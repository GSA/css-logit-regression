library(MASS)

dat_it <- read.csv(file="it_data.csv")
dat_hr<-read.csv(file='hr_data.csv')
dat_fm<-read.csv(file='fm_data.csv')
dat_acq<-read.csv(file='acq_data.csv')

dat_it.centered<-data.frame(scale(dat_it,center=TRUE,scale=FALSE))
dat_hr.centered<-data.frame(scale(dat_hr,center=TRUE,scale=FALSE))
dat_fm.centered<-data.frame(scale(dat_fm,center=TRUE,scale=FALSE))
dat_acq.centered<-data.frame(scale(dat_acq,center=TRUE,scale=FALSE))


#removed intercept
m_it<-polr(as.factor(IT.Function.Satisfaction) ~ intercept+O.M.Satisfaction+O.M.Importance+IT.Support.Satisfaction+IT.Support.Importance+
             IT.Function.System.Satisfaction+IT.Function.Strategic.Partner+IT.Equipment.Satisfaction+
             IT.Equipment.Importance+IT.Comms.Satisfaction+IT.Comms.Importance+DM.E.Satisfaction+DM.E.Importance+
             AGENCY_Social.Security.Administration+AGENCY_Small.Business.Administration+AGENCY_Office.of.Personnel.Management+
             AGENCY_Nuclear.Regulatory.Commission+AGENCY_National.Science.Foundation+AGENCY_National.Aeronautics.and.Space.Administration+
             AGENCY_General.Services.Administration+AGENCY_Environmental.Protection.Agency+AGENCY_Department.of.the.Treasury+
             AGENCY_Department.of.the.Interior+AGENCY_Department.of.Veterans.Affairs+AGENCY_Department.of.Transportation+
             AGENCY_Department.of.State+AGENCY_Department.of.Labor+AGENCY_Department.of.Justice+
             AGENCY_Department.of.Housing.and.Urban.Development+AGENCY_Department.of.Homeland.Security+
             AGENCY_Department.of.Health.and.Human.Services+AGENCY_Department.of.Energy+
             AGENCY_Department.of.Education+AGENCY_Department.of.Defense+AGENCY_Department.of.Commerce+TELEWORK + AGE+GRADELEVEL_GS.10+
             GRADELEVEL_GS.11+GRADELEVEL_GS.12+GRADELEVEL_GS.13+GRADELEVEL_GS.14+GRADELEVEL_GS.15+GRADELEVEL_GS.2+GRADELEVEL_GS.3+
             GRADELEVEL_GS.4+GRADELEVEL_GS.5+GRADELEVEL_GS.6+GRADELEVEL_GS.7+GRADELEVEL_GS.8+GRADELEVEL_GS.9+GRADELEVEL_SES+
             EDUCATION_LEVEL_Bachelor.s.Degree+EDUCATION_LEVEL_Doctoral.Professional.Degree+EDUCATION_LEVEL_High.School.Diploma..GED..or.Equivalent+
             EDUCATION_LEVEL_Less.than.High.School+EDUCATION_LEVEL_Master.s.Degree+EDUCATION_LEVEL_Some.College..no.degree.+
             EDUCATION_LEVEL_Trade.Technical.Certificate+LEAVING_Yes...for.personal.reasons..shorter.commute..family.responsibilities..health..further.education..spouse.relocation.etc..+
             LEAVING_Yes...to.retire+LEAVING_Yes...to.take.another.job,data=dat_it,Hess=TRUE)

m_fm<-polr(as.factor(Financial.Management.Satisfaction) ~ AGENCY_Social.Security.Administration+AGENCY_Small.Business.Administration+
             AGENCY_Office.of.Personnel.Management+AGENCY_Nuclear.Regulatory.Commission+AGENCY_National.Science.Foundation+
             AGENCY_National.Aeronautics.and.Space.Administration+AGENCY_General.Services.Administration+AGENCY_Environmental.Protection.Agency+
             AGENCY_Department.of.the.Treasury+AGENCY_Department.of.the.Interior+AGENCY_Department.of.Veterans.Affairs+
             AGENCY_Department.of.Transportation+AGENCY_Department.of.State+AGENCY_Department.of.Labor+AGENCY_Department.of.Justice+
             AGENCY_Department.of.Housing.and.Urban.Development+AGENCY_Department.of.Homeland.Security+AGENCY_Department.of.Health.and.Human.Services+
             AGENCY_Department.of.Energy+AGENCY_Department.of.Education+AGENCY_Department.of.Defense+AGENCY_Department.of.Commerce+TELEWORK + AGE+
             GRADELEVEL_GS.10+GRADELEVEL_GS.11+GRADELEVEL_GS.12+GRADELEVEL_GS.13+GRADELEVEL_GS.14+GRADELEVEL_GS.15+GRADELEVEL_GS.2+GRADELEVEL_GS.3+
             GRADELEVEL_GS.4+GRADELEVEL_GS.5+GRADELEVEL_GS.6+GRADELEVEL_GS.7+GRADELEVEL_GS.8+GRADELEVEL_GS.9+GRADELEVEL_SES+
             EDUCATION_LEVEL_Bachelor.s.Degree+EDUCATION_LEVEL_Doctoral.Professional.Degree+EDUCATION_LEVEL_High.School.Diploma..GED..or.Equivalent+
             EDUCATION_LEVEL_Less.than.High.School+EDUCATION_LEVEL_Master.s.Degree+EDUCATION_LEVEL_Some.College..no.degree.+
             EDUCATION_LEVEL_Trade.Technical.Certificate+LEAVING_Yes...for.personal.reasons..shorter.commute..family.responsibilities..health..further.education..spouse.relocation.etc..+
             LEAVING_Yes...to.retire+LEAVING_Yes...to.take.another.job+Benefits.Management.Importance+Benefits.Management.Satisfaction+
             Bill.Payment.Importance+Bill.Payment.Support.Satisfaction+Budget.Execution.Importance+Budget.Execution.Support.Satisfaction+
             Budget.Formulation.Importance+Budget.Formulation.Satisfaction+Financial.Management.Information.Satisfaction+
             Financial.Management.Information.and.Analysis.Importance+
             Financial.Management.Strategic.Partner.Satisfaction+Financial.Management.System+
             Financial.Risk.Management.Services.Importance+Financial.Risk.Management.Services.Satisfaction,data=dat_fm,Hess=TRUE)

m_hr<-polr(as.factor(Human.Capital.Satisfaction) ~ AGENCY_Social.Security.Administration+
             AGENCY_Small.Business.Administration+AGENCY_Office.of.Personnel.Management+AGENCY_Nuclear.Regulatory.Commission+
             AGENCY_National.Science.Foundation+AGENCY_National.Aeronautics.and.Space.Administration+AGENCY_General.Services.Administration+
             AGENCY_Environmental.Protection.Agency+AGENCY_Department.of.the.Treasury+AGENCY_Department.of.the.Interior+
             AGENCY_Department.of.Veterans.Affairs+AGENCY_Department.of.Transportation+AGENCY_Department.of.State+AGENCY_Department.of.Labor+
             AGENCY_Department.of.Justice+AGENCY_Department.of.Housing.and.Urban.Development+AGENCY_Department.of.Homeland.Security+
             AGENCY_Department.of.Health.and.Human.Services+AGENCY_Department.of.Energy+AGENCY_Department.of.Education+AGENCY_Department.of.Defense+
             AGENCY_Department.of.Commerce+TELEWORK + AGE+GRADELEVEL_GS.10+GRADELEVEL_GS.11+GRADELEVEL_GS.12+GRADELEVEL_GS.13+GRADELEVEL_GS.14+
             GRADELEVEL_GS.15+GRADELEVEL_GS.2+GRADELEVEL_GS.3+GRADELEVEL_GS.4+GRADELEVEL_GS.5+GRADELEVEL_GS.6+GRADELEVEL_GS.7+GRADELEVEL_GS.8+
             GRADELEVEL_GS.9+GRADELEVEL_SES+EDUCATION_LEVEL_Bachelor.s.Degree+EDUCATION_LEVEL_Doctoral.Professional.Degree+
             EDUCATION_LEVEL_High.School.Diploma..GED..or.Equivalent+EDUCATION_LEVEL_Less.than.High.School+EDUCATION_LEVEL_Master.s.Degree+
             EDUCATION_LEVEL_Some.College..no.degree.+EDUCATION_LEVEL_Trade.Technical.Certificate+LEAVING_Yes...for.personal.reasons..shorter.commute..family.responsibilities..health..further.education..spouse.relocation.etc..+
             LEAVING_Yes...to.retire+LEAVING_Yes...to.take.another.job+Employee.Relations.Services.Importance+
             Employee.Relations.Services.Satisfaction+Human.Capital.Strategic.Partner+Human.Capital.System+
             Labor.Relations.Services.Importance+Labor.Relations.Services.Satisfaction+Recruiting.and.Hiring.Services.Importance+
             Recruiting.and.Hiring.Services.Satisfaction+Reporting.Responsibilities.Awareness+Retirement.Planning.and.Processing.Importance+
             Retirement.Planning.and.Processing.Satisfaction+Time.and.Attendance.Management.Importance+
             Time.and.Attendance.Management.Satisfaction+Training.and.Development.Services.Importance+
             Workforce.and.Succession.Planning.Importance+Workforce.and.Succession.Planning.Satisfaction+Worklife.Support.Services.Importance+
             Worklife.Support.Services.Satisfaction,data=dat_hr,Hess=TRUE)

m_acq<-polr(as.factor(Contracting.Function.Quality.of.Support.Satisfaction) ~ AGENCY_Social.Security.Administration+
              AGENCY_Small.Business.Administration+AGENCY_Office.of.Personnel.Management+AGENCY_Nuclear.Regulatory.Commission+
              AGENCY_National.Science.Foundation+AGENCY_National.Aeronautics.and.Space.Administration+AGENCY_General.Services.Administration+
              AGENCY_Environmental.Protection.Agency+AGENCY_Department.of.the.Treasury+AGENCY_Department.of.the.Interior+
              AGENCY_Department.of.Veterans.Affairs+AGENCY_Department.of.Transportation+AGENCY_Department.of.State+AGENCY_Department.of.Labor+
              AGENCY_Department.of.Justice+AGENCY_Department.of.Housing.and.Urban.Development+AGENCY_Department.of.Homeland.Security+
              AGENCY_Department.of.Health.and.Human.Services+AGENCY_Department.of.Energy+AGENCY_Department.of.Education+AGENCY_Department.of.Defense+
              AGENCY_Department.of.Commerce+TELEWORK + AGE+GRADELEVEL_GS.10+GRADELEVEL_GS.11+GRADELEVEL_GS.12+GRADELEVEL_GS.13+GRADELEVEL_GS.14+
              GRADELEVEL_GS.15+GRADELEVEL_GS.2+GRADELEVEL_GS.3+GRADELEVEL_GS.4+GRADELEVEL_GS.5+GRADELEVEL_GS.6+GRADELEVEL_GS.7+GRADELEVEL_GS.8+
              GRADELEVEL_GS.9+GRADELEVEL_SES+EDUCATION_LEVEL_Bachelor.s.Degree+EDUCATION_LEVEL_Doctoral.Professional.Degree+
              EDUCATION_LEVEL_High.School.Diploma..GED..or.Equivalent+EDUCATION_LEVEL_Less.than.High.School+EDUCATION_LEVEL_Master.s.Degree+
              EDUCATION_LEVEL_Some.College..no.degree.+EDUCATION_LEVEL_Trade.Technical.Certificate+LEAVING_Yes...for.personal.reasons..shorter.commute..family.responsibilities..health..further.education..spouse.relocation.etc..+
              LEAVING_Yes...to.retire+LEAVING_Yes...to.take.another.job+Contract.Administration.Services.Importance+
              Contract.Administration.Services.Satisfaction+Contracting.Function.Partner.Satisfaction+
              Contracting.Function.Systems.Satisfaction+Pre.Award.Services.Importance+
              Pre.Award.Services.Satisfaction+Purchase.Card.Management.Services.Importance+Purchase.Card.Management.Services.Satisfaction,data=dat_acq,Hess=TRUE)


#
#centered data
#

m_it_cent<-polr(as.factor(IT.Function.Satisfaction) ~ intercept+O.M.Satisfaction+O.M.Importance+IT.Support.Satisfaction+IT.Support.Importance+
             IT.Function.System.Satisfaction+IT.Function.Strategic.Partner+IT.Equipment.Satisfaction+
             IT.Equipment.Importance+IT.Comms.Satisfaction+IT.Comms.Importance+DM.E.Satisfaction+DM.E.Importance+
             AGENCY_Social.Security.Administration+AGENCY_Small.Business.Administration+AGENCY_Office.of.Personnel.Management+
             AGENCY_Nuclear.Regulatory.Commission+AGENCY_National.Science.Foundation+AGENCY_National.Aeronautics.and.Space.Administration+
             AGENCY_General.Services.Administration+AGENCY_Environmental.Protection.Agency+AGENCY_Department.of.the.Treasury+
             AGENCY_Department.of.the.Interior+AGENCY_Department.of.Veterans.Affairs+AGENCY_Department.of.Transportation+
             AGENCY_Department.of.State+AGENCY_Department.of.Labor+AGENCY_Department.of.Justice+
             AGENCY_Department.of.Housing.and.Urban.Development+AGENCY_Department.of.Homeland.Security+
             AGENCY_Department.of.Health.and.Human.Services+AGENCY_Department.of.Energy+
             AGENCY_Department.of.Education+AGENCY_Department.of.Defense+AGENCY_Department.of.Commerce+TELEWORK + AGE+GRADELEVEL_GS.10+
             GRADELEVEL_GS.11+GRADELEVEL_GS.12+GRADELEVEL_GS.13+GRADELEVEL_GS.14+GRADELEVEL_GS.15+GRADELEVEL_GS.2+GRADELEVEL_GS.3+
             GRADELEVEL_GS.4+GRADELEVEL_GS.5+GRADELEVEL_GS.6+GRADELEVEL_GS.7+GRADELEVEL_GS.8+GRADELEVEL_GS.9+GRADELEVEL_SES+
             EDUCATION_LEVEL_Bachelor.s.Degree+EDUCATION_LEVEL_Doctoral.Professional.Degree+EDUCATION_LEVEL_High.School.Diploma..GED..or.Equivalent+
             EDUCATION_LEVEL_Less.than.High.School+EDUCATION_LEVEL_Master.s.Degree+EDUCATION_LEVEL_Some.College..no.degree.+
             EDUCATION_LEVEL_Trade.Technical.Certificate+LEAVING_Yes...for.personal.reasons..shorter.commute..family.responsibilities..health..further.education..spouse.relocation.etc..+
             LEAVING_Yes...to.retire+LEAVING_Yes...to.take.another.job,data=dat_it.centered,Hess=TRUE)

m_fm_cent<-polr(as.factor(Financial.Management.Satisfaction) ~ AGENCY_Social.Security.Administration+AGENCY_Small.Business.Administration+
             AGENCY_Office.of.Personnel.Management+AGENCY_Nuclear.Regulatory.Commission+AGENCY_National.Science.Foundation+
             AGENCY_National.Aeronautics.and.Space.Administration+AGENCY_General.Services.Administration+AGENCY_Environmental.Protection.Agency+
             AGENCY_Department.of.the.Treasury+AGENCY_Department.of.the.Interior+AGENCY_Department.of.Veterans.Affairs+
             AGENCY_Department.of.Transportation+AGENCY_Department.of.State+AGENCY_Department.of.Labor+AGENCY_Department.of.Justice+
             AGENCY_Department.of.Housing.and.Urban.Development+AGENCY_Department.of.Homeland.Security+AGENCY_Department.of.Health.and.Human.Services+
             AGENCY_Department.of.Energy+AGENCY_Department.of.Education+AGENCY_Department.of.Defense+AGENCY_Department.of.Commerce+TELEWORK + AGE+
             GRADELEVEL_GS.10+GRADELEVEL_GS.11+GRADELEVEL_GS.12+GRADELEVEL_GS.13+GRADELEVEL_GS.14+GRADELEVEL_GS.15+GRADELEVEL_GS.2+GRADELEVEL_GS.3+
             GRADELEVEL_GS.4+GRADELEVEL_GS.5+GRADELEVEL_GS.6+GRADELEVEL_GS.7+GRADELEVEL_GS.8+GRADELEVEL_GS.9+GRADELEVEL_SES+
             EDUCATION_LEVEL_Bachelor.s.Degree+EDUCATION_LEVEL_Doctoral.Professional.Degree+EDUCATION_LEVEL_High.School.Diploma..GED..or.Equivalent+
             EDUCATION_LEVEL_Less.than.High.School+EDUCATION_LEVEL_Master.s.Degree+EDUCATION_LEVEL_Some.College..no.degree.+
             EDUCATION_LEVEL_Trade.Technical.Certificate+LEAVING_Yes...for.personal.reasons..shorter.commute..family.responsibilities..health..further.education..spouse.relocation.etc..+
             LEAVING_Yes...to.retire+LEAVING_Yes...to.take.another.job+Benefits.Management.Importance+Benefits.Management.Satisfaction+
             Bill.Payment.Importance+Bill.Payment.Support.Satisfaction+Budget.Execution.Importance+Budget.Execution.Support.Satisfaction+
             Budget.Formulation.Importance+Budget.Formulation.Satisfaction+Financial.Management.Information.Satisfaction+
             Financial.Management.Information.and.Analysis.Importance+
             Financial.Management.Strategic.Partner.Satisfaction+Financial.Management.System+
             Financial.Risk.Management.Services.Importance+Financial.Risk.Management.Services.Satisfaction,data=dat_fm.centered,Hess=TRUE)

m_hr_cent<-polr(as.factor(Human.Capital.Satisfaction) ~ AGENCY_Social.Security.Administration+
             AGENCY_Small.Business.Administration+AGENCY_Office.of.Personnel.Management+AGENCY_Nuclear.Regulatory.Commission+
             AGENCY_National.Science.Foundation+AGENCY_National.Aeronautics.and.Space.Administration+AGENCY_General.Services.Administration+
             AGENCY_Environmental.Protection.Agency+AGENCY_Department.of.the.Treasury+AGENCY_Department.of.the.Interior+
             AGENCY_Department.of.Veterans.Affairs+AGENCY_Department.of.Transportation+AGENCY_Department.of.State+AGENCY_Department.of.Labor+
             AGENCY_Department.of.Justice+AGENCY_Department.of.Housing.and.Urban.Development+AGENCY_Department.of.Homeland.Security+
             AGENCY_Department.of.Health.and.Human.Services+AGENCY_Department.of.Energy+AGENCY_Department.of.Education+AGENCY_Department.of.Defense+
             AGENCY_Department.of.Commerce+TELEWORK + AGE+GRADELEVEL_GS.10+GRADELEVEL_GS.11+GRADELEVEL_GS.12+GRADELEVEL_GS.13+GRADELEVEL_GS.14+
             GRADELEVEL_GS.15+GRADELEVEL_GS.2+GRADELEVEL_GS.3+GRADELEVEL_GS.4+GRADELEVEL_GS.5+GRADELEVEL_GS.6+GRADELEVEL_GS.7+GRADELEVEL_GS.8+
             GRADELEVEL_GS.9+GRADELEVEL_SES+EDUCATION_LEVEL_Bachelor.s.Degree+EDUCATION_LEVEL_Doctoral.Professional.Degree+
             EDUCATION_LEVEL_High.School.Diploma..GED..or.Equivalent+EDUCATION_LEVEL_Less.than.High.School+EDUCATION_LEVEL_Master.s.Degree+
             EDUCATION_LEVEL_Some.College..no.degree.+EDUCATION_LEVEL_Trade.Technical.Certificate+LEAVING_Yes...for.personal.reasons..shorter.commute..family.responsibilities..health..further.education..spouse.relocation.etc..+
             LEAVING_Yes...to.retire+LEAVING_Yes...to.take.another.job+Employee.Relations.Services.Importance+
             Employee.Relations.Services.Satisfaction+Human.Capital.Strategic.Partner+Human.Capital.System+
             Labor.Relations.Services.Importance+Labor.Relations.Services.Satisfaction+Recruiting.and.Hiring.Services.Importance+
             Recruiting.and.Hiring.Services.Satisfaction+Reporting.Responsibilities.Awareness+Retirement.Planning.and.Processing.Importance+
             Retirement.Planning.and.Processing.Satisfaction+Time.and.Attendance.Management.Importance+
             Time.and.Attendance.Management.Satisfaction+Training.and.Development.Services.Importance+
             Workforce.and.Succession.Planning.Importance+Workforce.and.Succession.Planning.Satisfaction+Worklife.Support.Services.Importance+
             Worklife.Support.Services.Satisfaction,data=dat_hr.centered,Hess=TRUE)

m_acq_cent<-polr(as.factor(Contracting.Function.Quality.of.Support.Satisfaction) ~ AGENCY_Social.Security.Administration+
              AGENCY_Small.Business.Administration+AGENCY_Office.of.Personnel.Management+AGENCY_Nuclear.Regulatory.Commission+
              AGENCY_National.Science.Foundation+AGENCY_National.Aeronautics.and.Space.Administration+AGENCY_General.Services.Administration+
              AGENCY_Environmental.Protection.Agency+AGENCY_Department.of.the.Treasury+AGENCY_Department.of.the.Interior+
              AGENCY_Department.of.Veterans.Affairs+AGENCY_Department.of.Transportation+AGENCY_Department.of.State+AGENCY_Department.of.Labor+
              AGENCY_Department.of.Justice+AGENCY_Department.of.Housing.and.Urban.Development+AGENCY_Department.of.Homeland.Security+
              AGENCY_Department.of.Health.and.Human.Services+AGENCY_Department.of.Energy+AGENCY_Department.of.Education+AGENCY_Department.of.Defense+
              AGENCY_Department.of.Commerce+TELEWORK + AGE+GRADELEVEL_GS.10+GRADELEVEL_GS.11+GRADELEVEL_GS.12+GRADELEVEL_GS.13+GRADELEVEL_GS.14+
              GRADELEVEL_GS.15+GRADELEVEL_GS.2+GRADELEVEL_GS.3+GRADELEVEL_GS.4+GRADELEVEL_GS.5+GRADELEVEL_GS.6+GRADELEVEL_GS.7+GRADELEVEL_GS.8+
              GRADELEVEL_GS.9+GRADELEVEL_SES+EDUCATION_LEVEL_Bachelor.s.Degree+EDUCATION_LEVEL_Doctoral.Professional.Degree+
              EDUCATION_LEVEL_High.School.Diploma..GED..or.Equivalent+EDUCATION_LEVEL_Less.than.High.School+EDUCATION_LEVEL_Master.s.Degree+
              EDUCATION_LEVEL_Some.College..no.degree.+EDUCATION_LEVEL_Trade.Technical.Certificate+LEAVING_Yes...for.personal.reasons..shorter.commute..family.responsibilities..health..further.education..spouse.relocation.etc..+
              LEAVING_Yes...to.retire+LEAVING_Yes...to.take.another.job+Contract.Administration.Services.Importance+
              Contract.Administration.Services.Satisfaction+Contracting.Function.Partner.Satisfaction+
              Contracting.Function.Systems.Satisfaction+Pre.Award.Services.Importance+
              Pre.Award.Services.Satisfaction+Purchase.Card.Management.Services.Importance+Purchase.Card.Management.Services.Satisfaction,data=dat_acq.centered,Hess=TRUE)

summary(m_it)
summary(m_fm)
summary(m_hr)
summary(m_acq)

#
#summary centered
#
summary(m_it)
summary(m_fm)
summary(m_hr)
summary(m_acq)


#+YRSAGENCY2 +SUP_STATUS
#collin.diag = colldiag(mod=m, scale=FALSE, center=FALSE, add.intercept=FALSE)
#collin.diag# Check out the severe collinearity
#vif(m)

## store table
(ctableit <- coef(summary(m_it)))
(ctablefm <- coef(summary(m_fm)))
(ctablehr <- coef(summary(m_hr)))
(ctableacq <- coef(summary(m_acq)))

#centered
(ctableit_cent <- coef(summary(m_it_cent)))

## calculate and store p values
pit <- pnorm(abs(ctableit[, "t value"]), lower.tail = FALSE) * 2
pfm <- pnorm(abs(ctablefm[, "t value"]), lower.tail = FALSE) * 2
phr <- pnorm(abs(ctablehr[, "t value"]), lower.tail = FALSE) * 2
pacq <- pnorm(abs(ctableacq[, "t value"]), lower.tail = FALSE) * 2

#centered
pit_cent <- pnorm(abs(ctableit_cent[, "t value"]), lower.tail = FALSE) * 2

## combined table
(ctableit <- cbind(ctableit, "p value" = p))
(ctablefm <- cbind(ctablefm, "p value" = p))
(ctablehr <- cbind(ctablehr, "p value" = p))
(ctableacq <- cbind(ctableacq, "p value" = p))

#(ci <- confint(m)) # default method gives profiled CIs
confint.default(m_it) # CIs assuming normality
confint.default(m_fm)
confint.default(m_hr)
confint.default(m_acq)

## odds ratios
exp(coef(m_it))
exp(coef(m_fm))
exp(coef(m_hr))
exp(coef(m_acq))
exp(coef(m_it_cent))
exp(coef(m_fm_cent))
exp(coef(m_hr_cent))
exp(coef(m_acq_cent))

## OR and CI
exp(cbind(OR = coef(m_it), ci))
exp(cbind(OR = coef(m_fm), ci))
exp(cbind(OR = coef(m_hr), ci))
exp(cbind(OR = coef(m_acq), ci))




