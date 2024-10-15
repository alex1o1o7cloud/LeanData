import Mathlib

namespace NUMINAMATH_GPT_compute_x_y_power_sum_l1141_114171

noncomputable def pi : ℝ := Real.pi

theorem compute_x_y_power_sum
  (x y : ℝ)
  (h1 : 1 < x)
  (h2 : 1 < y)
  (h3 : (Real.log x / Real.log 2)^5 + (Real.log y / Real.log 3)^5 + 32 = 16 * (Real.log x / Real.log 2) * (Real.log y / Real.log 3)) :
  x^pi + y^pi = 2^(pi * (16:ℝ)^(1/5)) + 3^(pi * (16:ℝ)^(1/5)) :=
by
  sorry

end NUMINAMATH_GPT_compute_x_y_power_sum_l1141_114171


namespace NUMINAMATH_GPT_find_a_minus_b_l1141_114122

def f (a b x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -3 * x + 5
def h (a b x : ℝ) : ℝ := f a b (g x)
def h_inv (x : ℝ) : ℝ := x + 7

theorem find_a_minus_b (a b : ℝ) :
  (∀ x : ℝ, h a b x = -3 * a * x + 5 * a + b) ∧
  (∀ x : ℝ, h_inv (h a b x) = x) ∧
  (∀ x : ℝ, h a b x = x - 7) →
  a - b = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_a_minus_b_l1141_114122


namespace NUMINAMATH_GPT_correct_exponentiation_l1141_114133

variable (a : ℝ)

theorem correct_exponentiation : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_GPT_correct_exponentiation_l1141_114133


namespace NUMINAMATH_GPT_evaluate_exponent_l1141_114104

theorem evaluate_exponent : (3^3)^2 = 729 := by
  sorry

end NUMINAMATH_GPT_evaluate_exponent_l1141_114104


namespace NUMINAMATH_GPT_negation_exists_l1141_114181

theorem negation_exists (a : ℝ) :
  ¬ (∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ ∀ x : ℝ, x^2 - a * x + 1 ≥ 0 :=
sorry

end NUMINAMATH_GPT_negation_exists_l1141_114181


namespace NUMINAMATH_GPT_total_surface_area_prime_rectangular_solid_l1141_114157

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := Prime n

def prime_edge_lengths (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c

def volume (a b c : ℕ) : ℕ := a * b * c

def surface_area (a b c : ℕ) : ℕ := 2 * (a * b + b * c + c * a)

-- The main theorem statement
theorem total_surface_area_prime_rectangular_solid :
  ∃ (a b c : ℕ), prime_edge_lengths a b c ∧ volume a b c = 105 ∧ surface_area a b c = 142 :=
sorry

end NUMINAMATH_GPT_total_surface_area_prime_rectangular_solid_l1141_114157


namespace NUMINAMATH_GPT_factory_hours_per_day_l1141_114129

def hour_worked_forth_machine := 12
def production_rate_per_hour := 2
def selling_price_per_kg := 50
def total_earnings := 8100

def h := 23

theorem factory_hours_per_day
  (num_machines : ℕ)
  (num_machines := 3)
  (production_first_three : ℕ := num_machines * production_rate_per_hour * h)
  (production_fourth : ℕ := hour_worked_forth_machine * production_rate_per_hour)
  (total_production : ℕ := production_first_three + production_fourth)
  (total_earnings_eq : total_production * selling_price_per_kg = total_earnings) :
  h = 23 := by
  sorry

end NUMINAMATH_GPT_factory_hours_per_day_l1141_114129


namespace NUMINAMATH_GPT_find_a_l1141_114197

def star (x y : ℤ × ℤ) : ℤ × ℤ := (x.1 - y.1, x.2 + y.2)

theorem find_a :
  ∃ (a b : ℤ), 
  star (5, 2) (1, 1) = (a, b) ∧
  star (a, b) (0, 1) = (2, 5) ∧
  a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l1141_114197


namespace NUMINAMATH_GPT_tangent_product_20_40_60_80_l1141_114165

theorem tangent_product_20_40_60_80 :
  Real.tan (20 * Real.pi / 180) * Real.tan (40 * Real.pi / 180) * Real.tan (60 * Real.pi / 180) * Real.tan (80 * Real.pi / 180) = 3 :=
by
  sorry

end NUMINAMATH_GPT_tangent_product_20_40_60_80_l1141_114165


namespace NUMINAMATH_GPT_interval_solution_l1141_114115

theorem interval_solution :
  { x : ℝ | 2 < 3 * x ∧ 3 * x < 3 ∧ 2 < 4 * x ∧ 4 * x < 3 } =
  { x : ℝ | (2 / 3) < x ∧ x < (3 / 4) } :=
by
  sorry

end NUMINAMATH_GPT_interval_solution_l1141_114115


namespace NUMINAMATH_GPT_glucose_solution_volume_l1141_114174

theorem glucose_solution_volume (V : ℕ) (h : 500 / 10 = V / 20) : V = 1000 :=
sorry

end NUMINAMATH_GPT_glucose_solution_volume_l1141_114174


namespace NUMINAMATH_GPT_factorize_x_squared_plus_2x_l1141_114175

theorem factorize_x_squared_plus_2x (x : ℝ) : x^2 + 2*x = x*(x + 2) :=
by sorry

end NUMINAMATH_GPT_factorize_x_squared_plus_2x_l1141_114175


namespace NUMINAMATH_GPT_find_number_l1141_114162

theorem find_number (x: ℝ) (h1: 0.10 * x + 0.15 * 50 = 10.5) : x = 30 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1141_114162


namespace NUMINAMATH_GPT_solve_y_eq_l1141_114180

theorem solve_y_eq :
  ∀ y: ℝ, y ≠ -1 → (y^3 - 3 * y^2) / (y^2 + 2 * y + 1) + 2 * y = -1 → 
  y = 1 / Real.sqrt 3 ∨ y = -1 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_solve_y_eq_l1141_114180


namespace NUMINAMATH_GPT_all_numbers_are_2007_l1141_114132

noncomputable def sequence_five_numbers (a b c d e : ℤ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ 
  (a = 2007 ∨ b = 2007 ∨ c = 2007 ∨ d = 2007 ∨ e = 2007) ∧ 
  (∃ r1, b = r1 * a ∧ c = r1 * b ∧ d = r1 * c ∧ e = r1 * d) ∧
  (∃ r2, a = r2 * b ∧ c = r2 * a ∧ d = r2 * c ∧ e = r2 * d) ∧
  (∃ r3, a = r3 * c ∧ b = r3 * a ∧ d = r3 * b ∧ e = r3 * d) ∧
  (∃ r4, a = r4 * d ∧ b = r4 * a ∧ c = r4 * b ∧ e = r4 * d) ∧
  (∃ r5, a = r5 * e ∧ b = r5 * a ∧ c = r5 * b ∧ d = r5 * c)

theorem all_numbers_are_2007 (a b c d e : ℤ) 
  (h : sequence_five_numbers a b c d e) : 
  a = 2007 ∧ b = 2007 ∧ c = 2007 ∧ d = 2007 ∧ e = 2007 :=
sorry

end NUMINAMATH_GPT_all_numbers_are_2007_l1141_114132


namespace NUMINAMATH_GPT_imaginary_part_of_z_l1141_114121

open Complex

-- Condition
def equation_z (z : ℂ) : Prop := (z * (1 + I) * I^3) / (1 - I) = 1 - I

-- Problem statement
theorem imaginary_part_of_z (z : ℂ) (h : equation_z z) : z.im = -1 := 
by 
  sorry

end NUMINAMATH_GPT_imaginary_part_of_z_l1141_114121


namespace NUMINAMATH_GPT_solve_system_of_equations_l1141_114179

theorem solve_system_of_equations :
  ∃ x y : ℝ, (2 * x - 5 * y = -1) ∧ (-4 * x + y = -7) ∧ (x = 2) ∧ (y = 1) :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1141_114179


namespace NUMINAMATH_GPT_all_blue_figures_are_small_l1141_114173

variables (Shape : Type) (Large Blue Small Square Triangle : Shape → Prop)

-- Given conditions
axiom h1 : ∀ (x : Shape), Large x → Square x
axiom h2 : ∀ (x : Shape), Blue x → Triangle x

-- The goal to prove
theorem all_blue_figures_are_small : ∀ (x : Shape), Blue x → Small x :=
by
  sorry

end NUMINAMATH_GPT_all_blue_figures_are_small_l1141_114173


namespace NUMINAMATH_GPT_birds_more_than_nests_l1141_114188

theorem birds_more_than_nests : 
  let birds := 6 
  let nests := 3 
  (birds - nests) = 3 := 
by 
  sorry

end NUMINAMATH_GPT_birds_more_than_nests_l1141_114188


namespace NUMINAMATH_GPT_correct_f_l1141_114116

noncomputable def f : ℝ → ℝ := sorry

axiom f_0 : f 0 = 1
axiom functional_equation (x y : ℝ) : f (x * y + 1) = f x * f y - f y - x + 2

theorem correct_f (x : ℝ) : f x = x + 1 := sorry

end NUMINAMATH_GPT_correct_f_l1141_114116


namespace NUMINAMATH_GPT_find_A_d_minus_B_d_l1141_114137

variable {d : ℕ} (A B : ℕ) (h₁ : d > 6) (h₂ : (d^1 * A + B) + (d^1 * A + A) = 1 * d^2 + 6 * d^1 + 2)

theorem find_A_d_minus_B_d (h₁ : d > 6) (h₂ : (d^1 * A + B) + (d^1 * A + A) = 1 * d^2 + 6 * d^1 + 2) :
  A - B = 3 :=
sorry

end NUMINAMATH_GPT_find_A_d_minus_B_d_l1141_114137


namespace NUMINAMATH_GPT_kamal_marks_in_mathematics_l1141_114150

def kamal_marks_english : ℕ := 96
def kamal_marks_physics : ℕ := 82
def kamal_marks_chemistry : ℕ := 67
def kamal_marks_biology : ℕ := 85
def kamal_average_marks : ℕ := 79
def kamal_number_of_subjects : ℕ := 5

theorem kamal_marks_in_mathematics :
  let total_marks := kamal_average_marks * kamal_number_of_subjects
  let total_known_marks := kamal_marks_english + kamal_marks_physics + kamal_marks_chemistry + kamal_marks_biology
  total_marks - total_known_marks = 65 :=
by
  sorry

end NUMINAMATH_GPT_kamal_marks_in_mathematics_l1141_114150


namespace NUMINAMATH_GPT_sam_drove_distance_l1141_114172

theorem sam_drove_distance (m_distance : ℕ) (m_time : ℕ) (s_time : ℕ) (s_distance : ℕ)
  (m_distance_eq : m_distance = 120) (m_time_eq : m_time = 3) (s_time_eq : s_time = 4) :
  s_distance = (m_distance / m_time) * s_time :=
by
  sorry

end NUMINAMATH_GPT_sam_drove_distance_l1141_114172


namespace NUMINAMATH_GPT_nonneg_real_inequality_l1141_114144

theorem nonneg_real_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
    a^4 + b^4 + c^4 - 2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2) + a^2 * b * c + b^2 * a * c + c^2 * a * b ≥ 0 :=
sorry

end NUMINAMATH_GPT_nonneg_real_inequality_l1141_114144


namespace NUMINAMATH_GPT_capital_of_a_l1141_114114

variable (P P' TotalCapital Ca : ℝ)

theorem capital_of_a 
  (h1 : a_income_5_percent = (2/3) * P)
  (h2 : a_income_7_percent = (2/3) * P')
  (h3 : a_income_7_percent - a_income_5_percent = 200)
  (h4 : P = 0.05 * TotalCapital)
  (h5 : P' = 0.07 * TotalCapital)
  : Ca = (2/3) * TotalCapital :=
by
  sorry

end NUMINAMATH_GPT_capital_of_a_l1141_114114


namespace NUMINAMATH_GPT_max_positive_integers_on_circle_l1141_114161

theorem max_positive_integers_on_circle (a : ℕ → ℕ) (h: ∀ k : ℕ, 2 < k → a k > a (k-1) + a (k-2)) :
  ∃ n : ℕ, (∀ i < 2018, a i > 0 -> n ≤ 1009) :=
  sorry

end NUMINAMATH_GPT_max_positive_integers_on_circle_l1141_114161


namespace NUMINAMATH_GPT_Edward_money_left_l1141_114101

theorem Edward_money_left {initial_amount item_cost sales_tax_rate sales_tax total_cost money_left : ℝ} 
    (h_initial : initial_amount = 18) 
    (h_item : item_cost = 16.35) 
    (h_rate : sales_tax_rate = 0.075) 
    (h_sales_tax : sales_tax = item_cost * sales_tax_rate) 
    (h_sales_tax_rounded : sales_tax = 1.23) 
    (h_total : total_cost = item_cost + sales_tax) 
    (h_money_left : money_left = initial_amount - total_cost) :
    money_left = 0.42 :=
by sorry

end NUMINAMATH_GPT_Edward_money_left_l1141_114101


namespace NUMINAMATH_GPT_reciprocal_neg_one_thirteen_l1141_114156

theorem reciprocal_neg_one_thirteen : -(1:ℝ) / 13⁻¹ = -13 := 
sorry

end NUMINAMATH_GPT_reciprocal_neg_one_thirteen_l1141_114156


namespace NUMINAMATH_GPT_irrational_infinitely_many_approximations_l1141_114168

theorem irrational_infinitely_many_approximations (x : ℝ) (hx : Irrational x) (hx_pos : 0 < x) :
  ∃ᶠ (q : ℕ) in at_top, ∃ p : ℤ, |x - p / q| < 1 / q^2 :=
sorry

end NUMINAMATH_GPT_irrational_infinitely_many_approximations_l1141_114168


namespace NUMINAMATH_GPT_age_of_other_man_l1141_114145

theorem age_of_other_man
  (n : ℕ) (average_age_before : ℕ) (average_age_after : ℕ) (age_of_one_man : ℕ) (average_age_women : ℕ) 
  (h1 : n = 9)
  (h2 : average_age_after = average_age_before + 4)
  (h3 : age_of_one_man = 36)
  (h4 : average_age_women = 52) :
  (68 - 36 = 32) := 
by
  sorry

end NUMINAMATH_GPT_age_of_other_man_l1141_114145


namespace NUMINAMATH_GPT_find_some_ounce_size_l1141_114170

variable (x : ℕ)
variable (h_total : 122 = 6 * 5 + 4 * x + 15 * 4)

theorem find_some_ounce_size : x = 8 := by
  sorry

end NUMINAMATH_GPT_find_some_ounce_size_l1141_114170


namespace NUMINAMATH_GPT_investment_ratio_l1141_114106

theorem investment_ratio 
  (P Q : ℝ) 
  (profitP profitQ : ℝ)
  (h1 : profitP = 7 * (profitP + profitQ) / 17) 
  (h2 : profitQ = 10 * (profitP + profitQ) / 17)
  (tP : ℝ := 10)
  (tQ : ℝ := 20) 
  (h3 : profitP / profitQ = (P * tP) / (Q * tQ)) :
  P / Q = 7 / 5 := 
sorry

end NUMINAMATH_GPT_investment_ratio_l1141_114106


namespace NUMINAMATH_GPT_son_l1141_114111

theorem son's_age (S F : ℕ) (h1 : F = S + 26) (h2 : F + 2 = 2 * (S + 2)) : S = 24 :=
by
  sorry

end NUMINAMATH_GPT_son_l1141_114111


namespace NUMINAMATH_GPT_random_sampling_not_in_proving_methods_l1141_114147

inductive Method
| Comparison
| RandomSampling
| SyntheticAndAnalytic
| ProofByContradictionAndScaling

open Method

def proving_methods : List Method :=
  [Comparison, SyntheticAndAnalytic, ProofByContradictionAndScaling]

theorem random_sampling_not_in_proving_methods : 
  RandomSampling ∉ proving_methods :=
sorry

end NUMINAMATH_GPT_random_sampling_not_in_proving_methods_l1141_114147


namespace NUMINAMATH_GPT_speech_competition_score_l1141_114130

theorem speech_competition_score :
  let speech_content := 90
  let speech_skills := 80
  let speech_effects := 85
  let content_ratio := 4
  let skills_ratio := 2
  let effects_ratio := 4
  (speech_content * content_ratio + speech_skills * skills_ratio + speech_effects * effects_ratio) / (content_ratio + skills_ratio + effects_ratio) = 86 := by
  sorry

end NUMINAMATH_GPT_speech_competition_score_l1141_114130


namespace NUMINAMATH_GPT_proof_of_problem_statement_l1141_114139

noncomputable def problem_statement : Prop :=
  ∀ (k : ℝ) (m : ℝ),
    (0 < m ∧ m < 3/2) → 
    (-3/(4 * m) = k) → 
    (k < -1/2)

theorem proof_of_problem_statement : problem_statement :=
  sorry

end NUMINAMATH_GPT_proof_of_problem_statement_l1141_114139


namespace NUMINAMATH_GPT_initial_amount_in_cookie_jar_l1141_114185

theorem initial_amount_in_cookie_jar (M : ℝ) (h : 15 / 100 * (85 / 100 * (100 - 10) / 100 * (100 - 15) / 100 * M) = 15) : M = 24.51 :=
sorry

end NUMINAMATH_GPT_initial_amount_in_cookie_jar_l1141_114185


namespace NUMINAMATH_GPT_determine_positions_l1141_114149

-- Defining the participants
inductive Participant
| Olya
| Oleg
| Pasha

open Participant

-- Defining the possible places
inductive Place
| First
| Second
| Third

open Place

-- Define the conditions
def condition1 (pos : Participant → Place) : Prop := 
  pos Olya = First ∨ pos Oleg = First ∨ pos Pasha = First

def condition2 (pos : Participant → Place) : Prop :=
  (pos Olya = First ∧ pos Olya = Second ∧ pos Olya = Third) ∨
  (pos Oleg = First ∧ pos Oleg = Second ∧ pos Oleg = Third) ∨
  (pos Pasha = First ∧ pos Pasha = Second ∧ pos Pasha = Third)

def condition3 (pos : Participant → Place) : Prop :=
  ∀ p, pos p ≠ First ∧ pos p ≠ Second ∧ pos p ≠ Third

def condition4 (pos : Participant → Place) : Prop :=
  (pos Olya = First → (pos Oleg = First ∨ pos Pasha = First)) ∧
  (pos Oleg = First → pos Olya ≠ First) ∧
  (pos Pasha = First → (pos Oleg = First ∨ pos Olya = First))

def always_true_or_false : Prop :=
  (∀ p, p = Olya ∨ p = Oleg ∨ p = Pasha )

-- Main theorem
theorem determine_positions (pos : Participant → Place) :
  condition1 pos ∧ condition2 pos ∧ condition3 pos ∧ condition4 pos ∧ always_true_or_false →
  pos Oleg = First ∧ pos Pasha = Second ∧ pos Olya = Third := 
by
  sorry

end NUMINAMATH_GPT_determine_positions_l1141_114149


namespace NUMINAMATH_GPT_man_son_age_ratio_l1141_114103

-- Define the present age of the son
def son_age_present : ℕ := 22

-- Define the present age of the man based on the son's age
def man_age_present : ℕ := son_age_present + 24

-- Define the son's age in two years
def son_age_future : ℕ := son_age_present + 2

-- Define the man's age in two years
def man_age_future : ℕ := man_age_present + 2

-- Prove the ratio of the man's age to the son's age in two years is 2:1
theorem man_son_age_ratio : man_age_future / son_age_future = 2 := by
  sorry

end NUMINAMATH_GPT_man_son_age_ratio_l1141_114103


namespace NUMINAMATH_GPT_concentric_circles_area_difference_l1141_114146

/-- Two concentric circles with radii 12 cm and 7 cm have an area difference of 95π cm² between them. -/
theorem concentric_circles_area_difference :
  let r1 := 12
  let r2 := 7
  let area_larger := Real.pi * r1^2
  let area_smaller := Real.pi * r2^2
  let area_difference := area_larger - area_smaller
  area_difference = 95 * Real.pi := by
sorry

end NUMINAMATH_GPT_concentric_circles_area_difference_l1141_114146


namespace NUMINAMATH_GPT_park_area_l1141_114120

theorem park_area (length breadth : ℝ) (x : ℝ) 
  (h1 : length = 3 * x) 
  (h2 : breadth = x) 
  (h3 : 2 * length + 2 * breadth = 800) 
  (h4 : 12 * (4 / 60) * 1000 = 800) : 
  length * breadth = 30000 := by
sorry

end NUMINAMATH_GPT_park_area_l1141_114120


namespace NUMINAMATH_GPT_johnPaysPerYear_l1141_114125

-- Define the conditions
def epiPenCost : ℝ := 500
def insuranceCoverage : ℝ := 0.75
def epiPenFrequencyPerYear : ℝ := 2 -- Twice a year since 12 months / 6 months per EpiPen

-- Calculate the cost after insurance
def costAfterInsurance (cost : ℝ) (coverage : ℝ) : ℝ :=
  cost * (1 - coverage)

-- Calculate the annual cost
def annualCost (freq : ℝ) (cost : ℝ) : ℝ :=
  freq * cost

-- The statement asserting the proof
theorem johnPaysPerYear (cost : ℝ) (coverage : ℝ) (freq : ℝ) : 
  epiPenCost = cost → 
  insuranceCoverage = coverage → 
  epiPenFrequencyPerYear = freq → 
  annualCost freq (costAfterInsurance cost coverage) = 250 := 
by 
  intros _ _ _ 
  sorry

end NUMINAMATH_GPT_johnPaysPerYear_l1141_114125


namespace NUMINAMATH_GPT_person_A_boxes_average_unit_price_after_promotion_l1141_114143

-- Definitions based on the conditions.
def unit_price (x: ℕ) (y: ℕ) : ℚ := y / x

def person_A_spent : ℕ := 2400
def person_B_spent : ℕ := 3000
def promotion_discount : ℕ := 20
def boxes_difference : ℕ := 10

-- Main proofs
theorem person_A_boxes (unit_price: ℕ → ℕ → ℚ) 
  (person_A_spent person_B_spent boxes_difference: ℕ): 
  ∃ x, unit_price person_A_spent x = unit_price person_B_spent (x + boxes_difference) 
  ∧ x = 40 := 
by {
  sorry
}

theorem average_unit_price_after_promotion (unit_price: ℕ → ℕ → ℚ) 
  (promotion_discount: ℕ) (person_A_spent person_B_spent: ℕ) 
  (boxes_A_promotion boxes_B: ℕ): 
  person_A_spent / (boxes_A_promotion * 2) + 20 = 48 
  ∧ person_B_spent / (boxes_B * 2) + 20 = 50 :=
by {
  sorry
}

end NUMINAMATH_GPT_person_A_boxes_average_unit_price_after_promotion_l1141_114143


namespace NUMINAMATH_GPT_A_infinite_l1141_114191

noncomputable def f : ℝ → ℝ := sorry

def A : Set ℝ := { a : ℝ | f a > a ^ 2 }

theorem A_infinite
  (h_f_def : ∀ x : ℝ, ∃ y : ℝ, y = f x)
  (h_inequality: ∀ x : ℝ, (f x) ^ 2 ≤ 2 * x ^ 2 * f (x / 2))
  (h_A_nonempty : A ≠ ∅) :
  Set.Infinite A := 
sorry

end NUMINAMATH_GPT_A_infinite_l1141_114191


namespace NUMINAMATH_GPT_exam_combinations_l1141_114141

/-- In the "$3+1+2$" examination plan in Hubei Province, 2021,
there are three compulsory subjects: Chinese, Mathematics, and English.
Candidates must choose one subject from Physics and History.
Candidates must choose two subjects from Chemistry, Biology, Ideological and Political Education, and Geography.
Prove that the total number of different combinations of examination subjects is 12.
-/
theorem exam_combinations : exists n : ℕ, n = 12 :=
by
  have compulsory_choice := 1
  have physics_history_choice := 2
  have remaining_subjects_choice := Nat.choose 4 2
  exact Exists.intro (compulsory_choice * physics_history_choice * remaining_subjects_choice) sorry

end NUMINAMATH_GPT_exam_combinations_l1141_114141


namespace NUMINAMATH_GPT_tan_product_pi_over_6_3_2_undefined_l1141_114196

noncomputable def tan_pi_over_6 : ℝ := Real.tan (Real.pi / 6)
noncomputable def tan_pi_over_3 : ℝ := Real.tan (Real.pi / 3)
noncomputable def tan_pi_over_2 : ℝ := Real.tan (Real.pi / 2)

theorem tan_product_pi_over_6_3_2_undefined :
  ∃ (x y : ℝ), Real.tan (Real.pi / 6) = x ∧ Real.tan (Real.pi / 3) = y ∧ Real.tan (Real.pi / 2) = 0 :=
by
  sorry

end NUMINAMATH_GPT_tan_product_pi_over_6_3_2_undefined_l1141_114196


namespace NUMINAMATH_GPT_solve_for_x_l1141_114190

theorem solve_for_x (x : ℝ) (h : 6 * x ^ (1 / 3) - 3 * (x / x ^ (2 / 3)) = -1 + 2 * x ^ (1 / 3) + 4) :
  x = 27 :=
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l1141_114190


namespace NUMINAMATH_GPT_sum_cubes_div_product_eq_three_l1141_114182

-- Given that x, y, z are non-zero real numbers and x + y + z = 3,
-- we need to prove that the possible value of (x^3 + y^3 + z^3) / xyz is 3.

theorem sum_cubes_div_product_eq_three 
  (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (hxyz_sum : x + y + z = 3) : 
  (x^3 + y^3 + z^3) / (x * y * z) = 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_cubes_div_product_eq_three_l1141_114182


namespace NUMINAMATH_GPT_yards_after_8_marathons_l1141_114158

-- Define the constants and conditions
def marathon_miles := 26
def marathon_yards := 395
def yards_per_mile := 1760

-- Definition for total distance covered after 8 marathons
def total_miles := marathon_miles * 8
def total_yards := marathon_yards * 8

-- Convert the total yards into miles with remainder
def extra_miles := total_yards / yards_per_mile
def remainder_yards := total_yards % yards_per_mile

-- Prove the remainder yards is 1400
theorem yards_after_8_marathons : remainder_yards = 1400 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_yards_after_8_marathons_l1141_114158


namespace NUMINAMATH_GPT_dan_helmet_craters_l1141_114100

namespace HelmetCraters

variables {Dan Daniel Rin : ℕ}

/-- Condition 1: Dan's skateboarding helmet has ten more craters than Daniel's ski helmet. -/
def condition1 (C_d C_daniel : ℕ) : Prop := C_d = C_daniel + 10

/-- Condition 2: Rin's snorkel helmet has 15 more craters than Dan's and Daniel's helmets combined. -/
def condition2 (C_r C_d C_daniel : ℕ) : Prop := C_r = C_d + C_daniel + 15

/-- Condition 3: Rin's helmet has 75 craters. -/
def condition3 (C_r : ℕ) : Prop := C_r = 75

/-- The main theorem: Dan's skateboarding helmet has 35 craters given the conditions. -/
theorem dan_helmet_craters (C_d C_daniel C_r : ℕ) 
    (h1 : condition1 C_d C_daniel) 
    (h2 : condition2 C_r C_d C_daniel) 
    (h3 : condition3 C_r) : C_d = 35 :=
by {
    -- We state that the answer is 35 based on the conditions
    sorry
}

end HelmetCraters

end NUMINAMATH_GPT_dan_helmet_craters_l1141_114100


namespace NUMINAMATH_GPT_victor_decks_l1141_114127

theorem victor_decks (V : ℕ) (cost_per_deck total_spent friend_decks : ℕ) 
  (h1 : cost_per_deck = 8)
  (h2 : total_spent = 64)
  (h3 : friend_decks = 2) 
  (h4 : 8 * V + 8 * friend_decks = total_spent) : 
  V = 6 :=
by sorry

end NUMINAMATH_GPT_victor_decks_l1141_114127


namespace NUMINAMATH_GPT_inequality_holds_l1141_114192

theorem inequality_holds (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  x^4 + y^4 + 2 / (x^2 * y^2) ≥ 4 := 
by
  sorry

end NUMINAMATH_GPT_inequality_holds_l1141_114192


namespace NUMINAMATH_GPT_correct_time_fraction_l1141_114177

theorem correct_time_fraction : 
  let hours_with_glitch := [5]
  let minutes_with_glitch := [5, 15, 25, 35, 45, 55]
  let total_hours := 12
  let total_minutes_per_hour := 60
  let correct_hours := total_hours - hours_with_glitch.length
  let correct_minutes := total_minutes_per_hour - minutes_with_glitch.length
  (correct_hours * correct_minutes) / (total_hours * total_minutes_per_hour) = 33 / 40 :=
by
  sorry

end NUMINAMATH_GPT_correct_time_fraction_l1141_114177


namespace NUMINAMATH_GPT_equality_of_integers_l1141_114163

theorem equality_of_integers (a b : ℕ) (h1 : ∀ n : ℕ, ∃ m : ℕ, m > 0 ∧ (a^m + b^m) % (a^n + b^n) = 0) : a = b :=
sorry

end NUMINAMATH_GPT_equality_of_integers_l1141_114163


namespace NUMINAMATH_GPT_correct_options_l1141_114118

-- Given condition
def curve (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2 = 0

-- Option B assertion
def option_B (x y : ℝ) : Prop := (x^2 + y^2 - 4)/((x - 1)^2 + y^2 + 1) ≤ 2 + Real.sqrt 6

-- Option D assertion
def option_D (x y : ℝ) : Prop := x - Real.sqrt 2 * y + 2 = 0

-- Theorem to prove both options B and D are correct under the given condition
theorem correct_options {x y : ℝ} (h : curve x y) : option_B x y ∧ option_D x y := by
  sorry

end NUMINAMATH_GPT_correct_options_l1141_114118


namespace NUMINAMATH_GPT_square_area_l1141_114105

theorem square_area (x : ℝ) (h1 : x = 60) : x^2 = 1200 :=
by
  sorry

end NUMINAMATH_GPT_square_area_l1141_114105


namespace NUMINAMATH_GPT_coins_player_1_received_l1141_114164

def round_table := List Nat
def players := List Nat
def coins_received (table: round_table) (player_idx: Nat) : Nat :=
sorry -- the function to calculate coins received by player's index

-- Define the given conditions
def sectors : round_table := [1, 2, 3, 4, 5, 6, 7, 8, 9]
def num_players := 9
def num_rotations := 11
def player_4 := 4
def player_8 := 8
def player_1 := 1
def coins_player_4 := 90
def coins_player_8 := 35

theorem coins_player_1_received : coins_received sectors player_1 = 57 :=
by
  -- Setup the conditions
  have h1 : coins_received sectors player_4 = 90 := sorry
  have h2 : coins_received sectors player_8 = 35 := sorry
  -- Prove the target statement
  show coins_received sectors player_1 = 57
  sorry

end NUMINAMATH_GPT_coins_player_1_received_l1141_114164


namespace NUMINAMATH_GPT_daniel_total_earnings_l1141_114155

-- Definitions of conditions
def fabric_delivered_monday : ℕ := 20
def fabric_delivered_tuesday : ℕ := 2 * fabric_delivered_monday
def fabric_delivered_wednesday : ℕ := fabric_delivered_tuesday / 4
def total_fabric_delivered : ℕ := fabric_delivered_monday + fabric_delivered_tuesday + fabric_delivered_wednesday

def cost_per_yard : ℕ := 2
def total_earnings : ℕ := total_fabric_delivered * cost_per_yard

-- Proposition to be proved
theorem daniel_total_earnings : total_earnings = 140 := by
  sorry

end NUMINAMATH_GPT_daniel_total_earnings_l1141_114155


namespace NUMINAMATH_GPT_arithmetic_sum_sequence_l1141_114154

theorem arithmetic_sum_sequence (a : ℕ → ℝ) (d : ℝ)
  (h : ∀ n, a (n + 1) = a n + d) :
  ∃ d', 
    a 4 + a 5 + a 6 - (a 1 + a 2 + a 3) = d' ∧
    a 7 + a 8 + a 9 - (a 4 + a 5 + a 6) = d' :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sum_sequence_l1141_114154


namespace NUMINAMATH_GPT_theatre_lost_revenue_l1141_114123

def ticket_price (category : String) : Float :=
  match category with
  | "general" => 10.0
  | "children" => 6.0
  | "senior" => 8.0
  | "veteran" => 8.0  -- $10.00 - $2.00 discount
  | _ => 0.0

def vip_price (base_price : Float) : Float :=
  base_price + 5.0

def calculate_revenue_sold : Float :=
  let general_revenue := 12 * ticket_price "general" + 3 * (vip_price $ ticket_price "general") / 2
  let children_revenue := 3 * ticket_price "children" + vip_price (ticket_price "children")
  let senior_revenue := 4 * ticket_price "senior" + (vip_price (ticket_price "senior")) / 2
  let veteran_revenue := 2 * ticket_price "veteran" + vip_price (ticket_price "veteran")
  general_revenue + children_revenue + senior_revenue + veteran_revenue

def potential_total_revenue : Float :=
  40 * ticket_price "general" + 10 * vip_price (ticket_price "general")

def potential_revenue_lost : Float :=
  potential_total_revenue - calculate_revenue_sold

theorem theatre_lost_revenue : potential_revenue_lost = 224.0 :=
  sorry

end NUMINAMATH_GPT_theatre_lost_revenue_l1141_114123


namespace NUMINAMATH_GPT_compare_neg_fractions_l1141_114184

theorem compare_neg_fractions : (- (2 / 3) < - (1 / 2)) :=
sorry

end NUMINAMATH_GPT_compare_neg_fractions_l1141_114184


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_for_abs_eq_two_l1141_114176

theorem sufficient_but_not_necessary_for_abs_eq_two (a : ℝ) :
  (a = -2 → |a| = 2) ∧ (|a| = 2 → a = 2 ∨ a = -2) :=
by
   sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_for_abs_eq_two_l1141_114176


namespace NUMINAMATH_GPT_inverse_variation_l1141_114159

theorem inverse_variation (k : ℝ) : 
  (∀ (x y : ℝ), x * y^2 = k) → 
  (∀ (x y : ℝ), x = 1 → y = 2 → k = 4) → 
  (x = 0.1111111111111111) → 
  (y = 6) :=
by 
  -- Assume the given conditions
  intros h h0 hx
  -- Proof goes here...
  sorry

end NUMINAMATH_GPT_inverse_variation_l1141_114159


namespace NUMINAMATH_GPT_intersection_complement_eq_l1141_114109

def A : Set ℝ := { x | 1 ≤ x ∧ x < 3 }

def B : Set ℝ := { x | x^2 ≥ 4 }

def complementB : Set ℝ := { x | -2 < x ∧ x < 2 }

def intersection (A : Set ℝ) (B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∈ B }

theorem intersection_complement_eq : 
  intersection A complementB = { x | 1 ≤ x ∧ x < 2 } := 
sorry

end NUMINAMATH_GPT_intersection_complement_eq_l1141_114109


namespace NUMINAMATH_GPT_shaded_to_white_area_ratio_l1141_114199

-- Define the problem
theorem shaded_to_white_area_ratio :
  let total_triangles_shaded := 5
  let total_triangles_white := 3
  let ratio_shaded_to_white := total_triangles_shaded / total_triangles_white
  ratio_shaded_to_white = (5 : ℚ)/(3 : ℚ) := by
  -- Proof steps should be provided here, but "sorry" is used to skip the proof.
  sorry

end NUMINAMATH_GPT_shaded_to_white_area_ratio_l1141_114199


namespace NUMINAMATH_GPT_smallest_n_for_congruence_l1141_114142

theorem smallest_n_for_congruence :
  ∃ n : ℕ, n > 0 ∧ 7 ^ n % 4 = n ^ 7 % 4 ∧ ∀ m : ℕ, (m > 0 ∧ m < n → ¬ (7 ^ m % 4 = m ^ 7 % 4)) :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_congruence_l1141_114142


namespace NUMINAMATH_GPT_surface_area_of_sphere_l1141_114119

theorem surface_area_of_sphere (l w h : ℝ) (s t : ℝ) :
  l = 3 ∧ w = 2 ∧ h = 1 ∧ (s = (l^2 + w^2 + h^2).sqrt / 2) → t = 4 * Real.pi * s^2 → t = 14 * Real.pi :=
by
  intros
  sorry

end NUMINAMATH_GPT_surface_area_of_sphere_l1141_114119


namespace NUMINAMATH_GPT_area_of_given_triangle_l1141_114124

noncomputable def area_of_triangle (a A B : ℝ) : ℝ :=
  let C := Real.pi - A - B
  let b := a * (Real.sin B / Real.sin A)
  let S := (1 / 2) * a * b * Real.sin C
  S

theorem area_of_given_triangle : area_of_triangle 4 (Real.pi / 4) (Real.pi / 3) = 6 + 2 * Real.sqrt 3 := 
by 
  sorry

end NUMINAMATH_GPT_area_of_given_triangle_l1141_114124


namespace NUMINAMATH_GPT_total_clothing_ironed_l1141_114135

-- Definitions based on conditions
def shirts_per_hour := 4
def pants_per_hour := 3
def hours_ironing_shirts := 3
def hours_ironing_pants := 5

-- Theorem statement based on the problem and its solution
theorem total_clothing_ironed : 
  (shirts_per_hour * hours_ironing_shirts + pants_per_hour * hours_ironing_pants) = 27 := 
by
  sorry

end NUMINAMATH_GPT_total_clothing_ironed_l1141_114135


namespace NUMINAMATH_GPT_total_miles_l1141_114187

theorem total_miles (miles_Darius : Int) (miles_Julia : Int) (h1 : miles_Darius = 679) (h2 : miles_Julia = 998) :
  miles_Darius + miles_Julia = 1677 :=
by
  sorry

end NUMINAMATH_GPT_total_miles_l1141_114187


namespace NUMINAMATH_GPT_students_no_A_in_any_subject_l1141_114198

def total_students : ℕ := 50
def a_in_history : ℕ := 9
def a_in_math : ℕ := 15
def a_in_science : ℕ := 12
def a_in_math_and_history : ℕ := 5
def a_in_history_and_science : ℕ := 3
def a_in_science_and_math : ℕ := 4
def a_in_all_three : ℕ := 1

theorem students_no_A_in_any_subject : 
  (total_students - (a_in_history + a_in_math + a_in_science 
                      - a_in_math_and_history - a_in_history_and_science - a_in_science_and_math 
                      + a_in_all_three)) = 28 := by
  sorry

end NUMINAMATH_GPT_students_no_A_in_any_subject_l1141_114198


namespace NUMINAMATH_GPT_units_digit_of_product_is_eight_l1141_114113

def first_four_compos_comps : List Nat := [4, 6, 8, 9]

def product_of_comps : Nat := first_four_compos_comps.foldl (· * ·) 1

theorem units_digit_of_product_is_eight : product_of_comps % 10 = 8 := 
by 
  sorry

end NUMINAMATH_GPT_units_digit_of_product_is_eight_l1141_114113


namespace NUMINAMATH_GPT_third_number_correct_l1141_114138

-- Given that the row of Pascal's triangle with 51 numbers corresponds to the binomial coefficients of 50.
def third_number_in_51_pascal_row : ℕ := Nat.choose 50 2

-- Prove that the third number in this row of Pascal's triangle is 1225.
theorem third_number_correct : third_number_in_51_pascal_row = 1225 := 
by 
  -- Calculation part can be filled in for the full proof.
  sorry

end NUMINAMATH_GPT_third_number_correct_l1141_114138


namespace NUMINAMATH_GPT_polygon_sides_l1141_114108

theorem polygon_sides (n : ℕ) (D : ℕ) (hD : D = 77) (hFormula : D = n * (n - 3) / 2) (hVertex : n = n) : n + 1 = 15 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l1141_114108


namespace NUMINAMATH_GPT_books_sold_at_overall_loss_l1141_114128

-- Defining the conditions and values
def total_cost : ℝ := 540
def C1 : ℝ := 315
def loss_percentage_C1 : ℝ := 0.15
def gain_percentage_C2 : ℝ := 0.19
def C2 : ℝ := total_cost - C1
def loss_C1 := (loss_percentage_C1 * C1)
def SP1 := C1 - loss_C1
def gain_C2 := (gain_percentage_C2 * C2)
def SP2 := C2 + gain_C2
def total_selling_price := SP1 + SP2
def overall_loss := total_cost - total_selling_price

-- Formulating the theorem based on the conditions and required proof
theorem books_sold_at_overall_loss : overall_loss = 4.50 := 
by 
  sorry

end NUMINAMATH_GPT_books_sold_at_overall_loss_l1141_114128


namespace NUMINAMATH_GPT_true_proposition_l1141_114186

-- Define propositions p and q
variable (p q : Prop)

-- Assume p is true and q is false
axiom h1 : p
axiom h2 : ¬q

-- Prove that p ∧ ¬q is true
theorem true_proposition (p q : Prop) (h1 : p) (h2 : ¬q) : p ∧ ¬q :=
by
  sorry

end NUMINAMATH_GPT_true_proposition_l1141_114186


namespace NUMINAMATH_GPT_cubic_function_decreasing_l1141_114195

-- Define the given function
def f (a x : ℝ) : ℝ := a * x^3 - 1

-- Define the condition that the function is decreasing on ℝ
def is_decreasing_on_R (a : ℝ) : Prop :=
  ∀ x : ℝ, 3 * a * x^2 ≤ 0 

-- Main theorem and its statement
theorem cubic_function_decreasing (a : ℝ) (h : is_decreasing_on_R a) : a < 0 :=
sorry

end NUMINAMATH_GPT_cubic_function_decreasing_l1141_114195


namespace NUMINAMATH_GPT_final_price_correct_l1141_114102

-- Define the initial price of the iPhone
def initial_price : ℝ := 1000

-- Define the discount rates for the first and second month
def first_month_discount : ℝ := 0.10
def second_month_discount : ℝ := 0.20

-- Calculate the price after the first month's discount
def price_after_first_month (price : ℝ) : ℝ := price * (1 - first_month_discount)

-- Calculate the price after the second month's discount
def price_after_second_month (price : ℝ) : ℝ := price * (1 - second_month_discount)

-- Final price calculation after both discounts
def final_price : ℝ := price_after_second_month (price_after_first_month initial_price)

-- Proof statement
theorem final_price_correct : final_price = 720 := by
  sorry

end NUMINAMATH_GPT_final_price_correct_l1141_114102


namespace NUMINAMATH_GPT_mean_value_of_interior_angles_pentagon_l1141_114194

def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

theorem mean_value_of_interior_angles_pentagon :
  sum_of_interior_angles 5 / 5 = 108 :=
by
  sorry

end NUMINAMATH_GPT_mean_value_of_interior_angles_pentagon_l1141_114194


namespace NUMINAMATH_GPT_find_a_value_l1141_114167

-- Define the problem conditions
theorem find_a_value (a : ℝ) :
  let x_values := [0, 1, 3, 4]
  let y_values := [a, 4.3, 4.8, 6.7]
  let mean_x := (0 + 1 + 3 + 4) / 4
  let mean_y := (a + 4.3 + 4.8 + 6.7) / 4
  (mean_y = 0.95 * mean_x + 2.6) → a = 2.2 :=
by
  -- Let bindings are for convenience to follow the problem statement
  let x_values := [0, 1, 3, 4]
  let y_values := [a, 4.3, 4.8, 6.7]
  let mean_x := (0 + 1 + 3 + 4) / 4
  let mean_y := (a + 4.3 + 4.8 + 6.7) / 4
  intro h
  sorry

end NUMINAMATH_GPT_find_a_value_l1141_114167


namespace NUMINAMATH_GPT_photo_gallery_total_l1141_114117

theorem photo_gallery_total (initial_photos: ℕ) (first_day_photos: ℕ) (second_day_photos: ℕ)
  (h_initial: initial_photos = 400) 
  (h_first_day: first_day_photos = initial_photos / 2)
  (h_second_day: second_day_photos = first_day_photos + 120) : 
  initial_photos + first_day_photos + second_day_photos = 920 :=
by
  sorry

end NUMINAMATH_GPT_photo_gallery_total_l1141_114117


namespace NUMINAMATH_GPT_pq_or_l1141_114131

def p : Prop := 2 % 2 = 0
def q : Prop := 3 % 2 = 0

theorem pq_or : p ∨ q :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_pq_or_l1141_114131


namespace NUMINAMATH_GPT_box_height_is_6_l1141_114183

-- Defining the problem setup
variables (h : ℝ) (r_large r_small : ℝ)
variables (box_size : ℝ) (n_spheres : ℕ)

-- The conditions of the problem
def rectangular_box :=
  box_size = 5 ∧ r_large = 3 ∧ r_small = 1.5 ∧ n_spheres = 4 ∧
  (∀ k : ℕ, k < n_spheres → 
   ∃ C : ℝ, 
     (C = r_small) ∧ 
     -- Each smaller sphere is tangent to three sides of the box condition
     (C ≤ box_size))

def sphere_tangency (h r_large r_small : ℝ) :=
  h = 2 * r_large ∧ r_large + r_small = 4.5

def height_of_box (h : ℝ) := 2 * 3 = h

-- The mathematically equivalent proof problem
theorem box_height_is_6 (h : ℝ) (r_large : ℝ) (r_small : ℝ) (box_size : ℝ) (n_spheres : ℕ) 
  (conditions : rectangular_box box_size r_large r_small n_spheres) 
  (tangency : sphere_tangency h r_large r_small) :
  height_of_box h :=
by {
  -- Proof is omitted
  sorry
}

end NUMINAMATH_GPT_box_height_is_6_l1141_114183


namespace NUMINAMATH_GPT_atomic_weight_of_Calcium_l1141_114152

/-- Given definitions -/
def molecular_weight_CaOH₂ : ℕ := 74
def atomic_weight_O : ℕ := 16
def atomic_weight_H : ℕ := 1

/-- Given conditions -/
def total_weight_O_H : ℕ := 2 * atomic_weight_O + 2 * atomic_weight_H

/-- Problem statement -/
theorem atomic_weight_of_Calcium (H1 : molecular_weight_CaOH₂ = 74)
                                   (H2 : atomic_weight_O = 16)
                                   (H3 : atomic_weight_H = 1)
                                   (H4 : total_weight_O_H = 2 * atomic_weight_O + 2 * atomic_weight_H) :
  74 - (2 * 16 + 2 * 1) = 40 :=
by {
  sorry
}

end NUMINAMATH_GPT_atomic_weight_of_Calcium_l1141_114152


namespace NUMINAMATH_GPT_find_c_l1141_114178

def p (x : ℝ) : ℝ := 3 * x - 8
def q (x : ℝ) (c : ℝ) : ℝ := 5 * x - c

theorem find_c (c : ℝ) (h : p (q 3 c) = 14) : c = 23 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l1141_114178


namespace NUMINAMATH_GPT_find_a_l1141_114169

namespace MathProof

theorem find_a (a : ℕ) (h_pos : a > 0) (h_eq : (a : ℚ) / (a + 18) = 47 / 50) : a = 282 :=
by
  sorry

end MathProof

end NUMINAMATH_GPT_find_a_l1141_114169


namespace NUMINAMATH_GPT_friedas_probability_to_corner_l1141_114153

-- Define the grid size and positions
def grid_size : Nat := 4
def start_position : ℕ × ℕ := (3, 3)
def corner_positions : List (ℕ × ℕ) := [(1, 1), (1, 4), (4, 1), (4, 4)]

-- Define the number of hops allowed
def max_hops : Nat := 4

-- Define a function to calculate the probability of reaching a corner square
-- within the given number of hops starting from the initial position.
noncomputable def prob_reach_corner (grid_size : ℕ) (start_position : ℕ × ℕ) 
                                     (corner_positions : List (ℕ × ℕ)) 
                                     (max_hops : ℕ) : ℚ :=
  -- Implementation details skipped
  sorry

-- Define the main theorem that states the desired probability
theorem friedas_probability_to_corner : 
  prob_reach_corner grid_size start_position corner_positions max_hops = 17 / 64 :=
sorry

end NUMINAMATH_GPT_friedas_probability_to_corner_l1141_114153


namespace NUMINAMATH_GPT_first_discount_percentage_l1141_114126

theorem first_discount_percentage (original_price : ℝ) (final_price : ℝ) (additional_discount : ℝ) (first_discount : ℝ) : 
  original_price = 600 → final_price = 513 → additional_discount = 0.05 →
  600 * (1 - first_discount / 100) * (1 - 0.05) = 513 →
  first_discount = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_first_discount_percentage_l1141_114126


namespace NUMINAMATH_GPT_intersection_M_N_l1141_114166

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} :=
sorry

end NUMINAMATH_GPT_intersection_M_N_l1141_114166


namespace NUMINAMATH_GPT_sum_first_six_terms_l1141_114136

variable {S : ℕ → ℝ}

theorem sum_first_six_terms (h2 : S 2 = 4) (h4 : S 4 = 6) : S 6 = 7 := 
  sorry

end NUMINAMATH_GPT_sum_first_six_terms_l1141_114136


namespace NUMINAMATH_GPT_find_r_l1141_114110

theorem find_r (a b m p r : ℝ) (h_roots1 : a * b = 6) 
  (h_eq1 : ∀ x, x^2 - m*x + 6 = 0) 
  (h_eq2 : ∀ x, x^2 - p*x + r = 0) :
  r = 32 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_r_l1141_114110


namespace NUMINAMATH_GPT_find_expression_value_l1141_114151

theorem find_expression_value (m : ℝ) (h : m + 1/m = 10) : m^2 + 1/m^2 + m^3 + 1/m^3 + 4 = 1072 := 
by 
  sorry

end NUMINAMATH_GPT_find_expression_value_l1141_114151


namespace NUMINAMATH_GPT_phillip_initial_marbles_l1141_114193

theorem phillip_initial_marbles
  (dilan_marbles : ℕ) (martha_marbles : ℕ) (veronica_marbles : ℕ) 
  (total_after_redistribution : ℕ) 
  (individual_marbles_after : ℕ) :
  dilan_marbles = 14 →
  martha_marbles = 20 →
  veronica_marbles = 7 →
  total_after_redistribution = 4 * individual_marbles_after →
  individual_marbles_after = 15 →
  ∃phillip_marbles : ℕ, phillip_marbles = 19 :=
by
  intro h_dilan h_martha h_veronica h_total_after h_individual
  have total_initial := 60 - (14 + 20 + 7)
  existsi total_initial
  sorry

end NUMINAMATH_GPT_phillip_initial_marbles_l1141_114193


namespace NUMINAMATH_GPT_how_much_together_l1141_114160

def madeline_money : ℕ := 48
def brother_money : ℕ := madeline_money / 2

theorem how_much_together : madeline_money + brother_money = 72 := by
  sorry

end NUMINAMATH_GPT_how_much_together_l1141_114160


namespace NUMINAMATH_GPT_find_k_l1141_114112

theorem find_k : 
  ∃ (k : ℚ), 
    (∃ (x y : ℚ), y = 3 * x + 7 ∧ y = -4 * x + 1) ∧ 
    ∃ (x y : ℚ), y = 3 * x + 7 ∧ y = 2 * x + k ∧ k = 43 / 7 := 
sorry

end NUMINAMATH_GPT_find_k_l1141_114112


namespace NUMINAMATH_GPT_first_house_bottles_l1141_114189

theorem first_house_bottles (total_bottles : ℕ) 
  (cider_only : ℕ) (beer_only : ℕ) (half : ℕ → ℕ)
  (mixture : ℕ)
  (half_cider_bottles : ℕ)
  (half_beer_bottles : ℕ)
  (half_mixture_bottles : ℕ) : 
  total_bottles = 180 →
  cider_only = 40 →
  beer_only = 80 →
  mixture = total_bottles - (cider_only + beer_only) →
  half c = c / 2 →
  half_cider_bottles = half cider_only →
  half_beer_bottles = half beer_only →
  half_mixture_bottles = half mixture →
  half_cider_bottles + half_beer_bottles + half_mixture_bottles = 90 :=
by
  intros h_tot h_cid h_beer h_mix h_half half_cid half_beer half_mix
  sorry

end NUMINAMATH_GPT_first_house_bottles_l1141_114189


namespace NUMINAMATH_GPT_at_least_two_consecutive_heads_probability_l1141_114140

noncomputable def probability_at_least_two_consecutive_heads : ℚ := 
  let total_outcomes := 16
  let unfavorable_outcomes := 8
  1 - (unfavorable_outcomes / total_outcomes)

theorem at_least_two_consecutive_heads_probability :
  probability_at_least_two_consecutive_heads = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_at_least_two_consecutive_heads_probability_l1141_114140


namespace NUMINAMATH_GPT_two_dollar_coin_is_toonie_l1141_114134

/-- We define the $2 coin in Canada -/
def two_dollar_coin_name : String := "toonie"

/-- Antonella's wallet problem setup -/
def Antonella_has_ten_coins := 10
def loonies_value := 1
def toonies_value := 2
def coins_after_purchase := 11
def purchase_amount := 3
def initial_toonies := 4

/-- Proving that the $2 coin is called a "toonie" -/
theorem two_dollar_coin_is_toonie :
  two_dollar_coin_name = "toonie" :=
by
  -- Here, we place the logical steps to derive that two_dollar_coin_name = "toonie"
  sorry

end NUMINAMATH_GPT_two_dollar_coin_is_toonie_l1141_114134


namespace NUMINAMATH_GPT_max_sum_of_arithmetic_sequence_l1141_114148

theorem max_sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) 
(h1 : 3 * a 8 = 5 * a 13) 
(h2 : a 1 > 0)
(hS : ∀ n, S n = n / 2 * (2 * a 1 + (n - 1) * d)) :
S 20 > S 21 ∧ S 20 > S 10 ∧ S 20 > S 11 :=
sorry

end NUMINAMATH_GPT_max_sum_of_arithmetic_sequence_l1141_114148


namespace NUMINAMATH_GPT_John_meeting_percentage_l1141_114107

def hours_to_minutes (h : ℕ) : ℕ := 60 * h

def first_meeting_duration : ℕ := 30
def second_meeting_duration : ℕ := 60
def third_meeting_duration : ℕ := 2 * first_meeting_duration
def total_meeting_duration : ℕ := first_meeting_duration + second_meeting_duration + third_meeting_duration

def total_workday_duration : ℕ := hours_to_minutes 12

def percentage_of_meetings (total_meeting_time total_workday_time : ℕ) : ℕ := 
  (total_meeting_time * 100) / total_workday_time

theorem John_meeting_percentage : 
  percentage_of_meetings total_meeting_duration total_workday_duration = 21 :=
by
  sorry

end NUMINAMATH_GPT_John_meeting_percentage_l1141_114107
