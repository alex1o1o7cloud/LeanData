import Mathlib

namespace shaded_area_l2008_200813

noncomputable def area_of_shaded_region (AB : ℝ) (pi_approx : ℝ) : ℝ :=
  let R := AB / 2
  let r := R / 2
  let A_large := (1/2) * pi_approx * R^2
  let A_small := (1/2) * pi_approx * r^2
  2 * A_large - 4 * A_small

theorem shaded_area (h : area_of_shaded_region 40 3.14 = 628) : true :=
  sorry

end shaded_area_l2008_200813


namespace time_for_one_large_division_l2008_200883

/-- The clock face is divided into 12 equal parts by the 12 numbers (12 large divisions). -/
def num_large_divisions : ℕ := 12

/-- Each large division is further divided into 5 small divisions. -/
def num_small_divisions_per_large : ℕ := 5

/-- The second hand moves 1 small division every second. -/
def seconds_per_small_division : ℕ := 1

/-- The time it takes for the second hand to move one large division is 5 seconds. -/
def time_per_large_division : ℕ := num_small_divisions_per_large * seconds_per_small_division

theorem time_for_one_large_division : time_per_large_division = 5 := by
  sorry

end time_for_one_large_division_l2008_200883


namespace calculation_is_correct_l2008_200804

theorem calculation_is_correct : 450 / (6 * 5 - 10 / 2) = 18 :=
by {
  -- Let me provide an outline for solving this problem
  -- (6 * 5 - 10 / 2) must be determined first
  -- After that substituted into the fraction
  sorry
}

end calculation_is_correct_l2008_200804


namespace linear_eq_a_value_l2008_200887

theorem linear_eq_a_value (a : ℤ) (x : ℝ) 
  (h : x^(a-1) - 5 = 3) 
  (h_lin : ∃ b c : ℝ, x^(a-1) * b + c = 0 ∧ b ≠ 0):
  a = 2 :=
sorry

end linear_eq_a_value_l2008_200887


namespace percentage_of_only_cat_owners_l2008_200842

theorem percentage_of_only_cat_owners (total_students total_dog_owners total_cat_owners both_cat_dog_owners : ℕ) 
(h_total_students : total_students = 500)
(h_total_dog_owners : total_dog_owners = 120)
(h_total_cat_owners : total_cat_owners = 80)
(h_both_cat_dog_owners : both_cat_dog_owners = 40) :
( (total_cat_owners - both_cat_dog_owners : ℕ) * 100 / total_students ) = 8 := 
by
  sorry

end percentage_of_only_cat_owners_l2008_200842


namespace max_volume_cylinder_l2008_200800

theorem max_volume_cylinder (x : ℝ) (h1 : x > 0) (h2 : x < 10) : 
  (∀ x, 0 < x ∧ x < 10 → ∃ max_v, max_v = (4 * (10^3) * Real.pi) / 27) ∧ 
  ∃ x, x = 20/3 := 
by
  sorry

end max_volume_cylinder_l2008_200800


namespace relationship_among_abc_l2008_200823

theorem relationship_among_abc (e1 e2 : ℝ) (h1 : 0 ≤ e1) (h2 : e1 < 1) (h3 : e2 > 1) :
  let a := 3 ^ e1
  let b := 2 ^ (-e2)
  let c := Real.sqrt 5
  b < c ∧ c < a := by
  sorry

end relationship_among_abc_l2008_200823


namespace betty_age_l2008_200868

theorem betty_age : ∀ (A M B : ℕ), A = 2 * M → A = 4 * B → M = A - 10 → B = 5 :=
by
  intros A M B h1 h2 h3
  sorry

end betty_age_l2008_200868


namespace problem_statement_l2008_200899

variable {x y z : ℝ}

theorem problem_statement
  (h : x^2 + y^2 + z^2 + 9 = 4 * (x + y + z)) :
  x^4 + y^4 + z^4 + 16 * (x^2 + y^2 + z^2) ≥ 8 * (x^3 + y^3 + z^3) + 27 :=
by
  sorry

end problem_statement_l2008_200899


namespace A_share_of_profit_l2008_200801

-- Define necessary financial terms and operations
def initial_investment_A := 3000
def initial_investment_B := 4000

def withdrawal_A := 1000
def advanced_B := 1000

def duration_initial := 8
def duration_remaining := 4

def total_profit := 630

-- Calculate the equivalent investment duration for A and B
def investment_months_A_first := initial_investment_A * duration_initial
def investment_months_A_remaining := (initial_investment_A - withdrawal_A) * duration_remaining
def investment_months_A := investment_months_A_first + investment_months_A_remaining

def investment_months_B_first := initial_investment_B * duration_initial
def investment_months_B_remaining := (initial_investment_B + advanced_B) * duration_remaining
def investment_months_B := investment_months_B_first + investment_months_B_remaining

-- Prove that A's share of the profit is Rs. 240
theorem A_share_of_profit : 
  let ratio_A : ℚ := 4
  let ratio_B : ℚ := 6.5
  let total_ratio : ℚ := ratio_A + ratio_B
  let a_share : ℚ := (total_profit * ratio_A) / total_ratio
  a_share = 240 := 
by
  sorry

end A_share_of_profit_l2008_200801


namespace problem_sol_l2008_200830

-- Assume g is an invertible function
variable (g : ℝ → ℝ) (g_inv : ℝ → ℝ)
variable (h_invertible : ∀ y, g (g_inv y) = y ∧ g_inv (g y) = y)

-- Define p and q such that g(p) = 3 and g(q) = 5
variable (p q : ℝ)
variable (h1 : g p = 3) (h2 : g q = 5)

-- Goal to prove that p - q = 2
theorem problem_sol : p - q = 2 :=
by
  sorry

end problem_sol_l2008_200830


namespace point_on_line_l2008_200837

theorem point_on_line (s : ℝ) : 
  (∃ b : ℝ, ∀ x y : ℝ, (y = 3 * x + b) → 
    ((2 = x ∧ y = 8) ∨ (4 = x ∧ y = 14) ∨ (6 = x ∧ y = 20) ∨ (35 = x ∧ y = s))) → s = 107 :=
by
  sorry

end point_on_line_l2008_200837


namespace smallest_common_multiple_l2008_200822

theorem smallest_common_multiple (b : ℕ) (hb : b > 0) (h1 : b % 6 = 0) (h2 : b % 15 = 0) :
    b = 30 :=
sorry

end smallest_common_multiple_l2008_200822


namespace inverse_contrapositive_l2008_200897

theorem inverse_contrapositive (a b : ℝ) (h : a ≠ 0 ∨ b ≠ 0) : a^2 + b^2 ≠ 0 :=
sorry

end inverse_contrapositive_l2008_200897


namespace barrel_to_cask_ratio_l2008_200833

theorem barrel_to_cask_ratio
  (k : ℕ) -- k is the multiple
  (B C : ℕ) -- B is the amount a barrel can store, C is the amount a cask can store
  (h1 : C = 20) -- C stores 20 gallons
  (h2 : B = k * C + 3) -- A barrel stores 3 gallons more than k times the amount a cask stores
  (h3 : 4 * B + C = 172) -- The total storage capacity is 172 gallons
  : B / C = 19 / 10 :=
sorry

end barrel_to_cask_ratio_l2008_200833


namespace compare_neg_sqrt_l2008_200845

theorem compare_neg_sqrt :
  -5 > -Real.sqrt 26 := 
sorry

end compare_neg_sqrt_l2008_200845


namespace evaluate_expression_l2008_200832

theorem evaluate_expression : 
  (2 ^ 2003 * 3 ^ 2002 * 5) / (6 ^ 2003) = (5 / 3) :=
by sorry

end evaluate_expression_l2008_200832


namespace estimate_nearsighted_students_l2008_200860

theorem estimate_nearsighted_students (sample_size total_students nearsighted_sample : ℕ) 
  (h_sample_size : sample_size = 30)
  (h_total_students : total_students = 400)
  (h_nearsighted_sample : nearsighted_sample = 12):
  (total_students * nearsighted_sample) / sample_size = 160 := by
  sorry

end estimate_nearsighted_students_l2008_200860


namespace common_chord_condition_l2008_200856

theorem common_chord_condition 
    (h d1 d2 : ℝ) (C1 C2 D1 D2 : ℝ) 
    (hyp_len : (C1 * D1 = C2 * D2)) : 
    (C1 * D1 = C2 * D2) ↔ (1 / h^2 = 1 / d1^2 + 1 / d2^2) :=
by
  sorry

end common_chord_condition_l2008_200856


namespace time_between_ticks_at_6_l2008_200876

def intervals_12 := 11
def ticks_12 := 12
def seconds_12 := 77
def intervals_6 := 5
def ticks_6 := 6

theorem time_between_ticks_at_6 :
  let interval_time := seconds_12 / intervals_12
  let total_time_6 := intervals_6 * interval_time
  total_time_6 = 35 := sorry

end time_between_ticks_at_6_l2008_200876


namespace point_on_x_axis_equidistant_from_A_and_B_is_M_l2008_200867

theorem point_on_x_axis_equidistant_from_A_and_B_is_M :
  ∃ M : ℝ × ℝ × ℝ, (M = (-3 / 2, 0, 0)) ∧ 
  (dist M (1, -3, 1) = dist M (2, 0, 2)) := by
  sorry

end point_on_x_axis_equidistant_from_A_and_B_is_M_l2008_200867


namespace largest_digit_divisible_by_6_l2008_200855

theorem largest_digit_divisible_by_6 :
  ∃ N : ℕ, N ≤ 9 ∧ 4517 * 10 + N % 6 = 0 ∧ ∀ m : ℕ, m ≤ 9 ∧ 4517 * 10 + m % 6 = 0 → m ≤ N :=
by
  -- Proof omitted, replace with actual proof
  sorry

end largest_digit_divisible_by_6_l2008_200855


namespace set_intersection_complement_l2008_200859

open Set

universe u

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

/-- Given the universal set U={0,1,2,3,4,5}, sets A={0,2,4}, and B={0,5}, prove that
    the intersection of A and the complement of B in U is {2,4}. -/
theorem set_intersection_complement:
  U = {0, 1, 2, 3, 4, 5} →
  A = {0, 2, 4} →
  B = {0, 5} →
  A ∩ (U \ B) = {2, 4} := 
by
  intros hU hA hB
  sorry

end set_intersection_complement_l2008_200859


namespace min_value_of_reciprocals_l2008_200866

theorem min_value_of_reciprocals (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : m + n = 2) :
  (1 / m + 1 / n) = 2 :=
sorry

end min_value_of_reciprocals_l2008_200866


namespace total_spent_l2008_200811

theorem total_spent (deck_price : ℕ) (victor_decks : ℕ) (friend_decks : ℕ)
  (h1 : deck_price = 8)
  (h2 : victor_decks = 6)
  (h3 : friend_decks = 2) :
  deck_price * victor_decks + deck_price * friend_decks = 64 :=
by
  sorry

end total_spent_l2008_200811


namespace reciprocal_roots_condition_l2008_200803

theorem reciprocal_roots_condition (a b c : ℝ) (h : a ≠ 0) (roots_reciprocal : ∃ r s : ℝ, r * s = 1 ∧ r + s = -b/a ∧ r * s = c/a) : c = a :=
by
  sorry

end reciprocal_roots_condition_l2008_200803


namespace distance_to_campground_l2008_200805

-- definitions for speeds and times
def speed1 : ℤ := 50
def time1 : ℤ := 3
def speed2 : ℤ := 60
def time2 : ℤ := 2
def speed3 : ℤ := 55
def time3 : ℤ := 1
def speed4 : ℤ := 65
def time4 : ℤ := 2

-- definitions for calculating the distances
def distance1 : ℤ := speed1 * time1
def distance2 : ℤ := speed2 * time2
def distance3 : ℤ := speed3 * time3
def distance4 : ℤ := speed4 * time4

-- definition for the total distance
def total_distance : ℤ := distance1 + distance2 + distance3 + distance4

-- proof statement
theorem distance_to_campground : total_distance = 455 := by
  sorry -- proof omitted

end distance_to_campground_l2008_200805


namespace number_is_correct_l2008_200825

theorem number_is_correct (x : ℝ) (h : 0.35 * x = 0.25 * 50) : x = 35.7143 :=
by 
  sorry

end number_is_correct_l2008_200825


namespace smallest_solution_of_quartic_l2008_200824

theorem smallest_solution_of_quartic :
  ∃ x : ℝ, x^4 - 40*x^2 + 144 = 0 ∧ ∀ y : ℝ, (y^4 - 40*y^2 + 144 = 0) → x ≤ y :=
sorry

end smallest_solution_of_quartic_l2008_200824


namespace at_least_one_not_less_than_2_l2008_200827

theorem at_least_one_not_less_than_2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) :=
sorry

end at_least_one_not_less_than_2_l2008_200827


namespace circles_externally_tangent_l2008_200880

theorem circles_externally_tangent :
  let C1x := -3
  let C1y := 2
  let r1 := 2
  let C2x := 3
  let C2y := -6
  let r2 := 8
  let d := Real.sqrt ((C2x - C1x)^2 + (C2y - C1y)^2)
  (d = r1 + r2) → 
  ((x + 3)^2 + (y - 2)^2 = 4) → ((x - 3)^2 + (y + 6)^2 = 64) → 
  ∃ (P : ℝ × ℝ), (P.1 + 3)^2 + (P.2 - 2)^2 = 4 ∧ (P.1 - 3)^2 + (P.2 + 6)^2 = 64 :=
by
  intros
  sorry

end circles_externally_tangent_l2008_200880


namespace find_seating_capacity_l2008_200834

noncomputable def seating_capacity (buses : ℕ) (students_left : ℤ) : ℤ :=
  buses * 40 + students_left

theorem find_seating_capacity :
  (seating_capacity 4 30) = (seating_capacity 5 (-10)) :=
by
  -- Proof is not required, hence omitted.
  sorry

end find_seating_capacity_l2008_200834


namespace ravi_overall_profit_l2008_200820

-- Define the cost price of the refrigerator and the mobile phone
def cost_price_refrigerator : ℝ := 15000
def cost_price_mobile_phone : ℝ := 8000

-- Define the loss percentage for the refrigerator and the profit percentage for the mobile phone
def loss_percentage_refrigerator : ℝ := 0.05
def profit_percentage_mobile_phone : ℝ := 0.10

-- Calculate the loss amount and the selling price of the refrigerator
def loss_amount_refrigerator : ℝ := loss_percentage_refrigerator * cost_price_refrigerator
def selling_price_refrigerator : ℝ := cost_price_refrigerator - loss_amount_refrigerator

-- Calculate the profit amount and the selling price of the mobile phone
def profit_amount_mobile_phone : ℝ := profit_percentage_mobile_phone * cost_price_mobile_phone
def selling_price_mobile_phone : ℝ := cost_price_mobile_phone + profit_amount_mobile_phone

-- Calculate the total cost price and the total selling price
def total_cost_price : ℝ := cost_price_refrigerator + cost_price_mobile_phone
def total_selling_price : ℝ := selling_price_refrigerator + selling_price_mobile_phone

-- Calculate the overall profit or loss
def overall_profit_or_loss : ℝ := total_selling_price - total_cost_price

theorem ravi_overall_profit : overall_profit_or_loss = 50 := 
by
  sorry

end ravi_overall_profit_l2008_200820


namespace octal_to_decimal_7564_l2008_200848

theorem octal_to_decimal_7564 : 7 * 8^3 + 5 * 8^2 + 6 * 8^1 + 4 * 8^0 = 3956 :=
by
  sorry 

end octal_to_decimal_7564_l2008_200848


namespace ryegrass_percent_of_mixture_l2008_200875

noncomputable def mixture_percent_ryegrass (X_rye Y_rye portion_X : ℝ) : ℝ :=
  let portion_Y := 1 - portion_X
  let total_rye := (X_rye * portion_X) + (Y_rye * portion_Y)
  total_rye * 100

theorem ryegrass_percent_of_mixture :
  let X_rye := 40 / 100 
  let Y_rye := 25 / 100
  let portion_X := 1 / 3
  mixture_percent_ryegrass X_rye Y_rye portion_X = 30 :=
by
  sorry

end ryegrass_percent_of_mixture_l2008_200875


namespace john_salary_increase_l2008_200888

theorem john_salary_increase :
  let initial_salary : ℝ := 30
  let final_salary : ℝ := ((30 * 1.1) * 1.15) * 1.05
  (final_salary - initial_salary) / initial_salary * 100 = 32.83 := by
  sorry

end john_salary_increase_l2008_200888


namespace sum_of_ages_l2008_200893

-- Definitions of John's age and father's age according to the given conditions
def John's_age := 15
def Father's_age := 2 * John's_age + 32

-- The proof problem statement
theorem sum_of_ages : John's_age + Father's_age = 77 :=
by
  -- Here we would substitute and simplify according to the given conditions
  sorry

end sum_of_ages_l2008_200893


namespace hare_total_distance_l2008_200872

-- Define the conditions
def distance_between_trees : ℕ := 5
def number_of_trees : ℕ := 10

-- Define the question to be proved
theorem hare_total_distance : distance_between_trees * (number_of_trees - 1) = 45 :=
by
  sorry

end hare_total_distance_l2008_200872


namespace heather_heavier_than_emily_l2008_200854

theorem heather_heavier_than_emily :
  let Heather_weight := 87
  let Emily_weight := 9
  Heather_weight - Emily_weight = 78 :=
by
  -- Proof here
  sorry

end heather_heavier_than_emily_l2008_200854


namespace total_time_is_10_l2008_200841

-- Definitions based on conditions
def total_distance : ℕ := 224
def first_half_distance : ℕ := total_distance / 2
def second_half_distance : ℕ := total_distance / 2
def speed_first_half : ℕ := 21
def speed_second_half : ℕ := 24

-- Definition of time taken for each half of the journey
def time_first_half : ℚ := first_half_distance / speed_first_half
def time_second_half : ℚ := second_half_distance / speed_second_half

-- Total time is the sum of time taken for each half
def total_time : ℚ := time_first_half + time_second_half

-- Theorem stating the total time taken for the journey
theorem total_time_is_10 : total_time = 10 := by
  sorry

end total_time_is_10_l2008_200841


namespace Jason_current_cards_l2008_200885

-- Definitions based on the conditions
def Jason_original_cards : ℕ := 676
def cards_bought_by_Alyssa : ℕ := 224

-- Problem statement: Prove that Jason's current number of Pokemon cards is 452
theorem Jason_current_cards : Jason_original_cards - cards_bought_by_Alyssa = 452 := by
  sorry

end Jason_current_cards_l2008_200885


namespace expand_polynomial_l2008_200863

theorem expand_polynomial (x : ℝ) : (x + 4) * (5 * x - 10) = 5 * x ^ 2 + 10 * x - 40 := by
  sorry

end expand_polynomial_l2008_200863


namespace production_days_l2008_200861

theorem production_days (n : ℕ) (P : ℕ) (H1 : P = n * 50) (H2 : (P + 90) / (n + 1) = 52) : n = 19 :=
by
  sorry

end production_days_l2008_200861


namespace alicia_tax_deduction_is_50_cents_l2008_200896

def alicia_hourly_wage_dollars : ℝ := 25
def deduction_rate : ℝ := 0.02

def alicia_hourly_wage_cents : ℝ := alicia_hourly_wage_dollars * 100
def tax_deduction_cents : ℝ := alicia_hourly_wage_cents * deduction_rate

theorem alicia_tax_deduction_is_50_cents : tax_deduction_cents = 50 := by
  sorry

end alicia_tax_deduction_is_50_cents_l2008_200896


namespace find_number_l2008_200891

theorem find_number (a b some_number : ℕ) (h1 : a = 69842) (h2 : b = 30158) (h3 : (a^2 - b^2) / some_number = 100000) : some_number = 39684 :=
by {
  -- Proof skipped
  sorry
}

end find_number_l2008_200891


namespace proportion_check_option_B_l2008_200892

theorem proportion_check_option_B (a b c d : ℝ) (ha : a = 1) (hb : b = 2) (hc : c = 2) (hd : d = 4) :
  (a / b) = (c / d) :=
by {
  sorry
}

end proportion_check_option_B_l2008_200892


namespace exists_positive_integer_n_with_N_distinct_prime_factors_l2008_200870

open Nat

/-- Let \( N \) be a positive integer. Prove that there exists a positive integer \( n \) such that \( n^{2013} - n^{20} + n^{13} - 2013 \) has at least \( N \) distinct prime factors. -/
theorem exists_positive_integer_n_with_N_distinct_prime_factors (N : ℕ) (h : 0 < N) : 
  ∃ n : ℕ, 0 < n ∧ (n ^ 2013 - n ^ 20 + n ^ 13 - 2013).primeFactors.card ≥ N :=
sorry

end exists_positive_integer_n_with_N_distinct_prime_factors_l2008_200870


namespace constant_term_in_binomial_expansion_is_40_l2008_200821

-- Define the binomial coefficient C(n, k)
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the expression for the binomial expansion of (x^2 + 2/x^3)^5
def term (r : ℕ) : ℕ := binom 5 r * 2^r

theorem constant_term_in_binomial_expansion_is_40 
  (x : ℝ) (h : x ≠ 0) : 
  (∃ r : ℕ, 10 - 5 * r = 0) ∧ term 2 = 40 :=
by 
  sorry

end constant_term_in_binomial_expansion_is_40_l2008_200821


namespace evaluate_expression_l2008_200851

theorem evaluate_expression :
  (-3)^4 - (-3)^3 + (-3)^2 - 3^2 + 3^3 - 3^4 = 9 :=
by
  sorry

end evaluate_expression_l2008_200851


namespace airsickness_related_to_gender_l2008_200812

def a : ℕ := 28
def b : ℕ := 28
def c : ℕ := 28
def d : ℕ := 56
def n : ℕ := 140

def contingency_relation (a b c d n K2 : ℕ) : Prop := 
  let numerator := n * (a * d - b * c)^2
  let denominator := (a + b) * (c + d) * (a + c) * (b + d)
  K2 > 3841 / 1000

-- Goal statement for the proof
theorem airsickness_related_to_gender :
  contingency_relation a b c d n 3888 :=
  sorry

end airsickness_related_to_gender_l2008_200812


namespace min_tan_of_acute_angle_l2008_200819

def is_ocular_ray (u : ℚ) (x y : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 20 ∧ 1 ≤ y ∧ y ≤ 20 ∧ u = x / y

def acute_angle_tangent (u v : ℚ) : ℚ :=
  |(u - v) / (1 + u * v)|

theorem min_tan_of_acute_angle :
  ∃ θ : ℚ, (∀ u v : ℚ, (∃ x1 y1 x2 y2 : ℕ, is_ocular_ray u x1 y1 ∧ is_ocular_ray v x2 y2 ∧ u ≠ v) 
  → acute_angle_tangent u v ≥ θ) ∧ θ = 1 / 722 :=
sorry

end min_tan_of_acute_angle_l2008_200819


namespace initial_volume_proof_l2008_200890

-- Definitions for initial mixture and ratios
variables (x : ℕ)

def initial_milk := 4 * x
def initial_water := x
def initial_volume := initial_milk x + initial_water x

def add_water (water_added : ℕ) := initial_water x + water_added

def resulting_ratio := initial_milk x / add_water x 9 = 2

theorem initial_volume_proof (h : resulting_ratio x) : initial_volume x = 45 :=
by sorry

end initial_volume_proof_l2008_200890


namespace incorrect_statements_l2008_200846

-- Definitions for the points
def A := (-2, -3) 
def P := (1, 1)
def pt := (1, 3)

-- Definitions for the equations in the statements
def equationA (x y : ℝ) := x + y + 5 = 0
def equationB (m x y : ℝ) := 2*(m+1)*x + (m-3)*y + 7 - 5*m = 0
def equationC (θ x y : ℝ) := y - 1 = Real.tan θ * (x - 1)
def equationD (x₁ y₁ x₂ y₂ x y : ℝ) := (x₂ - x₁)*(y - y₁) = (y₂ - y₁)*(x - x₁)

-- Points of interest
def xA : ℝ := -2
def yA : ℝ := -3
def xP : ℝ := 1
def yP : ℝ := 1
def pt_x : ℝ := 1
def pt_y : ℝ := 3

-- Main proof to show which statements are incorrect
theorem incorrect_statements :
  ¬ equationA xA yA ∨ ¬ (∀ m, equationB m pt_x pt_y) ∨ (θ = (Real.pi / 2) → ¬ equationC θ xP yP) ∨
  ∀ x₁ y₁ x₂ y₂ x y, equationD x₁ y₁ x₂ y₂ x y :=
by {
  sorry
}

end incorrect_statements_l2008_200846


namespace compare_polynomials_l2008_200894

variable (x : ℝ)
variable (h : x > 1)

theorem compare_polynomials (h : x > 1) : x^3 + 6 * x > x^2 + 6 := 
by
  sorry

end compare_polynomials_l2008_200894


namespace percentage_increase_of_x_l2008_200879

theorem percentage_increase_of_x 
  (x1 y1 : ℝ) 
  (h1 : ∀ x2 y2, (x1 * y1 = x2 * y2) → (y2 = 0.7692307692307693 * y1) → x2 = x1 * 1.3) : 
  ∃ P : ℝ, P = 30 :=
by 
  have P := 30 
  use P 
  sorry

end percentage_increase_of_x_l2008_200879


namespace aquarium_visitors_not_ill_l2008_200895

theorem aquarium_visitors_not_ill :
  let visitors_monday := 300
  let visitors_tuesday := 500
  let visitors_wednesday := 400
  let ill_monday := (15 / 100) * visitors_monday
  let ill_tuesday := (30 / 100) * visitors_tuesday
  let ill_wednesday := (20 / 100) * visitors_wednesday
  let not_ill_monday := visitors_monday - ill_monday
  let not_ill_tuesday := visitors_tuesday - ill_tuesday
  let not_ill_wednesday := visitors_wednesday - ill_wednesday
  let total_not_ill := not_ill_monday + not_ill_tuesday + not_ill_wednesday
  total_not_ill = 925 := 
by
  sorry

end aquarium_visitors_not_ill_l2008_200895


namespace sufficient_but_not_necessary_condition_still_holds_when_not_positive_l2008_200809

theorem sufficient_but_not_necessary_condition (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (a > 0 ∧ b > 0) → (b / a + a / b ≥ 2) :=
by 
  sorry

theorem still_holds_when_not_positive (a b : ℝ) (h1 : a ≤ 0 ∨ b ≤ 0) :
  (b / a + a / b ≥ 2) :=
by
  sorry

end sufficient_but_not_necessary_condition_still_holds_when_not_positive_l2008_200809


namespace find_y_l2008_200802

def operation (x y : ℝ) : ℝ := 5 * x - 4 * y + 3 * x * y

theorem find_y : ∃ y : ℝ, operation 4 y = 21 ∧ y = 1 / 8 := by
  sorry

end find_y_l2008_200802


namespace smallest_delicious_integer_is_minus_2022_l2008_200865

def smallest_delicious_integer (sum_target : ℤ) : ℤ :=
  -2022

theorem smallest_delicious_integer_is_minus_2022
  (B : ℤ)
  (h : ∃ (s : List ℤ), s.sum = 2023 ∧ B ∈ s) :
  B = -2022 :=
sorry

end smallest_delicious_integer_is_minus_2022_l2008_200865


namespace gcf_of_294_and_108_l2008_200818

theorem gcf_of_294_and_108 : Nat.gcd 294 108 = 6 :=
by
  -- We are given numbers 294 and 108
  -- Their prime factorizations are 294 = 2 * 3 * 7^2 and 108 = 2^2 * 3^3
  -- The minimum power of the common prime factors are 2^1 and 3^1
  -- Thus, the GCF by multiplying these factors is 2^1 * 3^1 = 6
  sorry

end gcf_of_294_and_108_l2008_200818


namespace average_bowling_score_l2008_200835

theorem average_bowling_score 
    (gretchen_score : ℕ) (mitzi_score : ℕ) (beth_score : ℕ)
    (gretchen_eq : gretchen_score = 120)
    (mitzi_eq : mitzi_score = 113)
    (beth_eq : beth_score = 85) :
    (gretchen_score + mitzi_score + beth_score) / 3 = 106 := 
by
  sorry

end average_bowling_score_l2008_200835


namespace Kolya_correct_Valya_incorrect_l2008_200878

-- Kolya's Claim (Part a)
theorem Kolya_correct (x : ℝ) (p r : ℝ) (hpr : r = 1/(x+1) ∧ p = 1/x) : 
  (p / (1 - (1 - r) * (1 - p))) = (r / (1 - (1 - r) * (1 - p))) :=
sorry

-- Valya's Claim (Part b)
theorem Valya_incorrect (x : ℝ) (p r : ℝ) (q s : ℝ) (hprs : r = 1/(x+1) ∧ p = 1/x ∧ q = 1 - p ∧ s = 1 - r) : 
  ((q * r / (1 - s * q)) + (p * r / (1 - s * q))) = 1/2 :=
sorry

end Kolya_correct_Valya_incorrect_l2008_200878


namespace gcd_exponentiation_l2008_200877

theorem gcd_exponentiation (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) : 
  let a := 2^m - 2^n
  let b := 2^(m^2 + m * n + n^2) - 1
  let d := Nat.gcd a b
  d = 1 ∨ d = 7 :=
by
  sorry

end gcd_exponentiation_l2008_200877


namespace YaoMing_stride_impossible_l2008_200838

-- Defining the conditions as Lean definitions.
def XiaoMing_14_years_old (current_year : ℕ) : Prop := current_year = 14
def sum_of_triangle_angles (angles : ℕ) : Prop := angles = 180
def CCTV5_broadcasting_basketball_game : Prop := ∃ t : ℕ, true -- Random event placeholder
def YaoMing_stride (stride_length : ℕ) : Prop := stride_length = 10

-- The main statement: Prove that Yao Ming cannot step 10 meters in one stride.
theorem YaoMing_stride_impossible (h1: ∃ y : ℕ, XiaoMing_14_years_old y) 
                                  (h2: ∃ a : ℕ, sum_of_triangle_angles a) 
                                  (h3: CCTV5_broadcasting_basketball_game) 
: ¬ ∃ s : ℕ, YaoMing_stride s := sorry

end YaoMing_stride_impossible_l2008_200838


namespace intersection_A_B_l2008_200869

open Set

def universal_set : Set ℕ := {0, 1, 3, 5, 7, 9}
def complement_A : Set ℕ := {0, 5, 9}
def B : Set ℕ := {3, 5, 7}
def A : Set ℕ := universal_set \ complement_A

theorem intersection_A_B :
  A ∩ B = {3, 7} :=
by
  sorry

end intersection_A_B_l2008_200869


namespace simplify_expression_l2008_200852

theorem simplify_expression : 18 * (8 / 15) * (1 / 12) = 4 / 5 :=
by
  sorry

end simplify_expression_l2008_200852


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l2008_200871

theorem arccos_one_over_sqrt_two_eq_pi_over_four : 
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l2008_200871


namespace proof_1_proof_2_proof_3_proof_4_proof_5_proof_6_l2008_200840

noncomputable def problem_1 : Int :=
13 + (-5) - (-21) - 19

noncomputable def answer_1 : Int := 10

theorem proof_1 : problem_1 = answer_1 := 
by
  sorry

noncomputable def problem_2 : Rat :=
(0.125 : Rat) - (3 + 3 / 4 : Rat) + (-(3 + 1 / 8 : Rat)) - (-(10 + 2 / 3 : Rat)) - (1.25 : Rat)

noncomputable def answer_2 : Rat := 10 + 1 / 6

theorem proof_2 : problem_2 = answer_2 :=
by
  sorry

noncomputable def problem_3 : Rat :=
(36 : Int) / (-8) * (1 / 8 : Rat)

noncomputable def answer_3 : Rat := -9 / 16

theorem proof_3 : problem_3 = answer_3 :=
by
  sorry

noncomputable def problem_4 : Rat :=
((11 / 12 : Rat) - (7 / 6 : Rat) + (3 / 4 : Rat) - (13 / 24 : Rat)) * (-48)

noncomputable def answer_4 : Int := 2

theorem proof_4 : problem_4 = answer_4 :=
by
  sorry

noncomputable def problem_5 : Rat :=
(-(99 + 15 / 16 : Rat)) * 4

noncomputable def answer_5 : Rat := -(399 + 3 / 4 : Rat)

theorem proof_5 : problem_5 = answer_5 :=
by
  sorry

noncomputable def problem_6 : Rat :=
-(1 ^ 4 : Int) - ((1 - 0.5 : Rat) * (1 / 3 : Rat) * (2 - ((-3) ^ 2 : Int) : Int))

noncomputable def answer_6 : Rat := 1 / 6

theorem proof_6 : problem_6 = answer_6 :=
by
  sorry

end proof_1_proof_2_proof_3_proof_4_proof_5_proof_6_l2008_200840


namespace inequality_1_inequality_2_inequality_3_l2008_200858

variable (x : ℝ)

theorem inequality_1 (h : 2 * x^2 - 3 * x + 1 ≥ 0) : x ≤ 1 / 2 ∨ x ≥ 1 := 
  sorry

theorem inequality_2 (h : x^2 - 2 * x - 3 < 0) : -1 < x ∧ x < 3 := 
  sorry

theorem inequality_3 (h : -3 * x^2 + 5 * x - 2 > 0) : 2 / 3 < x ∧ x < 1 := 
  sorry

end inequality_1_inequality_2_inequality_3_l2008_200858


namespace rectangle_area_percentage_increase_l2008_200881

theorem rectangle_area_percentage_increase
  (L W : ℝ) -- Original length and width of the rectangle
  (L_new : L_new = 2 * L) -- New length of the rectangle
  (W_new : W_new = 2 * W) -- New width of the rectangle
  : (4 * L * W - L * W) / (L * W) * 100 = 300 := 
by
  sorry

end rectangle_area_percentage_increase_l2008_200881


namespace seventeen_divides_l2008_200839

theorem seventeen_divides (a b : ℤ) (h : 17 ∣ (2 * a + 3 * b)) : 17 ∣ (9 * a + 5 * b) :=
sorry

end seventeen_divides_l2008_200839


namespace circle_radius_tangent_to_semicircles_and_sides_l2008_200806

noncomputable def side_length_of_square : ℝ := 4
noncomputable def side_length_of_smaller_square : ℝ := side_length_of_square / 2
noncomputable def radius_of_semicircle : ℝ := side_length_of_smaller_square / 2
noncomputable def distance_from_center_to_tangent_point : ℝ := Real.sqrt (side_length_of_smaller_square^2 + radius_of_semicircle^2)

theorem circle_radius_tangent_to_semicircles_and_sides : 
  ∃ (r : ℝ), r = (Real.sqrt 5 - 1) / 2 :=
by
  have r : ℝ := (Real.sqrt 5 - 1) / 2
  use r
  sorry -- Proof omitted

end circle_radius_tangent_to_semicircles_and_sides_l2008_200806


namespace find_x0_l2008_200831

-- Defining the function f
def f (a c x : ℝ) : ℝ := a * x^2 + c

-- Defining the integral condition
def integral_condition (a c x0 : ℝ) : Prop :=
  (∫ x in (0 : ℝ)..(1 : ℝ), f a c x) = f a c x0

-- Proving the main statement
theorem find_x0 (a c x0 : ℝ) (h : a ≠ 0) (h_range : 0 ≤ x0 ∧ x0 ≤ 1) (h_integral : integral_condition a c x0) :
  x0 = Real.sqrt (1 / 3) :=
by
  sorry

end find_x0_l2008_200831


namespace production_today_l2008_200847

-- Definitions based on given conditions
def n := 9
def avg_past_days := 50
def avg_new_days := 55
def total_past_production := n * avg_past_days
def total_new_production := (n + 1) * avg_new_days

-- Theorem: Prove the number of units produced today
theorem production_today : total_new_production - total_past_production = 100 := by
  sorry

end production_today_l2008_200847


namespace block_path_length_l2008_200857

theorem block_path_length
  (length width height : ℝ) 
  (dot_distance : ℝ) 
  (rolls_to_return : ℕ) 
  (π : ℝ) 
  (k : ℝ)
  (H1 : length = 2) 
  (H2 : width = 1) 
  (H3 : height = 1)
  (H4 : dot_distance = 1)
  (H5 : rolls_to_return = 2) 
  (H6 : k = 4) 
  : (2 * rolls_to_return * length * π = k * π) :=
by sorry

end block_path_length_l2008_200857


namespace bus_final_count_l2008_200873

def initial_people : ℕ := 110
def first_stop_off : ℕ := 20
def first_stop_on : ℕ := 15
def second_stop_off : ℕ := 34
def second_stop_on : ℕ := 17
def third_stop_off : ℕ := 18
def third_stop_on : ℕ := 7
def fourth_stop_off : ℕ := 29
def fourth_stop_on : ℕ := 19
def fifth_stop_off : ℕ := 11
def fifth_stop_on : ℕ := 13
def sixth_stop_off : ℕ := 15
def sixth_stop_on : ℕ := 8
def seventh_stop_off : ℕ := 13
def seventh_stop_on : ℕ := 5
def eighth_stop_off : ℕ := 6
def eighth_stop_on : ℕ := 0

theorem bus_final_count :
  initial_people - first_stop_off + first_stop_on 
  - second_stop_off + second_stop_on 
  - third_stop_off + third_stop_on 
  - fourth_stop_off + fourth_stop_on 
  - fifth_stop_off + fifth_stop_on 
  - sixth_stop_off + sixth_stop_on 
  - seventh_stop_off + seventh_stop_on 
  - eighth_stop_off + eighth_stop_on = 48 :=
by sorry

end bus_final_count_l2008_200873


namespace ratio_female_to_male_l2008_200808

variable (m f : ℕ)

-- Average ages given in the conditions
def avg_female_age : ℕ := 35
def avg_male_age : ℕ := 45
def avg_total_age : ℕ := 40

-- Total ages based on number of members
def total_female_age (f : ℕ) : ℕ := avg_female_age * f
def total_male_age (m : ℕ) : ℕ := avg_male_age * m
def total_age (f m : ℕ) : ℕ := total_female_age f + total_male_age m

-- Equation based on average age of all members
def avg_age_eq (f m : ℕ) : Prop :=
  total_age f m / (f + m) = avg_total_age

theorem ratio_female_to_male : avg_age_eq f m → f = m :=
by
  sorry

end ratio_female_to_male_l2008_200808


namespace number_of_ordered_pairs_l2008_200817

noncomputable def count_valid_ordered_pairs (a b: ℝ) : Prop :=
  ∃ (x y : ℤ), a * (x : ℝ) + b * (y : ℝ) = 2 ∧ x^2 + y^2 = 65

theorem number_of_ordered_pairs : ∃ s : Finset (ℝ × ℝ), s.card = 128 ∧ ∀ (p : ℝ × ℝ), p ∈ s ↔ count_valid_ordered_pairs p.1 p.2 :=
by
  sorry

end number_of_ordered_pairs_l2008_200817


namespace share_difference_l2008_200889

theorem share_difference (p q r : ℕ) (x : ℕ) (h_ratio : p = 3 * x ∧ q = 7 * x ∧ r = 12 * x)
  (h_diff_qr : q - r = 5500) : q - p = 4400 :=
by
  sorry

end share_difference_l2008_200889


namespace seymour_fertilizer_requirement_l2008_200829

theorem seymour_fertilizer_requirement :
  let flats_petunias := 4
  let petunias_per_flat := 8
  let flats_roses := 3
  let roses_per_flat := 6
  let venus_flytraps := 2
  let fert_per_petunia := 8
  let fert_per_rose := 3
  let fert_per_venus_flytrap := 2

  let total_petunias := flats_petunias * petunias_per_flat
  let total_roses := flats_roses * roses_per_flat
  let fert_petunias := total_petunias * fert_per_petunia
  let fert_roses := total_roses * fert_per_rose
  let fert_venus_flytraps := venus_flytraps * fert_per_venus_flytrap

  let total_fertilizer := fert_petunias + fert_roses + fert_venus_flytraps
  total_fertilizer = 314 := sorry

end seymour_fertilizer_requirement_l2008_200829


namespace total_students_in_Lansing_l2008_200826

theorem total_students_in_Lansing:
  (number_of_schools : Nat) → 
  (students_per_school : Nat) → 
  (total_students : Nat) →
  number_of_schools = 25 → 
  students_per_school = 247 → 
  total_students = number_of_schools * students_per_school → 
  total_students = 6175 :=
by
  intros number_of_schools students_per_school total_students h_schools h_students h_total
  rw [h_schools, h_students] at h_total
  exact h_total

end total_students_in_Lansing_l2008_200826


namespace total_profit_equals_254000_l2008_200816

-- Definitions
def investment_A : ℕ := 8000
def investment_B : ℕ := 4000
def investment_C : ℕ := 6000
def investment_D : ℕ := 10000

def time_A : ℕ := 12
def time_B : ℕ := 8
def time_C : ℕ := 6
def time_D : ℕ := 9

def capital_months (investment : ℕ) (time : ℕ) : ℕ := investment * time

-- Given conditions
def A_capital_months := capital_months investment_A time_A
def B_capital_months := capital_months investment_B time_B
def C_capital_months := capital_months investment_C time_C
def D_capital_months := capital_months investment_D time_D

def total_capital_months : ℕ := A_capital_months + B_capital_months + C_capital_months + D_capital_months

def C_profit : ℕ := 36000

-- Proportion equation
def total_profit (C_capital_months : ℕ) (total_capital_months : ℕ) (C_profit : ℕ) : ℕ :=
  (C_profit * total_capital_months) / C_capital_months

-- Theorem statement
theorem total_profit_equals_254000 : total_profit C_capital_months total_capital_months C_profit = 254000 := by
  sorry

end total_profit_equals_254000_l2008_200816


namespace max_ab_min_expr_l2008_200844

variable {a b : ℝ}

-- Conditions
axiom a_pos : a > 0
axiom b_pos : b > 0
axiom add_eq_2 : a + b = 2

-- Statements to prove
theorem max_ab : (a * b) ≤ 1 := sorry
theorem min_expr : (2 / a + 8 / b) ≥ 9 := sorry

end max_ab_min_expr_l2008_200844


namespace min_tickets_to_ensure_match_l2008_200882

theorem min_tickets_to_ensure_match : 
  ∀ (host_ticket : Fin 50 → Fin 50),
  ∃ (tickets : Fin 26 → Fin 50 → Fin 50),
  ∀ (i : Fin 26), ∃ (k : Fin 50), host_ticket k = tickets i k :=
by sorry

end min_tickets_to_ensure_match_l2008_200882


namespace water_flow_rate_l2008_200850

theorem water_flow_rate
  (depth : ℝ := 4)
  (width : ℝ := 22)
  (flow_rate_kmph : ℝ := 2)
  (flow_rate_mpm : ℝ := (flow_rate_kmph * 1000) / 60)
  (cross_sectional_area : ℝ := depth * width)
  (volume_per_minute : ℝ := cross_sectional_area * flow_rate_mpm) :
  volume_per_minute = 2933.04 :=
  sorry

end water_flow_rate_l2008_200850


namespace solve_log_eq_l2008_200862

theorem solve_log_eq (x : ℝ) (hx : x > 0) 
  (h : 4^(Real.log x / Real.log 9 * 2) + Real.log 3 / (1/2 * Real.log 3) = 
       0.2 * (4^(2 + Real.log x / Real.log 9) - 4^(Real.log x / Real.log 9))) :
  x = 1 ∨ x = 3 :=
by sorry

end solve_log_eq_l2008_200862


namespace obtuse_triangle_acute_angles_l2008_200843

theorem obtuse_triangle_acute_angles (A B C : ℝ) (h : A + B + C = 180)
  (hA : A > 90) : (B < 90) ∧ (C < 90) :=
sorry

end obtuse_triangle_acute_angles_l2008_200843


namespace earliest_year_exceeds_target_l2008_200836

/-- Define the initial deposit and annual interest rate -/
def initial_deposit : ℝ := 100000
def annual_interest_rate : ℝ := 0.10

/-- Define the amount in the account after n years -/
def amount_after_years (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * (1 + r)^n

/-- Define the target amount to exceed -/
def target_amount : ℝ := 150100

/-- Define the year the initial deposit is made -/
def initial_year : ℕ := 2021

/-- Prove that the earliest year the amount exceeds the target is 2026 -/
theorem earliest_year_exceeds_target :
  ∃ n : ℕ, n > 0 ∧ amount_after_years initial_deposit annual_interest_rate n > target_amount ∧ (initial_year + n) = 2026 :=
by
  sorry

end earliest_year_exceeds_target_l2008_200836


namespace total_percentage_of_failed_candidates_is_correct_l2008_200853

def total_candidates : ℕ := 2000
def number_of_girls : ℕ := 900
def number_of_boys : ℕ := total_candidates - number_of_girls
def percentage_boys_passed : ℚ := 38 / 100
def percentage_girls_passed : ℚ := 32 / 100
def number_of_boys_passed : ℚ := percentage_boys_passed * number_of_boys
def number_of_girls_passed : ℚ := percentage_girls_passed * number_of_girls
def total_candidates_passed : ℚ := number_of_boys_passed + number_of_girls_passed
def total_candidates_failed : ℚ := total_candidates - total_candidates_passed
def total_percentage_failed : ℚ := (total_candidates_failed / total_candidates) * 100

theorem total_percentage_of_failed_candidates_is_correct :
  total_percentage_failed = 64.7 := by
  sorry

end total_percentage_of_failed_candidates_is_correct_l2008_200853


namespace significant_digits_of_square_side_l2008_200814

theorem significant_digits_of_square_side (A : ℝ) (s : ℝ) (h : A = 0.6400) (hs : s^2 = A) : 
  s = 0.8000 :=
sorry

end significant_digits_of_square_side_l2008_200814


namespace minimum_time_reach_distance_minimum_l2008_200864

/-- Given a right triangle with legs of length 1 meter, and two bugs starting crawling from the vertices
with speeds 5 cm/s and 10 cm/s respectively, prove that the minimum time after the start of their movement 
for the distance between the bugs to reach its minimum is 4 seconds. -/
theorem minimum_time_reach_distance_minimum (l : ℝ) (v_A v_B : ℝ) (h_l : l = 1) (h_vA : v_A = 5 / 100) (h_vB : v_B = 10 / 100) :
  ∃ t_min : ℝ, t_min = 4 := by
  -- Proof is omitted
  sorry

end minimum_time_reach_distance_minimum_l2008_200864


namespace simplify_expression_l2008_200898

variable (y : ℝ)

theorem simplify_expression : (3 * y)^3 + (4 * y) * (y^2) - 2 * y^3 = 29 * y^3 :=
by
  sorry

end simplify_expression_l2008_200898


namespace find_digits_sum_l2008_200828

theorem find_digits_sum (A B : ℕ) (h1 : A < 10) (h2 : B < 10) 
  (h3 : (A = 6) ∧ (B = 6))
  (h4 : (100 * A + 44610 + B) % 72 = 0) : A + B = 12 := 
by
  sorry

end find_digits_sum_l2008_200828


namespace total_shelves_needed_l2008_200849

def regular_shelf_capacity : Nat := 45
def large_shelf_capacity : Nat := 30
def regular_books : Nat := 240
def large_books : Nat := 75

def shelves_needed (book_count : Nat) (shelf_capacity : Nat) : Nat :=
  (book_count + shelf_capacity - 1) / shelf_capacity

theorem total_shelves_needed :
  shelves_needed regular_books regular_shelf_capacity +
  shelves_needed large_books large_shelf_capacity = 9 := by
sorry

end total_shelves_needed_l2008_200849


namespace temperature_lower_than_minus_three_l2008_200815

theorem temperature_lower_than_minus_three (a b : ℤ) (hx : a = -3) (hy : b = -6) : a + b = -9 :=
by
  sorry

end temperature_lower_than_minus_three_l2008_200815


namespace power_identity_l2008_200807

theorem power_identity (a b : ℕ) (R S : ℕ) (hR : R = 2^a) (hS : S = 5^b) : 
    20^(a * b) = R^(2 * b) * S^a := 
by 
    -- Insert the proof here
    sorry

end power_identity_l2008_200807


namespace arithmetic_sequence_sum_l2008_200886

theorem arithmetic_sequence_sum (c d : ℤ) (h1 : c = 24) (h2 : d = 31) :
  c + d = 55 :=
by
  rw [h1, h2]
  exact rfl

end arithmetic_sequence_sum_l2008_200886


namespace no_positive_integer_solutions_l2008_200874

theorem no_positive_integer_solutions (x : ℕ) : ¬(15 < 3 - 2 * x) := by
  sorry

end no_positive_integer_solutions_l2008_200874


namespace simplify_expression_l2008_200884

-- Define a variable x
variable (x : ℕ)

-- Statement of the problem
theorem simplify_expression : 120 * x - 75 * x = 45 * x := sorry

end simplify_expression_l2008_200884


namespace cindy_correct_method_l2008_200810

theorem cindy_correct_method (x : ℝ) (h : (x - 7) / 5 = 15) : (x - 5) / 7 = 11 := 
by
  sorry

end cindy_correct_method_l2008_200810
