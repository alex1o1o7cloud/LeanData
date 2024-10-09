import Mathlib

namespace Victor_more_scoops_l632_63239

def ground_almonds : ℝ := 1.56
def white_sugar : ℝ := 0.75

theorem Victor_more_scoops :
  ground_almonds - white_sugar = 0.81 :=
by
  sorry

end Victor_more_scoops_l632_63239


namespace income_to_expenditure_ratio_l632_63268

-- Define the constants based on the conditions in step a)
def income : ℕ := 36000
def savings : ℕ := 4000

-- Define the expenditure as a function of income and savings
def expenditure (I S : ℕ) : ℕ := I - S

-- Define the ratio of two natural numbers
def ratio (a b : ℕ) : ℚ := a / b

-- Statement to be proved
theorem income_to_expenditure_ratio : 
  ratio income (expenditure income savings) = 9 / 8 :=
by
  sorry

end income_to_expenditure_ratio_l632_63268


namespace ellipse_range_k_l632_63299

theorem ellipse_range_k (k : ℝ) (h1 : 3 + k > 0) (h2 : 2 - k > 0) (h3 : k ≠ -1 / 2) :
  k ∈ Set.Ioo (-3 : ℝ) (-1 / 2) ∪ Set.Ioo (-1 / 2) (2 : ℝ) :=
sorry

end ellipse_range_k_l632_63299


namespace division_scaling_l632_63255

theorem division_scaling (h : 204 / 12.75 = 16) : 2.04 / 1.275 = 16 :=
sorry

end division_scaling_l632_63255


namespace correct_calculation_l632_63297

variable (a : ℝ)

theorem correct_calculation : (a^2)^3 = a^6 := 
by sorry

end correct_calculation_l632_63297


namespace power_function_pass_through_point_l632_63206

theorem power_function_pass_through_point (a : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x ^ a) (h_point : f 2 = 16) : a = 4 :=
sorry

end power_function_pass_through_point_l632_63206


namespace chickens_on_farm_are_120_l632_63224

-- Given conditions
def Number_of_hens : ℕ := 52
def Difference_hens_roosters : ℕ := 16

-- Define the number of roosters based on the conditions
def Number_of_roosters : ℕ := Number_of_hens + Difference_hens_roosters

-- The total number of chickens is the sum of hens and roosters
def Total_number_of_chickens : ℕ := Number_of_hens + Number_of_roosters

-- Prove that the total number of chickens is 120
theorem chickens_on_farm_are_120 : Total_number_of_chickens = 120 := by
  -- leave this part unimplemented for proof.
  -- The steps would involve computing the values based on definitions
  sorry

end chickens_on_farm_are_120_l632_63224


namespace tenth_number_in_row_1_sum_of_2023rd_numbers_l632_63212

noncomputable def a (n : ℕ) := (-2)^n
noncomputable def b (n : ℕ) := a n + (n + 1)

theorem tenth_number_in_row_1 : a 10 = (-2)^10 := 
sorry

theorem sum_of_2023rd_numbers : a 2023 + b 2023 = -(2^2024) + 2024 := 
sorry

end tenth_number_in_row_1_sum_of_2023rd_numbers_l632_63212


namespace ratio_of_heights_l632_63269

theorem ratio_of_heights (a b : ℝ) (area_ratio_is_9_4 : a / b = 9 / 4) :
  ∃ h₁ h₂ : ℝ, h₁ / h₂ = 3 / 2 :=
by
  sorry

end ratio_of_heights_l632_63269


namespace find_duration_l632_63201

noncomputable def machine_times (x : ℝ) : Prop :=
  let tP := x + 5
  let tQ := x + 3
  let tR := 2 * (x * (x + 3) / 3)
  (1 / tP + 1 / tQ + 1 / tR = 1 / x) ∧ (tP > 0) ∧ (tQ > 0) ∧ (tR > 0)

theorem find_duration {x : ℝ} (h : machine_times x) : x = 3 :=
sorry

end find_duration_l632_63201


namespace solutions_to_equation_l632_63256

variable (x : ℝ)

def original_eq : Prop :=
  (3 * x - 9) / (x^2 - 6 * x + 8) = (x + 1) / (x - 2)

theorem solutions_to_equation : (original_eq 1 ∧ original_eq 5) :=
by
  sorry

end solutions_to_equation_l632_63256


namespace people_in_room_l632_63204

theorem people_in_room (people chairs : ℕ) (h1 : 5 / 8 * people = 4 / 5 * chairs)
  (h2 : chairs = 5 + 4 / 5 * chairs) : people = 32 :=
by
  sorry

end people_in_room_l632_63204


namespace find_b_in_geometric_sequence_l632_63280

theorem find_b_in_geometric_sequence 
  (a b c : ℝ) 
  (q : ℝ) 
  (h1 : -1 * q^4 = -9) 
  (h2 : a = -1 * q) 
  (h3 : b = a * q) 
  (h4 : c = b * q) 
  (h5 : -9 = c * q) : 
  b = -3 :=
by
  sorry

end find_b_in_geometric_sequence_l632_63280


namespace max_cars_and_quotient_l632_63214

-- Definition of the problem parameters
def car_length : ℕ := 5
def speed_per_car_length : ℕ := 10
def hour_in_seconds : ℕ := 3600
def one_kilometer_in_meters : ℕ := 1000
def distance_in_meters_per_hour (n : ℕ) : ℕ := (10 * n) * one_kilometer_in_meters
def unit_distance (n : ℕ) : ℕ := car_length * (n + 1)

-- Hypotheses
axiom car_spacing : ∀ n : ℕ, unit_distance n = car_length * (n + 1)
axiom car_speed : ∀ n : ℕ, distance_in_meters_per_hour n = (10 * n) * one_kilometer_in_meters

-- Maximum whole number of cars M that can pass in one hour and the quotient when M is divided by 10
theorem max_cars_and_quotient : ∃ (M : ℕ), M = 3000 ∧ M / 10 = 300 := by
  sorry

end max_cars_and_quotient_l632_63214


namespace problem_1_problem_2_l632_63235

def A (x : ℝ) : Prop := x^2 - 3*x - 10 ≤ 0
def B (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2*m - 1

theorem problem_1 (m : ℝ) : (∀ x, B m x → A x)  →  m ≤ 3 := 
sorry

theorem problem_2 (m : ℝ) : (¬ ∃ x, A x ∧ B m x) ↔ (m < 2 ∨ 4 < m) := 
sorry

end problem_1_problem_2_l632_63235


namespace complex_power_equality_l632_63271

namespace ComplexProof

open Complex

noncomputable def cos5 : ℂ := cos (5 * Real.pi / 180)

theorem complex_power_equality (w : ℂ) (h : w + 1 / w = 2 * cos5) : 
  w ^ 1000 + 1 / (w ^ 1000) = -((Real.sqrt 5 + 1) / 2) :=
sorry

end ComplexProof

end complex_power_equality_l632_63271


namespace james_carrot_sticks_l632_63247

theorem james_carrot_sticks (x : ℕ) (h : x + 15 = 37) : x = 22 :=
by {
  sorry
}

end james_carrot_sticks_l632_63247


namespace Total_toys_l632_63205

-- Definitions from the conditions
def Mandy_toys : ℕ := 20
def Anna_toys : ℕ := 3 * Mandy_toys
def Amanda_toys : ℕ := Anna_toys + 2

-- The statement to be proven
theorem Total_toys : Mandy_toys + Anna_toys + Amanda_toys = 142 :=
by
  -- Add proof here
  sorry

end Total_toys_l632_63205


namespace ball_bounces_below_2_feet_l632_63220

theorem ball_bounces_below_2_feet :
  ∃ k : ℕ, 500 * (2 / 3 : ℝ) ^ k < 2 ∧ ∀ n < k, 500 * (2 / 3 : ℝ) ^ n ≥ 2 :=
by
  sorry

end ball_bounces_below_2_feet_l632_63220


namespace atlantic_call_charge_l632_63229

theorem atlantic_call_charge :
  let united_base := 6.00
  let united_per_min := 0.25
  let atlantic_base := 12.00
  let same_bill_minutes := 120
  let atlantic_total (charge_per_minute : ℝ) := atlantic_base + charge_per_minute * same_bill_minutes
  let united_total := united_base + united_per_min * same_bill_minutes
  united_total = atlantic_total 0.20 :=
by
  sorry

end atlantic_call_charge_l632_63229


namespace average_new_data_set_is_5_l632_63208

variable {x1 x2 x3 x4 : ℝ}
variable (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0) (h4 : x4 > 0)
variable (var_sqr : ℝ) (h_var : var_sqr = (1 / 4) * (x1 ^ 2 + x2 ^ 2 + x3 ^ 2 + x4 ^ 2 - 16))

theorem average_new_data_set_is_5 (h_var : var_sqr = (1 / 4) * (x1 ^ 2 + x2 ^ 2 + x3 ^ 2 + x4 ^ 2 - 16)) : 
  (x1 + 3 + x2 + 3 + x3 + 3 + x4 + 3) / 4 = 5 := 
by 
  sorry

end average_new_data_set_is_5_l632_63208


namespace no_six_digit_number_meets_criteria_l632_63219

def valid_digit (n : ℕ) := 2 ≤ n ∧ n ≤ 8

theorem no_six_digit_number_meets_criteria :
  ¬ ∃ (digits : Finset ℕ), digits.card = 6 ∧ (∀ x ∈ digits, valid_digit x) ∧ (digits.sum id = 42) :=
by {
  sorry
}

end no_six_digit_number_meets_criteria_l632_63219


namespace verify_n_l632_63215

noncomputable def find_n (n : ℕ) : Prop :=
  let widget_rate1 := 3                             -- Widgets per worker-hour from the first condition
  let whoosit_rate1 := 2                            -- Whoosits per worker-hour from the first condition
  let widget_rate3 := 1                             -- Widgets per worker-hour from the third condition
  let minutes_per_widget := 1                       -- Arbitrary unit time for one widget
  let minutes_per_whoosit := 2                      -- 2 times unit time for one whoosit based on problem statement
  let whoosit_rate3 := 2 / 3                        -- Whoosits per worker-hour from the third condition
  let widget_rate2 := 540 / (90 * 3 : ℕ)            -- Widgets per hour in the second condition
  let whoosit_rate2 := n / (90 * 3 : ℕ)             -- Whoosits per hour in the second condition
  widget_rate2 = 2 ∧ whoosit_rate2 = 4 / 3 ∧
  (minutes_per_widget < minutes_per_whoosit) ∧
  (whoosit_rate2 = (4 / 3 : ℚ) ↔ n = 360)

theorem verify_n : find_n 360 :=
by sorry

end verify_n_l632_63215


namespace roots_situation_depends_on_k_l632_63202

theorem roots_situation_depends_on_k (k : ℝ) : 
  let a := 1
  let b := -3
  let c := 2 - k
  let Δ := b^2 - 4 * a * c
  (Δ > 0) ∨ (Δ = 0) ∨ (Δ < 0) :=
by
  intros
  sorry

end roots_situation_depends_on_k_l632_63202


namespace sequence_a_n_derived_conditions_derived_sequence_is_even_l632_63284

-- Statement of the first problem
theorem sequence_a_n_derived_conditions (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ)
  (h1 : b 1 = a n)
  (h2 : ∀ k, 2 ≤ k ∧ k ≤ n → b k = a (k - 1) + a k - b (k - 1))
  (h3 : b 1 = 5 ∧ b 2 = -2 ∧ b 3 = 7 ∧ b 4 = 2):
  a 1 = 2 ∧ a 2 = 1 ∧ a 3 = 4 ∧ a 4 = 5 :=
sorry

-- Statement of the second problem
theorem derived_sequence_is_even (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ) (n : ℕ)
  (h_even : n % 2 = 0)
  (h1 : b 1 = a n)
  (h2 : ∀ k, 2 ≤ k ∧ k ≤ n → b k = a (k - 1) + a k - b (k - 1))
  (h3 : c 1 = b n)
  (h4 : ∀ k, 2 ≤ k ∧ k ≤ n → c k = b (k - 1) + b k - c (k - 1)):
  ∀ i, 1 ≤ i ∧ i ≤ n → c i = a i :=
sorry

end sequence_a_n_derived_conditions_derived_sequence_is_even_l632_63284


namespace son_age_next_year_l632_63277

-- Definitions based on the given conditions
def my_current_age : ℕ := 35
def son_current_age : ℕ := my_current_age / 5

-- Theorem statement to prove the answer
theorem son_age_next_year : son_current_age + 1 = 8 :=
by
  -- Skipping the proof with 'sorry'
  sorry

end son_age_next_year_l632_63277


namespace tickets_spent_on_beanie_l632_63290

variable (initial_tickets won_tickets tickets_left tickets_spent: ℕ)

theorem tickets_spent_on_beanie
  (h1 : initial_tickets = 49)
  (h2 : won_tickets = 6)
  (h3 : tickets_left = 30)
  (h4 : tickets_spent = initial_tickets + won_tickets - tickets_left) :
  tickets_spent = 25 :=
by
  sorry

end tickets_spent_on_beanie_l632_63290


namespace shortest_paths_in_grid_l632_63272

-- Define a function that computes the binomial coefficient
def binom (m n : ℕ) : ℕ :=
  Nat.choose (m + n) n

-- Proof problem: Prove that the number of shortest paths in an m x n grid is binom(m, n)
theorem shortest_paths_in_grid (m n : ℕ) : binom m n = Nat.choose (m + n) n :=
by
  -- Intentionally left blank: proof is skipped
  sorry

end shortest_paths_in_grid_l632_63272


namespace find_fibonacci_x_l632_63217

def is_fibonacci (a b c : ℕ) : Prop :=
  c = a + b

theorem find_fibonacci_x (a b x : ℕ)
  (h₁ : a = 8)
  (h₂ : b = 13)
  (h₃ : is_fibonacci a b x) :
  x = 21 :=
by
  sorry

end find_fibonacci_x_l632_63217


namespace find_n_l632_63231

theorem find_n (n : ℕ) (h1 : 0 < n) : 
  ∃ n, n > 0 ∧ (Real.tan (Real.pi / (2 * n)) + Real.sin (Real.pi / (2 * n)) = n / 3) := 
sorry

end find_n_l632_63231


namespace round_to_nearest_whole_l632_63274

theorem round_to_nearest_whole (x : ℝ) (hx : x = 12345.49999) : round x = 12345 := by
  -- Proof omitted.
  sorry

end round_to_nearest_whole_l632_63274


namespace ratio_of_discounted_bricks_l632_63249

theorem ratio_of_discounted_bricks (total_bricks discounted_price full_price total_spending: ℝ) 
  (h1 : total_bricks = 1000) 
  (h2 : discounted_price = 0.25) 
  (h3 : full_price = 0.50) 
  (h4 : total_spending = 375) : 
  ∃ D : ℝ, (D / total_bricks = 1 / 2) ∧ (0.25 * D + 0.50 * (total_bricks - D) = total_spending) := 
  sorry

end ratio_of_discounted_bricks_l632_63249


namespace automotive_test_l632_63242

noncomputable def total_distance (D : ℝ) (t : ℝ) : ℝ := 3 * D

theorem automotive_test (D : ℝ) (h_time : (D / 4 + D / 5 + D / 6 = 37)) : total_distance D 37 = 180 :=
  by
    -- This skips the proof, only the statement is given
    sorry

end automotive_test_l632_63242


namespace exists_polynomial_f_divides_f_x2_sub_1_l632_63286

open Polynomial

theorem exists_polynomial_f_divides_f_x2_sub_1 (n : ℕ) :
    ∃ f : Polynomial ℝ, degree f = n ∧ f ∣ (f.comp (X ^ 2 - 1)) :=
by {
  sorry
}

end exists_polynomial_f_divides_f_x2_sub_1_l632_63286


namespace average_gas_mileage_round_trip_l632_63259

theorem average_gas_mileage_round_trip :
  let distance_to_conference := 150
  let distance_return_trip := 150
  let mpg_sedan := 25
  let mpg_hybrid := 40
  let total_distance := distance_to_conference + distance_return_trip
  let gas_used_sedan := distance_to_conference / mpg_sedan
  let gas_used_hybrid := distance_return_trip / mpg_hybrid
  let total_gas_used := gas_used_sedan + gas_used_hybrid
  let average_gas_mileage := total_distance / total_gas_used
  average_gas_mileage = 31 := by
    sorry

end average_gas_mileage_round_trip_l632_63259


namespace scientific_notation_conversion_l632_63248

theorem scientific_notation_conversion : 450000000 = 4.5 * 10^8 :=
by
  sorry

end scientific_notation_conversion_l632_63248


namespace zacks_friends_l632_63222

theorem zacks_friends (initial_marbles : ℕ) (marbles_kept : ℕ) (marbles_per_friend : ℕ) 
  (h_initial : initial_marbles = 65) (h_kept : marbles_kept = 5) 
  (h_per_friend : marbles_per_friend = 20) : (initial_marbles - marbles_kept) / marbles_per_friend = 3 :=
by
  sorry

end zacks_friends_l632_63222


namespace ratio_expenditure_l632_63294

variable (I : ℝ) -- Assume the income in the first year is I.

-- Conditions
def savings_first_year := 0.25 * I
def expenditure_first_year := 0.75 * I
def income_second_year := 1.25 * I
def savings_second_year := 2 * savings_first_year
def expenditure_second_year := income_second_year - savings_second_year
def total_expenditure_two_years := expenditure_first_year + expenditure_second_year

-- Statement to be proved
theorem ratio_expenditure 
  (savings_first_year : ℝ := 0.25 * I)
  (expenditure_first_year : ℝ := 0.75 * I)
  (income_second_year : ℝ := 1.25 * I)
  (savings_second_year : ℝ := 2 * savings_first_year)
  (expenditure_second_year : ℝ := income_second_year - savings_second_year)
  (total_expenditure_two_years : ℝ := expenditure_first_year + expenditure_second_year) :
  (total_expenditure_two_years / expenditure_first_year) = 2 := by
    sorry

end ratio_expenditure_l632_63294


namespace add_ab_equals_four_l632_63210

theorem add_ab_equals_four (a b : ℝ) (h₁ : a * (a - 4) = 5) (h₂ : b * (b - 4) = 5) (h₃ : a ≠ b) : a + b = 4 :=
by
  sorry

end add_ab_equals_four_l632_63210


namespace tan_alpha_eq_one_l632_63258

theorem tan_alpha_eq_one (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h : Real.cos (α + β) = Real.sin (α - β)) : Real.tan α = 1 :=
sorry

end tan_alpha_eq_one_l632_63258


namespace non_athletic_parents_l632_63253

-- Define the conditions
variables (total_students athletic_dads athletic_moms both_athletic : ℕ)

-- Assume the given conditions
axiom h1 : total_students = 45
axiom h2 : athletic_dads = 17
axiom h3 : athletic_moms = 20
axiom h4 : both_athletic = 11

-- Statement to be proven
theorem non_athletic_parents : total_students - (athletic_dads - both_athletic + athletic_moms - both_athletic + both_athletic) = 19 :=
by {
  -- We intentionally skip the proof here
  sorry
}

end non_athletic_parents_l632_63253


namespace max_trading_cards_l632_63233

theorem max_trading_cards (h : 10 ≥ 1.25 * nat):
  nat ≤ 8 :=
sorry

end max_trading_cards_l632_63233


namespace brooke_social_studies_problems_l632_63241

theorem brooke_social_studies_problems :
  ∀ (math_problems science_problems total_minutes : Nat) 
    (math_time_per_problem science_time_per_problem soc_studies_time_per_problem : Nat)
    (soc_studies_problems : Nat),
  math_problems = 15 →
  science_problems = 10 →
  total_minutes = 48 →
  math_time_per_problem = 2 →
  science_time_per_problem = 3 / 2 → -- converting 1.5 minutes to a fraction
  soc_studies_time_per_problem = 1 / 2 → -- converting 30 seconds to a fraction
  math_problems * math_time_per_problem + science_problems * science_time_per_problem + soc_studies_problems * soc_studies_time_per_problem = 48 →
  soc_studies_problems = 6 :=
by
  intros math_problems science_problems total_minutes math_time_per_problem science_time_per_problem soc_studies_time_per_problem soc_studies_problems
  intros h_math_problems h_science_problems h_total_minutes h_math_time_per_problem h_science_time_per_problem h_soc_studies_time_per_problem h_eq
  sorry

end brooke_social_studies_problems_l632_63241


namespace largest_non_zero_ending_factor_decreasing_number_l632_63293

theorem largest_non_zero_ending_factor_decreasing_number :
  ∃ n: ℕ, n = 180625 ∧ (n % 10 ≠ 0) ∧ (∃ m: ℕ, m < n ∧ (n % m = 0) ∧ (n / 10 ≤ m ∧ m * 10 > 0)) :=
by {
  sorry
}

end largest_non_zero_ending_factor_decreasing_number_l632_63293


namespace polynomial_transformation_l632_63250

theorem polynomial_transformation (g : Polynomial ℝ) (x : ℝ)
  (h : g.eval (x^2 + 2) = x^4 + 6 * x^2 + 8 * x) : 
  g.eval (x^2 - 1) = x^4 - 1 := by
  sorry

end polynomial_transformation_l632_63250


namespace friends_raise_funds_l632_63218

theorem friends_raise_funds (total_amount friends_count min_amount amount_per_person: ℕ)
  (h1 : total_amount = 3000)
  (h2 : friends_count = 10)
  (h3 : min_amount = 300)
  (h4 : amount_per_person = total_amount / friends_count) :
  amount_per_person = min_amount :=
by
  sorry

end friends_raise_funds_l632_63218


namespace range_of_m_minimum_value_ab_l632_63236

-- Define the given condition as a predicate on the real numbers
def domain_condition (m : ℝ) : Prop :=
  ∀ x : ℝ, |x + 2| + |x - 4| - m ≥ 0

-- Define the first part of the proof problem: range of m
theorem range_of_m :
  (∀ m : ℝ, domain_condition m) → ∀ m : ℝ, m ≤ 6 :=
sorry

-- Define the second part of the proof problem: minimum value of 4a + 7b
theorem minimum_value_ab (n : ℝ) (a b : ℝ) (h : n = 6) :
  (∀ a b : ℝ, (a > 0) ∧ (b > 0) ∧ (4 / (a + 5 * b) + 1 / (3 * a + 2 * b) = n)) → 
  ∃ (a b : ℝ), 4 * a + 7 * b = 3 / 2 :=
sorry

end range_of_m_minimum_value_ab_l632_63236


namespace expected_value_8_sided_die_l632_63227

/-- 
The expected value of rolling a standard 8-sided die is 4.5.
The die has 8 sides labeled 1 through 8, and each face has an equal probability of appearing,
which is 1/8. 
-/
theorem expected_value_8_sided_die : 
  (1/8:ℝ) * 1 + (1/8) * 2 + (1/8) * 3 + (1/8) * 4 + (1/8) * 5 + (1/8) * 6 + (1/8) * 7 + (1/8) * 8 = 4.5 :=
by 
  sorry

end expected_value_8_sided_die_l632_63227


namespace infinite_power_tower_solution_l632_63207

theorem infinite_power_tower_solution : 
  ∃ x : ℝ, (∀ y, y = x ^ y → y = 4) → x = Real.sqrt 2 :=
by
  sorry

end infinite_power_tower_solution_l632_63207


namespace no_pairs_for_arithmetic_progression_l632_63288

-- Define the problem in Lean
theorem no_pairs_for_arithmetic_progression :
  ¬ ∃ (a b : ℝ), (2 * a = 5 + b) ∧ (2 * b = a * (1 + b)) :=
sorry

end no_pairs_for_arithmetic_progression_l632_63288


namespace problem_statement_l632_63291

def S : Set Nat := {x | x ∈ Finset.range 13 \ Finset.range 1}

def n : Nat :=
  4^12 - 3 * 3^12 + 3 * 2^12

theorem problem_statement : n % 1000 = 181 :=
by
  sorry

end problem_statement_l632_63291


namespace inappropriate_survey_method_l632_63296

/-
Parameters:
- A: Using a sampling survey method to understand the water-saving awareness of middle school students in the city (appropriate).
- B: Investigating the capital city to understand the environmental pollution situation of the entire province (inappropriate due to lack of representativeness).
- C: Investigating the audience's evaluation of a movie by surveying those seated in odd-numbered seats (appropriate).
- D: Using a census method to understand the compliance rate of pilots' vision (appropriate).
-/

theorem inappropriate_survey_method (A B C D : Prop) 
  (hA : A = true)
  (hB : B = false)  -- This condition defines B as inappropriate
  (hC : C = true)
  (hD : D = true) : B = false :=
sorry

end inappropriate_survey_method_l632_63296


namespace number_of_tables_large_meeting_l632_63257

-- Conditions
def table_length : ℕ := 2
def table_width : ℕ := 1
def side_length_large_meeting : ℕ := 7

-- To be proved: number of tables needed for a large meeting is 12.
theorem number_of_tables_large_meeting : 
  let tables_per_side := side_length_large_meeting / (table_length + table_width)
  ∃ total_tables, total_tables = 4 * tables_per_side ∧ total_tables = 12 :=
by
  sorry

end number_of_tables_large_meeting_l632_63257


namespace trajectory_eq_l632_63225

theorem trajectory_eq {x y m : ℝ} (h : x^2 + y^2 - (4 * m + 2) * x - 2 * m * y + 4 * m^2 + 4 * m + 1 = 0) :
  x - 2 * y - 1 = 0 ∧ x ≠ 1 :=
sorry

end trajectory_eq_l632_63225


namespace c_sum_formula_l632_63226

noncomputable section

def arithmetic_sequence (a : Nat -> ℚ) : Prop :=
  a 3 = 2 ∧ (a 1 + 2 * ((a 2 - a 1) : ℚ)) = 2

def geometric_sequence (b : Nat -> ℚ) (a : Nat -> ℚ) : Prop :=
  b 1 = a 1 ∧ b 4 = a 15

def c_sequence (a : Nat -> ℚ) (b : Nat -> ℚ) (n : Nat) : ℚ :=
  a n + b n

def Tn (c : Nat -> ℚ) (n : Nat) : ℚ :=
  (Finset.range n).sum c

theorem c_sum_formula
  (a b c : Nat -> ℚ)
  (k : Nat) 
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b a)
  (hc : ∀ n, c n = c_sequence a b n) :
  Tn c k = k * (k + 3) / 4 + 2^k - 1 :=
by
  sorry

end c_sum_formula_l632_63226


namespace find_abc_sum_l632_63213

-- Definitions and statements directly taken from conditions
def Q1 (x y : ℝ) : Prop := y = x^2 + 51/50
def Q2 (x y : ℝ) : Prop := x = y^2 + 23/2
def common_tangent_rational_slope (a b c : ℤ) : Prop :=
  ∃ (x y : ℝ), (a * x + b * y = c) ∧ (Q1 x y ∨ Q2 x y)

theorem find_abc_sum :
  ∃ (a b c : ℕ), 
    gcd (a) (gcd (b) (c)) = 1 ∧
    common_tangent_rational_slope (a) (b) (c) ∧
    a + b + c = 9 :=
  by sorry

end find_abc_sum_l632_63213


namespace range_estimate_of_expression_l632_63251

theorem range_estimate_of_expression : 
  6 < (2 * Real.sqrt 2 + Real.sqrt 3) * Real.sqrt 2 ∧ 
       (2 * Real.sqrt 2 + Real.sqrt 3) * Real.sqrt 2 < 7 :=
by
  sorry

end range_estimate_of_expression_l632_63251


namespace bus_driver_hours_worked_l632_63240

-- Definitions based on the problem's conditions.
def regular_rate : ℕ := 20
def regular_hours : ℕ := 40
def overtime_rate : ℕ := regular_rate + (3 * (regular_rate / 4))  -- 75% higher
def total_compensation : ℕ := 1000

-- Theorem statement: The bus driver worked a total of 45 hours last week.
theorem bus_driver_hours_worked : 40 + ((total_compensation - (regular_rate * regular_hours)) / overtime_rate) = 45 := 
by 
  sorry

end bus_driver_hours_worked_l632_63240


namespace john_total_shirts_l632_63263

-- Define initial conditions
def initial_shirts : ℕ := 12
def additional_shirts : ℕ := 4

-- Statement of the problem
theorem john_total_shirts : initial_shirts + additional_shirts = 16 := by
  sorry

end john_total_shirts_l632_63263


namespace increasing_sequence_range_l632_63283

theorem increasing_sequence_range (a : ℝ) (f : ℝ → ℝ) (a_n : ℕ+ → ℝ) :
  (∀ n : ℕ+, a_n n = f n) →
  (∀ n m : ℕ+, n < m → a_n n < a_n m) →
  (∀ x : ℝ, f x = if  x ≤ 7 then (3 - a) * x - 3 else a ^ (x - 6) ) →
  2 < a ∧ a < 3 :=
by
  sorry

end increasing_sequence_range_l632_63283


namespace unique_y_for_diamond_l632_63265

def diamond (x y : ℝ) : ℝ := 5 * x - 4 * y + 2 * x * y + 1

theorem unique_y_for_diamond :
  ∃! y : ℝ, diamond 4 y = 21 :=
by
  sorry

end unique_y_for_diamond_l632_63265


namespace small_bottles_needed_l632_63289

noncomputable def small_bottle_capacity := 40 -- in milliliters
noncomputable def large_bottle_capacity := 540 -- in milliliters
noncomputable def worst_case_small_bottle_capacity := 38 -- in milliliters

theorem small_bottles_needed :
  let n_bottles := Int.ceil (large_bottle_capacity / worst_case_small_bottle_capacity : ℚ)
  n_bottles = 15 :=
by
  sorry

end small_bottles_needed_l632_63289


namespace monica_sees_121_individual_students_l632_63278

def students_count : ℕ :=
  let class1 := 20
  let class2 := 25
  let class3 := 25
  let class4 := class1 / 2
  let class5 := 28
  let class6 := 28
  let total_spots := class1 + class2 + class3 + class4 + class5 + class6
  let overlap12 := 5
  let overlap45 := 3
  let overlap36 := 7
  total_spots - overlap12 - overlap45 - overlap36

theorem monica_sees_121_individual_students : students_count = 121 := by
  sorry

end monica_sees_121_individual_students_l632_63278


namespace inequality_does_not_hold_l632_63282

theorem inequality_does_not_hold (a b : ℝ) (h₁ : a < b) (h₂ : b < 0) :
  ¬ (1 / (a - 1) < 1 / b) :=
by
  sorry

end inequality_does_not_hold_l632_63282


namespace Taehyung_mother_age_l632_63244

theorem Taehyung_mother_age (Taehyung_young_brother_age : ℕ) (Taehyung_age_diff : ℕ) (Mother_age_diff : ℕ) (H1 : Taehyung_young_brother_age = 7) (H2 : Taehyung_age_diff = 5) (H3 : Mother_age_diff = 31) :
  ∃ (Mother_age : ℕ), Mother_age = 43 := 
by
  have Taehyung_age : ℕ := Taehyung_young_brother_age + Taehyung_age_diff
  have Mother_age := Taehyung_age + Mother_age_diff
  existsi (Mother_age)
  sorry

end Taehyung_mother_age_l632_63244


namespace find_p_l632_63298

theorem find_p (a b p : ℝ) (h1: a ≠ 0) (h2: b ≠ 0) 
  (h3: a^2 - 4 * b = 0) 
  (h4: a + b = 5 * p) 
  (h5: a * b = 2 * p^3) : p = 3 := 
sorry

end find_p_l632_63298


namespace altitude_of_dolphin_l632_63281

theorem altitude_of_dolphin (h_submarine : altitude_submarine = -50) (h_dolphin : distance_above_submarine = 10) : altitude_dolphin = -40 :=
by
  -- Altitude of the dolphin is the altitude of the submarine plus the distance above it
  have h_dolphin_altitude : altitude_dolphin = altitude_submarine + distance_above_submarine := sorry
  -- Substitute the values
  rw [h_submarine, h_dolphin] at h_dolphin_altitude
  -- Simplify the expression
  exact h_dolphin_altitude

end altitude_of_dolphin_l632_63281


namespace blueberry_picking_l632_63209

-- Define the amounts y1 and y2 as a function of x
variable (x : ℝ)
def y1 : ℝ := 60 + 18 * x
def y2 : ℝ := 150 + 15 * x

-- State the theorem about the relationships given the condition 
theorem blueberry_picking (hx : x > 10) : 
  y1 x = 60 + 18 * x ∧ y2 x = 150 + 15 * x :=
by
  sorry

end blueberry_picking_l632_63209


namespace negation_of_exists_proposition_l632_63245

theorem negation_of_exists_proposition :
  (¬ ∃ n : ℕ, n^2 > 2^n) → (∀ n : ℕ, n^2 ≤ 2^n) := 
by 
  sorry

end negation_of_exists_proposition_l632_63245


namespace train_crossing_platform_time_l632_63234

theorem train_crossing_platform_time
  (length_train : ℝ)
  (length_platform : ℝ)
  (time_signal_pole : ℝ)
  (speed : ℝ)
  (time_platform_cross : ℝ)
  (v := length_train / time_signal_pole)
  (d := length_train + length_platform)
  (t := d / v) :
  length_train = 300 →
  length_platform = 250 →
  time_signal_pole = 18 →
  time_platform_cross = 33 →
  t = time_platform_cross := by
  sorry

end train_crossing_platform_time_l632_63234


namespace monotonic_decreasing_interval_l632_63246

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - Real.log x

theorem monotonic_decreasing_interval :
  {x : ℝ | 0 < x ∧ x ≤ 1} = {x : ℝ | ∃ ε > 0, ∀ y, y < x → f y > f x ∧ y > 0} :=
sorry

end monotonic_decreasing_interval_l632_63246


namespace soccer_team_total_games_l632_63261

variable (total_games : ℕ)
variable (won_games : ℕ)

-- Given conditions
def team_won_percentage (p : ℝ) := p = 0.60
def team_won_games (w : ℕ) := w = 78

-- The proof goal
theorem soccer_team_total_games 
    (h1 : team_won_percentage 0.60)
    (h2 : team_won_games 78) :
    total_games = 130 :=
sorry

end soccer_team_total_games_l632_63261


namespace smallest_degree_measure_for_WYZ_l632_63292

def angle_XYZ : ℝ := 130
def angle_XYW : ℝ := 100
def angle_WYZ : ℝ := angle_XYZ - angle_XYW

theorem smallest_degree_measure_for_WYZ : angle_WYZ = 30 :=
by
  sorry

end smallest_degree_measure_for_WYZ_l632_63292


namespace find_x_l632_63216

theorem find_x (x : ℝ) (h : x > 0) (area : 1 / 2 * (2 * x) * x = 72) : x = 6 * Real.sqrt 2 :=
by
  sorry

end find_x_l632_63216


namespace num_valid_n_l632_63267

theorem num_valid_n : ∃ k, k = 4 ∧ ∀ n : ℕ, (0 < n ∧ n < 50 ∧ ∃ m : ℕ, m > 0 ∧ n = m * (50 - n)) ↔ 
  (n = 25 ∨ n = 40 ∨ n = 45 ∨ n = 48) :=
by 
  sorry

end num_valid_n_l632_63267


namespace min_marbles_to_draw_l632_63264

theorem min_marbles_to_draw (reds greens blues yellows oranges purples : ℕ)
  (h_reds : reds = 35)
  (h_greens : greens = 25)
  (h_blues : blues = 24)
  (h_yellows : yellows = 18)
  (h_oranges : oranges = 15)
  (h_purples : purples = 12)
  : ∃ n : ℕ, n = 103 ∧ (∀ r g b y o p : ℕ, 
       r ≤ reds ∧ g ≤ greens ∧ b ≤ blues ∧ y ≤ yellows ∧ o ≤ oranges ∧ p ≤ purples ∧ 
       r < 20 ∧ g < 20 ∧ b < 20 ∧ y < 20 ∧ o < 20 ∧ p < 20 → r + g + b + y + o + p < n) ∧
      (∀ r g b y o p : ℕ, 
       r ≤ reds ∧ g ≤ greens ∧ b ≤ blues ∧ y ≤ yellows ∧ o ≤ oranges ∧ p ≤ purples ∧ 
       r + g + b + y + o + p = n → r = 20 ∨ g = 20 ∨ b = 20 ∨ y = 20 ∨ o = 20 ∨ p = 20) :=
sorry

end min_marbles_to_draw_l632_63264


namespace sally_initial_cards_l632_63254

def initial_baseball_cards (t w s a : ℕ) : Prop :=
  a = w + s + t

theorem sally_initial_cards :
  ∃ (initial_cards : ℕ), initial_baseball_cards 9 24 15 initial_cards ∧ initial_cards = 48 :=
by
  use 48
  sorry

end sally_initial_cards_l632_63254


namespace complement_intersection_l632_63223

open Set -- Open the Set namespace to simplify notation for set operations

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def M : Set ℤ := {-1, 0, 1, 3}
def N : Set ℤ := {-2, 0, 2, 3}

theorem complement_intersection : (U \ M) ∩ N = ({-2, 2} : Set ℤ) :=
by
  sorry

end complement_intersection_l632_63223


namespace instantaneous_velocity_at_t_2_l632_63243

theorem instantaneous_velocity_at_t_2 
  (t : ℝ) (x1 y1 x2 y2: ℝ) : 
  (t = 2) → 
  (x1 = 0) → (y1 = 4) → 
  (x2 = 12) → (y2 = -2) → 
  ((y2 - y1) / (x2 - x1) = -1 / 2) := 
by 
  intros ht hx1 hy1 hx2 hy2
  sorry

end instantaneous_velocity_at_t_2_l632_63243


namespace value_of_x_squared_minus_y_squared_l632_63237

theorem value_of_x_squared_minus_y_squared
  (x y : ℚ)
  (h1 : x + y = 8 / 15)
  (h2 : x - y = 2 / 15) :
  x^2 - y^2 = 16 / 225 :=
by
  sorry

end value_of_x_squared_minus_y_squared_l632_63237


namespace compare_abc_l632_63295

noncomputable def a : ℝ := 1 / (1 + Real.exp 2)
noncomputable def b : ℝ := 1 / Real.exp 1
noncomputable def c : ℝ := Real.log ((1 + Real.exp 2) / (Real.exp 2))

theorem compare_abc : b > c ∧ c > a := by
  sorry

end compare_abc_l632_63295


namespace cups_of_flour_already_put_in_correct_l632_63279

-- Let F be the number of cups of flour Mary has already put in
def cups_of_flour_already_put_in (F : ℕ) : Prop :=
  let total_flour_needed := 12
  let cups_of_salt := 7
  let additional_flour_needed := cups_of_salt + 3
  F = total_flour_needed - additional_flour_needed

-- Theorem stating that F = 2
theorem cups_of_flour_already_put_in_correct (F : ℕ) : cups_of_flour_already_put_in F → F = 2 :=
by
  intro h
  sorry

end cups_of_flour_already_put_in_correct_l632_63279


namespace op_assoc_l632_63230

open Real

def op (x y : ℝ) : ℝ := x + y - x * y

theorem op_assoc (x y z : ℝ) : op (op x y) z = op x (op y z) := by
  sorry

end op_assoc_l632_63230


namespace average_of_P_Q_R_is_correct_l632_63260

theorem average_of_P_Q_R_is_correct (P Q R : ℝ) 
  (h1 : 1001 * R - 3003 * P = 6006) 
  (h2 : 2002 * Q + 4004 * P = 8008) : 
  (P + Q + R)/3 = (2 * (P + 5))/3 :=
sorry

end average_of_P_Q_R_is_correct_l632_63260


namespace false_proposition_A_l632_63270

theorem false_proposition_A 
  (a b : ℝ)
  (root1_eq_1 : ∀ x, x^2 + a * x + b = 0 → x = 1)
  (root2_eq_3 : ∀ x, x^2 + a * x + b = 0 → x = 3)
  (sum_of_roots_eq_2 : -a = 2)
  (opposite_sign_roots : ∀ x1 x2, x1 * x2 < 0) :
  ∃ prop, prop = "A" :=
sorry

end false_proposition_A_l632_63270


namespace dhoni_leftover_percentage_l632_63275

variable (E : ℝ) (spent_on_rent : ℝ) (spent_on_dishwasher : ℝ)

def percent_spent_on_rent : ℝ := 0.40
def percent_spent_on_dishwasher : ℝ := 0.32

theorem dhoni_leftover_percentage (E : ℝ) :
  (1 - (percent_spent_on_rent + percent_spent_on_dishwasher)) * E / E = 0.28 :=
by
  sorry

end dhoni_leftover_percentage_l632_63275


namespace calculate_expression_l632_63276

theorem calculate_expression : 
  (10 - 9 * 8 + 7^2 / 2 - 3 * 4 + 6 - 5 = -48.5) :=
by
  -- Proof goes here
  sorry

end calculate_expression_l632_63276


namespace area_quotient_eq_correct_l632_63228

noncomputable def is_in_plane (x y z : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 2

def supports (x y z a b c : ℝ) : Prop :=
  (x ≥ a ∧ y ≥ b ∧ z < c) ∨ (x ≥ a ∧ y < b ∧ z ≥ c) ∨ (x < a ∧ y ≥ b ∧ z ≥ c)

def in_S (x y z : ℝ) : Prop :=
  is_in_plane x y z ∧ supports x y z 1 (2/3) (1/3)

noncomputable def area_S : ℝ := 
  -- Placeholder for the computed area of S
  sorry

noncomputable def area_T : ℝ := 
  -- Placeholder for the computed area of T
  sorry

theorem area_quotient_eq_correct :
  (area_S / area_T) = (3 / (8 * Real.sqrt 3)) := 
  sorry

end area_quotient_eq_correct_l632_63228


namespace exponent_division_l632_63285

theorem exponent_division (a : ℕ) (m n : ℕ) (h1 : 19 = a) (h2 : 11 = m) (h3 : 8 = n) : a^(m - n) = 6859 := by
  sorry

end exponent_division_l632_63285


namespace roots_of_quadratic_sum_of_sixth_powers_l632_63273

theorem roots_of_quadratic_sum_of_sixth_powers {u v : ℝ} 
  (h₀ : u^2 - 2*u*Real.sqrt 3 + 1 = 0)
  (h₁ : v^2 - 2*v*Real.sqrt 3 + 1 = 0)
  : u^6 + v^6 = 970 := 
by 
  sorry

end roots_of_quadratic_sum_of_sixth_powers_l632_63273


namespace radio_range_l632_63238

-- Define constants for speeds and time
def speed_team_1 : ℝ := 20
def speed_team_2 : ℝ := 30
def time : ℝ := 2.5

-- Define the distances each team travels
def distance_team_1 := speed_team_1 * time
def distance_team_2 := speed_team_2 * time

-- Define the total distance which is the range of the radios
def total_distance := distance_team_1 + distance_team_2

-- Prove that the total distance when they lose radio contact is 125 miles
theorem radio_range : total_distance = 125 := by
  sorry

end radio_range_l632_63238


namespace power_of_5_in_8_factorial_l632_63252

theorem power_of_5_in_8_factorial :
  let x := Nat.factorial 8
  ∃ (i k m p : ℕ), 0 < i ∧ 0 < k ∧ 0 < m ∧ 0 < p ∧ x = 2^i * 3^k * 5^m * 7^p ∧ m = 1 :=
by
  sorry

end power_of_5_in_8_factorial_l632_63252


namespace find_natural_numbers_l632_63211

theorem find_natural_numbers (n : ℕ) (p q : ℕ) (hp : p.Prime) (hq : q.Prime)
  (h : q = p + 2) (h1 : (2^n + p).Prime) (h2 : (2^n + q).Prime) :
    n = 1 ∨ n = 3 :=
by
  sorry

end find_natural_numbers_l632_63211


namespace commutative_otimes_l632_63287

def otimes (a b : ℝ) : ℝ := a * b + a + b

theorem commutative_otimes (a b : ℝ) : otimes a b = otimes b a :=
by
  /- The proof will go here, but we omit it and use sorry. -/
  sorry

end commutative_otimes_l632_63287


namespace remaining_inventory_l632_63232

def initial_inventory : Int := 4500
def bottles_sold_mon : Int := 2445
def bottles_sold_tue : Int := 906
def bottles_sold_wed : Int := 215
def bottles_sold_thu : Int := 457
def bottles_sold_fri : Int := 312
def bottles_sold_sat : Int := 239
def bottles_sold_sun : Int := 188

def bottles_received_tue : Int := 350
def bottles_received_thu : Int := 750
def bottles_received_sat : Int := 981

def total_bottles_sold : Int := bottles_sold_mon + bottles_sold_tue + bottles_sold_wed + bottles_sold_thu + bottles_sold_fri + bottles_sold_sat + bottles_sold_sun
def total_bottles_received : Int := bottles_received_tue + bottles_received_thu + bottles_received_sat

theorem remaining_inventory (initial_inventory bottles_sold_mon bottles_sold_tue bottles_sold_wed bottles_sold_thu bottles_sold_fri bottles_sold_sat bottles_sold_sun bottles_received_tue bottles_received_thu bottles_received_sat total_bottles_sold total_bottles_received : Int) :
  initial_inventory - total_bottles_sold + total_bottles_received = 819 :=
by
  sorry

end remaining_inventory_l632_63232


namespace min_value_expression_l632_63203

variable (a b : ℝ)

theorem min_value_expression :
  0 < a →
  1 < b →
  a + b = 2 →
  (∃ x, x = 3 + 2 * Real.sqrt 2 ∧ (∀ y, y = (2 / a) + (1 / (b - 1)) → y ≥ x)) :=
by
  sorry

end min_value_expression_l632_63203


namespace find_range_of_m_l632_63221

-- Define properties of ellipses and hyperbolas
def isEllipseY (m : ℝ) : Prop := (8 - m > 2 * m - 1 ∧ 2 * m - 1 > 0)
def isHyperbola (m : ℝ) : Prop := (m + 1) * (m - 2) < 0

-- The range of 'm' such that (p ∨ q) is true and (p ∧ q) is false
def p_or_q_true_p_and_q_false (m : ℝ) : Prop := 
  (isEllipseY m ∨ isHyperbola m) ∧ ¬ (isEllipseY m ∧ isHyperbola m)

-- The range of the real number 'm'
def range_of_m (m : ℝ) : Prop := 
  (-1 < m ∧ m ≤ 1/2) ∨ (2 ≤ m ∧ m < 3)

-- Prove that the above conditions imply the correct range for m
theorem find_range_of_m (m : ℝ) : p_or_q_true_p_and_q_false m → range_of_m m :=
by
  sorry

end find_range_of_m_l632_63221


namespace candy_difference_l632_63262

theorem candy_difference 
  (total_candies : ℕ)
  (strawberry_candies : ℕ)
  (total_eq : total_candies = 821)
  (strawberry_eq : strawberry_candies = 267) : 
  (total_candies - strawberry_candies - strawberry_candies = 287) :=
by
  sorry

end candy_difference_l632_63262


namespace triangle_third_side_range_l632_63200

theorem triangle_third_side_range {x : ℤ} : 
  (7 < x ∧ x < 17) → (4 ≤ x ∧ x ≤ 16) :=
by
  sorry

end triangle_third_side_range_l632_63200


namespace sequence_general_term_l632_63266

theorem sequence_general_term (a : ℕ → ℤ) (h1 : a 1 = 1) (h_rec : ∀ n, a (n + 1) = 2 * a n + 1) :
  ∀ n, a n = (2 ^ n) - 1 := 
sorry

end sequence_general_term_l632_63266
