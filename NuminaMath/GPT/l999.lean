import Mathlib

namespace sin_double_angle_l999_99907

theorem sin_double_angle (k α : ℝ) (h : Real.cos (π / 4 - α) = k) : Real.sin (2 * α) = 2 * k^2 - 1 := 
by
  sorry

end sin_double_angle_l999_99907


namespace monotonicity_of_f_range_of_a_l999_99919

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x

theorem monotonicity_of_f (a : ℝ) : 
  (a ≤ 0 → ∀ x y : ℝ, x < y → f x a < f y a) ∧ 
  (a > 0 → ∀ x y : ℝ, 
    (x < y ∧ y ≤ Real.log a → f x a > f y a) ∧ 
    (x > Real.log a → f x a < f y a)) :=
by
  sorry

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 0) ↔ 0 ≤ a ∧ a ≤ Real.exp 1 :=
by
  sorry

end monotonicity_of_f_range_of_a_l999_99919


namespace Q_is_234_l999_99957

def P : Set ℕ := {1, 2}
def Q : Set ℕ := {z | ∃ x y : ℕ, x ∈ P ∧ y ∈ P ∧ z = x + y}

theorem Q_is_234 : Q = {2, 3, 4} :=
by
  sorry

end Q_is_234_l999_99957


namespace find_real_solution_to_given_equation_l999_99913

noncomputable def sqrt_96_minus_sqrt_84 : ℝ := Real.sqrt 96 - Real.sqrt 84

theorem find_real_solution_to_given_equation (x : ℝ) (hx : x + 4 ≥ 0) :
  x^2 + 6 * x + 6 * x * Real.sqrt (x + 4) = 60 ↔ x = sqrt_96_minus_sqrt_84 := 
by
  sorry

end find_real_solution_to_given_equation_l999_99913


namespace joe_two_kinds_of_fruit_l999_99982

-- Definitions based on the conditions
def meals := ["breakfast", "lunch", "snack", "dinner"] -- 4 meals
def fruits := ["apple", "orange", "banana"] -- 3 kinds of fruits

-- Probability that Joe consumes the same fruit for all meals
noncomputable def prob_same_fruit := (1 / 3) ^ 4

-- Probability that Joe eats at least two different kinds of fruits
noncomputable def prob_at_least_two_kinds := 1 - 3 * prob_same_fruit

theorem joe_two_kinds_of_fruit :
  prob_at_least_two_kinds = 26 / 27 :=
by
  -- Proof omitted for this theorem
  sorry

end joe_two_kinds_of_fruit_l999_99982


namespace moles_of_Na2SO4_formed_l999_99977

/-- 
Given the following conditions:
1. 1 mole of H2SO4 reacts with 2 moles of NaOH.
2. In the presence of 0.5 moles of HCl and 0.5 moles of KOH.
3. At a temperature of 25°C and a pressure of 1 atm.
Prove that the moles of Na2SO4 formed is 1 mole.
-/

theorem moles_of_Na2SO4_formed
  (H2SO4 : ℝ) -- moles of H2SO4
  (NaOH : ℝ) -- moles of NaOH
  (HCl : ℝ) -- moles of HCl
  (KOH : ℝ) -- moles of KOH
  (T : ℝ) -- temperature in °C
  (P : ℝ) -- pressure in atm
  : H2SO4 = 1 ∧ NaOH = 2 ∧ HCl = 0.5 ∧ KOH = 0.5 ∧ T = 25 ∧ P = 1 → 
  ∃ Na2SO4 : ℝ, Na2SO4 = 1 :=
by
  sorry

end moles_of_Na2SO4_formed_l999_99977


namespace phone_purchase_initial_max_profit_additional_purchase_l999_99963

-- Definitions for phone purchase prices and selling prices
def purchase_price_A : ℕ := 3000
def selling_price_A : ℕ := 3400
def purchase_price_B : ℕ := 3500
def selling_price_B : ℕ := 4000

-- Definitions for total expenditure and profit
def total_spent : ℕ := 32000
def total_profit : ℕ := 4400

-- Definitions for initial number of units purchased
def initial_units_A : ℕ := 6
def initial_units_B : ℕ := 4

-- Definitions for the additional purchase constraints and profit calculation
def max_additional_units : ℕ := 30
def additional_units_A : ℕ := 10
def additional_units_B : ℕ := max_additional_units - additional_units_A 
def max_profit : ℕ := 14000

theorem phone_purchase_initial:
  3000 * initial_units_A + 3500 * initial_units_B = total_spent ∧
  (selling_price_A - purchase_price_A) * initial_units_A + (selling_price_B - purchase_price_B) * initial_units_B = total_profit := by
  sorry 

theorem max_profit_additional_purchase:
  additional_units_A + additional_units_B = max_additional_units ∧
  additional_units_B ≤ 2 * additional_units_A ∧
  (selling_price_A - purchase_price_A) * additional_units_A + (selling_price_B - purchase_price_B) * additional_units_B = max_profit := by
  sorry

end phone_purchase_initial_max_profit_additional_purchase_l999_99963


namespace find_Y_value_l999_99998

-- Define the conditions
def P : ℕ := 4020 / 4
def Q : ℕ := P * 2
def Y : ℤ := P - Q

-- State the theorem
theorem find_Y_value : Y = -1005 := by
  -- Proof goes here
  sorry

end find_Y_value_l999_99998


namespace xiaoming_interview_pass_probability_l999_99955

theorem xiaoming_interview_pass_probability :
  let p_correct := 0.7
  let p_fail_per_attempt := 1 - p_correct
  let p_fail_all_attempts := p_fail_per_attempt ^ 3
  let p_pass_interview := 1 - p_fail_all_attempts
  p_pass_interview = 0.973 := by
    let p_correct := 0.7
    let p_fail_per_attempt := 1 - p_correct
    let p_fail_all_attempts := p_fail_per_attempt ^ 3
    let p_pass_interview := 1 - p_fail_all_attempts
    sorry

end xiaoming_interview_pass_probability_l999_99955


namespace circle_tangent_unique_point_l999_99910

theorem circle_tangent_unique_point (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 1 → (x+4)^2 + (y-a)^2 = 25 → false) →
  (a = 2 * Real.sqrt 5 ∨ a = -2 * Real.sqrt 5 ∨ a = 0) :=
by
  sorry

end circle_tangent_unique_point_l999_99910


namespace cathy_total_money_l999_99943

variable (i d m : ℕ)
variable (h1 : i = 12)
variable (h2 : d = 25)
variable (h3 : m = 2 * d)

theorem cathy_total_money : i + d + m = 87 :=
by
  rw [h1, h2, h3]
  -- Continue proof steps here if necessary
  sorry

end cathy_total_money_l999_99943


namespace mean_home_runs_l999_99926

theorem mean_home_runs :
  let players_with_5 := 3
  let players_with_6 := 4
  let players_with_8 := 2
  let players_with_9 := 1
  let players_with_11 := 1
  let total_home_runs := (5 * players_with_5) + (6 * players_with_6) + (8 * players_with_8) + (9 * players_with_9) + (11 * players_with_11)
  let total_players := players_with_5 + players_with_6 + players_with_8 + players_with_9 + players_with_11
  (total_home_runs / total_players : ℚ) = 75 / 11 :=
by
  sorry

end mean_home_runs_l999_99926


namespace area_of_triangle_XYZ_l999_99906

noncomputable def centroid (p1 p2 p3 : (ℚ × ℚ)) : (ℚ × ℚ) :=
((p1.1 + p2.1 + p3.1) / 3, (p1.2 + p2.2 + p3.2) / 3)

noncomputable def triangle_area (p1 p2 p3 : (ℚ × ℚ)) : ℚ :=
abs ((p1.1 * p2.2 + p2.1 * p3.2 + p3.1 * p1.2 - p1.2 * p2.1 - p2.2 * p3.1 - p3.2 * p1.1) / 2)

noncomputable def point_A : (ℚ × ℚ) := (5, 12)
noncomputable def point_B : (ℚ × ℚ) := (0, 0)
noncomputable def point_C : (ℚ × ℚ) := (14, 0)

noncomputable def point_X : (ℚ × ℚ) :=
(109 / 13, 60 / 13)
noncomputable def point_Y : (ℚ × ℚ) :=
centroid point_A point_B point_X
noncomputable def point_Z : (ℚ × ℚ) :=
centroid point_B point_C point_Y

theorem area_of_triangle_XYZ : triangle_area point_X point_Y point_Z = 84 / 13 :=
sorry

end area_of_triangle_XYZ_l999_99906


namespace problem_one_problem_two_problem_three_l999_99972

noncomputable def f (x : ℝ) : ℝ := 3 - 2 * Real.log x / Real.log 2

noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def h (x : ℝ) : ℝ := (f x + 1) * g x

noncomputable def M (x : ℝ) : ℝ :=
  if f x >= g x then g x else f x

noncomputable def condition_one : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 8 → -6 ≤ h x ∧ h x ≤ 2

noncomputable def condition_two : Prop :=
  ∃ x, (M x = 1 ∧ 0 < x ∧ x ≤ 2) ∧ (∀ y, 0 < y ∧ y < x → M y < 1)

noncomputable def condition_three : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 8 → f (x^2) * f (Real.sqrt x) ≥ g x * -3

theorem problem_one : condition_one := sorry
theorem problem_two : condition_two := sorry
theorem problem_three : condition_three := sorry

end problem_one_problem_two_problem_three_l999_99972


namespace factor_x4_minus_81_l999_99922

variable (x : ℝ)

theorem factor_x4_minus_81 : 
  (x^4 - 81) = (x - 3) * (x + 3) * (x^2 + 9) :=
  by { -- proof steps would go here 
    sorry 
}

end factor_x4_minus_81_l999_99922


namespace candy_distribution_l999_99971

-- Define the required parameters and conditions.
def num_distinct_candies : ℕ := 9
def num_bags : ℕ := 3

-- The result that we need to prove
theorem candy_distribution :
  (3 ^ num_distinct_candies) - 3 * (2 ^ (num_distinct_candies - 1) - 2) = 18921 := by
  sorry

end candy_distribution_l999_99971


namespace third_angle_of_triangle_l999_99960

theorem third_angle_of_triangle (a b : ℝ) (ha : a = 50) (hb : b = 60) : 
  ∃ (c : ℝ), a + b + c = 180 ∧ c = 70 :=
by
  sorry

end third_angle_of_triangle_l999_99960


namespace cost_price_of_watch_l999_99942

theorem cost_price_of_watch (CP SP_loss SP_gain : ℝ) (h1 : SP_loss = 0.79 * CP)
  (h2 : SP_gain = 1.04 * CP) (h3 : SP_gain - SP_loss = 140) : CP = 560 := by
  sorry

end cost_price_of_watch_l999_99942


namespace mustard_bottles_total_l999_99987

theorem mustard_bottles_total (b1 b2 b3 : ℝ) (h1 : b1 = 0.25) (h2 : b2 = 0.25) (h3 : b3 = 0.38) :
  b1 + b2 + b3 = 0.88 :=
by
  sorry

end mustard_bottles_total_l999_99987


namespace min_people_wearing_both_hat_and_glove_l999_99938

theorem min_people_wearing_both_hat_and_glove (n : ℕ) (x : ℕ) 
  (h1 : 2 * n = 5 * (8 : ℕ)) -- 2/5 of n people wear gloves
  (h2 : 3 * n = 4 * (15 : ℕ)) -- 3/4 of n people wear hats
  (h3 : n = 20): -- total number of people is 20
  x = 3 := -- minimum number of people wearing both a hat and a glove is 3
by sorry

end min_people_wearing_both_hat_and_glove_l999_99938


namespace age_difference_problem_l999_99992

theorem age_difference_problem 
    (minimum_age : ℕ := 25)
    (current_age_Jane : ℕ := 28)
    (years_ahead : ℕ := 6)
    (Dara_age_in_6_years : ℕ := (current_age_Jane + years_ahead) / 2):
    minimum_age - (Dara_age_in_6_years - years_ahead) = 14 :=
by
  -- all definition parts: minimum_age, current_age_Jane, years_ahead,
  -- Dara_age_in_6_years are present
  sorry

end age_difference_problem_l999_99992


namespace first_alloy_mass_l999_99916

theorem first_alloy_mass (x : ℝ) : 
  (0.12 * x + 2.8) / (x + 35) = 9.454545454545453 / 100 → 
  x = 20 :=
by
  intro h
  sorry

end first_alloy_mass_l999_99916


namespace quadratic_root_form_eq_l999_99918

theorem quadratic_root_form_eq (c : ℚ) : 
  (∀ x : ℚ, x^2 - 7 * x + c = 0 → x = (7 + Real.sqrt (9 * c)) / 2 ∨ x = (7 - Real.sqrt (9 * c)) / 2) →
  c = 49 / 13 := 
by
  sorry

end quadratic_root_form_eq_l999_99918


namespace odot_property_l999_99999

def odot (x y : ℤ) := 2 * x + y

theorem odot_property (a b : ℤ) (h : odot a (-6 * b) = 4) : odot (a - 5 * b) (a + b) = 6 :=
by
  sorry

end odot_property_l999_99999


namespace find_N_l999_99917

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

theorem find_N (N : ℕ) (hN1 : N < 10000)
  (hN2 : N = 26 * sum_of_digits N) : N = 234 ∨ N = 468 := 
  sorry

end find_N_l999_99917


namespace value_of_x_l999_99973

theorem value_of_x (x : ℝ) : (2 : ℝ) = 1 / (4 * x + 2) → x = -3 / 8 := 
by
  intro h
  sorry

end value_of_x_l999_99973


namespace find_number_l999_99964

theorem find_number (x : ℤ) (h : 5 + x * 5 = 15) : x = 2 :=
by
  sorry

end find_number_l999_99964


namespace flat_rate_first_night_l999_99902

theorem flat_rate_first_night
  (f n : ℚ)
  (h1 : f + 3 * n = 210)
  (h2 : f + 6 * n = 350)
  : f = 70 :=
by
  sorry

end flat_rate_first_night_l999_99902


namespace system_solution_find_a_l999_99900

theorem system_solution (x y : ℝ) (a : ℝ) :
  (|16 + 6 * x - x ^ 2 - y ^ 2| + |6 * x| = 16 + 12 * x - x ^ 2 - y ^ 2)
  ∧ ((a + 15) * y + 15 * x - a = 0) →
  ( (x - 3) ^ 2 + y ^ 2 ≤ 25 ∧ x ≥ 0 ) :=
sorry

theorem find_a (a : ℝ) :
  ∃ (x y : ℝ), 
  ((a + 15) * y + 15 * x - a = 0 ∧ x ≥ 0 ∧ (x - 3) ^ 2 + y ^ 2 ≤ 25) ↔ 
  (a = -20 ∨ a = -12) :=
sorry

end system_solution_find_a_l999_99900


namespace number_of_houses_in_block_l999_99979

theorem number_of_houses_in_block (pieces_per_house pieces_per_block : ℕ) (h1 : pieces_per_house = 32) (h2 : pieces_per_block = 640) :
  pieces_per_block / pieces_per_house = 20 :=
by
  sorry

end number_of_houses_in_block_l999_99979


namespace pressure_on_trapezoidal_dam_l999_99952

noncomputable def water_pressure_on_trapezoidal_dam (ρ g h a b : ℝ) : ℝ :=
  ρ * g * (h^2) * (2 * a + b) / 6

theorem pressure_on_trapezoidal_dam
  (ρ g h a b : ℝ) : water_pressure_on_trapezoidal_dam ρ g h a b = ρ * g * (h^2) * (2 * a + b) / 6 := by
  sorry

end pressure_on_trapezoidal_dam_l999_99952


namespace monotonic_function_a_range_l999_99967

theorem monotonic_function_a_range :
  ∀ (f : ℝ → ℝ) (a : ℝ), 
  (f x = x^2 + (2 * a + 1) * x + 1) →
  (∀ x y, 1 ≤ x → x ≤ 2 → 1 ≤ y → y ≤ 2 → (f x ≤ f y ∨ f x ≥ f y)) ↔ 
  (a ∈ Set.Ici (-3/2) ∪ Set.Iic (-5/2)) := 
sorry

end monotonic_function_a_range_l999_99967


namespace verify_sum_l999_99970

-- Definitions and conditions
def C : ℕ := 1
def D : ℕ := 2
def E : ℕ := 5

-- Base-6 addition representation
def is_valid_base_6_addition (a b c d : ℕ) : Prop :=
  (a + b) % 6 = c ∧ (a + b) / 6 = d

-- Given the addition problem:
def addition_problem : Prop :=
  is_valid_base_6_addition 2 5 C 0 ∧
  is_valid_base_6_addition 4 C E 0 ∧
  is_valid_base_6_addition D 2 4 0

-- Goal to prove
theorem verify_sum : addition_problem → C + D + E = 6 :=
by
  sorry

end verify_sum_l999_99970


namespace units_digit_of_sequence_l999_99929

theorem units_digit_of_sequence : 
  (2 + 2 * 3 + 2 * 3^2 + 2 * 3^3 + 2 * 3^4 + 2 * 3^5 + 2 * 3^6 + 2 * 3^7 + 2 * 3^8 + 2 * 3^9) % 10 = 8 := 
by 
  sorry

end units_digit_of_sequence_l999_99929


namespace digit_number_is_203_l999_99978

theorem digit_number_is_203 {A B C : ℕ} (h1 : A + B + C = 10) (h2 : B = A + C) (h3 : 100 * C + 10 * B + A = 100 * A + 10 * B + C + 99) :
  100 * A + 10 * B + C = 203 :=
by
  sorry

end digit_number_is_203_l999_99978


namespace manufacturing_sector_angle_l999_99947

theorem manufacturing_sector_angle (h1 : 50 ≤ 100) (h2 : 360 = 4 * 90) : 0.50 * 360 = 180 := 
by
  sorry

end manufacturing_sector_angle_l999_99947


namespace four_digit_numbers_property_l999_99985

theorem four_digit_numbers_property : 
  ∃ count : ℕ, count = 3 ∧ 
  (∀ N : ℕ, 
    1000 ≤ N ∧ N < 10000 → -- N is a four-digit number
    (∃ a x : ℕ, 
      a ≥ 1 ∧ a < 10 ∧
      100 ≤ x ∧ x < 1000 ∧
      N = 1000 * a + x ∧
      N = 11 * x) → count = 3) :=
sorry

end four_digit_numbers_property_l999_99985


namespace symm_central_origin_l999_99935

noncomputable def f₁ (x : ℝ) : ℝ := 3^x

noncomputable def f₂ (x : ℝ) : ℝ := -3^(-x)

theorem symm_central_origin :
  ∀ x : ℝ, ∃ x' y y' : ℝ, (f₁ x = y) ∧ (f₂ x' = y') ∧ (x' = -x) ∧ (y' = -y) :=
by
  sorry

end symm_central_origin_l999_99935


namespace three_digit_multiples_of_7_l999_99965

theorem three_digit_multiples_of_7 :
  let a := 7 * Nat.ceil (100 / 7)
  let l := 7 * Nat.floor (999 / 7)
  let d := 7
  let n := (l - a) / d + 1
  n = 128 :=
by
  let a := 7 * Nat.ceil (100 / 7)
  let l := 7 * Nat.floor (999 / 7)
  let d := 7
  let n := (l - a) / d + 1
  have : a = 105 := sorry
  have : l = 994 := sorry
  have : n = (994 - 105) / 7 + 1 := sorry
  have : n = 128 := sorry
  exact this

end three_digit_multiples_of_7_l999_99965


namespace abs_abc_eq_one_l999_99969

variable (a b c : ℝ)

-- Conditions
axiom distinct_nonzero : (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0)
axiom condition : a^2 + 1/(b^2) = b^2 + 1/(c^2) ∧ b^2 + 1/(c^2) = c^2 + 1/(a^2)

theorem abs_abc_eq_one : |a * b * c| = 1 :=
by
  sorry

end abs_abc_eq_one_l999_99969


namespace avg_expenditure_Feb_to_July_l999_99950

noncomputable def avg_expenditure_Jan_to_Jun : ℝ := 4200
noncomputable def expenditure_January : ℝ := 1200
noncomputable def expenditure_July : ℝ := 1500
noncomputable def total_months_Jan_to_Jun : ℝ := 6
noncomputable def total_months_Feb_to_July : ℝ := 6

theorem avg_expenditure_Feb_to_July :
  (avg_expenditure_Jan_to_Jun * total_months_Jan_to_Jun - expenditure_January + expenditure_July) / total_months_Feb_to_July = 4250 :=
by sorry

end avg_expenditure_Feb_to_July_l999_99950


namespace m_in_A_l999_99909

variable (x : ℝ)
variable (A : Set ℝ := {x | x ≤ 2})
noncomputable def m : ℝ := Real.sqrt 2

theorem m_in_A : m ∈ A :=
sorry

end m_in_A_l999_99909


namespace distance_from_dorm_to_city_l999_99968

theorem distance_from_dorm_to_city (D : ℚ) (h1 : (1/3) * D = (1/3) * D) (h2 : (3/5) * D = (3/5) * D) (h3 : D - ((1 / 3) * D + (3 / 5) * D) = 2) :
  D = 30 := 
by sorry

end distance_from_dorm_to_city_l999_99968


namespace weight_of_original_piece_of_marble_l999_99996

theorem weight_of_original_piece_of_marble (W : ℝ) 
  (h1 : W > 0)
  (h2 : (0.75 * 0.56 * W) = 105) : 
  W = 250 :=
by
  sorry

end weight_of_original_piece_of_marble_l999_99996


namespace exists_m_in_range_l999_99975

theorem exists_m_in_range :
  ∃ m : ℝ, 0 ≤ m ∧ m < 1 ∧ ∀ x : ℕ, (x > m ∧ x < 2) ↔ (x = 1) :=
by
  sorry

end exists_m_in_range_l999_99975


namespace find_t_l999_99953

variable (t : ℝ)

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (1, 0)
def c (t : ℝ) : ℝ × ℝ := (3 + t, 4)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_t (h : dot_product (a) (c t) = dot_product (b) (c t)) : t = 5 := 
by 
  sorry

end find_t_l999_99953


namespace find_d_l999_99936

def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3

theorem find_d (c d : ℝ) (h : ∀ x : ℝ, f (g x c) c = 15 * x + d) : d = 18 :=
by
  sorry

end find_d_l999_99936


namespace sum_digits_3times_l999_99962

-- Define the sum of digits function
noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the 2006-th power of 2
noncomputable def power_2006 := 2 ^ 2006

-- State the theorem
theorem sum_digits_3times (n : ℕ) (h : n = power_2006) : 
  digit_sum (digit_sum (digit_sum n)) = 4 := by
  -- Add the proof steps here
  sorry

end sum_digits_3times_l999_99962


namespace problem_a_l999_99927

theorem problem_a (nums : Fin 101 → ℤ) : ∃ i j : Fin 101, i ≠ j ∧ (nums i - nums j) % 100 = 0 := sorry

end problem_a_l999_99927


namespace find_value_l999_99932

theorem find_value (x y : ℝ) (h1 : 3 * x + y = 5) (h2 : x + 3 * y = 8) : 5 * x^2 + 11 * x * y + 5 * y^2 = 89 :=
by
  sorry

end find_value_l999_99932


namespace root_polynomial_value_l999_99948

theorem root_polynomial_value (m : ℝ) (h : m^2 + 3 * m - 2022 = 0) : m^3 + 4 * m^2 - 2019 * m - 2023 = -1 :=
  sorry

end root_polynomial_value_l999_99948


namespace future_age_relation_l999_99984

-- Conditions
def son_present_age : ℕ := 8
def father_present_age : ℕ := 4 * son_present_age

-- Theorem statement
theorem future_age_relation : ∃ x : ℕ, 32 + x = 3 * (8 + x) ↔ x = 4 :=
by {
  sorry
}

end future_age_relation_l999_99984


namespace worth_of_each_gift_l999_99945

def workers_per_block : Nat := 200
def total_amount_for_gifts : Nat := 6000
def number_of_blocks : Nat := 15

theorem worth_of_each_gift (workers_per_block : Nat) (total_amount_for_gifts : Nat) (number_of_blocks : Nat) : 
  (total_amount_for_gifts / (workers_per_block * number_of_blocks)) = 2 := 
by 
  sorry

end worth_of_each_gift_l999_99945


namespace utility_bill_amount_l999_99923

/-- Mrs. Brown's utility bill amount given her payments in specific denominations. -/
theorem utility_bill_amount : 
  let fifty_bills := 3 * 50
  let ten_bills := 2 * 10
  fifty_bills + ten_bills = 170 := 
by
  rfl

end utility_bill_amount_l999_99923


namespace cost_of_candy_bar_l999_99901

def initial_amount : ℝ := 3.0
def remaining_amount : ℝ := 2.0

theorem cost_of_candy_bar :
  initial_amount - remaining_amount = 1.0 :=
by
  sorry

end cost_of_candy_bar_l999_99901


namespace gcf_75_100_l999_99995

theorem gcf_75_100 : Nat.gcd 75 100 = 25 :=
by
  sorry

end gcf_75_100_l999_99995


namespace yi_jianlian_shots_l999_99989

theorem yi_jianlian_shots (x y : ℕ) 
  (h1 : x + y = 16 - 3) 
  (h2 : 2 * x + y = 28 - 3 * 3) : 
  x = 6 ∧ y = 7 := 
by 
  sorry

end yi_jianlian_shots_l999_99989


namespace pasta_sauce_cost_l999_99939

theorem pasta_sauce_cost :
  let mustard_oil_cost := 2 * 13
  let penne_pasta_cost := 3 * 4
  let total_cost := 50 - 7
  let spent_on_oil_and_pasta := mustard_oil_cost + penne_pasta_cost
  let pasta_sauce_cost := total_cost - spent_on_oil_and_pasta
  pasta_sauce_cost = 5 :=
by
  let mustard_oil_cost := 2 * 13
  let penne_pasta_cost := 3 * 4
  let total_cost := 50 - 7
  let spent_on_oil_and_pasta := mustard_oil_cost + penne_pasta_cost
  let pasta_sauce_cost := total_cost - spent_on_oil_and_pasta
  sorry

end pasta_sauce_cost_l999_99939


namespace snow_at_least_once_prob_l999_99920

-- Define the conditions for the problem
def prob_snow_day1_to_day4 : ℚ := 1 / 2
def prob_no_snow_day1_to_day4 : ℚ := 1 - prob_snow_day1_to_day4

def prob_snow_day5_to_day7 : ℚ := 1 / 3
def prob_no_snow_day5_to_day7 : ℚ := 1 - prob_snow_day5_to_day7

-- Define the probability of no snow during the first week of February
def prob_no_snow_week : ℚ := (prob_no_snow_day1_to_day4 ^ 4) * (prob_no_snow_day5_to_day7 ^ 3)

-- Define the probability that it snows at least once during the first week of February
def prob_snow_at_least_once : ℚ := 1 - prob_no_snow_week

-- The theorem we want to prove
theorem snow_at_least_once_prob : prob_snow_at_least_once = 53 / 54 :=
by
  sorry

end snow_at_least_once_prob_l999_99920


namespace smallest_x_inequality_l999_99994

theorem smallest_x_inequality : ∃ x : ℝ, (x^2 - 8 * x + 15 ≤ 0) ∧ (∀ y : ℝ, (y^2 - 8 * y + 15 ≤ 0) → (3 ≤ y)) ∧ x = 3 := 
sorry

end smallest_x_inequality_l999_99994


namespace berry_ratio_l999_99934

-- Define the conditions
variables (S V R : ℕ) -- Number of berries Stacy, Steve, and Sylar have
axiom h1 : S + V + R = 1100
axiom h2 : S = 800
axiom h3 : V = 2 * R

-- Define the theorem to be proved
theorem berry_ratio (h1 : S + V + R = 1100) (h2 : S = 800) (h3 : V = 2 * R) : S / V = 4 :=
by
  sorry

end berry_ratio_l999_99934


namespace not_odd_iff_exists_ne_l999_99903

open Function

variable {f : ℝ → ℝ}

theorem not_odd_iff_exists_ne : (∃ x : ℝ, f (-x) ≠ -f x) ↔ ¬ (∀ x : ℝ, f (-x) = -f x) :=
by
  sorry

end not_odd_iff_exists_ne_l999_99903


namespace equation_solution_l999_99914

theorem equation_solution (x y : ℕ) :
  (x^2 + 1)^y - (x^2 - 1)^y = 2 * x^y ↔ 
  (x = 1 ∧ y = 1) ∨ (x = 0 ∧ ∃ k : ℕ, y = 2 * k ∧ k > 0) :=
by sorry

end equation_solution_l999_99914


namespace valid_a_value_l999_99974

theorem valid_a_value (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ a = 20 :=
by
  sorry

end valid_a_value_l999_99974


namespace maximum_gel_pens_l999_99915

theorem maximum_gel_pens 
  (x y z : ℕ) 
  (h1 : x + y + z = 20)
  (h2 : 10 * x + 50 * y + 80 * z = 1000)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0) 
  : y ≤ 13 :=
sorry

end maximum_gel_pens_l999_99915


namespace find_roots_l999_99976

theorem find_roots (x : ℝ) (h : 21 / (x^2 - 9) - 3 / (x - 3) = 1) : x = -3 ∨ x = 7 :=
by {
  sorry
}

end find_roots_l999_99976


namespace seven_people_arrangement_l999_99997

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def perm (n k : Nat) : Nat :=
  factorial n / factorial (n - k)

theorem seven_people_arrangement : 
  (perm 5 5) * (perm 6 2) = 3600 := by
sorry

end seven_people_arrangement_l999_99997


namespace arithmetic_sequence_geometric_subsequence_l999_99986

theorem arithmetic_sequence_geometric_subsequence (a : ℕ → ℤ) (a1 a3 a4 : ℤ) (d : ℤ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : d = 2)
  (h3 : a1 = a 1)
  (h4 : a3 = a 3)
  (h5 : a4 = a 4)
  (h6 : a3^2 = a1 * a4) :
  a 6 = 2 := 
by
  sorry

end arithmetic_sequence_geometric_subsequence_l999_99986


namespace no_such_function_exists_l999_99941

theorem no_such_function_exists :
  ¬ ∃ f : ℝ → ℝ, ∀ x : ℝ, f (f x) = x ^ 2 - 1996 :=
by
  sorry

end no_such_function_exists_l999_99941


namespace isosceles_triangle_area_l999_99921

theorem isosceles_triangle_area {a b h : ℝ} (h1 : a = 13) (h2 : b = 13) (h3 : h = 10) :
  ∃ (A : ℝ), A = 60 ∧ A = (1 / 2) * h * 12 :=
by
  sorry

end isosceles_triangle_area_l999_99921


namespace intersection_M_N_l999_99928

open Set Int

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by
  sorry

end intersection_M_N_l999_99928


namespace expand_and_simplify_l999_99944

theorem expand_and_simplify :
  ∀ (x : ℝ), 2 * x * (3 * x ^ 2 - 4 * x + 5) - (x ^ 2 - 3 * x) * (4 * x + 5) = 2 * x ^ 3 - x ^ 2 + 25 * x :=
by
  intro x
  sorry

end expand_and_simplify_l999_99944


namespace quadratic_expression_value_l999_99951

theorem quadratic_expression_value
  (x : ℝ)
  (h : x^2 + x - 2 = 0)
: x^3 + 2*x^2 - x + 2021 = 2023 :=
sorry

end quadratic_expression_value_l999_99951


namespace shift_upwards_l999_99949

theorem shift_upwards (a : ℝ) :
  (∀ x : ℝ, y = -2 * x + a) -> (a = 1) :=
by
  sorry

end shift_upwards_l999_99949


namespace count_points_l999_99940

theorem count_points (a b : ℝ) :
  (abs b = 2) ∧ (abs a = 4) → (∃ (P : ℝ × ℝ), P = (a, b) ∧ (abs b = 2) ∧ (abs a = 4) ∧
    ((a = 4 ∨ a = -4) ∧ (b = 2 ∨ b = -2)) ∧
    (P = (4, 2) ∨ P = (4, -2) ∨ P = (-4, 2) ∨ P = (-4, -2)) ∧
    ∃ n, n = 4) :=
sorry

end count_points_l999_99940


namespace trucks_sold_l999_99956

-- Definitions for conditions
def cars_and_trucks_total (T C : Nat) : Prop :=
  T + C = 69

def cars_more_than_trucks (T C : Nat) : Prop :=
  C = T + 27

-- Theorem statement
theorem trucks_sold (T C : Nat) (h1 : cars_and_trucks_total T C) (h2 : cars_more_than_trucks T C) : T = 21 :=
by
  -- This will be replaced by the proof
  sorry

end trucks_sold_l999_99956


namespace pregnant_fish_in_each_tank_l999_99905

/-- Mark has 3 tanks for pregnant fish. Each tank has a certain number of pregnant fish and each fish
gives birth to 20 young. Mark has 240 young fish at the end. Prove that there are 4 pregnant fish in
each tank. -/
theorem pregnant_fish_in_each_tank (x : ℕ) (h1 : 3 * 20 * x = 240) : x = 4 := by
  sorry

end pregnant_fish_in_each_tank_l999_99905


namespace delegate_arrangement_probability_l999_99931

theorem delegate_arrangement_probability :
  let delegates := 10
  let countries := 3
  let independent_delegate := 1
  let total_seats := 10
  let m := 379
  let n := 420
  delegates = 10 ∧ countries = 3 ∧ independent_delegate = 1 ∧ total_seats = 10 →
  Nat.gcd m n = 1 →
  m + n = 799 :=
by
  sorry

end delegate_arrangement_probability_l999_99931


namespace find_functions_l999_99983

open Function

theorem find_functions (f g : ℚ → ℚ) :
  (∀ x y : ℚ, f (g x - g y) = f (g x) - y) →
  (∀ x y : ℚ, g (f x - f y) = g (f x) - y) →
  ∃ c : ℚ, c ≠ 0 ∧ (∀ x : ℚ, f x = c * x) ∧ (∀ x : ℚ, g x = x / c) :=
by
  sorry

end find_functions_l999_99983


namespace plane_equation_l999_99961

theorem plane_equation (p q r : ℝ × ℝ × ℝ)
  (h₁ : p = (2, -1, 3))
  (h₂ : q = (0, -1, 5))
  (h₃ : r = (-1, -3, 4)) :
  ∃ A B C D : ℤ, A = 1 ∧ B = 2 ∧ C = -1 ∧ D = 3 ∧
               A > 0 ∧ Int.gcd (Int.gcd (Int.gcd A B) C) D = 1 ∧
               ∀ x y z : ℝ, A * x + B * y + C * z + D = 0 ↔
                             (x, y, z) = p ∨ (x, y, z) = q ∨ (x, y, z) = r :=
by
  sorry

end plane_equation_l999_99961


namespace committee_probability_l999_99966

def num_boys : ℕ := 10
def num_girls : ℕ := 15
def num_total : ℕ := 25
def committee_size : ℕ := 5

def num_ways_total : ℕ := Nat.choose num_total committee_size
def num_ways_boys_only : ℕ := Nat.choose num_boys committee_size
def num_ways_girls_only : ℕ := Nat.choose num_girls committee_size

def probability_boys_or_girls_only : ℚ :=
  (num_ways_boys_only + num_ways_girls_only) / num_ways_total

def probability_at_least_one_boy_and_one_girl : ℚ :=
  1 - probability_boys_or_girls_only

theorem committee_probability :
  probability_at_least_one_boy_and_one_girl = 475 / 506 :=
sorry

end committee_probability_l999_99966


namespace rectangle_area_l999_99954

-- Definitions based on the conditions
def radius := 6
def diameter := 2 * radius
def width := diameter
def length := 3 * width

-- Statement of the theorem
theorem rectangle_area : (width * length = 432) := by
  sorry

end rectangle_area_l999_99954


namespace min_value_a_l999_99959

theorem min_value_a (a : ℝ) :
  (∀ x : ℝ, |x + a| - |x + 1| ≤ 2 * a) → a ≥ 1 / 3 :=
by
  sorry

end min_value_a_l999_99959


namespace total_pools_l999_99980

def patsPools (numAStores numPStores poolsA ratio : ℕ) : ℕ :=
  numAStores * poolsA + numPStores * (ratio * poolsA)

theorem total_pools : 
  patsPools 6 4 200 3 = 3600 := 
by 
  sorry

end total_pools_l999_99980


namespace caffeine_over_l999_99958

section caffeine_problem

-- Definitions of the given conditions
def cups_of_coffee : Nat := 3
def cans_of_soda : Nat := 1
def cups_of_tea : Nat := 2

def caffeine_per_cup_coffee : Nat := 80
def caffeine_per_can_soda : Nat := 40
def caffeine_per_cup_tea : Nat := 50

def caffeine_goal : Nat := 200

-- Calculate the total caffeine consumption
def caffeine_from_coffee : Nat := cups_of_coffee * caffeine_per_cup_coffee
def caffeine_from_soda : Nat := cans_of_soda * caffeine_per_can_soda
def caffeine_from_tea : Nat := cups_of_tea * caffeine_per_cup_tea

def total_caffeine : Nat := caffeine_from_coffee + caffeine_from_soda + caffeine_from_tea

-- Calculate the caffeine amount over the goal
def caffeine_over_goal : Nat := total_caffeine - caffeine_goal

-- Theorem statement
theorem caffeine_over {total_caffeine caffeine_goal : Nat} (h : total_caffeine = 380) (g : caffeine_goal = 200) :
  caffeine_over_goal = 180 := by
  -- The proof goes here.
  sorry

end caffeine_problem

end caffeine_over_l999_99958


namespace omino_tilings_2_by_10_l999_99937

def fib : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n+2) => fib n + fib (n+1)

def omino_tilings (n : ℕ) : ℕ :=
  fib (n + 1)

theorem omino_tilings_2_by_10 : omino_tilings 10 = 3025 := by
  sorry

end omino_tilings_2_by_10_l999_99937


namespace problem1_problem2_l999_99990

-- Define that a quadratic is a root-multiplying equation if one root is twice the other
def is_root_multiplying (a b c : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 * x2 ≠ 0 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 ∧ (x2 = 2 * x1 ∨ x1 = 2 * x2)

-- Problem 1: Prove that x^2 - 3x + 2 = 0 is a root-multiplying equation
theorem problem1 : is_root_multiplying 1 (-3) 2 :=
  sorry

-- Problem 2: Given ax^2 + bx - 6 = 0 is a root-multiplying equation with one root being 2, determine a and b
theorem problem2 (a b : ℝ) : is_root_multiplying a b (-6) → (∃ x1 x2 : ℝ, x1 = 2 ∧ x1 ≠ 0 ∧ a * x1^2 + b * x1 - 6 = 0 ∧ a * x2^2 + b * x2 - 6 = 0 ∧ (x2 = 2 * x1 ∨ x1 = 2 * x2)) →
( (a = -3/4 ∧ b = 9/2) ∨ (a = -3 ∧ b = 9)) :=
  sorry

end problem1_problem2_l999_99990


namespace probability_five_chords_form_convex_pentagon_l999_99924

-- Definitions of problem conditions
variable (n : ℕ) (k : ℕ)

-- Eight points on a circle
def points_on_circle : ℕ := 8

-- Number of chords selected
def selected_chords : ℕ := 5

-- Total number of ways to select 5 chords from 28 possible chords
def total_ways : ℕ := Nat.choose 28 5

-- Number of ways to select 5 points from 8, forming a convex pentagon
def favorable_ways : ℕ := Nat.choose 8 5

-- The probability computation
def probability_pentagon (total : ℕ) (favorable : ℕ) : ℚ :=
  favorable / total

theorem probability_five_chords_form_convex_pentagon :
  probability_pentagon total_ways favorable_ways = 1 / 1755 :=
by
  sorry

end probability_five_chords_form_convex_pentagon_l999_99924


namespace average_other_marbles_l999_99911

def total_marbles : ℕ := 10 -- Define a hypothetical total number for computation
def clear_marbles : ℕ := total_marbles * 40 / 100
def black_marbles : ℕ := total_marbles * 20 / 100
def other_marbles : ℕ := total_marbles - clear_marbles - black_marbles
def marbles_taken : ℕ := 5

theorem average_other_marbles :
  marbles_taken * other_marbles / total_marbles = 2 := by
  sorry

end average_other_marbles_l999_99911


namespace interesting_quadruples_count_l999_99993

/-- Definition of interesting ordered quadruples (a, b, c, d) where 1 ≤ a < b < c < d ≤ 15 and a + b > c + d --/
def is_interesting_quadruple (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 15 ∧ a + b > c + d

/-- The number of interesting ordered quadruples (a, b, c, d) is 455 --/
theorem interesting_quadruples_count : 
  (∃ (s : Finset (ℕ × ℕ × ℕ × ℕ)), 
    s.card = 455 ∧ ∀ (a b c d : ℕ), 
    ((a, b, c, d) ∈ s ↔ is_interesting_quadruple a b c d)) :=
sorry

end interesting_quadruples_count_l999_99993


namespace baseball_card_problem_l999_99912

theorem baseball_card_problem:
  let initial_cards := 15
  let maria_takes := (initial_cards + 1) / 2
  let cards_after_maria := initial_cards - maria_takes
  let cards_after_peter := cards_after_maria - 1
  let final_cards := cards_after_peter * 3
  final_cards = 18 :=
by
  sorry

end baseball_card_problem_l999_99912


namespace tom_needs_more_blue_tickets_l999_99981

def yellow_to_red : ℕ := 10
def red_to_blue : ℕ := 10
def yellow_to_blue : ℕ := yellow_to_red * red_to_blue
def required_yellow_tickets : ℕ := 10
def required_blue_tickets : ℕ := required_yellow_tickets * yellow_to_blue

def toms_yellow_tickets : ℕ := 8
def toms_red_tickets : ℕ := 3
def toms_blue_tickets : ℕ := 7
def toms_total_blue_tickets : ℕ := 
  (toms_yellow_tickets * yellow_to_blue) + 
  (toms_red_tickets * red_to_blue) + 
  toms_blue_tickets

def additional_blue_tickets_needed : ℕ :=
  required_blue_tickets - toms_total_blue_tickets

theorem tom_needs_more_blue_tickets : additional_blue_tickets_needed = 163 := 
by sorry

end tom_needs_more_blue_tickets_l999_99981


namespace power_of_m_l999_99904

theorem power_of_m (m : ℕ) (h₁ : ∀ k : ℕ, m^k % 24 = 0) (h₂ : ∀ d : ℕ, d ∣ m → d ≤ 8) : ∃ k : ℕ, m^k = 24 :=
sorry

end power_of_m_l999_99904


namespace negation_of_existence_l999_99946

theorem negation_of_existence :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by sorry

end negation_of_existence_l999_99946


namespace derivatives_at_zero_l999_99908

noncomputable def f : ℝ → ℝ := sorry

axiom diff_f : ∀ n : ℕ, f (1 / (n + 1)) = (n + 1)^2 / ((n + 1)^2 + 1)

theorem derivatives_at_zero :
  f 0 = 1 ∧ 
  deriv f 0 = 0 ∧ 
  deriv (deriv f) 0 = -2 ∧ 
  ∀ k : ℕ, k ≥ 3 → deriv^[k] f 0 = 0 :=
by
  sorry

end derivatives_at_zero_l999_99908


namespace concentration_after_5_days_l999_99933

noncomputable def ozverin_concentration_after_iterations 
    (initial_volume : ℝ) (initial_concentration : ℝ)
    (drunk_volume : ℝ) (iterations : ℕ) : ℝ :=
initial_concentration * (1 - drunk_volume / initial_volume)^iterations

theorem concentration_after_5_days : 
  ozverin_concentration_after_iterations 0.5 0.4 0.05 5 = 0.236 :=
by
  sorry

end concentration_after_5_days_l999_99933


namespace morgan_hula_hooping_time_l999_99988

-- Definitions based on conditions
def nancy_can_hula_hoop : ℕ := 10
def casey_can_hula_hoop : ℕ := nancy_can_hula_hoop - 3
def morgan_can_hula_hoop : ℕ := 3 * casey_can_hula_hoop

-- Theorem statement to show the solution is correct
theorem morgan_hula_hooping_time : morgan_can_hula_hoop = 21 :=
by
  sorry

end morgan_hula_hooping_time_l999_99988


namespace water_for_1200ml_flour_l999_99925

-- Define the condition of how much water is mixed with a specific amount of flour
def water_per_flour (flour water : ℕ) : Prop :=
  water = (flour / 400) * 100

-- Given condition: Maria uses 100 mL of water for every 400 mL of flour
def condition : Prop := water_per_flour 400 100

-- Problem Statement: How many mL of water for 1200 mL of flour?
theorem water_for_1200ml_flour (h : condition) : water_per_flour 1200 300 :=
sorry

end water_for_1200ml_flour_l999_99925


namespace exists_n_sum_three_digit_identical_digit_l999_99991

theorem exists_n_sum_three_digit_identical_digit:
  ∃ (n : ℕ), (∃ (k : ℕ), (k ≥ 1 ∧ k ≤ 9) ∧ (n*(n+1)/2 = 111*k)) ∧ n = 36 :=
by
  -- Placeholder for the proof
  sorry

end exists_n_sum_three_digit_identical_digit_l999_99991


namespace distinct_paths_from_C_to_D_l999_99930

-- Definitions based on conditions
def grid_rows : ℕ := 7
def grid_columns : ℕ := 8
def total_steps : ℕ := grid_rows + grid_columns -- 15 in this case
def steps_right : ℕ := grid_columns -- 8 in this case

-- Theorem statement
theorem distinct_paths_from_C_to_D :
  Nat.choose total_steps steps_right = 6435 :=
by
  -- The proof itself
  sorry

end distinct_paths_from_C_to_D_l999_99930
