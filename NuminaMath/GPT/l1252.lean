import Mathlib

namespace total_vehicles_correct_l1252_125298

def num_trucks : ℕ := 20
def num_tanks (num_trucks : ℕ) : ℕ := 5 * num_trucks
def total_vehicles (num_trucks : ℕ) (num_tanks : ℕ) : ℕ := num_trucks + num_tanks

theorem total_vehicles_correct : total_vehicles num_trucks (num_tanks num_trucks) = 120 := by
  sorry

end total_vehicles_correct_l1252_125298


namespace cars_on_river_road_l1252_125251

theorem cars_on_river_road (B C : ℕ) (h1 : B = C - 60) (h2 : B * 13 = C) : C = 65 :=
sorry

end cars_on_river_road_l1252_125251


namespace evaluate_T_l1252_125225

def T (a b : ℤ) : ℤ := 4 * a - 7 * b

theorem evaluate_T : T 6 3 = 3 := by
  sorry

end evaluate_T_l1252_125225


namespace hitting_probability_l1252_125218

theorem hitting_probability (P_miss : ℝ) (P_6 P_7 P_8 P_9 P_10 : ℝ) :
  P_miss = 0.2 →
  P_6 = 0.1 →
  P_7 = 0.2 →
  P_8 = 0.3 →
  P_9 = 0.15 →
  P_10 = 0.05 →
  1 - P_miss = 0.8 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end hitting_probability_l1252_125218


namespace A_finishes_remaining_work_in_6_days_l1252_125232

-- Definitions for conditions
def A_workdays : ℕ := 18
def B_workdays : ℕ := 15
def B_worked_days : ℕ := 10

-- Proof problem statement
theorem A_finishes_remaining_work_in_6_days (A_workdays B_workdays B_worked_days : ℕ) :
  let rate_A := 1 / A_workdays
  let rate_B := 1 / B_workdays
  let work_done_by_B := B_worked_days * rate_B
  let remaining_work := 1 - work_done_by_B
  let days_A_needs := remaining_work / rate_A
  days_A_needs = 6 :=
by
  sorry

end A_finishes_remaining_work_in_6_days_l1252_125232


namespace value_of_f_at_pi_over_12_l1252_125203

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x - ω * Real.pi)

theorem value_of_f_at_pi_over_12 (ω : ℝ) (hω_pos : ω > 0) 
(h_period : ∀ x, f ω (x + Real.pi) = f ω x) : 
  f ω (Real.pi / 12) = 1 / 2 := 
sorry

end value_of_f_at_pi_over_12_l1252_125203


namespace maximum_area_of_triangle_l1252_125268

theorem maximum_area_of_triangle (A B C : ℝ) (a b c : ℝ) (hC : C = π / 6) (hSum : a + b = 12) :
  ∃ (S : ℝ), S = 9 ∧ ∀ S', S' ≤ S := 
sorry

end maximum_area_of_triangle_l1252_125268


namespace triangle_perimeter_l1252_125278

theorem triangle_perimeter (a : ℕ) (h1 : a < 8) (h2 : a > 4) (h3 : a % 2 = 0) : 2 + 6 + a = 14 :=
  by
  sorry

end triangle_perimeter_l1252_125278


namespace students_in_sample_l1252_125213

theorem students_in_sample (T : ℕ) (S : ℕ) (F : ℕ) (J : ℕ) (se : ℕ)
  (h1 : J = 22 * T / 100)
  (h2 : S = 25 * T / 100)
  (h3 : se = 160)
  (h4 : F = S + 64)
  (h5 : ∀ x, x ∈ ({F, S, J, se} : Finset ℕ) → x ≤ T ∧  x ≥ 0):
  T = 800 :=
by
  have h6 : T = F + S + J + se := sorry
  sorry

end students_in_sample_l1252_125213


namespace total_yearly_car_leasing_cost_l1252_125247

-- Define mileage per day
def mileage_per_day (day : String) : ℕ :=
  if day = "Monday" ∨ day = "Wednesday" ∨ day = "Friday" ∨ day = "Sunday" then 50
  else if day = "Tuesday" ∨ day = "Thursday" then 80
  else if day = "Saturday" then 120
  else 0

-- Define weekly mileage
def weekly_mileage : ℕ := 4 * 50 + 2 * 80 + 120

-- Define cost parameters
def cost_per_mile : ℕ := 1 / 10
def weekly_fee : ℕ := 100
def monthly_toll_parking_fees : ℕ := 50
def discount_every_5th_week : ℕ := 30
def number_of_weeks_in_year : ℕ := 52

-- Define total yearly cost
def total_cost_yearly : ℕ :=
  let total_weekly_cost := (weekly_mileage * cost_per_mile + weekly_fee)
  let total_yearly_cost := total_weekly_cost * number_of_weeks_in_year
  let total_discounts := (number_of_weeks_in_year / 5) * discount_every_5th_week
  let annual_cost_without_tolls := total_yearly_cost - total_discounts
  let total_toll_fees := monthly_toll_parking_fees * 12
  annual_cost_without_tolls + total_toll_fees

-- Define the main theorem
theorem total_yearly_car_leasing_cost : total_cost_yearly = 7996 := 
  by
    -- Proof omitted
    sorry

end total_yearly_car_leasing_cost_l1252_125247


namespace min_value_fraction_sum_l1252_125263

theorem min_value_fraction_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) : 
  ∃ (x y : ℝ), x = 2/5 ∧ y = 3/5 ∧ (∃ (k : ℝ), k = 4/x + 9/y ∧ k = 25) :=
by
  sorry

end min_value_fraction_sum_l1252_125263


namespace arithmetic_sequence_sufficient_but_not_necessary_condition_l1252_125279

-- Definitions
def is_arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def a_1_a_3_equals_2a_2 (a : ℕ → ℤ) :=
  a 1 + a 3 = 2 * a 2

-- Statement of the mathematical problem
theorem arithmetic_sequence_sufficient_but_not_necessary_condition (a : ℕ → ℤ) :
  is_arithmetic_sequence a → a_1_a_3_equals_2a_2 a ∧ (a_1_a_3_equals_2a_2 a → ¬ is_arithmetic_sequence a) :=
by
  sorry

end arithmetic_sequence_sufficient_but_not_necessary_condition_l1252_125279


namespace trig_identity_and_perimeter_l1252_125214

theorem trig_identity_and_perimeter
  (a b c : ℝ) (A B C : ℝ)
  (h1 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A))
  (a_val : a = 5)
  (cos_A : Real.cos A = 25 / 31) :
  (2 * a^2 = b^2 + c^2) ∧ (a + b + c = 14) :=
by
  sorry

end trig_identity_and_perimeter_l1252_125214


namespace negation_of_universal_proposition_l1252_125240

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) :=
by
  sorry

end negation_of_universal_proposition_l1252_125240


namespace arithmetic_sequence_sum_l1252_125281

theorem arithmetic_sequence_sum (c d : ℤ) (h₁ : d = 10 - 3)
    (h₂ : c = 17 + (10 - 3)) (h₃ : d = 24 + (10 - 3)) :
    c + d = 55 :=
sorry

end arithmetic_sequence_sum_l1252_125281


namespace cylinder_volume_calc_l1252_125261

def cylinder_volume (r h : ℝ) (π : ℝ) : ℝ := π * r^2 * h

theorem cylinder_volume_calc :
    cylinder_volume 5 (5 + 3) 3.14 = 628 :=
by
  -- We set r = 5, h = 8 (since h = r + 3), and π = 3.14 to calculate the volume
  sorry

end cylinder_volume_calc_l1252_125261


namespace part_a_part_b_l1252_125211

-- Define the natural numbers m and n
variable (m n : Nat)

-- Condition: m * n is divisible by m + n
def divisible_condition : Prop :=
  ∃ (k : Nat), m * n = k * (m + n)

-- Define prime number
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ d : Nat, d ∣ p → d = 1 ∨ d = p

-- Define n as the product of two distinct primes
def is_product_of_two_distinct_primes (n : Nat) : Prop :=
  ∃ (p₁ p₂ : Nat), is_prime p₁ ∧ is_prime p₂ ∧ p₁ ≠ p₂ ∧ n = p₁ * p₂

-- Problem (a): Prove that m is divisible by n when n is a prime number and m * n is divisible by m + n
theorem part_a (prime_n : is_prime n) (h : divisible_condition m n) : n ∣ m := sorry

-- Problem (b): Prove that m is not necessarily divisible by n when n is a product of two distinct prime numbers
theorem part_b (prod_of_primes_n : is_product_of_two_distinct_primes n) (h : divisible_condition m n) :
  ¬ (n ∣ m) := sorry

end part_a_part_b_l1252_125211


namespace bombardiers_shots_l1252_125224

theorem bombardiers_shots (x y z : ℕ) :
  x + y = z + 26 →
  x + y + 38 = y + z →
  x + z = y + 24 →
  x = 25 ∧ y = 64 ∧ z = 63 := by
  sorry

end bombardiers_shots_l1252_125224


namespace chi_squared_test_expectation_correct_distribution_table_correct_l1252_125230

-- Given data for the contingency table
def male_good := 52
def male_poor := 8
def female_good := 28
def female_poor := 12
def total := 100

-- Define the $\chi^2$ calculation
def chi_squared_value : ℚ :=
  (total * (male_good * female_poor - male_poor * female_good)^2) / 
  ((male_good + male_poor) * (female_good + female_poor) * (male_good + female_good) * (male_poor + female_poor))

-- The $\chi^2$ value to compare against for 99% confidence
def critical_value_99 : ℚ := 6.635

-- Prove that $\chi^2$ value is less than the critical value for 99% confidence
theorem chi_squared_test :
  chi_squared_value < critical_value_99 :=
by
  -- Sorry to skip the proof as instructed
  sorry

-- Probability data and expectations for successful shots
def prob_male_success : ℚ := 2 / 3
def prob_female_success : ℚ := 1 / 2

-- Probabilities of the number of successful shots
def prob_X_0 : ℚ := (1 - prob_male_success) ^ 2 * (1 - prob_female_success)
def prob_X_1 : ℚ := 2 * prob_male_success * (1 - prob_male_success) * (1 - prob_female_success) +
                    (1 - prob_male_success) ^ 2 * prob_female_success
def prob_X_2 : ℚ := prob_male_success ^ 2 * (1 - prob_female_success) +
                    2 * prob_male_success * (1 - prob_male_success) * prob_female_success
def prob_X_3 : ℚ := prob_male_success ^ 2 * prob_female_success

def expectation_X : ℚ :=
  0 * prob_X_0 + 
  1 * prob_X_1 + 
  2 * prob_X_2 + 
  3 * prob_X_3

-- The expected value of X
def expected_value_X : ℚ := 11 / 6

-- Prove the expected value is as calculated
theorem expectation_correct :
  expectation_X = expected_value_X :=
by
  -- Sorry to skip the proof as instructed
  sorry

-- Define the distribution table based on calculated probabilities
def distribution_table : List (ℚ × ℚ) :=
  [(0, prob_X_0), (1, prob_X_1), (2, prob_X_2), (3, prob_X_3)]

-- The correct distribution table
def correct_distribution_table : List (ℚ × ℚ) :=
  [(0, 1 / 18), (1, 5 / 18), (2, 4 / 9), (3, 2 / 9)]

-- Prove the distribution table is as calculated
theorem distribution_table_correct :
  distribution_table = correct_distribution_table :=
by
  -- Sorry to skip the proof as instructed
  sorry

end chi_squared_test_expectation_correct_distribution_table_correct_l1252_125230


namespace cakes_remaining_l1252_125272

theorem cakes_remaining (cakes_made : ℕ) (cakes_sold : ℕ) (h_made : cakes_made = 149) (h_sold : cakes_sold = 10) :
  (cakes_made - cakes_sold) = 139 :=
by
  cases h_made
  cases h_sold
  sorry

end cakes_remaining_l1252_125272


namespace gcd_gx_x_l1252_125290

theorem gcd_gx_x (x : ℤ) (hx : 34560 ∣ x) :
  Int.gcd ((3 * x + 4) * (8 * x + 5) * (15 * x + 11) * (x + 17)) x = 20 := 
by
  sorry

end gcd_gx_x_l1252_125290


namespace problem_solution_l1252_125264

universe u

variable (U : Set Nat) (M : Set Nat)
variable (complement_U_M : Set Nat)

axiom U_def : U = {1, 2, 3, 4, 5}
axiom complement_U_M_def : complement_U_M = {1, 3}
axiom M_def : M = U \ complement_U_M

theorem problem_solution : 2 ∈ M := by
  sorry

end problem_solution_l1252_125264


namespace roots_of_unity_l1252_125285

noncomputable def is_root_of_unity (z : ℂ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ z^n = 1

noncomputable def is_cube_root_of_unity (z : ℂ) : Prop :=
  z^3 = 1

theorem roots_of_unity (x y : ℂ) (hx : is_root_of_unity x) (hy : is_root_of_unity y) (hxy : x ≠ y) :
  is_root_of_unity (x + y) ↔ is_cube_root_of_unity (y / x) :=
sorry

end roots_of_unity_l1252_125285


namespace ratio_yuan_david_l1252_125277

-- Definitions
def yuan_age (david_age : ℕ) : ℕ := david_age + 7
def ratio (a b : ℕ) : ℚ := a / b

-- Conditions
variable (david_age : ℕ) (h_david : david_age = 7)

-- Proof Statement
theorem ratio_yuan_david : ratio (yuan_age david_age) david_age = 2 :=
by
  sorry

end ratio_yuan_david_l1252_125277


namespace complementary_angle_ratio_l1252_125291

theorem complementary_angle_ratio (x : ℝ) (h1 : 4 * x + x = 90) : x = 18 :=
by {
  sorry
}

end complementary_angle_ratio_l1252_125291


namespace b_in_terms_of_a_axis_of_symmetry_and_a_value_range_of_a_l1252_125248

-- Definitions based on problem conditions
def parabola (a b x : ℝ) : ℝ := a * x^2 + b * x
def passes_through_A (a b : ℝ) : Prop := parabola a b 3 = 3
def points_on_parabola (a b x1 x2 : ℝ) : Prop := x1 < x2 ∧ x1 + x2 = 2
def equal_y_values (a b x1 x2 : ℝ) : Prop := parabola a b x1 = parabola a b x2
def less_than_y_values (a b x1 x2 : ℝ) : Prop := parabola a b x1 < parabola a b x2

-- 1) Express b in terms of a
theorem b_in_terms_of_a (a : ℝ) (h : passes_through_A a (1 - 3 * a)) : True := sorry

-- 2) Axis of symmetry and the value of a when y1 = y2
theorem axis_of_symmetry_and_a_value (a : ℝ) (x1 x2 : ℝ) 
    (h1 : points_on_parabola a (1 - 3 * a) x1 x2)
    (h2 : equal_y_values a (1 - 3 * a) x1 x2) 
    : a = 1 ∧ -1 / 2 * (1 - 3 * a) / a = 1 := sorry

-- 3) Range of values for a when y1 < y2
theorem range_of_a (a x1 x2 : ℝ) 
    (h1 : points_on_parabola a (1 - 3 * a) x1 x2)
    (h2 : less_than_y_values a (1 - 3 * a) x1 x2) 
    (h3 : a ≠ 0) : 0 < a ∧ a < 1 := sorry

end b_in_terms_of_a_axis_of_symmetry_and_a_value_range_of_a_l1252_125248


namespace units_digit_2008_pow_2008_l1252_125217

theorem units_digit_2008_pow_2008 : (2008 ^ 2008) % 10 = 6 := 
by
  -- The units digits of powers of 8 repeat in a cycle: 8, 4, 2, 6
  -- 2008 mod 4 = 0 which implies it falls on the 4th position in the pattern cycle
  sorry

end units_digit_2008_pow_2008_l1252_125217


namespace systematic_sampling_method_l1252_125215

-- Defining the conditions of the problem as lean definitions
def sampling_interval_is_fixed (interval : ℕ) : Prop :=
  interval = 10

def production_line_uniformly_flowing : Prop :=
  true  -- Assumption

-- The main theorem formulation
theorem systematic_sampling_method :
  ∀ (interval : ℕ), sampling_interval_is_fixed interval → production_line_uniformly_flowing →
  (interval = 10 → true) :=
by {
  sorry
}

end systematic_sampling_method_l1252_125215


namespace selection_plans_l1252_125257

-- Definitions for the students
inductive Student
| A | B | C | D | E | F

open Student

-- Definitions for the subjects
inductive Subject
| Mathematics | Physics | Chemistry | Biology

open Subject

-- A function to count the number of valid selections such that A and B do not participate in Biology.
def countValidSelections : Nat :=
  let totalWays := Nat.factorial 6 / Nat.factorial 2 / Nat.factorial (6 - 4)
  let forbiddenWays := 2 * (Nat.factorial 5 / Nat.factorial 2 / Nat.factorial (5 - 3))
  totalWays - forbiddenWays

theorem selection_plans :
  countValidSelections = 240 :=
by
  sorry

end selection_plans_l1252_125257


namespace single_elimination_games_l1252_125284

theorem single_elimination_games (n : Nat) (h : n = 256) : n - 1 = 255 := by
  sorry

end single_elimination_games_l1252_125284


namespace extra_bananas_each_child_l1252_125287

theorem extra_bananas_each_child (total_children absent_children planned_bananas_per_child : ℕ) 
    (h1 : total_children = 660) (h2 : absent_children = 330) (h3 : planned_bananas_per_child = 2) : (1320 / (total_children - absent_children)) - planned_bananas_per_child = 2 := by
  sorry

end extra_bananas_each_child_l1252_125287


namespace sine_beta_value_l1252_125219

variable (α β : ℝ)
variable (h1 : 0 < α ∧ α < (π / 2))
variable (h2 : 0 < β ∧ β < (π / 2))
variable (h3 : Real.cos α = 4 / 5)
variable (h4 : Real.cos (α + β) = 3 / 5)

theorem sine_beta_value : Real.sin β = 7 / 25 :=
by
  -- The proof will go here
  sorry

end sine_beta_value_l1252_125219


namespace perimeter_proof_l1252_125255

noncomputable def perimeter (x : ℝ) : ℝ :=
  if x ≥ 0 ∧ x ≤ (Real.sqrt 3) / 3 then 3 * Real.sqrt 6 * x
  else if x > (Real.sqrt 3) / 3 ∧ x ≤ (2 * Real.sqrt 3) / 3 then 3 * Real.sqrt 2
  else if x > (2 * Real.sqrt 3) / 3 ∧ x ≤ Real.sqrt 3 then 3 * Real.sqrt 6 * (Real.sqrt 3 - x)
  else 0

theorem perimeter_proof (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ Real.sqrt 3) :
  perimeter x = 
    if x ≤ (Real.sqrt 3) / 3 then 3 * Real.sqrt 6 * x
    else if x ≤ (2 * Real.sqrt 3) / 3 then 3 * Real.sqrt 2
    else 3 * Real.sqrt 6 * (Real.sqrt 3 - x) :=
by 
  sorry

end perimeter_proof_l1252_125255


namespace pure_imaginary_solution_l1252_125207

theorem pure_imaginary_solution (b : ℝ) (z : ℂ) 
  (H : z = (b + Complex.I) / (2 + Complex.I))
  (H_imaginary : z.im = z ∧ z.re = 0) :
  b = -1 / 2 := 
by 
  sorry

end pure_imaginary_solution_l1252_125207


namespace projectile_reaches_75_feet_l1252_125253

def projectile_height (t : ℝ) : ℝ := -16 * t^2 + 80 * t

theorem projectile_reaches_75_feet :
  ∃ t : ℝ, projectile_height t = 75 ∧ t = 1.25 :=
by
  -- Skipping the proof as instructed
  sorry

end projectile_reaches_75_feet_l1252_125253


namespace polynomial_coeff_sum_l1252_125262

theorem polynomial_coeff_sum :
  let p1 : Polynomial ℝ := Polynomial.C 4 * Polynomial.X ^ 2 - Polynomial.C 6 * Polynomial.X + Polynomial.C 5
  let p2 : Polynomial ℝ := Polynomial.C 8 - Polynomial.C 3 * Polynomial.X
  let product : Polynomial ℝ := p1 * p2
  let a : ℝ := - (product.coeff 3)
  let b : ℝ := (product.coeff 2)
  let c : ℝ := - (product.coeff 1)
  let d : ℝ := (product.coeff 0)
  8 * a + 4 * b + 2 * c + d = 18 := sorry

end polynomial_coeff_sum_l1252_125262


namespace workers_production_l1252_125260

theorem workers_production
    (x y : ℝ)
    (h1 : x + y = 72)
    (h2 : 1.15 * x + 1.25 * y = 86) :
    1.15 * x = 46 ∧ 1.25 * y = 40 :=
by {
  sorry
}

end workers_production_l1252_125260


namespace P_plus_Q_is_26_l1252_125297

theorem P_plus_Q_is_26 (P Q : ℝ) (h : ∀ x : ℝ, x ≠ 3 → (P / (x - 3) + Q * (x + 2) = (-2 * x^2 + 8 * x + 34) / (x - 3))) : 
  P + Q = 26 :=
sorry

end P_plus_Q_is_26_l1252_125297


namespace average_bull_weight_l1252_125235

def ratioA : ℚ := 7 / 28  -- Ratio of cows to total cattle in section A
def ratioB : ℚ := 5 / 20  -- Ratio of cows to total cattle in section B
def ratioC : ℚ := 3 / 12  -- Ratio of cows to total cattle in section C

def total_cattle : ℕ := 1220  -- Total cattle on the farm
def total_bull_weight : ℚ := 200000  -- Total weight of bulls in kg

theorem average_bull_weight :
  ratioA = 7 / 28 ∧
  ratioB = 5 / 20 ∧
  ratioC = 3 / 12 ∧
  total_cattle = 1220 ∧
  total_bull_weight = 200000 →
  ∃ avg_weight : ℚ, avg_weight = 218.579 :=
sorry

end average_bull_weight_l1252_125235


namespace find_distance_between_stripes_l1252_125201

-- Define the problem conditions
def parallel_curbs (a b : ℝ) := ∀ g : ℝ, g * a = b
def crosswalk_conditions (curb_distance curb_length stripe_length : ℝ) := 
  curb_distance = 60 ∧ curb_length = 22 ∧ stripe_length = 65

-- State the theorem
theorem find_distance_between_stripes (curb_distance curb_length stripe_length : ℝ) 
  (h : ℝ) (H : crosswalk_conditions curb_distance curb_length stripe_length) :
  h = 264 / 13 :=
sorry

end find_distance_between_stripes_l1252_125201


namespace find_integer_n_cos_l1252_125212

theorem find_integer_n_cos : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 180 ∧ (Real.cos (n * Real.pi / 180) = Real.cos (1124 * Real.pi / 180)) ∧ n = 44 := by
  sorry

end find_integer_n_cos_l1252_125212


namespace years_ago_twice_age_l1252_125216

variables (H J x : ℕ)

def henry_age : ℕ := 20
def jill_age : ℕ := 13

axiom age_sum : H + J = 33
axiom age_difference : H - x = 2 * (J - x)

theorem years_ago_twice_age (H := henry_age) (J := jill_age) : x = 6 :=
by sorry

end years_ago_twice_age_l1252_125216


namespace min_distinct_values_l1252_125236

theorem min_distinct_values (n : ℕ) (mode_freq : ℕ) (total : ℕ)
  (h1 : total = 3000) (h2 : mode_freq = 15) :
  n = 215 :=
by
  sorry

end min_distinct_values_l1252_125236


namespace at_least_one_irrational_l1252_125254

theorem at_least_one_irrational (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) 
  (h₃ : a^2 * b^2 * (a^2 * b^2 + 4) = 2 * (a^6 + b^6)) : 
  ¬ (∀ a b : ℚ, a ≠ 0 ∧ b ≠ 0 → a^2 * b^2 * (a^2 * b^2 + 4) = 2 * (a^6 + b^6)) :=
by sorry

end at_least_one_irrational_l1252_125254


namespace solve_pond_fish_problem_l1252_125282

def pond_fish_problem 
  (tagged_fish : ℕ)
  (second_catch : ℕ)
  (tagged_in_second_catch : ℕ)
  (total_fish : ℕ) : Prop :=
  (tagged_in_second_catch : ℝ) / second_catch = (tagged_fish : ℝ) / total_fish →
  total_fish = 1750

theorem solve_pond_fish_problem : 
  pond_fish_problem 70 50 2 1750 :=
by
  sorry

end solve_pond_fish_problem_l1252_125282


namespace rate_ratio_l1252_125274

theorem rate_ratio
  (rate_up : ℝ) (time_up : ℝ) (distance_up : ℝ)
  (distance_down : ℝ) (time_down : ℝ) :
  rate_up = 4 → time_up = 2 → distance_up = rate_up * time_up →
  distance_down = 12 → time_down = 2 →
  (distance_down / time_down) / rate_up = 3 / 2 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end rate_ratio_l1252_125274


namespace prince_cd_total_spent_l1252_125210

theorem prince_cd_total_spent (total_cds : ℕ)
    (pct_20 : ℕ) (pct_15 : ℕ) (pct_10 : ℕ)
    (bought_20_pct : ℕ) (bought_15_pct : ℕ)
    (bought_10_pct : ℕ) (bought_6_pct : ℕ)
    (discount_cnt_4 : ℕ) (discount_amount_4 : ℕ)
    (discount_cnt_5 : ℕ) (discount_amount_5 : ℕ)
    (total_cost_no_discount : ℕ) (total_discount : ℕ) (total_spent : ℕ) :
    total_cds = 400 ∧
    pct_20 = 25 ∧ pct_15 = 30 ∧ pct_10 = 20 ∧
    bought_20_pct = 70 ∧ bought_15_pct = 40 ∧
    bought_10_pct = 80 ∧ bought_6_pct = 100 ∧
    discount_cnt_4 = 4 ∧ discount_amount_4 = 5 ∧
    discount_cnt_5 = 5 ∧ discount_amount_5 = 3 ∧
    total_cost_no_discount - total_discount = total_spent ∧
    total_spent = 3119 := by
  sorry

end prince_cd_total_spent_l1252_125210


namespace percentage_of_employees_in_manufacturing_l1252_125238

theorem percentage_of_employees_in_manufacturing (d total_degrees : ℝ) (h1 : d = 144) (h2 : total_degrees = 360) :
    (d / total_degrees) * 100 = 40 :=
by
  sorry

end percentage_of_employees_in_manufacturing_l1252_125238


namespace simplify_expression_value_at_3_value_at_4_l1252_125292

-- Define the original expression
def original_expr (x : ℕ) : ℚ := (1 - 1 / (x - 1)) / ((x^2 - 4) / (x^2 - 2 * x + 1))

-- Property 1: Simplify the expression
theorem simplify_expression (x : ℕ) (h1 : x ≠ 1) (h2 : x ≠ 2) : 
  original_expr x = (x - 1) / (x + 2) :=
sorry

-- Property 2: Evaluate the expression at x = 3
theorem value_at_3 : original_expr 3 = 2 / 5 :=
sorry

-- Property 3: Evaluate the expression at x = 4
theorem value_at_4 : original_expr 4 = 1 / 2 :=
sorry

end simplify_expression_value_at_3_value_at_4_l1252_125292


namespace last_bead_is_black_l1252_125294

-- Definition of the repeating pattern
def pattern := [1, 2, 3, 1, 2]  -- 1: black, 2: white, 3: gray (one full cycle)

-- Given constants
def total_beads : Nat := 91
def pattern_length : Nat := List.length pattern  -- This should be 9

-- Proof statement: The last bead is black
theorem last_bead_is_black : pattern[(total_beads % pattern_length) - 1] = 1 :=
by
  -- The following steps would be the proof which is not required
  sorry

end last_bead_is_black_l1252_125294


namespace sally_balloons_l1252_125266

theorem sally_balloons :
  (initial_orange_balloons : ℕ) → (lost_orange_balloons : ℕ) → 
  (remaining_orange_balloons : ℕ) → (doubled_orange_balloons : ℕ) → 
  initial_orange_balloons = 20 → 
  lost_orange_balloons = 5 →
  remaining_orange_balloons = initial_orange_balloons - lost_orange_balloons →
  doubled_orange_balloons = 2 * remaining_orange_balloons → 
  doubled_orange_balloons = 30 :=
by
  intro initial_orange_balloons lost_orange_balloons 
       remaining_orange_balloons doubled_orange_balloons
  intro h1 h2 h3 h4
  rw [h1, h2] at h3
  rw [h3] at h4
  sorry

end sally_balloons_l1252_125266


namespace kit_costs_more_l1252_125244

-- Defining the individual prices of the filters and the kit price
def price_filter1 := 16.45
def price_filter2 := 14.05
def price_filter3 := 19.50
def kit_price := 87.50

-- Calculating the total price of the filters if bought individually
def total_individual_price := (2 * price_filter1) + (2 * price_filter2) + price_filter3

-- Calculate the amount saved
def amount_saved := total_individual_price - kit_price

-- The theorem to show the amount saved 
theorem kit_costs_more : amount_saved = -7.00 := by
  sorry

end kit_costs_more_l1252_125244


namespace solve_arcsin_eq_l1252_125299

open Real

noncomputable def problem_statement (x : ℝ) : Prop :=
  arcsin (sin x) = (3 * x) / 4

theorem solve_arcsin_eq(x : ℝ) (h : problem_statement x) (h_range: - (2 * π) / 3 ≤ x ∧ x ≤ (2 * π) / 3) : x = 0 :=
sorry

end solve_arcsin_eq_l1252_125299


namespace find_number_l1252_125269

theorem find_number (x : ℝ) (h : 42 - 3 * x = 12) : x = 10 := 
by 
  sorry

end find_number_l1252_125269


namespace zero_point_of_function_l1252_125275

theorem zero_point_of_function : ∃ x : ℝ, 2 * x - 4 = 0 ∧ x = 2 :=
by
  sorry

end zero_point_of_function_l1252_125275


namespace single_line_points_l1252_125220

theorem single_line_points (S : ℝ) (h1 : 6 * S + 4 * (8 * S) = 38000) : S = 1000 :=
by
  sorry

end single_line_points_l1252_125220


namespace brenda_age_correct_l1252_125209

open Nat

noncomputable def brenda_age_proof : Prop :=
  ∃ (A B J : ℚ), 
  (A = 4 * B) ∧ 
  (J = B + 8) ∧ 
  (A = J) ∧ 
  (B = 8 / 3)

theorem brenda_age_correct : brenda_age_proof := 
  sorry

end brenda_age_correct_l1252_125209


namespace train_constant_speed_is_48_l1252_125206

theorem train_constant_speed_is_48 
  (d_12_00 d_12_15 d_12_45 : ℝ)
  (h1 : 72.5 ≤ d_12_00 ∧ d_12_00 < 73.5)
  (h2 : 61.5 ≤ d_12_15 ∧ d_12_15 < 62.5)
  (h3 : 36.5 ≤ d_12_45 ∧ d_12_45 < 37.5)
  (constant_speed : ℝ → ℝ): 
  (constant_speed d_12_15 - constant_speed d_12_00 = 48) ∧
  (constant_speed d_12_45 - constant_speed d_12_15 = 48) :=
by
  sorry

end train_constant_speed_is_48_l1252_125206


namespace proof_problem_l1252_125229

open Set

variable {U : Set ℕ} {A : Set ℕ} {B : Set ℕ}

def problem_statement (U A B : Set ℕ) : Prop :=
  ((U \ A) ∪ B) = {2, 3}

theorem proof_problem :
  problem_statement {0, 1, 2, 3} {0, 1, 2} {2, 3} :=
by
  unfold problem_statement
  simp
  sorry

end proof_problem_l1252_125229


namespace range_of_k_l1252_125250

theorem range_of_k (k : ℝ) 
  (h1 : ∀ x y : ℝ, (x^2 / (k-3) + y^2 / (2-k) = 1) → (k-3 < 0) ∧ (2-k > 0)) : 
  k < 2 := by
  sorry

end range_of_k_l1252_125250


namespace rate_of_interest_l1252_125295

theorem rate_of_interest (P R : ℝ) :
  (2 * P * R) / 100 = 320 ∧
  P * ((1 + R / 100) ^ 2 - 1) = 340 →
  R = 12.5 :=
by
  intro h
  sorry

end rate_of_interest_l1252_125295


namespace number_of_adults_l1252_125283

theorem number_of_adults (total_bill : ℕ) (cost_per_meal : ℕ) (num_children : ℕ) (total_cost_children : ℕ) 
  (remaining_cost_for_adults : ℕ) (num_adults : ℕ) 
  (H1 : total_bill = 56)
  (H2 : cost_per_meal = 8)
  (H3 : num_children = 5)
  (H4 : total_cost_children = num_children * cost_per_meal)
  (H5 : remaining_cost_for_adults = total_bill - total_cost_children)
  (H6 : num_adults = remaining_cost_for_adults / cost_per_meal) :
  num_adults = 2 :=
by
  sorry

end number_of_adults_l1252_125283


namespace find_cos_F1PF2_l1252_125226

noncomputable def cos_angle_P_F1_F2 : ℝ :=
  let F1 := (-(4:ℝ), 0)
  let F2 := ((4:ℝ), 0)
  let a := (5:ℝ)
  let b := (3:ℝ)
  let P : ℝ × ℝ := sorry -- P is a point on the ellipse
  let area_triangle : ℝ := 3 * Real.sqrt 3
  let cos_angle : ℝ := 1 / 2
  cos_angle

def cos_angle_F1PF2_lemma (F1 F2 : ℝ × ℝ) (ellipse_Area : ℝ) (cos_angle : ℝ) : Prop :=
  cos_angle = 1/2

theorem find_cos_F1PF2 (a b : ℝ) (F1 F2 : ℝ × ℝ) (P : ℝ × ℝ) (Area_PF1F2 : ℝ) :
  (F1 = (-(4:ℝ), 0) ∧ F2 = ((4:ℝ), 0)) ∧ (Area_PF1F2 = 3 * Real.sqrt 3) ∧
  (P.1^2 / (a^2) + P.2^2 / (b^2) = 1) → cos_angle_F1PF2_lemma F1 F2 Area_PF1F2 (cos_angle_P_F1_F2)
:=
  sorry

end find_cos_F1PF2_l1252_125226


namespace Clare_has_more_pencils_than_Jeanine_l1252_125200

def Jeanine_initial_pencils : ℕ := 250
def Clare_initial_pencils : ℤ := (-3 : ℤ) * Jeanine_initial_pencils / 5
def Jeanine_pencils_given_Abby : ℕ := (2 : ℕ) * Jeanine_initial_pencils / 7
def Jeanine_pencils_given_Lea : ℕ := (5 : ℕ) * Jeanine_initial_pencils / 11
def Clare_pencils_after_squaring : ℤ := Clare_initial_pencils ^ 2
def Clare_pencils_after_Jeanine_share : ℤ := Clare_pencils_after_squaring + (-1) * Jeanine_initial_pencils / 4

def Jeanine_final_pencils : ℕ := Jeanine_initial_pencils - Jeanine_pencils_given_Abby - Jeanine_pencils_given_Lea

theorem Clare_has_more_pencils_than_Jeanine :
  Clare_pencils_after_Jeanine_share - Jeanine_final_pencils = 22372 :=
sorry

end Clare_has_more_pencils_than_Jeanine_l1252_125200


namespace sin_double_angle_sin_multiple_angle_l1252_125245

-- Prove that |sin(2x)| <= 2|sin(x)| for any value of x
theorem sin_double_angle (x : ℝ) : |Real.sin (2 * x)| ≤ 2 * |Real.sin x| := 
by sorry

-- Prove that |sin(nx)| <= n|sin(x)| for any positive integer n and any value of x
theorem sin_multiple_angle (n : ℕ) (x : ℝ) (h : 0 < n) : |Real.sin (n * x)| ≤ n * |Real.sin x| :=
by sorry

end sin_double_angle_sin_multiple_angle_l1252_125245


namespace line_segment_is_symmetric_l1252_125228

def is_axial_symmetric (shape : Type) : Prop := sorry
def is_central_symmetric (shape : Type) : Prop := sorry

def equilateral_triangle : Type := sorry
def isosceles_triangle : Type := sorry
def parallelogram : Type := sorry
def line_segment : Type := sorry

theorem line_segment_is_symmetric : 
  is_axial_symmetric line_segment ∧ is_central_symmetric line_segment := 
by
  sorry

end line_segment_is_symmetric_l1252_125228


namespace tan_theta_expr_l1252_125286

variables {θ x : ℝ}

-- Let θ be an acute angle and let sin(θ/2) = sqrt((x - 2) / (3x)).
theorem tan_theta_expr (h₀ : 0 < θ) (h₁ : θ < (Real.pi / 2)) (h₂ : Real.sin (θ / 2) = Real.sqrt ((x - 2) / (3 * x))) :
  Real.tan θ = (3 * Real.sqrt (7 * x^2 - 8 * x - 16)) / (x + 4) :=
sorry

end tan_theta_expr_l1252_125286


namespace perimeter_after_adding_tiles_l1252_125270

theorem perimeter_after_adding_tiles (init_perimeter new_tiles : ℕ) (cond1 : init_perimeter = 14) (cond2 : new_tiles = 2) :
  ∃ new_perimeter : ℕ, new_perimeter = 18 :=
by
  sorry

end perimeter_after_adding_tiles_l1252_125270


namespace average_mark_excluded_students_l1252_125202

variables (N A E A_R A_E : ℕ)

theorem average_mark_excluded_students:
    N = 56 → A = 80 → E = 8 → A_R = 90 →
    N * A = E * A_E + (N - E) * A_R →
    A_E = 20 :=
by
  intros hN hA hE hAR hEquation
  rw [hN, hA, hE, hAR] at hEquation
  have h : 4480 = 8 * A_E + 4320 := hEquation
  sorry

end average_mark_excluded_students_l1252_125202


namespace doris_weeks_to_meet_expenses_l1252_125252

def doris_weekly_hours : Nat := 5 * 3 + 5 -- 5 weekdays (3 hours each) + 5 hours on Saturday
def doris_hourly_rate : Nat := 20 -- Doris earns $20 per hour
def doris_weekly_earnings : Nat := doris_weekly_hours * doris_hourly_rate -- The total earnings per week
def doris_monthly_expenses : Nat := 1200 -- Doris's monthly expense

theorem doris_weeks_to_meet_expenses : ∃ w : Nat, doris_weekly_earnings * w ≥ doris_monthly_expenses ∧ w = 3 :=
by
  sorry

end doris_weeks_to_meet_expenses_l1252_125252


namespace find_two_digit_number_l1252_125259

theorem find_two_digit_number (N : ℕ) (a b c : ℕ) 
  (h_end_digits : N % 1000 = c + 10 * b + 100 * a)
  (hN2_end_digits : N^2 % 1000 = c + 10 * b + 100 * a)
  (h_nonzero : a ≠ 0) :
  10 * a + b = 24 := 
by
  sorry

end find_two_digit_number_l1252_125259


namespace parallel_condition_l1252_125239

theorem parallel_condition (a : ℝ) : (a = -2) ↔ (∀ x y : ℝ, ax + 2 * y = 0 → (-a / 2) = 1) :=
by
  sorry

end parallel_condition_l1252_125239


namespace difference_in_girls_and_boys_l1252_125293

-- Given conditions as definitions
def boys : ℕ := 40
def ratio_boys_to_girls (b g : ℕ) : Prop := 5 * g = 13 * b

-- Statement of the problem
theorem difference_in_girls_and_boys (g : ℕ) (h : ratio_boys_to_girls boys g) : g - boys = 64 :=
by
  sorry

end difference_in_girls_and_boys_l1252_125293


namespace distance_calculation_l1252_125258

-- Define the given constants
def time_minutes : ℕ := 30
def average_speed : ℕ := 1
def seconds_per_minute : ℕ := 60

-- Define the total time in seconds
def time_seconds : ℕ := time_minutes * seconds_per_minute

-- The proof goal: that the distance covered is 1800 meters
theorem distance_calculation :
  time_seconds * average_speed = 1800 := by
  -- Calculation steps (using axioms and known values)
  sorry

end distance_calculation_l1252_125258


namespace probability_of_at_least_one_solving_l1252_125231

variable (P1 P2 : ℝ)

theorem probability_of_at_least_one_solving : 
  (1 - (1 - P1) * (1 - P2)) = P1 + P2 - P1 * P2 := 
sorry

end probability_of_at_least_one_solving_l1252_125231


namespace edward_initial_money_l1252_125242

theorem edward_initial_money (cars qty : Nat) (car_cost race_track_cost left_money initial_money : ℝ) 
    (h1 : cars = 4) 
    (h2 : car_cost = 0.95) 
    (h3 : race_track_cost = 6.00)
    (h4 : left_money = 8.00)
    (h5 : initial_money = (cars * car_cost) + race_track_cost + left_money) :
  initial_money = 17.80 := sorry

end edward_initial_money_l1252_125242


namespace foreign_stamps_count_l1252_125271

-- Define the conditions
variables (total_stamps : ℕ) (more_than_10_years_old : ℕ) (both_foreign_and_old : ℕ) (neither_foreign_nor_old : ℕ)

theorem foreign_stamps_count 
  (h1 : total_stamps = 200)
  (h2 : more_than_10_years_old = 60)
  (h3 : both_foreign_and_old = 20)
  (h4 : neither_foreign_nor_old = 70) : 
  ∃ (foreign_stamps : ℕ), foreign_stamps = 90 :=
by
  -- let foreign_stamps be the variable representing the number of foreign stamps
  let foreign_stamps := total_stamps - neither_foreign_nor_old - more_than_10_years_old + both_foreign_and_old
  use foreign_stamps
  -- the proof will develop here to show that foreign_stamps = 90
  sorry

end foreign_stamps_count_l1252_125271


namespace cheryl_gave_mms_to_sister_l1252_125280

-- Definitions for given conditions in the problem
def ate_after_lunch : ℕ := 7
def ate_after_dinner : ℕ := 5
def initial_mms : ℕ := 25

-- The statement to be proved
theorem cheryl_gave_mms_to_sister : (initial_mms - (ate_after_lunch + ate_after_dinner)) = 13 := by
  sorry

end cheryl_gave_mms_to_sister_l1252_125280


namespace total_sales_l1252_125296

theorem total_sales (S : ℝ) (remitted : ℝ) : 
  (∀ S, remitted = S - (0.05 * 10000 + 0.04 * (S - 10000)) → remitted = 31100) → S = 32500 :=
by
  sorry

end total_sales_l1252_125296


namespace m_le_n_l1252_125233

theorem m_le_n (k m n : ℕ) (hk_pos : 0 < k) (hm_pos : 0 < m) (hn_pos : 0 < n) (h : m^2 + n = k^2 + k) : m ≤ n := 
sorry

end m_le_n_l1252_125233


namespace june_walked_miles_l1252_125237

theorem june_walked_miles
  (step_counter_reset : ℕ)
  (resets_per_year : ℕ)
  (final_steps : ℕ)
  (steps_per_mile : ℕ)
  (h1 : step_counter_reset = 100000)
  (h2 : resets_per_year = 52)
  (h3 : final_steps = 30000)
  (h4 : steps_per_mile = 2000) :
  (resets_per_year * step_counter_reset + final_steps) / steps_per_mile = 2615 := 
by 
  sorry

end june_walked_miles_l1252_125237


namespace average_temperature_MTWT_l1252_125256

theorem average_temperature_MTWT (T_TWTF : ℝ) (T_M : ℝ) (T_F : ℝ) (T_MTWT : ℝ) :
    T_TWTF = 40 →
    T_M = 42 →
    T_F = 10 →
    T_MTWT = ((4 * T_TWTF - T_F + T_M) / 4) →
    T_MTWT = 48 := 
by
  intros hT_TWTF hT_M hT_F hT_MTWT
  rw [hT_TWTF, hT_M, hT_F] at hT_MTWT
  norm_num at hT_MTWT
  exact hT_MTWT

end average_temperature_MTWT_l1252_125256


namespace sufficiency_of_inequality_l1252_125222

theorem sufficiency_of_inequality (x : ℝ) (h : x > 5) : x^2 > 25 :=
sorry

end sufficiency_of_inequality_l1252_125222


namespace problem_statement_l1252_125246

variables (x y : ℝ)

def p : Prop := x > 1 ∧ y > 1
def q : Prop := x + y > 2

theorem problem_statement : (p x y → q x y) ∧ ¬(q x y → p x y) := sorry

end problem_statement_l1252_125246


namespace angle_C_triangle_area_l1252_125241

theorem angle_C 
  (a b c : ℝ) (A B C : ℝ)
  (h1 : a * Real.cos B + b * Real.cos A = -2 * c * Real.cos C) :
  C = 2 * Real.pi / 3 :=
sorry

theorem triangle_area 
  (a b c : ℝ) (C : ℝ)
  (h1 : a * Real.cos B + b * Real.cos A = -2 * c * Real.cos C)
  (h2 : c = Real.sqrt 7)
  (h3 : b = 2) :
  1 / 2 * a * b * Real.sin C = Real.sqrt 3 / 2 :=
sorry

end angle_C_triangle_area_l1252_125241


namespace Kara_books_proof_l1252_125227

-- Let's define the conditions and the proof statement in Lean 4

def Candice_books : ℕ := 18
def Amanda_books := Candice_books / 3
def Kara_books := Amanda_books / 2

theorem Kara_books_proof : Kara_books = 3 := by
  -- setting up the conditions based on the given problem.
  have Amanda_books_correct : Amanda_books = 6 := by
    exact Nat.div_eq_of_eq_mul_right (Nat.zero_lt_succ 2) (rfl) -- 18 / 3 = 6

  have Kara_books_correct : Kara_books = 3 := by
    exact Nat.div_eq_of_eq_mul_right (Nat.zero_lt_succ 1) Amanda_books_correct -- 6 / 2 = 3

  exact Kara_books_correct

end Kara_books_proof_l1252_125227


namespace number_of_heaps_is_5_l1252_125273

variable (bundles : ℕ) (bunches : ℕ) (heaps : ℕ) (total_removed : ℕ)
variable (sheets_per_bunch : ℕ) (sheets_per_bundle : ℕ) (sheets_per_heap : ℕ)

def number_of_heaps (bundles : ℕ) (sheets_per_bundle : ℕ)
                    (bunches : ℕ) (sheets_per_bunch : ℕ)
                    (total_removed : ℕ) (sheets_per_heap : ℕ) :=
  (total_removed - (bundles * sheets_per_bundle + bunches * sheets_per_bunch)) / sheets_per_heap

theorem number_of_heaps_is_5 :
  number_of_heaps 3 2 2 4 114 20 = 5 :=
by
  unfold number_of_heaps
  sorry

end number_of_heaps_is_5_l1252_125273


namespace fruit_weight_sister_and_dad_l1252_125276

-- Defining the problem statement and conditions
variable (strawberries_m blueberries_m raspberries_m : ℝ)
variable (strawberries_d blueberries_d raspberries_d : ℝ)
variable (strawberries_s blueberries_s raspberries_s : ℝ)
variable (total_weight : ℝ)

-- Given initial conditions
def conditions : Prop :=
  strawberries_m = 5 ∧
  blueberries_m = 3 ∧
  raspberries_m = 6 ∧
  strawberries_d = 2 * strawberries_m ∧
  blueberries_d = 2 * blueberries_m ∧
  raspberries_d = 2 * raspberries_m ∧
  strawberries_s = strawberries_m / 2 ∧
  blueberries_s = blueberries_m / 2 ∧
  raspberries_s = raspberries_m / 2 ∧
  total_weight = (strawberries_m + blueberries_m + raspberries_m) + 
                 (strawberries_d + blueberries_d + raspberries_d) + 
                 (strawberries_s + blueberries_s + raspberries_s)

-- Defining the property to prove
theorem fruit_weight_sister_and_dad :
  conditions strawberries_m blueberries_m raspberries_m strawberries_d blueberries_d raspberries_d strawberries_s blueberries_s raspberries_s total_weight →
  (strawberries_d + blueberries_d + raspberries_d) +
  (strawberries_s + blueberries_s + raspberries_s) = 35 := by
  sorry

end fruit_weight_sister_and_dad_l1252_125276


namespace functional_expression_point_M_coordinates_l1252_125208

variables (x y : ℝ) (k : ℝ)

-- Given conditions
def proportional_relation : Prop := y + 4 = k * (x - 3)
def initial_condition : Prop := (x = 1 → y = 0)
def point_M : Prop := ∃ m : ℝ, (m + 1, 2 * m) = (1, 0)

-- Proof of the functional expression
theorem functional_expression (h1 : proportional_relation x y k) (h2 : initial_condition x y) :
  ∃ k : ℝ, k = -2 ∧ y = -2 * x + 2 := 
sorry

-- Proof of the coordinates of point M
theorem point_M_coordinates (h : ∀ m : ℝ, (m + 1, 2 * m) = (1, 0)) :
  ∃ m : ℝ, m = 0 ∧ (m + 1, 2 * m) = (1, 0) := 
sorry

end functional_expression_point_M_coordinates_l1252_125208


namespace line_equation_through_points_and_area_l1252_125288

variable (a b S : ℝ)
variable (h_b_gt_a : b > a)
variable (h_area : S = 1/2 * (b - a) * (2 * S / (b - a)))

theorem line_equation_through_points_and_area :
  0 = -2 * S * x + (b - a)^2 * y + 2 * S * a - 2 * S * b := sorry

end line_equation_through_points_and_area_l1252_125288


namespace sara_staircase_l1252_125249

theorem sara_staircase (n : ℕ) (h : 2 * n * (n + 1) = 360) : n = 13 :=
sorry

end sara_staircase_l1252_125249


namespace greatest_possible_value_l1252_125289

theorem greatest_possible_value (x y : ℝ) (h1 : x^2 + y^2 = 98) (h2 : x * y = 40) : x + y = Real.sqrt 178 :=
by sorry

end greatest_possible_value_l1252_125289


namespace money_left_after_expenditures_l1252_125221

variable (initial_amount : ℝ) (P : initial_amount = 15000)
variable (gas_percentage food_fraction clothing_fraction entertainment_percentage : ℝ) 
variable (H1 : gas_percentage = 0.35) (H2 : food_fraction = 0.2) (H3 : clothing_fraction = 0.25) (H4 : entertainment_percentage = 0.15)

theorem money_left_after_expenditures
  (money_left : ℝ):
  money_left = initial_amount * (1 - gas_percentage) *
                (1 - food_fraction) * 
                (1 - clothing_fraction) * 
                (1 - entertainment_percentage) → 
  money_left = 4972.50 :=
by
  sorry

end money_left_after_expenditures_l1252_125221


namespace cost_option1_eq_cost_option2_eq_option1_final_cost_eq_option2_final_cost_eq_option1_more_cost_effective_optimal_cost_eq_l1252_125234

section price_calculations

variables {x : ℕ} (hx : x > 20)

-- Definitions based on the problem statement.
def suit_price : ℕ := 400
def tie_price : ℕ := 80

def option1_cost (x : ℕ) : ℕ :=
  20 * suit_price + tie_price * (x - 20)

def option2_cost (x : ℕ) : ℕ :=
  (20 * suit_price + tie_price * x) * 9 / 10

def option1_final_cost := option1_cost 30
def option2_final_cost := option2_cost 30

def optimal_cost : ℕ := 20 * suit_price + tie_price * 10 * 9 / 10

-- Proof obligations
theorem cost_option1_eq : option1_cost x = 80 * x + 6400 :=
by sorry

theorem cost_option2_eq : option2_cost x = 72 * x + 7200 :=
by sorry

theorem option1_final_cost_eq : option1_final_cost = 8800 :=
by sorry

theorem option2_final_cost_eq : option2_final_cost = 9360 :=
by sorry

theorem option1_more_cost_effective : option1_final_cost < option2_final_cost :=
by sorry

theorem optimal_cost_eq : optimal_cost = 8720 :=
by sorry

end price_calculations

end cost_option1_eq_cost_option2_eq_option1_final_cost_eq_option2_final_cost_eq_option1_more_cost_effective_optimal_cost_eq_l1252_125234


namespace cookie_distribution_l1252_125267

theorem cookie_distribution (b m l : ℕ)
  (h1 : b + m + l = 30)
  (h2 : m = 2 * b)
  (h3 : l = b + m) :
  b = 5 ∧ m = 10 ∧ l = 15 := 
by 
  sorry

end cookie_distribution_l1252_125267


namespace inequalities_no_solution_l1252_125223

theorem inequalities_no_solution (x n : ℝ) (h1 : x ≤ 1) (h2 : x ≥ n) : n > 1 :=
sorry

end inequalities_no_solution_l1252_125223


namespace parallelepiped_properties_l1252_125204

/--
In an oblique parallelepiped with the following properties:
- The height is 12 dm,
- The projection of the lateral edge on the base plane is 5 dm,
- A cross-section perpendicular to the lateral edge is a rhombus with:
  - An area of 24 dm²,
  - A diagonal of 8 dm,
Prove that:
1. The lateral surface area is 260 dm².
2. The volume is 312 dm³.
-/
theorem parallelepiped_properties
    (height : ℝ)
    (projection_lateral_edge : ℝ)
    (area_rhombus : ℝ)
    (diagonal_rhombus : ℝ)
    (lateral_surface_area : ℝ)
    (volume : ℝ) :
  height = 12 ∧
  projection_lateral_edge = 5 ∧
  area_rhombus = 24 ∧
  diagonal_rhombus = 8 ∧
  lateral_surface_area = 260 ∧
  volume = 312 :=
by
  sorry

end parallelepiped_properties_l1252_125204


namespace minimum_value_of_expression_l1252_125265

theorem minimum_value_of_expression {a c : ℝ} (h_pos : a > 0)
  (h_range : ∀ x, a * x ^ 2 - 4 * x + c ≥ 1) :
  ∃ a c, a > 0 ∧ (∀ x, a * x ^ 2 - 4 * x + c ≥ 1) ∧ (∃ a, a > 0 ∧ ∃ c, c - 1 = 4 / a ∧ (a / 4 + 9 / a = 3)) :=
by sorry

end minimum_value_of_expression_l1252_125265


namespace wendy_lost_lives_l1252_125243

theorem wendy_lost_lives (L : ℕ) (h1 : 10 - L + 37 = 41) : L = 6 :=
by
  sorry

end wendy_lost_lives_l1252_125243


namespace books_assigned_total_l1252_125205

-- Definitions for the conditions.
def Mcgregor_books := 34
def Floyd_books := 32
def remaining_books := 23

-- The total number of books assigned.
def total_books := Mcgregor_books + Floyd_books + remaining_books

-- The theorem that needs to be proven.
theorem books_assigned_total : total_books = 89 :=
by
  sorry

end books_assigned_total_l1252_125205
