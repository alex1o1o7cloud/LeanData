import Mathlib

namespace NUMINAMATH_GPT_smallest_number_of_cubes_filling_box_l1214_121412
open Nat

theorem smallest_number_of_cubes_filling_box (L W D : ℕ) (hL : L = 27) (hW : W = 15) (hD : D = 6) :
  let gcd := 3
  let cubes_along_length := L / gcd
  let cubes_along_width := W / gcd
  let cubes_along_depth := D / gcd
  cubes_along_length * cubes_along_width * cubes_along_depth = 90 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_cubes_filling_box_l1214_121412


namespace NUMINAMATH_GPT_maria_savings_l1214_121411

variable (S : ℝ) -- Define S as a real number (amount saved initially)

-- Conditions
def bike_cost : ℝ := 600
def additional_money : ℝ := 250 + 230

-- Theorem statement
theorem maria_savings : S + additional_money = bike_cost → S = 120 :=
by
  intro h -- Assume the hypothesis (condition)
  sorry -- Proof will go here

end NUMINAMATH_GPT_maria_savings_l1214_121411


namespace NUMINAMATH_GPT_discount_difference_l1214_121401

def original_amount : ℚ := 20000
def single_discount_rate : ℚ := 0.30
def first_discount_rate : ℚ := 0.25
def second_discount_rate : ℚ := 0.05

theorem discount_difference :
  (original_amount * (1 - single_discount_rate)) - (original_amount * (1 - first_discount_rate) * (1 - second_discount_rate)) = 250 := by
  sorry

end NUMINAMATH_GPT_discount_difference_l1214_121401


namespace NUMINAMATH_GPT_math_problem_l1214_121414

-- Define the mixed numbers as fractions
def mixed_3_1_5 := 16 / 5 -- 3 + 1/5 = 16/5
def mixed_4_1_2 := 9 / 2  -- 4 + 1/2 = 9/2
def mixed_2_3_4 := 11 / 4 -- 2 + 3/4 = 11/4
def mixed_1_2_3 := 5 / 3  -- 1 + 2/3 = 5/3

-- Define the main expression
def main_expr := 53 * (mixed_3_1_5 - mixed_4_1_2) / (mixed_2_3_4 + mixed_1_2_3)

-- Define the expected answer in its fractional form
def expected_result := -78 / 5

-- The theorem to prove the main expression equals the expected mixed number
theorem math_problem : main_expr = expected_result :=
by sorry

end NUMINAMATH_GPT_math_problem_l1214_121414


namespace NUMINAMATH_GPT_hyperbola_condition_l1214_121433

noncomputable def hyperbola_eccentricity_difference (a b : ℝ) (h1 : a > b) (h2 : b > 0) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  let e_2pi_over_3 := Real.sqrt 3 + 1
  let e_pi_over_3 := (Real.sqrt 3) / 3 + 1
  e_2pi_over_3 - e_pi_over_3

theorem hyperbola_condition (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  hyperbola_eccentricity_difference a b h1 h2 = (2 * Real.sqrt 3) / 3 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_condition_l1214_121433


namespace NUMINAMATH_GPT_find_a_l1214_121427

theorem find_a (a : ℝ) 
  (h1 : a < 0)
  (h2 : a < 1/3)
  (h3 : -2 * a + (1 - 3 * a) = 6) : 
  a = -1 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_l1214_121427


namespace NUMINAMATH_GPT_common_point_of_arithmetic_progression_lines_l1214_121443

theorem common_point_of_arithmetic_progression_lines 
  (a d : ℝ) 
  (h₁ : a ≠ 0)
  (h_d_ne_zero : d ≠ 0) 
  (h₃ : ∀ (x y : ℝ), (x = -1 ∧ y = 1) ↔ (∃ a d : ℝ, a ≠ 0 ∧ d ≠ 0 ∧ a*(x) + (a-d)*y = (a-2*d))) :
  (∀ (x y : ℝ), (a ≠ 0 ∧ d ≠ 0 ∧ a*(x) + (a-d)*y = a-2*d) → x = -1 ∧ y = 1) :=
by 
  sorry

end NUMINAMATH_GPT_common_point_of_arithmetic_progression_lines_l1214_121443


namespace NUMINAMATH_GPT_find_age_of_older_friend_l1214_121499

theorem find_age_of_older_friend (A B C : ℝ) 
  (h1 : A - B = 2.5)
  (h2 : A - C = 3.75)
  (h3 : A + B + C = 110.5)
  (h4 : B = 2 * C) : 
  A = 104.25 :=
by
  sorry

end NUMINAMATH_GPT_find_age_of_older_friend_l1214_121499


namespace NUMINAMATH_GPT_usage_difference_correct_l1214_121462

def computerUsageLastWeek : ℕ := 91

def computerUsageThisWeek : ℕ :=
  let first4days := 4 * 8
  let last3days := 3 * 10
  first4days + last3days

def computerUsageFollowingWeek : ℕ :=
  let weekdays := 5 * (5 + 3)
  let weekends := 2 * 12
  weekdays + weekends

def differenceThisWeek : ℕ := computerUsageLastWeek - computerUsageThisWeek
def differenceFollowingWeek : ℕ := computerUsageLastWeek - computerUsageFollowingWeek

theorem usage_difference_correct :
  differenceThisWeek = 29 ∧ differenceFollowingWeek = 27 := by
  sorry

end NUMINAMATH_GPT_usage_difference_correct_l1214_121462


namespace NUMINAMATH_GPT_plant_species_numbering_impossible_l1214_121429

theorem plant_species_numbering_impossible :
  ∀ (n m : ℕ), 2 ≤ n ∨ n ≤ 20000 ∧ 2 ≤ m ∨ m ≤ 20000 ∧ n ≠ m → 
  ∃ x y : ℕ, 2 ≤ x ∨ x ≤ 20000 ∧ 2 ≤ y ∨ y ≤ 20000 ∧ x ≠ y ∧
  (∀ k : ℕ, gcd x k = gcd n k ∧ gcd y k = gcd m k) :=
  by sorry

end NUMINAMATH_GPT_plant_species_numbering_impossible_l1214_121429


namespace NUMINAMATH_GPT_problem1_problem2_l1214_121446

def A := { x : ℝ | -2 < x ∧ x ≤ 4 }
def B := { x : ℝ | 2 - x < 1 }
def U := ℝ
def complement_B := { x : ℝ | x ≤ 1 }

theorem problem1 : { x : ℝ | 1 < x ∧ x ≤ 4 } = { x : ℝ | x ∈ A ∧ x ∈ B } := 
by sorry

theorem problem2 : { x : ℝ | x ≤ 4 } = { x : ℝ | x ∈ A ∨ x ∈ complement_B } := 
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1214_121446


namespace NUMINAMATH_GPT_sin_240_eq_neg_sqrt3_div_2_l1214_121447

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_240_eq_neg_sqrt3_div_2_l1214_121447


namespace NUMINAMATH_GPT_no_pair_of_primes_l1214_121484

theorem no_pair_of_primes (p q : ℕ) (hp_prime : Prime p) (hq_prime : Prime q) (h_gt : p > q) :
  ¬ (∃ (h : ℤ), 2 * (p^2 - q^2) = 8 * h + 4) :=
by
  sorry

end NUMINAMATH_GPT_no_pair_of_primes_l1214_121484


namespace NUMINAMATH_GPT_propane_tank_and_burner_cost_l1214_121463

theorem propane_tank_and_burner_cost
(Total_money: ℝ)
(Sheet_cost: ℝ)
(Rope_cost: ℝ)
(Helium_cost_per_oz: ℝ)
(Lift_per_oz: ℝ)
(Max_height: ℝ)
(ht: Total_money = 200)
(hs: Sheet_cost = 42)
(hr: Rope_cost = 18)
(hh: Helium_cost_per_oz = 1.50)
(hlo: Lift_per_oz = 113)
(hm: Max_height = 9492)
:
(Total_money - (Sheet_cost + Rope_cost) 
 - (Max_height / Lift_per_oz * Helium_cost_per_oz) 
 = 14) :=
by
  sorry

end NUMINAMATH_GPT_propane_tank_and_burner_cost_l1214_121463


namespace NUMINAMATH_GPT_cos_alpha_minus_beta_l1214_121465

theorem cos_alpha_minus_beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) (h_cos_add : Real.cos (α + β) = -5 / 13)
  (h_tan_sum : Real.tan α + Real.tan β = 3) :
  Real.cos (α - β) = 1 :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_minus_beta_l1214_121465


namespace NUMINAMATH_GPT_sum_of_reciprocals_l1214_121440

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 16) (h2 : x * y = 48) : (1 / x + 1 / y) = (1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l1214_121440


namespace NUMINAMATH_GPT_find_room_length_l1214_121456

variable (width : ℝ) (cost rate : ℝ) (length : ℝ)

theorem find_room_length (h_width : width = 4.75)
  (h_cost : cost = 34200)
  (h_rate : rate = 900)
  (h_area : cost / rate = length * width) :
  length = 8 :=
sorry

end NUMINAMATH_GPT_find_room_length_l1214_121456


namespace NUMINAMATH_GPT_imaginary_part_of_z_l1214_121441

theorem imaginary_part_of_z (z : ℂ) (h : (z / (1 - I)) = (3 + I)) : z.im = -2 :=
sorry

end NUMINAMATH_GPT_imaginary_part_of_z_l1214_121441


namespace NUMINAMATH_GPT_proof_ab_value_l1214_121466

theorem proof_ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := 
by
  sorry

end NUMINAMATH_GPT_proof_ab_value_l1214_121466


namespace NUMINAMATH_GPT_sqrt_equation_solution_l1214_121495

theorem sqrt_equation_solution (x : ℝ) :
  (Real.sqrt ((2 + Real.sqrt 3) ^ x) + Real.sqrt ((2 - Real.sqrt 3) ^ x) = 4) ↔ (x = 2 ∨ x = -2) := 
by
  sorry

end NUMINAMATH_GPT_sqrt_equation_solution_l1214_121495


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l1214_121404

theorem boat_speed_in_still_water (b s : ℝ) (h1 : b + s = 11) (h2 : b - s = 7) : b = 9 :=
by sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l1214_121404


namespace NUMINAMATH_GPT_second_hand_degrees_per_minute_l1214_121434

theorem second_hand_degrees_per_minute (clock_gains_5_minutes_per_hour : true) :
  (360 / 60 = 6) := 
by
  sorry

end NUMINAMATH_GPT_second_hand_degrees_per_minute_l1214_121434


namespace NUMINAMATH_GPT_yellow_balls_count_l1214_121497

theorem yellow_balls_count (r y : ℕ) (h1 : r = 9) (h2 : (r : ℚ) / (r + y) = 1 / 3) : y = 18 := 
by
  sorry

end NUMINAMATH_GPT_yellow_balls_count_l1214_121497


namespace NUMINAMATH_GPT_problem_lean_statement_l1214_121420

theorem problem_lean_statement (a b c : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 2) = 2 * x ^ 2 + 5 * x + 3)
  (h2 : ∀ x, f x = a * x ^ 2 + b * x + c) : a + b + c = 0 :=
by sorry

end NUMINAMATH_GPT_problem_lean_statement_l1214_121420


namespace NUMINAMATH_GPT_f_monotonic_m_range_l1214_121451

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.tan x - 2 * x

theorem f_monotonic {x : ℝ} (h : x ∈ Set.Ioo (-Real.pi / 2) (Real.pi / 2)) :
  Monotone f :=
sorry

theorem m_range {x : ℝ} (h : x ∈ Set.Ioo 0 (Real.pi / 2)) {m : ℝ} (hm : f x ≥ m * x^2) :
  m ≤ 0 :=
sorry

end NUMINAMATH_GPT_f_monotonic_m_range_l1214_121451


namespace NUMINAMATH_GPT_debra_probability_theorem_l1214_121452

-- Define event for Debra's coin flipping game starting with "HTT"
def debra_coin_game_event : Prop := 
  let heads_probability : ℝ := 0.5
  let tails_probability : ℝ := 0.5
  let initial_prob : ℝ := heads_probability * tails_probability * tails_probability
  let Q : ℝ := 1 / 3  -- the computed probability of getting HH after HTT
  let final_probability : ℝ := initial_prob * Q
  final_probability = 1 / 24

-- The theorem statement
theorem debra_probability_theorem :
  debra_coin_game_event := 
by
  sorry

end NUMINAMATH_GPT_debra_probability_theorem_l1214_121452


namespace NUMINAMATH_GPT_negation_proposition_real_l1214_121408

theorem negation_proposition_real :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ ∃ x : ℝ, x^2 + x + 1 ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_real_l1214_121408


namespace NUMINAMATH_GPT_average_speed_of_horse_l1214_121410

/-- Definitions of the conditions given in the problem. --/
def pony_speed : ℕ := 20
def pony_head_start_hours : ℕ := 3
def horse_chase_hours : ℕ := 4

-- Define a proof problem for the average speed of the horse.
theorem average_speed_of_horse : (pony_head_start_hours * pony_speed + horse_chase_hours * pony_speed) / horse_chase_hours = 35 := by
  -- Setting up the necessary distances
  let pony_head_start_distance := pony_head_start_hours * pony_speed
  let pony_additional_distance := horse_chase_hours * pony_speed
  let total_pony_distance := pony_head_start_distance + pony_additional_distance
  -- Asserting the average speed of the horse
  let horse_average_speed := total_pony_distance / horse_chase_hours
  show horse_average_speed = 35
  sorry

end NUMINAMATH_GPT_average_speed_of_horse_l1214_121410


namespace NUMINAMATH_GPT_heather_biked_per_day_l1214_121488

def total_kilometers_biked : ℝ := 320
def days_biked : ℝ := 8
def kilometers_per_day : ℝ := 40

theorem heather_biked_per_day : total_kilometers_biked / days_biked = kilometers_per_day := 
by
  -- Proof will be inserted here
  sorry

end NUMINAMATH_GPT_heather_biked_per_day_l1214_121488


namespace NUMINAMATH_GPT_frank_initial_candy_l1214_121438

theorem frank_initial_candy (n : ℕ) (h1 : n = 21) (h2 : 2 > 0) :
  2 * n = 42 :=
by
  --* Use the hypotheses to establish the required proof
  sorry

end NUMINAMATH_GPT_frank_initial_candy_l1214_121438


namespace NUMINAMATH_GPT_area_ratio_greater_than_two_ninths_l1214_121498

variable {α : Type*} [LinearOrder α] [LinearOrderedField α]

def area_triangle (A B C : α) : α := sorry -- Placeholder for the area function
noncomputable def triangle_division (A B C P Q R : α) : Prop :=
  -- Placeholder for division condition
  -- Here you would check that P, Q, and R divide the perimeter of triangle ABC into three equal parts
  sorry

theorem area_ratio_greater_than_two_ninths (A B C P Q R : α) :
  triangle_division A B C P Q R → area_triangle P Q R > (2 / 9) * area_triangle A B C :=
by
  sorry -- The proof goes here

end NUMINAMATH_GPT_area_ratio_greater_than_two_ninths_l1214_121498


namespace NUMINAMATH_GPT_ages_of_boys_l1214_121475

theorem ages_of_boys (a b c : ℕ) (h : a + b + c = 29) (h₁ : a = b) (h₂ : c = 11) : a = 9 ∧ b = 9 := 
by
  sorry

end NUMINAMATH_GPT_ages_of_boys_l1214_121475


namespace NUMINAMATH_GPT_exists_n_such_that_5_pow_n_has_six_consecutive_zeros_l1214_121469

theorem exists_n_such_that_5_pow_n_has_six_consecutive_zeros :
  ∃ n : ℕ, n < 1000000 ∧ ∃ k : ℕ, k = 20 ∧ 5 ^ n % (10 ^ k) < (10 ^ (k - 6)) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_exists_n_such_that_5_pow_n_has_six_consecutive_zeros_l1214_121469


namespace NUMINAMATH_GPT_power_mod_congruence_l1214_121481

theorem power_mod_congruence (h : 3^400 ≡ 1 [MOD 500]) : 3^800 ≡ 1 [MOD 500] :=
by {
  sorry
}

end NUMINAMATH_GPT_power_mod_congruence_l1214_121481


namespace NUMINAMATH_GPT_A_roster_method_l1214_121478

open Set

def A : Set ℤ := {x : ℤ | (∃ (n : ℤ), n > 0 ∧ 6 / (5 - x) = n) }

theorem A_roster_method :
  A = {-1, 2, 3, 4} :=
  sorry

end NUMINAMATH_GPT_A_roster_method_l1214_121478


namespace NUMINAMATH_GPT_distance_between_parallel_lines_eq_2_l1214_121461

def line1 (x y : ℝ) : Prop := 3 * x - 4 * y + 2 = 0
def line2 (x y : ℝ) : Prop := 3 * x - 4 * y - 8 = 0

theorem distance_between_parallel_lines_eq_2 :
  let A := 3
  let B := -4
  let c1 := 2
  let c2 := -8
  let d := (|c1 - c2| / Real.sqrt (A^2 + B^2))
  d = 2 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_parallel_lines_eq_2_l1214_121461


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1214_121494

theorem solution_set_of_inequality:
  {x : ℝ | 1 < abs (2 * x - 1) ∧ abs (2 * x - 1) < 3} = 
  {x : ℝ | -1 < x ∧ x < 0} ∪ 
  {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1214_121494


namespace NUMINAMATH_GPT_compare_fractions_l1214_121423

theorem compare_fractions (a b m : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : m > 0) : 
  (b / a) < ((b + m) / (a + m)) :=
sorry

end NUMINAMATH_GPT_compare_fractions_l1214_121423


namespace NUMINAMATH_GPT_car_rental_cost_per_mile_l1214_121449

def daily_rental_rate := 29.0
def total_amount_paid := 46.12
def miles_driven := 214.0

theorem car_rental_cost_per_mile : 
  (total_amount_paid - daily_rental_rate) / miles_driven = 0.08 := 
by
  sorry

end NUMINAMATH_GPT_car_rental_cost_per_mile_l1214_121449


namespace NUMINAMATH_GPT_rosy_fish_is_twelve_l1214_121407

/-- Let lilly_fish be the number of fish Lilly has. -/
def lilly_fish : ℕ := 10

/-- Let total_fish be the total number of fish Lilly and Rosy have together. -/
def total_fish : ℕ := 22

/-- Prove that the number of fish Rosy has is equal to 12. -/
theorem rosy_fish_is_twelve : (total_fish - lilly_fish) = 12 :=
by sorry

end NUMINAMATH_GPT_rosy_fish_is_twelve_l1214_121407


namespace NUMINAMATH_GPT_increasing_function_iff_l1214_121413

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a ^ x else (3 - a) * x + (1 / 2) * a

theorem increasing_function_iff (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ 2 ≤ a ∧ a < 3 :=
by
  sorry

end NUMINAMATH_GPT_increasing_function_iff_l1214_121413


namespace NUMINAMATH_GPT_fraction_calculation_l1214_121457

theorem fraction_calculation :
  (1 / 4) * (1 / 3) * (1 / 6) * 144 + (1 / 2) = (5 / 2) :=
by
  sorry

end NUMINAMATH_GPT_fraction_calculation_l1214_121457


namespace NUMINAMATH_GPT_max_value_of_linear_combination_of_m_n_k_l1214_121477

-- The style grants us maximum flexibility for definitions.
theorem max_value_of_linear_combination_of_m_n_k 
  (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ) (m n k : ℕ)
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ m → a i % 3 = 1)
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → b i % 3 = 2)
  (h3 : ∀ i, 1 ≤ i ∧ i ≤ k → c i % 3 = 0)
  (h4 : Function.Injective a)
  (h5 : Function.Injective b)
  (h6 : Function.Injective c)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ b j ∧ a i ≠ c j ∧ b i ≠ c j)
  (h_sum : (Finset.range m).sum a + (Finset.range n).sum b + (Finset.range k).sum c = 2007)
  : 4 * m + 3 * n + 5 * k ≤ 256 := by
  sorry

end NUMINAMATH_GPT_max_value_of_linear_combination_of_m_n_k_l1214_121477


namespace NUMINAMATH_GPT_value_of_x_squared_plus_y_squared_l1214_121416

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : x + y = -4) (h2 : x = 6 / y) : 
  x^2 + y^2 = 4 :=
  sorry

end NUMINAMATH_GPT_value_of_x_squared_plus_y_squared_l1214_121416


namespace NUMINAMATH_GPT_problem_2014_minus_4102_l1214_121493

theorem problem_2014_minus_4102 : 2014 - 4102 = -2088 := 
by
  -- The proof is omitted as per the requirement
  sorry

end NUMINAMATH_GPT_problem_2014_minus_4102_l1214_121493


namespace NUMINAMATH_GPT_exists_n_divisible_by_5_l1214_121479

open Int

theorem exists_n_divisible_by_5 
  (a b c d m : ℤ) 
  (h1 : 5 ∣ (a * m^3 + b * m^2 + c * m + d)) 
  (h2 : ¬ (5 ∣ d)) :
  ∃ n : ℤ, 5 ∣ (d * n^3 + c * n^2 + b * n + a) :=
by
  sorry

end NUMINAMATH_GPT_exists_n_divisible_by_5_l1214_121479


namespace NUMINAMATH_GPT_symmetric_circle_l1214_121409

theorem symmetric_circle (x y : ℝ) :
  let C := { p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 1 }
  let L := { p : ℝ × ℝ | p.1 + p.2 = 1 }
  ∃ C' : ℝ × ℝ → Prop, (∀ p, C' p ↔ (p.1)^2 + (p.2)^2 = 1) :=
sorry

end NUMINAMATH_GPT_symmetric_circle_l1214_121409


namespace NUMINAMATH_GPT_no_sum_of_three_squares_l1214_121430

theorem no_sum_of_three_squares (n : ℤ) (h : n % 8 = 7) : 
  ¬ ∃ a b c : ℤ, a^2 + b^2 + c^2 = n :=
by 
sorry

end NUMINAMATH_GPT_no_sum_of_three_squares_l1214_121430


namespace NUMINAMATH_GPT_smallest_positive_integer_n_l1214_121490

def contains_digit_9 (n : ℕ) : Prop := 
  ∃ m : ℕ, (10^m) ∣ n ∧ (n / 10^m) % 10 = 9

theorem smallest_positive_integer_n :
  ∃ n : ℕ, (∀ k : ℕ, k > 0 ∧ k < n → 
  (∃ a b : ℕ, k = 2^a * 5^b * 3) ∧ contains_digit_9 k ∧ (k % 3 = 0))
  → n = 90 :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_n_l1214_121490


namespace NUMINAMATH_GPT_total_peanuts_is_388_l1214_121437

def peanuts_total (jose kenya marcos : ℕ) : ℕ :=
  jose + kenya + marcos

theorem total_peanuts_is_388 :
  ∀ (jose kenya marcos : ℕ),
    (jose = 85) →
    (kenya = jose + 48) →
    (marcos = kenya + 37) →
    peanuts_total jose kenya marcos = 388 := 
by
  intros jose kenya marcos h_jose h_kenya h_marcos
  sorry

end NUMINAMATH_GPT_total_peanuts_is_388_l1214_121437


namespace NUMINAMATH_GPT_y_coordinate_of_third_vertex_eq_l1214_121480

theorem y_coordinate_of_third_vertex_eq (x1 x2 y1 y2 : ℝ)
    (h1 : x1 = 0) 
    (h2 : y1 = 3) 
    (h3 : x2 = 10) 
    (h4 : y2 = 3) 
    (h5 : x1 ≠ x2) 
    (h6 : y1 = y2) 
    : ∃ y3 : ℝ, y3 = 3 + 5 * Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_y_coordinate_of_third_vertex_eq_l1214_121480


namespace NUMINAMATH_GPT_maisy_new_job_hours_l1214_121471

-- Define the conditions
def current_job_earnings : ℚ := 80
def new_job_wage_per_hour : ℚ := 15
def new_job_bonus : ℚ := 35
def earnings_difference : ℚ := 15

-- Define the problem
theorem maisy_new_job_hours (h : ℚ) 
  (h1 : current_job_earnings = 80) 
  (h2 : new_job_wage_per_hour * h + new_job_bonus = current_job_earnings + earnings_difference) :
  h = 4 :=
  sorry

end NUMINAMATH_GPT_maisy_new_job_hours_l1214_121471


namespace NUMINAMATH_GPT_factor_expression_l1214_121473

theorem factor_expression (x : ℝ) : 5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1214_121473


namespace NUMINAMATH_GPT_sum_of_digits_second_smallest_mult_of_lcm_l1214_121425

theorem sum_of_digits_second_smallest_mult_of_lcm :
  let lcm12345678 := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8
  let M := 2 * lcm12345678
  (Nat.digits 10 M).sum = 15 := by
    -- Definitions from the problem statement
    let lcm12345678 := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8
    let M := 2 * lcm12345678
    sorry

end NUMINAMATH_GPT_sum_of_digits_second_smallest_mult_of_lcm_l1214_121425


namespace NUMINAMATH_GPT_chess_pieces_missing_l1214_121468

theorem chess_pieces_missing 
  (total_pieces : ℕ) (pieces_present : ℕ) (h1 : total_pieces = 32) (h2 : pieces_present = 28) : 
  total_pieces - pieces_present = 4 := 
by
  -- Sorry proof
  sorry

end NUMINAMATH_GPT_chess_pieces_missing_l1214_121468


namespace NUMINAMATH_GPT_percentage_managers_decrease_l1214_121442

theorem percentage_managers_decrease
  (employees : ℕ)
  (initial_percentage : ℝ)
  (managers_leave : ℝ)
  (new_percentage : ℝ)
  (h1 : employees = 200)
  (h2 : initial_percentage = 99)
  (h3 : managers_leave = 100)
  (h4 : new_percentage = 98) :
  ((initial_percentage / 100 * employees - managers_leave) / (employees - managers_leave) * 100 = new_percentage) :=
by
  -- To be proven
  sorry

end NUMINAMATH_GPT_percentage_managers_decrease_l1214_121442


namespace NUMINAMATH_GPT_total_books_l1214_121421

theorem total_books (books_last_month : ℕ) (goal_factor : ℕ) (books_this_month : ℕ) (total_books : ℕ) 
  (h1 : books_last_month = 4) 
  (h2 : goal_factor = 2) 
  (h3 : books_this_month = goal_factor * books_last_month) 
  (h4 : total_books = books_last_month + books_this_month) 
  : total_books = 12 := 
by
  sorry

end NUMINAMATH_GPT_total_books_l1214_121421


namespace NUMINAMATH_GPT_length_of_train_l1214_121454

theorem length_of_train :
  ∀ (L : ℝ) (V : ℝ),
  (∀ t p : ℝ, t = 14 → p = 535.7142857142857 → V = L / t) →
  (∀ t p : ℝ, t = 39 → p = 535.7142857142857 → V = (L + p) / t) →
  L = 300 :=
by
  sorry

end NUMINAMATH_GPT_length_of_train_l1214_121454


namespace NUMINAMATH_GPT_initial_invitation_count_l1214_121400

def people_invited (didnt_show : ℕ) (num_tables : ℕ) (people_per_table : ℕ) : ℕ :=
  didnt_show + num_tables * people_per_table

theorem initial_invitation_count (didnt_show : ℕ) (num_tables : ℕ) (people_per_table : ℕ)
    (h1 : didnt_show = 35) (h2 : num_tables = 5) (h3 : people_per_table = 2) :
  people_invited didnt_show num_tables people_per_table = 45 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end NUMINAMATH_GPT_initial_invitation_count_l1214_121400


namespace NUMINAMATH_GPT_cayli_combinations_l1214_121459

theorem cayli_combinations (art_choices sports_choices music_choices : ℕ)
  (h1 : art_choices = 2)
  (h2 : sports_choices = 3)
  (h3 : music_choices = 4) :
  art_choices * sports_choices * music_choices = 24 := by
  sorry

end NUMINAMATH_GPT_cayli_combinations_l1214_121459


namespace NUMINAMATH_GPT_total_weight_puffy_muffy_l1214_121453

def scruffy_weight : ℕ := 12
def muffy_weight : ℕ := scruffy_weight - 3
def puffy_weight : ℕ := muffy_weight + 5

theorem total_weight_puffy_muffy : puffy_weight + muffy_weight = 23 := 
by
  sorry

end NUMINAMATH_GPT_total_weight_puffy_muffy_l1214_121453


namespace NUMINAMATH_GPT_vegetarian_count_l1214_121491

theorem vegetarian_count (only_veg only_non_veg both_veg_non_veg : ℕ) 
  (h1 : only_veg = 19) (h2 : only_non_veg = 9) (h3 : both_veg_non_veg = 12) : 
  (only_veg + both_veg_non_veg = 31) :=
by
  -- We leave the proof here
  sorry

end NUMINAMATH_GPT_vegetarian_count_l1214_121491


namespace NUMINAMATH_GPT_ratio_ashley_mary_l1214_121426

-- Definitions based on conditions
def sum_ages (A M : ℕ) := A + M = 22
def ashley_age (A : ℕ) := A = 8

-- Theorem stating the ratio of Ashley's age to Mary's age
theorem ratio_ashley_mary (A M : ℕ) 
  (h1 : sum_ages A M)
  (h2 : ashley_age A) : 
  (A : ℚ) / (M : ℚ) = 4 / 7 :=
by
  -- Skipping the proof as specified
  sorry

end NUMINAMATH_GPT_ratio_ashley_mary_l1214_121426


namespace NUMINAMATH_GPT_time_after_2051_hours_l1214_121486

theorem time_after_2051_hours (h₀ : 9 ≤ 11): 
  (9 + 2051 % 12) % 12 = 8 :=
by {
  -- proving the statement here
  sorry
}

end NUMINAMATH_GPT_time_after_2051_hours_l1214_121486


namespace NUMINAMATH_GPT_sara_total_payment_l1214_121403

structure DecorationCosts where
  balloons: ℝ
  tablecloths: ℝ
  streamers: ℝ
  banners: ℝ
  confetti: ℝ
  change_received: ℝ

noncomputable def total_cost (c : DecorationCosts) : ℝ :=
  c.balloons + c.tablecloths + c.streamers + c.banners + c.confetti

noncomputable def amount_given (c : DecorationCosts) : ℝ :=
  total_cost c + c.change_received

theorem sara_total_payment : 
  ∀ (costs : DecorationCosts), 
    costs = ⟨3.50, 18.25, 9.10, 14.65, 7.40, 6.38⟩ →
    amount_given costs = 59.28 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sara_total_payment_l1214_121403


namespace NUMINAMATH_GPT_rice_flour_weights_l1214_121428

variables (r f : ℝ)

theorem rice_flour_weights :
  (8 * r + 6 * f = 550) ∧ (4 * r + 7 * f = 375) → (r = 50) ∧ (f = 25) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_rice_flour_weights_l1214_121428


namespace NUMINAMATH_GPT_original_number_not_800_l1214_121460

theorem original_number_not_800 (x : ℕ) (h : 10 * x = x + 720) : x ≠ 800 :=
by {
  sorry
}

end NUMINAMATH_GPT_original_number_not_800_l1214_121460


namespace NUMINAMATH_GPT_therapist_charge_difference_l1214_121424

theorem therapist_charge_difference :
  ∃ F A : ℝ, F + 4 * A = 350 ∧ F + A = 161 ∧ F - A = 35 :=
by {
  -- Placeholder for the actual proof.
  sorry
}

end NUMINAMATH_GPT_therapist_charge_difference_l1214_121424


namespace NUMINAMATH_GPT_problem1_monotonic_decreasing_problem2_monotonic_decreasing_pos_problem2_monotonic_decreasing_neg_l1214_121439

-- Problem 1: Monotonicity of f(x) = 1 - 3x on ℝ
theorem problem1_monotonic_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → (1 - 3 * x1) > (1 - 3 * x2) :=
by
  -- Proof (skipped)
  sorry

-- Problem 2: Monotonicity of g(x) = 1/x + 2 on (0, ∞) and (-∞, 0)
theorem problem2_monotonic_decreasing_pos : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → (1 / x1 + 2) > (1 / x2 + 2) :=
by
  -- Proof (skipped)
  sorry

theorem problem2_monotonic_decreasing_neg : ∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 < x2 → (1 / x1 + 2) > (1 / x2 + 2) :=
by
  -- Proof (skipped)
  sorry

end NUMINAMATH_GPT_problem1_monotonic_decreasing_problem2_monotonic_decreasing_pos_problem2_monotonic_decreasing_neg_l1214_121439


namespace NUMINAMATH_GPT_fourth_person_knight_l1214_121464

-- Let P1, P2, P3, and P4 be the statements made by the four people respectively.
def P1 := ∀ x y z w : Prop, x = y ∧ y = z ∧ z = w ∧ w = ¬w
def P2 := ∃! x y z w : Prop, x = true
def P3 := ∀ x y z w : Prop, (x = true ∧ y = true ∧ z = false) ∨ (x = true ∧ y = false ∧ z = true) ∨ (x = false ∧ y = true ∧ z = true)
def P4 := ∀ x : Prop, x = true → x = true

-- Now let's express the requirement of proving that the fourth person is a knight
theorem fourth_person_knight : P4 := by
  sorry

end NUMINAMATH_GPT_fourth_person_knight_l1214_121464


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_l1214_121487

theorem sum_of_arithmetic_sequence (S : ℕ → ℝ) (a₁ d : ℝ) 
  (h1 : ∀ n, S n = n * a₁ + (n - 1) * n / 2 * d)
  (h2 : S 1 / S 4 = 1 / 10) :
  S 3 / S 5 = 2 / 5 := 
sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_l1214_121487


namespace NUMINAMATH_GPT_cos_neg_pi_over_3_l1214_121467

noncomputable def angle := - (Real.pi / 3)

theorem cos_neg_pi_over_3 : Real.cos angle = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_neg_pi_over_3_l1214_121467


namespace NUMINAMATH_GPT_equal_roots_h_l1214_121415

theorem equal_roots_h (h : ℝ) : (∀ x : ℝ, 3 * x^2 - 4 * x + (h / 3) = 0) -> h = 4 :=
by 
  sorry

end NUMINAMATH_GPT_equal_roots_h_l1214_121415


namespace NUMINAMATH_GPT_common_ratio_solution_l1214_121417

-- Define the problem condition
def geometric_sum_condition (a1 : ℝ) (q : ℝ) : Prop :=
  (a1 * (1 - q^3)) / (1 - q) = 3 * a1

-- Define the theorem we want to prove
theorem common_ratio_solution (a1 : ℝ) (q : ℝ) (h : geometric_sum_condition a1 q) :
  q = 1 ∨ q = -2 :=
sorry

end NUMINAMATH_GPT_common_ratio_solution_l1214_121417


namespace NUMINAMATH_GPT_initial_production_rate_l1214_121458

theorem initial_production_rate 
  (x : ℝ)
  (h1 : 60 <= (60 * x) / 30 - 60 + 1800)
  (h2 : 60 <= 120)
  (h3 : 30 = (120 / (60 / x + 1))) : x = 20 := by
  sorry

end NUMINAMATH_GPT_initial_production_rate_l1214_121458


namespace NUMINAMATH_GPT_problem_solution_l1214_121432

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.rpow 3 (1 / 3)
noncomputable def c : ℝ := Real.log 2 / Real.log 3

theorem problem_solution : c < a ∧ a < b := 
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1214_121432


namespace NUMINAMATH_GPT_most_likely_number_of_cars_l1214_121436

theorem most_likely_number_of_cars 
    (cars_in_first_10_seconds : ℕ := 6) 
    (time_for_first_10_seconds : ℕ := 10) 
    (total_time_seconds : ℕ := 165) 
    (constant_speed : Prop := true) : 
    ∃ (num_cars : ℕ), num_cars = 100 :=
by
  sorry

end NUMINAMATH_GPT_most_likely_number_of_cars_l1214_121436


namespace NUMINAMATH_GPT_parallel_resistors_l1214_121444
noncomputable def resistance_R (x y z w : ℝ) : ℝ :=
  1 / (1/x + 1/y + 1/z + 1/w)

theorem parallel_resistors :
  resistance_R 5 7 3 9 = 315 / 248 :=
by
  sorry

end NUMINAMATH_GPT_parallel_resistors_l1214_121444


namespace NUMINAMATH_GPT_arithmetic_sequence_example_l1214_121422

theorem arithmetic_sequence_example (a : ℕ → ℝ) (d : ℝ) (h_arith : ∀ n, a (n + 1) = a n + d)
  (h2 : a 2 = 2) (h14 : a 14 = 18) : a 8 = 10 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_example_l1214_121422


namespace NUMINAMATH_GPT_expand_product_l1214_121482

theorem expand_product (x : ℝ) : (x^3 + 3) * (x^3 + 4) = x^6 + 7 * x^3 + 12 := 
  sorry

end NUMINAMATH_GPT_expand_product_l1214_121482


namespace NUMINAMATH_GPT_cheryl_material_used_l1214_121450

theorem cheryl_material_used :
  let material1 := (4 / 19 : ℚ)
  let material2 := (2 / 13 : ℚ)
  let bought := material1 + material2
  let leftover := (4 / 26 : ℚ)
  let used := bought - leftover
  used = (52 / 247 : ℚ) :=
by
  let material1 := (4 / 19 : ℚ)
  let material2 := (2 / 13 : ℚ)
  let bought := material1 + material2
  let leftover := (4 / 26 : ℚ)
  let used := bought - leftover
  have : used = (52 / 247 : ℚ) := sorry
  exact this

end NUMINAMATH_GPT_cheryl_material_used_l1214_121450


namespace NUMINAMATH_GPT_three_consecutive_odds_l1214_121489

theorem three_consecutive_odds (x : ℤ) (h3 : x + 4 = 133) : 
  x + (x + 4) = 3 * (x + 2) - 131 := 
by {
  sorry
}

end NUMINAMATH_GPT_three_consecutive_odds_l1214_121489


namespace NUMINAMATH_GPT_isosceles_triangle_side_length_l1214_121431

theorem isosceles_triangle_side_length (base : ℝ) (area : ℝ) (congruent_side : ℝ) 
  (h_base : base = 30) (h_area : area = 60) : congruent_side = Real.sqrt 241 :=
by 
  sorry

end NUMINAMATH_GPT_isosceles_triangle_side_length_l1214_121431


namespace NUMINAMATH_GPT_neg_p_is_necessary_but_not_sufficient_for_neg_q_l1214_121496

variables (p q : Prop)

-- Given conditions: (p → q) and ¬(q → p)
theorem neg_p_is_necessary_but_not_sufficient_for_neg_q
  (h1 : p → q)
  (h2 : ¬ (q → p)) :
  (¬ p → ¬ q) ∧ ¬ (¬ p ↔ ¬ q) :=
sorry

end NUMINAMATH_GPT_neg_p_is_necessary_but_not_sufficient_for_neg_q_l1214_121496


namespace NUMINAMATH_GPT_additional_hours_to_travel_l1214_121472

theorem additional_hours_to_travel (distance1 time1 rate distance2 : ℝ)
  (H1 : distance1 = 360)
  (H2 : time1 = 3)
  (H3 : rate = distance1 / time1)
  (H4 : distance2 = 240)
  :
  distance2 / rate = 2 := 
sorry

end NUMINAMATH_GPT_additional_hours_to_travel_l1214_121472


namespace NUMINAMATH_GPT_radius_ratio_ge_sqrt2plus1_l1214_121448

theorem radius_ratio_ge_sqrt2plus1 (r R a h : ℝ) (h1 : 2 * a ≠ 0) (h2 : h ≠ 0) 
  (hr : r = a * h / (a + Real.sqrt (a ^ 2 + h ^ 2)))
  (hR : R = (2 * a ^ 2 + h ^ 2) / (2 * h)) : 
  R / r ≥ 1 + Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_radius_ratio_ge_sqrt2plus1_l1214_121448


namespace NUMINAMATH_GPT_ratio_of_part_to_whole_l1214_121435

theorem ratio_of_part_to_whole (N : ℝ) (P : ℝ) (h1 : (1/4) * (2/5) * N = 17) (h2 : 0.40 * N = 204) :
  P = (2/5) * N → P / N = 2 / 5 :=
by
  intro h3
  sorry

end NUMINAMATH_GPT_ratio_of_part_to_whole_l1214_121435


namespace NUMINAMATH_GPT_solve_inequality_inequality_proof_l1214_121406

-- Problem 1: Solve the inequality |2x+1| - |x-4| > 2
theorem solve_inequality (x : ℝ) :
  (|2 * x + 1| - |x - 4| > 2) ↔ (x < -7 ∨ x > (5/3)) :=
sorry

-- Problem 2: Prove the inequality given a > 0 and b > 0
theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a / Real.sqrt b + b / Real.sqrt a) ≥ (Real.sqrt a + Real.sqrt b) :=
sorry

end NUMINAMATH_GPT_solve_inequality_inequality_proof_l1214_121406


namespace NUMINAMATH_GPT_class_overall_score_l1214_121483

def max_score : ℝ := 100
def percentage_study : ℝ := 0.4
def percentage_hygiene : ℝ := 0.25
def percentage_discipline : ℝ := 0.25
def percentage_activity : ℝ := 0.1

def score_study : ℝ := 85
def score_hygiene : ℝ := 90
def score_discipline : ℝ := 80
def score_activity : ℝ := 75

theorem class_overall_score :
  (score_study * percentage_study) +
  (score_hygiene * percentage_hygiene) +
  (score_discipline * percentage_discipline) +
  (score_activity * percentage_activity) = 84 :=
  by sorry

end NUMINAMATH_GPT_class_overall_score_l1214_121483


namespace NUMINAMATH_GPT_solve_for_x_l1214_121419

def star (a b : ℝ) : ℝ := 3 * a - b

theorem solve_for_x :
  ∃ x : ℝ, star 2 (star 5 x) = 1 ∧ x = 10 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1214_121419


namespace NUMINAMATH_GPT_profit_calculation_l1214_121485

theorem profit_calculation (cost_price_per_card_yuan : ℚ) (total_sales_yuan : ℚ)
  (n : ℕ) (sales_price_per_card_yuan : ℚ)
  (h1 : cost_price_per_card_yuan = 0.21)
  (h2 : total_sales_yuan = 14.57)
  (h3 : total_sales_yuan = n * sales_price_per_card_yuan)
  (h4 : sales_price_per_card_yuan ≤ 2 * cost_price_per_card_yuan) :
  (total_sales_yuan - n * cost_price_per_card_yuan = 4.7) :=
by
  sorry

end NUMINAMATH_GPT_profit_calculation_l1214_121485


namespace NUMINAMATH_GPT_minimum_value_w_l1214_121476

theorem minimum_value_w : 
  ∀ x y : ℝ, ∃ (w : ℝ), w = 2 * x^2 + 3 * y^2 + 8 * x - 5 * y + 30 → w ≥ 26.25 :=
by
  intro x y
  use 2 * x^2 + 3 * y^2 + 8 * x - 5 * y + 30
  sorry

end NUMINAMATH_GPT_minimum_value_w_l1214_121476


namespace NUMINAMATH_GPT_sum_of_edges_l1214_121492

theorem sum_of_edges (n : ℕ) (total_length large_edge small_edge : ℤ) : 
  n = 27 → 
  total_length = 828 → -- convert to millimeters
  large_edge = total_length / 12 → 
  small_edge = large_edge / 3 → 
  (large_edge + small_edge) / 10 = 92 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sum_of_edges_l1214_121492


namespace NUMINAMATH_GPT_unit_prices_minimize_cost_l1214_121418

theorem unit_prices (x y : ℕ) (h1 : x + 2 * y = 40) (h2 : 2 * x + 3 * y = 70) :
  x = 20 ∧ y = 10 :=
by {
  sorry -- proof would go here
}

theorem minimize_cost (total_pieces : ℕ) (cost_A cost_B : ℕ) 
  (total_cost : ℕ → ℕ)
  (h3 : total_pieces = 60) 
  (h4 : ∀ m, cost_A * m + cost_B * (total_pieces - m) = total_cost m) 
  (h5 : ∀ m, cost_A * m + cost_B * (total_pieces - m) ≥ 800) 
  (h6 : ∀ m, m ≥ (total_pieces - m) / 2) :
  total_cost 20 = 800 :=
by {
  sorry -- proof would go here
}

end NUMINAMATH_GPT_unit_prices_minimize_cost_l1214_121418


namespace NUMINAMATH_GPT_ratio_hours_per_day_l1214_121470

theorem ratio_hours_per_day 
  (h₁ : ∀ h : ℕ, h * 30 = 1200 + (h - 40) * 45 → 40 ≤ h ∧ 6 * 3 ≤ 40)
  (h₂ : 6 * 3 + (x - 6 * 3) / 2 = 24)
  (h₃ : x = 1290) :
  (24 / 2) / 6 = 2 := 
by
  sorry

end NUMINAMATH_GPT_ratio_hours_per_day_l1214_121470


namespace NUMINAMATH_GPT_dorothy_money_left_l1214_121474

def annual_income : ℝ := 60000
def tax_rate : ℝ := 0.18
def tax_amount : ℝ := annual_income * tax_rate
def money_left : ℝ := annual_income - tax_amount

theorem dorothy_money_left : money_left = 49200 := 
by
  sorry

end NUMINAMATH_GPT_dorothy_money_left_l1214_121474


namespace NUMINAMATH_GPT_new_cost_percentage_l1214_121402

variables (t c a x : ℝ) (n : ℕ)

def original_cost (t c a x : ℝ) (n : ℕ) : ℝ :=
  t * c * (a * x) ^ n

def new_cost (t c a x : ℝ) (n : ℕ) : ℝ :=
  t * (2 * c) * ((2 * a) * x) ^ (n + 2)

theorem new_cost_percentage (t c a x : ℝ) (n : ℕ) :
  new_cost t c a x n = 2^(n+1) * original_cost t c a x n * x^2 :=
by
  sorry

end NUMINAMATH_GPT_new_cost_percentage_l1214_121402


namespace NUMINAMATH_GPT_lines_connecting_intersections_l1214_121445

def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

theorem lines_connecting_intersections (n : ℕ) (h : n ≥ 2) :
  let N := binomial n 2
  binomial N 2 = (n * n * (n - 1) * (n - 1) - 2 * n * (n - 1)) / 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_lines_connecting_intersections_l1214_121445


namespace NUMINAMATH_GPT_wrapping_paper_area_l1214_121405

theorem wrapping_paper_area (length width : ℕ) (h1 : width = 6) (h2 : 2 * (length + width) = 28) : length * width = 48 :=
by
  sorry

end NUMINAMATH_GPT_wrapping_paper_area_l1214_121405


namespace NUMINAMATH_GPT_Jacob_eats_more_calories_than_planned_l1214_121455

theorem Jacob_eats_more_calories_than_planned 
  (planned_calories : ℕ) (actual_calories : ℕ)
  (h1 : planned_calories < 1800) 
  (h2 : actual_calories = 400 + 900 + 1100)
  : actual_calories - planned_calories = 600 := by
  sorry

end NUMINAMATH_GPT_Jacob_eats_more_calories_than_planned_l1214_121455
