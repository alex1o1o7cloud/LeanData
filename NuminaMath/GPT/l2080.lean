import Mathlib

namespace NUMINAMATH_GPT_ride_cost_l2080_208009

theorem ride_cost (joe_age_over_18 : Prop)
                   (joe_brother_age : Nat)
                   (joe_entrance_fee : ℝ)
                   (brother_entrance_fee : ℝ)
                   (total_spending : ℝ)
                   (rides_per_person : Nat)
                   (total_persons : Nat)
                   (total_entrance_fee : ℝ)
                   (amount_spent_on_rides : ℝ)
                   (total_rides : Nat) :
  joe_entrance_fee = 6 →
  brother_entrance_fee = 5 →
  total_spending = 20.5 →
  rides_per_person = 3 →
  total_persons = 3 →
  total_entrance_fee = 16 →
  amount_spent_on_rides = (total_spending - total_entrance_fee) →
  total_rides = (rides_per_person * total_persons) →
  (amount_spent_on_rides / total_rides) = 0.50 :=
by
  sorry

end NUMINAMATH_GPT_ride_cost_l2080_208009


namespace NUMINAMATH_GPT_no_solution_exists_l2080_208012

open Nat

theorem no_solution_exists : ¬ ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 2 ^ x + 3 ^ y - 5 ^ z = 2 * 11 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_exists_l2080_208012


namespace NUMINAMATH_GPT_sets_relationship_l2080_208008

def set_M : Set ℝ := {x | x^2 - 2 * x > 0}
def set_N : Set ℝ := {x | x > 3}

theorem sets_relationship : set_M ∩ set_N = set_N := by
  sorry

end NUMINAMATH_GPT_sets_relationship_l2080_208008


namespace NUMINAMATH_GPT_find_abc_solutions_l2080_208028

theorem find_abc_solutions :
  ∀ (a b c : ℕ),
    (2^(a) * 3^(b) = 7^(c) - 1) ↔
    ((a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 4 ∧ b = 1 ∧ c = 2)) :=
by
  sorry

end NUMINAMATH_GPT_find_abc_solutions_l2080_208028


namespace NUMINAMATH_GPT_abc_greater_than_n_l2080_208030

theorem abc_greater_than_n
  (a b c n : ℕ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : 1 < n)
  (h5 : a ^ n + b ^ n = c ^ n) :
  a > n ∧ b > n ∧ c > n :=
sorry

end NUMINAMATH_GPT_abc_greater_than_n_l2080_208030


namespace NUMINAMATH_GPT_inequality_add_six_l2080_208006

theorem inequality_add_six (x y : ℝ) (h : x < y) : x + 6 < y + 6 :=
sorry

end NUMINAMATH_GPT_inequality_add_six_l2080_208006


namespace NUMINAMATH_GPT_james_older_brother_age_l2080_208029

def johnAge : ℕ := 39

def ageCondition (johnAge : ℕ) (jamesAgeIn6 : ℕ) : Prop :=
  johnAge - 3 = 2 * jamesAgeIn6

def jamesOlderBrother (james : ℕ) : ℕ :=
  james + 4

theorem james_older_brother_age (johnAge jamesOlderBrotherAge : ℕ) (james : ℕ) :
  johnAge = 39 →
  (johnAge - 3 = 2 * (james + 6)) →
  jamesOlderBrotherAge = jamesOlderBrother james →
  jamesOlderBrotherAge = 16 :=
by
  sorry

end NUMINAMATH_GPT_james_older_brother_age_l2080_208029


namespace NUMINAMATH_GPT_final_price_correct_l2080_208065

open BigOperators

-- Define the constants used in the problem
def original_price : ℝ := 500
def first_discount : ℝ := 0.25
def second_discount : ℝ := 0.10
def state_tax : ℝ := 0.05

-- Define the calculation steps
def price_after_first_discount : ℝ := original_price * (1 - first_discount)
def price_after_second_discount : ℝ := price_after_first_discount * (1 - second_discount)
def final_price : ℝ := price_after_second_discount * (1 + state_tax)

-- Prove that the final price is 354.375
theorem final_price_correct : final_price = 354.375 :=
by
  sorry

end NUMINAMATH_GPT_final_price_correct_l2080_208065


namespace NUMINAMATH_GPT_polynomial_at_one_l2080_208023

def f (x : ℝ) : ℝ := x^4 - 7*x^3 - 9*x^2 + 11*x + 7

theorem polynomial_at_one :
  f 1 = 3 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_at_one_l2080_208023


namespace NUMINAMATH_GPT_calculate_speed_l2080_208058

theorem calculate_speed :
  ∀ (distance_ft time_sec miles_per_ft hours_per_sec approx_speed_mph : ℚ),
  distance_ft = 200 →
  time_sec = 2 →
  miles_per_ft = 1 / 5280 →
  hours_per_sec = 1 / 3600 →
  approx_speed_mph = 68.1818181818 →
  (distance_ft * miles_per_ft) / (time_sec * hours_per_sec) = approx_speed_mph :=
by
  intros distance_ft time_sec miles_per_ft hours_per_sec approx_speed_mph
  intro h_distance_eq h_time_eq h_miles_eq h_hours_eq h_speed_eq
  sorry

end NUMINAMATH_GPT_calculate_speed_l2080_208058


namespace NUMINAMATH_GPT_find_pairs_l2080_208078

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b

theorem find_pairs {a b : ℝ} :
  (0 < b) → (b ≤ 1) → (0 < a) → (a < 1) → (2 * a + b ≤ 2) →
  (∀ x y : ℝ, f a b (x * y) + f a b (x + y) ≥ f a b x * f a b y) :=
by
  intros h_b_gt_zero h_b_le_one h_a_gt_zero h_a_lt_one h_2a_b_le_2
  sorry

end NUMINAMATH_GPT_find_pairs_l2080_208078


namespace NUMINAMATH_GPT_num_true_propositions_eq_two_l2080_208010

open Classical

theorem num_true_propositions_eq_two (p q : Prop) :
  (if (p ∧ q) then 1 else 0) + (if (p ∨ q) then 1 else 0) + (if (¬p) then 1 else 0) + (if (¬q) then 1 else 0) = 2 :=
by sorry

end NUMINAMATH_GPT_num_true_propositions_eq_two_l2080_208010


namespace NUMINAMATH_GPT_sqrt_one_sixty_four_l2080_208057

theorem sqrt_one_sixty_four : Real.sqrt (1 / 64) = 1 / 8 :=
sorry

end NUMINAMATH_GPT_sqrt_one_sixty_four_l2080_208057


namespace NUMINAMATH_GPT_proof_problem_l2080_208081

noncomputable def log2 (n : ℝ) : ℝ := Real.log n / Real.log 2

theorem proof_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/2 * log2 x + 1/3 * log2 y = 1) : x^3 * y^2 = 64 := 
sorry 

end NUMINAMATH_GPT_proof_problem_l2080_208081


namespace NUMINAMATH_GPT_gcd_g_y_l2080_208051

def g (y : ℕ) : ℕ := (3*y + 4) * (8*y + 3) * (14*y + 9) * (y + 17)

theorem gcd_g_y (y : ℕ) (h : y % 42522 = 0) : Nat.gcd (g y) y = 102 := by
  sorry

end NUMINAMATH_GPT_gcd_g_y_l2080_208051


namespace NUMINAMATH_GPT_number_of_a_values_l2080_208053

theorem number_of_a_values (a : ℝ) :
  (∃ x : ℝ, y = x + 2*a ∧ y = x^3 - 3*a*x + a^3) → a = 0 :=
by
  sorry

end NUMINAMATH_GPT_number_of_a_values_l2080_208053


namespace NUMINAMATH_GPT_parabola_directrix_l2080_208077

theorem parabola_directrix (a : ℝ) :
  (∃ y : ℝ, y = ax^2 ∧ y = -2) → a = 1/8 :=
by
  -- Solution steps are omitted.
  sorry

end NUMINAMATH_GPT_parabola_directrix_l2080_208077


namespace NUMINAMATH_GPT_primes_ge_3_are_4k_pm1_infinitely_many_primes_4k_minus1_l2080_208059

-- Part 1: Prove that every prime number >= 3 is of the form 4k-1 or 4k+1
theorem primes_ge_3_are_4k_pm1 (p : ℕ) (hp_prime: Nat.Prime p) (hp_ge_3: p ≥ 3) : 
  ∃ k : ℕ, p = 4 * k + 1 ∨ p = 4 * k - 1 :=
by
  sorry

-- Part 2: Prove that there are infinitely many primes of the form 4k-1
theorem infinitely_many_primes_4k_minus1 : 
  ∀ (n : ℕ), ∃ (p : ℕ), Nat.Prime p ∧ p = 4 * k - 1 ∧ p > n :=
by
  sorry

end NUMINAMATH_GPT_primes_ge_3_are_4k_pm1_infinitely_many_primes_4k_minus1_l2080_208059


namespace NUMINAMATH_GPT_smaller_square_area_percentage_l2080_208089

noncomputable def percent_area_of_smaller_square (side_length_larger_square : ℝ) : ℝ :=
  let diagonal_larger_square := side_length_larger_square * Real.sqrt 2
  let radius_circle := diagonal_larger_square / 2
  let x := (2 + 4 * (side_length_larger_square / 2)) / ((side_length_larger_square / 2) * 2) -- Simplified quadratic solution
  let side_length_smaller_square := side_length_larger_square * x
  let area_smaller_square := side_length_smaller_square ^ 2
  let area_larger_square := side_length_larger_square ^ 2
  (area_smaller_square / area_larger_square) * 100

-- Statement to show that under given conditions, the area of the smaller square is 4% of the larger square's area
theorem smaller_square_area_percentage :
  percent_area_of_smaller_square 4 = 4 := 
sorry

end NUMINAMATH_GPT_smaller_square_area_percentage_l2080_208089


namespace NUMINAMATH_GPT_infinite_fractions_2_over_odd_l2080_208061

theorem infinite_fractions_2_over_odd (a b : ℕ) (n : ℕ) : 
  (a = 2 → 2 * b + 1 ≠ 0) ∧ ((b = 2 * n + 1) → (2 + 2) / (2 * (2 * n + 1)) = 2 / (2 * n + 1)) ∧ (a / b = 2 / (2 * n + 1)) :=
by
  sorry

end NUMINAMATH_GPT_infinite_fractions_2_over_odd_l2080_208061


namespace NUMINAMATH_GPT_combined_cost_price_correct_l2080_208088

def face_value_A : ℝ := 100
def discount_A : ℝ := 0.02
def face_value_B : ℝ := 100
def premium_B : ℝ := 0.015
def brokerage : ℝ := 0.002

def purchase_price_A := face_value_A * (1 - discount_A)
def brokerage_fee_A := purchase_price_A * brokerage
def total_cost_price_A := purchase_price_A + brokerage_fee_A

def purchase_price_B := face_value_B * (1 + premium_B)
def brokerage_fee_B := purchase_price_B * brokerage
def total_cost_price_B := purchase_price_B + brokerage_fee_B

def combined_cost_price := total_cost_price_A + total_cost_price_B

theorem combined_cost_price_correct :
  combined_cost_price = 199.899 :=
by
  sorry

end NUMINAMATH_GPT_combined_cost_price_correct_l2080_208088


namespace NUMINAMATH_GPT_train_speed_is_36_kph_l2080_208003

noncomputable def speed_of_train (length_train length_bridge time_to_pass : ℕ) : ℕ :=
  let total_distance := length_train + length_bridge
  let speed_mps := total_distance / time_to_pass
  let speed_kph := speed_mps * 3600 / 1000
  speed_kph

theorem train_speed_is_36_kph :
  speed_of_train 360 140 50 = 36 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_is_36_kph_l2080_208003


namespace NUMINAMATH_GPT_factorization_correct_l2080_208005

theorem factorization_correct :
  ∀ (y : ℝ), (y^2 - 1 = (y + 1) * (y - 1)) :=
by
  intro y
  sorry

end NUMINAMATH_GPT_factorization_correct_l2080_208005


namespace NUMINAMATH_GPT_incorrect_statement_C_l2080_208021

noncomputable def f (a x : ℝ) : ℝ := x^2 * (Real.log x - a) + a

theorem incorrect_statement_C :
  ¬ (∀ a : ℝ, a > 0 → ∀ x : ℝ, x > 0 → f a x ≥ 0) := sorry

end NUMINAMATH_GPT_incorrect_statement_C_l2080_208021


namespace NUMINAMATH_GPT_typeA_selling_price_maximize_profit_l2080_208097

theorem typeA_selling_price (sales_last_year : ℝ) (sales_increase_rate : ℝ) (price_increase : ℝ) 
                            (cars_sold_last_year : ℝ) : 
                            (sales_last_year = 32000) ∧ (sales_increase_rate = 1.25) ∧ 
                            (price_increase = 400) ∧ 
                            (sales_last_year / cars_sold_last_year = (sales_last_year * sales_increase_rate) / (cars_sold_last_year + price_increase)) → 
                            (cars_sold_last_year = 1600) :=
by
  sorry

theorem maximize_profit (typeA_price : ℝ) (typeB_price : ℝ) (typeA_cost : ℝ) (typeB_cost : ℝ) 
                        (total_cars : ℕ) :
                        (typeA_price = 2000) ∧ (typeB_price = 2400) ∧ 
                        (typeA_cost = 1100) ∧ (typeB_cost = 1400) ∧ 
                        (total_cars = 50) ∧ 
                        (∀ m : ℕ, m ≤ 50 / 3) → 
                        ∃ m : ℕ, (m = 17) ∧ (50 - m * 2 ≤ 33) :=
by
  sorry

end NUMINAMATH_GPT_typeA_selling_price_maximize_profit_l2080_208097


namespace NUMINAMATH_GPT_snakes_hiding_l2080_208091

/-- The statement that given the total number of snakes and the number of snakes not hiding,
we can determine the number of snakes hiding. -/
theorem snakes_hiding (total_snakes : ℕ) (snakes_not_hiding : ℕ) (h1 : total_snakes = 95) (h2 : snakes_not_hiding = 31) :
  total_snakes - snakes_not_hiding = 64 :=
by {
  sorry
}

end NUMINAMATH_GPT_snakes_hiding_l2080_208091


namespace NUMINAMATH_GPT_initial_pages_l2080_208011

variable (P : ℕ)
variable (h : 20 * P - 20 = 220)

theorem initial_pages (h : 20 * P - 20 = 220) : P = 12 := by
  sorry

end NUMINAMATH_GPT_initial_pages_l2080_208011


namespace NUMINAMATH_GPT_percentage_increase_direct_proportionality_l2080_208062

variable (x y k q : ℝ)
variable (h1 : x = k * y)
variable (h2 : x' = x * (1 + q / 100))

theorem percentage_increase_direct_proportionality :
  ∃ q_percent : ℝ, y' = y * (1 + q_percent / 100) ∧ q_percent = q := sorry

end NUMINAMATH_GPT_percentage_increase_direct_proportionality_l2080_208062


namespace NUMINAMATH_GPT_math_problem_l2080_208019

variable {f : ℝ → ℝ}
variable {g : ℝ → ℝ}

noncomputable def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
noncomputable def odd_function (g : ℝ → ℝ) := ∀ x : ℝ, g x = -g (-x)

theorem math_problem
  (hf_even : even_function f)
  (hf_0 : f 0 = 1)
  (hg_odd : odd_function g)
  (hgf : ∀ x : ℝ, g x = f (x - 1)) :
  f 2011 + f 2012 + f 2013 = 1 := sorry

end NUMINAMATH_GPT_math_problem_l2080_208019


namespace NUMINAMATH_GPT_crystal_final_segment_distance_l2080_208039

theorem crystal_final_segment_distance :
  let north_distance := 2
  let northwest_distance := 2
  let southwest_distance := 2
  let northwest_component := northwest_distance / Real.sqrt 2 -- as nx, ny
  let southwest_component := southwest_distance / Real.sqrt 2 -- as sx, sy
  let net_north := north_distance + northwest_component - southwest_component
  let net_west := northwest_component + southwest_component
  Real.sqrt (net_north^2 + net_west^2) = 2 * Real.sqrt 3 :=
by
  let north_distance := 2
  let northwest_distance := 2
  let southwest_distance := 2
  let northwest_component := northwest_distance / Real.sqrt 2
  let southwest_component := southwest_distance / Real.sqrt 2
  let net_north := north_distance + northwest_component - southwest_component
  let net_west := northwest_component + southwest_component
  exact sorry

end NUMINAMATH_GPT_crystal_final_segment_distance_l2080_208039


namespace NUMINAMATH_GPT_geometric_sequence_arithmetic_progression_l2080_208031

open Nat

/--
Given a geometric sequence \( \{a_n\} \) where \( a_1 = 1 \) and the sequence terms
\( 4a_1 \), \( 2a_2 \), \( a_3 \) form an arithmetic progression, prove that
the common ratio \( q = 2 \) and the sum of the first four terms \( S_4 = 15 \).
-/
theorem geometric_sequence_arithmetic_progression (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h₀ : a 1 = 1)
    (h₁ : ∀ n, S n = (1 - q^n) / (1 - q)) 
    (h₂ : ∀ k n, a (k + n) = a k * q ^ n) 
    (h₃ : 4 * a 1 + a 3 = 4 * a 2) :
  q = 2 ∧ S 4 = 15 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_arithmetic_progression_l2080_208031


namespace NUMINAMATH_GPT_remaining_regular_toenails_l2080_208045

def big_toenail_space := 2
def total_capacity := 100
def big_toenails_count := 20
def regular_toenails_count := 40

theorem remaining_regular_toenails : 
  total_capacity - (big_toenails_count * big_toenail_space + regular_toenails_count) = 20 := by
  sorry

end NUMINAMATH_GPT_remaining_regular_toenails_l2080_208045


namespace NUMINAMATH_GPT_real_solutions_of_equation_l2080_208026

theorem real_solutions_of_equation :
  (∃! x : ℝ, (5 * x) / (x^2 + 2 * x + 4) + (6 * x) / (x^2 - 6 * x + 4) = -4 / 3) :=
sorry

end NUMINAMATH_GPT_real_solutions_of_equation_l2080_208026


namespace NUMINAMATH_GPT_smallest_nine_digit_times_smallest_seven_digit_l2080_208071

theorem smallest_nine_digit_times_smallest_seven_digit :
  let smallest_nine_digit := 100000000
  let smallest_seven_digit := 1000000
  smallest_nine_digit = 100 * smallest_seven_digit :=
by
  sorry

end NUMINAMATH_GPT_smallest_nine_digit_times_smallest_seven_digit_l2080_208071


namespace NUMINAMATH_GPT_probability_multiple_of_100_is_zero_l2080_208034

def singleDigitMultiplesOf5 : Set ℕ := {5}
def primeNumbersLessThan50 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
def isMultipleOf100 (n : ℕ) : Prop := 100 ∣ n

theorem probability_multiple_of_100_is_zero :
  (∀ m ∈ singleDigitMultiplesOf5, ∀ p ∈ primeNumbersLessThan50, ¬ isMultipleOf100 (m * p)) →
  r = 0 :=
sorry

end NUMINAMATH_GPT_probability_multiple_of_100_is_zero_l2080_208034


namespace NUMINAMATH_GPT_cost_per_meter_of_fencing_l2080_208042

/-- A rectangular farm has area 1200 m², a short side of 30 m, and total job cost 1560 Rs.
    Prove that the cost of fencing per meter is 13 Rs. -/
theorem cost_per_meter_of_fencing
  (A : ℝ := 1200)
  (W : ℝ := 30)
  (job_cost : ℝ := 1560)
  (L : ℝ := A / W)
  (D : ℝ := Real.sqrt (L^2 + W^2))
  (total_length : ℝ := L + W + D) :
  job_cost / total_length = 13 := 
sorry

end NUMINAMATH_GPT_cost_per_meter_of_fencing_l2080_208042


namespace NUMINAMATH_GPT_sqrt_mixed_number_simplification_l2080_208094

theorem sqrt_mixed_number_simplification :
  Real.sqrt (7 + 9 / 16) = 11 / 4 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_mixed_number_simplification_l2080_208094


namespace NUMINAMATH_GPT_batsman_average_increase_l2080_208073

theorem batsman_average_increase (A : ℕ) (H1 : 16 * A + 85 = 17 * (A + 3)) : A + 3 = 37 :=
by {
  sorry
}

end NUMINAMATH_GPT_batsman_average_increase_l2080_208073


namespace NUMINAMATH_GPT_intersection_is_correct_complement_is_correct_l2080_208095

open Set

variable {U : Set ℝ} (A B : Set ℝ)

-- Define the universal set U
def U_def : Set ℝ := { x | 1 < x ∧ x < 7 }

-- Define set A
def A_def : Set ℝ := { x | 2 ≤ x ∧ x < 5 }

-- Define set B using the simplified condition from the inequality
def B_def : Set ℝ := { x | x ≥ 3 }

-- Proof statement that A ∩ B is as specified
theorem intersection_is_correct :
  (A_def ∩ B_def) = { x : ℝ | 3 ≤ x ∧ x < 5 } := by
  sorry

-- Proof statement for the complement of A relative to U
theorem complement_is_correct :
  (U_def \ A_def) = { x : ℝ | (1 < x ∧ x < 2) ∨ (5 ≤ x ∧ x < 7) } := by
  sorry

end NUMINAMATH_GPT_intersection_is_correct_complement_is_correct_l2080_208095


namespace NUMINAMATH_GPT_xy_range_l2080_208024

theorem xy_range (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
    (h_eqn : x + 3 * y + 2 / x + 4 / y = 10) :
    1 ≤ x * y ∧ x * y ≤ 8 / 3 :=
  sorry

end NUMINAMATH_GPT_xy_range_l2080_208024


namespace NUMINAMATH_GPT_original_price_l2080_208018

theorem original_price (P : ℝ) (h : P * 0.5 = 1200) : P = 2400 := 
by
  sorry

end NUMINAMATH_GPT_original_price_l2080_208018


namespace NUMINAMATH_GPT_gcd_9240_12240_33720_l2080_208032

theorem gcd_9240_12240_33720 : Nat.gcd (Nat.gcd 9240 12240) 33720 = 240 := by
  sorry

end NUMINAMATH_GPT_gcd_9240_12240_33720_l2080_208032


namespace NUMINAMATH_GPT_percentage_of_left_handed_women_l2080_208027

variable (x y : Nat) (h_ratio_rh_lh : 3 * x = 1 * x)
variable (h_ratio_men_women : 3 * y = 2 * y)
variable (h_rh_men_max : True)

theorem percentage_of_left_handed_women :
  (x / (4 * x)) * 100 = 25 :=
by sorry

end NUMINAMATH_GPT_percentage_of_left_handed_women_l2080_208027


namespace NUMINAMATH_GPT_first_problem_solution_set_second_problem_a_range_l2080_208064

-- Define the function f(x) = |2x - a| + |x - 1|
def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + abs (x - 1)

-- First problem: When a = 3, the solution set of the inequality f(x) ≥ 2
theorem first_problem_solution_set (x : ℝ) : (f x 3 ≥ 2) ↔ (x ≤ 2 / 3 ∨ x ≥ 2) :=
by sorry

-- Second problem: If f(x) ≥ 5 - x for ∀ x ∈ ℝ, find the range of the real number a
theorem second_problem_a_range (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 5 - x) ↔ (6 ≤ a) :=
by sorry

end NUMINAMATH_GPT_first_problem_solution_set_second_problem_a_range_l2080_208064


namespace NUMINAMATH_GPT_total_selling_price_l2080_208086

theorem total_selling_price
  (meters_cloth : ℕ)
  (profit_per_meter : ℕ)
  (cost_price_per_meter : ℕ)
  (selling_price_per_meter : ℕ := cost_price_per_meter + profit_per_meter)
  (total_selling_price : ℕ := selling_price_per_meter * meters_cloth)
  (h_mc : meters_cloth = 75)
  (h_ppm : profit_per_meter = 15)
  (h_cppm : cost_price_per_meter = 51)
  (h_spm : selling_price_per_meter = 66)
  (h_tsp : total_selling_price = 4950) : 
  total_selling_price = 4950 := 
  by
  -- Skipping the actual proof
  trivial

end NUMINAMATH_GPT_total_selling_price_l2080_208086


namespace NUMINAMATH_GPT_chandra_pairings_l2080_208068

variable (bowls : ℕ) (glasses : ℕ)

theorem chandra_pairings : 
  bowls = 5 → 
  glasses = 4 → 
  bowls * glasses = 20 :=
by intros; 
    sorry

end NUMINAMATH_GPT_chandra_pairings_l2080_208068


namespace NUMINAMATH_GPT_num_children_l2080_208066

-- Defining the conditions
def num_adults : Nat := 10
def price_adult_ticket : Nat := 8
def total_bill : Nat := 124
def price_child_ticket : Nat := 4

-- Statement to prove: Number of children
theorem num_children (num_adults : Nat) (price_adult_ticket : Nat) (total_bill : Nat) (price_child_ticket : Nat) : Nat :=
  let cost_adults := num_adults * price_adult_ticket
  let cost_child := total_bill - cost_adults
  cost_child / price_child_ticket

example : num_children 10 8 124 4 = 11 := sorry

end NUMINAMATH_GPT_num_children_l2080_208066


namespace NUMINAMATH_GPT_a2_add_a8_l2080_208076

variable (a : ℕ → ℝ) -- a_n is an arithmetic sequence
variable (d : ℝ) -- common difference

-- Condition stating that a_n is an arithmetic sequence with common difference d
axiom arithmetic_sequence : ∀ n, a (n + 1) = a n + d

-- Given condition a_3 + a_4 + a_5 + a_6 + a_7 = 450
axiom given_condition : a 3 + a 4 + a 5 + a 6 + a 7 = 450

theorem a2_add_a8 : a 2 + a 8 = 180 :=
by
  sorry

end NUMINAMATH_GPT_a2_add_a8_l2080_208076


namespace NUMINAMATH_GPT_will_can_buy_correct_amount_of_toys_l2080_208035

-- Define the initial conditions as constants
def initial_amount : Int := 57
def amount_spent : Int := 27
def cost_per_toy : Int := 6

-- Lemma stating the problem to prove.
theorem will_can_buy_correct_amount_of_toys : (initial_amount - amount_spent) / cost_per_toy = 5 :=
by
  sorry

end NUMINAMATH_GPT_will_can_buy_correct_amount_of_toys_l2080_208035


namespace NUMINAMATH_GPT_minimum_of_quadratic_l2080_208074

theorem minimum_of_quadratic : ∀ x : ℝ, 1 ≤ x^2 - 6 * x + 10 :=
by
  intro x
  have h : x^2 - 6 * x + 10 = (x - 3)^2 + 1 := by ring
  rw [h]
  have h_nonneg : (x - 3)^2 ≥ 0 := by apply sq_nonneg
  linarith

end NUMINAMATH_GPT_minimum_of_quadratic_l2080_208074


namespace NUMINAMATH_GPT_counting_unit_difference_l2080_208036

-- Definitions based on conditions
def magnitude_equality : Prop := 75 = 75.0
def counting_unit_75 : Nat := 1
def counting_unit_75_0 : Nat := 1 / 10

-- Proof problem stating that 75 and 75.0 do not have the same counting units.
theorem counting_unit_difference : 
  ¬ (counting_unit_75 = counting_unit_75_0) :=
by sorry

end NUMINAMATH_GPT_counting_unit_difference_l2080_208036


namespace NUMINAMATH_GPT_solve_for_m_l2080_208047

theorem solve_for_m (m : ℝ) (f g : ℝ → ℝ) :
  (∀ x : ℝ, f x = x^2 - 2 * x + m) →
  (∀ x : ℝ, g x = x^2 - 2 * x + 9 * m) →
  f 2 = 2 * g 2 →
  m = 0 :=
  by
    intros hf hg hs
    sorry

end NUMINAMATH_GPT_solve_for_m_l2080_208047


namespace NUMINAMATH_GPT_no_duplicate_among_expressions_l2080_208015

theorem no_duplicate_among_expressions
  (N a1 a2 b1 b2 c1 c2 d1 d2 : ℕ)
  (ha : a1 = x^2)
  (hb : b1 = y^3)
  (hc : c1 = z^5)
  (hd : d1 = w^7)
  (ha2 : a2 = m^2)
  (hb2 : b2 = n^3)
  (hc2 : c2 = p^5)
  (hd2 : d2 = q^7)
  (h1 : N = a1 - a2)
  (h2 : N = b1 - b2)
  (h3 : N = c1 - c2)
  (h4 : N = d1 - d2) :
  ¬ (a1 = b1 ∨ a1 = c1 ∨ a1 = d1 ∨ b1 = c1 ∨ b1 = d1 ∨ c1 = d1) :=
by
  -- Begin proof here
  sorry

end NUMINAMATH_GPT_no_duplicate_among_expressions_l2080_208015


namespace NUMINAMATH_GPT_x1x2_lt_one_l2080_208069

noncomputable section

open Real

def f (a : ℝ) (x : ℝ) : ℝ :=
  |exp x - exp 1| + exp x + a * x

theorem x1x2_lt_one (a x1 x2 : ℝ) 
  (ha : a < -exp 1) 
  (hzero1 : f a x1 = 0) 
  (hzero2 : f a x2 = 0) 
  (h_order : x1 < x2) : x1 * x2 < 1 := 
sorry

end NUMINAMATH_GPT_x1x2_lt_one_l2080_208069


namespace NUMINAMATH_GPT_cars_meeting_time_l2080_208093

def problem_statement (V_A V_B V_C V_D : ℝ) :=
  (V_A ≠ V_B) ∧ (V_A ≠ V_C) ∧ (V_A ≠ V_D) ∧
  (V_B ≠ V_C) ∧ (V_B ≠ V_D) ∧ (V_C ≠ V_D) ∧
  (V_A + V_C = V_B + V_D) ∧
  (53 * (V_A - V_B) / 46 = 7) ∧
  (53 * (V_D - V_C) / 46 = 7)

theorem cars_meeting_time (V_A V_B V_C V_D : ℝ) (h : problem_statement V_A V_B V_C V_D) : 
  ∃ t : ℝ, t = 53 := 
sorry

end NUMINAMATH_GPT_cars_meeting_time_l2080_208093


namespace NUMINAMATH_GPT_tom_driving_speed_l2080_208060

theorem tom_driving_speed
  (v : ℝ)
  (hKarenSpeed : 60 = 60) -- Karen drives at an average speed of 60 mph
  (hKarenLateStart: 4 / 60 = 1 / 15) -- Karen starts 4 minutes late, which is 1/15 hours
  (hTomDistance : 24 = 24) -- Tom drives 24 miles before Karen wins the bet
  (hTimeEquation: 24 / v = 8 / 15): -- The equation derived from given conditions
  v = 45 := 
by
  sorry

end NUMINAMATH_GPT_tom_driving_speed_l2080_208060


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l2080_208082

noncomputable def a : ℝ := Real.logb 0.5 0.2
noncomputable def b : ℝ := Real.logb 2 0.2
noncomputable def c : ℝ := Real.exp (0.2 * Real.log 2)

theorem relationship_among_a_b_c : b < c ∧ c < a :=
by
  sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l2080_208082


namespace NUMINAMATH_GPT_knights_on_red_chairs_l2080_208025

theorem knights_on_red_chairs (K L K_r L_b : ℕ) (h1: K + L = 20)
  (h2: K - K_r + L_b = 10) (h3: K_r + L - L_b = 10) (h4: K_r = L_b) : K_r = 5 := by
  sorry

end NUMINAMATH_GPT_knights_on_red_chairs_l2080_208025


namespace NUMINAMATH_GPT_true_statements_count_is_two_l2080_208085

def original_proposition (a : ℝ) : Prop :=
  a < 0 → ∃ x : ℝ, x^2 + x + a = 0

def contrapositive (a : ℝ) : Prop :=
  ¬ (∃ x : ℝ, x^2 + x + a = 0) → a ≥ 0

def converse (a : ℝ) : Prop :=
  (∃ x : ℝ, x^2 + x + a = 0) → a < 0

def negation (a : ℝ) : Prop :=
  a < 0 → ¬ ∃ x : ℝ, x^2 + x + a = 0

-- Prove that there are exactly 2 true statements among the four propositions: 
-- original_proposition, contrapositive, converse, and negation.

theorem true_statements_count_is_two : 
  ∀ (a : ℝ), original_proposition a ∧ contrapositive a ∧ ¬(converse a) ∧ ¬(negation a) → 
  (original_proposition a ∧ contrapositive a ∧ ¬(converse a) ∧ ¬(negation a)) ↔ (2 = 2) := 
by
  sorry

end NUMINAMATH_GPT_true_statements_count_is_two_l2080_208085


namespace NUMINAMATH_GPT_sum_of_medians_powers_l2080_208090

noncomputable def median_length_squared (a b c : ℝ) : ℝ :=
  (a^2 + b^2 - c^2) / 4

noncomputable def sum_of_fourth_powers_of_medians (a b c : ℝ) : ℝ :=
  let mAD := (median_length_squared a b c)^2
  let mBE := (median_length_squared b c a)^2
  let mCF := (median_length_squared c a b)^2
  mAD^2 + mBE^2 + mCF^2

theorem sum_of_medians_powers :
  sum_of_fourth_powers_of_medians 13 14 15 = 7644.25 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_medians_powers_l2080_208090


namespace NUMINAMATH_GPT_probability_no_neighbouring_same_color_l2080_208020

-- Given conditions
def red_beads : ℕ := 4
def white_beads : ℕ := 2
def blue_beads : ℕ := 2
def total_beads : ℕ := red_beads + white_beads + blue_beads

-- Total permutations
def total_orderings : ℕ := Nat.factorial total_beads / (Nat.factorial red_beads * Nat.factorial white_beads * Nat.factorial blue_beads)

-- Probability calculation proof
theorem probability_no_neighbouring_same_color : (30 / 420 : ℚ) = (1 / 14 : ℚ) :=
by
  -- proof steps
  sorry

end NUMINAMATH_GPT_probability_no_neighbouring_same_color_l2080_208020


namespace NUMINAMATH_GPT_trigonometric_identity_l2080_208083

theorem trigonometric_identity
  (θ : ℝ) 
  (h_tan : Real.tan θ = 3) :
  (1 - Real.cos θ) / (Real.sin θ) - (Real.sin θ) / (1 + (Real.cos θ)^2) = (11 * Real.sqrt 10 - 101) / 33 := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2080_208083


namespace NUMINAMATH_GPT_no_sphinx_tiling_l2080_208052

def equilateral_triangle_tiling_problem (side_length : ℕ) (pointing_up : ℕ) (pointing_down : ℕ) : Prop :=
  let total_triangles := side_length * side_length
  pointing_up + pointing_down = total_triangles ∧ 
  total_triangles = 36 ∧
  pointing_down = 1 + 2 + 3 + 4 + 5 ∧
  pointing_up = 1 + 2 + 3 + 4 + 5 + 6 ∧
  (pointing_up % 2 = 1) ∧
  (pointing_down % 2 = 1) ∧
  (2 * pointing_up + 4 * pointing_down ≠ total_triangles ∧ 4 * pointing_up + 2 * pointing_down ≠ total_triangles)

theorem no_sphinx_tiling : ¬equilateral_triangle_tiling_problem 6 21 15 :=
by
  sorry

end NUMINAMATH_GPT_no_sphinx_tiling_l2080_208052


namespace NUMINAMATH_GPT_solution_set_inequality_l2080_208033

noncomputable def f (x : ℝ) : ℝ := x * (1 - 3 * x)

theorem solution_set_inequality : {x : ℝ | f x > 0} = { x | (0 < x) ∧ (x < 1/3) } := by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l2080_208033


namespace NUMINAMATH_GPT_construction_costs_correct_l2080_208063

structure ConstructionCosts where
  landCostPerSqMeter : ℕ
  brickCostPerThousand : ℕ
  tileCostPerTile : ℕ
  landRequired : ℕ
  bricksRequired : ℕ
  tilesRequired : ℕ

noncomputable def totalConstructionCost (cc : ConstructionCosts) : ℕ :=
  let landCost := cc.landRequired * cc.landCostPerSqMeter
  let brickCost := (cc.bricksRequired / 1000) * cc.brickCostPerThousand
  let tileCost := cc.tilesRequired * cc.tileCostPerTile
  landCost + brickCost + tileCost

theorem construction_costs_correct (cc : ConstructionCosts)
  (h1 : cc.landCostPerSqMeter = 50)
  (h2 : cc.brickCostPerThousand = 100)
  (h3 : cc.tileCostPerTile = 10)
  (h4 : cc.landRequired = 2000)
  (h5 : cc.bricksRequired = 10000)
  (h6 : cc.tilesRequired = 500) :
  totalConstructionCost cc = 106000 := 
  by 
    sorry

end NUMINAMATH_GPT_construction_costs_correct_l2080_208063


namespace NUMINAMATH_GPT_sum_of_xy_l2080_208002

theorem sum_of_xy {x y : ℝ} (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := sorry

end NUMINAMATH_GPT_sum_of_xy_l2080_208002


namespace NUMINAMATH_GPT_sum_of_first_12_terms_geometric_sequence_l2080_208000

variable {α : Type*} [Field α]

def geometric_sequence (a : ℕ → α) : Prop :=
  ∃ r : α, ∀ n : ℕ, a (n + 1) = a n * r

noncomputable def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
  (Finset.range n).sum a

theorem sum_of_first_12_terms_geometric_sequence
  (a : ℕ → α)
  (h_geo : geometric_sequence a)
  (h_sum1 : sum_first_n_terms a 3 = 4)
  (h_sum2 : sum_first_n_terms a 6 - sum_first_n_terms a 3 = 8) :
  sum_first_n_terms a 12 = 60 := 
sorry

end NUMINAMATH_GPT_sum_of_first_12_terms_geometric_sequence_l2080_208000


namespace NUMINAMATH_GPT_complement_union_l2080_208080

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {1, 3, 4, 5}

theorem complement_union :
  (U \ A) ∪ (U \ B) = {1, 2, 3, 6} := 
by 
  sorry

end NUMINAMATH_GPT_complement_union_l2080_208080


namespace NUMINAMATH_GPT_area_difference_quarter_circles_l2080_208087

theorem area_difference_quarter_circles :
  let r1 := 28
  let r2 := 14
  let pi := (22 / 7)
  let quarter_area_big := (1 / 4) * pi * r1^2
  let quarter_area_small := (1 / 4) * pi * r2^2
  let rectangle_area := r1 * r2
  (quarter_area_big - (quarter_area_small + rectangle_area)) = 70 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_area_difference_quarter_circles_l2080_208087


namespace NUMINAMATH_GPT_find_cost_price_l2080_208013

theorem find_cost_price (C : ℝ) (SP : ℝ) (M : ℝ) (h1 : SP = 1.25 * C) (h2 : 0.90 * M = SP) (h3 : SP = 65.97) : 
  C = 52.776 :=
by
  sorry

end NUMINAMATH_GPT_find_cost_price_l2080_208013


namespace NUMINAMATH_GPT_find_a_equiv_l2080_208049

noncomputable def A (a : ℝ) : Set ℝ := {1, 3, a^2}
noncomputable def B (a : ℝ) : Set ℝ := {1, 2 + a}

theorem find_a_equiv (a : ℝ) (h : A a ∪ B a = A a) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_equiv_l2080_208049


namespace NUMINAMATH_GPT_distance_between_trees_l2080_208075

theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) (h1 : yard_length = 255) (h2 : num_trees = 18) : yard_length / (num_trees - 1) = 15 := by
  sorry

end NUMINAMATH_GPT_distance_between_trees_l2080_208075


namespace NUMINAMATH_GPT_janice_bottle_caps_l2080_208070

-- Define the conditions
def num_boxes : ℕ := 79
def caps_per_box : ℕ := 4

-- Define the question as a theorem to prove
theorem janice_bottle_caps : num_boxes * caps_per_box = 316 :=
by
  sorry

end NUMINAMATH_GPT_janice_bottle_caps_l2080_208070


namespace NUMINAMATH_GPT_minimum_unused_area_for_given_shapes_l2080_208048

def remaining_area (side_length : ℕ) (total_area used_area : ℕ) : ℕ :=
  total_area - used_area

theorem minimum_unused_area_for_given_shapes : (remaining_area 5 (5 * 5) (2 * 2 + 1 * 3 + 2 * 1) = 16) :=
by
  -- We skip the proof here, as instructed.
  sorry

end NUMINAMATH_GPT_minimum_unused_area_for_given_shapes_l2080_208048


namespace NUMINAMATH_GPT_average_speed_additional_hours_l2080_208084

theorem average_speed_additional_hours
  (time_first_part : ℝ) (speed_first_part : ℝ) (total_time : ℝ) (avg_speed_total : ℝ)
  (additional_hours : ℝ) (speed_additional_hours : ℝ) :
  time_first_part = 4 → speed_first_part = 35 → total_time = 24 → avg_speed_total = 50 →
  additional_hours = total_time - time_first_part →
  (time_first_part * speed_first_part + additional_hours * speed_additional_hours) / total_time = avg_speed_total →
  speed_additional_hours = 53 :=
by intros; sorry

end NUMINAMATH_GPT_average_speed_additional_hours_l2080_208084


namespace NUMINAMATH_GPT_solve_inequality_l2080_208001

theorem solve_inequality (x : ℝ) (h : |2 * x + 6| < 10) : -8 < x ∧ x < 2 :=
sorry

end NUMINAMATH_GPT_solve_inequality_l2080_208001


namespace NUMINAMATH_GPT_waiter_total_customers_l2080_208099

def numCustomers (T : ℕ) (totalTips : ℕ) (tipPerCustomer : ℕ) (numNoTipCustomers : ℕ) : ℕ :=
  T + numNoTipCustomers

theorem waiter_total_customers
  (T : ℕ)
  (h1 : 3 * T = 6)
  (numNoTipCustomers : ℕ := 5)
  (total := numCustomers T 6 3 numNoTipCustomers) :
  total = 7 := by
  sorry

end NUMINAMATH_GPT_waiter_total_customers_l2080_208099


namespace NUMINAMATH_GPT_unique_solution_positive_n_l2080_208041

theorem unique_solution_positive_n (n : ℝ) : 
  ( ∃ x : ℝ, 4 * x^2 + n * x + 16 = 0 ∧ ∀ y : ℝ, 4 * y^2 + n * y + 16 = 0 → y = x ) → n = 16 := 
by {
  sorry
}

end NUMINAMATH_GPT_unique_solution_positive_n_l2080_208041


namespace NUMINAMATH_GPT_polygon_sides_eq_seven_l2080_208050

theorem polygon_sides_eq_seven (n : ℕ) (h : 2 * n - (n * (n - 3)) / 2 = 0) : n = 7 :=
by sorry

end NUMINAMATH_GPT_polygon_sides_eq_seven_l2080_208050


namespace NUMINAMATH_GPT_find_additional_speed_l2080_208004

noncomputable def speed_initial : ℝ := 55
noncomputable def t_initial : ℝ := 4
noncomputable def speed_total : ℝ := 60
noncomputable def t_total : ℝ := 6

theorem find_additional_speed :
  let distance_initial := speed_initial * t_initial
  let distance_total := speed_total * t_total
  let t_additional := t_total - t_initial
  let distance_additional := distance_total - distance_initial
  let speed_additional := distance_additional / t_additional
  speed_additional = 70 :=
by
  sorry

end NUMINAMATH_GPT_find_additional_speed_l2080_208004


namespace NUMINAMATH_GPT_arithmetic_sequence_75th_term_diff_l2080_208054

noncomputable def sum_arith_sequence (n : ℕ) (a d : ℚ) : ℚ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_75th_term_diff {n : ℕ} {a d : ℚ}
  (hn : n = 150)
  (sum_seq : sum_arith_sequence n a d = 15000)
  (term_range : ∀ k, 0 ≤ k ∧ k < n → 20 ≤ a + k * d ∧ a + k * d ≤ 150)
  (t75th : ∃ L G, L = a + 74 * d ∧ G = a + 74 * d) :
  G - L = (7500 / 149) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_75th_term_diff_l2080_208054


namespace NUMINAMATH_GPT_statement_B_statement_C_l2080_208096

variable (a b c : ℝ)

-- Condition: a > b
def condition1 := a > b

-- Condition: a / c^2 > b / c^2
def condition2 := a / c^2 > b / c^2

-- Statement B: If a > b, then a - 1 > b - 2
theorem statement_B (ha_gt_b : condition1 a b) : a - 1 > b - 2 :=
by sorry

-- Statement C: If a / c^2 > b / c^2, then a > b
theorem statement_C (ha_div_csqr_gt_hb_div_csqr : condition2 a b c) : a > b :=
by sorry

end NUMINAMATH_GPT_statement_B_statement_C_l2080_208096


namespace NUMINAMATH_GPT_perfect_square_m_value_l2080_208072

theorem perfect_square_m_value (y m : ℤ) (h : ∃ k : ℤ, y^2 - 8 * y + m = (y - k)^2) : m = 16 :=
sorry

end NUMINAMATH_GPT_perfect_square_m_value_l2080_208072


namespace NUMINAMATH_GPT_find_number_l2080_208092

theorem find_number (x : ℕ) (h : 8 * x = 64) : x = 8 :=
sorry

end NUMINAMATH_GPT_find_number_l2080_208092


namespace NUMINAMATH_GPT_find_a_l2080_208056

theorem find_a (a : ℝ) (f : ℝ → ℝ) (h_def : ∀ x, f x = 3 * x^(a-2) - 2) (h_cond : f 2 = 4) : a = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2080_208056


namespace NUMINAMATH_GPT_reservoir_shortage_l2080_208067

noncomputable def reservoir_information := 
  let current_level := 14 -- million gallons
  let normal_level_due_to_yield := current_level / 2
  let percentage_of_capacity := 0.70
  let evaporation_factor := 0.90
  let total_capacity := current_level / percentage_of_capacity
  let normal_level_after_evaporation := normal_level_due_to_yield * evaporation_factor
  let shortage := total_capacity - normal_level_after_evaporation
  shortage

theorem reservoir_shortage :
  reservoir_information = 13.7 := 
by
  sorry

end NUMINAMATH_GPT_reservoir_shortage_l2080_208067


namespace NUMINAMATH_GPT_jake_watching_hours_l2080_208079

theorem jake_watching_hours
    (monday_hours : ℕ := 12) -- Half of 24 hours in a day is 12 hours for Monday
    (wednesday_hours : ℕ := 6) -- A quarter of 24 hours in a day is 6 hours for Wednesday
    (friday_hours : ℕ := 19) -- Jake watched 19 hours on Friday
    (total_hours : ℕ := 52) -- The entire show is 52 hours long
    (T : ℕ) -- To find the total number of hours on Tuesday
    (h : monday_hours + T + wednesday_hours + (monday_hours + T + wednesday_hours) / 2 + friday_hours = total_hours) :
    T = 4 := sorry

end NUMINAMATH_GPT_jake_watching_hours_l2080_208079


namespace NUMINAMATH_GPT_largest_prime_factor_of_4752_l2080_208038

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬ (m ∣ n)

def largest_prime_factor (n : ℕ) (p : ℕ) : Prop :=
  is_prime p ∧ p ∣ n ∧ (∀ q : ℕ, is_prime q ∧ q ∣ n → q ≤ p)

def pf_4752 : ℕ := 4752

theorem largest_prime_factor_of_4752 : largest_prime_factor pf_4752 11 :=
  by
  sorry

end NUMINAMATH_GPT_largest_prime_factor_of_4752_l2080_208038


namespace NUMINAMATH_GPT_least_three_digit_multiple_of_3_4_7_l2080_208017

theorem least_three_digit_multiple_of_3_4_7 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 3 = 0 ∧ n % 4 = 0 ∧ n % 7 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 → m % 3 = 0 ∧ m % 4 = 0 ∧ m % 7 = 0 → n ≤ m :=
  sorry

end NUMINAMATH_GPT_least_three_digit_multiple_of_3_4_7_l2080_208017


namespace NUMINAMATH_GPT_tank_breadth_l2080_208046

/-
  We need to define the conditions:
  1. The field dimensions.
  2. The tank dimensions (length and depth), and the unknown breadth.
  3. The relationship after the tank is dug.
-/

noncomputable def field_length : ℝ := 90
noncomputable def field_breadth : ℝ := 50
noncomputable def tank_length : ℝ := 25
noncomputable def tank_depth : ℝ := 4
noncomputable def rise_in_level : ℝ := 0.5

theorem tank_breadth (B : ℝ) (h : 100 * B = (field_length * field_breadth - tank_length * B) * rise_in_level) : B = 20 :=
by sorry

end NUMINAMATH_GPT_tank_breadth_l2080_208046


namespace NUMINAMATH_GPT_smaller_bills_denomination_correct_l2080_208044

noncomputable def denomination_of_smaller_bills : ℕ :=
  let total_money := 1000
  let part_smaller_bills := 3 / 10
  let smaller_bills_amount := part_smaller_bills * total_money
  let rest_of_money := total_money - smaller_bills_amount
  let bill_100_denomination := 100
  let total_bills := 13
  let num_100_bills := rest_of_money / bill_100_denomination
  let num_smaller_bills := total_bills - num_100_bills
  let denomination := smaller_bills_amount / num_smaller_bills
  denomination

theorem smaller_bills_denomination_correct : denomination_of_smaller_bills = 50 := by
  sorry

end NUMINAMATH_GPT_smaller_bills_denomination_correct_l2080_208044


namespace NUMINAMATH_GPT_find_a_l2080_208040

theorem find_a (a x : ℝ) (h1 : 3 * x + 2 * a = 2) (h2 : x = 1) : a = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2080_208040


namespace NUMINAMATH_GPT_discount_on_item_l2080_208037

noncomputable def discount_percentage : ℝ := 20
variable (total_cart_value original_price final_amount : ℝ)
variable (coupon_discount : ℝ)

axiom cart_value : total_cart_value = 54
axiom item_price : original_price = 20
axiom coupon : coupon_discount = 0.10
axiom final_price : final_amount = 45

theorem discount_on_item :
  ∃ x : ℝ, (total_cart_value - (x / 100) * original_price) * (1 - coupon_discount) = final_amount ∧ x = discount_percentage :=
by
  have eq1 := cart_value
  have eq2 := item_price
  have eq3 := coupon
  have eq4 := final_price
  sorry

end NUMINAMATH_GPT_discount_on_item_l2080_208037


namespace NUMINAMATH_GPT_value_of_V3_l2080_208055

-- Define the polynomial function using Horner's rule
def f (x : ℤ) := (((((2 * x + 0) * x - 3) * x + 2) * x + 1) * x - 3)

-- Define the value of x
def x : ℤ := 2

-- Prove the value of V_3 when x = 2
theorem value_of_V3 : f x = 12 := by
  sorry

end NUMINAMATH_GPT_value_of_V3_l2080_208055


namespace NUMINAMATH_GPT_fraction_equality_l2080_208016

theorem fraction_equality (x y z : ℝ) (k : ℝ) (hx : x = 3 * k) (hy : y = 5 * k) (hz : z = 7 * k) :
  (x - y + z) / (x + y - z) = 5 := 
  sorry

end NUMINAMATH_GPT_fraction_equality_l2080_208016


namespace NUMINAMATH_GPT_product_plus_one_is_square_l2080_208007

theorem product_plus_one_is_square (x y : ℕ) (h : x * y = (x + 2) * (y - 2)) : x * y + 1 = (x + 1) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_product_plus_one_is_square_l2080_208007


namespace NUMINAMATH_GPT_length_GH_of_tetrahedron_l2080_208043

noncomputable def tetrahedron_edge_length : ℕ := 24

theorem length_GH_of_tetrahedron
  (a b c d e f : ℕ)
  (h1 : a = 8) 
  (h2 : b = 16) 
  (h3 : c = 24) 
  (h4 : d = 35) 
  (h5 : e = 45) 
  (h6 : f = 55)
  (hEF : f = 55)
  (hEGF : e + b > f)
  (hEHG: e + c > a ∧ e + c > d) 
  (hFHG : b + c > a ∧ b + f > c ∧ c + a > b):
   tetrahedron_edge_length = c := 
sorry

end NUMINAMATH_GPT_length_GH_of_tetrahedron_l2080_208043


namespace NUMINAMATH_GPT_part1_smallest_period_part1_monotonic_interval_part2_value_of_a_l2080_208014

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x ^ 2 + Real.sin (7 * Real.pi / 6 - 2 * x) - 1

theorem part1_smallest_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := 
sorry

theorem part1_monotonic_interval :
  ∀ k : ℤ, ∀ x, (k * Real.pi - Real.pi / 3) ≤ x ∧ x ≤ (k * Real.pi + Real.pi / 6) →
  ∃ (b a c : ℝ) (A : ℝ), b + c = 2 * a ∧ 2 * A = A + Real.pi / 3 ∧ 
  f A = 1 / 2 ∧ a = 3 * Real.sqrt 2 := 
sorry

theorem part2_value_of_a :
  ∀ (A b c : ℝ), 
  (∃ (a : ℝ), 2 * a = b + c ∧ 
  f A = 1 / 2 ∧ 
  b * c = 18 ∧ 
  Real.cos A = 1 / 2) → 
  ∃ a, a = 3 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_part1_smallest_period_part1_monotonic_interval_part2_value_of_a_l2080_208014


namespace NUMINAMATH_GPT_larry_channel_reduction_l2080_208098

theorem larry_channel_reduction
  (initial_channels new_channels final_channels sports_package supreme_sports_package channels_at_end : ℕ)
  (h_initial : initial_channels = 150)
  (h_adjustment : new_channels = initial_channels - 20 + 12)
  (h_sports : sports_package = 8)
  (h_supreme_sports : supreme_sports_package = 7)
  (h_channels_at_end : channels_at_end = 147)
  (h_final : final_channels = channels_at_end - sports_package - supreme_sports_package) :
  initial_channels - 20 + 12 - final_channels = 10 := 
sorry

end NUMINAMATH_GPT_larry_channel_reduction_l2080_208098


namespace NUMINAMATH_GPT_cloves_needed_l2080_208022

theorem cloves_needed (cv_fp : 3 / 2 = 1.5) (cw_fp : 3 / 3 = 1) (vc_fp : 3 / 8 = 0.375) : 
  let cloves_for_vampires := 45
  let cloves_for_wights := 12
  let cloves_for_bats := 15
  30 * (3 / 2) + 12 * (3 / 3) + 40 * (3 / 8) = 72 := by
  sorry

end NUMINAMATH_GPT_cloves_needed_l2080_208022
