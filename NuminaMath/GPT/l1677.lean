import Mathlib

namespace NUMINAMATH_GPT_chuck_vs_dave_ride_time_l1677_167766

theorem chuck_vs_dave_ride_time (D E : ℕ) (h1 : D = 10) (h2 : E = 65) (h3 : E = 13 * C / 10) :
  (C / D = 5) :=
by
  sorry

end NUMINAMATH_GPT_chuck_vs_dave_ride_time_l1677_167766


namespace NUMINAMATH_GPT_power_of_fraction_l1677_167774

theorem power_of_fraction : ((1/3)^5 = (1/243)) :=
by
  sorry

end NUMINAMATH_GPT_power_of_fraction_l1677_167774


namespace NUMINAMATH_GPT_exists_language_spoken_by_at_least_three_l1677_167769

noncomputable def smallestValue_n (k : ℕ) : ℕ :=
  2 * k + 3

theorem exists_language_spoken_by_at_least_three (k n : ℕ) (P : Fin n → Set ℕ) (K : ℕ → ℕ) :
  (n = smallestValue_n k) →
  (∀ i, (K i) ≤ k) →
  (∀ (x y z : Fin n), ∃ l, l ∈ P x ∧ l ∈ P y ∧ l ∈ P z ∨ l ∈ P y ∧ l ∈ P z ∨ l ∈ P z ∧ l ∈ P x ∨ l ∈ P x ∧ l ∈ P y) →
  ∃ l, ∃ (a b c : Fin n), l ∈ P a ∧ l ∈ P b ∧ l ∈ P c :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_exists_language_spoken_by_at_least_three_l1677_167769


namespace NUMINAMATH_GPT_thread_length_l1677_167797

theorem thread_length (x : ℝ) (h : x + (3/4) * x = 21) : x = 12 :=
  sorry

end NUMINAMATH_GPT_thread_length_l1677_167797


namespace NUMINAMATH_GPT_find_judy_rotation_l1677_167773

-- Definition of the problem
def CarlaRotation := 480 % 360 -- This effectively becomes 120
def JudyRotation (y : ℕ) := (360 - 120) % 360 -- This should effectively be 240

-- Theorem stating the problem and solution
theorem find_judy_rotation (y : ℕ) (h : y < 360) : 360 - CarlaRotation = y :=
by 
  dsimp [CarlaRotation, JudyRotation] 
  sorry

end NUMINAMATH_GPT_find_judy_rotation_l1677_167773


namespace NUMINAMATH_GPT_strictly_increasing_0_to_e_l1677_167714

noncomputable def ln (x : ℝ) : ℝ := Real.log x

noncomputable def f (x : ℝ) : ℝ := ln x / x

theorem strictly_increasing_0_to_e :
  ∀ x : ℝ, 0 < x ∧ x < Real.exp 1 → 0 < (1 - ln x) / (x^2) :=
by
  sorry

end NUMINAMATH_GPT_strictly_increasing_0_to_e_l1677_167714


namespace NUMINAMATH_GPT_playground_ratio_l1677_167775

theorem playground_ratio (L B : ℕ) (playground_area landscape_area : ℕ) 
  (h1 : B = 8 * L)
  (h2 : B = 480)
  (h3 : playground_area = 3200)
  (h4 : landscape_area = L * B) : 
  (playground_area : ℚ) / landscape_area = 1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_playground_ratio_l1677_167775


namespace NUMINAMATH_GPT_johns_elevation_after_descent_l1677_167798

def starting_elevation : ℝ := 400
def rate_of_descent : ℝ := 10
def travel_time : ℝ := 5

theorem johns_elevation_after_descent :
  starting_elevation - (rate_of_descent * travel_time) = 350 :=
by
  sorry

end NUMINAMATH_GPT_johns_elevation_after_descent_l1677_167798


namespace NUMINAMATH_GPT_jiwon_distance_to_school_l1677_167721

theorem jiwon_distance_to_school
  (taehong_distance_meters jiwon_distance_meters : ℝ)
  (taehong_distance_km : ℝ := 1.05)
  (h1 : taehong_distance_meters = jiwon_distance_meters + 460)
  (h2 : taehong_distance_meters = taehong_distance_km * 1000) :
  jiwon_distance_meters / 1000 = 0.59 := 
sorry

end NUMINAMATH_GPT_jiwon_distance_to_school_l1677_167721


namespace NUMINAMATH_GPT_mary_has_34_lambs_l1677_167702

def mary_lambs (initial_lambs : ℕ) (lambs_with_babies : ℕ) (babies_per_lamb : ℕ) (traded_lambs : ℕ) (found_lambs : ℕ): ℕ :=
  initial_lambs + (lambs_with_babies * babies_per_lamb) - traded_lambs + found_lambs

theorem mary_has_34_lambs :
  mary_lambs 12 4 3 5 15 = 34 :=
by
  -- This line is in place of the actual proof.
  sorry

end NUMINAMATH_GPT_mary_has_34_lambs_l1677_167702


namespace NUMINAMATH_GPT_car_speed_second_hour_l1677_167794

theorem car_speed_second_hour
  (speed_first_hour : ℝ)
  (avg_speed : ℝ)
  (hours : ℝ)
  (total_time : ℝ)
  (total_distance : ℝ)
  (distance_first_hour : ℝ)
  (distance_second_hour : ℝ) :
  speed_first_hour = 90 →
  avg_speed = 75 →
  hours = 2 →
  total_time = hours →
  total_distance = avg_speed * total_time →
  distance_first_hour = speed_first_hour * 1 →
  distance_second_hour = total_distance - distance_first_hour →
  distance_second_hour / 1 = 60 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_car_speed_second_hour_l1677_167794


namespace NUMINAMATH_GPT_necessary_not_sufficient_l1677_167781

theorem necessary_not_sufficient (a b : ℝ) : (a > b - 1) ∧ ¬ (a > b - 1 → a > b) := 
sorry

end NUMINAMATH_GPT_necessary_not_sufficient_l1677_167781


namespace NUMINAMATH_GPT_solve_fractional_equation_l1677_167704

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 4) : (1 / (x - 1) = 2 / (1 - x) + 1) → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_fractional_equation_l1677_167704


namespace NUMINAMATH_GPT_one_head_two_tails_probability_l1677_167724

noncomputable def probability_of_one_head_two_tails :=
  let total_outcomes := 8
  let favorable_outcomes := 3
  favorable_outcomes / total_outcomes

theorem one_head_two_tails_probability :
  probability_of_one_head_two_tails = 3 / 8 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_one_head_two_tails_probability_l1677_167724


namespace NUMINAMATH_GPT_kitten_length_l1677_167738

theorem kitten_length (initial_length : ℕ) (doubled_length_1 : ℕ) (doubled_length_2 : ℕ) :
  initial_length = 4 →
  doubled_length_1 = 2 * initial_length →
  doubled_length_2 = 2 * doubled_length_1 →
  doubled_length_2 = 16 :=
by
  intros h1 h2 h3
  rw [h1] at h2
  rw [h2] at h3
  exact h3

end NUMINAMATH_GPT_kitten_length_l1677_167738


namespace NUMINAMATH_GPT_unique_solution_fraction_l1677_167763

theorem unique_solution_fraction (x : ℝ) :
  (2 * x^2 - 10 * x + 8 ≠ 0) → 
  (∃! (x : ℝ), (3 * x^2 - 15 * x + 12) / (2 * x^2 - 10 * x + 8) = x - 4) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_fraction_l1677_167763


namespace NUMINAMATH_GPT_find_y_value_l1677_167760

theorem find_y_value : (12 ^ 2 * 6 ^ 4) / 432 = 432 := by
  sorry

end NUMINAMATH_GPT_find_y_value_l1677_167760


namespace NUMINAMATH_GPT_polygon_interior_exterior_eq_l1677_167740

theorem polygon_interior_exterior_eq (n : ℕ) (hn : 3 ≤ n)
  (interior_sum_eq_exterior_sum : (n - 2) * 180 = 360) : n = 4 := by
  sorry

end NUMINAMATH_GPT_polygon_interior_exterior_eq_l1677_167740


namespace NUMINAMATH_GPT_stream_speed_l1677_167755

variable (B S : ℝ)

def downstream_eq : Prop := B + S = 13
def upstream_eq : Prop := B - S = 5

theorem stream_speed (h1 : downstream_eq B S) (h2 : upstream_eq B S) : S = 4 :=
by
  sorry

end NUMINAMATH_GPT_stream_speed_l1677_167755


namespace NUMINAMATH_GPT_population_reduction_l1677_167750

theorem population_reduction (initial_population : ℕ) (final_population : ℕ) (left_percentage : ℝ)
    (bombardment_percentage : ℝ) :
    initial_population = 7145 →
    final_population = 4555 →
    left_percentage = 0.75 →
    bombardment_percentage = 100 - 84.96 →
    ∃ (x : ℝ), bombardment_percentage = (100 - x) := 
by
    sorry

end NUMINAMATH_GPT_population_reduction_l1677_167750


namespace NUMINAMATH_GPT_cindy_olaf_earnings_l1677_167727
noncomputable def total_earnings (apples grapes : ℕ) (price_apple price_grape : ℝ) : ℝ :=
  apples * price_apple + grapes * price_grape

theorem cindy_olaf_earnings :
  total_earnings 15 12 2 1.5 = 48 :=
by
  sorry

end NUMINAMATH_GPT_cindy_olaf_earnings_l1677_167727


namespace NUMINAMATH_GPT_linear_function_no_second_quadrant_l1677_167706

theorem linear_function_no_second_quadrant (x y : ℝ) (h : y = 2 * x - 3) :
  ¬ ((x < 0) ∧ (y > 0)) :=
by {
  sorry
}

end NUMINAMATH_GPT_linear_function_no_second_quadrant_l1677_167706


namespace NUMINAMATH_GPT_min_value_x1x2_squared_inequality_ab_l1677_167705

def D : Set (ℝ × ℝ) := 
  { p | ∃ x1 x2, p = (x1, x2) ∧ x1 + x2 = 2 ∧ x1 > 0 ∧ x2 > 0 }

-- Part 1: Proving the minimum value of x1^2 + x2^2 in set D is 2
theorem min_value_x1x2_squared (x1 x2 : ℝ) (h : (x1, x2) ∈ D) : 
  x1^2 + x2^2 ≥ 2 := 
sorry

-- Part 2: Proving the inequality for any (a, b) in set D
theorem inequality_ab (a b : ℝ) (h : (a, b) ∈ D) : 
  (1 / (a + 2 * b) + 1 / (2 * a + b)) ≥ (2 / 3) := 
sorry

end NUMINAMATH_GPT_min_value_x1x2_squared_inequality_ab_l1677_167705


namespace NUMINAMATH_GPT_trapezoid_area_ratio_l1677_167733

theorem trapezoid_area_ratio (b h x : ℝ) 
  (base_relation : b + 150 = x)
  (area_ratio : (3 / 7) * h * (b + 75) = (1 / 2) * h * (b + x))
  (mid_segment : x = b + 150) 
  : ⌊x^3 / 1000⌋ = 142 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_area_ratio_l1677_167733


namespace NUMINAMATH_GPT_polynomial_proof_l1677_167784

variable (a b : ℝ)

-- Define the given monomial and the resulting polynomial 
def monomial := -3 * a ^ 2 * b
def result := 6 * a ^ 3 * b ^ 2 - 3 * a ^ 2 * b ^ 2 + 9 * a ^ 2 * b

-- Define the polynomial we want to prove
def poly := -2 * a * b + b - 3

-- Statement of the problem in Lean 4
theorem polynomial_proof :
  monomial * poly = result :=
by sorry

end NUMINAMATH_GPT_polynomial_proof_l1677_167784


namespace NUMINAMATH_GPT_correct_operation_B_l1677_167757

theorem correct_operation_B (x : ℝ) : 
  x - 2 * x = -x :=
sorry

end NUMINAMATH_GPT_correct_operation_B_l1677_167757


namespace NUMINAMATH_GPT_distance_to_SFL_is_81_l1677_167701

variable (Speed : ℝ)
variable (Time : ℝ)

def distance_to_SFL (Speed : ℝ) (Time : ℝ) := Speed * Time

theorem distance_to_SFL_is_81 : distance_to_SFL 27 3 = 81 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_SFL_is_81_l1677_167701


namespace NUMINAMATH_GPT_sum_of_roots_l1677_167745

theorem sum_of_roots : 
  ∀ x1 x2 : ℝ, 
  (x1^2 + 2023*x1 = 2024 ∧ x2^2 + 2023*x2 = 2024) → 
  x1 + x2 = -2023 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_roots_l1677_167745


namespace NUMINAMATH_GPT_school_should_purchase_bookshelves_l1677_167795

theorem school_should_purchase_bookshelves
  (x : ℕ)
  (h₁ : x ≥ 20)
  (cost_A : ℕ := 20 * 300 + 100 * (x - 20))
  (cost_B : ℕ := (20 * 300 + 100 * x) * 80 / 100)
  (h₂ : cost_A = cost_B) : x = 40 :=
by sorry

end NUMINAMATH_GPT_school_should_purchase_bookshelves_l1677_167795


namespace NUMINAMATH_GPT_combined_salaries_l1677_167725

variable (S_A S_B S_C S_D S_E : ℝ)

theorem combined_salaries 
    (h1 : S_C = 16000)
    (h2 : (S_A + S_B + S_C + S_D + S_E) / 5 = 9000) : 
    S_A + S_B + S_D + S_E = 29000 :=
by 
    sorry

end NUMINAMATH_GPT_combined_salaries_l1677_167725


namespace NUMINAMATH_GPT_constant_term_expansion_l1677_167717

-- auxiliary definitions and facts
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def term_constant (n k : ℕ) (a b x : ℂ) : ℂ :=
  binomial_coeff n k * (a * x)^(n-k) * (b / x)^k

-- main theorem statement
theorem constant_term_expansion : ∀ (x : ℂ), (term_constant 8 4 (5 : ℂ) (2 : ℂ) x).re = 1120 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_constant_term_expansion_l1677_167717


namespace NUMINAMATH_GPT_find_ages_l1677_167742

variables (H J A : ℕ)

def conditions := 
  H + J + A = 90 ∧ 
  H = 2 * J - 5 ∧ 
  H + J - 10 = A

theorem find_ages (h_cond : conditions H J A) : 
  H = 32 ∧ 
  J = 18 ∧ 
  A = 40 :=
sorry

end NUMINAMATH_GPT_find_ages_l1677_167742


namespace NUMINAMATH_GPT_f_at_zero_f_on_negative_l1677_167759

-- Define the odd function condition
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the function f(x) for x > 0 condition
def f_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x, x > 0 → f x = x^2 + x - 1

-- Lean statement for the first proof: f(0) = 0
theorem f_at_zero (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_positive : f_on_positive f) : f 0 = 0 :=
sorry

-- Lean statement for the second proof: for x < 0, f(x) = -x^2 + x + 1
theorem f_on_negative (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_positive : f_on_positive f) :
  ∀ x, x < 0 → f x = -x^2 + x + 1 :=
sorry

end NUMINAMATH_GPT_f_at_zero_f_on_negative_l1677_167759


namespace NUMINAMATH_GPT_exists_composite_power_sum_l1677_167744

def is_composite (n : ℕ) : Prop := ∃ p q : ℕ, 1 < p ∧ 1 < q ∧ n = p * q 

theorem exists_composite_power_sum (a : ℕ) (h1 : 1 < a) (h2 : a ≤ 100) : 
  ∃ n, (n > 0) ∧ (n ≤ 6) ∧ is_composite (a ^ (2 ^ n) + 1) :=
by
  sorry

end NUMINAMATH_GPT_exists_composite_power_sum_l1677_167744


namespace NUMINAMATH_GPT_average_speed_l1677_167716

-- Define the conditions
def initial_reading : ℕ := 2552
def final_reading : ℕ := 2992
def day1_time : ℕ := 6
def day2_time : ℕ := 8

-- Theorem: Proving the average speed is 31 miles per hour.
theorem average_speed :
  final_reading - initial_reading = 440 ∧ day1_time + day2_time = 14 ∧ 
  (final_reading - initial_reading) / (day1_time + day2_time) = 31 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_l1677_167716


namespace NUMINAMATH_GPT_show_spiders_l1677_167710

noncomputable def spiders_found (ants : ℕ) (ladybugs_initial : ℕ) (ladybugs_fly_away : ℕ) (total_insects_remaining : ℕ) : ℕ :=
  let ladybugs_remaining := ladybugs_initial - ladybugs_fly_away
  let insects_observed := ants + ladybugs_remaining
  total_insects_remaining - insects_observed

theorem show_spiders
  (ants : ℕ := 12)
  (ladybugs_initial : ℕ := 8)
  (ladybugs_fly_away : ℕ := 2)
  (total_insects_remaining : ℕ := 21) :
  spiders_found ants ladybugs_initial ladybugs_fly_away total_insects_remaining = 3 := by
  sorry

end NUMINAMATH_GPT_show_spiders_l1677_167710


namespace NUMINAMATH_GPT_cost_of_baseball_is_correct_l1677_167785

-- Define the costs and total amount spent
def cost_of_marbles : ℝ := 9.05
def cost_of_football : ℝ := 4.95
def total_amount_spent : ℝ := 20.52

-- Define the cost of the baseball
def cost_of_baseball : ℝ := total_amount_spent - (cost_of_marbles + cost_of_football)

-- The theorem we want to prove
theorem cost_of_baseball_is_correct :
  cost_of_baseball = 6.52 := by
  sorry

end NUMINAMATH_GPT_cost_of_baseball_is_correct_l1677_167785


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1677_167713

theorem hyperbola_eccentricity (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : b = a) 
  (h₄ : ∀ c, (c = Real.sqrt (a^2 + b^2)) → (b * c / Real.sqrt (a^2 + b^2) = a)) :
  (Real.sqrt (2) = (c / a)) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1677_167713


namespace NUMINAMATH_GPT_trig_identity_l1677_167783

theorem trig_identity (x : ℝ) (h : Real.sin (π / 6 - x) = 1 / 2) :
  Real.sin (19 * π / 6 - x) + Real.sin (-2 * π / 3 + x) ^ 2 = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l1677_167783


namespace NUMINAMATH_GPT_louisa_average_speed_l1677_167703

def average_speed (v : ℝ) : Prop :=
  (350 / v) - (200 / v) = 3

theorem louisa_average_speed :
  ∃ v : ℝ, average_speed v ∧ v = 50 := 
by
  use 50
  unfold average_speed
  sorry

end NUMINAMATH_GPT_louisa_average_speed_l1677_167703


namespace NUMINAMATH_GPT_find_DG_l1677_167799

theorem find_DG (a b k l : ℕ) (h1 : a * k = 37 * (a + b)) (h2 : b * l = 37 * (a + b)) : 
  k = 1406 :=
by
  sorry

end NUMINAMATH_GPT_find_DG_l1677_167799


namespace NUMINAMATH_GPT_three_digit_numbers_with_4_and_5_correct_l1677_167780

def count_three_digit_numbers_with_4_and_5 : ℕ :=
  48

theorem three_digit_numbers_with_4_and_5_correct :
  count_three_digit_numbers_with_4_and_5 = 48 :=
by
  sorry -- proof goes here

end NUMINAMATH_GPT_three_digit_numbers_with_4_and_5_correct_l1677_167780


namespace NUMINAMATH_GPT_number_of_factors_60_l1677_167788

def prime_factorization_60 : Prop := (60 = 2^2 * 3 * 5)

theorem number_of_factors_60 (h : prime_factorization_60) : 
  12 = ( (2 + 1) * (1 + 1) * (1 + 1) ) := 
by
  sorry

end NUMINAMATH_GPT_number_of_factors_60_l1677_167788


namespace NUMINAMATH_GPT_find_number_l1677_167734

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 9) : x = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1677_167734


namespace NUMINAMATH_GPT_find_a_value_l1677_167764

noncomputable def prob_sum_equals_one (a : ℝ) : Prop :=
  a * (1/2 + 1/4 + 1/8 + 1/16) = 1

theorem find_a_value (a : ℝ) (h : prob_sum_equals_one a) : a = 16/15 :=
sorry

end NUMINAMATH_GPT_find_a_value_l1677_167764


namespace NUMINAMATH_GPT_original_price_given_discounts_l1677_167749

theorem original_price_given_discounts (p q d : ℝ) (h : d > 0) :
  ∃ x : ℝ, x * (1 + (p - q) / 100 - p * q / 10000) = d :=
by
  sorry

end NUMINAMATH_GPT_original_price_given_discounts_l1677_167749


namespace NUMINAMATH_GPT_compare_abc_case1_compare_abc_case2_compare_abc_case3_l1677_167711

variable (a : ℝ)
variable (b : ℝ := (1 / 2) * (a + 3 / a))
variable (c : ℝ := (1 / 2) * (b + 3 / b))

-- First condition: if \(a > \sqrt{3}\), then \(a > b > c\)
theorem compare_abc_case1 (h1 : a > 0) (h2 : a > Real.sqrt 3) : a > b ∧ b > c := sorry

-- Second condition: if \(a = \sqrt{3}\), then \(a = b = c\)
theorem compare_abc_case2 (h1 : a > 0) (h2 : a = Real.sqrt 3) : a = b ∧ b = c := sorry

-- Third condition: if \(0 < a < \sqrt{3}\), then \(a < c < b\)
theorem compare_abc_case3 (h1 : a > 0) (h2 : a < Real.sqrt 3) : a < c ∧ c < b := sorry

end NUMINAMATH_GPT_compare_abc_case1_compare_abc_case2_compare_abc_case3_l1677_167711


namespace NUMINAMATH_GPT_fraction_multiplication_division_l1677_167761

-- We will define the fractions and state the equivalence
def fraction_1 : ℚ := 145 / 273
def fraction_2 : ℚ := 2 * (173 / 245) -- equivalent to 2 173/245
def fraction_3 : ℚ := 21 * (13 / 15) -- equivalent to 21 13/15

theorem fraction_multiplication_division :
  (frac1 * frac2 / frac3) = 7395 / 112504 := 
by sorry

end NUMINAMATH_GPT_fraction_multiplication_division_l1677_167761


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l1677_167715

theorem eccentricity_of_ellipse 
  (a b c m n : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : m > 0) 
  (h4 : n > 0) 
  (ellipse_eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 -> (m^2 + n^2 > x^2 + y^2))
  (hyperbola_eq : ∀ x y : ℝ, x^2 / m^2 - y^2 / n^2 = 1 -> (m^2 + n^2 > x^2 - y^2))
  (same_foci: ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 <= 1 → x^2 / m^2 - y^2 / n^2 = 1)
  (geometric_mean : c^2 = a * m)
  (arithmetic_mean : 2 * n^2 = 2 * m^2 + c^2) : 
  (c / a = 1 / 2) :=
sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l1677_167715


namespace NUMINAMATH_GPT_fresh_pineapples_left_l1677_167777

namespace PineappleStore

def initial := 86
def sold := 48
def rotten := 9

theorem fresh_pineapples_left (initial sold rotten : ℕ) (h_initial : initial = 86) (h_sold : sold = 48) (h_rotten : rotten = 9) :
  initial - sold - rotten = 29 :=
by sorry

end PineappleStore

end NUMINAMATH_GPT_fresh_pineapples_left_l1677_167777


namespace NUMINAMATH_GPT_number_of_cows_l1677_167730

-- Definitions
variables (a g e c : ℕ)
variables (six_two : 6 * e = 2 * a + 4 * g) (eight_two : 8 * e = 2 * a + 8 * g)

-- Theorem statement
theorem number_of_cows (a g e : ℕ) (six_two : 6 * e = 2 * a + 4 * g) (eight_two : 8 * e = 2 * a + 8 * g) :
  ∃ c : ℕ, c * e * 6 = 6 * a + 36 * g ∧ c = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cows_l1677_167730


namespace NUMINAMATH_GPT_slope_of_line_l1677_167756

theorem slope_of_line {x y : ℝ} : 
  (∃ (x y : ℝ), 0 = 3 * x + 4 * y + 12) → ∀ (m : ℝ), m = -3/4 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_l1677_167756


namespace NUMINAMATH_GPT_triangle_equi_if_sides_eq_sum_of_products_l1677_167790

theorem triangle_equi_if_sides_eq_sum_of_products (a b c : ℝ) (h : a^2 + b^2 + c^2 = ab + bc + ac) : a = b ∧ b = c :=
by sorry

end NUMINAMATH_GPT_triangle_equi_if_sides_eq_sum_of_products_l1677_167790


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1677_167751

theorem quadratic_inequality_solution (m : ℝ) : 
  (∀ x : ℝ, x^2 + m * x + 1 ≥ 0) ↔ (-2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1677_167751


namespace NUMINAMATH_GPT_rationalize_and_divide_l1677_167752

theorem rationalize_and_divide :
  (8 / Real.sqrt 8 / 2) = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_rationalize_and_divide_l1677_167752


namespace NUMINAMATH_GPT_negation_of_exists_solution_l1677_167719

theorem negation_of_exists_solution :
  ¬ (∃ c : ℝ, c > 0 ∧ ∃ x : ℝ, x^2 - x + c = 0) ↔ ∀ c : ℝ, c > 0 → ¬ (∃ x : ℝ, x^2 - x + c = 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_solution_l1677_167719


namespace NUMINAMATH_GPT_exists_sum_pair_l1677_167731

theorem exists_sum_pair (n : ℕ) (a b : List ℕ) (h₁ : ∀ x ∈ a, x < n) (h₂ : ∀ y ∈ b, y < n) 
  (h₃ : List.Nodup a) (h₄ : List.Nodup b) (h₅ : a.length + b.length ≥ n) : ∃ x ∈ a, ∃ y ∈ b, x + y = n := by
  sorry

end NUMINAMATH_GPT_exists_sum_pair_l1677_167731


namespace NUMINAMATH_GPT_coin_difference_is_eight_l1677_167747

theorem coin_difference_is_eight :
  let min_coins := 2  -- two 25-cent coins
  let max_coins := 10 -- ten 5-cent coins
  max_coins - min_coins = 8 :=
by
  sorry

end NUMINAMATH_GPT_coin_difference_is_eight_l1677_167747


namespace NUMINAMATH_GPT_mauve_red_paint_parts_l1677_167743

noncomputable def parts_of_red_in_mauve : ℕ :=
let fuchsia_red_ratio := 5
let fuchsia_blue_ratio := 3
let total_fuchsia := 16
let added_blue := 14
let mauve_blue_ratio := 6

let total_fuchsia_parts := fuchsia_red_ratio + fuchsia_blue_ratio
let red_in_fuchsia := (fuchsia_red_ratio * total_fuchsia) / total_fuchsia_parts
let blue_in_fuchsia := (fuchsia_blue_ratio * total_fuchsia) / total_fuchsia_parts
let blue_in_mauve := blue_in_fuchsia + added_blue
let ratio_red_to_blue_in_mauve := red_in_fuchsia / blue_in_mauve
ratio_red_to_blue_in_mauve * mauve_blue_ratio

theorem mauve_red_paint_parts : parts_of_red_in_mauve = 3 :=
by sorry

end NUMINAMATH_GPT_mauve_red_paint_parts_l1677_167743


namespace NUMINAMATH_GPT_max_p_l1677_167709

theorem max_p (p q r s t u v w : ℕ)
  (h1 : p + q + r + s = 35)
  (h2 : q + r + s + t = 35)
  (h3 : r + s + t + u = 35)
  (h4 : s + t + u + v = 35)
  (h5 : t + u + v + w = 35)
  (h6 : q + v = 14) :
  p ≤ 20 :=
sorry

end NUMINAMATH_GPT_max_p_l1677_167709


namespace NUMINAMATH_GPT_tax_percentage_excess_income_l1677_167728

theorem tax_percentage_excess_income :
  ∀ (rate : ℝ) (total_tax income : ℝ), 
  rate = 0.15 →
  total_tax = 8000 →
  income = 50000 →
  (total_tax - income * rate) / (income - 40000) = 0.2 :=
by
  intros rate total_tax income hrate htotal hincome
  -- proof omitted
  sorry

end NUMINAMATH_GPT_tax_percentage_excess_income_l1677_167728


namespace NUMINAMATH_GPT_average_apples_per_hour_l1677_167741

theorem average_apples_per_hour :
  (5.0 / 3.0) = 1.67 := 
sorry

end NUMINAMATH_GPT_average_apples_per_hour_l1677_167741


namespace NUMINAMATH_GPT_total_number_of_people_l1677_167792

variables (A : ℕ) -- Number of adults in the group

-- Conditions
-- Each adult meal costs $8 and the total cost was $72
def cost_per_adult_meal : ℕ := 8
def total_cost : ℕ := 72
def number_of_kids : ℕ := 2

-- Proof problem: Given the conditions, prove the total number of people in the group is 11
theorem total_number_of_people (h : A * cost_per_adult_meal = total_cost) : A + number_of_kids = 11 :=
sorry

end NUMINAMATH_GPT_total_number_of_people_l1677_167792


namespace NUMINAMATH_GPT_eugene_total_pencils_l1677_167768

-- Define the initial number of pencils Eugene has
def initial_pencils : ℕ := 51

-- Define the number of pencils Joyce gives to Eugene
def pencils_from_joyce : ℕ := 6

-- Define the expected total number of pencils
def expected_total_pencils : ℕ := 57

-- Theorem to prove the total number of pencils Eugene has
theorem eugene_total_pencils : initial_pencils + pencils_from_joyce = expected_total_pencils := 
by sorry

end NUMINAMATH_GPT_eugene_total_pencils_l1677_167768


namespace NUMINAMATH_GPT_no_four_digit_numbers_divisible_by_11_l1677_167739

theorem no_four_digit_numbers_divisible_by_11 (a b c d : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9) 
(h₃ : 0 ≤ b) (h₄ : b ≤ 9) (h₅ : 0 ≤ c) (h₆ : c ≤ 9) (h₇ : 0 ≤ d) (h₈ : d ≤ 9) 
(h₉ : a + b + c + d = 10) (h₁₀ : a + c = b + d) : 
0 = 0 :=
sorry

end NUMINAMATH_GPT_no_four_digit_numbers_divisible_by_11_l1677_167739


namespace NUMINAMATH_GPT_decrease_in_demand_correct_l1677_167762

noncomputable def proportionate_decrease_in_demand (p e : ℝ) : ℝ :=
  1 - (1 / (1 + e * p))

theorem decrease_in_demand_correct :
  proportionate_decrease_in_demand 0.20 1.5 = 0.23077 :=
by
  sorry

end NUMINAMATH_GPT_decrease_in_demand_correct_l1677_167762


namespace NUMINAMATH_GPT_geometric_progression_quadrilateral_exists_l1677_167770

theorem geometric_progression_quadrilateral_exists :
  ∃ (a1 r : ℝ), a1 > 0 ∧ r > 0 ∧ 
  (1 + r + r^2 > r^3) ∧
  (1 + r + r^3 > r^2) ∧
  (1 + r^2 + r^3 > r) ∧
  (r + r^2 + r^3 > 1) := 
sorry

end NUMINAMATH_GPT_geometric_progression_quadrilateral_exists_l1677_167770


namespace NUMINAMATH_GPT_find_tangent_line_l1677_167796

def curve := fun x : ℝ => x^3 + 2 * x + 1
def tangent_point := 1
def tangent_line (x y : ℝ) := 5 * x - y - 1 = 0

theorem find_tangent_line :
  tangent_line tangent_point (curve tangent_point) :=
by
  sorry

end NUMINAMATH_GPT_find_tangent_line_l1677_167796


namespace NUMINAMATH_GPT_number_with_150_quarters_is_37_point_5_l1677_167793

theorem number_with_150_quarters_is_37_point_5 (n : ℝ) (h : n / (1/4) = 150) : n = 37.5 := 
by 
  sorry

end NUMINAMATH_GPT_number_with_150_quarters_is_37_point_5_l1677_167793


namespace NUMINAMATH_GPT_constants_sum_l1677_167776

theorem constants_sum (c d : ℝ) (f : ℝ → ℝ) 
  (h₁ : ∀ x : ℝ, f x = if x ≤ 5 then c * x + d else 10 - 2 * x) 
  (h₂ : ∀ x : ℝ, f (f x) = x) : c + d = 6.5 := 
by sorry

end NUMINAMATH_GPT_constants_sum_l1677_167776


namespace NUMINAMATH_GPT_twelve_point_five_minutes_in_seconds_l1677_167791

-- Definitions
def minutes_to_seconds (m : ℝ) : ℝ := m * 60

-- Theorem: Prove that 12.5 minutes is 750 seconds
theorem twelve_point_five_minutes_in_seconds : minutes_to_seconds 12.5 = 750 :=
by 
  sorry

end NUMINAMATH_GPT_twelve_point_five_minutes_in_seconds_l1677_167791


namespace NUMINAMATH_GPT_rationalize_denominator_l1677_167778

theorem rationalize_denominator :
  (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = -9 - 4 * Real.sqrt 5 :=
by
  -- Commutative field properties and algebraic manipulation will be used here.
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l1677_167778


namespace NUMINAMATH_GPT_area_of_square_field_l1677_167771

-- Definitions
def cost_per_meter : ℝ := 1.40
def total_cost : ℝ := 932.40
def gate_width : ℝ := 1.0

-- Problem Statement
theorem area_of_square_field (s : ℝ) (A : ℝ) 
  (h1 : (4 * s - 2 * gate_width) * cost_per_meter = total_cost)
  (h2 : A = s^2) : A = 27889 := 
  sorry

end NUMINAMATH_GPT_area_of_square_field_l1677_167771


namespace NUMINAMATH_GPT_find_a_l1677_167707

theorem find_a (a : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x = abs (2 * x - a) + a)
  (h2 : ∀ x : ℝ, f x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) : 
  a = 1 := by
  sorry

end NUMINAMATH_GPT_find_a_l1677_167707


namespace NUMINAMATH_GPT_books_sold_on_Tuesday_l1677_167754

theorem books_sold_on_Tuesday 
  (initial_stock : ℕ)
  (books_sold_Monday : ℕ)
  (books_sold_Wednesday : ℕ)
  (books_sold_Thursday : ℕ)
  (books_sold_Friday : ℕ)
  (books_not_sold : ℕ) :
  initial_stock = 800 →
  books_sold_Monday = 60 →
  books_sold_Wednesday = 20 →
  books_sold_Thursday = 44 →
  books_sold_Friday = 66 →
  books_not_sold = 600 →
  ∃ (books_sold_Tuesday : ℕ), books_sold_Tuesday = 10
:= by
  intros h_initial h_monday h_wednesday h_thursday h_friday h_not_sold
  sorry

end NUMINAMATH_GPT_books_sold_on_Tuesday_l1677_167754


namespace NUMINAMATH_GPT_sum_of_x_coordinates_of_intersections_l1677_167732

def g : ℝ → ℝ := sorry  -- Definition of g is unspecified but it consists of five line segments.

theorem sum_of_x_coordinates_of_intersections 
  (h1 : ∃ x1, g x1 = x1 - 2 ∧ (x1 = -2 ∨ x1 = 1 ∨ x1 = 4))
  (h2 : ∃ x2, g x2 = x2 - 2 ∧ (x2 = -2 ∨ x2 = 1 ∨ x2 = 4))
  (h3 : ∃ x3, g x3 = x3 - 2 ∧ (x3 = -2 ∨ x3 = 1 ∨ x3 = 4)) 
  (hx1x2x3 : x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) :
  x1 + x2 + x3 = 3 := by
  -- Proof here
  sorry

end NUMINAMATH_GPT_sum_of_x_coordinates_of_intersections_l1677_167732


namespace NUMINAMATH_GPT_tan_difference_identity_l1677_167787

theorem tan_difference_identity {α : ℝ} (h : Real.tan α = 4 * Real.sin (7 * Real.pi / 3)) :
  Real.tan (α - Real.pi / 3) = Real.sqrt 3 / 7 := 
sorry

end NUMINAMATH_GPT_tan_difference_identity_l1677_167787


namespace NUMINAMATH_GPT_tom_wins_with_smallest_n_l1677_167753

def tom_and_jerry_game_proof_problem (n : ℕ) : Prop :=
  ∀ (pos : ℕ), pos ≥ 1 ∧ pos ≤ 2018 → 
  ∀ (move : ℕ), move ≥ 1 ∧ move ≤ n →
  (∃ n_min : ℕ, n_min ≤ n ∧ ∀ pos, (pos ≤ n_min ∨ pos > 2018 - n_min) → false)

theorem tom_wins_with_smallest_n : tom_and_jerry_game_proof_problem 1010 :=
sorry

end NUMINAMATH_GPT_tom_wins_with_smallest_n_l1677_167753


namespace NUMINAMATH_GPT_pencil_length_l1677_167782

theorem pencil_length :
  let purple := 1.5
  let black := 0.5
  let blue := 2
  purple + black + blue = 4 := by sorry

end NUMINAMATH_GPT_pencil_length_l1677_167782


namespace NUMINAMATH_GPT_find_a_l1677_167765

variable (a : ℝ)

def p (a : ℝ) : Set ℝ := {x | a-1 < x ∧ x < a+1}
def q : Set ℝ := {x | x^2 - 4*x + 3 ≥ 0}
def q_negation : Set ℝ := {x | 1 < x ∧ x < 3}

theorem find_a :
  (∀ x, q_negation x → p a x) → a = 2 := by
  sorry

end NUMINAMATH_GPT_find_a_l1677_167765


namespace NUMINAMATH_GPT_circumference_of_circle_of_given_area_l1677_167729

theorem circumference_of_circle_of_given_area (A : ℝ) (h : A = 225 * Real.pi) : 
  ∃ C : ℝ, C = 2 * Real.pi * 15 :=
by
  let r := 15
  let C := 2 * Real.pi * r
  use C
  sorry

end NUMINAMATH_GPT_circumference_of_circle_of_given_area_l1677_167729


namespace NUMINAMATH_GPT_triangle_angle_B_l1677_167737

theorem triangle_angle_B (a b A B : ℝ) (h1 : a * Real.cos B = 3 * b * Real.cos A) (h2 : B = A - Real.pi / 6) : 
  B = Real.pi / 6 := by
  sorry

end NUMINAMATH_GPT_triangle_angle_B_l1677_167737


namespace NUMINAMATH_GPT_relationship_between_heights_is_correlated_l1677_167700

theorem relationship_between_heights_is_correlated :
  (∃ r : ℕ, (r = 1 ∨ r = 2 ∨ r = 3 ∨ r = 4) ∧ r = 2) := by
  sorry

end NUMINAMATH_GPT_relationship_between_heights_is_correlated_l1677_167700


namespace NUMINAMATH_GPT_travel_ways_l1677_167746

theorem travel_ways (highways : ℕ) (railways : ℕ) (n : ℕ) :
  highways = 3 → railways = 2 → n = highways + railways → n = 5 :=
by
  intros h_eq r_eq n_eq
  rw [h_eq, r_eq] at n_eq
  exact n_eq

end NUMINAMATH_GPT_travel_ways_l1677_167746


namespace NUMINAMATH_GPT_restaurant_table_difference_l1677_167736

theorem restaurant_table_difference :
  ∃ (N O : ℕ), N + O = 40 ∧ 6 * N + 4 * O = 212 ∧ (N - O) = 12 :=
by
  sorry

end NUMINAMATH_GPT_restaurant_table_difference_l1677_167736


namespace NUMINAMATH_GPT_possible_to_fill_array_l1677_167772

open BigOperators

theorem possible_to_fill_array :
  ∃ (f : (Fin 10) × (Fin 10) → ℕ),
    (∀ i j : Fin 10, 
      (i ≠ 0 → f (i, j) ∣ f (i - 1, j) ∧ f (i, j) ≠ f (i - 1, j))) ∧
    (∀ i : Fin 10, ∃ n : ℕ, ∀ j : Fin 10, f (i, j) = n + j) :=
sorry

end NUMINAMATH_GPT_possible_to_fill_array_l1677_167772


namespace NUMINAMATH_GPT_forty_percent_of_number_l1677_167767

theorem forty_percent_of_number (N : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 20) : 0.40 * N = 240 :=
by
  sorry

end NUMINAMATH_GPT_forty_percent_of_number_l1677_167767


namespace NUMINAMATH_GPT_female_democrats_count_l1677_167720

theorem female_democrats_count :
  ∃ (F : ℕ) (M : ℕ),
    F + M = 750 ∧
    (F / 2) + (M / 4) = 250 ∧
    1 / 3 * 750 = 250 ∧
    F / 2 = 125 := sorry

end NUMINAMATH_GPT_female_democrats_count_l1677_167720


namespace NUMINAMATH_GPT_mike_practices_hours_on_saturday_l1677_167748

-- Definitions based on conditions
def weekday_hours : ℕ := 3
def weekdays_per_week : ℕ := 5
def total_hours : ℕ := 60
def weeks : ℕ := 3

def calculate_total_weekday_hours (weekday_hours weekdays_per_week weeks : ℕ) : ℕ :=
  weekday_hours * weekdays_per_week * weeks

def calculate_saturday_hours (total_hours total_weekday_hours weeks : ℕ) : ℕ :=
  (total_hours - total_weekday_hours) / weeks

-- Statement to prove
theorem mike_practices_hours_on_saturday :
  calculate_saturday_hours total_hours (calculate_total_weekday_hours weekday_hours weekdays_per_week weeks) weeks = 5 :=
by 
  sorry

end NUMINAMATH_GPT_mike_practices_hours_on_saturday_l1677_167748


namespace NUMINAMATH_GPT_solve_4_times_3_l1677_167789

noncomputable def custom_operation (x y : ℕ) : ℕ := x^2 - x * y + y^2

theorem solve_4_times_3 : custom_operation 4 3 = 13 := by
  -- Here the proof would be provided, for now we use sorry
  sorry

end NUMINAMATH_GPT_solve_4_times_3_l1677_167789


namespace NUMINAMATH_GPT_compute_expression_l1677_167718

theorem compute_expression (y : ℕ) (h : y = 3) : (y^8 + 10 * y^4 + 25) / (y^4 + 5) = 86 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_compute_expression_l1677_167718


namespace NUMINAMATH_GPT_solve_equation_l1677_167708

theorem solve_equation (x : ℝ) (h : (x^2 - x + 2) / (x - 1) = x + 3) : x = 5 / 3 := 
by sorry

end NUMINAMATH_GPT_solve_equation_l1677_167708


namespace NUMINAMATH_GPT_basketball_team_win_requirement_l1677_167722

theorem basketball_team_win_requirement :
  ∀ (games_won_first_60 : ℕ) (total_games : ℕ) (win_percentage : ℚ) (remaining_games : ℕ),
    games_won_first_60 = 45 →
    total_games = 110 →
    win_percentage = 0.75 →
    remaining_games = 50 →
    ∃ games_won_remaining, games_won_remaining = 38 ∧
    (games_won_first_60 + games_won_remaining) / total_games = win_percentage :=
by
  intros
  sorry

end NUMINAMATH_GPT_basketball_team_win_requirement_l1677_167722


namespace NUMINAMATH_GPT_average_marks_of_all_students_l1677_167786

theorem average_marks_of_all_students (n1 n2 a1 a2 : ℕ) (n1_eq : n1 = 12) (a1_eq : a1 = 40) 
  (n2_eq : n2 = 28) (a2_eq : a2 = 60) : 
  ((n1 * a1 + n2 * a2) / (n1 + n2) : ℕ) = 54 := 
by
  sorry

end NUMINAMATH_GPT_average_marks_of_all_students_l1677_167786


namespace NUMINAMATH_GPT_mouse_jump_less_than_frog_l1677_167723

-- Definitions for the given conditions
def grasshopper_jump : ℕ := 25
def frog_jump : ℕ := grasshopper_jump + 32
def mouse_jump : ℕ := 31

-- The statement we need to prove
theorem mouse_jump_less_than_frog :
  frog_jump - mouse_jump = 26 :=
by
  -- The proof will be filled in here
  sorry

end NUMINAMATH_GPT_mouse_jump_less_than_frog_l1677_167723


namespace NUMINAMATH_GPT_monotonically_increasing_interval_l1677_167779

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem monotonically_increasing_interval :
  ∀ x : ℝ, x > 1 / Real.exp 1 → (Real.log x + 1) > 0 :=
by
  intros x hx
  sorry

end NUMINAMATH_GPT_monotonically_increasing_interval_l1677_167779


namespace NUMINAMATH_GPT_average_minutes_per_day_is_correct_l1677_167712
-- Import required library for mathematics

-- Define the conditions
def sixth_grade_minutes := 10
def seventh_grade_minutes := 12
def eighth_grade_minutes := 8
def sixth_grade_ratio := 3
def eighth_grade_ratio := 1/2

-- We use noncomputable since we'll rely on some real number operations that are not trivially computable.
noncomputable def total_minutes_per_week (s : ℝ) : ℝ :=
  sixth_grade_minutes * (sixth_grade_ratio * s) * 2 + 
  seventh_grade_minutes * s * 2 + 
  eighth_grade_minutes * (eighth_grade_ratio * s) * 1

noncomputable def total_students (s : ℝ) : ℝ :=
  sixth_grade_ratio * s + s + eighth_grade_ratio * s

noncomputable def average_minutes_per_day : ℝ :=
  (total_minutes_per_week 1) / (total_students 1 / 5)

theorem average_minutes_per_day_is_correct : average_minutes_per_day = 176 / 9 :=
by
  sorry

end NUMINAMATH_GPT_average_minutes_per_day_is_correct_l1677_167712


namespace NUMINAMATH_GPT_sum_of_cubes_l1677_167735

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 11) (h2 : x * y = 12) : x^3 + y^3 = 935 := by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l1677_167735


namespace NUMINAMATH_GPT_div_by_64_l1677_167726

theorem div_by_64 (n : ℕ) (h : n ≥ 1) : 64 ∣ (3^(2*n + 2) - 8*n - 9) :=
sorry

end NUMINAMATH_GPT_div_by_64_l1677_167726


namespace NUMINAMATH_GPT_largest_composite_sequence_l1677_167758

theorem largest_composite_sequence (a b c d e f g : ℕ) (h₁ : a < b) (h₂ : b < c) (h₃ : c < d) (h₄ : d < e) (h₅ : e < f) (h₆ : f < g) 
  (h₇ : g < 50) (h₈ : a ≥ 10) (h₉ : g ≤ 32)
  (h₁₀ : ¬ Prime a) (h₁₁ : ¬ Prime b) (h₁₂ : ¬ Prime c) (h₁₃ : ¬ Prime d) 
  (h₁₄ : ¬ Prime e) (h₁₅ : ¬ Prime f) (h₁₆ : ¬ Prime g) :
  g = 32 :=
sorry

end NUMINAMATH_GPT_largest_composite_sequence_l1677_167758
