import Mathlib

namespace NUMINAMATH_GPT_posters_count_l1124_112420

-- Define the regular price per poster
def regular_price : ℕ := 4

-- Jeremy can buy 24 posters at regular price
def posters_at_regular_price : ℕ := 24

-- Total money Jeremy has is equal to the money needed to buy 24 posters
def total_money : ℕ := posters_at_regular_price * regular_price

-- The special deal: buy one get the second at half price
def cost_of_two_posters : ℕ := regular_price + regular_price / 2

-- Number of pairs Jeremy can buy with his total money
def number_of_pairs : ℕ := total_money / cost_of_two_posters

-- Total number of posters Jeremy can buy under the sale
def total_posters := number_of_pairs * 2

-- Prove that the total posters is 32
theorem posters_count : total_posters = 32 := by
  sorry

end NUMINAMATH_GPT_posters_count_l1124_112420


namespace NUMINAMATH_GPT_ratio_of_distance_l1124_112468

noncomputable def initial_distance : ℝ := 30 * 20

noncomputable def total_distance : ℝ := 2 * initial_distance

noncomputable def distance_after_storm : ℝ := initial_distance - 200

theorem ratio_of_distance (initial_distance : ℝ) (total_distance : ℝ) (distance_after_storm : ℝ) : 
  distance_after_storm / total_distance = 1 / 3 :=
by
  -- Given conditions
  have h1 : initial_distance = 30 * 20 := by sorry
  have h2 : total_distance = 2 * initial_distance := by sorry
  have h3 : distance_after_storm = initial_distance - 200 := by sorry
  -- Prove the ratio is 1 / 3
  sorry

end NUMINAMATH_GPT_ratio_of_distance_l1124_112468


namespace NUMINAMATH_GPT_identify_quadratic_equation_l1124_112440

theorem identify_quadratic_equation :
  (∀ b c d : Prop, ∀ (f : ℕ → Prop), f 0 → ¬ f 1 → ¬ f 2 → ¬ f 3 → b ∧ ¬ c ∧ ¬ d) →
  (∀ x y : ℝ,  (x^2 + 2 = 0) = (b ∧ ¬ b → c ∧ ¬ c → d ∧ ¬ d)) :=
by
  intros;
  sorry

end NUMINAMATH_GPT_identify_quadratic_equation_l1124_112440


namespace NUMINAMATH_GPT_diet_soda_bottles_l1124_112456

theorem diet_soda_bottles (R D : ℕ) 
  (h1 : R = 60)
  (h2 : R = D + 41) :
  D = 19 :=
by {
  sorry
}

end NUMINAMATH_GPT_diet_soda_bottles_l1124_112456


namespace NUMINAMATH_GPT_moles_KOH_combined_l1124_112431

-- Define the number of moles of KI produced
def moles_KI_produced : ℕ := 3

-- Define the molar ratio from the balanced chemical equation
def molar_ratio_KOH_NH4I_KI : ℕ := 1

-- The number of moles of KOH combined to produce the given moles of KI
theorem moles_KOH_combined (moles_KOH moles_NH4I : ℕ) (h : moles_NH4I = 3) 
  (h_produced : moles_KI_produced = 3) (ratio : molar_ratio_KOH_NH4I_KI = 1) :
  moles_KOH = 3 :=
by {
  -- Placeholder for proof, use sorry to skip proving
  sorry
}

end NUMINAMATH_GPT_moles_KOH_combined_l1124_112431


namespace NUMINAMATH_GPT_find_number_of_numbers_l1124_112452

theorem find_number_of_numbers (S : ℝ) (n : ℝ) (h1 : S - 30 = 16 * n) (h2 : S = 19 * n) : n = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_numbers_l1124_112452


namespace NUMINAMATH_GPT_neg_neg_two_neg_six_plus_six_neg_three_times_five_two_x_minus_three_x_l1124_112419

-- (1) Prove -(-2) = 2
theorem neg_neg_two : -(-2) = 2 := 
sorry

-- (2) Prove -6 + 6 = 0
theorem neg_six_plus_six : -6 + 6 = 0 := 
sorry

-- (3) Prove (-3) * 5 = -15
theorem neg_three_times_five : (-3) * 5 = -15 := 
sorry

-- (4) Prove 2x - 3x = -x
theorem two_x_minus_three_x (x : ℝ) : 2 * x - 3 * x = - x := 
sorry

end NUMINAMATH_GPT_neg_neg_two_neg_six_plus_six_neg_three_times_five_two_x_minus_three_x_l1124_112419


namespace NUMINAMATH_GPT_combination_eq_permutation_div_factorial_l1124_112447

-- Step d): Lean 4 Statement

variables (n k : ℕ)

-- Define combination C_n^k is any k-element subset of an n-element set
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define permutation A_n^k is the number of ways to arrange k elements out of n elements
def permutation (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

-- Statement to prove: C_n^k = A_n^k / k!
theorem combination_eq_permutation_div_factorial :
  combination n k = permutation n k / (Nat.factorial k) :=
by
  sorry

end NUMINAMATH_GPT_combination_eq_permutation_div_factorial_l1124_112447


namespace NUMINAMATH_GPT_least_area_of_prime_dim_l1124_112495

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem least_area_of_prime_dim (l w : ℕ) (h_perimeter : 2 * (l + w) = 120)
    (h_integer_dims : l > 0 ∧ w > 0) (h_prime_dim : is_prime l ∨ is_prime w) :
    l * w = 116 :=
sorry

end NUMINAMATH_GPT_least_area_of_prime_dim_l1124_112495


namespace NUMINAMATH_GPT_nat_n_divisibility_cond_l1124_112437

theorem nat_n_divisibility_cond (n : ℕ) : (n * 2^n + 1) % 3 = 0 ↔ (n % 3 = 1 ∨ n % 3 = 2) :=
by sorry

end NUMINAMATH_GPT_nat_n_divisibility_cond_l1124_112437


namespace NUMINAMATH_GPT_diagonals_in_30_sided_polygon_l1124_112465

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem diagonals_in_30_sided_polygon :
  number_of_diagonals 30 = 405 :=
by sorry

end NUMINAMATH_GPT_diagonals_in_30_sided_polygon_l1124_112465


namespace NUMINAMATH_GPT_probability_div_int_l1124_112499

theorem probability_div_int
    (r : ℤ) (k : ℤ)
    (hr : -5 < r ∧ r < 10)
    (hk : 1 < k ∧ k < 8)
    (hk_prime : Nat.Prime (Int.natAbs k)) :
    ∃ p q : ℕ, (p = 3 ∧ q = 14) ∧ p / q = 3 / 14 := 
by {
  sorry
}

end NUMINAMATH_GPT_probability_div_int_l1124_112499


namespace NUMINAMATH_GPT_david_money_left_l1124_112450

theorem david_money_left (S : ℤ) (h1 : S - 800 = 1800 - S) : 1800 - S = 500 :=
by
  sorry

end NUMINAMATH_GPT_david_money_left_l1124_112450


namespace NUMINAMATH_GPT_zeros_of_f_on_interval_l1124_112429

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.log x)

theorem zeros_of_f_on_interval : ∃ (S : Set ℝ), S ⊆ (Set.Ioo 0 1) ∧ S.Infinite ∧ ∀ x ∈ S, f x = 0 := by
  sorry

end NUMINAMATH_GPT_zeros_of_f_on_interval_l1124_112429


namespace NUMINAMATH_GPT_julie_same_hours_september_october_l1124_112467

-- Define Julie's hourly rates and work hours
def rate_mowing : ℝ := 4
def rate_weeding : ℝ := 8
def september_mowing_hours : ℕ := 25
def september_weeding_hours : ℕ := 3
def total_earnings_september_october : ℤ := 248

-- Define Julie's earnings for each activity and total earnings for September
def september_earnings_mowing : ℝ := september_mowing_hours * rate_mowing
def september_earnings_weeding : ℝ := september_weeding_hours * rate_weeding
def september_total_earnings : ℝ := september_earnings_mowing + september_earnings_weeding

-- Define earnings in October
def october_earnings : ℝ := total_earnings_september_october - september_total_earnings

-- Define the theorem to prove Julie worked the same number of hours in October as in September
theorem julie_same_hours_september_october :
  october_earnings = september_total_earnings :=
by
  sorry

end NUMINAMATH_GPT_julie_same_hours_september_october_l1124_112467


namespace NUMINAMATH_GPT_min_value_of_expression_l1124_112496

theorem min_value_of_expression {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 1/a + 2/b = 3) :
  (a + 1) * (b + 2) = 50/9 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l1124_112496


namespace NUMINAMATH_GPT_solve_for_x_l1124_112455

theorem solve_for_x (x : ℝ) (h : 0.05 * x + 0.12 * (30 + x) = 15.6) : x = 1200 / 17 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1124_112455


namespace NUMINAMATH_GPT_quadrilateral_circumscribed_circle_l1124_112477

theorem quadrilateral_circumscribed_circle (a : ℝ) :
  ((a + 2) * x + (1 - a) * y - 3 = 0) ∧ ((a - 1) * x + (2 * a + 3) * y + 2 = 0) →
  ( a = 1 ∨ a = -1 ) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_quadrilateral_circumscribed_circle_l1124_112477


namespace NUMINAMATH_GPT_difference_of_two_numbers_l1124_112441

-- Definitions as per conditions
def L : ℕ := 1656
def S : ℕ := 273
def quotient : ℕ := 6
def remainder : ℕ := 15

-- Statement of the proof problem
theorem difference_of_two_numbers (h1 : L = 6 * S + 15) : L - S = 1383 :=
by sorry

end NUMINAMATH_GPT_difference_of_two_numbers_l1124_112441


namespace NUMINAMATH_GPT_min_value_expression_l1124_112451

theorem min_value_expression :
  ∀ (x y z w : ℝ), x > 0 → y > 0 → z > 0 → w > 0 → x = y → x + y + z + w = 1 →
  (x + y + z) / (x * y * z * w) ≥ 1024 :=
by
  intros x y z w hx hy hz hw hxy hsum
  sorry

end NUMINAMATH_GPT_min_value_expression_l1124_112451


namespace NUMINAMATH_GPT_second_number_value_l1124_112483

theorem second_number_value (x y : ℝ) (h1 : (1/5) * x = (5/8) * y) 
                                      (h2 : x + 35 = 4 * y) : y = 40 := 
by 
  sorry

end NUMINAMATH_GPT_second_number_value_l1124_112483


namespace NUMINAMATH_GPT_scientific_notation_of_935000000_l1124_112408

theorem scientific_notation_of_935000000 :
  935000000 = 9.35 * 10^8 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_935000000_l1124_112408


namespace NUMINAMATH_GPT_max_xy_of_conditions_l1124_112457

noncomputable def max_xy : ℝ := 37.5

theorem max_xy_of_conditions (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 10 * x + 15 * y = 150) (h4 : x^2 + y^2 ≤ 100) :
  xy ≤ max_xy :=
by sorry

end NUMINAMATH_GPT_max_xy_of_conditions_l1124_112457


namespace NUMINAMATH_GPT_find_alpha_l1124_112460

theorem find_alpha (α : ℝ) (k : ℤ) 
  (h : ∃ (k : ℤ), α + 30 = k * 360 + 180) : 
  α = k * 360 + 150 :=
by 
  sorry

end NUMINAMATH_GPT_find_alpha_l1124_112460


namespace NUMINAMATH_GPT_mul_18396_9999_l1124_112481

theorem mul_18396_9999 :
  18396 * 9999 = 183941604 :=
by
  sorry

end NUMINAMATH_GPT_mul_18396_9999_l1124_112481


namespace NUMINAMATH_GPT_slips_with_3_l1124_112475

-- Definitions of the conditions
def num_slips : ℕ := 15
def expected_value : ℚ := 5.4

-- Theorem statement
theorem slips_with_3 (y : ℕ) (t : ℕ := num_slips) (E : ℚ := expected_value) :
  E = (3 * y + 8 * (t - y)) / t → y = 8 :=
by
  sorry

end NUMINAMATH_GPT_slips_with_3_l1124_112475


namespace NUMINAMATH_GPT_sum_of_N_values_eq_neg_one_l1124_112413

theorem sum_of_N_values_eq_neg_one (R : ℝ) :
  ∀ (N : ℝ), N ≠ 0 ∧ (N + N^2 - 5 / N = R) →
  (∃ N₁ N₂ N₃ : ℝ, N₁ + N₂ + N₃ = -1 ∧ N₁ ≠ 0 ∧ N₂ ≠ 0 ∧ N₃ ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_N_values_eq_neg_one_l1124_112413


namespace NUMINAMATH_GPT_tracey_initial_candies_l1124_112411

theorem tracey_initial_candies (x : ℕ) :
  (x % 4 = 0) ∧ (104 ≤ x) ∧ (x ≤ 112) ∧
  (∃ k : ℕ, 2 ≤ k ∧ k ≤ 6 ∧ (x / 2 - 40 - k = 10)) →
  (x = 108 ∨ x = 112) :=
by
  sorry

end NUMINAMATH_GPT_tracey_initial_candies_l1124_112411


namespace NUMINAMATH_GPT_pow_log_sqrt_l1124_112412

theorem pow_log_sqrt (a b c : ℝ) (h1 : a = 81) (h2 : b = 500) (h3 : c = 3) :
  ((a ^ (Real.log b / Real.log c)) ^ (1 / 2)) = 250000 :=
by
  sorry

end NUMINAMATH_GPT_pow_log_sqrt_l1124_112412


namespace NUMINAMATH_GPT_find_value_l1124_112473

variable (x y a c : ℝ)

-- Conditions
def condition1 : Prop := x * y = 2 * c
def condition2 : Prop := (1 / x ^ 2) + (1 / y ^ 2) = 3 * a

-- Proof statement
theorem find_value : condition1 x y c ∧ condition2 x y a ↔ (x + y) ^ 2 = 12 * a * c ^ 2 + 4 * c := 
by 
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_find_value_l1124_112473


namespace NUMINAMATH_GPT_ellipse_area_l1124_112418

-- Definitions based on the conditions
def cylinder_height : ℝ := 10
def cylinder_base_radius : ℝ := 1

-- Equivalent Proof Problem Statement
theorem ellipse_area
  (h : ℝ := cylinder_height)
  (r : ℝ := cylinder_base_radius)
  (ball_position_lower : ℝ := -4) -- derived from - (h / 2 - r)
  (ball_position_upper : ℝ := 4) -- derived from  (h / 2 - r)
  : (π * 4 * 2 = 16 * π) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_area_l1124_112418


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_l1124_112454

variable {α : Type*} [LinearOrderedField α]

noncomputable def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
∀ n, a (n + 1) - a n = d

noncomputable def sum_of_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
n * (a 1 + a n) / 2

theorem sum_of_arithmetic_sequence {a : ℕ → α} {d : α}
  (h3 : a 3 * a 7 = -16)
  (h4 : a 4 + a 6 = 0)
  (ha : is_arithmetic_sequence a d) :
  ∃ (s : α), s = n * (n - 9) ∨ s = -n * (n - 9) :=
sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_l1124_112454


namespace NUMINAMATH_GPT_floor_sqrt_245_l1124_112442

theorem floor_sqrt_245 : (Int.floor (Real.sqrt 245)) = 15 :=
by
  sorry

end NUMINAMATH_GPT_floor_sqrt_245_l1124_112442


namespace NUMINAMATH_GPT_total_steps_traveled_l1124_112497

def steps_per_mile : ℕ := 2000
def walk_to_subway : ℕ := 2000
def subway_ride_miles : ℕ := 7
def walk_to_rockefeller : ℕ := 3000
def cab_ride_miles : ℕ := 3

theorem total_steps_traveled :
  walk_to_subway +
  (subway_ride_miles * steps_per_mile) +
  walk_to_rockefeller +
  (cab_ride_miles * steps_per_mile)
  = 24000 := 
by 
  sorry

end NUMINAMATH_GPT_total_steps_traveled_l1124_112497


namespace NUMINAMATH_GPT_distinct_students_count_l1124_112402

theorem distinct_students_count
  (algebra_students : ℕ)
  (calculus_students : ℕ)
  (statistics_students : ℕ)
  (algebra_statistics_overlap : ℕ)
  (no_other_overlaps : algebra_students + calculus_students + statistics_students - algebra_statistics_overlap = 32) :
  algebra_students = 13 → calculus_students = 10 → statistics_students = 12 → algebra_statistics_overlap = 3 → 
  algebra_students + calculus_students + statistics_students - algebra_statistics_overlap = 32 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_distinct_students_count_l1124_112402


namespace NUMINAMATH_GPT_processing_rates_and_total_cost_l1124_112484

variables (products total_days total_days_A total_days_B daily_capacity_A daily_capacity_B total_cost_A total_cost_B : ℝ)

noncomputable def A_processing_rate : ℝ := daily_capacity_A
noncomputable def B_processing_rate : ℝ := daily_capacity_B

theorem processing_rates_and_total_cost
  (h1 : products = 1000)
  (h2 : total_days_A = total_days_B + 10)
  (h3 : daily_capacity_B = 1.25 * daily_capacity_A)
  (h4 : total_cost_A = 100 * total_days_A)
  (h5 : total_cost_B = 125 * total_days_B) :
  (daily_capacity_A = 20) ∧ (daily_capacity_B = 25) ∧ (total_cost_A + total_cost_B = 5000) :=
by
  sorry

end NUMINAMATH_GPT_processing_rates_and_total_cost_l1124_112484


namespace NUMINAMATH_GPT_division_problem_l1124_112436

theorem division_problem : 250 / (15 + 13 * 3 - 4) = 5 := by
  sorry

end NUMINAMATH_GPT_division_problem_l1124_112436


namespace NUMINAMATH_GPT_class_grades_l1124_112492

theorem class_grades (boys girls n : ℕ) (h1 : girls = boys + 3) (h2 : ∀ (fours fives : ℕ), fours = fives + 6) (h3 : ∀ (threes : ℕ), threes = 2 * (fives + 6)) : ∃ k, k = 2 ∨ k = 1 :=
by
  sorry

end NUMINAMATH_GPT_class_grades_l1124_112492


namespace NUMINAMATH_GPT_find_C_l1124_112458

theorem find_C (C : ℤ) (h : 4 * C + 3 = 31) : C = 7 := by
  sorry

end NUMINAMATH_GPT_find_C_l1124_112458


namespace NUMINAMATH_GPT_flight_duration_l1124_112446

theorem flight_duration (departure_time arrival_time : ℕ) (time_difference : ℕ) (h m : ℕ) (m_bound : 0 < m ∧ m < 60) 
  (h_val : h = 1) (m_val : m = 35)  : h + m = 36 := by
  sorry

end NUMINAMATH_GPT_flight_duration_l1124_112446


namespace NUMINAMATH_GPT_proper_subsets_count_l1124_112409

theorem proper_subsets_count (A : Set (Fin 4)) (h : A = {1, 2, 3}) : 
  ∃ n : ℕ, n = 7 ∧ ∃ (S : Finset (Set (Fin 4))), S.card = n ∧ (∀ B, B ∈ S → B ⊂ A) := 
by {
  sorry
}

end NUMINAMATH_GPT_proper_subsets_count_l1124_112409


namespace NUMINAMATH_GPT_chandler_saves_for_laptop_l1124_112480

theorem chandler_saves_for_laptop :
  ∃ x : ℕ, 140 + 20 * x = 800 ↔ x = 33 :=
by
  use 33
  sorry

end NUMINAMATH_GPT_chandler_saves_for_laptop_l1124_112480


namespace NUMINAMATH_GPT_slices_left_l1124_112464

-- Conditions
def total_slices : ℕ := 16
def fraction_eaten : ℚ := 3/4
def fraction_left : ℚ := 1 - fraction_eaten

-- Proof statement
theorem slices_left : total_slices * fraction_left = 4 := by
  sorry

end NUMINAMATH_GPT_slices_left_l1124_112464


namespace NUMINAMATH_GPT_find_numbers_l1124_112471

def is_solution (a b : ℕ) : Prop :=
  a + b = 432 ∧ (max a b) = 5 * (min a b) ∧ (max a b = 360 ∧ min a b = 72)

theorem find_numbers : ∃ a b : ℕ, is_solution a b :=
by
  sorry

end NUMINAMATH_GPT_find_numbers_l1124_112471


namespace NUMINAMATH_GPT_find_b_value_l1124_112444

theorem find_b_value : 
  ∀ (a b : ℝ), 
    (a^3 * b^4 = 2048) ∧ (a = 8) → b = Real.sqrt 2 := 
by 
sorry

end NUMINAMATH_GPT_find_b_value_l1124_112444


namespace NUMINAMATH_GPT_delta_max_success_ratio_l1124_112462

theorem delta_max_success_ratio :
  ∃ a b c d : ℕ, 
    0 < a ∧ a < b ∧ (40 * a) < (21 * b) ∧
    0 < c ∧ c < d ∧ (4 * c) < (3 * d) ∧
    b + d = 600 ∧
    (a + c) / 600 = 349 / 600 :=
by
  sorry

end NUMINAMATH_GPT_delta_max_success_ratio_l1124_112462


namespace NUMINAMATH_GPT_population_control_l1124_112430

   noncomputable def population_growth (initial_population : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
   initial_population * (1 + growth_rate / 100) ^ years

   theorem population_control {initial_population : ℝ} {threshold_population : ℝ} {growth_rate : ℝ} {years : ℕ} :
     initial_population = 1.3 ∧ threshold_population = 1.4 ∧ growth_rate = 0.74 ∧ years = 10 →
     population_growth initial_population growth_rate years < threshold_population :=
   by
     intros
     sorry
   
end NUMINAMATH_GPT_population_control_l1124_112430


namespace NUMINAMATH_GPT_sequence_is_aperiodic_l1124_112425

noncomputable def sequence_a (a : ℕ → ℕ) : Prop :=
∀ k n : ℕ, k < 2^n → a k ≠ a (k + 2^n)

theorem sequence_is_aperiodic (a : ℕ → ℕ) (h_a : sequence_a a) : ¬(∃ p : ℕ, ∀ n k : ℕ, a k = a (k + n * p)) :=
sorry

end NUMINAMATH_GPT_sequence_is_aperiodic_l1124_112425


namespace NUMINAMATH_GPT_find_a_m_l1124_112427

theorem find_a_m :
  ∃ a m : ℤ,
    (a = -2) ∧ (m = -1 ∨ m = 3) ∧ 
    (∀ x : ℝ, (a - 1) * x^2 + a * x + 1 = 0 → 
               (m^2 + m) * x^2 + 3 * m * x - 3 = 0) := sorry

end NUMINAMATH_GPT_find_a_m_l1124_112427


namespace NUMINAMATH_GPT_least_lcm_possible_l1124_112493

theorem least_lcm_possible (a b c : ℕ) (h1 : Nat.lcm a b = 24) (h2 : Nat.lcm b c = 18) : Nat.lcm a c = 12 :=
sorry

end NUMINAMATH_GPT_least_lcm_possible_l1124_112493


namespace NUMINAMATH_GPT_total_interest_obtained_l1124_112428

-- Define the interest rates and face values
def interest_16 := 0.16 * 100
def interest_12 := 0.12 * 100
def interest_20 := 0.20 * 100

-- State the theorem to be proved
theorem total_interest_obtained : 
  interest_16 + interest_12 + interest_20 = 48 :=
by
  sorry

end NUMINAMATH_GPT_total_interest_obtained_l1124_112428


namespace NUMINAMATH_GPT_interval_of_increase_l1124_112415

noncomputable def u (x : ℝ) : ℝ := x^2 - 5*x + 6

def increasing_interval (f : ℝ → ℝ) (interval : Set ℝ) : Prop :=
  ∀ (x y : ℝ), x ∈ interval → y ∈ interval → x < y → f x < f y

noncomputable def f (x : ℝ) : ℝ := Real.log (u x)

theorem interval_of_increase :
  increasing_interval f {x : ℝ | 3 < x} :=
sorry

end NUMINAMATH_GPT_interval_of_increase_l1124_112415


namespace NUMINAMATH_GPT_largest_common_divisor_l1124_112463

theorem largest_common_divisor (a b : ℕ) (h1 : a = 360) (h2 : b = 315) : 
  ∃ d : ℕ, d ∣ a ∧ d ∣ b ∧ ∀ e : ℕ, (e ∣ a ∧ e ∣ b) → e ≤ d ∧ d = 45 :=
by
  sorry

end NUMINAMATH_GPT_largest_common_divisor_l1124_112463


namespace NUMINAMATH_GPT_number_of_4_letter_words_with_vowel_l1124_112421

def is_vowel (c : Char) : Bool :=
c = 'A' ∨ c = 'E'

def count_4letter_words_with_vowels : Nat :=
  let total_words := 5^4
  let words_without_vowels := 3^4
  total_words - words_without_vowels

theorem number_of_4_letter_words_with_vowel :
  count_4letter_words_with_vowels = 544 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_number_of_4_letter_words_with_vowel_l1124_112421


namespace NUMINAMATH_GPT_ratio_alcohol_to_water_l1124_112407

-- Definitions of volume fractions for alcohol and water
def alcohol_volume_fraction : ℚ := 1 / 7
def water_volume_fraction : ℚ := 2 / 7

-- The theorem stating the ratio of alcohol to water volumes
theorem ratio_alcohol_to_water : (alcohol_volume_fraction / water_volume_fraction) = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_ratio_alcohol_to_water_l1124_112407


namespace NUMINAMATH_GPT_arith_seq_sum_l1124_112400

-- We start by defining what it means for a sequence to be arithmetic
def is_arith_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

-- We are given that a_2 = 5 and a_6 = 33 for an arithmetic sequence
variable (a : ℕ → ℤ)
variable (h_arith : is_arith_seq a)
variable (h1 : a 2 = 5)
variable (h2 : a 6 = 33)

-- The statement we want to prove
theorem arith_seq_sum (a : ℕ → ℤ) (h_arith : is_arith_seq a) (h1 : a 2 = 5) (h2 : a 6 = 33) :
  (a 3 + a 5) = 38 :=
  sorry

end NUMINAMATH_GPT_arith_seq_sum_l1124_112400


namespace NUMINAMATH_GPT_no_valid_weights_l1124_112474

theorem no_valid_weights (w_1 w_2 w_3 w_4 : ℝ) : 
  w_1 + w_2 + w_3 = 100 → w_1 + w_2 + w_4 = 101 → w_2 + w_3 + w_4 = 102 → 
  w_1 < 90 → w_2 < 90 → w_3 < 90 → w_4 < 90 → False :=
by 
  intros h1 h2 h3 hl1 hl2 hl3 hl4
  sorry

end NUMINAMATH_GPT_no_valid_weights_l1124_112474


namespace NUMINAMATH_GPT_cost_of_book_l1124_112478

theorem cost_of_book (cost_album : ℝ) (discount_rate : ℝ) (cost_CD : ℝ) (cost_book : ℝ) 
  (h1 : cost_album = 20)
  (h2 : discount_rate = 0.30)
  (h3 : cost_CD = cost_album * (1 - discount_rate))
  (h4 : cost_book = cost_CD + 4) :
  cost_book = 18 := by
  sorry

end NUMINAMATH_GPT_cost_of_book_l1124_112478


namespace NUMINAMATH_GPT_rearrange_to_rectangle_l1124_112405

-- Definition of a geometric figure and operations
structure Figure where
  parts : List (List (ℤ × ℤ)) -- List of parts represented by lists of coordinates

def is_cut_into_three_parts (fig : Figure) : Prop :=
  fig.parts.length = 3

def can_be_rearranged_to_form_rectangle (fig : Figure) : Prop := sorry

-- Initial given figure
variable (initial_figure : Figure)

-- Conditions
axiom figure_can_be_cut : is_cut_into_three_parts initial_figure
axiom cuts_not_along_grid_lines : True -- Replace with appropriate geometric operation when image is known
axiom parts_can_be_flipped : True -- Replace with operation allowing part flipping

-- Theorem to prove
theorem rearrange_to_rectangle : 
  is_cut_into_three_parts initial_figure →
  can_be_rearranged_to_form_rectangle initial_figure := 
sorry

end NUMINAMATH_GPT_rearrange_to_rectangle_l1124_112405


namespace NUMINAMATH_GPT_valid_documents_count_l1124_112426

-- Definitions based on the conditions
def total_papers : ℕ := 400
def invalid_percentage : ℝ := 0.40
def valid_percentage : ℝ := 1.0 - invalid_percentage

-- Question and answer formalized as a theorem
theorem valid_documents_count : total_papers * valid_percentage = 240 := by
  sorry

end NUMINAMATH_GPT_valid_documents_count_l1124_112426


namespace NUMINAMATH_GPT_avg_visitors_sundays_l1124_112461

-- Definitions
def days_in_month := 30
def avg_visitors_per_day_month := 750
def avg_visitors_other_days := 700
def sundays_in_month := 5
def other_days := days_in_month - sundays_in_month

-- Main statement to prove
theorem avg_visitors_sundays (S : ℕ) 
  (H1 : days_in_month = 30) 
  (H2 : avg_visitors_per_day_month = 750) 
  (H3 : avg_visitors_other_days = 700) 
  (H4 : sundays_in_month = 5) 
  (H5 : other_days = days_in_month - sundays_in_month) 
  :
  (sundays_in_month * S + other_days * avg_visitors_other_days) = avg_visitors_per_day_month * days_in_month 
  → S = 1000 :=
by 
  sorry

end NUMINAMATH_GPT_avg_visitors_sundays_l1124_112461


namespace NUMINAMATH_GPT_chef_initial_potatoes_l1124_112401

theorem chef_initial_potatoes (fries_per_potato : ℕ) (total_fries_needed : ℕ) (leftover_potatoes : ℕ) 
  (H1 : fries_per_potato = 25) 
  (H2 : total_fries_needed = 200) 
  (H3 : leftover_potatoes = 7) : 
  (total_fries_needed / fries_per_potato + leftover_potatoes = 15) :=
by
  sorry

end NUMINAMATH_GPT_chef_initial_potatoes_l1124_112401


namespace NUMINAMATH_GPT_joey_speed_on_way_back_eq_six_l1124_112482

theorem joey_speed_on_way_back_eq_six :
  ∃ (v : ℝ), 
    (∀ (d t : ℝ), 
      d = 2 ∧ t = 1 →  -- Joey runs a 2-mile distance in 1 hour
      (∀ (d_total t_avg : ℝ),
        d_total = 4 ∧ t_avg = 3 →  -- Round trip distance is 4 miles with average speed 3 mph
        (3 = 4 / (1 + 2 / v) → -- Given average speed equation
         v = 6))) := sorry

end NUMINAMATH_GPT_joey_speed_on_way_back_eq_six_l1124_112482


namespace NUMINAMATH_GPT_y_completion_time_l1124_112490

noncomputable def work_done (days : ℕ) (rate : ℚ) : ℚ := days * rate

theorem y_completion_time (X_days Y_remaining_days : ℕ) (X_rate Y_days : ℚ) :
  X_days = 40 →
  work_done 8 (1 / X_days) = 1 / 5 →
  work_done Y_remaining_days (4 / 5 / Y_remaining_days) = 4 / 5 →
  Y_days = 35 :=
by
  intros hX hX_work_done hY_work_done
  -- With the stated conditions, we should be able to conclude that Y_days is 35.
  sorry

end NUMINAMATH_GPT_y_completion_time_l1124_112490


namespace NUMINAMATH_GPT_students_like_burgers_l1124_112485

theorem students_like_burgers (total_students : ℕ) (french_fries_likers : ℕ) (both_likers : ℕ) (neither_likers : ℕ) 
    (h1 : total_students = 25) (h2 : french_fries_likers = 15) (h3 : both_likers = 6) (h4 : neither_likers = 6) : 
    (total_students - neither_likers) - (french_fries_likers - both_likers) = 10 :=
by
  -- The proof will go here.
  sorry

end NUMINAMATH_GPT_students_like_burgers_l1124_112485


namespace NUMINAMATH_GPT_fraction_of_alvin_age_l1124_112453

variable (A E F : ℚ)

-- Conditions
def edwin_older_by_six : Prop := E = A + 6
def total_age : Prop := A + E = 30.99999999
def age_relation_in_two_years : Prop := E + 2 = F * (A + 2) + 20

-- Statement to prove
theorem fraction_of_alvin_age
  (h1 : edwin_older_by_six A E)
  (h2 : total_age A E)
  (h3 : age_relation_in_two_years A E F) :
  F = 1 / 29 :=
sorry

end NUMINAMATH_GPT_fraction_of_alvin_age_l1124_112453


namespace NUMINAMATH_GPT_square_not_end_with_four_identical_digits_l1124_112470

theorem square_not_end_with_four_identical_digits (n : ℕ) (d : ℕ) :
  n = d * d → ¬ (d ≠ 0 ∧ (n % 10000 = d ^ 4)) :=
by
  sorry

end NUMINAMATH_GPT_square_not_end_with_four_identical_digits_l1124_112470


namespace NUMINAMATH_GPT_rate_mangoes_correct_l1124_112435

-- Define the conditions
def weight_apples : ℕ := 8
def rate_apples : ℕ := 70
def cost_apples := weight_apples * rate_apples

def total_payment : ℕ := 1145
def weight_mangoes : ℕ := 9
def cost_mangoes := total_payment - cost_apples

-- Define the rate per kg of mangoes
def rate_mangoes := cost_mangoes / weight_mangoes

-- Prove the rate per kg for mangoes
theorem rate_mangoes_correct : rate_mangoes = 65 := by
  -- all conditions and intermediate calculations already stated
  sorry

end NUMINAMATH_GPT_rate_mangoes_correct_l1124_112435


namespace NUMINAMATH_GPT_count_paths_word_l1124_112433

def move_right_or_down_paths (n : ℕ) : ℕ := 2^n

theorem count_paths_word (n : ℕ) (w : String) (start : Char) (end_ : Char) :
    w = "строка" ∧ start = 'C' ∧ end_ = 'A' ∧ n = 5 →
    move_right_or_down_paths n = 32 :=
by
  intro h
  cases h
  sorry

end NUMINAMATH_GPT_count_paths_word_l1124_112433


namespace NUMINAMATH_GPT_avg_highway_mpg_l1124_112410

noncomputable def highway_mpg (total_distance : ℕ) (fuel : ℕ) : ℝ :=
  total_distance / fuel
  
theorem avg_highway_mpg :
  highway_mpg 305 25 = 12.2 :=
by
  sorry

end NUMINAMATH_GPT_avg_highway_mpg_l1124_112410


namespace NUMINAMATH_GPT_product_of_fractions_l1124_112434

theorem product_of_fractions :
  (1 / 2) * (3 / 5) * (5 / 6) = 1 / 4 := 
by
  sorry

end NUMINAMATH_GPT_product_of_fractions_l1124_112434


namespace NUMINAMATH_GPT_range_of_a_l1124_112417

theorem range_of_a (h : ∃ x : ℝ, x > 0 ∧ 2^x * (x - a) < 1) : a > -1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1124_112417


namespace NUMINAMATH_GPT_handshakes_exchanged_l1124_112443

-- Let n be the number of couples
noncomputable def num_couples := 7

-- Total number of people at the gathering
noncomputable def total_people := num_couples * 2

-- Number of people each person shakes hands with
noncomputable def handshakes_per_person := total_people - 2

-- Total number of unique handshakes
noncomputable def total_handshakes := total_people * handshakes_per_person / 2

theorem handshakes_exchanged :
  total_handshakes = 77 :=
by
  sorry

end NUMINAMATH_GPT_handshakes_exchanged_l1124_112443


namespace NUMINAMATH_GPT_geom_seq_sum_half_l1124_112498

theorem geom_seq_sum_half (a : ℕ → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_sum : ∃ L, L = ∑' n, a n ∧ L = 1 / 2) (h_abs : |q| < 1) :
  a 0 ∈ (Set.Ioo 0 (1 / 2)) ∪ (Set.Ioo (1 / 2) 1) :=
sorry

end NUMINAMATH_GPT_geom_seq_sum_half_l1124_112498


namespace NUMINAMATH_GPT_find_n_l1124_112414

theorem find_n (n : ℕ) (h : 2 ^ 3 * 5 * n = Nat.factorial 10) : n = 45360 :=
sorry

end NUMINAMATH_GPT_find_n_l1124_112414


namespace NUMINAMATH_GPT_parallel_lines_slope_equal_l1124_112449

theorem parallel_lines_slope_equal (m : ℝ) : 
  (∃ m : ℝ, -(m+4)/(m+2) = -(m+2)/(m+1)) → m = 0 := 
by
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_equal_l1124_112449


namespace NUMINAMATH_GPT_problem1_problem2_l1124_112438

-- Theorem 1: Given a^2 - b^2 = 1940:
theorem problem1 
  (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_unit_digit : a^5 % 10 = b^5 % 10) : 
  a^2 - b^2 = 1940 → 
  (a = 102 ∧ b = 92) := 
by 
  sorry

-- Theorem 2: Given a^2 - b^2 = 1920:
theorem problem2 
  (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_unit_digit : a^5 % 10 = b^5 % 10) : 
  a^2 - b^2 = 1920 → 
  (a = 101 ∧ b = 91) ∨ 
  (a = 58 ∧ b = 38) ∨ 
  (a = 47 ∧ b = 17) ∨ 
  (a = 44 ∧ b = 4) := 
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1124_112438


namespace NUMINAMATH_GPT_distance_covered_l1124_112476

-- Define the conditions
def speed_still_water : ℕ := 30   -- 30 kmph
def current_speed : ℕ := 6        -- 6 kmph
def time_downstream : ℕ := 24     -- 24 seconds

-- Proving the distance covered downstream
theorem distance_covered (s_still s_current t : ℕ) (h_s_still : s_still = speed_still_water) (h_s_current : s_current = current_speed) (h_t : t = time_downstream):
  (s_still + s_current) * 1000 / 3600 * t = 240 :=
by sorry

end NUMINAMATH_GPT_distance_covered_l1124_112476


namespace NUMINAMATH_GPT_exist_consecutive_days_20_games_l1124_112439

theorem exist_consecutive_days_20_games 
  (a : ℕ → ℕ)
  (h_daily : ∀ n, a (n + 1) - a n ≥ 1)
  (h_weekly : ∀ n, a (n + 7) - a n ≤ 12) :
  ∃ i j, i < j ∧ a j - a i = 20 := by 
  sorry

end NUMINAMATH_GPT_exist_consecutive_days_20_games_l1124_112439


namespace NUMINAMATH_GPT_sum_of_areas_of_two_parks_l1124_112448

theorem sum_of_areas_of_two_parks :
  let side1 := 11
  let side2 := 5
  let area1 := side1 * side1
  let area2 := side2 * side2
  area1 + area2 = 146 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_areas_of_two_parks_l1124_112448


namespace NUMINAMATH_GPT_classroom_students_count_l1124_112469

-- Definitions from the conditions
def students (C S Sh : ℕ) : Prop :=
  S = 2 * C ∧
  S = Sh + 8 ∧
  Sh = C + 19

-- Proof statement
theorem classroom_students_count (C S Sh : ℕ) 
  (h : students C S Sh) : 3 * C = 81 :=
by
  sorry

end NUMINAMATH_GPT_classroom_students_count_l1124_112469


namespace NUMINAMATH_GPT_pyramid_volume_QEFGH_l1124_112487

noncomputable def volume_of_pyramid (EF FG QE : ℝ) : ℝ :=
  (1 / 3) * EF * FG * QE

theorem pyramid_volume_QEFGH :
  let EF := 10
  let FG := 5
  let QE := 9
  volume_of_pyramid EF FG QE = 150 := by
  sorry

end NUMINAMATH_GPT_pyramid_volume_QEFGH_l1124_112487


namespace NUMINAMATH_GPT_function_equality_l1124_112494

theorem function_equality (f : ℝ → ℝ)
  (hf : ∀ x : ℝ, f (2 * x + 1) = 2 * x^2 + 1) :
  ∀ x : ℝ, f x = (1/2) * x^2 - x + (3/2) :=
by
  sorry

end NUMINAMATH_GPT_function_equality_l1124_112494


namespace NUMINAMATH_GPT_solve_eqs_l1124_112459

theorem solve_eqs (x y : ℝ) 
  (h1 : x^2 + y^2 = 2)
  (h2 : x^2 / (2 - y) + y^2 / (2 - x) = 2) :
  x = 1 ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_eqs_l1124_112459


namespace NUMINAMATH_GPT_reciprocal_expression_equals_two_l1124_112466

theorem reciprocal_expression_equals_two (x y : ℝ) (h : x * y = 1) : 
  (x + 1 / y) * (2 * y - 1 / x) = 2 := by
  sorry

end NUMINAMATH_GPT_reciprocal_expression_equals_two_l1124_112466


namespace NUMINAMATH_GPT_floor_sum_correct_l1124_112404

def floor_sum_1_to_24 := 
  let sum := (3 * 1) + (5 * 2) + (7 * 3) + (9 * 4)
  sum

theorem floor_sum_correct : floor_sum_1_to_24 = 70 := by
  sorry

end NUMINAMATH_GPT_floor_sum_correct_l1124_112404


namespace NUMINAMATH_GPT_equal_roots_condition_l1124_112424

theorem equal_roots_condition (m : ℝ) :
  (m = 2 ∨ m = (9 + Real.sqrt 57) / 8 ∨ m = (9 - Real.sqrt 57) / 8) →
  ∃ a b c : ℝ, 
  (∀ x : ℝ, (a * x ^ 2 + b * x + c = 0) ↔
  (x * (x - 3) - (m + 2)) / ((x - 3) * (m - 2)) = x / m) ∧
  (b^2 - 4 * a * c = 0) :=
sorry

end NUMINAMATH_GPT_equal_roots_condition_l1124_112424


namespace NUMINAMATH_GPT_largest_perimeter_polygons_meeting_at_A_l1124_112489

theorem largest_perimeter_polygons_meeting_at_A
  (n : ℕ) 
  (r : ℝ)
  (h1 : n ≥ 3)
  (h2 : 2 * 180 * (n - 2) / n + 60 = 360) :
  2 * n * 2 = 24 := 
by
  sorry

end NUMINAMATH_GPT_largest_perimeter_polygons_meeting_at_A_l1124_112489


namespace NUMINAMATH_GPT_line_through_points_on_parabola_l1124_112406

theorem line_through_points_on_parabola 
  (x1 y1 x2 y2 : ℝ)
  (h_parabola_A : y1^2 = 4 * x1)
  (h_parabola_B : y2^2 = 4 * x2)
  (h_midpoint : (x1 + x2) / 2 = 2 ∧ (y1 + y2) / 2 = 2) :
  ∃ (m b : ℝ), m = 1 ∧ b = 2 ∧ (∀ x y : ℝ, y = m * x + b ↔ x - y = 0) :=
sorry

end NUMINAMATH_GPT_line_through_points_on_parabola_l1124_112406


namespace NUMINAMATH_GPT_sum_coeff_expansion_l1124_112472

theorem sum_coeff_expansion (x y : ℝ) : 
  (x + 2 * y)^4 = 81 := sorry

end NUMINAMATH_GPT_sum_coeff_expansion_l1124_112472


namespace NUMINAMATH_GPT_sin_gt_cos_range_l1124_112486

theorem sin_gt_cos_range (x : ℝ) : 
  0 < x ∧ x < 2 * Real.pi → (Real.sin x > Real.cos x ↔ (Real.pi / 4 < x ∧ x < 5 * Real.pi / 4)) := by
  sorry

end NUMINAMATH_GPT_sin_gt_cos_range_l1124_112486


namespace NUMINAMATH_GPT_ratio_diminished_to_total_l1124_112445

-- Definitions related to the conditions
def N := 240
def P := 60
def fifth_part_increased (N : ℕ) : ℕ := (N / 5) + 6
def part_diminished (P : ℕ) : ℕ := P - 6

-- The proof problem statement
theorem ratio_diminished_to_total 
  (h1 : fifth_part_increased N = part_diminished P) : 
  (P - 6) / N = 9 / 40 :=
by sorry

end NUMINAMATH_GPT_ratio_diminished_to_total_l1124_112445


namespace NUMINAMATH_GPT_remainder_of_power_is_41_l1124_112423

theorem remainder_of_power_is_41 : 
  ∀ (n k : ℕ), n = 2019 → k = 2018 → (n^k) % 100 = 41 :=
  by 
    intros n k hn hk 
    rw [hn, hk] 
    exact sorry

end NUMINAMATH_GPT_remainder_of_power_is_41_l1124_112423


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1124_112416

def a : Int := 1
def b : Int := -2

theorem simplify_and_evaluate :
  ((a * b - 3 * a^2) - 2 * b^2 - 5 * a * b - (a^2 - 2 * a * b)) = -8 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1124_112416


namespace NUMINAMATH_GPT_multiple_of_3_l1124_112479

theorem multiple_of_3 (a b : ℤ) (h1 : ∃ m : ℤ, a = 3 * m) (h2 : ∃ n : ℤ, b = 9 * n) : ∃ k : ℤ, a + b = 3 * k :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_3_l1124_112479


namespace NUMINAMATH_GPT_factor_of_7_l1124_112488

theorem factor_of_7 (a b : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : 7 ∣ (a + 2 * b)) : 7 ∣ (100 * a + 11 * b) :=
by sorry

end NUMINAMATH_GPT_factor_of_7_l1124_112488


namespace NUMINAMATH_GPT_find_r_l1124_112403

variable (a b c r : ℝ)

theorem find_r (h1 : a * (b - c) / (b * (c - a)) = r)
               (h2 : b * (c - a) / (c * (b - a)) = r)
               (h3 : r > 0) : 
               r = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_r_l1124_112403


namespace NUMINAMATH_GPT_no_zonk_probability_l1124_112491

theorem no_zonk_probability (Z C G : ℕ) (total_boxes : ℕ := 3) (tables : ℕ := 3)
  (no_zonk_prob : ℚ := 2 / 3) : (no_zonk_prob ^ tables) = 8 / 27 :=
by
  -- Here we would prove the theorem, but for the purpose of this task, we skip the proof.
  sorry

end NUMINAMATH_GPT_no_zonk_probability_l1124_112491


namespace NUMINAMATH_GPT_geometric_series_sum_l1124_112432

theorem geometric_series_sum {a r : ℚ} (n : ℕ) (h_a : a = 3/4) (h_r : r = 3/4) (h_n : n = 8) : 
       a * (1 - r^n) / (1 - r) = 176925 / 65536 :=
by
  -- Utilizing the provided conditions
  have h_a := h_a
  have h_r := h_r
  have h_n := h_n
  -- Proving the theorem using sorry as a placeholder for the detailed steps
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1124_112432


namespace NUMINAMATH_GPT_count_3digit_numbers_div_by_13_l1124_112422

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end NUMINAMATH_GPT_count_3digit_numbers_div_by_13_l1124_112422
