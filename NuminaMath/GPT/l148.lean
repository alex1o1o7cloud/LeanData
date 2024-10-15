import Mathlib

namespace NUMINAMATH_GPT_cos_A_value_find_c_l148_14836

theorem cos_A_value (a b c A B C : ℝ) (h : 2 * a * Real.cos A = c * Real.cos B + b * Real.cos C) : 
  Real.cos A = 1 / 2 := 
  sorry

theorem find_c (B C : ℝ) (A : B + C = Real.pi - A) (h1 : 1 = 1) 
  (h2 : Real.cos (B / 2) * Real.cos (B / 2) + Real.cos (C / 2) * Real.cos (C / 2) = 1 + Real.sqrt (3) / 4) : 
  ∃ (c : ℝ), c = 2 * Real.sqrt (3) / 3 ∨ c = Real.sqrt (3) / 3 := 
  sorry

end NUMINAMATH_GPT_cos_A_value_find_c_l148_14836


namespace NUMINAMATH_GPT_least_positive_integer_special_property_l148_14837

/-- 
  Prove that 9990 is the least positive integer whose digits sum to a multiple of 27 
  and the number itself is not a multiple of 27.
-/
theorem least_positive_integer_special_property : ∃ n : ℕ, 
  n > 0 ∧ 
  (Nat.digits 10 n).sum % 27 = 0 ∧ 
  n % 27 ≠ 0 ∧ 
  ∀ m : ℕ, (m > 0 ∧ (Nat.digits 10 m).sum % 27 = 0 ∧ m % 27 ≠ 0 → n ≤ m) := 
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_special_property_l148_14837


namespace NUMINAMATH_GPT_acute_angle_comparison_l148_14868

noncomputable def f : ℝ → ℝ := sorry

def even_function (f : ℝ → ℝ) :=
  ∀ x, f x = f (-x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) :=
  ∀ x, f (x + p) = f x

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem acute_angle_comparison (A B : ℝ)
  (hA : 0 < A ∧ A < π / 2)
  (hB : 0 < B ∧ B < π / 2)
  (f_even : even_function f)
  (f_periodic : ∀ x, f (x + 1) + f x = 0)
  (f_increasing : increasing_on_interval f 3 4) :
  f (Real.sin A) < f (Real.cos B) :=
sorry

end NUMINAMATH_GPT_acute_angle_comparison_l148_14868


namespace NUMINAMATH_GPT_clock_minutes_to_correct_time_l148_14839

def slow_clock_time_ratio : ℚ := 14 / 15

noncomputable def slow_clock_to_correct_time (slow_clock_time : ℚ) : ℚ :=
  slow_clock_time / slow_clock_time_ratio

theorem clock_minutes_to_correct_time :
  slow_clock_to_correct_time 14 = 15 :=
by
  sorry

end NUMINAMATH_GPT_clock_minutes_to_correct_time_l148_14839


namespace NUMINAMATH_GPT_original_population_l148_14870

-- Define the conditions
def population_increase (n : ℕ) : ℕ := n + 1200
def population_decrease (p : ℕ) : ℕ := (89 * p) / 100
def final_population (n : ℕ) : ℕ := population_decrease (population_increase n)

-- Claim that needs to be proven
theorem original_population (n : ℕ) (H : final_population n = n - 32) : n = 10000 :=
by
  sorry

end NUMINAMATH_GPT_original_population_l148_14870


namespace NUMINAMATH_GPT_number_of_lists_l148_14890

theorem number_of_lists (n k : ℕ) (h_n : n = 15) (h_k : k = 4) : (n ^ k) = 50625 := by
  have : 15 ^ 4 = 50625 := by norm_num
  rwa [h_n, h_k]

end NUMINAMATH_GPT_number_of_lists_l148_14890


namespace NUMINAMATH_GPT_closest_fraction_to_medals_won_l148_14897

theorem closest_fraction_to_medals_won :
  let won_ratio : ℚ := 35 / 225
  let choices : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]
  (closest : ℚ) = 1 / 6 → 
  (closest_in_choices : closest ∈ choices) →
  ∀ choice ∈ choices, abs ((7 / 45) - (1 / 6)) ≤ abs ((7 / 45) - choice) :=
by
  let won_ratio := 7 / 45
  let choices := [1/5, 1/6, 1/7, 1/8, 1/9]
  let closest := 1 / 6
  have closest_in_choices : closest ∈ choices := sorry
  intro choice h_choice_in_choices
  sorry

end NUMINAMATH_GPT_closest_fraction_to_medals_won_l148_14897


namespace NUMINAMATH_GPT_jenna_owes_amount_l148_14886

theorem jenna_owes_amount (initial_bill : ℝ) (rate : ℝ) (times : ℕ) : 
  initial_bill = 400 → rate = 0.02 → times = 3 → 
  owed_amount = (400 * (1 + 0.02)^3) := 
by
  intros
  sorry

end NUMINAMATH_GPT_jenna_owes_amount_l148_14886


namespace NUMINAMATH_GPT_sum_nonnegative_reals_l148_14833

variable {x y z : ℝ}

theorem sum_nonnegative_reals (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h1 : x^2 + y^2 + z^2 = 52) (h2 : x * y + y * z + z * x = 24) : 
  x + y + z = 10 := 
by sorry

end NUMINAMATH_GPT_sum_nonnegative_reals_l148_14833


namespace NUMINAMATH_GPT_expression_simplification_l148_14851

noncomputable def given_expression : ℝ :=
  1 / ((1 / (Real.sqrt 2 + 2)) + (3 / (2 * Real.sqrt 3 - 1)))

noncomputable def expected_expression : ℝ :=
  1 / (25 - 11 * Real.sqrt 2 + 6 * Real.sqrt 3)

theorem expression_simplification :
  given_expression = expected_expression :=
by
  sorry

end NUMINAMATH_GPT_expression_simplification_l148_14851


namespace NUMINAMATH_GPT_contradiction_even_odd_l148_14809

theorem contradiction_even_odd (a b c : ℕ) :
  (∃ x y z, (x = a ∧ y = b ∧ z = c) ∧ (¬((x % 2 = 0 ∧ y % 2 ≠ 0 ∧ z % 2 ≠ 0) ∨ 
                                          (x % 2 ≠ 0 ∧ y % 2 = 0 ∧ z % 2 ≠ 0) ∨ 
                                          (x % 2 ≠ 0 ∧ y % 2 ≠ 0 ∧ z % 2 = 0)))) → false :=
by
  sorry

end NUMINAMATH_GPT_contradiction_even_odd_l148_14809


namespace NUMINAMATH_GPT_expression_D_is_odd_l148_14825

namespace ProofProblem

def is_odd (n : ℤ) : Prop :=
  ∃ k : ℤ, n = 2 * k + 1

theorem expression_D_is_odd :
  is_odd (3 + 5 + 1) :=
by
  sorry

end ProofProblem

end NUMINAMATH_GPT_expression_D_is_odd_l148_14825


namespace NUMINAMATH_GPT_range_of_a_l148_14863

theorem range_of_a (a : ℝ) :
  (∀ θ : ℝ, 0 < θ ∧ θ < (π / 2) → a ≤ 1 / Real.sin θ + 1 / Real.cos θ) ↔ a ≤ 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l148_14863


namespace NUMINAMATH_GPT_teacher_allocation_l148_14884

theorem teacher_allocation :
  ∃ n : ℕ, n = 150 ∧ 
  (∀ t1 t2 t3 t4 t5 : Prop, -- represent the five teachers
    ∃ s1 s2 s3 : Prop, -- represent the three schools
      s1 ∧ s2 ∧ s3 ∧ -- each school receives at least one teacher
        ((t1 ∨ t2 ∨ t3 ∨ t4 ∨ t5) ∧ -- allocation condition
         (t1 ∨ t2 ∨ t3 ∨ t4 ∨ t5) ∧
         (t1 ∨ t2 ∨ t3 ∨ t4 ∨ t5))) := sorry

end NUMINAMATH_GPT_teacher_allocation_l148_14884


namespace NUMINAMATH_GPT_scientific_notation_l148_14819

theorem scientific_notation :
  56.9 * 10^9 = 5.69 * 10^(10 - 1) :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_l148_14819


namespace NUMINAMATH_GPT_isosceles_triangle_base_angle_l148_14896

theorem isosceles_triangle_base_angle (α : ℕ) (base_angle : ℕ) 
  (hα : α = 40) (hsum : α + 2 * base_angle = 180) : 
  base_angle = 70 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_base_angle_l148_14896


namespace NUMINAMATH_GPT_soccer_lineup_count_l148_14827

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem soccer_lineup_count : 
  let total_players := 18
  let goalies := 1
  let defenders := 6
  let forwards := 4
  18 * choose 17 6 * choose 11 4 = 73457760 :=
by
  sorry

end NUMINAMATH_GPT_soccer_lineup_count_l148_14827


namespace NUMINAMATH_GPT_smaller_cuboid_length_l148_14871

theorem smaller_cuboid_length
  (L : ℝ)
  (h1 : 32 * (L * 4 * 3) = 16 * 10 * 12) :
  L = 5 :=
by
  sorry

end NUMINAMATH_GPT_smaller_cuboid_length_l148_14871


namespace NUMINAMATH_GPT_minimum_value_of_f_l148_14889

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + 3 * x + 3) + Real.sqrt (x^2 - 3 * x + 3)

theorem minimum_value_of_f : (∃ x : ℝ, ∀ y : ℝ, f x ≤ f y) ∧ f 0 = 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l148_14889


namespace NUMINAMATH_GPT_sum_of_three_numbers_l148_14852

theorem sum_of_three_numbers (x : ℝ) (a b c : ℝ) (h1 : a = 5 * x) (h2 : b = x) (h3 : c = 4 * x) (h4 : c = 400) :
  a + b + c = 1000 := by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l148_14852


namespace NUMINAMATH_GPT_rectangle_circle_area_ratio_l148_14858

theorem rectangle_circle_area_ratio {d : ℝ} (h : d > 0) :
  let A_rectangle := 2 * d * d
  let A_circle := (π * d^2) / 4
  (A_rectangle / A_circle) = (8 / π) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_circle_area_ratio_l148_14858


namespace NUMINAMATH_GPT_correct_calculation_l148_14842

-- Define the statements for each option
def option_A (a : ℕ) : Prop := (a^2)^3 = a^5
def option_B (a : ℕ) : Prop := a^3 + a^2 = a^6
def option_C (a : ℕ) : Prop := a^6 / a^3 = a^3
def option_D (a : ℕ) : Prop := a^3 * a^2 = a^6

-- Define the theorem stating that option C is the only correct one
theorem correct_calculation (a : ℕ) : ¬option_A a ∧ ¬option_B a ∧ option_C a ∧ ¬option_D a := by
  sorry

end NUMINAMATH_GPT_correct_calculation_l148_14842


namespace NUMINAMATH_GPT_ab_gt_ac_l148_14876

variables {a b c : ℝ}

theorem ab_gt_ac (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c :=
sorry

end NUMINAMATH_GPT_ab_gt_ac_l148_14876


namespace NUMINAMATH_GPT_increasing_intervals_g_l148_14804

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

noncomputable def g (x : ℝ) : ℝ := f (2 - x^2)

theorem increasing_intervals_g : 
  (∀ x ∈ Set.Icc (-1 : ℝ) (0 : ℝ), ∀ y ∈ Set.Icc (-1 : ℝ) (0 : ℝ), x ≤ y → g x ≤ g y) ∧
  (∀ x ∈ Set.Ici (1 : ℝ), ∀ y ∈ Set.Ici (1 : ℝ), x ≤ y → g x ≤ g y) := 
sorry

end NUMINAMATH_GPT_increasing_intervals_g_l148_14804


namespace NUMINAMATH_GPT_cubic_conversion_l148_14816

theorem cubic_conversion (h : 1 = 100) : 1 = 1000000 :=
by
  sorry

end NUMINAMATH_GPT_cubic_conversion_l148_14816


namespace NUMINAMATH_GPT_sequence_problem_l148_14853

theorem sequence_problem (S : ℕ → ℚ) (a : ℕ → ℚ) (h : ∀ n, S n + a n = 2 * n) :
  a 1 = 1 ∧ a 2 = 3 / 2 ∧ a 3 = 7 / 4 ∧ a 4 = 15 / 8 ∧ 
  (∀ n : ℕ, n > 0 → a n = (2^n - 1) / 2^(n-1)) :=
by
  sorry

end NUMINAMATH_GPT_sequence_problem_l148_14853


namespace NUMINAMATH_GPT_distinct_ordered_pairs_count_l148_14893

theorem distinct_ordered_pairs_count :
  ∃ S : Finset (ℕ × ℕ), 
    (∀ p ∈ S, 1 ≤ p.1 ∧ 1 ≤ p.2 ∧ (1 / (p.1 : ℚ) + 1 / (p.2 : ℚ) = 1 / 6)) ∧
    S.card = 9 := 
by
  sorry

end NUMINAMATH_GPT_distinct_ordered_pairs_count_l148_14893


namespace NUMINAMATH_GPT_cos_sum_identity_l148_14877

theorem cos_sum_identity (α : ℝ) (h_cos : Real.cos α = 3 / 5) (h_alpha : 0 < α ∧ α < Real.pi / 2) :
  Real.cos (α + Real.pi / 3) = (3 - 4 * Real.sqrt 3) / 10 :=
by
  sorry

end NUMINAMATH_GPT_cos_sum_identity_l148_14877


namespace NUMINAMATH_GPT_westgate_high_school_chemistry_l148_14873

theorem westgate_high_school_chemistry :
  ∀ (total_players physics_both physics : ℕ),
    total_players = 15 →
    physics_both = 3 →
    physics = 8 →
    (total_players - (physics - physics_both)) - physics_both = 10 := by
  intros total_players physics_both physics h1 h2 h3
  sorry

end NUMINAMATH_GPT_westgate_high_school_chemistry_l148_14873


namespace NUMINAMATH_GPT_amoeba_after_ten_days_l148_14824

def amoeba_count (n : ℕ) : ℕ := 
  3^n

theorem amoeba_after_ten_days : amoeba_count 10 = 59049 := 
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_amoeba_after_ten_days_l148_14824


namespace NUMINAMATH_GPT_find_k_point_verification_l148_14826

-- Definition of the linear function
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x + 3

-- Condition that the point (2, 7) lies on the graph of the linear function
def passes_through (k : ℝ) : Prop := linear_function k 2 = 7

-- The actual proof task to verify the value of k
theorem find_k : ∃ k : ℝ, passes_through k ∧ k = 2 :=
by
  sorry

-- The condition that the point (-2, 1) is not on the graph with k = 2
def point_not_on_graph : Prop := ¬ (linear_function 2 (-2) = 1)

-- The actual proof task to verify the point (-2, 1) is not on the graph of y = 2x + 3
theorem point_verification : point_not_on_graph :=
by
  sorry

end NUMINAMATH_GPT_find_k_point_verification_l148_14826


namespace NUMINAMATH_GPT_math_proof_problem_l148_14847

open Set

noncomputable def alpha : ℝ := (3 - Real.sqrt 5) / 2

theorem math_proof_problem (α_pos : 0 < α) (α_lt_delta : α < alpha) :
  ∃ n p : ℕ, p > α * 2^n ∧ ∃ S T : Finset (Fin n) → Finset (Fin n), (∀ i j, (S i) ∩ (T j) ≠ ∅) :=
  sorry

end NUMINAMATH_GPT_math_proof_problem_l148_14847


namespace NUMINAMATH_GPT_measure_six_liters_l148_14838

-- Given conditions as constants
def container_capacity : ℕ := 40
def ten_liter_bucket_capacity : ℕ := 10
def nine_liter_jug_capacity : ℕ := 9
def five_liter_jug_capacity : ℕ := 5

-- Goal: Measure out exactly 6 liters of milk using the above containers
theorem measure_six_liters (container : ℕ) (ten_bucket : ℕ) (nine_jug : ℕ) (five_jug : ℕ) :
  container = 40 →
  ten_bucket ≤ 10 →
  nine_jug ≤ 9 →
  five_jug ≤ 5 →
  ∃ (sequence_of_steps : ℕ → ℕ) (final_ten_bucket : ℕ),
    final_ten_bucket = 6 ∧ final_ten_bucket ≤ ten_bucket :=
by
  intro hcontainer hten_bucket hnine_jug hfive_jug
  sorry

end NUMINAMATH_GPT_measure_six_liters_l148_14838


namespace NUMINAMATH_GPT_license_plate_palindrome_probability_l148_14883

-- Define the two-letter palindrome probability
def prob_two_letter_palindrome : ℚ := 1 / 26

-- Define the four-digit palindrome probability
def prob_four_digit_palindrome : ℚ := 1 / 100

-- Define the joint probability of both two-letter and four-digit palindrome
def prob_joint_palindrome : ℚ := prob_two_letter_palindrome * prob_four_digit_palindrome

-- Define the probability of at least one palindrome using Inclusion-Exclusion
def prob_at_least_one_palindrome : ℚ := prob_two_letter_palindrome + prob_four_digit_palindrome - prob_joint_palindrome

-- Convert the probability to the form of sum of two integers
def sum_of_integers : ℕ := 5 + 104

-- The final proof problem
theorem license_plate_palindrome_probability :
  (prob_at_least_one_palindrome = 5 / 104) ∧ (sum_of_integers = 109) := by
  sorry

end NUMINAMATH_GPT_license_plate_palindrome_probability_l148_14883


namespace NUMINAMATH_GPT_symmetric_origin_a_minus_b_l148_14823

noncomputable def A (a : ℝ) := (a, -2)
noncomputable def B (b : ℝ) := (4, b)
def symmetric (p q : ℝ × ℝ) : Prop := (q.1 = -p.1) ∧ (q.2 = -p.2)

theorem symmetric_origin_a_minus_b (a b : ℝ) (hA : A a = (-4, -2)) (hB : B b = (4, 2)) :
  a - b = -6 := by
  sorry

end NUMINAMATH_GPT_symmetric_origin_a_minus_b_l148_14823


namespace NUMINAMATH_GPT_ratio_diagonals_to_sides_l148_14898

-- Definition of the number of diagonals in a polygon with n sides
def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- Definition of the condition
def n : ℕ := 5

-- Proof statement that the ratio of the number of diagonals to the number of sides is 1
theorem ratio_diagonals_to_sides (n_eq_5 : n = 5) : 
  (number_of_diagonals n) / n = 1 :=
by {
  -- Proof would go here, but is omitted
  sorry
}

end NUMINAMATH_GPT_ratio_diagonals_to_sides_l148_14898


namespace NUMINAMATH_GPT_inscribed_circle_area_l148_14887

/-- Defining the inscribed circle problem and its area. -/
theorem inscribed_circle_area (l : ℝ) (h₁ : 90 = 90) (h₂ : true) : 
  ∃ r : ℝ, (r = (2 * (Real.sqrt 2 - 1) * l / Real.pi)) ∧ ((Real.pi * r ^ 2) = (12 - 8 * Real.sqrt 2) * l ^ 2 / Real.pi) :=
  sorry

end NUMINAMATH_GPT_inscribed_circle_area_l148_14887


namespace NUMINAMATH_GPT_hyperbola_eccentricity_ratio_hyperbola_condition_l148_14864

-- Part (a)
theorem hyperbola_eccentricity_ratio
  (a b c : ℝ) (h1 : c^2 = a^2 + b^2)
  (x0 y0 : ℝ) 
  (P : ℝ × ℝ) (h2 : P = (x0, y0))
  (F : ℝ × ℝ) (h3 : F = (c, 0))
  (D : ℝ) (h4 : D = a^2 / c)
  (d_PF : ℝ) (h5 : d_PF = ( (x0 - c)^2 + y0^2 )^(1/2))
  (d_PD : ℝ) (h6 : d_PD = |x0 - a^2 / c|)
  (e : ℝ) (h7 : e = c / a) :
  d_PF / d_PD = e :=
sorry

-- Part (b)
theorem hyperbola_condition
  (F_l : ℝ × ℝ) (h1 : F_l = (0, k))
  (X_l : ℝ × ℝ) (h2 : X_l = (x, l))
  (d_XF : ℝ) (h3 : d_XF = (x^2 + y^2)^(1/2))
  (d_Xl : ℝ) (h4 : d_Xl = |x - k|)
  (e : ℝ) (h5 : e > 1)
  (h6 : d_XF / d_Xl = e) :
  ∃ a b : ℝ, (x / a)^2 - (y / b)^2 = 1 :=
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_ratio_hyperbola_condition_l148_14864


namespace NUMINAMATH_GPT_pastrami_sandwich_cost_l148_14843

variable (X : ℕ)

theorem pastrami_sandwich_cost
  (h1 : 10 * X + 5 * (X + 2) = 55) :
  X + 2 = 5 := 
by
  sorry

end NUMINAMATH_GPT_pastrami_sandwich_cost_l148_14843


namespace NUMINAMATH_GPT_arithmetic_progression_cubic_eq_l148_14879

theorem arithmetic_progression_cubic_eq (x y z u : ℤ) (d : ℤ) :
  (x, y, z, u) = (3 * d, 4 * d, 5 * d, 6 * d) →
  x^3 + y^3 + z^3 = u^3 →
  ∃ d : ℤ, x = 3 * d ∧ y = 4 * d ∧ z = 5 * d ∧ u = 6 * d :=
by sorry

end NUMINAMATH_GPT_arithmetic_progression_cubic_eq_l148_14879


namespace NUMINAMATH_GPT_amount_added_to_doubled_number_l148_14821

theorem amount_added_to_doubled_number (N A : ℝ) (h1 : N = 6.0) (h2 : 2 * N + A = 17) : A = 5.0 :=
by
  sorry

end NUMINAMATH_GPT_amount_added_to_doubled_number_l148_14821


namespace NUMINAMATH_GPT_meal_combinations_count_l148_14822

/-- Define the number of menu items -/
def num_menu_items : ℕ := 15

/-- Define the number of distinct combinations of meals Maryam and Jorge can order,
    considering they may choose the same dish and distinguishing who orders what -/
theorem meal_combinations_count (maryam_dishes jorge_dishes : ℕ) : 
  maryam_dishes = num_menu_items ∧ jorge_dishes = num_menu_items → 
  maryam_dishes * jorge_dishes = 225 :=
by
  intros h
  simp only [num_menu_items] at h -- Utilize the definition of num_menu_items
  sorry

end NUMINAMATH_GPT_meal_combinations_count_l148_14822


namespace NUMINAMATH_GPT_range_of_k_tan_alpha_l148_14811

noncomputable def f (x k : Real) : Real := Real.sin x + k

theorem range_of_k (k : Real) : 
  (∃ x : Real, f x k = 1) ↔ (0 ≤ k ∧ k ≤ 2) :=
sorry

theorem tan_alpha (α k : Real) (h : α ∈ Set.Ioo (0 : Real) Real.pi) (hf : f α k = 1 / 3 + k) : 
  Real.tan α = Real.sqrt 2 / 4 :=
sorry

end NUMINAMATH_GPT_range_of_k_tan_alpha_l148_14811


namespace NUMINAMATH_GPT_algebra_identity_l148_14878

theorem algebra_identity (x y : ℝ) (h1 : x + y = 2) (h2 : x - y = 4) : x^2 - y^2 = 8 := by
  sorry

end NUMINAMATH_GPT_algebra_identity_l148_14878


namespace NUMINAMATH_GPT_g_analytical_expression_g_minimum_value_l148_14835

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 1
noncomputable def M (a : ℝ) : ℝ := if (a ≥ 1/3 ∧ a ≤ 1/2) then f a 1 else f a 3
noncomputable def N (a : ℝ) : ℝ := f a (1/a)
noncomputable def g (a : ℝ) : ℝ :=
  if a ≥ 1/3 ∧ a ≤ 1/2 then M a - N a 
  else if a > 1/2 ∧ a ≤ 1 then M a - N a
  else 0 -- outside the given interval, by definition may be kept as 0

theorem g_analytical_expression (a : ℝ) (h : 1/3 ≤ a ∧ a ≤ 1) : 
  g a = if (1/3 ≤ a ∧ a ≤ 1/2) then a + 1/a - 2 else 9 * a + 1/a - 6 := 
sorry

theorem g_minimum_value (a : ℝ) (h : 1/3 ≤ a ∧ a ≤ 1) :
  ∃ (a' : ℝ), 1/3 ≤ a' ∧ a' ≤ 1 ∧ (∀ a, 1/3 ≤ a ∧ a ≤ 1 → g a ≥ g a') ∧ g a' = 1/2 := 
sorry

end NUMINAMATH_GPT_g_analytical_expression_g_minimum_value_l148_14835


namespace NUMINAMATH_GPT_clock_hands_overlap_l148_14805

theorem clock_hands_overlap:
  ∃ x y: ℚ,
  -- Conditions
  (60 * 10 + x = 60 * 11 * 54 + 6 / 11) ∧
  (y - (5 / 60) * y = 60) ∧
  (65 * 5 / 11 = y) := sorry

end NUMINAMATH_GPT_clock_hands_overlap_l148_14805


namespace NUMINAMATH_GPT_christel_gave_andrena_l148_14850

theorem christel_gave_andrena (d m c a: ℕ) (h1: d = 20 - 2) (h2: c = 24) 
  (h3: a = c + 2) (h4: a = d + 3) : (24 - c = 5) :=
by { sorry }

end NUMINAMATH_GPT_christel_gave_andrena_l148_14850


namespace NUMINAMATH_GPT_sum_of_ages_l148_14899

theorem sum_of_ages (a b c d : ℕ) 
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
  (h7 : a * b = 24 ∨ a * c = 24 ∨ a * d = 24 ∨ b * c = 24 ∨ b * d = 24 ∨ c * d = 24)
  (h8 : a * b = 35 ∨ a * c = 35 ∨ a * d = 35 ∨ b * c = 35 ∨ b * d = 35 ∨ c * d = 35)
  (h9 : a < 10) (h10 : b < 10) (h11 : c < 10) (h12 : d < 10)
  (h13 : 0 < a) (h14 : 0 < b) (h15 : 0 < c) (h16 : 0 < d) :
  a + b + c + d = 23 := sorry

end NUMINAMATH_GPT_sum_of_ages_l148_14899


namespace NUMINAMATH_GPT_structure_cube_count_l148_14840

theorem structure_cube_count :
  let middle_layer := 16
  let other_layers := 4 * 24
  middle_layer + other_layers = 112 :=
by
  let middle_layer := 16
  let other_layers := 4 * 24
  have h : middle_layer + other_layers = 112 := by
    sorry
  exact h

end NUMINAMATH_GPT_structure_cube_count_l148_14840


namespace NUMINAMATH_GPT_mary_max_earnings_l148_14802

def regular_rate : ℝ := 8
def max_hours : ℝ := 60
def regular_hours : ℝ := 20
def overtime_rate : ℝ := regular_rate + 0.25 * regular_rate
def overtime_hours : ℝ := max_hours - regular_hours
def earnings_regular : ℝ := regular_hours * regular_rate
def earnings_overtime : ℝ := overtime_hours * overtime_rate
def total_earnings : ℝ := earnings_regular + earnings_overtime

theorem mary_max_earnings : total_earnings = 560 := by
  sorry

end NUMINAMATH_GPT_mary_max_earnings_l148_14802


namespace NUMINAMATH_GPT_find_solutions_to_system_l148_14872

theorem find_solutions_to_system (x y z : ℝ) 
    (h1 : 3 * (x^2 + y^2 + z^2) = 1) 
    (h2 : x^2 * y^2 + y^2 * z^2 + z^2 * x^2 = x * y * z * (x + y + z)^3) : 
    x = y ∧ y = z ∧ (x = 1 / 3 ∨ x = -1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_find_solutions_to_system_l148_14872


namespace NUMINAMATH_GPT_snowboard_price_after_discounts_l148_14857

noncomputable def final_snowboard_price (P_original : ℝ) (d_Friday : ℝ) (d_Monday : ℝ) : ℝ :=
  P_original * (1 - d_Friday) * (1 - d_Monday)

theorem snowboard_price_after_discounts :
  final_snowboard_price 100 0.50 0.30 = 35 :=
by 
  sorry

end NUMINAMATH_GPT_snowboard_price_after_discounts_l148_14857


namespace NUMINAMATH_GPT_determine_c_for_quadratic_eq_l148_14882

theorem determine_c_for_quadratic_eq (x1 x2 c : ℝ) 
  (h1 : x1 + x2 = 2)
  (h2 : x1 * x2 = c)
  (h3 : 7 * x2 - 4 * x1 = 47) : 
  c = -15 :=
sorry

end NUMINAMATH_GPT_determine_c_for_quadratic_eq_l148_14882


namespace NUMINAMATH_GPT_product_of_numbers_l148_14806

theorem product_of_numbers :
  ∃ (a b c : ℚ), a + b + c = 30 ∧
                 a = 2 * (b + c) ∧
                 b = 5 * c ∧
                 a + c = 22 ∧
                 a * b * c = 2500 / 9 :=
by
  sorry

end NUMINAMATH_GPT_product_of_numbers_l148_14806


namespace NUMINAMATH_GPT_factorial_division_l148_14812

theorem factorial_division (n : ℕ) (h : n = 9) : n.factorial / (n - 1).factorial = 9 :=
by 
  rw [h]
  sorry

end NUMINAMATH_GPT_factorial_division_l148_14812


namespace NUMINAMATH_GPT_conditional_probability_l148_14844

def P (event : ℕ → Prop) : ℝ := sorry

def A (n : ℕ) : Prop := n = 10000
def B (n : ℕ) : Prop := n = 15000

theorem conditional_probability :
  P A = 0.80 →
  P B = 0.60 →
  P B / P A = 0.75 :=
by
  intros hA hB
  sorry

end NUMINAMATH_GPT_conditional_probability_l148_14844


namespace NUMINAMATH_GPT_remaining_distance_l148_14807

theorem remaining_distance (speed time distance_covered total_distance remaining_distance : ℕ) 
  (h1 : speed = 60) 
  (h2 : time = 2) 
  (h3 : total_distance = 300)
  (h4 : distance_covered = speed * time) 
  (h5 : remaining_distance = total_distance - distance_covered) : 
  remaining_distance = 180 := 
by
  sorry

end NUMINAMATH_GPT_remaining_distance_l148_14807


namespace NUMINAMATH_GPT_average_stoppage_time_l148_14881

def bus_a_speed_excluding_stoppages := 54 -- kmph
def bus_a_speed_including_stoppages := 45 -- kmph

def bus_b_speed_excluding_stoppages := 60 -- kmph
def bus_b_speed_including_stoppages := 50 -- kmph

def bus_c_speed_excluding_stoppages := 72 -- kmph
def bus_c_speed_including_stoppages := 60 -- kmph

theorem average_stoppage_time :
  (bus_a_speed_excluding_stoppages - bus_a_speed_including_stoppages) / bus_a_speed_excluding_stoppages * 60
  + (bus_b_speed_excluding_stoppages - bus_b_speed_including_stoppages) / bus_b_speed_excluding_stoppages * 60
  + (bus_c_speed_excluding_stoppages - bus_c_speed_including_stoppages) / bus_c_speed_excluding_stoppages * 60
  = 30 / 3 :=
  by sorry

end NUMINAMATH_GPT_average_stoppage_time_l148_14881


namespace NUMINAMATH_GPT_rancher_loss_l148_14834

-- Define the necessary conditions
def initial_head_of_cattle := 340
def original_total_price := 204000
def cattle_died := 172
def price_reduction_per_head := 150

-- Define the original and new prices per head
def original_price_per_head := original_total_price / initial_head_of_cattle
def new_price_per_head := original_price_per_head - price_reduction_per_head

-- Define the number of remaining cattle
def remaining_cattle := initial_head_of_cattle - cattle_died

-- Define the total amount at the new price
def total_amount_new_price := new_price_per_head * remaining_cattle

-- Define the loss
def loss := original_total_price - total_amount_new_price

-- Prove that the loss is $128,400
theorem rancher_loss : loss = 128400 := by
  sorry

end NUMINAMATH_GPT_rancher_loss_l148_14834


namespace NUMINAMATH_GPT_fraction_to_decimal_l148_14875

theorem fraction_to_decimal : (5 : ℚ) / 8 = 0.625 := by
  -- Prove that the fraction 5/8 equals the decimal 0.625
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l148_14875


namespace NUMINAMATH_GPT_machine_c_more_bottles_l148_14815

theorem machine_c_more_bottles (A B C : ℕ) 
  (hA : A = 12)
  (hB : B = A - 2)
  (h_total : 10 * A + 10 * B + 10 * C = 370) :
  C - B = 5 :=
by
  sorry

end NUMINAMATH_GPT_machine_c_more_bottles_l148_14815


namespace NUMINAMATH_GPT_chewing_gums_count_l148_14841

-- Given conditions
def num_chocolate_bars : ℕ := 55
def num_candies : ℕ := 40
def total_treats : ℕ := 155

-- Definition to be proven
def num_chewing_gums : ℕ := total_treats - (num_chocolate_bars + num_candies)

-- Theorem statement
theorem chewing_gums_count : num_chewing_gums = 60 :=
by 
  -- here would be the proof steps, but it's omitted as per the instruction
  sorry

end NUMINAMATH_GPT_chewing_gums_count_l148_14841


namespace NUMINAMATH_GPT_average_xyz_l148_14888

theorem average_xyz (x y z : ℝ) (h1 : x = 3) (h2 : y = 2 * x) (h3 : z = 3 * y) : 
  (x + y + z) / 3 = 9 :=
by
  sorry

end NUMINAMATH_GPT_average_xyz_l148_14888


namespace NUMINAMATH_GPT_birds_remaining_l148_14891

variable (initial_birds : ℝ) (birds_flew_away : ℝ)

theorem birds_remaining (h1 : initial_birds = 12.0) (h2 : birds_flew_away = 8.0) : initial_birds - birds_flew_away = 4.0 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_birds_remaining_l148_14891


namespace NUMINAMATH_GPT_general_term_formula_l148_14854

theorem general_term_formula (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = 3^n - 1) →
  (∀ n, n ≥ 2 → a n = S n - S (n - 1)) →
  a 1 = 2 →
  ∀ n, a n = 2 * 3^(n - 1) :=
by
    intros hS ha h1 n
    sorry

end NUMINAMATH_GPT_general_term_formula_l148_14854


namespace NUMINAMATH_GPT_max_distance_between_sparkling_points_l148_14818

theorem max_distance_between_sparkling_points (a₁ b₁ a₂ b₂ : ℝ) 
  (h₁ : a₁^2 + b₁^2 = 1) (h₂ : a₂^2 + b₂^2 = 1) :
  ∃ d, d = 2 ∧ ∀ (x y : ℝ), x = a₂ - a₁ ∧ y = b₂ - b₁ → (x ^ 2 + y ^ 2 = d ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_max_distance_between_sparkling_points_l148_14818


namespace NUMINAMATH_GPT_max_value_of_a_l148_14862

theorem max_value_of_a :
  ∃ b : ℤ, ∃ (a : ℝ), 
    (a = 30285) ∧
    (a * b^2 / (a + 2 * b) = 2019) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_a_l148_14862


namespace NUMINAMATH_GPT_norma_initial_cards_l148_14848

def initial_card_count (lost: ℕ) (left: ℕ) : ℕ :=
  lost + left

theorem norma_initial_cards : initial_card_count 70 18 = 88 :=
  by
    -- skipping proof
    sorry

end NUMINAMATH_GPT_norma_initial_cards_l148_14848


namespace NUMINAMATH_GPT_coloring_impossible_l148_14820

-- Define vertices for the outer pentagon and inner star
inductive Vertex
| A | B | C | D | E | A' | B' | C' | D' | E'

open Vertex

-- Define segments in the figure
def Segments : List (Vertex × Vertex) :=
  [(A, B), (B, C), (C, D), (D, E), (E, A),
   (A, A'), (B, B'), (C, C'), (D, D'), (E, E'),
   (A', C), (C, E'), (E, B'), (B, D'), (D, A')]

-- Color type
inductive Color
| Red | Green | Blue

open Color

-- Condition for coloring: no two segments of the same color share a common endpoint
def distinct_color (c : Vertex → Color) : Prop :=
  ∀ (v1 v2 v3 : Vertex) (h1 : (v1, v2) ∈ Segments) (h2 : (v2, v3) ∈ Segments),
  c v1 ≠ c v2 ∧ c v2 ≠ c v3 ∧ c v1 ≠ c v3

-- Statement of the proof problem
theorem coloring_impossible : ¬ ∃ (c : Vertex → Color), distinct_color c := 
by 
  sorry

end NUMINAMATH_GPT_coloring_impossible_l148_14820


namespace NUMINAMATH_GPT_find_n_l148_14813

theorem find_n (n : ℤ) (h : n + (n + 1) + (n + 2) = 9) : n = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l148_14813


namespace NUMINAMATH_GPT_not_proportional_x2_y2_l148_14892

def directly_proportional (x y : ℝ) : Prop :=
∃ k : ℝ, x = k * y

def inversely_proportional (x y : ℝ) : Prop :=
∃ k : ℝ, x * y = k

theorem not_proportional_x2_y2 (x y : ℝ) :
  x^2 + y^2 = 16 → ¬directly_proportional x y ∧ ¬inversely_proportional x y :=
by
  sorry

end NUMINAMATH_GPT_not_proportional_x2_y2_l148_14892


namespace NUMINAMATH_GPT_roses_in_vase_l148_14814

theorem roses_in_vase (initial_roses added_roses : ℕ) (h₀ : initial_roses = 10) (h₁ : added_roses = 8) : initial_roses + added_roses = 18 :=
by
  sorry

end NUMINAMATH_GPT_roses_in_vase_l148_14814


namespace NUMINAMATH_GPT_roots_quartic_sum_l148_14894

theorem roots_quartic_sum (c d : ℝ) (h1 : c + d = 3) (h2 : c * d = 1) (hc : Polynomial.eval c (Polynomial.C (-1) + Polynomial.X ^ 4 - 6 * Polynomial.X ^ 3 - 4 * Polynomial.X) = 0) (hd : Polynomial.eval d (Polynomial.C (-1) + Polynomial.X ^ 4 - 6 * Polynomial.X ^ 3 - 4 * Polynomial.X) = 0) :
  c * d + c + d = 4 :=
by
  sorry

end NUMINAMATH_GPT_roots_quartic_sum_l148_14894


namespace NUMINAMATH_GPT_proof_w3_u2_y2_l148_14880

variable (x y z w u d : ℤ)

def arithmetic_sequence := x = 1370 ∧ z = 1070 ∧ w = -180 ∧ u = -6430 ∧ (z = x + 2 * d) ∧ (y = x + d)

theorem proof_w3_u2_y2 (h : arithmetic_sequence x y z w u d) : w^3 - u^2 + y^2 = -44200100 :=
  by
    sorry

end NUMINAMATH_GPT_proof_w3_u2_y2_l148_14880


namespace NUMINAMATH_GPT_intersection_point_x_coordinate_l148_14832

noncomputable def hyperbola (x y b : ℝ) := x^2 - (y^2 / b^2) = 1

noncomputable def c := 1 + Real.sqrt 3

noncomputable def distance (p1 p2 : ℝ × ℝ) := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_point_x_coordinate
  (x y b : ℝ)
  (h_hyperbola : hyperbola x y b)
  (h_distance_foci : distance (2 * c, 0) (0, 0) = 2 * c)
  (h_circle_center : distance (x, y) (0, 0) = c)
  (h_p_distance : distance (x, y) (2 * c, 0) = c + 2) :
  x = (Real.sqrt 3 + 1) / 2 :=
sorry

end NUMINAMATH_GPT_intersection_point_x_coordinate_l148_14832


namespace NUMINAMATH_GPT_platform_length_l148_14885

theorem platform_length (train_length : ℝ) (speed_kmph : ℝ) (time_sec : ℝ) (platform_length : ℝ) :
  train_length = 150 ∧ speed_kmph = 75 ∧ time_sec = 20 →
  platform_length = 1350 :=
by
  sorry

end NUMINAMATH_GPT_platform_length_l148_14885


namespace NUMINAMATH_GPT_roots_greater_than_two_range_l148_14810

theorem roots_greater_than_two_range (m : ℝ) :
  ∀ x1 x2 : ℝ, (x1^2 + (m - 4) * x1 + 6 - m = 0) ∧ (x2^2 + (m - 4) * x2 + 6 - m = 0) ∧ (x1 > 2) ∧ (x2 > 2) →
  -2 < m ∧ m ≤ 2 - 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_roots_greater_than_two_range_l148_14810


namespace NUMINAMATH_GPT_equivalent_operation_l148_14831

theorem equivalent_operation (x : ℚ) :
  (x * (2/3)) / (5/6) = x * (4/5) :=
by
  -- Normal proof steps might follow here
  sorry

end NUMINAMATH_GPT_equivalent_operation_l148_14831


namespace NUMINAMATH_GPT_least_possible_product_of_distinct_primes_gt_50_l148_14866

-- Define a predicate to check if a number is a prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Define the conditions: two distinct primes greater than 50
def distinct_primes_gt_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p > 50 ∧ q > 50 ∧ p ≠ q

-- The least possible product of two distinct primes each greater than 50
theorem least_possible_product_of_distinct_primes_gt_50 :
  ∃ p q : ℕ, distinct_primes_gt_50 p q ∧ p * q = 3127 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_product_of_distinct_primes_gt_50_l148_14866


namespace NUMINAMATH_GPT_range_of_m_l148_14803

open Set

theorem range_of_m (m : ℝ) :
  (∃ f : ℤ → Prop, (∀ x, f x ↔ x + 5 > 0 ∧ x - m ≤ 1) ∧ (∃ a b c : ℤ, f a ∧ f b ∧ f c))
  → (-3 ≤ m ∧ m < -2) := 
sorry

end NUMINAMATH_GPT_range_of_m_l148_14803


namespace NUMINAMATH_GPT_maximize_revenue_l148_14874

-- Define the revenue function
def revenue (p : ℝ) : ℝ :=
  p * (150 - 4 * p)

-- Define the price constraints
def price_constraint (p : ℝ) : Prop :=
  0 ≤ p ∧ p ≤ 30

-- The theorem statement to prove that p = 19 maximizes the revenue
theorem maximize_revenue : ∀ p: ℕ, price_constraint p → revenue p ≤ revenue 19 :=
by
  sorry

end NUMINAMATH_GPT_maximize_revenue_l148_14874


namespace NUMINAMATH_GPT_smallest_whole_number_larger_than_sum_l148_14856

noncomputable def mixed_number1 : ℚ := 3 + 2/3
noncomputable def mixed_number2 : ℚ := 4 + 1/4
noncomputable def mixed_number3 : ℚ := 5 + 1/5
noncomputable def mixed_number4 : ℚ := 6 + 1/6
noncomputable def mixed_number5 : ℚ := 7 + 1/7

noncomputable def sum_of_mixed_numbers : ℚ :=
  mixed_number1 + mixed_number2 + mixed_number3 + mixed_number4 + mixed_number5

theorem smallest_whole_number_larger_than_sum : 
  ∃ n : ℤ, (n : ℚ) > sum_of_mixed_numbers ∧ n = 27 :=
by
  sorry

end NUMINAMATH_GPT_smallest_whole_number_larger_than_sum_l148_14856


namespace NUMINAMATH_GPT_Marty_combinations_l148_14860

def unique_combinations (colors techniques : ℕ) : ℕ :=
  colors * techniques

theorem Marty_combinations :
  unique_combinations 6 5 = 30 := by
  sorry

end NUMINAMATH_GPT_Marty_combinations_l148_14860


namespace NUMINAMATH_GPT_largest_integer_square_two_digits_l148_14808

theorem largest_integer_square_two_digits : 
  ∃ M : ℤ, (M * M ≥ 10 ∧ M * M < 100) ∧ (∀ x : ℤ, (x * x ≥ 10 ∧ x * x < 100) → x ≤ M) ∧ M = 9 := 
by
  sorry

end NUMINAMATH_GPT_largest_integer_square_two_digits_l148_14808


namespace NUMINAMATH_GPT_lesser_fraction_solution_l148_14895

noncomputable def lesser_fraction (x y : ℚ) (h₁ : x + y = 7/8) (h₂ : x * y = 1/12) : ℚ :=
  if x ≤ y then x else y

theorem lesser_fraction_solution (x y : ℚ) (h₁ : x + y = 7/8) (h₂ : x * y = 1/12) :
  lesser_fraction x y h₁ h₂ = (7 - Real.sqrt 17) / 16 := by
  sorry

end NUMINAMATH_GPT_lesser_fraction_solution_l148_14895


namespace NUMINAMATH_GPT_right_triangle_inequality_l148_14801

variable (a b c : ℝ)

theorem right_triangle_inequality
  (h1 : b < a) -- shorter leg is less than longer leg
  (h2 : c = Real.sqrt (a^2 + b^2)) -- hypotenuse from Pythagorean theorem
  : a + b / 2 > c ∧ c > (8 / 9) * (a + b / 2) := 
sorry

end NUMINAMATH_GPT_right_triangle_inequality_l148_14801


namespace NUMINAMATH_GPT_ranking_Fiona_Giselle_Ella_l148_14830

-- Definitions of scores 
variable (score : String → ℕ)

-- Conditions based on the problem statement
def ella_not_highest : Prop := ¬ (score "Ella" = max (score "Ella") (max (score "Fiona") (score "Giselle")))
def giselle_not_lowest : Prop := ¬ (score "Giselle" = min (score "Ella") (score "Giselle"))

-- The goal is to rank the scores from highest to lowest
def score_ranking : Prop := (score "Fiona" > score "Giselle") ∧ (score "Giselle" > score "Ella")

theorem ranking_Fiona_Giselle_Ella :
  ella_not_highest score →
  giselle_not_lowest score →
  score_ranking score :=
by
  sorry

end NUMINAMATH_GPT_ranking_Fiona_Giselle_Ella_l148_14830


namespace NUMINAMATH_GPT_marie_lost_erasers_l148_14855

def initialErasers : ℕ := 95
def finalErasers : ℕ := 53

theorem marie_lost_erasers : initialErasers - finalErasers = 42 := by
  sorry

end NUMINAMATH_GPT_marie_lost_erasers_l148_14855


namespace NUMINAMATH_GPT_num_rectangular_tables_l148_14829

theorem num_rectangular_tables (R : ℕ) 
  (rectangular_tables_seat : R * 10 = 70) :
  R = 7 := by
  sorry

end NUMINAMATH_GPT_num_rectangular_tables_l148_14829


namespace NUMINAMATH_GPT_value_of_b_l148_14859

theorem value_of_b (x b : ℝ) (h₁ : x = 0.3) 
  (h₂ : (10 * x + b) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / 3) : 
  b = 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_b_l148_14859


namespace NUMINAMATH_GPT_min_value_of_f_l148_14800

noncomputable def f (x : ℝ) : ℝ := 7 * x^2 - 28 * x + 1425

theorem min_value_of_f : ∃ (x : ℝ), f x = 1397 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_f_l148_14800


namespace NUMINAMATH_GPT_valve_difference_l148_14865

theorem valve_difference (time_both : ℕ) (time_first : ℕ) (pool_capacity : ℕ) (V1 V2 diff : ℕ) :
  time_both = 48 → 
  time_first = 120 → 
  pool_capacity = 12000 → 
  V1 = pool_capacity / time_first → 
  V1 + V2 = pool_capacity / time_both → 
  diff = V2 - V1 → 
  diff = 50 :=
by sorry

end NUMINAMATH_GPT_valve_difference_l148_14865


namespace NUMINAMATH_GPT_find_y_minus_x_l148_14845

theorem find_y_minus_x (x y : ℝ) (h1 : x + y = 8) (h2 : y - 3 * x = 7) : y - x = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_find_y_minus_x_l148_14845


namespace NUMINAMATH_GPT_games_new_friends_l148_14828

-- Definitions based on the conditions
def total_games_all_friends : ℕ := 141
def games_old_friends : ℕ := 53

-- Statement of the problem
theorem games_new_friends {games_new_friends : ℕ} :
  games_new_friends = total_games_all_friends - games_old_friends :=
sorry

end NUMINAMATH_GPT_games_new_friends_l148_14828


namespace NUMINAMATH_GPT_mahesh_worked_days_l148_14869

-- Definitions
def mahesh_work_days := 45
def rajesh_work_days := 30
def total_work_days := 54

-- Theorem statement
theorem mahesh_worked_days (maheshrate : ℕ := mahesh_work_days) (rajeshrate : ℕ := rajesh_work_days) (totaldays : ℕ := total_work_days) :
  ∃ x : ℕ, x = totaldays - rajesh_work_days := by
  apply Exists.intro (54 - 30)
  simp
  sorry

end NUMINAMATH_GPT_mahesh_worked_days_l148_14869


namespace NUMINAMATH_GPT_largest_expr_l148_14817

noncomputable def A : ℝ := 2 * 1005 ^ 1006
noncomputable def B : ℝ := 1005 ^ 1006
noncomputable def C : ℝ := 1004 * 1005 ^ 1005
noncomputable def D : ℝ := 2 * 1005 ^ 1005
noncomputable def E : ℝ := 1005 ^ 1005
noncomputable def F : ℝ := 1005 ^ 1004

theorem largest_expr : A - B > B - C ∧ A - B > C - D ∧ A - B > D - E ∧ A - B > E - F :=
by
  sorry

end NUMINAMATH_GPT_largest_expr_l148_14817


namespace NUMINAMATH_GPT_bug_traverses_36_tiles_l148_14846

-- Define the dimensions of the rectangle and the bug's problem setup
def width : ℕ := 12
def length : ℕ := 25

-- Define the function to calculate the number of tiles traversed by the bug
def tiles_traversed (w l : ℕ) : ℕ :=
  w + l - Nat.gcd w l

-- Prove the number of tiles traversed by the bug is 36
theorem bug_traverses_36_tiles : tiles_traversed width length = 36 :=
by
  -- This part will be proven; currently, we add sorry
  sorry

end NUMINAMATH_GPT_bug_traverses_36_tiles_l148_14846


namespace NUMINAMATH_GPT_find_sum_of_variables_l148_14861

theorem find_sum_of_variables (x y : ℚ) (h1 : 5 * x - 3 * y = 17) (h2 : 3 * x + 5 * y = 1) : x + y = 21 / 17 := 
  sorry

end NUMINAMATH_GPT_find_sum_of_variables_l148_14861


namespace NUMINAMATH_GPT_graph_movement_l148_14867

noncomputable def f (x : ℝ) : ℝ := -2 * (x - 1) ^ 2 + 3

noncomputable def g (x : ℝ) : ℝ := -2 * x ^ 2

theorem graph_movement :
  ∀ (x y : ℝ),
  y = f x →
  g x = y → 
  (∃ Δx Δy, Δx = -1 ∧ Δy = -3 ∧ g (x + Δx) = y + Δy) :=
by
  sorry

end NUMINAMATH_GPT_graph_movement_l148_14867


namespace NUMINAMATH_GPT_max_tries_needed_to_open_lock_l148_14849

-- Definitions and conditions
def num_buttons : ℕ := 9
def sequence_length : ℕ := 4
def opposite_trigrams : ℕ := 2  -- assumption based on the problem's example
def total_combinations : ℕ := 3024

theorem max_tries_needed_to_open_lock :
  (total_combinations - (8 * 1 * 7 * 6 + 8 * 6 * 1 * 6 + 8 * 6 * 4 * 1)) = 2208 :=
by
  sorry

end NUMINAMATH_GPT_max_tries_needed_to_open_lock_l148_14849
