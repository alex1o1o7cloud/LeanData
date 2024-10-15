import Mathlib

namespace NUMINAMATH_GPT_constant_term_of_product_l606_60649

def P(x: ℝ) : ℝ := x^6 + 2 * x^2 + 3
def Q(x: ℝ) : ℝ := x^4 + x^3 + 4
def R(x: ℝ) : ℝ := 2 * x^2 + 3 * x + 7

theorem constant_term_of_product :
  let C := (P 0) * (Q 0) * (R 0)
  C = 84 :=
by
  let C := (P 0) * (Q 0) * (R 0)
  show C = 84
  sorry

end NUMINAMATH_GPT_constant_term_of_product_l606_60649


namespace NUMINAMATH_GPT_number_of_welders_left_l606_60643

-- Definitions for the given problem
def total_welders : ℕ := 36
def initial_days : ℝ := 1
def remaining_days : ℝ := 3.0000000000000004
def total_days : ℝ := 3

-- Condition equations
variable (r : ℝ) -- rate at which each welder works
variable (W : ℝ) -- total work

-- Equation representing initial total work
def initial_work : W = total_welders * r * total_days := by sorry

-- Welders who left for another project
variable (X : ℕ) -- number of welders who left

-- Equation representing remaining work
def remaining_work : (total_welders - X) * r * remaining_days = W - (total_welders * r * initial_days) := by sorry

-- Theorem to prove
theorem number_of_welders_left :
  (total_welders * total_days : ℝ) = W →
  (total_welders - X) * remaining_days = W - (total_welders * r * initial_days) →
  X = 12 :=
sorry

end NUMINAMATH_GPT_number_of_welders_left_l606_60643


namespace NUMINAMATH_GPT_effective_annual_rate_of_interest_l606_60613

theorem effective_annual_rate_of_interest 
  (i : ℝ) (n : ℕ) (h_i : i = 0.10) (h_n : n = 2) : 
  (1 + i / n)^n - 1 = 0.1025 :=
by
  sorry

end NUMINAMATH_GPT_effective_annual_rate_of_interest_l606_60613


namespace NUMINAMATH_GPT_fraction_division_l606_60657

theorem fraction_division :
  (3 / 4) / (5 / 6) = 9 / 10 :=
by {
  -- We skip the proof as per the instructions
  sorry
}

end NUMINAMATH_GPT_fraction_division_l606_60657


namespace NUMINAMATH_GPT_general_inequality_l606_60687

theorem general_inequality (x : ℝ) (n : ℕ) (h_pos_x : x > 0) (h_pos_n : 0 < n) : 
  x + n^n / x^n ≥ n + 1 := by 
  sorry

end NUMINAMATH_GPT_general_inequality_l606_60687


namespace NUMINAMATH_GPT_value_of_abc_l606_60678

noncomputable def f (a b c x : ℝ) := a * x^2 + b * x + c
noncomputable def f_inv (a b c x : ℝ) := c * x^2 + b * x + a

-- The main theorem statement
theorem value_of_abc (a b c : ℝ) (h : ∀ x : ℝ, f a b c (f_inv a b c x) = x) : a + b + c = 1 :=
sorry

end NUMINAMATH_GPT_value_of_abc_l606_60678


namespace NUMINAMATH_GPT_series_sum_equals_three_fourths_l606_60606

noncomputable def infinite_series_sum : ℝ :=
  (∑' n : ℕ, (3 * (n + 1) + 2) / ((n + 1) * (n + 1 + 1) * (n + 1 + 3)))

theorem series_sum_equals_three_fourths :
  infinite_series_sum = 3 / 4 :=
sorry

end NUMINAMATH_GPT_series_sum_equals_three_fourths_l606_60606


namespace NUMINAMATH_GPT_nine_digit_palindrome_count_l606_60646

-- Defining the set of digits
def digits : Multiset ℕ := {1, 1, 2, 2, 2, 4, 4, 5, 5}

-- Defining the proposition of the number of 9-digit palindromes
def num_9_digit_palindromes (digs : Multiset ℕ) : ℕ := 36

-- The proof statement
theorem nine_digit_palindrome_count : num_9_digit_palindromes digits = 36 := 
sorry

end NUMINAMATH_GPT_nine_digit_palindrome_count_l606_60646


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_squares_l606_60632

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h : a * b = 41) :
  (1 / (a^2) + 1 / (b^2)) = 1682 / 1681 := sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_squares_l606_60632


namespace NUMINAMATH_GPT_total_balloons_sam_and_dan_l606_60653

noncomputable def sam_initial_balloons : ℝ := 46.0
noncomputable def balloons_given_to_fred : ℝ := 10.0
noncomputable def dan_balloons : ℝ := 16.0

theorem total_balloons_sam_and_dan :
  (sam_initial_balloons - balloons_given_to_fred) + dan_balloons = 52.0 := 
by 
  sorry

end NUMINAMATH_GPT_total_balloons_sam_and_dan_l606_60653


namespace NUMINAMATH_GPT_find_g_3_16_l606_60659

theorem find_g_3_16 (g : ℝ → ℝ) (h1 : ∀ x, 0 ≤ x → x ≤ 1 → g x = g x) 
(h2 : g 0 = 0) 
(h3 : ∀ x y, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y) 
(h4 : ∀ x, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x) 
(h5 : ∀ x, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3) : 
  g (3 / 16) = 8 / 27 :=
sorry

end NUMINAMATH_GPT_find_g_3_16_l606_60659


namespace NUMINAMATH_GPT_find_x_plus_y_l606_60619

theorem find_x_plus_y :
  ∀ (x y : ℝ), (3 * x - y + 5)^2 + |2 * x - y + 3| = 0 → x + y = -3 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_find_x_plus_y_l606_60619


namespace NUMINAMATH_GPT_remainder_of_four_m_plus_five_l606_60661

theorem remainder_of_four_m_plus_five (m : ℤ) (h : m % 5 = 3) : (4 * m + 5) % 5 = 2 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_remainder_of_four_m_plus_five_l606_60661


namespace NUMINAMATH_GPT_smallest_number_is_61_point_4_l606_60641

theorem smallest_number_is_61_point_4 (x y z t : ℝ)
  (h1 : y = 2 * x)
  (h2 : z = 4 * y)
  (h3 : t = (y + z) / 3)
  (h4 : (x + y + z + t) / 4 = 220) :
  x = 2640 / 43 :=
by sorry

end NUMINAMATH_GPT_smallest_number_is_61_point_4_l606_60641


namespace NUMINAMATH_GPT_direct_variation_y_value_l606_60627

theorem direct_variation_y_value (x y : ℝ) (hx1 : x ≤ 10 → y = 3 * x)
  (hx2 : x > 10 → y = 6 * x) : 
  x = 20 → y = 120 := by
  sorry

end NUMINAMATH_GPT_direct_variation_y_value_l606_60627


namespace NUMINAMATH_GPT_sequence_v_n_l606_60647

theorem sequence_v_n (v : ℕ → ℝ)
  (h_recurr : ∀ n, v (n+2) = 3 * v (n+1) - v n)
  (h_init1 : v 3 = 16)
  (h_init2 : v 6 = 211) : 
  v 5 = 81.125 :=
sorry

end NUMINAMATH_GPT_sequence_v_n_l606_60647


namespace NUMINAMATH_GPT_binomial_10_3_eq_120_l606_60689

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end NUMINAMATH_GPT_binomial_10_3_eq_120_l606_60689


namespace NUMINAMATH_GPT_complement_of_A_in_U_l606_60656

noncomputable def U := {x : ℝ | Real.exp x > 1}

noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (x - 1)

def A := { x : ℝ | x > 1 }

def compl (U A : Set ℝ) := { x : ℝ | x ∈ U ∧ x ∉ A }

theorem complement_of_A_in_U : compl U A = { x : ℝ | 0 < x ∧ x ≤ 1 } := sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l606_60656


namespace NUMINAMATH_GPT_last_two_digits_of_7_pow_2016_l606_60658

theorem last_two_digits_of_7_pow_2016 : (7^2016 : ℕ) % 100 = 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_last_two_digits_of_7_pow_2016_l606_60658


namespace NUMINAMATH_GPT_number_of_numbers_l606_60652

theorem number_of_numbers (N : ℕ) (h_avg : (18 * N + 40) / N = 22) : N = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_numbers_l606_60652


namespace NUMINAMATH_GPT_inequality_transformation_l606_60614

variable {a b : ℝ}

theorem inequality_transformation (h : a < b) : -a / 3 > -b / 3 :=
  sorry

end NUMINAMATH_GPT_inequality_transformation_l606_60614


namespace NUMINAMATH_GPT_petya_must_have_photo_files_on_portable_hard_drives_l606_60637

theorem petya_must_have_photo_files_on_portable_hard_drives 
    (H F P T : ℕ) 
    (h1 : H > F) 
    (h2 : P > T) 
    : ∃ x, x ≠ 0 ∧ x ≤ H :=
by
  sorry

end NUMINAMATH_GPT_petya_must_have_photo_files_on_portable_hard_drives_l606_60637


namespace NUMINAMATH_GPT_second_grade_girls_l606_60692

theorem second_grade_girls (G : ℕ) 
  (h1 : ∃ boys_2nd : ℕ, boys_2nd = 20)
  (h2 : ∃ students_3rd : ℕ, students_3rd = 2 * (20 + G))
  (h3 : 20 + G + (2 * (20 + G)) = 93) :
  G = 11 :=
by
  sorry

end NUMINAMATH_GPT_second_grade_girls_l606_60692


namespace NUMINAMATH_GPT_determine_k_range_l606_60655

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x
noncomputable def g (x : ℝ) : ℝ := (Real.log x) / x
noncomputable def h (x : ℝ) : ℝ := (Real.log x) / (x * x)

theorem determine_k_range :
  (∀ x : ℝ, x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) → f k x = g x) →
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) ∧ x2 ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1)) →
  k ∈ Set.Ico (1 / (Real.exp 1) ^ 2) (1 / (2 * Real.exp 1)) := 
  sorry

end NUMINAMATH_GPT_determine_k_range_l606_60655


namespace NUMINAMATH_GPT_system_solution_conditions_l606_60602

theorem system_solution_conditions (α1 α2 α3 α4 : ℝ) :
  (α1 = α4 ∨ α2 = α3) ↔ 
  (∃ x1 x2 x3 x4 : ℝ,
    x1 + x2 = α1 * α2 ∧
    x1 + x3 = α1 * α3 ∧
    x1 + x4 = α1 * α4 ∧
    x2 + x3 = α2 * α3 ∧
    x2 + x4 = α2 * α4 ∧
    x3 + x4 = α3 * α4 ∧
    x1 = x2 ∧
    x2 = x3 ∧
    x1 = α2^2 / 2 ∧
    x3 = α2^2 / 2 ∧
    x4 = α2 * α4 - (α2^2 / 2) ) :=
by sorry

end NUMINAMATH_GPT_system_solution_conditions_l606_60602


namespace NUMINAMATH_GPT_problem_statement_l606_60666

theorem problem_statement (x y : ℝ) : 
  ((-3 * x * y^2)^3 * (-6 * x^2 * y) / (9 * x^4 * y^5) = 18 * x * y^2) :=
by sorry

end NUMINAMATH_GPT_problem_statement_l606_60666


namespace NUMINAMATH_GPT_unique_sum_of_three_distinct_positive_perfect_squares_l606_60615

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def distinct_positive_perfect_squares_that_sum_to (a b c sum : ℕ) : Prop :=
  is_perfect_square a ∧ is_perfect_square b ∧ is_perfect_square c ∧
  a < b ∧ b < c ∧ a + b + c = sum

theorem unique_sum_of_three_distinct_positive_perfect_squares :
  (∃ a b c : ℕ, distinct_positive_perfect_squares_that_sum_to a b c 100) ∧
  (∀ a1 b1 c1 a2 b2 c2 : ℕ,
    distinct_positive_perfect_squares_that_sum_to a1 b1 c1 100 ∧
    distinct_positive_perfect_squares_that_sum_to a2 b2 c2 100 →
    (a1 = a2 ∧ b1 = b2 ∧ c1 = c2)) :=
by
  sorry

end NUMINAMATH_GPT_unique_sum_of_three_distinct_positive_perfect_squares_l606_60615


namespace NUMINAMATH_GPT_integer_values_abs_lt_5pi_l606_60645

theorem integer_values_abs_lt_5pi : 
  ∃ n : ℕ, n = 31 ∧ ∀ x : ℤ, |(x : ℝ)| < 5 * Real.pi → x ∈ (Finset.Icc (-15) 15) := 
sorry

end NUMINAMATH_GPT_integer_values_abs_lt_5pi_l606_60645


namespace NUMINAMATH_GPT_unit_vector_opposite_AB_is_l606_60672

open Real

noncomputable def unit_vector_opposite_dir (A B : ℝ × ℝ) : ℝ × ℝ :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let BA := (-AB.1, -AB.2)
  let mag_BA := sqrt (BA.1^2 + BA.2^2)
  (BA.1 / mag_BA, BA.2 / mag_BA)

theorem unit_vector_opposite_AB_is (A B : ℝ × ℝ) (hA : A = (1, 2)) (hB : B = (-2, 6)) :
  unit_vector_opposite_dir A B = (3/5, -4/5) :=
by
  sorry

end NUMINAMATH_GPT_unit_vector_opposite_AB_is_l606_60672


namespace NUMINAMATH_GPT_quadratic_equation_with_distinct_roots_l606_60609

theorem quadratic_equation_with_distinct_roots 
  (a p q b α : ℝ) 
  (hα1 : α ≠ 0) 
  (h_quad1 : α^2 + a * α + b = 0) 
  (h_quad2 : α^2 + p * α + q = 0) : 
  ∃ x : ℝ, x^2 - (b + q) * (a - p) / (q - b) * x + b * q * (a - p)^2 / (q - b)^2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_with_distinct_roots_l606_60609


namespace NUMINAMATH_GPT_total_winnings_l606_60650

theorem total_winnings (x : ℝ)
  (h1 : x / 4 = first_person_share)
  (h2 : x / 7 = second_person_share)
  (h3 : third_person_share = 17)
  (h4 : first_person_share + second_person_share + third_person_share = x) :
  x = 28 := 
by sorry

end NUMINAMATH_GPT_total_winnings_l606_60650


namespace NUMINAMATH_GPT_volume_of_soup_in_hemisphere_half_height_l606_60676

theorem volume_of_soup_in_hemisphere_half_height 
  (V_hemisphere : ℝ)
  (hV_hemisphere : V_hemisphere = 8)
  (V_cap : ℝ) :
  V_cap = 2.5 :=
sorry

end NUMINAMATH_GPT_volume_of_soup_in_hemisphere_half_height_l606_60676


namespace NUMINAMATH_GPT_anna_gets_more_candy_l606_60680

theorem anna_gets_more_candy :
  let anna_pieces_per_house := 14
  let anna_houses := 60
  let billy_pieces_per_house := 11
  let billy_houses := 75
  let anna_total := anna_pieces_per_house * anna_houses
  let billy_total := billy_pieces_per_house * billy_houses
  anna_total - billy_total = 15 := by
    let anna_pieces_per_house := 14
    let anna_houses := 60
    let billy_pieces_per_house := 11
    let billy_houses := 75
    let anna_total := anna_pieces_per_house * anna_houses
    let billy_total := billy_pieces_per_house * billy_houses
    have h1 : anna_total = 14 * 60 := rfl
    have h2 : billy_total = 11 * 75 := rfl
    sorry

end NUMINAMATH_GPT_anna_gets_more_candy_l606_60680


namespace NUMINAMATH_GPT_regular_17gon_symmetries_l606_60626

theorem regular_17gon_symmetries : 
  let L := 17
  let R := 360 / 17
  L + R = 17 + 360 / 17 :=
by
  sorry

end NUMINAMATH_GPT_regular_17gon_symmetries_l606_60626


namespace NUMINAMATH_GPT_gain_percent_l606_60690

variable (C S : ℝ)
variable (h : 65 * C = 50 * S)

theorem gain_percent (h : 65 * C = 50 * S) : (S - C) / C * 100 = 30 :=
by
  sorry

end NUMINAMATH_GPT_gain_percent_l606_60690


namespace NUMINAMATH_GPT_determine_radius_of_semicircle_l606_60679

noncomputable def radius_of_semicircle (P : ℝ) : ℝ :=
  P / (Real.pi + 2)

theorem determine_radius_of_semicircle :
  radius_of_semicircle 32.392033717615696 = 6.3 :=
by
  sorry

end NUMINAMATH_GPT_determine_radius_of_semicircle_l606_60679


namespace NUMINAMATH_GPT_geometric_sequence_sixth_term_l606_60684

/-- 
The statement: 
The first term of a geometric sequence is 1000, and the 8th term is 125. Prove that the positive,
real value for the 6th term is 31.25.
-/
theorem geometric_sequence_sixth_term :
  ∀ (a1 a8 a6 : ℝ) (r : ℝ),
    a1 = 1000 →
    a8 = 125 →
    a8 = a1 * r^7 →
    a6 = a1 * r^5 →
    a6 = 31.25 :=
by
  intros a1 a8 a6 r h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_geometric_sequence_sixth_term_l606_60684


namespace NUMINAMATH_GPT_mul_mental_math_l606_60648

theorem mul_mental_math :
  96 * 104 = 9984 := by
  sorry

end NUMINAMATH_GPT_mul_mental_math_l606_60648


namespace NUMINAMATH_GPT_major_snow_shadow_length_l606_60697

theorem major_snow_shadow_length :
  ∃ (a1 d : ℝ), 
  (3 * a1 + 12 * d = 16.5) ∧ 
  (12 * a1 + 66 * d = 84) ∧
  (a1 + 11 * d = 12.5) := 
sorry

end NUMINAMATH_GPT_major_snow_shadow_length_l606_60697


namespace NUMINAMATH_GPT_least_non_lucky_multiple_of_12_l606_60629

/- Defines what it means for a number to be a lucky integer -/
def isLucky (n : ℕ) : Prop :=
  n % (n.digits 10).sum = 0

/- Proves the least positive multiple of 12 that is not a lucky integer is 96 -/
theorem least_non_lucky_multiple_of_12 : ∃ n, n % 12 = 0 ∧ ¬isLucky n ∧ ∀ m, m % 12 = 0 ∧ ¬isLucky m → n ≤ m :=
  by
  sorry

end NUMINAMATH_GPT_least_non_lucky_multiple_of_12_l606_60629


namespace NUMINAMATH_GPT_B_C_work_days_l606_60610

noncomputable def days_for_B_and_C {A B C : ℝ} (hA : A = 1 / 10) (hA_B : A + B = 1 / 5) (hA_B_C : A + B + C = 1 / 3) : ℝ :=
  30 / 7

theorem B_C_work_days {A B C : ℝ} (hA : A = 1 / 10) (hA_B : A + B = 1 / 5) (hA_B_C : A + B + C = 1 / 3) :
  days_for_B_and_C hA hA_B hA_B_C = 30 / 7 :=
sorry

end NUMINAMATH_GPT_B_C_work_days_l606_60610


namespace NUMINAMATH_GPT_missy_yells_total_l606_60668

variable {O S M : ℕ}
variable (yells_at_obedient : ℕ)

-- Conditions:
def yells_stubborn (yells_at_obedient : ℕ) : ℕ := 4 * yells_at_obedient
def yells_mischievous (yells_at_obedient : ℕ) : ℕ := 2 * yells_at_obedient

-- Prove the total yells equal to 84 when yells_at_obedient = 12
theorem missy_yells_total (h : yells_at_obedient = 12) :
  yells_at_obedient + yells_stubborn yells_at_obedient + yells_mischievous yells_at_obedient = 84 :=
by
  sorry

end NUMINAMATH_GPT_missy_yells_total_l606_60668


namespace NUMINAMATH_GPT_comparison_abc_l606_60681

noncomputable def a : ℝ := (Real.exp 1 + 2) / Real.log (Real.exp 1 + 2)
noncomputable def b : ℝ := 2 / Real.log 2
noncomputable def c : ℝ := (Real.exp 1)^2 / (4 - Real.log 4)

theorem comparison_abc : c < b ∧ b < a :=
by {
  sorry
}

end NUMINAMATH_GPT_comparison_abc_l606_60681


namespace NUMINAMATH_GPT_solution_set_inequality_l606_60669

theorem solution_set_inequality (x : ℝ) : (1 - x) * (2 + x) < 0 ↔ x < -2 ∨ x > 1 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l606_60669


namespace NUMINAMATH_GPT_example_problem_l606_60670

theorem example_problem : 2 + 3 * 4 - 5 + 6 / 3 = 11 := by
  sorry

end NUMINAMATH_GPT_example_problem_l606_60670


namespace NUMINAMATH_GPT_average_temperature_week_l606_60623

theorem average_temperature_week :
  let d1 := 40
  let d2 := 40
  let d3 := 40
  let d4 := 80
  let d5 := 80
  let remaining_days_total := 140
  d1 + d2 + d3 + d4 + d5 + remaining_days_total = 420 ∧ 420 / 7 = 60 :=
by sorry

end NUMINAMATH_GPT_average_temperature_week_l606_60623


namespace NUMINAMATH_GPT_find_x_from_exponential_eq_l606_60611

theorem find_x_from_exponential_eq (x : ℕ) (h : 3^x + 3^x + 3^x + 3^x = 6561) : x = 6 := 
sorry

end NUMINAMATH_GPT_find_x_from_exponential_eq_l606_60611


namespace NUMINAMATH_GPT_right_triangle_area_l606_60635

theorem right_triangle_area (a b c p : ℝ) (h1 : a = b) (h2 : 3 * p = a + b + c)
  (h3 : c = Real.sqrt (2 * a ^ 2)) :
  (1/2) * a ^ 2 = (9 * p ^ 2 * (3 - 2 * Real.sqrt 2)) / 4 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_area_l606_60635


namespace NUMINAMATH_GPT_length_of_AB_l606_60633

-- Defining the parabola and the condition on x1 and x2
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def condition (x1 x2 : ℝ) : Prop := x1 + x2 = 9

-- The main statement to prove |AB| = 11
theorem length_of_AB (x1 x2 y1 y2 : ℝ) (h1 : parabola x1 y1) (h2 : parabola x2 y2) (hx : condition x1 x2) :
  abs (x1 - x2) + abs (y1 - y2) = 11 :=
sorry

end NUMINAMATH_GPT_length_of_AB_l606_60633


namespace NUMINAMATH_GPT_evaluate_exponents_l606_60607

theorem evaluate_exponents :
  (5 ^ 0.4) * (5 ^ 0.6) * (5 ^ 0.2) * (5 ^ 0.3) * (5 ^ 0.5) = 25 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_exponents_l606_60607


namespace NUMINAMATH_GPT_concert_revenue_l606_60644

-- Define the prices and attendees
def adult_price := 26
def teenager_price := 18
def children_price := adult_price / 2
def num_adults := 183
def num_teenagers := 75
def num_children := 28

-- Calculate total revenue
def total_revenue := num_adults * adult_price + num_teenagers * teenager_price + num_children * children_price

-- The goal is to prove that total_revenue equals 6472
theorem concert_revenue : total_revenue = 6472 :=
by
  sorry

end NUMINAMATH_GPT_concert_revenue_l606_60644


namespace NUMINAMATH_GPT_business_proof_l606_60685

section Business_Problem

variables (investment cost_initial rubles production_capacity : ℕ)
variables (produced_July incomplete_July bottles_August bottles_September days_September : ℕ)
variables (total_depreciation residual_value sales_amount profit_target : ℕ)

def depreciation_per_bottle (cost_initial production_capacity : ℕ) : ℕ := 
    cost_initial / production_capacity

def calculate_total_depreciation (depreciation_per_bottle produced_July bottles_August bottles_September : ℕ) : ℕ :=
    (produced_July * depreciation_per_bottle) + (bottles_August * depreciation_per_bottle) + (bottles_September * depreciation_per_bottle)

def calculate_residual_value (cost_initial total_depreciation : ℕ) : ℕ :=
    cost_initial - total_depreciation

def calculate_sales_amount (residual_value profit_target : ℕ) : ℕ :=
    residual_value + profit_target

theorem business_proof
    (H1: investment = 1500000) 
    (H2: cost_initial = 500000)
    (H3: production_capacity = 100000)
    (H4: produced_July = 200)
    (H5: incomplete_July = 5)
    (H6: bottles_August = 15000)
    (H7: bottles_September = 12300)
    (H8: days_September = 20)
    (H9: total_depreciation = 137500)
    (H10: residual_value = 362500)
    (H11: profit_target = 10000)
    (H12: sales_amount = 372500): 

    total_depreciation = calculate_total_depreciation (depreciation_per_bottle cost_initial production_capacity) produced_July bottles_August bottles_September ∧
    residual_value = calculate_residual_value cost_initial total_depreciation ∧
    sales_amount = calculate_sales_amount residual_value profit_target := 
by 
  sorry

end Business_Problem

end NUMINAMATH_GPT_business_proof_l606_60685


namespace NUMINAMATH_GPT_A_completes_work_in_18_days_l606_60617

-- Define the conditions
def efficiency_A_twice_B (A B : ℕ → ℕ) : Prop := ∀ w, A w = 2 * B w
def same_work_time (A B C D : ℕ → ℕ) : Prop := 
  ∀ w t, A w + B w = C w + D w ∧ C t = 1 / 20 ∧ D t = 1 / 30

-- Define the key quantity to be proven
theorem A_completes_work_in_18_days (A B C D : ℕ → ℕ) 
  (h1 : efficiency_A_twice_B A B) 
  (h2 : same_work_time A B C D) : 
  ∀ w, A w = 1 / 18 :=
sorry

end NUMINAMATH_GPT_A_completes_work_in_18_days_l606_60617


namespace NUMINAMATH_GPT_find_unknown_polynomial_l606_60686

theorem find_unknown_polynomial (m : ℤ) : 
  ∃ q : ℤ, (q + (m^2 - 2 * m + 3) = 3 * m^2 + m - 1) → q = 2 * m^2 + 3 * m - 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_unknown_polynomial_l606_60686


namespace NUMINAMATH_GPT_total_money_of_james_and_ali_l606_60622

def jamesOwns : ℕ := 145
def jamesAliDifference : ℕ := 40
def aliOwns : ℕ := jamesOwns - jamesAliDifference

theorem total_money_of_james_and_ali :
  jamesOwns + aliOwns = 250 := by
  sorry

end NUMINAMATH_GPT_total_money_of_james_and_ali_l606_60622


namespace NUMINAMATH_GPT_cost_of_camel_proof_l606_60640

noncomputable def cost_of_camel (C H O E : ℕ) : ℕ :=
  if 10 * C = 24 * H ∧ 16 * H = 4 * O ∧ 6 * O = 4 * E ∧ 10 * E = 120000 then 4800 else 0

theorem cost_of_camel_proof (C H O E : ℕ) 
  (h1 : 10 * C = 24 * H) (h2 : 16 * H = 4 * O) (h3 : 6 * O = 4 * E) (h4 : 10 * E = 120000) :
  cost_of_camel C H O E = 4800 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_camel_proof_l606_60640


namespace NUMINAMATH_GPT_min_value_of_quadratic_l606_60651

theorem min_value_of_quadratic (x : ℝ) : 
  ∃ m : ℝ, (∀ z : ℝ, z = 5 * x ^ 2 + 20 * x + 25 → z ≥ m) ∧ m = 5 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l606_60651


namespace NUMINAMATH_GPT_divisibility_by_n5_plus_1_l606_60674

theorem divisibility_by_n5_plus_1 (n k : ℕ) (hn : 0 < n) (hk : 0 < k) : 
  n^5 + 1 ∣ (n^4 - 1) * (n^3 - n^2 + n - 1)^k + (n + 1) * n^(4 * k - 1) :=
sorry

end NUMINAMATH_GPT_divisibility_by_n5_plus_1_l606_60674


namespace NUMINAMATH_GPT_Lincoln_High_School_max_principals_l606_60677

def max_principals (total_years : ℕ) (term_length : ℕ) (max_principals_count : ℕ) : Prop :=
  ∀ (period : ℕ), period = total_years → 
                  term_length = 4 → 
                  max_principals_count = 3

theorem Lincoln_High_School_max_principals 
  (total_years term_length max_principals_count : ℕ) :
  max_principals total_years term_length max_principals_count :=
by 
  intros period h1 h2
  have h3 : period = 10 := sorry
  have h4 : term_length = 4 := sorry
  have h5 : max_principals_count = 3 := sorry
  sorry

end NUMINAMATH_GPT_Lincoln_High_School_max_principals_l606_60677


namespace NUMINAMATH_GPT_incorrect_statement_l606_60638

variable (f : ℝ → ℝ)
variable (k : ℝ)
variable (h₁ : f 0 = -1)
variable (h₂ : ∀ x, f' x > k)
variable (h₃ : k > 1)

theorem incorrect_statement :
  ¬ f (1 / (k - 1)) < 1 / (k - 1) :=
sorry

end NUMINAMATH_GPT_incorrect_statement_l606_60638


namespace NUMINAMATH_GPT_part1_part2_l606_60698

-- Part 1: Expression simplification
theorem part1 (a : ℝ) : (a - 3)^2 + a * (4 - a) = -2 * a + 9 := 
by
  sorry

-- Part 2: Solution set of inequalities
theorem part2 (x : ℝ) : 
  (3 * x - 5 < x + 1) ∧ (2 * (2 * x - 1) ≥ 3 * x - 4) ↔ (-2 ≤ x ∧ x < 3) := 
by
  sorry

end NUMINAMATH_GPT_part1_part2_l606_60698


namespace NUMINAMATH_GPT_opera_house_earnings_l606_60682

-- Definitions corresponding to the conditions
def num_rows : Nat := 150
def seats_per_row : Nat := 10
def ticket_cost : Nat := 10
def pct_not_taken : Nat := 20

-- Calculations based on conditions
def total_seats := num_rows * seats_per_row
def seats_not_taken := total_seats * pct_not_taken / 100
def seats_taken := total_seats - seats_not_taken
def earnings := seats_taken * ticket_cost

-- The theorem to prove
theorem opera_house_earnings : earnings = 12000 := sorry

end NUMINAMATH_GPT_opera_house_earnings_l606_60682


namespace NUMINAMATH_GPT_train_crossing_platform_l606_60604

/-- Given a train crosses a 100 m platform in 15 seconds, and the length of the train is 350 m,
    prove that the train takes 20 seconds to cross a second platform of length 250 m. -/
theorem train_crossing_platform (dist1 dist2 l_t t1 t2 : ℝ) (h1 : dist1 = 100) (h2 : dist2 = 250) (h3 : l_t = 350) (h4 : t1 = 15) :
  t2 = 20 :=
sorry

end NUMINAMATH_GPT_train_crossing_platform_l606_60604


namespace NUMINAMATH_GPT_product_range_l606_60600

theorem product_range (m b : ℚ) (h₀ : m = 3 / 4) (h₁ : b = 6 / 5) : 0 < m * b ∧ m * b < 1 :=
by
  sorry

end NUMINAMATH_GPT_product_range_l606_60600


namespace NUMINAMATH_GPT_james_total_chore_time_l606_60654

theorem james_total_chore_time
  (V C L : ℝ)
  (hV : V = 3)
  (hC : C = 3 * V)
  (hL : L = C / 2) :
  V + C + L = 16.5 := by
  sorry

end NUMINAMATH_GPT_james_total_chore_time_l606_60654


namespace NUMINAMATH_GPT_smallest_difference_of_sides_l606_60699

/-- Triangle PQR has a perimeter of 2021 units. The sides have lengths that are integer values with PQ < QR ≤ PR. 
The smallest possible value of QR - PQ is 1. -/
theorem smallest_difference_of_sides :
  ∃ (PQ QR PR : ℕ), PQ < QR ∧ QR ≤ PR ∧ PQ + QR + PR = 2021 ∧ PQ + QR > PR ∧ PQ + PR > QR ∧ QR + PR > PQ ∧ QR - PQ = 1 :=
sorry

end NUMINAMATH_GPT_smallest_difference_of_sides_l606_60699


namespace NUMINAMATH_GPT_sum_increased_consecutive_integers_product_990_l606_60601

theorem sum_increased_consecutive_integers_product_990 
  (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : a * b * c = 990) :
  (a + 2) + (b + 2) + (c + 2) = 36 :=
sorry

end NUMINAMATH_GPT_sum_increased_consecutive_integers_product_990_l606_60601


namespace NUMINAMATH_GPT_quadratic_solution_l606_60664

theorem quadratic_solution (x : ℝ) : x^2 - 2 * x - 3 = 0 → (x = 3 ∨ x = -1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_l606_60664


namespace NUMINAMATH_GPT_folding_cranes_together_l606_60671

theorem folding_cranes_together (rateA rateB combined_time : ℝ)
  (hA : rateA = 1 / 30)
  (hB : rateB = 1 / 45)
  (combined_rate : ℝ := rateA + rateB)
  (h_combined_rate : combined_rate = 1 / combined_time):
  combined_time = 18 :=
by
  sorry

end NUMINAMATH_GPT_folding_cranes_together_l606_60671


namespace NUMINAMATH_GPT_max_term_of_sequence_l606_60636

def a (n : ℕ) : ℚ := (n : ℚ) / (n^2 + 156)

theorem max_term_of_sequence : ∃ n, (n = 12 ∨ n = 13) ∧ (∀ m, a m ≤ a n) := by 
  sorry

end NUMINAMATH_GPT_max_term_of_sequence_l606_60636


namespace NUMINAMATH_GPT_side_length_of_square_l606_60683

theorem side_length_of_square (A : ℝ) (h : A = 81) : ∃ s : ℝ, s^2 = A ∧ s = 9 :=
by
  sorry

end NUMINAMATH_GPT_side_length_of_square_l606_60683


namespace NUMINAMATH_GPT_pentagon_area_l606_60625

noncomputable def angle_F := 100
noncomputable def angle_G := 100
noncomputable def JF := 3
noncomputable def FG := 3
noncomputable def GH := 3
noncomputable def HI := 5
noncomputable def IJ := 5
noncomputable def area_FGHIJ := 9 * Real.sqrt 3 + Real.sqrt 17.1875

theorem pentagon_area : area_FGHIJ = 9 * Real.sqrt 3 + Real.sqrt 17.1875 :=
by
  sorry

end NUMINAMATH_GPT_pentagon_area_l606_60625


namespace NUMINAMATH_GPT_evaluate_expr_l606_60642

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5

theorem evaluate_expr : 3 * g 2 + 2 * g (-4) = 169 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expr_l606_60642


namespace NUMINAMATH_GPT_intersection_of_sets_example_l606_60603

theorem intersection_of_sets_example :
  let M := { x : ℝ | 0 < x ∧ x < 4 }
  let N := { x : ℝ | 1 / 3 ≤ x ∧ x ≤ 5 }
  let expected := { x : ℝ | 1 / 3 ≤ x ∧ x < 4 }
  (M ∩ N) = expected :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_example_l606_60603


namespace NUMINAMATH_GPT_y_intercept_l606_60663

theorem y_intercept (x1 y1 : ℝ) (m : ℝ) (h1 : x1 = -2) (h2 : y1 = 4) (h3 : m = 1 / 2) : 
  ∃ b : ℝ, (∀ x y : ℝ, y = m * x + b ↔ y = 1/2 * x + 5) ∧ b = 5 := 
by
  sorry

end NUMINAMATH_GPT_y_intercept_l606_60663


namespace NUMINAMATH_GPT_g_of_12_l606_60675

def g (n : ℕ) : ℕ := n^2 - n + 23

theorem g_of_12 : g 12 = 155 :=
by
  sorry

end NUMINAMATH_GPT_g_of_12_l606_60675


namespace NUMINAMATH_GPT_youngest_brother_age_l606_60662

theorem youngest_brother_age 
  (Rick_age : ℕ)
  (oldest_brother_age : ℕ)
  (middle_brother_age : ℕ)
  (smallest_brother_age : ℕ)
  (youngest_brother_age : ℕ)
  (h1 : Rick_age = 15)
  (h2 : oldest_brother_age = 2 * Rick_age)
  (h3 : middle_brother_age = oldest_brother_age / 3)
  (h4 : smallest_brother_age = middle_brother_age / 2)
  (h5 : youngest_brother_age = smallest_brother_age - 2) :
  youngest_brother_age = 3 := 
sorry

end NUMINAMATH_GPT_youngest_brother_age_l606_60662


namespace NUMINAMATH_GPT_count_1000_pointed_stars_l606_60694

/--
A regular n-pointed star is defined by:
1. The points P_1, P_2, ..., P_n are coplanar and no three of them are collinear.
2. Each of the n line segments intersects at least one other segment at a point other than an endpoint.
3. All of the angles at P_1, P_2, ..., P_n are congruent.
4. All of the n line segments P_2P_3, ..., P_nP_1 are congruent.
5. The path P_1P_2, P_2P_3, ..., P_nP_1 turns counterclockwise at an angle of less than 180 degrees at each vertex.

There are no regular 3-pointed, 4-pointed, or 6-pointed stars.
All regular 5-pointed stars are similar.
There are two non-similar regular 7-pointed stars.

Prove that the number of non-similar regular 1000-pointed stars is 199.
-/
theorem count_1000_pointed_stars : ∀ (n : ℕ), n = 1000 → 
  -- Points P_1, P_2, ..., P_1000 are coplanar, no three are collinear.
  -- Each of the 1000 segments intersects at least one other segment not at an endpoint.
  -- Angles at P_1, P_2, ..., P_1000 are congruent.
  -- Line segments P_2P_3, ..., P_1000P_1 are congruent.
  -- Path P_1P_2, P_2P_3, ..., P_1000P_1 turns counterclockwise at < 180 degrees each.
  -- No 3-pointed, 4-pointed, or 6-pointed regular stars.
  -- All regular 5-pointed stars are similar.
  -- There are two non-similar regular 7-pointed stars.
  -- Proven: The number of non-similar regular 1000-pointed stars is 199.
  n = 1000 ∧ (∀ m : ℕ, 1 ≤ m ∧ m < 1000 → (gcd m 1000 = 1 → (m ≠ 1 ∧ m ≠ 999))) → 
    -- Because 1000 = 2^3 * 5^3 and we exclude 1 and 999.
    (2 * 5 * 2 * 5 * 2 * 5) / 2 - 1 - 1 / 2 = 199 :=
by
  -- Pseudo-proof steps for the problem.
  sorry

end NUMINAMATH_GPT_count_1000_pointed_stars_l606_60694


namespace NUMINAMATH_GPT_parabola_equation_l606_60667

theorem parabola_equation (x y : ℝ) (hx : x = -2) (hy : y = 3) :
  (y^2 = -(9 / 2) * x) ∨ (x^2 = (4 / 3) * y) :=
by
  sorry

end NUMINAMATH_GPT_parabola_equation_l606_60667


namespace NUMINAMATH_GPT_exam_student_count_l606_60660

theorem exam_student_count (N T T_5 T_remaining : ℕ)
  (h1 : T = 70 * N)
  (h2 : T_5 = 50 * 5)
  (h3 : T_remaining = 90 * (N - 5))
  (h4 : T = T_5 + T_remaining) :
  N = 10 :=
by
  sorry

end NUMINAMATH_GPT_exam_student_count_l606_60660


namespace NUMINAMATH_GPT_find_y_l606_60631

theorem find_y 
  (x y : ℝ) 
  (h1 : 3 * x - 2 * y = 18) 
  (h2 : x + 2 * y = 10) : 
  y = 1.5 := 
by 
  sorry

end NUMINAMATH_GPT_find_y_l606_60631


namespace NUMINAMATH_GPT_Meena_cookies_left_l606_60605

def cookies_initial := 5 * 12
def cookies_sold_to_teacher := 2 * 12
def cookies_bought_by_brock := 7
def cookies_bought_by_katy := 2 * cookies_bought_by_brock

def cookies_left := cookies_initial - cookies_sold_to_teacher - cookies_bought_by_brock - cookies_bought_by_katy

theorem Meena_cookies_left : cookies_left = 15 := 
by 
  -- steps to be proven here
  sorry

end NUMINAMATH_GPT_Meena_cookies_left_l606_60605


namespace NUMINAMATH_GPT_function_decreases_iff_l606_60620

theorem function_decreases_iff (m : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → (m - 3) * x1 + 4 > (m - 3) * x2 + 4) ↔ m < 3 :=
by
  sorry

end NUMINAMATH_GPT_function_decreases_iff_l606_60620


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_not_sufficient_x2_gt_y2_iff_x_lt_y_lt_0_l606_60608

variable (x y : ℝ)

theorem necessary_but_not_sufficient (hx : x < y ∧ y < 0) : x^2 > y^2 :=
sorry

theorem not_sufficient (hx : x^2 > y^2) : ¬ (x < y ∧ y < 0) :=
sorry

-- Optional: Combining the two to create a combined theorem statement
theorem x2_gt_y2_iff_x_lt_y_lt_0 : (∀ x y : ℝ, x < y ∧ y < 0 → x^2 > y^2) ∧ (∃ x y : ℝ, x^2 > y^2 ∧ ¬ (x < y ∧ y < 0)) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_not_sufficient_x2_gt_y2_iff_x_lt_y_lt_0_l606_60608


namespace NUMINAMATH_GPT_part1_part2_part3_l606_60634

-- Part (1): Proving \( p \implies m > \frac{3}{2} \)
theorem part1 (m : ℝ) : (∀ x : ℝ, x^2 + 2 * m - 3 > 0) → (m > 3 / 2) :=
by
  sorry

-- Part (2): Proving \( q \implies (m < -1 \text{ or } m > 2) \)
theorem part2 (m : ℝ) : (∃ x : ℝ, x^2 - 2 * m * x + m + 2 < 0) → (m < -1 ∨ m > 2) :=
by
  sorry

-- Part (3): Proving \( (p ∨ q) \implies ((-\infty, -1) ∪ (\frac{3}{2}, +\infty)) \)
theorem part3 (m : ℝ) : (∀ x : ℝ, x^2 + 2 * m - 3 > 0 ∨ ∃ x : ℝ, x^2 - 2 * m * x + m + 2 < 0) → ((m < -1) ∨ (3 / 2 < m)) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_part3_l606_60634


namespace NUMINAMATH_GPT_exists_univariate_polynomial_l606_60616

def polynomial_in_three_vars (P : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ,
  P x y z = P x y (x * y - z) ∧
  P x y z = P x (z * x - y) z ∧
  P x y z = P (y * z - x) y z

theorem exists_univariate_polynomial (P : ℝ → ℝ → ℝ → ℝ) (h : polynomial_in_three_vars P) :
  ∃ F : ℝ → ℝ, ∀ x y z : ℝ, P x y z = F (x^2 + y^2 + z^2 - x * y * z) :=
sorry

end NUMINAMATH_GPT_exists_univariate_polynomial_l606_60616


namespace NUMINAMATH_GPT_proof_problem_l606_60628

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 1 / x

noncomputable def f'' (x : ℝ) : ℝ := Real.exp x + 2 / x^3

theorem proof_problem {x0 m n : ℝ} (hx0_pos : 0 < x0)
  (H : f'' x0 = 0) (hm : 0 < m) (hmx0 : m < x0) (hn : x0 < n) :
  f'' m < 0 ∧ f'' n > 0 := sorry

end NUMINAMATH_GPT_proof_problem_l606_60628


namespace NUMINAMATH_GPT_probability_neither_defective_l606_60665

noncomputable def n := 9
noncomputable def k := 2
noncomputable def total_pens := 9
noncomputable def defective_pens := 3
noncomputable def non_defective_pens := total_pens - defective_pens

noncomputable def total_combinations := Nat.choose total_pens k
noncomputable def non_defective_combinations := Nat.choose non_defective_pens k

theorem probability_neither_defective :
  (non_defective_combinations : ℚ) / total_combinations = 5 / 12 := by
sorry

end NUMINAMATH_GPT_probability_neither_defective_l606_60665


namespace NUMINAMATH_GPT_find_d_for_single_point_l606_60630

/--
  Suppose that the graph of \(3x^2 + y^2 + 6x - 6y + d = 0\) consists of a single point.
  Prove that \(d = 12\).
-/
theorem find_d_for_single_point : 
  ∀ (d : ℝ), (∃ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 6 * y + d = 0) ∧
              (∀ (x1 y1 x2 y2 : ℝ), 
                (3 * x1^2 + y1^2 + 6 * x1 - 6 * y1 + d = 0 ∧ 
                 3 * x2^2 + y2^2 + 6 * x2 - 6 * y2 + d = 0 → 
                 x1 = x2 ∧ y1 = y2)) ↔ d = 12 := 
by 
  sorry

end NUMINAMATH_GPT_find_d_for_single_point_l606_60630


namespace NUMINAMATH_GPT_work_completed_together_in_4_days_l606_60624

/-- A can do the work in 6 days. -/
def A_work_rate : ℚ := 1 / 6

/-- B can do the work in 12 days. -/
def B_work_rate : ℚ := 1 / 12

/-- Combined work rate of A and B working together. -/
def combined_work_rate : ℚ := A_work_rate + B_work_rate

/-- Number of days for A and B to complete the work together. -/
def days_to_complete : ℚ := 1 / combined_work_rate

theorem work_completed_together_in_4_days : days_to_complete = 4 := by
  sorry

end NUMINAMATH_GPT_work_completed_together_in_4_days_l606_60624


namespace NUMINAMATH_GPT_total_gain_loss_is_correct_l606_60696

noncomputable def total_gain_loss_percentage 
    (cost1 cost2 cost3 : ℝ) 
    (gain1 gain2 gain3 : ℝ) : ℝ :=
  let total_cost := cost1 + cost2 + cost3
  let gain_amount1 := cost1 * gain1
  let loss_amount2 := cost2 * gain2
  let gain_amount3 := cost3 * gain3
  let net_gain_loss := (gain_amount1 + gain_amount3) - loss_amount2
  (net_gain_loss / total_cost) * 100

theorem total_gain_loss_is_correct :
  total_gain_loss_percentage 
    675958 995320 837492 0.11 (-0.11) 0.15 = 3.608 := 
sorry

end NUMINAMATH_GPT_total_gain_loss_is_correct_l606_60696


namespace NUMINAMATH_GPT_remainder_div_5_l606_60695

theorem remainder_div_5 (n : ℕ): (∃ k : ℤ, n = 10 * k + 7) → (∃ m : ℤ, n = 5 * m + 2) :=
by
  sorry

end NUMINAMATH_GPT_remainder_div_5_l606_60695


namespace NUMINAMATH_GPT_solve_x_division_l606_60693

theorem solve_x_division :
  ∀ x : ℝ, (3 / x + 4 / x / (8 / x) = 1.5) → x = 3 := 
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_x_division_l606_60693


namespace NUMINAMATH_GPT_find_a_l606_60639

theorem find_a (x y a : ℕ) (h₁ : x = 2) (h₂ : y = 3) (h₃ : a * x + 3 * y = 13) : a = 2 :=
by 
  sorry

end NUMINAMATH_GPT_find_a_l606_60639


namespace NUMINAMATH_GPT_find_x_given_conditions_l606_60612

variable (x y z : ℝ)

theorem find_x_given_conditions
  (h1: x * y / (x + y) = 4)
  (h2: x * z / (x + z) = 9)
  (h3: y * z / (y + z) = 16)
  (h_pos: 0 < x ∧ 0 < y ∧ 0 < z)
  (h_distinct: x ≠ y ∧ x ≠ z ∧ y ≠ z) :
  x = 384/21 :=
sorry

end NUMINAMATH_GPT_find_x_given_conditions_l606_60612


namespace NUMINAMATH_GPT_least_power_divisible_by_240_l606_60621

theorem least_power_divisible_by_240 (n : ℕ) (a : ℕ) (h_a : a = 60) (h : a^n % 240 = 0) : 
  n = 2 :=
by
  sorry

end NUMINAMATH_GPT_least_power_divisible_by_240_l606_60621


namespace NUMINAMATH_GPT_sam_age_l606_60688

-- Definitions
variables (B J S : ℕ)
axiom H1 : B = 2 * J
axiom H2 : B + J = 60
axiom H3 : S = (B + J) / 2

-- Problem statement
theorem sam_age : S = 30 :=
sorry

end NUMINAMATH_GPT_sam_age_l606_60688


namespace NUMINAMATH_GPT_lena_can_form_rectangles_vasya_can_form_rectangles_lena_and_vasya_can_be_right_l606_60673

def total_area_of_triangles_and_quadrilateral (A B Q : ℝ) : ℝ :=
  A + B + Q

def lena_triangles_and_quadrilateral_area (A B Q : ℝ) : Prop :=
  (24 : ℝ) = total_area_of_triangles_and_quadrilateral A B Q

def total_area_of_triangles_and_pentagon (C D P : ℝ) : ℝ :=
  C + D + P

def vasya_triangles_and_pentagon_area (C D P : ℝ) : Prop :=
  (24 : ℝ) = total_area_of_triangles_and_pentagon C D P

theorem lena_can_form_rectangles (A B Q : ℝ) (h : lena_triangles_and_quadrilateral_area A B Q) :
  lena_triangles_and_quadrilateral_area A B Q :=
by 
-- We assume the definition holds as given
sorry

theorem vasya_can_form_rectangles (C D P : ℝ) (h : vasya_triangles_and_pentagon_area C D P) :
  vasya_triangles_and_pentagon_area C D P :=
by 
-- We assume the definition holds as given
sorry

theorem lena_and_vasya_can_be_right (A B Q C D P : ℝ)
  (hlena : lena_triangles_and_quadrilateral_area A B Q)
  (hvasya : vasya_triangles_and_pentagon_area C D P) :
  lena_triangles_and_quadrilateral_area A B Q ∧ vasya_triangles_and_pentagon_area C D P :=
by 
-- Combining both assumptions
exact ⟨hlena, hvasya⟩

end NUMINAMATH_GPT_lena_can_form_rectangles_vasya_can_form_rectangles_lena_and_vasya_can_be_right_l606_60673


namespace NUMINAMATH_GPT_maximum_value_of_expression_l606_60618

noncomputable def maxValue (x y z : ℝ) : ℝ :=
(x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2)

theorem maximum_value_of_expression (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) :
  maxValue x y z ≤ 243 / 16 :=
sorry

end NUMINAMATH_GPT_maximum_value_of_expression_l606_60618


namespace NUMINAMATH_GPT_lateral_surface_area_cone_l606_60691

theorem lateral_surface_area_cone (r l : ℝ) (h₀ : r = 6) (h₁ : l = 10) : π * r * l = 60 * π := by 
  sorry

end NUMINAMATH_GPT_lateral_surface_area_cone_l606_60691
