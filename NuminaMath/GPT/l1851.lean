import Mathlib

namespace initial_students_count_l1851_185156

-- Definitions based on conditions
def initial_average_age (T : ℕ) (n : ℕ) : Prop := T = 14 * n
def new_average_age_after_adding (T : ℕ) (n : ℕ) : Prop := (T + 5 * 17) / (n + 5) = 15

-- Main proposition stating the problem
theorem initial_students_count (n : ℕ) (T : ℕ) 
  (h1 : initial_average_age T n)
  (h2 : new_average_age_after_adding T n) :
  n = 10 :=
by
  sorry

end initial_students_count_l1851_185156


namespace root_condition_l1851_185186

noncomputable def f (a : ℝ) (x : ℝ) := a * x^3 - 3 * x^2 + 1

theorem root_condition (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ = 0 ∧ ∀ x ≠ x₀, f a x ≠ 0 ∧ x₀ < 0) → a > 2 :=
sorry

end root_condition_l1851_185186


namespace correct_multiplication_result_l1851_185124

theorem correct_multiplication_result (x : ℕ) (h : x - 6 = 51) : x * 6 = 342 :=
  by
  sorry

end correct_multiplication_result_l1851_185124


namespace num_terms_arithmetic_sequence_is_41_l1851_185147

-- Definitions and conditions
def first_term : ℤ := 200
def common_difference : ℤ := -5
def last_term : ℤ := 0

-- Definition of the n-th term of arithmetic sequence
def nth_term (a : ℤ) (d : ℤ) (n : ℤ) : ℤ :=
  a + (n - 1) * d

-- Statement to prove
theorem num_terms_arithmetic_sequence_is_41 : 
  ∃ n : ℕ, nth_term first_term common_difference n = 0 ∧ n = 41 :=
by 
  sorry

end num_terms_arithmetic_sequence_is_41_l1851_185147


namespace fixed_point_coordinates_l1851_185157

theorem fixed_point_coordinates (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : (2, 2) ∈ {p : ℝ × ℝ | ∃ x : ℝ, p = (x, a^(x-2) + 1)} := 
by
  -- Proof goes here
  sorry

end fixed_point_coordinates_l1851_185157


namespace simplify_expression_l1851_185113

noncomputable def algebraic_expression (a : ℚ) (h1 : a ≠ -2) (h2 : a ≠ 2) (h3 : a ≠ 1) : ℚ :=
(1 - 3 / (a + 2)) / ((a^2 - 2 * a + 1) / (a^2 - 4))

theorem simplify_expression (a : ℚ) (h1 : a ≠ -2) (h2 : a ≠ 2) (h3 : a ≠ 1) :
  algebraic_expression a h1 h2 h3 = (a - 2) / (a - 1) :=
by
  sorry

end simplify_expression_l1851_185113


namespace six_a_seven_eight_b_div_by_45_l1851_185161

/-- If the number 6a78b is divisible by 45, then a + b = 6. -/
theorem six_a_seven_eight_b_div_by_45 (a b : ℕ) (h1: 0 ≤ a ∧ a < 10) (h2: 0 ≤ b ∧ b < 10)
  (h3 : (6 * 10^4 + a * 10^3 + 7 * 10^2 + 8 * 10 + b) % 45 = 0) : a + b = 6 := 
by
  sorry

end six_a_seven_eight_b_div_by_45_l1851_185161


namespace evaluate_f_at_7_l1851_185131

theorem evaluate_f_at_7 (f : ℝ → ℝ)
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, f (x + 4) = f x)
  (h3 : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = -x + 4) :
  f 7 = -3 :=
by
  sorry

end evaluate_f_at_7_l1851_185131


namespace consecutive_odd_integer_sum_l1851_185166

theorem consecutive_odd_integer_sum {n : ℤ} (h1 : n = 17 ∨ n + 2 = 17) (h2 : n + n + 2 ≥ 36) : (n = 17 → n + 2 = 19) ∧ (n + 2 = 17 → n = 15) :=
by
  sorry

end consecutive_odd_integer_sum_l1851_185166


namespace find_other_root_l1851_185115

theorem find_other_root (a b : ℝ) (h₁ : 3^2 + 3 * a - 2 * a = 0) (h₂ : ∀ x, x^2 + a * x - 2 * a = 0 → (x = 3 ∨ x = b)) :
  b = 6 := 
sorry

end find_other_root_l1851_185115


namespace smallest_value_arithmetic_geometric_seq_l1851_185180

theorem smallest_value_arithmetic_geometric_seq :
  ∃ (E F G H : ℕ), (E < F) ∧ (F < G) ∧ (F * 4 = G * 7) ∧ (E + G = 2 * F) ∧ (F * F * 49 = G * G * 16) ∧ (E + F + G + H = 97) := 
sorry

end smallest_value_arithmetic_geometric_seq_l1851_185180


namespace systematic_sampling_remove_l1851_185110

theorem systematic_sampling_remove (total_people : ℕ) (sample_size : ℕ) (remove_count : ℕ): 
  total_people = 162 → sample_size = 16 → remove_count = 2 → 
  (total_people - 1) % sample_size = sample_size - 1 :=
by
  sorry

end systematic_sampling_remove_l1851_185110


namespace general_formula_sum_first_n_terms_l1851_185174

open BigOperators

def geometric_sequence (a_3 : ℚ) (q : ℚ) : ℕ → ℚ
| 0       => 1 -- this is a placeholder since sequence usually start from 1
| (n + 1) => 1 * q ^ n

def sum_geometric_sequence (a_1 q : ℚ) (n : ℕ) : ℚ :=
  a_1 * (1 - q ^ n) / (1 - q)

theorem general_formula (a_3 : ℚ) (q : ℚ) (n : ℕ) (h_a3 : a_3 = 1 / 4) (h_q : q = -1 / 2) :
  geometric_sequence a_3 q (n + 1) = (-1 / 2) ^ n :=
by
  sorry

theorem sum_first_n_terms (a_1 q : ℚ) (n : ℕ) (h_a1 : a_1 = 1) (h_q : q = -1 / 2) :
  sum_geometric_sequence a_1 q n = 2 / 3 * (1 - (-1 / 2) ^ n) :=
by
  sorry

end general_formula_sum_first_n_terms_l1851_185174


namespace area_of_triangle_aef_l1851_185146

noncomputable def length_ab : ℝ := 10
noncomputable def width_ad : ℝ := 6
noncomputable def diagonal_ac : ℝ := Real.sqrt (length_ab^2 + width_ad^2)
noncomputable def segment_length_ac : ℝ := diagonal_ac / 4
noncomputable def area_aef : ℝ := (1/2) * segment_length_ac * ((60 * diagonal_ac) / diagonal_ac^2)

theorem area_of_triangle_aef : area_aef = 7.5 := by
  sorry

end area_of_triangle_aef_l1851_185146


namespace sum_of_digits_l1851_185126

theorem sum_of_digits (a b c d : ℕ) (h_diff : ∀ x y : ℕ, (x = a ∨ x = b ∨ x = c ∨ x = d) → (y = a ∨ y = b ∨ y = c ∨ y = d) → x ≠ y) (h1 : a + c = 10) (h2 : b + c = 8) (h3 : a + d = 11) : 
  a + b + c + d = 18 :=
by
  sorry

end sum_of_digits_l1851_185126


namespace negate_proposition_l1851_185132

theorem negate_proposition :
  (¬ ∃ x : ℝ, x^2 + x - 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + x - 2 > 0) := 
sorry

end negate_proposition_l1851_185132


namespace find_c_l1851_185171

theorem find_c (c : ℝ) (h1 : 0 < c) (h2 : c < 3) (h3 : abs (6 + 4 * c) = 14) : c = 2 :=
by {
  sorry
}

end find_c_l1851_185171


namespace parity_of_expression_l1851_185149

theorem parity_of_expression (o1 o2 n : ℕ) (h1 : o1 % 2 = 1) (h2 : o2 % 2 = 1) : 
  ((o1 * o1 + n * (o1 * o2)) % 2 = 1 ↔ n % 2 = 0) :=
by sorry

end parity_of_expression_l1851_185149


namespace part1_part2_l1851_185179

noncomputable def h (x : ℝ) : ℝ := x^2

noncomputable def phi (x : ℝ) : ℝ := 2 * Real.exp 1 * Real.log x

noncomputable def F (x : ℝ) : ℝ := h x - phi x

theorem part1 :
  ∃ (x : ℝ), x > 0 ∧ Real.log x = 1 ∧ F x = 0 :=
sorry

theorem part2 :
  ∃ (k b : ℝ), 
  (∀ x > 0, h x ≥ k * x + b) ∧
  (∀ x > 0, phi x ≤ k * x + b) ∧
  (k = 2 * Real.exp 1 ∧ b = -Real.exp 1) :=
sorry

end part1_part2_l1851_185179


namespace even_quadratic_increasing_l1851_185191

theorem even_quadratic_increasing (m : ℝ) (h : ∀ x : ℝ, (m-1)*x^2 + 2*m*x + 1 = (m-1)*(-x)^2 + 2*m*(-x) + 1) :
  ∀ x1 x2 : ℝ, x1 < x2 ∧ x2 ≤ 0 → ((m-1)*x1^2 + 2*m*x1 + 1) < ((m-1)*x2^2 + 2*m*x2 + 1) :=
sorry

end even_quadratic_increasing_l1851_185191


namespace gabrielle_total_crates_l1851_185101

theorem gabrielle_total_crates (monday tuesday wednesday thursday : ℕ)
  (h_monday : monday = 5)
  (h_tuesday : tuesday = 2 * monday)
  (h_wednesday : wednesday = tuesday - 2)
  (h_thursday : thursday = tuesday / 2) :
  monday + tuesday + wednesday + thursday = 28 :=
by
  sorry

end gabrielle_total_crates_l1851_185101


namespace sandy_age_correct_l1851_185102

def is_age_ratio (S M : ℕ) : Prop := S * 9 = M * 7
def is_age_difference (S M : ℕ) : Prop := M = S + 12

theorem sandy_age_correct (S M : ℕ) (h1 : is_age_ratio S M) (h2 : is_age_difference S M) : S = 42 := by
  sorry

end sandy_age_correct_l1851_185102


namespace arcsin_one_eq_pi_div_two_l1851_185158

noncomputable def arcsin : ℝ → ℝ := sorry -- Define arcsin function

theorem arcsin_one_eq_pi_div_two : arcsin 1 = Real.pi / 2 := sorry

end arcsin_one_eq_pi_div_two_l1851_185158


namespace find_x_set_l1851_185185

theorem find_x_set (a : ℝ) (h : 0 < a ∧ a < 1) : 
  {x : ℝ | a ^ (x + 3) > a ^ (2 * x)} = {x : ℝ | x > 3} :=
sorry

end find_x_set_l1851_185185


namespace intersection_A_B_l1851_185105

def A : Set Real := { y | ∃ x : Real, y = Real.cos x }
def B : Set Real := { x | x^2 < 9 }

theorem intersection_A_B : A ∩ B = { y | -1 ≤ y ∧ y ≤ 1 } :=
by
  sorry

end intersection_A_B_l1851_185105


namespace combine_like_terms_l1851_185145

variable (a : ℝ)

theorem combine_like_terms : 3 * a^2 + 5 * a^2 - a^2 = 7 * a^2 := 
by sorry

end combine_like_terms_l1851_185145


namespace chord_ratio_l1851_185128

variable (XQ WQ YQ ZQ : ℝ)

theorem chord_ratio (h1 : XQ = 5) (h2 : WQ = 7) (h3 : XQ * YQ = WQ * ZQ) : YQ / ZQ = 7 / 5 :=
by
  sorry

end chord_ratio_l1851_185128


namespace value_of_x0_l1851_185108

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / x
noncomputable def f_deriv (x : ℝ) : ℝ := ((x - 1) * Real.exp x) / (x * x)

theorem value_of_x0 (x0 : ℝ) (h : f_deriv x0 = -f x0) : x0 = 1 / 2 := by
  sorry

end value_of_x0_l1851_185108


namespace eight_digit_numbers_with_012_eight_digit_numbers_with_00012222_eight_digit_numbers_starting_with_1_0002222_l1851_185154

theorem eight_digit_numbers_with_012 :
  let total_sequences := 3^8 
  let invalid_sequences := 3^7 
  total_sequences - invalid_sequences = 4374 :=
by sorry

theorem eight_digit_numbers_with_00012222 :
  let total_sequences := Nat.factorial 8 / (Nat.factorial 3 * Nat.factorial 4)
  let invalid_sequences := Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 4)
  total_sequences - invalid_sequences = 175 :=
by sorry

theorem eight_digit_numbers_starting_with_1_0002222 :
  let number_starting_with_1 := Nat.factorial 7 / (Nat.factorial 3 * Nat.factorial 4)
  number_starting_with_1 = 35 :=
by sorry

end eight_digit_numbers_with_012_eight_digit_numbers_with_00012222_eight_digit_numbers_starting_with_1_0002222_l1851_185154


namespace f_positive_l1851_185148

noncomputable def f (x : ℝ) : ℝ := (1/3)^x - Real.log x / Real.log 2

variables (x0 x1 : ℝ)

theorem f_positive (hx0 : f x0 = 0) (hx1 : 0 < x1) (hx0_gt_x1 : x1 < x0) : 0 < f x1 :=
sorry

end f_positive_l1851_185148


namespace complement_of_intersection_l1851_185169

def S : Set ℝ := {-2, -1, 0, 1, 2}
def T : Set ℝ := {x | x + 1 ≤ 2}
def complement (A B : Set ℝ) : Set ℝ := {x ∈ B | x ∉ A}

theorem complement_of_intersection :
  complement (S ∩ T) S = {2} :=
by
  sorry

end complement_of_intersection_l1851_185169


namespace sqrt_mul_power_expr_l1851_185107

theorem sqrt_mul_power_expr : ( (Real.sqrt 3 + Real.sqrt 2) ^ 2023 * (Real.sqrt 3 - Real.sqrt 2) ^ 2022 ) = (Real.sqrt 3 + Real.sqrt 2) := 
  sorry

end sqrt_mul_power_expr_l1851_185107


namespace Spot_dog_reachable_area_l1851_185100

noncomputable def Spot_reachable_area (side_length tether_length : ℝ) : ℝ := 
  -- Note here we compute using the areas described in the problem
  6 * Real.pi * (tether_length^2) / 3 - Real.pi * (side_length^2)

theorem Spot_dog_reachable_area (side_length tether_length : ℝ)
  (H1 : side_length = 2) (H2 : tether_length = 3) :
    Spot_reachable_area side_length tether_length = (22 * Real.pi) / 3 := by
  sorry

end Spot_dog_reachable_area_l1851_185100


namespace car_average_speed_l1851_185167

theorem car_average_speed :
  let distance_uphill := 100
  let distance_downhill := 50
  let speed_uphill := 30
  let speed_downhill := 80
  let total_distance := distance_uphill + distance_downhill
  let time_uphill := distance_uphill / speed_uphill
  let time_downhill := distance_downhill / speed_downhill
  let total_time := time_uphill + time_downhill
  let average_speed := total_distance / total_time
  average_speed = 37.92 := by
  sorry

end car_average_speed_l1851_185167


namespace chinese_chess_draw_probability_l1851_185193

theorem chinese_chess_draw_probability (pMingNotLosing : ℚ) (pDongLosing : ℚ) : 
    pMingNotLosing = 3/4 → 
    pDongLosing = 1/2 → 
    (pMingNotLosing - (1 - pDongLosing)) = 1/4 :=
by
  intros
  sorry

end chinese_chess_draw_probability_l1851_185193


namespace no_int_solutions_a_b_l1851_185176

theorem no_int_solutions_a_b :
  ¬ ∃ (a b : ℤ), a^2 + 1998 = b^2 :=
by
  sorry

end no_int_solutions_a_b_l1851_185176


namespace julie_age_end_of_period_is_15_l1851_185111

-- Define necessary constants and variables
def hours_per_day : ℝ := 3
def pay_rate_per_hour_per_year : ℝ := 0.75
def total_days_worked : ℝ := 60
def total_earnings : ℝ := 810

-- Define Julie's age at the end of the four-month period
def julies_age_end_of_period (age: ℝ) : Prop :=
  hours_per_day * pay_rate_per_hour_per_year * age * total_days_worked = total_earnings

-- The final Lean 4 statement that needs proof
theorem julie_age_end_of_period_is_15 : ∃ age : ℝ, julies_age_end_of_period age ∧ age = 15 :=
by {
  sorry
}

end julie_age_end_of_period_is_15_l1851_185111


namespace rainfall_ratio_l1851_185198

theorem rainfall_ratio (R_1 R_2 : ℕ) (h1 : R_1 + R_2 = 25) (h2 : R_2 = 15) : R_2 / R_1 = 3 / 2 :=
by
  sorry

end rainfall_ratio_l1851_185198


namespace product_of_common_ratios_l1851_185175

theorem product_of_common_ratios (x p r a2 a3 b2 b3 : ℝ)
  (h1 : a2 = x * p) (h2 : a3 = x * p^2)
  (h3 : b2 = x * r) (h4 : b3 = x * r^2)
  (h5 : 3 * a3 - 4 * b3 = 5 * (3 * a2 - 4 * b2))
  (h_nonconstant : x ≠ 0) (h_diff_ratios : p ≠ r) :
  p * r = 9 :=
by
  sorry

end product_of_common_ratios_l1851_185175


namespace third_part_of_division_l1851_185122

noncomputable def divide_amount (total_amount : ℝ) : (ℝ × ℝ × ℝ) :=
  let part1 := (1/2)/(1/2 + 2/3 + 3/4) * total_amount
  let part2 := (2/3)/(1/2 + 2/3 + 3/4) * total_amount
  let part3 := (3/4)/(1/2 + 2/3 + 3/4) * total_amount
  (part1, part2, part3)

theorem third_part_of_division :
  divide_amount 782 = (261.0, 214.66666666666666, 306.0) :=
by
  sorry

end third_part_of_division_l1851_185122


namespace james_hours_per_day_l1851_185134

theorem james_hours_per_day (h : ℕ) (rental_rate : ℕ) (days_per_week : ℕ) (weekly_income : ℕ)
  (H1 : rental_rate = 20)
  (H2 : days_per_week = 4)
  (H3 : weekly_income = 640)
  (H4 : rental_rate * days_per_week * h = weekly_income) :
  h = 8 :=
sorry

end james_hours_per_day_l1851_185134


namespace riding_is_four_times_walking_l1851_185135

variable (D : ℝ) -- Total distance of the route
variable (v_r v_w : ℝ) -- Riding speed and walking speed
variable (t_r t_w : ℝ) -- Time spent riding and walking

-- Conditions given in the problem
axiom distance_riding : (2/3) * D = v_r * t_r
axiom distance_walking : (1/3) * D = v_w * t_w
axiom time_relation : t_w = 2 * t_r

-- Desired statement to prove
theorem riding_is_four_times_walking : v_r = 4 * v_w := by
  sorry

end riding_is_four_times_walking_l1851_185135


namespace tetrahedron_volume_from_cube_l1851_185127

theorem tetrahedron_volume_from_cube {s : ℝ} (h : s = 8) :
  let cube_volume := s^3
  let smaller_tetrahedron_volume := (1/3) * (1/2) * s * s * s
  let total_smaller_tetrahedron_volume := 4 * smaller_tetrahedron_volume
  let tetrahedron_volume := cube_volume - total_smaller_tetrahedron_volume
  tetrahedron_volume = 170.6666 :=
by
  sorry

end tetrahedron_volume_from_cube_l1851_185127


namespace quadratic_radical_same_type_l1851_185121

theorem quadratic_radical_same_type (a : ℝ) (h : (∃ (t : ℝ), t ^ 2 = 3 * a - 4) ∧ (∃ (t : ℝ), t ^ 2 = 8)) : a = 2 :=
by
  -- Extract the properties of the radicals
  sorry

end quadratic_radical_same_type_l1851_185121


namespace range_of_a_l1851_185129

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := if x ≥ 1 then x * Real.log x - a * x^2 else a^x

theorem range_of_a (a : ℝ) (f_decreasing : ∀ x y : ℝ, x ≤ y → f x a ≥ f y a) : 
  1/2 ≤ a ∧ a < 1 :=
by
  sorry

end range_of_a_l1851_185129


namespace largest_integral_x_l1851_185112

theorem largest_integral_x (x : ℤ) : 
  (1 / 4 : ℝ) < (x / 7) ∧ (x / 7) < (7 / 11 : ℝ) → x ≤ 4 := 
  sorry

end largest_integral_x_l1851_185112


namespace sin_alpha_beta_eq_l1851_185119

theorem sin_alpha_beta_eq 
  (α β : ℝ) 
  (h1 : π / 4 < α) (h2 : α < 3 * π / 4)
  (h3 : 0 < β) (h4 : β < π / 4)
  (h5: Real.sin (α + π / 4) = 3 / 5)
  (h6: Real.cos (π / 4 + β) = 5 / 13) :
  Real.sin (α + β) = 56 / 65 :=
sorry

end sin_alpha_beta_eq_l1851_185119


namespace probability_of_ge_four_is_one_eighth_l1851_185120

noncomputable def probability_ge_four : ℝ :=
sorry

theorem probability_of_ge_four_is_one_eighth :
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ 2) ∧ (0 ≤ y ∧ y ≤ 2) →
  (probability_ge_four = 1 / 8) :=
sorry

end probability_of_ge_four_is_one_eighth_l1851_185120


namespace combinations_problem_l1851_185181

open Nat

-- Definitions for combinations
def C (n k : Nat) : Nat :=
  factorial n / (factorial k * factorial (n - k))

-- Condition: Number of ways to choose 2 sergeants out of 6
def C_6_2 : Nat := C 6 2

-- Condition: Number of ways to choose 20 soldiers out of 60
def C_60_20 : Nat := C 60 20

-- Theorem statement for the problem
theorem combinations_problem :
  3 * C_6_2 * C_60_20 = 3 * 15 * C 60 20 := by
  simp [C_6_2, C_60_20, C]
  sorry

end combinations_problem_l1851_185181


namespace equivalent_integer_l1851_185138

theorem equivalent_integer (a b n : ℤ) (h1 : a ≡ 33 [ZMOD 60]) (h2 : b ≡ 85 [ZMOD 60]) (hn : 200 ≤ n ∧ n ≤ 251) : 
  a - b ≡ 248 [ZMOD 60] :=
sorry

end equivalent_integer_l1851_185138


namespace quadratic_distinct_real_roots_l1851_185152

theorem quadratic_distinct_real_roots (m : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ^ 2 - 2 * x₁ + m = 0 ∧ x₂ ^ 2 - 2 * x₂ + m = 0) ↔ m < 1 :=
by sorry

end quadratic_distinct_real_roots_l1851_185152


namespace determine_constants_l1851_185192

theorem determine_constants (P Q R : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
    (-2 * x^2 + 5 * x - 7) / (x^3 - x) = P / x + (Q * x + R) / (x^2 - 1)) ↔
    (P = 7 ∧ Q = -9 ∧ R = 5) :=
by
  sorry

end determine_constants_l1851_185192


namespace simplify_expression_l1851_185130

-- Definitions for conditions and parameters
variables {x y : ℝ}

-- The problem statement and proof
theorem simplify_expression : 12 * x^5 * y / (6 * x * y) = 2 * x^4 :=
by sorry

end simplify_expression_l1851_185130


namespace distance_between_points_l1851_185196

theorem distance_between_points : abs (3 - (-2)) = 5 := 
by
  sorry

end distance_between_points_l1851_185196


namespace hyperbola_eccentricity_asymptotes_l1851_185164

theorem hyperbola_eccentricity_asymptotes :
  (∃ e: ℝ, ∃ m: ℝ, 
    (∀ x y, (x^2 / 8 - y^2 / 4 = 1) → e = Real.sqrt 6 / 2 ∧ y = m * x) ∧ 
    (m = Real.sqrt 2 / 2 ∨ m = -Real.sqrt 2 / 2)) :=
sorry

end hyperbola_eccentricity_asymptotes_l1851_185164


namespace simplify_exponential_expression_l1851_185168

theorem simplify_exponential_expression :
  (3 * (-5)^2)^(3/4) = (75)^(3/4) := 
  sorry

end simplify_exponential_expression_l1851_185168


namespace relationship_of_a_b_l1851_185109

theorem relationship_of_a_b
  (a b : Real)
  (h1 : a < 0)
  (h2 : b > 0)
  (h3 : a + b < 0) : 
  -a > b ∧ b > -b ∧ -b > a := 
by
  sorry

end relationship_of_a_b_l1851_185109


namespace tan_angle_addition_l1851_185183

theorem tan_angle_addition (x : ℝ) (h : Real.tan x = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = -Real.sqrt 3 :=
sorry

end tan_angle_addition_l1851_185183


namespace sum_of_abs_values_l1851_185190

-- Define the problem conditions
variable (a b c d m : ℤ)
variable (h1 : a + b + c + d = 1)
variable (h2 : a * b + a * c + a * d + b * c + b * d + c * d = 0)
variable (h3 : a * b * c + a * b * d + a * c * d + b * c * d = -4023)
variable (h4 : a * b * c * d = m)

-- Prove the required sum of absolute values
theorem sum_of_abs_values : |a| + |b| + |c| + |d| = 621 :=
by
  sorry

end sum_of_abs_values_l1851_185190


namespace find_set_l1851_185163

/-- Definition of set A -/
def setA : Set ℝ := { x : ℝ | abs x < 4 }

/-- Definition of set B -/
def setB : Set ℝ := { x : ℝ | x^2 - 4 * x + 3 > 0 }

/-- Definition of the intersection A ∩ B -/
def intersectionAB : Set ℝ := { x : ℝ | abs x < 4 ∧ (x > 3 ∨ x < 1) }

/-- Definition of the set we want to find -/
def setDesired : Set ℝ := { x : ℝ | abs x < 4 ∧ ¬(abs x < 4 ∧ (x > 3 ∨ x < 1)) }

/-- The statement to prove -/
theorem find_set :
  setDesired = { x : ℝ | 1 ≤ x ∧ x ≤ 3 } :=
sorry

end find_set_l1851_185163


namespace circle_equation_l1851_185106

-- Defining the given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 4 * y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - x = 0

-- Defining the point
def point : ℝ × ℝ := (1, -1)

-- Proving the equation of the new circle that passes through the intersection points 
-- of the given circles and the given point
theorem circle_equation (x y : ℝ) :
  (circle1 x y ∧ circle2 x y ∧ x = 1 ∧ y = -1) → 9 * x^2 + 9 * y^2 - 14 * x + 4 * y = 0 :=
sorry

end circle_equation_l1851_185106


namespace sufficient_not_necessary_l1851_185177

variable (p q : Prop)

theorem sufficient_not_necessary (h1 : p ∧ q) (h2 : ¬¬p) : ¬¬p :=
by
  sorry

end sufficient_not_necessary_l1851_185177


namespace full_price_tickets_count_l1851_185197

def num_tickets_reduced := 5400
def total_tickets := 25200
def num_tickets_full := 5 * num_tickets_reduced

theorem full_price_tickets_count :
  num_tickets_reduced + num_tickets_full = total_tickets → num_tickets_full = 27000 :=
by
  sorry

end full_price_tickets_count_l1851_185197


namespace fraction_savings_on_makeup_l1851_185188

theorem fraction_savings_on_makeup (savings : ℝ) (sweater_cost : ℝ) (makeup_cost : ℝ) (h_savings : savings = 80) (h_sweater : sweater_cost = 20) (h_makeup : makeup_cost = savings - sweater_cost) : makeup_cost / savings = 3 / 4 := by
  sorry

end fraction_savings_on_makeup_l1851_185188


namespace pq_sum_l1851_185173

theorem pq_sum (p q : ℝ) 
  (h1 : p / 3 = 9) 
  (h2 : q / 3 = 15) : 
  p + q = 72 :=
sorry

end pq_sum_l1851_185173


namespace monotonicity_F_range_k_l1851_185136

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x) - Real.log (1 - x)
noncomputable def F (x : ℝ) (a : ℝ) : ℝ := f x + a * x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := f x - k * (x^3 - 3 * x)

theorem monotonicity_F (a : ℝ) (ha : a ≠ 0) :
(∀ x : ℝ, (-1 < x ∧ x < 1) → 
    (if (-2 ≤ a ∧ a < 0) ∨ (a > 0) then 0 ≤ (a - a * x^2 + 2) / (1 - x^2)
     else if a < -2 then 
        ((-1 < x ∧ x < -Real.sqrt ((a + 2) / a)) ∨ (Real.sqrt ((a + 2) / a) < x ∧ x < 1)) → 0 ≤ (a - a * x^2 + 2) / (1 - x^2) ∧ 
        (-Real.sqrt ((a + 2) / a) < x ∧ x < Real.sqrt ((a + 2) / a)) → 0 > (a - a * x^2 + 2) / (1 - x^2)
    else false)) :=
sorry

theorem range_k (k : ℝ) (hk : ∀ x : ℝ, (0 < x ∧ x < 1) → f x > k * (x^3 - 3 * x)) :
k ≥ -2 / 3 :=
sorry

end monotonicity_F_range_k_l1851_185136


namespace triangle_angle_sum_l1851_185178

theorem triangle_angle_sum (A B C : Type) (angle_ABC angle_BAC angle_ACB : ℝ)
  (h₁ : angle_ABC = 110)
  (h₂ : angle_BAC = 45)
  (triangle_sum : angle_ABC + angle_BAC + angle_ACB = 180) :
  angle_ACB = 25 :=
by
  sorry

end triangle_angle_sum_l1851_185178


namespace barbara_wins_gameA_l1851_185189

noncomputable def gameA_winning_strategy : Prop :=
∃ (has_winning_strategy : (ℤ → ℝ) → Prop),
  has_winning_strategy (fun n => n : ℤ → ℝ)

theorem barbara_wins_gameA :
  gameA_winning_strategy := sorry

end barbara_wins_gameA_l1851_185189


namespace daily_avg_for_entire_month_is_correct_l1851_185140

-- conditions
def avg_first_25_days := 63
def days_first_25 := 25
def avg_last_5_days := 33
def days_last_5 := 5
def total_days := days_first_25 + days_last_5

-- question: What is the daily average for the entire month?
theorem daily_avg_for_entire_month_is_correct : 
  (avg_first_25_days * days_first_25 + avg_last_5_days * days_last_5) / total_days = 58 := by
  sorry

end daily_avg_for_entire_month_is_correct_l1851_185140


namespace molecular_weight_cao_is_correct_l1851_185144

-- Define the atomic weights of calcium and oxygen
def atomic_weight_ca : ℝ := 40.08
def atomic_weight_o : ℝ := 16.00

-- Define the molecular weight of CaO
def molecular_weight_cao : ℝ := atomic_weight_ca + atomic_weight_o

-- State the theorem to prove
theorem molecular_weight_cao_is_correct : molecular_weight_cao = 56.08 :=
by
  sorry

end molecular_weight_cao_is_correct_l1851_185144


namespace james_goals_product_l1851_185123

theorem james_goals_product :
  ∃ (g7 g8 : ℕ), g7 < 7 ∧ g8 < 7 ∧ 
  (22 + g7) % 7 = 0 ∧ (22 + g7 + g8) % 8 = 0 ∧ 
  g7 * g8 = 24 :=
by
  sorry

end james_goals_product_l1851_185123


namespace ratio_of_tangent_to_circumference_l1851_185165

theorem ratio_of_tangent_to_circumference
  {r x : ℝ}  -- radius of the circle and length of the tangent
  (hT : x = 2 * π * r)  -- given the length of tangent PQ
  (hA : (1 / 2) * x * r = π * r^2)  -- given the area equivalence

  : (x / (2 * π * r)) = 1 :=  -- desired ratio
by
  -- proof omitted, just using sorry to indicate proof
  sorry

end ratio_of_tangent_to_circumference_l1851_185165


namespace percentage_markup_l1851_185117

/--
The owner of a furniture shop charges his customer a certain percentage more than the cost price.
A customer paid Rs. 3000 for a computer table, and the cost price of the computer table was Rs. 2500.
Prove that the percentage markup on the cost price is 20%.
-/
theorem percentage_markup (selling_price cost_price : ℝ) (h₁ : selling_price = 3000) (h₂ : cost_price = 2500) :
  ((selling_price - cost_price) / cost_price) * 100 = 20 :=
by
  -- proof omitted
  sorry

end percentage_markup_l1851_185117


namespace polynomial_factors_integers_l1851_185162

theorem polynomial_factors_integers (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 500)
  (h₃ : ∃ a : ℤ, n = a * (a + 1)) :
  n ≤ 21 :=
sorry

end polynomial_factors_integers_l1851_185162


namespace balloons_in_package_initially_l1851_185141

-- Definition of conditions
def friends : ℕ := 5
def balloons_given_back : ℕ := 11
def balloons_after_giving_back : ℕ := 39

-- Calculation for original balloons each friend had
def original_balloons_each_friend := balloons_after_giving_back + balloons_given_back

-- Theorem: Number of balloons in the package initially
theorem balloons_in_package_initially : 
  (original_balloons_each_friend * friends) = 250 :=
by
  sorry

end balloons_in_package_initially_l1851_185141


namespace kelly_total_snacks_l1851_185142

theorem kelly_total_snacks (peanuts raisins : ℝ) (h₁ : peanuts = 0.1) (h₂ : raisins = 0.4) :
  peanuts + raisins = 0.5 :=
by
  simp [h₁, h₂]
  sorry

end kelly_total_snacks_l1851_185142


namespace balls_problem_l1851_185187

noncomputable def red_balls_initial := 420
noncomputable def total_balls_initial := 600
noncomputable def percent_red_required := 60 / 100

theorem balls_problem :
  ∃ (x : ℕ), 420 - x = (3 / 5) * (600 - x) :=
by
  sorry

end balls_problem_l1851_185187


namespace price_of_skateboard_l1851_185184

-- Given condition (0.20 * p = 300)
variable (p : ℝ)
axiom upfront_payment : 0.20 * p = 300

-- Theorem statement to prove the price of the skateboard
theorem price_of_skateboard : p = 1500 := by
  sorry

end price_of_skateboard_l1851_185184


namespace unique_a_for_intersection_l1851_185104

def A (a : ℝ) : Set ℝ := {-4, 2 * a - 1, a^2}
def B (a : ℝ) : Set ℝ := {a - 5, 1 - a, 9}

theorem unique_a_for_intersection (a : ℝ) :
  (9 ∈ A a ∩ B a ∧ ∀ x, x ∈ A a ∩ B a → x = 9) ↔ a = -3 := by
  sorry

end unique_a_for_intersection_l1851_185104


namespace extremum_at_one_eq_a_one_l1851_185137

theorem extremum_at_one_eq_a_one 
  (a : ℝ) 
  (h : ∃ f' : ℝ → ℝ, (∀ x, f' x = 3 * a * x^2 - 3) ∧ f' 1 = 0) : 
  a = 1 :=
sorry

end extremum_at_one_eq_a_one_l1851_185137


namespace virus_infection_l1851_185103

theorem virus_infection (x : ℕ) (h : 1 + x + x^2 = 121) : x = 10 := 
sorry

end virus_infection_l1851_185103


namespace local_tax_deduction_in_cents_l1851_185195

def aliciaHourlyWageInDollars : ℝ := 25
def taxDeductionRate : ℝ := 0.02
def aliciaHourlyWageInCents := aliciaHourlyWageInDollars * 100

theorem local_tax_deduction_in_cents :
  taxDeductionRate * aliciaHourlyWageInCents = 50 :=
by 
  -- Proof goes here
  sorry

end local_tax_deduction_in_cents_l1851_185195


namespace overtakes_in_16_minutes_l1851_185151

def number_of_overtakes (track_length : ℕ) (speed_a : ℕ) (speed_b : ℕ) (time_minutes : ℕ) : ℕ :=
  let time_seconds := time_minutes * 60
  let relative_speed := speed_a - speed_b
  let time_per_overtake := track_length / relative_speed
  time_seconds / time_per_overtake

theorem overtakes_in_16_minutes :
  number_of_overtakes 200 6 4 16 = 9 :=
by
  -- We will insert calculations or detailed proof steps if needed
  sorry

end overtakes_in_16_minutes_l1851_185151


namespace smallest_m_exists_l1851_185194

theorem smallest_m_exists : ∃ (m : ℕ), (∀ n : ℕ, (n > 0) → ((10000 * n % 53 = 0) → (m ≤ n))) ∧ (10000 * m % 53 = 0) :=
by
  sorry

end smallest_m_exists_l1851_185194


namespace no_such_n_exists_l1851_185133

theorem no_such_n_exists : ∀ n : ℕ, n > 1 → ∀ (p1 p2 : ℕ), 
  (Nat.Prime p1) → (Nat.Prime p2) → n = p1^2 → n + 60 = p2^2 → False :=
by
  intro n hn p1 p2 hp1 hp2 h1 h2
  sorry

end no_such_n_exists_l1851_185133


namespace part1_l1851_185125

theorem part1 (x : ℝ) (hx : x > 0) : 
  (1 / (2 * Real.sqrt (x + 1))) < (Real.sqrt (x + 1) - Real.sqrt x) ∧ (Real.sqrt (x + 1) - Real.sqrt x) < (1 / (2 * Real.sqrt x)) := 
sorry

end part1_l1851_185125


namespace minimum_shirts_for_savings_l1851_185143

theorem minimum_shirts_for_savings (x : ℕ) : 75 + 8 * x < 16 * x ↔ 10 ≤ x :=
by
  sorry

end minimum_shirts_for_savings_l1851_185143


namespace average_ABC_l1851_185116

/-- Given three numbers A, B, and C such that 1503C - 3006A = 6012 and 1503B + 4509A = 7509,
their average is 3  -/
theorem average_ABC (A B C : ℚ) 
  (h1 : 1503 * C - 3006 * A = 6012) 
  (h2 : 1503 * B + 4509 * A = 7509) : 
  (A + B + C) / 3 = 3 :=
sorry

end average_ABC_l1851_185116


namespace determine_c_l1851_185159

-- Definitions of the sequence
def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 2 * a n + 1

-- Hypothesis for the sequence to be geometric
def geometric_seq (a : ℕ → ℕ) (r : ℕ) : Prop :=
  ∃ c, ∀ n, a (n + 1) + c = r * (a n + c)

-- The goal to prove
theorem determine_c (a : ℕ → ℕ) (c : ℕ) (r := 2) :
  seq a →
  geometric_seq a c →
  c = 1 :=
by
  intros h_seq h_geo
  sorry

end determine_c_l1851_185159


namespace sum_of_interior_edges_l1851_185114

-- Conditions
def width_of_frame_piece : ℝ := 1.5
def one_interior_edge : ℝ := 4.5
def total_frame_area : ℝ := 27

-- Statement of the problem as a theorem in Lean
theorem sum_of_interior_edges : 
  (∃ y : ℝ, (width_of_frame_piece * 2 + one_interior_edge) * (width_of_frame_piece * 2 + y) 
    - one_interior_edge * y = total_frame_area) →
  (4 * (one_interior_edge + y) = 12) :=
sorry

end sum_of_interior_edges_l1851_185114


namespace find_y_l1851_185150

theorem find_y (x y : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 24) : y = 120 :=
by
  sorry

end find_y_l1851_185150


namespace arithmetic_expression_evaluation_l1851_185155

theorem arithmetic_expression_evaluation :
  (-12 * 6) - (-4 * -8) + (-15 * -3) - (36 / (-2)) = -77 :=
by
  sorry

end arithmetic_expression_evaluation_l1851_185155


namespace blocks_needed_for_enclosure_l1851_185172

noncomputable def volume_of_rectangular_prism (length: ℝ) (width: ℝ) (height: ℝ) : ℝ :=
  length * width * height

theorem blocks_needed_for_enclosure 
  (length width height thickness : ℝ)
  (H_length : length = 15)
  (H_width : width = 12)
  (H_height : height = 6)
  (H_thickness : thickness = 1.5) :
  volume_of_rectangular_prism length width height - 
  volume_of_rectangular_prism (length - 2 * thickness) (width - 2 * thickness) (height - thickness) = 594 :=
by
  sorry

end blocks_needed_for_enclosure_l1851_185172


namespace total_people_on_bus_l1851_185153

def initial_people := 4
def added_people := 13

theorem total_people_on_bus : initial_people + added_people = 17 := by
  sorry

end total_people_on_bus_l1851_185153


namespace dandelions_initial_l1851_185160

theorem dandelions_initial (y w : ℕ) (h1 : y + w = 35) (h2 : y - 2 = 2 * (w - 6)) : y = 20 ∧ w = 15 :=
by
  sorry

end dandelions_initial_l1851_185160


namespace percent_savings_correct_l1851_185170

theorem percent_savings_correct :
  let cost_of_package := 9
  let num_of_rolls_in_package := 12
  let cost_per_roll_individually := 1
  let cost_per_roll_in_package := cost_of_package / num_of_rolls_in_package
  let savings_per_roll := cost_per_roll_individually - cost_per_roll_in_package
  let percent_savings := (savings_per_roll / cost_per_roll_individually) * 100
  percent_savings = 25 :=
by
  sorry

end percent_savings_correct_l1851_185170


namespace zero_in_A_l1851_185199

-- Define the set A
def A : Set ℝ := { x | x * (x - 2) = 0 }

-- State the theorem
theorem zero_in_A : 0 ∈ A :=
by {
  -- Skipping the actual proof with "sorry"
  sorry
}

end zero_in_A_l1851_185199


namespace determine_b_l1851_185118

theorem determine_b (b : ℝ) :
  (∀ x y : ℝ, 3 * y - 3 * b = 9 * x) ∧ (∀ x y : ℝ, y - 2 = (b + 9) * x) → 
  b = -6 :=
by
  sorry

end determine_b_l1851_185118


namespace average_marks_l1851_185182

theorem average_marks (A : ℝ) :
  let marks_first_class := 25 * A
  let marks_second_class := 30 * 60
  let total_marks := 55 * 50.90909090909091
  marks_first_class + marks_second_class = total_marks → A = 40 :=
by
  sorry

end average_marks_l1851_185182


namespace man_speed_42_minutes_7_km_l1851_185139

theorem man_speed_42_minutes_7_km 
  (distance : ℝ) (time_minutes : ℝ) (time_hours : ℝ)
  (h1 : distance = 7) 
  (h2 : time_minutes = 42) 
  (h3 : time_hours = time_minutes / 60) :
  distance / time_hours = 10 := by
  sorry

end man_speed_42_minutes_7_km_l1851_185139
