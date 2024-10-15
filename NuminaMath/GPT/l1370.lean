import Mathlib

namespace NUMINAMATH_GPT_ratio_of_ages_l1370_137097

theorem ratio_of_ages
  (Sandy_age : ℕ)
  (Molly_age : ℕ)
  (h1 : Sandy_age = 49)
  (h2 : Molly_age = Sandy_age + 14) : (Sandy_age : ℚ) / Molly_age = 7 / 9 :=
by
  -- To complete the proof.
  sorry

end NUMINAMATH_GPT_ratio_of_ages_l1370_137097


namespace NUMINAMATH_GPT_inequality_proof_l1370_137047

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (2 * x^2 / (y + z) + 2 * y^2 / (z + x) + 2 * z^2 / (x + y) ≥ x + y + z) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1370_137047


namespace NUMINAMATH_GPT_number_of_acceptable_ages_l1370_137037

theorem number_of_acceptable_ages (avg_age : ℤ) (std_dev : ℤ) (a b : ℤ) (h_avg : avg_age = 10) (h_std : std_dev = 8)
    (h1 : a = avg_age - std_dev) (h2 : b = avg_age + std_dev) :
    b - a + 1 = 17 :=
by {
    sorry
}

end NUMINAMATH_GPT_number_of_acceptable_ages_l1370_137037


namespace NUMINAMATH_GPT_gamma_minus_alpha_l1370_137090

theorem gamma_minus_alpha (α β γ : ℝ) (h1 : 0 < α) (h2 : α < β) (h3 : β < γ) (h4 : γ < 2 * Real.pi)
    (h5 : ∀ x : ℝ, Real.cos (x + α) + Real.cos (x + β) + Real.cos (x + γ) = 0) : 
    γ - α = (4 * Real.pi) / 3 :=
sorry

end NUMINAMATH_GPT_gamma_minus_alpha_l1370_137090


namespace NUMINAMATH_GPT_ninggao_intercity_project_cost_in_scientific_notation_l1370_137038

theorem ninggao_intercity_project_cost_in_scientific_notation :
  let length_kilometers := 55
  let cost_per_kilometer_million := 140
  let total_cost_million := length_kilometers * cost_per_kilometer_million
  let total_cost_scientific := 7.7 * 10^6
  total_cost_million = total_cost_scientific := 
  sorry

end NUMINAMATH_GPT_ninggao_intercity_project_cost_in_scientific_notation_l1370_137038


namespace NUMINAMATH_GPT_equilateral_triangle_perimeter_isosceles_triangle_leg_length_l1370_137089

-- Definitions for equilateral triangle problem
def side_length_equilateral : ℕ := 12
def perimeter_equilateral := side_length_equilateral * 3

-- Definitions for isosceles triangle problem
def perimeter_isosceles : ℕ := 72
def base_length_isosceles : ℕ := 28
def leg_length_isosceles := (perimeter_isosceles - base_length_isosceles) / 2

-- Theorem statement
theorem equilateral_triangle_perimeter : perimeter_equilateral = 36 := 
by
  sorry

theorem isosceles_triangle_leg_length : leg_length_isosceles = 22 := 
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_perimeter_isosceles_triangle_leg_length_l1370_137089


namespace NUMINAMATH_GPT_animal_population_l1370_137039

theorem animal_population
  (number_of_lions : ℕ)
  (number_of_leopards : ℕ)
  (number_of_elephants : ℕ)
  (h1 : number_of_lions = 200)
  (h2 : number_of_lions = 2 * number_of_leopards)
  (h3 : number_of_elephants = (number_of_lions + number_of_leopards) / 2) :
  number_of_lions + number_of_leopards + number_of_elephants = 450 :=
sorry

end NUMINAMATH_GPT_animal_population_l1370_137039


namespace NUMINAMATH_GPT_trigonometric_simplification_l1370_137010

noncomputable def tan : ℝ → ℝ := λ x => Real.sin x / Real.cos x
noncomputable def simp_expr : ℝ :=
  (tan (96 * Real.pi / 180) - tan (12 * Real.pi / 180) * (1 + 1 / Real.sin (6 * Real.pi / 180)))
  /
  (1 + tan (96 * Real.pi / 180) * tan (12 * Real.pi / 180) * (1 + 1 / Real.sin (6 * Real.pi / 180)))

theorem trigonometric_simplification : simp_expr = Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_simplification_l1370_137010


namespace NUMINAMATH_GPT_abs_quadratic_eq_linear_iff_l1370_137012

theorem abs_quadratic_eq_linear_iff (x : ℝ) : 
  (|x^2 - 5*x + 6| = x + 2) ↔ (x = 3 + Real.sqrt 5 ∨ x = 3 - Real.sqrt 5) :=
by
  sorry

end NUMINAMATH_GPT_abs_quadratic_eq_linear_iff_l1370_137012


namespace NUMINAMATH_GPT_largest_sum_of_two_3_digit_numbers_l1370_137042

theorem largest_sum_of_two_3_digit_numbers : 
  ∃ (a b c d e f : ℕ), 
    (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧ (1 ≤ c ∧ c ≤ 6) ∧
    (1 ≤ d ∧ d ≤ 6) ∧ (1 ≤ e ∧ e ≤ 6) ∧ (1 ≤ f ∧ f ≤ 6) ∧
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
     c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
     d ≠ e ∧ d ≠ f ∧ 
     e ≠ f) ∧ 
    (100 * (a + d) + 10 * (b + e) + (c + f) = 1173) :=
by
  sorry

end NUMINAMATH_GPT_largest_sum_of_two_3_digit_numbers_l1370_137042


namespace NUMINAMATH_GPT_min_squared_sum_l1370_137005

theorem min_squared_sum {x y z : ℝ} (h : 2 * x + y + 2 * z = 6) : x^2 + y^2 + z^2 ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_squared_sum_l1370_137005


namespace NUMINAMATH_GPT_tan_theta_eq_two_implies_expression_l1370_137082

theorem tan_theta_eq_two_implies_expression (θ : ℝ) (h : Real.tan θ = 2) :
    (1 - Real.sin (2 * θ)) / (2 * (Real.cos θ)^2) = 1 / 2 :=
by
  -- Define trig identities and given condition
  have h_sin_cos : Real.sin θ = 2 / Real.sqrt 5 ∧ Real.cos θ = 1 / Real.sqrt 5 :=
    sorry -- This will be derived from the given condition h
  
  -- Main proof
  sorry

end NUMINAMATH_GPT_tan_theta_eq_two_implies_expression_l1370_137082


namespace NUMINAMATH_GPT_correct_system_of_equations_l1370_137032

variable (x y : ℕ) -- We assume non-negative numbers for counts of chickens and rabbits

theorem correct_system_of_equations :
  (x + y = 35) ∧ (2 * x + 4 * y = 94) ↔
  (∃ (a b : ℕ), a = x ∧ b = y) :=
by
  sorry

end NUMINAMATH_GPT_correct_system_of_equations_l1370_137032


namespace NUMINAMATH_GPT_perfect_square_trinomial_l1370_137058

theorem perfect_square_trinomial (k : ℤ) : (∃ a : ℤ, (x : ℤ) → x^2 - k * x + 9 = (x - a)^2) → (k = 6 ∨ k = -6) :=
sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l1370_137058


namespace NUMINAMATH_GPT_find_y_z_l1370_137053

theorem find_y_z 
  (y z : ℝ) 
  (h_mean : (8 + 15 + 22 + 5 + y + z) / 6 = 12) 
  (h_diff : y - z = 6) : 
  y = 14 ∧ z = 8 := 
by
  sorry

end NUMINAMATH_GPT_find_y_z_l1370_137053


namespace NUMINAMATH_GPT_find_n_l1370_137083

theorem find_n (n : ℕ) : 5 ^ 29 * 4 ^ 15 = 2 * 10 ^ n → n = 29 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1370_137083


namespace NUMINAMATH_GPT_total_students_after_new_classes_l1370_137026

def initial_classes : ℕ := 15
def students_per_class : ℕ := 20
def new_classes : ℕ := 5

theorem total_students_after_new_classes :
  initial_classes * students_per_class + new_classes * students_per_class = 400 :=
by
  sorry

end NUMINAMATH_GPT_total_students_after_new_classes_l1370_137026


namespace NUMINAMATH_GPT_symmetric_function_cannot_be_even_l1370_137020

noncomputable def f : ℝ → ℝ := sorry

theorem symmetric_function_cannot_be_even :
  (∀ x, f (f x) = x^2) ∧ (∀ x ≥ 0, f (x^2) = x) → ¬ (∀ x, f x = f (-x)) :=
by 
  intros
  sorry -- Proof is not required

end NUMINAMATH_GPT_symmetric_function_cannot_be_even_l1370_137020


namespace NUMINAMATH_GPT_marks_in_social_studies_l1370_137096

def shekar_marks : ℕ := 82

theorem marks_in_social_studies 
  (marks_math : ℕ := 76)
  (marks_science : ℕ := 65)
  (marks_english : ℕ := 67)
  (marks_biology : ℕ := 55)
  (average_marks : ℕ := 69)
  (num_subjects : ℕ := 5) :
  marks_math + marks_science + marks_english + marks_biology + shekar_marks = average_marks * num_subjects :=
by
  sorry

end NUMINAMATH_GPT_marks_in_social_studies_l1370_137096


namespace NUMINAMATH_GPT_perpendicular_lines_m_value_l1370_137024

theorem perpendicular_lines_m_value
  (l1 : ∀ (x y : ℝ), x - 2 * y + 1 = 0)
  (l2 : ∀ (x y : ℝ), m * x + y - 3 = 0)
  (perpendicular : ∀ (m : ℝ) (l1_slope l2_slope : ℝ), l1_slope * l2_slope = -1) : 
  m = 2 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_m_value_l1370_137024


namespace NUMINAMATH_GPT_function_equivalence_l1370_137098

theorem function_equivalence (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2 * x) = 6 * x - 1) : ∀ x : ℝ, f x = 3 * x - 1 :=
by
  sorry

end NUMINAMATH_GPT_function_equivalence_l1370_137098


namespace NUMINAMATH_GPT_cos_pi_over_6_minus_a_eq_5_over_12_l1370_137031

theorem cos_pi_over_6_minus_a_eq_5_over_12 (a : ℝ) (h : Real.sin (Real.pi / 3 + a) = 5 / 12) :
  Real.cos (Real.pi / 6 - a) = 5 / 12 :=
by
  sorry

end NUMINAMATH_GPT_cos_pi_over_6_minus_a_eq_5_over_12_l1370_137031


namespace NUMINAMATH_GPT_plane_equation_parametric_l1370_137091

theorem plane_equation_parametric 
  (s t : ℝ)
  (v : ℝ × ℝ × ℝ)
  (x y z : ℝ) 
  (A B C D : ℤ)
  (h1 : v = (2 + s + 2 * t, 3 + 2 * s - t, 1 + s + 3 * t))
  (h2 : A = 7)
  (h3 : B = -1)
  (h4 : C = -5)
  (h5 : D = -6)
  (h6 : A > 0)
  (h7 : Int.gcd A (Int.gcd B (Int.gcd C D)) = 1) :
  7 * x - y - 5 * z - 6 = 0 := 
sorry

end NUMINAMATH_GPT_plane_equation_parametric_l1370_137091


namespace NUMINAMATH_GPT_initial_men_count_l1370_137070

theorem initial_men_count (M : ℕ) :
  let total_food := M * 22
  let food_after_2_days := total_food - 2 * M
  let remaining_food := 20 * M
  let new_total_men := M + 190
  let required_food_for_16_days := new_total_men * 16
  (remaining_food = required_food_for_16_days) → M = 760 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_initial_men_count_l1370_137070


namespace NUMINAMATH_GPT_tan_angle_sum_identity_l1370_137061

theorem tan_angle_sum_identity
  (θ : ℝ)
  (h1 : θ > π / 2 ∧ θ < π)
  (h2 : Real.cos θ = -3 / 5) :
  Real.tan (θ + π / 4) = -1 / 7 := by
  sorry

end NUMINAMATH_GPT_tan_angle_sum_identity_l1370_137061


namespace NUMINAMATH_GPT_expression_value_l1370_137011

theorem expression_value {a b : ℝ} (h : a * b = -3) : a * Real.sqrt (-b / a) + b * Real.sqrt (-a / b) = 0 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l1370_137011


namespace NUMINAMATH_GPT_sufficient_condition_not_monotonic_l1370_137066

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - 4 * a * x - Real.log x

def sufficient_not_monotonic (a : ℝ) : Prop :=
  (a > 1 / 6) ∨ (a < -1 / 2)

theorem sufficient_condition_not_monotonic (a : ℝ) :
  sufficient_not_monotonic a → ¬(∀ x y : ℝ, 1 < x ∧ x < 3 ∧ 1 < y ∧ y < 3 ∧ x ≠ y → ((f a x - f a y) / (x - y) ≥ 0 ∨ (f a y - f a x) / (y - x) ≥ 0)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_not_monotonic_l1370_137066


namespace NUMINAMATH_GPT_area_of_30_60_90_triangle_hypotenuse_6sqrt2_l1370_137016

theorem area_of_30_60_90_triangle_hypotenuse_6sqrt2 :
  ∀ (a b c : ℝ),
  a = 3 * Real.sqrt 2 →
  b = 3 * Real.sqrt 6 →
  c = 6 * Real.sqrt 2 →
  c = 2 * a →
  (1 / 2) * a * b = 18 * Real.sqrt 3 :=
by
  intro a b c ha hb hc h2a
  sorry

end NUMINAMATH_GPT_area_of_30_60_90_triangle_hypotenuse_6sqrt2_l1370_137016


namespace NUMINAMATH_GPT_expand_polynomial_l1370_137019

theorem expand_polynomial (t : ℝ) :
  (3 * t^3 - 2 * t^2 + t - 4) * (2 * t^2 - t + 3) = 6 * t^5 - 7 * t^4 + 5 * t^3 - 15 * t^2 + 7 * t - 12 :=
by sorry

end NUMINAMATH_GPT_expand_polynomial_l1370_137019


namespace NUMINAMATH_GPT_solution_set_f_leq_g_range_of_a_l1370_137052

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + abs (2 * x + 1)
noncomputable def g (x : ℝ) : ℝ := x + 2

theorem solution_set_f_leq_g (x : ℝ) : f x 1 ≤ g x ↔ (0 ≤ x ∧ x ≤ 2 / 3) := by
  sorry

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, f x a ≥ g x) : 2 ≤ a := by
  sorry

end NUMINAMATH_GPT_solution_set_f_leq_g_range_of_a_l1370_137052


namespace NUMINAMATH_GPT_color_changes_probability_l1370_137014

-- Define the durations of the traffic lights
def green_duration := 40
def yellow_duration := 5
def red_duration := 45

-- Define the total cycle duration
def total_cycle_duration := green_duration + yellow_duration + red_duration

-- Define the duration of the interval Mary watches
def watch_duration := 4

-- Define the change windows where the color changes can be witnessed
def change_windows :=
  [green_duration - watch_duration,
   green_duration + yellow_duration - watch_duration,
   green_duration + yellow_duration + red_duration - watch_duration]

-- Define the total change window duration
def total_change_window_duration := watch_duration * (change_windows.length)

-- Calculate the probability of witnessing a change
def probability_witnessing_change := (total_change_window_duration : ℚ) / total_cycle_duration

-- The theorem to prove
theorem color_changes_probability :
  probability_witnessing_change = 2 / 15 := by sorry

end NUMINAMATH_GPT_color_changes_probability_l1370_137014


namespace NUMINAMATH_GPT_minutes_before_noon_l1370_137081

theorem minutes_before_noon (x : ℕ) (h1 : x = 40)
  (h2 : ∀ (t : ℕ), t = 180 - (x + 40) ∧ t = 3 * x) : x = 35 :=
by {
  sorry
}

end NUMINAMATH_GPT_minutes_before_noon_l1370_137081


namespace NUMINAMATH_GPT_f_sum_zero_l1370_137099

noncomputable def f : ℝ → ℝ := sorry

axiom f_property_1 : ∀ x : ℝ, f (x ^ 3) = (f x) ^ 3
axiom f_property_2 : ∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 ≠ f x2

theorem f_sum_zero : f 0 + f (-1) + f 1 = 0 := by
  sorry

end NUMINAMATH_GPT_f_sum_zero_l1370_137099


namespace NUMINAMATH_GPT_simplify_expression_l1370_137049

variable (a : ℝ)

theorem simplify_expression : 
  (a^2 / (a^(1/2) * a^(2/3))) = a^(5/6) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1370_137049


namespace NUMINAMATH_GPT_largest_n_satisfying_equation_l1370_137043

theorem largest_n_satisfying_equation :
  ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ ∀ n : ℕ,
  (n * n = x * x + y * y + z * z + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 12) →
  n ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_largest_n_satisfying_equation_l1370_137043


namespace NUMINAMATH_GPT_shea_buys_corn_l1370_137007

noncomputable def num_pounds_corn (c b : ℚ) : ℚ :=
  if b + c = 24 ∧ 45 * b + 99 * c = 1809 then c else -1

theorem shea_buys_corn (c b : ℚ) : b + c = 24 ∧ 45 * b + 99 * c = 1809 → c = 13.5 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_shea_buys_corn_l1370_137007


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1370_137095

variable {I : Set ℝ} (f : ℝ → ℝ) (M : ℝ)

theorem necessary_but_not_sufficient :
  (∀ x ∈ I, f x ≤ M) ↔
  (∀ x ∈ I, f x ≤ M ∧ (∃ x ∈ I, f x = M) → M = M ∧ ∃ x ∈ I, f x = M) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1370_137095


namespace NUMINAMATH_GPT_lcm_12_15_18_l1370_137046

theorem lcm_12_15_18 : Nat.lcm (Nat.lcm 12 15) 18 = 180 := by 
  sorry

end NUMINAMATH_GPT_lcm_12_15_18_l1370_137046


namespace NUMINAMATH_GPT_measure_of_acute_angle_l1370_137018

theorem measure_of_acute_angle (x : ℝ) (h_complement : 90 - x = (1/2) * (180 - x) + 20) (h_acute : 0 < x ∧ x < 90) : x = 40 :=
  sorry

end NUMINAMATH_GPT_measure_of_acute_angle_l1370_137018


namespace NUMINAMATH_GPT_train_speed_in_km_per_hr_l1370_137045

def train_length : ℝ := 116.67 -- length of the train in meters
def crossing_time : ℝ := 7 -- time to cross the pole in seconds

theorem train_speed_in_km_per_hr : 
  (train_length / 1000) / (crossing_time / 3600) = 60 := by
  sorry

end NUMINAMATH_GPT_train_speed_in_km_per_hr_l1370_137045


namespace NUMINAMATH_GPT_ratio_of_areas_two_adjacent_triangles_to_one_triangle_l1370_137084

-- Definition of a regular hexagon divided into six equal triangles
def is_regular_hexagon_divided_into_six_equal_triangles (s : ℝ) : Prop :=
  s > 0 -- s is the area of one of the six triangles and must be positive

-- Definition of the area of a region formed by two adjacent triangles
def area_of_two_adjacent_triangles (s r : ℝ) : Prop :=
  r = 2 * s

-- The proof problem statement
theorem ratio_of_areas_two_adjacent_triangles_to_one_triangle (s r : ℝ)
  (hs : is_regular_hexagon_divided_into_six_equal_triangles s)
  (hr : area_of_two_adjacent_triangles s r) : 
  r / s = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_two_adjacent_triangles_to_one_triangle_l1370_137084


namespace NUMINAMATH_GPT_monotonicity_f_max_value_f_l1370_137088

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x - 1

theorem monotonicity_f :
  (∀ x, 0 < x ∧ x < Real.exp 1 → f x < f (Real.exp 1)) ∧
  (∀ x, x > Real.exp 1 → f x < f (Real.exp 1)) :=
sorry

theorem max_value_f (m : ℝ) (hm : m > 0) :
  (2 * m ≤ Real.exp 1 → ∃ x ∈ Set.Icc m (2 * m), f x = (Real.log (2 * m)) / (2 * m) - 1) ∧
  (m ≥ Real.exp 1 → ∃ x ∈ Set.Icc m (2 * m), f x = (Real.log m) / m - 1) ∧
  (Real.exp 1 / 2 < m ∧ m < Real.exp 1 → ∃ x ∈ Set.Icc m (2 * m), f x = 1 / Real.exp 1 - 1) :=
sorry

end NUMINAMATH_GPT_monotonicity_f_max_value_f_l1370_137088


namespace NUMINAMATH_GPT_a_eq_zero_l1370_137055

theorem a_eq_zero (a b : ℤ) (h : ∀ n : ℕ, ∃ x : ℤ, x^2 = 2^n * a + b) : a = 0 :=
sorry

end NUMINAMATH_GPT_a_eq_zero_l1370_137055


namespace NUMINAMATH_GPT_f_increasing_on_positive_l1370_137072

noncomputable def f (x : ℝ) : ℝ := - (1 / x) - 1

theorem f_increasing_on_positive (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 > x2) : f x1 > f x2 := by
  sorry

end NUMINAMATH_GPT_f_increasing_on_positive_l1370_137072


namespace NUMINAMATH_GPT_logarithmic_ratio_l1370_137073

theorem logarithmic_ratio (m n : ℝ) (h1 : Real.log 2 = m) (h2 : Real.log 3 = n) :
  (Real.log 12) / (Real.log 15) = (2 * m + n) / (1 - m + n) := 
sorry

end NUMINAMATH_GPT_logarithmic_ratio_l1370_137073


namespace NUMINAMATH_GPT_triangle_ratio_l1370_137063

theorem triangle_ratio (a b c : ℕ) (r s : ℕ) (h1 : a = 9) (h2 : b = 15) (h3 : c = 18) (h4 : r + s = a) (h5 : r < s) : r * 2 = s :=
by
  sorry

end NUMINAMATH_GPT_triangle_ratio_l1370_137063


namespace NUMINAMATH_GPT_expressions_equal_when_a_plus_b_plus_c_eq_1_l1370_137028

theorem expressions_equal_when_a_plus_b_plus_c_eq_1
  (a b c : ℝ) (h : a + b + c = 1) :
  a + b * c = (a + b) * (a + c) :=
sorry

end NUMINAMATH_GPT_expressions_equal_when_a_plus_b_plus_c_eq_1_l1370_137028


namespace NUMINAMATH_GPT_best_discount_sequence_l1370_137022

/-- 
The initial price of the book is 30.
Stay focused on two sequences of discounts.
Sequence 1: $5 off, then 10% off, then $2 off if applicable.
Sequence 2: 10% off, then $5 off, then $2 off if applicable.
Compare the final prices obtained from applying these sequences.
-/
noncomputable def initial_price : ℝ := 30
noncomputable def five_off (price : ℝ) : ℝ := price - 5
noncomputable def ten_percent_off (price : ℝ) : ℝ := 0.9 * price
noncomputable def additional_two_off_if_applicable (price : ℝ) : ℝ := 
  if price > 20 then price - 2 else price

noncomputable def sequence1_final_price : ℝ := 
  additional_two_off_if_applicable (ten_percent_off (five_off initial_price))

noncomputable def sequence2_final_price : ℝ := 
  additional_two_off_if_applicable (five_off (ten_percent_off initial_price))

theorem best_discount_sequence : 
  sequence2_final_price = 20 ∧ 
  sequence2_final_price < sequence1_final_price ∧ 
  sequence1_final_price - sequence2_final_price = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_best_discount_sequence_l1370_137022


namespace NUMINAMATH_GPT_f_neg_two_l1370_137021

def f (a b : ℝ) (x : ℝ) :=
  -a * x^5 - x^3 + b * x - 7

theorem f_neg_two (a b : ℝ) (h : f a b 2 = -9) : f a b (-2) = -5 :=
by sorry

end NUMINAMATH_GPT_f_neg_two_l1370_137021


namespace NUMINAMATH_GPT_not_difference_of_squares_2021_l1370_137017

theorem not_difference_of_squares_2021:
  ¬ ∃ (a b : ℕ), (a > b) ∧ (a^2 - b^2 = 2021) :=
sorry

end NUMINAMATH_GPT_not_difference_of_squares_2021_l1370_137017


namespace NUMINAMATH_GPT_weekly_salary_correct_l1370_137075

-- Define the daily salaries for each type of worker
def salary_A : ℝ := 200
def salary_B : ℝ := 250
def salary_C : ℝ := 300
def salary_D : ℝ := 350

-- Define the number of each type of worker
def num_A : ℕ := 3
def num_B : ℕ := 2
def num_C : ℕ := 3
def num_D : ℕ := 1

-- Define the total hours worked per day and the number of working days in a week
def hours_per_day : ℕ := 6
def working_days : ℕ := 7

-- Calculate the total daily salary for the team
def daily_salary_team : ℝ :=
  (num_A * salary_A) + (num_B * salary_B) + (num_C * salary_C) + (num_D * salary_D)

-- Calculate the total weekly salary for the team
def weekly_salary_team : ℝ := daily_salary_team * working_days

-- Problem: Prove that the total weekly salary for the team is Rs. 16,450
theorem weekly_salary_correct : weekly_salary_team = 16450 := by
  sorry

end NUMINAMATH_GPT_weekly_salary_correct_l1370_137075


namespace NUMINAMATH_GPT_cases_in_1995_l1370_137033

theorem cases_in_1995 (initial_cases cases_2010 : ℕ) (years_total : ℕ) (years_passed : ℕ) (cases_1995 : ℕ)
  (h1 : initial_cases = 700000) 
  (h2 : cases_2010 = 1000) 
  (h3 : years_total = 40) 
  (h4 : years_passed = 25)
  (h5 : cases_1995 = initial_cases - (years_passed * (initial_cases - cases_2010) / years_total)) : 
  cases_1995 = 263125 := 
sorry

end NUMINAMATH_GPT_cases_in_1995_l1370_137033


namespace NUMINAMATH_GPT_arrange_2015_integers_l1370_137050

theorem arrange_2015_integers :
  ∃ (f : Fin 2015 → Fin 2015),
    (∀ i, (Nat.gcd ((f i).val + (f (i + 1)).val) 4 = 1 ∨ Nat.gcd ((f i).val + (f (i + 1)).val) 7 = 1)) ∧
    Function.Injective f ∧ 
    (∀ i, 1 ≤ (f i).val ∧ (f i).val ≤ 2015) :=
sorry

end NUMINAMATH_GPT_arrange_2015_integers_l1370_137050


namespace NUMINAMATH_GPT_StepaMultiplication_l1370_137071

theorem StepaMultiplication {a : ℕ} (h1 : Grisha's_answer = (3 / 2) ^ 4 * a)
  (h2 : Grisha's_answer = 81) :
  (∃ (m n : ℕ), m * n = (3 / 2) ^ 3 * a ∧ m < 10 ∧ n < 10) :=
by
  sorry

end NUMINAMATH_GPT_StepaMultiplication_l1370_137071


namespace NUMINAMATH_GPT_anthony_transactions_more_percentage_l1370_137076

def transactions (Mabel Anthony Cal Jade : ℕ) : Prop := 
  Mabel = 90 ∧ 
  Jade = 84 ∧ 
  Jade = Cal + 18 ∧ 
  Cal = (2 * Anthony) / 3 ∧ 
  Anthony = Mabel + (Mabel * 10 / 100)

theorem anthony_transactions_more_percentage (Mabel Anthony Cal Jade : ℕ) 
    (h : transactions Mabel Anthony Cal Jade) : 
  (Anthony = Mabel + (Mabel * 10 / 100)) :=
by 
  sorry

end NUMINAMATH_GPT_anthony_transactions_more_percentage_l1370_137076


namespace NUMINAMATH_GPT_xyz_value_l1370_137003

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) : 
  x * y * z = 26 / 3 := 
by
  sorry

end NUMINAMATH_GPT_xyz_value_l1370_137003


namespace NUMINAMATH_GPT_distance_from_P_to_AD_is_correct_l1370_137079

noncomputable def P_distance_to_AD : ℝ :=
  let A : ℝ × ℝ := (0, 6)
  let D : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (6, 0)
  let M : ℝ × ℝ := (3, 0)
  let radius1 : ℝ := 5
  let radius2 : ℝ := 6
  let circle1_eq := fun (x y : ℝ) => (x - 3)^2 + y^2 = 25
  let circle2_eq := fun (x y : ℝ) => x^2 + (y - 6)^2 = 36
  let P := (24/5, 18/5)
  let AD := fun x y : ℝ => x = 0
  abs ((P.fst : ℝ) - 0)

theorem distance_from_P_to_AD_is_correct :
  P_distance_to_AD = 24 / 5 := by
  sorry

end NUMINAMATH_GPT_distance_from_P_to_AD_is_correct_l1370_137079


namespace NUMINAMATH_GPT_initial_wage_of_illiterate_l1370_137001

-- Definitions from the conditions
def illiterate_employees : ℕ := 20
def literate_employees : ℕ := 10
def total_employees := illiterate_employees + literate_employees

-- Given that the daily average wages of illiterate employees decreased to Rs. 10
def daily_wages_after_decrease : ℝ := 10
-- The total decrease in the average salary of all employees by Rs. 10 per day
def decrease_in_avg_wage : ℝ := 10

-- To be proved: the initial daily average wage of the illiterate employees was Rs. 25.
theorem initial_wage_of_illiterate (I : ℝ) :
  (illiterate_employees * I - illiterate_employees * daily_wages_after_decrease = total_employees * decrease_in_avg_wage) → 
  I = 25 := 
by
  sorry

end NUMINAMATH_GPT_initial_wage_of_illiterate_l1370_137001


namespace NUMINAMATH_GPT_arithmetic_sequence_diff_l1370_137044

theorem arithmetic_sequence_diff (a : ℕ → ℝ)
  (h1 : a 5 * a 7 = 6)
  (h2 : a 2 + a 10 = 5) :
  a 10 - a 6 = 2 ∨ a 10 - a 6 = -2 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_diff_l1370_137044


namespace NUMINAMATH_GPT_parabola_focus_distance_l1370_137092

open Real

noncomputable def parabola (P : ℝ × ℝ) : Prop := (P.2)^2 = 4 * P.1
def line_eq (P : ℝ × ℝ) : Prop := abs (P.1 + 2) = 6

theorem parabola_focus_distance (P : ℝ × ℝ) 
  (hp : parabola P) 
  (hl : line_eq P) : 
  dist P (1 / 4, 0) = 5 :=
sorry

end NUMINAMATH_GPT_parabola_focus_distance_l1370_137092


namespace NUMINAMATH_GPT_sequence_behavior_l1370_137002

theorem sequence_behavior (b : ℕ → ℕ) :
  (∀ n, b n = n) ∨ ∃ N, ∀ n, n ≥ N → b n = b N :=
sorry

end NUMINAMATH_GPT_sequence_behavior_l1370_137002


namespace NUMINAMATH_GPT_inequality_proof_l1370_137065

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (1 / (x^2 + y^2)) + (1 / x^2) + (1 / y^2) ≥ 10 / (x + y)^2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1370_137065


namespace NUMINAMATH_GPT_any_nat_in_frac_l1370_137056

theorem any_nat_in_frac (n : ℕ) : ∃ x y : ℕ, y ≠ 0 ∧ x^2 = y^3 * n := by
  sorry

end NUMINAMATH_GPT_any_nat_in_frac_l1370_137056


namespace NUMINAMATH_GPT_roots_reciprocal_l1370_137025

theorem roots_reciprocal (a b c x1 x2 x3 x4 : ℝ) 
  (h1 : a ≠ 0)
  (h2 : c ≠ 0)
  (hx1 : a * x1^2 + b * x1 + c = 0)
  (hx2 : a * x2^2 + b * x2 + c = 0)
  (hx3 : c * x3^2 + b * x3 + a = 0)
  (hx4 : c * x4^2 + b * x4 + a = 0) :
  (x3 = 1/x1 ∧ x4 = 1/x2) :=
  sorry

end NUMINAMATH_GPT_roots_reciprocal_l1370_137025


namespace NUMINAMATH_GPT_charles_richard_difference_in_dimes_l1370_137009

variable (q : ℕ)

-- Charles' quarters
def charles_quarters : ℕ := 5 * q + 1

-- Richard's quarters
def richard_quarters : ℕ := q + 5

-- Difference in quarters
def diff_quarters : ℕ := charles_quarters q - richard_quarters q

-- Difference in dimes
def diff_dimes : ℕ := (diff_quarters q) * 5 / 2

theorem charles_richard_difference_in_dimes : diff_dimes q = 10 * (q - 1) := by
  sorry

end NUMINAMATH_GPT_charles_richard_difference_in_dimes_l1370_137009


namespace NUMINAMATH_GPT_triangle_area_l1370_137004

-- Define the line equation as a condition.
def line_equation (x : ℝ) : ℝ :=
  4 * x + 8

-- Define the y-intercept (condition 1).
def y_intercept := line_equation 0

-- Define the x-intercept (condition 2).
def x_intercept := (-8) / 4

-- Define the area of the triangle given the intercepts and prove it equals 8 (question and correct answer).
theorem triangle_area :
  (1 / 2) * abs x_intercept * y_intercept = 8 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l1370_137004


namespace NUMINAMATH_GPT_num_lists_correct_l1370_137040

def num_balls : ℕ := 18
def num_draws : ℕ := 4

theorem num_lists_correct : (num_balls ^ num_draws) = 104976 :=
by
  sorry

end NUMINAMATH_GPT_num_lists_correct_l1370_137040


namespace NUMINAMATH_GPT_ac_length_l1370_137015

theorem ac_length (AB : ℝ) (H1 : AB = 100)
    (BC AC : ℝ)
    (H2 : AC = (1 + Real.sqrt 5)/2 * BC)
    (H3 : AC + BC = AB) : AC = 75 - 25 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_ac_length_l1370_137015


namespace NUMINAMATH_GPT_joanne_first_hour_coins_l1370_137057

theorem joanne_first_hour_coins 
  (X : ℕ)
  (H1 : 70 = 35 + 35)
  (H2 : 120 = X + 70 + 35)
  (H3 : 35 = 50 - 15) : 
  X = 15 :=
sorry

end NUMINAMATH_GPT_joanne_first_hour_coins_l1370_137057


namespace NUMINAMATH_GPT_find_multiplier_l1370_137041

theorem find_multiplier (n x : ℤ) (h1: n = 12) (h2: 4 * n - 3 = (n - 7) * x) : x = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_multiplier_l1370_137041


namespace NUMINAMATH_GPT_speed_of_man_l1370_137027

noncomputable def train_length : ℝ := 150
noncomputable def time_to_pass : ℝ := 6
noncomputable def train_speed_kmh : ℝ := 83.99280057595394

/-- The speed of the man in km/h -/
theorem speed_of_man (train_length time_to_pass train_speed_kmh : ℝ) (h_train_length : train_length = 150) (h_time_to_pass : time_to_pass = 6) (h_train_speed_kmh : train_speed_kmh = 83.99280057595394) : 
  (train_length / time_to_pass * 3600 / 1000 - train_speed_kmh) * 3600 / 1000 = 6.0072 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_man_l1370_137027


namespace NUMINAMATH_GPT_percentage_within_one_standard_deviation_l1370_137013

-- Define the constants
def m : ℝ := sorry     -- mean
def g : ℝ := sorry     -- standard deviation
def P : ℝ → ℝ := sorry -- cumulative distribution function

-- The condition that 84% of the distribution is less than m + g
def condition1 : Prop := P (m + g) = 0.84

-- The condition that the distribution is symmetric about the mean
def symmetric_distribution (P : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, P (m + (m - x)) = 1 - P x

-- The problem asks to prove that 68% of the distribution lies within one standard deviation of the mean
theorem percentage_within_one_standard_deviation 
  (h₁ : condition1)
  (h₂ : symmetric_distribution P m) : 
  P (m + g) - P (m - g) = 0.68 :=
sorry

end NUMINAMATH_GPT_percentage_within_one_standard_deviation_l1370_137013


namespace NUMINAMATH_GPT_avg_goals_per_game_l1370_137006

def carter_goals_per_game := 4
def shelby_goals_per_game := carter_goals_per_game / 2
def judah_goals_per_game := (2 * shelby_goals_per_game) - 3
def average_total_goals_team := carter_goals_per_game + shelby_goals_per_game + judah_goals_per_game

theorem avg_goals_per_game : average_total_goals_team = 7 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_avg_goals_per_game_l1370_137006


namespace NUMINAMATH_GPT_sum_floor_ceil_eq_seven_l1370_137093

theorem sum_floor_ceil_eq_seven (x : ℝ) 
  (h : ⌊x⌋ + ⌈x⌉ = 7) : 3 < x ∧ x < 4 := 
sorry

end NUMINAMATH_GPT_sum_floor_ceil_eq_seven_l1370_137093


namespace NUMINAMATH_GPT_vasya_made_mistake_l1370_137068

theorem vasya_made_mistake : 
  ∀ (total_digits : ℕ), 
    total_digits = 301 → 
    ¬∃ (n : ℕ), 
      (n ≤ 9 ∧ total_digits = (n * 1)) ∨ 
      (10 ≤ n ∧ n ≤ 99 ∧ total_digits = (9 * 1) + ((n - 9) * 2)) ∨ 
      (100 ≤ n ∧ total_digits = (9 * 1) + (90 * 2) + ((n - 99) * 3)) := 
by 
  sorry

end NUMINAMATH_GPT_vasya_made_mistake_l1370_137068


namespace NUMINAMATH_GPT_solve_inequality_l1370_137029

theorem solve_inequality :
  ∀ x : ℝ, (x - 2) * (x + 2) < 5 ↔ -3 < x ∧ x < 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1370_137029


namespace NUMINAMATH_GPT_no_sum_of_squares_of_rationals_l1370_137060

theorem no_sum_of_squares_of_rationals (p q r s : ℕ) (hq : q ≠ 0) (hs : s ≠ 0)
    (hpq : Nat.gcd p q = 1) (hrs : Nat.gcd r s = 1) :
    (↑p / q : ℚ) ^ 2 + (↑r / s : ℚ) ^ 2 ≠ 168 := by 
    sorry

end NUMINAMATH_GPT_no_sum_of_squares_of_rationals_l1370_137060


namespace NUMINAMATH_GPT_find_a_l1370_137008

theorem find_a (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y = 0 → 3*x + y + a = 0) →
  a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1370_137008


namespace NUMINAMATH_GPT_neither_necessary_nor_sufficient_l1370_137051

noncomputable def C1 (m n : ℝ) :=
  (m ^ 2 - 4 * n ≥ 0) ∧ (m > 0) ∧ (n > 0)

noncomputable def C2 (m n : ℝ) :=
  (m > 0) ∧ (n > 0) ∧ (m ≠ n)

theorem neither_necessary_nor_sufficient (m n : ℝ) :
  ¬(C1 m n → C2 m n) ∧ ¬(C2 m n → C1 m n) :=
sorry

end NUMINAMATH_GPT_neither_necessary_nor_sufficient_l1370_137051


namespace NUMINAMATH_GPT_min_value_3x_4y_l1370_137064

theorem min_value_3x_4y {x y : ℝ} (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 5 * x * y) :
    3 * x + 4 * y ≥ 5 :=
sorry

end NUMINAMATH_GPT_min_value_3x_4y_l1370_137064


namespace NUMINAMATH_GPT_xy_sum_is_2_l1370_137069

theorem xy_sum_is_2 (x y : ℝ) 
  (h1 : (x - 1) ^ 3 + 1997 * (x - 1) = -1)
  (h2 : (y - 1) ^ 3 + 1997 * (y - 1) = 1) : 
  x + y = 2 := 
  sorry

end NUMINAMATH_GPT_xy_sum_is_2_l1370_137069


namespace NUMINAMATH_GPT_candy_bar_multiple_l1370_137000

theorem candy_bar_multiple (s m x : ℕ) (h1 : s = m * x + 6) (h2 : x = 24) (h3 : s = 78) : m = 3 :=
by
  sorry

end NUMINAMATH_GPT_candy_bar_multiple_l1370_137000


namespace NUMINAMATH_GPT_pencils_undefined_l1370_137094

-- Definitions for the conditions given in the problem
def initial_crayons : Nat := 41
def added_crayons : Nat := 12
def total_crayons : Nat := 53

-- Theorem stating the problem's required proof
theorem pencils_undefined (initial_crayons : Nat) (added_crayons : Nat) (total_crayons : Nat) : Prop :=
  initial_crayons = 41 ∧ added_crayons = 12 ∧ total_crayons = 53 → 
  ∃ (pencils : Nat), true
-- Since the number of pencils is unknown and no direct information is given, we represent it as an existential statement that pencils exist in some quantity, but we cannot determine their exact number based on given information.

end NUMINAMATH_GPT_pencils_undefined_l1370_137094


namespace NUMINAMATH_GPT_range_of_a_l1370_137048

noncomputable def range_a : Set ℝ :=
  {a : ℝ | 0 < a ∧ a ≤ 1/2}

theorem range_of_a (O P : ℝ × ℝ) (Q : ℝ × ℝ)
  (hO : O = (0, 0))
  (hP : P = (a, 0))
  (ha : 0 < a)
  (hQ : ∃ m : ℝ, Q = (m^2, m))
  (hPQ_PO : ∀ Q, Q = (m^2, m) → dist P Q ≥ dist O P) :
  a ∈ range_a :=
sorry

end NUMINAMATH_GPT_range_of_a_l1370_137048


namespace NUMINAMATH_GPT_max_mn_sq_l1370_137067

theorem max_mn_sq {m n : ℤ} (h1: 1 ≤ m ∧ m ≤ 2005) (h2: 1 ≤ n ∧ n ≤ 2005) 
(h3: (n^2 + 2*m*n - 2*m^2)^2 = 1): m^2 + n^2 ≤ 702036 :=
sorry

end NUMINAMATH_GPT_max_mn_sq_l1370_137067


namespace NUMINAMATH_GPT_relationship_l1370_137077

noncomputable def a : ℝ := 3^(-1/3 : ℝ)
noncomputable def b : ℝ := Real.log 3 / Real.log 2⁻¹
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem relationship (a_def : a = 3^(-1/3 : ℝ)) 
                     (b_def : b = Real.log 3 / Real.log 2⁻¹) 
                     (c_def : c = Real.log 3 / Real.log 2) : 
  b < a ∧ a < c :=
  sorry

end NUMINAMATH_GPT_relationship_l1370_137077


namespace NUMINAMATH_GPT_solve_quadratic_l1370_137062

theorem solve_quadratic (x : ℝ) (h : x^2 = 9) : x = 3 ∨ x = -3 :=
sorry

end NUMINAMATH_GPT_solve_quadratic_l1370_137062


namespace NUMINAMATH_GPT_arithmetic_sequence_properties_l1370_137087

variable {a : ℕ → ℕ}
variable {n : ℕ}

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∃ a1 d, ∀ n, a n = a1 + (n - 1) * d

theorem arithmetic_sequence_properties 
  (a_3_eq_7 : a 3 = 7)
  (a_5_plus_a_7_eq_26 : a 5 + a 7 = 26) :
  (∃ a1 d, (a 1 = a1) ∧ (∀ n, a n = a1 + (n - 1) * d) ∧ d = 2) ∧
  (∀ n, a n = 2 * n + 1) ∧
  (∀ S_n, S_n = n^2 + 2 * n) ∧ 
  ∀ T_n n, (∃ b : (ℕ → ℕ) → ℕ → ℕ, b a n = 1 / (a n ^ 2 - 1)) 
  → T_n = n / (4 * (n + 1)) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_properties_l1370_137087


namespace NUMINAMATH_GPT_f_odd_f_monotonic_increasing_intervals_f_no_max_value_f_extreme_points_l1370_137023

noncomputable def f (x : ℝ) : ℝ := x^3 - x

theorem f_odd (x : ℝ) : f (-x) = -f (x) :=
by sorry

theorem f_monotonic_increasing_intervals :
  ∀ x : ℝ, (x < -Real.sqrt 3 / 3 ∨ x > Real.sqrt 3 / 3) → f x' > f x :=
by sorry

theorem f_no_max_value :
  ∀ x : ℝ, ¬(∃ M, f x ≤ M) :=
by sorry

theorem f_extreme_points :
  f (-Real.sqrt 3 / 3) = -2 * Real.sqrt 3 / 9 ∧ f (Real.sqrt 3 / 3) = -2 * Real.sqrt 3 / 9 :=
by sorry

end NUMINAMATH_GPT_f_odd_f_monotonic_increasing_intervals_f_no_max_value_f_extreme_points_l1370_137023


namespace NUMINAMATH_GPT_line_parallel_through_M_line_perpendicular_through_M_l1370_137080

-- Define the lines L1 and L2
def L1 (x y: ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def L2 (x y: ℝ) : Prop := x - 3 * y + 8 = 0

-- Define the parallel and perpendicular lines
def parallel_to_line (x y: ℝ) : Prop := 2 * x + y + 5 = 0
def perpendicular_to_line (x y: ℝ) : Prop := 2 * x + y + 5 = 0

-- Define the intersection points
def M : ℝ × ℝ := (-2, 2)

-- Define the lines that pass through point M and are parallel or perpendicular to the given line
def line_parallel (x y: ℝ) : Prop := 2 * x + y + 2 = 0
def line_perpendicular (x y: ℝ) : Prop := x - 2 * y + 6 = 0

-- The proof statements
theorem line_parallel_through_M : ∃ x y : ℝ, L1 x y ∧ L2 x y ∧ x = (-2) ∧ y = 2 -> line_parallel x y := by
  sorry

theorem line_perpendicular_through_M : ∃ x y : ℝ, L1 x y ∧ L2 x y ∧ x = (-2) ∧ y = 2 -> line_perpendicular x y := by
  sorry

end NUMINAMATH_GPT_line_parallel_through_M_line_perpendicular_through_M_l1370_137080


namespace NUMINAMATH_GPT_count_bases_for_last_digit_l1370_137035

theorem count_bases_for_last_digit (n : ℕ) : n = 729 → ∃ S : Finset ℕ, S.card = 2 ∧ ∀ b ∈ S, 2 ≤ b ∧ b ≤ 10 ∧ (n - 5) % b = 0 :=
by
  sorry

end NUMINAMATH_GPT_count_bases_for_last_digit_l1370_137035


namespace NUMINAMATH_GPT_triangle_isosceles_of_sin_condition_l1370_137054

noncomputable def isosceles_triangle (A B C : ℝ) : Prop :=
  A = B ∨ B = C ∨ C = A

theorem triangle_isosceles_of_sin_condition {A B C : ℝ} (h : 2 * Real.sin A * Real.cos B = Real.sin C) : 
  isosceles_triangle A B C :=
by
  sorry

end NUMINAMATH_GPT_triangle_isosceles_of_sin_condition_l1370_137054


namespace NUMINAMATH_GPT_max_slope_no_lattice_points_l1370_137074

theorem max_slope_no_lattice_points :
  (∃ b : ℚ, (∀ m : ℚ, 1 / 3 < m ∧ m < b → ∀ x : ℤ, 0 < x ∧ x ≤ 200 → ¬ ∃ y : ℤ, y = m * x + 3) ∧ b = 68 / 203) := 
sorry

end NUMINAMATH_GPT_max_slope_no_lattice_points_l1370_137074


namespace NUMINAMATH_GPT_min_value_a_l1370_137086

theorem min_value_a (a b c d : ℚ) (h₀ : a > 0)
  (h₁ : ∀ n : ℕ, (a * n^3 + b * n^2 + c * n + d).den = 1) :
  a = 1/6 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_min_value_a_l1370_137086


namespace NUMINAMATH_GPT_min_width_of_garden_l1370_137034

theorem min_width_of_garden (w : ℝ) (h : w*(w + 10) ≥ 150) : w ≥ 10 :=
by
  sorry

end NUMINAMATH_GPT_min_width_of_garden_l1370_137034


namespace NUMINAMATH_GPT_daily_practice_hours_l1370_137059

-- Define the conditions as given in the problem
def total_hours_practiced_this_week : ℕ := 36
def total_days_in_week : ℕ := 7
def days_could_not_practice : ℕ := 1
def actual_days_practiced := total_days_in_week - days_could_not_practice

-- State the theorem including the question and the correct answer, given the conditions
theorem daily_practice_hours :
  total_hours_practiced_this_week / actual_days_practiced = 6 := 
by
  sorry

end NUMINAMATH_GPT_daily_practice_hours_l1370_137059


namespace NUMINAMATH_GPT_arithmetic_sequence_b3b7_l1370_137036

theorem arithmetic_sequence_b3b7 (b : ℕ → ℤ) (d : ℤ)
  (h_arith_seq : ∀ n, b (n + 1) = b n + d)
  (h_increasing : ∀ n, b n < b (n + 1))
  (h_cond : b 4 * b 6 = 17) : 
  b 3 * b 7 = -175 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_b3b7_l1370_137036


namespace NUMINAMATH_GPT_triangle_area_l1370_137078

theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (h_perimeter : perimeter = 40) (h_inradius : inradius = 2.5) : 
  (inradius * (perimeter / 2)) = 50 :=
by
  -- Lean 4 statement code
  sorry

end NUMINAMATH_GPT_triangle_area_l1370_137078


namespace NUMINAMATH_GPT_general_formula_a_general_formula_c_l1370_137030

-- Definition of the sequence {a_n}
def S (n : ℕ) : ℕ := n^2 + 2 * n
def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem general_formula_a (n : ℕ) (hn : n > 0) : a n = 2 * n + 1 := sorry

-- Definitions for the second problem
def f (x : ℝ) : ℝ := x^2 + 2 * x
def f' (x : ℝ) : ℝ := 2 * x + 2
def k (n : ℕ) : ℝ := 2 * n + 2

def Q (k : ℝ) : Prop := ∃ (n : ℕ), k = 2 * n + 2
def R (k : ℝ) : Prop := ∃ (n : ℕ), k = 4 * n + 2

def c (n : ℕ) : ℕ := 12 * n - 6

theorem general_formula_c (n : ℕ) (hn1 : 0 < c 10)
    (hn2 : c 10 < 115) : c n = 12 * n - 6 := sorry

end NUMINAMATH_GPT_general_formula_a_general_formula_c_l1370_137030


namespace NUMINAMATH_GPT_age_difference_36_l1370_137085

noncomputable def jack_age (a b : ℕ) : ℕ := 10 * a + b
noncomputable def bill_age (b a : ℕ) : ℕ := 10 * b + a

theorem age_difference_36 (a b : ℕ) (h : 10 * a + b + 3 = 3 * (10 * b + a + 3)) :
  jack_age a b - bill_age b a = 36 :=
by sorry

end NUMINAMATH_GPT_age_difference_36_l1370_137085
