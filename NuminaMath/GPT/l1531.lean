import Mathlib

namespace NUMINAMATH_GPT_quadratic_function_properties_l1531_153199

noncomputable def quadratic_function (m : ℝ) (x : ℝ) : ℝ :=
  (m + 2) * x^(m^2 + m - 4)

theorem quadratic_function_properties :
  (∀ m, (m^2 + m - 4 = 2) → (m = -3 ∨ m = 2))
  ∧ (m = -3 → quadratic_function m 0 = 0) 
  ∧ (m = -3 → ∀ x, x > 0 → quadratic_function m x ≤ quadratic_function m 0 ∧ quadratic_function m x < 0)
  ∧ (m = -3 → ∀ x, x < 0 → quadratic_function m x ≤ quadratic_function m 0 ∧ quadratic_function m x < 0) :=
by
  -- Proof will be supplied here.
  sorry

end NUMINAMATH_GPT_quadratic_function_properties_l1531_153199


namespace NUMINAMATH_GPT_zara_goats_l1531_153146

noncomputable def total_animals_per_group := 48
noncomputable def total_groups := 3
noncomputable def total_cows := 24
noncomputable def total_sheep := 7

theorem zara_goats : 
  (total_groups * total_animals_per_group = 144) ∧ 
  (144 = total_cows + total_sheep + 113) →
  113 = 144 - total_cows - total_sheep := 
by sorry

end NUMINAMATH_GPT_zara_goats_l1531_153146


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l1531_153105

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 6)

theorem problem_part1 :
  f (Real.pi / 12) = 3 * Real.sqrt 3 / 2 :=
by
  sorry

theorem problem_part2 (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < Real.pi / 2) :
  Real.sin θ = 4 / 5 →
  f (5 * Real.pi / 12 - θ) = 72 / 25 :=
by
  sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l1531_153105


namespace NUMINAMATH_GPT_bags_sold_in_first_week_l1531_153139

def total_bags_sold : ℕ := 100
def bags_sold_week1 (X : ℕ) : ℕ := X
def bags_sold_week2 (X : ℕ) : ℕ := 3 * X
def bags_sold_week3_4 : ℕ := 40

theorem bags_sold_in_first_week (X : ℕ) (h : total_bags_sold = bags_sold_week1 X + bags_sold_week2 X + bags_sold_week3_4) : X = 15 :=
by
  sorry

end NUMINAMATH_GPT_bags_sold_in_first_week_l1531_153139


namespace NUMINAMATH_GPT_count_four_digit_multiples_of_5_l1531_153195

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def four_digit_multiples_of_5_count (lower upper : ℕ) :=
  (upper - lower + 1)

theorem count_four_digit_multiples_of_5 : 
  four_digit_multiples_of_5_count 200 1999 = 1800 :=
by
  sorry

end NUMINAMATH_GPT_count_four_digit_multiples_of_5_l1531_153195


namespace NUMINAMATH_GPT_sum_g_equals_half_l1531_153196

noncomputable def g (n : ℕ) : ℝ :=
  ∑' k, if k ≥ 3 then 1 / k ^ n else 0

theorem sum_g_equals_half : ∑' n : ℕ, g n.succ = 1 / 2 := 
sorry

end NUMINAMATH_GPT_sum_g_equals_half_l1531_153196


namespace NUMINAMATH_GPT_male_athletes_drawn_l1531_153161

theorem male_athletes_drawn (total_males : ℕ) (total_females : ℕ) (total_sample : ℕ)
  (h_males : total_males = 20) (h_females : total_females = 10) (h_sample : total_sample = 6) :
  (total_sample * total_males) / (total_males + total_females) = 4 := 
  by
  sorry

end NUMINAMATH_GPT_male_athletes_drawn_l1531_153161


namespace NUMINAMATH_GPT_shade_half_grid_additional_squares_l1531_153112

/-- A 4x5 grid consists of 20 squares, of which 3 are already shaded. 
Prove that the number of additional 1x1 squares needed to shade half the grid is 7. -/
theorem shade_half_grid_additional_squares (total_squares shaded_squares remaining_squares: ℕ) 
  (h1 : total_squares = 4 * 5)
  (h2 : shaded_squares = 3)
  (h3 : remaining_squares = total_squares / 2 - shaded_squares) :
  remaining_squares = 7 :=
by
  -- Proof not required.
  sorry

end NUMINAMATH_GPT_shade_half_grid_additional_squares_l1531_153112


namespace NUMINAMATH_GPT_no_solution_system_of_equations_l1531_153153

theorem no_solution_system_of_equations :
  ¬ ∃ (x y : ℝ), (2 * x - 3 * y = 6) ∧ (4 * x - 6 * y = 8) ∧ (5 * x - 5 * y = 15) :=
by {
  sorry
}

end NUMINAMATH_GPT_no_solution_system_of_equations_l1531_153153


namespace NUMINAMATH_GPT_same_color_probability_correct_l1531_153176

noncomputable def prob_same_color (green red blue : ℕ) : ℚ :=
  let total := green + red + blue
  (green / total) * (green / total) +
  (red / total) * (red / total) +
  (blue / total) * (blue / total)

theorem same_color_probability_correct :
  prob_same_color 5 7 3 = 83 / 225 :=
by
  sorry

end NUMINAMATH_GPT_same_color_probability_correct_l1531_153176


namespace NUMINAMATH_GPT_polynomial_coefficient_sum_l1531_153170

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (1 - 2 * x) ^ 4 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 →
  a₀ + a₁ + a₃ = -39 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_coefficient_sum_l1531_153170


namespace NUMINAMATH_GPT_max_bio_homework_time_l1531_153158

-- Define our variables as non-negative real numbers
variables (B H G : ℝ)

-- Given conditions
axiom h1 : H = 2 * B
axiom h2 : G = 6 * B
axiom h3 : B + H + G = 180

-- We need to prove that B = 20
theorem max_bio_homework_time : B = 20 :=
by
  sorry

end NUMINAMATH_GPT_max_bio_homework_time_l1531_153158


namespace NUMINAMATH_GPT_salt_solution_mixture_l1531_153111

/-- Let's define the conditions and hypotheses required for our proof. -/
def ounces_of_salt_solution 
  (percent_salt : ℝ) (amount : ℝ) : ℝ := percent_salt * amount

def final_amount (x : ℝ) : ℝ := x + 70
def final_salt_content (x : ℝ) : ℝ := 0.40 * (x + 70)

theorem salt_solution_mixture (x : ℝ) :
  0.60 * x + 0.20 * 70 = 0.40 * (x + 70) ↔ x = 70 :=
by {
  sorry
}

end NUMINAMATH_GPT_salt_solution_mixture_l1531_153111


namespace NUMINAMATH_GPT_right_triangle_expression_l1531_153125

theorem right_triangle_expression (a c b : ℝ) (h1 : c = a + 2) (h2 : a^2 + b^2 = c^2) : 
  b^2 = 4 * (a + 1) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_expression_l1531_153125


namespace NUMINAMATH_GPT_surveyed_individuals_not_working_percentage_l1531_153120

theorem surveyed_individuals_not_working_percentage :
  (55 / 100 * 0 + 35 / 100 * (1 / 8) + 10 / 100 * (1 / 4)) = 6.875 / 100 :=
by
  sorry

end NUMINAMATH_GPT_surveyed_individuals_not_working_percentage_l1531_153120


namespace NUMINAMATH_GPT_geometric_sequence_sixth_term_l1531_153187

theorem geometric_sequence_sixth_term (a : ℝ) (r : ℝ) 
  (h1 : a = 512) 
  (h2 : a * r^(7) = 2) :
  a * r^(5) = 16 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sixth_term_l1531_153187


namespace NUMINAMATH_GPT_items_sold_increase_by_20_percent_l1531_153165

-- Assume initial variables P (price per item without discount) and N (number of items sold without discount)
variables (P N : ℝ)

-- Define the conditions and the final proof goal
theorem items_sold_increase_by_20_percent 
  (h1 : ∀ (P N : ℝ), P > 0 → N > 0 → (P * N > 0))
  (h2 : ∀ (P : ℝ), P' = P * 0.90)
  (h3 : ∀ (P' N' : ℝ), P' * N' = P * N * 1.08)
  : (N' - N) / N * 100 = 20 := 
sorry

end NUMINAMATH_GPT_items_sold_increase_by_20_percent_l1531_153165


namespace NUMINAMATH_GPT_sin_gamma_isosceles_l1531_153145

theorem sin_gamma_isosceles (a c m_a m_c s_1 s_2 : ℝ) (γ : ℝ) 
  (h1 : a + m_c = s_1) (h2 : c + m_a = s_2) :
  Real.sin γ = (s_2 / (2 * s_1)) * Real.sqrt ((4 * s_1^2) - s_2^2) :=
sorry

end NUMINAMATH_GPT_sin_gamma_isosceles_l1531_153145


namespace NUMINAMATH_GPT_greatest_divisor_condition_l1531_153114

-- Define conditions
def leaves_remainder (a b k : ℕ) : Prop := ∃ q : ℕ, a = b * q + k

-- Define the greatest common divisor property
def gcd_of (a b k: ℕ) (g : ℕ) : Prop :=
  leaves_remainder a k g ∧ leaves_remainder b k g ∧ ∀ d : ℕ, (leaves_remainder a k d ∧ leaves_remainder b k d) → d ≤ g

theorem greatest_divisor_condition 
  (N : ℕ) (h1 : leaves_remainder 1657 N 6) (h2 : leaves_remainder 2037 N 5) :
  N = 127 :=
sorry

end NUMINAMATH_GPT_greatest_divisor_condition_l1531_153114


namespace NUMINAMATH_GPT_exists_initial_segment_of_power_of_2_l1531_153127

theorem exists_initial_segment_of_power_of_2 (m : ℕ) : ∃ n : ℕ, ∃ k : ℕ, k ≥ m ∧ 2^n = 10^k * m ∨ 2^n = 10^k * (m+1) := 
by
  sorry

end NUMINAMATH_GPT_exists_initial_segment_of_power_of_2_l1531_153127


namespace NUMINAMATH_GPT_complement_intersection_l1531_153167

def U : Set ℤ := {1, 2, 3, 4, 5}
def P : Set ℤ := {2, 4}
def Q : Set ℤ := {1, 3, 4, 6}
def C_U_P : Set ℤ := {x ∈ U | x ∉ P}

theorem complement_intersection :
  (C_U_P ∩ Q) = {1, 3} :=
by sorry

end NUMINAMATH_GPT_complement_intersection_l1531_153167


namespace NUMINAMATH_GPT_complex_expression_calculation_l1531_153181

noncomputable def complex_i := Complex.I -- Define the imaginary unit i

theorem complex_expression_calculation : complex_i * (1 - complex_i)^2 = 2 := by
  sorry

end NUMINAMATH_GPT_complex_expression_calculation_l1531_153181


namespace NUMINAMATH_GPT_find_N_l1531_153150

theorem find_N (N : ℕ) : (4^5)^2 * (2^5)^4 = 2^N → N = 30 :=
by
  intros h
  -- Sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_find_N_l1531_153150


namespace NUMINAMATH_GPT_f_at_pos_eq_l1531_153149

noncomputable def f (x : ℝ) : ℝ :=
  if h : x < 0 then x * (x - 1)
  else if h : x > 0 then -x * (x + 1)
  else 0

theorem f_at_pos_eq (x : ℝ) (hx : 0 < x) : f x = -x * (x + 1) :=
by
  -- Assume f is an odd function
  have h_odd : ∀ x : ℝ, f (-x) = -f x := sorry
  
  -- Given for x in (-∞, 0), f(x) = x * (x - 1)
  have h_neg : ∀ x : ℝ, x < 0 → f x = x * (x - 1) := sorry
  
  -- Prove for x > 0, f(x) = -x * (x + 1)
  sorry

end NUMINAMATH_GPT_f_at_pos_eq_l1531_153149


namespace NUMINAMATH_GPT_garden_area_garden_perimeter_l1531_153172

noncomputable def length : ℝ := 30
noncomputable def width : ℝ := length / 2
noncomputable def area : ℝ := length * width
noncomputable def perimeter : ℝ := 2 * (length + width)

theorem garden_area :
  area = 450 :=
sorry

theorem garden_perimeter :
  perimeter = 90 :=
sorry

end NUMINAMATH_GPT_garden_area_garden_perimeter_l1531_153172


namespace NUMINAMATH_GPT_new_difference_greater_l1531_153193

theorem new_difference_greater (x y a b : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x > y) (ha_pos : 0 < a) (hb_pos : 0 < b) (hab : a ≠ b) :
  (x + a) - (y - b) > x - y :=
by {
  sorry
}

end NUMINAMATH_GPT_new_difference_greater_l1531_153193


namespace NUMINAMATH_GPT_dice_minimum_rolls_l1531_153166

theorem dice_minimum_rolls (d1 d2 d3 d4 : ℕ) (h1 : 1 ≤ d1 ∧ d1 ≤ 6)
                           (h2 : 1 ≤ d2 ∧ d2 ≤ 6) (h3 : 1 ≤ d3 ∧ d3 ≤ 6) 
                           (h4 : 1 ≤ d4 ∧ d4 ≤ 6) :
  ∃ n, n = 43 ∧ ∀ (S : ℕ) (x : ℕ → ℕ), 
  (∀ i, 4 ≤ S ∧ S ≤ 24 ∧ x i = 4 ∧ (x i ≤ 6)) →
  (n ≤ 43) ∧ (∃ (k : ℕ), k ≥ 3) :=
sorry

end NUMINAMATH_GPT_dice_minimum_rolls_l1531_153166


namespace NUMINAMATH_GPT_num_terms_arithmetic_sequence_is_15_l1531_153109

theorem num_terms_arithmetic_sequence_is_15 :
  ∃ n : ℕ, (∀ (a : ℤ), a = -58 + (n - 1) * 7 → a = 44) ∧ n = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_num_terms_arithmetic_sequence_is_15_l1531_153109


namespace NUMINAMATH_GPT_mod_pow_eq_l1531_153160

theorem mod_pow_eq (m : ℕ) (h1 : 13^4 % 11 = m) (h2 : 0 ≤ m ∧ m < 11) : m = 5 := by
  sorry

end NUMINAMATH_GPT_mod_pow_eq_l1531_153160


namespace NUMINAMATH_GPT_sufficient_no_x_axis_intersections_l1531_153186

/-- Sufficient condition for no x-axis intersections -/
theorem sufficient_no_x_axis_intersections
    (a b c : ℝ)
    (h : a ≠ 0)
    (h_sufficient : b^2 - 4 * a * c < -1) :
    ∀ x : ℝ, ¬(a * x^2 + b * x + c = 0) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_no_x_axis_intersections_l1531_153186


namespace NUMINAMATH_GPT_geometric_progression_exists_l1531_153118

theorem geometric_progression_exists :
  ∃ (b1 b2 b3 b4: ℤ) (q: ℤ), 
    b2 = b1 * q ∧ 
    b3 = b1 * q^2 ∧ 
    b4 = b1 * q^3 ∧  
    b3 - b1 = 9 ∧ 
    b2 - b4 = 18 ∧ 
    b1 = 3 ∧ b2 = -6 ∧ b3 = 12 ∧ b4 = -24 :=
sorry

end NUMINAMATH_GPT_geometric_progression_exists_l1531_153118


namespace NUMINAMATH_GPT_A_runs_faster_l1531_153129

variable (v_A v_B : ℝ)  -- Speed of A and B
variable (k : ℝ)       -- Factor by which A is faster than B

-- Conditions as definitions in Lean:
def speed_relation (k : ℝ) (v_A v_B : ℝ) : Prop := v_A = k * v_B
def start_difference : ℝ := 60
def race_course_length : ℝ := 80
def reach_finish_same_time (v_A v_B : ℝ) : Prop := (80 / v_A) = ((80 - start_difference) / v_B)

theorem A_runs_faster
  (h1 : speed_relation k v_A v_B)
  (h2 : reach_finish_same_time v_A v_B) : k = 4 :=
by
  sorry

end NUMINAMATH_GPT_A_runs_faster_l1531_153129


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1531_153192

theorem quadratic_inequality_solution {a : ℝ} :
  (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ a < -1 ∨ a > 3 :=
by sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1531_153192


namespace NUMINAMATH_GPT_concentration_time_within_bounds_l1531_153142

-- Define the time bounds for the highest concentration of the drug in the blood
def highest_concentration_time_lower (base : ℝ) (tolerance : ℝ) : ℝ := base - tolerance
def highest_concentration_time_upper (base : ℝ) (tolerance : ℝ) : ℝ := base + tolerance

-- Define the base and tolerance values
def base_time : ℝ := 0.65
def tolerance_time : ℝ := 0.15

-- Define the specific time we want to prove is within the bounds
def specific_time : ℝ := 0.8

-- Theorem statement
theorem concentration_time_within_bounds : 
  highest_concentration_time_lower base_time tolerance_time ≤ specific_time ∧ 
  specific_time ≤ highest_concentration_time_upper base_time tolerance_time :=
by sorry

end NUMINAMATH_GPT_concentration_time_within_bounds_l1531_153142


namespace NUMINAMATH_GPT_angle_terminal_side_equiv_l1531_153140

theorem angle_terminal_side_equiv (k : ℤ) : 
  ∀ θ α : ℝ, θ = - (π / 3) → α = 5 * π / 3 → α = θ + 2 * k * π := by
  intro θ α hθ hα
  sorry

end NUMINAMATH_GPT_angle_terminal_side_equiv_l1531_153140


namespace NUMINAMATH_GPT_marble_count_l1531_153128

theorem marble_count (x : ℝ) (h1 : 0.5 * x = x / 2) (h2 : 0.25 * x = x / 4) (h3 : (27 + 14) = 41) (h4 : 0.25 * x = 27 + 14)
  : x = 164 :=
by
  sorry

end NUMINAMATH_GPT_marble_count_l1531_153128


namespace NUMINAMATH_GPT_find_smallest_N_l1531_153179

def smallest_possible_N (N : ℕ) : Prop :=
  ∃ (W : Fin N → ℝ), 
  (∀ i j, W i ≤ 1.25 * W j ∧ W j ≤ 1.25 * W i) ∧ 
  (∃ (P : Fin 10 → Finset (Fin N)), ∀ i j, i ≤ j →
    P i ≠ ∅ ∧ 
    Finset.sum (P i) W = Finset.sum (P j) W) ∧
  (∃ (V : Fin 11 → Finset (Fin N)), ∀ i j, i ≤ j →
    V i ≠ ∅ ∧ 
    Finset.sum (V i) W = Finset.sum (V j) W)

theorem find_smallest_N : smallest_possible_N 50 :=
sorry

end NUMINAMATH_GPT_find_smallest_N_l1531_153179


namespace NUMINAMATH_GPT_second_derivative_l1531_153110

noncomputable def y (x : ℝ) : ℝ := x^3 + Real.log x / Real.log 2 + Real.exp (-x)

theorem second_derivative (x : ℝ) : (deriv^[2] y x) = 3 * x^2 + (1 / (x * Real.log 2)) - Real.exp (-x) :=
by
  sorry

end NUMINAMATH_GPT_second_derivative_l1531_153110


namespace NUMINAMATH_GPT_diamond_of_2_and_3_l1531_153103

def diamond (a b : ℕ) : ℕ := a^3 * b^2 - b + 2

theorem diamond_of_2_and_3 : diamond 2 3 = 71 := by
  sorry

end NUMINAMATH_GPT_diamond_of_2_and_3_l1531_153103


namespace NUMINAMATH_GPT_negation_of_universal_l1531_153154

theorem negation_of_universal :
  ¬(∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
by sorry

end NUMINAMATH_GPT_negation_of_universal_l1531_153154


namespace NUMINAMATH_GPT_fixed_point_translation_l1531_153180

variable {R : Type*} [LinearOrderedField R]

def passes_through (f : R → R) (p : R × R) : Prop := f p.1 = p.2

theorem fixed_point_translation (f : R → R) (h : f 1 = 1) :
  passes_through (fun x => f (x + 2)) (-1, 1) :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_translation_l1531_153180


namespace NUMINAMATH_GPT_positive_number_property_l1531_153113

theorem positive_number_property (x : ℝ) (h_pos : 0 < x) (h_property : (x^2 / 100) = 9) : x = 30 :=
by
  sorry

end NUMINAMATH_GPT_positive_number_property_l1531_153113


namespace NUMINAMATH_GPT_disproving_proposition_l1531_153189

theorem disproving_proposition : ∃ (angle1 angle2 : ℝ), angle1 = angle2 ∧ angle1 + angle2 = 90 :=
by
  sorry

end NUMINAMATH_GPT_disproving_proposition_l1531_153189


namespace NUMINAMATH_GPT_perp_to_par_perp_l1531_153188

variable (m : Line)
variable (α β : Plane)

-- Conditions
axiom parallel_planes (α β : Plane) : Prop
axiom perp (m : Line) (α : Plane) : Prop

-- Statements
axiom parallel_planes_ax : parallel_planes α β
axiom perp_ax : perp m α

-- Goal
theorem perp_to_par_perp {m : Line} {α β : Plane} (h1 : perp m α) (h2 : parallel_planes α β) : perp m β := sorry

end NUMINAMATH_GPT_perp_to_par_perp_l1531_153188


namespace NUMINAMATH_GPT_transform_unit_square_l1531_153100

-- Define the unit square vertices in the xy-plane
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (0, 1)

-- Transformation functions from the xy-plane to the uv-plane
def transform_u (x y : ℝ) : ℝ := x^2 - y^2
def transform_v (x y : ℝ) : ℝ := x * y

-- Vertex transformation results
def O_image : ℝ × ℝ := (transform_u 0 0, transform_v 0 0)  -- (0,0)
def A_image : ℝ × ℝ := (transform_u 1 0, transform_v 1 0)  -- (1,0)
def B_image : ℝ × ℝ := (transform_u 1 1, transform_v 1 1)  -- (0,1)
def C_image : ℝ × ℝ := (transform_u 0 1, transform_v 0 1)  -- (-1,0)

-- The Lean 4 theorem statement
theorem transform_unit_square :
  O_image = (0, 0) ∧
  A_image = (1, 0) ∧
  B_image = (0, 1) ∧
  C_image = (-1, 0) :=
  by sorry

end NUMINAMATH_GPT_transform_unit_square_l1531_153100


namespace NUMINAMATH_GPT_root_of_equation_l1531_153115

theorem root_of_equation : 
  ∀ x : ℝ, x ≠ 3 → x ≠ -3 → (18 / (x^2 - 9) - 3 / (x - 3) = 2) → (x = -4.5) :=
by sorry

end NUMINAMATH_GPT_root_of_equation_l1531_153115


namespace NUMINAMATH_GPT_find_second_discount_l1531_153185

theorem find_second_discount 
    (list_price : ℝ)
    (final_price : ℝ)
    (first_discount : ℝ)
    (second_discount : ℝ)
    (h₁ : list_price = 65)
    (h₂ : final_price = 57.33)
    (h₃ : first_discount = 0.10)
    (h₄ : (list_price - (first_discount * list_price)) = 58.5)
    (h₅ : final_price = 58.5 - (second_discount * 58.5)) :
    second_discount = 0.02 := 
by
  sorry

end NUMINAMATH_GPT_find_second_discount_l1531_153185


namespace NUMINAMATH_GPT_smaller_angle_clock_1245_l1531_153198

theorem smaller_angle_clock_1245 
  (minute_rate : ℕ → ℝ) 
  (hour_rate : ℕ → ℝ) 
  (time : ℕ) 
  (minute_angle : ℝ) 
  (hour_angle : ℝ) 
  (larger_angle : ℝ) 
  (smaller_angle : ℝ) :
  (minute_rate 1 = 6) →
  (hour_rate 1 = 0.5) →
  (time = 45) →
  (minute_angle = minute_rate 45 * 45) →
  (hour_angle = hour_rate 45 * 45) →
  (larger_angle = |minute_angle - hour_angle|) →
  (smaller_angle = 360 - larger_angle) →
  smaller_angle = 112.5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_smaller_angle_clock_1245_l1531_153198


namespace NUMINAMATH_GPT_card_collection_average_l1531_153159

theorem card_collection_average (n : ℕ) (h : (2 * n + 1) / 3 = 2017) : n = 3025 :=
by
  sorry

end NUMINAMATH_GPT_card_collection_average_l1531_153159


namespace NUMINAMATH_GPT_log_sum_zero_l1531_153177

theorem log_sum_zero (a b c N : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_N : 0 < N) (h_neq_N : N ≠ 1) (h_geom_mean : b^2 = a * c) : 
  1 / Real.logb a N - 2 / Real.logb b N + 1 / Real.logb c N = 0 :=
  by
  sorry

end NUMINAMATH_GPT_log_sum_zero_l1531_153177


namespace NUMINAMATH_GPT_fifth_inequality_proof_l1531_153131

theorem fifth_inequality_proof : 
  1 + (1 / (2:ℝ)^2) + (1 / (3:ℝ)^2) + (1 / (4:ℝ)^2) + (1 / (5:ℝ)^2) + (1 / (6:ℝ)^2) < (11 / 6) :=
by {
  sorry
}

end NUMINAMATH_GPT_fifth_inequality_proof_l1531_153131


namespace NUMINAMATH_GPT_prob1_prob2_prob3_l1531_153148

-- Problem 1
theorem prob1 (f : ℝ → ℝ) (m : ℝ) (f_def : ∀ x, f x = Real.log x + (1/2) * m * x^2)
  (tangent_line_slope : ℝ) (perpendicular_line_eq : ℝ) :
  (tangent_line_slope = 1 + m) →
  (perpendicular_line_eq = -1/2) →
  (tangent_line_slope * perpendicular_line_eq = -1) →
  m = 1 := sorry

-- Problem 2
theorem prob2 (f : ℝ → ℝ) (m : ℝ) (f_def : ∀ x, f x = Real.log x + (1/2) * m * x^2) :
  (∀ x, f x ≤ m * x^2 + (m - 1) * x - 1) →
  ∃ (m_ : ℤ), m_ ≥ 2 := sorry

-- Problem 3
theorem prob3 (f : ℝ → ℝ) (F : ℝ → ℝ) (x1 x2 : ℝ) (m : ℝ) 
  (f_def : ∀ x, f x = Real.log x + (1/2) * x^2)
  (F_def : ∀ x, F x = f x + x)
  (hx1 : 0 < x1) (hx2: 0 < x2) :
  m = 1 →
  F x1 = -F x2 →
  x1 + x2 ≥ Real.sqrt 3 - 1 := sorry

end NUMINAMATH_GPT_prob1_prob2_prob3_l1531_153148


namespace NUMINAMATH_GPT_smallest_n_circle_l1531_153116

theorem smallest_n_circle (n : ℕ) 
    (h1 : ∀ i j : ℕ, i < j → j - i = 3 ∨ j - i = 4 ∨ j - i = 5) :
    n = 7 :=
sorry

end NUMINAMATH_GPT_smallest_n_circle_l1531_153116


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1531_153168

theorem necessary_but_not_sufficient (a b : ℝ) :
  (a - b > 0 → a^2 - b^2 > 0) ∧ ¬(a^2 - b^2 > 0 → a - b > 0) := by
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1531_153168


namespace NUMINAMATH_GPT_right_triangle_third_side_l1531_153152

theorem right_triangle_third_side (x : ℝ) : 
  (∃ (a b c : ℝ), (a = 3 ∧ b = 4 ∧ (a^2 + b^2 = c^2 ∧ (c = x ∨ x^2 + a^2 = b^2)))) → (x = 5 ∨ x = Real.sqrt 7) :=
by 
  sorry

end NUMINAMATH_GPT_right_triangle_third_side_l1531_153152


namespace NUMINAMATH_GPT_proof_problem_l1531_153144

theorem proof_problem
  (a b : ℝ)
  (h1 : a = -(-3))
  (h2 : b = - (- (1 / 2))⁻¹)
  (m n : ℝ) :
  (|m - a| + |n + b| = 0) → (a = 3 ∧ b = -2 ∧ m = 3 ∧ n = -2) :=
by {
  sorry
}

end NUMINAMATH_GPT_proof_problem_l1531_153144


namespace NUMINAMATH_GPT_how_many_cheburashkas_erased_l1531_153102

theorem how_many_cheburashkas_erased 
  (total_krakozyabras : ℕ)
  (characters_per_row_initial : ℕ) 
  (total_characters_initial : ℕ)
  (total_cheburashkas : ℕ)
  (total_rows : ℕ := 2)
  (total_krakozyabras := 29) :
  total_cheburashkas = 11 :=
by
  sorry

end NUMINAMATH_GPT_how_many_cheburashkas_erased_l1531_153102


namespace NUMINAMATH_GPT_find_value_l1531_153119

theorem find_value (a b : ℝ) (h : (a + 1)^2 + |b - 2| = 0) : a^2006 + (a + b)^2007 = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_value_l1531_153119


namespace NUMINAMATH_GPT_range_of_a_l1531_153174

noncomputable def f (a x : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 < x → (f a x) > 0) ↔ a ≤ 2 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1531_153174


namespace NUMINAMATH_GPT_find_C_l1531_153134

-- Define the sum of interior angles of a triangle
def sum_of_triangle_angles := 180

-- Define the total angles sum in a closed figure formed by multiple triangles
def total_internal_angles := 1080

-- Define the value to prove
def C := total_internal_angles - sum_of_triangle_angles

theorem find_C:
  C = 900 := by
  sorry

end NUMINAMATH_GPT_find_C_l1531_153134


namespace NUMINAMATH_GPT_monkey_climbing_time_l1531_153182

-- Define the conditions
def tree_height : ℕ := 20
def hop_distance : ℕ := 3
def slip_distance : ℕ := 2
def net_distance_per_hour : ℕ := hop_distance - slip_distance

-- Define the theorem statement
theorem monkey_climbing_time : ∃ (t : ℕ), t = 18 ∧ (net_distance_per_hour * (t - 1) + hop_distance) >= tree_height :=
by
  sorry

end NUMINAMATH_GPT_monkey_climbing_time_l1531_153182


namespace NUMINAMATH_GPT_jenny_eggs_in_each_basket_l1531_153101

theorem jenny_eggs_in_each_basket (n : ℕ) (h1 : 30 % n = 0) (h2 : 45 % n = 0) (h3 : n ≥ 5) : n = 15 :=
sorry

end NUMINAMATH_GPT_jenny_eggs_in_each_basket_l1531_153101


namespace NUMINAMATH_GPT_triangle_area_is_correct_l1531_153171

structure Point where
  x : ℝ
  y : ℝ

noncomputable def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * (abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)))

def A : Point := ⟨0, 0⟩
def B : Point := ⟨2, 2⟩
def C : Point := ⟨2, 0⟩

theorem triangle_area_is_correct : area_of_triangle A B C = 2 := by
  sorry

end NUMINAMATH_GPT_triangle_area_is_correct_l1531_153171


namespace NUMINAMATH_GPT_find_mark_age_l1531_153124

-- Define Mark and Aaron's ages
variables (M A : ℕ)

-- The conditions
def condition1 : Prop := M - 3 = 3 * (A - 3) + 1
def condition2 : Prop := M + 4 = 2 * (A + 4) + 2

-- The proof statement
theorem find_mark_age (h1 : condition1 M A) (h2 : condition2 M A) : M = 28 :=
by sorry

end NUMINAMATH_GPT_find_mark_age_l1531_153124


namespace NUMINAMATH_GPT_find_n_l1531_153138

-- We need a definition for permutations counting A_n^2 = n(n-1)
def permutations_squared (n : ℕ) : ℕ := n * (n - 1)

theorem find_n (n : ℕ) (h : permutations_squared n = 56) : n = 8 :=
by {
  sorry -- proof omitted as instructed
}

end NUMINAMATH_GPT_find_n_l1531_153138


namespace NUMINAMATH_GPT_solve_for_n_l1531_153122

theorem solve_for_n (n x y : ℤ) (h : n * (x + y) + 17 = n * (-x + y) - 21) (hx : x = 1) : n = -19 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_n_l1531_153122


namespace NUMINAMATH_GPT_not_possible_sum_2017_l1531_153163

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem not_possible_sum_2017 (A B : ℕ) (h1 : A + B = 2017) (h2 : sum_of_digits A = 2 * sum_of_digits B) : false := 
sorry

end NUMINAMATH_GPT_not_possible_sum_2017_l1531_153163


namespace NUMINAMATH_GPT_max_base_angle_is_7_l1531_153143

-- Define the conditions and the problem statement
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def isosceles_triangle (x : ℕ) : Prop :=
  is_prime x ∧ ∃ y : ℕ, 2 * x + y = 180 ∧ is_prime y

theorem max_base_angle_is_7 :
  ∃ (x : ℕ), isosceles_triangle x ∧ x = 7 :=
by
  sorry

end NUMINAMATH_GPT_max_base_angle_is_7_l1531_153143


namespace NUMINAMATH_GPT_smallest_positive_t_l1531_153147

theorem smallest_positive_t (x_1 x_2 x_3 x_4 x_5 t : ℝ) :
  (x_1 + x_3 = 2 * t * x_2) →
  (x_2 + x_4 = 2 * t * x_3) →
  (x_3 + x_5 = 2 * t * x_4) →
  (0 ≤ x_1) →
  (0 ≤ x_2) →
  (0 ≤ x_3) →
  (0 ≤ x_4) →
  (0 ≤ x_5) →
  (x_1 ≠ 0 ∨ x_2 ≠ 0 ∨ x_3 ≠ 0 ∨ x_4 ≠ 0 ∨ x_5 ≠ 0) →
  t = 1 / Real.sqrt 2 → 
  ∃ t, (0 < t) ∧ (x_1 + x_3 = 2 * t * x_2) ∧ (x_2 + x_4 = 2 * t * x_3) ∧ (x_3 + x_5 = 2 * t * x_4)
:=
sorry

end NUMINAMATH_GPT_smallest_positive_t_l1531_153147


namespace NUMINAMATH_GPT_number_of_balls_to_remove_l1531_153183

theorem number_of_balls_to_remove:
  ∀ (x : ℕ), 120 - x = (48 : ℕ) / (0.75 : ℝ) → x = 56 :=
by sorry

end NUMINAMATH_GPT_number_of_balls_to_remove_l1531_153183


namespace NUMINAMATH_GPT_number_of_subsets_l1531_153104

-- Define the set
def my_set : Set ℕ := {1, 2, 3}

-- Theorem statement
theorem number_of_subsets : Finset.card (Finset.powerset {1, 2, 3}) = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_subsets_l1531_153104


namespace NUMINAMATH_GPT_tangerine_initial_count_l1531_153191

theorem tangerine_initial_count 
  (X : ℕ) 
  (h1 : X - 9 + 5 = 20) : 
  X = 24 :=
sorry

end NUMINAMATH_GPT_tangerine_initial_count_l1531_153191


namespace NUMINAMATH_GPT_am_gm_inequality_l1531_153136

theorem am_gm_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 1) :
  (1 - x) * (1 - y) * (1 - z) ≥ 8 * x * y * z :=
by sorry

end NUMINAMATH_GPT_am_gm_inequality_l1531_153136


namespace NUMINAMATH_GPT_largest_reciprocal_l1531_153130

-- Definitions for the given numbers
def a := 1/4
def b := 3/7
def c := 2
def d := 10
def e := 2023

-- Statement to prove the problem
theorem largest_reciprocal :
  (1/a) > (1/b) ∧ (1/a) > (1/c) ∧ (1/a) > (1/d) ∧ (1/a) > (1/e) :=
by
  sorry

end NUMINAMATH_GPT_largest_reciprocal_l1531_153130


namespace NUMINAMATH_GPT_value_of_M_correct_l1531_153123

noncomputable def value_of_M : ℤ :=
  let d1 := 4        -- First column difference
  let d2 := -7       -- Row difference
  let d3 := 1        -- Second column difference
  let a1 := 25       -- First number in the row
  let a2 := 16 - d1  -- First number in the first column
  let a3 := a1 - d2 * 6  -- Last number in the row
  a3 + d3

theorem value_of_M_correct : value_of_M = -16 :=
  by
    let d1 := 4       -- First column difference
    let d2 := -7      -- Row difference
    let d3 := 1       -- Second column difference
    let a1 := 25      -- First number in the row
    let a2 := 16 - d1 -- First number in the first column
    let a3 := a1 - d2 * 6 -- Last number in the row
    have : a3 + d3 = -16
    · sorry
    exact this

end NUMINAMATH_GPT_value_of_M_correct_l1531_153123


namespace NUMINAMATH_GPT_eccentricity_squared_l1531_153173

-- Define the hyperbola and its properties
variables (a b c e : ℝ) (x₁ y₁ x₂ y₂ : ℝ)

-- Define the hyperbola equation and conditions
def hyperbola_eq (a b x y : ℝ) := (x^2)/(a^2) - (y^2)/(b^2) = 1

def midpoint_eq (x₁ y₁ x₂ y₂ : ℝ) := x₁ + x₂ = -4 ∧ y₁ + y₂ = 2

def slope_eq (a b c : ℝ) := -b / c = (b^2 * (-4)) / (a^2 * 2)

-- Define the proof
theorem eccentricity_squared :
  a > 0 → b > 0 → hyperbola_eq a b x₁ y₁ → hyperbola_eq a b x₂ y₂ → midpoint_eq x₁ y₁ x₂ y₂ →
  slope_eq a b c → c^2 = a^2 + b^2 → (e = c / a) → e^2 = (Real.sqrt 2 + 1) / 2 :=
by
  intro ha hb h1 h2 h3 h4 h5 he
  sorry

end NUMINAMATH_GPT_eccentricity_squared_l1531_153173


namespace NUMINAMATH_GPT_molecular_weight_CaSO4_2H2O_l1531_153141

def Ca := 40.08
def S := 32.07
def O := 16.00
def H := 1.008

def Ca_weight := 1 * Ca
def S_weight := 1 * S
def O_in_sulfate_weight := 4 * O
def O_in_water_weight := 4 * O
def H_in_water_weight := 4 * H

def total_weight := Ca_weight + S_weight + O_in_sulfate_weight + O_in_water_weight + H_in_water_weight

theorem molecular_weight_CaSO4_2H2O : total_weight = 204.182 := 
by {
  sorry
}

end NUMINAMATH_GPT_molecular_weight_CaSO4_2H2O_l1531_153141


namespace NUMINAMATH_GPT_smaller_angle_parallelogram_l1531_153135

theorem smaller_angle_parallelogram (x : ℕ) (h1 : ∀ a b : ℕ, a ≠ b ∧ a + b = 180) (h2 : ∃ y : ℕ, y = x + 70) : x = 55 :=
by
  sorry

end NUMINAMATH_GPT_smaller_angle_parallelogram_l1531_153135


namespace NUMINAMATH_GPT_trail_length_is_20_km_l1531_153157

-- Define the conditions and the question
def length_of_trail (L : ℝ) (hiked_percentage remaining_distance : ℝ) : Prop :=
  hiked_percentage = 0.60 ∧ remaining_distance = 8 ∧ 0.40 * L = remaining_distance

-- The statement: given the conditions, prove that length of trail is 20 km
theorem trail_length_is_20_km : ∃ L : ℝ, length_of_trail L 0.60 8 ∧ L = 20 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_trail_length_is_20_km_l1531_153157


namespace NUMINAMATH_GPT_solution_of_inequality_l1531_153137

theorem solution_of_inequality (a x : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  (x - a) * (x - a⁻¹) < 0 ↔ a < x ∧ x < a⁻¹ :=
by sorry

end NUMINAMATH_GPT_solution_of_inequality_l1531_153137


namespace NUMINAMATH_GPT_equivalent_single_discount_calculation_l1531_153121

-- Definitions for the successive discounts
def discount10 (x : ℝ) : ℝ := 0.90 * x
def discount15 (x : ℝ) : ℝ := 0.85 * x
def discount25 (x : ℝ) : ℝ := 0.75 * x

-- Final price after applying all discounts
def final_price (x : ℝ) : ℝ := discount25 (discount15 (discount10 x))

-- Equivalent single discount fraction
def equivalent_discount (x : ℝ) : ℝ := 0.57375 * x

theorem equivalent_single_discount_calculation (x : ℝ) : 
  final_price x = equivalent_discount x :=
sorry

end NUMINAMATH_GPT_equivalent_single_discount_calculation_l1531_153121


namespace NUMINAMATH_GPT_bus_speed_l1531_153108

theorem bus_speed (x y z : ℕ) (hx : x < 10) (hy : y < 10) (hz : z < 10)
    (h1 : 9 * (11 * y - x) = 5 * z)
    (h2 : z = 9) :
    ∀ speed, speed = 45 :=
by
  sorry

end NUMINAMATH_GPT_bus_speed_l1531_153108


namespace NUMINAMATH_GPT_original_price_l1531_153155

theorem original_price (saving : ℝ) (percentage : ℝ) (h_saving : saving = 10) (h_percentage : percentage = 0.10) :
  ∃ OP : ℝ, OP = 100 :=
by
  sorry

end NUMINAMATH_GPT_original_price_l1531_153155


namespace NUMINAMATH_GPT_total_votes_l1531_153197

theorem total_votes (P R : ℝ) (hP : P = 0.35) (diff : ℝ) (h_diff : diff = 1650) : 
  ∃ V : ℝ, P * V + (P * V + diff) = V ∧ V = 5500 :=
by
  use 5500
  sorry

end NUMINAMATH_GPT_total_votes_l1531_153197


namespace NUMINAMATH_GPT_totalHighlighters_l1531_153126

-- Define the number of each type of highlighter
def pinkHighlighters : ℕ := 10
def yellowHighlighters : ℕ := 15
def blueHighlighters : ℕ := 8

-- State the theorem to prove
theorem totalHighlighters :
  pinkHighlighters + yellowHighlighters + blueHighlighters = 33 :=
by
  -- Proof to be filled
  sorry

end NUMINAMATH_GPT_totalHighlighters_l1531_153126


namespace NUMINAMATH_GPT_proof_problem_l1531_153156

def work_problem :=
  ∃ (B : ℝ),
  (1 / 6) + (1 / B) + (1 / 24) = (1 / 3) ∧ B = 8

theorem proof_problem : work_problem :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1531_153156


namespace NUMINAMATH_GPT_exists_travel_route_l1531_153169

theorem exists_travel_route (n : ℕ) (cities : Finset ℕ) 
  (ticket_price : ℕ → ℕ → ℕ)
  (h1 : cities.card = n)
  (h2 : ∀ c1 c2, c1 ≠ c2 → ∃ p, (ticket_price c1 c2 = p ∧ ticket_price c1 c2 = ticket_price c2 c1))
  (h3 : ∀ p1 p2 c1 c2 c3 c4,
    p1 ≠ p2 ∧ (ticket_price c1 c2 = p1) ∧ (ticket_price c3 c4 = p2) →
    p1 ≠ p2) :
  ∃ city : ℕ, ∀ m : ℕ, m = n - 1 →
  ∃ route : Finset (ℕ × ℕ),
  route.card = m ∧
  ∀ (t₁ t₂ : ℕ × ℕ), t₁ ∈ route → t₂ ∈ route → (t₁ ≠ t₂ → ticket_price t₁.1 t₁.2 < ticket_price t₂.1 t₂.2) :=
by
  sorry

end NUMINAMATH_GPT_exists_travel_route_l1531_153169


namespace NUMINAMATH_GPT_non_degenerate_ellipse_l1531_153162

theorem non_degenerate_ellipse (k : ℝ) : (∃ a, a = -21) ↔ (k > -21) := by
  sorry

end NUMINAMATH_GPT_non_degenerate_ellipse_l1531_153162


namespace NUMINAMATH_GPT_cubic_inequality_l1531_153107

theorem cubic_inequality (x : ℝ) : x^3 - 4*x^2 + 4*x < 0 ↔ x < 0 :=
by
  sorry

end NUMINAMATH_GPT_cubic_inequality_l1531_153107


namespace NUMINAMATH_GPT_correct_transformation_l1531_153164

-- Given transformations
def transformation_A (a : ℝ) : Prop := - (1 / a) = -1 / a
def transformation_B (a b : ℝ) : Prop := (1 / a) + (1 / b) = 1 / (a + b)
def transformation_C (a b : ℝ) : Prop := (2 * b^2) / a^2 = (2 * b) / a
def transformation_D (a b : ℝ) : Prop := (a + a * b) / (b + a * b) = a / b

-- Correct transformation is A.
theorem correct_transformation (a b : ℝ) : transformation_A a ∧ ¬transformation_B a b ∧ ¬transformation_C a b ∧ ¬transformation_D a b :=
sorry

end NUMINAMATH_GPT_correct_transformation_l1531_153164


namespace NUMINAMATH_GPT_sum_of_digits_l1531_153190

noncomputable def A : ℕ := 3
noncomputable def B : ℕ := 9
noncomputable def C : ℕ := 2
noncomputable def BC : ℕ := B * 10 + C
noncomputable def ABC : ℕ := A * 100 + B * 10 + C

theorem sum_of_digits (H1: A ≠ 0) (H2: B ≠ 0) (H3: C ≠ 0) (H4: BC + ABC + ABC = 876):
  A + B + C = 14 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_l1531_153190


namespace NUMINAMATH_GPT_crow_distance_l1531_153133

theorem crow_distance (trips: ℕ) (hours: ℝ) (speed: ℝ) (distance: ℝ) :
  trips = 15 → hours = 1.5 → speed = 4 → (trips * 2 * distance) = (speed * hours) → distance = 200 / 1000 :=
by
  intros h_trips h_hours h_speed h_eq
  sorry

end NUMINAMATH_GPT_crow_distance_l1531_153133


namespace NUMINAMATH_GPT_cubic_km_to_cubic_m_l1531_153106

theorem cubic_km_to_cubic_m (km_to_m : 1 = 1000) : (1 : ℝ) ^ 3 = (1000 : ℝ) ^ 3 :=
by sorry

end NUMINAMATH_GPT_cubic_km_to_cubic_m_l1531_153106


namespace NUMINAMATH_GPT_length_of_NC_l1531_153178

noncomputable def semicircle_radius (AB : ℝ) : ℝ := AB / 2

theorem length_of_NC : 
  ∀ (AB CD AN NB N M C NC : ℝ),
    AB = 10 ∧ AB = CD ∧ AN = NB ∧ AN + NB = AB ∧ M = N ∧ AB / 2 = semicircle_radius AB ∧ (NC^2 + semicircle_radius AB^2 = (2 * semicircle_radius AB)^2) →
    NC = 5 * Real.sqrt 3 := 
by 
  intros AB CD AN NB N M C NC h 
  rcases h with ⟨hAB, hCD, hAN, hSumAN, hMN, hRadius, hPythag⟩
  sorry

end NUMINAMATH_GPT_length_of_NC_l1531_153178


namespace NUMINAMATH_GPT_determine_number_l1531_153194

noncomputable def is_valid_number (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000 ∧
  (∃ d1 d2 d3, 
    n = d1 * 100 + d2 * 10 + d3 ∧ 
    (
      (d1 = 5 ∨ d1 = 1 ∨ d1 = 5 ∨ d1 = 2) ∧
      (d2 = 4 ∨ d2 = 4 ∨ d2 = 4) ∧
      (d3 = 3 ∨ d3 = 2 ∨ d3 = 6)
    ) ∧
    (
      (d1 ≠ 1 ∧ d1 ≠ 2 ∧ d1 ≠ 6) ∧
      (d2 ≠ 5 ∧ d2 ≠ 4 ∧ d2 ≠ 6 ∧ d2 ≠ 2) ∧
      (d3 ≠ 5 ∧ d3 ≠ 4 ∧ d3 ≠ 1 ∧ d3 ≠ 2)
    )
  )

theorem determine_number : ∃ n : ℕ, is_valid_number n ∧ n = 163 :=
by 
  existsi 163
  unfold is_valid_number
  sorry

end NUMINAMATH_GPT_determine_number_l1531_153194


namespace NUMINAMATH_GPT_trigonometric_inequality_l1531_153132

theorem trigonometric_inequality (x : ℝ) : 0 ≤ 5 + 8 * Real.cos x + 4 * Real.cos (2 * x) + Real.cos (3 * x) ∧ 
                                            5 + 8 * Real.cos x + 4 * Real.cos (2 * x) + Real.cos (3 * x) ≤ 18 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_inequality_l1531_153132


namespace NUMINAMATH_GPT_ac_le_bc_if_a_gt_b_and_c_le_zero_a_sq_gt_b_sq_if_ac_sq_gt_bc_sq_and_b_ge_zero_log_a1_gt_log_b1_if_a_gt_b_and_b_gt_neg1_inv_a_lt_inv_b_if_a_gt_b_and_ab_gt_zero_l1531_153184

section

variable {a b c : ℝ}

-- Statement 1
theorem ac_le_bc_if_a_gt_b_and_c_le_zero (h1 : a > b) (h2 : c ≤ 0) : a * c ≤ b * c := 
  sorry

-- Statement 2
theorem a_sq_gt_b_sq_if_ac_sq_gt_bc_sq_and_b_ge_zero (h1 : a * c ^ 2 > b * c ^ 2) (h2 : b ≥ 0) : a ^ 2 > b ^ 2 := 
  sorry

-- Statement 3
theorem log_a1_gt_log_b1_if_a_gt_b_and_b_gt_neg1 (h1 : a > b) (h2 : b > -1) : Real.log (a + 1) > Real.log (b + 1) := 
  sorry

-- Statement 4
theorem inv_a_lt_inv_b_if_a_gt_b_and_ab_gt_zero (h1 : a > b) (h2 : a * b > 0) : 1 / a < 1 / b := 
  sorry

end

end NUMINAMATH_GPT_ac_le_bc_if_a_gt_b_and_c_le_zero_a_sq_gt_b_sq_if_ac_sq_gt_bc_sq_and_b_ge_zero_log_a1_gt_log_b1_if_a_gt_b_and_b_gt_neg1_inv_a_lt_inv_b_if_a_gt_b_and_ab_gt_zero_l1531_153184


namespace NUMINAMATH_GPT_price_of_magic_card_deck_l1531_153175

def initial_decks : ℕ := 5
def remaining_decks : ℕ := 3
def total_earnings : ℕ := 4
def decks_sold := initial_decks - remaining_decks
def price_per_deck := total_earnings / decks_sold

theorem price_of_magic_card_deck : price_per_deck = 2 := by
  sorry

end NUMINAMATH_GPT_price_of_magic_card_deck_l1531_153175


namespace NUMINAMATH_GPT_factor_expression_l1531_153151

theorem factor_expression (x y z : ℝ) : 
  ( (x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3 ) / 
  ( (x - y)^3 + (y - z)^3 + (z - x)^3 ) = 
  (x + y) * (y + z) * (z + x) := 
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1531_153151


namespace NUMINAMATH_GPT_loaned_books_during_month_l1531_153117

-- Definitions corresponding to the conditions
def initial_books : ℕ := 75
def returned_percent : ℚ := 0.65
def end_books : ℕ := 68

-- Proof statement
theorem loaned_books_during_month (x : ℕ) 
  (h1 : returned_percent = 0.65)
  (h2 : initial_books = 75)
  (h3 : end_books = 68) :
  (0.35 * x : ℚ) = (initial_books - end_books) :=
sorry

end NUMINAMATH_GPT_loaned_books_during_month_l1531_153117
