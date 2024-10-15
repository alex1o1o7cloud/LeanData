import Mathlib

namespace NUMINAMATH_GPT_inscribed_circle_equals_arc_length_l1187_118782

open Real

theorem inscribed_circle_equals_arc_length 
  (R : ℝ) 
  (hR : 0 < R) 
  (θ : ℝ)
  (hθ : θ = (2 * π) / 3)
  (r : ℝ)
  (h_r : r = R / 2) 
  : 2 * π * r = 2 * π * R * (θ / (2 * π)) := by
  sorry

end NUMINAMATH_GPT_inscribed_circle_equals_arc_length_l1187_118782


namespace NUMINAMATH_GPT_pies_from_36_apples_l1187_118723

-- Definitions of conditions
def pies_from_apples (apples : Nat) : Nat :=
  apples / 4  -- because 12 apples = 3 pies implies 1 pie = 4 apples

-- Theorem to prove
theorem pies_from_36_apples : pies_from_apples 36 = 9 := by
  sorry

end NUMINAMATH_GPT_pies_from_36_apples_l1187_118723


namespace NUMINAMATH_GPT_sum_reciprocal_of_shifted_roots_l1187_118735

noncomputable def roots_of_cubic (a b c : ℝ) : Prop := 
    ∀ x : ℝ, x^3 - x - 2 = (x - a) * (x - b) * (x - c)

theorem sum_reciprocal_of_shifted_roots (a b c : ℝ) 
    (h : roots_of_cubic a b c) : 
    (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_reciprocal_of_shifted_roots_l1187_118735


namespace NUMINAMATH_GPT_line_within_plane_correct_l1187_118729

-- Definitions of sets representing a line and a plane
variable {Point : Type}
variable (l α : Set Point)

-- Definition of the statement
def line_within_plane : Prop := l ⊆ α

-- Proof statement (without the actual proof)
theorem line_within_plane_correct (h : l ⊆ α) : line_within_plane l α :=
by
  sorry

end NUMINAMATH_GPT_line_within_plane_correct_l1187_118729


namespace NUMINAMATH_GPT_gcd_power_of_two_sub_one_l1187_118766

def a : ℤ := 2^1100 - 1
def b : ℤ := 2^1122 - 1
def c : ℤ := 2^22 - 1

theorem gcd_power_of_two_sub_one :
  Int.gcd (2^1100 - 1) (2^1122 - 1) = 2^22 - 1 := by
  sorry

end NUMINAMATH_GPT_gcd_power_of_two_sub_one_l1187_118766


namespace NUMINAMATH_GPT_trigonometric_identity_l1187_118759

theorem trigonometric_identity : 
  4 * Real.cos (50 * Real.pi / 180) - Real.tan (40 * Real.pi / 180) = Real.sqrt 3 :=
by
  -- Here we assume standard trigonometric identities and basic properties already handled by Mathlib
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1187_118759


namespace NUMINAMATH_GPT_vector_subtraction_correct_l1187_118793

def vector_a : ℝ × ℝ := (2, -1)
def vector_b : ℝ × ℝ := (-4, 2)

theorem vector_subtraction_correct :
  vector_a - 2 • vector_b = (10, -5) :=
sorry

end NUMINAMATH_GPT_vector_subtraction_correct_l1187_118793


namespace NUMINAMATH_GPT_unique_solution_f_l1187_118751

theorem unique_solution_f (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, f (x + f y) ≥ f (f x + y))
  (h2 : f 0 = 0) :
  ∀ x : ℝ, f x = x :=
sorry

end NUMINAMATH_GPT_unique_solution_f_l1187_118751


namespace NUMINAMATH_GPT_certain_number_calculation_l1187_118710

theorem certain_number_calculation (x : ℝ) (h : (15 * x) / 100 = 0.04863) : x = 0.3242 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_calculation_l1187_118710


namespace NUMINAMATH_GPT_evaluate_expression_l1187_118795

theorem evaluate_expression 
  (d a b c : ℚ)
  (h1 : d = a + 1)
  (h2 : a = b - 3)
  (h3 : b = c + 5)
  (h4 : c = 6)
  (nz1 : d + 3 ≠ 0)
  (nz2 : a + 2 ≠ 0)
  (nz3 : b - 5 ≠ 0)
  (nz4 : c + 7 ≠ 0) :
  (d + 5) / (d + 3) * (a + 3) / (a + 2) * (b - 3) / (b - 5) * (c + 10) / (c + 7) = 1232 / 585 :=
sorry

end NUMINAMATH_GPT_evaluate_expression_l1187_118795


namespace NUMINAMATH_GPT_circle_tangency_problem_l1187_118796

theorem circle_tangency_problem :
  let u1 := ∀ (x y : ℝ), x^2 + y^2 + 8 * x - 30 * y - 63 = 0
  let u2 := ∀ (x y : ℝ), x^2 + y^2 - 6 * x - 30 * y + 99 = 0
  let line := ∀ (b x : ℝ), y = b * x
  ∃ p q : ℕ, gcd p q = 1 ∧ n^2 = (p : ℚ) / (q : ℚ) ∧ p + q = 7 :=
sorry

end NUMINAMATH_GPT_circle_tangency_problem_l1187_118796


namespace NUMINAMATH_GPT_lines_intersect_l1187_118733

def line1 (t : ℝ) : ℝ × ℝ :=
  (1 - 2 * t, 2 + 4 * t)

def line2 (u : ℝ) : ℝ × ℝ :=
  (3 + u, 5 + 3 * u)

theorem lines_intersect :
  ∃ t u : ℝ, line1 t = line2 u ∧ line1 t = (1.2, 1.6) :=
by
  sorry

end NUMINAMATH_GPT_lines_intersect_l1187_118733


namespace NUMINAMATH_GPT_passes_after_6_l1187_118709

-- Define the sequence a_n where a_n represents the number of ways the ball is in A's hands after n passes
def passes : ℕ → ℕ
| 0       => 1       -- Initially, the ball is in A's hands (1 way)
| (n + 1) => 2^n - passes n

-- Theorem to prove the number of different passing methods after 6 passes
theorem passes_after_6 : passes 6 = 22 := by
  sorry

end NUMINAMATH_GPT_passes_after_6_l1187_118709


namespace NUMINAMATH_GPT_problem_expression_l1187_118780

theorem problem_expression (x y : ℝ) (h1 : x - y = 5) (h2 : x * y = 4) : x^2 + y^2 = 33 :=
by sorry

end NUMINAMATH_GPT_problem_expression_l1187_118780


namespace NUMINAMATH_GPT_isabella_jumped_farthest_l1187_118743

-- defining the jumping distances
def ricciana_jump : ℕ := 4
def margarita_jump : ℕ := 2 * ricciana_jump - 1
def isabella_jump : ℕ := ricciana_jump + 3 

-- defining the total distances
def ricciana_total : ℕ := 20 + ricciana_jump
def margarita_total : ℕ := 18 + margarita_jump
def isabella_total : ℕ := 22 + isabella_jump

-- stating the theorem
theorem isabella_jumped_farthest : isabella_total = 29 :=
by sorry

end NUMINAMATH_GPT_isabella_jumped_farthest_l1187_118743


namespace NUMINAMATH_GPT_find_maximum_value_l1187_118701

open Real

noncomputable def maximum_value (a b c : ℝ) (h₁ : a + b + c = 1) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) : ℝ :=
  2 + sqrt 5

theorem find_maximum_value (a b c : ℝ) (h₁ : a + b + c = 1) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) :
  sqrt (4 * a + 1) + sqrt (4 * b + 1) + sqrt (4 * c + 1) > maximum_value a b c h₁ h₂ h₃ h₄ :=
by
  sorry

end NUMINAMATH_GPT_find_maximum_value_l1187_118701


namespace NUMINAMATH_GPT_find_f_six_minus_a_l1187_118785

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then 2^(x-2) - 2 else -Real.logb 2 (x + 1)

variable (a : ℝ)
axiom h : f a = -3

theorem find_f_six_minus_a : f (6 - a) = - 15 / 8 :=
by
  sorry

end NUMINAMATH_GPT_find_f_six_minus_a_l1187_118785


namespace NUMINAMATH_GPT_g_at_3_l1187_118719

def g (x : ℝ) : ℝ := 7 * x^3 - 5 * x^2 + 3 * x - 6

theorem g_at_3 : g 3 = 147 :=
by
  -- Proof omitted for brevity
  sorry

end NUMINAMATH_GPT_g_at_3_l1187_118719


namespace NUMINAMATH_GPT_valid_k_sum_correct_l1187_118791

def sum_of_valid_k : ℤ :=
  (List.range 17).sum * 1734 + (List.range 17).sum * 3332

theorem valid_k_sum_correct : sum_of_valid_k = 5066 := by
  sorry

end NUMINAMATH_GPT_valid_k_sum_correct_l1187_118791


namespace NUMINAMATH_GPT_cannot_be_covered_by_dominoes_l1187_118786

-- Definitions for each board
def board_3x4_squares : ℕ := 3 * 4
def board_3x5_squares : ℕ := 3 * 5
def board_4x4_one_removed_squares : ℕ := 4 * 4 - 1
def board_5x5_squares : ℕ := 5 * 5
def board_6x3_squares : ℕ := 6 * 3

-- Parity check
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Mathematical proof problem statement
theorem cannot_be_covered_by_dominoes :
  ¬ is_even board_3x5_squares ∧
  ¬ is_even board_4x4_one_removed_squares ∧
  ¬ is_even board_5x5_squares :=
by
  -- Checking the conditions that must hold
  sorry

end NUMINAMATH_GPT_cannot_be_covered_by_dominoes_l1187_118786


namespace NUMINAMATH_GPT_side_length_of_cube_l1187_118761

theorem side_length_of_cube (s : ℝ) (h : s^3 = 6 * s^2) : s = 6 :=
by
  sorry

end NUMINAMATH_GPT_side_length_of_cube_l1187_118761


namespace NUMINAMATH_GPT_total_pictures_painted_l1187_118711

def pictures_painted_in_june : ℕ := 2
def pictures_painted_in_july : ℕ := 2
def pictures_painted_in_august : ℕ := 9

theorem total_pictures_painted : 
  pictures_painted_in_june + pictures_painted_in_july + pictures_painted_in_august = 13 :=
by
  sorry

end NUMINAMATH_GPT_total_pictures_painted_l1187_118711


namespace NUMINAMATH_GPT_younger_son_age_after_30_years_l1187_118798

-- Definitions based on given conditions
def age_difference : Nat := 10
def elder_son_current_age : Nat := 40

-- We need to prove that given these conditions, the younger son will be 60 years old 30 years from now
theorem younger_son_age_after_30_years : (elder_son_current_age - age_difference) + 30 = 60 := by
  -- Proof should go here, but we will skip it as per the instructions
  sorry

end NUMINAMATH_GPT_younger_son_age_after_30_years_l1187_118798


namespace NUMINAMATH_GPT_find_c_for_equal_real_roots_l1187_118752

theorem find_c_for_equal_real_roots
  (c : ℝ)
  (h : ∀ x : ℝ, x^2 + 6 * x + c = 0 → x = -3) : c = 9 :=
sorry

end NUMINAMATH_GPT_find_c_for_equal_real_roots_l1187_118752


namespace NUMINAMATH_GPT_prove_inequality_l1187_118748

-- Defining properties of f
variable {α : Type*} [LinearOrderedField α] (f : α → α)

-- Condition 1: f is even function
def is_even_function (f : α → α) : Prop := ∀ x : α, f (-x) = f x

-- Condition 2: f is monotonically increasing on (0, ∞)
def is_monotonically_increasing_on_positive (f : α → α) : Prop := ∀ ⦃x y : α⦄, 0 < x → 0 < y → x < y → f x < f y

-- Define the main theorem we need to prove:
theorem prove_inequality (h1 : is_even_function f) (h2 : is_monotonically_increasing_on_positive f) : 
  f (-1) < f 2 ∧ f 2 < f (-3) :=
by
  sorry

end NUMINAMATH_GPT_prove_inequality_l1187_118748


namespace NUMINAMATH_GPT_value_of_x_l1187_118706

variable (w x y : ℝ)

theorem value_of_x 
  (h_avg : (w + x) / 2 = 0.5)
  (h_eq : (7 / w) + (7 / x) = 7 / y)
  (h_prod : w * x = y) :
  x = 0.5 :=
sorry

end NUMINAMATH_GPT_value_of_x_l1187_118706


namespace NUMINAMATH_GPT_parallel_lines_slope_l1187_118765

theorem parallel_lines_slope (m : ℝ) :
  ((m + 2) * (2 * m - 1) = 3 * 1) →
  m = - (5 / 2) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_l1187_118765


namespace NUMINAMATH_GPT_diagonal_ratio_l1187_118758

variable (a b : ℝ)
variable (d1 : ℝ) -- diagonal length of the first square
variable (r : ℝ := 1.5) -- ratio between perimeters

theorem diagonal_ratio (h : 4 * a / (4 * b) = r) (hd1 : d1 = a * Real.sqrt 2) : 
  (b * Real.sqrt 2) = (2/3) * d1 := 
sorry

end NUMINAMATH_GPT_diagonal_ratio_l1187_118758


namespace NUMINAMATH_GPT_find_n_l1187_118714

theorem find_n (n : ℕ) (a_n D_n d_n : ℕ) (h1 : n > 5) (h2 : D_n - d_n = a_n) : n = 9 := 
by 
  sorry

end NUMINAMATH_GPT_find_n_l1187_118714


namespace NUMINAMATH_GPT_complex_quadrant_l1187_118734

theorem complex_quadrant (z : ℂ) (h : z * (2 - I) = 2 + I) : 0 < z.re ∧ 0 < z.im := 
sorry

end NUMINAMATH_GPT_complex_quadrant_l1187_118734


namespace NUMINAMATH_GPT_circle_diameter_l1187_118740

theorem circle_diameter (r d : ℝ) (h₀ : ∀ (r : ℝ), ∃ (d : ℝ), d = 2 * r) (h₁ : π * r^2 = 9 * π) :
  d = 6 :=
by
  rcases h₀ r with ⟨d, hd⟩
  sorry

end NUMINAMATH_GPT_circle_diameter_l1187_118740


namespace NUMINAMATH_GPT_correct_transformation_D_l1187_118774

theorem correct_transformation_D : ∀ x, 2 * (x + 1) = x + 7 → x = 5 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_correct_transformation_D_l1187_118774


namespace NUMINAMATH_GPT_cubicroots_expression_l1187_118756

theorem cubicroots_expression (a b c : ℝ)
  (h₁ : a + b + c = 6)
  (h₂ : a * b + b * c + c * a = 11)
  (h₃ : a * b * c = 6) :
  1 / a^3 + 1 / b^3 + 1 / c^3 = 251 / 216 :=
by sorry

end NUMINAMATH_GPT_cubicroots_expression_l1187_118756


namespace NUMINAMATH_GPT_price_increase_percentage_l1187_118781

theorem price_increase_percentage (original_price : ℝ) (discount : ℝ) (reduced_price : ℝ) : 
  reduced_price = original_price * (1 - discount) →
  (original_price / reduced_price - 1) * 100 = 8.7 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_price_increase_percentage_l1187_118781


namespace NUMINAMATH_GPT_cone_lateral_surface_area_l1187_118727

theorem cone_lateral_surface_area (r h l S : ℝ) (π_pos : 0 < π) (r_eq : r = 6)
  (V : ℝ) (V_eq : V = 30 * π)
  (vol_eq : V = (1/3) * π * r^2 * h)
  (h_eq : h = 5 / 2)
  (l_eq : l = Real.sqrt (r^2 + h^2))
  (S_eq : S = π * r * l) :
  S = 39 * π :=
  sorry

end NUMINAMATH_GPT_cone_lateral_surface_area_l1187_118727


namespace NUMINAMATH_GPT_rod_length_l1187_118717

/--
Prove that given the number of pieces that can be cut from the rod is 40 and the length of each piece is 85 cm, the length of the rod is 3400 cm.
-/
theorem rod_length (number_of_pieces : ℕ) (length_of_each_piece : ℕ) (h_pieces : number_of_pieces = 40) (h_length_piece : length_of_each_piece = 85) : number_of_pieces * length_of_each_piece = 3400 := 
by
  -- We need to prove that 40 * 85 = 3400
  sorry

end NUMINAMATH_GPT_rod_length_l1187_118717


namespace NUMINAMATH_GPT_relationship_between_xyz_l1187_118741

theorem relationship_between_xyz (x y z : ℝ) (h1 : x - z < y) (h2 : x + z > y) : -z < x - y ∧ x - y < z :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_xyz_l1187_118741


namespace NUMINAMATH_GPT_find_p_q_l1187_118776

def op (a b c d : ℝ) : ℝ × ℝ := (a * c - b * d, a * d + b * c)

theorem find_p_q :
  (∀ (a b c d : ℝ), (a = c ∧ b = d) ↔ (a, b) = (c, d)) →
  (op 1 2 p q = (5, 0)) →
  (p, q) = (1, -2) :=
by
  intro h
  intro eq_op
  sorry

end NUMINAMATH_GPT_find_p_q_l1187_118776


namespace NUMINAMATH_GPT_total_cost_of_fencing_l1187_118787

def costOfFencing (lengths rates : List ℝ) : ℝ :=
  List.sum (List.zipWith (· * ·) lengths rates)

theorem total_cost_of_fencing :
  costOfFencing [14, 20, 35, 40, 15, 30, 25]
                [2.50, 3.00, 3.50, 4.00, 2.75, 3.25, 3.75] = 610.00 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_fencing_l1187_118787


namespace NUMINAMATH_GPT_touchdowns_points_l1187_118792

theorem touchdowns_points 
    (num_touchdowns : ℕ) (total_points : ℕ) 
    (h1 : num_touchdowns = 3) 
    (h2 : total_points = 21) : 
    total_points / num_touchdowns = 7 :=
by
    sorry

end NUMINAMATH_GPT_touchdowns_points_l1187_118792


namespace NUMINAMATH_GPT_math_problem_proof_l1187_118745

noncomputable def problem_expr : ℚ :=
  ((11 + 1/9) - (3 + 2/5) * (1 + 2/17)) - (8 + 2/5) / (36/10) / (2 + 6/25)

theorem math_problem_proof : problem_expr = 20 / 9 := by
  sorry

end NUMINAMATH_GPT_math_problem_proof_l1187_118745


namespace NUMINAMATH_GPT_find_rate_of_current_l1187_118731

-- Parameters and definitions
variables (r w : Real)

-- Conditions of the problem
def original_journey := 3 * r^2 - 23 * w^2 = 0
def modified_journey := 6 * r^2 - 2 * w^2 + 40 * w = 0

-- Main theorem to prove
theorem find_rate_of_current (h1 : original_journey r w) (h2 : modified_journey r w) :
  w = 10 / 11 :=
sorry

end NUMINAMATH_GPT_find_rate_of_current_l1187_118731


namespace NUMINAMATH_GPT_poly_divisible_coeff_sum_eq_one_l1187_118754

theorem poly_divisible_coeff_sum_eq_one (C D : ℂ) :
  (∀ x : ℂ, x^2 + x + 1 = 0 → x^100 + C * x^2 + D * x + 1 = 0) →
  C + D = 1 :=
by
  sorry

end NUMINAMATH_GPT_poly_divisible_coeff_sum_eq_one_l1187_118754


namespace NUMINAMATH_GPT_repeating_decimal_transform_l1187_118712

theorem repeating_decimal_transform (n : ℕ) (s : String) (k : ℕ) (m : ℕ)
  (original : s = "2345678") (len : k = 7) (position : n = 2011)
  (effective_position : m = n - 1) (mod_position : m % k = 3) :
  "0.1" ++ s = "0.12345678" :=
sorry

end NUMINAMATH_GPT_repeating_decimal_transform_l1187_118712


namespace NUMINAMATH_GPT_complex_number_division_l1187_118788

theorem complex_number_division (i : ℂ) (h_i : i^2 = -1) :
  2 / (i * (3 - i)) = (1 - 3 * i) / 5 :=
by
  sorry

end NUMINAMATH_GPT_complex_number_division_l1187_118788


namespace NUMINAMATH_GPT_exponent_value_l1187_118771

theorem exponent_value (y k : ℕ) (h1 : 9^y = 3^k) (h2 : y = 7) : k = 14 := by
  sorry

end NUMINAMATH_GPT_exponent_value_l1187_118771


namespace NUMINAMATH_GPT_min_value_geometric_sequence_l1187_118784

noncomputable def geometric_min_value (b1 b2 b3 : ℝ) (s : ℝ) : ℝ :=
  3 * b2 + 4 * b3

theorem min_value_geometric_sequence (s : ℝ) :
  ∃ s : ℝ, 2 = b1 ∧ b2 = 2 * s ∧ b3 = 2 * s^2 ∧ 3 * b2 + 4 * b3 = -9 / 8 :=
by
  sorry

end NUMINAMATH_GPT_min_value_geometric_sequence_l1187_118784


namespace NUMINAMATH_GPT_p_at_zero_l1187_118764

-- We state the conditions: p is a polynomial of degree 6, and p(3^n) = 1/(3^n) for n = 0 to 6
def p : Polynomial ℝ := sorry

axiom p_degree : p.degree = 6
axiom p_values : ∀ (n : ℕ), n ≤ 6 → p.eval (3^n) = 1 / (3^n)

-- We want to prove that p(0) = 29523 / 2187
theorem p_at_zero : p.eval 0 = 29523 / 2187 := by sorry

end NUMINAMATH_GPT_p_at_zero_l1187_118764


namespace NUMINAMATH_GPT_fraction_identity_l1187_118700

variable {a b x : ℝ}

-- Conditions
axiom h1 : x = a / b
axiom h2 : a ≠ b
axiom h3 : b ≠ 0

-- Question to prove
theorem fraction_identity :
  (a + b) / (a - b) = (x + 1) / (x - 1) :=
by
  sorry

end NUMINAMATH_GPT_fraction_identity_l1187_118700


namespace NUMINAMATH_GPT_expand_product_l1187_118722

-- We need to state the problem as a theorem
theorem expand_product (y : ℝ) (hy : y ≠ 0) : (3 / 7) * (7 / y + 14 * y^3) = 3 / y + 6 * y^3 :=
by
  sorry -- Skipping the proof

end NUMINAMATH_GPT_expand_product_l1187_118722


namespace NUMINAMATH_GPT_sum_is_2000_l1187_118763

theorem sum_is_2000 (x y : ℝ) (h : x ≠ y) (h_eq : x^2 - 2000 * x = y^2 - 2000 * y) : x + y = 2000 := by
  sorry

end NUMINAMATH_GPT_sum_is_2000_l1187_118763


namespace NUMINAMATH_GPT_area_triangle_PQR_eq_2sqrt2_l1187_118738

noncomputable def areaOfTrianglePQR : ℝ :=
  let sideAB := 3
  let altitudeAE := 6
  let EB := Real.sqrt (sideAB^2 + altitudeAE^2)
  let ED := EB
  let EC := Real.sqrt ((sideAB * Real.sqrt 2)^2 + altitudeAE^2)
  let EP := (2 / 3) * EB
  let EQ := EP
  let ER := (1 / 3) * EC
  let PR := Real.sqrt (ER^2 + EP^2 - 2 * ER * EP * (EB^2 + EC^2 - sideAB^2) / (2 * EB * EC))
  let PQ := 2
  let RS := Real.sqrt (PR^2 - (PQ / 2)^2)
  (1 / 2) * PQ * RS

theorem area_triangle_PQR_eq_2sqrt2 : areaOfTrianglePQR = 2 * Real.sqrt 2 :=
  sorry

end NUMINAMATH_GPT_area_triangle_PQR_eq_2sqrt2_l1187_118738


namespace NUMINAMATH_GPT_danica_planes_l1187_118742

def smallestAdditionalPlanes (n k : ℕ) : ℕ :=
  let m := k * (n / k + 1)
  m - n

theorem danica_planes : smallestAdditionalPlanes 17 7 = 4 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_danica_planes_l1187_118742


namespace NUMINAMATH_GPT_arun_deepak_age_ratio_l1187_118797

-- Define the current age of Arun based on the condition that after 6 years he will be 26 years old
def Arun_current_age : ℕ := 26 - 6

-- Define Deepak's current age based on the given condition
def Deepak_current_age : ℕ := 15

-- The present ratio between Arun's age and Deepak's age
theorem arun_deepak_age_ratio : Arun_current_age / Nat.gcd Arun_current_age Deepak_current_age = (4 : ℕ) ∧ Deepak_current_age / Nat.gcd Arun_current_age Deepak_current_age = (3 : ℕ) := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_arun_deepak_age_ratio_l1187_118797


namespace NUMINAMATH_GPT_trigonometric_identity_l1187_118762

theorem trigonometric_identity :
  (3 / (Real.sin (20 * Real.pi / 180))^2) - (1 / (Real.cos (20 * Real.pi / 180))^2) + 64 * (Real.sin (20 * Real.pi / 180))^2 = 32 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1187_118762


namespace NUMINAMATH_GPT_sacks_per_day_proof_l1187_118757

-- Definitions based on the conditions in the problem
def totalUnripeOranges : ℕ := 1080
def daysOfHarvest : ℕ := 45

-- Mathematical statement to prove
theorem sacks_per_day_proof : totalUnripeOranges / daysOfHarvest = 24 :=
by sorry

end NUMINAMATH_GPT_sacks_per_day_proof_l1187_118757


namespace NUMINAMATH_GPT_ratio_of_jumps_l1187_118726

theorem ratio_of_jumps (run_ric: ℕ) (jump_ric: ℕ) (run_mar: ℕ) (extra_dist: ℕ)
    (h1 : run_ric = 20)
    (h2 : jump_ric = 4)
    (h3 : run_mar = 18)
    (h4 : extra_dist = 1) :
    (run_mar + extra_dist - run_ric - jump_ric) / jump_ric = 7 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_jumps_l1187_118726


namespace NUMINAMATH_GPT_complex_number_fourth_quadrant_l1187_118728

theorem complex_number_fourth_quadrant (m : ℝ) (h1 : 2/3 < m) (h2 : m < 1) : 
  (3 * m - 2) > 0 ∧ (m - 1) < 0 := 
by 
  sorry

end NUMINAMATH_GPT_complex_number_fourth_quadrant_l1187_118728


namespace NUMINAMATH_GPT_infinite_series_sum_eq_seven_l1187_118768

noncomputable def infinite_series_sum : ℝ :=
  ∑' k : ℕ, (1 + k)^2 / 3^(1 + k)

theorem infinite_series_sum_eq_seven : infinite_series_sum = 7 :=
sorry

end NUMINAMATH_GPT_infinite_series_sum_eq_seven_l1187_118768


namespace NUMINAMATH_GPT_right_triangle_perimeter_area_ratio_l1187_118702

theorem right_triangle_perimeter_area_ratio 
  (a b : ℝ) (h : a > 0 ∧ b > 0) 
  (hyp : ∀ c, c = Real.sqrt (a^2 + b^2))
  : (a + b + Real.sqrt (a^2 + b^2)) / (0.5 * a * b) = 5 → (∃! x y : ℝ, x + y + Real.sqrt (x^2 + y^2) / (0.5 * x * y) = 5) :=
by
  sorry   -- Proof is omitted as per instructions.

end NUMINAMATH_GPT_right_triangle_perimeter_area_ratio_l1187_118702


namespace NUMINAMATH_GPT_find_angle_A_find_minimum_bc_l1187_118790

open Real

variables (A B C a b c : ℝ)

-- Conditions
def side_opposite_angles_condition : Prop :=
  A > 0 ∧ A < π ∧ (A + B + C) = π

def collinear_vectors_condition (B C : ℝ) : Prop :=
  ∃ (k : ℝ), (2 * cos B * cos C + 1, 2 * sin B) = k • (sin C, 1)

-- Questions translated to proof statements
theorem find_angle_A (h1 : side_opposite_angles_condition A B C) (h2 : collinear_vectors_condition B C) :
  A = π / 3 :=
sorry

theorem find_minimum_bc (h1 : side_opposite_angles_condition A B C) (h2 : collinear_vectors_condition B C)
  (h3 : (1 / 2) * b * c * sin A = sqrt 3) :
  b + c = 4 :=
sorry

end NUMINAMATH_GPT_find_angle_A_find_minimum_bc_l1187_118790


namespace NUMINAMATH_GPT_problem_A_minus_B_no_x_or_x2_implies_a2_plus_b2_eq_13_l1187_118750

variable (x y a b : ℝ)

def A : ℝ := 2*x^2 + a*x - y + 6
def B : ℝ := b*x^2 - 3*x + 5*y - 1

theorem problem_A_minus_B_no_x_or_x2_implies_a2_plus_b2_eq_13 
  (h : A x y a - B x y b = -6*y + 7) : a^2 + b^2 = 13 := by
  sorry

end NUMINAMATH_GPT_problem_A_minus_B_no_x_or_x2_implies_a2_plus_b2_eq_13_l1187_118750


namespace NUMINAMATH_GPT_luke_fish_catching_l1187_118777

theorem luke_fish_catching :
  ∀ (days : ℕ) (fillets_per_fish : ℕ) (total_fillets : ℕ),
  days = 30 → fillets_per_fish = 2 → total_fillets = 120 →
  (total_fillets / fillets_per_fish) / days = 2 :=
by
  intros days fillets_per_fish total_fillets days_eq fillets_eq fillets_total_eq
  sorry

end NUMINAMATH_GPT_luke_fish_catching_l1187_118777


namespace NUMINAMATH_GPT_train_overtake_l1187_118704

theorem train_overtake :
  let speedA := 30 -- speed of Train A in miles per hour
  let speedB := 38 -- speed of Train B in miles per hour
  let lead_timeA := 2 -- lead time of Train A in hours
  let distanceA := speedA * lead_timeA -- distance traveled by Train A in the lead time
  let t := 7.5 -- time in hours Train B travels to catch up Train A
  let total_distanceB := speedB * t -- total distance traveled by Train B in time t
  total_distanceB = 285 := 
by
  sorry

end NUMINAMATH_GPT_train_overtake_l1187_118704


namespace NUMINAMATH_GPT_packet_weight_l1187_118789

theorem packet_weight
  (tons_to_pounds : ℕ := 2600) -- 1 ton = 2600 pounds
  (total_tons : ℕ := 13)       -- Total capacity in tons
  (num_packets : ℕ := 2080)    -- Number of packets
  (expected_weight_per_packet : ℚ := 16.25) : 
  total_tons * tons_to_pounds / num_packets = expected_weight_per_packet := 
sorry

end NUMINAMATH_GPT_packet_weight_l1187_118789


namespace NUMINAMATH_GPT_jim_travel_distance_l1187_118721

theorem jim_travel_distance :
  ∀ (John Jill Jim : ℝ),
  John = 15 →
  Jill = (John - 5) →
  Jim = (0.2 * Jill) →
  Jim = 2 :=
by
  intros John Jill Jim hJohn hJill hJim
  sorry

end NUMINAMATH_GPT_jim_travel_distance_l1187_118721


namespace NUMINAMATH_GPT_novelists_count_l1187_118705

theorem novelists_count (n p : ℕ) (h1 : n / (n + p) = 5 / 8) (h2 : n + p = 24) : n = 15 :=
sorry

end NUMINAMATH_GPT_novelists_count_l1187_118705


namespace NUMINAMATH_GPT_initial_mixture_equals_50_l1187_118747

theorem initial_mixture_equals_50 (x : ℝ) (h1 : 0.10 * x + 10 = 0.25 * (x + 10)) : x = 50 :=
by
  sorry

end NUMINAMATH_GPT_initial_mixture_equals_50_l1187_118747


namespace NUMINAMATH_GPT_jancy_currency_notes_l1187_118773

theorem jancy_currency_notes (x y : ℕ) (h1 : 70 * x + 50 * y = 5000) (h2 : y = 2) : x + y = 72 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_jancy_currency_notes_l1187_118773


namespace NUMINAMATH_GPT_biff_break_even_hours_l1187_118713

def totalSpent (ticket drinks snacks headphones : ℕ) : ℕ :=
  ticket + drinks + snacks + headphones

def netEarningsPerHour (earningsCost wifiCost : ℕ) : ℕ :=
  earningsCost - wifiCost

def hoursToBreakEven (totalSpent netEarnings : ℕ) : ℕ :=
  totalSpent / netEarnings

-- given conditions
def given_ticket : ℕ := 11
def given_drinks : ℕ := 3
def given_snacks : ℕ := 16
def given_headphones : ℕ := 16
def given_earningsPerHour : ℕ := 12
def given_wifiCostPerHour : ℕ := 2

theorem biff_break_even_hours :
  hoursToBreakEven (totalSpent given_ticket given_drinks given_snacks given_headphones) 
                   (netEarningsPerHour given_earningsPerHour given_wifiCostPerHour) = 3 :=
by
  sorry

end NUMINAMATH_GPT_biff_break_even_hours_l1187_118713


namespace NUMINAMATH_GPT_retail_price_percentage_l1187_118769

variable (P : ℝ)
variable (wholesale_cost : ℝ)
variable (employee_price : ℝ)

axiom wholesale_cost_def : wholesale_cost = 200
axiom employee_price_def : employee_price = 192
axiom employee_discount_def : employee_price = 0.80 * (wholesale_cost + (P / 100 * wholesale_cost))

theorem retail_price_percentage (P : ℝ) (wholesale_cost : ℝ) (employee_price : ℝ)
    (H1 : wholesale_cost = 200)
    (H2 : employee_price = 192)
    (H3 : employee_price = 0.80 * (wholesale_cost + (P / 100 * wholesale_cost))) :
    P = 20 :=
  sorry

end NUMINAMATH_GPT_retail_price_percentage_l1187_118769


namespace NUMINAMATH_GPT_area_of_square_l1187_118772

-- Define the parabola and the line
def parabola (x : ℝ) : ℝ := x^2 + 4 * x + 3
def line (y : ℝ) : Prop := y = 7

-- Define the roots of the quadratic equation derived from the conditions
noncomputable def root1 : ℝ := -2 + 2 * Real.sqrt 2
noncomputable def root2 : ℝ := -2 - 2 * Real.sqrt 2

-- Define the side length of the square
noncomputable def side_length : ℝ := abs (root1 - root2)

-- Define the area of the square
noncomputable def area_square : ℝ := side_length^2

-- Theorem statement for the problem
theorem area_of_square : area_square = 32 :=
sorry

end NUMINAMATH_GPT_area_of_square_l1187_118772


namespace NUMINAMATH_GPT_problem_statement_l1187_118715

theorem problem_statement (a b : ℝ) (h : a > b) : -2 * a < -2 * b := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1187_118715


namespace NUMINAMATH_GPT_quadrilateral_area_ABCDEF_l1187_118703

theorem quadrilateral_area_ABCDEF :
  ∀ (A B C D E : Type)
  (AC CD AE : ℝ) 
  (angle_ABC angle_ACD : ℝ),
  angle_ABC = 90 ∧
  angle_ACD = 90 ∧
  AC = 20 ∧
  CD = 30 ∧
  AE = 5 →
  ∃ S : ℝ, S = 360 :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_area_ABCDEF_l1187_118703


namespace NUMINAMATH_GPT_range_of_b_l1187_118736

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 1

theorem range_of_b (b : ℝ) : 
  (∃ (x1 x2 x3 : ℝ), f x1 = -b ∧ f x2 = -b ∧ f x3 = -b ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) ↔ (-1 < b ∧ b < 0) :=
by
  sorry

end NUMINAMATH_GPT_range_of_b_l1187_118736


namespace NUMINAMATH_GPT_black_car_overtakes_red_car_in_one_hour_l1187_118778

def red_car_speed : ℕ := 40
def black_car_speed : ℕ := 50
def initial_gap : ℕ := 10

theorem black_car_overtakes_red_car_in_one_hour (h_red_car_speed : red_car_speed = 40)
                                               (h_black_car_speed : black_car_speed = 50)
                                               (h_initial_gap : initial_gap = 10) :
  initial_gap / (black_car_speed - red_car_speed) = 1 :=
by
  sorry

end NUMINAMATH_GPT_black_car_overtakes_red_car_in_one_hour_l1187_118778


namespace NUMINAMATH_GPT_number_of_even_three_digit_numbers_l1187_118708

theorem number_of_even_three_digit_numbers : 
  ∃ (count : ℕ), 
  count = 12 ∧ 
  (∀ (d1 d2 : ℕ), (0 ≤ d1 ∧ d1 ≤ 4) ∧ (Even d1) ∧ (0 ≤ d2 ∧ d2 ≤ 4) ∧ (Even d2) ∧ d1 ≠ d2 →
   ∃ (d3 : ℕ), (d3 = 1 ∨ d3 = 3) ∧ 
   ∃ (units tens hundreds : ℕ), 
     (units ∈ [0, 2, 4]) ∧ 
     (tens ∈ [0, 2, 4]) ∧ 
     (hundreds ∈ [1, 3]) ∧ 
     (units ≠ tens) ∧ 
     (units ≠ hundreds) ∧ 
     (tens ≠ hundreds) ∧ 
     ((units + tens * 10 + hundreds * 100) % 2 = 0) ∧ 
     count = 12) :=
sorry

end NUMINAMATH_GPT_number_of_even_three_digit_numbers_l1187_118708


namespace NUMINAMATH_GPT_John_works_5_days_a_week_l1187_118767

theorem John_works_5_days_a_week
  (widgets_per_hour : ℕ)
  (hours_per_day : ℕ)
  (widgets_per_week : ℕ)
  (H1 : widgets_per_hour = 20)
  (H2 : hours_per_day = 8)
  (H3 : widgets_per_week = 800) :
  widgets_per_week / (widgets_per_hour * hours_per_day) = 5 :=
by
  sorry

end NUMINAMATH_GPT_John_works_5_days_a_week_l1187_118767


namespace NUMINAMATH_GPT_radius_ratio_l1187_118775

theorem radius_ratio (V₁ V₂ : ℝ) (hV₁ : V₁ = 432 * Real.pi) (hV₂ : V₂ = 108 * Real.pi) : 
  (∃ (r₁ r₂ : ℝ), V₁ = (4/3) * Real.pi * r₁^3 ∧ V₂ = (4/3) * Real.pi * r₂^3) →
  ∃ k : ℝ, k = r₂ / r₁ ∧ k = 1 / 2^(2/3) := 
by
  sorry

end NUMINAMATH_GPT_radius_ratio_l1187_118775


namespace NUMINAMATH_GPT_range_of_h_l1187_118794

theorem range_of_h 
  (y1 y2 y3 k : ℝ)
  (h : ℝ)
  (H1 : y1 = (-3 - h)^2 + k)
  (H2 : y2 = (-1 - h)^2 + k)
  (H3 : y3 = (1 - h)^2 + k)
  (H_ord : y2 < y1 ∧ y1 < y3) : 
  -2 < h ∧ h < -1 :=
sorry

end NUMINAMATH_GPT_range_of_h_l1187_118794


namespace NUMINAMATH_GPT_germs_per_dish_l1187_118725

/--
Given:
- the total number of germs is \(5.4 \times 10^6\),
- the number of petri dishes is 10,800,

Prove:
- the number of germs per dish is 500.
-/
theorem germs_per_dish (total_germs : ℝ) (petri_dishes: ℕ) (h₁: total_germs = 5.4 * 10^6) (h₂: petri_dishes = 10800) :
  (total_germs / petri_dishes = 500) :=
sorry

end NUMINAMATH_GPT_germs_per_dish_l1187_118725


namespace NUMINAMATH_GPT_distinct_digit_sum_equation_l1187_118744

theorem distinct_digit_sum_equation :
  ∃ (F O R T Y S I X : ℕ), 
    F ≠ O ∧ F ≠ R ∧ F ≠ T ∧ F ≠ Y ∧ F ≠ S ∧ F ≠ I ∧ F ≠ X ∧ 
    O ≠ R ∧ O ≠ T ∧ O ≠ Y ∧ O ≠ S ∧ O ≠ I ∧ O ≠ X ∧ 
    R ≠ T ∧ R ≠ Y ∧ R ≠ S ∧ R ≠ I ∧ R ≠ X ∧ 
    T ≠ Y ∧ T ≠ S ∧ T ≠ I ∧ T ≠ X ∧ 
    Y ≠ S ∧ Y ≠ I ∧ Y ≠ X ∧ 
    S ≠ I ∧ S ≠ X ∧ 
    I ≠ X ∧ 
    FORTY = 10000 * F + 1000 * O + 100 * R + 10 * T + Y ∧ 
    TEN = 100 * T + 10 * E + N ∧ 
    SIXTY = 10000 * S + 1000 * I + 100 * X + 10 * T + Y ∧ 
    FORTY + TEN + TEN = SIXTY ∧ 
    SIXTY = 31486 :=
sorry

end NUMINAMATH_GPT_distinct_digit_sum_equation_l1187_118744


namespace NUMINAMATH_GPT_phone_price_in_october_l1187_118755

variable (a : ℝ) (P_October : ℝ) (r : ℝ)

noncomputable def price_in_january := a
noncomputable def price_in_october (a : ℝ) (r : ℝ) := a * r^9

theorem phone_price_in_october :
  r = 0.97 →
  P_October = price_in_october a r →
  P_October = a * (0.97)^9 :=
by
  intros h1 h2
  rw [h1] at h2
  exact h2

end NUMINAMATH_GPT_phone_price_in_october_l1187_118755


namespace NUMINAMATH_GPT_problem_statement_l1187_118724

def g (x : ℝ) : ℝ := 3 * x + 2

theorem problem_statement : g (g (g 3)) = 107 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1187_118724


namespace NUMINAMATH_GPT_probability_same_carriage_l1187_118730

theorem probability_same_carriage (num_carriages num_people : ℕ) (h1 : num_carriages = 10) (h2 : num_people = 3) : 
  ∃ p : ℚ, p = 7/25 ∧ p = 1 - (10 * 9 * 8) / (10^3) :=
by
  sorry

end NUMINAMATH_GPT_probability_same_carriage_l1187_118730


namespace NUMINAMATH_GPT_speed_of_first_plane_l1187_118760

theorem speed_of_first_plane
  (v : ℕ)
  (travel_time : ℚ := 44 / 11)
  (relative_speed : ℚ := v + 90)
  (distance : ℚ := 800) :
  (relative_speed * travel_time = distance) → v = 110 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_first_plane_l1187_118760


namespace NUMINAMATH_GPT_area_of_triangle_XPQ_l1187_118799

noncomputable def area_triangle_XPQ (XY YZ XZ XP XQ : ℝ) (hXY : XY = 12) (hYZ : YZ = 13) (hXZ : XZ = 15) (hXP : XP = 5) (hXQ : XQ = 9) : ℝ :=
  let s := (XY + YZ + XZ) / 2
  let area_XYZ := Real.sqrt (s * (s - XY) * (s - YZ) * (s - XZ))
  let cosX := (XY^2 + YZ^2 - XZ^2) / (2 * XY * YZ)
  let sinX := Real.sqrt (1 - cosX^2)
  (1 / 2) * XP * XQ * sinX

theorem area_of_triangle_XPQ :
  area_triangle_XPQ 12 13 15 5 9 (by rfl) (by rfl) (by rfl) (by rfl) (by rfl) = 45 * Real.sqrt 1400 / 78 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_XPQ_l1187_118799


namespace NUMINAMATH_GPT_original_number_unique_l1187_118783

theorem original_number_unique (N : ℤ) (h : (N - 31) % 87 = 0) : N = 118 :=
by
  sorry

end NUMINAMATH_GPT_original_number_unique_l1187_118783


namespace NUMINAMATH_GPT_find_range_of_a_l1187_118720

theorem find_range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x > a) ∨ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0) ∧ 
  ¬ ((∀ x : ℝ, x^2 - 2 * x > a) ∧ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0)) → 
  a ∈ Set.Ioo (-2:ℝ) (-1:ℝ) ∪ Set.Ici (1:ℝ) :=
sorry

end NUMINAMATH_GPT_find_range_of_a_l1187_118720


namespace NUMINAMATH_GPT_father_current_age_l1187_118718

namespace AgeProof

def daughter_age : ℕ := 10
def years_future : ℕ := 20

def father_age (D : ℕ) : ℕ := 4 * D

theorem father_current_age :
  ∀ D : ℕ, ∀ F : ℕ, (F = father_age D) →
  (F + years_future = 2 * (D + years_future)) →
  D = daughter_age →
  F = 40 :=
by
  intro D F h1 h2 h3
  sorry

end AgeProof

end NUMINAMATH_GPT_father_current_age_l1187_118718


namespace NUMINAMATH_GPT_rectangle_side_ratio_l1187_118746

theorem rectangle_side_ratio
  (s : ℝ) -- side length of inner square
  (x y : ℝ) -- longer side and shorter side of the rectangle
  (h_inner_square : y = s) -- shorter side aligns to form inner square
  (h_outer_area : (3 * s) ^ 2 = 9 * s ^ 2) -- area of outer square is 9 times the inner square
  (h_outer_side_relation : x + s = 3 * s) -- outer side length relation
  : x / y = 2 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_side_ratio_l1187_118746


namespace NUMINAMATH_GPT_range_of_a_if_f_decreasing_l1187_118737

noncomputable def f (a x : ℝ) : ℝ := Real.sqrt (x^2 - a * x + 4)

theorem range_of_a_if_f_decreasing:
  ∀ (a : ℝ),
    (∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x < y → f a y < f a x) →
    2 ≤ a ∧ a ≤ 5 :=
by
  intros a h
  sorry

end NUMINAMATH_GPT_range_of_a_if_f_decreasing_l1187_118737


namespace NUMINAMATH_GPT_largest_value_is_D_l1187_118749

theorem largest_value_is_D :
  let A := 15432 + 1/3241
  let B := 15432 - 1/3241
  let C := 15432 * (1/3241)
  let D := 15432 / (1/3241)
  let E := 15432.3241
  max (max (max A B) (max C D)) E = D := by
{
  sorry -- proof not required
}

end NUMINAMATH_GPT_largest_value_is_D_l1187_118749


namespace NUMINAMATH_GPT_fifth_stair_area_and_perimeter_stair_for_area_78_stair_for_perimeter_100_l1187_118753

-- Conditions
def square_side : ℕ := 1
def area_per_square : ℕ := square_side * square_side
def area_of_stair (n : ℕ) : ℕ := (n * (n + 1)) / 2
def perimeter_of_stair (n : ℕ) : ℕ := 4 * n

-- Part (a)
theorem fifth_stair_area_and_perimeter :
  area_of_stair 5 = 15 ∧ perimeter_of_stair 5 = 20 := by
  sorry

-- Part (b)
theorem stair_for_area_78 :
  ∃ n, area_of_stair n = 78 ∧ n = 12 := by
  sorry

-- Part (c)
theorem stair_for_perimeter_100 :
  ∃ n, perimeter_of_stair n = 100 ∧ n = 25 := by
  sorry

end NUMINAMATH_GPT_fifth_stair_area_and_perimeter_stair_for_area_78_stair_for_perimeter_100_l1187_118753


namespace NUMINAMATH_GPT_num_perfect_squares_in_range_l1187_118716

-- Define the range for the perfect squares
def lower_bound := 75
def upper_bound := 400

-- Define the smallest integer whose square is greater than lower_bound
def lower_int := 9

-- Define the largest integer whose square is less than or equal to upper_bound
def upper_int := 20

-- State the proof problem
theorem num_perfect_squares_in_range : 
  (upper_int - lower_int + 1) = 12 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_num_perfect_squares_in_range_l1187_118716


namespace NUMINAMATH_GPT_painting_area_l1187_118732

theorem painting_area (c t A : ℕ) (h1 : c = 15) (h2 : t = 840) (h3 : c * A = t) : A = 56 := 
by
  sorry -- proof to demonstrate A = 56

end NUMINAMATH_GPT_painting_area_l1187_118732


namespace NUMINAMATH_GPT_expression_evaluation_l1187_118739

theorem expression_evaluation (a : ℝ) (h : a = Real.sqrt 2 - 3) : 
  (2 * a + Real.sqrt 3) * (2 * a - Real.sqrt 3) - 3 * a * (a - 2) + 3 = -7 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1187_118739


namespace NUMINAMATH_GPT_combination_multiplication_and_addition_l1187_118707

theorem combination_multiplication_and_addition :
  (Nat.choose 10 3) * (Nat.choose 8 3) + (Nat.choose 5 2) = 6730 :=
by
  sorry

end NUMINAMATH_GPT_combination_multiplication_and_addition_l1187_118707


namespace NUMINAMATH_GPT_cats_weight_more_than_puppies_l1187_118779

theorem cats_weight_more_than_puppies :
  let num_puppies := 4
  let weight_per_puppy := 7.5
  let num_cats := 14
  let weight_per_cat := 2.5
  (num_cats * weight_per_cat) - (num_puppies * weight_per_puppy) = 5 :=
by 
  let num_puppies := 4
  let weight_per_puppy := 7.5
  let num_cats := 14
  let weight_per_cat := 2.5
  sorry

end NUMINAMATH_GPT_cats_weight_more_than_puppies_l1187_118779


namespace NUMINAMATH_GPT_textbook_cost_l1187_118770

theorem textbook_cost 
  (credits : ℕ) 
  (cost_per_credit : ℕ) 
  (facility_fee : ℕ) 
  (total_cost : ℕ) 
  (num_textbooks : ℕ) 
  (total_spent : ℕ) 
  (h1 : credits = 14) 
  (h2 : cost_per_credit = 450) 
  (h3 : facility_fee = 200) 
  (h4 : total_spent = 7100) 
  (h5 : num_textbooks = 5) :
  (total_cost - (credits * cost_per_credit + facility_fee)) / num_textbooks = 120 :=
by
  sorry

end NUMINAMATH_GPT_textbook_cost_l1187_118770
