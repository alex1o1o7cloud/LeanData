import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.Basic
import Mathlib.Algebra.Trigonometry
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.DifferentialEquations
import Mathlib.Analysis.Geometry.Area
import Mathlib.Analysis.Polynomial.Basic
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecificFunctions.Trigonometric
import Mathlib.Analysis.Trigonometry
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Combination
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Graph
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Defs
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Init.Data.Int.Basic
import Mathlib.LinearAlgebra.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tools.Tauto
import Mathlib.Topology.Algebra.InfiniteSum
import Real

namespace exam_question_attempted_l735_735234

-- Defining the conditions as hypotheses
def marking_scheme (total_questions correct_questions total_marks : ℕ) : Prop :=
  let incorrect_questions := total_questions - correct_questions in
  (correct_questions * 4) - (incorrect_questions * 1) = total_marks

-- Example theorem stating the given problem and its answer
theorem exam_question_attempted (total_questions : ℕ) :
  marking_scheme total_questions 40 120 → total_questions = 80 := 
by
  sorry

end exam_question_attempted_l735_735234


namespace simplify_product_of_fractions_l735_735308

theorem simplify_product_of_fractions :
  (25 / 24) * (18 / 35) * (56 / 45) = (50 / 3) :=
by sorry

end simplify_product_of_fractions_l735_735308


namespace log_arithmetic_sequence_l735_735882

noncomputable def f (x : ℝ) : ℝ := 2^x

theorem log_arithmetic_sequence :
  ∀ (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = a n + 2),
  f (a 2 + a 4 + a 6 + a 8 + a 10) = 4 →
  log 2 (f (a 1) * f (a 2) * f (a 3) * f (a 4) * f (a 5) * f (a 6) * f (a 7) * f (a 8) * f (a 9) * f (a 10)) = -6 := 
by 
  sorry

end log_arithmetic_sequence_l735_735882


namespace find_b_minus_a_l735_735292

noncomputable def rotate_90_counterclockwise (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  let (a, b) := p1
  let (c, d) := p2
  ((a - c) * 0 - (b - d) * 1 + c, (a - c) * 1 + (b - d) * 0 + d)

def reflect_about_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (-y, -x)

def P_after_transformations (P : ℝ × ℝ) : ℝ × ℝ :=
  reflect_about_y_eq_neg_x (rotate_90_counterclockwise P (1, 5))

theorem find_b_minus_a (a b : ℝ) : P_after_transformations (a, b) = (-6, 3) → b - a = 7 := by
  intros h
  sorry

end find_b_minus_a_l735_735292


namespace percent_decrease_of_y_l735_735314

theorem percent_decrease_of_y (k x y q : ℝ) (h_inv_prop : x * y = k) (h_pos : 0 < x ∧ 0 < y) (h_q : 0 < q) :
  let x' := x * (1 + q / 100)
  let y' := y * 100 / (100 + q)
  (y - y') / y * 100 = (100 * q) / (100 + q) :=
by
  sorry

end percent_decrease_of_y_l735_735314


namespace real_number_m_values_l735_735850

def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

def on_circle (z : ℂ) (m : ℝ) : Prop :=
  (z.re)^2 + (z.im + 3 * m)^2 = 17

theorem real_number_m_values {m : ℝ} :
  let z := (m^2 - 1) + (m^2 - 3 * m + 2) * complex.I in
  (is_pure_imaginary z → m = -1) ∧ (on_circle z m → m = sqrt 2 ∨ m = -sqrt 2) :=
by
  sorry

end real_number_m_values_l735_735850


namespace sum_of_fourth_powers_eq_82_l735_735209

theorem sum_of_fourth_powers_eq_82 (x y : ℝ) (hx : x + y = -2) (hy : x * y = -3) :
  x^4 + y^4 = 82 :=
by
  sorry

end sum_of_fourth_powers_eq_82_l735_735209


namespace exists_valid_net_and_cut_square_l735_735297

-- Define rectangular paper prism
def RectangularPrism2x1x1 : Type := 
  {faces : List (Fin 10) // faces.length = 10}

-- Define the net after cutting along edges and removing one square
def possibleNetWithOneSquareRemoved : Prop :=
  ∃ (net: RectangularPrism2x1x1), ∃ cutOffSquare: Fin 10,
    let remainingSquares := net.faces.erase cutOffSquare in
    remainingSquares.length = 9

-- Lean 4 statement proving the existence of a valid configuration
theorem exists_valid_net_and_cut_square :
  possibleNetWithOneSquareRemoved :=
sorry

end exists_valid_net_and_cut_square_l735_735297


namespace triangle_NMC_area_l735_735069

variables {A B C D L M N K : Type} [Square A B C D]
variables {x y : ℝ} (h : 1 - x - y = sqrt (x^2 + y^2))

theorem triangle_NMC_area 
  (h_sq : IsSquare A B C D)
  (h_L_on_AB : LiesOn L A B)
  (h_M_on_BC : LiesOn M B C)
  (h_N_on_CD : LiesOn N C D)
  (h_K_on_AD : LiesOn K A D)
  (h_AL : SegmentLength A L = x)
  (h_AK : SegmentLength A K = y)
  : TriangleArea N M C = 1 / 4 :=
by 
  sorry

end triangle_NMC_area_l735_735069


namespace continuous_function_fixed_point_l735_735329

variable (f : ℝ → ℝ)
variable (h_cont : Continuous f)
variable (h_comp : ∀ x : ℝ, ∃ n : ℕ, n > 0 ∧ (f^[n] x = 1))

theorem continuous_function_fixed_point : f 1 = 1 := 
by
  sorry

end continuous_function_fixed_point_l735_735329


namespace angle_size_and_max_y_l735_735863

noncomputable def triangle_sides := {a b c : ℝ}

theorem angle_size_and_max_y
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : (2 * b - a) * Real.cos C = c * Real.cos A) :
  (C = π / 3) ∧ 
  (∃ y, y = (-4) * Real.sqrt 3 * (Real.sin (A / 2)) ^ 2 + 2 * Real.sin (C - B) ∧ 
          y ≤ 2 - 2 * Real.sqrt 3 ∧ 
          ∀ y', y' = (-4) * Real.sqrt 3 * (Real.sin (A / 2)) ^ 2 + 2 * Real.sin (C - B) → 
          y' ≤ y →
          ∃ t, t = 2 - 2 * Real.sqrt 3 ∧ 
               (B = π / 2 ∨ C = π / 2 ∨ A = π / 2)) :=
  sorry

end angle_size_and_max_y_l735_735863


namespace quadratic_polynomial_discriminant_l735_735143

def P (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∃ x : ℝ, P a b c x = x - 2 ∧ (discriminant a (b - 1) (c + 2) = 0))
  (h₂ : ∃ x : ℝ, P a b c x = 1 - x / 2 ∧ (discriminant a (b + 1 / 2) (c - 1) = 0)) :
  discriminant a b c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l735_735143


namespace combined_squirrel_count_is_28_l735_735293

def squirrel_count_combined(first_student_count : ℕ, second_student_addition_fraction : ℚ) : ℕ :=
  let second_student_count := first_student_count + (second_student_addition_fraction * first_student_count).natAbs
  in first_student_count + second_student_count

theorem combined_squirrel_count_is_28 (h1 : first_student_count = 12) (h2 : second_student_addition_fraction = 1/3) :
  squirrel_count_combined first_student_count second_student_addition_fraction = 28 := by
  sorry

end combined_squirrel_count_is_28_l735_735293


namespace welders_started_on_other_project_l735_735417

theorem welders_started_on_other_project
  (r : ℝ) (x : ℝ) (W : ℝ)
  (h1 : 16 * r * 8 = W)
  (h2 : (16 - x) * r * 24 = W - 16 * r) :
  x = 11 :=
by
  sorry

end welders_started_on_other_project_l735_735417


namespace find_number_l735_735765

theorem find_number (divisor quotient remainder : ℕ) (h_divisor : divisor = 20) (h_quotient : quotient = 10) (h_remainder : remainder = 10) :
  (divisor * quotient + remainder) = 210 :=
by
  rw [h_divisor, h_quotient, h_remainder]
  sorry

end find_number_l735_735765


namespace find_fourth_number_l735_735737

variables (A B C D E F : ℝ)

theorem find_fourth_number
  (h1 : A + B + C + D + E + F = 180)
  (h2 : A + B + C + D = 100)
  (h3 : D + E + F = 105) :
  D = 25 :=
by
  sorry

end find_fourth_number_l735_735737


namespace area_of_triangle_l735_735584

-- Definitions
variables {A B C : Type}
variables {i j k : ℕ}
variables (AB AC : ℝ)
variables (s t : ℝ)
variables (sinA : ℝ) (cosA : ℝ)

-- Conditions 
axiom sin_A : sinA = 4 / 5
axiom dot_product : s * t * cosA = 6

-- The problem theorem
theorem area_of_triangle : (1 / 2) * s * t * sinA = 4 :=
by
  sorry

end area_of_triangle_l735_735584


namespace sum_exponents_binary_3400_l735_735219

theorem sum_exponents_binary_3400 : 
  ∃ (a b c d e : ℕ), 
    3400 = 2^a + 2^b + 2^c + 2^d + 2^e ∧ 
    a > b ∧ b > c ∧ c > d ∧ d > e ∧ 
    a + b + c + d + e = 38 :=
sorry

end sum_exponents_binary_3400_l735_735219


namespace cannot_transform_with_swap_rows_and_columns_l735_735408

def initialTable : Matrix (Fin 3) (Fin 3) ℕ :=
![![1, 2, 3], ![4, 5, 6], ![7, 8, 9]]

def goalTable : Matrix (Fin 3) (Fin 3) ℕ :=
![![1, 4, 7], ![2, 5, 8], ![3, 6, 9]]

theorem cannot_transform_with_swap_rows_and_columns :
  ¬ ∃ (is_transformed_by_swapping : Matrix (Fin 3) (Fin 3) ℕ → Matrix (Fin 3) (Fin 3) ℕ → Prop),
    is_transformed_by_swapping initialTable goalTable :=
by sorry

end cannot_transform_with_swap_rows_and_columns_l735_735408


namespace painted_cubes_with_two_sides_l735_735436

def cuboid_width := 4
def cuboid_length := 3
def cuboid_height := 3
def total_small_cubes := 36

theorem painted_cubes_with_two_sides :
  ∃ (small_cube_side : ℕ),
    (cuboid_width / small_cube_side) * (cuboid_length / small_cube_side) * (cuboid_height / small_cube_side) = total_small_cubes ∧
    (∃ n, n = 16 ∧ n = number_of_cubes_with_two_painted_sides (cuboid_width / small_cube_side)
                                                        (cuboid_length / small_cube_side)
                                                        (cuboid_height / small_cube_side)) :=
sorry

end painted_cubes_with_two_sides_l735_735436


namespace concurrent_lines_in_hexagon_of_equidistant_points_l735_735233

theorem concurrent_lines_in_hexagon_of_equidistant_points
  (A B C : Point)
  (hABC : equilateral_triangle ABC)
  (A1 A2 B1 B2 C1 C2 : Point)
  (hA1A2_on_BC : A1 ∈ segment B C ∧ A2 ∈ segment B C)
  (hB1B2_on_CA : B1 ∈ segment C A ∧ B2 ∈ segment C A)
  (hC1C2_on_AB : C1 ∈ segment A B ∧ C2 ∈ segment A B)
  (hAllSidesEqual : ∀ (P Q : Point), P ∈ [A1, A2, B1, B2, C1, C2] → Q ∈ [A1, A2, B1, B2, C1, C2] → length (segment P Q) = length (segment A1 A2)) :
  are_concurrent [line_through A1 B2, line_through B1 C2, line_through C1 A2] := 
sorry

end concurrent_lines_in_hexagon_of_equidistant_points_l735_735233


namespace terminal_side_in_second_quadrant_l735_735521

def is_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0
def is_second_quadrant (α : ℝ) : Prop := π / 2 < α ∧ α < π
def tan (α : ℝ) : ℝ := Real.sin α / Real.cos α

theorem terminal_side_in_second_quadrant (α : ℝ) 
  (h1 : is_third_quadrant (tan α) (Real.cos α)) : is_second_quadrant α :=
by
  sorry

end terminal_side_in_second_quadrant_l735_735521


namespace smallest_combination_is_145_l735_735643

-- Define the conditions: having cards 1, 4, and 5
def card1 : ℕ := 1
def card4 : ℕ := 4
def card5 : ℕ := 5

-- Define the smallest number you can make with those cards
def smallest_number := 145

-- Statement to prove smallest_number is indeed the smallest combination
theorem smallest_combination_is_145 : 
  ∀ (a b c : ℕ), {a, b, c} = {card1, card4, card5} → (∃ n : ℕ, n = card1 ∨ n = card4 ∨ n = card5 ∧
  smallest_number = a * 100 + b * 10 + c) → smallest_number = 145 :=
by 
  intros a b c h1 h2
  sorry

end smallest_combination_is_145_l735_735643


namespace xy_2yz_3zx_eq_24_sqrt_3_l735_735302

/-- 
Given three positive real numbers x, y, z satisfying the following system of equations:
1. x^2 + xy + y^2 / 3 = 25
2. y^2 / 3 + z^2 = 9
3. z^2 + zx + x^2 = 16,
prove that xy + 2yz + 3zx = 24 * real.sqrt 3.
-/
theorem xy_2yz_3zx_eq_24_sqrt_3 (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (eq1 : x^2 + x * y + y^2 / 3 = 25)
  (eq2 : y^2 / 3 + z^2 = 9)
  (eq3 : z^2 + z * x + x^2 = 16) :
  x * y + 2 * y * z + 3 * z * x = 24 * real.sqrt 3 := 
sorry

end xy_2yz_3zx_eq_24_sqrt_3_l735_735302


namespace count_divisors_not_by_2_or_3_l735_735564

theorem count_divisors_not_by_2_or_3 : 
  ∃ (n : ℕ), n = 210 ∧ (∀ d : ℕ, d > 0 ∧ d ∣ n → ¬(2 ∣ d) ∧ ¬(3 ∣ d) → d = 1 ∨ d = 5 ∨ d = 7 ∨ d = 35) ∧ (set.toFinset {d : ℕ | d > 0 ∧ d ∣ 210 ∧ ¬(2 ∣ d) ∧ ¬(3 ∣ d)}).card = 4 :=
by
  sorry

end count_divisors_not_by_2_or_3_l735_735564


namespace angle_KOL_gt_90_l735_735277

open EuclideanGeometry

theorem angle_KOL_gt_90 (O : Circle) (A B C P K L : Point)
  (h_eq_triABC : EquilateralTriangle A B C)
  (h_pointP_arcBC : P ∈ arc BC O)
  (h_tangent : Tangent O P)
  (h_intersect_K : LineThrough P ∩ extendedLineThrough A B = K)
  (h_intersect_L : LineThrough P ∩ extendedLineThrough A C = L) :
  angle K O L > 90 := 
sorry

end angle_KOL_gt_90_l735_735277


namespace domain_of_f_l735_735386

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x - 5) + real.cbrt (x + 4)

theorem domain_of_f :
  {x : ℝ | ∃ y, y = f x} = {x : ℝ | x ≥ 5} :=
by
  sorry

end domain_of_f_l735_735386


namespace hurricane_average_rainfall_l735_735760

noncomputable def total_rainfall (rainfalls : List ℝ) : ℝ :=
  rainfalls.sum

noncomputable def overall_average_rainfall (total_rainfall : ℝ) (total_time : ℝ) : ℝ :=
  total_rainfall / total_time

theorem hurricane_average_rainfall :
  let stormA_rainfall_1 := 5
  let stormA_rainfall_2 := 2.5
  let stormB_rainfall_1 := 3
  let stormB_rainfall_2 := 4.5
  let stormC_rainfall_rate := 1 / 2
  let stormA_time := 1
  let stormB_time := 1 + (45 / 60)
  let stormC_time := 3
  let total_rainfall := total_rainfall [stormA_rainfall_1 + stormA_rainfall_2, stormB_rainfall_1 + stormB_rainfall_2, stormC_rainfall_rate * stormC_time]
  let total_time := stormA_time + stormB_time + stormC_time
  overall_average_rainfall total_rainfall total_time = 2.87 :=
by
  let stormA_rainfall := stormA_rainfall_1 + stormA_rainfall_2
  have hA : stormA_rainfall = 7.5 := by norm_num
  let stormB_rainfall := stormB_rainfall_1 + stormB_rainfall_2
  have hB : stormB_rainfall = 7.5 := by norm_num
  let stormC_rainfall := stormC_rainfall_rate * stormC_time
  have hC : stormC_rainfall = 1.5 := by norm_num
  let total_rainfall := total_rainfall [stormA_rainfall, stormB_rainfall, stormC_rainfall]
  have hr : total_rainfall = 16.5 :=
    by simp [total_rainfall, hA, hB, hC]; norm_num
  let total_time := stormA_time + stormB_time + stormC_time
  have ht : total_time = 5.75 :=
    by simp [stormA_time, stormB_time, stormC_time]; norm_num
  have ha : overall_average_rainfall total_rainfall total_time = 2.87 :=
    by simp [overall_average_rainfall, hr, ht]; norm_num
  exact ha

end hurricane_average_rainfall_l735_735760


namespace length_of_side_b_l735_735919

theorem length_of_side_b (B C : ℝ) (c b : ℝ) (hB : B = 45 * Real.pi / 180) (hC : C = 60 * Real.pi / 180) (hc : c = 1) :
  b = Real.sqrt 6 / 3 :=
by
  sorry

end length_of_side_b_l735_735919


namespace intersection_A_B_range_m_l735_735896

-- Define set A when m = 3 as given
def A_set (x : ℝ) : Prop := 3 - 2 * x - x^2 ≥ 0
def A (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 1

-- Define set B when m = 3 as given
def B_set (x : ℝ) (m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0
def B (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 4

-- The intersection of A and B should be: -2 ≤ x ≤ 1
theorem intersection_A_B : ∀ (x : ℝ), A x ∧ B x ↔ (-2 ≤ x ∧ x ≤ 1) := sorry

-- Define A for general m > 0
def A_set_general (x : ℝ) : Prop := 3 - 2 * x - x^2 ≥ 0

-- Define B for general m
def B_set_general (x : ℝ) (m : ℝ) : Prop := (x - 1)^2 ≤ m^2

-- Prove the range for m such that A ⊆ B
theorem range_m (m : ℝ) (h : m > 0) : (∀ x, A_set_general x → B_set_general x m) ↔ m ≥ 4 := sorry

end intersection_A_B_range_m_l735_735896


namespace correct_operation_l735_735400

theorem correct_operation :
    (∀ x, x^2 * x^3 ≠ x^6) ∧
    (∀ a, a^6 / a^2 ≠ a^3) ∧
    (∀ a b, (a^3 * b)^2 = a^6 * b^2) ∧
    (∀ a, 5 * a - 3 * a ≠ 2) := by
  sorry

end correct_operation_l735_735400


namespace exact_time_now_is_approx_4_25_l735_735947

theorem exact_time_now_is_approx_4_25 (t : ℝ) (h1 : 0 < t ∧ t < 60)
  (h2 : abs ((6 * t + 60) - (120 + 0.5 * (t - 5))) = 180) :
  t ≈ 25 :=
by sorry

end exact_time_now_is_approx_4_25_l735_735947


namespace _l735_735810

noncomputable theorem distinct_pos_numbers_sum_to_22
  (a b c : ℝ) 
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : a ≠ b)
  (h5 : b ≠ c)
  (h6 : a ≠ c)
  (eq1 : a^2 + b * c = 115)
  (eq2 : b^2 + a * c = 127)
  (eq3 : c^2 + a * b = 115) :
  a + b + c = 22 :=
  sorry

end _l735_735810


namespace red_tetrahedron_volume_l735_735422

def side_length := 8
def cube_volume := side_length^3
def tetra_volume (base_area height : ℝ) := (base_area * height) / 3
def base_triangle_area := (side_length * side_length) / 2
def blue_tetra_volume := 4 * (tetra_volume base_triangle_area side_length)
def red_tetra_volume := cube_volume - blue_tetra_volume

theorem red_tetrahedron_volume :
  red_tetra_volume = 170.67 := by
  sorry

end red_tetrahedron_volume_l735_735422


namespace no_positive_divisor_of_2n2_square_l735_735621

theorem no_positive_divisor_of_2n2_square (n : ℕ) (hn : n > 0) : 
  ∀ d : ℕ, d > 0 → d ∣ 2 * n ^ 2 → ¬∃ x : ℕ, x ^ 2 = d ^ 2 * n ^ 2 + d ^ 3 := 
by
  sorry

end no_positive_divisor_of_2n2_square_l735_735621


namespace correct_operation_l735_735026

theorem correct_operation (x : ℝ) (f : ℝ → ℝ) (h : ∀ x, (x / 10) = 0.01 * f x) : 
  f x = 10 * x :=
by
  sorry

end correct_operation_l735_735026


namespace problem_1_problem_2_problem_3_l735_735065

-- 1. Prove that 3a + b = 9 has positive integer solutions {a = 2, b = 3} and {a = 1, b = 6}.
theorem problem_1 :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ (3 * a + b = 9) ∧ ((a = 2 ∧ b = 3) ∨ (a = 1 ∧ b = 6)) := 
sorry

-- 2. Given 2 (2 - a) = 5b - 2m and 9 - 3b = 5a + m, prove 12a + 11b = 22.
theorem problem_2 (a b m : ℤ) :
  2 * (2 - a) = 5 * b - 2 * m →
  9 - 3 * b = 5 * a + m →
  12 * a + 11 * b = 22 := 
sorry

-- 3. Given a > 1, prove M ≥ N.
theorem problem_3 (a b : ℕ) (h : a > 1) :
  let M := a * b * (a * b - b),
  let N := b * (b - a * b)
  in M ≥ N :=
sorry

end problem_1_problem_2_problem_3_l735_735065


namespace arc_length_of_regular_octagon_l735_735427

def regular_octagon_in_circle_arc_length (r : ℝ) (n : ℕ) (s : ℝ) : ℝ :=
  if n = 8 ∧ s = 5 then (s / r) * (π * r)
  else 0

theorem arc_length_of_regular_octagon 
  (r : ℝ) (s : ℝ) (h₁ : s = 5) (h₂ : r = 5) :
  regular_octagon_in_circle_arc_length r 8 s = (5 * π / 4) :=
by {
  -- Using the conditions and properties derived before
  rw [h₁, h₂, regular_octagon_in_circle_arc_length],
  norm_num,
  have : (5 / 5) * (π * 5) = 5 * π / 4,
  {
    norm_num,
    simp,
  },
  exact this,
}

end arc_length_of_regular_octagon_l735_735427


namespace odd_of_odd_comp_l735_735625

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f(x)

theorem odd_of_odd_comp (f : ℝ → ℝ) (h : is_odd_function f) : is_odd_function (λ x, f (f x)) :=
by 
  intros x
  have h₁ : f (-x) = -f(x), from h x
  calc
    (f ∘ f) (-x) = f (f (-x)) : rfl
              ... = f (-f (x)) : by rw [h₁]
              ... = -f (f (x)) : by rw [h]; exact f(x)

end odd_of_odd_comp_l735_735625


namespace validate_sales_data_l735_735208

noncomputable def average (values : List ℕ) : ℚ :=
  (values.sum : ℚ) / values.length

theorem validate_sales_data :
  let unit_prices := [4, 5, 6, 7, 8, 9]
      sales_volumes := [90, 84, 83, 80, 75, 68]
      avg_y := average sales_volumes
      slope := -4
      intercept := 106
      empirical_regression (x : ℚ) : ℚ := slope * x + intercept in
  average sales_volumes = 80 ∧
  (∀ (x₁ x₂ : ℚ), empirical_regression x₁ - empirical_regression x₂ = slope * (x₁ - x₂)) ∧
  intercept ≠ 26 ∧
  empirical_regression 10 = 66 :=
by
  intros
  sorry

end validate_sales_data_l735_735208


namespace geometric_sequence_sum_eq_l735_735106

variable {α : Type*} [LinearOrderedField α]

/-- For a geometric sequence with sums of the first n, 2n, and 3n terms being A, B, and C respectively,
    the equation B(B-A) = A(C-A) always holds true. -/
theorem geometric_sequence_sum_eq (a_n : ℕ → α) 
  (n : ℕ) (A B C : α)
  (h1 : ∑ i in Finset.range n, a_n i = A)
  (h2 : ∑ i in Finset.range (2 * n), a_n i = B)
  (h3 : ∑ i in Finset.range (3 * n), a_n i = C) :
  B * (B - A) = A * (C - A) := 
sorry

end geometric_sequence_sum_eq_l735_735106


namespace discriminant_of_P_l735_735153

theorem discriminant_of_P (a b c : ℚ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1/2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1/2 := 
begin
  -- Proof omitted for brevity
  sorry
end

end discriminant_of_P_l735_735153


namespace sum_of_reciprocals_of_roots_is_neg_p_l735_735057

open Complex Polynomial

noncomputable def sum_of_reciprocals_of_roots (p q r : ℂ) : ℂ :=
  let roots := (λ (r : ℂ), r ≠ 0 ∧ (z^3 + p*z^2 + q*z + r).eval r = 0 ∧ abs r = 1)
  let reciproals := (λ (r : ℂ), 1/r)
  let sum_of_reciproals := (roots.sum reciproals)
  sum_of_reciproals

theorem sum_of_reciprocals_of_roots_is_neg_p
  (p q r : ℂ)
  (z1 z2 z3 : ℂ)
  (h1 : (z^3 + p*z^2+ q*z + r).eval z1 = 0)
  (h2 : (z^3 + p*z^2+ q*z + r).eval z2 = 0)
  (h3 : (z^3 + p*z^2+ q*z + r).eval z3 = 0)
  (hz1 : abs z1 = 1)
  (hz2 : abs z2 = 1)
  (hz3 : abs z3 = 1)
  : 1/z1 + 1/z2 + 1/z3 = -p :=
sorry

end sum_of_reciprocals_of_roots_is_neg_p_l735_735057


namespace find_discriminant_l735_735157

variables {a b c : ℝ}
variables (P : ℝ → ℝ)
def is_quadratic_polynomial (P : ℝ → ℝ) : Prop := ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, P x = a * x^2 + b * x + c)

theorem find_discriminant (h1 : is_quadratic_polynomial P)
  (h2 : ∃ x, P x = x - 2)
  (h3 : ∃ y, P y = 1 - y / 2)
  : ∃ D, D = -1/2 := 
sorry

end find_discriminant_l735_735157


namespace parabola_equation_l735_735913

theorem parabola_equation (directrix : ℝ) (h : directrix = -7) : ∃ p, y^2 = 4 * p * x ∧ 2 * p = 28 :=
by
  sorry

#eval parabola_equation -7 rfl -- test to ensure it compiles 

end parabola_equation_l735_735913


namespace find_axis_of_symmetry_l735_735873

noncomputable def f : ℝ → ℝ := λ x, Real.sin x + λ * Real.cos x
noncomputable def g : ℝ → ℝ := λ x, λ * Real.sin x * Real.cos x + Real.sin x ^ 2
axiom lambda_value : λ = -Real.sqrt 3
axiom center_of_symmetry : f (Real.pi / 3) = 0
axiom symmetry_axis : g (x) = g (-x)

theorem find_axis_of_symmetry {x : ℝ} (hx : x = -Real.pi / 3) : g (x) = g (-x) :=
sorry

end find_axis_of_symmetry_l735_735873


namespace sum_of_abc_l735_735811

theorem sum_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) 
  (eq1 : a^2 + b * c = 115) (eq2 : b^2 + a * c = 127) (eq3 : c^2 + a * b = 115) :
  a + b + c = 22 := by
  sorry

end sum_of_abc_l735_735811


namespace minimal_operations_to_compute_x2006_l735_735278

-- Define the conditions: multiply and division operations on x
def can_compute_power (x : ℕ → ℕ) : Prop :=
∀ n : ℕ, ∃ (ops : list (ℕ → ℕ → ℕ)), (∀ f ∈ ops, f = (· * ·) ∨ f = (· / ·)) ∧
  ∃ (steps : list (ℕ × ℕ)), (0 < n → n < 2^steps.length) ∧ 
  (steps.head = (x, x)) ∧ 
  (steps.last.fst = x^n) ∧
  (∀ ⟨a, b⟩ ∈ steps, ∃ f ∈ ops, b = f a b) ∧
  ops.length = 17

-- Problem statement: minimal number of operations to compute x^2006
theorem minimal_operations_to_compute_x2006 (x : ℕ) : can_compute_power (λ n => x^2006) :=
sorry

end minimal_operations_to_compute_x2006_l735_735278


namespace zach_needs_more_tickets_l735_735404

theorem zach_needs_more_tickets {ferris_wheel_tickets roller_coaster_tickets log_ride_tickets zach_tickets : ℕ} :
  ferris_wheel_tickets = 2 ∧
  roller_coaster_tickets = 7 ∧
  log_ride_tickets = 1 ∧
  zach_tickets = 1 →
  (ferris_wheel_tickets + roller_coaster_tickets + log_ride_tickets - zach_tickets = 9) :=
by
  intro h
  sorry

end zach_needs_more_tickets_l735_735404


namespace jill_lavender_candles_l735_735615

theorem jill_lavender_candles
  (lavender_ml_per_candle : ℕ := 10)
  (coconut_ml_per_candle : ℕ := 8)
  (almond_ml_per_candle : ℕ := 12)
  (jasmine_ml_per_candle : ℕ := 9)
  (coconut_candles : ℕ)
  (lavender_candles : ℕ := 3 * coconut_candles)
  (jasmine_candles : ℕ)
  (almond_candles : ℕ := 2 * jasmine_candles)
  (almond_scent_total_ml : ℕ := 144)
  (coconut_scent_total_ml : ℕ := 2.5 * almond_scent_total_ml)
  (coconut_candles_calc : coconut_scent_total_ml / coconut_ml_per_candle = 45)
  (jasmine_candles_calc : almond_scent_total_ml / almond_ml_per_candle / 2 = 6)
  : lavender_candles = 135 := by
  sorry

end jill_lavender_candles_l735_735615


namespace second_train_length_correct_l735_735418

-- Define the initial conditions
def length_first_train : ℝ := 270  -- in meters
def speed_first_train_kmph : ℝ := 120  -- in km/h
def speed_second_train_kmph : ℝ := 80  -- in km/h
def crossing_time : ℝ := 9  -- in seconds

-- Conversion factors and calculations
def kmph_to_mps (speed: ℝ) : ℝ := speed * 1000 / 3600
def speed_first_train_mps : ℝ := kmph_to_mps speed_first_train_kmph
def speed_second_train_mps : ℝ := kmph_to_mps speed_second_train_kmph

def relative_speed_mps : ℝ := speed_first_train_mps + speed_second_train_mps  -- relative speed in m/s

-- The length of the second train, to be proven
def length_second_train : ℝ := 229.95  -- in meters

-- Lean proof statement
theorem second_train_length_correct :
  length_first_train + length_second_train = relative_speed_mps * crossing_time :=
by
  sorry

end second_train_length_correct_l735_735418


namespace combined_squirrel_count_is_28_l735_735294

def squirrel_count_combined(first_student_count : ℕ, second_student_addition_fraction : ℚ) : ℕ :=
  let second_student_count := first_student_count + (second_student_addition_fraction * first_student_count).natAbs
  in first_student_count + second_student_count

theorem combined_squirrel_count_is_28 (h1 : first_student_count = 12) (h2 : second_student_addition_fraction = 1/3) :
  squirrel_count_combined first_student_count second_student_addition_fraction = 28 := by
  sorry

end combined_squirrel_count_is_28_l735_735294


namespace angle_C_measure_l735_735641

theorem angle_C_measure (p q : Line) (A B C : Angle) 
  (hpq : p ∥ q) 
  (hA : ∠ A = 2/9 * ∠ B) : ∠ C = 360 / 11 :=
by
  sorry

end angle_C_measure_l735_735641


namespace value_of_expression_l735_735522

theorem value_of_expression (a b c : ℚ) (h1 : a * b * c < 0) (h2 : a + b + c = 0) :
    (a - b - c) / |a| + (b - c - a) / |b| + (c - a - b) / |c| = 2 :=
by
  sorry

end value_of_expression_l735_735522


namespace order_of_magnitude_l735_735898

noncomputable def a : ℝ := 0.2 ^ (-3)
noncomputable def b : ℝ := Real.log 0.2 / Real.log 3
noncomputable def c : ℝ := 3 ^ 0.2

theorem order_of_magnitude : b < c ∧ c < a :=
by
  sorry

end order_of_magnitude_l735_735898


namespace find_fourth_power_subset_l735_735270

/-- Let M be a set consisting of 1985 different positive integers,
 each of whose prime factors are no greater than 26. Prove that
 it is possible to find a subset of M containing at least four distinct numbers 
 such that the product of these four numbers is equal to the fourth power of some positive integer.
 -/
theorem find_fourth_power_subset (M : Finset ℕ) (hM_size : M.card = 1985) 
  (hM_primes : ∀ x ∈ M, ∀ p : ℕ, p.prime → p ∣ x → p ≤ 26) :
  ∃ a b c d ∈ M, (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
  ∃ k : ℕ, a * b * c * d = k ^ 4 :=
by
  sorry

end find_fourth_power_subset_l735_735270


namespace statement1_statement2_statement3_statement4_l735_735909

noncomputable def cyclic_sym (f : ℝ → ℝ → ℝ → ℝ) :=
  ∀ a b c : ℝ, f a b c = f b c a ∧ f a b c = f c a b

def f1 (x y z : ℝ) : ℝ := x^2 - y^2 + z
def f2 (x y z : ℝ) : ℝ := x^2 * (y - z) + y^2 * (z - x) + z^2 * (x - y)
noncomputable def f4 (A B C : ℝ) : ℝ := 2 + cos C * cos (A - B) - (cos C)^2

theorem statement1 : ¬ cyclic_sym f1 := sorry
theorem statement2 : cyclic_sym f2 := sorry
theorem statement3 (f g : ℝ → ℝ → ℝ → ℝ) (h_f : cyclic_sym f) (h_g : cyclic_sym g) : cyclic_sym (λ x y z, f x y z - g x y z) := sorry
theorem statement4 : cyclic_sym f4 := sorry

end statement1_statement2_statement3_statement4_l735_735909


namespace validate_sales_data_l735_735207

noncomputable def average (values : List ℕ) : ℚ :=
  (values.sum : ℚ) / values.length

theorem validate_sales_data :
  let unit_prices := [4, 5, 6, 7, 8, 9]
      sales_volumes := [90, 84, 83, 80, 75, 68]
      avg_y := average sales_volumes
      slope := -4
      intercept := 106
      empirical_regression (x : ℚ) : ℚ := slope * x + intercept in
  average sales_volumes = 80 ∧
  (∀ (x₁ x₂ : ℚ), empirical_regression x₁ - empirical_regression x₂ = slope * (x₁ - x₂)) ∧
  intercept ≠ 26 ∧
  empirical_regression 10 = 66 :=
by
  intros
  sorry

end validate_sales_data_l735_735207


namespace cars_meet_first_time_l735_735841

-- Definitions based on conditions
def car (t : ℕ) (v : ℕ) : ℕ := t * v
def car_meet (t : ℕ) (v1 v2 : ℕ) : Prop := ∃ n, v1 * t + v2 * t = n

-- Given conditions
variables (v_A v_B v_C v_D : ℕ) (pairwise_different : v_A ≠ v_B ∧ v_B ≠ v_C ∧ v_C ≠ v_D ∧ v_D ≠ v_A)
variables (t1 t2 t3 : ℕ) (time_AC : t1 = 7) (time_BD : t1 = 7) (time_AB : t2 = 53)
variables (condition1 : car_meet t1 v_A v_C) (condition2 : car_meet t1 v_B v_D)
variables (condition3 : ∃ k, (v_A - v_B) * t2 = k)

-- Theorem statement
theorem cars_meet_first_time : ∃ t, (t = 371) := sorry

end cars_meet_first_time_l735_735841


namespace merlin_fired_probability_l735_735409

-- Definitions
variables (p : ℝ) (h_cond : 0 ≤ p ∧ p ≤ 1)
noncomputable def q := 1 - p

-- Question restatement
theorem merlin_fired_probability : (1 / 4 : ℝ) := by
  sorry

end merlin_fired_probability_l735_735409


namespace problem1_problem2_l735_735311

-- Problem 1
theorem problem1 (x : ℝ) (h1 : 2 * x > 1 - x) (h2 : x + 2 < 4 * x - 1) : x > 1 := 
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ)
  (h1 : (2 / 3) * x + 5 > 1 - x)
  (h2 : x - 1 ≤ (3 / 4) * x - 1 / 8) :
  -12 / 5 < x ∧ x ≤ 7 / 2 := 
by
  sorry

end problem1_problem2_l735_735311


namespace power_of_two_square_l735_735749

theorem power_of_two_square (n : ℕ) : ∃ k : ℕ, 2^6 + 2^9 + 2^n = k^2 ↔ n = 10 :=
by
  sorry

end power_of_two_square_l735_735749


namespace remainder_div_by_3_not_divisible_by_9_l735_735710

theorem remainder_div_by_3 (x : ℕ) (h : x = 1493826) : x % 3 = 0 :=
by sorry

theorem not_divisible_by_9 (x : ℕ) (h : x = 1493826) : x % 9 ≠ 0 :=
by sorry

end remainder_div_by_3_not_divisible_by_9_l735_735710


namespace equalize_balloons_l735_735384

def largest_prime_divisor (n : Nat) : Nat :=
  sorry -- Assume the existence of such a function

theorem equalize_balloons (n k : Nat) (k := largest_prime_divisor 2001) :
  (∀ (sizes : List ℕ), sizes.length = n → 
    (∃ (steps : ℕ), all_balloons_equal_after_steps sizes k steps)) :=
begin
  -- Problem statement
  sorry
end

end equalize_balloons_l735_735384


namespace distance_B_to_O_l735_735928

-- Lean definitions based on conditions and question.
variable {x : ℝ}

-- Assumptions based on the problem statement.
axiom right_triangle_equal_legs (AC BC : ℝ) (h_eq : AC = BC) : x = AC ∧ x = BC
axiom circle_diameter_AC (r : ℝ) (h_radius : r = AC / 2) : True
axiom point_M_on_AB (B M : ℝ) (h_BM : BM = sqrt 2) : True

-- Conclusion we need to prove.
theorem distance_B_to_O (A B C O : point) 
    (h_triangle : right_triangle_equal_legs AC BC h_eq) 
    (h_circle : circle_diameter_AC r h_radius) 
    (h_M : point_M_on_AB B M h_BM) : distance B O = sqrt 5 := by
  sorry

end distance_B_to_O_l735_735928


namespace simplify_expression_l735_735050

theorem simplify_expression : 
  ( (√98) + (√32) + (3√27) = (11 * √2) + 3 ) :=
by sorry

end simplify_expression_l735_735050


namespace min_value_of_reciprocal_sums_l735_735554

variable {a b : ℝ}

theorem min_value_of_reciprocal_sums (ha : a ≠ 0) (hb : b ≠ 0) (h : 4 * a ^ 2 + b ^ 2 = 1) :
  (1 / a ^ 2) + (1 / b ^ 2) = 9 := by
  sorry

end min_value_of_reciprocal_sums_l735_735554


namespace valid_systematic_sample_l735_735772

-- Definitions of the conditions
def is_systematic_sample (numbers : List ℕ) (interval total count : ℕ) : Prop :=
  ∀ n i, i < List.length numbers - 1 → numbers.get i + interval = numbers.get (i + 1)

def missile_numbers := List.range' 1 50  -- represents the range from 1 to 50

-- Stating the main theorem
theorem valid_systematic_sample : 
  is_systematic_sample [3, 13, 23, 33, 43] 10 50 5 :=
  sorry

end valid_systematic_sample_l735_735772


namespace factorization_l735_735479

def quadratic_expr (x : ℝ) : ℝ := 16 * x^2 - 40 * x + 25

theorem factorization (x : ℝ) : quadratic_expr x = (4 * x - 5)^2 := 
by sorry

end factorization_l735_735479


namespace law_school_applicants_l735_735992

theorem law_school_applicants 
  (ps : ℕ) (gpa_gt_3 : ℕ) (not_ps_gpa_leq_3 : ℕ) (ps_gpa_gt_3 : ℕ) :
  ps = 15 → gpa_gt_3 = 20 → not_ps_gpa_leq_3 = 10 → ps_gpa_gt_3 = 5 →
  ps + gpa_gt_3 - ps_gpa_gt_3 + not_ps_gpa_leq_3 = 40 := 
by
  intros h_ps h_gpa_gt_3 h_not_ps_gpa_leq_3 h_ps_gpa_gt_3
  rw [h_ps, h_gpa_gt_3, h_not_ps_gpa_leq_3, h_ps_gpa_gt_3]
  norm_num
  sorry

end law_school_applicants_l735_735992


namespace length_of_hypotenuse_l735_735380

variable {R : Type*} [Real R]

/-- Given a right triangle DEF with legs DE and DF, where points Z and W lie on DE and DF respectively, such that DZ:ZE = 1:3, DW:WF = 1:3, EW = 18 units, FW = 24 units, the length of the hypotenuse EF is 24 units. --/
theorem length_of_hypotenuse {DE DF EF DZ ZE DW WF EW FW : R} 
  (right_triangle : right_triangle DE DF EF) 
  (ratio_Z : DZ / ZE = 1 / 3) 
  (ratio_W : DW / WF = 1 / 3) 
  (length_EW : EW = 18) 
  (length_FW : FW = 24) : 
  EF = 24 := 
sorry

end length_of_hypotenuse_l735_735380


namespace distinct_sums_cardinality_l735_735856

theorem distinct_sums_cardinality {n : ℕ} (a : fin n → ℝ) (h : ∀ i j, i ≠ j → a i > 0 ∧ a i ≠ a j) :
  ∃ S : set ℝ, (∀ s ∈ S, ∃ t : finset (fin n), s = t.sum a) ∧ S.card ≥ n * (n + 1) / 2 := 
sorry

end distinct_sums_cardinality_l735_735856


namespace area_triangle_NMC_l735_735068

theorem area_triangle_NMC (x y : ℝ) (h : 1 - x - y = real.sqrt (x^2 + y^2)) :
  1 / 2 * (1 - x) * (1 - y) = 1 / 4 :=
by sorry

end area_triangle_NMC_l735_735068


namespace volume_of_solid_is_correct_l735_735344

noncomputable def volume_of_solid (v : ℝ × ℝ × ℝ) :=
  let ⟨x, y, z⟩ := v in
  (x^2 + y^2 + z^2 = 12 * x - 32 * y + 6 * z)

theorem volume_of_solid_is_correct :
  ∃ v : ℝ × ℝ × ℝ, volume_of_solid v → (4 / 3) * Real.pi * (301^(3/2)) := 
sorry

end volume_of_solid_is_correct_l735_735344


namespace boys_love_marbles_l735_735922

def total_marbles : ℕ := 26
def marbles_per_boy : ℕ := 2
def num_boys_love_marbles : ℕ := total_marbles / marbles_per_boy

theorem boys_love_marbles : num_boys_love_marbles = 13 := by
  rfl

end boys_love_marbles_l735_735922


namespace find_a_l735_735884

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x ≥ 0 then a * 2^x else Real.log (3 - x) / Real.log 2

theorem find_a :
  (∃ a : ℝ, (f a (f a (-1)) = 1)) → a = 1 / 4 :=
by
  sorry

end find_a_l735_735884


namespace f_has_unique_zero_on_Ici_max_value_a_for_increasing_g_l735_735880

-- Definitions based on conditions
def f (x : ℝ) : ℝ := (x - 2) * Real.log x + 2 * x - 3
def g (x a : ℝ) : ℝ := (x - a) * Real.log x + a * (x - 1) / x

-- Proof problems based on questions and correct answers
theorem f_has_unique_zero_on_Ici (f : ℝ → ℝ) (h : ∀ x, f x = (x - 2) * Real.log x + 2 * x - 3) :
  ∃! x, x ≥ 1 ∧ f x = 0 := sorry

theorem max_value_a_for_increasing_g (g : ℝ → ℝ → ℝ) (h : ∀ (x a : ℝ), g x a = (x - a) * Real.log x + a * (x - 1) / x) :
  ∀ {x}, g x 6 ≤ g x ↑ 6 := sorry

end f_has_unique_zero_on_Ici_max_value_a_for_increasing_g_l735_735880


namespace vector_CD_l735_735585

-- Define the vector space and the vectors a and b
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D : V)
variables (a b : V)

-- Define the conditions
def is_on_line (D A B : V) := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (D = t • A + (1 - t) • B)
def da_eq_2bd (D A B : V) := (A - D) = 2 • (D - B)

-- Define the triangle ABC and the specific vectors CA and CB
variables (CA := C - A) (CB := C - B)
variable (H1 : is_on_line D A B)
variable (H2 : da_eq_2bd D A B)
variable (H3 : CA = a)
variable (H4 : CB = b)

-- Prove the conclusion
theorem vector_CD (H1 : is_on_line D A B) (H2 : da_eq_2bd D A B)
  (H3 : CA = a) (H4 : CB = b) : 
  (C - D) = (1/3 : ℝ) • a + (2/3 : ℝ) • b :=
sorry

end vector_CD_l735_735585


namespace min_abs_a1_b1_l735_735799

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem min_abs_a1_b1 (a b : ℕ) (hab : a = 47 ∧ b = 43) :
  a + b = 90 ∧ abs (a - b) = 4 := 
by 
  sorry

end min_abs_a1_b1_l735_735799


namespace min_count_to_ensure_multiple_of_5_l735_735116

theorem min_count_to_ensure_multiple_of_5 (n : ℕ) (S : Finset ℕ) (hS : S = Finset.range 31) :
  25 ≤ S.card ∧ (∀ (T : Finset ℕ), T ⊆ S → T.card = 24 → ↑(∃ x ∈ T, x % 5 = 0)) :=
by sorry

end min_count_to_ensure_multiple_of_5_l735_735116


namespace expenses_denoted_as_negative_l735_735324

theorem expenses_denoted_as_negative (income_yuan expenses_yuan : Int) (h : income_yuan = 6) : 
  expenses_yuan = -4 :=
by
  sorry

end expenses_denoted_as_negative_l735_735324


namespace closest_root_of_quadratic_l735_735695

def quadratic_function (c : ℝ) (x : ℝ) := 3 * x ^ 2 - 6 * x + c

theorem closest_root_of_quadratic (c : ℝ)
  (h0 : quadratic_function c 0 = 2.24)
  (h2 : quadratic_function c 0.2 = 1.16)
  (h4 : quadratic_function c 0.4 = 0.32)
  (h6 : quadratic_function c 0.6 = -0.28)
  (h8 : quadratic_function c 0.8 = -0.64) :
  0.4 = closest_root (quadratic_function c) [0.2, 0.4, 0.6, 0.8] :=
sorry

end closest_root_of_quadratic_l735_735695


namespace area_of_gray_region_l735_735459

open Real

-- Define the circles and the radii.
def circleC_center : Prod Real Real := (5, 5)
def radiusC : Real := 5

def circleD_center : Prod Real Real := (15, 5)
def radiusD : Real := 5

-- The main theorem stating the area of the gray region bound by the circles and the x-axis.
theorem area_of_gray_region : 
  let area_rectangle := (10:Real) * (5:Real)
  let area_sectors := (2:Real) * ((1/4) * (5:Real)^2 * π)
  area_rectangle - area_sectors = 50 - 12.5 * π :=
by
  sorry

end area_of_gray_region_l735_735459


namespace sum_f_values_l735_735532

variable {R : Type*} [LinearOrder R]

noncomputable def f : R → R :=
sorry

axiom even_function : ∀ x, f (-x) = f x
axiom shifted_odd_function : ∀ x, f (-x + 1) = -f (x + 1)
axiom f_2_eq_3 : f 2 = 3

theorem sum_f_values : 
  f 0 + f 1 + f 2 + f 3 + (Finset.range 2013).sum (λ i, f (i + 4)) = -3 :=
sorry

end sum_f_values_l735_735532


namespace smallest_n_for_average_abs_diff_l735_735854

def abs_diff_nearest_square (k : ℕ) : ℕ :=
  let root := Int.sqrt k
  let square1 := root * root
  let square2 := (root + 1) * (root + 1)
  min (abs (k - square1)) (abs (k - square2))

theorem smallest_n_for_average_abs_diff : ∃ (n : ℕ), 
  (∀ (i : ℕ), 1 ≤ i → i ≤ n → ∑_{i = 1}^n abs_diff_nearest_square i) / n = 100 ∧
  n = 89601 := 
  by
    sorry

end smallest_n_for_average_abs_diff_l735_735854


namespace increasing_interval_cos_2alpha_l735_735547

noncomputable theory

open Real

-- Definition of the given function f
def f (x : ℝ) : ℝ := cos x * (sin x + cos x) - 1 / 2

-- Statement of the first proof: Interval where f(x) is monotonically increasing
theorem increasing_interval (k : ℤ) : 
  ∃ (a b : ℝ), [a, b] = [k * π - 3 * π / 8, k * π + π / 8] ∧ ∀ x y, 
  x ∈ [a, b] → y ∈ [a, b] → x < y → f x < f y := 
sorry

-- Statement of the second proof: value of cos(2α) given a specific condition
theorem cos_2alpha (α : ℝ) 
  (hα : α ∈ Ioo (π / 8) (3 * π / 8)) 
  (h : f α = sqrt 2 / 6) : 
  cos (2 * α) = (sqrt 2 - 4) / 6 := 
sorry

end increasing_interval_cos_2alpha_l735_735547


namespace equation_of_line_l735_735676

theorem equation_of_line
  (a : ℝ) 
  (h : a < 3)
  (l : ℝ → ℝ)
  (H : ∀ (x y : ℝ), (x^2 + y^2 + 2*x - 4*y + a = 0) → ((∃ k, l k = (x, y)) ∧ l (l k) = (0, 1))):
  l = (λ x : ℝ, x + 1) :=
by
  sorry

end equation_of_line_l735_735676


namespace sum_series_eq_one_l735_735794

theorem sum_series_eq_one : 
  (∑ k in Nat.range ∞, (∑ k = 1 to ∞, 12 ^ k / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1))))) = 1 := 
by sorry

end sum_series_eq_one_l735_735794


namespace only_odd_and_decreasing_l735_735040

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = - f x

def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x ≥ f y

def f1 (x : ℝ) : ℝ := x + 1
def f2 (x : ℝ) : ℝ := -x^2
def f3 (x : ℝ) : ℝ := 1 / x
def f4 (x : ℝ) : ℝ := -x * abs x

theorem only_odd_and_decreasing :
  odd_function f1 = false ∧
  odd_function f2 = false ∧
  (odd_function f3 ∧ decreasing_function f3 = false) ∧
  (odd_function f4 ∧ decreasing_function f4 = true) :=
sorry

end only_odd_and_decreasing_l735_735040


namespace complex_problem_solution_l735_735865

theorem complex_problem_solution (z : ℂ) (h : z = -⟨1 - complex.I, real.sqrt(2)⟩) :
  100 + z^100 = 99 :=
by sorry

end complex_problem_solution_l735_735865


namespace min_value_t2_t1_l735_735498

open Real

noncomputable def h (t : ℝ) : ℝ := 
  if t > 1 / 2 then exp (t - 1) else 0

noncomputable def g (t : ℝ) : ℝ := 
  if t > 1 / 2 then log (2 * t - 1) + 2 else 0

theorem min_value_t2_t1 : ∃ t1 t2 : ℝ, t1 > 1/2 ∧ t2 > 1/2 ∧ h t1 = g t2 ∧ (t2 - t1 = -log 2) :=
by
  sorry

end min_value_t2_t1_l735_735498


namespace factorial_division_identity_l735_735056

theorem factorial_division_identity : (10! / (6! * 4!)) = 210 := 
by
  -- We simplify using the properties of factorials and combinations.
  sorry

end factorial_division_identity_l735_735056


namespace find_number_l735_735083

theorem find_number (x : ℝ) (h : 0.6667 * x + 1 = 0.75 * x) : x = 12 :=
sorry

end find_number_l735_735083


namespace geometric_sequence_a2_value_l735_735605

theorem geometric_sequence_a2_value
  (a : ℕ → ℝ)
  (a1 a2 a3 : ℝ)
  (h1 : a 1 = a1)
  (h2 : a 2 = a2)
  (h3 : a 3 = a3)
  (h_pos : ∀ n, 0 < a n)
  (h_geo : ∀ n, a (n + 1) = a 1 * (a 2) ^ n)
  (h_sum : a1 + a2 + a3 = 18)
  (h_inverse_sum : 1/a1 + 1/a2 + 1/a3 = 2)
  : a2 = 3 :=
sorry

end geometric_sequence_a2_value_l735_735605


namespace triangle_inequality_cond_III_l735_735299

variable (a b c d : ℝ)

theorem triangle_inequality_cond_III :
  distinct_points A B C D E → 
  on_straight_line A B C D E →
  segment_length AB = a → 
  segment_length BD = b → 
  segment_length DE = c → 
  segment_length EC = d →
  a + d > b + c → 
  b + c > a + d → 
  a + b + c > d → 
  b < a + c :=
by
  sorry

end triangle_inequality_cond_III_l735_735299


namespace compound_interest_sum_l735_735690

noncomputable def simple_interest (P R T : ℝ) := (P * R * T) / 100

noncomputable def compound_interest (P R T : ℝ) := P * ((1 + R / 100) ^ T - 1)

theorem compound_interest_sum :
  let P := 2015.625 in
  let R_S := 12 in
  let T_S := 4 in
  let S_I := simple_interest P R_S T_S in
  let C_I := 2 * S_I in
  let R_C := 15 in
  let T_C := 2 in
  let P_C := 6000 in
  C_I = compound_interest P_C R_C T_C :=
by
  sorry

end compound_interest_sum_l735_735690


namespace lowest_probability_red_side_up_l735_735359

def card_flip_probability (k : ℕ) (n : ℕ) : ℚ :=
  if k ≤ n/2 then (n-k)^2/(n^2) + k^2/(n^2)
  else card_flip_probability (n+1-k) n 

theorem lowest_probability_red_side_up :
  (card_flip_probability 13 50) = (card_flip_probability 38 50) ∧ 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 25 → (card_flip_probability k 50 ≥ card_flip_probability 13 50)) :=
begin
  sorry
end

end lowest_probability_red_side_up_l735_735359


namespace discriminant_of_P_l735_735151

theorem discriminant_of_P (a b c : ℚ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1/2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1/2 := 
begin
  -- Proof omitted for brevity
  sorry
end

end discriminant_of_P_l735_735151


namespace complement_of_A_in_U_l735_735551

open Set

variable (U : Set ℕ) (A : Set ℕ)

theorem complement_of_A_in_U (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {2, 4, 6}) :
  (U \ A) = {1, 3, 5} := by 
  sorry

end complement_of_A_in_U_l735_735551


namespace rationalize_denominator_eqn_l735_735660

theorem rationalize_denominator_eqn : 
  let expr := (3 + Real.sqrt 2) / (2 - Real.sqrt 5)
  let rationalized := -6 - 3 * Real.sqrt 5 - 2 * Real.sqrt 2 - Real.sqrt 10
  let A := -6
  let B := -2
  let C := 2
  expr = rationalized ∧ A * B * C = -24 :=
by
  sorry

end rationalize_denominator_eqn_l735_735660


namespace set_properties_l735_735280

/-- Given sets A, B and C, prove the intersections and range conditions -/
theorem set_properties 
  (A : Set ℝ) (hA : A = {x | 2 < x ∧ x ≤ 3})
  (B : Set ℝ) (hB : B = {x | 1 < x ∧ x < 3})
  (C : ℝ → Set ℝ) (m : ℝ) (hC : C m = {x | m ≤ x}) :
  
  -- Part 1: Intersection of the complement of A with B.
  (Set.compl A ∩ B) = {x | 1 < x ∧ x ≤ 2} ∧  
  -- Part 2: Range condition for non-empty intersection of (A ∪ B) and C(m)
  (Set.Union A B ∩ C m).Nonempty ↔ m ≤ 3 :=
  
  begin
    sorry
  end

end set_properties_l735_735280


namespace johns_payment_l735_735954

-- Define the value of the camera
def camera_value : ℕ := 5000

-- Define the rental fee rate per week as a percentage
def rental_fee_rate : ℝ := 0.1

-- Define the rental period in weeks
def rental_period : ℕ := 4

-- Define the friend's contribution rate as a percentage
def friend_contribution_rate : ℝ := 0.4

-- Theorem: Calculate how much John pays for the camera rental
theorem johns_payment :
  let weekly_rental_fee := camera_value * rental_fee_rate
  let total_rental_fee := weekly_rental_fee * rental_period
  let friends_contribution := total_rental_fee * friend_contribution_rate
  let johns_payment := total_rental_fee - friends_contribution
  johns_payment = 1200 :=
by
  sorry

end johns_payment_l735_735954


namespace compatible_pairs_within_100_compatible_pairs_within_1000_with_one_greater_than_99_l735_735768

-- Helper functions to calculate sum and product of digits
def sum_of_digits (n: Nat) : Nat :=
  n.digits.sum

def product_of_digits (n: Nat) : Nat :=
  n.digits.foldl (· * ·) 1

-- Condition for compatibility
def are_compatible (a b : Nat) : Prop :=
  (a = sum_of_digits b) ∧ (b = product_of_digits a)

-- The main theorem statement for part 1
theorem compatible_pairs_within_100 :
  { (a, b) : Nat × Nat | a < 100 ∧ b < 100 ∧ are_compatible a b } = 
    {(9, 11), (12, 36)} :=
  sorry

-- The main theorem statement for part 2
theorem compatible_pairs_within_1000_with_one_greater_than_99 :
  { (a, b) : Nat × Nat | a < 1000 ∧ b < 1000 ∧ (a > 99 ∨ b > 99) ∧ are_compatible a b } =
    {(135, 19), (144, 19)} :=
  sorry

end compatible_pairs_within_100_compatible_pairs_within_1000_with_one_greater_than_99_l735_735768


namespace compute_value_l735_735968

noncomputable def y : ℂ := complex.exp (complex.I * (2 * real.pi / 9))

theorem compute_value :
  let y := complex.exp (complex.I * (2 * real.pi / 9)) in
  (2 * y + y^2) * 
  (2 * y^2 + y^4) * 
  (2 * y^3 + y^6) * 
  (2 * y^4 + y^8) * 
  (2 * y^5 + y^10) * 
  (2 * y^6 + y^12) = 43 := sorry

end compute_value_l735_735968


namespace sports_club_membership_l735_735929

theorem sports_club_membership :
  (17 + 21 - 10 + 2 = 30) :=
by
  sorry

end sports_club_membership_l735_735929


namespace range_of_m_l735_735895

noncomputable def quadratic_function (m x : ℝ) : ℝ := (m-2) * x^2 + 2 * m * x - (3 - m)

theorem range_of_m (m : ℝ) (h_vertex_third_quadrant : (-(m) / (m-2) < 0) ∧ ((-5)*m + 6) / (m-2) < 0)
                   (h_parabola_opens_upwards : m - 2 > 0)
                   (h_intersects_negative_y_axis : m < 3) : 2 < m ∧ m < 3 :=
by {
    sorry
}

end range_of_m_l735_735895


namespace denominator_exceeds_numerator_by_13_l735_735961

-- Define the repeating decimal F
def F : ℚ := 86 / 99

-- State the theorem about the difference between numerator and denominator
theorem denominator_exceeds_numerator_by_13 : 
  let num_den_diff := F.denom - F.num in num_den_diff = 13 :=
sorry

end denominator_exceeds_numerator_by_13_l735_735961


namespace find_y_l735_735237

def ABC_is_straight_line : Prop := True -- Placeholder as actual geometric representation is complex.

def angle_ABD : ℝ := 120
def angle_CBD : ℝ := 54
def exterior_angle_theorem (x y : ℝ) : Prop := x = y + 54

theorem find_y :
  ∀ (y : ℝ), ABC_is_straight_line → 
    angle_ABD = 120 → 
    angle_CBD = 54 → 
    exterior_angle_theorem angle_ABD y → 
    y = 66 :=
by
  intro y _ h₁ h₂ h₃
  sorry

end find_y_l735_735237


namespace solve_inequality_l735_735416

theorem solve_inequality (a x : ℝ) (h : a > 0 ∧ a ≠ 1) :
  x^Real.log a x > x^(9/2) / a^2 ↔ 
  if a > 1 then (x > a^4 ∨ 0 < x ∧ x < a^(1/2)) else (a^4 < x ∧ x < a^(1/2)) :=
sorry

end solve_inequality_l735_735416


namespace kris_suspension_days_per_instance_is_three_l735_735618

-- Define the basic parameters given in the conditions
def total_fingers_toes : ℕ := 20
def total_bullying_instances : ℕ := 20
def multiplier : ℕ := 3

-- Define total suspension days according to the conditions
def total_suspension_days : ℕ := multiplier * total_fingers_toes

-- Define the goal: to find the number of suspension days per instance
def suspension_days_per_instance : ℕ := total_suspension_days / total_bullying_instances

-- The theorem to prove that Kris was suspended for 3 days per instance
theorem kris_suspension_days_per_instance_is_three : suspension_days_per_instance = 3 := by
  -- Skip the actual proof, focus only on the statement
  sorry

end kris_suspension_days_per_instance_is_three_l735_735618


namespace exists_n_plus_Sn_eq_1980_l735_735982

def S (n : ℕ) : ℕ :=
  n.digits.sum

theorem exists_n_plus_Sn_eq_1980 : ∃ n : ℕ, n + S n = 1980 :=
sorry

end exists_n_plus_Sn_eq_1980_l735_735982


namespace half_angle_in_third_quadrant_l735_735862

theorem half_angle_in_third_quadrant
  (α : ℝ) (k : ℤ)
  (h1 : 2 * k * π < α ∧ α < 2 * k * π + π / 2)
  (h2 : |cos (α / 2)| = -cos (α / 2)) :
  k * π < α / 2 ∧ α / 2 < k * π + π / 2 :=
by
  sorry

end half_angle_in_third_quadrant_l735_735862


namespace alpha_beta_sum_l735_735530

-- Definitions based on the given conditions
def quadratic_roots (a b c x : ℝ) : Prop := (a * x^2 + b * x + c = 0)

-- Explicitly stating the conditions
def tan_alpha_beta_roots (α β : ℝ) : Prop :=
  quadratic_roots 1 (3 * real.sqrt 3) 4 (real.tan α) ∧
  quadratic_roots 1 (3 * real.sqrt 3) 4 (real.tan β)

-- Stating the conditions about the interval
def in_interval (x : ℝ) : Prop :=
  x > -real.pi / 2 ∧ x < real.pi / 2

-- Theorem statement
theorem alpha_beta_sum (α β : ℝ) (h1 : tan_alpha_beta_roots α β) (h2 : in_interval α) (h3 : in_interval β) :
  α + β = -2 * real.pi / 3 :=
begin
  sorry
end

end alpha_beta_sum_l735_735530


namespace half_angle_quadrants_l735_735527

variable (k : ℤ) (α : ℝ)

-- Conditions
def is_second_quadrant (α : ℝ) (k : ℤ) : Prop :=
  2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi

-- Question: Determine the quadrant(s) in which α / 2 lies under the given condition.
theorem half_angle_quadrants (α : ℝ) (k : ℤ) 
  (h : is_second_quadrant α k) : 
  ((k * Real.pi + Real.pi / 4 < α / 2) ∧ (α / 2 < k * Real.pi + Real.pi / 2)) ↔ 
  (∃ (m : ℤ), (2 * m * Real.pi < α / 2 ∧ α / 2 < 2 * m * Real.pi + Real.pi)) ∨ ( ∃ (m : ℤ), (2 * m * Real.pi + Real.pi < α / 2 ∧ α / 2 < 2 * m * Real.pi + 2 * Real.pi)) := 
sorry

end half_angle_quadrants_l735_735527


namespace part_I_part_II_l735_735123

noncomputable def f (x : ℝ) (a b : ℝ) := 2 * x^2 - 2 * a * x + b

def f_vertex_condition (a b : ℝ) := f (-1) a b = -8

def vertex_form (x : ℝ) := 2 * (x + 1)^2 - 8

def A (x : ℝ) : set ℝ := { x | f x 1 16 > 0 }

def B (t : ℝ) : set ℝ := { x | abs (x - t) ≤ 1 }

def complement (s : set ℝ) : set ℝ := { x | x ∉ s }

theorem part_I :
  (complement (A 1.0)) ∪ (B 1.0) = {x | -3 ≤ x ∧ x ≤ 2} := 
by sorry

theorem part_II (t : ℝ) : 
  (∀ x, x ∈ A 1.0 → x ∉ B t) → (-2 ≤ t ∧ t ≤ 0) := 
by sorry

end part_I_part_II_l735_735123


namespace area_rectangle_l735_735996

/-- Given parameters for the rectangle inscribed in a semicircle -/
variables (DA FD : ℕ) (r : ℕ) (AB : ℝ)

-- Conditions derived from the problem statement
axiom DA_Val : DA = 20
axiom FD_Val : FD = 12
axiom r_Val : r = 22

-- Define the Pythagorean relationship derived from the problem condition
axiom AB_Val : AB = real.sqrt (r * r - FD * FD)

-- Assertion for the area of the rectangle
theorem area_rectangle : (DA : ℝ) * AB = 20 * real.sqrt 340 :=
by 
  rw [DA_Val, AB_Val, real.sqrt_inj (by norm_num : 0 ≤ 340)],
  exact sorry

end area_rectangle_l735_735996


namespace find_value_l735_735118

-- Define the constants and conditions
variable (a b : ℝ)
variable (h1 : 3^a = 5)
variable (h2 : 3^b = 8)

-- State the problem as a theorem
theorem find_value : 3^(3 * a - 2 * b) = 125 / 64 :=
by {
  sorry
}

end find_value_l735_735118


namespace pebbles_in_10th_picture_l735_735046

theorem pebbles_in_10th_picture :
  (∃ f : ℕ → ℕ, f 0 = 1 ∧ f 1 = 5 ∧ (∀ n : ℕ, f (n + 1) = f n + 3 * (n + 1) - 2)) →
  f 9 = 145 :=
by
  sorry

end pebbles_in_10th_picture_l735_735046


namespace angle_ratio_l735_735939

theorem angle_ratio (A B C P Q M : Point) (h1 : Trisects (∠ A C B) (∠ A C P) (∠ P C Q) (∠ Q C B))
  (h2 : Bisects (∠ P C Q) (∠ M C P) (∠ M C Q)) :
  (measure (∠ M C Q)) / (measure (∠ A C Q)) = 1 / 4 := by
  sorry

end angle_ratio_l735_735939


namespace move_line_up_l735_735224

theorem move_line_up (x : ℝ) :
  let y_initial := 4 * x - 1
  let y_moved := y_initial + 2
  y_moved = 4 * x + 1 :=
by
  let y_initial := 4 * x - 1
  let y_moved := y_initial + 2
  show y_moved = 4 * x + 1
  sorry

end move_line_up_l735_735224


namespace area_of_trapezoid_l735_735321

variables (A B C D : Type) [linear_ordered_field A] [linear_ordered_field B] [linear_ordered_field C] [linear_ordered_field D]

def circle_radius := 5 : ℝ 
def longer_base := 10 : ℝ
def shorter_base := 6 : ℝ
def height (AH DH : ℝ) : ℝ := sqrt (AH * DH)

theorem area_of_trapezoid 
  (AH DH CH : ℝ) 
  (circumcircle_center_on_longer_base : longer_base = circle_radius * 2)
  (shorter_base_eq_6 : shorter_base = 6)
  (AH_eq : AH = (longer_base + shorter_base) / 2)
  (DH_eq : DH = (longer_base - shorter_base) / 2)
  (CH_eq : CH = height AH DH) : 
  (1/2) * (shorter_base + longer_base) * CH = 32 := 
sorry

end area_of_trapezoid_l735_735321


namespace tetrahedron_AC_value_l735_735596

variables (A B C D : Type)
variables [inner_product_space ℝ A]

noncomputable def AC_distance (AB BD CD : ℝ) (angle_ABD_BCD : ℝ) : ℝ :=
  if angle_ABD_BCD = real.pi / 3 then 
    (AB^2 + BD^2 + CD^2 + 2 * AB * BD * real.cos (real.pi / 3) + 2 * AB * CD * real.cos (real.pi / 3) + 2 * BD * CD * real.cos (real.pi / 3)).sqrt
  else
    (AB^2 + BD^2 + CD^2 + 2 * AB * BD * real.cos (2 * real.pi / 3) + 2 * AB * CD * real.cos (2 * real.pi / 3) + 2 * BD * CD * real.cos (2 * real.pi / 3)).sqrt

axiom given_conditions : AB ⊥ BD ∧ CD ⊥ BD ∧ AB = 3 ∧ BD = 2 ∧ CD = 4 ∧ angle_between_planes = real.pi / 3

theorem tetrahedron_AC_value :
  let AC := AC_distance 3 2 4 (real.pi / 3)
  in AC = real.sqrt 17 :=
by
  sorry

end tetrahedron_AC_value_l735_735596


namespace sin_alpha_plus_pi_over_4_equals_sqrt2_div_3_angle_between_OB_OC_l735_735977

-- Problem 1
theorem sin_alpha_plus_pi_over_4_equals_sqrt2_div_3
  (α : ℝ)
  (h_dot : let A := (3, 0) in let B := (0, 3) in let C := (Real.cos α, Real.sin α) in
          (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = -1) :
  Real.sin (α + Real.pi / 4) = Real.sqrt 2 / 3 :=
sorry

-- Problem 2
theorem angle_between_OB_OC
  (α : ℝ)
  (h_condition : let A := (3, 0) in let C := (Real.cos α, Real.sin α) in
                 Real.sqrt ((C.1 - A.1) ^ 2 + (C.2 - A.2) ^ 2) = Real.sqrt 13)
  (h_interval : 0 < α ∧ α < Real.pi) :
  let B := (0, 3) in
  let angle := α - Real.pi / 2 in
  angle = Real.pi / 6 :=
sorry

end sin_alpha_plus_pi_over_4_equals_sqrt2_div_3_angle_between_OB_OC_l735_735977


namespace log_base_change_l735_735477

theorem log_base_change (h : 1 / 32 = 2 ^ (-5)) : log 3 (1 / 32) = -5 * log 3 2 :=
by
  sorry

end log_base_change_l735_735477


namespace unique_integer_solution_l735_735215

theorem unique_integer_solution (n : ℤ) : 
  (\left\lfloor (n^2 + 6 : ℤ) / 4 \right\rfloor - \left\lfloor n / 2 \right\rfloor ^ 2 = 2) → 
  n = 3 := 
by
  sorry

end unique_integer_solution_l735_735215


namespace increasing_interval_f_l735_735680

noncomputable def f (x : ℝ) : ℝ := (x^2) / (Real.exp x)

theorem increasing_interval_f : ∀ x, 0 ≤ x ∧ x ≤ 2 → (f x).derivative ≥ 0 :=
by
  sorry

end increasing_interval_f_l735_735680


namespace tangent_slope_k_l735_735916

theorem tangent_slope_k :
  ∃ x0 : ℝ, (k = 3 * x0^2 - 6 * x0 + 2) ∧ ((0, 0) ∈ set_of (λ p : ℝ × ℝ, p.2 = (3 * x0^2 - 6 * x0 + 2) * p.1 - (3 * x0^2 - 6 * x0 + 2) * x0 + (x0^3 - 3 * x0^2 + 2 * x0)))
  →
  k = 2 ∨ k = - (1 / 4) :=
by
  sorry

end tangent_slope_k_l735_735916


namespace conference_session_time_l735_735013

def conference_duration_hours : ℕ := 8
def conference_duration_minutes : ℕ := 45
def break_time : ℕ := 30

theorem conference_session_time :
  (conference_duration_hours * 60 + conference_duration_minutes) - break_time = 495 :=
by sorry

end conference_session_time_l735_735013


namespace minimum_x_value_of_quadratic_l735_735061

theorem minimum_x_value_of_quadratic (p : ℝ) (hp : 0 < p) : 
  ∃ x : ℝ, x = -(3 * p) / 2 ∧ ∀ y, y = x^2 + 3 * p * x + 2 * p^2 → y = (-(3 * p) / 2)^2 + 3 * p * (-(3 * p) / 2) + 2 * p^2
  sorry

end minimum_x_value_of_quadratic_l735_735061


namespace evaluate_powers_of_i_l735_735077

theorem evaluate_powers_of_i : (complex.I ^ 45) + (complex.I ^ 345) = 2 * complex.I :=
by
  -- The proof will go here
  sorry

end evaluate_powers_of_i_l735_735077


namespace minimal_flip_probability_l735_735363

def flip_probability (n : ℕ) (k : ℕ) : ℚ :=
  if k <= 25 then
    (2 * k^2 - 52 * k + 676) / 676
  else
    let mirrored_k := 51 - k in
    (2 * mirrored_k^2 - 52 * mirrored_k + 676) / 676

theorem minimal_flip_probability :
  ∀ k, (13 ≤ k ∧ k ≤ 13) ∨ (38 ≤ k ∧ k ≤ 38) :=
by
  intro k
  sorry

end minimal_flip_probability_l735_735363


namespace X_set_equals_Y_set_l735_735410

-- Define the problem
def blue_red_points (n : ℕ) :=
  { p : ℕ × ℕ // p.1 + p.2 < n }

def is_red (n : ℕ) (red_points : set (ℕ × ℕ)) (p : ℕ × ℕ) : Prop :=
  (p ∈ red_points) ∧ (∀ q ∈ red_points, q.1 ≤ p.1 ∧ q.2 ≤ p.2)

def X_set (n : ℕ) (blue_points : set (ℕ × ℕ)) : Prop :=
  ∃ (points : fin n → ℕ × ℕ), (∀ i j, i ≠ j → (points i).1 ≠ (points j).1) ∧
  ∀ i, points i ∈ blue_points

def Y_set (n : ℕ) (blue_points : set (ℕ × ℕ)) : Prop :=
  ∃ (points : fin n → ℕ × ℕ), (∀ i j, i ≠ j → (points i).2 ≠ (points j).2) ∧
  ∀ i, points i ∈ blue_points

theorem X_set_equals_Y_set (n : ℕ) (red_points blue_points : set (ℕ × ℕ)) :
  (∀ p ∈ blue_points, ∃ (x1 x2), red_points (x1, x2) → x1 > p.1 ∧ x2 > p.2) →
  (∀ x (y1 y2), y1 ≠ y2 → (x, y1) ∈ blue_points → (x, y2) ∈ blue_points → false) →
  X_set n blue_points → Y_set n blue_points :=
by
  sorry

end X_set_equals_Y_set_l735_735410


namespace percent_of_motorists_receive_speeding_tickets_l735_735289

theorem percent_of_motorists_receive_speeding_tickets
    (p_exceed : ℝ)
    (p_no_ticket : ℝ)
    (h1 : p_exceed = 0.125)
    (h2 : p_no_ticket = 0.20) : 
    (0.8 * p_exceed) * 100 = 10 :=
by
  sorry

end percent_of_motorists_receive_speeding_tickets_l735_735289


namespace cards_least_likely_red_after_flips_l735_735352

theorem cards_least_likely_red_after_flips :
  ∃ (k1 k2 : ℕ), 1 ≤ k1 ∧ k1 ≤ 50 ∧ 1 ≤ k2 ∧ k2 ≤ 50 ∧ (k1 = 13 ∧ k2 = 38) ∧ 
  (∀ k ∈ finset.range 1 51, 
    let p := (if k ≤ 25 then ((26 - k) ^ 2 + k ^ 2) / 676 else ((26 - (51 - k)) ^ 2 + (51 - k) ^ 2) / 676) in
    p ≥ (if k = 13 ∨ k = 38 then ((26 - k) ^ 2 + k ^ 2) / 676 else p)) :=
sorry

end cards_least_likely_red_after_flips_l735_735352


namespace sin_double_angle_pi_six_l735_735117

theorem sin_double_angle_pi_six (α : ℝ)
  (h : 2 * Real.sin α = 1 + 2 * Real.sqrt 3 * Real.cos α) :
  Real.sin (2 * α - Real.pi / 6) = 7 / 8 :=
sorry

end sin_double_angle_pi_six_l735_735117


namespace partition_triangles_l735_735657

-- Define the 27 triangles and their areas
noncomputable def areas : Fin 27 → ℝ := sorry

-- The main theorem statement
theorem partition_triangles :
  ∃ (group1 group2 : Finset (Fin 27)),
    (group1 ∪ group2 = Finset.univ) ∧
    (group1 ∩ group2 = ∅) ∧
    (∑ i in group1, areas i = ∑ i in group2, areas i) :=
sorry

end partition_triangles_l735_735657


namespace sum_of_perfect_square_divisors_is_infinite_l735_735097

theorem sum_of_perfect_square_divisors_is_infinite (a b : ℤ) :
  let n := complex.mk a b in
  let n_squared := n * n in
  let product := 544 * n_squared in
  ∀ s, s ∈ divisors product → is_perfect_square s → false :=
by sorry

end sum_of_perfect_square_divisors_is_infinite_l735_735097


namespace sqrt_eq_cond_l735_735483

theorem sqrt_eq_cond (x : ℝ) (h : x > 9) : (sqrt (x - 4 * sqrt (x - 9)) + 3 = sqrt (x + 4 * sqrt (x - 9)) - 3) ↔ (x ∈ set.Ici 12) :=
by
  sorry

end sqrt_eq_cond_l735_735483


namespace smallest_multiple_63_odd_ones_l735_735461

/-- 
  This proof statement verifies that the smallest multiple of 63 
  with an odd number of ones in its binary representation is 4221.
-/
theorem smallest_multiple_63_odd_ones : 
  ∃ (n : ℕ), n % 63 = 0 ∧ 
             (nat_popcount n % 2 = 1) ∧ 
             (∀ (m : ℕ), m % 63 = 0 ∧ nat_popcount m % 2 = 1 → n ≤ m) ∧ 
             n = 4221 :=
sorry

end smallest_multiple_63_odd_ones_l735_735461


namespace smallest_divisor_l735_735488

theorem smallest_divisor (k n : ℕ) (x y : ℤ) :
  (∃ n : ℕ, k ∣ 2^n + 15) ∧ (∃ x y : ℤ, k = 3 * x^2 - 4 * x * y + 3 * y^2) → k = 23 := by
  sorry

end smallest_divisor_l735_735488


namespace cleaner_for_dog_stain_l735_735645

theorem cleaner_for_dog_stain (D : ℝ) (H : 6 * D + 3 * 4 + 1 * 1 = 49) : D = 6 :=
by 
  -- Proof steps would go here, but we are skipping the proof.
  sorry

end cleaner_for_dog_stain_l735_735645


namespace alpha_beta_sum_pi_over_2_l735_735558

theorem alpha_beta_sum_pi_over_2 (α β : ℝ) (hα : 0 < α) (hα_lt : α < π / 2) (hβ : 0 < β) (hβ_lt : β < π / 2) (h : Real.sin (α + β) = Real.sin α ^ 2 + Real.sin β ^ 2) : α + β = π / 2 :=
by
  -- Proof steps would go here
  sorry

end alpha_beta_sum_pi_over_2_l735_735558


namespace evaluate_expression_l735_735478

theorem evaluate_expression : (733 * 733) - (732 * 734) = 1 :=
by
  sorry

end evaluate_expression_l735_735478


namespace find_remainder_q_neg2_l735_735714

-- Define q(x)
def q (x : ℝ) (D E F : ℝ) : ℝ := D * x^4 + E * x^2 + F * x + 6

-- The given conditions in the problem
variable {D E F : ℝ}
variable (h_q_2 : q 2 D E F = 14)

-- The statement we aim to prove
theorem find_remainder_q_neg2 (h_q_2 : q 2 D E F = 14) : q (-2) D E F = 14 :=
sorry

end find_remainder_q_neg2_l735_735714


namespace intersection_of_planes_dividing_space_into_six_parts_l735_735717

theorem intersection_of_planes_dividing_space_into_six_parts 
    (scenario1 : ∃ (P1 P2 P3 : Plane), (P1 ∥ P2) ∧ ¬(P3 ∥ P1) ∧ ¬(P3 ∥ P2))
    (scenario2 : ∃ (P1 P2 P3 : Plane), ∃ L : Line, (L ∈ P1 ∧ L ∈ P2 ∧ L ∈ P3)) :
    (1 = 1 ∨ 2 = 2) → (lines_of_intersection scenario1 = 1 ∨ lines_of_intersection scenario1 = 2) :=
by
  sorry

end intersection_of_planes_dividing_space_into_six_parts_l735_735717


namespace interval_intersection_l735_735092

theorem interval_intersection (x : ℝ) :
  (4 * x > 2 ∧ 4 * x < 3) ∧ (5 * x > 2 ∧ 5 * x < 3) ↔ (x > 1/2 ∧ x < 3/5) :=
by
  sorry

end interval_intersection_l735_735092


namespace laurent_series_expansion_l735_735818

theorem laurent_series_expansion (f : ℂ → ℂ) (z : ℂ) (h : |z + 1| > 2) :
  f z = ∑ k in (Finset.range ∞), (λ k, ((-2 : ℂ) ^ k + 1) / (z + 1) ^ (k + 1)) :=
  sorry

end laurent_series_expansion_l735_735818


namespace new_circumference_of_circle_l735_735774

theorem new_circumference_of_circle (w h : ℝ) (d_multiplier : ℝ) 
  (h_w : w = 7) (h_h : h = 24) (h_d_multiplier : d_multiplier = 1.5) : 
  (π * (d_multiplier * (Real.sqrt (w^2 + h^2)))) = 37.5 * π :=
by
  sorry

end new_circumference_of_circle_l735_735774


namespace tangent_point_l735_735611

noncomputable def f (x : ℝ) (p : ℝ) := x^p
noncomputable def g (x : ℝ) := Real.log x

theorem tangent_point (p : ℝ) (x y : ℝ) 
  (h_tangent_values: f x p = g x) 
  (h_tangent_deriv: (deriv (λ x, f x p) x) = (deriv g x))
  : p = 1 / Real.exp 1 ∧ x = Real.exp (1 / p) ∧ y = Real.exp 1 :=
by
  -- Proof to be filled in
  sorry

end tangent_point_l735_735611


namespace quadratic_polynomial_discriminant_l735_735136

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₁ : a ≠ 0)
  (h₂ : ∃ x : ℝ, a * x^2 + b * x + c = x - 2 ∧ (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h₃ : ∃ x : ℝ, a * x^2 + b * x + c = 1 - x / 2 ∧ (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l735_735136


namespace profit_percentage_l735_735729

theorem profit_percentage (cost_price selling_price : ℝ) (h₁ : cost_price = 32) (h₂ : selling_price = 56) : 
  ((selling_price - cost_price) / cost_price) * 100 = 75 :=
by
  sorry

end profit_percentage_l735_735729


namespace sum_abcd_eq_42_l735_735338

variable {a b c d : ℕ}

theorem sum_abcd_eq_42
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_prods : ∃ (p1 p2 p3 p4 : ℕ), 
              {p1, p2, p3, p4} = {64, 88, 120, 165} ∧ 
              (p1 = a * b ∧ p2 = b * c ∧ p3 = c * d ∧ p4 = d * a ∨ 
              p1 = a * b ∧ p2 = b * c ∧ p3 = d * a ∧ p4 = c * d ∨
              -- Include all 24 permutations here --
              )) : 
  a + b + c + d = 42 :=
sorry

end sum_abcd_eq_42_l735_735338


namespace many_people_sharing_car_l735_735043

theorem many_people_sharing_car (x y : ℤ) 
  (h1 : 3 * (y - 2) = x) 
  (h2 : 2 * y + 9 = x) : 
  3 * (y - 2) = 2 * y + 9 := 
by
  -- by assumption h1 and h2, we already have the setup, refute/validate consistency
  sorry

end many_people_sharing_car_l735_735043


namespace fencing_rate_correct_l735_735089

noncomputable def rate_per_meter (d : ℝ) (cost : ℝ) : ℝ :=
  cost / (Real.pi * d)

theorem fencing_rate_correct : rate_per_meter 26 122.52211349000194 = 1.5 := by
  sorry

end fencing_rate_correct_l735_735089


namespace perpendicular_lines_l735_735912

theorem perpendicular_lines {a : ℝ} :
  a*(a-1) + (1-a)*(2*a+3) = 0 → (a = 1 ∨ a = -3) := 
by
  intro h
  sorry

end perpendicular_lines_l735_735912


namespace num_funcs_satisfying_condition_l735_735628

noncomputable def A : Set ℂ := {1, -1, Complex.I, -Complex.I}

def satisfies_condition (f : ℂ → ℂ) : Prop :=
  ∀ x y ∈ A, f (x * y) = f x * f y

def is_function_modulo_A (f : ℂ → ℂ) : Prop :=
  ∀ x ∈ A, f x ∈ A

theorem num_funcs_satisfying_condition : 
  (∃ f : ℂ → ℂ, is_function_modulo_A f ∧ satisfies_condition f ∧ (Finset.univ.filter (λ f : ℂ → ℂ, is_function_modulo_A f ∧ satisfies_condition f)).card = 2) := 
sorry

end num_funcs_satisfying_condition_l735_735628


namespace student_arrangements_l735_735004

def num_arrangements (n : ℕ) (not_far_left : ℕ) (not_far_right : ℕ) : ℕ :=
  let total_permutations := n!
  let invalid_arrangements_far_left := 2 * ((n - 1)!) -- A at far left, B at far right
  let invalid_arrangements_far_right := ((n - 2)!) -- A at far left and B also at far right
  total_permutations - invalid_arrangements_far_left + invalid_arrangements_far_right

theorem student_arrangements :
  num_arrangements 5 1 1 = 78 :=
by
sorry

end student_arrangements_l735_735004


namespace ten_exp_eq_25_then_ten_exp_sub_eq_two_l735_735571

theorem ten_exp_eq_25_then_ten_exp_sub_eq_two (x : ℝ) (h : 10^(2 * x) = 25) : 10^(1 - x) = 2 :=
by
  sorry

end ten_exp_eq_25_then_ten_exp_sub_eq_two_l735_735571


namespace systematic_sampling_firefighters_l735_735763

theorem systematic_sampling_firefighters :
  ∃ (a b c : ℕ), a = 25 ∧ b = 17 ∧ c = 8 ∧
    (∃ seq : ℕ → ℕ, (∀ n : ℕ, seq n = 3 + 12 * (n - 1) ∧
      (∃ total : ℕ, total = 600 ∧
        (∃ sample_size : ℕ, sample_size = 50 ∧ 
          (∀ n : ℕ, 1 ≤ seq n ∧ seq n ≤ total) ∧
          a = count_in_interval seq 1 300 ∧ 
          b = count_in_interval seq 301 495 ∧
          c = count_in_interval seq 496 600
)))) :=
begin
  existsi 25, existsi 17, existsi 8,
  repeat {split},
  { sorry },
  { existsi (λ n, 3 + 12 * (n - 1)),
    intros n,
    split,
    { sorry },
    existsi 600,
    split,
    { sorry },
    existsi 50,
    split,
    { sorry },
    { intros n,
      split,
      { sorry },
      { sorry },
    },
    { sorry },
    { sorry },
    { sorry }
  }
end

end systematic_sampling_firefighters_l735_735763


namespace triangle_AD_eq_8sqrt2_l735_735943

/-- Given a triangle ABC where AB = 13, AC = 20, and
    D is the foot of the perpendicular from A to BC,
    with the ratio BD : CD = 3 : 4, prove that AD = 8√2. -/
theorem triangle_AD_eq_8sqrt2 
  (AB AC : ℝ) (BD CD AD : ℝ) 
  (h₁ : AB = 13)
  (h₂ : AC = 20)
  (h₃ : BD / CD = 3 / 4)
  (h₄ : BD^2 = AB^2 - AD^2)
  (h₅ : CD^2 = AC^2 - AD^2) :
  AD = 8 * Real.sqrt 2 :=
by
  sorry

end triangle_AD_eq_8sqrt2_l735_735943


namespace sum_of_simplified_fraction_l735_735716

theorem sum_of_simplified_fraction : 
  let num := 49
  let denom := 84
  let gcd := Nat.gcd num denom
  let simple_num := num / gcd
  let simple_denom := denom / gcd
  simple_num + simple_denom = 19 := 
by
  let num := 49
  let denom := 84
  have gcd_pos : gcd > 0 := Nat.gcd_pos_of_pos_left num denom (by norm_num)
  have h_gcd : gcd = 7 := by norm_num
  have h_simple_num : simple_num = 49 / 7 := by rw [h_gcd, Nat.div_self gcd_pos]
  have h_simple_denom : simple_denom = 84 / 7 := by rw [h_gcd, Nat.div_self gcd_pos]
  have h_simple_num_val : simple_num = 7 := by norm_num [h_simple_num]
  have h_simple_denom_val : simple_denom = 12 := by norm_num [h_simple_denom]
  rw [h_simple_num_val, h_simple_denom_val]
  norm_num

end sum_of_simplified_fraction_l735_735716


namespace projection_impossible_as_triangle_l735_735725

theorem projection_impossible_as_triangle 
  (rect : Type) 
  (projection : rect → Type) 
  (cardboard : rect) 
  (conditions : ∀ (orientation : rect), 
    orientation = "parallel to sunlight" → projection orientation = "line segment" ∧
    orientation = "parallel to ground" → projection orientation = "rectangle" ∧
    orientation = "inclined" → projection orientation = "parallelogram") : 
  ¬ ∃ (orientation : rect), projection orientation = "triangle" :=
by sorry

end projection_impossible_as_triangle_l735_735725


namespace minimum_distance_proof_l735_735838

noncomputable def minimize_distance : ℝ := 1

noncomputable def point_on_curve : ℝ × ℝ :=
(1 - real.sqrt 2 / 2, -real.sqrt 2 / 2)

theorem minimum_distance_proof :
  ∃ θ : ℝ, (1 + real.cos θ, real.sin θ) = point_on_curve ∧
  (∀t:ℝ, abs (((1 + real.cos θ) + real.sin θ + 2 * real.sqrt 2 - (1 + t / 2) + (1 - t / 2)) / real.sqrt 2) = minimize_distance) :=
begin
  sorry
end

end minimum_distance_proof_l735_735838


namespace hyperbolas_count_l735_735108

def binomial (m n : ℕ) : ℕ :=
  Nat.factorial m / (Nat.factorial n * Nat.factorial (m - n))

theorem hyperbolas_count :
  (Finset.card (Finset.filter
    (λ n : ℕ × ℕ, 1 ≤ n.2 ∧ n.2 ≤ n.1 ∧ n.1 ≤ 5 ∧ binomial n.1 n.2 > 1)
    (Finset.product (Finset.range 6) (Finset.range 6)))) = 6 :=
by
  sorry

end hyperbolas_count_l735_735108


namespace cost_of_each_candy_bar_l735_735379

-- Definitions of the conditions
def initial_amount : ℕ := 20
def final_amount : ℕ := 12
def number_of_candy_bars : ℕ := 4

-- Statement of the proof problem: prove the cost of each candy bar
theorem cost_of_each_candy_bar :
  (initial_amount - final_amount) / number_of_candy_bars = 2 := by
  sorry

end cost_of_each_candy_bar_l735_735379


namespace highest_a_value_l735_735610

theorem highest_a_value (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 143) : a = 23 :=
sorry

end highest_a_value_l735_735610


namespace circle_area_increase_l735_735581

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * (new_radius)^2
  let increase := new_area - original_area
  let percentage_increase := (increase / original_area) * 100
  percentage_increase = 125 := 
by {
  -- The proof will be written here.
  sorry
}

end circle_area_increase_l735_735581


namespace exists_arrangement_l735_735243

theorem exists_arrangement :
  ∃ (M : Matrix (Fin 3) (Fin 3) ℕ), 
  (∀ i j, M i j ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (Multiset.of_fn (λ i j => M i j)).nodup ∧
  (∀ i, (Finset.univ.sum (λ j => M i j)) % 9 = 0) ∧
  (∀ j, (Finset.univ.sum (λ i => M i j)) % 9 = 0) := by
  sorry

end exists_arrangement_l735_735243


namespace planes_parallel_if_perpendicular_to_same_line_l735_735439

variables {V : Type*} [inner_product_space ℝ V]

/-- Definition of a plane in ℝ³ as an affine subspace --/
def is_plane (α : affine_subspace ℝ V) : Prop := ∃ P ∈ α, ∃ v ∈ α.direction, ∃ w ∈ α.direction, v ≠ 0 ∧ w ≠ 0 ∧ inner v w = 0

/-- Definition of a line being perpendicular to a plane --/
def line_perpendicular_to_plane (l : affine_subspace ℝ V) (α : affine_subspace ℝ V) [affine_subspace ℝ l] [affine_subspace ℝ α] : Prop :=
∃ v ∈ l.direction, ∃ n ∈ α.direction, v ≠ 0 ∧ inner v n = 0 -- Direction of the line is perpendicular to normal vector of the plane

-- Theorem: Given that line l is perpendicular to planes α and β, planes α and β are parallel.
theorem planes_parallel_if_perpendicular_to_same_line
  {l α β : affine_subspace ℝ V} [affine_subspace ℝ l] [affine_subspace ℝ α] [affine_subspace ℝ β]
  (hα : line_perpendicular_to_plane l α)
  (hβ : line_perpendicular_to_plane l β) :
  α.direction = β.direction :=
sorry

end planes_parallel_if_perpendicular_to_same_line_l735_735439


namespace supermarket_profit_and_discount_l735_735420

theorem supermarket_profit_and_discount :
  ∃ (x : ℕ) (nB1 nB2 : ℕ) (discount_rate : ℝ),
    22*x + 30*(nB1) = 6000 ∧
    nB1 = (1 / 2 : ℝ) * x + 15 ∧
    150 * (29 - 22) + 90 * (40 - 30) = 1950 ∧
    nB2 = 3 * nB1 ∧
    150 * (29 - 22) + 270 * (40 * (1 - discount_rate / 100) - 30) = 2130 ∧
    discount_rate = 8.5 := sorry

end supermarket_profit_and_discount_l735_735420


namespace arithmetical_puzzle_l735_735942

theorem arithmetical_puzzle (S I X T W E N : ℕ) 
  (h1 : S = 1) 
  (h2 : N % 2 = 0) 
  (h3 : (1 * 100 + I * 10 + X) * 3 = T * 1000 + W * 100 + E * 10 + N) 
  (h4 : ∀ (a b c d e f : ℕ), 
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
        c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
        d ≠ e ∧ d ≠ f ∧
        e ≠ f) :
  T = 5 := sorry

end arithmetical_puzzle_l735_735942


namespace find_inverse_mod_36_l735_735525

-- Given condition
def inverse_mod_17 := (17 * 23) % 53 = 1

-- Definition for the problem statement
def inverse_mod_36 : Prop := (36 * 30) % 53 = 1

theorem find_inverse_mod_36 (h : inverse_mod_17) : inverse_mod_36 :=
sorry

end find_inverse_mod_36_l735_735525


namespace hypotenuse_length_l735_735590

theorem hypotenuse_length (a b c : ℝ) (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 := 
by
  sorry

end hypotenuse_length_l735_735590


namespace option_two_correct_l735_735120

variables (α : set ℝ) (m n : set ℝ)
  (h1 : m ≠ n)
  (h2 : ∃ p ∈ α, ∀ x ∈ m, p ≠ x)
  (h3 : m ∩ α = ∅)
  (h4 : n ⊆ α)

theorem option_two_correct (h5 : m ∩ α = ∅) (h6 : n ⊆ α) : 
  ∀ x ∈ n, x ∈ α → x ∈ m → x ∈ α ∩ m := 
begin
  sorry
end

end option_two_correct_l735_735120


namespace Jovana_added_shells_l735_735251

theorem Jovana_added_shells :
  ∀ (initial new added : ℝ), initial = 5.75 → new = 28.3 → added = new - initial → added = 22.55 :=
by
  intros initial new added h_initial h_new h_added
  rw [h_initial, h_new] at h_added
  have : added = 22.55 := by norm_num at h_added 
  exact this

end Jovana_added_shells_l735_735251


namespace tangent_line_at_point_intervals_of_monotonicity_max_min_values_on_interval_l735_735196

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3 * x^2 + 9 * x

def point : ℝ × ℝ := (1, 11)

theorem tangent_line_at_point : ∃ (m b : ℝ), m = 12 ∧ b = -1 ∧ ∀ x, 12 * x - f x - 1 = 0 := by sorry

theorem intervals_of_monotonicity : 
  (∀ x, -1 < x < 3 → f' x > 0) ∧ (∀ x, x < -1 ∨ x > 3 → f' x < 0) := by sorry

theorem max_min_values_on_interval : 
  ∃ (max_val min_val : ℝ), max_val = 22 ∧ min_val = -5 := by sorry

end tangent_line_at_point_intervals_of_monotonicity_max_min_values_on_interval_l735_735196


namespace find_n_l735_735481

theorem find_n (n : ℕ) (h_pos : n > 0) (h_factor : ∀ p : ℕ, prime p → p ∣ (2^n - 1) → p ≤ 7) : n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 6 :=
sorry

end find_n_l735_735481


namespace proof_l735_735259

noncomputable def problem := 
  let side_length := 3
  let n := 48
  let square_area := 27 + 18 * Real.sqrt 3
  let a := 27
  let b := 18
  let c := 3
  (a + b + c = 48) ∧ (a + b * Real.sqrt c = square_area)

theorem proof : problem :=
by {
  let side_length := 3
  let n := 48
  let square_area := 27 + 18 * Real.sqrt 3
  let a := 27
  let b := 18
  let c := 3
  have h1 : a + b + c = 48 := by sorry
  have h2 : a + b * Real.sqrt c = square_area := by sorry
  exact ⟨h1, h2⟩
}

end proof_l735_735259


namespace median_of_siblings_l735_735923

-- List of siblings
def siblings : List ℕ := [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5]

-- Function to compute the median (specifically for this predefined ordered list)
def median (l : List ℕ) : ℕ :=
  l[(l.length + 1) / 2 - 1] -- Note: Lean uses 0-based indexing

theorem median_of_siblings :
  median siblings = 2 :=
by sorry

end median_of_siblings_l735_735923


namespace eval_fx_log2_3_l735_735195

def f : ℝ → ℝ
| x := if x ≥ 4 then (1 / 2) ^ x else f (x + 1)

theorem eval_fx_log2_3 : f (Real.log 3 / Real.log 2) = 1 / 24 := by
  sorry

end eval_fx_log2_3_l735_735195


namespace median_of_special_list_l735_735599

theorem median_of_special_list : ∀ n, (1 ≤ n ∧ n ≤ 200 → median_of (multiset.bind (multiset.range n) (λ k, k::k::multiset.repeat k (k-1))) = 141) := by
  intro n hn
  sorry

end median_of_special_list_l735_735599


namespace angle_ratio_l735_735940

theorem angle_ratio (A B C P Q M : Point) (h1 : Trisects (∠ A C B) (∠ A C P) (∠ P C Q) (∠ Q C B))
  (h2 : Bisects (∠ P C Q) (∠ M C P) (∠ M C Q)) :
  (measure (∠ M C Q)) / (measure (∠ A C Q)) = 1 / 4 := by
  sorry

end angle_ratio_l735_735940


namespace minimum_faces_polyhedron_l735_735213

theorem minimum_faces_polyhedron : ∃ (n : ℕ), n = 4 ∧ ∀ p : Polyhedron, faces p ≥ n := 
sorry

end minimum_faces_polyhedron_l735_735213


namespace investment_simple_compound_l735_735573

theorem investment_simple_compound (P y : ℝ) 
    (h1 : 600 = P * y * 2 / 100)
    (h2 : 615 = P * (1 + y/100)^2 - P) : 
    P = 285.71 :=
by
    sorry

end investment_simple_compound_l735_735573


namespace expenses_denoted_as_negative_l735_735325

theorem expenses_denoted_as_negative (income_yuan expenses_yuan : Int) (h : income_yuan = 6) : 
  expenses_yuan = -4 :=
by
  sorry

end expenses_denoted_as_negative_l735_735325


namespace equilateral_triangle_side_length_l735_735335

theorem equilateral_triangle_side_length (total_length : ℕ) (h1 : total_length = 78) : (total_length / 3) = 26 :=
by
  sorry

end equilateral_triangle_side_length_l735_735335


namespace inequality_solution_set_l735_735538

noncomputable def f : ℝ → ℝ := sorry

noncomputable def g : ℝ → ℝ := λ x, f x - 2

-- conditions
axiom deriv_f : ∀ x : ℝ, (deriv f x = f'' x)
axiom f_lt_f'' : ∀ x : ℝ, f x < f'' x
axiom g_odd : ∀ x : ℝ, g x = -g (-x)

-- proof problem statement
theorem inequality_solution_set : {x : ℝ | f x > 2 * real.exp x} = {x : ℝ | 0 < x} :=
sorry

end inequality_solution_set_l735_735538


namespace coefficient_x2_in_expansion_l735_735238

theorem coefficient_x2_in_expansion (a b : ℤ) (n k : ℕ) (h_a : a = 2) (h_b : b = x) (h_n : n = 4) (h_k : k = 2) :
  (Nat.choose n k * a^(n - k)) = 24 := by
  sorry

end coefficient_x2_in_expansion_l735_735238


namespace sum_of_integer_solutions_to_equation_l735_735101

theorem sum_of_integer_solutions_to_equation :
  (∑ x in {x : ℤ | x^4 - 13 * x^2 + 36 = 0}.toFinset, x) = 0 :=
by
  sorry

end sum_of_integer_solutions_to_equation_l735_735101


namespace polynomial_at_3mnplus1_l735_735507

noncomputable def polynomial_value (x : ℤ) : ℤ := x^2 + 4 * x + 6

theorem polynomial_at_3mnplus1 (m n : ℤ) (h₁ : 2 * m + n + 2 = m + 2 * n) (h₂ : m - n + 2 ≠ 0) :
  polynomial_value (3 * (m + n + 1)) = 3 := 
by 
  sorry

end polynomial_at_3mnplus1_l735_735507


namespace percentage_decrease_in_denominator_l735_735682

variable (N D : ℝ)
variable (x : ℝ)

-- Given conditions
def original_fraction := N / D = 0.75
def new_numerator := 1.12 * N
def new_denominator := D * (1 - x / 100)
def new_fraction := 1.12 * N / (D * (1 - x / 100)) = 6 / 7

theorem percentage_decrease_in_denominator :
  original_fraction N D →
  new_numerator N D →
  new_denominator N D x →
  new_fraction N D x →
  x = 6 :=
by
  intro h1 h2 h3 h4
  sorry

end percentage_decrease_in_denominator_l735_735682


namespace steven_has_19_peaches_l735_735612

-- Conditions
def jill_peaches : ℕ := 6
def steven_peaches : ℕ := jill_peaches + 13

-- Statement to prove
theorem steven_has_19_peaches : steven_peaches = 19 :=
by {
    -- Proof steps would go here
    sorry
}

end steven_has_19_peaches_l735_735612


namespace find_m_value_l735_735607

def symmetric_inverse (g : ℝ → ℝ) (h : ℝ → ℝ) :=
  ∀ x, g (h x) = x ∧ h (g x) = x

def symmetric_y_axis (f : ℝ → ℝ) (g : ℝ → ℝ) :=
  ∀ x, f x = g (-x)

theorem find_m_value :
  (∀ g, symmetric_inverse g (Real.exp) → (∀ f, symmetric_y_axis f g → (∀ m, f m = -1 → m = - (1 / Real.exp 1)))) := by
  sorry

end find_m_value_l735_735607


namespace least_number_to_add_l735_735743

def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.sum

def next_multiple_of_9 (n : Nat) : Nat :=
  ((n / 9) + 1) * 9

theorem least_number_to_add (x y : Nat) (hx : x = 51234) (hy : y = 3) :
  ∑ d in (Nat.digits 10 x), d + 3 = next_multiple_of_9 (sum_of_digits x) :=
by
  sorry

end least_number_to_add_l735_735743


namespace evaluate_infinite_series_l735_735081

noncomputable def infinite_series_s : ℝ := ∑ k in (∀ k : ℕ, 1 ≤ k), (k^3) / (3^k)

theorem evaluate_infinite_series : infinite_series_s = 6 :=
by
  sorry

end evaluate_infinite_series_l735_735081


namespace range_of_k_l735_735575

noncomputable def h (x : ℝ) (k : ℝ) : ℝ := 2 * x - k / x + k / 3

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, 1 < x → 2 + k / x^2 > 0) ↔ k ≥ -2 :=
by
  sorry

end range_of_k_l735_735575


namespace range_of_a_l735_735915

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, abs (2 * x - 1) + abs (x + 2) ≥ a^2 + (1 / 2) * a + 2) →
  -1 ≤ a ∧ a ≤ (1 / 2) := by
sorry

end range_of_a_l735_735915


namespace solve_for_x_l735_735111

theorem solve_for_x : 
  ∃ x : ℝ, 10^x * 1000^(2*x) = 10000^3 ↔ x = 12/7 :=
by 
  -- Expressing the given in terms of 10
  have h1 : 1000 = 10^3 := by norm_num,   
  have h2 : 10000 = 10^4 := by norm_num,
  -- Original equation reformulated
  calc
    10^x * (10^3)^(2*x) = (10^4)^3 : by sorry
        -- Simplifying to prove x = 12/7
end

end solve_for_x_l735_735111


namespace calculate_g_18_75_l735_735758

theorem calculate_g_18_75 :
  (∀ x, g x x = x) →
  (∀ x y, g x y = g y x) →
  (∀ x y, (x + y) * g x y = x * g y (x + y)) →
  g 18 75 = 2616 :=
by
  intros h1 h2 h3
  sorry

end calculate_g_18_75_l735_735758


namespace number_of_possible_pairs_l735_735665

theorem number_of_possible_pairs :
  let f := fun (s : List Bool) => s.zip (s.rotate 1).count (λ (x : Bool × Bool), x.1 ≠ x.2 ∧ x.1)
  let m := fun (s : List Bool) => s.zip (s.rotate 1).count (λ (x : Bool × Bool), x.1 ≠ x.2 ∧ ¬ x.1)
  ∀ (s : List Bool), (s.length = 6) →
    (∃ (n : ℕ), n = s.count id) → -- n is the number of females
    let pairs := {(f s, m s) | ∀ s : List Bool, s.length = 6 ∧ (∃ n, n = s.count id)} in
    pairs.to_list.length = 9 :=
by
  sorry

end number_of_possible_pairs_l735_735665


namespace roundness_of_900000_l735_735487

theorem roundness_of_900000 : 
  let n := 900000 in
  let prime_factorization (n : ℕ) : List (ℕ × ℕ) := [(3, 2), (2, 5), (5, 5)] in
  (prime_factorization n).foldl (λ acc p, acc + p.snd) 0 = 12 :=
by
  let n := 900000
  let prime_factorization (n : ℕ) : List (ℕ × ℕ) := [(3, 2), (2, 5), (5, 5)]
  show (prime_factorization n).foldl (λ acc p, acc + p.snd) 0 = 12
  sorry

end roundness_of_900000_l735_735487


namespace probability_a_2b_3c_gt_5_l735_735374

def isInUnitCube (a b c : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1

theorem probability_a_2b_3c_gt_5 (a b c : ℝ) :
  isInUnitCube a b c → ¬(a + 2 * b + 3 * c > 5) :=
by
  intro h
  -- The proof goes here, currently using sorry as placeholder
  sorry

end probability_a_2b_3c_gt_5_l735_735374


namespace product_gcd_lcm_8_12_l735_735389

def gcd (a b : ℕ) : ℕ := a.gcd b
def lcm (a b : ℕ) : ℕ := a.lcm b
noncomputable def product_gcd_lcm (a b : ℕ) : ℕ := gcd a b * lcm a b

theorem product_gcd_lcm_8_12 : product_gcd_lcm 8 12 = 96 := by
  sorry

end product_gcd_lcm_8_12_l735_735389


namespace trapezoid_angles_l735_735413

-- Define the geometric entities and conditions
variables {A B C D M O : Point}
variables {θ1 θ2 θ3 θ4 : ℝ}

-- Axiom: ABCD is a trapezoid with bases AD and BC.
axiom trapezoidABCD (h : Trapezoid A B C D)

-- Axiom: The midpoints M and O of diagonals AC and BD are on the circle.
axiom midpointM (hM : Midpoint M A C)
axiom midpointO (hO : Midpoint O B D)

-- Axiom: The circle with diameter BC passes through M and O and is tangent to AD.
axiom circleDiameterBC (hC1 : Circle Diameter B C)
axiom circlePassesThrough (hC2 : hC1.PassesThrough M)
axiom circlePassesThrough (hC3 : hC1.PassesThrough O)
axiom circleTangentAD (hCT : hC1.Tangent AD)

-- Theorem: The angles of the trapezoid are 30°, 150°, 30°, and 150° respectively.
theorem trapezoid_angles :
  ∃ θ1 θ2 θ3 θ4 ,
    θ1 = 30 ∧ θ2 = 150 ∧ θ3 = 30 ∧ θ4 = 150 :=
sorry

end trapezoid_angles_l735_735413


namespace quadratic_polynomial_discriminant_l735_735144

def P (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∃ x : ℝ, P a b c x = x - 2 ∧ (discriminant a (b - 1) (c + 2) = 0))
  (h₂ : ∃ x : ℝ, P a b c x = 1 - x / 2 ∧ (discriminant a (b + 1 / 2) (c - 1) = 0)) :
  discriminant a b c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l735_735144


namespace find_denominator_l735_735019

theorem find_denominator (x : ℕ) (dec_form_of_frac_4128 : ℝ) (h1: 4128 / x = dec_form_of_frac_4128) 
    : x = 4387 :=
by
  have h: dec_form_of_frac_4128 = 0.9411764705882353 := sorry
  sorry

end find_denominator_l735_735019


namespace quadratic_polynomial_discriminant_l735_735148

def P (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∃ x : ℝ, P a b c x = x - 2 ∧ (discriminant a (b - 1) (c + 2) = 0))
  (h₂ : ∃ x : ℝ, P a b c x = 1 - x / 2 ∧ (discriminant a (b + 1 / 2) (c - 1) = 0)) :
  discriminant a b c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l735_735148


namespace oil_level_drop_l735_735757

-- Define the given conditions
def r_stationary : ℝ := 100
def h_stationary : ℝ := 25
def r_truck : ℝ := 8
def h_truck : ℝ := 10

-- The volume formula for a cylinder
def volume (r h : ℝ) : ℝ := π * r^2 * h

-- The cross-sectional area of the stationary tank
def A_stationary : ℝ := π * r_stationary^2

-- The volume of the truck's tank
def V_truck : ℝ := volume r_truck h_truck

-- The height drop in stationary tank
def h_drop : ℝ := V_truck / A_stationary

-- The theorem to prove
theorem oil_level_drop : h_drop = 0.064 := by
  sorry

end oil_level_drop_l735_735757


namespace find_f_neg_5_div_2_add_f_2_l735_735914

noncomputable def f : ℝ → ℝ := sorry -- We'll define this later based on conditions

-- Given f is periodic with period 2
axiom periodic_f : ∀ x : ℝ, f (x + 2) = f x

-- Given f is odd
axiom odd_f : ∀ x : ℝ, f (-x) = -f x

-- Given f(x) = 4^x in the interval 0 < x < 1
axiom f_interval : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = 4 ^ x

-- Prove the required equality
theorem find_f_neg_5_div_2_add_f_2 : f (-5 / 2) + f 2 = -2 := by
  sorry

end find_f_neg_5_div_2_add_f_2_l735_735914


namespace compute_g_l735_735634

noncomputable def f (x : ℝ) : ℝ := 1 + x + x^2 + x^3 + ...

noncomputable def g (x : ℝ) : ℝ := 1 - x

theorem compute_g (x : ℝ) (hx : |x| < 1) : f(x) * g(x) = 1 := by
  sorry

end compute_g_l735_735634


namespace maximum_value_squared_l735_735964

theorem maximum_value_squared (a b : ℝ) (h₁ : 0 < b) (h₂ : b ≤ a) :
  (∃ x y : ℝ, 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < b ∧ a^2 + y^2 = b^2 + x^2 ∧ b^2 + x^2 = (a + x)^2 + (b - y)^2) →
  (a / b)^2 ≤ 4 / 3 := 
sorry

end maximum_value_squared_l735_735964


namespace oranges_remaining_l735_735376

theorem oranges_remaining (Michaela_oranges : ℕ) 
                          (Cassandra_oranges : ℕ) 
                          (total_oranges : ℕ) 
                          (h1 : Michaela_oranges = 20) 
                          (h2 : Cassandra_oranges = 2 * Michaela_oranges) 
                          (h3 : total_oranges = 90) : 
    total_oranges - (Michaela_oranges + Cassandra_oranges) = 30 :=
by 
  rw [h1, h2, h3]
  -- simplifying the given conditions directly step by step
  calc
    90 - (20 + 2 * 20)
    = 90 - (20 + 40) : by rw [mul_succ] -- replace twice 20 with 40
    = 90 - 60 : by rfl
    = 30 : by rfl

/-

Explanation:
- Michaela_oranges = 20
- Cassandra_oranges = 2 * Michaela_oranges = 40
- total_oranges = 90
So,
total_oranges - (Michaela_oranges + Cassandra_oranges)
    = 90 - (20 + 40)
    = 90 - 60
    = 30

-/

end oranges_remaining_l735_735376


namespace phone_price_increase_is_40_percent_l735_735048

-- Definitions based on the conditions
def initial_price_tv := 500
def increased_fraction_tv := 2 / 5
def initial_price_phone := 400
def total_amount_received := 1260

-- The price increase of the TV
def final_price_tv := initial_price_tv * (1 + increased_fraction_tv)

-- The final price of the phone
def final_price_phone := total_amount_received - final_price_tv

-- The percentage increase in the phone's price
def percentage_increase_phone := ((final_price_phone - initial_price_phone) / initial_price_phone) * 100

-- The theorem to prove
theorem phone_price_increase_is_40_percent :
  percentage_increase_phone = 40 := by
  sorry

end phone_price_increase_is_40_percent_l735_735048


namespace area_ratio_of_extended_equilateral_triangles_l735_735622

theorem area_ratio_of_extended_equilateral_triangles
  (x : ℝ)
  (hx : 0 < x) :
  let area_ABC := (sqrt 3 / 4) * x^2,
      area_A'B'C' := (sqrt 3 / 4) * (3 * x)^2 in
  area_A'B'C' / area_ABC = 9 :=
by
  let area_ABC := (sqrt 3 / 4) * x^2
  let area_A'B'C' := (sqrt 3 / 4) * (3 * x)^2
  have h : area_A'B'C' / area_ABC = 9 := by sorry
  exact h

end area_ratio_of_extended_equilateral_triangles_l735_735622


namespace question1_question2_l735_735282

namespace MathProofs

variable (U : Set ℝ) (A : Set ℝ) (B : Set ℝ)

-- Definitions based on conditions
def isA := ∀ x, A x ↔ (-3 < x ∧ x < 2)
def isB := ∀ x, B x ↔ (Real.exp (x - 1) ≥ 1)
def isCuA := ∀ x, (U \ A) x ↔ (x ≤ -3 ∨ x ≥ 2)

-- Proof of Question 1
theorem question1 : (∀ x, (A ∪ B) x ↔ (x > -3)) := by
  sorry

-- Proof of Question 2
theorem question2 : (∀ x, ((U \ A) ∩ B) x ↔ (x ≥ 2)) := by
  sorry

end MathProofs

end question1_question2_l735_735282


namespace not_all_products_square_not_2_5_13_l735_735272

open Nat

theorem not_all_products_square_not_2_5_13 (d : ℕ) (h1 : d ≠ 2) (h2 : d ≠ 5) (h3 : d ≠ 13) :
  ∃ (a b : ℕ), a ≠ b ∧ a ∈ {2, 5, 13, d} ∧ b ∈ {2, 5, 13, d} ∧ ¬ is_square (a * b - 1) :=
begin
  sorry
end

end not_all_products_square_not_2_5_13_l735_735272


namespace polar_to_rectangular_correct_l735_735467

noncomputable def polar_to_rectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_rectangular_correct :
  polar_to_rectangular 10 (5 * Real.pi / 4) = (-5 * Real.sqrt 2, -5 * Real.sqrt 2) := by
    have cos_value : Real.cos (5 * Real.pi / 4) = -1 / Real.sqrt 2 := Real.cos_pi_div_four
    have sin_value : Real.sin (5 * Real.pi / 4) = -1 / Real.sqrt 2 := Real.sin_pi_div_four
    sorry

end polar_to_rectangular_correct_l735_735467


namespace sum_evaluation_l735_735817

theorem sum_evaluation :
  (∑ k in Finset.range 10, ((10 + (k + 1))^2 - (k + 1)^2)) = 2100 := by
  sorry

end sum_evaluation_l735_735817


namespace quadratic_polynomial_discriminant_l735_735133

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₁ : a ≠ 0)
  (h₂ : ∃ x : ℝ, a * x^2 + b * x + c = x - 2 ∧ (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h₃ : ∃ x : ℝ, a * x^2 + b * x + c = 1 - x / 2 ∧ (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l735_735133


namespace counting_three_digit_integers_l735_735560

theorem counting_three_digit_integers :
  let S := {4, 7, 8, 9} in
  let max_count := λ d, if d = 4 then 4 else if d ∈ {7, 8, 9} then 1 else 0 in
  let choices := finset.prod (finset.Ico 100 1000) (λ n,
    let digits := [n / 100, (n / 10 % 10), (n % 10)] in
    ∀ d ∈ digits.to_finset, digits.count d ≤ max_count d) in
  choices.card = 34 :=
by sorry

end counting_three_digit_integers_l735_735560


namespace base_b_sum_correct_l735_735797

def sum_double_digit_numbers (b : ℕ) : ℕ :=
  (b * (b - 1) * (b ^ 2 - b + 1)) / 2

def base_b_sum (b : ℕ) : ℕ :=
  b ^ 2 + 12 * b + 5

theorem base_b_sum_correct : ∃ b : ℕ, sum_double_digit_numbers b = base_b_sum b ∧ b = 15 :=
by
  sorry

end base_b_sum_correct_l735_735797


namespace find_discriminant_l735_735164

variables {a b c : ℝ}
variables (P : ℝ → ℝ)
def is_quadratic_polynomial (P : ℝ → ℝ) : Prop := ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, P x = a * x^2 + b * x + c)

theorem find_discriminant (h1 : is_quadratic_polynomial P)
  (h2 : ∃ x, P x = x - 2)
  (h3 : ∃ y, P y = 1 - y / 2)
  : ∃ D, D = -1/2 := 
sorry

end find_discriminant_l735_735164


namespace polyhedron_center_of_symmetry_l735_735434

-- Definitions for a polyhedron and centers of symmetry
variables (P : Type) [polyhedron P]
variables (has_symmetric_faces : ∀ (f : face P), has_center_of_symmetry f)

-- Statement to be proved
theorem polyhedron_center_of_symmetry :
  has_center_of_symmetry P :=
sorry

end polyhedron_center_of_symmetry_l735_735434


namespace sequence_property_l735_735526

def sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  a 1 = 3/2 ∧ 
  (∀ n, n ≥ 2 → 2 * S n - S (n - 1) = n^2 + 3 * n - 1) ∧ 
  (∀ n, S n = ∑ i in finset.range (n+1), a i)

theorem sequence_property (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) 
  (h : sequence a S) :
  a n = 2 * n - 1 / 2^n :=
sorry

end sequence_property_l735_735526


namespace triangle_angle_sine_cos_identity_l735_735304

theorem triangle_angle_sine_cos_identity 
  (α β γ : ℝ)
  (hαβγ : α + β + γ = π) :
  (sin β ^ 2 + sin γ ^ 2 - 2 * sin β * sin γ * cos α) / sin α ^ 2 = 1 ∧
  (sin α ^ 2 + sin γ ^ 2 - 2 * sin α * sin γ * cos β) / sin β ^ 2 = 1 ∧
  (sin α ^ 2 + sin β ^ 2 - 2 * sin α * sin β * cos γ) / sin γ ^ 2 = 1 :=
by 
  sorry

end triangle_angle_sine_cos_identity_l735_735304


namespace average_rate_of_interest_l735_735426

noncomputable def interest_problem : Prop :=
  let total_investment : ℝ := 6000
  let rate_a : ℝ := 0.03
  let rate_b : ℝ := 0.05
  let avg_rate : ℝ := 0.0375
  ∃ (x : ℝ), 
    0 ≤ x ∧ x ≤ total_investment ∧ 0.03 * (total_investment - x) = 0.05 * x ∧ 
    (0.03 * (total_investment - x) + 0.05 * x) / total_investment = avg_rate

theorem average_rate_of_interest :
  interest_problem :=
begin
  sorry
end

end average_rate_of_interest_l735_735426


namespace numA_is_irrational_numB_is_rational_numC_is_rational_numD_is_rational_l735_735720

-- Define the numbers in question
def numA : ℝ := Real.sqrt 2
def numB : ℝ := 0.5
def numC : ℝ := 1 / 3
def numD : ℝ := Real.sqrt 4

-- Definition of a rational number
def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- Definition of an irrational number
def is_irrational (x : ℝ) : Prop := ¬ is_rational x

-- The theorem to prove
theorem numA_is_irrational : is_irrational numA :=
by
  sorry

-- Additional facts (not required for the main problem but can be stated)
theorem numB_is_rational : is_rational numB :=
by
  sorry

theorem numC_is_rational : is_rational numC :=
by
  sorry

theorem numD_is_rational : is_rational numD :=
by
  sorry

end numA_is_irrational_numB_is_rational_numC_is_rational_numD_is_rational_l735_735720


namespace hypotenuse_length_l735_735589

theorem hypotenuse_length (a b c : ℝ) (h_right_angled : c^2 = a^2 + b^2) (h_sum_of_squares : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l735_735589


namespace approx_people_per_group_l735_735105

theorem approx_people_per_group : 
    ∃ (S R : ℕ), 
    (R = 3 * S) ∧
    (10 * R + 5 * S = 28350) ∧
    let total_minutes := (8 * 60 + 55) in
    let num_intervals := total_minutes / 5 in
    let total_tickets := S + R in
    let people_per_group := total_tickets / num_intervals in
    people_per_group ≈ 30 := sorry

end approx_people_per_group_l735_735105


namespace rowing_upstream_speed_l735_735764

noncomputable def still_water_speed := 41 -- Speed of the man in still water
def downstream_speed := 48 -- Speed of the man rowing downstream

theorem rowing_upstream_speed :
  let V_m := still_water_speed in
  let V_downstream := downstream_speed in
  let V_s := V_downstream - V_m in
  V_m - V_s = 34 := 
by
  intro V_m V_downstream V_s
  sorry

end rowing_upstream_speed_l735_735764


namespace marco_total_time_l735_735984

def marco_run_time (laps distance1 distance2 speed1 speed2 : ℕ ) : ℝ :=
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  laps * (time1 + time2)

theorem marco_total_time :
  marco_run_time 7 150 350 3 4 = 962.5 :=
by
  sorry

end marco_total_time_l735_735984


namespace fractions_with_same_denominators_fractions_with_same_numerators_fractions_with_different_numerators_and_denominators_l735_735555

theorem fractions_with_same_denominators {a b c : ℤ} (h_c : c ≠ 0) :
  (a > b → a / (c:ℚ) > b / (c:ℚ)) ∧ (a < b → a / (c:ℚ) < b / (c:ℚ)) :=
by sorry

theorem fractions_with_same_numerators {a c d : ℤ} (h_c : c ≠ 0) (h_d : d ≠ 0) :
  (c < d → a / (c:ℚ) > a / (d:ℚ)) ∧ (c > d → a / (c:ℚ) < a / (d:ℚ)) :=
by sorry

theorem fractions_with_different_numerators_and_denominators {a b c d : ℤ} (h_c : c ≠ 0) (h_d : d ≠ 0) :
  a > b ∧ c < d → a / (c:ℚ) > b / (d:ℚ) :=
by sorry

end fractions_with_same_denominators_fractions_with_same_numerators_fractions_with_different_numerators_and_denominators_l735_735555


namespace largest_side_of_similar_triangle_l735_735877

theorem largest_side_of_similar_triangle (x : ℝ) (y : ℝ) (z : ℝ) (k : ℝ) (h_ratio : (x : y : z) = (2 : 3 : 4)) (h_smallest : x = 12) :
  z = 24 :=
by
  -- Proof goes here
  sorry

end largest_side_of_similar_triangle_l735_735877


namespace students_not_participating_l735_735925

theorem students_not_participating (total_students 
participating_modeling participating_programming both_activities : ℕ)
  (h1 : total_students = 15)
  (h2 : participating_modeling = 7)
  (h3 : participating_programming = 9)
  (h4 : both_activities = 3) : 
  total_students - (participating_modeling + participating_programming - both_activities) = 2 := 
by
  rw [h1, h2, h3, h4]
  exact by norm_num

end students_not_participating_l735_735925


namespace increasing_f2_on_R_l735_735437

-- Define the functions
def f1 (x : ℝ) := exp (-x)
def f2 (x : ℝ) := x^3
def f3 (x : ℝ) := log x
def f4 (x : ℝ) := abs x

-- Propose the theorem to be proved
theorem increasing_f2_on_R : 
  (∀ x y : ℝ, x < y → f2 x < f2 y) ∧ 
  (∀ x : ℝ, ∃ y : ℝ, x ≠ y → f2 x ≠ f2 y) ∧ 
  (∀ x : ℝ, differentiable_at ℝ f2 x ∧ deriv f2 x > 0) :=
by 
  split
  -- each part of the proof will be here but, we will place sorry instead to skip proof steps
  sorry
  sorry
  sorry

end increasing_f2_on_R_l735_735437


namespace average_of_div6_between_30_and_50_l735_735821

theorem average_of_div6_between_30_and_50 : 
  let nums := {n : ℕ | 30 ≤ n ∧ n ≤ 50 ∧ n % 6 = 0}
  (∑ n in nums, n) / nums.card = 39 :=
by
  -- Mathematical proof steps would follow here
  sorry

end average_of_div6_between_30_and_50_l735_735821


namespace find_f_l735_735541

noncomputable def f (x : ℝ) : ℝ := 2^x + 1

theorem find_f (x : ℝ) :
  (∀ x > 1, f (x) = 2 ^ x + 1 ↔ f (y) = log 2 (x - 1))
  → (f = λ x, 2^x + 1) :=
sorry

end find_f_l735_735541


namespace neither_sufficient_nor_necessary_condition_l735_735217

theorem neither_sufficient_nor_necessary_condition
  (a1 b1 c1 a2 b2 c2 : ℝ) (h1 : a1 ≠ 0) (h2 : b1 ≠ 0) (h3 : c1 ≠ 0)
  (h4 : a2 ≠ 0) (h5 : b2 ≠ 0) (h6 : c2 ≠ 0) :
  (a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2) ↔
  ¬(∀ x, a1 * x^2 + b1 * x + c1 > 0 ↔ a2 * x^2 + b2 * x + c2 > 0) :=
sorry

end neither_sufficient_nor_necessary_condition_l735_735217


namespace perpendicular_vectors_x_l735_735918

theorem perpendicular_vectors_x (x : ℝ) : 
  let a := (1, 0)
  let b := (2, 1)
  let c := (x, 1)
  let lhs := 3 * a - b
  (lhs.1 * c.1 + lhs.2 * c.2 = 0) → x = 1 :=
by {
  intros a b c lhs perp_cond,
  calc
    x - 1 = 0 : by sorry
}

end perpendicular_vectors_x_l735_735918


namespace lattice_points_on_hyperbola_l735_735561

theorem lattice_points_on_hyperbola : 
  {p : ℤ × ℤ | p.1^2 - p.2^2 = 65}.finite ∧ 
  {p : ℤ × ℤ | p.1^2 - p.2^2 = 65}.to_finset.card = 8 :=
by
  sorry

end lattice_points_on_hyperbola_l735_735561


namespace part1_part2_l735_735523

open Set Real

def A := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem part1 (m : ℝ) (h : Disjoint A (B m)) : m ∈ Iio 2 ∪ Ioi 4 := 
sorry

theorem part2 (m : ℝ) (h : A ∪ (univ \ (B m)) = univ) : m ∈ Iic 3 := 
sorry

end part1_part2_l735_735523


namespace cost_to_fill_can_V_equals_16_l735_735732

noncomputable def cost_to_fill_can_V : ℝ :=
  let r := 1 -- assume any positive value for r
  let h := 1 -- assume any positive value for h
  let cost_half_can_B := 4
  let radius_can_V := 2 * r
  let height_can_V := h / 2
  let volume_can_B := real.pi * r^2 * h
  let volume_can_V := real.pi * (2 * r)^2 * (h / 2)
  let cost_can_B := 2 * cost_half_can_B
  2 * cost_can_B

theorem cost_to_fill_can_V_equals_16 :
  cost_to_fill_can_V = 16 := sorry

end cost_to_fill_can_V_equals_16_l735_735732


namespace wage_on_11th_day_eq_l735_735475

variable (A B C D E : ℝ)

axiom average_wage_total : A = 100
axiom average_wage_first_5 : B = 90
axiom average_wage_second_5 : C = 110
axiom average_wage_third_5 : D = 115.50
axiom average_wage_last_5 : E = 81

theorem wage_on_11th_day_eq :
  0.05 * C + C = D :=
begin
  sorry
end

end wage_on_11th_day_eq_l735_735475


namespace min_value_l735_735890

noncomputable def g : ℝ → ℝ := λ x, x - 1

noncomputable def f : ℝ → ℝ
| x := 
  if x ∈ (0,1] 
  then x^2 - x 
  else -2 * f (x - 1) - 1

theorem min_value (x_1 : ℝ) (x_2 : ℝ) (h1 : 1 < x_1) (h2 : x_1 ≤ 2) : 
  ∃ x_2 : ℝ, (x_1 - x_2)^2 + (f x_1 - g x_2)^2 = 49 / 128 :=
sorry

end min_value_l735_735890


namespace range_of_x_f_x_gt_0_l735_735639

noncomputable def f : ℝ → ℝ := sorry

def odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def derivative_at (f : ℝ → ℝ) (x : ℝ) (y : ℝ) : Prop :=
∀ε > 0, ∃δ > 0, ∀ x', abs (x' - x) < δ → abs ((f x' - f x) / (x' - x) - y) < ε

theorem range_of_x_f_x_gt_0
    (h1 : odd f)
    (h2 : derivative_at f (-1) 0)
    (h3 : ∀ x, x > 0 → x * (derivative (f x)) + f x > 0)
    : {x : ℝ | f x > 0} = set.Ioo (-1) 0 ∪ set.Ioi 1 :=
by sorry

end range_of_x_f_x_gt_0_l735_735639


namespace quadratic_polynomial_discriminant_l735_735145

def P (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∃ x : ℝ, P a b c x = x - 2 ∧ (discriminant a (b - 1) (c + 2) = 0))
  (h₂ : ∃ x : ℝ, P a b c x = 1 - x / 2 ∧ (discriminant a (b + 1 / 2) (c - 1) = 0)) :
  discriminant a b c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l735_735145


namespace triangle_perimeter_l735_735684

theorem triangle_perimeter (A r p : ℝ) (hA : A = 75) (hr : r = 2.5) :
  A = r * (p / 2) → p = 60 := by
  intros
  sorry

end triangle_perimeter_l735_735684


namespace combined_lifting_capacity_correct_l735_735617

-- Define the initial lifting capacities for John, Alice, and Mark
def initial_CJ_john := 80
def initial_snatch_john := 50
def initial_CJ_alice := 90
def initial_snatch_alice := 55
def initial_CJ_mark := 100
def initial_snatch_mark := 65

-- Define the increases after one year for each weightlifter
def increase_CJ_john := 2
def increase_snatch_john := 1.80
def increase_CJ_alice := 1.50
def increase_snatch_alice := 1.90
def increase_CJ_mark := 1.75
def increase_snatch_mark := 1.70

-- New lifting capacities after one year
def new_CJ_john := initial_CJ_john * increase_CJ_john
def new_snatch_john := initial_snatch_john * increase_snatch_john
def new_CJ_alice := initial_CJ_alice * increase_CJ_alice
def new_snatch_alice := initial_snatch_alice * increase_snatch_alice
def new_CJ_mark := initial_CJ_mark * increase_CJ_mark
def new_snatch_mark := initial_snatch_mark * increase_snatch_mark

-- Combined lifting capacities
def combined_CJ := new_CJ_john + new_CJ_alice + new_CJ_mark
def combined_snatch := new_snatch_john + new_snatch_alice + new_snatch_mark

-- New combined total lifting capacity
def new_combined_total_lifting_capacity := combined_CJ + combined_snatch

-- Prove that the new combined total lifting capacity is 775 kg
theorem combined_lifting_capacity_correct :
  new_combined_total_lifting_capacity = 775 := by
  have h1 : new_CJ_john = 160 := rfl
  have h2 : new_snatch_john = 90 := rfl
  have h3 : new_CJ_alice = 135 := rfl
  have h4 : new_snatch_alice = 104.5 := rfl
  have h5 : new_CJ_mark = 175 := rfl
  have h6 : new_snatch_mark = 110.5 := rfl
  have h7 : combined_CJ = 470 := by simp [combined_CJ, h1, h3, h5]
  have h8 : combined_snatch = 305 := by simp [combined_snatch, h2, h4, h6]
  have h9 : new_combined_total_lifting_capacity = combined_CJ + combined_snatch := rfl
  simp [h7, h8, h9]
  sorry

end combined_lifting_capacity_correct_l735_735617


namespace calculate_expression_l735_735450

theorem calculate_expression : (0.25)^(-0.5) + (1/27)^(-1/3) - 625^(0.25) = 0 := 
by 
  sorry

end calculate_expression_l735_735450


namespace difference_between_3rd_and_2nd_smallest_l735_735368

theorem difference_between_3rd_and_2nd_smallest :
  let numbers := {10, 11, 12, 13}
  ∃ a b : ℕ, a ∈ numbers ∧ b ∈ numbers ∧ is_nth_smallest a numbers 3 ∧ is_nth_smallest b numbers 2 ∧ (a - b) = 1 :=
by
  sorry

end difference_between_3rd_and_2nd_smallest_l735_735368


namespace false_proposition_among_given_l735_735786

theorem false_proposition_among_given (a b c : Prop) : 
  (a = ∀ x : ℝ, ∃ y : ℝ, x = y) ∧
  (b = (∃ a b c : ℕ, a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2)) ∧
  (c = ∀ α β : ℝ, α = β ∧ ∃ P : Type, ∃ vertices : P, α = β ) → ¬c := by
  sorry

end false_proposition_among_given_l735_735786


namespace cards_least_likely_red_after_flips_l735_735348

theorem cards_least_likely_red_after_flips :
  ∃ (k1 k2 : ℕ), 1 ≤ k1 ∧ k1 ≤ 50 ∧ 1 ≤ k2 ∧ k2 ≤ 50 ∧ (k1 = 13 ∧ k2 = 38) ∧ 
  (∀ k ∈ finset.range 1 51, 
    let p := (if k ≤ 25 then ((26 - k) ^ 2 + k ^ 2) / 676 else ((26 - (51 - k)) ^ 2 + (51 - k) ^ 2) / 676) in
    p ≥ (if k = 13 ∨ k = 38 then ((26 - k) ^ 2 + k ^ 2) / 676 else p)) :=
sorry

end cards_least_likely_red_after_flips_l735_735348


namespace max_value_of_f_value_f_at_3pi_over_2_max_value_problem_l735_735486

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

theorem max_value_of_f : 
  max (f (Real.pi / 2)) (f (3 * Real.pi / 2)) = f (3 * Real.pi / 2) := 
sorry
theorem value_f_at_3pi_over_2 : 
  f (3 * Real.pi / 2) = (3 * Real.pi / 2) + 1 := 
sorry

theorem max_value_problem : 
  ∃! x ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 2), 
  f x = (3 * Real.pi / 2) + 1 := by
exact ⟨ 3 * Real.pi / 2, ⟨ ⟨ le_refl (3 * Real.pi / 2), le_refl (3 * Real.pi / 2) ⟩, value_f_at_3pi_over_2 ⟩, λ y hy, sorry ⟩

end max_value_of_f_value_f_at_3pi_over_2_max_value_problem_l735_735486


namespace complex_expression_evaluation_l735_735533

-- Definitions based on given conditions
def i : ℂ := complex.I
def z : ℂ := 1 + complex.I
def conj_z : ℂ := complex.conj z

-- Lean statement that represents the proof problem
theorem complex_expression_evaluation :
  (z / i + i * conj_z) = 2 := by
sorry

end complex_expression_evaluation_l735_735533


namespace find_sum_of_5_missing_numbers_l735_735059

-- Define the conditions as constants and variables.
constant first10_avg : ℕ := 48
constant last10_avg : ℕ := 41
constant middle7_avg : ℕ := 45
constant overall_avg : ℕ := 44
constant num_first10 : ℕ := 10
constant num_last10 : ℕ := 10
constant num_middle7 : ℕ := 7
constant num_overall : ℕ := 21

-- Derived sums based on the given conditions.
constant S1 : ℕ := num_first10 * first10_avg
constant S2 : ℕ := num_last10 * last10_avg
constant total_sum : ℕ := num_overall * overall_avg
constant S3 : ℕ := num_middle7 * middle7_avg

-- The values of A and B from the given solution.
constant A : ℕ := S1 - (9 * first10_avg)
constant B : ℕ := S2 - (9 * last10_avg)

-- Define the goal: finding the sum of the 5 unknown numbers in the middle set
noncomputable def sum_of_5_missing_numbers : ℕ :=
  S3 - (A + B)

-- The theorem stating the goal and asserting it equals 226
theorem find_sum_of_5_missing_numbers : sum_of_5_missing_numbers = 226 :=
by
  sorry

end find_sum_of_5_missing_numbers_l735_735059


namespace intersection_of_A_and_B_l735_735978

open Set

theorem intersection_of_A_and_B :
  let A := {x : ℝ | x^2 - x ≤ 0}
  let B := ({0, 1, 2} : Set ℝ)
  A ∩ B = ({0, 1} : Set ℝ) :=
by
  sorry

end intersection_of_A_and_B_l735_735978


namespace range_of_t_l735_735855

noncomputable def S_n (n : ℕ) : ℝ := n^2 + 2 * n

noncomputable def a_n (n : ℕ) : ℝ :=
  if n = 1 then 3 else S_n n - S_n (n - 1)

noncomputable def b_n (n : ℕ) : ℝ :=
  a_n n * a_n (n + 1) * Real.cos ((n + 1) * Real.pi)

noncomputable def T_n (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, b_n i

/-- Prove that T_n ≥ tn^2 implies t ≤ -5, for n ≥ 1 -/
theorem range_of_t (t : ℝ) (n : ℕ) (hn : n ≠ 0) : T_n n ≥ t * n^2 → t ≤ -5 :=
sorry

end range_of_t_l735_735855


namespace frobenius_two_vars_l735_735257

theorem frobenius_two_vars (a b n : ℤ) (ha : 0 < a) (hb : 0 < b) (hgcd : Int.gcd a b = 1) (hn : n > a * b - a - b) :
  ∃ x y : ℕ, n = a * x + b * y :=
by
  sorry

end frobenius_two_vars_l735_735257


namespace evaluate_powers_of_i_l735_735078

theorem evaluate_powers_of_i : (complex.I ^ 45) + (complex.I ^ 345) = 2 * complex.I :=
by
  -- The proof will go here
  sorry

end evaluate_powers_of_i_l735_735078


namespace find_m_and_n_l735_735881

noncomputable def f (x : ℝ) (m n : ℝ) := m / Real.exp x + n * x

theorem find_m_and_n (m n : ℝ) :
  (let f' x := -(m / Real.exp x) + n in
   (f' 0 = -3) ∧ (f 0 m n = 2)) →
  (m = 2 ∧ n = -1) :=
by
  sorry

end find_m_and_n_l735_735881


namespace slopes_of_line_intersecting_ellipse_l735_735762

theorem slopes_of_line_intersecting_ellipse (m : ℝ) : 
  (m ∈ Set.Iic (-1 / Real.sqrt 624) ∨ m ∈ Set.Ici (1 / Real.sqrt 624)) ↔
  ∃ x y, y = m * x + 10 ∧ 4 * x^2 + 25 * y^2 = 100 :=
by
  sorry

end slopes_of_line_intersecting_ellipse_l735_735762


namespace roses_in_each_bouquet_l735_735017

theorem roses_in_each_bouquet (R : ℕ)
(roses_bouquets daisies_bouquets total_bouquets total_flowers daisies_per_bouquet total_daisies : ℕ)
(h1 : total_bouquets = 20)
(h2 : roses_bouquets = 10)
(h3 : daisies_bouquets = 10)
(h4 : total_flowers = 190)
(h5 : daisies_per_bouquet = 7)
(h6 : total_daisies = daisies_bouquets * daisies_per_bouquet)
(h7 : total_flowers - total_daisies = roses_bouquets * R) :
R = 12 :=
by
  sorry

end roses_in_each_bouquet_l735_735017


namespace math_proof_problem_l735_735930

variables {A B C a b c : ℝ}

def is_acute_triangle (A B C : ℝ) : Prop := 0 < A ∧ A < π/2 ∧
                                             0 < B ∧ B < π/2 ∧
                                             0 < C ∧ C < π/2 ∧
                                             A + B + C = π

def side_opposite_angle (a b c A B C : ℝ) : Prop := a^2 = b^2 + c^2 - 2*b*c*cos A

noncomputable def problem_conditions (a b c A B C : ℝ) : Prop :=
is_acute_triangle A B C ∧
2 * (cos((B + C) / 2))^2 + sin (2 * A) = 1 ∧
a = 2 * sqrt 3 - 2 ∧
1/2 * b * c * sin A = 2

theorem math_proof_problem (A B C a b c : ℝ) 
  (h_cond : problem_conditions a b c A B C)
  : A = π / 6 ∧ (a = 2 * sqrt 3 - 2 → 1/2 * b * c * sin A = 2 → b + c = 4 * sqrt 2) :=
by
  sorry

end math_proof_problem_l735_735930


namespace find_cos_phi_l735_735264

variables (a b d : E) [InnerProductSpace ℝ E]
variables (norm_a : ∥a∥ = 2) (norm_b : ∥b∥ = 6) (norm_d : ∥d∥ = 10)
variables (triple_product : a × (b × a) = d)

noncomputable def cos_phi : ℝ :=
  let φ := Real.arccos (a ⬝ b / (∥a∥ * ∥b∥)) in
  Real.cos φ

theorem find_cos_phi : cos_phi a b d = 1 / 12 :=
by
  -- Use the conditions and the given vector triple product identity
  sorry

end find_cos_phi_l735_735264


namespace g_g_x_eq_five_has_five_solutions_l735_735635

def g (x : ℝ) : ℝ :=
if x ≤ 1 then -x + 4 else 3*x - 7

theorem g_g_x_eq_five_has_five_solutions :
  {x : ℝ | g (g x) = 5}.finite.to_finset.card = 5 :=
by
  sorry

end g_g_x_eq_five_has_five_solutions_l735_735635


namespace positive_difference_l735_735054

def C : ℤ := (List.sum (List.map (λ n, (2 * n + 3) * (2 * n + 4)) (List.range 21))) + 45
def D : ℤ := 3 + List.sum (List.map (λ n, (2 * n + 4) * (2 * n + 5)) (List.range 21))

theorem positive_difference : abs (C - D) = 882 := by
  sorry

end positive_difference_l735_735054


namespace collinear_X_Y_Z_l735_735241

-- Assume the basic geometric entities and points
variables {A B C : Point} 
variable (I_a : Excenter A B C)
variable (ω : Circle A I_a)

variables {X Y : Point}
variable (H1 : ω.IntersectsExtension A B X)
variable (H2 : ω.IntersectsExtension A C Y)

variables {S T : Point}
variable (H3 : S ∈ Segment I_a B)
variable (H4 : T ∈ Segment I_a C)
variable (angle_cond1 : ∠AXI_a = ∠BTI_a)
variable (angle_cond2 : ∠AYI_a = ∠CSI_a)

variables {K Z : Point}
variable (H5 : Line BT ∩ Line CS = K)
variable (H6 : Line KI_a ∩ Line TS = Z)

-- Goal: Prove X, Y, Z are collinear
theorem collinear_X_Y_Z : Collinear X Y Z := 
  sorry

end collinear_X_Y_Z_l735_735241


namespace perpendicular_line_plane_l735_735181

variables (m n : Set Point) (α β : Set Plane)
  (line_l : Set Line)

-- Given Conditions
variable (hm : ∀ (p : Point), p ∈ m → p ∈ α)
variable (hn : ∀ (p : Point), p ∈ n → p ∈ β)

-- Additional Conditions
variable (h_diff_mn : m ≠ n)
variable (h_diff_alphabeta : α ≠ β)
variable (h_line_l_intersection : line_l = α ∩ β)
variable (h_planes_perp : α ⊥ β)
variable (h_m_perp_l : m ⊥ line_l)

-- The statement to prove
theorem perpendicular_line_plane :
  m ⊥ β :=
sorry

end perpendicular_line_plane_l735_735181


namespace omar_kite_height_l735_735948

noncomputable theory

def omarRate (jasperRate : ℕ) : ℕ := jasperRate / 3

def height (rate : ℕ) (time : ℕ) : ℕ := rate * time

theorem omar_kite_height :
  ∃ (jasperRate omarRate timeOmar timeJasper : ℕ) (heightJasper heightOmar : ℕ),
    jasperRate = 600 / 10 ∧
    omarRate = jasperRate / 3 ∧
    timeOmar = 12 ∧
    timeJasper = 10 ∧
    heightJasper = jasperRate * timeJasper ∧
    heightOmar = omarRate * timeOmar ∧
    heightOmar = 240 :=
begin
  -- values for Jasper's rate and Omar's rate
  let jasperRate := 60,
  let omarRate := jasperRate / 3,

  -- time taken by Omar and Jasper
  let timeOmar := 12,
  let timeJasper := 10,

  -- height calculations
  let heightJasper := jasperRate * timeJasper,
  let heightOmar := omarRate * timeOmar,

  -- proof of final height
  use [jasperRate, omarRate, timeOmar, timeJasper, heightJasper, heightOmar],
  split, -- for jasperRate
  { exact rfl, },
  split, -- for omarRate
  { exact rfl, },
  split, -- for timeOmar
  { exact rfl, },
  split, -- for timeJasper
  { exact rfl, },
  split, -- for heightJasper
  { exact rfl, },
  split, -- for heightOmar
  { exact rfl, },
  -- final result
  exact rfl,
end

end omar_kite_height_l735_735948


namespace find_line_m_l735_735473

noncomputable def reflect_point_across_line 
  (P : ℝ × ℝ) (a b c : ℝ) : ℝ × ℝ :=
  let line_vector := (a, b)
  let scaling_factor := -2 * ((a * P.1 + b * P.2 + c) / (a^2 + b^2))
  ((P.1 + scaling_factor * a), (P.2 + scaling_factor * b))

theorem find_line_m (P P'' : ℝ × ℝ) (a b : ℝ) (c : ℝ := 0)
  (h₁ : P = (2, -3))
  (h₂ : a * 1 + b * 4 = 0)
  (h₃ : P'' = (1, 4))
  (h₄ : reflect_point_across_line (reflect_point_across_line P a b c) a b c = P'') :
  4 * P''.1 - P''.2 = 0 :=
by
  sorry

end find_line_m_l735_735473


namespace sum_sq_lengths_div_by_a1_l735_735271

theorem sum_sq_lengths_div_by_a1
  (a : ℕ → ℕ)
  (n : ℕ)
  (h1 : ∀ i j, i < j ∧ j ≤ n → a i < a j)
  (h2 : ∀ i j, i ≠ j → Nat.coprime (a i) (a j))
  (h3 : Prime (a 1))
  (h4 : a 1 ≥ n + 2) :
  let A := ∏ i in Finset.range n, a (i + 1)
  in (Finset.sum (Finset.finRange (A+1)) (λ x,
        if ∃ i, a i ∣ x 
        then let S := (Finset.range (x+2)).filter (λ y, ¬ (∃ i, a i ∣ y))
             in Finset.sum S (λ y, y^2)
        else 0)) % a 1 = 0 :=
by {
  sorry
}

end sum_sq_lengths_div_by_a1_l735_735271


namespace relationship_y1_y2_l735_735540

theorem relationship_y1_y2 (k b y1 y2 : ℝ) (h₀ : k < 0) (h₁ : y1 = k * (-1) + b) (h₂ : y2 = k * 1 + b) : y1 > y2 := 
by
  sorry

end relationship_y1_y2_l735_735540


namespace cost_per_mile_l735_735701

def miles_per_week : ℕ := 3 * 50 + 4 * 100
def weeks_per_year : ℕ := 52
def miles_per_year : ℕ := miles_per_week * weeks_per_year
def weekly_fee : ℕ := 100
def yearly_total_fee : ℕ := 7800
def yearly_weekly_fees : ℕ := 52 * weekly_fee
def yearly_mile_fees := yearly_total_fee - yearly_weekly_fees
def pay_per_mile := yearly_mile_fees / miles_per_year

theorem cost_per_mile : pay_per_mile = 909 / 10000 := by
  -- proof will be added here
  sorry

end cost_per_mile_l735_735701


namespace longest_path_in_circle_l735_735594

theorem longest_path_in_circle (O : Type) [MetricSpace O] [NormedGroup O]
  (A B C D P : O) (d : ℝ) (h : dist A B = 10)
  (h₁ : dist A C = 4) (h₂ : dist B D = 4) (h₃ : d ∈ Icc (dist A B) (dist A B))
  (circle : Metric.ball (midpoint ℝ A B) 5 = set.univ) 
  (on_circle : P ∈ Metric.ball (midpoint ℝ A B) 5) :
  dist C P + dist P D = 2 * Real.sqrt 26 → dist C P = dist P D :=
sorry

end longest_path_in_circle_l735_735594


namespace pyramid_volume_proof_l735_735428

noncomputable def volume_of_right_pyramid (total_surface_area : ℝ) (area_tri_face : ℝ) (area_hex_base : ℝ) : ℝ :=
  (1 / 3) * area_hex_base * (height total_surface_area area_tri_face area_hex_base)
  where height := λ total_surface_area area_tri_face area_hex_base,
    (by sorry)  -- Define height based on given conditions

theorem pyramid_volume_proof :
  ∀ (total_surface_area : ℝ) (area_tri_face : ℝ) (area_hex_base : ℝ) (height : ℝ), 
  total_surface_area = 972 → 
  area_tri_face = area_hex_base / 3 → 
  height = (by sorry) → 
  volume_of_right_pyramid total_surface_area area_tri_face area_hex_base = 27 * sqrt(3) * height := 
by
  intros total_surface_area area_tri_face area_hex_base height
  assume h1 : total_surface_area = 972
  assume h2 : area_tri_face = area_hex_base / 3
  assume h3 : height = (by sorry)
  sorry  -- Proof using the given conditions and the formula for volume

end pyramid_volume_proof_l735_735428


namespace prime_p_square_condition_l735_735805

theorem prime_p_square_condition (p : ℕ) (h_prime : Prime p) (h_square : ∃ n : ℤ, 5^p + 4 * p^4 = n^2) :
  p = 31 :=
sorry

end prime_p_square_condition_l735_735805


namespace slope_of_BC_l735_735515

theorem slope_of_BC
  (h₁ : ∀ x y : ℝ, (x^2 / 8) + (y^2 / 2) = 1)
  (h₂ : ∀ A : ℝ × ℝ, A = (2, 1))
  (h₃ : ∀ k₁ k₂ : ℝ, k₁ + k₂ = 0) :
  ∃ k : ℝ, k = 1 / 2 :=
by
  sorry

end slope_of_BC_l735_735515


namespace intersection_A_B_l735_735062
-- Lean 4 code statement

def set_A : Set ℝ := {x | |x - 1| > 2}
def set_B : Set ℝ := {x | x * (x - 5) < 0}
def set_intersection : Set ℝ := {x | 3 < x ∧ x < 5}

theorem intersection_A_B :
  (set_A ∩ set_B) = set_intersection := by
  sorry

end intersection_A_B_l735_735062


namespace earphone_cost_l735_735731

/-- 
The cost of the earphone purchased on Friday can be calculated given:
1. The mean expenditure over 7 days is 500.
2. The expenditures for Monday, Tuesday, Wednesday, Thursday, Saturday, and Sunday are 450, 600, 400, 500, 550, and 300, respectively.
3. On Friday, the expenditures include a pen costing 30 and a notebook costing 50.
-/
theorem earphone_cost
  (mean_expenditure : ℕ)
  (mon tue wed thur sat sun : ℕ)
  (pen_cost notebook_cost : ℕ)
  (mean_expenditure_eq : mean_expenditure = 500)
  (mon_eq : mon = 450)
  (tue_eq : tue = 600)
  (wed_eq : wed = 400)
  (thur_eq : thur = 500)
  (sat_eq : sat = 550)
  (sun_eq : sun = 300)
  (pen_cost_eq : pen_cost = 30)
  (notebook_cost_eq : notebook_cost = 50)
  : ∃ (earphone_cost : ℕ), earphone_cost = 620 := 
by
  sorry

end earphone_cost_l735_735731


namespace k_connected_iff_internally_disjoint_paths_k_edge_connected_iff_edge_disjoint_paths_l735_735415

variables {V : Type*} [fintype V] [decidable_eq V]

-- Internally disjoint paths definition
def internally_disjoint_paths (G : simple_graph V) (u v : V) (k : ℕ) : Prop :=
∃ (paths : fin k → list V),
  (∀ i, u = paths i.head ∧ v = paths i.last) ∧
  (∀ i j, i ≠ j → list.disjoint (paths i) (paths j)) ∧
  (∀ i, paths i.nodup)

-- Edge-disjoint paths definition
def edge_disjoint_paths (G : simple_graph V) (u v : V) (k : ℕ) : Prop :=
∃ (paths : fin k → list (V × V)),
  (∀ i, (u, v) ∈ paths i) ∧
  (∀ i j, i ≠ j → list.disjoint (paths i) (paths j)) ∧
  (∀ i, paths i.nodup)

-- Part (i) Lean statement
theorem k_connected_iff_internally_disjoint_paths (G : simple_graph V) (k : ℕ) :
  G.is_k_connected k ↔ ∀ u v : V, 
  u ≠ v → internally_disjoint_paths G u v k :=
sorry

-- Part (ii) Lean statement
theorem k_edge_connected_iff_edge_disjoint_paths (G : simple_graph V) (k : ℕ) :
  G.is_k_edge_connected k ↔ ∀ u v : V, 
  u ≠ v → edge_disjoint_paths G u v k :=
sorry

end k_connected_iff_internally_disjoint_paths_k_edge_connected_iff_edge_disjoint_paths_l735_735415


namespace exist_line_l1_exist_line_l2_l735_735875

noncomputable def P : ℝ × ℝ := ⟨3, 2⟩

def line1_eq (x y : ℝ) : Prop := 2 * x - y - 4 = 0
def line2_eq (x y : ℝ) : Prop := x - 2 * y + 1 = 0
def perpend_line_eq (x y : ℝ) : Prop := 3 * x + 4 * y - 15 = 0

def line_l1 (x y : ℝ) : Prop := 4 * x - 3 * y - 6 = 0
def line_l2_case1 (x y : ℝ) : Prop := 2 * x - 3 * y = 0
def line_l2_case2 (x y : ℝ) : Prop := x + y - 5 = 0

theorem exist_line_l1 : ∃ (x y : ℝ), line1_eq x y ∧ line2_eq x y ∧ perpend_line_eq x y → line_l1 x y :=
by
  sorry

theorem exist_line_l2 : ∃ (x y : ℝ), line1_eq x y ∧ line2_eq x y ∧ ((line_l2_case1 x y) ∨ (line_l2_case2 x y)) :=
by
  sorry

end exist_line_l1_exist_line_l2_l735_735875


namespace school_boys_number_l735_735926

theorem school_boys_number (B G : ℕ) (h1 : B / G = 5 / 13) (h2 : G = B + 80) : B = 50 :=
by
  sorry

end school_boys_number_l735_735926


namespace max_distance_parallel_lines_l735_735519

open Real

theorem max_distance_parallel_lines (k m : Real) (h1 : k ≠ 0) (h2 : m ≠ -1) :
  let l1 := λ (x y : Real), x + k*y + 1 = 0
  let l2 := λ (x y : Real), (m+1)*x - y + 1 = 0
  let parallel := k = -(m+1)
  parallel → (∀ (x1 y1 : Real), l1 x1 y1 → ∀ (x2 y2 : Real), l2 x2 y2 → dist (x1, y1) (x2, y2) ≤ 2) :=
begin
  intros h_parallel,
  sorry
end

end max_distance_parallel_lines_l735_735519


namespace train_speed_approximation_l735_735783

theorem train_speed_approximation
  (train_length : ℝ)
  (cross_time : ℝ)
  (h_train_length : train_length = 320)
  (h_cross_time : cross_time = 12) :
  (train_length / cross_time * 3.6) ≈ 96.01 :=
by
  sorry

end train_speed_approximation_l735_735783


namespace eval_ceiling_expr_l735_735076

theorem eval_ceiling_expr :
  (Int.ceil (4 * (8 - (3 / 4) + 2))) = 37 :=
by
  sorry

end eval_ceiling_expr_l735_735076


namespace projection_impossible_as_triangle_l735_735724

theorem projection_impossible_as_triangle 
  (rect : Type) 
  (projection : rect → Type) 
  (cardboard : rect) 
  (conditions : ∀ (orientation : rect), 
    orientation = "parallel to sunlight" → projection orientation = "line segment" ∧
    orientation = "parallel to ground" → projection orientation = "rectangle" ∧
    orientation = "inclined" → projection orientation = "parallelogram") : 
  ¬ ∃ (orientation : rect), projection orientation = "triangle" :=
by sorry

end projection_impossible_as_triangle_l735_735724


namespace fraction_calculation_l735_735052

theorem fraction_calculation : (3/10 : ℚ) + (5/100 : ℚ) - (2/1000 : ℚ) = 348/1000 := 
by 
  sorry

end fraction_calculation_l735_735052


namespace _l735_735809

noncomputable theorem distinct_pos_numbers_sum_to_22
  (a b c : ℝ) 
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : a ≠ b)
  (h5 : b ≠ c)
  (h6 : a ≠ c)
  (eq1 : a^2 + b * c = 115)
  (eq2 : b^2 + a * c = 127)
  (eq3 : c^2 + a * b = 115) :
  a + b + c = 22 :=
  sorry

end _l735_735809


namespace real_part_of_Z_l735_735220

theorem real_part_of_Z (Z : ℂ) : (1 + I) * Z = complex.abs (3 + 4 * I) → Z.re = 5 / 2 :=
by
  intro h
  sorry

end real_part_of_Z_l735_735220


namespace angle_ACB_eq_60_l735_735702

theorem angle_ACB_eq_60 (A B C D E F : Type) 
  (hABC : Triangle A B C) 
  (hAC_2AB : dist A C = 2 * dist A B)
  (hD_ab : ∃ (d : ℝ), 0 ≤ d ∧ d ≤ 1 ∧ (LineSegment A B).contains d)
  (hE_bc : ∃ (e : ℝ), 0 ≤ e ∧ e ≤ 1 ∧ (LineSegment B C).contains e)
  (hAngle_BAE_ACD : ∡ B A E = ∡ A C D)
  (hF_inter : F = intersection (Line A E) (Line C D))
  (hTriangle_AFE_equilateral : TriangleEquilateral A F E) :
  ∡ A C B = 60 :=
sorry

end angle_ACB_eq_60_l735_735702


namespace sheets_usage_l735_735767

noncomputable def average_sheets_per_day
  (sheets_per_pad: ℕ)
  (days_off_per_year: ℕ)
  (weeks_per_year: ℕ)
  (days_per_week: ℕ)
  (sheets_used_day1: ℕ)
  (sheets_used_day2: ℕ)
  (sheets_used_day3: ℕ)
  (sheets_lent_per_week: ℕ) : ℝ :=
  let total_sheets_per_week := sheets_used_day1 + sheets_used_day2 + sheets_used_day3 + sheets_lent_per_week
  let total_off_weeks :=  (days_off_per_year : ℝ) / (days_per_week : ℝ)
  let total_working_weeks := (weeks_per_year : ℝ) - total_off_weeks
  let total_sheets_per_year := (total_sheets_per_week : ℝ)  * total_working_weeks
  let total_working_days_per_year := total_working_weeks * (days_per_week : ℝ)
  total_sheets_per_year / total_working_days_per_year

theorem sheets_usage (h : average_sheets_per_day 60 8 48 3 2 4 8 3 = 5.67) : true :=
begin
  sorry
end

end sheets_usage_l735_735767


namespace maximum_possible_ratio_l735_735430

-- Definitions following conditions:
def board := ℕ
def cells (n : ℕ) := fin (n*n)
def pattern (c : cells 8 → Prop) := ∀ (x y : fin 8), (mod2 x + mod2 y = 0 ↔ ¬c ⟨x.val * 8 + y.val, sorry⟩)
def area_ratio_upper_bound (c : cells 8 → nat) (w b : cells 8 → bool) := ∀ (x y : cells 8), w x = tt → b y = tt → (c x / c y ≤ 2)
def total_area (c : cells 8 → nat) (w : cells 8 → bool) := ∑ (x : cells 8), if w x then c x else 0

-- Target statement:
theorem maximum_possible_ratio (board_div : board) (cell_size : cells 8 → nat) (white_black_pattern : cells 8 → bool)
  (h_pattern : pattern white_black_pattern)
  (h_area_ratio_bound : area_ratio_upper_bound cell_size white_black_pattern (λ x, ¬white_black_pattern x)) :
  ∃ (max_ratio : ℚ), max_ratio = 5 / 4 ∧
  (total_area cell_size white_black_pattern / total_area cell_size (λ x, ¬white_black_pattern x)) ≤ max_ratio := by
  sorry

end maximum_possible_ratio_l735_735430


namespace graph_shift_eq_f_of_g_shifted_l735_735332

noncomputable def f (x : ℝ) : ℝ :=
  sin x * cos x - sqrt 3 * cos x ^ 2

noncomputable def g (x : ℝ) : ℝ :=
  sin (2 * x + π / 3) - sqrt 3 / 2

theorem graph_shift_eq_f_of_g_shifted (k : ℝ) (hk : k = π / 3) :
  (f = λ x, g (x - k)) :=
by simp [f, g, hk]; sorry

end graph_shift_eq_f_of_g_shifted_l735_735332


namespace ruth_gave_janet_53_stickers_l735_735246

-- Definitions: Janet initially has 3 stickers, after receiving more from Ruth, she has 56 stickers in total.
def janet_initial : ℕ := 3
def janet_total : ℕ := 56

-- The statement to prove: Ruth gave Janet 53 stickers.
def stickers_from_ruth (initial: ℕ) (total: ℕ) : ℕ :=
  total - initial

theorem ruth_gave_janet_53_stickers : stickers_from_ruth janet_initial janet_total = 53 :=
by sorry

end ruth_gave_janet_53_stickers_l735_735246


namespace problem1_problem2_l735_735312

-- Problem 1
theorem problem1 (x : ℝ) (h1 : 2 * x > 1 - x) (h2 : x + 2 < 4 * x - 1) : x > 1 := 
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ)
  (h1 : (2 / 3) * x + 5 > 1 - x)
  (h2 : x - 1 ≤ (3 / 4) * x - 1 / 8) :
  -12 / 5 < x ∧ x ≤ 7 / 2 := 
by
  sorry

end problem1_problem2_l735_735312


namespace checkerboard_repaint_impossible_l735_735671

theorem checkerboard_repaint_impossible : 
  ∀ (board : array (array bool 100) 100), 
  (∀ (i : fin 100), (array.count (λ j, board[i][j]) (λ x, x = tt) = 50)) ∧ 
  (∀ (j : fin 100), (array.count (λ i, board[i][j]) (λ x, x = tt) = 50)) → 
  ¬ (∃ (repainted_board : array (array bool 100) 100), 
     count_diff_board repainted_board board = 2018 ∧ 
     (∀ (i : fin 100), (array.count (λ j, repainted_board[i][j]) (λ x, x = tt) = 50)) ∧ 
     (∀ (j : fin 100), (array.count (λ i, repainted_board[i][j]) (λ x, x = tt) = 50))) := 
begin
  sorry
end

def count_diff_board (board1 : array (array bool 100) 100) (board2 : array (array bool 100) 100) : nat :=
  array.foldl (λ acc i, acc + array.foldl (λ acc j, if board1[i][j] ≠ board2[i][j] then acc + 1 else acc) 0 (array.enum 100)) 0 (array.enum 100)

end checkerboard_repaint_impossible_l735_735671


namespace functions_of_same_family_count_l735_735572

-- Definitions and conditions based on the problem statement
def analytic_expression (x : ℝ) : ℝ := x^2
def range_set : Set ℝ := {1, 4}
def domain_set : Set ℝ := {1, -1, 2, -2}

-- Problem statement in Lean 4
theorem functions_of_same_family_count : 
  -- Given Conditions
  --   1. Analytic expression: y = x^2
  --   2. Range: {1, 4}
  --   3. Definition of "functions of the same family": Functions with the same analytic expression, the same range, but different domains.

  -- Prove that the number of such functions is 9.
  (number_of_functions_of_same_family analytic_expression range_set domain_set = 9) :=
sorry

end functions_of_same_family_count_l735_735572


namespace problem1_problem2_problem3_problem4_l735_735608

def is_expressible (m : ℤ) : Prop :=
  ∃ (σ : Fin 9 → ℤ), (∀ i, σ i ∈ {1, -1} ∧ (i : ℤ) + 1 * σ i) = m

theorem problem1 : is_expressible 13 := sorry

theorem problem2 : ¬ is_expressible 14 := sorry

theorem problem3 : ∃ n, n = 46 ∧ (∀ m, -45 ≤ m ∧ m ≤ 45 → is_expressible m → m % 2 = 1) := sorry

theorem problem4 : ∃ n, n = 8 ∧ (count_ways_to_express 27) = n := sorry

noncomputable def count_ways_to_express (m : ℤ) : ℕ := sorry

end problem1_problem2_problem3_problem4_l735_735608


namespace solution1_solution2_l735_735197

-- Define the function f(x) = log_a(x + 1)
def f (a : ℝ) (x : ℝ) : ℝ := log a (x + 1)

-- Define the conditions for the first problem
def condition1 (a : ℝ) (x : ℝ) : Prop :=
a > 0 ∧ a ≠ 1 ∧ f(a, x) > 0

-- Prove the solution set for f(x) > 0
theorem solution1 (a : ℝ) (x : ℝ) : condition1 a x → ((a > 1 → x ∈ set.Ioi 0) ∧ (0 < a → a < 1 → x ∈ set.Ioo (-1) 0)) :=
sorry -- Proof to be filled in

-- Define the conditions for the second problem
def condition2 (a : ℝ) (p m n : ℝ) : Prop :=
a > 1 ∧ -1 < m ∧ m < 0 ∧ m < n ∧
f(a, m) = log a (p / m) ∧ f(a, n) = log a (p / n)

-- Prove the range of the real number p
theorem solution2 (a : ℝ) (p m n : ℝ) : condition2 a p m n → -1 / 4 < p ∧ p < 0 :=
sorry -- Proof to be filled in

end solution1_solution2_l735_735197


namespace length_QR_l735_735689

-- Let's define the given conditions and the theorem to prove

-- Define the lengths of the sides of the triangle
def PQ : ℝ := 4
def PR : ℝ := 7
def PM : ℝ := 3.5

-- Define the median formula
def median_formula (PQ PR QR PM : ℝ) := PM = 0.5 * Real.sqrt (2 * PQ^2 + 2 * PR^2 - QR^2)

-- The theorem to prove: QR = 9
theorem length_QR 
  (hPQ : PQ = 4) 
  (hPR : PR = 7) 
  (hPM : PM = 3.5) 
  (hMedian : median_formula PQ PR QR PM) : 
  QR = 9 :=
sorry  -- proof will be here

end length_QR_l735_735689


namespace remainder_of_largest_multiple_of_9_no_repeat_digits_1_through_6_div_100_l735_735260

theorem remainder_of_largest_multiple_of_9_no_repeat_digits_1_through_6_div_100 :
  ∃ M : ℕ, (M % 9 = 0) ∧ (∀ d ∈ int.to_digits 10 M, d ∈ {1, 2, 3, 4, 5, 6}) ∧ (∀ i j, i ≠ j → int.to_digits 10 M !! i ≠ int.to_digits 10 M !! j) ∧ ((M % 100) = 32) :=
sorry

end remainder_of_largest_multiple_of_9_no_repeat_digits_1_through_6_div_100_l735_735260


namespace part1_complement_intersection_part2_range_m_l735_735267

open Set

-- Define set A
def A : Set ℝ := { x | -1 ≤ x ∧ x < 4 }

-- Define set B parameterized by m
def B (m : ℝ) : Set ℝ := { x | m ≤ x ∧ x ≤ m + 2 }

-- Part (1): Prove the complement of the intersection for m = 3
theorem part1_complement_intersection :
  ∀ x : ℝ, x ∉ (A ∩ B 3) ↔ x < 3 ∨ x ≥ 4 :=
by
  sorry

-- Part (2): Prove the range of m for A ∩ B = ∅
theorem part2_range_m (m : ℝ) :
  (A ∩ B m = ∅) ↔ m < -3 ∨ m ≥ 4 :=
by
  sorry

end part1_complement_intersection_part2_range_m_l735_735267


namespace vector_statements_l735_735177

section VectorProofs

variables {x : ℚ} (a b : ℚ × ℚ)
def parallel (a b : ℚ × ℚ) : Prop := ∃ k : ℚ, a = (k * b.1, k * b.2)
def orthogonal (a b : ℚ × ℚ) : Prop := a.1 * b.1 + a.2 * b.2 = 0
def projection (u v : ℚ × ℚ) : ℚ × ℚ := 
  let denom := (v.1 * v.1 + v.2 * v.2) 
  in ((u.1 * v.1 + u.2 * v.2) / denom * v.1, 
      (u.1 * v.1 + u.2 * v.2) / denom * v.2)

noncomputable def conditions := (3, -4) ∧ b = (x, 3)

theorem vector_statements
  (par_cond : parallel (3, -4) b ↔ x = -9 / 4)
  (ortho_cond : orthogonal (3, -4) b → projection (3, -4) (3 * (3, -4) + 4 * b) = (3, 0)) :
  (parallel (3, -4) (x, 3) → x = -9 / 4) ∧ 
  (orthogonal (3, -4) (x, 3) → projection (3, -4) (3 * (3, -4) + 4 * (x, 3)) = (3, 0)) :=
begin
  split;
  intro H;
    cases H with k hk;
    sorry,
  sorry
end

end VectorProofs

end vector_statements_l735_735177


namespace equal_segments_l735_735771

variable (A B C D O M N A1 B1 C1 D1: Type) [Point] (AB CD: Line) (circle: Circle) (quadrilateral_abcd: CyclicQuadrilateral AB CD) 
  (mid_m: Midpoint M AB) (mid_n: Midpoint N CD) 
  (congruent_quads: CongruentQuadrilaterals (Quadrilateral A A1 B1 B) (Quadrilateral C C1 D1 D))

theorem equal_segments : segment A1 B1 = segment C1 D1 :=
by
  sorry

end equal_segments_l735_735771


namespace estimated_white_balls_l735_735933

theorem estimated_white_balls (total_balls : ℕ) (freq_white_ball : ℝ) :
  total_balls = 10 → freq_white_ball = 0.4 → 
  (freq_white_ball * total_balls).floor.cast = 4 := by
  sorry

end estimated_white_balls_l735_735933


namespace james_muffins_correct_l735_735047

-- Arthur baked 115 muffins
def arthur_muffins : ℕ := 115

-- James baked 12 times as many muffins as Arthur
def james_multiplier : ℕ := 12

-- The number of muffins James baked
def james_muffins : ℕ := arthur_muffins * james_multiplier

-- The expected result
def expected_james_muffins : ℕ := 1380

-- The statement we want to prove
theorem james_muffins_correct : james_muffins = expected_james_muffins := by
  sorry

end james_muffins_correct_l735_735047


namespace equal_angles_quadrilateral_l735_735653

theorem equal_angles_quadrilateral (ABCD : Type) 
  (isQuadrilateral : ∀ (A B C D : ABCD), quadrilateral A B C D)
  (equal_diagonals : ∀ (A B C D : ABCD), diagonal_length A C = diagonal_length B D) :
  ∀ (m1 m2 : ABCD), midline_intersects_diagonals_at_equal_angles m1 m2 :=
by
  sorry

end equal_angles_quadrilateral_l735_735653


namespace manufacturing_cost_before_decrease_l735_735801

variable (P : ℝ)
variable (C_old : ℝ)
variable (C_new : ℝ := 50)

def initial_profit := 0.20 * P
def new_profit := 0.50 * P
def old_manufacturing_cost := P - initial_profit

theorem manufacturing_cost_before_decrease
  (h1 : C_new = P - new_profit) :
  C_old = old_manufacturing_cost :=
by
  sorry

end manufacturing_cost_before_decrease_l735_735801


namespace minimal_flip_probability_l735_735367

def flip_probability (n : ℕ) (k : ℕ) : ℚ :=
  if k <= 25 then
    (2 * k^2 - 52 * k + 676) / 676
  else
    let mirrored_k := 51 - k in
    (2 * mirrored_k^2 - 52 * mirrored_k + 676) / 676

theorem minimal_flip_probability :
  ∀ k, (13 ≤ k ∧ k ≤ 13) ∨ (38 ≤ k ∧ k ≤ 38) :=
by
  intro k
  sorry

end minimal_flip_probability_l735_735367


namespace advertisement_revenue_l735_735756

theorem advertisement_revenue
  (cost_per_program : ℝ)
  (num_programs : ℕ)
  (selling_price_per_program : ℝ)
  (desired_profit : ℝ)
  (total_cost_production : ℝ)
  (total_revenue_sales : ℝ)
  (total_revenue_needed : ℝ)
  (revenue_from_advertisements : ℝ) :
  cost_per_program = 0.70 →
  num_programs = 35000 →
  selling_price_per_program = 0.50 →
  desired_profit = 8000 →
  total_cost_production = cost_per_program * num_programs →
  total_revenue_sales = selling_price_per_program * num_programs →
  total_revenue_needed = total_cost_production + desired_profit →
  revenue_from_advertisements = total_revenue_needed - total_revenue_sales →
  revenue_from_advertisements = 15000 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end advertisement_revenue_l735_735756


namespace two_trains_crossing_time_l735_735705

-- Define the conditions
def train_speed (length time : ℝ) : ℝ := length / time

def relative_speed (speed1 speed2 : ℝ) : ℝ := speed1 + speed2

def crossing_time (total_distance speed : ℝ) : ℝ := total_distance / speed

-- Constants given in the problem
def length_of_train : ℝ := 120
def time_to_cross_post_train1 : ℝ := 15
def time_to_cross_post_train2 : ℝ := 20

-- Calculations based on the problem
def speed_train1 : ℝ := train_speed length_of_train time_to_cross_post_train1
def speed_train2 : ℝ := train_speed length_of_train time_to_cross_post_train2

def total_distance : ℝ := length_of_train + length_of_train
def total_relative_speed : ℝ := relative_speed speed_train1 speed_train2

-- Theorem to prove
theorem two_trains_crossing_time :
  crossing_time total_distance total_relative_speed = 17.14 := by
    sorry

end two_trains_crossing_time_l735_735705


namespace part1_part2_l735_735318

-- Definition for part 1
def exists_finite_set_geometric_mean_integer (n : ℕ) (n_pos : n > 0) : Prop :=
  ∃ S : Finset ℕ, S.card = n ∧ 
  (∀ T ⊆ S, T.nonempty → ∃ k : ℕ, ∏ x in T, x ^ (1 / T.card : ℝ) = k)

-- Statement for part 1
theorem part1 (n : ℕ) (n_pos : n > 0) : exists_finite_set_geometric_mean_integer n n_pos :=
  sorry

-- Definition for part 2
def exists_infinite_set_geometric_mean_integer : Prop :=
  ∃ S : Set ℕ, S.Infinite ∧ 
  (∀ T : Finset ℕ, T ⊆ S.to_finset → T.nonempty → ∃ k : ℕ, ∏ x in T, x ^ (1 / T.card : ℝ) = k)

-- Statement for part 2
theorem part2 : ¬ exists_infinite_set_geometric_mean_integer :=
  sorry

end part1_part2_l735_735318


namespace evaluate_expression_l735_735476

theorem evaluate_expression (a b : ℕ) (h1 : a = 3) (h2 : b = 2) : (a^b)^b + (b^a)^a = 593 := by
  sorry

end evaluate_expression_l735_735476


namespace minimum_switches_for_sorted_sequence_l735_735403

theorem minimum_switches_for_sorted_sequence : 
  ∀ (P Q : list ℕ), 
    P = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1] → 
    Q = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] →
    (∃ (n : ℕ), n = 120 ∧ ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 16 ∧ 1 ≤ j ∧ j ≤ 16 ∧ i < j → 
    (P !! i) > (P !! j)) := 
by 
  intros P Q hP hQ 
  use 120 
  split 
  { 
    exact rfl
  }
  { 
    intros i j hi hj hij 
    sorry 
  }

end minimum_switches_for_sorted_sequence_l735_735403


namespace unique_one_to_one_mapping_l735_735085

theorem unique_one_to_one_mapping (f : ℕ → ℕ) (h_inj : function.injective f)
  (h_cond : ∀ n : ℕ, n > 0 → f(f(n)) ≤ (n + f(n)) / 2) :
  ∀ n : ℕ, f(n) = n :=
by
  sorry

end unique_one_to_one_mapping_l735_735085


namespace f_mul_eq_three_l735_735539

-- Define the conditions
def is_even {f : ℝ → ℝ} : Prop := ∀ x : ℝ, f x = f (-x)
def is_periodic_4 {f : ℝ → ℝ} : Prop := ∀ x : ℝ, f (x + 2) = f (x - 2)
def def_on_interval {f : ℝ → ℝ} : Prop := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f x = 2^x - 1

-- Define the function and problem statement
theorem f_mul_eq_three (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_periodic : is_periodic_4 f)
  (h_def_interval : def_on_interval f) : 
  f (-2017) * f 2018 = 3 :=
sorry

end f_mul_eq_three_l735_735539


namespace problem_statement_l735_735565

def count_possible_N : ℕ :=
  let N_values := { N : ℕ | ∃ x : ℝ, 0 < N ∧ N < 1500 ∧ N = x ^ (⌊ x ⌋) } in
  N_values.card

theorem problem_statement : count_possible_N = 581 :=
  sorry

end problem_statement_l735_735565


namespace all_math_majors_consecutive_probability_correct_l735_735317

noncomputable def probability_all_math_majors_consecutive (total_people : ℕ) (math_majors : ℕ) 
    (physics_majors : ℕ) (chemistry_majors : ℕ) : ℚ :=
  if total_people = 10 ∧ math_majors = 5 ∧ physics_majors = 3 ∧ chemistry_majors = 2 then
    5 / 126
  else
    0

theorem all_math_majors_consecutive_probability_correct (h : true) :
  probability_all_math_majors_consecutive 10 5 3 2 = 5 / 126 :=
begin
  sorry,
end

end all_math_majors_consecutive_probability_correct_l735_735317


namespace minimum_distance_between_points_l735_735874

theorem minimum_distance_between_points :
  (∃ l : ℝ → ℝ, ∀ x : ℝ, (e (-2 * x + 1) = l (x)) ∧ (l(x) = -x - 1)) →
  ∀ P Q : ℝ × ℝ,
    (P = (λ x, (x, e (-2 * x + 1)))) ∧
    (Q = (λ x, (x, (ln (-x - 1) - 3) / 2))) →
    ∃ d : ℝ, d = (sqrt 2 * (4 + log 2) / 2) := sorry

end minimum_distance_between_points_l735_735874


namespace eval_expression_l735_735451

theorem eval_expression : (2 ^ (-1 : ℤ)) + (Real.sin (Real.pi / 6)) - (Real.pi - 3.14) ^ (0 : ℤ) + abs (-3) - Real.sqrt 9 = 0 := by
  sorry

end eval_expression_l735_735451


namespace Ethan_read_pages_at_night_10_l735_735816

noncomputable def pagesReadOnSaturdayNight (total_pages read_morning read_remaining read_twice: ℕ) := 
  let read_on_saturday := read_morning + x
  let read_on_sunday := 2 * read_on_saturday
  read_morning + x + read_on_sunday = total_pages - read_remaining

theorem Ethan_read_pages_at_night_10 :
  ∀ (total_pages read_morning read_remaining : ℕ), total_pages = 360 → read_morning = 40 → read_remaining = 210 →
  (pagesReadOnSaturdayNight total_pages read_morning read_remaining 2 10 = true) :=
  by
    intros total_pages read_morning read_remaining h_total_pages h_read_morning h_read_remaining
    unfold pagesReadOnSaturdayNight
    sorry

end Ethan_read_pages_at_night_10_l735_735816


namespace angle_ratio_l735_735938

theorem angle_ratio (CP CQ CM : Point -> Point -> Angle) (A C B M : Point) (P Q : Point) (∠ACB ∠MCQ y : ℝ)
    (trisect : ∠ACB / 3 = CP ∠ C ∧ ∠ACB / 3 = CQ ∠ C)
    (bisect : CM ∠ C P = CM ∠ C Q) :
    ∠MCQ / (4 * y) = 1 / 4 :=
by
    sorry

end angle_ratio_l735_735938


namespace tan_squared_half_C_eq_tan_alpha_mul_tan_beta_l735_735501

/-- 
Given a triangle ABC, with C as the vertex. 
A median and an angle bisector are drawn from vertex C forming an angle α. 
Let β be the acute angle formed by the angle bisector and side AB. 
We need to prove the trigonometric identity: 
tan²(C/2) = tan(α) * tan(β) 
--/
theorem tan_squared_half_C_eq_tan_alpha_mul_tan_beta
  (α β C : ℝ)
  (hC : 0 < C ∧ C < π)  -- C is an angle of a triangle
  (hα : α ≠ 0)  -- α is the angle between the median and the angle bisector
  (hβ : β ≠ 0)  -- β is the acute angle on side AB
  (h_condition : Some additional geometric conditions if needed) 
  : tan (C / 2) ^ 2 = tan α * tan β :=
sorry

end tan_squared_half_C_eq_tan_alpha_mul_tan_beta_l735_735501


namespace sphere_radius_from_cone_l735_735035

-- Define the conditions of the problem
def volume_sphere (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3
def volume_cone (r h : ℝ) : ℝ := (1 / 3) * Real.pi * r^2 * h
def cone_height (r : ℝ) : ℝ := 6 * r

theorem sphere_radius_from_cone (r R : ℝ) (h : ℝ)
  (hr : r = 3 * Real.cbrt 2) 
  (hh : h = 6 * r)
  (hv : volume_sphere R = volume_cone r h) :
  R = 3 * Real.cbrt 3 :=
by {
  sorry
}

end sphere_radius_from_cone_l735_735035


namespace john_saves_1200_per_year_l735_735250

-- Definitions of the conditions
def formerRentPerSqFt : ℝ := 2
def sizeOfFormerApartment : ℝ := 750
def totalNewRent : ℝ := 2800
def newRentSplit : ℝ := totalNewRent / 2

-- Definition of the monthly savings
def monthlySavings : ℝ :=
  (formerRentPerSqFt * sizeOfFormerApartment) - newRentSplit

-- Definition of the annual savings
def annualSavings : ℝ := monthlySavings * 12

-- The theorem to prove John saves $1200 a year by moving to the new apartment with a roommate
theorem john_saves_1200_per_year :
  annualSavings = 1200 := by
  sorry

end john_saves_1200_per_year_l735_735250


namespace area_of_rectangle_PQRS_l735_735595

-- Definitions based on given conditions
variables (PQRS : Type) [rectangle PQRS]
variables (P Q R S H G : PQRS)
variables (angle : ℝ)
variables [trisects_angle R angle RG]
variables [trisects_angle R angle RH]
variables (PH SG : ℝ)

-- Given conditions
def conditions : Prop :=
  is_on H PQ ∧ is_on G PS ∧ PH = 8 ∧ SG = 4

-- The theorem to prove
theorem area_of_rectangle_PQRS (h : conditions PQRS P Q R S H G PH SG)
  : area PQRS = 192 :=
sorry

end area_of_rectangle_PQRS_l735_735595


namespace cos_double_angle_zero_l735_735844

open Real

theorem cos_double_angle_zero (α : ℝ) (h : sin (π / 6 - α) = cos (π / 6 + α)) : cos (2 * α) = 0 :=
by
  sorry

end cos_double_angle_zero_l735_735844


namespace prove_average_sales_volume_and_regression_correctness_l735_735205

-- Definitions based on the problem statement
def unit_prices : List ℕ := [4, 5, 6, 7, 8, 9]
def sales_volumes : List ℕ := [90, 84, 83, 80, 75, 68]
def regression_slope : ℤ := -4
def regression_intercept (prices : List ℕ) (volumes : List ℕ) : ℤ := 106 -- from solution

-- Problem statement in Lean format
theorem prove_average_sales_volume_and_regression_correctness :
  (List.sum sales_volumes / sales_volumes.length = 80) ∧
  (∃ a : ℤ, regression_intercept unit_prices sales_volumes = a ∧
    ∀ x: ℕ, x ≥ 0 → x ∈ unit_prices → (∃ y : ℤ, y = regression_slope * x + a ∧ y ∈ sales_volumes)) :=
by
  sorry

end prove_average_sales_volume_and_regression_correctness_l735_735205


namespace number_of_dogs_l735_735646

theorem number_of_dogs (cost_price selling_price total_amount : ℝ) (profit_percentage : ℝ)
    (h1 : cost_price = 1000)
    (h2 : profit_percentage = 0.30)
    (h3 : total_amount = 2600)
    (h4 : selling_price = cost_price + (profit_percentage * cost_price)) :
    total_amount / selling_price = 2 :=
by
  sorry

end number_of_dogs_l735_735646


namespace median_of_list_l735_735601

def list_spec : List ℕ := List.join (List.map (λ n => List.repeat n n) (List.range' 1 200))

theorem median_of_list : (list_spec.nth 10049).iget = 141 ∧ (list_spec.nth 10050).iget = 141 :=
by
  sorry

end median_of_list_l735_735601


namespace Points_PQRC_are_concyclic_l735_735932

open EuclideanGeometry

variable {A B C A1 B1 P S Q R : Point}

-- Definitions of the problem's conditions
def is_acute_angled_triangle (A B C : Point) : Prop := is_triangle A B C ∧ acute_angle (∠BC A) ∧ acute_angle (∠CA B) ∧ acute_angle (∠AB C)
def intersects (circle : set Point) (segment : set Point) (P : Point) : Prop := P ∈ circle ∧ on_line_segment P segment
def reflection (S : Point) (line : set Point) (R : Point) : Prop := ∃ M : Point, on_midpoint M S R ∧ on_line M line

-- Given data
axiom h1 : is_acute_angled_triangle A B C
axiom h2 : intersects (circumcircle A B) (line_segment C A) A1
axiom h3 : intersects (circumcircle A B) (line_segment C B) B1
axiom h4 : intersects (circumcircle A C B) (circumcircle A1 B1 C) P
axiom h5 : intersects (line_segment A B1) (line_segment B A1) S
axiom h6 : reflection S (line_segment C A) Q
axiom h7 : reflection S (line_segment C B) R

-- Goal to prove
theorem Points_PQRC_are_concyclic : concyclic P Q R C := 
sorry

end Points_PQRC_are_concyclic_l735_735932


namespace total_area_is_8_units_l735_735780

-- Let s be the side length of the original square and x be the leg length of each isosceles right triangle
variables (s x : ℕ)

-- The side length of the smaller square is 8 units
axiom smaller_square_length : s - 2 * x = 8

-- The area of one isosceles right triangle
def area_triangle : ℕ := x * x / 2

-- There are four triangles
def total_area_triangles : ℕ := 4 * area_triangle x

-- The aim is to prove that the total area of the removed triangles is 8 square units
theorem total_area_is_8_units : total_area_triangles x = 8 :=
sorry

end total_area_is_8_units_l735_735780


namespace find_fx_y_l735_735273

noncomputable def f : ℝ → ℝ → ℝ := sorry

axiom cond1 : f 1 2 = 2

axiom cond2 : ∀ (x y : ℝ), y * f x (f x y) = (f x y) ^ 2 ∧ x * f (f x y) y = (f x y) ^ 2

theorem find_fx_y : f = λ x y, x * y :=
by
  sorry

end find_fx_y_l735_735273


namespace small_beaker_salt_fraction_l735_735034

theorem small_beaker_salt_fraction
  (S L : ℝ) 
  (h1 : L = 5 * S)
  (h2 : L * (1 / 5) = S)
  (h3 : L * 0.3 = S * 1.5)
  : (S * 0.5) / S = 0.5 :=
by 
  sorry

end small_beaker_salt_fraction_l735_735034


namespace check_double_root_statements_l735_735911

-- Condition Definitions
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ r : ℝ, a * r^2 + b * r + c = 0 ∧ a * (2 * r)^2 + b * (2 * r) + c = 0

-- Statement ①
def statement_1 : Prop := ¬is_double_root_equation 1 2 (-8)

-- Statement ②
def statement_2 : Prop := is_double_root_equation 1 (-3) 2

-- Statement ③
def statement_3 (m n : ℝ) : Prop := 
  (∃ r : ℝ, (r - 2) * (m * r + n) = 0 ∧ (m * (2 * r) + n = 0) ∧ r = 2) → 4 * m^2 + 5 * m * n + n^2 = 0

-- Statement ④
def statement_4 (p q : ℝ) : Prop := 
  (p * q = 2 → is_double_root_equation p 3 q)

-- Main proof problem statement
theorem check_double_root_statements (m n p q : ℝ) : 
  statement_1 ∧ statement_2 ∧ statement_3 m n ∧ statement_4 p q :=
by
  sorry

end check_double_root_statements_l735_735911


namespace equivalent_problem_l735_735627

-- Definitions that correspond to conditions
def valid_n (n : ℕ) : Prop := n < 13 ∧ (4 * n) % 13 = 1

-- The equivalent proof problem
theorem equivalent_problem (n : ℕ) (h : valid_n n) : ((3 ^ n) ^ 4 - 3) % 13 = 6 := by
  sorry

end equivalent_problem_l735_735627


namespace sum_of_integer_solutions_l735_735099

theorem sum_of_integer_solutions (x : ℤ) :
    (x^4 - 13 * x^2 + 36 = 0) →
    (∃ s : ℤ, s = ∑ i in {2, -2, 3, -3}, i) ∧ s = 0 :=
by
  sorry

end sum_of_integer_solutions_l735_735099


namespace a_plus_b_equals_10_l735_735574

-- Given the conditions
variables {a b x : ℝ}
def equation_infinite_solutions := ∀ x : ℝ, a * x - 4 = 14 * x + b

-- The theorem to prove the required result
theorem a_plus_b_equals_10 (h : equation_infinite_solutions) : a + b = 10 :=
sorry

end a_plus_b_equals_10_l735_735574


namespace number_of_roses_is_44_l735_735347

noncomputable def flowers : ℕ → ℕ → ℕ
| b, g => b * g

theorem number_of_roses_is_44 :
  ∃ (x : ℕ), let b := 11 in let g := 17 in
  (28 = b + g) ∧ (17 * x = flowers b g) ∧ (4 * x = 44) :=
begin
  sorry
end

end number_of_roses_is_44_l735_735347


namespace discriminant_of_P_l735_735149

theorem discriminant_of_P (a b c : ℚ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1/2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1/2 := 
begin
  -- Proof omitted for brevity
  sorry
end

end discriminant_of_P_l735_735149


namespace inclination_angle_of_line_l735_735334

theorem inclination_angle_of_line (α : ℝ) (hα : 0 < α ∧ α < π / 2) :
  ∃ θ : ℝ, θ = (π / 2) + α ∧ 
           (∃ (m : ℝ), m = -Real.cot α ∧ (∀ x y : ℝ, x * Real.cos α + y * Real.sin α + 1 = 0 → m = -(x / y))) ∧ θ = Real.arctan (0) :=
begin
  sorry
end

end inclination_angle_of_line_l735_735334


namespace intersection_of_M_and_N_l735_735204

def M : Set ℤ := { x | x^2 ≤ 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }

theorem intersection_of_M_and_N :
  M ∩ N = {0, 1} := by
  sorry

end intersection_of_M_and_N_l735_735204


namespace number_of_freshmen_to_sample_l735_735022

-- Define parameters
def total_students : ℕ := 900
def sample_size : ℕ := 45
def freshmen_count : ℕ := 400
def sophomores_count : ℕ := 300
def juniors_count : ℕ := 200

-- Define the stratified sampling calculation
def stratified_sampling_calculation (group_size : ℕ) (total_size : ℕ) (sample_size : ℕ) : ℕ :=
  (group_size * sample_size) / total_size

-- Theorem stating that the number of freshmen to be sampled is 20
theorem number_of_freshmen_to_sample : stratified_sampling_calculation freshmen_count total_students sample_size = 20 := by
  sorry

end number_of_freshmen_to_sample_l735_735022


namespace shark_sightings_in_cape_may_l735_735453

theorem shark_sightings_in_cape_may (x : ℕ) (hx : x + (2 * x - 8) = 40) : 2 * x - 8 = 24 := 
by 
  sorry

end shark_sightings_in_cape_may_l735_735453


namespace alpha_inverse_proportional_beta_l735_735313

theorem alpha_inverse_proportional_beta (α β : ℝ) (k : ℝ) :
  (∀ β1 α1, α1 * β1 = k) → (4 * 2 = k) → (β = -3) → (α = -8/3) :=
by
  sorry

end alpha_inverse_proportional_beta_l735_735313


namespace friendly_two_digit_count_l735_735029

def is_friendly (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  n % digits.sum = 0

def is_two_digit (n : ℕ) : Prop := 
  10 ≤ n ∧ n < 100

theorem friendly_two_digit_count : 
  {n : ℕ | is_two_digit n ∧ is_friendly n}.to_finset.card = 23 := 
sorry

end friendly_two_digit_count_l735_735029


namespace second_markdown_percentage_l735_735779

theorem second_markdown_percentage (P : ℝ) (h1 : P > 0)
    (h2 : ∃ x : ℝ, x = 0.50 * P) -- First markdown
    (h3 : ∃ y : ℝ, y = 0.45 * P) -- Final price
    : ∃ X : ℝ, X = 10 := 
sorry

end second_markdown_percentage_l735_735779


namespace no_real_solutions_l735_735840

theorem no_real_solutions (k d : ℝ) :
  (∀ (x y : ℝ), x^3 + y^3 = 2 → y = k * x + d → false) ↔ (k = -1 ∧ 0 < d ∧ d < 2 * real.sqrt 2) :=
sorry

end no_real_solutions_l735_735840


namespace wire_lengths_l735_735041

variables (total_length first second third fourth : ℝ)

def wire_conditions : Prop :=
  total_length = 72 ∧
  first = second + 3 ∧
  third = 2 * second - 2 ∧
  fourth = 0.5 * (first + second + third) ∧
  second + first + third + fourth = total_length

theorem wire_lengths 
  (h : wire_conditions total_length first second third fourth) :
  second = 11.75 ∧ first = 14.75 ∧ third = 21.5 ∧ fourth = 24 :=
sorry

end wire_lengths_l735_735041


namespace probability_of_no_defective_pencils_l735_735924

open Nat

-- Define the total number of ways to select 5 pencils out of 15
def total_ways_to_choose : ℕ := (choose 15 5)

-- Define the number of ways to choose 5 non-defective pencils from 11 non-defective
def non_defective_ways_to_choose : ℕ := (choose 11 5)

-- Define the probability as a rational number
def probability_none_defective : ℚ := (non_defective_ways_to_choose : ℚ) / (total_ways_to_choose : ℚ)

theorem probability_of_no_defective_pencils :
  probability_none_defective = 154 / 1001 :=
by
  -- We assume this proof can be constructed from provided combinatorial calculations
  -- Add the necessary definitions to calculate total ways and non-defective ways
  sorry

end probability_of_no_defective_pencils_l735_735924


namespace O_is_incenter_l735_735842

variable {n : ℕ}
variable (A : Fin n → ℝ × ℝ)
variable (O : ℝ × ℝ)

-- Conditions
def inside_convex_ngon (O : ℝ × ℝ) (A : Fin n → ℝ × ℝ) : Prop := sorry
def angles_acute (O : ℝ × ℝ) (A : Fin n → ℝ × ℝ) : Prop := sorry
def angles_inequality (O : ℝ × ℝ) (A : Fin n → ℝ × ℝ) : Prop := sorry

-- This is the statement that we need to prove.
theorem O_is_incenter 
  (h1 : inside_convex_ngon O A)
  (h2 : angles_acute O A) 
  (h3 : angles_inequality O A) 
: sorry := sorry

end O_is_incenter_l735_735842


namespace circle_circumference_l735_735582

theorem circle_circumference (r : ℝ) (h : 2 * r^2 = real.pi * r^2) : 2 * real.pi * r = 4 * r :=
by
  sorry

end circle_circumference_l735_735582


namespace nancy_money_in_euros_l735_735986

noncomputable def convert_to_euros (quarters dozens : ℕ) (five_dollar_bills : ℕ) (dimes_dozens : ℕ) (exchange_rate : ℝ) : ℝ :=
  let quarter_value := 0.25
  let dime_value := 0.10
  let total_dollars := (dozens * 12 * quarter_value) + (five_dollar_bills * 5) + (dimes_dozens * 24 * dime_value)
  total_dollars / exchange_rate

theorem nancy_money_in_euros :
  convert_to_euros 1 3 2 1.12 = 18.21 :=
by
  sorry

end nancy_money_in_euros_l735_735986


namespace dragon_cannot_be_killed_l735_735015

def initial_heads := 100

def head_change (cut : ℕ) : ℤ :=
  if cut = 15 then 24 - 15
  else if cut = 17 then 2 - 17
  else if cut = 20 then 14 - 20
  else if cut = 5 then 17 - 5
  else 0

theorem dragon_cannot_be_killed : 
  (∀ (cut : ℕ), cut = 15 ∨ cut = 17 ∨ cut = 20 ∨ cut = 5 →
    let final_heads := initial_heads + head_change cut in final_heads > 0) :=
by
  sorry

end dragon_cannot_be_killed_l735_735015


namespace number_of_ways_difference_of_squares_l735_735620

-- Lean statement
theorem number_of_ways_difference_of_squares (n k : ℕ) (h1 : n > 10^k) (h2 : n % 10^k = 0) (h3 : k ≥ 2) :
  ∃ D, D = k^2 - 1 ∧ ∀ (a b : ℕ), n = a^2 - b^2 → D = k^2 - 1 :=
by
  sorry

end number_of_ways_difference_of_squares_l735_735620


namespace find_initial_term_common_difference_l735_735180

noncomputable def arithmetic_geometric_sequences (a b : ℕ → ℝ) (d : ℝ) : Prop :=
  (∀ n, a n = a 0 + n * d) ∧
  (b 0 = (a 0)^2) ∧ (b 1 = (a 1)^2) ∧ (b 2 = (a 2)^2) ∧
  (b 1 / b 0 = (b 2 / b 1)^2) ∧
  (∃ l, ∀ ε > 0, ∃ N, ∀ n ≥ N, abs ((∑ i in finset.range (n + 1), b i) - l) < ε) ∧
  (l = sqrt 2 + 1)

theorem find_initial_term_common_difference 
  {a b : ℕ → ℝ} {d a1 : ℝ} 
  (h_arithmetic_geometric_sequences : arithmetic_geometric_sequences a b d) :
  a 0 = -sqrt 2 ∧ d = 2 * sqrt 2 - 2 :=
sorry

end find_initial_term_common_difference_l735_735180


namespace find_k_l735_735222

theorem find_k (k : ℤ) :
  (∃ a b c : ℤ, a = 49 + k ∧ b = 441 + k ∧ c = 961 + k ∧
  (∃ r : ℚ, b = r * a ∧ c = r * r * a)) ↔ k = 1152 := by
  sorry

end find_k_l735_735222


namespace role_of_scatter_plot_correct_l735_735688

-- Definitions for problem context
def role_of_scatter_plot (role : String) : Prop :=
  role = "Roughly judging whether variables are linearly related"

-- Problem and conditions
theorem role_of_scatter_plot_correct :
  role_of_scatter_plot "Roughly judging whether variables are linearly related" :=
by 
  sorry

end role_of_scatter_plot_correct_l735_735688


namespace AK_squared_eq_LK_mul_KM_l735_735991

variables {A B C D K L M : Type*}
variables [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]
variables [V : Module ℝ K] [V : Module ℝ L] [V : Module ℝ M]

-- Define the parallelogram ABCD
-- This uses the fact that diagonal cuts through and K is a point on BD
noncomputable def isParallelogram (A B C D K L M : Type*)
  [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]
  [V : Module ℝ K] [V : Module ℝ L] [V : Module ℝ M] := sorry

-- Define the intersections AK ∩ BC = L and AK ∩ CD = M
noncomputable def intersection_AK_BC_CD (A B C D K L M : Type*)
  [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]
  [V : Module ℝ K] [V : Module ℝ L] [V : Module ℝ M] := sorry

theorem AK_squared_eq_LK_mul_KM
  (A B C D K L M : Type*)
  [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]
  [isParallelogram A B C D K L M]
  [intersection_AK_BC_CD A B C D K L M] :
  (dist A K) ^ 2 = (dist L K) * (dist K M) :=
sorry

end AK_squared_eq_LK_mul_KM_l735_735991


namespace ideal_complex_number_condition_l735_735226

def is_ideal_complex_number (z : ℂ) : Prop :=
  z.re = -z.im

noncomputable def complex_z (a b : ℝ) : ℂ :=
  a / (1 - 2 * complex.I) + b * complex.I

theorem ideal_complex_number_condition (a b : ℝ) (h : is_ideal_complex_number (complex_z a b)) : 3 * a + 5 * b = 0 :=
  sorry

end ideal_complex_number_condition_l735_735226


namespace polynomial_discriminant_l735_735127

theorem polynomial_discriminant (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
by
  sorry

end polynomial_discriminant_l735_735127


namespace triangle_NMC_area_l735_735070

variables {A B C D L M N K : Type} [Square A B C D]
variables {x y : ℝ} (h : 1 - x - y = sqrt (x^2 + y^2))

theorem triangle_NMC_area 
  (h_sq : IsSquare A B C D)
  (h_L_on_AB : LiesOn L A B)
  (h_M_on_BC : LiesOn M B C)
  (h_N_on_CD : LiesOn N C D)
  (h_K_on_AD : LiesOn K A D)
  (h_AL : SegmentLength A L = x)
  (h_AK : SegmentLength A K = y)
  : TriangleArea N M C = 1 / 4 :=
by 
  sorry

end triangle_NMC_area_l735_735070


namespace eccentricity_of_hyperbola_l735_735652

variable {a b c : ℝ} (eq_c : ∀ x y : ℝ, (x / a) ^ 2 - (y / b) ^ 2 = 1)
variable (a_pos : a > 0) (b_pos : b > 0)
variable (M_center : c : ℝ, ∃ n : ℝ, c ^ 2 - (n * b) ^ 2 = 1 ∧ n > 0)
variable (circle_center : ∃ M : ℝ × ℝ, M = (c, (b ^ 2 / a)) ∧ M.2 > 0)
variable (tangent_to_x : ∃ F : ℝ × ℝ, F.1 = c ∧ F.2 = 0)
variable (eq_triangle : ∃ P Q : ℝ × ℝ, dist P Q = dist P (c, b^2 / a) ∧ dist Q (c, b^2 / a) = dist P (c, b^2 / a))

theorem eccentricity_of_hyperbola :
  ∃ e : ℝ, e = Real.sqrt 3 :=
  sorry

end eccentricity_of_hyperbola_l735_735652


namespace purchase_price_first_batch_lowest_discount_needed_l735_735781

-- Definitions based on the problem conditions
def batch1_cost : ℝ := 1050
def batch1_price_per_box (x : ℝ) := x
def batch2_cost : ℝ := 1440
def batch2_price_per_box (x : ℝ) := 1.2 * x

def boxes_count_first_batch (x : ℝ) := batch1_cost / x
def boxes_count_second_batch (x : ℝ) := batch2_cost / (1.2 * x) + 10

-- Part (1): Purchase price per box for the first batch
theorem purchase_price_first_batch :
  ∃ x : ℝ, boxes_count_first_batch x = batch2_cost / (1.2 * x) - 10 ∧ x = 15 := sorry

-- Conditions for second part: 
def profit_first_half_sold (x : ℝ) := (24 - batch2_price_per_box x) * (1/2) * (boxes_count_second_batch x)
def profit_second_half_sold (x : ℝ) (m : ℝ) := (24 * (m / 10) - batch2_price_per_box x) * (1/2) * (boxes_count_second_batch x)
def total_profit (x : ℝ) (m : ℝ) := profit_first_half_sold x + profit_second_half_sold x m

-- Part (2): Lowest discount required
theorem lowest_discount_needed :
  ∃ (x m : ℝ), x = 15 ∧ total_profit x m ≥ 288 ∧ m = 8 := sorry

end purchase_price_first_batch_lowest_discount_needed_l735_735781


namespace equation_involving_x_and_y_l735_735552

variable (x y : ℝ)

theorem equation_involving_x_and_y :
  (x + y) / 3 = 1.888888888888889 → 
  x + 2y = 10 →
  x + y = 5.666666666666667 :=
by
  sorry

end equation_involving_x_and_y_l735_735552


namespace simplify_and_evaluate_expression_l735_735663

theorem simplify_and_evaluate_expression (m : ℕ) (h : m = 2) :
  ( (↑m + 1) / (↑m - 1) + 1 ) / ( (↑m + m^2) / (m^2 - 2*m + 1) ) - ( 2 - 2*↑m ) / ( m^2 - 1 ) = 4 / 3 :=
by sorry

end simplify_and_evaluate_expression_l735_735663


namespace smallest_m_for_9_step_sequence_l735_735616

theorem smallest_m_for_9_step_sequence :
  ∃ m : ℕ, 
    (∀ (seq : ℕ → ℕ), 
      (seq 0 = m) ∧
      (∀ n : ℕ, seq (n + 1) = seq n - (nat.sqrt seq n)^2) ∧
      seq (9) = 0) →
    m = 73 :=
begin
  sorry
end

end smallest_m_for_9_step_sequence_l735_735616


namespace dot_product_of_wz_l735_735972
noncomputable def unit_vectors {n : ℕ} (w x y z : ℝ^n) : Prop :=
  -- Defined unit vector properties
  ∥w∥ = 1 ∧ ∥x∥ = 1 ∧ ∥y∥ = 1 ∧ ∥z∥ = 1 ∧ 
  w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z

noncomputable def dot_product_properties {n : ℕ} (w x y z : ℝ^n) : Prop :=
  -- Given properties for dot products
  w • x = 0 ∧ 
  w • y = 1 / 5 ∧ 
  x • y = -1 / 5 ∧ 
  x • z = 1 / 7 ∧ 
  y • z = -2 / 7

theorem dot_product_of_wz {n : ℕ} (w x y z : ℝ^n) 
  (h1 : unit_vectors w x y z) 
  (h2 : dot_product_properties w x y z) :
  w • z = 2 / 35 := sorry

end dot_product_of_wz_l735_735972


namespace quadratic_two_distinct_zeros_probability_l735_735121

open Real Set

noncomputable def given_set : Set ℝ :=
  {log 2 + log 5, log 3 / log 4, (1 / 3) ^ (-3 / 5), tan 1}

def probability_two_distinct_zeros (f : ℝ → ℝ) : ℝ :=
  let pairs := {p : ℝ × ℝ | p.1 ∈ given_set ∧ p.2 ∈ given_set}
  let satisfying_pairs := {p : ℝ × ℝ | p.1 ∈ given_set ∧ p.2 ∈ given_set ∧ p.1 > p.2}
  (satisfying_pairs.to_finset.card : ℝ) / (pairs.to_finset.card : ℝ)

#eval probability_two_distinct_zeros (λ x, x^2 + 2 * (log 2 + log 5) * x + (log 2 + log 5)^2)

theorem quadratic_two_distinct_zeros_probability :
  probability_two_distinct_zeros (λ x, x^2 + 2 * (log 2 + log 5) * x + (log 2 + log 5)^2) = 3 / 8 :=
sorry

end quadratic_two_distinct_zeros_probability_l735_735121


namespace quadratic_polynomial_discriminant_l735_735134

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₁ : a ≠ 0)
  (h₂ : ∃ x : ℝ, a * x^2 + b * x + c = x - 2 ∧ (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h₃ : ∃ x : ℝ, a * x^2 + b * x + c = 1 - x / 2 ∧ (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l735_735134


namespace discriminant_of_P_l735_735155

theorem discriminant_of_P (a b c : ℚ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1/2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1/2 := 
begin
  -- Proof omitted for brevity
  sorry
end

end discriminant_of_P_l735_735155


namespace sally_total_spent_l735_735662

-- Define the prices paid by Sally for peaches after the coupon and for cherries
def P_peaches : ℝ := 12.32
def C_cherries : ℝ := 11.54

-- State the problem to prove that the total amount Sally spent is 23.86
theorem sally_total_spent : P_peaches + C_cherries = 23.86 := by
  sorry

end sally_total_spent_l735_735662


namespace linear_regression_equation_l735_735030

-- Given conditions
variables {n : ℕ} (x y : Fin n → ℝ)
variables (Σ_x : ℝ) (Σ_y : ℝ) (Σ_xy : ℝ) (Σ_x2 : ℝ)
variables [Fact (n = 10)] [Fact (Σ_x = 80)] [Fact (Σ_y = 20)]
          [Fact (Σ_xy = 184)] [Fact (Σ_x2 = 720)]

-- Calculated mean values
noncomputable def x̄ : ℝ := Σ_x / n
noncomputable def ȳ : ℝ := Σ_y / n

-- Calculated sums of squares
noncomputable def l_xx : ℝ := Σ_x2 - n * x̄ ^ 2
noncomputable def l_xy : ℝ := Σ_xy - n * x̄ * ȳ

-- Calculated regression coefficients
noncomputable def b̂ : ℝ := l_xy / l_xx
noncomputable def â : ℝ := ȳ - b̂ * x̄

-- Predicted regression equation
noncomputable def ŷ (x : ℝ) : ℝ := b̂ * x + â

-- Lean 4 statement for the proof problem
theorem linear_regression_equation :
  (ŷ x = 0.3 * x - 0.4) ∧ (b̂ > 0) ∧ (ŷ 7 = 1.7) :=
by
  -- The proof is omitted
  sorry

end linear_regression_equation_l735_735030


namespace math_problem_l735_735258

noncomputable def problem_statement (f : ℚ → ℝ) : Prop :=
  (∀ r s : ℚ, ∃ n : ℤ, f (r + s) = f r + f s + n) →
  ∃ (q : ℕ) (p : ℤ), abs (f (1 / q) - p) ≤ 1 / 2012

-- To state this problem as a theorem in Lean 4
theorem math_problem (f : ℚ → ℝ) :
  problem_statement f :=
sorry

end math_problem_l735_735258


namespace tangent_circumcircles_iff_perpendicular_l735_735800

variables (A B C D I K L M N E F X Y Z T : Type)
variables [ConvexCircumscribedQuadrilateral A B C D I K L M N E F X Y Z T]

def circumcircle_of_triangle_XFY (A B X F Y : Type) : Circle A B := sorry
def circle_with_diameter_EI (E I : Type) : Circle E I := sorry
def circumcircle_of_triangle_TEZ (T E Z : Type) : Circle T E := sorry
def circle_with_diameter_FI (F I : Type) : Circle F I := sorry
def perpendicular (X Y : Type) : Prop := sorry

theorem tangent_circumcircles_iff_perpendicular:
  Tangent (circumcircle_of_triangle_XFY A B X F Y) (circle_with_diameter_EI E I) ↔
  Tangent (circumcircle_of_triangle_TEZ T E Z) (circle_with_diameter_FI F I) :=
begin
  let KM := sorry,
  let LN := sorry,
  exact KM ⊥ LN,
end

end tangent_circumcircles_iff_perpendicular_l735_735800


namespace angle_bc_l735_735531

variables (a b c : ℝ → ℝ → Prop) (theta : ℝ)

-- Definitions of parallelism and angle conditions
def parallel (x y : ℝ → ℝ → Prop) : Prop := ∀ p q r s : ℝ, x p q → y r s → p - q = r - s

def angle_between (x y : ℝ → ℝ → Prop) (θ : ℝ) : Prop := sorry  -- Assume we have a definition for angle between lines

-- Given conditions
axiom parallel_ab : parallel a b
axiom angle_ac : angle_between a c theta

-- Theorem statement
theorem angle_bc : angle_between b c theta :=
sorry

end angle_bc_l735_735531


namespace discriminant_of_P_l735_735150

theorem discriminant_of_P (a b c : ℚ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1/2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1/2 := 
begin
  -- Proof omitted for brevity
  sorry
end

end discriminant_of_P_l735_735150


namespace graph_with_degree_3_contains_cycle_l735_735230

theorem graph_with_degree_3_contains_cycle
  (G : Type) [graph G]
  (degree_3 : ∀ (v : vertex G), degree v = 3) :
  ∃ (C : cycle G), true := 
by {
  sorry
}

end graph_with_degree_3_contains_cycle_l735_735230


namespace lines_parallel_coeff_l735_735579

theorem lines_parallel_coeff (a : ℝ) :
  (∀ x y: ℝ, a * x + 2 * y = 0 → 3 * x + (a + 1) * y + 1 = 0) ↔ (a = -3 ∨ a = 2) :=
by
  sorry

end lines_parallel_coeff_l735_735579


namespace polynomial_integer_values_l735_735549

theorem polynomial_integer_values (a b c d : ℤ) (h1 : ∃ (n : ℤ), n = (a * (-1)^3 + b * (-1)^2 - c * (-1) - d))
  (h2 : ∃ (n : ℤ), n = (a * 0^3 + b * 0^2 - c * 0 - d))
  (h3 : ∃ (n : ℤ), n = (a * 1^3 + b * 1^2 - c * 1 - d))
  (h4 : ∃ (n : ℤ), n = (a * 2^3 + b * 2^2 - c * 2 - d)) :
  ∀ x : ℤ, ∃ m : ℤ, m = a * x^3 + b * x^2 - c * x - d :=
by {
  -- proof goes here
  sorry
}

end polynomial_integer_values_l735_735549


namespace trapezoid_proof_l735_735446

variables {A B C D E F G : Point}

-- Definitions and conditions from the problem
def is_isosceles_trapezoid (ABCD : quadrilateral) : Prop :=
ABCD.A, ABD.C ∥ BCD.C

def incircle_touches_point (BCD : triangle) (E : Point) : Prop :=
BCD.incircle.touches_side BCD.CD E

def lies_on_angle_bisector (DAC : angle) (F : Point) : Prop :=
DAC.bisector E ∋ F

def perpendicularly_intersects (EF : line) (CD : line) : Prop :=
⊥ EF CD

def circumcircle_intersects_again (ACF : triangle) (CD : line) (G : Point) : Prop :=
triangle.Circumcircle ACF ∩ CD = {G}

theorem trapezoid_proof 
  (h1 : is_isosceles_trapezoid AB_CD)
  (h2 : incircle_touches_point BCD E)
  (h3 : lies_on_angle_bisector DAC F)
  (h4 : perpendicularly_intersects EF CD)
  (h5 : circumcircle_intersects_again ACF CD G) :
  dist A F = dist F G :=
sorry

end trapezoid_proof_l735_735446


namespace interval_solution_l735_735904

def smallest_integer_greater_than (a: ℝ) : ℤ := ⌈a⌉
def largest_integer_not_greater_than (a: ℝ) : ℤ := ⌊a⌋

theorem interval_solution (x y : ℝ) 
    (hx : 3 * largest_integer_not_greater_than x + 2 * smallest_integer_greater_than y = 18)
    (hy : 3 * smallest_integer_greater_than x - largest_integer_not_greater_than y = 4) :
    (2 <= x ∧ x < 3 ∧ 5 <= y ∧ y < 6) :=
    sorry

end interval_solution_l735_735904


namespace count_numbers_without_seven_l735_735214

def does_not_contain_seven (n : ℕ) : Prop :=
  ¬(n.digits 10).contains 7

theorem count_numbers_without_seven :
  (Finset.range 2000).filter (λ n, does_not_contain_seven (n + 1)).card = 1457 := 
sorry

end count_numbers_without_seven_l735_735214


namespace triangle_BMN_perimeter_range_l735_735517

def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def intersection_point : ℝ := 2 / 3

def point_B : ℝ × ℝ := (1, 0)

noncomputable def f (x : ℝ) : ℝ :=
if x < intersection_point then 2 * real.sqrt x
else real.sqrt (3 * (4 - x^2)) / 2

theorem triangle_BMN_perimeter_range:
  ∀ (y : ℝ), (0 < y) → (y < 2 * real.sqrt (2 / 3)) → 
  ∃ (x₁ x₂ : ℝ), f x₁ = y ∧ f x₂ = y ∧ x₁ ≠ x₂ ∧ 
  (10 / 3) < 3 + (1 / 2) * x₂ ∧ 3 + (1 / 2) * x₂ < 4 := by
  sorry

end triangle_BMN_perimeter_range_l735_735517


namespace reflected_ray_deviation_l735_735023

noncomputable def deviation_angle (α β : ℝ) : ℝ :=
  2 * Real.arcsin (Real.sin α * Real.sin β)

theorem reflected_ray_deviation (α β : ℝ) :
  ∃ φ, φ = deviation_angle α β := by
  use 2 * Real.arcsin (Real.sin α * Real.sin β)
  simp [deviation_angle]
  sorry

end reflected_ray_deviation_l735_735023


namespace sufficient_condition_perpendicular_l735_735262

variables {Plane : Type*} [plane : linear_ordered_ring Plane]
variables (α β : Plane) 
variables (n m : set Plane)
variables [decidable_eq Plane]

/-- Define perpendicularity between two planes --/
def perp_planes (p q : set Plane) : Prop := sorry

/-- Define the condition that a line is perpendicular to a plane --/
def perp_line_plane (l : set Plane) (p : set Plane) : Prop := sorry

/-- Theorem: Given conditions, the line m is perpendicular to plane β --/
theorem sufficient_condition_perpendicular 
  (n_perp_α : perp_line_plane n α) 
  (n_perp_β : perp_line_plane n β) 
  (m_perp_α : perp_line_plane m α) : perp_line_plane m β := sorry

end sufficient_condition_perpendicular_l735_735262


namespace quadratic_discriminant_l735_735171

variable {a b c : ℝ}
variable (h₁ : a ≠ 0)
variable (h₂ : (b - 1)^2 - 4 * a * (c + 2) = 0)
variable (h₃ : (b + 1/2)^2 - 4 * a * (c - 1) = 0)

theorem quadratic_discriminant : b^2 - 4 * a * c = -1 / 2 := 
by
  have h₁' : (b - 1)^2 - 4 * a * (c + 2) = 0 := h₂
  have h₂' : (b + 1/2)^2 - 4 * a * (c - 1) = 0 := h₃
  sorry

end quadratic_discriminant_l735_735171


namespace probability_interval_l735_735513

noncomputable def ξ : ℝ → ℝ := sorry -- Definition of the random variable ξ
def σ^2 : ℝ := sorry -- Definition to specify the variance 

axiom ξ_normal : ∀ x, ξ x ∈ Normal 0 σ^2 -- ξ follows the normal distribution N(0, σ^2)
axiom P_ξ_gt_2 : Prob (ξ > 2) = 0.023 -- P(ξ > 2) = 0.023

theorem probability_interval :
  Prob (-2 ≤ ξ ∧ ξ ≤ 2) = 0.954 := 
sorry

end probability_interval_l735_735513


namespace area_of_A1B1C1D1_is_five_l735_735853

variable (Points : Type) [MetricSpace Points]

structure Parallelogram :=
(A B C D : Points)
(AA1 BB1 CC1 DD1 : Points)
(mid_A_DD1 : midpoint A (A : Points) DD1 = A)
(mid_B_AA1 : midpoint B (B : Points) AA1 = B)
(mid_C_BB1 : midpoint C (C : Points) BB1 = C)
(mid_D_CC1 : midpoint D (D : Points) CC1 = D)
(area_ABCD : area A B C D = 1)

noncomputable def area_A1B1C1D1 (p : Parallelogram) : ℝ :=
area p.AA1 p.BB1 p.CC1 p.DD1

theorem area_of_A1B1C1D1_is_five (p : Parallelogram) : area_A1B1C1D1 p = 5 :=
sorry

end area_of_A1B1C1D1_is_five_l735_735853


namespace parabola_segment_length_l735_735025

theorem parabola_segment_length {A B : Real × Real} {midpoint_x : Real} (h_parabola : ∀ (P : Real × Real), P ∈ [A, B] → (P.2)^2 = 8 * P.1)
  (h_midpoint_x : midpoint_x = 3)
  (h_midpoint : (A.1 + B.1) / 2 = midpoint_x) :
  dist A B = 10 := 
  sorry

end parabola_segment_length_l735_735025


namespace quadratic_polynomial_discriminant_l735_735135

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₁ : a ≠ 0)
  (h₂ : ∃ x : ℝ, a * x^2 + b * x + c = x - 2 ∧ (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h₃ : ∃ x : ℝ, a * x^2 + b * x + c = 1 - x / 2 ∧ (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l735_735135


namespace percentage_small_square_in_rectangle_l735_735031

-- Define the conditions
variables (s : ℝ) (w : ℝ) (l : ℝ) (k : ℝ)
  (h_w : w = 3/2 * s)
  (h_l : l = 3/2 * w)
  (h_k : k = 1/2 * s)

-- Define the areas
def area_large_square : ℝ := s^2
def area_small_square : ℝ := k^2
def area_rectangle : ℝ := w * l

-- Define the ratio and expected percentage
def ratio_small_to_large : ℝ := (area_small_square k) / (area_rectangle w l) * 100

-- The theorem stating the expected outcome
theorem percentage_small_square_in_rectangle : ratio_small_to_large s w l k h_w h_l h_k = 7.41 :=
  sorry

end percentage_small_square_in_rectangle_l735_735031


namespace parabola_directrix_l735_735090

theorem parabola_directrix (y : ℝ) : (∃ d : ℝ, d = 3) := 
  let h := 2
  let a := (-1 : ℝ) / 4
  by
    obtain ⟨focus_x, focus_y⟩ : ∃ x y : ℝ, (x, y) = (h - 1 / (4 * a), 0) :=
      ⟨(h - 1 / (4 * a)), 0, rfl⟩ -- focus calculation
    have directrix_x := (h + 1 / (4 * a))
    exact ⟨directrix_x, rfl⟩

end parabola_directrix_l735_735090


namespace corrected_mean_l735_735679

theorem corrected_mean (n : ℕ) (incorrect_mean : ℝ) (incorrect_observation correct_observation : ℝ)
  (h_n : n = 50)
  (h_incorrect_mean : incorrect_mean = 30)
  (h_incorrect_observation : incorrect_observation = 23)
  (h_correct_observation : correct_observation = 48) :
  (incorrect_mean * n - incorrect_observation + correct_observation) / n = 30.5 :=
by
  sorry

end corrected_mean_l735_735679


namespace sum_of_all_four_is_zero_l735_735586

variables {a b c d : ℤ}

theorem sum_of_all_four_is_zero (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_sum_rows : a + b = c + d) 
  (h_product_columns : a * c = b * d) :
  a + b + c + d = 0 := 
sorry

end sum_of_all_four_is_zero_l735_735586


namespace range_of_a1_l735_735239

noncomputable def infinite_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * q ^ n

lemma geometric_series_sum (a1 q : ℝ) (h : |q| < 1) :
  ∑' (n : ℕ), infinite_geometric_sequence a1 q n = a1 / (1 - q) := sorry

theorem range_of_a1 (a1 : ℝ) (q : ℝ) 
  (h : |q| < 1) 
  (h_sum : ∑' (n : ℕ), infinite_geometric_sequence a1 q n = 1 / 2) : 
  a1 ∈ (set.Ioo 0 (1/2) ∪ set.Ioo (1/2) 1) :=
begin
  sorry
end

end range_of_a1_l735_735239


namespace selling_price_of_radio_l735_735672

theorem selling_price_of_radio (cost_price : ℝ) (loss_percentage : ℝ) (selling_price : ℝ) :
  cost_price = 3300 →
  loss_percentage = 62.727272727272734 →
  selling_price = cost_price - (loss_percentage / 100) * cost_price →
  selling_price = 1230 :=
by
  intros h_cost h_loss h_eq
  rw [h_cost, h_loss] at h_eq
  exact h_eq

end selling_price_of_radio_l735_735672


namespace angle_construction_theoretical_basis_l735_735715

theorem angle_construction_theoretical_basis : 
  (∃ A B C : Type, ∀ (a1 a2 b1 b2 c1 c2 : Type),
  triangle_congruent A B C a1 a2 b1 b2 c1 c2 → equal_angles a1 a2 b1 b2 c1 c2) → 
  (SSS : Type) := 
by 
  sorry

end angle_construction_theoretical_basis_l735_735715


namespace find_m_l735_735804

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n - 4 else n / 3

def is_odd (n : ℤ) : Prop := n % 2 = 1

theorem find_m (m : ℤ) : is_odd m → g (g (g m)) = 5 ↔ m = 17 :=
by
  intro m_odd,
  split
  · intro h,
    sorry -- Proof omitted
  · intro h,
    subst h,
    sorry -- Proof omitted

end find_m_l735_735804


namespace circle_radius_five_iff_l735_735807

noncomputable def circle_eq_radius (x y : ℝ) (k : ℝ) : Prop :=
  x^2 + 8*x + y^2 + 4*y - k = 0

def is_circle_with_radius (r : ℝ) (x y : ℝ) (k : ℝ) : Prop :=
  circle_eq_radius x y k ↔ r = 5 ∧ k = 5

theorem circle_radius_five_iff (k : ℝ) :
  (∃ x y : ℝ, circle_eq_radius x y k) ↔ k = 5 :=
sorry

end circle_radius_five_iff_l735_735807


namespace geometric_sequence_sum_l735_735640

theorem geometric_sequence_sum (a1 r : ℝ) (S : ℕ → ℝ) :
  S 2 = 3 → S 4 = 15 →
  (∀ n, S n = a1 * (1 - r^n) / (1 - r)) → S 6 = 63 :=
by
  intros hS2 hS4 hSn
  sorry

end geometric_sequence_sum_l735_735640


namespace cherry_lollipops_are_integer_l735_735458

def bouquets (cherry_per_bouquet : Nat) (orange_per_bouquet : Nat) (num_bouquets : Nat) : Nat :=
  num_bouquets * cherry_per_bouquet

theorem cherry_lollipops_are_integer (cherry_per_bouquet orange_per_bouquet : Nat) (num_bouquets : Nat) 
  (num_bouquets = 2) (orange_per_bouquet = 6) :
    ∃ C : Nat, cherry_per_bouquet = C :=
by
  sorry

end cherry_lollipops_are_integer_l735_735458


namespace complex_fifth_roots_wrong_statement_l735_735967

noncomputable def x : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)
noncomputable def y : ℂ := Complex.exp (-2 * Real.pi * Complex.I / 5)

theorem complex_fifth_roots_wrong_statement :
  ¬(x^5 + y^5 = 1) :=
sorry

end complex_fifth_roots_wrong_statement_l735_735967


namespace represent_sum_and_product_eq_231_l735_735661

theorem represent_sum_and_product_eq_231 :
  ∃ (x y z w : ℕ), x = 3 ∧ y = 7 ∧ z = 11 ∧ w = 210 ∧ (231 = x + y + z + w) ∧ (231 = x * y * z) :=
by
  -- The proof is omitted here.
  sorry

end represent_sum_and_product_eq_231_l735_735661


namespace problem1_problem2_l735_735796

variable a b c d e : ℚ

-- Condition definitions
def condition1 := a = 3 / (-1 / 2)
def condition2 := b = (2 / 5 - 1 / 3) * 15
def condition3 := c = (-3)^2
def condition4 := d = (-2)^3
def condition5 := e = -1 + 6

-- Problem statement for part 1
theorem problem1 (h1 : condition1) (h2 : condition2) : a - b = -7 := 
by {
  rw h1,
  rw h2,
  sorry
}

-- Problem statement for part 2
theorem problem2 (h3 : condition3) (h4 : condition4) (h5 : condition5) : 
  c - d * (-1 / 4) - e = 2 := 
by {
  rw h3,
  rw h4,
  rw h5,
  sorry
}

end problem1_problem2_l735_735796


namespace find_number_l735_735751

theorem find_number (x : ℝ) (h : 0.50 * x = 48 + 180) : x = 456 :=
sorry

end find_number_l735_735751


namespace perpendicularity_of_MG_to_h_l735_735788

noncomputable def ellipse (F G : Point) (a : ℝ) : Ellipse :=
  sorry

noncomputable def intersect_line_ellipse (h : Line) (ellipse : Ellipse) : Point × Point :=
  sorry

noncomputable def circle_with_radius (center : Point) (radius : ℝ) : Circle :=
  sorry

noncomputable def tangents_intersect (circle : Circle) (P1 P2 : Point) : Point :=
  sorry

theorem perpendicularity_of_MG_to_h :
  ∀ (F G M P1 P2 : Point) (a : ℝ) (h : Line),
    let e := ellipse F G a in
    let (P1, P2) := intersect_line_ellipse h e in
    let c := circle_with_radius F (2 * a) in
    let M := tangents_intersect c P1 P2 in
    MG ⊥ h := 
sorry

end perpendicularity_of_MG_to_h_l735_735788


namespace dorothy_needs_more_money_l735_735813

structure Person :=
  (age : ℕ)

def Discount (age : ℕ) : ℝ :=
  if age <= 11 then 0.5 else
  if age >= 65 then 0.8 else
  if 12 <= age && age <= 18 then 0.7 else 1.0

def ticketCost (age : ℕ) : ℝ :=
  (10 : ℝ) * Discount age

def specialExhibitCost : ℝ := 5

def totalCost (family : List Person) : ℝ :=
  (family.map (λ p => ticketCost p.age + specialExhibitCost)).sum

def salesTaxRate : ℝ := 0.1

def finalCost (family : List Person) : ℝ :=
  let total := totalCost family
  total + (total * salesTaxRate)

def dorothy_money_after_trip (dorothy_money : ℝ) (family : List Person) : ℝ :=
  dorothy_money - finalCost family

theorem dorothy_needs_more_money :
  dorothy_money_after_trip 70 [⟨15⟩, ⟨10⟩, ⟨40⟩, ⟨42⟩, ⟨65⟩] = -1.5 := by
  sorry

end dorothy_needs_more_money_l735_735813


namespace proj_6v_w_l735_735963

open Real
open Matrix

-- Define the projection function
noncomputable def proj (w v : Vector ℝ) : Vector ℝ :=
  (v ⬝ w) / (w ⬝ w) * w

-- Let v and w be vectors in ℝ²
variables (v w : Vector ℝ)
-- Given condition
axiom proj_v_w : proj w v = ⟨[4, 3]⟩

-- Prove that proj w (6 * v) = ⟨[24, 18]⟩
theorem proj_6v_w : proj w (6 • v) = ⟨[24, 18]⟩ :=
sorry

end proj_6v_w_l735_735963


namespace triangle_AGE_area_l735_735291

-- Define point and square
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the square
def A : Point := ⟨0, 0⟩
def B : Point := ⟨5, 0⟩
def C : Point := ⟨5, 5⟩
def D : Point := ⟨0, 5⟩

-- Define point E
def E : Point := ⟨5, 2⟩

-- Define point G and prove that the area of triangle AGE is 4.25
theorem triangle_AGE_area : 
  let G := Point.mk (real.sqrt 2.25) (5 - (real.sqrt 2.25)) in
  let area (A G E : Point) : ℝ := 
    0.5 * abs (A.x * (G.y - E.y) + G.x * (E.y - A.y) + E.x * (A.y - G.y)) in
  area A G E = 4.25 :=
by
  let G : Point := ⟨(3 : ℝ), ((5 - 3) : ℝ)⟩
  let area : ℝ := 
    0.5 * abs (A.x * (G.y - E.y) + G.x * (E.y - A.y) + E.x * (A.y - G.y))
  have h : area = 4.25, by sorry
  exact h

end triangle_AGE_area_l735_735291


namespace problem1_problem2_problem3_l735_735414

-- Proof for part 1
theorem problem1 (α : ℝ) (h : Real.tan α = 3) :
  (3 * Real.sin α + Real.cos α) / (Real.sin α - 2 * Real.cos α) = 10 :=
sorry

-- Proof for part 2
theorem problem2 (α : ℝ) :
  (-Real.sin (Real.pi + α) + Real.sin (-α) - Real.tan (2 * Real.pi + α)) / 
  (Real.tan (α + Real.pi) + Real.cos (-α) + Real.cos (Real.pi - α)) = -1 :=
sorry

-- Proof for part 3
theorem problem3 (α : ℝ) (h : Real.sin α + Real.cos α = 1 / 2) (hα : 0 < α ∧ α < Real.pi) :
  Real.sin α * Real.cos α = -3 / 8 :=
sorry

end problem1_problem2_problem3_l735_735414


namespace largest_common_divisor_510_399_l735_735708

theorem largest_common_divisor_510_399 : ∃ d, d ∣ 510 ∧ d ∣ 399 ∧ ∀ e, e ∣ 510 ∧ e ∣ 399 → e ≤ d :=
begin
  use 57,
  split,
  { sorry },  -- placeholder for proof that 57 divides 510
  split,
  { sorry },  -- placeholder for proof that 57 divides 399
  { assume e h,
    sorry }  -- placeholder for proof that any common divisor must be <= 57
end

end largest_common_divisor_510_399_l735_735708


namespace boxes_calculation_proof_l735_735697

variable (baskets : ℕ) (eggs_per_basket : ℕ) (eggs_per_box : ℕ)
variable (total_eggs : ℕ := baskets * eggs_per_basket)
variable (boxes_needed : ℕ := total_eggs / eggs_per_box)

theorem boxes_calculation_proof :
  baskets = 21 →
  eggs_per_basket = 48 →
  eggs_per_box = 28 →
  boxes_needed = 36 :=
by
  intros
  sorry

end boxes_calculation_proof_l735_735697


namespace inequality_solution_range_l735_735839

theorem inequality_solution_range (a : ℝ) :
  (∃ x ∈ set.Icc (1 : ℝ) 4, x^2 + a * x - 2 < 0) ↔ a < 1 := 
sorry

end inequality_solution_range_l735_735839


namespace polynomial_discriminant_l735_735132

theorem polynomial_discriminant (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
by
  sorry

end polynomial_discriminant_l735_735132


namespace angle_difference_equals_arc_measure_l735_735122

theorem angle_difference_equals_arc_measure
  (O : Point)
  (ω : Circle O)
  (P B A Q : Point)
  (hPA : P ∈ ω)
  (hBA : B ∈ ω)
  (hAA : A ∈ ω)
  (hQ : IsInterior Q ω)
  (hAngle : ∠ PAQ = 90)
  (hEq : PQ = BQ) :
  (∠ AQB - ∠ PQA) = (arc_length AB) :=
sorry

end angle_difference_equals_arc_measure_l735_735122


namespace imaginary_part_of_z_plus_inv_z_l735_735194

-- Define the complex number z
def z : ℂ := 1 + I

-- Define the statement to prove
theorem imaginary_part_of_z_plus_inv_z : (z + (1 / z)).im = 1 / 2 :=
by
  sorry

end imaginary_part_of_z_plus_inv_z_l735_735194


namespace units_digit_n_l735_735492

theorem units_digit_n (m n : ℕ) (hm : m % 10 = 9) (h : m * n = 18^5) : n % 10 = 2 :=
sorry

end units_digit_n_l735_735492


namespace cards_least_likely_red_after_flips_l735_735351

theorem cards_least_likely_red_after_flips :
  ∃ (k1 k2 : ℕ), 1 ≤ k1 ∧ k1 ≤ 50 ∧ 1 ≤ k2 ∧ k2 ≤ 50 ∧ (k1 = 13 ∧ k2 = 38) ∧ 
  (∀ k ∈ finset.range 1 51, 
    let p := (if k ≤ 25 then ((26 - k) ^ 2 + k ^ 2) / 676 else ((26 - (51 - k)) ^ 2 + (51 - k) ^ 2) / 676) in
    p ≥ (if k = 13 ∨ k = 38 then ((26 - k) ^ 2 + k ^ 2) / 676 else p)) :=
sorry

end cards_least_likely_red_after_flips_l735_735351


namespace range_of_a_l735_735893

-- Defining propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Icc (-1 : ℝ) 1, 3 * x^2 - a < 0
def q (a : ℝ) : Prop := a^2 - 4 < 0

-- Main theorem stating the equivalent problem
theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ (-2 < a ∧ a < 2) ∨ (3 ≤ a) :=
by
  sorry

end range_of_a_l735_735893


namespace count_two_digit_numbers_with_double_sum_of_digits_five_l735_735962

def sum_of_digits (x : ℕ) : ℕ :=
  x.digits.sum

def double_sum_of_digits_is_five (x : ℕ) : Prop :=
  sum_of_digits (sum_of_digits x) = 5

def is_two_digit (x : ℕ) : Prop :=
  10 ≤ x ∧ x < 100

theorem count_two_digit_numbers_with_double_sum_of_digits_five :
  (finset.range 100).filter (λ x, is_two_digit x ∧ double_sum_of_digits_is_five x).card = 10 :=
by sorry

end count_two_digit_numbers_with_double_sum_of_digits_five_l735_735962


namespace angle_ratio_l735_735937

theorem angle_ratio (CP CQ CM : Point -> Point -> Angle) (A C B M : Point) (P Q : Point) (∠ACB ∠MCQ y : ℝ)
    (trisect : ∠ACB / 3 = CP ∠ C ∧ ∠ACB / 3 = CQ ∠ C)
    (bisect : CM ∠ C P = CM ∠ C Q) :
    ∠MCQ / (4 * y) = 1 / 4 :=
by
    sorry

end angle_ratio_l735_735937


namespace trigonometric_expression_l735_735103

theorem trigonometric_expression : 
  (sin 10 * sin 80) / (cos 35 ^ 2 - sin 35 ^ 2) = 1 / 2 :=
by
  sorry

end trigonometric_expression_l735_735103


namespace beautiful_value_part1_beautiful_value_part2_beautiful_value_part3_l735_735066

def beautiful_two_var_eqn (a b x y : ℝ) := a * x + y = b

/-- Given a beautiful two-variable linear equation 5x - y = 1, prove the beautiful value is 1/3 -/
theorem beautiful_value_part1 : ∀ (x : ℝ), 5 * x - 2 * x = 1 → x = 1 / 3 := 
by 
  intro x h
  simp [←beautiful_two_var_eqn] at h
  sorry

/-- Given a beautiful two-variable linear equation 1/3x + y = m and beautiful value is -3, prove m = -7 -/
theorem beautiful_value_part2 (m : ℝ) : beautiful_two_var_eqn (1 / 3) m (-3) (2 * -3) → m = -7 := 
by 
  intro h
  simp [beautiful_two_var_eqn] at h
  sorry

/-- Prove there exists n such that the beautiful values of 5/2x + y = n and 4x - y = n - 2 are the same, and find n and the beautiful value -/
theorem beautiful_value_part3 (n : ℝ) : 
  beautiful_two_var_eqn (5 / 2) n ((9 * n) / 2) (2 * ((9 * n) / 2)) →
  beautiful_two_var_eqn 4 (n - 2) ((n - 2) / 2) (2 * ((n - 2) / 2)) →
  (n = 18 / 5 ∧ (2 * n / 9) = 4 / 5) := 
by 
  intros h1 h2
  simp [beautiful_two_var_eqn] at *
  sorry

end beautiful_value_part1_beautiful_value_part2_beautiful_value_part3_l735_735066


namespace largest_product_from_set_l735_735485

def largest_product (s : Set ℤ) : ℤ :=
  let products := { x * y | x ∈ s, y ∈ s, x ≠ y }
  Finset.sup products id

theorem largest_product_from_set :
  largest_product {-6, -4, -2, 1, 5} = 24 :=
by
  sorry

end largest_product_from_set_l735_735485


namespace probability_reach_origin_from_3_3_l735_735027

noncomputable def P : ℕ → ℕ → ℚ
| 0, 0 => 1
| x+1, 0 => 0
| 0, y+1 => 0
| x+1, y+1 => (1/3) * P x (y+1) + (1/3) * P (x+1) y + (1/3) * P x y

theorem probability_reach_origin_from_3_3 : P 3 3 = 1 / 27 := by
  sorry

end probability_reach_origin_from_3_3_l735_735027


namespace sum_is_multiple_of_6_12_24_not_48_l735_735449

theorem sum_is_multiple_of_6_12_24_not_48 :
  let y := 72 + 144 + 216 + 288 + 576 + 720 + 4608 in
  (6 ∣ y) ∧ (12 ∣ y) ∧ (24 ∣ y) ∧ ¬ (48 ∣ y) :=
by
  let y := 72 + 144 + 216 + 288 + 576 + 720 + 4608
  sorry

end sum_is_multiple_of_6_12_24_not_48_l735_735449


namespace area_triangle_NMC_l735_735067

theorem area_triangle_NMC (x y : ℝ) (h : 1 - x - y = real.sqrt (x^2 + y^2)) :
  1 / 2 * (1 - x) * (1 - y) = 1 / 4 :=
by sorry

end area_triangle_NMC_l735_735067


namespace curves_intersection_count_l735_735668

noncomputable def intersecting_curves_count : ℕ := 180

theorem curves_intersection_count :
  ∃ (A B C D : ℕ), 
    A ∈ {1, 2, 3, 4, 5, 6} ∧ B ∈ {1, 2, 3, 4, 5, 6} ∧ C ∈ {1, 2, 3, 4, 5, 6} ∧ D ∈ {1, 2, 3, 4, 5, 6} ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    ((A = C ∧ B ≠ D) ∨ (A ≠ C)) ∧ 
  intersecting_curves_count = 180 :=
sorry

end curves_intersection_count_l735_735668


namespace surface_area_of_cube_l735_735346

theorem surface_area_of_cube (sum_edges : ℝ) (h : sum_edges = 72) : 
  ∃ (area : ℝ), area = 216 :=
by
  -- let l be the length of one edge of the cube
  let l := sum_edges / 12 in
  -- we write the correct surface area computation formally
  let area := 6 * (l * l) in
  use area
  have : l = 6 := by
    sorry -- Proof of edge length from sum_edges; skip actual proof
  rw this
  show 6 * (6 * 6) = 216
  norm_num

end surface_area_of_cube_l735_735346


namespace find_custom_operation_value_l735_735279

noncomputable def custom_operation (a b : ℤ) : ℚ := (1 : ℚ)/a + (1 : ℚ)/b

theorem find_custom_operation_value (a b : ℤ) (h1 : a + b = 12) (h2 : a * b = 32) :
  custom_operation a b = 3 / 8 := by
  sorry

end find_custom_operation_value_l735_735279


namespace car_travel_distance_l735_735396

def travel_distance (n : ℕ) : ℤ :=
  35 - (n - 1) * 9

def total_travel_distance : ℤ :=
  35 + (∑ i in Finset.range 4, travel_distance (i + 1))

theorem car_travel_distance : total_travel_distance = 121 := by
  unfold total_travel_distance travel_distance
  calc
    35 + (35 + 26 + 17 + 8) = 35 + 86 := by simp
    35 + 86 = 121 := by simp

end car_travel_distance_l735_735396


namespace mutually_exclusive_event_l735_735770

def shooting_twice : Type := 
  { hit_first : Bool // hit_first = true ∨ hit_first = false }

def hitting_at_least_once (shoots : shooting_twice) : Prop :=
  shoots.1 ∨ (¬shoots.1 ∧ true)

def missing_both_times (shoots : shooting_twice) : Prop :=
  ¬shoots.1 ∧ (¬true ∨ true)

def mutually_exclusive (A : Prop) (B : Prop) : Prop :=
  A ∨ B → ¬ (A ∧ B)

theorem mutually_exclusive_event :
  ∀ shoots : shooting_twice, 
  mutually_exclusive (hitting_at_least_once shoots) (missing_both_times shoots) :=
by
  intro shoots
  unfold mutually_exclusive
  sorry

end mutually_exclusive_event_l735_735770


namespace population_hypothetical_town_l735_735339

theorem population_hypothetical_town :
  ∃ (a b c : ℕ), a^2 + 150 = b^2 + 1 ∧ b^2 + 1 + 150 = c^2 ∧ a^2 = 5476 :=
by {
  sorry
}

end population_hypothetical_town_l735_735339


namespace f_denominator_not_multiple_of_prime_l735_735440

theorem f_denominator_not_multiple_of_prime {f : ℕ × ℕ → ℚ}
  (h1 : ∀ i > 1, f (1, i) = 1 / i)
  (h2 : ∀ n i, i > n + 1 → f (n + 1, i) = (n + 1)/i * ∑ k in finset.Ico n (i - 1), f (n, k))
  (p : ℕ) (hp : nat.prime p) (n : ℕ) (hn : n > 1) :
  ¬ p ∣ rat.denom (f (n, p)) := by
  sorry

end f_denominator_not_multiple_of_prime_l735_735440


namespace five_letter_arrangements_l735_735212

theorem five_letter_arrangements : 
  let letters := ['A', 'B', 'C', 'D', 'E', 'F', 'G']
  let first_letter := 'D'
  let second_letter := 'G'
  (∀ l, List.length l = 5 ∧ 
       List.head l = first_letter ∧ 
       List.head (List.tail l) = second_letter ∧ 
       'B' ∈ List.drop 2 l →
       List.allDifferent l → 
       List.permutations letters) :=
  36 :=
sorry

end five_letter_arrangements_l735_735212


namespace boat_mass_and_pressure_l735_735753

/-- Given the conditions of the boat, water displacement, and additional weight, this theorem
    proves the mass of the man and the pressure he exerts on the boat. -/
theorem boat_mass_and_pressure (L B h : ℝ) (weight_supplies : ℝ) (rho g : ℝ)
  (H_L : L = 6) (H_B : B = 3) (H_h : h = 0.01) (H_weight_supplies : weight_supplies = 15)
  (H_rho : rho = 1000) (H_g : g = 9.81) :
  let V := L * B * h,
      m_water := V * rho,
      m_man := m_water - weight_supplies,
      A := L * B,
      F_man := m_man * g,
      P := F_man / A
  in m_man = 165 ∧ P ≈ 89.925 :=
by
  simp [H_L, H_B, H_h, H_weight_supplies, H_rho, H_g]
  sorry

end boat_mass_and_pressure_l735_735753


namespace altitude_of_equilateral_triangle_correct_l735_735441

noncomputable def altitude_of_equilateral_triangle (area : ℝ) := 
  let s := real.sqrt (640 * real.sqrt 3) in  -- side length of the triangle
  (s * real.sqrt 3) / 2  -- altitude of the triangle

theorem altitude_of_equilateral_triangle_correct : altitude_of_equilateral_triangle 480 = 40 * real.sqrt 3 :=
 by
  sorry

end altitude_of_equilateral_triangle_correct_l735_735441


namespace cheese_fries_cost_l735_735249

def jim_money : ℝ := 20
def cousin_money : ℝ := 10
def combined_money : ℝ := jim_money + cousin_money
def expenditure : ℝ := 0.80 * combined_money
def cheeseburger_cost : ℝ := 3
def milkshake_cost : ℝ := 5
def cheeseburgers_cost : ℝ := 2 * cheeseburger_cost
def milkshakes_cost : ℝ := 2 * milkshake_cost
def meal_cost : ℝ := cheeseburgers_cost + milkshakes_cost

theorem cheese_fries_cost :
  let cheese_fries_cost := expenditure - meal_cost 
  cheese_fries_cost = 8 := 
by
  sorry

end cheese_fries_cost_l735_735249


namespace floor_sum_inequality_l735_735746

theorem floor_sum_inequality (x : ℝ) (n : ℕ) (hx : 0 < x) (hn : 0 < n) :
  ⌊n * x⌋ ≥ ∑ k in Finset.range (n + 1) \ {0}, ⌊k * x⌋ / k :=
by
  sorry

end floor_sum_inequality_l735_735746


namespace area_ratio_of_S2_to_S1_l735_735253

-- Definition of the sets S1 and S2
def S1 : Set (ℝ × ℝ) := 
  { p | log 10 (3 + p.1 * p.1 + p.2 * p.2) ≤ 1 + log 10 (p.1 + p.2) }

def S2 : Set (ℝ × ℝ) := 
  { p | log 10 (5 + p.1 * p.1 + p.2 * p.2) ≤ 2 + log 10 (p.1 + p.2) }

-- The statement of the theorem proving the ratio of the areas
theorem area_ratio_of_S2_to_S1 : 
  (MeasureTheory.Measure.pi * 4995) / (MeasureTheory.Measure.pi * 47) = 4995 / 47 :=
  sorry

end area_ratio_of_S2_to_S1_l735_735253


namespace length_of_common_chord_eq_l735_735115

noncomputable def length_common_chord (A B B1 C C1 : Point) (α a : ℝ) : ℝ :=
  if |AB - AC| = a ∧ |AB1 - AC1| = a ∧ α > 0 then 
    a / (2 * (Real.sin (α / 2)))
  else
    0

theorem length_of_common_chord_eq (A B B1 C C1 : Point) (α a : ℝ) 
  (h1 : |AB - AC| = a) (h2 : |AB1 - AC1| = a) (hα : α > 0) : 
  length_common_chord A B B1 C C1 α a = a / (2 * (Real.sin (α / 2))) :=
  sorry

end length_of_common_chord_eq_l735_735115


namespace trig_identity_l735_735528

-- Define the given conditions
def condition (x : ℝ) : Prop := cos (π / 6 - x) = - (sqrt 3) / 3

-- State the theorem to prove
theorem trig_identity (x : ℝ) (h : condition x) : 
  cos (5 * π / 6 + x) + sin (2 * π / 3 - x) = 0 := 
by
  sorry

end trig_identity_l735_735528


namespace max_cos_arg_difference_correct_l735_735176

noncomputable def max_cos_arg_difference (z ω : ℂ) : ℝ :=
  if (z + ω + 3 = 0 ∧ |z| + |ω| = 4) then (1 / 8) else 0

theorem max_cos_arg_difference_correct (z ω : ℂ):
  (z + ω + 3 = 0) → (|z| + |ω| = 4) → 
  (cos (arg z - arg ω)) ≤ (max_cos_arg_difference z ω) :=
by
  intros hz ho
  rw [max_cos_arg_difference]
  simp [hz, ho]
  sorry

end max_cos_arg_difference_correct_l735_735176


namespace number_of_correct_propositions_l735_735002

-- Definitions based on the conditions
def prop1 (P Q : Plane) (l : Line) : Prop := Parallel P l ∧ Parallel Q l → Parallel P Q
def prop2 (P Q R : Plane) : Prop := Parallel P R ∧ Parallel Q R → Parallel P Q
def prop3 (a b c : Line) : Prop := Perpendicular a c ∧ Perpendicular b c → Parallel a b
def prop4 (a b : Line) (R : Plane) : Prop := Perpendicular a R ∧ Perpendicular b R → Parallel a b

-- The proof problem
theorem number_of_correct_propositions :
  (¬ (∀ (P Q : Plane) (l : Line), prop1 P Q l)) ∧
  (∀ (P Q R : Plane), prop2 P Q R) ∧
  (¬ (∀ (a b c : Line), prop3 a b c)) ∧
  (∀ (a b : Line) (R : Plane), prop4 a b R) →
  2 = 2 :=
sorry

end number_of_correct_propositions_l735_735002


namespace range_of_x0_l735_735958

theorem range_of_x0 (n : ℕ) (a b : ℝ) (x : ℕ → ℝ) (hₙ : n > 0)
  (h_sum : ∑ i in Finset.range (n+1), x i = a)
  (h_sum_sq : ∑ i in Finset.range (n+1), (x i)^2 = b) :
  let D := Real.sqrt (n * ((n+1)*b - a^2)) in 
  (a - D) / (n+1) ≤ x 0 ∧ x 0 ≤ (a + D) / (n+1) :=
by
  let D := Real.sqrt (n * ((n+1)*b - a^2))
  have h1 : (a - D) / (n + 1) ≤ x 0, sorry
  have h2 : x 0 ≤ (a + D) / (n + 1), sorry
  exact ⟨h1, h2⟩

end range_of_x0_l735_735958


namespace valid_boxes_count_l735_735927

noncomputable def num_valid_boxes (total_marbles : ℕ) (divisor_limit : ℕ) : ℕ :=
  let divisors := (List.range (total_marbles + 1)).filter (λ d, total_marbles % d = 0)
  let valid_divisors := divisors.filter (λ d, d > (total_marbles / 360) ∧ d ∣ divisor_limit ∧ d > 2)
  valid_divisors.length

theorem valid_boxes_count : num_valid_boxes 720 120 = 13 :=
by
  sorry

end valid_boxes_count_l735_735927


namespace diana_paint_statues_l735_735733

theorem diana_paint_statues (remaining_paint : ℚ) (paint_per_statue : ℚ)
  (h1 : remaining_paint = 7 / 8) (h2 : paint_per_statue = 1 / 8) : 
  (remaining_paint / paint_per_statue = 7) :=
by
  -- Import of Lean 4 libraries
  -- skipping the actual proof
  sorry

end diana_paint_statues_l735_735733


namespace sections_capacity_l735_735373

theorem sections_capacity (total_people sections : ℕ) 
  (h1 : total_people = 984) 
  (h2 : sections = 4) : 
  total_people / sections = 246 := 
by
  sorry

end sections_capacity_l735_735373


namespace fourth_term_eq_156_l735_735462

-- Definition of the sequence term
def seq_term (n : ℕ) : ℕ :=
  (List.range n).map (λ k => 5^k) |>.sum

-- Theorem to prove the fourth term equals 156
theorem fourth_term_eq_156 : seq_term 4 = 156 :=
sorry

end fourth_term_eq_156_l735_735462


namespace logarithm_problem_l735_735568

section LogarithmProblem

variables (x : ℝ)

-- Define the given condition
def condition : Prop := log 27 (x - 3) = 1 / 3

-- Define the target proof statement
def target : Prop := log 343 x = (1 / 3) * log 7 6

-- State the theorem
theorem logarithm_problem (h : condition x) : target x :=
by sorry

end LogarithmProblem

end logarithm_problem_l735_735568


namespace remainder_division_l735_735650

theorem remainder_division : ∃ (r : ℕ), 271 = 30 * 9 + r ∧ r = 1 :=
by
  -- Details of the proof would be filled here
  sorry

end remainder_division_l735_735650


namespace find_c_l735_735833

theorem find_c (x c : ℝ) (h : ((5 * x + 38 + c) / 5) = (x + 4) + 5) : c = 7 :=
by
  sorry

end find_c_l735_735833


namespace find_radius_l735_735042

noncomputable def radius_equilateral_inscribed_circle (r : ℝ) : Prop :=
  let s := r * real.sqrt 3 in
  let perimeter_triangle := 3 * s in
  let side_square := r * real.sqrt 2 in
  let area_square := side_square ^ 2 in
  perimeter_triangle = area_square → r = (3 * real.sqrt 3 / 4)

theorem find_radius (r : ℝ) (h : radius_equilateral_inscribed_circle r) : 
  r = (3 * real.sqrt 3 / 4) :=
by {
  exact h
}

end find_radius_l735_735042


namespace tanya_traveled_distance_l735_735468

theorem tanya_traveled_distance (N : ℤ) (T : ℤ) (D : ℤ) (x : ℤ)
  (h1 : D = 6)
  (h2 : x = N - D)
  (h3 : x + (N - T) = 1.5 * (D - T)) :
  N = 7 := 
sorry

end tanya_traveled_distance_l735_735468


namespace vector_dot_product_l735_735871

-- Define the vectors and the conditions
variables (a b : EuclideanSpace ℝ) -- assume 2D or 3D space for vectors
variables (angle_ab : Real) (norm_a : Real) (norm_b : Real)

-- Given conditions
def condition1 : angle_ab = π / 3 := by sorry -- 60 degrees in radians
def condition2 : norm a = 1 := by sorry
def condition3 : norm b = 2 := by sorry

-- Prove the goal
theorem vector_dot_product :
  b ⬝ (2 • a + b) = 6 :=
by 
  have h1 : a ⬝ b = 1 := by sorry
  have h2 : b ⬝ b = 4 := by sorry
  calc
    b ⬝ (2 • a + b) = 2 * (a ⬝ b) + (b ⬝ b) : sorry
                  ... = 2 * 1 + 4           : sorry
                  ... = 6                   : sorry

end vector_dot_product_l735_735871


namespace fraction_of_dark_tiles_half_l735_735009

-- Define a uniformly tiled floor such that each corner resembles the other
def uniformly_tiled (floor : Set (ℕ × ℕ)) : Prop :=
  ∀ (x y : ℕ), floor (x, y) ↔ (x % 2 = y % 2)

-- Prove that the fraction of the entire floor covered by dark tiles is 1/2
theorem fraction_of_dark_tiles_half (floor : Set (ℕ × ℕ)) 
  (h : uniformly_tiled floor) : 
  (∃ (dark_tiles : Set (ℕ × ℕ)), 
  ∀ (x y : ℕ), dark_tiles (x, y) ↔ (x % 2 = 1 ∧ y % 2 = 1) ∨ (x % 2 = 0 ∧ y % 2 = 0) 
  ∧ (dark_tiles.count / ((4 : ℕ) * 4) : ℚ) = 1 / 2) := 
sorry

end fraction_of_dark_tiles_half_l735_735009


namespace profit_15_percent_on_fruits_l735_735424

def selling_price_mango := 14.0
def loss_percentage_mango := 0.15
def target_profit_percentage := 0.15

def selling_price_apple := 20.0
def loss_percentage_apple := 0.10

def selling_price_orange := 30.0
def profit_percentage_orange := 0.05

noncomputable def cost_price (SP : ℝ) (loss_or_profit_percentage : ℝ) (is_loss : Bool) :=
  if is_loss then SP / (1 - loss_or_profit_percentage) else SP / (1 + loss_or_profit_percentage)

noncomputable def new_selling_price (CP : ℝ) (profit_percentage : ℝ) :=
  CP * (1 + profit_percentage)

theorem profit_15_percent_on_fruits :
  let CP_mango := cost_price selling_price_mango loss_percentage_mango true in
  let CP_apple := cost_price selling_price_apple loss_percentage_apple true in
  let CP_orange := cost_price selling_price_orange profit_percentage_orange false in
  new_selling_price CP_mango target_profit_percentage ≈ 18.94 ∧
  new_selling_price CP_apple target_profit_percentage ≈ 25.55 ∧
  new_selling_price CP_orange target_profit_percentage ≈ 32.86 :=
by
  sorry

end profit_15_percent_on_fruits_l735_735424


namespace smallest_n_for_y_n_integer_l735_735058

noncomputable def y (n : ℕ) : ℝ :=
  if n = 0 then 0 else
  if n = 1 then (5 : ℝ)^(1/3) else
  if n = 2 then ((5 : ℝ)^(1/3))^((5 : ℝ)^(1/3)) else
  y (n-1)^((5 : ℝ)^(1/3))

theorem smallest_n_for_y_n_integer : ∃ n : ℕ, y n = 5 ∧ ∀ m < n, y m ≠ ((⌊y m⌋:ℝ)) :=
by
  sorry

end smallest_n_for_y_n_integer_l735_735058


namespace simplify_sqrt_fraction_l735_735664

-- Definition of the problem
theorem simplify_sqrt_fraction : 
  (sqrt 450 / sqrt 200 + sqrt 392 / sqrt 98) = (7 / 2) := 
by
  sorry

end simplify_sqrt_fraction_l735_735664


namespace solution_set_of_3x2_minus_7x_gt_6_l735_735071

theorem solution_set_of_3x2_minus_7x_gt_6 (x : ℝ) :
  3 * x^2 - 7 * x > 6 ↔ (x < -2 / 3 ∨ x > 3) := 
by
  sorry

end solution_set_of_3x2_minus_7x_gt_6_l735_735071


namespace waiting_room_people_l735_735370

-- Define the conditions
def interview_room_people : ℕ := 5
def waiting_room_people_after_arrival (P : ℕ) : ℕ := P + 3
def interview_to_waiting_ratio (P : ℕ) : ℕ := 5 * interview_room_people

-- Statement of the proof problem
theorem waiting_room_people :
  ∃ P : ℕ, waiting_room_people_after_arrival(P) = interview_to_waiting_ratio(P) ∧ P = 22 :=
by
  sorry

end waiting_room_people_l735_735370


namespace cards_least_likely_red_after_flips_l735_735349

theorem cards_least_likely_red_after_flips :
  ∃ (k1 k2 : ℕ), 1 ≤ k1 ∧ k1 ≤ 50 ∧ 1 ≤ k2 ∧ k2 ≤ 50 ∧ (k1 = 13 ∧ k2 = 38) ∧ 
  (∀ k ∈ finset.range 1 51, 
    let p := (if k ≤ 25 then ((26 - k) ^ 2 + k ^ 2) / 676 else ((26 - (51 - k)) ^ 2 + (51 - k) ^ 2) / 676) in
    p ≥ (if k = 13 ∨ k = 38 then ((26 - k) ^ 2 + k ^ 2) / 676 else p)) :=
sorry

end cards_least_likely_red_after_flips_l735_735349


namespace mrs_hilt_hot_dogs_l735_735985

theorem mrs_hilt_hot_dogs (total_cost cost_per_hot_dog : ℕ) (h1 : total_cost = 300) (h2 : cost_per_hot_dog = 50) :
  total_cost / cost_per_hot_dog = 6 :=
by {
  rw [h1, h2],
  norm_num,
  sorry,
}

end mrs_hilt_hot_dogs_l735_735985


namespace existence_of_n1_existence_of_n2_l735_735769

-- Define the geometric series term
def geom_series (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, 1.06^i

-- Define the dividend condition (part 1)
def dividend_condition1 (n : ℕ) : Prop :=
  0.24 * geom_series n ≥ 1

-- Define the dividend condition (part 2)
def dividend_condition2 (n : ℕ) : Prop :=
  0.24 * geom_series n ≥ 1.06^n

-- The theorem statements to be proven
theorem existence_of_n1 : ∃ n : ℕ, dividend_condition1 n :=
by sorry

theorem existence_of_n2 : ∃ n : ℕ, dividend_condition2 n :=
by sorry

end existence_of_n1_existence_of_n2_l735_735769


namespace PepaIsLying_l735_735112

-- Definitions of friends and directions
inductive Direction
| North
| South
| East
| West

structure FriendDirection where
  name : Type
  direction : Direction

def Karel : FriendDirection := { name := "Karel", direction := sorry }
def Mojmir : FriendDirection := { name := "Mojmir", direction := Direction.South }
def Pepa : FriendDirection := { name := "Pepa", direction := Direction.North }
def Zdenda : FriendDirection := { name := "Zdenda", direction := sorry }

axiom StatementFalse (f: FriendDirection) : Bool

-- Conditions based on the problem
axiom KarelCondition : (Karel.direction ≠ Direction.North ∧ Karel.direction ≠ Direction.South)
axiom MojmirCondition : (Mojmir.direction = Direction.South)
axiom PepaCondition : (Pepa.direction = Direction.North)
axiom ZdendaCondition : (Zdenda.direction ≠ Direction.South)

-- There is one false statement among the friends
axiom OneFalseStatement : ∃ f : FriendDirection, StatementFalse f

-- The main theorem we want to prove
theorem PepaIsLying : StatementFalse Pepa := by
  sorry

end PepaIsLying_l735_735112


namespace problem_statement_l735_735624

noncomputable def roots1 (p : ℝ) : set ℝ := {α | ∃ β, α + β = -p ∧ α * β = 2 }
noncomputable def roots2 (q : ℝ) : set ℝ := {γ | ∃ δ, γ + δ = -q ∧ γ * δ = -3 }

theorem problem_statement (p q α β γ δ : ℝ) 
  (h1 : α ∈ roots1 p)
  (h2 : β ∈ roots1 p)
  (h3 : γ ∈ roots2 q)
  (h4 : δ ∈ roots2 q) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = 3 * (q^2 - p^2) + 15 :=
by
  sorry

end problem_statement_l735_735624


namespace number_of_paths_l735_735563

theorem number_of_paths (m n : ℕ) : 
  (m + n).choose m = nat.factorial (m + n) / (nat.factorial m * nat.factorial n) := by
  sorry

end number_of_paths_l735_735563


namespace proof_problem_l735_735109

-- Define the custom operations as given in conditions.
def custom_minus (x y : ℝ) : ℝ := x * y - x / 2

def custom_oplus (x y : ℝ) : ℝ := x + y / 2

-- Prove the required statements using these definitions.
theorem proof_problem :
  custom_minus 3.6 2 = 5.4 ∧ 0.12 - custom_oplus 7.5 4.8 = -9.78 :=
by
  sorry

end proof_problem_l735_735109


namespace range_of_m_l735_735330

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * Real.log x - m * x^2

def has_two_extreme_points (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f m x₁ = f m x₂ ∧ (∀ x, x = x₁ ∨ x = x₂ ∨ f m x ≤ f m x₁ ∨ f m x ≤ f m x₂)

theorem range_of_m :
  ∀ m : ℝ, has_two_extreme_points (m) ↔ 0 < m ∧ m < 1 / 2 := 
by
  sorry

end range_of_m_l735_735330


namespace trapezoid_upper_side_length_l735_735432

theorem trapezoid_upper_side_length 
  (base1 : ℝ) (height : ℝ) (area : ℝ) (base2 : ℝ) 
  (h_base1 : base1 = 25)
  (h_height : height = 13) 
  (h_area : area = 286) 
  (h_formula : area = 0.5 * (base1 + base2) * height) : 
  base2 = 19 :=
by {
  rw [h_base1, h_height, h_area] at h_formula,
  simp at h_formula,
  linarith,
  rw [h_base1, h_height],
  simp,
  linarith,
}

end trapezoid_upper_side_length_l735_735432


namespace max_intersection_l735_735834

open Finset

def n (S : Finset α) : ℕ := (2 : ℕ) ^ S.card

theorem max_intersection (A B C : Finset ℕ)
  (h1 : A.card = 2016)
  (h2 : B.card = 2016)
  (h3 : n A + n B + n C = n (A ∪ B ∪ C)) :
  (A ∩ B ∩ C).card ≤ 2015 :=
sorry

end max_intersection_l735_735834


namespace derivative_at_x0_l735_735626

open Function Filter

variable {α : Type*} {f : α → ℝ} {x_0 : α}

theorem derivative_at_x0 
  (h_diff : DifferentiableAt ℝ f x_0)
  (h_limit : tendsto (λ Δx : ℝ, (f (x_0 - 3 * Δx) - f x_0) / Δx) (𝓝 0) (𝓝 1)) :
  deriv f x_0 = -1/3 :=
by
  sorry

end derivative_at_x0_l735_735626


namespace max_k_value_l735_735876

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_k_value :
  (∀ x : ℝ, 0 < x → (∃ k : ℝ, k * x = Real.log x ∧ k ≤ f x)) ∧
  (∀ x : ℝ, 0 < x → f x ≤ 1 / Real.exp 1) ∧
  (∀ x : ℝ, 0 < x → (k = f x → k ≤ 1 / Real.exp 1)) := 
sorry

end max_k_value_l735_735876


namespace time_to_clear_proof_l735_735407

noncomputable def train1_length : ℝ := 111 -- meters
noncomputable def train2_length : ℝ := 165 -- meters
noncomputable def train1_speed_kmh : ℝ := 60 -- km/h
noncomputable def train2_speed_kmh : ℝ := 90 -- km/h

-- Conversion factor from km/h to m/s
noncomputable def kmh_to_ms (v : ℝ) : ℝ :=
  v * (1000 / 3600)

noncomputable def train1_speed_ms : ℝ := kmh_to_ms train1_speed_kmh
noncomputable def train2_speed_ms : ℝ := kmh_to_ms train2_speed_kmh

noncomputable def total_length : ℝ := train1_length + train2_length
noncomputable def relative_speed : ℝ := train1_speed_ms + train2_speed_ms

noncomputable def time_to_clear : ℝ := total_length / relative_speed

theorem time_to_clear_proof : time_to_clear ≈ 6.62 := 
by
  sorry

end time_to_clear_proof_l735_735407


namespace quartic_real_roots_approx_l735_735820

noncomputable def N : ℝ := 10^10

theorem quartic_real_roots_approx :
  ∀ (x : ℝ), x = 99999.9984 ∨ x = 100000.0016 →
  x^4 - (2 * N + 1) * x^2 - x + N^2 + N - 1 = 0 :=
by
  intros x h
  cases h
  { rw h, sorry }
  { rw h, sorry }

end quartic_real_roots_approx_l735_735820


namespace dalton_needs_more_money_l735_735802

theorem dalton_needs_more_money
  (cost_jump_rope : ℕ)
  (cost_board_game : ℕ)
  (cost_playground_ball : ℕ)
  (money_saved : ℕ)
  (money_from_uncle : ℕ) :
  cost_jump_rope = 7 →
  cost_board_game = 12 →
  cost_playground_ball = 4 →
  money_saved = 6 →
  money_from_uncle = 13 →
  cost_jump_rope + cost_board_game + cost_playground_ball - (money_saved + money_from_uncle) = 4 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end dalton_needs_more_money_l735_735802


namespace minimum_distance_sum_l735_735184

noncomputable theory

def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 2 * p.1
def distance (a b : ℝ × ℝ) : ℝ := real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

def P := (x: ℝ) -> (y: ℝ) -> parabola (x, y)

def focus : ℝ × ℝ := (1 / 2, 0)
def point_A : ℝ × ℝ := (0, 2)
def directrix := λ x : ℝ, x = -(1/2)

theorem minimum_distance_sum :
  ∀ (x y : ℝ), parabola (x, y) → 
  (distance (x, y) focus + distance (x, y) point_A) ≥ distance focus point_A :=
by
  intros x y h
  sorry

end minimum_distance_sum_l735_735184


namespace sum_over_eighth_roots_of_unity_eq_10_plus_4_sqrt_2_l735_735393

noncomputable def sum_over_roots_of_unity : ℂ :=
  ∑ (k : ℕ) in finset.range 8, 1 / (abs (1 - complex.exp (2 * real.pi * complex.I * k / 8)))^2

theorem sum_over_eighth_roots_of_unity_eq_10_plus_4_sqrt_2 :
  sum_over_roots_of_unity = 10 + 4 * real.sqrt 2 :=
sorry

end sum_over_eighth_roots_of_unity_eq_10_plus_4_sqrt_2_l735_735393


namespace discriminant_of_P_l735_735154

theorem discriminant_of_P (a b c : ℚ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1/2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1/2 := 
begin
  -- Proof omitted for brevity
  sorry
end

end discriminant_of_P_l735_735154


namespace sqrt_sum_le_sqrt3_l735_735845

theorem sqrt_sum_le_sqrt3 (a b c : ℝ) (h : a + b + c = 1) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) : 
  sqrt a + sqrt b + sqrt c ≤ sqrt 3 := 
sorry

end sqrt_sum_le_sqrt3_l735_735845


namespace median_of_special_list_l735_735600

theorem median_of_special_list : ∀ n, (1 ≤ n ∧ n ≤ 200 → median_of (multiset.bind (multiset.range n) (λ k, k::k::multiset.repeat k (k-1))) = 141) := by
  intro n hn
  sorry

end median_of_special_list_l735_735600


namespace lowest_probability_red_side_up_l735_735361

def card_flip_probability (k : ℕ) (n : ℕ) : ℚ :=
  if k ≤ n/2 then (n-k)^2/(n^2) + k^2/(n^2)
  else card_flip_probability (n+1-k) n 

theorem lowest_probability_red_side_up :
  (card_flip_probability 13 50) = (card_flip_probability 38 50) ∧ 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 25 → (card_flip_probability k 50 ≥ card_flip_probability 13 50)) :=
begin
  sorry
end

end lowest_probability_red_side_up_l735_735361


namespace problem_solution_l735_735889

noncomputable def f (x : ℝ) : ℝ := Real.exp (x + 1)

def a : ℝ := f (Real.log 0.2 / Real.log 2)
def b : ℝ := f (Real.exp (0.2 * Real.log 2))
def c : ℝ := f (Real.exp (0.3 * Real.log 0.2))

theorem problem_solution : a < c ∧ c < b := 
by {
  -- proof goes here
  sorry
}

end problem_solution_l735_735889


namespace vector_relationship_l735_735866

noncomputable def f (x : ℝ) : ℝ :=
  ln (x + 1)

theorem vector_relationship (x y f_prime_O : ℝ)
  (h : ∀ (A B C O : Type) (OA OB OC : O → ℝ) (l : set O),
      A ∈ l → B ∈ l → C ∈ l →
      (OA - (y + 2 * f_prime_O) • OB + ln (x + 1) • OC = 0)) :
  f x = ln (x + 1) :=
sorry

end vector_relationship_l735_735866


namespace inradius_length_l735_735266

noncomputable def inradius (BC AB AC IC : ℝ) (r : ℝ) : Prop :=
  ∀ (r : ℝ), ((BC = 40) ∧ (AB = AC) ∧ (IC = 24)) →
    r = 4 * Real.sqrt 11

theorem inradius_length (BC AB AC IC : ℝ) (r : ℝ) :
  (BC = 40) ∧ (AB = AC) ∧ (IC = 24) →
  r = 4 * Real.sqrt 11 := 
by
  sorry

end inradius_length_l735_735266


namespace sin_cos_18_36_l735_735073

theorem sin_cos_18_36 (deg18 : ℝ := real.pi * 18 / 180) (deg36 : ℝ := real.pi * 36 / 180) :
  real.sin(deg18) * real.cos(deg36) = 1 / 4 :=
by
  sorry

end sin_cos_18_36_l735_735073


namespace jill_water_filled_jars_l735_735735

variable (gallons : ℕ) (quart_halfGallon_gallon : ℕ)
variable (h_eq : gallons = 14)
variable (h_eq_n : quart_halfGallon_gallon = 3 * 8)
variable (h_total : quart_halfGallon_gallon = 24)

theorem jill_water_filled_jars :
  3 * (gallons * 4 / 7) = 24 :=
sorry

end jill_water_filled_jars_l735_735735


namespace prove_average_sales_volume_and_regression_correctness_l735_735206

-- Definitions based on the problem statement
def unit_prices : List ℕ := [4, 5, 6, 7, 8, 9]
def sales_volumes : List ℕ := [90, 84, 83, 80, 75, 68]
def regression_slope : ℤ := -4
def regression_intercept (prices : List ℕ) (volumes : List ℕ) : ℤ := 106 -- from solution

-- Problem statement in Lean format
theorem prove_average_sales_volume_and_regression_correctness :
  (List.sum sales_volumes / sales_volumes.length = 80) ∧
  (∃ a : ℤ, regression_intercept unit_prices sales_volumes = a ∧
    ∀ x: ℕ, x ≥ 0 → x ∈ unit_prices → (∃ y : ℤ, y = regression_slope * x + a ∧ y ∈ sales_volumes)) :=
by
  sorry

end prove_average_sales_volume_and_regression_correctness_l735_735206


namespace smallest_palindrome_divisible_by_4_is_1881_l735_735388

def is_palindrome (n : ℕ) : Prop :=
  let str := n.repr in
  str = str.reverse

def smallest_4_digit_palindrome_divisible_by_4 : ℕ :=
  1881

theorem smallest_palindrome_divisible_by_4_is_1881 (n : ℕ) :
  (1000 ≤ n ∧ n < 10000) ∧ is_palindrome n ∧ n % 4 = 0 → n = smallest_4_digit_palindrome_divisible_by_4 :=
begin
  sorry
end

end smallest_palindrome_divisible_by_4_is_1881_l735_735388


namespace find_discriminant_l735_735161

variables {a b c : ℝ}
variables (P : ℝ → ℝ)
def is_quadratic_polynomial (P : ℝ → ℝ) : Prop := ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, P x = a * x^2 + b * x + c)

theorem find_discriminant (h1 : is_quadratic_polynomial P)
  (h2 : ∃ x, P x = x - 2)
  (h3 : ∃ y, P y = 1 - y / 2)
  : ∃ D, D = -1/2 := 
sorry

end find_discriminant_l735_735161


namespace log_56_eq_a_mul_b_plus_3_l735_735529

theorem log_56_eq_a_mul_b_plus_3 (a b : ℝ) (log2_val : Real.log 2 = a) (log2_7_val : Real.logBase 2 7 = b) :
  Real.log 56 = a * (b + 3) :=
by
  sorry

end log_56_eq_a_mul_b_plus_3_l735_735529


namespace min_red_up_probability_card_l735_735354

theorem min_red_up_probability_card (cards : Fin 50) :
  (cards = 13) ∨ (cards = 38) ↔
  -- Conditions for Vasya and Asya's actions:
  ∃ (select_vasya : Fin 26 → Fin 50) (select_asya : Fin 26 → Fin 50),
    (∀ k : Fin 50, 
      probability.card_pos (select_vasya k.left ≤ k.to_nat ∧ k.to_nat < k.left + 25 → 
                            select_asya k.right ≤ k.to_nat ∧ k.to_nat < k.right + 25 → 
                            k.val = 13 ∨ k.val = 38))

end min_red_up_probability_card_l735_735354


namespace solve_problem_l735_735808

open Real

noncomputable def problem_statement : ℝ :=
  2 * log (sqrt 2) + (log 5 / log 2) * log 2

theorem solve_problem : problem_statement = 1 := by
  sorry

end solve_problem_l735_735808


namespace angle_XYZ_is_90_l735_735265

-- Variables representing points in an affine space
variables (X Y Z A D B E C F : ℝ)

-- Given conditions: Let AD, BE, CF be the altitudes of an acute triangle XYZ
-- and the vector equation involving these altitudes
axiom altitude_AD : ∃ k : ℝ, AD = k * A
axiom altitude_BE : ∃ l : ℝ, BE = l * B
axiom altitude_CF : ∃ m : ℝ, CF = m * C
axiom vector_sum : 5 * (D - A) + 3 * (E - B) + 8 * (F - C) = 0

-- We need to prove that the angle XYZ is 90 degrees
theorem angle_XYZ_is_90 (hAD : altitude_AD) (hBE : altitude_BE) (hCF : altitude_CF)
                (hVecSum : vector_sum) : 
                ∠XYZ = 90 := 
sorry

end angle_XYZ_is_90_l735_735265


namespace find_t_l735_735830

theorem find_t (t : ℝ) (h : (1 / (t+3) + 3 * t / (t+3) - 4 / (t+3)) = 5) : t = -9 :=
by
  sorry

end find_t_l735_735830


namespace quadratic_discriminant_l735_735169

variable {a b c : ℝ}
variable (h₁ : a ≠ 0)
variable (h₂ : (b - 1)^2 - 4 * a * (c + 2) = 0)
variable (h₃ : (b + 1/2)^2 - 4 * a * (c - 1) = 0)

theorem quadratic_discriminant : b^2 - 4 * a * c = -1 / 2 := 
by
  have h₁' : (b - 1)^2 - 4 * a * (c + 2) = 0 := h₂
  have h₂' : (b + 1/2)^2 - 4 * a * (c - 1) = 0 := h₃
  sorry

end quadratic_discriminant_l735_735169


namespace range_of_a_l735_735535

theorem range_of_a (x y z a : ℝ) 
    (h1 : x > 0) 
    (h2 : y > 0) 
    (h3 : z > 0) 
    (h4 : x + y + z = 1) 
    (h5 : a / (x * y * z) = 1 / x + 1 / y + 1 / z - 2) : 
    0 < a ∧ a ≤ 7 / 27 := 
  sorry

end range_of_a_l735_735535


namespace greatest_three_digit_divisible_by_3_5_6_l735_735387

theorem greatest_three_digit_divisible_by_3_5_6 : 
    ∃ n : ℕ, 
        (100 ≤ n ∧ n ≤ 999) ∧ 
        (∃ k₃ : ℕ, n = 3 * k₃) ∧ 
        (∃ k₅ : ℕ, n = 5 * k₅) ∧ 
        (∃ k₆ : ℕ, n = 6 * k₆) ∧ 
        (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999) ∧ (∃ k₃ : ℕ, m = 3 * k₃) ∧ (∃ k₅ : ℕ, m = 5 * k₅) ∧ (∃ k₆ : ℕ, m = 6 * k₆) → m ≤ 990) := by
  sorry

end greatest_three_digit_divisible_by_3_5_6_l735_735387


namespace correct_algebraic_expression_l735_735399

theorem correct_algebraic_expression
  (A : String := "1 1/2 a")
  (B : String := "a × b")
  (C : String := "a ÷ b")
  (D : String := "2a") :
  D = "2a" :=
by {
  -- Explanation based on the conditions provided
  -- A: "1 1/2 a" is not properly formatted. Correct format involves improper fraction for multiplication.
  -- B: "a × b" should avoid using the multiplication sign explicitly.
  -- C: "a ÷ b" should be written as a fraction a/b.
  -- D: "2a" is correctly formatted.
  sorry
}

end correct_algebraic_expression_l735_735399


namespace problem_statement_l735_735556

noncomputable def vector_m (ω x : ℝ) : ℝ × ℝ := (2 * sin (ω * x), 2 * cos (ω * x))
def vector_n : ℝ × ℝ := (1, 1)

noncomputable def norm (v : ℝ × ℝ) : ℝ := real.sqrt ((v.1)^2 + (v.2)^2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

noncomputable def projection (v1 v2 : ℝ × ℝ) : ℝ := dot_product v1 v2 / norm v2

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := projection (vector_m ω x) vector_n

theorem problem_statement (ω : ℝ) (h1 : ω > 0) (h2 : ω * (π / 4) + π / 4 = π / 2) 
  (θ : ℝ) (h3 : cos θ = 4 / 5) (h4 : θ ∈ set.Ioc (3 * π / 2) (2 * π)) :
  (f x ω = 2 * sin (x + π / 4)) ∧ (f (θ + π / 12) 1 = (4 * real.sqrt 3 - 3) / 5) :=
sorry

end problem_statement_l735_735556


namespace solution_set_of_inequality_system_l735_735345

theorem solution_set_of_inequality_system (x : ℝ) : (2 - x > 0) ∧ (2x + 3 > 1) → (-1 < x) ∧ (x < 2) :=
by
  sorry

end solution_set_of_inequality_system_l735_735345


namespace partial_fractions_sum_zero_l735_735310

noncomputable def sum_of_coefficients (A B C D E : ℝ) : Prop :=
  (A + B + C + D + E = 0)

theorem partial_fractions_sum_zero :
  ∀ (A B C D E : ℝ),
    (∀ x : ℝ, 1 = A*(x+1)*(x+2)*(x+3)*(x+5) + B*x*(x+2)*(x+3)*(x+5) + 
              C*x*(x+1)*(x+3)*(x+5) + D*x*(x+1)*(x+2)*(x+5) + 
              E*x*(x+1)*(x+2)*(x+3)) →
    sum_of_coefficients A B C D E :=
by sorry

end partial_fractions_sum_zero_l735_735310


namespace perfect_square_iff_n_eq_5_l735_735482

theorem perfect_square_iff_n_eq_5 (n : ℕ) (h_pos : 0 < n) :
  ∃ m : ℕ, n * 2^(n-1) + 1 = m^2 ↔ n = 5 := by
  sorry

end perfect_square_iff_n_eq_5_l735_735482


namespace polynomial_discriminant_l735_735129

theorem polynomial_discriminant (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
by
  sorry

end polynomial_discriminant_l735_735129


namespace triangle_similarity_l735_735255

-- Define the setup with the necessary points and conditions
variables {A B C A' B' C' A1 B1 C1 D E F : Point} 
(noncomputable def : Triangle := {A, B, C})
(noncomputable def : ExcircleTouchPoints := {A', B', C'})
(noncomputable def : ContactTriangle := {D, E, F})

-- Define the specific intersections with the circumcircle
axiom A1_property : is_intersection A1 (circumcircle ABC) (circumcircle A'B'C')
axiom B1_property : is_intersection B1 (circumcircle ABC) (circumcircle AB'C')
axiom C1_property : is_intersection C1 (circumcircle ABC) (circumcircle A'BC')

-- Main theorem statement
theorem triangle_similarity 
  (hA1 : A1 ≠ A) (hB1 : B1 ≠ B) (hC1 : C1 ≠ C) :
  similar (triangle A1 B1 C1) (triangle D E F) :=
sorry -- Proof will be provided later 

end triangle_similarity_l735_735255


namespace part1_part2_l735_735894

def f (x : ℝ) (q : ℝ) : ℝ := x^2 - 16 * x + q + 3

theorem part1 (q : ℝ) : (-20 ≤ q ∧ q ≤ 12) ↔ 
  ∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), (f x q ≠ 0) := sorry

theorem part2 : ∃ t : ℝ, t ≥ 0 ∧ (∀ x ∈ set.Icc t (10 : ℝ), 
  f x 0 ∈ set.Icc (10 : ℝ)(12-t)) ∧ (t = 8 ∨ t = 9) := sorry

end part1_part2_l735_735894


namespace symmetric_point_is_correct_l735_735597

/-- A point in 2D Cartesian coordinates -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Defining the point P with given coordinates -/
def P : Point := {x := 2, y := 3}

/-- Defining the symmetry of a point with respect to the origin -/
def symmetric_origin (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- States that the symmetric point of P (2, 3) with respect to the origin is (-2, -3) -/
theorem symmetric_point_is_correct :
  symmetric_origin P = {x := -2, y := -3} :=
by
  sorry

end symmetric_point_is_correct_l735_735597


namespace number_x_is_greater_than_9_l735_735583
noncomputable def certain_number (x : ℤ) : Prop :=
  5 < x ∧ x < 21 ∧
  7 < x ∧ x < 18 ∧
  x < 13 ∧
  9 < x ∧ x < 12 ∧
  x + 1 < 13

theorem number_x_is_greater_than_9 (x : ℤ) (h : certain_number x) : 9 < x :=
by {
  cases h with _ h1,
  cases h1 with _ h2,
  cases h2 with _ h3,
  cases h3 with _ h4,
  cases h4 with _ h5,
  exact h5.left,
}

end number_x_is_greater_than_9_l735_735583


namespace venus_speed_in_miles_per_hour_l735_735742

theorem venus_speed_in_miles_per_hour (v : ℝ) (h : v = 21.9) : 
  v * 3600 = 78840 := 
by 
  rw [h]
  norm_num
  sorry -- This is where the detailed proof would go

end venus_speed_in_miles_per_hour_l735_735742


namespace part_a_part_b_part_c_part_d_l735_735245

-- Definitions of natural numbers and the properties we need to prove
open Nat

-- Part (a): ∃ n ∈ ℕ, √[n]{1000} < 1.001.
theorem part_a : ∃ n : ℕ, Real.rootn 1000 n < 1.001 := sorry

-- Part (b): ∃ n ∈ ℕ, √[n]{n} < 1.001.
theorem part_b : ∃ n : ℕ, Real.rootn n n < 1.001 := sorry

-- Part (c): ∃ n ∈ ℕ, √(n + 1) - √n < 0.1.
theorem part_c : ∃ n : ℕ, Real.sqrt (n + 1) - Real.sqrt n < 0.1 := sorry

-- Part (d): ∀ n ∈ ℕ, √(n² + n) - n ≥ 0.1.
theorem part_d : ∀ n : ℕ, Real.sqrt (n^2 + n) - n ≥ 0.1 := sorry

end part_a_part_b_part_c_part_d_l735_735245


namespace value_of_stamp_collection_l735_735305

theorem value_of_stamp_collection :
  (∀ n : ℕ, n = 24 → ∃ v : ℝ, v = 2.5 → 8 * v = ∃ t : ℝ, t = 20) → 
  ∀ n : ℕ, n = 24 → (24 * 2.5 = 60) :=
begin
  sorry
end

end value_of_stamp_collection_l735_735305


namespace sequence_count_l735_735566

theorem sequence_count : 
  let sequences := { s : Fin 30 → bool // ∃ n, (∀ i < n, s i = false) ∨ (∀ i ≥ n, s i = true) } in
  sequences.card = 988 :=
  sorry

end sequence_count_l735_735566


namespace sum_of_x_coords_above_line_l735_735988

def lies_above_line (p : ℝ × ℝ) : Prop :=
  p.2 > 3 * p.1 + 5

def points : List (ℝ × ℝ) :=
  [ (4, 20), (7, 30), (15, 50), (20, 65), (25, 80) ]

def sum_x_coords_of_points_above_line : ℝ :=
  (points.filter lies_above_line).map Prod.fst

theorem sum_of_x_coords_above_line :
  (sum_x_coords_of_points_above_line points) = 11 :=
begin
  sorry
end

end sum_of_x_coords_above_line_l735_735988


namespace domain_of_f_l735_735326

-- Define the functions g and h
def g (x : ℝ) : ℝ :=
  1 / real.sqrt (2 - x)

def h (x : ℝ) : ℝ :=
  real.log (x + 1)

-- Define the function f as g + h
def f (x : ℝ) : ℝ :=
  g x + h x

-- Define the conditions for x
def condition1 (x : ℝ) : Prop :=
  2 - x > 0

def condition2 (x : ℝ) : Prop :=
  x + 1 > 0

-- Prove that the domain of f is (-1, 2)
theorem domain_of_f (x : ℝ) : (condition1 x) ∧ (condition2 x) ↔ -1 < x ∧ x < 2 :=
by
  -- omitted proof
  sorry

end domain_of_f_l735_735326


namespace right_triangle_tan_B_l735_735240

theorem right_triangle_tan_B (A B C : Type) [metric_space A] [inner_product_space ℝ A]
  (h_right : angle A B C = 90) (AB BC : ℝ)
  (h_AB : AB = 16) (h_BC : BC = 24) :
  tan B = 3 / 2 :=
sorry

end right_triangle_tan_B_l735_735240


namespace find_discriminant_l735_735162

variables {a b c : ℝ}
variables (P : ℝ → ℝ)
def is_quadratic_polynomial (P : ℝ → ℝ) : Prop := ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, P x = a * x^2 + b * x + c)

theorem find_discriminant (h1 : is_quadratic_polynomial P)
  (h2 : ∃ x, P x = x - 2)
  (h3 : ∃ y, P y = 1 - y / 2)
  : ∃ D, D = -1/2 := 
sorry

end find_discriminant_l735_735162


namespace R_m_is_fibonacci_l735_735256

-- Definition of the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fibonacci n + fibonacci (n + 1)

-- Definition of R_m
def R_m (m : ℕ) : ℕ :=
  if h : m > 2 then
    (∏ k in Finset.range (fibonacci m), k ^ k) % fibonacci m
  else
    0 -- R_m is not defined for m ≤ 2

-- Theorem to be proven
theorem R_m_is_fibonacci (m : ℕ) (h : m > 2) : ∃ n : ℕ, R_m m = fibonacci n :=
begin
  sorry
end

end R_m_is_fibonacci_l735_735256


namespace min_red_up_probability_card_l735_735355

theorem min_red_up_probability_card (cards : Fin 50) :
  (cards = 13) ∨ (cards = 38) ↔
  -- Conditions for Vasya and Asya's actions:
  ∃ (select_vasya : Fin 26 → Fin 50) (select_asya : Fin 26 → Fin 50),
    (∀ k : Fin 50, 
      probability.card_pos (select_vasya k.left ≤ k.to_nat ∧ k.to_nat < k.left + 25 → 
                            select_asya k.right ≤ k.to_nat ∧ k.to_nat < k.right + 25 → 
                            k.val = 13 ∨ k.val = 38))

end min_red_up_probability_card_l735_735355


namespace find_hyperbola_a_l735_735852

theorem find_hyperbola_a
  (p t : ℝ) (hp : p > 0) (ht : t > 0) (M_focus_dist : (M : (ℝ × ℝ)), M = (1, t) → ∃ focus: (ℝ × ℝ), dist M focus = 5) 
  (a : ℝ) (ha : a > 0) : 
  ∀ M, M = (1, (4:ℝ)) → (∃ A, A = (-a, 0)) → (∃ k, k = (4 / (1 + a))) → parallel (3 / a) (4 / (1 + a)) → a = 3 :=
by sorry

end find_hyperbola_a_l735_735852


namespace g_at_8_l735_735331

noncomputable def g : ℝ → ℝ := sorry

axiom g_property : ∀ x y : ℝ, g(x) + g(3*x + y) + 7*x*y = g(4*x - y) + 3*x^2 + 4

theorem g_at_8 : g(8) = 420 :=
by
  sorry

end g_at_8_l735_735331


namespace lines_intersect_at_point_l735_735269

theorem lines_intersect_at_point {A B C I P Q R S : Type*}
  [incircle : incenter A B C I]
  [circle_k : circle k A B]
  [line_AI : line AI A P]
  [line_BI : line BI B Q]
  [line_AC : line AC A R]
  [line_BC : line BC B S]
  (distinct_points : A ≠ B ∧ A ≠ P ∧ A ≠ Q ∧ A ≠ R ∧ A ≠ S ∧ B ≠ P ∧ B ≠ Q ∧ B ≠ R ∧ B ≠ S ∧ P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S)
  (R_in_AC : R ∈ segment AC)
  (S_in_BC : S ∈ segment BC) :
  (∃ point : Type*, collinear PS QR point ∧ collinear PS CI point ∧ collinear QR CI point) :=
sorry

end lines_intersect_at_point_l735_735269


namespace probability_of_triangle_segments_from_15gon_l735_735777

/-- A proof problem that calculates the probability that three randomly selected segments 
    from a regular 15-gon inscribed in a circle form a triangle with positive area. -/
theorem probability_of_triangle_segments_from_15gon : 
  let n := 15
  let total_segments := (n * (n - 1)) / 2 
  let total_combinations := total_segments * (total_segments - 1) * (total_segments - 2) / 6 
  let valid_probability := 943 / 1365
  valid_probability = (total_combinations - count_violating_combinations) / total_combinations :=
sorry

end probability_of_triangle_segments_from_15gon_l735_735777


namespace jeff_bought_6_pairs_l735_735247

theorem jeff_bought_6_pairs (price_of_shoes : ℝ) (num_of_shoes : ℕ) (price_of_jersey : ℝ)
  (h1 : price_of_jersey = (1 / 4) * price_of_shoes)
  (h2 : num_of_shoes * price_of_shoes = 480)
  (h3 : num_of_shoes * price_of_shoes + 4 * price_of_jersey = 560) :
  num_of_shoes = 6 :=
sorry

end jeff_bought_6_pairs_l735_735247


namespace probability_sum_even_l735_735832

theorem probability_sum_even :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37],
      num_selection := 5,
      num_primes := 12,
      num_cases := (primes.length.choose num_selection) in
  let num_even_cases := (primes.erase_chosen 2).length.choose (num_selection - 1) in
  (num_even_cases.fraction num_cases) = (7/12) :=
by sorry

end probability_sum_even_l735_735832


namespace quadratic_polynomial_discriminant_l735_735146

def P (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∃ x : ℝ, P a b c x = x - 2 ∧ (discriminant a (b - 1) (c + 2) = 0))
  (h₂ : ∃ x : ℝ, P a b c x = 1 - x / 2 ∧ (discriminant a (b + 1 / 2) (c - 1) = 0)) :
  discriminant a b c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l735_735146


namespace eccentricity_is_correct_l735_735674

noncomputable def eccentricity_of_hyperbola
  (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0)
  (intersects : ∃ A B : ℝ × ℝ,
    (A.1^2 / a^2 - A.2^2 / b^2 = 1 ∧ A.2^2 = 2 * p * A.1) ∧
    (B.1^2 / a^2 - B.2^2 / b^2 = 1 ∧ B.2^2 = 2 * p * B.1) ∧
    (B.1 = A.1) ∧ (B.2 = -A.2) ∧ (B.1 = A.1 = p / 2)) : ℝ :=
1 + real.sqrt 2

theorem eccentricity_is_correct
  (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0)
  (intersects : ∃ A B : ℝ × ℝ,
    (A.1^2 / a^2 - A.2^2 / b^2 = 1 ∧ A.2^2 = 2 * p * A.1) ∧
    (B.1^2 / a^2 - B.2^2 / b^2 = 1 ∧ B.2^2 = 2 * p * B.1) ∧
    (B.1 = A.1) ∧ (B.2 = -A.2) ∧ (B.1 = A.1 = p / 2)) :
  eccentricity_of_hyperbola a b p ha hb hp intersects = 1 + real.sqrt 2 :=
sorry

end eccentricity_is_correct_l735_735674


namespace rounding_to_thousandth_precision_of_scientific_notation_l735_735997

-- Definition and theorem for rounding to the nearest thousandth
theorem rounding_to_thousandth (x : ℝ) (h : x = 3.1415926) : round (x * 1000) / 1000 = 3.142 :=
by
  rw [h]
  norm_num
  -- Simplify the rounding directly leads to 3.142
  sorry

-- Definition and theorem for determining the precision of an approximate number in scientific notation
theorem precision_of_scientific_notation (a n : ℝ) (h : a * 10^n = 3.0 * 10^6) : 
  a = 3.0 ∧ n = 6 → (precision 3.0 = "ten-thousandths place") :=
by
  intros ha hn
  have h1 : a = 3.0 := ha
  have h2 : n = 6 := hn
  -- Considering the significant digits of 3.0 which are counted as one digit
  -- Leading to precision at ten-thousandths place
  sorry

end rounding_to_thousandth_precision_of_scientific_notation_l735_735997


namespace hotel_charge_problem_l735_735739

theorem hotel_charge_problem (R G P : ℝ) 
  (h1 : P = 0.5 * R) 
  (h2 : P = 0.9 * G) : 
  (R - G) / G * 100 = 80 :=
by
  sorry

end hotel_charge_problem_l735_735739


namespace tangent_sum_formula_l735_735524

noncomputable def theta : ℝ := sorry -- Placeholder for specific value of θ

theorem tangent_sum_formula :
  (θ ∈ Ioo (-π / 2) 0) →
  (cos θ = sqrt 17 / 17) →
  tan (θ + π / 4) = -3 / 5 :=
by
  intros h1 h2
  sorry

end tangent_sum_formula_l735_735524


namespace length_of_train_l735_735036

def speed_km_per_hr : ℝ := 72
def speed_m_per_s : ℝ := speed_km_per_hr * (1000 / 3600)

def time_s : ℝ := 11.099112071034318
def length_bridge_m : ℝ := 112

def total_distance_m : ℝ := speed_m_per_s * time_s
def length_train_m : ℝ := total_distance_m - length_bridge_m

theorem length_of_train :
  length_train_m = 109.98224142068636 :=
by
  sorry

end length_of_train_l735_735036


namespace polynomial_discriminant_l735_735126

theorem polynomial_discriminant (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
by
  sorry

end polynomial_discriminant_l735_735126


namespace prove_relationship_l735_735851

variable (f : ℝ → ℝ) (x : ℝ)
variable (f'' : ℝ → ℝ)
variable a b c : ℝ
axiom fx_symmetry : ∀ x : ℝ, f x = f (2 - x)
axiom condition : ∀ x : ℝ, x < 1 → (x - 1) * f'' x < 0
def a := f 0
def b := f (1 / 2)
def c := f 3

theorem prove_relationship : c < a ∧ a < b :=
sorry

end prove_relationship_l735_735851


namespace count_more_fives_than_twos_in_book_l735_735815

open Nat

def count_digit_in_range (digit start end : ℕ) : ℕ :=
  (List.range' start (end + 1 - start)).sumBy (λ n => (digits 10 n).count digit)

theorem count_more_fives_than_twos_in_book :
  count_digit_in_range 5 1 625 - count_digit_in_range 2 1 625 = 20 :=
  sorry

end count_more_fives_than_twos_in_book_l735_735815


namespace maximize_area_difference_l735_735235

-- Define the conditions
def point_P : ℝ × ℝ := (1, 1)
def circle_region (x y : ℝ) : Prop := x^2 + y^2 ≤ 4

-- Define the line equation to be proven
def line_eq (x y : ℝ) : Prop := x + y - 2 = 0

-- The theorem statement
theorem maximize_area_difference : 
  ∃ l : ℝ → ℝ → Prop, 
    (∀ x y, l x y ↔ line_eq x y)
    ∧ (P line_eq point_P.1 point_P.2)
    ∧ (divides_circle_maximally l) := 
  sorry

end maximize_area_difference_l735_735235


namespace students_with_exactly_two_talents_l735_735429

def students_total : ℕ := 120
def students_cannot_sing : ℕ := 50
def students_cannot_dance : ℕ := 75
def students_cannot_act : ℕ := 45
def students_no_talents : ℕ := 15

theorem students_with_exactly_two_talents :
  let students_can_sing := students_total - students_cannot_sing in
  let students_can_dance := students_total - students_cannot_dance in
  let students_can_act := students_total - students_cannot_act in
  let students_with_talents := students_total - students_no_talents in
  let total_talents := students_can_sing + students_can_dance + students_can_act in
  let students_with_one_or_two_talents := total_talents - students_with_talents in
  let students_with_two_talents := students_with_one_or_two_talents - (students_with_talents - students_total)
  in students_with_two_talents = 70 := by
  sorry

end students_with_exactly_two_talents_l735_735429


namespace perpendicular_bisector_eq_l735_735328

theorem perpendicular_bisector_eq :
  ∃ (m : ℝ) (x₀ y₀ : ℝ), x₀ = 3 ∧ y₀ = 1 ∧ m = 1 ∧
  ∀ x y, y - 1 = m * (x - 3) ↔ x - y - 2 = 0 :=
begin
  sorry
end

end perpendicular_bisector_eq_l735_735328


namespace triangle_inequality_proof_l735_735654

-- Define the sides of the triangle as elements of the reals
variables (a b c : ℝ)

-- Define semiperimeter s
def s : ℝ := (a + b + c) / 2

-- Define the conditions: a, b, c are sides of a triangle
def triangle_inequality : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the final inequality that we need to prove
theorem triangle_inequality_proof (h : triangle_inequality a b c) : 
  (a * b) / (s a b c - c) + (b * c) / (s a b c - a) + (c * a) / (s a b c - b) ≥ 4 * (s a b c) := 
sorry

end triangle_inequality_proof_l735_735654


namespace sector_area_correct_l735_735580

-- Definitions based on the conditions
def sector_perimeter := 16 -- cm
def central_angle := 2 -- radians
def radius := 4 -- The radius computed from perimeter condition

-- Lean 4 statement to prove the equivalent math problem
theorem sector_area_correct : ∃ (s : ℝ), 
  (∀ (r : ℝ), (2 * r + r * central_angle = sector_perimeter → r = 4) → 
  (s = (1 / 2) * central_angle * (radius) ^ 2) → 
  s = 16) :=
by 
  sorry

end sector_area_correct_l735_735580


namespace boiling_point_of_water_celsius_l735_735706

theorem boiling_point_of_water_celsius :
  (boiling_point_F : ℝ) 
  (melting_point_F : ℝ) 
  (melting_point_C : ℝ)
  (temp_pot_C : ℝ) 
  (temp_pot_F : ℝ) :
  boiling_point_F = 212 ∧
  melting_point_F = 32 ∧ 
  melting_point_C = 0 ∧ 
  temp_pot_C = 60 ∧ 
  temp_pot_F = 140 →
  boiling_point_C = 100 :=
by
  intros h
  sorry

end boiling_point_of_water_celsius_l735_735706


namespace find_p_l735_735463

variable (p q : ℝ) (k : ℕ)

theorem find_p (h_sum : ∀ (α β : ℝ), α + β = 2) (h_prod : ∀ (α β : ℝ), α * β = k) (hk : k > 0) :
  p = -2 := by
  sorry

end find_p_l735_735463


namespace probability_correct_l735_735887

def region (a b : ℝ) : Prop :=
  a + b ≤ 6 ∧ a > 0 ∧ b > 0

def has_root (a b : ℝ) : Prop :=
  b > a^2

def total_area : ℝ :=
  18  -- The area of the region a + b <= 6, a > 0, b > 0

noncomputable def favorable_area : ℝ :=
  ∫ x in 0..2, (6 - x - x^2)

def probability_has_root : ℝ :=
  favorable_area / total_area

theorem probability_correct :
  probability_has_root = 11 / 27 :=
by sorry

end probability_correct_l735_735887


namespace chord_length_and_distance_of_parabola_focus_l735_735822

theorem chord_length_and_distance_of_parabola_focus:
  ∀ (A B : ℝ × ℝ),
    let parabola := λ p : ℝ × ℝ, p.2 ^ 2 = 8 * p.1
    let mid := (A.1 + B.1 / 2, (A.2 + B.2) / 2)
    let line_eq := λ p : ℝ × ℝ, p.2 = p.1 - 2
    parabola A ∧ parabola B ∧ line_eq A ∧ line_eq B
    → distance A B = 16 ∧ distance (mid.1, mid.2) (-2, mid.2) = 8 :=
begin
  sorry
end

end chord_length_and_distance_of_parabola_focus_l735_735822


namespace incenter_bisector_ratio_l735_735174

-- Define the given sides of Triangle ABC
def AB := 15
def BC := 12
def AC := 18

-- Define the statement for the ratio 
theorem incenter_bisector_ratio (ΔABC : Triangle) (I : Point) (C : Angle) (AB = 15) (BC = 12) (AC = 18) :
  divides I (angle_bisector C) (2, 1) :=
sorry

end incenter_bisector_ratio_l735_735174


namespace coefficient_of_x_squared_term_in_expansion_l735_735484

-- Define the polynomials
def poly1 := 2 * X^3 + 4 * X^2 - 3 * X
def poly2 := 3 * X^2 - 8 * X - 5

-- The theorem statement
theorem coefficient_of_x_squared_term_in_expansion :
  coefficient (poly1 * poly2) 2 = 4 :=
by
  sorry

end coefficient_of_x_squared_term_in_expansion_l735_735484


namespace prisha_other_number_l735_735993

def prisha_numbers (a b : ℤ) : Prop :=
  3 * a + 2 * b = 105 ∧ (a = 15 ∨ b = 15)

theorem prisha_other_number (a b : ℤ) (h : prisha_numbers a b) : b = 30 :=
sorry

end prisha_other_number_l735_735993


namespace valid_two_digit_number_count_l735_735901

def is_prime (n : ℕ) : Prop := ∃ p, nat.prime p ∧ p = n

def is_perfect_square_less_than_10 (n : ℕ) : Prop := ∃ m, m * m = n ∧ n < 10

def valid_digit (n : ℕ) : Prop := is_prime n ∨ is_perfect_square_less_than_10 n

def is_valid_two_digit_number (n : ℕ) : Prop :=
  let tens := n / 10
  let ones := n % 10
  10 ≤ n ∧ n < 100 ∧ valid_digit tens ∧ valid_digit ones

theorem valid_two_digit_number_count : 
  (∃ count : ℕ, count = 36 ∧ ∀ n, is_valid_two_digit_number n ↔ n ≤ count) :=
sorry

end valid_two_digit_number_count_l735_735901


namespace apple_problem_l735_735727

theorem apple_problem (x : ℕ) (h1 : 1430 % x = 0)
  (h2 : 1430 % (x + 45) = 0)
  (h3 : 1430 / x - 1430 / (x + 45) = 9) :
  1430 / x = 22 :=
begin
  -- sorry, to be filled with proof
  sorry
end

end apple_problem_l735_735727


namespace rectangle_area_inscribed_circle_l735_735011

theorem rectangle_area_inscribed_circle (r l w : ℝ) (h_r : r = 7)
(h_ratio : l / w = 2) (h_w : w = 2 * r) :
  l * w = 392 :=
by sorry

end rectangle_area_inscribed_circle_l735_735011


namespace sum_heartsuits_l735_735494

def heartsuit (x : ℝ) : ℝ := (x + x^2) / 2

theorem sum_heartsuits : heartsuit 2 + heartsuit 3 + heartsuit 4 + heartsuit 5 = 34 :=
by
  sorry

end sum_heartsuits_l735_735494


namespace point_in_third_quadrant_l735_735536

open Complex

-- Define that i is the imaginary unit
def imaginary_unit : ℂ := Complex.I

-- Define the condition i * z = 1 - 2i
def condition (z : ℂ) : Prop := imaginary_unit * z = (1 : ℂ) - 2 * imaginary_unit

-- Prove that the point corresponding to the complex number z is located in the third quadrant
theorem point_in_third_quadrant (z : ℂ) (h : condition z) : z.re < 0 ∧ z.im < 0 := sorry

end point_in_third_quadrant_l735_735536


namespace probability_floor_sqrt_100x_eq_180_given_floor_sqrt_x_eq_18_l735_735969

open Real

noncomputable def probability_event : ℝ :=
  ((327.61 - 324) / (361 - 324))

theorem probability_floor_sqrt_100x_eq_180_given_floor_sqrt_x_eq_18 :
  probability_event = 361 / 3700 :=
by
  -- Conditions and calculations supplied in the problem
  sorry

end probability_floor_sqrt_100x_eq_180_given_floor_sqrt_x_eq_18_l735_735969


namespace find_income_before_taxes_l735_735785

def income_before_taxes (gross_income : ℝ) : ℝ :=
  if gross_income > 3000 then gross_income - 0.10 * (gross_income - 3000) else gross_income

theorem find_income_before_taxes (net_income : ℝ) (gross_income : ℝ) :
  income_before_taxes gross_income = net_income → net_income = 12000 → gross_income = 13000 :=
by
  intros h1 h2
  sorry

end find_income_before_taxes_l735_735785


namespace conditional_probability_B_given_A_l735_735814

def P : Type := ℝ

noncomputable def probability_of_event_A : P := 0.15
noncomputable def probability_of_event_AB : P := 0.05

theorem conditional_probability_B_given_A :
  (probability_of_event_AB / probability_of_event_A) = 1 / 3 := sorry

end conditional_probability_B_given_A_l735_735814


namespace line_parallel_to_intersection_l735_735867

variables {α β : Type} [plane α] [plane β]
variable (a : line)

-- Conditions
def line_parallel_to_planes :=
  parallel a α ∧ parallel a β

def planes_intersect_in_line (b : line) :=
  intersection α β = b

-- The theorem we want to prove
theorem line_parallel_to_intersection (b : line)
  (h1 : line_parallel_to_planes a α β)
  (h2 : planes_intersect_in_line b α β) :
  parallel a b :=
sorry

end line_parallel_to_intersection_l735_735867


namespace distribution_of_balls_into_boxes_l735_735902

noncomputable def partitions_of_6_into_4_boxes : ℕ := 9

theorem distribution_of_balls_into_boxes :
  let balls := 6
  let boxes := 4
  let ways := partitions_of_6_into_4_boxes
  ways = 9 :=
by
  let balls := 6
  let boxes := 4
  let ways := partitions_of_6_into_4_boxes
  sorry

end distribution_of_balls_into_boxes_l735_735902


namespace sector_angle_l735_735186

theorem sector_angle (r l : ℝ) (h₁ : 2 * r + l = 4) (h₂ : 1/2 * l * r = 1) : l / r = 2 :=
by
  sorry

end sector_angle_l735_735186


namespace tangent_angle_range_l735_735397

noncomputable def curve (x : ℝ) : ℝ := x^3 - x + 2/3

theorem tangent_angle_range : 
  ∃ α : ℝ, ∀ P : ℝ, 
  let m := 3 * P^2 - 1 in 
  (m ∈ [-1, ∞) ∧ (α = arctan m ∧ (α ∈ [0, π/2) ∨ α ∈ [3*π/4, π]))) :=
sorry

end tangent_angle_range_l735_735397


namespace distance_P_to_base_AB_l735_735798

-- Given conditions
variables (A B C P : Type)
variable [triangle : Triangle ABC]
variable [base_AB : segment A B]
variable [altitude_C : altitude C (segment A B)]
variable [height_C : distance C (segment A B) = 3]
variable (line_parallel : ∃ l, line l ∧ parallel l base_AB ∧ ∃ P_inside, point P_inside ∧ P ∈ l ∧ divides_triangle l (segment A B) ABC 1 2)
variable (area_ratio : area_ratio (portion_below_line P triangle base_AB) (triangle_area ABC) = 1/3)

-- The problem: to find the distance from point P to the base AB
theorem distance_P_to_base_AB : (distance P base_AB) = 1 := by
  sorry

end distance_P_to_base_AB_l735_735798


namespace quadratic_polynomial_discriminant_l735_735142

def P (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∃ x : ℝ, P a b c x = x - 2 ∧ (discriminant a (b - 1) (c + 2) = 0))
  (h₂ : ∃ x : ℝ, P a b c x = 1 - x / 2 ∧ (discriminant a (b + 1 / 2) (c - 1) = 0)) :
  discriminant a b c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l735_735142


namespace count_integers_l735_735110

theorem count_integers (f : ℕ → ℕ!) (n_min n_max : ℕ) 
    (h_min : n_min = 1) (h_max : n_max = 100) 
    (P : ℕ → Bool) (hP : ∀ n, 1 ≤ n ∧ n ≤ 100 → P(n) = (∃ k : ℤ, k * ((n!) ^ n * (n - 1)) = (f n))) :
  (Finset.filter P (Finset.range (n_max + 1))).card = 74 :=
sorry

end count_integers_l735_735110


namespace half_last_k_digits_of_power_of_2_are_9_l735_735995

theorem half_last_k_digits_of_power_of_2_are_9 (k : ℕ) (h : k > 1) : 
  ∃ (n : ℕ), last_k_digits_are_at_least_half_9 (2^n) k :=
by
  -- formalization of the mathematical proof
  sorry

-- Definitions required for the proof
def last_k_digits (n : ℕ) (k : ℕ) : ℕ :=
  n % 10^k

def count_digit_9 (n : ℕ) : ℕ :=
  -- Function to count the number of digit 9s in the number n 
  sorry

def last_k_digits_are_at_least_half_9 (n : ℕ) (k : ℕ) : Prop :=
  count_digit_9 (last_k_digits n k) ≥ k / 2

end half_last_k_digits_of_power_of_2_are_9_l735_735995


namespace quadratic_polynomial_discriminant_l735_735140

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₁ : a ≠ 0)
  (h₂ : ∃ x : ℝ, a * x^2 + b * x + c = x - 2 ∧ (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h₃ : ∃ x : ℝ, a * x^2 + b * x + c = 1 - x / 2 ∧ (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l735_735140


namespace A_and_B_know_own_results_l735_735500

def students : Type := {A : Type} → {B : Type} → {C : Type} → {D : Type} → Prop

def result : students → (students → students → Prop) → students → Prop

-- Define the properties given in conditions
def conditions (s : students) : Prop :=
  (∃ excellent good : students, ∃ a b c d : students,
   excellent ≠ good ∧
   (result s b c ∧ result s a d) ∧
   (∀ b c, result s b c → ¬ result s b) ∧
   (∀ a d, result s a d → ¬ result s a))

-- The theorem to prove
theorem A_and_B_know_own_results (s : students) (cond : conditions s) : Prop :=
  ∃ a b : students, (result s a ∧ result s b)

end A_and_B_know_own_results_l735_735500


namespace time_to_return_l735_735038

-- Given conditions
def distance : ℝ := 1000
def return_speed : ℝ := 142.85714285714286

-- Goal to prove
theorem time_to_return : distance / return_speed = 7 := 
by
  sorry

end time_to_return_l735_735038


namespace find_length_BE_l735_735298

/-- Point F is taken on the extension of side AD of parallelogram ABCD.
    BF intersects diagonal AC at E and side DC at G.
    Given EF = 18 and GF = 30, find the length of BE such that BE = 6 * sqrt 6. --/
theorem find_length_BE
  (A B C D F E G : Type) 
  [parallelogram A B C D]
  (F_on_extension_AD : F ∈ (extension A D))
  (intersects_BF_AC : E ∈ (intersection (line B F) (line A C)))
  (intersects_BF_DC : G ∈ (intersection (line B F) (line D C)))
  (EF_eq : EF = 18)
  (GF_eq : GF = 30) :
  BE = 6 * Real.sqrt 6 := 
sorry

end find_length_BE_l735_735298


namespace num_valid_pairs_l735_735032

theorem num_valid_pairs (a b : ℕ) (hb : b > a) (h_unpainted_area : ab = 3 * (a - 4) * (b - 4)) :
  (∃ (a b : ℕ), b > a ∧ ab = 3 * (a-4) * (b-4) ∧ (a-6) * (b-6) = 12 ∧ ((a, b) = (7, 18) ∨ (a, b) = (8, 12))) ∧
  (2 = 2) :=
by sorry

end num_valid_pairs_l735_735032


namespace wet_surface_area_of_cistern_l735_735730

-- Conditions
def cistern_length := 9 -- meters
def cistern_width := 6 -- meters
def water_height_breadth := 2.25 -- meters

-- Correct Answer for the proof
def total_wet_surface_area := 121.5 -- square meters

-- The proof statement
theorem wet_surface_area_of_cistern :
  (cistern_length * cistern_width) + -- Bottom Area
  (2 * (cistern_length * water_height_breadth)) + -- Two longer sides
  (2 * (cistern_width * water_height_breadth)) = -- Two shorter sides
  total_wet_surface_area :=
by
  sorry

end wet_surface_area_of_cistern_l735_735730


namespace fraction_eq_l735_735010

theorem fraction_eq (x : ℝ) (h1 : x * 180 = 24) (h2 : x < 20 / 100) : x = 2 / 15 :=
sorry

end fraction_eq_l735_735010


namespace triangle_KLM_angles_l735_735175

-- Definitions and Assumptions
variables {A B C K L M : Type}
variables (α β γ : ℝ)

def isosceles_triangle (A B C : Type) (α : ℝ) : Prop :=
  ∃ K, angle A K B = α ∧ angle B K C = (π - α)

def sum_of_angles_eq (α β γ : ℝ) : Prop := α + β + γ = 2 * π

-- Main Theorem Statement
theorem triangle_KLM_angles
  (hABC : ∀ (A B C : Type), triangle A B C)
  (hAKB : isosceles_triangle A K B α)
  (hBLC : isosceles_triangle B L C β)
  (hCMA : isosceles_triangle C M A γ)
  (h_sum : sum_of_angles_eq α β γ) :
  ∃ K L M, triangle K L M ∧ angle K L M = α / 2 ∧ angle L M K = β / 2 ∧ angle M K L = γ / 2 :=
sorry

end triangle_KLM_angles_l735_735175


namespace mod_division_l735_735651

theorem mod_division (N : ℕ) (h₁ : N = 5 * 2 + 0) : N % 4 = 2 :=
by sorry

end mod_division_l735_735651


namespace arithmetic_sequence_general_formula_sum_first_n_terms_l735_735495

variable {a : ℕ → ℕ}
variable {b : ℕ → ℕ}

-- Definitions based on conditions
def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) := ∀ n : ℕ, a (n + 1) = a n + d

def sum_terms_condition (a : ℕ → ℕ) (d : ℕ) :=
  a 2 + 2 * a 4 = 20

def geometric_sequence_condition (a : ℕ → ℕ) :=
  a 1 ≠ 0 ∧ (a 3)^2 = a 1 * a 9

-- General formula for a_n
theorem arithmetic_sequence_general_formula (d : ℕ) (h_d : d ≠ 0)
  (h_sum_terms : sum_terms_condition a d)
  (h_geom_seq : geometric_sequence_condition a) :
  a = λ n, 2 * n := sorry

-- Definition b_n and theorem for sum of first n terms
def b_n (a : ℕ → ℕ) (n : ℕ) := 2^(n-1) * a n

def T_n (b : ℕ → ℕ) (n : ℕ) := ∑ k in finset.range n, b (k + 1)

theorem sum_first_n_terms (h_arith_seq : a = λ n, 2 * n) :
  ∀ n, T_n (b_n a) n = (n-1) * 2^(n+1) + 2 := sorry

end arithmetic_sequence_general_formula_sum_first_n_terms_l735_735495


namespace squirrel_count_l735_735295

theorem squirrel_count (n m : ℕ) (h1 : n = 12) (h2 : m = 12 + 12 / 3) : n + m = 28 := by
  sorry

end squirrel_count_l735_735295


namespace complement_of_A_l735_735897

-- Definition of the universal set U and the set A
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | (x - 1) * (x + 2) > 0}

-- Theorem statement for the complement of A in U
theorem complement_of_A:
  (U \ A) = {x | -2 ≤ x ∧ x ≤ 1} :=
  sorry

end complement_of_A_l735_735897


namespace multiplication_correct_l735_735392

theorem multiplication_correct :
  72514 * 99999 = 7250675486 :=
by
  sorry

end multiplication_correct_l735_735392


namespace sum_of_coefficients_of_y_terms_is_56_l735_735398

-- Define the terms
def exp1 : ℚ[x, y] := 2*x + 3*y + 2
def exp2 : ℚ[x, y] := x + 5*y + 6

-- Calculate the product
def product_expansion : ℚ[x, y] := exp1 * exp2

-- Extract the terms and calculate the sum
def sum_of_coefficients_of_y_terms : ℚ :=
  (product_expansion.coeff (1, 0) * 0) + 
  (product_expansion.coeff (0, 1) * 1) + 
  (product_expansion.coeff (0, 2) * 2 * 15) -- represents the coefficients of xy, y, y^2 terms respectively

-- The proof statement
theorem sum_of_coefficients_of_y_terms_is_56 : sum_of_coefficients_of_y_terms = 56 := by
  sorry

end sum_of_coefficients_of_y_terms_is_56_l735_735398


namespace hyperbola_n_range_l735_735543

noncomputable def hyperbola_range_n (m n : ℝ) : Set ℝ :=
  {n | ∃ (m : ℝ), (m^2 + n) + (3 * m^2 - n) = 4 ∧ ((m^2 + n) * (3 * m^2 - n) > 0) }

theorem hyperbola_n_range : ∀ n : ℝ, n ∈ hyperbola_range_n m n ↔ -1 < n ∧ n < 3 :=
by
  sorry

end hyperbola_n_range_l735_735543


namespace min_red_up_probability_card_l735_735357

theorem min_red_up_probability_card (cards : Fin 50) :
  (cards = 13) ∨ (cards = 38) ↔
  -- Conditions for Vasya and Asya's actions:
  ∃ (select_vasya : Fin 26 → Fin 50) (select_asya : Fin 26 → Fin 50),
    (∀ k : Fin 50, 
      probability.card_pos (select_vasya k.left ≤ k.to_nat ∧ k.to_nat < k.left + 25 → 
                            select_asya k.right ≤ k.to_nat ∧ k.to_nat < k.right + 25 → 
                            k.val = 13 ∨ k.val = 38))

end min_red_up_probability_card_l735_735357


namespace solve_for_x_l735_735493

-- Definitions of the conditions
def condition (x : ℚ) : Prop :=
  (x^2 - 6 * x + 8) / (x^2 - 9 * x + 14) = (x^2 - 3 * x - 18) / (x^2 - 2 * x - 24)

-- Statement of the theorem
theorem solve_for_x (x : ℚ) (h : condition x) : x = -5 / 4 :=
by 
  sorry

end solve_for_x_l735_735493


namespace polynomial_discriminant_l735_735130

theorem polynomial_discriminant (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
by
  sorry

end polynomial_discriminant_l735_735130


namespace matrix_vector_computation_l735_735263

-- Setup vectors and their corresponding matrix multiplication results
variables {R : Type*} [Field R]
variables {M : Matrix (Fin 2) (Fin 2) R} {u z : Fin 2 → R}

-- Conditions given in (a)
def condition1 : M.mulVec u = ![3, -4] :=
  sorry

def condition2 : M.mulVec z = ![-1, 6] :=
  sorry

-- Statement equivalent to the proof problem given in (c)
theorem matrix_vector_computation :
  M.mulVec (3 • u - 2 • z) = ![11, -24] :=
by
  -- Use the conditions to prove the theorem
  sorry

end matrix_vector_computation_l735_735263


namespace johns_payment_l735_735953

-- Define the value of the camera
def camera_value : ℕ := 5000

-- Define the rental fee rate per week as a percentage
def rental_fee_rate : ℝ := 0.1

-- Define the rental period in weeks
def rental_period : ℕ := 4

-- Define the friend's contribution rate as a percentage
def friend_contribution_rate : ℝ := 0.4

-- Theorem: Calculate how much John pays for the camera rental
theorem johns_payment :
  let weekly_rental_fee := camera_value * rental_fee_rate
  let total_rental_fee := weekly_rental_fee * rental_period
  let friends_contribution := total_rental_fee * friend_contribution_rate
  let johns_payment := total_rental_fee - friends_contribution
  johns_payment = 1200 :=
by
  sorry

end johns_payment_l735_735953


namespace minimal_flip_probability_l735_735364

def flip_probability (n : ℕ) (k : ℕ) : ℚ :=
  if k <= 25 then
    (2 * k^2 - 52 * k + 676) / 676
  else
    let mirrored_k := 51 - k in
    (2 * mirrored_k^2 - 52 * mirrored_k + 676) / 676

theorem minimal_flip_probability :
  ∀ k, (13 ≤ k ∧ k ≤ 13) ∨ (38 ≤ k ∧ k ≤ 38) :=
by
  intro k
  sorry

end minimal_flip_probability_l735_735364


namespace set_range_of_three_numbers_l735_735778

theorem set_range_of_three_numbers (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : (a + b + c) / 3 = 6) 
(h4 : b = 6) (h5 : c = 10) : c - a = 8 := by
  sorry

end set_range_of_three_numbers_l735_735778


namespace cut_and_rearrange_square_l735_735244

theorem cut_and_rearrange_square (shape : fin 6 × fin 6 → Prop) 
  (h_shape_area : ∑ x in finset.univ, ∑ y in finset.univ, if shape (x, y) then 1 else 0 = 36) :
  ∃ (parts : list (fin 6 × fin 6 → Prop)), 
    parts.length = 4 ∧
    (∀ part ∈ parts, (∑ x in finset.univ, ∑ y in finset.univ, if part (x, y) then 1 else 0) = 9) ∧ 
    (∃ (new_shape : fin 3 × fin 3 → Prop), 
      (∀ x y, new_shape (x, y) ↔ 
        ∃ (part : fin 6 × fin 6 → Prop) (hx : x < 6) (hy : y < 6), 
          part (fin.mk ((3 * (x/3) + (x % 3)) % 6, hx), fin.mk ((3 * (y/3) + (y % 3)) % 6, hy)) ≡ shape (fin.mk (x, hx), fin.mk (y, hy))))):
sorry

end cut_and_rearrange_square_l735_735244


namespace max_value_PM_PF1_l735_735336

noncomputable def ellipse : Set (ℝ × ℝ) :=
  { p | let ⟨x, y⟩ := p in x^2 / 25 + y^2 / 9 = 1 }

noncomputable def circle : Set (ℝ × ℝ) :=
  { p | let ⟨x, y⟩ := p in x^2 + (y - 2 * Real.sqrt 5)^2 = 1 }

def F1 : ℝ × ℝ := (-4, 0)

def max_PM_PF1 : ℝ := 17

theorem max_value_PM_PF1 : ∀ (P M : (ℝ × ℝ)),
  P ∈ ellipse ∧ M ∈ circle → |dist P M + dist P F1| ≤ max_PM_PF1 := by
  sorry

end max_value_PM_PF1_l735_735336


namespace integer_sum_of_prime_factor_reciprocals_l735_735858

open Nat

theorem integer_sum_of_prime_factor_reciprocals
  (n : ℕ) (hn1 : 1 < n)
  (hdiv : n ∣ ∑ i in range (n + 1), i ^ (totient n)) :
  let p_factors := unique_factorization_monoid.factors n in
  ∀ p ∈ p_factors, Prime p → 
    (∑ p in p_factors, 1 / p + 1 / (∏ q in p_factors, q) : ℝ).denom = 1 := sorry

end integer_sum_of_prime_factor_reciprocals_l735_735858


namespace train_length_l735_735037

variable (bridge_length : ℝ) (time_seconds : ℝ) (speed_kmh : ℝ)

theorem train_length (h_bridge : bridge_length = 180) (h_time : time_seconds = 20) (h_speed : speed_kmh = 77.4) : 
  let speed_mps := speed_kmh * 1000 / 3600 in
  let distance_covered := speed_mps * time_seconds in
  let length_train := distance_covered - bridge_length in
  length_train = 250 :=
by
  sorry

end train_length_l735_735037


namespace product_of_positive_integer_c_with_real_roots_l735_735824

theorem product_of_positive_integer_c_with_real_roots :
  (∏ c in finset.filter (λ c : ℕ, 12 * 1^2 + 19 * 1 + c < 361 / 48) (finset.range 8)) = 5040 :=
by {
  sorry
}

end product_of_positive_integer_c_with_real_roots_l735_735824


namespace sum_of_abc_l735_735812

theorem sum_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) 
  (eq1 : a^2 + b * c = 115) (eq2 : b^2 + a * c = 127) (eq3 : c^2 + a * b = 115) :
  a + b + c = 22 := by
  sorry

end sum_of_abc_l735_735812


namespace fraction_proof_l735_735119

variables (m n p q : ℚ)

theorem fraction_proof
  (h1 : m / n = 18)
  (h2 : p / n = 9)
  (h3 : p / q = 1 / 15) :
  m / q = 2 / 15 :=
by sorry

end fraction_proof_l735_735119


namespace intersection_point_sum_l735_735276

theorem intersection_point_sum {p q r s : ℕ} : 
  let A := (0,0) in
  let B := (2,4) in
  let C := (3,3) in
  let D := (5,0) in
  ∃ (p q r s : ℕ), ((p + q + r + s = 30) ∧ gcd p q = 1 ∧ gcd r s = 1 ∧ 
  (intersection_point (line_through A (divides_equal_area_line A B C D)) (line_segment C D)) = 
  (p / q, r / s))
  .
sorry

end intersection_point_sum_l735_735276


namespace sum_of_cubes_of_roots_l735_735721

theorem sum_of_cubes_of_roots (x₁ x₂ : ℝ) :
    (3 * x₁^2 - 5 * x₁ - 2 = 0) →
    (3 * x₂^2 - 5 * x₂ - 2 = 0) →
    (x₁ + x₂ = (5:ℝ)/3) →
    (x₁ * x₂ = -(2:ℝ)/3) →
    x₁^3 + x₂^3 = 215 / 27 :=
by assume
    h1 : (3 * x₁^2 - 5 * x₁ - 2 = 0)
    h2 : (3 * x₂^2 - 5 * x₂ - 2 = 0)
    h3 : (x₁ + x₂ = (5:ℝ)/3)
    h4 : (x₁ * x₂ = -(2:ℝ)/3)
suffices sorry -- proof goes here

end sum_of_cubes_of_roots_l735_735721


namespace minimal_flip_probability_l735_735365

def flip_probability (n : ℕ) (k : ℕ) : ℚ :=
  if k <= 25 then
    (2 * k^2 - 52 * k + 676) / 676
  else
    let mirrored_k := 51 - k in
    (2 * mirrored_k^2 - 52 * mirrored_k + 676) / 676

theorem minimal_flip_probability :
  ∀ k, (13 ≤ k ∧ k ≤ 13) ∨ (38 ≤ k ∧ k ≤ 38) :=
by
  intro k
  sorry

end minimal_flip_probability_l735_735365


namespace bankers_discount_is_281_25_l735_735738

theorem bankers_discount_is_281_25 (BG : ℝ) (R : ℝ) (T : ℝ) (BD : ℝ) : 
  BG = 180 → R = 0.12 → T = 3 → BD = BG / (1 - (R * T)) → BD = 281.25 :=
by
  intros hBG hR hT hBD
  rw [hBG, hR, hT, hBD]
  have : 180 / (1 - (0.12 * 3)) = 281.25 := sorry
  exact this

end bankers_discount_is_281_25_l735_735738


namespace problem_solution_l735_735053

noncomputable def sum_of_terms : ℚ :=
  (1 / (2 : ℚ) ^ 1980) * (Finset.range 496).sum (λ n, (-3 : ℚ) ^ n * Nat.choose 1980 (4 * n))

theorem problem_solution :
  sum_of_terms = -1 / 2 :=
sorry

end problem_solution_l735_735053


namespace tan_a2_a12_l735_735189

noncomputable def arithmetic_term (a d : ℝ) (n : ℕ) : ℝ := a + d * (n - 1)

theorem tan_a2_a12 (a d : ℝ) (h : a + (a + 6 * d) + (a + 12 * d) = 4 * Real.pi) :
  Real.tan (arithmetic_term a d 2 + arithmetic_term a d 12) = - Real.sqrt 3 :=
by
  sorry

end tan_a2_a12_l735_735189


namespace A_and_C_together_work_time_l735_735755

theorem A_and_C_together_work_time
  (A : ℝ) (B : ℝ) (C : ℝ) (W : ℝ) 
  (hA : A = W / 4)
  (hBC : B + C = W / 2)
  (hB : B = W / 4) :
  (A + C = W / 2) -> C = W / 4 :=
begin
  intros hAC,
  sorry
end

end A_and_C_together_work_time_l735_735755


namespace smallest_integer_k_l735_735423

theorem smallest_integer_k (k : ℕ) : 
  (∀ k, 
    (∃ placement : fin 6 × fin 6 → option (fin 6 × fin 6), 
      (∀ (x y : fin 6 × fin 6), 
        (placement x = some y → placement y = some x) ∧ 
        ((x ≠ y → placement x ≠ some y) ∧ 
        (x = y → placement x = none))) ∧ 
      (∀ (x y : fin 6 × fin 6), 
        (placement x = some y → (x ≠ y) ∧ 
        (placement y = some x) ∧ 
        (x.fst = y.fst ∨ x.snd = y.snd)))) 
  → ((∑ x, (if placement x = none then 1 else 0) = 2 * (13 + k)))) 
  → k = 5 :=
sorry

end smallest_integer_k_l735_735423


namespace lychee_production_increase_l735_735983

variable (x : ℝ) (hx : x > 0)

theorem lychee_production_increase (x : ℝ) (hx : x > 0) :
  let initial_production := 1 in
  let production_2005 := initial_production * (1 + x / 100) in
  let production_2006 := production_2005 * (1 + x / 100) in
  production_2006 = initial_production * ((1 + x / 100) * (1 + x / 100)) :=
by
  sorry

end lychee_production_increase_l735_735983


namespace triangle_equilateral_l735_735944

variables (A B C P A' B' C' : Type) [triangle : Geometry.Triangle A B C P]

-- Setup the conditions as hypotheses
hypothesis (H1 : ∠ B P C = ∠ A + 60°)
hypothesis (H2 : ∠ A P C = ∠ B + 60°)
hypothesis (H3 : ∠ A P B = ∠ C + 60°)
hypothesis (HA' : A' ∈ circumcircle(ABC ∧ (A', P, A) are collinear))
hypothesis (HB' : B' ∈ circumcircle(ABC ∧ (B', P, B) are collinear))
hypothesis (HC' : C' ∈ circumcircle(ABC ∧ (C', P, C) are collinear))

-- Define the theorem to prove that triangle A'B'C' is equilateral
theorem triangle_equilateral :
  is_equilateral △ A' B' C' :=
sorry

end triangle_equilateral_l735_735944


namespace dot_product_q_r_l735_735216

open Real 

variables {V : Type*} [inner_product_space ℝ V] [normed_space ℝ V] [finite_dimensional ℝ V]
variables (p q r : V)
hypothesis (hp : ∥p∥ = 1)
hypothesis (hq : ∥q∥ = 1)
hypothesis (hpq : ∥p + q∥ = 2)
hypothesis (hr : r - 2 • p - 3 • q = 4 • (p ×ₑ q))

theorem dot_product_q_r : inner q r = 5 :=
by
  sorry

end dot_product_q_r_l735_735216


namespace expected_other_cars_l735_735447

namespace CarDealership

def sports_to_sedan_ratio (s : ℕ) (d : ℕ) : Prop := 5 * d = 8 * s

def sedan_to_other_ratio (d : ℕ) (o : ℕ) : Prop := d = 2 * o

def dealership_sales (expected_sports : ℕ) (expected_sedans : ℕ) (expected_others : ℕ) : Prop :=
  expected_sports = 35 ∧
  sports_to_sedan_ratio expected_sports expected_sedans ∧
  sedan_to_other_ratio expected_sedans expected_others

theorem expected_other_cars :
  ∃ (expected_others : ℕ), dealership_sales 35 56 expected_others ∧ expected_others = 28 :=
by
  apply Exists.intro 28
  split
  . split
    . exact eq.refl 35
    . unfold sports_to_sedan_ratio
      exact eq.refl (5 * 56)
    . unfold sedan_to_other_ratio
      exact eq.refl 56
  . exact eq.refl 28

end CarDealership

end expected_other_cars_l735_735447


namespace area_of_right_triangle_l735_735941

theorem area_of_right_triangle
  (BC AC : ℝ)
  (h1 : BC * AC = 16) : 
  0.5 * BC * AC = 8 := by 
  sorry

end area_of_right_triangle_l735_735941


namespace derivative_of_y_l735_735744

noncomputable def y (x : ℝ) : ℝ := 
  (6^x * (sin (4 * x) * log 6 - 4 * cos (4 * x))) / (16 + (log 6)^2)

theorem derivative_of_y (x : ℝ) : 
  deriv y x = 6^x * sin (4 * x) :=
by
  sorry

end derivative_of_y_l735_735744


namespace triangle_height_l735_735559

variables (AB BC AC : ℝ) (h : ℝ)

def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

theorem triangle_height (h : ℝ) :
  AB = 20 → BC = 99 → AC = 101 →
  let s := semi_perimeter AB BC AC in
  h = (2 * 990) / BC :=
by
  sorry

end triangle_height_l735_735559


namespace locus_of_points_l735_735210

variables (A B C D P : ℝ) -- Representing the points as real numbers on a line
-- Declaring the conditions that A, B, C, D are distinct
variables (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D) (h4 : B ≠ C) (h5 : B ≠ D) (h6 : C ≠ D)

theorem locus_of_points (hABCD : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D): 
  ∃ (O : ℝ) (r : ℝ), 
    ∀ P : ℝ, 
    (circle_through_three_points A B P ∩ 
     circle_through_three_points C D P ≠ ∅) → 
    circle O r := sorry

end locus_of_points_l735_735210


namespace students_in_grades_v_vi_l735_735681

theorem students_in_grades_v_vi (n a b c p q : ℕ) (h1 : n = 100*a + 10*b + c)
  (h2 : a * b * c = p) (h3 : (p / 10) * (p % 10) = q) : n = 144 :=
sorry

end students_in_grades_v_vi_l735_735681


namespace tangent_line_slope_4_tangent_line_at_point_tangent_line_through_origin_l735_735888

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 16

theorem tangent_line_slope_4 :
  ∀ {l : LinearMap ℝ ℝ}, l.slope = 4 →
  (l = λ x, 4*x-18) ∨ (l = λ x, 4*x-14) :=
by
  sorry

theorem tangent_line_at_point :
  tangent_line (λ x, x^3 + x - 16) (2, -6) = λ x, 13*x-32 :=
by
  sorry

theorem tangent_line_through_origin :
  tangent_line_passing_origin (λ x, x^3 + x - 16) = λ x, 13*x :=
by
  sorry

end tangent_line_slope_4_tangent_line_at_point_tangent_line_through_origin_l735_735888


namespace element_in_set_l735_735401

theorem element_in_set : 1 ∈ ({0, 1} : Set ℕ) := 
by 
  -- Proof goes here
  sorry

end element_in_set_l735_735401


namespace arithmetic_sequence_identity_l735_735593

theorem arithmetic_sequence_identity 
  (a : ℕ → ℤ) 
  (a_2_eq : a 2 = a 1 + 1) 
  (a_7_eq : a 7 = a 1 + 6) 
  (a_12_eq : a 12 = a 1 + 11) 
  (h : a 2 + 4 * a 7 + a 12 = 96) : 
  2 * (a 3) + a 15 = 48 :=
begin
  sorry
end

end arithmetic_sequence_identity_l735_735593


namespace leak_empties_in_72_hours_l735_735425

def rate_without_leak : ℝ := 1 / 8 -- The rate of filling the tank without the leak
def rate_with_leak : ℝ := 1 / 9  -- The combined rate of filling with the leak present

noncomputable def leak_rate : ℝ := rate_without_leak - rate_with_leak

theorem leak_empties_in_72_hours : 1 / leak_rate = 72 :=
by
  have h1 : rate_without_leak = 1 / 8 := rfl
  have h2 : rate_with_leak = 1 / 9 := rfl
  have h3 : leak_rate = rate_without_leak - rate_with_leak := rfl
  have h4 : leak_rate = (1 / 8) - (1 / 9)
  rw [h1, h2]
  have h5 : (1 / 8) - (1 / 9) = (9 - 8) / 72
  norm_num [h4, h5]
  rw [h5]
  have h6 : 1 / ((1 / 72) : ℝ) = 72 := by
  rw [inv_div, one_mul]
  norm_num
  have h7 : 1 / leak_rate = 72 := by
  rw [h3, h4, h6]
  exact h7
  sorry -- the final proof steps are skipped

end leak_empties_in_72_hours_l735_735425


namespace evaluate_operation_l735_735466

def table : Fin 4 → Fin 4 → Fin 4 :=
λ i j,
  match i, j with
  | 0, 0 => 0
  | 0, 1 => 2
  | 0, 2 => 1
  | 0, 3 => 3
  | 1, 0 => 2
  | 1, 1 => 0
  | 1, 2 => 3
  | 1, 3 => 1
  | 2, 0 => 1
  | 2, 1 => 3
  | 2, 2 => 0
  | 2, 3 => 2
  | 3, 0 => 3
  | 3, 1 => 1
  | 3, 2 => 2
  | 3, 3 => 0

theorem evaluate_operation : table 2 (table 3 1) = 0 :=
by {
  -- given the operation/condition table
  -- 3 corresponds to 2, 4 corresponds to 3, 1 corresponds to 0, 2 corresponds to 1
  have h1 : table 2 3 = 3 := 
    by decide, -- 3*2 from the table defined
  have h2 : table 3 0 = 3 := 
    by decide, -- 4*1 from the table defined
  rw [h1, h2],
  exact rfl,
}

end evaluate_operation_l735_735466


namespace simplify_expression_correct_l735_735309

noncomputable def simplify_expression (m n : ℝ) : ℝ :=
  ( (2 - n) / (n - 1) + 4 * ((m - 1) / (m - 2)) ) /
  ( n^2 * ((m - 1) / (n - 1)) + m^2 * ((2 - n) / (m - 2)) )

theorem simplify_expression_correct :
  simplify_expression (Real.rpow 400 (1/4)) (Real.sqrt 5) = (Real.sqrt 5) / 5 := 
sorry

end simplify_expression_correct_l735_735309


namespace correct_option_is_B_l735_735872

variable (f : ℝ → ℝ)
variable (h0 : f 0 = 2)
variable (h1 : ∀ x : ℝ, deriv f x > f x + 1)

theorem correct_option_is_B : 3 * Real.exp (1 : ℝ) < f 2 + 1 := sorry

end correct_option_is_B_l735_735872


namespace problem_statement_l735_735633

theorem problem_statement
  (a1 a2 a3 S : ℝ)
  (h1 : a1 > 1)
  (h2 : a2 > 1)
  (h3 : a3 > 1)
  (h_sum : a1 + a2 + a3 = S)
  (h_ineq : ∀ i ∈ {1, 2, 3}, (if i = 1 then a1 else if i = 2 then a2 else a3) ^ 2 / (if i = 1 then a1 else if i = 2 then a2 else a3 - 1) > S) :
  (1 / (a1 + a2)) + (1 / (a2 + a3)) + (1 / (a3 + a1)) > 1 := 
sorry

end problem_statement_l735_735633


namespace triangle_plane_l735_735039

def intersect_lines_determine_plane (l1 l2 : Set Point) : Prop := 
  ∃ P Q : Point, P ∈ l1 ∧ Q ∈ l1 ∧ P ∈ l2 ∧ Q ∈ l2 ∧ P ≠ Q

def parallel_lines_determine_plane (l1 l2 : Set Point) : Prop := 
  ∃ P1 P2 : Point, P1 ∈ l1 ∧ P2 ∈ l2 ∧ 
  ∀ P ∈ l1, ∀ Q ∈ l2, (P - Q) ∥ (P1 - P2)

def line_and_point_determine_plane (l : Set Point) (P : Point) : Prop := 
  P ∉ l ∧ ∃ Q : Point, Q ∈ l

def triangle (A B C : Point) : Prop := 
  ¬ collinear {A, B, C}

def three_non_collinear_points_determine_plane (A B C : Point) : Prop := 
  ¬ collinear {A, B, C}

theorem triangle_plane (A B C : Point) 
  (h : triangle A B C) : three_non_collinear_points_determine_plane A B C :=
sorry

end triangle_plane_l735_735039


namespace quadratic_function_expression_l735_735905

noncomputable def f : ℝ → ℝ := sorry

theorem quadratic_function_expression :
  (∀ x, ∃ a b c, f x = a * x^2 + b * x + c) ∧
  f 0 = 3 ∧
  (∀ x, f (x + 2) - f x = 4 * x + 2) →
  f = λ x, x^2 - x + 3 :=
by
  intro h,
  cases h with quadratic_eq h0,
  cases h0 with initial h_diff,
  sorry

end quadratic_function_expression_l735_735905


namespace pumpkin_count_sunshine_orchard_l735_735979

def y (x : ℕ) : ℕ := 3 * x^2 + 12

theorem pumpkin_count_sunshine_orchard :
  y 14 = 600 :=
by
  sorry

end pumpkin_count_sunshine_orchard_l735_735979


namespace equal_intercepts_l735_735891

theorem equal_intercepts (a : ℝ) (h : ∃p, (a * p, 0) = (0, a - 2)) : a = 1 ∨ a = 2 :=
sorry

end equal_intercepts_l735_735891


namespace shift_graph_sin_cos_l735_735700

open Real

theorem shift_graph_sin_cos :
  ∀ x : ℝ, sin (2 * x + π / 3) = cos (2 * (x + π / 12) - π / 3) :=
by
  sorry

end shift_graph_sin_cos_l735_735700


namespace circumcenter_XSY_equidistant_l735_735511

open EuclideanGeometry

variable {A B C D S X Y P M O' : Point}
variable {ABC : Triangle}
variable (h_nonisosceles : ¬ Isosceles ABC)
variable (h_acute : AcuteTriangle ABC)
variable (h_euler_line : EulerLine ABC A D)
variable (h_γ_passes_through_AD : Circle Γ S A D)
variable (h_γ_intersects_AB_AC : Intersects Γ AB X AC Y)
variable (h_projection_P : Projection A BC P)
variable (h_midpoint_M : Midpoint B C M)

theorem circumcenter_XSY_equidistant :
  let O' := Circumcenter (triangle.mk X S Y) in
  dist O' P = dist O' M := 
sorry

end circumcenter_XSY_equidistant_l735_735511


namespace work_done_in_give_days_l735_735405

theorem work_done_in_give_days (a b c : ℚ) :
  (a + b + c) = (1 / 4) ∧ a = (1 / 12) ∧ b = (1 / 9) → c = (1 / 18) :=
by
  intro h
  sorry

end work_done_in_give_days_l735_735405


namespace range_of_a_l735_735883

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then Real.log x / Real.log 2 else Real.log (-x) / Real.log (1 / 2)

theorem range_of_a (a : ℝ) (h : f a > f (-a)) : a ∈ Set.Ioo (-1 : ℝ) 0 ∨ a ∈ Set.Ioo 0 1 :=
sorry

end range_of_a_l735_735883


namespace prism_volume_l735_735124

open Real

def regular_triangular_prism_volume (a b : ℝ) (h : ℝ) : ℝ :=
  (sqrt 3 / 4) * a^2 * h

theorem prism_volume :
  ∃ (V : ℝ), 
    ((regular_triangular_prism_volume 4 6 (6 / sqrt 3) = V) ∨ (regular_triangular_prism_volume 4 6 (4 / sqrt 3) = V)) →
    (V = 4 * sqrt 3) ∨ (V = (8 * sqrt 3) / 3) :=
begin
  sorry
end

end prism_volume_l735_735124


namespace factorial_fraction_simplification_l735_735460

theorem factorial_fraction_simplification :
  (7.factorial + 8.factorial) / (5.factorial * 6.factorial) = 21 / 40 :=
by
  sorry

end factorial_fraction_simplification_l735_735460


namespace maxwell_meets_brad_l735_735736

theorem maxwell_meets_brad :
  ∃ t : ℝ, t = 2 ∧ 
  (∀ distance max_speed brad_speed start_time, 
   distance = 14 ∧ 
   max_speed = 4 ∧ 
   brad_speed = 6 ∧ 
   start_time = 1 → 
   max_speed * (t + start_time) + brad_speed * t = distance) :=
by
  use 1
  sorry

end maxwell_meets_brad_l735_735736


namespace intersection_of_sets_l735_735179

theorem intersection_of_sets (A B : Set ℝ) (A_def : A = {x | 1 < 3^x ∧ 3^x ≤ 9}) (B_def : B = {x | ∃ y, y = Real.log 2 (2 - x)}) : 
  A ∩ B = {x | 0 < x ∧ x < 2} :=
sorry

end intersection_of_sets_l735_735179


namespace part_I_monotonic_intervals_part_II_slope_condition_part_III_extreme_points_l735_735545

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * Real.log x + x^2 - a * x

-- Part I
theorem part_I_monotonic_intervals (a : ℝ) (h_a : a = 5) :
  (∀ x, 0 < x ∧ x < (1/2) → f x a > f (1/2) a) ∧ 
  (∀ x, x > 2 → f x a > f 2 a) ∧
  (∀ x, (1/2) < x ∧ x < 2 → f x a < f (1/2) a ∧ f x a < f 2 a) :=
sorry

-- Part II
theorem part_II_slope_condition (a : ℝ) :
  (∀ x1 x2, x1 ≠ x2 → (f x2 a - f x1 a) / (x2 - x1) > 1) → a ≤ 3 :=
sorry

-- Part III
theorem part_III_extreme_points (x1 x2 m a : ℝ) 
  (h_extreme : x1 < x2 ∧ x2 > real.exp 1) 
  (h_f_extreme : f x1 a - f x2 a ≥ m) :
  m ≤ real.exp 2 - 1 / real.exp 2 - 4 :=
sorry

end part_I_monotonic_intervals_part_II_slope_condition_part_III_extreme_points_l735_735545


namespace jacques_initial_gumballs_l735_735951

def joanna_initial_gumballs : ℕ := 40
def each_shared_gumballs_after_purchase : ℕ := 250

theorem jacques_initial_gumballs (J : ℕ) (h : 2 * (joanna_initial_gumballs + J + 4 * (joanna_initial_gumballs + J)) = 2 * each_shared_gumballs_after_purchase) : J = 60 :=
by
  sorry

end jacques_initial_gumballs_l735_735951


namespace transform_quadratic_to_squared_form_l735_735686

theorem transform_quadratic_to_squared_form :
  ∀ x : ℝ, 2 * x^2 - 3 * x + 1 = 0 → (x - 3 / 4)^2 = 1 / 16 :=
by
  intro x h
  sorry

end transform_quadratic_to_squared_form_l735_735686


namespace problem_1_problem_2_l735_735199

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a|

theorem problem_1 (x : ℝ) : (∀ x, f x 4 < 8 - |x - 1|) → x ∈ Set.Ioo (-1 : ℝ) (13 / 3) :=
by sorry

theorem problem_2 (a : ℝ) : (∃ x, f x a > 8 + |2 * x - 1|) → a > 9 ∨ a < -7 :=
by sorry

end problem_1_problem_2_l735_735199


namespace odd_planet_not_observed_l735_735990

theorem odd_planet_not_observed (n : ℕ) (h_odd : n % 2 = 1) (h_unique_distances : ∀ i j : ℕ, i ≠ j → dist i j ≠ dist j i) : 
  ∃ p : ℕ, ∀ q : ℕ, (q ≠ p →  ¬ observer q = p) := 
by
  sorry

end odd_planet_not_observed_l735_735990


namespace max_neg_integers_l735_735975

theorem max_neg_integers (
  a b c d e f g h : ℤ
) (h_a : a ≠ 0) (h_c : c ≠ 0) (h_e : e ≠ 0)
  (h_ineq : (a * b^2 + c * d * e^3) * (f * g^2 * h + f^3 - g^2) < 0)
  (h_abs : |d| < |f| ∧ |f| < |h|)
  : ∃ s, s = 5 ∧ ∀ (neg_count : ℕ), neg_count ≤ s := 
sorry

end max_neg_integers_l735_735975


namespace number_of_poles_needed_l735_735775

def length := 90
def width := 40
def distance_between_poles := 5

noncomputable def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

theorem number_of_poles_needed (l w d : ℕ) : perimeter l w / d = 52 :=
by
  rw [perimeter]
  sorry

end number_of_poles_needed_l735_735775


namespace area_of_region_l735_735806

theorem area_of_region :
  let C := {p : ℝ × ℝ | (p.1 ^ 2 + p.2 ^ 2 - 6 * p.1 + 14 * p.2 = -45)} in
  ∃ r : ℝ, ∃ center : ℝ × ℝ, 
  let A := {q : ℝ × ℝ | (q.1 - center.1) ^ 2 + (q.2 - center.2) ^ 2 = r ^ 2} in
  (r = sqrt 13) ∧ (area_of_circle r = 13 * Real.pi) ∧ C = A :=
sorry

end area_of_region_l735_735806


namespace polynomial_discriminant_l735_735125

theorem polynomial_discriminant (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
by
  sorry

end polynomial_discriminant_l735_735125


namespace place_stone_in_1000_l735_735371

/--
There is an infinite one-way checkered strip with cells numbered by natural numbers and a bag with 10 stones.
Initially, there are no stones in the cells of the strip. The following actions are possible:
- Move a stone from the bag to the first cell of the strip or back;
- If there is a stone in the cell numbered \(i\), you can move a stone from the bag to the cell numbered \(i+1\) or back.
We need to prove that it is possible to place a stone in cell number \(1000\) following these rules.
-/
theorem place_stone_in_1000 (bag : fin 10) (strip : ℕ → ℕ) :
  ∃ (strip' : ℕ → ℕ), strip' 1000 = 1 :=
sorry

end place_stone_in_1000_l735_735371


namespace valid_integers_count_l735_735835

theorem valid_integers_count : 
  ∃ count : ℕ, count = 96 ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 → (Nat.factorial (n ^ 2 - 4) / (Nat.factorial n) ^ (n - 2)).denom = 1 → n ≥ 5) :=
by
  -- Mathematical proof skipped
  sorry

end valid_integers_count_l735_735835


namespace minimal_loss_arithmetic_progression_l735_735093

theorem minimal_loss_arithmetic_progression :
  ∃ (P A C X O Д : ℕ), 
    (A = P + 1) ∧
    (C = A + 1) ∧
    (X = C + 1) ∧
    (O = X + 1) ∧
    (Д = O + 1) ∧
    (P = 1) ∧
    (EXPENSE = 100000 * P + 10000 * A + 1000 * C + 100 * X + 10 * O + Д) ∧
    (INCOME = 100000 * A + 10000 * C + 1000 * X + 100 * O + 10 * Д + P) ∧
    (LOSS = EXPENSE - INCOME) ∧ 
    (EXPENSE - INCOME = 58000) := sorry

end minimal_loss_arithmetic_progression_l735_735093


namespace meet_after_9_steps_probability_l735_735987

open Real

def binomial (n k : ℕ) : ℕ := Nat.choose n k

def A_paths (i : ℕ) : ℕ := binomial 9 i
def B_paths (i : ℕ) : ℕ := binomial 9 (8 - i)

def meet_probability : ℝ :=
  (1 / 2 ^ 18) * ∑ i in Finset.range 10, (A_paths i) * (B_paths i)

theorem meet_after_9_steps_probability : meet_probability = 48620 / 262144 := sorry

end meet_after_9_steps_probability_l735_735987


namespace compute_r_l735_735636

variables {j p t m n x y r : ℝ}

theorem compute_r
    (h1 : j = 0.75 * p)
    (h2 : j = 0.80 * t)
    (h3 : t = p - r * p / 100)
    (h4 : m = 1.10 * p)
    (h5 : n = 0.70 * m)
    (h6 : j + p + t = m * n)
    (h7 : x = 1.15 * j)
    (h8 : y = 0.80 * n)
    (h9 : x * y = (j + p + t) ^ 2) : r = 6.25 := by
  sorry

end compute_r_l735_735636


namespace largest_n_l735_735878

/-- Given the sum of the first n terms of the sequence {a_n} is S_n = n^2 + n,
    the common terms of the sequences {a_n} and {2^(n-1)} are arranged in ascending order
    to form the sequence {b_n}.
    T_n be the sum of the first n terms of the sequence {a_n * b_n}.
    - Prove that the largest positive integer n such that T_n < 2023 is n = 6 -/
theorem largest_n (S_n : ℕ → ℕ) (a_n b_n : ℕ → ℕ) (T_n : ℕ → ℕ) :
  (∀ n, S_n n = n^2 + n) →
  (∀ n, a_n n = S_n n - S_n (n-1)) →
  a_n 1 = 2 →
  (∀ n, b_n n = 2^n) →
  (∀ n, T_n n = ∑ k in finset.range n, a_n (k+1) * b_n (k+1)) →
  ∃ n, T_n n < 2023 ∧ ∀ m, T_n m < 2023 → m ≤ n :=
  by
    sorry

end largest_n_l735_735878


namespace I_circumcircle_ABK_iff_K_circumcircle_CDJ_l735_735553

-- Assume required geometric entities and properties
variables {A B C D E F I J K : Point}
variables {AB CD EF BC AD : Line}
variables {ω : Circle}

-- Given conditions
axiom trapezoid_ABCD : parallel (Line.mk A B) (Line.mk C D)
axiom E_on_line_BC_outside : ∃ E, on_line E (Line.mk B C) ∧ ¬on_segment E (Segment.mk B C)
axiom F_in_segment_AD : ∃ F, on_segment F (Segment.mk A D)
axiom angle_DAE_eq_CBF : ∃ E F, ∠ D A E = ∠ C B F
axiom I_intersection : I = intersection (Line.mk C D) (Line.mk E F)
axiom J_intersection : J = intersection (Line.mk A B) (Line.mk E F)
axiom K_midpoint_EF : K = midpoint E F ∧ ¬on_line K (Line.mk A B)

-- The proof goal
theorem I_circumcircle_ABK_iff_K_circumcircle_CDJ :
  (I ∈ circumcircle (Triangle.mk A B K)) ↔ (K ∈ circumcircle (Triangle.mk C D J)) :=
sorry -- proof omitted

end I_circumcircle_ABK_iff_K_circumcircle_CDJ_l735_735553


namespace distance_from_O_to_plane_ABC_l735_735537

noncomputable def distance_from_center_to_plane (O A B C S : Point) (SA SB SC AB : ℝ) : ℝ :=
  let SD := Real.sqrt (SA^2 - (AB / 2)^2)
  in SD / 3

theorem distance_from_O_to_plane_ABC 
  (O A B C S : Point) 
  (SA SB SC AB : ℝ) 
  (h1 : triangle.is_isosceles_right_triangle A B C) 
  (h2 : S.dist A = 2)
  (h3 : S.dist B = 2) 
  (h4 : S.dist C = 2) 
  (h5 : A.dist B = 2) 
  (h6 : S.is_on_sphere O A B C) :
  distance_from_center_to_plane O A B C S 2 2 2 2 = Real.sqrt 3 / 3 := sorry

end distance_from_O_to_plane_ABC_l735_735537


namespace general_term_arithmetic_seq_l735_735857

-- Definitions of sequences and given conditions
def arithmetic_seq (a b : ℕ) (n : ℕ) : ℕ := a + (n - 1) * b
def geometric_seq (b a : ℕ) (n : ℕ) : ℕ := b * a^(n - 1)

variables {a b : ℕ}
variables (a_pos : 1 < a) (b_pos : 1 < b)
variables (h1 : a < b) (h2 : b * 2 < a + 2 * b)

theorem general_term_arithmetic_seq :
  (∀ n : ℕ, ∃ m : ℕ, arithmetic_seq a b m + 3 = geometric_seq b a (n + 1)) →
  ∀ n : ℕ, arithmetic_seq 2 5 n = 5 * n - 3 :=
begin
  intros h n,
  sorry
end

end general_term_arithmetic_seq_l735_735857


namespace geometric_problem_solution_l735_735508

open Real

variables (O H P Q C D : Point) 
variables (R d a b : ℝ) 
variables (l : Line)
variables (A B : Fin 2008 → Point)

axiom circle_O (x y : ℝ) : circle O R
axiom line_l (x y : ℝ) : line l
axiom H_projection (h_pos : O.projection_on l = H) 
axiom OH_distance (h_dist : dist O H = d)
axiom CD_intersects_circle (h_inter : seg OH ∩ circle O R = {C, D})
axiom CH_greater_DH (h_disc : dist C H > dist D H)
axiom points_on_l (h_PQ_on_l : P ∈ l ∧ Q ∈ l)
axiom PH_distance (h_PH : dist P H = a)
axiom QH_distance (h_QH : dist Q H = b)
axiom a_gt_b (h_order : a > b)
axiom chords_parallel (h_chords : ∀ i, parallel (line_through (A i) (B i)) l)
axiom CD_divided_evenly (h_div_even : evenly_divides (segment C D) 2008 (intersection_points))

noncomputable def proof_problem : ℝ :=
  (1 / 2008) * Σ i, (dist P (A i))^2 + (dist P (B i))^2 + 
  (dist Q (A i))^2 + (dist Q (B i))

theorem geometric_problem_solution : 
  proof_problem O H P Q C D R d a b l A B = 2 * a^2 + 2 * b^2 + 4 * R^2 + 4 * d^2 := by
  sorry

end geometric_problem_solution_l735_735508


namespace logarithm_identity_l735_735072

noncomputable section

open Real

theorem logarithm_identity : 
  log 10 = (log (sqrt 5) / log 10 + (1 / 2) * log 20) :=
sorry

end logarithm_identity_l735_735072


namespace necessary_condition_l735_735974

theorem necessary_condition (a b : ℝ) (h : b ≠ 0) (h2 : a > b) (h3 : b > 0) : (1 / a < 1 / b) :=
sorry

end necessary_condition_l735_735974


namespace total_area_correct_l735_735444

def section_areas : list ℝ := [0.5, 3, 16, 2, 1, 2.5, 3, 0.5, 1.5, 12, 3, 2, 0.5, 3, 0.5, 1, 2, 1.5, 0.5]

theorem total_area_correct :
  ∑ i in section_areas, i = 56 :=
by
  -- Proof omitted here
  sorry

end total_area_correct_l735_735444


namespace determine_cube_edge_length_l735_735707

noncomputable def original_cube_edge_length (n : ℕ) (painted_faces_cube_three_painted_faces_ratio : ℕ → ℕ) (painted_faces_cube_two_painted_faces_ratio : ℕ → ℕ →  ℕ → ℕ): Prop :=
    let three_painted_faces := 8 in
    let two_painted_faces := 12 * (n - 2) in
    (painted_faces_cube_three_painted_faces_ratio 10 15 ≠ 12).1 n ∧  (painted_faces_cube_two_painted_faces_ratio 2 15 three_painted_faces = 15).2 n ∧ n = 12

axiom is_lukas_claim_incorrect : ∀ {n : ℕ}, (10 * 8 ≠ 12 * (n - 2))
axiom is_martina_claim_correct : ∀ {n : ℕ}, (15 * 8 = 12 * (n - 2)) → n = 12

theorem determine_cube_edge_length : ∃ n : ℕ, (original_cube_edge_length n is_lukas_claim_incorrect is_martina_claim_correct)
by
    use 12
    split
    exact is_lukas_claim_incorrect
    exact (is_martina_claim_correct).2
    rfl

end determine_cube_edge_length_l735_735707


namespace sum_of_consecutive_even_numbers_l735_735698

theorem sum_of_consecutive_even_numbers :
  ∃ n : ℤ, 
  let a := n, b := n + 2, c := n + 4, d := n + 6 in
  a^2 + b^2 + c^2 + d^2 = 344 ∧ a + b + c + d = 36 :=
begin
  -- We need to fill this in with the proof, but it's omitted here.
  sorry
end

end sum_of_consecutive_even_numbers_l735_735698


namespace total_bouncy_balls_l735_735642

-- Definitions of the given quantities
def r : ℕ := 4 -- number of red packs
def y : ℕ := 8 -- number of yellow packs
def g : ℕ := 4 -- number of green packs
def n : ℕ := 10 -- number of balls per pack

-- Proof statement to show the correct total number of balls
theorem total_bouncy_balls : r * n + y * n + g * n = 160 := by
  sorry

end total_bouncy_balls_l735_735642


namespace find_discriminant_l735_735158

variables {a b c : ℝ}
variables (P : ℝ → ℝ)
def is_quadratic_polynomial (P : ℝ → ℝ) : Prop := ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, P x = a * x^2 + b * x + c)

theorem find_discriminant (h1 : is_quadratic_polynomial P)
  (h2 : ∃ x, P x = x - 2)
  (h3 : ∃ y, P y = 1 - y / 2)
  : ∃ D, D = -1/2 := 
sorry

end find_discriminant_l735_735158


namespace find_n_in_given_set_l735_735908

theorem find_n_in_given_set (a b : ℤ) (h_a : a ≡ 16 [MOD 44]) (h_b : b ≡ 77 [MOD 44]) :
  ∃ n ∈ ({120, 121, 122, ..., 160, 161} : Set ℤ), (a - b) ≡ n [MOD 44] :=
by 
  have h_b' : b ≡ 33 [MOD 44] := by sorry
  have h_ab : (a - b) ≡ -17 [MOD 44] := by sorry
  have h_pos : -17 ≡ 27 [MOD 44] := by sorry
  have h_n : 27 + 44 * 3 = 159 := rfl
  use 159
  sorry

end find_n_in_given_set_l735_735908


namespace analysis_method_sufficient_conditions_l735_735442

theorem analysis_method_sufficient_conditions (P : Prop) (analysis_method : ∀ (Q : Prop), (Q → P) → Q) :
  ∀ Q, (Q → P) → Q :=
by
  -- Proof is skipped
  sorry

end analysis_method_sufficient_conditions_l735_735442


namespace exists_c_divisible_by_2009_l735_735254

theorem exists_c_divisible_by_2009 (f : ℤ[X]) (a b : ℤ) (h1 : f.eval a = 41) (h2 : f.eval b = 49) : 
  ∃ c : ℤ, 2009 ∣ f.eval c := 
by
  sorry

end exists_c_divisible_by_2009_l735_735254


namespace polynomial_discriminant_l735_735128

theorem polynomial_discriminant (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
by
  sorry

end polynomial_discriminant_l735_735128


namespace repayment_ways_l735_735766

theorem repayment_ways :
  ∃ ways : ℕ, 
    ways = 4 ∧
    (∀ seq, (seq.length > 0) →
            (seq.sum = 50) →
            (∀ i, i < seq.length - 1 → 
                   (seq.get i = 10 ∧ seq.get (i + 1) = 20) ∨
                   (seq.get i = 20 ∧ seq.get (i + 1) = 10) ) →
            (ways = 4)) := sorry

end repayment_ways_l735_735766


namespace phone_answered_within_two_rings_l735_735340

def probability_of_first_ring : ℝ := 0.5
def probability_of_second_ring : ℝ := 0.3
def probability_of_within_two_rings : ℝ := 0.8

theorem phone_answered_within_two_rings :
  probability_of_first_ring + probability_of_second_ring = probability_of_within_two_rings :=
by
  sorry

end phone_answered_within_two_rings_l735_735340


namespace length_of_other_side_l735_735248

-- Defining the conditions
def roofs := 3
def sides_per_roof := 2
def length_of_one_side := 40 -- measured in feet
def shingles_per_square_foot := 8
def total_shingles := 38400

-- The proof statement
theorem length_of_other_side : 
    ∃ (L : ℕ), (total_shingles / shingles_per_square_foot / roofs / sides_per_roof = 40 * L) ∧ L = 20 :=
by
  sorry

end length_of_other_side_l735_735248


namespace sum_of_solutions_l735_735390

theorem sum_of_solutions :
  (∑ x in {x : ℝ | x = abs (3 * x - abs (45 - 3 * x))}.to_finset) = 423 / 7 :=
by
  sorry

end sum_of_solutions_l735_735390


namespace additional_savings_is_300_l735_735431

-- Define constants
def price_per_window : ℕ := 120
def discount_threshold : ℕ := 10
def discount_per_window : ℕ := 10
def free_window_threshold : ℕ := 5

-- Define the number of windows Alice needs
def alice_windows : ℕ := 9

-- Define the number of windows Bob needs
def bob_windows : ℕ := 12

-- Define the function to calculate total cost without discount
def cost_without_discount (n : ℕ) : ℕ := n * price_per_window

-- Define the function to calculate cost with discount
def cost_with_discount (n : ℕ) : ℕ :=
  let full_windows := n - n / free_window_threshold
  let discounted_price := if n > discount_threshold then price_per_window - discount_per_window else price_per_window
  full_windows * discounted_price

-- Define the function to calculate savings when windows are bought separately
def savings_separately : ℕ :=
  (cost_without_discount alice_windows + cost_without_discount bob_windows) 
  - (cost_with_discount alice_windows + cost_with_discount bob_windows)

-- Define the function to calculate savings when windows are bought together
def savings_together : ℕ :=
  let combined_windows := alice_windows + bob_windows
  cost_without_discount combined_windows - cost_with_discount combined_windows

-- Prove that the additional savings when buying together is $300
theorem additional_savings_is_300 : savings_together - savings_separately = 300 := by
  -- missing proof
  sorry

end additional_savings_is_300_l735_735431


namespace find_m_plus_n_l735_735819

theorem find_m_plus_n :
  ∃ (m n : ℤ), ¬ ∃ (p : ℕ), Prime p ∧ p^2 ∣ m ∧
  let Δ := (11^2 - 4 * 2 * 5) in
  Δ = 81 ∧
  let r1 := (11 + 9) / 4, r2 := (11 - 9) / 4 in
  (r1 - r2) = 4.5 ∧ (4.5 = Real.sqrt m / n) ∧
  m + n = 3 :=
by
  sorry

end find_m_plus_n_l735_735819


namespace positive_integer_sum_l735_735491

theorem positive_integer_sum (n : ℤ) :
  (n > 0) ∧
  (∀ stamps_cannot_form : ℤ, (∀ a b c : ℤ, 7 * a + n * b + (n + 2) * c ≠ 120) ↔
  ¬ (0 ≤ 7*a ∧ 7*a ≤ 120 ∧ 0 ≤ n*b ∧ n*b ≤ 120 ∧ 0 ≤ (n + 2)*c ∧ (n + 2)*c ≤ 120)) ∧
  (∀ postage_formed : ℤ, (120 < postage_formed ∧ postage_formed ≤ 125 →
  ∃ a b c : ℤ, 7 * a + n * b + (n + 2) * c = postage_formed)) →
  n = 21 :=
by {
  -- proof omittted 
  sorry
}

end positive_integer_sum_l735_735491


namespace find_m_value_l735_735981

noncomputable def geometric_seq_sum := fun (a₁ r : ℝ) (n : ℕ) =>
  a₁ * (1 - r^(n + 1)) / (1 - r)

theorem find_m_value (Sm Sm₁ Sm₋₁ : ℕ → ℝ) : 
  (∀ m : ℕ, Sm (m - 1) = 5 → Sm m = -11 → Sm (m + 1) = 21 → m = 5) := 
by
  intros m hSm₋₁ hSm hSm₁
  sorry

end find_m_value_l735_735981


namespace median_of_list_l735_735602

def list_spec : List ℕ := List.join (List.map (λ n => List.repeat n n) (List.range' 1 200))

theorem median_of_list : (list_spec.nth 10049).iget = 141 ∧ (list_spec.nth 10050).iget = 141 :=
by
  sorry

end median_of_list_l735_735602


namespace solve_for_x_l735_735903

variable (a b c x y z : ℝ)
variable (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

theorem solve_for_x (h1 : (x * y) / (x + y) = a)
                   (h2 : (x * z) / (x + z) = b)
                   (h3 : (y * z) / (y + z) = c) :
                   x = (2 * a * b * c) / (a * c + b * c - a * b) :=
by 
  sorry

end solve_for_x_l735_735903


namespace quadratic_discriminant_l735_735165

variable {a b c : ℝ}
variable (h₁ : a ≠ 0)
variable (h₂ : (b - 1)^2 - 4 * a * (c + 2) = 0)
variable (h₃ : (b + 1/2)^2 - 4 * a * (c - 1) = 0)

theorem quadratic_discriminant : b^2 - 4 * a * c = -1 / 2 := 
by
  have h₁' : (b - 1)^2 - 4 * a * (c + 2) = 0 := h₂
  have h₂' : (b + 1/2)^2 - 4 * a * (c - 1) = 0 := h₃
  sorry

end quadratic_discriminant_l735_735165


namespace lowest_probability_red_side_up_l735_735358

def card_flip_probability (k : ℕ) (n : ℕ) : ℚ :=
  if k ≤ n/2 then (n-k)^2/(n^2) + k^2/(n^2)
  else card_flip_probability (n+1-k) n 

theorem lowest_probability_red_side_up :
  (card_flip_probability 13 50) = (card_flip_probability 38 50) ∧ 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 25 → (card_flip_probability k 50 ≥ card_flip_probability 13 50)) :=
begin
  sorry
end

end lowest_probability_red_side_up_l735_735358


namespace trajectory_of_other_focus_is_hyperbola_l735_735178

structure Point :=
(x: ℝ)
(y: ℝ)

def A : Point := { x := -7, y := 0 }
def B : Point := { x := 7, y := 0 }
def C : Point := { x := 2, y := -12 }

theorem trajectory_of_other_focus_is_hyperbola :
  ∃ F : Point, (trajectory of point F is part of a hyperbola) :=
sorry

end trajectory_of_other_focus_is_hyperbola_l735_735178


namespace bridget_bought_17_apples_l735_735793

noncomputable def total_apples (x : ℕ) : Prop :=
  (2 * x / 3) - 5 = 6

theorem bridget_bought_17_apples : ∃ x : ℕ, total_apples x ∧ x = 17 :=
  sorry

end bridget_bought_17_apples_l735_735793


namespace conditional_probability_l735_735114

-- Definitions of events A and B given the conditions
def total_bills : ℕ := 20
def counterfeit_bills : ℕ := 5

def event_A := "at least one of the two selected bills is counterfeit"
def event_B := "both selected bills are counterfeit"

-- We need to prove P(B|A) = 2/17 under these conditions
theorem conditional_probability (P : (event_A → Prop) → (event_B → Prop) → ℚ) :
  P event_B given event_A = 2 / 17 := 
sorry

end conditional_probability_l735_735114


namespace range_of_a_l735_735303

open Real

noncomputable def has_no_real_roots (a : ℝ) : Prop :=
  a^2 - 8 < 0

noncomputable def is_monotonically_increasing (a : ℝ) : Prop :=
  a > 1

noncomputable def p : Prop := ∃ a : ℝ, has_no_real_roots a
noncomputable def q : Prop := ∃ a : ℝ, is_monotonically_increasing a

theorem range_of_a (a : ℝ) :
  ¬ (p ∧ q) ∧ (p ∨ q) ↔ a ∈ Ioc (-sqrt 8) 1 ∪ Ici (sqrt 8) :=
sorry

end range_of_a_l735_735303


namespace sum_A_H_l735_735598

variable (A B C D E F G H : ℤ)

-- Given conditions
def condition1 : C = 5 := sorry
def condition2 : (∀ i, A + B + C = 30 ∧ B + C + D = 30 ∧ C + D + E = 30 ∧ D + E + F = 30 ∧ E + F + G = 30 ∧ F + G + H = 30) := sorry

-- Proof statement
theorem sum_A_H : C = 5 → (∀ i, A + B + C = 30 ∧ B + C + D = 30 ∧ C + D + E = 30 ∧ D + E + F = 30 ∧ E + F + G = 30∧ F + G + H = 30) → A + H = 25 :=
by
  intros
  sorry

end sum_A_H_l735_735598


namespace vodka_shot_size_l735_735949

theorem vodka_shot_size (x : ℝ) (h1 : 8 / 2 = 4) (h2 : 4 * x = 2 * 3) : x = 1.5 :=
by
  sorry

end vodka_shot_size_l735_735949


namespace triangle_ABC_right_angle_l735_735860

def point := (ℝ × ℝ)
def line (P: point) := P.1 = 5 ∨ ∃ a: ℝ, P.1 - 5 = a * (P.2 + 2)
def parabola (P: point) := P.2 ^ 2 = 4 * P.1
def perpendicular_slopes (k1 k2: ℝ) := k1 * k2 = -1

theorem triangle_ABC_right_angle (A B C: point) (P: point) 
  (hA: A = (1, 2))
  (hP: P = (5, -2))
  (h_line: line B ∧ line C)
  (h_parabola: parabola B ∧ parabola C):
  (∃ k_AB k_AC: ℝ, perpendicular_slopes k_AB k_AC) →
  ∃k_AB k_AC: ℝ, k_AB * k_AC = -1 :=
by sorry

end triangle_ABC_right_angle_l735_735860


namespace distance_from_circle_center_to_line_l735_735604

-- Definitions
def circle_polar (rho theta : ℝ) : Prop := rho = -2 * Real.cos theta
def line_polar (rho theta : ℝ) : Prop := 2 * rho * Real.cos theta + rho * Real.sin theta - 2 = 0

-- Cartesian coordinates for the circle's center
def circle_center : ℝ × ℝ := (-1, 0)

-- Cartesian form of the line
def line_cartesian (x y : ℝ) : Prop := 2 * x + y - 2 = 0

-- Distance from point to line
def distance_point_to_line (C : ℝ × ℝ) (A B C0 : ℝ) : ℝ :=
(abs (A * C.1 + B * C.2 + C0) / (Real.sqrt (A^2 + B^2)))

-- Problem statement in Lean
theorem distance_from_circle_center_to_line : 
  circle_center = (-1, 0) → 
  line_cartesian 2 1 2 →
  distance_point_to_line circle_center 2 1 (-2) = 4 * Real.sqrt 5 / 5 :=
by 
  intro h_center h_line
  sorry

end distance_from_circle_center_to_line_l735_735604


namespace length_AB_intercepted_by_parabola_l735_735879

def parabola : Set (ℝ × ℝ) := 
  {p | ∃ x y : ℝ, p = (x, y) ∧ y^2 = 4 * x}

def line (α t : ℝ) : Set (ℝ × ℝ) := 
  {p | ∃ t : ℝ, p = (t * Real.cos α, 1 + t * Real.sin α)}

theorem length_AB_intercepted_by_parabola (α : ℝ) (hα : 0 ≤ α ∧ α ≤ π) :
  (1,0) ∈ (line α) →
  ∃ (A B : ℝ × ℝ), A ∈ parabola ∧ B ∈ parabola ∧ 
  A ∈ line α ∧ B ∈ line α ∧ 
  ∃ l : ℝ, l = Real.dist A B ∧ l = 8 := sorry

end length_AB_intercepted_by_parabola_l735_735879


namespace parallel_lines_slope_l735_735223

theorem parallel_lines_slope (a : ℝ) : 
  (∀ x y : ℝ, ax + 2 * y + 1 = 0 → ∀ x y : ℝ, x + y - 2 = 0 → True) → 
  a = 2 :=
by
  sorry

end parallel_lines_slope_l735_735223


namespace find_a_l735_735886

noncomputable def f (x a : ℝ) : ℝ := (Real.sin x) ^ 2 + Real.cos x + (5 / 8) * a - 3 / 2

theorem find_a (a : ℝ) : 
  (∀ x ∈ set.Icc 0 (Real.pi / 2), f x a ≥ 2) → a = 4 :=
by
  sorry

end find_a_l735_735886


namespace intersection_of_sets_l735_735203

def set_a : Set ℝ := { x | -x^2 + 2 * x ≥ 0 }
def set_b : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def set_intersection : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem intersection_of_sets : (set_a ∩ set_b) = set_intersection := by 
  sorry

end intersection_of_sets_l735_735203


namespace range_of_t_max_min_of_f_l735_735638

noncomputable def f (x : ℝ) := (Real.log x / Real.log 2)^2 + 3 * (Real.log x / Real.log 2) + 2

theorem range_of_t (x : ℝ) (h : 1/4 ≤ x ∧ x ≤ 4) : 
  -2 ≤ Real.log x / Real.log 2 ∧ Real.log x / Real.log 2 ≤ 2 := 
sorry

theorem max_min_of_f :
  let x₁ := (1/2:ℝ)^(-3/2)
  let f_min := -1/4
  let f_max := 12
  let x₂ := 4
  (∀ x : ℝ, 1/4 ≤ x ∧ x ≤ 4 → f x ≥ f_min) ∧ f x₁ = f_min ∧ 
  (∀ x : ℝ, 1/4 ≤ x ∧ x ≤ 4 → f x ≤ f_max) ∧ f x₂ = f_max := 
sorry

end range_of_t_max_min_of_f_l735_735638


namespace hypotenuse_length_l735_735588

theorem hypotenuse_length (a b c : ℝ) (h_right_angled : c^2 = a^2 + b^2) (h_sum_of_squares : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l735_735588


namespace a_n_is_square_of_rational_inequality_holds_l735_735836

namespace ArithmeticGeometricMeans

noncomputable def A (a b : ℝ) : ℝ := (a + b) / 2
noncomputable def G (a b : ℝ) : ℝ := Real.sqrt (a * b)

def a_seq (n : ℕ) : ℕ → ℝ 
| 0       := 0
| 1       := 1
| (m+2) := A (A (a_seq m) (a_seq (m + 1))) (G (a_seq m) (a_seq (m + 1)))

theorem a_n_is_square_of_rational (n : ℕ) (h₀ : n > 0) : ∃ b_n : ℚ, (0 ≤ b_n) ∧ (a_seq n = b_n ^ 2) :=
sorry

theorem inequality_holds (n : ℕ) (h₀ : n > 0) : abs (a_seq n - 2 / 3) < 1 / 2^n :=
sorry

end ArithmeticGeometricMeans

end a_n_is_square_of_rational_inequality_holds_l735_735836


namespace percentage_of_students_from_A_to_C_l735_735229

variables 
  (total_students : ℕ)
  (school_A_percent : ℕ)
  (school_B_percent : ℕ)
  (school_B_to_C_percent : ℕ)
  (total_to_C_percent : ℕ)
  (school_A_to_C_percent : ℕ)

def students_in_A := total_students * school_A_percent / 100
def students_in_B := total_students * school_B_percent / 100
def expected_students_in_C_from_B := students_in_B * school_B_to_C_percent / 100
def expected_students_in_C := total_students * total_to_C_percent / 100
def expected_students_in_C_from_A := students_in_A * school_A_to_C_percent / 100

axiom conditions : 
  total_students = 100 ∧
  school_A_percent = 60 ∧
  school_B_percent = 40 ∧
  school_B_to_C_percent = 40 ∧
  total_to_C_percent = 34 ∧
  expected_students_in_C = expected_students_in_C_from_A + expected_students_in_C_from_B

theorem percentage_of_students_from_A_to_C : 
  school_A_to_C_percent = 30 :=
by
  sorry

end percentage_of_students_from_A_to_C_l735_735229


namespace smaller_number_of_ratio_4_5_lcm_180_l735_735741

theorem smaller_number_of_ratio_4_5_lcm_180 {a b : ℕ} (h_ratio : 4 * b = 5 * a) (h_lcm : Nat.lcm a b = 180) : a = 144 :=
by
  sorry

end smaller_number_of_ratio_4_5_lcm_180_l735_735741


namespace consecutive_odd_integers_count_l735_735369

theorem consecutive_odd_integers_count (n : ℕ) (h_mean : (∑ i in finset.range n, (313 + 2 * i) / n = 414))
    (h_least : 313 = 313) : n = 102 :=
sorry

end consecutive_odd_integers_count_l735_735369


namespace divisor_count_l735_735268

theorem divisor_count : 
  let n := (2^31) * (3^19) in
  let count_factors_less_than := ((2 * 31 + 1) * (2 * 19 + 1) - 1) / 2 in
  let count_factors_n_less_than := (31 + 1) * (19 + 1) - 1 in
  (count_factors_less_than - count_factors_n_less_than) = 589 :=
by
  let n := (2^31) * (3^19)
  let count_factors_less_than := ((2 * 31 + 1) * (2 * 19 + 1) - 1) / 2
  let count_factors_n_less_than := (31 + 1) * (19 + 1) - 1
  exact sorry

end divisor_count_l735_735268


namespace sum_of_common_ratios_l735_735274

theorem sum_of_common_ratios (k p r : ℝ) (h : k ≠ 0) (h1 : k * p ≠ k * r)
  (h2 : k * p ^ 2 - k * r ^ 2 = 3 * (k * p - k * r)) : p + r = 3 :=
by
  sorry

end sum_of_common_ratios_l735_735274


namespace find_N_product_of_primes_l735_735337

theorem find_N_product_of_primes (N : ℕ) (a b : ℕ) (h1 : prime a) (h2 : prime b) (h3 : N = a * b) (h4 : 1 + a + b = 2014) : N = 4022 :=
sorry

end find_N_product_of_primes_l735_735337


namespace find_deleted_files_l735_735064

def original_files : Nat := 21
def remaining_files : Nat := 7
def deleted_files : Nat := 14

theorem find_deleted_files : original_files - remaining_files = deleted_files := by
  sorry

end find_deleted_files_l735_735064


namespace real_numbers_x_condition_l735_735470

theorem real_numbers_x_condition (x : ℝ) (h : ∀ n : ℤ, (x^n + x^(-n)) ∈ ℤ) :
  ∃ k : ℤ, |k| ≥ 2 ∧ (x = (k + Real.sqrt (k^2 - 4)) / 2 ∨ x = (k - Real.sqrt (k^2 - 4)) / 2) :=
begin
  sorry
end

end real_numbers_x_condition_l735_735470


namespace kind_wizard_goal_achievable_l735_735791

-- Defining the concept of being achievable as a predicate
def achievable (n : ℕ) : Prop :=
  n % 2 = 1 ∧ n > 1 ∧
  ∃ (pairs : set (ℕ × ℕ)),  -- the set of pairs defining friendships
    pairs.card = 2 * n ∧
    (∀ e ∈ pairs, (∃ i, e = (i, i + 1 % n))) ∧  -- condition of immediate neighbors friendship
    (∀ unfriendly_set ⊆ pairs, unfriendly_set.card = n → 
      -- Any subset of n pairs being unfriends should still satisfy the condition
      ∃ friendly_set ⊆ pairs, friendly_set.card = n ∧ 
        ∀ (i : ℕ), friendly_set ∩ ({i, (i + 1) % n} : set (ℕ × ℕ)) ≠ ∅)

-- The theorem statement
theorem kind_wizard_goal_achievable :
  ∀ n : ℕ, n % 2 = 1 → n > 1 → achievable n :=
by
  intros n hn1 hn_gt1
  unfold achievable
  -- We leave the proof as an exercise with 'sorry' to denote incompleteness
  sorry

end kind_wizard_goal_achievable_l735_735791


namespace find_width_of_rectangular_plot_l735_735776

-- Conditions
def length : ℝ := 90
def distance_between_poles : ℝ := 4
def number_of_poles : ℝ := 70

-- Calculation based on the conditions
def total_length_of_fencing : ℝ := (number_of_poles - 1) * distance_between_poles
def perimeter : ℝ := 2 * length + 2 * width

-- Proof statement
theorem find_width_of_rectangular_plot :
  total_length_of_fencing = 276 → width = 48 :=
by
  intro h1
  sorry

end find_width_of_rectangular_plot_l735_735776


namespace quadratic_roots_negative_l735_735712

theorem quadratic_roots_negative (m : ℝ) :
  ∀ (x₁ x₂ : ℝ), 3 * x₁ ^ 2 + 6 * x₁ + m = 0 ∧ 3 * x₂ ^ 2 + 6 * x₂ + m = 0 ∧ x₁ ≠ x₂ ∧ x₁ + x₂ = -2 ∧ 0 < x₁ * x₂ ↔ m = 1 ∨ m = 2 ∨ m = 3 := 
by
  sorry

end quadratic_roots_negative_l735_735712


namespace find_natural_numbers_l735_735084

open Nat

theorem find_natural_numbers (n : ℕ) (h : ∃ m : ℤ, 2^n + 33 = m^2) : n = 4 ∨ n = 8 :=
sorry

end find_natural_numbers_l735_735084


namespace baking_time_correct_l735_735283

/-- Mark lets the bread rise for 120 minutes twice. -/
def rising_time : ℕ := 120 * 2

/-- Mark spends 10 minutes kneading the bread. -/
def kneading_time : ℕ := 10

/-- Total time taken to finish making the bread. -/
def total_time : ℕ := 280

/-- Calculate the baking time based on the given conditions. -/
def baking_time (rising kneading total : ℕ) : ℕ := total - (rising + kneading)

theorem baking_time_correct :
  baking_time rising_time kneading_time total_time = 30 := 
by 
  -- Proof is omitted
  sorry

end baking_time_correct_l735_735283


namespace minimal_loss_arithmetic_progression_l735_735094

theorem minimal_loss_arithmetic_progression :
  ∃ (P A C X O Д : ℕ), 
    (A = P + 1) ∧
    (C = A + 1) ∧
    (X = C + 1) ∧
    (O = X + 1) ∧
    (Д = O + 1) ∧
    (P = 1) ∧
    (EXPENSE = 100000 * P + 10000 * A + 1000 * C + 100 * X + 10 * O + Д) ∧
    (INCOME = 100000 * A + 10000 * C + 1000 * X + 100 * O + 10 * Д + P) ∧
    (LOSS = EXPENSE - INCOME) ∧ 
    (EXPENSE - INCOME = 58000) := sorry

end minimal_loss_arithmetic_progression_l735_735094


namespace vertical_distance_to_max_l735_735587

-- Definitions based on the conditions
def Rachel_coords : ℝ × ℝ := (4, -15)
def Sarah_coords : ℝ × ℝ := (-2, 12)
def Max_coords : ℝ × ℝ := (1, -5)

-- Defining the midpoint calculation function
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, 
   (p1.2 + p2.2) / 2)

-- Calculating the vertical distance between two points
def vertical_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2).abs

-- Theorem statement translating to: given the coordinates, prove that the vertical distance between the midpoint and Max is 3.5
theorem vertical_distance_to_max :
  let M := midpoint Rachel_coords Sarah_coords in
  vertical_distance M Max_coords = 3.5 := by
  let M := midpoint Rachel_coords Sarah_coords
  show vertical_distance M Max_coords = 3.5
  sorry

end vertical_distance_to_max_l735_735587


namespace twenty_fifth_number_in_base_six_l735_735934

def decimal_to_base_six (n : ℕ) : ℕ := 
let r1 := n % 6 in
let q1 := n / 6 in
let r2 := q1 % 6 in
let q2 := q1 / 6 in
r2 * 10 + r1

theorem twenty_fifth_number_in_base_six : decimal_to_base_six 25 = 41 :=
by
  sorry

end twenty_fifth_number_in_base_six_l735_735934


namespace ratio_BD_BE_areas_of_BDE_and_BDF_l735_735375

variables {A B C D E F : Point}
variables {k l : Circle}

-- Given Conditions
variables (circ_intersect : CircleIntersect k l) (A_on_BC : A ∈ LineSegment B C)
variables (from_A : LineThrough A D ∧ LineThrough A E)
variables (tangent_BD : TangentToCircle B D k) (tangent_BE : TangentToCircle B E l)
variables (AD_len : dist A D = 8) (AE_len : dist A E = 2)
variables (CD_extends_BF : CD_ExtensionIntersectsCircle D F l)

-- Part (a): To find the ratio BD : BE
theorem ratio_BD_BE : dist B D / dist B E = 2 :=
sorry

-- Part (b): To compare areas of triangles BDE and BDF
theorem areas_of_BDE_and_BDF : TriangleArea B D E = TriangleArea B D F :=
sorry

end ratio_BD_BE_areas_of_BDE_and_BDF_l735_735375


namespace cone_shape_in_spherical_coordinates_l735_735107

-- Define the conditions as given in the problem
def spherical_coordinates (rho theta phi c : ℝ) : Prop := 
  rho = c * Real.sin phi

-- Define the main statement to prove
theorem cone_shape_in_spherical_coordinates (rho theta phi c : ℝ) (hpos : 0 < c) :
  spherical_coordinates rho theta phi c → 
  ∃ cone : Prop, cone :=
sorry

end cone_shape_in_spherical_coordinates_l735_735107


namespace cape_may_shark_sightings_l735_735455

def total_shark_sightings (D C : ℕ) : Prop :=
  D + C = 40

def cape_may_sightings (D C : ℕ) : Prop :=
  C = 2 * D - 8

theorem cape_may_shark_sightings : 
  ∃ (C D : ℕ), total_shark_sightings D C ∧ cape_may_sightings D C ∧ C = 24 :=
by
  sorry

end cape_may_shark_sightings_l735_735455


namespace problem_l735_735496

noncomputable
def differentiable_on_R (f : ℝ → ℝ) := differentiable ℝ f

theorem problem (f : ℝ → ℝ) (h_f_diff : differentiable_on_R f) 
  (h_f' : ∀ x, x * (deriv f x) ≥ 0) : f (-1) + f 1 ≥ 2 * f 0 :=
by sorry

end problem_l735_735496


namespace find_expression_value_l735_735028

def g (p q r s : ℝ) (x : ℝ) : ℝ := p * x^3 + q * x^2 + r * x + s

theorem find_expression_value (p q r s : ℝ) (h1 : g p q r s (-1) = 2) (h2 : g p q r s (-2) = -1) (h3 : g p q r s (1) = -2) :
  9 * p - 3 * q + 3 * r - s = -2 :=
by
  sorry

end find_expression_value_l735_735028


namespace prob_hat_given_sunglasses_l735_735649

theorem prob_hat_given_sunglasses :
  let total_sunglasses := 60
  let total_hats := 45
  let prob_hat_given_sunglasses := 3/5
  let both := (3/5) * 45
  show (both / total_sunglasses) = (9 / 20) :=
begin
  -- Sorry is used here to skip the actual proof.
  sorry
end

end prob_hat_given_sunglasses_l735_735649


namespace johns_payment_l735_735955

def camera_value : ℕ := 5000
def rental_fee_perc : ℕ := 10
def rental_duration : ℕ := 4
def friend_contrib_perc : ℕ := 40

theorem johns_payment :
  let total_rental_fee := (rental_fee_perc * camera_value / 100) * rental_duration,
      friend_payment := (friend_contrib_perc * total_rental_fee / 100),
      johns_payment := total_rental_fee - friend_payment
  in johns_payment = 1200 :=
by
  sorry

end johns_payment_l735_735955


namespace opposite_of_cube_root_neg27_l735_735683

theorem opposite_of_cube_root_neg27 : - ( -3 : ℤ ) = 3 := 
by {
  have root_eq : ( -3 : ℤ )^3 = -27, from by norm_num,
  have eq_root : ∃ x : ℤ, x^3 = -27 ∧ x = -3, from ⟨-3, by {split; assumption}⟩,
  change - ( -3 ) = 3,
  norm_num
}

end opposite_of_cube_root_neg27_l735_735683


namespace polynomial_discriminant_l735_735131

theorem polynomial_discriminant (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
by
  sorry

end polynomial_discriminant_l735_735131


namespace lcm_of_coprime_product_l735_735685

variable {a b : ℕ} (h_coprime : Nat.coprime a b) (h_product : a * b = 117)

theorem lcm_of_coprime_product : Nat.lcm a b = 117 := by
  sorry

end lcm_of_coprime_product_l735_735685


namespace hidden_lattice_points_l735_735464

-- Definition of a lattice point being hidden
def is_hidden (O A : ℤ × ℤ) : Prop :=
  ∃ B : ℤ × ℤ, B ≠ O ∧ B ≠ A ∧ B.1 ∈ set.Ioo O.1 A.1 ∧ B.2 ∈ set.Ioo O.2 A.2

-- Main theorem stating that for any positive integer n, 
-- there exists a square of side length n such that all its lattice points are hidden
theorem hidden_lattice_points (n : ℕ) (hn : 0 < n) :
  ∃ a b : ℤ, ∀ i j ∈ finset.range n, gcd (a + i) (b + j) > 1 :=
begin
  sorry
end

end hidden_lattice_points_l735_735464


namespace time_spent_per_piece_l735_735659

-- Conditions
def number_of_chairs : ℕ := 7
def number_of_tables : ℕ := 3
def total_furniture : ℕ := number_of_chairs + number_of_tables
def total_time_spent : ℕ := 40

-- Proof statement
theorem time_spent_per_piece : total_time_spent / total_furniture = 4 :=
by
  -- Proof goes here
  sorry

end time_spent_per_piece_l735_735659


namespace range_of_2a_plus_b_range_of_a_minus_b_range_of_a_div_b_l735_735843

variable {a b : ℝ}

theorem range_of_2a_plus_b (h1 : -6 < a) (h2 : a < 8) (h3 : 2 < b) (h4 : b < 3) : -10 < 2*a + b ∧ 2*a + b < 19 :=
by
  sorry

theorem range_of_a_minus_b (h1 : -6 < a) (h2 : a < 8) (h3 : 2 < b) (h4 : b < 3) : -9 < a - b ∧ a - b < 6 :=
by
  sorry

theorem range_of_a_div_b (h1 : -6 < a) (h2 : a < 8) (h3 : 2 < b) (h4 : b < 3) : -2 < a / b ∧ a / b < 4 :=
by
  sorry

end range_of_2a_plus_b_range_of_a_minus_b_range_of_a_div_b_l735_735843


namespace largest_common_divisor_510_399_l735_735709

theorem largest_common_divisor_510_399 : ∃ d, d ∣ 510 ∧ d ∣ 399 ∧ ∀ e, e ∣ 510 ∧ e ∣ 399 → e ≤ d :=
begin
  use 57,
  split,
  { sorry },  -- placeholder for proof that 57 divides 510
  split,
  { sorry },  -- placeholder for proof that 57 divides 399
  { assume e h,
    sorry }  -- placeholder for proof that any common divisor must be <= 57
end

end largest_common_divisor_510_399_l735_735709


namespace solve_boys_left_l735_735699

--given conditions
variable (boys_initial girls_initial boys_left girls_entered children_end: ℕ)
variable (h_boys_initial : boys_initial = 5)
variable (h_girls_initial : girls_initial = 4)
variable (h_girls_entered : girls_entered = 2)
variable (h_children_end : children_end = 8)

-- Problem definition
def boys_left_proof : Prop :=
  ∃ (B : ℕ), boys_left = B ∧ boys_initial - B + girls_initial + girls_entered = children_end ∧ B = 3

-- The statement to be proven
theorem solve_boys_left : boys_left_proof boys_initial girls_initial boys_left girls_entered children_end := by
  -- Proof will be provided here
  sorry

end solve_boys_left_l735_735699


namespace domain_of_g_l735_735471

def g (x : ℝ) : ℝ := (4 * x + 2) / Real.sqrt (x - 7)

theorem domain_of_g :
  {x : ℝ | x > 7} = set.Ioi 7 :=
by
  sorry

end domain_of_g_l735_735471


namespace no_solution_exists_l735_735394

theorem no_solution_exists (x : ℝ) : 3 * |x + 2| + 2 = 0 → false :=
by
  intro h
  have h1 : 3 * |x + 2| = -2 := by rw [← sub_eq_zero, h]
  have h2 : |x + 2| = -2 / 3 := by rw [← mul_div_assoc, ← div_neg, ← h1]
  have h3 : -2 / 3 < 0 := by norm_num
  have h4 : 0 ≤ |x + 2| := abs_nonneg (x + 2)
  linarith

end no_solution_exists_l735_735394


namespace diorama_time_subtraction_l735_735044

theorem diorama_time_subtraction (P B X : ℕ) (h1 : B = 3 * P - X) (h2 : B = 49) (h3 : P + B = 67) : X = 5 :=
by
  sorry

end diorama_time_subtraction_l735_735044


namespace graph_remains_connected_after_deletions_l735_735014

theorem graph_remains_connected_after_deletions
  (G : SimpleGraph (Fin 1998))
  (h_connected : G.IsConnected)
  (h_degree : ∀ v : Fin 1998, G.degree v = 3)
  (vertex_set : Finset (Fin 1998))
  (h_card_vertex_set : vertex_set.card = 200)
  (h_no_adjacent : ∀ v w ∈ vertex_set, ¬G.Adj v w) :
  G.deleteVertices vertex_set).IsConnected :=
begin
  sorry 
end

end graph_remains_connected_after_deletions_l735_735014


namespace students_brought_both_types_l735_735406

theorem students_brought_both_types :
  (∃ (B : ℕ), 12 + 8 - 2 * B = 10 ∧ B = 5) :=
begin
  sorry
end

end students_brought_both_types_l735_735406


namespace find_b_if_parallel_l735_735829

theorem find_b_if_parallel : 
  ∀ b : ℝ, (∀ x y : ℝ, 3 * y - 3 * b = 9 * x → y = 3 * x + b) ∧
            (∀ x y : ℝ, y - 2 = (b + 9) * x → y = (b + 9) * x + 2) →
            (3 = b + 9) → b = -6 :=
by
  intros b h_eq_slope_1 h_eq_slope_2 h_parallel
  sorry

end find_b_if_parallel_l735_735829


namespace water_usage_eq_13_l735_735421

theorem water_usage_eq_13 (m x : ℝ) (h : 16 * m = 10 * m + (x - 10) * 2 * m) : x = 13 :=
by sorry

end water_usage_eq_13_l735_735421


namespace min_red_up_probability_card_l735_735356

theorem min_red_up_probability_card (cards : Fin 50) :
  (cards = 13) ∨ (cards = 38) ↔
  -- Conditions for Vasya and Asya's actions:
  ∃ (select_vasya : Fin 26 → Fin 50) (select_asya : Fin 26 → Fin 50),
    (∀ k : Fin 50, 
      probability.card_pos (select_vasya k.left ≤ k.to_nat ∧ k.to_nat < k.left + 25 → 
                            select_asya k.right ≤ k.to_nat ∧ k.to_nat < k.right + 25 → 
                            k.val = 13 ∨ k.val = 38))

end min_red_up_probability_card_l735_735356


namespace five_points_concyclic_l735_735000

variable {A B C D E F G H K L : Point}

-- Definition of the convex pentagon and the external triangles formed by the extended sides.
def is_convex_pentagon (A B C D E : Point) : Prop := 
  -- Convexity condition (abstract, replace with a concrete condition if needed)
  sorry

def external_triangle (A B C D E : Point) (F G H K L : Point) : Prop :=
  -- Some abstract geometrical condition defining the external triangles
  sorry

-- Definition that the five circumcircles of the triangles intersect at five points
def circumcircles_intersect_at_five_points (A B C D E F G H K L A' B' C' D' E' : Point) : Prop :=
  -- Some abstract geometrical condition about the intersections
  sorry

-- Main theorem statement
theorem five_points_concyclic
  (A B C D E A' B' C' D' E' : Point)
  (h_convex: is_convex_pentagon A B C D E)
  (h_external_triangles: external_triangle A B C D E F G H K L)
  (h_intersections: circumcircles_intersect_at_five_points A B C D E F G H K L A' B' C' D' E') :
  is_concyclic A' B' C' D' E' :=
sorry

end five_points_concyclic_l735_735000


namespace quadratic_discriminant_l735_735166

variable {a b c : ℝ}
variable (h₁ : a ≠ 0)
variable (h₂ : (b - 1)^2 - 4 * a * (c + 2) = 0)
variable (h₃ : (b + 1/2)^2 - 4 * a * (c - 1) = 0)

theorem quadratic_discriminant : b^2 - 4 * a * c = -1 / 2 := 
by
  have h₁' : (b - 1)^2 - 4 * a * (c + 2) = 0 := h₂
  have h₂' : (b + 1/2)^2 - 4 * a * (c - 1) = 0 := h₃
  sorry

end quadratic_discriminant_l735_735166


namespace eagle_speed_l735_735787

theorem eagle_speed (E : ℕ) 
  (falcon_speed : ℕ := 46)
  (pelican_speed : ℕ := 33)
  (hummingbird_speed : ℕ := 30)
  (total_distance : ℕ := 248)
  (flight_time : ℕ := 2)
  (falcon_distance := falcon_speed * flight_time)
  (pelican_distance := pelican_speed * flight_time)
  (hummingbird_distance := hummingbird_speed * flight_time) :
  2 * E + falcon_distance + pelican_distance + hummingbird_distance = total_distance →
  E = 15 :=
by
  -- Proof will be provided here
  sorry

end eagle_speed_l735_735787


namespace speed_of_stream_l735_735419

-- Definitions of the given conditions
def speed_boat_still_water : ℝ := 22
def distance_downstream : ℝ := 108
def time_downstream : ℝ := 4

-- Calculated effective speed downstream (from the given problem)
def effective_speed_downstream := distance_downstream / time_downstream

-- Statement of the proof problem
theorem speed_of_stream :
  ∃ v : ℝ, speed_boat_still_water + v = effective_speed_downstream ∧ v = 5 :=
by
  sorry

end speed_of_stream_l735_735419


namespace part1_part2_part3_l735_735936

/-- Proof for part (1): If the point P lies on the x-axis, then m = -1. -/
theorem part1 (m : ℝ) (hx : 3 * m + 3 = 0) : m = -1 := 
by {
  sorry
}

/-- Proof for part (2): If point P lies on a line passing through A(-5, 1) and parallel to the y-axis, 
then the coordinates of point P are (-5, -12). -/
theorem part2 (m : ℝ) (hy : 2 * m + 5 = -5) : (2 * m + 5, 3 * m + 3) = (-5, -12) := 
by {
  sorry
}

/-- Proof for part (3): If point P is moved 2 right and 3 up to point M, 
and point M lies in the third quadrant with a distance of 7 from the y-axis, then the coordinates of M are (-7, -15). -/
theorem part3 (m : ℝ) 
  (hc : 2 * m + 7 = -7)
  (config : 3 * m + 6 < 0) : (2 * m + 7, 3 * m + 6) = (-7, -15) := 
by {
  sorry
}

end part1_part2_part3_l735_735936


namespace original_number_of_people_l735_735232

theorem original_number_of_people (x : ℕ) (h1 : x ≠ 0) 
  (h2 : 1 / 3 * (x : ℝ) : ℝ)
  (h3 : 1 / 2 * 2 / 3 * (x : ℝ) : ℝ)
  (h4 : 2 / 3 * (x : ℝ) - 1 / 3 * (x : ℝ) = (18 : ℝ)) : 
  x = 54 := 
by 
  sorry

end original_number_of_people_l735_735232


namespace david_john_work_together_l735_735803

theorem david_john_work_together : 
  let david_rate := 1 / 5
      john_rate := 1 / 9
      combined_rate := david_rate + john_rate
  in combined_rate = 1 / (45 / 14) :=
by
  sorry

end david_john_work_together_l735_735803


namespace average_weight_of_all_children_l735_735670

theorem average_weight_of_all_children 
  (Boys: ℕ) (Girls: ℕ) (Additional: ℕ)
  (avgWeightBoys: ℚ) (avgWeightGirls: ℚ) (avgWeightAdditional: ℚ) :
  Boys = 8 ∧ Girls = 5 ∧ Additional = 3 ∧ 
  avgWeightBoys = 160 ∧ avgWeightGirls = 130 ∧ avgWeightAdditional = 145 →
  ((Boys * avgWeightBoys + Girls * avgWeightGirls + Additional * avgWeightAdditional) / (Boys + Girls + Additional) = 148) :=
by
  intros
  sorry

end average_weight_of_all_children_l735_735670


namespace problem_solution_l735_735497

noncomputable def f1 (x : ℝ) : ℝ := -2 * x + 2 * Real.sqrt 2
noncomputable def f2 (x : ℝ) : ℝ := Real.sin x
noncomputable def f3 (x : ℝ) : ℝ := x + (1 / x)
noncomputable def f4 (x : ℝ) : ℝ := Real.exp x
noncomputable def f5 (x : ℝ) : ℝ := -2 * Real.log x

def has_inverse_proportion_point (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∃ x ∈ domain, x * f x = 1

theorem problem_solution :
  (has_inverse_proportion_point f1 univ) ∧
  (has_inverse_proportion_point f2 (Set.Icc 0 (2 * Real.pi))) ∧
  ¬ (has_inverse_proportion_point f3 (Set.Ioi 0)) ∧
  (has_inverse_proportion_point f4 univ) ∧
  ¬ (has_inverse_proportion_point f5 (Set.Ioi 0)) :=
by
  sorry

end problem_solution_l735_735497


namespace part_I_part_II_l735_735505

universe u

variable (U : Type u) [LinearOrder U] {x a : U}

def A := {x : U | 1 ≤ x ∧ x ≤ 4}
def B (a : U) := {x : U | a ≤ x ∧ x ≤ a + 2}
def not_U (A : set U) := {x : U | ¬ A x}

-- Part I
theorem part_I (a : U) (ha : a = 3) :
  (A U ∪ B U a = {x : U | 1 ≤ x ∧ x ≤ 5}) ∧
  (B U a ∩ not_U U (A U) = {x : U | 4 < x ∧ x ≤ 5}) :=
  sorry

-- Part II
theorem part_II (h : B U a ⊆ A U) :
  1 ≤ a ∧ a ≤ 2 :=
  sorry

end part_I_part_II_l735_735505


namespace inequality_solution_system_l735_735692

theorem inequality_solution_system {x : ℝ} :
  (2 * x - 1 > 5) ∧ (-x < -6) ↔ (x > 6) :=
by
  sorry

end inequality_solution_system_l735_735692


namespace prime_mod_30_l735_735656

theorem prime_mod_30 (a : ℕ) (ha : Nat.Prime a) (q r : ℕ) 
  (h : a = 30 * q + r) (hr : 0 ≤ r ∧ r < 30) : 
  r ∈ {1, 7, 11, 13, 17, 19, 23, 29} :=
sorry

end prime_mod_30_l735_735656


namespace binomial_distrib_not_equiv_binom_expansion_l735_735319

theorem binomial_distrib_not_equiv_binom_expansion (a b : ℝ) (n : ℕ) (p : ℝ) (h1: a = p) (h2: b = 1 - p):
    ¬ (∃ k : ℕ, p ^ k * (1 - p) ^ (n - k) = (a + b) ^ n) := sorry

end binomial_distrib_not_equiv_binom_expansion_l735_735319


namespace kyle_car_payment_l735_735252

theorem kyle_car_payment (income rent utilities retirement groceries insurance miscellaneous gas x : ℕ)
  (h_income : income = 3200)
  (h_rent : rent = 1250)
  (h_utilities : utilities = 150)
  (h_retirement : retirement = 400)
  (h_groceries : groceries = 300)
  (h_insurance : insurance = 200)
  (h_miscellaneous : miscellaneous = 200)
  (h_gas : gas = 350)
  (h_expenses : rent + utilities + retirement + groceries + insurance + miscellaneous + gas + x = income) :
  x = 350 :=
by sorry

end kyle_car_payment_l735_735252


namespace area_ADE_eq_area_BCE_l735_735971

-- Define the given properties and assumptions
def is_trapezoid (A B C D : Point) : Prop :=
  ∃ (l₁ l₂ : Line), (A ∈ l₁) ∧ (B ∈ l₁) ∧ (C ∈ l₂) ∧ (D ∈ l₂) ∧ parallel l₁ l₂

def intersection_point (A C B D : Point) : Point :=
  let l₁ := line_through A C in
  let l₂ := line_through B D in
  intersection l₁ l₂ sorry -- assumes the intersection exists

-- Define the area function for triangles
noncomputable def area (X Y Z : Point) : ℝ := sorry

-- Prove the problem statement
theorem area_ADE_eq_area_BCE
  (A B C D E : Point)
  (h₁ : is_trapezoid A B C D)
  (h₂ : E = intersection_point A C B D) :
  area A D E = area B C E := sorry

end area_ADE_eq_area_BCE_l735_735971


namespace part1_am_eq_ln_am1_minus_1_part2_am_le_am1_minus_2_part3_k_is_3_l735_735847

noncomputable def f (x : ℝ) := Real.log x
noncomputable def deriv_f (x : ℝ) := 1 / x

theorem part1_am_eq_ln_am1_minus_1 (a_n : ℕ → ℝ) (m : ℕ) (h : m ≥ 2) :
  a_n m = Real.log (a_n (m - 1)) - 1 :=
sorry

theorem part2_am_le_am1_minus_2 (a_n : ℕ → ℝ) (m : ℕ) (h : m ≥ 2) :
  a_n m ≤ a_n (m - 1) - 2 :=
sorry

theorem part3_k_is_3 (a_n : ℕ → ℝ) :
  ∃ k : ℕ, k = 3 ∧ ∀ n : ℕ, n ≤ k → (a_n n) - (a_n (n - 1)) = (a_n 2) - (a_n 1) :=
sorry

end part1_am_eq_ln_am1_minus_1_part2_am_le_am1_minus_2_part3_k_is_3_l735_735847


namespace bug_stops_eventually_l735_735008

def point_on_curve (x y : ℝ) : Prop := x^2 + y^2 + x * y = 6

def bug_moves_parallel_axes (start: ℝ × ℝ) : ℕ → ℝ × ℝ
| 0     := start
| (n+1) := let (x, y) := bug_moves_parallel_axes n in
            if x^2 + y^2 + x*y = 6 
            then if x = 0 
                 then (x, -x - y)
                 else if y = 0 
                 then (-x - y, y)
                 else if n % 2 = 0 then (x, -x - y) else (-x - y, y)
            else (x, y)

def bug_stopped (position: ℝ × ℝ) : Prop :=
  ∃ n, bug_moves_parallel_axes start n = position ∧
       (∀ m, m < n → bug_moves_parallel_axes start m ≠ position)

theorem bug_stops_eventually (start: ℝ × ℝ) (h: point_on_curve start.1 start.2):
  ∃ n, bug_stopped (bug_moves_parallel_axes start n) :=
by
  sorry

end bug_stops_eventually_l735_735008


namespace binomial_identity_l735_735411

open BigOperators

theorem binomial_identity :
  (∑ k in ({1, 3, 5} : Finset ℕ), if even k then 0 else 2 * binom 10 k) - binom 10 3 + (binom 10 5) / 2 = 2^4 := 
by
  sorry

end binomial_identity_l735_735411


namespace gcd_sum_l735_735790

theorem gcd_sum (n : ℕ) (h : 0 < n) : ∑ k in {d | d = gcd (5*n + 6) n}, d = 12 :=
by
  sorry

end gcd_sum_l735_735790


namespace unit_vector_opposite_correct_l735_735861

def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, -1)

def vectorAB (A B : ℝ × ℝ) : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)
def unitVectorOpposite (v : ℝ × ℝ) : ℝ × ℝ := 
  let mag := magnitude v
  (- v.1 / mag, - v.2 / mag)

theorem unit_vector_opposite_correct :
  unitVectorOpposite (vectorAB A B) = (- 3 / 5, 4 / 5) :=
by 
  sorry

end unit_vector_opposite_correct_l735_735861


namespace find_real_m_such_that_z_is_pure_imaginary_l735_735192

theorem find_real_m_such_that_z_is_pure_imaginary (m : ℝ) :
  let z := (1 + complex.i) / (1 - complex.i) + m * (1 - complex.i) * complex.i in
  complex.re z = 0 → m = 0 := by
  intro h
  sorry

end find_real_m_such_that_z_is_pure_imaginary_l735_735192


namespace impossible_projection_of_rectangle_l735_735722

theorem impossible_projection_of_rectangle (A B C D : Type) (cardboard_projection : A → B → C → D) 
  (line_segment_projections : ∃ p : A, cardboard_projection p = B) 
  (rectangle_projections : ∃ q : A, cardboard_projection q = C)
  (parallelogram_projections : ∃ r : A, cardboard_projection r = D) : 
  ∀ t : A, cardboard_projection t ≠ Triangle :=
by sorry

end impossible_projection_of_rectangle_l735_735722


namespace find_minimum_value_M_l735_735823

theorem find_minimum_value_M : (∃ (M : ℝ), (∀ (x : ℝ), -x^2 + 2 * x ≤ M) ∧ M = 1) := 
sorry

end find_minimum_value_M_l735_735823


namespace minute_first_catch_hour_l735_735784

theorem minute_first_catch_hour :
  ∃ (t : ℚ), t = 60 * (1 + (5 / 11)) :=
sorry

end minute_first_catch_hour_l735_735784


namespace maximal_overlap_area_l735_735703

theorem maximal_overlap_area : 
  ∀ (T1 T2 : Triangle) (x : ℝ), 
  (is_right_angled_isosceles T1 ∧ T1.base_length = 1) ∧ 
  (is_right_angled_isosceles T2 ∧ T2.base_length = 1) ∧ 
  slides_along_line T1 T2 x → 
  overlaps_area_max T1 T2 = 2 / 9 :=
by
  sorry

end maximal_overlap_area_l735_735703


namespace find_b_for_smallest_c_l735_735828

theorem find_b_for_smallest_c (c b : ℝ) (h_c_pos : 0 < c) (h_b_pos : 0 < b)
  (polynomial_condition : ∀ x : ℝ, (x^4 - c*x^3 + b*x^2 - c*x + 1 = 0) → real) :
  c = 4 → b = 6 :=
by
  intros h_c_eq_4
  sorry

end find_b_for_smallest_c_l735_735828


namespace midpoints_of_triangle_sides_l735_735300

open Point Distance
open segment

variables {A B C P M K : Point}
variables {AM BK CP : segment}

-- Define the conditions
def points_on_sides : Prop := P ∈ AB ∧ M ∈ BC ∧ K ∈ AC
def segments_intersect : Prop := intersects_at_one_point AM BK CP
def vector_sum_zero : Prop := (vector AM + vector BK + vector CP = vector.zero)

-- Define the final proof statement
theorem midpoints_of_triangle_sides (h1 : points_on_sides) 
                                    (h2 : segments_intersect)
                                    (h3 : vector_sum_zero) 
                                    : (is_midpoint P AB ∧ is_midpoint M BC ∧ is_midpoint K AC) := 
sorry -- proof is omitted

end midpoints_of_triangle_sides_l735_735300


namespace maximum_function_value_l735_735678

noncomputable def y (x : ℝ) : ℝ := x^2 * (1 - 3 * x)

theorem maximum_function_value : ∃ x (h : 0 < x ∧ x < 1/3), y x = 1/12 :=
by
  sorry

end maximum_function_value_l735_735678


namespace tin_can_allocation_l735_735377

-- Define the total number of sheets of tinplate available
def total_sheets := 108

-- Define the number of sheets used for can bodies
variable (x : ℕ)

-- Define the number of can bodies a single sheet makes
def can_bodies_per_sheet := 15

-- Define the number of can bottoms a single sheet makes
def can_bottoms_per_sheet := 42

-- Define the equation to be proven
theorem tin_can_allocation :
  2 * can_bodies_per_sheet * x = can_bottoms_per_sheet * (total_sheets - x) :=
  sorry

end tin_can_allocation_l735_735377


namespace zhuzhuxia_defeats_monsters_l735_735435

theorem zhuzhuxia_defeats_monsters {a : ℕ} (H1 : zhuzhuxia_total_defeated_monsters = 20) :
  zhuzhuxia_total_defeated_by_monsters = 8 :=
sorry

end zhuzhuxia_defeats_monsters_l735_735435


namespace johns_payment_l735_735956

def camera_value : ℕ := 5000
def rental_fee_perc : ℕ := 10
def rental_duration : ℕ := 4
def friend_contrib_perc : ℕ := 40

theorem johns_payment :
  let total_rental_fee := (rental_fee_perc * camera_value / 100) * rental_duration,
      friend_payment := (friend_contrib_perc * total_rental_fee / 100),
      johns_payment := total_rental_fee - friend_payment
  in johns_payment = 1200 :=
by
  sorry

end johns_payment_l735_735956


namespace solve_for_x_l735_735544

theorem solve_for_x (x : Real) (k : ℤ) : 
  (0 < Real.cos x ∧ Real.cos x < 1) → 
  Real.logBase (Real.cos x) 4 * Real.logBase ((Real.cos x) ^ 2) 2 = 1 → 
  ∃ (k : ℤ), x = ((6 * k + 1) * (Real.pi / 3)) ∨ x = ((6 * k - 1) * (Real.pi / 3)) :=
by
  sorry

end solve_for_x_l735_735544


namespace power_eq_l735_735795

theorem power_eq (a b c : ℝ) (h₁ : a = 81) (h₂ : b = 4 / 3) : (a ^ b) = 243 * (3 ^ (1 / 3)) := by
  sorry

end power_eq_l735_735795


namespace gage_skating_time_l735_735502

theorem gage_skating_time :
  let minutes_skated_first_5_days := 75 * 5
  let minutes_skated_next_3_days := 90 * 3
  let total_minutes_skated_8_days := minutes_skated_first_5_days + minutes_skated_next_3_days
  let total_minutes_required := 85 * 9
  let minutes_needed_ninth_day := total_minutes_required - total_minutes_skated_8_days
  minutes_needed_ninth_day = 120 :=
by
  let minutes_skated_first_5_days := 75 * 5
  let minutes_skated_next_3_days := 90 * 3
  let total_minutes_skated_8_days := minutes_skated_first_5_days + minutes_skated_next_3_days
  let total_minutes_required := 85 * 9
  let minutes_needed_ninth_day := total_minutes_required - total_minutes_skated_8_days
  sorry

end gage_skating_time_l735_735502


namespace find_S11_l735_735190

noncomputable theory

open_locale classical

variables {a : ℕ → ℝ} (S : ℕ → ℝ)

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∃ a₁ : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop := ∀ n : ℕ, S n = n * (a 1 + a n) / 2

def special_condition (a : ℕ → ℝ) : Prop := a 5 + a 7 - (a 6) ^ 2 = 0

-- Problem statement
theorem find_S11 (h_arith : is_arithmetic_sequence a) (h_sum : sum_first_n_terms S a) (h_cond : special_condition a) : S 11 = 22 :=
sorry

end find_S11_l735_735190


namespace digit_in_decimal_expansion_l735_735088

theorem digit_in_decimal_expansion :
  (∀ n: ℕ, decimal_expansion 7 26 (1501) = 3) :=
sorry

-- Additional definitions might be required to identify the recurring decimal and the modular arithmetic.
axiom decimal_expansion : ℕ → ℕ → ℕ → ℕ

end digit_in_decimal_expansion_l735_735088


namespace find_cd_l735_735713

noncomputable def g (c d : ℝ) (x : ℝ) : ℝ := c * x^3 - 8 * x^2 + d * x - 7

theorem find_cd (c d : ℝ) 
  (h1 : g c d 2 = -7) 
  (h2 : g c d (-1) = -25) : 
  (c, d) = (2, 8) := 
by
  sorry

end find_cd_l735_735713


namespace parallel_vectors_eq_fraction_l735_735900

variables (x : ℝ)

def a : ℝ × ℝ := (Real.sin x, 3 / 2)
def b : ℝ × ℝ := (Real.cos x, -1)

theorem parallel_vectors_eq_fraction :
  (a x).snd * (b x).fst = (a x).fst * (b x).snd →
  (2 * Real.sin x - Real.cos x) / (4 * Real.sin x + 3 * Real.cos x) = 4 / 3 :=
by
  intro h
  have h1 : Real.sin x + 3 / 2 * Real.cos x = 0 := by sorry
  have h2 : Real.sin x = -3 / 2 * Real.cos x := by sorry
  calc
    (2 * Real.sin x - Real.cos x) / (4 * Real.sin x + 3 * Real.cos x)
        = (2*(-3/2)*Real.cos x - Real.cos x) / (4*(-3/2)*Real.cos x + 3*Real.cos x) : by sorry
    ... = (4*Real.cos x) / (-3*Real.cos x) : by sorry
    ... = 4 / 3 : by sorry

end parallel_vectors_eq_fraction_l735_735900


namespace proof_of_2a_minus_b_l735_735675

theorem proof_of_2a_minus_b :
  let a := 2
  let b := Real.sqrt 5 - a
  (2 * a - b) = (6 - Real.sqrt 5) :=
by {
  let a := 2,
  let b := Real.sqrt 5 - a,
  sorry
}

end proof_of_2a_minus_b_l735_735675


namespace largest_value_among_bases_l735_735438

theorem largest_value_among_bases :
  let n1 := 8 * 9 + 5,
      n2 := 2^6 - 1,
      n3 := 4^3,
      n4 := 2 * 36 + 1 * 6
  in max n1 (max n2 (max n3 n4)) = n4 :=
by
  let n1 := 77
  let n2 := 63
  let n3 := 64
  let n4 := 78
  sorry

end largest_value_among_bases_l735_735438


namespace intercepts_equal_l735_735892

theorem intercepts_equal (a : ℝ) : 
  (∃ x y : ℝ, (ax + y - 2 - a = 0) ∧ (x = (a + 2) / a) ∧ (y = 2 + a) ∧ (x = y)) ↔ (a = -2 ∨ a = 1) :=
begin
  sorry
end

end intercepts_equal_l735_735892


namespace hexagon_area_l735_735021

theorem hexagon_area (s t_height : ℕ) (tri_area rect_area : ℕ) :
    s = 2 →
    t_height = 4 →
    tri_area = 1 / 2 * s * t_height →
    rect_area = (s + s + s) * (t_height + t_height) →
    rect_area - 4 * tri_area = 32 :=
by
  sorry

end hexagon_area_l735_735021


namespace shark_sightings_in_cape_may_l735_735454

theorem shark_sightings_in_cape_may (x : ℕ) (hx : x + (2 * x - 8) = 40) : 2 * x - 8 = 24 := 
by 
  sorry

end shark_sightings_in_cape_may_l735_735454


namespace solve_equation_for_x_l735_735667

theorem solve_equation_for_x (x : ℝ) : 125 = 5 * (25)^(x - 1) ↔ x = 2 := 
by sorry

end solve_equation_for_x_l735_735667


namespace max_subset_size_l735_735980

open Set

variable (n : ℕ) (S : Set ℕ) (T : Set ℕ)

-- Assume n is a positive integer
axiom n_pos : n > 0

-- Define the set S
def S := {x | x ∈ Finset.range (3 * n) ∧ x > 0}

-- Define the condition on subset T
def valid_subset (T : Set ℕ) : Prop := ∀ x y z ∈ T, ¬(x + y + z ∈ T)

-- Define the proof for the maximum number of elements in such a subset T
theorem max_subset_size : ∀ (T : Set ℕ), (valid_subset n T) → (∃ N, N = 2 * n ∧ ∀ T', valid_subset n T' → Finset.card T' ≤ N) :=
by
  sorry

end max_subset_size_l735_735980


namespace lowest_probability_red_side_up_l735_735360

def card_flip_probability (k : ℕ) (n : ℕ) : ℚ :=
  if k ≤ n/2 then (n-k)^2/(n^2) + k^2/(n^2)
  else card_flip_probability (n+1-k) n 

theorem lowest_probability_red_side_up :
  (card_flip_probability 13 50) = (card_flip_probability 38 50) ∧ 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 25 → (card_flip_probability k 50 ≥ card_flip_probability 13 50)) :=
begin
  sorry
end

end lowest_probability_red_side_up_l735_735360


namespace find_discriminant_l735_735160

variables {a b c : ℝ}
variables (P : ℝ → ℝ)
def is_quadratic_polynomial (P : ℝ → ℝ) : Prop := ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, P x = a * x^2 + b * x + c)

theorem find_discriminant (h1 : is_quadratic_polynomial P)
  (h2 : ∃ x, P x = x - 2)
  (h3 : ∃ y, P y = 1 - y / 2)
  : ∃ D, D = -1/2 := 
sorry

end find_discriminant_l735_735160


namespace quadratic_root_k_value_l735_735074

theorem quadratic_root_k_value (k : ℝ) :
  (∀ x : ℝ, (x = -2 + real.sqrt 6 ∨ x = -2 - real.sqrt 6) → 2 * x^2 + 8 * x + k = 0) → k = -4 :=
by
  sorry

end quadratic_root_k_value_l735_735074


namespace product_of_odd_implies_sum_is_odd_l735_735655

theorem product_of_odd_implies_sum_is_odd (a b c : ℤ) (h : a * b * c % 2 = 1) : (a + b + c) % 2 = 1 :=
sorry

end product_of_odd_implies_sum_is_odd_l735_735655


namespace expenses_notation_l735_735323

theorem expenses_notation (income expense : ℤ) (h_income : income = 6) (h_expense : -expense = income) : expense = -4 := 
by
  sorry

end expenses_notation_l735_735323


namespace pieces_of_gum_per_nickel_l735_735658

-- Definitions based on the given conditions
def initial_nickels : ℕ := 5
def remaining_nickels : ℕ := 2
def total_gum_pieces : ℕ := 6

-- We need to prove that Quentavious gets 2 pieces of gum per nickel.
theorem pieces_of_gum_per_nickel 
  (initial_nickels remaining_nickels total_gum_pieces : ℕ)
  (h1 : initial_nickels = 5)
  (h2 : remaining_nickels = 2)
  (h3 : total_gum_pieces = 6) :
  total_gum_pieces / (initial_nickels - remaining_nickels) = 2 :=
by {
  sorry
}

end pieces_of_gum_per_nickel_l735_735658


namespace circle_coloring_l735_735290

theorem circle_coloring :
  ¬ ∃ (red_points blue_points : Finset ℕ),
    red_points.card = 1007 ∧
    blue_points.card = 1007 ∧
    (∀ point ∈ red_points, (red_points ∩ point.neighbors).card % 2 = 1) ∧
    (∀ point ∈ blue_points, (blue_points ∩ point.neighbors).card % 2 = 0) :=
by
  intro h
  sorry

end circle_coloring_l735_735290


namespace remainder_of_polynomial_division_l735_735826

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 7 * x^4 - 16 * x^3 + 3 * x^2 - 5 * x - 20

-- Define the divisor D(x)
def D (x : ℝ) : ℝ := 2 * x - 4

-- The remainder theorem sets x to 2 and evaluates P(x)
theorem remainder_of_polynomial_division : P 2 = -34 :=
by
  -- We will substitute x=2 directly into P(x)
  sorry

end remainder_of_polynomial_division_l735_735826


namespace manoel_score_l735_735644

/-- 
  Manoel tests his aim by shooting five arrows that hit the target at points A, B, C, D, and E,
  with coordinates A = (1, -1), B = (2, 5), C = (-1, 4), D = (-4, -4), E = (6, 5).
  The smallest circle has a radius of sqrt(2) and is centered at the origin.

  - Points scoring inside the smallest circle: 300 points.
  - B: 100 points.
  - C: 50 points.
  - D: 50 points.
  - E: 0 points.

  Prove that the total score of Manoel is 500 points.
-/
theorem manoel_score 
    (A : ℝ × ℝ := (1, -1))
    (B : ℝ × ℝ := (2.5, 1))
    (C : ℝ × ℝ := (-1, 4))
    (D : ℝ × ℝ := (-4, -4))
    (E : ℝ × ℝ := (6, 5))
    (dist : ℝ → ℝ → ℝ := λ x y => Real.sqrt(x^2 + y^2)) :
    let score :=
        (if dist A.1 A.2 < Real.sqrt(2) then 300 else 0) +
        100 + 50 + 50 + 0 in
    score = 500 := 
by
  -- Implementation steps skipped
  sorry

end manoel_score_l735_735644


namespace real_part_of_z_is_zero_l735_735848

noncomputable def z : ℂ := (2 + complex.i) / (-2 * complex.i + 1)

theorem real_part_of_z_is_zero : z.re = 0 := by
  sorry

end real_part_of_z_is_zero_l735_735848


namespace product_PA_sum_squares_PA_l735_735504

variables {n : ℕ} {r : ℝ} {O P : ℂ}
variables (A : fin n → ℂ) -- A_i as vertices of regular n-polygon

-- Condition: A is a regular polygon inscribed in a circle with radius r centered at O
-- And P is a point on the extension of O A_i

definition regular_n_polygon (A : fin n → ℂ) (O : ℂ) (r : ℝ) : Prop :=
∃ (θ : ℂ) (a : ℝ), 
  (θ = exp (2 * π * I / ↑n) ∧ a = r ∧ ∀ i, A i = O + a * θ^(i : ℂ))

def extended_point_on_OA (P O : ℂ) (A_i : ℂ) : Prop :=
∃ k : ℝ, P = O + k * (A_i - O) ∧ k > 1

theorem product_PA (h_reg_poly : regular_n_polygon A O r) (h_ext : ∀ i, extended_point_on_OA P O (A i)) :
  (∏ i : fin n, abs (P - A i)) = abs (P - O)^n - r^n :=
sorry

theorem sum_squares_PA (h_reg_poly : regular_n_polygon A O r) (h_ext : ∀ i, extended_point_on_OA P O (A i)) :
  (∑ i : fin n, abs (P - A i)^2) = n * (r^2 + abs (P - O)^2) :=
sorry

end product_PA_sum_squares_PA_l735_735504


namespace cost_of_bananas_and_cantaloupe_l735_735503

variable (a b c d : ℝ)

theorem cost_of_bananas_and_cantaloupe :
  (a + b + c + d = 30) →
  (d = 3 * a) →
  (c = a - b) →
  (b + c = 6) :=
by
  intros h1 h2 h3
  sorry

end cost_of_bananas_and_cantaloupe_l735_735503


namespace one_third_eleven_y_plus_three_l735_735910

theorem one_third_eleven_y_plus_three (y : ℝ) : 
  (1/3) * (11 * y + 3) = 11 * y / 3 + 1 :=
by
  sorry

end one_third_eleven_y_plus_three_l735_735910


namespace find_x_l735_735395

theorem find_x :
  ∃ x : ℝ, ((x * 0.85) / 2.5) - (8 * 2.25) = 5.5 ∧
  x = 69.11764705882353 :=
by
  sorry

end find_x_l735_735395


namespace smaller_samovar_cools_faster_l735_735381

-- Define the conditions and the expected conclusion in Lean
theorem smaller_samovar_cools_faster (V A : ℝ) (n : ℝ) (hV : V > 0) (hA : A > 0) (hn : n > 1) :
  let large_volume := n^3 * V
      small_volume := V
      large_area := n^2 * A
      small_area := A in
  (small_area / small_volume) > (large_area / large_volume) → 
  "smaller" = "маленький" :=
sorry

end smaller_samovar_cools_faster_l735_735381


namespace intersection_of_A_and_Z_l735_735202

def A : set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def Z := set.univ : set ℤ  -- ℤ can be used directly as the set of all integers

theorem intersection_of_A_and_Z :
  A ∩ (coe '' Z) = {-1, 0, 1} :=
by sorry

end intersection_of_A_and_Z_l735_735202


namespace find_discriminant_l735_735163

variables {a b c : ℝ}
variables (P : ℝ → ℝ)
def is_quadratic_polynomial (P : ℝ → ℝ) : Prop := ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, P x = a * x^2 + b * x + c)

theorem find_discriminant (h1 : is_quadratic_polynomial P)
  (h2 : ∃ x, P x = x - 2)
  (h3 : ∃ y, P y = 1 - y / 2)
  : ∃ D, D = -1/2 := 
sorry

end find_discriminant_l735_735163


namespace max_distance_on_curve_and_ellipse_l735_735868

noncomputable def max_distance_between_P_and_Q : ℝ :=
  6 * Real.sqrt 2

theorem max_distance_on_curve_and_ellipse :
  ∃ P Q, (P ∈ { p : ℝ × ℝ | p.1^2 + (p.2 - 6)^2 = 2 }) ∧ 
         (Q ∈ { q : ℝ × ℝ | q.1^2 / 10 + q.2^2 = 1 }) ∧ 
         (dist P Q = max_distance_between_P_and_Q) := 
sorry

end max_distance_on_curve_and_ellipse_l735_735868


namespace complex_expression_eq_l735_735965

-- Define the imaginary unit i
def I : ℂ := Complex.I

-- Define the expression
def expr : ℂ := (1 + I)^3 / (1 - I)^2

-- State the theorem
theorem complex_expression_eq : expr = -1 - I := by
  sorry

end complex_expression_eq_l735_735965


namespace magnitude_a_plus_b_angle_between_a_and_b_value_x_parallel_l735_735211

-- Definitions of vectors
def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (2, 1)

-- Magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Problem 1: Magnitude of a + b
theorem magnitude_a_plus_b :
  magnitude (a.1 + b.1, a.2 + b.2) = 5 :=
sorry

-- Problem 2: Angle between a and b
theorem angle_between_a_and_b :
  let θ := real.arccos (dot_product a b / (magnitude a * magnitude b)) in
  θ = real.pi / 4 :=
sorry

-- Problem 3: Value of x making x * a + 3 * b parallel to 3 * a - 2 * b
theorem value_x_parallel :
  ∃ x : ℝ, let v1 := (3 * x + 6, -x + 3),
                v2 := (9 - 4, -3 - 2) in
  v1.1 / v2.1 = v1.2 / v2.2 → x = -9 / 2 :=
sorry

end magnitude_a_plus_b_angle_between_a_and_b_value_x_parallel_l735_735211


namespace smaller_samovar_cools_faster_l735_735382

-- Define the conditions and the expected conclusion in Lean
theorem smaller_samovar_cools_faster (V A : ℝ) (n : ℝ) (hV : V > 0) (hA : A > 0) (hn : n > 1) :
  let large_volume := n^3 * V
      small_volume := V
      large_area := n^2 * A
      small_area := A in
  (small_area / small_volume) > (large_area / large_volume) → 
  "smaller" = "маленький" :=
sorry

end smaller_samovar_cools_faster_l735_735382


namespace determine_quadrant_l735_735182

variable {α : ℝ}

-- Conditions
def in_third_quadrant (α : ℝ) : Prop :=
  π < α ∧ α < 3 * π / 2

def cosine_condition (α : ℝ) : Prop :=
  |cos (α / 3)| = -cos (α / 3)

-- Theorem to prove
theorem determine_quadrant (h1 : in_third_quadrant α) (h2 : cosine_condition α) :
  π < α / 3 ∧ α / 3 < 3 * π / 2 :=
sorry

end determine_quadrant_l735_735182


namespace sum_of_intercepts_eq_eight_l735_735917

theorem sum_of_intercepts_eq_eight (m n k : ℝ) :
  (∀ x y : ℝ, (x^2 + y^2 + 3 * x + y + k = 0) → 
             (y = 3 * x + m) ∨ (y = 3 * x + n)) →
  (∃ a b c d : ℝ, ((a, b), (c, d)), ((a, d), (c, b)) are four distinct points of intersection such that they form a rectangle) →
  m + n = 8 :=
by
  sorry

end sum_of_intercepts_eq_eight_l735_735917


namespace area_triangle_MAB_l735_735976

/-- Define the first curve C1 in cartesian coordinates --/
def curve_C1 (P : ℝ × ℝ) : Prop := (P.1 - 2)^2 + P.2^2 = 4

/-- Define the first curve C1 in polar coordinates --/
def curve_C1_polar (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

/-- Define the second curve C2 in polar coordinates --/
def curve_C2_polar (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

/-- Define the point M --/
def point_M : ℝ × ℝ := (2, 0)

/-- Define the ray θ = π/3 --/
def ray (ρ θ : ℝ) : Prop := θ = Real.pi / 3 ∧ ρ > 0

/-- Prove the area of triangle MAB is 3 - √3 --/
theorem area_triangle_MAB : 
  let A := (4 * Real.cos (Real.pi / 3), Real.pi / 3) in
  let B := (4 * Real.sin (Real.pi / 3), Real.pi / 3) in
  let d := 2 * Real.sin (Real.pi / 3) in
  let AB := 4 * (Real.sin (Real.pi / 3) - Real.cos (Real.pi / 3)) in
  let S := (1 / 2) * AB * d in
  S = 3 - Real.sqrt 3 :=
by sorry

end area_triangle_MAB_l735_735976


namespace compare_a_b_c_l735_735864

variable {f : ℝ → ℝ}

theorem compare_a_b_c
  (h_even : ∀ x, f x = f (-x))
  (h_decreasing : ∀ x y, x < y → x ≤ 0 → y ≤ 0 → f x ≥ f y)
  (a_def : f (Real.logb 4 7) = a)
  (b_def : f (Real.logb (1/2) 3) = b)
  (c_def : f ((0.2)^0.6) = c) :
  c < a ∧ a < b :=
by
  sorry

end compare_a_b_c_l735_735864


namespace unique_functional_form_l735_735480

-- Define the problem conditions
def positive_reals := {x : ℝ // x > 0}

def satisfies_condition1 (f : positive_reals → positive_reals) := 
  ∀ x y : positive_reals, f ⟨x.val * f y.val, mul_pos x.property (y.property)⟩ = ⟨y.val * f x.val, mul_pos y.property (x.property)⟩ 

def satisfies_condition2 (f : positive_reals → positive_reals) := 
  filter.tendsto (λ x : positive_reals, f x.val) filter.at_top (nhds 0)

-- Define the main theorem
theorem unique_functional_form :
  ∀ f : positive_reals → positive_reals, satisfies_condition1 f → satisfies_condition2 f → (∀ x : positive_reals, f x = ⟨1 / x.val, one_div_pos.mpr x.property⟩) :=
by
  sorry

end unique_functional_form_l735_735480


namespace min_transport_time_8_hours_l735_735007

theorem min_transport_time_8_hours (v : ℝ) :
  (∀ (d_a_b : ℝ) (num_trains : ℕ) (safety_distance : ℝ), 
    d_a_b = 400 ∧ num_trains = 17 ∧ 
    safety_distance = (v / 20) ^ 2) →
  (∀ t, t = (400 / v) + (v / 25) ∧ v = 100) →
  ∃ v, v = 100 ∧ (400 / v) + (v / 25) = 8 := 
by 
  intro h _,
  exists 100,
  exact ⟨rfl, by norm_cast⟩,
sorry

end min_transport_time_8_hours_l735_735007


namespace johns_elevation_final_l735_735952

theorem johns_elevation_final (init_elev : ℕ) 
                              (first_rate : ℕ) (first_duration : ℕ) 
                              (second_rate : ℕ) (second_duration : ℕ) 
                              (third_rate : ℕ) (third_duration : ℕ) 
                              (first_total : ℕ) (second_total : ℕ) (third_total : ℕ) 
                              (total_descent : ℕ) (final_elev : ℕ) :
  init_elev = 400 →
  first_rate = 10 → first_duration = 5 → first_total = 50 →
  second_rate = 15 → second_duration = 3 → second_total = 45 →
  third_rate = 12 → third_duration = 6 → third_total = 72 →
  total_descent = 167 →
  final_elev = 233 →
  final_elev  = init_elev - total_descent := 
begin
  sorry
end

end johns_elevation_final_l735_735952


namespace exists_1985_distinct_integers_sum_of_squares_is_cube_and_sum_of_cubes_is_square_l735_735474

theorem exists_1985_distinct_integers_sum_of_squares_is_cube_and_sum_of_cubes_is_square :
  ∃ (a : Fin 1985 → ℕ),
    (Function.Injective a) ∧
    (∃ k, (∑ i, a i ^ 2) = k^3) ∧ 
    (∃ m, (∑ i, a i ^ 3) = m^2) :=
sorry

end exists_1985_distinct_integers_sum_of_squares_is_cube_and_sum_of_cubes_is_square_l735_735474


namespace quadratic_polynomial_discriminant_l735_735138

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₁ : a ≠ 0)
  (h₂ : ∃ x : ℝ, a * x^2 + b * x + c = x - 2 ∧ (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h₃ : ∃ x : ℝ, a * x^2 + b * x + c = 1 - x / 2 ∧ (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l735_735138


namespace discriminant_of_P_l735_735156

theorem discriminant_of_P (a b c : ℚ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1/2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1/2 := 
begin
  -- Proof omitted for brevity
  sorry
end

end discriminant_of_P_l735_735156


namespace collinear_T_O_O1_l735_735959

open EuclideanGeometry

-- Definitions based on provided conditions

variables {Ω : Type*} [MetricSpace Ω] [EuclideanSpace 3 Ω]

variables 
  {W X Y Z O T A B C D P Q R O1 : Ω}
  (circum_omega : Circle 3)
  (center_O : circum_omega.center = O)
  (tang_points : Set Ω)
  (points_on_circumference : {W, X, Y, Z} ⊆ circum_omega)
  (wy_perpendicular_xz : ∟ (W -ᵥ Y) (X -ᵥ Z))
  (intersection_T : T = line_through W Y ∩ line_through X Z)
  (tangency_AB : tangent_line circum_omega W ⟂ A B)
  (tangency_BC : tangent_line circum_omega X ⟂ B C)
  (tangency_CD : tangent_line circum_omega Y ⟂ C D)
  (tangency_DA : tangent_line circum_omega Z ⟂ D A)
  (line_PA : P = perpendicular_bisector O A A)
  (line_PB : P = perpendicular_bisector O B B)
  (line_QB : Q = perpendicular_bisector O B B)
  (line_QC : Q = perpendicular_bisector O C C)
  (line_RC : R = perpendicular_bisector O C C)
  (line_RD : R = perpendicular_bisector O D D)
  (circle_O1 : Circumcenter P Q R O1)

theorem collinear_T_O_O1
  : Collinear {T, O, O1} := sorry

end collinear_T_O_O1_l735_735959


namespace hypotenuse_length_l735_735591

theorem hypotenuse_length (a b c : ℝ) (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 := 
by
  sorry

end hypotenuse_length_l735_735591


namespace max_area_triangle_AM_l735_735187

noncomputable def CircleE (a b : ℝ) : (ℝ → ℝ → Prop) :=
  λ x y, (x^2 / a^2) + (y^2 / b^2) = 1

def isEquilateralTriangle (A B C : ℝ × ℝ) (side_length : ℝ) : Prop :=
  dist A B = side_length ∧ dist B C = side_length ∧ dist C A = side_length

theorem max_area_triangle_AM :
  ∃ M A B : ℝ × ℝ, 
  (∀ a b : ℝ, b > 0 → CircleE a b M.1 M.2) →
  isEquilateralTriangle (0, √3) M A 2 →
  isEquilateralTriangle M A B 2 →
  (∃ AM BM AB : ℝ, AM ≠ 0 ∧ BM ≠ 0 ∧ AB ≠ 0 ∧ AM * BM < 0) →
  ∃ S : ℝ, S = sqrt 3 / 2 := sorry

end max_area_triangle_AM_l735_735187


namespace no_real_roots_iff_k_gt_1_div_4_l735_735221

theorem no_real_roots_iff_k_gt_1_div_4 (k : ℝ) :
  (∀ x : ℝ, ¬ (x^2 - x + k = 0)) ↔ k > 1 / 4 :=
by
  sorry

end no_real_roots_iff_k_gt_1_div_4_l735_735221


namespace quadratic_polynomial_discriminant_l735_735141

def P (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∃ x : ℝ, P a b c x = x - 2 ∧ (discriminant a (b - 1) (c + 2) = 0))
  (h₂ : ∃ x : ℝ, P a b c x = 1 - x / 2 ∧ (discriminant a (b + 1 / 2) (c - 1) = 0)) :
  discriminant a b c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l735_735141


namespace cape_may_shark_sightings_l735_735456

def total_shark_sightings (D C : ℕ) : Prop :=
  D + C = 40

def cape_may_sightings (D C : ℕ) : Prop :=
  C = 2 * D - 8

theorem cape_may_shark_sightings : 
  ∃ (C D : ℕ), total_shark_sightings D C ∧ cape_may_sightings D C ∧ C = 24 :=
by
  sorry

end cape_may_shark_sightings_l735_735456


namespace value_of_c_l735_735225

theorem value_of_c :
  (∀ (x y: ℝ), (∃ (k : ℕ), k > 0 ∧ k < 4 ∧ ((sqrt ((x - 1)^2 + y^2)) / (sqrt ((x - 4)^2 + y^2)) = 1/2)) ∧
  (∀ (c : ℝ), (∃ (M : ℕ), 
    ((abs c) / (sqrt 2) = 1 ∧ 
    M = 3) ∧  
      (∃ (d: ℝ), d = (sqrt ((0 - 0 + c)^2) / sqrt(1^2 + (-1)^2)) ∧
        d = abs(sqrt 2))))  →
  (c = sqrt 2 ∨ c = -sqrt 2)) := by
  sorry

end value_of_c_l735_735225


namespace cosine_of_angle_opposite_9_units_leg_l735_735514

theorem cosine_of_angle_opposite_9_units_leg :
  ∀ (a b c : ℝ),
    a = 9 →
    c = 15 →
    a^2 + b^2 = c^2 →
    ∃ (θ : ℝ), cos θ = a / c :=
by
  intros a b c ha hc h
  rw [ha, hc] at h
  let b := 12 -- Use Pythagorean theorem to calculate b
  use (Real.arccos (a / c))
  rw [ha, hc]
  exact sorry

end cosine_of_angle_opposite_9_units_leg_l735_735514


namespace multiply_123_32_125_l735_735452

theorem multiply_123_32_125 : 123 * 32 * 125 = 492000 := by
  sorry

end multiply_123_32_125_l735_735452


namespace roots_quadratic_fraction_l735_735906

theorem roots_quadratic_fraction :
  (∀ (x : ℝ), (Polynomial.eval x (Polynomial.C 1 * Polynomial.X ^ 2 - Polynomial.C 2 * Polynomial.X - Polynomial.C 4) = 0) → 
  let x1 := (-2 + Real.sqrt (4 + 16)) / (2 * 1)
  let x2 := (-2 - Real.sqrt (4 + 16)) / (2 * 1)
  (x1 + x2) / (x1 * x2) = -1 / 2) := 
sorry

end roots_quadratic_fraction_l735_735906


namespace set_in_proportion_l735_735402

theorem set_in_proportion : 
  let a1 := 3
  let a2 := 9
  let b1 := 10
  let b2 := 30
  (a1 * b2 = a2 * b1) := 
by {
  sorry
}

end set_in_proportion_l735_735402


namespace total_fish_caught_l735_735499

-- Definitions based on conditions
def sums : List ℕ := [7, 9, 14, 14, 19, 21]

-- Statement of the proof problem
theorem total_fish_caught : 
  (∃ (a b c d : ℕ), [a+b, a+c, a+d, b+c, b+d, c+d] = sums) → 
  ∃ (a b c d : ℕ), a + b + c + d = 28 :=
by 
  sorry

end total_fish_caught_l735_735499


namespace solution_set_f_l735_735576

noncomputable def f (a x : ℝ) : ℝ := a * exp (-x) - exp x

-- conditions: f(x) is an odd function, a = 1
theorem solution_set_f (a : ℝ)
  (h : ∀ x : ℝ, f a x = -f a (-x))
  (ha : a = 1) :
  {x : ℝ | f a (x - 1) < exp 1 - 1 / exp 1} = {x : ℝ | 0 < x} :=
by
  sorry

end solution_set_f_l735_735576


namespace _l735_735752

-- Definitions based on conditions
def total_students : ℕ := 600
def sample_size : ℕ := 50
def first_drawn : ℕ := 3
def sampling_interval : ℕ := total_students / sample_size
def camp_I : set ℕ := {n | 1 ≤ n ∧ n ≤ 300}
def camp_II : set ℕ := {n | 301 ≤ n ∧ n ≤ 495}
def camp_III : set ℕ := {n | 496 ≤ n ∧ n ≤ 600}

-- Main theorem statement based on correct answer
lemma count_camp_III_in_sample : 
  let seq := (list.range sample_size).map (λ i, first_drawn + i * sampling_interval)
  in seq.filter (λ n, n ∈ camp_III).length = 8 :=
by
  sorry

end _l735_735752


namespace minimal_flip_probability_l735_735366

def flip_probability (n : ℕ) (k : ℕ) : ℚ :=
  if k <= 25 then
    (2 * k^2 - 52 * k + 676) / 676
  else
    let mirrored_k := 51 - k in
    (2 * mirrored_k^2 - 52 * mirrored_k + 676) / 676

theorem minimal_flip_probability :
  ∀ k, (13 ≤ k ∧ k ≤ 13) ∨ (38 ≤ k ∧ k ≤ 38) :=
by
  intro k
  sorry

end minimal_flip_probability_l735_735366


namespace apples_per_pie_l735_735284

-- Definitions of the conditions
def number_of_pies : ℕ := 10
def harvested_apples : ℕ := 50
def to_buy_apples : ℕ := 30
def total_apples_needed : ℕ := harvested_apples + to_buy_apples

-- The theorem to prove
theorem apples_per_pie :
  (total_apples_needed / number_of_pies) = 8 := 
sorry

end apples_per_pie_l735_735284


namespace first_player_wins_9_first_player_wins_10_l735_735288

-- Define the game conditions and the main theorem
def game {n : ℕ} (board : list bool) (player : ℕ) : Prop :=
  -- Placeholder for actual game logic
  sorry

theorem first_player_wins_9 : game (list.replicate 9 ff) 1 := 
sorry

theorem first_player_wins_10 : game (list.replicate 10 ff) 1 := 
sorry

end first_player_wins_9_first_player_wins_10_l735_735288


namespace range_of_a_l735_735188

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, |x - a| + |x + 1| ≤ 2) ↔ a ∈ (Set.Iio (-3) ∪ Set.Ioi 1) :=
by sorry

end range_of_a_l735_735188


namespace min_value_x_l735_735218

open Real 

variable (x : ℝ)

theorem min_value_x (hx_pos : 0 < x) 
    (ineq : log x ≥ 2 * log 3 + (1 / 3) * log x + 1) : 
    x ≥ 27 * exp (3 / 2) :=
by 
  sorry

end min_value_x_l735_735218


namespace problem_l735_735825

-- Define the point P
def P (m : ℝ) : ℝ × ℝ := (m, 1)

-- Define the line 3x + 4y = 0 in terms of A, B, and C
def A : ℝ := 3
def B : ℝ := 4
def C : ℝ := 0

-- Define the distance formula from a point to a line
def distance (P : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  abs (A * P.1 + B * P.2 + C) / real.sqrt (A^2 + B^2)

-- State the problem in Lean 4
theorem problem (m : ℝ) (d : ℝ) (h1: d = distance (P m) A B C) (h2: d > 1) : 
  m < -3 ∨ m > 1 / 3 :=
by
  sorry

end problem_l735_735825


namespace no_integer_solutions_l735_735307

theorem no_integer_solutions (a b c : ℤ) : ¬ (a^2 + b^2 = 8 * c + 6) :=
sorry

end no_integer_solutions_l735_735307


namespace length_of_AX_l735_735242

theorem length_of_AX (A B C X : Type) (AB BC CA : ℝ) (AX BX CX : ℝ)
  (h1 : AB = 35) 
  (h2 : BC = 40) 
  (h3 : CA = 45)
  (h4 : BX = 21)
  (h5 : CA/AB = CX/BX) :
  AX = 14 :=
by 
  -- introduce conditions
  rw [h1] at *,
  rw [h3] at *,
  rw [h4] at *,
  rw [h5] at *,
  -- perform calculations (assuming non-trivial division and multiplication are correct)
  sorry

end length_of_AX_l735_735242


namespace polynomial_divisibility_l735_735994

theorem polynomial_divisibility (a b x y : ℤ) : 
  ∃ k : ℤ, (a * x + b * y)^3 + (b * x + a * y)^3 = k * (a + b) * (x + y) := by
  sorry

end polynomial_divisibility_l735_735994


namespace quadrant_of_z_l735_735193

noncomputable def z : ℂ := (i + 2) / i

theorem quadrant_of_z : ∃ (quadrant: ℕ), quadrant = 4 ∧ (1 - 2*i).re > 0 ∧ (1 - 2*i).im < 0 := by
  use 4
  simp
  sorry

end quadrant_of_z_l735_735193


namespace chess_tournament_bound_l735_735748

-- The main theorem to prove
theorem chess_tournament_bound (p n : ℕ) (h1 : 2 < p)
  (h2 : ∀ (i j : ℕ), i ≠ j → i, j < p → i and j play at most one game against each other) 
  (h3 : ∀ G : graph ℕ, ( ∀ {i j : ℕ}, G n ≤ p(2)) -- More precision required here But setting as a base constraint
  (h4 : ∀ s : set ℕ, #s = 3 → ∃ i j ∈ s, i ≠ j ∧ ¬G.adj i j) :
  n ≤ p^2 / 4 := 
sorry

end chess_tournament_bound_l735_735748


namespace quadratic_solution_l735_735183

theorem quadratic_solution (a : ℝ) (h : 2^2 - 3 * 2 + a = 0) : 2 * a - 1 = 3 :=
by {
  sorry
}

end quadratic_solution_l735_735183


namespace cost_per_rug_proof_l735_735285

noncomputable def cost_per_rug (price_sold : ℝ) (number_rugs : ℕ) (profit : ℝ) : ℝ :=
  let total_revenue := number_rugs * price_sold
  let total_cost := total_revenue - profit
  total_cost / number_rugs

theorem cost_per_rug_proof : cost_per_rug 60 20 400 = 40 :=
by
  -- Lean will need the proof steps here, which are skipped
  -- The solution steps illustrate how Lean would derive this in a proof
  sorry

end cost_per_rug_proof_l735_735285


namespace count_valid_a₁_l735_735973

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

def a (n : ℕ) (a₁ : ℕ) : ℕ :=
  if n = 1 then a₁
  else if is_even (a (n - 1) a₁) then a (n - 1) a₁ / 2
  else if is_multiple_of_5 (a (n - 1) a₁) then 2 * a (n - 1) a₁ + 1
  else 3 * a (n - 1) a₁ + 1

theorem count_valid_a₁ : 
  (∑ k in finset.range 750, 1) = 750 :=
by
  sorry

end count_valid_a₁_l735_735973


namespace min_loss_expense_income_l735_735096

def arithmetic_progression (a d : ℕ) (n : ℕ) : ℕ := a + n * d

theorem min_loss_expense_income :
  let P := arithmetic_progression 1 1 0,
      A := arithmetic_progression 1 1 1,
      C := arithmetic_progression 1 1 2,
      X := arithmetic_progression 1 1 3,
      O := arithmetic_progression 1 1 4,
      Д := arithmetic_progression 1 1 5,
      EXPENSE := P * 100000 + A * 10000 + C * 1000 + X * 100 + O * 10 + Д,
      INCOME := A * 10000 + C * 1000 + X * 100 + O * 10 + Д
  in EXPENSE - INCOME = 58000 :=
  by
  sorry

end min_loss_expense_income_l735_735096


namespace ending_number_of_SetB_l735_735998

-- Definition of Set A
def SetA : Set ℕ := {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

-- Definition of Set B
def SetB_ends_at (n : ℕ) : Set ℕ := {i | 6 ≤ i ∧ i ≤ n}

-- The main theorem statement
theorem ending_number_of_SetB : ∃ n, SetA ∩ SetB_ends_at n = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15} ∧ 10 ∈ SetA ∩ SetB_ends_at n := 
sorry

end ending_number_of_SetB_l735_735998


namespace intersect_once_l735_735465

theorem intersect_once (x : ℝ) : 
  (∀ y, y = 3 * Real.log x ↔ y = Real.log (3 * x)) → (∃! x, 3 * Real.log x = Real.log (3 * x)) :=
by 
  sorry

end intersect_once_l735_735465


namespace james_calories_per_minute_l735_735950

variable (classes_per_week : ℕ) (hours_per_class : ℝ) (total_calories_per_week : ℕ)

theorem james_calories_per_minute
  (h1 : classes_per_week = 3)
  (h2 : hours_per_class = 1.5)
  (h3 : total_calories_per_week = 1890) :
  total_calories_per_week / (classes_per_week * (hours_per_class * 60)) = 7 := 
by
  sorry

end james_calories_per_minute_l735_735950


namespace find_B_find_a_plus_c_l735_735920

variable {A B C a b c : ℝ}
variable (b_4 : b = 4) (A_π_B : b * cos A = (2 * c + a) * cos (π - B))
variable (area : 1 / 2 * a * c * sin B = sqrt 3)

theorem find_B (h : b * cos A = (2 * c + a) * cos (π - B)) : B = 2 * π / 3 :=
  by sorry

theorem find_a_plus_c (b_4 : b = 4) (area : 1 / 2 * a * c * sin B = sqrt 3) (B_eq : B = 2 * π / 3) :
  a + c = 2 * sqrt 5 :=
  by sorry

end find_B_find_a_plus_c_l735_735920


namespace find_a_l735_735569

theorem find_a (a : ℝ) (h : a^2 + a^2 / 4 = 5) : a = 2 ∨ a = -2 := 
sorry

end find_a_l735_735569


namespace smallest_n_exists_l735_735631

open Finset

def pairwise_coprime (s : Finset ℕ) : Prop :=
  ∀ (x y ∈ s), x ≠ y → Nat.coprime x y

def subset_contains_pairwise_coprime (S : Finset ℕ) (n : ℕ) : Prop :=
  ∀ (T : Finset ℕ), T ⊆ S ∧ T.card = n → 
    ∃ (C : Finset ℕ), C ⊆ T ∧ C.card = 4 ∧ pairwise_coprime C

theorem smallest_n_exists :
  let S := (finset.range 101).erase 0 in
  ∃ n, subset_contains_pairwise_coprime S n ∧ 
       ∀ m, (m < n → ¬ subset_contains_pairwise_coprime S m) := 
  sorry

end smallest_n_exists_l735_735631


namespace number_of_zeros_l735_735548

noncomputable def f (a : ℝ) (x : ℝ) := a * x^2 - 2 * a * x + a + 1
noncomputable def g (b : ℝ) (x : ℝ) := b * x^3 - 2 * b * x^2 + b * x - 4 / 27

theorem number_of_zeros (a b : ℝ) (ha : a > 0) (hb : b > 1) :
  ∃! (x : ℝ), g b (f a x) = 0 := sorry

end number_of_zeros_l735_735548


namespace quadratic_polynomial_discriminant_l735_735139

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₁ : a ≠ 0)
  (h₂ : ∃ x : ℝ, a * x^2 + b * x + c = x - 2 ∧ (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h₃ : ∃ x : ℝ, a * x^2 + b * x + c = 1 - x / 2 ∧ (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l735_735139


namespace line_through_point_with_negative_reciprocal_intercepts_l735_735091

theorem line_through_point_with_negative_reciprocal_intercepts :
  ∃ a b : ℝ, (2, -3) ∈ set_of (λ p : ℝ × ℝ, 3 * p.1 + 2 * p.2 = 0) ∧ (b = -a) ∧ (3 * (2 : ℝ) + 2 * (-3 : ℝ) = 0) :=
by
  sorry

end line_through_point_with_negative_reciprocal_intercepts_l735_735091


namespace quadratic_polynomial_discriminant_l735_735137

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₁ : a ≠ 0)
  (h₂ : ∃ x : ℝ, a * x^2 + b * x + c = x - 2 ∧ (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h₃ : ∃ x : ℝ, a * x^2 + b * x + c = 1 - x / 2 ∧ (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l735_735137


namespace graph_shift_l735_735378

-- Define the functions
def f1 (x : ℝ) := Real.sin (3 * x - Real.pi / 4)
def f2 (x : ℝ) := Real.cos (3 * x)

-- State the theorem to be proven
theorem graph_shift : ∀ (x : ℝ), f2 (x - Real.pi / 4) = f1 x := by
  -- Proof body to be filled in
  sorry

end graph_shift_l735_735378


namespace minimize_distance_PQ_l735_735236

open Real

def curve_C1 (φ : ℝ) : ℝ × ℝ :=
  (-4 + 2 * cos φ, 3 + 2 * sin φ)

def curve_C2 (θ : ℝ) : ℝ × ℝ :=
  let ρ := 4 / (cos θ - sin θ)
  (ρ * cos θ, ρ * sin θ)

theorem minimize_distance_PQ :
  let P := (-4 + sqrt 2, 3 - sqrt 2)
  ∀ Q ∈ {q : ℝ × ℝ | ∃ θ, q = curve_C2 θ}, 
  ∀ P' ∈ {p : ℝ × ℝ | ∃ φ, p = curve_C1 φ}, 
  dist P Q ≤ dist P' Q :=
sorry

end minimize_distance_PQ_l735_735236


namespace painted_cube_faces_l735_735018

theorem painted_cube_faces :
  let total_cubes := 4 * 4 * 4 in
  let edge_cubes_with_two_faces := 12 * 2 in
  total_cubes = 64 ∧
  edge_cubes_with_two_faces = 24 ∧
  (forall (cube : ℤ), cube ∈ (from 1 to total_cubes) → 
    (painted_faces cube = 2 → paint_count cube = 24)) :=
begin
  sorry -- Proof omitted as per instructions
end

end painted_cube_faces_l735_735018


namespace sum_of_squares_of_roots_l735_735827

theorem sum_of_squares_of_roots (s₁ s₂ : ℝ) (h1 : s₁ + s₂ = 10) (h2 : s₁ * s₂ = 9) : 
  s₁^2 + s₂^2 = 82 := by
  sorry

end sum_of_squares_of_roots_l735_735827


namespace limit_of_ratio_a_b_l735_735343

-- Definitions based on initial conditions and recursive relations
def a_nat : ℕ → ℝ
| 0       := 1  -- Lean's nat is 0-based, so a_0 corresponds to a_1 in the problem
| (n + 1) := b_nat n - 2 * a_nat n

def b_nat : ℕ → ℝ
| 0       := 7  -- Similarly, b_0 corresponds to b_1 in the problem
| (n + 1) := 3 * b_nat n - 4 * a_nat n

-- Statement of the problem rephrased as a formal theorem in Lean
theorem limit_of_ratio_a_b (H : ∀ n, b_nat n ≠ 0) :
  tendsto (λ n, (a_nat n / b_nat n)) at_top (𝓝 (1/4)) :=
sorry

end limit_of_ratio_a_b_l735_735343


namespace find_norm_b_projection_of_b_on_a_l735_735870

open Real EuclideanSpace

noncomputable def a : ℝ := 4

noncomputable def angle_ab : ℝ := π / 4  -- 45 degrees in radians

noncomputable def inner_prod_condition (b : ℝ) : ℝ := 
  (1 / 2 * a) * (2 * a) + 
  (1 / 2 * a) * (-3 * b) + 
  b * (2 * a) + 
  b * (-3 * b) - 12

theorem find_norm_b (b : ℝ) (hb : inner_prod_condition b = 0) : b = sqrt 2 :=
  sorry

theorem projection_of_b_on_a (b : ℝ) (hb : inner_prod_condition b = 0) : 
  (b * cos angle_ab) = 1 :=
  sorry

end find_norm_b_projection_of_b_on_a_l735_735870


namespace min_constant_l735_735275

theorem min_constant (n : ℕ) (x : ℕ → ℝ)
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → x i ≥ -1)
  (h2 : ∑ i in Finset.range n, x i ^ 3 = 0) :
  (∑ i in Finset.range n, x i ^ 2) ≤ (4 / 3) * n :=
sorry

end min_constant_l735_735275


namespace inequality_solution_set_l735_735691

open Set -- Open the Set namespace to work with sets in Lean

theorem inequality_solution_set (x : ℝ) :
  (3 * x - 1) / (2 - x) ≥ 1 ↔ (x ∈ Icc (3 / 4) 2 \ {2}) := 
by
  sorry

end inequality_solution_set_l735_735691


namespace problem_statement_l735_735191

theorem problem_statement
    (a : ℝ)
    (M : ℝ × ℝ := (1, a))
    (O : set (ℝ × ℝ) := { p | p.1^2 + p.2^2 = 4 })
    (line_tangent_M : M ∈ O → ∃! l : set (ℝ × ℝ), is_tangent l O ∧ M ∈ l)
    (chords_AC_BD : true → ∃ (AC BD: set (ℝ × ℝ)), is_chord AC O ∧ is_chord BD O ∧ is_perpendicular AC BD ∧ |AC| + |BD| ≤ 2*real.sqrt 10)
    : (M ∈ O → (a = real.sqrt 3 ∨ a = -real.sqrt 3) ∧
      ((a = real.sqrt 3 → ∃ l, l = { p | p.1 + real.sqrt 3 * p.2 - 4 = 0 } ∨ l = { p | p.1 - real.sqrt 3 * p.2 - 4 = 0 })),
      (a = real.sqrt 2 → |AC| + |BD| = 2 * real.sqrt 10)) :=
by
  sorry

end problem_statement_l735_735191


namespace range_of_a_l735_735546

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 2 then -x^2 + 4 * x else Real.logb 2 x - a

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) → 1 < a :=
sorry

end range_of_a_l735_735546


namespace function_inequality_l735_735185

noncomputable def f : ℝ → ℝ
| x => if -1 ≤ x ∧ x ≤ 1 then x * (1 - (2 / (Real.exp x + 1))) else f (x % 2)

theorem function_inequality (x : ℝ) :
  (∀ x, f x = f (x + 2)) →
  (∀ x, f (x-1) = f (x+1)) →
  (∀ x ∈ Icc (-1 : ℝ) (1 : ℝ), f x = x * (1 - 2 / (Real.exp x + 1))) →
  (f 2 < f (5 / 2) ∧ f (5 / 2) < f (-3)) :=
by
  sorry

end function_inequality_l735_735185


namespace ratio_of_AB_to_BC_is_2sqrt2_over_pi_l735_735301

-- Define the setup as per conditions
variables {r : ℝ} (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable [Circle (radius := r)] -- Assume this represents points A, B, C on a circle with radius r

-- Definitions based on the conditions:
variable (AB AC BC : ℝ)
variable (arc_length_BC : ℝ) 

-- Conditions
def condition1 := A ∈ Circle r ∧ B ∈ Circle r ∧ C ∈ Circle r -- Points A, B, C on a circle of radius r
def condition2 := AB = AC -- AB = AC
def condition3 := AB > r -- AB > r
def condition4 := arc_length_BC = π * r / 2 -- Length of minor arc BC is πr/2

-- The theorem to be proved
theorem ratio_of_AB_to_BC_is_2sqrt2_over_pi
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3)
  (h4 : condition4) :
  AB / BC = 2 * sqrt 2 / π :=
sorry

end ratio_of_AB_to_BC_is_2sqrt2_over_pi_l735_735301


namespace ellipse_area_l735_735445

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

end ellipse_area_l735_735445


namespace measles_cases_1993_l735_735921

theorem measles_cases_1993 :
  ∀ (cases_1970 cases_1986 cases_2000 : ℕ)
    (rate1 rate2 : ℕ),
  cases_1970 = 600000 →
  cases_1986 = 30000 →
  cases_2000 = 600 →
  rate1 = 35625 →
  rate2 = 2100 →
  cases_1986 - 7 * rate2 = 15300 :=
by {
  sorry
}

end measles_cases_1993_l735_735921


namespace distinct_placing_methods_l735_735750

theorem distinct_placing_methods (n : ℕ) (h : n = 100) :
  ∃ (method1 method2 : set (ℝ × ℝ × ℝ)), method1 ≠ method2 ∧
  (method1 ∪ method2).card = n ∧
  ∀ p ∈ (method1 ∪ method2), p ∈ (surface_of_cube) :=
by
  sorry

end distinct_placing_methods_l735_735750


namespace m_n_divisible_by_4_l735_735773

def T_tetromino_tiling (m n : ℕ) : Prop :=
  ∃ f : (Fin 4 → Fin 4) → Fin 2, 
  ∀ i j : Fin 4, ∀ k l : Fin 4, f i = f j → (i ≠ j → (i ∈ {0,1,2,3}) → (j ∈ {0,1,2,3})
  → ∃!y : Fin 2, y ≡ i ≡ j [MOD 2] -- Well definedness conditions are necessary

theorem m_n_divisible_by_4 (m n : ℕ) (h : T_tetromino_tiling m n) :
  m % 4 = 0 ∧ n % 4 = 0 :=
sorry

end m_n_divisible_by_4_l735_735773


namespace value_of_q_l735_735629

-- Define the problem in Lean 4

variable (a d q : ℝ) (h0 : a ≠ 0)
variables (M P : Set ℝ)
variable (hM : M = {a, a + d, a + 2 * d})
variable (hP : P = {a, a * q, a * q * q})
variable (hMP : M = P)

theorem value_of_q : q = -1 :=
by
  sorry

end value_of_q_l735_735629


namespace unique_real_root_and_series_l735_735747

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 + 5 * x - 2

theorem unique_real_root_and_series :
  (∃! r : ℝ, (f r = 0) ∧ (0 < r) ∧ (r < 1/2)) ∧
  (∃! (a : ℕ → ℕ), strict_mono a ∧ (∀ n, 0 < a n) ∧
  (∑' n, (r : ℝ) ^ (a n) = 2 / 5)) := sorry

end unique_real_root_and_series_l735_735747


namespace calculate_molecular_weight_CaBr2_l735_735051

def atomic_weight_Ca : ℝ := 40.08                 -- The atomic weight of calcium (Ca)
def atomic_weight_Br : ℝ := 79.904                -- The atomic weight of bromine (Br)
def molecular_weight_CaBr2 : ℝ := atomic_weight_Ca + 2 * atomic_weight_Br  -- Definition of molecular weight of CaBr₂

theorem calculate_molecular_weight_CaBr2 : molecular_weight_CaBr2 = 199.888 := by
  sorry

end calculate_molecular_weight_CaBr2_l735_735051


namespace fill_tank_time_l735_735704

theorem fill_tank_time (R1 R2 : ℝ) (R1_rate : R1 = 1 / 20) (R2_rate : R2 = 1 / 30)
  (leak : ℝ → ℝ := λ (rate : ℝ), (2 / 3) * rate) :
  let R_combined := R1 + R2 in
  let R_effective := leak R_combined in
  (1 / R_effective) = 18 :=
by
  sorry

end fill_tank_time_l735_735704


namespace shorter_piece_length_l735_735005

theorem shorter_piece_length (total_length : ℝ) (ratio : ℝ) 
(h_total : total_length = 70) 
(h_ratio : ratio = 2 / 3) : 
  ∃ x : ℝ, x = 26.25 ∧ x + x * ratio = total_length :=
by
  let x := 26.25
  have h1 : x + (x + x * ratio) = total_length, from sorry
  exact ⟨x, rfl, h1⟩

end shorter_piece_length_l735_735005


namespace batsman_average_after_12_innings_l735_735728

noncomputable def batsman_average (runs_in_12th: ℕ) (average_increase: ℕ) (original_avg: ℕ) : ℕ :=
  let new_avg := original_avg + average_increase in
  let total_runs_11_innings := 11 * original_avg in
  let total_runs_12_innings := total_runs_11_innings + runs_in_12th in
  total_runs_12_innings / 12

theorem batsman_average_after_12_innings :
  ∀ (original_avg : ℕ),
    (11 * original_avg + 48) / 12 = 26 → original_avg = 24 :=
by
  intros original_avg h
  sorry

end batsman_average_after_12_innings_l735_735728


namespace seungho_more_marbles_l735_735306

variable (S H : ℕ)

-- Seungho gave 273 marbles to Hyukjin
def given_marbles : ℕ := 273

-- After giving 273 marbles, Seungho has 477 more marbles than Hyukjin
axiom marbles_condition : S - given_marbles = (H + given_marbles) + 477

theorem seungho_more_marbles (S H : ℕ) (marbles_condition : S - 273 = (H + 273) + 477) : S = H + 1023 :=
by
  sorry

end seungho_more_marbles_l735_735306


namespace smallest_x_satisfies_palindrome_l735_735711

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

theorem smallest_x_satisfies_palindrome :
  ∃ x : ℕ, x > 0 ∧ x + 7654 = 7677 :=
by
  use 23
  split
  · exact Nat.zero_lt_succ 22
  · norm_num

end smallest_x_satisfies_palindrome_l735_735711


namespace cindy_pens_ratio_is_one_l735_735726

noncomputable def pens_owned_initial : ℕ := 25
noncomputable def pens_given_by_mike : ℕ := 22
noncomputable def pens_given_to_sharon : ℕ := 19
noncomputable def pens_owned_final : ℕ := 75

def pens_before_cindy (initial_pens mike_pens : ℕ) : ℕ := initial_pens + mike_pens
def pens_before_sharon (final_pens sharon_pens : ℕ) : ℕ := final_pens + sharon_pens
def pens_given_by_cindy (pens_before_sharon pens_before_cindy : ℕ) : ℕ := pens_before_sharon - pens_before_cindy
def ratio_pens_given_cindy (cindy_pens pens_before_cindy : ℕ) : ℚ := cindy_pens / pens_before_cindy

theorem cindy_pens_ratio_is_one :
    ratio_pens_given_cindy
        (pens_given_by_cindy (pens_before_sharon pens_owned_final pens_given_to_sharon)
                             (pens_before_cindy pens_owned_initial pens_given_by_mike))
        (pens_before_cindy pens_owned_initial pens_given_by_mike) = 1 := by
    sorry

end cindy_pens_ratio_is_one_l735_735726


namespace part1_part2_l735_735931

variable (A B C E F P : Type)
variable [AcuteTriangle A B C]
variable [ABG : ∃ g : Real, AB g A B] [αB : (A, B) > (A, C)]
variable [CosSum : (cos B) + (cos C) = 1]
variable [RightAngleABF : ∀ (B: Type),  RightAngle (AB F)]
variable [RightAngleACE : ∀ (C: Type),  RightAngle (AC E)]
variable [OnExtensionLineE : ∀ (E: Type), (OnExtensionLine AB E) ]
variable [OnExtensionLineF : ∀ (F: Type), (OnExtensionLine AC F) ]

theorem part1 : BE + CF = EF := sorry
theorem part2 (BisectorEBC : (AngleBisector E B C P)) : (AngleBisector C P (AngleBCF)) := sorry

end part1_part2_l735_735931


namespace part_I_part_II_l735_735606

def C1_parametric_to_cartesian (phi : ℝ) : ℝ × ℝ :=
  let x := 2 + 2 * Real.cos phi
      y := 2 * Real.sin phi
  (x, y)

def C2_polar_to_cartesian (theta : ℝ) : ℝ :=
  let ρ := 4 * Real.sin theta
  ρ

theorem part_I :
  (∀ (phi : ℝ),
    C1_parametric_to_cartesian phi = (2 + 2 * Real.cos phi, 2 * Real.sin phi)) ∧
  (∀ (theta : ℝ), C2_polar_to_cartesian theta = 4 * Real.sin theta) :=
by sorry

def C1_cartesian_to_polar (theta : ℝ) : ℝ :=
  4 * Real.cos theta

theorem part_II (α : ℝ) (A B : ℝ × ℝ) (h1 : 0 < α ∧ α < π) (h2 : |A - B| = (4 * sqrt 2)): 
  α = 3 * π / 4 :=
by sorry

end part_I_part_II_l735_735606


namespace min_val_N_l735_735315

theorem min_val_N (N : ℕ) (a b : ℕ) (h1 : 100 ≤ N ∧ N < 1000)
    (h2 : N % 7 = 0)
    (h3 : let N' := 10 * (N / 100) + (N % 10) in N' % 7 = 0) :
    N = 154 :=
by
  sorry

end min_val_N_l735_735315


namespace min_loss_expense_income_l735_735095

def arithmetic_progression (a d : ℕ) (n : ℕ) : ℕ := a + n * d

theorem min_loss_expense_income :
  let P := arithmetic_progression 1 1 0,
      A := arithmetic_progression 1 1 1,
      C := arithmetic_progression 1 1 2,
      X := arithmetic_progression 1 1 3,
      O := arithmetic_progression 1 1 4,
      Д := arithmetic_progression 1 1 5,
      EXPENSE := P * 100000 + A * 10000 + C * 1000 + X * 100 + O * 10 + Д,
      INCOME := A * 10000 + C * 1000 + X * 100 + O * 10 + Д
  in EXPENSE - INCOME = 58000 :=
  by
  sorry

end min_loss_expense_income_l735_735095


namespace sum_of_integer_solutions_to_equation_l735_735100

theorem sum_of_integer_solutions_to_equation :
  (∑ x in {x : ℤ | x^4 - 13 * x^2 + 36 = 0}.toFinset, x) = 0 :=
by
  sorry

end sum_of_integer_solutions_to_equation_l735_735100


namespace book_purchasing_options_l735_735782

theorem book_purchasing_options :
  ∃ (n : ℕ), n = 4 ∧ 3.choose 2 + 1 = n :=
begin
  use 4,
  split,
  { refl },
  { sorry }
end

end book_purchasing_options_l735_735782


namespace complex_root_expression_l735_735261

noncomputable def alpha : ℂ := sorry -- Nonreal root of x^4 = 1

theorem complex_root_expression : (1 - alpha + alpha^2 - alpha^3)^4 + (1 + alpha - alpha^2 + alpha^3)^4 = 32 :=
by
  have h : alpha^4 = 1 := sorry -- Given root condition
  have nonreal : ¬ isReal alpha := sorry -- α is non-real
  sorry

end complex_root_expression_l735_735261


namespace num_non_congruent_triangles_with_perimeter_12_l735_735562

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter_12_integer_triangles : list (ℕ × ℕ × ℕ) :=
  [(2, 5, 5), (3, 4, 5), (4, 4, 4)]

theorem num_non_congruent_triangles_with_perimeter_12 :
  (filter (λ t, is_valid_triangle t.1 t.2 t.3) perimeter_12_integer_triangles).length = 3 := 
sorry

end num_non_congruent_triangles_with_perimeter_12_l735_735562


namespace beads_problem_l735_735045

theorem beads_problem :
  ∃ b : ℕ, (b % 6 = 5) ∧ (b % 8 = 3) ∧ (b % 9 = 7) ∧ (b = 179) :=
by
  sorry

end beads_problem_l735_735045


namespace eval_g_pi_l735_735999

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)
def shift_left (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x, f (x + a)
def stretch_horizontal (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x, f (k * x)
def g := stretch_horizontal (shift_left f (Real.pi / 3)) (1 / 2)

theorem eval_g_pi : g π = - Real.sqrt 3 / 2 := by
  sorry

end eval_g_pi_l735_735999


namespace probability_sine_interval_l735_735603

noncomputable def sine_probability (x : ℝ) (hx : x ∈ Icc 0 real.pi) : Prop :=
  let lengths := [real.pi / 6, real.pi / 6]
  let total_length := real.pi
  (len_sum := (lengths.sum = real.pi / 3))
  probability_eq_one_third : ((real.pi / 3) / total_length) = (1 / 3)

theorem probability_sine_interval :
  ∀ x ∈ Icc 0 real.pi, sine_probability x = (1 / 3) := by
  sorry

end probability_sine_interval_l735_735603


namespace circle_tangent_to_line_at_point_A_l735_735327

theorem circle_tangent_to_line_at_point_A :
  ∃ b, (center : ℝ) (radius : ℝ), center = (0, b) ∧ radius = sqrt (4 + (b - 2)^2) ∧
    (∀ p : ℝ × ℝ, (p.1, p.2) = (2, 2) → 2 * p.1 + 3 * p.2 - 10 = 0 → distance (p.1, p.2) (center.1, center.2) = radius) ∧
    (circle_eq : p.1 ^ 2 + (p.2 + b) ^ 2 = 13) :=
begin
  use -1,
  use (0, -1),
  use 2 * real.sqrt 13,
  sorry
end

end circle_tangent_to_line_at_point_A_l735_735327


namespace a_eq_0_iff_b_eq_0_l735_735385

noncomputable def polynomial_with_equal_roots_abs
  (a b c : ℝ)
  (p q r : ℂ)
  (h_pq_ineq : |p| = |q|)
  (h_qr_ineq : |q| = |r|)
  (h_roots : p * q * r = -c) :=
{ polynomial := ∀ x : ℂ, x^3 + a * x^2 + b * x + c,
  root_p := polynomial p = 0,
  root_q := polynomial q = 0,
  root_r := polynomial r = 0,
  pq_plus_qr_plus_rp := p * q + q * r + r * p
}

theorem a_eq_0_iff_b_eq_0
  (a b c : ℝ)
  (p q r : ℂ)
  (h_pq_ineq : |p| = |q|)
  (h_qr_ineq : |q| = |r|)
  (h_pq_plus_qr_plus_rp : p * q + q * r + r * p = b)
  (h_p_plus_q_plus_r : p + q + r = -a) :
  (a = 0 ↔ b = 0) :=
sorry

end a_eq_0_iff_b_eq_0_l735_735385


namespace age_ratio_l735_735113

-- Definitions as per the conditions
variable (j e x : ℕ)

-- Conditions from the problem
def condition1 : Prop := j - 4 = 2 * (e - 4)
def condition2 : Prop := j - 10 = 3 * (e - 10)

-- The statement we need to prove
theorem age_ratio (j e x : ℕ) (h1 : condition1 j e)
(h2 : condition2 j e) :
(j + x) * 2 = (e + x) * 3 ↔ x = 8 :=
sorry

end age_ratio_l735_735113


namespace unique_solution_l735_735086

theorem unique_solution (m n : ℕ) (h1 : n^4 ∣ 2 * m^5 - 1) (h2 : m^4 ∣ 2 * n^5 + 1) : m = 1 ∧ n = 1 :=
by
  sorry

end unique_solution_l735_735086


namespace convex_polygon_sides_equal_to_longest_diagonal_l735_735567

open Polygon

theorem convex_polygon_sides_equal_to_longest_diagonal (P : Polygon) (hconvex : P.is_convex) 
    (hsize : P.n ≥ 3) :
  ∃ k : ℕ, k ∈ {0, 1, 2} ∧ ∀ (d : ℕ), is_longest_diagonal P d → count (λ s, s = d) P.sides = k :=
sorry

end convex_polygon_sides_equal_to_longest_diagonal_l735_735567


namespace squirrel_count_l735_735296

theorem squirrel_count (n m : ℕ) (h1 : n = 12) (h2 : m = 12 + 12 / 3) : n + m = 28 := by
  sorry

end squirrel_count_l735_735296


namespace correct_statements_l735_735003

def problem_statements :=
  [ "The negation of the statement 'There exists an x ∈ ℝ such that x^2 - 3x + 3 = 0' is true.",
    "The statement '-1/2 < x < 0' is a necessary but not sufficient condition for '2x^2 - 5x - 3 < 0'.",
    "The negation of the statement 'If xy = 0, then at least one of x or y is equal to 0' is true.",
    "The curves x^2/25 + y^2/9 = 1 and x^2/(25 − k) + y^2/(9 − k) = 1 (9 < k < 25) share the same foci.",
    "There exists a unique line that passes through the point (1,3) and is tangent to the parabola y^2 = 4x."
  ]

theorem correct_statements :
  (∀ x : ℝ, ¬(x^2 - 3 * x + 3 = 0)) ∧ 
  ¬ (¬-1/2 < x ∧ x < 0 → 2 * x^2 - 5*x - 3 < 0) ∧ 
  (∀ x y : ℝ, xy ≠ 0 → x ≠ 0 ∧ y ≠ 0) ∧ 
  (∀ k : ℝ, 9 < k ∧ k < 25 → ∀ x y : ℝ, (x^2 / (25 - k) + y^2 / (9 - k) = 1) → (x^2 / 25 + y^2 / 9 = 1) → (x ≠ 0 ∨ y ≠ 0)) ∧ 
  ¬ (∃ l : ℝ, ∀ pt : ℝ × ℝ, pt = (1, 3) → ∀ y : ℝ, y^2 = 4 * pt.1 → y = 2 * pt.2)
:= 
  sorry

end correct_statements_l735_735003


namespace star_computation_l735_735469

-- Define the operation X \star Y
def star (X Y : ℝ) : ℝ := (X + Y) / 4

-- Statement to prove that (3 \star 11) \star 7 = 2.625
theorem star_computation : (star (star 3 11) 7) = 2.625 := 
by
  sorry

end star_computation_l735_735469


namespace initial_friends_l735_735759

theorem initial_friends (added_friends total_friends initial_friends : ℕ)
  (h1 : added_friends = 3)
  (h2 : total_friends = 7)
  (h3 : total_friends = initial_friends + added_friends) : initial_friends = 4 :=
by
  rw [h2, h1, h3]
  sorry

end initial_friends_l735_735759


namespace distance_l1_l2_l735_735520

noncomputable def distance_between_parallel_lines
  (l1 : ℝ → ℝ → Prop) (l2 : ℝ → ℝ → Prop) : ℝ :=
  -- Definitions of the lines according to their equations
  let l1 := ∀ x y : ℝ, 2 * x + 3 * 2 * y - 2 + 2 = 0 in
  let l2 := ∀ x y : ℝ, 2 * x + 6 * y - 4 = 0 in
  -- Calculation of the distance between two parallel lines
  (abs (0 - (-2))) / (sqrt ((1 ^ 2) + (3 ^ 2)))

-- The theorem statement for proving the distance
theorem distance_l1_l2 :
  let l1 := ∀ x y : ℝ, 2 * x + 3 * 2 * y - 2 + 2 = 0 in
  let l2 := ∀ x y : ℝ, 2 * x + 6 * y - 4 = 0 in
  (abs (0 - (-2))) / (sqrt ((1 ^ 2) + (3 ^ 2))) = (√10 / 5) :=
by sorry


end distance_l1_l2_l735_735520


namespace relationship_among_log_exp_l735_735846

theorem relationship_among_log_exp (
  a : ℝ := Real.log 0.8 / Real.log 1.2,
  b : ℝ := Real.log 0.8 / Real.log 0.7,
  c : ℝ := 1.2 ^ 0.8
) : a < b ∧ b < c :=
by
  sorry  -- to be proven

end relationship_among_log_exp_l735_735846


namespace quadratic_discriminant_l735_735172

variable {a b c : ℝ}
variable (h₁ : a ≠ 0)
variable (h₂ : (b - 1)^2 - 4 * a * (c + 2) = 0)
variable (h₃ : (b + 1/2)^2 - 4 * a * (c - 1) = 0)

theorem quadratic_discriminant : b^2 - 4 * a * c = -1 / 2 := 
by
  have h₁' : (b - 1)^2 - 4 * a * (c + 2) = 0 := h₂
  have h₂' : (b + 1/2)^2 - 4 * a * (c - 1) = 0 := h₃
  sorry

end quadratic_discriminant_l735_735172


namespace number_of_elements_in_S_inter_even_l735_735509

variables (F : Type) [field F] [fintype F] (p : ℕ) (h_prime : nat.prime p) (h_card : fintype.card F = p^2)

def S (F : Type) [field F] [fintype F] : finset F :=
finset.filter (λ x, x ≠ 0 ∧ (¬(∃ a, x = a) ∨ (∃ b, x = -b))) (finset.univ : finset F)

theorem number_of_elements_in_S_inter_even (F : Type) [field F] [fintype F] (p : ℕ) [fact (nat.prime p)] (h_card : fintype.card F = p^2) :
  ∃ S : finset F, S.card = (p^2 - 1) / 2 ∧ (∀ a ∈ S, -a ∉ S) → nat.even (finset.card (finset.inter S (finset.image (λ x, 2 * x) S))) :=
begin
  intros S h_S_card h_S_condition,
  -- Proof to be added
  sorry
end

end number_of_elements_in_S_inter_even_l735_735509


namespace quadratic_discriminant_l735_735168

variable {a b c : ℝ}
variable (h₁ : a ≠ 0)
variable (h₂ : (b - 1)^2 - 4 * a * (c + 2) = 0)
variable (h₃ : (b + 1/2)^2 - 4 * a * (c - 1) = 0)

theorem quadratic_discriminant : b^2 - 4 * a * c = -1 / 2 := 
by
  have h₁' : (b - 1)^2 - 4 * a * (c + 2) = 0 := h₂
  have h₂' : (b + 1/2)^2 - 4 * a * (c - 1) = 0 := h₃
  sorry

end quadratic_discriminant_l735_735168


namespace length_segment_pq_l735_735989

theorem length_segment_pq 
  (P Q R S T : ℝ)
  (h1 : (dist P Q + dist P R + dist P S + dist P T = 67))
  (h2 : (dist Q P + dist Q R + dist Q S + dist Q T = 34)) :
  dist P Q = 11 :=
sorry

end length_segment_pq_l735_735989


namespace vowels_written_correctly_l735_735792

theorem vowels_written_correctly :
  let vowels := 5 in
  let times := 2 in
  vowels * times = 10 :=
by
  let vowels := 5
  let times := 2
  show vowels * times = 10
  sorry

end vowels_written_correctly_l735_735792


namespace min_red_up_probability_card_l735_735353

theorem min_red_up_probability_card (cards : Fin 50) :
  (cards = 13) ∨ (cards = 38) ↔
  -- Conditions for Vasya and Asya's actions:
  ∃ (select_vasya : Fin 26 → Fin 50) (select_asya : Fin 26 → Fin 50),
    (∀ k : Fin 50, 
      probability.card_pos (select_vasya k.left ≤ k.to_nat ∧ k.to_nat < k.left + 25 → 
                            select_asya k.right ≤ k.to_nat ∧ k.to_nat < k.right + 25 → 
                            k.val = 13 ∨ k.val = 38))

end min_red_up_probability_card_l735_735353


namespace complex_expr_eq_zero_l735_735831

noncomputable def complex_expr : ℂ := (1 + complex.i) / (1 - complex.i) + complex.i^3

theorem complex_expr_eq_zero : complex_expr = 0 := 
by
  sorry

end complex_expr_eq_zero_l735_735831


namespace sum_abs_roots_l735_735102

theorem sum_abs_roots (P : Polynomial ℂ) (hP : P = Polynomial.C (1 : ℂ) * X^4 - Polynomial.C (6 : ℂ) * X^3 - Polynomial.C (3 : ℂ) * X^2 + Polynomial.C (18 : ℂ) * X - Polynomial.C (9 : ℂ)) :
  (∑ r in P.roots.to_finset, Complex.abs r) = 6 := 
sorry

end sum_abs_roots_l735_735102


namespace mu_bounds_l735_735506

noncomputable def mu (x y : ℝ) := (sin y) + (cos x)^2

theorem mu_bounds (x y : ℝ) (h1 : sin x + sin y = 1 / 3)
  (hx : -1 ≤ sin x ∧ sin x ≤ 1) (hy : -1 ≤ sin y ∧ sin y ≤ 1) :
  -2/3 ≤ mu x y ∧ mu x y ≤ 19/12 := 
by
  sorry

end mu_bounds_l735_735506


namespace third_integer_ordered_l735_735577

theorem third_integer_ordered (l : List Int) (h : l = [-7, 10, 9, 0, -9]) :
    (l.qsort (· < ·)).nthLe 2 (by decide) = 0 := by sorry

end third_integer_ordered_l735_735577


namespace value_range_x_sq_minus_2x_l735_735696

theorem value_range_x_sq_minus_2x (x : ℝ) (hx : 0 ≤ x) : ∃ y : ℝ, y = x^2 - 2 * x ∧ y ∈ set.Ici (-1) :=
by
  sorry

end value_range_x_sq_minus_2x_l735_735696


namespace infinite_points_in_circle_radius_one_sum_squares_three_l735_735966

variables {A B P : ℝ × ℝ}

/--
The number of points P inside a circle with radius 1 such that the sum
of squares of the distances from P to the endpoints A and B of a given
diameter is 3, is infinite.
-/
theorem infinite_points_in_circle_radius_one_sum_squares_three
  {radius : ℝ} (h_radius : radius = 1)
  (hA : A = (-1, 0))
  (hB : B = (1, 0))
  (h_distance : ∀ P : ℝ × ℝ, P.1^2 + P.2^2 < radius^2)
  (h_sum_squares : ∀ P : ℝ × ℝ, ((P.1 - A.1)^2 + (P.2 - A.2)^2) + ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 3) :
  ∃∞ P : ℝ × ℝ, P ∈ {P : ℝ × ℝ | (P.1)^2 + (P.2)^2 < radius^2 ∧ ((P.1 - A.1)^2 + (P.2 - A.2)^2) + ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 3} :=
sorry

end infinite_points_in_circle_radius_one_sum_squares_three_l735_735966


namespace circle_eq_l735_735694

theorem circle_eq (x y : ℝ) (h k r : ℝ) (hc : h = 3) (kc : k = 1) (rc : r = 5) :
  (x - h)^2 + (y - k)^2 = r^2 ↔ (x - 3)^2 + (y - 1)^2 = 25 :=
by
  sorry

end circle_eq_l735_735694


namespace find_lambda_l735_735899

-- Define the vectors a and b
def vec_a : ℝ × ℝ × ℝ := (-2, 1, 3)
def vec_b : ℝ × ℝ × ℝ := (-1, 2, 1)

-- Define the dot product of two vectors
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Given condition: a is orthogonal to (a - λb)
def orthogonal_condition (λ : ℝ) : Prop :=
  dot_product vec_a (vec_a.1 + λ * vec_b.1, vec_a.2 - 2 * λ * vec_b.2, vec_a.3 - λ * vec_b.3) = 0

-- Prove that λ = 2 satisfies the orthogonal condition
theorem find_lambda : ∃ λ : ℝ, orthogonal_condition λ :=
begin
  use 2,
  sorry
end

end find_lambda_l735_735899


namespace square_center_in_triangle_or_boundary_l735_735433

open Set

-- Define the problem conditions.
structure Triangle (α : Type _) :=
(a b c : α)
(len_a : a ≥ 1)
(len_b : b ≥ 1)
(len_c : c ≥ 1)

-- Define the square properties.
structure Square (α : Type _) :=
(center : α)
(side_length : ℝ)

theorem square_center_in_triangle_or_boundary 
  {α : Type _} [LinearOrderedField α] [TopologicalSpace α] [AffineSpace α ℝ] 
  (T : Triangle α) 
  (S : Square α)
  (side_length_eq_one : S.side_length = 1)
  (triangle_inside_square : ∀ p ∈ {T.a, T.b, T.c}, p ∈ S) :
  S.center ∈ ConvexHull ℝ {T.a, T.b, T.c} :=
sorry

end square_center_in_triangle_or_boundary_l735_735433


namespace no_triangle_with_perfect_square_sides_l735_735075

theorem no_triangle_with_perfect_square_sides :
  ∃ (a b : ℕ), a > 1000 ∧ b > 1000 ∧
    ∀ (c : ℕ), (∃ d : ℕ, c = d^2) → 
    ¬ (a + b > c ∧ b + c > a ∧ a + c > b) :=
sorry

end no_triangle_with_perfect_square_sides_l735_735075


namespace sum_of_integer_solutions_l735_735098

theorem sum_of_integer_solutions (x : ℤ) :
    (x^4 - 13 * x^2 + 36 = 0) →
    (∃ s : ℤ, s = ∑ i in {2, -2, 3, -3}, i) ∧ s = 0 :=
by
  sorry

end sum_of_integer_solutions_l735_735098


namespace grades_calculation_l735_735342

-- Defining the conditions
def total_students : ℕ := 22800
def students_per_grade : ℕ := 75

-- Stating the theorem to be proved
theorem grades_calculation : total_students / students_per_grade = 304 := sorry

end grades_calculation_l735_735342


namespace prove_inequality_l735_735970

noncomputable def inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : Prop :=
    Real.root 4 (x * y * z * (x + y + z) / 3) ≤ Real.root 3 ((x + y) / 2 * (y + z) / 2 * (z + x) / 2)

theorem prove_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : inequality_proof x y z hx hy hz :=
    sorry

end prove_inequality_l735_735970


namespace triangle_area_ratio_l735_735945

variables {V : Type*} [inner_product_space ℝ V] (A B C Q : V) 

def point_Q_condition (A B C Q : V) : Prop :=
  A - Q + 3 * (B - Q) + 4 * (C - Q) = 0

def area_ratio (ABC AQC : ℝ) : Prop :=
  ABC / AQC = 3

theorem triangle_area_ratio (ABC AQC : ℝ) (A B C Q : V)
  (hQ : point_Q_condition A B C Q) : 
  area_ratio ABC AQC :=
sorry

end triangle_area_ratio_l735_735945


namespace solve_equation_121_x_squared_plus_54_eq_0_l735_735489

theorem solve_equation_121_x_squared_plus_54_eq_0 :
  ∃ x : ℂ, (121 * x^2 + 54 = 0) ↔ (x = 3 * complex.sqrt 6 * complex.I / 11 ∨ x = -3 * complex.sqrt 6 * complex.I / 11) :=
sorry

end solve_equation_121_x_squared_plus_54_eq_0_l735_735489


namespace sum_of_squares_inequality_l735_735647

theorem sum_of_squares_inequality (n : ℕ) (hn : 0 < n) :
  ∑ i in Finset.range (n + 1), (1 / (i + 1) ^ 2 : ℝ) < (2 * n + 1) / (n + 1) := by
  sorry

end sum_of_squares_inequality_l735_735647


namespace jason_earnings_after_school_l735_735614

variable (earnings_sat_per_hour : ℝ) (total_hours : ℝ)
variable (total_earnings : ℝ) (hours_sat : ℝ)

theorem jason_earnings_after_school :
  let earnings_after_school := total_earnings - hours_sat * earnings_sat_per_hour in
  let hours_after_school := total_hours - hours_sat in
  let rate_after_school := earnings_after_school / hours_after_school in
  earnings_sat_per_hour = 6 →
  total_hours = 18 →
  total_earnings = 88 →
  hours_sat = 8 →
  rate_after_school = 4 :=
by
  intros
  sorry

end jason_earnings_after_school_l735_735614


namespace symmetric_point_x_axis_l735_735609

-- Define the coordinates of point A
def pointA : ℝ × ℝ × ℝ := (1, 2, 1)

-- Define the symmetry transformation with respect to the x-axis
def symmetric_x_axis (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, p.2, -p.3)

-- Prove that the symmetric point is as stated
theorem symmetric_point_x_axis :
  symmetric_x_axis pointA = (1, 2, -1) :=
by
  simp [pointA, symmetric_x_axis]
  exact rfl

end symmetric_point_x_axis_l735_735609


namespace probability_of_drawing_ball_1_is_2_over_5_l735_735754

noncomputable def probability_of_drawing_ball_1 : ℚ :=
  let total_balls := [1, 2, 3, 4, 5]
  let draw_pairs := [ (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5) ]
  let favorable_pairs := [ (1, 2), (1, 3), (1, 4), (1, 5) ]
  (favorable_pairs.length : ℚ) / (draw_pairs.length : ℚ)

theorem probability_of_drawing_ball_1_is_2_over_5 :
  probability_of_drawing_ball_1 = 2 / 5 :=
by sorry

end probability_of_drawing_ball_1_is_2_over_5_l735_735754


namespace minimum_perimeter_of_polygon_formed_by_Q_zeros_l735_735630

-- Define the polynomial Q(z)
noncomputable def Q (z : ℂ) : ℂ :=
  z^8 + (3*real.sqrt 3 + 5)*z^4 - (3*real.sqrt 3 + 8)

-- State the theorem about the minimum perimeter of the 8-sided polygon formed by the zeros of Q(z)
theorem minimum_perimeter_of_polygon_formed_by_Q_zeros :
  let zeros := (λ (z : ℂ), Q(z) = 0) in
  let distances := (λ (z1 z2 : ℂ), abs (z1 - z2)) in
  let perimeter := ∑ (i : fin 8), distances (zeros i) (zeros ((i + 1) % 8)) in
  perimeter = 8 * real.sqrt 2 :=
sorry

end minimum_perimeter_of_polygon_formed_by_Q_zeros_l735_735630


namespace solve_riccati_eqn_l735_735666

noncomputable def riccati_solution (C : ℝ) : ℝ → ℝ :=
  λ x, real.exp x - (1 / (x + C))

theorem solve_riccati_eqn (C : ℝ) :
  ∀ y, (∃ x, y = real.exp x + (real.exp x - (1 / (x + C)))) →
       (deriv y - y^2 + 2 * real.exp x * y = real.exp (2 * x) + real.exp x) :=
begin
  sorry
end

end solve_riccati_eqn_l735_735666


namespace eccentricity_of_hyperbola_l735_735673

noncomputable def hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 1)
  (h4 : ∃ (A B : ℝ × ℝ), (x - 1)^2 + y^2 = a^2 ∧ abs (A - B) = b)
  : ℝ :=
3 * sqrt 5 / 5

theorem eccentricity_of_hyperbola (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 1)
  (hyp : ∃ A B : ℝ × ℝ, ((A.1 - 1)^2 + A.2^2 = a^2) ∧ ((B.1 - 1)^2 + B.2^2 = a^2) ∧
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 = b^2))
  : hyperbola_eccentricity a b h1 h2 h3 hyp = 3 * sqrt 5 / 5 :=
by sorry

end eccentricity_of_hyperbola_l735_735673


namespace curve_intersection_condition_l735_735719

theorem curve_intersection_condition (a : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = a^2 ∧ y = x^2 - 1 ∧
  (∀ x1 y1 x2 y2, (x1, y1) ≠ (x2, y2) → x1^2 + y1^2 = a^2 ∧ y1 = x1^2 - 1 ∧ x2^2 + y2^2 = a^2 ∧ y2 = x2^2 - 1)) ↔ a ≥ Real.sqrt 2 :=
begin
  sorry
end

end curve_intersection_condition_l735_735719


namespace find_surface_area_of_ball_l735_735173

noncomputable def surface_area_of_ball : ℝ :=
  let tetrahedron_edge := 4
  let tetrahedron_volume := (1 / 3) * (Real.sqrt 3 / 4) * tetrahedron_edge ^ 2 * (Real.sqrt 16 - 16 / 3)
  let water_volume := (7 / 8) * tetrahedron_volume
  let remaining_volume := (1 / 8) * tetrahedron_volume
  let remaining_edge := 2
  let ball_radius := Real.sqrt 6 / 6
  let surface_area := 4 * Real.pi * ball_radius ^ 2
  surface_area

theorem find_surface_area_of_ball :
  let tetrahedron_edge := 4
  let tetrahedron_volume := (1 / 3) * (Real.sqrt 3 / 4) * tetrahedron_edge ^ 2 * (Real.sqrt 16 - 16 / 3)
  let water_volume := (7 / 8) * tetrahedron_volume
  let remaining_volume := (1 / 8) * tetrahedron_volume
  let remaining_edge := 2
  let ball_radius := Real.sqrt 6 / 6
  let surface_area := 4 * Real.pi * ball_radius ^ 2
  surface_area = (2 / 3) * Real.pi :=
by
  let tetrahedron_edge := 4
  let tetrahedron_volume := (1 / 3) * (Real.sqrt 3 / 4) * tetrahedron_edge ^ 2 * (Real.sqrt 16 - 16 / 3)
  let water_volume := (7 / 8) * tetrahedron_volume
  let remaining_volume := (1 / 8) * tetrahedron_volume
  let remaining_edge := 2
  let ball_radius := Real.sqrt 6 / 6
  let surface_area := 4 * Real.pi * ball_radius ^ 2
  sorry

end find_surface_area_of_ball_l735_735173


namespace linear_system_reduction_transformation_l735_735718

theorem linear_system_reduction_transformation :
  ∀ (use_substitution_or_elimination : Bool), 
    (use_substitution_or_elimination = true) ∨ (use_substitution_or_elimination = false) → 
    "Reduction and transformation" = "Reduction and transformation" :=
by
  intro use_substitution_or_elimination h
  sorry

end linear_system_reduction_transformation_l735_735718


namespace quadratic_discriminant_l735_735167

variable {a b c : ℝ}
variable (h₁ : a ≠ 0)
variable (h₂ : (b - 1)^2 - 4 * a * (c + 2) = 0)
variable (h₃ : (b + 1/2)^2 - 4 * a * (c - 1) = 0)

theorem quadratic_discriminant : b^2 - 4 * a * c = -1 / 2 := 
by
  have h₁' : (b - 1)^2 - 4 * a * (c + 2) = 0 := h₂
  have h₂' : (b + 1/2)^2 - 4 * a * (c - 1) = 0 := h₃
  sorry

end quadratic_discriminant_l735_735167


namespace positive_integer_sum_l735_735490

theorem positive_integer_sum (n : ℤ) :
  (n > 0) ∧
  (∀ stamps_cannot_form : ℤ, (∀ a b c : ℤ, 7 * a + n * b + (n + 2) * c ≠ 120) ↔
  ¬ (0 ≤ 7*a ∧ 7*a ≤ 120 ∧ 0 ≤ n*b ∧ n*b ≤ 120 ∧ 0 ≤ (n + 2)*c ∧ (n + 2)*c ≤ 120)) ∧
  (∀ postage_formed : ℤ, (120 < postage_formed ∧ postage_formed ≤ 125 →
  ∃ a b c : ℤ, 7 * a + n * b + (n + 2) * c = postage_formed)) →
  n = 21 :=
by {
  -- proof omittted 
  sorry
}

end positive_integer_sum_l735_735490


namespace aubrey_tomatoes_l735_735448

theorem aubrey_tomatoes :
  (∀ r, r % 3 = 0 → r / 3 * 8 * 3 = 120) :=
by
  intros r hrmod hrdiv
  sorry

end aubrey_tomatoes_l735_735448


namespace range_of_a_l735_735510

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + 2 * x - Real.exp x

theorem range_of_a (h : ∀ m n : ℝ, 0 < m → 0 < n → m > n → (f a m - f a n) / (m - n) < 2) :
  a ≤ Real.exp 1 / (2 * 1) := 
sorry

end range_of_a_l735_735510


namespace transformed_data_properties_l735_735033

-- Definitions of the initial mean and variance
def initial_mean : ℝ := 2.8
def initial_variance : ℝ := 3.6

-- Definitions of transformation constants
def multiplier : ℝ := 2
def increment : ℝ := 60

-- New mean after transformation
def new_mean : ℝ := multiplier * initial_mean + increment

-- New variance after transformation
def new_variance : ℝ := (multiplier ^ 2) * initial_variance

-- Theorem statement
theorem transformed_data_properties :
  new_mean = 65.6 ∧ new_variance = 14.4 :=
by
  sorry

end transformed_data_properties_l735_735033


namespace problem_l735_735960

def f (n : ℕ) : ℕ :=
  List.prod (List.map (λ i, 2 * i + 1) (List.range n))

theorem problem : (List.sum (List.map f (List.range 2016.succ)) % 100 = 74) :=
  by
  sorry

end problem_l735_735960


namespace maximize_profit_l735_735946

def fixed_cost := 40 -- in million dollars
def variable_cost (x : ℝ) := 0.16 * x -- x is in thousand units, variable cost in million dollars
def revenue (x : ℝ) : ℝ :=
  if x ≤ 40 then 400 - 6 * x 
  else (7400 / x) - (40000 / x^2)

def profit (x : ℝ) : ℝ :=
  if x ≤ 40 then
    x * revenue x - (16 * x + fixed_cost)
  else
    x * revenue x - (16 * x + fixed_cost)

theorem maximize_profit :
  ∃ x : ℝ, x = 32 ∧ ∀ y : ℝ, 0 < y → profit y ≤ 6104 :=
by sorry

end maximize_profit_l735_735946


namespace expenses_notation_l735_735322

theorem expenses_notation (income expense : ℤ) (h_income : income = 6) (h_expense : -expense = income) : expense = -4 := 
by
  sorry

end expenses_notation_l735_735322


namespace tank_cost_minimization_l735_735016

def volume := 4800
def depth := 3
def cost_per_sqm_bottom := 150
def cost_per_sqm_walls := 120

theorem tank_cost_minimization (x : ℝ) 
  (S1 : ℝ := volume / depth)
  (S2 : ℝ := 6 * (x + (S1 / x)))
  (cost := cost_per_sqm_bottom * S1 + cost_per_sqm_walls * S2) :
  (x = 40) → cost = 297600 :=
sorry

end tank_cost_minimization_l735_735016


namespace line_AB_passes_through_fixed_point_l735_735859

theorem line_AB_passes_through_fixed_point 
    (C : set (ℝ × ℝ)) 
    (P : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ p ∈ C, p.1^2 + p.2^2 = 4)
    (hP : P.1 + 2 * P.2 = 9)
    (tangent_PA : (A ∈ C) ∧ (¬ ∃ k : ℝ, P.1 = A.1 + k * (P.1 - A.1) ∧ P.2 = A.2 + k * (P.2 - A.2)))
    (tangent_PB : (B ∈ C) ∧ (¬ ∃ k : ℝ, P.1 = B.1 + k * (P.1 - B.1) ∧ P.2 = B.2 + k * (P.2 - B.2))) :
    ∃ O : ℝ × ℝ, 
    (O = (4 / 9, 8 / 9)) ∧
    (∀ m : ℝ, P = (9 - 2 * m, m) → (P.1 + (2 * P.2) = 9) →
        (∃ O ∈ (set (ℝ × ℝ)), line_through A B O)) :=
by
    sorry

end line_AB_passes_through_fixed_point_l735_735859


namespace Mike_given_total_cookies_l735_735383

-- All given conditions
variables (total Tim fridge Mike Anna : Nat)
axiom h1 : total = 256
axiom h2 : Tim = 15
axiom h3 : fridge = 188
axiom h4 : Anna = 2 * Tim
axiom h5 : total = Tim + Anna + fridge + Mike

-- The goal of the proof
theorem Mike_given_total_cookies : Mike = 23 :=
by
  sorry

end Mike_given_total_cookies_l735_735383


namespace cards_least_likely_red_after_flips_l735_735350

theorem cards_least_likely_red_after_flips :
  ∃ (k1 k2 : ℕ), 1 ≤ k1 ∧ k1 ≤ 50 ∧ 1 ≤ k2 ∧ k2 ≤ 50 ∧ (k1 = 13 ∧ k2 = 38) ∧ 
  (∀ k ∈ finset.range 1 51, 
    let p := (if k ≤ 25 then ((26 - k) ^ 2 + k ^ 2) / 676 else ((26 - (51 - k)) ^ 2 + (51 - k) ^ 2) / 676) in
    p ≥ (if k = 13 ∨ k = 38 then ((26 - k) ^ 2 + k ^ 2) / 676 else p)) :=
sorry

end cards_least_likely_red_after_flips_l735_735350


namespace juliette_and_marco_money_comparison_l735_735669

noncomputable def euro_to_dollar (eur : ℝ) : ℝ := eur * 1.5

theorem juliette_and_marco_money_comparison :
  (600 - euro_to_dollar 350) / 600 * 100 = 12.5 := by
sorry

end juliette_and_marco_money_comparison_l735_735669


namespace magistrate_arrangement_l735_735648

theorem magistrate_arrangement : 
  let E := 2, S := 2, W := 2, F := 1, I := 1, Sp := 1, A := 1 in
  let total := E + S + W + F + I + Sp + A in
  let factorial (n : Nat) : Nat :=
    if n = 0 then 1 else n * factorial (n - 1) in
  (E > 1 ∧ S > 1 ∧ W > 1 ∧ total = 10) →
  (factorial total - (3 * (factorial 9 * factorial 2) - 3 * (factorial 8 * (factorial 2 * factorial 2)) + (factorial 7 * (factorial 2 * factorial 2 * factorial 2))) = 1895040) :=
sorry

end magistrate_arrangement_l735_735648


namespace number_of_white_cats_l735_735231

theorem number_of_white_cats (total_cats : ℕ) (percent_black : ℤ) (grey_cats : ℕ) : 
  total_cats = 16 → 
  percent_black = 25 →
  grey_cats = 10 → 
  (total_cats - (total_cats * percent_black / 100 + grey_cats)) = 2 :=
by
  intros
  sorry

end number_of_white_cats_l735_735231


namespace length_of_bridge_correct_l735_735740

noncomputable def L_train : ℝ := 180
noncomputable def v_km_per_hr : ℝ := 60  -- speed in km/hr
noncomputable def t : ℝ := 25

-- Convert speed from km/hr to m/s
noncomputable def km_per_hr_to_m_per_s (v: ℝ) : ℝ := v * (1000 / 3600)
noncomputable def v : ℝ := km_per_hr_to_m_per_s v_km_per_hr

-- Distance covered by the train while crossing the bridge
noncomputable def d : ℝ := v * t

-- Length of the bridge
noncomputable def L_bridge : ℝ := d - L_train

theorem length_of_bridge_correct :
  L_bridge = 236.75 :=
  by
    sorry

end length_of_bridge_correct_l735_735740


namespace gcd_228_2008_l735_735333

theorem gcd_228_2008 : Int.gcd 228 2008 = 4 := by
  sorry

end gcd_228_2008_l735_735333


namespace power_function_point_l735_735542

theorem power_function_point (a : ℝ) (h : (2 : ℝ) ^ a = (1 / 2 : ℝ)) : a = -1 :=
by sorry

end power_function_point_l735_735542


namespace area_triangle_is_constant_equation_of_circle_l735_735557

-- Define the circle and its interaction with the axes
def circle_center (t : ℝ) (ht : t ≠ 0) : ℝ × ℝ := (t, 2 / t)

def circle_equation (t : ℝ) (ht : t ≠ 0) (x y : ℝ) : Prop :=
  (x - t)^2 + (y - 2 / t)^2 = t^2 + 4 / t^2

def intersects_x_axis (t : ℝ) (ht : t ≠ 0) (y : ℝ) : y = 0 ∨ y = 4 / t := sorry

def intersects_y_axis (t : ℝ) (ht : t ≠ 0) (x : ℝ) : x = 0 ∨ x = 2 * t := sorry

-- Prove that the area of the triangle is constant
theorem area_triangle_is_constant (t : ℝ) (ht : t ≠ 0) :
  (1 / 2) * 2 * t * 4 / t = 4 := sorry

-- Prove the equation of the circle given the line intersection conditions
theorem equation_of_circle (t : ℝ) (ht : t ≠ 0) (h_slope : ∀ x, (2 / t) = (1 / 2) * t)
   : (t = 2 ∨ t = -2) → (y = -2 * x + 4 → (x - 2)^2 + (y - 1)^2 = 5) := sorry

end area_triangle_is_constant_equation_of_circle_l735_735557


namespace sum_of_first_n_terms_general_formula_geometric_sequence_l735_735020

noncomputable def geometric_sequence := 
  ∃ a_n : ℕ → ℝ,
    (a_n 3) * (a_n 4) * (a_n 5) = 512 ∧ 
    (a_n 3) + (a_n 4) + (a_n 5) = 28 ∧ 
    (∀ n : ℕ, n > 0 → a_n n = 2^(n-1)) 

noncomputable def sequence_sum (n : ℕ) : ℝ := 
  let a_n := λ n : ℕ, 2^(n-1)
  let b_n := λ n : ℕ, (2*n - 1)
  sum (finset.range n) (λ k, a_n k + b_n k)

theorem sum_of_first_n_terms (n : ℕ) :
  sequence_sum n = (2^n - 1) + n^2 :=
by 
    intros
    sorry

theorem general_formula_geometric_sequence :
  ∃ a_n : ℕ → ℝ, 
    (a_n 3) * (a_n 4) * (a_n 5) = 512 ∧ 
    (a_n 3) + (a_n 4) + (a_n 5) = 28 ∧ 
    (∀ n : ℕ, n > 0 → a_n n = 2^(n-1)) :=
by 
    intros
    sorry

end sum_of_first_n_terms_general_formula_geometric_sequence_l735_735020


namespace quadratic_discriminant_l735_735170

variable {a b c : ℝ}
variable (h₁ : a ≠ 0)
variable (h₂ : (b - 1)^2 - 4 * a * (c + 2) = 0)
variable (h₃ : (b + 1/2)^2 - 4 * a * (c - 1) = 0)

theorem quadratic_discriminant : b^2 - 4 * a * c = -1 / 2 := 
by
  have h₁' : (b - 1)^2 - 4 * a * (c + 2) = 0 := h₂
  have h₂' : (b + 1/2)^2 - 4 * a * (c - 1) = 0 := h₃
  sorry

end quadratic_discriminant_l735_735170


namespace min_value_expression_l735_735869

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + 3 * y = 4) :
  (1 / (2 * x + 1) + 1 / (3 * y + 2)) = (3 / 8 + real.sqrt 2 / 4) :=
by
  sorry

end min_value_expression_l735_735869


namespace bakery_rolls_combinations_l735_735006

theorem bakery_rolls_combinations : 
  ∃ n : ℕ, n = 10 ∧ 
  (∃ k : ℕ, ∃ t : ℕ, (k = 4) ∧ (t = 2) ∧ (n = k * t + (n - k * t)) 
    ∧ (nat.choose (n - k * t + k - 1) (k - 1)) = 10) := by
  sorry

end bakery_rolls_combinations_l735_735006


namespace rectangle_square_perimeter_l735_735578

theorem rectangle_square_perimeter (l w : ℕ) (hl : l = 10) (hw : w = 8) :
  let s := min l w,
  let rem_l := if l > w then l - s else l,
  let rem_w := if w > l then w - s else w,
  4 * s = 32 ∧ 2 * (rem_l + rem_w) = 20 :=
by
  -- Definitions based on conditions
  have hs: s = w := by 
    -- s is the shortest side, which must be w in this problem
    exact min_eq_right hw.ge
  -- Perimeter of the square
  have perimeter_square: 4 * s = 32 := by 
    -- P = 4 * side length
    rw [hs, hw]
  -- Dimensions of remaining rectangle
  have rem_rect_dims: (rem_l, rem_w) = (l - s, s) := 
    by
    split_ifs
    -- l > w
    case h_1 => rw [hs, hl, nat.sub_sub_self hw.le, hl]
  -- Perimeter of remaining rectangle
  have perimeter_remaining: 2 * (l - s + s) = 20 := 
    by
    rw [hs, hl]
  -- Combine results
  exact ⟨perimeter_square, perimeter_remaining⟩

end rectangle_square_perimeter_l735_735578


namespace minimum_value_of_expression_l735_735534

noncomputable def f (x y : ℝ) : ℝ := 2 / (x - 2) + 2 / y

theorem minimum_value_of_expression (x y : ℝ) (h1 : x > 2) (h2 : y > 0) (h3 : 2^(x * 2^y) = 16) :
  ∃ (xmin : ℝ) (ymin : ℝ), x = xmin ∧ y = ymin ∧ f x y = 4 :=
by
  sorry

end minimum_value_of_expression_l735_735534


namespace impossible_projection_of_rectangle_l735_735723

theorem impossible_projection_of_rectangle (A B C D : Type) (cardboard_projection : A → B → C → D) 
  (line_segment_projections : ∃ p : A, cardboard_projection p = B) 
  (rectangle_projections : ∃ q : A, cardboard_projection q = C)
  (parallelogram_projections : ∃ r : A, cardboard_projection r = D) : 
  ∀ t : A, cardboard_projection t ≠ Triangle :=
by sorry

end impossible_projection_of_rectangle_l735_735723


namespace ellipse_C_eq_line_l_eq_min_major_axis_l735_735201

noncomputable def hyperbola_eq (a b x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

noncomputable def ellipse_eq (c b x y : ℝ) : Prop :=
  (x^2 / c^2) + (y^2 / b^2) = 1

noncomputable def line_eq_A (k x : ℝ) : ℝ → Prop :=
  λ y, y = k * x + 1/2

noncomputable def line_eq_B (k x : ℝ) : ℝ → Prop :=
  λ y, y = k * x + 1/2

noncomputable def line_eq_C (k x : ℝ) : ℝ → Prop :=
  λ y, y = x + 5

-- Problem (1) 
theorem ellipse_C_eq (x y : ℝ) : ellipse_eq 2 1 x y ↔ (x^2 / 4) + y^2 = 1 := 
sorry

-- Problem (2)
theorem line_l_eq (k : ℝ) (x1 x2 y1 y2 : ℝ) (h_a : ellipse_eq 2 1 x1 y1) (h_b : ellipse_eq 2 1 x2 y2) :
  (k = sqrt 15 / 10) ∨ (k = - sqrt 15 / 10) :=
sorry

-- Problem (3)
theorem min_major_axis (b x y : ℝ) (h : ∃ x y, ellipse_eq (sqrt(b^2 + 1)) b x y ∧ line_eq_C x y = y) :
  2 * sqrt(13) ≤ major_axis := 
sorry

end ellipse_C_eq_line_l_eq_min_major_axis_l735_735201


namespace order_of_magnitudes_l735_735849

theorem order_of_magnitudes 
  (a b c : ℝ)
  (ha : a = (2 : ℝ)^(-3 / 2))
  (hb : b = (2 / 5 : ℝ)^3)
  (hc : c = (1 / 2 : ℝ)^3) :
  a > c ∧ c > b := 
by {
  sorry
}

end order_of_magnitudes_l735_735849


namespace range_of_a_l735_735885

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x >= 0 then -x + 3 * a else x^2 - a * x + 1

theorem range_of_a (a : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → f a x1 ≥ f a x2) ↔ (0 <= a ∧ a <= 1/3) :=
by
  sorry

end range_of_a_l735_735885


namespace interest_difference_l735_735789

theorem interest_difference :
  ∀ (P R : ℕ) (T1 T2 : ℚ),
    P = 640 →
    R = 15 →
    T1 = 3.5 →
    T2 = 5 →
    ((P * R * T2) / 100 - (P * R * T1) / 100 = 144) :=
by
  intros P R T1 T2 hP hR hT1 hT2
  rw [hP, hR, hT1, hT2]
  norm_num
  exact sorry

end interest_difference_l735_735789


namespace correct_equations_for_rhombus_l735_735837

variables (A B C D : Type) [AddCommGroup A] [VectorSpace ℝ A]

def is_rhombus (AB BC AD CD : A) : Prop :=
  (AB ≠ BC ∧ ∥AB∥ = ∥BC∥ ∧ ∥AB - CD∥ = ∥AD + BC∥ ∧ ∥AC∥^2 + ∥BD∥^2 = 4 * ∥AB∥^2)

theorem correct_equations_for_rhombus 
  (AB BC AD CD AC BD : A)
  (h1 : is_rhombus AB BC AD CD) :
  ∃ (n : ℕ), n = 3 :=
sorry

end correct_equations_for_rhombus_l735_735837


namespace distance_between_J_and_Y_l735_735457

noncomputable def distance_between_stations 
  (s_A s_B s_C : ℝ) -- speeds of cars A, B, and C
  (t_meet_BC: ℝ) -- time in hours between meetings of A with B and C
  (meet_time_AB: ℝ)
  (combined_time: ℝ) : ℝ :=
  let combined_speed_AC := s_A + s_C in
  let distance_AC := combined_speed_AC * t_meet_BC in
  let relative_speed_BC := s_B - s_C in
  let time_AB := distance_AC / relative_speed_BC in
  let combined_speed_AB := s_A + s_B in
  combined_speed_AB * combined_time

theorem distance_between_J_and_Y :
  distance_between_stations 90 80 60 (1/3) 2.5 = 425 := 
by
  sorry

end distance_between_J_and_Y_l735_735457


namespace evaluate_i_powers_l735_735080

theorem evaluate_i_powers :
  let i : ℂ := complex.I in
  (i^45 + i^345) = 2 * i :=
by
  sorry

end evaluate_i_powers_l735_735080


namespace overall_average_birds_l735_735957

theorem overall_average_birds :
  let monday := (5, 7)
  let tuesday := (5, 5)
  let wednesday := (10, 8)
  let thursday := (7, 10)
  let friday := (3, 6)
  let saturday := (8, 12)
  let sunday := (4, 9)
  let total_birds := 35 + 25 + 80 + 70 + 18 + 96 + 36
  let total_sites := 5 + 5 + 10 + 7 + 3 + 8 + 4
  let overall_average := total_birds / total_sites
  in overall_average ≈ 8.57 :=
by sorry

end overall_average_birds_l735_735957


namespace arable_land_decrease_max_l735_735001

theorem arable_land_decrease_max
  (A₀ : ℕ := 100000)
  (grain_yield_increase : ℝ := 1.22)
  (per_capita_increase : ℝ := 1.10)
  (pop_growth_rate : ℝ := 0.01)
  (years : ℕ := 10) :
  ∃ (max_decrease : ℕ), max_decrease = 4 := sorry

end arable_land_decrease_max_l735_735001


namespace factor_of_polynomial_l735_735472

theorem factor_of_polynomial (p : Polynomial ℤ) (h : p = Polynomial.C 1 * Polynomial.X^3 + Polynomial.C 3 * Polynomial.X^2 - Polynomial.C 4 * Polynomial.X - Polynomial.C 12) :
  Polynomial.isFactor (Polynomial.X - Polynomial.C 2) p :=
by {
  have hp : p = Polynomial.monomial 3 1 + Polynomial.monomial 2 3 - Polynomial.monomial 1 4 - Polynomial.C 12 := h,
  sorry
}

end factor_of_polynomial_l735_735472


namespace pattern1_cannot_form_cube_with_unique_adjacent_colors_l735_735060

def pattern1 := 
{ 
  shape := "cross",
  color_sequence := ["blue", "green", "red", "blue", "yellow", "green"]
}
def pattern2 := 
{ 
  shape := "T",
  color_sequence := ["blue", "green", "red", "blue", "yellow"]
}
def pattern3 := 
{ 
  shape := "two rows of three with one additional",
  color_sequence := ["blue", "green", "red", "blue", "yellow", "green", "red"]
}
def pattern4 := 
{ 
  shape := "linear",
  color_sequence := ["blue", "green", "red", "blue", "yellow", "green"]
}

theorem pattern1_cannot_form_cube_with_unique_adjacent_colors : 
  ∃ p, p = pattern1 ∧ 
    (∀ (c1 c2 : String), (c1 ≠ c2) → adjacent_faces pattern1 c1 c2 → False) := sorry

end pattern1_cannot_form_cube_with_unique_adjacent_colors_l735_735060


namespace committee_role_permutations_l735_735518

theorem committee_role_permutations
  (members : Set String) (roles : Set String) 
  (h1 : members = {'Alice', 'Bob', 'Carol', 'Dave'}) 
  (h2 : roles = {'president', 'secretary', 'treasurer', 'vice-president'}) 
  (h3 : members.size = roles.size) : 
  Set.Perm.multiplicity members roles = 24 := 
sorry

end committee_role_permutations_l735_735518


namespace line_y2_does_not_pass_through_fourth_quadrant_l735_735024

theorem line_y2_does_not_pass_through_fourth_quadrant (k b : ℝ) (h1 : k < 0) (h2 : b > 0) : 
  ¬(∃ x y : ℝ, (y = b * x - k ∧ x > 0 ∧ y < 0)) := 
by 
  sorry

end line_y2_does_not_pass_through_fourth_quadrant_l735_735024


namespace geometric_sequence_min_value_l735_735512

theorem geometric_sequence_min_value
  (a : ℕ → ℝ)
  (h1 : ∀ n, 0 < a n) 
  (h2 : a 9 = 9 * a 7)
  (exists_m_n : ∃ m n, a m * a n = 9 * (a 1)^2):
  ∀ m n, (m + n = 4) → (1 / m + 9 / n) ≥ 4 :=
by
  intros m n h
  sorry

end geometric_sequence_min_value_l735_735512


namespace solve_ineq_l735_735087

noncomputable def f (x : ℝ) : ℝ := (2 / (x + 2)) + (4 / (x + 8)) - (7 / 3)

theorem solve_ineq (x : ℝ) : 
  (f x ≤ 0) ↔ (x ∈ Set.Ioc (-8) 4) := 
sorry

end solve_ineq_l735_735087


namespace intersection_of_diagonals_in_parallelogram_l735_735935

theorem intersection_of_diagonals_in_parallelogram
  (EFGH : Parallelogram)
  (E F G H J K L : Point)
  (hEF : EFGH.edge EF)
  (hEH : EFGH.edge EH)
  (hJ : J ∈ segment E F)
  (hK : K ∈ segment E H)
  (hL : L ∈ intersection (line E G) (line J K))
  (hEJ_EF : EJ_ratio J E F (1 / 4))
  (hEK_EH : EK_ratio K E H (1 / 3))
  : EG_div_EL_ratio E G L = 1 := sorry

end intersection_of_diagonals_in_parallelogram_l735_735935


namespace find_c_of_parabola_l735_735320

theorem find_c_of_parabola (a b c : ℚ) (h_vertex : (5 : ℚ) = a * (3 : ℚ)^2 + b * (3 : ℚ) + c)
    (h_point : (7 : ℚ) = a * (1 : ℚ)^2 + b * (1 : ℚ) + c) :
  c = 19 / 2 :=
by
  sorry

end find_c_of_parabola_l735_735320


namespace determine_120_percent_of_y_l735_735228

def x := 0.80 * 350
def y := 0.60 * x
def result := 1.20 * y

theorem determine_120_percent_of_y : result = 201.6 := by
  sorry

end determine_120_percent_of_y_l735_735228


namespace total_population_expression_l735_735592

variables (b g t: ℕ)

-- Assuming the given conditions
def condition1 := b = 4 * g
def condition2 := g = 8 * t

-- The theorem to prove
theorem total_population_expression (h1 : condition1 b g) (h2 : condition2 g t) :
    b + g + t = 41 * b / 32 := sorry

end total_population_expression_l735_735592


namespace num_divisors_l735_735637

section
variables {p q r : ℕ} (m : ℕ) (n : ℕ)

/-- p, q, and r are distinct prime numbers -/
variable (hp : prime p) (hq : prime q) (hr : prime r)
variable (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r)

/-- m is a positive integer -/
variable (hm : m > 0)

/-- n is defined as 7^m * p^2 * q * r -/
def n := 7^m * p^2 * q * r

/-- The number of divisors of n is 12 times (m + 1) -/
theorem num_divisors (hp : prime p) (hq : prime q) (hr : prime r) (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) (hm : m > 0) :
  num_divisors n = 12 * (m + 1) := 
sorry

end

end num_divisors_l735_735637


namespace total_wrappers_collected_l735_735443

theorem total_wrappers_collected :
  let Andy_wrappers := 34
  let Max_wrappers := 15
  let Zoe_wrappers := 25
  Andy_wrappers + Max_wrappers + Zoe_wrappers = 74 :=
by
  let Andy_wrappers := 34
  let Max_wrappers := 15
  let Zoe_wrappers := 25
  show Andy_wrappers + Max_wrappers + Zoe_wrappers = 74
  sorry

end total_wrappers_collected_l735_735443


namespace sum_of_fourth_powers_of_distances_l735_735619

theorem sum_of_fourth_powers_of_distances
  (P : ℝ × ℝ)
  (h : P.1 ^ 2 + P.2 ^ 2 = 1) :
  (P.1)^4 + (P.2)^4 + ((P.1 - P.2) / Math.sqrt 2)^4 + ((P.1 + P.2) / Math.sqrt 2)^4 = 3 / 2 :=
by sorry

end sum_of_fourth_powers_of_distances_l735_735619


namespace find_number_l735_735570

theorem find_number (a b x : ℝ) (H1 : 2 * a = x * b) (H2 : a * b ≠ 0) (H3 : (a / 3) / (b / 2) = 1) : x = 3 :=
by
  sorry

end find_number_l735_735570


namespace initial_salary_increase_l735_735341

variables (S x : ℝ)

theorem initial_salary_increase (h : S * (1 + x / 100) * 0.75 = S * 1.0625) : x = 41.67 := by
  have S_ne_zero : S ≠ 0 := sorry
  have h_cancel_S := (mul_left_cancel' S_ne_zero).mp h
  have h_eq : 0.75 * (1 + x / 100) = 1.0625 := by
    rw [mul_assoc, h_cancel_S]
  have temp_eq : (1 + x / 100) = 1.41666667 := by
    calc 
      (1 + x / 100) = 1.0625 / 0.75 := by { sorry }
      ... = 1.41666667 := by { sorry }
  have x_eq : x / 100 = 0.41666667 := by
    rw [eq_div_iff, temp_eq]
    norm_num
  calc
    x = 41.666667 := by
    linarith

end initial_salary_increase_l735_735341


namespace compute_expression_l735_735055

theorem compute_expression : 
  (real.sqrt 12) + (2⁻¹) + (real.cos (real.pi / 3)) - 3 * (real.tan (real.pi / 6)) = real.sqrt 3 + 1 :=
  by sorry

end compute_expression_l735_735055


namespace cone_sections_equal_surface_area_l735_735012

theorem cone_sections_equal_surface_area {m r : ℝ} (h_r_pos : r > 0) (h_m_pos : m > 0) :
  ∃ (m1 m2 : ℝ), 
  (m1 = m / Real.sqrt 3) ∧ 
  (m2 = m / 3 * Real.sqrt 6) :=
sorry

end cone_sections_equal_surface_area_l735_735012


namespace range_of_function_l735_735687

theorem range_of_function : (∃ y, (∃ x, y = sqrt (16 - 4^x)) ↔ 0 ≤ y ∧ y < 4) :=
sorry

end range_of_function_l735_735687


namespace account_balance_after_one_year_l735_735734

-- Define the initial conditions
variable (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ)

-- Define the compound interest formula
noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

-- Given conditions
def P_amount := 700
def r_rate := 0.10
def n_times := 2
def t_years := 1

theorem account_balance_after_one_year : compound_interest P_amount r_rate n_times t_years = 770.75 :=
by
  -- Self-contained proof can be skipped with sorry
  sorry

end account_balance_after_one_year_l735_735734


namespace part1_part2_part3_l735_735632

-- Define the set M and the mapping F
def M (a b : ℝ) : set (ℝ → ℝ) :=
  {f | ∃ (a b : ℝ), f = λ x, a * real.cos x + b * real.sin x}

def F (a b : ℝ) : ℝ → ℝ :=
  λ x, a * real.cos x + b * real.sin x

-- Prove that two different points do not correspond to the same function.
theorem part1 (a b c d : ℝ) :
  F a b = F c d → a = c ∧ b = d := 
sorry

-- Prove that when f₀(x) ∈ M, f₁(x) = f₀(x + t) ∈ M, where t is a constant.
theorem part2 (a₀ b₀ t : ℝ) :
  (λ x, a₀ * real.cos x + b₀ * real.sin x) ∈ M a₀ b₀ → 
  (λ x, a₀ * real.cos (x + t) + b₀ * real.sin (x + t)) ∈ M a₀ b₀ :=
sorry

-- Describe the preimage of M₁ under the mapping F for a fixed function f₀(x) = a₀ cos x + b₀ sin x.
theorem part3 (a₀ b₀ : ℝ) :
  F a₀ b₀ ∈ M a₀ b₀ → 
  ∀ t : ℝ, F (a₀ * real.cos t + b₀ * real.sin t) (b₀ * real.cos t - a₀ * real.sin t) ∈ M a₀ b₀ →
  F⁻¹ (λ t : ℝ, (a₀ * real.cos t + b₀ * real.sin t, b₀ * real.cos t - a₀ * real.sin t)) = 
  {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = a₀ ^ 2 + b₀ ^ 2} :=
sorry

end part1_part2_part3_l735_735632


namespace sum_of_sequence_l735_735281

def a (n : ℕ) : ℕ :=
  if n = 1 then 1 else 2^(n - 2)

noncomputable def S (n : ℕ) : ℝ :=
  (∑ k in Finset.range n, (k + 1 : ℕ) / a (k + 1))

theorem sum_of_sequence (n : ℕ) (hn : 1 ≤ n) :
  S n = 7 - (n + 2) / 2^(n - 2) := sorry

end sum_of_sequence_l735_735281


namespace Rachel_total_score_l735_735412

theorem Rachel_total_score
    (points_per_treasure : ℕ)
    (treasures_first_level : ℕ)
    (treasures_second_level : ℕ)
    (h1 : points_per_treasure = 9)
    (h2 : treasures_first_level = 5)
    (h3 : treasures_second_level = 2) : 
    (points_per_treasure * treasures_first_level + points_per_treasure * treasures_second_level = 63) :=
by
    sorry

end Rachel_total_score_l735_735412


namespace range_of_a_l735_735200

theorem range_of_a (a : ℝ) : 
  (∀ n : ℕ, n ≥ 8 → (a * (n^2) + n + 5) > (a * ((n + 1)^2) + (n + 1) + 5)) → 
  (a * (1^2) + 1 + 5 < a * (2^2) + 2 + 5) →
  (a * (2^2) + 2 + 5 < a * (3^2) + 3 + 5) →
  (a * (3^2) + 3 + 5 < a * (4^2) + 4 + 5) →
  (- (1 / 7) < a ∧ a < - (1 / 17)) :=
by
  sorry

end range_of_a_l735_735200


namespace square_area_l735_735745

variable {s : ℝ}
variables (P : Point)

def is_on_diagonal (P : Point) : Prop :=
  -- Definition that P is on diagonal AC
  on_diagonal AC P

def AP_is_quarter_side (P : Point) : Prop :=
  -- Definition that AP is one-fourth of the side length
  AP P = s / 4

def quadrilateral_area_ABPD_one (s : ℝ) (P : Point) : Prop :=
  -- Definition that the area of ABPD is 1 square unit
  area (ABPD P) = 1

theorem square_area (P : Point) (h1: is_on_diagonal P) (h2: AP_is_quarter_side P) (h3: quadrilateral_area_ABPD_one s P) 
  : area (square s) = 4 * real.sqrt 2 := 
  sorry

end square_area_l735_735745


namespace nathan_tokens_l735_735286

theorem nathan_tokens
  (hockey_games : Nat := 5)
  (hockey_cost : Nat := 4)
  (basketball_games : Nat := 7)
  (basketball_cost : Nat := 5)
  (skee_ball_games : Nat := 3)
  (skee_ball_cost : Nat := 3)
  : hockey_games * hockey_cost + basketball_games * basketball_cost + skee_ball_games * skee_ball_cost = 64 := 
by
  sorry

end nathan_tokens_l735_735286


namespace total_distance_in_land_miles_l735_735049

-- Definitions based on conditions
def speed_one_sail : ℕ := 25
def time_one_sail : ℕ := 4
def distance_one_sail := speed_one_sail * time_one_sail

def speed_two_sails : ℕ := 50
def time_two_sails : ℕ := 4
def distance_two_sails := speed_two_sails * time_two_sails

def conversion_factor : ℕ := 115  -- Note: 1.15 * 100 for simplicity with integers

-- Theorem to prove the total distance in land miles
theorem total_distance_in_land_miles : (distance_one_sail + distance_two_sails) * conversion_factor / 100 = 345 := by
  sorry

end total_distance_in_land_miles_l735_735049


namespace intersection_distance_l735_735677

theorem intersection_distance (p q : ℕ) (h1 : p = 65) (h2 : q = 2) :
  p - q = 63 := 
by
  sorry

end intersection_distance_l735_735677


namespace sum_of_prime_factors_of_247520_l735_735391

theorem sum_of_prime_factors_of_247520 :
  (∑ p in (finset.filter nat.prime (nat.factors 247520)).to_finset, p) = 113 :=
by sorry

end sum_of_prime_factors_of_247520_l735_735391


namespace at_least_two_equations_have_solutions_l735_735516

theorem at_least_two_equations_have_solutions 
  (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  (∃ x : ℝ, (x - a) * (x - b) = x - c) + (∃ x : ℝ, (x - b) * (x - c) = x - a) + (∃ x : ℝ, (x - c) * (x - a) = x - b) ≥ 2 :=
sorry

end at_least_two_equations_have_solutions_l735_735516


namespace lowest_probability_red_side_up_l735_735362

def card_flip_probability (k : ℕ) (n : ℕ) : ℚ :=
  if k ≤ n/2 then (n-k)^2/(n^2) + k^2/(n^2)
  else card_flip_probability (n+1-k) n 

theorem lowest_probability_red_side_up :
  (card_flip_probability 13 50) = (card_flip_probability 38 50) ∧ 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 25 → (card_flip_probability k 50 ≥ card_flip_probability 13 50)) :=
begin
  sorry
end

end lowest_probability_red_side_up_l735_735362


namespace discriminant_of_P_l735_735152

theorem discriminant_of_P (a b c : ℚ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1/2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1/2 := 
begin
  -- Proof omitted for brevity
  sorry
end

end discriminant_of_P_l735_735152


namespace quadratic_polynomial_discriminant_l735_735147

def P (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∃ x : ℝ, P a b c x = x - 2 ∧ (discriminant a (b - 1) (c + 2) = 0))
  (h₂ : ∃ x : ℝ, P a b c x = 1 - x / 2 ∧ (discriminant a (b + 1 / 2) (c - 1) = 0)) :
  discriminant a b c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l735_735147


namespace number_of_possible_b_values_l735_735316

theorem number_of_possible_b_values : 
  let b_values := {b : ℕ | 7 ≤ b ∧ b ≤ 16} in
  #b_values = 10 := sorry

end number_of_possible_b_values_l735_735316


namespace evaluate_i_powers_l735_735079

theorem evaluate_i_powers :
  let i : ℂ := complex.I in
  (i^45 + i^345) = 2 * i :=
by
  sorry

end evaluate_i_powers_l735_735079


namespace minimum_people_l735_735372

theorem minimum_people (V E : Type)
  [Fintype V] [Fintype E]
  (G : SimpleGraph V)
  (h_deg : ∀ v : V, G.degree v = 3)
  (h_three_subset : ∀ (s : Finset V), s.card = 3 → ∃ (x y : V), x ∈ s ∧ y ∈ s ∧ ¬ G.Adj x y) :
  Fintype.card V = 6 :=
sorry

end minimum_people_l735_735372


namespace parabola_line_ratio_l735_735761

noncomputable def parabola_focus (p : ℝ) (h : p > 0) : ℝ × ℝ := (p / 2, 0)

noncomputable def line_through_focus_at_angle (p : ℝ) (h : p > 0) : ℝ → ℝ := 
  λ x, Real.sqrt 3 * (x - p / 2)

def parabola (p : ℝ) (h : p > 0) (x y : ℝ) : Prop :=
  y^2 = 2 * p * x

def line (p : ℝ) (h : p > 0) (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * (x - p / 2)

theorem parabola_line_ratio (p : ℝ) (h : p > 0) (A B : ℝ) :
  ∃ A_x A_y B_x B_y : ℝ, 
    parabola p h A_x A_y ∧ 
    line p h A_x A_y ∧ 
    parabola p h B_x B_y ∧ 
    line p h B_x B_y ∧ 
    A_x + p / 2 = 2 * p ∧ 
    B_x + p / 2 = 2 * p / 3 ∧
    abs (A_x + p / 2) / abs (B_x + p / 2) = 3 :=
begin
  sorry
end

end parabola_line_ratio_l735_735761


namespace yellow_marbles_count_l735_735613

-- Definitions based on given conditions
def blue_marbles : ℕ := 10
def green_marbles : ℕ := 5
def black_marbles : ℕ := 1
def probability_black : ℚ := 1 / 28
def total_marbles : ℕ := 28

-- Problem statement to prove
theorem yellow_marbles_count :
  (total_marbles = blue_marbles + green_marbles + black_marbles + n) →
  (probability_black = black_marbles / total_marbles) →
  n = 12 :=
by
  intros; sorry

end yellow_marbles_count_l735_735613


namespace librarian_donated_200_books_this_year_l735_735104

noncomputable def total_books_five_years_ago : ℕ := 500
noncomputable def books_bought_two_years_ago : ℕ := 300
noncomputable def books_bought_last_year : ℕ := books_bought_two_years_ago + 100
noncomputable def total_books_current : ℕ := 1000

-- The Lean statement to prove the librarian donated 200 old books this year
theorem librarian_donated_200_books_this_year :
  total_books_five_years_ago + books_bought_two_years_ago + books_bought_last_year - total_books_current = 200 :=
by sorry

end librarian_donated_200_books_this_year_l735_735104


namespace part1_part2_part3_l735_735198

noncomputable def f (x a : ℝ) : ℝ := x^2 + (x - 1) * |x - a|

-- Part 1
theorem part1 (a : ℝ) (x : ℝ) (h : a = -1) : 
  (f x a = 1) ↔ (x ≤ -1 ∨ x = 1) :=
sorry

-- Part 2
theorem part2 (a : ℝ) : 
  (∀ x y : ℝ, x < y → f x a < f y a) ↔ (a ≥ 1 / 3) :=
sorry

-- Part 3
theorem part3 (a : ℝ) (h1 : a < 1) (h2 : ∀ x : ℝ, f x a ≥ 2 * x - 3) : 
  -3 ≤ a ∧ a < 1 :=
sorry

end part1_part2_part3_l735_735198


namespace quadratic_radical_same_type_l735_735227

theorem quadratic_radical_same_type (x : ℝ) (h₀ : sqrt (x^2 - 2) = sqrt (2 * x - 2)) : x = 2 :=
by
  sorry

end quadratic_radical_same_type_l735_735227


namespace solve_quadratic_l735_735693

theorem solve_quadratic : ∀ x : ℝ, 2 * x^2 + 5 * x = 0 ↔ x = 0 ∨ x = -5/2 :=
by
  intro x
  sorry

end solve_quadratic_l735_735693


namespace parabolas_intersect_l735_735063

theorem parabolas_intersect :
  let eq1 (x : ℝ) := 3 * x^2 - 4 * x + 2
  let eq2 (x : ℝ) := -x^2 + 6 * x + 8
  (∃ x y : ℝ, y = eq1 x ∧ y = eq2 x ∧ x = -0.5 ∧ y = 4.75) ∧
  (∃ x y : ℝ, y = eq1 x ∧ y = eq2 x ∧ x = 3 ∧ y = 17) :=
by sorry

end parabolas_intersect_l735_735063


namespace find_discriminant_l735_735159

variables {a b c : ℝ}
variables (P : ℝ → ℝ)
def is_quadratic_polynomial (P : ℝ → ℝ) : Prop := ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, P x = a * x^2 + b * x + c)

theorem find_discriminant (h1 : is_quadratic_polynomial P)
  (h2 : ∃ x, P x = x - 2)
  (h3 : ∃ y, P y = 1 - y / 2)
  : ∃ D, D = -1/2 := 
sorry

end find_discriminant_l735_735159


namespace problem_solution_l735_735550

-- Define the set A
def A : Set ℝ := {x | |x| > 1}

-- Define the set B
def B : Set ℝ := {y | ∃ x : ℝ, y = x^2}

-- Prove that (ℝ \ A) ∩ B = { x | 0 ≤ x ∧ x ≤ 1 }
theorem problem_solution :
  (set.compl A) ∩ B = {x | 0 ≤ x ∧ x ≤ 1} :=
by
  sorry

end problem_solution_l735_735550


namespace limit_example_l735_735082

theorem limit_example :
  (∀ (f : ℝ → ℝ) (l : ℝ), (∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x - l| < ε) → 
    (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → |f x - l| < ε) ) →
  (∀ (lim_x : ℝ), {f : ℝ → ℝ // continuous f } →
    (∀ ε > 0, ∃ δ > 0, ∀ x, |x - lim_x| < δ → |f x - 1| < ε)) →
  (∀ (lim_x : ℝ), {f : ℝ → ℝ // continuous f } →
    (∀ ε > 0, ∃ δ > 0, ∀ x, |x - lim_x| < δ → |f x - 3| < ε)) →
  (∀ (lim_x : ℝ), {f : ℝ → ℝ // continuous f } →
    (∀ ε > 0, ∃ δ > 0, ∀ x, |x - lim_x| < δ → |f x - 3| < ε)) →
  (∀ (lim_x : ℝ), {f : ℝ → ℝ // continuous f } →
    (∀ ε > 0, ∃ δ > 0, ∀ x, |x - lim_x| < δ → |f x - 2| < ε)) →
∃ ε > 0 ,  ∀ g h : ℝ × ℝ → ℝ, ∃ δ > 0, ∀ x, (x ≠ 0 ∧ |x| < δ) → 
g ⟨x, x * (sin (3 * x) - (tan (2 * x)) )⟩ / 
h ⟨x, x * (e ^ x - e^(3 * x)⟩ < ε) := 
by {

  sorry
}

end limit_example_l735_735082


namespace noemi_initial_money_l735_735287

variable (money_lost_roulette : ℕ := 400)
variable (money_lost_blackjack : ℕ := 500)
variable (money_left : ℕ)
variable (money_started : ℕ)

axiom money_left_condition : money_left > 0
axiom total_loss_condition : money_lost_roulette + money_lost_blackjack = 900

theorem noemi_initial_money (h1 : money_lost_roulette = 400) (h2 : money_lost_blackjack = 500)
    (h3 : money_started - 900 = money_left) (h4 : money_left > 0) :
    money_started > 900 := by
  sorry

end noemi_initial_money_l735_735287


namespace fraction_pos_integer_l735_735907

theorem fraction_pos_integer (p : ℕ) (hp : 0 < p) : (∃ (k : ℕ), k = 1 + (2 * p + 53) / (3 * p - 8)) ↔ p = 3 := 
by
  sorry

end fraction_pos_integer_l735_735907


namespace problem_q_convex_quadrilateral_orthocenters_l735_735623

variables {V : Type*} [inner_product_space ℝ V] [finite_dimensional ℝ V]

-- Define the quadrilateral ABCD
variables (A B C D : V)

-- Define the orthocenters H_A, H_B, H_C, H_D of the respective triangles BCD, ACD, ABD, ABC
variables (H_A H_B H_C H_D : V)

-- Given that ABCD is a convex quadrilateral and H_A, H_B, H_C, H_D are the orthocenters
-- as defined in the problem statement
theorem problem_q_convex_quadrilateral_orthocenters 
  (convex_ABCD : convex_hull ℝ (set.insert A (set.insert B (set.insert C {D})))) 
  (HA_def : ∀ G_A, centroid ℝ ({B, C, D} : set V) = G_A → H_A = B + C + D - 2 • G_A)
  (HB_def : ∀ G_B, centroid ℝ ({A, C, D} : set V) = G_B → H_B = A + C + D - 2 • G_B)
  (HC_def : ∀ G_C, centroid ℝ ({A, B, D} : set V) = G_C → H_C = A + B + D - 2 • G_C)
  (HD_def : ∀ G_D, centroid ℝ ({A, B, C} : set V) = G_D → H_D = A + B + C - 2 • G_D):
  [convex_hull ℝ {H_A, H_B, H_C, H_D}] = (4 / 9 : ℝ) * [convex_hull ℝ {A, B, C, D}] :=
sorry

end problem_q_convex_quadrilateral_orthocenters_l735_735623
