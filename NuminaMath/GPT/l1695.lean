import Mathlib

namespace NUMINAMATH_GPT_range_of_a_l1695_169521

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, 2^(2 * x) + 2^x * a + a + 1 = 0) : a ≤ 2 - 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1695_169521


namespace NUMINAMATH_GPT_expand_simplify_correct_l1695_169593

noncomputable def expand_and_simplify (x : ℕ) : ℕ :=
  (x + 4) * (x - 9)

theorem expand_simplify_correct (x : ℕ) : 
  (x + 4) * (x - 9) = x^2 - 5*x - 36 := 
by
  sorry

end NUMINAMATH_GPT_expand_simplify_correct_l1695_169593


namespace NUMINAMATH_GPT_sin_alpha_in_second_quadrant_l1695_169554

theorem sin_alpha_in_second_quadrant
  (α : ℝ)
  (h1 : π/2 < α ∧ α < π)
  (h2 : Real.tan α = - (8 / 15)) :
  Real.sin α = 8 / 17 :=
sorry

end NUMINAMATH_GPT_sin_alpha_in_second_quadrant_l1695_169554


namespace NUMINAMATH_GPT_remaining_amount_needed_l1695_169510

def goal := 150
def earnings_from_3_families := 3 * 10
def earnings_from_15_families := 15 * 5
def total_earnings := earnings_from_3_families + earnings_from_15_families
def remaining_amount := goal - total_earnings

theorem remaining_amount_needed : remaining_amount = 45 := by
  sorry

end NUMINAMATH_GPT_remaining_amount_needed_l1695_169510


namespace NUMINAMATH_GPT_find_pairs_l1695_169584

noncomputable def f (k : ℤ) (x y : ℝ) : ℝ :=
  if k = 0 then 0 else (x^k + y^k + (-1)^k * (x + y)^k) / k

theorem find_pairs (x y : ℝ) (hxy : x ≠ 0 ∧ y ≠ 0 ∧ x + y ≠ 0) :
  ∃ (m n : ℤ), m ≠ 0 ∧ n ≠ 0 ∧ m ≤ n ∧ m + n ≠ 0 ∧ 
    (∀ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ x + y ≠ 0 → f m x y * f n x y = f (m + n) x y) :=
  sorry

end NUMINAMATH_GPT_find_pairs_l1695_169584


namespace NUMINAMATH_GPT_find_x_given_y_l1695_169528

-- Given x varies inversely as the square of y, we define the relationship
def varies_inversely (x y k : ℝ) : Prop := x = k / y^2

theorem find_x_given_y (k : ℝ) (h_k : k = 4) :
  ∀ (y : ℝ), varies_inversely x y k → y = 2 → x = 1 :=
by
  intros y h_varies h_y_eq
  -- We need to prove the statement here
  sorry

end NUMINAMATH_GPT_find_x_given_y_l1695_169528


namespace NUMINAMATH_GPT_resistance_between_opposite_vertices_of_cube_l1695_169568

-- Define the parameters of the problem
def resistance_cube_edge : ℝ := 1

-- Define the function to calculate the equivalent resistance
noncomputable def equivalent_resistance_opposite_vertices (R : ℝ) : ℝ :=
  let R1 := R / 3
  let R2 := R / 6
  let R3 := R / 3
  R1 + R2 + R3

-- State the theorem to prove the resistance between two opposite vertices
theorem resistance_between_opposite_vertices_of_cube :
  equivalent_resistance_opposite_vertices resistance_cube_edge = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_resistance_between_opposite_vertices_of_cube_l1695_169568


namespace NUMINAMATH_GPT_b_plus_d_l1695_169597

noncomputable def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem b_plus_d 
  (a b c d : ℝ) 
  (h1 : f a b c d 1 = 20) 
  (h2 : f a b c d (-1) = 16) 
: b + d = 18 :=
sorry

end NUMINAMATH_GPT_b_plus_d_l1695_169597


namespace NUMINAMATH_GPT_trapezoid_area_l1695_169511

theorem trapezoid_area (l : ℝ) (r : ℝ) (a b : ℝ) (h : ℝ) (A : ℝ) :
  l = 9 →
  r = 4 →
  a + b = l + l →
  h = 2 * r →
  (a + b) / 2 * h = A →
  A = 72 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_trapezoid_area_l1695_169511


namespace NUMINAMATH_GPT_constructed_expression_equals_original_l1695_169550

variable (a : ℝ)

theorem constructed_expression_equals_original : 
  a ≠ 0 → 
  ((1/a) / ((1/a) * (1/a)) - (1/a)) / (1/a) = (a + 1) * (a - 1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_constructed_expression_equals_original_l1695_169550


namespace NUMINAMATH_GPT_expression_value_l1695_169556

theorem expression_value : 2 + 3 * 5 + 2 = 19 := by
  sorry

end NUMINAMATH_GPT_expression_value_l1695_169556


namespace NUMINAMATH_GPT_size_relationship_l1695_169526

theorem size_relationship (a b : ℝ) (h₀ : a + b > 0) :
  a / (b^2) + b / (a^2) ≥ 1 / a + 1 / b :=
by
  sorry

end NUMINAMATH_GPT_size_relationship_l1695_169526


namespace NUMINAMATH_GPT_max_k_value_l1695_169555

theorem max_k_value (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) : 
  (∃ k : ℝ, (∀ m, 0 < m → m < 1/2 → (1/m + 2/(1-2*m) ≥ k)) ∧ k = 8) := 
sorry

end NUMINAMATH_GPT_max_k_value_l1695_169555


namespace NUMINAMATH_GPT_minimum_value_sine_shift_l1695_169534

theorem minimum_value_sine_shift :
  ∀ (f : ℝ → ℝ) (φ : ℝ), (∀ x, f x = Real.sin (2 * x + φ)) → |φ| < Real.pi / 2 →
  (∀ x, f (x + Real.pi / 6) = f (-x)) →
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = - Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_sine_shift_l1695_169534


namespace NUMINAMATH_GPT_total_ceilings_to_paint_l1695_169523

theorem total_ceilings_to_paint (ceilings_painted_this_week : ℕ) 
                                (ceilings_painted_next_week : ℕ)
                                (ceilings_left_to_paint : ℕ) 
                                (h1 : ceilings_painted_this_week = 12) 
                                (h2 : ceilings_painted_next_week = ceilings_painted_this_week / 4) 
                                (h3 : ceilings_left_to_paint = 13) : 
    ceilings_painted_this_week + ceilings_painted_next_week + ceilings_left_to_paint = 28 :=
by
  sorry

end NUMINAMATH_GPT_total_ceilings_to_paint_l1695_169523


namespace NUMINAMATH_GPT_renu_work_rate_l1695_169512

theorem renu_work_rate (R : ℝ) :
  (∀ (renu_rate suma_rate combined_rate : ℝ),
    renu_rate = 1 / R ∧
    suma_rate = 1 / 6 ∧
    combined_rate = 1 / 3 ∧    
    combined_rate = renu_rate + suma_rate) → 
    R = 6 :=
by
  sorry

end NUMINAMATH_GPT_renu_work_rate_l1695_169512


namespace NUMINAMATH_GPT_prime_in_form_x_squared_plus_16y_squared_prime_in_form_4x_squared_plus_4xy_plus_5y_squared_l1695_169538

theorem prime_in_form_x_squared_plus_16y_squared (p : ℕ) (hprime : Prime p) (h1 : p % 8 = 1) :
  ∃ x y : ℤ, p = x^2 + 16 * y^2 :=
by
  sorry

theorem prime_in_form_4x_squared_plus_4xy_plus_5y_squared (p : ℕ) (hprime : Prime p) (h5 : p % 8 = 5) :
  ∃ x y : ℤ, p = 4 * x^2 + 4 * x * y + 5 * y^2 :=
by
  sorry

end NUMINAMATH_GPT_prime_in_form_x_squared_plus_16y_squared_prime_in_form_4x_squared_plus_4xy_plus_5y_squared_l1695_169538


namespace NUMINAMATH_GPT_decimal_2_09_is_209_percent_l1695_169503

-- Definition of the conversion from decimal to percentage
def decimal_to_percentage (x : ℝ) := x * 100

-- Theorem statement
theorem decimal_2_09_is_209_percent : decimal_to_percentage 2.09 = 209 :=
by sorry

end NUMINAMATH_GPT_decimal_2_09_is_209_percent_l1695_169503


namespace NUMINAMATH_GPT_ratio_of_sheep_to_cow_l1695_169575

noncomputable def sheep_to_cow_ratio 
  (S : ℕ) 
  (h1 : 12 + 4 * S = 108) 
  (h2 : S ≠ 0) : ℕ × ℕ := 
if h3 : 12 = 0 then (0, 0) else (2, 1)

theorem ratio_of_sheep_to_cow 
  (S : ℕ) 
  (h1 : 12 + 4 * S = 108) 
  (h2 : S ≠ 0) : sheep_to_cow_ratio S h1 h2 = (2, 1) := 
sorry

end NUMINAMATH_GPT_ratio_of_sheep_to_cow_l1695_169575


namespace NUMINAMATH_GPT_option_a_not_right_triangle_option_b_right_triangle_option_c_right_triangle_option_d_right_triangle_l1695_169578

noncomputable def triangle (A B C : ℝ) := A + B + C = 180

-- Define the conditions for options A, B, C, and D
def option_a := ∀ A B C : ℝ, triangle A B C → A = B ∧ B = 3 * C
def option_b := ∀ A B C : ℝ, triangle A B C → A + B = C
def option_c := ∀ A B C : ℝ, triangle A B C → A = B ∧ B = (1/2) * C
def option_d := ∀ A B C : ℝ, triangle A B C → ∃ x : ℝ, A = x ∧ B = 2 * x ∧ C = 3 * x

-- Define that option A does not form a right triangle
theorem option_a_not_right_triangle (A B C : ℝ) (hA : triangle A B C) : 
  option_a → A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 :=
sorry

-- Check that options B, C, and D do form right triangles
theorem option_b_right_triangle (A B C : ℝ) (hA : triangle A B C) : 
  option_b → C = 90 :=
sorry

theorem option_c_right_triangle (A B C : ℝ) (hA : triangle A B C) : 
  option_c → C = 90 :=
sorry

theorem option_d_right_triangle (A B C : ℝ) (hA : triangle A B C) : 
  option_d → C = 90 :=
sorry

end NUMINAMATH_GPT_option_a_not_right_triangle_option_b_right_triangle_option_c_right_triangle_option_d_right_triangle_l1695_169578


namespace NUMINAMATH_GPT_hexagon_perimeter_l1695_169552

-- Define the length of one side of the hexagon
def side_length : ℕ := 5

-- Define the number of sides of a hexagon
def num_sides : ℕ := 6

-- Problem statement: Prove the perimeter of a regular hexagon with the given side length
theorem hexagon_perimeter (s : ℕ) (n : ℕ) : s = side_length ∧ n = num_sides → n * s = 30 :=
by sorry

end NUMINAMATH_GPT_hexagon_perimeter_l1695_169552


namespace NUMINAMATH_GPT_real_solutions_l1695_169542

theorem real_solutions : 
  ∃ x : ℝ, (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 8) ∧ (x = 7 ∨ x = -2) :=
sorry

end NUMINAMATH_GPT_real_solutions_l1695_169542


namespace NUMINAMATH_GPT_point_same_side_of_line_l1695_169527

def same_side (p₁ p₂ : ℝ × ℝ) (a b c : ℝ) : Prop :=
  (a * p₁.1 + b * p₁.2 + c > 0) ↔ (a * p₂.1 + b * p₂.2 + c > 0)

theorem point_same_side_of_line :
  same_side (1, 2) (1, 0) 2 (-1) 1 :=
by
  unfold same_side
  sorry

end NUMINAMATH_GPT_point_same_side_of_line_l1695_169527


namespace NUMINAMATH_GPT_value_of_neg_a_squared_sub_3a_l1695_169579

variable (a : ℝ)
variable (h : a^2 + 3 * a - 5 = 0)

theorem value_of_neg_a_squared_sub_3a : -a^2 - 3*a = -5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_neg_a_squared_sub_3a_l1695_169579


namespace NUMINAMATH_GPT_exit_condition_l1695_169514

-- Define the loop structure in a way that is consistent with how the problem is described
noncomputable def program_loop (k : ℕ) : ℕ :=
  if k < 7 then 35 else sorry -- simulate the steps of the program

-- The proof goal is to show that the condition which stops the loop when s = 35 is k ≥ 7
theorem exit_condition (k : ℕ) (s : ℕ) : 
  (program_loop k = 35) → (k ≥ 7) :=
by {
  sorry
}

end NUMINAMATH_GPT_exit_condition_l1695_169514


namespace NUMINAMATH_GPT_solve_inequality_l1695_169561

theorem solve_inequality (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) : 
  (x^2 - 9) / (x^2 - 1) > 0 ↔ (x > 3 ∨ x < -3 ∨ (-1 < x ∧ x < 1)) :=
sorry

end NUMINAMATH_GPT_solve_inequality_l1695_169561


namespace NUMINAMATH_GPT_vector_subtraction_magnitude_l1695_169563

theorem vector_subtraction_magnitude (a b : EuclideanSpace ℝ (Fin 2))
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 3) 
  (h3 : ‖a + b‖ = Real.sqrt 19) : 
  ‖a - b‖ = Real.sqrt 7 :=
sorry

end NUMINAMATH_GPT_vector_subtraction_magnitude_l1695_169563


namespace NUMINAMATH_GPT_rectangle_height_l1695_169541

variable (h : ℕ) -- Define h as a natural number for the height

-- Given conditions
def width : ℕ := 32
def area_divided_by_diagonal : ℕ := 576

-- Math proof problem
theorem rectangle_height :
  (1 / 2 * (width * h) = area_divided_by_diagonal) → h = 36 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_height_l1695_169541


namespace NUMINAMATH_GPT_first_term_of_arithmetic_progression_l1695_169544

theorem first_term_of_arithmetic_progression 
  (a : ℕ) (d : ℕ) (n : ℕ) 
  (nth_term_eq : a + (n - 1) * d = 26)
  (common_diff : d = 2)
  (term_num : n = 10) : 
  a = 8 := 
by 
  sorry

end NUMINAMATH_GPT_first_term_of_arithmetic_progression_l1695_169544


namespace NUMINAMATH_GPT_eval_g_five_l1695_169532

def g (x : ℝ) : ℝ := 4 * x - 2

theorem eval_g_five : g 5 = 18 := by
  sorry

end NUMINAMATH_GPT_eval_g_five_l1695_169532


namespace NUMINAMATH_GPT_cube_root_equation_l1695_169571

theorem cube_root_equation (x : ℝ) (h : (2 * x - 14)^(1/3) = -2) : 2 * x + 3 = 9 := by
  sorry

end NUMINAMATH_GPT_cube_root_equation_l1695_169571


namespace NUMINAMATH_GPT_right_triangle_angle_l1695_169588

open Real

theorem right_triangle_angle (a b c : ℝ) (h : a^2 + b^2 = c^2) (h2 : c^2 = 2 * a * b) : 
  ∃ θ : ℝ, θ = 45 ∧ tan θ = a / b := 
by sorry

end NUMINAMATH_GPT_right_triangle_angle_l1695_169588


namespace NUMINAMATH_GPT_number_of_teams_l1695_169509

theorem number_of_teams (n : ℕ) (h : (n * (n - 1)) / 2 * 10 = 1050) : n = 15 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_teams_l1695_169509


namespace NUMINAMATH_GPT_intersection_complement_l1695_169536

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5}

theorem intersection_complement : P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l1695_169536


namespace NUMINAMATH_GPT_eval_g_at_3_l1695_169548

def g (x : ℤ) : ℤ := 5 * x^3 + 7 * x^2 - 3 * x - 6

theorem eval_g_at_3 : g 3 = 183 := by
  sorry

end NUMINAMATH_GPT_eval_g_at_3_l1695_169548


namespace NUMINAMATH_GPT_record_loss_of_10_l1695_169599

-- Definition of profit and loss recording
def record (x : Int) : Int :=
  if x ≥ 0 then x else -x

-- Condition: A profit of $20 should be recorded as +$20
axiom profit_recording : ∀ (p : Int), p ≥ 0 → record p = p

-- Condition: A loss should be recorded as a negative amount
axiom loss_recording : ∀ (l : Int), l < 0 → record l = l

-- Question: How should a loss of $10 be recorded?
-- Prove that if a small store lost $10, it should be recorded as -$10
theorem record_loss_of_10 : record (-10) = -10 :=
by sorry

end NUMINAMATH_GPT_record_loss_of_10_l1695_169599


namespace NUMINAMATH_GPT_ratio_of_arithmetic_sequence_sums_l1695_169566

-- Definitions of the arithmetic sequences based on the conditions
def numerator_seq (n : ℕ) : ℕ := 3 + (n - 1) * 3
def denominator_seq (m : ℕ) : ℕ := 4 + (m - 1) * 4

-- Definitions of the number of terms based on the conditions
def num_terms_num : ℕ := 32
def num_terms_den : ℕ := 16

-- Definitions of the sums based on the sequences
def sum_numerator_seq : ℕ := (num_terms_num / 2) * (3 + 96)
def sum_denominator_seq : ℕ := (num_terms_den / 2) * (4 + 64)

-- Calculate the ratio of the sums
def ratio_of_sums : ℚ := sum_numerator_seq / sum_denominator_seq

-- Proof statement
theorem ratio_of_arithmetic_sequence_sums : ratio_of_sums = 99 / 34 := by
  sorry

end NUMINAMATH_GPT_ratio_of_arithmetic_sequence_sums_l1695_169566


namespace NUMINAMATH_GPT_kevin_total_hops_l1695_169533

/-- Define the hop function for Kevin -/
def hop (remaining_distance : ℚ) : ℚ :=
  remaining_distance / 4

/-- Summing the series for five hops -/
def total_hops (start_distance : ℚ) (hops : ℕ) : ℚ :=
  let h0 := hop start_distance
  let h1 := hop (start_distance - h0)
  let h2 := hop (start_distance - h0 - h1)
  let h3 := hop (start_distance - h0 - h1 - h2)
  let h4 := hop (start_distance - h0 - h1 - h2 - h3)
  h0 + h1 + h2 + h3 + h4

/-- Final proof statement: after five hops from starting distance of 2, total distance hopped should be 1031769/2359296 -/
theorem kevin_total_hops :
  total_hops 2 5 = 1031769 / 2359296 :=
sorry

end NUMINAMATH_GPT_kevin_total_hops_l1695_169533


namespace NUMINAMATH_GPT_k_range_l1695_169591

def y_increasing (k : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → k * x₁ + 1 < k * x₂ + 1
def y_max_min (k : ℝ) : Prop := (∃ x : ℝ, 0 ≤ x ∧ x ≤ k ∧ (x^2 - 2 * x + 3 = 2)) ∧ (∃ x : ℝ, 0 ≤ x ∧ x ≤ k ∧ (x^2 - 2 * x + 3 = 3))

theorem k_range (k : ℝ) (hk : (¬ (0 < k ∧ y_max_min k) ∧ (0 < k ∨ y_max_min k))) : 
  (0 < k ∧ k < 1) ∨ (k > 2) :=
sorry

end NUMINAMATH_GPT_k_range_l1695_169591


namespace NUMINAMATH_GPT_find_a_for_no_x2_term_l1695_169564

theorem find_a_for_no_x2_term :
  ∀ a : ℝ, (∀ x : ℝ, (3 * x^2 + 2 * a * x + 1) * (-3 * x) - 4 * x^2 = -9 * x^3 + (-6 * a - 4) * x^2 - 3 * x) →
  (¬ ∃ x : ℝ, (-6 * a - 4) * x^2 ≠ 0) →
  a = -2 / 3 :=
by
  intros a h1 h2
  sorry

end NUMINAMATH_GPT_find_a_for_no_x2_term_l1695_169564


namespace NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l1695_169547

-- Define the function f
def f (a x : ℝ) : ℝ := |3 * x - 1| + a * x + 3

-- Problem 1: When a = 1, solve the inequality f(x) ≤ 5
theorem part1_solution_set : 
  { x : ℝ | f 1 x ≤ 5 } = {x : ℝ | -1 / 2 ≤ x ∧ x ≤ 3 / 4} := 
  by 
  sorry

-- Problem 2: Determine the range of a for which f(x) has a minimum
theorem part2_range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, x < 1/3 → f a x ≤ f a 1/3) → 
           (∀ x : ℝ, x ≥ 1/3 → f a x ≥ f a 1/3) ↔ 
           (-3 ≤ a ∧ a ≤ 3) := 
  by
  sorry

end NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l1695_169547


namespace NUMINAMATH_GPT_parallel_if_perp_to_plane_l1695_169572

variable {α m n : Type}

variables (plane : α) (line_m line_n : m)

-- Define what it means for lines to be perpendicular to a plane
def perpendicular_to_plane (line : m) (pl : α) : Prop := sorry

-- Define what it means for lines to be parallel
def parallel (line1 line2 : m) : Prop := sorry

-- The conditions
axiom perp_1 : perpendicular_to_plane line_m plane
axiom perp_2 : perpendicular_to_plane line_n plane

-- The theorem to prove
theorem parallel_if_perp_to_plane : parallel line_m line_n := sorry

end NUMINAMATH_GPT_parallel_if_perp_to_plane_l1695_169572


namespace NUMINAMATH_GPT_cos_A_is_one_l1695_169508

-- Definitions as per Lean's requirement
variable (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]

-- Declaring the conditions are given
variables (α : ℝ) (cos_A : ℝ)
variables (AB CD AD BC : ℝ)
def is_convex_quadrilateral (A B C D : Type) : Prop := 
  sorry -- This would be a formal definition of convex quadrilateral

-- The conditions are specified in Lean terms
variables (h1 : is_convex_quadrilateral A B C D)
variables (h2 : α = 0) -- α = 0 implies cos(α) = 1
variables (h3 : AB = 240)
variables (h4 : CD = 240)
variables (h5 : AD ≠ BC)
variables (h6 : AB + CD + AD + BC = 960)

-- The proof statement to indicate that cos(α) = 1 under the given conditions
theorem cos_A_is_one : cos_A = 1 :=
by
  sorry -- Proof not included as per the instruction

end NUMINAMATH_GPT_cos_A_is_one_l1695_169508


namespace NUMINAMATH_GPT_ruda_received_clock_on_correct_date_l1695_169590

/-- Ruda's clock problem -/
def ruda_clock_problem : Prop :=
  ∃ receive_date : ℕ → ℕ × ℕ × ℕ, -- A function mapping the number of presses to a date (Year, Month, Day)
  (∀ days_after_received, 
    receive_date days_after_received = 
    if days_after_received <= 45 then (2022, 10, 27 - (45 - days_after_received)) -- Calculating the receive date.
    else receive_date 45)
  ∧
  receive_date 45 = (2022, 12, 11) -- The day he checked the clock has to be December 11th

-- We want to prove that:
theorem ruda_received_clock_on_correct_date : ruda_clock_problem :=
by
  sorry

end NUMINAMATH_GPT_ruda_received_clock_on_correct_date_l1695_169590


namespace NUMINAMATH_GPT_multiply_negatives_l1695_169594

theorem multiply_negatives : (-3) * (-4) * (-1) = -12 := 
by sorry

end NUMINAMATH_GPT_multiply_negatives_l1695_169594


namespace NUMINAMATH_GPT_root_and_value_of_a_equation_has_real_roots_l1695_169529

theorem root_and_value_of_a (a : ℝ) (other_root : ℝ) :
  (∃ x : ℝ, x^2 + a * x + a - 1 = 0 ∧ x = 2) → a = -1 ∧ other_root = -1 :=
by sorry

theorem equation_has_real_roots (a : ℝ) :
  ∃ x : ℝ, x^2 + a * x + a - 1 = 0 :=
by sorry

end NUMINAMATH_GPT_root_and_value_of_a_equation_has_real_roots_l1695_169529


namespace NUMINAMATH_GPT_tens_digit_of_2013_squared_minus_2013_l1695_169539

theorem tens_digit_of_2013_squared_minus_2013 : (2013^2 - 2013) % 100 / 10 = 5 := by
  sorry

end NUMINAMATH_GPT_tens_digit_of_2013_squared_minus_2013_l1695_169539


namespace NUMINAMATH_GPT_gcd_xyz_square_of_diff_l1695_169504

theorem gcd_xyz_square_of_diff {x y z : ℕ} 
    (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) : 
    ∃ n : ℕ, Nat.gcd x (Nat.gcd y z) * (y - x) = n ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_gcd_xyz_square_of_diff_l1695_169504


namespace NUMINAMATH_GPT_max_value_trig_expression_l1695_169565

variable (a b φ θ : ℝ)

theorem max_value_trig_expression :
  ∃ θ : ℝ, a * Real.cos θ + b * Real.sin (θ + φ) ≤ Real.sqrt (a^2 + 2 * a * b * Real.sin φ + b^2) := sorry

end NUMINAMATH_GPT_max_value_trig_expression_l1695_169565


namespace NUMINAMATH_GPT_max_intersections_l1695_169507

/-- Given two different circles and three different straight lines, the maximum number of
points of intersection on a plane is 17. -/
theorem max_intersections (c1 c2 : Circle) (l1 l2 l3 : Line) (h_distinct_cir : c1 ≠ c2) (h_distinct_lines : ∀ (l1 l2 : Line), l1 ≠ l2) :
  ∃ (n : ℕ), n = 17 :=
by
  sorry

end NUMINAMATH_GPT_max_intersections_l1695_169507


namespace NUMINAMATH_GPT_max_value_of_k_proof_l1695_169589

noncomputable def maximum_value_of_k (x y k : ℝ) (h1: x > 0) (h2: y > 0) (h3: k > 0) 
(h4: 5 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) : Prop :=
  k = (-1 + Real.sqrt 17) / 2

-- This is the statement that needs to be proven:
theorem max_value_of_k_proof (x y k : ℝ) (h1: x > 0) (h2: y > 0) (h3: k > 0) 
(h4: 5 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) : maximum_value_of_k x y k h1 h2 h3 h4 :=
sorry

end NUMINAMATH_GPT_max_value_of_k_proof_l1695_169589


namespace NUMINAMATH_GPT_find_p_4_l1695_169582

-- Define the polynomial p(x)
def p (x : ℕ) : ℚ := sorry

-- Given conditions
axiom h1 : p 1 = 1
axiom h2 : p 2 = 1 / 4
axiom h3 : p 3 = 1 / 9
axiom h4 : p 5 = 1 / 25

-- Prove that p(4) = -1/30
theorem find_p_4 : p 4 = -1 / 30 := 
  by sorry

end NUMINAMATH_GPT_find_p_4_l1695_169582


namespace NUMINAMATH_GPT_number_of_violas_l1695_169573

theorem number_of_violas (V : ℕ) 
  (cellos : ℕ := 800) 
  (pairs : ℕ := 70) 
  (probability : ℝ := 0.00014583333333333335) 
  (h : probability = pairs / (cellos * V)) : V = 600 :=
by
  sorry

end NUMINAMATH_GPT_number_of_violas_l1695_169573


namespace NUMINAMATH_GPT_negation_of_all_men_are_tall_l1695_169506

variable {α : Type}
variable (man : α → Prop) (tall : α → Prop)

theorem negation_of_all_men_are_tall :
  (¬ ∀ x, man x → tall x) ↔ ∃ x, man x ∧ ¬ tall x :=
sorry

end NUMINAMATH_GPT_negation_of_all_men_are_tall_l1695_169506


namespace NUMINAMATH_GPT_cost_of_60_tulips_l1695_169501

-- Definition of conditions
def cost_of_bouquet (n : ℕ) : ℝ :=
  if n ≤ 40 then n * 2
  else 40 * 2 + (n - 40) * 3

-- The main statement
theorem cost_of_60_tulips : cost_of_bouquet 60 = 140 := by
  sorry

end NUMINAMATH_GPT_cost_of_60_tulips_l1695_169501


namespace NUMINAMATH_GPT_find_reals_abc_d_l1695_169505

theorem find_reals_abc_d (a b c d : ℝ)
  (h1 : a * b * c + a * b + b * c + c * a + a + b + c = 1)
  (h2 : b * c * d + b * c + c * d + d * b + b + c + d = 9)
  (h3 : c * d * a + c * d + d * a + a * c + c + d + a = 9)
  (h4 : d * a * b + d * a + a * b + b * d + d + a + b = 9) :
  a = b ∧ b = c ∧ c = (2 : ℝ)^(1/3) - 1 ∧ d = 5 * (2 : ℝ)^(1/3) - 1 :=
sorry

end NUMINAMATH_GPT_find_reals_abc_d_l1695_169505


namespace NUMINAMATH_GPT_selling_price_decreased_l1695_169516

theorem selling_price_decreased (d m : ℝ) (hd : d = 0.10) (hm : m = 0.10) :
  (1 - d) * (1 + m) < 1 :=
by
  rw [hd, hm]
  sorry

end NUMINAMATH_GPT_selling_price_decreased_l1695_169516


namespace NUMINAMATH_GPT_sum_of_integers_l1695_169574

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 144) : x + y = 24 :=
sorry

end NUMINAMATH_GPT_sum_of_integers_l1695_169574


namespace NUMINAMATH_GPT_initial_stock_decaf_percentage_l1695_169557

variable (x : ℝ)
variable (initialStock newStock totalStock initialDecaf newDecaf totalDecaf: ℝ)

theorem initial_stock_decaf_percentage :
  initialStock = 400 ->
  newStock = 100 ->
  totalStock = 500 ->
  initialDecaf = initialStock * x / 100 ->
  newDecaf = newStock * 60 / 100 ->
  totalDecaf = 180 ->
  initialDecaf + newDecaf = totalDecaf ->
  x = 30 := by
  intros h₁ h₂ h₃ h₄ h₅ h₆ h₇
  sorry

end NUMINAMATH_GPT_initial_stock_decaf_percentage_l1695_169557


namespace NUMINAMATH_GPT_min_value_four_l1695_169520

noncomputable def min_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : y > 2 * x) : ℝ :=
  (y^2 - 2 * x * y + x^2) / (x * y - 2 * x^2)

theorem min_value_four (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hy_gt_2x : y > 2 * x) :
  min_value x y hx_pos hy_pos hy_gt_2x = 4 := 
sorry

end NUMINAMATH_GPT_min_value_four_l1695_169520


namespace NUMINAMATH_GPT_zero_in_interval_l1695_169522

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem zero_in_interval (h_mono : ∀ x y, 0 < x → x < y → f x < f y) (h_f2 : f 2 < 0) (h_f3 : 0 < f 3) :
  ∃ x₀ ∈ (Set.Ioo 2 3), f x₀ = 0 :=
by
  sorry

end NUMINAMATH_GPT_zero_in_interval_l1695_169522


namespace NUMINAMATH_GPT_morning_snowfall_l1695_169570

theorem morning_snowfall (afternoon_snowfall total_snowfall : ℝ) (h₀ : afternoon_snowfall = 0.5) (h₁ : total_snowfall = 0.63):
  total_snowfall - afternoon_snowfall = 0.13 :=
by 
  sorry

end NUMINAMATH_GPT_morning_snowfall_l1695_169570


namespace NUMINAMATH_GPT_first_dimension_length_l1695_169543

-- Definitions for conditions
def tank_surface_area (x : ℝ) : ℝ := 14 * x + 20
def cost_per_sqft : ℝ := 20
def total_cost (x : ℝ) : ℝ := (tank_surface_area x) * cost_per_sqft

-- The theorem we need to prove
theorem first_dimension_length : ∃ x : ℝ, total_cost x = 1520 ∧ x = 4 := by 
  sorry

end NUMINAMATH_GPT_first_dimension_length_l1695_169543


namespace NUMINAMATH_GPT_max_value_expression_correct_l1695_169577

noncomputable def max_value_expression (a b c d : ℝ) : ℝ :=
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_value_expression_correct :
  ∃ a b c d : ℝ, a ∈ Set.Icc (-13.5) 13.5 ∧ b ∈ Set.Icc (-13.5) 13.5 ∧ 
                  c ∈ Set.Icc (-13.5) 13.5 ∧ d ∈ Set.Icc (-13.5) 13.5 ∧ 
                  max_value_expression a b c d = 756 := 
sorry

end NUMINAMATH_GPT_max_value_expression_correct_l1695_169577


namespace NUMINAMATH_GPT_remainder_13_pow_2031_mod_100_l1695_169592

theorem remainder_13_pow_2031_mod_100 : (13^2031) % 100 = 17 :=
by sorry

end NUMINAMATH_GPT_remainder_13_pow_2031_mod_100_l1695_169592


namespace NUMINAMATH_GPT_exchange_rate_change_2014_l1695_169551

theorem exchange_rate_change_2014 :
  let init_rate := 32.6587
  let final_rate := 56.2584
  let change := final_rate - init_rate
  let rounded_change := Float.round change
  rounded_change = 24 :=
by
  sorry

end NUMINAMATH_GPT_exchange_rate_change_2014_l1695_169551


namespace NUMINAMATH_GPT_coin_flip_sequences_l1695_169517

theorem coin_flip_sequences (n : ℕ) : (2^10 = 1024) := 
by rfl

end NUMINAMATH_GPT_coin_flip_sequences_l1695_169517


namespace NUMINAMATH_GPT_not_parallel_to_a_l1695_169596

noncomputable def is_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * u.1, k * u.2)

theorem not_parallel_to_a : ∀ k : ℝ, ¬ is_parallel (k^2 + 1, k^2 + 1) (1, -2) :=
sorry

end NUMINAMATH_GPT_not_parallel_to_a_l1695_169596


namespace NUMINAMATH_GPT_percentage_of_girls_l1695_169576

theorem percentage_of_girls (B G : ℕ) (h1 : B + G = 900) (h2 : B = 90) :
  (G / (B + G) : ℚ) * 100 = 90 :=
  by
  sorry

end NUMINAMATH_GPT_percentage_of_girls_l1695_169576


namespace NUMINAMATH_GPT_train_crosses_second_platform_l1695_169513

theorem train_crosses_second_platform (
  length_train length_platform1 length_platform2 : ℝ) 
  (time_platform1 : ℝ) 
  (H1 : length_train = 100)
  (H2 : length_platform1 = 200)
  (H3 : length_platform2 = 300)
  (H4 : time_platform1 = 15) :
  ∃ t : ℝ, t = 20 := by
  sorry

end NUMINAMATH_GPT_train_crosses_second_platform_l1695_169513


namespace NUMINAMATH_GPT_calculate_result_l1695_169546

def multiply (a b : ℕ) : ℕ := a * b
def subtract (a b : ℕ) : ℕ := a - b
def three_fifths (a : ℕ) : ℕ := 3 * a / 5

theorem calculate_result :
  let result := three_fifths (subtract (multiply 12 10) 20)
  result = 60 :=
by
  sorry

end NUMINAMATH_GPT_calculate_result_l1695_169546


namespace NUMINAMATH_GPT_male_contestants_count_l1695_169569

-- Define the total number of contestants.
def total_contestants : ℕ := 18

-- A third of the contestants are female.
def female_contestants : ℕ := total_contestants / 3

-- The rest are male.
def male_contestants : ℕ := total_contestants - female_contestants

-- The theorem to prove
theorem male_contestants_count : male_contestants = 12 := by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_male_contestants_count_l1695_169569


namespace NUMINAMATH_GPT_piggy_bank_balance_l1695_169583

theorem piggy_bank_balance (original_amount : ℕ) (taken_out : ℕ) : original_amount = 5 ∧ taken_out = 2 → original_amount - taken_out = 3 :=
by sorry

end NUMINAMATH_GPT_piggy_bank_balance_l1695_169583


namespace NUMINAMATH_GPT_value_of_2a_plus_b_l1695_169560

theorem value_of_2a_plus_b (a b : ℤ) (h1 : |a - 1| = 4) (h2 : |b| = 7) (h3 : |a + b| ≠ a + b) :
  2 * a + b = 3 ∨ 2 * a + b = -13 := sorry

end NUMINAMATH_GPT_value_of_2a_plus_b_l1695_169560


namespace NUMINAMATH_GPT_simplify_fraction_l1695_169525

theorem simplify_fraction : 5 * (18 / 7) * (21 / -54) = -5 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1695_169525


namespace NUMINAMATH_GPT_perimeter_of_park_l1695_169535

def length := 300
def breadth := 200

theorem perimeter_of_park : 2 * (length + breadth) = 1000 := by
  sorry

end NUMINAMATH_GPT_perimeter_of_park_l1695_169535


namespace NUMINAMATH_GPT_find_quadratic_function_l1695_169500

-- Define the quadratic function
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant condition
def discriminant_zero (a b c : ℝ) : Prop := b^2 - 4 * a * c = 0

-- Given derivative
def given_derivative (x : ℝ) : ℝ := 2 * x + 2

-- Prove that if these conditions hold, then f(x) = x^2 + 2x + 1
theorem find_quadratic_function :
  ∃ (a b c : ℝ), (∀ x, quadratic_function a b c x = 0 → discriminant_zero a b c) ∧
                (∀ x, (2 * a * x + b) = given_derivative x) ∧
                (quadratic_function a b c x = x^2 + 2 * x + 1) := 
by
  sorry

end NUMINAMATH_GPT_find_quadratic_function_l1695_169500


namespace NUMINAMATH_GPT_emily_spending_l1695_169581

theorem emily_spending (X Y : ℝ) 
  (h1 : (X + 2*X + 3*X + 12*X) = Y) : 
  X = Y / 18 := 
by
  sorry

end NUMINAMATH_GPT_emily_spending_l1695_169581


namespace NUMINAMATH_GPT_average_speed_correct_l1695_169519

-- Define the speeds for each hour
def speed_hour1 := 90 -- km/h
def speed_hour2 := 40 -- km/h
def speed_hour3 := 60 -- km/h
def speed_hour4 := 80 -- km/h
def speed_hour5 := 50 -- km/h

-- Define the total time of the journey
def total_time := 5 -- hours

-- Calculate the sum of distances
def total_distance := speed_hour1 + speed_hour2 + speed_hour3 + speed_hour4 + speed_hour5

-- Define the average speed calculation
def average_speed := total_distance / total_time

-- The proof problem: average speed is 64 km/h
theorem average_speed_correct : average_speed = 64 := by
  sorry

end NUMINAMATH_GPT_average_speed_correct_l1695_169519


namespace NUMINAMATH_GPT_weight_of_b_l1695_169562

-- Definitions based on conditions
variables (A B C : ℝ)

def avg_abc := (A + B + C) / 3 = 45
def avg_ab := (A + B) / 2 = 40
def avg_bc := (B + C) / 2 = 44

-- The theorem to prove
theorem weight_of_b (h1 : avg_abc A B C) (h2 : avg_ab A B) (h3 : avg_bc B C) :
  B = 33 :=
sorry

end NUMINAMATH_GPT_weight_of_b_l1695_169562


namespace NUMINAMATH_GPT_area_of_EPGQ_l1695_169586

noncomputable def area_of_region (length_rect width_rect half_length_rect : ℝ) : ℝ :=
  half_length_rect * width_rect

theorem area_of_EPGQ :
  let length_rect := 10.0
  let width_rect := 6.0
  let P_half_length := length_rect / 2
  let Q_half_length := length_rect / 2
  (area_of_region length_rect width_rect P_half_length) = 30.0 :=
by
  sorry

end NUMINAMATH_GPT_area_of_EPGQ_l1695_169586


namespace NUMINAMATH_GPT_solve_absolute_value_equation_l1695_169585

theorem solve_absolute_value_equation (y : ℝ) : (|y - 4| + 3 * y = 15) ↔ (y = 19 / 4) := by
  sorry

end NUMINAMATH_GPT_solve_absolute_value_equation_l1695_169585


namespace NUMINAMATH_GPT_vladimir_can_invest_more_profitably_l1695_169502

-- Conditions and parameters
def p_buckwheat_initial : ℝ := 70 -- initial price of buckwheat in RUB/kg
def p_buckwheat_2017 : ℝ := 85 -- price of buckwheat in early 2017 in RUB/kg
def rate_2015 : ℝ := 0.16 -- interest rate for annual deposit in 2015
def rate_2016 : ℝ := 0.10 -- interest rate for annual deposit in 2016
def rate_2yr : ℝ := 0.15 -- interest rate for two-year deposit per year

-- Amounts after investments
def amount_annual : ℝ := p_buckwheat_initial * (1 + rate_2015) * (1 + rate_2016)
def amount_2yr : ℝ := p_buckwheat_initial * (1 + rate_2yr)^2

-- Prove that the best investment amount is greater than the 2017 buckwheat price
theorem vladimir_can_invest_more_profitably : max amount_annual amount_2yr > p_buckwheat_2017 := by
  sorry

end NUMINAMATH_GPT_vladimir_can_invest_more_profitably_l1695_169502


namespace NUMINAMATH_GPT_carpenter_material_cost_l1695_169559

theorem carpenter_material_cost (total_estimate hourly_rate num_hours : ℝ) 
    (h1 : total_estimate = 980)
    (h2 : hourly_rate = 28)
    (h3 : num_hours = 15) : 
    total_estimate - hourly_rate * num_hours = 560 := 
by
  sorry

end NUMINAMATH_GPT_carpenter_material_cost_l1695_169559


namespace NUMINAMATH_GPT_problem_1_l1695_169515

variable (x : ℝ) (a : ℝ)

theorem problem_1 (h1 : x - 1/x = 3) (h2 : a = x^2 + 1/x^2) : a = 11 := sorry

end NUMINAMATH_GPT_problem_1_l1695_169515


namespace NUMINAMATH_GPT_James_leftover_money_l1695_169558

variable (W : ℝ)
variable (M : ℝ)

theorem James_leftover_money 
  (h1 : M = (W / 2 - 2))
  (h2 : M + 114 = W) : 
  M = 110 := sorry

end NUMINAMATH_GPT_James_leftover_money_l1695_169558


namespace NUMINAMATH_GPT_horse_revolutions_l1695_169531

theorem horse_revolutions :
  ∀ (r_1 r_2 : ℝ) (n : ℕ),
    r_1 = 30 → r_2 = 10 → n = 25 → (r_1 * n) / r_2 = 75 := by
  sorry

end NUMINAMATH_GPT_horse_revolutions_l1695_169531


namespace NUMINAMATH_GPT_neg_exists_eq_forall_ne_l1695_169518

variable (x : ℝ)

theorem neg_exists_eq_forall_ne : (¬ ∃ x : ℝ, x^2 - 2 * x = 0) ↔ ∀ x : ℝ, x^2 - 2 * x ≠ 0 := by
  sorry

end NUMINAMATH_GPT_neg_exists_eq_forall_ne_l1695_169518


namespace NUMINAMATH_GPT_money_lent_to_B_l1695_169549

theorem money_lent_to_B (total_money : ℕ) (interest_A_rate : ℚ) (interest_B_rate : ℚ) (interest_difference : ℚ) (years : ℕ) 
  (x y : ℚ) 
  (h1 : total_money = 10000)
  (h2 : interest_A_rate = 0.15)
  (h3 : interest_B_rate = 0.18)
  (h4 : interest_difference = 360)
  (h5 : years = 2)
  (h6 : y = total_money - x)
  (h7 : ((x * interest_A_rate * years) = ((y * interest_B_rate * years) + interest_difference))) : 
  y = 4000 := 
sorry

end NUMINAMATH_GPT_money_lent_to_B_l1695_169549


namespace NUMINAMATH_GPT_probability_light_change_l1695_169595

noncomputable def total_cycle_duration : ℕ := 45 + 5 + 50
def change_intervals : ℕ := 15

theorem probability_light_change :
  (15 : ℚ) / total_cycle_duration = 3 / 20 :=
by
  sorry

end NUMINAMATH_GPT_probability_light_change_l1695_169595


namespace NUMINAMATH_GPT_exam_passing_marks_l1695_169530

theorem exam_passing_marks (T P : ℝ) 
  (h1 : 0.30 * T = P - 60) 
  (h2 : 0.40 * T + 10 = P) 
  (h3 : 0.50 * T - 5 = P + 40) : 
  P = 210 := 
sorry

end NUMINAMATH_GPT_exam_passing_marks_l1695_169530


namespace NUMINAMATH_GPT_gcd_1729_867_l1695_169580

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_1729_867_l1695_169580


namespace NUMINAMATH_GPT_scheme_A_yield_percentage_l1695_169587

-- Define the initial investments and yields
def initial_investment_A : ℝ := 300
def initial_investment_B : ℝ := 200
def yield_B : ℝ := 0.5 -- 50% yield

-- Define the equation given in the problem
def yield_A_equation (P : ℝ) : Prop :=
  initial_investment_A + (initial_investment_A * (P / 100)) = initial_investment_B + (initial_investment_B * yield_B) + 90

-- The proof statement we need to prove
theorem scheme_A_yield_percentage : yield_A_equation 30 :=
by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_scheme_A_yield_percentage_l1695_169587


namespace NUMINAMATH_GPT_duck_travel_days_l1695_169537

theorem duck_travel_days (x : ℕ) (h1 : 40 + 2 * 40 + x = 180) : x = 60 := by
  sorry

end NUMINAMATH_GPT_duck_travel_days_l1695_169537


namespace NUMINAMATH_GPT_find_xyz_l1695_169540

variables (x y z s : ℝ)

theorem find_xyz (h₁ : (x + y + z) * (x * y + x * z + y * z) = 12)
    (h₂ : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 0)
    (hs : x + y + z = s) : xyz = -8 :=
by
  sorry

end NUMINAMATH_GPT_find_xyz_l1695_169540


namespace NUMINAMATH_GPT_alice_reeboks_sold_l1695_169553

theorem alice_reeboks_sold
  (quota : ℝ)
  (price_adidas : ℝ)
  (price_nike : ℝ)
  (price_reeboks : ℝ)
  (num_nike : ℕ)
  (num_adidas : ℕ)
  (excess : ℝ)
  (total_sales_goal : ℝ)
  (total_sales : ℝ)
  (sales_nikes_adidas : ℝ)
  (sales_reeboks : ℝ)
  (num_reeboks : ℕ) :
  quota = 1000 →
  price_adidas = 45 →
  price_nike = 60 →
  price_reeboks = 35 →
  num_nike = 8 →
  num_adidas = 6 →
  excess = 65 →
  total_sales_goal = quota + excess →
  total_sales = 1065 →
  sales_nikes_adidas = price_nike * num_nike + price_adidas * num_adidas →
  sales_reeboks = total_sales - sales_nikes_adidas →
  num_reeboks = sales_reeboks / price_reeboks →
  num_reeboks = 9 :=
by
  intros
  sorry

end NUMINAMATH_GPT_alice_reeboks_sold_l1695_169553


namespace NUMINAMATH_GPT_selected_number_in_first_group_is_7_l1695_169545

def N : ℕ := 800
def k : ℕ := 50
def interval : ℕ := N / k
def selected_number : ℕ := 39
def second_group_start : ℕ := 33
def second_group_end : ℕ := 48

theorem selected_number_in_first_group_is_7 
  (h1 : interval = 16)
  (h2 : selected_number ≥ second_group_start ∧ selected_number ≤ second_group_end)
  (h3 : ∃ n, selected_number = second_group_start + interval * n - 1) :
  selected_number % interval = 7 :=
sorry

end NUMINAMATH_GPT_selected_number_in_first_group_is_7_l1695_169545


namespace NUMINAMATH_GPT_distance_Tim_covers_l1695_169598

theorem distance_Tim_covers (initial_distance : ℕ) (tim_speed elan_speed : ℕ) (double_speed_time : ℕ)
  (h_initial_distance : initial_distance = 30)
  (h_tim_speed : tim_speed = 10)
  (h_elan_speed : elan_speed = 5)
  (h_double_speed_time : double_speed_time = 1) :
  ∃ t d : ℕ, d = 20 ∧ t ∈ {t | t = d / tim_speed + (initial_distance - d) / (tim_speed * 2)} :=
sorry

end NUMINAMATH_GPT_distance_Tim_covers_l1695_169598


namespace NUMINAMATH_GPT_cost_to_fill_sandbox_l1695_169567

-- Definitions for conditions
def side_length : ℝ := 3
def volume_per_bag : ℝ := 3
def cost_per_bag : ℝ := 4

-- Theorem statement
theorem cost_to_fill_sandbox : (side_length ^ 3 / volume_per_bag * cost_per_bag) = 36 := by
  sorry

end NUMINAMATH_GPT_cost_to_fill_sandbox_l1695_169567


namespace NUMINAMATH_GPT_product_of_two_numbers_l1695_169524

-- Define the conditions
def two_numbers (x y : ℝ) : Prop :=
  x + y = 27 ∧ x - y = 7

-- Define the product function
def product_two_numbers (x y : ℝ) : ℝ := x * y

-- State the theorem
theorem product_of_two_numbers : ∃ x y : ℝ, two_numbers x y ∧ product_two_numbers x y = 170 := by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1695_169524
