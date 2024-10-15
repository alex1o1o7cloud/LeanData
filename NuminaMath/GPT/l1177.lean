import Mathlib

namespace NUMINAMATH_GPT_older_sister_age_l1177_117765

theorem older_sister_age (x : ℕ) (older_sister_age : ℕ) (h1 : older_sister_age = 3 * x)
  (h2 : older_sister_age + 2 = 2 * (x + 2)) : older_sister_age = 6 :=
by
  sorry

end NUMINAMATH_GPT_older_sister_age_l1177_117765


namespace NUMINAMATH_GPT_find_k_l1177_117768

theorem find_k (k : ℝ) 
  (h1 : ∀ (r s : ℝ), r + s = -k ∧ r * s = 8 → (r + 3) + (s + 3) = k) : 
  k = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1177_117768


namespace NUMINAMATH_GPT_part_a_part_b_l1177_117726

-- Part (a)
theorem part_a (students : Fin 67) (answers : Fin 6 → Bool) :
  ∃ (s1 s2 : Fin 67), s1 ≠ s2 ∧ answers s1 = answers s2 := by
  sorry

-- Part (b)
theorem part_b (students : Fin 67) (points : Fin 6 → ℤ)
  (h_points : ∀ k, points k = k ∨ points k = -k) :
  ∃ (scores : Fin 67 → ℤ), ∃ (s1 s2 s3 s4 : Fin 67),
  s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s2 ≠ s3 ∧ s2 ≠ s4 ∧ s3 ≠ s4 ∧
  scores s1 = scores s2 ∧ scores s1 = scores s3 ∧ scores s1 = scores s4 := by
  sorry

end NUMINAMATH_GPT_part_a_part_b_l1177_117726


namespace NUMINAMATH_GPT_eval_arith_expression_l1177_117716

theorem eval_arith_expression : 2 + 3^2 * 4 - 5 + 6 / 2 = 36 := 
by sorry

end NUMINAMATH_GPT_eval_arith_expression_l1177_117716


namespace NUMINAMATH_GPT_find_a_l1177_117730

variable (a b c : ℤ)

theorem find_a (h1 : a + b = 2) (h2 : b + c = 0) (h3 : |c| = 1) : a = 3 ∨ a = 1 := 
sorry

end NUMINAMATH_GPT_find_a_l1177_117730


namespace NUMINAMATH_GPT_range_of_a_l1177_117729

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + a * x + 1 ≥ 0) ↔ -2 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1177_117729


namespace NUMINAMATH_GPT_intersecting_lines_angle_difference_l1177_117745

-- Define the conditions
def angle_y : ℝ := 40
def straight_angle_sum : ℝ := 180

-- Define the variables representing the angles
variable (x y : ℝ)

-- Define the proof problem
theorem intersecting_lines_angle_difference : 
  ∀ x y : ℝ, 
  y = angle_y → 
  (∃ (a b : ℝ), a + b = straight_angle_sum ∧ a = y ∧ b = x) → 
  x - y = 100 :=
by
  intros x y hy h
  sorry

end NUMINAMATH_GPT_intersecting_lines_angle_difference_l1177_117745


namespace NUMINAMATH_GPT_cricket_initial_matches_l1177_117720

theorem cricket_initial_matches (x : ℝ) :
  (0.28 * x + 60 = 0.52 * (x + 60)) → x = 120 :=
by
  sorry

end NUMINAMATH_GPT_cricket_initial_matches_l1177_117720


namespace NUMINAMATH_GPT_find_number_l1177_117797

theorem find_number (N x : ℝ) (h1 : x = 1) (h2 : N / (4 + 1 / x) = 1) : N = 5 := 
by 
  sorry

end NUMINAMATH_GPT_find_number_l1177_117797


namespace NUMINAMATH_GPT_smallest_value_of_n_l1177_117708

theorem smallest_value_of_n : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (n + 6) % 7 = 0 ∧ (n - 9) % 4 = 0 ∧ n = 113 :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_of_n_l1177_117708


namespace NUMINAMATH_GPT_solve_for_X_l1177_117796

theorem solve_for_X (X : ℝ) (h : (X ^ (5 / 4)) = 32 * (32 ^ (1 / 16))) :
  X =  16 * (2 ^ (1 / 4)) :=
sorry

end NUMINAMATH_GPT_solve_for_X_l1177_117796


namespace NUMINAMATH_GPT_intersection_complement_eq_l1177_117707

def setA : Set ℝ := { x | (x - 6) * (x + 1) ≤ 0 }
def setB : Set ℝ := { x | x ≥ 2 }

theorem intersection_complement_eq :
  setA ∩ (Set.univ \ setB) = { x | -1 ≤ x ∧ x < 2 } := 
by 
  sorry

end NUMINAMATH_GPT_intersection_complement_eq_l1177_117707


namespace NUMINAMATH_GPT_range_of_m_l1177_117744

open Real

theorem range_of_m (a m y1 y2 : ℝ) (h_a_pos : a > 0)
  (hA : y1 = a * (m - 1)^2 + 4 * a * (m - 1) + 3)
  (hB : y2 = a * m^2 + 4 * a * m + 3)
  (h_y1_lt_y2 : y1 < y2) : 
  m > -3 / 2 := 
sorry

end NUMINAMATH_GPT_range_of_m_l1177_117744


namespace NUMINAMATH_GPT_find_y_l1177_117724

theorem find_y (x y : ℤ) (h1 : x + y = 250) (h2 : x - y = 200) : y = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1177_117724


namespace NUMINAMATH_GPT_pigeon_distance_l1177_117786

-- Define the conditions
def pigeon_trip (d : ℝ) (v : ℝ) (wind : ℝ) (time_nowind : ℝ) (time_wind : ℝ) :=
  (2 * d / v = time_nowind) ∧
  (d / (v + wind) + d / (v - wind) = time_wind)

-- Define the theorems to be proven
theorem pigeon_distance : ∃ (d : ℝ), pigeon_trip d 40 10 3.75 4 ∧ d = 75 :=
  by {
  sorry
}

end NUMINAMATH_GPT_pigeon_distance_l1177_117786


namespace NUMINAMATH_GPT_trapezoid_equilateral_triangle_ratio_l1177_117700

theorem trapezoid_equilateral_triangle_ratio (s d : ℝ) (AB CD : ℝ) 
  (h1 : AB = s) 
  (h2 : CD = 2 * d)
  (h3 : d = s) : 
  AB / CD = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_trapezoid_equilateral_triangle_ratio_l1177_117700


namespace NUMINAMATH_GPT_min_right_triangle_side_l1177_117784

theorem min_right_triangle_side (s : ℕ) : 
  (7^2 + 24^2 = s^2 ∧ 7 + 24 > s ∧ 24 + s > 7 ∧ 7 + s > 24) → s = 25 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_min_right_triangle_side_l1177_117784


namespace NUMINAMATH_GPT_single_elimination_matches_l1177_117723

theorem single_elimination_matches (players byes : ℕ)
  (h1 : players = 100)
  (h2 : byes = 28) :
  (players - 1) = 99 :=
by
  -- The proof would go here if it were needed
  sorry

end NUMINAMATH_GPT_single_elimination_matches_l1177_117723


namespace NUMINAMATH_GPT_spacy_subsets_15_l1177_117721

def spacy (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | 1     => 2
  | 2     => 3
  | n + 3 => spacy n + spacy (n-2)

theorem spacy_subsets_15 : spacy 15 = 406 := 
  sorry

end NUMINAMATH_GPT_spacy_subsets_15_l1177_117721


namespace NUMINAMATH_GPT_eventually_constant_sequence_a_floor_l1177_117731

noncomputable def sequence_a (n : ℕ) : ℝ := sorry
noncomputable def sequence_b (n : ℕ) : ℝ := sorry

axiom base_conditions : 
  (sequence_a 1 = 1) ∧
  (sequence_b 1 = 2) ∧
  (∀ n, sequence_a (n + 1) * sequence_b n = 1 + sequence_a n + sequence_a n * sequence_b n) ∧
  (∀ n, sequence_b (n + 1) * sequence_a n = 1 + sequence_b n + sequence_a n * sequence_b n)

theorem eventually_constant_sequence_a_floor:
  (∃ N, ∀ n ≥ N, 4 < sequence_a n ∧ sequence_a n < 5) →
  (∃ N, ∀ n ≥ N, Int.floor (sequence_a n) = 4) :=
sorry

end NUMINAMATH_GPT_eventually_constant_sequence_a_floor_l1177_117731


namespace NUMINAMATH_GPT_find_e_of_conditions_l1177_117776

noncomputable def Q (x : ℝ) (d e f : ℝ) : ℝ := x^3 + d * x^2 + e * x + f

theorem find_e_of_conditions (d e f : ℝ) 
  (h1 : f = 6) 
  (h2 : -d / 3 = -f)
  (h3 : -f = d + e + f - 1) : 
  e = -30 :=
by 
  sorry

end NUMINAMATH_GPT_find_e_of_conditions_l1177_117776


namespace NUMINAMATH_GPT_log_ordering_l1177_117792

noncomputable def P : ℝ := Real.log 3 / Real.log 2
noncomputable def Q : ℝ := Real.log 2 / Real.log 3
noncomputable def R : ℝ := Real.log (Real.log 2 / Real.log 3) / Real.log 2

theorem log_ordering (P Q R : ℝ) (h₁ : P = Real.log 3 / Real.log 2)
  (h₂ : Q = Real.log 2 / Real.log 3) (h₃ : R = Real.log (Real.log 2 / Real.log 3) / Real.log 2) :
  R < Q ∧ Q < P := by
  sorry

end NUMINAMATH_GPT_log_ordering_l1177_117792


namespace NUMINAMATH_GPT_product_of_variables_l1177_117752

theorem product_of_variables (a b c d : ℚ)
  (h1 : 4 * a + 5 * b + 7 * c + 9 * d = 56)
  (h2 : 4 * (d + c) = b)
  (h3 : 4 * b + 2 * c = a)
  (h4 : c - 2 = d) :
  a * b * c * d = 58653 / 10716361 := 
sorry

end NUMINAMATH_GPT_product_of_variables_l1177_117752


namespace NUMINAMATH_GPT_regression_line_l1177_117742

theorem regression_line (x y : ℝ) (m : ℝ) (x1 y1 : ℝ)
  (h_slope : m = 6.5)
  (h_point : (x1, y1) = (2, 3)) :
  (y - y1) = m * (x - x1) ↔ y = 6.5 * x - 10 :=
by
  sorry

end NUMINAMATH_GPT_regression_line_l1177_117742


namespace NUMINAMATH_GPT_range_of_m_for_hyperbola_l1177_117799

theorem range_of_m_for_hyperbola (m : ℝ) :
  (∃ (x y : ℝ), (m+2) ≠ 0 ∧ (m-2) ≠ 0 ∧ (x^2)/(m+2) + (y^2)/(m-2) = 1) ↔ (-2 < m ∧ m < 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_for_hyperbola_l1177_117799


namespace NUMINAMATH_GPT_find_a_l1177_117735

theorem find_a (a : ℝ) (h1 : 0 < a)
  (c1 : ∀ x y : ℝ, x^2 + y^2 = 4)
  (c2 : ∀ x y : ℝ, x^2 + y^2 + 2 * a * y - 6 = 0)
  (h_chord : (2 * Real.sqrt 3) = 2 * Real.sqrt 3) :
  a = 1 := 
sorry

end NUMINAMATH_GPT_find_a_l1177_117735


namespace NUMINAMATH_GPT_smallest_integer_among_three_l1177_117713

theorem smallest_integer_among_three 
  (x y z : ℕ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y)
  (hz_pos : 0 < z)
  (hxy : y - x ≤ 6)
  (hxz : z - x ≤ 6) 
  (hprod : x * y * z = 2808) : 
  x = 12 := 
sorry

end NUMINAMATH_GPT_smallest_integer_among_three_l1177_117713


namespace NUMINAMATH_GPT_painted_sphere_area_proportionality_l1177_117748

theorem painted_sphere_area_proportionality
  (r : ℝ)
  (R_inner R_outer : ℝ)
  (A_inner : ℝ)
  (h_r : r = 1)
  (h_R_inner : R_inner = 4)
  (h_R_outer : R_outer = 6)
  (h_A_inner : A_inner = 47) :
  ∃ A_outer : ℝ, A_outer = 105.75 :=
by
  have ratio := (R_outer / R_inner)^2
  have A_outer := A_inner * ratio
  use A_outer
  sorry

end NUMINAMATH_GPT_painted_sphere_area_proportionality_l1177_117748


namespace NUMINAMATH_GPT_sqrt_720_simplified_l1177_117767

theorem sqrt_720_simplified : Real.sqrt 720 = 6 * Real.sqrt 5 := sorry

end NUMINAMATH_GPT_sqrt_720_simplified_l1177_117767


namespace NUMINAMATH_GPT_todd_ingredients_l1177_117703

variables (B R N : ℕ) (P A : ℝ) (I : ℝ)

def todd_problem (B R N : ℕ) (P A I : ℝ) : Prop := 
  B = 100 ∧ 
  R = 110 ∧ 
  N = 200 ∧ 
  P = 0.75 ∧ 
  A = 65 ∧ 
  I = 25

theorem todd_ingredients :
  todd_problem 100 110 200 0.75 65 25 :=
by sorry

end NUMINAMATH_GPT_todd_ingredients_l1177_117703


namespace NUMINAMATH_GPT_positive_real_solution_unique_l1177_117770

theorem positive_real_solution_unique :
  (∃! x : ℝ, 0 < x ∧ x^12 + 5 * x^11 - 3 * x^10 + 2000 * x^9 - 1500 * x^8 = 0) :=
sorry

end NUMINAMATH_GPT_positive_real_solution_unique_l1177_117770


namespace NUMINAMATH_GPT_maximize_expr_at_neg_5_l1177_117763

-- Definition of the expression
def expr (x : ℝ) : ℝ := 1 - (x + 5) ^ 2

-- Prove that when x = -5, the expression has its maximum value
theorem maximize_expr_at_neg_5 : ∀ x : ℝ, expr x ≤ expr (-5) :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_maximize_expr_at_neg_5_l1177_117763


namespace NUMINAMATH_GPT_max_value_of_a_l1177_117781

theorem max_value_of_a :
  ∀ (a : ℚ),
  (∀ (m : ℚ), 1/3 < m ∧ m < a →
   (∀ (x : ℤ), 0 < x ∧ x ≤ 200 →
    ¬ (∃ (y : ℤ), y = m * x + 3 ∨ y = m * x + 1))) →
  a = 68/201 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_a_l1177_117781


namespace NUMINAMATH_GPT_count_multiples_of_four_between_100_and_350_l1177_117787

-- Define the problem conditions
def is_multiple_of_four (n : ℕ) : Prop := n % 4 = 0
def in_range (n : ℕ) : Prop := 100 < n ∧ n < 350

-- Problem statement
theorem count_multiples_of_four_between_100_and_350 : 
  ∃ (k : ℕ), k = 62 ∧ ∀ n : ℕ, is_multiple_of_four n ∧ in_range n ↔ (100 < n ∧ n < 350 ∧ is_multiple_of_four n)
:= sorry

end NUMINAMATH_GPT_count_multiples_of_four_between_100_and_350_l1177_117787


namespace NUMINAMATH_GPT_total_blocks_traveled_l1177_117712

-- Given conditions as definitions
def annie_walked_blocks : ℕ := 5
def annie_rode_blocks : ℕ := 7

-- The total blocks Annie traveled
theorem total_blocks_traveled : annie_walked_blocks + annie_rode_blocks + (annie_walked_blocks + annie_rode_blocks) = 24 := by
  sorry

end NUMINAMATH_GPT_total_blocks_traveled_l1177_117712


namespace NUMINAMATH_GPT_min_value_expr_l1177_117747

noncomputable def min_value (a b c : ℝ) := 4 * a^3 + 8 * b^3 + 18 * c^3 + 1 / (9 * a * b * c)

theorem min_value_expr (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  min_value a b c ≥ 8 / Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expr_l1177_117747


namespace NUMINAMATH_GPT_rectangle_area_constant_l1177_117766

theorem rectangle_area_constant (d : ℝ) (x : ℝ)
  (length width : ℝ)
  (h_length : length = 5 * x)
  (h_width : width = 4 * x)
  (h_diagonal : d = Real.sqrt (length ^ 2 + width ^ 2)) :
  (exists k : ℝ, k = 20 / 41 ∧ (length * width = k * d ^ 2)) :=
by
  use 20 / 41
  sorry

end NUMINAMATH_GPT_rectangle_area_constant_l1177_117766


namespace NUMINAMATH_GPT_right_triangle_area_l1177_117740

-- Define the lengths of the legs of the right triangle
def leg_length : ℝ := 1

-- State the theorem
theorem right_triangle_area (a b : ℝ) (h1 : a = leg_length) (h2 : b = leg_length) : 
  (1 / 2) * a * b = 1 / 2 :=
by
  rw [h1, h2]
  -- From the substitutions above, it simplifies to:
  sorry

end NUMINAMATH_GPT_right_triangle_area_l1177_117740


namespace NUMINAMATH_GPT_last_two_digits_7_pow_2018_l1177_117758

theorem last_two_digits_7_pow_2018 : 
  (7 ^ 2018) % 100 = 49 := 
sorry

end NUMINAMATH_GPT_last_two_digits_7_pow_2018_l1177_117758


namespace NUMINAMATH_GPT_solve_inequality_system_l1177_117795

theorem solve_inequality_system (x : ℝ) 
  (h1 : 3 * x - 1 > x + 1) 
  (h2 : (4 * x - 5) / 3 ≤ x) 
  : 1 < x ∧ x ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_system_l1177_117795


namespace NUMINAMATH_GPT_intersection_A_B_find_a_b_l1177_117785

noncomputable def A : Set ℝ := { x | x^2 - 5 * x + 6 > 0 }
noncomputable def B : Set ℝ := { x | Real.log (x + 1) / Real.log 2 < 2 }

theorem intersection_A_B :
  A ∩ B = { x | -1 < x ∧ x < 2 } :=
by
  -- Proof will be provided
  sorry

theorem find_a_b :
  ∃ a b : ℝ, (∀ x : ℝ, x^2 + a * x - b < 0 ↔ -1 < x ∧ x < 2) ∧ a = -1 ∧ b = 2 :=
by
  -- Proof will be provided
  sorry

end NUMINAMATH_GPT_intersection_A_B_find_a_b_l1177_117785


namespace NUMINAMATH_GPT_valid_values_l1177_117733

noncomputable def is_defined (x : ℝ) : Prop := 
  (x^2 - 4*x + 3 > 0) ∧ (5 - x^2 > 0)

theorem valid_values (x : ℝ) : 
  is_defined x ↔ (-Real.sqrt 5 < x ∧ x < 1) ∨ (3 < x ∧ x < Real.sqrt 5) := by
  sorry

end NUMINAMATH_GPT_valid_values_l1177_117733


namespace NUMINAMATH_GPT_sum_of_solutions_l1177_117779

theorem sum_of_solutions (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (∃ x₁ x₂ : ℝ, (3 * x₁ + 2) * (x₁ - 4) = 0 ∧ (3 * x₂ + 2) * (x₂ - 4) = 0 ∧
  x₁ ≠ 1 ∧ x₁ ≠ -1 ∧ x₂ ≠ 1 ∧ x₂ ≠ -1 ∧ x₁ + x₂ = 10 / 3) :=
sorry

end NUMINAMATH_GPT_sum_of_solutions_l1177_117779


namespace NUMINAMATH_GPT_Anne_Katherine_savings_l1177_117764

theorem Anne_Katherine_savings :
  ∃ A K : ℕ, (A - 150 = K / 3) ∧ (2 * K = 3 * A) ∧ (A + K = 750) := 
sorry

end NUMINAMATH_GPT_Anne_Katherine_savings_l1177_117764


namespace NUMINAMATH_GPT_find_rhombus_acute_angle_l1177_117754

-- Definitions and conditions
def rhombus_angle (V1 V2 : ℝ) (α : ℝ) : Prop :=
  V1 / V2 = 1 / (2 * Real.sqrt 5)
  
-- Theorem statement
theorem find_rhombus_acute_angle (V1 V2 a : ℝ) (α : ℝ) (h : rhombus_angle V1 V2 α) :
  α = Real.arccos (1 / 9) :=
sorry

end NUMINAMATH_GPT_find_rhombus_acute_angle_l1177_117754


namespace NUMINAMATH_GPT_find_f_1_0_plus_f_2_0_general_form_F_l1177_117741

variable {F : ℝ → ℝ → ℝ}

-- Conditions
axiom cond1 : ∀ a, F a a = a
axiom cond2 : ∀ (k a b : ℝ), F (k * a) (k * b) = k * F a b
axiom cond3 : ∀ (a1 a2 b1 b2 : ℝ), F (a1 + a2) (b1 + b2) = F a1 b1 + F a2 b2
axiom cond4 : ∀ (a b : ℝ), F a b = F b ((a + b) / 2)

-- Proof problem
theorem find_f_1_0_plus_f_2_0 : F 1 0 + F 2 0 = 0 :=
sorry

theorem general_form_F : ∀ (x y : ℝ), F x y = y :=
sorry

end NUMINAMATH_GPT_find_f_1_0_plus_f_2_0_general_form_F_l1177_117741


namespace NUMINAMATH_GPT_not_divisible_by_121_l1177_117743

theorem not_divisible_by_121 (n : ℤ) : ¬ ∃ t : ℤ, (n^2 + 3*n + 5) = 121 * t ∧ (n^2 - 3*n + 5) = 121 * t := sorry

end NUMINAMATH_GPT_not_divisible_by_121_l1177_117743


namespace NUMINAMATH_GPT_triangle_coordinates_sum_l1177_117738

noncomputable def coordinates_of_triangle_A (p q : ℚ) : Prop :=
  let B := (12, 19)
  let C := (23, 20)
  let area := ((B.1 * C.2 + C.1 * q + p * B.2) - (B.2 * C.1 + C.2 * p + q * B.1)) / 2 
  let M := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let median_slope := (q - M.2) / (p - M.1)
  area = 60 ∧ median_slope = 3 

theorem triangle_coordinates_sum (p q : ℚ) 
(h : coordinates_of_triangle_A p q) : p + q = 52 := 
sorry

end NUMINAMATH_GPT_triangle_coordinates_sum_l1177_117738


namespace NUMINAMATH_GPT_ordered_pair_a_82_a_28_l1177_117757

-- Definitions for the conditions
def a (i j : ℕ) : ℕ :=
  if i % 2 = 1 then
    if j = 1 then i * i else i * i - (j - 1)
  else
    if j = 1 then (i-1) * i + 1 else i * i - (j - 1)

theorem ordered_pair_a_82_a_28 : (a 8 2, a 2 8) = (51, 63) := by
  sorry

end NUMINAMATH_GPT_ordered_pair_a_82_a_28_l1177_117757


namespace NUMINAMATH_GPT_flower_combinations_count_l1177_117798

/-- Prove that there are exactly 3 combinations of tulips and sunflowers that sum up to $60,
    where tulips cost $4 each and sunflowers cost $3 each, and the number of sunflowers is greater than the number 
    of tulips. -/
theorem flower_combinations_count :
  ∃ n : ℕ, n = 3 ∧
    ∃ t s : ℕ, 4 * t + 3 * s = 60 ∧ s > t :=
by {
  sorry
}

end NUMINAMATH_GPT_flower_combinations_count_l1177_117798


namespace NUMINAMATH_GPT_max_non_multiples_of_3_l1177_117761

theorem max_non_multiples_of_3 (a b c d e f : ℕ) (h1 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) (h2 : a * b * c * d * e * f % 3 = 0) : 
  ¬ ∃ (count : ℕ), count > 5 ∧ (∀ x ∈ [a, b, c, d, e, f], x % 3 ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_max_non_multiples_of_3_l1177_117761


namespace NUMINAMATH_GPT_sqrt_23_range_l1177_117769

theorem sqrt_23_range : 4.5 < Real.sqrt 23 ∧ Real.sqrt 23 < 5 := by
  sorry

end NUMINAMATH_GPT_sqrt_23_range_l1177_117769


namespace NUMINAMATH_GPT_gcd_12012_21021_l1177_117750

-- Definitions
def factors_12012 : List ℕ := [2, 2, 3, 7, 11, 13] -- Factors of 12,012
def factors_21021 : List ℕ := [3, 7, 7, 11, 13] -- Factors of 21,021

def common_factors := [3, 7, 11, 13] -- Common factors between 12,012 and 21,021

def gcd (ls : List ℕ) : ℕ :=
ls.foldr Nat.gcd 0 -- Function to calculate gcd of list of numbers

-- Main statement
theorem gcd_12012_21021 : gcd common_factors = 1001 := by
  -- Proof is not required, so we use sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_gcd_12012_21021_l1177_117750


namespace NUMINAMATH_GPT_boxes_needed_l1177_117719

-- Let's define the conditions
def total_paper_clips : ℕ := 81
def paper_clips_per_box : ℕ := 9

-- Define the target of our proof, which is that the number of boxes needed is 9
theorem boxes_needed : total_paper_clips / paper_clips_per_box = 9 := by
  sorry

end NUMINAMATH_GPT_boxes_needed_l1177_117719


namespace NUMINAMATH_GPT_least_three_digit_multiple_l1177_117775

def LCM (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

theorem least_three_digit_multiple (n : ℕ) :
  (n >= 100) ∧ (n < 1000) ∧ (n % 36 = 0) ∧ (∀ m, (m >= 100) ∧ (m < 1000) ∧ (m % 36 = 0) → n <= m) ↔ n = 108 :=
sorry

end NUMINAMATH_GPT_least_three_digit_multiple_l1177_117775


namespace NUMINAMATH_GPT_total_games_played_l1177_117772

-- Define the conditions as parameters
def ratio_games_won_lost (W L : ℕ) : Prop := W / 2 = L / 3

-- Let's state the problem formally in Lean
theorem total_games_played (W L : ℕ) (h1 : ratio_games_won_lost W L) (h2 : W = 18) : W + L = 30 :=
by 
  sorry  -- The proof will be filled in


end NUMINAMATH_GPT_total_games_played_l1177_117772


namespace NUMINAMATH_GPT_one_thirds_in_eight_halves_l1177_117728

theorem one_thirds_in_eight_halves : (8 / 2) / (1 / 3) = 12 := by
  sorry

end NUMINAMATH_GPT_one_thirds_in_eight_halves_l1177_117728


namespace NUMINAMATH_GPT_Masha_thought_of_numbers_l1177_117788

theorem Masha_thought_of_numbers : ∃ a b : ℕ, a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a + b = 28) ∧ (a % 2 = 0 ∨ b % 2 = 0) ∧ (a = 12 ∧ b = 16 ∨ a = 16 ∧ b = 12) :=
by
  sorry

end NUMINAMATH_GPT_Masha_thought_of_numbers_l1177_117788


namespace NUMINAMATH_GPT_three_angles_difference_is_2pi_over_3_l1177_117710

theorem three_angles_difference_is_2pi_over_3 (α β γ : ℝ) 
    (h1 : 0 ≤ α) (h2 : α ≤ β) (h3 : β < γ) (h4 : γ ≤ 2 * Real.pi)
    (h5 : Real.cos α + Real.cos β + Real.cos γ = 0)
    (h6 : Real.sin α + Real.sin β + Real.sin γ = 0) :
    β - α = 2 * Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_three_angles_difference_is_2pi_over_3_l1177_117710


namespace NUMINAMATH_GPT_tomatoes_left_after_yesterday_correct_l1177_117734

def farmer_initial_tomatoes := 160
def tomatoes_picked_yesterday := 56
def tomatoes_left_after_yesterday : ℕ := farmer_initial_tomatoes - tomatoes_picked_yesterday

theorem tomatoes_left_after_yesterday_correct :
  tomatoes_left_after_yesterday = 104 :=
by
  unfold tomatoes_left_after_yesterday
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_tomatoes_left_after_yesterday_correct_l1177_117734


namespace NUMINAMATH_GPT_projection_of_A_onto_Oxz_is_B_l1177_117706

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def projection_onto_Oxz (A : Point3D) : Point3D :=
  { x := A.x, y := 0, z := A.z }

theorem projection_of_A_onto_Oxz_is_B :
  let A := Point3D.mk 2 3 6
  let B := Point3D.mk 2 0 6
  projection_onto_Oxz A = B :=
by
  let A := Point3D.mk 2 3 6
  let B := Point3D.mk 2 0 6
  have h : projection_onto_Oxz A = B := rfl
  exact h

end NUMINAMATH_GPT_projection_of_A_onto_Oxz_is_B_l1177_117706


namespace NUMINAMATH_GPT_solve_system_a_solve_system_b_l1177_117732

-- For problem (a):
theorem solve_system_a (x y : ℝ) :
  (x + y + x * y = 5) ∧ (x * y * (x + y) = 6) → 
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) := 
by
  sorry

-- For problem (b):
theorem solve_system_b (x y : ℝ) :
  (x^3 + y^3 + 2 * x * y = 4) ∧ (x^2 - x * y + y^2 = 1) → 
  (x = 1 ∧ y = 1) := 
by
  sorry

end NUMINAMATH_GPT_solve_system_a_solve_system_b_l1177_117732


namespace NUMINAMATH_GPT_student_average_grade_l1177_117782

noncomputable def average_grade_two_years : ℝ :=
  let year1_courses := 6
  let year1_average_grade := 100
  let year1_total_points := year1_courses * year1_average_grade

  let year2_courses := 5
  let year2_average_grade := 40
  let year2_total_points := year2_courses * year2_average_grade

  let total_courses := year1_courses + year2_courses
  let total_points := year1_total_points + year2_total_points

  total_points / total_courses

theorem student_average_grade : average_grade_two_years = 72.7 :=
by
  sorry

end NUMINAMATH_GPT_student_average_grade_l1177_117782


namespace NUMINAMATH_GPT_area_increase_percentage_area_percentage_increase_length_to_width_ratio_l1177_117739

open Real

-- Part (a)
theorem area_increase_percentage (a b : ℝ) :
  (1.12 * a) * (1.15 * b) = 1.288 * (a * b) :=
  sorry

theorem area_percentage_increase (a b : ℝ) :
  ((1.12 * a) * (1.15 * b)) / (a * b) = 1.288 :=
  sorry

-- Part (b)
theorem length_to_width_ratio (a b : ℝ) (h : 2 * ((1.12 * a) + (1.15 * b)) = 1.13 * 2 * (a + b)) :
  a = 2 * b :=
  sorry

end NUMINAMATH_GPT_area_increase_percentage_area_percentage_increase_length_to_width_ratio_l1177_117739


namespace NUMINAMATH_GPT_find_side_a_l1177_117727

theorem find_side_a (a b c : ℝ) (B : ℝ)
  (h1 : c = Real.sqrt 2)
  (h2 : b = Real.sqrt 6)
  (h3 : B = 120) :
  a = Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_find_side_a_l1177_117727


namespace NUMINAMATH_GPT_triangle_side_length_l1177_117715

theorem triangle_side_length (A : ℝ) (b : ℝ) (S : ℝ) (hA : A = 120) (hb : b = 4) (hS: S = 2 * Real.sqrt 3) : 
  ∃ c : ℝ, c = 2 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_side_length_l1177_117715


namespace NUMINAMATH_GPT_no_solution_for_given_m_l1177_117777

theorem no_solution_for_given_m (x m : ℝ) (h1 : x ≠ 5) (h2 : x ≠ 8) :
  (∀ y : ℝ, (y - 2) / (y - 5) = (y - m) / (y - 8) → false) ↔ m = 5 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_for_given_m_l1177_117777


namespace NUMINAMATH_GPT_junior_score_l1177_117753

theorem junior_score (total_students : ℕ) (juniors_percentage : ℝ) (seniors_percentage : ℝ)
  (class_average : ℝ) (senior_average : ℝ) (juniors_same_score : Prop) 
  (h1 : juniors_percentage = 0.2) (h2 : seniors_percentage = 0.8)
  (h3 : class_average = 85) (h4 : senior_average = 84) : 
  ∃ junior_score : ℝ, juniors_same_score → junior_score = 89 :=
by
  sorry

end NUMINAMATH_GPT_junior_score_l1177_117753


namespace NUMINAMATH_GPT_inequality_proof_l1177_117756

theorem inequality_proof
  (x y z : ℝ) (hxpos : 0 < x) (hypos : 0 < y) (hzpos : 0 < z)
  (hineq : x * y + y * z + z * x ≤ 1) :
  (x + 1 / x) * (y + 1 / y) * (z + 1 / z) ≥ 8 * (x + y) * (y + z) * (z + x) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1177_117756


namespace NUMINAMATH_GPT_infinite_product_value_l1177_117778

noncomputable def infinite_product : ℝ :=
  ∏' n : ℕ, 9^(1/(3^n))

theorem infinite_product_value : infinite_product = 27 := 
  by sorry

end NUMINAMATH_GPT_infinite_product_value_l1177_117778


namespace NUMINAMATH_GPT_min_value_at_neg7_l1177_117736

noncomputable def f (x : ℝ) : ℝ := x^2 + 14 * x + 24

theorem min_value_at_neg7 : ∀ x : ℝ, f (-7) ≤ f x :=
by
  sorry

end NUMINAMATH_GPT_min_value_at_neg7_l1177_117736


namespace NUMINAMATH_GPT_max_voters_is_five_l1177_117718

noncomputable def max_voters_after_T (x : ℕ) : ℕ :=
if h : 0 ≤ (x - 11) then x - 11 else 0

theorem max_voters_is_five (x : ℕ) (hx : 0 ≤ x ∧ x ≤ 10) :
  max_voters_after_T x = 5 :=
by
  sorry

end NUMINAMATH_GPT_max_voters_is_five_l1177_117718


namespace NUMINAMATH_GPT_pencil_sharpening_and_breaking_l1177_117722

/-- Isha's pencil initially has a length of 31 inches. After sharpening, it has a length of 14 inches.
Prove that:
1. The pencil was shortened by 17 inches.
2. Each half of the pencil, after being broken in half, is 7 inches long. -/
theorem pencil_sharpening_and_breaking 
  (initial_length : ℕ) 
  (length_after_sharpening : ℕ) 
  (sharpened_length : ℕ) 
  (half_length : ℕ) 
  (h1 : initial_length = 31) 
  (h2 : length_after_sharpening = 14) 
  (h3 : sharpened_length = initial_length - length_after_sharpening) 
  (h4 : half_length = length_after_sharpening / 2) : 
  sharpened_length = 17 ∧ half_length = 7 := 
by {
  sorry
}

end NUMINAMATH_GPT_pencil_sharpening_and_breaking_l1177_117722


namespace NUMINAMATH_GPT_triangle_perimeter_l1177_117789

theorem triangle_perimeter (a b c : ℕ) (ha : a = 7) (hb : b = 10) (hc : c = 15) :
  a + b + c = 32 :=
by
  -- Given the lengths of the sides
  have H1 : a = 7 := ha
  have H2 : b = 10 := hb
  have H3 : c = 15 := hc
  
  -- Therefore, we need to prove the sum
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l1177_117789


namespace NUMINAMATH_GPT_percentage_more_likely_to_lose_both_l1177_117771

def first_lawsuit_win_probability : ℝ := 0.30
def first_lawsuit_lose_probability : ℝ := 0.70
def second_lawsuit_win_probability : ℝ := 0.50
def second_lawsuit_lose_probability : ℝ := 0.50

theorem percentage_more_likely_to_lose_both :
  (second_lawsuit_lose_probability * first_lawsuit_lose_probability - second_lawsuit_win_probability * first_lawsuit_win_probability) / (second_lawsuit_win_probability * first_lawsuit_win_probability) * 100 = 133.33 :=
by
  sorry

end NUMINAMATH_GPT_percentage_more_likely_to_lose_both_l1177_117771


namespace NUMINAMATH_GPT_tony_quilt_square_side_length_l1177_117790

theorem tony_quilt_square_side_length (length width : ℝ) (h_length : length = 6) (h_width : width = 24) : 
  ∃ s, s * s = length * width ∧ s = 12 :=
by
  sorry

end NUMINAMATH_GPT_tony_quilt_square_side_length_l1177_117790


namespace NUMINAMATH_GPT_used_computer_lifespan_l1177_117759

-- Problem statement
theorem used_computer_lifespan (cost_new : ℕ) (lifespan_new : ℕ) (cost_used : ℕ) (num_used : ℕ) (savings : ℕ) :
  cost_new = 600 →
  lifespan_new = 6 →
  cost_used = 200 →
  num_used = 2 →
  savings = 200 →
  ((cost_new - savings = num_used * cost_used) → (2 * (lifespan_new / 2) = 6) → lifespan_new / 2 = 3)
:= by
  intros
  sorry

end NUMINAMATH_GPT_used_computer_lifespan_l1177_117759


namespace NUMINAMATH_GPT_correct_arrangements_l1177_117709

open Finset Nat

-- Definitions for combinations and powers
def comb (n k : ℕ) : ℕ := choose n k

-- The number of computer rooms
def num_computer_rooms : ℕ := 6

-- The number of arrangements to open at least 2 out of 6 computer rooms
def arrangement_count1 : ℕ := 2^num_computer_rooms - (comb num_computer_rooms 0 + comb num_computer_rooms 1)

-- Another calculation for the number of arrangements to open at least 2 out of 6 computer rooms
def arrangement_count2 : ℕ := comb num_computer_rooms 2 + comb num_computer_rooms 3 + comb num_computer_rooms 4 + 
                               comb num_computer_rooms 5 + comb num_computer_rooms 6

theorem correct_arrangements :
  arrangement_count1 = arrangement_count2 := 
  sorry

end NUMINAMATH_GPT_correct_arrangements_l1177_117709


namespace NUMINAMATH_GPT_sum_of_series_equals_negative_682_l1177_117714

noncomputable def geometric_sum : ℤ :=
  let a := 2
  let r := -2
  let n := 10
  (a * (r ^ n - 1)) / (r - 1)

theorem sum_of_series_equals_negative_682 : geometric_sum = -682 := 
by sorry

end NUMINAMATH_GPT_sum_of_series_equals_negative_682_l1177_117714


namespace NUMINAMATH_GPT_car_speed_l1177_117749

/-- Given a car covers a distance of 624 km in 2 3/5 hours,
    prove that the speed of the car is 240 km/h. -/
theorem car_speed (distance : ℝ) (time : ℝ)
  (h_distance : distance = 624)
  (h_time : time = 13 / 5) :
  (distance / time) = 240 :=
by
  sorry

end NUMINAMATH_GPT_car_speed_l1177_117749


namespace NUMINAMATH_GPT_combined_avg_of_remaining_two_subjects_l1177_117725

noncomputable def avg (scores : List ℝ) : ℝ :=
  scores.foldl (· + ·) 0 / scores.length

theorem combined_avg_of_remaining_two_subjects 
  (S1_avg S2_part_avg all_avg : ℝ)
  (S1_count S2_part_count S2_total_count : ℕ)
  (h1 : S1_avg = 85) 
  (h2 : S2_part_avg = 78) 
  (h3 : all_avg = 80) 
  (h4 : S1_count = 3)
  (h5 : S2_part_count = 5)
  (h6 : S2_total_count = 7) :
  avg [all_avg * (S1_count + S2_total_count) 
       - S1_count * S1_avg 
       - S2_part_count * S2_part_avg] / (S2_total_count - S2_part_count)
  = 77.5 := by
  sorry

end NUMINAMATH_GPT_combined_avg_of_remaining_two_subjects_l1177_117725


namespace NUMINAMATH_GPT_no_integer_soln_x_y_l1177_117737

theorem no_integer_soln_x_y (x y : ℤ) : x^2 + 5 ≠ y^3 := 
sorry

end NUMINAMATH_GPT_no_integer_soln_x_y_l1177_117737


namespace NUMINAMATH_GPT_ned_shirts_problem_l1177_117780

theorem ned_shirts_problem
  (long_sleeve_shirts : ℕ)
  (total_shirts_washed : ℕ)
  (total_shirts_had : ℕ)
  (h1 : long_sleeve_shirts = 21)
  (h2 : total_shirts_washed = 29)
  (h3 : total_shirts_had = total_shirts_washed + 1) :
  ∃ short_sleeve_shirts : ℕ, short_sleeve_shirts = total_shirts_had - total_shirts_washed - 1 :=
by
  sorry

end NUMINAMATH_GPT_ned_shirts_problem_l1177_117780


namespace NUMINAMATH_GPT_one_sixth_time_l1177_117774

-- Conditions
def total_kids : ℕ := 40
def kids_less_than_6_minutes : ℕ := total_kids * 10 / 100
def kids_less_than_8_minutes : ℕ := 3 * kids_less_than_6_minutes
def remaining_kids : ℕ := total_kids - (kids_less_than_6_minutes + kids_less_than_8_minutes)
def kids_more_than_certain_minutes : ℕ := 4
def one_sixth_remaining_kids : ℕ := remaining_kids / 6

-- Statement to prove the equivalence
theorem one_sixth_time :
  one_sixth_remaining_kids = kids_more_than_certain_minutes := 
sorry

end NUMINAMATH_GPT_one_sixth_time_l1177_117774


namespace NUMINAMATH_GPT_find_number_l1177_117783

/--
A number is added to 5, then multiplied by 5, then subtracted by 5, and then divided by 5. 
The result is still 5. Prove that the number is 1.
-/
theorem find_number (x : ℝ) (h : ((5 * (x + 5) - 5) / 5 = 5)) : x = 1 := 
  sorry

end NUMINAMATH_GPT_find_number_l1177_117783


namespace NUMINAMATH_GPT_cylinder_lateral_surface_area_l1177_117762
noncomputable def lateralSurfaceArea (S : ℝ) : ℝ :=
  let l := Real.sqrt S
  let d := l
  let r := d / 2
  let h := l
  2 * Real.pi * r * h

theorem cylinder_lateral_surface_area (S : ℝ) (hS : S ≥ 0) : 
  lateralSurfaceArea S = Real.pi * S := by
  sorry

end NUMINAMATH_GPT_cylinder_lateral_surface_area_l1177_117762


namespace NUMINAMATH_GPT_root_in_interval_l1177_117773

noncomputable def f (a b x : ℝ) : ℝ := a^x + x - b

theorem root_in_interval (a b : ℝ) (ha : a > 1) (hb : 0 < b ∧ b < 1) : 
  ∃ x : ℝ, -1 < x ∧ x < 0 ∧ f a b x = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_root_in_interval_l1177_117773


namespace NUMINAMATH_GPT_members_not_playing_either_l1177_117794

variable (total_members badminton_players tennis_players both_players : ℕ)

theorem members_not_playing_either (h1 : total_members = 40)
                                   (h2 : badminton_players = 20)
                                   (h3 : tennis_players = 18)
                                   (h4 : both_players = 3) :
  total_members - (badminton_players + tennis_players - both_players) = 5 := by
  sorry

end NUMINAMATH_GPT_members_not_playing_either_l1177_117794


namespace NUMINAMATH_GPT_contrapositive_proposition_l1177_117755

theorem contrapositive_proposition (a b : ℝ) :
  (¬ ((a - b) * (a + b) = 0) → ¬ (a - b = 0)) :=
sorry

end NUMINAMATH_GPT_contrapositive_proposition_l1177_117755


namespace NUMINAMATH_GPT_polynomial_expansion_l1177_117702

theorem polynomial_expansion :
  ∃ A B C D : ℝ, (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) 
  ∧ (A + B + C + D = 36) :=
by {
  sorry
}

end NUMINAMATH_GPT_polynomial_expansion_l1177_117702


namespace NUMINAMATH_GPT_star_comm_l1177_117791

section SymmetricOperation

variable {S : Type*} 
variable (star : S → S → S)
variable (symm : ∀ a b : S, star a b = star (star b a) (star b a)) 

theorem star_comm (a b : S) : star a b = star b a := 
by 
  sorry

end SymmetricOperation

end NUMINAMATH_GPT_star_comm_l1177_117791


namespace NUMINAMATH_GPT_kyle_delivers_daily_papers_l1177_117760

theorem kyle_delivers_daily_papers (x : ℕ) (h : 6 * x + (x - 10) + 30 = 720) : x = 100 :=
by
  sorry

end NUMINAMATH_GPT_kyle_delivers_daily_papers_l1177_117760


namespace NUMINAMATH_GPT_find_a_for_exactly_two_solutions_l1177_117705

theorem find_a_for_exactly_two_solutions :
  ∃ a : ℝ, (∀ x : ℝ, (|x + a| = 1/x) ↔ (a = -2) ∧ (x ≠ 0)) ∧ ∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (|x1 + a| = 1/x1) ∧ (|x2 + a| = 1/x2) :=
sorry

end NUMINAMATH_GPT_find_a_for_exactly_two_solutions_l1177_117705


namespace NUMINAMATH_GPT_minimum_squares_required_l1177_117704

theorem minimum_squares_required (length : ℚ) (width : ℚ) (M N : ℕ) :
  (length = 121 / 2) → (width = 143 / 3) → (M / N = 33 / 26) → (M * N = 858) :=
by
  intros hL hW hMN
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_minimum_squares_required_l1177_117704


namespace NUMINAMATH_GPT_max_new_cars_l1177_117717

theorem max_new_cars (b₁ : ℕ) (r : ℝ) (M : ℕ) (L : ℕ) (x : ℝ) (h₀ : b₁ = 30) (h₁ : r = 0.94) (h₂ : M = 600000) (h₃ : L = 300000) :
  x ≤ (3.6 * 10^4) :=
sorry

end NUMINAMATH_GPT_max_new_cars_l1177_117717


namespace NUMINAMATH_GPT_longer_side_of_rectangle_l1177_117711

theorem longer_side_of_rectangle
  (r : ℝ) (A_rect A_circle L S : ℝ) (h1 : r = 6)
  (h2 : A_circle = π * r^2)
  (h3 : A_rect = 3 * A_circle)
  (h4 : S = 2 * r)
  (h5 : A_rect = S * L) : L = 9 * π :=
by
  sorry

end NUMINAMATH_GPT_longer_side_of_rectangle_l1177_117711


namespace NUMINAMATH_GPT_find_a_range_empty_solution_set_l1177_117701

theorem find_a_range_empty_solution_set :
  ∀ a : ℝ, (∀ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0 → false) ↔ (-2 ≤ a ∧ a < 6 / 5) :=
by sorry

end NUMINAMATH_GPT_find_a_range_empty_solution_set_l1177_117701


namespace NUMINAMATH_GPT_solve_for_y_l1177_117751

theorem solve_for_y (y : ℝ) (h : (8 * y^2 + 50 * y + 3) / (4 * y + 21) = 2 * y + 1) : y = 4.5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_solve_for_y_l1177_117751


namespace NUMINAMATH_GPT_distinct_sequences_l1177_117793

theorem distinct_sequences (N : ℕ) (α : ℝ) 
  (cond1 : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ N → 1 ≤ j ∧ j ≤ N → i ≠ j → 
    Int.floor (i * α) ≠ Int.floor (j * α)) 
  (cond2 : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ N → 1 ≤ j ∧ j ≤ N → i ≠ j → 
    Int.floor (i / α) ≠ Int.floor (j / α)) : 
  (↑(N - 1) / ↑N : ℝ) ≤ α ∧ α ≤ (↑N / ↑(N - 1) : ℝ) := 
sorry

end NUMINAMATH_GPT_distinct_sequences_l1177_117793


namespace NUMINAMATH_GPT_third_divisor_l1177_117746

theorem third_divisor (x : ℕ) (h12 : 12 ∣ (x + 3)) (h15 : 15 ∣ (x + 3)) (h40 : 40 ∣ (x + 3)) :
  ∃ d : ℕ, d ≠ 12 ∧ d ≠ 15 ∧ d ≠ 40 ∧ d ∣ (x + 3) ∧ d = 2 :=
by
  sorry

end NUMINAMATH_GPT_third_divisor_l1177_117746
