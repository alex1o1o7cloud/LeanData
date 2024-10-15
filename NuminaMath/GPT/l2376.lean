import Mathlib

namespace NUMINAMATH_GPT_region_area_l2376_237617

theorem region_area : 
  (∃ (x y : ℝ), abs (4 * x - 16) + abs (3 * y + 9) ≤ 6) →
  (∀ (A : ℝ), (∀ x y : ℝ, abs (4 * x - 16) + abs (3 * y + 9) ≤ 6 → 0 ≤ A ∧ A = 6)) :=
by
  intro h exist_condtion
  sorry

end NUMINAMATH_GPT_region_area_l2376_237617


namespace NUMINAMATH_GPT_evaluate_expression_l2376_237625

theorem evaluate_expression : (16 ^ 24) / (64 ^ 8) = 16 ^ 12 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l2376_237625


namespace NUMINAMATH_GPT_largest_of_A_B_C_l2376_237615

noncomputable def A : ℝ := (2010 / 2009) + (2010 / 2011)
noncomputable def B : ℝ := (2010 / 2011) + (2012 / 2011)
noncomputable def C : ℝ := (2011 / 2010) + (2011 / 2012)

theorem largest_of_A_B_C : B > A ∧ B > C := by
  sorry

end NUMINAMATH_GPT_largest_of_A_B_C_l2376_237615


namespace NUMINAMATH_GPT_debby_jogged_total_l2376_237683

theorem debby_jogged_total :
  let monday_distance := 2
  let tuesday_distance := 5
  let wednesday_distance := 9
  monday_distance + tuesday_distance + wednesday_distance = 16 :=
by
  sorry

end NUMINAMATH_GPT_debby_jogged_total_l2376_237683


namespace NUMINAMATH_GPT_quadratic_trinomial_negative_value_l2376_237651

theorem quadratic_trinomial_negative_value
  (a b c : ℝ)
  (h1 : b^2 ≥ 4 * c)
  (h2 : 1 ≥ 4 * a * c)
  (h3 : b^2 ≥ 4 * a) :
  ∃ x : ℝ, a * x^2 + b * x + c < 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_trinomial_negative_value_l2376_237651


namespace NUMINAMATH_GPT_find_radius_l2376_237631

theorem find_radius
  (r_1 r_2 r_3 : ℝ)
  (h_cone : r_2 = 2 * r_1 ∧ r_3 = 3 * r_1 ∧ r_1 + r_2 + r_3 = 18) :
  r_1 = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_radius_l2376_237631


namespace NUMINAMATH_GPT_constant_condition_for_quadrant_I_solution_l2376_237614

-- Define the given conditions
def equations (c : ℚ) (x y : ℚ) : Prop :=
  (x - 2 * y = 5) ∧ (c * x + 3 * y = 2)

-- Define the condition for the solution to be in Quadrant I
def isQuadrantI (x y : ℚ) : Prop :=
  (x > 0) ∧ (y > 0)

-- The theorem to be proved
theorem constant_condition_for_quadrant_I_solution (c : ℚ) :
  (∃ x y : ℚ, equations c x y ∧ isQuadrantI x y) ↔ (-3/2 < c ∧ c < 2/5) :=
by
  sorry

end NUMINAMATH_GPT_constant_condition_for_quadrant_I_solution_l2376_237614


namespace NUMINAMATH_GPT_paint_fraction_used_l2376_237647

theorem paint_fraction_used (initial_paint: ℕ) (first_week_fraction: ℚ) (total_paint_used: ℕ) (remaining_paint_after_first_week: ℕ) :
  initial_paint = 360 →
  first_week_fraction = 1/3 →
  total_paint_used = 168 →
  remaining_paint_after_first_week = initial_paint - initial_paint * first_week_fraction →
  (total_paint_used - initial_paint * first_week_fraction) / remaining_paint_after_first_week = 1/5 := 
by
  sorry

end NUMINAMATH_GPT_paint_fraction_used_l2376_237647


namespace NUMINAMATH_GPT_geometric_sequence_sum_l2376_237681

-- Definition of the sum of the first n terms of a geometric sequence
variable (S : ℕ → ℝ)

-- Conditions given in the problem
def S_n_given (n : ℕ) : Prop := S n = 36
def S_2n_given (n : ℕ) : Prop := S (2 * n) = 42

-- Theorem to prove
theorem geometric_sequence_sum (n : ℕ) (S : ℕ → ℝ) 
    (h1 : S n = 36) (h2 : S (2 * n) = 42) : S (3 * n) = 48 := sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l2376_237681


namespace NUMINAMATH_GPT_power_of_two_divides_sub_one_l2376_237612

theorem power_of_two_divides_sub_one (k : ℕ) (h_odd : k % 2 = 1) : ∀ n ≥ 1, 2^(n+2) ∣ k^(2^n) - 1 :=
by
  sorry

end NUMINAMATH_GPT_power_of_two_divides_sub_one_l2376_237612


namespace NUMINAMATH_GPT_triangle_side_lengths_l2376_237671

-- Define the problem
variables {r: ℝ} (h_a h_b h_c a b c : ℝ)
variable (sum_of_heights : h_a + h_b + h_c = 13)
variable (r_value : r = 4 / 3)
variable (height_relation : 1/h_a + 1/h_b + 1/h_c = 3/4)

-- Define the theorem to be proven
theorem triangle_side_lengths (h_a h_b h_c : ℝ)
  (sum_of_heights : h_a + h_b + h_c = 13) 
  (r_value : r = 4 / 3)
  (height_relation : 1/h_a + 1/h_b + 1/h_c = 3/4) :
  (a, b, c) = (32 / Real.sqrt 15, 24 / Real.sqrt 15, 16 / Real.sqrt 15) := 
sorry

end NUMINAMATH_GPT_triangle_side_lengths_l2376_237671


namespace NUMINAMATH_GPT_find_p_q_l2376_237675

theorem find_p_q 
  (p q: ℚ)
  (a : ℚ × ℚ × ℚ × ℚ := (4, p, -2, 1))
  (b : ℚ × ℚ × ℚ × ℚ := (3, 2, q, -1))
  (orthogonal : (4 * 3 + p * 2 + (-2) * q + 1 * (-1) = 0))
  (equal_magnitudes : (4^2 + p^2 + (-2)^2 + 1^2 = 3^2 + 2^2 + q^2 + (-1)^2))
  : p = -93/44 ∧ q = 149/44 := 
  by 
    sorry

end NUMINAMATH_GPT_find_p_q_l2376_237675


namespace NUMINAMATH_GPT_number_153_satisfies_l2376_237650

noncomputable def sumOfCubes (n : ℕ) : ℕ :=
  (n % 10)^3 + ((n / 10) % 10)^3 + ((n / 100) % 10)^3

theorem number_153_satisfies :
  (sumOfCubes 153) = 153 ∧ 
  (153 % 10 ≠ 0) ∧ ((153 / 10) % 10 ≠ 0) ∧ ((153 / 100) % 10 ≠ 0) ∧ 
  153 ≠ 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_153_satisfies_l2376_237650


namespace NUMINAMATH_GPT_max_x_lcm_max_x_lcm_value_l2376_237678

theorem max_x_lcm (x : ℕ) (h1 : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 :=
  sorry

theorem max_x_lcm_value (x : ℕ) (h1 : Nat.lcm (Nat.lcm x 15) 21 = 105) : x = 105 :=
  sorry

end NUMINAMATH_GPT_max_x_lcm_max_x_lcm_value_l2376_237678


namespace NUMINAMATH_GPT_math_problem_l2376_237670

noncomputable def proof_problem (a b c d : ℝ) : Prop :=
  36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧
  4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48

theorem math_problem (a b c d : ℝ)
  (h1 : a + b + c + d = 6)
  (h2 : a^2 + b^2 + c^2 + d^2 = 12) :
  proof_problem a b c d :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l2376_237670


namespace NUMINAMATH_GPT_geometric_sequence_sum_four_l2376_237642

theorem geometric_sequence_sum_four (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a (n + 1) = a n * q)
  (h2 : q ≠ 1)
  (h3 : -3 * a 0 = -2 * a 1 - a 2)
  (h4 : a 0 = 1) : 
  S 4 = -20 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_four_l2376_237642


namespace NUMINAMATH_GPT_problem_l2376_237620

theorem problem (n : ℕ) (h : n ∣ (2^n - 2)) : (2^n - 1) ∣ (2^(2^n - 1) - 2) :=
by
  sorry

end NUMINAMATH_GPT_problem_l2376_237620


namespace NUMINAMATH_GPT_combination_sum_l2376_237674

theorem combination_sum :
  (Nat.choose 7 4) + (Nat.choose 7 3) = 70 := by
  sorry

end NUMINAMATH_GPT_combination_sum_l2376_237674


namespace NUMINAMATH_GPT_ratio_of_radii_l2376_237652

theorem ratio_of_radii (r R : ℝ) (k : ℝ) (h1 : R > r) (h2 : π * R^2 - π * r^2 = k * π * r^2) :
  R / r = Real.sqrt (k + 1) :=
sorry

end NUMINAMATH_GPT_ratio_of_radii_l2376_237652


namespace NUMINAMATH_GPT_product_squared_inequality_l2376_237609

theorem product_squared_inequality (n : ℕ) (a : Fin n → ℝ) (h : (Finset.univ.prod (λ i => a i)) = 1) :
    (Finset.univ.prod (λ i => (1 + (a i)^2))) ≥ 2^n := 
sorry

end NUMINAMATH_GPT_product_squared_inequality_l2376_237609


namespace NUMINAMATH_GPT_solve_linear_eq_l2376_237697

theorem solve_linear_eq : (∃ x : ℝ, 2 * x - 1 = 0) ↔ (∃ x : ℝ, x = 1/2) :=
by
  sorry

end NUMINAMATH_GPT_solve_linear_eq_l2376_237697


namespace NUMINAMATH_GPT_parabola_intersection_sum_zero_l2376_237658

theorem parabola_intersection_sum_zero
  (x_1 x_2 x_3 x_4 y_1 y_2 y_3 y_4 : ℝ)
  (h1 : ∀ x, ∃ y, y = (x - 2)^2 + 1)
  (h2 : ∀ y, ∃ x, x - 1 = (y + 2)^2)
  (h_intersect : (∃ x y, (y = (x - 2)^2 + 1) ∧ (x - 1 = (y + 2)^2))) :
  x_1 + x_2 + x_3 + x_4 + y_1 + y_2 + y_3 + y_4 = 0 :=
sorry

end NUMINAMATH_GPT_parabola_intersection_sum_zero_l2376_237658


namespace NUMINAMATH_GPT_expression_simplifies_to_62_l2376_237637

theorem expression_simplifies_to_62 (a b c : ℕ) (h1 : a = 14) (h2 : b = 19) (h3 : c = 29) :
  (a^2 * (1 / b - 1 / c) + b^2 * (1 / c - 1 / a) + c^2 * (1 / a - 1 / b)) / 
  (a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)) = 62 := by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_expression_simplifies_to_62_l2376_237637


namespace NUMINAMATH_GPT_power_sums_equal_l2376_237603

theorem power_sums_equal (x y a b : ℝ)
  (h1 : x + y = a + b)
  (h2 : x^2 + y^2 = a^2 + b^2) :
  ∀ n : ℕ, x^n + y^n = a^n + b^n :=
by
  sorry

end NUMINAMATH_GPT_power_sums_equal_l2376_237603


namespace NUMINAMATH_GPT_weight_of_mixture_is_correct_l2376_237659

def weight_of_mixture (weight_a_per_l : ℕ) (weight_b_per_l : ℕ) 
                      (total_volume : ℕ) (ratio_a : ℕ) (ratio_b : ℕ) : ℚ :=
  let volume_a := (ratio_a : ℚ) / (ratio_a + ratio_b) * total_volume
  let volume_b := (ratio_b : ℚ) / (ratio_a + ratio_b) * total_volume
  let weight_a := volume_a * weight_a_per_l
  let weight_b := volume_b * weight_b_per_l
  (weight_a + weight_b) / 1000

theorem weight_of_mixture_is_correct :
  weight_of_mixture 800 850 3 3 2 = 2.46 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_mixture_is_correct_l2376_237659


namespace NUMINAMATH_GPT_g_five_eq_one_l2376_237640

noncomputable def g : ℝ → ℝ := sorry

axiom g_mul (x y : ℝ) : g (x * y) = g x * g y
axiom g_zero_ne_zero : g 0 ≠ 0

theorem g_five_eq_one : g 5 = 1 := by
  sorry

end NUMINAMATH_GPT_g_five_eq_one_l2376_237640


namespace NUMINAMATH_GPT_function_decreasing_iff_a_neg_l2376_237653

variable (a : ℝ)

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 ≥ f x2

theorem function_decreasing_iff_a_neg (h : ∀ x : ℝ, (7 * a * x ^ 6) ≤ 0) : a < 0 :=
by
  sorry

end NUMINAMATH_GPT_function_decreasing_iff_a_neg_l2376_237653


namespace NUMINAMATH_GPT_previous_salary_l2376_237648

theorem previous_salary (P : ℝ) (h : 1.05 * P = 2100) : P = 2000 :=
by
  sorry

end NUMINAMATH_GPT_previous_salary_l2376_237648


namespace NUMINAMATH_GPT_proportion_x_l2376_237621

theorem proportion_x (x : ℝ) (h : 0.60 / x = 6 / 4) : x = 0.4 :=
sorry

end NUMINAMATH_GPT_proportion_x_l2376_237621


namespace NUMINAMATH_GPT_min_x_squared_plus_y_squared_l2376_237605

theorem min_x_squared_plus_y_squared (x y : ℝ) (h : (x + 3) * (y - 3) = 0) : x^2 + y^2 = 18 :=
sorry

end NUMINAMATH_GPT_min_x_squared_plus_y_squared_l2376_237605


namespace NUMINAMATH_GPT_min_abs_sum_l2376_237606

theorem min_abs_sum (x y : ℝ) : (|x - 1| + |x| + |y - 1| + |y + 1|) ≥ 3 :=
sorry

end NUMINAMATH_GPT_min_abs_sum_l2376_237606


namespace NUMINAMATH_GPT_Gloria_pine_tree_price_l2376_237635

theorem Gloria_pine_tree_price :
  ∀ (cabin_cost cash cypress_count pine_count maple_count cypress_price maple_price left_over_price : ℕ)
  (cypress_total maple_total total_required total_from_cypress_and_maple total_needed amount_per_pine : ℕ),
    cabin_cost = 129000 →
    cash = 150 →
    cypress_count = 20 →
    pine_count = 600 →
    maple_count = 24 →
    cypress_price = 100 →
    maple_price = 300 →
    left_over_price = 350 →
    cypress_total = cypress_count * cypress_price →
    maple_total = maple_count * maple_price →
    total_required = cabin_cost - cash + left_over_price →
    total_from_cypress_and_maple = cypress_total + maple_total →
    total_needed = total_required - total_from_cypress_and_maple →
    amount_per_pine = total_needed / pine_count →
    amount_per_pine = 200 :=
by
  intros
  sorry

end NUMINAMATH_GPT_Gloria_pine_tree_price_l2376_237635


namespace NUMINAMATH_GPT_max_ab_l2376_237604

theorem max_ab {a b : ℝ} (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 6) : ab ≤ 9 :=
sorry

end NUMINAMATH_GPT_max_ab_l2376_237604


namespace NUMINAMATH_GPT_percent_of_12356_equals_1_2356_l2376_237654

theorem percent_of_12356_equals_1_2356 (p : ℝ) (h : p * 12356 = 1.2356) : p = 0.0001 := sorry

end NUMINAMATH_GPT_percent_of_12356_equals_1_2356_l2376_237654


namespace NUMINAMATH_GPT_min_square_side_length_l2376_237636

theorem min_square_side_length (s : ℝ) (h : s^2 ≥ 625) : s ≥ 25 :=
sorry

end NUMINAMATH_GPT_min_square_side_length_l2376_237636


namespace NUMINAMATH_GPT_parallelogram_rectangle_l2376_237633

/-- A quadrilateral is a parallelogram if both pairs of opposite sides are equal,
and it is a rectangle if its diagonals are equal. -/
structure Quadrilateral :=
  (side1 side2 side3 side4 : ℝ)
  (diag1 diag2 : ℝ)

structure Parallelogram extends Quadrilateral :=
  (opposite_sides_equal : side1 = side3 ∧ side2 = side4)

def is_rectangle (p : Parallelogram) : Prop :=
  p.diag1 = p.diag2 → (p.side1^2 + p.side2^2 = p.side3^2 + p.side4^2)

theorem parallelogram_rectangle (p : Parallelogram) : is_rectangle p :=
  sorry

end NUMINAMATH_GPT_parallelogram_rectangle_l2376_237633


namespace NUMINAMATH_GPT_range_of_function_l2376_237693

theorem range_of_function :
  ∀ y : ℝ, (∃ x : ℝ, x ≠ -2 ∧ y = (x^2 + 5*x + 6)/(x + 2)) ↔ (y ∈ Set.Iio 1 ∨ y ∈ Set.Ioi 1) := 
sorry

end NUMINAMATH_GPT_range_of_function_l2376_237693


namespace NUMINAMATH_GPT_chord_line_eq_l2376_237641

open Real

def ellipse (x y : ℝ) : Prop :=
  x^2 / 36 + y^2 / 9 = 1

def bisecting_point (a b : ℝ) : Prop :=
  a = 4 ∧ b = 2

theorem chord_line_eq :
  (∃ (k : ℝ), ∀ (x y : ℝ), ellipse x y → bisecting_point ((x + y) / 2) ((x + y) / 2) → y - 2 = k * (x - 4)) →
  (∃ (x y : ℝ), ellipse x y ∧ x + 2 * y - 8 = 0) :=
by
  sorry

end NUMINAMATH_GPT_chord_line_eq_l2376_237641


namespace NUMINAMATH_GPT_largest_k_l2376_237699

theorem largest_k (k n : ℕ) (h1 : 2^11 = (k * (2 * n + k + 1)) / 2) : k = 1 := sorry

end NUMINAMATH_GPT_largest_k_l2376_237699


namespace NUMINAMATH_GPT_triangle_area_l2376_237686

noncomputable def s (a b c : ℝ) : ℝ := (a + b + c) / 2
noncomputable def area (a b c : ℝ) : ℝ := Real.sqrt (s a b c * (s a b c - a) * (s a b c - b) * (s a b c - c))

theorem triangle_area (a b c : ℝ) (ha : a = 13) (hb : b = 12) (hc : c = 5) : area a b c = 30 := by
  rw [ha, hb, hc]
  show area 13 12 5 = 30
  -- manually calculate and reduce the expression to verify the theorem
  sorry

end NUMINAMATH_GPT_triangle_area_l2376_237686


namespace NUMINAMATH_GPT_sign_of_k_l2376_237639

variable (k x y : ℝ)
variable (A B : ℝ × ℝ)
variable (y₁ y₂ : ℝ)
variable (h₁ : A = (-2, y₁))
variable (h₂ : B = (5, y₂))
variable (h₃ : y₁ = k / -2)
variable (h₄ : y₂ = k / 5)
variable (h₅ : y₁ > y₂)
variable (h₀ : k ≠ 0)

-- We need to prove that k < 0
theorem sign_of_k (A B : ℝ × ℝ) (y₁ y₂ k : ℝ) 
  (h₁ : A = (-2, y₁)) 
  (h₂ : B = (5, y₂)) 
  (h₃ : y₁ = k / -2) 
  (h₄ : y₂ = k / 5) 
  (h₅ : y₁ > y₂) 
  (h₀ : k ≠ 0) : k < 0 := 
by
  sorry

end NUMINAMATH_GPT_sign_of_k_l2376_237639


namespace NUMINAMATH_GPT_ratio_of_y_and_z_l2376_237692

variable (x y z : ℝ)

theorem ratio_of_y_and_z (h1 : x + y = 2 * x + z) (h2 : x - 2 * y = 4 * z) (h3 : x + y + z = 21) : y / z = -5 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_y_and_z_l2376_237692


namespace NUMINAMATH_GPT_range_of_x_div_y_l2376_237627

theorem range_of_x_div_y {x y : ℝ} (hx : 1 < x ∧ x < 6) (hy : 2 < y ∧ y < 8) : 
  (1/8 < x / y) ∧ (x / y < 3) :=
sorry

end NUMINAMATH_GPT_range_of_x_div_y_l2376_237627


namespace NUMINAMATH_GPT_polynomial_roots_to_determinant_l2376_237602

noncomputable def determinant_eq (a b c m p q : ℂ) : Prop :=
  (Matrix.det ![
    ![a, 1, 1],
    ![1, b, 1],
    ![1, 1, c]
  ] = 2 - m - q)

theorem polynomial_roots_to_determinant (a b c m p q : ℂ) 
  (h1 : Polynomial.eval a (Polynomial.C q + Polynomial.monomial 1 p + Polynomial.monomial 2 m + Polynomial.monomial 3 1) = 0)
  (h2 : Polynomial.eval b (Polynomial.C q + Polynomial.monomial 1 p + Polynomial.monomial 2 m + Polynomial.monomial 3 1) = 0)
  (h3 : Polynomial.eval c (Polynomial.C q + Polynomial.monomial 1 p + Polynomial.monomial 2 m + Polynomial.monomial 3 1) = 0)
  : determinant_eq a b c m p q :=
by sorry

end NUMINAMATH_GPT_polynomial_roots_to_determinant_l2376_237602


namespace NUMINAMATH_GPT_no_solutions_l2376_237608

theorem no_solutions (x : ℝ) (h : x ≠ 0) : 4 * Real.sin x - 3 * Real.cos x ≠ 5 + 1 / |x| := 
by
  sorry

end NUMINAMATH_GPT_no_solutions_l2376_237608


namespace NUMINAMATH_GPT_oranges_in_bin_l2376_237626

theorem oranges_in_bin (initial_oranges thrown_out new_oranges : ℕ) (h1 : initial_oranges = 34) (h2 : thrown_out = 20) (h3 : new_oranges = 13) :
  (initial_oranges - thrown_out + new_oranges = 27) :=
by
  sorry

end NUMINAMATH_GPT_oranges_in_bin_l2376_237626


namespace NUMINAMATH_GPT_sum_first_10_log_a_l2376_237610

-- Given sum of the first n terms of the sequence
def S (n : ℕ) : ℕ := 2^n - 1

-- Function to get general term log_2 a_n
def log_a (n : ℕ) : ℕ := n - 1

-- The statement to prove
theorem sum_first_10_log_a : (List.range 10).sum = 45 := by 
  sorry

end NUMINAMATH_GPT_sum_first_10_log_a_l2376_237610


namespace NUMINAMATH_GPT_geometric_sequence_problem_l2376_237676

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∀ n, a n = a 0 * (1 / 2) ^ n

theorem geometric_sequence_problem 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : ∀ n, S n = (a 0 * (1 - (1 / 2 : ℝ) ^ n)) / (1 - (1 / 2)))
  (h3 : a 0 + a 2 = 5 / 2)
  (h4 : a 1 + a 3 = 5 / 4) :
  ∀ n, S n / a n = 2 ^ n - 1 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l2376_237676


namespace NUMINAMATH_GPT_day_of_week_nminus1_l2376_237660

theorem day_of_week_nminus1 (N : ℕ) 
  (h1 : (250 % 7 = 3 ∧ (250 / 7 * 7 + 3 = 250)) ∧ (150 % 7 = 3 ∧ (150 / 7 * 7 + 3 = 150))) :
  (50 % 7 = 0 ∧ (50 / 7 * 7 = 50)) := 
sorry

end NUMINAMATH_GPT_day_of_week_nminus1_l2376_237660


namespace NUMINAMATH_GPT_ax5_by5_eq_28616_l2376_237667

variables (a b x y : ℝ)

theorem ax5_by5_eq_28616
  (h1 : a * x + b * y = 1)
  (h2 : a * x^2 + b * y^2 = 9)
  (h3 : a * x^3 + b * y^3 = 28)
  (h4 : a * x^4 + b * y^4 = 96) :
  a * x^5 + b * y^5 = 28616 :=
sorry

end NUMINAMATH_GPT_ax5_by5_eq_28616_l2376_237667


namespace NUMINAMATH_GPT_evaluate_expression_l2376_237655

theorem evaluate_expression : (1:ℤ)^10 + (-1:ℤ)^8 + (-1:ℤ)^7 + (1:ℤ)^5 = 2 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2376_237655


namespace NUMINAMATH_GPT_min_value_expression_l2376_237630

noncomputable def expression (x : ℝ) : ℝ :=
  (15 - x) * (8 - x) * (15 + x) * (8 + x)

theorem min_value_expression : ∃ x : ℝ, expression x = -6480.25 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l2376_237630


namespace NUMINAMATH_GPT_B_is_criminal_l2376_237607

-- Introduce the conditions
variable (A B C : Prop)  -- A, B, and C represent whether each individual is the criminal.

-- A says they did not commit the crime
axiom A_says_innocent : ¬A

-- Exactly one of A_says_innocent must hold true (A says ¬A, so B or C must be true)
axiom exactly_one_assertion_true : (¬A ∨ B ∨ C)

-- Problem Statement: Prove that B is the criminal
theorem B_is_criminal : B :=
by
  -- Solution steps would go here
  sorry

end NUMINAMATH_GPT_B_is_criminal_l2376_237607


namespace NUMINAMATH_GPT_petya_time_comparison_l2376_237680

variables (D V : ℝ) (hD_pos : D > 0) (hV_pos : V > 0)

theorem petya_time_comparison (hD_pos : D > 0) (hV_pos : V > 0) :
  (41 * D / (40 * V)) > (D / V) :=
by
  sorry

end NUMINAMATH_GPT_petya_time_comparison_l2376_237680


namespace NUMINAMATH_GPT_no_positive_integer_solutions_l2376_237695

theorem no_positive_integer_solutions :
  ¬ ∃ (x y : ℕ) (h1 : x > 0) (h2 : y > 0), 21 * x * y = 7 - 3 * x - 4 * y :=
by
  sorry

end NUMINAMATH_GPT_no_positive_integer_solutions_l2376_237695


namespace NUMINAMATH_GPT_factorize_a_squared_plus_2a_l2376_237691

theorem factorize_a_squared_plus_2a (a : ℝ) : a^2 + 2*a = a * (a + 2) :=
sorry

end NUMINAMATH_GPT_factorize_a_squared_plus_2a_l2376_237691


namespace NUMINAMATH_GPT_quadratic_eq_solution_trig_expression_calc_l2376_237643

-- Part 1: Proof for the quadratic equation solution
theorem quadratic_eq_solution : ∀ (x : ℝ), x^2 - 4 * x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7 :=
by
  sorry

-- Part 2: Proof for trigonometric expression calculation
theorem trig_expression_calc : (-1 : ℝ) ^ 2 + 2 * Real.sin (Real.pi / 3) - Real.tan (Real.pi / 4) = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_eq_solution_trig_expression_calc_l2376_237643


namespace NUMINAMATH_GPT_smallest_non_multiple_of_5_abundant_l2376_237657

def proper_divisors (n : ℕ) : List ℕ := List.filter (fun d => d ∣ n ∧ d < n) (List.range (n + 1))

def is_abundant (n : ℕ) : Prop := (proper_divisors n).sum > n

def is_not_multiple_of_5 (n : ℕ) : Prop := ¬ (5 ∣ n)

theorem smallest_non_multiple_of_5_abundant : ∃ n, is_abundant n ∧ is_not_multiple_of_5 n ∧ 
  ∀ m, is_abundant m ∧ is_not_multiple_of_5 m → n ≤ m :=
  sorry

end NUMINAMATH_GPT_smallest_non_multiple_of_5_abundant_l2376_237657


namespace NUMINAMATH_GPT_fraction_of_money_left_l2376_237661

theorem fraction_of_money_left (m c : ℝ) 
   (h1 : (1/5) * m = (1/3) * c) :
   (m - ((3/5) * m) = (2/5) * m) := by
  sorry

end NUMINAMATH_GPT_fraction_of_money_left_l2376_237661


namespace NUMINAMATH_GPT_oranges_less_per_student_l2376_237665

def total_students : ℕ := 12
def total_oranges : ℕ := 108
def bad_oranges : ℕ := 36

theorem oranges_less_per_student :
  (total_oranges / total_students) - ((total_oranges - bad_oranges) / total_students) = 3 :=
by
  sorry

end NUMINAMATH_GPT_oranges_less_per_student_l2376_237665


namespace NUMINAMATH_GPT_multiple_of_denominator_l2376_237646

def denominator := 5
def numerator := denominator + 4

theorem multiple_of_denominator:
  (numerator + 6) = 3 * denominator :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_multiple_of_denominator_l2376_237646


namespace NUMINAMATH_GPT_sequence_an_l2376_237645

theorem sequence_an (a : ℕ → ℝ) (h0 : a 1 = 1)
  (h1 : ∀ n, 4 * a n * a (n + 1) = (a n + a (n + 1) - 1)^2)
  (h2 : ∀ n > 1, a n > a (n - 1)) :
  ∀ n, a n = n^2 := 
sorry

end NUMINAMATH_GPT_sequence_an_l2376_237645


namespace NUMINAMATH_GPT_sqrt_mixed_number_eq_l2376_237682

noncomputable def mixed_number : ℝ := 8 + 1 / 9

theorem sqrt_mixed_number_eq : Real.sqrt (8 + 1 / 9) = Real.sqrt 73 / 3 := by
  sorry

end NUMINAMATH_GPT_sqrt_mixed_number_eq_l2376_237682


namespace NUMINAMATH_GPT_henry_present_age_l2376_237634

theorem henry_present_age (H J : ℕ) (h1 : H + J = 41) (h2 : H - 7 = 2 * (J - 7)) : H = 25 :=
sorry

end NUMINAMATH_GPT_henry_present_age_l2376_237634


namespace NUMINAMATH_GPT_mean_of_combined_sets_l2376_237613

theorem mean_of_combined_sets 
  (mean1 mean2 mean3 : ℚ)
  (count1 count2 count3 : ℕ)
  (h1 : mean1 = 15)
  (h2 : mean2 = 20)
  (h3 : mean3 = 12)
  (hc1 : count1 = 7)
  (hc2 : count2 = 8)
  (hc3 : count3 = 5) :
  ((count1 * mean1 + count2 * mean2 + count3 * mean3) / (count1 + count2 + count3)) = 16.25 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_combined_sets_l2376_237613


namespace NUMINAMATH_GPT_sum_a_c_eq_l2376_237629

theorem sum_a_c_eq
  (a b c d : ℝ)
  (h1 : a * b + a * c + b * c + b * d + c * d + a * d = 40)
  (h2 : b^2 + d^2 = 29) :
  a + c = 8.4 :=
by
  sorry

end NUMINAMATH_GPT_sum_a_c_eq_l2376_237629


namespace NUMINAMATH_GPT_exist_distinct_indices_l2376_237601

theorem exist_distinct_indices (n : ℕ) (h1 : n > 3)
  (a : Fin n.succ → ℕ) 
  (h2 : StrictMono a) 
  (h3 : a n ≤ 2 * n - 3) :
  ∃ (i j k l m : Fin n.succ), 
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧ 
    j ≠ k ∧ j ≠ l ∧ j ≠ m ∧ 
    k ≠ l ∧ k ≠ m ∧ l ≠ m ∧ 
    a i + a j = a k + a l ∧ 
    a k + a l = a m := 
sorry

end NUMINAMATH_GPT_exist_distinct_indices_l2376_237601


namespace NUMINAMATH_GPT_compare_abc_l2376_237690

theorem compare_abc (a b c : Real) (h1 : a = Real.sqrt 3) (h2 : b = Real.log 2) (h3 : c = Real.logb 3 (Real.sin (Real.pi / 6))) :
  a > b ∧ b > c :=
by
  sorry

end NUMINAMATH_GPT_compare_abc_l2376_237690


namespace NUMINAMATH_GPT_problem_statement_l2376_237611

def diamond (x y : ℝ) : ℝ := (x + y) ^ 2 * (x - y) ^ 2

theorem problem_statement : diamond 2 (diamond 3 4) = 5745329 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l2376_237611


namespace NUMINAMATH_GPT_tan_pi_minus_alpha_l2376_237616

theorem tan_pi_minus_alpha 
  (α : ℝ) 
  (h1 : Real.sin α = 1 / 3) 
  (h2 : π / 2 < α) 
  (h3 : α < π) :
  Real.tan (π - α) = Real.sqrt 2 / 4 :=
by
  sorry

end NUMINAMATH_GPT_tan_pi_minus_alpha_l2376_237616


namespace NUMINAMATH_GPT_sqrt_four_eq_two_or_neg_two_l2376_237618

theorem sqrt_four_eq_two_or_neg_two (x : ℝ) : x^2 = 4 ↔ x = 2 ∨ x = -2 :=
by 
  sorry

end NUMINAMATH_GPT_sqrt_four_eq_two_or_neg_two_l2376_237618


namespace NUMINAMATH_GPT_change_received_after_discounts_and_taxes_l2376_237672

theorem change_received_after_discounts_and_taxes :
  let price_wooden_toy : ℝ := 20
  let price_hat : ℝ := 10
  let tax_rate : ℝ := 0.08
  let discount_wooden_toys : ℝ := 0.15
  let discount_hats : ℝ := 0.10
  let quantity_wooden_toys : ℝ := 3
  let quantity_hats : ℝ := 4
  let amount_paid : ℝ := 200
  let cost_wooden_toys := quantity_wooden_toys * price_wooden_toy
  let discounted_cost_wooden_toys := cost_wooden_toys - (discount_wooden_toys * cost_wooden_toys)
  let cost_hats := quantity_hats * price_hat
  let discounted_cost_hats := cost_hats - (discount_hats * cost_hats)
  let total_cost_before_tax := discounted_cost_wooden_toys + discounted_cost_hats
  let tax := tax_rate * total_cost_before_tax
  let total_cost_after_tax := total_cost_before_tax + tax
  let change_received := amount_paid - total_cost_after_tax
  change_received = 106.04 := by
  -- All the conditions and intermediary steps are defined above, from problem to solution.
  sorry

end NUMINAMATH_GPT_change_received_after_discounts_and_taxes_l2376_237672


namespace NUMINAMATH_GPT_ursula_hourly_wage_l2376_237628

def annual_salary : ℝ := 16320
def hours_per_day : ℝ := 8
def days_per_month : ℝ := 20
def months_per_year : ℝ := 12

theorem ursula_hourly_wage : 
  (annual_salary / months_per_year) / (hours_per_day * days_per_month) = 8.50 := by 
  sorry

end NUMINAMATH_GPT_ursula_hourly_wage_l2376_237628


namespace NUMINAMATH_GPT_range_of_m_exists_l2376_237698

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3 * x

-- Proof problem statement
theorem range_of_m_exists (m : ℝ) (h : m ∈ Set.Icc (-2 : ℝ) (0 : ℝ)) : 
  ∃ x ∈ Set.Icc (0 : ℝ) (1 : ℝ), f x = m := 
by
  sorry

end NUMINAMATH_GPT_range_of_m_exists_l2376_237698


namespace NUMINAMATH_GPT_ten_times_hundred_eq_thousand_ten_times_thousand_eq_ten_thousand_hundreds_in_ten_thousand_tens_in_one_thousand_l2376_237663

theorem ten_times_hundred_eq_thousand : 10 * 100 = 1000 := 
by sorry

theorem ten_times_thousand_eq_ten_thousand : 10 * 1000 = 10000 := 
by sorry

theorem hundreds_in_ten_thousand : 10000 / 100 = 100 := 
by sorry

theorem tens_in_one_thousand : 1000 / 10 = 100 := 
by sorry

end NUMINAMATH_GPT_ten_times_hundred_eq_thousand_ten_times_thousand_eq_ten_thousand_hundreds_in_ten_thousand_tens_in_one_thousand_l2376_237663


namespace NUMINAMATH_GPT_number_decomposition_l2376_237632

theorem number_decomposition (n : ℕ) : n = 6058 → (n / 1000 = 6) ∧ ((n % 100) / 10 = 5) ∧ (n % 10 = 8) :=
by
  -- Actual proof will go here
  sorry

end NUMINAMATH_GPT_number_decomposition_l2376_237632


namespace NUMINAMATH_GPT_correct_division_incorrect_addition_incorrect_multiplication_incorrect_squaring_only_correct_operation_l2376_237656

theorem correct_division (x : ℝ) : x^6 / x^3 = x^3 := by 
  sorry

theorem incorrect_addition (x : ℝ) : ¬(x^2 + x^3 = 2 * x^5) := by 
  sorry

theorem incorrect_multiplication (x : ℝ) : ¬(x^2 * x^3 = x^6) := by 
  sorry

theorem incorrect_squaring (x : ℝ) : ¬((-x^3) ^ 2 = -x^6) := by 
  sorry

theorem only_correct_operation (x : ℝ) : 
  (x^6 / x^3 = x^3) ∧ ¬(x^2 + x^3 = 2 * x^5) ∧ ¬(x^2 * x^3 = x^6) ∧ ¬((-x^3) ^ 2 = -x^6) := 
  by
    exact ⟨correct_division x, incorrect_addition x, incorrect_multiplication x,
           incorrect_squaring x⟩

end NUMINAMATH_GPT_correct_division_incorrect_addition_incorrect_multiplication_incorrect_squaring_only_correct_operation_l2376_237656


namespace NUMINAMATH_GPT_inequality_proof_l2376_237600

theorem inequality_proof (a b c : ℝ) (hab : a * b < 0) : 
  a^2 + b^2 + c^2 > 2 * a * b + 2 * b * c + 2 * c * a := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l2376_237600


namespace NUMINAMATH_GPT_A_eq_B_l2376_237644

noncomputable def A := Real.sqrt 5 + Real.sqrt (22 + 2 * Real.sqrt 5)
noncomputable def B := Real.sqrt (11 + 2 * Real.sqrt 29) 
                      + Real.sqrt (16 - 2 * Real.sqrt 29 
                                   + 2 * Real.sqrt (55 - 10 * Real.sqrt 29))

theorem A_eq_B : A = B := 
  sorry

end NUMINAMATH_GPT_A_eq_B_l2376_237644


namespace NUMINAMATH_GPT_average_daily_low_temperature_l2376_237677

theorem average_daily_low_temperature (temps : List ℕ) (h_len : temps.length = 5) 
  (h_vals : temps = [40, 47, 45, 41, 39]) : 
  (temps.sum / 5 : ℝ) = 42.4 := 
by
  sorry

end NUMINAMATH_GPT_average_daily_low_temperature_l2376_237677


namespace NUMINAMATH_GPT_Masha_initial_ball_count_l2376_237673

theorem Masha_initial_ball_count (r w n p : ℕ) (h1 : r + n * w = 101) (h2 : p * r + w = 103) (hn : n ≠ 0) :
  r + w = 51 ∨ r + w = 68 :=
  sorry

end NUMINAMATH_GPT_Masha_initial_ball_count_l2376_237673


namespace NUMINAMATH_GPT_abi_suji_age_ratio_l2376_237623

theorem abi_suji_age_ratio (A S : ℕ) (h1 : S = 24) 
  (h2 : (A + 3) / (S + 3) = 11 / 9) : A / S = 5 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_abi_suji_age_ratio_l2376_237623


namespace NUMINAMATH_GPT_range_of_m_l2376_237619

theorem range_of_m (m x1 x2 y1 y2 : ℝ) (h₁ : x1 < x2) (h₂ : y1 < y2)
  (A_on_line : y1 = (2 * m - 1) * x1 + 1)
  (B_on_line : y2 = (2 * m - 1) * x2 + 1) :
  m > 0.5 :=
sorry

end NUMINAMATH_GPT_range_of_m_l2376_237619


namespace NUMINAMATH_GPT_total_cats_in_center_l2376_237684

def cats_training_center : ℕ := 45
def cats_can_fetch : ℕ := 25
def cats_can_meow : ℕ := 40
def cats_jump_and_fetch : ℕ := 15
def cats_fetch_and_meow : ℕ := 20
def cats_jump_and_meow : ℕ := 23
def cats_all_three : ℕ := 10
def cats_none : ℕ := 5

theorem total_cats_in_center :
  (cats_training_center - (cats_jump_and_fetch + cats_jump_and_meow - cats_all_three)) +
  (cats_all_three) +
  (cats_fetch_and_meow - cats_all_three) +
  (cats_jump_and_fetch - cats_all_three) +
  (cats_jump_and_meow - cats_all_three) +
  cats_none = 67 := by
  sorry

end NUMINAMATH_GPT_total_cats_in_center_l2376_237684


namespace NUMINAMATH_GPT_azalea_profit_l2376_237696

def num_sheep : Nat := 200
def wool_per_sheep : Nat := 10
def price_per_pound : Nat := 20
def shearer_cost : Nat := 2000

theorem azalea_profit : 
  (num_sheep * wool_per_sheep * price_per_pound) - shearer_cost = 38000 := 
by
  sorry

end NUMINAMATH_GPT_azalea_profit_l2376_237696


namespace NUMINAMATH_GPT_quotient_of_501_div_0_point_5_l2376_237679

theorem quotient_of_501_div_0_point_5 : 501 / 0.5 = 1002 := by
  sorry

end NUMINAMATH_GPT_quotient_of_501_div_0_point_5_l2376_237679


namespace NUMINAMATH_GPT_pyramid_top_row_missing_number_l2376_237662

theorem pyramid_top_row_missing_number (a b c d e f g : ℕ)
  (h₁ : b * c = 720)
  (h₂ : a * b = 240)
  (h₃ : c * d = 1440)
  (h₄ : c = 6)
  : a = 120 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_top_row_missing_number_l2376_237662


namespace NUMINAMATH_GPT_ring_revolutions_before_stopping_l2376_237624

variable (R ω μ m g : ℝ) -- Declare the variables as real numbers

-- Statement of the theorem
theorem ring_revolutions_before_stopping
  (h_positive_R : 0 < R)
  (h_positive_ω : 0 < ω)
  (h_positive_μ : 0 < μ)
  (h_positive_m : 0 < m)
  (h_positive_g : 0 < g) :
  let N1 := m * g / (1 + μ^2)
  let N2 := μ * m * g / (1 + μ^2)
  let K_initial := (1 / 2) * m * R^2 * ω^2
  let A_friction := -2 * π * R * n * μ * (N1 + N2)
  ∃ n : ℝ, n = ω^2 * R * (1 + μ^2) / (4 * π * g * μ * (1 + μ)) :=
by sorry

end NUMINAMATH_GPT_ring_revolutions_before_stopping_l2376_237624


namespace NUMINAMATH_GPT_Moe_has_least_amount_of_money_l2376_237694

variables {B C F J M Z : ℕ}

theorem Moe_has_least_amount_of_money
  (h1 : Z > F) (h2 : F > B) (h3 : Z > C) (h4 : B > M) (h5 : C > M) (h6 : Z > J) (h7 : J > M) :
  ∀ x, x ≠ M → x > M :=
by {
  sorry
}

end NUMINAMATH_GPT_Moe_has_least_amount_of_money_l2376_237694


namespace NUMINAMATH_GPT_percentage_caught_sampling_candy_l2376_237687

theorem percentage_caught_sampling_candy
  (S : ℝ) (C : ℝ)
  (h1 : 0.1 * S = 0.1 * 24.444444444444443) -- 10% of the customers who sample the candy are not caught
  (h2 : S = 24.444444444444443)  -- The total percent of all customers who sample candy is 24.444444444444443%
  :
  C = 0.9 * 24.444444444444443 := -- Equivalent \( C \approx 22 \% \)
by
  sorry

end NUMINAMATH_GPT_percentage_caught_sampling_candy_l2376_237687


namespace NUMINAMATH_GPT_expression_rewrite_l2376_237668

theorem expression_rewrite :
  ∃ (d r s : ℚ), (∀ k : ℚ, 8*k^2 - 6*k + 16 = d*(k + r)^2 + s) ∧ s / r = -118 / 3 :=
by sorry

end NUMINAMATH_GPT_expression_rewrite_l2376_237668


namespace NUMINAMATH_GPT_product_of_abc_l2376_237669

-- Define the constants and conditions
variables (a b c m : ℝ)
axiom h1 : a + b + c = 180
axiom h2 : 5 * a = m
axiom h3 : b = m + 12
axiom h4 : c = m - 6

-- Prove that the product of a, b, and c is 42184
theorem product_of_abc : a * b * c = 42184 :=
by {
  sorry
}

end NUMINAMATH_GPT_product_of_abc_l2376_237669


namespace NUMINAMATH_GPT_frog_hops_ratio_l2376_237638

theorem frog_hops_ratio (S T F : ℕ) (h1 : S = 2 * T) (h2 : S = 18) (h3 : F + S + T = 99) :
  F / S = 4 / 1 :=
by
  sorry

end NUMINAMATH_GPT_frog_hops_ratio_l2376_237638


namespace NUMINAMATH_GPT_AgOH_moles_formed_l2376_237685

noncomputable def number_of_moles_of_AgOH (n_AgNO3 n_NaOH : ℕ) : ℕ :=
  if n_AgNO3 = n_NaOH then n_AgNO3 else 0

theorem AgOH_moles_formed :
  number_of_moles_of_AgOH 3 3 = 3 := by
  sorry

end NUMINAMATH_GPT_AgOH_moles_formed_l2376_237685


namespace NUMINAMATH_GPT_pipes_fill_tank_in_7_minutes_l2376_237688

theorem pipes_fill_tank_in_7_minutes (T : ℕ) (R_A R_B R_combined : ℚ) 
  (h1 : R_A = 1 / 56) 
  (h2 : R_B = 7 * R_A)
  (h3 : R_combined = R_A + R_B)
  (h4 : T = 1 / R_combined) : 
  T = 7 := by 
  sorry

end NUMINAMATH_GPT_pipes_fill_tank_in_7_minutes_l2376_237688


namespace NUMINAMATH_GPT_find_k_l2376_237664

-- Definition of the vertices and conditions
variables {t k : ℝ}
def A : (ℝ × ℝ) := (0, 3)
def B : (ℝ × ℝ) := (0, k)
def C : (ℝ × ℝ) := (t, 10)
def D : (ℝ × ℝ) := (t, 0)

-- Condition that the area of the quadrilateral is 50 square units
def area_cond (height base1 base2 : ℝ) : Prop :=
  50 = (1 / 2) * height * (base1 + base2)

-- Stating the problem in Lean
theorem find_k
  (ht : t = 5)
  (hk : k > 3) 
  (t_pos : t > 0)
  (area : area_cond t (k - 3) 10) :
  k = 13 :=
  sorry

end NUMINAMATH_GPT_find_k_l2376_237664


namespace NUMINAMATH_GPT_same_color_points_exist_l2376_237622

theorem same_color_points_exist (d : ℝ) (colored_plane : ℝ × ℝ → Prop) :
  (∃ (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧ colored_plane p1 = colored_plane p2 ∧ dist p1 p2 = d) := 
sorry

end NUMINAMATH_GPT_same_color_points_exist_l2376_237622


namespace NUMINAMATH_GPT_F_atoms_in_compound_l2376_237649

-- Given conditions
def atomic_weight_Al : Real := 26.98
def atomic_weight_F : Real := 19.00
def molecular_weight : Real := 84

-- Defining the assertion: number of F atoms in the compound
def number_of_F_atoms (n : Real) : Prop :=
  molecular_weight = atomic_weight_Al + n * atomic_weight_F

-- Proving the assertion that the number of F atoms is approximately 3
theorem F_atoms_in_compound : number_of_F_atoms 3 :=
  by
  sorry

end NUMINAMATH_GPT_F_atoms_in_compound_l2376_237649


namespace NUMINAMATH_GPT_jeans_sold_l2376_237666

-- Definitions based on conditions
def price_per_jean : ℤ := 11
def price_per_tee : ℤ := 8
def tees_sold : ℤ := 7
def total_money : ℤ := 100

-- Proof statement
theorem jeans_sold (J : ℤ)
  (h1 : price_per_jean = 11)
  (h2 : price_per_tee = 8)
  (h3 : tees_sold = 7)
  (h4 : total_money = 100) :
  J = 4 :=
by
  sorry

end NUMINAMATH_GPT_jeans_sold_l2376_237666


namespace NUMINAMATH_GPT_intersection_complement_l2376_237689

-- Defining the sets A and B
def setA : Set ℝ := { x | -3 < x ∧ x < 3 }
def setB : Set ℝ := { x | x < -2 }
def complementB : Set ℝ := { x | x ≥ -2 }

-- The theorem to be proved
theorem intersection_complement :
  setA ∩ complementB = { x | -2 ≤ x ∧ x < 3 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l2376_237689
