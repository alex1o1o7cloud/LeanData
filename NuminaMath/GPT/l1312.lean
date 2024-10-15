import Mathlib

namespace NUMINAMATH_GPT_pascal_row_12_sum_pascal_row_12_middle_l1312_131271

open Nat

/-- Definition of the sum of all numbers in a given row of Pascal's Triangle -/
def pascal_sum (n : ℕ) : ℕ :=
  2^n

/-- Definition of the binomial coefficient -/
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- Pascal Triangle Row 12 sum -/
theorem pascal_row_12_sum : pascal_sum 12 = 4096 :=
by
  sorry

/-- Pascal Triangle Row 12 middle number -/
theorem pascal_row_12_middle : binomial 12 6 = 924 :=
by
  sorry

end NUMINAMATH_GPT_pascal_row_12_sum_pascal_row_12_middle_l1312_131271


namespace NUMINAMATH_GPT_parallel_lines_value_of_m_l1312_131299

theorem parallel_lines_value_of_m (m : ℝ) 
  (h1 : ∀ x y : ℝ, x + m * y - 2 = 0 = (2 * x + (1 - m) * y + 2 = 0)) : 
  m = 1 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_parallel_lines_value_of_m_l1312_131299


namespace NUMINAMATH_GPT_vector_solution_l1312_131274

theorem vector_solution :
  let u := -6 / 41
  let v := -46 / 41
  let vec1 := (⟨3, -2⟩: ℝ × ℝ)
  let vec2 := (⟨5, -7⟩: ℝ × ℝ)
  let vec3 := (⟨0, 3⟩: ℝ × ℝ)
  let vec4 := (⟨-3, 4⟩: ℝ × ℝ)
  (vec1 + u • vec2 = vec3 + v • vec4) := by
  sorry

end NUMINAMATH_GPT_vector_solution_l1312_131274


namespace NUMINAMATH_GPT_roots_of_quadratic_l1312_131248

theorem roots_of_quadratic :
  ∃ (b c : ℝ), ( ∀ (x : ℝ), x^2 + b * x + c = 0 ↔ x = 1 ∨ x = -2) :=
sorry

end NUMINAMATH_GPT_roots_of_quadratic_l1312_131248


namespace NUMINAMATH_GPT_rectangle_shorter_side_length_l1312_131264

theorem rectangle_shorter_side_length (rope_length : ℕ) (long_side : ℕ) : 
  rope_length = 100 → long_side = 28 → 
  ∃ short_side : ℕ, (2 * long_side + 2 * short_side = rope_length) ∧ short_side = 22 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_shorter_side_length_l1312_131264


namespace NUMINAMATH_GPT_distance_center_to_plane_l1312_131238

theorem distance_center_to_plane (r : ℝ) (a b : ℝ) (h : a ^ 2 + b ^ 2 = 10 ^ 2) (d : ℝ) : 
  r = 13 → a = 6 → b = 8 → d = 12 := 
by 
  sorry

end NUMINAMATH_GPT_distance_center_to_plane_l1312_131238


namespace NUMINAMATH_GPT_bananas_to_oranges_l1312_131282

theorem bananas_to_oranges (B A O : ℕ) 
    (h1 : 4 * B = 3 * A) 
    (h2 : 7 * A = 5 * O) : 
    28 * B = 15 * O :=
by
  sorry

end NUMINAMATH_GPT_bananas_to_oranges_l1312_131282


namespace NUMINAMATH_GPT_hannah_monday_run_l1312_131288

-- Definitions of the conditions
def ran_on_wednesday : ℕ := 4816
def ran_on_friday : ℕ := 2095
def extra_on_monday : ℕ := 2089

-- Translations to set the total combined distance and the distance ran on Monday
def combined_distance := ran_on_wednesday + ran_on_friday
def ran_on_monday := combined_distance + extra_on_monday

-- A statement to show she ran 9 kilometers on Monday
theorem hannah_monday_run :
  ran_on_monday = 9000 / 1000 * 1000 := sorry

end NUMINAMATH_GPT_hannah_monday_run_l1312_131288


namespace NUMINAMATH_GPT_cost_of_adult_ticket_l1312_131247

theorem cost_of_adult_ticket (A : ℝ) (H1 : ∀ (cost_child : ℝ), cost_child = 7) 
                             (H2 : ∀ (num_adults : ℝ), num_adults = 2) 
                             (H3 : ∀ (num_children : ℝ), num_children = 2) 
                             (H4 : ∀ (total_cost : ℝ), total_cost = 58) :
    A = 22 :=
by
  -- You can assume variables for children's cost, number of adults, and number of children
  let cost_child := 7
  let num_adults := 2
  let num_children := 2
  let total_cost := 58
  
  -- Formalize the conditions given
  have H_children_cost : num_children * cost_child = 14 := by simp [cost_child, num_children]
  
  -- Establish the total cost equation
  have H_total_equation : num_adults * A + num_children * cost_child = total_cost := 
    by sorry  -- (Total_equation_proof)
  
  -- Solve for A
  sorry  -- Proof step

end NUMINAMATH_GPT_cost_of_adult_ticket_l1312_131247


namespace NUMINAMATH_GPT_ellipse_equation_l1312_131204

open Real

theorem ellipse_equation (x y : ℝ) (h₁ : (- sqrt 15) = x) (h₂ : (5 / 2) = y)
  (h₃ : ∃ (a b : ℝ), (a > b) ∧ (b > 0) ∧ (a^2 = b^2 + 5) 
  ∧ b^2 = 20 ∧ a^2 = 25) :
  (x^2 / 20 + y^2 / 25 = 1) :=
sorry

end NUMINAMATH_GPT_ellipse_equation_l1312_131204


namespace NUMINAMATH_GPT_binomial_expansion_conditions_l1312_131223

noncomputable def binomial_expansion (a b : ℝ) (x y : ℝ) (n : ℕ) : ℝ :=
(1 + a*x + b*y)^n

theorem binomial_expansion_conditions
  (a b : ℝ) (n : ℕ) 
  (h1 : (1 + b)^n = 243)
  (h2 : (1 + |a|)^n = 32) :
  a = 1 ∧ b = 2 ∧ n = 5 := by
  sorry

end NUMINAMATH_GPT_binomial_expansion_conditions_l1312_131223


namespace NUMINAMATH_GPT_tan_pi_minus_alpha_l1312_131228

theorem tan_pi_minus_alpha (α : ℝ) (h : 3 * Real.sin α = Real.cos α) : Real.tan (π - α) = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_pi_minus_alpha_l1312_131228


namespace NUMINAMATH_GPT_find_a8_l1312_131216

variable (a : ℕ → ℝ)
variable (q : ℝ)

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem find_a8 
  (hq : is_geometric_sequence a q)
  (h1 : a 1 * a 3 = 4)
  (h2 : a 9 = 256) : 
  a 8 = 128 ∨ a 8 = -128 :=
by
  sorry

end NUMINAMATH_GPT_find_a8_l1312_131216


namespace NUMINAMATH_GPT_range_of_b_distance_when_b_eq_one_l1312_131221

-- Definitions for conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1
def line (x y b : ℝ) : Prop := y = x + b
def intersect (x y b : ℝ) : Prop := ellipse x y ∧ line x y b

-- Prove the range of b for which there are two distinct intersection points
theorem range_of_b (b : ℝ) : (∃ x1 y1 x2 y2, x1 ≠ x2 ∧ intersect x1 y1 b ∧ intersect x2 y2 b) ↔ (-Real.sqrt 3 < b ∧ b < Real.sqrt 3) :=
by sorry

-- Prove the distance between points A and B when b = 1
theorem distance_when_b_eq_one : 
  ∃ x1 y1 x2 y2, intersect x1 y1 1 ∧ intersect x2 y2 1 ∧ Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 4 * Real.sqrt 2 / 3 :=
by sorry

end NUMINAMATH_GPT_range_of_b_distance_when_b_eq_one_l1312_131221


namespace NUMINAMATH_GPT_exists_K_p_l1312_131286

noncomputable def constant_K_p (p : ℝ) (hp : p > 1) : ℝ :=
  (p * p) / (p - 1)

theorem exists_K_p (p : ℝ) (hp : p > 1) :
  ∃ K_p > 0, ∀ x y : ℝ, |x|^p + |y|^p = 2 → (x - y)^2 ≤ K_p * (4 - (x + y)^2) :=
by
  use constant_K_p p hp
  sorry

end NUMINAMATH_GPT_exists_K_p_l1312_131286


namespace NUMINAMATH_GPT_angle_D_measure_l1312_131229

theorem angle_D_measure 
  (A B C D : Type)
  (angleA : ℝ)
  (angleB : ℝ)
  (angleC : ℝ)
  (angleD : ℝ)
  (BD_bisector : ℝ → ℝ) :
  angleA = 85 ∧ angleB = 50 ∧ angleC = 25 ∧ BD_bisector angleB = 25 →
  angleD = 130 :=
by
  intro h
  have hA := h.1
  have hB := h.2.1
  have hC := h.2.2.1
  have hBD := h.2.2.2
  sorry

end NUMINAMATH_GPT_angle_D_measure_l1312_131229


namespace NUMINAMATH_GPT_sqrt_expr_evaluation_l1312_131250

theorem sqrt_expr_evaluation : 
  (Real.sqrt 24) - 3 * (Real.sqrt (1 / 6)) + (Real.sqrt 6) = (5 * Real.sqrt 6) / 2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_expr_evaluation_l1312_131250


namespace NUMINAMATH_GPT_paco_initial_sweet_cookies_l1312_131232

theorem paco_initial_sweet_cookies
    (x : ℕ)  -- Paco's initial number of sweet cookies
    (eaten_sweet : ℕ)  -- number of sweet cookies Paco ate
    (left_sweet : ℕ)  -- number of sweet cookies Paco had left
    (h1 : eaten_sweet = 15)  -- Paco ate 15 sweet cookies
    (h2 : left_sweet = 19)  -- Paco had 19 sweet cookies left
    (h3 : x - eaten_sweet = left_sweet)  -- After eating, Paco had 19 sweet cookies left
    : x = 34 :=  -- Paco initially had 34 sweet cookies
sorry

end NUMINAMATH_GPT_paco_initial_sweet_cookies_l1312_131232


namespace NUMINAMATH_GPT_line_always_passes_through_fixed_point_l1312_131239

theorem line_always_passes_through_fixed_point :
  ∀ (m : ℝ), ∃ (x y : ℝ), (y = m * x + 2 * m + 1) ∧ (x = -2) ∧ (y = 1) :=
by
  sorry

end NUMINAMATH_GPT_line_always_passes_through_fixed_point_l1312_131239


namespace NUMINAMATH_GPT_value_of_a_b_c_l1312_131206

noncomputable def absolute_value (x : ℤ) : ℤ := abs x

theorem value_of_a_b_c (a b c : ℤ)
  (ha : absolute_value a = 1)
  (hb : absolute_value b = 2)
  (hc : absolute_value c = 3)
  (h : a > b ∧ b > c) :
  a + b - c = 2 ∨ a + b - c = 0 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_b_c_l1312_131206


namespace NUMINAMATH_GPT_find_a_l1312_131279

theorem find_a (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 3) (h₂ : 3 / a + 6 / b = 2 / 3) : 
  a = 9 * b / (2 * b - 18) :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1312_131279


namespace NUMINAMATH_GPT_smallest_m_l1312_131203

-- Definitions of lengths and properties of the pieces
variable {lengths : Fin 21 → ℝ} 
variable (h_all_pos : ∀ i, lengths i > 0)
variable (h_total_length : (Finset.univ : Finset (Fin 21)).sum lengths = 21)
variable (h_max_factor : ∀ i j, max (lengths i) (lengths j) ≤ 3 * min (lengths i) (lengths j))

-- Proof statement
theorem smallest_m (m : ℝ) (hm : ∀ i j, max (lengths i) (lengths j) ≤ m * min (lengths i) (lengths j)) : 
  m ≥ 1 := 
sorry

end NUMINAMATH_GPT_smallest_m_l1312_131203


namespace NUMINAMATH_GPT_area_difference_l1312_131270

noncomputable def speed_ratio_A_B : ℚ := 3 / 2
noncomputable def side_length : ℝ := 100
noncomputable def perimeter : ℝ := 4 * side_length

noncomputable def distance_A := (3 / 5) * perimeter
noncomputable def distance_B := perimeter - distance_A

noncomputable def EC := distance_A - 2 * side_length
noncomputable def DE := distance_B - side_length

noncomputable def area_ADE := 0.5 * DE * side_length
noncomputable def area_BCE := 0.5 * EC * side_length

theorem area_difference :
  (area_ADE - area_BCE) = 1000 :=
by
  sorry

end NUMINAMATH_GPT_area_difference_l1312_131270


namespace NUMINAMATH_GPT_solution_set_a_eq_1_find_a_min_value_3_l1312_131205

open Real

noncomputable def f (x a : ℝ) := 2 * abs (x + 1) + abs (x - a)

-- The statement for the first question
theorem solution_set_a_eq_1 (x : ℝ) : f x 1 ≥ 5 ↔ x ≤ -2 ∨ x ≥ (4 / 3) := 
by sorry

-- The statement for the second question
theorem find_a_min_value_3 (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 3) ∧ (∃ x : ℝ, f x a = 3) ↔ a = 2 ∨ a = -4 := 
by sorry

end NUMINAMATH_GPT_solution_set_a_eq_1_find_a_min_value_3_l1312_131205


namespace NUMINAMATH_GPT_exists_invisible_square_l1312_131278

def invisible (p q : ℤ) : Prop := Int.gcd p q > 1

theorem exists_invisible_square (n : ℤ) (h : 0 < n) : 
  ∃ (a b : ℤ), ∀ i j : ℤ, (0 ≤ i) ∧ (i < n) ∧ (0 ≤ j) ∧ (j < n) → invisible (a + i) (b + j) :=
by {
  sorry
}

end NUMINAMATH_GPT_exists_invisible_square_l1312_131278


namespace NUMINAMATH_GPT_integer_sequence_count_l1312_131284

theorem integer_sequence_count (a₀ : ℕ) (step : ℕ → ℕ) (n : ℕ) 
  (h₀ : a₀ = 5184)
  (h_step : ∀ k, k < n → step k = (a₀ / 4^k))
  (h_stop : a₀ = (4 ^ (n - 1)) * 81) :
  n = 4 := 
sorry

end NUMINAMATH_GPT_integer_sequence_count_l1312_131284


namespace NUMINAMATH_GPT_no_solution_for_n_eq_neg1_l1312_131292

theorem no_solution_for_n_eq_neg1 (x y z : ℝ) : ¬ (∃ x y z, (-1) * x^2 + y = 2 ∧ (-1) * y^2 + z = 2 ∧ (-1) * z^2 + x = 2) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_for_n_eq_neg1_l1312_131292


namespace NUMINAMATH_GPT_total_square_miles_of_plains_l1312_131211

-- Defining conditions
def region_east_of_b : ℕ := 200
def region_east_of_a : ℕ := region_east_of_b - 50

-- To test this statement in Lean 4
theorem total_square_miles_of_plains : region_east_of_a + region_east_of_b = 350 := by
  sorry

end NUMINAMATH_GPT_total_square_miles_of_plains_l1312_131211


namespace NUMINAMATH_GPT_find_n_l1312_131291

variable (x n : ℝ)

-- Definitions
def positive (x : ℝ) : Prop := x > 0
def equation (x n : ℝ) : Prop := x / n + x / 25 = 0.06 * x

-- Theorem statement
theorem find_n (h1 : positive x) (h2 : equation x n) : n = 50 :=
sorry

end NUMINAMATH_GPT_find_n_l1312_131291


namespace NUMINAMATH_GPT_trader_sells_cloth_l1312_131296

theorem trader_sells_cloth
  (total_SP : ℝ := 4950)
  (profit_per_meter : ℝ := 15)
  (cost_price_per_meter : ℝ := 51)
  (SP_per_meter : ℝ := cost_price_per_meter + profit_per_meter)
  (x : ℝ := total_SP / SP_per_meter) :
  x = 75 :=
by
  sorry

end NUMINAMATH_GPT_trader_sells_cloth_l1312_131296


namespace NUMINAMATH_GPT_graph_passes_through_point_l1312_131261

theorem graph_passes_through_point (a : ℝ) (ha : 0 < a) (ha_ne_one : a ≠ 1) :
  let f := fun x : ℝ => a^(x - 3) + 2
  f 3 = 3 := by
  sorry

end NUMINAMATH_GPT_graph_passes_through_point_l1312_131261


namespace NUMINAMATH_GPT_initial_position_l1312_131259

variable (x : Int)

theorem initial_position 
  (h: x - 5 + 4 + 2 - 3 + 1 = 6) : x = 7 := 
  by 
  sorry

end NUMINAMATH_GPT_initial_position_l1312_131259


namespace NUMINAMATH_GPT_problem1_problem2_l1312_131237

-- Define the sets of balls and boxes
inductive Ball
| ball1 | ball2 | ball3 | ball4

inductive Box
| boxA | boxB | boxC

-- Define the arrangements for the first problem
def arrangements_condition1 (arrangement : Ball → Box) : Prop :=
  (arrangement Ball.ball3 = Box.boxB) ∧
  (∃ b1 b2 b3 : Box, b1 ≠ b2 ∧ b2 ≠ b3 ∧ b3 ≠ b1 ∧ 
    ∃ (f : Ball → Box), 
      (f Ball.ball1 = b1) ∧ (f Ball.ball2 = b2) ∧ (f Ball.ball3 = Box.boxB) ∧ (f Ball.ball4 = b3))

-- Define the proof statement for the first problem
theorem problem1 : ∃ n : ℕ, (∀ arrangement : Ball → Box, arrangements_condition1 arrangement → n = 7) :=
sorry

-- Define the arrangements for the second problem
def arrangements_condition2 (arrangement : Ball → Box) : Prop :=
  (arrangement Ball.ball1 ≠ Box.boxA) ∧
  (arrangement Ball.ball2 ≠ Box.boxB)

-- Define the proof statement for the second problem
theorem problem2 : ∃ n : ℕ, (∀ arrangement : Ball → Box, arrangements_condition2 arrangement → n = 36) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1312_131237


namespace NUMINAMATH_GPT_part1_minimum_b_over_a_l1312_131254

noncomputable def f (x a : ℝ) : ℝ := Real.log x - a * x

-- Prove part 1
theorem part1 (x : ℝ) : (0 < x ∧ x < 1 → (f x 1 / (1/x - 1) > 0)) ∧ (1 < x → (f x 1 / (1/x - 1) < 0)) := sorry

-- Prove part 2
lemma part2 (a b : ℝ) (h : ∀ x > 0, f x a ≤ b - a) (ha : a ≠ 0) : ∃ x > 0, f x a = b - a := sorry

theorem minimum_b_over_a (a : ℝ) (ha : a ≠ 0) (h : ∀ x > 0, f x a ≤ b - a) : b/a ≥ 0 := sorry

end NUMINAMATH_GPT_part1_minimum_b_over_a_l1312_131254


namespace NUMINAMATH_GPT_hyperbola_asymptote_passing_through_point_l1312_131215

theorem hyperbola_asymptote_passing_through_point (a : ℝ) (h_pos : a > 0) :
  (∃ m : ℝ, ∃ b : ℝ, ∀ x y : ℝ, y = m * x + b ∧ (x, y) = (2, 1) ∧ m = 2 / a) → a = 4 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_passing_through_point_l1312_131215


namespace NUMINAMATH_GPT_compare_times_l1312_131260

variable {v : ℝ} (h_v_pos : 0 < v)

/-- 
  Jones covered a distance of 80 miles on his first trip at speed v.
  On a later trip, he traveled 360 miles at four times his original speed.
  Prove that his new time is (9/8) times his original time.
-/
theorem compare_times :
  let t1 := 80 / v
  let t2 := 360 / (4 * v)
  t2 = (9 / 8) * t1 :=
by
  sorry

end NUMINAMATH_GPT_compare_times_l1312_131260


namespace NUMINAMATH_GPT_tan_ratio_alpha_beta_l1312_131244

theorem tan_ratio_alpha_beta 
  (α β : ℝ) 
  (h1 : Real.sin (α + β) = 1 / 5) 
  (h2 : Real.sin (α - β) = 3 / 5) : 
  Real.tan α / Real.tan β = -1 :=
sorry

end NUMINAMATH_GPT_tan_ratio_alpha_beta_l1312_131244


namespace NUMINAMATH_GPT_tan_cot_theta_l1312_131273

theorem tan_cot_theta 
  (θ : ℝ) 
  (h1 : Real.sin θ + Real.cos θ = (Real.sqrt 2) / 3) 
  (h2 : Real.pi / 2 < θ ∧ θ < Real.pi) : 
  Real.tan θ - (1 / Real.tan θ) = - (8 * Real.sqrt 2) / 7 := 
sorry

end NUMINAMATH_GPT_tan_cot_theta_l1312_131273


namespace NUMINAMATH_GPT_set_representation_l1312_131225

theorem set_representation :
  {p : ℕ × ℕ | 2 * p.1 + 3 * p.2 = 16} = {(2, 4), (5, 2), (8, 0)} :=
by
  sorry

end NUMINAMATH_GPT_set_representation_l1312_131225


namespace NUMINAMATH_GPT_opposite_of_neg3_l1312_131257

theorem opposite_of_neg3 : -(-3) = 3 := 
by 
  sorry

end NUMINAMATH_GPT_opposite_of_neg3_l1312_131257


namespace NUMINAMATH_GPT_find_x2_plus_y2_l1312_131266

theorem find_x2_plus_y2
  (x y : ℝ)
  (h1 : x * y = 8)
  (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^2 + y^2 = 5104 / 81 := 
by
  sorry

end NUMINAMATH_GPT_find_x2_plus_y2_l1312_131266


namespace NUMINAMATH_GPT_sqrt_pow_mul_l1312_131233

theorem sqrt_pow_mul (a b : ℝ) : (a = 3) → (b = 5) → (Real.sqrt (a^2 * b^6) = 375) :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end NUMINAMATH_GPT_sqrt_pow_mul_l1312_131233


namespace NUMINAMATH_GPT_sasha_salt_factor_l1312_131281

theorem sasha_salt_factor (x y : ℝ) : 
  (y = 2 * x) →
  (x + y = 2 * x + y / 2) →
  (3 * x / (2 * x) = 1.5) :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_sasha_salt_factor_l1312_131281


namespace NUMINAMATH_GPT_units_digit_42_3_plus_27_2_l1312_131277

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_42_3_plus_27_2 : units_digit (42^3 + 27^2) = 7 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_42_3_plus_27_2_l1312_131277


namespace NUMINAMATH_GPT_ab_div_c_eq_2_l1312_131276

variable (a b c : ℝ)

def condition1 (a b c : ℝ) : Prop := a * b - c = 3
def condition2 (a b c : ℝ) : Prop := a * b * c = 18

theorem ab_div_c_eq_2 (h1 : condition1 a b c) (h2 : condition2 a b c) : a * b / c = 2 :=
by sorry

end NUMINAMATH_GPT_ab_div_c_eq_2_l1312_131276


namespace NUMINAMATH_GPT_correct_car_selection_l1312_131295

-- Define the production volumes
def production_emgrand : ℕ := 1600
def production_king_kong : ℕ := 6000
def production_freedom_ship : ℕ := 2000

-- Define the total number of cars produced
def total_production : ℕ := production_emgrand + production_king_kong + production_freedom_ship

-- Define the number of cars selected for inspection
def cars_selected_for_inspection : ℕ := 48

-- Calculate the sampling ratio
def sampling_ratio : ℚ := cars_selected_for_inspection / total_production

-- Define the expected number of cars to be selected from each model using the sampling ratio
def cars_selected_emgrand : ℚ := sampling_ratio * production_emgrand
def cars_selected_king_kong : ℚ := sampling_ratio * production_king_kong
def cars_selected_freedom_ship : ℚ := sampling_ratio * production_freedom_ship

theorem correct_car_selection :
  cars_selected_emgrand = 8 ∧ cars_selected_king_kong = 30 ∧ cars_selected_freedom_ship = 10 := by
  sorry

end NUMINAMATH_GPT_correct_car_selection_l1312_131295


namespace NUMINAMATH_GPT_part1_part2_l1312_131294

variable (a : ℝ)

-- Proposition A
def propA (a : ℝ) := ∀ x : ℝ, ¬ (x^2 + (2*a-1)*x + a^2 ≤ 0)

-- Proposition B
def propB (a : ℝ) := 0 < a^2 - 1 ∧ a^2 - 1 < 1

theorem part1 (ha : propA a ∨ propB a) : 
  (-Real.sqrt 2 < a ∧ a < -1) ∨ (a > 1/4) :=
  sorry

theorem part2 (ha : ¬ propA a) (hb : propB a) : 
  (-Real.sqrt 2 < a ∧ a < -1) → (a^3 + 1 < a^2 + a) :=
  sorry

end NUMINAMATH_GPT_part1_part2_l1312_131294


namespace NUMINAMATH_GPT_triangle_properties_l1312_131289

theorem triangle_properties (a b c : ℝ) (h1 : a / b = 5 / 12) (h2 : b / c = 12 / 13) (h3 : a + b + c = 60) :
  (a^2 + b^2 = c^2) ∧ ((1 / 2) * a * b > 100) :=
by
  sorry

end NUMINAMATH_GPT_triangle_properties_l1312_131289


namespace NUMINAMATH_GPT_minimum_value_l1312_131201

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 / x + 1 / y = 1) : 3 * x + 4 * y ≥ 25 :=
sorry

end NUMINAMATH_GPT_minimum_value_l1312_131201


namespace NUMINAMATH_GPT_quadratic_eq1_solution_quadratic_eq2_solution_l1312_131298

-- Define the first problem and its conditions
theorem quadratic_eq1_solution :
  ∀ x : ℝ, 4 * x^2 + x - (1 / 2) = 0 ↔ (x = -1 / 2 ∨ x = 1 / 4) :=
by
  -- The proof is omitted
  sorry

-- Define the second problem and its conditions
theorem quadratic_eq2_solution :
  ∀ y : ℝ, (y - 2) * (y + 3) = 6 ↔ (y = -4 ∨ y = 3) :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_quadratic_eq1_solution_quadratic_eq2_solution_l1312_131298


namespace NUMINAMATH_GPT_determine_x_l1312_131200

noncomputable def proof_problem (x : ℝ) (y : ℝ) : Prop :=
  y > 0 → 2 * (x * y^2 + x^2 * y + 2 * y^2 + 2 * x * y) / (x + y) > 3 * x^2 * y

theorem determine_x (x : ℝ) : 
  (∀ (y : ℝ), y > 0 → proof_problem x y) ↔ 0 ≤ x ∧ x < (1 + Real.sqrt 13) / 3 := 
sorry

end NUMINAMATH_GPT_determine_x_l1312_131200


namespace NUMINAMATH_GPT_expr_value_l1312_131210

variable (a : ℝ)
variable (h : a^2 - 3 * a - 1011 = 0)

theorem expr_value : 2 * a^2 - 6 * a + 1 = 2023 :=
by
  -- insert proof here
  sorry

end NUMINAMATH_GPT_expr_value_l1312_131210


namespace NUMINAMATH_GPT_region_of_inequality_l1312_131236

theorem region_of_inequality (x y : ℝ) : (x + y - 6 < 0) → y < -x + 6 := by
  sorry

end NUMINAMATH_GPT_region_of_inequality_l1312_131236


namespace NUMINAMATH_GPT_weight_shaina_receives_l1312_131213

namespace ChocolateProblem

-- Definitions based on conditions
def total_chocolate : ℚ := 60 / 7
def piles : ℚ := 5
def weight_per_pile : ℚ := total_chocolate / piles
def shaina_piles : ℚ := 2

-- Proposition to represent the question and correct answer
theorem weight_shaina_receives : 
  (weight_per_pile * shaina_piles) = 24 / 7 := 
by
  sorry

end ChocolateProblem

end NUMINAMATH_GPT_weight_shaina_receives_l1312_131213


namespace NUMINAMATH_GPT_third_intermission_served_l1312_131256

def total_served : ℚ :=  0.9166666666666666
def first_intermission : ℚ := 0.25
def second_intermission : ℚ := 0.4166666666666667

theorem third_intermission_served : first_intermission + second_intermission ≤ total_served →
  (total_served - (first_intermission + second_intermission)) = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_third_intermission_served_l1312_131256


namespace NUMINAMATH_GPT_quarters_spent_l1312_131227

variable (q_initial q_left q_spent : ℕ)

theorem quarters_spent (h1 : q_initial = 11) (h2 : q_left = 7) : q_spent = q_initial - q_left ∧ q_spent = 4 :=
by
  sorry

end NUMINAMATH_GPT_quarters_spent_l1312_131227


namespace NUMINAMATH_GPT_probability_of_hitting_10_or_9_probability_of_hitting_at_least_7_probability_of_hitting_less_than_8_l1312_131290

-- Definitions of the probabilities
def P_A := 0.24
def P_B := 0.28
def P_C := 0.19
def P_D := 0.16
def P_E := 0.13

-- Prove that the probability of hitting the 10 or 9 rings is 0.52
theorem probability_of_hitting_10_or_9 : P_A + P_B = 0.52 :=
  by sorry

-- Prove that the probability of hitting at least the 7 ring is 0.87
theorem probability_of_hitting_at_least_7 : P_A + P_B + P_C + P_D = 0.87 :=
  by sorry

-- Prove that the probability of hitting less than 8 rings is 0.29
theorem probability_of_hitting_less_than_8 : P_D + P_E = 0.29 :=
  by sorry

end NUMINAMATH_GPT_probability_of_hitting_10_or_9_probability_of_hitting_at_least_7_probability_of_hitting_less_than_8_l1312_131290


namespace NUMINAMATH_GPT_blue_whale_tongue_weight_in_tons_l1312_131255

-- Define the conditions
def weight_of_tongue_pounds : ℕ := 6000
def pounds_per_ton : ℕ := 2000

-- Define the theorem stating the question and its answer
theorem blue_whale_tongue_weight_in_tons :
  (weight_of_tongue_pounds / pounds_per_ton) = 3 :=
by sorry

end NUMINAMATH_GPT_blue_whale_tongue_weight_in_tons_l1312_131255


namespace NUMINAMATH_GPT_seventh_term_in_geometric_sequence_l1312_131265

theorem seventh_term_in_geometric_sequence :
  ∃ r, (4 * r^8 = 2097152) ∧ (4 * r^6 = 1048576) :=
by
  sorry

end NUMINAMATH_GPT_seventh_term_in_geometric_sequence_l1312_131265


namespace NUMINAMATH_GPT_simplify_expression_l1312_131245

noncomputable def y := 
  Real.cos (2 * Real.pi / 15) + 
  Real.cos (4 * Real.pi / 15) + 
  Real.cos (8 * Real.pi / 15) + 
  Real.cos (14 * Real.pi / 15)

theorem simplify_expression : 
  y = (-1 + Real.sqrt 61) / 4 := 
sorry

end NUMINAMATH_GPT_simplify_expression_l1312_131245


namespace NUMINAMATH_GPT_mika_initial_stickers_l1312_131242

theorem mika_initial_stickers :
  let store_stickers := 26.0
  let birthday_stickers := 20.0 
  let sister_stickers := 6.0 
  let mother_stickers := 58.0 
  let total_stickers := 130.0 
  ∃ x : Real, x + store_stickers + birthday_stickers + sister_stickers + mother_stickers = total_stickers ∧ x = 20.0 := 
by 
  sorry

end NUMINAMATH_GPT_mika_initial_stickers_l1312_131242


namespace NUMINAMATH_GPT_problem1_proof_problem2_proof_l1312_131285

noncomputable def problem1_statement : Prop :=
  (2 * Real.sin (Real.pi / 6) - Real.sin (Real.pi / 4) * Real.cos (Real.pi / 4) = 1 / 2)

noncomputable def problem2_statement : Prop :=
  ((-1)^2023 + 2 * Real.sin (Real.pi / 4) - Real.cos (Real.pi / 6) + Real.sin (Real.pi / 3) + Real.tan (Real.pi / 3)^2 = 2 + Real.sqrt 2)

theorem problem1_proof : problem1_statement :=
by
  sorry

theorem problem2_proof : problem2_statement :=
by
  sorry

end NUMINAMATH_GPT_problem1_proof_problem2_proof_l1312_131285


namespace NUMINAMATH_GPT_sum_of_numerator_and_denominator_of_repeating_decimal_l1312_131246

theorem sum_of_numerator_and_denominator_of_repeating_decimal (x : ℚ) (h : x = 34 / 99) : (x.den + x.num : ℤ) = 133 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numerator_and_denominator_of_repeating_decimal_l1312_131246


namespace NUMINAMATH_GPT_robin_made_more_cupcakes_l1312_131287

theorem robin_made_more_cupcakes (initial final sold made: ℕ)
  (h1 : initial = 42)
  (h2 : sold = 22)
  (h3 : final = 59)
  (h4 : initial - sold + made = final) :
  made = 39 :=
  sorry

end NUMINAMATH_GPT_robin_made_more_cupcakes_l1312_131287


namespace NUMINAMATH_GPT_divisibility_of_n_l1312_131218

theorem divisibility_of_n
  (n : ℕ) (n_gt_1 : n > 1)
  (h : n ∣ (6^n - 1)) : 5 ∣ n :=
by
  sorry

end NUMINAMATH_GPT_divisibility_of_n_l1312_131218


namespace NUMINAMATH_GPT_solve_quadratic_eq_solve_cubic_eq_l1312_131230

-- Problem 1: 4x^2 - 9 = 0 implies x = ± 3/2
theorem solve_quadratic_eq (x : ℝ) : 4 * x^2 - 9 = 0 ↔ x = 3/2 ∨ x = -3/2 :=
by sorry

-- Problem 2: 64 * (x + 1)^3 = -125 implies x = -9/4
theorem solve_cubic_eq (x : ℝ) : 64 * (x + 1)^3 = -125 ↔ x = -9/4 :=
by sorry

end NUMINAMATH_GPT_solve_quadratic_eq_solve_cubic_eq_l1312_131230


namespace NUMINAMATH_GPT_race_track_cost_l1312_131297

def toy_car_cost : ℝ := 0.95
def num_toy_cars : ℕ := 4
def total_money : ℝ := 17.80
def money_left : ℝ := 8.00

theorem race_track_cost :
  total_money - num_toy_cars * toy_car_cost - money_left = 6.00 :=
by
  sorry

end NUMINAMATH_GPT_race_track_cost_l1312_131297


namespace NUMINAMATH_GPT_min_f_in_interval_l1312_131240

open Real

noncomputable def f (ω x : ℝ) : ℝ := sin (ω * x) - 2 * sqrt 3 * sin (ω * x / 2) ^ 2 + sqrt 3

theorem min_f_in_interval (ω : ℝ) (hω : ω > 0) :
  (∀ x, 0 <= x ∧ x <= π / 2 → f 1 x >= f 1 (π / 3)) :=
by sorry

end NUMINAMATH_GPT_min_f_in_interval_l1312_131240


namespace NUMINAMATH_GPT_oranges_per_pack_correct_l1312_131207

-- Definitions for the conditions.
def num_trees : Nat := 10
def oranges_per_tree_per_day : Nat := 12
def price_per_pack : Nat := 2
def total_earnings : Nat := 840
def weeks : Nat := 3
def days_per_week : Nat := 7

-- Theorem statement:
theorem oranges_per_pack_correct :
  let oranges_per_day := num_trees * oranges_per_tree_per_day
  let total_days := weeks * days_per_week
  let total_oranges := oranges_per_day * total_days
  let num_packs := total_earnings / price_per_pack
  total_oranges / num_packs = 6 :=
by
  sorry

end NUMINAMATH_GPT_oranges_per_pack_correct_l1312_131207


namespace NUMINAMATH_GPT_functional_equation_solution_l1312_131275

open Nat

theorem functional_equation_solution :
  (∀ (f : ℕ → ℕ), 
    (∀ (x y : ℕ), 0 ≤ y + f x - (Nat.iterate f (f y) x) ∧ (y + f x - (Nat.iterate f (f y) x) ≤ 1)) →
    (∀ n, f n = n + 1)) :=
by
  intro f h
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l1312_131275


namespace NUMINAMATH_GPT_S_10_eq_110_l1312_131208

-- Conditions
def a (n : ℕ) : ℕ := sorry  -- Assuming general term definition of arithmetic sequence
def S (n : ℕ) : ℕ := sorry  -- Assuming sum definition of arithmetic sequence

axiom a_3_eq_16 : a 3 = 16
axiom S_20_eq_20 : S 20 = 20

-- Prove
theorem S_10_eq_110 : S 10 = 110 :=
  by
  sorry

end NUMINAMATH_GPT_S_10_eq_110_l1312_131208


namespace NUMINAMATH_GPT_find_m_given_slope_condition_l1312_131258

variable (m : ℝ)

theorem find_m_given_slope_condition
  (h : (m - 4) / (3 - 2) = 1) : m = 5 :=
sorry

end NUMINAMATH_GPT_find_m_given_slope_condition_l1312_131258


namespace NUMINAMATH_GPT_factorize_expression_l1312_131222

theorem factorize_expression (a m n : ℝ) : a * m^2 - 2 * a * m * n + a * n^2 = a * (m - n)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1312_131222


namespace NUMINAMATH_GPT_integer_values_of_x_for_positive_star_l1312_131253

-- Definition of the operation star
def star (a b : ℕ) : ℚ := (a^2 : ℕ) / b

-- Problem statement
theorem integer_values_of_x_for_positive_star :
  ∃ (count : ℕ), count = 9 ∧ (∀ x : ℕ, (10^2 % x = 0) → (∃ n : ℕ, star 10 x = n)) :=
sorry

end NUMINAMATH_GPT_integer_values_of_x_for_positive_star_l1312_131253


namespace NUMINAMATH_GPT_problem_solution_l1312_131272

def f (x y : ℝ) : ℝ :=
  (x - y) * x * y * (x + y) * (2 * x^2 - 5 * x * y + 2 * y^2)

theorem problem_solution :
  (∀ x y : ℝ, f x y + f y x = 0) ∧
  (∀ x y : ℝ, f x (x + y) + f y (x + y) = 0) :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1312_131272


namespace NUMINAMATH_GPT_simplify_sqrt_product_l1312_131251

theorem simplify_sqrt_product (x : ℝ) : 
  (Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x)) = 120 * x * Real.sqrt (2 * x) := 
by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_product_l1312_131251


namespace NUMINAMATH_GPT_Dave_needs_31_gallons_l1312_131214

noncomputable def numberOfGallons (numberOfTanks : ℕ) (height : ℝ) (diameter : ℝ) (coveragePerGallon : ℝ) : ℕ :=
  let radius := diameter / 2
  let lateral_surface_area := 2 * Real.pi * radius * height
  let total_surface_area := lateral_surface_area * numberOfTanks
  let gallons_needed := total_surface_area / coveragePerGallon
  Nat.ceil gallons_needed

theorem Dave_needs_31_gallons :
  numberOfGallons 20 24 8 400 = 31 :=
by
  sorry

end NUMINAMATH_GPT_Dave_needs_31_gallons_l1312_131214


namespace NUMINAMATH_GPT_number_of_sides_of_regular_polygon_l1312_131283

theorem number_of_sides_of_regular_polygon (h: ∀ (n: ℕ), (180 * (n - 2) / n) = 135) : ∃ n, n = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_sides_of_regular_polygon_l1312_131283


namespace NUMINAMATH_GPT_residue_5_pow_1234_mod_13_l1312_131234

theorem residue_5_pow_1234_mod_13 : ∃ k : ℤ, 5^1234 = 13 * k + 12 :=
by
  sorry

end NUMINAMATH_GPT_residue_5_pow_1234_mod_13_l1312_131234


namespace NUMINAMATH_GPT_minor_axis_length_l1312_131202

theorem minor_axis_length (h : ∀ x y : ℝ, x^2 / 4 + y^2 / 36 = 1) : 
  ∃ b : ℝ, b = 2 ∧ 2 * b = 4 :=
by
  sorry

end NUMINAMATH_GPT_minor_axis_length_l1312_131202


namespace NUMINAMATH_GPT_radius_of_arch_bridge_l1312_131293

theorem radius_of_arch_bridge :
  ∀ (AB CD AD r : ℝ),
    AB = 12 →
    CD = 4 →
    AD = AB / 2 →
    r^2 = AD^2 + (r - CD)^2 →
    r = 6.5 :=
by
  intros AB CD AD r hAB hCD hAD h_eq
  sorry

end NUMINAMATH_GPT_radius_of_arch_bridge_l1312_131293


namespace NUMINAMATH_GPT_ribbon_initial_amount_l1312_131280

theorem ribbon_initial_amount (x : ℕ) (gift_count : ℕ) (ribbon_per_gift : ℕ) (ribbon_left : ℕ)
  (H1 : ribbon_per_gift = 2) (H2 : gift_count = 6) (H3 : ribbon_left = 6)
  (H4 : x = gift_count * ribbon_per_gift + ribbon_left) : x = 18 :=
by
  rw [H1, H2, H3] at H4
  exact H4

end NUMINAMATH_GPT_ribbon_initial_amount_l1312_131280


namespace NUMINAMATH_GPT_convert_3241_quinary_to_septenary_l1312_131235

/-- Convert quinary number 3241_(5) to septenary number, yielding 1205_(7). -/
theorem convert_3241_quinary_to_septenary : 
  let quinary := 3 * 5^3 + 2 * 5^2 + 4 * 5^1 + 1 * 5^0
  let septenary := 1 * 7^3 + 2 * 7^2 + 0 * 7^1 + 5 * 7^0
  quinary = 446 → septenary = 1205 :=
by
  intros
  -- Quinary to Decimal
  have h₁ : 3 * 5^3 + 2 * 5^2 + 4 * 5^1 + 1 * 5^0 = 446 := by norm_num
  -- Decimal to Septenary
  have h₂ : 446 = 1 * 7^3 + 2 * 7^2 + 0 * 7^1 + 5 * 7^0 := by norm_num
  exact sorry

end NUMINAMATH_GPT_convert_3241_quinary_to_septenary_l1312_131235


namespace NUMINAMATH_GPT_math_problem_l1312_131231

variables {x y z a b c : ℝ}

theorem math_problem
  (h₁ : x / a + y / b + z / c = 4)
  (h₂ : a / x + b / y + c / z = 2) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 12 :=
sorry

end NUMINAMATH_GPT_math_problem_l1312_131231


namespace NUMINAMATH_GPT_road_construction_days_l1312_131262

theorem road_construction_days
  (length_of_road : ℝ)
  (initial_men : ℕ)
  (completed_length : ℝ)
  (completed_days : ℕ)
  (extra_men : ℕ)
  (initial_days : ℕ)
  (remaining_length : ℝ)
  (remaining_days : ℕ)
  (total_men : ℕ) :
  length_of_road = 15 →
  initial_men = 30 →
  completed_length = 2.5 →
  completed_days = 100 →
  extra_men = 45 →
  initial_days = initial_days →
  remaining_length = length_of_road - completed_length →
  remaining_days = initial_days - completed_days →
  total_men = initial_men + extra_men →
  initial_days = 700 :=
by
  intros
  sorry

end NUMINAMATH_GPT_road_construction_days_l1312_131262


namespace NUMINAMATH_GPT_find_values_of_c_x1_x2_l1312_131268

theorem find_values_of_c_x1_x2 (x₁ x₂ c : ℝ)
    (h1 : x₁ + x₂ = -2)
    (h2 : x₁ * x₂ = c)
    (h3 : x₁^2 + x₂^2 = c^2 - 2 * c) :
    c = -2 ∧ x₁ = -1 + Real.sqrt 3 ∧ x₂ = -1 - Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_find_values_of_c_x1_x2_l1312_131268


namespace NUMINAMATH_GPT_minimum_button_presses_l1312_131241

theorem minimum_button_presses :
  ∃ (r y g : ℕ), 
    2 * y - r = 3 ∧ 2 * g - y = 3 ∧ r + y + g = 9 :=
by sorry

end NUMINAMATH_GPT_minimum_button_presses_l1312_131241


namespace NUMINAMATH_GPT_arithmetic_seq_sum_2013_l1312_131243

noncomputable def a1 : ℤ := -2013
noncomputable def S (n d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem arithmetic_seq_sum_2013 :
  ∃ d : ℤ, (S 12 d / 12 - S 10 d / 10 = 2) → S 2013 d = -2013 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_2013_l1312_131243


namespace NUMINAMATH_GPT_exists_adj_diff_gt_3_max_min_adj_diff_l1312_131209
-- Import needed libraries

-- Definition of the given problem and statement of the parts (a) and (b)

-- Part (a)
theorem exists_adj_diff_gt_3 (arrangement : Fin 18 → Fin 18) (adj : Fin 18 → Fin 18 → Prop) :
  (∀ i j : Fin 18, adj i j → i ≠ j) →
  (∃ i j : Fin 18, adj i j ∧ |arrangement i - arrangement j| > 3) :=
sorry

-- Part (b)
theorem max_min_adj_diff (arrangement : Fin 18 → Fin 18) (adj : Fin 18 → Fin 18 → Prop) :
  (∀ i j : Fin 18, adj i j → i ≠ j) →
  (∀ i j : Fin 18, adj i j → |arrangement i - arrangement j| ≥ 6) :=
sorry

end NUMINAMATH_GPT_exists_adj_diff_gt_3_max_min_adj_diff_l1312_131209


namespace NUMINAMATH_GPT_calculate_f_g_f_l1312_131269

def f (x : ℤ) : ℤ := 5 * x + 5
def g (x : ℤ) : ℤ := 6 * x + 5

theorem calculate_f_g_f : f (g (f 3)) = 630 := by
  sorry

end NUMINAMATH_GPT_calculate_f_g_f_l1312_131269


namespace NUMINAMATH_GPT_find_a_l1312_131252

theorem find_a (a b c d : ℕ) (h1 : a + b = d) (h2 : b + c = 6) (h3 : c + d = 7) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1312_131252


namespace NUMINAMATH_GPT_studentC_spending_l1312_131226

-- Definitions based on the problem conditions

-- Prices of Type A and Type B notebooks, respectively
variables (x y : ℝ)

-- Number of each type of notebook bought by Student A
def studentA : Prop := x + y = 3

-- Number of Type A notebooks bought by Student B
variables (a : ℕ)

-- Total cost and number of notebooks bought by Student B
def studentB : Prop := (x * a + y * (8 - a) = 11)

-- Constraints on the number of Type A and B notebooks bought by Student C
def studentC_notebooks : Prop := ∃ b : ℕ, b = 8 - a ∧ b = a

-- The total amount spent by Student C
def studentC_cost : ℝ := (8 - a) * x + a * y

-- The statement asserting the cost is 13 yuan
theorem studentC_spending (x y : ℝ) (a : ℕ) (hA : studentA x y) (hB : studentB x y a) (hC : studentC_notebooks a) : studentC_cost x y a = 13 := sorry

end NUMINAMATH_GPT_studentC_spending_l1312_131226


namespace NUMINAMATH_GPT_equal_angles_not_necessarily_vertical_l1312_131220

-- Define what it means for angles to be vertical
def is_vertical_angle (a b : ℝ) : Prop :=
∃ l1 l2 : ℝ, a = 180 - b ∧ (l1 + l2 == 180 ∨ l1 == 0 ∨ l2 == 0)

-- Define what it means for angles to be equal
def are_equal_angles (a b : ℝ) : Prop := a = b

-- Proposition to be proved
theorem equal_angles_not_necessarily_vertical (a b : ℝ) (h : are_equal_angles a b) : ¬ is_vertical_angle a b :=
by
  sorry

end NUMINAMATH_GPT_equal_angles_not_necessarily_vertical_l1312_131220


namespace NUMINAMATH_GPT_exists_infinitely_many_gcd_condition_l1312_131212

theorem exists_infinitely_many_gcd_condition (a : ℕ → ℕ) (h : ∀ n : ℕ, ∃ m : ℕ, a m = n) :
  ∃ᶠ i in at_top, Nat.gcd (a i) (a (i + 1)) ≤ (3 * i) / 4 :=
sorry

end NUMINAMATH_GPT_exists_infinitely_many_gcd_condition_l1312_131212


namespace NUMINAMATH_GPT_sequence_odd_for_all_n_greater_than_1_l1312_131224

theorem sequence_odd_for_all_n_greater_than_1 (a : ℕ → ℤ) :
  (a 1 = 2) →
  (a 2 = 7) →
  (∀ n, 2 ≤ n → (-1/2 : ℚ) < (a (n + 1) : ℚ) - ((a n : ℚ) ^ 2) / (a (n - 1) : ℚ) ∧ (a (n + 1) : ℚ) - ((a n : ℚ) ^ 2) / (a (n - 1) : ℚ) ≤ (1/2 : ℚ)) →
  ∀ n, 1 < n → Odd (a n) := 
sorry

end NUMINAMATH_GPT_sequence_odd_for_all_n_greater_than_1_l1312_131224


namespace NUMINAMATH_GPT_part1_part2_l1312_131219

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)
noncomputable def g (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem part1 (x : ℝ) : (f x)^2 - (g x)^2 = -4 :=
by sorry

theorem part2 (x y : ℝ) (h1 : f x * f y = 4) (h2 : g x * g y = 8) : 
  g (x + y) / g (x - y) = 3 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1312_131219


namespace NUMINAMATH_GPT_gcd_45_75_l1312_131217

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end NUMINAMATH_GPT_gcd_45_75_l1312_131217


namespace NUMINAMATH_GPT_find_b_and_c_l1312_131263

variable (U : Set ℝ) -- Define the universal set U
variable (A : Set ℝ) -- Define the set A
variables (b c : ℝ) -- Variables for coefficients

-- Conditions that U = {2, 3, 5} and A = { x | x^2 + bx + c = 0 }
def cond_universal_set := U = {2, 3, 5}
def cond_set_A := A = { x | x^2 + b * x + c = 0 }

-- Condition for the complement of A w.r.t U being {2}
def cond_complement := (U \ A) = {2}

-- The statement to be proved
theorem find_b_and_c : 
  cond_universal_set U →
  cond_set_A A b c →
  cond_complement U A →
  b = -8 ∧ c = 15 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_b_and_c_l1312_131263


namespace NUMINAMATH_GPT_find_a_l1312_131249

theorem find_a 
  (a b c : ℚ) 
  (h1 : a + b = c) 
  (h2 : b + c + 2 * b = 11) 
  (h3 : c = 7) :
  a = 17 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1312_131249


namespace NUMINAMATH_GPT_rabbit_speed_correct_l1312_131267

-- Define the conditions given in the problem
def rabbit_speed (x : ℝ) : Prop :=
2 * (2 * x + 4) = 188

-- State the main theorem using the defined conditions
theorem rabbit_speed_correct : ∃ x : ℝ, rabbit_speed x ∧ x = 45 :=
by
  sorry

end NUMINAMATH_GPT_rabbit_speed_correct_l1312_131267
