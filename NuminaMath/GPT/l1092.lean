import Mathlib

namespace NUMINAMATH_GPT_fraction_of_raisins_in_mixture_l1092_109255

def cost_of_raisins (R : ℝ) := 3 * R
def cost_of_nuts (R : ℝ) := 3 * (3 * R)
def total_cost (R : ℝ) := cost_of_raisins R + cost_of_nuts R

theorem fraction_of_raisins_in_mixture (R : ℝ) (hR_pos : R > 0) : 
  cost_of_raisins R / total_cost R = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_raisins_in_mixture_l1092_109255


namespace NUMINAMATH_GPT_total_number_of_balls_in_fish_tank_l1092_109230

-- Definitions as per conditions
def num_goldfish := 3
def num_platyfish := 10
def red_balls_per_goldfish := 10
def white_balls_per_platyfish := 5

-- Theorem statement
theorem total_number_of_balls_in_fish_tank : 
  (num_goldfish * red_balls_per_goldfish + num_platyfish * white_balls_per_platyfish) = 80 := 
by
  sorry

end NUMINAMATH_GPT_total_number_of_balls_in_fish_tank_l1092_109230


namespace NUMINAMATH_GPT_bag_of_chips_weight_l1092_109261

theorem bag_of_chips_weight (c : ℕ) : 
  (∀ (t : ℕ), t = 9) → 
  (∀ (b : ℕ), b = 6) → 
  (∀ (x : ℕ), x = 4 * 6) → 
  (21 * 16 = 336) →
  (336 - 24 * 9 = 6 * c) → 
  c = 20 :=
by
  intros ht hb hx h_weight_total h_weight_chips
  sorry

end NUMINAMATH_GPT_bag_of_chips_weight_l1092_109261


namespace NUMINAMATH_GPT_interval_of_decrease_l1092_109200

theorem interval_of_decrease (f : ℝ → ℝ) (h_decreasing : ∀ x y : ℝ, x ≤ y → f y ≤ f x) :
  ∀ x0 : ℝ, ∀ x1 : ℝ, x0 ≥ 3 → x0 ≤ x1 → f (x1 - 3) ≤ f (x0 - 3) := sorry

end NUMINAMATH_GPT_interval_of_decrease_l1092_109200


namespace NUMINAMATH_GPT_shared_property_l1092_109293

-- Definitions of the shapes
structure Parallelogram where
  sides_equal    : Bool -- Parallelograms have opposite sides equal but not necessarily all four.

structure Rectangle where
  sides_equal    : Bool -- Rectangles have opposite sides equal.
  diagonals_equal: Bool

structure Rhombus where
  sides_equal: Bool -- Rhombuses have all sides equal.
  diagonals_perpendicular: Bool -- Diagonals of a rhombus are perpendicular.

structure Square where
  sides_equal: Bool -- Squares have all sides equal.
  diagonals_perpendicular: Bool -- Diagonals of a square are perpendicular.
  diagonals_equal: Bool -- Diagonals of a square are equal in length.

-- Definitions of properties
def all_sides_equal (p1 p2 p3 p4 : Parallelogram) := p1.sides_equal ∧ p2.sides_equal ∧ p3.sides_equal ∧ p4.sides_equal
def diagonals_equal (r1 r2 r3 : Rectangle) (s1 s2 : Square) := r1.diagonals_equal ∧ r2.diagonals_equal ∧ s1.diagonals_equal ∧ s2.diagonals_equal
def diagonals_perpendicular (r1 : Rhombus) (s1 s2 : Square) := r1.diagonals_perpendicular ∧ s1.diagonals_perpendicular ∧ s2.diagonals_perpendicular
def diagonals_bisect_each_other (p1 p2 p3 p4 : Parallelogram) (r1 : Rectangle) (r2 : Rhombus) (s1 s2 : Square) := True -- All these shapes have diagonals that bisect each other.

-- The statement we need to prove
theorem shared_property (p1 p2 p3 p4 : Parallelogram) (r1 r2 : Rectangle) (r3 : Rhombus) (s1 s2 : Square) : 
  (diagonals_bisect_each_other p1 p2 p3 p4 r1 r3 s1 s2) :=
by
  sorry

end NUMINAMATH_GPT_shared_property_l1092_109293


namespace NUMINAMATH_GPT_factor_quadratic_l1092_109208

theorem factor_quadratic (m p : ℝ) (h : (m - 8) ∣ (m^2 - p * m - 24)) : p = 5 :=
sorry

end NUMINAMATH_GPT_factor_quadratic_l1092_109208


namespace NUMINAMATH_GPT_find_s_of_2_l1092_109258

-- Define t and s as per the given conditions
def t (x : ℚ) : ℚ := 4 * x - 9
def s (x : ℚ) : ℚ := x^2 + 4 * x - 5

-- The theorem that we need to prove
theorem find_s_of_2 : s 2 = 217 / 16 := by
  sorry

end NUMINAMATH_GPT_find_s_of_2_l1092_109258


namespace NUMINAMATH_GPT_a_9_value_l1092_109270

-- Define the sequence and its sum of the first n terms
def S (n : ℕ) : ℕ := n^2

-- Define the terms of the sequence
def a (n : ℕ) : ℕ := if n = 1 then S 1 else S n - S (n - 1)

-- The main statement to be proved
theorem a_9_value : a 9 = 17 :=
by
  sorry

end NUMINAMATH_GPT_a_9_value_l1092_109270


namespace NUMINAMATH_GPT_weighted_arithmetic_geometric_mean_l1092_109204
-- Importing required library

-- Definitions of the problem variables and conditions
variables (a b c : ℝ)

-- Non-negative constraints on the lengths of the line segments
variables (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)

-- Problem statement, we need to prove
theorem weighted_arithmetic_geometric_mean :
  0.2 * a + 0.3 * b + 0.5 * c ≥ (a * b * c)^(1/3) :=
sorry

end NUMINAMATH_GPT_weighted_arithmetic_geometric_mean_l1092_109204


namespace NUMINAMATH_GPT_hyperbola_focus_and_distance_l1092_109227

noncomputable def right_focus_of_hyperbola (a b : ℝ) : ℝ × ℝ := 
  (Real.sqrt (a^2 + b^2), 0)

noncomputable def distance_to_asymptote (a b : ℝ) : ℝ := 
  let c := Real.sqrt (a^2 + b^2)
  abs c / Real.sqrt (1 + (b/a)^2)

theorem hyperbola_focus_and_distance (a b : ℝ) (h₁ : a^2 = 6) (h₂ : b^2 = 3) :
  right_focus_of_hyperbola a b = (3, 0) ∧ distance_to_asymptote a b = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_focus_and_distance_l1092_109227


namespace NUMINAMATH_GPT_computation_problems_count_l1092_109266

theorem computation_problems_count : 
  ∃ (x y : ℕ), 3 * x + 5 * y = 110 ∧ x + y = 30 ∧ x = 20 :=
by
  sorry

end NUMINAMATH_GPT_computation_problems_count_l1092_109266


namespace NUMINAMATH_GPT_no_such_function_exists_l1092_109213

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), ∀ x : ℝ, f (Real.sin x) + f (Real.cos x) = Real.sin x :=
by
  sorry

end NUMINAMATH_GPT_no_such_function_exists_l1092_109213


namespace NUMINAMATH_GPT_check_ratio_l1092_109209

theorem check_ratio (initial_balance check_amount new_balance : ℕ) 
  (h1 : initial_balance = 150) (h2 : check_amount = 50) (h3 : new_balance = initial_balance + check_amount) :
  (check_amount : ℚ) / new_balance = 1 / 4 := 
by { 
  sorry 
}

end NUMINAMATH_GPT_check_ratio_l1092_109209


namespace NUMINAMATH_GPT_uncolored_vertex_not_original_hexagon_vertex_l1092_109229

theorem uncolored_vertex_not_original_hexagon_vertex
    (point_index : ℕ)
    (orig_hex_vertices : Finset ℕ) -- Assuming the vertices of the original hexagon are represented as a finite set of indices.
    (num_parts : ℕ := 1000) -- Each hexagon side is divided into 1000 parts
    (label : ℕ → Fin 3) -- A function labeling each point with 0, 1, or 2.
    (is_valid_labeling : ∀ (i j k : ℕ), label i ≠ label j ∧ label j ≠ label k ∧ label k ≠ label i) -- No duplicate labeling within a triangle.
    (is_single_uncolored : ∀ (p : ℕ), (p ∈ orig_hex_vertices ∨ ∃ (v : ℕ), v ∈ orig_hex_vertices ∧ p = v) → p ≠ point_index) -- Only one uncolored point
    : point_index ∉ orig_hex_vertices :=
by sorry

end NUMINAMATH_GPT_uncolored_vertex_not_original_hexagon_vertex_l1092_109229


namespace NUMINAMATH_GPT_edith_novel_count_l1092_109288

-- Definitions based on conditions
variables (N W : ℕ)

-- Conditions from the problem
def condition1 : Prop := N = W / 2
def condition2 : Prop := N + W = 240

-- Target statement
theorem edith_novel_count (N W : ℕ) (h1 : N = W / 2) (h2 : N + W = 240) : N = 80 :=
by
  sorry

end NUMINAMATH_GPT_edith_novel_count_l1092_109288


namespace NUMINAMATH_GPT_problem_sum_of_pairwise_prime_product_l1092_109274

theorem problem_sum_of_pairwise_prime_product:
  ∃ a b c d: ℕ, a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1 ∧
  a * b * c * d = 288000 ∧
  gcd a b = 1 ∧ gcd a c = 1 ∧ gcd a d = 1 ∧
  gcd b c = 1 ∧ gcd b d = 1 ∧ gcd c d = 1 ∧
  a + b + c + d = 390 :=
sorry

end NUMINAMATH_GPT_problem_sum_of_pairwise_prime_product_l1092_109274


namespace NUMINAMATH_GPT_arrangement_of_70616_l1092_109215

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangement_count (digits : List ℕ) : ℕ :=
  let count := digits.length
  let duplicates := List.length (List.filter (fun x => x = 6) digits)
  factorial count / factorial duplicates

theorem arrangement_of_70616 : arrangement_count [7, 0, 6, 6, 1] = 4 * 12 := by
  -- We need to prove that the number of ways to arrange the digits 7, 0, 6, 6, 1 without starting with 0 is 48
  sorry

end NUMINAMATH_GPT_arrangement_of_70616_l1092_109215


namespace NUMINAMATH_GPT_original_number_fraction_l1092_109233

theorem original_number_fraction (x : ℚ) (h : 1 + 1/x = 9/4) : x = 4/5 := by
  sorry

end NUMINAMATH_GPT_original_number_fraction_l1092_109233


namespace NUMINAMATH_GPT_find_A_and_evaluate_A_minus_B_l1092_109216

-- Given definitions
def B (x y : ℝ) : ℝ := 4 * x ^ 2 - 3 * y - 1
def result (x y : ℝ) : ℝ := 6 * x ^ 2 - y

-- Defining the polynomial A based on the first condition
def A (x y : ℝ) : ℝ := 2 * x ^ 2 + 2 * y + 1

-- The main theorem to be proven
theorem find_A_and_evaluate_A_minus_B :
  (∀ x y : ℝ, B x y + A x y = result x y) →
  (∀ x y : ℝ, |x - 1| * (y + 1) ^ 2 = 0 → A x y - B x y = -5) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_find_A_and_evaluate_A_minus_B_l1092_109216


namespace NUMINAMATH_GPT_coin_problem_exists_l1092_109245

theorem coin_problem_exists (n : ℕ) : 
  (∃ n, n % 8 = 6 ∧ n % 7 = 5 ∧ (∀ m, (m % 8 = 6 ∧ m % 7 = 5) → n ≤ m)) →
  (∃ n, (n % 8 = 6) ∧ (n % 7 = 5) ∧ (n % 9 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_coin_problem_exists_l1092_109245


namespace NUMINAMATH_GPT_roberto_outfit_combinations_l1092_109234

-- Define the components of the problem
def trousers_count : ℕ := 5
def shirts_count : ℕ := 7
def jackets_count : ℕ := 4
def disallowed_combinations : ℕ := 7

-- Define the requirements
theorem roberto_outfit_combinations :
  (trousers_count * shirts_count * jackets_count) - disallowed_combinations = 133 := by
  sorry

end NUMINAMATH_GPT_roberto_outfit_combinations_l1092_109234


namespace NUMINAMATH_GPT_fraction_age_28_to_32_l1092_109206

theorem fraction_age_28_to_32 (F : ℝ) (total_participants : ℝ) 
  (next_year_fraction_increase : ℝ) (next_year_fraction : ℝ) 
  (h1 : total_participants = 500)
  (h2 : next_year_fraction_increase = (1 / 8 : ℝ))
  (h3 : next_year_fraction = 0.5625) 
  (h4 : F + next_year_fraction_increase * F = next_year_fraction) :
  F = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_age_28_to_32_l1092_109206


namespace NUMINAMATH_GPT_height_of_table_l1092_109295

variable (h l w h3 : ℝ)

-- Conditions from the problem
def condition1 : Prop := h3 = 4
def configurationA : Prop := l + h - w = 50
def configurationB : Prop := w + h + h3 - l = 44

-- Statement to prove
theorem height_of_table (h l w h3 : ℝ) 
  (cond1 : condition1 h3)
  (confA : configurationA h l w)
  (confB : configurationB h l w h3) : 
  h = 45 := 
by 
  sorry

end NUMINAMATH_GPT_height_of_table_l1092_109295


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1092_109260

theorem sufficient_but_not_necessary_condition (k : ℝ) : 
  (k = 1 → ∃ x y : ℝ, x^2 + y^2 = 1 ∧ x - y + k = 0) ∧ 
  ¬(∀ k : ℝ, ∃ x y : ℝ, x^2 + y^2 = 1 ∧ x - y + k = 0 → k = 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1092_109260


namespace NUMINAMATH_GPT_complement_union_M_N_eq_16_l1092_109296

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the subsets M and N
def M : Set ℕ := {2, 3, 4}
def N : Set ℕ := {4, 5}

-- Define the union of M and N
def unionMN : Set ℕ := M ∪ N

-- Define the complement of M ∪ N in U
def complementUnionMN : Set ℕ := U \ unionMN

-- State the theorem that the complement is {1, 6}
theorem complement_union_M_N_eq_16 : complementUnionMN = {1, 6} := by
  sorry

end NUMINAMATH_GPT_complement_union_M_N_eq_16_l1092_109296


namespace NUMINAMATH_GPT_average_of_remaining_two_numbers_l1092_109202

theorem average_of_remaining_two_numbers 
  (a b c d e f : ℚ)
  (h1 : (a + b + c + d + e + f) / 6 = 6.40)
  (h2 : (a + b) / 2 = 6.2)
  (h3 : (c + d) / 2 = 6.1) : 
  ((e + f) / 2 = 6.9) :=
by
  sorry

end NUMINAMATH_GPT_average_of_remaining_two_numbers_l1092_109202


namespace NUMINAMATH_GPT_ellipse_slope_product_l1092_109252

theorem ellipse_slope_product (x₀ y₀ : ℝ) (hp : x₀^2 / 4 + y₀^2 / 3 = 1) :
  (y₀ / (x₀ + 2)) * (y₀ / (x₀ - 2)) = -3 / 4 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_ellipse_slope_product_l1092_109252


namespace NUMINAMATH_GPT_equation_is_correct_l1092_109298

-- Define the numbers
def n1 : ℕ := 2
def n2 : ℕ := 2
def n3 : ℕ := 11
def n4 : ℕ := 11

-- Define the mathematical expression and the target result
def expression : ℚ := (n1 + n2 / n3) * n4
def target_result : ℚ := 24

-- The proof statement
theorem equation_is_correct : expression = target_result := by
  sorry

end NUMINAMATH_GPT_equation_is_correct_l1092_109298


namespace NUMINAMATH_GPT_problem_statement_l1092_109299

variable {R : Type*} [LinearOrderedField R]

theorem problem_statement
  (x1 x2 x3 y1 y2 y3 : R)
  (h1 : x1 + x2 + x3 = 0)
  (h2 : y1 + y2 + y3 = 0)
  (h3 : x1 * y1 + x2 * y2 + x3 * y3 = 0)
  (h4 : (x1^2 + x2^2 + x3^2) * (y1^2 + y2^2 + y3^2) > 0) :
  (x1^2 / (x1^2 + x2^2 + x3^2) + y1^2 / (y1^2 + y2^2 + y3^2) = 2 / 3) := 
sorry

end NUMINAMATH_GPT_problem_statement_l1092_109299


namespace NUMINAMATH_GPT_delta_x_not_zero_l1092_109279

noncomputable def average_rate_of_change (f : ℝ → ℝ) (x : ℝ) (delta_x : ℝ) : ℝ :=
  (f (x + delta_x) - f x) / delta_x

theorem delta_x_not_zero (f : ℝ → ℝ) (x delta_x : ℝ) (h_neq : delta_x ≠ 0):
  average_rate_of_change f x delta_x ≠ 0 := 
by
  sorry

end NUMINAMATH_GPT_delta_x_not_zero_l1092_109279


namespace NUMINAMATH_GPT_tank_full_after_50_minutes_l1092_109297

-- Define the conditions as constants
def tank_capacity : ℕ := 850
def pipe_a_rate : ℕ := 40
def pipe_b_rate : ℕ := 30
def pipe_c_rate : ℕ := 20
def cycle_duration : ℕ := 3  -- duration of each cycle in minutes
def net_water_per_cycle : ℕ := pipe_a_rate + pipe_b_rate - pipe_c_rate  -- net liters added per cycle

-- Define the statement to be proved: the tank will be full at exactly 50 minutes
theorem tank_full_after_50_minutes :
  ∀ minutes_elapsed : ℕ, (minutes_elapsed = 50) →
  ((minutes_elapsed / cycle_duration) * net_water_per_cycle = tank_capacity - pipe_c_rate) :=
sorry

end NUMINAMATH_GPT_tank_full_after_50_minutes_l1092_109297


namespace NUMINAMATH_GPT_find_x_value_l1092_109277

theorem find_x_value (x : ℝ) (h1 : Real.sin (π / 2 - x) = -Real.sqrt 3 / 2) (h2 : π < x ∧ x < 2 * π) : x = 7 * π / 6 :=
sorry

end NUMINAMATH_GPT_find_x_value_l1092_109277


namespace NUMINAMATH_GPT_cos_triple_angle_l1092_109238

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (3 * θ) = -117/125 := 
by
  sorry

end NUMINAMATH_GPT_cos_triple_angle_l1092_109238


namespace NUMINAMATH_GPT_min_value_expr_l1092_109210

noncomputable def expr (x : ℝ) : ℝ := (Real.sin x)^8 + (Real.cos x)^8 + 3 / (Real.sin x)^6 + (Real.cos x)^6 + 3

theorem min_value_expr : ∃ x : ℝ, expr x = 14 / 31 := 
by
  sorry

end NUMINAMATH_GPT_min_value_expr_l1092_109210


namespace NUMINAMATH_GPT_min_value_PQ_l1092_109235

variable (t : ℝ) (x y : ℝ)

-- Parametric equations of line l
def line_l : Prop := (x = 4 * t - 1) ∧ (y = 3 * t - 3 / 2)

-- Polar equation of circle C
def polar_eq_circle_c (ρ θ : ℝ) : Prop :=
  ρ^2 = 2 * Real.sqrt 2 * ρ * Real.sin (θ - Real.pi / 4)

-- General equation of line l
def general_eq_line_l (x y : ℝ) : Prop := 3 * x - 4 * y = 3

-- Rectangular equation of circle C
def rectangular_eq_circle_c (x y : ℝ) : Prop :=
  (x + 1)^2 + (y - 1)^2 = 2

-- Definition of the condition where P is on line l
def p_on_line_l (x y : ℝ) : Prop := ∃ t : ℝ, line_l t x y

-- Minimum value of |PQ|
theorem min_value_PQ :
  p_on_line_l x y →
  general_eq_line_l x y →
  rectangular_eq_circle_c x y →
  ∃ d : ℝ, d = Real.sqrt 2 :=
by intros; sorry

end NUMINAMATH_GPT_min_value_PQ_l1092_109235


namespace NUMINAMATH_GPT_correct_ordering_l1092_109219

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f (-x) = f x
axiom monotonicity (x1 x2 : ℝ) (h1 : 0 ≤ x1) (h2 : 0 ≤ x2) (h3 : x1 ≠ x2) : (x1 - x2) * (f x1 - f x2) > 0

theorem correct_ordering : f 1 < f (-2) ∧ f (-2) < f 3 :=
by sorry

end NUMINAMATH_GPT_correct_ordering_l1092_109219


namespace NUMINAMATH_GPT_ratio_out_of_school_friends_to_classmates_l1092_109207

variable (F : ℕ) (classmates : ℕ := 20) (parents : ℕ := 2) (sister : ℕ := 1) (total : ℕ := 33)

theorem ratio_out_of_school_friends_to_classmates (h : classmates + F + parents + sister = total) :
  (F : ℚ) / classmates = 1 / 2 := by
    -- sorry allows this to build even if proof is not provided
    sorry

end NUMINAMATH_GPT_ratio_out_of_school_friends_to_classmates_l1092_109207


namespace NUMINAMATH_GPT_miles_remaining_l1092_109225

theorem miles_remaining (total_miles driven_miles : ℕ) (h1 : total_miles = 1200) (h2 : driven_miles = 768) :
    total_miles - driven_miles = 432 := by
  sorry

end NUMINAMATH_GPT_miles_remaining_l1092_109225


namespace NUMINAMATH_GPT_smallest_sum_of_squares_value_l1092_109257

noncomputable def collinear_points_min_value (A B C D E P : ℝ): Prop :=
  let AB := 3
  let BC := 2
  let CD := 5
  let DE := 4
  let pos_A := 0
  let pos_B := pos_A + AB
  let pos_C := pos_B + BC
  let pos_D := pos_C + CD
  let pos_E := pos_D + DE
  let P := P
  let AP := (P - pos_A)
  let BP := (P - pos_B)
  let CP := (P - pos_C)
  let DP := (P - pos_D)
  let EP := (P - pos_E)
  let sum_squares := AP^2 + BP^2 + CP^2 + DP^2 + EP^2
  (sum_squares = 85.2)

theorem smallest_sum_of_squares_value : ∃ (A B C D E P : ℝ), collinear_points_min_value A B C D E P :=
sorry

end NUMINAMATH_GPT_smallest_sum_of_squares_value_l1092_109257


namespace NUMINAMATH_GPT_area_relation_l1092_109244

noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := 
  0.5 * (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_relation (A B C A' B' C' : ℝ × ℝ) (hAA'BB'CC'parallel: 
  ∃ k : ℝ, (A'.1 - A.1 = k * (B'.1 - B.1)) ∧ (A'.2 - A.2 = k * (B'.2 - B.2)) ∧ 
           (B'.1 - B.1 = k * (C'.1 - C.1)) ∧ (B'.2 - B.2 = k * (C'.2 - C.2))) :
  3 * (area_triangle A B C + area_triangle A' B' C') = 
    area_triangle A B' C' + area_triangle B C' A' + area_triangle C A' B' +
    area_triangle A' B C + area_triangle B' C A + area_triangle C' A B := 
sorry

end NUMINAMATH_GPT_area_relation_l1092_109244


namespace NUMINAMATH_GPT_probability_sequence_rw_10_l1092_109280

noncomputable def probability_red_white_red : ℚ :=
  (4 / 10) * (6 / 9) * (3 / 8)

theorem probability_sequence_rw_10 :
    probability_red_white_red = 1 / 10 := by
  sorry

end NUMINAMATH_GPT_probability_sequence_rw_10_l1092_109280


namespace NUMINAMATH_GPT_area_of_triangle_OAB_is_5_l1092_109276

-- Define the parameters and assumptions
def OA : ℝ × ℝ := (-2, 1)
def OB : ℝ × ℝ := (4, 3)

noncomputable def area_triangle_OAB (OA OB : ℝ × ℝ) : ℝ :=
  1 / 2 * (OA.1 * OB.2 - OA.2 * OB.1)

-- The theorem we want to prove:
theorem area_of_triangle_OAB_is_5 : area_triangle_OAB OA OB = 5 := by
  sorry

end NUMINAMATH_GPT_area_of_triangle_OAB_is_5_l1092_109276


namespace NUMINAMATH_GPT_rectangle_area_theorem_l1092_109241

def rectangle_area (d : ℝ) (area : ℝ) : Prop :=
  ∃ w : ℝ, 0 < w ∧ 9 * w^2 + w^2 = d^2 ∧ area = 3 * w^2

theorem rectangle_area_theorem (d : ℝ) : rectangle_area d (3 * d^2 / 10) :=
sorry

end NUMINAMATH_GPT_rectangle_area_theorem_l1092_109241


namespace NUMINAMATH_GPT_line_equation_l1092_109201

theorem line_equation (x y : ℝ) (m : ℝ) (h1 : (1, 2) = (x, y)) (h2 : m = 3) :
  y = 3 * x - 1 :=
by
  sorry

end NUMINAMATH_GPT_line_equation_l1092_109201


namespace NUMINAMATH_GPT_polynomial_divisible_by_7_polynomial_divisible_by_12_l1092_109205

theorem polynomial_divisible_by_7 (x : ℤ) : (x^7 - x) % 7 = 0 := 
sorry

theorem polynomial_divisible_by_12 (x : ℤ) : (x^4 - x^2) % 12 = 0 := 
sorry

end NUMINAMATH_GPT_polynomial_divisible_by_7_polynomial_divisible_by_12_l1092_109205


namespace NUMINAMATH_GPT_smallest_class_size_l1092_109294

variable (x : ℕ) 

theorem smallest_class_size
  (h1 : 5 * x + 2 > 40)
  (h2 : x ≥ 0) : 
  5 * 8 + 2 = 42 :=
by sorry

end NUMINAMATH_GPT_smallest_class_size_l1092_109294


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1092_109249

theorem solution_set_of_inequality (x : ℝ) (h : 2 * x + 3 ≤ 1) : x ≤ -1 :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1092_109249


namespace NUMINAMATH_GPT_number_of_team_members_l1092_109246

theorem number_of_team_members (x x1 x2 : ℕ) (h₀ : x = x1 + x2) (h₁ : 3 * x1 + 4 * x2 = 33) : x = 6 :=
sorry

end NUMINAMATH_GPT_number_of_team_members_l1092_109246


namespace NUMINAMATH_GPT_present_age_ratio_l1092_109265

-- Define the conditions as functions in Lean.
def age_difference (M R : ℝ) : Prop := M - R = 7.5
def future_age_ratio (M R : ℝ) : Prop := (R + 10) / (M + 10) = 2 / 3

-- Define the goal as a proof problem in Lean.
theorem present_age_ratio (M R : ℝ) 
  (h1 : age_difference M R) 
  (h2 : future_age_ratio M R) : 
  R / M = 2 / 5 := 
by 
  sorry  -- Proof to be completed

end NUMINAMATH_GPT_present_age_ratio_l1092_109265


namespace NUMINAMATH_GPT_minimum_omega_l1092_109239

/-- Given function f and its properties, determine the minimum valid ω. -/
theorem minimum_omega {f : ℝ → ℝ} 
  (Hf : ∀ x : ℝ, f x = (1 / 2) * Real.cos (ω * x + φ) + 1)
  (Hsymmetry : ∃ k : ℤ, ω * (π / 3) + φ = k * π)
  (Hvalue : ∃ n : ℤ, f (π / 12) = 1 ∧ ω * (π / 12) + φ = n * π + π / 2)
  (Hpos : ω > 0) : ω = 2 := 
sorry

end NUMINAMATH_GPT_minimum_omega_l1092_109239


namespace NUMINAMATH_GPT_tomato_plants_per_row_l1092_109203

-- Definitions based on given conditions.
variables (T C P : ℕ)

-- Condition 1: For each row of tomato plants, she is planting 2 rows of cucumbers
def cucumber_rows (T : ℕ) := 2 * T

-- Condition 2: She has enough room for 15 rows of plants in total
def total_rows (T : ℕ) (C : ℕ) := T + C = 15

-- Condition 3: If each plant produces 3 tomatoes, she will have 120 tomatoes in total
def total_tomatoes (P : ℕ) := 5 * P * 3 = 120

-- The task is to prove that P = 8
theorem tomato_plants_per_row : 
  ∀ T C P : ℕ, cucumber_rows T = C → total_rows T C → total_tomatoes P → P = 8 :=
by
  -- The actual proof will go here.
  sorry

end NUMINAMATH_GPT_tomato_plants_per_row_l1092_109203


namespace NUMINAMATH_GPT_quadratic_has_one_real_root_l1092_109236

theorem quadratic_has_one_real_root (m : ℝ) (h : (6 * m)^2 - 4 * 1 * 4 * m = 0) : m = 4 / 9 :=
by sorry

end NUMINAMATH_GPT_quadratic_has_one_real_root_l1092_109236


namespace NUMINAMATH_GPT_factorization_option_a_factorization_option_b_factorization_option_c_factorization_option_d_correct_factorization_b_l1092_109224

-- Definitions from conditions
theorem factorization_option_a (a b : ℝ) : a^4 * b - 6 * a^3 * b + 9 * a^2 * b = a^2 * b * (a^2 - 6 * a + 9) ↔ a^2 * b * (a - 3)^2 ≠ a^2 * b * (a^2 - 6 * a - 9) := sorry

theorem factorization_option_b (x : ℝ) : (x^2 - x + 1/4) = (x - 1/2)^2 := sorry

theorem factorization_option_c (x : ℝ) : x^2 - 2 * x + 4 = (x - 2)^2 ↔ x^2 - 2 * x + 4 ≠ x^2 - 4 * x + 4 := sorry

theorem factorization_option_d (x y : ℝ) : 4 * x^2 - y^2 = (2 * x + y) * (2 * x - y) ↔ (4 * x + y) * (4 * x - y) ≠ (2 * x + y) * (2 * x - y) := sorry

-- Main theorem that states option B's factorization is correct
theorem correct_factorization_b (x : ℝ) (h1 : x^2 - x + 1/4 = (x - 1/2)^2)
                                (h2 : ∀ (a b : ℝ), a^4 * b - 6 * a^3 * b + 9 * a^2 * b ≠ a^2 * b * (a^2 - 6 * a - 9))
                                (h3 : ∀ (x : ℝ), x^2 - 2 * x + 4 ≠ (x - 2)^2)
                                (h4 : ∀ (x y : ℝ), 4 * x^2 - y^2 ≠ (4 * x + y) * (4 * x - y)) : 
                                (x^2 - x + 1/4 = (x - 1/2)^2) := 
                                by 
                                sorry

end NUMINAMATH_GPT_factorization_option_a_factorization_option_b_factorization_option_c_factorization_option_d_correct_factorization_b_l1092_109224


namespace NUMINAMATH_GPT_conversion_rate_false_l1092_109228

-- Definition of conversion rates between units
def conversion_rate_hour_minute : ℕ := 60
def conversion_rate_minute_second : ℕ := 60

-- Theorem stating that the rate being 100 is false under the given conditions
theorem conversion_rate_false (h1 : conversion_rate_hour_minute = 60) 
  (h2 : conversion_rate_minute_second = 60) : 
  ¬ (conversion_rate_hour_minute = 100 ∧ conversion_rate_minute_second = 100) :=
by {
  sorry
}

end NUMINAMATH_GPT_conversion_rate_false_l1092_109228


namespace NUMINAMATH_GPT_customers_in_other_countries_l1092_109222

def total_customers : ℕ := 7422
def us_customers : ℕ := 723
def other_customers : ℕ := total_customers - us_customers

theorem customers_in_other_countries : other_customers = 6699 := by
  sorry

end NUMINAMATH_GPT_customers_in_other_countries_l1092_109222


namespace NUMINAMATH_GPT_a_gt_b_neither_sufficient_nor_necessary_a2_gt_b2_a_gt_b_necessary_not_sufficient_ac2_gt_bc2_l1092_109218

variable {a b c : ℝ}

theorem a_gt_b_neither_sufficient_nor_necessary_a2_gt_b2 :
  ¬((a > b) → (a^2 > b^2)) ∧ ¬((a^2 > b^2) → (a > b)) :=
sorry

theorem a_gt_b_necessary_not_sufficient_ac2_gt_bc2 :
  ¬((a > b) → (a * c^2 > b * c^2)) ∧ ((a * c^2 > b * c^2) → (a > b)) :=
sorry

end NUMINAMATH_GPT_a_gt_b_neither_sufficient_nor_necessary_a2_gt_b2_a_gt_b_necessary_not_sufficient_ac2_gt_bc2_l1092_109218


namespace NUMINAMATH_GPT_number_of_ways_to_win_championships_l1092_109253

-- Definitions for the problem
def num_athletes := 5
def num_events := 3

-- Proof statement
theorem number_of_ways_to_win_championships : 
  (num_athletes ^ num_events) = 125 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_ways_to_win_championships_l1092_109253


namespace NUMINAMATH_GPT_green_fraction_is_three_fifths_l1092_109221

noncomputable def fraction_green_after_tripling (total_balloons : ℕ) : ℚ :=
  let green_balloons := total_balloons / 3
  let new_green_balloons := green_balloons * 3
  let new_total_balloons := total_balloons * (5 / 3)
  new_green_balloons / new_total_balloons

theorem green_fraction_is_three_fifths (total_balloons : ℕ) (h : total_balloons > 0) : 
  fraction_green_after_tripling total_balloons = 3 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_green_fraction_is_three_fifths_l1092_109221


namespace NUMINAMATH_GPT_jane_mistake_l1092_109286

theorem jane_mistake (x y z : ℤ) (h1 : x - y + z = 15) (h2 : x - y - z = 7) : x - y = 11 :=
by sorry

end NUMINAMATH_GPT_jane_mistake_l1092_109286


namespace NUMINAMATH_GPT_triangle_inequality_l1092_109263
open Real

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_abc : a + b > c) (h_acb : a + c > b) (h_bca : b + c > a) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l1092_109263


namespace NUMINAMATH_GPT_distance_from_point_to_focus_l1092_109273

noncomputable def point_on_parabola (P : ℝ × ℝ) (y : ℝ) : Prop :=
  y^2 = 16 * P.1 ∧ (P.2 = y ∨ P.2 = -y)

noncomputable def parabola_focus : ℝ × ℝ :=
  (4, 0)

theorem distance_from_point_to_focus
  (P : ℝ × ℝ) (y : ℝ)
  (h1 : point_on_parabola P y)
  (h2 : dist P (0, P.2) = 12) :
  dist P parabola_focus = 13 :=
sorry

end NUMINAMATH_GPT_distance_from_point_to_focus_l1092_109273


namespace NUMINAMATH_GPT_angies_age_l1092_109281

variable (A : ℝ)

theorem angies_age (h : 2 * A + 4 = 20) : A = 8 :=
sorry

end NUMINAMATH_GPT_angies_age_l1092_109281


namespace NUMINAMATH_GPT_no_integer_solutions_l1092_109278

theorem no_integer_solutions :
  ∀ (m n : ℤ), (m^3 + 4 * m^2 + 3 * m ≠ 8 * n^3 + 12 * n^2 + 6 * n + 1) := by
  sorry

end NUMINAMATH_GPT_no_integer_solutions_l1092_109278


namespace NUMINAMATH_GPT_graph_always_passes_fixed_point_l1092_109223

theorem graph_always_passes_fixed_point (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : 
  ∃ A : ℝ × ℝ, A = (-2, -1) ∧ (∀ x : ℝ, y = a^(x+2)-2 → y = -1 ∧ x = -2) :=
by
  use (-2, -1)
  sorry

end NUMINAMATH_GPT_graph_always_passes_fixed_point_l1092_109223


namespace NUMINAMATH_GPT_boy_running_time_l1092_109242

theorem boy_running_time (s : ℝ) (v : ℝ) (h1 : s = 35) (h2 : v = 9) : 
  (4 * s) / (v * 1000 / 3600) = 56 := by
  sorry

end NUMINAMATH_GPT_boy_running_time_l1092_109242


namespace NUMINAMATH_GPT_problem1_l1092_109264

theorem problem1
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ)
  (h₁ : (3*x - 2)^(6) = a₀ + a₁ * (2*x - 1) + a₂ * (2*x - 1)^2 + a₃ * (2*x - 1)^3 + a₄ * (2*x - 1)^4 + a₅ * (2*x - 1)^5 + a₆ * (2*x - 1)^6)
  (h₂ : a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 1)
  (h₃ : a₀ - a₁ + a₂ - a₃ + a₄ - a₅ + a₆ = 64) :
  (a₁ + a₃ + a₅) / (a₀ + a₂ + a₄ + a₆) = -63 / 65 := by
  sorry

end NUMINAMATH_GPT_problem1_l1092_109264


namespace NUMINAMATH_GPT_range_of_m_l1092_109272

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 - 4 * x ≥ m) → m ≤ -3 := 
sorry

end NUMINAMATH_GPT_range_of_m_l1092_109272


namespace NUMINAMATH_GPT_range_of_k_l1092_109283

theorem range_of_k (k : ℝ) : 
  (∀ x : ℝ, k * x ^ 2 + 2 * k * x + 3 ≠ 0) ↔ (0 ≤ k ∧ k < 3) :=
by sorry

end NUMINAMATH_GPT_range_of_k_l1092_109283


namespace NUMINAMATH_GPT_initial_y_percentage_proof_l1092_109285

variable (initial_volume : ℝ) (added_volume : ℝ) (initial_percentage_x : ℝ) (result_percentage_x : ℝ)

-- Conditions
def initial_volume_condition : Prop := initial_volume = 80
def added_volume_condition : Prop := added_volume = 20
def initial_percentage_x_condition : Prop := initial_percentage_x = 0.30
def result_percentage_x_condition : Prop := result_percentage_x = 0.44

-- Question
def initial_percentage_y (initial_volume added_volume initial_percentage_x result_percentage_x : ℝ) : ℝ :=
  1 - initial_percentage_x

-- Theorem
theorem initial_y_percentage_proof 
  (h1 : initial_volume_condition initial_volume)
  (h2 : added_volume_condition added_volume)
  (h3 : initial_percentage_x_condition initial_percentage_x)
  (h4 : result_percentage_x_condition result_percentage_x) :
  initial_percentage_y initial_volume added_volume initial_percentage_x result_percentage_x = 0.70 := 
sorry

end NUMINAMATH_GPT_initial_y_percentage_proof_l1092_109285


namespace NUMINAMATH_GPT_maximum_sets_l1092_109251

-- define the initial conditions
def dinner_forks : Nat := 6
def knives : Nat := dinner_forks + 9
def soup_spoons : Nat := 2 * knives
def teaspoons : Nat := dinner_forks / 2
def dessert_forks : Nat := teaspoons / 3
def butter_knives : Nat := 2 * dessert_forks

def max_capacity_g : Nat := 20000

def weight_dinner_fork : Nat := 80
def weight_knife : Nat := 100
def weight_soup_spoon : Nat := 85
def weight_teaspoon : Nat := 50
def weight_dessert_fork : Nat := 70
def weight_butter_knife : Nat := 65

-- Calculate the total weight of the existing cutlery
def total_weight_existing : Nat := 
  (dinner_forks * weight_dinner_fork) + 
  (knives * weight_knife) + 
  (soup_spoons * weight_soup_spoon) + 
  (teaspoons * weight_teaspoon) + 
  (dessert_forks * weight_dessert_fork) + 
  (butter_knives * weight_butter_knife)

-- Calculate the weight of one 2-piece cutlery set (1 knife + 1 dinner fork)
def weight_set : Nat := weight_knife + weight_dinner_fork

-- The remaining capacity in the drawer
def remaining_capacity_g : Nat := max_capacity_g - total_weight_existing

-- The maximum number of 2-piece cutlery sets that can be added
def max_2_piece_sets : Nat := remaining_capacity_g / weight_set

-- Theorem: maximum number of 2-piece cutlery sets that can be added is 84
theorem maximum_sets : max_2_piece_sets = 84 :=
by
  sorry

end NUMINAMATH_GPT_maximum_sets_l1092_109251


namespace NUMINAMATH_GPT_kate_spent_on_mouse_l1092_109269

theorem kate_spent_on_mouse :
  let march := 27
  let april := 13
  let may := 28
  let saved := march + april + may
  let keyboard := 49
  let left := 14
  saved - left - keyboard = 5 :=
by
  let march := 27
  let april := 13
  let may := 28
  let saved := march + april + may
  let keyboard := 49
  let left := 14
  show saved - left - keyboard = 5
  sorry

end NUMINAMATH_GPT_kate_spent_on_mouse_l1092_109269


namespace NUMINAMATH_GPT_green_fish_count_l1092_109291

theorem green_fish_count (B O G : ℕ) (h1 : B = (2 / 5) * 200)
  (h2 : O = 2 * B - 30) (h3 : G = (3 / 2) * O) (h4 : B + O + G = 200) : 
  G = 195 :=
by
  sorry

end NUMINAMATH_GPT_green_fish_count_l1092_109291


namespace NUMINAMATH_GPT_sandy_initial_fish_l1092_109226

theorem sandy_initial_fish (bought_fish : ℕ) (total_fish : ℕ) (h1 : bought_fish = 6) (h2 : total_fish = 32) :
  total_fish - bought_fish = 26 :=
by
  sorry

end NUMINAMATH_GPT_sandy_initial_fish_l1092_109226


namespace NUMINAMATH_GPT_total_apples_l1092_109287

theorem total_apples (A B C : ℕ) (h1 : A + B = 11) (h2 : B + C = 18) (h3 : A + C = 19) : A + B + C = 24 :=  
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_total_apples_l1092_109287


namespace NUMINAMATH_GPT_baker_sold_cakes_l1092_109290

theorem baker_sold_cakes :
  ∀ (C : ℕ),  -- C is the number of cakes Baker sold
    (∃ (cakes pastries : ℕ), 
      cakes = 14 ∧ 
      pastries = 153 ∧ 
      (∃ (sold_pastries : ℕ), sold_pastries = 8 ∧ 
      C = 89 + sold_pastries)) 
  → C = 97 :=
by
  intros C h
  rcases h with ⟨cakes, pastries, cakes_eq, pastries_eq, ⟨sold_pastries, sold_pastries_eq, C_eq⟩⟩
  -- Fill in the proof details
  sorry

end NUMINAMATH_GPT_baker_sold_cakes_l1092_109290


namespace NUMINAMATH_GPT_box_weight_difference_l1092_109248

theorem box_weight_difference:
  let w1 := 2
  let w2 := 3
  let w3 := 13
  let w4 := 7
  let w5 := 10
  (max (max (max (max w1 w2) w3) w4) w5) - (min (min (min (min w1 w2) w3) w4) w5) = 11 :=
by
  sorry

end NUMINAMATH_GPT_box_weight_difference_l1092_109248


namespace NUMINAMATH_GPT_min_value_of_quadratic_l1092_109237

theorem min_value_of_quadratic :
  (∀ x : ℝ, 3 * x^2 - 12 * x + 908 ≥ 896) ∧ (∃ x : ℝ, 3 * x^2 - 12 * x + 908 = 896) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l1092_109237


namespace NUMINAMATH_GPT_minimum_area_triangle_ABC_l1092_109247

-- Define the vertices of the triangle
def A : ℤ × ℤ := (0,0)
def B : ℤ × ℤ := (30,18)

-- Define a function to calculate the area of the triangle using the Shoelace formula
def area_of_triangle (A B C : ℤ × ℤ) : ℤ := 15 * (C.2).natAbs

-- State the theorem
theorem minimum_area_triangle_ABC : 
  ∀ C : ℤ × ℤ, C ≠ (0,0) → area_of_triangle A B C ≥ 15 :=
by
  sorry -- Skip the proof

end NUMINAMATH_GPT_minimum_area_triangle_ABC_l1092_109247


namespace NUMINAMATH_GPT_office_distance_l1092_109259

theorem office_distance (d t : ℝ) 
    (h1 : d = 40 * (t + 1.5)) 
    (h2 : d - 40 = 60 * (t - 2)) : 
    d = 340 :=
by
  -- The detailed proof omitted
  sorry

end NUMINAMATH_GPT_office_distance_l1092_109259


namespace NUMINAMATH_GPT_space_shuttle_speed_kmph_l1092_109275

-- Question: Prove that the speed of the space shuttle in kilometers per hour is 32400, given it travels at 9 kilometers per second and there are 3600 seconds in an hour.
theorem space_shuttle_speed_kmph :
  (9 * 3600 = 32400) :=
by
  sorry

end NUMINAMATH_GPT_space_shuttle_speed_kmph_l1092_109275


namespace NUMINAMATH_GPT_cylinder_height_l1092_109232

noncomputable def height_of_cylinder_inscribed_in_sphere : ℝ := 4 * Real.sqrt 10

theorem cylinder_height :
  ∀ (R_cylinder R_sphere : ℝ), R_cylinder = 3 → R_sphere = 7 →
  (height_of_cylinder_inscribed_in_sphere = 4 * Real.sqrt 10) := by
  intros R_cylinder R_sphere h1 h2
  sorry

end NUMINAMATH_GPT_cylinder_height_l1092_109232


namespace NUMINAMATH_GPT_total_balloons_correct_l1092_109289

-- Definitions based on the conditions
def brookes_initial_balloons : Nat := 12
def brooke_additional_balloons : Nat := 8

def tracys_initial_balloons : Nat := 6
def tracy_additional_balloons : Nat := 24

-- Calculate the number of balloons each person has after the additions and Tracy popping half
def brookes_final_balloons : Nat := brookes_initial_balloons + brooke_additional_balloons
def tracys_balloons_after_addition : Nat := tracys_initial_balloons + tracy_additional_balloons
def tracys_final_balloons : Nat := tracys_balloons_after_addition / 2

-- Total number of balloons
def total_balloons : Nat := brookes_final_balloons + tracys_final_balloons

-- The proof statement
theorem total_balloons_correct : total_balloons = 35 := by
  -- Proof would go here (but we'll skip with sorry)
  sorry

end NUMINAMATH_GPT_total_balloons_correct_l1092_109289


namespace NUMINAMATH_GPT_bill_left_with_money_l1092_109267

def foolsgold (ounces_sold : Nat) (price_per_ounce : Nat) (fine : Nat): Int :=
  (ounces_sold * price_per_ounce) - fine

theorem bill_left_with_money :
  foolsgold 8 9 50 = 22 :=
by
  sorry

end NUMINAMATH_GPT_bill_left_with_money_l1092_109267


namespace NUMINAMATH_GPT_max_rectangle_area_with_prime_dimension_l1092_109282

theorem max_rectangle_area_with_prime_dimension :
  ∃ (l w : ℕ), 2 * (l + w) = 120 ∧ (Prime l ∨ Prime w) ∧ l * w = 899 :=
by
  sorry

end NUMINAMATH_GPT_max_rectangle_area_with_prime_dimension_l1092_109282


namespace NUMINAMATH_GPT_find_p_at_8_l1092_109211

noncomputable def h (x : ℝ) : ℝ := x^3 - x^2 + x - 1

noncomputable def p (x : ℝ) : ℝ :=
  let a := sorry ; -- root 1 of h
  let b := sorry ; -- root 2 of h
  let c := sorry ; -- root 3 of h
  let B := 2 / ((1 - a^3) * (1 - b^3) * (1 - c^3))
  B * (x - a^3) * (x - b^3) * (x - c^3)

theorem find_p_at_8 : p 8 = 1008 := sorry

end NUMINAMATH_GPT_find_p_at_8_l1092_109211


namespace NUMINAMATH_GPT_prime_number_solution_l1092_109220

theorem prime_number_solution (X Y : ℤ) (h_prime : Prime (X^4 + 4 * Y^4)) :
  (X = 1 ∧ Y = 1) ∨ (X = -1 ∧ Y = -1) :=
sorry

end NUMINAMATH_GPT_prime_number_solution_l1092_109220


namespace NUMINAMATH_GPT_triangle_side_lengths_approx_l1092_109268

noncomputable def approx_side_lengths (AB : ℝ) (BAC ABC : ℝ) : ℝ × ℝ :=
  let α := BAC * Real.pi / 180
  let β := ABC * Real.pi / 180
  let c := AB
  let β1 := (90 - (BAC)) * Real.pi / 180
  let m := 2 * c * α * (β1 + 3) / (9 - α * β1)
  let c1 := 2 * c * β1 * (α + 3) / (9 - α * β1)
  let β2 := β1 - β
  let γ1 := α + β
  let a1 := β2 / γ1 * (γ1 + 3) / (β2 + 3) * m
  let a := (9 - β2 * γ1) / (2 * γ1 * (β2 + 3)) * m
  let b := c1 - a1
  (a, b)

theorem triangle_side_lengths_approx (AB : ℝ) (BAC ABC : ℝ) (hAB : AB = 441) (hBAC : BAC = 16.2) (hABC : ABC = 40.6) :
  approx_side_lengths AB BAC ABC = (147, 344) := by
  sorry

end NUMINAMATH_GPT_triangle_side_lengths_approx_l1092_109268


namespace NUMINAMATH_GPT_count_perfect_square_factors_l1092_109271

theorem count_perfect_square_factors : 
  let n := (2^10) * (3^12) * (5^15) * (7^7)
  ∃ (count : ℕ), count = 1344 ∧
    (∀ (a b c d : ℕ), 0 ≤ a ∧ a ≤ 10 ∧ 0 ≤ b ∧ b ≤ 12 ∧ 0 ≤ c ∧ c ≤ 15 ∧ 0 ≤ d ∧ d ≤ 7 →
      ((a % 2 = 0) ∧ (b % 2 = 0) ∧ (c % 2 = 0) ∧ (d % 2 = 0) →
        ∃ (k : ℕ), (2^a * 3^b * 5^c * 7^d) = k ∧ k ∣ n)) :=
by
  sorry

end NUMINAMATH_GPT_count_perfect_square_factors_l1092_109271


namespace NUMINAMATH_GPT_hexagon_tiling_min_colors_l1092_109214

theorem hexagon_tiling_min_colors :
  ∀ (s₁ s₂ : ℝ) (hex_area : ℝ) (tile_area : ℝ) (tiles_needed : ℕ) (n : ℕ),
    s₁ = 6 →
    s₂ = 0.5 →
    hex_area = (3 * Real.sqrt 3 / 2) * s₁^2 →
    tile_area = (Real.sqrt 3 / 4) * s₂^2 →
    tiles_needed = hex_area / tile_area →
    tiles_needed ≤ (Nat.choose n 3) →
    n ≥ 19 :=
by
  intros s₁ s₂ hex_area tile_area tiles_needed n
  intros s₁_eq s₂_eq hex_area_eq tile_area_eq tiles_needed_eq color_constraint
  sorry

end NUMINAMATH_GPT_hexagon_tiling_min_colors_l1092_109214


namespace NUMINAMATH_GPT_donald_juice_l1092_109254

variable (P D : ℕ)

theorem donald_juice (h1 : P = 3) (h2 : D = 2 * P + 3) : D = 9 := by
  sorry

end NUMINAMATH_GPT_donald_juice_l1092_109254


namespace NUMINAMATH_GPT_min_z_value_l1092_109256

variable (x y z : ℝ)

theorem min_z_value (h1 : x - 1 ≥ 0) (h2 : 2 * x - y - 1 ≤ 0) (h3 : x + y - 3 ≤ 0) :
  z = x - y → z = -1 :=
by sorry

end NUMINAMATH_GPT_min_z_value_l1092_109256


namespace NUMINAMATH_GPT_average_percentage_revenue_fall_l1092_109231

theorem average_percentage_revenue_fall
  (initial_revenue_A final_revenue_A : ℝ)
  (initial_revenue_B final_revenue_B : ℝ) (exchange_rate_B : ℝ)
  (initial_revenue_C final_revenue_C : ℝ) (exchange_rate_C : ℝ) :
  initial_revenue_A = 72.0 →
  final_revenue_A = 48.0 →
  initial_revenue_B = 20.0 →
  final_revenue_B = 15.0 →
  exchange_rate_B = 1.30 →
  initial_revenue_C = 6000.0 →
  final_revenue_C = 5500.0 →
  exchange_rate_C = 0.0091 →
  (33.33 + 25 + 8.33) / 3 = 22.22 :=
by
  sorry

end NUMINAMATH_GPT_average_percentage_revenue_fall_l1092_109231


namespace NUMINAMATH_GPT_part1_am_eq_ln_am1_minus_1_part2_am_le_am1_minus_2_part3_k_is_3_l1092_109212

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

end NUMINAMATH_GPT_part1_am_eq_ln_am1_minus_1_part2_am_le_am1_minus_2_part3_k_is_3_l1092_109212


namespace NUMINAMATH_GPT_remainder_3_pow_2n_plus_8_l1092_109240

theorem remainder_3_pow_2n_plus_8 (n : Nat) : (3 ^ (2 * n) + 8) % 8 = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_3_pow_2n_plus_8_l1092_109240


namespace NUMINAMATH_GPT_meaningful_sqrt_range_l1092_109284

theorem meaningful_sqrt_range (x : ℝ) (h : 0 ≤ x + 3) : -3 ≤ x :=
by sorry

end NUMINAMATH_GPT_meaningful_sqrt_range_l1092_109284


namespace NUMINAMATH_GPT_eddie_rate_l1092_109250

variables (hours_sam hours_eddie rate_sam total_crates rate_eddie : ℕ)

def sam_conditions :=
  hours_sam = 6 ∧ rate_sam = 60

def eddie_conditions :=
  hours_eddie = 4 ∧ total_crates = hours_sam * rate_sam

theorem eddie_rate (hs : sam_conditions hours_sam rate_sam)
                   (he : eddie_conditions hours_sam hours_eddie rate_sam total_crates) :
  rate_eddie = 90 :=
by sorry

end NUMINAMATH_GPT_eddie_rate_l1092_109250


namespace NUMINAMATH_GPT_strike_time_10_times_l1092_109243

def time_to_strike (n : ℕ) : ℝ :=
  if n = 0 then 0 else (n - 1) * 6

theorem strike_time_10_times : time_to_strike 10 = 60 :=
  by {
    -- Proof outline
    -- time_to_strike 10 = (10 - 1) * 6 = 9 * 6 = 54. Thanks to provided solution -> we shall consider that time take 10 seconds for the clock to start striking.
    sorry
  }

end NUMINAMATH_GPT_strike_time_10_times_l1092_109243


namespace NUMINAMATH_GPT_seq_nat_eq_n_l1092_109262

theorem seq_nat_eq_n (a : ℕ → ℕ) (h_inc : ∀ n, a n < a (n + 1))
  (h_le : ∀ n, a n ≤ n + 2020)
  (h_div : ∀ n, a (n + 1) ∣ (n^3 * a n - 1)) :
  ∀ n, a n = n :=
by
  sorry

end NUMINAMATH_GPT_seq_nat_eq_n_l1092_109262


namespace NUMINAMATH_GPT_strips_overlap_area_l1092_109292

theorem strips_overlap_area (L1 L2 AL AR S : ℝ) (hL1 : L1 = 9) (hL2 : L2 = 7) (hAL : AL = 27) (hAR : AR = 18) 
    (hrel : (AL + S) / (AR + S) = L1 / L2) : S = 13.5 := 
by
  sorry

end NUMINAMATH_GPT_strips_overlap_area_l1092_109292


namespace NUMINAMATH_GPT_crayons_per_color_in_each_box_l1092_109217

def crayons_in_each_box : ℕ := 2

theorem crayons_per_color_in_each_box
  (colors : ℕ)
  (boxes_per_hour : ℕ)
  (crayons_in_4_hours : ℕ)
  (hours : ℕ)
  (total_boxes : ℕ := boxes_per_hour * hours)
  (crayons_per_box : ℕ := crayons_in_4_hours / total_boxes)
  (crayons_per_color : ℕ := crayons_per_box / colors)
  (colors_eq : colors = 4)
  (boxes_per_hour_eq : boxes_per_hour = 5)
  (crayons_in_4_hours_eq : crayons_in_4_hours = 160)
  (hours_eq : hours = 4) : crayons_per_color = crayons_in_each_box :=
by {
  sorry
}

end NUMINAMATH_GPT_crayons_per_color_in_each_box_l1092_109217
