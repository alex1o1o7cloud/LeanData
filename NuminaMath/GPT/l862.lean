import Mathlib

namespace NUMINAMATH_GPT_distance_to_x_axis_l862_86207

theorem distance_to_x_axis (x y : ℝ) :
  (x^2 / 9 - y^2 / 16 = 1) →
  (x^2 + y^2 = 25) →
  abs y = 16 / 5 :=
by
  -- Conditions: x^2 / 9 - y^2 / 16 = 1, x^2 + y^2 = 25
  -- Conclusion: abs y = 16 / 5 
  intro h1 h2
  sorry

end NUMINAMATH_GPT_distance_to_x_axis_l862_86207


namespace NUMINAMATH_GPT_equivalent_expression_l862_86283

theorem equivalent_expression :
  (2 + 5) * (2^2 + 5^2) * (2^4 + 5^4) * (2^8 + 5^8) * (2^16 + 5^16) * (2^32 + 5^32) * (2^64 + 5^64) =
  5^128 - 2^128 := by
  sorry

end NUMINAMATH_GPT_equivalent_expression_l862_86283


namespace NUMINAMATH_GPT_determine_m_to_satisfy_conditions_l862_86272

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 5) * x^(m - 1)

theorem determine_m_to_satisfy_conditions : 
  ∃ (m : ℝ), (m = 3) ∧ ∀ (x : ℝ), (0 < x → (m^2 - m - 5 > 0) ∧ (m - 1 > 0)) :=
by
  sorry

end NUMINAMATH_GPT_determine_m_to_satisfy_conditions_l862_86272


namespace NUMINAMATH_GPT_not_satisfiable_conditions_l862_86265

theorem not_satisfiable_conditions (x y : ℕ) (h1 : 0 ≤ x ∧ x ≤ 9) (h2 : 0 ≤ y ∧ y ≤ 9) 
    (h3 : 10 * x + y % 80 = 0) (h4 : x + y = 2) : false := 
by 
  -- The proof is omitted because we are only asked for the statement.
  sorry

end NUMINAMATH_GPT_not_satisfiable_conditions_l862_86265


namespace NUMINAMATH_GPT_x_squared_plus_y_squared_l862_86247

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 10) : x^2 + y^2 = 29 := 
by
  sorry

end NUMINAMATH_GPT_x_squared_plus_y_squared_l862_86247


namespace NUMINAMATH_GPT_Jenny_walked_distance_l862_86266

-- Given: Jenny ran 0.6 mile.
-- Given: Jenny ran 0.2 miles farther than she walked.
-- Prove: Jenny walked 0.4 miles.

variable (r w : ℝ)

theorem Jenny_walked_distance
  (h1 : r = 0.6) 
  (h2 : r = w + 0.2) : 
  w = 0.4 :=
sorry

end NUMINAMATH_GPT_Jenny_walked_distance_l862_86266


namespace NUMINAMATH_GPT_t_plus_reciprocal_l862_86254

theorem t_plus_reciprocal (t : ℝ) (h : t^2 - 3 * t + 1 = 0) (ht : t ≠ 0) : t + 1/t = 3 :=
by sorry

end NUMINAMATH_GPT_t_plus_reciprocal_l862_86254


namespace NUMINAMATH_GPT_solution_l862_86209

noncomputable def F (a b c : ℝ) := a * (b ^ 3) + c

theorem solution (a : ℝ) (h : F a 2 3 = F a 3 10) : a = -7 / 19 := sorry

end NUMINAMATH_GPT_solution_l862_86209


namespace NUMINAMATH_GPT_molecular_weight_C8H10N4O6_eq_258_22_l862_86214

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

def number_C : ℕ := 8
def number_H : ℕ := 10
def number_N : ℕ := 4
def number_O : ℕ := 6

def molecular_weight : ℝ :=
    (number_C * atomic_weight_C) +
    (number_H * atomic_weight_H) +
    (number_N * atomic_weight_N) +
    (number_O * atomic_weight_O)

theorem molecular_weight_C8H10N4O6_eq_258_22 :
  molecular_weight = 258.22 :=
  by
    sorry

end NUMINAMATH_GPT_molecular_weight_C8H10N4O6_eq_258_22_l862_86214


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l862_86201

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 ≥ 1) → (¬(x ≥ 1) ∨ (x ≥ 1)) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l862_86201


namespace NUMINAMATH_GPT_triangle_side_difference_l862_86230

theorem triangle_side_difference (x : ℕ) : 3 < x ∧ x < 17 → (∃ a b : ℕ, 3 < a ∧ a < 17 ∧ 3 < b ∧ b < 17 ∧ a - b = 12) :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_difference_l862_86230


namespace NUMINAMATH_GPT_eval_expr_l862_86277
open Real

theorem eval_expr : 4 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 8000 := by
  sorry

end NUMINAMATH_GPT_eval_expr_l862_86277


namespace NUMINAMATH_GPT_unique_ellipse_through_points_with_perpendicular_axes_infinite_ellipses_when_points_coincide_l862_86234

-- Definitions of points and lines
structure Point (α : Type*) := (x : α) (y : α)
structure Line (α : Type*) := (a : α) (b : α) -- Represented as ax + by = 0

-- Given conditions
variables {α : Type*} [Field α]
variables (P Q : Point α)
variables (L1 L2 : Line α) -- L1 and L2 are perpendicular

-- Proof problem statement
theorem unique_ellipse_through_points_with_perpendicular_axes (P Q : Point α) (L1 L2 : Line α) (h_perp : L1.a * L2.b = - (L1.b * L2.a)) :
(P ≠ Q) → 
∃! (E : Set (Point α)), -- E represents the ellipse as a set of points
(∀ (p : Point α), p ∈ E → (p = P ∨ p = Q)) ∧ -- E passes through P and Q
(∀ (p : Point α), ∃ (u v : α), p.x = u ∨ p.y = v) := -- E has axes along L1 and L2
sorry

theorem infinite_ellipses_when_points_coincide (P : Point α) (L1 L2 : Line α) (h_perp : L1.a * L2.b = - (L1.b * L2.a)) :
∃ (E : Set (Point α)), -- E represents an ellipse
(∀ (p : Point α), p ∈ E → p = P) ∧ -- E passes through P
(∀ (p : Point α), ∃ (u v : α), p.x = u ∨ p.y = v) := -- E has axes along L1 and L2
sorry

end NUMINAMATH_GPT_unique_ellipse_through_points_with_perpendicular_axes_infinite_ellipses_when_points_coincide_l862_86234


namespace NUMINAMATH_GPT_locate_z_in_fourth_quadrant_l862_86270

def z_in_quadrant_fourth (z : ℂ) : Prop :=
  (z.re > 0) ∧ (z.im < 0)

theorem locate_z_in_fourth_quadrant (z : ℂ) (i : ℂ) (h : i * i = -1) 
(hz : z * (1 + i) = 1) : z_in_quadrant_fourth z :=
sorry

end NUMINAMATH_GPT_locate_z_in_fourth_quadrant_l862_86270


namespace NUMINAMATH_GPT_max_value_of_f_l862_86228

noncomputable def f (x : Real) := 2 * (Real.sin x) ^ 2 - (Real.tan x) ^ 2

theorem max_value_of_f : 
  ∃ (x : Real), f x = 3 - 2 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_max_value_of_f_l862_86228


namespace NUMINAMATH_GPT_PolyCoeffInequality_l862_86240

open Real

variable (p q : ℝ[X])
variable (a : ℝ)
variable (n : ℕ)
variable (h k : ℝ)
variable (deg_p : p.degree = n)
variable (deg_q : q.degree = n - 1)
variable (hp : ∀ i, i ≤ n → |p.coeff i| ≤ h)
variable (hq : ∀ i, i < n → |q.coeff i| ≤ k)
variable (hpq : p = (X + C a) * q)

theorem PolyCoeffInequality : k ≤ h^n := by
  sorry

end NUMINAMATH_GPT_PolyCoeffInequality_l862_86240


namespace NUMINAMATH_GPT_min_value_expression_l862_86216

theorem min_value_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
    (h3 : ∀ x y : ℝ, x + y + a = 0 → (x - b)^2 + (y - 1)^2 = 2) : 
    (∃ c : ℝ,  c = 4 ∧ ∀ a b : ℝ, (0 < a → 0 < b → x + y + a = 0 → (x - b)^2 + (y - 1)^2 = 2 →  (3 - 2 * b)^2 / (2 * a) ≥ c)) :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l862_86216


namespace NUMINAMATH_GPT_balloon_permutations_l862_86289

theorem balloon_permutations : 
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  (Nat.factorial total_letters) / ((Nat.factorial repetitions_L) * (Nat.factorial repetitions_O)) = 1260 := 
by
  sorry

end NUMINAMATH_GPT_balloon_permutations_l862_86289


namespace NUMINAMATH_GPT_matrix_cubic_l862_86285

noncomputable def matrix_entries (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x, y, z], ![y, z, x], ![z, x, y]]

theorem matrix_cubic (x y z : ℝ) (N : Matrix (Fin 3) (Fin 3) ℝ)
    (hN : N = matrix_entries x y z)
    (hn : N ^ 2 = 2 • (1 : Matrix (Fin 3) (Fin 3) ℝ))
    (hxyz : x * y * z = -2) :
  x^3 + y^3 + z^3 = -6 + 2 * Real.sqrt 2 ∨ x^3 + y^3 + z^3 = -6 - 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_matrix_cubic_l862_86285


namespace NUMINAMATH_GPT_integer_points_inequality_l862_86236

theorem integer_points_inequality
  (a b : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a * b + a - b - 5 = 0)
  (M := max ((a : ℤ)^2 + (b : ℤ)^2)) :
  (3 * x^2 + 2 * y^2 <= M) → ∃ (n : ℕ), n = 51 :=
by sorry

end NUMINAMATH_GPT_integer_points_inequality_l862_86236


namespace NUMINAMATH_GPT_prove_a_minus_b_plus_c_eq_3_l862_86200

variable {a b c m n : ℝ}

theorem prove_a_minus_b_plus_c_eq_3 
    (h : ∀ x : ℝ, m * x^2 - n * x + 3 = a * (x - 1)^2 + b * (x - 1) + c) :
    a - b + c = 3 :=
sorry

end NUMINAMATH_GPT_prove_a_minus_b_plus_c_eq_3_l862_86200


namespace NUMINAMATH_GPT_find_number_l862_86255

theorem find_number (x : ℝ) (h₁ : |x| + 1/x = 0) (h₂ : x ≠ 0) : x = -1 :=
sorry

end NUMINAMATH_GPT_find_number_l862_86255


namespace NUMINAMATH_GPT_san_antonio_to_austin_buses_passed_l862_86227

def departure_schedule (departure_time_A_to_S departure_time_S_to_A travel_time : ℕ) : Prop :=
  ∀ t, (t < travel_time) →
       (∃ n, t = (departure_time_A_to_S + n * 60)) ∨
       (∃ m, t = (departure_time_S_to_A + m * 60)) →
       t < travel_time

theorem san_antonio_to_austin_buses_passed :
  let departure_time_A_to_S := 30  -- Austin to San Antonio buses leave every hour on the half-hour (e.g., 00:30, 1:30, ...)
  let departure_time_S_to_A := 0   -- San Antonio to Austin buses leave every hour on the hour (e.g., 00:00, 1:00, ...)
  let travel_time := 6 * 60        -- The trip takes 6 hours, or 360 minutes
  departure_schedule departure_time_A_to_S departure_time_S_to_A travel_time →
  ∃ count, count = 12 := 
by
  sorry

end NUMINAMATH_GPT_san_antonio_to_austin_buses_passed_l862_86227


namespace NUMINAMATH_GPT_slope_of_asymptotes_is_one_l862_86286

-- Given definitions and axioms
variables (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (A1 : ℝ × ℝ := (-a, 0))
  (A2 : ℝ × ℝ := (a, 0))
  (B : ℝ × ℝ := (c, b^2 / a))
  (C : ℝ × ℝ := (c, -b^2 / a))
  (Perp : (b^2 / a) / (c + a) * -(b^2 / a) / (c - a) = -1)

-- Proof goal
theorem slope_of_asymptotes_is_one : a = b → (∀ m : ℝ, m = (b / a) ∨ m = -(b / a)) ↔ ∀ m : ℝ, m = 1 ∨ m = -1 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_asymptotes_is_one_l862_86286


namespace NUMINAMATH_GPT_find_C_and_D_l862_86250

variables (C D : ℝ)

theorem find_C_and_D (h : 4 * C + 2 * D + 5 = 30) : C = 5.25 ∧ D = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_C_and_D_l862_86250


namespace NUMINAMATH_GPT_problem_solution_l862_86219

open Nat

def sum_odd (n : ℕ) : ℕ :=
  n ^ 2

def sum_even (n : ℕ) : ℕ :=
  n * (n + 1)

theorem problem_solution : 
  sum_odd 1010 - sum_even 1009 = 1010 :=
by
  -- Here the proof would go
  sorry

end NUMINAMATH_GPT_problem_solution_l862_86219


namespace NUMINAMATH_GPT_product_of_four_consecutive_integers_is_product_of_two_consecutive_even_numbers_l862_86290

theorem product_of_four_consecutive_integers_is_product_of_two_consecutive_even_numbers (n : ℤ) :
  let T := (n - 1) * n * (n + 1) * (n + 2)
  let M := n * (n + 1)
  T = (M - 2) * M :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_product_of_four_consecutive_integers_is_product_of_two_consecutive_even_numbers_l862_86290


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_for_f_to_be_odd_l862_86217

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def f (a b x : ℝ) : ℝ :=
  x * abs (x + a) + b

theorem necessary_and_sufficient_condition_for_f_to_be_odd (a b : ℝ) :
  is_odd_function (f a b) ↔ sorry :=
by
  -- This is where the proof would go.
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_for_f_to_be_odd_l862_86217


namespace NUMINAMATH_GPT_arithmetic_arrangement_result_l862_86278

theorem arithmetic_arrangement_result :
    (1 / 8) * (1 / 9) * (1 / 28) = 1 / 2016 ∨ ((1 / 8) - (1 / 9)) * (1 / 28) = 1 / 2016 :=
by {
    sorry
}

end NUMINAMATH_GPT_arithmetic_arrangement_result_l862_86278


namespace NUMINAMATH_GPT_curve_is_circle_l862_86248

theorem curve_is_circle (r : ℝ) (θ : ℝ) (h : r = 3) : 
  ∃ (c : ℝ) (p : ℝ × ℝ), c = 3 ∧ p = (3 * Real.cos θ, 3 * Real.sin θ) := 
sorry

end NUMINAMATH_GPT_curve_is_circle_l862_86248


namespace NUMINAMATH_GPT_number_of_neutrons_l862_86253

def mass_number (element : Type) : ℕ := 61
def atomic_number (element : Type) : ℕ := 27

theorem number_of_neutrons (element : Type) : mass_number element - atomic_number element = 34 :=
by
  -- Place the proof here
  sorry

end NUMINAMATH_GPT_number_of_neutrons_l862_86253


namespace NUMINAMATH_GPT_phoneExpences_l862_86233

structure PhonePlan where
  fixed_fee : ℝ
  free_minutes : ℕ
  excess_rate : ℝ -- rate per minute

def JanuaryUsage : ℕ := 15 * 60 + 17 -- 15 hours 17 minutes in minutes
def FebruaryUsage : ℕ := 9 * 60 + 55 -- 9 hours 55 minutes in minutes

def computeBill (plan : PhonePlan) (usage : ℕ) : ℝ :=
  let excess_minutes := (usage - plan.free_minutes).max 0
  plan.fixed_fee + (excess_minutes * plan.excess_rate)

theorem phoneExpences (plan : PhonePlan) :
  plan = { fixed_fee := 18.00, free_minutes := 600, excess_rate := 0.03 } →
  computeBill plan JanuaryUsage + computeBill plan FebruaryUsage = 45.51 := by
  sorry

end NUMINAMATH_GPT_phoneExpences_l862_86233


namespace NUMINAMATH_GPT_tetrahedron_edges_vertices_product_l862_86259

theorem tetrahedron_edges_vertices_product :
  let vertices := 4
  let edges := 6
  edges * vertices = 24 :=
by
  let vertices := 4
  let edges := 6
  sorry

end NUMINAMATH_GPT_tetrahedron_edges_vertices_product_l862_86259


namespace NUMINAMATH_GPT_common_number_in_sequence_l862_86242

theorem common_number_in_sequence 
  (a b c d e f g h i j : ℕ) 
  (h1 : (a + b + c + d + e) / 5 = 4) 
  (h2 : (f + g + h + i + j) / 5 = 9)
  (h3 : (a + b + c + d + e + f + g + h + i + j) / 10 = 7)
  (h4 : e = f) :
  e = 5 :=
by
  sorry

end NUMINAMATH_GPT_common_number_in_sequence_l862_86242


namespace NUMINAMATH_GPT_net_effect_sale_value_net_effect_sale_value_percentage_increase_l862_86229

def sale_value (P Q : ℝ) : ℝ := P * Q

theorem net_effect_sale_value (P Q : ℝ) :
  sale_value (0.8 * P) (1.8 * Q) = 1.44 * sale_value P Q :=
by
  sorry

theorem net_effect_sale_value_percentage_increase (P Q : ℝ) :
  (sale_value (0.8 * P) (1.8 * Q) - sale_value P Q) / sale_value P Q = 0.44 :=
by
  sorry

end NUMINAMATH_GPT_net_effect_sale_value_net_effect_sale_value_percentage_increase_l862_86229


namespace NUMINAMATH_GPT_sacks_after_6_days_l862_86226

theorem sacks_after_6_days (sacks_per_day : ℕ) (days : ℕ) 
  (h1 : sacks_per_day = 83) (h2 : days = 6) : 
  sacks_per_day * days = 498 :=
by
  sorry

end NUMINAMATH_GPT_sacks_after_6_days_l862_86226


namespace NUMINAMATH_GPT_simplify_expression_l862_86235

variable (a b : ℕ)

theorem simplify_expression (a b : ℕ) : 5 * a * b - 7 * a * b + 3 * a * b = a * b := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l862_86235


namespace NUMINAMATH_GPT_rods_in_one_mile_l862_86292

-- Definitions of the conditions
def mile_to_furlong := 10
def furlong_to_rod := 50

-- Theorem statement corresponding to the proof problem
theorem rods_in_one_mile : mile_to_furlong * furlong_to_rod = 500 := 
by sorry

end NUMINAMATH_GPT_rods_in_one_mile_l862_86292


namespace NUMINAMATH_GPT_valid_plantings_count_l862_86221

-- Define the grid structure
structure Grid3x3 :=
  (sections : Fin 9 → String)

noncomputable def crops := ["corn", "wheat", "soybeans", "potatoes", "oats"]

-- Define the adjacency relationships and restrictions as predicates
def adjacent (i j : Fin 9) : Prop :=
  (i = j + 1 ∧ j % 3 ≠ 2) ∨ (i = j - 1 ∧ i % 3 ≠ 2) ∨ (i = j + 3) ∨ (i = j - 3)

def valid_crop_planting (g : Grid3x3) : Prop :=
  ∀ i j, adjacent i j →
    (¬(g.sections i = "corn" ∧ g.sections j = "wheat") ∧ 
    ¬(g.sections i = "wheat" ∧ g.sections j = "corn") ∧
    ¬(g.sections i = "soybeans" ∧ g.sections j = "potatoes") ∧
    ¬(g.sections i = "potatoes" ∧ g.sections j = "soybeans") ∧
    ¬(g.sections i = "oats" ∧ g.sections j = "potatoes") ∧ 
    ¬(g.sections i = "potatoes" ∧ g.sections j = "oats"))

noncomputable def count_valid_plantings : Nat :=
  -- Placeholder for the actual count computing function
  sorry

theorem valid_plantings_count : count_valid_plantings = 5 :=
  sorry

end NUMINAMATH_GPT_valid_plantings_count_l862_86221


namespace NUMINAMATH_GPT_problem_solution_l862_86239

noncomputable def equilateral_triangle_area_to_perimeter_square_ratio (s : ℝ) (h : s = 10) : ℝ :=
  let altitude := s * Real.sqrt 3 / 2
  let area := 1 / 2 * s * altitude
  let perimeter := 3 * s
  let perimeter_squared := perimeter^2
  area / perimeter_squared

theorem problem_solution :
  equilateral_triangle_area_to_perimeter_square_ratio 10 rfl = Real.sqrt 3 / 36 :=
sorry

end NUMINAMATH_GPT_problem_solution_l862_86239


namespace NUMINAMATH_GPT_continuous_function_triples_l862_86246

theorem continuous_function_triples (f g h : ℝ → ℝ) (h₁ : Continuous f) (h₂ : Continuous g) (h₃ : Continuous h)
  (h₄ : ∀ x y : ℝ, f (x + y) = g x + h y) :
  ∃ (c a b : ℝ), (∀ x : ℝ, f x = c * x + a + b) ∧ (∀ x : ℝ, g x = c * x + a) ∧ (∀ x : ℝ, h x = c * x + b) :=
sorry

end NUMINAMATH_GPT_continuous_function_triples_l862_86246


namespace NUMINAMATH_GPT_max_value_of_expression_l862_86261

theorem max_value_of_expression (x y z : ℝ) (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) (h_sum : x + y + z = 3) :
  (xy / (x + y + 1) + xz / (x + z + 1) + yz / (y + z + 1)) ≤ 1 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l862_86261


namespace NUMINAMATH_GPT_coordinates_of_OC_l862_86269

-- Define the given vectors
def OP : ℝ × ℝ := (2, 1)
def OA : ℝ × ℝ := (1, 7)
def OB : ℝ × ℝ := (5, 1)

-- Define the dot product for ℝ × ℝ
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define OC as a point on line OP, parameterized by t
def OC (t : ℝ) : ℝ × ℝ := (2 * t, t)

-- Define CA and CB
def CA (t : ℝ) : ℝ × ℝ := (OA.1 - (OC t).1, OA.2 - (OC t).2)
def CB (t : ℝ) : ℝ × ℝ := (OB.1 - (OC t).1, OB.2 - (OC t).2)

-- Prove that minimization of dot_product (CA t) (CB t) occurs at OC = (4, 2)
noncomputable def find_coordinates_at_min_dot_product : Prop :=
  ∃ (t : ℝ), t = 2 ∧ OC t = (4, 2)

-- The theorem statement
theorem coordinates_of_OC : find_coordinates_at_min_dot_product :=
sorry

end NUMINAMATH_GPT_coordinates_of_OC_l862_86269


namespace NUMINAMATH_GPT_calculate_total_weight_l862_86212

-- Define the given conditions as constants and calculations
def silverware_weight_per_piece : ℕ := 4
def pieces_per_setting : ℕ := 3
def plate_weight_per_piece : ℕ := 12
def plates_per_setting : ℕ := 2
def tables : ℕ := 15
def settings_per_table : ℕ := 8
def backup_settings : ℕ := 20

-- Calculate individual weights and total settings
def silverware_weight_per_setting := silverware_weight_per_piece * pieces_per_setting
def plate_weight_per_setting := plate_weight_per_piece * plates_per_setting
def weight_per_setting := silverware_weight_per_setting + plate_weight_per_setting
def total_settings := (tables * settings_per_table) + backup_settings

-- Calculate the total weight of all settings
def total_weight : ℕ := total_settings * weight_per_setting

-- The theorem to prove that the total weight is 5040 ounces
theorem calculate_total_weight : total_weight = 5040 :=
by
  -- The proof steps are omitted
  sorry

end NUMINAMATH_GPT_calculate_total_weight_l862_86212


namespace NUMINAMATH_GPT_r_pow_four_solution_l862_86238

theorem r_pow_four_solution (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/(r^4) = 7 := 
by
  sorry

end NUMINAMATH_GPT_r_pow_four_solution_l862_86238


namespace NUMINAMATH_GPT_find_number_l862_86243

theorem find_number (N : ℝ) 
    (h : 0.20 * ((0.05)^3 * 0.35 * (0.70 * N)) = 182.7) : 
    N = 20880000 :=
by
  -- proof to be filled
  sorry

end NUMINAMATH_GPT_find_number_l862_86243


namespace NUMINAMATH_GPT_machine_value_after_two_years_l862_86282

theorem machine_value_after_two_years (initial_value : ℝ) (decrease_rate : ℝ) (years : ℕ) (value_after_two_years : ℝ) :
  initial_value = 8000 ∧ decrease_rate = 0.30 ∧ years = 2 → value_after_two_years = 3200 := by
  intros h
  sorry

end NUMINAMATH_GPT_machine_value_after_two_years_l862_86282


namespace NUMINAMATH_GPT_zongzi_profit_l862_86251

def initial_cost : ℕ := 10
def initial_price : ℕ := 16
def initial_bags_sold : ℕ := 200
def additional_sales_per_yuan (x : ℕ) : ℕ := 80 * x
def profit_per_bag (x : ℕ) : ℕ := initial_price - x - initial_cost
def number_of_bags_sold (x : ℕ) : ℕ := initial_bags_sold + additional_sales_per_yuan x
def total_profit (profit_per_bag : ℕ) (number_of_bags_sold : ℕ) : ℕ := profit_per_bag * number_of_bags_sold

theorem zongzi_profit (x : ℕ) : 
  total_profit (profit_per_bag x) (number_of_bags_sold x) = 1440 := 
sorry

end NUMINAMATH_GPT_zongzi_profit_l862_86251


namespace NUMINAMATH_GPT_paper_clips_in_two_cases_l862_86287

-- Defining the problem statement in Lean 4
theorem paper_clips_in_two_cases (c b : ℕ) :
  2 * (c * b * 300) = 2 * c * b * 300 :=
by
  sorry

end NUMINAMATH_GPT_paper_clips_in_two_cases_l862_86287


namespace NUMINAMATH_GPT_find_number_l862_86218

-- Define the number x that satisfies the given condition
theorem find_number (x : ℤ) (h : x + 12 - 27 = 24) : x = 39 :=
by {
  -- This is where the proof steps will go, but we'll use sorry to indicate it's incomplete
  sorry
}

end NUMINAMATH_GPT_find_number_l862_86218


namespace NUMINAMATH_GPT_rita_remaining_money_l862_86268

-- Defining the conditions
def num_dresses := 5
def price_dress := 20
def num_pants := 3
def price_pant := 12
def num_jackets := 4
def price_jacket := 30
def transport_cost := 5
def initial_amount := 400

-- Calculating the total cost
def total_cost : ℕ :=
  (num_dresses * price_dress) + 
  (num_pants * price_pant) + 
  (num_jackets * price_jacket) + 
  transport_cost

-- Stating the proof problem 
theorem rita_remaining_money : initial_amount - total_cost = 139 := by
  sorry

end NUMINAMATH_GPT_rita_remaining_money_l862_86268


namespace NUMINAMATH_GPT_midpoint_coord_sum_l862_86256

theorem midpoint_coord_sum (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 10) (hy1 : y1 = -2) (hx2 : x2 = -4) (hy2 : y2 = 8)
: (x1 + x2) / 2 + (y1 + y2) / 2 = 6 :=
by
  rw [hx1, hx2, hy1, hy2]
  /-
  Have (10 + (-4)) / 2 + (-2 + 8) / 2 = (6 / 2) + (6 / 2)
  Prove that (6 / 2) + (6 / 2) = 6
  -/
  sorry

end NUMINAMATH_GPT_midpoint_coord_sum_l862_86256


namespace NUMINAMATH_GPT_smallest_positive_integer_satisfying_conditions_l862_86262

theorem smallest_positive_integer_satisfying_conditions :
  ∃ (N : ℕ), N = 242 ∧
    ( ∃ (i : Fin 4), (N + i) % 8 = 0 ) ∧
    ( ∃ (i : Fin 4), (N + i) % 9 = 0 ) ∧
    ( ∃ (i : Fin 4), (N + i) % 25 = 0 ) ∧
    ( ∃ (i : Fin 4), (N + i) % 121 = 0 ) :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_satisfying_conditions_l862_86262


namespace NUMINAMATH_GPT_problem1_problem2_l862_86220

-- Problem 1
theorem problem1 : 3 * (Real.sqrt 3 + Real.sqrt 2) - 2 * (Real.sqrt 3 - Real.sqrt 2) = Real.sqrt 3 + 5 * Real.sqrt 2 :=
by
  sorry

-- Problem 2
theorem problem2 : abs (Real.sqrt 3 - Real.sqrt 2) + abs (Real.sqrt 3 - 2) + Real.sqrt 4 = 4 - Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l862_86220


namespace NUMINAMATH_GPT_flea_never_lands_on_all_points_l862_86288

noncomputable def a_n (n : ℕ) : ℕ := (n * (n + 1) / 2) % 300

theorem flea_never_lands_on_all_points :
  ∃ k : ℕ, k < 300 ∧ ∀ n : ℕ, a_n n ≠ k :=
sorry

end NUMINAMATH_GPT_flea_never_lands_on_all_points_l862_86288


namespace NUMINAMATH_GPT_average_income_correct_l862_86263

def incomes : List ℕ := [250, 400, 750, 400, 500]

noncomputable def average : ℕ := (incomes.sum) / incomes.length

theorem average_income_correct : average = 460 :=
by 
  sorry

end NUMINAMATH_GPT_average_income_correct_l862_86263


namespace NUMINAMATH_GPT_kia_vehicle_count_l862_86295

theorem kia_vehicle_count (total_vehicles : Nat) (dodge_vehicles : Nat) (hyundai_vehicles : Nat) 
    (h1 : total_vehicles = 400)
    (h2 : dodge_vehicles = total_vehicles / 2)
    (h3 : hyundai_vehicles = dodge_vehicles / 2) : 
    (total_vehicles - dodge_vehicles - hyundai_vehicles) = 100 := 
by sorry

end NUMINAMATH_GPT_kia_vehicle_count_l862_86295


namespace NUMINAMATH_GPT_find_b_squared_l862_86275

theorem find_b_squared
    (b : ℝ)
    (c_ellipse c_hyperbola a_ellipse a2_hyperbola b2_hyperbola : ℝ)
    (h1: a_ellipse^2 = 25)
    (h2 : b2_hyperbola = 9 / 4)
    (h3 : a2_hyperbola = 4)
    (h4 : c_hyperbola = Real.sqrt (a2_hyperbola + b2_hyperbola))
    (h5 : c_ellipse = c_hyperbola)
    (h6 : b^2 = a_ellipse^2 - c_ellipse^2)
: b^2 = 75 / 4 :=
sorry

end NUMINAMATH_GPT_find_b_squared_l862_86275


namespace NUMINAMATH_GPT_cube_volume_given_surface_area_l862_86296

theorem cube_volume_given_surface_area (A : ℝ) (V : ℝ) :
  A = 96 → V = 64 :=
by
  sorry

end NUMINAMATH_GPT_cube_volume_given_surface_area_l862_86296


namespace NUMINAMATH_GPT_messages_tuesday_l862_86274

theorem messages_tuesday (T : ℕ) (h1 : 300 + T + (T + 300) + 2 * (T + 300) = 2000) : 
  T = 200 := by
  sorry

end NUMINAMATH_GPT_messages_tuesday_l862_86274


namespace NUMINAMATH_GPT_sequence_length_l862_86257

theorem sequence_length (a : ℕ) (h : a = 10800) (h1 : ∀ n, (n ≠ 0 → ∃ m, n = 2 * m ∧ m ≠ 0) ∧ 2 ∣ n)
  : ∃ k : ℕ, k = 5 := 
sorry

end NUMINAMATH_GPT_sequence_length_l862_86257


namespace NUMINAMATH_GPT_find_value_of_k_l862_86299

noncomputable def line_parallel_and_point_condition (k : ℝ) :=
  ∃ (m : ℝ), m = -5/4 ∧ (22 - (-8)) / (k - 3) = m

theorem find_value_of_k : ∃ k : ℝ, line_parallel_and_point_condition k ∧ k = -21 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_k_l862_86299


namespace NUMINAMATH_GPT_find_birth_rate_l862_86241

noncomputable def average_birth_rate (B : ℕ) : Prop :=
  let death_rate := 3
  let net_increase_per_2_seconds := B - death_rate
  let seconds_per_hour := 3600
  let hours_per_day := 24
  let seconds_per_day := seconds_per_hour * hours_per_day
  let net_increase_times := seconds_per_day / 2
  let total_net_increase := net_increase_times * net_increase_per_2_seconds
  total_net_increase = 172800

theorem find_birth_rate (B : ℕ) (h : average_birth_rate B) : B = 7 :=
  sorry

end NUMINAMATH_GPT_find_birth_rate_l862_86241


namespace NUMINAMATH_GPT_tan_alpha_value_l862_86211

theorem tan_alpha_value (α : ℝ) (h1 : Real.sin α = 3 / 5) (h2 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) : Real.tan α = -3 / 4 := 
sorry

end NUMINAMATH_GPT_tan_alpha_value_l862_86211


namespace NUMINAMATH_GPT_Molly_age_now_l862_86215

/- Definitions -/
def Sandy_curr_age : ℕ := 60
def Molly_curr_age (S : ℕ) : ℕ := 3 * S / 4
def Sandy_age_in_6_years (S : ℕ) : ℕ := S + 6

/- Theorem to prove -/
theorem Molly_age_now 
  (ratio_condition : ∀ S M : ℕ, S / M = 4 / 3 → M = 3 * S / 4)
  (age_condition : Sandy_age_in_6_years Sandy_curr_age = 66) : 
  Molly_curr_age Sandy_curr_age = 45 :=
by
  sorry

end NUMINAMATH_GPT_Molly_age_now_l862_86215


namespace NUMINAMATH_GPT_quoted_price_of_shares_l862_86280

theorem quoted_price_of_shares (investment : ℝ) (face_value : ℝ) (rate_dividend : ℝ) (annual_income : ℝ) (num_shares : ℝ) (quoted_price : ℝ) :
  investment = 4455 ∧ face_value = 10 ∧ rate_dividend = 0.12 ∧ annual_income = 648 ∧ num_shares = annual_income / (rate_dividend * face_value) →
  quoted_price = investment / num_shares :=
by sorry

end NUMINAMATH_GPT_quoted_price_of_shares_l862_86280


namespace NUMINAMATH_GPT_range_of_a_l862_86293

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l862_86293


namespace NUMINAMATH_GPT_chord_length_l862_86245

-- Define radii of the circles
def r1 : ℝ := 5
def r2 : ℝ := 12
def r3 : ℝ := r1 + r2

-- Define the centers of the circles
variable (O1 O2 O3 : ℝ)

-- Define the points of tangency and foot of the perpendicular
def T1 : ℝ := O1 + r1
def T2 : ℝ := O2 + r2
def T : ℝ := O3 - r3

-- Given the conditions
theorem chord_length (m n p : ℤ) : 
  (∃ (C1 C2 C3 : ℝ) (tangent1 tangent2 : ℝ),
    C1 = r1 ∧ C2 = r2 ∧ C3 = r3 ∧
    -- Externally tangent: distance between centers of C1 and C2 is r1 + r2
    dist O1 O2 = r1 + r2 ∧
    -- Internally tangent: both C1 and C2 are tangent to C3
    dist O1 O3 = r3 - r1 ∧
    dist O2 O3 = r3 - r2 ∧
    -- The chord in C3 is a common external tangent to C1 and C2
    tangent1 = O3 + ((O1 * O2) - (O1 * O3)) / r1 ∧
    tangent2 = O3 + ((O2 * O1) - (O2 * O3)) / r2 ∧
    m = 10 ∧ n = 546 ∧ p = 7 ∧
    m + n + p = 563)
  := sorry

end NUMINAMATH_GPT_chord_length_l862_86245


namespace NUMINAMATH_GPT_base4_more_digits_than_base9_l862_86206

def base4_digits_1234 : ℕ := 6
def base9_digits_1234 : ℕ := 4

theorem base4_more_digits_than_base9 :
  base4_digits_1234 - base9_digits_1234 = 2 :=
by
  sorry

end NUMINAMATH_GPT_base4_more_digits_than_base9_l862_86206


namespace NUMINAMATH_GPT_difference_increased_decreased_l862_86204

theorem difference_increased_decreased (x : ℝ) (hx : x = 80) : 
  ((x * 1.125) - (x * 0.75)) = 30 := by
  have h1 : x * 1.125 = 90 := by rw [hx]; norm_num
  have h2 : x * 0.75 = 60 := by rw [hx]; norm_num
  rw [h1, h2]
  norm_num
  done

end NUMINAMATH_GPT_difference_increased_decreased_l862_86204


namespace NUMINAMATH_GPT_theta_value_l862_86224

theorem theta_value (Theta : ℕ) (h_digit : Θ < 10) (h_eq : 252 / Θ = 30 + 2 * Θ) : Θ = 6 := 
by
  sorry

end NUMINAMATH_GPT_theta_value_l862_86224


namespace NUMINAMATH_GPT_min_ab_value_l862_86205

theorem min_ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 1 / a + 4 / b = Real.sqrt (a * b)) :
  a * b = 4 :=
  sorry

end NUMINAMATH_GPT_min_ab_value_l862_86205


namespace NUMINAMATH_GPT_melted_mixture_weight_l862_86298

theorem melted_mixture_weight (Z C : ℝ) (ratio : 9 / 11 = Z / C) (zinc_weight : Z = 28.8) : Z + C = 64 :=
by
  sorry

end NUMINAMATH_GPT_melted_mixture_weight_l862_86298


namespace NUMINAMATH_GPT_continuous_stripe_encircling_tetrahedron_probability_l862_86202

noncomputable def tetrahedron_continuous_stripe_probability : ℚ :=
  let total_combinations := 3^4
  let favorable_combinations := 2 
  favorable_combinations / total_combinations

theorem continuous_stripe_encircling_tetrahedron_probability :
  tetrahedron_continuous_stripe_probability = 2 / 81 :=
by
  -- the proof would be here
  sorry

end NUMINAMATH_GPT_continuous_stripe_encircling_tetrahedron_probability_l862_86202


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l862_86244

theorem arithmetic_sequence_sum :
  let a₁ := 3
  let d := 4
  let n := 30
  let aₙ := a₁ + (n - 1) * d
  let Sₙ := (n / 2) * (a₁ + aₙ)
  Sₙ = 1830 :=
by
  intros
  let a₁ := 3
  let d := 4
  let n := 30
  let aₙ := a₁ + (n - 1) * d
  let Sₙ := (n / 2) * (a₁ + aₙ)
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l862_86244


namespace NUMINAMATH_GPT_find_special_integer_l862_86281

theorem find_special_integer :
  ∃ (n : ℕ), n > 0 ∧ (21 ∣ n) ∧ 30 ≤ Real.sqrt n ∧ Real.sqrt n ≤ 30.5 ∧ n = 903 := 
sorry

end NUMINAMATH_GPT_find_special_integer_l862_86281


namespace NUMINAMATH_GPT_number_of_clothes_hangers_l862_86210

noncomputable def total_money : ℝ := 60
noncomputable def spent_on_tissues : ℝ := 34.8
noncomputable def price_per_hanger : ℝ := 1.6

theorem number_of_clothes_hangers : 
  let remaining_money := total_money - spent_on_tissues
  let hangers := remaining_money / price_per_hanger
  Int.floor hangers = 15 := 
by
  sorry

end NUMINAMATH_GPT_number_of_clothes_hangers_l862_86210


namespace NUMINAMATH_GPT_sticks_left_is_correct_l862_86294

-- Define the initial conditions
def initial_popsicle_sticks : ℕ := 170
def popsicle_sticks_per_group : ℕ := 15
def number_of_groups : ℕ := 10

-- Define the total number of popsicle sticks given out to the groups
def total_sticks_given : ℕ := popsicle_sticks_per_group * number_of_groups

-- Define the number of popsicle sticks left
def sticks_left : ℕ := initial_popsicle_sticks - total_sticks_given

-- Prove that the number of sticks left is 20
theorem sticks_left_is_correct : sticks_left = 20 :=
by
  sorry

end NUMINAMATH_GPT_sticks_left_is_correct_l862_86294


namespace NUMINAMATH_GPT_speed_of_A_is_24_speed_of_A_is_18_l862_86203

-- Definitions for part 1
def speed_of_B (x : ℝ) := x
def speed_of_A_1 (x : ℝ) := 1.2 * x
def distance_AB := 30 -- kilometers
def distance_B_rides_first := 2 -- kilometers
def time_A_catches_up := 0.5 -- hours

theorem speed_of_A_is_24 (x : ℝ) (h1 : 0.6 * x = 2 + 0.5 * x) : speed_of_A_1 x = 24 := by
  sorry

-- Definitions for part 2
def speed_of_A_2 (y : ℝ) := 1.2 * y
def time_B_rides_first := 1/3 -- hours
def time_difference := 1/3 -- hours

theorem speed_of_A_is_18 (y : ℝ) (h2 : (30 / y) - (30 / (1.2 * y)) = 1/3) : speed_of_A_2 y = 18 := by
  sorry

end NUMINAMATH_GPT_speed_of_A_is_24_speed_of_A_is_18_l862_86203


namespace NUMINAMATH_GPT_harmonic_arithmetic_sequence_common_difference_l862_86297

theorem harmonic_arithmetic_sequence_common_difference (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) : 
  (∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * d)) →
  (∀ n, a n = a 1 + (n - 1) * d) →
  (a 1 = 1) →
  (d ≠ 0) →
  (∃ k, ∀ n, S n / S (2 * n) = k) →
  d = 2 :=
by
  sorry

end NUMINAMATH_GPT_harmonic_arithmetic_sequence_common_difference_l862_86297


namespace NUMINAMATH_GPT_digit_problem_l862_86279

variable {x y : ℕ}

theorem digit_problem (h1 : 10 * x + y - (10 * y + x) = 36) (h2 : x * 2 = y) :
  (x + y) - (x - y) = 16 :=
by sorry

end NUMINAMATH_GPT_digit_problem_l862_86279


namespace NUMINAMATH_GPT_gary_current_weekly_eggs_l862_86276

noncomputable def egg_laying_rates : List ℕ := [6, 5, 7, 4]

def total_eggs_per_day (rates : List ℕ) : ℕ :=
  rates.foldl (· + ·) 0

def total_eggs_per_week (eggs_per_day : ℕ) : ℕ :=
  eggs_per_day * 7

theorem gary_current_weekly_eggs : 
  total_eggs_per_week (total_eggs_per_day egg_laying_rates) = 154 :=
by
  sorry

end NUMINAMATH_GPT_gary_current_weekly_eggs_l862_86276


namespace NUMINAMATH_GPT_number_of_integers_divisible_by_18_or_21_but_not_both_l862_86284

theorem number_of_integers_divisible_by_18_or_21_but_not_both :
  let num_less_2019_div_by_18 := 112
  let num_less_2019_div_by_21 := 96
  let num_less_2019_div_by_both := 16
  num_less_2019_div_by_18 + num_less_2019_div_by_21 - 2 * num_less_2019_div_by_both = 176 :=
by
  sorry

end NUMINAMATH_GPT_number_of_integers_divisible_by_18_or_21_but_not_both_l862_86284


namespace NUMINAMATH_GPT_correct_negation_l862_86222

-- Define a triangle with angles A, B, and C
variables (α β γ : ℝ)

-- Define properties of the angles
def is_triangle (α β γ : ℝ) : Prop := α + β + γ = 180
def is_right_angle (angle : ℝ) : Prop := angle = 90
def is_acute_angle (angle : ℝ) : Prop := angle > 0 ∧ angle < 90

-- Original statement to be negated
def original_statement (α β γ : ℝ) : Prop := 
  is_triangle α β γ ∧ is_right_angle γ → is_acute_angle α ∧ is_acute_angle β

-- Negation of the original statement
def negated_statement (α β γ : ℝ) : Prop := 
  is_triangle α β γ ∧ ¬ is_right_angle γ → ¬ (is_acute_angle α ∧ is_acute_angle β)

-- Proof statement: prove that the negated statement is the correct negation
theorem correct_negation (α β γ : ℝ) :
  negated_statement α β γ = ¬ original_statement α β γ :=
sorry

end NUMINAMATH_GPT_correct_negation_l862_86222


namespace NUMINAMATH_GPT_distance_from_neg2_eq4_l862_86271

theorem distance_from_neg2_eq4 (x : ℤ) : |x + 2| = 4 ↔ x = 2 ∨ x = -6 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_neg2_eq4_l862_86271


namespace NUMINAMATH_GPT_base_any_number_l862_86208

open Nat

theorem base_any_number (n k : ℕ) (h1 : k ≥ 0) (h2 : (30 ^ k) ∣ 929260) (h3 : n ^ k - k ^ 3 = 1) : true :=
by
  sorry

end NUMINAMATH_GPT_base_any_number_l862_86208


namespace NUMINAMATH_GPT_maximize_profit_l862_86273

noncomputable def g (x : ℝ) : ℝ :=
if h : (0 < x ∧ x ≤ 10) then 13.5 - (1 / 30) * x^2
else if h : (x > 10) then 168 / x - 2000 / (3 * x^2)
else 0 -- default case included for totality

noncomputable def y (x : ℝ) : ℝ :=
if h : (0 < x ∧ x ≤ 10) then 8.1 * x - (1 / 30) * x^3 - 20
else if h : (x > 10) then 148 - 2 * (1000 / (3 * x) + 2.7 * x)
else 0 -- default case included for totality

theorem maximize_profit (x : ℝ) : 0 < x → y 9 = 28.6 :=
by sorry

end NUMINAMATH_GPT_maximize_profit_l862_86273


namespace NUMINAMATH_GPT_findValuesForFibSequence_l862_86264

noncomputable def maxConsecutiveFibonacciTerms (A B C : ℝ) : ℝ :=
  if A ≠ 0 then 4 else 0

theorem findValuesForFibSequence :
  maxConsecutiveFibonacciTerms (1/2) (-1/2) 2 = 4 ∧ maxConsecutiveFibonacciTerms (1/2) (1/2) 2 = 4 :=
by
  -- This statement will follow from the given conditions and the solution provided.
  sorry

end NUMINAMATH_GPT_findValuesForFibSequence_l862_86264


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l862_86223

-- Given conditions
variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_periodic : ∀ x, f (x - 4) = -f x)
variable (h_increasing : ∀ x y : ℝ, 0 ≤ x → x ≤ 2 → x ≤ y → y ≤ 2 → f x ≤ f y)

-- Problem statements
theorem problem1 : f 2012 = 0 := sorry

theorem problem2 : ∀ x, f (4 - x) = -f (4 + x) := sorry

theorem problem3 : f (-25) < f 80 ∧ f 80 < f 11 := sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l862_86223


namespace NUMINAMATH_GPT_square_can_be_divided_into_40_smaller_squares_l862_86258

theorem square_can_be_divided_into_40_smaller_squares 
: ∃ (n : ℕ), n * n = 40 := 
sorry

end NUMINAMATH_GPT_square_can_be_divided_into_40_smaller_squares_l862_86258


namespace NUMINAMATH_GPT_find_value_of_2a10_minus_a12_l862_86291

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the given conditions
def condition (a : ℕ → ℝ) : Prop :=
  is_arithmetic_sequence a ∧ (a 4 + a 6 + a 8 + a 10 + a 12 = 120)

-- State the theorem
theorem find_value_of_2a10_minus_a12 (a : ℕ → ℝ) (h : condition a) : 2 * a 10 - a 12 = 24 :=
by sorry

end NUMINAMATH_GPT_find_value_of_2a10_minus_a12_l862_86291


namespace NUMINAMATH_GPT_base_conversion_correct_l862_86267

def convert_base_9_to_10 (n : ℕ) : ℕ :=
  3 * 9^2 + 6 * 9^1 + 1 * 9^0

def convert_base_13_to_10 (n : ℕ) (C : ℕ) : ℕ :=
  4 * 13^2 + C * 13^1 + 5 * 13^0

theorem base_conversion_correct :
  convert_base_9_to_10 361 + convert_base_13_to_10 4 12 = 1135 :=
by
  sorry

end NUMINAMATH_GPT_base_conversion_correct_l862_86267


namespace NUMINAMATH_GPT_douglas_votes_in_county_X_l862_86260

theorem douglas_votes_in_county_X (V : ℝ) :
  (0.64 * (2 * V + V) - 0.4000000000000002 * V) / (2 * V) * 100 = 76 := by
sorry

end NUMINAMATH_GPT_douglas_votes_in_county_X_l862_86260


namespace NUMINAMATH_GPT_michael_pets_kangaroos_l862_86213

theorem michael_pets_kangaroos :
  let total_pets := 24
  let fraction_dogs := 1 / 8
  let fraction_not_cows := 3 / 4
  let fraction_not_cats := 2 / 3
  let num_dogs := fraction_dogs * total_pets
  let num_cows := (1 - fraction_not_cows) * total_pets
  let num_cats := (1 - fraction_not_cats) * total_pets
  let num_kangaroos := total_pets - num_dogs - num_cows - num_cats
  num_kangaroos = 7 :=
by
  sorry

end NUMINAMATH_GPT_michael_pets_kangaroos_l862_86213


namespace NUMINAMATH_GPT_conjugate_system_solution_l862_86237

theorem conjugate_system_solution (a b : ℝ) :
  (∀ x y : ℝ,
    (x + (2-a) * y = b + 1) ∧ ((2*a-7) * x + y = -5 - b)
    ↔ x + (2*a-7) * y = -5 - b ∧ (x + (2-a) * y = b + 1))
  ↔ a = 3 ∧ b = -3 := by
  sorry

end NUMINAMATH_GPT_conjugate_system_solution_l862_86237


namespace NUMINAMATH_GPT_find_special_N_l862_86252

theorem find_special_N : ∃ N : ℕ, 
  (Nat.digits 10 N).length = 1112 ∧
  (Nat.digits 10 N).sum % 2000 = 0 ∧
  (Nat.digits 10 (N + 1)).sum % 2000 = 0 ∧
  (Nat.digits 10 N).contains 1 ∧
  (N = 9 * 10^1111 + 1 * 10^221 + 9 * (10^220 - 1) / 9 + 10^890 - 1) :=
sorry

end NUMINAMATH_GPT_find_special_N_l862_86252


namespace NUMINAMATH_GPT_number_of_male_animals_l862_86249

def total_original_animals : ℕ := 100 + 29 + 9
def animals_bought_by_brian : ℕ := total_original_animals / 2
def animals_after_brian : ℕ := total_original_animals - animals_bought_by_brian
def animals_after_jeremy : ℕ := animals_after_brian + 37

theorem number_of_male_animals : animals_after_jeremy / 2 = 53 :=
by
  sorry

end NUMINAMATH_GPT_number_of_male_animals_l862_86249


namespace NUMINAMATH_GPT_avg_age_decrease_l862_86231

/-- Define the original average age of the class -/
def original_avg_age : ℕ := 40

/-- Define the number of original students -/
def original_strength : ℕ := 17

/-- Define the average age of the new students -/
def new_students_avg_age : ℕ := 32

/-- Define the number of new students joining -/
def new_students_strength : ℕ := 17

/-- Define the total original age of the class -/
def total_original_age : ℕ := original_strength * original_avg_age

/-- Define the total age of the new students -/
def total_new_students_age : ℕ := new_students_strength * new_students_avg_age

/-- Define the new total strength of the class after joining of new students -/
def new_total_strength : ℕ := original_strength + new_students_strength

/-- Define the new total age of the class after joining of new students -/
def new_total_age : ℕ := total_original_age + total_new_students_age

/-- Define the new average age of the class -/
def new_avg_age : ℕ := new_total_age / new_total_strength

/-- Prove that the average age decreased by 4 years when the new students joined -/
theorem avg_age_decrease : original_avg_age - new_avg_age = 4 := by
  sorry

end NUMINAMATH_GPT_avg_age_decrease_l862_86231


namespace NUMINAMATH_GPT_ratio_calc_l862_86225

theorem ratio_calc :
  (14^4 + 484) * (26^4 + 484) * (38^4 + 484) * (50^4 + 484) * (62^4 + 484) /
  ((8^4 + 484) * (20^4 + 484) * (32^4 + 484) * (44^4 + 484) * (56^4 + 484)) = -423 := 
by
  sorry

end NUMINAMATH_GPT_ratio_calc_l862_86225


namespace NUMINAMATH_GPT_unique_four_digit_numbers_l862_86232

theorem unique_four_digit_numbers (digits : Finset ℕ) (odd_digits : Finset ℕ) :
  digits = {2, 3, 4, 5, 6} → 
  odd_digits = {3, 5} → 
  ∃ (n : ℕ), n = 14 :=
by
  sorry

end NUMINAMATH_GPT_unique_four_digit_numbers_l862_86232
