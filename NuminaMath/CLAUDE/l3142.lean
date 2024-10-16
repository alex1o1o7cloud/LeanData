import Mathlib

namespace NUMINAMATH_CALUDE_tan_to_sin_cos_ratio_l3142_314227

theorem tan_to_sin_cos_ratio (α : Real) (h : Real.tan α = 2) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_to_sin_cos_ratio_l3142_314227


namespace NUMINAMATH_CALUDE_parent_son_age_ratio_l3142_314248

/-- The ratio of a parent's age to their son's age -/
def age_ratio (parent_age : ℕ) (son_age : ℕ) : ℚ :=
  parent_age / son_age

theorem parent_son_age_ratio :
  let parent_age : ℕ := 35
  let son_age : ℕ := 7
  age_ratio parent_age son_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_parent_son_age_ratio_l3142_314248


namespace NUMINAMATH_CALUDE_consistency_condition_l3142_314259

theorem consistency_condition (a b c d x y z : ℝ) 
  (eq1 : y + z = a)
  (eq2 : x + y = b)
  (eq3 : x + z = c)
  (eq4 : x + y + z = d) :
  a + b + c = 2 * d := by
  sorry

end NUMINAMATH_CALUDE_consistency_condition_l3142_314259


namespace NUMINAMATH_CALUDE_fraction_equality_l3142_314230

theorem fraction_equality (a b : ℝ) (h : a / b = 5 / 4) :
  (4 * a + 3 * b) / (4 * a - 3 * b) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3142_314230


namespace NUMINAMATH_CALUDE_number_divided_by_ratio_l3142_314257

theorem number_divided_by_ratio (x : ℝ) : 
  0.55 * x = 4.235 → x / 0.55 = 14 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_ratio_l3142_314257


namespace NUMINAMATH_CALUDE_money_exchange_equations_l3142_314224

/-- Represents the money exchange problem from "Nine Chapters on the Mathematical Art" --/
theorem money_exchange_equations (x y : ℝ) : 
  (x + 1/2 * y = 50 ∧ y + 2/3 * x = 50) ↔ 
  (∃ (a b : ℝ), 
    a = x ∧ 
    b = y ∧ 
    a + 1/2 * b = 50 ∧ 
    b + 2/3 * a = 50) :=
by sorry

end NUMINAMATH_CALUDE_money_exchange_equations_l3142_314224


namespace NUMINAMATH_CALUDE_parabola_properties_incorrect_statement_l3142_314208

-- Define the parabola
def parabola (x : ℝ) : ℝ := -(x - 1)^2 + 4

-- Statements to prove
theorem parabola_properties :
  -- The parabola opens downwards
  (∀ x : ℝ, parabola x ≤ parabola 1) ∧
  -- The shape is the same as y = x^2
  (∃ c : ℝ, ∀ x : ℝ, parabola x = c - x^2) ∧
  -- The vertex is (1,4)
  (parabola 1 = 4 ∧ ∀ x : ℝ, parabola x ≤ 4) ∧
  -- The axis of symmetry is the line x = 1
  (∀ x : ℝ, parabola (1 + x) = parabola (1 - x)) :=
by sorry

-- Statement C is incorrect
theorem incorrect_statement :
  ¬(parabola (-1) = 4 ∧ ∀ x : ℝ, parabola x ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_incorrect_statement_l3142_314208


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l3142_314240

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ) : 
  (∀ x : ℤ, (2 - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 665 := by
sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l3142_314240


namespace NUMINAMATH_CALUDE_inequality_range_l3142_314261

theorem inequality_range : 
  ∀ x : ℝ, (∀ a b : ℝ, a > 0 ∧ b > 0 → x^2 + x < a/b + b/a) ↔ -2 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l3142_314261


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3142_314253

/-- Given an equilateral triangle with perimeter 60 and an isosceles triangle with perimeter 55,
    where one side of the equilateral triangle is a side of the isosceles triangle,
    the base of the isosceles triangle is 15 units long. -/
theorem isosceles_triangle_base_length
  (equilateral_perimeter : ℝ)
  (isosceles_perimeter : ℝ)
  (h_equilateral_perimeter : equilateral_perimeter = 60)
  (h_isosceles_perimeter : isosceles_perimeter = 55)
  (h_shared_side : equilateral_perimeter / 3 = isosceles_perimeter / 2 - isosceles_base / 2) :
  isosceles_base = 15 :=
by
  sorry

#check isosceles_triangle_base_length

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3142_314253


namespace NUMINAMATH_CALUDE_parabola_equation_l3142_314296

-- Define the parabola type
structure Parabola where
  -- The equation of the parabola is either y² = ax or x² = by
  a : ℝ
  b : ℝ
  along_x_axis : Bool

-- Define the properties of the parabola
def satisfies_conditions (p : Parabola) : Prop :=
  -- Vertex at origin (implied by the standard form of equation)
  -- Axis of symmetry along one of the coordinate axes (implied by the structure)
  -- Passes through the point (-2, 3)
  (p.along_x_axis ∧ 3^2 = -p.a * (-2)) ∨
  (¬p.along_x_axis ∧ (-2)^2 = p.b * 3)

-- Theorem statement
theorem parabola_equation :
  ∀ p : Parabola, satisfies_conditions p →
    (p.along_x_axis ∧ p.a = -9/2) ∨ (¬p.along_x_axis ∧ p.b = 4/3) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l3142_314296


namespace NUMINAMATH_CALUDE_reverse_increase_l3142_314219

def reverse_number (n : ℕ) : ℕ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d3 * 100 + d2 * 10 + d1

theorem reverse_increase (n : ℕ) : 
  n = 253 → 
  (n / 100 + (n / 10) % 10 + n % 10 = 10) → 
  ((n / 10) % 10 = (n / 100 + n % 10)) → 
  reverse_number n - n = 99 :=
by sorry

end NUMINAMATH_CALUDE_reverse_increase_l3142_314219


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l3142_314226

def f (x : ℝ) : ℝ := x^2014

theorem triangle_angle_inequality (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : α + β > π/2) : 
  f (Real.sin α) > f (Real.cos β) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l3142_314226


namespace NUMINAMATH_CALUDE_function_value_at_negative_l3142_314214

theorem function_value_at_negative (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x + 1 / x - 1) (h2 : f a = 2) :
  f (-a) = -4 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_negative_l3142_314214


namespace NUMINAMATH_CALUDE_flower_shop_cost_l3142_314221

/-- The total cost of buying roses and lilies with given conditions -/
theorem flower_shop_cost : 
  let num_roses : ℕ := 20
  let num_lilies : ℕ := (3 * num_roses) / 4
  let cost_per_rose : ℕ := 5
  let cost_per_lily : ℕ := 2 * cost_per_rose
  let total_cost : ℕ := num_roses * cost_per_rose + num_lilies * cost_per_lily
  total_cost = 250 := by
  sorry

end NUMINAMATH_CALUDE_flower_shop_cost_l3142_314221


namespace NUMINAMATH_CALUDE_probability_theorem_l3142_314206

-- Define the probabilities of events A1, A2, A3
def P_A1 : ℚ := 1/2
def P_A2 : ℚ := 1/5
def P_A3 : ℚ := 3/10

-- Define the conditional probabilities
def P_B_given_A1 : ℚ := 5/11
def P_B_given_A2 : ℚ := 4/11
def P_B_given_A3 : ℚ := 4/11

-- Define the probability of event B
def P_B : ℚ := 9/22

-- Define the theorem to be proved
theorem probability_theorem :
  (P_B_given_A1 = 5/11) ∧
  (P_B = 9/22) ∧
  (P_A1 + P_A2 + P_A3 = 1) := by
  sorry

#check probability_theorem

end NUMINAMATH_CALUDE_probability_theorem_l3142_314206


namespace NUMINAMATH_CALUDE_arithmetic_sequence_120th_term_l3142_314258

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 120th term of the specific arithmetic sequence -/
def term_120 : ℝ :=
  arithmetic_sequence 6 6 120

theorem arithmetic_sequence_120th_term :
  term_120 = 720 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_120th_term_l3142_314258


namespace NUMINAMATH_CALUDE_henry_margo_meeting_l3142_314211

/-- The time it takes Henry to complete one lap around the track -/
def henry_lap_time : ℕ := 7

/-- The time it takes Margo to complete one lap around the track -/
def margo_lap_time : ℕ := 12

/-- The time when Henry and Margo meet at the starting line -/
def meeting_time : ℕ := 84

theorem henry_margo_meeting :
  Nat.lcm henry_lap_time margo_lap_time = meeting_time := by
  sorry

end NUMINAMATH_CALUDE_henry_margo_meeting_l3142_314211


namespace NUMINAMATH_CALUDE_min_sum_squares_l3142_314209

theorem min_sum_squares (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 3) :
  ∃ (m : ℝ), m = 9 ∧ ∀ (a b c : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → a + b + c = 3 → a^2 + b^2 + c^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3142_314209


namespace NUMINAMATH_CALUDE_unique_pair_solution_l3142_314256

theorem unique_pair_solution : 
  ∃! (x y : ℕ), 
    x > 0 ∧ y > 0 ∧  -- Positive integers
    y ≥ x ∧          -- y ≥ x
    x + y ≤ 20 ∧     -- Sum constraint
    ¬(Nat.Prime (x * y)) ∧  -- Product is composite
    (∀ (a b : ℕ), a > 0 ∧ b > 0 ∧ b ≥ a ∧ a + b ≤ 20 ∧ a * b = x * y → a + b = x + y) ∧  -- Unique sum given product and constraints
    x = 2 ∧ y = 11 :=
by sorry

end NUMINAMATH_CALUDE_unique_pair_solution_l3142_314256


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3142_314200

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptote
def asymptote (x y : ℝ) : Prop :=
  2 * x - y = 0

-- Theorem statement
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ x y, hyperbola a b x y ∧ asymptote x y) →
  let c := Real.sqrt (a^2 + b^2)
  c / a = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3142_314200


namespace NUMINAMATH_CALUDE_problem_solution_l3142_314207

-- Define proposition p
def p : Prop := ∃ x : ℝ, Real.tan x = 1

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 - 3*x + 2 < 0 ↔ 1 < x ∧ x < 2

-- Theorem to prove
theorem problem_solution :
  (p ∧ q) ∧
  ¬(p ∧ ¬q) ∧
  (¬p ∨ q) ∧
  ¬(¬p ∨ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3142_314207


namespace NUMINAMATH_CALUDE_michaels_coins_value_l3142_314268

theorem michaels_coins_value (p n : ℕ) : 
  p + n = 15 ∧ 
  n + 2 = 2 * (p - 2) →
  p * 1 + n * 5 = 47 :=
by sorry

end NUMINAMATH_CALUDE_michaels_coins_value_l3142_314268


namespace NUMINAMATH_CALUDE_square_sum_equals_product_implies_zero_l3142_314249

theorem square_sum_equals_product_implies_zero (x y z : ℤ) :
  x^2 + y^2 + z^2 = 2*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_product_implies_zero_l3142_314249


namespace NUMINAMATH_CALUDE_max_discount_percentage_l3142_314231

theorem max_discount_percentage (cost : ℝ) (price : ℝ) (min_margin : ℝ) :
  cost = 400 →
  price = 500 →
  min_margin = 0.0625 →
  ∃ x : ℝ, x = 15 ∧
    ∀ y : ℝ, 0 ≤ y → y ≤ x →
      price * (1 - y / 100) - cost ≥ cost * min_margin ∧
      ∀ z : ℝ, z > x →
        price * (1 - z / 100) - cost < cost * min_margin :=
by sorry

end NUMINAMATH_CALUDE_max_discount_percentage_l3142_314231


namespace NUMINAMATH_CALUDE_original_equals_scientific_l3142_314282

/-- The number to be expressed in scientific notation -/
def original_number : ℝ := 1650000

/-- The scientific notation representation -/
def scientific_notation : ℝ := 1.65 * (10 ^ 6)

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific : original_number = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l3142_314282


namespace NUMINAMATH_CALUDE_simplify_fraction_l3142_314292

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  3 / (x - 1) + (x - 3) / (1 - x^2) = (2*x + 6) / (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3142_314292


namespace NUMINAMATH_CALUDE_triangle_side_length_l3142_314228

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = √3, b = 3, and B = 2A, then c = 2√3 -/
theorem triangle_side_length (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧          -- Sum of angles in a triangle
  a = Real.sqrt 3 ∧        -- Given: a = √3
  b = 3 ∧                  -- Given: b = 3
  B = 2 * A →              -- Given: B = 2A
  c = 2 * Real.sqrt 3 :=   -- Conclusion: c = 2√3
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3142_314228


namespace NUMINAMATH_CALUDE_boat_round_trip_average_speed_l3142_314238

/-- The average speed of a boat on a round trip, given its upstream and downstream speeds -/
theorem boat_round_trip_average_speed (distance : ℝ) (upstream_speed downstream_speed : ℝ) 
  (h1 : upstream_speed = 6)
  (h2 : downstream_speed = 3)
  (h3 : distance > 0) :
  (2 * distance) / ((distance / upstream_speed) + (distance / downstream_speed)) = 4 := by
  sorry

#check boat_round_trip_average_speed

end NUMINAMATH_CALUDE_boat_round_trip_average_speed_l3142_314238


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l3142_314299

/-- Two triangles are similar -/
structure SimilarTriangles (T1 T2 : Type) :=
  (ratio : ℝ)
  (similar : T1 → T2 → Prop)

/-- Triangle XYZ -/
structure TriangleXYZ :=
  (X Y Z : ℝ × ℝ)
  (YZ : ℝ)
  (XZ : ℝ)

/-- Triangle MNP -/
structure TriangleMNP :=
  (M N P : ℝ × ℝ)
  (MN : ℝ)
  (NP : ℝ)

theorem similar_triangles_side_length 
  (XYZ : TriangleXYZ) 
  (MNP : TriangleMNP) 
  (sim : SimilarTriangles TriangleXYZ TriangleMNP) 
  (h_sim : sim.similar XYZ MNP) 
  (h_YZ : XYZ.YZ = 10) 
  (h_XZ : XYZ.XZ = 7) 
  (h_MN : MNP.MN = 4.2) : 
  MNP.NP = 6 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l3142_314299


namespace NUMINAMATH_CALUDE_equation_proof_l3142_314237

theorem equation_proof : (5568 / 87 : ℝ)^(1/3) + (72 * 2 : ℝ)^(1/2) = (256 : ℝ)^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l3142_314237


namespace NUMINAMATH_CALUDE_unique_solution_system_l3142_314281

theorem unique_solution_system (x y z : ℝ) : 
  (x^2 - 23*y - 25*z = -681) ∧
  (y^2 - 21*x - 21*z = -419) ∧
  (z^2 - 19*x - 21*y = -313) ↔
  (x = 20 ∧ y = 22 ∧ z = 23) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3142_314281


namespace NUMINAMATH_CALUDE_opposite_of_half_l3142_314286

-- Define the concept of opposite
def opposite (x : ℝ) : ℝ := -x

-- Theorem statement
theorem opposite_of_half : opposite 0.5 = -0.5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_half_l3142_314286


namespace NUMINAMATH_CALUDE_complex_exponential_thirteen_pi_over_two_equals_i_l3142_314266

theorem complex_exponential_thirteen_pi_over_two_equals_i :
  Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_thirteen_pi_over_two_equals_i_l3142_314266


namespace NUMINAMATH_CALUDE_exists_a_C_is_line_C_passes_through_origin_L_intersects_C_l3142_314271

-- Define the curve C
def C (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1^2 + a * p.2^2 - 2 * p.1 - 2 * p.2 = 0}

-- Define a straight line
def isLine (S : Set (ℝ × ℝ)) : Prop :=
  ∃ A B C : ℝ, ∀ p : ℝ × ℝ, p ∈ S ↔ A * p.1 + B * p.2 + C = 0

-- Statement 1: C is a straight line for some a
theorem exists_a_C_is_line : ∃ a : ℝ, isLine (C a) := by sorry

-- Statement 2: C passes through (0, 0) for all a
theorem C_passes_through_origin : ∀ a : ℝ, (0, 0) ∈ C a := by sorry

-- Define the line x + 2y = 0
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + 2 * p.2 = 0}

-- Statement 3: When a = 1, L intersects C
theorem L_intersects_C : (C 1) ∩ L ≠ ∅ := by sorry

end NUMINAMATH_CALUDE_exists_a_C_is_line_C_passes_through_origin_L_intersects_C_l3142_314271


namespace NUMINAMATH_CALUDE_simultaneous_inequalities_condition_l3142_314202

theorem simultaneous_inequalities_condition (a b : ℝ) :
  (a > b ∧ 1 / a > 1 / b) ↔ (a > 0 ∧ 0 > b) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_inequalities_condition_l3142_314202


namespace NUMINAMATH_CALUDE_parabola_vertex_l3142_314270

/-- The equation of a parabola is x^2 - 4x + 3y + 8 = 0. -/
def parabola_equation (x y : ℝ) : Prop := x^2 - 4*x + 3*y + 8 = 0

/-- The vertex of a parabola is the point where it reaches its maximum or minimum y-value. -/
def is_vertex (x y : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x' y', eq x' y' → y ≤ y' ∨ y ≥ y'

/-- The vertex of the parabola defined by x^2 - 4x + 3y + 8 = 0 is (2, -4/3). -/
theorem parabola_vertex : is_vertex 2 (-4/3) parabola_equation := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3142_314270


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l3142_314264

theorem rectangle_area_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  let new_length := 1.3 * l
  let new_width := 1.2 * w
  let original_area := l * w
  let new_area := new_length * new_width
  (new_area - original_area) / original_area = 0.56 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l3142_314264


namespace NUMINAMATH_CALUDE_inequality_implies_a_range_l3142_314273

open Real

theorem inequality_implies_a_range (a : ℝ) : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ < π / 2 → 
    sqrt 2 * (2 * a + 3) * cos (θ - π / 4) + 6 / (sin θ + cos θ) - 2 * sin (2 * θ) < 3 * a + 6) →
  a > 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_a_range_l3142_314273


namespace NUMINAMATH_CALUDE_problem_solution_l3142_314254

theorem problem_solution (a b x y : ℝ) 
  (eq1 : 2*a*x + 2*b*y = 6)
  (eq2 : 3*a*x^2 + 3*b*y^2 = 21)
  (eq3 : 4*a*x^3 + 4*b*y^3 = 64)
  (eq4 : 5*a*x^4 + 5*b*y^4 = 210) :
  6*a*x^5 + 6*b*y^5 = 5372 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3142_314254


namespace NUMINAMATH_CALUDE_mikey_has_125_jelly_beans_l3142_314297

/-- The number of jelly beans each person has -/
structure JellyBeans where
  napoleon : ℕ
  sedrich : ℕ
  daphne : ℕ
  alondra : ℕ
  mikey : ℕ

/-- The conditions of the jelly bean problem -/
def jelly_bean_conditions (jb : JellyBeans) : Prop :=
  jb.napoleon = 56 ∧
  jb.sedrich = 3 * jb.napoleon + 9 ∧
  jb.daphne = 2 * (jb.sedrich - jb.napoleon) ∧
  jb.alondra = (jb.napoleon + jb.sedrich + jb.daphne) / 3 - 8 ∧
  jb.napoleon + jb.sedrich + jb.daphne + jb.alondra = 5 * jb.mikey

/-- The theorem stating that under the given conditions, Mikey has 125 jelly beans -/
theorem mikey_has_125_jelly_beans (jb : JellyBeans) 
  (h : jelly_bean_conditions jb) : jb.mikey = 125 := by
  sorry


end NUMINAMATH_CALUDE_mikey_has_125_jelly_beans_l3142_314297


namespace NUMINAMATH_CALUDE_prob_at_least_one_diamond_l3142_314215

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of diamond cards in a standard deck -/
def diamondCardCount : ℕ := 13

/-- Probability of drawing at least one diamond when drawing two cards without replacement -/
def probAtLeastOneDiamond : ℚ :=
  1 - (standardDeckSize - diamondCardCount) * (standardDeckSize - diamondCardCount - 1) /
      (standardDeckSize * (standardDeckSize - 1))

theorem prob_at_least_one_diamond :
  probAtLeastOneDiamond = 15 / 34 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_diamond_l3142_314215


namespace NUMINAMATH_CALUDE_function_value_at_two_l3142_314235

/-- Given a function f : ℝ → ℝ satisfying f(x) + 2f(1/x) = 3x for all x ≠ 0,
    prove that f(2) = -1 -/
theorem function_value_at_two (f : ℝ → ℝ) 
    (h : ∀ (x : ℝ), x ≠ 0 → f x + 2 * f (1 / x) = 3 * x) : 
    f 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l3142_314235


namespace NUMINAMATH_CALUDE_base_conversion_156_to_234_l3142_314279

-- Define a function to convert a base-8 number to base-10
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

-- Theorem statement
theorem base_conversion_156_to_234 :
  156 = base8ToBase10 [4, 3, 2] :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_156_to_234_l3142_314279


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l3142_314295

-- Define the hyperbola C
def hyperbola (m : ℝ) (x y : ℝ) : Prop := x^2 / m - y^2 = 1

-- Define the asymptote of C
def asymptote (m : ℝ) (x y : ℝ) : Prop := Real.sqrt 3 * x + m * y = 0

-- Theorem statement
theorem hyperbola_focal_length (m : ℝ) (h1 : m > 0) :
  (∀ x y, hyperbola m x y ↔ asymptote m x y) →
  ∃ a b c : ℝ, a^2 = m ∧ b^2 = 1 ∧ c^2 = a^2 + b^2 ∧ 2*c = 4 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l3142_314295


namespace NUMINAMATH_CALUDE_intersection_slope_l3142_314285

/-- Given two lines that intersect at a specific point, prove the slope of one line. -/
theorem intersection_slope (m : ℝ) : 
  (∀ x y, y = 5 * x + 3 → (x = 2 ∧ y = 13)) →  -- Line p passes through (2, 13)
  (∀ x y, y = m * x + 1 → (x = 2 ∧ y = 13)) →  -- Line q passes through (2, 13)
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_intersection_slope_l3142_314285


namespace NUMINAMATH_CALUDE_parabola_directrix_l3142_314289

/-- The equation of a parabola -/
def parabola (x y : ℝ) : Prop := y = 3 * x^2 - 6 * x + 1

/-- The equation of the directrix -/
def directrix (y : ℝ) : Prop := y = -25/12

/-- Theorem: The directrix of the given parabola is y = -25/12 -/
theorem parabola_directrix : 
  ∀ x y : ℝ, parabola x y → ∃ d : ℝ, directrix d ∧ 
  (∀ p q : ℝ, parabola p q → (p - x)^2 + (q - y)^2 = (q - d)^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3142_314289


namespace NUMINAMATH_CALUDE_stating_min_drops_required_stating_drops_18_sufficient_min_drops_is_18_l3142_314234

/-- Represents the number of floors in the building -/
def num_floors : ℕ := 163

/-- Represents the number of test phones available -/
def num_phones : ℕ := 2

/-- Represents a strategy for dropping phones -/
structure DropStrategy where
  num_drops : ℕ
  can_determine_all_cases : Bool

/-- 
Theorem stating that 18 drops is the minimum number required 
to determine the breaking floor or conclude the phone is unbreakable
-/
theorem min_drops_required : 
  ∀ (s : DropStrategy), s.can_determine_all_cases → s.num_drops ≥ 18 := by
  sorry

/-- 
Theorem stating that 18 drops is sufficient to determine 
the breaking floor or conclude the phone is unbreakable
-/
theorem drops_18_sufficient : 
  ∃ (s : DropStrategy), s.can_determine_all_cases ∧ s.num_drops = 18 := by
  sorry

/-- 
Main theorem combining the above results to prove that 18 
is the minimum number of drops required
-/
theorem min_drops_is_18 : 
  (∃ (s : DropStrategy), s.can_determine_all_cases ∧ 
    (∀ (t : DropStrategy), t.can_determine_all_cases → s.num_drops ≤ t.num_drops)) ∧
  (∃ (s : DropStrategy), s.can_determine_all_cases ∧ s.num_drops = 18) := by
  sorry

end NUMINAMATH_CALUDE_stating_min_drops_required_stating_drops_18_sufficient_min_drops_is_18_l3142_314234


namespace NUMINAMATH_CALUDE_P_bounds_l3142_314232

/-- A convex n-gon divided into triangles by non-intersecting diagonals -/
structure ConvexNGon (n : ℕ) where
  (n_ge_3 : n ≥ 3)

/-- Transformation that replaces triangles ABC and ACD with ABD and BCD -/
def transformation (n : ℕ) (g : ConvexNGon n) : ConvexNGon n := sorry

/-- P(n) is the minimum number of transformations required to convert any partition into any other partition -/
def P (n : ℕ) : ℕ := sorry

/-- Main theorem about bounds on P(n) -/
theorem P_bounds (n : ℕ) (g : ConvexNGon n) :
  P n ≥ n - 3 ∧
  P n ≤ 2*n - 7 ∧
  (n ≥ 13 → P n ≤ 2*n - 10) :=
sorry

end NUMINAMATH_CALUDE_P_bounds_l3142_314232


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l3142_314217

/-- Represents an isosceles triangle DEF -/
structure IsoscelesTriangle where
  /-- The length of the base of the triangle -/
  base : ℝ
  /-- The area of the triangle -/
  area : ℝ
  /-- The length of one of the congruent sides -/
  side : ℝ
  /-- Assertion that the base is positive -/
  base_pos : base > 0
  /-- Assertion that the area is positive -/
  area_pos : area > 0
  /-- Assertion that the side is positive -/
  side_pos : side > 0

/-- Theorem stating the relationship between the base, area, and side length of an isosceles triangle -/
theorem isosceles_triangle_side_length (t : IsoscelesTriangle) 
  (h1 : t.base = 30) 
  (h2 : t.area = 75) : 
  t.side = 5 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l3142_314217


namespace NUMINAMATH_CALUDE_figure_to_square_possible_l3142_314210

/-- A figure on a grid paper -/
structure GridFigure where
  area : ℕ

/-- Represents a dissection of a figure into parts -/
structure Dissection where
  parts : ℕ

/-- Represents a square shape -/
structure Square where
  side_length : ℕ

/-- A function that checks if a figure can be dissected into parts and formed into a square -/
def can_form_square (figure : GridFigure) (d : Dissection) (s : Square) : Prop :=
  figure.area = s.side_length ^ 2 ∧ d.parts = 3

theorem figure_to_square_possible (figure : GridFigure) (d : Dissection) (s : Square) 
  (h_area : figure.area = 16) (h_parts : d.parts = 3) (h_side : s.side_length = 4) : 
  can_form_square figure d s := by
  sorry

#check figure_to_square_possible

end NUMINAMATH_CALUDE_figure_to_square_possible_l3142_314210


namespace NUMINAMATH_CALUDE_ninth_ninety_ninth_digit_sum_l3142_314218

def decimal_expansion (n : ℕ) (d : ℕ) : ℚ := n / d

def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

theorem ninth_ninety_ninth_digit_sum (n : ℕ) : 
  nth_digit_after_decimal (decimal_expansion 2 9 + decimal_expansion 3 11 + decimal_expansion 5 13) 999 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ninth_ninety_ninth_digit_sum_l3142_314218


namespace NUMINAMATH_CALUDE_algae_coverage_day21_l3142_314272

-- Define the algae coverage function
def algaeCoverage (day : ℕ) : ℚ :=
  1 / 2^(24 - day)

-- State the theorem
theorem algae_coverage_day21 :
  algaeCoverage 24 = 1 ∧ (∀ d : ℕ, algaeCoverage (d + 1) = 2 * algaeCoverage d) →
  algaeCoverage 21 = 1/8 :=
by
  sorry

end NUMINAMATH_CALUDE_algae_coverage_day21_l3142_314272


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3142_314246

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 10*x - 24) → (∃ y : ℝ, y^2 = 10*y - 24 ∧ x + y = 10) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3142_314246


namespace NUMINAMATH_CALUDE_perfect_squares_condition_l3142_314298

theorem perfect_squares_condition (n : ℕ) : 
  (∃ k m : ℕ, 2 * n + 1 = k^2 ∧ 3 * n + 1 = m^2) ↔ 
  (∃ a b : ℕ, n + 1 = a^2 + (a + 1)^2 ∧ ∃ c : ℕ, n + 1 = c^2 + 2 * (c + 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_condition_l3142_314298


namespace NUMINAMATH_CALUDE_arrangements_count_l3142_314284

/-- The number of volunteers --/
def num_volunteers : ℕ := 4

/-- The number of elderly persons --/
def num_elderly : ℕ := 1

/-- The total number of people --/
def total_people : ℕ := num_volunteers + num_elderly

/-- The position of the elderly person --/
def elderly_position : ℕ := (total_people + 1) / 2

theorem arrangements_count :
  (num_volunteers.factorial * (num_volunteers + 1 - elderly_position).factorial) = 24 :=
sorry

end NUMINAMATH_CALUDE_arrangements_count_l3142_314284


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_equation_l3142_314263

/-- Given a parabola and a hyperbola in the Cartesian coordinate plane, 
    with a point of intersection and a condition on the focus of the parabola, 
    prove that the asymptotes of the hyperbola have a specific equation. -/
theorem hyperbola_asymptotes_equation (b : ℝ) (A : ℝ × ℝ) (F : ℝ × ℝ) :
  b > 0 →
  A.2^2 = 4 * A.1 →
  A.1^2 / 4 - A.2^2 / b^2 = 1 →
  F = (1, 0) →
  (A.1 - F.1)^2 + (A.2 - F.2)^2 = 25 →
  ∃ (k : ℝ), k = 2 * Real.sqrt 3 / 3 ∧
    (∀ (x y : ℝ), (x^2 / 4 - y^2 / b^2 = 1) → (y = k * x ∨ y = -k * x)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_equation_l3142_314263


namespace NUMINAMATH_CALUDE_gcd_8_factorial_6_factorial_squared_l3142_314293

theorem gcd_8_factorial_6_factorial_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8_factorial_6_factorial_squared_l3142_314293


namespace NUMINAMATH_CALUDE_calculation_proof_l3142_314262

theorem calculation_proof : 
  |(-1/2 : ℝ)| + (2023 - Real.pi)^0 - (27 : ℝ)^(1/3) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3142_314262


namespace NUMINAMATH_CALUDE_vowel_probability_is_three_thirteenths_l3142_314244

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The set of vowels including W -/
def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U', 'W'}

/-- The probability of selecting a vowel from the alphabet -/
def vowel_probability : ℚ := (Finset.card vowels : ℚ) / alphabet_size

theorem vowel_probability_is_three_thirteenths : 
  vowel_probability = 3 / 13 := by sorry

end NUMINAMATH_CALUDE_vowel_probability_is_three_thirteenths_l3142_314244


namespace NUMINAMATH_CALUDE_roberts_chocolates_l3142_314288

theorem roberts_chocolates (nickel_chocolates : ℕ) (difference : ℕ) : nickel_chocolates = 2 → difference = 7 → nickel_chocolates + difference = 9 := by
  sorry

end NUMINAMATH_CALUDE_roberts_chocolates_l3142_314288


namespace NUMINAMATH_CALUDE_parallelogram_area_error_l3142_314250

/-- Calculates the percentage error in the area of a parallelogram given measurement errors -/
theorem parallelogram_area_error (x y : ℝ) (z : Real) (hx : x > 0) (hy : y > 0) (hz : 0 < z ∧ z < pi) :
  let actual_area := x * y * Real.sin z
  let measured_area := (1.05 * x) * (1.07 * y) * Real.sin z
  (measured_area - actual_area) / actual_area * 100 = 12.35 := by
sorry


end NUMINAMATH_CALUDE_parallelogram_area_error_l3142_314250


namespace NUMINAMATH_CALUDE_probability_of_specific_colors_l3142_314265

def black_balls : ℕ := 5
def white_balls : ℕ := 7
def green_balls : ℕ := 2
def blue_balls : ℕ := 3
def red_balls : ℕ := 4

def total_balls : ℕ := black_balls + white_balls + green_balls + blue_balls + red_balls

def favorable_outcomes : ℕ := black_balls * green_balls * red_balls

def total_outcomes : ℕ := (total_balls.choose 3)

theorem probability_of_specific_colors : 
  (favorable_outcomes : ℚ) / total_outcomes = 4 / 133 := by sorry

end NUMINAMATH_CALUDE_probability_of_specific_colors_l3142_314265


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3142_314267

theorem arithmetic_calculation : (-1 + 2) * 3 + 2^2 / (-4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3142_314267


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_l3142_314201

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem largest_two_digit_prime_factor :
  ∃ (p : ℕ), is_prime p ∧ 
             p ≥ 10 ∧ p < 100 ∧
             p ∣ binomial_coefficient 300 150 ∧
             ∀ (q : ℕ), is_prime q → q ≥ 10 → q < 100 → q ∣ binomial_coefficient 300 150 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_l3142_314201


namespace NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l3142_314277

-- Equation 1
theorem solve_equation_one (x : ℝ) : 1 - 2 * (2 * x + 3) = -3 * (2 * x + 1) ↔ x = 1 := by sorry

-- Equation 2
theorem solve_equation_two (x : ℝ) : (x - 3) / 2 - (4 * x + 1) / 5 = 1 ↔ x = -9 := by sorry

end NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l3142_314277


namespace NUMINAMATH_CALUDE_common_factor_proof_l3142_314213

theorem common_factor_proof (x y : ℝ) : ∃ (k : ℝ), 5*x^2 - 25*x^2*y = 5*x^2 * k :=
sorry

end NUMINAMATH_CALUDE_common_factor_proof_l3142_314213


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l3142_314247

theorem inequality_solution_sets 
  (a b : ℝ) 
  (h1 : ∀ x, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) :
  ∀ x, b*x^2 - a*x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l3142_314247


namespace NUMINAMATH_CALUDE_correct_number_of_guesses_l3142_314233

/-- The number of valid guesses for three prizes with given digits -/
def number_of_valid_guesses : ℕ :=
  let digits : List ℕ := [2, 2, 2, 4, 4, 4, 4]
  let min_price : ℕ := 1
  let max_price : ℕ := 9999
  420

/-- Theorem stating that the number of valid guesses is 420 -/
theorem correct_number_of_guesses :
  number_of_valid_guesses = 420 := by sorry

end NUMINAMATH_CALUDE_correct_number_of_guesses_l3142_314233


namespace NUMINAMATH_CALUDE_log_x2y2_value_l3142_314225

theorem log_x2y2_value (x y : ℝ) (h1 : Real.log (x * y^2) = 2) (h2 : Real.log (x^3 * y) = 2) :
  Real.log (x^2 * y^2) = 12/5 := by sorry

end NUMINAMATH_CALUDE_log_x2y2_value_l3142_314225


namespace NUMINAMATH_CALUDE_widget_earnings_proof_l3142_314280

/-- Calculates the earnings per widget given the hourly rate, weekly hours, target earnings, and required widget production. -/
def earnings_per_widget (hourly_rate : ℚ) (weekly_hours : ℕ) (target_earnings : ℚ) (widget_production : ℕ) : ℚ :=
  (target_earnings - hourly_rate * weekly_hours) / widget_production

/-- Proves that the earnings per widget is $0.16 given the specified conditions. -/
theorem widget_earnings_proof :
  let hourly_rate : ℚ := 25 / 2
  let weekly_hours : ℕ := 40
  let target_earnings : ℚ := 620
  let widget_production : ℕ := 750
  earnings_per_widget hourly_rate weekly_hours target_earnings widget_production = 16 / 100 := by
  sorry

#eval earnings_per_widget (25/2) 40 620 750

end NUMINAMATH_CALUDE_widget_earnings_proof_l3142_314280


namespace NUMINAMATH_CALUDE_complex_expression_equals_negative_two_l3142_314275

theorem complex_expression_equals_negative_two :
  (2023 * Real.pi) ^ 0 + (-1/2)⁻¹ + |1 - Real.sqrt 3| - 2 * Real.sin (Real.pi / 3) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_negative_two_l3142_314275


namespace NUMINAMATH_CALUDE_second_term_value_l3142_314216

-- Define a sequence type
def Sequence := ℕ → ℝ

-- Define the Δ operator
def delta (A : Sequence) : Sequence :=
  λ n => A (n + 1) - A n

-- Main theorem
theorem second_term_value (A : Sequence) 
  (h1 : ∀ n, delta (delta A) n = 1)
  (h2 : A 12 = 0)
  (h3 : A 22 = 0) : 
  A 2 = 100 := by
sorry


end NUMINAMATH_CALUDE_second_term_value_l3142_314216


namespace NUMINAMATH_CALUDE_sum_of_roots_l3142_314222

theorem sum_of_roots (x : ℝ) : x + 16 / x = 12 → ∃ y : ℝ, y + 16 / y = 12 ∧ x + y = 12 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3142_314222


namespace NUMINAMATH_CALUDE_smallest_m_is_20_l3142_314203

theorem smallest_m_is_20 (m : ℕ) (y : ℕ → ℝ)
  (h1 : ∀ i, i ∈ Finset.range m → |y i| ≤ 1/2)
  (h2 : (Finset.range m).sum (fun i => |y i|) = 10 + |(Finset.range m).sum y|) :
  m ≥ 20 ∧ ∃ (y' : ℕ → ℝ), 
    (∀ i, i ∈ Finset.range 20 → |y' i| ≤ 1/2) ∧
    (Finset.range 20).sum (fun i => |y' i|) = 10 + |(Finset.range 20).sum y'| :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_is_20_l3142_314203


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_square_area_l3142_314252

theorem shaded_region_perimeter_square_area (PS PQ QR RS : ℝ) : 
  PS = 4 ∧ PQ + QR + RS = PS →
  let shaded_perimeter := (π/2) * (PS + PQ + QR + RS)
  let square_side := shaded_perimeter / 4
  square_side ^ 2 = π ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_square_area_l3142_314252


namespace NUMINAMATH_CALUDE_coin_count_l3142_314276

theorem coin_count (num_25_cent num_10_cent : ℕ) : 
  num_25_cent = 17 → num_10_cent = 17 → num_25_cent + num_10_cent = 34 := by
  sorry

end NUMINAMATH_CALUDE_coin_count_l3142_314276


namespace NUMINAMATH_CALUDE_rational_equation_result_l3142_314212

theorem rational_equation_result (x y : ℚ) 
  (h : |2*x - 3*y + 1| + (x + 3*y + 5)^2 = 0) : 
  (-2*x*y)^2 * (-y^2) * 6*x*y^2 = 192 := by
sorry

end NUMINAMATH_CALUDE_rational_equation_result_l3142_314212


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3142_314287

theorem quadratic_roots_property (m n : ℝ) : 
  m^2 + 1994*m + 7 = 0 → 
  n^2 + 1994*n + 7 = 0 → 
  (m^2 + 1993*m + 6) * (n^2 + 1995*n + 8) = 1986 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3142_314287


namespace NUMINAMATH_CALUDE_fourteen_sided_figure_area_l3142_314205

/-- A fourteen-sided figure constructed on a 1 cm × 1 cm grid -/
structure FourteenSidedFigure where
  /-- The number of full unit squares inside the figure -/
  full_squares : ℕ
  /-- The number of small right-angled triangles along the boundaries -/
  boundary_triangles : ℕ
  /-- The figure has 14 sides -/
  sides : ℕ
  sides_eq : sides = 14

/-- The area of the fourteen-sided figure is 16 cm² -/
theorem fourteen_sided_figure_area (f : FourteenSidedFigure) : 
  f.full_squares + f.boundary_triangles / 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fourteen_sided_figure_area_l3142_314205


namespace NUMINAMATH_CALUDE_f_properties_l3142_314283

def f (x : ℝ) := (x - 2)^2

theorem f_properties :
  (∀ x, f (x + 2) = f (-x - 2)) ∧
  (∀ x < 2, ∀ y < x, f y < f x) ∧
  (∀ x > 2, ∀ y > x, f y > f x) ∧
  (∀ x y, x < y → f (y + 2) - f y > f (x + 2) - f x) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3142_314283


namespace NUMINAMATH_CALUDE_correct_average_calculation_l3142_314278

theorem correct_average_calculation (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 ∧ initial_avg = 17 ∧ incorrect_num = 26 ∧ correct_num = 56 →
  (n : ℚ) * initial_avg - incorrect_num + correct_num = n * 20 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_calculation_l3142_314278


namespace NUMINAMATH_CALUDE_child_height_calculation_l3142_314239

/-- Given a child's previous height and growth, calculate the current height -/
def current_height (previous_height growth : ℝ) : ℝ :=
  previous_height + growth

/-- Theorem: The child's current height is 41.5 inches -/
theorem child_height_calculation : 
  current_height 38.5 3 = 41.5 := by
  sorry

end NUMINAMATH_CALUDE_child_height_calculation_l3142_314239


namespace NUMINAMATH_CALUDE_binary_divisible_by_seven_l3142_314229

def K (x y z : Fin 2) : ℕ :=
  524288 + 131072 + 65536 + 16384 + 4096 + 1024 + 256 + 64 * y.val + 32 * x.val + 16 * z.val + 8 + 2

theorem binary_divisible_by_seven (x y z : Fin 2) :
  K x y z % 7 = 0 → x = 0 ∧ y = 1 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_binary_divisible_by_seven_l3142_314229


namespace NUMINAMATH_CALUDE_gary_egg_collection_l3142_314204

/-- Calculates the number of eggs collected per week given the initial number of chickens,
    the multiplication factor after two years, eggs laid per chicken per day, and days in a week. -/
def eggs_per_week (initial_chickens : ℕ) (multiplication_factor : ℕ) (eggs_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  initial_chickens * multiplication_factor * eggs_per_day * days_in_week

/-- Proves that Gary collects 1344 eggs per week given the initial conditions. -/
theorem gary_egg_collection :
  eggs_per_week 4 8 6 7 = 1344 :=
by sorry

end NUMINAMATH_CALUDE_gary_egg_collection_l3142_314204


namespace NUMINAMATH_CALUDE_cube_through_cube_l3142_314290

theorem cube_through_cube (a : ℝ) (h : a > 0) : ∃ (s : ℝ), s > a ∧ s = (2 * a * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_through_cube_l3142_314290


namespace NUMINAMATH_CALUDE_divisibility_concatenation_l3142_314220

theorem divisibility_concatenation (a b : ℕ) : 
  100 ≤ a ∧ a < 1000 ∧ 100 ≤ b ∧ b < 1000 →  -- a and b are three-digit numbers
  ¬(37 ∣ a) →  -- a is not divisible by 37
  ¬(37 ∣ b) →  -- b is not divisible by 37
  (37 ∣ (a + b)) →  -- a + b is divisible by 37
  (37 ∣ (1000 * a + b))  -- 1000a + b is divisible by 37
  := by sorry

end NUMINAMATH_CALUDE_divisibility_concatenation_l3142_314220


namespace NUMINAMATH_CALUDE_quadratic_equation_problem_l3142_314236

theorem quadratic_equation_problem (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + 2*(a-1)*x + a^2 - 7*a - 4 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁*x₂ - 3*x₁ - 3*x₂ - 2 = 0 →
  (1 + 4/(a^2 - 4)) * (a + 2)/a = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_problem_l3142_314236


namespace NUMINAMATH_CALUDE_circle_m_range_l3142_314245

-- Define the equation of the circle
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + m*x + 2*m*y + 2*m^2 + m - 1 = 0

-- Theorem statement
theorem circle_m_range :
  (∃ x y : ℝ, circle_equation x y m) → -2 < m ∧ m < 2/3 :=
sorry

end NUMINAMATH_CALUDE_circle_m_range_l3142_314245


namespace NUMINAMATH_CALUDE_total_amount_proof_l3142_314291

/-- Proves that given the spending conditions, the total amount is $93,750 -/
theorem total_amount_proof (raw_materials : ℝ) (machinery : ℝ) (cash_percentage : ℝ) 
  (h1 : raw_materials = 35000)
  (h2 : machinery = 40000)
  (h3 : cash_percentage = 0.20)
  (total : ℝ)
  (h4 : total = raw_materials + machinery + cash_percentage * total) :
  total = 93750 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_proof_l3142_314291


namespace NUMINAMATH_CALUDE_sara_grew_four_onions_l3142_314251

/-- The number of onions grown by Sara, given the total number of onions and the numbers grown by Sally and Fred. -/
def saras_onions (total : ℕ) (sallys : ℕ) (freds : ℕ) : ℕ :=
  total - sallys - freds

/-- Theorem stating that Sara grew 4 onions given the conditions of the problem. -/
theorem sara_grew_four_onions :
  let total := 18
  let sallys := 5
  let freds := 9
  saras_onions total sallys freds = 4 := by
  sorry

end NUMINAMATH_CALUDE_sara_grew_four_onions_l3142_314251


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l3142_314223

theorem quadratic_perfect_square (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 200*x + c = (x + a)^2) → c = 10000 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l3142_314223


namespace NUMINAMATH_CALUDE_julie_lettuce_purchase_l3142_314269

/-- The total pounds of lettuce Julie bought -/
def total_lettuce (green_cost red_cost price_per_pound : ℚ) : ℚ :=
  green_cost / price_per_pound + red_cost / price_per_pound

/-- Proof that Julie bought 7 pounds of lettuce -/
theorem julie_lettuce_purchase : 
  total_lettuce 8 6 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_julie_lettuce_purchase_l3142_314269


namespace NUMINAMATH_CALUDE_det_max_value_l3142_314242

open Real

-- Define the determinant function
noncomputable def det (θ : ℝ) : ℝ :=
  let a := 1 + sin θ
  let b := 1 + cos θ
  a * (1 - a^2) - b * (1 - a * b) + (1 - b * a)

-- State the theorem
theorem det_max_value :
  ∀ θ : ℝ, det θ ≤ -1 ∧ ∃ θ₀ : ℝ, det θ₀ = -1 :=
by sorry

end NUMINAMATH_CALUDE_det_max_value_l3142_314242


namespace NUMINAMATH_CALUDE_car_journey_distance_l3142_314294

theorem car_journey_distance (S : ℝ) (D : ℝ) : 
  D = S * 7 ∧ D = (S + 12) * 5 → D = 210 := by sorry

end NUMINAMATH_CALUDE_car_journey_distance_l3142_314294


namespace NUMINAMATH_CALUDE_smallest_prime_sum_of_five_primes_l3142_314241

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

/-- A function that checks if a list of natural numbers contains distinct elements -/
def isDistinct (list : List ℕ) : Prop := list.Nodup

/-- The theorem stating that 43 is the smallest prime that is the sum of five distinct primes -/
theorem smallest_prime_sum_of_five_primes :
  ∃ (p₁ p₂ p₃ p₄ p₅ : ℕ),
    isPrime p₁ ∧ isPrime p₂ ∧ isPrime p₃ ∧ isPrime p₄ ∧ isPrime p₅ ∧
    isDistinct [p₁, p₂, p₃, p₄, p₅] ∧
    p₁ + p₂ + p₃ + p₄ + p₅ = 43 ∧
    isPrime 43 ∧
    (∀ (q : ℕ), q < 43 →
      ¬∃ (q₁ q₂ q₃ q₄ q₅ : ℕ),
        isPrime q₁ ∧ isPrime q₂ ∧ isPrime q₃ ∧ isPrime q₄ ∧ isPrime q₅ ∧
        isDistinct [q₁, q₂, q₃, q₄, q₅] ∧
        q₁ + q₂ + q₃ + q₄ + q₅ = q ∧
        isPrime q) :=
by sorry


end NUMINAMATH_CALUDE_smallest_prime_sum_of_five_primes_l3142_314241


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_32_l3142_314243

theorem smallest_positive_multiple_of_32 :
  ∀ n : ℕ, n > 0 → 32 * 1 ≤ 32 * n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_32_l3142_314243


namespace NUMINAMATH_CALUDE_carlos_earnings_l3142_314260

/-- Carlos's work hours and earnings problem -/
theorem carlos_earnings (hours_week1 hours_week2 : ℕ) (extra_earnings : ℚ) :
  hours_week1 = 12 →
  hours_week2 = 18 →
  extra_earnings = 36 →
  ∃ (hourly_wage : ℚ),
    hourly_wage * (hours_week2 - hours_week1) = extra_earnings ∧
    hourly_wage * (hours_week1 + hours_week2) = 180 :=
by sorry


end NUMINAMATH_CALUDE_carlos_earnings_l3142_314260


namespace NUMINAMATH_CALUDE_panthers_score_l3142_314255

theorem panthers_score (total_score cougars_margin : ℕ) 
  (h1 : total_score = 48)
  (h2 : cougars_margin = 20) :
  ∃ (panthers_score cougars_score : ℕ),
    panthers_score + cougars_score = total_score ∧
    cougars_score = panthers_score + cougars_margin ∧
    panthers_score = 14 :=
by sorry

end NUMINAMATH_CALUDE_panthers_score_l3142_314255


namespace NUMINAMATH_CALUDE_matrix_power_eight_l3142_314274

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, -1; 1, 1]

theorem matrix_power_eight :
  A^8 = !![16, 0; 0, 16] := by sorry

end NUMINAMATH_CALUDE_matrix_power_eight_l3142_314274
