import Mathlib

namespace possible_S_n_plus_1_l1433_143352

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Property: n ≡ S(n) (mod 9) for all natural numbers n -/
axiom S_mod_9 (n : ℕ) : n % 9 = S n % 9

theorem possible_S_n_plus_1 (n : ℕ) (h : S n = 3096) : 
  ∃ m : ℕ, m = n + 1 ∧ S m = 3097 := by sorry

end possible_S_n_plus_1_l1433_143352


namespace hcd_problem_l1433_143364

theorem hcd_problem : (Nat.gcd 2548 364 + 8) - 12 = 360 := by sorry

end hcd_problem_l1433_143364


namespace perimeter_ABCDE_l1433_143306

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
axiom AB_eq_BC : dist A B = dist B C
axiom AB_eq_4 : dist A B = 4
axiom AE_eq_8 : dist A E = 8
axiom ED_eq_9 : dist E D = 9
axiom right_angle_EAB : (A.1 - E.1) * (B.1 - A.1) + (A.2 - E.2) * (B.2 - A.2) = 0
axiom right_angle_ABC : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0
axiom right_angle_AED : (E.1 - A.1) * (D.1 - E.1) + (E.2 - A.2) * (D.2 - E.2) = 0

-- Define the perimeter function
def perimeter (A B C D E : ℝ × ℝ) : ℝ :=
  dist A B + dist B C + dist C D + dist D E + dist E A

-- State the theorem
theorem perimeter_ABCDE :
  perimeter A B C D E = 25 + Real.sqrt 41 := by
  sorry

end perimeter_ABCDE_l1433_143306


namespace percentage_chain_ten_percent_of_thirty_percent_of_fifty_percent_of_7000_l1433_143322

theorem percentage_chain (n : ℝ) : n * 0.5 * 0.3 * 0.1 = n * 0.015 := by sorry

theorem ten_percent_of_thirty_percent_of_fifty_percent_of_7000 :
  7000 * 0.5 * 0.3 * 0.1 = 105 := by sorry

end percentage_chain_ten_percent_of_thirty_percent_of_fifty_percent_of_7000_l1433_143322


namespace triangle_abc_area_l1433_143392

/-- Reflection of a point over the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflection of a point over the line y = -x -/
def reflect_neg_x (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, -p.1)

/-- Calculate the area of a triangle given three points -/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  0.5 * abs ((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))

theorem triangle_abc_area :
  let A : ℝ × ℝ := (2, 3)
  let B : ℝ × ℝ := reflect_y_axis A
  let C : ℝ × ℝ := reflect_neg_x B
  triangle_area A B C = 10 := by sorry

end triangle_abc_area_l1433_143392


namespace shape_relationships_l1433_143377

-- Define the basic geometric shapes
class Shape

-- Define specific shapes
class Rectangle extends Shape
class Rhombus extends Shape
class Triangle extends Shape
class Parallelogram extends Shape
class Square extends Shape
class Polygon extends Shape

-- Define specific types of triangles
class RightTriangle extends Triangle
class IsoscelesTriangle extends Triangle
class AcuteTriangle extends Triangle
class EquilateralTriangle extends IsoscelesTriangle
class ObtuseTriangle extends Triangle
class ScaleneTriangle extends Triangle

-- Define the relationships between shapes
theorem shape_relationships :
  -- Case 1
  (∃ x : Rectangle, ∃ y : Rhombus, True) ∧
  -- Case 2
  (∃ x : RightTriangle, ∃ y : IsoscelesTriangle, ∃ z : AcuteTriangle, True) ∧
  -- Case 3
  (∃ x : Parallelogram, ∃ y : Rectangle, ∃ z : Square, ∃ u : Rhombus, True) ∧
  -- Case 4
  (∃ x : Polygon, ∃ y : Triangle, ∃ z : IsoscelesTriangle, ∃ u : EquilateralTriangle, ∃ t : RightTriangle, True) ∧
  -- Case 5
  (∃ x : RightTriangle, ∃ y : IsoscelesTriangle, ∃ z : ObtuseTriangle, ∃ u : ScaleneTriangle, True) :=
by
  sorry


end shape_relationships_l1433_143377


namespace starting_elevation_l1433_143316

/-- Calculates the starting elevation of a person climbing a hill --/
theorem starting_elevation (final_elevation horizontal_distance : ℝ) : 
  final_elevation = 1450 ∧ 
  horizontal_distance = 2700 → 
  final_elevation - (horizontal_distance / 2) = 100 :=
by
  sorry

end starting_elevation_l1433_143316


namespace value_of_a_l1433_143331

-- Define the conversion rate between paise and rupees
def paise_per_rupee : ℚ := 100

-- Define the given percentage as a rational number
def given_percentage : ℚ := 1 / 200

-- Define the given amount in paise
def given_paise : ℚ := 95

-- Theorem statement
theorem value_of_a (a : ℚ) 
  (h : given_percentage * a = given_paise) : 
  a = 190 := by sorry

end value_of_a_l1433_143331


namespace visual_range_increase_l1433_143366

theorem visual_range_increase (original_range new_range : ℝ) (h1 : original_range = 60) (h2 : new_range = 150) :
  (new_range - original_range) / original_range * 100 = 150 := by
  sorry

end visual_range_increase_l1433_143366


namespace mass_is_not_vector_l1433_143315

-- Define the properties of a physical quantity
structure PhysicalQuantity where
  has_magnitude : Bool
  has_direction : Bool

-- Define what makes a quantity a vector
def is_vector (q : PhysicalQuantity) : Prop :=
  q.has_magnitude ∧ q.has_direction

-- Define the physical quantities
def mass : PhysicalQuantity :=
  { has_magnitude := true, has_direction := false }

def velocity : PhysicalQuantity :=
  { has_magnitude := true, has_direction := true }

def displacement : PhysicalQuantity :=
  { has_magnitude := true, has_direction := true }

def force : PhysicalQuantity :=
  { has_magnitude := true, has_direction := true }

-- Theorem to prove
theorem mass_is_not_vector : ¬(is_vector mass) := by
  sorry

end mass_is_not_vector_l1433_143315


namespace ceiling_floor_calculation_l1433_143333

theorem ceiling_floor_calculation : 
  ⌈(15 : ℝ) / 8 * (-45 : ℝ) / 4⌉ - ⌊(15 : ℝ) / 8 * ⌊(-45 : ℝ) / 4⌋⌋ = 2 := by sorry

end ceiling_floor_calculation_l1433_143333


namespace unique_solution_l1433_143385

/-- Two lines in a 2D plane -/
structure TwoLines where
  l₁ : ℝ → ℝ → ℝ  -- represents ax - by + 4 = 0
  l₂ : ℝ → ℝ → ℝ  -- represents (a - 1)x + y + b = 0
  a : ℝ
  b : ℝ

/-- Condition that l₁ is perpendicular to l₂ -/
def perpendicular (lines : TwoLines) : Prop :=
  lines.a * (lines.a - 1) - lines.b = 0

/-- Condition that l₁ passes through point (-3, -1) -/
def passes_through (lines : TwoLines) : Prop :=
  lines.l₁ (-3) (-1) = 0

/-- Condition that l₁ is parallel to l₂ -/
def parallel (lines : TwoLines) : Prop :=
  lines.a / lines.b = 1 - lines.a

/-- Condition that the distance from origin to both lines is equal -/
def equal_distance (lines : TwoLines) : Prop :=
  4 / lines.b = -lines.b

/-- The main theorem -/
theorem unique_solution (lines : TwoLines) :
  perpendicular lines ∧ passes_through lines →
  parallel lines ∧ equal_distance lines →
  lines.a = 2 ∧ lines.b = -2 :=
sorry

end unique_solution_l1433_143385


namespace water_filling_solution_l1433_143304

/-- Represents the water filling problem -/
def WaterFillingProblem (canCapacity : ℝ) (initialCans : ℕ) (initialFillRatio : ℝ) (initialTime : ℝ) (targetCans : ℕ) : Prop :=
  let initialWaterFilled := canCapacity * initialFillRatio * initialCans
  let fillRate := initialWaterFilled / initialTime
  let targetWaterToFill := canCapacity * targetCans
  targetWaterToFill / fillRate = 5

/-- Theorem stating the solution to the water filling problem -/
theorem water_filling_solution :
  WaterFillingProblem 8 20 (3/4) 3 25 := by
  sorry

end water_filling_solution_l1433_143304


namespace scrabble_champions_years_l1433_143373

theorem scrabble_champions_years (total_champions : ℕ) 
  (women_percent : ℚ) (men_with_beard_percent : ℚ) (men_with_beard : ℕ) : 
  women_percent = 3/5 →
  men_with_beard_percent = 2/5 →
  men_with_beard = 4 →
  total_champions = 25 := by
sorry

end scrabble_champions_years_l1433_143373


namespace road_graveling_cost_l1433_143345

theorem road_graveling_cost (lawn_length lawn_width road_width gravel_cost : ℝ) :
  lawn_length = 80 ∧
  lawn_width = 60 ∧
  road_width = 10 ∧
  gravel_cost = 5 →
  (lawn_length * road_width + (lawn_width - road_width) * road_width) * gravel_cost = 6500 :=
by sorry

end road_graveling_cost_l1433_143345


namespace student_count_l1433_143370

theorem student_count (right_rank left_rank : ℕ) 
  (h1 : right_rank = 13) 
  (h2 : left_rank = 8) : 
  right_rank + left_rank - 1 = 20 := by
  sorry

end student_count_l1433_143370


namespace problem_1_problem_2_l1433_143348

theorem problem_1 (α β : Real) 
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - π / 4) = 1 / 4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 / 22 := by
sorry

theorem problem_2 (α β : Real)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.cos (α + β) = Real.sqrt 5 / 5)
  (h4 : Real.sin (α - β) = Real.sqrt 10 / 10) :
  β = π / 8 := by
sorry

end problem_1_problem_2_l1433_143348


namespace sum_of_ab_l1433_143359

theorem sum_of_ab (a b : ℝ) (h1 : a * b = 5) (h2 : 1 / a^2 + 1 / b^2 = 0.6) : 
  a + b = 5 ∨ a + b = -5 := by
  sorry

end sum_of_ab_l1433_143359


namespace small_circle_radius_l1433_143342

/-- A design consisting of a small circle surrounded by four equal quarter-circle arcs -/
structure CircleDesign where
  /-- The radius of the large arcs -/
  R : ℝ
  /-- The radius of the small circle -/
  r : ℝ
  /-- The width of the design is 2 cm -/
  width_eq : R + r = 2

/-- The radius of the small circle in a CircleDesign with width 2 cm is 2 - √2 cm -/
theorem small_circle_radius (d : CircleDesign) : d.r = 2 - Real.sqrt 2 := by
  sorry

end small_circle_radius_l1433_143342


namespace simplify_and_rationalize_l1433_143301

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end simplify_and_rationalize_l1433_143301


namespace intersection_line_of_circles_l1433_143368

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 10
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 14 = 0

-- Define the line
def line (x y : ℝ) : Prop := x + y - 2 = 0

-- Theorem statement
theorem intersection_line_of_circles :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line x y :=
by sorry

end intersection_line_of_circles_l1433_143368


namespace problem_statement_l1433_143308

theorem problem_statement (a b : ℝ) 
  (ha : |a| = 3)
  (hb : |b| = 5)
  (hab_sum : a + b > 0)
  (hab_prod : a * b < 0) :
  a^3 + 2*b = -17 := by
  sorry

end problem_statement_l1433_143308


namespace points_opposite_sides_iff_a_in_range_l1433_143325

/-- The coordinates of point A satisfy the given equation. -/
def point_A_equation (a x y : ℝ) : Prop :=
  5 * a^2 - 6 * a * x - 4 * a * y + 2 * x^2 + 2 * x * y + y^2 = 0

/-- The equation of the circle centered at point B. -/
def circle_B_equation (a x y : ℝ) : Prop :=
  a^2 * x^2 + a^2 * y^2 - 6 * a^2 * x - 2 * a^3 * y + 4 * a * y + a^4 + 4 = 0

/-- Points A and B lie on opposite sides of the line y = 1. -/
def opposite_sides (ya yb : ℝ) : Prop :=
  (ya - 1) * (yb - 1) < 0

/-- The main theorem statement. -/
theorem points_opposite_sides_iff_a_in_range (a : ℝ) :
  (∃ (xa ya xb yb : ℝ),
    point_A_equation a xa ya ∧
    circle_B_equation a xb yb ∧
    opposite_sides ya yb) ↔
  (a > -1 ∧ a < 0) ∨ (a > 1 ∧ a < 2) :=
sorry

end points_opposite_sides_iff_a_in_range_l1433_143325


namespace tenth_term_is_144_l1433_143300

def fibonacci_like_sequence : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | n + 2 => fibonacci_like_sequence n + fibonacci_like_sequence (n + 1)

theorem tenth_term_is_144 : fibonacci_like_sequence 9 = 144 := by
  sorry

end tenth_term_is_144_l1433_143300


namespace y_derivative_y_derivative_at_zero_l1433_143302

-- Define y as a function of x
variable (y : ℝ → ℝ)

-- Define the condition e^y + xy = e
variable (h : ∀ x, Real.exp (y x) + x * (y x) = Real.exp 1)

-- Theorem for y'
theorem y_derivative (x : ℝ) : 
  deriv y x = -(y x) / (Real.exp (y x) + x) := by sorry

-- Theorem for y'(0)
theorem y_derivative_at_zero : 
  deriv y 0 = -(1 / Real.exp 1) := by sorry

end y_derivative_y_derivative_at_zero_l1433_143302


namespace alpha_beta_sum_l1433_143328

theorem alpha_beta_sum (α β : ℝ) 
  (hα : α^3 - 3*α^2 + 5*α = 1) 
  (hβ : β^3 - 3*β^2 + 5*β = 5) : 
  α + β = 2 := by sorry

end alpha_beta_sum_l1433_143328


namespace quadratic_equation_rational_solutions_product_of_c_values_l1433_143374

theorem quadratic_equation_rational_solutions (c : ℕ+) : 
  (∃ x : ℚ, 3 * x^2 + 17 * x + c.val = 0) ↔ (c.val = 14 ∨ c.val = 24) :=
sorry

theorem product_of_c_values : 
  (∃ c₁ c₂ : ℕ+, c₁ ≠ c₂ ∧ 
    (∃ x : ℚ, 3 * x^2 + 17 * x + c₁.val = 0) ∧ 
    (∃ x : ℚ, 3 * x^2 + 17 * x + c₂.val = 0) ∧
    c₁.val * c₂.val = 336) :=
sorry

end quadratic_equation_rational_solutions_product_of_c_values_l1433_143374


namespace trains_crossing_time_l1433_143351

/-- The time taken for two trains to cross each other -/
theorem trains_crossing_time (train_length : ℝ) (faster_speed : ℝ) : 
  train_length = 100 →
  faster_speed = 40 →
  (10 : ℝ) / 3 = (2 * train_length) / (faster_speed + faster_speed / 2) := by
  sorry

#check trains_crossing_time

end trains_crossing_time_l1433_143351


namespace solve_for_b_l1433_143307

theorem solve_for_b (a b : ℝ) 
  (eq1 : a * (a - 4) = 21)
  (eq2 : b * (b - 4) = 21)
  (neq : a ≠ b)
  (sum : a + b = 4) :
  b = -3 := by
sorry

end solve_for_b_l1433_143307


namespace square_ratio_side_length_sum_l1433_143346

theorem square_ratio_side_length_sum (area_ratio : ℚ) : 
  area_ratio = 50 / 98 →
  ∃ (a b c : ℕ), 
    (a * (b.sqrt : ℝ) / c : ℝ) ^ 2 = area_ratio ∧
    a = 5 ∧ b = 14 ∧ c = 49 ∧
    a + b + c = 68 :=
by sorry

end square_ratio_side_length_sum_l1433_143346


namespace parabola_tangent_secant_relation_l1433_143371

/-- A parabola with its axis parallel to the y-axis -/
structure Parabola where
  a : ℝ
  f : ℝ → ℝ
  f_eq : f = fun x ↦ a * x^2

/-- A point on the parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y = p.f x

/-- Tangent of the angle of inclination of the tangent at a point -/
def tangentSlope (p : Parabola) (point : PointOnParabola p) : ℝ :=
  2 * p.a * point.x

/-- Tangent of the angle of inclination of the secant line between two points -/
def secantSlope (p : Parabola) (p1 p2 : PointOnParabola p) : ℝ :=
  p.a * (p1.x + p2.x)

/-- The main theorem -/
theorem parabola_tangent_secant_relation (p : Parabola) 
    (A1 A2 A3 : PointOnParabola p) : 
    tangentSlope p A1 = secantSlope p A1 A2 + secantSlope p A1 A3 - secantSlope p A2 A3 := by
  sorry

end parabola_tangent_secant_relation_l1433_143371


namespace shaded_area_is_925_l1433_143311

-- Define the vertices of the square
def square_vertices : List (ℝ × ℝ) := [(0, 0), (40, 0), (40, 40), (0, 40)]

-- Define the vertices of the shaded polygon
def shaded_vertices : List (ℝ × ℝ) := [(0, 0), (15, 0), (40, 30), (30, 40), (0, 20)]

-- Function to calculate the area of a polygon given its vertices
def polygon_area (vertices : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem stating that the area of the shaded region is 925 square units
theorem shaded_area_is_925 :
  polygon_area shaded_vertices = 925 :=
sorry

end shaded_area_is_925_l1433_143311


namespace inequality_proof_l1433_143340

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z ≥ 1) :
  (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + z^2 + x^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 :=
by sorry

end inequality_proof_l1433_143340


namespace complex_modulus_10_minus_26i_l1433_143362

theorem complex_modulus_10_minus_26i :
  Complex.abs (10 - 26 * Complex.I) = 2 * Real.sqrt 194 := by
  sorry

end complex_modulus_10_minus_26i_l1433_143362


namespace joan_gained_two_balloons_l1433_143369

/-- The number of blue balloons Joan gained -/
def balloons_gained (initial final : ℕ) : ℕ := final - initial

/-- Proof that Joan gained 2 blue balloons -/
theorem joan_gained_two_balloons :
  let initial : ℕ := 9
  let final : ℕ := 11
  balloons_gained initial final = 2 := by sorry

end joan_gained_two_balloons_l1433_143369


namespace line_through_points_with_45_degree_slope_l1433_143350

/-- Given a line passing through points (3, m) and (2, 4) with a slope angle of 45°, prove that m = 5. -/
theorem line_through_points_with_45_degree_slope (m : ℝ) :
  (∃ (line : Set (ℝ × ℝ)), 
    (3, m) ∈ line ∧ 
    (2, 4) ∈ line ∧ 
    (∀ (x y : ℝ), (x, y) ∈ line → y - 4 = x - 2)) → 
  m = 5 := by
sorry

end line_through_points_with_45_degree_slope_l1433_143350


namespace smallest_m_is_24_l1433_143367

/-- The set of complex numbers with real part between 1/2 and 2/3 -/
def S : Set ℂ :=
  {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ 2/3}

/-- Definition of the property we want to prove for m -/
def has_nth_root_of_unity (m : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ m → ∃ z ∈ S, z^n = 1

/-- The theorem stating that 24 is the smallest positive integer satisfying the property -/
theorem smallest_m_is_24 :
  has_nth_root_of_unity 24 ∧ ∀ m : ℕ, 0 < m → m < 24 → ¬has_nth_root_of_unity m :=
sorry

end smallest_m_is_24_l1433_143367


namespace sufficient_not_necessary_l1433_143382

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perpToPlane : Line → Plane → Prop)
variable (para : Line → Plane → Prop)

-- Define the given objects
variable (l m : Line) (α : Plane)

-- Define the condition that l is perpendicular to α
variable (h : perpToPlane l α)

-- State the theorem
theorem sufficient_not_necessary :
  (∀ m, para m α → perp m l) ∧
  (∃ m, perp m l ∧ ¬para m α) :=
sorry

end sufficient_not_necessary_l1433_143382


namespace not_prime_n4_plus_n2_plus_1_l1433_143323

theorem not_prime_n4_plus_n2_plus_1 (n : ℤ) (h : n ≥ 2) :
  ¬(Nat.Prime (n^4 + n^2 + 1).natAbs) := by
  sorry

end not_prime_n4_plus_n2_plus_1_l1433_143323


namespace equality_identity_l1433_143383

theorem equality_identity (a : ℝ) : 
  (∃ a : ℝ, (a^4 - 1)^6 ≠ (a^6 - 1)^4) ∧ 
  (∀ a : ℝ, a = -1 ∨ a = 0 ∨ a = 1 → (a^4 - 1)^6 = (a^6 - 1)^4) := by
  sorry


end equality_identity_l1433_143383


namespace triangle_area_in_circle_l1433_143313

/-- Given a triangle with side lengths in the ratio 2:3:4 inscribed in a circle of radius 5,
    the area of the triangle is 18.75. -/
theorem triangle_area_in_circle (a b c : ℝ) (r : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Ensure positive side lengths
  r = 5 →  -- Circle radius is 5
  b = (3/2) * a →  -- Side length ratio 2:3
  c = 2 * a →  -- Side length ratio 2:4
  c = 2 * r →  -- Diameter of the circle
  (1/2) * a * b = 18.75 :=  -- Area of the triangle
by sorry


end triangle_area_in_circle_l1433_143313


namespace constant_a_value_l1433_143318

theorem constant_a_value (x y : ℝ) (a : ℝ) 
  (h1 : (a * x + 4 * y) / (x - 2 * y) = 13)
  (h2 : x / (2 * y) = 5 / 2) :
  a = 7 := by
  sorry

end constant_a_value_l1433_143318


namespace ab_value_l1433_143339

theorem ab_value (a b c : ℝ) 
  (h1 : a - b = 3)
  (h2 : a^2 + b^2 = 27)
  (h3 : a + b + c = 10)
  (h4 : a^3 - b^3 = 36) :
  a * b = -15 := by
  sorry

end ab_value_l1433_143339


namespace hybrid_rice_yield_and_conversion_l1433_143388

-- Define the yield per acre of ordinary rice
def ordinary_yield : ℝ := 600

-- Define the yield per acre of hybrid rice
def hybrid_yield : ℝ := 1200

-- Define the acreage difference between fields
def acreage_difference : ℝ := 4

-- Define the harvest of field A (hybrid rice)
def field_A_harvest : ℝ := 9600

-- Define the harvest of field B (ordinary rice)
def field_B_harvest : ℝ := 7200

-- Define the total yield goal
def total_yield_goal : ℝ := 17700

-- Define the minimum acres to be converted
def min_acres_converted : ℝ := 1.5

-- Theorem statement
theorem hybrid_rice_yield_and_conversion :
  (hybrid_yield = 2 * ordinary_yield) ∧
  (field_B_harvest / ordinary_yield - field_A_harvest / hybrid_yield = acreage_difference) ∧
  (field_A_harvest + ordinary_yield * (field_B_harvest / ordinary_yield - min_acres_converted) + hybrid_yield * min_acres_converted ≥ total_yield_goal) := by
  sorry

end hybrid_rice_yield_and_conversion_l1433_143388


namespace tangent_circle_radius_l1433_143332

/-- A 45°-45°-90° triangle inscribed in the first quadrant -/
structure RightIsoscelesTriangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  first_quadrant : X.1 ≥ 0 ∧ X.2 ≥ 0 ∧ Y.1 ≥ 0 ∧ Y.2 ≥ 0 ∧ Z.1 ≥ 0 ∧ Z.2 ≥ 0
  right_angle : (Z.1 - X.1) * (Y.1 - X.1) + (Z.2 - X.2) * (Y.2 - X.2) = 0
  isosceles : (Y.1 - X.1)^2 + (Y.2 - X.2)^2 = (Z.1 - Y.1)^2 + (Z.2 - Y.2)^2
  hypotenuse_length : (Z.1 - X.1)^2 + (Z.2 - X.2)^2 = 32  -- 4√2 squared

/-- A circle tangent to x-axis, y-axis, and hypotenuse of the triangle -/
structure TangentCircle (t : RightIsoscelesTriangle) where
  O : ℝ × ℝ
  r : ℝ
  tangent_x : O.2 = r
  tangent_y : O.1 = r
  tangent_hypotenuse : ((t.Z.1 - t.X.1) * (O.1 - t.X.1) + (t.Z.2 - t.X.2) * (O.2 - t.X.2))^2 = 
                       r^2 * ((t.Z.1 - t.X.1)^2 + (t.Z.2 - t.X.2)^2)

theorem tangent_circle_radius 
  (t : RightIsoscelesTriangle) 
  (c : TangentCircle t) 
  (h : (t.Y.1 - t.X.1)^2 + (t.Y.2 - t.X.2)^2 = 16) : 
  c.r = 2 := by
  sorry

end tangent_circle_radius_l1433_143332


namespace extra_crayons_l1433_143363

theorem extra_crayons (packs : ℕ) (crayons_per_pack : ℕ) (total_crayons : ℕ) : 
  packs = 4 → crayons_per_pack = 10 → total_crayons = 46 → 
  total_crayons - (packs * crayons_per_pack) = 6 := by
  sorry

end extra_crayons_l1433_143363


namespace combined_original_price_l1433_143357

/-- Given a pair of shoes with a 20% discount sold for $480 and a dress with a 30% discount sold for $350,
    prove that the combined original price of the shoes and dress is $1100. -/
theorem combined_original_price 
  (shoes_discount : Real) (dress_discount : Real)
  (shoes_discounted_price : Real) (dress_discounted_price : Real)
  (h1 : shoes_discount = 0.2)
  (h2 : dress_discount = 0.3)
  (h3 : shoes_discounted_price = 480)
  (h4 : dress_discounted_price = 350) :
  (shoes_discounted_price / (1 - shoes_discount)) + (dress_discounted_price / (1 - dress_discount)) = 1100 := by
  sorry

end combined_original_price_l1433_143357


namespace complex_equation_solution_l1433_143337

theorem complex_equation_solution :
  ∀ (a b : ℝ), (Complex.I : ℂ) * 2 + 1 * (a : ℂ) + (b : ℂ) = Complex.I * 2 →
  a = 1 ∧ b = -1 := by
sorry

end complex_equation_solution_l1433_143337


namespace congruent_implies_similar_similar_scale_one_implies_congruent_congruent_subset_similar_l1433_143396

-- Define geometric figures
structure GeometricFigure where
  -- Add necessary properties for a geometric figure
  -- This is a simplified representation
  shape : ℕ
  size : ℝ

-- Define congruence relation
def congruent (a b : GeometricFigure) : Prop :=
  a.shape = b.shape ∧ a.size = b.size

-- Define similarity relation with scale factor
def similar (a b : GeometricFigure) (scale : ℝ) : Prop :=
  a.shape = b.shape ∧ a.size = scale * b.size

-- Theorem: Congruent figures are similar with scale factor 1
theorem congruent_implies_similar (a b : GeometricFigure) :
  congruent a b → similar a b 1 := by
  sorry

-- Theorem: Similar figures with scale factor 1 are congruent
theorem similar_scale_one_implies_congruent (a b : GeometricFigure) :
  similar a b 1 → congruent a b := by
  sorry

-- Theorem: Congruent figures are a subset of similar figures
theorem congruent_subset_similar (a b : GeometricFigure) :
  congruent a b → ∃ scale, similar a b scale := by
  sorry

end congruent_implies_similar_similar_scale_one_implies_congruent_congruent_subset_similar_l1433_143396


namespace cosine_symmetry_axis_l1433_143397

/-- Given a function f(x) = cos(x - π/4), prove that its axis of symmetry is x = π/4 + kπ where k ∈ ℤ -/
theorem cosine_symmetry_axis (f : ℝ → ℝ) (k : ℤ) :
  (∀ x, f x = Real.cos (x - π/4)) →
  (∀ x, f (π/4 + k * π + x) = f (π/4 + k * π - x)) :=
by sorry

end cosine_symmetry_axis_l1433_143397


namespace sum_ages_in_three_years_l1433_143360

/-- The sum of Josiah and Hans' ages in three years -/
def sum_ages (hans_age : ℕ) (josiah_multiplier : ℕ) (years_later : ℕ) : ℕ :=
  (hans_age * josiah_multiplier + hans_age) + 2 * years_later

/-- Theorem stating the sum of Josiah and Hans' ages in three years -/
theorem sum_ages_in_three_years :
  sum_ages 15 3 3 = 66 := by
  sorry

#eval sum_ages 15 3 3

end sum_ages_in_three_years_l1433_143360


namespace binary_sum_theorem_l1433_143314

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binaryToDecimal (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Represents a binary number as a list of bits -/
def Binary := List Bool

/-- The binary number 1011₂ -/
def b1 : Binary := [true, false, true, true]

/-- The binary number 101₂ -/
def b2 : Binary := [true, false, true]

/-- The binary number 11001₂ -/
def b3 : Binary := [true, true, false, false, true]

/-- The binary number 1110₂ -/
def b4 : Binary := [true, true, true, false]

/-- The binary number 100101₂ -/
def b5 : Binary := [true, false, false, true, false, true]

/-- The expected sum 1111010₂ -/
def expectedSum : Binary := [true, true, true, true, false, true, false]

/-- Theorem stating that the sum of the given binary numbers equals the expected sum -/
theorem binary_sum_theorem :
  binaryToDecimal b1 + binaryToDecimal b2 + binaryToDecimal b3 + 
  binaryToDecimal b4 + binaryToDecimal b5 = binaryToDecimal expectedSum := by
  sorry

end binary_sum_theorem_l1433_143314


namespace fraction_to_percentage_decimal_seven_fifteenths_to_decimal_l1433_143344

theorem fraction_to_percentage_decimal (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = (n : ℚ) / (d : ℚ) := by sorry

theorem seven_fifteenths_to_decimal :
  (7 : ℚ) / 15 = 0.4666666666666667 := by sorry

end fraction_to_percentage_decimal_seven_fifteenths_to_decimal_l1433_143344


namespace siblings_comparison_l1433_143353

/-- Given information about siblings of Masud, Janet, Carlos, and Stella, prove that Janet has 16 fewer siblings than Carlos and Stella combined. -/
theorem siblings_comparison (masud janet carlos stella : ℕ) : 
  masud = 40 →
  janet = 4 * masud - 60 →
  carlos = 3 * masud / 4 + 12 →
  stella = 2 * (carlos - 12) - 8 →
  janet = 100 →
  carlos = 64 →
  stella = 52 →
  janet = carlos + stella - 16 := by
  sorry


end siblings_comparison_l1433_143353


namespace runners_speed_ratio_l1433_143356

/-- Two runners with different speeds start d miles apart. When running towards each other,
    they meet in s hours. When running in the same direction, the faster runner catches up
    to the slower one in u hours. This theorem proves that the ratio of their speeds is 2. -/
theorem runners_speed_ratio
  (d : ℝ) -- distance between starting points
  (s : ℝ) -- time to meet when running towards each other
  (u : ℝ) -- time for faster runner to catch up when running in same direction
  (h_d : d > 0)
  (h_s : s > 0)
  (h_u : u > 0) :
  ∃ (v_f v_s : ℝ), v_f > v_s ∧ v_f / v_s = 2 ∧
    v_f + v_s = d / s ∧
    (v_f - v_s) * u = v_s * u :=
by sorry

end runners_speed_ratio_l1433_143356


namespace wrapping_paper_area_l1433_143338

/-- A rectangular box with a square base -/
structure Box where
  base_side : ℝ
  height : ℝ
  height_eq_double_base : height = 2 * base_side

/-- A square sheet of wrapping paper -/
structure WrappingPaper where
  side_length : ℝ

/-- The configuration of the box on the wrapping paper -/
structure BoxWrappingConfiguration where
  box : Box
  paper : WrappingPaper
  box_centrally_placed : True
  vertices_on_midlines : True
  paper_folds_to_top_center : True

theorem wrapping_paper_area (config : BoxWrappingConfiguration) :
  config.paper.side_length ^ 2 = 16 * config.box.base_side ^ 2 := by
  sorry

end wrapping_paper_area_l1433_143338


namespace january_salary_l1433_143379

/-- Represents the salary structure for five months --/
structure SalaryStructure where
  jan : ℕ
  feb : ℕ
  mar : ℕ
  apr : ℕ
  may : ℕ

/-- Theorem stating the salary for January given the conditions --/
theorem january_salary (s : SalaryStructure) : s.jan = 4000 :=
  sorry

/-- The average salary for the first four months is 8000 --/
axiom avg_first_four (s : SalaryStructure) : 
  (s.jan + s.feb + s.mar + s.apr) / 4 = 8000

/-- The average salary for the last four months (including bonus) is 8800 --/
axiom avg_last_four (s : SalaryStructure) : 
  (s.feb + s.mar + s.apr + s.may + 1500) / 4 = 8800

/-- The salary for May (excluding bonus) is 6500 --/
axiom may_salary (s : SalaryStructure) : s.may = 6500

/-- February had a deduction of 700 --/
axiom feb_deduction (s : SalaryStructure) (feb_original : ℕ) : 
  s.feb = feb_original - 700

/-- No deductions in other months --/
axiom no_other_deductions (s : SalaryStructure) 
  (jan_original mar_original apr_original : ℕ) : 
  s.jan = jan_original ∧ s.mar = mar_original ∧ s.apr = apr_original

end january_salary_l1433_143379


namespace n_range_l1433_143320

/-- The function f(x) with parameters m and n -/
def f (m n x : ℝ) : ℝ := m * x^2 - (5 * m + n) * x + n

/-- Theorem stating the range of n given the conditions -/
theorem n_range :
  ∀ n : ℝ,
  (∃ m : ℝ, -2 < m ∧ m < -1 ∧
    ∃ x : ℝ, 3 < x ∧ x < 5 ∧ f m n x = 0) →
  0 < n ∧ n ≤ 3 :=
by sorry

end n_range_l1433_143320


namespace five_balls_four_boxes_l1433_143389

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 6 ways to distribute 5 indistinguishable balls into 4 indistinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 6 := by sorry

end five_balls_four_boxes_l1433_143389


namespace x_sixth_minus_six_x_when_three_l1433_143330

theorem x_sixth_minus_six_x_when_three :
  let x : ℝ := 3
  x^6 - 6*x = 711 := by
  sorry

end x_sixth_minus_six_x_when_three_l1433_143330


namespace ceiling_sqrt_225_l1433_143390

theorem ceiling_sqrt_225 : ⌈Real.sqrt 225⌉ = 15 := by sorry

end ceiling_sqrt_225_l1433_143390


namespace basketball_club_boys_l1433_143310

theorem basketball_club_boys (total : ℕ) (attendance : ℕ) (boys : ℕ) (girls : ℕ) :
  total = 30 →
  attendance = 18 →
  total = boys + girls →
  attendance = boys + (girls / 3) →
  boys = 12 := by
sorry

end basketball_club_boys_l1433_143310


namespace percentage_increase_l1433_143336

theorem percentage_increase (initial : ℝ) (final : ℝ) (percentage : ℝ) : 
  initial = 240 → final = 288 → percentage = 20 →
  (final - initial) / initial * 100 = percentage := by
  sorry

end percentage_increase_l1433_143336


namespace new_students_count_l1433_143324

theorem new_students_count (initial_students : Nat) (left_students : Nat) (final_students : Nat) :
  initial_students = 11 →
  left_students = 6 →
  final_students = 47 →
  final_students - (initial_students - left_students) = 42 :=
by sorry

end new_students_count_l1433_143324


namespace smallest_sum_of_reciprocals_l1433_143394

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 20 → 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 20 → (x : ℕ) + y ≤ (a : ℕ) + b) →
  (x : ℕ) + y = 81 :=
by sorry

end smallest_sum_of_reciprocals_l1433_143394


namespace two_digit_addition_equation_l1433_143334

theorem two_digit_addition_equation (A B : ℕ) : 
  A ≠ B →
  A < 10 →
  B < 10 →
  6 * A + 10 * B + 2 = 77 →
  B = 1 := by sorry

end two_digit_addition_equation_l1433_143334


namespace pizza_party_group_size_l1433_143341

/-- Given a group of people consisting of children and adults, where the number of children
    is twice the number of adults and there are 80 children, prove that the total number
    of people in the group is 120. -/
theorem pizza_party_group_size :
  ∀ (num_children num_adults : ℕ),
    num_children = 80 →
    num_children = 2 * num_adults →
    num_children + num_adults = 120 :=
by
  sorry

end pizza_party_group_size_l1433_143341


namespace area_of_right_triangle_PQR_l1433_143309

/-- A right triangle PQR in the xy-plane with specific properties -/
structure RightTrianglePQR where
  -- P, Q, R are points in ℝ²
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  -- PQR is a right triangle with right angle at R
  is_right_triangle : (P.1 - R.1) * (Q.1 - R.1) + (P.2 - R.2) * (Q.2 - R.2) = 0
  -- Length of hypotenuse PQ is 50
  hypotenuse_length : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 50^2
  -- Median through P lies along y = x + 2
  median_P : ∃ t : ℝ, (P.1 + R.1) / 2 = t ∧ (P.2 + R.2) / 2 = t + 2
  -- Median through Q lies along y = 2x + 3
  median_Q : ∃ t : ℝ, (Q.1 + R.1) / 2 = t ∧ (Q.2 + R.2) / 2 = 2*t + 3

/-- The area of the right triangle PQR is 500/3 -/
theorem area_of_right_triangle_PQR (t : RightTrianglePQR) : 
  abs ((t.P.1 - t.R.1) * (t.Q.2 - t.R.2) - (t.Q.1 - t.R.1) * (t.P.2 - t.R.2)) / 2 = 500 / 3 :=
sorry

end area_of_right_triangle_PQR_l1433_143309


namespace expand_product_l1433_143349

theorem expand_product (x : ℝ) : (x + 3) * (x^2 + 4*x + 6) = x^3 + 7*x^2 + 18*x + 18 := by
  sorry

end expand_product_l1433_143349


namespace team_size_is_eight_l1433_143375

/-- The number of players in the basketball team -/
def n : ℕ := sorry

/-- The initial average height of the team in centimeters -/
def initial_average : ℝ := 190

/-- The height of the player leaving the team in centimeters -/
def height_leaving : ℝ := 197

/-- The height of the player joining the team in centimeters -/
def height_joining : ℝ := 181

/-- The new average height of the team after the player change in centimeters -/
def new_average : ℝ := 188

/-- Theorem stating that the number of players in the team is 8 -/
theorem team_size_is_eight :
  (n : ℝ) * initial_average - (height_leaving - height_joining) = n * new_average ∧ n = 8 := by
  sorry

end team_size_is_eight_l1433_143375


namespace range_of_sine_function_l1433_143391

open Set
open Real

theorem range_of_sine_function (x : ℝ) (h : 0 < x ∧ x < 2*π/3) :
  ∃ y, y ∈ Ioo 0 1 ∧ y = 2 * sin (x + π/6) - 1 ∧
  ∀ z, z = 2 * sin (x + π/6) - 1 → z ∈ Ioc 0 1 :=
sorry

end range_of_sine_function_l1433_143391


namespace max_distance_between_circles_l1433_143335

/-- The maximum distance between the centers of two circles with 6-inch diameters
    placed within a 12-inch by 14-inch rectangle without extending beyond it. -/
def max_circle_centers_distance : ℝ := 10

/-- The width of the rectangle -/
def rectangle_width : ℝ := 12

/-- The height of the rectangle -/
def rectangle_height : ℝ := 14

/-- The diameter of each circle -/
def circle_diameter : ℝ := 6

/-- Theorem stating that the maximum distance between the centers of the circles is 10 inches -/
theorem max_distance_between_circles :
  ∀ (center1 center2 : ℝ × ℝ),
  (0 ≤ center1.1 ∧ center1.1 ≤ rectangle_width) →
  (0 ≤ center1.2 ∧ center1.2 ≤ rectangle_height) →
  (0 ≤ center2.1 ∧ center2.1 ≤ rectangle_width) →
  (0 ≤ center2.2 ∧ center2.2 ≤ rectangle_height) →
  (∀ (x y : ℝ), (x - center1.1)^2 + (y - center1.2)^2 ≤ (circle_diameter / 2)^2 →
    0 ≤ x ∧ x ≤ rectangle_width ∧ 0 ≤ y ∧ y ≤ rectangle_height) →
  (∀ (x y : ℝ), (x - center2.1)^2 + (y - center2.2)^2 ≤ (circle_diameter / 2)^2 →
    0 ≤ x ∧ x ≤ rectangle_width ∧ 0 ≤ y ∧ y ≤ rectangle_height) →
  (center1.1 - center2.1)^2 + (center1.2 - center2.2)^2 ≤ max_circle_centers_distance^2 :=
by sorry

end max_distance_between_circles_l1433_143335


namespace man_double_son_age_l1433_143305

/-- Calculates the number of years until a man's age is twice his son's age. -/
def yearsUntilDoubleAge (manAge sonAge : ℕ) : ℕ :=
  sorry

/-- Proves that the number of years until the man's age is twice his son's age is 2. -/
theorem man_double_son_age :
  let sonAge : ℕ := 14
  let manAge : ℕ := sonAge + 16
  yearsUntilDoubleAge manAge sonAge = 2 := by
  sorry

end man_double_son_age_l1433_143305


namespace enemy_plane_hit_probability_l1433_143361

theorem enemy_plane_hit_probability 
  (prob_A : ℝ) 
  (prob_B : ℝ) 
  (h_prob_A : prob_A = 0.7) 
  (h_prob_B : prob_B = 0.5) : 
  1 - (1 - prob_A) * (1 - prob_B) = 0.85 := by
sorry

end enemy_plane_hit_probability_l1433_143361


namespace crayon_selection_count_l1433_143384

-- Define the number of crayons of each color
def red_crayons : ℕ := 4
def blue_crayons : ℕ := 5
def green_crayons : ℕ := 3
def yellow_crayons : ℕ := 3

-- Define the total number of crayons
def total_crayons : ℕ := red_crayons + blue_crayons + green_crayons + yellow_crayons

-- Define the number of crayons to be selected
def select_count : ℕ := 5

-- Function to calculate combinations
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- Define the theorem
theorem crayon_selection_count :
  ∃ (x : ℕ),
    x = combination total_crayons select_count -
        (combination (total_crayons - red_crayons) select_count +
         combination (total_crayons - blue_crayons) select_count +
         combination (total_crayons - green_crayons) select_count +
         combination (total_crayons - yellow_crayons) select_count) +
        -- Placeholder for corrections due to over-subtraction
        0 :=
by
  sorry

end crayon_selection_count_l1433_143384


namespace parabola_vertex_l1433_143358

/-- The parabola is defined by the equation y = (x - 1)^2 - 2 -/
def parabola (x : ℝ) : ℝ := (x - 1)^2 - 2

/-- The vertex of a parabola is the point where it reaches its maximum or minimum -/
def is_vertex (x y : ℝ) : Prop :=
  ∀ t : ℝ, parabola t ≥ parabola x ∨ parabola t ≤ parabola x

theorem parabola_vertex :
  is_vertex 1 (-2) :=
sorry

end parabola_vertex_l1433_143358


namespace max_value_problem_1_l1433_143399

theorem max_value_problem_1 (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  ∃ (max_y : ℝ), ∀ y : ℝ, y = 1/2 * x * (1 - 2*x) → y ≤ max_y ∧ max_y = 1/16 := by
  sorry


end max_value_problem_1_l1433_143399


namespace min_max_sum_bound_l1433_143321

theorem min_max_sum_bound (a b c d e f g : ℝ) 
  (non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0 ∧ g ≥ 0) 
  (sum_one : a + b + c + d + e + f + g = 1) : 
  ∃ (x : ℝ), x ≥ 1/3 ∧ 
    (∀ y, y = max (a+b+c) (max (b+c+d) (max (c+d+e) (max (d+e+f) (e+f+g)))) → y ≤ x) ∧
    (∃ (a' b' c' d' e' f' g' : ℝ),
      a' ≥ 0 ∧ b' ≥ 0 ∧ c' ≥ 0 ∧ d' ≥ 0 ∧ e' ≥ 0 ∧ f' ≥ 0 ∧ g' ≥ 0 ∧
      a' + b' + c' + d' + e' + f' + g' = 1 ∧
      max (a'+b'+c') (max (b'+c'+d') (max (c'+d'+e') (max (d'+e'+f') (e'+f'+g')))) = 1/3) :=
by sorry

end min_max_sum_bound_l1433_143321


namespace second_group_has_ten_students_l1433_143380

/-- The number of students in the second kindergartner group -/
def second_group_size : ℕ := 10

/-- The number of students in the first kindergartner group -/
def first_group_size : ℕ := 9

/-- The number of students in the third kindergartner group -/
def third_group_size : ℕ := 11

/-- The number of tissues in each mini tissue box -/
def tissues_per_box : ℕ := 40

/-- The total number of tissues brought by all kindergartner groups -/
def total_tissues : ℕ := 1200

/-- Theorem stating that the second group has 10 students -/
theorem second_group_has_ten_students :
  second_group_size = 10 :=
by
  sorry

#check second_group_has_ten_students

end second_group_has_ten_students_l1433_143380


namespace least_addition_for_divisibility_l1433_143326

theorem least_addition_for_divisibility : 
  ∃! x : ℕ, x > 0 ∧ 
  (∀ y : ℕ, y > 0 → (1056 + y) % 29 = 0 ∧ (1056 + y) % 37 = 0 ∧ (1056 + y) % 43 = 0 → x ≤ y) ∧
  (1056 + x) % 29 = 0 ∧ (1056 + x) % 37 = 0 ∧ (1056 + x) % 43 = 0 ∧
  x = 44597 := by
sorry

end least_addition_for_divisibility_l1433_143326


namespace monotonic_cubic_function_l1433_143329

/-- A function f(x) = -x³ + ax² - x - 1 is monotonic on ℝ iff a ∈ [-√3, √3] -/
theorem monotonic_cubic_function (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => -x^3 + a*x^2 - x - 1)) ↔ 
  a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end monotonic_cubic_function_l1433_143329


namespace power_of_product_l1433_143347

theorem power_of_product (a b : ℝ) (m : ℕ+) : (a * b) ^ (m : ℕ) = a ^ (m : ℕ) * b ^ (m : ℕ) := by
  sorry

end power_of_product_l1433_143347


namespace pencil_count_l1433_143317

theorem pencil_count :
  ∀ (pens pencils : ℕ),
  (pens : ℚ) / (pencils : ℚ) = 5 / 6 →
  pencils = pens + 5 →
  pencils = 30 :=
by
  sorry

end pencil_count_l1433_143317


namespace fifth_result_proof_l1433_143381

theorem fifth_result_proof (total_average : ℚ) (first_five_average : ℚ) (last_seven_average : ℚ) 
  (h1 : total_average = 42)
  (h2 : first_five_average = 49)
  (h3 : last_seven_average = 52) :
  ∃ (fifth_result : ℚ), fifth_result = 147 ∧ 
    (5 * first_five_average + 7 * last_seven_average - fifth_result) / 11 = total_average := by
  sorry

end fifth_result_proof_l1433_143381


namespace erased_number_l1433_143312

theorem erased_number (a : ℤ) (b : ℤ) (h1 : -4 ≤ b ∧ b ≤ 4) 
  (h2 : (a - 4) + (a - 3) + (a - 2) + (a - 1) + a + (a + 1) + (a + 2) + (a + 3) + (a + 4) - (a + b) = 1703) : 
  a + b = 214 := by
sorry

end erased_number_l1433_143312


namespace probability_sum_greater_than_five_l1433_143393

def roll_die : Finset ℕ := Finset.range 6

theorem probability_sum_greater_than_five :
  let outcomes := (roll_die.product roll_die).filter (λ p => p.1 + p.2 > 5)
  (outcomes.card : ℚ) / (roll_die.card * roll_die.card) = 13 / 18 := by
sorry

end probability_sum_greater_than_five_l1433_143393


namespace function_difference_l1433_143354

theorem function_difference (F : ℝ → ℤ) (h1 : F 3 = 3) (h2 : F 1 = 2) :
  F 3 - F 1 = 1 := by
  sorry

end function_difference_l1433_143354


namespace peters_age_one_third_of_jacobs_l1433_143386

/-- Proves the number of years ago when Peter's age was one-third of Jacob's age -/
theorem peters_age_one_third_of_jacobs (peter_current_age jacob_current_age years_ago : ℕ) :
  peter_current_age = 16 →
  jacob_current_age = peter_current_age + 12 →
  peter_current_age - years_ago = (jacob_current_age - years_ago) / 3 →
  years_ago = 10 := by sorry

end peters_age_one_third_of_jacobs_l1433_143386


namespace new_average_is_65_l1433_143378

/-- Calculates the new average marks per paper after additional marks are added. -/
def new_average_marks (num_papers : ℕ) (original_average : ℚ) (additional_marks_geo : ℕ) (additional_marks_hist : ℕ) : ℚ :=
  (num_papers * original_average + additional_marks_geo + additional_marks_hist) / num_papers

/-- Proves that the new average marks per paper is 65 given the specified conditions. -/
theorem new_average_is_65 :
  new_average_marks 11 63 20 2 = 65 := by
  sorry

end new_average_is_65_l1433_143378


namespace surface_sum_bounds_l1433_143343

/-- Represents a small cube with numbers on its faces -/
structure SmallCube :=
  (faces : Fin 6 → Nat)
  (opposite_sum_seven : ∀ i : Fin 3, faces i + faces (i + 3) = 7)
  (valid_numbers : ∀ i : Fin 6, 1 ≤ faces i ∧ faces i ≤ 6)

/-- Represents the larger cube assembled from 64 small cubes -/
structure LargeCube :=
  (small_cubes : Fin 64 → SmallCube)

/-- The sum of visible numbers on the surface of the larger cube -/
def surface_sum (lc : LargeCube) : Nat :=
  sorry

theorem surface_sum_bounds (lc : LargeCube) :
  144 ≤ surface_sum lc ∧ surface_sum lc ≤ 528 := by
  sorry

end surface_sum_bounds_l1433_143343


namespace absolute_value_inequality_l1433_143327

theorem absolute_value_inequality (a b : ℝ) (h : a ≠ 0) :
  ∃ (m : ℝ), m = 2 ∧ (∀ (x y : ℝ) (h' : x ≠ 0), |x + y| + |x - y| ≥ m * |x|) ∧
  (∀ (m' : ℝ), (∀ (x y : ℝ) (h' : x ≠ 0), |x + y| + |x - y| ≥ m' * |x|) → m' ≤ m) :=
sorry

end absolute_value_inequality_l1433_143327


namespace cone_base_radius_l1433_143372

/-- Given a cone formed from a sector of a circle with a central angle of 120° and a radius of 6,
    the radius of the base circle of the cone is 2. -/
theorem cone_base_radius (sector_angle : ℝ) (sector_radius : ℝ) (base_radius : ℝ) : 
  sector_angle = 120 * π / 180 ∧ 
  sector_radius = 6 ∧ 
  base_radius = sector_angle / (2 * π) * sector_radius → 
  base_radius = 2 := by
  sorry

end cone_base_radius_l1433_143372


namespace geometric_series_problem_l1433_143355

theorem geometric_series_problem (a r : ℝ) 
  (h1 : |r| < 1) 
  (h2 : a / (1 - r) = 7)
  (h3 : a * r / (1 - r^2) = 3) : 
  a + r = 5/2 := by sorry

end geometric_series_problem_l1433_143355


namespace quadratic_inequality_equivalence_l1433_143395

theorem quadratic_inequality_equivalence :
  ∀ x : ℝ, x * (2 * x + 3) < -15 ↔ x ∈ Set.Ioo (-5/2) 3 := by
  sorry

end quadratic_inequality_equivalence_l1433_143395


namespace inequality_system_solution_l1433_143319

theorem inequality_system_solution (x : ℝ) :
  (x - 3 * (x - 2) ≥ 4 ∧ (1 + 2 * x) / 3 > x - 1) ↔ x ≤ 1 := by
  sorry

end inequality_system_solution_l1433_143319


namespace bad_carrots_count_l1433_143365

theorem bad_carrots_count (carol_carrots mom_carrots brother_carrots good_carrots : ℕ) 
  (h1 : carol_carrots = 29)
  (h2 : mom_carrots = 16)
  (h3 : brother_carrots = 23)
  (h4 : good_carrots = 52) :
  carol_carrots + mom_carrots + brother_carrots - good_carrots = 16 := by
sorry

end bad_carrots_count_l1433_143365


namespace simplify_and_evaluate_l1433_143387

theorem simplify_and_evaluate (a b : ℤ) (h1 : a = -2) (h2 : b = 1) :
  (a^2 + 2*a*b) - 2*(a^2 + 4*a*b - b) = 10 := by
  sorry

end simplify_and_evaluate_l1433_143387


namespace path_length_along_squares_l1433_143303

theorem path_length_along_squares (PQ : ℝ) (h : PQ = 73) : 
  3 * PQ = 219 := by
  sorry

end path_length_along_squares_l1433_143303


namespace total_letters_received_l1433_143376

theorem total_letters_received (brother_letters : ℕ) (greta_extra : ℕ) (mother_multiplier : ℕ) : 
  brother_letters = 40 → 
  greta_extra = 10 → 
  mother_multiplier = 2 → 
  (brother_letters + (brother_letters + greta_extra) + 
   mother_multiplier * (brother_letters + (brother_letters + greta_extra))) = 270 :=
by
  sorry

end total_letters_received_l1433_143376


namespace boxes_sold_tuesday_l1433_143398

/-- The number of boxes Kim sold on different days of the week -/
structure BoxesSold where
  friday : ℕ
  thursday : ℕ
  wednesday : ℕ
  tuesday : ℕ

/-- The conditions of the problem -/
def problem_conditions (b : BoxesSold) : Prop :=
  b.friday = 600 ∧
  b.thursday = (3/2 : ℚ) * b.friday ∧
  b.wednesday = 2 * b.thursday ∧
  b.tuesday = 3 * b.wednesday

/-- The theorem stating that under the given conditions, Kim sold 5400 boxes on Tuesday -/
theorem boxes_sold_tuesday (b : BoxesSold) (h : problem_conditions b) : b.tuesday = 5400 := by
  sorry

end boxes_sold_tuesday_l1433_143398
