import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_alpha_min_beta_l1194_119425

/-- Definition of M as a function of x, y, and z -/
noncomputable def M (x y z : ℝ) : ℝ :=
  (Real.sqrt (x^2 + x*y + y^2) * Real.sqrt (y^2 + y*z + z^2)) +
  (Real.sqrt (y^2 + y*z + z^2) * Real.sqrt (z^2 + z*x + x^2)) +
  (Real.sqrt (z^2 + z*x + x^2) * Real.sqrt (x^2 + x*y + y^2))

/-- The theorem stating the maximum value of α and minimum value of β -/
theorem max_alpha_min_beta (α β : ℝ) :
  (∀ x y z : ℝ, α * (x*y + y*z + z*x) ≤ M x y z ∧ M x y z ≤ β * (x^2 + y^2 + z^2)) →
  α ≤ 3 ∧ β ≥ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_alpha_min_beta_l1194_119425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1194_119467

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def condition1 (t : Triangle) : Prop :=
  2 * Real.cos (Real.pi + t.A) + Real.sin (Real.pi / 2 + 2 * t.A) + 3 / 2 = 0

def condition2 (t : Triangle) : Prop :=
  t.c - t.b = (Real.sqrt 3) / 3 * t.a

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h1 : condition1 t) (h2 : condition2 t) : 
  t.A = Real.pi / 3 ∧ (t.B = Real.pi / 2 ∨ t.C = Real.pi / 2) := by sorry

-- Note: The existence of a right angle is now explicitly stated as either B or C being π/2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1194_119467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1194_119435

theorem trigonometric_identities (θ : Real) 
  (h1 : Real.sin θ = 3/5) 
  (h2 : π/2 < θ ∧ θ < π) : 
  Real.tan θ = 3/4 ∧ Real.cos (2*θ - π/3) = (7 - 24*Real.sqrt 3)/50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1194_119435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_present_value_l1194_119423

/-- The present value that grows to a future value with compound interest -/
noncomputable def presentValue (futureValue : ℝ) (interestRate : ℝ) (years : ℕ) : ℝ :=
  futureValue / (1 + interestRate) ^ years

/-- Theorem: The present value that grows to $600,000 in 12 years with 6% annual compound interest is approximately $303,912.29 -/
theorem investment_present_value :
  let futureValue : ℝ := 600000
  let interestRate : ℝ := 0.06
  let years : ℕ := 12
  let calculatedValue := presentValue futureValue interestRate years
  ∃ ε > 0, |calculatedValue - 303912.29| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_present_value_l1194_119423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_value_max_area_l1194_119415

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the given condition
def condition (t : Triangle) : Prop :=
  2 * Real.sin (7 * Real.pi / 6) * Real.sin (Real.pi / 6 + t.C) + Real.cos t.C = -1/2

-- Theorem 1: Prove that C = π/3
theorem angle_C_value (t : Triangle) (h : condition t) : t.C = Real.pi/3 := by
  sorry

-- Theorem 2: Prove the maximum area when c = 2√3
theorem max_area (t : Triangle) (h : condition t) (hc : t.c = 2 * Real.sqrt 3) :
  (∃ (area : ℝ), area = (1/2) * t.a * t.b * Real.sin t.C ∧ 
   ∀ (other_area : ℝ), other_area = (1/2) * t.a * t.b * Real.sin t.C → other_area ≤ 3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_value_max_area_l1194_119415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_calculation_l1194_119480

theorem remainder_calculation (x y : ℕ) (h1 : (x : ℚ) / (y : ℚ) = 96.25) (h2 : y = 36) :
  x % y = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_calculation_l1194_119480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_counterexample_l1194_119488

-- Define the sample space
def Ω : Set Nat := {0, 1}

-- Define the σ-algebra
def F : Set (Set Nat) := {∅, Ω}

-- Define the function ξ
def ξ : Nat → ℝ
  | 0 => -1
  | 1 => 1
  | _ => 0 -- Add a default case for completeness

-- Define the measurability of a function
def is_measurable (f : Nat → ℝ) (A : Set (Set Nat)) : Prop :=
  ∀ c : ℝ, {ω ∈ Ω | f ω ≤ c} ∈ A

theorem exists_counterexample :
  ∃ (ξ : Nat → ℝ) (F : Set (Set Nat)),
    is_measurable (fun ω => |ξ ω|) F ∧ ¬is_measurable ξ F :=
by
  -- Use the definitions from above
  let ξ := ξ
  let F := F
  
  -- Prove that |ξ| is measurable
  have h1 : is_measurable (fun ω => |ξ ω|) F := by
    sorry -- Proof omitted
  
  -- Prove that ξ is not measurable
  have h2 : ¬is_measurable ξ F := by
    sorry -- Proof omitted
  
  -- Combine the results
  exact ⟨ξ, F, h1, h2⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_counterexample_l1194_119488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1194_119401

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a point lies on a line -/
def Line.containsPoint (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

/-- Converts an angle in degrees to radians -/
noncomputable def degToRad (deg : ℝ) : ℝ :=
  deg * (Real.pi / 180)

theorem line_equation_proof :
  ∃ (l : Line),
    l.containsPoint (-1) 2 ∧
    l.slope = Real.tan (degToRad 45) ∧
    ∀ (x y : ℝ), l.containsPoint x y ↔ x - y + 3 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1194_119401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1194_119483

/-- The speed of the train in km/hr -/
noncomputable def train_speed : ℝ := 120

/-- The time taken by the train to cross a pole in seconds -/
noncomputable def crossing_time : ℝ := 9

/-- Conversion factor from km/hr to m/s -/
noncomputable def km_hr_to_m_s : ℝ := 5 / 18

/-- The length of the train in meters -/
noncomputable def train_length : ℝ := train_speed * km_hr_to_m_s * crossing_time

theorem train_length_calculation :
  ∃ ε > 0, |train_length - 299.97| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1194_119483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l1194_119417

theorem max_m_value (a b m : ℝ) 
  (h1 : ∀ x, a*x^2 + b*x + 1 > 0 ↔ -1/2 < x ∧ x < 1) 
  (h2 : ∀ x ≥ 4, b*x^2 - m*x - 2*a ≥ 0) : 
  ∃ m_max : ℝ, m_max = 5 ∧ ∀ m', (∀ x ≥ 4, b*x^2 - m'*x - 2*a ≥ 0) → m' ≤ m_max :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l1194_119417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_ratio_l1194_119444

/-- The sum of a geometric series -/
noncomputable def geometricSum (a r : ℝ) : ℝ := a / (1 - r)

/-- The sum of odd-power terms in a geometric series -/
noncomputable def oddPowerSum (a r : ℝ) : ℝ := (a * r) / (1 - r^2)

/-- The sum of even-power terms in a geometric series -/
noncomputable def evenPowerSum (a r : ℝ) : ℝ := a / (1 - r^2)

theorem geometric_series_ratio (a r : ℝ) (hr : r ≠ 1) (hr2 : r^2 ≠ 1) :
  geometricSum a r = 24 ∧ oddPowerSum a r = 8 → r = 1/2 := by
  sorry

#check geometric_series_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_ratio_l1194_119444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_doubled_sides_l1194_119476

/-- Given a triangle with sides a, b, c and an angle θ between sides a and b,
    doubling all sides while keeping θ constant quadruples the area. -/
theorem triangle_area_doubled_sides (a b c θ : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hθ : 0 < θ ∧ θ < π) :
  (1/2) * (2*a) * (2*b) * Real.sin θ = 4 * ((1/2) * a * b * Real.sin θ) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_doubled_sides_l1194_119476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_is_30_percent_of_a_l1194_119400

-- Define variables
variable (a b c : ℝ)

-- Define conditions
def c_is_25_percent_of_b (c b : ℝ) : Prop := c = 0.25 * b
def b_is_120_percent_of_a (b a : ℝ) : Prop := b = 1.2 * a

-- Theorem statement
theorem c_is_30_percent_of_a 
  (h1 : c_is_25_percent_of_b c b)
  (h2 : b_is_120_percent_of_a b a) :
  c = 0.3 * a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_is_30_percent_of_a_l1194_119400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_is_seven_halves_l1194_119413

/-- Sequence b_n defined recursively -/
def b : ℕ → ℚ
  | 0 => 3  -- Added case for 0
  | 1 => 3
  | 2 => 5
  | (n + 3) => b (n + 2) + 2 * b (n + 1)

/-- The sum of the series -/
noncomputable def seriesSum : ℝ := ∑' n, (b n : ℝ) / 3^(n + 1)

/-- Theorem stating that the sum of the series equals 7/2 -/
theorem series_sum_is_seven_halves : seriesSum = 7/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_is_seven_halves_l1194_119413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equations_l1194_119458

theorem diophantine_equations :
  (∃ f : ℕ → ℕ × ℕ × ℕ, ∀ n : ℕ,
    let (x, y, z) := f n
    x^2 + 2*y^2 = z^2 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) ∧
  (¬ ∃ x y z t : ℤ,
    x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ t ≠ 0 ∧
    x^2 + 2*y^2 = z^2 ∧ 2*x^2 + y^2 = t^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equations_l1194_119458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_springs_stiffness_l1194_119449

/-- Two light springs with different stiffnesses -/
structure Springs where
  k₁ : ℝ
  k₂ : ℝ

/-- Problem parameters -/
noncomputable def m : ℝ := 3
noncomputable def g : ℝ := 10
noncomputable def x₁ : ℝ := 0.4
noncomputable def x₂ : ℝ := 0.075

/-- Combined stiffness when springs are in series -/
noncomputable def series_stiffness (s : Springs) : ℝ :=
  (s.k₁ * s.k₂) / (s.k₁ + s.k₂)

/-- Combined stiffness when springs are in parallel -/
noncomputable def parallel_stiffness (s : Springs) : ℝ :=
  s.k₁ + s.k₂

/-- The main theorem stating the stiffnesses of the springs -/
theorem springs_stiffness :
  ∃ s : Springs,
    series_stiffness s * x₁ = m * g ∧
    parallel_stiffness s * x₂ = m * g ∧
    s.k₁ = 300 ∧
    s.k₂ = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_springs_stiffness_l1194_119449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l1194_119440

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (a b c1 c2 : ℝ) : ℝ :=
  abs (c1 - c2) / Real.sqrt (a^2 + b^2)

/-- Proof that the distance between 3x + 4y + 3 = 0 and 6x + 8y + 11 = 0 is 1/2 -/
theorem distance_between_given_lines :
  distance_between_parallel_lines 3 4 3 (11/2) = 1/2 := by
  -- Unfold the definition of distance_between_parallel_lines
  unfold distance_between_parallel_lines
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check distance_between_given_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l1194_119440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_line_l1194_119454

/-- The circle C with equation x^2 + y^2 = 4 -/
def circleC (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The point P with coordinates (1, 1) -/
def point_P : ℝ × ℝ := (1, 1)

/-- The line L with equation x + y - 2 = 0 -/
def line_L (x y : ℝ) : Prop := x + y - 2 = 0

/-- The theorem stating that line L passing through point P cuts the shortest chord on circle C -/
theorem shortest_chord_line :
  (∀ x y : ℝ, line_L x y → (x, y) ≠ point_P → circleC x y) →
  (∀ m b : ℝ, m ≠ -1 → (∀ x y : ℝ, y = m * x + b → (x, y) ≠ point_P → circleC x y) →
    ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ line_L x₁ y₁ ∧ line_L x₂ y₂ ∧ circleC x₁ y₁ ∧ circleC x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 < (x - x₁)^2 + (y - y₁)^2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_line_l1194_119454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_efficiency_is_nine_l1194_119490

/-- Represents the fuel efficiency of a car -/
structure CarFuelEfficiency where
  highway_miles_per_tank : ℚ
  city_miles_per_tank : ℚ
  efficiency_difference : ℚ

/-- Calculates the city fuel efficiency of a car given its characteristics -/
noncomputable def city_fuel_efficiency (car : CarFuelEfficiency) : ℚ :=
  let tank_size := car.highway_miles_per_tank / (car.city_miles_per_tank / (car.highway_miles_per_tank / car.efficiency_difference + 1) + car.efficiency_difference)
  car.city_miles_per_tank / tank_size

/-- Theorem stating that for a car with given characteristics, its city fuel efficiency is 9 miles per gallon -/
theorem city_efficiency_is_nine :
  let car := CarFuelEfficiency.mk 560 336 6
  city_fuel_efficiency car = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_efficiency_is_nine_l1194_119490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_increase_twenty_percent_l1194_119465

/-- Represents the company's profits over three years -/
structure CompanyProfit where
  year1 : ℝ
  year2 : ℝ
  year3 : ℝ

/-- Conditions for the company's profits -/
def validProfit (p : CompanyProfit) : Prop :=
  p.year3 = 1.20 * p.year2 ∧ p.year3 = 1.44 * p.year1

/-- The percent increase from year1 to year2 -/
noncomputable def percentIncrease (p : CompanyProfit) : ℝ :=
  (p.year2 - p.year1) / p.year1 * 100

/-- Theorem stating that the percent increase from year1 to year2 is 20% -/
theorem profit_increase_twenty_percent (p : CompanyProfit) (h : validProfit p) :
  percentIncrease p = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_increase_twenty_percent_l1194_119465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_grid_theorem_l1194_119427

/-- A regular hexagon with side length 5 divided into equilateral triangles with side length 1 -/
structure HexagonGrid :=
  (side_length : ℝ)
  (triangle_side_length : ℝ)
  (is_regular : Bool)
  (is_divided : Bool)

/-- A node in the hexagon grid -/
structure Node :=
  (x : ℝ)
  (y : ℝ)
  (is_marked : Bool)

/-- The set of all nodes in the hexagon grid -/
def all_nodes (grid : HexagonGrid) : Finset Node := sorry

/-- The set of marked nodes in the hexagon grid -/
def marked_nodes (grid : HexagonGrid) : Finset Node := sorry

/-- Checks if five nodes lie on the same circle -/
def five_nodes_on_circle (nodes : Finset Node) : Prop := sorry

/-- The main theorem -/
theorem hexagon_grid_theorem (grid : HexagonGrid) 
  (h1 : grid.side_length = 5)
  (h2 : grid.triangle_side_length = 1)
  (h3 : grid.is_regular = true)
  (h4 : grid.is_divided = true)
  (h5 : (marked_nodes grid).card > (all_nodes grid).card / 2) :
  ∃ (circle_nodes : Finset Node), circle_nodes ⊆ marked_nodes grid ∧ 
                                  circle_nodes.card = 5 ∧ 
                                  five_nodes_on_circle circle_nodes := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_grid_theorem_l1194_119427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1194_119441

noncomputable def α : ℝ := Real.arctan (-3/4)

noncomputable def P : ℝ × ℝ := (-4, 3)
noncomputable def a : ℝ × ℝ := (3, 1)
noncomputable def b : ℝ × ℝ := (Real.sin α, Real.cos α)

theorem problem_solution :
  (P.2 / P.1 = -Real.tan α) ∧
  (a.1 * b.2 = a.2 * b.1) →
  ((Real.cos (π/2 + α) * Real.sin (-π - α)) / (Real.cos (11*π/2 - α) * Real.sin (9*π/2 + α)) = -3/4) ∧
  ((4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5/7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1194_119441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_intersection_l1194_119452

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 - y^2 = 2

-- Define the foci
def focus1 : ℝ × ℝ := (-2, 0)
def focus2 : ℝ × ℝ := (2, 0)

-- Define point P on the hyperbola
noncomputable def point_P : ℝ × ℝ := (3, Real.sqrt 7)

-- Define point A
def point_A : ℝ × ℝ := (0, 2)

-- Define the line l passing through A
def line_l (k : ℝ) (x : ℝ) : ℝ := k * x + 2

-- Define the area of triangle OEF
noncomputable def triangle_area (E F : ℝ × ℝ) : ℝ := abs (E.1 * F.2 - F.1 * E.2) / 2

theorem hyperbola_line_intersection :
  ∀ k : ℝ,
  hyperbola_C point_P.1 point_P.2 →
  (∃ E F : ℝ × ℝ,
    E ≠ F ∧
    hyperbola_C E.1 E.2 ∧
    hyperbola_C F.1 F.2 ∧
    E.2 = line_l k E.1 ∧
    F.2 = line_l k F.1 ∧
    triangle_area E F = 2 * Real.sqrt 2) →
  k = Real.sqrt 2 ∨ k = -Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_intersection_l1194_119452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_range_l1194_119489

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the directrix intersection with x-axis
def D : ℝ × ℝ := (-1, 0)

-- Define a line passing through the focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 1)

-- Define the area of triangle DAB
noncomputable def triangle_area (A B : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  abs ((x₁ - (-1)) * (y₂ - 0) - (x₂ - (-1)) * (y₁ - 0)) / 2

-- Theorem statement
theorem parabola_triangle_area_range :
  ∀ (k : ℝ) (A B : ℝ × ℝ),
    parabola A.1 A.2 →
    parabola B.1 B.2 →
    line_through_focus k A.1 A.2 →
    line_through_focus k B.1 B.2 →
    A ≠ B →
    triangle_area A B ≥ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_range_l1194_119489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_parallelogram_is_major_premise_l1194_119496

-- Define the types for our geometric shapes
def Rectangle : Type := Unit
def Parallelogram : Type := Unit
def Triangle : Type := Unit

-- Define the structure of a syllogism
structure Syllogism where
  major_premise : Prop
  minor_premise : Prop
  conclusion : Prop

-- Define the given syllogism
def given_syllogism : Syllogism :=
  { major_premise := (∀ x : Rectangle, ∃ y : Parallelogram, x = y),
    minor_premise := (∀ x : Triangle, ¬ ∃ y : Parallelogram, x = y),
    conclusion := (∀ x : Triangle, ¬ ∃ y : Rectangle, x = y) }

-- Define the property of being a major premise
def is_major_premise (s : Syllogism) (p : Prop) : Prop :=
  p = s.major_premise

-- Theorem statement
theorem rectangle_parallelogram_is_major_premise :
  is_major_premise given_syllogism (∀ x : Rectangle, ∃ y : Parallelogram, x = y) :=
by
  -- The proof is omitted for now
  sorry

#check rectangle_parallelogram_is_major_premise

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_parallelogram_is_major_premise_l1194_119496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1194_119471

open Real

theorem trig_identity (α : ℝ) : 
  (1 / sin (-α) - sin (π + α)) / (1 / cos (3*π - α) + cos (2*π - α)) = 1 / tan α ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1194_119471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_equals_five_l1194_119442

noncomputable def f (x : ℕ+) : ℝ :=
  1 / (((x.val^2 + 2*x.val + 1 : ℝ)^(1/3) : ℝ) + ((x.val^2 - 1 : ℝ)^(1/3) : ℝ) + ((x.val^2 - 2*x.val + 1 : ℝ)^(1/3) : ℝ))

noncomputable def sum_f : ℝ := (Finset.range 499).sum (fun k => f ⟨2*k+1, by {
  have h : 2*k+1 ≥ 1 := by {
    apply Nat.succ_le_of_lt
    exact Nat.zero_lt_succ (2*k)
  }
  exact h
}⟩) - f ⟨999, by norm_num⟩

theorem sum_f_equals_five : sum_f = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_equals_five_l1194_119442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_area_and_same_perimeter_l1194_119433

-- Define a right-angled triangle
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : c^2 = a^2 + b^2
  positive : a > 0 ∧ b > 0 ∧ c > 0

noncomputable def semi_perimeter (t : RightTriangle) : ℝ := (t.a + t.b + t.c) / 2

noncomputable def rho (t : RightTriangle) : ℝ := (t.a + t.b - t.c) / 2

theorem half_area_and_same_perimeter (t : RightTriangle) :
  let new_a := t.a - rho t
  let new_b := t.b - rho t
  let new_c := Real.sqrt (new_a^2 + new_b^2)
  let new_triangle := RightTriangle.mk new_a new_b new_c
    (by sorry) -- Proof that new_c^2 = new_a^2 + new_b^2
    (by sorry) -- Proof that new_a > 0 ∧ new_b > 0 ∧ new_c > 0
  let new_rectangle_perimeter := 2 * (t.a - rho t / 2) + 2 * (t.b - rho t / 2)
  (new_triangle.a * new_triangle.b) / 2 = (t.a * t.b) / 4 ∧
  new_rectangle_perimeter = t.a + t.b + t.c :=
by
  sorry -- Proof of the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_area_and_same_perimeter_l1194_119433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_triple_volume_diameter_l1194_119434

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

noncomputable def sphere_radius (v : ℝ) : ℝ := ((3 * v) / (4 * Real.pi))^(1/3)

def sphere_diameter (r : ℝ) : ℝ := 2 * r

theorem sphere_triple_volume_diameter :
  let r₁ : ℝ := 6
  let v₁ : ℝ := sphere_volume r₁
  let v₂ : ℝ := 3 * v₁
  let r₂ : ℝ := sphere_radius v₂
  let d₂ : ℝ := sphere_diameter r₂
  d₂ = 12 * (3 : ℝ)^(1/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_triple_volume_diameter_l1194_119434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_concentration_theorem_l1194_119429

/-- Represents the state of the two glasses --/
structure GlassState where
  water1 : ℝ
  alcohol1 : ℝ
  water2 : ℝ
  alcohol2 : ℝ

/-- Represents a transfer of liquid between glasses --/
structure Transfer where
  amount : ℝ
  fromFirst : Bool

/-- Calculates the new state after a transfer --/
def applyTransfer (state : GlassState) (transfer : Transfer) : GlassState :=
  sorry

/-- Calculates the alcohol concentration in a glass --/
noncomputable def alcoholConcentration (water : ℝ) (alcohol : ℝ) : ℝ :=
  alcohol / (water + alcohol)

/-- Theorem: The alcohol concentration in the first glass never exceeds that in the second glass --/
theorem alcohol_concentration_theorem (initialWater : ℝ) (initialAlcohol : ℝ) 
  (transfers : List Transfer) (h1 : initialWater > 0) (h2 : initialAlcohol > 0) :
  let finalState := transfers.foldl applyTransfer { water1 := initialWater, alcohol1 := 0, 
                                                    water2 := 0, alcohol2 := initialAlcohol }
  alcoholConcentration finalState.water1 finalState.alcohol1 ≤ 
  alcoholConcentration finalState.water2 finalState.alcohol2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_concentration_theorem_l1194_119429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_preimage_of_compact_is_compact_l1194_119484

-- Define the metric spaces X and Y
variable {X Y : Type*} [MetricSpace X] [MetricSpace Y]

-- Define the continuous function f: X → Y
variable (f : X → Y) (hf : Continuous f)

-- Define the function f₁: X × ℝ → Y × ℝ
def f₁ (f : X → Y) : X × ℝ → Y × ℝ := λ (x, t) ↦ (f x, t)

-- State the theorem
theorem preimage_of_compact_is_compact
  (hf₁ : IsClosed (Set.range (f₁ f)))
  (K : Set Y) (hK : IsCompact K) :
  IsCompact (f ⁻¹' K) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_preimage_of_compact_is_compact_l1194_119484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_averages_permutation_l1194_119497

/-- A permutation of the first n positive integers. -/
def IsValidPermutation (n : ℕ) (a : ℕ → ℕ) : Prop :=
  (∀ i, i < n → a i < n) ∧ Function.Injective a

/-- The property that all partial sums divided by their length are integers. -/
def HasIntegerAverages (n : ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ k, k ∈ Finset.range n → (Finset.sum (Finset.range k) a) % k = 0

/-- The main theorem stating that only n = 1 and n = 3 satisfy the conditions. -/
theorem integer_averages_permutation (n : ℕ) :
  (n > 0 ∧ ∃ a : ℕ → ℕ, IsValidPermutation n a ∧ HasIntegerAverages n a) ↔ n = 1 ∨ n = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_averages_permutation_l1194_119497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l1194_119421

/-- An arithmetic sequence -/
structure ArithmeticSequence (α : Type*) [Add α] [Mul α] where
  first : α
  common_difference : α

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def sum_n_terms (seq : ArithmeticSequence ℝ) (n : ℕ) : ℝ :=
  n * (2 * seq.first + (n - 1) * seq.common_difference) / 2

/-- Theorem about the ratio of nth terms of two arithmetic sequences -/
theorem arithmetic_sequence_ratio (a b : ArithmeticSequence ℝ) (n : ℕ) :
  (sum_n_terms a n) / (sum_n_terms b n) = 2 * n / (3 * n + 1) →
  a.first / b.first = (2 * n - 1) / (3 * n - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l1194_119421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1194_119426

noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) / (x^2 + 4)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem function_properties (a b : ℝ) :
  (∀ x, x ∈ Set.Icc (-2 : ℝ) 2 → is_odd (f a b))
  → f a b (-2) = -1/4
  → (a = 1 ∧ b = 0)
  ∧ (∀ x y, x ∈ Set.Icc (-2 : ℝ) 2 → y ∈ Set.Icc (-2 : ℝ) 2 → x < y → f a b x < f a b y)
  ∧ (∀ m : ℝ, (∀ x t, x ∈ Set.Icc (-2 : ℝ) 2 → t ∈ Set.Icc (-2 : ℝ) 2 → f a b x ≤ m^2 - m*t - 11/4) 
    → m ≤ -3 ∨ m ≥ 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1194_119426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1194_119460

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (x / 2 + Real.pi / 6) + 3

theorem function_properties :
  let period := 4 * Real.pi
  let amplitude := 3
  let initial_phase := Real.pi / 6
  let axis_of_symmetry (k : ℤ) := 2 * (k : ℝ) * Real.pi + 2 * Real.pi / 3
  ∀ x : ℝ,
    (∀ k : ℤ, f (x + period) = f x) ∧
    (abs (f x - 3) ≤ amplitude) ∧
    (f (2 * initial_phase) = 3) ∧
    (∀ k : ℤ, ∀ t : ℝ, f (axis_of_symmetry k + t) = f (axis_of_symmetry k - t)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1194_119460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_arithmetic_sequence_l1194_119466

theorem sin_arithmetic_sequence (a : ℝ) :
  0 < a * (π / 180) ∧ a * (π / 180) < 2 * π →
  (∃ r : ℝ, Real.sin (a * (π / 180)) + r = Real.sin (2 * a * (π / 180)) ∧
               Real.sin (2 * a * (π / 180)) + r = Real.sin (3 * a * (π / 180))) →
  a = 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_arithmetic_sequence_l1194_119466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1194_119445

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The focal length of a hyperbola -/
noncomputable def focal_length (h : Hyperbola) : ℝ := Real.sqrt (h.a^2 + h.b^2)

/-- Check if a point (x, y) lies on the asymptote of a hyperbola -/
def on_asymptote (h : Hyperbola) (x y : ℝ) : Prop := y * h.a = x * h.b

/-- The main theorem -/
theorem hyperbola_equation (h : Hyperbola) 
  (h_focal : focal_length h = 10) 
  (h_asymptote : on_asymptote h 2 1) : 
  h.a = 2 * Real.sqrt 5 ∧ h.b = Real.sqrt 5 := by
  sorry

#check hyperbola_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1194_119445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_monotonic_increasing_l1194_119408

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 * Real.sin x + Real.cos x) * (Real.sqrt 3 * Real.cos x - Real.sin x)

-- Theorem for the smallest positive period
theorem f_period : ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧ T = Real.pi := by
  sorry

-- Theorem for the monotonic increasing interval
theorem f_monotonic_increasing (k : ℤ) : 
  StrictMonoOn f (Set.Icc (k * Real.pi - 5 * Real.pi / 12) (k * Real.pi + Real.pi / 12)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_monotonic_increasing_l1194_119408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_g_well_defined_l1194_119414

-- Define the function f with domain [0,8]
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domainF : Set ℝ := Set.Icc 0 8

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f (2 * x) / (3 - x)

-- Define the domain of g
def domainG : Set ℝ := Set.Ioc 0 3 ∪ Set.Ioc 3 4

-- Theorem statement
theorem domain_of_g :
  ∀ x, x ∈ domainG ↔ x ∈ Set.Ioc 0 3 ∪ Set.Ioc 3 4 :=
by sorry

-- Theorem to show that g is well-defined on its domain
theorem g_well_defined :
  ∀ x ∈ domainG, 2 * x ∈ domainF ∧ 3 - x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_g_well_defined_l1194_119414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_return_speed_l1194_119457

/-- Proves that the average speed for the return trip is 10 miles per hour given the conditions of the cycling problem. -/
theorem cyclist_return_speed (total_distance : ℝ) (first_speed : ℝ) (second_speed : ℝ) (total_time : ℝ) : 
  total_distance = 32 →
  first_speed = 8 →
  second_speed = 10 →
  total_time = 6.8 →
  (total_distance / 2 / first_speed + total_distance / 2 / second_speed + total_distance / (total_distance / (total_time - (total_distance / 2 / first_speed + total_distance / 2 / second_speed)))) = total_time →
  total_distance / (total_time - (total_distance / 2 / first_speed + total_distance / 2 / second_speed)) = 10 := by
  sorry

#check cyclist_return_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_return_speed_l1194_119457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_packing_height_difference_l1194_119486

/-- The diameter of each cylindrical rod in cm -/
noncomputable def rod_diameter : ℝ := 8

/-- The number of rods in each container -/
def num_rods : ℕ := 150

/-- The height of the vertical stacking in Container X -/
noncomputable def height_x : ℝ := num_rods * rod_diameter

/-- The vertical distance between centers of rods in successive rows in hexagonal close packing -/
noncomputable def d : ℝ := (Real.sqrt 3 / 2) * rod_diameter

/-- The height of the hexagonal close packing in Container Y -/
noncomputable def height_y : ℝ := (num_rods - 1 : ℝ) * d + rod_diameter

/-- The positive difference in total heights between the two containers -/
noncomputable def height_difference : ℝ := height_x - height_y

theorem packing_height_difference :
  height_difference = 1192 - 596 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_packing_height_difference_l1194_119486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l1194_119479

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x - Real.pi / 3)

theorem max_omega_value (ω : ℝ) :
  ω > 0 →
  f ω (2 * Real.pi / 3) = f ω (5 * Real.pi / 6) →
  (∃ (max_val : ℝ), ∀ x ∈ Set.Ioo (2 * Real.pi / 3) (5 * Real.pi / 6), f ω x ≤ max_val) →
  (¬∃ (min_val : ℝ), ∀ x ∈ Set.Ioo (2 * Real.pi / 3) (5 * Real.pi / 6), min_val ≤ f ω x) →
  (∀ ω' > ω, ¬(f ω' (2 * Real.pi / 3) = f ω' (5 * Real.pi / 6) ∧
    (∃ (max_val : ℝ), ∀ x ∈ Set.Ioo (2 * Real.pi / 3) (5 * Real.pi / 6), f ω' x ≤ max_val) ∧
    (¬∃ (min_val : ℝ), ∀ x ∈ Set.Ioo (2 * Real.pi / 3) (5 * Real.pi / 6), min_val ≤ f ω' x))) →
  ω = 100 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l1194_119479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_reaches_height_l1194_119462

/-- The time at which a projectile reaches a specific height -/
noncomputable def time_to_height (a b c h : ℝ) : ℝ := 
  let t₁ := (b - Real.sqrt (b^2 - 4*a*(c-h))) / (2*a)
  let t₂ := (b + Real.sqrt (b^2 - 4*a*(c-h))) / (2*a)
  min t₁ t₂

/-- Theorem: The projectile reaches 35 meters at 10/7 seconds -/
theorem projectile_reaches_height :
  let a : ℝ := -4.9
  let b : ℝ := 29.5
  let c : ℝ := 0
  let h : ℝ := 35
  time_to_height a b c h = 10/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_reaches_height_l1194_119462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l1194_119420

theorem min_value_expression (a b c d : ℚ) : 
  a ∈ ({2, 3, 5, 7} : Set ℚ) → 
  b ∈ ({2, 3, 5, 7} : Set ℚ) → 
  c ∈ ({2, 3, 5, 7} : Set ℚ) → 
  d ∈ ({2, 3, 5, 7} : Set ℚ) → 
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d → 
  (((a + b) / (c - d)) / 2) ≥ -5/4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l1194_119420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_calculation_l1194_119499

noncomputable section

-- Define the total distance in miles
def total_distance : ℝ := 200

-- Define the total time in hours
def total_time : ℝ := 7

-- Define the average speed
def average_speed : ℝ := total_distance / total_time

-- Theorem to prove
theorem average_speed_calculation :
  average_speed = 28 + 4 / 7 :=
by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Simplify the fraction
  norm_num
  -- The proof is complete
  rfl

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_calculation_l1194_119499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_theorem_l1194_119450

/-- Represents a right circular cone water tank -/
structure WaterTank where
  baseRadius : ℝ
  height : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (r : ℝ) (h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- Calculates the height of water in the tank given a fill percentage -/
noncomputable def waterHeight (tank : WaterTank) (fillPercentage : ℝ) : ℝ :=
  tank.height * (fillPercentage)^(1/3)

theorem water_height_theorem (tank : WaterTank) (h : tank.baseRadius = 20 ∧ tank.height = 120) :
  waterHeight tank 0.4 = 48 * Real.rpow 2 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_theorem_l1194_119450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_proportional_to_volume_l1194_119463

/-- Represents a cylindrical jar with its dimensions and price -/
structure Jar where
  diameter : ℝ
  height : ℝ
  price : ℝ

/-- Calculates the volume of a cylindrical jar -/
noncomputable def jarVolume (j : Jar) : ℝ := Real.pi * (j.diameter / 2) ^ 2 * j.height

theorem price_proportional_to_volume (j1 j2 : Jar) 
  (h_diameter : j2.diameter = 2 * j1.diameter)
  (h_height : j2.height = 2 * j1.height)
  (h_price_prop : ∃ k : ℝ, j1.price = k * jarVolume j1 ∧ j2.price = k * jarVolume j2) :
  j2.price = 8 * j1.price := by
  sorry

#check price_proportional_to_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_proportional_to_volume_l1194_119463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_steps_approx_l1194_119448

/-- Represents a position on the 10x10 grid -/
structure Position where
  x : Fin 10
  y : Fin 10

/-- Represents the state of the pigeon and burrito -/
structure State where
  pigeon : Position
  burrito : Position
  burritoSize : ℚ

/-- Calculates the Manhattan distance between two positions -/
def manhattanDistance (p1 p2 : Position) : ℕ :=
  (Int.natAbs (p1.x.val - p2.x.val)) + (Int.natAbs (p1.y.val - p2.y.val))

/-- Simulates one step of the pigeon-burrito process -/
def simulateStep (s : State) : State :=
  sorry

/-- Calculates the expected number of steps until the burrito is fully eaten -/
noncomputable def expectedSteps : ℝ :=
  sorry

theorem expected_steps_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |expectedSteps - 71.8| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_steps_approx_l1194_119448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angles_trig_l1194_119418

theorem acute_angles_trig (α β : ℝ) 
  (h_acute_α : 0 < α ∧ α < Real.pi / 2)
  (h_acute_β : 0 < β ∧ β < Real.pi / 2)
  (h_sin_α : Real.sin α = 4 / 5)
  (h_cos_sum : Real.cos (α + β) = 5 / 13) : 
  Real.tan (2 * α) = -24 / 7 ∧ Real.sin β = 16 / 65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angles_trig_l1194_119418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equality_l1194_119453

theorem trigonometric_equality (θ φ : Real) :
  (Real.cos θ ^ 6 / Real.cos φ ^ 2 + Real.sin θ ^ 6 / Real.sin φ ^ 2 = 2) →
  (Real.sin φ ^ 6 / Real.sin θ ^ 2 + Real.cos φ ^ 6 / Real.cos θ ^ 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equality_l1194_119453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l1194_119416

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 - y - 2 * Real.log (Real.sqrt x) = 0

-- Define the line
def line (x y : ℝ) : Prop := 4 * x + 4 * y + 1 = 0

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (4 * x + 4 * y + 1) / Real.sqrt 32

-- Theorem statement
theorem min_distance_to_line :
  ∀ x y : ℝ, curve x y →
    (∀ x' y' : ℝ, curve x' y' → distance_to_line x y ≤ distance_to_line x' y') →
    distance_to_line x y = (Real.sqrt 2 / 2) * (1 + Real.log 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l1194_119416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_28_equals_inverse_of_one_minus_x_l1194_119447

noncomputable def f₁ (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

noncomputable def f_n : ℕ → (ℝ → ℝ)
  | 0 => id
  | 1 => f₁
  | n + 1 => λ x => f₁ (f_n n x)

theorem f_28_equals_inverse_of_one_minus_x :
  ∀ x : ℝ, x ≠ -1 → x ≠ 1 → f_n 35 = f_n 5 → f_n 28 x = 1 / (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_28_equals_inverse_of_one_minus_x_l1194_119447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_with_zero_derivative_l1194_119419

open Set

theorem constant_function_with_zero_derivative 
  {f : ℝ → ℝ} {a b : ℝ} (hab : a ≤ b) 
  (hf : DifferentiableOn ℝ f (Icc a b))
  (hf' : ∀ x, x ∈ Icc a b → deriv f x = 0) :
  ∀ x y, x ∈ Icc a b → y ∈ Icc a b → f x = f y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_with_zero_derivative_l1194_119419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_weighings_correct_l1194_119422

/-- Given a number of coins, returns the minimum number of weighings needed to find the counterfeit coin -/
noncomputable def min_weighings (n : ℕ) : ℕ :=
  ⌈(Real.log (n : ℝ) / Real.log 3 : ℝ)⌉.toNat

/-- Proves that the minimum number of weighings to find a counterfeit coin among n coins is ⌈log₃ n⌉ -/
theorem min_weighings_correct (n : ℕ) (h : n > 0) :
  ∀ k : ℕ, (∀ m : ℕ, m ≤ n → (3 : ℝ)^(k-1) < m → m ≤ (3 : ℝ)^k) ↔ k = min_weighings n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_weighings_correct_l1194_119422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l1194_119493

noncomputable def f (x : ℝ) := Real.exp (x * Real.log 3) + 3 * x - 8

theorem root_exists_in_interval :
  ∀ (a b c : ℝ), 1 < a ∧ a < b ∧ b < c ∧ c < 2 →
  ContinuousOn f (Set.Icc 1 2) →
  f 1 < 0 →
  f a < 0 →
  f c > 0 →
  ∃ x, x ∈ Set.Ioo a c ∧ f x = 0 :=
by
  sorry

#check root_exists_in_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l1194_119493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_calculation_l1194_119404

/-- Work done by a force F on a mass m moving along the x-axis -/
noncomputable def work (F : ℝ → ℝ) (a b : ℝ) : ℝ := ∫ x in a..b, F x

/-- The force function F(x) = x^2 + 1 -/
def F (x : ℝ) : ℝ := x^2 + 1

theorem work_calculation :
  work F 1 10 = 342 := by
  -- Unfold the definition of work
  unfold work
  -- Evaluate the integral
  simp [F]
  -- The rest of the proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_calculation_l1194_119404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_dimension_theorem_l1194_119455

/-- The dimension z of a hexagon obtained from cutting a 7×21 rectangle -/
noncomputable def z : ℝ := (7 * Real.sqrt 3) / 3

/-- The side length of the square formed by repositioning the hexagons -/
noncomputable def s : ℝ := Real.sqrt 147

theorem hexagon_dimension_theorem :
  let rectangle_area : ℝ := 7 * 21
  let square_area : ℝ := s ^ 2
  rectangle_area = square_area ∧ z = s / 3 → z = (7 * Real.sqrt 3) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_dimension_theorem_l1194_119455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_l1194_119436

theorem tank_capacity (initial_fill final_fill water_added capacity : ℝ) :
  initial_fill = 1/8 →
  final_fill = 2/3 →
  water_added = 150 →
  capacity = 277 →
  (initial_fill * capacity) + water_added = final_fill * capacity := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_l1194_119436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_l1194_119469

noncomputable section

-- Define the cone
structure Cone where
  lateralArea : ℝ
  slantBaseAngle : ℝ

-- Define our specific cone
def myCone : Cone where
  lateralArea := 9 * Real.sqrt 2 * Real.pi
  slantBaseAngle := Real.pi / 4

-- Theorem statement
theorem cone_volume (c : Cone) (h : c = myCone) : 
  ∃ (volume : ℝ), volume = 9 * Real.pi := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_l1194_119469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_transformation_l1194_119428

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (∀ x : ℂ, x^3 - 3*x^2 + 5 = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) →
  (∀ x : ℂ, x^3 - 9*x^2 + 135 = 0 ↔ x = 3*r₁ ∨ x = 3*r₂ ∨ x = 3*r₃) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_transformation_l1194_119428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_manufacturing_department_percentage_l1194_119451

/-- The percentage of employees in a department represented by a sector in a circle graph -/
noncomputable def department_percentage (sector_degrees : ℝ) : ℝ :=
  (sector_degrees / 360) * 100

/-- Theorem: The percentage of employees in a department represented by a sector of 18° in a circle graph is 5% -/
theorem manufacturing_department_percentage :
  department_percentage 18 = 5 := by
  -- Unfold the definition of department_percentage
  unfold department_percentage
  -- Simplify the expression
  simp [div_mul_eq_mul_div]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_manufacturing_department_percentage_l1194_119451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l1194_119431

theorem angle_in_second_quadrant (θ : Real) (h1 : Real.sin θ > 0) (h2 : Real.cos θ < 0) :
  ∃ (α : Real), 0 < α ∧ α < Real.pi ∧ θ = α + 2 * Real.pi * Int.floor (θ / (2 * Real.pi)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l1194_119431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_milk_tea_sales_l1194_119411

theorem chocolate_milk_tea_sales (total : ℕ) (winter_melon_ratio : ℚ) (okinawa_ratio : ℚ) 
  (h1 : total = 50)
  (h2 : winter_melon_ratio = 2/5)
  (h3 : okinawa_ratio = 3/10)
  (h4 : winter_melon_ratio + okinawa_ratio < 1) :
  total - (winter_melon_ratio * ↑total).floor - (okinawa_ratio * ↑total).floor = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_milk_tea_sales_l1194_119411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l1194_119464

-- Define the points
def P : ℝ × ℝ := (2, 3)
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, -3)

-- Define the slope angle θ
def θ_range (x : ℝ) : Prop := 0 ≤ x ∧ x < Real.pi

-- Define the condition that line l intersects with line segment AB
def intersects_AB (θ : ℝ) : Prop :=
  (Real.tan θ ≥ 2) ∨ (Real.tan θ ≤ -1)

-- Theorem statement
theorem slope_angle_range :
  ∀ θ, θ_range θ → intersects_AB θ →
  θ ∈ Set.Icc (Real.arctan 2) (3 * Real.pi / 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l1194_119464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divisible_by_three_l1194_119485

/-- Number formed by arranging 1 to 2018 in order -/
def N : ℕ := 
  -- Implementation details omitted for brevity
  sorry

/-- Function to count occurrences of 8 in a number -/
def count_8 (n : ℕ) : ℕ := 
  sorry

/-- Function to sum digits of a number -/
def sum_digits (n : ℕ) : ℕ := 
  sorry

/-- Function to remove all 8s from a number -/
def remove_8 (n : ℕ) : ℕ := 
  sorry

/-- Theorem stating that the number formed after removing 8s is not divisible by 3 -/
theorem not_divisible_by_three : ¬(∃ k : ℕ, remove_8 N = 3 * k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divisible_by_three_l1194_119485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_side_relation_l1194_119409

/-- Given a triangle ABC where 4A = B = C, prove that a^3 + b^3 = 3ab^2 -/
theorem triangle_angle_side_relation (A B C a b c : ℝ) 
  (h_triangle : A + B + C = π) 
  (h_angles : 4 * A = B ∧ B = C) 
  (h_sides : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C) : 
  a^3 + b^3 = 3 * a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_side_relation_l1194_119409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1194_119456

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := (Real.sin (ω * x) + Real.cos (ω * x))^2 + 2 * (Real.cos (ω * x))^2

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := f ω (x - Real.pi / 2)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem problem_solution (ω : ℝ) (h1 : ω > 0) (h2 : has_period (f ω) (2 * Real.pi / 3)) :
  ω = 3 / 2 ∧
  ∀ k : ℤ, StrictMonoOn (g ω) (Set.Icc (2 * Real.pi * k / 3 + Real.pi / 4) (2 * Real.pi * k / 3 + 7 * Real.pi / 12)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1194_119456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_mixture_percentage_l1194_119424

/-- Proves that a 15-liter mixture of alcohol and water, when mixed with 2 liters of water,
    results in a new mixture with 17.647058823529413% alcohol if and only if
    the original mixture contained 20% alcohol. -/
theorem alcohol_mixture_percentage : 
  ∀ (original_percentage : ℝ),
  (original_percentage / 100 * 15 = 17.647058823529413 / 100 * 17) ↔
  (original_percentage = 20) := by
  intro original_percentage
  have h1 : (original_percentage / 100 * 15 = 17.647058823529413 / 100 * 17) ↔
            (original_percentage = 20) := by
    apply Iff.intro
    · intro h
      -- Here you would prove that if the equation holds, then original_percentage = 20
      sorry
    · intro h
      -- Here you would prove that if original_percentage = 20, then the equation holds
      sorry
  exact h1

#check alcohol_mixture_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_mixture_percentage_l1194_119424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_for_quadratic_inequality_l1194_119468

theorem x_range_for_quadratic_inequality (a : ℝ) (h : a ∈ Set.Icc 1 3) :
  (∀ x : ℝ, a * x^2 + (a - 2) * x - 2 > 0) →
  (∀ x : ℝ, x < -1 ∨ x > 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_for_quadratic_inequality_l1194_119468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friends_meet_after_eight_hours_meeting_point_closer_to_X_l1194_119403

/-- The distance between X and Y in miles -/
noncomputable def total_distance : ℝ := 120

/-- A's constant speed in miles per hour -/
noncomputable def speed_A : ℝ := 5

/-- B's initial speed in miles per hour -/
noncomputable def initial_speed_B : ℝ := 4

/-- B's speed increase per hour in miles per hour -/
noncomputable def speed_increase_B : ℝ := 0.75

/-- The time it takes for A and B to meet in hours -/
noncomputable def meeting_time : ℝ := 8

/-- The distance traveled by A in miles -/
noncomputable def distance_A (t : ℝ) : ℝ := speed_A * t

/-- The distance traveled by B in miles -/
noncomputable def distance_B (t : ℝ) : ℝ := t / 2 * (2 * initial_speed_B + (t - 1) * speed_increase_B)

/-- Theorem stating that A and B meet after 8 hours -/
theorem friends_meet_after_eight_hours :
  distance_A meeting_time + distance_B meeting_time = total_distance := by
  sorry

/-- Theorem stating that the meeting point is closer to X than Y -/
theorem meeting_point_closer_to_X :
  ∃ (x : ℕ), distance_A meeting_time = (total_distance / 2 : ℝ) - (x : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friends_meet_after_eight_hours_meeting_point_closer_to_X_l1194_119403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_23pi_6_l1194_119446

noncomputable def periodicity (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + Real.pi) = f x + Real.sin x

theorem f_value_at_23pi_6
  (f : ℝ → ℝ)
  (h1 : periodicity f)
  (h2 : ∀ x : ℝ, 0 ≤ x → x < Real.pi → f x = 0) :
  f (23 * Real.pi / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_23pi_6_l1194_119446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_count_l1194_119459

theorem equation_solutions_count :
  ∃! (s : Finset ℝ), (∀ x ∈ s, (2 * x^2 - 7)^2 = 49) ∧ s.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_count_l1194_119459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base6_difference_theorem_l1194_119439

/-- Represents a single digit in base 6 -/
def Base6Digit := Fin 6

/-- Converts a natural number to its base 6 representation -/
def toBase6 (n : ℕ) : List Base6Digit := sorry

/-- Converts a list of base 6 digits to a natural number -/
def fromBase6 (l : List Base6Digit) : ℕ := sorry

/-- Performs addition in base 6 -/
def addBase6 (a b : List Base6Digit) : List Base6Digit := sorry

/-- Helper function to create a Base6Digit from a Nat -/
def natToBase6Digit (n : ℕ) : Base6Digit :=
  ⟨n % 6, by exact Nat.mod_lt n (Nat.zero_lt_succ 5)⟩

theorem base6_difference_theorem (C D : Base6Digit) :
  let CDC : List Base6Digit := [C, D, D]
  let D52 : List Base6Digit := [natToBase6Digit 5, natToBase6Digit 2, D]
  let C34 : List Base6Digit := [C, natToBase6Digit 3, natToBase6Digit 4]
  let C213 : List Base6Digit := [C, natToBase6Digit 2, natToBase6Digit 1, natToBase6Digit 3]
  addBase6 (addBase6 CDC D52) C34 = C213 →
  toBase6 (Int.natAbs (fromBase6 [C] - fromBase6 [D])) = [natToBase6Digit 2] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base6_difference_theorem_l1194_119439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_ratio_l1194_119475

/-- The ratio of the area of a regular octagon circumscribed about a circle
    to the area of a regular octagon inscribed in the same circle is 1. -/
theorem octagon_area_ratio (r : ℝ) (r_pos : r > 0) : 
  (let inscribed_side := r * Real.sqrt 2
   let circumscribed_side := r * Real.sqrt 2
   let octagon_area (s : ℝ) := 2 * (1 + Real.sqrt 2) * s^2
   octagon_area circumscribed_side / octagon_area inscribed_side) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_ratio_l1194_119475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1194_119405

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) - Real.sqrt 3 * Real.cos (ω * x)

theorem function_properties (ω : ℝ) (h_ω_pos : ω > 0) 
  (h_period : ∀ x, f ω (x + π) = f ω x) :
  ω = 2 ∧ f ω (-5 * π / 6) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1194_119405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_functional_equation_l1194_119470

theorem polynomial_functional_equation (P : ℝ → ℝ) :
  (∀ x : ℝ, P (x^2) = x * P x) ↔ (∃ c : ℝ, ∀ x : ℝ, P x = c * x) ∨ (∀ x : ℝ, P x = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_functional_equation_l1194_119470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1194_119402

/-- Represents an ellipse with foci on the x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_a_gt_b : a > b
  h_b_gt_0 : b > 0

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The standard equation of an ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Theorem stating the properties of the specific ellipse -/
theorem ellipse_properties :
  ∃ (e : Ellipse),
    e.equation (-3) 0 ∧
    e.eccentricity = Real.sqrt 5 / 3 ∧
    e.equation = fun x y ↦ x^2 / 9 + y^2 / 4 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1194_119402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apothem_inequality_l1194_119461

/-- Apothem of a regular k-gon inscribed in a circle of radius R -/
noncomputable def apothem (k : ℕ) (R : ℝ) : ℝ := R * Real.cos (Real.pi / k)

/-- Theorem: For any positive integer n and radius R, the inequality holds -/
theorem apothem_inequality (n : ℕ) (R : ℝ) (h_n_pos : n > 0) (h_R_pos : R > 0) :
  (n + 1 : ℝ) * apothem (n + 1) R - n * apothem n R > R := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apothem_inequality_l1194_119461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_sum_l1194_119494

/-- Given a function f(x) = ax^5 + b*sin(x) + c, prove that if f(-1) + f(1) = 2, then c = 1 -/
theorem function_value_sum (a b c : ℝ) :
  (fun (x : ℝ) => a * x^5 + b * Real.sin x + c) (-1) +
  (fun (x : ℝ) => a * x^5 + b * Real.sin x + c) 1 = 2 →
  c = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_sum_l1194_119494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_euler_characteristic_l1194_119473

/-- A rectangular prism is a three-dimensional shape with six rectangular faces. -/
structure RectangularPrism where
  -- We don't need to define any specific properties for this problem

/-- The number of faces in a rectangular prism -/
def number_of_faces (rp : RectangularPrism) : ℕ := 6

/-- The number of edges in a rectangular prism -/
def number_of_edges (rp : RectangularPrism) : ℕ := 12

/-- The number of vertices in a rectangular prism -/
def number_of_vertices (rp : RectangularPrism) : ℕ := 8

/-- The Euler characteristic of a rectangular prism is the sum of its faces, edges, and vertices. -/
def euler_characteristic (rp : RectangularPrism) : ℕ :=
  number_of_faces rp + number_of_edges rp + number_of_vertices rp

/-- The Euler characteristic of a rectangular prism is 26 -/
theorem rectangular_prism_euler_characteristic (rp : RectangularPrism) :
  euler_characteristic rp = 26 := by
  unfold euler_characteristic
  unfold number_of_faces
  unfold number_of_edges
  unfold number_of_vertices
  simp
  -- The proof is completed by simplification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_euler_characteristic_l1194_119473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_blue_tile_is_seven_fiftieths_l1194_119437

/-- A tile is blue if its number is congruent to 3 mod 7 -/
def is_blue (n : ℕ) : Bool := n % 7 = 3

/-- The count of blue tiles in the range 1 to 100 -/
def count_blue_tiles : ℕ := (Finset.range 100).filter (fun n => is_blue n) |>.card

/-- The probability of selecting a blue tile -/
def prob_blue_tile : ℚ := count_blue_tiles / 100

/-- Theorem stating that the probability of selecting a blue tile is 7/50 -/
theorem prob_blue_tile_is_seven_fiftieths : prob_blue_tile = 7 / 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_blue_tile_is_seven_fiftieths_l1194_119437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_chest_contents_l1194_119492

-- Define the types of coins
inductive CoinType
| Gold
| Silver
| Copper

-- Define the chests
structure Chest where
  label : CoinType
  content : CoinType

-- Define the problem setup
def chestSetup : List Chest :=
  [
    { label := CoinType.Gold, content := CoinType.Silver },
    { label := CoinType.Silver, content := CoinType.Gold },
    { label := CoinType.Gold, content := CoinType.Copper }
  ]

-- The theorem to prove
theorem correct_chest_contents :
  ∀ (setup : List Chest),
    setup.length = 3 →
    (∀ chest, chest ∈ setup → chest.label ≠ chest.content) →
    (∃! chest, chest ∈ setup ∧ chest.content = CoinType.Gold) →
    (∃! chest, chest ∈ setup ∧ chest.content = CoinType.Silver) →
    (∃! chest, chest ∈ setup ∧ chest.content = CoinType.Copper) →
    setup = chestSetup := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_chest_contents_l1194_119492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sister_age_this_year_l1194_119472

-- Define the ages of the sisters as functions of the year
def younger_sister_age : ℕ → ℕ := sorry
def older_sister_age : ℕ → ℕ := sorry

-- Define the current year
def current_year : ℕ := sorry

-- Conditions
axiom condition1 : older_sister_age current_year = 3 * younger_sister_age current_year
axiom condition2 : older_sister_age (current_year + 2) = 2 * younger_sister_age (current_year + 2)

-- Theorem to prove
theorem sister_age_this_year : older_sister_age current_year = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sister_age_this_year_l1194_119472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_l1194_119474

/-- Represents the total number of voters -/
def n : ℕ → ℕ := id

/-- Represents the number of voters in the smaller district -/
def m : ℕ → ℕ := id

/-- Represents the probability of Miraflores winning in the larger district -/
noncomputable def prob_win_larger (n m : ℕ) : ℚ := (n - m : ℚ) / (2*n - m : ℚ)

/-- Represents the probability of Miraflores winning in the smaller district -/
noncomputable def prob_win_smaller (m : ℕ) : ℚ := 1 / m

/-- Represents the total probability of Miraflores winning the election -/
noncomputable def total_prob (n m : ℕ) : ℚ := (prob_win_larger n m) * (prob_win_smaller m)

theorem optimal_strategy (n : ℕ) (h : n > 0) :
  ∀ m : ℕ, 0 < m ∧ m < 2*n → total_prob n m ≤ total_prob n 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_l1194_119474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_purchase_frequency_and_discount_l1194_119495

/-- Represents the daily cost function without discount -/
noncomputable def daily_cost (n : ℝ) : ℝ := n + 100 / n + 1501

/-- Represents the daily cost function with discount -/
noncomputable def daily_cost_discounted (m : ℝ) : ℝ := m + 100 / m + 1426

theorem optimal_purchase_frequency_and_discount :
  (∀ n : ℝ, n > 0 → daily_cost n ≥ daily_cost 10) ∧
  (∀ m : ℝ, m ≥ 20 → daily_cost_discounted m < daily_cost 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_purchase_frequency_and_discount_l1194_119495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_polynomial_and_multiple_l1194_119430

theorem gcd_polynomial_and_multiple (a : ℤ) (h : ∃ k : ℤ, a = 456 * k) :
  Int.gcd (3 * a^3 + a^2 + 4 * a + 57) a = 57 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_polynomial_and_multiple_l1194_119430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equality_l1194_119482

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_equality (t : Triangle) 
  (h : t.a^2 - t.b^2 = (1/2) * t.c^2) : 
  (2 * t.a * Real.cos t.B) / t.c = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equality_l1194_119482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_y_l1194_119478

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 + Real.log x / Real.log 3

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := (f x)^2 + f (x^2)

-- State the theorem
theorem range_of_y :
  ∀ a, a ∈ Set.Icc 6 13 ↔ ∃ x ∈ Set.Icc 1 9, y x = a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_y_l1194_119478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_statements_true_l1194_119410

-- Define the four statements
def statement1 : Prop := ∀ (x : ℝ), x^2 ≥ 0 ↔ ¬∃ (x₀ : ℝ), x₀^2 ≤ 0

def statement2 : Prop := ∀ (m : ℝ), m > (1/2) → ¬∃ (x : ℝ), m*x^2 + 2*x + 2 = 0

def statement3 : Prop := ∀ (x : ℝ), (x ≠ 3 → abs x ≠ 3) ∧ ¬(abs x ≠ 3 → x ≠ 3)

noncomputable def statement4 : Prop := ∀ (A B : ℝ), 0 < A ∧ A < Real.pi/2 ∧ 0 < B ∧ B < Real.pi/2 ∧ Real.pi/2 < A + B ∧ A + B < Real.pi →
  Real.cos B < Real.sin A ∧ Real.sin A < Real.tan A

-- Theorem stating that exactly two of the statements are true
theorem exactly_two_statements_true : 
  (statement1 = false ∧ statement2 = true ∧ statement3 = false ∧ statement4 = true) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_statements_true_l1194_119410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kathleen_savings_l1194_119491

noncomputable def savings_june : ℚ := 21
noncomputable def savings_july : ℚ := 46
noncomputable def savings_august : ℚ := 45
noncomputable def savings_september : ℚ := 32
noncomputable def savings_october : ℚ := savings_august / 2

noncomputable def expense_school : ℚ := 12
noncomputable def expense_clothes : ℚ := 54
noncomputable def expense_gift : ℚ := 37

noncomputable def charity_donation : ℚ := 10
noncomputable def aunt_gift : ℚ := 25
noncomputable def savings_threshold : ℚ := 125

theorem kathleen_savings (
  total_savings : ℚ := savings_june + savings_july + savings_august + savings_september + savings_october
  ) (
  total_expenses : ℚ := expense_school + expense_clothes + expense_gift
  ) (
  h_savings : total_savings > savings_threshold
  ) : 
  total_savings - total_expenses - charity_donation + aunt_gift = 78.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kathleen_savings_l1194_119491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crossSectionArea_theorem_l1194_119443

/-- Regular quadrilateral pyramid with base side length a -/
structure RegularQuadPyramid where
  a : ℝ
  a_pos : 0 < a

/-- The area of a cross-section passing through the apex of a regular quadrilateral pyramid,
    perpendicular to the opposite edge, given that a lateral edge forms an angle of 30° with the height -/
noncomputable def crossSectionArea (p : RegularQuadPyramid) : ℝ :=
  (p.a^2 * Real.sqrt 3) / 3

/-- Theorem stating the area of the cross-section -/
theorem crossSectionArea_theorem (p : RegularQuadPyramid) :
  let h := p.a / Real.sqrt 3  -- Height of the pyramid
  let l := 2 * h / Real.sqrt 3  -- Lateral edge length
  (∃ θ : ℝ, θ = 30 * π / 180 ∧ Real.cos θ = h / l) →  -- 30° angle condition
  crossSectionArea p = (p.a^2 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crossSectionArea_theorem_l1194_119443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_cubic_equation_l1194_119407

theorem root_of_cubic_equation :
  let x : ℝ := (4 + Real.sqrt 80)^(1/3) - (Real.sqrt 80 - 4)^(1/3)
  x^3 + 12*x - 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_cubic_equation_l1194_119407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_negative_fifteen_l1194_119406

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x - 7

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 3 * ((f⁻¹ x) ^ 2) - 2 * (f⁻¹ x) + 1

-- State the theorem
theorem g_of_negative_fifteen : g (-15) = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_negative_fifteen_l1194_119406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_present_value_calculation_l1194_119432

/-- Calculates the present value given future value, interest rate, and time period -/
noncomputable def presentValue (futureValue : ℝ) (interestRate : ℝ) (years : ℕ) : ℝ :=
  futureValue / (1 + interestRate) ^ years

/-- The future value after 15 years -/
def futureValue : ℝ := 600000

/-- The annual interest rate -/
def interestRate : ℝ := 0.04

/-- The time period in years -/
def timePeriod : ℕ := 15

/-- Theorem stating that the present value is approximately $333,087.66 -/
theorem present_value_calculation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |presentValue futureValue interestRate timePeriod - 333087.66| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_present_value_calculation_l1194_119432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1194_119487

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - 1 else Real.exp (x * Real.log 2)

-- Define the set of a that satisfy the condition
def S : Set ℝ := {a | f (f a) = Real.exp ((f a) * Real.log 2)}

-- Theorem statement
theorem range_of_a : S = Set.Ici (2/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1194_119487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_plus_b_eq_neg_two_l1194_119412

-- Define the slope angle of line l
def slope_angle_l : ℝ := 135

-- Define points A and B
def point_A : ℝ × ℝ := (3, 2)
def point_B (a : ℝ) : ℝ × ℝ := (a, -1)

-- Define line l₁ passing through A and B
def line_l₁ (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • point_A + t • (point_B a)}

-- Define line l₂
def line_l₂ (b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 + b * p.2 + 1 = 0}

-- State the theorem
theorem a_plus_b_eq_neg_two (a b : ℝ) : 
  (∃ l : Set (ℝ × ℝ), 
    (∀ p ∈ l, (p.2 - 0) / (p.1 - 0) = Real.tan (slope_angle_l * π / 180)) ∧
    (∀ p q : ℝ × ℝ, p ∈ line_l₁ a → q ∈ line_l₁ a → (p.2 - q.2) * (p.1 - q.1) = -(q.1 - p.1) * (q.2 - p.2)) ∧
    (∀ p q r s : ℝ × ℝ, p ∈ line_l₁ a → q ∈ line_l₁ a → r ∈ line_l₂ b → s ∈ line_l₂ b → 
      (p.2 - q.2) / (p.1 - q.1) = (r.2 - s.2) / (r.1 - s.1))) →
  a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_plus_b_eq_neg_two_l1194_119412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_at_most_one_zero_odd_symmetry_l1194_119477

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define the property of a function being decreasing
def Decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- Define the property of a function being odd
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem 1: If f is decreasing, then f(x) = 0 has at most one solution
theorem decreasing_at_most_one_zero (h : Decreasing f) :
  (∃! x, f x = 0) ∨ (∀ x, f x ≠ 0) :=
sorry

-- Theorem 2: If f is odd and f(a) = 1 for some a, then f(b) = -1 for some b
theorem odd_symmetry (h : IsOdd f) (a : ℝ) (ha : f a = 1) :
  ∃ b, f b = -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_at_most_one_zero_odd_symmetry_l1194_119477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_to_geometric_sequence_l1194_119498

theorem arithmetic_to_geometric_sequence (a b c : ℝ) : 
  (b - a = c - b) ∧  -- arithmetic sequence
  (a : ℝ) / 3 = b / 4 ∧ b / 4 = c / 5 ∧  -- ratio 3:4:5
  ((a + 1) * c = b * b) →  -- geometric sequence after increasing smallest number
  a = 15 ∧ b = 20 ∧ c = 25 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_to_geometric_sequence_l1194_119498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vremyankin_arrives_first_l1194_119438

/-- Represents the total distance between Morning Town and Evening Town -/
def D : ℝ := sorry

/-- Represents Vremyankin's total travel time -/
def T : ℝ := sorry

/-- Vremyankin's distance covered in the first half of time -/
noncomputable def vremyankin_first_half : ℝ := 5 * (T / 2)

/-- Vremyankin's distance covered in the second half of time -/
noncomputable def vremyankin_second_half : ℝ := 4 * (T / 2)

/-- Puteykin's time for the first half of distance -/
noncomputable def puteykin_first_half : ℝ := D / (2 * 4)

/-- Puteykin's time for the second half of distance -/
noncomputable def puteykin_second_half : ℝ := D / (2 * 5)

/-- The theorem stating that Vremyankin arrives first -/
theorem vremyankin_arrives_first (hD : D > 0) (hT : T > 0) :
  vremyankin_first_half + vremyankin_second_half = D →
  T < puteykin_first_half + puteykin_second_half :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vremyankin_arrives_first_l1194_119438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_term_prime_l1194_119481

def sequenceItem (n : ℕ) : ℕ :=
  if n = 1 then 47
  else 47 * (List.range n).foldl (fun acc i => acc + 10^(2*i)) 1

theorem only_first_term_prime :
  ∀ n : ℕ, n > 1 → ¬(Nat.Prime (sequenceItem n)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_term_prime_l1194_119481
