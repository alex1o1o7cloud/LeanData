import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinates_in_new_basis_l469_46955

open LinearAlgebra

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the original basis vectors
variable (a b c : V)

-- Assume the original basis is linearly independent
variable (h : LinearIndependent ℝ ![a, b, c])

-- Define vector p in terms of the original basis
def p (a b c : V) : V := 2 • a + 1 • b + (-1) • c

-- Define the new basis vectors
def new_basis_1 (a b : V) : V := a + b
def new_basis_2 (a b : V) : V := a - b
def new_basis_3 (c : V) : V := c

-- State the theorem
theorem coordinates_in_new_basis (a b c : V) :
  ∃ (x y z : ℝ), p a b c = x • new_basis_1 a b + y • new_basis_2 a b + z • new_basis_3 c ∧
                 x = 3/2 ∧ y = 1/2 ∧ z = -1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinates_in_new_basis_l469_46955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_ratio_l469_46932

/-- 
Given an isosceles triangle with a vertex angle of 36°, 
the ratio of the base to the leg is (√5 - 1) / 2.
-/
theorem isosceles_triangle_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  36 = 36 → a / b = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_ratio_l469_46932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_wheel_rpm_l469_46906

/-- Calculates the revolutions per minute of a bus wheel -/
noncomputable def wheel_rpm (radius : ℝ) (speed : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let speed_cm_per_minute := speed * 100000 / 60
  speed_cm_per_minute / circumference

/-- Theorem stating that a bus wheel with radius 175 cm on a bus traveling at 66 km/h 
    rotates approximately 1000 times per minute -/
theorem bus_wheel_rpm :
  let radius := 175
  let speed := 66
  Int.floor (wheel_rpm radius speed) = 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_wheel_rpm_l469_46906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_large_triangle_l469_46918

noncomputable def triangle_area (n : ℕ+) : ℝ :=
  let v1 := (n : ℂ) + Complex.I
  let v2 := v1 ^ 2
  let v3 := v1 ^ 4
  (1/2) * Complex.abs (v1.re * v2.im + v2.re * v3.im + v3.re * v1.im - 
                       v1.im * v2.re - v2.im * v3.re - v3.im * v1.re)

theorem smallest_n_for_large_triangle : 
  (∀ k : ℕ+, k < 9 → triangle_area k ≤ 5000) ∧ triangle_area 9 > 5000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_large_triangle_l469_46918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_intersection_perimeter_l469_46959

/-- Given a square with side length 2a and a line y = x/3 intersecting it,
    the perimeter of one resulting quadrilateral divided by a
    is equal to (14 + 2√10) / 3 -/
theorem square_intersection_perimeter (a : ℝ) (a_pos : a > 0) :
  let square := {p : ℝ × ℝ | max (|p.1|) (|p.2|) = a}
  let line := {p : ℝ × ℝ | p.2 = p.1 / 3}
  let intersection := square ∩ line
  let quadrilateral := {p ∈ square | p.1 ≥ 0 ∧ p.2 ≥ p.1 / 3}
  let perimeter := Real.sqrt ((2 * a)^2 + (2 * a / 3)^2) +
                    (a - a / 3) + 2 * a + (a - (-a / 3))
  perimeter / a = (14 + 2 * Real.sqrt 10) / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_intersection_perimeter_l469_46959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_bisects_circle_implies_relation_l469_46984

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y - 4 = 0

-- Define the line
def line_eq (m n x y : ℝ) : Prop := m*x + 2*n*y - 4 = 0

-- State the theorem
theorem line_bisects_circle_implies_relation (m n : ℝ) :
  (∀ x y : ℝ, circle_eq x y → line_eq m n x y → 
    ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ 
      line_eq m n x₁ y₁ ∧ line_eq m n x₂ y₂ ∧
      (x₁ - x)^2 + (y₁ - y)^2 = (x₂ - x)^2 + (y₂ - y)^2) →
  m - n - 2 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_bisects_circle_implies_relation_l469_46984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_prime_probability_l469_46939

def spinner_numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 9]

def is_prime (n : ℕ) : Bool := Nat.Prime n

def count_primes (numbers : List ℕ) : ℕ :=
  numbers.filter is_prime |>.length

theorem spinner_prime_probability :
  let total_sectors := spinner_numbers.length
  let prime_sectors := count_primes spinner_numbers
  (prime_sectors : ℚ) / total_sectors = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_prime_probability_l469_46939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_fourth_proposition_true_l469_46970

open Real

-- Define the propositions
def proposition1 (k : ℝ) : Prop :=
  (k = 1) ↔ (∀ x, cos (k*x)^2 - sin (k*x)^2 = cos (k*(x+π))^2 - sin (k*(x+π))^2) ∧
            (∀ p, p > 0 → (∀ x, cos (k*x)^2 - sin (k*x)^2 = cos (k*(x+p))^2 - sin (k*(x+p))^2) → p ≥ π)

def proposition2 : Prop :=
  ∀ x, sin (2*x - π/6) = cos (2*(x - π/6))

def proposition3 : Prop :=
  (∀ a : ℝ, (∀ x : ℝ, a*x^2 - 2*a*x + 1 > 0) → (0 < a ∧ a < 1))

def proposition4 : Prop :=
  ∀ A B C O : ℝ × ℝ,
    (O.1 > min A.1 (min B.1 C.1) ∧ O.1 < max A.1 (max B.1 C.1) ∧
     O.2 > min A.2 (min B.2 C.2) ∧ O.2 < max A.2 (max B.2 C.2)) →
    ((O.1 - A.1, O.2 - A.2) + (O.1 - C.1, O.2 - C.2) = (-2*(O.1 - B.1), -2*(O.2 - B.2))) →
    2 * abs ((O.1 - A.1)*(O.2 - B.2) - (O.2 - A.2)*(O.1 - B.1)) =
    abs ((O.1 - A.1)*(O.2 - C.2) - (O.2 - A.2)*(O.1 - C.1))

theorem only_fourth_proposition_true : 
  (¬ ∃ k, proposition1 k) ∧ 
  ¬ proposition2 ∧ 
  ¬ proposition3 ∧ 
  proposition4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_fourth_proposition_true_l469_46970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_in_triangle_l469_46956

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

-- Define the vector from one point to another
def vector_between (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  (p2.1 - p1.1, p2.2 - p1.2)

-- Define the angle between three points
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  sorry -- This would be defined properly in a full implementation

-- Theorem statement
theorem min_dot_product_in_triangle (t : Triangle) :
  t.C.1 - t.B.1 = 4 ∧ 
  Real.cos (angle t.B t.A t.C) = -1/2 →
  ∃ (min_value : ℝ), 
    min_value = -8/3 ∧
    ∀ (t' : Triangle), 
      t'.C.1 - t'.B.1 = 4 ∧ 
      Real.cos (angle t'.B t'.A t'.C) = -1/2 →
      dot_product (vector_between t'.A t'.B) (vector_between t'.A t'.C) ≥ min_value :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_in_triangle_l469_46956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_shift_equals_g_l469_46979

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 3)
noncomputable def g (x : ℝ) := Real.cos (2 * x)

theorem f_shift_equals_g : ∀ x : ℝ, f (x + Real.pi / 12) = g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_shift_equals_g_l469_46979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_negative_money_l469_46952

/-- Represents the state of the game at any point -/
structure GameState where
  n : Nat
  money : List Int
  deriving Repr

/-- Defines a single step of the game -/
def gameStep (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Runs the game until it's over -/
def runGame (initialState : GameState) : GameState :=
  sorry

/-- Checks if any student has negative money -/
def hasNegativeMoney (state : GameState) : Bool :=
  sorry

/-- The main theorem to prove -/
theorem no_negative_money (n : Nat) (initialMoney : List Int) :
  n ≥ 2 →
  (∀ m ∈ initialMoney, m ≥ 0) →
  initialMoney.length = n →
  initialMoney.sum ≥ n^2 - 3*n + 2 →
  ¬(hasNegativeMoney (runGame ⟨n, initialMoney⟩)) :=
by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_negative_money_l469_46952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_range_of_m_l469_46961

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∃ x, f x < |1 - 2*a|} = Set.Ioi (5/2) ∪ Set.Iio (-3/2) :=
sorry

-- Theorem for the range of m
theorem range_of_m :
  {m : ℝ | ∃ t, t^2 + 2*Real.sqrt 6*t + f m = 0} = Set.Icc (-1) 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_range_of_m_l469_46961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_values_l469_46900

theorem expression_values (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ (x : ℝ), x ∈ ({4, 0, -4} : Set ℝ) ∧
  (a / abs a + b / abs b + c / abs c + (a * b * c) / abs (a * b * c) = x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_values_l469_46900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_tax_rate_is_15_percent_l469_46993

/-- Represents the tax system in Country X -/
structure TaxSystem where
  /-- The tax rate (as a percentage) on the first $40,000 of income -/
  baseTaxRate : ℚ
  /-- The tax rate (as a percentage) on income exceeding $40,000 -/
  excessTaxRate : ℚ
  /-- The income threshold for the base tax rate -/
  baseThreshold : ℚ

/-- Calculates the total tax for a given income and tax system -/
def calculateTax (income : ℚ) (system : TaxSystem) : ℚ :=
  let baseTax := min income system.baseThreshold * (system.baseTaxRate / 100)
  let excessTax := max (income - system.baseThreshold) 0 * (system.excessTaxRate / 100)
  baseTax + excessTax

/-- Theorem: Given the specific conditions, the base tax rate is 15% -/
theorem base_tax_rate_is_15_percent 
  (system : TaxSystem)
  (h1 : system.excessTaxRate = 20)
  (h2 : system.baseThreshold = 40000)
  (h3 : calculateTax 50000 system = 8000) :
  system.baseTaxRate = 15 := by
  sorry

/-- Example calculation -/
def example_calculation : ℚ :=
  calculateTax 50000 { baseTaxRate := 15, excessTaxRate := 20, baseThreshold := 40000 }

#eval example_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_tax_rate_is_15_percent_l469_46993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_gain_approx_l469_46969

-- Define the given values
noncomputable def manufacturing_cost : ℝ := 180
noncomputable def transportation_cost_per_100 : ℝ := 500
noncomputable def selling_price : ℝ := 222

-- Define the function to calculate the percentage gain
noncomputable def percentage_gain : ℝ :=
  let transportation_cost := transportation_cost_per_100 / 100
  let total_cost := manufacturing_cost + transportation_cost
  let gain := selling_price - total_cost
  (gain / selling_price) * 100

-- Theorem statement
theorem percentage_gain_approx :
  abs (percentage_gain - 16.67) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_gain_approx_l469_46969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_shaded_area_l469_46986

/-- Given a rectangle ABCD inscribed in a circle where AC is a diameter of length t
    and CD = 2AD, the area of the region between the circle and the rectangle
    is (t²/4)π - 2t²/5 -/
theorem inscribed_rectangle_shaded_area (t : ℝ) (h : t > 0) :
  let r := t / 2
  let circle_area := π * r^2
  let ad := r / Real.sqrt 5
  let cd := 2 * ad
  let rectangle_area := ad * cd
  circle_area - rectangle_area = (t^2 / 4) * π - 2 * t^2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_shaded_area_l469_46986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_D3_type_iff_g_is_D2_type_l469_46980

-- Define D(k) type function
def is_D_k_type (f : ℝ → ℝ) (D : Set ℝ) (k : ℕ) : Prop :=
  ∀ x ∈ D, (1 : ℝ) / k < f x ∧ f x < k

-- Define the domains
def D1 : Set ℝ := Set.Icc (-3) (-1) ∪ Set.Icc 1 3
def D2 : Set ℝ := Set.Ioo 0 2

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := a * abs x
noncomputable def g (x : ℝ) : ℝ := Real.exp x - x^2 - x

-- Theorem 1
theorem f_is_D3_type_iff (a : ℝ) :
  is_D_k_type (f a) D1 3 ↔ a ∈ Set.Ioo (1/3) 1 := by
  sorry

-- Theorem 2
theorem g_is_D2_type :
  is_D_k_type g D2 2 := by
  sorry

-- Given fact about e^2
axiom e_squared_bounds : 7 < Real.exp 2 ∧ Real.exp 2 < 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_D3_type_iff_g_is_D2_type_l469_46980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_relation_l469_46943

-- Define the cylinders
structure Cylinder where
  radius : ℝ
  height : ℝ

-- Define the volume of a cylinder
noncomputable def volume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

-- State the theorem
theorem cylinder_volume_relation (h : ℝ) (h_pos : h > 0) :
  let A : Cylinder := { radius := h/3, height := h }
  let B : Cylinder := { radius := h, height := h }
  volume B = 3 * volume A →
  volume B = (1/3) * Real.pi * h^3 := by
  intro A B hyp
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_relation_l469_46943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_pairs_in_cube_l469_46933

/-- A cube is a three-dimensional shape with six square faces -/
structure Cube where
  -- We don't need to define the specifics of a cube for this problem

/-- A line in a cube determined by two vertices -/
structure Line (c : Cube) where
  -- We don't need to define the specifics of a line for this problem

/-- A plane in a cube containing four vertices -/
structure Plane (c : Cube) where
  -- We don't need to define the specifics of a plane for this problem

/-- A perpendicular line-plane pair in a cube -/
structure PerpendicularPair (c : Cube) where
  line : Line c
  plane : Plane c
  is_perpendicular : Bool -- We assume there's a way to determine if a line is perpendicular to a plane

/-- The number of perpendicular line-plane pairs in a cube -/
def num_perpendicular_pairs (c : Cube) : Nat :=
  -- This function would count the number of perpendicular pairs
  sorry

theorem perpendicular_pairs_in_cube (c : Cube) :
  num_perpendicular_pairs c = 36 := by
  sorry

#check perpendicular_pairs_in_cube

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_pairs_in_cube_l469_46933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_36m_minus_5n_l469_46946

theorem min_value_36m_minus_5n :
  (∃ (m₀ n₀ : ℕ), (36^m₀ : ℤ) - (5^n₀ : ℤ) = 11) ∧
  (∀ (m n : ℕ), |(36^m : ℤ) - (5^n : ℤ)| ≥ 11) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_36m_minus_5n_l469_46946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_f_leq_neg_two_g_four_zeros_l469_46913

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 2 then (x - 2) * (a - x)
  else if 0 ≤ x ∧ x < 2 then x * (2 - x)
  else (x + 2) * (a - x)

noncomputable def g (a m : ℝ) (x : ℝ) : ℝ := f a x - m

theorem f_even (a : ℝ) : ∀ x, f a x = f a (-x) := by sorry

theorem f_leq_neg_two (a : ℝ) : ∀ x ≤ -2, f a x = (x + 2) * (a - x) := by sorry

theorem g_four_zeros (a m : ℝ) :
  (∃ x₁ x₂ x₃ x₄, x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧
    g a m x₁ = 0 ∧ g a m x₂ = 0 ∧ g a m x₃ = 0 ∧ g a m x₄ = 0 ∧
    x₂ - x₁ = x₃ - x₂ ∧ x₃ - x₂ = x₄ - x₃) ↔
  ((a < Real.sqrt 3 + 2 ∧ m = 3/4) ∨
   (a = 4 ∧ m = 1) ∨
   (a > (10 + 4 * Real.sqrt 7)/3 ∧ m = -(3*a^2 - 20*a + 12)/16)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_f_leq_neg_two_g_four_zeros_l469_46913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_sum_l469_46927

/-- The sum of an infinite geometric series with first term a and common ratio r -/
noncomputable def geometricSeriesSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- The first term of the geometric series -/
def a : ℝ := 6

/-- The common ratio of the geometric series -/
def r : ℝ := -0.5

theorem infinite_geometric_series_sum :
  geometricSeriesSum a r = 4 := by
  -- Unfold the definition of geometricSeriesSum
  unfold geometricSeriesSum
  -- Unfold the definitions of a and r
  unfold a r
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_sum_l469_46927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_equidistant_point_l469_46991

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the moving point type
structure MovingPoint where
  position : ℝ → ℝ × ℝ
  speed : ℝ

-- Define the problem setup
def IntersectingCircles (c1 c2 : Circle) : Prop :=
  ∃ A : ℝ × ℝ, A ∈ {x | ‖x - c1.center‖ = c1.radius} ∩ {x | ‖x - c2.center‖ = c2.radius}

def SimultaneousRevolution (p1 p2 : MovingPoint) (A : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 0 ∧ p1.position 0 = A ∧ p2.position 0 = A ∧ 
           p1.position t = A ∧ p2.position t = A

-- Main theorem
theorem fixed_equidistant_point 
  (c1 c2 : Circle) 
  (p1 p2 : MovingPoint) 
  (A : ℝ × ℝ) 
  (h1 : IntersectingCircles c1 c2) 
  (h2 : SimultaneousRevolution p1 p2 A) : 
  ∃ P : ℝ × ℝ, ∀ t : ℝ, ‖P - p1.position t‖ = ‖P - p2.position t‖ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_equidistant_point_l469_46991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l469_46975

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the perimeter of a quadrilateral given its four vertices -/
noncomputable def perimeter (a b c d : Point) : ℝ :=
  distance a b + distance b c + distance c d + distance d a

/-- Theorem: The perimeter of the given trapezoid ABCD is 40 -/
theorem trapezoid_perimeter :
  let a : Point := ⟨0, 0⟩
  let b : Point := ⟨3, 4⟩
  let c : Point := ⟨15, 4⟩
  let d : Point := ⟨18, 0⟩
  perimeter a b c d = 40 := by
  sorry

#eval println! "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l469_46975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_proof_l469_46930

def A : ℝ × ℝ := (-5, -4)
def C : ℝ × ℝ := (-3, 0)

def divides_segment (A B C : ℝ × ℝ) (ratio : ℚ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ C = (1 - t) • A + t • B ∧ ratio = (t / (1 - t))

theorem segment_length_proof (B : ℝ × ℝ) 
  (h : divides_segment A B C (2/3)) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 5 * Real.sqrt 5 := by
  sorry

#check segment_length_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_proof_l469_46930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l469_46910

noncomputable def f (x : ℝ) : ℝ := 2 * x * Real.tan x

theorem tangent_line_equation (x y : ℝ) :
  f (π / 4) = y ∧ 
  (deriv f) (π / 4) * (x - π / 4) = y - f (π / 4) →
  (2 + π) * x - y - π^2 / 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l469_46910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_equal_sines_different_sides_l469_46902

/-- A convex polygon with n sides. -/
structure ConvexPolygon (n : ℕ) where
  -- Vertices of the polygon
  vertices : Fin n → ℝ × ℝ
  -- Convexity condition (placeholder)
  convex : True

/-- Helper function to calculate the angle at a vertex of a polygon. -/
noncomputable def angle (p : ConvexPolygon n) (i : Fin n) : ℝ :=
  sorry

/-- Helper function to calculate the length of a side of a polygon. -/
noncomputable def sideLength (p : ConvexPolygon n) (i : Fin n) : ℝ :=
  sorry

/-- Predicate to check if all sines of angles in a polygon are equal. -/
def AllSinesEqual (p : ConvexPolygon n) : Prop :=
  ∀ i j : Fin n, Real.sin (angle p i) = Real.sin (angle p j)

/-- Predicate to check if all side lengths in a polygon are different. -/
def AllSidesDifferent (p : ConvexPolygon n) : Prop :=
  ∀ i j : Fin n, i ≠ j → sideLength p i ≠ sideLength p j

/-- The smallest n for which a convex n-gon with equal sines of angles and different side lengths exists is 5. -/
theorem smallest_n_equal_sines_different_sides :
  (∃ (p : ConvexPolygon 5), AllSinesEqual p ∧ AllSidesDifferent p) ∧
  (∀ n < 5, ¬∃ (p : ConvexPolygon n), AllSinesEqual p ∧ AllSidesDifferent p) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_equal_sines_different_sides_l469_46902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_is_42_l469_46974

/-- Represents a cricket batsman's performance -/
structure Batsman where
  initial_innings : ℕ
  initial_average : ℚ
  new_innings_score : ℚ
  average_increase : ℚ

/-- Calculates the new average after an additional innings -/
def new_average (b : Batsman) : ℚ :=
  (b.initial_innings * b.initial_average + b.new_innings_score) / (b.initial_innings + 1)

/-- Theorem: Given the conditions, prove that the new average is 42 -/
theorem new_average_is_42 (b : Batsman) 
    (h1 : b.initial_innings = 8)
    (h2 : b.new_innings_score = 90)
    (h3 : b.average_increase = 6)
    (h4 : new_average b = b.initial_average + b.average_increase) :
  new_average b = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_is_42_l469_46974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l469_46934

-- Define the function (marked as noncomputable due to dependency on Real)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 * x - 3) + 1 / (x - 3)

-- Define the domain
def domain : Set ℝ := {x | x ≥ 3/2 ∧ x ≠ 3}

-- Theorem statement
theorem f_domain : 
  {x : ℝ | ∃ y, f x = y} = domain := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l469_46934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_property_l469_46965

/-- Given a function g(x) = (2ax - b) / (bx + 2a) where ab ≠ 0,
    if g(g(x)) = x for all x in the domain of g, then 2a - b = 0 -/
theorem inverse_function_property (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  let g := fun (x : ℝ) => (2 * a * x - b) / (b * x + 2 * a)
  (∀ x, x ∈ Set.univ → g (g x) = x) →
  2 * a - b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_property_l469_46965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_some_expression_proof_l469_46936

theorem some_expression_proof (x y : ℝ) (h : x * y = 1) :
  (4 : ℝ) ^ ((x + y) ^ 2) / (4 : ℝ) ^ ((x + y - 2) ^ 2) = 256 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_some_expression_proof_l469_46936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thabo_books_l469_46937

/-- The number of books Thabo owns -/
def total_books : ℕ := 200

/-- The number of hardcover nonfiction books Thabo owns -/
def hardcover_nonfiction : ℕ := 35

/-- The number of paperback nonfiction books Thabo owns -/
def paperback_nonfiction : ℕ := hardcover_nonfiction + 20

/-- The number of paperback fiction books Thabo owns -/
def paperback_fiction : ℕ := 2 * paperback_nonfiction

theorem thabo_books : 
  hardcover_nonfiction + paperback_nonfiction + paperback_fiction = total_books := by
  sorry

#eval hardcover_nonfiction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thabo_books_l469_46937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_true_l469_46908

-- Define the statements
def statement1 : Prop := ∀ x : ℝ, (x ≠ 1 → x^2 - 3*x + 2 ≠ 0) ↔ (x^2 - 3*x + 2 = 0 → x = 1)
def statement2 : Prop := (¬∀ x : ℝ, x^2 + x + 1 ≠ 0) ↔ (∃ x : ℝ, x^2 + x + 1 = 0)
def statement3 : Prop := ∀ p q : Prop, (p ∨ q) → (p ∧ q)
def statement4 : Prop := ∀ x : ℝ, (x > 3 → x^2 - 3*x + 2 > 0) ∧ (∃ y : ℝ, y ≤ 3 ∧ y^2 - 3*y + 2 > 0)
def statement5 : Prop := ∀ A B C : ℝ, 0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧ A + B + C = Real.pi → (A > B → Real.sin A > Real.sin B)

-- Theorem stating that exactly 4 out of 5 statements are true
theorem exactly_four_true : 
  (statement1 ∧ statement2 ∧ ¬statement3 ∧ statement4 ∧ statement5) ∨
  (statement1 ∧ statement2 ∧ statement3 ∧ ¬statement4 ∧ statement5) ∨
  (statement1 ∧ statement2 ∧ statement3 ∧ statement4 ∧ ¬statement5) ∨
  (statement1 ∧ ¬statement2 ∧ statement3 ∧ statement4 ∧ statement5) ∨
  (¬statement1 ∧ statement2 ∧ statement3 ∧ statement4 ∧ statement5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_true_l469_46908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stamp_collection_difference_l469_46940

/-- Represents a stamp collection -/
structure StampCollection where
  total : ℕ
  international : ℕ
  historical : ℕ
  animal : ℕ
  space : ℕ

/-- Calculate the combined difference in historical and animal stamps between two collections -/
def combinedDifference (a b : StampCollection) : ℕ :=
  (a.historical.max b.historical - a.historical.min b.historical) +
  (a.animal.max b.animal - a.animal.min b.animal)

/-- Carl's stamp collection -/
def carl : StampCollection :=
  { total := 200, international := 60, historical := 70, animal := 40, space := 30 }

/-- Kevin's stamp collection -/
def kevin : StampCollection :=
  { total := 150, international := 50, historical := 45, animal := 25, space := 30 }

/-- Susan's stamp collection -/
def susan : StampCollection :=
  { total := 180, international := 55, historical := 60, animal := 35, space := 30 }

theorem stamp_collection_difference :
  combinedDifference carl kevin = 40 ∧ combinedDifference carl susan = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stamp_collection_difference_l469_46940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_m_value_l469_46920

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 4*x - m = 0

-- Define the area of the circle
noncomputable def circle_area (r : ℝ) : ℝ :=
  Real.pi * r^2

-- Theorem statement
theorem circle_m_value :
  ∀ m : ℝ, 
  (∃ r : ℝ, circle_area r = Real.pi) ∧
  (∀ x y : ℝ, circle_equation x y m ↔ (x - 2)^2 + y^2 = 4 + m) →
  m = -3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_m_value_l469_46920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_phi_value_l469_46909

theorem max_phi_value (ω φ : ℝ) (h_ω_pos : ω > 0) (h_φ_range : 0 < φ ∧ φ < π) : 
  let f := λ x => Real.tan (ω * x - φ)
  (∀ x, f (x + π / ω) = f x) →  -- period is π/ω
  (π / ω = π / 3) →             -- minimum period is π/3
  (∀ x, f (x + π/12) = -f (-x - π/12)) →  -- f(x + π/12) is an odd function
  φ ≤ 3 * π / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_phi_value_l469_46909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequality_l469_46921

theorem count_integers_satisfying_inequality : 
  ∃! k : ℕ, k = (Finset.filter 
    (λ n : ℕ ↦ (Finset.prod (Finset.range 49) (λ i ↦ n - (2*i + 2))) < 0) 
    (Finset.range 99)).card ∧ k = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequality_l469_46921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l469_46981

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def sum_arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1 : ℚ) * d) / 2

theorem arithmetic_sequence_max_sum 
  (n : ℕ) 
  (h₁ : (11 : ℚ) / 7 ≤ n) 
  (h₂ : n ≤ (13 : ℚ) / 5) 
  (a₁ : ℚ) 
  (h₃ : a₁ = (Nat.choose (5 * n) (11 - 2 * n) : ℚ) - (Nat.factorial (11 - 3 * n) / (Nat.factorial (2 * n - 2) * Nat.factorial (11 - 3 * n - (2 * n - 2))))) 
  (d : ℚ) 
  (h₄ : d = -4) :
  (∃ k : ℕ, (k = 25 ∨ k = 26) ∧ 
    sum_arithmetic_sequence a₁ d k = 1300 ∧ 
    ∀ m : ℕ, sum_arithmetic_sequence a₁ d m ≤ 1300) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l469_46981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_rectangular_equivalence_l469_46995

/-- The polar coordinate equation of a circle -/
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.cos (θ + Real.pi / 3)

/-- The rectangular coordinate equation of a circle -/
def rectangular_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - x + Real.sqrt 3 * y = 0

/-- Theorem stating the equivalence of polar and rectangular equations -/
theorem polar_rectangular_equivalence :
  ∀ (x y ρ θ : ℝ), 
    x = ρ * Real.cos θ →
    y = ρ * Real.sin θ →
    polar_equation ρ θ ↔ rectangular_equation x y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_rectangular_equivalence_l469_46995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_tile_weight_calculation_l469_46989

/-- Represents a square tile with given side length and weight -/
structure Tile where
  side : ℝ
  weight : ℝ

/-- Calculates the weight of a larger tile given an original tile -/
noncomputable def largerTileWeight (original : Tile) (largerSide : ℝ) : ℝ :=
  (largerSide^2 / original.side^2) * original.weight

theorem larger_tile_weight_calculation (original : Tile) (larger : Tile) :
  original.side = 4 →
  original.weight = 10 →
  larger.side = 6 →
  largerTileWeight original larger.side = 22.5 :=
by
  intros h1 h2 h3
  unfold largerTileWeight
  simp [h1, h2, h3]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_tile_weight_calculation_l469_46989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_isosceles_triangular_prism_l469_46998

/-- A right prism with isosceles triangular bases -/
structure IsoscelesTriangularPrism where
  /-- Base side length of the isosceles triangle -/
  a : ℝ
  /-- Height of the prism -/
  h : ℝ
  /-- Angle between the equal sides of the base triangle -/
  θ : ℝ
  /-- Constraints on the angle -/
  angle_pos : 0 < θ
  angle_lt_pi : θ < π

/-- The sum of the areas of three mutually adjacent faces is 30 -/
noncomputable def adjacent_faces_area_sum (p : IsoscelesTriangularPrism) : ℝ :=
  2 * p.a * p.h + 1/2 * p.a^2 * Real.sin p.θ

/-- The volume of the prism -/
noncomputable def volume (p : IsoscelesTriangularPrism) : ℝ :=
  1/2 * p.a^2 * p.h * Real.sin p.θ

/-- The theorem stating the maximum volume of the prism -/
theorem max_volume_isosceles_triangular_prism :
  ∃ (p : IsoscelesTriangularPrism),
    adjacent_faces_area_sum p = 30 ∧
    ∀ (q : IsoscelesTriangularPrism),
      adjacent_faces_area_sum q = 30 →
      volume q ≤ 15 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_isosceles_triangular_prism_l469_46998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_thousandths_place_of_five_thirty_seconds_l469_46958

theorem ten_thousandths_place_of_five_thirty_seconds : 
  ∃ (n m : ℕ), (5 : ℚ) / 32 = n / 10000 + (2 : ℚ) / 10000 + m / 100000 ∧ m < 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_thousandths_place_of_five_thirty_seconds_l469_46958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_typhoon_influence_duration_l469_46996

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Duration of typhoon influence on pier A -/
theorem typhoon_influence_duration
  (pier_A : Point)
  (typhoon_initial : Point)
  (typhoon_speed : ℝ)
  (typhoon_radius : ℝ)
  (h1 : distance pier_A typhoon_initial = 400)
  (h2 : typhoon_initial.x - pier_A.x = 400 * Real.cos (60 * Real.pi / 180))
  (h3 : typhoon_initial.y - pier_A.y = 400 * Real.sin (60 * Real.pi / 180))
  (h4 : typhoon_speed = 40)
  (h5 : typhoon_radius = 350) :
  (2 * Real.sqrt (typhoon_radius^2 - (typhoon_initial.x - pier_A.x)^2)) / typhoon_speed = 2.5 := by
  sorry

#check typhoon_influence_duration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_typhoon_influence_duration_l469_46996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_problem_l469_46999

/-- Simple interest calculation --/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Final amount calculation --/
noncomputable def final_amount (principal : ℝ) (interest : ℝ) : ℝ :=
  principal + interest

/-- Initial amount problem --/
theorem initial_amount_problem (final_amount_value : ℝ) (rate : ℝ) (time : ℝ) :
  final_amount_value = 950 →
  rate = 9.230769230769232 →
  time = 5 →
  ∃ (initial_amount : ℝ),
    (final_amount initial_amount (simple_interest initial_amount rate time) = final_amount_value) ∧
    (abs (initial_amount - 650) < 0.01) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_problem_l469_46999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sqrt_2_no_solution_1_solution_2_l469_46912

-- Define the function for the left side of the equation
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + Real.sqrt (2*x - 1)) + Real.sqrt (x - Real.sqrt (2*x - 1))

-- Theorem for case A = √2
theorem solution_sqrt_2 : 
  ∀ x : ℝ, x ∈ Set.Icc (1/2) 1 → f x = Real.sqrt 2 :=
by sorry

-- Theorem for case A = 1
theorem no_solution_1 :
  ¬ ∃ x : ℝ, f x = 1 :=
by sorry

-- Theorem for case A = 2
theorem solution_2 :
  f (3/2) = 2 ∧ ∀ x : ℝ, f x = 2 → x = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sqrt_2_no_solution_1_solution_2_l469_46912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l469_46905

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * 2^x - 4/3 * a) / Real.log 4

noncomputable def g (x : ℝ) : ℝ := Real.log (4^x + 1) / Real.log 4 - 1/2 * x

theorem unique_intersection (a : ℝ) : 
  (a ≠ 0) → 
  (∃! x : ℝ, f a x = g x) ↔ (a > 1 ∨ a = -3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l469_46905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_complex_l469_46903

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def complex_equation (a b : ℝ) : Prop :=
  (3 + b • i) / (1 - i) = (a : ℂ) + b • i

-- State the theorem
theorem magnitude_of_complex (a b : ℝ) :
  complex_equation a b → Complex.abs ((a : ℂ) + b • i) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_complex_l469_46903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l469_46977

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/2) * x - Real.sin x

-- State the theorem
theorem range_of_f :
  ∃ (a b : ℝ), a = Real.pi/6 - Real.sqrt 3/2 ∧ b = Real.pi/2 ∧
  (∀ y : ℝ, (∃ x : ℝ, x ∈ Set.Icc 0 Real.pi ∧ f x = y) ↔ y ∈ Set.Icc a b) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l469_46977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_in_triangle_l469_46929

noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

theorem max_sin_in_triangle (D E F : ℝ × ℝ) : 
  let DE := Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2)
  let EF := Real.sqrt ((E.1 - F.1)^2 + (E.2 - F.2)^2)
  DE = 25 → EF = 20 → Real.sin (angle D E F) ≤ 4/5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_in_triangle_l469_46929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_chords_equal_arcs_l469_46922

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

-- Define a chord
structure Chord (c : Circle) where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ
  on_circle : (endpoint1.1 - c.center.1)^2 + (endpoint1.2 - c.center.2)^2 = c.radius^2 ∧
              (endpoint2.1 - c.center.1)^2 + (endpoint2.2 - c.center.2)^2 = c.radius^2

-- Define an arc
structure Arc (c : Circle) where
  start_point : ℝ × ℝ
  end_point : ℝ × ℝ
  on_circle : (start_point.1 - c.center.1)^2 + (start_point.2 - c.center.2)^2 = c.radius^2 ∧
              (end_point.1 - c.center.1)^2 + (end_point.2 - c.center.2)^2 = c.radius^2

-- Define chord length
noncomputable def chord_length (c : Circle) (ch : Chord c) : ℝ :=
  Real.sqrt ((ch.endpoint1.1 - ch.endpoint2.1)^2 + (ch.endpoint1.2 - ch.endpoint2.2)^2)

-- Define arc length
noncomputable def arc_length (c : Circle) (a : Arc c) : ℝ :=
  c.radius * (Real.arccos ((a.start_point.1 - c.center.1) * (a.end_point.1 - c.center.1) +
                           (a.start_point.2 - c.center.2) * (a.end_point.2 - c.center.2)) / c.radius^2)

-- Theorem: Chords that subtend equal arcs are equal
theorem equal_chords_equal_arcs (c : Circle) (ch1 ch2 : Chord c) (a1 a2 : Arc c) 
  (h1 : arc_length c a1 = arc_length c a2)
  (h2 : ch1.endpoint1 = a1.start_point ∧ ch1.endpoint2 = a1.end_point)
  (h3 : ch2.endpoint1 = a2.start_point ∧ ch2.endpoint2 = a2.end_point) :
  chord_length c ch1 = chord_length c ch2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_chords_equal_arcs_l469_46922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l469_46938

theorem hyperbola_eccentricity (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (Real.tan (π / 6) = b / a) → 
  Real.sqrt (1 + (b / a)^2) = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l469_46938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_divisibility_l469_46966

theorem sequence_divisibility (a : Fin 3031 → ℕ+) 
  (h : ∀ n : Fin 3029, 2 * a (n + 2) = a (n + 1) + 4 * a n) :
  ∃ i : Fin 3031, (2^2020 : ℕ) ∣ (a i : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_divisibility_l469_46966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_b_speed_l469_46911

/-- The speed of Train B given the conditions of the problem -/
noncomputable def speed_of_train_b (speed_a : ℝ) (delay : ℝ) (overtake_distance : ℝ) : ℝ :=
  (overtake_distance + speed_a * delay) / (overtake_distance / speed_a)

/-- Theorem stating the speed of Train B under the given conditions -/
theorem train_b_speed : 
  let speed_a : ℝ := 30
  let delay : ℝ := 2
  let overtake_distance : ℝ := 285
  speed_of_train_b speed_a delay overtake_distance = 345 / (285/30) := by
  sorry

-- Remove the #eval statement as it's not compatible with noncomputable definitions
-- #eval speed_of_train_b 30 2 285

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_b_speed_l469_46911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_initial_investment_l469_46941

/-- Represents the initial investment of A in rupees -/
def A_investment : ℚ := sorry

/-- Represents B's investment in rupees -/
def B_investment : ℚ := 9000

/-- The number of months A's money was invested -/
def A_months : ℚ := 12

/-- The number of months B's money was invested -/
def B_months : ℚ := 7

/-- The ratio of A's profit share to B's profit share -/
def profit_ratio : ℚ := 2 / 3

theorem A_initial_investment :
  (A_investment * A_months) / (B_investment * B_months) = profit_ratio →
  A_investment = 3500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_initial_investment_l469_46941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l469_46982

/-- The time (in seconds) it takes for a train to pass a bridge -/
noncomputable def train_pass_time (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating that a 700m long train moving at 21 km/h passes a 130m bridge in approximately 142.31 seconds -/
theorem train_bridge_passing_time :
  let result := train_pass_time 700 130 21
  abs (result - 142.31) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passing_time_l469_46982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_coordinates_l469_46917

-- Define the function f(x) = e^x
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the tangent line equation
noncomputable def tangent_line (x : ℝ) : ℝ := Real.exp x

-- Theorem statement
theorem tangent_point_coordinates :
  ∃ (x y : ℝ), x = 1 ∧ y = Real.exp 1 ∧
  f x = y ∧ tangent_line x = y ∧
  (∀ t : ℝ, f t ≤ tangent_line t) := by
  sorry

#check tangent_point_coordinates

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_coordinates_l469_46917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_estate_commission_l469_46987

/-- Calculates the commission percentage given the commission amount and selling price -/
noncomputable def commission_percentage (commission : ℝ) (selling_price : ℝ) : ℝ :=
  (commission / selling_price) * 100

/-- Theorem stating that the commission percentage for the given problem is 6% -/
theorem real_estate_commission : 
  let commission : ℝ := 8880
  let selling_price : ℝ := 148000
  commission_percentage commission selling_price = 6 := by
  -- Unfold the definition of commission_percentage
  unfold commission_percentage
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_estate_commission_l469_46987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stationary_points_of_cubic_l469_46942

/-- The function f(x) = x³ + px² + qx -/
def f (p q x : ℝ) : ℝ := x^3 + p * x^2 + q * x

/-- The derivative of f(x) -/
def f_derivative (p q x : ℝ) : ℝ := 3 * x^2 + 2 * p * x + q

theorem stationary_points_of_cubic (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  ∀ x : ℝ, f_derivative p q x = 0 ↔ 
    x = (-p + Real.sqrt (p^2 - 3*q)) / 3 ∨ 
    x = (-p - Real.sqrt (p^2 - 3*q)) / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stationary_points_of_cubic_l469_46942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_2005_with_equal_digits_l469_46964

-- Helper function (defined before the theorem)
def count_occurrences (d : Fin 10) (n : ℕ) : ℕ :=
  (n.digits 10).count d

theorem multiples_of_2005_with_equal_digits : 
  ∃ (seq : ℕ → ℕ), 
    (∀ n : ℕ, 2005 ∣ seq n) ∧ 
    (∀ n : ℕ, ∀ d₁ d₂ : Fin 10, count_occurrences d₁ (seq n) = count_occurrences d₂ (seq n)) ∧
    (∀ n : ℕ, seq n > 0) ∧
    (∀ n m : ℕ, n ≠ m → seq n ≠ seq m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_2005_with_equal_digits_l469_46964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_g_l469_46988

/-- The function g(x) = x^3 + 3x + 2√x -/
noncomputable def g (x : ℝ) : ℝ := x^3 + 3*x + 2*Real.sqrt x

/-- Theorem: 3g(3) - 2g(9) = -1416 + 6√3 -/
theorem evaluate_g : 3 * g 3 - 2 * g 9 = -1416 + 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_g_l469_46988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_line_properties_l469_46914

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y = 0

/-- The line perpendicular to the given line -/
def perp_line (x y : ℝ) : Prop := x + 2*y = 0

/-- The bisecting line -/
def bisecting_line (x y : ℝ) : Prop := 2*x - y = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (1, 2)

theorem bisecting_line_properties :
  (∀ x y, perp_line x y → bisecting_line x y → x + 2*y = 0) ∧ 
  (bisecting_line circle_center.1 circle_center.2) ∧
  (∀ x y, circle_eq x y ↔ (x - 1)^2 + (y - 2)^2 = 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_line_properties_l469_46914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_prime_and_power_l469_46901

-- Define the distance to the nearest integer function
noncomputable def distToNearestInt (x : ℝ) : ℝ := min (x - ⌊x⌋) (⌈x⌉ - x)

-- State the theorem
theorem existence_of_prime_and_power (a b : ℕ+) :
  ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ Odd p ∧
    distToNearestInt (a.1 / (p : ℝ)^k) + 
    distToNearestInt (b.1 / (p : ℝ)^k) + 
    distToNearestInt ((a.1 + b.1) / (p : ℝ)^k) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_prime_and_power_l469_46901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematics_letter_probability_l469_46997

theorem mathematics_letter_probability : 
  let alphabet_size : ℕ := 26
  let unique_letters : ℕ := 8
  let probability_mathematics_letter : ℚ := unique_letters / alphabet_size
  probability_mathematics_letter = 4 / 13 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematics_letter_probability_l469_46997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_vectors_l469_46968

/-- The angle between two 2D vectors -/
noncomputable def angle_between (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem angle_between_specific_vectors :
  let a : ℝ × ℝ := (Real.sqrt 3 / 2, 1 / 2)
  let b : ℝ × ℝ := (Real.sqrt 3, -1)
  angle_between a b = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_vectors_l469_46968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_in_plane_l469_46915

-- Define the plane α
def plane_α : Set (Fin 3 → ℝ) :=
  {x | (-2 * (x 0) + 3 * (x 1) + 1 * (x 2)) - (-2 * 1 + 3 * 1 + 1 * 2) = 0}

-- Define the points
def P : Fin 3 → ℝ := ![1, 1, 2]
def B : Fin 3 → ℝ := ![0, 0, 3]
def C : Fin 3 → ℝ := ![3, 2, 3]
def D : Fin 3 → ℝ := ![2, 1, 4]

-- Theorem statement
theorem points_in_plane :
  P ∈ plane_α ∧ B ∈ plane_α ∧ C ∈ plane_α ∧ D ∈ plane_α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_in_plane_l469_46915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_C_properties_l469_46935

/-- Definition of the hyperbola C -/
def hyperbola_C (x y : ℝ) : Prop := x^2 - y^2/2 = 1

/-- The length of the real axis of hyperbola C -/
def real_axis_length : ℝ := 2

/-- The eccentricity of hyperbola C -/
noncomputable def eccentricity : ℝ := Real.sqrt 3

/-- Theorem stating the properties of hyperbola C -/
theorem hyperbola_C_properties :
  (∀ x y, hyperbola_C x y → 
    (real_axis_length = 2 ∧ eccentricity = Real.sqrt 3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_C_properties_l469_46935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_sum_of_coefficients_l469_46963

-- Define the original expression
noncomputable def original_expr : ℝ := 3 / (4 * Real.sqrt 7 + 3 * Real.sqrt 3)

-- Define the rationalized expression
noncomputable def rationalized_expr : ℝ := (12 * Real.sqrt 7 - 9 * Real.sqrt 3) / 85

-- Theorem statement
theorem rationalize_denominator :
  original_expr = rationalized_expr ∧
  7 < 3 ∧
  (∀ k : ℕ, k > 1 → ¬(85 % k = 0 ∧ (12 % k = 0 ∨ 9 % k = 0))) :=
by
  sorry

-- Additional theorem to show the sum of A, B, C, D, and E
theorem sum_of_coefficients : 12 + 7 + (-9) + 3 + 85 = 98 :=
by
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_sum_of_coefficients_l469_46963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_increase_from_4_to_5_l469_46953

def question_values : List ℕ := [150, 300, 450, 600, 800, 1500, 3000, 6000, 12000, 24000, 48000, 96000, 192000, 384000, 768000]

def percent_increase (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def smallest_increase (increases : List ℚ) : ℚ :=
  increases.foldl min (increases.head!)

theorem smallest_increase_from_4_to_5 :
  let increases := [
    percent_increase (question_values[3]!) (question_values[4]!),
    percent_increase (question_values[4]!) (question_values[5]!),
    percent_increase (question_values[11]!) (question_values[12]!),
    percent_increase (question_values[13]!) (question_values[14]!),
    percent_increase (question_values[0]!) (question_values[1]!)
  ]
  smallest_increase increases = percent_increase (question_values[3]!) (question_values[4]!) := by
  sorry

#eval smallest_increase [
  percent_increase (question_values[3]!) (question_values[4]!),
  percent_increase (question_values[4]!) (question_values[5]!),
  percent_increase (question_values[11]!) (question_values[12]!),
  percent_increase (question_values[13]!) (question_values[14]!),
  percent_increase (question_values[0]!) (question_values[1]!)
]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_increase_from_4_to_5_l469_46953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_number_theorem_l469_46923

def bracket (m : ℕ) : ℚ :=
  if m % 2 = 1 then 3 * m else (1/2) * m

theorem odd_number_theorem (n : ℕ) (h_odd : n % 2 = 1) :
  bracket n * bracket 10 = 45 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_number_theorem_l469_46923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_successive_price_increase_l469_46957

theorem successive_price_increase (original_price : ℝ) (h : original_price > 0) :
  let increase_factor := 1 + 8 / 100
  let final_price := original_price * increase_factor * increase_factor
  let equivalent_single_increase := (final_price / original_price - 1) * 100
  ∃ ε > 0, |equivalent_single_increase - 16.64| < ε := by
  sorry

#check successive_price_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_successive_price_increase_l469_46957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_to_cos_shift_cos_shift_eq_cos_l469_46976

theorem sin_to_cos_shift (x : ℝ) : 
  Real.sin (2 * x) = Real.cos (2 * (x + π/12 - π/4)) := by sorry

theorem cos_shift_eq_cos (x : ℝ) :
  Real.cos (2 * (x + π/12 - π/4)) = Real.cos (2 * x - π/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_to_cos_shift_cos_shift_eq_cos_l469_46976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l469_46962

/-- The standard equation of a hyperbola passing through (2,1) with foci (-√3,0) and (√3,0) -/
theorem hyperbola_standard_equation :
  ∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔
      (x = 2 ∧ y = 1) ∨
      (∃ t : ℝ, (x + Real.sqrt 3)^2 + y^2 = t^2 ∧ 
                (x - Real.sqrt 3)^2 + y^2 = (t + 2 * Real.sqrt 3)^2)) ∧
    a^2 = 2 ∧ b^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l469_46962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_number_ninety_five_two_one_zero_is_valid_l469_46960

def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 5 ∧
  digits[3]! > digits[4]! ∧
  digits[2]! > digits[3]! + digits[4]! ∧
  digits[1]! > digits[2]! + digits[3]! + digits[4]! ∧
  digits[0]! > digits[1]! + digits[2]! + digits[3]! + digits[4]!

theorem largest_valid_number :
  ∀ n : ℕ, is_valid_number n → n ≤ 95210 := by
  sorry

theorem ninety_five_two_one_zero_is_valid :
  is_valid_number 95210 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_number_ninety_five_two_one_zero_is_valid_l469_46960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_alpha_pi_fourth_l469_46950

noncomputable def curve_C (θ : Real) : Real := 4 * Real.cos θ / (1 - Real.cos θ ^ 2)

noncomputable def line_l (t α : Real) : Real × Real := (2 + t * Real.cos α, 2 + t * Real.sin α)

theorem intersection_implies_alpha_pi_fourth (α : Real) 
  (h1 : 0 ≤ α ∧ α < Real.pi) 
  (h2 : ∃ t1 t2 : Real, t1 ≠ t2 ∧ 
        (line_l t1 α).1 ^ 2 + (line_l t1 α).2 ^ 2 = (curve_C (Real.arctan ((line_l t1 α).2 / (line_l t1 α).1)))^2 ∧
        (line_l t2 α).1 ^ 2 + (line_l t2 α).2 ^ 2 = (curve_C (Real.arctan ((line_l t2 α).2 / (line_l t2 α).1)))^2 ∧
        ((line_l t1 α).1 + (line_l t2 α).1) / 2 = 2 ∧
        ((line_l t1 α).2 + (line_l t2 α).2) / 2 = 2) :
  α = Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_alpha_pi_fourth_l469_46950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_circles_l469_46971

theorem shaded_area_circles (P Q : ℝ × ℝ) : 
  let r₁ : ℝ := 1  -- radius of larger circles
  let d : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)  -- distance between P and Q
  let r₂ : ℝ := d / 2  -- radius of smaller circle
  let area_larger : ℝ := 2 * π * r₁^2  -- area of two larger circles
  let area_smaller : ℝ := π * r₂^2  -- area of smaller circle
  let area_common : ℝ := 2 * ((π / 3) * r₁^2 - (Real.sqrt 3 / 4) * r₁^2)  -- area common to larger circles
  let area_shaded : ℝ := area_larger - area_common - area_smaller
  d = r₁ → area_shaded = (5 / 12) * π - Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_circles_l469_46971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tangent_l469_46948

noncomputable section

theorem parallel_vectors_tangent (θ : Real) 
  (h1 : 0 < θ) (h2 : θ < π/2)
  (a b : Fin 2 → ℝ)
  (ha : a = λ i => if i = 0 then Real.sin (2*θ) else Real.cos θ)
  (hb : b = λ i => if i = 0 then Real.cos θ else 1)
  (hparallel : ∃ (k : ℝ), a = k • b) :
  Real.tan θ = 1/2 := by
  sorry

#check parallel_vectors_tangent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tangent_l469_46948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_inequality_l469_46983

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the absolute value of complex numbers
noncomputable def complex_abs (z : ℂ) : ℝ := Complex.abs z

-- State the theorem
theorem complex_inequality : complex_abs (2 - i) > 2 * (i^4).re := by
  -- Convert the complex expressions to their real representations
  have h1 : complex_abs (2 - i) = Real.sqrt 5 := by
    sorry
  have h2 : (i^4).re = 1 := by
    sorry
  
  -- Rewrite the inequality using the real representations
  rw [h1, h2]
  
  -- Prove the inequality
  norm_num
  
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_inequality_l469_46983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equivalence_l469_46994

noncomputable def angle_in_degrees : ℝ := -1120

noncomputable def angle_in_radians : ℝ := angle_in_degrees * (Real.pi / 180)

def k : ℤ := -4

noncomputable def α : ℝ := 16 * Real.pi / 9

noncomputable def β₁ : ℝ := -2 * Real.pi / 9

noncomputable def β₂ : ℝ := -20 * Real.pi / 9

theorem angle_equivalence :
  (angle_in_radians = 2 * ↑k * Real.pi + α) ∧
  (0 ≤ α) ∧ (α < 2 * Real.pi) ∧
  (β₁ ≥ -4 * Real.pi) ∧ (β₁ ≤ 0) ∧
  (β₂ ≥ -4 * Real.pi) ∧ (β₂ ≤ 0) ∧
  (∃ (m n : ℤ), β₁ = 2 * ↑m * Real.pi + α ∧ β₂ = 2 * ↑n * Real.pi + α) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equivalence_l469_46994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l469_46925

/-- Helper function to calculate the area of a triangle given its side lengths -/
noncomputable def area_triangle (a b c : Real) : Real :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Given a triangle ABC with circumradius R, this theorem proves
    properties about angle C and the maximum area of the triangle. -/
theorem triangle_properties (A B C : Real) (R : Real) : 
  -- Conditions
  (A + B + C = π) →  -- Sum of angles in a triangle
  (2 * Real.sin ((A + B) / 2) ^ 2 - Real.cos (2 * C) = 1) →  -- Given equation
  (R = 2) →  -- Given circumradius
  -- Conclusions
  (C = 2 * π / 3) ∧ 
  (∃ (S : Real), S = Real.sqrt 3 ∧ 
    ∀ (a b c : Real), area_triangle a b c ≤ S) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l469_46925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_additional_rum_l469_46972

def rum_from_sally : ℝ := 10
def max_multiplier : ℝ := 3
def rum_consumed_earlier : ℝ := 12

theorem max_additional_rum : 
  max_multiplier * rum_from_sally - rum_consumed_earlier = 18 := by
  -- Proof goes here
  sorry

#eval max_multiplier * rum_from_sally - rum_consumed_earlier

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_additional_rum_l469_46972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_x_geq_24_l469_46928

/-- The function x(t) representing the coordinate of a moving point -/
noncomputable def x (t a : ℝ) : ℝ := 5 * (t + 1)^2 + a / (t + 1)^5

/-- The theorem stating the minimum value of a such that x(t) ≥ 24 for all t ≥ 0 -/
theorem min_a_for_x_geq_24 :
  ∃ a_min : ℝ, a_min = 2 * Real.sqrt ((24/7)^7) ∧
  (∀ a t : ℝ, t ≥ 0 → (∀ t' ≥ 0, x t' a ≥ 24) → a ≥ a_min) ∧
  (∀ t : ℝ, t ≥ 0 → x t a_min ≥ 24) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_x_geq_24_l469_46928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_is_three_l469_46985

noncomputable def mySequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 3
  | 1 => 5
  | n + 2 => 15 / mySequence (n + 1)

theorem fifteenth_term_is_three :
  mySequence 14 = 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_is_three_l469_46985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_neq_ln_one_third_l469_46944

noncomputable def f (x : ℝ) : ℝ := Real.log (2 + 3*x) - (3/2)*x^2

theorem inequality_holds_iff_a_neq_ln_one_third :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (1/6 : ℝ) (1/3 : ℝ) →
    |a - Real.log x| + Real.log ((deriv f) x + 3*x) > 0) ↔
  a ≠ Real.log (1/3 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_neq_ln_one_third_l469_46944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_values_l469_46916

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.tan (x + Real.pi/4)

-- State the theorem
theorem beta_values (β : ℝ) (h1 : 0 < β) (h2 : β < Real.pi) 
  (h3 : f β = 2 * Real.cos (β - Real.pi/4)) : 
  β = Real.pi/12 ∨ β = 3*Real.pi/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_values_l469_46916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l469_46978

noncomputable def sequence_a (n : ℕ) : ℝ := 3^n

noncomputable def S (n : ℕ) : ℝ := (3/2) * sequence_a n - (1/2) * sequence_a 1

noncomputable def b (n : ℕ) : ℝ := sequence_a (n+1) / (S n * S (n+1))

noncomputable def T (n : ℕ) : ℝ := (2/3) * (1/2 - 1/(3^(n+1) - 1))

theorem sequence_properties :
  (∀ n, S n = (3/2) * sequence_a n - (1/2) * sequence_a 1) ∧
  (sequence_a 1 + sequence_a 3 = 2 * (sequence_a 2 + 6)) →
  (∀ n, sequence_a n = 3^n) ∧
  (∀ n, T n = (2/3) * (1/2 - 1/(3^(n+1) - 1))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l469_46978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cleaning_time_together_l469_46954

/-- Represents the cleaning rates and times for John and Nick -/
structure CleaningData where
  john_full_time : ℚ  -- Time for John to clean the entire house
  john_half_time : ℚ  -- Time for John to clean half the house
  nick_full_time : ℚ  -- Time for Nick to clean the entire house

/-- Calculates the time it takes John and Nick to clean the house together -/
def time_together (data : CleaningData) : ℚ :=
  1 / (1 / data.john_full_time + 1 / data.nick_full_time)

/-- Theorem stating that under given conditions, John and Nick clean the house together in 3.6 hours -/
theorem cleaning_time_together (data : CleaningData) 
  (h1 : data.john_full_time = 6)
  (h2 : data.john_half_time = data.nick_full_time / 3)
  (h3 : data.john_half_time = data.john_full_time / 2) :
  time_together data = 18/5 := by
  sorry

#eval time_together { john_full_time := 6, john_half_time := 3, nick_full_time := 9 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cleaning_time_together_l469_46954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_covering_l469_46931

/-- Represents a position on the chessboard --/
structure Position where
  row : Nat
  col : Nat
deriving Repr

/-- Represents a trimino piece --/
structure Trimino where
  pos1 : Position
  pos2 : Position
  pos3 : Position
deriving Repr

/-- The chessboard --/
def Chessboard := Fin 8 → Fin 8 → Option Trimino

/-- Possible positions for the uncovered square --/
def uncovered_positions : List Position :=
  [⟨1, 1⟩, ⟨1, 4⟩, ⟨4, 1⟩, ⟨4, 4⟩]

/-- Check if a position is valid on the 8x8 chessboard --/
def is_valid_position (p : Position) : Prop :=
  1 ≤ p.row ∧ p.row ≤ 8 ∧ 1 ≤ p.col ∧ p.col ≤ 8

/-- Check if a trimino is valid (covers exactly 3 adjacent squares) --/
def is_valid_trimino (t : Trimino) : Prop :=
  is_valid_position t.pos1 ∧ is_valid_position t.pos2 ∧ is_valid_position t.pos3
  -- Add conditions for adjacency here if needed

/-- The main theorem --/
theorem chessboard_covering
  (board : Chessboard)
  (triminos : List Trimino)
  (h1 : triminos.length = 21)
  (h2 : ∀ t ∈ triminos, is_valid_trimino t)
  (h3 : ∀ p : Position, is_valid_position p → 
        (p ∈ uncovered_positions ∨ ∃ t ∈ triminos, p = t.pos1 ∨ p = t.pos2 ∨ p = t.pos3))
  : ∃ p ∈ uncovered_positions, ∀ t ∈ triminos, p ≠ t.pos1 ∧ p ≠ t.pos2 ∧ p ≠ t.pos3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_covering_l469_46931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_theorem_l469_46945

noncomputable def class_average_score (total_students : ℕ) (num_boys : ℕ) (girls_avg : ℝ) (boys_avg : ℝ) : ℝ :=
  let num_girls : ℕ := total_students - num_boys
  let total_score : ℝ := (girls_avg * (num_girls : ℝ)) + (boys_avg * (num_boys : ℝ))
  total_score / (total_students : ℝ)

theorem class_average_theorem :
  class_average_score 50 20 85 80 = 83 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_theorem_l469_46945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l469_46951

theorem division_problem (n j : ℕ) (h1 : n % j = 28) (h2 : (n : ℝ) / (j : ℝ) = 142.07) : j = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l469_46951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_speed_approx_48_l469_46967

/-- The speed of the faster train given the conditions of the problem -/
noncomputable def faster_train_speed (train_length : ℝ) (crossing_time : ℝ) : ℝ :=
  let relative_speed := 2 * train_length / crossing_time
  let slower_speed := relative_speed / 3
  let faster_speed := 2 * slower_speed
  faster_speed * 3.6

/-- Theorem stating the speed of the faster train under the given conditions -/
theorem faster_train_speed_approx_48 :
  ∃ ε > 0, |faster_train_speed 100 10 - 48| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_speed_approx_48_l469_46967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_48kmh_l469_46907

/-- Calculates the speed of a car in km/h given the tire's rotation rate and circumference. -/
noncomputable def carSpeed (revPerMin : ℝ) (tireCircumference : ℝ) : ℝ :=
  (revPerMin * tireCircumference * 60) / 1000

/-- Proves that a car with a tire rotating at 400 revolutions per minute and
    a circumference of 2 meters travels at 48 km/h. -/
theorem car_speed_48kmh (revPerMin : ℝ) (tireCircumference : ℝ)
    (h1 : revPerMin = 400)
    (h2 : tireCircumference = 2) :
    carSpeed revPerMin tireCircumference = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_48kmh_l469_46907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_students_in_pe_or_music_l469_46924

theorem fraction_of_students_in_pe_or_music (total_students : ℚ) : 
  (let pe_students := (1/2) * total_students
   let theatre_students := (1/3) * total_students
   let music_students := total_students - pe_students - theatre_students
   let pe_left := (1/3) * pe_students
   let theatre_left := (1/4) * theatre_students
   let total_left := pe_left + theatre_left
   let remaining_students := total_students - total_left
   let remaining_pe_or_music := pe_students + music_students - theatre_left
   remaining_pe_or_music / remaining_students) = 7/12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_students_in_pe_or_music_l469_46924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_george_says_365_l469_46992

def skip_pattern (n : ℕ) : ℕ → ℕ
  | 0 => 3 * n - 1
  | m + 1 => 3 * (skip_pattern n m) - 1

def george_number : ℕ := skip_pattern 1 6

theorem george_says_365 :
  george_number = 365 ∧ george_number ≤ 1000 ∧ skip_pattern 1 7 > 1000 := by
  sorry

#eval george_number
#eval skip_pattern 1 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_george_says_365_l469_46992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_squares_are_negative_l469_46949

-- Define the set of positive real numbers
def R_pos : Set ℝ := {x : ℝ | x > 0}

-- Define the set of negative real numbers
def R_neg : Set ℝ := {x : ℝ | x < 0}

-- Define the set of pure imaginary numbers
def X : Set ℂ := {z : ℂ | ∃ (b : ℝ), z = Complex.I * b ∧ b ≠ 0}

-- Define the set of squares of pure imaginary numbers
def imaginary_squares : Set ℝ := {x : ℝ | ∃ (z : ℂ), z ∈ X ∧ x = (z.re^2 + z.im^2)}

-- The theorem to prove
theorem imaginary_squares_are_negative : 
  imaginary_squares = R_neg := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_squares_are_negative_l469_46949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_max_l469_46990

noncomputable def a (x m : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, m + Real.cos x)
noncomputable def b (x m : ℝ) : ℝ × ℝ := (Real.cos x, -m + Real.cos x)

noncomputable def f (x m : ℝ) : ℝ := (a x m).1 * (b x m).1 + (a x m).2 * (b x m).2

theorem vector_dot_product_max (m : ℝ) :
  ∃ (x_max : ℝ),
    x_max ∈ Set.Icc (-π/6) (π/3) ∧
    f x_max m = Real.sin (2*x_max + π/6) + 1/2 - m^2 ∧
    f x_max m = 3/2 - m^2 ∧
    x_max = π/6 ∧
    ∀ (x : ℝ), x ∈ Set.Icc (-π/6) (π/3) → f x m ≤ f x_max m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_max_l469_46990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_sum_l469_46919

theorem reciprocal_of_sum : (3 / 4 + 4 / 5 : ℚ)⁻¹ = 20 / 31 := by
  -- Convert fractions to rational numbers
  have h1 : (3 / 4 : ℚ) = 3 / 4 := by norm_num
  have h2 : (4 / 5 : ℚ) = 4 / 5 := by norm_num
  
  -- Add the fractions
  have sum : (3 / 4 + 4 / 5 : ℚ) = 31 / 20 := by
    rw [h1, h2]
    norm_num
  
  -- Take the reciprocal
  rw [sum]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_sum_l469_46919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radii_order_l469_46973

-- Define the radii of the circles
noncomputable def radius_A : ℝ := Real.sqrt 10
def radius_B : ℝ := 5  -- Derived from circumference = 2πr = 10π
def radius_C : ℝ := 4  -- Derived from area = πr² = 16π

-- Theorem to prove the order of radii
theorem circle_radii_order :
  radius_C < radius_A ∧ radius_A < radius_B :=
by
  -- Split the conjunction
  apply And.intro
  -- Prove radius_C < radius_A
  · sorry
  -- Prove radius_A < radius_B
  · sorry

#check circle_radii_order

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radii_order_l469_46973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l469_46904

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) : Prop where
  a_pos : 0 < a
  b_pos : 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The left focus of the hyperbola -/
def left_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- A point on the hyperbola in the second quadrant -/
def point_on_hyperbola (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- A point on the upper asymptote -/
def point_on_upper_asymptote (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- A point on the lower asymptote -/
def point_on_lower_asymptote (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The origin -/
def origin : ℝ × ℝ := (0, 0)

theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) 
  (m n : ℝ) :
  let O := origin
  let P := point_on_hyperbola h
  let M := point_on_upper_asymptote h
  let N := point_on_lower_asymptote h
  (P - O = m • (M - O) + n • (N - O)) →
  (m * n = 1 / 8) →
  eccentricity h = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l469_46904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_gpa_calculation_l469_46947

/-- Given a class where one third of the students have a GPA of 30 and the remaining two thirds have a GPA of 33, prove that the GPA of the entire class is 32. -/
theorem class_gpa_calculation (n : ℕ) (h : n > 0) : 
  (n * 30 + 2 * n * 33) / (3 * n) = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_gpa_calculation_l469_46947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_not_collinear_line_intersecting_skew_lines_determines_two_planes_l469_46926

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a line in 3D space
structure Line3D where
  point : Point3D
  direction : Point3D

-- Define a plane in 3D space
structure Plane3D where
  point : Point3D
  normal : Point3D

-- Define coplanarity for four points
def coplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

-- Define collinearity for three points
def collinear (p1 p2 p3 : Point3D) : Prop := sorry

-- Define skew lines
def skew (l1 l2 : Line3D) : Prop := sorry

-- Define intersection of two lines
def intersects_lines (l1 l2 : Line3D) : Prop := sorry

-- Define intersection of a line and a plane
def intersects_line_plane (l : Line3D) (p : Plane3D) : Prop := sorry

-- Theorem 1: Among four non-coplanar points, any three points are not collinear
theorem three_points_not_collinear (p1 p2 p3 p4 : Point3D) 
  (h : ¬ coplanar p1 p2 p3 p4) : 
  ¬ collinear p1 p2 p3 ∧ ¬ collinear p1 p2 p4 ∧ ¬ collinear p1 p3 p4 ∧ ¬ collinear p2 p3 p4 := by
  sorry

-- Theorem 2: A line that intersects with two skew lines can determine two planes
theorem line_intersecting_skew_lines_determines_two_planes (l1 l2 l3 : Line3D) 
  (h : skew l1 l2) (h1 : intersects_lines l3 l1) (h2 : intersects_lines l3 l2) :
  ∃ (p1 p2 : Plane3D), p1 ≠ p2 ∧ intersects_line_plane l3 p1 ∧ intersects_line_plane l3 p2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_not_collinear_line_intersecting_skew_lines_determines_two_planes_l469_46926
