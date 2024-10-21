import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_sixth_max_sin_sum_in_triangle_l1267_126786

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sin (π/4 + x) * sin (π/4 - x) + Real.sqrt 3 * sin x * cos x

-- Theorem 1: f(π/6) = 1
theorem f_pi_sixth : f (π/6) = 1 := by sorry

-- Theorem 2: In triangle ABC, if f(A/2) = 1, then max(sin(B) + sin(C)) = √3
theorem max_sin_sum_in_triangle (A B C : ℝ) (h1 : 0 < A ∧ A < π) (h2 : 0 < B ∧ B < π) 
  (h3 : 0 < C ∧ C < π) (h4 : A + B + C = π) (h5 : f (A/2) = 1) : 
  (∀ x y : ℝ, 0 < x ∧ x < π ∧ 0 < y ∧ y < π ∧ x + y = π - A → sin x + sin y ≤ Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_sixth_max_sin_sum_in_triangle_l1267_126786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_hexagon_arrangement_l1267_126762

/-- A type representing a hexagonal arrangement of 19 numbers -/
def HexagonArrangement := Fin 19 → ℕ

/-- A function that checks if a given arrangement satisfies the sum condition -/
def is_valid_arrangement (arr : HexagonArrangement) : Prop :=
  ∀ (line : Fin 12), ∃ (a b c : Fin 19), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    arr a + arr b + arr c = 23

/-- The theorem stating that there exists a unique valid hexagon arrangement -/
theorem unique_hexagon_arrangement : 
  ∃! arr : HexagonArrangement, 
    (∀ n : Fin 19, arr n ∈ Set.range (fun i : Fin 19 => i.val + 1)) ∧
    is_valid_arrangement arr :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_hexagon_arrangement_l1267_126762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_cost_calculation_l1267_126785

/-- Calculate the final cost of buying oranges and mangoes with price increase, bulk discount, and sales tax -/
theorem final_cost_calculation (orange_price mango_price : ℚ)
  (price_increase bulk_discount sales_tax : ℚ)
  (orange_quantity mango_quantity : ℕ) :
  let new_orange_price := orange_price * (1 + price_increase)
  let new_mango_price := mango_price * (1 + price_increase)
  let orange_total := new_orange_price * orange_quantity
  let mango_total := new_mango_price * mango_quantity
  let total_before_discount := orange_total + mango_total
  let discount := if orange_quantity ≥ 10 ∧ mango_quantity ≥ 10 then
    bulk_discount * total_before_discount else 0
  let total_after_discount := total_before_discount - discount
  let final_total := total_after_discount * (1 + sales_tax)
  orange_price = 40 ∧
  mango_price = 50 ∧
  price_increase = 15/100 ∧
  bulk_discount = 10/100 ∧
  sales_tax = 8/100 ∧
  orange_quantity = 10 ∧
  mango_quantity = 10 →
  final_total = 1006.02 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_cost_calculation_l1267_126785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_ellipse_standard_equation_l1267_126780

noncomputable def ellipse_equation (x y : ℝ) : Prop := x^2 / 40 + y^2 / 15 = 1

noncomputable def hyperbola_equation (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

def eccentricity : ℚ := 5 / 4

theorem hyperbola_standard_equation :
  ∀ x y : ℝ, hyperbola_equation x y ↔ 
  (∃ c : ℝ, c^2 = 25 ∧ 
    (∀ x₀ y₀ : ℝ, ellipse_equation x₀ y₀ → 
      (x₀ - c)^2 + y₀^2 + (x₀ + c)^2 + y₀^2 = 
      (x - c)^2 + y^2 + (x + c)^2 + y^2)) ∧
  (eccentricity : ℝ) = (5:ℝ) / 4 :=
by
  sorry

-- Part 2: Ellipse equation
noncomputable def ellipse_equation_2 (x y : ℝ) : Prop := x^2 / 45 + y^2 / 5 = 1

theorem ellipse_standard_equation :
  ∀ x y : ℝ, ellipse_equation_2 x y ↔
  (x^2 / 45 + y^2 / 5 = 1 ∧
   ∃ a b : ℝ, a > b ∧ b > 0 ∧
   x^2 / a^2 + y^2 / b^2 = 1 ∧
   ellipse_equation_2 3 2 ∧
   2 * a = 3 * (2 * b)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_ellipse_standard_equation_l1267_126780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_congruent_squares_l1267_126746

/-- A lattice point on a 2D grid -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A square on a lattice grid -/
structure LatticeSquare where
  vertices : Fin 4 → LatticePoint

/-- The size of our grid -/
def gridSize : ℕ := 6

/-- Predicate to check if a LatticeSquare is within our grid bounds -/
def isWithinGrid (s : LatticeSquare) : Prop :=
  ∀ i, 0 ≤ (s.vertices i).x ∧ (s.vertices i).x ≤ gridSize ∧
       0 ≤ (s.vertices i).y ∧ (s.vertices i).y ≤ gridSize

/-- Predicate to check if two LatticeSquares are congruent -/
def areCongruent (s1 s2 : LatticeSquare) : Prop :=
  sorry  -- Definition of congruence would go here

/-- The set of all valid LatticeSquares on our grid -/
def validSquares : Set LatticeSquare :=
  { s | isWithinGrid s }

/-- The set of non-congruent LatticeSquares on our grid -/
noncomputable def nonCongruentSquares : Finset LatticeSquare :=
  sorry  -- Definition using validSquares and areCongruent would go here

theorem count_non_congruent_squares :
  Finset.card nonCongruentSquares = 245 :=
by
  sorry  -- Proof would go here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_congruent_squares_l1267_126746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_for_monotonic_sine_l1267_126766

/-- The maximum value of ω for which f(x) = 2sin(ωx + π/6) is monotonic on [-π/6, π/6] --/
theorem max_omega_for_monotonic_sine (f : ℝ → ℝ) (ω : ℝ) : 
  (∀ x, f x = 2 * Real.sin (ω * x + π / 6)) →
  ω > 0 →
  (∀ x y, x ∈ Set.Icc (-π / 6) (π / 6) → y ∈ Set.Icc (-π / 6) (π / 6) → x < y → f x < f y) →
  ω ≤ 2 ∧ ∀ ε > 0, ∃ x y, x ∈ Set.Icc (-π / 6) (π / 6) ∧ y ∈ Set.Icc (-π / 6) (π / 6) ∧ x < y ∧ f x ≥ f y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_for_monotonic_sine_l1267_126766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_polynomial_l1267_126733

/-- The degree of the polynomial x^2 - 2x + 3 is 2. -/
theorem degree_of_polynomial : 
  Polynomial.degree (X^2 - 2*X + 3 : Polynomial ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_polynomial_l1267_126733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_extreme_points_implies_a_geq_one_l1267_126748

/-- A cubic function with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + x^2 + a*x - 5

/-- The derivative of f with respect to x -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a

/-- A function has no extreme value points if its derivative has at most one real root -/
def has_no_extreme_points (a : ℝ) : Prop :=
  ∀ x y : ℝ, x ≠ y → (f' a x = 0 → f' a y ≠ 0)

theorem no_extreme_points_implies_a_geq_one (a : ℝ) :
  has_no_extreme_points a → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_extreme_points_implies_a_geq_one_l1267_126748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_radius_of_cylinder_l1267_126708

/-- The volume of a cylinder with radius r and height h -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The theorem stating the original radius of the cylinder -/
theorem original_radius_of_cylinder (y : ℝ) :
  let h₀ : ℝ := 3
  let r₀ : ℝ := 3 + Real.sqrt 21
  cylinderVolume (r₀ + 4) h₀ - cylinderVolume r₀ h₀ = y ∧
  cylinderVolume r₀ (h₀ + 4) - cylinderVolume r₀ h₀ = y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_radius_of_cylinder_l1267_126708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_100_abs_eq_one_l1267_126724

noncomputable def z : ℕ → ℂ
  | 0 => 0
  | 1 => 0
  | n + 2 => z (n + 1) ^ 2 - 1

theorem z_100_abs_eq_one : Complex.abs (z 100) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_100_abs_eq_one_l1267_126724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_equation_m_value_l1267_126799

/-- A quadratic equation is a perfect square if it can be written as (x - a)² = 0 for some real a -/
def is_perfect_square_equation (a b c : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (x - k)^2

theorem perfect_square_equation_m_value :
  ∀ m : ℝ, is_perfect_square_equation 1 (-m) 49 → m = 14 ∨ m = -14 := by
  sorry

#check perfect_square_equation_m_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_equation_m_value_l1267_126799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l1267_126719

-- Define the line
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 3

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 4

-- Define the intersection condition
def intersects (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ circle_eq x₁ (line k x₁) ∧ circle_eq x₂ (line k x₂)

-- Define the chord length condition
def chord_length_condition (k : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → circle_eq x₁ (line k x₁) → circle_eq x₂ (line k x₂) →
    (x₁ - x₂)^2 + (line k x₁ - line k x₂)^2 ≥ 12

-- Define the slope angle
noncomputable def slope_angle (k : ℝ) : ℝ := Real.arctan k

-- Theorem statement
theorem slope_angle_range (k : ℝ) :
  intersects k → chord_length_condition k →
  (slope_angle k ∈ Set.Icc 0 (π/6) ∪ Set.Ico (5*π/6) π) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l1267_126719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pit_depth_calculation_l1267_126736

/-- Calculates the depth of a pit in a field given specific conditions -/
theorem pit_depth_calculation (field_length field_width pit_length pit_width height_rise : ℝ) 
  (h1 : field_length = 20)
  (h2 : field_width = 10)
  (h3 : pit_length = 8)
  (h4 : pit_width = 5)
  (h5 : height_rise = 0.5) : 
  (field_length * field_width - pit_length * pit_width) * height_rise / (pit_length * pit_width) = 2 := by
  -- Define intermediate calculations
  let field_area := field_length * field_width
  let pit_area := pit_length * pit_width
  let covered_area := field_area - pit_area
  let earth_volume := covered_area * height_rise
  let pit_depth := earth_volume / pit_area
  
  -- Proof steps (to be filled in)
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pit_depth_calculation_l1267_126736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1267_126756

/-- The sequence {a_n} defined by the given recurrence relation -/
noncomputable def a (a₀ : ℝ) : ℕ → ℝ
  | 0 => a₀
  | n + 1 => Real.sqrt ((1 - Real.sqrt (1 - (a a₀ n)^2)) / 2)

/-- The theorem stating the general term formula for the sequence {a_n} -/
theorem a_formula (a₀ : ℝ) (h : 0 < a₀ ∧ a₀ < 1) :
  ∀ n : ℕ, n ≥ 1 → a a₀ n = Real.sin (Real.arcsin a₀ / 2^(n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1267_126756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_point_y_coordinate_l1267_126709

/-- Area function for a triangle given its three vertices -/
noncomputable def area (triangle : Set (ℝ × ℝ)) : ℝ :=
  sorry -- Definition of area calculation

/-- Given two triangles, one with vertices (6,0), (10,0), and (x,y), and another with vertices (2,0), (8,12), and (14,0), 
    if the area of the first triangle is 1/9 of the area of the second triangle, then y = 4 -/
theorem third_point_y_coordinate 
  (x y : ℝ) 
  (triangle1 : Set (ℝ × ℝ) := {(6,0), (10,0), (x,y)})
  (triangle2 : Set (ℝ × ℝ) := {(2,0), (8,12), (14,0)})
  (h : area triangle1 = (1/9) * area triangle2) : 
  y = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_point_y_coordinate_l1267_126709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1267_126729

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- A point on the ellipse -/
structure EllipsePoint (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Helper function for area of a triangle -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Helper function for perimeter of a triangle -/
noncomputable def perimeter_triangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- The theorem to be proved -/
theorem ellipse_theorem (e : Ellipse) 
    (h_ecc : e.eccentricity = 3/5)
    (h_minor : 2 * e.b = 8)
    (M N : EllipsePoint e)
    (h_circumference : 2 * Real.pi * (area_triangle (M.x, M.y) (N.x, N.y) (-3, 0) / perimeter_triangle (M.x, M.y) (N.x, N.y) (-3, 0)) = Real.pi) :
    |M.y - N.y| = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1267_126729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_exists_l1267_126713

-- Define the given lengths
variable (AD BD CD : ℝ)

-- Assume the given lengths are positive
variable (hAD : AD > 0)
variable (hBD : BD > 0)
variable (hCD : CD > 0)

-- Define the triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the condition for D being on AB
def D_on_AB (AD BD : ℝ) (t : Triangle) : Prop :=
  ∃ (k : ℝ), 0 < k ∧ k < 1 ∧ 
    t.A.1 + k * (t.B.1 - t.A.1) = t.A.1 + AD ∧
    t.A.2 + k * (t.B.2 - t.A.2) = t.A.2

-- Define the condition for CD being the angle bisector
def CD_is_bisector (AD BD : ℝ) (t : Triangle) : Prop :=
  let D := (t.A.1 + AD, t.A.2)
  let AC := ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2).sqrt
  let BC := ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2).sqrt
  AC / AD = BC / BD

-- Theorem statement
theorem triangle_exists (AD BD CD : ℝ) (hAD : AD > 0) (hBD : BD > 0) (hCD : CD > 0) : 
  ∃ (t : Triangle), 
    D_on_AB AD BD t ∧ 
    CD_is_bisector AD BD t ∧
    ((t.C.1 - (t.A.1 + AD))^2 + (t.C.2 - t.A.2)^2).sqrt = CD :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_exists_l1267_126713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_eq_two_l1267_126751

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (x * e^x) / (e^(ax) - 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (x * Real.exp x) / (Real.exp (a * x) - 1)

theorem f_even_implies_a_eq_two (a : ℝ) :
  IsEven (f a) → a = 2 := by
  sorry

#check f_even_implies_a_eq_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_eq_two_l1267_126751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gynecologist_distance_is_50_l1267_126707

/-- Represents the distance and fuel consumption for a round trip to two destinations -/
structure TwoDestinationTrip where
  distance_to_dermatologist : ℚ
  car_efficiency : ℚ
  total_fuel_used : ℚ

/-- Calculates the distance to the gynecologist given a TwoDestinationTrip -/
def distance_to_gynecologist (trip : TwoDestinationTrip) : ℚ :=
  ((trip.car_efficiency * trip.total_fuel_used) / 2) - trip.distance_to_dermatologist

/-- Theorem stating that the distance to the gynecologist is 50 miles for the given conditions -/
theorem gynecologist_distance_is_50 (trip : TwoDestinationTrip)
    (h1 : trip.distance_to_dermatologist = 30)
    (h2 : trip.car_efficiency = 20)
    (h3 : trip.total_fuel_used = 8) :
    distance_to_gynecologist trip = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gynecologist_distance_is_50_l1267_126707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_b_greater_than_a_l1267_126783

def A : Finset ℕ := {1, 2, 3, 4, 5}
def B : Finset ℕ := {1, 2, 3}

def favorable_outcomes : Finset (ℕ × ℕ) :=
  A.product B |>.filter (fun (a, b) => b > a)

theorem probability_b_greater_than_a :
  (favorable_outcomes.card : ℚ) / ((A.card * B.card) : ℚ) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_b_greater_than_a_l1267_126783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_example_l1267_126726

/-- The total surface area of a right pyramid with a square base --/
noncomputable def pyramid_surface_area (base_side : ℝ) (height : ℝ) : ℝ :=
  let base_area := base_side ^ 2
  let slant_height := Real.sqrt (height ^ 2 + (base_side / 2) ^ 2)
  let triangular_face_area := (base_side * slant_height) / 2
  base_area + 4 * triangular_face_area

/-- Theorem: The surface area of a right pyramid with base side 6 and height 15 is approximately 219.564 --/
theorem pyramid_surface_area_example : 
  ∃ ε > 0, |pyramid_surface_area 6 15 - 219.564| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_example_l1267_126726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_benzene_weight_l1267_126789

/-- The molecular weight of a compound given its composition and the number of moles -/
def molecular_weight (carbon_count : ℕ) (hydrogen_count : ℕ) (carbon_weight : ℝ) (hydrogen_weight : ℝ) (moles : ℝ) : ℝ :=
  moles * (carbon_count * carbon_weight + hydrogen_count * hydrogen_weight)

/-- Theorem stating that the molecular weight of approximately 3.994 moles of C6H6 is 312 g -/
theorem benzene_weight :
  let carbon_count : ℕ := 6
  let hydrogen_count : ℕ := 6
  let carbon_weight : ℝ := 12.01
  let hydrogen_weight : ℝ := 1.008
  let moles : ℝ := 3.994
  ∃ ε > 0, |molecular_weight carbon_count hydrogen_count carbon_weight hydrogen_weight moles - 312| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_benzene_weight_l1267_126789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_C_intersects_D_l1267_126700

/-- A line in the form y = kx - 1 -/
structure Line where
  k : ℝ

/-- A circle with center (a, b) and radius r -/
structure Circle where
  a : ℝ
  b : ℝ
  r : ℝ

/-- Distance from a point (x, y) to a line y = kx - 1 -/
noncomputable def distanceToLine (x y : ℝ) (l : Line) : ℝ :=
  |l.k * x - y - 1| / Real.sqrt (l.k^2 + 1)

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  distanceToLine c.a c.b l = c.r

/-- Checks if a line intersects a circle -/
def intersects (l : Line) (c : Circle) : Prop :=
  distanceToLine c.a c.b l < c.r

theorem line_tangent_to_C_intersects_D (l : Line) :
  isTangent l { a := 2, b := 1, r := Real.sqrt 2 } →
  intersects l { a := 2, b := 0, r := Real.sqrt 3 } := by
  sorry

#check line_tangent_to_C_intersects_D

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_C_intersects_D_l1267_126700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_separate_l1267_126727

/-- Two circles are separate if the distance between their centers is greater than the sum of their radii -/
def circles_are_separate (center1 center2 : ℝ × ℝ) (radius1 radius2 : ℝ) : Prop :=
  Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2) > radius1 + radius2

/-- Given two circles, prove they are separate -/
theorem circles_separate :
  let circle1 := fun (x y : ℝ) ↦ x^2 + y^2 = 1
  let circle2 := fun (x y : ℝ) ↦ (x-3)^2 + (y+4)^2 = 9
  circles_are_separate (0, 0) (3, -4) 1 3 := by
  sorry

#check circles_separate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_separate_l1267_126727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_lambda_range_l1267_126755

/-- A sequence is increasing if each term is greater than the previous one. -/
def IsIncreasing (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n < a (n + 1)

/-- The theorem states that for a sequence with the given general term,
    if it's increasing, then λ must be in the specified range. -/
theorem increasing_sequence_lambda_range (lambda : ℝ) :
  let a : ℕ+ → ℝ := fun n => (n : ℝ) ^ 2 - 2 * lambda * n + 1
  IsIncreasing a → lambda < 3 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_lambda_range_l1267_126755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_theorem_l1267_126758

-- Define the ellipse C
noncomputable def ellipse (x y a b : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2/a^2)

theorem ellipse_slope_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : eccentricity a b = Real.sqrt 2 / 2)
  (h4 : ellipse 1 (Real.sqrt 2 / 2) a b)
  (h5 : ∃ k : ℝ, (∀ x y : ℝ, ellipse x y a b → y = k * (x - a) ∨ y = -1/k * x + b) 
              ∧ (k ≠ 0)) :
  ∃ x1 y1 x2 y2 : ℝ, 
    ellipse x1 y1 a b ∧ 
    ellipse x2 y2 a b ∧ 
    (y2 - y1) / (x2 - x1) = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_theorem_l1267_126758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_l1267_126787

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem function_and_range 
  (A ω φ : ℝ) 
  (h1 : A > 0) 
  (h2 : ω > 0) 
  (h3 : 0 < φ ∧ φ < Real.pi / 2) 
  (h4 : ∀ x y : ℝ, f A ω φ x = 0 → f A ω φ y = 0 → y ≠ x → |y - x| ≥ Real.pi / 2) 
  (h5 : f A ω φ (2 * Real.pi / 3) = -2) :
  (∀ x : ℝ, f A ω φ x = 2 * Real.sin (2 * x + Real.pi / 6)) ∧
  (Set.Icc (Real.pi / 12) (Real.pi / 2) ⊆ (f A ω φ) ⁻¹' (Set.Icc (-1) 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_l1267_126787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_circle_equation_l1267_126776

-- Define the circle C
def circle_C (p : ℝ × ℝ) : Prop := (p.1 - 1)^2 + (p.2 - 1)^2 = 1

-- Define point P
def point_P : ℝ × ℝ := (-1, -1)

-- Define the property of being tangent to circle C at two points
def tangent_to_C (f : ℝ × ℝ → Prop) : Prop :=
  ∃ A B : ℝ × ℝ, A ≠ B ∧ circle_C A ∧ circle_C B ∧
    (∀ p : ℝ × ℝ, f p → ¬(circle_C p)) ∧
    (∃ ε > 0, ∀ p : ℝ × ℝ, (p.1 - A.1)^2 + (p.2 - A.2)^2 < ε^2 → (f p ∨ circle_C p)) ∧
    (∃ ε > 0, ∀ p : ℝ × ℝ, (p.1 - B.1)^2 + (p.2 - B.2)^2 < ε^2 → (f p ∨ circle_C p))

-- Define the new circle
def new_circle (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 = 2

-- Theorem statement
theorem new_circle_equation :
  tangent_to_C new_circle ∧ new_circle point_P :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_circle_equation_l1267_126776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rectangle_area_l1267_126781

-- Define the curve C in polar coordinates
noncomputable def C (θ : ℝ) : ℝ × ℝ := 
  let ρ := 2 * (Real.sin θ + Real.cos θ + 1 / (2 * (Real.sin θ + Real.cos θ)))
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the area of rectangle OAPB for a given point on C
noncomputable def rectangle_area (θ : ℝ) : ℝ :=
  let (x, y) := C θ
  abs (x * y)

-- Theorem statement
theorem max_rectangle_area :
  ∃ (θ : ℝ), ∀ (φ : ℝ), rectangle_area θ ≥ rectangle_area φ ∧ rectangle_area θ = 3 + 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rectangle_area_l1267_126781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_axis_endpoint_distance_l1267_126767

/-- Given an ellipse with equation 4(x+2)^2 + 16y^2 = 64, 
    the distance between an endpoint of its major axis 
    and an endpoint of its minor axis is 2√5. -/
theorem ellipse_axis_endpoint_distance : 
  ∃ (C D : ℝ × ℝ),
    (∀ (x y : ℝ), 4 * (x + 2)^2 + 16 * y^2 = 64 ↔ (x, y) ∈ {p : ℝ × ℝ | (p.1 + 2)^2 / 16 + p.2^2 / 4 = 1}) →
    (C ∈ {p : ℝ × ℝ | (p.1 + 2)^2 / 16 + p.2^2 / 4 = 1}) →
    (D ∈ {p : ℝ × ℝ | (p.1 + 2)^2 / 16 + p.2^2 / 4 = 1}) →
    (C.2 = 0 ∧ C.1 = -2 + 4 ∨ C.1 = -2 - 4) →
    (D.1 = -2 ∧ D.2 = 2 ∨ D.2 = -2) →
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_axis_endpoint_distance_l1267_126767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramids_to_cube_l1267_126797

-- Define the edge length of the pyramids
noncomputable def pyramid_edge : ℝ := 1

-- Define the side length of the cube
noncomputable def cube_side : ℝ := 1 / Real.sqrt 2

-- Volume of a regular tetrahedron (triangular pyramid)
noncomputable def tetrahedron_volume (edge : ℝ) : ℝ := 
  (edge ^ 3) / (6 * Real.sqrt 2)

-- Volume of a quadrangular pyramid
noncomputable def quad_pyramid_volume (edge : ℝ) : ℝ := 
  (edge ^ 3) * (Real.sqrt 2 - 1) / 3

-- Volume of a cube
noncomputable def cube_volume (side : ℝ) : ℝ := side ^ 3

-- Theorem statement
theorem pyramids_to_cube : 
  tetrahedron_volume pyramid_edge + quad_pyramid_volume pyramid_edge = 
  cube_volume cube_side := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramids_to_cube_l1267_126797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_not_closed_under_subtraction_l1267_126734

def v : Set ℕ := {1, 4, 9, 16, 25}

theorem v_not_closed_under_subtraction :
  ¬ (∀ (a b : ℕ), a ∈ v → b ∈ v → a ≠ b → (a - b) ∈ v) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_not_closed_under_subtraction_l1267_126734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_to_decimal_l1267_126728

theorem fraction_to_decimal (n d : ℕ) (h : d = 2^5 * 5) :
  (n : ℚ) / d = 0.33125 :=
by
  sorry

#eval (53 : ℚ) / 160

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_to_decimal_l1267_126728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l1267_126740

/-- Given a line L1 with equation x - 2y + 3 = 0 and a point P (-1, 3),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation 2x + y - 1 = 0 -/
theorem perpendicular_line_equation 
  (L1 L2 : Set (ℝ × ℝ)) 
  (P : ℝ × ℝ) :
  (∀ x y, (x, y) ∈ L1 ↔ x - 2 * y + 3 = 0) →
  P = (-1, 3) →
  P ∈ L2 →
  (∀ x y x' y', (x, y) ∈ L1 → (x', y') ∈ L2 → 
    (x - x') * (x - x') + (y - y') * (y - y') = 0) →
  (∀ x y, (x, y) ∈ L2 ↔ 2 * x + y - 1 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l1267_126740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_deduction_proof_l1267_126769

/-- Represents the hourly wage in dollars -/
noncomputable def hourly_wage : ℚ := 25

/-- Represents the local tax rate as a percentage -/
noncomputable def tax_rate : ℚ := 21/10

/-- Represents the number of cents in a dollar -/
noncomputable def cents_per_dollar : ℚ := 100

/-- Calculates the amount of tax deducted in cents -/
noncomputable def tax_deducted_cents : ℚ :=
  hourly_wage * cents_per_dollar * (tax_rate / 100)

theorem tax_deduction_proof :
  tax_deducted_cents = 525/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_deduction_proof_l1267_126769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1267_126792

/-- Given a triangle ABC with side BC = 2, angle A = π/3, and angle B = π/4,
    the length of side AB is (3√2 + √6) / 3 -/
theorem triangle_side_length (A B C : ℝ) (BC : ℝ) (angleA : ℝ) (angleB : ℝ) :
  BC = 2 →
  angleA = π / 3 →
  angleB = π / 4 →
  let angleC := π - angleA - angleB
  let AB := BC * (Real.sin angleA) / (Real.sin angleC)
  AB = (3 * Real.sqrt 2 + Real.sqrt 6) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1267_126792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_cardinality_l1267_126731

-- Define the universal set U
def U : Finset Char := {'a', 'b', 'c', 'd', 'e', 'f'}

-- Define set A
def A : Finset Char := {'a', 'b', 'c', 'd', 'e'}

-- Define set B
def B : Finset Char := {'c', 'd', 'e', 'f'}

-- Theorem statement
theorem complement_intersection_cardinality :
  (U \ (A ∩ B)).card = 3 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_cardinality_l1267_126731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_values_l1267_126763

noncomputable def a : ℝ := 21.2

noncomputable def b : ℝ := Real.sqrt 450 - 0.8

noncomputable def c : ℝ := 2 * (Real.log 2 / Real.log 5)

theorem order_of_values : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_values_l1267_126763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_approx_1_6b_l1267_126764

-- Define a and b as given in the problem
noncomputable def a : ℝ := Real.log 576 / Real.log 16
noncomputable def b : ℝ := Real.log 24 / Real.log 4

-- State the theorem to be proved
theorem a_approx_1_6b : ∃ ε > 0, |a - 1.6 * b| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_approx_1_6b_l1267_126764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_dozen_pens_l1267_126716

/-- The cost of stationery items -/
structure StationeryCost where
  pen_cost : ℕ
  pencil_cost : ℕ
  total_cost : ℕ
  pen_pencil_ratio : ℚ

/-- The given conditions for the stationery cost problem -/
def given_conditions : StationeryCost :=
  { pen_cost := 65
  , pencil_cost := 13  -- Derived from the ratio, not directly given
  , total_cost := 260
  , pen_pencil_ratio := 5/1 }

/-- Theorem stating the cost of one dozen pens -/
theorem cost_of_dozen_pens (sc : StationeryCost) 
  (h1 : sc.pen_cost = 65)
  (h2 : sc.pen_pencil_ratio = 5/1)
  (h3 : sc.total_cost = 260)
  (h4 : 3 * sc.pen_cost + 5 * sc.pencil_cost = sc.total_cost) :
  12 * sc.pen_cost = 780 :=
by sorry

#eval 12 * given_conditions.pen_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_dozen_pens_l1267_126716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_product_l1267_126795

def Q (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem roots_product (a b c d : ℝ) :
  (∀ x : ℝ, Q a b c d x = 0 ↔ x ∈ ({Real.cos (π/9), Real.cos (2*π/9), Real.cos (4*π/9), Real.cos (8*π/9)} : Set ℝ)) →
  a * b * c * d = -1/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_product_l1267_126795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_measure_of_angle_A_area_of_triangle_ABC_l1267_126757

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) : Prop :=
  ∃ (a b c : ℝ),
    a = 8 ∧ b = 7 ∧
    Real.cos B = -1/7 ∧
    -- Additional conditions to ensure it's a valid triangle
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem for the measure of angle A
theorem measure_of_angle_A (A B C : ℝ) :
  triangle_ABC A B C → A = π/3 :=
by sorry

-- Theorem for the area of triangle ABC
theorem area_of_triangle_ABC (A B C : ℝ) :
  triangle_ABC A B C → (1/2) * 8 * 7 * Real.sin C = 6 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_measure_of_angle_A_area_of_triangle_ABC_l1267_126757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_hiring_manager_acceptance_range_l1267_126782

/-- Represents the hiring manager's acceptance criteria for job applicants based on age. -/
structure HiringCriteria where
  average_age : ℝ
  std_dev : ℝ
  max_different_ages : ℕ

/-- 
Calculates the number of standard deviations from the average age that the hiring manager 
is willing to accept, given the hiring criteria.
-/
noncomputable def calculate_std_dev_range (criteria : HiringCriteria) : ℝ :=
  ((criteria.max_different_ages : ℝ) - 1) / (2 * criteria.std_dev)

/-- 
Theorem stating that for the given hiring criteria, the number of standard deviations 
from the average age that the hiring manager is willing to accept is 1.
-/
theorem hiring_manager_acceptance_range 
  (criteria : HiringCriteria) 
  (h1 : criteria.average_age = 31)
  (h2 : criteria.std_dev = 5)
  (h3 : criteria.max_different_ages = 11) : 
  calculate_std_dev_range criteria = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_hiring_manager_acceptance_range_l1267_126782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_count_proof_l1267_126718

theorem disk_count_proof (blue yellow green : ℕ) : 
  blue + yellow + green > 0 →
  (blue : ℕ) / 3 = (yellow : ℕ) / 7 ∧ (yellow : ℕ) / 7 = (green : ℕ) / 8 →
  green = blue + 40 →
  blue + yellow + green = 144 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_count_proof_l1267_126718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1267_126741

theorem simplify_expression : (-2) + (-6) - (-3) - 2 = -2 - 6 + 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1267_126741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l1267_126794

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x + 1) / Real.log (1/2)

-- State the theorem
theorem monotonic_increase_interval :
  ∀ x y : ℝ, x < y ∧ x < 1 ∧ y < 1 → f x < f y :=
by
  -- We use 'sorry' to skip the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l1267_126794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_arithmetic_sum_l1267_126747

-- Define the geometric sequence and its sum
noncomputable def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁
  else a₁ * (1 - q^n) / (1 - q)

-- State the theorem
theorem geometric_sequence_arithmetic_sum 
  (a₁ : ℝ) (q : ℝ) (n : ℕ) (h₁ : a₁ ≠ 0) (h₂ : q ≠ 1) :
  (∀ n : ℕ, 2 * geometric_sum a₁ q n = geometric_sum a₁ q (n + 1) + geometric_sum a₁ q (n + 2)) →
  q = -2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_arithmetic_sum_l1267_126747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_approx_l1267_126745

/-- Represents the height of the frustum in decimeters -/
noncomputable def m : ℝ := 3 / Real.pi

/-- Represents the inclination angle of the slant height in radians -/
noncomputable def α : ℝ := (43 + 40/60 + 42.2/3600) * Real.pi / 180

/-- Represents the volume of the complete cone in cubic meters -/
def k : ℝ := 1

/-- Calculates the volume of a frustum given the height of the complete cone and the height of the frustum -/
noncomputable def frustumVolume (H h : ℝ) : ℝ :=
  k * (1 - (H - h)^3 / H^3)

/-- Theorem stating that the volume of the frustum is approximately 0.79 cubic meters -/
theorem frustum_volume_approx :
  ∃ (H h : ℝ), H > h ∧ h > 0 ∧ H - h = m ∧ 
  abs (frustumVolume H h - 0.79) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_approx_l1267_126745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_values_l1267_126737

noncomputable def point_on_terminal_side (α : ℝ) (x y : ℝ) : Prop :=
  x = r * Real.cos α ∧ y = r * Real.sin α
  where r : ℝ := Real.sqrt (x^2 + y^2)

theorem angle_values (α : ℝ) (a : ℝ) :
  point_on_terminal_side α (-4) a →
  Real.sin α * Real.cos α = Real.sqrt 3 / 4 →
  a = -4 * Real.sqrt 3 ∨ a = -4 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_values_l1267_126737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisibility_l1267_126715

theorem smallest_number_divisibility (n : ℕ) :
  n = 1020 - 12 →
  (∀ d : ℕ, d ∈ ({12, 24, 36, 48, 56} : Set ℕ) → d ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisibility_l1267_126715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dawn_commissioned_paintings_l1267_126772

/-- Calculates the number of paintings commissioned given the painting time, total earnings, and hourly rate. -/
noncomputable def paintings_commissioned (painting_time : ℝ) (total_earnings : ℝ) (hourly_rate : ℝ) : ℝ :=
  total_earnings / (painting_time * hourly_rate)

/-- Proves that Dawn was commissioned to paint 12 paintings. -/
theorem dawn_commissioned_paintings :
  paintings_commissioned 2 3600 150 = 12 := by
  -- Unfold the definition of paintings_commissioned
  unfold paintings_commissioned
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dawn_commissioned_paintings_l1267_126772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_data_set_variance_l1267_126714

noncomputable def data_set : List ℝ := [9.8, 9.9, 10, 10.1, 10.2]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let μ := mean xs
  (xs.map (fun x => (x - μ) ^ 2)).sum / xs.length

theorem data_set_variance :
  mean data_set = 10 → variance data_set = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_data_set_variance_l1267_126714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_replacement_factor_five_l1267_126754

theorem digit_replacement_factor_five (N : ℕ) :
  (∃ (k : ℕ) (a : ℕ), 
    N = 125 * a * 10^197 ∧ 
    a ∈ ({1, 2, 3} : Finset ℕ) ∧
    k < 200) ↔ 
  (∃ (m n : ℕ) (k : ℕ) (a : ℕ),
    N = m + 10^k * a + 10^(k+1) * n ∧
    k < 200 ∧
    m < 10^k ∧
    a < 10 ∧
    m + 10^(k+1) * n = N / 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_replacement_factor_five_l1267_126754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_angle_approximation_l1267_126717

/-- Given a circle with radius 10 meters and a sector area of 36.67 square meters,
    the central angle of the sector is approximately 42 degrees. -/
theorem sector_angle_approximation (r : ℝ) (area : ℝ) (θ : ℝ) :
  r = 10 →
  area = 36.67 →
  area = (θ / 360) * Real.pi * r^2 →
  ∃ ε > 0, |θ - 42| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_angle_approximation_l1267_126717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moments_of_X_l1267_126752

/-- Probability density function for the random variable X -/
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 2 then 0.5 * x else 0

/-- Raw moment of order k -/
noncomputable def raw_moment (k : ℕ) : ℝ :=
  ∫ x in Set.Icc 0 2, x^k * f x

/-- Central moment of order k -/
noncomputable def central_moment (k : ℕ) : ℝ :=
  ∫ x in Set.Icc 0 2, (x - raw_moment 1)^k * f x

/-- Theorem stating the values of raw and central moments -/
theorem moments_of_X :
  (raw_moment 1 = 4/3) ∧
  (raw_moment 2 = 2) ∧
  (raw_moment 3 = 16/5) ∧
  (raw_moment 4 = 16/3) ∧
  (central_moment 2 = 2/9) ∧
  (central_moment 3 = -8/135) ∧
  (central_moment 4 = 16/135) := by
  sorry

#check moments_of_X

end NUMINAMATH_CALUDE_ERRORFEEDBACK_moments_of_X_l1267_126752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_term_coefficient_l1267_126774

theorem linear_term_coefficient (x : ℝ) : 
  let p := fun x : ℝ ↦ x^2 - 2*x - 3
  (deriv p x - deriv (deriv p) x * x) / 2 = -2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_term_coefficient_l1267_126774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_distinct_t_is_28_div_3_l1267_126749

/-- A polynomial of the form x^2 - 7x + t with only positive integer roots -/
def PolyWithPositiveIntegerRoots (t : ℤ) : Prop :=
  ∃ (r₁ r₂ : ℤ), r₁ > 0 ∧ r₂ > 0 ∧ r₁ + r₂ = 7 ∧ r₁ * r₂ = t

/-- The set of all possible values of t -/
def PossibleT : Set ℤ :=
  {t : ℤ | PolyWithPositiveIntegerRoots t}

/-- The list of distinct possible values of t -/
def DistinctPossibleT : List ℤ := [6, 10, 12]

/-- The average of all distinct possible values of t -/
noncomputable def AverageOfDistinctT : ℚ :=
  (DistinctPossibleT.sum) / (DistinctPossibleT.length : ℚ)

theorem average_of_distinct_t_is_28_div_3 :
  AverageOfDistinctT = 28 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_distinct_t_is_28_div_3_l1267_126749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_equation_trajectory_of_M_l1267_126753

-- Define the circle C
noncomputable def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define point P
def point_P : ℝ × ℝ := (3, 1)

-- Define a general point Q on the circle
def point_Q (x y : ℝ) : Prop := circle_C x y

-- Define the midpoint M of PQ
noncomputable def midpoint_M (x y : ℝ) : ℝ × ℝ := ((x + 3) / 2, (y + 1) / 2)

-- Theorem for the equation of line AB
theorem line_AB_equation : 
  ∃ (x_A y_A x_B y_B : ℝ),
    circle_C x_A y_A ∧ 
    circle_C x_B y_B ∧ 
    (∀ (x y : ℝ), circle_C x y → (x - 3) * (y - y_A) = (y - 1) * (x - x_A)) ∧
    (∀ (x y : ℝ), circle_C x y → (x - 3) * (y - y_B) = (y - 1) * (x - x_B)) ∧
    (2 * x_A + y_A - 3 = 0) ∧
    (2 * x_B + y_B - 3 = 0) :=
by sorry

-- Theorem for the trajectory of midpoint M
theorem trajectory_of_M : 
  ∀ (x y : ℝ),
    (∃ (x_Q y_Q : ℝ), point_Q x_Q y_Q ∧ midpoint_M x_Q y_Q = (x, y)) ↔
    (x - 2)^2 + (y - 1/2)^2 = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_equation_trajectory_of_M_l1267_126753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_ratio_implies_100th_term_l1267_126742

/-- Arithmetic sequence with common difference 3 -/
noncomputable def arithmetic_sequence (a : ℝ) : ℕ → ℝ := fun n ↦ a + 3 * (n - 1 : ℝ)

/-- Sum of first k terms of the arithmetic sequence -/
noncomputable def S (a : ℝ) (k : ℕ) : ℝ := k * (2 * a + 3 * ((k : ℝ) - 1)) / 2

/-- Ratio of S_3n to S_n -/
noncomputable def ratio (a : ℝ) (n : ℕ) : ℝ := S a (3 * n) / S a n

theorem constant_ratio_implies_100th_term (a : ℝ) :
  (∀ n : ℕ, ratio a n = 9) →
  arithmetic_sequence a 100 = 298.5 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_ratio_implies_100th_term_l1267_126742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_with_nines_l1267_126725

theorem power_of_two_with_nines (k : ℕ) (hk : k > 1) :
  ∃ n : ℕ, ∃ m : ℕ, (2^n) % (10^k) = m ∧ 
  (∃ digits : List ℕ, digits.length = k ∧ 
   digits = (fun i => (m / 10^i) % 10) <$> List.range k ∧
   (digits.filter (fun x => x = 9)).length ≥ k / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_with_nines_l1267_126725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_distribution_is_correct_l1267_126793

/-- Represents the division of bread and money between two woodcutters and a hunter. -/
structure WoodcuttersProblem where
  ivan_loaves : ℕ
  prokhor_loaves : ℕ
  total_loaves : ℕ
  total_kopecks : ℕ
  people : ℕ

/-- Represents the fair distribution of money between the woodcutters. -/
structure FairDistribution where
  ivan_kopecks : ℕ
  prokhor_kopecks : ℕ
deriving Repr

/-- Calculates the fair distribution of money based on the problem setup. -/
def calculate_fair_distribution (p : WoodcuttersProblem) : FairDistribution :=
  let loaves_per_person := p.total_loaves / p.people
  let kopecks_per_loaf := p.total_kopecks / p.total_loaves
  let ivan_shared := max (p.ivan_loaves - loaves_per_person) 0
  let prokhor_shared := max (p.prokhor_loaves - loaves_per_person) 0
  { ivan_kopecks := ivan_shared * kopecks_per_loaf,
    prokhor_kopecks := prokhor_shared * kopecks_per_loaf + (p.total_kopecks - ivan_shared * kopecks_per_loaf) }

/-- Theorem stating that the calculated distribution is fair given the problem conditions. -/
theorem fair_distribution_is_correct (p : WoodcuttersProblem) 
    (h1 : p.ivan_loaves = 4)
    (h2 : p.prokhor_loaves = 8)
    (h3 : p.total_loaves = 12)
    (h4 : p.total_kopecks = 60)
    (h5 : p.people = 3) :
  let d := calculate_fair_distribution p
  d.ivan_kopecks = 20 ∧ d.prokhor_kopecks = 40 := by
  sorry

#eval calculate_fair_distribution { ivan_loaves := 4, prokhor_loaves := 8, total_loaves := 12, total_kopecks := 60, people := 3 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_distribution_is_correct_l1267_126793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_badminton_match_probability_l1267_126721

/-- The probability of player A winning a single game -/
def p : ℚ := 2/3

/-- The number of games needed to win the match -/
def games_to_win : ℕ := 3

/-- The total number of games played when A wins 3:1 -/
def total_games : ℕ := 4

/-- The probability of player A winning a badminton match with a score of 3:1 -/
theorem badminton_match_probability :
  (Finset.card (Finset.filter (λ i => i < games_to_win) (Finset.range total_games)) : ℚ) * p^games_to_win * (1 - p) = 8/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_badminton_match_probability_l1267_126721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_limit_l1267_126712

noncomputable def my_sequence (n : ℕ) : ℝ :=
  (Real.sqrt ((n^3 + 1) * (n^2 + 3)) - Real.sqrt (n * (n^4 + 2))) / (2 * Real.sqrt n)

theorem my_sequence_limit : 
  Filter.Tendsto my_sequence Filter.atTop (nhds (3/4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_limit_l1267_126712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_proof_l1267_126739

/-- The cost price of a ball, given the selling price and loss condition -/
def cost_price_of_ball (selling_price : ℕ) (num_balls : ℕ) (loss_balls : ℕ) : ℕ :=
  let cost_price := 48
  cost_price

#eval cost_price_of_ball 720 20 5

theorem cost_price_proof (selling_price num_balls loss_balls : ℕ) 
  (h1 : selling_price = 720)
  (h2 : num_balls = 20)
  (h3 : loss_balls = 5) : 
  num_balls * (cost_price_of_ball selling_price num_balls loss_balls) = 
    selling_price + loss_balls * (cost_price_of_ball selling_price num_balls loss_balls) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_proof_l1267_126739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_midpoint_existence_l1267_126705

-- Define the trajectory
def trajectory (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line
def line_k (k x y : ℝ) : Prop := y = k*(x+4) ∧ k ≠ 0

-- Main theorem
theorem trajectory_and_midpoint_existence :
  (∀ x y : ℝ, x^2 + y^2 = 4 → ∃ x' y' : ℝ, trajectory x' y' ∧ 2*y' = Real.sqrt 3*y) ∧
  (∃ k : ℝ, k ≠ 0 ∧
    ∃ x₁ y₁ x₂ y₂ x y : ℝ,
      trajectory x₁ y₁ ∧ trajectory x₂ y₂ ∧
      line_k k x₁ y₁ ∧ line_k k x₂ y₂ ∧
      x = (x₁ + x₂)/2 ∧ y = (y₁ + y₂)/2 ∧
      x^2 + y^2 = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_midpoint_existence_l1267_126705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polynomial_is_identity_l1267_126798

/-- A real-coefficient polynomial satisfying specific conditions -/
def SpecialPolynomial (f : ℝ → ℝ) : Prop :=
  (∀ a : ℝ, f (a + 1) = f a + f 1) ∧
  (∃ (k₁ : ℝ) (n : ℕ) (k : Fin (n + 1) → ℝ), 
    k₁ ≠ 0 ∧ n > 0 ∧
    (∀ i : Fin n, f (k i) = k i.succ) ∧
    f (k (Fin.last (n + 1))) = k₁)

/-- The only real-coefficient polynomial satisfying the SpecialPolynomial conditions is f(x) = x -/
theorem special_polynomial_is_identity :
  ∀ f : ℝ → ℝ, SpecialPolynomial f → f = id := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polynomial_is_identity_l1267_126798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ramu_loss_percent_l1267_126773

noncomputable def car_purchase_price : ℝ := 42000
noncomputable def repair_cost : ℝ := 13000
noncomputable def tax_usd : ℝ := 500
noncomputable def usd_to_inr_rate : ℝ := 75
noncomputable def insurance_php : ℝ := 3000
noncomputable def php_to_inr_rate : ℝ := 1.5
noncomputable def selling_price : ℝ := 83000

noncomputable def total_cost : ℝ := car_purchase_price + repair_cost + tax_usd * usd_to_inr_rate + insurance_php * php_to_inr_rate

noncomputable def loss : ℝ := total_cost - selling_price

noncomputable def loss_percent : ℝ := (loss / total_cost) * 100

theorem ramu_loss_percent :
  abs (loss_percent - 14.43) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ramu_loss_percent_l1267_126773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_manufacturer_pricing_l1267_126744

/-- Calculates the minimum selling price per unit to ensure a specified monthly profit -/
noncomputable def minimum_selling_price (units : ℕ) (production_cost : ℚ) (variable_cost : ℚ) (desired_profit : ℚ) : ℚ :=
  (units * (production_cost + variable_cost) + desired_profit) / units

theorem chocolate_manufacturer_pricing :
  let units : ℕ := 400
  let production_cost : ℚ := 40
  let variable_cost : ℚ := 10
  let desired_profit : ℚ := 40000
  minimum_selling_price units production_cost variable_cost desired_profit = 150 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_manufacturer_pricing_l1267_126744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_chord_theorem_l1267_126743

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the left focus
def left_focus : ℝ × ℝ := (-2, 0)

-- Define the chord AB
def chord_AB (x y : ℝ) : Prop :=
  y = (Real.sqrt 3 / 3) * (x + 2)

-- Define that AB passes through F1
def AB_through_F1 (A B : ℝ × ℝ) : Prop :=
  chord_AB A.1 A.2 ∧ chord_AB B.1 B.2 ∧
  (A = left_focus ∨ B = left_focus)

-- Theorem statement
theorem hyperbola_chord_theorem 
  (A B : ℝ × ℝ) (h1 : AB_through_F1 A B) 
  (h2 : hyperbola A.1 A.2) (h3 : hyperbola B.1 B.2) :
  ∃ (AB_length : ℝ) (perimeter : ℝ),
    AB_length = 3 ∧ perimeter = 3 + 3 * Real.sqrt 3 := by
  let right_focus : ℝ × ℝ := (2, 0)
  let AB_length := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let AF2_length := Real.sqrt ((A.1 - right_focus.1)^2 + (A.2 - right_focus.2)^2)
  let BF2_length := Real.sqrt ((B.1 - right_focus.1)^2 + (B.2 - right_focus.2)^2)
  let perimeter := AB_length + AF2_length + BF2_length
  exists AB_length, perimeter
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_chord_theorem_l1267_126743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l1267_126765

-- Define the circle
def my_circle (R : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = R^2}

-- Define the line
def my_line (x₀ y₀ R : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | x₀ * p.1 + y₀ * p.2 = R^2}

-- Define what it means for a point to be outside the circle
def outside_circle (x₀ y₀ R : ℝ) : Prop := x₀^2 + y₀^2 > R^2

-- Define what it means for a line to intersect a circle
def intersects (s t : Set (ℝ × ℝ)) : Prop := ∃ p, p ∈ s ∧ p ∈ t

-- The theorem
theorem line_intersects_circle (x₀ y₀ R : ℝ) (h : outside_circle x₀ y₀ R) :
  intersects (my_line x₀ y₀ R) (my_circle R) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l1267_126765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_assignment_count_l1267_126706

/-- The number of ways to assign four students to three different classes -/
def assign_students : ℕ := 36

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of classes -/
def num_classes : ℕ := 3

/-- Theorem stating that the number of ways to assign four students to three different classes,
    with each class having at least one student, is equal to 36 -/
theorem student_assignment_count :
  (∀ (assignment : Fin num_students → Fin num_classes),
    (∀ c : Fin num_classes, ∃ s : Fin num_students, assignment s = c) →
    ∃! (perm : Equiv.Perm (Fin num_students)), assignment = fun s => assignment (perm s)) →
  assign_students = 36 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_assignment_count_l1267_126706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_centroid_focus_l1267_126788

/-- The parabola y^2 = 4√2x -/
def parabola (x y : ℝ) : Prop := y^2 = 4 * Real.sqrt 2 * x

/-- The focus of the parabola -/
noncomputable def focus : ℝ × ℝ := (Real.sqrt 2, 0)

/-- A point on the parabola -/
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

/-- An inscribed triangle of the parabola -/
structure InscribedTriangle where
  A : PointOnParabola
  B : PointOnParabola
  C : PointOnParabola

/-- The centroid of a triangle -/
noncomputable def centroid (t : InscribedTriangle) : ℝ × ℝ :=
  ((t.A.x + t.B.x + t.C.x) / 3, (t.A.y + t.B.y + t.C.y) / 3)

/-- The squared distance between two points -/
def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- The theorem to be proved -/
theorem inscribed_triangle_centroid_focus (t : InscribedTriangle) :
  centroid t = focus →
  distanceSquared (t.A.x, t.A.y) focus +
  distanceSquared (t.B.x, t.B.y) focus +
  distanceSquared (t.C.x, t.C.y) focus = 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_centroid_focus_l1267_126788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_l1267_126722

-- Define the function f(x) = ln x + 2x - 6
noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

-- Theorem statement
theorem solution_in_interval :
  ∃! x : ℝ, 2 < x ∧ x < 3 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_l1267_126722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_intersection_sum_zero_l1267_126761

/-- Represents the possible values in each cell of the grid -/
inductive CellValue
  | PosOne
  | NegOne
  | Zero

/-- Represents a grid with rows columns, where each cell contains a CellValue -/
def Grid (rows columns : ℕ) := Fin rows → Fin columns → CellValue

/-- Computes the sum of all values in the grid -/
def gridSum (grid : Grid m n) : ℤ :=
  (Finset.sum (Finset.univ : Finset (Fin m)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin n)) fun j =>
      match grid i j with
      | CellValue.PosOne => 1
      | CellValue.NegOne => -1
      | CellValue.Zero => 0)

/-- Checks if the sum of four cells at the intersection of two rows and two columns is zero -/
def fourCellSumZero (grid : Grid m n) (r1 r2 : Fin m) (c1 c2 : Fin n) : Prop :=
  let sum := match grid r1 c1, grid r1 c2, grid r2 c1, grid r2 c2 with
    | CellValue.PosOne, CellValue.PosOne, CellValue.NegOne, CellValue.NegOne => 0
    | CellValue.PosOne, CellValue.NegOne, CellValue.PosOne, CellValue.NegOne => 0
    | CellValue.PosOne, CellValue.NegOne, CellValue.NegOne, CellValue.PosOne => 0
    | CellValue.NegOne, CellValue.PosOne, CellValue.PosOne, CellValue.NegOne => 0
    | CellValue.NegOne, CellValue.PosOne, CellValue.NegOne, CellValue.PosOne => 0
    | CellValue.NegOne, CellValue.NegOne, CellValue.PosOne, CellValue.PosOne => 0
    | CellValue.Zero, CellValue.Zero, CellValue.Zero, CellValue.Zero => 0
    | _, _, _, _ => 1  -- Any other combination
  sum = 0

theorem grid_intersection_sum_zero
  (grid : Grid 1980 1981)
  (h_sum : gridSum grid = 0) :
  ∃ (r1 r2 : Fin 1980) (c1 c2 : Fin 1981),
    r1 ≠ r2 ∧ c1 ≠ c2 ∧ fourCellSumZero grid r1 r2 c1 c2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_intersection_sum_zero_l1267_126761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_equiv_domain_l1267_126775

noncomputable def f (x : ℝ) := Real.log (4 - x^2)

theorem domain_of_f :
  ∀ x : ℝ, x ∈ Set.Ioo (-2) 2 ↔ (4 - x^2) > 0 :=
by sorry

theorem equiv_domain :
  Set.Ioo (-2) 2 = {x : ℝ | f x ∈ Set.range Real.log} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_equiv_domain_l1267_126775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_FOH_l1267_126704

/-- Trapezoid EFGH with given properties -/
structure Trapezoid :=
  (EF : ℝ)
  (GH : ℝ)
  (area : ℝ)
  (EF_positive : EF > 0)
  (GH_positive : GH > 0)
  (area_positive : area > 0)
  (diagonals_perpendicular : Bool)

/-- The area of triangle FOH in the trapezoid -/
noncomputable def triangle_FOH_area (t : Trapezoid) : ℝ :=
  t.GH * (t.area / (t.EF + t.GH)) / 2

/-- Theorem: The area of triangle FOH is 108 square units -/
theorem area_of_triangle_FOH (t : Trapezoid) 
  (h1 : t.EF = 24)
  (h2 : t.GH = 36)
  (h3 : t.area = 360)
  (h4 : t.diagonals_perpendicular = true) :
  triangle_FOH_area t = 108 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_FOH_l1267_126704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_identification_l1267_126735

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation 3x(x-4) = 0 -/
noncomputable def eq_A (x : ℝ) : ℝ := 3 * x * (x - 4)

/-- The equation x^2 + y - 3 = 0 -/
noncomputable def eq_B (x y : ℝ) : ℝ := x^2 + y - 3

/-- The equation 1/x^2 + x = 2 -/
noncomputable def eq_C (x : ℝ) : ℝ := 1 / x^2 + x - 2

/-- The equation x^3 - 3x + 8 = 0 -/
noncomputable def eq_D (x : ℝ) : ℝ := x^3 - 3 * x + 8

theorem quadratic_equation_identification :
  is_quadratic_equation eq_A ∧
  ¬is_quadratic_equation (λ x ↦ eq_B x 0) ∧
  ¬is_quadratic_equation eq_C ∧
  ¬is_quadratic_equation eq_D :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_identification_l1267_126735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1267_126790

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.sin x, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, 2 * Real.sqrt 3 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧
    ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≤ 2 + Real.sqrt 3) ∧
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = 2 + Real.sqrt 3) ∧
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≥ 0) ∧
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1267_126790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_through_two_points_supplement_greater_than_complement_l1267_126710

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Define a line type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define an angle type
structure Angle where
  measure : ℝ

def acuteAngle (α : Angle) : Prop := 0 < α.measure ∧ α.measure < Real.pi/2

noncomputable def supplement (α : Angle) : Angle :=
  ⟨Real.pi - α.measure⟩

noncomputable def complement (α : Angle) : Angle :=
  ⟨Real.pi/2 - α.measure⟩

-- Define membership for Point in Line
instance : Membership Point Line where
  mem p l := l.a * p.x + l.b * p.y + l.c = 0

theorem unique_line_through_two_points (P Q : Point) (P_ne_Q : P ≠ Q) :
  ∃! L : Line, P ∈ L ∧ Q ∈ L :=
sorry

theorem supplement_greater_than_complement (α : Angle) (h : acuteAngle α) :
  (supplement α).measure > (complement α).measure :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_through_two_points_supplement_greater_than_complement_l1267_126710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curvilinearTrapezoidArea_eq_ln2_l1267_126759

/-- The area of the curvilinear trapezoid bounded by f(x) = 1/x, x = 1, x = 2, and y = 0 -/
noncomputable def curvilinearTrapezoidArea : ℝ := ∫ x in (1:ℝ)..2, 1/x

/-- Theorem stating that the area of the curvilinear trapezoid is ln 2 -/
theorem curvilinearTrapezoidArea_eq_ln2 : curvilinearTrapezoidArea = Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curvilinearTrapezoidArea_eq_ln2_l1267_126759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_interest_calculation_l1267_126796

/-- Calculate simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Convert years, months, and days to years -/
noncomputable def to_years (years : ℕ) (months : ℕ) (days : ℕ) : ℝ :=
  (years : ℝ) + ((months : ℝ) / 12) + ((days : ℝ) / 365)

theorem investment_interest_calculation :
  let principal : ℝ := 30000
  let rate : ℝ := 0.237
  let time : ℝ := to_years 5 8 12
  let interest : ℝ := simple_interest principal rate time
  ∃ ε > 0, |interest - 40524| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_interest_calculation_l1267_126796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_range_of_a_l1267_126760

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log x

noncomputable def g (x : ℝ) : ℝ := f (x + 1) - x

-- State the theorem for the maximum value of g
theorem max_value_of_g :
  ∃ (M : ℝ), M = 0 ∧ ∀ x, g x ≤ M :=
sorry

-- State the theorem for the range of a
theorem range_of_a :
  ∀ a : ℝ, (∀ x > 0, f x ≤ a * x ∧ a * x ≤ x^2 + 1) ↔ (1 / Real.exp 1 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_range_of_a_l1267_126760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_problem_solution_l1267_126791

/-- Represents a train with its speed in km/h -/
structure Train where
  speed : ℝ

/-- Represents the problem setup -/
structure TrainProblem where
  distance : ℝ
  express : Train
  slow : Train

/-- Time for trains to meet when traveling towards each other -/
noncomputable def time_to_meet (p : TrainProblem) : ℝ :=
  p.distance / (p.express.speed + p.slow.speed)

/-- Time for express train to catch up when traveling in the same direction -/
noncomputable def time_to_catch_up (p : TrainProblem) (head_start : ℝ) : ℝ :=
  (p.distance + p.slow.speed * head_start) / (p.express.speed - p.slow.speed)

theorem train_problem_solution (p : TrainProblem) (h : p.distance = 360 ∧ p.express.speed = 160 ∧ p.slow.speed = 80) :
  time_to_meet p = 1.5 ∧ time_to_catch_up p 0.5 = 5 := by
  sorry

-- Remove #eval statements as they are not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_problem_solution_l1267_126791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_intersection_equality_l1267_126768

-- Define the universal set U
def U : Finset Char := {'a', 'b', 'c', 'd', 'e'}

-- Define sets A, B, and C
def A : Finset Char := {'a', 'c', 'd'}
def B : Finset Char := {'a', 'b'}
def C : Finset Char := {'d', 'e'}

-- State the theorem
theorem complement_union_intersection_equality :
  (U \ A) ∪ (B ∩ C) = {'b', 'e'} :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_intersection_equality_l1267_126768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_short_bingo_first_column_possibilities_short_bingo_first_column_count_l1267_126703

theorem short_bingo_first_column_possibilities : Nat.factorial 15 / Nat.factorial 10 = 360360 := by
  -- Define the number of rows in the BINGO card
  let n : Nat := 5
  -- Define the range of numbers to choose from
  let range : Nat := 15
  
  -- Calculate the number of possibilities using factorial
  have h : Nat.factorial 15 / Nat.factorial 10 = 15 * 14 * 13 * 12 * 11 := by
    sorry -- Skip the proof
  
  -- Show that this equals 360360
  have h2 : 15 * 14 * 13 * 12 * 11 = 360360 := by
    sorry -- Skip the proof
  
  -- Combine the results
  rw [h, h2]

-- Additional theorem to state the result more directly
theorem short_bingo_first_column_count : 360360 = 360360 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_short_bingo_first_column_possibilities_short_bingo_first_column_count_l1267_126703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1267_126738

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 6 ≤ 0}
def B : Set ℝ := {x | x > 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ioc 2 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1267_126738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cylinder_surface_area_l1267_126723

-- Define the volume of the cylinder
noncomputable def cylinder_volume : ℝ := 16 * Real.pi

-- Theorem stating the minimum surface area of the cylinder
theorem min_cylinder_surface_area :
  ∀ (r h : ℝ), r > 0 → h > 0 →
  Real.pi * r^2 * h = cylinder_volume →
  2 * Real.pi * r^2 + 2 * Real.pi * r * h ≥ 24 * Real.pi := by
  sorry

#check min_cylinder_surface_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cylinder_surface_area_l1267_126723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_product_l1267_126702

def p (x : ℝ) : ℝ := 3 * x^3 + 4 * x^2 + 5 * x + 6
def q (x : ℝ) : ℝ := 7 * x^3 + 8 * x^2 + 9 * x + 10

theorem coefficient_x_cubed_in_product :
  ∃ (a b c : ℝ), (p * q) = (λ x => 148 * x^3 + a * x^2 + b * x + c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_product_l1267_126702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hare_wins_by_30_minutes_l1267_126777

/-- Race parameters --/
noncomputable def race_distance : ℝ := 1000
noncomputable def tortoise_speed : ℝ := 20
noncomputable def hare_speed : ℝ := 200
noncomputable def hare_stop_distance : ℝ := 400
noncomputable def hare_rest_time : ℝ := 15

/-- Calculate race times --/
noncomputable def tortoise_time : ℝ := race_distance / tortoise_speed

noncomputable def hare_time : ℝ :=
  (hare_stop_distance / hare_speed) +
  hare_rest_time +
  ((race_distance - hare_stop_distance) / hare_speed)

/-- Theorem statement --/
theorem hare_wins_by_30_minutes :
  tortoise_time - hare_time = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hare_wins_by_30_minutes_l1267_126777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_l1267_126779

/-- Given three vectors in a plane, prove that l² + m² = 12 under specific conditions. -/
theorem vector_equation (a b c : ℝ × ℝ) (l m : ℝ) : 
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- Angle between a and b is 90°
  (a.1^2 + a.2^2 = 1) →  -- |a| = 1
  (b.1^2 + b.2^2 = 1) →  -- |b| = 1
  (c.1^2 + c.2^2 = 12) →  -- |c| = 2√3
  (c = (l * a.1 + m * b.1, l * a.2 + m * b.2)) →  -- c = la + mb
  l^2 + m^2 = 12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_l1267_126779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_equation_l1267_126730

theorem exponent_equation (w : ℤ) : (3 : ℝ)^8 * (3 : ℝ)^w = 81 → w = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_equation_l1267_126730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_over_vector_l1267_126732

/-- Reflects a vector over another vector -/
noncomputable def reflect (v u : Fin 2 → ℝ) : Fin 2 → ℝ :=
  let p := ((v • u) / (u • u)) • u
  2 • p - v

theorem reflection_over_vector 
  (v : Fin 2 → ℝ) 
  (u : Fin 2 → ℝ) 
  (t : Fin 2 → ℝ) 
  (h1 : v 0 = 2 ∧ v 1 = -1)
  (h2 : u 0 = -2 ∧ u 1 = 1)
  (h3 : t 0 = 1 ∧ t 1 = 2) :
  reflect (v + t) u = fun i => if i = 0 then 1 else -3 := by
  sorry

#check reflection_over_vector

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_over_vector_l1267_126732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quadrilateral_pyramid_l1267_126750

theorem regular_quadrilateral_pyramid (m n : ℤ) :
  (4 * m * n)^2 + 2 * (m^2 - 2 * n^2)^2 = 2 * (m^2 + 2 * n^2)^2 := by
  -- Let's define x, y, and z
  let x := 4 * m * n
  let y := m^2 + 2 * n^2
  let z := m^2 - 2 * n^2
  
  -- Now we can prove the equation
  calc
    (4 * m * n)^2 + 2 * (m^2 - 2 * n^2)^2
    = x^2 + 2 * z^2 := by rfl
    _ = 2 * y^2 := by sorry
    _ = 2 * (m^2 + 2 * n^2)^2 := by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quadrilateral_pyramid_l1267_126750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_67_36_bar_l1267_126720

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to its rational representation. -/
def toRational (x : RepeatingDecimal) : ℚ :=
  x.integerPart + x.repeatingPart / (100 - 1 : ℚ)

/-- Rounds a rational number to the nearest hundredth. -/
def roundToHundredth (x : ℚ) : ℚ :=
  ⌊(x * 100 + 1/2)⌋ / 100

/-- The repeating decimal 67.36363636... -/
def number : RepeatingDecimal :=
  { integerPart := 67, repeatingPart := 36 }

theorem round_67_36_bar : roundToHundredth (toRational number) = 67.36 := by
  sorry

#eval roundToHundredth (toRational number)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_67_36_bar_l1267_126720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_theorem_l1267_126711

/-- Calculates the time it takes for a train to cross a signal pole given its length, 
    the platform length, and the time it takes to cross the platform. -/
noncomputable def time_to_cross_signal_pole (train_length platform_length time_to_cross_platform : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let train_speed := total_distance / time_to_cross_platform
  train_length / train_speed

/-- Theorem stating that a 300m train crossing a 200m platform in 30s 
    takes approximately 18s to cross a signal pole -/
theorem train_crossing_time_theorem :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ 
  |time_to_cross_signal_pole 300 200 30 - 18| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_theorem_l1267_126711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_diff_roots_quadratic_l1267_126778

theorem abs_diff_roots_quadratic : 
  let r₁ := (7 + Real.sqrt (7^2 - 4*1*12)) / (2*1)
  let r₂ := (7 - Real.sqrt (7^2 - 4*1*12)) / (2*1)
  |r₁ - r₂| = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_diff_roots_quadratic_l1267_126778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1267_126771

noncomputable def f (x : ℝ) := Real.log (2 * x - 1)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > 1/2} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1267_126771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1267_126784

/-- The number of days it takes y to complete the work -/
noncomputable def y_days : ℝ := 45

/-- The number of days it takes x and y together to complete the work -/
noncomputable def xy_days : ℝ := 11.25

/-- The number of days it takes x to complete the work -/
noncomputable def x_days : ℝ := 15

/-- The rate at which x completes the work -/
noncomputable def x_rate : ℝ := 1 / x_days

/-- The rate at which y completes the work -/
noncomputable def y_rate : ℝ := 1 / y_days

/-- The combined rate at which x and y complete the work -/
noncomputable def xy_rate : ℝ := 1 / xy_days

theorem work_completion_time :
  x_rate + y_rate = xy_rate :=
by
  -- Expand the definitions
  unfold x_rate y_rate xy_rate x_days y_days xy_days
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1267_126784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_radius_semiperimeter_ratio_l1267_126770

/-- Given an acute-angled triangle with semiperimeter p, circumradius R, inradius r, 
    and q as the semiperimeter of the triangle formed by the feet of the altitudes, 
    prove that R/r = p/q. -/
theorem triangle_radius_semiperimeter_ratio 
  (p R r q : ℝ) 
  (h_acute : IsAcuteTriangle) 
  (h_p : p > 0)
  (h_R : R > 0)
  (h_r : r > 0)
  (h_q : q > 0) : 
  R / r = p / q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_radius_semiperimeter_ratio_l1267_126770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_unique_coefficients_l1267_126701

/-- A quadratic function y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-value of a quadratic function at a given x -/
def QuadraticFunction.yValue (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- The x-coordinate of the vertex of a quadratic function -/
noncomputable def QuadraticFunction.vertexX (f : QuadraticFunction) : ℝ :=
  -f.b / (2 * f.a)

/-- The y-coordinate of the vertex of a quadratic function -/
noncomputable def QuadraticFunction.vertexY (f : QuadraticFunction) : ℝ :=
  f.yValue f.vertexX

/-- The discriminant of a quadratic function -/
def QuadraticFunction.discriminant (f : QuadraticFunction) : ℝ :=
  f.b^2 - 4 * f.a * f.c

theorem quadratic_function_unique_coefficients 
  (f : QuadraticFunction) 
  (h_max : f.vertexY = 10 ∧ f.vertexX = 3) 
  (h_intercept : Real.sqrt (f.discriminant / f.a^2) = 4) : 
  f.a = -5/2 ∧ f.b = 15 ∧ f.c = -25/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_unique_coefficients_l1267_126701
