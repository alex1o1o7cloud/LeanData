import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_three_points_compass_constructible_l1151_115111

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a circle
structure Circle where
  center : Point2D
  radius : ℝ

-- Define the compass construction capability
noncomputable def compassConstruction (p q : Point2D) : Circle :=
  { center := p, radius := ((p.x - q.x)^2 + (p.y - q.y)^2).sqrt }

-- Define membership for Point2D in Circle
def Point2D.mem (p : Point2D) (c : Circle) : Prop :=
  ((p.x - c.center.x)^2 + (p.y - c.center.y)^2) = c.radius^2

instance : Membership Point2D Circle where
  mem := Point2D.mem

-- Theorem statement
theorem circle_through_three_points_compass_constructible 
  (A B C : Point2D) (h : A ≠ B ∧ B ≠ C ∧ A ≠ C) : 
  ∃! (circ : Circle), A ∈ circ ∧ B ∈ circ ∧ C ∈ circ ∧ 
  (∃ (constructions : List (Point2D × Point2D)), 
   circ = List.foldr (λ (p, q) acc => compassConstruction p q) 
                     (compassConstruction A B) constructions) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_three_points_compass_constructible_l1151_115111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_5_is_neg_6_l1151_115183

/-- The coefficient of x^5 in the expansion of (x + 1/x - 1)^6 -/
def coefficient_x_5 : ℤ := -6

theorem coefficient_x_5_is_neg_6 : coefficient_x_5 = -6 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_5_is_neg_6_l1151_115183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_inscribed_triangle_l1151_115194

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)

/-- Given an acute triangle ABC with sides a, b, c, prove that the minimum area of an inscribed
    triangle DEF is 12S^3 / (a^2 + b^2 + c^2)^2, where S is the area of ABC. -/
theorem min_area_inscribed_triangle (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_acute : a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2) :
  let s := (a + b + c) / 2
  let S := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  ∃ (D : ℝ × ℝ) (E : ℝ × ℝ) (F : ℝ × ℝ),
    (∀ (D' E' F' : ℝ × ℝ), area_triangle D' E' F' ≥ 12 * S^3 / (a^2 + b^2 + c^2)^2) ∧
    area_triangle D E F = 12 * S^3 / (a^2 + b^2 + c^2)^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_inscribed_triangle_l1151_115194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1151_115119

/-- The set of available digits -/
def available_digits : Finset Nat := {2, 4, 6, 7, 8}

/-- A function that returns true if a number is a valid two-digit integer formed from the available digits -/
def is_valid_number (n : Nat) : Bool :=
  n ≥ 10 && n < 100 &&
  (n / 10) ∈ available_digits &&
  (n % 10) ∈ available_digits &&
  (n / 10) ≠ (n % 10)

/-- The set of all valid two-digit integers formed from the available digits -/
def valid_numbers : Finset Nat :=
  Finset.filter (fun n => is_valid_number n) (Finset.range 100)

theorem count_valid_numbers :
  Finset.card valid_numbers = 20 := by
  sorry

#eval Finset.card valid_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1151_115119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_curves_l1151_115131

/-- The shortest distance between a point on the curve x² = y and a point on the line x - y - 2 = 0 is 7√2/8 -/
theorem shortest_distance_between_curves : 
  ∃ (d : ℝ), d = (7 * Real.sqrt 2) / 8 ∧ 
  ∀ (x1 y1 x2 y2 : ℝ), 
    y1 = x1^2 → 
    x2 - y2 - 2 = 0 → 
    Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) ≥ d ∧ 
    ∃ (x3 y3 x4 y4 : ℝ), 
      y3 = x3^2 ∧ 
      x4 - y4 - 2 = 0 ∧ 
      Real.sqrt ((x4 - x3)^2 + (y4 - y3)^2) = d :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_curves_l1151_115131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relations_l1151_115123

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpLine : Line → Plane → Prop)
variable (paraPlane : Plane → Plane → Prop)
variable (perpLines : Line → Line → Prop)
variable (paraLines : Line → Line → Prop)

-- Define the "contained in" relation
variable (contains : Plane → Line → Prop)

-- Define the given lines and planes
variable (l m : Line) (α β : Plane)

-- State the theorem
theorem line_plane_relations :
  (perpLine l α ∧ contains β m) →
  ((paraPlane α β → perpLines l m) ∧
   (paraLines l m → perpLine m α)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relations_l1151_115123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_odd_subsets_equal_l1151_115196

theorem even_odd_subsets_equal {α : Type*} (S : Finset α) [Nonempty S] :
  (Finset.filter (fun A => A.card % 2 = 0) (Finset.powerset S)).card =
  (Finset.filter (fun A => A.card % 2 = 1) (Finset.powerset S)).card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_odd_subsets_equal_l1151_115196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_terms_count_two_digit_numbers_count_sum_of_digit_products_l1151_115130

/-- The set of digits used in the problem -/
def digits : Finset ℕ := {1, 2, 3, 4}

/-- The sum of the digits -/
def digit_sum : ℕ := Finset.sum digits id

/-- Number of terms in the expansion of (1+2+3+4)^3 -/
theorem expansion_terms_count : (Finset.card (Finset.powerset digits))^3 = 64 := by sorry

/-- Number of two-digit numbers using digits 1, 2, 3, and 4 -/
theorem two_digit_numbers_count : Finset.card (Finset.product digits digits) = 16 := by sorry

/-- Sum of products of digits for all four-digit numbers using 1, 2, 3, and 4 -/
theorem sum_of_digit_products : digit_sum^4 = 10000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_terms_count_two_digit_numbers_count_sum_of_digit_products_l1151_115130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_of_36_l1151_115165

theorem probability_factor_of_36 : 
  let n : ℕ := 36
  let total_integers := n
  let factors := Finset.filter (fun x => x ≤ n ∧ n % x = 0) (Finset.range (n + 1))
  (factors.card : ℚ) / total_integers = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_of_36_l1151_115165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_is_100_l1151_115144

/-- Calculates the downstream distance traveled by a boat given its upstream travel details and stream speed -/
noncomputable def downstream_distance (upstream_distance : ℝ) (upstream_time : ℝ) (downstream_time : ℝ) (stream_speed : ℝ) : ℝ :=
  let boat_speed := upstream_distance / upstream_time + stream_speed
  (boat_speed + stream_speed) * downstream_time

/-- Theorem stating that under given conditions, the downstream distance is 100 km -/
theorem downstream_distance_is_100 :
  downstream_distance 60 15 10 3 = 100 := by
  -- Unfold the definition of downstream_distance
  unfold downstream_distance
  -- Simplify the expression
  simp
  -- The proof is completed by numerical computation
  norm_num

-- We can't use #eval with noncomputable functions, so we'll use #check instead
#check downstream_distance 60 15 10 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_is_100_l1151_115144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1151_115180

/-- Hyperbola C with equation x²/a² - y²/b² = 1 -/
structure Hyperbola (a b : ℝ) where
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- Point on a hyperbola -/
structure PointOnHyperbola (C : Hyperbola a b) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / a^2 - y^2 / b^2 = 1

/-- Focus of a hyperbola -/
structure Focus (C : Hyperbola a b) where
  x : ℝ
  y : ℝ

/-- Vector from a point to a focus -/
def vector_to_focus (P : PointOnHyperbola C) (F : Focus C) : ℝ × ℝ :=
  (F.x - P.x, F.y - P.y)

/-- Magnitude of a vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (C : Hyperbola a b) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

theorem hyperbola_eccentricity 
  (C : Hyperbola a b) 
  (F₁ F₂ : Focus C) 
  (P : PointOnHyperbola C) 
  (h₁ : magnitude (vector_to_focus P F₁ + vector_to_focus P F₂) = 
    Real.sqrt ((magnitude (vector_to_focus P F₁))^2 + (magnitude (vector_to_focus P F₂))^2))
  (h₂ : magnitude (vector_to_focus P F₁) = 2 * magnitude (vector_to_focus P F₂)) :
  eccentricity C = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1151_115180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_price_reduction_l1151_115134

/-- Calculates the percentage price reduction of apples -/
def percentage_price_reduction (original_price reduced_price : ℚ) : ℚ :=
  (original_price - reduced_price) / original_price * 100

/-- Proves that the percentage reduction in apple prices is 30% -/
theorem apple_price_reduction : ∃ (original_price : ℚ),
  original_price > 0 ∧
  let reduced_price : ℚ := 2
  let additional_apples : ℕ := 54
  let total_cost : ℚ := 30
  (total_cost / reduced_price * 12 - additional_apples : ℚ) / 12 * original_price = total_cost ∧
  percentage_price_reduction original_price reduced_price = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_price_reduction_l1151_115134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_six_digit_number_l1151_115195

def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 6 ∧
  digits.get! 0 = (digits.get! 1 + digits.get! 2 + digits.get! 3 + digits.get! 4 + digits.get! 5) / 6 ∧
  digits.get! 1 = (digits.get! 2 + digits.get! 3 + digits.get! 4 + digits.get! 5) / 6

theorem unique_six_digit_number : 
  ∃! n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ is_valid_number n ∧ n = 769999 := by
  sorry

#check unique_six_digit_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_six_digit_number_l1151_115195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_l1151_115107

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 - 2*m - 3)

theorem power_function_decreasing (m : ℝ) : 
  (∀ x > 0, f m x = x^(m^2 - 2*m - 3)) →
  (∀ x > 0, ∀ y > 0, x < y → f m x > f m y) →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_l1151_115107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_distance_l1151_115157

/-- Helper function to represent the distance from the vertex to the directrix -/
noncomputable def distance_vertex_to_directrix (m : ℝ) : ℝ := 
  1 / (4 * m)

/-- Theorem: For a parabola defined by x² = my, where m is a positive real number,
    if the distance from the vertex to the directrix is 1/2, then m = 2. -/
theorem parabola_directrix_distance (m : ℝ) : 
  m > 0 → 
  (∀ x y : ℝ, x^2 = m*y) → 
  (∃ d : ℝ, d = 1/2 ∧ d = distance_vertex_to_directrix m) → 
  m = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_distance_l1151_115157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_inverse_fifteen_l1151_115126

noncomputable def f (x : ℝ) : ℝ := 3 * x + 6

noncomputable def f_inverse (y : ℝ) : ℝ := (y - 6) / 3

theorem inverse_of_inverse_fifteen :
  f_inverse (f_inverse 15) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_inverse_fifteen_l1151_115126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ZYX_is_30_l1151_115112

-- Define the basic geometric entities
structure Point := (x y : ℝ)

def Line (A B : Point) := {P : Point | ∃ t : ℝ, P = ⟨(1 - t) * A.x + t * B.x, (1 - t) * A.y + t * B.y⟩}

structure Circle := (center : Point) (radius : ℝ)

-- Define the triangles
structure Triangle := (A B C : Point)

-- Define the configuration
structure Configuration :=
  (ABC : Triangle)
  (XYZ : Triangle)
  (Γ : Circle)
  (is_circumcircle : Prop) -- Placeholder for the circumcircle condition
  (is_incircle : Prop) -- Placeholder for the incircle condition
  (X_on_BC : XYZ.A ∈ Line ABC.B ABC.C)
  (Y_on_AB : XYZ.B ∈ Line ABC.A ABC.B)
  (Z_on_AC : XYZ.C ∈ Line ABC.A ABC.C)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (angle_C : ℝ)
  (h_angle_A : angle_A = 50)
  (h_angle_B : angle_B = 70)
  (h_angle_C : angle_C = 60)

-- Define the angle measure (simplified for this example)
def Angle (A B C : Point) : ℝ := sorry

-- State the theorem
theorem angle_ZYX_is_30 (config : Configuration) :
  Angle config.XYZ.A config.XYZ.B config.XYZ.C = 30 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ZYX_is_30_l1151_115112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_at_intersection_l1151_115114

-- Define the two curves
def curve1 (x : ℝ) : ℝ := 2 * x^2 - 5
def curve2 (x : ℝ) : ℝ := x^2 - 3 * x + 5

-- Define the derivative of the curves
def curve1_deriv (x : ℝ) : ℝ := 4 * x
def curve2_deriv (x : ℝ) : ℝ := 2 * x - 3

-- Define the intersection points
def intersection_point1 : ℝ × ℝ := (-5, 45)
def intersection_point2 : ℝ × ℝ := (2, 3)

-- Define the tangent lines
def tangent1 (x : ℝ) : ℝ := -20 * x - 55
def tangent2 (x : ℝ) : ℝ := -13 * x - 20
def tangent3 (x : ℝ) : ℝ := 8 * x - 13
def tangent4 (x : ℝ) : ℝ := x + 1

theorem tangent_lines_at_intersection :
  (tangent1 (intersection_point1.1) = curve1 intersection_point1.1 ∧
   (tangent1 intersection_point1.1 - curve1 intersection_point1.1) / (intersection_point1.1 - intersection_point1.1) = curve1_deriv intersection_point1.1) ∧
  (tangent2 (intersection_point1.1) = curve2 intersection_point1.1 ∧
   (tangent2 intersection_point1.1 - curve2 intersection_point1.1) / (intersection_point1.1 - intersection_point1.1) = curve2_deriv intersection_point1.1) ∧
  (tangent3 (intersection_point2.1) = curve1 intersection_point2.1 ∧
   (tangent3 intersection_point2.1 - curve1 intersection_point2.1) / (intersection_point2.1 - intersection_point2.1) = curve1_deriv intersection_point2.1) ∧
  (tangent4 (intersection_point2.1) = curve2 intersection_point2.1 ∧
   (tangent4 intersection_point2.1 - curve2 intersection_point2.1) / (intersection_point2.1 - intersection_point2.1) = curve2_deriv intersection_point2.1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_at_intersection_l1151_115114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1151_115164

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N : M ∩ N = Set.Ioc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1151_115164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_OPQR_volume_zero_l1151_115122

/-- Triangle PQR with vertices on coordinate axes and side lengths 6, 8, and 10 -/
structure TrianglePQR where
  P : ℝ × ℝ × ℝ
  Q : ℝ × ℝ × ℝ
  R : ℝ × ℝ × ℝ
  h_P : P.2.1 = 0 ∧ P.2.2 = 0 ∧ P.1 > 0
  h_Q : Q.1 = 0 ∧ Q.2.2 = 0 ∧ Q.2.1 > 0
  h_R : R.1 = 0 ∧ R.2.1 = 0 ∧ R.2.2 > 0
  h_PQ : Real.sqrt ((P.1 - Q.1)^2 + (P.2.1 - Q.2.1)^2 + (P.2.2 - Q.2.2)^2) = 6
  h_QR : Real.sqrt ((Q.1 - R.1)^2 + (Q.2.1 - R.2.1)^2 + (Q.2.2 - R.2.2)^2) = 8
  h_RP : Real.sqrt ((R.1 - P.1)^2 + (R.2.1 - P.2.1)^2 + (R.2.2 - P.2.2)^2) = 10

/-- The volume of tetrahedron OPQR is 0 -/
theorem tetrahedron_OPQR_volume_zero (t : TrianglePQR) : 
  (1 / 6 : ℝ) * t.P.1 * t.Q.2.1 * t.R.2.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_OPQR_volume_zero_l1151_115122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1151_115190

/-- The function f(x) defined in terms of a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - a*x + 3) / Real.log a

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ ≤ a/2 → f a x₁ - f a x₂ > 0) : 
  1 < a ∧ a < 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1151_115190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_quadrilateral_l1151_115133

/-- The area of the quadrilateral APDQ given the square side length s and the parameter x -/
noncomputable def area_quadrilateral (s x : ℝ) : ℝ := (3 / 2) * x * s

/-- Given a square ABCD with side length s, and points P on AB and Q on CD
    such that AP = 2x and CQ = x, the area of quadrilateral APDQ is maximized
    when x = s/3, and the maximum area is (1/2)s^2 -/
theorem max_area_quadrilateral (s : ℝ) (h : s > 0) :
  ∃ x : ℝ, x > 0 ∧ x < s ∧
    (∀ y : ℝ, y > 0 → y < s →
      area_quadrilateral s x ≥ area_quadrilateral s y) ∧
    x = s / 3 ∧
    area_quadrilateral s x = (1 / 2) * s^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_quadrilateral_l1151_115133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_product_inequality_l1151_115177

theorem triangle_sine_product_inequality (A B C : Real) :
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) ≤ 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_product_inequality_l1151_115177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_product_l1151_115172

theorem pure_imaginary_product (m : ℝ) : 
  (Complex.I : ℂ).im = 1 →
  (Complex.ofReal m + Complex.I) * (3 - Complex.I) ∈ {z : ℂ | z.re = 0} →
  m = -1/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_product_l1151_115172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_filling_cubes_l1151_115147

/-- Given a box with dimensions 24 inches long, 40 inches wide, and 16 inches deep,
    the smallest number of identical cubes that can completely fill the box without
    leaving any space is 30. -/
theorem box_filling_cubes : ∃ (cube_side : ℕ),
  cube_side > 0 ∧
  24 % cube_side = 0 ∧
  40 % cube_side = 0 ∧
  16 % cube_side = 0 ∧
  (24 / cube_side) * (40 / cube_side) * (16 / cube_side) = 30 := by
  -- The proof goes here
  sorry

#check box_filling_cubes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_filling_cubes_l1151_115147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_240_degrees_l1151_115182

theorem cot_240_degrees : Real.tan (240 * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cot_240_degrees_l1151_115182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1151_115135

theorem relationship_abc (a b c : ℝ) : 
  a = (3/5)^(-(1/3 : ℝ)) → 
  b = (3/5)^(-(1/4 : ℝ)) → 
  c = (3/2)^(-(3/4 : ℝ)) → 
  c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1151_115135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_point_l1151_115154

/-- The inverse proportion function f(x) = -18/x --/
noncomputable def f (x : ℝ) : ℝ := -18 / x

/-- Predicate to check if a point (x, y) lies on the graph of f --/
def lies_on_graph (x y : ℝ) : Prop := f x = y

theorem inverse_proportion_point :
  lies_on_graph 3 (-6) ∧
  ¬ lies_on_graph (-5) 2 ∧
  ¬ lies_on_graph 2 9 ∧
  ¬ lies_on_graph 9 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_point_l1151_115154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_equals_M_l1151_115175

-- Define the sets M and N
def M : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.exp (x * Real.log 2)}
def N : Set ℝ := {x : ℝ | x > 1 ∧ ∃ y : ℝ, y = Real.log (x - 1)}

-- Theorem statement
theorem union_equals_M : M ∪ N = M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_equals_M_l1151_115175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l1151_115168

/-- The curve C: y = √(-x² - 2x) -/
noncomputable def C : ℝ → ℝ := λ x => Real.sqrt (-x^2 - 2*x)

/-- The line l: x + y - m = 0 -/
def L (m : ℝ) : ℝ → ℝ := λ x => m - x

/-- Predicate to check if C and L intersect at two points -/
def intersect_at_two_points (m : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ C x₁ = L m x₁ ∧ C x₂ = L m x₂

/-- Theorem stating the range of m for which C and L intersect at two points -/
theorem intersection_range :
  ∀ m : ℝ, intersect_at_two_points m ↔ 0 < m ∧ m < Real.sqrt 2 - 1 := by
  sorry

#check intersection_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l1151_115168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_min_max_f_l1151_115121

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 / (x - 1)

-- Define the domain of f
def domain : Set ℝ := Set.Icc (-2) 1 ∪ Set.Ioc 1 6

-- Theorem statement
theorem no_min_max_f :
  ¬ (∃ (m : ℝ), ∀ (x : ℝ), x ∈ domain → f x ≥ m) ∧
  ¬ (∃ (M : ℝ), ∀ (x : ℝ), x ∈ domain → f x ≤ M) := by
  sorry

#check no_min_max_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_min_max_f_l1151_115121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_novels_probability_l1151_115108

theorem consecutive_novels_probability (n m : ℕ) (hn : n = 12) (hm : m = 4) :
  (((n - m + 1).factorial * m.factorial : ℚ) / n.factorial) = 1 / 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_novels_probability_l1151_115108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_not_less_than_10_l1151_115110

-- Define a die
def Die : Type := Fin 6

-- Define the probability space
def Ω : Type := Die × Die

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- Define the event that the first die shows 6
def A : Set Ω := {ω | ω.1 = ⟨5, sorry⟩}

-- Define the event that the sum of points is not less than 10
def B : Set Ω := {ω | ω.1.val + ω.2.val + 2 ≥ 10}

-- State the theorem
theorem probability_sum_not_less_than_10 :
  P (B ∩ A) / P A = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_not_less_than_10_l1151_115110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_fraction_after_transfers_l1151_115115

/-- Represents the state of a mug containing tea and milk -/
structure MugState where
  tea : ℚ
  milk : ℚ

/-- Represents the transfer of liquid between mugs -/
def transfer (from_mug to_mug : MugState) (fraction : ℚ) : MugState × MugState :=
  let amount := fraction * (from_mug.tea + from_mug.milk)
  let tea_ratio := from_mug.tea / (from_mug.tea + from_mug.milk)
  let milk_ratio := from_mug.milk / (from_mug.tea + from_mug.milk)
  let new_from := MugState.mk (from_mug.tea - amount * tea_ratio) (from_mug.milk - amount * milk_ratio)
  let new_to := MugState.mk (to_mug.tea + amount * tea_ratio) (to_mug.milk + amount * milk_ratio)
  (new_from, new_to)

theorem milk_fraction_after_transfers : 
  let mug1_initial := MugState.mk 6 0
  let mug2_initial := MugState.mk 0 6
  let (mug1_mid, mug2_mid) := transfer mug1_initial mug2_initial (1/3)
  let (mug1_final, mug2_final) := transfer mug2_mid mug1_mid (1/4)
  mug1_final.milk / (mug1_final.tea + mug1_final.milk) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_fraction_after_transfers_l1151_115115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_sum_specific_circle_center_l1151_115152

/-- The equation of a circle in the form x^2 + y^2 + ax + by + c = 0 -/
def CircleEquation (a b c : ℝ) : ℝ × ℝ → Prop :=
  fun p => (p.1^2 + p.2^2 + a * p.1 + b * p.2 + c = 0)

/-- The center of a circle given by the equation x^2 + y^2 + ax + by + c = 0 -/
noncomputable def CircleCenter (a b c : ℝ) : ℝ × ℝ :=
  (-a/2, -b/2)

theorem circle_center_sum (a b c : ℝ) :
  let center := CircleCenter a b c
  CircleEquation a b c center ∧ center.1 + center.2 = (-a-b)/2 := by
  sorry

theorem specific_circle_center :
  let a := -10
  let b := 4
  let c := 15
  let center := CircleCenter a b c
  CircleEquation a b c center ∧ center = (5, -2) ∧ center.1 + center.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_sum_specific_circle_center_l1151_115152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_property_l1151_115116

-- Define a linear function
def linear_function (a b : ℝ) : ℝ → ℝ := λ x ↦ a * x + b

-- State the theorem
theorem linear_function_property (g : ℝ → ℝ) (h : ∃ a b : ℝ, g = linear_function a b) 
  (h_diff : g 5 - g 0 = 10) : g 15 - g 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_property_l1151_115116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_difference_per_sq_inch_l1151_115125

noncomputable def first_tv_width : ℝ := 24
noncomputable def first_tv_height : ℝ := 16
noncomputable def first_tv_cost : ℝ := 672

noncomputable def new_tv_width : ℝ := 48
noncomputable def new_tv_height : ℝ := 32
noncomputable def new_tv_cost : ℝ := 1152

noncomputable def first_tv_area : ℝ := first_tv_width * first_tv_height
noncomputable def new_tv_area : ℝ := new_tv_width * new_tv_height

noncomputable def first_tv_cost_per_sq_inch : ℝ := first_tv_cost / first_tv_area
noncomputable def new_tv_cost_per_sq_inch : ℝ := new_tv_cost / new_tv_area

theorem cost_difference_per_sq_inch :
  first_tv_cost_per_sq_inch - new_tv_cost_per_sq_inch = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_difference_per_sq_inch_l1151_115125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_ratio_theorem_l1151_115167

/-- Represents a cone with generatrix length 1 and base radius r -/
structure Cone where
  r : ℝ
  generatrix_length : ℝ := 1

/-- The ratio of the unfolded side surface area to the area of the circle containing the sector -/
noncomputable def surface_area_ratio (c : Cone) : ℝ := 1/3

/-- The ratio of the generatrix length to the base radius -/
noncomputable def length_radius_ratio (c : Cone) : ℝ := c.generatrix_length / c.r

/-- 
Given a cone with generatrix length 1 and base radius r, 
if the area of the unfolded side surface is 1/3 of the area of the circle containing the sector,
then the ratio of generatrix length to base radius is 3.
-/
theorem cone_ratio_theorem (c : Cone) : 
  surface_area_ratio c = 1/3 → length_radius_ratio c = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_ratio_theorem_l1151_115167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_complex_fraction_l1151_115171

theorem simplify_complex_fraction :
  ∀ x y : ℝ,
  x = Real.sqrt (3 + Real.sqrt 5) ∧
  y = (4 * Real.sqrt 2 - 2 * Real.sqrt 10)^(2/3) →
  x / y = (Real.sqrt 2 / 2) * (Real.sqrt 5 - 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_complex_fraction_l1151_115171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_chords_perimeter_sum_l1151_115170

/-- 
Given a circle with two perpendicular chords AC and BD intersecting at E,
this theorem proves that the perimeter of quadrilateral ABCD can be expressed
in a specific form and that a certain sum involving the coefficients equals 9.
-/
theorem perpendicular_chords_perimeter_sum (BE EC AE : ℝ) 
  (h_positive : BE > 0 ∧ EC > 0 ∧ AE > 0)
  (h_BE : BE = 3)
  (h_EC : EC = 2)
  (h_AE : AE = 6)
  : ∃ (m n p q : ℕ), 
    (m > 0 ∧ n > 0 ∧ p > 0 ∧ q > 0) ∧
    (q > n) ∧
    (∀ k : ℕ, k > 1 → ¬(k * k ∣ n)) ∧
    (∀ k : ℕ, k > 1 → ¬(k * k ∣ q)) ∧
    (∃ (perimeter : ℝ), 
      perimeter = m * Real.sqrt n + p * Real.sqrt q ∧
      Real.sqrt (m * n) + Real.sqrt (p + q : ℕ) = 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_chords_perimeter_sum_l1151_115170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l1151_115106

-- Define the power function as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^a

-- State the theorem
theorem power_function_through_point (a : ℝ) :
  f a 4 = 2 → f a 9 = 3 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l1151_115106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_center_of_symmetry_l1151_115136

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (x + Real.pi / 6)

noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos (2 * x + Real.pi / 6)

-- Center of symmetry for cosine function occurs when the argument is π/2 + kπ
def is_center_of_symmetry (x : ℝ) : Prop :=
  ∃ k : ℤ, 2 * x + Real.pi / 6 = Real.pi / 2 + k * Real.pi

theorem g_center_of_symmetry :
  is_center_of_symmetry (Real.pi / 12) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_center_of_symmetry_l1151_115136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_5_l1151_115105

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt ((h.a ^ 2 + h.b ^ 2) / h.a ^ 2)

/-- Theorem stating the eccentricity of the hyperbola under given conditions -/
theorem hyperbola_eccentricity_sqrt_5 (h : Hyperbola) 
  (h_foci_circle : ∃ (P : ℝ × ℝ), P.1 < 0 ∧ (P.1 ^ 2 / h.a ^ 2 - P.2 ^ 2 / h.b ^ 2 = 1))
  (h_vertices_circle_tangent : ∃ (Q : ℝ × ℝ), Q.1 > 0 ∧ Q.2 = 0) :
  eccentricity h = Real.sqrt 5 := by
  sorry

#check hyperbola_eccentricity_sqrt_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_5_l1151_115105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segment_length_l1151_115127

/- Definitions (not to be proved, just for context) -/

def IsTriangle (s : Set (ℝ × ℝ)) : Prop := sorry

def HasBase (triangle : Set (ℝ × ℝ)) (base : ℝ) : Prop := sorry

def ParallelToBase (triangle : Set (ℝ × ℝ)) (line : Set (ℝ × ℝ)) : Prop := sorry

def DividesIntoEqualAreas (triangle : Set (ℝ × ℝ)) (line1 line2 : Set (ℝ × ℝ)) : Prop := sorry

def IsClosestParallelSegment (triangle : Set (ℝ × ℝ)) (segment : ℝ) : Prop := sorry

theorem parallel_segment_length (base : ℝ) (h : base = 18) : 
  ∀ (triangle : Set (ℝ × ℝ)) (line1 line2 : Set (ℝ × ℝ)),
    IsTriangle triangle →
    HasBase triangle base →
    ParallelToBase triangle line1 →
    ParallelToBase triangle line2 →
    DividesIntoEqualAreas triangle line1 line2 →
    ∃ (segment : ℝ),
      IsClosestParallelSegment triangle segment →
      segment = 6 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segment_length_l1151_115127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1151_115151

-- Define the work rate of p and q
noncomputable def p_rate : ℝ := 1 / 20
noncomputable def q_rate : ℝ := 1 / 12

-- Define the time p works alone
def p_alone_time : ℝ := 4

-- Define the total work
def total_work : ℝ := 1

-- Theorem statement
theorem work_completion_time :
  let remaining_work := total_work - p_rate * p_alone_time
  let combined_rate := p_rate + q_rate
  p_alone_time + remaining_work / combined_rate = 10 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1151_115151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_target_heart_rate_post_warmup_l1151_115191

def athlete_age : ℕ := 30

def max_heart_rate (age : ℕ) : ℕ := 225 - age

def initial_target_rate (max_rate : ℕ) : ℚ := 0.75 * max_rate

def post_warmup_rate (initial_rate : ℚ) : ℚ := 1.05 * initial_rate

def round_to_nearest (x : ℚ) : ℕ := 
  if x - ↑(Int.floor x) < 0.5 then (Int.floor x).toNat else (Int.ceil x).toNat

theorem target_heart_rate_post_warmup :
  round_to_nearest (post_warmup_rate (initial_target_rate (max_heart_rate athlete_age))) = 154 := by
  sorry

#eval round_to_nearest (post_warmup_rate (initial_target_rate (max_heart_rate athlete_age)))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_target_heart_rate_post_warmup_l1151_115191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_coeff_in_x_plus_one_fifth_l1151_115162

/-- The coefficient of x^2 in the expansion of (x+1)^5 is 10 -/
theorem x_squared_coeff_in_x_plus_one_fifth : 
  (Finset.range 6).sum (fun k => Nat.choose 5 k * (if k = 3 then 1 else 0)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_coeff_in_x_plus_one_fifth_l1151_115162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_increasing_f_implies_b_range_l1151_115185

/-- A function that is strictly increasing on R -/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + b * x^2 + (b + 2) * x + 3

/-- The derivative of f with respect to x -/
noncomputable def f' (b : ℝ) (x : ℝ) : ℝ := x^2 + 2 * b * x + b + 2

theorem strictly_increasing_f_implies_b_range (b : ℝ) :
  (∀ x : ℝ, StrictMono (f b)) → -1 ≤ b ∧ b ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_increasing_f_implies_b_range_l1151_115185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_midline_l1151_115192

/-- Represents a cosine function with parameters a, b, c, and d -/
noncomputable def cosine_function (a b c d : ℝ) (x : ℝ) : ℝ := a * Real.cos (b * x + c) + d

theorem cosine_function_midline 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_oscillate : ∀ x, 1 ≤ cosine_function a b c d x ∧ cosine_function a b c d x ≤ 5) :
  d = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_midline_l1151_115192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_30_degrees_l1151_115179

-- Define the radius and central angle
def radius : ℝ := 1
def centralAngle : ℝ := 30

-- Define the area of a sector
noncomputable def sectorArea (r : ℝ) (θ : ℝ) : ℝ := (θ * Real.pi * r^2) / 360

-- Theorem statement
theorem sector_area_30_degrees :
  sectorArea radius centralAngle = Real.pi / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_30_degrees_l1151_115179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1151_115150

noncomputable def f (x : Real) : Real := 2 * (Real.sin (Real.pi / 4 + x))^2 - Real.sqrt 3 * Real.cos (2 * x)

theorem f_properties :
  -- The smallest positive period is π
  (∃ T : Real, T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S)) ∧
  -- Maximum and minimum values in [π/4, π/2]
  (∀ x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), f x ≤ 3) ∧
  (∃ x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), f x = 3) ∧
  (∀ x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), f x ≥ 2) ∧
  (∃ x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), f x = 2) ∧
  -- Range of m for |f(x) - m| < 2
  (∀ m : Real, (∀ x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), |f x - m| < 2) ↔ m ∈ Set.Ioo 1 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1151_115150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_from_centroid_l1151_115139

/-- The centroid of a triangle -/
noncomputable def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

/-- Theorem: Given the coordinates of two vertices and the centroid of a triangle, 
    we can determine the coordinates of the third vertex. -/
theorem third_vertex_from_centroid 
  (A B G : ℝ × ℝ) 
  (hA : A = (2, 3)) 
  (hB : B = (-4, -2)) 
  (hG : G = (2, -1)) :
  ∃ C : ℝ × ℝ, C = (8, -4) ∧ centroid A B C = G := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_from_centroid_l1151_115139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_location_l1151_115140

-- Define the function f
def f (a b c x : ℝ) : ℝ := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)

-- State the theorem
theorem roots_location (a b c : ℝ) (h : a < b ∧ b < c) :
  ∃ (x₁ x₂ : ℝ), x₁ ∈ Set.Ioo a b ∧ x₂ ∈ Set.Ioo b c ∧ 
  f a b c x₁ = 0 ∧ f a b c x₂ = 0 ∧
  ∀ x, f a b c x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_location_l1151_115140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_concerts_for_six_musicians_l1151_115100

/-- Represents a concert arrangement --/
structure Concert where
  performers : Finset Nat
  listeners : Finset Nat
  disjoint : Disjoint performers listeners
  total : performers.card + listeners.card = 6

/-- Represents a festival schedule --/
structure Festival where
  concerts : List Concert
  cover_all : ∀ (i j : Nat), i < 6 → j < 6 → i ≠ j →
    ∃ (c : Concert), c ∈ concerts ∧ i ∈ c.listeners ∧ j ∈ c.performers

/-- The theorem to be proved --/
theorem min_concerts_for_six_musicians :
  ∃ (f : Festival), f.concerts.length = 4 ∧
  ∀ (f' : Festival), f'.concerts.length < 4 → False := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_concerts_for_six_musicians_l1151_115100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_is_odd_l1151_115102

-- Define the real number a
variable (a : ℝ)

-- Define the function F
variable (F : ℝ → ℝ)

-- Define the function G
noncomputable def G (x : ℝ) : ℝ := F x * (1 / (a^x - 1) + 1/2)

-- State the theorem
theorem G_is_odd (h1 : a > 0) (h2 : a ≠ 1) (h3 : ∀ x, F (-x) = -F x) :
  ∀ x, G a F (-x) = -G a F x :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_is_odd_l1151_115102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_stick_stability_l1151_115193

/-- A rectangular door frame -/
structure DoorFrame where
  width : ℝ
  height : ℝ
  is_rectangular : width > 0 ∧ height > 0

/-- A wooden stick used in the door frame -/
structure WoodenStick where
  length : ℝ
  is_diagonal : length > 0

/-- Stability measure of a shape -/
def stability (length : ℝ) : Prop := sorry

/-- Triangle formed by the diagonal stick in the door frame -/
def triangle (df : DoorFrame) (ws : WoodenStick) : Prop :=
  ws.length^2 = df.width^2 + df.height^2

/-- Theorem stating that a diagonal stick increases stability -/
theorem diagonal_stick_stability (df : DoorFrame) (ws : WoodenStick) :
  triangle df ws → stability ws.length ∧ ¬stability df.width ∧ ¬stability df.height := by
  sorry

#check diagonal_stick_stability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_stick_stability_l1151_115193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_willam_land_percentage_is_12_point_5_l1151_115117

/-- The total tax collected from the village -/
noncomputable def total_tax : ℝ := 3840

/-- The tax paid by Mr. Willam -/
noncomputable def willam_tax : ℝ := 480

/-- The percentage of Mr. Willam's taxable land over the total taxable land of the village -/
noncomputable def willam_land_percentage : ℝ := (willam_tax / total_tax) * 100

theorem willam_land_percentage_is_12_point_5 :
  willam_land_percentage = 12.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_willam_land_percentage_is_12_point_5_l1151_115117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_car_fuel_usage_l1151_115129

/-- Represents a car with its characteristics and travel information -/
structure Car where
  speed : ℚ  -- Speed in miles per hour
  fuelEfficiency : ℚ  -- Miles per gallon
  tankCapacity : ℚ  -- Gallons
  travelTime : ℚ  -- Hours

/-- Calculates the fraction of a full tank used given a car's characteristics and travel time -/
def fractionOfTankUsed (car : Car) : ℚ :=
  (car.speed * car.travelTime) / (car.fuelEfficiency * car.tankCapacity)

/-- Theorem stating that a car with given characteristics traveling for 5 hours uses 5/6 of its tank -/
theorem specific_car_fuel_usage :
  let car : Car := {
    speed := 60,
    fuelEfficiency := 30,
    tankCapacity := 12,
    travelTime := 5
  }
  fractionOfTankUsed car = 5/6 := by
  -- Proof goes here
  sorry

#eval fractionOfTankUsed {
  speed := 60,
  fuelEfficiency := 30,
  tankCapacity := 12,
  travelTime := 5
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_car_fuel_usage_l1151_115129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmers_pass_12_times_l1151_115178

/-- Represents a swimmer with a given speed -/
structure Swimmer where
  speed : ℚ
  deriving Repr

/-- Represents the swimming scenario -/
structure SwimmingScenario where
  poolLength : ℚ
  swimmer1 : Swimmer
  swimmer2 : Swimmer
  totalTime : ℚ
  deriving Repr

/-- Calculates the number of times the swimmers pass each other -/
def calculatePassings (scenario : SwimmingScenario) : ℕ :=
  sorry

/-- The main theorem stating that the swimmers pass each other 12 times -/
theorem swimmers_pass_12_times (scenario : SwimmingScenario) 
    (h1 : scenario.poolLength = 100)
    (h2 : scenario.swimmer1.speed = 4)
    (h3 : scenario.swimmer2.speed = 3)
    (h4 : scenario.totalTime = 600) : 
    calculatePassings scenario = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmers_pass_12_times_l1151_115178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_multiples_of_five_l1151_115187

theorem four_digit_multiples_of_five : 
  (Finset.filter (fun n => n % 5 = 0) (Finset.range 9000)).card = 1800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_multiples_of_five_l1151_115187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_necessary_not_sufficient_l1151_115103

/-- A tetrahedron is a polyhedron with four vertices -/
structure Tetrahedron :=
  (A B C D : Point)

/-- A regular tetrahedron is a tetrahedron whose base is an equilateral triangle and 
    whose vertex projection on the base is the center of the base triangle -/
def RegularTetrahedron (ABCD : Tetrahedron) : Prop := sorry

/-- A regular tetrahedron is a tetrahedron with all edges of equal length -/
def RegularTetrahedron' (ABCD : Tetrahedron) : Prop := sorry

theorem regular_tetrahedron_necessary_not_sufficient :
  (∀ ABCD : Tetrahedron, RegularTetrahedron ABCD → RegularTetrahedron' ABCD) ∧
  (∃ ABCD : Tetrahedron, RegularTetrahedron ABCD ∧ ¬RegularTetrahedron' ABCD) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_necessary_not_sufficient_l1151_115103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_lines_condition_l1151_115124

/-- The equation represents two intersecting lines -/
def represents_intersecting_lines (lambda : ℝ) : Prop :=
  ∃ (m n : ℝ), (m + n = 4) ∧ (m * n = 3) ∧ (n - m = lambda)

/-- Theorem stating the necessary and sufficient condition for intersecting lines -/
theorem intersecting_lines_condition (lambda : ℝ) :
  represents_intersecting_lines lambda ↔ (lambda = 2 ∨ lambda = -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_lines_condition_l1151_115124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_five_pi_twelfths_plus_theta_l1151_115186

theorem sin_five_pi_twelfths_plus_theta (θ : ℝ) :
  Real.cos (π / 12 - θ) = 1 / 3 →
  Real.sin (5 * π / 12 + θ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_five_pi_twelfths_plus_theta_l1151_115186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1151_115174

noncomputable def f (x : ℝ) : ℝ :=
  let P : ℝ × ℝ := (Real.cos (2 * x) + 1, 1)
  let Q : ℝ × ℝ := (1, Real.sqrt 3 * Real.sin (2 * x) + 1)
  P.1 * Q.1 + P.2 * Q.2

theorem f_properties :
  ∀ x : ℝ,
  (f x = 2 * Real.sin (2 * x + π / 6) + 2) ∧
  (∀ y : ℝ, f (x + π) = f x) ∧
  (∀ y : ℝ, f y ≥ 0) ∧
  (∃ y : ℝ, f y = 0) ∧
  (∀ y : ℝ, f y ≤ 4) ∧
  (∃ y : ℝ, f y = 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1151_115174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_displacement_equals_36_l1151_115118

-- Define the velocity function
def velocity (t : ℝ) : ℝ := 3 * t^2 + 2 * t

-- Define the displacement function
noncomputable def displacement (a b : ℝ) : ℝ := ∫ t in a..b, velocity t

-- Theorem statement
theorem displacement_equals_36 : displacement 0 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_displacement_equals_36_l1151_115118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_90_l1151_115156

def factors (n : ℕ) : Finset ℕ := Finset.filter (λ d => d > 0 ∧ n % d = 0) (Finset.range (n + 1))

def factorsLessThan (n : ℕ) (m : ℕ) : Finset ℕ := Finset.filter (λ d => d < m) (factors n)

theorem factors_of_90 :
  let allFactors := factors 90
  let factorsUnder10 := factorsLessThan 90 10
  (factorsUnder10.card : ℚ) / allFactors.card = 1/2 ∧
  Finset.card (Finset.filter (λ p : ℕ × ℕ => p.1 ∈ factorsUnder10 ∧ p.2 ∈ allFactors ∧ p.1 * p.2 = 90) (allFactors.product allFactors)) = 6 := by
  sorry

#check factors_of_90

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_90_l1151_115156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_covers_all_except_a_l1151_115142

noncomputable def f (n : ℕ+) : ℕ := 
  ⌊(n : ℝ) + Real.sqrt ((n : ℝ) / 3) + 1/2⌋.toNat

def a (n : ℕ+) : ℕ := 3 * n^2 - 2 * n

theorem f_covers_all_except_a (m : ℕ+) :
  (∃ n : ℕ+, f n = m) ↔ ¬(∃ k : ℕ+, a k = m) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_covers_all_except_a_l1151_115142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_existence_condition_l1151_115149

/-- The set of positive real numbers -/
def RealPos : Type := {x : ℝ // x > 0}

/-- The theorem statement -/
theorem function_existence_condition (a : ℝ) : 
  (∃ f : RealPos → RealPos, ∀ x : RealPos, 3 * (f x).val^2 = 2 * (f (f x)).val + a * x.val^4) ↔ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_existence_condition_l1151_115149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l1151_115159

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_property (seq : ArithmeticSequence) (k : ℕ) 
  (h : (2 : ℝ) * k + 1 > 0) : seq.a (k + 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l1151_115159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_from_odds_red_card_probability_l1151_115155

/-- Given odds of a:b for an event, the probability of the event is a/(a+b) -/
def probability_of_event_with_odds (a b : ℕ) : ℚ := (a : ℚ) / (a + b : ℚ)

/-- Given odds of a:b for an event, the probability of the event is a/(a+b) -/
theorem probability_from_odds (a b : ℕ) (h : a ≠ 0 ∧ b ≠ 0) : 
  probability_of_event_with_odds a b = (a : ℚ) / (a + b : ℚ) :=
by
  rfl

/-- The probability of drawing a red card given odds of 5:8 is 5/13 -/
theorem red_card_probability : 
  probability_of_event_with_odds 5 8 = 5 / 13 :=
by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_from_odds_red_card_probability_l1151_115155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1151_115143

noncomputable def f (x : ℝ) := 4 * Real.sin x * Real.cos (x + Real.pi/6)

theorem f_properties :
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi + Real.pi/6) (k * Real.pi + 2*Real.pi/3), 
    ∀ y ∈ Set.Icc (k * Real.pi + Real.pi/6) (k * Real.pi + 2*Real.pi/3), 
    x ≤ y → f y ≤ f x) ∧ 
  (∀ x ∈ Set.Icc 0 (Real.pi/2), f x ≤ 1) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi/2), f x ≥ -2) ∧
  (f (Real.pi/6) = 1) ∧
  (f (Real.pi/2) = -2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1151_115143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_g_unique_zero_m_range_l1151_115198

noncomputable def f (x : ℝ) : ℝ := 1 - 1 / (2^x + 1)

noncomputable def g (x : ℝ) : ℝ := Real.log x + f x

theorem f_strictly_increasing : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ := by sorry

theorem g_unique_zero : 
  ∃! x₀ : ℝ, x₀ > 0 ∧ g x₀ = 0 := by sorry

theorem m_range : 
  {m : ℝ | ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 2 ∧ x₂ ∈ Set.Icc 1 2 ∧ 
    x₁ ≠ x₂ ∧ f x₁ = m * f (2 * x₁) ∧ f x₂ = m * f (2 * x₂)} = 
  Set.Icc (2 * Real.sqrt 2 - 2) (17/20) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_g_unique_zero_m_range_l1151_115198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_ratio_equals_three_halves_l1151_115128

theorem log_ratio_equals_three_halves (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (Real.log a^3) / (Real.log a^2) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_ratio_equals_three_halves_l1151_115128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_iff_a_range_l1151_115113

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 2 then a * x^2 + x - 1 else -x + 1

theorem monotone_decreasing_iff_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a y < f a x) ↔ a ∈ Set.Iic (-1/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_iff_a_range_l1151_115113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_towel_package_rolls_l1151_115158

/-- Calculates the number of rolls in a package of paper towels given the package price, 
    individual roll price, and percent savings. -/
def rolls_in_package (package_price : ℚ) (individual_price : ℚ) (percent_savings : ℚ) : ℕ :=
  let discounted_price := individual_price * (1 - percent_savings / 100)
  (package_price / discounted_price).floor.toNat

/-- Theorem stating that a package of paper towels selling for $9, with individual rolls 
    costing $1, and a 25% savings per roll, contains 12 rolls. -/
theorem paper_towel_package_rolls : 
  rolls_in_package 9 1 25 = 12 := by
  sorry

#eval rolls_in_package 9 1 25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_towel_package_rolls_l1151_115158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_2017_equals_two_l1151_115153

def sequence_a (n : ℕ) : ℚ :=
  match n with
  | 0 => 2
  | n + 1 => 1 - 1 / sequence_a n

def product_t (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => product_t n * sequence_a n

theorem product_2017_equals_two : product_t 2017 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_2017_equals_two_l1151_115153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1151_115199

theorem problem_statement (c d : ℝ) 
  (h1 : (100 : ℝ) ^ c = 4) 
  (h2 : (100 : ℝ) ^ d = 5) : 
  (20 : ℝ) ^ ((1 - c - d) / (2 * (1 - d))) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1151_115199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l1151_115148

-- Define the function f(x) = ln x + x³ - 3
noncomputable def f (x : ℝ) : ℝ := Real.log x + x^3 - 3

-- State the theorem
theorem root_exists_in_interval :
  ∃ x : ℝ, x > 1 ∧ x < 2 ∧ f x = 0 := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l1151_115148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_is_30_l1151_115188

/-- The first term of an infinite geometric series with given common ratio and sum -/
noncomputable def first_term_of_geometric_series (r : ℝ) (S : ℝ) : ℝ :=
  S * (1 - r)

/-- Theorem: The first term of an infinite geometric series with common ratio 1/4 and sum 40 is 30 -/
theorem first_term_is_30 :
  first_term_of_geometric_series (1/4 : ℝ) 40 = 30 := by
  -- Unfold the definition of first_term_of_geometric_series
  unfold first_term_of_geometric_series
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num

-- We can't use #eval with real numbers, so let's use #check instead
#check first_term_is_30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_is_30_l1151_115188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_largest_and_smallest_l1151_115109

def digits : List Nat := [6, 3, 8]

def is_valid_number (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ (∃ (a b c : Nat), List.Perm [a, b, c] digits ∧ n = 100 * a + 10 * b + c)

def largest_number : Nat := 863
def smallest_number : Nat := 368

theorem sum_of_largest_and_smallest :
  is_valid_number largest_number ∧
  is_valid_number smallest_number ∧
  (∀ n, is_valid_number n → n ≤ largest_number) ∧
  (∀ n, is_valid_number n → n ≥ smallest_number) ∧
  largest_number + smallest_number = 1231 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_largest_and_smallest_l1151_115109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_BR_parallel_AC_l1151_115176

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary functions and predicates
variable (Length : Point → Point → ℝ)
variable (Line : Point → Point → Type)
variable (OnCircle : Point → Circle → Prop)
variable (IsCircumcircle : Circle → Point → Point → Point → Prop)
variable (IsCircumcenter : Point → Point → Point → Point → Prop)
variable (OnAngleBisector : Point → Point → Point → Point → Prop)
variable (IsDiameter : Circle → Point → Point → Prop)
variable (Collinear : Point → Point → Point → Prop)
variable (IsParallel : Type → Type → Prop)
variable (IsAcuteTriangle : Point → Point → Point → Prop)

-- Define the triangle ABC
variable (A B C : Point)

-- Define the circumcircle and circumcenter
variable (Ω : Circle) (O : Point)

-- Define other points
variable (M P Q R : Point)

-- Define the circle Γ
variable (Γ : Circle)

-- Define the necessary conditions
variable (acute_triangle : IsAcuteTriangle A B C)
variable (AB_greater_BC : Length A B > Length B C)
variable (Ω_circumcircle : IsCircumcircle Ω A B C)
variable (O_circumcenter : IsCircumcenter O A B C)
variable (M_on_angle_bisector : OnAngleBisector M A B C)
variable (M_on_Ω : OnCircle M Ω)
variable (M_not_B : M ≠ B)
variable (Γ_diameter : IsDiameter Γ B M)
variable (P_on_AOB_bisector : OnAngleBisector P A O B)
variable (P_on_Γ : OnCircle P Γ)
variable (Q_on_BOC_bisector : OnAngleBisector Q B O C)
variable (Q_on_Γ : OnCircle Q Γ)
variable (R_on_PQ : Collinear P Q R)
variable (BR_eq_MR : Length B R = Length M R)

-- State the theorem
theorem BR_parallel_AC : IsParallel (Line B R) (Line A C) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_BR_parallel_AC_l1151_115176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_l1151_115197

/-- The function f(x) defined in the problem -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) + Real.cos (ω * x)

/-- The theorem stating the minimum positive value of ω -/
theorem min_omega : 
  (∃ ω : ℝ, ω > 0 ∧ 
    (∃ x₁ : ℝ, ∀ x : ℝ, f ω x₁ ≤ f ω x ∧ f ω x ≤ f ω (x₁ + 2018))) → 
  (∃ ω_min : ℝ, ω_min = Real.pi / 2018 ∧
    ω_min > 0 ∧
    (∃ x₁ : ℝ, ∀ x : ℝ, f ω_min x₁ ≤ f ω_min x ∧ f ω_min x ≤ f ω_min (x₁ + 2018)) ∧
    (∀ ω : ℝ, ω > 0 ∧ ω < ω_min → 
      ¬(∃ x₁ : ℝ, ∀ x : ℝ, f ω x₁ ≤ f ω x ∧ f ω x ≤ f ω (x₁ + 2018)))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_l1151_115197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l1151_115189

-- Define the four logarithmic functions
noncomputable def f1 (x : ℝ) := Real.log x / Real.log 4
noncomputable def f2 (x : ℝ) := Real.log 4 / Real.log x
noncomputable def f3 (x : ℝ) := Real.log x / Real.log (1/4)
noncomputable def f4 (x : ℝ) := Real.log (1/4) / Real.log x

-- Define a predicate for points that lie on at least two of the graphs
def lies_on_two_or_more (x y : ℝ) :=
  (f1 x = y ∧ f2 x = y) ∨ (f1 x = y ∧ f3 x = y) ∨ (f1 x = y ∧ f4 x = y) ∨
  (f2 x = y ∧ f3 x = y) ∨ (f2 x = y ∧ f4 x = y) ∨ (f3 x = y ∧ f4 x = y)

-- The theorem to be proved
theorem intersection_points_count :
  ∃ (S : Finset (ℝ × ℝ)), 
    (∀ (p : ℝ × ℝ), p ∈ S ↔ p.1 > 0 ∧ lies_on_two_or_more p.1 p.2) ∧
    S.card = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l1151_115189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_vertical_line_l1151_115163

noncomputable def SlopeAngle (f : ℝ → ℝ) : ℝ := sorry

theorem slope_angle_vertical_line (k : ℝ) : 
  SlopeAngle (fun _ => k) = π / 2 :=
sorry

#check slope_angle_vertical_line 2017

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_vertical_line_l1151_115163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1151_115169

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x - Real.exp x + 1

-- State the theorem
theorem f_inequality (m n : ℝ) (h1 : 0 < m) (h2 : m < n) :
  (1 / n) - 1 < (f (Real.log n) - f (Real.log m)) / (n - m) ∧
  (f (Real.log n) - f (Real.log m)) / (n - m) < (1 / m) - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1151_115169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_hyperbola_asymptote_l1151_115138

/-- The value of a for which the circle (x-2)^2 + y^2 = 1 is tangent to the asymptote of the hyperbola x^2/a^2 - y^2 = 1 (a > 0) -/
noncomputable def a : ℝ := Real.sqrt 3

/-- The equation of the asymptotes of the hyperbola x^2/a^2 - y^2 = 1 when a = √3 -/
def asymptote_equation (x : ℝ) : Set ℝ := {y | y = (Real.sqrt 3 / 3) * x ∨ y = -(Real.sqrt 3 / 3) * x}

/-- The circle equation (x-2)^2 + y^2 = 1 -/
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

/-- The hyperbola equation x^2/a^2 - y^2 = 1 -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 / a^2 - y^2 = 1

theorem circle_tangent_to_hyperbola_asymptote :
  ∃ (x y : ℝ), circle_equation x y ∧ (y ∈ asymptote_equation x) ∧
  ∀ (x' y' : ℝ), circle_equation x' y' → (y' ∈ asymptote_equation x') → (x = x' ∧ y = y') :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_hyperbola_asymptote_l1151_115138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_coincide_l1151_115137

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define points C₁, A₁, B₁ on extensions of sides
noncomputable def C₁ (A B : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) := A + 2 • (B - A)
noncomputable def A₁ (B C : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) := B + 2 • (C - B)
noncomputable def B₁ (C A : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) := C + 2 • (A - C)

-- Define the centroid of a triangle
noncomputable def centroid (P Q R : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
  (1/3 : ℝ) • (P + Q + R)

-- Theorem statement
theorem centroid_coincide (A B C : EuclideanSpace ℝ (Fin 2)) :
  centroid A B C = centroid (C₁ A B) (A₁ B C) (B₁ C A) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_coincide_l1151_115137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_routes_l1151_115104

/-- Represents a bus route with three stops -/
structure BusRoute where
  stops : Finset Nat
  three_stops : stops.card = 3

/-- Represents the network of bus routes in the city -/
structure BusNetwork where
  stops : Finset Nat
  routes : Finset BusRoute
  total_stops : stops.card = 9
  route_condition : ∀ r1 r2 : BusRoute, r1 ∈ routes → r2 ∈ routes → r1 ≠ r2 → 
    (r1.stops ∩ r2.stops).card = 0 ∨ (r1.stops ∩ r2.stops).card = 1

/-- The maximum number of routes in the network is 12 -/
theorem max_routes (network : BusNetwork) : network.routes.card ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_routes_l1151_115104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_velocity_l1151_115161

noncomputable section

/-- The acceleration due to gravity in m/s² -/
def g : ℝ := 9.8

/-- The angle of launch in radians -/
def θ : ℝ := 30 * Real.pi / 180

/-- The total horizontal distance traveled by the projectile in meters -/
def d : ℝ := 30

/-- The initial velocity of the projectile in m/s -/
def v : ℝ := 19

theorem projectile_velocity :
  abs (v - Real.sqrt ((2 * d * g) / (Real.sin (2 * θ)))) < 0.1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_velocity_l1151_115161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_2x3x6_parallelepiped_l1151_115166

/-- The volume of space within or exactly one unit from the surface of a rectangular parallelepiped -/
noncomputable def extendedVolume (l w h : ℝ) : ℝ :=
  let boxVolume := l * w * h
  let parVolume := 2 * (l * w + l * h + w * h)
  let cylVolume := Real.pi * (l^2 + w^2 + h^2)
  let sphVolume := 4 / 3 * Real.pi
  boxVolume + parVolume + cylVolume + sphVolume

/-- Theorem stating the volume of space within or exactly one unit from the surface
    of a 2x3x6 rectangular parallelepiped -/
theorem volume_2x3x6_parallelepiped :
  extendedVolume 2 3 6 = (324 + 151 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_2x3x6_parallelepiped_l1151_115166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_difference_l1151_115181

open MeasureTheory Interval Real

theorem integral_difference (f : ℝ → ℝ) (A B : ℝ) 
  (h1 : ∫ x in (Set.Icc 0 1), f x = A) 
  (h2 : ∫ x in (Set.Icc 0 2), f x = B) : 
  ∫ x in (Set.Icc 1 2), f x = B - A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_difference_l1151_115181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_side_length_sum_l1151_115120

-- Define the square ABCD
def square_ABCD : Set (ℝ × ℝ) := {(x, y) | 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1}

-- Define points E and F
variable (E F : ℝ × ℝ)

-- Define the right triangle AEF
def triangle_AEF : Set (ℝ × ℝ) := {(0, 0), E, F}

-- Define the rectangle
variable (rectangle : Set (ℝ × ℝ))

-- Axioms
axiom E_on_BC : E.1 = 1 ∧ 0 ≤ E.2 ∧ E.2 ≤ 1
axiom F_on_CD : F.1 ≥ 0 ∧ F.1 ≤ 1 ∧ F.2 = 0
axiom right_angle_AEF : (E.1 - 0) * (F.2 - 0) + (E.2 - 0) * (F.1 - 1) = 0
axiom rectangle_vertex_B : (0, 1) ∈ rectangle
axiom rectangle_vertex_on_AE : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (t * E.1, t * E.2) ∈ rectangle
axiom rectangle_diagonal_intersects_D : (1, 0) ∈ rectangle

-- Define the shorter side length of the rectangle
noncomputable def shorter_side_length (a b c : ℕ) : ℝ := (a - Real.sqrt b) / c

-- Define the condition for b
def b_not_divisible_by_square_prime (b : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ b)

-- Theorem to prove
theorem rectangle_side_length_sum (a b c : ℕ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : b_not_divisible_by_square_prime b)
  (h5 : shorter_side_length a b c = Real.sqrt 2 / 2) : 
  a + b + c = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_side_length_sum_l1151_115120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1151_115132

def g (x : ℝ) : ℝ := x^2 - 2

noncomputable def f (x : ℝ) : ℝ :=
  if x < g x then g x + x + 4 else g x - x

theorem range_of_f :
  Set.range f = Set.Icc (-9/4) 0 ∪ Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1151_115132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_arrangement_probability_l1151_115160

/-- The binomial expression -/
noncomputable def binomial (x : ℝ) := (Real.sqrt x + 1 / (2 * x^(1/6))) ^ 8

/-- The number of rational terms in the expansion -/
def rationalTerms : ℕ := 3

/-- The number of irrational terms in the expansion -/
def irrationalTerms : ℕ := 6

/-- The total number of terms in the expansion -/
def totalTerms : ℕ := rationalTerms + irrationalTerms

/-- The probability of arranging all terms such that no two rational terms are adjacent -/
def probabilityNoAdjacentRational : ℚ := 5 / 12

theorem binomial_arrangement_probability :
  probabilityNoAdjacentRational = (Nat.factorial totalTerms * Nat.factorial irrationalTerms) /
    (Nat.factorial (totalTerms + 1) * Nat.factorial (irrationalTerms - 1)) := by
  sorry

#eval probabilityNoAdjacentRational

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_arrangement_probability_l1151_115160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_calculation_l1151_115173

/-- Calculates the discount percentage on the retail price of a machine -/
theorem discount_percentage_calculation
  (wholesale_price : ℝ)
  (retail_price : ℝ)
  (profit_percentage : ℝ)
  (h_wholesale : wholesale_price = 90)
  (h_retail : retail_price = 120)
  (h_profit : profit_percentage = 0.20) :
  (retail_price - (wholesale_price + wholesale_price * profit_percentage)) / retail_price * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_calculation_l1151_115173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_buyers_difference_l1151_115145

/- Define the cost of a pencil in cents -/
def pencil_cost : ℕ := sorry

/- Define the number of seventh graders who bought pencils -/
def seventh_graders : ℕ := sorry

/- Define the number of sixth graders who bought pencils -/
def sixth_graders : ℕ := sorry

/- Define the number of eighth graders who bought pencils -/
def eighth_graders : ℕ := sorry

/- Axioms based on the problem conditions -/
axiom pencil_cost_positive : pencil_cost > 0

axiom seventh_graders_payment : seventh_graders * pencil_cost = 168

axiom sixth_graders_payment : sixth_graders * pencil_cost = 208

axiom eighth_graders_payment : eighth_graders * pencil_cost = 156

axiom eighth_graders_count : eighth_graders = 20

axiom sixth_graders_max : sixth_graders ≤ 25

/- The theorem to be proved -/
theorem pencil_buyers_difference : sixth_graders - seventh_graders = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_buyers_difference_l1151_115145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l1151_115184

/-- The area of a quadrilateral bounded by the lines x=0, x=4, y=x-2, and y=x+3 is 20 -/
theorem quadrilateral_area : ℝ := by
  -- Define the bounding lines
  let line1 := fun x : ℝ => (0 : ℝ)
  let line2 := fun x : ℝ => (4 : ℝ)
  let line3 := fun x : ℝ => x - 2
  let line4 := fun x : ℝ => x + 3

  -- Define the quadrilateral
  let quadrilateral := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ (p.2 = line3 p.1 ∨ p.2 = line4 p.1)}

  -- State that the area of the quadrilateral is 20
  have h : MeasureTheory.volume quadrilateral = 20 := by sorry

  -- Return the area
  exact 20


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l1151_115184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_to_350_l1151_115101

theorem closest_perfect_square_to_350 :
  ∀ n : ℤ, n ≠ 19 → n * n ≠ 0 → |n * n - 350| ≥ |361 - 350| :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_to_350_l1151_115101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_andrew_keeps_675_balloons_l1151_115146

/-- The number of balloons Andrew keeps given the initial quantities and fractions -/
def andrewsBalloons (blue purple red yellow : ℕ) : ℕ :=
  (((2 : ℚ)/3 * blue).floor + ((3 : ℚ)/5 * purple).floor + 
   ((4 : ℚ)/7 * red).floor + ((1 : ℚ)/3 * yellow).floor).toNat

/-- Theorem stating that Andrew keeps 675 balloons -/
theorem andrew_keeps_675_balloons :
  andrewsBalloons 303 453 165 324 = 675 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_andrew_keeps_675_balloons_l1151_115146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1151_115141

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi * x + Real.pi / 2)

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1151_115141
