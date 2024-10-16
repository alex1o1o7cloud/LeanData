import Mathlib

namespace NUMINAMATH_CALUDE_equation_represents_pair_of_lines_l3627_362722

theorem equation_represents_pair_of_lines : 
  ∃ (m₁ m₂ : ℝ), ∀ (x y : ℝ), 4 * x^2 - 9 * y^2 = 0 ↔ (y = m₁ * x ∨ y = m₂ * x) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_pair_of_lines_l3627_362722


namespace NUMINAMATH_CALUDE_similar_triangles_hypotenuse_l3627_362764

-- Define the properties of the triangles
def smallTriangleArea : ℝ := 8
def largeTriangleArea : ℝ := 200
def smallTriangleHypotenuse : ℝ := 10

-- Define the theorem
theorem similar_triangles_hypotenuse :
  ∃ (smallLeg1 smallLeg2 largeLeg1 largeLeg2 largeHypotenuse : ℝ),
    -- Conditions for the smaller triangle
    smallLeg1 > 0 ∧ smallLeg2 > 0 ∧
    smallLeg1 * smallLeg2 / 2 = smallTriangleArea ∧
    smallLeg1^2 + smallLeg2^2 = smallTriangleHypotenuse^2 ∧
    -- Conditions for the larger triangle
    largeLeg1 > 0 ∧ largeLeg2 > 0 ∧
    largeLeg1 * largeLeg2 / 2 = largeTriangleArea ∧
    largeLeg1^2 + largeLeg2^2 = largeHypotenuse^2 ∧
    -- Similarity condition
    largeLeg1 / smallLeg1 = largeLeg2 / smallLeg2 ∧
    -- Conclusion
    largeHypotenuse = 50 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_hypotenuse_l3627_362764


namespace NUMINAMATH_CALUDE_patio_table_cost_l3627_362730

/-- The cost of the patio table given the total cost and chair costs -/
theorem patio_table_cost (total_cost : ℕ) (chair_cost : ℕ) (num_chairs : ℕ) :
  total_cost = 135 →
  chair_cost = 20 →
  num_chairs = 4 →
  total_cost - (num_chairs * chair_cost) = 55 :=
by sorry

end NUMINAMATH_CALUDE_patio_table_cost_l3627_362730


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_equal_area_l3627_362710

/-- An isosceles trapezoid -/
structure IsoscelesTrapezoid where
  /-- The length of the diagonal -/
  diagonalLength : ℝ
  /-- The angle between the diagonals -/
  diagonalAngle : ℝ
  /-- The area of the trapezoid -/
  area : ℝ

/-- 
Theorem: If two isosceles trapezoids have equal diagonal lengths and equal angles between their diagonals, 
then their areas are equal.
-/
theorem isosceles_trapezoid_equal_area 
  (t1 t2 : IsoscelesTrapezoid) 
  (h1 : t1.diagonalLength = t2.diagonalLength) 
  (h2 : t1.diagonalAngle = t2.diagonalAngle) : 
  t1.area = t2.area :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_equal_area_l3627_362710


namespace NUMINAMATH_CALUDE_unique_solution_3_and_7_equation_l3627_362759

theorem unique_solution_3_and_7_equation :
  ∀ a y : ℕ, a ≥ 1 → y ≥ 1 →
  (3 ^ (2 * a - 1) + 3 ^ a + 1 = 7 ^ y) →
  (a = 1 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_3_and_7_equation_l3627_362759


namespace NUMINAMATH_CALUDE_threeTangentLines_l3627_362745

/-- Represents a circle in the 2D plane --/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The first circle: x^2 + y^2 + 4x - 4y + 7 = 0 --/
def circle1 : Circle := { a := 1, b := 1, c := 4, d := -4, e := 7 }

/-- The second circle: x^2 + y^2 - 4x - 10y + 13 = 0 --/
def circle2 : Circle := { a := 1, b := 1, c := -4, d := -10, e := 13 }

/-- Count the number of lines tangent to both circles --/
def countTangentLines (c1 c2 : Circle) : ℕ :=
  sorry

/-- Theorem stating that there are exactly 3 lines tangent to both circles --/
theorem threeTangentLines : countTangentLines circle1 circle2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_threeTangentLines_l3627_362745


namespace NUMINAMATH_CALUDE_even_increasing_inequality_l3627_362757

/-- A function f is even if f(-x) = f(x) for all x -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function f is increasing on [0, +∞) if f(x) ≤ f(y) for all 0 ≤ x ≤ y -/
def IncreasingOnNonnegatives (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ y → f x ≤ f y

theorem even_increasing_inequality (f : ℝ → ℝ) (a : ℝ) 
    (heven : EvenFunction f) (hincr : IncreasingOnNonnegatives f) :
    f (-1) < f (a^2 - 2*a + 3) := by
  sorry

end NUMINAMATH_CALUDE_even_increasing_inequality_l3627_362757


namespace NUMINAMATH_CALUDE_plane_equation_proof_l3627_362721

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the coefficients of a plane equation Ax + By + Cz + D = 0 -/
structure PlaneCoefficients where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane given by its coefficients -/
def pointLiesOnPlane (p : Point3D) (coeff : PlaneCoefficients) : Prop :=
  coeff.A * p.x + coeff.B * p.y + coeff.C * p.z + coeff.D = 0

/-- The greatest common divisor of the absolute values of four integers is 1 -/
def gcdIsOne (a b c d : ℤ) : Prop :=
  Nat.gcd (Nat.gcd (Int.natAbs a) (Int.natAbs b)) (Nat.gcd (Int.natAbs c) (Int.natAbs d)) = 1

theorem plane_equation_proof (p1 p2 p3 : Point3D) (coeff : PlaneCoefficients) : 
  p1 = ⟨2, -3, 1⟩ →
  p2 = ⟨6, -3, 3⟩ →
  p3 = ⟨4, -5, 2⟩ →
  coeff = ⟨2, 3, -4, 9⟩ →
  pointLiesOnPlane p1 coeff ∧
  pointLiesOnPlane p2 coeff ∧
  pointLiesOnPlane p3 coeff ∧
  coeff.A > 0 ∧
  gcdIsOne coeff.A coeff.B coeff.C coeff.D := by
  sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l3627_362721


namespace NUMINAMATH_CALUDE_product_sign_l3627_362748

theorem product_sign (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x^4 - y^4 > x) (h2 : y^4 - x^4 > y) : x * y > 0 := by
  sorry

end NUMINAMATH_CALUDE_product_sign_l3627_362748


namespace NUMINAMATH_CALUDE_critical_point_iff_a_in_range_l3627_362715

/-- The function f(x) = x³ - ax² + ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + a*x + 3

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x + a

/-- A critical point of f exists if and only if f'(x) = 0 has real solutions -/
def has_critical_point (a : ℝ) : Prop := ∃ x : ℝ, f' a x = 0

/-- The main theorem: f(x) has a critical point iff a ∈ (-∞, 0) ∪ (3, +∞) -/
theorem critical_point_iff_a_in_range (a : ℝ) :
  has_critical_point a ↔ a < 0 ∨ a > 3 := by sorry

end NUMINAMATH_CALUDE_critical_point_iff_a_in_range_l3627_362715


namespace NUMINAMATH_CALUDE_shoe_price_after_changes_lous_shoe_price_l3627_362777

/-- The price of shoes after a price increase followed by a discount -/
theorem shoe_price_after_changes (initial_price : ℝ) 
  (increase_percent : ℝ) (discount_percent : ℝ) : ℝ := by
  -- Define the price after increase
  let price_after_increase := initial_price * (1 + increase_percent / 100)
  -- Define the final price after discount
  let final_price := price_after_increase * (1 - discount_percent / 100)
  -- Prove that when initial_price = 40, increase_percent = 10, and discount_percent = 10,
  -- the final_price is 39.60
  sorry

/-- The specific case for Lou's Fine Shoes -/
theorem lous_shoe_price : 
  shoe_price_after_changes 40 10 10 = 39.60 := by sorry

end NUMINAMATH_CALUDE_shoe_price_after_changes_lous_shoe_price_l3627_362777


namespace NUMINAMATH_CALUDE_exponential_decreasing_range_l3627_362718

/-- Given a function f(x) = a^x where a > 0 and a ≠ 1, 
    if f(m) < f(n) for all m > n, then 0 < a < 1 -/
theorem exponential_decreasing_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ m n : ℝ, m > n → a^m < a^n) → 0 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_exponential_decreasing_range_l3627_362718


namespace NUMINAMATH_CALUDE_equation_system_equivalence_l3627_362750

theorem equation_system_equivalence (x y : ℝ) :
  (3 * x^2 + 9 * x + 7 * y + 2 = 0) ∧ (3 * x + 2 * y + 4 = 0) →
  4 * y^2 + 19 * y - 14 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_system_equivalence_l3627_362750


namespace NUMINAMATH_CALUDE_remainder_3_305_mod_13_l3627_362789

theorem remainder_3_305_mod_13 : 3^305 % 13 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_305_mod_13_l3627_362789


namespace NUMINAMATH_CALUDE_efqs_equals_qrst_l3627_362709

/-- Assigns a value to each letter of the alphabet -/
def letter_value (c : Char) : ℕ :=
  (c.toNat - 'A'.toNat + 1)

/-- Calculates the product of values assigned to a list of characters -/
def list_product (s : List Char) : ℕ :=
  s.map letter_value |>.foldl (·*·) 1

/-- Checks if a list of characters contains distinct elements -/
def distinct_chars (s : List Char) : Prop :=
  s.toFinset.card = s.length

theorem efqs_equals_qrst : ∃ (e f q s : Char), 
  distinct_chars ['E', 'F', 'Q', 'S'] ∧
  list_product ['E', 'F', 'Q', 'S'] = list_product ['Q', 'R', 'S', 'T'] :=
by sorry

end NUMINAMATH_CALUDE_efqs_equals_qrst_l3627_362709


namespace NUMINAMATH_CALUDE_number_of_children_l3627_362749

theorem number_of_children : ∃ n : ℕ, 
  (∃ b : ℕ, b = 3 * n + 4) ∧ 
  (∃ b : ℕ, b = 4 * n - 3) ∧ 
  n = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_of_children_l3627_362749


namespace NUMINAMATH_CALUDE_light_path_in_cube_l3627_362765

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- Represents a light path in the cube -/
structure LightPath where
  start : Point3D
  reflection : Point3D
  length : ℝ

/-- Theorem stating the properties of the light path in the cube -/
theorem light_path_in_cube (cube : Cube) (path : LightPath) :
  cube.sideLength = 12 →
  path.start = Point3D.mk 0 0 0 →
  path.reflection = Point3D.mk 12 5 7 →
  ∃ (m n : ℕ), 
    path.length = m * Real.sqrt n ∧ 
    ¬ ∃ (p : ℕ), Prime p ∧ p^2 ∣ n ∧
    m + n = 230 := by
  sorry

end NUMINAMATH_CALUDE_light_path_in_cube_l3627_362765


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l3627_362714

theorem trigonometric_equation_solution (x : ℝ) :
  (8.4743 * Real.tan (2 * x) - 4 * Real.tan (3 * x) = Real.tan (3 * x)^2 * Real.tan (2 * x)) ↔
  (∃ k : ℤ, x = k * Real.pi ∨ x = Real.arctan (Real.sqrt (3 / 5)) + k * Real.pi ∨ 
   x = -Real.arctan (Real.sqrt (3 / 5)) + k * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l3627_362714


namespace NUMINAMATH_CALUDE_b_52_mod_55_l3627_362778

/-- Definition of b_n as the integer obtained by writing all integers from 1 to n from left to right -/
def b (n : ℕ) : ℕ := sorry

/-- Theorem stating that the remainder of b_52 divided by 55 is 2 -/
theorem b_52_mod_55 : b 52 % 55 = 2 := by sorry

end NUMINAMATH_CALUDE_b_52_mod_55_l3627_362778


namespace NUMINAMATH_CALUDE_line_perpendicular_plane_implies_planes_perpendicular_l3627_362720

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the contained relation between a line and a plane
variable (contained : Line → Plane → Prop)

-- Define the perpendicular relation between two planes
variable (perpendicularPlanes : Plane → Plane → Prop)

-- Theorem statement
theorem line_perpendicular_plane_implies_planes_perpendicular
  (m : Line) (α β : Plane)
  (h1 : perpendicular m α)
  (h2 : contained m β) :
  perpendicularPlanes α β :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_plane_implies_planes_perpendicular_l3627_362720


namespace NUMINAMATH_CALUDE_orthogonal_circles_on_radical_axis_l3627_362736

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the orthogonality condition
def is_orthogonal (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  c1.radius^2 = (x1 - x2)^2 + (y1 - y2)^2 - c2.radius^2

-- Define the radical axis
def on_radical_axis (p : ℝ × ℝ) (c1 c2 : Circle) : Prop :=
  let (x, y) := p
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x - x1)^2 + (y - y1)^2 - c1.radius^2 = (x - x2)^2 + (y - y2)^2 - c2.radius^2

-- Main theorem
theorem orthogonal_circles_on_radical_axis (S1 S2 : Circle) (O : ℝ × ℝ) :
  (∃ r : ℝ, r > 0 ∧ is_orthogonal ⟨O, r⟩ S1 ∧ is_orthogonal ⟨O, r⟩ S2) ↔
  (on_radical_axis O S1 S2 ∧ O ≠ S1.center ∧ O ≠ S2.center) :=
sorry

end NUMINAMATH_CALUDE_orthogonal_circles_on_radical_axis_l3627_362736


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l3627_362724

theorem cyclic_sum_inequality (a b c : ℝ) (k : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hk : k ≥ 2) (habc : a * b * c = 1) :
  (a^k / (a + b)) + (b^k / (b + c)) + (c^k / (c + a)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l3627_362724


namespace NUMINAMATH_CALUDE_calculate_expression_l3627_362770

theorem calculate_expression : 500 * 4020 * 0.0402 * 20 = 1616064000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3627_362770


namespace NUMINAMATH_CALUDE_two_from_four_is_six_l3627_362732

/-- The number of ways to choose 2 items from a set of 4 distinct items -/
def choose_two_from_four : ℕ := Nat.choose 4 2

/-- Theorem stating that choosing 2 items from 4 distinct items results in 6 possibilities -/
theorem two_from_four_is_six : choose_two_from_four = 6 := by
  sorry

end NUMINAMATH_CALUDE_two_from_four_is_six_l3627_362732


namespace NUMINAMATH_CALUDE_journey_speed_problem_l3627_362791

theorem journey_speed_problem (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) :
  total_distance = 448 →
  total_time = 20 →
  second_half_speed = 24 →
  ∃ first_half_speed : ℝ,
    first_half_speed * (total_time / 2) = total_distance / 2 ∧
    second_half_speed * (total_time / 2) = total_distance / 2 ∧
    first_half_speed = 21 := by
  sorry


end NUMINAMATH_CALUDE_journey_speed_problem_l3627_362791


namespace NUMINAMATH_CALUDE_product_103_97_l3627_362735

theorem product_103_97 : 103 * 97 = 9991 := by
  sorry

end NUMINAMATH_CALUDE_product_103_97_l3627_362735


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3627_362738

open Set

def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def M : Set ℕ := {0, 3, 5}
def N : Set ℕ := {1, 4, 5}

theorem intersection_complement_equality : M ∩ (U \ N) = {0, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3627_362738


namespace NUMINAMATH_CALUDE_real_number_line_bijection_l3627_362729

/-- A point on the number line -/
structure NumberLinePoint where
  position : ℝ

/-- The bijective function between real numbers and points on the number line -/
def realToPoint : ℝ → NumberLinePoint :=
  λ x ↦ ⟨x⟩

theorem real_number_line_bijection :
  Function.Bijective realToPoint :=
sorry

end NUMINAMATH_CALUDE_real_number_line_bijection_l3627_362729


namespace NUMINAMATH_CALUDE_right_triangle_check_l3627_362737

/-- Checks if three numbers can form a right-angled triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

theorem right_triangle_check :
  ¬ is_right_triangle 1 3 4 ∧
  ¬ is_right_triangle 2 3 4 ∧
  ¬ is_right_triangle 1 1 (Real.sqrt 3) ∧
  is_right_triangle 5 12 13 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_check_l3627_362737


namespace NUMINAMATH_CALUDE_fraction_division_addition_l3627_362766

theorem fraction_division_addition : (3 / 7 : ℚ) / 4 + 2 = 59 / 28 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_addition_l3627_362766


namespace NUMINAMATH_CALUDE_unit_distance_preservation_implies_all_distance_preservation_l3627_362706

/-- A function that maps points on a plane to other points on the same plane -/
def PlaneMap (Plane : Type*) := Plane → Plane

/-- Distance function between two points on a plane -/
def distance (Plane : Type*) := Plane → Plane → ℝ

/-- A function preserves unit distances if the distance between the images of any two points
    that are one unit apart is also one unit -/
def preserves_unit_distances (Plane : Type*) (f : PlaneMap Plane) (d : distance Plane) :=
  ∀ (P Q : Plane), d P Q = 1 → d (f P) (f Q) = 1

/-- A function preserves all distances if the distance between the images of any two points
    is equal to the distance between the original points -/
def preserves_all_distances (Plane : Type*) (f : PlaneMap Plane) (d : distance Plane) :=
  ∀ (P Q : Plane), d (f P) (f Q) = d P Q

/-- Main theorem: if a plane map preserves unit distances, it preserves all distances -/
theorem unit_distance_preservation_implies_all_distance_preservation
  (Plane : Type*) (f : PlaneMap Plane) (d : distance Plane) :
  preserves_unit_distances Plane f d → preserves_all_distances Plane f d :=
by
  sorry

end NUMINAMATH_CALUDE_unit_distance_preservation_implies_all_distance_preservation_l3627_362706


namespace NUMINAMATH_CALUDE_imaginary_part_of_iz_l3627_362792

theorem imaginary_part_of_iz (z : ℂ) (h : z^2 - 4*z + 5 = 0) : 
  Complex.im (Complex.I * z) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_iz_l3627_362792


namespace NUMINAMATH_CALUDE_hyperbola_ratio_l3627_362779

/-- Given a point M(x, 5/x) in the first quadrant on the hyperbola y = 5/x,
    with A(x, 0), B(0, 5/x), C(x, 3/x), and D(3/y, y) where y = 5/x,
    prove that the ratio CD:AB = 2:5 -/
theorem hyperbola_ratio (x : ℝ) (hx : x > 0) : 
  let y := 5 / x
  let m := (x, y)
  let a := (x, 0)
  let b := (0, y)
  let c := (x, 3 / x)
  let d := (3 / y, y)
  let cd := Real.sqrt ((x - 3 / y)^2 + (3 / x - y)^2)
  let ab := Real.sqrt ((x - 0)^2 + (0 - y)^2)
  cd / ab = 2 / 5 := by
sorry


end NUMINAMATH_CALUDE_hyperbola_ratio_l3627_362779


namespace NUMINAMATH_CALUDE_coin_flip_heads_l3627_362726

theorem coin_flip_heads (total_flips : ℕ) (tail_head_diff : ℕ) (h_total : total_flips = 211) (h_diff : tail_head_diff = 81) :
  let heads := (total_flips - tail_head_diff) / 2
  heads = 65 := by
sorry

end NUMINAMATH_CALUDE_coin_flip_heads_l3627_362726


namespace NUMINAMATH_CALUDE_special_polygon_diagonals_l3627_362728

/-- A polygon with 10 vertices, where 4 vertices lie on a straight line
    and the remaining 6 form a regular hexagon. -/
structure SpecialPolygon where
  vertices : Fin 10
  line_vertices : Fin 4
  hexagon_vertices : Fin 6

/-- The number of diagonals in the special polygon. -/
def num_diagonals (p : SpecialPolygon) : ℕ := 33

/-- Theorem stating that the number of diagonals in the special polygon is 33. -/
theorem special_polygon_diagonals (p : SpecialPolygon) : num_diagonals p = 33 := by
  sorry

end NUMINAMATH_CALUDE_special_polygon_diagonals_l3627_362728


namespace NUMINAMATH_CALUDE_max_value_expression_l3627_362733

theorem max_value_expression (a b c d : ℝ) 
  (ha : -6.5 ≤ a ∧ a ≤ 6.5)
  (hb : -6.5 ≤ b ∧ b ≤ 6.5)
  (hc : -6.5 ≤ c ∧ c ≤ 6.5)
  (hd : -6.5 ≤ d ∧ d ≤ 6.5) :
  (a + 2*b + c + 2*d - a*b - b*c - c*d - d*a) ≤ 182 :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l3627_362733


namespace NUMINAMATH_CALUDE_tickets_sold_is_525_l3627_362774

/-- Represents the total number of tickets sold given ticket prices, total money collected, and number of general admission tickets. -/
def total_tickets_sold (student_price general_price total_collected general_tickets : ℕ) : ℕ :=
  let student_tickets := (total_collected - general_price * general_tickets) / student_price
  student_tickets + general_tickets

/-- Theorem stating that given the specific conditions, the total number of tickets sold is 525. -/
theorem tickets_sold_is_525 :
  total_tickets_sold 4 6 2876 388 = 525 := by
  sorry

end NUMINAMATH_CALUDE_tickets_sold_is_525_l3627_362774


namespace NUMINAMATH_CALUDE_log_less_than_zero_implies_x_between_zero_and_one_l3627_362701

theorem log_less_than_zero_implies_x_between_zero_and_one (x : ℝ) :
  (∃ (y : ℝ), y = Real.log x ∧ y < 0) → 0 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_log_less_than_zero_implies_x_between_zero_and_one_l3627_362701


namespace NUMINAMATH_CALUDE_equation_solution_l3627_362705

theorem equation_solution : ∃ x : ℝ, 0.6 * x + (0.2 * 0.4) = 0.56 ∧ x = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3627_362705


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3627_362788

theorem sum_of_fractions : 
  (19 / ((2^3 - 1) * (3^3 - 1)) + 
   37 / ((3^3 - 1) * (4^3 - 1)) + 
   61 / ((4^3 - 1) * (5^3 - 1)) + 
   91 / ((5^3 - 1) * (6^3 - 1))) = 208 / 1505 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3627_362788


namespace NUMINAMATH_CALUDE_tire_price_problem_l3627_362794

theorem tire_price_problem (regular_price : ℝ) : 
  (3 * regular_price + 10 = 310) → regular_price = 100 := by
  sorry

end NUMINAMATH_CALUDE_tire_price_problem_l3627_362794


namespace NUMINAMATH_CALUDE_cheerleader_count_l3627_362763

theorem cheerleader_count (size2 : ℕ) (size6 : ℕ) : 
  size2 = 4 → size6 = 10 → size2 + size6 + (size6 / 2) = 19 := by
  sorry

end NUMINAMATH_CALUDE_cheerleader_count_l3627_362763


namespace NUMINAMATH_CALUDE_fruit_basket_count_l3627_362799

/-- The number of possible fruit baskets with at least one piece of fruit -/
def num_fruit_baskets (num_apples : Nat) (num_oranges : Nat) : Nat :=
  (num_apples + 1) * (num_oranges + 1) - 1

/-- Theorem stating the number of fruit baskets with 7 apples and 12 oranges -/
theorem fruit_basket_count :
  num_fruit_baskets 7 12 = 103 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_count_l3627_362799


namespace NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l3627_362790

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, (n : ℝ) > Real.sqrt 500 ∧ ∀ m : ℕ, (m : ℝ) > Real.sqrt 500 → m ≥ n :=
  sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l3627_362790


namespace NUMINAMATH_CALUDE_marlon_gift_card_balance_l3627_362796

def gift_card_balance (initial_balance : ℝ) (monday_fraction : ℝ) (tuesday_fraction : ℝ) : ℝ :=
  let remaining_after_monday := initial_balance * (1 - monday_fraction)
  remaining_after_monday * (1 - tuesday_fraction)

theorem marlon_gift_card_balance :
  gift_card_balance 200 (1/2) (1/4) = 75 := by
  sorry

end NUMINAMATH_CALUDE_marlon_gift_card_balance_l3627_362796


namespace NUMINAMATH_CALUDE_maria_coin_count_l3627_362768

/-- Represents the number of stacks for each coin type -/
structure CoinStacks where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Represents the number of coins in each stack for each coin type -/
structure CoinsPerStack where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total number of coins given the number of stacks and coins per stack -/
def totalCoins (stacks : CoinStacks) (perStack : CoinsPerStack) : ℕ :=
  stacks.pennies * perStack.pennies +
  stacks.nickels * perStack.nickels +
  stacks.dimes * perStack.dimes

theorem maria_coin_count :
  let stacks : CoinStacks := { pennies := 3, nickels := 5, dimes := 7 }
  let perStack : CoinsPerStack := { pennies := 10, nickels := 8, dimes := 4 }
  totalCoins stacks perStack = 98 := by
  sorry

end NUMINAMATH_CALUDE_maria_coin_count_l3627_362768


namespace NUMINAMATH_CALUDE_sum_div_four_l3627_362704

theorem sum_div_four : (4 + 44 + 444) / 4 = 123 := by
  sorry

end NUMINAMATH_CALUDE_sum_div_four_l3627_362704


namespace NUMINAMATH_CALUDE_sin_five_alpha_identity_l3627_362756

theorem sin_five_alpha_identity (α : ℝ) : 
  16 * (Real.sin α)^5 - 20 * (Real.sin α)^3 + 5 * Real.sin α = Real.sin (5 * α) := by
  sorry

end NUMINAMATH_CALUDE_sin_five_alpha_identity_l3627_362756


namespace NUMINAMATH_CALUDE_roots_are_correct_all_roots_found_l3627_362785

/-- The roots of the equation 5x^4 - 28x^3 + 49x^2 - 28x + 5 = 0 -/
def roots : Set ℝ :=
  {2, 1/2, (5 + Real.sqrt 21)/5, (5 - Real.sqrt 21)/5}

/-- The polynomial function corresponding to the equation -/
def f (x : ℝ) : ℝ := 5*x^4 - 28*x^3 + 49*x^2 - 28*x + 5

theorem roots_are_correct : ∀ x ∈ roots, f x = 0 := by
  sorry

theorem all_roots_found : ∀ x, f x = 0 → x ∈ roots := by
  sorry

end NUMINAMATH_CALUDE_roots_are_correct_all_roots_found_l3627_362785


namespace NUMINAMATH_CALUDE_point_on_exponential_graph_l3627_362731

theorem point_on_exponential_graph (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f := fun x => a^(x - 1)
  f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_point_on_exponential_graph_l3627_362731


namespace NUMINAMATH_CALUDE_square_of_six_y_minus_two_l3627_362773

theorem square_of_six_y_minus_two (y : ℝ) (h : 3 * y^2 + 6 = 2 * y + 10) : (6 * y - 2)^2 = 52 := by
  sorry

end NUMINAMATH_CALUDE_square_of_six_y_minus_two_l3627_362773


namespace NUMINAMATH_CALUDE_minimum_value_problem_l3627_362700

theorem minimum_value_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 6) : 
  (9 / x) + (25 / y) + (49 / z) ≥ 37.5 ∧ 
  ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ x' + y' + z' = 6 ∧ 
    (9 / x') + (25 / y') + (49 / z') = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_problem_l3627_362700


namespace NUMINAMATH_CALUDE_composition_equality_l3627_362786

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 5*x + a
def g (a : ℝ) (x : ℝ) : ℝ := a*x^2 + 1

-- State the theorem
theorem composition_equality (a : ℝ) :
  ∃ b : ℝ, ∀ x : ℝ, f a (g a x) = a^2*x^4 + 5*a*x^2 + b → b = 6 + a :=
by sorry

end NUMINAMATH_CALUDE_composition_equality_l3627_362786


namespace NUMINAMATH_CALUDE_not_right_angled_triangle_l3627_362713

theorem not_right_angled_triangle (A B C : ℝ) (h1 : A = B) (h2 : A = 2 * C) 
  (h3 : A + B + C = 180) : A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := by
  sorry

#check not_right_angled_triangle

end NUMINAMATH_CALUDE_not_right_angled_triangle_l3627_362713


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l3627_362708

open Set

-- Define the universal set U as the real numbers
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | 2 - x > 0}

-- Define set N
def N : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- Define the result set
def result : Set ℝ := {x | 2 ≤ x ∧ x < 3}

-- Theorem statement
theorem set_intersection_theorem : (U \ M) ∩ N = result := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l3627_362708


namespace NUMINAMATH_CALUDE_bug_position_after_2010_jumps_l3627_362717

/-- Represents the points on the circle -/
inductive Point
| one
| two
| three
| four
| five

/-- Determines if a point is odd-numbered -/
def is_odd (p : Point) : Bool :=
  match p with
  | Point.one => true
  | Point.two => false
  | Point.three => true
  | Point.four => false
  | Point.five => true

/-- Calculates the next point based on the current point -/
def next_point (p : Point) : Point :=
  match p with
  | Point.one => Point.three
  | Point.two => Point.five
  | Point.three => Point.five
  | Point.four => Point.two
  | Point.five => Point.two

/-- Calculates the point after n jumps -/
def point_after_jumps (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | Nat.succ m => next_point (point_after_jumps start m)

theorem bug_position_after_2010_jumps :
  point_after_jumps Point.five 2010 = Point.two :=
sorry

end NUMINAMATH_CALUDE_bug_position_after_2010_jumps_l3627_362717


namespace NUMINAMATH_CALUDE_ryan_study_difference_l3627_362782

/-- Ryan's daily study schedule -/
structure StudySchedule where
  english_hours : ℕ
  chinese_hours : ℕ

/-- The difference in hours between English and Chinese study time -/
def study_time_difference (schedule : StudySchedule) : ℤ :=
  schedule.english_hours - schedule.chinese_hours

/-- Theorem: Ryan spends 4 more hours on English than Chinese -/
theorem ryan_study_difference :
  ∀ (schedule : StudySchedule),
  schedule.english_hours = 6 →
  schedule.chinese_hours = 2 →
  study_time_difference schedule = 4 := by
sorry

end NUMINAMATH_CALUDE_ryan_study_difference_l3627_362782


namespace NUMINAMATH_CALUDE_initial_meals_correct_l3627_362784

/-- The number of meals Colt and Curt initially prepared -/
def initial_meals : ℕ := 113

/-- The number of meals Sole Mart provided -/
def sole_mart_meals : ℕ := 50

/-- The number of meals given away -/
def meals_given_away : ℕ := 85

/-- The number of meals left to be distributed -/
def meals_left : ℕ := 78

/-- Theorem stating that the initial number of meals is correct -/
theorem initial_meals_correct : 
  initial_meals + sole_mart_meals = meals_given_away + meals_left := by
  sorry

end NUMINAMATH_CALUDE_initial_meals_correct_l3627_362784


namespace NUMINAMATH_CALUDE_contractor_engagement_days_l3627_362758

/-- Proves that the contractor was engaged for 20 days given the problem conditions --/
theorem contractor_engagement_days : 
  ∀ (daily_wage : ℚ) (daily_fine : ℚ) (total_amount : ℚ) (absent_days : ℕ),
    daily_wage = 25 →
    daily_fine = (15/2) →
    total_amount = 425 →
    absent_days = 10 →
    ∃ (engaged_days : ℕ), 
      engaged_days * daily_wage - absent_days * daily_fine = total_amount ∧
      engaged_days = 20 := by
  sorry

end NUMINAMATH_CALUDE_contractor_engagement_days_l3627_362758


namespace NUMINAMATH_CALUDE_angle_POQ_is_72_degrees_l3627_362795

-- Define the regular pentagon
structure RegularPentagon where
  side_length : ℝ
  internal_angle : ℝ
  internal_angle_eq : internal_angle = 108

-- Define the inscribed circle
structure InscribedCircle (p : RegularPentagon) where
  center : Point
  radius : ℝ
  tangent_point1 : Point
  tangent_point2 : Point
  corner : Point
  is_tangent : Bool
  intersects_other_sides : Bool

-- Define the angle POQ
def angle_POQ (p : RegularPentagon) (c : InscribedCircle p) : ℝ :=
  sorry

-- Define the bisector property
def is_bisector (p : RegularPentagon) (c : InscribedCircle p) : Prop :=
  sorry

-- Theorem statement
theorem angle_POQ_is_72_degrees 
  (p : RegularPentagon) 
  (c : InscribedCircle p) 
  (h1 : c.is_tangent = true) 
  (h2 : c.intersects_other_sides = true) 
  (h3 : is_bisector p c) : 
  angle_POQ p c = 72 := by
  sorry

end NUMINAMATH_CALUDE_angle_POQ_is_72_degrees_l3627_362795


namespace NUMINAMATH_CALUDE_expression_value_l3627_362772

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 2) :
  3 * x - 4 * y + 2 * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3627_362772


namespace NUMINAMATH_CALUDE_sequence_existence_l3627_362754

theorem sequence_existence (n : ℕ) (hn : n ≥ 3) :
  (∃ (a : ℕ → ℝ), 
    (a 1 = a (n + 1)) ∧ 
    (a 2 = a (n + 2)) ∧ 
    (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i * a (i + 1) + 1 = a (i + 2)))
  ↔ 
  (∃ k : ℕ, n = 3 * k) :=
by sorry

end NUMINAMATH_CALUDE_sequence_existence_l3627_362754


namespace NUMINAMATH_CALUDE_base7_divisibility_l3627_362744

/-- Converts a base-7 number of the form 3dd6_7 to base 10 -/
def base7ToBase10 (d : ℕ) : ℕ := 3 * 7^3 + d * 7^2 + d * 7 + 6

/-- Checks if a number is a valid base-7 digit -/
def isValidBase7Digit (d : ℕ) : Prop := d ≤ 6

theorem base7_divisibility :
  ∀ d : ℕ, isValidBase7Digit d → (base7ToBase10 d % 13 = 0 ↔ d = 4) :=
by sorry

end NUMINAMATH_CALUDE_base7_divisibility_l3627_362744


namespace NUMINAMATH_CALUDE_exists_unique_marking_scheme_l3627_362781

/-- Represents a cell in the grid -/
structure Cell :=
  (row : Nat)
  (col : Nat)

/-- Represents a marking scheme for the grid -/
def MarkingScheme := Set Cell

/-- Represents a 10x10 sub-square in the grid -/
structure SubSquare :=
  (topLeft : Cell)

/-- Counts the number of marked cells in a sub-square -/
def countMarkedCells (scheme : MarkingScheme) (square : SubSquare) : Nat :=
  sorry

/-- Checks if all sub-squares have unique counts -/
def allSubSquaresUnique (scheme : MarkingScheme) : Prop :=
  sorry

/-- Main theorem: There exists a marking scheme where all sub-squares have unique counts -/
theorem exists_unique_marking_scheme :
  ∃ (scheme : MarkingScheme),
    (∀ c : Cell, c.row < 19 ∧ c.col < 19) →
    (∀ s : SubSquare, s.topLeft.row ≤ 9 ∧ s.topLeft.col ≤ 9) →
    allSubSquaresUnique scheme :=
  sorry

end NUMINAMATH_CALUDE_exists_unique_marking_scheme_l3627_362781


namespace NUMINAMATH_CALUDE_gpa_probability_at_least_3_6_l3627_362746

/-- Represents the possible grades a student can receive. -/
inductive Grade
| A
| B
| C
| D

/-- Converts a grade to its point value. -/
def gradeToPoints (g : Grade) : ℕ :=
  match g with
  | Grade.A => 4
  | Grade.B => 3
  | Grade.C => 2
  | Grade.D => 1

/-- Calculates the GPA given a list of grades. -/
def calculateGPA (grades : List Grade) : ℚ :=
  (grades.map gradeToPoints).sum / 5

/-- Represents the probability distribution of grades for a class. -/
structure GradeProbability where
  probA : ℚ
  probB : ℚ
  probC : ℚ
  probD : ℚ

/-- The probability distribution for English grades. -/
def englishProb : GradeProbability :=
  { probA := 1/4, probB := 1/3, probC := 5/12, probD := 0 }

/-- The probability distribution for History grades. -/
def historyProb : GradeProbability :=
  { probA := 1/3, probB := 1/4, probC := 5/12, probD := 0 }

/-- Theorem stating the probability of achieving a GPA of at least 3.6. -/
theorem gpa_probability_at_least_3_6 :
  let allGrades := [Grade.A, Grade.A, Grade.A] -- Math, Science, Art
  let probAtLeast3_6 := (
    englishProb.probA * historyProb.probA +
    englishProb.probA * historyProb.probB +
    englishProb.probB * historyProb.probA +
    englishProb.probB * historyProb.probB
  )
  probAtLeast3_6 = 49/144 := by sorry

end NUMINAMATH_CALUDE_gpa_probability_at_least_3_6_l3627_362746


namespace NUMINAMATH_CALUDE_fraction_sum_lower_bound_l3627_362723

theorem fraction_sum_lower_bound (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / (y + z) + y / (z + x) + z / (x + y) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_lower_bound_l3627_362723


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3627_362793

theorem polynomial_simplification (x : ℝ) : (3 * x^2 - 4 * x + 5) - (2 * x^2 - 6 * x - 8) = x^2 + 2 * x + 13 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3627_362793


namespace NUMINAMATH_CALUDE_probability_with_replacement_probability_without_replacement_l3627_362753

-- Define the number of disco and classical cassettes
def disco_cassettes : ℕ := 12
def classical_cassettes : ℕ := 18

-- Define the total number of cassettes
def total_cassettes : ℕ := disco_cassettes + classical_cassettes

-- Theorem for the probability with replacement
theorem probability_with_replacement :
  (disco_cassettes : ℚ) / total_cassettes * (disco_cassettes : ℚ) / total_cassettes = 4 / 25 :=
sorry

-- Theorem for the probability without replacement
theorem probability_without_replacement :
  (disco_cassettes : ℚ) / total_cassettes * ((disco_cassettes - 1) : ℚ) / (total_cassettes - 1) = 22 / 145 :=
sorry

end NUMINAMATH_CALUDE_probability_with_replacement_probability_without_replacement_l3627_362753


namespace NUMINAMATH_CALUDE_problem_solution_l3627_362798

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem problem_solution :
  (B (1/5) ⊂ A) ∧
  ({a : ℝ | A ∩ B a = B a} = {0, 1/3, 1/5}) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3627_362798


namespace NUMINAMATH_CALUDE_rational_terms_not_adjacent_probability_l3627_362743

theorem rational_terms_not_adjacent_probability :
  let total_terms : ℕ := 9
  let rational_terms : ℕ := 3
  let irrational_terms : ℕ := 6
  let total_arrangements := Nat.factorial total_terms
  let favorable_arrangements := Nat.factorial irrational_terms * (Nat.factorial (irrational_terms + 1)).choose rational_terms
  (favorable_arrangements : ℚ) / total_arrangements = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_rational_terms_not_adjacent_probability_l3627_362743


namespace NUMINAMATH_CALUDE_final_jellybeans_count_l3627_362741

-- Define the initial number of jellybeans
def initial_jellybeans : ℕ := 90

-- Define the number of jellybeans Samantha took
def samantha_took : ℕ := 24

-- Define the number of jellybeans Shelby ate
def shelby_ate : ℕ := 12

-- Define the function to calculate the final number of jellybeans
def final_jellybeans : ℕ :=
  initial_jellybeans - (samantha_took + shelby_ate) + (samantha_took + shelby_ate) / 2

-- Theorem statement
theorem final_jellybeans_count : final_jellybeans = 72 := by
  sorry

end NUMINAMATH_CALUDE_final_jellybeans_count_l3627_362741


namespace NUMINAMATH_CALUDE_smallest_a_value_l3627_362742

/-- Represents a parabola with equation y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem stating the smallest possible value of a for the given parabola -/
theorem smallest_a_value (p : Parabola) 
  (vertex_x : p.a * (1/3)^2 + p.b * (1/3) + p.c = -5/9) 
  (a_positive : p.a > 0)
  (sum_integer : ∃ n : ℤ, p.a + p.b + p.c = n) :
  p.a ≥ 5/4 ∧ ∃ (q : Parabola), q.a = 5/4 ∧ 
    q.a * (1/3)^2 + q.b * (1/3) + q.c = -5/9 ∧ 
    q.a > 0 ∧ 
    (∃ n : ℤ, q.a + q.b + q.c = n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_value_l3627_362742


namespace NUMINAMATH_CALUDE_football_practice_hours_l3627_362702

/-- Calculates the daily practice hours for a football team -/
def dailyPracticeHours (totalHours weekDays missedDays : ℕ) : ℚ :=
  totalHours / (weekDays - missedDays)

/-- Proves that the football team practices 5 hours daily -/
theorem football_practice_hours :
  let totalHours : ℕ := 30
  let weekDays : ℕ := 7
  let missedDays : ℕ := 1
  dailyPracticeHours totalHours weekDays missedDays = 5 := by
sorry

end NUMINAMATH_CALUDE_football_practice_hours_l3627_362702


namespace NUMINAMATH_CALUDE_theme_park_youngest_child_age_l3627_362711

theorem theme_park_youngest_child_age (father_charge : ℝ) (age_cost : ℝ) (total_cost : ℝ) :
  father_charge = 6.5 →
  age_cost = 0.55 →
  total_cost = 15.95 →
  ∃ (twin_age : ℕ) (youngest_age : ℕ),
    youngest_age < twin_age ∧
    youngest_age + 4 * twin_age = 17 ∧
    (youngest_age = 1 ∨ youngest_age = 5) :=
by sorry

end NUMINAMATH_CALUDE_theme_park_youngest_child_age_l3627_362711


namespace NUMINAMATH_CALUDE_sector_circumradius_l3627_362780

theorem sector_circumradius (r : ℝ) (θ : ℝ) (h1 : r = 8) (h2 : θ = 2 * π / 3) :
  let R := r / (2 * Real.sin (θ / 2))
  R = 8 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_sector_circumradius_l3627_362780


namespace NUMINAMATH_CALUDE_original_ratio_l3627_362751

theorem original_ratio (x y : ℝ) (h1 : y = 40) (h2 : (x + 10) / (y + 10) = 4/5) :
  x / y = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_original_ratio_l3627_362751


namespace NUMINAMATH_CALUDE_charles_whistle_count_l3627_362797

/-- The number of whistles Sean has -/
def sean_whistles : ℕ := 45

/-- The difference between Sean's and Charles' whistles -/
def whistle_difference : ℕ := 32

/-- The number of whistles Charles has -/
def charles_whistles : ℕ := sean_whistles - whistle_difference

theorem charles_whistle_count : charles_whistles = 13 := by
  sorry

end NUMINAMATH_CALUDE_charles_whistle_count_l3627_362797


namespace NUMINAMATH_CALUDE_pipe_filling_time_l3627_362727

/-- Proves that Pipe A takes 20 minutes to fill the tank alone given the conditions -/
theorem pipe_filling_time (t : ℝ) : 
  t > 0 →  -- Pipe A fills the tank in t minutes (t must be positive)
  (t / 4 > 0) →  -- Pipe B fills the tank in t/4 minutes (t/4 must be positive)
  (1 / t + 1 / (t / 4) = 1 / 4) →  -- When both pipes are open, it takes 4 minutes to fill the tank
  t = 20 := by
sorry


end NUMINAMATH_CALUDE_pipe_filling_time_l3627_362727


namespace NUMINAMATH_CALUDE_complementary_angles_sum_l3627_362776

theorem complementary_angles_sum (a b : ℝ) : 
  a > 0 → b > 0 → a / b = 3 / 5 → a + b = 90 → a + b = 90 := by sorry

end NUMINAMATH_CALUDE_complementary_angles_sum_l3627_362776


namespace NUMINAMATH_CALUDE_keanu_refills_l3627_362769

/-- Calculates the number of refills needed for a round trip given the tank capacity, fuel consumption rate, and one-way distance. -/
def refills_needed (tank_capacity : ℚ) (consumption_per_40_miles : ℚ) (one_way_distance : ℚ) : ℚ :=
  let consumption_per_mile := consumption_per_40_miles / 40
  let round_trip_distance := one_way_distance * 2
  let total_consumption := round_trip_distance * consumption_per_mile
  (total_consumption / tank_capacity).ceil

/-- Theorem stating that for the given conditions, 14 refills are needed. -/
theorem keanu_refills :
  refills_needed 8 8 280 = 14 := by
  sorry

end NUMINAMATH_CALUDE_keanu_refills_l3627_362769


namespace NUMINAMATH_CALUDE_monotonic_increasing_sufficient_not_necessary_l3627_362747

-- Define a monotonically increasing function on ℝ
def MonotonicIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x ≤ f y

-- Define the existence of x₁ < x₂ such that f(x₁) < f(x₂)
def ExistsStrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ < f x₂

-- Theorem stating that monotonic increasing is sufficient but not necessary
-- for the existence of strictly increasing points
theorem monotonic_increasing_sufficient_not_necessary (f : ℝ → ℝ) :
  (MonotonicIncreasing f → ExistsStrictlyIncreasing f) ∧
  ¬(ExistsStrictlyIncreasing f → MonotonicIncreasing f) :=
sorry

end NUMINAMATH_CALUDE_monotonic_increasing_sufficient_not_necessary_l3627_362747


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_l3627_362787

/-- Given a quadratic expression 3x^2 + 9x - 24, when written in the form a(x - h)^2 + k, h = -1.5 -/
theorem quadratic_vertex_form (x : ℝ) : 
  ∃ (a k : ℝ), 3*x^2 + 9*x - 24 = a*(x - (-1.5))^2 + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_l3627_362787


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l3627_362752

theorem polynomial_multiplication :
  ∀ x : ℝ, (7 * x^2 + 3 * x + 1) * (5 * x^3 + 2 * x + 6) = 
    35 * x^5 + 15 * x^4 + 19 * x^3 + 48 * x^2 + 20 * x + 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l3627_362752


namespace NUMINAMATH_CALUDE_complex_equation_implies_sum_l3627_362725

def complex (a b : ℝ) : ℂ := Complex.mk a b

theorem complex_equation_implies_sum (a b : ℝ) :
  complex 9 3 * complex a b = complex 10 4 →
  a + b = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_implies_sum_l3627_362725


namespace NUMINAMATH_CALUDE_square_sum_equality_l3627_362712

theorem square_sum_equality : (-2)^2 + 2^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l3627_362712


namespace NUMINAMATH_CALUDE_iron_content_calculation_l3627_362771

theorem iron_content_calculation (initial_mass : ℝ) (impurities_mass : ℝ) 
  (impurities_iron_percent : ℝ) (iron_content_increase : ℝ) :
  initial_mass = 500 →
  impurities_mass = 200 →
  impurities_iron_percent = 12.5 →
  iron_content_increase = 20 →
  ∃ (remaining_iron : ℝ),
    remaining_iron = 187.5 ∧
    remaining_iron = 
      (initial_mass * ((impurities_mass * impurities_iron_percent / 100) / 
      (initial_mass - impurities_mass) + iron_content_increase / 100) / 100) * 
      (initial_mass - impurities_mass) -
      (impurities_mass * impurities_iron_percent / 100) := by
  sorry

end NUMINAMATH_CALUDE_iron_content_calculation_l3627_362771


namespace NUMINAMATH_CALUDE_intersection_P_Q_l3627_362767

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x * (x - 1) ≥ 0}
def Q : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 3 * x^2 + 1}

-- Theorem statement
theorem intersection_P_Q : P ∩ Q = {x : ℝ | x > 1} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l3627_362767


namespace NUMINAMATH_CALUDE_square_carpet_side_length_l3627_362703

theorem square_carpet_side_length (area : ℝ) (h : area = 10) :
  ∃ (side : ℝ), side * side = area ∧ 3 < side ∧ side < 4 := by
  sorry

end NUMINAMATH_CALUDE_square_carpet_side_length_l3627_362703


namespace NUMINAMATH_CALUDE_vlad_height_l3627_362755

/-- Proves that Vlad is 3 inches taller than 6 feet given the conditions of the problem -/
theorem vlad_height (vlad_feet : ℕ) (vlad_inches : ℕ) (sister_feet : ℕ) (sister_inches : ℕ) 
  (height_difference : ℕ) :
  vlad_feet = 6 →
  sister_feet = 2 →
  sister_inches = 10 →
  height_difference = 41 →
  vlad_inches = 3 :=
by
  sorry

#check vlad_height

end NUMINAMATH_CALUDE_vlad_height_l3627_362755


namespace NUMINAMATH_CALUDE_delegates_without_badges_l3627_362762

theorem delegates_without_badges (total : Nat) (preprinted : Nat) : 
  total = 36 → preprinted = 16 → (total - preprinted - (total - preprinted) / 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_delegates_without_badges_l3627_362762


namespace NUMINAMATH_CALUDE_function_property_l3627_362783

-- Define the functions
def f1 (x : ℝ) := |x|
def f2 (x : ℝ) := x - |x|
def f3 (x : ℝ) := x + 1
def f4 (x : ℝ) := -x

-- Define the property we're checking
def satisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 * x) = 2 * f x

-- Theorem statement
theorem function_property :
  satisfiesProperty f1 ∧
  satisfiesProperty f2 ∧
  ¬satisfiesProperty f3 ∧
  satisfiesProperty f4 :=
sorry

end NUMINAMATH_CALUDE_function_property_l3627_362783


namespace NUMINAMATH_CALUDE_smallest_y_with_24_factors_l3627_362761

theorem smallest_y_with_24_factors (y : ℕ) 
  (h1 : (Nat.divisors y).card = 24)
  (h2 : 20 ∣ y)
  (h3 : 35 ∣ y) :
  y ≥ 1120 ∧ ∃ (z : ℕ), z ≥ 1120 ∧ (Nat.divisors z).card = 24 ∧ 20 ∣ z ∧ 35 ∣ z :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_with_24_factors_l3627_362761


namespace NUMINAMATH_CALUDE_round_robin_tournament_l3627_362739

theorem round_robin_tournament (n : ℕ) (h : n * (n - 1) / 2 = 15) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_round_robin_tournament_l3627_362739


namespace NUMINAMATH_CALUDE_power_calculation_l3627_362775

theorem power_calculation : (2 : ℝ)^2021 * (-1/2 : ℝ)^2022 = 1/2 := by sorry

end NUMINAMATH_CALUDE_power_calculation_l3627_362775


namespace NUMINAMATH_CALUDE_restaurant_hiring_l3627_362740

/-- Given a restaurant with cooks and waiters, prove the number of newly hired waiters. -/
theorem restaurant_hiring (initial_cooks initial_waiters new_waiters : ℕ) : 
  initial_cooks * 11 = initial_waiters * 3 →  -- Initial ratio of cooks to waiters is 3:11
  initial_cooks * 5 = (initial_waiters + new_waiters) * 1 →  -- New ratio is 1:5
  initial_cooks = 9 →  -- There are 9 cooks
  new_waiters = 12 :=  -- Prove that 12 waiters were hired
by sorry

end NUMINAMATH_CALUDE_restaurant_hiring_l3627_362740


namespace NUMINAMATH_CALUDE_min_value_of_f_l3627_362707

-- Define the function f(x)
def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

-- State the theorem
theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y m ≤ f x m) ∧  -- f has a maximum on [-2, 2]
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 3) →                               -- The maximum value is 3
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ f y m ∧ f x m = -37) :=  -- The minimum value is -37
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3627_362707


namespace NUMINAMATH_CALUDE_product_of_max_min_a_l3627_362716

theorem product_of_max_min_a (a b c : ℝ) 
  (sum_eq : a + b + c = 15) 
  (sum_squares_eq : a^2 + b^2 + c^2 = 100) : 
  let f := fun x : ℝ => (5 + (5 * Real.sqrt 6) / 3) * (5 - (5 * Real.sqrt 6) / 3)
  f a = 25 / 3 := by
sorry

end NUMINAMATH_CALUDE_product_of_max_min_a_l3627_362716


namespace NUMINAMATH_CALUDE_factor_implies_sum_l3627_362719

theorem factor_implies_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 - 2*X + 5) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) →
  P + Q = 31 :=
by sorry

end NUMINAMATH_CALUDE_factor_implies_sum_l3627_362719


namespace NUMINAMATH_CALUDE_no_valid_decagon_labeling_l3627_362760

/-- Represents a labeling of a regular decagon with center -/
def DecagonLabeling := Fin 11 → Fin 10

/-- The sum of digits on a line through the center of the decagon -/
def line_sum (l : DecagonLabeling) (i j : Fin 11) : ℕ :=
  l i + l j + l 10

/-- Checks if a labeling is valid according to the problem constraints -/
def is_valid_labeling (l : DecagonLabeling) : Prop :=
  (∀ i j : Fin 11, i ≠ j → l i ≠ l j) ∧
  (line_sum l 0 4 = line_sum l 1 5) ∧
  (line_sum l 0 4 = line_sum l 2 6) ∧
  (line_sum l 0 4 = line_sum l 3 7) ∧
  (line_sum l 0 4 = line_sum l 4 8)

theorem no_valid_decagon_labeling :
  ¬∃ l : DecagonLabeling, is_valid_labeling l :=
sorry

end NUMINAMATH_CALUDE_no_valid_decagon_labeling_l3627_362760


namespace NUMINAMATH_CALUDE_max_value_fraction_l3627_362734

theorem max_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x * y / (4 * x + 9 * y) ≤ a * b / (4 * a + 9 * b)) → 
  a * b / (4 * a + 9 * b) = 1 / 25 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l3627_362734
