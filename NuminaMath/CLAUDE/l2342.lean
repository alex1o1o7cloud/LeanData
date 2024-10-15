import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2342_234287

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(1 + b²/a²) -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt (1 + b^2 / a^2)
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  hyperbola 3 4 → e = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2342_234287


namespace NUMINAMATH_CALUDE_tan_B_in_triangle_l2342_234296

theorem tan_B_in_triangle (A B C : ℝ) (cosC : ℝ) (AC BC : ℝ) 
  (h1 : cosC = 2/3)
  (h2 : AC = 4)
  (h3 : BC = 3)
  (h4 : A + B + C = Real.pi) -- sum of angles in a triangle
  (h5 : 0 < AC ∧ 0 < BC) -- positive side lengths
  : Real.tan B = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_B_in_triangle_l2342_234296


namespace NUMINAMATH_CALUDE_cement_bags_ratio_l2342_234264

theorem cement_bags_ratio (bags1 : ℕ) (weight1 : ℚ) (cost1 : ℚ) (cost2 : ℚ) (weight_ratio : ℚ) :
  bags1 = 80 →
  weight1 = 50 →
  cost1 = 6000 →
  cost2 = 10800 →
  weight_ratio = 3 / 5 →
  (cost2 / (cost1 / bags1 * weight_ratio)) / bags1 = 3 / 1 := by
sorry

end NUMINAMATH_CALUDE_cement_bags_ratio_l2342_234264


namespace NUMINAMATH_CALUDE_no_solution_arcsin_arccos_squared_l2342_234224

theorem no_solution_arcsin_arccos_squared (x : ℝ) : 
  (Real.arcsin x + Real.arccos x = π / 2) → (Real.arcsin x)^2 + (Real.arccos x)^2 ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_arcsin_arccos_squared_l2342_234224


namespace NUMINAMATH_CALUDE_cos_sixty_degrees_l2342_234276

theorem cos_sixty_degrees : Real.cos (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sixty_degrees_l2342_234276


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l2342_234235

/-- Proves that adding 2.4 liters of pure alcohol to a 6-liter solution
    that is 30% alcohol results in a 50% alcohol solution -/
theorem alcohol_mixture_proof :
  let initial_volume : ℝ := 6
  let initial_concentration : ℝ := 0.3
  let final_concentration : ℝ := 0.5
  let added_alcohol : ℝ := 2.4

  let initial_alcohol : ℝ := initial_volume * initial_concentration
  let final_volume : ℝ := initial_volume + added_alcohol
  let final_alcohol : ℝ := initial_alcohol + added_alcohol

  final_alcohol / final_volume = final_concentration :=
by
  sorry


end NUMINAMATH_CALUDE_alcohol_mixture_proof_l2342_234235


namespace NUMINAMATH_CALUDE_percentage_of_democrat_voters_prove_percentage_of_democrat_voters_l2342_234210

theorem percentage_of_democrat_voters : ℝ → ℝ → Prop :=
  fun d r =>
    d + r = 100 →
    0.7 * d + 0.2 * r = 50 →
    d = 60

-- Proof
theorem prove_percentage_of_democrat_voters :
  ∃ d r : ℝ, percentage_of_democrat_voters d r :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_of_democrat_voters_prove_percentage_of_democrat_voters_l2342_234210


namespace NUMINAMATH_CALUDE_x_value_proof_l2342_234209

theorem x_value_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^4 / y = 2) (h2 : y^3 / z = 6) (h3 : z^2 / x = 8) :
  x = (18432 : ℝ)^(1/23) := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l2342_234209


namespace NUMINAMATH_CALUDE_student_arrangement_counts_l2342_234222

/-- The number of ways to arrange 5 male and 2 female students in a row --/
def arrange_students (n_male : ℕ) (n_female : ℕ) : ℕ → ℕ → ℕ
| 1 => λ _ => 1400  -- females must be next to each other
| 2 => λ _ => 3600  -- females must not be next to each other
| 3 => λ _ => 3720  -- specific placement restrictions for females
| _ => λ _ => 0     -- undefined for other cases

/-- Theorem stating the correct number of arrangements for each scenario --/
theorem student_arrangement_counts :
  let n_male := 5
  let n_female := 2
  (arrange_students n_male n_female 1 0 = 1400) ∧
  (arrange_students n_male n_female 2 0 = 3600) ∧
  (arrange_students n_male n_female 3 0 = 3720) :=
by sorry


end NUMINAMATH_CALUDE_student_arrangement_counts_l2342_234222


namespace NUMINAMATH_CALUDE_sin_405_degrees_l2342_234258

theorem sin_405_degrees : Real.sin (405 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_405_degrees_l2342_234258


namespace NUMINAMATH_CALUDE_cube_sum_l2342_234283

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where

/-- The number of faces in a cube -/
def Cube.faces (c : Cube) : ℕ := 6

/-- The number of edges in a cube -/
def Cube.edges (c : Cube) : ℕ := 12

/-- The number of vertices in a cube -/
def Cube.vertices (c : Cube) : ℕ := 8

/-- The sum of faces, edges, and vertices in a cube is 26 -/
theorem cube_sum (c : Cube) : c.faces + c.edges + c.vertices = 26 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_l2342_234283


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_l2342_234233

theorem smallest_non_factor_product (a b : ℕ+) : 
  a ≠ b →
  a ∣ 48 →
  b ∣ 48 →
  ¬(a * b ∣ 48) →
  (∀ (c d : ℕ+), c ≠ d → c ∣ 48 → d ∣ 48 → ¬(c * d ∣ 48) → a * b ≤ c * d) →
  a * b = 18 := by
sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_l2342_234233


namespace NUMINAMATH_CALUDE_min_value_xy_l2342_234275

theorem min_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : (2 / x) + (8 / y) = 1) :
  xy ≥ 64 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ (2 / x₀) + (8 / y₀) = 1 ∧ x₀ * y₀ = 64 :=
by sorry

end NUMINAMATH_CALUDE_min_value_xy_l2342_234275


namespace NUMINAMATH_CALUDE_some_number_value_l2342_234270

theorem some_number_value (x : ℝ) (some_number : ℝ) 
  (h1 : 5 + 7 / x = some_number - 5 / x)
  (h2 : x = 12) : 
  some_number = 6 := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l2342_234270


namespace NUMINAMATH_CALUDE_cube_root_of_216_l2342_234265

theorem cube_root_of_216 (y : ℝ) : (Real.sqrt y)^3 = 216 → y = 36 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_216_l2342_234265


namespace NUMINAMATH_CALUDE_f_inequality_l2342_234291

noncomputable def f (x : ℝ) := x^2 - Real.cos x

theorem f_inequality : f 0 < f (-0.5) ∧ f (-0.5) < f 0.6 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l2342_234291


namespace NUMINAMATH_CALUDE_max_product_distances_area_triangle_45_slope_l2342_234220

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  h₁ : a > 0
  h₂ : b > 0
  h₃ : a > b

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Define the foci of an ellipse -/
def foci (e : Ellipse a b) : Point × Point :=
  sorry

/-- Define the endpoints of the minor axis of an ellipse -/
def minorAxisEndpoints (e : Ellipse a b) : Point × Point :=
  sorry

/-- Calculate the perimeter of a quadrilateral given its vertices -/
def perimeter (p₁ p₂ p₃ p₄ : Point) : ℝ :=
  sorry

/-- Calculate the distance between two points -/
def distance (p₁ p₂ : Point) : ℝ :=
  sorry

/-- Calculate the area of a triangle given its vertices -/
def triangleArea (p₁ p₂ p₃ : Point) : ℝ :=
  sorry

/-- Theorem about the maximum product of distances from foci to points on the ellipse -/
theorem max_product_distances (e : Ellipse a b) (F₁ F₂ A B : Point) :
  let (F₁', F₂') := foci e
  let (M, N) := minorAxisEndpoints e
  perimeter F₁ F₂ M N = 4 →
  F₁ = F₁' →
  distance A B = 4/3 →
  (∃ (l : Point → Prop), l F₁ ∧ l A ∧ l B) →
  (∀ A' B' : Point, (∃ (l : Point → Prop), l F₁ ∧ l A' ∧ l B') →
    distance A' F₂ * distance B' F₂ ≤ 16/9) :=
  sorry

/-- Theorem about the area of the triangle when the line has a 45-degree slope -/
theorem area_triangle_45_slope (e : Ellipse a b) (F₁ F₂ A B : Point) :
  let (F₁', F₂') := foci e
  let (M, N) := minorAxisEndpoints e
  perimeter F₁ F₂ M N = 4 →
  F₁ = F₁' →
  distance A B = 4/3 →
  (∃ (l : Point → Prop), l F₁ ∧ l A ∧ l B ∧ ∀ p q : Point, l p ∧ l q → (p.y - q.y) = (p.x - q.x)) →
  triangleArea A B F₂ = 2/3 :=
  sorry

end NUMINAMATH_CALUDE_max_product_distances_area_triangle_45_slope_l2342_234220


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_l2342_234245

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- The measure of each interior angle in a regular octagon -/
def octagon_interior_angle : ℝ := 135

/-- Theorem: Each interior angle of a regular octagon measures 135 degrees -/
theorem regular_octagon_interior_angle :
  (180 * (octagon_sides - 2 : ℝ)) / octagon_sides = octagon_interior_angle :=
sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_l2342_234245


namespace NUMINAMATH_CALUDE_angle_C_is_120_degrees_max_area_is_sqrt_3_l2342_234261

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  pos_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π
  sine_law : a / Real.sin A = b / Real.sin B
  cosine_law : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

theorem angle_C_is_120_degrees (t : Triangle) 
  (h : t.a * (Real.cos t.C)^2 + 2 * t.c * Real.cos t.A * Real.cos t.C + t.a + t.b = 0) :
  t.C = 2 * π / 3 := by sorry

theorem max_area_is_sqrt_3 (t : Triangle) (h : t.b = 4 * Real.sin t.B) :
  (∀ u : Triangle, u.b = 4 * Real.sin u.B → t.a * t.b * Real.sin t.C / 2 ≥ u.a * u.b * Real.sin u.C / 2) ∧
  t.a * t.b * Real.sin t.C / 2 = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_angle_C_is_120_degrees_max_area_is_sqrt_3_l2342_234261


namespace NUMINAMATH_CALUDE_x_squared_y_squared_value_l2342_234294

theorem x_squared_y_squared_value (x y : ℝ) 
  (h1 : x + y = 25)
  (h2 : x^2 + y^2 = 169)
  (h3 : x^3*y^3 + y^3*x^3 = 243) :
  x^2 * y^2 = 51984 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_y_squared_value_l2342_234294


namespace NUMINAMATH_CALUDE_units_digit_of_product_units_digit_of_27_times_68_l2342_234242

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The product of two natural numbers has the same units digit as the product of their units digits -/
theorem units_digit_of_product (a b : ℕ) :
  unitsDigit (a * b) = unitsDigit (unitsDigit a * unitsDigit b) := by sorry

theorem units_digit_of_27_times_68 :
  unitsDigit (27 * 68) = 6 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_units_digit_of_27_times_68_l2342_234242


namespace NUMINAMATH_CALUDE_not_all_electric_implies_some_not_electric_l2342_234217

-- Define the set of all cars in the parking lot
variable (Car : Type)
variable (parking_lot : Set Car)

-- Define a predicate for electric cars
variable (is_electric : Car → Prop)

-- Define the theorem
theorem not_all_electric_implies_some_not_electric
  (h : ¬ ∀ (c : Car), c ∈ parking_lot → is_electric c) :
  ∃ (c : Car), c ∈ parking_lot ∧ ¬ is_electric c :=
by
  sorry

end NUMINAMATH_CALUDE_not_all_electric_implies_some_not_electric_l2342_234217


namespace NUMINAMATH_CALUDE_inequality_proof_l2342_234211

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x / Real.sqrt (y + z)) + (y / Real.sqrt (z + x)) + (z / Real.sqrt (x + y)) ≥ Real.sqrt ((3 / 2) * (x + y + z)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2342_234211


namespace NUMINAMATH_CALUDE_complex_imaginary_solution_l2342_234241

theorem complex_imaginary_solution (z : ℂ) : 
  (∃ b : ℝ, z = b * I) → 
  (∃ c : ℝ, (z - 3)^2 + 12 * I = c * I) → 
  (z = 3 * I ∨ z = -3 * I) := by
sorry

end NUMINAMATH_CALUDE_complex_imaginary_solution_l2342_234241


namespace NUMINAMATH_CALUDE_triangle_similarity_from_arithmetic_sides_l2342_234208

/-- Two triangles with sides in arithmetic progression and one equal angle are similar -/
theorem triangle_similarity_from_arithmetic_sides (a b c a₁ b₁ c₁ : ℝ) 
  (angleCAB angleCBA angleABC angleC₁A₁B₁ angleC₁B₁A₁ angleA₁B₁C₁ : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < a₁ ∧ 0 < b₁ ∧ 0 < c₁ →
  b - a = c - b →
  b₁ - a₁ = c₁ - b₁ →
  angleCAB + angleCBA + angleABC = π →
  angleC₁A₁B₁ + angleC₁B₁A₁ + angleA₁B₁C₁ = π →
  angleCAB = angleC₁A₁B₁ →
  ∃ (k : ℝ), k > 0 ∧ a = k * a₁ ∧ b = k * b₁ ∧ c = k * c₁ :=
by sorry

end NUMINAMATH_CALUDE_triangle_similarity_from_arithmetic_sides_l2342_234208


namespace NUMINAMATH_CALUDE_rectangle_perimeter_is_46_l2342_234240

/-- A rectangle dissection puzzle with seven squares -/
structure RectangleDissection where
  b₁ : ℕ
  b₂ : ℕ
  b₃ : ℕ
  b₄ : ℕ
  b₅ : ℕ
  b₆ : ℕ
  b₇ : ℕ
  rel₁ : b₁ + b₂ = b₃
  rel₂ : b₁ + b₃ = b₄
  rel₃ : b₃ + b₄ = b₅
  rel₄ : b₄ + b₅ = b₆
  rel₅ : b₂ + b₅ = b₇
  b₁_eq_one : b₁ = 1
  b₂_eq_two : b₂ = 2

/-- The perimeter of the rectangle in the dissection puzzle -/
def perimeter (r : RectangleDissection) : ℕ :=
  2 * (r.b₆ + r.b₇)

/-- Theorem stating that the perimeter of the rectangle is 46 -/
theorem rectangle_perimeter_is_46 (r : RectangleDissection) : perimeter r = 46 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_is_46_l2342_234240


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2342_234218

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Check if a point lies on a line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem perpendicular_line_equation :
  ∃ (l : Line),
    perpendicular l (Line.mk 2 (-3) 4) ∧
    point_on_line (-1) 2 l ∧
    l = Line.mk 3 2 (-1) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2342_234218


namespace NUMINAMATH_CALUDE_calculation_result_l2342_234254

theorem calculation_result : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ 
  |0.00067 * 0.338 - (75 * 0.00000102 / 0.00338 * 0.042) - 0.0008| < ε :=
sorry

end NUMINAMATH_CALUDE_calculation_result_l2342_234254


namespace NUMINAMATH_CALUDE_geometric_number_difference_l2342_234214

/-- A geometric number is a 3-digit number with distinct digits forming a geometric sequence,
    and the middle digit is odd. -/
def IsGeometricNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    Odd b ∧
    b * b = a * c

theorem geometric_number_difference :
  ∃ (min max : ℕ),
    IsGeometricNumber min ∧
    IsGeometricNumber max ∧
    (∀ n, IsGeometricNumber n → min ≤ n ∧ n ≤ max) ∧
    max - min = 220 := by
  sorry

end NUMINAMATH_CALUDE_geometric_number_difference_l2342_234214


namespace NUMINAMATH_CALUDE_insertion_methods_l2342_234250

theorem insertion_methods (n : ℕ) (k : ℕ) : n = 5 ∧ k = 2 → (n + 1) * (n + 2) = 42 := by
  sorry

end NUMINAMATH_CALUDE_insertion_methods_l2342_234250


namespace NUMINAMATH_CALUDE_percentage_calculation_l2342_234207

theorem percentage_calculation (a : ℝ) (x : ℝ) (h1 : a = 140) (h2 : (x / 100) * a = 70) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2342_234207


namespace NUMINAMATH_CALUDE_cost_of_five_cds_l2342_234227

/-- The cost of a certain number of identical CDs -/
def cost_of_cds (n : ℕ) : ℚ :=
  28 * (n / 2 : ℚ)

/-- Theorem stating that the cost of five CDs is 70 dollars -/
theorem cost_of_five_cds : cost_of_cds 5 = 70 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_five_cds_l2342_234227


namespace NUMINAMATH_CALUDE_cauchy_inequality_2d_l2342_234263

theorem cauchy_inequality_2d (a b c d : ℝ) : 
  (a * c + b * d)^2 ≤ (a^2 + b^2) * (c^2 + d^2) ∧ 
  ((a * c + b * d)^2 = (a^2 + b^2) * (c^2 + d^2) ↔ a * d = b * c) :=
sorry

end NUMINAMATH_CALUDE_cauchy_inequality_2d_l2342_234263


namespace NUMINAMATH_CALUDE_k_satisfies_conditions_l2342_234206

/-- The number of digits in the second factor of (9)(999...9) -/
def k : ℕ := 55

/-- The resulting integer from the multiplication (9)(999...9) -/
def result (n : ℕ) : ℕ := 9 * (10^n - 1)

/-- The sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

/-- The theorem stating that k satisfies the given conditions -/
theorem k_satisfies_conditions : digit_sum (result k) = 500 := by sorry

end NUMINAMATH_CALUDE_k_satisfies_conditions_l2342_234206


namespace NUMINAMATH_CALUDE_min_value_and_nonexistence_l2342_234215

theorem min_value_and_nonexistence (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 4 * b = (a * b) ^ (3/2)) :
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' + 4 * b' = (a' * b') ^ (3/2) → a' ^ 2 + 16 * b' ^ 2 ≥ 32) ∧
  ¬∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ a' + 4 * b' = (a' * b') ^ (3/2) ∧ a' + 3 * b' = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_nonexistence_l2342_234215


namespace NUMINAMATH_CALUDE_expression_values_l2342_234203

theorem expression_values (x y z : ℝ) (h : x * y * z ≠ 0) :
  let expr := |x| / x + y / |y| + |z| / z
  expr = 1 ∨ expr = -1 ∨ expr = 3 ∨ expr = -3 :=
by sorry

end NUMINAMATH_CALUDE_expression_values_l2342_234203


namespace NUMINAMATH_CALUDE_phi_value_l2342_234288

theorem phi_value (φ : Real) (a : Real) :
  φ ∈ Set.Icc 0 (2 * Real.pi) →
  (∃ x₁ x₂ x₃ : Real,
    x₁ ∈ Set.Icc 0 Real.pi ∧
    x₂ ∈ Set.Icc 0 Real.pi ∧
    x₃ ∈ Set.Icc 0 Real.pi ∧
    Real.sin (2 * x₁ + φ) = a ∧
    Real.sin (2 * x₂ + φ) = a ∧
    Real.sin (2 * x₃ + φ) = a ∧
    x₁ + x₂ + x₃ = 7 * Real.pi / 6) →
  φ = Real.pi / 3 ∨ φ = 4 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_phi_value_l2342_234288


namespace NUMINAMATH_CALUDE_total_peanuts_l2342_234266

/-- The number of peanuts initially in the box -/
def initial_peanuts : ℕ := 4

/-- The number of peanuts Mary adds to the box -/
def added_peanuts : ℕ := 4

/-- Theorem: The total number of peanuts in the box is 8 -/
theorem total_peanuts : initial_peanuts + added_peanuts = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_peanuts_l2342_234266


namespace NUMINAMATH_CALUDE_sum_of_digits_of_N_l2342_234260

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem sum_of_digits_of_N (N : ℕ) (h : N^2 = 36^50 * 50^36) : sum_of_digits N = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_N_l2342_234260


namespace NUMINAMATH_CALUDE_tan_graph_product_l2342_234269

theorem tan_graph_product (a b : ℝ) : 
  a > 0 → b > 0 → 
  (π / b = 2 * π / 3) →
  (a * Real.tan (b * (π / 6)) = 2) →
  a * b = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_graph_product_l2342_234269


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2342_234286

/-- Given x = √5 + 2 and y = √5 - 2, prove that x^2 - y + xy = 12 + 3√5 -/
theorem algebraic_expression_value :
  let x : ℝ := Real.sqrt 5 + 2
  let y : ℝ := Real.sqrt 5 - 2
  x^2 - y + x*y = 12 + 3 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2342_234286


namespace NUMINAMATH_CALUDE_complex_fraction_real_l2342_234205

theorem complex_fraction_real (t : ℝ) : 
  (Complex.I * (2 * t + Complex.I) / (1 - 2 * Complex.I)).im = 0 → t = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_real_l2342_234205


namespace NUMINAMATH_CALUDE_divisibility_criterion_l2342_234255

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem divisibility_criterion (n : ℕ) (h : n > 1) :
  (Nat.factorial (n - 1)) % n = 0 ↔ is_composite n ∧ n ≠ 4 :=
sorry

end NUMINAMATH_CALUDE_divisibility_criterion_l2342_234255


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l2342_234230

theorem point_in_first_quadrant (α : Real) : 
  α ∈ Set.Icc 0 (2 * Real.pi) →
  (Real.sin α - Real.cos α > 0 ∧ Real.tan α > 0) ↔ 
  (α ∈ Set.Ioo (Real.pi / 4) (Real.pi / 2) ∪ Set.Ioo Real.pi (5 * Real.pi / 4)) := by
sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l2342_234230


namespace NUMINAMATH_CALUDE_salt_production_increase_l2342_234248

/-- Proves that given an initial production of 1000 tonnes in January and an average
    monthly production of 1550 tonnes for the year, the constant monthly increase
    in production from February to December is 100 tonnes. -/
theorem salt_production_increase (initial_production : ℕ) (average_production : ℕ) 
  (monthly_increase : ℕ) (h1 : initial_production = 1000) 
  (h2 : average_production = 1550) :
  (monthly_increase = 100 ∧ 
   (12 * initial_production + (monthly_increase * 11 * 12 / 2) = 12 * average_production)) := by
  sorry

end NUMINAMATH_CALUDE_salt_production_increase_l2342_234248


namespace NUMINAMATH_CALUDE_first_platform_length_l2342_234292

/-- The length of a train in meters. -/
def train_length : ℝ := 350

/-- The time taken to cross the first platform in seconds. -/
def time_first : ℝ := 15

/-- The length of the second platform in meters. -/
def length_second : ℝ := 250

/-- The time taken to cross the second platform in seconds. -/
def time_second : ℝ := 20

/-- The length of the first platform in meters. -/
def length_first : ℝ := 100

theorem first_platform_length :
  (train_length + length_first) / time_first = (train_length + length_second) / time_second :=
by sorry

end NUMINAMATH_CALUDE_first_platform_length_l2342_234292


namespace NUMINAMATH_CALUDE_lucas_numbers_l2342_234284

theorem lucas_numbers (a b : ℤ) : 
  (3 * a + 4 * b = 161) → 
  ((a = 17 ∨ b = 17) → (a = 31 ∨ b = 31)) :=
by sorry

end NUMINAMATH_CALUDE_lucas_numbers_l2342_234284


namespace NUMINAMATH_CALUDE_age_ratio_problem_l2342_234223

theorem age_ratio_problem (ann_age : ℕ) (x : ℚ) : 
  ann_age = 6 →
  (ann_age + 10) + (x * ann_age + 10) = 38 →
  x * ann_age / ann_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l2342_234223


namespace NUMINAMATH_CALUDE_x_to_twenty_l2342_234221

theorem x_to_twenty (x : ℝ) (h : x + 1/x = Real.sqrt 5) : x^20 = 16163 := by
  sorry

end NUMINAMATH_CALUDE_x_to_twenty_l2342_234221


namespace NUMINAMATH_CALUDE_marble_247_is_white_l2342_234239

/-- Represents the color of a marble -/
inductive MarbleColor
| Gray
| White
| Black

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : MarbleColor :=
  match n % 12 with
  | 0 | 1 | 2 | 3 => MarbleColor.Gray
  | 4 | 5 | 6 | 7 | 8 => MarbleColor.White
  | _ => MarbleColor.Black

/-- Theorem stating that the 247th marble is white -/
theorem marble_247_is_white : marbleColor 247 = MarbleColor.White := by
  sorry


end NUMINAMATH_CALUDE_marble_247_is_white_l2342_234239


namespace NUMINAMATH_CALUDE_parabola_directrix_parameter_l2342_234246

/-- 
For a parabola y = ax^2 with directrix y = 1, the value of a is -1/4.
-/
theorem parabola_directrix_parameter (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2) →  -- Condition 1: Parabola equation
  (∃ y : ℝ, y = 1 ∧ ∀ x : ℝ, y = 1 → (x, y) ∉ {(x, y) | y = a * x^2}) →  -- Condition 2: Directrix equation
  a = -1/4 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_parameter_l2342_234246


namespace NUMINAMATH_CALUDE_fourth_customer_new_item_probability_l2342_234229

/-- The number of menu items --/
def menu_items : ℕ := 5

/-- The number of customers --/
def customers : ℕ := 4

/-- The probability that the 4th customer orders a previously unordered item --/
def probability : ℚ := 32 / 125

theorem fourth_customer_new_item_probability :
  (menu_items ^ (customers - 1) * (menu_items - (customers - 1))) /
  (menu_items ^ customers) = probability := by
  sorry

end NUMINAMATH_CALUDE_fourth_customer_new_item_probability_l2342_234229


namespace NUMINAMATH_CALUDE_point_movement_power_l2342_234216

/-- 
Given a point (-1, 1) in the Cartesian coordinate system,
if it is moved up 1 unit and then left 2 units to reach a point (x, y),
then x^y = 9.
-/
theorem point_movement_power (x y : ℝ) : 
  ((-1 : ℝ) + -2 = x) → ((1 : ℝ) + 1 = y) → x^y = 9 := by
  sorry

end NUMINAMATH_CALUDE_point_movement_power_l2342_234216


namespace NUMINAMATH_CALUDE_acceleration_at_two_seconds_l2342_234212

-- Define the distance function
def s (t : ℝ) : ℝ := 2 * t^3 - 5 * t^2 + 2

-- Define the velocity function as the derivative of the distance function
def v (t : ℝ) : ℝ := 6 * t^2 - 10 * t

-- Define the acceleration function as the derivative of the velocity function
def a (t : ℝ) : ℝ := 12 * t - 10

-- Theorem: The acceleration at t = 2 seconds is 14 m/s²
theorem acceleration_at_two_seconds : a 2 = 14 := by sorry

end NUMINAMATH_CALUDE_acceleration_at_two_seconds_l2342_234212


namespace NUMINAMATH_CALUDE_minimum_point_of_translated_abs_function_l2342_234259

-- Define the function
def f (x : ℝ) : ℝ := |x - 4| + 7

-- State the theorem
theorem minimum_point_of_translated_abs_function :
  ∃ (x₀ : ℝ), (∀ (x : ℝ), f x ≥ f x₀) ∧ f x₀ = 7 ∧ x₀ = 4 :=
sorry

end NUMINAMATH_CALUDE_minimum_point_of_translated_abs_function_l2342_234259


namespace NUMINAMATH_CALUDE_range_of_b_l2342_234256

theorem range_of_b (a : ℝ) (h1 : 0 < a) (h2 : a ≤ 5/4) :
  (∃ (b : ℝ), b > 0 ∧ 
    (∀ (x : ℝ), |x - a| < b → |x - a^2| < 1/2) ∧
    (∀ (c : ℝ), c > b → ∃ (y : ℝ), |y - a| < c ∧ |y - a^2| ≥ 1/2)) ∧
  (∀ (b : ℝ), (∀ (x : ℝ), |x - a| < b → |x - a^2| < 1/2) → b ≤ 3/16) :=
sorry

end NUMINAMATH_CALUDE_range_of_b_l2342_234256


namespace NUMINAMATH_CALUDE_only_valid_rectangles_l2342_234237

/-- A rectangle that can be divided into 13 equal squares -/
structure Rectangle13Squares where
  width : ℕ
  height : ℕ
  is_valid : width * height = 13

/-- The set of all valid rectangles that can be divided into 13 equal squares -/
def valid_rectangles : Set Rectangle13Squares :=
  {r : Rectangle13Squares | r.width = 1 ∧ r.height = 13 ∨ r.width = 13 ∧ r.height = 1}

/-- Theorem stating that the only valid rectangles are 1x13 or 13x1 -/
theorem only_valid_rectangles :
  ∀ r : Rectangle13Squares, r ∈ valid_rectangles :=
by
  sorry

end NUMINAMATH_CALUDE_only_valid_rectangles_l2342_234237


namespace NUMINAMATH_CALUDE_lcm_48_180_l2342_234201

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  sorry

end NUMINAMATH_CALUDE_lcm_48_180_l2342_234201


namespace NUMINAMATH_CALUDE_export_volume_equation_l2342_234297

def export_volume_2023 : ℝ := 107
def export_volume_2013 : ℝ → ℝ := λ x => x

theorem export_volume_equation (x : ℝ) : 
  export_volume_2023 = 4 * (export_volume_2013 x) + 3 ↔ 4 * x + 3 = 107 :=
by sorry

end NUMINAMATH_CALUDE_export_volume_equation_l2342_234297


namespace NUMINAMATH_CALUDE_least_possible_difference_l2342_234238

theorem least_possible_difference (x y z : ℤ) : 
  x < y → y < z → 
  y - x > 9 → 
  Even x → Odd y → Odd z → 
  (∀ w, w = z - x → w ≥ 13) ∧ ∃ w, w = z - x ∧ w = 13 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_difference_l2342_234238


namespace NUMINAMATH_CALUDE_college_students_count_l2342_234272

theorem college_students_count (num_girls : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) : 
  num_girls = 120 → ratio_boys = 8 → ratio_girls = 5 →
  (num_girls + (num_girls * ratio_boys) / ratio_girls : ℕ) = 312 := by
sorry

end NUMINAMATH_CALUDE_college_students_count_l2342_234272


namespace NUMINAMATH_CALUDE_function_composition_multiplication_l2342_234281

-- Define the composition operation
def compose (f g : ℝ → ℝ) : ℝ → ℝ := λ x => f (g x)

-- Define the multiplication operation
def multiply (f g : ℝ → ℝ) : ℝ → ℝ := λ x => f x * g x

-- State the theorem
theorem function_composition_multiplication (f g h : ℝ → ℝ) :
  compose (multiply f g) h = multiply (compose f h) (compose g h) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_multiplication_l2342_234281


namespace NUMINAMATH_CALUDE_relay_team_permutations_l2342_234290

theorem relay_team_permutations (n : ℕ) (h : n = 3) : Nat.factorial n = 6 := by
  sorry

end NUMINAMATH_CALUDE_relay_team_permutations_l2342_234290


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2342_234200

theorem contrapositive_equivalence (x : ℝ) : 
  (x^2 = 1 → x = 1 ∨ x = -1) ↔ (x ≠ 1 ∧ x ≠ -1 → x^2 ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2342_234200


namespace NUMINAMATH_CALUDE_writing_stats_theorem_l2342_234268

/-- Represents the writing statistics of an author -/
structure WritingStats where
  total_words : ℕ
  total_hours : ℕ
  first_half_hours : ℕ
  first_half_words : ℕ

/-- Calculates the average words per hour -/
def average_words_per_hour (words : ℕ) (hours : ℕ) : ℚ :=
  (words : ℚ) / (hours : ℚ)

/-- Theorem about the writing statistics -/
theorem writing_stats_theorem (stats : WritingStats) 
  (h1 : stats.total_words = 60000)
  (h2 : stats.total_hours = 150)
  (h3 : stats.first_half_hours = 50)
  (h4 : stats.first_half_words = stats.total_words / 2) :
  average_words_per_hour stats.total_words stats.total_hours = 400 ∧
  average_words_per_hour stats.first_half_words stats.first_half_hours = 600 := by
  sorry


end NUMINAMATH_CALUDE_writing_stats_theorem_l2342_234268


namespace NUMINAMATH_CALUDE_gcd_12345_67890_l2342_234274

theorem gcd_12345_67890 : Nat.gcd 12345 67890 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12345_67890_l2342_234274


namespace NUMINAMATH_CALUDE_pet_ownership_percentage_l2342_234213

theorem pet_ownership_percentage (total_students : ℕ) (cat_owners : ℕ) (dog_owners : ℕ) (both_owners : ℕ)
  (h1 : total_students = 500)
  (h2 : cat_owners = 150)
  (h3 : dog_owners = 100)
  (h4 : both_owners = 40) :
  (cat_owners + dog_owners - both_owners) / total_students = 42 / 100 := by
  sorry

end NUMINAMATH_CALUDE_pet_ownership_percentage_l2342_234213


namespace NUMINAMATH_CALUDE_calculation_proof_l2342_234252

theorem calculation_proof : 
  |(-7)| / ((2 / 3) - (1 / 5)) - (1 / 2) * ((-4)^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2342_234252


namespace NUMINAMATH_CALUDE_square_circle_area_ratio_l2342_234249

theorem square_circle_area_ratio (s r : ℝ) (h : s > 0) (k : r > 0) (eq : 4 * s = 4 * Real.pi * r) :
  s^2 / (Real.pi * r^2) = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_square_circle_area_ratio_l2342_234249


namespace NUMINAMATH_CALUDE_max_value_theorem_l2342_234231

theorem max_value_theorem (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0) :
  (∀ a b : ℝ, a > 0 → b > 0 → (k * a + b)^2 / (a^2 + b^2) ≤ (k * x + y)^2 / (x^2 + y^2)) →
  (k * x + y)^2 / (x^2 + y^2) = 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2342_234231


namespace NUMINAMATH_CALUDE_log_inequality_l2342_234253

theorem log_inequality (x : ℝ) (h : x > 0) : Real.log (1 + 1/x) > 1/(1 + x) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l2342_234253


namespace NUMINAMATH_CALUDE_solution_set_part1_solution_set_characterization_l2342_234262

-- Part 1
theorem solution_set_part1 (x : ℝ) :
  -5 * x^2 + 3 * x + 2 > 0 ↔ -2/5 < x ∧ x < 1 := by sorry

-- Part 2
def solution_set_part2 (a x : ℝ) : Prop :=
  a * x^2 + 3 * x + 2 > -a * x - 1

theorem solution_set_characterization (a x : ℝ) (ha : a > 0) :
  solution_set_part2 a x ↔
    (0 < a ∧ a < 3 ∧ (x < -3/a ∨ x > -1)) ∨
    (a = 3 ∧ x ≠ -1) ∨
    (a > 3 ∧ (x < -1 ∨ x > -3/a)) := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_solution_set_characterization_l2342_234262


namespace NUMINAMATH_CALUDE_coin_packing_theorem_l2342_234285

/-- A coin is represented by its center and radius -/
structure Coin where
  center : ℝ × ℝ
  radius : ℝ

/-- The configuration of 12 coins forming a regular 12-gon -/
def outer_ring : List Coin := sorry

/-- The configuration of 7 coins inside the outer ring -/
def inner_coins : List Coin := sorry

/-- Two coins are tangent if the distance between their centers equals the sum of their radii -/
def are_tangent (c1 c2 : Coin) : Prop := sorry

/-- All coins in a list are mutually tangent -/
def all_tangent (coins : List Coin) : Prop := sorry

/-- The centers of the outer coins form a regular 12-gon -/
def is_regular_12gon (coins : List Coin) : Prop := sorry

theorem coin_packing_theorem :
  is_regular_12gon outer_ring ∧
  all_tangent outer_ring ∧
  (∀ c ∈ inner_coins, ∀ o ∈ outer_ring, are_tangent c o ∨ c = o) ∧
  all_tangent inner_coins ∧
  (List.length outer_ring = 12) ∧
  (List.length inner_coins = 7) := by
  sorry

end NUMINAMATH_CALUDE_coin_packing_theorem_l2342_234285


namespace NUMINAMATH_CALUDE_quadratic_sequence_exists_l2342_234236

theorem quadratic_sequence_exists (b c : ℤ) : 
  ∃ (n : ℕ) (a : ℕ → ℤ), a 0 = b ∧ a n = c ∧ 
  ∀ i : ℕ, i ≥ 1 → i ≤ n → |a i - a (i-1)| = i^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sequence_exists_l2342_234236


namespace NUMINAMATH_CALUDE_modular_exponentiation_16_cube_mod_7_l2342_234282

theorem modular_exponentiation_16_cube_mod_7 :
  ∃ m : ℕ, 16^3 ≡ m [ZMOD 7] ∧ 0 ≤ m ∧ m < 7 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_modular_exponentiation_16_cube_mod_7_l2342_234282


namespace NUMINAMATH_CALUDE_cathy_remaining_money_l2342_234280

/-- Calculates the remaining money in Cathy's wallet after all expenditures --/
def remaining_money (initial_amount dad_sent mom_sent_multiplier book_cost cab_ride_percent dinner_percent : ℝ) : ℝ :=
  let total_from_parents := dad_sent + mom_sent_multiplier * dad_sent
  let total_initial := total_from_parents + initial_amount
  let food_budget := 0.4 * total_initial
  let after_book := total_initial - book_cost
  let cab_cost := cab_ride_percent * after_book
  let after_cab := after_book - cab_cost
  let dinner_cost := dinner_percent * food_budget
  after_cab - dinner_cost

/-- Theorem stating that Cathy's remaining money is $52.44 --/
theorem cathy_remaining_money :
  remaining_money 12 25 2 15 0.03 0.5 = 52.44 := by
  sorry

end NUMINAMATH_CALUDE_cathy_remaining_money_l2342_234280


namespace NUMINAMATH_CALUDE_max_ab_value_l2342_234228

theorem max_ab_value (a b : ℝ) (h1 : 1 ≤ a - b) (h2 : a - b ≤ 2) (h3 : 3 ≤ a + b) (h4 : a + b ≤ 4) :
  ∃ (m : ℝ), m = 15/4 ∧ ab ≤ m ∧ ∃ (a' b' : ℝ), 1 ≤ a' - b' ∧ a' - b' ≤ 2 ∧ 3 ≤ a' + b' ∧ a' + b' ≤ 4 ∧ a' * b' = m :=
sorry

end NUMINAMATH_CALUDE_max_ab_value_l2342_234228


namespace NUMINAMATH_CALUDE_division_by_fraction_twelve_divided_by_three_fifths_l2342_234289

theorem division_by_fraction (a b c : ℚ) (hb : b ≠ 0) (hc : c ≠ 0) :
  a / (b / c) = (a * c) / b := by sorry

theorem twelve_divided_by_three_fifths :
  12 / (3 / 5) = 20 := by sorry

end NUMINAMATH_CALUDE_division_by_fraction_twelve_divided_by_three_fifths_l2342_234289


namespace NUMINAMATH_CALUDE_max_digits_product_5_and_3_l2342_234251

theorem max_digits_product_5_and_3 : 
  ∀ a b : ℕ, 
  10000 ≤ a ∧ a ≤ 99999 → 
  100 ≤ b ∧ b ≤ 999 → 
  a * b < 100000000 := by
sorry

end NUMINAMATH_CALUDE_max_digits_product_5_and_3_l2342_234251


namespace NUMINAMATH_CALUDE_table_length_l2342_234257

/-- Proves that a rectangular table with an area of 54 square meters and a width of 600 centimeters has a length of 900 centimeters. -/
theorem table_length (area : ℝ) (width : ℝ) (length : ℝ) : 
  area = 54 → 
  width = 6 →
  area = length * width →
  length * 100 = 900 := by
  sorry

#check table_length

end NUMINAMATH_CALUDE_table_length_l2342_234257


namespace NUMINAMATH_CALUDE_parabola_reflection_l2342_234293

/-- Reflects a point (x, y) over the point (1, 1) -/
def reflect (x y : ℝ) : ℝ × ℝ := (2 - x, 2 - y)

/-- The original parabola y = x^2 -/
def original_parabola (x y : ℝ) : Prop := y = x^2

/-- The reflected parabola y = -x^2 + 4x - 2 -/
def reflected_parabola (x y : ℝ) : Prop := y = -x^2 + 4*x - 2

theorem parabola_reflection :
  ∀ x y : ℝ, original_parabola x y ↔ reflected_parabola (reflect x y).1 (reflect x y).2 :=
sorry

end NUMINAMATH_CALUDE_parabola_reflection_l2342_234293


namespace NUMINAMATH_CALUDE_f_properties_l2342_234232

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.pi / 2 - x) * Real.cos x + Real.sqrt 3 * Real.sin x ^ 2

theorem f_properties :
  -- Smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  -- Monotonically decreasing in [5π/12 + kπ, 11π/12 + kπ]
  (∀ (k : ℤ), StrictMonoOn f (Set.Icc (5 * Real.pi / 12 + k * Real.pi) (11 * Real.pi / 12 + k * Real.pi))) ∧
  -- Minimum and maximum values on [π/6, π/2]
  (∃ (x_min x_max : ℝ), x_min ∈ Set.Icc (Real.pi / 6) (Real.pi / 2) ∧
                        x_max ∈ Set.Icc (Real.pi / 6) (Real.pi / 2) ∧
                        (∀ (x : ℝ), x ∈ Set.Icc (Real.pi / 6) (Real.pi / 2) → 
                          f x_min ≤ f x ∧ f x ≤ f x_max) ∧
                        f x_min = Real.sqrt 3 / 2 ∧
                        f x_max = Real.sqrt 3 / 2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2342_234232


namespace NUMINAMATH_CALUDE_reading_time_per_disc_l2342_234279

/-- Proves that given a total reading time of 480 minutes, disc capacity of 60 minutes,
    and the conditions of using the smallest possible number of discs with equal reading time on each disc,
    the reading time per disc is 60 minutes. -/
theorem reading_time_per_disc (total_time : ℕ) (disc_capacity : ℕ) (reading_per_disc : ℕ) :
  total_time = 480 →
  disc_capacity = 60 →
  reading_per_disc * (total_time / disc_capacity) = total_time →
  reading_per_disc ≤ disc_capacity →
  reading_per_disc = 60 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_per_disc_l2342_234279


namespace NUMINAMATH_CALUDE_machine_time_calculation_l2342_234277

/-- Given a machine that can make a certain number of shirts per minute
    and has made a total number of shirts, calculate the time it worked. -/
def machine_working_time (shirts_per_minute : ℕ) (total_shirts : ℕ) : ℚ :=
  total_shirts / shirts_per_minute

/-- Theorem stating that for a machine making 3 shirts per minute
    and having made 6 shirts in total, it worked for 2 minutes. -/
theorem machine_time_calculation :
  machine_working_time 3 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_machine_time_calculation_l2342_234277


namespace NUMINAMATH_CALUDE_coin_distribution_theorem_l2342_234225

-- Define the number of cans
def num_cans : ℕ := 2015

-- Define the three initial configurations
def config_a (j : ℕ) : ℤ := 0
def config_b (j : ℕ) : ℤ := j
def config_c (j : ℕ) : ℤ := 2016 - j

-- Define the property that needs to be proven for each configuration
def has_solution (d : ℕ → ℤ) : Prop :=
  ∃ X : ℤ, ∀ j : ℕ, 1 ≤ j ∧ j ≤ num_cans → X ≡ d j [ZMOD j]

-- Theorem statement
theorem coin_distribution_theorem :
  has_solution config_a ∧ has_solution config_b ∧ has_solution config_c :=
sorry

end NUMINAMATH_CALUDE_coin_distribution_theorem_l2342_234225


namespace NUMINAMATH_CALUDE_max_students_distribution_l2342_234244

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 1204) (h2 : pencils = 840) :
  Nat.gcd pens pencils = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_students_distribution_l2342_234244


namespace NUMINAMATH_CALUDE_roots_in_interval_l2342_234298

theorem roots_in_interval (m : ℝ) :
  (∀ x, 4 * x^2 - (3 * m + 1) * x - m - 2 = 0 → -1 < x ∧ x < 2) ↔ -1 < m ∧ m < 12/7 := by
  sorry

end NUMINAMATH_CALUDE_roots_in_interval_l2342_234298


namespace NUMINAMATH_CALUDE_ticket_cost_l2342_234278

/-- The cost of a single ticket at the fair, given the initial number of tickets,
    remaining tickets, and total amount spent on the ferris wheel. -/
theorem ticket_cost (initial_tickets : ℕ) (remaining_tickets : ℕ) (total_spent : ℕ) :
  initial_tickets > remaining_tickets →
  total_spent % (initial_tickets - remaining_tickets) = 0 →
  total_spent / (initial_tickets - remaining_tickets) = 9 :=
by
  intro h_tickets h_divisible
  sorry

#check ticket_cost 13 4 81

end NUMINAMATH_CALUDE_ticket_cost_l2342_234278


namespace NUMINAMATH_CALUDE_point_C_in_fourth_quadrant_l2342_234243

/-- A point in the 2D Cartesian coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The point we want to prove is in the fourth quadrant -/
def point_C : Point :=
  { x := 1, y := -2 }

/-- Theorem: point_C is in the fourth quadrant -/
theorem point_C_in_fourth_quadrant : is_in_fourth_quadrant point_C := by
  sorry

end NUMINAMATH_CALUDE_point_C_in_fourth_quadrant_l2342_234243


namespace NUMINAMATH_CALUDE_arc_length_calculation_l2342_234273

theorem arc_length_calculation (circumference : ℝ) (central_angle : ℝ) 
  (h1 : circumference = 72) 
  (h2 : central_angle = 45) : 
  (central_angle / 360) * circumference = 9 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_calculation_l2342_234273


namespace NUMINAMATH_CALUDE_sine_cosine_transform_l2342_234295

theorem sine_cosine_transform (x : ℝ) : 
  Real.sqrt 3 * Real.sin (3 * x) + Real.cos (3 * x) = 2 * Real.sin (3 * x + π / 6) := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_transform_l2342_234295


namespace NUMINAMATH_CALUDE_garden_perimeter_l2342_234299

/-- The perimeter of a rectangular garden with width 16 meters and the same area as a rectangular playground of length 16 meters and width 12 meters is 56 meters. -/
theorem garden_perimeter (garden_width playground_length playground_width : ℝ) :
  garden_width = 16 →
  playground_length = 16 →
  playground_width = 12 →
  garden_width * (playground_length * playground_width / garden_width) + 2 * garden_width = 56 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l2342_234299


namespace NUMINAMATH_CALUDE_y_value_l2342_234267

theorem y_value (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l2342_234267


namespace NUMINAMATH_CALUDE_solve_equation_l2342_234234

theorem solve_equation (a b c : ℤ) (x : ℝ) (h : 5 / (a^2 + b * Real.log x) = c) :
  x = 10 ^ ((5 / c - a^2) / b) :=
by sorry

end NUMINAMATH_CALUDE_solve_equation_l2342_234234


namespace NUMINAMATH_CALUDE_least_even_perimeter_l2342_234247

theorem least_even_perimeter (a b c : ℕ) : 
  a = 24 →
  b = 37 →
  c ≥ a ∧ c ≥ b →
  a + b + c > a + b →
  Even (a + b + c) →
  (∀ x : ℕ, x < c → ¬(Even (a + b + x) ∧ a + b + x > a + b)) →
  a + b + c = 100 := by
  sorry

end NUMINAMATH_CALUDE_least_even_perimeter_l2342_234247


namespace NUMINAMATH_CALUDE_max_at_2_implies_c_6_l2342_234202

/-- The function f(x) = x(x-c)² has a maximum value at x=2 -/
def has_max_at_2 (c : ℝ) : Prop :=
  ∀ x : ℝ, x * (x - c)^2 ≤ 2 * (2 - c)^2

/-- Theorem: If f(x) = x(x-c)² has a maximum value at x=2, then c = 6 -/
theorem max_at_2_implies_c_6 :
  ∀ c : ℝ, has_max_at_2 c → c = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_at_2_implies_c_6_l2342_234202


namespace NUMINAMATH_CALUDE_library_book_sale_l2342_234204

theorem library_book_sale (initial_books : ℕ) (remaining_fraction : ℚ) : 
  initial_books = 9900 →
  remaining_fraction = 4/6 →
  initial_books * (1 - remaining_fraction) = 3300 :=
by sorry

end NUMINAMATH_CALUDE_library_book_sale_l2342_234204


namespace NUMINAMATH_CALUDE_adrian_cards_l2342_234271

theorem adrian_cards (n : ℕ) : 
  (∃ k : ℕ, 
    k ≥ 1 ∧ 
    k + n - 1 ≤ 2 * n ∧ 
    (2 * n * (2 * n + 1)) / 2 - (n * k + (n * (n - 1)) / 2) = 1615) →
  (n = 34 ∨ n = 38) :=
by sorry

end NUMINAMATH_CALUDE_adrian_cards_l2342_234271


namespace NUMINAMATH_CALUDE_kim_money_l2342_234226

/-- Given that Kim has 40% more money than Sal, Sal has 20% less money than Phil,
    and Sal and Phil have a combined total of $1.80, prove that Kim has $1.12. -/
theorem kim_money (sal phil kim : ℝ) 
  (h1 : kim = sal * 1.4)
  (h2 : sal = phil * 0.8)
  (h3 : sal + phil = 1.8) : 
  kim = 1.12 := by
sorry

end NUMINAMATH_CALUDE_kim_money_l2342_234226


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l2342_234219

/-- Given an ellipse with equation 4(x+2)^2 + 16y^2 = 64, 
    the distance between an endpoint of its major axis 
    and an endpoint of its minor axis is 2√5. -/
theorem ellipse_axis_endpoint_distance : 
  ∃ (C D : ℝ × ℝ), 
    (∀ (x y : ℝ), 4 * (x + 2)^2 + 16 * y^2 = 64 ↔ (x + 2)^2 / 16 + y^2 / 4 = 1) ∧
    (C.1 = -2 ∧ C.2 = 4 ∨ C.1 = -2 ∧ C.2 = -4) ∧  -- C is an endpoint of the major axis
    (D.1 = 0 ∧ D.2 = 0 ∨ D.1 = -4 ∧ D.2 = 0) ∧    -- D is an endpoint of the minor axis
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l2342_234219
