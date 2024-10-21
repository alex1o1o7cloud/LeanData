import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l96_9638

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.sin x)^2 + a

theorem function_properties :
  ∃ (a : ℝ),
    f (π/3) a = 3 ∧
    (∀ (x : ℝ), f (x + π) a = f x a) ∧
    (∀ (m : ℝ), (∀ (x : ℝ), x ∈ Set.Icc 0 m → f x a ≥ 0) → m ≤ 2*π/3) ∧
    (∃ (m : ℝ), m = 2*π/3 ∧ ∀ (x : ℝ), x ∈ Set.Icc 0 m → f x a ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l96_9638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l96_9621

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

theorem min_translation_for_symmetry :
  ∃ (φ : ℝ), φ > 0 ∧
  (∀ (x : ℝ), f (x - φ) = f (-x - φ)) ∧
  (∀ (ψ : ℝ), ψ > 0 ∧ (∀ (x : ℝ), f (x - ψ) = f (-x - ψ)) → ψ ≥ φ) ∧
  φ = 3 * Real.pi / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l96_9621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_area_l96_9605

/-- The area of a triangle given its vertex coordinates -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem triangle_ABC_area :
  triangleArea (-3) 4 1 7 4 (-1) = 32.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_area_l96_9605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relations_l96_9673

/-- A plane vector -/
structure PlaneVector where
  x : ℝ
  y : ℝ

/-- Dot product of two plane vectors -/
def dot (a b : PlaneVector) : ℝ := a.x * b.x + a.y * b.y

/-- Magnitude (norm) of a plane vector -/
noncomputable def norm (a : PlaneVector) : ℝ := Real.sqrt (a.x^2 + a.y^2)

/-- Scalar multiplication of a plane vector -/
def scalarMult (r : ℝ) (a : PlaneVector) : PlaneVector :=
  { x := r * a.x, y := r * a.y }

/-- Zero vector -/
def zeroVector : PlaneVector := { x := 0, y := 0 }

theorem vector_relations (a b c : PlaneVector) : 
  (∃! n : ℕ, n = 2 ∧ 
    (scalarMult 0 a = zeroVector) = false ∧
    (dot a b = dot b a) = true ∧
    (dot a a = norm a ^ 2) = true ∧
    (∀ (x y z : ℝ), x * (y * z) = (x * y) * z) = false ∧
    (|dot a b| ≤ dot a b) = false) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relations_l96_9673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l96_9611

-- Define the hyperbola C
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the focus F, vertex A, and point B
noncomputable def F (a b : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 + b^2), 0)
def A (a : ℝ) : ℝ × ℝ := (a, 0)
noncomputable def B (a b : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 + b^2), b^2 / a)

-- Define the slope of AB
def slope_AB : ℝ := 3

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2) / a

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  hyperbola a b (B a b).1 (B a b).2 ∧
  slope_AB = ((B a b).2 - (A a).2) / ((B a b).1 - (A a).1) →
  eccentricity a b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l96_9611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l96_9679

noncomputable section

/-- The original cosine function -/
def f (x : ℝ) : ℝ := 3 * Real.cos (2 * x + Real.pi / 3)

/-- The shifted cosine function -/
def g (m : ℝ) (x : ℝ) : ℝ := 3 * Real.cos (2 * (x - m) + Real.pi / 3)

/-- A function is symmetric about the origin if f(x) = -f(-x) for all x -/
def symmetric_about_origin (h : ℝ → ℝ) : Prop :=
  ∀ x, h x = -h (-x)

theorem min_shift_for_symmetry :
  ∃ m : ℝ, m > 0 ∧ 
    symmetric_about_origin (g m) ∧
    (∀ m' : ℝ, m' > 0 → symmetric_about_origin (g m') → m ≤ m') ∧
    m = 5 * Real.pi / 12 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l96_9679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_remainder_count_l96_9622

theorem three_digit_remainder_count : 
  (Finset.filter (fun n : ℕ => 
    100 ≤ n ∧ n < 1000 ∧ 
    n % 7 = 4 ∧ 
    n % 8 = 3 ∧ 
    n % 10 = 2) 
  (Finset.range 1000)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_remainder_count_l96_9622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l96_9677

/-- Calculates the time (in seconds) for a train to cross an electric pole. -/
noncomputable def trainCrossingTime (trainLength : ℝ) (trainSpeed : ℝ) : ℝ :=
  trainLength / (trainSpeed * 1000 / 3600)

/-- Proves that a train of length 250 meters, traveling at 200 km/hr, crosses an electric pole in approximately 4.5 seconds. -/
theorem train_crossing_time_approx :
  let trainLength : ℝ := 250
  let trainSpeed : ℝ := 200
  let crossingTime := trainCrossingTime trainLength trainSpeed
  ∃ ε > 0, |crossingTime - 4.5| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l96_9677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_and_triangle_area_l96_9691

/-- Pyramid with rectangular base -/
structure RectangularPyramid where
  baseLength : ℝ
  baseWidth : ℝ
  height : ℝ
  heightPerpendicular : Prop

/-- Calculate the volume of a rectangular pyramid -/
noncomputable def pyramidVolume (p : RectangularPyramid) : ℝ :=
  (1 / 3) * p.baseLength * p.baseWidth * p.height

/-- Calculate the area of a right triangle -/
noncomputable def rightTriangleArea (base width : ℝ) : ℝ :=
  (1 / 2) * base * width

/-- Theorem about the volume of pyramid PABCD and area of triangle ABC -/
theorem pyramid_volume_and_triangle_area :
  ∀ (p : RectangularPyramid),
    p.baseLength = 12 ∧
    p.baseWidth = 6 ∧
    p.height = 10 ∧
    p.heightPerpendicular →
    pyramidVolume p = 240 ∧
    rightTriangleArea p.baseLength p.baseWidth = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_and_triangle_area_l96_9691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_game_shots_theorem_l96_9671

/-- Represents a basketball player's shooting performance -/
structure ShootingPerformance where
  initial_shots : ℕ
  initial_made : ℕ
  seventh_game_shots : ℕ
  final_average : ℚ

/-- Calculates the number of shots made in the seventh game -/
def shots_made_seventh_game (performance : ShootingPerformance) : ℕ :=
  let total_shots := performance.initial_shots + performance.seventh_game_shots
  let total_made := (total_shots : ℚ) * performance.final_average
  (⌊total_made⌋ : ℤ).toNat - performance.initial_made

/-- Theorem stating that under given conditions, the player made 9 shots in the seventh game -/
theorem seventh_game_shots_theorem (performance : ShootingPerformance) 
  (h1 : performance.initial_shots = 50)
  (h2 : performance.initial_made = 20)
  (h3 : performance.seventh_game_shots = 15)
  (h4 : performance.final_average = 45/100) :
  shots_made_seventh_game performance = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_game_shots_theorem_l96_9671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ground_area_calculation_l96_9635

/-- The area of ground covered by rainfall, given the depth of rainfall and volume of water --/
noncomputable def ground_area (rainfall_depth : ℝ) (water_volume : ℝ) : ℝ :=
  water_volume / rainfall_depth

/-- Theorem stating that the area of the ground is 15000 square meters --/
theorem ground_area_calculation :
  let rainfall_depth : ℝ := 0.05  -- 5 cm converted to meters
  let water_volume : ℝ := 750    -- 750 cubic meters
  ground_area rainfall_depth water_volume = 15000 := by
  -- Unfold the definition of ground_area
  unfold ground_area
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ground_area_calculation_l96_9635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_lines_l96_9645

/-- A line in 2D space represented by slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The perpendicular slope of a given slope -/
noncomputable def perpendicular_slope (m : ℝ) : ℝ := -1 / m

/-- The intersection point of two lines -/
noncomputable def intersection_point (l1 l2 : Line) : ℝ × ℝ :=
  let x := (l2.y_intercept - l1.y_intercept) / (l1.slope - l2.slope)
  let y := l1.slope * x + l1.y_intercept
  (x, y)

/-- Theorem: The intersection of y = 3x + 10 and its perpendicular line through (3, 2) is (-2.1, 3.7) -/
theorem intersection_of_lines :
  let l1 : Line := { slope := 3, y_intercept := 10 }
  let l2 : Line := 
    { slope := perpendicular_slope l1.slope,
      y_intercept := 2 - perpendicular_slope l1.slope * 3 }
  intersection_point l1 l2 = (-2.1, 3.7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_lines_l96_9645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_sixty_percent_l96_9616

noncomputable section

/-- The side length of the square ABCD -/
def side_length : ℝ := 5

/-- The area of the entire square ABCD -/
def total_area : ℝ := side_length ^ 2

/-- The area of the first shaded part (1x1 square) -/
def shaded_area_1 : ℝ := 1 ^ 2 - 0 ^ 2

/-- The area of the second shaded part (3x3 square with 2x2 removed) -/
def shaded_area_2 : ℝ := 3 ^ 2 - 2 ^ 2

/-- The area of the third shaded part (5x5 square with 4x4 removed) -/
def shaded_area_3 : ℝ := 5 ^ 2 - 4 ^ 2

/-- The total shaded area -/
def total_shaded_area : ℝ := shaded_area_1 + shaded_area_2 + shaded_area_3

/-- The percentage of the square that is shaded -/
def shaded_percentage : ℝ := (total_shaded_area / total_area) * 100

theorem shaded_area_is_sixty_percent :
  shaded_percentage = 60 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_sixty_percent_l96_9616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_plus_icing_area_equals_36_l96_9634

/-- Represents a trapezoidal piece cut from a cube --/
structure TrapezoidalPiece where
  cube_edge : ℝ
  small_base : ℝ
  large_base : ℝ
  height : ℝ

/-- Calculates the volume of the trapezoidal piece --/
noncomputable def volume (p : TrapezoidalPiece) : ℝ :=
  ((p.small_base + p.large_base) / 2) * p.height * p.cube_edge

/-- Calculates the icing area of the trapezoidal piece --/
noncomputable def icing_area (p : TrapezoidalPiece) : ℝ :=
  ((p.small_base + p.large_base) / 2) * p.height +
  p.small_base * p.cube_edge +
  p.large_base * p.cube_edge +
  p.height * p.cube_edge

/-- Theorem stating that the sum of volume and icing area is 36 for the given cube --/
theorem volume_plus_icing_area_equals_36 (p : TrapezoidalPiece)
    (h1 : p.cube_edge = 3)
    (h2 : p.small_base = 1)
    (h3 : p.large_base = 2)
    (h4 : p.height = 3) :
    volume p + icing_area p = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_plus_icing_area_equals_36_l96_9634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sheridan_has_37_cats_l96_9695

/-- The number of cats Mrs. Garrett has -/
def garrett_cats : ℕ := 24

/-- The difference between Mrs. Sheridan's cats and Mrs. Garrett's cats -/
def cat_difference : ℕ := 13

/-- Mrs. Sheridan's number of cats -/
def sheridan_cats : ℕ := garrett_cats + cat_difference

theorem sheridan_has_37_cats : sheridan_cats = 37 := by
  rfl  -- reflexivity proves this trivial equality


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sheridan_has_37_cats_l96_9695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l96_9643

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  (∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M)) :=
by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l96_9643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l96_9644

-- Define a power function
noncomputable def power_function (α : ℝ) : ℝ → ℝ := λ x ↦ x ^ α

-- Theorem statement
theorem power_function_value (α : ℝ) :
  power_function α 4 = (1/2) → power_function α (1/4) = 2 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l96_9644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l96_9619

/-- Given positive differentiable functions f and g on ℝ, 
    if f'(x)g(x) - f(x)g'(x) < 0 for all x, 
    then f(x)g(b) > f(b)g(x) for a < x < b -/
theorem inequality_theorem 
  (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g)
  (hpos_f : ∀ x, f x > 0)
  (hpos_g : ∀ x, g x > 0)
  (h_deriv : ∀ x, deriv f x * g x - f x * deriv g x < 0)
  {a b x : ℝ} 
  (hx : a < x ∧ x < b) : 
  f x * g b > f b * g x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l96_9619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_relation_l96_9690

theorem cube_volume_relation (base_cube_volume : ℝ) (target_cube_surface_area_factor : ℝ) : 
  base_cube_volume = 8 →
  target_cube_surface_area_factor = 3 →
  (let base_cube_side := base_cube_volume ^ (1/3 : ℝ)
   let base_cube_surface_area := 6 * base_cube_side ^ 2
   let target_cube_surface_area := target_cube_surface_area_factor * base_cube_surface_area
   let target_cube_side := (target_cube_surface_area / 6) ^ (1/2 : ℝ)
   let target_cube_volume := target_cube_side ^ 3
   target_cube_volume) = 24 * Real.sqrt 3 := by
  sorry

#check cube_volume_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_relation_l96_9690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frank_original_amount_l96_9608

/-- The original amount of money Frank had, given his spending pattern and final balance. -/
def original_amount (final_balance : ℚ) : ℚ :=
  final_balance * 70 / 6

/-- Theorem stating that given Frank's spending pattern and final balance, his original amount was as calculated. -/
theorem frank_original_amount :
  let final_balance : ℚ := 600
  let calculated_amount : ℚ := original_amount final_balance
  ∃ (x : ℚ),
    x > 0 ∧
    (4/5 * x) > 0 ∧
    (3/4 * 4/5 * x) > 0 ∧
    (6/7 * 3/4 * 4/5 * x) > 0 ∧
    (2/3 * 6/7 * 3/4 * 4/5 * x) = final_balance ∧
    x = calculated_amount := by
  sorry

#eval original_amount 600

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frank_original_amount_l96_9608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_capri_sun_cost_theorem_l96_9662

/-- Calculates the cost per pouch in cents given the number of boxes, pouches per box,
    discount rate, tax rate, and total paid amount. -/
def cost_per_pouch (boxes : ℕ) (pouches_per_box : ℕ) (discount_rate : ℚ) 
                   (tax_rate : ℚ) (total_paid : ℚ) : ℚ :=
  let total_pouches := boxes * pouches_per_box
  let discounted_price := total_paid / (1 + tax_rate) / (1 - discount_rate)
  (discounted_price / total_pouches) * 100

theorem capri_sun_cost_theorem :
  let boxes := 10
  let pouches_per_box := 6
  let discount_rate := 15 / 100
  let tax_rate := 8 / 100
  let total_paid := 12
  Int.floor (cost_per_pouch boxes pouches_per_box discount_rate tax_rate total_paid) = 21 := by
  sorry

#eval Int.floor (cost_per_pouch 10 6 (15/100) (8/100) 12)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_capri_sun_cost_theorem_l96_9662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_radius_l96_9625

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the spheres
structure Sphere where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def SphereProblem (t : Triangle) (s1 s2 s3 : Sphere) : Prop :=
  -- Spheres 1 and 2 touch the plane of triangle ABC at B and C
  (s1.center = t.B ∧ s2.center = t.C) ∧
  -- Sum of radii of spheres 1 and 2 is 7
  (s1.radius + s2.radius = 7) ∧
  -- Distance between centers of spheres 1 and 2 is 17
  (Real.sqrt ((s1.center.1 - s2.center.1)^2 + (s1.center.2 - s2.center.2)^2) = 17) ∧
  -- Center of sphere 3 is at point A with radius 8
  (s3.center = t.A ∧ s3.radius = 8) ∧
  -- Sphere 3 externally touches spheres 1 and 2
  (Real.sqrt ((s3.center.1 - s1.center.1)^2 + (s3.center.2 - s1.center.2)^2) = s3.radius + s1.radius) ∧
  (Real.sqrt ((s3.center.1 - s2.center.1)^2 + (s3.center.2 - s2.center.2)^2) = s3.radius + s2.radius)

-- Theorem statement
theorem circumcircle_radius (t : Triangle) (s1 s2 s3 : Sphere) 
  (h : SphereProblem t s1 s2 s3) : 
  ∃ (center : ℝ × ℝ), 
    Real.sqrt ((center.1 - t.A.1)^2 + (center.2 - t.A.2)^2) = 2 * Real.sqrt 15 ∧
    Real.sqrt ((center.1 - t.B.1)^2 + (center.2 - t.B.2)^2) = 2 * Real.sqrt 15 ∧
    Real.sqrt ((center.1 - t.C.1)^2 + (center.2 - t.C.2)^2) = 2 * Real.sqrt 15 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_radius_l96_9625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_tan_sin_inequality_l96_9640

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := tan (tan x) - tan x + sin x - sin (sin x)

-- State the theorem
theorem tan_tan_sin_inequality {x : ℝ} (h : x ∈ Set.Ioo 0 (π/2)) : f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_tan_sin_inequality_l96_9640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_valid_row_sizes_eq_92_l96_9687

/-- The sum of all valid row sizes for a graduation ceremony seating arrangement --/
def sum_valid_row_sizes : ℕ :=
  (Finset.filter (fun x : ℕ => 
    x ∣ 360 ∧ 
    x ≥ 18 ∧ 
    360 / x ≥ 12) 
  (Finset.range 361)).sum id

/-- The sum of all valid row sizes for the graduation ceremony is 92 --/
theorem sum_valid_row_sizes_eq_92 : sum_valid_row_sizes = 92 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_valid_row_sizes_eq_92_l96_9687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_parallel_lines_l96_9627

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a plane -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Predicate to check if a circle is tangent to a line -/
def isTangent (c : Circle) (l : Line) : Prop := sorry

/-- The locus of centers of circles tangent to a fixed line -/
def locus (a : ℝ) (l : Line) : Set (ℝ × ℝ) :=
  {p | ∃ c : Circle, c.radius = a ∧ c.center = p ∧ isTangent c l}

/-- Two parallel lines in a plane -/
structure ParallelLines where
  line1 : Line
  line2 : Line
  isParallel : line1.direction = line2.direction

/-- Convert a Line to a Set of points -/
def Line.toSet (l : Line) : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, p = l.point + t • l.direction}

/-- Main theorem: The locus is two parallel lines -/
theorem locus_is_parallel_lines (a : ℝ) (l : Line) :
  ∃ pl : ParallelLines, locus a l = pl.line1.toSet ∪ pl.line2.toSet := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_parallel_lines_l96_9627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_equation_l96_9629

-- Define the circles in polar coordinates
noncomputable def circle_O1 (θ : ℝ) : ℝ := 4 * Real.cos θ
noncomputable def circle_O2 (θ : ℝ) : ℝ := -Real.sin θ

-- Define the conversion from polar to Cartesian coordinates
noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the circles in Cartesian coordinates
def circle_O1_cartesian (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x = 0

def circle_O2_cartesian (x y : ℝ) : Prop :=
  x^2 + y^2 + y = 0

-- State the theorem
theorem intersection_line_equation :
  ∀ x y : ℝ,
  (circle_O1_cartesian x y ∧ circle_O2_cartesian x y) →
  4*x + y = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_equation_l96_9629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_7_eq_28_l96_9660

/-- An arithmetic sequence with specified terms -/
structure ArithSeq where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  a3_eq_3 : a 3 = 3
  a10_eq_10 : a 10 = 10

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithSeq) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

/-- The main theorem: S₇ = 28 for the given arithmetic sequence -/
theorem sum_7_eq_28 (seq : ArithSeq) : sum_n_terms seq 7 = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_7_eq_28_l96_9660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_at_one_implies_a_range_no_a_exists_for_nonpositive_f_l96_9694

-- Define the function f as noncomputable
noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * (2*a + 1) * x^2 - 2*(a + 1) * x

-- Theorem for part 1
theorem max_at_one_implies_a_range (a : ℝ) :
  (∀ x : ℝ, f a x ≤ f a 1) → a < -3/2 :=
by sorry

-- Theorem for part 2
theorem no_a_exists_for_nonpositive_f :
  ¬ ∃ a : ℝ, ∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ f a x ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_at_one_implies_a_range_no_a_exists_for_nonpositive_f_l96_9694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l96_9615

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 1 + Real.log x / Real.log 5
  else 2 * x - 1

-- State the theorem
theorem problem_solution (m : ℝ) : f (f 0 + m) = 2 → m = 6 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l96_9615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_AB_l96_9665

/-- The distance between two points A and B, where two people walk between them meeting twice --/
def distance_AB : ℝ := 21

/-- The distance of the first meeting point from A --/
def first_meeting : ℝ := 10

/-- The distance of the second meeting point from B --/
def second_meeting : ℝ := 12

/-- Theorem stating that the distance between A and B is 21 km --/
theorem distance_between_AB : 
  ∀ (v_A v_B : ℝ), v_A > 0 → v_B > 0 →
  (first_meeting / v_A = (distance_AB - first_meeting) / v_B) →
  ((2 * distance_AB - second_meeting) / v_A = (distance_AB + second_meeting) / v_B) →
  distance_AB = 21 := by
  sorry

#check distance_between_AB

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_AB_l96_9665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_min_value_g_min_set_tan_x_plus_pi_4_l96_9618

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sin x + cos x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := cos x - sin x

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f x * f' x

-- Theorem for the minimum value of g(x)
theorem g_min_value : ∃ (x : ℝ), g x = -1 ∧ ∀ (y : ℝ), g y ≥ -1 := by sorry

-- Theorem for the set of x values that minimize g(x)
theorem g_min_set : ∀ (x : ℝ), g x = -1 ↔ ∃ (k : ℤ), x = -π/2 + k*π := by sorry

-- Theorem for tan(x + π/4) when f(x) = 2f'(x)
theorem tan_x_plus_pi_4 : ∀ (x : ℝ), f x = 2 * f' x → tan (x + π/4) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_min_value_g_min_set_tan_x_plus_pi_4_l96_9618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_range_of_m_l96_9681

noncomputable def f (x : ℝ) : ℝ := (Real.log (2 * x)) / x

-- Theorem for the minimum value of f(x) on [1, a]
theorem min_value_f (a : ℝ) (h : a > 1) :
  (∃ (x : ℝ), x ∈ Set.Icc 1 a ∧ 
    ∀ (y : ℝ), y ∈ Set.Icc 1 a → f x ≤ f y) ∧
  (a ≤ 2 → ∀ (x : ℝ), x ∈ Set.Icc 1 a → f x ≥ Real.log 2) ∧
  (a > 2 → ∀ (x : ℝ), x ∈ Set.Icc 1 a → f x ≥ f a) := by
  sorry

-- Theorem for the range of m
theorem range_of_m :
  ∃ (m : ℝ), -Real.log 2 < m ∧ m ≤ -(1/3) * Real.log 6 ∧
    (∀ (x : ℤ), (f (x : ℝ))^2 + m * f (x : ℝ) > 0) ∧
    (∃ (x y : ℤ), x ≠ y ∧ (f (x : ℝ))^2 + m * f (x : ℝ) > 0 ∧ (f (y : ℝ))^2 + m * f (y : ℝ) > 0) ∧
    (∀ (z : ℤ), z ≠ x ∧ z ≠ y → (f (z : ℝ))^2 + m * f (z : ℝ) ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_range_of_m_l96_9681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_monotonicity_of_f_and_range_of_m_l96_9663

noncomputable def f (x : ℝ) := x - 6 / (x + 1)

noncomputable def g (m : ℝ) (x : ℝ) := x^2 - m*x + m

theorem symmetry_and_monotonicity_of_f_and_range_of_m :
  -- The center of symmetry of f(x) is (-1, -1)
  (∀ x, f (-1 + x) + f (-1 - x) = -2) ∧
  -- f(x) is increasing on (0, +∞)
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
  -- The range of m for which g(x₁) = f(x₂) holds is [-2, 4]
  (∃ m₁ m₂, m₁ = -2 ∧ m₂ = 4 ∧
    ∀ m, m₁ ≤ m ∧ m ≤ m₂ ↔
      (∀ x₁, x₁ ∈ Set.Icc 0 2 →
        ∃ x₂, x₂ ∈ Set.Icc 1 5 ∧ g m x₁ = f x₂) ∧
      (∀ x, x ∈ Set.Icc 0 1 → g m (2 - x) = g m x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_monotonicity_of_f_and_range_of_m_l96_9663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_minor_arc_theorem_l96_9680

-- Define the circle perimeter
def circle_perimeter : ℚ := 3

-- Define the probability function
noncomputable def prob_minor_arc_less_than_one (perimeter : ℚ) : ℚ := 1 / 3

-- Theorem statement
theorem prob_minor_arc_theorem :
  prob_minor_arc_less_than_one circle_perimeter = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_minor_arc_theorem_l96_9680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_score_calculation_l96_9624

/-- Represents a test with a maximum score and a direct proportion between preparation time and score up to the maximum. -/
structure Test where
  maxScore : ℚ
  maxPossibleScore : ℚ
  scorePerHour : ℚ

/-- Calculates the score for a given preparation time. -/
def calculateScore (test : Test) (hours : ℚ) : ℚ :=
  min test.maxPossibleScore (test.scorePerHour * hours)

theorem test_score_calculation (test : Test) (h1 : test.maxScore = 150) 
    (h2 : test.maxPossibleScore = 140) (h3 : calculateScore test 5 = 90) :
  calculateScore test 7 = 126 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_score_calculation_l96_9624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l96_9652

def A : Set ℕ := {x | x ≤ 3}
def B : Set ℕ := {0, 1, 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l96_9652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l96_9630

variable (a b : ℝ × ℝ)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def vector_length (v : ℝ × ℝ) : ℝ := Real.sqrt (dot_product v v)

noncomputable def angle_between (v w : ℝ × ℝ) : ℝ := Real.arccos ((dot_product v w) / (vector_length v * vector_length w))

theorem angle_between_vectors (h1 : dot_product (2 • a - b) (a + b) = 6)
                              (h2 : vector_length a = 2)
                              (h3 : vector_length b = 1) :
  angle_between a b = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l96_9630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l96_9682

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + Real.log x
def g (a : ℝ) (x : ℝ) : ℝ := x^2 - f a x

-- State the theorem
theorem function_properties :
  (∃ (x : ℝ), ∀ (y : ℝ), f 3 x ≤ f 3 y) ∧
  f 3 1 = -2 ∧
  (∃ (a : ℝ), a = 1 ∧ 
    (∀ (x : ℝ), x ∈ Set.Icc 1 (Real.exp 1) → g a x ≥ 1) ∧
    (∃ (x : ℝ), x ∈ Set.Icc 1 (Real.exp 1) ∧ g a x = 1)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l96_9682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l96_9699

theorem trig_problem (θ : ℝ) (h : Real.cos (2*θ) + Real.cos θ = 0) :
  Real.sin (2*θ) + Real.sin θ ∈ ({0, Real.sqrt 3, -Real.sqrt 3} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l96_9699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l96_9698

/-- The directrix of a parabola y = ax^2 is given by y = -1/(4a) -/
noncomputable def directrix_equation (a : ℝ) : ℝ := -1 / (4 * a)

/-- For a parabola y = 2x^2, prove that its directrix has the equation y = -1/8 -/
theorem parabola_directrix (x y : ℝ) :
  y = 2 * x^2 → directrix_equation 2 = -1/8 := by
  intro h
  unfold directrix_equation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l96_9698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lily_coffee_money_l96_9626

/-- Represents the cost of an item including tax -/
structure ItemCost where
  pretax : ℚ
  tax : ℚ
  deriving Repr

/-- Calculates the total cost of an item including tax -/
def totalCost (item : ItemCost) : ℚ :=
  item.pretax + item.tax

/-- Represents Lily's shopping trip -/
structure LilyShopping where
  budget : ℚ := 85
  taxRate : ℚ := 6/100
  celery : ItemCost := ⟨6, 6 * (6/100)⟩
  cereal : ItemCost := ⟨14, 14 * (6/100)⟩
  bread : ItemCost := ⟨15.30, 15.30 * (6/100)⟩
  milk : ItemCost := ⟨10, 10 * (6/100)⟩
  potatoes : ItemCost := ⟨7.50, 7.50 * (6/100)⟩
  cookies : ItemCost := ⟨15, 15 * (6/100)⟩
  deriving Repr

theorem lily_coffee_money (shopping : LilyShopping) :
  shopping.budget - (totalCost shopping.celery + totalCost shopping.cereal +
    totalCost shopping.bread + totalCost shopping.milk +
    totalCost shopping.potatoes + totalCost shopping.cookies) = 12.13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lily_coffee_money_l96_9626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perfect_square_sequence_no_perfect_square_sequence_with_constant_l96_9664

theorem no_perfect_square_sequence (a b : ℕ) : 
  ¬(∀ n : ℕ, ∃ k : ℕ, (a * 2^n + b * 5^n : ℕ) = k^2) :=
sorry

theorem no_perfect_square_sequence_with_constant (a b c : ℕ) :
  ¬(∀ n : ℕ, ∃ k : ℕ, (a * 2^n + b * 5^n + c : ℕ) = k^2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perfect_square_sequence_no_perfect_square_sequence_with_constant_l96_9664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daughter_cookies_l96_9656

def total_cookies : ℕ := 200
def wife_percentage : ℚ := 30 / 100
def uneaten_cookies : ℕ := 50

theorem daughter_cookies :
  ∃ (daughter_cookies : ℕ),
    daughter_cookies = total_cookies - 
      (wife_percentage * total_cookies).floor - 
      ((total_cookies - (wife_percentage * total_cookies).floor - daughter_cookies) / 2) - 
      uneaten_cookies :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_daughter_cookies_l96_9656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l96_9646

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

noncomputable def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l96_9646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l96_9692

/-- Given two vectors a and b in ℝ², prove properties about their sum and angle between linear combinations. -/
theorem vector_properties (a b : ℝ × ℝ) 
    (ha : a = (2, 0)) 
    (hb : b = (1/2, Real.sqrt 3/2)) : 
  let sum := (a.1 + b.1, a.2 + b.2)
  let magnitude := Real.sqrt ((sum.1)^2 + (sum.2)^2)
  let unit_vector := (sum.1 / magnitude, sum.2 / magnitude)
  (unit_vector = (5 * Real.sqrt 7 / 14, Real.sqrt 21 / 14) ∧ 
  ∀ t : ℝ, (2*t*a.1 + 7*b.1) * (a.1 + t*b.1) + 
           (2*t*a.2 + 7*b.2) * (a.2 + t*b.2) < 0 ↔ 
    (t > -7 ∧ t < -Real.sqrt 14/2) ∨ (t > -Real.sqrt 14/2 ∧ t < -1/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l96_9692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l96_9633

/-- Given that in the expansion of (√x + 1/x)^n, the sum of coefficients
    of all quadratic terms is 64, prove that the constant term is 15. -/
theorem constant_term_of_expansion (n : ℕ) : 
  (∃ (coeff_sum : ℕ), coeff_sum = 64 ∧ 
    coeff_sum = (Finset.sum (Finset.range (n+1)) (λ i => 
      (n.choose i) * (if i % 3 = 2 then 1 else 0)))) →
  ((n.choose (n/3)) : ℕ) = 15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_expansion_l96_9633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_selected_individual_is_one_l96_9693

/-- Represents a random number table as a list of integers -/
def RandomNumberTable : List (List Nat) :=
  [[78, 16, 65, 72, 08, 02, 63, 14, 07, 02, 43, 69, 97, 28, 01, 98],
   [32, 04, 92, 34, 49, 35, 82, 00, 36, 23, 48, 69, 69, 38, 74, 81]]

/-- Represents the population size -/
def PopulationSize : Nat := 20

/-- Represents the number of individuals to be selected -/
def SampleSize : Nat := 5

/-- Function to check if a number is valid (less than or equal to PopulationSize) -/
def isValidNumber (n : Nat) : Bool :=
  n ≤ PopulationSize ∧ n ≠ 0

/-- Function to select valid numbers from the random number table -/
def selectValidNumbers (table : List (List Nat)) : List Nat :=
  (table.join.map (fun n => if n < 10 then n else n % 100))
    |> List.filter isValidNumber
    |> List.take SampleSize

/-- The main theorem to prove -/
theorem fifth_selected_individual_is_one :
  (selectValidNumbers RandomNumberTable).get? 4 = some 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_selected_individual_is_one_l96_9693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_winning_strategy_l96_9666

/-- Represents a card in a standard deck -/
inductive Card
  | Heart
  | Diamond
  | Spade
  | Club

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (cards : Finset Card)
  (well_shuffled : cards.card = 52)

/-- Represents a strategy for deciding when to stop -/
def Strategy := List Card → Bool

/-- The probability of winning given a strategy and a deck -/
noncomputable def winningProbability (s : Strategy) (d : Deck) : ℝ := sorry

/-- Theorem stating that no strategy can have a winning probability greater than 0.5 -/
theorem no_winning_strategy (s : Strategy) (d : Deck) : 
  winningProbability s d ≤ 0.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_winning_strategy_l96_9666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_20_formula_l96_9669

noncomputable def v : ℕ → ℝ → ℝ
  | 0, _ => 0  -- Define a base case for 0
  | 1, b => b
  | n + 2, b => -2 / (v (n + 1) b + 2)

theorem v_20_formula (b : ℝ) (h : b > 0) :
  v 20 b = -(2 * b - 4) / (b - 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_20_formula_l96_9669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l96_9685

-- Define the circles C₁ and C₂
noncomputable def circle_C1 (c : ℝ) (x y : ℝ) : Prop := x^2 + 2*c*x + y^2 = 0
noncomputable def circle_C2 (c : ℝ) (x y : ℝ) : Prop := x^2 - 2*c*x + y^2 = 0

-- Define the ellipse C
noncomputable def ellipse_C (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define the semi-latus rectum
noncomputable def semi_latus_rectum (a b : ℝ) : ℝ := b^2/a

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2/a^2)

-- Theorem statement
theorem eccentricity_range (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y, circle_C1 c x y → ellipse_C a b x y) →
  (∀ x y, circle_C2 c x y → ellipse_C a b x y) →
  c = semi_latus_rectum a b →
  0 < eccentricity a b ∧ eccentricity a b < 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l96_9685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_condition_max_min_values_non_monotonicity_condition_l96_9631

def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + 1) * (x + a)

theorem extreme_values_condition (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ ∀ z : ℝ, f a z ≤ max (f a x) (f a y)) ↔ (a > Real.sqrt 3 ∨ a < -Real.sqrt 3) :=
sorry

theorem max_min_values (a : ℝ) (h : (deriv (f a)) 1 = 0) :
  (∀ x ∈ Set.Icc (-1) (1/2), f a x ≤ 58/27) ∧
  (∀ x ∈ Set.Icc (-1) (1/2), f a x ≥ -2) ∧
  (∃ x ∈ Set.Icc (-1) (1/2), f a x = 58/27) ∧
  (∃ x ∈ Set.Icc (-1) (1/2), f a x = -2) :=
sorry

theorem non_monotonicity_condition (a : ℝ) :
  (∃ x y z : ℝ, x < y ∧ y < z ∧ x ≥ -1 ∧ z ≤ 1/2 ∧ f a x > f a y ∧ f a y < f a z) ↔ (a > Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_condition_max_min_values_non_monotonicity_condition_l96_9631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_f_increasing_neg_l96_9614

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / x

-- Theorem 1: Find the value of a
theorem find_a : ∃ a : ℝ, 2 * (f a 1) = f a 2 ∧ a = 3 := by
  -- The proof is omitted for now
  sorry

-- Theorem 2: Prove f is strictly increasing on (-∞, 0)
theorem f_increasing_neg : 
  ∀ x y : ℝ, x < y → x < 0 → y < 0 → f 3 x < f 3 y := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_f_increasing_neg_l96_9614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedra_lines_concur_l96_9672

-- Define the types for points and tetrahedra
def Point := ℝ × ℝ × ℝ
def Tetrahedron := Fin 4 → Point

-- Define the property of being a regular tetrahedron
def IsRegularTetrahedron (t : Tetrahedron) : Prop := sorry

-- Define the property of two tetrahedra not being congruent
def NotCongruent (t1 t2 : Tetrahedron) : Prop := sorry

-- Define the midpoint of a line segment
def Midpoint (a b : Point) : Point := sorry

-- Define a line passing through two points
def Line (a b : Point) : Set Point := sorry

-- Define the property of lines being concurrent
def AreConcurrent (l1 l2 l3 l4 : Set Point) : Prop := sorry

theorem tetrahedra_lines_concur
  (A B C : Tetrahedron)
  (hA : IsRegularTetrahedron A)
  (hB : IsRegularTetrahedron B)
  (hC : IsRegularTetrahedron C)
  (hAB : NotCongruent A B)
  (hBC : NotCongruent B C)
  (hAC : NotCongruent A C)
  (hMidpoint : ∀ i : Fin 4, C i = Midpoint (A i) (B i)) :
  AreConcurrent
    (Line (A 0) (B 0))
    (Line (A 1) (B 1))
    (Line (A 2) (B 2))
    (Line (A 3) (B 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedra_lines_concur_l96_9672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_pi_third_value_l96_9603

noncomputable def f (ω φ : ℝ) : ℝ → ℝ := fun x ↦ (1/2) * Real.cos (ω * x + φ)

noncomputable def g (ω φ : ℝ) : ℝ → ℝ := fun x ↦ 3 * Real.sin (ω * x + φ) - 2

theorem g_pi_third_value (ω φ : ℝ) 
  (h : ∀ x, f ω φ (π/3 - x) = f ω φ (π/3 + x)) : 
  g ω φ (π/3) = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_pi_third_value_l96_9603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rooks_is_16_l96_9678

/-- A configuration of rooks on a 10x10 board. -/
def RookConfiguration := Fin 10 → Fin 10 → Bool

/-- A cell is under attack if there's a rook in its row or column. -/
def isUnderAttack (config : RookConfiguration) (row col : Fin 10) : Prop :=
  ∃ i : Fin 10, config i col ∨ config row i

/-- The number of rooks in a configuration. -/
def rookCount (config : RookConfiguration) : Nat :=
  Finset.sum (Finset.univ : Finset (Fin 10)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 10)) fun j =>
      if config i j then 1 else 0

/-- A configuration is valid if removing any rook leaves at least one cell no longer under attack. -/
def isValidConfiguration (config : RookConfiguration) : Prop :=
  ∀ rook_row rook_col : Fin 10, config rook_row rook_col →
    ∃ cell_row cell_col : Fin 10,
      isUnderAttack config cell_row cell_col ∧
      ¬isUnderAttack (Function.update config rook_row (Function.const _ false)) cell_row cell_col

/-- The maximum number of rooks in a valid configuration is 16. -/
theorem max_rooks_is_16 :
  ∃ config : RookConfiguration, isValidConfiguration config ∧ rookCount config = 16 ∧
  ∀ other : RookConfiguration, isValidConfiguration other → rookCount other ≤ 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rooks_is_16_l96_9678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_potato_yield_calculation_l96_9667

-- Define the garden dimensions in steps
def garden_length_steps : ℕ := 18
def garden_width_steps : ℕ := 25

-- Define the stride length in inches
def stride_length_inches : ℕ := 30

-- Define the yield criteria
noncomputable def yield_rate (area : ℝ) : ℝ :=
  if area ≤ 1200 then 0.4 else 0.6

-- Define the function to calculate the expected potato yield
noncomputable def expected_potato_yield : ℝ :=
  let steps_to_feet : ℝ := (stride_length_inches : ℝ) / 12
  let garden_length_feet : ℝ := (garden_length_steps : ℝ) * steps_to_feet
  let garden_width_feet : ℝ := (garden_width_steps : ℝ) * steps_to_feet
  let garden_area : ℝ := garden_length_feet * garden_width_feet
  garden_area * yield_rate garden_area

-- Theorem statement
theorem potato_yield_calculation :
  expected_potato_yield = 1687.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_potato_yield_calculation_l96_9667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l96_9637

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | 3 * x - 9 ≥ 0}

def B : Set ℝ := {x | 2 * x - 13 < 1}

theorem set_operations :
  (A ∩ B = {x : ℝ | 3 ≤ x ∧ x < 7}) ∧
  (A ∪ B = U) ∧
  ((Set.univ \ A) ∩ B = {x : ℝ | x < 3}) ∧
  ((Set.univ \ A) ∪ B = {x : ℝ | x < 7}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l96_9637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l96_9641

/-- A hyperbola passing through the point (2, √2) with an asymptote y = x has the equation x²/2 - y²/2 = 1 -/
theorem hyperbola_equation (h : Set (ℝ × ℝ)) 
  (contains_point : (2, Real.sqrt 2) ∈ h)
  (asymptote : ∃ (m b : ℝ), m = 1 ∧ (∀ (x y : ℝ), y = m * x + b → (x, y) ∉ h ∧ (∀ ε > 0, ∃ (x' y' : ℝ), (x', y') ∈ h ∧ |(y - y') - m * (x - x')| < ε))) :
  h = {(x, y) | x^2/2 - y^2/2 = 1} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l96_9641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_is_five_sixteenths_l96_9675

-- Define a sphere with volume V
structure Sphere where
  volume : ℝ
  volume_pos : volume > 0

-- Define the configuration of two intersecting spheres
structure IntersectingSpheres where
  sphere1 : Sphere
  sphere2 : Sphere
  centers_on_surface : True  -- This represents the condition that the centers are on each other's surface

-- Define the volume of intersection
noncomputable def intersection_volume (spheres : IntersectingSpheres) : ℝ :=
  (5 / 16) * spheres.sphere1.volume

-- Theorem statement
theorem intersection_volume_is_five_sixteenths (spheres : IntersectingSpheres) :
  intersection_volume spheres = (5 / 16) * spheres.sphere1.volume :=
by
  -- Unfold the definition of intersection_volume
  unfold intersection_volume
  -- The rest of the proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_is_five_sixteenths_l96_9675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_athlete_high_jump_l96_9607

/-- Represents an athlete's jump distances -/
structure AthleteJumps where
  long_jump : ℝ
  triple_jump : ℝ
  high_jump : ℝ

/-- Calculates the average jump distance for an athlete -/
noncomputable def average_jump (jumps : AthleteJumps) : ℝ :=
  (jumps.long_jump + jumps.triple_jump + jumps.high_jump) / 3

theorem first_athlete_high_jump
  (first_athlete : AthleteJumps)
  (second_athlete : AthleteJumps)
  (winner_average : ℝ)
  (h1 : first_athlete.long_jump = 26)
  (h2 : first_athlete.triple_jump = 30)
  (h3 : second_athlete.long_jump = 24)
  (h4 : second_athlete.triple_jump = 34)
  (h5 : second_athlete.high_jump = 8)
  (h6 : winner_average = 22)
  (h7 : average_jump first_athlete = winner_average) :
  first_athlete.high_jump = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_athlete_high_jump_l96_9607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_principal_calculation_l96_9657

/-- Calculates the total interest rate for a loan with varying interest rates over different periods -/
def total_interest_rate (rate1 rate2 rate3 : ℚ) (period1 period2 period3 : ℕ) : ℚ :=
  rate1 * period1 + rate2 * period2 + rate3 * period3

/-- Calculates the principal amount given the total interest and interest rate -/
def calculate_principal (total_interest : ℚ) (interest_rate : ℚ) : ℚ :=
  total_interest / interest_rate

theorem loan_principal_calculation 
  (rate1 rate2 rate3 : ℚ) 
  (period1 period2 period3 : ℕ) 
  (total_interest : ℚ) :
  let interest_rate := total_interest_rate rate1 rate2 rate3 period1 period2 period3
  let principal := calculate_principal total_interest interest_rate
  rate1 = 8 / 100 ∧ 
  rate2 = 10 / 100 ∧ 
  rate3 = 12 / 100 ∧
  period1 = 4 ∧ 
  period2 = 6 ∧ 
  period3 = 5 ∧
  total_interest = 12160 
  → principal = 8000 := by
  sorry

#eval total_interest_rate (8/100) (10/100) (12/100) 4 6 5
#eval calculate_principal 12160 (152/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_principal_calculation_l96_9657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sequence_sum_2017_l96_9647

/-- A sequence of operations, where true represents addition and false represents subtraction. -/
def OperationSequence := List Bool

/-- Applies a sequence of operations to a list of integers. -/
def applyOperations (nums : List Int) (ops : OperationSequence) : Int :=
  match ops with
  | [] => nums.head!
  | op :: rest =>
    let result := nums.head!
    let remaining := nums.tail!
    remaining.zip rest |>.foldl (fun acc (n, o) => if o then acc + n else acc - n) result

theorem exists_sequence_sum_2017 :
  ∃ (ops : OperationSequence),
    ops.length = 78 ∧
    applyOperations (List.replicate 79 1) ops = 2017 := by
  sorry

#eval applyOperations (List.replicate 79 1) (List.replicate 78 true)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sequence_sum_2017_l96_9647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_x_plus_three_l96_9649

theorem integral_sqrt_x_plus_three : ∫ x in (0:ℝ)..(1:ℝ), (Real.sqrt x + 3) = 11/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_x_plus_three_l96_9649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l96_9623

noncomputable def f (x y : ℝ) : ℝ := (x^2 * y) / (x^3 + y^3)

theorem min_value_of_f :
  ∃ (min : ℝ), min = 12/35 ∧
  ∀ (x y : ℝ), 1/3 ≤ x ∧ x ≤ 2/3 ∧ 1/4 ≤ y ∧ y ≤ 1/2 →
  f x y ≥ min := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l96_9623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_ratio_l96_9602

theorem circle_radius_ratio (square_area : Real) (small_circle_circumference : Real) 
  (h1 : square_area = 784) 
  (h2 : small_circle_circumference = 8) : Real := by
  -- Define the radius of the larger circle
  let large_radius : Real := Real.sqrt square_area / 2
  -- Define the radius of the smaller circle
  let small_radius : Real := small_circle_circumference / (2 * Real.pi)
  -- The ratio of the radii
  let ratio : Real := large_radius / small_radius
  -- Prove that this ratio equals 3.5π
  have : ratio = 3.5 * Real.pi := by sorry
  exact ratio


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_ratio_l96_9602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l96_9620

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k + Real.sqrt (x + 2)

-- Define the theorem
theorem range_of_k (k : ℝ) :
  (∃ (D : Set ℝ) (a b : ℝ), 
    (Set.Icc a b).Subset D ∧
    (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc a b ∩ D, f k x = y)) →
  k ∈ Set.Ioo (-9/4) (-2) ∪ {-2} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l96_9620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_and_triangle_properties_l96_9650

-- Define the points
def origin : ℝ × ℝ := (0, 0)
def point1 : ℝ × ℝ := (12, -5)
def point2 : ℝ × ℝ := (12, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem distance_and_triangle_properties :
  (distance origin point1 = 13) ∧
  (∃ (a b c : ℝ × ℝ), a = origin ∧ b = point2 ∧ c = point1 ∧
    (distance a b)^2 + (distance b c)^2 = (distance a c)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_and_triangle_properties_l96_9650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_quadrilateral_area_l96_9632

/-- Three non-intersecting circles consecutively inscribed in a 60° angle -/
structure InscribedCircles where
  S₁ : Real
  S₂ : Real
  S₃ : Real
  angle : Real
  inscribed : Bool
  angle_measure : angle = 60

/-- The quadrilateral formed by intersection points of common internal tangents -/
def TangentQuadrilateral (circles : InscribedCircles) : Real := 
  sorry

/-- The radius of the middle circle -/
def middle_radius (circles : InscribedCircles) : Real := circles.S₂

/-- The difference between radii of the largest and smallest circles -/
def radii_difference (circles : InscribedCircles) : Real := circles.S₃ - circles.S₁

/-- Theorem stating the area of the tangent quadrilateral -/
theorem tangent_quadrilateral_area (circles : InscribedCircles) :
  TangentQuadrilateral circles = (radii_difference circles) * (middle_radius circles) * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_quadrilateral_area_l96_9632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_coverage_criterion_l96_9609

/-- A hook is a shape made up of 6 unit squares. -/
structure Hook where
  squares : Fin 6 → Nat × Nat

/-- A rectangle is defined by its width and height. -/
structure Rectangle where
  m : Nat
  n : Nat

/-- Predicate to check if hooks cover a rectangle without gaps or overlaps. -/
def CoverRectangle (rect : Rectangle) (hooks : List Hook) : Prop :=
  sorry

/-- Predicate to check if a rectangle can be covered by hooks. -/
def CanBeCoveredByHooks (rect : Rectangle) : Prop :=
  ∃ (hooks : List Hook), CoverRectangle rect hooks

theorem rectangle_coverage_criterion (rect : Rectangle) :
  CanBeCoveredByHooks rect ↔ 12 ∣ (rect.m * rect.n) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_coverage_criterion_l96_9609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_plus_x_plus_x_cubed_l96_9653

theorem integral_sqrt_plus_x_plus_x_cubed : 
  ∫ x in (Set.Icc 0 1), (Real.sqrt (1 - x^2) + x + x^3) = (π + 3) / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_plus_x_plus_x_cubed_l96_9653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_polygon_sides_l96_9659

/-- The number of sides in the original polygon -/
def original_sides (n : ℕ) : Prop := n ≥ 3

/-- The sum of interior angles of a polygon with n sides -/
def interior_angle_sum (n : ℕ) : ℝ := (n - 2) * 180

/-- Represents the removal of one angle from the original polygon -/
def angle_removed (m n : ℕ) : Prop := m = n + 1

theorem original_polygon_sides :
  ∀ n : ℕ, 
    (∃ m : ℕ, angle_removed m n ∧ interior_angle_sum n = 1080) →
    (original_sides 7 ∨ original_sides 8 ∨ original_sides 9) :=
by
  intro n h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_polygon_sides_l96_9659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_of_inscribed_triangles_l96_9606

/-- Parabola definition -/
def parabola (x : ℝ) : ℝ := x^2

/-- Point M on the parabola -/
def M : ℝ × ℝ := (1, 1)

/-- Condition for a point to be on the parabola -/
def on_parabola (p : ℝ × ℝ) : Prop :=
  p.2 = parabola p.1

/-- Condition for two vectors to be perpendicular -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- Definition of a right-angled triangle inscribed in the parabola -/
structure RightTriangleInParabola where
  A : ℝ × ℝ
  B : ℝ × ℝ
  hA : on_parabola A
  hB : on_parabola B
  hRight : perpendicular (A.1 - M.1, A.2 - M.2) (B.1 - M.1, B.2 - M.2)

/-- Definition of a line segment -/
def line_segment (a b : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (1 - t) • a + t • b}

/-- The theorem to be proved -/
theorem intersection_point_of_inscribed_triangles
  (triangle1 triangle2 : RightTriangleInParabola) :
  ∃ (E : ℝ × ℝ), E = (-1, 2) ∧
  E ∈ line_segment triangle1.A triangle1.B ∧
  E ∈ line_segment triangle2.A triangle2.B :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_of_inscribed_triangles_l96_9606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_x_less_than_2y_is_half_l96_9658

/-- The probability that a randomly chosen point (x,y) from a rectangle
    satisfies x < 2y, where the rectangle has vertices (0,0), (4,0), (4,2), and (0,2) -/
noncomputable def probability_x_less_than_2y : ℝ := 1/2

/-- The rectangle from which the point is chosen -/
def rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

/-- The region where x < 2y within the rectangle -/
def region_x_less_than_2y : Set (ℝ × ℝ) :=
  {p ∈ rectangle | p.1 < 2 * p.2}

/-- The volume (area) of a set in ℝ² -/
noncomputable def volume (s : Set (ℝ × ℝ)) : ℝ := sorry

theorem probability_x_less_than_2y_is_half :
  probability_x_less_than_2y = (volume region_x_less_than_2y) / (volume rectangle) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_x_less_than_2y_is_half_l96_9658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_gt_M_not_limit_to_infinity_l96_9617

/-- The number of solutions to n² + x² = y² in natural numbers x, y > n -/
noncomputable def a (n : ℕ) : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    p.1 > n ∧ p.2 > n ∧ n^2 + p.1^2 = p.2^2) (Finset.range 1000 ×ˢ Finset.range 1000)).card

/-- For any M, there exists n such that a(n) > M -/
theorem exists_n_gt_M (M : ℕ) : ∃ n : ℕ, a n > M := by
  sorry

/-- The limit of a(n) as n approaches infinity is not infinity -/
theorem not_limit_to_infinity : ¬ (∀ M : ℕ, ∃ N : ℕ, ∀ n ≥ N, a n > M) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_gt_M_not_limit_to_infinity_l96_9617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivatives_at_zero_l96_9683

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - Real.exp (-x^2))

theorem derivatives_at_zero :
  let f' := deriv f
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < x → x < δ → |f' x - 1| < ε) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, -δ < x → x < 0 → |f' x + 1| < ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivatives_at_zero_l96_9683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l96_9648

/-- The time (in seconds) it takes for a train to cross a platform -/
noncomputable def time_to_cross (train_length platform_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating the time it takes for the train to cross the platform -/
theorem train_crossing_time :
  let train_length := (480 : ℝ)
  let platform_length := (620 : ℝ)
  let train_speed_kmh := (55 : ℝ)
  let crossing_time := time_to_cross train_length platform_length train_speed_kmh
  ∃ ε > 0, |crossing_time - 71.98| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l96_9648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gray_squares_count_l96_9636

/-- Represents a square on the game board -/
inductive Square
| Gray
| NonGray

/-- Represents the game board as a 2D array of squares -/
def GameBoard := Array (Array Square)

/-- Represents a path on the game board as a list of coordinates -/
def GamePath := List (Nat × Nat)

/-- Rotates the game board clockwise by the specified number of 90-degree turns -/
def rotateBoard (board : GameBoard) (turns : Nat) : GameBoard :=
  sorry

/-- Counts the number of gray squares a path passes through on a given board -/
def countGraySquares (board : GameBoard) (path : GamePath) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem gray_squares_count 
  (board : GameBoard) 
  (path : GamePath) : 
  ∃ (initial_board : GameBoard),
    (countGraySquares initial_board path = 7) ∧
    (countGraySquares (rotateBoard initial_board 1) path = 8) ∧
    (countGraySquares (rotateBoard initial_board 2) path = 4) ∧
    (countGraySquares (rotateBoard initial_board 3) path = 7) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gray_squares_count_l96_9636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_l96_9688

/-- Represents a hyperbola with center (h,k), focus distance c, and vertex distance a -/
structure Hyperbola where
  h : ℝ
  k : ℝ
  c : ℝ
  a : ℝ

/-- The sum of h, k, a, and b for the given hyperbola is 10 -/
theorem hyperbola_sum (H : Hyperbola) (h_center : H.h = 0 ∧ H.k = 3) 
    (h_focus : H.c = 5) (h_vertex : H.a = 3) : 
  H.h + H.k + H.a + Real.sqrt (H.c^2 - H.a^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_l96_9688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_ratio_l96_9604

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

noncomputable def focusOfParabola (para : Parabola) : Point :=
  { x := para.p / 2, y := 0 }

noncomputable def lineThrough45Deg (para : Parabola) : Line :=
  { m := 1, b := -para.p / 2 }

noncomputable def intersectionPoints (para : Parabola) (l : Line) : (Point × Point) :=
  sorry

theorem parabola_line_intersection_ratio (para : Parabola) : 
  let F := focusOfParabola para
  let l := lineThrough45Deg para
  let (A, B) := intersectionPoints para l
  let AF := ((A.x - F.x)^2 + (A.y - F.y)^2).sqrt
  let BF := ((B.x - F.x)^2 + (B.y - F.y)^2).sqrt
  AF / BF = 3 + 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_ratio_l96_9604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_one_woman_l96_9696

theorem probability_at_least_one_woman (men women selected : ℕ) : 
  men = 8 → women = 4 → selected = 4 →
  (1 - (Nat.choose men selected : ℚ) / (Nat.choose (men + women) selected : ℚ)) = 85/99 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_one_woman_l96_9696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_and_monotonicity_l96_9686

noncomputable def f (a x : ℝ) : ℝ := (1/2) * x^2 - (a + 1) * x + a * Real.log x

theorem extremum_and_monotonicity (a : ℝ) :
  (∃ (h : ℝ), h > 0 ∧ ∀ x, x ∈ Set.Ioo (2 - h) (2 + h) → (deriv (f a)) x = 0) →
  (a = 2 ∧
   (∀ x, x > 0 → x < 1 → (deriv (f a)) x < 0) ∧
   (∀ x, x > 1 → (deriv (f a)) x > 0)) ∨
  (a < 0 ∧
   (∀ x, x > 0 → x < 1 → (deriv (f a)) x < 0) ∧
   (∀ x, x > 1 → (deriv (f a)) x > 0)) ∨
  (a > 0 ∧ a < 1 ∧
   (∀ x, x > a → x < 1 → (deriv (f a)) x < 0) ∧
   (∀ x, x > 0 → x < a → (deriv (f a)) x > 0) ∧
   (∀ x, x > 1 → (deriv (f a)) x > 0)) ∨
  (a = 1 ∧
   (∀ x, x > 0 → (deriv (f a)) x > 0)) ∨
  (a > 1 ∧
   (∀ x, x > 1 → x < a → (deriv (f a)) x < 0) ∧
   (∀ x, x > 0 → x < 1 → (deriv (f a)) x > 0) ∧
   (∀ x, x > a → (deriv (f a)) x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_and_monotonicity_l96_9686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_to_product_l96_9642

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (5 * x) = 2 * Real.sin (4 * x) * Real.cos x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_to_product_l96_9642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_l96_9654

/-- Calculates the final amount for an investment with annual compounding -/
noncomputable def annual_compound (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Calculates the final amount for an investment with monthly compounding -/
noncomputable def monthly_compound (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate / 12) ^ (years * 12)

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem investment_difference :
  let principal : ℝ := 100000
  let rate : ℝ := 0.05
  let years : ℕ := 3
  let monthly_result := monthly_compound principal rate years
  let annual_result := annual_compound principal rate years
  round_to_nearest (monthly_result - annual_result) = 405 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_l96_9654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_normal_l96_9661

/-- Parabola structure -/
structure Parabola where
  f : ℝ → ℝ
  eq : f = λ x => x^2 + 4*x + 2

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y = p.f x

/-- Tangent line to a parabola at a point -/
def tangent_line (p : Parabola) (point : PointOnParabola p) : ℝ → ℝ :=
  λ x => 6*x + 1

/-- Normal line to a parabola at a point -/
noncomputable def normal_line (p : Parabola) (point : PointOnParabola p) : ℝ → ℝ :=
  λ y => (-1/6)*y + 43/6

theorem parabola_tangent_normal (p : Parabola) 
  (point : PointOnParabola p) (h : point.x = 1 ∧ point.y = 7) :
  (tangent_line p point = λ x => 6*x + 1) ∧
  (∀ x y, normal_line p point y = x ↔ x + 6*y - 43 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_normal_l96_9661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_puzzle_l96_9600

theorem fraction_puzzle (n m : ℚ) 
  (h1 : n / (m - 1) = 1 / 3)
  (h2 : (n + 4) / m = 1 / 2) :
  n / m = 7 / 22 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_puzzle_l96_9600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_abc_l96_9676

/-- The diameter of the inscribed circle in a triangle with sides a, b, and c --/
noncomputable def inscribed_circle_diameter (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  2 * area / s

/-- The theorem stating the diameter of the inscribed circle in the given triangle --/
theorem inscribed_circle_diameter_abc :
  inscribed_circle_diameter 13 8 10 = Real.sqrt (400 / 15) := by
  sorry

-- We can't use #eval for noncomputable functions, so we'll use #check instead
#check inscribed_circle_diameter 13 8 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_abc_l96_9676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_worked_42_hours_l96_9628

/-- Represents Michael's weekly work and earnings structure -/
structure MichaelWork where
  regularRate : ℚ  -- Regular hourly rate
  overtimeRate : ℚ  -- Overtime hourly rate
  regularHours : ℚ  -- Maximum regular hours
  totalEarnings : ℚ  -- Total earnings for the week

/-- Calculates the total hours worked given Michael's work structure -/
def totalHoursWorked (w : MichaelWork) : ℚ :=
  w.regularHours + (w.totalEarnings - w.regularRate * w.regularHours) / w.overtimeRate

/-- Theorem stating that Michael worked 42 hours -/
theorem michael_worked_42_hours :
  let w : MichaelWork := {
    regularRate := 7,
    overtimeRate := 14,
    regularHours := 40,
    totalEarnings := 320
  }
  ⌊totalHoursWorked w⌋ = 42 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_worked_42_hours_l96_9628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_problem_l96_9610

theorem cosine_problem (f : ℝ → ℝ) (A α β : ℝ) :
  (∀ x, f x = A * Real.cos (x / 4 + π / 6)) →
  f (π / 3) = Real.sqrt 2 →
  α ∈ Set.Icc 0 (π / 2) →
  β ∈ Set.Icc 0 (π / 2) →
  f (4 * α + 4 * π / 3) = -30 / 17 →
  f (4 * β - 2 * π / 3) = 8 / 5 →
  A = 2 ∧ Real.cos (α + β) = -13 / 85 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_problem_l96_9610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ge_sphere_area_l96_9613

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop

/-- Chord of a parabola -/
structure Chord (para : Parabola) where
  start : ℝ × ℝ
  end_ : ℝ × ℝ
  passes_through_focus : Prop
  on_parabola : Prop

/-- Projection of a chord on the directrix -/
def projection (para : Parabola) (c : Chord para) : ℝ × ℝ := sorry

/-- Area of surface formed by rotating chord around directrix -/
def surface_area (para : Parabola) (c : Chord para) : ℝ := sorry

/-- Area of sphere with projection as diameter -/
def sphere_area (proj : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem surface_area_ge_sphere_area (para : Parabola) (c : Chord para) :
  surface_area para c ≥ sphere_area (projection para c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ge_sphere_area_l96_9613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DEF_l96_9651

noncomputable def D : ℝ × ℝ := (2, 5)

noncomputable def E : ℝ × ℝ := (-D.1, D.2)

noncomputable def F : ℝ × ℝ := (E.2, E.1)

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem area_of_triangle_DEF : triangle_area D E F = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DEF_l96_9651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_one_inequality_holds_l96_9689

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^x - 1/(2^(abs x))

-- Part 1: Prove f(1) = 3/2
theorem f_at_one : f 1 = 3/2 := by sorry

-- Part 2: Prove the inequality holds for m ≥ -5 and t ∈ [1,2]
theorem inequality_holds (m : ℝ) (t : ℝ) (h1 : m ≥ -5) (h2 : t ∈ Set.Icc 1 2) :
  2^t * f (2*t) + m * f t ≥ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_one_inequality_holds_l96_9689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l96_9670

/-- The function f(x) as defined in the problem -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * x) - Real.sqrt 3 * Real.cos (2 * ω * x)

/-- The theorem stating the relationship between ω and the distance between symmetry axes -/
theorem omega_value (ω : ℝ) :
  (∃ (d : ℝ), d > 0 ∧ ∀ (x : ℝ), f ω x = f ω (x + d) ∧ d = π / 3) →
  ω = 3 / 2 ∨ ω = -3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l96_9670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l96_9684

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then -x + 3*a else a^x

theorem decreasing_f_implies_a_range (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : ∀ x y : ℝ, x < y → f a x > f a y) : 
  a ∈ Set.Icc (1/3) 1 ∧ a < 1 := by
  sorry

#check decreasing_f_implies_a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l96_9684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotic_lines_l96_9601

/-- Definition of a hyperbola with equation x^2 - y^2 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

/-- Definition of an asymptotic line for the hyperbola -/
noncomputable def is_asymptotic_line (m : ℝ) : Prop :=
  ∀ ε > 0, ∃ x₀ > 0, ∀ x ≥ x₀, |Real.sqrt (x^2 - 1) - m*x| < ε

/-- Theorem: The asymptotic lines of the hyperbola x^2 - y^2 = 1 are y = x and y = -x -/
theorem hyperbola_asymptotic_lines :
  (is_asymptotic_line 1) ∧ (is_asymptotic_line (-1)) := by
  sorry

#check hyperbola_asymptotic_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotic_lines_l96_9601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_line_slope_angle_l96_9697

/-- A line in a 2D coordinate system -/
structure Line where
  equation : ℝ → Prop

/-- The slope angle of a line -/
noncomputable def slopeAngle (l : Line) : ℝ := sorry

/-- A line with equation x = 2 -/
def vertical_line : Line :=
  { equation := fun x ↦ x = 2 }

/-- Theorem: The slope angle of a line with equation x = 2 is 90° -/
theorem vertical_line_slope_angle :
  slopeAngle vertical_line = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_line_slope_angle_l96_9697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_through_origin_l96_9674

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp (2 - x)

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := -Real.exp (2 - x)

-- Theorem statement
theorem tangent_line_through_origin :
  ∃ (x₀ : ℝ), 
    (f x₀ = -f' x₀ * x₀) ∧ 
    (∀ x y : ℝ, y = -Real.exp 3 * x ↔ y = f' x₀ * (x - x₀) + f x₀) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_through_origin_l96_9674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_knowledgeable_person_l96_9655

/-- Represents a person in the room -/
structure Person where
  id : Nat
  isKnowledgeable : Bool

/-- Represents the result of asking a person about another person's information -/
inductive QueryResult
  | Knows
  | DoesntKnow

/-- The main theorem stating that the knowledgeable person can be identified in 29 queries -/
theorem identify_knowledgeable_person 
  (people : Finset Person) 
  (h_count : people.card = 30) 
  (h_unique : ∃! p, p ∈ people ∧ p.isKnowledgeable) 
  (query : Person → Person → QueryResult) 
  (h_query : ∀ (p1 p2 : Person), p1 ∈ people → p2 ∈ people → 
    query p1 p2 = QueryResult.Knows ↔ (p1.isKnowledgeable ∨ p1 = p2)) :
  ∃ (queries : Finset (Person × Person)), 
    queries.card = 29 ∧ 
    ∃! (p : Person), p ∈ people ∧ 
      ∀ (q : Person × Person), q ∈ queries → 
        (query q.fst q.snd = QueryResult.DoesntKnow → q.fst ≠ p) ∧
        (query q.fst q.snd = QueryResult.Knows → q.snd ≠ p) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_knowledgeable_person_l96_9655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_theorem_l96_9612

/-- An isosceles trapezoid inscribed in a circle -/
structure IsoscelesTrapezoid where
  /-- Ratio of leg to shorter base -/
  k : ℝ
  /-- Condition that k > 1 -/
  k_gt_one : k > 1

/-- Angles of the isosceles trapezoid -/
noncomputable def trapezoid_angles (t : IsoscelesTrapezoid) : ℝ × ℝ :=
  (Real.arccos (1 - 1 / t.k), Real.pi - Real.arccos (1 - 1 / t.k))

theorem isosceles_trapezoid_theorem (t : IsoscelesTrapezoid) :
  let (α, β) := trapezoid_angles t
  α = Real.arccos (1 - 1 / t.k) ∧
  β = Real.pi - Real.arccos (1 - 1 / t.k) ∧
  t.k > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_theorem_l96_9612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_4_is_5_l96_9668

-- Define the polynomials
def p (x : ℝ) : ℝ := x^5 - 2*x^4 + 3*x^3 - 4*x + 2
def q (x : ℝ) : ℝ := 3*x^2 - x + 5

-- Define the product of the polynomials
def product (x : ℝ) : ℝ := p x * q x

-- Theorem statement
theorem coefficient_of_x_4_is_5 :
  ∃ (a b c d e : ℝ), ∀ x, product x = a*x^6 + 5*x^4 + b*x^3 + c*x^2 + d*x + e :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_4_is_5_l96_9668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_division_exists_l96_9639

-- Define a polygon type
structure Polygon where
  vertices : Set (ℝ × ℝ)
  is_polygon : Prop

-- Define a rotation of 360°/7
noncomputable def rotation_360_div_7 (center : ℝ × ℝ) (p : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define invariance under rotation
def invariant_under_rotation (poly : Polygon) (center : ℝ × ℝ) : Prop :=
  ∀ p, p ∈ poly.vertices → rotation_360_div_7 center p ∈ poly.vertices

-- Define minimum side length
noncomputable def min_side_length (poly : Polygon) : ℝ := sorry

-- Theorem statement
theorem plane_division_exists :
  ∃ (division : Set Polygon),
    (∀ p : ℝ × ℝ, ∃! poly, poly ∈ division ∧ p ∈ poly.vertices) ∧
    (∀ poly ∈ division, ∃ center : ℝ × ℝ, invariant_under_rotation poly center) ∧
    (∀ poly ∈ division, min_side_length poly > 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_division_exists_l96_9639
