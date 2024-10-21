import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_triangles_theorem_l1012_101271

/-- A triangle represented by its three sides and three angles. -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

/-- Two triangles are congruent if they have the same shape and size. -/
def CongruentTriangles (t1 t2 : Triangle) : Prop :=
  t1.side1 = t2.side1 ∧ t1.side2 = t2.side2 ∧ t1.side3 = t2.side3

/-- A triangle is right-angled if one of its angles is 90 degrees. -/
def RightAngledTriangle (t : Triangle) : Prop :=
  t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

/-- A triangle is isosceles if at least two of its sides are equal. -/
def IsoscelesTriangle (t : Triangle) : Prop :=
  t.side1 = t.side2 ∨ t.side2 = t.side3 ∨ t.side1 = t.side3

/-- An angle is obtuse if it is greater than 90 degrees. -/
def ObtuseAngle (angle : ℝ) : Prop := angle > 90

theorem congruent_triangles_theorem :
  ∀ (t1 t2 : Triangle),
    (RightAngledTriangle t1 ∧ RightAngledTriangle t2 ∧ 
     ∃ (s1 s2 : ℝ), t1.side1 = s1 ∧ t1.side2 = s2 ∧ t2.side1 = s1 ∧ t2.side2 = s2) →
    ¬ CongruentTriangles t1 t2 ∧
    
    (RightAngledTriangle t1 ∧ RightAngledTriangle t2 ∧
     ∃ (s : ℝ) (a : ℝ), t1.side1 = s ∧ t2.side1 = s ∧ t1.angle1 = a ∧ t2.angle1 = a) →
    ¬ CongruentTriangles t1 t2 ∧
    
    (IsoscelesTriangle t1 ∧ IsoscelesTriangle t2 ∧
     t1.side1 = 1 ∧ t1.side2 = 1 ∧ t1.side3 = 1 ∧
     t2.side1 = 1 ∧ t2.side2 = 1 ∧ t2.side3 = 1) →
    CongruentTriangles t1 t2 ∧
    
    (IsoscelesTriangle t1 ∧ IsoscelesTriangle t2 ∧
     ∃ (a : ℝ), ObtuseAngle a ∧ t1.angle1 = a ∧ t2.angle1 = a) →
    ¬ CongruentTriangles t1 t2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_triangles_theorem_l1012_101271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l1012_101298

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (Real.sqrt (a - 1) > Real.sqrt (b - 1)) → (a > b ∧ b > 0) ∧
  ¬(∀ a b : ℝ, (a > b ∧ b > 0) → Real.sqrt (a - 1) > Real.sqrt (b - 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l1012_101298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l1012_101293

/-- Calculates the average speed of a train given two trips -/
noncomputable def average_speed (distance1 : ℝ) (time1 : ℝ) (distance2 : ℝ) (time2 : ℝ) : ℝ :=
  (distance1 + distance2) / (time1 + time2)

/-- The average speed of the train is approximately 71.82 km/h -/
theorem train_average_speed :
  let speed := average_speed 125 2.5 270 3
  ∃ ε > 0, |speed - 71.82| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l1012_101293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_metropolis_taxi_distance_l1012_101243

/-- Represents the fare structure for taxis in Metropolis -/
structure FareStructure where
  initial_fare : ℚ
  initial_distance : ℚ
  additional_fare : ℚ
  additional_distance : ℚ

/-- Calculates the total distance that can be traveled given a fare structure and total amount -/
def calculate_distance (fs : FareStructure) (total_amount : ℚ) (tip : ℚ) : ℚ :=
  let fare_amount := total_amount - tip
  let additional_miles := (fare_amount - fs.initial_fare) / (fs.additional_fare / fs.additional_distance)
  fs.initial_distance + additional_miles

/-- Theorem stating that given the specific fare structure and total amount including tip,
    the maximum distance that can be traveled is 4.35 miles -/
theorem metropolis_taxi_distance :
  let fs : FareStructure := {
    initial_fare := 3,
    initial_distance := 3/4,
    additional_fare := 1/4,
    additional_distance := 1/10
  }
  let total_amount := 15
  let tip := 3
  calculate_distance fs total_amount tip = 87/20 := by
  sorry

#eval calculate_distance
  { initial_fare := 3,
    initial_distance := 3/4,
    additional_fare := 1/4,
    additional_distance := 1/10 }
  15 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_metropolis_taxi_distance_l1012_101243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_2_2019_closest_to_4_l1012_101204

noncomputable def harmonicMean (a b : ℝ) : ℝ := 2 * a * b / (a + b)

theorem harmonic_mean_2_2019_closest_to_4 :
  let hm := harmonicMean 2 2019
  ∀ n : ℤ, n ≠ 4 → |hm - 4| < |hm - (n : ℝ)| := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_2_2019_closest_to_4_l1012_101204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_eq_six_l1012_101280

/-- The number of integers x such that (x, -2x) lies inside or on the circle 
    with radius 10 centered at (7, 3) -/
def count_integers : ℕ := 6

/-- The circle with radius 10 centered at (7, 3) -/
def in_circle (x y : ℝ) : Prop :=
  (x - 7)^2 + (y - 3)^2 ≤ 100

theorem count_integers_eq_six : 
  count_integers = 6 ∧ 
  (∀ x : ℤ, (in_circle (x : ℝ) (-2 * x) ↔ -2 ≤ x ∧ x ≤ 3)) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_eq_six_l1012_101280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequality_l1012_101289

theorem count_integers_satisfying_inequality :
  ∃ (S : Finset ℤ), (∀ n ∈ S, (n + 1) * (n - 5) < 0) ∧ Finset.card S = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequality_l1012_101289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divides_n_squared_plus_n_plus_one_l1012_101263

theorem divides_n_squared_plus_n_plus_one (n : ℕ) : 
  n < 589 ∧ 589 ∣ (n^2 + n + 1) ↔ n ∈ ({49, 216, 315, 482} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divides_n_squared_plus_n_plus_one_l1012_101263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_faces_polyhedron_with_pentagons_l1012_101200

/-- Definition of a convex polyhedron -/
structure ConvexPolyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ
  pentagonalFaces : ℕ
  convex : Bool
  euler : faces + vertices = edges + 2
  minEdgesPerVertex : 2 * edges ≥ 3 * vertices
  pentagonalFacesCondition : pentagonalFaces ≥ 3

/-- A convex polyhedron with at least three pentagonal faces has at least 7 faces. -/
theorem min_faces_polyhedron_with_pentagons (P : ConvexPolyhedron) : P.faces ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_faces_polyhedron_with_pentagons_l1012_101200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_volume_bound_and_increasing_main_sales_theorem_l1012_101224

/-- Sequence representing total sales volume -/
def a : ℕ → ℚ
  | 0 => 1  -- Adding case for 0 to cover all natural numbers
  | 1 => 1
  | 2 => 3/2
  | 3 => 15/8
  | n + 4 => 2 * a (n + 3) - (1/2) * (a (n + 3))^2

/-- The sequence a_n is always less than 2 and strictly increasing -/
theorem sales_volume_bound_and_increasing :
  ∀ n : ℕ, n ≥ 1 → a n < a (n + 1) ∧ a (n + 1) < 2 := by
  sorry

/-- The main theorem combining both parts of the proof -/
theorem main_sales_theorem :
  (a 1 = 1 ∧ a 2 = 3/2 ∧ a 3 = 15/8) ∧
  (∀ n : ℕ, n ≥ 1 → a n < a (n + 1) ∧ a (n + 1) < 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_volume_bound_and_increasing_main_sales_theorem_l1012_101224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_l1012_101260

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 * Real.pi * Real.arcsin x - (Real.arccos (-x))^2

-- State the theorem
theorem max_min_difference :
  ∃ (M m : ℝ), (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ M ∧ m ≤ f x) ∧ M - m = 3 * Real.pi^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_l1012_101260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_purchase_cost_l1012_101277

/-- Represents the cost of apples in dollars -/
noncomputable def appleCost (pounds : ℝ) : ℝ :=
  let baseRate := 6 / 6  -- $6 per 6 pounds
  let baseCost := baseRate * pounds
  if pounds > 20 then
    baseCost * 0.9  -- 10% discount
  else
    baseCost

theorem apple_purchase_cost :
  appleCost 30 = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_purchase_cost_l1012_101277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_ratios_l1012_101233

/-- An isosceles trapezoid inscribed in a circle -/
structure IsoscelesTrapezoid where
  a : ℝ  -- length of base BC
  b : ℝ  -- length of base AD
  c : ℝ  -- length of leg AB and CD
  r : ℝ  -- radius of inscribed circle
  h : ℝ  -- height of trapezoid
  α : ℝ  -- angle at the base

/-- Properties of the isosceles trapezoid -/
def IsoscelesTrapezoidProperties (t : IsoscelesTrapezoid) : Prop :=
  t.h = 2 * t.r ∧
  2 * t.c = t.a + t.b ∧
  t.a + t.b = 8 * t.r ∧
  t.α = Real.pi / 6 ∧
  t.c = 4 * t.r

/-- Area of the trapezoid -/
def AreaTrapezoid (t : IsoscelesTrapezoid) : ℝ := (t.a + t.b) * t.r

/-- Area of the inscribed circle -/
noncomputable def AreaInscribedCircle (t : IsoscelesTrapezoid) : ℝ := Real.pi * t.r^2

/-- Area of the circumscribed circle -/
noncomputable def AreaCircumscribedCircle (t : IsoscelesTrapezoid) : ℝ := 5 * Real.pi * t.c^2 / 4

/-- Main theorem -/
theorem isosceles_trapezoid_area_ratios 
  (t : IsoscelesTrapezoid) 
  (h : IsoscelesTrapezoidProperties t) : 
  AreaTrapezoid t / AreaInscribedCircle t = 8 / Real.pi ∧
  AreaTrapezoid t / AreaCircumscribedCircle t = 2 / (5 * Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_ratios_l1012_101233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_range_l1012_101254

theorem quadratic_equation_roots_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ 
   x₁ ≠ x₂ ∧ 
   x₁^2 + (m+3)*x₁ - m = 0 ∧ 
   x₂^2 + (m+3)*x₂ - m = 0) ↔ 
  m ∈ Set.Iic (-9) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_range_l1012_101254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_BPD_is_one_l1012_101210

-- Define the points
variable (A B C D E P : ℝ × ℝ)

-- Define the angles
noncomputable def angle_BPC : ℝ := Real.arccos ((3 : ℝ) / 5)
noncomputable def angle_CPD : ℝ := Real.arccos ((4 : ℝ) / 5)
noncomputable def angle_BPD : ℝ := angle_BPC + angle_CPD

-- State the theorem
theorem sin_angle_BPD_is_one 
  (h1 : B.1 - A.1 = C.1 - B.1) 
  (h2 : C.1 - B.1 = D.1 - C.1) 
  (h3 : D.1 - C.1 = E.1 - D.1) 
  (h4 : B.2 = C.2) (h5 : C.2 = D.2) (h6 : D.2 = E.2) 
  (h7 : Real.cos angle_BPC = (3 : ℝ) / 5) 
  (h8 : Real.cos angle_CPD = (4 : ℝ) / 5) : 
  Real.sin angle_BPD = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_BPD_is_one_l1012_101210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_non_prime_power_numerators_l1012_101242

def harmonic_sum (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => 1 / (i + 1 : ℚ))

def numerator_of_irreducible (q : ℚ) : ℕ :=
  q.num.natAbs

def is_prime_power (n : ℕ) : Prop :=
  ∃ p k, Nat.Prime p ∧ k > 0 ∧ n = p ^ k

theorem infinitely_many_non_prime_power_numerators :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    ∀ n ∈ S, ¬ is_prime_power (numerator_of_irreducible (harmonic_sum n)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_non_prime_power_numerators_l1012_101242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_on_interval_l1012_101261

-- Define the function f(x) = x / ln(x)
noncomputable def f (x : ℝ) : ℝ := x / Real.log x

-- State the theorem
theorem f_strictly_decreasing_on_interval :
  StrictMonoOn f (Set.Ioo 1 (Real.exp 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_on_interval_l1012_101261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_120_degrees_l1012_101227

namespace Triangle

structure ABC where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_triangle : a + b > c ∧ b + c > a ∧ c + a > b

variable (t : ABC)

noncomputable def cosB : ℝ := (t.a^2 + t.c^2 - t.b^2) / (2 * t.a * t.c)

theorem angle_B_is_120_degrees (h : t.a^2 = t.b^2 - t.c^2 - t.a * t.c) :
  cosB t = -1/2 := by
  -- Proof steps would go here
  sorry

end Triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_120_degrees_l1012_101227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumradius_l1012_101231

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the incenter
noncomputable def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the circumradius
noncomputable def circumradius (t : Triangle) : ℝ := sorry

-- Define the angle function
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_circumradius (t : Triangle) :
  -- Conditions
  (distance t.A (incenter t) + distance t.B (incenter t) + distance t.C (incenter t) = 
   distance t.A t.B + distance t.B t.C + distance t.C t.A) →  -- Incenter property
  (distance t.A (incenter t) = distance t.B (incenter t)) →   -- Equidistant from A and B
  (distance t.B (incenter t) = 3) →                           -- Distance from B to incenter is 3
  (distance t.C (incenter t) = 4) →                           -- Distance from C to incenter is 4
  (Real.cos (angle t.B t.A t.C) = 1/2) →                      -- Angle A is 60°
  -- Conclusion
  circumradius t = Real.sqrt (37/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumradius_l1012_101231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_I_is_positive_l1012_101238

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem I_is_positive (n : ℕ) : (n + 1)^2 + n - (floor (Real.sqrt ((n + 1)^2 + n + 1)))^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_I_is_positive_l1012_101238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1012_101240

noncomputable def angle_between (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem angle_between_vectors :
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (0, -2)
  angle_between a b = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1012_101240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_and_radicals_l1012_101215

/-- The internal angle of a regular octagon in degrees -/
noncomputable def internal_angle_regular_octagon : ℝ := 135

/-- The square root of 27 -/
noncomputable def sqrt_27 : ℝ := Real.sqrt 27

/-- The square root of 1/3 -/
noncomputable def sqrt_one_third : ℝ := Real.sqrt (1/3)

/-- Two radical expressions are similar if they differ only by a rational factor -/
def are_similar_radicals (a b : ℝ) : Prop :=
  ∃ (q : ℚ), a = q * b ∨ b = q * a

theorem regular_octagon_and_radicals :
  (internal_angle_regular_octagon = 135) ∧
  (are_similar_radicals sqrt_27 sqrt_one_third) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_and_radicals_l1012_101215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_divisibility_pairs_l1012_101249

theorem three_digit_divisibility_pairs :
  (Finset.filter (fun pairs : ℕ × ℕ =>
    let (a, b) := pairs
    100 ≤ b ∧ b ≤ 999 ∧
    (a + 1) ∣ (b - 1) ∧
    b ∣ (a^2 + a + 2)) (Finset.product (Finset.range 1000) (Finset.range 1000))).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_divisibility_pairs_l1012_101249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_and_intersection_l1012_101295

/-- Curve C in parametric form -/
noncomputable def curve_C (t : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos (2 * t), 2 * Real.sin t)

/-- Line l in polar form -/
def line_l (ρ θ m : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 3) + m = 0

/-- Cartesian equation of line l -/
def line_l_cartesian (x y m : ℝ) : Prop := Real.sqrt 3 * x + y + 2 * m = 0

/-- Theorem stating the Cartesian equation of line l and its intersection with curve C -/
theorem line_equation_and_intersection :
  ∀ m : ℝ,
  (∀ x y : ℝ, line_l_cartesian x y m ↔ ∃ ρ θ, line_l ρ θ m ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧
  (∃ t : ℝ, ∃ x y : ℝ, curve_C t = (x, y) ∧ line_l_cartesian x y m) ↔ m ≥ -19/12 ∧ m ≤ 5/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_and_intersection_l1012_101295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolic_arch_height_l1012_101239

/-- Represents a parabolic arch -/
structure ParabolicArch where
  a : ℝ
  k : ℝ

/-- The height of the arch at a given x-coordinate -/
def ParabolicArch.height (arch : ParabolicArch) (x : ℝ) : ℝ :=
  arch.a * x^2 + arch.k

theorem parabolic_arch_height :
  ∃ (arch : ParabolicArch),
    (arch.height 0 = 25) ∧
    (arch.height 25 = 0) ∧
    (arch.height 10 = 21) := by
  -- Construct the arch
  let arch : ParabolicArch := ⟨-1/25, 25⟩
  
  -- Prove it satisfies the conditions
  use arch
  constructor
  · -- Height at center (x = 0)
    simp [ParabolicArch.height]
  constructor
  · -- Height at edge (x = 25)
    simp [ParabolicArch.height]
    norm_num
  · -- Height at 10 inches from center
    simp [ParabolicArch.height]
    norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolic_arch_height_l1012_101239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_product_l1012_101266

theorem polynomial_root_product (p q r : ℝ) : 
  (∃ Q : ℝ → ℝ, Q = λ x ↦ x^3 + p*x^2 + q*x + r) ∧
  (∀ x : ℝ, x^3 + p*x^2 + q*x + r = 0 ↔ 
    (x = Real.cos (π/9) ∨ x = Real.cos (2*π/9) ∨ x = Real.cos (4*π/9))) →
  p * q * r = 1 / 576 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_product_l1012_101266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l1012_101222

theorem complex_fraction_simplification (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^(1/6:ℝ) - y^(1/6:ℝ)) / (x^(1/2:ℝ) + x^(1/3:ℝ) * y^(1/6:ℝ)) *
  ((x^(1/3:ℝ) + y^(1/3:ℝ))^2 - 4 * (x*y)^(1/3:ℝ)) / (x^(5/6:ℝ) * y^(1/3:ℝ) - x^(1/2:ℝ) * y^(2/3:ℝ)) +
  2 * x^(-2/3:ℝ) * y^(-1/6:ℝ) =
  (x^(1/3:ℝ) + y^(1/3:ℝ)) / (x^(5/6:ℝ) * y^(1/3:ℝ)) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l1012_101222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1012_101230

open Real

/-- A function satisfying the given differential equation and initial condition -/
noncomputable def f : ℝ → ℝ := sorry

/-- The domain of f is (0, +∞) -/
axiom f_domain (x : ℝ) : x > 0 → DifferentiableAt ℝ f x

/-- The differential equation satisfied by f -/
axiom f_diff_eq (x : ℝ) (h : x > 0) : 
  x * deriv f x - f x = (x - 1) * Real.exp x

/-- The initial condition f(1) = 0 -/
axiom f_init : f 1 = 0

theorem f_properties : 
  (3 * f 2 < 2 * f 3) ∧ 
  (∀ x : ℝ, x > 0 → ∃ y : ℝ, y > 0 ∧ f y > f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1012_101230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_vertical_line_l1012_101217

/-- The inclination angle of a line is the angle between the positive x-axis and the line --/
noncomputable def inclination_angle (l : ℝ → Prop) : ℝ :=
  sorry

/-- The inclination angle of a vertical line is π/2 --/
theorem inclination_angle_vertical_line :
  let l : ℝ → Prop := λ x => x = Real.tan (-π/6)
  ∀ θ, (∀ x y, l x → l y → x = y) →
    (θ = inclination_angle l) → θ = π/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_vertical_line_l1012_101217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_is_4_sqrt_6_plus_21_l1012_101225

-- Define the sales volume function
noncomputable def F (x : ℝ) : ℝ := -1/3 * x^3 + 6 * x + 24

-- Define the profit function
noncomputable def profit (x : ℝ) : ℝ := F x - 3

-- Theorem statement
theorem max_profit_is_4_sqrt_6_plus_21 :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧
  ∀ y : ℝ, 0 ≤ y ∧ y ≤ 3 → profit x ≥ profit y ∧
  profit x = 4 * Real.sqrt 6 + 21 := by
  sorry

#check max_profit_is_4_sqrt_6_plus_21

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_is_4_sqrt_6_plus_21_l1012_101225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_arithmetic_sum_l1012_101270

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

noncomputable def is_arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

noncomputable def arithmetic_sum (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (b 1 + b n) / 2

theorem geometric_arithmetic_sum 
  (a b : ℕ → ℝ) 
  (h_geo : is_geometric_sequence a)
  (h_arith : is_arithmetic_sequence b)
  (h_relation : a 4 * a 6 = 2 * a 5)
  (h_b5 : b 5 = 2 * a 5) :
  arithmetic_sum b 9 = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_arithmetic_sum_l1012_101270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_m_values_l1012_101292

/-- Triangle with vertices (0,0), (2,2), and (4m,0) -/
structure Triangle (m : ℝ) where
  v1 : ℝ × ℝ := (0, 0)
  v2 : ℝ × ℝ := (2, 2)
  v3 : ℝ × ℝ := (4*m, 0)

/-- Line y = 2mx -/
def dividingLine (m : ℝ) (x : ℝ) : ℝ := 2*m*x

/-- The line divides the triangle into two equal area triangles -/
def dividesEquallyArea (m : ℝ) : Prop := sorry

/-- The sum of all possible values of m is -1/2 -/
theorem sum_of_m_values (m : ℝ) :
  dividesEquallyArea m →
  (∃ m₁ m₂ : ℝ, m = m₁ ∨ m = m₂) ∧ (Finset.sum {m₁, m₂} id = -1/2) :=
sorry

#check sum_of_m_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_m_values_l1012_101292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_disjoint_quadrilaterals_l1012_101288

-- Define a type for grid points
def GridPoint := ℤ × ℤ

-- Define a type for quadrilaterals
def Quadrilateral := Finset GridPoint

-- Define the property of a valid selection of quadrilaterals
def ValidSelection (S : Set Quadrilateral) :=
  ∀ (coloring : GridPoint → Nat),
    ∃ (q : Quadrilateral), q ∈ S ∧ 
      (∀ (p₁ p₂ : GridPoint), p₁ ∈ q.val → p₂ ∈ q.val → coloring p₁ = coloring p₂)

-- Define the property of quadrilaterals not sharing vertices
def NoSharedVertices (q₁ q₂ : Quadrilateral) :=
  ∀ (p : GridPoint), p ∈ q₁.val → p ∉ q₂.val

-- The main theorem
theorem infinite_disjoint_quadrilaterals 
  (S : Set Quadrilateral) (h : ValidSelection S) :
  ∃ (f : Nat → Quadrilateral), 
    (∀ n, f n ∈ S) ∧ 
    (∀ i j, i ≠ j → NoSharedVertices (f i) (f j)) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_disjoint_quadrilaterals_l1012_101288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1012_101274

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 2) + 1 / (x - 3)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ 2 ∧ x ≠ 3} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1012_101274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_row_greater_than_column_l1012_101286

/-- Represents a cell in the rectangle grid -/
structure Cell (M N : ℕ) where
  row : Fin M
  col : Fin N

/-- Represents the configuration of stars in the rectangle -/
def StarConfiguration (M N : ℕ) := Cell M N → Bool

theorem star_row_greater_than_column 
  (M N : ℕ) 
  (h_dim : N > M) 
  (stars : StarConfiguration M N)
  (h_row : ∀ r, ∃ c, stars ⟨r, c⟩ = true)
  (h_col : ∀ c, ∃ r, stars ⟨r, c⟩ = true) :
  ∃ (i : Fin M) (j : Fin N), 
    stars ⟨i, j⟩ = true ∧ 
    (Finset.filter (fun c => stars ⟨i, c⟩ = true) (Finset.univ : Finset (Fin N))).card > 
    (Finset.filter (fun r => stars ⟨r, j⟩ = true) (Finset.univ : Finset (Fin M))).card :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_row_greater_than_column_l1012_101286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l1012_101269

theorem distance_range (θ : Real) : 
  let point := (Real.sin θ, Real.cos θ)
  let line_dist (x y : Real) := |x * Real.cos θ + y * Real.sin θ + 1| / Real.sqrt (Real.cos θ ^ 2 + Real.sin θ ^ 2)
  let d := line_dist (Real.sin θ) (Real.cos θ)
  0 ≤ d ∧ d ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l1012_101269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_from_skew_lines_l1012_101219

/-- A line in 3D space --/
structure Line3D where
  direction : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Three lines are mutually skew if no two are parallel and no two intersect --/
def mutually_skew (f g h : Line3D) : Prop :=
  (f.direction ≠ g.direction ∧ f.direction ≠ h.direction ∧ g.direction ≠ h.direction) ∧
  (¬∃ (t s : ℝ), f.point + t • f.direction = g.point + s • g.direction) ∧
  (¬∃ (t s : ℝ), g.point + t • g.direction = h.point + s • h.direction) ∧
  (¬∃ (t s : ℝ), h.point + t • h.direction = f.point + s • f.direction)

/-- A parallelepiped in 3D space --/
structure Parallelepiped where
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- A line lies on an edge of a parallelepiped --/
def line_on_edge (l : Line3D) (p : Parallelepiped) : Prop :=
  ∃ (i j : Fin 8), i ≠ j ∧ 
    ∃ (t : ℝ), l.point + t • l.direction = p.vertices i ∧
               l.point + (t + 1) • l.direction = p.vertices j

/-- Main theorem --/
theorem parallelepiped_from_skew_lines (f g h : Line3D) 
  (H : mutually_skew f g h) :
  ∃! (p : Parallelepiped), line_on_edge f p ∧ line_on_edge g p ∧ line_on_edge h p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_from_skew_lines_l1012_101219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1012_101234

/-- The function f(x) -/
noncomputable def f (x : ℝ) : ℝ := 
  4 * (Real.sin (Real.pi / 4 + x))^2 - 2 * Real.sqrt 3 * Real.cos (2 * x) - 1

/-- Condition p -/
def p (x : ℝ) : Prop := x < Real.pi / 4 ∨ x > Real.pi / 2

/-- Condition q -/
def q (x m : ℝ) : Prop := -3 < f x - m ∧ f x - m < 3

/-- The main theorem -/
theorem range_of_m :
  (∀ x : ℝ, ¬(p x) → q x m) →
  ∃ a b : ℝ, a = 2 ∧ b = 6 ∧ ∀ m : ℝ, a < m ∧ m < b := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1012_101234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1012_101223

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / (x - 2)

-- State the theorem
theorem f_range :
  (∀ y ∈ Set.range f, y ≠ 0) ∧
  (∀ y : ℝ, y ≠ 0 → ∃ x : ℝ, f x = y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1012_101223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_F_l1012_101211

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := 3^(x - b)

-- Define the inverse function of f
noncomputable def f_inv (b : ℝ) (x : ℝ) : ℝ := 2 + (Real.log x) / (Real.log 3)

-- Define the function F
noncomputable def F (b : ℝ) (x : ℝ) : ℝ := (f_inv b x)^2 - f_inv b (x^2)

-- State the theorem
theorem range_of_F :
  ∃ (b : ℝ), 
    (∀ x ∈ Set.Icc 2 4, f b x = 3^(x - b)) ∧ 
    (f b 2 = 1) ∧
    (Set.range (F b) = Set.Icc 2 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_F_l1012_101211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_relationship_l1012_101265

-- Define a, b, and c as noncomputable real numbers
noncomputable def a : ℝ := (1/3) ^ (Real.log 3 / Real.log ((1/3) ^ (Real.log 3 / Real.log ((1/3) ^ (Real.log 3 / Real.log 1)))))
noncomputable def b : ℝ := (1/3) ^ (Real.log 4 / Real.log a)
noncomputable def c : ℝ := 3 ^ Real.log 3

-- State the theorem
theorem abc_relationship : c = a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_relationship_l1012_101265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_same_properties_l1012_101259

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 2^x - 2^(-x)
def g (x : ℝ) : ℝ := x^3

-- State the theorem
theorem f_g_same_properties :
  (∀ x, (∃ y, f x = y) ↔ (∃ y, g x = y)) ∧  -- Same domain (ℝ)
  (∀ x y, x < y → f x < f y ↔ g x < g y) ∧  -- Same monotonicity
  (∀ x, f (-x) = -f x ↔ g (-x) = -g x) :=  -- Same parity
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_same_properties_l1012_101259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_circumscribing_cone_l1012_101296

/-- The surface area of a sphere circumscribed around a cone -/
theorem sphere_surface_area_circumscribing_cone 
  (R h : ℝ) (hR : R > 0) (hh : h > 0) : 
  (4 : ℝ) * π * ((R^2 + h^2) / (2*h))^2 = π * (R^2 + h^2)^2 / h^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_circumscribing_cone_l1012_101296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_one_range_l1012_101267

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then |3*x - 4|
  else 2 / (x - 1)

theorem f_geq_one_range :
  {x : ℝ | f x ≥ 1} = Set.Iic 1 ∪ Set.Icc (5/3) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_one_range_l1012_101267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_pass_through_common_point_l1012_101241

-- Define the circle and points
variable (circle : Set (ℝ × ℝ))
variable (A B C : ℝ × ℝ)

-- Define the arcs
def arc_AB : Set (ℝ × ℝ) := sorry
def arc_BC : Set (ℝ × ℝ) := sorry
def arc_CA : Set (ℝ × ℝ) := sorry

-- Define the midpoints
noncomputable def D : ℝ × ℝ := sorry
noncomputable def E : ℝ × ℝ := sorry
noncomputable def F : ℝ × ℝ := sorry

-- Define the constructed circles
def circle_D : Set (ℝ × ℝ) := sorry
def circle_E : Set (ℝ × ℝ) := sorry
def circle_F : Set (ℝ × ℝ) := sorry

-- Define the incenter of triangle ABC
noncomputable def incenter : ℝ × ℝ := sorry

-- State the theorem
theorem circles_pass_through_common_point :
  A ∈ circle → B ∈ circle → C ∈ circle →
  D ∈ arc_BC → E ∈ arc_CA → F ∈ arc_AB →
  B ∈ circle_D → C ∈ circle_D →
  C ∈ circle_E → A ∈ circle_E →
  A ∈ circle_F → B ∈ circle_F →
  incenter ∈ circle_D ∧ incenter ∈ circle_E ∧ incenter ∈ circle_F := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_pass_through_common_point_l1012_101241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1012_101275

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 1)

noncomputable def vec_b (x m : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.sin (2 * x) + m)

noncomputable def f (x m : ℝ) : ℝ := (vec_a x).1 * (vec_b x m).1 + (vec_a x).2 * (vec_b x m).2

theorem f_properties (m : ℝ) :
  (∀ x, f (x + Real.pi) m = f x m) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 6), Monotone (fun y ↦ f y m)) ∧
  (∀ x ∈ Set.Icc (2 * Real.pi / 3) Real.pi, Monotone (fun y ↦ f y m)) ∧
  ((∀ x ∈ Set.Icc 0 (Real.pi / 6), -4 < f x m ∧ f x m < 4) →
   -6 < m ∧ m < 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1012_101275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_in_still_water_l1012_101203

/-- Proves that the speed of a boat in still water is 24 km/hr given the conditions -/
theorem boat_speed_in_still_water
  (stream_speed : ℝ) (travel_time : ℝ) (distance : ℝ) (boat_speed : ℝ)
  (h1 : stream_speed = 4)
  (h2 : travel_time = 2)
  (h3 : distance = 56)
  (h4 : distance = (boat_speed + stream_speed) * travel_time) :
  boat_speed = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_in_still_water_l1012_101203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_not_representable_l1012_101237

theorem consecutive_integers_not_representable :
  ∃ (n : ℕ), ∀ (k : ℕ) (x y : ℤ), 
    n ≤ k ∧ k < n + 8 → (7 * x^2 + 9 * x * y - 5 * y^2).natAbs ≠ k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_not_representable_l1012_101237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_is_centroid_l1012_101262

/-- The centroid of a triangle given its three vertices -/
noncomputable def centroid (p1 p2 p3 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1 + p3.1) / 3, (p1.2 + p2.2 + p3.2) / 3)

/-- Alice's position -/
def alice : ℝ × ℝ := (1, 5)

/-- Bob's position -/
def bob : ℝ × ℝ := (-3, -3)

/-- Carol's position -/
def carol : ℝ × ℝ := (3, 1)

/-- The meeting point is the centroid of the triangle formed by Alice, Bob, and Carol's positions -/
theorem meeting_point_is_centroid :
  centroid alice bob carol = (1/3, 1) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_is_centroid_l1012_101262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_number_is_981_l1012_101245

/-- The sequence of integers that are either powers of 3 or sums of distinct powers of 3 -/
def sequenceThree : ℕ → ℕ := sorry

/-- The property that a natural number is either a power of 3 or a sum of distinct powers of 3 -/
def is_valid_number (n : ℕ) : Prop := sorry

theorem hundredth_number_is_981 :
  ∀ n : ℕ, n ≤ 100 → is_valid_number (sequenceThree n) ∧
  (∀ m : ℕ, m < n → sequenceThree m < sequenceThree n) ∧
  sequenceThree 100 = 981 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_number_is_981_l1012_101245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_negative_two_equals_reciprocal_l1012_101250

theorem power_negative_two_equals_reciprocal (a : ℝ) (h : a ≠ 0) :
  (2 * a)^(-2 : ℤ) = 1 / (4 * a^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_negative_two_equals_reciprocal_l1012_101250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_gallons_bought_l1012_101253

def rental_cost : ℝ := 150
def gas_price : ℝ := 3.50
def mileage_cost : ℝ := 0.50
def total_distance : ℝ := 320
def total_expense : ℝ := 338

theorem gas_gallons_bought :
  (total_expense - rental_cost - mileage_cost * total_distance) / gas_price = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_gallons_bought_l1012_101253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_conversion_l1012_101282

noncomputable def spherical_to_rectangular (ρ θ φ : Real) : (Real × Real × Real) :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_conversion :
  let ρ : Real := 3
  let θ : Real := 3 * Real.pi / 4
  let φ : Real := Real.pi / 3
  spherical_to_rectangular ρ θ φ = (-3 * Real.sqrt 6 / 4, 3 * Real.sqrt 6 / 4, 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_conversion_l1012_101282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asian_games_mascot_sale_theorem_l1012_101221

/-- Represents the cost and profit calculations for the Asian Games mascot sale --/
def AsianGamesMascotSale (x : ℝ) : Prop :=
  let a (z : ℝ) := 90 / z  -- a is inversely proportional to quantity z
  let costPrice (z : ℝ) := 50 + a z
  let salesVolume := 10 - x / 10
  let totalRevenue := x * salesVolume
  let totalCost := costPrice salesVolume * salesVolume + 10  -- Including other expenses
  let totalProfit := totalRevenue - totalCost
  let profitPerMascot := x - costPrice salesVolume - 1 / salesVolume

  -- Condition: x < 100
  x < 100 ∧
  -- Question 1: Total profit when selling price is 70 yuan
  (x = 70 → totalProfit = -40) ∧
  -- Question 2: Selling price that maximizes profit per mascot
  (∀ y, y < 100 → profitPerMascot ≤ (fun _ => profitPerMascot) 90)

theorem asian_games_mascot_sale_theorem :
  ∀ x, AsianGamesMascotSale x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asian_games_mascot_sale_theorem_l1012_101221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_to_273_l1012_101294

theorem closest_perfect_square_to_273 :
  ∀ n : ℤ, n * n ≠ 289 → |n * n - 273| ≥ |289 - 273| :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_to_273_l1012_101294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_exactly_two_satisfied_l1012_101290

def probability_satisfied : ℝ := 0.65

def number_of_residents : ℕ := 4

def number_satisfied : ℕ := 2

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_exactly_two_satisfied :
  |binomial_coefficient number_of_residents number_satisfied *
    probability_satisfied ^ number_satisfied *
    (1 - probability_satisfied) ^ (number_of_residents - number_satisfied) -
    0.3106| < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_exactly_two_satisfied_l1012_101290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_program_selection_count_l1012_101207

def courses : Nat := 7
def program_size : Nat := 5
def required_course : Nat := 1
def math_courses : Nat := 2
def min_math_courses : Nat := 2

theorem program_selection_count :
  (Nat.choose (courses - 1) (program_size - 1)) -
  (Nat.choose (courses - 1 - math_courses) (program_size - 1)) -
  (Nat.choose (courses - 1 - math_courses) (program_size - 1 - 1)) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_program_selection_count_l1012_101207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABHFGD_l1012_101213

noncomputable section

-- Define the squares and point H
def square_ABCD : ℝ := 25
def square_EFGD : ℝ := 16

-- Define the side lengths of the squares
noncomputable def side_ABCD : ℝ := Real.sqrt square_ABCD
noncomputable def side_EFGD : ℝ := Real.sqrt square_EFGD

-- Define H as midpoint
noncomputable def BH : ℝ := side_ABCD / 2
noncomputable def EH : ℝ := side_EFGD / 2

-- Define the overlap area
noncomputable def overlap_area : ℝ := side_EFGD * EH

-- Theorem to prove
theorem area_ABHFGD : 
  square_ABCD + square_EFGD - overlap_area = 33 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABHFGD_l1012_101213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_product_l1012_101208

/-- Represents a hyperbola with semi-major axis a, semi-minor axis b, and focal distance c -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  ecc_eq : c / a = 2 * Real.sqrt 3 / 3
  focal_dist : c = a * a / 2

/-- A point on the hyperbola -/
structure HyperbolaPoint (h : Hyperbola) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The left focus of the hyperbola -/
def leftFocus (h : Hyperbola) : ℝ × ℝ := (-h.c, 0)

/-- The right focus of the hyperbola -/
def rightFocus (h : Hyperbola) : ℝ × ℝ := (h.c, 0)

/-- Vector from a point to the left focus -/
def vectorToLeftFocus (h : Hyperbola) (p : HyperbolaPoint h) : ℝ × ℝ :=
  (p.x + h.c, p.y)

/-- Vector from a point to the right focus -/
def vectorToRightFocus (h : Hyperbola) (p : HyperbolaPoint h) : ℝ × ℝ :=
  (p.x - h.c, p.y)

/-- Dot product of two 2D vectors -/
def dotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Magnitude of a 2D vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

/-- Main theorem to prove -/
theorem hyperbola_focal_product (h : Hyperbola) (p : HyperbolaPoint h)
    (dot_prod_eq : dotProduct (vectorToLeftFocus h p) (vectorToRightFocus h p) = 2) :
    magnitude (vectorToLeftFocus h p) * magnitude (vectorToRightFocus h p) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_product_l1012_101208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roller_coaster_median_l1012_101228

def roller_coaster_times : List ℕ := sorry

-- The list has 17 elements
axiom list_length : roller_coaster_times.length = 17

-- The list is sorted in ascending order
axiom list_sorted : List.Sorted (· ≤ ·) roller_coaster_times

-- The 9th element (0-indexed) is 2 minutes and 43 seconds
axiom ninth_element : roller_coaster_times[8]! = 2 * 60 + 43

-- Define the median for an odd-length sorted list
def median (l : List ℕ) : ℕ := l[(l.length - 1) / 2]!

-- Theorem statement
theorem roller_coaster_median :
  median roller_coaster_times = 163 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roller_coaster_median_l1012_101228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_special_triangle_l1012_101273

-- Define a right triangle XYZ
structure RightTriangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  is_right : (Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2) = 0

-- Define the properties of the specific triangle
def special_triangle (t : RightTriangle) : Prop :=
  let XY := ((t.Y.1 - t.X.1)^2 + (t.Y.2 - t.X.2)^2).sqrt
  let YZ := ((t.Z.1 - t.Y.1)^2 + (t.Z.2 - t.Y.2)^2).sqrt
  let ZX := ((t.X.1 - t.Z.1)^2 + (t.X.2 - t.Z.2)^2).sqrt
  (YZ^2 = XY^2 + ZX^2) ∧ (XY^2 = YZ^2 + ZX^2) ∧
  XY = 10

-- Theorem statement
theorem area_of_special_triangle (t : RightTriangle) (h : special_triangle t) :
  (1/2 : ℝ) * 10 * 10 = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_special_triangle_l1012_101273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_quotients_distinct_l1012_101256

theorem not_all_quotients_distinct (n : ℕ) (h_n : n > 3) : 
  ∀ (x : Fin n → ℕ), 
    (∀ i j : Fin n, i.val < j.val → x i < x j) →
    (∀ i : Fin n, x i < Nat.factorial (n - 1)) →
    ¬(∀ i j : Fin n, i.val < j.val → 
      ∀ k l : Fin n, k.val < l.val → 
        (x j / x i : ℕ) ≠ (x l / x k : ℕ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_quotients_distinct_l1012_101256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_intersect_Q_l1012_101285

def P : Set ℝ := {x : ℝ | x < 1}
def Q : Set ℝ := {x : ℝ | ∃ n : ℤ, x = n ∧ n^2 < 4}

theorem P_intersect_Q : P ∩ Q = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_intersect_Q_l1012_101285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_for_quadratic_inequality_l1012_101268

theorem x_range_for_quadratic_inequality (a : ℝ) (h1 : a ∈ Set.Icc (-1) 1) 
  (h2 : ∀ x : ℝ, x^2 + (a - 4) * x + 4 - 2 * a > 0) :
  {x : ℝ | x < 1 ∨ x > 3} = Set.univ := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_for_quadratic_inequality_l1012_101268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_eight_times_shorts_l1012_101201

/-- The cost of football equipment relative to shorts -/
structure FootballEquipmentCost where
  shorts : ℝ
  tshirt : ℝ
  boots : ℝ
  shin_guards : ℝ

/-- Conditions for the football equipment cost problem -/
structure FootballEquipmentProblem where
  cost : FootballEquipmentCost
  shorts_positive : cost.shorts > 0
  tshirt_condition : cost.shorts + cost.tshirt = 2 * cost.shorts
  boots_condition : cost.shorts + cost.boots = 5 * cost.shorts
  shin_guards_condition : cost.shorts + cost.shin_guards = 3 * cost.shorts

/-- The theorem stating that the total cost is 8 times the cost of shorts -/
theorem total_cost_is_eight_times_shorts (problem : FootballEquipmentProblem) :
  problem.cost.shorts + problem.cost.tshirt + problem.cost.boots + problem.cost.shin_guards
  = 8 * problem.cost.shorts := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_eight_times_shorts_l1012_101201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_equations_l1012_101202

-- Define the circle C
def circle_C (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line y = x + 1
def line_y_eq_x_plus_1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 + 1}

-- Define the point (3, 0)
def point_3_0 : ℝ × ℝ := (3, 0)

-- Define the points P and Q
def point_P : ℝ × ℝ := (3, 6)
def point_Q : ℝ × ℝ := (5, 6)

-- Define the line l
def line_l (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | k * p.1 - p.2 - 3 * k = 0}

-- Define the vertical line x = 3
def line_x_eq_3 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 3}

-- Define the chord length
noncomputable def chord_length (C : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) : ℝ := 2

theorem circle_and_line_equations :
  ∃ (center : ℝ × ℝ) (k : ℝ),
    center ∈ line_y_eq_x_plus_1 ∧
    point_P ∈ circle_C center (Real.sqrt 2) ∧
    point_Q ∈ circle_C center (Real.sqrt 2) ∧
    point_3_0 ∈ line_l k ∧
    chord_length (circle_C center (Real.sqrt 2)) (line_l k) = 2 ∧
    (
      (circle_C center (Real.sqrt 2) = circle_C (4, 5) (Real.sqrt 2)) ∧
      (line_l k = line_l (12/5) ∨ line_l k = line_x_eq_3)
    ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_equations_l1012_101202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_term_of_sequence_l1012_101232

/-- Given a sequence with the first few terms -b/a, 3b/a², 5b/a³, 7b/a⁴, ...,
    this theorem states that the nth term of the sequence is (-1)ⁿ * (2n-1)b / aⁿ. -/
theorem nth_term_of_sequence (a b : ℝ) (n : ℕ) :
  let sequence := fun k => (-1)^k * (2*k - 1) * b / a^k
  sequence 1 = -b/a ∧
  sequence 2 = 3*b/a^2 ∧
  sequence 3 = 5*b/a^3 ∧
  sequence 4 = 7*b/a^4 →
  ∀ k, sequence k = (-1)^k * (2*k - 1) * b / a^k :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_term_of_sequence_l1012_101232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_detergent_ranking_l1012_101257

structure DetergentBox where
  size : String
  cost : ℚ
  quantity : ℚ

def small : DetergentBox := { size := "S", cost := 1, quantity := 5 }
def medium : DetergentBox := { size := "M", cost := 3/2, quantity := 8 }
def large : DetergentBox := { size := "L", cost := 39/20, quantity := 10 }

def costPerOunce (box : DetergentBox) : ℚ := box.cost / box.quantity

theorem detergent_ranking :
  (costPerOunce medium < costPerOunce large) ∧
  (costPerOunce large < costPerOunce small) ∧
  (medium.cost = 3/2 * small.cost) ∧
  (medium.quantity = 4/5 * large.quantity) ∧
  (large.quantity = 2 * small.quantity) ∧
  (large.cost = 13/10 * medium.cost) := by
  sorry

#eval costPerOunce small
#eval costPerOunce medium
#eval costPerOunce large

end NUMINAMATH_CALUDE_ERRORFEEDBACK_detergent_ranking_l1012_101257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_max_profit_at_optimal_reduction_l1012_101226

/-- Represents the profit function for a product with given cost and sales parameters. -/
def profit_function (x : ℝ) : ℝ := -20 * x^2 + 100 * x + 6000

/-- Represents the constraint on the price reduction. -/
def price_reduction_constraint (x : ℝ) : Prop := 0 ≤ x ∧ x < 20

/-- Theorem stating that the profit function is maximized at x = 2.5 within the given constraints. -/
theorem profit_maximization :
  ∃ (x : ℝ), price_reduction_constraint x ∧
    profit_function x = 6125 ∧
    ∀ (y : ℝ), price_reduction_constraint y → profit_function y ≤ profit_function x :=
by sorry

/-- Corollary stating that the maximum profit is achieved at x = 2.5. -/
theorem max_profit_at_optimal_reduction :
  ∃ (x : ℝ), x = 2.5 ∧ price_reduction_constraint x ∧ profit_function x = 6125 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_max_profit_at_optimal_reduction_l1012_101226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_28_l1012_101255

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point over the x-axis -/
def reflectOverXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- Reflects a point over the line y = x -/
def reflectOverYEqualsX (p : Point) : Point :=
  { x := p.y, y := p.x }

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (a b c : Point) : ℝ :=
  let base := |b.y - a.y|
  let height := |c.x - a.x|
  (1 / 2) * base * height

theorem triangle_area_is_28 : 
  let a : Point := { x := 3, y := 4 }
  let b : Point := reflectOverXAxis a
  let c : Point := reflectOverYEqualsX b
  triangleArea a b c = 28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_28_l1012_101255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_speed_proof_l1012_101279

/-- The speed of a bike given its distance traveled and time taken -/
noncomputable def bike_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

/-- Theorem: A bike traveling 350 meters in 7 seconds has a speed of 50 meters per second -/
theorem bike_speed_proof :
  bike_speed 350 7 = 50 := by
  -- Unfold the definition of bike_speed
  unfold bike_speed
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_speed_proof_l1012_101279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_undefined_at_two_l1012_101284

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (x - 5)

-- State the theorem
theorem inverse_f_undefined_at_two :
  ∀ x : ℝ, f x = 2 → x = 5 :=
by
  intro x
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

-- Note: We're proving that if f(x) = 2, then x = 5, which is equivalent to
-- showing that f^(-1)(2) is undefined (as it would lead to division by zero in the original function).

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_undefined_at_two_l1012_101284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_30cm_square_net_tetrahedron_l1012_101281

/-- A tetrahedron with an unfolded net that forms a square --/
structure SquareNetTetrahedron where
  /-- The side length of the square formed by the unfolded net --/
  side_length : ℝ
  /-- Assumption that the side length is positive --/
  side_length_pos : side_length > 0

/-- Calculate the volume of a tetrahedron given its unfolded net forms a square --/
noncomputable def tetrahedron_volume (t : SquareNetTetrahedron) : ℝ :=
  (t.side_length ^ 3) / 4

/-- Theorem stating that a tetrahedron with an unfolded net forming a 30 cm square has a volume of 1125 cm³ --/
theorem volume_of_30cm_square_net_tetrahedron :
  let t : SquareNetTetrahedron := ⟨30, by norm_num⟩
  tetrahedron_volume t = 1125 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_30cm_square_net_tetrahedron_l1012_101281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acid_mixture_replacement_l1012_101220

theorem acid_mixture_replacement (initial_concentration : ℝ) 
                                 (replacement_concentration : ℝ) 
                                 (final_concentration : ℝ) 
                                 (total_volume : ℝ) :
  initial_concentration = 0.5 →
  replacement_concentration = 0.2 →
  final_concentration = 0.35 →
  total_volume > 0 →
  (initial_concentration - final_concentration) / 
  (initial_concentration - replacement_concentration) * total_volume / total_volume = 1 / 2 := by
  sorry

#check acid_mixture_replacement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acid_mixture_replacement_l1012_101220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_g_g_l1012_101287

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x - 1)

-- Theorem statement
theorem smallest_x_in_domain_of_g_g :
  ∀ x : ℝ, (∃ y : ℝ, g (g x) = y) → x ≥ 2 ∧
  ∀ z : ℝ, z < 2 → ¬(∃ y : ℝ, g (g z) = y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_g_g_l1012_101287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_cos_greater_than_one_l1012_101218

theorem negation_of_exists_cos_greater_than_one :
  (¬ ∃ x : ℝ, Real.cos x > 1) ↔ (∀ x : ℝ, Real.cos x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_cos_greater_than_one_l1012_101218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_conic_section_l1012_101247

-- Define the geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 0 = 1 ∧ a 4 = 81 ∧ ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the conic section
def conic_section (a₂ : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 / a₂ = 1

-- Define eccentricity
noncomputable def eccentricity (a₂ : ℝ) : ℝ :=
  Real.sqrt (a₂ - 1) / Real.sqrt a₂

-- Theorem statement
theorem eccentricity_of_conic_section (a : ℕ → ℝ) :
  geometric_sequence a →
  eccentricity (a 2) = 2 * Real.sqrt 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_conic_section_l1012_101247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beetle_reaches_all_even_columns_l1012_101258

/-- Represents the beetle's position on the grid -/
structure Position where
  x : Int
  y : Int

/-- Represents the possible directions the beetle can face -/
inductive Direction
  | Up
  | Right
  | Down
  | Left

/-- Represents a step the beetle can take -/
structure Step where
  dir : Direction
  distance : Nat

/-- The set of possible steps the beetle can take -/
def possibleSteps : List Step :=
  [ ⟨Direction.Right, 2⟩
  , ⟨Direction.Left, 4⟩
  , ⟨Direction.Up, 3⟩
  , ⟨Direction.Down, 5⟩ ]

/-- Function to determine if a column is even -/
def isEvenColumn (p : Position) : Prop :=
  p.x % 2 = 0

/-- Function to apply a step to a position -/
def applyStep (p : Position) (s : Step) : Position :=
  match s.dir with
  | Direction.Right => ⟨p.x + s.distance, p.y⟩
  | Direction.Left => ⟨p.x - s.distance, p.y⟩
  | Direction.Up => ⟨p.x, p.y + s.distance⟩
  | Direction.Down => ⟨p.x, p.y - s.distance⟩

/-- The starting position of the beetle -/
def startPosition : Position := ⟨0, 0⟩

/-- Theorem stating that the beetle can reach all even-numbered columns -/
theorem beetle_reaches_all_even_columns :
  ∀ (p : Position), isEvenColumn p →
    ∃ (steps : List Step),
      (∀ s ∈ steps, s ∈ possibleSteps) ∧
      steps.foldl applyStep startPosition = p :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beetle_reaches_all_even_columns_l1012_101258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approximation_l1012_101272

/-- The length of the train in meters -/
noncomputable def train_length : ℝ := 160

/-- The time taken for the train to cross a stationary point, in seconds -/
noncomputable def crossing_time : ℝ := 9

/-- The speed of the train in meters per second -/
noncomputable def train_speed : ℝ := train_length / crossing_time

theorem train_speed_approximation : 
  ∃ ε > 0, |train_speed - 17.78| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approximation_l1012_101272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_sum_theorem_l1012_101244

theorem matrix_sum_theorem (x y z : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![x, y, z; y, z, x; z, x, y]
  ¬(IsUnit (Matrix.det M)) →
  (x / (y + z) + y / (z + x) + z / (x + y) = -3) ∨
  (x / (y + z) + y / (z + x) + z / (x + y) = 3/2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_sum_theorem_l1012_101244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mass_ratio_is_three_halves_l1012_101248

/-- The maximum ratio of bead mass to hoop mass for which the hoop remains grounded -/
noncomputable def max_mass_ratio : ℝ := 3/2

/-- Represents the setup of two beads sliding on a vertical hoop -/
structure BeadHoopSystem where
  m : ℝ     -- mass of each bead
  m_h : ℝ   -- mass of the hoop
  R : ℝ     -- radius of the hoop
  g : ℝ     -- gravitational acceleration

/-- Condition for the hoop to remain in contact with the ground -/
def hoop_grounded (sys : BeadHoopSystem) (α : ℝ) : Prop :=
  sys.m_h * sys.g ≥ 2 * sys.m * sys.g * (2 - 3 * Real.cos α) * Real.cos α

/-- Theorem stating that the maximum mass ratio is 3/2 -/
theorem max_mass_ratio_is_three_halves (sys : BeadHoopSystem) :
  (∀ α, hoop_grounded sys α) → sys.m / sys.m_h ≤ max_mass_ratio := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mass_ratio_is_three_halves_l1012_101248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_M_l1012_101297

noncomputable def f (x : ℝ) : ℝ := Real.log x

noncomputable def M : ℝ × ℝ := (Real.exp 1, 1)

def tangent_line (x y : ℝ) : Prop := x - Real.exp 1 * y = 0

theorem tangent_line_at_M :
  let (a, b) := M
  (∀ x, x > 0 → HasDerivAt f (1 / x) x) →
  tangent_line a b ∧
  ∃ k, ∀ x y, tangent_line x y ↔ y - b = k * (x - a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_M_l1012_101297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_plane_l1012_101252

/-- Given a plane ABC and a point M in that plane, prove that if for any point O in space,
    OM = x * OA + (1/3) * OB + (1/3) * OC, then x = 1/3 -/
theorem point_in_plane (A B C M : EuclideanSpace ℝ (Fin 3)) (x : ℝ) : 
  (∃ (a b c : ℝ), a • A + b • B + c • C = M ∧ a + b + c = 1) →
  (∀ (O : EuclideanSpace ℝ (Fin 3)), M - O = x • (A - O) + (1/3) • (B - O) + (1/3) • (C - O)) →
  x = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_plane_l1012_101252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_hexagon_ABQCDP_l1012_101264

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a trapezoid ABCD -/
structure Trapezoid where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Predicate to check if a point is on the angle bisector -/
def is_angle_bisector (A B C P : Point) : Prop :=
  sorry

/-- Predicate to check if an angle bisector divides an angle into given degrees -/
def divides_angle_into (deg1 deg2 : ℝ) (A B C P : Point) : Prop :=
  sorry

/-- Function to calculate the area of the hexagon ABQCDP -/
noncomputable def area_hexagon (ABCD : Trapezoid) : ℝ :=
  sorry

/-- Theorem: Area of hexagon ABQCDP in trapezoid ABCD -/
theorem area_hexagon_ABQCDP (ABCD : Trapezoid) 
  (AB_parallel_CD : (ABCD.B.y - ABCD.A.y) / (ABCD.B.x - ABCD.A.x) = (ABCD.D.y - ABCD.C.y) / (ABCD.D.x - ABCD.C.x))
  (AB_length : Real.sqrt ((ABCD.B.x - ABCD.A.x)^2 + (ABCD.B.y - ABCD.A.y)^2) = 13)
  (BC_length : Real.sqrt ((ABCD.C.x - ABCD.B.x)^2 + (ABCD.C.y - ABCD.B.y)^2) = 7)
  (CD_length : Real.sqrt ((ABCD.D.x - ABCD.C.x)^2 + (ABCD.D.y - ABCD.C.y)^2) = 23)
  (DA_length : Real.sqrt ((ABCD.A.x - ABCD.D.x)^2 + (ABCD.A.y - ABCD.D.y)^2) = 9)
  (angle_A_bisector : ∃ P : Point, is_angle_bisector ABCD.A ABCD.B ABCD.D P ∧ divides_angle_into 30 60 ABCD.A ABCD.B ABCD.D P)
  (angle_D_bisector : ∃ P : Point, is_angle_bisector ABCD.D ABCD.C ABCD.A P ∧ divides_angle_into 30 60 ABCD.D ABCD.C ABCD.A P)
  (angle_B_bisector : ∃ Q : Point, is_angle_bisector ABCD.B ABCD.A ABCD.C Q)
  (angle_C_bisector : ∃ Q : Point, is_angle_bisector ABCD.C ABCD.D ABCD.B Q)
  : area_hexagon ABCD = 14 * Real.sqrt 37.44 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_hexagon_ABQCDP_l1012_101264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_blocks_l1012_101251

/-- Represents the number of blocks in a company --/
noncomputable def number_of_blocks (total_amount : ℚ) (gift_worth : ℚ) (workers_per_block : ℚ) : ℚ :=
  (total_amount / gift_worth) / workers_per_block

/-- Theorem stating that the number of blocks in the company is 15 --/
theorem company_blocks :
  let total_amount : ℚ := 6000
  let gift_worth : ℚ := 2
  let workers_per_block : ℚ := 200
  number_of_blocks total_amount gift_worth workers_per_block = 15 := by
  -- Unfold the definition and simplify
  unfold number_of_blocks
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_blocks_l1012_101251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_tower_eventually_constant_l1012_101229

def power_tower (i : ℕ) : ℕ :=
  match i with
  | 0 => 2
  | n + 1 => 2^(power_tower n)

theorem power_tower_eventually_constant (n : ℕ) (hn : n ≥ 1) :
  ∃ K : ℕ, ∀ k ≥ K, power_tower k % n = power_tower (k + 1) % n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_tower_eventually_constant_l1012_101229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_on_line_l1012_101235

/-- Given vectors and a point on a line, prove the minimum dot product -/
theorem min_dot_product_on_line (O P A B M : ℝ × ℝ) : 
  O = (0, 0) →
  P = (2, 1) →
  A = (1, 7) →
  B = (5, 1) →
  ∃ (t : ℝ), M = (2*t, t) →
  ∀ (t : ℝ), (A.1 - M.1) * (B.1 - M.1) + (A.2 - M.2) * (B.2 - M.2) ≥ -8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_on_line_l1012_101235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_equivalence_l1012_101246

-- Define the points in a 2D Euclidean space
variable (A B C D E F M : EuclideanSpace ℝ (Fin 2))

-- Define the circle
variable (circle : Sphere (EuclideanSpace ℝ (Fin 2)) ℝ)

-- Define the conditions
variable (h1 : A ∈ circle)
variable (h2 : B ∈ circle)
variable (h3 : D ∈ circle ∩ (Segment A C).toSet)
variable (h4 : E ∈ circle ∩ (Segment B C).toSet)
variable (h5 : F ∈ (Line.throughPts B A).toSet ∩ (Line.throughPts E D).toSet)
variable (h6 : M ∈ (Line.throughPts B D).toSet ∩ (Line.throughPts C F).toSet)

-- State the theorem
theorem geometric_equivalence :
  dist M F = dist M C ↔ dist M B * dist M D = (dist M C) ^ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_equivalence_l1012_101246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1012_101206

-- Define the hyperbola and its properties
def Hyperbola (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0

-- Define the foci and a point on the right branch
def Foci (F₁ F₂ : ℝ × ℝ) (a b : ℝ) : Prop :=
  Hyperbola a b ∧ F₁.1 < F₂.1

def RightBranchPoint (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) (a b : ℝ) : Prop :=
  Foci F₁ F₂ a b ∧ P.1 > 0

-- Define the conditions
def ConditionA (P F₁ F₂ : ℝ × ℝ) : Prop :=
  dist P F₂ = dist F₁ F₂

-- Replace distancePointLine with a placeholder function
def distancePointToLine (point : ℝ × ℝ) (line : Set (ℝ × ℝ)) : ℝ := sorry

def ConditionB (P F₁ F₂ : ℝ × ℝ) (a : ℝ) : Prop :=
  distancePointToLine F₂ {x | ∃ t, x = (1 - t) • P + t • F₁} = 2 * a

-- Define eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := sorry

-- Theorem statement
theorem hyperbola_eccentricity
  (a b : ℝ) (F₁ F₂ P : ℝ × ℝ)
  (h₁ : Hyperbola a b)
  (h₂ : RightBranchPoint P F₁ F₂ a b)
  (h₃ : ConditionA P F₁ F₂)
  (h₄ : ConditionB P F₁ F₂ a) :
  eccentricity a b = 5 / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1012_101206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_locus_proof_l1012_101216

-- Define the ellipse parameters
noncomputable def a : ℝ := 5
noncomputable def b : ℝ := 3
noncomputable def e : ℝ := 4/5

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define the directrix
def directrix (x : ℝ) : Prop := x = 25/4

-- Define point A
def A : ℝ × ℝ := (6, 0)

-- Define the locus equation
def locus (x y : ℝ) : Prop := (y-6)^2/25 + (x-6)^2/9 = 1

-- Theorem statement
theorem ellipse_and_locus_proof :
  (a > b ∧ b ≥ 0) →
  (∀ x y, ellipse x y ↔ x^2/25 + y^2/9 = 1) ∧
  (∀ x y, locus x y ↔ 
    ∃ x₀ y₀, ellipse x₀ y₀ ∧ 
    (x - 6)^2 + y^2 = (x₀ - 6)^2 + y₀^2 ∧
    (x - 6) * (x₀ - 6) + y * y₀ = 0) :=
by
  sorry

#check ellipse_and_locus_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_locus_proof_l1012_101216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_pi_twelfth_l1012_101236

theorem cos_minus_sin_pi_twelfth : Real.cos (π / 12) - Real.sin (π / 12) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_pi_twelfth_l1012_101236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_largest_and_smallest_l1012_101205

def numbers : List ℕ := [10, 11, 12]

theorem sum_of_largest_and_smallest : 
  (List.maximum numbers).getD 0 + (List.minimum numbers).getD 0 = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_largest_and_smallest_l1012_101205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_right_triangle_l1012_101276

/-- A triangle is a right triangle if one of its angles is a right angle -/
def IsRightTriangle (triangle : Set (ℝ × ℝ)) : Prop := sorry

/-- A point is the midpoint of a line segment if it's equidistant from both endpoints -/
def IsMidpoint (M A B : ℝ × ℝ) : Prop := sorry

/-- Given a right triangle DEF with DE = 5 and EF = 12, and N the midpoint of EF,
    prove that the length of median DN is 6.5 -/
theorem median_length_right_triangle (D E F N : ℝ × ℝ) : 
  let triangle := {D, E, F}
  let DE := Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2)
  let EF := Real.sqrt ((E.1 - F.1)^2 + (E.2 - F.2)^2)
  let DF := Real.sqrt ((D.1 - F.1)^2 + (D.2 - F.2)^2)
  let DN := Real.sqrt ((D.1 - N.1)^2 + (D.2 - N.2)^2)
  IsRightTriangle triangle ∧
  DE = 5 ∧
  EF = 12 ∧
  IsMidpoint N E F →
  DN = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_right_triangle_l1012_101276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_profit_share_l1012_101214

-- Define the capital shares
def a_share : ℚ := 1 / 3
def b_share : ℚ := 1 / 4
def c_share : ℚ := 1 / 5
def d_share : ℚ := 1 - (a_share + b_share + c_share)

-- Define the total profit
def total_profit : ℚ := 2415

-- Theorem statement
theorem a_profit_share : 
  a_share * total_profit = 805 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_profit_share_l1012_101214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_determine_crossing_time_l1012_101212

-- Define the train lengths
def train1_length : ℝ := 120
def train2_length : ℝ := 150

-- Define the time they take to cross each other when moving in the same direction
def crossing_time : ℝ := 135

-- Define the speeds of the trains as variables
variable (v1 v2 : ℝ)

-- Define the relative speed of the trains
def relative_speed (v1 v2 : ℝ) : ℝ := v2 - v1

-- Theorem stating that we cannot determine the time for the 120m train to cross the stationary man
theorem cannot_determine_crossing_time :
  ∀ (t : ℝ), relative_speed v1 v2 * crossing_time = train1_length + train2_length →
  ¬ (∃! (unique_t : ℝ), t = train1_length / v1) :=
by
  sorry

#check cannot_determine_crossing_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_determine_crossing_time_l1012_101212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_multiples_difference_l1012_101209

theorem smallest_multiples_difference (c d : ℕ) : 
  (c ≥ 10000 ∧ c < 100000 ∧ c % 7 = 0 ∧ ∀ x : ℕ, x ≥ 10000 ∧ x < 100000 ∧ x % 7 = 0 → c ≤ x) →
  (d ≥ 1000 ∧ d < 10000 ∧ d % 9 = 0 ∧ ∀ y : ℕ, y ≥ 1000 ∧ y < 10000 ∧ y % 9 = 0 → d ≤ y) →
  c - d = 8995 :=
by
  sorry

#check smallest_multiples_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_multiples_difference_l1012_101209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_10_plus_3_power_property_l1012_101299

noncomputable def P (x : ℝ) := x - ⌊x⌋

theorem sqrt_10_plus_3_power_property (n : ℕ) :
  let x := (Real.sqrt 10 + 3) ^ (2 * n + 1)
  let I := ⌊x⌋
  let F := x - I
  P (I + F) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_10_plus_3_power_property_l1012_101299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_point_nine_recurring_and_seven_l1012_101291

/-- The product of 0.999... and 7 is 7 -/
theorem product_of_point_nine_recurring_and_seven :
  (∃ (x : ℝ), (∀ (n : ℕ), x = 1 - 1 / (10^n)) ∧ x * 7 = 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_point_nine_recurring_and_seven_l1012_101291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l1012_101283

/-- The angle of inclination of a line given by parametric equations -/
noncomputable def angle_of_inclination (x y : ℝ → ℝ) : ℝ := sorry

/-- Parametric equation for x -/
noncomputable def x (t : ℝ) : ℝ := 1 - (1/2) * t

/-- Parametric equation for y -/
noncomputable def y (t : ℝ) : ℝ := (Real.sqrt 3 / 2) * t

/-- The angle of inclination of the line is 2π/3 (120 degrees) -/
theorem line_inclination :
  angle_of_inclination x y = 2 * Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_l1012_101283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_projection_l1012_101278

/-- Parabola structure -/
structure Parabola where
  focus : ℝ × ℝ
  vertex : ℝ × ℝ

/-- Point on a parabola -/
def PointOnParabola (p : Parabola) (P : ℝ × ℝ) : Prop :=
  (P.1)^2 = 4 * P.2

/-- Distance between two points -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- Circle intersects x-axis -/
def CircleIntersectsXAxis (center : ℝ × ℝ) (radius : ℝ) (A B : ℝ × ℝ) : Prop :=
  A.2 = 0 ∧ B.2 = 0 ∧ distance center A = radius ∧ distance center B = radius

/-- Dot product of two vectors -/
def dotProduct (A B C : ℝ × ℝ) : ℝ :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)

theorem parabola_point_projection (p : Parabola) (P A B : ℝ × ℝ) :
  PointOnParabola p P →
  distance P p.focus = 5 →
  CircleIntersectsXAxis P 5 A B →
  dotProduct A P B = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_projection_l1012_101278
