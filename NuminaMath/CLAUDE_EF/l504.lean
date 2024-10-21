import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_right_focus_to_line_l504_50445

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define the line
def line (x y : ℝ) : Prop := x + 2*y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus : ℝ × ℝ := (3, 0)

-- Define the distance function from a point to a line
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

-- Theorem statement
theorem distance_right_focus_to_line :
  distance_point_to_line right_focus.1 right_focus.2 1 2 (-8) = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_right_focus_to_line_l504_50445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nectar_water_content_l504_50443

/-- The percentage of water in honey -/
noncomputable def honey_water_percentage : ℝ := 30

/-- The mass of flower-nectar needed to produce 1 kg of honey -/
noncomputable def nectar_to_honey_ratio : ℝ := 1.4

/-- The mass of honey produced -/
noncomputable def honey_mass : ℝ := 1

/-- The percentage of water in flower-nectar -/
noncomputable def nectar_water_percentage : ℝ := (honey_water_percentage * honey_mass) / nectar_to_honey_ratio

theorem nectar_water_content :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |nectar_water_percentage - 21.43| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nectar_water_content_l504_50443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiring_probabilities_l504_50431

/-- Represents the number of applicant sequences where the kth best applicant is hired -/
def A (k : ℕ) : ℕ := sorry

/-- The total number of possible applicant sequences -/
def total_sequences : ℕ := Nat.factorial 10

/-- The probability of hiring the kth best applicant -/
noncomputable def P (k : ℕ) : ℚ := (A k : ℚ) / total_sequences

theorem hiring_probabilities :
  (∀ k ∈ Finset.range 7, A k > A (k + 1)) ∧
  (A 8 = A 9) ∧ (A 9 = A 10) ∧
  (P 1 + P 2 + P 3 > 7/10) ∧
  (P 8 + P 9 + P 10 < 1/10) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiring_probabilities_l504_50431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_compromise_function_l504_50455

/-- The domain of the functions -/
def D : Set ℝ := Set.Icc 1 (2 * Real.exp 1)

/-- The function f -/
def f (k : ℝ) (x : ℝ) : ℝ := (k - 1) * x - 1

/-- The function g -/
def g (x : ℝ) : ℝ := 0

/-- The function h -/
noncomputable def h (x : ℝ) : ℝ := (x + 1) * Real.log x

/-- The property of being a compromise function -/
def isCompromiseFunction (k : ℝ) : Prop :=
  ∀ x ∈ D, g x ≤ f k x ∧ f k x ≤ h x

theorem unique_compromise_function :
  ∃! k : ℝ, isCompromiseFunction k ∧ k = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_compromise_function_l504_50455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_A_l504_50464

theorem find_A (x y A : ℝ) (h1 : (2 : ℝ)^x = (7 : ℝ)^(2*y)) (h2 : (2 : ℝ)^x = A) (h3 : (7 : ℝ)^(2*y) = A) 
  (h4 : 1/x + 2/y = 2) : A = 7 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_A_l504_50464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jake_jogging_distance_l504_50441

/-- The distance between Jake's home and the park in kilometers. -/
noncomputable def park_distance : ℝ := 4

/-- The fraction of the remaining distance Jake jogs each time. -/
noncomputable def jog_fraction : ℝ := 1/2

/-- The point closer to Jake's home that he approaches. -/
noncomputable def C : ℝ := 4/3

/-- The point closer to the park that Jake approaches. -/
noncomputable def D : ℝ := 8/3

/-- Theorem stating that the absolute difference between the two points
Jake fluctuates between is 4/3 kilometers. -/
theorem jake_jogging_distance : |C - D| = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jake_jogging_distance_l504_50441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_tangent_length_is_correct_l504_50499

/-- Circle C1 with center (3, 4) and radius 5 -/
def C1 : Set (ℝ × ℝ) :=
  {p | (p.1 - 3)^2 + (p.2 - 4)^2 = 25}

/-- Circle C2 with center (-6, -5) and radius 4 -/
def C2 : Set (ℝ × ℝ) :=
  {p | (p.1 + 6)^2 + (p.2 + 5)^2 = 16}

/-- The length of the shortest line segment PQ that is tangent to C1 at P and to C2 at Q -/
noncomputable def shortestTangentLength : ℝ := 9 * Real.sqrt 2 - 9

theorem shortest_tangent_length_is_correct :
  ∃ (P Q : ℝ × ℝ), P ∈ C1 ∧ Q ∈ C2 ∧
  (∀ (P' Q' : ℝ × ℝ), P' ∈ C1 → Q' ∈ C2 →
    Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) ≥ shortestTangentLength) ∧
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = shortestTangentLength :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_tangent_length_is_correct_l504_50499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_of_two_l504_50482

def f (x : ℝ) : ℝ := x^2 - 2

theorem greatest_power_of_two (a b : ℕ) (h1 : Nat.Coprime a b) 
  (h2 : f (f (f (f (f (f (f (2.5))))))) = a / b) : 
  (Finset.sup (Finset.filter (λ n => (2^n : ℕ) ∣ a*b + a + b - 1) (Finset.range 129)) id) = 128 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_of_two_l504_50482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_from_area_l504_50406

-- Define the circle's area
noncomputable def circle_area : ℝ := 49 * Real.pi

-- Theorem statement
theorem circle_diameter_from_area :
  ∃ (d : ℝ), d = 14 ∧ circle_area = Real.pi * ((d / 2) ^ 2) :=
by
  -- Introduce the diameter
  let d : ℝ := 14
  
  -- Prove the existence
  use d
  
  constructor
  
  -- Prove d = 14
  · rfl
  
  -- Prove the area equation
  · calc
      circle_area = 49 * Real.pi := rfl
      _ = Real.pi * 49 := by ring
      _ = Real.pi * (7 ^ 2) := by ring
      _ = Real.pi * ((14 / 2) ^ 2) := by ring
      _ = Real.pi * ((d / 2) ^ 2) := by rfl
  
  -- The proof is complete

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_from_area_l504_50406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_is_eleven_l504_50461

def sequenceNumbers : List ℕ := [11, 23, 47, 83, 131, 191, 263, 347, 443, 551, 671]

theorem first_number_is_eleven : sequenceNumbers.head? = some 11 := by
  rfl

#check first_number_is_eleven

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_is_eleven_l504_50461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_camping_hike_distance_l504_50400

/-- Represents the distance hiked between two consecutive landmarks -/
def HikeSegment : Type := ℝ

/-- Calculates the total distance hiked given the distances of individual segments -/
def totalDistance (segments : List ℝ) : ℝ :=
  segments.sum

theorem camping_hike_distance 
  (car_to_stream stream_to_meadow meadow_to_campsite : ℝ)
  (h1 : car_to_stream = 0.2)
  (h2 : stream_to_meadow = 0.4)
  (h3 : meadow_to_campsite = 0.1) :
  totalDistance [car_to_stream, stream_to_meadow, meadow_to_campsite] = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_camping_hike_distance_l504_50400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_tank_capacity_approx_l504_50415

/-- Given a tank with a leak and an inlet pipe, this theorem proves the capacity of the tank. -/
theorem tank_capacity
  (leak_empty_time : ℝ)  -- Time to empty the tank with only the leak active
  (inlet_rate : ℝ)       -- Rate at which the inlet fills the tank
  (combined_empty_time : ℝ)  -- Time to empty the tank with both leak and inlet active
  (h1 : leak_empty_time = 6)  -- The leak empties the full tank in 6 hours
  (h2 : inlet_rate = 4.5)     -- The inlet fills at 4.5 liters per minute
  (h3 : combined_empty_time = 8)  -- The tank empties in 8 hours with both leak and inlet active
  : ℝ :=
  6480 / 7  -- The capacity of the tank in liters

/-- The capacity of the tank is approximately 925.71 liters. -/
theorem tank_capacity_approx :
  |tank_capacity 6 4.5 8 rfl rfl rfl - 925.71| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_tank_capacity_approx_l504_50415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_direction_vector_l504_50444

def projection_matrix : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![1/9, 1/9, 1/3],
    ![1/9, 1/9, 1/3],
    ![1/3, 1/3, 1/3]]

def direction_vector : Fin 3 → ℚ :=
  ![1, 1, 1]

theorem projection_direction_vector :
  ∃ (k : ℚ), k ≠ 0 ∧ ∀ (v : Fin 3 → ℚ),
    Matrix.mulVec projection_matrix v = k • direction_vector := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_direction_vector_l504_50444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_journey_length_is_107_l504_50402

/-- Represents a city in the country -/
structure City where
  id : Nat
  roads : Finset Nat

/-- Represents the country with its cities and road network -/
structure Country where
  cities : Finset City
  total_cities : Nat
  road_network : City → City → Bool

/-- Represents the driver's journey through the cities -/
def Journey (country : Country) (path : List City) : Prop :=
  sorry

/-- The maximum possible value of N for a valid journey -/
def max_journey_length (country : Country) : Nat :=
  sorry

theorem max_journey_length_is_107 (country : Country) :
  country.total_cities = 110 →
  (∀ c : City, c ∈ country.cities → c.id ≤ 110) →
  (∀ c1 c2 : City, c1 ∈ country.cities → c2 ∈ country.cities → 
    country.road_network c1 c2 ∨ ¬country.road_network c1 c2) →
  (∃ journey : List City, Journey country journey ∧ 
    (∀ k, k < journey.length → (journey.get ⟨k, by sorry⟩).roads.card = k + 1)) →
  max_journey_length country = 107 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_journey_length_is_107_l504_50402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_in_rectangle_l504_50481

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A rectangle in a 2D plane -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Check if a point is inside a rectangle -/
def inRectangle (p : Point) (r : Rectangle) : Prop :=
  0 ≤ p.x ∧ p.x ≤ r.width ∧ 0 ≤ p.y ∧ p.y ≤ r.height

theorem points_in_rectangle (points : Finset Point) (r : Rectangle) :
  r.width = 4 →
  r.height = 3 →
  points.card = 6 →
  (∀ p ∈ points, inRectangle p r) →
  ∃ p q : Point, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧ distance p q ≤ Real.sqrt 5 := by
  sorry

#check points_in_rectangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_in_rectangle_l504_50481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l504_50454

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.exp x + x

-- Define the point of tangency
def x₀ : ℝ := 0
def y₀ : ℝ := 1

-- Define the slope of the tangent line
noncomputable def m : ℝ := Real.exp x₀ + 1

-- Statement of the theorem
theorem tangent_line_equation :
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = 2 * x + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l504_50454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_one_plus_z_eq_sqrt_two_l504_50457

noncomputable def z : ℂ := (1 - Complex.I) / (Complex.I + 1)

theorem abs_one_plus_z_eq_sqrt_two : Complex.abs (1 + z) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_one_plus_z_eq_sqrt_two_l504_50457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_congruence_criteria_l504_50430

/-- Definition of a triangle -/
structure Triangle where
  -- We'll use this as a placeholder for now
  mk :: 

/-- Definition of triangle congruence -/
def triangles_congruent (t1 t2 : Triangle) : Prop := sorry

/-- Two triangles with equal corresponding sides -/
def equal_sides (t1 t2 : Triangle) : Prop := sorry

/-- Two triangles with equal corresponding angles -/
def equal_angles (t1 t2 : Triangle) : Prop := sorry

/-- Two triangles with two equal sides and the included angle -/
def equal_two_sides_included_angle (t1 t2 : Triangle) : Prop := sorry

/-- Two triangles with two equal angles and the opposite side of one angle -/
def equal_two_angles_opposite_side (t1 t2 : Triangle) : Prop := sorry

/-- Two triangles with two equal sides and the opposite angle of one side -/
def equal_two_sides_opposite_angle (t1 t2 : Triangle) : Prop := sorry

theorem triangle_congruence_criteria :
  (∀ t1 t2 : Triangle, equal_sides t1 t2 → triangles_congruent t1 t2) ∧
  (∃ t1 t2 : Triangle, equal_angles t1 t2 ∧ ¬triangles_congruent t1 t2) ∧
  (∀ t1 t2 : Triangle, equal_two_sides_included_angle t1 t2 → triangles_congruent t1 t2) ∧
  (∀ t1 t2 : Triangle, equal_two_angles_opposite_side t1 t2 → triangles_congruent t1 t2) ∧
  (∃ t1 t2 : Triangle, equal_two_sides_opposite_angle t1 t2 ∧ ¬triangles_congruent t1 t2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_congruence_criteria_l504_50430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_decimal_mean_l504_50401

/-- 
Represents a fraction q/p as an infinite periodic decimal,
where p is an odd prime not equal to 5.
-/
structure PeriodicDecimal where
  q : ℕ
  p : ℕ
  h_prime : Nat.Prime p
  h_odd : Odd p
  h_not_five : p ≠ 5
  period : List ℕ
  h_period : ∀ d ∈ period, d < 10

/-- The length of the period of the decimal representation -/
def period_length (d : PeriodicDecimal) : ℕ := d.period.length

/-- The arithmetic mean of the digits in the period -/
def period_mean (d : PeriodicDecimal) : ℚ :=
  (d.period.sum : ℚ) / d.period.length

theorem periodic_decimal_mean (d : PeriodicDecimal) :
  Even (period_length d) → period_mean d = 9/2 ∧
  Odd (period_length d) → period_mean d ≠ 9/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_decimal_mean_l504_50401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l504_50466

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  sum_angles : A + B + C = Real.pi
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

-- Define the main theorem
theorem triangle_property (t : Triangle) 
  (h : (2 * t.a - t.c) * Real.cos t.B = t.b * Real.cos t.C) : 
  t.B = Real.pi / 3 ∧ 
  (∀ A : Real, A ∈ Set.Ioo 0 (2 * Real.pi / 3) → -Real.sin A + 1 ≥ 0) ∧
  (∃ A : Real, A ∈ Set.Ioo 0 (2 * Real.pi / 3) ∧ -Real.sin A + 1 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l504_50466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_sale_price_l504_50451

/-- The sale price that results in a profit equal to the loss when sold for $448 -/
def S : ℝ := sorry

/-- The cost price of the article -/
def C : ℝ := sorry

/-- The profit earned when the article is sold for S -/
def profit : ℝ := S - C

/-- The loss incurred when the article is sold for $448 -/
def loss : ℝ := C - 448

/-- The sale price for a 50% profit -/
def sale_price_50_percent : ℝ := 975

theorem article_sale_price :
  (profit = loss) →
  (sale_price_50_percent = C + 0.5 * C) →
  S = 852 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_sale_price_l504_50451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_harmonic_mean_of_5_10_20_l504_50405

noncomputable def harmonic_mean (a b c : ℝ) : ℝ := 3 / (1/a + 1/b + 1/c)

theorem square_harmonic_mean_of_5_10_20 :
  (harmonic_mean 5 10 20) ^ 2 = 3600 / 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_harmonic_mean_of_5_10_20_l504_50405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l504_50476

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  16 * x^2 + 64 * x - 4 * y^2 + 8 * y + 36 = 0

/-- The distance between the vertices of the hyperbola -/
noncomputable def vertex_distance : ℝ := Real.sqrt 6

theorem hyperbola_vertex_distance :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola_equation x₁ y₁ ∧
    hyperbola_equation x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = vertex_distance^2 ∧
    ∀ (x y : ℝ), hyperbola_equation x y →
      (x - x₁)^2 + (y - y₁)^2 ≤ vertex_distance^2 ∧
      (x - x₂)^2 + (y - y₂)^2 ≤ vertex_distance^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l504_50476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focal_radius_points_l504_50490

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola -/
structure Parabola where
  focus : Point
  directrix : Line

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Distance from a point to a line -/
noncomputable def distanceToLine (p : Point) (l : Line) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Theorem: There are exactly two points on the focal radius (or its extension) of a parabola
    such that the difference of their distances to the focus and directrix is a given length -/
theorem parabola_focal_radius_points (p : Parabola) (l : ℝ) :
  ∃! (n1 n2 : Point),
    (∃ t : ℝ, n1 = Point.mk (p.focus.x + t * (p.focus.x - p.directrix.a)) (p.focus.y + t * (p.focus.y - p.directrix.b))) ∧
    (∃ t : ℝ, n2 = Point.mk (p.focus.x + t * (p.focus.x - p.directrix.a)) (p.focus.y + t * (p.focus.y - p.directrix.b))) ∧
    n1 ≠ n2 ∧
    abs (distance n1 p.focus - distanceToLine n1 p.directrix) = l ∧
    abs (distance n2 p.focus - distanceToLine n2 p.directrix) = l :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focal_radius_points_l504_50490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_point_two_neg_three_l504_50463

/-- For an angle α whose terminal side passes through the point (2, -3), sinα = -3√13 / 13 -/
theorem sin_alpha_point_two_neg_three :
  ∃ (α : ℝ), 
  let x : ℝ := 2
  let y : ℝ := -3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  Real.sin α = y / r :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_point_two_neg_three_l504_50463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_radius_eq_8_sec_half_phi_l504_50495

/-- The radius of the circle circumscribed about a sector with central angle φ cut from a circle of radius 8 -/
noncomputable def circumscribed_radius (φ : Real) : Real :=
  8 / Real.cos (φ / 2)

/-- Theorem: The radius of the circle circumscribed about a sector with central angle φ cut from a circle of radius 8 is equal to 8 sec(φ/2) -/
theorem circumscribed_radius_eq_8_sec_half_phi (φ : Real) (h : φ > 0 ∧ φ < Real.pi) :
  circumscribed_radius φ = 8 / Real.cos (φ / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_radius_eq_8_sec_half_phi_l504_50495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_implies_root_product_l504_50486

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- The product of roots of a quadratic equation ax^2 + bx + c = 0 -/
noncomputable def productOfRoots (a b c : ℝ) : ℝ := c / a

theorem segment_length_implies_root_product :
  ∀ a : ℝ, distance (3 * a) (a - 5) 7 0 = 3 * Real.sqrt 10 →
  productOfRoots 5 (-26) (-8) = -8/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_implies_root_product_l504_50486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_rate_theorem_combined_work_rate_positive_l504_50407

/-- The amount of work done by two people working together -/
noncomputable def combined_work_rate (a b : ℝ) : ℝ := 1/a + 1/b

/-- Theorem: The amount of work done by persons A and B working together in one day -/
theorem work_rate_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  combined_work_rate a b = 1/a + 1/b :=
by
  -- Unfold the definition of combined_work_rate
  unfold combined_work_rate
  -- The equality now holds by definition
  rfl

/-- Corollary: The combined work rate is positive when a and b are positive -/
theorem combined_work_rate_positive (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  combined_work_rate a b > 0 :=
by
  unfold combined_work_rate
  -- Use the fact that the sum of positive reals is positive
  exact add_pos (one_div_pos.mpr ha) (one_div_pos.mpr hb)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_rate_theorem_combined_work_rate_positive_l504_50407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lucas_lemonade_calories_l504_50435

/-- Represents the recipe and calorie information for lemonade --/
structure LemonadeInfo where
  lemon_juice_weight : ℚ
  sugar_weight : ℚ
  water_weight : ℚ
  lemon_juice_calories_per_100g : ℚ
  sugar_calories_per_100g : ℚ

/-- Calculates the calories in a given weight of lemonade --/
def calories_in_lemonade (info : LemonadeInfo) (weight : ℚ) : ℚ :=
  let total_weight := info.lemon_juice_weight + info.sugar_weight + info.water_weight
  let total_calories := (info.lemon_juice_weight / 100) * info.lemon_juice_calories_per_100g +
                        (info.sugar_weight / 100) * info.sugar_calories_per_100g
  (total_calories / total_weight) * weight

/-- The main theorem stating that 200g of Lucas's lemonade contains 145 calories --/
theorem lucas_lemonade_calories :
  let info : LemonadeInfo := {
    lemon_juice_weight := 200
    sugar_weight := 100
    water_weight := 300
    lemon_juice_calories_per_100g := 25
    sugar_calories_per_100g := 386
  }
  calories_in_lemonade info 200 = 145 := by
  sorry

#eval calories_in_lemonade {
  lemon_juice_weight := 200
  sugar_weight := 100
  water_weight := 300
  lemon_juice_calories_per_100g := 25
  sugar_calories_per_100g := 386
} 200

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lucas_lemonade_calories_l504_50435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domains_and_complement_l504_50417

-- Define the universal set U as ℝ
def U := ℝ

-- Define the functions
noncomputable def f₁ (x : ℝ) := Real.sqrt (x - 2) + Real.sqrt (x + 1)
noncomputable def f₂ (x : ℝ) := Real.sqrt (2 * x + 4) / (x - 3)

-- Define the domains A and B
def A : Set ℝ := {x | x ≥ 2}
def B : Set ℝ := {x | x ≥ -2 ∧ x ≠ 3}

-- Theorem to prove
theorem domains_and_complement :
  (A = {x : ℝ | x ≥ 2}) ∧
  (B = {x : ℝ | x ≥ -2 ∧ x ≠ 3}) ∧
  ((Aᶜ ∪ Bᶜ) = {x : ℝ | x < 2 ∨ x = 3}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domains_and_complement_l504_50417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_problem_l504_50477

/-- The minimum value of 1/a + 2/b given the conditions -/
theorem min_value_problem (m : ℝ) (a b : ℝ) : 
  m > 0 ∧ 
  (∃ (s : ℕ → ℝ), Monotone s ∧ StrictMono s ∧
    (∀ n, Real.sqrt 3 * Real.sin (s n) - Real.cos (s n) = m) ∧
    (∃ d, ∀ n, s (n + 1) - s n = d)) ∧
  a * 1 + b * m = 2 ∧
  a > 0 ∧ b > 0 →
  (1 / a + 2 / b) ≥ 9 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_problem_l504_50477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_equivalence_l504_50479

theorem problem_equivalence :
  (- (1 : ℝ) ^ 2022 - |2 - Real.sqrt 2| - ((-8) ^ (1/3 : ℝ)) = Real.sqrt 2 - 1) ∧
  (Real.sqrt ((-2 : ℝ) ^ 2) + |1 - Real.sqrt 2| - (2 * Real.sqrt 2 - 1) = 2 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_equivalence_l504_50479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_solutions_count_l504_50493

theorem positive_integer_solutions_count : 
  let solution_set := {p : ℕ × ℕ | 4 * p.1 + 7 * p.2 = 2001}
  Finset.card (Finset.filter (λ p => p ∈ solution_set) (Finset.product (Finset.range 501) (Finset.range 287))) = 71 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_solutions_count_l504_50493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_desk_chair_purchasing_problem_l504_50414

def total_sets : ℕ := 200

def cost_difference : ℕ := 40

def sample_cost : ℕ := 1640

def max_total_cost : ℕ := 40880

def type_a_cost : ℕ := 180

def type_b_cost : ℕ := 220

def valid_plans : ℕ := 3

def min_expenditure : ℕ := 40800

theorem desk_chair_purchasing_problem 
  (h1 : 3 * type_a_cost + 5 * type_b_cost = sample_cost)
  (h2 : type_b_cost = type_a_cost + cost_difference)
  (h3 : ∀ x : ℕ, x ≤ total_sets → 
    (x * type_a_cost + (total_sets - x) * type_b_cost ≤ max_total_cost ∧
     x ≤ 2 * (total_sets - x) / 3) →
    x = 78 ∨ x = 79 ∨ x = 80) :
  type_a_cost = 180 ∧ 
  type_b_cost = 220 ∧ 
  valid_plans = 3 ∧
  min_expenditure = 40800 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_desk_chair_purchasing_problem_l504_50414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_set_integer_distances_collinear_l504_50434

-- Define a type for points in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a type for lines in a plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to calculate the distance from a point to a line
noncomputable def distancePointToLine (p : Point) (l : Line) : ℝ :=
  (abs (l.a * p.x + l.b * p.y + l.c)) / Real.sqrt (l.a^2 + l.b^2)

-- Define a function to check if three points are collinear
def areCollinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

-- Define the theorem
theorem infinite_set_integer_distances_collinear (S : Set Point) :
  (∀ A B C : Point, A ∈ S → B ∈ S → C ∈ S → 
    ∃ n : ℤ, (distancePointToLine A (Line.mk (B.y - C.y) (C.x - B.x) (B.x * C.y - C.x * B.y)) = n)) →
  Set.Infinite S →
  ∀ p1 p2 p3, p1 ∈ S → p2 ∈ S → p3 ∈ S → areCollinear p1 p2 p3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_set_integer_distances_collinear_l504_50434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_m_n_congruence_l504_50438

/-- The set S(m,n) of coprime pairs within given bounds -/
def S (m n : ℕ) : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 ≥ 1 ∧ p.1 ≤ m ∧ p.2 ≥ 1 ∧ p.2 ≤ n ∧ Nat.gcd p.1 p.2 = 1)
    (Finset.product (Finset.range m) (Finset.range n))

/-- Main theorem: For any d and r, there exist m and n such that |S(m,n)| ≡ r (mod d) -/
theorem exists_m_n_congruence (d r : ℕ) (hd : d > 0) (hr : r > 0) :
  ∃ m n : ℕ, m ≥ d ∧ n ≥ d ∧ (Finset.card (S m n)) % d = r % d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_m_n_congruence_l504_50438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_multiple_intersections_l504_50420

-- Define a curve as a function from ℝ to ℝ
noncomputable def Curve := ℝ → ℝ

-- Define a tangent line at a point x₀ for a given curve f
noncomputable def TangentLine (f : Curve) (x₀ : ℝ) : ℝ → ℝ := 
  λ x => f x₀ + (deriv f x₀) * (x - x₀)

-- Theorem: There exists a curve and a point such that the tangent line at that point
-- intersects the curve at more than one point
theorem tangent_line_multiple_intersections :
  ∃ (f : Curve) (x₀ x₁ : ℝ), x₀ ≠ x₁ ∧ f x₁ = TangentLine f x₀ x₁ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_multiple_intersections_l504_50420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_investment_is_10000_l504_50440

/-- Calculates the initial investment given the interest earned, time period, and annual interest rate --/
noncomputable def calculate_initial_investment (interest : ℝ) (months : ℝ) (annual_rate : ℝ) : ℝ :=
  interest / (annual_rate * (months / 12))

/-- Theorem stating that given the specified conditions, the initial investment is $10,000 --/
theorem initial_investment_is_10000 :
  let interest : ℝ := 300
  let months : ℝ := 9
  let annual_rate : ℝ := 0.04
  calculate_initial_investment interest months annual_rate = 10000 := by
  -- Proof steps would go here
  sorry

-- Remove the #eval statement as it's not compatible with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_investment_is_10000_l504_50440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lucas_fib_relation_lucas_fib_determinant_main_theorem_l504_50411

def lucas : ℕ → ℤ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

def matrix_power (n : ℕ) : Matrix (Fin 2) (Fin 2) ℤ :=
  (Matrix.of !![1, 1; 1, 0]) ^ n

theorem lucas_fib_relation (n : ℕ) :
  matrix_power n = Matrix.of !![lucas (n + 1), fib n; fib n, lucas n] :=
  sorry

theorem lucas_fib_determinant (n : ℕ) :
  lucas (n + 1) * lucas n - fib n * fib n = (-1)^n :=
  sorry

theorem main_theorem :
  lucas 785 * lucas 783 - fib 784 * fib 784 = 1 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lucas_fib_relation_lucas_fib_determinant_main_theorem_l504_50411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balanced_subset_l504_50449

def isBalanced (S : Finset ℕ) : Prop :=
  ∀ a ∈ S, ∃ b ∈ S, b ≠ a ∧ (a + b) / 2 ∈ S

theorem balanced_subset (k : ℕ) (h_k : k > 1) :
  let n := 2^k
  ∀ S : Finset ℕ, S ⊆ Finset.range n.succ →
    S.card > (3 * n) / 4 →
    isBalanced S :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balanced_subset_l504_50449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l504_50470

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := (a * x^2 + x + a) / (2 * Real.exp x)

-- Part I
theorem part_one (a : ℝ) (h1 : a ≥ 0) (h2 : ∀ x : ℝ, f a x ≤ 5 / (2 * Real.exp 1)) 
  (h3 : ∃ x : ℝ, f a x = 5 / (2 * Real.exp 1)) : a = 2 := by
  sorry

-- Part II
theorem part_two (b : ℝ) 
  (h : ∀ a x : ℝ, a ≤ 0 → x ≥ 0 → f a x ≤ b * Real.log (x + 1) / 2) : b ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l504_50470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_with_greater_than_l504_50426

theorem inequalities_with_greater_than (a b : ℝ) (h : a > b) : a^3 > b^3 ∧ (2:ℝ)^a > (2:ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_with_greater_than_l504_50426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_integral_value_l504_50459

/-- The distance function d(t) from the origin to the line (ae^t)x + (be^(-t))y = 1 -/
noncomputable def d (a b t : ℝ) : ℝ := 1 / Real.sqrt (a^2 * Real.exp (2*t) + b^2 * Real.exp (-2*t))

/-- The integral we want to minimize -/
noncomputable def I (a b : ℝ) : ℝ := ∫ t in Set.Icc 0 1, (1 / d a b t^2)

theorem min_integral_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  ∃ (min_val : ℝ), min_val = Real.exp 1 - Real.exp (-1) ∧ 
  ∀ (a' b' : ℝ), a' > 0 → b' > 0 → a' * b' = 1 → I a' b' ≥ min_val := by
  sorry

#check min_integral_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_integral_value_l504_50459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l504_50484

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2*x - 1) / (x + 1)

-- Define the domain
def domain : Set ℝ := { x | 3 ≤ x ∧ x ≤ 5 }

-- Theorem statement
theorem f_properties :
  (∀ x ∈ domain, ∀ y ∈ domain, x < y → f x < f y) ∧ 
  (∀ x ∈ domain, f x ≥ 5/4) ∧
  (∀ x ∈ domain, f x ≤ 3/2) ∧
  (∃ x ∈ domain, f x = 5/4) ∧
  (∃ x ∈ domain, f x = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l504_50484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_when_y_is_10_l504_50412

-- Define the constant ratio
noncomputable def constant_ratio (x y : ℝ) : ℝ := (2 * x - 3) / (2 * y + 10)

-- State the theorem
theorem x_value_when_y_is_10 :
  ∀ (x₀ y₀ x₁ y₁ : ℝ),
  (x₀ = 4 ∧ y₀ = 5) →  -- Given condition
  (y₁ = 10) →  -- New condition
  (constant_ratio x₀ y₀ = constant_ratio x₁ y₁) →  -- Constant ratio property
  x₁ = 5.25 :=
by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_when_y_is_10_l504_50412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_is_25_l504_50478

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The square defined by its vertices -/
structure Square where
  E : Point
  F : Point
  G : Point
  H : Point

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the area of a square given its side length -/
def squareArea (side : ℝ) : ℝ :=
  side^2

/-- Theorem: The area of the given square is 25 square units -/
theorem square_area_is_25 (s : Square) 
    (h1 : s.E = ⟨1, 2⟩) 
    (h2 : s.F = ⟨1, -3⟩) 
    (h3 : s.G = ⟨-4, -3⟩) 
    (h4 : s.H = ⟨-4, 2⟩) : 
    squareArea (distance s.E s.F) = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_is_25_l504_50478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_mult_chain_l504_50436

/-- Custom multiplication operation -/
def star_mult (a b : ℚ) : ℚ := (a - b) / (1 - a * b)

/-- Recursive application of star_mult -/
def recursive_star_mult : ℕ → ℚ
  | 0 => 1000  -- Base case for 0
  | 1 => 1000  -- Base case for 1
  | n + 1 => star_mult (n : ℚ) (recursive_star_mult n)

/-- Main theorem statement -/
theorem star_mult_chain : star_mult 2 (recursive_star_mult 3) = 2001 / 1995 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_mult_chain_l504_50436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_range_l504_50473

-- Define the function f(x) = 2/x - x
noncomputable def f (x : ℝ) : ℝ := 2/x - x

-- Define the theorem
theorem inequality_solution_range :
  ∀ a : ℝ, (∃ x : ℝ, x ∈ Set.Icc 1 4 ∧ x^2 + a*x - 2 < 0) ↔ a ∈ Set.Iio 1 := by
  sorry

-- Define a lemma for the maximum value of f(x) in [1,4]
lemma f_max_in_interval :
  ∀ x : ℝ, x ∈ Set.Icc 1 4 → f x ≤ f 1 := by
  sorry

-- Define a lemma for the monotonicity of f(x) in [1,4]
lemma f_decreasing_in_interval :
  ∀ x y : ℝ, x ∈ Set.Icc 1 4 → y ∈ Set.Icc 1 4 → x < y → f x > f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_range_l504_50473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rebus_puzzle_solution_exists_and_is_unique_l504_50485

theorem rebus_puzzle_solution_exists_and_is_unique :
  ∃! (a b c d f g h k l m n p q r s t u v w x y z : ℕ),
    a * 10 + b = 110 ∧
    c = 0 ∧
    d = 1 ∧
    f = 1 ∧
    60 + g = 68 ∧
    10 + h = 17 ∧
    k = 4 ∧
    l = 1 ∧
    m = 0 ∧
    n = 1 ∧
    p = 0 ∧
    q = 1 ∧
    r = 4 ∧
    10 + s = 12 ∧
    t = 1 ∧
    u = 0 ∧
    v = 1 ∧
    w = 4 ∧
    x = 1 ∧
    100 + y * 10 + z = 142 ∧
    a + 6 + l + q + t = 11 ∧
    1 + g + m + 2 + u = 11 ∧
    b + h + n + r + v = 14 ∧
    c + 7 + p + s + w = 20 ∧
    f + k + 0 + 8 + x = 14 ∧
    0 + 0 + 0 + 0 + z = 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rebus_puzzle_solution_exists_and_is_unique_l504_50485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fortieth_number_is_eighteen_l504_50432

/-- Represents the sequence described in the problem -/
def sequenceValue (n : ℕ) : ℕ := 3 * n

/-- Calculates the cumulative count of numbers up to a given row -/
def cumulativeCount (row : ℕ) : ℕ :=
  (row * (row + 1))

/-- Finds the row containing the kth number -/
def findRow (k : ℕ) : ℕ :=
  match k with
  | 0 => 0
  | k + 1 => if cumulativeCount k < k + 1 then k + 1 else findRow k

/-- The 40th number in the sequence is 18 -/
theorem fortieth_number_is_eighteen : sequenceValue (findRow 40) = 18 := by
  sorry

#eval sequenceValue (findRow 40)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fortieth_number_is_eighteen_l504_50432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_proof_l504_50498

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = 2 * x - 20

-- Define the function g
def g (t : ℝ) : ℝ := 10 * t + 7.5

-- Define the parameterization
def parameterization (t : ℝ) : ℝ × ℝ := (g t, 20 * t - 5)

-- Theorem statement
theorem line_parameterization_proof :
  ∀ t : ℝ, line_equation (g t) (20 * t - 5) ∧ parameterization t = (g t, 20 * t - 5) := by
  intro t
  constructor
  · -- Prove that the point (g t, 20 * t - 5) satisfies the line equation
    calc
      20 * t - 5 = 2 * (10 * t + 7.5) - 20 := by ring
      _ = 2 * (g t) - 20 := by rfl
  · -- Prove that parameterization t equals (g t, 20 * t - 5)
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_proof_l504_50498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_three_iff_parallel_not_coincident_l504_50488

noncomputable section

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- Two lines are coincident if they have the same slope and y-intercept -/
def are_coincident (m1 b1 m2 b2 : ℝ) : Prop := m1 = m2 ∧ b1 = b2

/-- The slope-intercept form of a line ax + by + c = 0 is y = mx + k where m = -a/b and k = -c/b -/
def slope_intercept (a b c : ℝ) : ℝ × ℝ := (-a/b, -c/b)

/-- The condition for the given lines to be parallel and not coincident -/
def parallel_not_coincident (a : ℝ) : Prop :=
  let (m1, b1) := slope_intercept a 1 (3*a)
  let (m2, b2) := slope_intercept 3 (a-2) (a-8)
  are_parallel m1 m2 ∧ ¬are_coincident m1 b1 m2 b2

/-- The theorem to be proved -/
theorem a_equals_three_iff_parallel_not_coincident :
  ∀ a : ℝ, a = 3 ↔ parallel_not_coincident a := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_three_iff_parallel_not_coincident_l504_50488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_property_l504_50416

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := Real.sqrt 5 / x
def g (x : ℝ) : ℝ := x + 5

-- Define variables for the intersection point
variable (a b : ℝ)

-- Define the intersection point
def P : ℝ × ℝ := (a, b)

-- State the theorem
theorem intersection_point_property :
  (∀ x > 0, f x = g x → x = a) →  -- Intersection condition
  b = f a →                       -- Point satisfies first function
  b = g a →                       -- Point satisfies second function
  1 / a - 1 / b = Real.sqrt 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_property_l504_50416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_succ_beta_l504_50447

-- Define the type for the polynomials T
def T (α : ℕ) : ℝ → ℝ → ℝ → ℝ := sorry

-- Define the ordering relation ≻
def succ (α β : ℕ) : Prop := sorry

-- State the theorem
theorem alpha_succ_beta (α β : ℕ) 
  (h : ∀ (x y z : ℝ), x ≥ 0 → y ≥ 0 → z ≥ 0 → T α x y z ≥ T β x y z) : 
  succ α β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_succ_beta_l504_50447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_polygon_long_side_l504_50471

/-- Represents a grid polygon -/
structure GridPolygon where
  area : ℕ
  perimeter : ℕ
  sides : Set ℝ
  no_holes : Prop

/-- Predicate to check if a side length is greater than 1 -/
def has_side_longer_than_one (p : GridPolygon) : Prop :=
  ∃ (side : ℝ), side ∈ p.sides ∧ side > 1

/-- Theorem stating that a grid polygon with area 300 and perimeter 300 has a side longer than 1 -/
theorem grid_polygon_long_side (p : GridPolygon) 
  (h_area : p.area = 300) 
  (h_perimeter : p.perimeter = 300) 
  (h_no_holes : p.no_holes) : 
  has_side_longer_than_one p :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_polygon_long_side_l504_50471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_relationship_l504_50413

/-- Given a job and workers A, B, and C, prove the relationship between their working times. -/
theorem job_completion_time_relationship (m n : ℝ) (h_mn : m * n ≠ 1) : 
  ∃ x : ℝ, ∀ a b c : ℝ, 
    (3 * 40 * a = m * (1 / b + 1 / c) ∧ 
     1 / b = n * (1 / a + 1 / c) ∧ 
     1 / c = x * (1 / a + 1 / b)) →
    x = (m + n + 2) / (m * n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_relationship_l504_50413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_a_range_l504_50460

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Theorem for the minimum value of f(x)
theorem f_minimum_value : ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≥ f x ∧ f x = -1 / Real.exp 1 := by sorry

-- Theorem for the range of a
theorem a_range (a : ℝ) : (∀ (x : ℝ), x ≥ 1 → f x ≥ a * x - 1) ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_a_range_l504_50460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_face_is_E_l504_50439

-- Define the faces of the cube
inductive Face : Type
| X | A | B | C | D | E

-- Define the structure of the cube net
structure CubeNet where
  faces : List Face
  adjacent_to_x : List Face
  top_face : Face

-- Define the folding instructions
structure FoldingInstructions where
  upward_from_x : Face
  rightward_from_x : Face

-- Helper function to determine the opposite face (not implemented)
def opposite_face (net : CubeNet) (face : Face) : Face :=
  sorry

-- Define the theorem
theorem opposite_face_is_E (net : CubeNet) (fold : FoldingInstructions) : 
  net.faces = [Face.X, Face.A, Face.B, Face.C, Face.D, Face.E] →
  net.adjacent_to_x = [Face.A, Face.B] →
  net.top_face = Face.C →
  fold.upward_from_x = Face.A →
  fold.rightward_from_x = Face.B →
  (opposite_face net Face.X) = Face.E :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_face_is_E_l504_50439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_C_line_l_l504_50487

-- Define curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos α, Real.sin α)

-- Define line l
def line_l (t : ℝ) : ℝ × ℝ := (t, 3 - t)

-- Define the distance function between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem min_distance_curve_C_line_l :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 2 / 2 ∧
  ∀ (α t : ℝ), distance (curve_C α) (line_l t) ≥ min_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_C_line_l_l504_50487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_directional_vector_l504_50433

-- Define the two curves
noncomputable def curve1 (x : ℝ) : ℝ := Real.exp x - 1
noncomputable def curve2 (x : ℝ) : ℝ := Real.log (x + 1)

-- Define the derivative of the curves
noncomputable def curve1_derivative (x : ℝ) : ℝ := Real.exp x
noncomputable def curve2_derivative (x : ℝ) : ℝ := 1 / (x + 1)

-- Define the common tangent line
def common_tangent (x : ℝ) : ℝ := x

-- Theorem statement
theorem common_tangent_directional_vector :
  ∃ (s t : ℝ),
    -- The tangent line touches both curves
    curve1 s = common_tangent s ∧
    curve2 t = common_tangent t ∧
    -- The slopes of the curves at the tangent points are equal to the slope of the common tangent
    curve1_derivative s = 1 ∧
    curve2_derivative t = 1 ∧
    -- (1, 1) is a directional vector of the common tangent
    ∀ (x : ℝ), common_tangent (x + 1) = common_tangent x + 1 := by
  sorry

#check common_tangent_directional_vector

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_directional_vector_l504_50433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_OCD_l504_50421

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := y^2 / 8 + x^2 / 2 = 1

/-- Point on the ellipse -/
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse x y

/-- Upper vertex of the ellipse -/
noncomputable def A : ℝ × ℝ := (0, 2 * Real.sqrt 2)

/-- Lower vertex of the ellipse -/
noncomputable def B : ℝ × ℝ := (0, -2 * Real.sqrt 2)

/-- The x-coordinate of the intersection line -/
def intersection_x : ℝ := -2

/-- The origin -/
def O : ℝ × ℝ := (0, 0)

/-- Calculate the area of triangle OCD given a point P on the ellipse -/
noncomputable def triangle_area (P : PointOnEllipse) : ℝ :=
  sorry

/-- The minimum area of triangle OCD -/
noncomputable def min_triangle_area : ℝ := 8 - 4 * Real.sqrt 2

/-- The main theorem: the minimum area of triangle OCD is 8 - 4√2 -/
theorem min_area_triangle_OCD :
  ∃ (P : PointOnEllipse), ∀ (Q : PointOnEllipse), triangle_area P ≤ triangle_area Q ∧ triangle_area P = min_triangle_area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_OCD_l504_50421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l504_50452

noncomputable def f (x : ℝ) : ℝ := |x - 4| + |x - 1|

noncomputable def g (x m : ℝ) : ℝ := (2017 * x - 2016) / (f x + 2 * m)

theorem problem_solution :
  (∀ x : ℝ, f x ≤ 5 ↔ 0 ≤ x ∧ x ≤ 5) ∧
  (∀ m : ℝ, (∀ x : ℝ, g x m ≠ 0) → m > -3/2) :=
by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l504_50452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_imaginary_roots_l504_50497

theorem quadratic_imaginary_roots (l : ℝ) : 
  (∀ x : ℂ, (1 - Complex.I) * x^2 + (l + Complex.I) * x + (1 + Complex.I * l) = 0 → x.im ≠ 0) ↔ 
  l ≠ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_imaginary_roots_l504_50497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l504_50410

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- Represents a point on a hyperbola -/
structure PointOnHyperbola (h : Hyperbola) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / h.a^2 - y^2 / h.b^2 = 1
  right_branch : x > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- Theorem stating the range of eccentricity for a hyperbola with specific conditions -/
theorem eccentricity_range (h : Hyperbola) (p : PointOnHyperbola h) 
  (hf : ∃ (xf1 yf1 xf2 yf2 : ℝ), 
    distance p.x p.y xf1 yf1 = 4 * distance p.x p.y xf2 yf2) : 
  1 < eccentricity h ∧ eccentricity h ≤ 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l504_50410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_areas_sum_eq_9pi_div_8_l504_50408

open Real

/-- The sum of the areas of an infinite sequence of circles -/
noncomputable def circle_areas_sum : ℝ :=
  let radius (n : ℕ) : ℝ := (1 : ℝ) / (3 ^ n)
  let area (n : ℕ) : ℝ := π * (radius n) ^ 2
  ∑' n, area n

/-- Theorem: The sum of the areas of an infinite sequence of circles,
    where the radius of each circle is 1/3 of the previous one and
    the first circle has a radius of 1 inch, is equal to 9π/8 square inches. -/
theorem circle_areas_sum_eq_9pi_div_8 :
  circle_areas_sum = 9 * π / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_areas_sum_eq_9pi_div_8_l504_50408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_distance_l504_50446

/-- The distance from Anglchester to Klinkerton in miles -/
noncomputable def distance : ℝ := 200

/-- The initial speed of the train in miles per hour -/
noncomputable def initial_speed : ℝ := 50

/-- The time in hours before the malfunction occurred -/
noncomputable def time_before_malfunction : ℝ := 1

/-- The factor by which the speed was reduced after the malfunction -/
noncomputable def speed_reduction_factor : ℝ := 3/5

/-- The delay in hours caused by the malfunction -/
noncomputable def delay : ℝ := 2

/-- The distance in miles that would have resulted in an earlier arrival if the malfunction occurred later -/
noncomputable def additional_distance : ℝ := 50

/-- The time in hours by which the train would have arrived earlier if the malfunction occurred later -/
noncomputable def earlier_arrival : ℝ := 2/3

theorem train_journey_distance :
  distance = initial_speed * 4 ∧
  time_before_malfunction * initial_speed + 
    (distance - time_before_malfunction * initial_speed) / (speed_reduction_factor * initial_speed) = 
    distance / initial_speed + delay ∧
  time_before_malfunction + additional_distance / initial_speed + 
    (distance - additional_distance) / (speed_reduction_factor * initial_speed) = 
    distance / initial_speed + delay - earlier_arrival :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_distance_l504_50446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_diameter_approx_wheel_diameter_satisfies_conditions_l504_50492

/-- The diameter of a wheel that covers 1320 cm in 15.013648771610555 revolutions --/
noncomputable def wheel_diameter : ℝ :=
  1320 / (15.013648771610555 * Real.pi)

/-- Theorem stating that the wheel diameter is approximately 28.01 cm --/
theorem wheel_diameter_approx :
  abs (wheel_diameter - 28.01) < 0.01 := by sorry

/-- Function to calculate the circumference of a circle given its diameter --/
noncomputable def circumference (d : ℝ) : ℝ := Real.pi * d

/-- Function to calculate the total distance covered by a wheel given number of revolutions and circumference --/
def total_distance (n : ℝ) (c : ℝ) : ℝ := n * c

/-- Theorem proving that the calculated wheel diameter satisfies the given conditions --/
theorem wheel_diameter_satisfies_conditions :
  abs (total_distance 15.013648771610555 (circumference wheel_diameter) - 1320) < 0.000001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_diameter_approx_wheel_diameter_satisfies_conditions_l504_50492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_40π_l504_50423

-- Define the circle ω
def ω : Set (ℝ × ℝ) := sorry

-- Define points A and B
def A : ℝ × ℝ := (4, 15)
def B : ℝ × ℝ := (14, 9)

-- Axiom: A and B lie on circle ω
axiom A_on_ω : A ∈ ω
axiom B_on_ω : B ∈ ω

-- Define what it means for a line to be tangent to a circle at a point
def IsTangentLine (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop := sorry

-- Axiom: Tangent lines at A and B intersect on x-axis
axiom tangent_intersection_on_x_axis : 
  ∃ C : ℝ × ℝ, C.2 = 0 ∧ 
  (∃ l₁ l₂ : Set (ℝ × ℝ), IsTangentLine l₁ ω A ∧ IsTangentLine l₂ ω B ∧ C ∈ l₁ ∩ l₂)

-- Define the area of a circle
noncomputable def circle_area (c : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem to prove
theorem circle_area_is_40π : circle_area ω = 40 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_40π_l504_50423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_minimum_distance_l504_50422

/-- The parabola defined by y^2 = 5x -/
def parabola (x y : ℝ) : Prop := y^2 = 5*x

/-- The focus of the parabola -/
noncomputable def focus : ℝ × ℝ := (5/4, 0)

/-- Point A -/
def point_A : ℝ × ℝ := (3, 1)

/-- A point on the parabola -/
def point_on_parabola (p : ℝ × ℝ) : Prop :=
  parabola p.1 p.2

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The sum of distances from a point to A and F -/
noncomputable def sum_of_distances (p : ℝ × ℝ) : ℝ :=
  distance p point_A + distance p focus

/-- The statement to be proved -/
theorem parabola_minimum_distance :
  let M : ℝ × ℝ := (1/5, 1)
  point_on_parabola M ∧
  ∀ p : ℝ × ℝ, point_on_parabola p → sum_of_distances M ≤ sum_of_distances p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_minimum_distance_l504_50422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_children_on_airplane_l504_50437

theorem children_on_airplane (total_passengers : ℕ) 
  (adult_percentage : ℚ) (h1 : total_passengers = 240) 
  (h2 : adult_percentage = 60 / 100) : 
  (1 - adult_percentage) * total_passengers = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_children_on_airplane_l504_50437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_decrease_percentage_l504_50475

/-- Calculates the percentage decrease between two values -/
noncomputable def percentageDecrease (oldValue newValue : ℝ) : ℝ :=
  ((oldValue - newValue) / oldValue) * 100

theorem revenue_decrease_percentage :
  let oldRevenue : ℝ := 69.0
  let newRevenue : ℝ := 48.0
  abs (percentageDecrease oldRevenue newRevenue - 30.43) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_decrease_percentage_l504_50475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_y_coordinate_sum_abcd_is_15_l504_50467

-- Define the points
def A : ℝ × ℝ := (-4, 1)
def B : ℝ × ℝ := (-1, 2)
def C : ℝ × ℝ := (1, 2)
def D : ℝ × ℝ := (4, 1)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem point_P_y_coordinate :
  ∃ P : ℝ × ℝ,
  (distance P A + distance P D = 10) ∧
  (distance P B + distance P C = 10) ∧
  P.2 = (-2 + 2 * Real.sqrt 6) / 5 :=
by sorry

-- Define the sum of a, b, c, and d
def sum_abcd : ℕ := 2 + 2 + 6 + 5

-- Prove that the sum is correct
theorem sum_abcd_is_15 : sum_abcd = 15 :=
by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_y_coordinate_sum_abcd_is_15_l504_50467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_needs_all_as_l504_50453

/-- Represents Lisa's quiz performance and goal --/
structure QuizPerformance where
  total_quizzes : ℕ
  goal_percentage : ℚ
  completed_quizzes : ℕ
  completed_as : ℕ

/-- Calculates the maximum number of remaining quizzes where Lisa can earn a grade lower than an A --/
def max_non_a_quizzes (performance : QuizPerformance) : ℕ :=
  let remaining_quizzes := performance.total_quizzes - performance.completed_quizzes
  let required_as := Int.ceil (performance.goal_percentage * performance.total_quizzes)
  let additional_as_needed := required_as - performance.completed_as
  (remaining_quizzes - additional_as_needed).toNat.max 0

/-- Theorem stating that Lisa can't afford any non-A grades on remaining quizzes --/
theorem lisa_needs_all_as (performance : QuizPerformance) 
    (h_total : performance.total_quizzes = 60)
    (h_goal : performance.goal_percentage = 85 / 100)
    (h_completed : performance.completed_quizzes = 40)
    (h_completed_as : performance.completed_as = 30) :
    max_non_a_quizzes performance = 0 := by
  sorry

#eval max_non_a_quizzes { total_quizzes := 60, goal_percentage := 85/100, completed_quizzes := 40, completed_as := 30 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_needs_all_as_l504_50453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l504_50428

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define points D, E, F on the sides of the triangle
noncomputable def D : ℝ × ℝ := (2/5 * B.1 + 3/5 * C.1, 2/5 * B.2 + 3/5 * C.2)
noncomputable def E : ℝ × ℝ := (2/5 * A.1 + 3/5 * C.1, 2/5 * A.2 + 3/5 * C.2)
noncomputable def F : ℝ × ℝ := (3/5 * A.1 + 2/5 * B.1, 3/5 * A.2 + 2/5 * B.2)

-- Define the intersections P, Q, R
noncomputable def P : ℝ × ℝ := sorry
noncomputable def Q : ℝ × ℝ := sorry
noncomputable def R : ℝ × ℝ := sorry

-- Define the area function
noncomputable def area (p q r : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_ratio : 
  area P Q R / area A B C = 6/25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l504_50428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_composition_l504_50474

-- Define the real logarithm with base 3
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- Define the function g first
noncomputable def g (x : ℝ) : ℝ := -log3 (-x + 1)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then log3 (x + 1) else g x

-- State the theorem
theorem odd_function_composition :
  (∀ x, f (-x) = -f x) →  -- f is an odd function
  g (f (-8)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_composition_l504_50474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_negative_numbers_l504_50424

theorem min_negative_numbers (a b c d : ℝ) (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (h_sum : (a + b + c < d) ∧ (a + b + d < c) ∧ (a + c + d < b) ∧ (b + c + d < a)) :
  (Bool.toNat (a < 0) + Bool.toNat (b < 0) + Bool.toNat (c < 0) + Bool.toNat (d < 0)) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_negative_numbers_l504_50424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simpsons_rule_x_squared_equals_21_simpsons_rule_exact_for_x_squared_l504_50494

/-- Simpson's Rule function for n subdivisions -/
noncomputable def simpsonsRule (f : ℝ → ℝ) (a b : ℝ) (n : ℕ) : ℝ :=
  let h := (b - a) / n
  let x := fun i => a + i * h
  let y := fun i => f (x i)
  (h / 3) * (y 0 + y n + 
    4 * (Finset.sum (Finset.range (n / 2)) (fun i => y (2 * i + 1))) +
    2 * (Finset.sum (Finset.range ((n - 2) / 2)) (fun i => y (2 * i + 2))))

/-- The function to be integrated: f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- Theorem: Simpson's Rule with 10 subdivisions for ∫₁⁴ x² dx equals 21 -/
theorem simpsons_rule_x_squared_equals_21 :
  simpsonsRule f 1 4 10 = 21 := by
  sorry  -- The proof is omitted as per the instructions

/-- The exact value of the integral ∫₁⁴ x² dx -/
def exact_integral : ℝ := 21

/-- Theorem: Simpson's Rule gives the exact result for ∫₁⁴ x² dx -/
theorem simpsons_rule_exact_for_x_squared :
  simpsonsRule f 1 4 10 = exact_integral := by
  sorry  -- The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simpsons_rule_x_squared_equals_21_simpsons_rule_exact_for_x_squared_l504_50494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_Q_on_x_axis_l504_50480

noncomputable section

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the line l
def l (x y : ℝ) : Prop := y = x - 1

-- Define the intersection points P₁ and P₂
def intersection_points (P₁ P₂ : ℝ × ℝ) : Prop :=
  C P₁.1 P₁.2 ∧ C P₂.1 P₂.2 ∧ l P₁.1 P₁.2 ∧ l P₂.1 P₂.2 ∧ P₁ ≠ P₂

-- Define the area of a triangle given three points
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

-- Main theorem
theorem exists_Q_on_x_axis (P₁ P₂ : ℝ × ℝ) (h_intersection : intersection_points P₁ P₂) :
  ∃ Q : ℝ × ℝ, Q.2 = 0 ∧ triangle_area P₁ P₂ Q = 6 * Real.sqrt 2 ∧ (Q.1 = 8 ∨ Q.1 = -6) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_Q_on_x_axis_l504_50480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parallel_lines_intersection_l504_50448

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define a line
def is_on_line (y k : ℝ) : Prop := y = k

-- Define the number of intersection points
noncomputable def num_intersections (k : ℝ) : ℕ :=
  if k^2 < 1 then 2 else 0

-- Theorem statement
theorem ellipse_parallel_lines_intersection :
  ∀ k₁ k₂ : ℝ, k₁ ≠ k₂ →  -- Two distinct parallel lines
  k₁ ≠ 1 ∧ k₁ ≠ -1 ∧ k₂ ≠ 1 ∧ k₂ ≠ -1 →  -- Lines are not tangents
  (num_intersections k₁ + num_intersections k₂) ∈ ({0, 2, 4} : Set ℕ) := by
  sorry

#check ellipse_parallel_lines_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parallel_lines_intersection_l504_50448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_secant_theorem_l504_50404

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define a function to check if a point is on a circle
def isOnCircle (p : Point) (c : Circle) : Prop :=
  distance p c.center = c.radius

-- Define a function to check if a point is outside a circle
def isOutsideCircle (p : Point) (c : Circle) : Prop :=
  distance p c.center > c.radius

-- Theorem statement
theorem tangent_secant_theorem 
  (c : Circle) (A M B C : Point) : 
  isOutsideCircle A c →
  isOnCircle M c →
  isOnCircle B c →
  isOnCircle C c →
  distance A M = c.radius →  -- AM is tangent
  distance A C < distance A B →  -- C is between A and B
  distance A C * distance A B = (distance A M)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_secant_theorem_l504_50404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_profit_calculation_l504_50462

noncomputable section

/-- Calculate the final selling price after markup, discount, and tax -/
def finalSellingPrice (cost markup discount tax : ℝ) : ℝ :=
  let priceAfterMarkup := cost * (1 + markup)
  let priceAfterDiscount := priceAfterMarkup * (1 - discount)
  priceAfterDiscount * (1 + tax)

/-- Calculate the total revenue for a given number of units -/
def totalRevenue (units : ℕ) (finalPrice : ℝ) : ℝ :=
  (units : ℝ) * finalPrice

/-- Calculate the total cost for a given number of units -/
def totalCost (units : ℕ) (cost : ℝ) : ℝ :=
  (units : ℝ) * cost

/-- Calculate the percentage profit -/
def percentageProfit (revenue cost : ℝ) : ℝ :=
  (revenue - cost) / cost * 100

theorem merchant_profit_calculation :
  let cost1 : ℝ := 30
  let markup1 : ℝ := 0.45
  let discount1 : ℝ := 0.15
  let units1 : ℕ := 100
  let cost2 : ℝ := 60
  let markup2 : ℝ := 0.75
  let discount2 : ℝ := 0.30
  let units2 : ℕ := 50
  let tax : ℝ := 0.08

  let finalPrice1 := finalSellingPrice cost1 markup1 discount1 tax
  let finalPrice2 := finalSellingPrice cost2 markup2 discount2 tax
  let revenue1 := totalRevenue units1 finalPrice1
  let revenue2 := totalRevenue units2 finalPrice2
  let totalRevenue := revenue1 + revenue2
  let totalCost := totalCost units1 cost1 + totalCost units2 cost2
  let profit := percentageProfit totalRevenue totalCost

  abs (profit - 32.71) < 0.01 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_profit_calculation_l504_50462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_interest_rate_l504_50425

/-- The bank profit model -/
structure BankModel where
  k : ℝ
  k_pos : k > 0

/-- The bank's profit function -/
def profit (b : BankModel) (x : ℝ) : ℝ :=
  0.072 * b.k * x^2 - b.k * x^3

/-- The derivative of the profit function -/
def profit_derivative (b : BankModel) (x : ℝ) : ℝ :=
  3 * b.k * x * (0.048 - x)

/-- Theorem: The optimal interest rate for depositors is 4.8% -/
theorem optimal_interest_rate (b : BankModel) :
  ∃ (x : ℝ), x > 0 ∧ x = 0.048 ∧
  ∀ (y : ℝ), y > 0 → profit b x ≥ profit b y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_interest_rate_l504_50425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_90_l504_50429

-- Define the profit function
noncomputable def profit (x : ℝ) : ℝ :=
  if x < 80 then
    -0.5 * x^2 + 60 * x - 5
  else
    1680 - (x + 8100 / x)

-- Theorem statement
theorem max_profit_at_90 :
  ∃ (max_profit : ℝ), max_profit = 1500 ∧
  ∀ (x : ℝ), x > 0 → profit x ≤ max_profit ∧
  profit 90 = max_profit := by
  sorry

#check max_profit_at_90

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_90_l504_50429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_by_five_and_seven_l504_50491

theorem smallest_divisible_by_five_and_seven (x : ℕ) : 
  (∀ y : ℕ, y > 0 → y < 35 → ¬(5 ∣ 67 * 89 * y) ∨ ¬(7 ∣ 67 * 89 * y)) ∧
  (5 ∣ 67 * 89 * 35) ∧ 
  (7 ∣ 67 * 89 * 35) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_by_five_and_seven_l504_50491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_bisection_l504_50469

/-- Two circles in a plane -/
structure TwoCircles where
  Γ₁ : Set (ℝ × ℝ)
  Γ₂ : Set (ℝ × ℝ)

/-- The intersection points of two circles -/
def IntersectionPoints (tc : TwoCircles) : Set (ℝ × ℝ) :=
  tc.Γ₁ ∩ tc.Γ₂

/-- A common tangent to two circles -/
structure CommonTangent (tc : TwoCircles) where
  line : Set (ℝ × ℝ)
  touches_Γ₁ : line ∩ tc.Γ₁ ≠ ∅
  touches_Γ₂ : line ∩ tc.Γ₂ ≠ ∅

/-- Points of tangency for a common tangent -/
def TangencyPoints (tc : TwoCircles) (ct : CommonTangent tc) : Set (ℝ × ℝ) :=
  (ct.line ∩ tc.Γ₁) ∪ (ct.line ∩ tc.Γ₂)

/-- The midpoint of a line segment -/
noncomputable def Midpoint (a b : ℝ × ℝ) : ℝ × ℝ :=
  ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

/-- A line through two points -/
def LineThroughPoints (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {x : ℝ × ℝ | ∃ t : ℝ, x = (p.1 + t * (q.1 - p.1), p.2 + t * (q.2 - p.2))}

/-- The theorem to be proved -/
theorem tangent_bisection
  (tc : TwoCircles)
  (ct : CommonTangent tc)
  (C D : ℝ × ℝ)
  (A B : ℝ × ℝ)
  (h₁ : {C, D} ⊆ IntersectionPoints tc)
  (h₂ : {A, B} = TangencyPoints tc ct)
  : Midpoint A B ∈ LineThroughPoints C D :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_bisection_l504_50469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_slope_angle_l504_50489

/-- The time taken for a frictionless particle to slide down a slope -/
noncomputable def slide_time (α : Real) (x : Real) (g : Real) : Real :=
  Real.sqrt ((2 * x) / (g * Real.sin (2 * α)))

/-- Theorem: The angle that minimizes the slide time is 45 degrees -/
theorem optimal_slope_angle (x : Real) (g : Real) (h₁ : x > 0) (h₂ : g > 0) :
  ∃ (α : Real), ∀ (β : Real), 0 < β ∧ β < π / 2 → slide_time α x g ≤ slide_time β x g ∧ α = π / 4 := by
  sorry

#check optimal_slope_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_slope_angle_l504_50489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_eccentricity_l504_50458

noncomputable def geometric_mean (a b : ℝ) : ℝ := Real.sqrt (a * b)

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

theorem conic_eccentricity (m : ℝ) :
  m = geometric_mean 2 8 →
  (∃ e : ℝ, (e = Real.sqrt 3 / 2 ∨ e = Real.sqrt 5) ∧
   e = eccentricity 1 (Real.sqrt m)) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_eccentricity_l504_50458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_example_l504_50450

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(a² + b²) / a -/
noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2) / a

/-- The hyperbola equation x²/16 - y²/9 = 1 has eccentricity 5/4 -/
theorem hyperbola_eccentricity_example : hyperbola_eccentricity 4 3 = 5/4 := by
  -- Unfold the definition of hyperbola_eccentricity
  unfold hyperbola_eccentricity
  -- Simplify the expression
  simp [Real.sqrt_eq_rpow]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_example_l504_50450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_396_l504_50409

def digit := Fin 10

def permutation (p : Fin 5 → digit) : Prop :=
  Function.Injective p ∧ Function.Surjective p

def number (p : Fin 5 → digit) : ℕ :=
  3 * 10^17 + (p 0).val * 10^16 + 4 * 10^15 + (p 1).val * 10^14 + 1 * 10^13 + 
  (p 2).val * 10^12 + 0 * 10^11 + (p 3).val * 10^10 + 8 * 10^9 + 2 * 10^8 + 
  (p 4).val * 10^7 + 40923 * 10^6 + 0 * 10^5 + 320 * 10^3 + 2 * 10^2 + 56

theorem divisibility_by_396 (p : Fin 5 → digit) (h : permutation p) : 
  (number p) % 396 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_396_l504_50409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_constraint_l504_50456

theorem intersection_constraint (a : ℤ) : 
  let M : Set ℤ := {a, 0}
  let N : Set ℤ := {x : ℤ | 2*x^2 - 5*x < 0}
  (M ∩ N).Nonempty → a = 1 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_constraint_l504_50456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_ratio_is_seven_fifths_l504_50496

/-- Represents a cylindrical stage of a rocket --/
structure Stage where
  length : ℝ
  mass : ℝ

/-- Represents a three-stage rocket --/
structure Rocket where
  stage1 : Stage
  stage2 : Stage
  stage3 : Stage

/-- The ratio of the lengths of the first and third stages of a rocket --/
noncomputable def lengthRatio (r : Rocket) : ℝ := r.stage1.length / r.stage3.length

/-- Conditions for a valid three-stage rocket --/
def validRocket (r : Rocket) : Prop :=
  -- The length of the middle stage is half the sum of the lengths of the first and third stages
  r.stage2.length = (r.stage1.length + r.stage3.length) / 2 ∧
  -- The mass of the middle stage is 13/6 times less than the combined mass of the first and third stages
  r.stage2.mass = (r.stage1.mass + r.stage3.mass) * 6 / 13 ∧
  -- The mass of each stage is proportional to the cube of its length
  ∃ k : ℝ, k > 0 ∧
    r.stage1.mass = k * r.stage1.length ^ 3 ∧
    r.stage2.mass = k * r.stage2.length ^ 3 ∧
    r.stage3.mass = k * r.stage3.length ^ 3

/-- The main theorem: For any valid rocket, the ratio of the lengths of the first and third stages is 7/5 --/
theorem length_ratio_is_seven_fifths (r : Rocket) (h : validRocket r) : lengthRatio r = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_ratio_is_seven_fifths_l504_50496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_number_is_valid_987654_largest_six_digit_sum_21_l504_50465

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 100000 ∧ n < 1000000) ∧  -- six-digit number
  (Nat.digits 10 n).Nodup ∧  -- all digits are different
  (Nat.digits 10 n).sum = 21 ∧  -- sum of digits is 21
  0 ∉ Nat.digits 10 n  -- doesn't include 0

theorem largest_valid_number :
  ∀ n : ℕ, is_valid_number n → n ≤ 987654 :=
by sorry

theorem is_valid_987654 : is_valid_number 987654 :=
by sorry

theorem largest_six_digit_sum_21 :
  ∃ n : ℕ, is_valid_number n ∧ ∀ m, is_valid_number m → m ≤ n :=
by
  use 987654
  constructor
  · exact is_valid_987654
  · intro m hm
    exact largest_valid_number m hm

#check largest_six_digit_sum_21

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_number_is_valid_987654_largest_six_digit_sum_21_l504_50465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_calculation_l504_50427

/-- Represents a cube with its dimensions and surface area -/
structure Cube where
  side : ℕ
  surface_area : ℕ

/-- Calculates the surface area of a cube given its side length -/
def cube_surface_area (side : ℕ) : ℕ :=
  6 * side * side

/-- Represents the problem setup -/
def problem_setup : Prop :=
  ∃ (large_cube : Cube) (small_cube : Cube),
    large_cube.side = 12 ∧
    small_cube.side = 4 ∧
    large_cube.surface_area = cube_surface_area large_cube.side ∧
    small_cube.surface_area = cube_surface_area small_cube.side

/-- The final surface area after removals -/
def final_surface_area : ℕ := 2880

/-- The main theorem stating that the final surface area is correct -/
theorem surface_area_calculation :
  problem_setup → final_surface_area = 2880 := by
  intro h
  -- The proof would go here
  sorry

#check surface_area_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_calculation_l504_50427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nails_for_small_planks_l504_50472

/-- Given the total number of nails and the number of nails for large planks,
    proves that the number of nails for small planks is their difference. -/
theorem nails_for_small_planks (total_nails large_plank_nails : ℕ) :
  total_nails ≥ large_plank_nails →
  total_nails - large_plank_nails = total_nails - large_plank_nails :=
by
  intro h
  rfl

/-- Calculates the number of nails needed for small planks. -/
def solve_problem : ℕ :=
  20 - 15

#eval solve_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nails_for_small_planks_l504_50472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_well_expenditure_approx_l504_50403

/-- Calculates the expenditure for digging a cylindrical well -/
noncomputable def well_expenditure (depth : ℝ) (diameter : ℝ) (cost_per_cubic_meter : ℝ) : ℝ :=
  let radius := diameter / 2
  let volume := Real.pi * radius^2 * depth
  volume * cost_per_cubic_meter

/-- Theorem stating that the expenditure for the given well specifications is approximately 1583.36 Rs -/
theorem well_expenditure_approx :
  ∃ ε > 0, |well_expenditure 14 3 16 - 1583.36| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_well_expenditure_approx_l504_50403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_uniqueness_l504_50419

def is_valid_function (ω φ : Real) : Prop :=
  ω > 0 ∧ -Real.pi/2 < φ ∧ φ < Real.pi/2

def has_period_two (ω : Real) : Prop :=
  2 * Real.pi / ω = 2

def point_on_graph (ω φ : Real) : Prop :=
  Real.sin (ω * (1/3) + φ) = 1

theorem sine_function_uniqueness (ω φ : Real) 
  (h1 : is_valid_function ω φ) 
  (h2 : has_period_two ω) 
  (h3 : point_on_graph ω φ) : 
  ω = Real.pi ∧ φ = Real.pi/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_uniqueness_l504_50419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_arrangement_areas_l504_50483

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ

/-- Represents the arrangement of three squares -/
structure SquareArrangement where
  small : Square
  medium : Square
  large : Square

/-- The line segment connecting the bottom left of the smallest square to the upper right of the largest square -/
structure DiagonalLine where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ

/-- Theorem stating the areas of the trapezoid and triangle -/
theorem square_arrangement_areas (arr : SquareArrangement) (diag : DiagonalLine) : 
  arr.small.sideLength = 3 ∧ 
  arr.medium.sideLength = 6 ∧ 
  arr.large.sideLength = 9 ∧
  diag.start = (0, 0) ∧
  diag.endpoint = (18, 9) →
  ∃ (trapezoidArea triangleArea : ℝ),
    trapezoidArea = 18 ∧
    triangleArea = 20.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_arrangement_areas_l504_50483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_P_l504_50468

noncomputable section

open Real

def EquilateralTriangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

def PointOnLine (D : ℝ × ℝ) (B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (1 - t) • B + t • C

def Circumcenter (O : ℝ × ℝ) (A B D : ℝ × ℝ) : Prop :=
  dist O A = dist O B ∧ dist O B = dist O D

def Incenter (I : ℝ × ℝ) (A B D : ℝ × ℝ) : Prop :=
  ∃ (p q r : ℝ), p > 0 ∧ q > 0 ∧ r > 0 ∧
    I = ((p • A.1 + q • B.1 + r • D.1) / (p + q + r),
         (p • A.2 + q • B.2 + r • D.2) / (p + q + r))

def LinesIntersect (P : ℝ × ℝ) (O₁ I₁ O₂ I₂ : ℝ × ℝ) : Prop :=
  ∃ (t s : ℝ), P = (1 - t) • O₁ + t • I₁ ∧ P = (1 - s) • O₂ + s • I₂

theorem locus_of_P (A B C D O₁ I₁ O₂ I₂ P : ℝ × ℝ) :
  EquilateralTriangle A B C →
  PointOnLine D B C →
  Circumcenter O₁ A B D →
  Incenter I₁ A B D →
  Circumcenter O₂ A D C →
  Incenter I₂ A D C →
  LinesIntersect P O₁ I₁ O₂ I₂ →
  ∃ (x : ℝ), -1 < x ∧ x < 1 ∧ P = (x, -sqrt ((x^2 + 3) / 3)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_P_l504_50468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_e_opposite_a_l504_50442

-- Define the structure of a cube net
structure CubeNet where
  faces : Finset Char
  adjacent : Char → Char → Prop
  opposite : Char → Char → Prop
  tShape : Prop

-- Define the properties of our specific cube net
def specificNet : CubeNet where
  faces := {'A', 'B', 'C', 'D', 'E', 'F'}
  adjacent := λ x y ↦ (x = 'A' ∧ y = 'B') ∨ (x = 'A' ∧ y = 'D') ∨ (x = 'B' ∧ y = 'A') ∨ (x = 'D' ∧ y = 'A')
  opposite := λ x y ↦ x = 'A' ∧ y = 'E'
  tShape := True

-- Theorem stating that E is opposite to A in the specific net
theorem e_opposite_a (net : CubeNet) (h : net = specificNet) :
  net.opposite 'A' 'E' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_e_opposite_a_l504_50442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l504_50418

-- Define a triangle with sides a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_ratio (t : Triangle) 
  (h1 : t.b * Real.cos t.C + t.c * Real.cos t.B = 2 * t.b) : 
  t.b / t.a = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l504_50418
