import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_contains_triple_l619_61970

def X : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem partition_contains_triple (A B : Set Nat) 
  (h_partition : A ∪ B = X) (h_disjoint : A ∩ B = ∅) : 
  (∃ a b c, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b = 2 * c) ∨ 
  (∃ a b c, a ∈ B ∧ b ∈ B ∧ c ∈ B ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b = 2 * c) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_contains_triple_l619_61970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airline_route_within_republic_l619_61949

/-- Represents a city in the country -/
structure City where
  id : Nat
  republic : Nat
  routes : Nat
  deriving Repr, DecidableEq

/-- The country configuration -/
structure Country where
  cities : Finset City
  num_republics : Nat

/-- Predicate to check if a city is a "millionaire city" -/
def is_millionaire_city (c : City) : Bool :=
  c.routes ≥ 70

/-- Main theorem -/
theorem airline_route_within_republic (country : Country)
  (h_total_cities : country.cities.card = 100)
  (h_num_republics : country.num_republics = 3)
  (h_millionaire_cities : (country.cities.filter (fun c => is_millionaire_city c)).card ≥ 70) :
  ∃ (c1 c2 : City), c1 ∈ country.cities ∧ c2 ∈ country.cities ∧ 
    c1.republic = c2.republic ∧ c1 ≠ c2 :=
by
  sorry

#check airline_route_within_republic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_airline_route_within_republic_l619_61949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_valid_polynomials_l619_61910

/-- The number of distinct prime divisors of a positive integer -/
noncomputable def omega (n : ℕ) : ℕ := sorry

/-- An integer coefficient polynomial -/
def IntPolynomial := ℕ → ℤ

/-- The set of polynomials satisfying the given conditions -/
def ValidPolynomials : Set IntPolynomial :=
  {Q | (∀ n : ℕ, n > 0 → Q n ≥ 1) ∧
       (∀ m n : ℕ, m > 0 → n > 0 → omega (Int.natAbs (Q (m * n))) = omega (Int.natAbs ((Q m) * (Q n))))}

/-- The theorem to be proved -/
theorem characterize_valid_polynomials :
  ∀ Q ∈ ValidPolynomials, ∃ (c : ℕ) (d : ℕ), c > 0 ∧ ∀ x : ℕ, x > 0 → Q x = c * x ^ d :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_valid_polynomials_l619_61910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tethered_dog_area_l619_61942

/-- The area outside a regular pentagon reachable by a tethered point -/
theorem tethered_dog_area (side_length : ℝ) (rope_length : ℝ) : 
  side_length = 1 → rope_length = 3 → 
  let pentagon_area := (side_length^2 * (5 * Real.tan (π / 5))) / 4
  let total_area := π * rope_length^2
  total_area - pentagon_area = (37 * π) / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tethered_dog_area_l619_61942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_distance_is_42_l619_61975

/-- Represents the taxi fare calculation problem -/
structure TaxiFare where
  mike_start_fee : ℝ
  cost_per_mile : ℝ
  bridge_toll : ℝ
  annie_distance : ℝ
  mike_distance : ℝ

/-- The taxi fare problem with given conditions -/
def taxi_problem (tf : TaxiFare) : Prop :=
  tf.mike_start_fee = 2.50 ∧
  tf.cost_per_mile = 0.25 ∧
  tf.bridge_toll = 5.00 ∧
  tf.annie_distance = 22 ∧
  tf.mike_start_fee + tf.cost_per_mile * tf.mike_distance =
    tf.mike_start_fee + tf.bridge_toll + tf.cost_per_mile * tf.annie_distance

/-- Theorem stating that Mike's ride was 42 miles long -/
theorem mike_distance_is_42 (tf : TaxiFare) :
  taxi_problem tf → tf.mike_distance = 42 := by
  intro h
  sorry -- The proof is omitted for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_distance_is_42_l619_61975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_negative_sqrt3_over_2_l619_61965

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.sin (Real.pi * x / 2)
  else 1/6 - Real.log x / Real.log 3

theorem f_composition_equals_negative_sqrt3_over_2 :
  f (f (3 * Real.sqrt 3)) = -(Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_negative_sqrt3_over_2_l619_61965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_a_value_l619_61958

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x ∈ Set.Ioo 0 2 then Real.log x - a * x else 0

-- State the theorem
theorem odd_function_a_value (a : ℝ) :
  (∀ x, f a x = -f a (-x)) → -- f is odd
  (a > 1/2) → -- a > 1/2
  (∀ x ∈ Set.Ioo 0 2, f a x = Real.log x - a * x) → -- definition of f for x ∈ (0, 2)
  (∃ m, ∀ x ∈ Set.Ioo (-2) 0, f a x ≥ m ∧ (∃ y ∈ Set.Ioo (-2) 0, f a y = m)) → -- minimum value exists in (-2, 0)
  (∃ m, ∀ x ∈ Set.Ioo (-2) 0, f a x ≥ m ∧ (∃ y ∈ Set.Ioo (-2) 0, f a y = m) ∧ m = 1) → -- minimum value is 1
  a = 1 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_a_value_l619_61958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_landscape_ratio_l619_61966

/-- Represents a rectangular landscape with a playground -/
structure Landscape where
  length : ℝ
  breadth : ℝ
  playground_area : ℝ

/-- The ratio of breadth to length in a landscape -/
noncomputable def breadth_to_length_ratio (l : Landscape) : ℝ :=
  l.breadth / l.length

theorem landscape_ratio (l : Landscape) 
  (h1 : ∃ k : ℝ, l.breadth = k * l.length)
  (h2 : l.breadth = 420)
  (h3 : l.playground_area = 4200)
  (h4 : l.playground_area = (1/7) * l.length * l.breadth) :
  breadth_to_length_ratio l = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_landscape_ratio_l619_61966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l619_61914

def set_A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}
def set_B : Set ℝ := {x : ℝ | Real.rpow 2 (x-1) ≥ 1}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = Set.Icc 1 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l619_61914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_abc_l619_61923

-- Define the constants as noncomputable
noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

-- State the theorem
theorem ordering_of_abc : c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_abc_l619_61923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_formula_l619_61983

/-- A right prism with an equilateral triangle base --/
structure RightPrism where
  /-- Side length of the equilateral triangle base --/
  a : ℝ
  /-- Assumption that a is positive --/
  a_pos : 0 < a

/-- The lateral surface area of the right prism --/
noncomputable def lateral_surface_area (p : RightPrism) : ℝ :=
  (p.a^2 * (2 * Real.sqrt 3 + 2 * Real.sqrt 13)) / Real.sqrt 3

/-- Center of the equilateral triangle base --/
def center_of_equilateral_triangle (side_length : ℝ) : ℝ × ℝ := sorry

/-- A vertex of the top face of the prism --/
def vertex_of_top_face (p : RightPrism) : ℝ × ℝ × ℝ := sorry

/-- Segment connecting a vertex on the base to the corresponding vertex on the top face --/
def segment_connecting_base_to_top (p : RightPrism) : ℝ × ℝ × ℝ := sorry

/-- Orthogonal projection of a point onto the base plane --/
def orthogonal_projection (point : ℝ × ℝ × ℝ) : ℝ × ℝ := sorry

/-- Angle between a line segment and the base plane --/
def angle_with_base (segment : ℝ × ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the lateral surface area of the right prism --/
theorem lateral_surface_area_formula (p : RightPrism) :
  let O := center_of_equilateral_triangle p.a
  let A₁ := vertex_of_top_face p
  let lateral_edge := segment_connecting_base_to_top p
  (orthogonal_projection A₁ = O) →
  (angle_with_base lateral_edge = 60 * π / 180) →
  lateral_surface_area p = (p.a^2 * (2 * Real.sqrt 3 + 2 * Real.sqrt 13)) / Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_formula_l619_61983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_value_l619_61907

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem find_a_value (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (∀ x ∈ Set.Icc 1 2, f a x ≤ f a 1) ∧
  (∀ x ∈ Set.Icc 1 2, f a x ≥ f a 2) ∧
  (f a 1 - f a 2 = a / 2) →
  a = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_value_l619_61907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_value_l619_61918

/-- An ellipse with foci on the x-axis -/
structure Ellipse where
  m : ℝ
  equation : ∀ (x y : ℝ), m * x^2 + y^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := 
  Real.sqrt (1 - (1 / e.m))

/-- Theorem: For an ellipse with equation mx^2 + y^2 = 1, 
    where the foci are on the x-axis and the eccentricity is 1/2, 
    the value of m is 3/4 -/
theorem ellipse_m_value (e : Ellipse) : 
  eccentricity e = 1/2 → e.m = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_value_l619_61918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_angle_triangles_l619_61988

/-- A triangle with integer angles in degrees -/
structure IntegerAngleTriangle where
  α : ℕ
  β : ℕ
  γ : ℕ
  sum_180 : α + β + γ = 180
  positive : 0 < α ∧ 0 < β ∧ 0 < γ

/-- The set of all distinct integer angle triangles -/
def AllIntegerAngleTriangles : Set IntegerAngleTriangle :=
  {t : IntegerAngleTriangle | t.α ≤ t.β ∧ t.β ≤ t.γ}

/-- Fintype instance for AllIntegerAngleTriangles -/
instance : Fintype AllIntegerAngleTriangles := by
  sorry  -- The actual implementation would go here

/-- The number of integer angle triangles is 2700 -/
theorem count_integer_angle_triangles : 
  Fintype.card AllIntegerAngleTriangles = 2700 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_angle_triangles_l619_61988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_area_from_axonometric_l619_61935

/-- The area of a triangle after oblique axonometric drawing -/
noncomputable def axonometric_area : ℝ := 1

/-- The ratio of the height in axonometric drawing to the original height -/
noncomputable def height_ratio : ℝ := Real.sqrt 2 / 4

/-- The original area of the triangle -/
noncomputable def original_area : ℝ := 2 * Real.sqrt 2

/-- Theorem stating the relationship between axonometric area and original area -/
theorem original_area_from_axonometric :
  axonometric_area = height_ratio * original_area := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_area_from_axonometric_l619_61935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_squared_l619_61901

theorem sin_minus_cos_squared (α : ℝ) 
  (h1 : Real.sin (2 * α) = 1 / 2) 
  (h2 : 0 < α ∧ α < Real.pi / 2) : 
  (Real.sin α - Real.cos α)^2 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_squared_l619_61901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_root_simplification_l619_61972

theorem fifth_root_simplification :
  ((2^11 * 3^5 : ℕ) : ℝ)^(1/5) = 32 * 486^(1/5 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_root_simplification_l619_61972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangency_l619_61989

/-- Represents a parabola with equation y² = 2px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a circle with equation x² + y² + ax + b = 0 -/
structure Circle where
  a : ℝ
  b : ℝ

/-- The center of a circle -/
noncomputable def Circle.center (c : Circle) : ℝ × ℝ := (-c.a/2, 0)

/-- The radius of a circle -/
noncomputable def Circle.radius (c : Circle) : ℝ := Real.sqrt ((c.a/2)^2 - c.b)

/-- The directrix of a parabola -/
noncomputable def Parabola.directrix (p : Parabola) : ℝ := -p.p/2

/-- States that the directrix of a parabola is tangent to a circle -/
def is_tangent (p : Parabola) (c : Circle) : Prop :=
  |Parabola.directrix p - (Circle.center c).1| = Circle.radius c

theorem parabola_circle_tangency (p : Parabola) (c : Circle) 
  (h_eq1 : c.a = 6) (h_eq2 : c.b = 8) (h_tangent : is_tangent p c) : 
  p.p = 4 ∨ p.p = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangency_l619_61989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_theorem_l619_61948

theorem subset_intersection_theorem (n : ℕ) (A : Fin n → Finset (Fin n)) :
  (∀ i : Fin n, (A i).card = 3) →
  ∃ S : Finset (Fin n), 
    S.card ≤ ⌊(2 * n : ℝ) / 5⌋ ∧ 
    ∀ i : Fin n, ∃ x ∈ A i, x ∉ S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_theorem_l619_61948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chile_earthquake_third_major_l619_61990

/-- Represents an earthquake event -/
structure Earthquake where
  magnitude : Float
  date : String
  latitude : Float
  longitude : Float
  depth : Float

/-- Checks if an earthquake has a magnitude of 8.0 or greater -/
def isMajorEarthquake (e : Earthquake) : Bool :=
  e.magnitude ≥ 8.0

/-- The list of major earthquakes in Chile in the past five years -/
def chileEarthquakes : List Earthquake := [
  { magnitude := 8.8, date := "2010-02-27", latitude := -35.9, longitude := -72.7, depth := 35.0 },
  { magnitude := 8.1, date := "2014-04-02", latitude := -19.6, longitude := -70.8, depth := 25.0 },
  { magnitude := 8.2, date := "2015-09-17", latitude := -31.6, longitude := -71.6, depth := 20.0 }
]

/-- Theorem stating that the 2015 Chile earthquake is the third major earthquake in five years -/
theorem chile_earthquake_third_major :
  (chileEarthquakes.filter isMajorEarthquake).length = 3 ∧
  (chileEarthquakes.getLast?.map isMajorEarthquake) = some true := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chile_earthquake_third_major_l619_61990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_182_train_length_unique_l619_61934

/-- The length of a train given specific time and distance conditions -/
def train_length (L : ℝ) : Prop :=
  ∃ (V : ℝ), V = L / 8 ∧ V = (L + 273) / 20

/-- Theorem stating that the train length is 182 meters -/
theorem train_length_is_182 : ∃ L : ℝ, train_length L ∧ L = 182 :=
  sorry

/-- Proof that the train length is unique -/
theorem train_length_unique : ∀ L₁ L₂ : ℝ, train_length L₁ → train_length L₂ → L₁ = L₂ :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_182_train_length_unique_l619_61934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_and_line_intersection_l619_61954

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Helper functions
def Circle.points (S : Circle) : Set (ℝ × ℝ) :=
  {P | (P.1 - S.center.1)^2 + (P.2 - S.center.2)^2 = S.radius^2}

def Line.points (L : Line) : Set (ℝ × ℝ) :=
  {P | ∃ t : ℝ, P = (L.point1.1 + t * (L.point2.1 - L.point1.1), 
                     L.point1.2 + t * (L.point2.2 - L.point1.2))}

-- Define the theorem
theorem circle_line_intersection_and_line_intersection 
  (S : Circle) 
  (AB : Line) 
  (A₁B₁ : Line) 
  (A₂B₂ : Line) : 
  (∃ P : ℝ × ℝ, P ∈ S.points ∧ P ∈ AB.points) ∧ 
  (∃ Q : ℝ × ℝ, Q ∈ A₁B₁.points ∧ Q ∈ A₂B₂.points) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_and_line_intersection_l619_61954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_slope_sum_l619_61959

-- Define the slopes and points for the two lines
def m1 : ℚ := -5
def m2 : ℚ := 3
def p1 : ℚ × ℚ := (2, 4)
def p2 : ℚ × ℚ := (1, 2)

-- Define the intersection point and sum of slopes
def intersection_point : ℚ × ℚ := (15/8, 37/8)
def sum_of_slopes : ℚ := -2

-- Theorem statement
theorem intersection_and_slope_sum :
  let line1 := λ (x : ℚ) => m1 * x + (p1.2 - m1 * p1.1)
  let line2 := λ (x : ℚ) => m2 * x + (p2.2 - m2 * p2.1)
  (∃ x : ℚ, line1 x = line2 x ∧ 
    (x, line1 x) = intersection_point) ∧
  m1 + m2 = sum_of_slopes := by
  sorry

#check intersection_and_slope_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_slope_sum_l619_61959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_class_students_l619_61927

theorem first_class_students (avg_first : ℝ) (num_second : ℕ) (avg_second : ℝ) (avg_total : ℝ) :
  avg_first = 45 →
  num_second = 55 →
  avg_second = 65 →
  avg_total = 57.22222222222222 →
  ∃ (num_first : ℕ), num_first = 551 ∧
    (avg_first * (num_first : ℝ) + avg_second * (num_second : ℝ)) / ((num_first : ℝ) + (num_second : ℝ)) = avg_total :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_class_students_l619_61927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_angle_probability_l619_61903

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hexagon defined by its vertices -/
structure Hexagon where
  F : Point
  G : Point
  H : Point
  I : Point
  J : Point
  K : Point

/-- The hexagon FGHIJK as defined in the problem -/
noncomputable def problemHexagon : Hexagon := {
  F := { x := 0, y := 3 },
  G := { x := 5, y := 0 },
  H := { x := 2 * Real.pi + 2, y := 0 },
  I := { x := 2 * Real.pi + 2, y := 5 },
  J := { x := 0, y := 5 },
  K := { x := 3, y := 5 }
}

/-- The probability that angle FQG is obtuse when Q is randomly selected from the interior of the hexagon -/
noncomputable def probabilityOfObtuseAngle (h : Hexagon) : ℝ :=
  17 / (40 * Real.pi + 60)

/-- Theorem stating that the probability of ∠FQG being obtuse is 17 / (40π + 60) -/
theorem obtuse_angle_probability :
  probabilityOfObtuseAngle problemHexagon = 17 / (40 * Real.pi + 60) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_angle_probability_l619_61903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_theorem_l619_61991

/-- Calculates the return speed given the distance, outbound speed, and average speed of a round trip -/
noncomputable def return_speed (distance : ℝ) (outbound_speed : ℝ) (average_speed : ℝ) : ℝ :=
  (2 * distance * average_speed) / (2 * distance - outbound_speed * (distance / average_speed))

/-- The return speed for the given problem conditions -/
noncomputable def cyclist_return_speed : ℝ :=
  return_speed 150 30 25

theorem cyclist_speed_theorem :
  ∃ ε > 0, abs (cyclist_return_speed - 750 / 35) < ε :=
by sorry

#eval Float.toString (750 / 35 : Float)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_theorem_l619_61991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kineticEnergyRatioIsHalf_l619_61967

/-- Represents a perfectly elastic collision between two masses on a frictionless surface. -/
structure ElasticCollision where
  m : ℝ  -- Mass of the smaller object
  v₀ : ℝ  -- Initial velocity of the smaller object

/-- Calculates the ratio of final to initial kinetic energy in the elastic collision. -/
noncomputable def kineticEnergyRatio (collision : ElasticCollision) : ℝ :=
  let m := collision.m
  let v₀ := collision.v₀
  let finalKE := (3 * m * (v₀ / 2)^2 + m * (v₀ / 2)^2) / 2
  let initialKE := m * v₀^2 / 2
  finalKE / initialKE

/-- Theorem stating that the kinetic energy ratio is 1/2 for any elastic collision. -/
theorem kineticEnergyRatioIsHalf (collision : ElasticCollision) :
  kineticEnergyRatio collision = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kineticEnergyRatioIsHalf_l619_61967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l619_61902

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x - Real.cos x, -1)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - 1/2

theorem function_properties :
  ∀ x : ℝ,
  -- 1. Expression of f(x)
  f x = Real.sqrt 2 / 2 * Real.sin (2 * x - π / 4) ∧
  -- 2. Axis of symmetry
  (∃ k : ℤ, x = k * π / 2 + 3 * π / 8 →
    ∀ y : ℝ, f (x + y) = f (x - y)) ∧
  -- 3. Center of symmetry
  (∃ k : ℤ, (k * π / 2 + π / 8, 0) = (x, f x) →
    ∀ y : ℝ, f (x + y) = f (x - y)) ∧
  -- 4. Solution set
  (f x ≥ 1/2 ↔ ∃ k : ℤ, π / 4 + k * π ≤ x ∧ x ≤ π / 2 + k * π) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l619_61902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_factorial_divisible_by_100000_l619_61961

theorem least_factorial_divisible_by_100000 :
  ∀ n : ℕ, n ≥ 1 → (∀ k : ℕ, k ≥ 1 ∧ k < 25 → ¬(100000 ∣ Nat.factorial k)) ∧ (100000 ∣ Nat.factorial 25) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_factorial_divisible_by_100000_l619_61961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_f_l619_61921

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 * Real.sin x + Real.cos x) * (Real.sqrt 3 * Real.cos x - Real.sin x)

/-- The period of the function f(x) -/
noncomputable def period : ℝ := Real.pi

/-- Theorem stating that the minimum positive period of f(x) is π -/
theorem min_positive_period_f :
  ∀ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) → p ≥ period := by
  sorry

#check min_positive_period_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_f_l619_61921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_passes_probability_l619_61913

theorem only_one_passes_probability 
  (prob_A prob_B prob_C : ℚ)
  (h_A : prob_A = 4/5)
  (h_B : prob_B = 3/5)
  (h_C : prob_C = 7/10)
  (h_independence : True) -- Assuming independence
  : prob_A * (1 - prob_B) * (1 - prob_C) + 
    (1 - prob_A) * prob_B * (1 - prob_C) + 
    (1 - prob_A) * (1 - prob_B) * prob_C = 47/250 := by
  sorry

#eval (4/5 : ℚ) * (1 - 3/5) * (1 - 7/10) + 
      (1 - 4/5) * (3/5) * (1 - 7/10) + 
      (1 - 4/5) * (1 - 3/5) * (7/10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_passes_probability_l619_61913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_product_l619_61981

theorem repeating_decimal_product : 
  (246 / 999 : ℚ) + (135 / 999 : ℚ) * (369 / 999 : ℚ) = 140529 / 998001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_product_l619_61981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_when_a_zero_f_negative_condition_l619_61998

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * (x - 1)^2 - x + 1

-- Theorem for part I
theorem f_minimum_when_a_zero :
  ∃ (x : ℝ), x > 0 ∧ f 0 x = 0 ∧ ∀ y > 0, f 0 y ≥ f 0 x :=
sorry

-- Theorem for part II
theorem f_negative_condition (a : ℝ) :
  (∀ x > 1, f a x < 0) ↔ a ≥ 1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_when_a_zero_f_negative_condition_l619_61998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_five_iff_c_eq_neg_eight_l619_61944

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y c : ℝ) : Prop :=
  x^2 + 8*x + y^2 - 2*y + c = 0

/-- The radius of a circle given its center (h, k) and a point (x, y) on the circle -/
noncomputable def circle_radius (h k x y : ℝ) : ℝ :=
  ((x - h)^2 + (y - k)^2).sqrt

theorem circle_radius_five_iff_c_eq_neg_eight :
  ∀ c : ℝ, (∀ x y : ℝ, circle_equation x y c → circle_radius (-4) 1 x y = 5) ↔ c = -8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_five_iff_c_eq_neg_eight_l619_61944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l619_61916

theorem sin_alpha_value (α β : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : Real.pi / 2 < β ∧ β < Real.pi)
  (h3 : Real.sin (α + β) = 3 / 5)
  (h4 : Real.cos β = -5 / 13) :
  Real.sin α = 33 / 65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l619_61916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_digits_0_to_99_l619_61924

def sum_digits (n : ℕ) : ℕ := 
  if n < 10 then n else n % 10 + sum_digits (n / 10)

def sum_range_digits (a b : ℕ) : ℕ := 
  (List.range (b - a + 1)).map (λ i => sum_digits (a + i)) |>.sum

theorem sum_digits_0_to_99 : 
  (sum_range_digits 0 99 = 900) ∧ (sum_range_digits 18 21 = 24) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_digits_0_to_99_l619_61924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l619_61976

/-- Calculates the length of a platform given train parameters -/
noncomputable def platformLength (trainLength : ℝ) (timePlatform : ℝ) (timePole : ℝ) : ℝ :=
  (trainLength * timePlatform / timePole) - trainLength

/-- Theorem stating the platform length calculation -/
theorem platform_length_calculation :
  let trainLength : ℝ := 300
  let timePlatform : ℝ := 39
  let timePole : ℝ := 18
  let calculatedLength := platformLength trainLength timePlatform timePole
  ∃ ε > 0, abs (calculatedLength - 350.13) < ε :=
by
  sorry

-- Remove the #eval statement as it's not necessary for the proof
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l619_61976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_C_is_3_covering_l619_61960

def is_subsequence (s t : List ℕ) : Prop :=
  ∃ (indices : List ℕ), List.Sorted (· < ·) indices ∧ 
    indices.length = s.length ∧ 
    (∀ i, i < s.length → s.get! i = t.get! (indices.get! i))

def is_n_covering_sequence (n : ℕ) (s : List ℕ) : Prop :=
  ∀ p : List ℕ, p.length = n → (∀ i, i < n → p.get! i < n) → 
    p.toFinset = Finset.range n → is_subsequence p s

theorem sequence_C_is_3_covering : 
  is_n_covering_sequence 3 [1, 2, 3, 1, 2, 1, 3] :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_C_is_3_covering_l619_61960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nicky_pace_is_three_l619_61971

/-- Race between Nicky and Cristina --/
structure Race where
  head_start : ℚ
  cristina_pace : ℚ
  catch_up_time : ℚ

/-- Calculate Nicky's pace given the race conditions --/
def nicky_pace (race : Race) : ℚ :=
  (race.cristina_pace * race.catch_up_time - race.head_start) / race.catch_up_time

/-- Theorem stating that Nicky's pace is 3 meters per second --/
theorem nicky_pace_is_three (race : Race) 
  (h1 : race.head_start = 36)
  (h2 : race.cristina_pace = 4)
  (h3 : race.catch_up_time = 36) :
  nicky_pace race = 3 := by
  sorry

/-- Compute Nicky's pace for the given race conditions --/
def main : IO Unit := do
  let race : Race := { head_start := 36, cristina_pace := 4, catch_up_time := 36 }
  IO.println s!"Nicky's pace: {nicky_pace race}"

#eval main

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nicky_pace_is_three_l619_61971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_points_a_prob_a_b_same_points_l619_61962

-- Define the probabilities
noncomputable def p_a_wins_b : ℝ := 2/5
noncomputable def p_draw_ab : ℝ := 1/5
noncomputable def p_a_wins_c : ℝ := 1/3
noncomputable def p_draw_ac : ℝ := 1/3
noncomputable def p_b_wins_c : ℝ := 1/2
noncomputable def p_draw_bc : ℝ := 1/6

-- Define the point system
def win_points : ℝ := 3
def draw_points : ℝ := 1
def lose_points : ℝ := 0

-- Define the expected points function
noncomputable def expected_points (p_win p_draw : ℝ) : ℝ :=
  win_points * p_win + draw_points * p_draw + lose_points * (1 - p_win - p_draw)

-- Theorem for the expected points of player A
theorem expected_points_a :
  expected_points p_a_wins_b p_draw_ab + expected_points p_a_wins_c p_draw_ac = 41/15 := by
  sorry

-- Theorem for the probability that A and B have the same points
theorem prob_a_b_same_points :
  let p1 := p_draw_ab * (1 - p_a_wins_c - p_draw_ac) * (1 - p_b_wins_c - p_draw_bc)
  let p2 := p_draw_ab * p_draw_ac * p_draw_bc
  let p3 := p_a_wins_b * p_b_wins_c * (1 - p_a_wins_c - p_draw_ac) +
            p_a_wins_b * (1 - p_b_wins_c - p_draw_bc) * (1 - p_a_wins_c - p_draw_ac)
  let p4 := p_draw_ab * p_a_wins_c * p_b_wins_c
  p1 + p2 + p3 + p4 = 8/45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_points_a_prob_a_b_same_points_l619_61962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_bounds_local_max_condition_l619_61995

-- Part 1
theorem sin_bounds (x : ℝ) (h : 0 < x ∧ x < 1) : x - x^2 < Real.sin x ∧ Real.sin x < x := by sorry

-- Part 2
noncomputable def f (a x : ℝ) : ℝ := Real.cos (a * x) - Real.log (1 - x^2)

theorem local_max_condition (a : ℝ) : 
  (∃ δ > 0, ∀ x ∈ Set.Ioo (-δ) δ, f a x ≤ f a 0) ↔ 
  (a < -Real.sqrt 2 ∨ a > Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_bounds_local_max_condition_l619_61995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jones_elementary_boys_calculation_l619_61932

/-- Represents the number of students that make up x percent of the boys at Jones Elementary School -/
noncomputable def students_representing_x_percent_of_boys (x : ℝ) : ℝ :=
  (x / 100) * 73.48469228349535

theorem jones_elementary_boys_calculation (x : ℝ) :
  let total_students : ℝ := 122.47448713915891
  let boys_percentage : ℝ := 60
  let boys_count : ℝ := total_students * (boys_percentage / 100)
  students_representing_x_percent_of_boys x = (x / 100) * boys_count :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jones_elementary_boys_calculation_l619_61932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distinct_prime_divisors_of_1728_l619_61997

theorem sum_of_distinct_prime_divisors_of_1728 : 
  (Finset.sum (Finset.filter Nat.Prime (Nat.divisors 1728)) id) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distinct_prime_divisors_of_1728_l619_61997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_ride_time_l619_61900

/-- The time taken to cover a 1-mile stretch of highway on a bike following a quarter-circle path -/
theorem bike_ride_time 
  (highway_length : ℝ)
  (highway_width : ℝ)
  (bike_speed : ℝ)
  (feet_per_mile : ℝ)
  (h1 : highway_length = 1)
  (h2 : highway_width = 50)
  (h3 : bike_speed = 3)
  (h4 : feet_per_mile = 5280) :
  (highway_length * feet_per_mile / (highway_width / 2) * (π / 4) * (highway_width / 2)) / (bike_speed * feet_per_mile) = π / 6 := by
  sorry

#check bike_ride_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_ride_time_l619_61900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l619_61955

/-- The equation of a parabola -/
def parabola (x y : ℝ) : Prop := y = (x^2 - 8*x + 12) / 16

/-- The equation of the directrix -/
def directrix (y : ℝ) : Prop := y = -17/4

/-- Theorem stating that the given directrix is correct for the parabola -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola x y → ∃ d : ℝ, directrix d ∧ 
  d = -17/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l619_61955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_4034_implies_c_equals_2017_l619_61933

noncomputable def f (a b c x : ℝ) : ℝ := a * Real.sin x + b / x + c

theorem f_sum_equals_4034_implies_c_equals_2017 (a b c : ℝ) :
  (∀ x ∈ (Set.Icc (-5 * Real.pi) 0 ∪ Set.Ioc 0 (5 * Real.pi)), f a b c x = a * Real.sin x + b / x + c) →
  f a b c 1 + f a b c (-1) = 4034 →
  c = 2017 := by
  intro h1 h2
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_4034_implies_c_equals_2017_l619_61933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_lambda_l619_61908

/-- Given points A, B, C in R², and a point P that satisfies the vector equation
    AP = AB + lambda * AC for some real lambda, prove that if P is in the third quadrant,
    then lambda < -1. -/
theorem range_of_lambda (A B C P : ℝ × ℝ) (lambda : ℝ) :
  A = (2, 3) →
  B = (5, 4) →
  C = (7, 10) →
  P.1 < 0 →
  P.2 < 0 →
  P - A = (B - A) + lambda • (C - A) →
  lambda < -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_lambda_l619_61908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_O_equation_and_properties_l619_61936

noncomputable section

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the circle M
def circle_M (x y r : ℝ) : Prop := (x + 2)^2 + (y + 2)^2 = r^2 ∧ r > 0

-- Define the line l
def line_l (x y : ℝ) : Prop := y = 1/2 * x - 2

-- Define the point N
def point_N : ℝ × ℝ := (1, Real.sqrt 2 / 2)

theorem circle_O_equation_and_properties :
  ∃ (a b : ℝ),
    -- Circle O passes through A(1,1)
    circle_O 1 1 ∧
    -- O is symmetric to M with respect to x+y+2=0
    (∀ (x y r : ℝ), circle_M x y r → 
      ∃ (x' y' : ℝ), circle_O x' y' ∧ symmetry_line ((x + x') / 2) ((y + y') / 2)) →
    -- The equation of circle O
    (∀ (x y : ℝ), circle_O x y ↔ x^2 + y^2 = 2) ∧
    -- Maximum area of quadrilateral EGFH
    (∀ (E F G H : ℝ × ℝ),
      let (ex, ey) := E
      let (fx, fy) := F
      let (gx, gy) := G
      let (hx, hy) := H
      circle_O ex ey ∧ circle_O fx fy ∧ circle_O gx gy ∧ circle_O hx hy →
      (ex - gx) * (fy - hy) = (fx - hx) * (ey - gy) →
      ((ex - fx) * (ey - fy) + (gx - hx) * (gy - hy) = 0) →
      Real.sqrt ((ex - gx)^2 + (ey - gy)^2) * Real.sqrt ((fx - hx)^2 + (fy - hy)^2) / 2 ≤ 5/2) ∧
    -- Line CD passes through (1/2, -1)
    (∀ (P C D : ℝ × ℝ),
      let (px, py) := P
      let (cx, cy) := C
      let (dx, dy) := D
      line_l px py →
      circle_O cx cy ∧ circle_O dx dy →
      ((cx - px)^2 + (cy - py)^2) * ((dx - px)^2 + (dy - py)^2) = 
        ((cx - dx)^2 + (cy - dy)^2) * ((px)^2 + (py)^2) →
      ∃ (t : ℝ), cx + t * (dx - cx) = 1/2 ∧ cy + t * (dy - cy) = -1) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_O_equation_and_properties_l619_61936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_a_range_l619_61925

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 7 then (3 - a) * x - 3 else a^(x - 6)

theorem f_monotone_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  a ∈ Set.Icc (9/4) 3 := by
  sorry

#check f_monotone_a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_a_range_l619_61925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l619_61931

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₂ - c₁| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance between the parallel lines 4x + 3y - 4 = 0 and 8x + 6y - 9 = 0 is 1/10 -/
theorem distance_between_given_lines :
  distance_between_parallel_lines 4 3 4 (9/2) = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l619_61931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_perimeter_l619_61987

theorem right_triangle_perimeter : ∀ a b : ℝ,
  (2 * a^2 - 8 * a + 7 = 0) →
  (2 * b^2 - 8 * b + 7 = 0) →
  a ≠ b →
  let c := Real.sqrt (a^2 + b^2)
  a + b + c = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_perimeter_l619_61987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_keystone_arch_angle_l619_61968

/-- Represents the number of trapezoids in the keystone arch. -/
def num_trapezoids : ℕ := 10

/-- Represents the measure of the central angle for each trapezoid in degrees. -/
noncomputable def central_angle : ℝ := 360 / num_trapezoids

/-- Represents the measure of half the central angle in degrees. -/
noncomputable def half_central_angle : ℝ := central_angle / 2

/-- Represents the measure of the vertex angle of each trapezoid in degrees. -/
noncomputable def vertex_angle : ℝ := 180 - half_central_angle

/-- Represents the measure of the smaller interior angle of each trapezoid in degrees. -/
noncomputable def smaller_interior_angle : ℝ := vertex_angle / 2

/-- Represents the measure of the larger interior angle of each trapezoid in degrees. -/
noncomputable def larger_interior_angle : ℝ := 180 - smaller_interior_angle

theorem keystone_arch_angle :
  larger_interior_angle = 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_keystone_arch_angle_l619_61968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_calculation_total_profit_value_l619_61994

/-- Represents the total profit for the year --/
def total_profit : ℝ := 18750

/-- A's investment --/
def A_investment : ℝ := 5000

/-- B's interest rate on half of A's investment --/
def B_interest_rate : ℝ := 0.10

/-- A's monthly income as working partner --/
def A_monthly_income : ℝ := 500

/-- Number of months in a year --/
def months_in_year : ℝ := 12

/-- The ratio of A's income to B's income --/
def income_ratio : ℝ := 2

theorem total_profit_calculation : 
  total_profit = A_investment * B_interest_rate / 2 + 
                 A_monthly_income * months_in_year + 
                 2 * ((total_profit - (A_investment * B_interest_rate / 2 + A_monthly_income * months_in_year)) / 2) := by
  sorry

theorem total_profit_value : total_profit = 18750 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_calculation_total_profit_value_l619_61994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_analysis_inequality_solution_condition_absolute_difference_bound_l619_61963

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.exp x

-- Statement 1
theorem extremum_analysis (x : ℝ) (hx : x > 0) :
  ∃ c > 0, ∀ x₁ x₂, 0 < x₁ ∧ x₁ < c ∧ c < x₂ → 
    (-1 + 1/x₁ > 0) ∧ (-1 + 1/x₂ < 0) :=
sorry

-- Statement 2
theorem inequality_solution_condition (m : ℝ) :
  (∃ x, g x < x + m) ↔ m > 1 :=
sorry

-- Statement 3
theorem absolute_difference_bound (x : ℝ) (hx : x > 0) :
  |f 0 x - g x| > 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_analysis_inequality_solution_condition_absolute_difference_bound_l619_61963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l619_61946

theorem complex_fraction_equality : 
  (1 + 3 * Complex.I) / (1 - Complex.I) = -1 + 2 * Complex.I := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l619_61946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l619_61938

-- Define the domain M
def M : Set ℝ := {x : ℝ | x > 3 ∨ x < 1}

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2*x + 2 - 3 * (4^x)

-- Theorem statement
theorem f_extrema :
  (∀ y ∈ M, ∃ x ∈ M, f x ≥ f y) ∧
  (¬∃ z ∈ M, ∀ w ∈ M, f z ≤ f w) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l619_61938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l619_61909

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 - (2 + a)*x + 2*a

-- Define the inequality
def inequality (a x : ℝ) : Prop := x^2 + 2*a*x + 2*a ≤ 0

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ,
  (∃! x, x ∈ Set.Icc (-1 : ℝ) 1 ∧ quadratic a x = 0) ∧
  (∃ x : ℝ, inequality a x) →
  a ∈ Set.Icc (-1 : ℝ) 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l619_61909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l619_61979

noncomputable def f (x : ℝ) := Real.cos x ^ 4 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - Real.sin x ^ 4

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ (∀ x, f (x + T) = f x) ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6))) ∧
  (∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ 2 ∧ f x ≥ -1) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l619_61979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l619_61940

theorem equation_solution :
  let S := {x : ℝ | x^9 - 21*x^3 - Real.sqrt 22 = 0}
  S = {(22 : ℝ)^(1/6), ((- Real.sqrt 22 + 3 * Real.sqrt 2) / 2)^(1/3), ((- Real.sqrt 22 - 3 * Real.sqrt 2) / 2)^(1/3)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l619_61940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ravi_jumps_39_inches_l619_61905

-- Define the jump heights of the three next highest jumpers
noncomputable def jumper1_height : ℝ := 23
noncomputable def jumper2_height : ℝ := 27
noncomputable def jumper3_height : ℝ := 28

-- Define Ravi's jump multiplier
noncomputable def ravi_multiplier : ℝ := 1.5

-- Define Ravi's jump height
noncomputable def ravi_jump_height : ℝ := ravi_multiplier * ((jumper1_height + jumper2_height + jumper3_height) / 3)

-- Theorem stating Ravi's jump height is 39 inches
theorem ravi_jumps_39_inches : ravi_jump_height = 39 := by
  -- Unfold the definitions
  unfold ravi_jump_height ravi_multiplier jumper1_height jumper2_height jumper3_height
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ravi_jumps_39_inches_l619_61905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magazine_purchase_ways_l619_61920

theorem magazine_purchase_ways : (Nat.choose 3 2 * Nat.choose 8 4) + (Nat.choose 8 5) = 266 := by
  -- Define the total number of magazine types
  let total_types : ℕ := 11
  -- Define the number of types priced at 2 yuan
  let types_2yuan : ℕ := 8
  -- Define the number of types priced at 1 yuan
  let types_1yuan : ℕ := 3
  -- Define the total amount spent
  let total_spent : ℕ := 10
  -- Define the maximum number of each type that can be bought
  let max_per_type : ℕ := 1

  -- The proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magazine_purchase_ways_l619_61920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricketer_wickets_last_match_l619_61973

/-- Represents a cricketer's bowling statistics -/
structure BowlingStats where
  initialAverage : ℚ
  runsLastMatch : ℚ
  averageDecrease : ℚ
  previousWickets : ℕ

/-- Calculates the number of wickets taken in the last match -/
def wicketsLastMatch (stats : BowlingStats) : ℕ :=
  let totalRunsBefore := stats.initialAverage * stats.previousWickets
  let newAverage := stats.initialAverage - stats.averageDecrease
  let totalRunsAfter := totalRunsBefore + stats.runsLastMatch
  let totalWicketsAfter := (totalRunsAfter / newAverage).ceil.toNat
  totalWicketsAfter - stats.previousWickets

/-- Theorem stating that the cricketer took 5 wickets in the last match -/
theorem cricketer_wickets_last_match :
  let stats : BowlingStats := {
    initialAverage := 12.4,
    runsLastMatch := 26,
    averageDecrease := 0.4,
    previousWickets := 85
  }
  wicketsLastMatch stats = 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricketer_wickets_last_match_l619_61973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_equals_2_l619_61945

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Add this case for 0
  | 1 => 2
  | n + 1 => 1 / (1 - sequence_a n)

theorem a_2017_equals_2 : sequence_a 2017 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_equals_2_l619_61945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_together_theorem_actual_time_approx_l619_61986

/-- The time taken for three workers to complete a job together -/
noncomputable def time_together (time_a time_b time_c : ℝ) : ℝ :=
  1 / (1 / time_a + 1 / time_b + 1 / time_c)

/-- Theorem stating that the time taken for three workers to complete a job together
    is equal to the reciprocal of the sum of their individual rates -/
theorem time_together_theorem (time_a time_b time_c : ℝ) 
    (ha : time_a > 0) (hb : time_b > 0) (hc : time_c > 0) :
  time_together time_a time_b time_c = 
    1 / (1 / time_a + 1 / time_b + 1 / time_c) := by
  rfl

/-- The actual time taken for the three workers in the problem -/
noncomputable def actual_time : ℝ := time_together 8 10 12

/-- Theorem stating that the actual time is approximately 3.24 hours -/
theorem actual_time_approx : 
  ∃ ε > 0, |actual_time - 3.24| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_together_theorem_actual_time_approx_l619_61986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l619_61978

/-- The volume of a regular triangular pyramid with given perpendicular length and dihedral angle -/
theorem regular_triangular_pyramid_volume 
  (p : ℝ) -- perpendicular length
  (α : ℝ) -- dihedral angle
  (h_p : p > 0)
  (h_α : 0 < α ∧ α < π)
  : ∃ (V : ℝ), V = (9 * p^3 * Real.tan (α/2)^3) / (4 * Real.sqrt (3 * Real.tan (α/2)^2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l619_61978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_inequality_solve_combination_sum_l619_61999

-- Part 1
def permutation (n k : ℕ) : ℕ := 
  if k ≤ n then (Nat.factorial n) / (Nat.factorial (n - k)) else 0

theorem solve_inequality : 
  ∃! x : ℕ, x > 0 ∧ x ≤ 6 ∧ permutation 6 x < 4 * permutation 6 (x - 2) ∧ x = 6 :=
by sorry

-- Part 2
def combination (n k : ℕ) : ℕ := 
  if k ≤ n then (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k)) else 0

theorem solve_combination_sum : 
  ∃! n : ℕ, n > 0 ∧ (Finset.sum (Finset.range (n - 2)) (fun i => combination (i + 3) 2)) = 55 ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_inequality_solve_combination_sum_l619_61999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_range_l619_61974

theorem geometric_sequence_common_ratio_range 
  (a : ℕ+ → ℝ) 
  (b : ℕ+ → ℝ) 
  (q : ℝ) 
  (h1 : ∀ n : ℕ+, a n = 2 * n)
  (h2 : ∀ n : ℕ+, b (n + 1) = q * b n)
  (h3 : ∀ n : ℕ+, b n ≥ a n)
  (h4 : b 4 = a 4) :
  q ∈ Set.Icc (5 / 4 : ℝ) (4 / 3 : ℝ) := by
  sorry

#check geometric_sequence_common_ratio_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_range_l619_61974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moving_circle_trajectory_l619_61941

-- Define the two fixed circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 5 = 0

-- Define a moving circle
def movingCircle (cx cy r : ℝ) (x y : ℝ) : Prop := (x - cx)^2 + (y - cy)^2 = r^2

-- Define tangency condition
def isTangent (c1 c2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), c1 x y ∧ c2 x y ∧ ∀ (x' y' : ℝ), c1 x' y' ∧ c2 x' y' → (x = x' ∧ y = y')

-- Theorem statement
theorem moving_circle_trajectory :
  ∀ (cx cy r : ℝ),
    (isTangent (movingCircle cx cy r) circle1) →
    (isTangent (movingCircle cx cy r) circle2) →
    -- The trajectory of (cx, cy) is one branch of a hyperbola
    ∃ (a b : ℝ), (cx/a)^2 - (cy/b)^2 = 1 ∨ (cx/a)^2 - (cy/b)^2 = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_moving_circle_trajectory_l619_61941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_francine_normal_frogs_l619_61917

/-- Represents the number of frogs Francine caught -/
def total_frogs : ℕ := 27

/-- Represents the number of mutated frogs -/
def mutated_frogs : ℕ := 9

/-- Represents the percentage of mutated frogs -/
def mutation_rate : ℚ := 33 / 100

/-- Represents the number of normal frogs -/
def normal_frogs : ℕ := 18

/-- Theorem stating the number of normal frogs Francine caught -/
theorem francine_normal_frogs : 
  (mutated_frogs : ℚ) / mutation_rate = total_frogs ∧
  (total_frogs : ℚ) * (1 - mutation_rate) = normal_frogs ∧
  normal_frogs = 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_francine_normal_frogs_l619_61917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gravitational_field_equality_l619_61969

/-- The gravitational field above the surface of a spherical planet -/
noncomputable def gravitational_field_above (R : ℝ) (x : ℝ) : ℝ := (R ^ 3) / ((R + x) ^ 2)

/-- The gravitational field below the surface of a spherical planet -/
def gravitational_field_below (R : ℝ) (x : ℝ) : ℝ := (R - x)

/-- Theorem stating the value of x that equalizes the gravitational field above and below the surface -/
theorem gravitational_field_equality (R : ℝ) (h : R > 0) :
  ∃ x : ℝ, x > 0 ∧ gravitational_field_above R x = gravitational_field_below R x ∧
            x = R * ((-1 + Real.sqrt 5) / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gravitational_field_equality_l619_61969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cot_squared_sum_l619_61964

theorem tan_cot_squared_sum (α : Real) 
  (h : Real.sin α + Real.cos α = 1/2) : 
  Real.tan α ^ 2 + (1 / Real.tan α) ^ 2 = 46/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cot_squared_sum_l619_61964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_leg_of_similar_triangle_main_proof_l619_61953

/-- Represents a right triangle with known leg and hypotenuse lengths -/
structure RightTriangle where
  knownLeg : ℝ
  hypotenuse : ℝ

/-- Calculates the length of the unknown leg in a right triangle -/
noncomputable def unknownLeg (t : RightTriangle) : ℝ :=
  Real.sqrt (t.hypotenuse ^ 2 - t.knownLeg ^ 2)

/-- Theorem about the shortest leg of a similar triangle -/
theorem shortest_leg_of_similar_triangle 
  (t1 : RightTriangle)
  (h1 : t1.knownLeg = 15)
  (h2 : t1.hypotenuse = 34)
  (h3 : unknownLeg t1 * 1.33 < t1.knownLeg * 2) :
  let t2 := RightTriangle.mk (unknownLeg t1 * 1.33) (t1.hypotenuse * 2)
  (t2.knownLeg : ℝ) = 30 := by
  sorry

/-- Proof of the main theorem -/
theorem main_proof : ∃ (t1 : RightTriangle), 
  t1.knownLeg = 15 ∧ 
  t1.hypotenuse = 34 ∧ 
  unknownLeg t1 * 1.33 < t1.knownLeg * 2 ∧
  let t2 := RightTriangle.mk (unknownLeg t1 * 1.33) (t1.hypotenuse * 2)
  (t2.knownLeg : ℝ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_leg_of_similar_triangle_main_proof_l619_61953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_digits_l619_61993

/-- Represents a nonzero digit (1-9) -/
def NonzeroDigit := { d : ℕ // 1 ≤ d ∧ d ≤ 9 }

/-- Constructs An as an n-digit number with all digits equal to a -/
def An (a : NonzeroDigit) (n : ℕ+) : ℕ :=
  a.val * ((10 ^ n.val - 1) / 9)

/-- Constructs Bn as an n-digit number with all digits equal to b -/
def Bn (b : NonzeroDigit) (n : ℕ+) : ℕ :=
  b.val * ((10 ^ n.val - 1) / 9)

/-- Constructs Dn as a 3n-digit number with all digits equal to d -/
def Dn (d : NonzeroDigit) (n : ℕ+) : ℕ :=
  d.val * ((10 ^ (3 * n.val) - 1) / 9)

/-- Predicate to check if the equation Dn - Bn = An^3 holds for given a, b, d, and n -/
def EquationHolds (a b d : NonzeroDigit) (n : ℕ+) : Prop :=
  Dn d n - Bn b n = (An a n) ^ 3

theorem max_sum_of_digits :
  ∃ (a b d : NonzeroDigit),
    (∃ (n₁ n₂ : ℕ+), n₁ ≠ n₂ ∧ EquationHolds a b d n₁ ∧ EquationHolds a b d n₂) ∧
    (∀ (a' b' d' : NonzeroDigit),
      (∃ (n₁ n₂ : ℕ+), n₁ ≠ n₂ ∧ EquationHolds a' b' d' n₁ ∧ EquationHolds a' b' d' n₂) →
      a'.val + b'.val + d'.val ≤ a.val + b.val + d.val) ∧
    a.val + b.val + d.val = 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_digits_l619_61993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_1_simplify_expression_2_l619_61985

-- Problem 1
theorem simplify_expression_1 : (-1 : ℝ)^2021 + Real.sqrt 8 - 4 * Real.sin (π / 4) + |(-2 : ℝ)| + (1/2)^(-2 : ℝ) = 5 := by
  sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ -3) :
  (x^2 - 9) / (x^2 + 2*x + 1) / (x + (3 - x^2) / (x + 1)) = (x - 3) / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_1_simplify_expression_2_l619_61985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_increasing_interval_l619_61912

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 3)

theorem function_increasing_interval
  (ω α β : ℝ)
  (h1 : f ω α = -2)
  (h2 : f ω β = 0)
  (h3 : |α - β| ≥ Real.pi / 2)
  (h4 : ∀ δ, 0 < δ → Real.pi / 2 < |α - β| + δ) :
  ∃ k : ℤ, StrictMonoOn (f ω) (Set.Icc (2 * ↑k * Real.pi - 5 * Real.pi / 6) (2 * ↑k * Real.pi + Real.pi / 6)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_increasing_interval_l619_61912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l619_61926

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Adding case for 0
  | 1 => 1
  | 2 => 2
  | (n + 3) => -sequence_a n

def sequence_b (n : ℕ) : ℚ := sequence_a (n + 1) - sequence_a n

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → sequence_b (n + 1) = -sequence_b n) ∧
  (∀ n : ℕ, sequence_a n = (1 + (-1)^n) / 2) := by
  sorry

#eval sequence_a 0  -- Test case
#eval sequence_a 1  -- Test case
#eval sequence_a 2  -- Test case
#eval sequence_a 3  -- Test case

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l619_61926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aluminum_percentage_bounds_l619_61930

/-- Represents the composition of an alloy -/
structure Alloy where
  aluminum : ℚ
  copper : ℚ
  magnesium : ℚ
  sum_to_one : aluminum + copper + magnesium = 1

/-- The three original alloys -/
def alloy1 : Alloy := ⟨6/10, 15/100, 25/100, by norm_num⟩
def alloy2 : Alloy := ⟨0, 3/10, 7/10, by norm_num⟩
def alloy3 : Alloy := ⟨45/100, 0, 55/100, by norm_num⟩

/-- The percentage of copper in the new alloy -/
def new_copper_percentage : ℚ := 1/5

/-- Theorem stating the bounds on aluminum percentage in the new alloy -/
theorem aluminum_percentage_bounds :
  ∃ (x₁ x₂ x₃ : ℚ),
    x₁ + x₂ + x₃ = 1 ∧
    x₁ * alloy1.copper + x₂ * alloy2.copper + x₃ * alloy3.copper = new_copper_percentage ∧
    0 ≤ x₁ ∧ 0 ≤ x₂ ∧ 0 ≤ x₃ ∧
    15/100 ≤ x₁ * alloy1.aluminum + x₂ * alloy2.aluminum + x₃ * alloy3.aluminum ∧
    x₁ * alloy1.aluminum + x₂ * alloy2.aluminum + x₃ * alloy3.aluminum ≤ 2/5 :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_aluminum_percentage_bounds_l619_61930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_typing_time_theorem_l619_61939

/-- The time taken to type a paper together given individual typing times -/
noncomputable def time_to_type_together (t1 t2 t3 : ℝ) : ℝ :=
  1 / (1 / t1 + 1 / t2 + 1 / t3)

/-- Theorem stating that the time taken to type the paper together is approximately 13.85 minutes -/
theorem typing_time_theorem :
  let randy_time := (30 : ℝ)
  let candy_time := (45 : ℝ)
  let sandy_time := (60 : ℝ)
  ∃ ε > 0, |time_to_type_together randy_time candy_time sandy_time - 13.85| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_typing_time_theorem_l619_61939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_number_l619_61980

open Nat

/-- Function to check if all digits of a number are odd -/
def allDigitsOdd (n : ℕ) : Prop :=
  ∀ d, d ∈ digits 10 n → d % 2 = 1

/-- Function to check if the number of each odd digit (1,3,5,7,9) is equal -/
def equalOddDigitCounts (n : ℕ) : Prop :=
  let oddDigits := [1, 3, 5, 7, 9]
  ∀ d₁ d₂, d₁ ∈ oddDigits → d₂ ∈ oddDigits → 
    (digits 10 n).count d₁ = (digits 10 n).count d₂

/-- Function to check if a number is divisible by all 20-digit numbers obtained by deleting its digits -/
def divisibleByAllDeletions (n : ℕ) : Prop :=
  ∀ m, (digits 10 m).length = 20 → (∃ k, n = m * k) → (digits 10 m).toFinset ⊆ (digits 10 n).toFinset

/-- The main theorem -/
theorem exists_special_number : ∃ N : ℕ,
  N > 10^20 ∧
  allDigitsOdd N ∧
  equalOddDigitCounts N ∧
  divisibleByAllDeletions N :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_number_l619_61980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_conversion_l619_61956

-- Define the conversion factor from radians to degrees
noncomputable def pi_to_deg : ℝ := 180

-- Define the angle in radians
noncomputable def angle_rad : ℝ := 23 / 12 * Real.pi

-- Theorem statement
theorem angle_conversion :
  angle_rad * (pi_to_deg / Real.pi) = -345 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_conversion_l619_61956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_podcast_length_l619_61996

/-- Represents the duration of a podcast in minutes -/
abbrev PodcastLength := Nat

/-- Calculates the total length of podcasts in minutes -/
def total_podcast_length (p1 p2 p3 p4 p5 : PodcastLength) : Nat :=
  p1 + p2 + p3 + p4 + p5

/-- Converts minutes to hours -/
def minutes_to_hours (minutes : Nat) : Nat :=
  minutes / 60

theorem fifth_podcast_length 
  (p1 : PodcastLength) 
  (p2 : PodcastLength) 
  (p3 : PodcastLength) 
  (p4 : PodcastLength) 
  (h1 : p1 = 45)
  (h2 : p2 = 2 * p1)
  (h3 : p3 = 105)
  (h4 : p4 = 60)
  : ∃ (p5 : PodcastLength), 
    minutes_to_hours (total_podcast_length p1 p2 p3 p4 p5) = 6 ∧ 
    p5 = 60 :=
by
  sorry

#check fifth_podcast_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_podcast_length_l619_61996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_arc_length_l619_61957

/-- The length of the arc intercepted by one side of a regular octagon in a larger circle -/
noncomputable def arc_length_octagon (R : ℝ) : ℝ :=
  2.5 * Real.pi

/-- Theorem: The length of the arc intercepted by one side of a regular octagon in a circle with radius 10 units is 2.5π units, given that the octagon is inscribed in a smaller concentric circle with radius 5 units. -/
theorem octagon_arc_length :
  let small_radius : ℝ := 5
  let large_radius : ℝ := 10
  arc_length_octagon large_radius = 2.5 * Real.pi :=
by
  -- Unfold the definition of arc_length_octagon
  unfold arc_length_octagon
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_arc_length_l619_61957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_equation_solution_l619_61906

/-- The operation ⋆ defined as a ⋆ b = √(a + 2b) / √(a - b) -/
noncomputable def star (a b : ℝ) : ℝ := (Real.sqrt (a + 2*b)) / (Real.sqrt (a - b))

/-- Theorem: If y ⋆ 10 = 5, then y = 45/4 -/
theorem star_equation_solution (y : ℝ) (h : star y 10 = 5) : y = 45/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_equation_solution_l619_61906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_combinability_with_sqrt_two_l619_61915

theorem sqrt_combinability_with_sqrt_two :
  (∃ (q : ℚ), (Real.sqrt (1/8 : ℝ)) = q * Real.sqrt 2) ∧
  (∀ (q : ℚ), (Real.sqrt (12 : ℝ)) ≠ q * Real.sqrt 2) ∧
  (∀ (q : ℚ), (Real.sqrt (1/5 : ℝ)) ≠ q * Real.sqrt 2) ∧
  (∀ (q : ℚ), (Real.sqrt (27 : ℝ)) ≠ q * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_combinability_with_sqrt_two_l619_61915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_eight_divisors_l619_61943

theorem factorial_eight_divisors : 
  (Finset.card (Finset.filter (λ d => d > 0 ∧ d ∣ Nat.factorial 8) (Finset.range (Nat.factorial 8 + 1)))) = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_eight_divisors_l619_61943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_open_closed_interval_l619_61947

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {y | ∃ x, y = 2^x}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem stating that the intersection of A and B is equal to (0, 2]
theorem intersection_equals_open_closed_interval :
  A_intersect_B = Set.Ioc 0 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_open_closed_interval_l619_61947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_slope_angle_l619_61951

/-- The slope angle of a line given a hyperbola with specific eccentricity -/
theorem hyperbola_line_slope_angle 
  (m n : ℝ) 
  (h_hyperbola : ∀ x y : ℝ, x^2 / m^2 - y^2 / n^2 = 1)
  (h_eccentricity : m^2 + n^2 = 4 * m^2) :
  let k := -m / n
  ∃ θ : ℝ, (θ = π/6 ∨ θ = 5*π/6) ∧ k = Real.tan θ :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_slope_angle_l619_61951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_unique_l619_61952

theorem quadratic_function_unique (f : ℝ → ℝ) 
  (hf : ∃ a b c, ∀ x, f x = a * x^2 + b * x + c) :
  f 1 = 0 →
  (deriv f) 1 = 2 →
  (∫ x in Set.Icc 0 1, f x) = 0 →
  ∀ x, f x = 3 * x^2 - 4 * x + 1 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_unique_l619_61952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l619_61950

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- A point on the ellipse -/
structure EllipsePoint (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The theorem stating the properties of the ellipse and the ratio |BP|/|BQ| -/
theorem ellipse_properties (e : Ellipse) 
  (M : EllipsePoint e)
  (h_M : M.x = 2 ∧ M.y = 0)
  (h_ecc : Real.sqrt (e.a^2 - e.b^2) / e.a = 1/2)
  (A B : EllipsePoint e)
  (h_slopes : (A.y / A.x) * (B.y / B.x) = -3/4)
  (P : ℝ × ℝ)
  (h_P : P.1 = 3 * A.x ∧ P.2 = 3 * A.y)
  (Q : EllipsePoint e)
  (h_collinear : ∃ (t : ℝ), P.1 - B.x = t * (Q.x - B.x) ∧ P.2 - B.y = t * (Q.y - B.y)) :
  (e.a = 2 ∧ e.b = Real.sqrt 3) ∧ 
  ((P.1 - B.x)^2 + (P.2 - B.y)^2) / ((Q.x - B.x)^2 + (Q.y - B.y)^2) = 25 := by
  sorry

#check ellipse_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l619_61950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_max_cos_sum_l619_61922

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Properties of the triangle --/
def TriangleProperties (t : Triangle) : Prop :=
  0 < t.a ∧ 0 < t.b ∧ 0 < t.c ∧
  0 < t.A ∧ 0 < t.B ∧ 0 < t.C ∧
  t.A + t.B + t.C = Real.pi ∧
  t.b^2 + t.c^2 - t.a^2 = t.b * t.c ∧
  t.b = 2 ∧
  Real.sin t.C = 2 * Real.sin t.B

/-- The area of the triangle --/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  1/2 * t.b * t.c * Real.sin t.A

theorem triangle_area_and_max_cos_sum (t : Triangle) 
  (h : TriangleProperties t) : 
  triangleArea t = 2 * Real.sqrt 3 ∧ 
  ∃ (x : ℝ), ∀ (y : ℝ), Real.cos t.B + Real.cos t.C ≤ x ∧ x = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_max_cos_sum_l619_61922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percent_for_given_ratio_l619_61982

theorem profit_percent_for_given_ratio : 
  ∀ (x : ℝ), x > 0 →
  (5 * x - 4 * x) / (4 * x) * 100 = 25 := by
  intro x hx
  field_simp
  ring
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percent_for_given_ratio_l619_61982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_japanese_students_fraction_l619_61977

theorem japanese_students_fraction (j : ℝ) (j_pos : j > 0) : 
  (let senior_class := 2 * j
   let sophomore_class := (3/4) * j
   let seniors_japanese := (3/8) * senior_class
   let juniors_japanese := (1/4) * j
   let sophomores_japanese := (2/5) * sophomore_class
   let seniors_both := (1/6) * senior_class
   let juniors_both := (1/12) * j
   let sophomores_both := (1/10) * sophomore_class
   let total_students := j + senior_class + sophomore_class
   let total_japanese := (seniors_japanese - seniors_both) + 
                         (juniors_japanese - juniors_both) + 
                         (sophomores_japanese - sophomores_both)
   total_japanese / total_students) = 97 / 450 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_japanese_students_fraction_l619_61977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_eccentricity_l619_61919

/-- The line y = x + 2 -/
def line (x : ℝ) : ℝ := x + 2

/-- Point A, one of the foci -/
def A : ℝ × ℝ := (-1, 0)

/-- Point B, the other focus -/
def B : ℝ × ℝ := (1, 0)

/-- A point P on the line -/
noncomputable def P : ℝ × ℝ := sorry

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The eccentricity of the ellipse -/
noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

/-- The theorem stating the maximum eccentricity -/
theorem max_eccentricity :
  ∀ (P : ℝ × ℝ), 
  P.2 = line P.1 →  -- P is on the line y = x + 2
  ∃ (a : ℝ), a > 0 ∧ 
  distance P A + distance P B = 2 * a →  -- P is on the ellipse
  eccentricity a 1 ≤ Real.sqrt 10 / 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_eccentricity_l619_61919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_inequality_l619_61937

noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

theorem inverse_proportion_inequality (k x₁ x₂ y₁ y₂ : ℝ) 
  (hk : k < 0) 
  (hx : x₁ < x₂ ∧ x₂ < 0) 
  (hy₁ : y₁ = inverse_proportion k x₁)
  (hy₂ : y₂ = inverse_proportion k x₂) :
  y₂ > y₁ ∧ y₁ > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_inequality_l619_61937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_rearrangements_two_increasing_pairs_l619_61984

/-- The number of rearrangements of numbers from 1 to n with exactly two increasing pairs of consecutive elements -/
def P (n : ℕ) : ℕ := 3^n - (n+1) * 2^n + n*(n+1)/2

/-- Auxiliary function to represent the number of rearrangements with two increasing pairs -/
def number_of_rearrangements_with_two_increasing_pairs (n : ℕ) : ℕ := 
  -- This is a placeholder. In a real implementation, this would be defined or computed.
  0

/-- Theorem stating that P(n) correctly counts the number of rearrangements -/
theorem count_rearrangements_two_increasing_pairs (n : ℕ) :
  (number_of_rearrangements_with_two_increasing_pairs n) = P n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_rearrangements_two_increasing_pairs_l619_61984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_of_exponent_four_l619_61929

def f (n : ℕ) : ℕ := (Finset.range (n^2 - 3)).prod (λ i => i + 4)

def g (n : ℕ) : ℕ := (Finset.range n).prod (λ i => (i + 1)^2)

theorem base_of_exponent_four (n : ℕ) (h : n = 3) :
  ∃ (p : ℕ) (m : ℕ), 
    Nat.Prime p ∧ 
    (f n / g n).factorization p = 4 ∧ 
    p = 2 := by
  sorry

#eval f 3
#eval g 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_of_exponent_four_l619_61929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_l619_61928

open Real

theorem angle_relation (α β : ℝ) : 
  α ∈ Set.Ioo 0 π → 
  β ∈ Set.Ioo 0 π → 
  Real.tan (α - β) = 1/2 → 
  Real.tan β = -1/7 → 
  2*α - β = -3*π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_l619_61928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l619_61904

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := 9 * y^2 - 16 * x^2 = 144

-- Define the real semi-axis length
noncomputable def real_semi_axis : ℝ := 4

-- Define the imaginary semi-axis length
noncomputable def imaginary_semi_axis : ℝ := 3

-- Define the foci coordinates
def foci : Set (ℝ × ℝ) := {(0, -5), (0, 5)}

-- Define the eccentricity
noncomputable def eccentricity : ℝ := 5/4

-- Define the asymptote equations
def asymptotes (x y : ℝ) : Prop := y = (4/3) * x ∨ y = -(4/3) * x

theorem hyperbola_properties :
  (∀ x y, hyperbola x y → 
    real_semi_axis = 4 ∧
    imaginary_semi_axis = 3 ∧
    foci = {(0, -5), (0, 5)} ∧
    eccentricity = 5/4 ∧
    asymptotes x y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l619_61904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_is_60_l619_61992

/-- Represents a square-based pyramid container -/
structure PyramidContainer where
  base_side : ℝ
  height : ℝ

/-- Calculates the volume of water collected based on rainfall -/
noncomputable def water_volume (base_area : ℝ) (rainfall : ℝ) : ℝ :=
  base_area * rainfall

/-- Calculates the depth of water in the container -/
noncomputable def water_depth (container : PyramidContainer) (rainfall : ℝ) : ℝ :=
  let base_area := container.base_side ^ 2
  let water_vol := water_volume base_area rainfall
  let x := (3 * water_vol / (base_area * container.height)) ^ (1/3 : ℝ)
  container.height / x

/-- Theorem stating that for the given container and rainfall, the water depth is 60 cm -/
theorem water_depth_is_60 :
  let container := PyramidContainer.mk 23 120
  water_depth container 5 = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_is_60_l619_61992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_Y_at_pi_third_l619_61911

noncomputable def Y (x : ℝ) : ℝ := (Real.sin x - Real.cos x) / (2 * Real.cos x)

theorem derivative_Y_at_pi_third :
  deriv Y (π/3) = 2 := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_Y_at_pi_third_l619_61911
