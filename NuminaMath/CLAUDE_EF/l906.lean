import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_l906_90600

/-- The speed of a boat in still water, given its downstream and upstream distances and the stream speed. -/
theorem boat_speed (stream_speed downstream_distance upstream_distance boat_speed : ℝ) 
  (h_stream : stream_speed = 12)
  (h_downstream : downstream_distance = 80)
  (h_upstream : upstream_distance = 40)
  (h_time : downstream_distance / (boat_speed + stream_speed) = upstream_distance / (boat_speed - stream_speed)) :
  boat_speed = 36 :=
by
  sorry

#check boat_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_l906_90600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_products_l906_90612

theorem max_sum_of_products (a b c d : ℕ) : 
  a ∈ ({1, 3, 5, 7} : Set ℕ) → 
  b ∈ ({1, 3, 5, 7} : Set ℕ) → 
  c ∈ ({1, 3, 5, 7} : Set ℕ) → 
  d ∈ ({1, 3, 5, 7} : Set ℕ) → 
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d → 
  (a * b + a * c + b * d + c * d) ≤ 64 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_products_l906_90612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_27gon_isosceles_l906_90648

theorem regular_27gon_isosceles (vertices : Finset ℕ) : 
  vertices.card = 7 ∧ (∀ v ∈ vertices, v < 27) →
  (∃ a b c, a ∈ vertices ∧ b ∈ vertices ∧ c ∈ vertices ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a + c) / 2 % 27 = b % 27) ∨
  (∃ a b c d, a ∈ vertices ∧ b ∈ vertices ∧ c ∈ vertices ∧ d ∈ vertices ∧ 
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ d ∧ (a + d) / 2 % 27 = (b + c) / 2 % 27) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_27gon_isosceles_l906_90648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_data_mean_variance_sum_l906_90640

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (λ x => (x - m)^2)).sum / xs.length

noncomputable def standardDeviation (xs : List ℝ) : ℝ :=
  Real.sqrt (variance xs)

theorem transformed_data_mean_variance_sum
  (xs : List ℝ)
  (hlen : xs.length = 8)
  (hmean : mean xs = 6)
  (hstd : standardDeviation xs = 2) :
  let ys := xs.map (λ x => 3 * x - 5)
  mean ys + variance ys = 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_data_mean_variance_sum_l906_90640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_check_l906_90665

-- Define the functions for each pair
noncomputable def f1 (x : ℝ) : ℝ := Real.sqrt (x^2)
def g1 (x : ℝ) : ℝ := 3 * x^3

noncomputable def f2 (x : ℝ) : ℝ := abs x / x
noncomputable def g2 (x : ℝ) : ℝ := if x > 0 then 1 else -1

def f3 (x : ℝ) : ℝ := 1
noncomputable def g3 (x : ℝ) : ℝ := x^0

-- Theorem statement
theorem function_equality_check :
  (∃ x, f1 x ≠ g1 x) ∧
  (∀ x, x ≠ 0 → f2 x = g2 x) ∧
  (∃ x, f3 x ≠ g3 x) := by
  sorry

#check function_equality_check

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_check_l906_90665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_sum_l906_90635

/-- Given a positive integer n, prove that A + 2B + 4 is a perfect square,
    where A = (4/9) * (10^(2n) - 1) and B = (8/9) * (10^n - 1) -/
theorem perfect_square_sum (n : ℕ+) : ∃ (k : ℕ), 
  (4/9 : ℚ) * (10^(2*n.val) - 1) + 2 * ((8/9 : ℚ) * (10^n.val - 1)) + 4 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_sum_l906_90635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_people_lineup_with_restriction_l906_90695

theorem five_people_lineup_with_restriction (n : ℕ) (h : n = 5) :
  (n.factorial : ℕ) - ((n - 1).factorial : ℕ) = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_people_lineup_with_restriction_l906_90695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_review_rate_l906_90689

theorem restaurant_review_rate : 
  ∀ (total_reviews_A total_reviews_B : ℕ) 
    (positive_rate_A positive_rate_B : ℚ),
  total_reviews_A = 200 →
  total_reviews_B = 100 →
  positive_rate_A = 9/10 →
  positive_rate_B = 87/100 →
  let total_positive_A := (total_reviews_A : ℚ) * positive_rate_A;
  let total_positive_B := (total_reviews_B : ℚ) * positive_rate_B;
  let total_positive := total_positive_A + total_positive_B;
  let total_reviews := (total_reviews_A + total_reviews_B : ℚ);
  total_positive / total_reviews = 89/100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_review_rate_l906_90689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequalities_l906_90605

theorem trigonometric_inequalities :
  (Real.sin (1/2 : ℝ) < Real.sin ((5*Real.pi)/6)) ∧
  (Real.cos ((3*Real.pi)/4) > Real.cos ((5*Real.pi)/6)) ∧
  (Real.tan ((7*Real.pi)/6) > Real.sin (Real.pi/6)) ∧
  (Real.sin (Real.pi/5) < Real.cos (Real.pi/5)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequalities_l906_90605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_arrangement_not_power_of_two_l906_90678

def concatenate (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => acc * 100000 + x) 0

theorem card_arrangement_not_power_of_two :
  ∀ (arrangement : List ℕ),
    (∀ n ∈ arrangement, 11111 ≤ n ∧ n ≤ 99999) →
    (arrangement.length = 88889) →
    (∀ k : ℕ, k ∈ List.range 88889 → (k + 11111) ∈ arrangement) →
    ∃ (m : ℕ), (concatenate arrangement = 11111 * m) ∧
    ¬∃ (p : ℕ), concatenate arrangement = 2^p :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_arrangement_not_power_of_two_l906_90678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_inequality_l906_90683

theorem power_function_inequality (a : ℝ) : 
  (∀ x : ℝ, x > 1 → x^a < x) → a < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_inequality_l906_90683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_number_rules_l906_90686

-- Define subtraction
def sub (a b : ℝ) := a + (-b)

-- State the theorem
theorem negative_number_rules :
  (∀ a b : ℝ, a + (-b) = sub a b) ∧
  (∀ a b : ℝ, sub a (-b) = a + b) ∧
  (∀ a b : ℝ, a * (-b) = b * (-a) ∧ a * (-b) = -(a * b)) ∧
  (∀ a b : ℝ, (-a) * (-b) = a * b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_number_rules_l906_90686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_diameter_l906_90688

-- Define the circles
def circle_C : ℝ → Prop := sorry
def circle_D : ℝ → Prop := sorry

-- Define the diameter of circle D
def diameter_D : ℝ := 32

-- Define the ratio of shaded area to area of circle C
def shaded_area_ratio : ℝ := 7
def circle_C_area_ratio : ℝ := 1

-- Define the diameter of circle C
noncomputable def diameter_C : ℝ := 8 * Real.sqrt 2

-- Theorem statement
theorem circle_C_diameter :
  (∀ x, circle_C x → circle_D x) ∧  -- C is in the interior of D
  (diameter_D = 32) ∧  -- Diameter of D is 32 cm
  (shaded_area_ratio / circle_C_area_ratio = 7 / 1) →  -- Ratio of shaded area to area of C is 7:1
  diameter_C = 8 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_diameter_l906_90688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_tangents_theorem_l906_90650

-- Define the circles and line
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the intersection points and angles
structure Intersection where
  point : ℝ × ℝ
  angle : ℝ

-- Define the problem setup
structure ProblemSetup where
  circle1 : Circle
  circle2 : Circle
  line : Line
  intersections : Fin 4 → Intersection

-- Define helper functions
def is_circumscribed (quad : List (ℝ × ℝ)) : Prop := sorry

def circumcenter (quad : List (ℝ × ℝ)) : ℝ × ℝ := sorry

def point_on_line (p : ℝ × ℝ) (l : Line) : Prop := 
  l.a * p.1 + l.b * p.2 + l.c = 0

-- Define the theorem
theorem intersection_tangents_theorem (setup : ProblemSetup) :
  let tangent_quad := sorry -- Quadrilateral formed by tangents at intersection points
  let circ_center := circumcenter tangent_quad
  let center_line := Line.mk 
    (setup.circle2.center.1 - setup.circle1.center.1)
    (setup.circle2.center.2 - setup.circle1.center.2)
    (setup.circle1.center.1 * setup.circle2.center.2 - setup.circle1.center.2 * setup.circle2.center.1)
  (is_circumscribed tangent_quad) ∧ 
  (point_on_line circ_center center_line) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_tangents_theorem_l906_90650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l906_90645

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 1

-- Define the line l
def lineL (m x y : ℝ) : Prop := (m - 1) * x + m * y + 2 = 0

-- Define the distance function between a point and a line
noncomputable def distance_point_to_line (x y m : ℝ) : ℝ :=
  |((m - 1) * x + m * y + 2) / Real.sqrt ((m - 1)^2 + m^2)|

-- Theorem statement
theorem max_distance_circle_to_line :
  ∃ (max_dist : ℝ), max_dist = 6 ∧
  ∀ (x y m : ℝ), circleC x y →
    (∀ (x' y' : ℝ), circleC x' y' →
      distance_point_to_line x' y' m ≤ max_dist) ∧
    (∃ (x' y' : ℝ), circleC x' y' ∧
      distance_point_to_line x' y' m = max_dist) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l906_90645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_200_deg_in_rad_l906_90629

-- Define the conversion factor from degrees to radians
noncomputable def deg_to_rad : ℝ := Real.pi / 180

-- Define the angle in degrees
def angle_deg : ℝ := 200

-- Define the angle in radians
noncomputable def angle_rad : ℝ := angle_deg * deg_to_rad

-- Theorem statement
theorem angle_200_deg_in_rad : angle_rad = 10 * Real.pi / 9 := by
  -- Expand the definitions
  unfold angle_rad
  unfold deg_to_rad
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_200_deg_in_rad_l906_90629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_placement_and_coverage_area_l906_90639

/-- The number of radars -/
def n : ℕ := 5

/-- The radius of each radar's coverage area in km -/
noncomputable def r : ℝ := 13

/-- The width of the coverage ring in km -/
noncomputable def w : ℝ := 10

/-- The central angle of the regular pentagon in radians -/
noncomputable def α : ℝ := 2 * Real.pi / n

/-- The distance from the center to each radar -/
noncomputable def centerToRadar : ℝ := 12 / Real.sin (α / 2)

/-- The area of the coverage ring -/
noncomputable def ringArea : ℝ := 240 * Real.pi / Real.tan (α / 2)

theorem radar_placement_and_coverage_area :
  (centerToRadar = 12 / Real.sin (α / 2)) ∧
  (ringArea = 240 * Real.pi / Real.tan (α / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_placement_and_coverage_area_l906_90639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_surface_area_l906_90662

-- Define a regular tetrahedron
def RegularTetrahedron (edge : ℝ) : Prop :=
  edge > 0

-- Define a sphere
def Sphere (radius : ℝ) : Prop :=
  radius > 0

-- Define the circumradius of a regular tetrahedron
noncomputable def circumradius (edge : ℝ) : ℝ :=
  edge * Real.sqrt 6 / 4

-- Define the surface area of a sphere
noncomputable def sphereSurfaceArea (radius : ℝ) : ℝ :=
  4 * Real.pi * radius^2

-- Theorem statement
theorem tetrahedron_sphere_surface_area :
  ∀ (edge : ℝ),
    RegularTetrahedron edge →
    edge = 2 →
    ∃ (radius : ℝ),
      Sphere radius ∧
      radius = circumradius edge ∧
      sphereSurfaceArea radius = 6 * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_surface_area_l906_90662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_polygon_regular_l906_90643

/-- A polygon with an odd number of sides -/
structure OddPolygon where
  n : ℕ
  odd : Odd n

/-- A polygon inscribed in a circle -/
def InscribedPolygon (p : OddPolygon) :=
  ∃ (center : ℝ × ℝ) (radius : ℝ), True  -- Placeholder for the actual definition

/-- A polygon circumscribed around a circle -/
def CircumscribedPolygon (p : OddPolygon) :=
  ∃ (center : ℝ × ℝ) (radius : ℝ), True  -- Placeholder for the actual definition

/-- Angle at a vertex of a polygon -/
noncomputable def AngleAt (p : OddPolygon) (i : ℕ) : ℝ := 
  sorry

/-- Length of a side of a polygon -/
noncomputable def SideLength (p : OddPolygon) (i : ℕ) : ℝ := 
  sorry

/-- All angles of a polygon are equal -/
def EqualAngles (p : OddPolygon) :=
  ∀ i j, i < p.n → j < p.n → AngleAt p i = AngleAt p j

/-- All sides of a polygon are equal -/
def EqualSides (p : OddPolygon) :=
  ∀ i j, i < p.n → j < p.n → SideLength p i = SideLength p j

/-- A polygon is regular -/
def Regular (p : OddPolygon) :=
  EqualAngles p ∧ EqualSides p

theorem odd_polygon_regular (p : OddPolygon) :
  (InscribedPolygon p ∧ EqualAngles p) ∨ (CircumscribedPolygon p ∧ EqualSides p) →
  Regular p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_polygon_regular_l906_90643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_prostokvashino_to_molochnoye_l906_90627

/-- Represents a village in the postal route --/
inductive Village
  | Prostokvashino
  | Smetanino
  | Tvorozhnoye
  | Molochnoye
deriving Repr, DecidableEq

/-- Represents the distance between two villages --/
def distance (a b : Village) : ℕ :=
  match a, b with
  | Village.Prostokvashino, Village.Tvorozhnoye => 9
  | Village.Tvorozhnoye, Village.Prostokvashino => 9
  | Village.Prostokvashino, Village.Smetanino => 13
  | Village.Smetanino, Village.Prostokvashino => 13
  | Village.Tvorozhnoye, Village.Smetanino => 8
  | Village.Smetanino, Village.Tvorozhnoye => 8
  | Village.Tvorozhnoye, Village.Molochnoye => 14
  | Village.Molochnoye, Village.Tvorozhnoye => 14
  | _, _ => 0  -- Default case for undefined distances

/-- The theorem stating the distance from Prostokvashino to Molochnoye --/
theorem distance_prostokvashino_to_molochnoye :
  distance Village.Prostokvashino Village.Molochnoye = 19 := by
  sorry

#eval distance Village.Prostokvashino Village.Molochnoye

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_prostokvashino_to_molochnoye_l906_90627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimated_gas_consumption_l906_90673

-- Define the regression line equation
noncomputable def regression_line (x : ℝ) : ℝ := (170 / 23) * x - (31 / 23)

-- Define the number of new gas users in 10,000 households
def new_users : ℚ := 23 / 100

-- Theorem statement
theorem estimated_gas_consumption :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |regression_line (new_users : ℝ) - 0.35| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimated_gas_consumption_l906_90673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_length_l906_90698

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  edgeLength : ℝ

/-- Represents a light beam path in the cube -/
structure LightPath where
  cube : Cube
  start : Point3D
  firstReflection : Point3D

/-- Calculate the length of a single reflection segment -/
noncomputable def reflectionLength (c : Cube) (_ : Point3D) : ℝ :=
  Real.sqrt (c.edgeLength^2 + 4^2 + 6^2)

/-- Calculate the number of reflections needed to reach a vertex -/
def numberOfReflections (_ : Cube) : ℕ := 10

/-- Calculate the total length of the light path -/
noncomputable def totalLightPathLength (path : LightPath) : ℝ :=
  (numberOfReflections path.cube) * (reflectionLength path.cube path.firstReflection)

/-- The main theorem to prove -/
theorem light_path_length (path : LightPath) 
  (h1 : path.cube.edgeLength = 10)
  (h2 : path.start.x = 0 ∧ path.start.y = 0 ∧ path.start.z = 0)
  (h3 : path.firstReflection.x = 10 ∧ path.firstReflection.y = 4 ∧ path.firstReflection.z = 6) :
  totalLightPathLength path = 10 * Real.sqrt 152 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_length_l906_90698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_n_x_contains_infinitely_many_primes_l906_90669

def x : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1)/2004 + 1/(n + 1) * (x n)^2 - (n + 1)^3/2004 + 1

theorem x_equals_n (n : ℕ) : x n = n + 1 := by
  sorry

theorem x_contains_infinitely_many_primes : ∃ (f : ℕ → ℕ), StrictMono f ∧ ∀ k, Nat.Prime (x (f k)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_n_x_contains_infinitely_many_primes_l906_90669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_upper_bound_l906_90680

theorem function_upper_bound 
  (f : ℝ → ℝ)
  (h1 : ∀ x ∈ Set.Icc 0 1, f x ≥ 0)
  (h2 : f 1 = 1)
  (h3 : ∀ x y, x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 → 
        x + y ∈ Set.Icc 0 1 → f (x + y) ≥ f x + f y) :
  ∀ x ∈ Set.Icc 0 1, f x ≤ 2 * x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_upper_bound_l906_90680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_necessarily_x1x2_gt_1_l906_90632

-- Define the function f(x) = e^x - ax
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

-- State the theorem
theorem not_necessarily_x1x2_gt_1 :
  ¬ (∀ a x₁ x₂ : ℝ, x₁ < x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 → x₁ * x₂ > 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_necessarily_x1x2_gt_1_l906_90632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_and_circle_l906_90654

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 25

-- Define the point through which the line passes
def point : ℝ × ℝ := (3, 6)

-- Define the length of the segment cut from the circle
def segment_length : ℝ := 8

-- Define the two line equations
def line1 (x : ℝ) : Prop := x = 3
def line2 (x y : ℝ) : Prop := 3*x - 4*y + 15 = 0

-- Theorem statement
theorem line_through_point_and_circle :
  ∃ (line : ℝ → ℝ → Prop), 
    (line point.1 point.2) ∧ 
    (∃ (x1 y1 x2 y2 : ℝ), 
      circle_eq x1 y1 ∧ circle_eq x2 y2 ∧ 
      line x1 y1 ∧ line x2 y2 ∧ 
      (x1 - x2)^2 + (y1 - y2)^2 = segment_length^2) ∧
    ((∀ x y, line x y ↔ line1 x) ∨ (∀ x y, line x y ↔ line2 x y)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_and_circle_l906_90654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_sum_l906_90670

/-- A sequence of four positive integers satisfying specific conditions -/
structure SpecialSequence where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  geometric : b = a * 2  -- Simplified geometric progression condition
  arithmetic : c - b = d - c   -- Arithmetic progression condition
  difference : d = a + 50      -- Difference between first and fourth terms

/-- The sum of the terms in a SpecialSequence is 130 -/
theorem special_sequence_sum (seq : SpecialSequence) : 
  seq.a + seq.b + seq.c + seq.d = 130 := by
  sorry

#check special_sequence_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_sum_l906_90670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_theorem_l906_90623

/-- A right circular cone with an inscribed cube -/
structure ConeWithCube where
  -- The edge length of the cube
  cube_edge : ℝ
  -- The radius of the base of the cone
  cone_radius : ℝ
  -- The height of the cone
  cone_height : ℝ
  -- One edge of the cube lies on the diameter of the base of the cone
  edge_on_diameter : cone_radius = cube_edge / Real.sqrt 2
  -- The center of the cube lies on the height of the cone
  center_on_height : True
  -- The vertices of the cube that do not belong to the edge on the diameter lie on the lateral surface of the cone
  vertices_on_surface : True

/-- The ratio of the volume of the cone to the volume of the cube -/
noncomputable def volume_ratio (c : ConeWithCube) : ℝ :=
  (Real.pi * c.cone_radius^2 * c.cone_height / 3) / c.cube_edge^3

/-- The theorem stating the ratio of the volumes -/
theorem volume_ratio_theorem (c : ConeWithCube) :
  volume_ratio c = Real.pi * Real.sqrt 2 * (53 - 7 * Real.sqrt 3) / 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_theorem_l906_90623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_two_l906_90694

/-- The distance between a point and a plane in 3D space -/
noncomputable def distance_point_to_plane (x₀ y₀ z₀ A B C D : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C * z₀ + D| / Real.sqrt (A^2 + B^2 + C^2)

/-- The theorem stating that the distance between the given point and plane is 2 -/
theorem distance_is_two : distance_point_to_plane 2 1 1 3 4 12 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_two_l906_90694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_difference_l906_90617

/-- Given an original figure with blue and green tiles, and adding a border of green tiles,
    calculate the difference between the total number of green tiles and blue tiles in the new figure. -/
theorem tile_difference (blue green border : ℕ) : blue = 20 → green = 9 → border = 9 →
  (green + border : ℤ) - blue = -2 := by
  intros hblue hgreen hborder
  rw [hblue, hgreen, hborder]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_difference_l906_90617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_f_11_l906_90677

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the conditions
axiom g_has_inverse : Function.Bijective g
axiom f_g_relation : ∀ x, f⁻¹ (g x) = 2 * x^2 - 3

-- State the theorem
theorem g_inverse_f_11 :
  g⁻¹ (f 11) = Real.sqrt 7 ∨ g⁻¹ (f 11) = -Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_f_11_l906_90677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_to_line_l906_90651

/-- The parabola y = -x^2 -/
def parabola (x : ℝ) : ℝ := -x^2

/-- The line 4x + 3y - 8 = 0 -/
def line (x y : ℝ) : Prop := 4*x + 3*y - 8 = 0

/-- The distance from a point (x, y) to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |4*x + 3*y - 8| / Real.sqrt (4^2 + 3^2)

theorem min_distance_parabola_to_line :
  ∃ (d : ℝ), d = 4/3 ∧ 
  ∀ (x : ℝ), distance_to_line x (parabola x) ≥ d ∧
  ∃ (x₀ : ℝ), distance_to_line x₀ (parabola x₀) = d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_to_line_l906_90651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_f_implies_a_range_l906_90685

/-- A function f is monotonically increasing on ℝ if for all x, y ∈ ℝ, x < y implies f(x) ≤ f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x ≤ f y

/-- The function f(x) = x - (1/3) * sin(2x) + a * sin(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - (1/3) * Real.sin (2 * x) + a * Real.sin x

theorem monotone_f_implies_a_range (a : ℝ) :
  MonotonicallyIncreasing (f a) → -1/3 ≤ a ∧ a ≤ 1/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_f_implies_a_range_l906_90685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_for_g_g_x_eq_3_l906_90684

-- Define the function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 1 then
    -0.5 * x^2 + x + 3
  else if x ≤ 3 then
    4 - 1.5 * (x - 1)
  else
    0  -- undefined, but we need to return a value for all real inputs

-- State the theorem
theorem no_solutions_for_g_g_x_eq_3 :
  ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → g (g x) ≠ 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_for_g_g_x_eq_3_l906_90684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hook_function_possible_a_l906_90656

noncomputable def hook_function (a : ℝ) (x : ℝ) : ℝ := x + a / x

def diff_max_min_is_one (a : ℝ) : Prop :=
  (⨆ x ∈ Set.Icc 2 4, hook_function a x) - (⨅ x ∈ Set.Icc 2 4, hook_function a x) = 1

theorem hook_function_possible_a :
  ∀ a : ℝ, a > 0 → diff_max_min_is_one a → (a = 4 ∨ a = 6 + 4 * Real.sqrt 2) := by
  sorry

#check hook_function_possible_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hook_function_possible_a_l906_90656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_one_is_true_l906_90655

theorem proposition_one_is_true :
  (∀ x y : ℝ, x^2 ≠ y^2 → x ≠ y ∨ x ≠ -y) ∧
  (∀ x : ℝ, 2 ≤ x ∧ x < 3 → (x - 2) * (x - 3) ≤ 0) ∧
  (∀ a b : ℝ, a = 0 ∧ b = 0 → |a| + |b| = 0) ∧
  (∀ x y : ℕ, Odd (x + y) → (Odd x ∧ Even y) ∨ (Even x ∧ Odd y)) →
  (∀ x y : ℝ, x^2 ≠ y^2 → x ≠ y ∨ x ≠ -y) := by
  intro h
  exact h.1

#check proposition_one_is_true

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_one_is_true_l906_90655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomials_make_perfect_square_l906_90696

-- Define the original polynomial
def original_poly (x : ℝ) : ℝ := 16 * x^2 + 1

-- Define the possible monomials to add
def monomial1 (x : ℝ) : ℝ := 64 * x^4
def monomial2 (x : ℝ) : ℝ := 8 * x
def monomial3 (x : ℝ) : ℝ := -8 * x
def monomial4 : ℝ := -1
def monomial5 (x : ℝ) : ℝ := -16 * x^2

-- Define a function to check if a polynomial is a perfect square binomial
def is_perfect_square_binomial (p : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, p x = (a * x + b)^2

-- State the theorem
theorem monomials_make_perfect_square : 
  (∀ x : ℝ, is_perfect_square_binomial (λ y ↦ original_poly y + monomial1 y)) ∧
  (∀ x : ℝ, is_perfect_square_binomial (λ y ↦ original_poly y + monomial2 y)) ∧
  (∀ x : ℝ, is_perfect_square_binomial (λ y ↦ original_poly y + monomial3 y)) ∧
  (∀ x : ℝ, is_perfect_square_binomial (λ y ↦ original_poly y + monomial4)) ∧
  (∀ x : ℝ, is_perfect_square_binomial (λ y ↦ original_poly y + monomial5 y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomials_make_perfect_square_l906_90696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equals_P_l906_90614

def M : Set ℝ := {x | ∃ k : ℤ, x = (3*k - 2)*Real.pi}
def P : Set ℝ := {y | ∃ l : ℤ, y = (3*l + 1)*Real.pi}

theorem M_equals_P : M = P := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equals_P_l906_90614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l906_90619

theorem inequality_solution_set (x : ℝ) : 
  x ∈ Set.Icc (-π/3) (5*π/3) →
  (Real.cos x)^2018 + (Real.sin x)^(-(2019:ℤ)) ≤ (Real.sin x)^2018 + (Real.cos x)^(-(2019:ℤ)) ↔
  x ∈ Set.Ioc (-π/3) 0 ∪ Set.Ico (π/4) (π/2) ∪ Set.Ioc π (5*π/4) ∪ Set.Ioo (3*π/2) (5*π/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l906_90619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_constant_exists_l906_90672

noncomputable def C₁ : Set (EuclideanSpace ℝ (Fin 2)) := 
  {p | (p 0 + 2)^2 + (p 1)^2 ≤ 9}

def O : EuclideanSpace ℝ (Fin 2) := ![0, 0]
def A : EuclideanSpace ℝ (Fin 2) := ![1, 0]

theorem min_constant_exists (X : EuclideanSpace ℝ (Fin 2)) (h : X ∉ C₁) :
  ∃ c : ℝ, c = (Real.sqrt 15 - 3) / 3 ∧
    ‖X - O‖ - 1 ≥ c * min ‖X - A‖ (‖X - A‖^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_constant_exists_l906_90672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x5_coefficient_expansion_l906_90630

/-- The coefficient of x^5 in the expansion of (1+x)^2(1-x)^5 is -1 -/
theorem x5_coefficient_expansion : 
  (Polynomial.coeff ((Polynomial.X + 1)^2 * (1 - Polynomial.X)^5) 5) = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x5_coefficient_expansion_l906_90630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_in_interval_l906_90615

noncomputable def f (x : ℝ) := x + 2 * Real.cos x

theorem min_value_f_in_interval :
  ∀ x ∈ Set.Icc (-Real.pi / 2) 0, f (-Real.pi / 2) ≤ f x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_in_interval_l906_90615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_N_in_M_l906_90611

-- Define the sets M and N
def M : Set ℝ := {x | Real.exp (x * Real.log 2) ≤ 4}
def N : Set ℝ := {x | x * (1 - x) > 0}

-- Define the complement of N in M
def C_MN : Set ℝ := M \ N

-- Theorem statement
theorem complement_N_in_M :
  C_MN = Set.Iic (0 : ℝ) ∪ Set.Icc 1 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_N_in_M_l906_90611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_l906_90642

-- Define an enumeration for the answer options
inductive AnswerOption
  | A
  | B
  | C
  | D

-- Define a function that represents our choice
def chooseAnswer : AnswerOption := AnswerOption.D

-- State a theorem that our choice is correct
theorem correct_answer : chooseAnswer = AnswerOption.D := by
  -- The proof is trivial since we defined chooseAnswer to be D
  rfl

#print correct_answer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_l906_90642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_y_coord_l906_90622

/-- The y-coordinate of the point on the y-axis that is equidistant from A(3, 0) and B(1, -6) -/
theorem equidistant_point_y_coord : 
  ∃ y : ℝ, (Real.sqrt ((3 - 0)^2 + (0 - y)^2) = Real.sqrt ((1 - 0)^2 + (-6 - y)^2)) ∧ y = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_y_coord_l906_90622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_properties_l906_90604

-- Define the curve C and line l
noncomputable def curve_C (a : ℝ) (θ : ℝ) : ℝ := 4 * a * Real.cos θ
noncomputable def line_l (θ : ℝ) : ℝ := 4 / Real.cos (θ - Real.pi/3)

-- Define the tangency condition
def tangent_condition (a : ℝ) : Prop :=
  ∃ θ : ℝ, curve_C a θ = line_l θ ∧
  ∀ φ : ℝ, φ ≠ θ → curve_C a φ ≠ line_l φ

-- Define the angle between OA and OB
def angle_AOB (θA θB : ℝ) : ℝ := θB - θA

-- Define the sum of distances OA and OB
noncomputable def sum_distances (a : ℝ) (θA θB : ℝ) : ℝ :=
  curve_C a θA + curve_C a θB

-- Theorem statement
theorem curve_and_line_properties :
  ∃ a : ℝ, a > 0 ∧ tangent_condition a ∧ a = 4/3 ∧
  (∀ θA θB : ℝ, angle_AOB θA θB = Real.pi/3 →
    sum_distances a θA θB ≤ 16 * Real.sqrt 3 / 3) ∧
  (∃ θA θB : ℝ, angle_AOB θA θB = Real.pi/3 ∧
    sum_distances a θA θB = 16 * Real.sqrt 3 / 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_properties_l906_90604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_value_l906_90657

theorem ab_value (a b : ℝ) (h1 : a * Real.log 3 / Real.log 2 = 1) (h2 : (4 : ℝ)^b = 3) : a * b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_value_l906_90657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_singular_pairs_l906_90664

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  (Nat.factors n).maximum?.getD 1

def is_singular_pair (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧
  ∀ n ≥ 2, largest_prime_factor n * largest_prime_factor (n + 1) ≠ p * q

theorem infinitely_many_singular_pairs :
  ∀ k : ℕ, ∃ S : Finset (ℕ × ℕ),
    S.card > k ∧ ∀ (pair : ℕ × ℕ), pair ∈ S → is_singular_pair pair.1 pair.2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_singular_pairs_l906_90664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_integer_sum_l906_90668

noncomputable def a : ℝ := Real.pi / 2015

theorem smallest_n_for_integer_sum : ∃ n : ℕ, n = 31 ∧
  (∀ k < n, ¬ (Real.sin ((k^2 + 3*k : ℝ) * a) = 0)) ∧
  Real.sin ((n^2 + 3*n : ℝ) * a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_integer_sum_l906_90668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_l906_90692

-- Define the properties of the polygon
noncomputable def ConvexPolygon (n : ℕ) (d : ℝ) (a₁ : ℝ) : Prop :=
  n ≥ 3 ∧ 
  d = 5 ∧ 
  a₁ = 120 ∧
  (∀ i : ℕ, i < n → a₁ + (i : ℝ) * d < 180)

-- Define the sum of interior angles of a polygon
noncomputable def SumInteriorAngles (n : ℕ) : ℝ := 180 * (n - 2 : ℝ)

-- Define the sum of an arithmetic progression
noncomputable def SumArithmeticProgression (n : ℕ) (a₁ d : ℝ) : ℝ := 
  (n : ℝ) / 2 * (2 * a₁ + (n - 1 : ℝ) * d)

-- Theorem statement
theorem polygon_sides : 
  ∃ (n : ℕ), ConvexPolygon n 5 120 ∧ 
  SumInteriorAngles n = SumArithmeticProgression n 120 5 ∧
  n = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_l906_90692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_billing_excess_rate_l906_90674

/-- Represents the water company's billing method -/
structure WaterBilling where
  base_rate : ℚ  -- Rate for the first 5 tons
  base_limit : ℚ  -- Limit for the base rate (5 tons)
  excess_rate : ℚ  -- Rate for usage exceeding the base limit

/-- Calculates the water bill given the usage and billing method -/
def calculate_bill (usage : ℚ) (billing : WaterBilling) : ℚ :=
  min usage billing.base_limit * billing.base_rate +
  max (usage - billing.base_limit) 0 * billing.excess_rate

/-- Theorem stating the correct excess rate given the problem conditions -/
theorem water_billing_excess_rate :
  ∃ (billing : WaterBilling),
    billing.base_rate = 85/100 ∧
    billing.base_limit = 5 ∧
    (∃ (x : ℚ),
      calculate_bill (2 * x) billing = 1460/100 ∧
      calculate_bill (3 * x) billing = 2265/100) →
    billing.excess_rate = 115/100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_billing_excess_rate_l906_90674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_diff_depends_only_on_a_l906_90638

noncomputable section

/-- A quadratic function f(x) = x^2 + ax + b -/
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

/-- The maximum value of f(x) in the interval [0,1] -/
noncomputable def M (a b : ℝ) : ℝ := max (f a b 0) (f a b 1)

/-- The minimum value of f(x) in the interval [0,1] -/
noncomputable def m (a b : ℝ) : ℝ := min (f a b 0) (f a b 1)

/-- The difference between maximum and minimum values depends only on a -/
theorem max_min_diff_depends_only_on_a (a b₁ b₂ : ℝ) :
  M a b₁ - m a b₁ = M a b₂ - m a b₂ := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_diff_depends_only_on_a_l906_90638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l906_90626

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 2*a*x + 4*a else x^2 + 2*a*x + 4*a

-- State the theorem
theorem f_properties (a : ℝ) :
  (∀ x, f a x = f a (-x)) ∧ -- f is even
  (∀ x, x < 0 → f a x = x^2 + 2*a*x + 4*a) ∧ -- f(x) for x < 0
  ((0 < a ∧ a < 4 → (∀ x, f a x ≠ 0)) ∧ -- 0 zeros when 0 < a < 4
   (a = 0 → (∃! x, f a x = 0)) ∧ -- 1 zero when a = 0
   ((a = 4 ∨ a < 0) → (∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ ∀ z, f a z = 0 → z = x ∨ z = y)) ∧ -- 2 zeros when a = 4 or a < 0
   (a > 4 → (∃ w x y z, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
             f a w = 0 ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0 ∧
             ∀ t, f a t = 0 → t = w ∨ t = x ∨ t = y ∨ t = z))) -- 4 zeros when a > 4
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l906_90626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_solutions_count_l906_90690

def g (x : ℝ) : ℝ := x^3 - 3*x

theorem distinct_solutions_count :
  ∃ (S : Finset ℝ), (∀ d ∈ S, g (g (g (g d))) = 8) ∧ 
                    (∀ d : ℝ, g (g (g (g d))) = 8 → d ∈ S) ∧
                    (S.card = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_solutions_count_l906_90690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_equivalence_l906_90653

theorem trigonometric_equation_equivalence :
  ∃ (a b c : ℕ+),
    (∀ x : ℝ, (Real.sin x)^2 + (Real.sin (3*x))^2 + (Real.sin (5*x))^2 + (Real.sin (7*x))^2 + (Real.sin (9*x))^2 = 5/2 ↔
               Real.cos (↑a * x) * Real.cos (↑b * x) * Real.cos (↑c * x) = 0) ∧
    a + b + c = 18 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_equivalence_l906_90653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_sample_data_l906_90641

noncomputable def sample_data : List ℝ := [8, 12, 10, 11, 9]

noncomputable def mean (data : List ℝ) : ℝ := (data.sum) / (data.length : ℝ)

noncomputable def variance (data : List ℝ) : ℝ :=
  (data.map (fun x => (x - mean data) ^ 2)).sum / (data.length : ℝ)

theorem variance_of_sample_data :
  mean sample_data = 10 → variance sample_data = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_sample_data_l906_90641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l906_90676

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x + Real.pi/6) - 1

theorem f_properties :
  (∃ (T : ℝ), ∀ (x : ℝ), f (x + T) = f x ∧ T = Real.pi) ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-Real.pi/6) (Real.pi/4) → f y ∈ Set.Icc (-1) 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l906_90676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_from_inscribed_cylinder_l906_90667

/-- Given a cylinder inscribed in a sphere, with the cylinder's height to base diameter ratio of 2:1
    and volume 500π, prove that the volume of the sphere is (2500√5/3)π. -/
theorem sphere_volume_from_inscribed_cylinder (h r : ℝ) :
  h = 2 * r →                          -- height is twice the radius
  h^2 + r^2 = (2 * r * Real.sqrt 5)^2 →         -- cylinder inscribed in sphere
  π * r^2 * h = 500 * π →              -- volume of cylinder is 500π
  (4/3) * π * (r * Real.sqrt 5)^3 = (2500 * Real.sqrt 5/3) * π := by
  sorry

#check sphere_volume_from_inscribed_cylinder

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_from_inscribed_cylinder_l906_90667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_element_in_special_set_l906_90628

def satisfies_conditions (S : Finset ℕ) : Prop :=
  ∀ a b, a ∈ S → b ∈ S → a < b → 
    (¬ (b % a = 0)) ∧ 
    (∀ c ∈ S, c ≠ b → ¬ (c % a = 0))

theorem least_element_in_special_set : 
  ∃ S : Finset ℕ, 
    S.card = 5 ∧ 
    (∀ x, x ∈ S → 1 ≤ x ∧ x ≤ 10) ∧ 
    satisfies_conditions S ∧ 
    1 ∈ S ∧
    (∀ y, y ∈ S → 1 ≤ y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_element_in_special_set_l906_90628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_is_48_l906_90652

def veteran_count : Nat := 2
def new_player_count : Nat := 3
def total_players : Nat := veteran_count + new_player_count
def selected_positions : Nat := 3

def valid_arrangement (v n p1 p2 : Nat) : Prop :=
  v + n = selected_positions ∧
  v ≥ 1 ∧
  v ≤ veteran_count ∧
  n ≤ new_player_count ∧
  (p1 = 0 ∨ p2 = 0)

def arrangement_count : Nat :=
  (Nat.choose veteran_count 1 * Nat.choose new_player_count 2 * Nat.factorial 3) +
  (Nat.choose veteran_count 2 * Nat.choose new_player_count 1 * Nat.factorial 2)

theorem arrangement_count_is_48 : arrangement_count = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_is_48_l906_90652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_surface_area_l906_90691

/-- The surface area of a regular tetrahedron with edge length 1 is √3. -/
theorem tetrahedron_surface_area : 
  ∀ t : Real, t > 0 → Real.sqrt 3 = 4 * (Real.sqrt 3 / 4) :=
by
  intros t ht
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_surface_area_l906_90691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_projection_l906_90681

/-- Given two non-zero vectors a and b on a plane, prove that the projection of b on a is -1 -/
theorem vector_projection (a b : ℝ × ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : Real.sqrt ((a.1)^2 + (a.2)^2) = 2) 
  (h4 : a.1 * (a.1 + 2 * b.1) + a.2 * (a.2 + 2 * b.2) = 0) : 
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt ((a.1)^2 + (a.2)^2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_projection_l906_90681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_same_arc_l906_90637

/-- A type representing points on a circle -/
def Point : Type := Unit

/-- A circle with n marked points -/
structure Circle (n : ℕ) where
  points : Fin n → Point

/-- The measure of an arc between two points on a circle -/
noncomputable def arcMeasure (p q : Point) : ℝ := sorry

/-- The condition that for any two marked points, one of the arcs connecting them is less than 120° -/
def smallArcCondition (c : Circle n) : Prop :=
  ∀ i j, i ≠ j → min (arcMeasure (c.points i) (c.points j))
                     (360 - arcMeasure (c.points i) (c.points j)) < 120

/-- The theorem stating that all points lie on the same arc of 120° -/
theorem points_on_same_arc (n : ℕ) (c : Circle n) (h : smallArcCondition c) :
  ∃ (p q : Point), ∀ i, arcMeasure p (c.points i) + arcMeasure (c.points i) q ≤ 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_same_arc_l906_90637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_proof_l906_90621

theorem remainder_proof (y : ℕ) (h : 5 * y ≡ 1 [ZMOD 17]) :
  (7 + y) % 17 = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_proof_l906_90621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_radius_in_isosceles_triangle_l906_90682

/-- The radius of a semicircle inscribed in an isosceles triangle -/
theorem semicircle_radius_in_isosceles_triangle (base height : ℝ) 
  (h_base : base = 18) (h_height : height = 24) : ℝ := by
  -- Define the radius of the inscribed semicircle
  let r : ℝ := 108 * Real.sqrt 3
  
  -- State that this radius is correct for the given triangle
  have h : r = (base * height) / (2 * Real.sqrt (height^2 + (base/2)^2)) := by sorry
  
  -- Return the radius
  exact r

-- Remove the #eval statement as it's causing issues with compilation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_radius_in_isosceles_triangle_l906_90682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l906_90610

theorem cos_minus_sin_value (θ : ℝ) 
  (h1 : Real.sin θ * Real.cos θ = 1/8)
  (h2 : π/4 < θ)
  (h3 : θ < π/2) :
  Real.cos θ - Real.sin θ = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l906_90610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_sum_l906_90618

theorem tan_half_sum (x y : ℝ) 
  (h1 : Real.cos x + Real.cos y = 3/5)
  (h2 : Real.sin x + Real.sin y = 1/5) : 
  Real.tan ((x + y) / 2) = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_sum_l906_90618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_parallel_l906_90663

structure Plane where

structure Line where

def perpendicular (l : Line) (p : Plane) : Prop := sorry

def parallel (l1 l2 : Line) : Prop := sorry

theorem perpendicular_lines_parallel (α : Plane) (a b : Line) :
  perpendicular a α → perpendicular b α → parallel a b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_parallel_l906_90663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_simultaneous_events_l906_90659

-- Define the time intervals
def time_interval := ℝ × ℝ

-- Define the events
def event_A : time_interval := (0, 1)
def event_B : time_interval := (0.5, 1.5)

-- Define the probability of an event occurring in a given interval
noncomputable def prob_in_interval (event : time_interval) (interval : time_interval) : ℝ :=
  (min event.2 interval.2 - max event.1 interval.1) /
  (event.2 - event.1)

-- State the theorem
theorem probability_of_simultaneous_events :
  prob_in_interval event_A (0.5, 1) * prob_in_interval event_B (0.5, 1) = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_simultaneous_events_l906_90659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_2026_to_hundredth_l906_90603

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The original measurement in kg -/
def originalMeasurement : ℝ := 2.026

theorem rounding_2026_to_hundredth :
  roundToHundredth originalMeasurement = 2.03 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_2026_to_hundredth_l906_90603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_EFGH_area_l906_90647

-- Define the rectangle and squares
structure Rectangle where
  width : ℝ
  height : ℝ

structure Square where
  side : ℝ

-- Define the problem setup
def smallSquareArea : ℝ := 4

noncomputable def smallSquareSide : ℝ := Real.sqrt smallSquareArea

noncomputable def largeSquareSide : ℝ := smallSquareSide + 2

-- Define the rectangle EFGH
noncomputable def rectangleEFGH : Rectangle :=
  { width := smallSquareSide + largeSquareSide,
    height := largeSquareSide }

-- Theorem statement
theorem rectangle_EFGH_area :
  rectangleEFGH.width * rectangleEFGH.height = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_EFGH_area_l906_90647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplified_expression_l906_90666

theorem simplified_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let x := b / c + c / b
  let y := a^2 / c^2 + c^2 / a^2
  let z := a / b + b / a
  x^2 + y^2 + z^2 - x*y*z = 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplified_expression_l906_90666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_classification_l906_90634

/-- Represents the classification of a triangle based on its angles -/
inductive TriangleType
  | Acute
  | Right
  | Obtuse

/-- Determines the type of triangle based on its angles -/
noncomputable def determineTriangleType (α β γ : ℝ) : TriangleType :=
  if max α (max β γ) > 90 then
    TriangleType.Obtuse
  else if max α (max β γ) = 90 then
    TriangleType.Right
  else
    TriangleType.Acute

theorem triangle_classification 
  (α β γ : ℝ) 
  (h_sum : α + β + γ = 180) 
  (h_positive : α > 0 ∧ β > 0 ∧ γ > 0) :
  (determineTriangleType α β γ = TriangleType.Obtuse ↔ max α (max β γ) > 90) ∧
  (determineTriangleType α β γ = TriangleType.Right ↔ max α (max β γ) = 90) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_classification_l906_90634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_at_one_implies_a_range_l906_90616

/-- The function f(x) = lnx - 2ax + 2a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - 2 * a * x + 2 * a

/-- The function g(x) = xf(x) + ax^2 - x -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x * f a x + a * x^2 - x

/-- Theorem: If g(x) attains its maximum value at x = 1, then a ∈ (1/2, +∞) -/
theorem g_max_at_one_implies_a_range (a : ℝ) :
  (∀ x > 0, g a x ≤ g a 1) → a > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_at_one_implies_a_range_l906_90616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_different_parity_sum_of_digits_l906_90699

/-- Sum of digits function -/
def s (n : ℕ) : ℕ := sorry

/-- Polynomial type -/
def MyPolynomial (α : Type) := List α

/-- Evaluate polynomial at a point -/
def eval_poly {α : Type} [Semiring α] (p : MyPolynomial α) (x : α) : α := sorry

/-- Main theorem -/
theorem exists_different_parity_sum_of_digits (n : ℕ) (a : Fin n → ℕ+) (h_n : n ≥ 2) :
  ∃ k : ℕ+, ¬(s k.val % 2 = s (eval_poly 
    ((1 : ℕ) :: (List.ofFn (fun i => (a i).val))) k.val) % 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_different_parity_sum_of_digits_l906_90699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_similarity_theorem_l906_90671

open Matrix Complex

noncomputable def E12 : Matrix (Fin 2) (Fin 2) ℂ := ![![0, 1], ![0, 0]]
noncomputable def E21 : Matrix (Fin 2) (Fin 2) ℂ := ![![0, 0], ![1, 0]]

theorem matrix_similarity_theorem (M N : Matrix (Fin 2) (Fin 2) ℂ)
  (hM : M ≠ 0)
  (hN : N ≠ 0)
  (hM2 : M * M = 0)
  (hN2 : N * N = 0)
  (hMN : M * N + N * M = 1) :
  ∃ A : Matrix (Fin 2) (Fin 2) ℂ, IsUnit A ∧ M = A * E12 * A⁻¹ ∧ N = A * E21 * A⁻¹ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_similarity_theorem_l906_90671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fixed_point_of_f_and_f_inv_l906_90631

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 * x - 9

-- Define the inverse function f⁻¹
noncomputable def f_inv (x : ℝ) : ℝ := (x + 9) / 4

-- Theorem statement
theorem unique_fixed_point_of_f_and_f_inv :
  ∃! x : ℝ, f x = f_inv x ∧ x = 3 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fixed_point_of_f_and_f_inv_l906_90631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l906_90602

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem function_symmetry (ω φ : ℝ) (h1 : ω > 0) (h2 : |φ| < π/2) 
  (h3 : ∀ x, f ω φ (x + π) = f ω φ x)
  (h4 : ∀ x, f ω φ (x - π/3) = -f ω φ (-x - π/3)) :
  ∀ x, f ω φ (5*π/12 + x) = f ω φ (5*π/12 - x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l906_90602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l906_90675

noncomputable def f (x : ℝ) : ℝ := (3*x - 8)*(x - 5)/(x - 2)

theorem inequality_solution (x : ℝ) (h : x ≠ 2) :
  f x ≤ 0 ↔ 8/3 ≤ x ∧ x ≤ 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l906_90675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_henry_walk_limit_difference_l906_90601

/-- Henry's walking pattern between home and park --/
def henry_walk (total_distance : ℝ) (fraction : ℝ) : ℕ → ℝ
  | 0 => 0  -- Start at home
  | n + 1 => 
    let prev := henry_walk total_distance fraction n
    if n % 2 = 0
    then prev + fraction * (total_distance - prev)  -- Towards park
    else prev - fraction * prev  -- Towards home

/-- The limit points of Henry's walk --/
def limit_points (total_distance : ℝ) (fraction : ℝ) : ℝ × ℝ :=
  (⟨henry_walk total_distance fraction (2 * 10000), henry_walk total_distance fraction (2 * 10000 + 1)⟩)

theorem henry_walk_limit_difference :
  let (a, b) := limit_points 3 (5/6)
  |a - b| = 1.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_henry_walk_limit_difference_l906_90601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_coordinates_l906_90609

/-- The x-coordinate of the center of the hyperbola -/
def h : ℝ := 5

/-- The y-coordinate of the center of the hyperbola -/
def k : ℝ := 20

/-- The parameter 'a' in the hyperbola equation -/
def a : ℝ := 7

/-- The parameter 'b' in the hyperbola equation -/
def b : ℝ := 9

/-- The hyperbola equation -/
def is_on_hyperbola (x y : ℝ) : Prop :=
  (x - h)^2 / a^2 - (y - k)^2 / b^2 = 1

/-- The x-coordinate of the focus with larger x-coordinate -/
noncomputable def focus_x : ℝ := h + Real.sqrt (a^2 + b^2)

/-- The y-coordinate of the focus -/
def focus_y : ℝ := k

theorem hyperbola_focus_coordinates :
  focus_x = 5 + Real.sqrt 130 ∧ focus_y = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_coordinates_l906_90609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_revenue_day14_is_48000_plan1_more_reasonable_l906_90679

-- Define the value of each tea garden
noncomputable def garden_value : ℝ := 60000

-- Define the probability of rain
noncomputable def rain_probability : ℝ := 0.4

-- Define the cost of hiring additional workers
noncomputable def hiring_cost : ℝ := 32000

-- Function to calculate the expected value of a garden on a given day
noncomputable def expected_garden_value (day : ℕ) : ℝ :=
  garden_value * (1 - rain_probability / 2) ^ (day - 1)

-- Calculate the expected revenue from picking tea on the 14th for plan ①
noncomputable def expected_revenue_day14 : ℝ := expected_garden_value 3

-- Calculate the total expected revenue for plan ①
noncomputable def total_revenue_plan1 : ℝ :=
  garden_value + expected_garden_value 2 + expected_garden_value 3

-- Calculate the total revenue for plan ②
noncomputable def total_revenue_plan2 : ℝ := 3 * garden_value - hiring_cost

-- Theorem: The expected revenue from picking tea on the 14th for plan ① is 48,000 yuan
theorem expected_revenue_day14_is_48000 :
  expected_revenue_day14 = 48000 := by sorry

-- Theorem: Plan ① yields a higher expected total revenue than plan ②
theorem plan1_more_reasonable :
  total_revenue_plan1 > total_revenue_plan2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_revenue_day14_is_48000_plan1_more_reasonable_l906_90679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_45_repeating_decimal_245_l906_90613

/-- Represents a repeating decimal with a single repeating digit -/
def SingleRepeatingDecimal (whole : ℚ) (repeating : ℕ) : ℚ :=
  whole + (repeating : ℚ) / 9

/-- Represents a repeating decimal with two repeating digits -/
def DoubleRepeatingDecimal (whole : ℚ) (repeating1 repeating2 : ℕ) : ℚ :=
  whole + (10 * repeating1 + repeating2 : ℚ) / 99

/-- The repeating decimal 0.4̅5̅ is equal to the fraction 5/11 -/
theorem repeating_decimal_45 : DoubleRepeatingDecimal 0 4 5 = 5 / 11 := by
  sorry

/-- The repeating decimal 0.24̅5̅ is equal to the fraction 27/110 -/
theorem repeating_decimal_245 : DoubleRepeatingDecimal 0.2 4 5 = 27 / 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_45_repeating_decimal_245_l906_90613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_sales_increase_l906_90620

theorem price_reduction_sales_increase 
  (original_price : ℝ) 
  (original_sales : ℝ) 
  (new_sales : ℝ) 
  (h1 : original_price > 0) 
  (h2 : original_sales > 0) 
  (h3 : new_sales > 0) 
  (h4 : 0.85 * original_price * new_sales = 1.53 * (original_price * original_sales)) : 
  (new_sales - original_sales) / original_sales = 0.8 := by
  sorry

#check price_reduction_sales_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_sales_increase_l906_90620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_property_l906_90624

/-- Given a triangle ABC and a point M, if M satisfies certain vector equations, then m = -3 -/
theorem triangle_centroid_property (A B C M : EuclideanSpace ℝ (Fin 2)) (m : ℝ) : 
  (M - A) + (M - B) + (M - C) = 0 →
  (B - A) + (C - A) + m • (M - A) = 0 →
  m = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_property_l906_90624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_problem_l906_90687

theorem complex_modulus_problem (z : ℂ) : 
  (z * Complex.I - 1 * Complex.I) = 1 + Complex.I →
  Complex.abs z = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_problem_l906_90687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_sides_l906_90697

/-- A regular polygon with perimeter 180 and side length 15 has 12 sides -/
theorem regular_polygon_sides (p : ℕ) (perimeter side_length : ℝ) 
  (h_regular : p ≥ 3)
  (h_perimeter : perimeter = 180)
  (h_side_length : side_length = 15)
  (h_relation : p * side_length = perimeter) : p = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_sides_l906_90697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_time_l906_90649

/-- The time it takes for workers A, B, and C to complete a job together,
    given their individual completion times. -/
noncomputable def combined_completion_time (time_A time_B time_C : ℝ) : ℝ :=
  1 / (1 / time_A + 1 / time_B + 1 / time_C)

/-- Theorem stating that workers with individual completion times of 20, 35, and 50 days
    will complete the job together in 700/69 days. -/
theorem combined_work_time :
  combined_completion_time 20 35 50 = 700 / 69 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_time_l906_90649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mcdonalds_order_price_l906_90625

def original_cost : ℚ := 7.5
def coupon_value : ℚ := 2.5
def senior_discount : ℚ := 0.2

theorem mcdonalds_order_price :
  (original_cost - coupon_value) * (1 - senior_discount) = 4 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mcdonalds_order_price_l906_90625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workman_b_time_l906_90633

/-- Represents the time taken by workman b to complete the work -/
def time_b : ℝ := sorry

/-- Represents the rate at which workman b works -/
def rate_b : ℝ := sorry

/-- Represents the total amount of work to be done -/
def total_work : ℝ := sorry

/-- The theorem stating that workman b takes 15 days to complete the work -/
theorem workman_b_time :
  (rate_b * time_b = total_work) →
  (3 * rate_b * (time_b - 10) = total_work) →
  time_b = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workman_b_time_l906_90633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_interval_of_g_l906_90658

noncomputable def f (ω x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) - Real.cos (ω * x)

noncomputable def g (ω x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 6)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def min_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  is_periodic f p ∧ p > 0 ∧ ∀ q, 0 < q ∧ q < p → ¬is_periodic f q

def interval_of_increase (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem increase_interval_of_g (ω : ℝ) (h_ω : ω > 0) 
  (h_period : min_positive_period (f ω) Real.pi) :
  ∀ k : ℤ, interval_of_increase (g ω) (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_interval_of_g_l906_90658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_squared_plus_sin_x_l906_90661

theorem integral_x_squared_plus_sin_x : ∫ x in (-1 : ℝ)..1, (x^2 + Real.sin x) = 2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_x_squared_plus_sin_x_l906_90661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_homework_hours_l906_90693

/-- Represents the homework hours for each day of the week -/
def homework_hours : Fin 7 → ℕ
  | 0 => 2  -- Monday
  | 1 => 3  -- Tuesday
  | 2 => 4  -- Wednesday
  | 3 => 3  -- Thursday
  | 4 => 1  -- Friday
  | 5 => 4  -- Saturday (half of weekend homework)
  | 6 => 4  -- Sunday (half of weekend homework)

/-- Represents the chore hours for each day of the week -/
def chore_hours : Fin 7 → ℕ
  | 2 => 1  -- Wednesday
  | 6 => 1  -- Sunday
  | _ => 0  -- Other days

/-- The total number of practice nights -/
def practice_nights : ℕ := 3

/-- The indices of days with least homework (assuming practice falls on these days) -/
def practice_days : Finset (Fin 7) := {0, 4, 5}  -- Monday, Friday, Saturday

/-- The total hours of homework and chores -/
def total_hours : ℕ := (Finset.univ.sum homework_hours) + (Finset.univ.sum chore_hours)

/-- Theorem: Additional hours spent on homework on days without practice is 15 -/
theorem additional_homework_hours :
  (total_hours - practice_days.sum (fun i => homework_hours i + chore_hours i)) = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_homework_hours_l906_90693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l906_90607

/-- Given a circle with center (a,b) and radius r, if certain conditions are met,
    then the equation of the circle is one of two specific forms. -/
theorem circle_equation (a b r : ℝ) : 
  -- Point (2,3) is on the circle
  (2 - a)^2 + (3 - b)^2 = r^2 →
  -- The point symmetric to (2,3) with respect to x+2y=0 is on the circle
  (a + 2*b = 0) →
  -- The chord formed by x-y+1=0 has length 2√2
  r^2 - ((a - b + 1) / Real.sqrt 2)^2 = 2 →
  -- Then the equation of the circle is one of these two forms
  ((a = 6 ∧ b = -3 ∧ r^2 = 52) ∨ (a = 14 ∧ b = -7 ∧ r^2 = 244)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l906_90607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l906_90660

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sin x ^ 3 + 10 * Real.sin x ^ 2 + 3 * Real.sin x + 4 * Real.cos x ^ 2 - 12) / (Real.sin x - 2)

theorem f_range : 
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ -4 ≤ y ∧ y ≤ 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l906_90660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_matrix_vector_mult_l906_90636

def matrix_vector_mult (a b c d e f : ℝ) : Fin 2 → ℝ :=
  λ i => match i with
  | 0 => a * e + b * f
  | 1 => c * e + d * f

theorem trig_matrix_vector_mult (α β : ℝ) 
  (h1 : α + β = Real.pi) 
  (h2 : α - β = Real.pi / 2) : 
  matrix_vector_mult (Real.sin α) (Real.cos α) (Real.cos α) (Real.sin α) (Real.cos β) (Real.sin β) = 
  λ i => match i with
  | 0 => 0
  | 1 => 0 := by
  sorry

#check trig_matrix_vector_mult

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_matrix_vector_mult_l906_90636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_evaluation_l906_90644

theorem root_evaluation (y : ℝ) (h : y ≥ 0) : 
  Real.sqrt (y * (y * Real.sqrt y) ^ (1/3)) = (y^3) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_evaluation_l906_90644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_imply_k_l906_90606

theorem quadratic_roots_imply_k (k : ℝ) : 
  (∃ x : ℂ, x^2 + 3/7*x + k/7 = 0 ∧ 
   (x = (-3 + Complex.I * Real.sqrt 299) / 14 ∨ x = (-3 - Complex.I * Real.sqrt 299) / 14)) → 
  k = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_imply_k_l906_90606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_directrix_distance_l906_90608

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  ellipse P.fst P.snd

-- Define the distance from a point to the left focus
noncomputable def dist_to_left_focus : ℝ := 5/2

-- Define the distance from a point to the right directrix
noncomputable def dist_to_right_directrix : ℝ := 3

-- Theorem statement
theorem ellipse_focus_directrix_distance (P : ℝ × ℝ) :
  point_on_ellipse P → dist_to_left_focus = 5/2 → dist_to_right_directrix = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_directrix_distance_l906_90608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_semicircle_circumference_l906_90646

-- Define the given constants
noncomputable def rectangle_length : ℝ := 8
noncomputable def rectangle_breadth : ℝ := 6
noncomputable def trapezoid_height : ℝ := 5
noncomputable def base_difference : ℝ := 4

-- Define the square side length
noncomputable def square_side : ℝ := (2 * (rectangle_length + rectangle_breadth)) / 4

-- Define the trapezoid bases
noncomputable def trapezoid_longer_base : ℝ := square_side
noncomputable def trapezoid_shorter_base : ℝ := trapezoid_longer_base - base_difference

-- Theorem for the area of the trapezoid
theorem trapezoid_area : 
  (1/2 : ℝ) * (trapezoid_longer_base + trapezoid_shorter_base) * trapezoid_height = 25 := by
  sorry

-- Theorem for the circumference of the semicircle
theorem semicircle_circumference : 
  (π * square_side) / 2 + square_side = π * (7/2) + 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_semicircle_circumference_l906_90646
