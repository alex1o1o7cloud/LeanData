import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_bounds_l248_24822

-- Define the function f(x) = √(1 + x³)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 + x^3)

-- State the theorem
theorem integral_bounds :
  Continuous f ∧ MonotoneOn f (Set.Icc 0 2) →
  2 ≤ ∫ x in (0 : ℝ)..(2 : ℝ), f x ∧ ∫ x in (0 : ℝ)..(2 : ℝ), f x ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_bounds_l248_24822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_spherical_coords_l248_24871

/-- The radius of the circle formed by points with spherical coordinates (2, θ, π/3) is √3 --/
theorem circle_radius_spherical_coords :
  let r : ℝ → ℝ → ℝ → ℝ := λ ρ θ φ => Real.sqrt ((ρ * Real.sin φ * Real.cos θ)^2 + (ρ * Real.sin φ * Real.sin θ)^2)
  ∀ θ : ℝ, r 2 θ (π/3) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_spherical_coords_l248_24871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_domain_l248_24865

noncomputable def v (x : ℝ) := Real.sqrt (2 * x - 4) + (x - 5) ^ (1/4 : ℝ)

theorem v_domain : Set.Ici (5 : ℝ) = {x : ℝ | v x ∈ Set.univ} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_domain_l248_24865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_b_fill_time_l248_24845

/-- Represents the time (in hours) it takes for a pipe to fill or empty a tank -/
structure PipeTime where
  hours : ℝ
  isPositive : 0 < hours

/-- Represents the rate at which a pipe fills or empties a tank (fraction of tank per hour) -/
noncomputable def fillRate (t : PipeTime) : ℝ := 1 / t.hours

theorem pipe_b_fill_time 
  (pipe_a : PipeTime)
  (pipe_b : PipeTime)
  (pipe_c : PipeTime)
  (h1 : pipe_a.hours = 6)
  (h2 : pipe_c.hours = 12)
  (h3 : fillRate pipe_a + fillRate pipe_b - fillRate pipe_c = 1 / 3) :
  pipe_b.hours = 4 := by
  sorry

#check pipe_b_fill_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_b_fill_time_l248_24845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vba_player_count_l248_24859

/-- Represents the Vista Basketball Association --/
structure VBA where
  sock_cost : ℚ
  tshirt_cost : ℚ
  hat_cost : ℚ
  total_expenditure : ℚ

/-- Calculates the number of players in the VBA --/
noncomputable def number_of_players (vba : VBA) : ℚ :=
  vba.total_expenditure / (2 * (vba.sock_cost + vba.tshirt_cost + vba.hat_cost))

/-- Theorem stating the number of players in the VBA --/
theorem vba_player_count (vba : VBA) 
  (h1 : vba.sock_cost = 6)
  (h2 : vba.tshirt_cost = vba.sock_cost + 8)
  (h3 : vba.hat_cost = vba.tshirt_cost - 3)
  (h4 : vba.total_expenditure = 4950) : 
  number_of_players vba = 80 := by
    sorry

#check vba_player_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vba_player_count_l248_24859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_cost_savings_l248_24898

noncomputable def old_apartment_cost : ℝ := 1200
def price_increase_percentage : ℝ := 40
def number_of_people : ℕ := 3
def months_in_year : ℕ := 12

noncomputable def new_apartment_cost : ℝ := old_apartment_cost * (1 + price_increase_percentage / 100)
noncomputable def individual_share : ℝ := new_apartment_cost / number_of_people
noncomputable def monthly_savings : ℝ := old_apartment_cost - individual_share
noncomputable def yearly_savings : ℝ := monthly_savings * months_in_year

theorem apartment_cost_savings : yearly_savings = 7680 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_cost_savings_l248_24898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l248_24862

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (1 - Real.sqrt (3 - Real.sqrt (4 - x)))

-- Define the domain of g
def domain_g : Set ℝ := {x : ℝ | 1 - Real.sqrt (3 - Real.sqrt (4 - x)) ≥ 0}

-- Theorem stating that the domain of g is [-5, 0]
theorem domain_of_g : domain_g = Set.Icc (-5) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l248_24862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_room_size_l248_24883

/-- The length of the rectangular table -/
def table_length : ℝ := 12

/-- The width of the rectangular table -/
def table_width : ℝ := 9

/-- The diagonal length of the table -/
noncomputable def table_diagonal : ℝ := Real.sqrt (table_length ^ 2 + table_width ^ 2)

/-- The side length of the square room -/
def room_side : ℕ := 15

/-- Theorem stating the minimum room size requirement -/
theorem min_room_size :
  (∀ n : ℕ, n < room_side → (n : ℝ) < table_diagonal) ∧
  ((room_side : ℝ) ≥ table_diagonal) := by
  sorry

#eval room_side

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_room_size_l248_24883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_in_binary_l248_24850

/-- Converts a binary number represented as a list of bits to a natural number. -/
def binary_to_nat (bits : List Bool) : Nat :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Converts a natural number to its binary representation as a list of bits. -/
def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_bits (m : Nat) : List Bool :=
    if m = 0 then [] else
    (m % 2 = 1) :: to_bits (m / 2)
  to_bits n

theorem product_in_binary :
  let a := [true, false, true, true]  -- 1101₂
  let b := [true, true, true]         -- 111₂
  let product := [true, true, false, true, true, false, true]  -- 1011011₂
  binary_to_nat a * binary_to_nat b = binary_to_nat product :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_in_binary_l248_24850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l248_24869

theorem quadratic_inequality_solution_set 
  (f : ℝ → ℝ) (m n a b : ℝ) :
  (∀ x, f x = x^2 - m*x + n) →
  m > 0 →
  n > 0 →
  f a = 0 →
  f b = 0 →
  a ≠ b →
  ((∃ r : ℝ, r ∈ ({a, b, -1} : Set ℝ) ∧ 
    r + (r + (r + r)) ∈ ({a, b, -1} : Set ℝ) ∧ 
    r + r ∈ ({a, b, -1} : Set ℝ)) ∨
   (∃ r q : ℝ, r ∈ ({a, b, -1} : Set ℝ) ∧ 
    r * q ∈ ({a, b, -1} : Set ℝ) ∧ 
    r * q * q ∈ ({a, b, -1} : Set ℝ) ∧ 
    q ≠ 0 ∧ q ≠ 1)) →
  {x : ℝ | (x - m) / (x - n) ≥ 0} = {x : ℝ | x < 1 ∨ x ≥ 5/2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l248_24869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l248_24800

/-- The total distance between the foci of an ellipse -/
noncomputable def total_focal_distance (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

/-- Theorem: For an ellipse with semi-major axis length 4 and semi-minor axis length 3,
    the total distance between the foci is 2√7 -/
theorem ellipse_focal_distance :
  total_focal_distance 4 3 = 2 * Real.sqrt 7 := by
  -- Unfold the definition of total_focal_distance
  unfold total_focal_distance
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l248_24800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_exists_l248_24868

theorem equation_solution_exists : ∃ x : ℝ, x > 0 ∧ 
  (1 / 3) * (7 * x^2 - 3) = (x^2 - 70 * x - 20) * (x^2 + 35 * x + 7) ∧ 
  |x - 56.48| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_exists_l248_24868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_increasing_l248_24872

noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1 / Real.exp x

theorem f_is_odd_and_increasing :
  (∀ x, f (-x) = -f x) ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < 1 → f x < f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_increasing_l248_24872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_inequality_l248_24888

theorem negation_of_sin_inequality :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_inequality_l248_24888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_of_gyration_l248_24849

/-- The area of the ring between two concentric circles is 25π square inches.
    The length of the radius of gyration of the larger circle,
    which touches the smaller circle's perimeter twice at its endpoints, is √(100 + 4r²). -/
theorem radius_of_gyration (r R : ℝ) (h : R^2 - r^2 = 25) :
  (2*R)^2 = 100 + 4*r^2 := by
  -- Proof steps will go here
  sorry

#check radius_of_gyration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_of_gyration_l248_24849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_time_is_60_seconds_l248_24852

/-- Represents the problem of a train passing over a bridge -/
structure TrainBridgeProblem where
  train_length : ℝ  -- Length of the train in meters
  train_speed : ℝ   -- Speed of the train in km/hour
  bridge_length : ℝ -- Length of the bridge in meters

/-- Calculates the time (in seconds) for a train to pass a bridge -/
noncomputable def time_to_pass (problem : TrainBridgeProblem) : ℝ :=
  let total_distance := problem.train_length + problem.bridge_length
  let speed_in_mps := problem.train_speed * 1000 / 3600
  total_distance / speed_in_mps

/-- Theorem stating that for the given problem, the time to pass is 60 seconds -/
theorem train_bridge_time_is_60_seconds :
  let problem : TrainBridgeProblem := {
    train_length := 360,
    train_speed := 30,
    bridge_length := 140
  }
  time_to_pass problem = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_time_is_60_seconds_l248_24852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosC_value_max_area_l248_24835

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the condition from the problem
def condition (t : Triangle) : Prop :=
  (5 * t.a - 3 * t.b) * Real.cos t.C = 3 * t.c * Real.cos t.B

-- Part I: Prove cos C = 3/5
theorem cosC_value (t : Triangle) (h : condition t) : Real.cos t.C = 3/5 := by
  sorry

-- Helper function to calculate area
noncomputable def area (t : Triangle) : ℝ :=
  1/2 * t.a * t.b * Real.sin t.C

-- Part II: Prove maximum area occurs when a = b = 2√5
theorem max_area (t : Triangle) (h1 : t.c = 4) (h2 : Real.cos t.C = 3/5) :
  (∀ s : Triangle, s.c = 4 → area s ≤ area t) → t.a = 2 * Real.sqrt 5 ∧ t.b = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosC_value_max_area_l248_24835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l248_24838

-- Define the piecewise function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x ≤ -1 then x + 2
  else if -1 < x ∧ x < 2 then x^2
  else 0  -- This else case is added to make the function total

-- Theorem statement
theorem f_properties :
  (f 0 = 0) ∧
  (Set.range f = Set.Iio 4) ∧
  ({x : ℝ | f x < 1} = Set.Iio (-1) ∪ Set.Ioo (-1) 1) ∧
  ({x : ℝ | f x = 3 ∧ -1 < x ∧ x < 2} = {Real.sqrt 3}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l248_24838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_plane_l248_24837

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Represents a triangle in 3D space -/
structure Triangle3D where
  a : Point3D
  b : Point3D
  c : Point3D

/-- Represents a tetrahedron -/
structure Tetrahedron where
  p : Point3D
  base : Triangle3D

def isEquilateral (t : Triangle3D) : Prop := sorry

noncomputable def surfaceArea (s : Sphere) : ℝ := 4 * Real.pi * s.radius^2

def sideLength (t : Triangle3D) : ℝ := sorry

def distanceToPlane (point : Point3D) (plane : Triangle3D) : ℝ := sorry

def onSphere (p : Point3D) (s : Sphere) : Prop :=
  (p.x - s.center.x)^2 + (p.y - s.center.y)^2 + (p.z - s.center.z)^2 = s.radius^2

theorem max_distance_to_plane 
  (o : Sphere) 
  (tetra : Tetrahedron) 
  (h1 : onSphere tetra.p o)
  (h2 : onSphere tetra.base.a o)
  (h3 : onSphere tetra.base.b o)
  (h4 : onSphere tetra.base.c o)
  (h5 : isEquilateral tetra.base)
  (h6 : sideLength tetra.base = Real.sqrt 3)
  (h7 : surfaceArea o = 36 * Real.pi) :
  ∃ (p : Point3D), onSphere p o ∧ 
    distanceToPlane p tetra.base = 3 + 2 * Real.sqrt 2 ∧
    ∀ (q : Point3D), onSphere q o → 
      distanceToPlane q tetra.base ≤ 3 + 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_plane_l248_24837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_is_11_sqrt3_div_2_minus_7_l248_24873

/-- Represents the shape described in the problem -/
structure Shape where
  square : Set (ℝ × ℝ)
  triangles : Fin 2 → Set (ℝ × ℝ)
  added_squares : Fin 6 → Set (ℝ × ℝ)

/-- The region P formed by the union of the original square, triangles, and squares -/
def P (s : Shape) : Set (ℝ × ℝ) :=
  s.square ∪ (⋃ i, s.triangles i) ∪ (⋃ i, s.added_squares i)

/-- The smallest convex polygon Q that contains P -/
def Q (s : Shape) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Main theorem statement -/
theorem area_difference_is_11_sqrt3_div_2_minus_7 (s : Shape) 
    (h1 : area s.square = 1)
    (h2 : ∀ i, area (s.triangles i) = Real.sqrt 3 / 4)
    (h3 : ∀ i, area (s.added_squares i) = 1)
    (h4 : Disjoint s.square (⋃ i, s.triangles i))
    (h5 : Disjoint s.square (⋃ i, s.added_squares i))
    (h6 : Disjoint (⋃ i, s.triangles i) (⋃ i, s.added_squares i)) :
    area (Q s \ P s) = 11 * Real.sqrt 3 / 2 - 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_is_11_sqrt3_div_2_minus_7_l248_24873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_sixth_l248_24803

/-- A rectangle in the xy-plane --/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- The specific rectangle in the problem --/
def problemRectangle : Rectangle where
  x_min := 0
  x_max := 6
  y_min := 0
  y_max := 1
  h_x := by norm_num
  h_y := by norm_num

/-- The area of a rectangle --/
noncomputable def rectangleArea (r : Rectangle) : ℝ :=
  (r.x_max - r.x_min) * (r.y_max - r.y_min)

/-- The region where x < 2y within the rectangle --/
noncomputable def regionArea (r : Rectangle) : ℝ :=
  (min r.x_max (2 * r.y_max) - r.x_min) * (r.y_max - r.y_min) / 2

/-- The probability of x < 2y for a random point in the rectangle --/
noncomputable def probability (r : Rectangle) : ℝ :=
  regionArea r / rectangleArea r

theorem probability_is_one_sixth :
  probability problemRectangle = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_sixth_l248_24803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_size_related_to_subway_preference_p_minus_quarter_is_geometric_sequence_p5_greater_than_q5_l248_24854

-- Define the contingency table
structure ContingencyTable :=
  (a b c d : ℕ)

-- Define the chi-square statistic
noncomputable def chi_square (table : ContingencyTable) : ℝ :=
  let n := table.a + table.b + table.c + table.d
  (n : ℝ) * (table.a * table.d - table.b * table.c)^2 / 
    ((table.a + table.b) * (table.c + table.d) * (table.a + table.c) * (table.b + table.d))

-- Define the sequence of probabilities
noncomputable def p : ℕ → ℝ
| 0 => 1  -- Added case for 0
| 1 => 1
| 2 => 0
| n + 3 => -1/3 * p (n + 2) + 1/3

noncomputable def q (n : ℕ) : ℝ := 1/3 * (1 - p n)

-- Statement 1: City size is related to subway preference
theorem city_size_related_to_subway_preference (table : ContingencyTable) :
  chi_square table > 6.635 → true := by sorry

-- Statement 2: {pₙ - 1/4} forms a geometric sequence
theorem p_minus_quarter_is_geometric_sequence :
  ∀ n : ℕ, n ≥ 2 → p (n + 1) - 1/4 = -1/3 * (p n - 1/4) := by sorry

-- Statement 3: p₅ > q₅
theorem p5_greater_than_q5 : p 5 > q 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_size_related_to_subway_preference_p_minus_quarter_is_geometric_sequence_p5_greater_than_q5_l248_24854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_forms_two_cones_l248_24824

/-- A right-angled triangle -/
structure RightTriangle where
  base : ℝ
  height : ℝ
  hypotenuse : ℝ
  right_angle : base^2 + height^2 = hypotenuse^2

/-- A cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- The result of rotating a right-angled triangle 360° around its hypotenuse -/
def rotate_triangle (t : RightTriangle) : (Cone × Cone) :=
  sorry

/-- Theorem stating that rotating a right-angled triangle 360° around its hypotenuse forms two cones -/
theorem rotation_forms_two_cones (t : RightTriangle) :
  ∃ (c1 c2 : Cone), rotate_triangle t = (c1, c2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_forms_two_cones_l248_24824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_decimal_to_fraction_l248_24886

theorem periodic_decimal_to_fraction : (5 : ℚ) / 10 + (12 : ℚ) / 990 = 41 / 66 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_decimal_to_fraction_l248_24886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_approx_l248_24801

/-- The volume of lead in cubic decimeters -/
def lead_volume : ℝ := 2.2

/-- The diameter of the wire in centimeters -/
def wire_diameter : ℝ := 0.50

/-- Conversion factor from cubic decimeters to cubic centimeters -/
def dm3_to_cm3 : ℝ := 1000

/-- Conversion factor from centimeters to meters -/
def cm_to_m : ℝ := 0.01

/-- The length of the wire in meters -/
noncomputable def wire_length : ℝ :=
  let volume_cm3 := lead_volume * dm3_to_cm3
  let radius := wire_diameter / 2
  let length_cm := volume_cm3 / (Real.pi * radius^2)
  length_cm * cm_to_m

theorem wire_length_approx :
  ∃ ε > 0, |wire_length - 112.09| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_approx_l248_24801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_satisfies_conditions_l248_24842

/-- The equation of the given circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0

/-- The equation of the line on which the center of the circle lies -/
def center_line (x y : ℝ) : Prop := x - 2*y = 0

/-- The equation of the circle we want to prove -/
def target_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x - y + 1 = 0

theorem circle_satisfies_conditions :
  ∃ (h k : ℝ),
    (∀ x y, target_circle x y ↔ (x - h)^2 + (y - k)^2 = (h - 1)^2 + (k - 1)^2) ∧
    center_line h k ∧
    (∀ x y, target_circle x y → C₁ x y → x = 1 ∧ y = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_satisfies_conditions_l248_24842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l248_24881

noncomputable def g (x : ℝ) : ℝ := (x + 1) / (x^2 + x + 2)

theorem g_range : Set.range g = Set.Icc (-1/7 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l248_24881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exhibition_ratio_closest_to_one_l248_24814

/-- The ratio of adults to children at an exhibition that is closest to 1, given the admission fees and total collection. -/
theorem exhibition_ratio_closest_to_one :
  ∃ (a c : ℕ), 
    a ≥ 1 ∧ 
    c ≥ 1 ∧ 
    30 * a + 15 * c = 1950 ∧ 
    ∀ (a' c' : ℕ), 
      a' ≥ 1 → 
      c' ≥ 1 → 
      30 * a' + 15 * c' = 1950 → 
      |((a : ℚ) / c) - 1| ≤ |((a' : ℚ) / c') - 1| ∧
    a = 43 ∧ 
    c = 44 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exhibition_ratio_closest_to_one_l248_24814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l248_24806

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.exp (-x)

-- Define the second derivative of f
noncomputable def f_second_derivative (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f_second_derivative a x ≥ 2 * Real.sqrt 6) → a ≥ 6 := by
  intro h
  -- Proof goes here
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l248_24806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annipanni_has_winning_strategy_l248_24813

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a 100-digit number as a list of digits -/
def Number := List Digit

/-- Calculates the remainder when a number is divided by 11 -/
def remainderMod11 (n : Number) : ℕ :=
  (n.enum.foldl (fun acc (i, d) => acc + (if i % 2 = 0 then d.val else 11 - d.val)) 0) % 11

/-- Defines the game between Annipanni and Boribon -/
structure Game where
  firstMove : Digit
  strategy : Digit → Digit

/-- Determines if Annipanni wins given a game strategy -/
def annipaaniWins (g : Game) : Prop :=
  ∀ (opponentMoves : List Digit),
    opponentMoves.length = 49 →  -- Boribon makes 49 moves
    let finalNumber : Number := 
      (g.firstMove :: List.join (List.map (λ m => [m, g.strategy m]) opponentMoves))
    remainderMod11 finalNumber ≠ 5

/-- Main theorem: There exists a winning strategy for Annipanni -/
theorem annipanni_has_winning_strategy : 
  ∃ (g : Game), g.firstMove.val ≠ 0 ∧ annipaaniWins g := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_annipanni_has_winning_strategy_l248_24813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_granary_circumference_l248_24885

/-- Proves that a cylindrical granary with given height and volume has a specific base circumference -/
theorem cylindrical_granary_circumference :
  ∀ (height volume radius circumference : ℝ),
    height = 13 →
    volume = 1950 * 1.62 →
    volume = 3 * radius^2 * height →
    circumference = 2 * 3 * radius →
    ∃ (ε : ℝ), ε > 0 ∧ |circumference - 54| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_granary_circumference_l248_24885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l248_24878

/-- The time taken for a train to cross a man walking in the same direction --/
noncomputable def time_to_cross (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : ℝ :=
  train_length / (train_speed - man_speed)

/-- Conversion factor from km/hr to m/s --/
noncomputable def km_per_hr_to_m_per_s (speed : ℝ) : ℝ :=
  speed * 1000 / 3600

theorem train_crossing_time :
  let train_length : ℝ := 400
  let train_speed_km_hr : ℝ := 63
  let man_speed_km_hr : ℝ := 3
  let train_speed_m_s := km_per_hr_to_m_per_s train_speed_km_hr
  let man_speed_m_s := km_per_hr_to_m_per_s man_speed_km_hr
  let crossing_time := time_to_cross train_length train_speed_m_s man_speed_m_s
  ∃ ε > 0, |crossing_time - 24| < ε :=
by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval time_to_cross 400 (km_per_hr_to_m_s 63) (km_per_hr_to_m_s 3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l248_24878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_relation_l248_24839

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x + 2 * Real.cos x + 1

-- State the theorem
theorem function_relation (a b c : ℝ) :
  (∀ x : ℝ, a * f x + b * f (x - c) = 1) →
  b * Real.cos c / a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_relation_l248_24839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l248_24896

-- Define the points
def A : ℝ × ℝ := (2, 5)
def B : ℝ × ℝ := (-2, 5)  -- Reflection of A over y-axis
def C : ℝ × ℝ := (5, -2)  -- Reflection of B over y=x

-- Define the area of triangle ABC
def area_ABC : ℝ := 14

-- Theorem statement
theorem area_of_triangle_ABC : 
  (1/2) * Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) * |B.2 - C.2| = area_ABC := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l248_24896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l248_24809

open Real

theorem triangle_property (A B C a b c : ℝ) : 
  0 < A ∧ A < π ∧ 
  0 < B ∧ B < π ∧
  0 < C ∧ C < π/2 ∧
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / sin A = b / sin B ∧
  b / sin B = c / sin C ∧
  Real.sqrt 3 * a * cos ((A + C) / 2) = b * sin A →
  B = π/3 ∧ 
  (1 + Real.sqrt 3) / 2 < (a + b) / c ∧ (a + b) / c < 2 + Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l248_24809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_a_on_itself_l248_24874

noncomputable def a : ℝ × ℝ := (3, 4)
noncomputable def b : ℝ → ℝ × ℝ := λ t => (t, -6)

def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = k • w ∨ w = k • v

noncomputable def proj_vector (v w : ℝ × ℝ) : ℝ :=
  (v.1 * w.1 + v.2 * w.2) / (w.1^2 + w.2^2)

theorem projection_of_a_on_itself (t : ℝ) :
  collinear a (b t) → proj_vector a a = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_a_on_itself_l248_24874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_MN_MP_l248_24846

-- Define the line l: y = x + 1
def line_l (x y : ℝ) : Prop := y = x + 1

-- Define the circle C: (x-4)^2 + y^2 = 16
def circle_C (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 16

-- Define point M as the intersection of line l with x-axis
def point_M : ℝ × ℝ := (-1, 0)

-- Define point N as the intersection of line l with y-axis
def point_N : ℝ × ℝ := (0, 1)

-- Define points A and B as intersections of line l with circle C
def point_A : ℝ × ℝ := (0, 1)
def point_B : ℝ × ℝ := (3, 4)

-- Define point P as the midpoint of AB
noncomputable def point_P : ℝ × ℝ := ((point_A.1 + point_B.1) / 2, (point_A.2 + point_B.2) / 2)

-- Define vectors MN and MP
noncomputable def vector_MN : ℝ × ℝ := (point_N.1 - point_M.1, point_N.2 - point_M.2)
noncomputable def vector_MP : ℝ × ℝ := (point_P.1 - point_M.1, point_P.2 - point_M.2)

-- Theorem statement
theorem dot_product_MN_MP :
  vector_MN.1 * vector_MP.1 + vector_MN.2 * vector_MP.2 = 5 := by
  -- Expand the definitions
  unfold vector_MN vector_MP point_M point_N point_P point_A point_B
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_MN_MP_l248_24846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_inequality_l248_24861

/-- A convex quadrilateral with side lengths a, b, c, d and diagonal lengths e and f -/
structure ConvexQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  convex : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0

/-- The property of being a rhombus -/
def is_rhombus (q : ConvexQuadrilateral) : Prop :=
  q.a = q.b ∧ q.b = q.c ∧ q.c = q.d

theorem quadrilateral_diagonal_inequality (q : ConvexQuadrilateral) :
  2 * min (min (min q.a q.b) q.c) q.d ≤ Real.sqrt (q.e^2 + q.f^2) ∧
  (2 * min (min (min q.a q.b) q.c) q.d = Real.sqrt (q.e^2 + q.f^2) ↔ is_rhombus q) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_inequality_l248_24861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_next_sales_is_30_million_l248_24817

/-- Represents the sales and profit data for a technology company --/
structure SalesData where
  initialSales : ℚ
  initialProfit : ℚ
  nextProfit : ℚ
  profitRatioIncrease : ℚ

/-- Calculates the amount of next sales based on the given sales data --/
def calculateNextSales (data : SalesData) : ℚ :=
  (data.nextProfit * data.initialSales) / (data.initialProfit * (1 + data.profitRatioIncrease))

/-- Theorem stating that the next sales amount is 30 million given the specified conditions --/
theorem next_sales_is_30_million (data : SalesData) 
  (h1 : data.initialSales = 15)
  (h2 : data.initialProfit = 5)
  (h3 : data.nextProfit = 12)
  (h4 : data.profitRatioIncrease = 1 / 5) :
  calculateNextSales data = 30 := by
  sorry

def main : IO Unit := do
  let result := calculateNextSales { 
    initialSales := 15, 
    initialProfit := 5, 
    nextProfit := 12, 
    profitRatioIncrease := 1 / 5
  }
  IO.println s!"The next sales amount is: {result}"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_next_sales_is_30_million_l248_24817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equation_l248_24821

theorem factorial_equation : ∃ n : ℕ, 2^7 * 3^3 * n = Nat.factorial 10 ∧ n = 525 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equation_l248_24821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_is_two_l248_24851

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle --/
noncomputable def area (t : Triangle) : ℝ := (1/2) * t.a * t.b * sin t.C

/-- The conditions given in the problem --/
def satisfies_conditions (t : Triangle) : Prop :=
  t.a = 2 ∧ 3 * t.b * sin t.C = 5 * t.c * sin t.B * cos t.A

/-- The theorem statement --/
theorem max_area_is_two :
  ∃ (t : Triangle), satisfies_conditions t ∧
    ∀ (t' : Triangle), satisfies_conditions t' → area t' ≤ 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_is_two_l248_24851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_calculation_l248_24830

/-- Represents the daily savings amount in cents -/
def daily_savings : ℕ → Prop := sorry

/-- Represents the total savings after 20 days in dimes -/
def total_savings : ℕ → Prop := sorry

/-- Theorem stating that if a person saves a constant amount daily for 20 days
    and accumulates 2 dimes in total, then the daily savings amount is 1 cent -/
theorem savings_calculation (x : ℕ) :
  daily_savings x →
  total_savings 2 →
  x = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_calculation_l248_24830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_remainder_l248_24855

/-- The sequence starts with 3 and increases by 7 each time -/
def sequence_term (n : ℕ) : ℕ := 3 + 7 * (n - 1)

/-- The last term of the sequence -/
def last_term : ℕ := 304

/-- The number of terms in the sequence -/
def num_terms : ℕ := (last_term - 3) / 7 + 1

/-- The sum of the sequence -/
def sequence_sum : ℕ := (num_terms * (3 + last_term)) / 2

theorem sequence_sum_remainder (h : sequence_term num_terms = last_term) : 
  sequence_sum % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_remainder_l248_24855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l248_24891

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

-- State the theorem
theorem f_is_odd : ∀ x ∈ Set.Ioo (-1 : ℝ) 1, f (-x) = -f x := by
  -- The proof is omitted and replaced with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l248_24891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_locus_l248_24820

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the fixed points A and B
def point_A : ℝ × ℝ := (0, 1)
def point_B : ℝ × ℝ := (0, -1)

-- Define a parabola passing through A and B with directrix tangent to the circle
def parabola_through_AB_tangent_to_circle (f : ℝ × ℝ) (d : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), circle_eq a b ∧
  (∀ (x y : ℝ), d x y = a*x + b*y - 4) ∧
  (∀ (p : ℝ × ℝ), p = point_A ∨ p = point_B → 
    (p.1 - f.1)^2 + (p.2 - f.2)^2 = (d p.1 p.2)^2 / (a^2 + b^2))

-- Theorem statement
theorem focus_locus (x y : ℝ) (h : x ≠ 0) :
  (∃ (f : ℝ × ℝ) (d : ℝ → ℝ → ℝ), 
    parabola_through_AB_tangent_to_circle f d ∧ f = (x, y)) ↔ 
  x^2/3 + y^2/4 = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_locus_l248_24820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_y_l248_24840

noncomputable def y (x : ℝ) : ℝ := Real.cos (Real.sin 3) ^ 2 + (Real.sin (29 * x) ^ 2) / (29 * Real.cos (58 * x))

theorem derivative_y (x : ℝ) :
  deriv y x = Real.tan (58 * x) / Real.cos (58 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_y_l248_24840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_property_l248_24887

/-- Represents a rational function -/
structure RationalFunction where
  numerator : Polynomial ℝ
  denominator : Polynomial ℝ

/-- Counts the number of holes in a rational function -/
noncomputable def count_holes (f : RationalFunction) : ℕ := sorry

/-- Counts the number of vertical asymptotes of a rational function -/
noncomputable def count_vertical_asymptotes (f : RationalFunction) : ℕ := sorry

/-- Counts the number of horizontal asymptotes of a rational function -/
noncomputable def count_horizontal_asymptotes (f : RationalFunction) : ℕ := sorry

/-- Counts the number of oblique asymptotes of a rational function -/
noncomputable def count_oblique_asymptotes (f : RationalFunction) : ℕ := sorry

/-- The main theorem stating the property of the given rational function -/
theorem rational_function_property : 
  let f : RationalFunction := {
    numerator := Polynomial.X^2 - Polynomial.X - 2,
    denominator := Polynomial.X^3 - 3*Polynomial.X^2 + 2*Polynomial.X
  }
  let p := count_holes f
  let q := count_vertical_asymptotes f
  let r := count_horizontal_asymptotes f
  let s := count_oblique_asymptotes f
  p + 2*q + 3*r + 4*s = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_property_l248_24887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_coverage_l248_24805

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The area covered by a set of circles -/
noncomputable def area_covered (circles : Set Circle) : ℝ := sorry

/-- Predicate to check if two circles overlap -/
def overlaps (c1 c2 : Circle) : Prop := sorry

theorem circle_coverage (M : ℝ) (circles : Set Circle) :
  (area_covered circles = M) →
  (∃ (subset : Set Circle), subset ⊆ circles ∧
    (∀ c1 c2, c1 ∈ subset → c2 ∈ subset → c1 ≠ c2 → ¬(overlaps c1 c2)) ∧
    area_covered subset ≥ M / 9) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_coverage_l248_24805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_a_value_l248_24895

-- Define the hyperbola
noncomputable def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 4 = 1

-- Define the eccentricity
noncomputable def eccentricity (a : ℝ) : ℝ :=
  Real.sqrt (a^2 + 4) / a

-- Theorem statement
theorem hyperbola_a_value :
  ∀ a : ℝ, 
    a > 0 → 
    eccentricity a = Real.sqrt 5 / 2 → 
    a = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_a_value_l248_24895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_derivative_bound_l248_24894

noncomputable def f (x : ℝ) : ℝ := if x ≠ 0 then Real.sin x / x else 1

theorem nth_derivative_bound (n : ℕ) (x : ℝ) (h : x > 0) :
  ∃ (f_nth_deriv : ℝ → ℝ), (∀ (y : ℝ), y > 0 → HasDerivAt f_nth_deriv (deriv^[n] f y) y) ∧
    |f_nth_deriv x| ≤ 1 / (n + 1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_derivative_bound_l248_24894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l248_24884

/-- The distance between two points in 3D space -/
noncomputable def distance (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

/-- The coordinates of point A -/
noncomputable def A : ℝ × ℝ × ℝ := (0, 0, -13/3)

/-- The coordinates of point B -/
def B : ℝ × ℝ × ℝ := (7, 0, -15)

/-- The coordinates of point C -/
def C : ℝ × ℝ × ℝ := (2, 10, -12)

/-- Theorem: Point A is equidistant from points B and C -/
theorem equidistant_point :
  distance A.1 A.2.1 A.2.2 B.1 B.2.1 B.2.2 =
  distance A.1 A.2.1 A.2.2 C.1 C.2.1 C.2.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l248_24884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_9_is_99_l248_24833

/-- An arithmetic sequence with specific conditions -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_147 : a 1 + a 4 + a 7 = 39
  sum_369 : a 3 + a 6 + a 9 = 27

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1) * (seq.a 2 - seq.a 1))

/-- The theorem to be proved -/
theorem sum_9_is_99 (seq : ArithmeticSequence) : sum_n seq 9 = 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_9_is_99_l248_24833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l248_24841

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2 - Real.sqrt (-x^2 + 4*x)

-- Define the domain
def domain (x : ℝ) : Prop := -x^2 + 4*x ≥ 0

-- Theorem statement
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, domain x ∧ f x = y) ↔ 0 ≤ y ∧ y ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l248_24841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_composition_has_no_roots_l248_24892

-- Define the quadratic trinomials f and g
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c
def g (d e : ℝ) (x : ℝ) : ℝ := x^2 + d*x + e

-- Define the property of having no real roots
def has_no_real_roots (h : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, h x ≠ 0

-- Main theorem
theorem at_least_one_composition_has_no_roots
  (b c d e : ℝ)
  (h1 : has_no_real_roots (λ x ↦ f b c (g d e x)))
  (h2 : has_no_real_roots (λ x ↦ g d e (f b c x))) :
  has_no_real_roots (λ x ↦ f b c (f b c x)) ∨
  has_no_real_roots (λ x ↦ g d e (g d e x)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_composition_has_no_roots_l248_24892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_hexagon_area_l248_24819

/-- A hexagon with the following properties:
  - Six equal-length sides
  - Adjacent angles are either 120° or 60°
  - Two distinct sets of parallel pairs of sides
  - Shorter sides have length 1
-/
structure SpecialHexagon where
  -- We don't need to define all properties explicitly,
  -- as some are implied by others
  shortSideLength : ℝ
  shortSideLength_eq_one : shortSideLength = 1

/-- The area of the special hexagon -/
noncomputable def hexagonArea (h : SpecialHexagon) : ℝ := 9 * Real.sqrt 3 / 4

/-- Theorem stating that the area of the special hexagon is 9√3/4 -/
theorem special_hexagon_area (h : SpecialHexagon) :
  hexagonArea h = 9 * Real.sqrt 3 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_hexagon_area_l248_24819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_product_8_factorial_l248_24863

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem min_sum_of_product_8_factorial (a b c d : ℕ+) : 
  a * b * c * d = factorial 8 → 
  (∀ w x y z : ℕ+, w * x * y * z = factorial 8 → a + b + c + d ≤ w + x + y + z) →
  a + b + c + d = 61 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_product_8_factorial_l248_24863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l248_24811

-- Define the equation
noncomputable def equation (x : ℂ) : ℂ := x^3 + 4*x^2*Real.sqrt 3 + 12*x + 8*Real.sqrt 3 + 2*x + Real.sqrt 3

-- State the theorem
theorem equation_solutions :
  let sol₁ : ℂ := -Real.sqrt 3
  let sol₂ : ℂ := -Real.sqrt 3 + Complex.I * Real.sqrt 2
  let sol₃ : ℂ := -Real.sqrt 3 - Complex.I * Real.sqrt 2
  (equation sol₁ = 0) ∧ (equation sol₂ = 0) ∧ (equation sol₃ = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l248_24811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gorilla_exhibit_visitors_l248_24880

def visitors : List ℕ := [50, 70, 90, 100, 70, 60, 80, 50]

def gorilla_percentages : List ℚ := [
  80/100, 75/100, 60/100, 40/100, 55/100, 70/100, 60/100, 80/100
]

def gorilla_visitors (visitors : List ℕ) (percentages : List ℚ) : ℚ :=
  List.sum (List.zipWith (λ v p ↦ (v : ℚ) * p) visitors percentages)

theorem gorilla_exhibit_visitors :
  gorilla_visitors visitors gorilla_percentages = 355 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gorilla_exhibit_visitors_l248_24880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_segment_length_is_4_8_l248_24858

/-- An isosceles triangle with base 8 and legs 12 -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  is_isosceles : base = 8 ∧ leg = 12

/-- The length of the segment connecting the points where the angle bisectors 
    of the angles at the base intersect the legs of the triangle -/
noncomputable def angle_bisector_segment_length (t : IsoscelesTriangle) : ℝ := 
  24 / 5

/-- Theorem stating that the length of the segment connecting the points where 
    the angle bisectors of the angles at the base intersect the legs of the 
    isosceles triangle with base 8 and legs 12 is 4.8 -/
theorem angle_bisector_segment_length_is_4_8 (t : IsoscelesTriangle) : 
  angle_bisector_segment_length t = 4.8 := by
  unfold angle_bisector_segment_length
  norm_num
  
#check angle_bisector_segment_length_is_4_8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_segment_length_is_4_8_l248_24858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_half_in_second_quadrant_l248_24877

open Real

-- Define the property of being in the fourth quadrant
def is_in_fourth_quadrant (θ : ℝ) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi + 3 * Real.pi / 2 ≤ θ ∧ θ ≤ 2 * k * Real.pi + 2 * Real.pi

-- Define the property of being in the second quadrant
def is_in_second_quadrant (θ : ℝ) : Prop :=
  ∃ k : ℤ, k * Real.pi + Real.pi / 2 < θ ∧ θ < k * Real.pi + Real.pi

-- State the theorem
theorem angle_half_in_second_quadrant (θ : ℝ) :
  is_in_fourth_quadrant θ → |Real.cos (θ / 2)| = -Real.cos (θ / 2) →
  is_in_second_quadrant (θ / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_half_in_second_quadrant_l248_24877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l248_24882

theorem trigonometric_identities (α : Real) 
  (h1 : Real.sin α = Real.sqrt 5 / 5) 
  (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.tan (Real.pi + α) + Real.sin (Real.pi / 2 - α) / Real.cos (Real.pi - α) = -1 / 2 ∧ 
  Real.tan (α + Real.pi / 4) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l248_24882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_intervals_f_increasing_intervals_l248_24808

noncomputable def f' (x : ℝ) := 2 * Real.cos (2 * x + Real.pi / 6)

def monotonic_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

theorem f_monotonic_intervals :
  (∀ x ∈ Set.Icc 0 (Real.pi / 6), f' x ≥ 0) ∧
  (∀ x ∈ Set.Icc (2 * Real.pi / 3) Real.pi, f' x ≥ 0) :=
sorry

theorem f_increasing_intervals :
  (monotonic_increasing f 0 (Real.pi / 6)) ∧
  (monotonic_increasing f (2 * Real.pi / 3) Real.pi) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_intervals_f_increasing_intervals_l248_24808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l248_24802

/-- The circle represented by the equation x^2 + y^2 + 2x - 4y + 1 = 0 -/
def my_circle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

/-- The line represented by the equation 2ax - by + 2 = 0 -/
def my_line (a b x y : ℝ) : Prop := 2*a*x - b*y + 2 = 0

/-- The condition that the line cuts a chord of length 4 from the circle -/
def cuts_chord_of_length_4 (a b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    my_circle x₁ y₁ ∧ my_circle x₂ y₂ ∧ 
    my_line a b x₁ y₁ ∧ my_line a b x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16

theorem min_value_of_expression (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_chord : cuts_chord_of_length_4 a b) :
  (1 / a + 4 / b) ≥ 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l248_24802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_george_solo_work_days_l248_24857

/-- The number of days it takes Paul to complete the entire work alone -/
noncomputable def paul_solo_days : ℝ := 30

/-- The fraction of work George completes alone -/
noncomputable def george_work_fraction : ℝ := 3/5

/-- The number of days it takes George and Paul to complete the remaining work together -/
noncomputable def days_working_together : ℝ := 4

/-- The number of days George works alone -/
noncomputable def george_solo_days : ℝ := 90

theorem george_solo_work_days :
  let total_work : ℝ := 1
  let remaining_work : ℝ := total_work - george_work_fraction
  let paul_work_rate : ℝ := 1 / paul_solo_days
  let george_work_rate : ℝ := george_work_fraction / george_solo_days
  (george_work_rate + paul_work_rate) * days_working_together = remaining_work :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_george_solo_work_days_l248_24857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_valid_l248_24870

open Real

-- Define the differential equation
noncomputable def diff_eq (y : ℝ → ℝ) (x : ℝ) : Prop :=
  x * ((deriv (deriv y)) x) + 2 * ((deriv y) x) + x * (y x) = 0

-- Define the particular solution
noncomputable def y₁ (x : ℝ) : ℝ := sin x / x

-- Define the general solution
noncomputable def general_solution (C₁ C₂ : ℝ) (x : ℝ) : ℝ :=
  C₁ * (cos x / x) + C₂ * (sin x / x)

-- Statement of the theorem
theorem general_solution_valid (C₁ C₂ : ℝ) :
  (∀ x : ℝ, x ≠ 0 → diff_eq y₁ x) →
  (∀ x : ℝ, x ≠ 0 → diff_eq (general_solution C₁ C₂) x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_valid_l248_24870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_l248_24899

theorem negation_of_existence :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_l248_24899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tomato_plants_count_l248_24815

theorem tomato_plants_count
  (sunflower_count : ℕ)
  (corn_count : ℕ)
  (max_plants_per_row : ℕ)
  (tomato_rows : ℕ)
  (h1 : sunflower_count = 45)
  (h2 : corn_count = 81)
  (h3 : max_plants_per_row = 9)
  (h4 : tomato_rows = sunflower_count / max_plants_per_row + corn_count / max_plants_per_row)
  : tomato_rows * max_plants_per_row = 126 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tomato_plants_count_l248_24815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complementary_events_l248_24876

-- Define the sample space
inductive Ball : Type
| Red : Ball
| White : Ball

-- Define equality for Ball
instance : DecidableEq Ball :=
  fun a b => match a, b with
  | Ball.Red, Ball.Red => isTrue rfl
  | Ball.White, Ball.White => isTrue rfl
  | Ball.Red, Ball.White => isFalse (fun h => Ball.noConfusion h)
  | Ball.White, Ball.Red => isFalse (fun h => Ball.noConfusion h)

-- Define the bag contents
def bag : Multiset Ball := 2 • {Ball.Red} + 2 • {Ball.White}

-- Define the event of drawing two balls
def draw : Finset (Ball × Ball) := (bag.toFinset.product bag.toFinset).filter (fun p => p.1 ≠ p.2)

-- Define the event "at least one white ball"
def at_least_one_white : Set (Ball × Ball) :=
  {p | p.1 = Ball.White ∨ p.2 = Ball.White}

-- Define the event "both are red balls"
def both_red : Set (Ball × Ball) :=
  {p | p.1 = Ball.Red ∧ p.2 = Ball.Red}

-- Theorem stating that the events are complementary
theorem complementary_events :
  at_least_one_white ∪ both_red = draw.toSet ∧
  at_least_one_white ∩ both_red = ∅ := by
  sorry

#check complementary_events

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complementary_events_l248_24876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_video_game_sales_total_video_game_sales_total_eq_770_l248_24807

theorem video_game_sales_total (zachary_games : ℕ) (price_per_game : ℚ) 
  (jason_percentage : ℚ) (ryan_extra : ℚ) : ℚ :=
  let zachary_earnings := zachary_games * price_per_game
  let jason_earnings := zachary_earnings + jason_percentage * zachary_earnings
  let ryan_earnings := jason_earnings + ryan_extra
  let total_earnings := zachary_earnings + jason_earnings + ryan_earnings
  have h1 : zachary_games = 40 := by sorry
  have h2 : price_per_game = 5 := by sorry
  have h3 : jason_percentage = 0.3 := by sorry
  have h4 : ryan_extra = 50 := by sorry
  total_earnings

theorem video_game_sales_total_eq_770 : 
  video_game_sales_total 40 5 0.3 50 = 770 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_video_game_sales_total_video_game_sales_total_eq_770_l248_24807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l248_24810

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x + Real.pi / 3)

theorem phase_shift_of_f :
  ∃ (C : ℝ), ∀ (x : ℝ), f x = 2 * Real.cos (2 * (x - C)) ∧ C = -Real.pi / 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l248_24810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scalar_mult_zero_implies_factor_zero_l248_24860

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem scalar_mult_zero_implies_factor_zero (a : V) (r : ℝ) :
  r • a = 0 → r = 0 ∨ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scalar_mult_zero_implies_factor_zero_l248_24860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_on_interval_max_convex_interval_max_interval_is_two_l248_24844

noncomputable section

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := (1/12) * x^4 - (1/6) * m * x^3 - (3/2) * x^2

-- Define the second derivative of f(x)
def f'' (m : ℝ) (x : ℝ) : ℝ := x^2 - m * x - 3

-- Part I: Convexity on (-1,3) implies m = 2
theorem convex_on_interval (m : ℝ) :
  (∀ x ∈ Set.Ioo (-1) 3, f'' m x < 0) → m = 2 :=
by sorry

-- Part II: Maximum interval for convexity when |m| ≤ 2
theorem max_convex_interval (a b : ℝ) :
  (∀ m : ℝ, |m| ≤ 2 → ∀ x ∈ Set.Ioo a b, f'' m x < 0) →
  b - a ≤ 2 :=
by sorry

-- The maximum value of b-a is indeed 2
theorem max_interval_is_two :
  ∃ a b : ℝ, b - a = 2 ∧ 
  (∀ m : ℝ, |m| ≤ 2 → ∀ x ∈ Set.Ioo a b, f'' m x < 0) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_on_interval_max_convex_interval_max_interval_is_two_l248_24844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_tangent_properties_l248_24879

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x / Real.exp x

def tangent_line (b : ℝ) (x : ℝ) : ℝ := x + b

theorem curve_and_tangent_properties (a b m : ℝ) :
  (∀ x, x ∈ Set.Ioo (1/2) (3/2) → f a x < 1 / (m + 6*x - 3*x^2)) →
  (∀ x, tangent_line b x = f a x + (deriv (f a)) 0 * (x - 0)) →
  a = 1 ∧ b = 0 ∧ m ∈ Set.Icc (-9/4) (Real.exp 1 - 3) := by
  sorry

#check curve_and_tangent_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_tangent_properties_l248_24879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_perpendicular_tangent_l248_24836

/-- The minimum distance between a point on y = x² and the intersection of the perpendicular to the tangent line with the curve -/
theorem min_distance_parabola_perpendicular_tangent :
  ∃ x₀ : ℝ, 
    let f : ℝ → ℝ := fun x ↦ x^2
    let A : ℝ → ℝ × ℝ := fun x₀ ↦ (x₀, f x₀)
    let m : ℝ → ℝ → ℝ := fun x₀ x ↦ 2*x₀*x - 2*x₀^2 + f x₀  -- Tangent line at A
    let n : ℝ → ℝ → ℝ := fun x₀ x ↦ -(1/(2*x₀))*(x - x₀) + f x₀  -- Line perpendicular to m at A
    let B : ℝ → ℝ × ℝ := fun x₀ ↦ (-1/(2*x₀) - x₀, f (-1/(2*x₀) - x₀))  -- Intersection of n with y = x²
    let dist : ℝ → ℝ := fun x₀ ↦ Real.sqrt ((A x₀).1 - (B x₀).1)^2 + ((A x₀).2 - (B x₀).2)^2
    (∀ y : ℝ, dist x₀ ≤ dist y) ∧ dist x₀ = 3*Real.sqrt 3/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_perpendicular_tangent_l248_24836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_current_rate_l248_24853

/-- The man's usual rowing speed in still water (miles per hour) -/
noncomputable def r : ℝ := sorry

/-- The speed of the stream's current (miles per hour) -/
noncomputable def w : ℝ := sorry

/-- The distance rowed (miles) -/
def distance : ℝ := 20

/-- Condition: Downstream time is 7 hours less than upstream time at usual rate -/
axiom usual_rate_condition : distance / (r + w) + 7 = distance / (r - w)

/-- Condition: When the man triples his usual rate, downstream time is 2 hours less than upstream time -/
axiom triple_rate_condition : distance / (3 * r + w) + 2 = distance / (3 * r - w)

/-- Theorem: The rate of the stream's current is 3 miles per hour -/
theorem stream_current_rate : w = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_current_rate_l248_24853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l248_24843

-- Define the types for lines and planes
structure Line : Type
structure Plane : Type

-- Define perpendicular relation between a line and a plane
axiom perpendicular_line_plane : Line → Plane → Prop

-- Define perpendicular relation between two lines
axiom perpendicular_lines : Line → Line → Prop

-- Define the property of a line being within a plane
axiom line_in_plane : Line → Plane → Prop

-- Define sufficient but not necessary condition
def sufficient_but_not_necessary (P Q : Prop) : Prop :=
  (P → Q) ∧ ¬(Q → P)

-- Theorem statement
theorem perpendicular_condition (l m n : Line) (α : Plane) 
  (h1 : line_in_plane m α) (h2 : line_in_plane n α) :
  sufficient_but_not_necessary 
    (perpendicular_line_plane l α) 
    (perpendicular_lines l m ∧ perpendicular_lines l n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l248_24843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_associate_professor_pencils_l248_24864

/-- Represents the number of associate professors -/
def A : ℕ := sorry

/-- Represents the number of assistant professors -/
def B : ℕ := sorry

/-- Represents the number of pencils each associate professor brings -/
def P : ℕ := sorry

/-- The total number of people present is 5 -/
axiom total_people : A + B = 5

/-- The total number of pencils brought is 10 -/
axiom total_pencils : P * A + B = 10

/-- The total number of charts brought is 5 -/
axiom total_charts : A + 2 * B = 5

theorem associate_professor_pencils : P = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_associate_professor_pencils_l248_24864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_on_interval_l248_24866

-- Define the function f(x) = e^x - x
noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

-- State the theorem
theorem max_value_of_f_on_interval :
  ∃ (max : ℝ), max = Real.exp 1 - 1 ∧
  ∀ x ∈ Set.Icc (-1) 1, f x ≤ max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_on_interval_l248_24866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_symmetry_phase_l248_24889

open Real

-- Define the function f
noncomputable def f (ω φ x : ℝ) : ℝ := sin (ω * x + φ)

-- State the theorem
theorem sine_symmetry_phase (ω φ : ℝ) : 
  ω > 0 → 
  0 < φ → φ < π → 
  (∀ x, f ω φ (π/6 - x) = f ω φ (π/6 + x)) →
  (∀ x, f ω φ (2*π/3 - x) = f ω φ (2*π/3 + x)) →
  f ω φ (2*π/3) = 0 →
  φ = π/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_symmetry_phase_l248_24889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixtieth_integer_is_32541_l248_24867

def is_valid_integer (n : ℕ) : Bool :=
  if n ≥ 10000 ∧ n < 100000 then
    let digits := [n / 10000, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10]
    digits.toFinset.card == 5 ∧ 
    digits.all (λ d => d ∈ [1, 2, 3, 4, 5])
  else
    false

def valid_integers : List ℕ :=
  (List.range 100000).filter is_valid_integer

theorem sixtieth_integer_is_32541 : valid_integers[59]! = 32541 := by
  sorry

#eval valid_integers[59]!

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixtieth_integer_is_32541_l248_24867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetrical_circle_equation_min_area_parallelogram_ABCD_l248_24804

-- Define the circle O
def circle_O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

-- Define points A and B
def point_A : ℝ × ℝ := (-3, 0)
def point_B : ℝ × ℝ := (-1, 2)

-- Define the line AB
def line_AB : Set (ℝ × ℝ) := {p | p.2 = p.1 + 3}

-- Define the symmetrical circle
def symmetrical_circle : Set (ℝ × ℝ) := {p | (p.1 + 3)^2 + (p.2 - 3)^2 = 4}

-- Theorem for the symmetrical circle
theorem symmetrical_circle_equation : 
  ∀ p : ℝ × ℝ, p ∈ symmetrical_circle ↔ (p.1 + 3)^2 + (p.2 - 3)^2 = 4 :=
by sorry

-- Define a parallelogram ABCD with C and D on circle O
def is_parallelogram_ABCD (C D : ℝ × ℝ) : Prop :=
  C ∈ circle_O ∧ D ∈ circle_O ∧
  (C.1 - point_A.1 = D.1 - point_B.1) ∧
  (C.2 - point_A.2 = D.2 - point_B.2)

-- Helper function to calculate the area of a quadrilateral
noncomputable def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ := 
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2) +
       (C.1 - A.1) * (D.2 - A.2) - (D.1 - A.1) * (C.2 - A.2)) / 2

-- Theorem for the minimum area of parallelogram ABCD
theorem min_area_parallelogram_ABCD :
  ∃ (C D : ℝ × ℝ), is_parallelogram_ABCD C D ∧
  ∀ (E F : ℝ × ℝ), is_parallelogram_ABCD E F →
  area_quadrilateral point_A point_B E F ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetrical_circle_equation_min_area_parallelogram_ABCD_l248_24804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_bound_l248_24848

open Complex

/-- Represents a point in the complex plane -/
def Point := ℂ

/-- Represents a circle in the complex plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a counter-clockwise rotation in the complex plane -/
def Rotation := Point → ℝ → Point

/-- Given three points on a circle, construct the rotation functions -/
noncomputable def constructRotations (A₁ A₂ A₃ : Point) : Fin 3 → Rotation := sorry

/-- Construct the points P₁, P₂, P₃ given the rotations and initial point P -/
noncomputable def constructPPoints (τ : Fin 3 → Rotation) (P : Point) : Fin 3 → Point := sorry

/-- Calculate the radius of the circumcircle of a triangle -/
noncomputable def circumradius (P₁ P₂ P₃ : Point) : ℝ := sorry

/-- Check if three points are in counter-clockwise order -/
def IsCounterclockwise (A B C : Point) : Prop := sorry

/-- Check if a point is on a circle -/
def OnCircle (P : Point) (Γ : Circle) : Prop := sorry

theorem circumradius_bound (Γ : Circle) (A₁ A₂ A₃ P : Point) 
  (h₁ : OnCircle A₁ Γ) (h₂ : OnCircle A₂ Γ) (h₃ : OnCircle A₃ Γ) 
  (h₄ : IsCounterclockwise A₁ A₂ A₃) : 
  let τ := constructRotations A₁ A₂ A₃
  let P₁ := constructPPoints τ P 0
  let P₂ := constructPPoints τ P 1
  let P₃ := constructPPoints τ P 2
  circumradius P₁ P₂ P₃ ≤ Γ.radius := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_bound_l248_24848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_sin_4theta_value_l248_24831

noncomputable def a (x : ℝ) : ℝ × ℝ := (1 + Real.sin (2 * x), Real.sin x - Real.cos x)

noncomputable def b (x : ℝ) : ℝ × ℝ := (1, Real.sin x + Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_max_value :
  ∃ (x : ℝ), f x = 1 + Real.sqrt 2 ∧ ∀ (y : ℝ), f y ≤ 1 + Real.sqrt 2 :=
by sorry

theorem sin_4theta_value (θ : ℝ) (h : f θ = 8/5) :
  Real.sin (4 * θ) = 16/25 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_sin_4theta_value_l248_24831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_h_is_three_l248_24812

-- Define the polynomials f and g
def f (x : ℝ) : ℝ := 2 - 15*x + 4*x^2 - 5*x^3 + 6*x^4
def g (x : ℝ) : ℝ := 4 - 3*x - 7*x^3 + 12*x^4

-- Define the combined polynomial h
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

-- Define a function to get the degree of a polynomial
def polynomialDegree (p : ℝ → ℝ) : ℕ := sorry

-- Theorem statement
theorem degree_of_h_is_three :
  ∃ (c : ℝ), c = -1/2 ∧ 
  polynomialDegree (h c) = 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_h_is_three_l248_24812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_integer_T_l248_24826

/-- Sum of reciprocals of non-zero digits from 1 to 9 -/
def L : ℚ := (1 : ℚ) + (1/2) + (1/3) + (1/4) + (1/5) + (1/6) + (1/7) + (1/8) + (1/9)

/-- The denominator of L when expressed in its simplest form -/
def D : ℕ := 2^3 * 3^2 * 5 * 7

/-- Sum of reciprocals of non-zero digits from 1 to 5^n -/
def T (n : ℕ) : ℚ := n * (5^(n-1) : ℚ) * L + 1

/-- Predicate to check if a rational number is an integer -/
def isInteger (q : ℚ) : Prop := ∃ (z : ℤ), q = z

/-- The smallest positive integer n for which T_n is an integer -/
theorem smallest_n_for_integer_T : (∀ m < 504, ¬(isInteger (T m))) ∧ (isInteger (T 504)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_integer_T_l248_24826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_diagonal_ratio_l248_24823

/-- A regular hexagon with vertices A, B, C, D, E, F -/
structure RegularHexagon where
  vertices : Fin 6 → ℂ
  is_regular : ∀ i, vertices ((i + 1) % 6) = vertices i * (Complex.exp (Complex.I * Real.pi / 3))

/-- The ratio r that divides the diagonals -/
def diagonal_ratio (h : RegularHexagon) (r : ℝ) : Prop :=
  let A := h.vertices 0
  let C := h.vertices 2
  let E := h.vertices 4
  let M := (1 - r) • A + r • C
  let N := (1 - r) • C + r • E
  ∃ t, M = (1 - t) • h.vertices 1 + t • N

theorem hexagon_diagonal_ratio (h : RegularHexagon) :
  ∃ (r : ℝ), diagonal_ratio h r ∧ r = Real.sqrt 3 / 3 := by
  sorry

#check hexagon_diagonal_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_diagonal_ratio_l248_24823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_range_l248_24875

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 4

-- Define the line passing through (0,3)
def line_eq (x y k : ℝ) : Prop := y = k * x + 3

-- Define point Q
def Q : ℝ × ℝ := (2, 2)

-- Define the vector sum →QA + →QB
def vector_sum (A B : ℝ × ℝ) : ℝ × ℝ :=
  (A.1 + B.1 - 2*Q.1, A.2 + B.2 - 2*Q.2)

-- Define the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

-- Theorem statement
theorem vector_sum_range :
  ∀ (A B : ℝ × ℝ) (k : ℝ),
    circle_eq A.1 A.2 →
    circle_eq B.1 B.2 →
    line_eq A.1 A.2 k →
    line_eq B.1 B.2 k →
    4 ≤ magnitude (vector_sum A B) ∧ magnitude (vector_sum A B) ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_range_l248_24875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_specific_line_l248_24893

/-- A line passing through two points -/
structure Line where
  x1 : ℚ
  y1 : ℚ
  x2 : ℚ
  y2 : ℚ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℚ :=
  (l.x1 * l.y2 - l.x2 * l.y1) / (l.y2 - l.y1)

/-- Theorem: The x-intercept of the line passing through (-2, 2) and (2, 10) is -3 -/
theorem x_intercept_specific_line :
  x_intercept ⟨-2, 2, 2, 10⟩ = -3 := by
  -- Unfold the definition of x_intercept
  unfold x_intercept
  -- Simplify the arithmetic
  simp [add_div, sub_div, mul_div_cancel]
  -- The result should now be obvious to Lean
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_specific_line_l248_24893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_midpoint_equation_l248_24847

/-- Parabola defined by the equation y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Line passing through two points (x₁, y₁) and (x₂, y₂) -/
def line (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (x - x₁) * (y₂ - y₁)

/-- Midpoint of two points (x₁, y₁) and (x₂, y₂) -/
def midpoint_of (x₁ y₁ x₂ y₂ xm ym : ℝ) : Prop :=
  xm = (x₁ + x₂) / 2 ∧ ym = (y₁ + y₂) / 2

theorem parabola_line_midpoint_equation :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ →
    parabola x₂ y₂ →
    midpoint_of x₁ y₁ x₂ y₂ 2 2 →
    ∀ (x y : ℝ), line x₁ y₁ x₂ y₂ x y ↔ x = y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_midpoint_equation_l248_24847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coat_price_calculation_l248_24890

/-- The final price of a coat after discounts and tax -/
noncomputable def final_price (original_price : ℝ) (initial_discount_percent : ℝ) 
  (coupon_discount : ℝ) (tax_percent : ℝ) : ℝ :=
  let price_after_initial_discount := original_price * (1 - initial_discount_percent / 100)
  let price_after_coupon := price_after_initial_discount - coupon_discount
  let final_price := price_after_coupon * (1 + tax_percent / 100)
  final_price

/-- Theorem stating the final price of the coat -/
theorem coat_price_calculation :
  final_price 150 25 10 10 = 112.75 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval final_price 150 25 10 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coat_price_calculation_l248_24890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_at_one_implies_a_b_values_l248_24832

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x = 1 then 1
  else if x ≠ -1 ∧ x ≠ 1 then a / (1 - x) - b / (1 - x^2)
  else 0  -- arbitrary value for x = -1

-- State the theorem
theorem continuous_at_one_implies_a_b_values (a b : ℝ) :
  ContinuousAt (f a b) 1 → a = -2 ∧ b = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_at_one_implies_a_b_values_l248_24832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l248_24897

def sequence_a : ℕ → ℚ
  | 0 => 3  -- Add this case for n = 0
  | 1 => 3
  | (n + 2) => (3 * sequence_a (n + 1) - 4) / (sequence_a (n + 1) - 2)

theorem sequence_a_formula (n : ℕ) : 
  sequence_a n = ((-2)^(n+2) - 1) / ((-2)^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l248_24897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_special_right_triangle_l248_24829

-- Define a right triangle PQR
structure RightTriangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  is_right_angle : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0

-- Define the properties of the triangle
def isosceles_right_triangle (t : RightTriangle) : Prop :=
  let PQ := Real.sqrt ((t.Q.1 - t.P.1)^2 + (t.Q.2 - t.P.2)^2)
  let PR := Real.sqrt ((t.R.1 - t.P.1)^2 + (t.R.2 - t.P.2)^2)
  PQ = PR

noncomputable def hypotenuse_length (t : RightTriangle) : ℝ :=
  Real.sqrt ((t.Q.1 - t.R.1)^2 + (t.Q.2 - t.R.2)^2)

-- Theorem statement
theorem area_of_special_right_triangle (t : RightTriangle) 
  (h1 : isosceles_right_triangle t) 
  (h2 : hypotenuse_length t = 6 * Real.sqrt 2) : 
  (1/2) * (Real.sqrt ((t.Q.1 - t.P.1)^2 + (t.Q.2 - t.P.2)^2)) * 
          (Real.sqrt ((t.R.1 - t.P.1)^2 + (t.R.2 - t.P.2)^2)) = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_special_right_triangle_l248_24829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumscribed_circle_diameter_l248_24825

/-- The diameter of a triangle's circumscribed circle, given one side length and its opposite angle -/
noncomputable def circumscribed_circle_diameter (side_length : ℝ) (opposite_angle : ℝ) : ℝ :=
  side_length / Real.sin opposite_angle

theorem triangle_circumscribed_circle_diameter :
  let side_length : ℝ := 15
  let opposite_angle : ℝ := π / 4  -- 45° in radians
  circumscribed_circle_diameter side_length opposite_angle = 15 * Real.sqrt 2 :=
by
  -- Unfold the definition of circumscribed_circle_diameter
  unfold circumscribed_circle_diameter
  -- Simplify the expression
  simp [Real.sin_pi_div_four]
  -- The proof is completed with sorry for now
  sorry

#check triangle_circumscribed_circle_diameter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumscribed_circle_diameter_l248_24825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l248_24856

theorem expression_evaluation : 
  (1/4)^(-1 : ℤ) + |-(Real.sqrt 3)| - (Real.pi - 3)^(0 : ℤ) + 3 * Real.tan (30 * Real.pi / 180) = 3 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l248_24856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_of_f_l248_24834

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 + 2*x - 3) / Real.log (1/2)

-- State the theorem
theorem increasing_interval_of_f :
  {x : ℝ | ∀ y, y < x → f y < f x} = Set.Iio (-3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_of_f_l248_24834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_4_minus_x_squared_definite_integral_circle_area_l248_24828

/-- The definite integral of √(4-x^2) from -2 to 2 equals 2π -/
theorem integral_sqrt_4_minus_x_squared :
  (∫ x in (-2 : ℝ)..2, Real.sqrt (4 - x^2)) = 2 * π := by sorry

/-- A more general theorem about definite integrals and circle areas -/
theorem definite_integral_circle_area (f : ℝ → ℝ) :
  (∫ x in (-2 : ℝ)..2, f x) = 2 * π → ∀ x, f x = Real.sqrt (4 - x^2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_4_minus_x_squared_definite_integral_circle_area_l248_24828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eq_or_eq_neg_of_sq_eq_line_slope_l248_24818

/-- Two real numbers are equal up to sign if their squares are equal -/
theorem eq_or_eq_neg_of_sq_eq {x y : ℝ} (h : x^2 = y^2) : x = y ∨ x = -y := by sorry

/-- The equation of the line passing through (1,2) with slope k intercepted by two parallel lines -/
noncomputable def line_equation (k : ℝ) (x : ℝ) : ℝ := k * (x - 1) + 2

/-- The x-coordinate of point A (intersection with first line) -/
noncomputable def point_A_x (k : ℝ) : ℝ := (3*k - 7) / (3*k + 4)

/-- The y-coordinate of point A (intersection with first line) -/
noncomputable def point_A_y (k : ℝ) : ℝ := (-5*k + 8) / (3*k + 4)

/-- The x-coordinate of point B (intersection with second line) -/
noncomputable def point_B_x (k : ℝ) : ℝ := (3*k - 12) / (3*k + 4)

/-- The y-coordinate of point B (intersection with second line) -/
noncomputable def point_B_y (k : ℝ) : ℝ := (8 - 10*k) / (3*k + 4)

/-- The squared distance between points A and B -/
noncomputable def distance_squared (k : ℝ) : ℝ := 
  (point_B_x k - point_A_x k)^2 + (point_B_y k - point_A_y k)^2

theorem line_slope (k : ℝ) : 
  (∀ x, 4*x + 3*(line_equation k x) + 1 = 0 → 4*x + 3*(line_equation k x) + 6 = 0 → False) →
  distance_squared k = 2 →
  k = 7 ∨ k = -1/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eq_or_eq_neg_of_sq_eq_line_slope_l248_24818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l248_24827

/-- A function f(x) with specific properties -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b / x

/-- The theorem stating the properties of f and its derivative -/
theorem f_properties (a b : ℝ) :
  f a b 1 = -2 ∧
  (deriv (f a b)) 1 = 0 →
  (deriv (f a b)) 2 = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l248_24827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_cartesian_equation_line_l_cartesian_equation_line_n_slope_angle_l248_24816

noncomputable section

-- Define the curve C
def curve_C (m : ℝ) : ℝ × ℝ := (|m + 1/(2*m)|, m - 1/(2*m))

-- Define the point M
def point_M : ℝ × ℝ := (2, 0)

-- Define the line l in polar form
def line_l (ρ θ : ℝ) : Prop := ρ * Real.cos (θ + Real.pi/3) = 1

-- Define the line n
def line_n (θ : ℝ) (t : ℝ) : ℝ × ℝ := (2 + t * Real.cos θ, t * Real.sin θ)

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem curve_C_cartesian_equation :
  ∀ x y : ℝ, x ≥ Real.sqrt 2 →
  (∃ m : ℝ, curve_C m = (x, y)) ↔ x^2/2 - y^2/2 = 1 := by sorry

theorem line_l_cartesian_equation :
  ∀ x y : ℝ, (∃ ρ θ : ℝ, line_l ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ↔
  x - Real.sqrt 3 * y - 2 = 0 := by sorry

theorem line_n_slope_angle :
  ∀ θ : ℝ, (∃ t₁ t₂ : ℝ, 
    (∃ m : ℝ, curve_C m = line_n θ t₁) ∧
    (∃ m : ℝ, curve_C m = line_n θ t₂) ∧
    distance (line_n θ t₁) (line_n θ t₂) = 4 * Real.sqrt 2) →
  θ = Real.pi/3 ∨ θ = 2*Real.pi/3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_cartesian_equation_line_l_cartesian_equation_line_n_slope_angle_l248_24816
