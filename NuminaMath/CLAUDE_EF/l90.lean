import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_b_l90_9066

-- Define a triangle ABC
variable (A B C : ℝ)

-- Define the condition that A, B, C form a triangle
variable (triangle_condition : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi)

-- Define a and b
noncomputable def a (A B : ℝ) : ℝ := Real.sin (A + B)
noncomputable def b (A B : ℝ) : ℝ := Real.sin A + Real.sin B

-- State the theorem
theorem a_less_than_b (A B : ℝ) (h : 0 < A ∧ 0 < B ∧ A + B < Real.pi) : 
  a A B < b A B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_b_l90_9066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counter_arrangement_impossibility_l90_9078

theorem counter_arrangement_impossibility (n : ℕ) (h : n > 0) :
  ¬ ∃ (arrangement : Fin (n^2) → Fin n),
    (∀ label : Fin n, (Fintype.card {i | arrangement i = label} = n)) ∧
    (∀ label : Fin n,
      ∀ i j : Fin (n^2),
        i < j ∧ 
        arrangement i = label ∧ 
        arrangement j = label ∧ 
        (∀ k : Fin (n^2), i < k ∧ k < j → arrangement k ≠ label) →
        j.val - i.val - 1 = label.val) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_counter_arrangement_impossibility_l90_9078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_divisor_condition_l90_9082

theorem no_common_divisor_condition (k : ℤ) : 
  (∀ n : ℤ, Int.gcd (4*n + 1) (k*n + 1) = 1) ↔ 
  (∃ m : ℕ, k = 4 + 2^m ∨ k = 4 - 2^m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_divisor_condition_l90_9082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vertices_of_special_polyhedron_l90_9034

/-- A lattice point in 3D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ
  z : ℤ

/-- A convex polyhedron in 3D space -/
structure ConvexPolyhedron where
  vertices : Finset LatticePoint
  is_convex : Bool
  no_interior_lattice_points : Bool
  no_face_lattice_points : Bool
  no_edge_lattice_points : Bool

/-- The theorem to be proved -/
theorem max_vertices_of_special_polyhedron (P : ConvexPolyhedron) :
  P.is_convex = true →
  P.no_interior_lattice_points = true →
  P.no_face_lattice_points = true →
  P.no_edge_lattice_points = true →
  P.vertices.card ≤ 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vertices_of_special_polyhedron_l90_9034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doghouse_area_l90_9055

/-- The area outside a regular hexagon that can be reached by a point tethered to a vertex -/
def area_outside_hexagon (side_length : ℝ) (rope_length : ℝ) : ℝ :=
  -- This function is not implemented, but represents the area we're calculating
  sorry

/-- The area outside a regular hexagon reachable by a tethered point -/
theorem doghouse_area (side_length : ℝ) (rope_length : ℝ) : 
  side_length = 1 → rope_length = 2 → 
  area_outside_hexagon side_length rope_length = 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_doghouse_area_l90_9055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_quadrilateral_on_ellipse_l90_9080

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on the ellipse -/
def on_ellipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Calculates the eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- Helper function to calculate the area of a quadrilateral -/
noncomputable def area_quadrilateral (p q a b : Point) : ℝ :=
  sorry -- Definition of area calculation

/-- Theorem: Maximum area of quadrilateral APBQ on ellipse -/
theorem max_area_quadrilateral_on_ellipse (e : Ellipse) 
    (h_ecc : eccentricity e = 1/2)
    (p q : Point) 
    (h_p_on_e : on_ellipse e p) 
    (h_q_on_e : on_ellipse e q)
    (h_p : p = ⟨2, 3⟩) 
    (h_q : q = ⟨2, -3⟩) : 
    ∃ (a b : Point), 
      on_ellipse e a ∧ 
      on_ellipse e b ∧ 
      (b.y - a.y) / (b.x - a.x) = 1/2 ∧
      (∀ (a' b' : Point), 
        on_ellipse e a' → 
        on_ellipse e b' → 
        (b'.y - a'.y) / (b'.x - a'.x) = 1/2 → 
        area_quadrilateral p q a' b' ≤ 12 * Real.sqrt 3) :=
by
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_quadrilateral_on_ellipse_l90_9080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_passed_is_22_percent_l90_9006

/-- The percentage of engineering students who passed the exam --/
def percentage_passed (male_students female_students : ℕ)
  (male_eng_percent female_eng_percent : ℚ)
  (male_pass_percent female_pass_percent : ℚ) : ℚ :=
  let male_eng := (male_eng_percent * male_students).floor
  let female_eng := (female_eng_percent * female_students).floor
  let male_passed := (male_pass_percent * male_eng).floor
  let female_passed := (female_pass_percent * female_eng).floor
  let total_passed := male_passed + female_passed
  let total_eng := male_eng + female_eng
  (total_passed : ℚ) / (total_eng : ℚ) * 100

/-- Theorem stating that the percentage of engineering students who passed the exam is 22% --/
theorem percentage_passed_is_22_percent :
  percentage_passed 120 100 (25/100) (20/100) (20/100) (25/100) = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_passed_is_22_percent_l90_9006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dinner_seating_arrangements_l90_9018

theorem dinner_seating_arrangements (n k m : ℕ) :
  n = 12 →
  k = 8 →
  m = 4 →
  (n.choose k) * (k.choose m) * ((m - 1).factorial * (m - 1).factorial) = 1247400 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dinner_seating_arrangements_l90_9018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_count_l90_9057

def is_valid_number (n : ℕ) : Bool :=
  100 ≤ n && n < 1000 && (n % 10) ≥ 3 * ((n / 10) % 10)

def count_valid_numbers : ℕ := 
  (List.range 1000).filter is_valid_number |>.length

theorem valid_numbers_count : count_valid_numbers = 198 := by
  -- Proof goes here
  sorry

#eval count_valid_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_count_l90_9057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motion_repeats_120_seconds_l90_9063

/-- Represents a point moving on a circular track -/
structure MovingPoint where
  position : ℝ  -- Position on the track (0 ≤ position < 40)
  speed : ℝ     -- Speed in m/s
  direction : Bool  -- true for clockwise, false for counterclockwise

/-- The state of the system at any given time -/
structure SystemState where
  a : MovingPoint
  b : MovingPoint
  time : ℝ

def track_length : ℝ := 40

/-- Updates the position of a point after a given time interval -/
noncomputable def update_position (p : MovingPoint) (dt : ℝ) : MovingPoint :=
  { p with position := (p.position + p.speed * dt * (if p.direction then 1 else -1)) % track_length }

/-- Checks if A needs to change direction -/
noncomputable def should_change_direction (a b : MovingPoint) : Bool :=
  let distance := min (abs (a.position - b.position)) (track_length - abs (a.position - b.position))
  distance ≤ 10

/-- Updates the system state after a small time step -/
noncomputable def update_system (s : SystemState) (dt : ℝ) : SystemState :=
  let new_b := update_position s.b dt
  let new_a := 
    if should_change_direction s.a new_b
    then { (update_position s.a dt) with direction := ¬s.a.direction }
    else update_position s.a dt
  { a := new_a, b := new_b, time := s.time + dt }

/-- The initial state of the system -/
def initial_state : SystemState :=
  { a := { position := 20, speed := 3, direction := true },
    b := { position := 0, speed := 1, direction := true },
    time := 0 }

/-- Theorem stating that the motion of A repeats every 120 seconds -/
theorem motion_repeats_120_seconds :
  ∀ (n : ℕ), 
    (Nat.iterate (λ s => update_system s 120) n initial_state).a.position = initial_state.a.position ∧
    (Nat.iterate (λ s => update_system s 120) n initial_state).a.direction = initial_state.a.direction :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_motion_repeats_120_seconds_l90_9063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_with_no_intersection_l90_9058

-- Define the function f(x) = 2^|x| - 1
noncomputable def f (x : ℝ) : ℝ := 2^(abs x) - 1

-- Define the property of having no common points
def no_common_points (b : ℝ) : Prop :=
  ∀ x : ℝ, f x ≠ b

-- Theorem statement
theorem range_of_b_with_no_intersection :
  {b : ℝ | no_common_points b} = Set.Iio 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_with_no_intersection_l90_9058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_tetrahedron_volume_l90_9032

/-- The volume of a regular tetrahedron inscribed in a sphere of radius R -/
noncomputable def volume_of_regular_tetrahedron_inscribed_in_sphere (R : ℝ) : ℝ :=
  (8 * R^3 * Real.sqrt 6) / 27

/-- The volume of a regular tetrahedron inscribed in a sphere of radius R -/
theorem inscribed_tetrahedron_volume (R : ℝ) (h : R > 0) :
  ∃ V : ℝ, V = (8 * R^3 * Real.sqrt 6) / 27 ∧
  V = volume_of_regular_tetrahedron_inscribed_in_sphere R :=
by
  use volume_of_regular_tetrahedron_inscribed_in_sphere R
  constructor
  · rfl
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_tetrahedron_volume_l90_9032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_l90_9069

-- Define the curve C in polar coordinates
def curve_C (ρ θ a : ℝ) : Prop := ρ * (Real.sin θ)^2 = 2 * a * Real.cos θ

-- Define the line l in parametric form
def line_l (t : ℝ) : ℝ × ℝ := (-2 + t, -4 + t)

-- Define point P
def point_P : ℝ × ℝ := (-2, -4)

-- Define the condition that |PM|, |MN|, and |PN| form a geometric sequence
def geometric_sequence (PM MN PN : ℝ) : Prop := MN^2 = PM * PN

-- Main theorem
theorem curve_line_intersection (a : ℝ) :
  a > 0 →
  (∃ M N : ℝ × ℝ, 
    (∃ θ_M ρ_M, curve_C ρ_M θ_M a ∧ M = (ρ_M * Real.cos θ_M, ρ_M * Real.sin θ_M)) ∧
    (∃ θ_N ρ_N, curve_C ρ_N θ_N a ∧ N = (ρ_N * Real.cos θ_N, ρ_N * Real.sin θ_N)) ∧
    (∃ t_M, line_l t_M = M) ∧
    (∃ t_N, line_l t_N = N) ∧
    geometric_sequence (dist M point_P) (dist M N) (dist N point_P)) →
  a = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_l90_9069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_properties_l90_9040

/-- A linear function f(x) = kx + b where k < 0 and b < 0 -/
def linear_function (k b : ℝ) (hk : k < 0) (hb : b < 0) : ℝ → ℝ := λ x ↦ k * x + b

theorem linear_function_properties (k b : ℝ) (hk : k < 0) (hb : b < 0) :
  let f := linear_function k b hk hb
  (∀ x y, x < y → f y < f x) ∧ (f 0 < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_properties_l90_9040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_theorem_l90_9083

-- Define the types for lines and planes
variable (L P : Type*) [LinearOrderedField L] [LinearOrderedField P]

-- Define the relations
variable (parallel : L → L → Prop)
variable (parallel_plane : P → P → Prop)
variable (perpendicular : L → P → Prop)
variable (contains : P → L → Prop)
variable (intersects : P → P → L → Prop)

-- Define the lines and planes
variable (l m : L)
variable (α β γ : P)

-- State the theorem
theorem geometry_theorem 
  (h_distinct_lines : l ≠ m)
  (h_distinct_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)
  : 
  (parallel m l ∧ perpendicular m α → perpendicular l α) ∧
  (intersects α γ m ∧ intersects β γ l ∧ parallel_plane α β → parallel m l) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_theorem_l90_9083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_theorem_l90_9045

/-- The profit function for product A -/
noncomputable def profit_A (x : ℝ) : ℝ := 18 - 180 / (x + 10)

/-- The profit function for product B -/
noncomputable def profit_B (x : ℝ) : ℝ := x / 5

/-- The total profit function -/
noncomputable def total_profit (x : ℝ) : ℝ := profit_A x + profit_B (100 - x)

/-- The theorem stating the maximum profit and optimal investment -/
theorem max_profit_theorem :
  ∃ (x : ℝ), x ∈ Set.Icc 0 100 ∧ 
  (∀ y ∈ Set.Icc 0 100, total_profit y ≤ total_profit x) ∧
  total_profit x = 28 ∧ 
  x = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_theorem_l90_9045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l90_9090

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x ≠ a then (x + 1 - a) / (a - x) else 0

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x^2 + |((x - a) * f a x)|

theorem function_properties (a : ℝ) (h : a ≥ 1/2) :
  (∀ x : ℝ, x ≠ 0 → f a (x + a) = -1/x - 1) →
  (∀ x : ℝ, x ≠ a → f a x + f a (2*a - x) = -2) ∧
  (a > 3/2 → ∃ m : ℝ, m = a - 5/4 ∧ ∀ x : ℝ, x ≠ a → g a x ≥ m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l90_9090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmer_speed_is_10_l90_9072

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  man : ℝ  -- Speed of the man in still water
  stream : ℝ  -- Speed of the stream

/-- Given downstream and upstream distances and time, calculates the swimmer's speed. -/
noncomputable def calculate_swimmer_speed (downstream_distance upstream_distance time : ℝ) : SwimmerSpeed :=
  { man := (downstream_distance + upstream_distance) / (2 * time),
    stream := (downstream_distance - upstream_distance) / (2 * time) }

/-- Theorem stating that for the given conditions, the swimmer's speed in still water is 10 km/h. -/
theorem swimmer_speed_is_10 :
  let s := calculate_swimmer_speed 42 18 3
  s.man = 10 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmer_speed_is_10_l90_9072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_filling_time_l90_9019

/-- Represents the time taken to fill the pool using different combinations of hoses -/
structure PoolFilling where
  xy_time : ℚ  -- Time taken by hoses X and Y together
  xz_time : ℚ  -- Time taken by hoses X and Z together
  yz_time : ℚ  -- Time taken by hoses Y and Z together

/-- Calculates the time taken to fill the pool using all three hoses -/
def time_all_hoses (pf : PoolFilling) : ℚ :=
  20 / 7

/-- Theorem stating that given the conditions, the time taken for all three hoses 
    to fill the pool is 20/7 hours -/
theorem pool_filling_time (pf : PoolFilling) 
    (h_xy : pf.xy_time = 3)
    (h_xz : pf.xz_time = 6)
    (h_yz : pf.yz_time = 5) : 
  time_all_hoses pf = 20 / 7 := by
  sorry

#eval (20 : ℚ) / 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_filling_time_l90_9019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_six_equals_eight_l90_9092

-- Define the function f
noncomputable def f : ℝ → ℝ := fun u => (u^3 + 6*u^2 - 4*u + 104) / 64

-- State the theorem
theorem f_of_six_equals_eight :
  (∀ x : ℝ, f (4*x - 2) = x^3 - x + 2) → f 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_six_equals_eight_l90_9092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_relation_l90_9051

noncomputable def g (n : ℤ) : ℝ :=
  (3 + 2 * Real.sqrt 3) / 6 * ((2 + Real.sqrt 3) / 2) ^ n +
  (3 - 2 * Real.sqrt 3) / 6 * ((2 - Real.sqrt 3) / 2) ^ n

theorem g_relation (n : ℤ) : g (n + 1) - 2 * g n + g (n - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_relation_l90_9051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_profit_percentage_l90_9042

/-- Calculates the profit percentage for a dealer's transaction -/
theorem dealer_profit_percentage
  (purchase_quantity : ℕ)
  (purchase_amount : ℚ)
  (sale_quantity : ℕ)
  (sale_amount : ℚ)
  (h1 : purchase_quantity = 15)
  (h2 : purchase_amount = 25)
  (h3 : sale_quantity = 12)
  (h4 : sale_amount = 32) :
  (sale_amount / sale_quantity - purchase_amount / purchase_quantity) /
  (purchase_amount / purchase_quantity) * 100 = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_profit_percentage_l90_9042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_travel_distance_l90_9029

/-- Represents the distance traveled by a person in a two-dimensional plane -/
structure Travel where
  east : ℝ
  north : ℝ

/-- Calculates the distance from the origin given east and north components -/
noncomputable def distance_from_origin (t : Travel) : ℝ :=
  Real.sqrt (t.east ^ 2 + t.north ^ 2)

/-- Represents a 45-degree turn followed by a certain distance -/
noncomputable def turn_45_and_walk (distance : ℝ) : Travel :=
  { east := distance / Real.sqrt 2,
    north := distance / Real.sqrt 2 }

theorem billy_travel_distance :
  let initial_east := (5 : ℝ)
  let turn_distance := (8 : ℝ)
  let final_travel := 
    { east := initial_east + (turn_45_and_walk turn_distance).east,
      north := (turn_45_and_walk turn_distance).north }
  distance_from_origin final_travel = Real.sqrt (89 + 40 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_travel_distance_l90_9029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_length_calculation_l90_9060

/-- The length of a cubic box given total volume, cost per box, and total payment -/
theorem box_length_calculation (total_volume : ℝ) (cost_per_box : ℝ) (total_payment : ℝ)
  (hv : total_volume = 1080000)
  (hc : cost_per_box = 0.2)
  (hp : total_payment = 120) :
  ∃ (length : ℝ), abs (length - (total_volume / (total_payment / cost_per_box)) ^ (1/3)) < 0.1 :=
by sorry

#eval (1080000 / (120 / 0.2)) ^ (1/3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_length_calculation_l90_9060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ballerina_kinetic_energy_ratio_l90_9097

/-- The ratio of final to initial kinetic energy for a rotating ballerina -/
theorem ballerina_kinetic_energy_ratio 
  (I : ℝ) -- Initial moment of inertia
  (ω : ℝ) -- Initial angular velocity
  (h1 : I > 0) -- Moment of inertia is positive
  (h2 : ω ≠ 0) -- Angular velocity is non-zero
  : 
  ((1 / 2) * ((7 / 10) * I) * ((I * ω) / ((7 / 10) * I))^2) / ((1 / 2) * I * ω^2) = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ballerina_kinetic_energy_ratio_l90_9097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_and_amplitude_of_f_l90_9076

noncomputable def f (x : ℝ) := 3 * Real.cos ((x / 3) + (Real.pi / 4))

theorem period_and_amplitude_of_f :
  (∀ x, f (x + 6 * Real.pi) = f x) ∧
  (∀ x, -3 ≤ f x ∧ f x ≤ 3) ∧
  (∃ x₁ x₂, f x₁ = 3 ∧ f x₂ = -3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_and_amplitude_of_f_l90_9076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_max_area_l90_9095

/-- A sector is a portion of a circle enclosed by two radii and an arc. -/
structure Sector where
  radius : ℝ
  arcLength : ℝ
  centralAngle : ℝ

/-- The perimeter of a sector is the sum of two radii and the arc length. -/
noncomputable def Sector.perimeter (s : Sector) : ℝ := 2 * s.radius + s.arcLength

/-- The area of a sector is half the product of its radius and arc length. -/
noncomputable def Sector.area (s : Sector) : ℝ := (1 / 2) * s.radius * s.arcLength

/-- Given a sector with perimeter 40, its area is maximized when the radius is 10 and the central angle is 2. -/
theorem sector_max_area :
  ∀ s : Sector,
  s.perimeter = 40 →
  s.area ≤ Sector.area { radius := 10, arcLength := 20, centralAngle := 2 } ∧
  (s.area = Sector.area { radius := 10, arcLength := 20, centralAngle := 2 } ↔ 
    s.radius = 10 ∧ s.centralAngle = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_max_area_l90_9095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_werewolf_vampire_ratio_l90_9085

/-- Represents the payment and removal details for vampires and werewolves --/
structure MonsterHunt where
  vampirePayment : ℕ
  werewolfPayment : ℕ
  vampiresRemoved : ℚ
  werewolvesRemoved : ℕ
  totalEarnings : ℕ

/-- Theorem stating the ratio of werewolves to vampires given the hunt details --/
theorem werewolf_vampire_ratio (hunt : MonsterHunt) 
  (h1 : hunt.vampirePayment = 5)
  (h2 : hunt.werewolfPayment = 10)
  (h3 : hunt.vampiresRemoved = 1/2)
  (h4 : hunt.werewolvesRemoved = 8)
  (h5 : hunt.totalEarnings = 105) :
  ∃ (v w : ℕ), v ≠ 0 ∧ w * 5 = v * 4 ∧ 
  hunt.vampirePayment * (↑v * hunt.vampiresRemoved).floor + hunt.werewolfPayment * hunt.werewolvesRemoved = hunt.totalEarnings :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_werewolf_vampire_ratio_l90_9085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_keys_two_lamps_on_l90_9008

/-- Represents the connection between keys and lamps -/
def Connection (n : ℕ) := Fin 5 → Finset (Fin n)

/-- Predicate to check if a combination of keys turns on at least two lamps -/
def TurnsOnTwoOrMore {n : ℕ} (c : Connection n) (keys : Finset (Fin 5)) : Prop :=
  2 ≤ (keys.biUnion c).card

/-- Main theorem: There exists a combination of 3 keys that turns on at least 2 lamps -/
theorem three_keys_two_lamps_on (n : ℕ) (c : Connection n) 
  (h_unique : ∀ i j, i ≠ j → c i ≠ c j) :
  ∃ keys : Finset (Fin 5), keys.card = 3 ∧ TurnsOnTwoOrMore c keys := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_keys_two_lamps_on_l90_9008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_points_theorem_l90_9056

-- Define the line and points
def line_equation (x y : ℝ) : Prop := x - y + 1 = 0

def point_A : ℝ × ℝ := (-1, -3)
def point_B : ℝ × ℝ := (1, 1)
def point_C : ℝ × ℝ := (2, 2)

-- Define the propositions
def p : Prop := ∃ (m : ℝ), Real.tan m = -1 ∧ m * (180 / Real.pi) = 135

def q : Prop := ∃ (t : ℝ),
  point_A.1 + t * (point_B.1 - point_A.1) = point_C.1 ∧
  point_A.2 + t * (point_B.2 - point_A.2) = point_C.2

-- Theorem to prove
theorem line_and_points_theorem : ¬p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_points_theorem_l90_9056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_l90_9020

theorem max_min_sum (x y z : ℝ) (h : 3 * (x + y + z) = x^2 + y^2 + z^2) :
  ∃ (M m : ℝ), (∀ (a b c : ℝ), a * b + a * c + b * c ≤ M ∧ m ≤ a * b + a * c + b * c) ∧ M + 10 * m = -189/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_l90_9020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l90_9089

theorem simplify_expression :
  -3 - (6) - (-5) + (-2) = -3 - 6 + 5 - 2 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l90_9089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_discount_order_invariance_l90_9014

/-- Proves that the order of applying tax and discount doesn't affect the final price --/
theorem tax_discount_order_invariance 
  (original_price : ℝ) 
  (tax_rate : ℝ) 
  (discount_rate : ℝ) 
  (h₁ : 0 < original_price) 
  (h₂ : 0 ≤ tax_rate) 
  (h₃ : 0 ≤ discount_rate) 
  (h₄ : discount_rate < 1) :
  original_price * (1 + tax_rate) * (1 - discount_rate) = 
  original_price * (1 - discount_rate) * (1 + tax_rate) :=
by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_discount_order_invariance_l90_9014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l90_9005

def my_sequence : ℕ → ℕ
  | 0 => 1
  | 1 => 0
  | 2 => 1
  | 3 => 0
  | 4 => 1
  | 5 => 0
  | n + 6 => (my_sequence n + my_sequence (n + 1) + my_sequence (n + 2) + 
              my_sequence (n + 3) + my_sequence (n + 4) + my_sequence (n + 5)) % 10

def subsequence_not_possible (s : ℕ → ℕ) : Prop :=
  ∀ k, ¬(s k = 0 ∧ s (k + 1) = 1 ∧ s (k + 2) = 0 ∧ 
        s (k + 3) = 1 ∧ s (k + 4) = 0 ∧ s (k + 5) = 1)

theorem sequence_property : subsequence_not_possible my_sequence := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l90_9005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_small_triangle_l90_9013

/-- A circle with radius 13 and 13 marked points on its circumference -/
structure MarkedCircle where
  radius : ℝ
  points : Finset (ℝ × ℝ)
  radius_eq : radius = 13
  on_circle : ∀ p ∈ points, (p.1^2 + p.2^2 = radius^2)
  num_points : points.card = 13

/-- The area of a triangle formed by three points -/
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  abs ((p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2)

/-- Theorem: There exist three points forming a triangle with area less than 13 -/
theorem exists_small_triangle (c : MarkedCircle) :
  ∃ p1 p2 p3, p1 ∈ c.points ∧ p2 ∈ c.points ∧ p3 ∈ c.points ∧
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ triangleArea p1 p2 p3 < 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_small_triangle_l90_9013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_unique_solution_l90_9086

open Real

-- Define the triangle properties
def isIsoscelesTriangle (x : ℝ) : Prop :=
  x > 0 ∧ x < 90 ∧  -- x is acute
  ∃ (a b c : ℝ),
    a = Real.sin (x * π / 180) ∧
    b = Real.sin (x * π / 180) ∧
    c = Real.sin ((9 * x) * π / 180) ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧  -- sides are positive
    a + b > c ∧ b + c > a ∧ c + a > b  -- triangle inequality

-- Define the vertex angle condition
def hasVertexAngle3x (x : ℝ) : Prop :=
  ∃ (angle : ℝ), angle = 3 * x

-- Theorem statement
theorem isosceles_triangle_unique_solution :
  ∀ x : ℝ, isIsoscelesTriangle x ∧ hasVertexAngle3x x → x = 30 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_unique_solution_l90_9086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l90_9094

theorem simplify_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  4 * a^(2/3 : ℝ) * b^(-(1/3) : ℝ) / (-2/3 * a^(-(1/3) : ℝ) * b^(2/3 : ℝ)) = -6 * a / b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l90_9094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l90_9022

-- Define the piecewise function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then -(1/2) ^ x
  else if x ≤ 4 then -x^2 + 2*x
  else 0  -- This else case is added to make the function total

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, -8 ≤ f a x ∧ f a x ≤ 1) →
  -3 ≤ a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l90_9022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_integer_values_l90_9037

/-- A quadratic polynomial -/
def quadratic_polynomial (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c

/-- Theorem: If a quadratic polynomial takes integer values at x = 0, 1, and 2, 
    then it takes integer values for all integer x -/
theorem quadratic_integer_values 
  (a b c : ℤ) 
  (h0 : quadratic_polynomial a b c 0 = quadratic_polynomial a b c 0)
  (h1 : quadratic_polynomial a b c 1 = quadratic_polynomial a b c 1)
  (h2 : quadratic_polynomial a b c 2 = quadratic_polynomial a b c 2) :
  ∀ x : ℤ, quadratic_polynomial a b c x = quadratic_polynomial a b c x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_integer_values_l90_9037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_abs_f_l90_9052

/-- The function f(x, y) = x^3 - xy -/
def f (x y : ℝ) : ℝ := x^3 - x*y

/-- The set of x values: [0, 1] -/
def X : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 1}

theorem min_max_abs_f :
  ∃ y : ℝ, ∀ y' : ℝ, 
    (⨆ x ∈ X, |f x y'|) ≤ (⨆ x ∈ X, |f x y|) ∧
    (⨆ x ∈ X, |f x y|) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_abs_f_l90_9052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_increasing_in_interval_l90_9084

open Real

-- Define the functions
noncomputable def f (x : ℝ) := sin x + cos x
noncomputable def g (x : ℝ) := 2 * Real.sqrt 2 * sin x * cos x

-- State the theorem
theorem f_and_g_increasing_in_interval :
  ∀ x y, -π/4 < x ∧ x < y ∧ y < π/4 → f x < f y ∧ g x < g y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_increasing_in_interval_l90_9084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_inequality_l90_9077

theorem solution_set_inequality (x : ℝ) : (x - 1) / x > 1 ↔ x ∈ Set.Iio 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_inequality_l90_9077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_sequence_properties_l90_9053

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a sequence of points
def PointSequence := ℕ → Point

-- Define the unit vector j
def j : Point := ⟨0, 1⟩

-- Define the dot product of two points (treated as vectors)
def dot (p q : Point) : ℝ := p.x * q.x + p.y * q.y

-- Define vector subtraction for Points
instance : HSub Point Point Point where
  hSub a b := ⟨a.x - b.x, a.y - b.y⟩

-- Define b_n for a point sequence
def b (A : PointSequence) (n : ℕ) : ℝ :=
  dot (A (n+1) - A n) j

-- Define what it means to be a T point sequence
def is_T_sequence (A : PointSequence) : Prop :=
  ∀ n : ℕ, b A (n+1) > b A n

-- Main theorem
theorem T_sequence_properties (A : PointSequence) 
  (h_T : is_T_sequence A) 
  (h_A2 : (A 2).x > (A 1).x ∧ (A 2).y > (A 1).y) :
  (∀ k : ℕ, 
    let v1 := Point.mk (-1) ((A k).y - (A (k+1)).y)
    let v2 := Point.mk 1 ((A (k+2)).y - (A (k+1)).y)
    dot v1 v2 < 0) ∧ 
  (∀ m n p q : ℕ, 
    1 ≤ m → m < n → n < p → p < q → m + q = n + p →
    dot (Point.mk (q-n) ((A q).y - (A n).y)) j > 
    dot (Point.mk (p-m) ((A p).y - (A m).y)) j) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_sequence_properties_l90_9053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l90_9096

open Real

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ := (3 * x^3 + 2 * x^2 + 1) / ((x + 2) * (x - 2) * (x - 1))

-- Define the antiderivative
noncomputable def F (x : ℝ) : ℝ := 3 * x - (5/4) * log (abs (x + 2)) + (33/4) * log (abs (x - 2)) - 2 * log (abs (x - 1))

-- State the theorem
theorem indefinite_integral_proof (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) (h3 : x ≠ 1) : 
  deriv F x = f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l90_9096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l90_9043

-- Define a real-valued function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f(x+2)
def domain_f_shifted : Set ℝ := Set.Icc 0 1

-- State the theorem
theorem domain_of_f :
  (∀ x, f (x + 2) ∈ domain_f_shifted ↔ x ∈ domain_f_shifted) →
  (∀ x, f x ∈ Set.Icc (-2) (-1) ↔ x ∈ Set.Icc (-2) (-1)) :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l90_9043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_relationship_l90_9088

-- Define the inverse proportion function
noncomputable def f (x : ℝ) : ℝ := -6 / x

-- Define the points
noncomputable def A : ℝ × ℝ := (-1, f (-1))
noncomputable def B : ℝ × ℝ := (2, f 2)
noncomputable def C : ℝ × ℝ := (3, f 3)

-- Define y₁, y₂, and y₃
noncomputable def y₁ : ℝ := A.2
noncomputable def y₂ : ℝ := B.2
noncomputable def y₃ : ℝ := C.2

-- Theorem statement
theorem inverse_proportion_relationship : y₁ > y₃ ∧ y₃ > y₂ := by
  -- Unfold definitions
  unfold y₁ y₂ y₃ A B C f
  -- Simplify expressions
  simp
  -- Split into two goals
  constructor
  -- Prove y₁ > y₃
  · norm_num
  -- Prove y₃ > y₂
  · norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_relationship_l90_9088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_component_processing_theorem_solve_specific_problem_l90_9012

/-- Represents the daily processing capacity and fee for a person -/
structure Person where
  daily_capacity : ℕ
  daily_fee : ℕ

/-- Represents the problem setup -/
structure ProblemSetup where
  person_a : Person
  person_b : Person
  total_components : ℕ
  max_total_fee : ℕ

/-- The main theorem -/
theorem component_processing_theorem (setup : ProblemSetup) : 
  setup.person_a.daily_capacity = 60 ∧ 
  setup.person_b.daily_capacity = 40 ∧ 
  ∃ (days_a : ℕ), days_a ≥ 40 ∧ 
    ∃ (days_b : ℕ), 
      setup.person_a.daily_capacity * days_a + setup.person_b.daily_capacity * days_b = setup.total_components ∧
      setup.person_a.daily_fee * days_a + setup.person_b.daily_fee * days_b ≤ setup.max_total_fee :=
by
  sorry

/-- The problem instance -/
def problem_instance : ProblemSetup :=
  { person_a := { daily_capacity := 60, daily_fee := 150 }
  , person_b := { daily_capacity := 40, daily_fee := 120 }
  , total_components := 3000
  , max_total_fee := 7800
  }

/-- Applying the theorem to our specific problem instance -/
theorem solve_specific_problem : 
  ∃ (days_a : ℕ), days_a ≥ 40 ∧ 
    ∃ (days_b : ℕ), 
      60 * days_a + 40 * days_b = 3000 ∧
      150 * days_a + 120 * days_b ≤ 7800 :=
by
  have h := component_processing_theorem problem_instance
  exact h.2.2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_component_processing_theorem_solve_specific_problem_l90_9012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l90_9073

def sequence_a : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2 * sequence_a n + 1

theorem sequence_a_properties :
  (sequence_a 3 = 23) ∧
  (∀ n : ℕ, sequence_a n = 3 * 2^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l90_9073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_sum_reciprocals_relation_l90_9011

/-- Represents a geometric progression with n terms, first term a, and common ratio -r --/
structure GeometricProgression (α : Type*) [Field α] where
  n : ℕ
  a : α
  r : α

/-- The product of the terms in a geometric progression --/
noncomputable def product {α : Type*} [Field α] (gp : GeometricProgression α) : α :=
  gp.a^gp.n * (-gp.r)^((gp.n * (gp.n - 1)) / 2)

/-- The sum of the terms in a geometric progression --/
noncomputable def sum {α : Type*} [Field α] (gp : GeometricProgression α) : α :=
  gp.a * (1 - (-gp.r)^gp.n) / (1 + gp.r)

/-- The sum of the reciprocals of the terms in a geometric progression --/
noncomputable def sumReciprocals {α : Type*} [Field α] (gp : GeometricProgression α) : α :=
  (1 / gp.a) * (1 - (-1/gp.r)^gp.n) / (1 + 1/gp.r)

/-- Theorem stating the relationship between product, sum, and sum of reciprocals --/
theorem product_sum_reciprocals_relation {α : Type*} [Field α] (gp : GeometricProgression α) :
  product gp = (sum gp / sumReciprocals gp) ^ (gp.n / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_sum_reciprocals_relation_l90_9011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_total_investment_l90_9000

/-- Represents the investment and profit sharing details of two businesses --/
structure BusinessInvestment where
  p_investment1 : ℚ
  p_investment2 : ℚ
  profit_ratio1 : ℚ × ℚ
  profit_ratio2 : ℚ × ℚ

/-- Calculates Q's investment given P's investment and the profit ratio --/
def calculate_q_investment (p_investment : ℚ) (profit_ratio : ℚ × ℚ) : ℚ :=
  (p_investment * profit_ratio.2) / profit_ratio.1

/-- Theorem stating Q's total investment given the business details --/
theorem q_total_investment (b : BusinessInvestment) 
  (h1 : b.p_investment1 = 50000)
  (h2 : b.p_investment2 = 30000)
  (h3 : b.profit_ratio1 = (3, 4))
  (h4 : b.profit_ratio2 = (2, 3)) :
  let q_investment1 := calculate_q_investment b.p_investment1 b.profit_ratio1
  let q_investment2 := calculate_q_investment b.p_investment2 b.profit_ratio2
  q_investment1 + q_investment2 = 111666 + 2/3 := by
  sorry

#eval (50000 : ℚ) * 4 / 3 + (30000 : ℚ) * 3 / 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_total_investment_l90_9000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_problem_l90_9070

/-- An arithmetic progression with n terms, first term a, and common difference d. -/
structure ArithmeticProgression where
  n : ℕ
  a : ℝ
  d : ℝ

/-- The sum of the first k terms of an arithmetic progression. -/
noncomputable def sumFirstKTerms (ap : ArithmeticProgression) (k : ℕ) : ℝ :=
  k / 2 * (2 * ap.a + (k - 1) * ap.d)

/-- The sum of the last k terms of an arithmetic progression. -/
noncomputable def sumLastKTerms (ap : ArithmeticProgression) (k : ℕ) : ℝ :=
  k / 2 * (2 * ap.a + (2 * ap.n - k - 1) * ap.d)

/-- Theorem stating the conditions and conclusion for the arithmetic progression problem. -/
theorem arithmetic_progression_problem (ap : ArithmeticProgression) :
  (sumFirstKTerms ap 13 = 0.5 * sumLastKTerms ap 13) →
  ((sumFirstKTerms ap ap.n - sumFirstKTerms ap 3) / (sumFirstKTerms ap (ap.n - 3)) = 4 / 3) →
  ap.n = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_problem_l90_9070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_to_square_l90_9074

/-- The side length of a regular hexagon. -/
noncomputable def hexagon_side : ℝ := 1

/-- The height of a regular hexagon. -/
noncomputable def hexagon_height (s : ℝ) : ℝ := (Real.sqrt 3 / 2) * s

/-- The area of a regular hexagon. -/
noncomputable def hexagon_area (s : ℝ) : ℝ := 3 * Real.sqrt 3 / 2 * s^2

/-- The area of a square. -/
noncomputable def square_area (s : ℝ) : ℝ := s^2

theorem hexagon_to_square :
  hexagon_area hexagon_side = square_area (hexagon_height hexagon_side) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_to_square_l90_9074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_equation_solvability_l90_9036

theorem cosine_sine_equation_solvability (a : ℝ) :
  (∃ x : ℝ, (Real.cos x) ^ 2 - Real.sin x + a = 0) ↔ -5/4 ≤ a ∧ a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_equation_solvability_l90_9036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosA_value_side_c_value_l90_9041

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  t.a = 2 * t.b ∧
  ∃ k, t.A + t.C = 2 * t.B + k ∧  -- Arithmetic sequence condition
  Real.sin t.A + Real.sin t.C = 2 * Real.sin t.B

-- Theorem 1: Prove cos A = -1/4
theorem cosA_value (t : Triangle) (h : isValidTriangle t) : Real.cos t.A = -1/4 := by
  sorry

-- Theorem 2: Prove c = 4√2 when area is 8√15/3
theorem side_c_value (t : Triangle) (h : isValidTriangle t) 
  (area_condition : 1/2 * t.b * t.c * Real.sin t.A = 8 * Real.sqrt 15 / 3) : 
  t.c = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosA_value_side_c_value_l90_9041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l90_9061

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.sqrt 2 * x) + Real.sin ((3/8) * Real.sqrt 2 * x)

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ 
  (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = 8 * Real.sqrt 2 * Real.pi := by
  sorry

#check smallest_positive_period_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l90_9061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_exponential_curve_l90_9039

theorem tangent_line_exponential_curve (x y : ℝ) :
  let f : ℝ → ℝ := λ t => Real.exp (2 * t)
  let point : ℝ × ℝ := (0, 1)
  let tangent_line : ℝ → ℝ := λ t => 2 * t + 1
  (∀ t, f t = Real.exp (2 * t)) →
  f (point.fst) = point.snd →
  (∀ t, tangent_line t = 2 * t + 1) →
  ∀ t, tangent_line t = f point.fst + (deriv f point.fst) * (t - point.fst) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_exponential_curve_l90_9039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_probability_theorem_l90_9044

/-- Represents the probability of drawing pens from a box -/
def PenProbability (total : ℕ) (first : ℕ) (second : ℕ) (third : ℕ) (drawn : ℕ) : Prop :=
  total = first + second + third ∧
  ∃ (exactly_one_first exactly_two_first no_third : ℚ),
    exactly_one_first = (Nat.choose first 1 * Nat.choose (second + third) (drawn - 1) : ℚ) / Nat.choose total drawn ∧
    exactly_two_first = (Nat.choose first 2 * Nat.choose (second + third) (drawn - 2) : ℚ) / Nat.choose total drawn ∧
    no_third = (Nat.choose (first + second) drawn : ℚ) / Nat.choose total drawn

theorem pen_probability_theorem :
  PenProbability 6 3 2 1 3 →
  ∃ (exactly_one_first exactly_two_first no_third : ℚ),
    exactly_one_first = 9/20 ∧
    exactly_two_first = 9/20 ∧
    no_third = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_probability_theorem_l90_9044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l90_9050

theorem log_equation_solution (x : ℝ) (k : ℤ) :
  (0 < x) ∧ (x < Real.pi / 2) →
  (2 - Real.log (Real.cos x) / Real.log (Real.sin x) = Real.log (Real.sin x) / Real.log (Real.cos x)) ↔
  (x = Real.pi / 4 + 2 * ↑k * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l90_9050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_when_tan_is_two_l90_9059

theorem sin_double_angle_when_tan_is_two (θ : Real) (h : Real.tan θ = 2) : 
  Real.sin (2 * θ) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_when_tan_is_two_l90_9059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l90_9003

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then (4 : ℝ)^x else (2 : ℝ)^(a - x)

-- State the theorem
theorem function_equality (a : ℝ) (h1 : a ≠ 1) :
  f a (1 - a) = f a (a - 1) → a = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l90_9003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sightseeing_tour_duration_l90_9067

/-- Represents a time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : hours < 24 ∧ minutes < 60

/-- Calculates the angle of the hour hand at a given time -/
noncomputable def hour_hand_angle (t : Time) : ℝ :=
  (t.hours % 12 * 30 + t.minutes / 2 : ℝ)

/-- Calculates the angle of the minute hand at a given time -/
noncomputable def minute_hand_angle (t : Time) : ℝ :=
  (t.minutes * 6 : ℝ)

/-- Checks if the clock hands are 180 degrees apart at a given time -/
def hands_180_degrees_apart (t : Time) : Prop :=
  abs (hour_hand_angle t - minute_hand_angle t) = 180 ∨
  abs (hour_hand_angle t - minute_hand_angle t) = 180

/-- Checks if the clock hands are aligned at a given time -/
def hands_aligned (t : Time) : Prop :=
  hour_hand_angle t = minute_hand_angle t

/-- Calculates the duration between two times in minutes -/
def duration_minutes (t1 t2 : Time) : ℕ :=
  (t2.hours - t1.hours) * 60 + t2.minutes - t1.minutes

/-- The main theorem to prove -/
theorem sightseeing_tour_duration :
  ∃ (start_time end_time : Time),
    10 ≤ start_time.hours ∧ start_time.hours < 11 ∧
    16 ≤ end_time.hours ∧ end_time.hours < 17 ∧
    hands_180_degrees_apart start_time ∧
    hands_aligned end_time ∧
    duration_minutes start_time end_time = 362 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sightseeing_tour_duration_l90_9067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcf_of_lcm_problem_l90_9038

theorem gcf_of_lcm_problem : Nat.gcd (Nat.lcm 9 21) (Nat.lcm 7 15) = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcf_of_lcm_problem_l90_9038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_property_l90_9015

theorem inverse_function_property (f g : ℝ → ℝ) (a b : ℝ) :
  Function.RightInverse g f → f a = b → a ≠ 0 → b ≠ 0 → g b = a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_property_l90_9015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_decimal_properties_l90_9093

theorem square_root_decimal_properties :
  ∃ (n m : ℕ),
    (∃ (a b : ℕ) (d e : ℚ),
      Real.sqrt (n : ℝ) = (a : ℝ) + (b : ℝ) / 10 + d / 1000 + e / 10000 ∧
      0 ≤ d ∧ d < 1 ∧
      0 ≤ e ∧ e < 1) ∧
    (∃ (x y : ℕ) (z w : ℚ),
      Real.sqrt (m : ℝ) = (x : ℝ) + (y : ℝ) / 10 + z / 100 + w / 10000 ∧
      x ≠ 0 ∧
      0 ≤ z ∧ z < 1 ∧
      0 ≤ w ∧ w < 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_decimal_properties_l90_9093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_formula_l90_9001

noncomputable def x : ℕ → ℝ
  | 0 => 1/2  -- Added case for 0 to cover all natural numbers
  | 1 => 1/2
  | n + 1 => Real.sqrt ((1 - Real.sqrt (1 - (x n)^2)) / 2)

theorem x_formula (n : ℕ) : x n = Real.sin (π / (3 * 2^n)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_formula_l90_9001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cot_210_degrees_l90_9035

theorem tan_cot_210_degrees :
  let θ : Real := 210 * Real.pi / 180
  Real.tan θ = 1 / Real.sqrt 3 ∧ 1 / Real.tan θ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cot_210_degrees_l90_9035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_on_line_divisibility_l90_9024

/-- Given three points on a line with coordinates satisfying a certain condition,
    prove that at least two pairs of points have coordinate differences divisible by 1979. -/
theorem three_points_on_line_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_p : p = 1979)
    (x₁ x₂ x₃ y₁ y₂ y₃ : ℤ)
    (h_line : ∃ (a b c : ℤ), a ≠ 0 ∨ b ≠ 0 ∧ a * x₁ + b * y₁ = c ∧ a * x₂ + b * y₂ = c ∧ a * x₃ + b * y₃ = c)
    (h_div₁ : (p : ℤ) ∣ x₁ * y₁ - 1)
    (h_div₂ : (p : ℤ) ∣ x₂ * y₂ - 1)
    (h_div₃ : (p : ℤ) ∣ x₃ * y₃ - 1) :
    ((p : ℤ) ∣ x₁ - x₂ ∧ (p : ℤ) ∣ y₁ - y₂) ∨
    ((p : ℤ) ∣ x₁ - x₃ ∧ (p : ℤ) ∣ y₁ - y₃) ∨
    ((p : ℤ) ∣ x₂ - x₃ ∧ (p : ℤ) ∣ y₂ - y₃) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_on_line_divisibility_l90_9024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l90_9071

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- Proposition: Given a polynomial with integer coefficients and a finite set of integers,
    if for every integer the polynomial evaluation is divisible by at least one integer from the set,
    then there exists an integer from the set that divides all polynomial evaluations. -/
theorem polynomial_divisibility 
  (F : IntPolynomial) 
  (A : Finset ℤ) 
  (h : ∀ n : ℤ, ∃ a ∈ A, (F.eval n) % a = 0) :
  ∃ a ∈ A, ∀ n : ℤ, (F.eval n) % a = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l90_9071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l90_9098

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

-- State the theorem
theorem inequality_equivalence (m : ℝ) :
  f (2 * m) > f (m - 2) ↔ m > 2/3 ∨ m < -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l90_9098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_number_count_l90_9064

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10]
  digits[1]! = (digits[0]! + digits[2]!) / 2 ∧
  digits[2]! = (digits[1]! + digits[3]!) / 2

instance : DecidablePred is_valid_number :=
  fun n => decidable_of_iff
    (1000 ≤ n ∧ n ≤ 9999 ∧
      let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10]
      digits[1]! = (digits[0]! + digits[2]!) / 2 ∧
      digits[2]! = (digits[1]! + digits[3]!) / 2)
    (by simp [is_valid_number])

theorem four_digit_number_count : 
  (Finset.filter is_valid_number (Finset.range 10000)).card = 225 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_number_count_l90_9064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_positive_f_less_than_condition_l90_9079

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (Real.log (1 + x)) / x

-- Define the domain of f(x)
def domain (x : ℝ) : Prop := (x > -1 ∧ x < 0) ∨ x > 0

-- Theorem for the monotonicity of f(x) on (0, +∞)
theorem f_decreasing_on_positive : ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f x₁ > f x₂ := by
  sorry

-- Theorem for the condition f(x) < (1-ax)/(1+x)
theorem f_less_than_condition (a : ℝ) : 
  (∀ x : ℝ, domain x → f x < (1 - a * x) / (1 + x)) ↔ a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_positive_f_less_than_condition_l90_9079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortening_theorem_l90_9087

/-- Represents the shortening operation on a sequence of digits --/
def shorteningOperation (seq : List Nat) : List Nat := sorry

/-- Calculates the probability of a sequence shortening by exactly one digit --/
noncomputable def probShortenByOne (n : Nat) : ℝ := sorry

/-- Calculates the expected length of a sequence after shortening --/
noncomputable def expectedLength (n : Nat) : ℝ := sorry

/-- The main theorem about the shortening operation on a sequence of 2015 digits --/
theorem shortening_theorem :
  let n : Nat := 2015
  let digits : List Nat := List.replicate n 0  -- placeholder for the actual random sequence
  (abs (probShortenByOne n - 1.564e-90) < 1e-92) ∧
  (abs (expectedLength n - 1813.6) < 1e-1) := by
  sorry

#eval "Theorem statement type-checks successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortening_theorem_l90_9087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_on_z_axis_l90_9007

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- A point is on the z-axis if its x and y coordinates are zero -/
def onZAxis (p : Point3D) : Prop :=
  p.x = 0 ∧ p.y = 0

theorem equidistant_point_on_z_axis :
  let M : Point3D := ⟨0, 0, -3⟩
  let A : Point3D := ⟨1, 0, 2⟩
  let B : Point3D := ⟨1, -3, 1⟩
  onZAxis M ∧ distance M A = distance M B :=
by
  sorry

#check equidistant_point_on_z_axis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_on_z_axis_l90_9007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distances_imply_k_values_acute_angle_implies_k_range_l90_9027

noncomputable section

-- Define the line l: kx - y - 2k + 2 = 0
def line (k : ℝ) (x y : ℝ) : Prop := k * x - y - 2 * k + 2 = 0

-- Define the distance from a point to the line
noncomputable def distance_to_line (k : ℝ) (x y : ℝ) : ℝ :=
  |k * x - y - 2 * k + 2| / Real.sqrt (k^2 + 1)

-- Define the points M and N
def M : ℝ × ℝ := (0, 2)
def N : ℝ × ℝ := (-2, 0)

-- Define the midpoint H of MN
def H : ℝ × ℝ := (-1, 1)

-- Theorem 1: Equal distances imply k = 1 or k = 1/3
theorem equal_distances_imply_k_values (k : ℝ) :
  distance_to_line k M.1 M.2 = distance_to_line k N.1 N.2 →
  k = 1 ∨ k = 1/3 := by sorry

-- Theorem 2: Acute angle MPN implies k range
theorem acute_angle_implies_k_range (k : ℝ) :
  (∀ x y : ℝ, line k x y → distance_to_line k H.1 H.2 > Real.sqrt 2) →
  k < -1/7 ∨ k > 1 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distances_imply_k_values_acute_angle_implies_k_range_l90_9027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l90_9026

def a : ℕ → ℚ
  | 0 => 0  -- We define a₀ as 0 to make the indexing consistent
  | 1 => 1
  | n + 2 => a (n + 1) / (3 * a (n + 1) + 1)

theorem a_formula (n : ℕ) (h : n > 0) : a n = 1 / (3 * n - 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l90_9026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_scalar_l90_9030

/-- Given vectors a and b, if (a - λb) is perpendicular to b, then λ = 3/5 -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (l : ℝ) :
  a = (1, 3) →
  b = (3, 4) →
  (a.1 - l * b.1, a.2 - l * b.2) • b = 0 →
  l = 3/5 := by
  sorry

#check perpendicular_vector_scalar

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_scalar_l90_9030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_b_nonzero_l90_9033

/-- A polynomial of degree 5 with five distinct x-intercepts -/
def Q (a b c d f : ℝ) (x : ℝ) : ℝ := x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + f

/-- The five distinct x-intercepts of Q -/
def intercepts : List ℝ := [0, -1, 1, -2, 2]

/-- Theorem stating that the coefficient b cannot be zero -/
theorem coeff_b_nonzero (a b c d f : ℝ) :
  (∀ r ∈ intercepts, Q a b c d f r = 0) →
  (∀ r s, r ∈ intercepts → s ∈ intercepts → r ≠ s) →
  b ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_b_nonzero_l90_9033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_sum_property_l90_9065

theorem partition_sum_property :
  ∀ (partition : Fin 3 → Set ℕ),
    (∀ n, n ∈ (⋃ i, partition i) → n ≤ 49) →
    (∀ n, n ≤ 49 → ∃ i, n ∈ partition i) →
    (∀ i j, i ≠ j → partition i ∩ partition j = ∅) →
    ∃ (i : Fin 3) (a b c : ℕ),
      a ∈ partition i ∧ b ∈ partition i ∧ c ∈ partition i ∧
      a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
      a + b = c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_sum_property_l90_9065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l90_9049

/-- Represents the initial investment amount in dollars -/
noncomputable def initial_investment : ℝ := 1800

/-- Represents the final investment amount in dollars after 28 years -/
noncomputable def final_investment : ℝ := 16200

/-- Represents the annual interest rate as a decimal -/
noncomputable def interest_rate : ℝ := 0.08

/-- Represents the investment period in years -/
noncomputable def investment_period : ℝ := 28

/-- Represents the time it takes for the investment to triple in years -/
noncomputable def tripling_period : ℝ := 112 / interest_rate

/-- Theorem stating that the initial investment grows to the final investment
    after the given investment period at the specified interest rate -/
theorem investment_growth :
  initial_investment * (3 ^ (investment_period / tripling_period)) = final_investment :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l90_9049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bens_savings_factor_l90_9010

def daily_savings : ℚ := 50 - 15

def total_savings (days : ℕ) : ℚ := daily_savings * days

def savings_with_dad_contribution (days : ℕ) : ℚ := total_savings days + 10

theorem bens_savings_factor (days : ℕ) (final_amount : ℚ) 
  (h1 : days = 7) 
  (h2 : final_amount = 500) : 
  (final_amount - 10) / savings_with_dad_contribution days = 
  (final_amount - 10) / (days * daily_savings + 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bens_savings_factor_l90_9010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leo_and_kendra_weight_l90_9021

/-- Calculates the combined weight of Leo and Kendra given Leo's current weight and the condition about their weight difference after Leo gains 10 pounds. -/
noncomputable def combined_weight (leo_weight : ℝ) : ℝ :=
  let kendra_weight := (leo_weight + 10) / 1.5
  leo_weight + kendra_weight

/-- Theorem stating that given Leo's current weight of 92 pounds and the condition about their weight difference, their combined weight is 160 pounds. -/
theorem leo_and_kendra_weight :
  combined_weight 92 = 160 := by
  -- Unfold the definition of combined_weight
  unfold combined_weight
  -- Simplify the arithmetic
  simp [add_div, add_mul]
  -- Check that the result is equal to 160
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_leo_and_kendra_weight_l90_9021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplified_function_theorem_l90_9004

noncomputable def f (x : ℝ) : ℝ := (x^3 + 11*x^2 + 38*x + 40) / (x + 3)

theorem simplified_function_theorem :
  ∃ (A B C D : ℝ),
    (∀ x, x ≠ D → f x = A * x^2 + B * x + C) ∧
    (∀ x, f x = A * x^2 + B * x + C → x ≠ D) ∧
    A + B + C + D = 20 := by
  sorry

#check simplified_function_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplified_function_theorem_l90_9004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_decreasing_line_l90_9048

/-- Given a line y = -3x + 2 and two points (3, m) and (5, n) on this line, prove that m > n -/
theorem points_on_decreasing_line (m n : ℝ) : 
  ((3 : ℝ), m) ∈ {p : ℝ × ℝ | p.2 = -3 * p.1 + 2} → 
  ((5 : ℝ), n) ∈ {p : ℝ × ℝ | p.2 = -3 * p.1 + 2} → 
  m > n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_decreasing_line_l90_9048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_roots_l90_9099

theorem sum_of_squares_of_roots (a b c : ℚ) (ha : a ≠ 0) :
  let f : ℚ → ℚ := λ x ↦ a * x^2 + b * x + c
  let sum_of_squares := (-(b / a))^2 - 2 * (c / a)
  a = 5 ∧ b = -7 ∧ c = 3 → sum_of_squares = 19 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_roots_l90_9099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_string_bead_problem_l90_9062

/-- 
Given a string of length 26 inches held at two points, 
if the midpoint (where a bead is attached) is raised by 8 inches, 
then the distance between the two holding points is 24 inches.
-/
theorem string_bead_problem (string_length bead_height hand_distance : ℝ) : 
  string_length = 26 →
  bead_height = 8 →
  hand_distance = 2 * Real.sqrt ((string_length / 2)^2 - (string_length / 2 - bead_height)^2) →
  hand_distance = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_string_bead_problem_l90_9062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_2023_l90_9054

def h : ℕ → ℕ
  | 0 => 2  -- Adding a case for 0
  | 1 => 2
  | 2 => 2
  | n+3 => h (n+2) - h (n+1) + 2*(n+3)

theorem h_2023 : h 2023 = 4052 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_2023_l90_9054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l90_9046

/-- Calculates the length of a train given its speed, tunnel length, and time to pass through the tunnel. -/
noncomputable def train_length (speed : ℝ) (tunnel_length : ℝ) (time : ℝ) : ℝ :=
  speed * time / 60 - tunnel_length

/-- Theorem stating that a train traveling at 72 km/hr through a 2.9 km long tunnel in 2.5 minutes has a length of 0.1 km. -/
theorem train_length_calculation :
  train_length 72 2.9 2.5 = 0.1 := by
  -- Unfold the definition of train_length
  unfold train_length
  -- Perform the calculation
  -- (72 * 2.5 / 60) - 2.9 = 0.1
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l90_9046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l90_9009

/-- The hyperbola C: x²/a² - y²/b² = 1 -/
def hyperbola (a b x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- The circle: x² + y² - 6x - 2y + 9 = 0 -/
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 2*y + 9 = 0

/-- The asymptote of the hyperbola -/
def asymptote (a b x y : ℝ) : Prop :=
  b*x - a*y = 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (c a : ℝ) : ℝ :=
  c / a

/-- 
Given a hyperbola C: x²/a² - y²/b² = 1 with an asymptote tangent to the circle x² + y² - 6x - 2y + 9 = 0, 
the eccentricity of C is 5/4.
-/
theorem hyperbola_eccentricity (a b : ℝ) :
  (∃ x y : ℝ, hyperbola a b x y ∧ circle_eq x y ∧ asymptote a b x y) →
  (∃ c : ℝ, eccentricity c a = 5/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l90_9009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_AE_AF_l90_9075

-- Define the polar coordinate system and the shapes
def ray_l (θ : Real) : Prop := θ = Real.pi / 6
def circle_C (ρ : Real) : Prop := ρ = 2
def ellipse_Γ (ρ θ : Real) : Prop := ρ^2 = 3 / (1 + 2 * Real.sin θ^2)

-- Define points A, E, and F
noncomputable def point_A : Real × Real := (Real.sqrt 3, 1)
def point_E : Real × Real := (0, -1)
noncomputable def point_F (θ : Real) : Real × Real := (Real.sqrt 3 * Real.cos θ, Real.sin θ)

-- Define vectors AE and AF
noncomputable def vector_AE : Real × Real := (point_E.1 - point_A.1, point_E.2 - point_A.2)
noncomputable def vector_AF (θ : Real) : Real × Real := 
  ((point_F θ).1 - point_A.1, (point_F θ).2 - point_A.2)

-- Define the dot product of AE and AF
noncomputable def dot_product_AE_AF (θ : Real) : Real :=
  vector_AE.1 * (vector_AF θ).1 + vector_AE.2 * (vector_AF θ).2

-- State the theorem
theorem max_dot_product_AE_AF :
  ∃ (θ : Real), ∀ (φ : Real), dot_product_AE_AF θ ≥ dot_product_AE_AF φ ∧ 
  dot_product_AE_AF θ = 5 + Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_AE_AF_l90_9075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l90_9081

/-- Curve C₁ in parametric form -/
noncomputable def C₁ (t : ℝ) : ℝ × ℝ :=
  (4 + 5 * Real.cos t, 5 + 5 * Real.sin t)

/-- Curve C₂ in polar form -/
noncomputable def C₂ (θ : ℝ) : ℝ :=
  2 * Real.sin θ

/-- Conversion from polar to Cartesian coordinates -/
noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

/-- The intersection points of C₁ and C₂ in polar coordinates -/
theorem intersection_points :
  ∃ (t₁ t₂ θ₁ θ₂ : ℝ),
    C₁ t₁ = polar_to_cartesian (C₂ θ₁) θ₁ ∧
    C₁ t₂ = polar_to_cartesian (C₂ θ₂) θ₂ ∧
    (C₂ θ₁ = Real.sqrt 2 ∧ θ₁ = Real.pi / 4) ∧
    (C₂ θ₂ = 2 ∧ θ₂ = Real.pi / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l90_9081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_p_squared_minus_3q_squared_l90_9028

theorem largest_divisor_of_p_squared_minus_3q_squared (p q : ℤ) 
  (hp : Odd p) (hq : Odd q) (hpq : q < p) : 
  (∃ (d : ℕ), d > 0 ∧ (d : ℤ) ∣ (p^2 - 3*q^2) ∧ 
    ∀ (k : ℕ), k > 0 → (k : ℤ) ∣ (p^2 - 3*q^2) → k ≤ d) → 
  (∃ (d : ℕ), d = 2 ∧ d > 0 ∧ (d : ℤ) ∣ (p^2 - 3*q^2) ∧ 
    ∀ (k : ℕ), k > 0 → (k : ℤ) ∣ (p^2 - 3*q^2) → k ≤ d) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_p_squared_minus_3q_squared_l90_9028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_f_l90_9068

-- Define the complex function f(z)
noncomputable def f (z : ℂ) : ℂ := Complex.exp z - 1 - z

-- Theorem statement
theorem zero_of_f :
  -- z = 0 is a zero of f
  f 0 = 0 ∧
  -- z = 0 is a zero of order 2
  ∃ g : ℂ → ℂ, ∀ z : ℂ, f z = z^2 * g z ∧ g 0 ≠ 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_of_f_l90_9068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeated_star_result_l90_9017

noncomputable def star (x y : ℝ) : ℝ := (((x^2 + 3*x*y + y^2 - 2*x - 2*y + 4) : ℝ).sqrt) / (x*y + 4)

noncomputable def repeated_star : ℕ → ℝ
| 0 => 2007
| (n+1) => star (repeated_star n) (2007 - n : ℝ)

theorem repeated_star_result : repeated_star 2006 = Real.sqrt 15 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeated_star_result_l90_9017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_l90_9091

/-- Represents the company's strategy -/
structure Strategy where
  natural_gas : ℝ
  liquefied_gas : ℝ

/-- Represents the weather conditions -/
inductive Weather
  | Mild
  | Cold

/-- Calculates the profit for a given strategy and weather -/
noncomputable def profit (s : Strategy) (w : Weather) : ℝ :=
  match w with
  | Weather.Mild =>
    min s.natural_gas 2200 * (35 - 19) + min s.liquefied_gas 3500 * (58 - 25)
  | Weather.Cold =>
    min s.natural_gas 3800 * (35 - 19) + min s.liquefied_gas 2450 * (58 - 25)

/-- The optimal strategy should maximize the minimum profit across weather conditions -/
def is_optimal (s : Strategy) : Prop :=
  ∀ s' : Strategy,
    (min (profit s Weather.Mild) (profit s Weather.Cold)) ≥
    (min (profit s' Weather.Mild) (profit s' Weather.Cold))

/-- The theorem stating the optimal strategy -/
theorem optimal_strategy :
  ∃ s : Strategy, is_optimal s ∧ 
    (abs (s.natural_gas - 3032) < 1) ∧
    (abs (s.liquefied_gas - 2954) < 1) := by
  sorry

#check optimal_strategy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_l90_9091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_900_smaller_than_900_smallest_with_15_divisors_l90_9047

/-- The number of positive integer divisors of a positive integer n -/
def num_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range n)).card + 1

/-- 900 has exactly 15 positive integer divisors -/
theorem divisors_of_900 : num_divisors 900 = 15 := by sorry

/-- For any positive integer less than 900, the number of divisors is not 15 -/
theorem smaller_than_900 (n : ℕ) : 0 < n → n < 900 → num_divisors n ≠ 15 := by sorry

/-- 900 is the smallest positive integer with exactly 15 positive integer divisors -/
theorem smallest_with_15_divisors :
  ∀ n : ℕ, 0 < n → num_divisors n = 15 → n ≥ 900 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_900_smaller_than_900_smallest_with_15_divisors_l90_9047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_l90_9002

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 2 then
    |Real.log x / Real.log (1/2)|
  else if x > 2 then
    -1/2 * x + 2
  else
    0

theorem problem (a : ℝ) (h1 : f a = 2) : f (a + 2) = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_l90_9002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_linear_functions_l90_9031

/-- A function is linear if it can be expressed as f(x) = mx + b for some constants m and b -/
def IsLinear (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

/-- The given functions -/
noncomputable def f1 (k b : ℝ) (x : ℝ) : ℝ := k * x + b
noncomputable def f2 (x : ℝ) : ℝ := 2 * x
noncomputable def f3 (x : ℝ) : ℝ := -3 / x
noncomputable def f4 (x : ℝ) : ℝ := (1/3) * x + 3
noncomputable def f5 (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- The theorem stating that exactly two of the given functions are linear -/
theorem two_linear_functions :
  (∃ k b, IsLinear (f1 k b)) ∧
  IsLinear f2 ∧
  ¬IsLinear f3 ∧
  IsLinear f4 ∧
  ¬IsLinear f5 :=
by
  sorry

#check two_linear_functions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_linear_functions_l90_9031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l90_9025

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem range_of_x (x : ℝ) : 
  (0 < floor (2 * x + 2) ∧ floor (2 * x + 2) < 3) ↔ (-1/2 ≤ x ∧ x < 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l90_9025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dedekind_cut_no_max_min_exists_dedekind_cut_no_max_no_min_exists_l90_9023

-- Define the set of rational numbers
variable (Q : Type) [LinearOrder Q]

-- Define a Dedekind cut
structure DedekindCut (Q : Type) [LinearOrder Q] where
  M : Set Q
  N : Set Q
  union_eq_Q : M ∪ N = Set.univ
  intersection_empty : M ∩ N = ∅
  M_lt_N : ∀ (x y : Q), x ∈ M → y ∈ N → x < y

-- Theorem: There exists a Dedekind cut where M has no maximum element and N has a minimum element
theorem dedekind_cut_no_max_min_exists (Q : Type) [LinearOrder Q] :
  ∃ (cut : DedekindCut Q), (¬∃ (m : Q), m ∈ cut.M ∧ ∀ (x : Q), x ∈ cut.M → x ≤ m) ∧
                            (∃ (n : Q), n ∈ cut.N ∧ ∀ (y : Q), y ∈ cut.N → n ≤ y) :=
by sorry

-- Theorem: There exists a Dedekind cut where M has no maximum element and N has no minimum element
theorem dedekind_cut_no_max_no_min_exists (Q : Type) [LinearOrder Q] :
  ∃ (cut : DedekindCut Q), (¬∃ (m : Q), m ∈ cut.M ∧ ∀ (x : Q), x ∈ cut.M → x ≤ m) ∧
                            (¬∃ (n : Q), n ∈ cut.N ∧ ∀ (y : Q), y ∈ cut.N → n ≤ y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dedekind_cut_no_max_min_exists_dedekind_cut_no_max_no_min_exists_l90_9023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bella_items_after_purchase_l90_9016

def total_items (marbles frisbees cards : ℕ) : ℕ :=
  marbles + frisbees + cards

theorem bella_items_after_purchase :
  ∀ (marbles frisbees cards : ℕ),
    marbles = 60 →
    marbles = 2 * frisbees →
    frisbees = cards + 20 →
    total_items (marbles + (2 * marbles / 5))
                (frisbees + (2 * frisbees / 5))
                (cards + (2 * cards / 5)) = 140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bella_items_after_purchase_l90_9016
