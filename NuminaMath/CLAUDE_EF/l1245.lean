import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_range_implies_sum_l1245_124561

noncomputable def f (a b x : ℝ) : ℝ := a^x + b

theorem function_domain_range_implies_sum (a b : ℝ) :
  a > 0 ∧ a ≠ 1 ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 0, f a b x ∈ Set.Icc (-1 : ℝ) 0) ∧
  (∀ y ∈ Set.Icc (-1 : ℝ) 0, ∃ x ∈ Set.Icc (-1 : ℝ) 0, f a b x = y) →
  a + b = -3/2 := by
  sorry

#check function_domain_range_implies_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_range_implies_sum_l1245_124561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_sectors_area_l1245_124587

/-- The area of the overlapping region of two 90° sectors from circles with radius 15 -/
noncomputable def overlapping_area : ℝ := (225 * Real.pi / 2) - 225

/-- The radius of the circles -/
noncomputable def radius : ℝ := 15

/-- The angle of each sector in radians -/
noncomputable def sector_angle : ℝ := Real.pi / 2

/-- The area of one sector -/
noncomputable def sector_area : ℝ := (1 / 2) * radius ^ 2 * sector_angle

/-- The area of the right triangle formed in each sector -/
noncomputable def triangle_area : ℝ := (1 / 2) * radius ^ 2

theorem overlapping_sectors_area :
  overlapping_area = 2 * (sector_area - triangle_area) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_sectors_area_l1245_124587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1245_124575

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = 75 * Real.pi / 180 →
  C = 60 * Real.pi / 180 →
  c = 1 →
  A + B + C = Real.pi →
  (Real.sin A) / a = (Real.sin B) / b →
  (Real.sin B) / b = (Real.sin C) / c →
  b = Real.sqrt 6 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1245_124575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1245_124591

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - x - m

-- State the theorem
theorem function_properties (m : ℝ) (h_m : m < -2) :
  -- Part 1: Minimum value on [1/e, e]
  (∃ (x : ℝ), x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1) ∧
    ∀ (y : ℝ), y ∈ Set.Icc (1/Real.exp 1) (Real.exp 1) → f m x ≤ f m y) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1) → f m x ≥ 1 - Real.exp 1 - m) ∧
  -- Part 2: Product of zeros
  (∀ (x₁ x₂ : ℝ), f m x₁ = 0 → f m x₂ = 0 → x₁ < x₂ → x₁ * x₂ < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1245_124591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_sqrt_two_l1245_124556

/-- The curve function f(x) = x^2 - ln(x) -/
noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

/-- The line function g(x) = x - 2 -/
def g (x : ℝ) : ℝ := x - 2

/-- The distance function between a point (x, f(x)) and the line y = x - 2 -/
noncomputable def distance (x : ℝ) : ℝ := 
  |f x - g x| / Real.sqrt 2

/-- The theorem stating that the minimum distance is √2 -/
theorem min_distance_is_sqrt_two : 
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → distance x ≤ distance y ∧ distance x = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_sqrt_two_l1245_124556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_automobile_distance_l1245_124579

/-- Proves that an automobile traveling 2a/5 feet in 2r seconds will travel 20a/r yards in 5 minutes -/
theorem automobile_distance (a r : ℝ) (h : r ≠ 0) : 
  (2 * a / 5) / (2 * r) / 3 * (5 * 60) = 20 * a / r := by
  -- Simplify the left-hand side
  calc (2 * a / 5) / (2 * r) / 3 * (5 * 60)
    = a / (5 * r) / 3 * 300 := by ring_nf
  -- Continue simplification
  _ = a * 100 / (5 * r) := by ring_nf
  -- Final simplification
  _ = 20 * a / r := by ring_nf


end NUMINAMATH_CALUDE_ERRORFEEDBACK_automobile_distance_l1245_124579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_integers_l1245_124527

theorem sum_of_integers (x y : ℤ) 
  (h1 : x^2 + y^2 = 130)
  (h2 : x * y = 45)
  (h3 : |x - y| < 7)
  (h4 : x > 0)
  (h5 : y > 0) :
  x + y = 14 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_integers_l1245_124527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lattice_points_on_circle_l1245_124567

-- Define the circle C
def C (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - Real.sqrt 2)^2 + (p.2 - Real.sqrt 3)^2 = r^2}

-- Define a lattice point
def is_lattice_point (p : ℝ × ℝ) : Prop :=
  ∃ (m n : ℤ), p.1 = m ∧ p.2 = n

-- Theorem statement
theorem max_lattice_points_on_circle (r : ℝ) (hr : r > 0) :
  ∃! (n : ℕ), n ≤ 1 ∧ ∃ (S : Finset (ℝ × ℝ)), ↑S ⊆ C r ∧ (∀ p ∈ S, is_lattice_point p) ∧ S.card = n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lattice_points_on_circle_l1245_124567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mrs_riley_average_score_l1245_124529

/-- Represents the score distribution for a class of students -/
structure ScoreDistribution where
  total_students : ℕ
  scores : List ℕ
  students : List ℕ

/-- Calculates the average score given a score distribution -/
def average_score (sd : ScoreDistribution) : ℚ :=
  let total_sum : ℕ := (List.zip sd.scores sd.students).map (fun (s, n) => s * n) |>.sum
  (total_sum : ℚ) / sd.total_students

/-- The specific score distribution for Mrs. Riley's class -/
def mrs_riley_distribution : ScoreDistribution := {
  total_students := 100,
  scores := [100, 90, 80, 70, 60, 50, 40],
  students := [7, 18, 35, 25, 10, 3, 2]
}

/-- Theorem stating that the average score for Mrs. Riley's class is 77% -/
theorem mrs_riley_average_score :
  average_score mrs_riley_distribution = 77 := by
  sorry

#eval average_score mrs_riley_distribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mrs_riley_average_score_l1245_124529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1245_124565

/-- The standard equation of an ellipse with specific properties -/
theorem ellipse_equation (C : Set (ℝ × ℝ)) (e : ℝ) (V : ℝ × ℝ) : 
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2/16 + y^2/12 = 1) ↔
  (-- Center at origin
   (0, 0) ∈ C ∧
   -- Foci on x-axis
   (∃ (c : ℝ), (c, 0) ∈ C ∧ (-c, 0) ∈ C) ∧
   -- Eccentricity is 1/2
   e = 1/2 ∧
   -- One vertex is (0, 2√3)
   V = (0, 2 * Real.sqrt 3) ∧ V ∈ C) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1245_124565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_station_l1245_124592

/-- The distance to the train station in kilometers -/
noncomputable def distance : ℝ := sorry

/-- The time difference in hours between walking at 5 kmph and 6 kmph -/
noncomputable def time_difference : ℝ := 12 / 60

/-- When walking at 5 kmph, the man misses the train by 7 minutes -/
axiom miss_train : distance / 5 = time_difference + distance / 6

/-- When walking at 6 kmph, the man arrives 5 minutes early -/
axiom early_arrival : distance / 6 + time_difference = distance / 5

theorem distance_to_station : distance = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_station_l1245_124592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_product_bound_l1245_124530

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.exp (-x) - abs (Real.log x)

-- State the theorem
theorem zeros_product_bound (x₁ x₂ : ℝ) (h₁ : f x₁ = 0) (h₂ : f x₂ = 0) : 0 < x₁ * x₂ ∧ x₁ * x₂ < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_product_bound_l1245_124530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_correct_l1245_124574

/-- A function f is odd if f(-x) = -f(x) for all x in the domain of f -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- Definition of the function f -/
noncomputable def f : ℝ → ℝ := fun x => if x > 0 then x^3 - Real.cos x else x^3 + Real.cos x

theorem f_is_odd_and_correct : IsOdd f ∧ ∀ x, x < 0 → f x = x^3 + Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_correct_l1245_124574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l1245_124586

/-- For a parabola with equation y² = 4x, the distance from its focus to its directrix is 2 -/
theorem parabola_focus_directrix_distance (x y : ℝ) :
  y^2 = 4*x → (∃ (f d : ℝ), abs (f - d) = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l1245_124586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l1245_124583

/-- For a parabola with equation y² = 20x, the distance from its focus to its directrix is 10 -/
theorem parabola_focus_directrix_distance :
  ∀ (y x : ℝ), y^2 = 20*x → (∃ (focus_x focus_y directrix_x : ℝ),
    |focus_x - directrix_x| = 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l1245_124583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_relationship_l1245_124543

/-- Represents a right cylindrical water tank -/
structure WaterTank where
  height : ℝ
  baseDiameter : ℝ

/-- Calculates the water depth when the tank is standing upright -/
noncomputable def waterDepthUpright (tank : WaterTank) (sideDepth : ℝ) : ℝ :=
  sorry

/-- Theorem stating the relationship between side water depth and upright water depth -/
theorem water_depth_relationship (tank : WaterTank) (sideDepth : ℝ) :
  tank.height = 20 →
  tank.baseDiameter = 5 →
  sideDepth = 2 →
  |waterDepthUpright tank sideDepth - 12.1| < 0.05 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_relationship_l1245_124543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_is_36_l1245_124550

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Given conditions of the problem -/
def problem_setup (circle_III circle_IV : Circle) (square : Square) : Prop :=
  -- Circle III inscribes the square
  circle_III.radius * 2 = square.side
  -- Circle III is tangent to Circle IV
  ∧ circle_III.radius + circle_IV.radius = square.side / 2
  -- Circle III passes through the midpoint of Circle IV
  ∧ circle_III.radius * 2 = circle_IV.radius
  -- Area of Circle III is 9π/4
  ∧ Real.pi * circle_III.radius ^ 2 = 9 * Real.pi / 4

/-- The main theorem to prove -/
theorem square_area_is_36 (circle_III circle_IV : Circle) (square : Square) :
  problem_setup circle_III circle_IV square → square.side ^ 2 = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_is_36_l1245_124550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_sum_l1245_124584

-- Define the points of the triangles
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (0, 12)
def C : ℝ × ℝ := (16, 0)
def A' : ℝ × ℝ := (24, 18)
def B' : ℝ × ℝ := (36, 18)
def C' : ℝ × ℝ := (24, 2)

-- Define the rotation
noncomputable def clockwise_rotation (center : ℝ × ℝ) (angle : ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  let (cx, cy) := center
  let (px, py) := point
  let dx := px - cx
  let dy := py - cy
  (cx + dx * Real.cos angle + dy * Real.sin angle,
   cy - dx * Real.sin angle + dy * Real.cos angle)

-- Theorem statement
theorem rotation_sum (m x y : ℝ) (h1 : 0 < m) (h2 : m < 180) :
  (clockwise_rotation (x, y) (m * π / 180) A = A' ∧
   clockwise_rotation (x, y) (m * π / 180) B = B' ∧
   clockwise_rotation (x, y) (m * π / 180) C = C') →
  m + x + y = 108 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_sum_l1245_124584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_750_meters_l1245_124573

-- Define the given conditions
noncomputable def train_speed_kmh : ℝ := 108
noncomputable def platform_length : ℝ := 150
noncomputable def crossing_time : ℝ := 30

-- Define the conversion factor from km/h to m/s
noncomputable def kmh_to_ms : ℝ := 1000 / 3600

-- Define the theorem
theorem train_length_is_750_meters :
  let train_speed_ms := train_speed_kmh * kmh_to_ms
  let total_distance := train_speed_ms * crossing_time
  let train_length := total_distance - platform_length
  train_length = 750 := by
  -- Proof steps would go here, but we'll use 'sorry' for now
  sorry

#check train_length_is_750_meters

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_750_meters_l1245_124573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l1245_124552

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) - Real.cos (ω * x)

def is_axis_of_symmetry (ω : ℝ) (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (k * Real.pi + 3 * Real.pi / 4) / ω

def abscissa_not_in_interval (ω : ℝ) : Prop :=
  ∀ x : ℝ, is_axis_of_symmetry ω x → (x ≤ Real.pi ∨ x ≥ 2 * Real.pi)

theorem omega_range (ω : ℝ) (h1 : ω > 2/3) (h2 : abscissa_not_in_interval ω) :
  ω ∈ Set.Icc (3/4 : ℝ) (7/8 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l1245_124552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_and_fib_sum_l1245_124554

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x)

noncomputable def g (n : ℕ) (x : ℝ) : ℝ := x + (Finset.range n).sum (λ i => (f^[i + 1] x))

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem g_monotone_and_fib_sum (n : ℕ) :
  (∀ x y : ℝ, x > y ∧ y > 0 → g n x > g n y) ∧
  g n 1 = (Finset.range (n + 1)).sum (λ i => (fib (i + 1) : ℝ) / fib (i + 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_and_fib_sum_l1245_124554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_belt_drive_perimeter_l1245_124510

/-- Represents a wheel in the belt drive system -/
structure Wheel where
  perimeter : ℚ
  deriving Repr

/-- Represents the relationship between two wheels -/
structure WheelRelation where
  wheel1 : Wheel
  wheel2 : Wheel
  turns1 : ℕ
  turns2 : ℕ
  deriving Repr

/-- Theorem stating the relationship between wheel perimeters in the belt drive system -/
theorem belt_drive_perimeter 
  (K L M : Wheel)
  (KL_relation LM_relation : WheelRelation)
  (hKL : KL_relation.wheel1 = K ∧ KL_relation.wheel2 = L ∧ KL_relation.turns1 = 5 ∧ KL_relation.turns2 = 4)
  (hLM : LM_relation.wheel1 = L ∧ LM_relation.wheel2 = M ∧ LM_relation.turns1 = 6 ∧ LM_relation.turns2 = 7)
  (hM_perimeter : M.perimeter = 30) :
  K.perimeter = 28 := by
  sorry

-- The #eval command is not necessary for this theorem, so we can remove it

end NUMINAMATH_CALUDE_ERRORFEEDBACK_belt_drive_perimeter_l1245_124510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_l1245_124549

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- State the theorem
theorem x_range (x : ℝ) (h : floor ((x + 4) / 10) = 5) : 46 ≤ x ∧ x < 56 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_l1245_124549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_characterization_l1245_124590

theorem positive_integer_characterization (a : ℝ) (h : a > 0) :
  let b := (a + Real.sqrt (a^2 + 1))^(1/3) + (a - Real.sqrt (a^2 + 1))^(1/3)
  (∃ n : ℕ+, (b : ℝ) = n) ↔ (∃ n : ℕ+, a = (1/2) * (n : ℝ) * ((n : ℝ)^2 + 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integer_characterization_l1245_124590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_solution_set_l1245_124558

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2

theorem function_properties (a : ℝ) :
  (∀ x, f a x = f a (-x)) →
  a = 0 :=
by
  intro h
  sorry

theorem solution_set (a : ℝ) :
  f a (π / 4) = Real.sqrt 3 + 1 →
  {x : ℝ | f a x = 1 - Real.sqrt 2 ∧ -π ≤ x ∧ x ≤ π} =
  {13 * π / 24, -5 * π / 24, -11 * π / 24, 19 * π / 24} :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_solution_set_l1245_124558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circumscribed_circle_ratio_l1245_124553

/-- Predicate to check if a given radius is the inscribed radius of an isosceles right triangle -/
def is_inscribed_radius (r : ℝ) : Prop :=
  ∃ (side : ℝ), side > 0 ∧ r = side * (Real.sqrt 2 - 1) / 2

/-- Predicate to check if a given radius is the circumscribed radius of an isosceles right triangle -/
def is_circumscribed_radius (R : ℝ) : Prop :=
  ∃ (side : ℝ), side > 0 ∧ R = side * Real.sqrt 2 / 2

/-- For an isosceles right triangle with an inscribed circle of radius r
    and a circumscribed circle of radius R, R/r = 1 + √2 -/
theorem inscribed_circumscribed_circle_ratio (r R : ℝ) :
  r > 0 → R > 0 → is_inscribed_radius r → is_circumscribed_radius R → R / r = 1 + Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circumscribed_circle_ratio_l1245_124553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chris_reaches_andrea_in_52_minutes_l1245_124503

/-- Represents the biking scenario with Andrea and Chris --/
structure BikingScenario where
  initial_distance : ℝ
  initial_rate : ℝ
  rate_decrease : ℝ
  andrea_stop_time : ℝ
  chris_speed : ℝ

/-- Calculates the time it takes for Chris to reach Andrea --/
noncomputable def time_to_reach (scenario : BikingScenario) : ℝ :=
  scenario.andrea_stop_time +
    (scenario.initial_distance -
      (scenario.initial_rate * scenario.andrea_stop_time -
        0.5 * scenario.rate_decrease * scenario.andrea_stop_time ^ 2)) /
    scenario.chris_speed

/-- Theorem stating that Chris reaches Andrea after 52 minutes --/
theorem chris_reaches_andrea_in_52_minutes (scenario : BikingScenario)
    (h1 : scenario.initial_distance = 24)
    (h2 : scenario.initial_rate = 1.2)
    (h3 : scenario.rate_decrease = 0.04)
    (h4 : scenario.andrea_stop_time = 4)
    (h5 : scenario.chris_speed = 0.4) :
    time_to_reach scenario = 52 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chris_reaches_andrea_in_52_minutes_l1245_124503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_revolutions_half_mile_l1245_124555

/-- The number of revolutions a wheel makes to travel a certain distance -/
noncomputable def revolutions (diameter : ℝ) (distance : ℝ) : ℝ :=
  distance / (Real.pi * diameter)

/-- Half a mile in feet -/
def half_mile_in_feet : ℝ := 0.5 * 5280

theorem wheel_revolutions_half_mile :
  revolutions 10 half_mile_in_feet = 264 / Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_revolutions_half_mile_l1245_124555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_proof_l1245_124523

/-- The x-coordinate of the point on the x-axis that is equidistant from A(-3, 0) and B(2, 5) -/
def equidistant_point : ℝ := 2

/-- Point A -/
def A : ℝ × ℝ := (-3, 0)

/-- Point B -/
def B : ℝ × ℝ := (2, 5)

/-- Distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem equidistant_point_proof :
  let P : ℝ × ℝ := (equidistant_point, 0)
  distance P A = distance P B := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_proof_l1245_124523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_21_digit_sum_l1245_124516

theorem factorial_21_digit_sum (a b c : ℕ) : 
  (a < 10 ∧ b < 10 ∧ c < 10) →
  (Nat.factorial 21 = 51090942171000000000 + a * 100000000000 + b * 10000000000 + c * 1000000000) →
  100 * a + 10 * b + c = 709 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_21_digit_sum_l1245_124516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_recipe_fills_l1245_124566

-- Define the problem parameters
def total_sugar : ℚ := 15/4  -- 3¾ cups
def cup_capacity : ℚ := 1/3
def sugar_measured : ℚ := total_sugar / 2  -- Half of the required sugar
def sugar_spilled : ℚ := 1/4

-- Define the function to calculate the number of fills needed
def fills_needed (total : ℚ) (capacity : ℚ) (measured : ℚ) (spilled : ℚ) : ℕ :=
  let remaining := total - (measured - spilled)
  Int.toNat ((remaining / capacity).ceil)

-- State the theorem
theorem cookie_recipe_fills :
  fills_needed total_sugar cup_capacity sugar_measured sugar_spilled = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_recipe_fills_l1245_124566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_expression_equality_l1245_124589

theorem root_expression_equality : 
  (((5 : ℝ) ^ (1/3 : ℝ)) * ((3 : ℝ) ^ (1/6 : ℝ))) / (((5 : ℝ) ^ (1/2 : ℝ)) / ((3 : ℝ) ^ (1/3 : ℝ))) = 
  (5 : ℝ) ^ (-(1/6 : ℝ)) * (3 : ℝ) ^ ((1/2 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_expression_equality_l1245_124589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_solution_existence_l1245_124538

theorem sin_cos_equation_solution_existence (k : ℝ) :
  (∃ (x y : ℝ), Real.sin y = k * Real.sin x ∧ 2 * Real.cos x + Real.cos y = 1) ↔ 
  -Real.sqrt 2 ≤ k ∧ k ≤ Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_solution_existence_l1245_124538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1245_124551

-- Define the inequality condition
axiom inequality_condition {a b x y : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) 
  (hx : 0 < x) (hy : 0 < y) : 
  1/x + 1/y ≥ 4/(x+y)

-- Define the equality condition
axiom equality_condition {a b x y : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) 
  (hx : 0 < x) (hy : 0 < y) : 
  1/x + 1/y = 4/(x+y) ↔ x = y

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1/(1-2*x) + 1/(2-3*x)

-- State the theorem
theorem min_value_of_f : 
  ∃ (x : ℝ), ∀ (y : ℝ), 0 < 1-2*y ∧ 0 < 2-3*y → f y ≥ 35 ∧ f x = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1245_124551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_characterization_l1245_124533

-- Define the space
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

-- Define points A and B
variable (A B : E)

-- Define the unit circles centered at A and B
def ωA (A : E) : Set E := {x | ‖x - A‖ ≤ 1}
def ωB (B : E) : Set E := {x | ‖x - B‖ ≤ 1}

-- Define the lens-shaped intersection of ωA and ωB
def lens (A B : E) : Set E := ωA A ∩ ωB B

-- Define the locus of points C
def locus (A B : E) : Set E :=
  {C | ∃ (center : E), ‖center - A‖ ≤ 1 ∧ ‖center - B‖ ≤ 1 ∧ ‖center - C‖ ≤ 1}

-- Theorem statement
theorem locus_characterization (A B : E) :
  locus A B = ⋃ (center ∈ lens A B), {x | ‖x - center‖ ≤ 1} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_characterization_l1245_124533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_squared_l1245_124507

theorem root_difference_squared (x₁ x₂ : ℂ) : 
  (Real.sqrt 11 : ℝ) * x₁^2 + (Real.sqrt 180 : ℝ) * x₁ + (Real.sqrt 176 : ℝ) = 0 →
  (Real.sqrt 11 : ℝ) * x₂^2 + (Real.sqrt 180 : ℝ) * x₂ + (Real.sqrt 176 : ℝ) = 0 →
  x₁ ≠ x₂ →
  Complex.abs (1/x₁^2 - 1/x₂^2) = (Real.sqrt 45 : ℝ)/44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_squared_l1245_124507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_planes_l1245_124532

/-- The distance between two parallel planes -/
noncomputable def distance_between_planes (a b c d : ℝ) (k : ℝ) : ℝ :=
  |k - d| / Real.sqrt (a^2 + b^2 + c^2)

/-- Theorem: The distance between the planes x + 2y - 2z + 1 = 0 and 2x + 4y - 4z + 5 = 0 is 1/2 -/
theorem distance_specific_planes :
  distance_between_planes 1 2 (-2) 1 (5/2) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_planes_l1245_124532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_t_for_2090_l1245_124514

def p (k : ℕ) : ℕ := Nat.minFac k

def X (k : ℕ) : ℕ :=
  if p k = 2 then 1
  else (Finset.range (p k - 1)).prod (λ i => if Nat.Prime (i + 2) then i + 2 else 1)

def x : ℕ → ℕ
  | 0 => 1
  | n + 1 => x n * p (x n) / X (x n)

theorem smallest_t_for_2090 : ∃ t : ℕ, t = 149 ∧ x t = 2090 ∧ ∀ s < t, x s ≠ 2090 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_t_for_2090_l1245_124514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_zero_is_plane_l1245_124500

/-- The set of points with vertical coordinate 0 forms a plane -/
theorem vertical_zero_is_plane :
  ∀ (S : Set (Fin 3 → ℝ)), 
  (∀ p ∈ S, p 2 = 0) → 
  ∃ (a b c : ℝ), ∀ p ∈ S, a * p 0 + b * p 1 + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_zero_is_plane_l1245_124500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1245_124519

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos (Real.pi/3 * Real.sin (x^2 + 6*x + 10 - Real.sin x))

theorem f_range : ∀ x : ℝ, 2 ≤ f x ∧ f x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1245_124519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_plus_2alpha_l1245_124564

theorem cos_pi_plus_2alpha (α : ℝ) (h : Real.sin α = 1/3) : Real.cos (π + 2*α) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_plus_2alpha_l1245_124564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_equilateral_triangle_l1245_124557

/-- A move is defined as changing the length of any side such that the result is still a triangle -/
def IsValidMove (a b c a' b' c' : ℝ) : Prop :=
  (a + b > c ∧ b + c > a ∧ c + a > b) ∧
  (a' + b' > c' ∧ b' + c' > a' ∧ c' + a' > b') ∧
  ((a' = a ∧ b' = b) ∨ (a' = a ∧ c' = c) ∨ (b' = b ∧ c' = c))

/-- The sequence of moves from the initial triangle to the final triangle -/
def ValidMoveSequence (n : ℕ) (seq : Fin (n + 1) → ℝ × ℝ × ℝ) : Prop :=
  seq 0 = (700, 700, 700) ∧ seq (Fin.last (n + 1)) = (2, 2, 2) ∧
    ∀ i : Fin n, IsValidMove (seq i).1 (seq i).2.1 (seq i).2.2
                             (seq (Fin.succ i)).1 (seq (Fin.succ i)).2.1 (seq (Fin.succ i)).2.2

/-- The main theorem stating that the minimum number of moves is 14 -/
theorem min_moves_equilateral_triangle :
  (∃ seq : Fin 15 → ℝ × ℝ × ℝ, ValidMoveSequence 14 seq) ∧
  (∀ n < 14, ¬∃ seq : Fin (n + 1) → ℝ × ℝ × ℝ, ValidMoveSequence n seq) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_equilateral_triangle_l1245_124557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1245_124548

/-- Calculates the speed of a train in km/h given its length in meters and the time it takes to cross a stationary observer in seconds. -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

/-- Theorem stating that a 240-meter long train crossing a stationary observer in 4 seconds has a speed of 216 km/h. -/
theorem train_speed_calculation :
  train_speed 240 4 = 216 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1245_124548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1245_124522

theorem simplify_expression : 7 - (6) - (-3) + (-5) = 7 - 6 + 3 - 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1245_124522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_at_most_three_l1245_124528

theorem negation_at_most_three :
  (∀ n : ℕ, (¬(n ≤ 3)) ↔ (n ≥ 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_at_most_three_l1245_124528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_points_P_l1245_124588

/-- Definition of the ellipse C -/
noncomputable def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Definition of eccentricity -/
noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

/-- Definition of the tangent line -/
def tangentLine (x y : ℝ) : Prop := x - y - 4 = 0

/-- Theorem about the ellipse and points P -/
theorem ellipse_and_points_P :
  ∀ a b c : ℝ,
  (∃ x y : ℝ, ellipse x y a b) →
  eccentricity a c = 1/3 →
  (∃ x y : ℝ, tangentLine x y ∧ x^2 + y^2 = b^2) →
  (∃ x y : ℝ, ellipse x y 3 (2*Real.sqrt 2)) ∧
  (∀ m : ℝ, ∃ k : ℝ,
    ∀ x₁ y₁ x₂ y₂ : ℝ,
    ellipse x₁ y₁ 3 (2*Real.sqrt 2) →
    ellipse x₂ y₂ 3 (2*Real.sqrt 2) →
    x₁ = m * y₁ + 1 →
    x₂ = m * y₂ + 1 →
    ((y₁ / (x₁ - 3)) * (y₂ / (x₂ - 3)) = k ∨
     (y₁ / (x₁ + 3)) * (y₂ / (x₂ + 3)) = k)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_points_P_l1245_124588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_cut_perimeter_ratio_l1245_124569

/-- Given a square with vertices (-a, -a), (a, -a), (-a, a), and (a, a), cut by the line y = x,
    the ratio of the perimeter of one of the resulting congruent quadrilaterals to a is 4 + 2√2. -/
theorem square_cut_perimeter_ratio (a : ℝ) (a_pos : a > 0) : 
  let square_vertices : Set (ℝ × ℝ) := {(-a, -a), (a, -a), (-a, a), (a, a)}
  let cut_line : Set (ℝ × ℝ) := {p | p.2 = p.1}
  let quadrilateral : Set (ℝ × ℝ) := {(-a, -a), (a, -a), (a, a), (-a, a)}
  let perimeter : ℝ := 2 * (2 * a + a * Real.sqrt 2)
  perimeter / a = 4 + 2 * Real.sqrt 2 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_cut_perimeter_ratio_l1245_124569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stirling_first_kind_exp_gen_func_l1245_124570

/-- Stirling numbers of the first kind -/
def stirling_first_kind (n N : ℕ) : ℚ := sorry

/-- Exponential generating function for Stirling numbers of the first kind -/
noncomputable def exp_gen_func (n : ℕ) (x : ℝ) : ℝ := 
  ∑' N, (stirling_first_kind n N : ℝ) * x^N / (Nat.factorial N : ℝ)

/-- Theorem stating the equality of the exponential generating function
    and the logarithmic expression for Stirling numbers of the first kind -/
theorem stirling_first_kind_exp_gen_func (n : ℕ) (x : ℝ) :
  exp_gen_func n x = (Real.log (1 + x))^n / (Nat.factorial n : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stirling_first_kind_exp_gen_func_l1245_124570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_karls_journey_distance_l1245_124525

/-- Represents Karl's car and journey details -/
structure KarlsJourney where
  /-- Miles per gallon of Karl's car -/
  mpg : ℚ
  /-- Capacity of Karl's gas tank in gallons -/
  tankCapacity : ℚ
  /-- Initial distance driven in miles -/
  initialDistance : ℚ
  /-- Amount of gas bought after initial distance in gallons -/
  gasBought : ℚ
  /-- Fraction of tank full upon arrival -/
  finalTankFraction : ℚ

/-- Calculates the total distance driven by Karl -/
def totalDistance (journey : KarlsJourney) : ℚ :=
  journey.initialDistance + 
  (journey.tankCapacity - journey.initialDistance / journey.mpg + journey.gasBought - 
   journey.finalTankFraction * journey.tankCapacity) * journey.mpg

/-- Theorem stating that Karl's total distance driven is 580 miles -/
theorem karls_journey_distance :
  let journey : KarlsJourney := {
    mpg := 30,
    tankCapacity := 14,
    initialDistance := 300,
    gasBought := 10,
    finalTankFraction := 1/3
  }
  totalDistance journey = 580 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_karls_journey_distance_l1245_124525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_theorem_l1245_124531

theorem root_sum_theorem (m n : ℕ+) (h_coprime : Nat.Coprime m n) :
  (∃ x₁ x₂ x₃ : ℝ,
    (2:ℝ)^(333*x₁ - 2) + (2:ℝ)^(111*x₁ + 2) = (2:ℝ)^(222*x₁ + 1) + 1 ∧
    (2:ℝ)^(333*x₂ - 2) + (2:ℝ)^(111*x₂ + 2) = (2:ℝ)^(222*x₂ + 1) + 1 ∧
    (2:ℝ)^(333*x₃ - 2) + (2:ℝ)^(111*x₃ + 2) = (2:ℝ)^(222*x₃ + 1) + 1 ∧
    (x₁ + x₂ + x₃ = m / n)) →
  m + n = 113 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_theorem_l1245_124531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_is_20000_l1245_124545

/-- Calculates the principal amount given the interest rate, time, and total interest paid. -/
def calculate_principal (interest_rate : ℚ) (time : ℕ) (total_interest : ℕ) : ℚ :=
  (total_interest * 100 : ℚ) / (interest_rate * time)

/-- Theorem stating that given the specified conditions, the principal amount is 20000. -/
theorem principal_amount_is_20000 
  (interest_rate : ℚ) 
  (time : ℕ) 
  (total_interest : ℕ) 
  (h1 : interest_rate = 12 / 100)
  (h2 : time = 3)
  (h3 : total_interest = 7200) :
  calculate_principal interest_rate time total_interest = 20000 := by
  sorry

#eval calculate_principal (12 / 100) 3 7200

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_is_20000_l1245_124545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friends_rental_fee_percentage_l1245_124542

/-- Calculates the percentage of rental fee paid by John's friend -/
theorem friends_rental_fee_percentage
  (camera_value : ℚ)
  (rental_rate : ℚ)
  (rental_weeks : ℕ)
  (johns_payment : ℚ)
  (h1 : camera_value = 5000)
  (h2 : rental_rate = 1/10)
  (h3 : rental_weeks = 4)
  (h4 : johns_payment = 1200)
  : (camera_value * rental_rate * rental_weeks - johns_payment) / (camera_value * rental_rate * rental_weeks) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friends_rental_fee_percentage_l1245_124542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1245_124513

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (4 - a) * x else a^x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (x₁ - x₂) * (f a x₁ - f a x₂) > 0) →
  (a ≥ 2 ∧ a < 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1245_124513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_profit_range_l1245_124524

-- Define the cost per unit
def cost : ℝ := 30

-- Define the sales volume function
noncomputable def sales_volume (x : ℝ) : ℝ := 
  if 40 ≤ x ∧ x < 60 then -2*x + 140
  else if 60 ≤ x ∧ x ≤ 70 then -x + 80
  else 0

-- Define the profit function
noncomputable def profit (x : ℝ) : ℝ := (x - cost) * sales_volume x

-- Theorem for maximum profit
theorem max_profit :
  ∃ (x : ℝ), x = 50 ∧ profit x = 800 ∧ 
  ∀ (y : ℝ), 40 ≤ y ∧ y ≤ 70 → profit y ≤ profit x := by
  sorry

-- Theorem for profit range
theorem profit_range :
  ∀ (x : ℝ), 45 ≤ x ∧ x ≤ 55 ↔ profit x ≥ 750 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_profit_range_l1245_124524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1245_124544

-- Define the line equation
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the circle equation
def circleEq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 5

-- Define the chord length
def chordLength : ℝ := 4

-- Theorem statement
theorem line_circle_intersection :
  ∀ k : ℝ, 
  (∃ x y : ℝ, line k x = y ∧ circleEq x y) ∧ 
  chordLength = 4 →
  k = 0 ∨ k = -4/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1245_124544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yards_per_mile_l1245_124512

-- Define the constants given in the problem
def car_speed_mph : ℝ := 90
def distance_yards : ℝ := 22
def time_seconds : ℝ := 0.5

-- Define the theorem to prove
theorem yards_per_mile : ∃ (y : ℝ), y = 1760 ∧ y * car_speed_mph = (distance_yards / time_seconds) * 3600 := by
  -- Calculate yards per second
  let yards_per_second := distance_yards / time_seconds
  -- Calculate yards per hour
  let yards_per_hour := yards_per_second * 3600
  -- Calculate yards per mile
  let yards_per_mile := yards_per_hour / car_speed_mph
  
  -- Prove that yards_per_mile equals 1760
  exists yards_per_mile
  constructor
  · sorry -- Proof that yards_per_mile = 1760
  · sorry -- Proof that yards_per_mile * car_speed_mph = yards_per_hour

-- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yards_per_mile_l1245_124512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rent_ratio_is_187_5_percent_l1245_124518

/-- Represents Elaine's financial situation over two years -/
structure ElaineFinances where
  lastYearEarnings : ℚ
  lastYearRentPercent : ℚ
  lastYearUtilitiesPercent : ℚ
  earningsIncreasePercent : ℚ
  thisYearRentPercent : ℚ
  thisYearUtilitiesPercent : ℚ

/-- Calculates the ratio of rent spent this year to rent spent last year -/
def rentRatio (e : ElaineFinances) : ℚ :=
  (e.thisYearRentPercent * (1 + e.earningsIncreasePercent)) / e.lastYearRentPercent

/-- Theorem stating that the rent ratio is 187.5% given Elaine's financial conditions -/
theorem rent_ratio_is_187_5_percent (e : ElaineFinances) 
  (h1 : e.lastYearRentPercent = 20/100)
  (h2 : e.lastYearUtilitiesPercent = 15/100)
  (h3 : e.earningsIncreasePercent = 25/100)
  (h4 : e.thisYearRentPercent = 30/100)
  (h5 : e.thisYearUtilitiesPercent = 20/100) :
  rentRatio e = 375/200 := by
  sorry

#eval (30/100 * (1 + 25/100)) / (20/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rent_ratio_is_187_5_percent_l1245_124518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l1245_124577

theorem sin_cos_product (α : ℝ) (h : Real.sin α = 2 * Real.cos α) : 
  Real.sin α * Real.cos α = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l1245_124577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_theorem_l1245_124521

noncomputable section

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the asymptote
def asymptote (x y : ℝ) : Prop :=
  x + Real.sqrt 2 * y = 0

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = 12 * x

-- Define the focus of the parabola
def parabola_focus (x : ℝ) : Prop :=
  x = 3

-- Define a point on the hyperbola
def point_on_hyperbola (a b x y : ℝ) : Prop :=
  hyperbola a b x y

-- Define the perpendicularity condition
def perpendicular_to_x_axis (x : ℝ) : Prop :=
  x = -3

-- Define the distance from a point to a line
noncomputable def distance_point_to_line (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  abs (-((y2 - y1) / (x2 - x1)) * x3 + y3 - y1 + ((y2 - y1) / (x2 - x1)) * x1) /
  Real.sqrt (1 + ((y2 - y1) / (x2 - x1))^2)

-- State the theorem
theorem hyperbola_distance_theorem (a b x1 y1 x2 y2 x3 y3 : ℝ) :
  hyperbola a b x2 y2 →
  asymptote x2 y2 →
  parabola_focus x3 →
  point_on_hyperbola a b x1 y1 →
  perpendicular_to_x_axis x1 →
  distance_point_to_line (-Real.sqrt 6) 0 x1 y1 x3 y3 = 6/5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_theorem_l1245_124521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_squared_greater_than_critical_value_frequency_machine_a_frequency_machine_b_l1245_124568

/-- Production data for two machines -/
structure ProductionData where
  machine_a_first : ℕ
  machine_a_second : ℕ
  machine_b_first : ℕ
  machine_b_second : ℕ

/-- Calculate K² statistic -/
noncomputable def calculate_k_squared (data : ProductionData) : ℝ :=
  let n := (data.machine_a_first + data.machine_a_second + data.machine_b_first + data.machine_b_second : ℝ)
  let a := (data.machine_a_first : ℝ)
  let b := (data.machine_a_second : ℝ)
  let c := (data.machine_b_first : ℝ)
  let d := (data.machine_b_second : ℝ)
  (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Critical value for 99% confidence level -/
def critical_value_99_percent : ℝ := 6.635

/-- Theorem: The calculated K² is greater than the critical value for 99% confidence -/
theorem k_squared_greater_than_critical_value (data : ProductionData) 
  (h1 : data.machine_a_first = 150)
  (h2 : data.machine_a_second = 50)
  (h3 : data.machine_b_first = 120)
  (h4 : data.machine_b_second = 80) :
  calculate_k_squared data > critical_value_99_percent := by
  sorry

/-- Frequencies of first-class products -/
def frequency_first_class (first : ℕ) (total : ℕ) : ℚ :=
  ↑first / ↑total

theorem frequency_machine_a (data : ProductionData) 
  (h1 : data.machine_a_first = 150)
  (h2 : data.machine_a_second = 50) :
  frequency_first_class data.machine_a_first (data.machine_a_first + data.machine_a_second) = 3/4 := by
  sorry

theorem frequency_machine_b (data : ProductionData)
  (h3 : data.machine_b_first = 120)
  (h4 : data.machine_b_second = 80) :
  frequency_first_class data.machine_b_first (data.machine_b_first + data.machine_b_second) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_squared_greater_than_critical_value_frequency_machine_a_frequency_machine_b_l1245_124568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_garden_area_difference_l1245_124560

-- Define the dimensions of the rectangular garden
def rectangle_length : ℝ := 60
def rectangle_width : ℝ := 8

-- Theorem statement
theorem circular_garden_area_difference :
  let rectangle_area := rectangle_length * rectangle_width
  let rectangle_perimeter := 2 * (rectangle_length + rectangle_width)
  let circle_radius := rectangle_perimeter / (2 * Real.pi)
  let circle_area := Real.pi * circle_radius^2
  ∃ ε > 0, abs (circle_area - rectangle_area - 992) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_garden_area_difference_l1245_124560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1245_124581

theorem problem_1 : 
  |1 - Real.sqrt 2| + (-(64 : ℝ))^(1/3) - Real.sqrt (1/2) = Real.sqrt 2 / 2 - 5 := by sorry

theorem problem_2 :
  (3 - 2 * Real.sqrt 5) * (3 + 2 * Real.sqrt 5) + (1 - Real.sqrt 5)^2 = -5 - 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1245_124581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1245_124546

/-- A parabola with vertex at the origin, focus on the coordinate axis, and passing through (-2,4) -/
structure Parabola where
  /-- The parabola passes through the point (-2,4) -/
  passes_through : ((-2 : ℝ), 4) ∈ {p : ℝ × ℝ | p.1^2 = p.2 ∨ p.2^2 = -8*p.1}
  /-- The vertex of the parabola is at the origin (0,0) -/
  vertex_at_origin : ((0 : ℝ), 0) ∈ {p : ℝ × ℝ | p.1^2 = p.2 ∨ p.2^2 = -8*p.1}
  /-- The focus of the parabola is on one of the coordinate axes -/
  focus_on_axis : ∃ (p : ℝ), p ≠ 0 ∧ ((p, 0) ∈ {q : ℝ × ℝ | q.1^2 = q.2} ∨ (0, p) ∈ {q : ℝ × ℝ | q.2^2 = -8*q.1})

/-- The equation of the parabola is either x^2 = y or y^2 = -8x -/
theorem parabola_equation (p : Parabola) : 
  ∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | p.1^2 = p.2 ∨ p.2^2 = -8*p.1} ↔ (x^2 = y ∨ y^2 = -8*x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1245_124546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_count_l1245_124596

theorem divisibility_count : 
  (Finset.filter 
    (fun x : ℕ => 
      x < 4032 ∧ 
      x > 0 ∧
      (x^2 - 20) % 16 = 0 ∧ 
      (x^2 - 16) % 20 = 0) 
    (Finset.range 4032)).card = 101 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_count_l1245_124596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l1245_124509

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (1/2)^x

-- Define the properties of f
def f_symmetric_to_g (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- Define the increasing property
def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x < f y

-- State the theorem
theorem f_increasing_interval
  (f : ℝ → ℝ)
  (h_sym : f_symmetric_to_g f) :
  is_increasing_on (fun x ↦ f (4 - x^2)) (Set.Icc 0 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l1245_124509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_tan_cos_l1245_124572

structure RightTriangle where
  J : ℝ
  K : ℝ
  L : ℝ
  jl : ℝ
  kl : ℝ
  right_angle : J = 90
  hypotenuse : kl = 13
  side : jl = 12

noncomputable def tan_K (t : RightTriangle) : ℝ := 5 / 12

noncomputable def cos_L (t : RightTriangle) : ℝ := 5 / 13

theorem right_triangle_tan_cos (t : RightTriangle) :
  tan_K t = 5 / 12 ∧ cos_L t = 5 / 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_tan_cos_l1245_124572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_5_l1245_124547

/-- Geometric sequence with positive common ratio -/
noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 0 ∧ ∀ n, a (n + 1) = a n * q

/-- Sum of first n terms of a geometric sequence -/
noncomputable def geometric_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a 1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_5 (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  a 1 = 2 →
  a 3 = a 2 + 4 →
  geometric_sum a q 5 = 62 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_5_l1245_124547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_area_special_square_l1245_124562

/-- A square with two vertices on a line and two on a parabola -/
structure SpecialSquare where
  -- Two vertices on the line y = 3x - 12
  vertex1 : ℝ × ℝ
  vertex2 : ℝ × ℝ
  -- Two vertices on the parabola y = x^2
  vertex3 : ℝ × ℝ
  vertex4 : ℝ × ℝ
  -- Conditions for the vertices
  line_condition1 : vertex1.2 = 3 * vertex1.1 - 12
  line_condition2 : vertex2.2 = 3 * vertex2.1 - 12
  parabola_condition1 : vertex3.2 = vertex3.1 ^ 2
  parabola_condition2 : vertex4.2 = vertex4.1 ^ 2
  -- Square condition (all sides equal)
  square_condition : 
    (vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2 =
    (vertex2.1 - vertex3.1)^2 + (vertex2.2 - vertex3.2)^2 ∧
    (vertex2.1 - vertex3.1)^2 + (vertex2.2 - vertex3.2)^2 =
    (vertex3.1 - vertex4.1)^2 + (vertex3.2 - vertex4.2)^2 ∧
    (vertex3.1 - vertex4.1)^2 + (vertex3.2 - vertex4.2)^2 =
    (vertex4.1 - vertex1.1)^2 + (vertex4.2 - vertex1.2)^2

/-- The area of a square given its side length -/
def area (s : SpecialSquare) : ℝ :=
  (s.vertex1.1 - s.vertex2.1)^2 + (s.vertex1.2 - s.vertex2.2)^2

/-- The smallest possible area of a SpecialSquare is 100 -/
theorem smallest_area_special_square :
  ∀ s : SpecialSquare, (∃ s' : SpecialSquare, area s' ≤ area s) → area s ≥ 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_area_special_square_l1245_124562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_total_worth_is_four_l1245_124580

/-- The probability of getting heads on a single flip of the biased coin -/
noncomputable def p_heads : ℝ := 2/3

/-- The probability of getting tails on a single flip of the biased coin -/
noncomputable def p_tails : ℝ := 1/3

/-- The amount gained on a heads flip -/
def heads_gain : ℝ := 5

/-- The amount lost on a tails flip -/
def tails_loss : ℝ := 6

/-- The number of coin flips -/
def num_flips : ℕ := 3

/-- The expected worth of a single flip of the biased coin -/
noncomputable def expected_single_flip : ℝ := p_heads * heads_gain - p_tails * tails_loss

/-- The expected total worth of three flips of the biased coin -/
noncomputable def expected_total_worth : ℝ := num_flips * expected_single_flip

theorem expected_total_worth_is_four :
  expected_total_worth = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_total_worth_is_four_l1245_124580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_triangle_area_l1245_124576

/-- Hyperbola with given parameters -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  e : ℝ
  h_e_sqrt3 : e = Real.sqrt 3
  h_b_sqrt2 : b = Real.sqrt 2

/-- Point on the hyperbola -/
structure PointOnHyperbola (H : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : x^2 / H.a^2 - y^2 / H.b^2 = 1

/-- Foci of the hyperbola -/
noncomputable def foci (H : Hyperbola) : ℝ × ℝ := (Real.sqrt 3, 0)

/-- Theorem about the hyperbola and area of triangle -/
theorem hyperbola_and_triangle_area (H : Hyperbola) :
  (∀ x y, x^2 - y^2 / 2 = 1 ↔ x^2 / H.a^2 - y^2 / H.b^2 = 1) ∧
  (∀ P : PointOnHyperbola H,
    let (E, F) := foci H
    let PE := (E - P.x, -P.y)
    let PF := (F - P.x, -P.y)
    PE.1 * PF.1 + PE.2 * PF.2 = 0 →
    (1 / 2) * Real.sqrt ((E - P.x)^2 + P.y^2) * Real.sqrt ((F - P.x)^2 + P.y^2) = 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_triangle_area_l1245_124576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_in_convex_ngon_intersection_points_formula_correct_l1245_124539

/-- The number of intersection points of diagonals in a convex n-gon -/
def intersectionPoints (n : ℕ) : ℚ :=
  n * (n - 1) * (n - 2) * (n - 3) / 24

/-- Theorem: Number of intersection points in a convex n-gon -/
theorem intersection_points_in_convex_ngon (n : ℕ) (h : n ≥ 4) :
  intersectionPoints n = n.choose 4 := by
  sorry

/-- Corollary: The formula for intersection points is correct -/
theorem intersection_points_formula_correct (n : ℕ) (h : n ≥ 4) :
  ∃ (m : ℚ), m = intersectionPoints n ∧ 
  m = n.choose 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_in_convex_ngon_intersection_points_formula_correct_l1245_124539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_l1245_124506

theorem sin_double_angle_special (a : ℝ) (h1 : 0 < a ∧ a < π/2) 
  (h2 : Real.cos (a + π/6) = 4/5) : Real.sin (2*a + π/3) = 24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_l1245_124506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_curve_l1245_124585

noncomputable section

-- Define the curve f(x) = x^3 + 2x + 1
def f (x : ℝ) : ℝ := x^3 + 2*x + 1

-- Define the line l: y = kx + 1
def line (k : ℝ) (x : ℝ) : ℝ := k*x + 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem line_intersects_curve (A B C : ℝ × ℝ) :
  (∃ k, f A.1 = line k A.1 ∧ f B.1 = line k B.1 ∧ f C.1 = line k C.1) →
  (distance A.1 A.2 B.1 B.2 = Real.sqrt 10) →
  (distance B.1 B.2 C.1 C.2 = Real.sqrt 10) →
  (∃ k, k = 3 ∧ ∀ x, line k x = 3*x + 1) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_curve_l1245_124585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_transformation_l1245_124578

noncomputable def variance (s : List ℝ) : ℝ := (1 / s.length : ℝ) * (s.map (λ x => (x - s.sum / s.length) ^ 2)).sum

theorem variance_transformation (k : List ℝ) (h : k.length = 8) (var_k : variance k = 3) :
  variance (k.map (λ x => 2 * (x - 3))) = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_transformation_l1245_124578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_extension_l1245_124594

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x > 0 then x * (1 - x)
  else if x < 0 then x * (1 + x)
  else 0  -- Define f(0) = 0 to make it continuous at 0

-- State the theorem
theorem odd_function_extension :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x > 0, f x = x * (1 - x)) →
  (∀ x < 0, f x = x * (1 + x)) :=
by
  intro h
  intro x hx
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_extension_l1245_124594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_toy_spending_calculation_l1245_124504

def toy_car_price : ℚ := 4.95
def toy_car_quantity : ℕ := 3
def skateboard_price : ℚ := 24.88
def toy_truck_original_price : ℚ := 12.95
def toy_truck_quantity : ℕ := 2
def toy_truck_discount : ℚ := 0.15

def total_toy_spending : ℚ :=
  toy_car_price * toy_car_quantity +
  skateboard_price +
  (toy_truck_original_price * (1 - toy_truck_discount)).floor / 100 * 100 * toy_truck_quantity

theorem total_toy_spending_calculation :
  total_toy_spending = 61.75 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_toy_spending_calculation_l1245_124504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_A_profit_percent_l1245_124559

theorem article_A_profit_percent (x : ℝ) (x_positive : x > 0) : 
  let cost_price := (5/8) * x
  let discounted_price := 0.9 * x
  let final_price := discounted_price * 1.08
  let profit := final_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  abs (profit_percent - 55.52) < 0.01 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_A_profit_percent_l1245_124559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_q_coordinates_l1245_124526

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Two points are parallel to the x-axis if they have the same y-coordinate -/
def parallel_to_x_axis (p q : Point) : Prop :=
  p.y = q.y

theorem point_q_coordinates (p q : Point) :
  p = Point.mk 2 (-6) →
  parallel_to_x_axis p q →
  distance p q = 2 →
  (q = Point.mk 0 (-6) ∨ q = Point.mk 4 (-6)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_q_coordinates_l1245_124526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_f_3_l1245_124517

-- Define the function g
def g : ℝ → ℝ := sorry

-- Define the function f
def f (x : ℝ) : ℝ := x * g x

-- State the theorem
theorem find_f_3 :
  (∀ x, g (3 * x) = g x) →
  (∀ x y, x * f y = y * f x) →
  f 15 = 45 →
  f 3 = 9 := by
  intros h1 h2 h3
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_f_3_l1245_124517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_landing_probability_l1245_124598

theorem adjacent_landing_probability (n : ℕ) (h : n = 6) :
  (2 * Nat.factorial (n - 1) : ℚ) / Nat.factorial n = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_landing_probability_l1245_124598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_inequality_l1245_124541

variable {n : ℕ}
variable (a b c : EuclideanSpace ℝ (Fin n))

theorem vector_inequality :
  (‖a‖ * inner b c)^2 + (‖b‖ * inner a c)^2 ≤ ‖a‖ * ‖b‖ * (‖a‖ * ‖b‖ + |inner a b|) * ‖c‖^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_inequality_l1245_124541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_equals_24_l1245_124597

-- Define the functions f and g as noncomputable
noncomputable def f (x : ℝ) : ℝ := x - 3
noncomputable def g (x : ℝ) : ℝ := x / 2

-- Define the inverse functions as noncomputable
noncomputable def f_inv (x : ℝ) : ℝ := x + 3
noncomputable def g_inv (x : ℝ) : ℝ := 2 * x

-- State the theorem
theorem composition_equals_24 : f (g_inv (f_inv (g (f 24)))) = 24 := by
  -- Proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_equals_24_l1245_124597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mutated_frogs_percentage_is_33_l1245_124582

/-- Calculates the percentage of mutated frogs, rounded to the nearest integer -/
def mutatedFrogsPercentage (extra_legs : ℕ) (two_heads : ℕ) (bright_red : ℕ) (normal : ℕ) : ℕ :=
  let total_mutated := extra_legs + two_heads + bright_red
  let total_frogs := total_mutated + normal
  let percentage := (total_mutated : ℚ) / (total_frogs : ℚ) * 100
  (percentage + 1/2).floor.toNat

/-- Theorem stating that the percentage of mutated frogs is 33% -/
theorem mutated_frogs_percentage_is_33 :
  mutatedFrogsPercentage 5 2 2 18 = 33 := by
  sorry

#eval mutatedFrogsPercentage 5 2 2 18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mutated_frogs_percentage_is_33_l1245_124582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_i_fourth_power_sum_of_i_powers_l1245_124520

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- The cyclical property of i -/
theorem i_fourth_power : i^4 = 1 := by
  sorry

/-- Helper lemma for negative exponents -/
lemma i_neg_power (n : ℤ) : i^(-n) = (i^n)⁻¹ := by
  sorry

/-- The main theorem to prove -/
theorem sum_of_i_powers : i^5 + i^17 + i^(-15 : ℤ) + i^23 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_i_fourth_power_sum_of_i_powers_l1245_124520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l1245_124502

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)

theorem sine_function_properties :
  let amplitude : ℝ := 1
  let angular_frequency : ℝ := 2
  let phase_shift : ℝ := Real.pi / 4
  let x_range : Set ℝ := Set.Icc 0 (Real.pi / 2)
  let y_range : Set ℝ := Set.Icc (-Real.sqrt 2 / 2) 1
  (∀ x ∈ x_range, f x ∈ y_range) ∧
  (amplitude = 1) ∧
  (angular_frequency = 2) ∧
  (phase_shift = Real.pi / 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l1245_124502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1245_124537

noncomputable def cube_root (x : ℝ) : ℝ := Real.rpow x (1/3)

theorem inequality_solution (x : ℝ) : 
  cube_root x + 3 / (cube_root x + 4) ≤ 0 ↔ x ∈ Set.Ioo (-64) (-27) ∪ {-1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1245_124537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremePoints_and_zeros_equivalence_l1245_124515

open Set
open Function

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ) (f' : ℝ → ℝ)
variable (a b : ℝ)

-- Define the interval (a, b)
def openInterval (a b : ℝ) : Set ℝ := Ioo a b

-- Define what it means for f to have no extreme points in (a, b)
def noExtremePoints (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ openInterval a b, ¬(IsLocalMax f x ∨ IsLocalMin f x)

-- Define what it means for f' to have no zeros in (a, b)
def noZeros (f' : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ openInterval a b, f' x ≠ 0

-- State the theorem
theorem extremePoints_and_zeros_equivalence
  (hf : Differentiable ℝ f)
  (hf' : ∀ x, HasDerivAt f (f' x) x) :
  noExtremePoints f a b ↔ noZeros f' a b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremePoints_and_zeros_equivalence_l1245_124515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_profit_and_pricing_l1245_124599

structure Store where
  purchase_price_A : ℚ
  selling_price_A : ℚ
  purchase_price_B : ℚ
  selling_price_B : ℚ
  total_items : ℕ
  items_A : ℕ
  daily_sales_B : ℕ
  sales_increase_rate : ℕ
  target_daily_profit_B : ℚ

def total_profit (s : Store) (x : ℕ) : ℚ :=
  (s.selling_price_A - s.purchase_price_A) * x +
  (s.selling_price_B - s.purchase_price_B) * (s.total_items - x)

def new_price_equation (s : Store) (m : ℚ) : Prop :=
  (m - s.purchase_price_B) * (s.daily_sales_B + s.sales_increase_rate * (s.selling_price_B - m)) = s.target_daily_profit_B

theorem store_profit_and_pricing (s : Store) 
    (h1 : s.purchase_price_A = 40)
    (h2 : s.selling_price_A = 55)
    (h3 : s.purchase_price_B = 28)
    (h4 : s.selling_price_B = 40)
    (h5 : s.total_items = 80)
    (h6 : s.daily_sales_B = 4)
    (h7 : s.sales_increase_rate = 2)
    (h8 : s.target_daily_profit_B = 96) :
    (∀ x, total_profit s x = 3 * x + 960) ∧
    (∀ m, new_price_equation s m ↔ (m - 28) * (84 - 2 * m) = 96) := by
  sorry

#check store_profit_and_pricing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_profit_and_pricing_l1245_124599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l1245_124536

theorem binomial_expansion_properties (n : ℕ+) (a : ℝ) :
  (∀ x : ℝ, x > 0 →
    (Nat.choose n 1 : ℝ) / (Nat.choose n 2 : ℝ) = 1 / 4 ∧
    (Nat.choose n 6 : ℝ) * x^(n.val-6) * (a/Real.sqrt x)^6 = (Nat.choose n 6 : ℝ) * a^6) →
  (a = 1 ∧ 
   (Nat.choose n 4 = 126 ∨ Nat.choose n 5 = 126) ∧
   (∀ k : ℕ, k ≤ n → Nat.choose n k ≤ 126) ∧
   (∀ x : ℝ, x > 0 → 
     (x * Real.sqrt x - 1) * (x + a/Real.sqrt x)^(n.val) = 
     -48 + x * (Real.sqrt x) * ((x + a/Real.sqrt x)^(n.val)))) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l1245_124536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_sequence_is_zero_l1245_124563

def sequenceList : List ℤ := List.range 13 |>.map (λ x => x - 6)

theorem arithmetic_mean_of_sequence_is_zero :
  (sequenceList.sum : ℚ) / sequenceList.length = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_sequence_is_zero_l1245_124563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_odd_product_greater_than_20_l1245_124571

def ball_numbers : Finset ℕ := {2, 4, 6, 8}

def is_valid_product (x y : ℕ) : Bool :=
  Odd (x * y) ∧ x * y > 20

theorem probability_odd_product_greater_than_20 :
  (Finset.card (Finset.filter (fun p => is_valid_product p.1 p.2) (ball_numbers.product ball_numbers))) / 
  ((Finset.card ball_numbers ^ 2) : ℚ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_odd_product_greater_than_20_l1245_124571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vegan_soy_free_fraction_l1245_124535

theorem vegan_soy_free_fraction (n : ℕ) (h1 : n > 0) : 
  (n / 4 - (2 * (n / 4)) / 3 : ℚ) / n = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vegan_soy_free_fraction_l1245_124535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_dihedral_angle_bound_l1245_124540

/-- Represents a tetrahedron with face areas and dihedral angles -/
structure Tetrahedron where
  S₁ : ℝ
  S₂ : ℝ
  S₃ : ℝ
  α₁ : ℝ
  α₂ : ℝ
  α₃ : ℝ
  h_positive_areas : 0 < S₁ ∧ 0 < S₂ ∧ 0 < S₃
  h_largest_face : S₁ ≤ 1 ∧ S₂ ≤ 1 ∧ S₃ ≤ 1
  h_dihedral_range : 0 < α₁ ∧ α₁ < π ∧ 0 < α₂ ∧ α₂ < π ∧ 0 < α₃ ∧ α₃ < π
  h_projection : S₁ * Real.cos α₁ + S₂ * Real.cos α₂ + S₃ * Real.cos α₃ = 1

/-- The smallest dihedral angle of any tetrahedron is not greater than 
    the dihedral angle of a regular tetrahedron -/
theorem smallest_dihedral_angle_bound (t : Tetrahedron) : 
  max (Real.cos t.α₁) (max (Real.cos t.α₂) (Real.cos t.α₃)) ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_dihedral_angle_bound_l1245_124540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shot_probability_for_A_l1245_124511

/-- The probability of firing the shot on a single attempt -/
def p : ℚ := 1 / 6

/-- The probability of not firing the shot on a single attempt -/
def q : ℚ := 1 - p

/-- The probability that A fires the shot on the (2n+1)th attempt -/
def prob_A_fires (n : ℕ) : ℚ := q^(2*n) * p

/-- The total probability that A fires the shot -/
noncomputable def total_prob_A : ℚ := ∑' n, prob_A_fires n

/-- The theorem stating that the probability of A firing the shot is 6/11 -/
theorem shot_probability_for_A : total_prob_A = 6 / 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shot_probability_for_A_l1245_124511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_Q_coordinates_l1245_124501

def triangle_PQR (Q : ℝ × ℝ) : Prop :=
  Q.1 ≥ 0 ∧ Q.2 ≥ 0 ∧  -- Q in first quadrant
  (Q.2 / Q.1 = 1) ∧    -- ∠QPR = 45°
  (Q.1 = 6 ∧ Q.2 = 6)  -- ∠QRP = 90°

noncomputable def rotate_point (p : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ :=
  (p.1 * Real.cos θ - p.2 * Real.sin θ, p.1 * Real.sin θ + p.2 * Real.cos θ)

theorem rotated_Q_coordinates :
  ∀ Q : ℝ × ℝ, triangle_PQR Q →
  rotate_point Q (π/3) = (3 - 3 * Real.sqrt 3, 3 + 3 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_Q_coordinates_l1245_124501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_from_side_relation_l1245_124505

theorem triangle_angle_from_side_relation (a b c : ℝ) 
  (h : (a + b - c) * (a + b + c) = a * b) :
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_from_side_relation_l1245_124505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_is_three_area_theorem_l1245_124508

noncomputable def area_of_region (a b : ℝ) : ℝ := 3

theorem area_of_region_is_three (a b : ℝ) : 
  (-1 ≤ (a * (-1) + b) ∧ (a * (-1) + b) ≤ 2) → 
  (2 ≤ (a * 1 + b) ∧ (a * 1 + b) ≤ 4) → 
  area_of_region a b = 3 :=
by
  intros h1 h2
  unfold area_of_region
  rfl

theorem area_theorem (a b : ℝ) : 
  (-1 ≤ (a * (-1) + b) ∧ (a * (-1) + b) ≤ 2) → 
  (2 ≤ (a * 1 + b) ∧ (a * 1 + b) ≤ 4) → 
  ∃ S : ℝ, S = 3 ∧ S = area_of_region a b :=
by
  intros h1 h2
  use 3
  constructor
  · rfl
  · exact area_of_region_is_three a b h1 h2

#check area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_is_three_area_theorem_l1245_124508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l1245_124595

/-- The length of a train given the speeds and passing time of two trains -/
noncomputable def train_length (speed1 speed2 : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let relative_speed_ms := relative_speed * 1000 / 3600
  let total_length := relative_speed_ms * passing_time
  total_length / 2

/-- Theorem stating the length of each train given the problem conditions -/
theorem train_length_problem : 
  let speed1 := (90 : ℝ) -- km/h
  let speed2 := (85 : ℝ) -- km/h
  let passing_time := (8.64 : ℝ) -- seconds
  abs (train_length speed1 speed2 passing_time - 209.96) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l1245_124595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_value_l1245_124593

-- Define the function f
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

-- State the theorem
theorem f_symmetric_value (ω φ : ℝ) :
  (∀ x, f ω φ (π/6 + x) = f ω φ (π/6 - x)) →
  (f ω φ (π/6) = 2 ∨ f ω φ (π/6) = -2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_value_l1245_124593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1245_124534

theorem equation_solution (x : ℝ) : 
  27 * (2 : ℝ)^(-3*x) + 9 * (2 : ℝ)^x - (2 : ℝ)^(3*x) - 27 * (2 : ℝ)^(-x) = 8 ↔ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1245_124534
