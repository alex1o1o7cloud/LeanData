import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_time_difference_l1110_111059

/-- Represents a route with its characteristics -/
structure Route where
  total_distance : ℚ
  normal_distance : ℚ
  reduced_distance : ℚ
  normal_speed : ℚ
  reduced_speed : ℚ

/-- Calculates the time taken for a route in minutes -/
def time_taken (r : Route) : ℚ :=
  (r.normal_distance / r.normal_speed + r.reduced_distance / r.reduced_speed) * 60

/-- Route A characteristics -/
def route_a : Route :=
  { total_distance := 10
  , normal_distance := 8
  , reduced_distance := 2
  , normal_speed := 30
  , reduced_speed := 15 }

/-- Route B characteristics -/
def route_b : Route :=
  { total_distance := 8
  , normal_distance := 7
  , reduced_distance := 1
  , normal_speed := 40
  , reduced_speed := 10 }

/-- Theorem stating the time difference between routes A and B -/
theorem route_time_difference : time_taken route_a - time_taken route_b = 15/2 := by
  -- Expand the definitions and perform the calculation
  simp [time_taken, route_a, route_b]
  -- The proof is completed by computation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_time_difference_l1110_111059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_at_negative_one_l1110_111083

/-- Definition of the function f for positive x -/
noncomputable def f_pos (x : ℝ) : ℝ := x^2 + 1/x

/-- Main theorem: For an odd function f where f(x) = x^2 + 1/x for x > 0, f(-1) = -2 -/
theorem odd_function_value_at_negative_one
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_pos : ∀ x > 0, f x = f_pos x) :
  f (-1) = -2 := by
  have h1 : f 1 = f_pos 1 := h_pos 1 (by norm_num)
  have h2 : f_pos 1 = 2 := by
    unfold f_pos
    norm_num
  have h3 : f 1 = 2 := h1.trans h2
  have h4 : f (-1) = -f 1 := h_odd 1
  rw [h3] at h4
  exact h4


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_at_negative_one_l1110_111083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l1110_111087

theorem diophantine_equation_solutions :
  ∀ a b c : ℕ,
    a > 0 → b > 0 → c > 0 →
    (2 : ℕ)^a * (3 : ℕ)^b + 9 = c^2 ↔
    ((a, b, c) = (4, 0, 5) ∨
     (a, b, c) = (4, 5, 51) ∨
     (a, b, c) = (3, 3, 15) ∨
     (a, b, c) = (4, 3, 21) ∨
     (a, b, c) = (3, 2, 9)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l1110_111087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1110_111003

theorem problem_statement (a b : ℝ) 
  (h : ({0, b, b/a} : Set ℝ) = {1, a, a+b}) : a + 2*b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1110_111003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_on_circle_l1110_111053

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on a circle
def PointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Define a parallelogram
def IsParallelogram (a b c d : ℝ × ℝ) : Prop :=
  (b.1 - a.1 = d.1 - c.1) ∧ (b.2 - a.2 = d.2 - c.2) ∧
  (c.1 - a.1 = d.1 - b.1) ∧ (c.2 - a.2 = d.2 - b.2)

-- Theorem statement
theorem parallelogram_on_circle (Γ : Circle) (A C : ℝ × ℝ) :
  ∃ (B D : ℝ × ℝ), PointOnCircle Γ B ∧ PointOnCircle Γ D ∧ IsParallelogram A B C D := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_on_circle_l1110_111053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_speed_theorem_l1110_111084

noncomputable def average_speed_round_trip (upstream_speed downstream_speed : ℝ) : ℝ :=
  2 / (1 / upstream_speed + 1 / downstream_speed)

theorem round_trip_speed_theorem :
  average_speed_round_trip 4 7 = 56 / 11 := by
  unfold average_speed_round_trip
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_speed_theorem_l1110_111084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_representation_l1110_111016

theorem cosine_sum_product_representation :
  ∃ (a b c d : ℕ+),
    (∀ x : ℝ, Real.cos (2 * x) + Real.cos (4 * x) + Real.cos (8 * x) + Real.cos (14 * x) = 
      (a : ℝ) * Real.cos (b * x) * Real.cos (c * x) * Real.cos (d * x)) ∧
    a + b + c + d = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_representation_l1110_111016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_is_99_l1110_111063

def a : ℕ → ℕ
  | 0 => 3  -- Added case for 0
  | 1 => 3
  | n + 1 => 3^(a n)

def b : ℕ → ℕ
  | 0 => 100  -- Added case for 0
  | 1 => 100
  | n + 1 => 100^(b n)

def is_smallest_m (m : ℕ) : Prop :=
  m > 0 ∧ b m > a 100 ∧ ∀ k, 0 < k ∧ k < m → b k ≤ a 100

theorem smallest_m_is_99 : is_smallest_m 99 := by
  sorry

#eval a 2  -- This line is added to test the function
#eval b 2  -- This line is added to test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_is_99_l1110_111063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_of_100_factorial_l1110_111002

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

/-- Theorem about the last two nonzero digits of 100! -/
theorem last_two_nonzero_digits_of_100_factorial :
  ∃ k : ℕ, factorial 100 = k * 10^24 * 100 ∧ k % 100 ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_of_100_factorial_l1110_111002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_l1110_111033

-- Define the sign function
noncomputable def sign (a : ℝ) : ℝ :=
  if a > 0 then 1
  else if a < 0 then -1
  else 0

-- Define the system of equations
def satisfies_equations (x y z : ℝ) : Prop :=
  x = 2018 - 2019 * sign (y + z) ∧
  y = 2018 - 2019 * sign (x + z) ∧
  z = 2018 - 2019 * sign (x + y)

-- Theorem statement
theorem exactly_three_solutions :
  ∃! (s : Finset (ℝ × ℝ × ℝ)), 
    (∀ (x y z : ℝ), (x, y, z) ∈ s ↔ satisfies_equations x y z) ∧
    Finset.card s = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_l1110_111033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1110_111031

noncomputable def ellipse_foci : (ℝ × ℝ) × (ℝ × ℝ) := ((1, 3), (1, 9))
def passing_point : ℝ × ℝ := (10, 1)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def is_on_ellipse (p : ℝ × ℝ) (f1 f2 : ℝ × ℝ) (a b h k : ℝ) : Prop :=
  (p.1 - h)^2 / a^2 + (p.2 - k)^2 / b^2 = 1 ∧
  distance p f1 + distance p f2 = 2 * a

theorem ellipse_equation (a b h k : ℝ) :
  a > 0 ∧ b > 0 ∧
  is_on_ellipse passing_point ellipse_foci.1 ellipse_foci.2 a b h k ∧
  a = 8 * Real.sqrt 2 ∧ b = 11 ∧ h = 1 ∧ k = 6 := by
  sorry

#check ellipse_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1110_111031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_is_110_l1110_111035

/-- Represents a machine with its production quantity and rate -/
structure Machine where
  quantity : ℕ
  rate : ℕ
  mk_machine : rate > 0

/-- Calculates the total time for shirt production and malfunction repairs -/
def totalTime (machineA machineB machineC : Machine) (numMalfunctions timeMalfunction : ℕ) : ℕ :=
  let productionTimes := [machineA.quantity / machineA.rate, machineB.quantity / machineB.rate, machineC.quantity / machineC.rate]
  let maxProductionTime := productionTimes.maximum?
  match maxProductionTime with
  | some time => time + numMalfunctions * timeMalfunction
  | none => 0

/-- Theorem stating that the total time is 110 minutes -/
theorem total_time_is_110 (machineA machineB machineC : Machine) 
  (h1 : machineA.quantity = 360 ∧ machineA.rate = 4)
  (h2 : machineB.quantity = 480 ∧ machineB.rate = 5)
  (h3 : machineC.quantity = 300 ∧ machineC.rate = 3)
  (h4 : numMalfunctions = 2)
  (h5 : timeMalfunction = 5) :
  totalTime machineA machineB machineC numMalfunctions timeMalfunction = 110 := by
  sorry

#eval totalTime ⟨360, 4, by norm_num⟩ ⟨480, 5, by norm_num⟩ ⟨300, 3, by norm_num⟩ 2 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_is_110_l1110_111035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1110_111075

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x + 1)) / x

theorem domain_of_f :
  {x : ℝ | x ∈ Set.Ici (-1) ∧ x ≠ 0} = {x : ℝ | f x ∈ Set.univ} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1110_111075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l1110_111023

theorem power_equality (x : ℝ) : 10^(3*x) = (1000 : ℝ) → 10^(-x) = (1/10 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l1110_111023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_in_xz_plane_l1110_111060

/-- The point in the xz-plane equidistant from three given points -/
theorem equidistant_point_in_xz_plane :
  let p : ℝ × ℝ × ℝ := (31/10, 0, 1/5)
  let p1 : ℝ × ℝ × ℝ := (1, -1, 0)
  let p2 : ℝ × ℝ × ℝ := (2, 1, 2)
  let p3 : ℝ × ℝ × ℝ := (3, 2, -1)
  let dist (a b : ℝ × ℝ × ℝ) := Real.sqrt ((a.1 - b.1)^2 + (a.2.1 - b.2.1)^2 + (a.2.2 - b.2.2)^2)
  dist p p1 = dist p p2 ∧ dist p p2 = dist p p3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_in_xz_plane_l1110_111060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_T_l1110_111071

-- Define the solid T
def T : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p;
                    (|x| + |y| ≤ 2) ∧ (|x| + |z| ≤ 1) ∧ (|z| + |y| ≤ 1)}

-- Define the volume of a set in ℝ³
noncomputable def volume (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem volume_of_T : volume T = 4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_T_l1110_111071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l1110_111021

-- Define the sign function
noncomputable def sgn (x : ℝ) : ℝ :=
  if x > 0 then 1
  else if x = 0 then 0
  else -1

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := sgn (Real.log x) - Real.log x

-- Theorem statement
theorem f_has_three_zeros :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  f a = 0 ∧ f b = 0 ∧ f c = 0 ∧
  ∀ x > 0, f x = 0 → x = a ∨ x = b ∨ x = c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l1110_111021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_ratio_l1110_111046

/-- Given a line y = 1 - x and an ellipse ax^2 + by^2 = 1 intersecting at points A and B,
    if the slope of the line passing through the origin and the midpoint of AB is √3/2,
    then a/b = √3/2 -/
theorem ellipse_line_intersection_ratio (a b : ℝ) (A B : ℝ × ℝ) :
  (∃ x y, y = 1 - x ∧ a * x^2 + b * y^2 = 1) →  -- Line and ellipse intersection condition
  (A.1^2 + A.2^2 = 1 ∧ B.1^2 + B.2^2 = 1) →     -- A and B are on the ellipse
  (A.2 = 1 - A.1 ∧ B.2 = 1 - B.1) →             -- A and B are on the line
  (let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2);
   M.2 / M.1 = Real.sqrt 3 / 2) →               -- Slope of OM is √3/2
  a / b = Real.sqrt 3 / 2 :=                    -- Conclusion: a/b = √3/2
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_ratio_l1110_111046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_duration_proof_l1110_111055

/-- Represents the duration of a single class in hours -/
def class_duration : ℚ := 2

/-- Represents the initial number of classes -/
def initial_classes : ℕ := 4

/-- Represents the number of classes after dropping one -/
def remaining_classes : ℕ := initial_classes - 1

/-- Represents the total class time after dropping one class, in hours -/
def total_class_time : ℚ := 6

theorem class_duration_proof :
  (class_duration : ℚ) * (remaining_classes : ℚ) = total_class_time ∧
  class_duration > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_duration_proof_l1110_111055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_plane_property_l1110_111036

-- Define the basic structures
structure Line where

structure Plane where

-- Define the parallel relationship
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

def parallel_line_line (l1 l2 : Line) : Prop := sorry

def line_in_plane (l : Line) (p : Plane) : Prop := sorry

-- Define the concept of "countless" lines
def countless_parallel_lines (l : Line) (p : Plane) : Prop :=
  ∃ (S : Set Line), (∀ l' ∈ S, line_in_plane l' p ∧ parallel_line_line l l') ∧ Set.Infinite S

-- State the theorem
theorem parallel_line_plane_property (l : Line) (a : Plane) :
  (parallel_line_plane l a → countless_parallel_lines l a) ∧
  ∃ l a, countless_parallel_lines l a ∧ ¬parallel_line_plane l a := by
  sorry

#check parallel_line_plane_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_plane_property_l1110_111036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l1110_111018

/-- Yield function W(x) -/
noncomputable def W (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then 5 * (x^2 + 3)
  else if 2 < x ∧ x ≤ 5 then 50 * x / (1 + x)
  else 0

/-- Profit function f(x) -/
noncomputable def f (x : ℝ) : ℝ :=
  15 * W x - 30 * x

/-- The domain of x -/
def valid_x (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 5

theorem max_profit :
  ∃ (x : ℝ), valid_x x ∧ f x = 480 ∧ ∀ (y : ℝ), valid_x y → f y ≤ f x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l1110_111018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spiral_stripe_length_is_10_sqrt_10_l1110_111049

/-- Represents a cylindrical container with a spiral stripe -/
structure SpiralCylinder where
  height : ℝ
  baseCircumference : ℝ
  windingCount : ℕ

/-- Calculates the length of the spiral stripe on the cylinder -/
noncomputable def spiralStripeLength (c : SpiralCylinder) : ℝ :=
  Real.sqrt ((c.windingCount * c.baseCircumference) ^ 2 + c.height ^ 2)

/-- Theorem stating the length of the spiral stripe for the given conditions -/
theorem spiral_stripe_length_is_10_sqrt_10 :
  let c := SpiralCylinder.mk 10 15 2
  spiralStripeLength c = 10 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spiral_stripe_length_is_10_sqrt_10_l1110_111049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_theorem_l1110_111022

/-- Regular quadrilateral pyramid with base side length and slant height angle -/
structure RegularQuadPyramid where
  a : ℝ  -- base side length
  α : ℝ  -- slant height angle

/-- Represents the intersecting plane and inscribed circle -/
structure IntersectionPlane (p : RegularQuadPyramid) where
  -- Assumption that the plane is parallel to a base diagonal and a slant edge
  parallel_to_diagonal : Prop
  parallel_to_slant : Prop
  -- Assumption that a circle can be inscribed in the resulting section
  has_inscribed_circle : Prop

/-- The radius of the inscribed circle in the intersection plane -/
noncomputable def inscribed_circle_radius (p : RegularQuadPyramid) (plane : IntersectionPlane p) : ℝ :=
  (p.a * Real.sqrt 2) / (1 + 2 * Real.cos p.α + Real.sqrt (4 * (Real.cos p.α)^2 + 1))

theorem inscribed_circle_radius_theorem (p : RegularQuadPyramid) (plane : IntersectionPlane p) :
  inscribed_circle_radius p plane = (p.a * Real.sqrt 2) / (1 + 2 * Real.cos p.α + Real.sqrt (4 * (Real.cos p.α)^2 + 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_theorem_l1110_111022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_exists_l1110_111077

/-- Represents a cell in the grid -/
structure Cell where
  x : Fin 5
  y : Fin 7

/-- Represents a shape on the grid -/
structure Shape where
  cells : Set Cell

/-- The original 5x7 grid -/
def fullGrid : Shape := { cells := Set.univ }

/-- The three removed cells -/
def removedCells : Set Cell := sorry

/-- The remaining shape after removing cells -/
def remainingShape : Shape := { cells := fullGrid.cells \ removedCells }

/-- A part of the division -/
structure DivisionPart where
  shape : Shape
  cellCount : Nat
  isValid : cellCount = 4

/-- Predicate for whether two shapes can overlay -/
def CanOverlay (s1 s2 : Shape) : Prop := sorry

/-- A division of the remaining shape -/
structure Division where
  parts : Fin 8 → DivisionPart
  coversAll : (∀ c ∈ remainingShape.cells, ∃ i, c ∈ (parts i).shape.cells)
  noOverlap : ∀ i j, i ≠ j → Disjoint (parts i).shape.cells (parts j).shape.cells
  allEqual : ∀ i j, CanOverlay (parts i).shape (parts j).shape

theorem division_exists : ∃ d : Division, True := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_exists_l1110_111077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_foci_on_x_axis_foci_distance_l1110_111091

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop := x^2 + 2 * y^2 = 4

/-- The coordinates of a focus of the ellipse -/
noncomputable def focus_coordinate : ℝ × ℝ := (Real.sqrt 2, 0)

/-- Theorem stating that the given coordinates are the foci of the ellipse -/
theorem ellipse_foci :
  let (a, b) := focus_coordinate
  (ellipse_equation a b ∧
   ∀ x y, ellipse_equation x y → (x - a)^2 + y^2 + (x + a)^2 + y^2 = 4 * Real.sqrt ((x^2 + y^2) * (4 - x^2 - 2*y^2))) :=
by sorry

/-- Theorem proving that the foci are on the x-axis -/
theorem foci_on_x_axis :
  let (a, b) := focus_coordinate
  b = 0 :=
by sorry

/-- Theorem proving the distance between the foci -/
theorem foci_distance :
  let (a, b) := focus_coordinate
  2 * a = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_foci_on_x_axis_foci_distance_l1110_111091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_V_l1110_111041

/-- The set V in R³ -/
def V : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p
                   0 ≤ x ∧ x ≤ 1 ∧
                   ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 →
                     0 ≤ x * t^2 + y * t + z ∧
                     x * t^2 + y * t + z ≤ 1}

/-- The volume of set V -/
theorem volume_of_V : MeasureTheory.volume V = 17/18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_V_l1110_111041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_theorem_l1110_111050

-- Define the speed conversion factor from km/h to m/s
noncomputable def km_per_hour_to_m_per_sec : ℚ := 5 / 18

-- Define the given conditions
def train_speed_km_per_hour : ℚ := 144
def time_to_cross_post_seconds : ℚ := 16

-- Define the theorem
theorem train_length_theorem :
  let train_speed_m_per_sec := train_speed_km_per_hour * km_per_hour_to_m_per_sec
  let train_length_meters := train_speed_m_per_sec * time_to_cross_post_seconds
  train_length_meters = 640 := by
  -- Unfold the definitions
  unfold train_speed_km_per_hour
  unfold time_to_cross_post_seconds
  unfold km_per_hour_to_m_per_sec
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_theorem_l1110_111050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_selection_l1110_111024

/-- The sequence of numbers as described in the problem -/
def problem_sequence (n : ℕ) : List ℝ :=
  List.map (fun k => 3 * 2^k) (List.range (2*n))

/-- Predicate to check if the ratio of two numbers is neither 2 nor 1/2 -/
def valid_ratio (a b : ℝ) : Prop :=
  a/b ≠ 2 ∧ a/b ≠ 1/2

/-- The main theorem: The maximum number of elements that can be selected
    from the sequence with valid ratios is n -/
theorem max_selection (n : ℕ) :
  ∃ (s : List ℝ), s.length = n ∧ 
    (∀ x, x ∈ s → x ∈ problem_sequence n) ∧
    (∀ a b, a ∈ s → b ∈ s → a ≠ b → valid_ratio a b) ∧
    (∀ (t : List ℝ), t.length > n →
      (∀ x, x ∈ t → x ∈ problem_sequence n) →
      ¬(∀ a b, a ∈ t → b ∈ t → a ≠ b → valid_ratio a b)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_selection_l1110_111024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l1110_111015

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

/-- Theorem: For a geometric sequence with common ratio 3 and sum of first 4 terms equal to 80, the first term is 2 -/
theorem geometric_sequence_first_term :
  ∃ (a₁ : ℝ), geometric_sum a₁ 3 4 = 80 ∧ a₁ = 2 := by
  use 2
  simp [geometric_sum]
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l1110_111015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kit_ice_cream_time_kit_ice_cream_problem_l1110_111034

/-- Given Kit's movement rate and remaining distance, calculate the time needed to reach the ice cream stand -/
theorem kit_ice_cream_time (rate : ℚ) (distance : ℚ) (time : ℚ) : 
  rate = 2 → distance = 300 → time = distance / rate → time = 150 := by sorry

/-- Convert yards to feet -/
def yards_to_feet (yards : ℚ) : ℚ := yards * 3

/-- Calculate movement rate given distance and time -/
def calculate_rate (distance : ℚ) (time : ℚ) : ℚ := distance / time

/-- Initial conditions -/
def initial_movement : ℚ := 90
def initial_time : ℚ := 45
def remaining_distance_yards : ℚ := 100

/-- Kit's problem statement -/
theorem kit_ice_cream_problem :
  let rate := calculate_rate initial_movement initial_time
  let remaining_distance := yards_to_feet remaining_distance_yards
  let time_to_ice_cream := remaining_distance / rate
  time_to_ice_cream = 150 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kit_ice_cream_time_kit_ice_cream_problem_l1110_111034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_sum_l1110_111009

/-- The ellipse with equation x²/4 + y²/3 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1}

/-- The vertical line x = 1 -/
def VerticalLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 1}

/-- The right focus of the ellipse -/
def RightFocus : ℝ × ℝ := (1, 0)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem ellipse_intersection_sum (M N : ℝ × ℝ) 
  (hM : M ∈ Ellipse ∩ VerticalLine)
  (hN : N ∈ Ellipse ∩ VerticalLine)
  (hMN : M ≠ N) :
  1 / distance M RightFocus + 1 / distance N RightFocus = 4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_sum_l1110_111009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l1110_111017

/-- The volume of the top section of a right square pyramid -/
noncomputable def topPyramidVolume (baseEdge slantEdge cutHeight : ℝ) : ℝ :=
  let fullHeight := Real.sqrt (slantEdge ^ 2 - (baseEdge ^ 2) / 2)
  let newHeight := fullHeight - cutHeight
  let newBaseEdge := baseEdge * (newHeight / fullHeight)
  (1 / 3) * newBaseEdge ^ 2 * newHeight

/-- Theorem: The volume of the top section of a specific right square pyramid -/
theorem specific_pyramid_volume :
  topPyramidVolume 12 15 4 = (1 / 3) * ((144 * (153 - 8 * Real.sqrt 153)) / 153) * (Real.sqrt 153 - 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l1110_111017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_theorem_l1110_111011

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 9

-- Define the function g
noncomputable def g (m : ℝ) : ℝ :=
  if m ≥ 2 then 2 * m^2 - 8 * m + 9
  else if 0 < m ∧ m < 2 then 1
  else 2 * m^2 + 1

-- State the theorem
theorem quadratic_function_theorem :
  (∀ x, f x ≥ 1) ∧  -- Minimum value of f(x) is 1
  f 0 = 9 ∧ f 4 = 9 ∧  -- f(0) = f(4) = 9
  (∀ m, g m = min (f m) (f (m + 2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_theorem_l1110_111011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ethanol_percentage_in_fuel_B_l1110_111005

/-- Given a fuel tank with the following properties:
  - Tank capacity is 218 gallons
  - Fuel A contains 12% ethanol by volume
  - Full tank contains 30 gallons of ethanol
  - 122 gallons of fuel A were added
  This theorem proves that the percentage of ethanol in fuel B by volume is 16%. -/
theorem ethanol_percentage_in_fuel_B (tank_capacity : ℝ) (ethanol_A_percentage : ℝ) 
  (total_ethanol : ℝ) (fuel_A_volume : ℝ) : ℝ :=
  by
  -- Define given values
  have h1 : tank_capacity = 218 := by sorry
  have h2 : ethanol_A_percentage = 0.12 := by sorry
  have h3 : total_ethanol = 30 := by sorry
  have h4 : fuel_A_volume = 122 := by sorry
  
  -- Define the volume of fuel B
  let fuel_B_volume := tank_capacity - fuel_A_volume
  
  -- Define the amount of ethanol in fuel A
  let ethanol_A := ethanol_A_percentage * fuel_A_volume
  
  -- Define the amount of ethanol in fuel B
  let ethanol_B := total_ethanol - ethanol_A
  
  -- Calculate the percentage of ethanol in fuel B
  let ethanol_B_percentage := (ethanol_B / fuel_B_volume) * 100
  
  -- Return the result
  exact ethanol_B_percentage

-- Example usage (commented out to avoid compilation issues)
-- #eval ethanol_percentage_in_fuel_B 218 0.12 30 122

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ethanol_percentage_in_fuel_B_l1110_111005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_not_hearing_favorite_song_l1110_111097

-- Define the number of songs
def num_songs : ℕ := 12

-- Define the duration of the shortest song in seconds
def shortest_song_duration : ℕ := 45

-- Define the duration increment between songs in seconds
def song_duration_increment : ℕ := 45

-- Define the duration of the favorite song in seconds
def favorite_song_duration : ℕ := 4 * 60 + 15

-- Define the total play time we're considering in seconds
def total_play_time : ℕ := 5 * 60

-- Function to calculate the duration of a song given its position
def song_duration (position : ℕ) : ℕ :=
  shortest_song_duration + (position - 1) * song_duration_increment

-- Theorem stating the probability of not hearing the entire favorite song
theorem probability_not_hearing_favorite_song :
  (num_songs - 1 : ℚ) / num_songs =
  (1 - (Finset.filter (λ i ↦ song_duration i + favorite_song_duration ≤ total_play_time)
     (Finset.range num_songs)).card / num_songs : ℚ) := by
  sorry

#check probability_not_hearing_favorite_song

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_not_hearing_favorite_song_l1110_111097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_filtrations_required_l1110_111037

theorem minimum_filtrations_required (initial_impurity : Real) 
  (market_requirement : Real) (filtration_reduction : Real) : Nat :=
  let n : Nat := 6
  have h1 : initial_impurity = 0.01 := by sorry
  have h2 : market_requirement = 0.001 := by sorry
  have h3 : filtration_reduction = 2/3 := by sorry
  have h4 : initial_impurity * (filtration_reduction ^ n) ≤ market_requirement := by sorry
  have h5 : ∀ m : Nat, m < n → initial_impurity * (filtration_reduction ^ m) > market_requirement := by sorry
  n

/- Proof skipped -/

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_filtrations_required_l1110_111037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1110_111020

/-- The function g(x) = (x+1)/(x^2 + 1) -/
noncomputable def g (x : ℝ) : ℝ := (x + 1) / (x^2 + 1)

/-- The range of g is exactly {1/2} -/
theorem range_of_g :
  Set.range g = {1/2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1110_111020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_minor_axis_length_l1110_111073

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- An ellipse with axes parallel to the coordinate axes -/
structure Ellipse where
  center : Point
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

/-- Check if a point lies on an ellipse -/
def point_on_ellipse (p : Point) (e : Ellipse) : Prop :=
  ((p.x - e.center.x)^2 / e.semi_major_axis^2) + ((p.y - e.center.y)^2 / e.semi_minor_axis^2) = 1

/-- The five given points -/
noncomputable def points : List Point := [
  ⟨1, 0⟩,
  ⟨1, 3⟩,
  ⟨4, 0⟩,
  ⟨4, 3⟩,
  ⟨6, 3/2⟩
]

/-- The theorem statement -/
theorem ellipse_minor_axis_length :
  ∃ (e : Ellipse),
    (∀ p ∈ points, point_on_ellipse p e) ∧
    e.semi_minor_axis * 2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_minor_axis_length_l1110_111073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_cube_root_l1110_111057

theorem closest_integer_to_cube_root :
  ∃ (n : ℤ), n = 9 ∧ ∀ (m : ℤ), |((6^3 + 8^3 : ℝ) ^ (1/3 : ℝ)) - n| ≤ |((6^3 + 8^3 : ℝ) ^ (1/3 : ℝ)) - m| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_cube_root_l1110_111057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_transformation_l1110_111032

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the domain of f(x+1)
def domain_f_x_plus_1 : Set ℝ := Set.Icc 0 1

-- Define the domain of f(2^x - 2)
noncomputable def domain_f_2_pow_x_minus_2 : Set ℝ := Set.Icc (Real.log 3 / Real.log 2) 2

-- Theorem statement
theorem domain_transformation :
  (∀ x ∈ domain_f_x_plus_1, f (x + 1) ∈ Set.Icc 1 2) →
  (∀ x ∈ domain_f_2_pow_x_minus_2, f (2^x - 2) ∈ Set.Icc 1 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_transformation_l1110_111032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1110_111069

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x - 1) / Real.sqrt (3 * x + 2) + x^0

-- Define the domain of f
def domain_f : Set ℝ := {x | x > -2/3 ∧ x ≠ 0}

-- Theorem stating that the domain of f is (-2/3, 0) ∪ (0, +∞)
theorem domain_of_f : {x : ℝ | ∃ y, f x = y} = domain_f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1110_111069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_reflection_product_l1110_111078

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a triangle is isosceles -/
def isIsosceles (t : Triangle) : Prop :=
  distance t.A t.B = distance t.A t.C

/-- Check if a point is on a line segment -/
def isOnSegment (p : Point) (a b : Point) : Prop :=
  distance a p + distance p b = distance a b

/-- Reflect a point over a line -/
noncomputable def reflect (p : Point) (l1 l2 : Point) : Point :=
  sorry

/-- Find the intersection of two lines -/
noncomputable def lineIntersection (l1a l1b l2a l2b : Point) : Point :=
  sorry

theorem isosceles_triangle_reflection_product (t : Triangle) (D E F : Point) :
  isIsosceles t →
  distance t.A t.B = Real.sqrt 5 →
  isOnSegment D t.B t.C →
  D ≠ Point.mk ((t.B.x + t.C.x) / 2) ((t.B.y + t.C.y) / 2) →
  E = reflect t.C t.A D →
  F = lineIntersection E t.B t.A D →
  distance t.A D * distance t.A F = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_reflection_product_l1110_111078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1110_111019

-- Define the function f(x) = lg(2-x)
noncomputable def f (x : ℝ) : ℝ := Real.log (2 - x) / Real.log 10

-- Theorem stating that the domain of f is (-∞, 2)
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = Set.Iio 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1110_111019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carton_height_from_max_boxes_l1110_111080

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of boxes that can fit in a carton -/
def maxBoxesInCarton (carton : BoxDimensions) (box : BoxDimensions) : ℕ :=
  (carton.length / box.length) * (carton.width / box.width) * (carton.height / box.height)

/-- The theorem stating the relationship between carton height and max boxes -/
theorem carton_height_from_max_boxes 
  (carton_length carton_width carton_height : ℕ)
  (box_length box_width box_height : ℕ)
  (max_boxes : ℕ)
  (h1 : carton_length = 42)
  (h2 : carton_width = 25)
  (h3 : box_length = 8)
  (h4 : box_width = 7)
  (h5 : box_height = 5)
  (h6 : max_boxes = 210)
  (h7 : maxBoxesInCarton 
    { length := carton_length, width := carton_width, height := carton_height } 
    { length := box_length, width := box_width, height := box_height } = max_boxes) :
  carton_height = 70 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carton_height_from_max_boxes_l1110_111080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_min_distance_in_unit_disk_l1110_111013

/-- The largest possible minimum distance between n points in a unit disk -/
noncomputable def D (n : ℕ) : ℝ :=
  if n = 2 then 2
  else if n = 3 then Real.sqrt 3
  else if n = 4 then Real.sqrt 2
  else if n = 5 then 2 * Real.sin (Real.pi / 5)
  else if n = 6 then 1
  else if n = 7 then 1
  else 0  -- undefined for other values of n

/-- Theorem stating the largest possible minimum distance for n points in a unit disk -/
theorem largest_min_distance_in_unit_disk (n : ℕ) (h : 2 ≤ n ∧ n ≤ 7) :
  ∃ (points : Fin n → ℝ × ℝ),
    (∀ i, (points i).1^2 + (points i).2^2 ≤ 1) ∧
    (∀ i j, i ≠ j → ((points i).1 - (points j).1)^2 + ((points i).2 - (points j).2)^2 ≥ (D n)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_min_distance_in_unit_disk_l1110_111013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1110_111012

-- Define the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the slope of the line
noncomputable def line_slope : ℝ := -Real.sqrt 3

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ x, p.1 = x ∧ p.2 = parabola x ∧ p.2 - (1/8) = line_slope * (p.1 - 0)}

-- Theorem statement
theorem chord_length :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1110_111012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_shape_perimeter_l1110_111068

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℝ
  side_length_positive : side_length > 0

/-- Calculates the perimeter of an equilateral triangle -/
noncomputable def perimeter (t : EquilateralTriangle) : ℝ := 3 * t.side_length

/-- Represents a compound shape formed by two equilateral triangles sharing a vertex -/
structure CompoundShape where
  triangle1 : EquilateralTriangle
  triangle2 : EquilateralTriangle

/-- Calculates the total perimeter of the compound shape -/
noncomputable def total_perimeter (shape : CompoundShape) : ℝ :=
  perimeter shape.triangle1 + perimeter shape.triangle2 - min shape.triangle1.side_length shape.triangle2.side_length

/-- The theorem stating that the total perimeter of the specific compound shape is 42 -/
theorem compound_shape_perimeter :
  ∃ (shape : CompoundShape),
    shape.triangle1.side_length = 10 ∧
    shape.triangle2.side_length = 6 ∧
    total_perimeter shape = 42 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_shape_perimeter_l1110_111068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_coordinates_l1110_111061

/-- The problem setup -/
def Problem (x y z a : ℝ) : Prop :=
  x + y - 2 ≤ 0 ∧
  x - 2*y - 2 ≤ 0 ∧
  2*x - y + 2 ≥ 0 ∧
  z = y - a*x ∧
  a > 0 ∧
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ 
    (x₁ + y₁ - 2 ≤ 0 ∧ x₁ - 2*y₁ - 2 ≤ 0 ∧ 2*x₁ - y₁ + 2 ≥ 0 ∧ z = y₁ - a*x₁) ∧ 
    (x₂ + y₂ - 2 ≤ 0 ∧ x₂ - 2*y₂ - 2 ≤ 0 ∧ 2*x₂ - y₂ + 2 ≥ 0 ∧ z = y₂ - a*x₂) ∧
    ∀ (x' y' : ℝ), (x' + y' - 2 ≤ 0 ∧ x' - 2*y' - 2 ≤ 0 ∧ 2*x' - y' + 2 ≥ 0) → y' - a*x' ≤ z

/-- The theorem to be proved -/
theorem focus_coordinates (x y z a : ℝ) (h : Problem x y z a) : 
  ∃ (f : ℝ × ℝ), f = (0, 1/8) ∧ f.2 = 1/(4*a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_coordinates_l1110_111061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_horizontal_line_slope_angle_y_plus_3_eq_0_l1110_111045

/-- Slope angle of a line --/
noncomputable def SlopeAngle (f : ℝ → ℝ) : ℝ := sorry

/-- The slope angle of a line parallel to the x-axis is 0° --/
theorem slope_angle_horizontal_line (y : ℝ → ℝ) (c : ℝ) :
  (∀ x, y x + c = 0) → SlopeAngle y = 0 := by sorry

/-- The slope angle of the line y + 3 = 0 is 0° --/
theorem slope_angle_y_plus_3_eq_0 :
  SlopeAngle (λ x => -3) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_horizontal_line_slope_angle_y_plus_3_eq_0_l1110_111045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1110_111072

-- Define the curve C in polar coordinates
def curve_C (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

-- Define the line l in polar coordinates
def line_l (ρ θ : ℝ) : Prop := Real.sqrt 2 * ρ * Real.sin (θ + Real.pi/4) = 1

-- Define point Q
def point_Q (ρ : ℝ) : Prop := line_l ρ (Real.pi/2)

-- Theorem statement
theorem intersection_distance_sum (ρ_Q : ℝ) : 
  point_Q ρ_Q →
  ∃ (ρ_A θ_A ρ_B θ_B : ℝ),
    curve_C ρ_A θ_A ∧ line_l ρ_A θ_A ∧
    curve_C ρ_B θ_B ∧ line_l ρ_B θ_B ∧
    (ρ_A, θ_A) ≠ (ρ_B, θ_B) →
    Real.sqrt ((ρ_A * Real.cos θ_A)^2 + (ρ_A * Real.sin θ_A - ρ_Q)^2) +
    Real.sqrt ((ρ_B * Real.cos θ_B)^2 + (ρ_B * Real.sin θ_B - ρ_Q)^2) = 3 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1110_111072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_tangent_l1110_111028

theorem triangle_angle_tangent (A : ℝ) :
  (Real.sqrt 3 * Real.cos A + Real.sin A) / (Real.sqrt 3 * Real.sin A - Real.cos A) = Real.tan (-7 * π / 12) →
  Real.tan A = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_tangent_l1110_111028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_64_x_approx_l1110_111081

noncomputable def log_16_x_minus_5 (x : ℝ) : ℝ := Real.log (x - 5) / Real.log 16

noncomputable def log_64_x (x : ℝ) : ℝ := Real.log x / Real.log 64

theorem log_64_x_approx (x : ℝ) (h : log_16_x_minus_5 x = 1/2) :
  abs (log_64_x x - 0.525) < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_64_x_approx_l1110_111081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monster_perimeter_l1110_111056

/-- The perimeter of a circular sector with radius 1 and central angle 240° -/
theorem monster_perimeter : 
  (1 : ℝ) * (240 * (π / 180)) + 2 * 1 = (4/3) * π + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monster_perimeter_l1110_111056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_mixture_theorem_l1110_111079

/-- Represents the alcohol content of a solution -/
structure AlcoholContent where
  value : ℝ
  nonneg : 0 ≤ value
  le_one : value ≤ 1

/-- Represents the volume of a solution in milliliters -/
structure Volume where
  value : ℝ
  nonneg : 0 ≤ value

/-- Calculates the amount of pure alcohol in a solution -/
def pureAlcohol (content : AlcoholContent) (volume : Volume) : ℝ :=
  content.value * volume.value

/-- Theorem: Adding 200 mL of 30% alcohol solution to 200 mL of 10% alcohol solution results in a 20% alcohol solution -/
theorem alcohol_mixture_theorem (x_content y_content result_content : AlcoholContent) 
                                (x_volume y_volume : Volume) :
  x_content.value = 0.1 →
  y_content.value = 0.3 →
  result_content.value = 0.2 →
  x_volume.value = 200 →
  y_volume.value = 200 →
  pureAlcohol x_content x_volume + pureAlcohol y_content y_volume = 
  pureAlcohol result_content ⟨x_volume.value + y_volume.value, by sorry⟩ := by
  sorry

#check alcohol_mixture_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_mixture_theorem_l1110_111079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_of_75_factorial_l1110_111030

theorem last_two_nonzero_digits_of_75_factorial (m : ℕ) : 
  m = 32 → ∃ k : ℕ, Nat.factorial 75 = 100 * k + m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_of_75_factorial_l1110_111030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_perimeter_calculation_l1110_111094

/-- Calculates the perimeter of a square garden with a surrounding path -/
noncomputable def garden_perimeter (garden_area : ℝ) (path_width : ℝ) : ℝ :=
  4 * (Real.sqrt garden_area + 2 * path_width)

/-- Theorem: The perimeter of a square garden with area 144 sq meters and a 1-meter wide path is 56 meters -/
theorem garden_perimeter_calculation :
  garden_perimeter 144 1 = 56 := by
  -- Unfold the definition of garden_perimeter
  unfold garden_perimeter
  -- Simplify the expression
  simp [Real.sqrt_eq_rpow]
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_perimeter_calculation_l1110_111094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_area_theorem_l1110_111004

/-- A cross made of congruent squares -/
structure Cross where
  /-- Number of congruent squares in the cross -/
  num_squares : ℕ
  /-- Perimeter of the cross -/
  perimeter : ℝ

/-- The area of a cross made of congruent squares -/
noncomputable def cross_area (c : Cross) : ℝ :=
  let side_length := c.perimeter / (c.num_squares * 2 + 2)
  c.num_squares * (side_length ^ 2)

/-- Theorem: The area of a cross made of 5 congruent squares with perimeter 72 is 180 -/
theorem cross_area_theorem :
  ∀ (c : Cross), c.num_squares = 5 → c.perimeter = 72 → cross_area c = 180 :=
by
  intro c h1 h2
  unfold cross_area
  simp [h1, h2]
  -- The actual proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_area_theorem_l1110_111004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_A_l1110_111066

variable (A B C : ℝ)

def f (A B : ℝ) (x : ℝ) : ℝ := A * x^2 - 3 * B^3
def g (B C : ℝ) (x : ℝ) : ℝ := B * x^2 + C

theorem solve_for_A (hB : B ≠ 0) (hC : C ≠ 0) (h : f A B (g B C 2) = 0) :
  A = 3 * B^3 / (4 * B + C)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_A_l1110_111066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_two_zeros_l1110_111092

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := Real.sin (Real.log x) + 1/2

-- Define the interval (0, e^π)
def interval : Set ℝ := {x | 0 < x ∧ x < Real.exp Real.pi}

-- Theorem statement
theorem g_has_two_zeros : 
  ∃ (x₁ x₂ : ℝ), x₁ ∈ interval ∧ x₂ ∈ interval ∧ x₁ ≠ x₂ ∧ 
  g x₁ = 0 ∧ g x₂ = 0 ∧ 
  ∀ (x : ℝ), x ∈ interval ∧ g x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

#check g_has_two_zeros

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_two_zeros_l1110_111092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_arrangement_l1110_111065

def is_valid_arrangement (n : ℕ) (arr : List ℕ) : Prop :=
  arr.length = n ∧
  (∀ i : Fin n,
    (∀ j : Fin 40, arr[i]! > arr[(i + j) % n]!) ∨
    (∀ j : Fin 30, arr[i]! < arr[(i + j) % n]!))

theorem smallest_valid_arrangement :
  ∃ (arr : List ℕ),
    is_valid_arrangement 70 arr ∧
    ∀ m < 70, ¬∃ (arr : List ℕ), is_valid_arrangement m arr := by
  sorry

#check smallest_valid_arrangement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_arrangement_l1110_111065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_range_in_interval_l1110_111099

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin x * Real.cos (x + Real.pi/6) + 1

-- Theorem for the smallest positive period
theorem smallest_positive_period :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi :=
sorry

-- Theorem for the range of f(x) when x is in [-7π/12, 0]
theorem range_in_interval :
  ∀ (y : ℝ), (∃ (x : ℝ), x ∈ Set.Icc (-7*Real.pi/12) 0 ∧ f x = y) ↔ y ∈ Set.Icc (-2) 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_range_in_interval_l1110_111099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_to_girls_ratio_l1110_111038

/-- Represents the ratio of two numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- The college student population -/
structure College where
  total_students : ℕ
  girls : ℕ

def boys (c : College) : ℕ := c.total_students - c.girls

def simplify_ratio (r : Ratio) : Ratio :=
  let gcd := Nat.gcd r.numerator r.denominator
  { numerator := r.numerator / gcd, denominator := r.denominator / gcd }

theorem boys_to_girls_ratio (c : College) 
  (h1 : c.total_students = 520) 
  (h2 : c.girls = 200) : 
  simplify_ratio { numerator := boys c, denominator := c.girls } = { numerator := 8, denominator := 5 } := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_to_girls_ratio_l1110_111038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_problem_l1110_111040

theorem divisibility_problem (a : ℤ) (h1 : 0 ≤ a) (h2 : a < 13) 
  (h3 : (51^2016 + a) % 13 = 0) : a = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_problem_l1110_111040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_hundred_degrees_terminal_side_l1110_111014

def same_terminal_side (θ : ℝ) : Set ℝ :=
  {α | ∃ k : ℤ, α = k * 360 + θ}

theorem negative_hundred_degrees_terminal_side :
  same_terminal_side (-100) = {α : ℝ | ∃ k : ℤ, α = k * 360 - 100} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_hundred_degrees_terminal_side_l1110_111014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_with_constraints_l1110_111043

def is_valid_digit (d : ℕ) (valid_digits : List ℕ) : Prop :=
  d ∈ valid_digits

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def digits (n : ℕ) : ℕ × ℕ × ℕ :=
  (n / 100, (n / 10) % 10, n % 10)

theorem max_difference_with_constraints :
  ∃ (minuend subtrahend : ℕ),
    is_three_digit_number minuend ∧
    is_three_digit_number subtrahend ∧
    (let (a, b, c) := digits minuend;
     let (d, e, f) := digits subtrahend;
     let (g, h, i) := digits (minuend - subtrahend);
     is_valid_digit a [3, 5, 9] ∧
     is_valid_digit b [2, 3, 7] ∧
     is_valid_digit c [3, 4, 8, 9] ∧
     is_valid_digit d [2, 3, 7] ∧
     is_valid_digit e [3, 5, 9] ∧
     is_valid_digit f [1, 4, 7] ∧
     is_valid_digit g [4, 5, 9] ∧
     h = 2 ∧
     is_valid_digit i [4, 5, 9] ∧
     minuend - subtrahend = 529 ∧
     ∀ (m s : ℕ),
       is_three_digit_number m →
       is_three_digit_number s →
       (let (am, bm, cm) := digits m;
        let (ds, es, fs) := digits s;
        let (gd, hd, id) := digits (m - s);
        is_valid_digit am [3, 5, 9] →
        is_valid_digit bm [2, 3, 7] →
        is_valid_digit cm [3, 4, 8, 9] →
        is_valid_digit ds [2, 3, 7] →
        is_valid_digit es [3, 5, 9] →
        is_valid_digit fs [1, 4, 7] →
        is_valid_digit gd [4, 5, 9] →
        hd = 2 →
        is_valid_digit id [4, 5, 9] →
        m - s ≤ minuend - subtrahend))
:= by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_with_constraints_l1110_111043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircles_tangent_l1110_111052

-- Define a convex quadrilateral
structure ConvexQuadrilateral (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
(A B C D : V)
(convex : Convex ℝ {A, B, C, D})

-- Define the property of the quadrilateral
def HasEqualSumOfOppositeSides {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (quad : ConvexQuadrilateral V) : Prop :=
  ‖quad.A - quad.B‖ + ‖quad.C - quad.D‖ = ‖quad.B - quad.C‖ + ‖quad.D - quad.A‖

-- Define an incircle
noncomputable def Incircle (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (A B C : V) : Set V :=
  { p : V | ∃ (r : ℝ), r > 0 ∧ 
    ‖p - A‖ = r ∧ ‖p - B‖ = r ∧ ‖p - C‖ = r ∧
    ∀ q : V, ‖q - A‖ ≤ r ∧ ‖q - B‖ ≤ r ∧ ‖q - C‖ ≤ r }

-- Define tangency of two circles
def AreTangent {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (c1 c2 : Set V) : Prop :=
  ∃ p : V, p ∈ c1 ∧ p ∈ c2 ∧ 
  ∀ q : V, q ∈ c1 ∧ q ∈ c2 → q = p

-- The theorem statement
theorem incircles_tangent 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (quad : ConvexQuadrilateral V)
  (h : HasEqualSumOfOppositeSides quad) :
  AreTangent 
    (Incircle V quad.A quad.B quad.C)
    (Incircle V quad.A quad.C quad.D) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircles_tangent_l1110_111052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_theorem_l1110_111025

/-- The angle between two 2D vectors -/
noncomputable def angle (v w : ℝ × ℝ) : ℝ := 
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

/-- Theorem: If vectors a = (√3, 1) and b = (m, 1) form an angle of 2π/3, then m = -√3 -/
theorem vector_angle_theorem (m : ℝ) : 
  let a : ℝ × ℝ := (Real.sqrt 3, 1)
  let b : ℝ × ℝ := (m, 1)
  angle a b = 2 * Real.pi / 3 → m = -Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_theorem_l1110_111025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_proper_subsets_l1110_111039

def A : Finset ℕ := {1, 3, 5, 7}
def B : Finset ℕ := {2, 3, 4, 5}

theorem intersection_proper_subsets :
  Finset.card (Finset.powerset (A ∩ B) \ {A ∩ B}) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_proper_subsets_l1110_111039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_honey_production_equations_l1110_111048

/-- Represents the amount of nectar from Type A flowers in kg -/
def x : ℝ := sorry

/-- Represents the amount of nectar from Type B flowers in kg -/
def y : ℝ := sorry

/-- The water content fraction in Type A nectar -/
def water_content_A : ℝ := 0.7

/-- The water content fraction in Type B nectar -/
def water_content_B : ℝ := 0.5

/-- The desired water content fraction in the final honey -/
def final_water_content : ℝ := 0.3

/-- The fraction of initial water content lost through evaporation -/
def evaporation_rate : ℝ := 0.15

/-- The weight of the final honey product in kg -/
def final_honey_weight : ℝ := 1

/-- 
Theorem stating that the system of equations representing the relationship
between x and y for producing 1 kg of honey with 30% water content, given
the conditions on nectar water content and evaporation, is:
x + y = 1 and 0.595x + 0.425y = 0.3
-/
theorem honey_production_equations :
  (x + y = final_honey_weight) ∧
  ((water_content_A * x + water_content_B * y) * (1 - evaporation_rate) = final_water_content * final_honey_weight) ↔
  (x + y = 1 ∧ 0.595 * x + 0.425 * y = 0.3) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_honey_production_equations_l1110_111048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_sum_l1110_111090

/-- The sum of areas of an infinite series of equilateral triangles -/
noncomputable def triangle_area_sum (A : ℝ) : ℝ :=
  let area_ratio := (1 : ℝ) / 4
  A / (1 - area_ratio)

/-- Theorem: The sum of areas of an infinite series of equilateral triangles -/
theorem equilateral_triangle_area_sum (A : ℝ) (h : A > 0) :
  triangle_area_sum A = 4 * A / 3 := by
  -- Unfold the definition of triangle_area_sum
  unfold triangle_area_sum
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry

#check equilateral_triangle_area_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_sum_l1110_111090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_of_specific_arithmetic_sequence_l1110_111074

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => arithmetic_sequence a₁ d n + d

theorem tenth_term_of_specific_arithmetic_sequence :
  let a₁ := 1
  let d := 3
  arithmetic_sequence a₁ d 9 = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_of_specific_arithmetic_sequence_l1110_111074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_convex_pentagon_l1110_111054

/-- The measure of the largest angle in a convex pentagon with specific angle measures -/
theorem largest_angle_convex_pentagon (x : ℝ) 
  (angle1 : ℝ := 2*x + 2)
  (angle2 : ℝ := 3*x - 3)
  (angle3 : ℝ := 4*x + 1)
  (angle4 : ℝ := 5*x)
  (angle5 : ℝ := 6*x - 5)
  (h_sum : angle1 + angle2 + angle3 + angle4 + angle5 = 540) :
  max angle1 (max angle2 (max angle3 (max angle4 angle5))) = 158.5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_convex_pentagon_l1110_111054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_relations_l1110_111089

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation
variable (perp : Plane → Plane → Prop)
variable (perpLL : Line → Line → Prop)
variable (perpLP : Line → Plane → Prop)

-- Notation for perpendicularity
local infix:50 " ⊥ " => perp
local infix:50 " ⊥L " => perpLL
local infix:50 " ⊥P " => perpLP

-- Define the planes and lines
variable (α β : Plane) (m n : Line)

-- Define the theorem
theorem perpendicular_relations :
  ((α ⊥ β) ∧ (m ⊥P β) ∧ (n ⊥P α) → (m ⊥L n)) ∨
  ((m ⊥L n) ∧ (m ⊥P β) ∧ (n ⊥P α) → (α ⊥ β)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_relations_l1110_111089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_cone_heights_l1110_111027

-- Define the frustum properties
noncomputable def frustum_altitude : ℝ := 30
noncomputable def lower_base_area : ℝ := 400 * Real.pi
noncomputable def upper_base_area : ℝ := 100 * Real.pi

-- Define the radii of the bases
noncomputable def lower_radius : ℝ := (lower_base_area / Real.pi) ^ (1/2)
noncomputable def upper_radius : ℝ := (upper_base_area / Real.pi) ^ (1/2)

-- Define the theorem
theorem frustum_cone_heights :
  let ratio : ℝ := upper_radius / lower_radius
  let total_height : ℝ := frustum_altitude / (1 - ratio)
  let small_cone_height : ℝ := ratio * total_height
  small_cone_height = 30 ∧ total_height = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_cone_heights_l1110_111027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_area_is_five_halves_l1110_111096

/-- The polynomial whose roots form the kite -/
noncomputable def p (z : ℂ) : ℂ := z^4 + 4*Complex.I*z^3 + (7 - 7*Complex.I)*z^2 + (10 + 2*Complex.I)*z - (1 + 12*Complex.I)

/-- The set of roots of the polynomial p -/
noncomputable def roots : Set ℂ := {z : ℂ | p z = 0}

/-- The kite formed by the roots of p -/
structure Kite where
  vertices : Set ℂ
  is_roots : vertices = roots
  is_kite : ∃ (a b c d : ℂ), vertices = {a, b, c, d} ∧ 
    ∃ (p q : ℝ), ‖a - (-Complex.I)‖ = ‖c - (-Complex.I)‖ ∧ 
                  ‖a - (-Complex.I)‖ = p ∧
                  ‖b - (-Complex.I)‖ = ‖d - (-Complex.I)‖ ∧
                  ‖b - (-Complex.I)‖ = q ∧
                  (a - c).re * (b - d).re + (a - c).im * (b - d).im = 0  -- perpendicular diagonals

/-- The area of the kite -/
noncomputable def kite_area (k : Kite) : ℝ := 5/2

/-- The main theorem: the area of the kite formed by the roots of p is 5/2 -/
theorem kite_area_is_five_halves (k : Kite) : kite_area k = 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_area_is_five_halves_l1110_111096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_sqrt_plus_x_l1110_111047

open MeasureTheory Interval Real

theorem definite_integral_sqrt_plus_x :
  ∫ x in (Set.Icc 0 1), (Real.sqrt x + x) = 7/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_sqrt_plus_x_l1110_111047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_phi_value_l1110_111085

/-- The original function f(x) -/
noncomputable def f (x φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ)

/-- The shifted function g(x) -/
noncomputable def g (x φ : ℝ) : ℝ := f (x + Real.pi/3) φ

/-- Theorem stating the maximum value of φ -/
theorem max_phi_value (φ : ℝ) 
  (h1 : φ < 0)
  (h2 : ∀ x, g x φ = g (-x) φ) -- g is an even function
  : φ ≤ -Real.pi/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_phi_value_l1110_111085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_heads_is_47_l1110_111042

/-- A fair coin that can be flipped up to four times -/
structure Coin where
  flips : Fin 4 → Bool

/-- The probability of getting heads on a single flip -/
noncomputable def p_heads : ℝ := 1 / 2

/-- The probability of getting heads within four flips -/
noncomputable def p_heads_within_four : ℝ :=
  p_heads + (1 - p_heads) * p_heads + (1 - p_heads)^2 * p_heads + (1 - p_heads)^3 * p_heads

/-- The number of coins -/
def num_coins : ℕ := 50

/-- The expected number of coins showing heads after up to four tosses -/
noncomputable def expected_heads : ℝ := num_coins * p_heads_within_four

theorem expected_heads_is_47 : 
  ⌊expected_heads⌋₊ = 47 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_heads_is_47_l1110_111042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bottles_needed_l1110_111088

/-- The number of fluid ounces in one liter -/
noncomputable def fl_oz_per_liter : ℚ := 33.8

/-- The volume of each milk bottle in milliliters -/
def bottle_volume : ℚ := 250

/-- The minimum amount of milk Christine needs to buy in fluid ounces -/
def min_milk_needed : ℚ := 60

/-- Converts fluid ounces to milliliters -/
noncomputable def fl_oz_to_ml (x : ℚ) : ℚ := x * 1000 / fl_oz_per_liter

/-- Calculates the number of bottles needed to contain a given volume of milk in milliliters -/
noncomputable def bottles_needed (ml : ℚ) : ℕ := (Int.ceil (ml / bottle_volume)).toNat

theorem min_bottles_needed : bottles_needed (fl_oz_to_ml min_milk_needed) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bottles_needed_l1110_111088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_fencing_cost_l1110_111076

/-- Represents a rectangular park with sides in ratio 3:2 and area 3750 sq m -/
structure Park where
  length : ℝ
  width : ℝ
  ratio : length = 3/2 * width
  area : length * width = 3750

/-- The cost of fencing in paise per meter -/
noncomputable def fencing_cost_paise : ℝ := 50

/-- Conversion rate from paise to rupees -/
noncomputable def paise_to_rupees : ℝ := 1/100

theorem park_fencing_cost (p : Park) :
  (2 * (p.length + p.width)) * (fencing_cost_paise * paise_to_rupees) = 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_fencing_cost_l1110_111076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l1110_111010

-- Define the circle equation
def circle_equation (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x^2 + 4*x + y^2 - 6*y - 40 = 0

-- Define the center of a circle
def is_center (h k : ℝ) (eq : ℝ × ℝ → Prop) : Prop :=
  ∀ p, eq p ↔ (p.1 - h)^2 + (p.2 - k)^2 = (p.1 + h)^2 + (p.2 + k)^2

-- Theorem statement
theorem circle_center :
  is_center (-2) 3 circle_equation :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l1110_111010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_f_upper_bound_achievable_l1110_111044

/-- The function f(x) defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := (2022 * x^2 * Real.log (x + 2022)) / ((Real.log (x + 2022))^3 + 2*x^3)

/-- Theorem stating that f(x) has an upper bound of 674 for positive real x -/
theorem f_upper_bound (x : ℝ) (hx : x > 0) : f x ≤ 674 := by sorry

/-- Theorem stating that the upper bound of 674 for f(x) is achievable -/
theorem f_upper_bound_achievable : ∃ x : ℝ, x > 0 ∧ f x = 674 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_f_upper_bound_achievable_l1110_111044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_figure_l1110_111001

/-- The area of the figure bounded by x = 3 cos t, y = 8 sin t, and y = 4 (where y ≥ 4) -/
noncomputable def boundedArea : ℝ := 8 * Real.pi - 6 * Real.sqrt 3

/-- The parametric equations of the boundary -/
noncomputable def boundaryEquations (t : ℝ) : ℝ × ℝ := (3 * Real.cos t, 8 * Real.sin t)

/-- The horizontal line that bounds the figure -/
def horizontalBoundary : ℝ := 4

theorem area_of_bounded_figure :
  ∃ (t_min t_max : ℝ),
    t_min < t_max ∧
    (∀ t, t_min ≤ t ∧ t ≤ t_max → (boundaryEquations t).2 ≥ horizontalBoundary) ∧
    (∫ (t : ℝ) in t_min..t_max, 
      (boundaryEquations t).1 * (8 * Real.cos t)) = boundedArea := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_figure_l1110_111001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1110_111051

/-- Represents a train with its speed and length -/
structure Train where
  speed : ℝ  -- Speed in m/s
  length : ℝ  -- Length in meters

/-- Represents a platform with its length -/
structure Platform where
  length : ℝ  -- Length in meters

/-- Calculates the time taken for a train to cross its own length -/
noncomputable def time_to_cross_train (t : Train) : ℝ :=
  t.length / t.speed

/-- Calculates the time taken for a train to cross a platform -/
noncomputable def time_to_cross_platform (t : Train) (p : Platform) : ℝ :=
  p.length / t.speed

/-- The main theorem to prove -/
theorem train_crossing_time 
  (t : Train) 
  (p : Platform) 
  (h1 : t.speed = 10)  -- 36 kmph converted to m/s
  (h2 : time_to_cross_train t = 12)
  (h3 : time_to_cross_platform t p + time_to_cross_train t = 48.997) :
  time_to_cross_platform t p = 36.997 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1110_111051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_l1110_111086

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

-- Define the left focus of the hyperbola
noncomputable def left_focus (a : ℝ) : ℝ × ℝ := (-Real.sqrt (a^2 + 1), 0)

-- Define the directrix of the parabola
def directrix : ℝ := -4

-- Theorem statement
theorem hyperbola_parabola_intersection (a : ℝ) (h1 : a > 0) 
  (h2 : (left_focus a).1 = directrix) : a = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_l1110_111086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_is_19_hours_l1110_111082

/-- Represents the journey of a boat between points A, B, and C -/
structure BoatJourney where
  stream_velocity : ℝ
  boat_speed : ℝ
  distance_AB : ℝ
  distance_BC : ℝ

/-- Calculates the total time for the boat journey -/
noncomputable def total_journey_time (j : BoatJourney) : ℝ :=
  let downstream_speed := j.boat_speed + j.stream_velocity
  let upstream_speed := j.boat_speed - j.stream_velocity
  let time_downstream := j.distance_AB / downstream_speed
  let time_upstream := j.distance_BC / upstream_speed
  time_downstream + time_upstream

/-- Theorem stating that the total journey time is 19 hours -/
theorem journey_time_is_19_hours (j : BoatJourney) 
    (h1 : j.stream_velocity = 4)
    (h2 : j.boat_speed = 14)
    (h3 : j.distance_AB = 180)
    (h4 : j.distance_BC = j.distance_AB / 2) : 
  total_journey_time j = 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_is_19_hours_l1110_111082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_caterpillars_per_jar_l1110_111067

/-- Proves the number of caterpillars in each jar given the problem conditions -/
theorem caterpillars_per_jar 
  (num_jars : ℕ) 
  (butterfly_rate : ℝ) 
  (price_per_butterfly : ℝ) 
  (total_revenue : ℝ) 
  (h1 : num_jars = 4)
  (h2 : butterfly_rate = 0.6)
  (h3 : price_per_butterfly = 3)
  (h4 : total_revenue = 72)
  : ∃ (caterpillars_per_jar : ℝ), 
    num_jars * butterfly_rate * caterpillars_per_jar * price_per_butterfly = total_revenue ∧ 
    caterpillars_per_jar = 10 := by
  sorry

#check caterpillars_per_jar

end NUMINAMATH_CALUDE_ERRORFEEDBACK_caterpillars_per_jar_l1110_111067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1110_111029

/-- The given function f(x) -/
noncomputable def f (x : ℝ) : ℝ := 4 * (Real.sin x)^2 + 4 * Real.sqrt 3 * Real.sin x * Real.cos x

/-- The smallest positive period of f(x) -/
noncomputable def smallest_period : ℝ := Real.pi

/-- The increasing interval of f(x) -/
noncomputable def increasing_interval (k : ℤ) : Set ℝ := 
  Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3)

/-- The x-coordinate of the center of symmetry -/
noncomputable def center_x (k : ℤ) : ℝ := k * Real.pi / 2 + Real.pi / 12

/-- The y-coordinate of the center of symmetry -/
def center_y : ℝ := 2

theorem f_properties :
  (∀ x : ℝ, f (x + smallest_period) = f x) ∧ 
  (∀ k : ℤ, ∀ x y : ℝ, x ∈ increasing_interval k → y ∈ increasing_interval k → x < y → f x < f y) ∧
  (∀ k : ℤ, ∀ x : ℝ, f (center_x k + x) + f (center_x k - x) = 2 * center_y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1110_111029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_n_is_perfect_cube_l1110_111008

/-- The function that constructs A_n as described in the problem -/
def construct_A (n : ℕ) : ℕ :=
  (10^n.succ - 1) * 10^(2*n + 3) + 7 * 10^(2*n + 2) + 2 * 10^(n + 1) + 9

/-- The theorem stating that A_n is always a perfect cube -/
theorem A_n_is_perfect_cube (n : ℕ) :
  construct_A n = (10^(n.succ) - 1)^3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_n_is_perfect_cube_l1110_111008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_paths_in_three_by_three_grid_l1110_111093

/-- A grid of squares -/
structure Grid :=
  (size : ℕ)

/-- A path in the grid -/
inductive GridPath (g : Grid)
  | nil : GridPath g
  | cons : (ℕ × ℕ) → GridPath g → GridPath g

/-- The number of valid paths in a grid -/
def num_paths (g : Grid) : ℕ :=
  sorry

/-- Theorem: In a 3x3 grid, there are 9 unique paths from one corner to the opposite corner -/
theorem nine_paths_in_three_by_three_grid :
  ∀ (g : Grid), g.size = 3 → num_paths g = 9 :=
by
  sorry

#check nine_paths_in_three_by_three_grid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_paths_in_three_by_three_grid_l1110_111093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_zero_l1110_111070

-- Define the ellipse
noncomputable def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the foci
def are_foci (F₁ F₂ : ℝ × ℝ) : Prop := 
  ∃ c : ℝ, F₁ = (c, 0) ∧ F₂ = (-c, 0) ∧ c^2 = 3

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  is_on_ellipse P.1 P.2

-- Define the area of the triangle
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

-- The main theorem
theorem dot_product_zero (F₁ F₂ P : ℝ × ℝ) :
  are_foci F₁ F₂ →
  point_on_ellipse P →
  triangle_area F₁ P F₂ = 1 →
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_zero_l1110_111070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ways_to_sum_1800_eq_4_pow_300_l1110_111026

/-- The number of ways to represent 1800 as a sum of ones, twos, and threes -/
def ways_to_sum_1800 : ℕ := sorry

/-- 1800 can be expressed as 300 groups of 6 -/
axiom groups_of_six : 1800 = 300 * 6

/-- Each group of 6 can be represented in 4 ways using ones, twos, and threes -/
def ways_per_group : ℕ := 4

/-- The number of ways to sum 1800 is equal to 4^300 -/
theorem ways_to_sum_1800_eq_4_pow_300 : ways_to_sum_1800 = 4^300 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ways_to_sum_1800_eq_4_pow_300_l1110_111026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_approximation_l1110_111064

-- Define the base of the logarithm (common logarithm, base 10)
def log_base : ℝ := 10

-- Define the logarithm function
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log log_base

-- State the theorem
theorem x_value_approximation (x : ℝ) (h : log x = 0.3364) : 
  ∃ ε > 0, |x - 2.186| < ε := by
  sorry

#check x_value_approximation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_approximation_l1110_111064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expressions_evaluation_l1110_111058

-- Define the expressions
noncomputable def expr1 : ℝ := Real.sqrt 9 - (3 - Real.pi) ^ (0 : ℤ) + (1 / 5) ^ (-1 : ℤ)
noncomputable def expr2 : ℝ := (1 - Real.sqrt 3) ^ 2 + Real.sqrt 12

-- State the theorem
theorem expressions_evaluation :
  expr1 = 6 ∧ expr2 = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expressions_evaluation_l1110_111058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_bound_for_complex_sum_product_l1110_111098

open Complex

theorem smallest_bound_for_complex_sum_product : ∃ (lambda : ℝ), 
  (lambda > 0) ∧ 
  (∀ (z1 z2 z3 : ℂ), (abs z1 < 1) → (abs z2 < 1) → (abs z3 < 1) → 
    (z1 + z2 + z3 = 0) → 
    (abs (z1*z2 + z2*z3 + z3*z1))^2 + (abs (z1*z2*z3))^2 < lambda) ∧
  (∀ (mu : ℝ), (mu > 0) → 
    (∀ (z1 z2 z3 : ℂ), (abs z1 < 1) → (abs z2 < 1) → (abs z3 < 1) → 
      (z1 + z2 + z3 = 0) → 
      (abs (z1*z2 + z2*z3 + z3*z1))^2 + (abs (z1*z2*z3))^2 < mu) → 
    lambda ≤ mu) ∧
  lambda = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_bound_for_complex_sum_product_l1110_111098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_f_decreasing_intervals_l1110_111095

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x + 3 * Real.cos x ^ 2

-- Theorem for the minimum value and where it occurs
theorem f_minimum :
  (∃ (min : ℝ), ∀ (x : ℝ), f x ≥ min ∧ min = 2 - Real.sqrt 2) ∧
  (∀ (x : ℝ), f x = 2 - Real.sqrt 2 ↔ ∃ (k : ℤ), x = -3 * π / 8 + k * π) := by
  sorry

-- Theorem for the decreasing intervals
theorem f_decreasing_intervals :
  ∀ (x y : ℝ) (k : ℤ),
    π / 8 + k * π ≤ x ∧
    x < y ∧
    y ≤ 5 * π / 8 + k * π →
    f y < f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_f_decreasing_intervals_l1110_111095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_and_rectangle_area_increase_l1110_111007

theorem square_and_rectangle_area_increase :
  ∀ (s : ℝ), s > 0 →
  let original_area := s^2
  let new_square_side := 1.15 * s
  let new_square_area := new_square_side^2
  let square_area_increase := (new_square_area - original_area) / original_area * 100
  let rectangle_side1 := 1.1 * s
  let rectangle_side2 := 1.2 * s
  let rectangle_area := rectangle_side1 * rectangle_side2
  let rectangle_area_increase := (rectangle_area - original_area) / original_area * 100
  square_area_increase = 32.25 ∧ rectangle_area_increase = 32 :=
by
  intro s hs
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_and_rectangle_area_increase_l1110_111007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1110_111000

/-- The function f(x) = (x^2 + 5x + 6) / (x^2 + 2x + 3) -/
noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x^2 + 2*x + 3)

/-- The range of f is all real numbers -/
theorem range_of_f : Set.range f = Set.univ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1110_111000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1110_111006

/-- A function f defined on the interval [-2, 2] with specific properties -/
noncomputable def f : ℝ → ℝ := sorry

/-- The domain of f is [-2, 2] -/
axiom f_domain (x : ℝ) : f x = f x → x ∈ Set.Icc (-2) 2

/-- f is an even function -/
axiom f_even (x : ℝ) : x ∈ Set.Icc (-2) 2 → f (-x) = f x

/-- f is strictly decreasing on [0, 2] -/
axiom f_decreasing (a b : ℝ) : a ∈ Set.Icc 0 2 → b ∈ Set.Icc 0 2 → a ≠ b → (f a - f b) / (a - b) < 0

/-- The main theorem: range of m that satisfies f(1-m) < f(m) -/
theorem range_of_m (m : ℝ) : f (1 - m) < f m ↔ m ∈ Set.Ioc (-1) (1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1110_111006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_in_hexagon_l1110_111062

-- Definitions
def is_regular_hexagon (S : Set (ℝ × ℝ)) (side_length : ℝ) : Prop := sorry
def is_square (S : Set (ℝ × ℝ)) (side_length : ℝ) : Prop := sorry
def can_turn_within (inner outer : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem
theorem square_in_hexagon (s : ℝ) :
  let hexagon_side : ℝ := 1
  (∃ (square : Set (ℝ × ℝ)) (hexagon : Set (ℝ × ℝ)),
    is_regular_hexagon hexagon hexagon_side ∧
    is_square square s ∧
    can_turn_within square hexagon) ↔
  3 - Real.sqrt 3 ≥ s ∧ s ≥ Real.sqrt (3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_in_hexagon_l1110_111062
