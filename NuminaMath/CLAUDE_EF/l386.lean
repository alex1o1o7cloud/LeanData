import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_properties_l386_38681

-- Define the curves C₁ and C₂
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := (Real.sqrt 2 * Real.cos θ, Real.sin θ)
noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := 1 / (Real.cos θ + Real.sin θ)
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Define point M
def M : ℝ × ℝ := (-1, 2)

-- Statement of the theorem
theorem curves_intersection_properties :
  -- 1. Rectangular equations of C₁ and C₂
  (∀ x y, (x, y) ∈ Set.range C₁ ↔ x^2/2 + y^2 = 1) ∧
  (∀ x y, (x, y) ∈ Set.range C₂ ↔ x + y = 1) ∧
  -- 2. Distance between A and B
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 2 / 3 ∧
  -- 3. Product of distances from M to A and B
  Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) * 
  Real.sqrt ((M.1 - B.1)^2 + (M.2 - B.2)^2) = 14/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_properties_l386_38681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_scoring_schemes_for_C_l386_38623

/-- Represents the possible scores a participant can receive. -/
inductive Score
  | F  -- Failed
  | G  -- Good
  | E  -- Exceptional
deriving DecidableEq

/-- Represents a scoring scheme for a participant. -/
def ScoringScheme := Fin 8 → Score

/-- Returns true if two scoring schemes differ in at least n positions. -/
def differInAtLeastN (s1 s2 : ScoringScheme) (n : Nat) : Prop :=
  (Finset.filter (fun i => s1 i ≠ s2 i) (Finset.univ : Finset (Fin 8))).card ≥ n

/-- Returns true if two scoring schemes are the same in exactly n positions. -/
def sameInExactlyN (s1 s2 : ScoringScheme) (n : Nat) : Prop :=
  (Finset.filter (fun i => s1 i = s2 i) (Finset.univ : Finset (Fin 8))).card = n

/-- The main theorem stating the number of valid scoring schemes for C. -/
theorem number_of_scoring_schemes_for_C :
  ∃ (a b : ScoringScheme),
    sameInExactlyN a b 4 ∧
    (∀ c : ScoringScheme,
      differInAtLeastN a c 4 ∧ differInAtLeastN b c 4 →
      (Finset.filter (fun _ => True) {c}).card = 2401) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_scoring_schemes_for_C_l386_38623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_error_percentage_l386_38632

/-- The percentage of error when dividing by 10 instead of multiplying by 2 -/
theorem error_percentage (N : ℝ) : (((2 * N - N / 10) / (2 * N)) * 100 = 95) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_error_percentage_l386_38632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_length_l386_38617

/-- The length of a goods train given its speed and platform crossing time -/
theorem goods_train_length
  (train_speed : ℝ)
  (platform_length : ℝ)
  (crossing_time : ℝ)
  (h1 : train_speed = 72)  -- km/hr
  (h2 : platform_length = 220)  -- meters
  (h3 : crossing_time = 26)  -- seconds
  : ℝ :=
by
  -- Convert speed to m/s
  let speed_ms := train_speed * 1000 / 3600
  
  -- Calculate total distance covered
  let total_distance := speed_ms * crossing_time
  
  -- Calculate train length
  let train_length := total_distance - platform_length
  
  -- Return the train length
  exact train_length

-- Example usage (commented out to avoid evaluation issues)
-- #eval goods_train_length 72 220 26

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_length_l386_38617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_quadratic_sum_l386_38660

/-- A quadratic function tangent to two lines -/
structure TangentQuadratic where
  a : ℚ
  b : ℚ
  tangent_to_line1 : ∃ x : ℚ, x^2 + a*x + b = -5*x + 6 ∧ 2*x + a = -5
  tangent_to_line2 : ∃ x : ℚ, x^2 + a*x + b = x - 1 ∧ 2*x + a = 1

/-- The sum of coefficients as a fraction -/
def sum_of_coefficients (f : TangentQuadratic) : ℚ :=
  1 + f.a + f.b

theorem tangent_quadratic_sum (f : TangentQuadratic) 
  (h : ∃ p q : ℕ, sum_of_coefficients f = p / q ∧ Nat.Coprime p q) :
  ∃ p q : ℕ, sum_of_coefficients f = p / q ∧ Nat.Coprime p q ∧ 100 * p + q = 2509 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_quadratic_sum_l386_38660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_l386_38687

/-- The speed of a train crossing a bridge -/
noncomputable def train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) : ℝ :=
  (train_length + bridge_length) / crossing_time

/-- Theorem: The speed of the train is approximately 10.91 m/s -/
theorem train_speed_approx :
  let train_length : ℝ := 120
  let bridge_length : ℝ := 480
  let crossing_time : ℝ := 55
  abs (train_speed train_length bridge_length crossing_time - 10.91) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_l386_38687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_a_value_l386_38621

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + x^2 + 3*x + a

-- State the theorem
theorem f_properties (a : ℝ) :
  (∀ x y, x ≤ -1 ∧ y ≤ -1 ∧ x ≤ y → f a y ≤ f a x) ∧
  (∀ x y, x ≥ 3 ∧ y ≥ 3 ∧ x ≤ y → f a y ≤ f a x) ∧
  (∀ x y, -1 ≤ x ∧ x ≤ y ∧ y ≤ 3 → f a x ≤ f a y) ∧
  (∀ x ∈ Set.Icc (-3) 3, f a x ≥ 7/3 → a = 13/3) :=
by
  sorry

-- Additional theorem for the value of a
theorem a_value :
  (∀ x ∈ Set.Icc (-3) 3, f (13/3) x ≥ 7/3) ∧
  (∃ x ∈ Set.Icc (-3) 3, f (13/3) x = 7/3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_a_value_l386_38621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_board_number_is_91_l386_38676

/-- A natural number is composed of n consecutive two-digit numbers in descending order -/
def is_composed_of_consecutive_numbers (x : ℕ) (n : ℕ) : Prop :=
  ∃ (start : ℕ), 10 ≤ start ∧ start < 100 ∧
    x = Finset.sum (Finset.range n) (λ i => (start - i) * 10^(n - 1 - i))

/-- The last digit of a natural number is not 7 -/
def last_digit_not_seven (x : ℕ) : Prop :=
  x % 10 ≠ 7

/-- A natural number has exactly two prime factors -/
def has_exactly_two_prime_factors (x : ℕ) : Prop :=
  ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ x = p * q

/-- Two natural numbers differ by 4 -/
def differ_by_four (a b : ℕ) : Prop :=
  b = a + 4 ∨ a = b + 4

theorem board_number_is_91 :
  ∃ (x n : ℕ),
    is_composed_of_consecutive_numbers x n ∧
    last_digit_not_seven x ∧
    has_exactly_two_prime_factors x ∧
    (∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ x = p * q ∧ differ_by_four p q) ∧
    x = 91 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_board_number_is_91_l386_38676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_is_three_fourths_l386_38608

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 3 = 0

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y + 1 = k * (x + 1)

-- Define the point P
def point_P : ℝ × ℝ := (-1, -1)

-- Define the theorem
theorem line_slope_is_three_fourths :
  ∃ (k : ℝ) (A B C : ℝ × ℝ),
    -- Line l passes through point P
    line_l k point_P.1 point_P.2 ∧
    -- Line l is not vertical to the y-axis
    k ≠ 0 ∧
    -- A and B are on both line l and circle M
    line_l k A.1 A.2 ∧ circle_M A.1 A.2 ∧
    line_l k B.1 B.2 ∧ circle_M B.1 B.2 ∧
    -- C is on circle M
    circle_M C.1 C.2 ∧
    -- ABC is a right triangle
    (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0 →
    -- The slope of line l is 3/4
    k = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_is_three_fourths_l386_38608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_slope_range_line_equation_given_circle_condition_l386_38696

/-- Ellipse C -/
def ellipse (x y : ℝ) : Prop := x^2/3 + y^2 = 1

/-- Line l passing through (0,2) with slope k -/
def line (k x y : ℝ) : Prop := y = k*x + 2

/-- Circle passing through three points -/
def circle_through (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  (x1 - x3) * (y2 - y3) = (x2 - x3) * (y1 - y3)

theorem ellipse_line_intersection_slope_range (k : ℝ) :
  (∃ x1 y1 x2 y2, 
    ellipse x1 y1 ∧ ellipse x2 y2 ∧ 
    line k x1 y1 ∧ line k x2 y2 ∧ 
    (x1 ≠ x2 ∨ y1 ≠ y2)) →
  k ∈ Set.Ioi 1 ∪ Set.Iio (-1) := by
  sorry

theorem line_equation_given_circle_condition (k : ℝ) :
  (∃ x1 y1 x2 y2, 
    ellipse x1 y1 ∧ ellipse x2 y2 ∧ 
    line k x1 y1 ∧ line k x2 y2 ∧
    circle_through x1 y1 x2 y2 1 0) →
  k = -7/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_slope_range_line_equation_given_circle_condition_l386_38696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_interval_l386_38659

/-- A cubic function with parameters a and b -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + x^2 + a*x + b

/-- The derivative of f with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*x + a

theorem max_value_on_interval (a b : ℝ) :
  (f_deriv a 3 = 0) →
  (f a b 3 = 4) →
  (∃ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) 1 ∧ f a b x = -4/3) ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-2 : ℝ) 1 → f a b y ≤ -4/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_interval_l386_38659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_given_ellipse_and_angle_l386_38625

/-- A structure representing a conic section (ellipse or hyperbola) -/
structure Conic where
  a : ℝ  -- semi-major axis (for ellipse) or real semi-axis (for hyperbola)
  c : ℝ  -- half the distance between foci
  is_ellipse : Bool

/-- The eccentricity of a conic section -/
noncomputable def eccentricity (conic : Conic) : ℝ :=
  conic.c / conic.a

/-- The angle between three points in 2D space -/
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

theorem hyperbola_eccentricity_given_ellipse_and_angle 
  (ellipse : Conic) (hyperbola : Conic) (P : ℝ × ℝ) :
  ellipse.is_ellipse = true →
  hyperbola.is_ellipse = false →
  ellipse.c = hyperbola.c →
  eccentricity ellipse = Real.sin (π / 4) →
  let F₁ : ℝ × ℝ := (-ellipse.c, 0)
  let F₂ : ℝ × ℝ := (ellipse.c, 0)
  Real.cos (angle F₁ P F₂) = 1 / 2 →
  eccentricity hyperbola = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_given_ellipse_and_angle_l386_38625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_precision_l386_38693

/-- Represents the precision level of a monetary value -/
inductive Precision
  | HundredBillion
  | Billion
  | HundredMillion
  | Percent

/-- Determines the precision of a given monetary value in billions of yuan -/
noncomputable def determinePrecision (value : ℝ) : Precision :=
  if value * 100 % 1 ≠ 0 then Precision.HundredMillion
  else if value * 10 % 1 ≠ 0 then Precision.Billion
  else if value % 1 ≠ 0 then Precision.HundredBillion
  else Precision.Percent

/-- The total local general budget revenue of Shenzhen from January to May -/
def revenue : ℝ := 21.658

theorem revenue_precision :
  determinePrecision revenue = Precision.HundredMillion := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_precision_l386_38693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_monotone_increasing_a_l386_38618

/-- The function f(x) = e^x(-x^2 + 2x + a) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (-x^2 + 2*x + a)

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (-x^2 + a + 2)

/-- The function is monotonically increasing if its derivative is non-negative -/
def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc a (a + 1), f_deriv a x ≥ 0

/-- The maximum value of a for which f(x) is monotonically increasing in [a, a+1] -/
noncomputable def max_a : ℝ := (Real.sqrt 5 - 1) / 2

/-- Theorem stating that if f(x) is monotonically increasing in [a, a+1], then a ≤ max_a -/
theorem max_monotone_increasing_a :
  ∀ a : ℝ, is_monotone_increasing a → a ≤ max_a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_monotone_increasing_a_l386_38618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_thirteen_l386_38626

theorem divisible_by_thirteen (n : ℕ+) : ∃ k : ℤ, 4^(2*n.val+1) + 3^(n.val+2) = 13*k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_thirteen_l386_38626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distinct_interior_angles_overlapping_quadrilaterals_l386_38622

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define the configuration of two overlapping quadrilaterals
structure OverlappingQuadrilaterals where
  ABCD : Quadrilateral
  PQRS : Quadrilateral
  C_equals_S : ABCD.C = PQRS.C
  D_equals_R : ABCD.D = PQRS.D

-- Define the sum of interior angles of a quadrilateral
def sum_interior_angles (_ : Quadrilateral) : ℝ := 360

-- Theorem: The sum of all distinct interior angles in the overlapping quadrilaterals is 540°
theorem sum_distinct_interior_angles_overlapping_quadrilaterals 
  (config : OverlappingQuadrilaterals) : ℝ := by
  have h1 : sum_interior_angles config.ABCD = 360 := rfl
  have h2 : sum_interior_angles config.PQRS = 360 := rfl
  exact 360 + 360 - 180


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distinct_interior_angles_overlapping_quadrilaterals_l386_38622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_inequality_solution_l386_38688

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the solution set for f(cos x) ≥ 0
def cos_solution_set : Set ℝ := {x | 0 ≤ x ∧ x ≤ Real.pi / 2}

-- Assumption that f(cos x) ≥ 0 for x in the cos_solution_set
axiom f_cos_nonneg : ∀ x ∈ cos_solution_set, f (Real.cos x) ≥ 0

-- Define the solution set for f(sin x) ≥ 0
def sin_solution_set : Set ℝ := {x | ∃ k : ℤ, 2 * k * Real.pi ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi}

-- Theorem statement
theorem sin_inequality_solution :
  ∀ x : ℝ, f (Real.sin x) ≥ 0 ↔ x ∈ sin_solution_set :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_inequality_solution_l386_38688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_figure_l386_38633

noncomputable def curve_x (t : ℝ) : ℝ := 4 * Real.sqrt 2 * (Real.cos t) ^ 3
noncomputable def curve_y (t : ℝ) : ℝ := 2 * Real.sqrt 2 * (Real.sin t) ^ 3

def boundary_line : ℝ := 2

def valid_t_range : Set ℝ := Set.Icc (-Real.pi/4) (Real.pi/4)

noncomputable def area_calculation : ℝ := 2 * ∫ t in valid_t_range, curve_y t * (deriv curve_x t)

theorem area_of_bounded_figure :
  area_calculation = 3 * Real.pi / 2 - 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_figure_l386_38633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_to_origin_l386_38609

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Check if a point is on the hyperbola -/
def isOnHyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The origin point (0, 0) -/
def origin : Point := ⟨0, 0⟩

/-- The right focus of the hyperbola -/
noncomputable def rightFocus (h : Hyperbola) : Point :=
  ⟨Real.sqrt (h.a^2 + h.b^2), 0⟩

theorem hyperbola_distance_to_origin 
  (h : Hyperbola) 
  (p : Point) 
  (h_on : isOnHyperbola h p) 
  (h_dist : distance p (rightFocus h) = 1) : 
  distance p origin = h.a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_to_origin_l386_38609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_and_evaluation_l386_38602

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (2 * x) + Real.log x

theorem f_derivative_and_evaluation :
  (∀ x, x ≠ 0 → HasDerivAt f (-1 / (2 * x^2) + 1 / x) x) ∧
  deriv f (-1) = -3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_and_evaluation_l386_38602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l386_38624

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x * Real.exp x else -x / Real.exp x

-- State the theorem
theorem solution_set_of_inequality (x : ℝ) :
  f (x - 2) < Real.exp 1 ↔ 1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l386_38624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l386_38694

def set_A : Set ℝ := {x : ℝ | |x - 1| ≤ 3}

def set_B : Set ℝ := {x : ℝ | x ≠ 1 ∧ 1 / (x - 1) ≤ 1}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = Set.Ioc (-2) 1 ∪ Set.Icc 2 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l386_38694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_one_l386_38648

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x + Real.sqrt (x^2 + 1))

-- State the theorem
theorem sum_equals_one (a b : ℝ) (h : f a + f (b - 1) = 0) : a + b = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_one_l386_38648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_properties_l386_38698

-- Define the points A and B
def A : ℝ × ℝ := (8, -6)
def B : ℝ × ℝ := (2, 2)

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Define the perpendicular bisector equation
def perp_bisector (x y : ℝ) : Prop := 3*x - 4*y - 23 = 0

-- Define the parallel tangent lines equations
def parallel_tangent_line1 (x y : ℝ) : Prop := 4*x + 3*y + 20 = 0
def parallel_tangent_line2 (x y : ℝ) : Prop := 4*x + 3*y - 20 = 0

theorem geometric_properties :
  -- Part 1: Perpendicular bisector
  (∀ x y : ℝ, perp_bisector x y ↔ 
    (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2) ∧
  -- Part 2: Parallel tangent lines
  (∀ x y : ℝ, (parallel_tangent_line1 x y ∨ parallel_tangent_line2 x y) ↔
    (∃ t : ℝ, x = A.1 + t*(B.1 - A.1) ∧ y = A.2 + t*(B.2 - A.2) ∧ circle_eq x y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_properties_l386_38698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scenario_is_systematic_sampling_l386_38601

/-- Represents a sampling method -/
inductive SamplingMethod
  | Systematic
  | Stratified
  | Lottery
  | Random
deriving Repr

/-- Represents a class of students -/
structure StudentClass where
  size : Nat
deriving Repr

/-- Represents a sampling scenario -/
structure SamplingScenario where
  classes : List StudentClass
  selectedNumber : Nat
deriving Repr

/-- Determines if a sampling scenario uses systematic sampling -/
def isSystematicSampling (scenario : SamplingScenario) : Prop :=
  scenario.classes.all (fun c => c.size ≥ scenario.selectedNumber) ∧
  scenario.classes.length > 1 ∧
  scenario.selectedNumber > 0

/-- Theorem: The given scenario uses systematic sampling -/
theorem scenario_is_systematic_sampling (scenario : SamplingScenario)
  (h1 : scenario.classes.length ≥ 12)
  (h2 : ∀ c ∈ scenario.classes, c.size = 50)
  (h3 : scenario.selectedNumber = 14) :
  isSystematicSampling scenario := by
  sorry

#eval SamplingMethod.Systematic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scenario_is_systematic_sampling_l386_38601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_cosine_identity_l386_38673

open Real

theorem trigonometric_identities 
  (α : ℝ) 
  (h1 : Real.sin α - Real.cos α = 1/5)
  (h2 : π < α ∧ α < 3*π/2) :
  Real.sin α * Real.cos α = 12/25 ∧ Real.sin α + Real.cos α = -7/5 := by sorry

theorem cosine_identity 
  (x : ℝ) 
  (h1 : Real.cos (40 * π/180 + x) = 1/4)
  (h2 : -π < x ∧ x < -π/2) :
  Real.cos (140 * π/180 - x) + (Real.cos (50 * π/180 - x))^2 = 11/16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_cosine_identity_l386_38673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l386_38614

theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  let angle : ℝ := (3 * π) / 4
  (a = (1, 1)) →
  (‖a - 2 • b‖ = Real.sqrt 10) →
  ((a.1 * b.1 + a.2 * b.2) = ‖a‖ * ‖b‖ * Real.cos angle) →
  ‖b‖ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l386_38614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_monotonicity_l386_38603

-- Define the piecewise function f
noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  if x > 0 then -x^2 + 2*x
  else if x = 0 then 0
  else x^2 + m*x

-- State the theorem
theorem odd_function_and_monotonicity (m : ℝ) (a : ℝ) :
  (∀ x, f x m = -f (-x) m) →  -- f is an odd function
  (m = 2 ∧ 
   (∀ x y, x ∈ Set.Icc (-1) (a - 2) → y ∈ Set.Icc (-1) (a - 2) → x ≤ y → f x m ≤ f y m) ↔ 
   a ∈ Set.Ioo 1 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_monotonicity_l386_38603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_x_of_quadratic_through_points_l386_38645

/-- A quadratic function passing through three given points -/
noncomputable def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- The x-coordinate of the vertex of a quadratic function -/
noncomputable def VertexX (a b c : ℝ) : ℝ := -b / (2 * a)

theorem vertex_x_of_quadratic_through_points
  (a b c : ℝ) (h1 : QuadraticFunction a b c (-1) = 7)
  (h2 : QuadraticFunction a b c 5 = 7)
  (h3 : QuadraticFunction a b c 6 = 10) :
  VertexX a b c = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_x_of_quadratic_through_points_l386_38645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_even_implies_even_degree_l386_38615

/-- Ω function that calculates the sum of exponents in the prime factorization -/
def Ω : ℕ → ℕ := sorry

/-- P is a polynomial of degree n with positive integer coefficients -/
def P (n : ℕ) (a : Fin n → ℕ) : ℕ → ℕ := sorry

/-- Main theorem: If Ω(P(k)) is even for all k, then n is even -/
theorem omega_even_implies_even_degree (n : ℕ) (a : Fin n → ℕ) :
  (∀ k : ℕ, Even (Ω (P n a k))) → Even n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_even_implies_even_degree_l386_38615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_g_l386_38653

open MeasureTheory Interval Real

/-- K is a positive continuous function on [0, 1] × [0, 1] -/
noncomputable def K : ℝ → ℝ → ℝ := sorry

/-- f is a positive continuous function on [0, 1] -/
noncomputable def f : ℝ → ℝ := sorry

/-- g is a positive continuous function on [0, 1] -/
noncomputable def g : ℝ → ℝ := sorry

theorem f_eq_g (hK : Continuous (λ p : ℝ × ℝ => K p.1 p.2) ∧ ∀ x y, x ∈ I ∧ y ∈ I → K x y > 0)
               (hf : Continuous f ∧ ∀ x, x ∈ I → f x > 0)
               (hg : Continuous g ∧ ∀ x, x ∈ I → g x > 0)
               (h1 : ∀ x ∈ I, ∫ y in I, f y * K x y = g x)
               (h2 : ∀ x ∈ I, ∫ y in I, g y * K x y = f x) :
               ∀ x ∈ I, f x = g x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_g_l386_38653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l386_38628

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.cos (ω * x) * Real.sin (ω * x - Real.pi/3) + Real.sqrt 3 * (Real.cos (ω * x))^2 - Real.sqrt 3 / 4

def has_symmetry_center (f : ℝ → ℝ) (d : ℝ) : Prop :=
  ∃ c : ℝ, ∀ x : ℝ, f (c + x) = f (c - x) ∧ 
  ∀ y : ℝ, y ≠ c → |y - c| ≥ d

theorem f_properties (ω : ℝ) (A B : ℝ) (a b : ℝ) :
  ω > 0 →
  has_symmetry_center (f ω) (Real.pi/4) →
  f ω A = 0 →
  Real.sin B = 4/5 →
  a = Real.sqrt 3 →
  (ω = 1 ∧ 
   (∃ k : ℤ, f ω (k * Real.pi/2 + Real.pi/12) = 0) ∧
   b = 2/5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l386_38628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_in_second_quadrant_l386_38655

theorem tan_value_in_second_quadrant (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin α = 4 / 5) : 
  Real.tan α = -4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_in_second_quadrant_l386_38655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_inequality_impossibility_l386_38647

/-- Theorem: Impossibility of simultaneous vector inequalities -/
theorem vector_inequality_impossibility {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b c : V) : ¬(Real.sqrt 3 * ‖a‖ < ‖b - c‖ ∧ 
                  Real.sqrt 3 * ‖b‖ < ‖c - a‖ ∧ 
                  Real.sqrt 3 * ‖c‖ < ‖a - b‖) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_inequality_impossibility_l386_38647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_42_l386_38637

/-- The distance between locations A and B in kilometers -/
noncomputable def distance_AB : ℝ := 42

/-- The speed of person A in km/hour before 7:00 -/
noncomputable def speed_A : ℝ := sorry

/-- The speed of person B in km/hour before 7:00 -/
noncomputable def speed_B : ℝ := sorry

/-- The time (in hours) they travel before 7:00 when starting at 6:50 -/
noncomputable def time_before_peak : ℝ := 1/6

/-- The time (in hours) they travel after 7:00 when starting at 6:50 -/
noncomputable def time_during_peak : ℝ := sorry

/-- Assertion that A and B meet 24 km from A when both start at 6:50 -/
axiom meet_at_24 : speed_A * (time_before_peak + time_during_peak / 2) + 
                    speed_B * (time_before_peak + time_during_peak / 2) = 24

/-- Assertion that A and B meet 20 km from A when B starts 20 minutes earlier -/
axiom meet_at_20 : speed_A * (time_before_peak + time_during_peak / 2) + 
                    speed_B * (time_before_peak + time_during_peak / 2 + 1/3) = 20

/-- Assertion that A and B meet at the midpoint when A starts 20 minutes later -/
axiom meet_at_midpoint : speed_A * time_during_peak / 2 + 
                         speed_B * (time_before_peak + time_during_peak) = distance_AB / 2

/-- Theorem stating that the distance between A and B is 42 km -/
theorem distance_is_42 : distance_AB = 42 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_42_l386_38637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_balanced_numbers_l386_38679

-- Define the number of positive divisors function
def d (k : ℕ) : ℕ := (Nat.divisors k).card

-- Define what it means for a number to be balanced
def is_balanced (n : ℕ) : Prop :=
  (d (n - 1) ≤ d n ∧ d n ≤ d (n + 1)) ∨ (d (n - 1) ≥ d n ∧ d n ≥ d (n + 1))

-- Define the set of balanced numbers
def balanced_numbers : Set ℕ := {n | is_balanced n}

-- Theorem statement
theorem infinitely_many_balanced_numbers : ∃ f : ℕ → ℕ, StrictMono f ∧ ∀ n, is_balanced (f n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_balanced_numbers_l386_38679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_divisors_count_l386_38651

/-- A function that returns true if a number has an odd number of positive divisors --/
def has_odd_divisors (n : ℕ) : Bool :=
  (Nat.divisors n).card % 2 = 1

/-- The count of positive integers less than 100 with an odd number of divisors --/
def count_odd_divisors : ℕ :=
  (Finset.range 100).filter (fun n => has_odd_divisors n) |>.card

/-- Theorem stating that the count of positive integers less than 100 
    with an odd number of divisors is 9 --/
theorem odd_divisors_count : count_odd_divisors = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_divisors_count_l386_38651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_percentage_approx_l386_38631

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℚ

/-- Calculates the area of a square -/
def Square.area (s : Square) : ℚ := s.sideLength ^ 2

/-- Represents the main square EFGH -/
def mainSquare : Square := { sideLength := 6 }

/-- Represents the small square at the bottom left corner -/
def bottomLeftSquare : Square := { sideLength := 1 }

/-- Represents the square extending upward from 2 units right of the bottom left corner -/
def middleSquare : Square := { sideLength := 3 }

/-- Represents the square extending downward from 2 units right of the top left corner -/
def topRightSquare : Square := { sideLength := 4 }

/-- Calculates the total shaded area -/
def totalShadedArea : ℚ :=
  bottomLeftSquare.area + middleSquare.area + topRightSquare.area

/-- Calculates the percentage of the main square that is shaded -/
noncomputable def shadedPercentage : ℚ :=
  (totalShadedArea / mainSquare.area) * 100

/-- Theorem stating that the shaded percentage is approximately 72.22% -/
theorem shaded_percentage_approx :
  ∃ ε > 0, |shadedPercentage - 72.22| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_percentage_approx_l386_38631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_increase_q3_to_q4_l386_38690

noncomputable def question_values : List ℚ := [100, 250, 400, 600, 1000, 1800, 3500, 6000]

def percent_increase (a b : ℚ) : ℚ := (b - a) / a * 100

def min_percent_increase (values : List ℚ) : Nat × Nat :=
  (List.range (values.length - 1)).foldl
    (fun (acc : Nat × Nat) i =>
      let curr := percent_increase (values.get! i) (values.get! (i+1))
      let prev := percent_increase (values.get! acc.1) (values.get! (acc.1 + 1))
      if curr < prev then (i, i+1) else acc)
    (0, 1)

theorem smallest_increase_q3_to_q4 :
  min_percent_increase question_values = (2, 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_increase_q3_to_q4_l386_38690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commute_speed_theorem_l386_38680

/-- Calculates the average speed for the return trip given the conditions of the commute -/
noncomputable def average_speed_return (speed_to_work : ℝ) (total_commute_time : ℝ) (distance : ℝ) : ℝ :=
  let time_to_work := distance / speed_to_work
  let time_from_work := total_commute_time - time_to_work
  distance / time_from_work

/-- Theorem stating that under the given conditions, the average speed returning from work is 30 mph -/
theorem commute_speed_theorem (speed_to_work : ℝ) (total_commute_time : ℝ) (distance : ℝ)
  (h1 : speed_to_work = 45)
  (h2 : total_commute_time = 1)
  (h3 : distance = 18) :
  average_speed_return speed_to_work total_commute_time distance = 30 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval average_speed_return 45 1 18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_commute_speed_theorem_l386_38680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_fraction_at_event_l386_38684

/-- Represents a school with a given number of students and ratio of boys to girls -/
structure School where
  total_students : ℕ
  boys_ratio : ℕ
  girls_ratio : ℕ

/-- Calculates the number of girls in a school -/
def girls_count (s : School) : ℕ :=
  s.total_students * s.girls_ratio / (s.boys_ratio + s.girls_ratio)

/-- The fraction of girls at the joint event is 221/440 -/
theorem girls_fraction_at_event (rawlings waverly : School)
    (h1 : rawlings.total_students = 240)
    (h2 : rawlings.boys_ratio = 3)
    (h3 : rawlings.girls_ratio = 2)
    (h4 : waverly.total_students = 200)
    (h5 : waverly.boys_ratio = 3)
    (h6 : waverly.girls_ratio = 5) :
    (girls_count rawlings + girls_count waverly : ℚ) / (rawlings.total_students + waverly.total_students) = 221 / 440 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_fraction_at_event_l386_38684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l386_38666

/-- Given a parabola y² = 2px (p > 0) and a line y = √3x - √3 passing through M(1,0), 
    intersecting the parabola at B and the directrix at A, 
    if AM = MB, then the equation of the parabola is y² = 4x. -/
theorem parabola_equation (p : ℝ) (A B M : ℝ × ℝ) (h1 : p > 0) 
    (h2 : M = (1, 0)) 
    (h3 : ∀ x y, y = Real.sqrt 3 * x - Real.sqrt 3 → (x, y) ∈ {(x, y) | y^2 = 2 * p * x} → (x, y) = B)
    (h4 : ∀ x y, y = Real.sqrt 3 * x - Real.sqrt 3 → (x, y) ∈ {(x, y) | x = -p/2} → (x, y) = A)
    (h5 : A - M = M - B) : 
  ∀ x y, y^2 = 2 * p * x ↔ y^2 = 4 * x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l386_38666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_eq_one_sufficient_not_necessary_l386_38675

/-- The distance from the origin to the line ax + by + c = 0 --/
noncomputable def distanceToLine (a b c : ℝ) : ℝ := |c| / Real.sqrt (a^2 + b^2)

/-- A line intersects a unit circle if the distance from the origin to the line is less than 1 --/
def lineIntersectsUnitCircle (a b c : ℝ) : Prop := distanceToLine a b c < 1

/-- The specific line equation x - y + k = 0 --/
def specificLine (k : ℝ) : ℝ → ℝ → Prop := fun x y ↦ x - y + k = 0

theorem k_eq_one_sufficient_not_necessary :
  (∀ k, k = 1 → ∃ x y, specificLine k x y ∧ x^2 + y^2 = 1) ∧
  ¬(∀ x y, x^2 + y^2 = 1 → ∃ k, k = 1 ∧ specificLine k x y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_eq_one_sufficient_not_necessary_l386_38675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_weights_unique_l386_38668

-- Define the fruits as an enumeration
inductive Fruit
| Banana
| Pear
| Melon
| Kiwi
| Apple

-- Define the weight function
def weight : Fruit → Nat
| Fruit.Banana => 230
| Fruit.Pear => 170
| Fruit.Melon => 1600
| Fruit.Kiwi => 210
| Fruit.Apple => 150

-- State the theorem
theorem fruit_weights_unique :
  -- The melon weighs more than the other 4 items
  (∀ f : Fruit, f ≠ Fruit.Melon → weight Fruit.Melon > weight f) ∧
  -- The pear and kiwi together weigh as much as the banana and apple together
  (weight Fruit.Pear + weight Fruit.Kiwi = weight Fruit.Banana + weight Fruit.Apple) ∧
  -- The kiwi weighs less than the banana but more than the pear
  (weight Fruit.Kiwi < weight Fruit.Banana ∧ weight Fruit.Kiwi > weight Fruit.Pear) ∧
  -- All weights are among the given values
  (∀ f : Fruit, (weight f = 150) ∨ (weight f = 170) ∨ (weight f = 210) ∨ (weight f = 230) ∨ (weight f = 1600)) ∧
  -- The weight function is injective (each fruit has a unique weight)
  (∀ f g : Fruit, weight f = weight g → f = g) :=
by sorry

#eval weight Fruit.Banana
#eval weight Fruit.Pear
#eval weight Fruit.Melon
#eval weight Fruit.Kiwi
#eval weight Fruit.Apple

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_weights_unique_l386_38668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_angle_specific_triangle_l386_38634

/-- A right triangle with side lengths -/
structure RightTriangle where
  /-- Length of the hypotenuse -/
  c : ℝ
  /-- Length of one leg -/
  a : ℝ
  /-- Length of the other leg -/
  b : ℝ
  /-- The triangle satisfies the Pythagorean theorem -/
  pyth : c^2 = a^2 + b^2

/-- Tangent of an angle in a right triangle -/
noncomputable def tan_angle (t : RightTriangle) : ℝ := t.a / t.b

theorem tan_angle_specific_triangle :
  ∃ (t : RightTriangle), t.c = 13 ∧ t.a = 12 ∧ tan_angle t = 12/5 :=
by
  sorry

#check tan_angle_specific_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_angle_specific_triangle_l386_38634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_iff_a_eq_one_l386_38646

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 / (3^x + 1) - a

-- Define what it means for a function to be odd
def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- Theorem statement
theorem f_is_odd_iff_a_eq_one :
  ∀ a : ℝ, is_odd_function (f a) ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_iff_a_eq_one_l386_38646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_man_weight_is_83_l386_38620

/-- The weight of the new man given the initial number of men, average weight increase, and weight of the replaced man. -/
def weight_of_new_man (initial_men : ℕ) (avg_weight_increase : ℚ) (replaced_man_weight : ℕ) : ℕ :=
  replaced_man_weight + (initial_men * (avg_weight_increase * 2).num / 2).toNat

/-- Theorem stating that the weight of the new man is 83 kg under the given conditions. -/
theorem new_man_weight_is_83 :
  weight_of_new_man 10 (5/2) 58 = 83 := by
  sorry

#eval weight_of_new_man 10 (5/2) 58

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_man_weight_is_83_l386_38620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_real_domain_l386_38606

/-- The function f(x) defined by (x-4)/(mx^2+4mx+3) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (x - 4) / (m * x^2 + 4 * m * x + 3)

/-- The domain of f is ℝ -/
def domain_is_real (m : ℝ) : Prop := ∀ x, m * x^2 + 4 * m * x + 3 ≠ 0

/-- The theorem stating the range of m for which the domain of f is ℝ -/
theorem range_of_m_for_real_domain :
  ∀ m : ℝ, domain_is_real m ↔ (0 ≤ m ∧ m < 3/4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_real_domain_l386_38606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l386_38678

/-- The function f(x) = |sin(ωx + π/3)| -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := |Real.sin (ω * x + Real.pi / 3)|

/-- ω is greater than 1 -/
def omega_gt_one (ω : ℝ) : Prop := ω > 1

/-- f is monotonically decreasing in the interval [π, 5π/4] -/
def f_decreasing (ω : ℝ) : Prop :=
  ∀ x y, Real.pi ≤ x ∧ x < y ∧ y ≤ 5*Real.pi/4 → f ω x ≥ f ω y

/-- The theorem stating the range of ω -/
theorem omega_range (ω : ℝ) : 
  omega_gt_one ω → f_decreasing ω → 7/6 ≤ ω ∧ ω ≤ 4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l386_38678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_when_a_8_range_of_a_l386_38607

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - (sin x) / (cos x)^3

-- Define the domain
def domain : Set ℝ := { x | 0 < x ∧ x < π / 2 }

-- Theorem for monotonicity when a = 8
theorem monotonicity_when_a_8 :
  ∀ x ∈ domain, 
    (x < π / 4 → (∀ y ∈ domain, x < y → f 8 x < f 8 y)) ∧
    (x > π / 4 → (∀ y ∈ domain, x < y → f 8 x > f 8 y)) :=
by sorry

-- Theorem for the range of a
theorem range_of_a :
  ∀ a : ℝ, (∀ x ∈ domain, f a x < sin (2 * x)) ↔ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_when_a_8_range_of_a_l386_38607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_of_555_l386_38611

theorem prime_divisors_of_555 : 
  (Finset.filter (fun p : ℕ => Nat.Prime p ∧ p ∣ 555) (Finset.range 556)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_of_555_l386_38611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_a_upper_bound_l386_38605

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x + Real.exp 2 - 7

-- Part 1: Tangent line equation
theorem tangent_line_equation (a : ℝ) (h : a = 2) :
  let f' := λ x => Real.exp x - a
  (λ x => (Real.exp 2 - 2) * x - 7) = λ x => f a 2 + f' 2 * (x - 2) := by sorry

-- Part 2: Range of a
theorem a_upper_bound (a : ℝ) :
  (∀ x ≥ 0, f a x ≥ (7/4) * x^2) → a ≤ Real.exp 2 - 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_a_upper_bound_l386_38605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_angles_theorem_l386_38671

-- Define IsAngle as a local definition
def IsAngle (x : ℝ) : Prop := 0 ≤ x ∧ x < 360

-- Define AreParallel as a local definition
def AreParallel (α β : ℝ) : Prop := sorry  -- Definition for parallel sides of angles

theorem parallel_angles_theorem (α β : ℝ) : 
  (IsAngle α ∧ IsAngle β) →  -- α and β are angles
  AreParallel α β →        -- The sides of α and β are parallel
  (α = 4 * β - 30) →         -- One angle is 30° less than four times the other
  ((α = 38 ∧ β = 42) ∨ (α = 10 ∧ β = 10)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_angles_theorem_l386_38671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_password_probability_l386_38697

def password_length : ℕ := 6
def known_digits : ℕ := 5
def even_numbers : Finset ℕ := {0, 2, 4, 6, 8}

theorem password_probability :
  (1 : ℚ) / Finset.card even_numbers = (1 : ℚ) / 5 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_password_probability_l386_38697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_transformation_l386_38642

/-- A move on the table replaces a -1 with 0 and multiplies neighboring cells by -1 -/
def Move (m n : ℕ) := (Fin m × Fin n) → ℤ → ℤ

/-- The table is valid if it contains exactly one -1 and the rest are +1 -/
def ValidInitialTable {m n : ℕ} (t : Fin m × Fin n → ℤ) : Prop :=
  (∃! p, t p = -1) ∧ (∀ p, t p = 1 ∨ t p = -1)

/-- The table contains only zeroes -/
def AllZeroes {m n : ℕ} (t : Fin m × Fin n → ℤ) : Prop :=
  ∀ p, t p = 0

/-- A sequence of moves is valid if each move follows the rules -/
def ValidMoveSequence (m n : ℕ) (moves : List (Move m n)) : Prop :=
  sorry

/-- The main theorem: the table can be transformed to all zeroes iff at least one of m or n is odd -/
theorem table_transformation (m n : ℕ) :
  (∃ (t : Fin m × Fin n → ℤ) (moves : List (Move m n)),
    ValidInitialTable t ∧ ValidMoveSequence m n moves ∧
    AllZeroes (moves.foldl (fun t move => fun p => move p (t p)) t)) ↔
  m % 2 = 1 ∨ n % 2 = 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_transformation_l386_38642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_principle_students_l386_38667

theorem pigeonhole_principle_students (n m : ℕ) (h1 : n = 38) (h2 : m = 12) :
  ∃ (month : Fin m), 4 ≤ (Finset.univ.filter (λ student : Fin n ↦ student.val % m = month.val)).card :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_principle_students_l386_38667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_work_days_b_work_days_is_48_l386_38610

noncomputable def work_rate (days : ℝ) : ℝ := 1 / days

theorem b_work_days 
  (combined_days : ℝ) 
  (a_days : ℝ) 
  (h1 : combined_days = 16) 
  (h2 : a_days = 24) : 
  ℝ := by
  let combined_rate := work_rate combined_days
  let a_rate := work_rate a_days
  let b_rate := combined_rate - a_rate
  exact 1 / b_rate

theorem b_work_days_is_48 
  (combined_days : ℝ) 
  (a_days : ℝ) 
  (h1 : combined_days = 16) 
  (h2 : a_days = 24) : 
  b_work_days combined_days a_days h1 h2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_work_days_b_work_days_is_48_l386_38610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_x_coordinate_range_l386_38629

/-- The range of x-coordinates for the center of circle C -/
def valid_x_range : Set ℝ := Set.Icc 0 (12/5)

/-- The line l: y = 2x - 4 -/
def line_l (x : ℝ) : ℝ := 2 * x - 4

/-- Circle C with radius 1 and center (a, 2a - 4) -/
def circle_C (a x y : ℝ) : Prop :=
  (x - a)^2 + (y - (2 * a - 4))^2 = 1

/-- Circle D: x² + (y+1)² = 4 -/
def circle_D (x y : ℝ) : Prop :=
  x^2 + (y + 1)^2 = 4

/-- The main theorem -/
theorem center_x_coordinate_range :
  ∀ a : ℝ,
  (∃ x y : ℝ, circle_C a x y ∧ circle_D x y) →
  a ∈ valid_x_range :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_x_coordinate_range_l386_38629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_externally_tangent_l386_38616

/-- Curve C₁ in polar coordinates -/
noncomputable def C₁ (θ : ℝ) : ℝ := 2 * Real.cos θ

/-- Circle C₂ in polar coordinates -/
def C₂ (ρ θ : ℝ) : Prop := ρ^2 - 2 * Real.sqrt 3 * ρ * Real.sin θ + 2 = 0

/-- Center of C₁ in Cartesian coordinates -/
def center_C₁ : ℝ × ℝ := (1, 0)

/-- Center of C₂ in Cartesian coordinates -/
noncomputable def center_C₂ : ℝ × ℝ := (0, Real.sqrt 3)

/-- Radius of C₁ -/
def radius_C₁ : ℝ := 1

/-- Radius of C₂ -/
def radius_C₂ : ℝ := 1

/-- Distance between centers of C₁ and C₂ -/
noncomputable def distance_between_centers : ℝ := Real.sqrt ((1 - 0)^2 + (0 - Real.sqrt 3)^2)

theorem curves_externally_tangent :
  distance_between_centers = radius_C₁ + radius_C₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_externally_tangent_l386_38616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_jogs_one_and_half_hours_l386_38644

/-- The number of days in two weeks -/
def days_in_two_weeks : ℕ := 14

/-- The total time spent jogging in hours -/
def total_jogging_time : ℚ := 21

/-- The time Mr. John jogs each morning in hours -/
noncomputable def daily_jogging_time : ℚ := total_jogging_time / days_in_two_weeks

/-- Theorem stating that Mr. John jogs for 1.5 hours each morning -/
theorem john_jogs_one_and_half_hours : daily_jogging_time = 3/2 := by
  unfold daily_jogging_time
  unfold total_jogging_time
  unfold days_in_two_weeks
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_jogs_one_and_half_hours_l386_38644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_theorem_l386_38669

theorem election_votes_theorem (total_votes : ℕ) : 
  (2 : ℕ) ≤ total_votes →  -- at least two candidates
  (0.7 * (total_votes : ℝ) - 0.3 * (total_votes : ℝ) = 176) →
  total_votes = 440 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_theorem_l386_38669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_multiplication_l386_38613

theorem binary_multiplication :
  let a : Nat := 0b1101101
  let b : Nat := 0b111
  a * b = 0b10010010111 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_multiplication_l386_38613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_equal_remainder_l386_38619

def primes : List Nat := [11, 13, 17, 19]

def R (n : Nat) : Nat :=
  (primes.map fun p => n % p).sum

theorem smallest_n_equal_remainder : 
  (∀ m : Nat, m > 0 → m < 37 → R m ≠ R (m + 2)) ∧ 
  R 37 = R 39 := by
  sorry

#eval R 37
#eval R 39

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_equal_remainder_l386_38619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_intersections_count_l386_38670

/-- The number of intersection points of diagonals in a convex n-gon -/
def diagonalIntersections (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2) * (n - 3) / 24

/-- Axiom: The polygon is convex -/
axiom is_convex_polygon (n : ℕ) : Prop

/-- Axiom: No three diagonals intersect at a single point -/
axiom no_triple_intersections (n : ℕ) : Prop

/-- The actual number of intersection points (to be proved equal to diagonalIntersections) -/
axiom number_of_intersection_points (n : ℕ) : ℕ

/-- Theorem: Number of intersection points of diagonals in a convex n-gon -/
theorem diagonal_intersections_count (n : ℕ) (h1 : n > 3) :
  diagonalIntersections n = number_of_intersection_points n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_intersections_count_l386_38670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_length_and_perimeter_l386_38672

/-- SimilarTriangles type to represent similar triangles -/
def SimilarTriangles (t1 t2 : Triangle) : Prop :=
  sorry

/-- Triangle type to represent a triangle -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to calculate the perimeter of a triangle -/
def Perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Given two similar triangles MNP and XYZ, prove the length of XZ and the perimeter of XYZ -/
theorem similar_triangles_length_and_perimeter 
  (MNP XYZ : Triangle)
  (MN NP XY : ℝ) 
  (h_similar : SimilarTriangles MNP XYZ) 
  (h_MN : MN = 8) 
  (h_NP : NP = 10) 
  (h_XY : XY = 20) : 
  ∃ (XZ : ℝ), XZ = 25 ∧ Perimeter XYZ = 65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_length_and_perimeter_l386_38672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_proof_l386_38650

/-- The length of a bridge where two people meet in the middle -/
noncomputable def bridge_length (v : ℝ) : ℝ := (2 * v + 1) / 6

/-- Theorem stating the length of the bridge given the conditions -/
theorem bridge_length_proof (v : ℝ) :
  let person_a_speed := v + 2
  let person_b_speed := v - 1
  let meeting_time := 10 / 60 -- 10 minutes in hours
  let half_bridge := meeting_time * person_a_speed
  half_bridge = meeting_time * person_b_speed →
  bridge_length v = 2 * half_bridge :=
by
  intro h
  unfold bridge_length
  -- The proof steps would go here
  sorry

#check bridge_length_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_proof_l386_38650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2011_is_sin_l386_38640

-- Define the sequence of functions
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => λ x => Real.cos x
  | (n + 1) => λ x => deriv (f n) x

-- State the theorem
theorem f_2011_is_sin : f 2011 = λ x => Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2011_is_sin_l386_38640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_staircase_theorem_l386_38695

/-- Represents the staircase construction problem --/
def StaircaseProblem (totalToothpicks : ℕ) : Prop :=
  ∃ (n : ℕ),
    (∀ k : ℕ, k ≤ n → 2 * k = 2 * k) ∧  -- This line is modified
    (2 * n * (n + 1) = totalToothpicks) ∧
    (∀ m : ℕ, m > n → 2 * m * (m + 1) > totalToothpicks)

/-- The theorem statement for the staircase problem --/
theorem staircase_theorem :
  StaircaseProblem 360 → ∃ (n : ℕ), n = 13 := by
  sorry

#check staircase_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_staircase_theorem_l386_38695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_value_remainder_value_l386_38638

/-- Definition of a 1/500-array entry -/
def array_entry (r c : ℕ) : ℚ :=
  (1 / (2 * 500)^r) * (1 / 500^c)

/-- Sum of squares of all terms in a 1/500-array -/
noncomputable def sum_of_squares : ℚ :=
  ∑' r, ∑' c, (array_entry r c)^2

/-- Theorem: The sum of squares of all terms in a 1/500-array is 1/249997000001 -/
theorem sum_of_squares_value : sum_of_squares = 1 / 249997000001 := by
  sorry

/-- The remainder when numerator + denominator is divided by 500 -/
def remainder : ℕ :=
  (1 + 249997000001) % 500

/-- Theorem: The remainder is 2 -/
theorem remainder_value : remainder = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_value_remainder_value_l386_38638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_homogeneous_exact_constant_l386_38635

/-- A homogeneous function of degree n -/
def IsHomogeneous (f : ℝ → ℝ → ℝ) (n : ℝ) :=
  ∀ (t x y : ℝ), t ≠ 0 → f (t * x) (t * y) = t^n * f x y

/-- An exact differential equation -/
def IsExact (M N : ℝ → ℝ → ℝ) :=
  ∀ (x y : ℝ), (deriv (fun y => M x y) y) = (deriv (fun x => N x y) x)

/-- The differential equation M dx + N dy = 0 -/
def DiffEq (M N : ℝ → ℝ → ℝ) (y : ℝ → ℝ) :=
  ∀ x, M x (y x) + N x (y x) * (deriv y x) = 0

theorem homogeneous_exact_constant 
  (M N : ℝ → ℝ → ℝ) (n : ℝ) (y : ℝ → ℝ) :
  IsHomogeneous M n → IsHomogeneous N n → IsExact M N → 
  DiffEq M N y → 
  ∃ (c : ℝ), ∀ x, x * M x (y x) + (y x) * N x (y x) = c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_homogeneous_exact_constant_l386_38635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_given_z_less_than_3_l386_38689

theorem range_of_a_given_z_less_than_3 (a b : ℝ) (z : ℂ) (i : ℂ) :
  i * i = -1 →
  z = a^2 - b + (b - 2*a) * i →
  z.re < 3 ∧ z.im = 0 →
  -1 < a ∧ a < 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_given_z_less_than_3_l386_38689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_domain_real_l386_38674

/-- The function c(x) with parameter m -/
noncomputable def c (m : ℝ) (x : ℝ) : ℝ := (3 * x + m) / (3 * x^2 + m * x + 7)

/-- The theorem stating the condition for c(x) to have domain ℝ -/
theorem c_domain_real (m : ℝ) : 
  (∀ x, c m x ≠ 0) ↔ m ∈ Set.Ioo (-2 * Real.sqrt 21) (2 * Real.sqrt 21) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_domain_real_l386_38674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_labeling_possible_l386_38656

-- Define the possible contents of a box
inductive BoxContents
  | Oranges
  | Guavas
  | Both

-- Define a box with a label and actual contents
structure Box where
  label : BoxContents
  contents : BoxContents

-- Define the market setup
def MarketSetup : Prop :=
  ∃ (box1 box2 box3 : Box),
    -- There are three boxes with different contents
    box1.contents ≠ box2.contents ∧
    box2.contents ≠ box3.contents ∧
    box1.contents ≠ box3.contents ∧
    -- Each box has a different label
    box1.label ≠ box2.label ∧
    box2.label ≠ box3.label ∧
    box1.label ≠ box3.label ∧
    -- Labels are incorrect
    box1.label ≠ box1.contents ∧
    box2.label ≠ box2.contents ∧
    box3.label ≠ box3.contents ∧
    -- One box is labeled as "Oranges and Guavas"
    (box1.label = BoxContents.Both ∨ box2.label = BoxContents.Both ∨ box3.label = BoxContents.Both)

-- Theorem statement
theorem correct_labeling_possible (setup : MarketSetup) :
  ∃ (examined_box : Box),
    examined_box.label = BoxContents.Both ∧
    (∀ (fruit : BoxContents),
      (fruit = BoxContents.Oranges ∨ fruit = BoxContents.Guavas) →
      (examined_box.contents = fruit ∨ examined_box.contents = BoxContents.Both) →
      ∃ (box1 box2 box3 : Box),
        box1.label ≠ box1.contents ∧
        box2.label ≠ box2.contents ∧
        box3.label ≠ box3.contents ∧
        (box1.contents = examined_box.contents ∧
         box2.contents ≠ examined_box.contents ∧
         box3.contents ≠ examined_box.contents)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_labeling_possible_l386_38656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l386_38643

noncomputable def f (x a b : ℝ) : ℝ := 4^x + a * 2^x + b

theorem function_properties :
  ∃ (a b : ℝ),
    (f 0 a b = 1) ∧
    (f (-1) a b = -5/4) ∧
    (∀ x, f x a b = 4^x - 3 * 2^x + 3) ∧
    (∀ x, 0 ≤ x ∧ x ≤ 2 → 1 ≤ f x a b ∧ f x a b ≤ 25) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l386_38643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l386_38654

noncomputable def f (x φ : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x - φ) - Real.cos (2 * x - φ)

theorem max_value_of_f (φ : ℝ) (h1 : |φ| < π / 2) 
  (h2 : ∀ x, f x φ = f (-x) φ) : 
  ∃ x ∈ Set.Icc (-π/6) (π/3), ∀ y ∈ Set.Icc (-π/6) (π/3), f x φ ≥ f y φ ∧ f x φ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l386_38654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l386_38692

-- Define the parabola
def parabola (P : ℝ × ℝ) : Prop := P.1^2 = 2 * P.2

-- Define the focus of the parabola
noncomputable def focus : ℝ × ℝ := (0, 1/4)

-- Define the fixed point M
def M : ℝ × ℝ := (1, 2)

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statement
theorem min_sum_distances :
  ∃ (min : ℝ), min = 5/2 ∧
  ∀ (P : ℝ × ℝ), parabola P →
    distance P M + distance P focus ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l386_38692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l386_38685

/-- A line from (c,0) to (4,4) divides six unit squares into two equal areas -/
theorem equal_area_division (c : ℝ) : 
  (2 * (4 - c) = 3) ↔ c = 5/2 := by
  sorry

/-- Helper lemma to show the total area is 6 -/
lemma total_area_is_six : 
  (6 : ℝ) = 6 * (1 : ℝ) := by
  sorry

/-- Helper lemma to show half the total area is 3 -/
lemma half_total_area_is_three :
  (3 : ℝ) = (6 : ℝ) / 2 := by
  sorry

/-- Helper lemma for the triangle area formula -/
lemma triangle_area_formula (c : ℝ) :
  2 * (4 - c) = (1/2) * (4 - c) * 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l386_38685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_right_angle_l386_38686

/-- Circle C with equation x^2 + y^2 - 4x + 2y + m = 0 -/
def Circle (m : ℝ) := {(x, y) : ℝ × ℝ | x^2 + y^2 - 4*x + 2*y + m = 0}

/-- Points A and B are the intersections of Circle C with the y-axis -/
def intersectsYAxis (m : ℝ) : Prop :=
  ∃ (a b : ℝ), (0, a) ∈ Circle m ∧ (0, b) ∈ Circle m ∧ a ≠ b

/-- The angle ACB is 90 degrees -/
def rightAngleACB (m : ℝ) : Prop :=
  ∃ (a b : ℝ), (0, a) ∈ Circle m ∧ (0, b) ∈ Circle m ∧
  ((2 - 0)^2 + (-1 - a)^2 + (2 - 0)^2 + (-1 - b)^2 = (a - b)^2)

theorem circle_intersection_right_angle (m : ℝ) :
  intersectsYAxis m → rightAngleACB m → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_right_angle_l386_38686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_line_l386_38677

/-- Given a line in polar coordinates with θ = 2π/3, prove its Cartesian equation is √3x + y = 0 -/
theorem polar_to_cartesian_line :
  ∀ x y : ℝ, (∃ r : ℝ, r * Real.cos (2 * π / 3) = x ∧ r * Real.sin (2 * π / 3) = y) ↔ Real.sqrt 3 * x + y = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_line_l386_38677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l386_38661

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ π) ∧
  (∀ x ∈ Set.Icc (-π/6) (π/4), f x ≤ 2) ∧
  (∀ x ∈ Set.Icc (-π/6) (π/4), f x ≥ -1) ∧
  (∃ x ∈ Set.Icc (-π/6) (π/4), f x = 2) ∧
  (∃ x ∈ Set.Icc (-π/6) (π/4), f x = -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l386_38661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_proof_l386_38600

noncomputable def f (x : ℝ) := (3 * x^2 / Real.sqrt (1 - x)) + Real.log (3 * x + 1) / Real.log 10

def domain_of_f : Set ℝ := { x | -1/3 < x ∧ x < 1 }

theorem domain_proof : 
  ∀ x : ℝ, x ∈ domain_of_f ↔ (1 - x > 0 ∧ 3 * x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_proof_l386_38600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficients_not_divisible_by_65_l386_38699

theorem coefficients_not_divisible_by_65 :
  let expansion := (Polynomial.X + 1 : Polynomial ℤ)^65
  (Finset.filter (fun k => ¬ (65 ∣ expansion.coeff k)) (Finset.range 66)).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficients_not_divisible_by_65_l386_38699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_problem_l386_38652

theorem lcm_problem : Nat.lcm (Nat.lcm 12 16) (Nat.lcm 18 24) = 144 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_problem_l386_38652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rss_decreases_as_r_squared_increases_statement_a_is_correct_l386_38657

-- Define the correlation coefficient R² as a real number between 0 and 1
def R_squared : Type := {r : ℝ // 0 ≤ r ∧ r ≤ 1}

-- Define the residual sum of squares (RSS) as a non-negative real number
def RSS : Type := {rss : ℝ // 0 ≤ rss}

-- Define a function that maps R² to RSS
noncomputable def rss_function : R_squared → RSS := sorry

-- Theorem: As R² increases, RSS decreases
theorem rss_decreases_as_r_squared_increases 
  (r1 r2 : R_squared) (h : r1.val < r2.val) :
  (rss_function r2).val < (rss_function r1).val := by
  sorry

-- Statement A is correct
theorem statement_a_is_correct :
  ∀ (r1 r2 : R_squared), r1.val < r2.val → 
  (rss_function r2).val < (rss_function r1).val := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rss_decreases_as_r_squared_increases_statement_a_is_correct_l386_38657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l386_38627

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*y + 2 = 0

-- Define the line
def line (h t : ℝ) : ℝ × ℝ := (h - 10 + t, t)

-- Define the condition for the line to be tangent to the circle
def is_tangent (h : ℝ) : Prop :=
  ∃ t : ℝ, circle_eq ((line h t).1) ((line h t).2) ∧
  ∀ s : ℝ, s ≠ t → ¬ circle_eq ((line h s).1) ((line h s).2)

-- Theorem statement
theorem line_tangent_to_circle : 
  ∃ h : ℝ, is_tangent h ∧ h = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l386_38627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_x_minus_y_minus_two_l386_38664

/-- The angle of inclination of a line given its equation -/
noncomputable def angle_of_inclination (a b c : ℝ) : ℝ :=
  Real.arctan (a / b)

/-- The line equation: ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

theorem angle_of_inclination_x_minus_y_minus_two (line : LineEquation) :
  line.a = 1 ∧ line.b = -1 ∧ line.c = -2 →
  angle_of_inclination line.a line.b line.c = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_x_minus_y_minus_two_l386_38664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_to_tetrahedron_ratio_l386_38662

/-- Represents a polyhedron with faces, edges, and vertices. -/
structure Polyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- An octahedron is a polyhedron with 8 faces, 12 edges, and 6 vertices. -/
def Octahedron : Polyhedron :=
  { faces := 8, edges := 12, vertices := 6 }

/-- A tetrahedron is a polyhedron with 4 faces, 6 edges, and 4 vertices. -/
def Tetrahedron : Polyhedron :=
  { faces := 4, edges := 6, vertices := 4 }

/-- Represents the volume of a polyhedron. -/
noncomputable def volume (p : Polyhedron) : ℝ := sorry

/-- Represents the surface area of a polyhedron. -/
noncomputable def surfaceArea (p : Polyhedron) : ℝ := sorry

/-- 
  Theorem: When a regular tetrahedron is formed by extending four non-adjacent faces of an octahedron,
  the volume and surface area of the tetrahedron are both twice those of the octahedron.
-/
theorem octahedron_to_tetrahedron_ratio :
  volume Tetrahedron = 2 * volume Octahedron ∧
  surfaceArea Tetrahedron = 2 * surfaceArea Octahedron :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_to_tetrahedron_ratio_l386_38662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_l386_38641

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on the given ellipse -/
def on_ellipse (p : Point) : Prop :=
  p.x^2 / 16 + p.y^2 / 36 = 1

/-- Checks if a point lies on the given parabola -/
def on_parabola (p : Point) : Prop :=
  p.x = -p.y^2 / (8 * Real.sqrt 5) - 2 * Real.sqrt 5

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem statement for the intersection of ellipse and parabola -/
theorem ellipse_parabola_intersection :
  ∃ (p1 p2 : Point),
    on_ellipse p1 ∧ on_ellipse p2 ∧
    on_parabola p1 ∧ on_parabola p2 ∧
    ∀ (q1 q2 : Point),
      on_ellipse q1 ∧ on_ellipse q2 ∧
      on_parabola q1 ∧ on_parabola q2 →
      distance q1 q2 ≤ distance p1 p2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_l386_38641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_correct_l386_38691

/-- The minimum number of moves required to make all numbers the same -/
def min_moves (n : ℕ) : ℕ := n^2 + n

/-- The set of divisors of 10^n -/
def divisors_of_10_pow (n : ℕ) : Finset ℕ := sorry

/-- A single move in the game -/
def move (s : Finset ℕ) : Finset ℕ := sorry

theorem min_moves_correct (n : ℕ) :
  ∃ (k : ℕ), ∃ (s : Finset ℕ),
    (s.card = 1) ∧ 
    (∃ (initial : Finset ℕ), 
      initial = divisors_of_10_pow n ∧
      (∃ (moves : ℕ → Finset ℕ),
        moves 0 = initial ∧
        moves k = s ∧
        ∀ i : ℕ, i < k → moves (i + 1) = move (moves i))) ∧
    k = min_moves n ∧
    ∀ (m : ℕ), m < k → 
      ¬∃ (t : Finset ℕ), (t.card = 1) ∧ 
        (∃ (initial : Finset ℕ), 
          initial = divisors_of_10_pow n ∧
          (∃ (moves : ℕ → Finset ℕ),
            moves 0 = initial ∧
            moves m = t ∧
            ∀ i : ℕ, i < m → moves (i + 1) = move (moves i))) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_correct_l386_38691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l386_38604

/-- A quadratic function with zeros at -3 and 1, and minimum value -4 -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

/-- A function g defined in terms of f -/
def g (m : ℝ) (x : ℝ) : ℝ := m * f x + 2

theorem quadratic_function_properties :
  (∀ x, f x = 0 ↔ x = -3 ∨ x = 1) ∧
  (∃ x, ∀ y, f x ≤ f y ∧ f x = -4) ∧
  (∀ x, f x = x^2 + 2*x - 3) ∧
  (∀ m < 0, ∃! x, x ≥ -3 ∧ g m x = 0) ∧
  (∀ m > 0, m ≤ 8/7 →
    (∀ x ∈ Set.Icc (-3) (3/2), |g m x| ≤ 9/4*m + 2) ∧
    (∃ x ∈ Set.Icc (-3) (3/2), |g m x| = 9/4*m + 2)) ∧
  (∀ m > 8/7,
    (∀ x ∈ Set.Icc (-3) (3/2), |g m x| ≤ 4*m - 2) ∧
    (∃ x ∈ Set.Icc (-3) (3/2), |g m x| = 4*m - 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l386_38604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equality_l386_38636

theorem trigonometric_expression_equality : 
  (Real.sin (38 * π / 180) * Real.sin (38 * π / 180) + 
   Real.cos (38 * π / 180) * Real.sin (52 * π / 180) - 
   Real.tan (15 * π / 180)^2) / (3 * Real.tan (15 * π / 180)) = 
  2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equality_l386_38636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_22_5_deg_formula_l386_38658

open Real

theorem tan_22_5_deg_formula : 
  (Real.tan (22.5 * π / 180)) / (1 - (Real.tan (22.5 * π / 180))^2) = 1/2 := by
  have h1 : Real.tan (45 * π / 180) = 1 := by sorry
  have h2 : Real.tan (45 * π / 180) = (2 * Real.tan (22.5 * π / 180)) / (1 - (Real.tan (22.5 * π / 180))^2) := by sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_22_5_deg_formula_l386_38658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l386_38639

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 3) + Real.cos (2 * x + Real.pi / 6)

theorem f_properties :
  -- 1. Maximum value is √2
  (∀ x, f x ≤ Real.sqrt 2) ∧ (∃ x, f x = Real.sqrt 2) ∧
  -- 2. Smallest positive period is π
  (∀ x, f (x + Real.pi) = f x) ∧ (∀ t > 0, t < Real.pi → ∃ x, f (x + t) ≠ f x) ∧
  -- 3. Decreasing in the interval (π/24, 13π/24)
  (∀ x y, Real.pi/24 < x ∧ x < y ∧ y < 13*Real.pi/24 → f y < f x) ∧
  -- 4. Equivalent to √2cos(2(x - π/24))
  (∀ x, f x = Real.sqrt 2 * Real.cos (2 * (x - Real.pi / 24))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l386_38639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_height_of_cut_cone_l386_38630

/-- Represents a cone cut into a frustum -/
structure CutCone where
  /-- Radius of the top base of the frustum -/
  r : ℝ
  /-- Ratio of the radii of the top and bottom bases of the frustum -/
  ratio : ℝ
  /-- Slant height of the frustum -/
  frustum_slant_height : ℝ

/-- The slant height of the original cone -/
noncomputable def slant_height (c : CutCone) : ℝ :=
  3 * Real.sqrt 6

/-- Theorem stating the slant height of the original cone given the conditions -/
theorem slant_height_of_cut_cone (c : CutCone) 
    (h_ratio : c.ratio = 2)
    (h_frustum_height : c.frustum_slant_height = 6) :
    slant_height c = 3 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_height_of_cut_cone_l386_38630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_opposite_faces_l386_38649

/-- Represents the faces of a cube -/
structure CubeFaces :=
  (faces : Finset ℕ)
  (distinct : faces.card = 6)
  (range : ∀ n ∈ faces, 6 ≤ n ∧ n ≤ 11)

/-- Represents the sum of numbers on four lateral faces -/
def lateral_sum (c : CubeFaces) (s : ℕ) : Prop :=
  ∃ (f₁ f₂ : Finset ℕ), f₁ ⊆ c.faces ∧ f₂ ⊆ c.faces ∧ f₁.card = 4 ∧ f₂.card = 2 ∧ 
    f₁ ∩ f₂ = ∅ ∧ f₁ ∪ f₂ = c.faces ∧ f₁.sum id = s

theorem cube_opposite_faces (c : CubeFaces) 
  (h₁ : lateral_sum c 36)
  (h₂ : lateral_sum c 33)
  (h₃ : 10 ∈ c.faces) :
  ∃ f₁ f₂ : Finset ℕ, f₁ ⊆ c.faces ∧ f₂ ⊆ c.faces ∧ 
    f₁.card = 1 ∧ f₂.card = 1 ∧ f₁ ∩ f₂ = ∅ ∧
    10 ∈ f₁ ∧ 8 ∈ f₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_opposite_faces_l386_38649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l386_38612

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ a then Real.cos x else 1 / x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, -1 ≤ f a x ∧ f a x ≤ 1) → a ∈ Set.Ici 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l386_38612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_louisa_average_speed_l386_38665

/-- Proves that given the conditions of Louisa's travel, her average speed was 100/3 miles per hour -/
theorem louisa_average_speed :
  ∀ (v : ℝ), 
  v > 0 →  -- Ensure speed is positive
  350 / v = 250 / v + 3 →  -- Condition from the problem
  v = 100 / 3 := by
  intro v hv_pos hv_eq
  -- Proof steps would go here
  sorry

#check louisa_average_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_louisa_average_speed_l386_38665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_proof_l386_38663

/-- The principal amount -/
noncomputable def P : ℝ := 9999.99999999988

/-- The time period in years -/
def t : ℝ := 1

/-- The difference between compound interest and simple interest -/
def diff : ℝ := 25

/-- Simple interest calculation -/
noncomputable def simpleInterest (r : ℝ) : ℝ := P * r * t

/-- Compound interest calculation (half-yearly) -/
noncomputable def compoundInterest (r : ℝ) : ℝ := P * ((1 + r / 2) ^ (2 * t) - 1)

/-- The interest rate that satisfies the given conditions -/
def interestRate : ℝ := 0.1

theorem interest_rate_proof :
  compoundInterest interestRate - simpleInterest interestRate = diff := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_proof_l386_38663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_megatek_manufacturing_percentage_l386_38683

/-- The angle in degrees that represents a full circle. -/
noncomputable def full_circle : ℝ := 360

/-- The angle in degrees of the manufacturing department's sector in the circle graph. -/
noncomputable def manufacturing_sector : ℝ := 54

/-- The percentage of employees in the manufacturing department. -/
noncomputable def manufacturing_percentage : ℝ := (manufacturing_sector / full_circle) * 100

/-- Theorem stating that the percentage of Megatek employees in manufacturing is 15%. -/
theorem megatek_manufacturing_percentage : manufacturing_percentage = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_megatek_manufacturing_percentage_l386_38683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_ceiling_distance_l386_38682

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Theorem: Distance from fly to ceiling in a rectangular room -/
theorem fly_ceiling_distance (fly : Point3D) (P : Point3D) : 
  fly.x = 3 → 
  fly.y = 5 → 
  fly.z = 6 → 
  distance fly P = 10 → 
  ∃ (ceiling : Point3D), distance fly ceiling = Real.sqrt 66 := by
  sorry

#check fly_ceiling_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_ceiling_distance_l386_38682
