import Mathlib

namespace NUMINAMATH_CALUDE_union_equals_B_implies_m_leq_neg_three_l475_47511

def A : Set ℝ := {x | x^2 - 3*x - 10 < 0}
def B (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 1 - 3*m}

theorem union_equals_B_implies_m_leq_neg_three (m : ℝ) : A ∪ B m = B m → m ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_B_implies_m_leq_neg_three_l475_47511


namespace NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l475_47554

/-- The number of different rectangles with sides parallel to the grid
    that can be formed by connecting four dots in a 5x5 square array of dots. -/
def num_rectangles_in_5x5_grid : ℕ :=
  (Nat.choose 5 2) * (Nat.choose 5 2)

/-- Theorem stating that the number of rectangles in a 5x5 grid is 100. -/
theorem rectangles_in_5x5_grid :
  num_rectangles_in_5x5_grid = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l475_47554


namespace NUMINAMATH_CALUDE_billy_ticket_difference_l475_47536

/-- The difference between initial tickets and remaining tickets after purchases -/
def ticket_difference (initial_tickets yoyo_cost keychain_cost plush_toy_cost : ℝ) : ℝ :=
  initial_tickets - (initial_tickets - (yoyo_cost + keychain_cost + plush_toy_cost))

/-- Theorem stating the ticket difference for Billy's specific case -/
theorem billy_ticket_difference :
  ticket_difference 48.5 11.7 6.3 16.2 = 14.3 := by
  sorry

end NUMINAMATH_CALUDE_billy_ticket_difference_l475_47536


namespace NUMINAMATH_CALUDE_garden_cut_percentage_l475_47589

theorem garden_cut_percentage (rows : ℕ) (flowers_per_row : ℕ) (remaining : ℕ) :
  rows = 50 →
  flowers_per_row = 400 →
  remaining = 8000 →
  (rows * flowers_per_row - remaining : ℚ) / (rows * flowers_per_row) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_garden_cut_percentage_l475_47589


namespace NUMINAMATH_CALUDE_f_is_odd_l475_47594

def f (x : ℝ) : ℝ := |x + 1| - |x - 1|

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_f_is_odd_l475_47594


namespace NUMINAMATH_CALUDE_unique_integer_solution_l475_47559

theorem unique_integer_solution (a b c : ℤ) :
  a^2 + b^2 + c^2 + 3 < a*b + 3*b + 2*c ↔ a = 1 ∧ b = 2 ∧ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l475_47559


namespace NUMINAMATH_CALUDE_parabola_equation_l475_47572

/-- A parabola in the Cartesian coordinate system -/
structure Parabola where
  -- The equation of the parabola in the form y^2 = ax
  a : ℝ
  -- Condition that the parabola is symmetric with respect to the x-axis
  x_axis_symmetry : True
  -- Condition that the vertex is at the origin
  vertex_at_origin : True
  -- Condition that the parabola passes through the point (2, 4)
  passes_through_point : a * 2 = 4^2

/-- Theorem stating that the parabola y^2 = 8x satisfies the given conditions -/
theorem parabola_equation : ∃ (p : Parabola), p.a = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l475_47572


namespace NUMINAMATH_CALUDE_range_of_a_l475_47538

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * Real.sin x + Real.cos x < 2) → 
  -Real.sqrt 3 < a ∧ a < Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l475_47538


namespace NUMINAMATH_CALUDE_factorization_problems_l475_47514

variables (a b m n : ℝ)

theorem factorization_problems :
  (m^2 * (a - b) + 4 * n^2 * (b - a) = (a - b) * (m + 2*n) * (m - 2*n)) ∧
  (-a^3 + 2*a^2*b - a*b^2 = -a * (a - b)^2) := by sorry

end NUMINAMATH_CALUDE_factorization_problems_l475_47514


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l475_47548

def is_circle (t : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 - 6*t*x + 8*t*y + 25 = 0

theorem sufficient_not_necessary :
  (∀ t : ℝ, t > 1 → is_circle t) ∧
  (∃ t : ℝ, is_circle t ∧ ¬(t > 1)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l475_47548


namespace NUMINAMATH_CALUDE_prime_sum_of_powers_l475_47500

theorem prime_sum_of_powers (n : ℕ) : 
  (∃ (a b c : ℤ), a + b + c = 0 ∧ Nat.Prime (Int.natAbs (a^n + b^n + c^n))) ↔ Even n := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_of_powers_l475_47500


namespace NUMINAMATH_CALUDE_adam_final_money_l475_47593

/-- Calculates the final amount of money Adam has after a series of transactions --/
theorem adam_final_money (initial : ℚ) (game_cost : ℚ) (snack_cost : ℚ) (found : ℚ) (allowance : ℚ) :
  initial = 5.25 →
  game_cost = 2.30 →
  snack_cost = 1.75 →
  found = 1.00 →
  allowance = 5.50 →
  initial - game_cost - snack_cost + found + allowance = 7.70 := by
  sorry

end NUMINAMATH_CALUDE_adam_final_money_l475_47593


namespace NUMINAMATH_CALUDE_vector_at_negative_one_l475_47567

/-- A line in 3D space parameterized by t -/
structure ParametricLine where
  -- The vector on the line at t = 0
  v0 : ℝ × ℝ × ℝ
  -- The vector on the line at t = 1
  v1 : ℝ × ℝ × ℝ

/-- The vector on the line at a given t -/
def vectorAtT (line : ParametricLine) (t : ℝ) : ℝ × ℝ × ℝ :=
  let (x0, y0, z0) := line.v0
  let (x1, y1, z1) := line.v1
  (x0 + t * (x1 - x0), y0 + t * (y1 - y0), z0 + t * (z1 - z0))

theorem vector_at_negative_one (line : ParametricLine) 
  (h1 : line.v0 = (2, 6, 16)) 
  (h2 : line.v1 = (1, 1, 4)) : 
  vectorAtT line (-1) = (3, 11, 28) := by
  sorry

end NUMINAMATH_CALUDE_vector_at_negative_one_l475_47567


namespace NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l475_47597

/-- Proves that the ratio of the time taken to row upstream to the time taken to row downstream is 2:1 -/
theorem upstream_downstream_time_ratio 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : boat_speed = 78) 
  (h2 : stream_speed = 26) : 
  (boat_speed - stream_speed) / (boat_speed + stream_speed) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l475_47597


namespace NUMINAMATH_CALUDE_total_cubes_is_seven_l475_47537

/-- Represents a stack of unit cubes -/
structure CubeStack where
  bottomLayer : Nat
  middleLayer : Nat
  topLayer : Nat

/-- The total number of cubes in a stack -/
def totalCubes (stack : CubeStack) : Nat :=
  stack.bottomLayer + stack.middleLayer + stack.topLayer

/-- Given stack of unit cubes -/
def givenStack : CubeStack :=
  { bottomLayer := 4
  , middleLayer := 2
  , topLayer := 1 }

/-- Theorem: The total number of unit cubes in the given stack is 7 -/
theorem total_cubes_is_seven : totalCubes givenStack = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_cubes_is_seven_l475_47537


namespace NUMINAMATH_CALUDE_exponent_multiplication_l475_47557

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l475_47557


namespace NUMINAMATH_CALUDE_shaded_area_theorem_total_shaded_area_l475_47582

-- Define the length of the diagonal
def diagonal_length : ℝ := 8

-- Define the number of congruent squares
def num_squares : ℕ := 25

-- Theorem statement
theorem shaded_area_theorem (diagonal : ℝ) (num_squares : ℕ) 
  (h1 : diagonal = diagonal_length) 
  (h2 : num_squares = num_squares) : 
  (diagonal^2 / 2) = 32 := by
  sorry

-- Main theorem connecting the given conditions to the final area
theorem total_shaded_area : 
  (diagonal_length^2 / 2) = 32 := by
  exact shaded_area_theorem diagonal_length num_squares rfl rfl

end NUMINAMATH_CALUDE_shaded_area_theorem_total_shaded_area_l475_47582


namespace NUMINAMATH_CALUDE_point_coordinates_l475_47507

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the third quadrant -/
def in_third_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Distance from a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates (p : Point) 
  (h1 : in_third_quadrant p)
  (h2 : distance_to_x_axis p = 8)
  (h3 : distance_to_y_axis p = 5) :
  p = Point.mk (-5) (-8) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l475_47507


namespace NUMINAMATH_CALUDE_perimeter_of_triangle_l475_47560

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 9

-- Define the foci
def left_focus : ℝ × ℝ := sorry
def right_focus : ℝ × ℝ := sorry

-- Define points P and Q on the left branch
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- State that P and Q are on the hyperbola
axiom P_on_hyperbola : hyperbola P.1 P.2
axiom Q_on_hyperbola : hyperbola Q.1 Q.2

-- State that PQ passes through the left focus
axiom PQ_through_left_focus : sorry

-- Define the length of PQ
def PQ_length : ℝ := 7

-- Define the property of hyperbola for P and Q
axiom hyperbola_property_P : dist P right_focus - dist P left_focus = 6
axiom hyperbola_property_Q : dist Q right_focus - dist Q left_focus = 6

-- Theorem to prove
theorem perimeter_of_triangle : 
  dist P right_focus + dist Q right_focus + PQ_length = 26 := sorry

end NUMINAMATH_CALUDE_perimeter_of_triangle_l475_47560


namespace NUMINAMATH_CALUDE_neg_one_quad_residue_iff_prime_mod_four_infinite_primes_mod_four_l475_47569

/-- For an odd prime p, -1 is a quadratic residue modulo p if and only if p ≡ 1 (mod 4) -/
theorem neg_one_quad_residue_iff_prime_mod_four (p : Nat) (h_prime : Nat.Prime p) (h_odd : p % 2 = 1) :
  (∃ x, x^2 % p = (p - 1) % p) ↔ p % 4 = 1 := by sorry

/-- There are infinitely many prime numbers congruent to 1 modulo 4 -/
theorem infinite_primes_mod_four :
  ∀ n, ∃ p, p > n ∧ Nat.Prime p ∧ p % 4 = 1 := by sorry

end NUMINAMATH_CALUDE_neg_one_quad_residue_iff_prime_mod_four_infinite_primes_mod_four_l475_47569


namespace NUMINAMATH_CALUDE_cubic_polynomial_inequality_iff_coeff_conditions_l475_47585

/-- A cubic polynomial -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Evaluation of a cubic polynomial at a point -/
def eval (p : CubicPolynomial) (x : ℝ) : ℝ :=
  p.a * x^3 + p.b * x^2 + p.c * x + p.d

/-- The inequality condition for the polynomial -/
def satisfiesInequality (p : CubicPolynomial) : Prop :=
  ∀ x y, x ≥ 0 → y ≥ 0 → eval p (x + y) ≥ eval p x + eval p y

/-- The conditions on the coefficients -/
def satisfiesCoeffConditions (p : CubicPolynomial) : Prop :=
  p.a > 0 ∧ p.d ≤ 0 ∧ 8 * p.b^3 ≥ 243 * p.a^2 * p.d

theorem cubic_polynomial_inequality_iff_coeff_conditions (p : CubicPolynomial) :
  satisfiesInequality p ↔ satisfiesCoeffConditions p := by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_inequality_iff_coeff_conditions_l475_47585


namespace NUMINAMATH_CALUDE_derivative_sqrt_l475_47508

theorem derivative_sqrt (x : ℝ) (h : x > 0) :
  deriv (fun x => Real.sqrt x) x = 1 / (2 * Real.sqrt x) := by sorry

end NUMINAMATH_CALUDE_derivative_sqrt_l475_47508


namespace NUMINAMATH_CALUDE_largest_odd_integer_with_coprime_primes_l475_47550

theorem largest_odd_integer_with_coprime_primes : ∃ (n : ℕ), 
  n = 105 ∧ 
  n % 2 = 1 ∧
  (∀ k : ℕ, 1 < k → k < n → k % 2 = 1 → Nat.gcd k n = 1 → Nat.Prime k) ∧
  (∀ m : ℕ, m > n → m % 2 = 1 → 
    ∃ k : ℕ, 1 < k ∧ k < m ∧ k % 2 = 1 ∧ Nat.gcd k m = 1 ∧ ¬Nat.Prime k) :=
by sorry

end NUMINAMATH_CALUDE_largest_odd_integer_with_coprime_primes_l475_47550


namespace NUMINAMATH_CALUDE_alvin_wood_needed_l475_47543

/-- The number of wood pieces Alvin needs for his house -/
def total_wood_needed (friend_pieces brother_pieces more_pieces : ℕ) : ℕ :=
  friend_pieces + brother_pieces + more_pieces

/-- Theorem: Alvin needs 376 pieces of wood in total -/
theorem alvin_wood_needed :
  total_wood_needed 123 136 117 = 376 := by
  sorry

end NUMINAMATH_CALUDE_alvin_wood_needed_l475_47543


namespace NUMINAMATH_CALUDE_anie_work_schedule_l475_47524

/-- Represents Anie's work schedule and project details -/
structure WorkSchedule where
  normal_hours : ℝ        -- Normal work hours per day
  extra_hours : ℝ         -- Extra hours worked per day
  project_hours : ℝ       -- Total hours for the project
  days_to_finish : ℝ      -- Number of days to finish the project

/-- Calculates Anie's normal work schedule given the conditions -/
def calculate_normal_schedule (w : WorkSchedule) : Prop :=
  w.extra_hours = 5 ∧ 
  w.project_hours = 1500 ∧ 
  w.days_to_finish = 100 ∧
  w.normal_hours = 10

/-- Theorem stating that Anie's normal work schedule is 10 hours per day -/
theorem anie_work_schedule : 
  ∀ w : WorkSchedule, calculate_normal_schedule w → w.normal_hours = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_anie_work_schedule_l475_47524


namespace NUMINAMATH_CALUDE_high_school_language_study_l475_47520

theorem high_school_language_study (total_students : ℕ) 
  (spanish_min spanish_max french_min french_max : ℕ) :
  total_students = 2001 →
  spanish_min = 1601 →
  spanish_max = 1700 →
  french_min = 601 →
  french_max = 800 →
  let m := spanish_min + french_min - total_students
  let M := spanish_max + french_max - total_students
  M - m = 298 := by
sorry

end NUMINAMATH_CALUDE_high_school_language_study_l475_47520


namespace NUMINAMATH_CALUDE_average_rate_round_trip_l475_47529

/-- Calculates the average rate of a round trip given the distance, running speed, and swimming speed. -/
theorem average_rate_round_trip 
  (distance : ℝ) 
  (running_speed : ℝ) 
  (swimming_speed : ℝ) 
  (h1 : distance = 4) 
  (h2 : running_speed = 10) 
  (h3 : swimming_speed = 6) : 
  (2 * distance) / (distance / running_speed + distance / swimming_speed) / 60 = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_average_rate_round_trip_l475_47529


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l475_47551

/-- The hyperbola and parabola intersect at two points A and B -/
structure IntersectionPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- The common focus of the hyperbola and parabola -/
def CommonFocus : ℝ × ℝ := (1, 2)

/-- The hyperbola equation -/
def isOnHyperbola (a b : ℝ) (p : ℝ × ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (p.1^2 / a^2) - (p.2^2 / b^2) = 1

/-- The parabola equation -/
def isOnParabola (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

/-- Line AB passes through the common focus -/
def lineABThroughFocus (points : IntersectionPoints) : Prop :=
  ∃ (t : ℝ), (1 - t) * points.A.1 + t * points.B.1 = CommonFocus.1 ∧
             (1 - t) * points.A.2 + t * points.B.2 = CommonFocus.2

/-- Theorem: The length of the real axis of the hyperbola is 2√2 - 2 -/
theorem hyperbola_real_axis_length (a b : ℝ) (points : IntersectionPoints) :
  isOnHyperbola a b CommonFocus →
  isOnParabola CommonFocus →
  lineABThroughFocus points →
  2 * a = 2 * Real.sqrt 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l475_47551


namespace NUMINAMATH_CALUDE_shadow_relation_sets_l475_47541

def is_shadow_relation (A : Set ℝ) : Prop :=
  ∀ x ∈ A, (1 / x) ∈ A

def set_A : Set ℝ := {-1, 1}
def set_B : Set ℝ := {1/2, 2}
def set_C : Set ℝ := {x : ℝ | x^2 > 1}
def set_D : Set ℝ := {x : ℝ | x > 0}

theorem shadow_relation_sets :
  is_shadow_relation set_A ∧
  is_shadow_relation set_B ∧
  is_shadow_relation set_D ∧
  ¬is_shadow_relation set_C :=
sorry

end NUMINAMATH_CALUDE_shadow_relation_sets_l475_47541


namespace NUMINAMATH_CALUDE_triangle_circumcircle_l475_47525

/-- Given a triangle with sides defined by three linear equations, 
    prove that its circumscribed circle has the specified equation. -/
theorem triangle_circumcircle 
  (line1 : ℝ → ℝ → Prop) 
  (line2 : ℝ → ℝ → Prop)
  (line3 : ℝ → ℝ → Prop)
  (h1 : ∀ x y, line1 x y ↔ x - 3*y = 2)
  (h2 : ∀ x y, line2 x y ↔ 7*x - y = 34)
  (h3 : ∀ x y, line3 x y ↔ x + 2*y = -8) :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -2) ∧ 
    radius = 5 ∧
    (∀ x y, (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ 
      (x - 1)^2 + (y + 2)^2 = 25) :=
sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_l475_47525


namespace NUMINAMATH_CALUDE_lunch_break_duration_l475_47553

/-- Represents the painting rate of a person in house/hour -/
structure PaintingRate :=
  (rate : ℝ)

/-- Represents a day's painting session -/
structure PaintingDay :=
  (duration : ℝ)  -- in hours
  (totalRate : ℝ)  -- combined rate of painters
  (portionPainted : ℝ)  -- portion of house painted

def calculateLunchBreak (monday : PaintingDay) (tuesday : PaintingDay) (wednesday : PaintingDay) : ℝ :=
  sorry

theorem lunch_break_duration :
  let paula : PaintingRate := ⟨0.5⟩
  let helpers : PaintingRate := ⟨0.25⟩  -- Combined rate of two helpers
  let apprentice : PaintingRate := ⟨0⟩
  let monday : PaintingDay := ⟨9, paula.rate + helpers.rate + apprentice.rate, 0.6⟩
  let tuesday : PaintingDay := ⟨7, helpers.rate + apprentice.rate, 0.3⟩
  let wednesday : PaintingDay := ⟨1.2, paula.rate + apprentice.rate, 0.1⟩
  calculateLunchBreak monday tuesday wednesday = 1.4 :=
by sorry

end NUMINAMATH_CALUDE_lunch_break_duration_l475_47553


namespace NUMINAMATH_CALUDE_rectangle_construction_l475_47584

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Checks if four points form a rectangle -/
def isRectangle (r : Rectangle) : Prop := sorry

/-- Calculates the aspect ratio of a rectangle -/
def aspectRatio (r : Rectangle) : ℝ := sorry

/-- Checks if a point lies on a line segment between two other points -/
def onSegment (p q r : Point) : Prop := sorry

/-- Theorem: A rectangle with a given aspect ratio can be constructed
    given one point on each of its sides -/
theorem rectangle_construction
  (a : ℝ)
  (A B C D : Point)
  (h_a : a > 0) :
  ∃ (r : Rectangle),
    isRectangle r ∧
    aspectRatio r = a ∧
    onSegment r.P A r.Q ∧
    onSegment r.Q B r.R ∧
    onSegment r.R C r.S ∧
    onSegment r.S D r.P :=
by sorry

end NUMINAMATH_CALUDE_rectangle_construction_l475_47584


namespace NUMINAMATH_CALUDE_allocation_schemes_l475_47544

/-- The number of ways to allocate teachers to buses -/
def allocate_teachers (n : ℕ) (m : ℕ) : ℕ :=
  sorry

/-- There are 3 buses -/
def num_buses : ℕ := 3

/-- There are 5 teachers -/
def num_teachers : ℕ := 5

/-- Each bus must have at least one teacher -/
axiom at_least_one_teacher (b : ℕ) : b ≤ num_buses → b > 0

theorem allocation_schemes :
  allocate_teachers num_teachers num_buses = 150 :=
sorry

end NUMINAMATH_CALUDE_allocation_schemes_l475_47544


namespace NUMINAMATH_CALUDE_inverse_function_ln_l475_47561

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1)

noncomputable def g (x : ℝ) : ℝ := Real.exp x + 1

theorem inverse_function_ln (x : ℝ) (hx : x > 2) :
  Function.Injective f ∧
  Function.Surjective f ∧
  (∀ y, y > 0 → g y > 2) ∧
  (∀ y, y > 0 → f (g y) = y) ∧
  (∀ x, x > 2 → g (f x) = x) := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_ln_l475_47561


namespace NUMINAMATH_CALUDE_factory_sampling_is_systematic_l475_47579

/-- Represents a sampling method -/
inductive SamplingMethod
  | Stratified
  | SimpleRandom
  | Systematic
  | Other

/-- Represents a factory with a conveyor belt and sampling process -/
structure Factory where
  sampleInterval : ℕ  -- Time interval between samples in minutes
  sampleLocation : String  -- Description of the sample location

/-- Determines the sampling method based on the factory's sampling process -/
def determineSamplingMethod (f : Factory) : SamplingMethod :=
  sorry

/-- Theorem stating that the described sampling method is systematic sampling -/
theorem factory_sampling_is_systematic (f : Factory) 
  (h1 : f.sampleInterval = 10)
  (h2 : f.sampleLocation = "specific location on the conveyor belt") :
  determineSamplingMethod f = SamplingMethod.Systematic :=
sorry

end NUMINAMATH_CALUDE_factory_sampling_is_systematic_l475_47579


namespace NUMINAMATH_CALUDE_square_root_problem_l475_47539

theorem square_root_problem (y z x : ℝ) (hy : y > 0) (hx : x > 0) :
  y^z = (Real.sqrt 16)^3 → x^2 = y^z → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l475_47539


namespace NUMINAMATH_CALUDE_alice_spending_percentage_l475_47528

theorem alice_spending_percentage (alice_initial : ℝ) (bob_initial : ℝ) (alice_final : ℝ)
  (h1 : bob_initial = 0.9 * alice_initial)
  (h2 : alice_final = 0.9 * bob_initial) :
  (alice_initial - alice_final) / alice_initial = 0.19 :=
by sorry

end NUMINAMATH_CALUDE_alice_spending_percentage_l475_47528


namespace NUMINAMATH_CALUDE_joes_mens_haircuts_l475_47510

def women_haircut_time : ℕ := 50
def men_haircut_time : ℕ := 15
def kids_haircut_time : ℕ := 25
def num_women : ℕ := 3
def num_kids : ℕ := 3
def total_time : ℕ := 255

theorem joes_mens_haircuts :
  ∃ (num_men : ℕ),
    num_men * men_haircut_time +
    num_women * women_haircut_time +
    num_kids * kids_haircut_time = total_time ∧
    num_men = 2 := by
  sorry

end NUMINAMATH_CALUDE_joes_mens_haircuts_l475_47510


namespace NUMINAMATH_CALUDE_tulip_price_is_two_l475_47592

/-- Represents the price of a tulip in dollars -/
def tulip_price : ℝ := 2

/-- Represents the price of a rose in dollars -/
def rose_price : ℝ := 3

/-- Calculates the total revenue for the three days -/
def total_revenue (tulip_price : ℝ) : ℝ :=
  -- First day
  (30 * tulip_price + 20 * rose_price) +
  -- Second day
  (60 * tulip_price + 40 * rose_price) +
  -- Third day
  (6 * tulip_price + 16 * rose_price)

theorem tulip_price_is_two :
  total_revenue tulip_price = 420 :=
by sorry

end NUMINAMATH_CALUDE_tulip_price_is_two_l475_47592


namespace NUMINAMATH_CALUDE_sum_mod_thirteen_l475_47570

theorem sum_mod_thirteen : (9753 + 9754 + 9755 + 9756) % 13 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_thirteen_l475_47570


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_relation_l475_47583

/-- Prove that for an arithmetic sequence, R = 2n²d -/
theorem arithmetic_sequence_sum_relation 
  (a d n : ℝ) 
  (S₁ : ℝ := n / 2 * (2 * a + (n - 1) * d))
  (S₂ : ℝ := n * (2 * a + (2 * n - 1) * d))
  (S₃ : ℝ := 3 * n / 2 * (2 * a + (3 * n - 1) * d))
  (R : ℝ := S₃ - S₂ - S₁) :
  R = 2 * n^2 * d := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_relation_l475_47583


namespace NUMINAMATH_CALUDE_tims_golf_balls_l475_47545

-- Define the number of dozens Tim has
def tims_dozens : ℕ := 13

-- Define the number of items in a dozen
def items_per_dozen : ℕ := 12

-- Theorem to prove
theorem tims_golf_balls : tims_dozens * items_per_dozen = 156 := by
  sorry

end NUMINAMATH_CALUDE_tims_golf_balls_l475_47545


namespace NUMINAMATH_CALUDE_turns_for_both_buckets_l475_47555

/-- Represents the capacity of bucket Q -/
def capacity_Q : ℝ := 1

/-- Represents the capacity of bucket P -/
def capacity_P : ℝ := 3 * capacity_Q

/-- Represents the number of turns it takes bucket P to fill the drum -/
def turns_P : ℕ := 80

/-- Represents the capacity of the drum -/
def drum_capacity : ℝ := turns_P * capacity_P

/-- Represents the combined capacity of buckets P and Q -/
def combined_capacity : ℝ := capacity_P + capacity_Q

/-- 
Proves that the number of turns it takes for both buckets P and Q together 
to fill the drum is 60, given the conditions stated in the problem.
-/
theorem turns_for_both_buckets : 
  (drum_capacity / combined_capacity : ℝ) = 60 := by sorry

end NUMINAMATH_CALUDE_turns_for_both_buckets_l475_47555


namespace NUMINAMATH_CALUDE_remainder_4_100_div_9_l475_47565

theorem remainder_4_100_div_9 : (4^100) % 9 = 4 := by sorry

end NUMINAMATH_CALUDE_remainder_4_100_div_9_l475_47565


namespace NUMINAMATH_CALUDE_not_necessarily_parallel_lines_l475_47512

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (intersect_planes : Plane → Plane → Line → Prop)

-- State the theorem
theorem not_necessarily_parallel_lines 
  (α β : Plane) (m n : Line) 
  (h1 : α ≠ β) 
  (h2 : m ≠ n) 
  (h3 : parallel_line_plane m α) 
  (h4 : intersect_planes α β n) : 
  ¬ (∀ m n, parallel_lines m n) :=
sorry

end NUMINAMATH_CALUDE_not_necessarily_parallel_lines_l475_47512


namespace NUMINAMATH_CALUDE_inclination_angle_expression_l475_47596

theorem inclination_angle_expression (θ : Real) : 
  (2 : Real) * Real.tan θ = -1 → 
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_inclination_angle_expression_l475_47596


namespace NUMINAMATH_CALUDE_sams_walking_speed_l475_47530

/-- Proves that Sam's walking speed is equal to Fred's given the problem conditions -/
theorem sams_walking_speed (total_distance : ℝ) (fred_speed : ℝ) (sam_distance : ℝ) :
  total_distance = 50 →
  fred_speed = 5 →
  sam_distance = 25 →
  let fred_distance := total_distance - sam_distance
  let time := fred_distance / fred_speed
  let sam_speed := sam_distance / time
  sam_speed = fred_speed :=
by sorry

end NUMINAMATH_CALUDE_sams_walking_speed_l475_47530


namespace NUMINAMATH_CALUDE_sum_of_roots_l475_47513

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 15*a^2 + 25*a - 75 = 0)
  (hb : 8*b^3 - 60*b^2 - 310*b + 2675 = 0) :
  a + b = 15/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l475_47513


namespace NUMINAMATH_CALUDE_bottles_per_case_is_25_l475_47517

/-- A company produces bottles of water and packs them in cases. -/
structure WaterBottleCompany where
  /-- The number of bottles produced per day -/
  daily_production : ℕ
  /-- The number of cases required for daily production -/
  cases_required : ℕ

/-- Calculate the number of bottles per case -/
def bottles_per_case (company : WaterBottleCompany) : ℕ :=
  company.daily_production / company.cases_required

/-- Theorem stating that for a company producing 50,000 bottles per day
    and requiring 2,000 cases, the number of bottles per case is 25 -/
theorem bottles_per_case_is_25 
  (company : WaterBottleCompany) 
  (h1 : company.daily_production = 50000) 
  (h2 : company.cases_required = 2000) : 
  bottles_per_case company = 25 := by
  sorry

end NUMINAMATH_CALUDE_bottles_per_case_is_25_l475_47517


namespace NUMINAMATH_CALUDE_doctors_who_quit_correct_number_of_doctors_quit_l475_47563

theorem doctors_who_quit (initial_doctors : ℕ) (initial_nurses : ℕ) 
  (nurses_quit : ℕ) (final_total : ℕ) : ℕ :=
  let doctors_quit := initial_doctors + initial_nurses - nurses_quit - final_total
  doctors_quit

theorem correct_number_of_doctors_quit : 
  doctors_who_quit 11 18 2 22 = 5 := by sorry

end NUMINAMATH_CALUDE_doctors_who_quit_correct_number_of_doctors_quit_l475_47563


namespace NUMINAMATH_CALUDE_total_marks_calculation_l475_47533

theorem total_marks_calculation (num_candidates : ℕ) (average_mark : ℚ) :
  num_candidates = 120 →
  average_mark = 35 →
  (num_candidates : ℚ) * average_mark = 4200 := by
  sorry

end NUMINAMATH_CALUDE_total_marks_calculation_l475_47533


namespace NUMINAMATH_CALUDE_x_squared_plus_y_squared_l475_47573

theorem x_squared_plus_y_squared (x y : ℚ) 
  (h : 2002 * (x - 1)^2 + |x - 12*y + 1| = 0) : 
  x^2 + y^2 = 37 / 36 := by
sorry

end NUMINAMATH_CALUDE_x_squared_plus_y_squared_l475_47573


namespace NUMINAMATH_CALUDE_reflection_composition_l475_47505

theorem reflection_composition :
  let x_reflection : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, -1]
  let y_reflection : Matrix (Fin 2) (Fin 2) ℝ := !![-1, 0; 0, 1]
  x_reflection * y_reflection = !![-1, 0; 0, -1] := by
sorry

end NUMINAMATH_CALUDE_reflection_composition_l475_47505


namespace NUMINAMATH_CALUDE_paperboy_delivery_ways_l475_47566

/-- Represents the number of ways to deliver newspapers to n houses without missing four consecutive houses. -/
def delivery_ways : ℕ → ℕ
  | 0 => 1  -- base case: one way to deliver to zero houses
  | 1 => 2  -- base case: two ways to deliver to one house
  | 2 => 4  -- base case: four ways to deliver to two houses
  | 3 => 8  -- base case: eight ways to deliver to three houses
  | n + 4 => delivery_ways (n + 3) + delivery_ways (n + 2) + delivery_ways (n + 1) + delivery_ways n

/-- Theorem stating that there are 2872 ways for a paperboy to deliver newspapers to 12 houses without missing four consecutive houses. -/
theorem paperboy_delivery_ways :
  delivery_ways 12 = 2872 := by
  sorry

end NUMINAMATH_CALUDE_paperboy_delivery_ways_l475_47566


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_18_with_sqrt_between_30_and_30_2_l475_47590

theorem unique_integer_divisible_by_18_with_sqrt_between_30_and_30_2 :
  ∃! n : ℕ+, 18 ∣ n ∧ 30 < (n : ℝ).sqrt ∧ (n : ℝ).sqrt < 30.2 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_18_with_sqrt_between_30_and_30_2_l475_47590


namespace NUMINAMATH_CALUDE_absolute_value_sum_l475_47519

theorem absolute_value_sum (x p : ℝ) : 
  (|x - 2| = p) → (x > 2) → (x + p = 2*p + 2) := by sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l475_47519


namespace NUMINAMATH_CALUDE_problem_solution_l475_47531

theorem problem_solution (x y : ℝ) : (x + 3)^2 + Real.sqrt (2 - y) = 0 → (x + y)^2021 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l475_47531


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l475_47576

theorem gcd_digits_bound (a b : ℕ) : 
  10000 ≤ a ∧ a < 100000 →
  10000 ≤ b ∧ b < 100000 →
  100000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 1000000000 →
  Nat.gcd a b < 100 :=
by sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l475_47576


namespace NUMINAMATH_CALUDE_brand_comparison_l475_47580

/-- Distribution of timing errors for brand A -/
def dist_A : List (ℝ × ℝ) := [(-1, 0.1), (0, 0.8), (1, 0.1)]

/-- Distribution of timing errors for brand B -/
def dist_B : List (ℝ × ℝ) := [(-2, 0.1), (-1, 0.2), (0, 0.4), (1, 0.2), (2, 0.1)]

/-- Expected value of a discrete random variable -/
def expected_value (dist : List (ℝ × ℝ)) : ℝ :=
  (dist.map (fun (x, p) => x * p)).sum

/-- Variance of a discrete random variable -/
def variance (dist : List (ℝ × ℝ)) : ℝ :=
  (dist.map (fun (x, p) => x^2 * p)).sum - (expected_value dist)^2

/-- Theorem stating the properties of brands A and B -/
theorem brand_comparison :
  expected_value dist_A = 0 ∧
  expected_value dist_B = 0 ∧
  variance dist_A = 0.2 ∧
  variance dist_B = 1.2 ∧
  variance dist_A < variance dist_B := by
  sorry

#check brand_comparison

end NUMINAMATH_CALUDE_brand_comparison_l475_47580


namespace NUMINAMATH_CALUDE_cos_arcsin_three_fifths_l475_47571

theorem cos_arcsin_three_fifths : 
  Real.cos (Real.arcsin (3/5)) = 4/5 := by sorry

end NUMINAMATH_CALUDE_cos_arcsin_three_fifths_l475_47571


namespace NUMINAMATH_CALUDE_undefined_expression_l475_47574

theorem undefined_expression (y : ℝ) : 
  (y^2 - 10*y + 25 = 0) ↔ (y = 5) := by
  sorry

#check undefined_expression

end NUMINAMATH_CALUDE_undefined_expression_l475_47574


namespace NUMINAMATH_CALUDE_radio_cost_price_l475_47588

/-- Proves that the cost price of a radio is 1500 Rs. given the selling price and loss percentage --/
theorem radio_cost_price (selling_price : ℝ) (loss_percentage : ℝ) : 
  selling_price = 1110 → loss_percentage = 26 → 
  ∃ (cost_price : ℝ), cost_price = 1500 ∧ selling_price = cost_price * (1 - loss_percentage / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_radio_cost_price_l475_47588


namespace NUMINAMATH_CALUDE_number_percentage_equality_l475_47522

theorem number_percentage_equality (x : ℚ) : 
  (35 : ℚ) / 100 * x = (25 : ℚ) / 100 * 40 → x = 200 / 7 := by
  sorry

end NUMINAMATH_CALUDE_number_percentage_equality_l475_47522


namespace NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l475_47586

theorem cube_volume_from_space_diagonal (d : ℝ) (h : d = 6 * Real.sqrt 3) :
  ∃ (s : ℝ), s > 0 ∧ s * Real.sqrt 3 = d ∧ s^3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l475_47586


namespace NUMINAMATH_CALUDE_total_animals_count_l475_47542

def animal_count : ℕ → ℕ → ℕ → ℕ
| snakes, arctic_foxes, leopards =>
  let bee_eaters := 12 * leopards
  let cheetahs := snakes / 3
  let alligators := 2 * (arctic_foxes + leopards)
  snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators

theorem total_animals_count :
  animal_count 100 80 20 = 673 :=
by sorry

end NUMINAMATH_CALUDE_total_animals_count_l475_47542


namespace NUMINAMATH_CALUDE_five_digit_multiple_of_nine_l475_47535

theorem five_digit_multiple_of_nine :
  ∃ (n : ℕ), n = 56781 ∧ n % 9 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_five_digit_multiple_of_nine_l475_47535


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l475_47526

-- Define the operation ⊗
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + 3 + Real.sqrt (b + 3 + Real.sqrt (b + 3 + Real.sqrt (b + 3))))

-- Theorem statement
theorem bowtie_equation_solution (x : ℝ) :
  bowtie 3 x = 12 → x = 69 := by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l475_47526


namespace NUMINAMATH_CALUDE_rest_time_calculation_l475_47518

theorem rest_time_calculation (walking_rate : ℝ) (total_distance : ℝ) (total_time : ℝ) :
  walking_rate = 10 →
  total_distance = 50 →
  total_time = 328 →
  (∃ (rest_time : ℝ),
    rest_time * 4 = total_time - (total_distance / walking_rate * 60) ∧
    rest_time = 7) := by
  sorry

end NUMINAMATH_CALUDE_rest_time_calculation_l475_47518


namespace NUMINAMATH_CALUDE_complex_equation_proof_l475_47599

theorem complex_equation_proof (z : ℂ) (h : Complex.abs z = 1 + 3 * I - z) :
  ((1 + I)^2 * (3 + 4*I)) / (2 * z) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_proof_l475_47599


namespace NUMINAMATH_CALUDE_retired_faculty_surveys_l475_47549

/-- Given a total number of surveys and a ratio of surveys from different groups,
    calculate the number of surveys from the retired faculty. -/
theorem retired_faculty_surveys
  (total_surveys : ℕ)
  (retired_ratio : ℕ)
  (current_ratio : ℕ)
  (student_ratio : ℕ)
  (h1 : total_surveys = 300)
  (h2 : retired_ratio = 2)
  (h3 : current_ratio = 8)
  (h4 : student_ratio = 40) :
  (total_surveys * retired_ratio) / (retired_ratio + current_ratio + student_ratio) = 12 := by
  sorry

#check retired_faculty_surveys

end NUMINAMATH_CALUDE_retired_faculty_surveys_l475_47549


namespace NUMINAMATH_CALUDE_f_properties_l475_47575

noncomputable def f (x : ℝ) : ℝ := 2 * x / Real.log x

theorem f_properties :
  let e := Real.exp 1
  -- 1. f'(e^2) = 1/2
  (deriv f (e^2) = 1/2) ∧
  -- 2. f is monotonically decreasing on (0, 1) and (1, e)
  (∀ x y, 0 < x ∧ x < y ∧ y < 1 → f y < f x) ∧
  (∀ x y, 1 < x ∧ x < y ∧ y < e → f y < f x) ∧
  -- 3. For all x > 0, x ≠ 1, f(x) > 2 / ln(x) + 2√x
  (∀ x, x > 0 ∧ x ≠ 1 → f x > 2 / Real.log x + 2 * Real.sqrt x) :=
by sorry


end NUMINAMATH_CALUDE_f_properties_l475_47575


namespace NUMINAMATH_CALUDE_infinite_solutions_equation_l475_47509

theorem infinite_solutions_equation :
  ∃ (S : Set (ℕ × ℕ × ℕ)), 
    (∀ (x y z : ℕ), (x, y, z) ∈ S → 
      x > 2008 ∧ y > 2008 ∧ z > 2008 ∧ 
      x^2 + y^2 + z^2 - x*y*z + 10 = 0) ∧
    Set.Infinite S :=
sorry

end NUMINAMATH_CALUDE_infinite_solutions_equation_l475_47509


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l475_47577

theorem largest_n_satisfying_inequality :
  ∀ n : ℕ, (1/4 : ℚ) + n/8 + 1/8 < 1 ↔ n ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l475_47577


namespace NUMINAMATH_CALUDE_Z_three_seven_l475_47506

def Z (a b : ℝ) : ℝ := b + 15 * a - a^2

theorem Z_three_seven : Z 3 7 = 43 := by sorry

end NUMINAMATH_CALUDE_Z_three_seven_l475_47506


namespace NUMINAMATH_CALUDE_distance_is_130_km_l475_47547

/-- The distance between two vehicles at both 5 hours before and 5 hours after they pass each other -/
def distance_between_vehicles (speed1 speed2 : ℝ) : ℝ :=
  (speed1 + speed2) * 2.5

/-- Theorem stating that the distance between the vehicles is 130 km -/
theorem distance_is_130_km :
  distance_between_vehicles 37 15 = 130 := by sorry

end NUMINAMATH_CALUDE_distance_is_130_km_l475_47547


namespace NUMINAMATH_CALUDE_tank_capacity_l475_47556

theorem tank_capacity (initial_fraction : ℚ) (final_fraction : ℚ) (added_amount : ℚ) :
  initial_fraction = 5 / 8 →
  final_fraction = 19 / 24 →
  added_amount = 15 →
  ∃ (total_capacity : ℚ),
    initial_fraction * total_capacity + added_amount = final_fraction * total_capacity ∧
    total_capacity = 90 := by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l475_47556


namespace NUMINAMATH_CALUDE_circle_area_difference_radius_l475_47578

theorem circle_area_difference_radius 
  (r₁ : ℝ) (r₂ : ℝ) (r₃ : ℝ) 
  (h₁ : r₁ = 21) (h₂ : r₂ = 31) 
  (h₃ : π * r₃^2 = π * r₂^2 - π * r₁^2) : 
  r₃ = 2 * Real.sqrt 130 := by
sorry

end NUMINAMATH_CALUDE_circle_area_difference_radius_l475_47578


namespace NUMINAMATH_CALUDE_thirty_percent_greater_than_88_l475_47598

theorem thirty_percent_greater_than_88 (x : ℝ) : 
  x = 88 * (1 + 30 / 100) → x = 114.4 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_greater_than_88_l475_47598


namespace NUMINAMATH_CALUDE_hares_per_rabbit_l475_47523

theorem hares_per_rabbit (dog : Nat) (cats : Nat) (rabbits_per_cat : Nat) (total_animals : Nat) :
  dog = 1 →
  cats = 4 →
  rabbits_per_cat = 2 →
  total_animals = 37 →
  ∃ hares_per_rabbit : Nat, 
    total_animals = dog + cats + (cats * rabbits_per_cat) + (cats * rabbits_per_cat * hares_per_rabbit) ∧
    hares_per_rabbit = 3 := by
  sorry

end NUMINAMATH_CALUDE_hares_per_rabbit_l475_47523


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l475_47546

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A nonagon is a polygon with 9 sides -/
def nonagon_sides : ℕ := 9

theorem nonagon_diagonals : num_diagonals nonagon_sides = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l475_47546


namespace NUMINAMATH_CALUDE_male_salmon_count_l475_47562

theorem male_salmon_count (female_salmon : ℕ) (total_salmon : ℕ) 
  (h1 : female_salmon = 259378) 
  (h2 : total_salmon = 971639) : 
  total_salmon - female_salmon = 712261 := by
  sorry

end NUMINAMATH_CALUDE_male_salmon_count_l475_47562


namespace NUMINAMATH_CALUDE_photo_arrangement_count_photo_arrangement_count_is_correct_l475_47540

/-- The number of different ways to adjust the arrangement of students in a group photo. -/
theorem photo_arrangement_count : ℕ :=
  let total_students : ℕ := 10
  let initial_front_row : ℕ := 3
  let initial_back_row : ℕ := 7
  let students_to_move : ℕ := 2
  let final_front_row : ℕ := initial_front_row + students_to_move

  Nat.choose initial_back_row students_to_move *
  (Nat.factorial final_front_row / Nat.factorial (final_front_row - students_to_move))

/-- The correct number of arrangements is equal to C(7,2) * A(5,2) -/
theorem photo_arrangement_count_is_correct :
  photo_arrangement_count = Nat.choose 7 2 * (Nat.factorial 5 / Nat.factorial 3) := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangement_count_photo_arrangement_count_is_correct_l475_47540


namespace NUMINAMATH_CALUDE_proportion_not_recent_boarders_l475_47521

/-- Represents the proportion of passengers who boarded at a given dock -/
def boardingProportion : ℚ := 1/4

/-- Represents the proportion of departing passengers who boarded at the previous dock -/
def previousDockProportion : ℚ := 1/10

/-- Calculates the proportion of passengers who boarded at either of the two previous docks -/
def recentBoardersProportion : ℚ := 2 * boardingProportion - boardingProportion * previousDockProportion

/-- Theorem stating the proportion of passengers who did not board at either of the two previous docks -/
theorem proportion_not_recent_boarders :
  1 - recentBoardersProportion = 21/40 := by sorry

end NUMINAMATH_CALUDE_proportion_not_recent_boarders_l475_47521


namespace NUMINAMATH_CALUDE_monochromatic_isosceles_independent_of_coloring_l475_47502

/-- A regular polygon with 6n+1 sides -/
structure RegularPolygon (n : ℕ) where
  sides : ℕ
  is_regular : sides = 6 * n + 1

/-- A coloring of the vertices of a regular polygon -/
structure Coloring (n : ℕ) where
  polygon : RegularPolygon n
  red_vertices : ℕ
  valid_coloring : red_vertices ≤ polygon.sides

/-- An isosceles triangle in a regular polygon -/
structure IsoscelesTriangle (n : ℕ) where
  polygon : RegularPolygon n

/-- A monochromatic isosceles triangle (all vertices same color) -/
structure MonochromaticIsoscelesTriangle (n : ℕ) extends IsoscelesTriangle n where
  coloring : Coloring n

/-- The number of monochromatic isosceles triangles in a colored regular polygon -/
def num_monochromatic_isosceles_triangles (n : ℕ) (c : Coloring n) : ℕ := sorry

/-- The main theorem: the number of monochromatic isosceles triangles is independent of coloring -/
theorem monochromatic_isosceles_independent_of_coloring (n : ℕ) 
  (c1 c2 : Coloring n) (h : c1.red_vertices = c2.red_vertices) :
  num_monochromatic_isosceles_triangles n c1 = num_monochromatic_isosceles_triangles n c2 := by
  sorry

end NUMINAMATH_CALUDE_monochromatic_isosceles_independent_of_coloring_l475_47502


namespace NUMINAMATH_CALUDE_cone_base_diameter_l475_47532

/-- Represents a cone with given properties -/
structure Cone where
  surfaceArea : ℝ
  lateralSurfaceIsSemicircle : Prop

/-- Theorem stating that a cone with surface area 3π and lateral surface unfolding 
    into a semicircle has a base diameter of √6 -/
theorem cone_base_diameter (c : Cone) 
    (h1 : c.surfaceArea = 3 * Real.pi)
    (h2 : c.lateralSurfaceIsSemicircle) : 
    ∃ (d : ℝ), d = Real.sqrt 6 ∧ d = 2 * (Real.sqrt ((3 : ℝ) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_cone_base_diameter_l475_47532


namespace NUMINAMATH_CALUDE_area_circle_inscribed_square_l475_47501

/-- The area of a circle inscribed in a square with diagonal 10 meters is 12.5π square meters. -/
theorem area_circle_inscribed_square (d : ℝ) (A : ℝ) :
  d = 10 → A = π * (d / (2 * Real.sqrt 2))^2 → A = 12.5 * π := by
  sorry

end NUMINAMATH_CALUDE_area_circle_inscribed_square_l475_47501


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_smallest_perimeter_l475_47503

/-- A quadrilateral represented by four points in a 2D plane -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Predicate to check if a quadrilateral is cyclic -/
def isCyclic (q : Quadrilateral) : Prop := sorry

/-- Predicate to check if a quadrilateral is inscribed in another -/
def isInscribed (inner outer : Quadrilateral) : Prop := sorry

/-- Function to calculate the perimeter of a quadrilateral -/
def perimeter (q : Quadrilateral) : ℝ := sorry

/-- Theorem about the existence of inscribed quadrilaterals with smallest perimeter -/
theorem inscribed_quadrilateral_smallest_perimeter (ABCD : Quadrilateral) :
  (isCyclic ABCD → ∃ (S : Set Quadrilateral), 
    (∀ q ∈ S, isInscribed q ABCD) ∧ 
    (∀ q ∈ S, ∀ p, isInscribed p ABCD → perimeter q ≤ perimeter p) ∧
    (Set.Infinite S)) ∧
  (¬isCyclic ABCD → ¬∃ q : Quadrilateral, 
    isInscribed q ABCD ∧ 
    (∀ p, isInscribed p ABCD → perimeter q ≤ perimeter p) ∧
    (q.A ≠ q.B ∧ q.B ≠ q.C ∧ q.C ≠ q.D ∧ q.D ≠ q.A)) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_smallest_perimeter_l475_47503


namespace NUMINAMATH_CALUDE_perpendicular_lines_l475_47515

-- Define the coefficients of the two lines
def line1_coeff (a : ℝ) : ℝ × ℝ := (2, a)
def line2_coeff (a : ℝ) : ℝ × ℝ := (a, 2*a - 1)

-- Define the perpendicularity condition
def are_perpendicular (a : ℝ) : Prop :=
  (line1_coeff a).1 * (line2_coeff a).1 + (line1_coeff a).2 * (line2_coeff a).2 = 0

-- State the theorem
theorem perpendicular_lines (a : ℝ) :
  are_perpendicular a ↔ a = -1/2 ∨ a = 0 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l475_47515


namespace NUMINAMATH_CALUDE_space_diagonals_of_specific_polyhedron_l475_47558

structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ
  pentagonal_faces : ℕ

def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  let total_segments := (Q.vertices.choose 2)
  let face_diagonals := 2 * Q.quadrilateral_faces + 5 * Q.pentagonal_faces
  total_segments - Q.edges - face_diagonals

theorem space_diagonals_of_specific_polyhedron :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 70,
    faces := 40,
    triangular_faces := 20,
    quadrilateral_faces := 15,
    pentagonal_faces := 5
  }
  space_diagonals Q = 310 := by sorry

end NUMINAMATH_CALUDE_space_diagonals_of_specific_polyhedron_l475_47558


namespace NUMINAMATH_CALUDE_cupcake_packages_l475_47568

/-- Given the initial number of cupcakes, the number eaten, and the number of cupcakes per package,
    calculate the number of full packages that can be made. -/
def fullPackages (initial : ℕ) (eaten : ℕ) (perPackage : ℕ) : ℕ :=
  (initial - eaten) / perPackage

/-- Theorem stating that with 60 initial cupcakes, 22 eaten, and 10 cupcakes per package,
    the number of full packages is 3. -/
theorem cupcake_packages : fullPackages 60 22 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_packages_l475_47568


namespace NUMINAMATH_CALUDE_daniel_speed_l475_47516

/-- The speeds of runners in a marathon preparation --/
def marathon_speeds (eugene_speed : ℚ) : ℚ × ℚ × ℚ × ℚ :=
  let brianna_speed := (3 / 4) * eugene_speed
  let katie_speed := (4 / 3) * brianna_speed
  let daniel_speed := (5 / 6) * katie_speed
  (eugene_speed, brianna_speed, katie_speed, daniel_speed)

/-- Theorem stating Daniel's speed given Eugene's speed --/
theorem daniel_speed (eugene_speed : ℚ) : 
  (marathon_speeds eugene_speed).2.2.2 = 25 / 6 :=
by
  sorry

#eval marathon_speeds 5

end NUMINAMATH_CALUDE_daniel_speed_l475_47516


namespace NUMINAMATH_CALUDE_xy_squared_sum_l475_47534

theorem xy_squared_sum (x y : ℝ) (h1 : x + y = 3) (h2 : x * y = 2) :
  x^2 * y + x * y^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_xy_squared_sum_l475_47534


namespace NUMINAMATH_CALUDE_area_ADBC_l475_47595

/-- Given a triangle ABC in the xy-plane where:
    A is at the origin (0, 0)
    B lies on the positive x-axis
    C is in the upper right quadrant
    ∠A = 30°, ∠B = 60°, ∠C = 90°
    Length BC = 1
    D is the intersection of the angle bisector of ∠C with the y-axis

    The area of quadrilateral ADBC is (5√3 + 9) / 4 -/
theorem area_ADBC (A B C D : ℝ × ℝ) : 
  A = (0, 0) →
  B.1 > 0 ∧ B.2 = 0 →
  C.1 > 0 ∧ C.2 > 0 →
  Real.cos (π/6) * (C.1 - A.1) = Real.sin (π/6) * (C.2 - A.2) →
  Real.cos (π/3) * (C.1 - B.1) = Real.sin (π/3) * (C.2 - B.2) →
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 1 →
  D.1 = 0 →
  (C.2 - D.2) / (C.1 - D.1) = (C.2 - A.2) / (C.1 - A.1) →
  let area := abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2 +
               abs ((C.1 - A.1) * (D.2 - A.2) - (D.1 - A.1) * (C.2 - A.2)) / 2
  area = (5 * Real.sqrt 3 + 9) / 4 := by
  sorry


end NUMINAMATH_CALUDE_area_ADBC_l475_47595


namespace NUMINAMATH_CALUDE_divisible_by_42_l475_47581

theorem divisible_by_42 (a : ℤ) : ∃ k : ℤ, a^7 - a = 42 * k := by sorry

end NUMINAMATH_CALUDE_divisible_by_42_l475_47581


namespace NUMINAMATH_CALUDE_twirly_tea_cups_capacity_l475_47564

/-- The 'Twirly Tea Cups' ride capacity problem -/
theorem twirly_tea_cups_capacity 
  (people_per_teacup : ℕ) 
  (number_of_teacups : ℕ) 
  (h1 : people_per_teacup = 9)
  (h2 : number_of_teacups = 7) : 
  people_per_teacup * number_of_teacups = 63 := by
  sorry

end NUMINAMATH_CALUDE_twirly_tea_cups_capacity_l475_47564


namespace NUMINAMATH_CALUDE_trenton_earnings_goal_l475_47552

/-- Trenton's weekly earnings calculation --/
def weekly_earnings (base_pay : ℝ) (commission_rate : ℝ) (sales : ℝ) : ℝ :=
  base_pay + commission_rate * sales

theorem trenton_earnings_goal :
  let base_pay : ℝ := 190
  let commission_rate : ℝ := 0.04
  let min_sales : ℝ := 7750
  let goal : ℝ := 500
  weekly_earnings base_pay commission_rate min_sales = goal := by
sorry

end NUMINAMATH_CALUDE_trenton_earnings_goal_l475_47552


namespace NUMINAMATH_CALUDE_ant_movement_l475_47527

-- Define the grid
structure Grid :=
  (has_black_pairs : Bool)
  (has_white_pairs : Bool)

-- Define the ant's position
inductive Position
  | Black
  | White

-- Define a single move
def move (p : Position) : Position :=
  match p with
  | Position.Black => Position.White
  | Position.White => Position.Black

-- Define the number of moves
def num_moves : Nat := 4

-- Define the function to count black finishing squares
def count_black_finish (g : Grid) : Nat :=
  sorry

-- Theorem statement
theorem ant_movement (g : Grid) :
  g.has_black_pairs = true →
  g.has_white_pairs = true →
  count_black_finish g = 6 :=
sorry

end NUMINAMATH_CALUDE_ant_movement_l475_47527


namespace NUMINAMATH_CALUDE_discriminant_of_specific_quadratic_l475_47587

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of 5x^2 - 9x + 1 is 61 -/
theorem discriminant_of_specific_quadratic : discriminant 5 (-9) 1 = 61 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_specific_quadratic_l475_47587


namespace NUMINAMATH_CALUDE_hotdog_competition_ratio_l475_47591

/-- Hotdog eating competition rates and ratios -/
theorem hotdog_competition_ratio :
  let first_rate := 10 -- hot dogs per minute
  let second_rate := 3 * first_rate
  let third_rate := 300 / 5 -- 300 hot dogs in 5 minutes
  third_rate / second_rate = 2 := by
  sorry

end NUMINAMATH_CALUDE_hotdog_competition_ratio_l475_47591


namespace NUMINAMATH_CALUDE_vasims_share_l475_47504

/-- Proves that Vasim's share is 1500 given the specified conditions -/
theorem vasims_share (total : ℕ) (faruk vasim ranjith : ℕ) : 
  faruk + vasim + ranjith = total →
  3 * faruk = 3 * vasim →
  3 * faruk = 3 * vasim ∧ 3 * vasim = 7 * ranjith →
  ranjith - faruk = 2000 →
  vasim = 1500 := by
sorry

end NUMINAMATH_CALUDE_vasims_share_l475_47504
