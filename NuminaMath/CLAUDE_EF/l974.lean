import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_decreasing_min_value_of_a_l974_97430

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + x

-- Part 1: Monotonicity of g
theorem g_monotone_decreasing (a : ℝ) :
  (∃ (t : ℝ), t - 1 = (1 - a) * (t - 1) ∧ g a 1 = 1 ∧ g a t = 2) →
  ∀ x ∈ Set.Ioo 0 2, StrictMonoOn (g a) (Set.Ioo 0 2) := by sorry

-- Part 2: Minimum value of a
theorem min_value_of_a :
  ∃ a_min : ℝ, a_min = 2 - 4 * Real.log 2 ∧
  (∀ a : ℝ, (∀ x ∈ Set.Ioo 0 (1/2), f a x ≠ 0) → a ≥ a_min) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_decreasing_min_value_of_a_l974_97430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_projective_transformation_circle_to_infinity_l974_97437

-- Define a circle on a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line on a plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- ax + by + c = 0

-- Define a projective transformation
structure ProjectiveTransformation where
  matrix : Matrix (Fin 3) (Fin 3) ℝ
  det_nonzero : Matrix.det matrix ≠ 0

-- Define the line at infinity
def LineAtInfinity : Line := ⟨0, 0, 1⟩

-- Function to check if a line intersects a circle
def intersects (l : Line) (c : Circle) : Prop := sorry

-- Function to check if a projective transformation maps a circle to a circle
def mapsToCircle (t : ProjectiveTransformation) (c1 c2 : Circle) : Prop := sorry

-- Function to check if a projective transformation maps a line to the line at infinity
def mapsToLineAtInfinity (t : ProjectiveTransformation) (l : Line) : Prop := sorry

-- The main theorem
theorem exists_projective_transformation_circle_to_infinity 
  (c : Circle) (l : Line) (h : ¬intersects l c) : 
  ∃ (t : ProjectiveTransformation) (c' : Circle), 
    mapsToCircle t c c' ∧ mapsToLineAtInfinity t l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_projective_transformation_circle_to_infinity_l974_97437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_coplanar_implies_not_collinear_l974_97405

-- Define a type for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a predicate for collinearity of three points
def collinear (p q r : Point3D) : Prop :=
  ∃ (t : ℝ), (q.x - p.x, q.y - p.y, q.z - p.z) = t • (r.x - p.x, r.y - p.y, r.z - p.z)

-- Define a predicate for four points being non-coplanar
def nonCoplanar (p q r s : Point3D) : Prop :=
  ¬∃ (a b c : ℝ), a * (q.x - p.x) + b * (r.x - p.x) + c * (s.x - p.x) = 0 ∧
                   a * (q.y - p.y) + b * (r.y - p.y) + c * (s.y - p.y) = 0 ∧
                   a * (q.z - p.z) + b * (r.z - p.z) + c * (s.z - p.z) = 0 ∧
                   (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)

-- Theorem statement
theorem non_coplanar_implies_not_collinear (p q r s : Point3D) :
  nonCoplanar p q r s → ¬(collinear p q r ∨ collinear p q s ∨ collinear p r s ∨ collinear q r s) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_coplanar_implies_not_collinear_l974_97405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_l974_97499

def is_valid_permutation (a b c d : ℕ) : Prop :=
  Multiset.ofList [a, b, c, d] = Multiset.ofList [2, 3, 4, 5]

def product_sum (a b c d : ℕ) : ℕ :=
  a * b + b * c + c * d + d * a

theorem max_product_sum :
  ∀ a b c d : ℕ, is_valid_permutation a b c d →
    product_sum a b c d ≤ 48 ∧
    ∃ w x y z : ℕ, is_valid_permutation w x y z ∧ product_sum w x y z = 48 :=
by
  sorry

#check max_product_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_l974_97499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_traffic_distance_l974_97478

/-- Represents the speed of a person -/
structure Speed where
  normal : ℝ
  peak : ℝ

/-- Represents a person's journey -/
structure Journey where
  start_time : ℝ
  speed : Speed

/-- The problem setup -/
structure CityTraffic where
  d : ℝ            -- distance between points A and B
  person_a : Journey
  person_b : Journey
  peak_start : ℝ   -- start time of peak traffic
  peak_end : ℝ     -- end time of peak traffic
  first_meet : ℝ   -- distance from A of first meeting point
  mid_meet : ℝ     -- time person A starts for midpoint meeting
  early_meet : ℝ   -- distance from A of early meeting point
  early_start : ℝ   -- time person B starts for early meeting

/-- The theorem to prove -/
theorem city_traffic_distance (ct : CityTraffic) : 
  (ct.person_a.start_time = 6 + 50/60) →
  (ct.person_b.start_time = 6 + 50/60) →
  (ct.peak_start = 7) →
  (ct.peak_end = 8) →
  (ct.person_a.speed.peak = ct.person_a.speed.normal / 2) →
  (ct.person_b.speed.peak = ct.person_b.speed.normal / 2) →
  (ct.first_meet = 24) →
  (ct.mid_meet = ct.person_a.start_time + 20/60) →
  (ct.early_meet = 20) →
  (ct.early_start = ct.person_b.start_time - 20/60) →
  ct.d = 42 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_traffic_distance_l974_97478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_nine_max_profit_value_l974_97406

-- Define the revenue function
noncomputable def g (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then 13.5 - (1/30) * x^2
  else if x > 10 then 168/x - 2000/(3*x^2)
  else 0

-- Define the profit function
noncomputable def profit (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then 8.1*x - (1/30)*x^3 - 20
  else if x > 10 then 148 - 2*(1000/(3*x) + 2.7*x)
  else 0

-- Theorem stating the maximum profit occurs at x = 9
theorem max_profit_at_nine :
  ∀ x : ℝ, x > 0 → profit x ≤ profit 9 := by
  sorry

-- Theorem stating the maximum profit value
theorem max_profit_value :
  profit 9 = 28.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_nine_max_profit_value_l974_97406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_theorem_unique_percentage_increase_l974_97469

/-- Given an initial salary, a percentage increase, and a final salary after a 5% decrease,
    this function calculates the final salary. -/
noncomputable def finalSalary (initialSalary : ℝ) (percentageIncrease : ℝ) : ℝ :=
  initialSalary * (1 + percentageIncrease / 100) * (1 - 5 / 100)

/-- Theorem stating that if the initial salary of 3000 is increased by 10% and then
    decreased by 5%, the final salary will be 3135. -/
theorem salary_increase_theorem :
  finalSalary 3000 10 = 3135 := by
  sorry

/-- Theorem proving that 10% is the unique percentage increase that results in
    a final salary of 3135, given an initial salary of 3000 and a 5% decrease. -/
theorem unique_percentage_increase :
  ∀ x : ℝ, finalSalary 3000 x = 3135 ↔ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_theorem_unique_percentage_increase_l974_97469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_period_trig_identity_l974_97484

-- Theorem 1: Periodicity of tangent function
theorem tan_period (x : ℝ) : Real.tan (x + π) = Real.tan x := by sorry

-- Theorem 2: Trigonometric identity
theorem trig_identity (α : ℝ) : Real.sin (-α) / Real.tan (2 * π - α) = Real.cos α := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_period_trig_identity_l974_97484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_of_five_l974_97491

open Nat

-- Define the number of divisors function
noncomputable def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

-- Define the statement
theorem greatest_power_of_five (n : ℕ) :
  n > 0 ∧ num_divisors n = 120 ∧ num_divisors (5 * n) = 144 →
  ∃ k : ℕ, k = 4 ∧ 5^k ∣ n ∧ ∀ m : ℕ, 5^m ∣ n → m ≤ k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_of_five_l974_97491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_line_l974_97401

/-- The minimum distance between a point on the circle (x-1)² + y² = 1 and a point on the line 2x - y + 2 = 0 is (4√5)/5 - 1 -/
theorem min_distance_circle_line : 
  let circle := {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 1}
  let line := {q : ℝ × ℝ | 2 * q.1 - q.2 + 2 = 0}
  ∃ (min_dist : ℝ), min_dist = (4 * Real.sqrt 5) / 5 - 1 ∧
    ∀ (p q : ℝ × ℝ), p ∈ circle → q ∈ line → 
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ min_dist :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_line_l974_97401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_condition_for_x_greater_than_e_l974_97479

theorem necessary_but_not_sufficient_condition_for_x_greater_than_e (x : ℝ) :
  (x > Real.exp 1 → x > 1) ∧ ¬(x > 1 → x > Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_condition_for_x_greater_than_e_l974_97479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_parallelepiped_lateral_surface_area_l974_97448

/-- The lateral surface area of a rectangular parallelepiped -/
noncomputable def lateralSurfaceArea (d α β : ℝ) : ℝ :=
  2 * Real.sqrt 2 * d^2 * Real.sin α * Real.tan β * Real.sin (α + Real.pi/4)

/-- Theorem: The lateral surface area of a rectangular parallelepiped
    given the diagonal of the base (d), the angle between the diagonal
    and one side of the base (α), and the angle between a plane through
    one side and the opposite side of the upper base and the plane of
    the base (β) -/
theorem rectangular_parallelepiped_lateral_surface_area
  (d α β : ℝ) (h_d : d > 0) (h_α : 0 < α ∧ α < Real.pi/2) (h_β : 0 < β ∧ β < Real.pi/2) :
  lateralSurfaceArea d α β =
    2 * Real.sqrt 2 * d^2 * Real.sin α * Real.tan β * Real.sin (α + Real.pi/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_parallelepiped_lateral_surface_area_l974_97448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l974_97400

def sequence_a : ℕ → ℚ
  | 0 => 1
  | n + 1 => sequence_a n / (sequence_a n + 2)

def sequence_b (lambda : ℚ) : ℕ → ℚ
  | 0 => -lambda
  | n + 1 => (n + 1 - lambda) * (1 / sequence_a n + 1)

theorem lambda_range (lambda : ℚ) :
  (∀ n : ℕ, sequence_b lambda n ≤ sequence_b lambda (n + 1)) →
  lambda < 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l974_97400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_uniform_distribution_l974_97467

/-- A random variable uniformly distributed on the interval [2,8] -/
noncomputable def X : ℝ → ℝ := sorry

/-- The probability density function of X -/
noncomputable def f (x : ℝ) : ℝ := 
  if 2 ≤ x ∧ x ≤ 8 then 1/6 else 0

/-- The expected value of X -/
noncomputable def E (X : ℝ → ℝ) : ℝ := ∫ x in Set.Icc 2 8, x * f x

theorem expected_value_uniform_distribution :
  E X = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_uniform_distribution_l974_97467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_red_yellow_flash_at_60_l974_97487

/-- Red light flash interval in seconds -/
def red_interval : ℕ := 3

/-- Yellow light flash interval in seconds -/
def yellow_interval : ℕ := 4

/-- Green light flash interval in seconds -/
def green_interval : ℕ := 8

/-- Checks if a given time is when only red and yellow lights flash -/
def only_red_yellow_flash (t : ℕ) : Bool :=
  t % red_interval = 0 && t % yellow_interval = 0 && t % green_interval ≠ 0

/-- Counts the number of times only red and yellow lights have flashed up to time t -/
def count_red_yellow_flashes (t : ℕ) : ℕ :=
  (List.range (t + 1)).filter only_red_yellow_flash |>.length

/-- The theorem stating that the third occurrence of only red and yellow lights flashing together happens at 60 seconds -/
theorem third_red_yellow_flash_at_60 : count_red_yellow_flashes 60 = 3 ∧ 
  ∀ t < 60, count_red_yellow_flashes t < 3 := by
  sorry

#eval count_red_yellow_flashes 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_red_yellow_flash_at_60_l974_97487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_divides_xy_l974_97480

theorem seven_divides_xy (x y z : ℕ+) 
  (h1 : Nat.Coprime x y) 
  (h2 : x^2 + y^2 = z^4) : 
  7 ∣ (x * y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_divides_xy_l974_97480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_virus_radius_l974_97422

/-- Given a virus with a diameter of 120 nanometers, prove that its radius in meters (in scientific notation) is 6 × 10^-8 -/
theorem virus_radius (diameter : ℝ) (nanometer_to_meter : ℝ) : 
  diameter = 120 → nanometer_to_meter = 10^(-9 : ℤ) → 
  (diameter / 2) * nanometer_to_meter = 6 * 10^(-8 : ℤ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_virus_radius_l974_97422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cameron_work_days_l974_97471

/-- The number of days Cameron worked alone before Sandra joined him --/
noncomputable def days_cameron_worked_alone : ℝ := 9

/-- Cameron's work rate (task per day) --/
noncomputable def cameron_rate : ℝ := 1 / 18

/-- Combined work rate of Cameron and Sandra (task per day) --/
noncomputable def combined_rate : ℝ := 1 / 7

/-- The time it took for both to complete the remaining task after Cameron worked alone --/
noncomputable def remaining_time : ℝ := 3.5

theorem cameron_work_days :
  days_cameron_worked_alone * cameron_rate + remaining_time * combined_rate = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cameron_work_days_l974_97471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l974_97427

/-- Compound interest calculation for semi-annual compounding -/
noncomputable def compound_interest (principal : ℝ) (annual_rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + annual_rate / 2) ^ (2 * years)

/-- The problem statement -/
theorem investment_growth :
  let principal := 2500
  let annual_rate := 0.045
  let years := 21
  let final_amount := compound_interest principal annual_rate years
  ∃ ε > 0, |final_amount - 5077.14| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l974_97427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_through_foci_l974_97460

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis

/-- Represents a diameter of an ellipse -/
structure Diameter where
  start : Point
  finish : Point  -- Using 'finish' instead of 'end' as 'end' is a reserved keyword

/-- Represents a rectangular hyperbola -/
structure RectangularHyperbola where
  points : List Point

/-- Returns the foci of an ellipse -/
noncomputable def Ellipse.foci (e : Ellipse) : (Point × Point) :=
  ({ x := e.center.x + Real.sqrt (e.a^2 - e.b^2), y := e.center.y },
   { x := e.center.x - Real.sqrt (e.a^2 - e.b^2), y := e.center.y })

/-- Checks if two diameters are perpendicular -/
def arePerpendicular (d1 d2 : Diameter) : Prop := sorry

/-- Checks if a diameter is conjugate to another diameter -/
def isConjugate (e : Ellipse) (d1 d2 : Diameter) : Prop := sorry

/-- Checks if a point lies on a rectangular hyperbola -/
def isOnHyperbola (p : Point) (h : RectangularHyperbola) : Prop := sorry

/-- Main theorem -/
theorem hyperbola_through_foci (e : Ellipse) (aob cod : Diameter) (apobp cpodp : Diameter)
  (h : RectangularHyperbola) :
  arePerpendicular aob cod →
  isConjugate e aob apobp →
  isConjugate e cod cpodp →
  isOnHyperbola apobp.start h →
  isOnHyperbola apobp.finish h →
  isOnHyperbola cpodp.start h →
  isOnHyperbola cpodp.finish h →
  isOnHyperbola (Ellipse.foci e).1 h ∧ isOnHyperbola (Ellipse.foci e).2 h := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_through_foci_l974_97460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_onto_same_vector_l974_97498

noncomputable def v1 : ℝ × ℝ := (5, 2)
noncomputable def v2 : ℝ × ℝ := (2, 6)
noncomputable def q : ℝ × ℝ := (104/25, 78/25)

theorem projection_onto_same_vector (u : ℝ × ℝ) : 
  ∃ (k1 k2 : ℝ), 
    k1 • u = q ∧ 
    k2 • u = q ∧ 
    (v1.1 - q.1) * u.1 + (v1.2 - q.2) * u.2 = 0 ∧
    (v2.1 - q.1) * u.1 + (v2.2 - q.2) * u.2 = 0 :=
by sorry

#check projection_onto_same_vector

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_onto_same_vector_l974_97498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_root_over_seventh_root_of_eleven_l974_97443

theorem fifth_root_over_seventh_root_of_eleven : 
  (11 : ℝ) ^ (1/5 : ℝ) / (11 : ℝ) ^ (1/7 : ℝ) = (11 : ℝ) ^ (2/35 : ℝ) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_root_over_seventh_root_of_eleven_l974_97443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l974_97494

noncomputable def f (x : ℝ) : ℝ := (x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7))

theorem inequality_solution_set :
  ∀ x : ℝ, f x > 0 ↔ (x > 4 ∧ x < 5) ∨ (x > 5 ∧ x < 6) ∨ (x > 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l974_97494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_remaining_speed_l974_97433

/-- A journey with a specific speed profile -/
structure Journey where
  totalDistance : ℝ
  totalTime : ℝ
  initialSpeed : ℝ
  initialDistanceFraction : ℝ
  initialTimeFraction : ℝ

/-- Calculates the required speed for the remaining part of the journey -/
noncomputable def remainingSpeed (j : Journey) : ℝ :=
  (j.totalDistance * (1 - j.initialDistanceFraction)) /
  (j.totalTime * (1 - j.initialTimeFraction))

/-- Theorem stating the required speed for the remaining part of the journey -/
theorem journey_remaining_speed (j : Journey)
  (h1 : j.initialSpeed = 50)
  (h2 : j.initialDistanceFraction = 2/3)
  (h3 : j.initialTimeFraction = 1/3)
  (h4 : j.initialSpeed * j.totalTime * j.initialTimeFraction = j.totalDistance * j.initialDistanceFraction) :
  remainingSpeed j = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_remaining_speed_l974_97433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_S_l974_97441

/-- The number of positive divisors of a positive integer n -/
def d (n : ℕ+) : ℕ := (Nat.divisors n.val).card

/-- The set of positive integers n that satisfy n + d(n) = d(n)^2 -/
def S : Set ℕ+ := {n | n.val + d n = (d n)^2}

theorem characterization_of_S : S = {2, 56, 132, 1260} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_S_l974_97441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repaired_shoes_last_one_year_l974_97463

/-- The cost to repair used shoes -/
noncomputable def repair_cost : ℝ := 13.50

/-- The cost of new shoes -/
noncomputable def new_shoes_cost : ℝ := 32.00

/-- The duration new shoes last in years -/
noncomputable def new_shoes_duration : ℝ := 2

/-- The percentage increase in average cost per year of new shoes compared to repaired shoes -/
noncomputable def cost_increase_percentage : ℝ := 18.52

/-- The duration repaired shoes last in years -/
noncomputable def repaired_shoes_duration : ℝ := repair_cost / (new_shoes_cost / new_shoes_duration / (1 + cost_increase_percentage / 100))

theorem repaired_shoes_last_one_year :
  ∃ (ε : ℝ), ε > 0 ∧ |repaired_shoes_duration - 1| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repaired_shoes_last_one_year_l974_97463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_l974_97442

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  sine_law : a / (Real.sin A) = b / (Real.sin B)

noncomputable def line1 (t : Triangle) (x y : ℝ) : ℝ := 
  Real.sin t.A * x - t.a * y - t.c

noncomputable def line2 (t : Triangle) (x y : ℝ) : ℝ := 
  t.b * x + Real.sin t.B * y + Real.sin t.C

theorem lines_perpendicular (t : Triangle) : 
  (Real.sin t.A / t.a) * (-t.b / Real.sin t.B) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_l974_97442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triangle_property_solutions_l974_97495

/-- A complex number z satisfies the triangle property if 0, z, and z^3 form
    the three distinct vertices of an equilateral triangle in the complex plane. -/
def satisfies_triangle_property (z : ℂ) : Prop :=
  z ≠ 0 ∧
  z ≠ z^3 ∧
  Complex.abs z = Complex.abs (z^3 - z) ∧
  Complex.abs z = Complex.abs z^3

/-- The set of complex numbers satisfying the triangle property -/
def triangle_property_set : Set ℂ :=
  {z : ℂ | satisfies_triangle_property z}

/-- There are exactly 4 complex numbers satisfying the triangle property -/
theorem count_triangle_property_solutions :
  ∃ (s : Finset ℂ), s.card = 4 ∧ ∀ z : ℂ, z ∈ s ↔ satisfies_triangle_property z := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triangle_property_solutions_l974_97495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_min_modulus_l974_97420

-- Define the complex quadratic equation
def quadratic_equation (z : ℂ) (x : ℝ) : ℂ := x^2 - z*x + (4 : ℂ) + (3 : ℂ)*Complex.I

-- Define the condition for real roots
def has_real_roots (z : ℂ) : Prop :=
  ∃ x₁ x₂ : ℝ, quadratic_equation z x₁ = 0 ∧ quadratic_equation z x₂ = 0

-- Define the minimum modulus value
noncomputable def min_modulus : ℝ := Real.sqrt 170 / 3

-- Define the optimal complex number
noncomputable def optimal_z : ℂ := (9 * Real.sqrt 5 / 5) + (3 * Real.sqrt 5 / 5) * Complex.I

-- State the theorem
theorem quadratic_min_modulus :
  ∀ z : ℂ, has_real_roots z → Complex.abs z ≥ min_modulus ∧
  (Complex.abs z = min_modulus ↔ z = optimal_z ∨ z = -optimal_z) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_min_modulus_l974_97420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_selected_number_l974_97457

def random_table : List (List Nat) := [
  [1622, 7794, 3949, 5443, 5482, 1737, 9323, 7887, 3520, 9643],
  [8626, 3491, 6484, 4217, 5331, 5724, 5506, 8877, 0474, 4767]
]

def total_products : Nat := 500
def sample_size : Nat := 20
def start_row : Nat := 0
def start_col : Nat := 5

def is_valid_number (n : Nat) : Bool :=
  n > 0 && n <= total_products

def find_nth_valid_number (n : Nat) : Option Nat := do
  let flattened_table := random_table.join
  let valid_numbers := flattened_table.filter is_valid_number
  valid_numbers[n - 1]?

theorem fourth_selected_number :
  find_nth_valid_number 4 = some 173 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_selected_number_l974_97457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_value_from_point_l974_97434

theorem angle_value_from_point (θ : Real) (P : ℝ × ℝ) : 
  P.1 = Real.sqrt 3 / 2 →
  P.2 = -1 / 2 →
  θ ∈ Set.Icc 0 (2 * Real.pi) →
  P ∈ {p : ℝ × ℝ | ∃ r, p.1 = r * Real.cos θ ∧ p.2 = r * Real.sin θ} →
  θ = 11 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_value_from_point_l974_97434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_division_equal_areas_grid_division_equal_areas_proof_l974_97408

/-- A point in a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A line passing through two points -/
structure GridLine where
  p1 : GridPoint
  p2 : GridPoint

/-- The area of a right triangle formed by a line and the grid edge -/
def triangleArea (line : GridLine) : ℚ :=
  (abs (line.p2.x - line.p1.x) * abs (line.p2.y - line.p1.y)) / 2

theorem grid_division_equal_areas (gridSize : ℕ) (centerPoint : GridPoint)
    (line1 line2 : GridLine) : Prop :=
  gridSize = 6 ∧
  centerPoint = ⟨3, 3⟩ ∧
  line1.p1 = centerPoint ∧
  line2.p1 = centerPoint ∧
  line1.p2 = ⟨0, 4⟩ ∧
  line2.p2 = ⟨6, 4⟩ →
  triangleArea line1 = (gridSize * gridSize : ℚ) / 3 ∧
  triangleArea line2 = (gridSize * gridSize : ℚ) / 3 ∧
  (gridSize * gridSize : ℚ) / 3 = gridSize * gridSize - triangleArea line1 - triangleArea line2

theorem grid_division_equal_areas_proof : grid_division_equal_areas 6 ⟨3, 3⟩ 
    ⟨⟨3, 3⟩, ⟨0, 4⟩⟩ ⟨⟨3, 3⟩, ⟨6, 4⟩⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_division_equal_areas_grid_division_equal_areas_proof_l974_97408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_a_prefers_scenario_b_l974_97462

/-- Represents the three cards in the game -/
inductive Card
  | Red3
  | Red4
  | Black5

/-- Represents the outcome of two draws -/
structure DrawOutcome where
  first : Card
  second : Card

/-- Defines the game with its rules -/
structure CardGame where
  outcomes : List DrawOutcome
  all_outcomes : ∀ c1 c2 : Card, DrawOutcome.mk c1 c2 ∈ outcomes

/-- Checks if both cards in a draw outcome are red (hearts) -/
def bothRed (outcome : DrawOutcome) : Bool :=
  match outcome with
  | ⟨Card.Red3, Card.Red3⟩ => true
  | ⟨Card.Red3, Card.Red4⟩ => true
  | ⟨Card.Red4, Card.Red3⟩ => true
  | ⟨Card.Red4, Card.Red4⟩ => true
  | _ => false

/-- Checks if both cards in a draw outcome have the same suit -/
def sameSuit (outcome : DrawOutcome) : Bool :=
  match outcome with
  | ⟨Card.Red3, Card.Red3⟩ => true
  | ⟨Card.Red3, Card.Red4⟩ => true
  | ⟨Card.Red4, Card.Red3⟩ => true
  | ⟨Card.Red4, Card.Red4⟩ => true
  | ⟨Card.Black5, Card.Black5⟩ => true
  | _ => false

/-- Checks if the sum of card numbers in a draw outcome is odd -/
def sumIsOdd (outcome : DrawOutcome) : Bool :=
  match outcome with
  | ⟨Card.Red3, Card.Red4⟩ => true
  | ⟨Card.Red4, Card.Red3⟩ => true
  | ⟨Card.Red3, Card.Black5⟩ => true
  | ⟨Card.Red4, Card.Black5⟩ => true
  | ⟨Card.Black5, Card.Red3⟩ => true
  | ⟨Card.Black5, Card.Red4⟩ => true
  | _ => false

/-- Calculates the probability of an event in the card game -/
def probability (game : CardGame) (event : DrawOutcome → Bool) : Rat :=
  (game.outcomes.filter event).length / game.outcomes.length

theorem player_a_prefers_scenario_b (game : CardGame) :
  probability game sumIsOdd > probability game sameSuit := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_a_prefers_scenario_b_l974_97462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_B_inv_l974_97426

theorem det_A_B_inv {n : Type*} [Fintype n] [DecidableEq n] 
  (A B : Matrix n n ℝ) 
  (h1 : Matrix.det A = 3) 
  (h2 : Matrix.det B = 8) : 
  Matrix.det (A * B⁻¹) = 3/8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_B_inv_l974_97426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_blue_is_three_twentieths_l974_97453

/-- A tile is blue if its number is congruent to 3 mod 7 -/
def isBlue (n : ℕ) : Prop := n % 7 = 3

/-- The total number of tiles in the box -/
def totalTiles : ℕ := 60

/-- The number of blue tiles in the box -/
noncomputable def blueTiles : ℕ := (Finset.range totalTiles).filter (fun n => n % 7 = 3) |>.card

/-- The probability of randomly selecting a blue tile -/
noncomputable def probabilityBlue : ℚ := blueTiles / totalTiles

theorem probability_blue_is_three_twentieths :
  probabilityBlue = 3 / 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_blue_is_three_twentieths_l974_97453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_rotation_surface_areas_l974_97428

/-- The curved surface area of a conical frustum formed by rotating a right triangle around one of its legs. -/
noncomputable def curved_surface_area (leg : ℝ) (hypotenuse : ℝ) : ℝ :=
  Real.pi * leg * Real.sqrt ((hypotenuse - leg)^2 + (Real.sqrt (hypotenuse^2 - leg^2))^2)

theorem right_triangle_rotation_surface_areas :
  let a : ℝ := 3
  let b : ℝ := 4
  let c : ℝ := 5
  (a^2 + b^2 = c^2) →
  (curved_surface_area a c = 6 * Real.pi * Real.sqrt 5) ∧
  (curved_surface_area b c = 4 * Real.pi * Real.sqrt 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_rotation_surface_areas_l974_97428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_condition_l974_97421

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if (sin A - sin C)(a + c) / b = sin A - sin B, then C = π/3 -/
theorem triangle_special_condition (a b c A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  (Real.sin A - Real.sin C) * (a + c) / b = Real.sin A - Real.sin B →
  C = Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_condition_l974_97421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l974_97468

theorem function_property (f : ℕ → ℕ) 
  (h1 : ∀ n, f (f n) + f n = 2 * n + 3) 
  (h2 : f 0 = 1) : 
  ∀ n, f n = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l974_97468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ones_digit_of_sum_of_powers_2013_l974_97447

/-- The sum of powers from 1 to n, each raised to the power k -/
def sum_of_powers (n : ℕ) (k : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (fun i => (i + 1) ^ k)

/-- The ones digit of a natural number -/
def ones_digit (n : ℕ) : ℕ :=
  n % 10

theorem ones_digit_of_sum_of_powers_2013 :
  ones_digit (sum_of_powers 2013 2013) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ones_digit_of_sum_of_powers_2013_l974_97447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l974_97482

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos x * (2 * Real.sqrt 3 * Real.sin x + Real.cos x) + 
  Real.sin (x + 3 * Real.pi / 4) * Real.cos (x - Real.pi / 4) + 1 / 2

noncomputable def g (x : ℝ) : ℝ :=
  f ((x - Real.pi / 6) * 2)

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x : ℝ), x > Real.pi / 12 ∧ x < Real.pi / 2 → f x > 0 ∧ f x ≤ 3) ∧
  (∃ (α : ℝ), α ≥ -2 * Real.pi / 3 ∧ α < -Real.pi / 2 ∧
    (∃ (s : Finset ℝ), s.card = 6 ∧
      (∀ (x : ℝ), x ∈ s → α < x ∧ x < Real.pi ∧ g x = 0) ∧
      (∀ (x : ℝ), α < x ∧ x < Real.pi ∧ g x = 0 → x ∈ s))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l974_97482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_condition_l974_97464

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 + a * x + 3)

theorem domain_condition (a : ℝ) : 
  (∀ x, f a x ∈ Set.univ) ↔ a ∈ Set.Icc 0 12 ∧ a ≠ 12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_condition_l974_97464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_possible_scores_l974_97444

/-- Represents a basketball game where a player made 7 baskets, each worth either 2 or 3 points -/
structure BasketballGame where
  scores : Finset ℕ
  two_pointers : ℕ
  three_pointers : ℕ
  sum_constraint : two_pointers + three_pointers = 7
  scores_def : scores = Finset.image (fun x => 2 * (7 - x) + 3 * x) (Finset.range 8)

/-- The number of different possible total scores in the basketball game -/
theorem number_of_possible_scores (game : BasketballGame) : game.scores.card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_possible_scores_l974_97444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_square_factors_l974_97446

open Finset Nat

def set_100 : Finset Nat := range 100

def perfect_squares : List Nat := [4, 9, 16, 25, 36, 49, 64, 81]

def has_square_factor (n : Nat) : Bool :=
  perfect_squares.any (fun s => n % s == 0)

theorem count_numbers_with_square_factors :
  (set_100.filter (fun n => has_square_factor n)).card = 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_square_factors_l974_97446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cape_may_sightings_count_l974_97476

/-- The number of shark sightings in Daytona Beach -/
def daytona_sightings : ℕ := sorry

/-- The number of shark sightings in Cape May -/
def cape_may_sightings : ℕ := sorry

/-- The total number of shark sightings in both locations -/
def total_sightings : ℕ := 40

/-- Cape May has 8 less than double the number of shark sightings of Daytona Beach -/
axiom cape_may_relation : cape_may_sightings = 2 * daytona_sightings - 8

/-- The sum of shark sightings in both locations equals the total -/
axiom total_sum : daytona_sightings + cape_may_sightings = total_sightings

theorem cape_may_sightings_count : cape_may_sightings = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cape_may_sightings_count_l974_97476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_theorem_l974_97490

noncomputable def triangle_ABC (a b c A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi ∧
  a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C)

theorem triangle_ABC_theorem (a b c A B C : ℝ) 
  (h_triangle : triangle_ABC a b c A B C) 
  (h_condition : (2 * b - c) / a = Real.cos C / Real.cos A) : 
  A = Real.pi / 3 ∧ 
  Set.Icc (1 : ℝ) 2 = { y | ∃ B C, triangle_ABC a b c (Real.pi/3) B C ∧ 
                              y = Real.sqrt 3 * Real.sin B + Real.sin (C - Real.pi/6) } :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_theorem_l974_97490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_coloring_books_l974_97466

/-- The number of pictures in Rachel's coloring books -/
structure ColoringBooks where
  first : ℕ 
  second : ℕ 
  colored : ℕ 
  remaining : ℕ 

/-- Rachel's coloring book problem -/
theorem rachel_coloring_books 
  (books : ColoringBooks) 
  (h1 : books.second = 32)
  (h2 : books.colored = 44)
  (h3 : books.remaining = 11) :
  books.first = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_coloring_books_l974_97466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sixth_term_l974_97488

/-- An arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmeticSequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmeticSum (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem arithmetic_sequence_sixth_term :
  ∀ (a₁ d : ℝ),
    a₁ = 2 →
    arithmeticSum a₁ d 3 = 12 →
    arithmeticSequence a₁ d 6 = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sixth_term_l974_97488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_selection_theorem_l974_97486

/-- A chord is represented by a pair of integers (a, b) where 1 ≤ a < b ≤ n -/
def Chord (n : ℕ) := { p : ℕ × ℕ // 1 ≤ p.1 ∧ p.1 < p.2 ∧ p.2 ≤ n }

/-- Two chords intersect if and only if a < c < b < d or c < a < d < b -/
def intersect {n : ℕ} (c1 c2 : Chord n) : Prop :=
  (c1.val.1 < c2.val.1 ∧ c2.val.1 < c1.val.2 ∧ c1.val.2 < c2.val.2) ∨
  (c2.val.1 < c1.val.1 ∧ c1.val.1 < c2.val.2 ∧ c2.val.2 < c1.val.2)

theorem chord_selection_theorem (n k : ℕ) (h1 : 2 * k + 1 < n) 
  (chords : Finset (Chord n)) (h2 : chords.card = n * k + 1) :
  ∃ (non_intersecting : Finset (Chord n)), non_intersecting ⊆ chords ∧ 
  non_intersecting.card = k + 1 ∧ 
  ∀ c1 c2, c1 ∈ non_intersecting → c2 ∈ non_intersecting → c1 ≠ c2 → ¬(intersect c1 c2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_selection_theorem_l974_97486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_theorem_l974_97472

theorem divisibility_theorem (a b c : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a^3 % b = 0)
  (h2 : b^3 % c = 0)
  (h3 : c^3 % a = 0) :
  (a + b + c)^13 % (a * b * c) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_theorem_l974_97472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trigonometric_relations_l974_97432

/-- Given a triangle ABC with incircle radius r, circumcircle radius R, area S,
    incenter I, circumcenter O, and points K and H as defined:
    K is the sum of the complex numbers of the points where the incircle touches the sides,
    H corresponds to the complex number z₁ + z₂ + z₃ where z₁, z₂, z₃ are the complex
    numbers corresponding to vertices A, B, C respectively. -/
theorem triangle_trigonometric_relations (A B C : ℝ) (r R S : ℝ) (IK OH : ℝ) :
  0 < r ∧ 0 < R ∧ 0 < S ∧ 0 ≤ A ∧ A < 2 * Real.pi ∧ 0 ≤ B ∧ B < 2 * Real.pi ∧ 0 ≤ C ∧ C < 2 * Real.pi ∧
  A + B + C = Real.pi →
  (Real.cos A + Real.cos B + Real.cos C = 3/2 - IK^2/(2*r^2)) ∧
  (Real.sin (2*A) + Real.sin (2*B) + Real.sin (2*C) = 2*S/R^2) ∧
  (Real.cos (2*A) + Real.cos (2*B) + Real.cos (2*C) = OH^2/(2*R^2) - 3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trigonometric_relations_l974_97432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_photo_probability_l974_97439

/-- Rachel's lap time in seconds -/
noncomputable def rachel_lap_time : ℝ := 100

/-- Robert's lap time in seconds -/
noncomputable def robert_lap_time : ℝ := 70

/-- Duration of the photo opportunity in seconds -/
noncomputable def photo_duration : ℝ := 60

/-- Fraction of the track captured in the photo -/
noncomputable def photo_coverage : ℝ := 1/3

/-- Time when the photo opportunity starts (in seconds after the start of the run) -/
noncomputable def photo_start_time : ℝ := 8 * 60

theorem runner_photo_probability :
  let rachel_photo_time := photo_coverage * rachel_lap_time
  let robert_photo_time := photo_coverage * robert_lap_time
  let rachel_position := photo_start_time % rachel_lap_time
  let robert_position := photo_start_time % robert_lap_time
  let rachel_overlap := min rachel_photo_time (rachel_lap_time - rachel_position)
  let robert_overlap := min robert_photo_time (robert_lap_time - robert_position)
  let overlap_time := min rachel_overlap robert_overlap
  overlap_time / photo_duration = 11/20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_photo_probability_l974_97439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_mike_pairing_probability_l974_97470

/-- Represents a class of students -/
structure StudentClass where
  size : ℕ
  mark_in_class : Prop
  mike_in_class : Prop

/-- Represents the pairing process -/
def random_pairing (c : StudentClass) : Prop :=
  c.size % 2 = 0 ∧ c.size ≥ 2

/-- The probability of Mark being paired with Mike -/
def prob_mark_with_mike (c : StudentClass) : ℚ :=
  1 / (c.size - 1 : ℚ)

/-- Theorem stating the probability of Mark being paired with Mike in a class of 16 students -/
theorem mark_mike_pairing_probability (c : StudentClass) 
    (h1 : c.size = 16) 
    (h2 : c.mark_in_class) 
    (h3 : c.mike_in_class) 
    (h4 : random_pairing c) : 
  prob_mark_with_mike c = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_mike_pairing_probability_l974_97470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aaron_earnings_l974_97418

-- Define the work hours for each day
def monday_hours : ℚ := 2
def tuesday_hours : ℚ := 5/4
def wednesday_hours : ℚ := 3
def thursday_hours : ℚ := 3/4

-- Define the pay rates
def base_rate : ℚ := 5
def overtime_rate : ℚ := 7

-- Define a function to calculate daily earnings
noncomputable def daily_earnings (hours : ℚ) : ℚ :=
  if hours ≤ 2 then hours * base_rate
  else 2 * base_rate + (hours - 2) * overtime_rate

-- Define the total earnings for the week
noncomputable def total_earnings : ℚ :=
  daily_earnings monday_hours +
  daily_earnings tuesday_hours +
  daily_earnings wednesday_hours +
  daily_earnings thursday_hours

-- Theorem statement
theorem aaron_earnings : total_earnings = 37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_aaron_earnings_l974_97418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_imply_m_value_l974_97481

-- Define the variables
variable (a b d m : ℝ)

-- Define the equation as a function
def equation (a b d m : ℝ) (x : ℝ) : Prop :=
  (x^2 - (b+d)*x) / (a*x - d) = (m-2) / (m+2)

-- Define the condition that roots are numerically equal but opposite in sign
def roots_condition (a b d m : ℝ) : Prop :=
  ∃ (α : ℝ), (equation a b d m α ∧ equation a b d m (-α))

-- State the theorem
theorem roots_imply_m_value (a b d : ℝ) (h : roots_condition a b d m) : 
  m = (2*(a-b-d)) / (a+b+d) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_imply_m_value_l974_97481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_approx_verify_future_value_l974_97461

/-- The interest rate that causes an investment to grow from $8000 to $32000 over 36 years -/
noncomputable def interest_rate : ℝ :=
  (((32000 : ℝ) / 8000) ^ (1 / 36) - 1) * 100

/-- Theorem stating that the interest rate is approximately 3.63% -/
theorem interest_rate_approx :
  abs (interest_rate - 3.63) < 0.01 := by
  sorry

/-- Function to calculate the future value of an investment -/
noncomputable def future_value (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate / 100) ^ years

/-- Theorem verifying that the calculated interest rate results in the correct future value -/
theorem verify_future_value :
  abs (future_value 8000 interest_rate 36 - 32000) < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_approx_verify_future_value_l974_97461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_ratio_l974_97477

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The ratio of distances for a specific parabola and line configuration -/
theorem parabola_distance_ratio 
  (C : Parabola) 
  (F G A B : Point) 
  (h1 : F.x = C.p / 2 ∧ F.y = 0)  -- Focus position
  (h2 : G.x = -C.p / 2 ∧ G.y = 0)  -- Directrix intersection with x-axis
  (h3 : A.y = 4/3 * (A.x - F.x))  -- Line equation through F with slope 4/3
  (h4 : B.y = 4/3 * (B.x - F.x))  -- Line equation through F with slope 4/3
  (h5 : A.y^2 = 2 * C.p * A.x)  -- A is on the parabola
  (h6 : B.y^2 = 2 * C.p * B.x)  -- B is on the parabola
  (h7 : A.y > 0)  -- A is above x-axis
  : distance G A / distance G B = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_ratio_l974_97477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_quarters_point_l974_97438

def start : ℚ := 1/5
def finish : ℚ := 4/5
def weight_start : ℚ := 1
def weight_finish : ℚ := 3

theorem three_quarters_point :
  (weight_start * start + weight_finish * finish) / (weight_start + weight_finish) = 13/20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_quarters_point_l974_97438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_equality_l974_97423

-- Define a Point type
structure Point where
  x : ℝ
  y : ℝ

-- Define a Line type
structure Line where
  p1 : Point
  p2 : Point

structure Triangle where
  A : Point
  B : Point
  C : Point

structure Parallelogram where
  P : Point
  Q : Point
  R : Point
  S : Point

-- Define a function to construct a parallelogram on a side of a triangle
def Triangle.constructParallelogram (t : Triangle) (side : Fin 3) : Parallelogram :=
  sorry

-- Define an area function for parallelograms
noncomputable def Parallelogram.area (p : Parallelogram) : ℝ :=
  sorry

-- Define a distance function between two points
def Point.dist (p1 p2 : Point) : ℝ :=
  sorry

-- Define an intersection function for two lines
def Line.intersect (l1 l2 : Line) : Point :=
  sorry

-- Define a parallel check for two lines
def Line.parallel (l1 l2 : Line) : Prop :=
  sorry

theorem parallelogram_area_equality 
  (ABC : Triangle) 
  (ACDE : Parallelogram)
  (BCFG : Parallelogram)
  (ABML : Parallelogram)
  (h1 : ACDE = ABC.constructParallelogram 0)
  (h2 : BCFG = ABC.constructParallelogram 1)
  (h3 : ABML = ABC.constructParallelogram 2)
  (h4 : ABML.P.dist ABML.Q = (Line.intersect (Line.mk ACDE.R ACDE.S) (Line.mk BCFG.R BCFG.S)).dist ABC.C)
  (h5 : Line.parallel (Line.mk ABML.P ABML.S) (Line.mk ABC.C (Line.intersect (Line.mk ACDE.R ACDE.S) (Line.mk BCFG.R BCFG.S))))
  : ABML.area = ACDE.area + BCFG.area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_equality_l974_97423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_at_B_l974_97424

-- Define the square and pentagon
def square_side_length : ℝ := 4
def AF : ℝ := 2
def FB : ℝ := 1

-- Define the point P on AB
def P (x : ℝ) : ℝ := x

-- Define the area of rectangle PNDM as a function of x
def area_PNDM (x : ℝ) : ℝ := (2 + 2*x) * (4 - x)

-- Theorem statement
theorem max_area_at_B :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 →
    area_PNDM x ≤ area_PNDM 1 :=
by
  sorry

#check max_area_at_B

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_at_B_l974_97424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_l974_97417

/-- The slope angle of line m in degrees -/
noncomputable def slope_angle : ℝ := 45

/-- The equation of line l₁ -/
def l₁ (x y : ℝ) : Prop := x + y - 1 = 0

/-- The equation of line l₂ -/
def l₂ (x y : ℝ) : Prop := 2*x + 2*y - 5 = 0

/-- l₁ and l₂ are parallel -/
axiom parallel_lines : ∃ (k : ℝ), ∀ (x y : ℝ), l₁ x y ↔ l₂ (k*x) (k*y)

/-- The length of segment AB -/
noncomputable def length_AB : ℝ := 3 * Real.sqrt 2 / 4

theorem segment_length :
  ∀ (A B : ℝ × ℝ),
  (∃ (x₁ y₁ x₂ y₂ : ℝ), A = (x₁, y₁) ∧ B = (x₂, y₂) ∧ l₁ x₁ y₁ ∧ l₂ x₂ y₂) →
  (∃ (m : ℝ), Real.tan (slope_angle * π / 180) = m) →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = length_AB :=
by
  sorry

#check segment_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_l974_97417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_implies_identity_l974_97410

/-- A function satisfying the given functional equation is the identity function -/
theorem functional_equation_implies_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (y - f x) = f x - 2*x + f (f y)) : 
  ∀ y : ℝ, f y = y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_implies_identity_l974_97410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_minimum_iff_l974_97407

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < a then -a * x + 4 else (x - 2)^2

-- Define the property of having a minimum value
def has_minimum (g : ℝ → ℝ) : Prop :=
  ∃ (m : ℝ), ∀ (x : ℝ), g x ≥ m

-- State the theorem
theorem f_has_minimum_iff (a : ℝ) :
  has_minimum (f a) ↔ 0 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_minimum_iff_l974_97407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_bounds_l974_97411

/-- The ellipse on which point P moves -/
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

/-- The circle N -/
def circle_n (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

/-- The dot product of vectors PE and PF -/
def dot_product (px py ex ey fx fy : ℝ) : ℝ :=
  (px - ex) * (px - fx) + (py - ey) * (py - fy)

theorem dot_product_bounds :
  ∀ px py ex ey fx fy : ℝ,
  ellipse px py →
  circle_n ex ey →
  circle_n fx fy →
  (ex - fx)^2 + (ey - fy)^2 = 4 →  -- EF is a diameter
  dot_product px py ex ey fx fy ≤ 19 ∧
  dot_product px py ex ey fx fy ≥ 12 - 4 * Real.sqrt 3 :=
by
  sorry

#check dot_product_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_bounds_l974_97411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_at_vertex_l974_97483

/-- Regular triangular pyramid -/
structure RegularTriangularPyramid where
  /-- Base of the pyramid -/
  base : Triangle
  /-- Vertex of the pyramid -/
  vertex : Point
  /-- The pyramid is regular -/
  regular : Prop

/-- Plane passing through a vertex of the base -/
structure IntersectingPlane (pyramid : RegularTriangularPyramid) where
  /-- The plane passes through a vertex of the base -/
  passes_through_base_vertex : Prop
  /-- The plane is perpendicular to the opposite lateral face -/
  perpendicular_to_lateral_face : Prop
  /-- The plane is parallel to the opposite side of the base -/
  parallel_to_base_side : Prop
  /-- The plane forms an angle α with the base plane -/
  angle_with_base : ℝ

/-- Dihedral angle at the vertex of a pyramid -/
def DihedralAngleAtVertex (pyramid : RegularTriangularPyramid) : ℝ := 
  sorry

/-- Theorem: Dihedral angle at the vertex of a regular triangular pyramid -/
theorem dihedral_angle_at_vertex (pyramid : RegularTriangularPyramid) 
  (plane : IntersectingPlane pyramid) :
  DihedralAngleAtVertex pyramid = 2 * Real.arctan (Real.sqrt 3 * Real.sin plane.angle_with_base) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_at_vertex_l974_97483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_angle_ratio_is_right_angled_l974_97475

/-- A triangle with angles in the ratio 1:5:6 is a right-angled triangle -/
theorem triangle_with_angle_ratio_is_right_angled (A B C : ℝ) 
  (h_angles : A + B + C = 180) 
  (h_ratio : (A, B, C) = (15, 75, 90)) : 
  A = 15 ∧ B = 75 ∧ C = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_angle_ratio_is_right_angled_l974_97475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_breadth_is_six_l974_97415

/-- Calculates the breadth of a room given its length, carpet width, carpet cost per meter, and total cost. -/
noncomputable def room_breadth (room_length : ℝ) (carpet_width : ℝ) (carpet_cost_per_meter : ℝ) (total_cost : ℝ) : ℝ :=
  (total_cost / carpet_cost_per_meter * carpet_width) / room_length

/-- Theorem stating that the breadth of the room is 6 meters given the specified conditions. -/
theorem room_breadth_is_six :
  let room_length : ℝ := 15
  let carpet_width : ℝ := 0.75  -- 75 cm converted to meters
  let carpet_cost_per_meter : ℝ := 0.30  -- 30 paise converted to rupees
  let total_cost : ℝ := 36
  room_breadth room_length carpet_width carpet_cost_per_meter total_cost = 6 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_breadth_is_six_l974_97415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_period_omega_l974_97452

/-- The minimum positive period of sin(ωx - π/3) is π when ω = 2 -/
theorem sin_period_omega (ω : ℝ) (h₁ : ω > 0) :
  (∀ x : ℝ, Real.sin (ω * x - Real.pi / 3) = Real.sin (ω * (x + Real.pi) - Real.pi / 3)) → ω = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_period_omega_l974_97452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_positive_factors_count_l974_97436

def n : ℕ := 2^4 * 3^3 * 5 * 7

theorem even_positive_factors_count : 
  (Finset.filter (fun x => x ∣ n ∧ Even x ∧ x > 0) (Finset.range (n + 1))).card = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_positive_factors_count_l974_97436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_day_after_2_pow_20_l974_97425

/-- Enumeration of days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

/-- Function to calculate the day of the week after a given number of days -/
def dayAfter (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => dayAfter (nextDay start) n

/-- Theorem stating that after 2^20 days from Monday, it will be Friday -/
theorem day_after_2_pow_20 : dayAfter DayOfWeek.Monday (2^20) = DayOfWeek.Friday := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_day_after_2_pow_20_l974_97425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_price_calculation_l974_97456

/-- Calculates the original price of an item given the final price after discounts and tax --/
noncomputable def calculate_original_price (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (discount3 : ℝ) (tax_rate : ℝ) : ℝ :=
  let price_before_tax := final_price / (1 + tax_rate)
  let discount_factor := (1 - discount1) * (1 - discount2) * (1 - discount3)
  price_before_tax / discount_factor

/-- Theorem stating that given the specified discounts, tax rate, and final price, 
    the original price is approximately $32.91 --/
theorem original_price_calculation (final_price : ℝ) (h_final_price : final_price = 17) 
  (discount1 : ℝ) (h_discount1 : discount1 = 0.25)
  (discount2 : ℝ) (h_discount2 : discount2 = 0.25)
  (discount3 : ℝ) (h_discount3 : discount3 = 0.15)
  (tax_rate : ℝ) (h_tax_rate : tax_rate = 0.08) :
  ∃ ε > 0, |calculate_original_price final_price discount1 discount2 discount3 tax_rate - 32.91| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_price_calculation_l974_97456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_k_value_l974_97409

theorem orthogonal_vectors_k_value 
  (a b c : ℝ × ℝ)
  (ha : a = (2, 3))
  (hb : b = (1, 4))
  (hc : c = (k, 3))
  (h_orthogonal : (a.1 + b.1, a.2 + b.2) • c = 0) :
  k = -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_k_value_l974_97409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l974_97431

-- Part I
theorem part_one : Real.sqrt 0.04 + (-2) - Real.sqrt (1/4) + 2 = -0.3 := by
  sorry

-- Part II
theorem part_two : 
  {x : ℝ | 4 * (x + 5)^2 = 16} = {-7, -3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l974_97431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_greater_than_N_l974_97404

-- Define the variables and conditions
axiom a : ℝ
axiom b : ℝ
axiom m : ℝ
axiom n : ℝ
axiom a_pos : a > 0
axiom b_pos : b > 0
axiom inequality : m^2 * n^2 > a^2 * m^2 + b^2 * n^2

-- Define M and N
noncomputable def M : ℝ := Real.sqrt (m^2 + n^2)
noncomputable def N : ℝ := a + b

-- State the theorem
theorem M_greater_than_N : M > N := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_greater_than_N_l974_97404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_partition_exists_l974_97429

/-- Represents a piece of the partitioned square -/
structure Piece where
  cells : Set (Fin 8 × Fin 8)
  perimeter : ℕ

/-- Represents a partition of the 8x8 square -/
structure Partition where
  pieces : Finset Piece
  valid : pieces.card = 7

/-- The 8x8 square -/
def Square : Set (Fin 8 × Fin 8) :=
  Set.univ

/-- Checks if a partition is valid according to the problem conditions -/
def isValidPartition (p : Partition) : Prop :=
  -- All pieces are disjoint and cover the entire square
  (∀ (i j : Piece), i ∈ p.pieces → j ∈ p.pieces → i ≠ j → i.cells ∩ j.cells = ∅) ∧
  (⋃ (piece ∈ p.pieces), piece.cells) = Square ∧
  -- All pieces have equal perimeters
  (∀ (i j : Piece), i ∈ p.pieces → j ∈ p.pieces → i.perimeter = j.perimeter) ∧
  -- Pieces are formed along cell boundaries (implicit in the Piece structure)
  True

/-- The main theorem stating that a valid partition exists -/
theorem valid_partition_exists : ∃ (p : Partition), isValidPartition p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_partition_exists_l974_97429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_46_l974_97435

def repeating_decimal (whole : ℕ) (repeating : ℕ) : ℚ :=
  (whole : ℚ) + (repeating : ℚ) / (100 - 1)

theorem repeating_decimal_46 :
  repeating_decimal 0 46 = 46 / 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_46_l974_97435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_cube_root_difference_l974_97416

theorem smallest_k_for_cube_root_difference (n : ℕ) (hn : n = 2016) :
  ∃ k : ℕ, k = 13 ∧
  (∀ S : Finset ℕ, S.card = k → S ⊆ Finset.range n →
    ∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ |Real.rpow a (1/3 : ℝ) - Real.rpow b (1/3 : ℝ)| < 1) ∧
  (∀ m : ℕ, m < k →
    ∃ T : Finset ℕ, T.card = m ∧ T ⊆ Finset.range n ∧
      ∀ a b : ℕ, a ∈ T → b ∈ T → a ≠ b → |Real.rpow a (1/3 : ℝ) - Real.rpow b (1/3 : ℝ)| ≥ 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_cube_root_difference_l974_97416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_expression_natural_number_l974_97449

theorem square_expression_natural_number : 
  ∀ n : ℕ, (∃ m : ℕ, n^2 - 4*n + 11 = m^2) ↔ n = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_expression_natural_number_l974_97449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_semicircles_area_l974_97402

/-- The side length of the regular octagon -/
noncomputable def octagon_side_length : ℝ := 4

/-- The number of sides in the octagon -/
def octagon_sides : ℕ := 8

/-- The area of a regular octagon with side length s -/
noncomputable def octagon_area (s : ℝ) : ℝ := 2 * (1 + Real.sqrt 2) * s^2

/-- The area of a semicircle with radius r -/
noncomputable def semicircle_area (r : ℝ) : ℝ := Real.pi * r^2 / 2

/-- The theorem stating the area of the region inside the octagon but outside all semicircles -/
theorem octagon_semicircles_area :
  octagon_area octagon_side_length - octagon_sides * semicircle_area (octagon_side_length / 2) =
  32 * (1 + Real.sqrt 2) - 16 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_semicircles_area_l974_97402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_wins_by_15_meters_l974_97450

/-- A race between two contestants A and B -/
structure Race where
  length : ℚ
  speedRatio : ℚ × ℚ
  headStart : ℚ

/-- Calculate the winning margin for contestant A -/
def winningMargin (race : Race) : ℚ :=
  let aSpeed := race.speedRatio.fst
  let bSpeed := race.speedRatio.snd
  let timeForB := race.length / bSpeed
  let aDistance := aSpeed * timeForB + race.headStart
  aDistance - race.length

/-- Theorem stating that A wins the race by 15 meters -/
theorem a_wins_by_15_meters :
  let race : Race := {
    length := 500,
    speedRatio := (3, 4),
    headStart := 140
  }
  winningMargin race = 15 := by
  -- Proof goes here
  sorry

#eval winningMargin { length := 500, speedRatio := (3, 4), headStart := 140 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_wins_by_15_meters_l974_97450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_weight_bounds_l974_97414

/-- Represents Arun's weight in kilograms -/
def arun_weight : ℝ := sorry

/-- Arun's opinion: lower bound -/
def arun_lower_bound : ℝ := 62

/-- Arun's opinion: upper bound -/
def arun_upper_bound : ℝ := 72

/-- Brother's opinion: lower bound -/
def brother_lower_bound : ℝ := 60

/-- Brother's opinion: upper bound -/
def brother_upper_bound : ℝ := 70

/-- Average of probable weights -/
def average_weight : ℝ := 64

/-- Mother's opinion: upper bound (to be proved) -/
def mother_upper_bound : ℝ := 66

theorem arun_weight_bounds :
  arun_lower_bound < arun_weight ∧ 
  arun_weight < arun_upper_bound ∧
  brother_lower_bound < arun_weight ∧ 
  arun_weight < brother_upper_bound ∧
  arun_weight ≤ mother_upper_bound ∧
  average_weight = (arun_lower_bound + mother_upper_bound) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_weight_bounds_l974_97414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_not_subset_of_curve_l974_97419

-- Define the function f and the curve C
variable {X : Type} [TopologicalSpace X]
variable (f : X → X → ℝ)
variable (C : Set (X × X))

-- Define the solution set of f(x, y) = 0
def SolutionSet (f : X → X → ℝ) : Set (X × X) :=
  {p : X × X | f p.1 p.2 = 0}

-- Theorem statement
theorem solution_not_subset_of_curve
  (h1 : Set.Nonempty (SolutionSet f))
  (h2 : ¬(SolutionSet f ⊆ C)) :
  ∃ p : X × X, p ∈ SolutionSet f ∧ p ∉ C := by
  sorry

#check solution_not_subset_of_curve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_not_subset_of_curve_l974_97419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l974_97412

def my_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧
  (∀ k, a (2 * k) = a (2 * k - 1) + a k) ∧
  (∀ k, a (2 * k + 1) = a (2 * k))

theorem sequence_inequality (a : ℕ → ℕ) (h : my_sequence a) :
  ∀ n : ℕ, n ≥ 3 → a (2^n) < 2^(n^2/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l974_97412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_w_is_468_l974_97413

/-- The prime factorization of 1452 -/
def factor_1452 : ℕ := 2^2 * 3^3 * 13

/-- The smallest possible value of w -/
def w : ℕ := 468

/-- The prime factorization of w -/
def factor_w : ℕ := 2^2 * 3^2 * 13

theorem smallest_w_is_468 :
  (∀ k : ℕ, k > 0 ∧ k < w → ¬(2^4 ∣ (factor_1452 * k))) ∧
  2^4 ∣ (factor_1452 * w) ∧
  3^3 ∣ (factor_1452 * w) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_w_is_468_l974_97413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_internally_l974_97474

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 16 = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + (y+1)^2 = 5

-- Define the centers and radii
def center_C1 : ℝ × ℝ := (2, 0)
noncomputable def radius_C1 : ℝ := Real.sqrt 20
def center_C2 : ℝ × ℝ := (0, -1)
noncomputable def radius_C2 : ℝ := Real.sqrt 5

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 5

-- Theorem statement
theorem circles_tangent_internally :
  distance_between_centers = radius_C2 ∧ radius_C1 > radius_C2 :=
by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_internally_l974_97474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_rounded_to_hundredth_l974_97455

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The sum of 91.234 and 42.7689 rounded to the nearest hundredth is 134.00 -/
theorem sum_rounded_to_hundredth : round_to_hundredth (91.234 + 42.7689) = 134.00 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_rounded_to_hundredth_l974_97455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_selection_probability_l974_97454

def total_pairs : ℕ := 17
def black_pairs : ℕ := 8
def brown_pairs : ℕ := 4
def gray_pairs : ℕ := 3
def blue_pairs : ℕ := 2

theorem shoe_selection_probability : 
  (let total_shoes := total_pairs * 2
   let prob_black := (black_pairs * 2 : ℚ) / total_shoes * (black_pairs : ℚ) / (total_shoes - 1)
   let prob_brown := (brown_pairs * 2 : ℚ) / total_shoes * (brown_pairs : ℚ) / (total_shoes - 1)
   let prob_gray := (gray_pairs * 2 : ℚ) / total_shoes * (gray_pairs : ℚ) / (total_shoes - 1)
   let prob_blue := (blue_pairs * 2 : ℚ) / total_shoes * (blue_pairs : ℚ) / (total_shoes - 1)
   prob_black + prob_brown + prob_gray + prob_blue) = 31 / 187 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_selection_probability_l974_97454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_is_odd_l974_97459

noncomputable def f (x : ℝ) : ℝ := (3 : ℝ) ^ x

noncomputable def F (x : ℝ) : ℝ := f x - 1 / f x

theorem F_is_odd : ∀ x : ℝ, F (-x) = -F x := by
  intro x
  simp [F, f]
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_is_odd_l974_97459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_condition_l974_97485

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Conditions for a valid triangle
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = Real.pi
  -- Law of sines
  law_of_sines : a / (Real.sin A) = b / (Real.sin B)

-- Define what it means for a triangle to be obtuse
def is_obtuse (t : Triangle) : Prop :=
  t.A > Real.pi/2 ∨ t.B > Real.pi/2 ∨ t.C > Real.pi/2

-- State the theorem
theorem obtuse_triangle_condition (t : Triangle) (h : t.c < t.b * (Real.cos t.A)) :
  is_obtuse t :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_condition_l974_97485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mother_age_ratio_year_l974_97465

/-- Represents Jessie's age -/
def jessie_age (year : ℕ) : ℕ := year - 2000

/-- Represents Jessie's mother's age -/
def mother_age (year : ℕ) : ℕ := 5 * (jessie_age 2010) + (year - 2010)

/-- The theorem to prove -/
theorem mother_age_ratio_year : 
  (∃ (year : ℕ), year > 2010 ∧ 
    (mother_age year : ℚ) = (5/2 : ℚ) * (jessie_age year : ℚ) ∧
    ∀ (y : ℕ), 2010 < y ∧ y < year → 
      (mother_age y : ℚ) ≠ (5/2 : ℚ) * (jessie_age y : ℚ)) → 2027 = 2027 := by
  sorry

#eval mother_age 2027 -- To verify the calculation
#eval jessie_age 2027 -- To verify the calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mother_age_ratio_year_l974_97465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_freeze_opponent_l974_97458

/-- The number of sides on each die -/
def sides : ℕ := 12

/-- The sum needed to freeze the opponent -/
def target_sum : ℕ := 15

/-- The probability of rolling a specific sum with two fair dice -/
def prob_sum (n : ℕ) : ℚ :=
  (Finset.filter (λ (x : ℕ × ℕ) => x.1 + x.2 = n) (Finset.product (Finset.range sides) (Finset.range sides))).card /
  (sides * sides)

theorem prob_freeze_opponent : prob_sum target_sum = 5 / 36 := by
  sorry

#eval prob_sum target_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_freeze_opponent_l974_97458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_exists_l974_97451

/-- Returns the largest digit in the decimal representation of a natural number -/
def max_digit (n : ℕ) : ℕ := sorry

/-- Generates the next number in the sequence -/
def next_number (n : ℕ) : ℕ := n - max_digit n

/-- Predicate to check if a number is even -/
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

/-- Theorem stating that a sequence of more than 12 numbers satisfying the conditions exists -/
theorem sequence_exists : ∃ (a : ℕ → ℕ), 
  (∀ i, is_even (a i)) ∧ 
  (∀ i, 2 ≤ i ∧ i ≤ 13 → a i = next_number (a (i-1))) :=
sorry

#check sequence_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_exists_l974_97451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l974_97492

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

theorem f_properties :
  (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f (-x) = -f x) ∧  -- f is odd
  f (1/2) = 2/5 ∧  -- f(1/2) = 2/5
  (∀ x y, x ∈ Set.Ioo (-1 : ℝ) 1 → y ∈ Set.Ioo (-1 : ℝ) 1 → x < y → f x < f y)  -- f is strictly increasing
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l974_97492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_field_ratio_is_seven_to_five_l974_97493

/-- Represents a rectangular field -/
structure RectangularField where
  width : ℚ
  length : ℚ

/-- Calculates the perimeter of a rectangular field -/
def perimeter (field : RectangularField) : ℚ :=
  2 * (field.width + field.length)

/-- Calculates the ratio of length to width as a pair of integers -/
def lengthToWidthRatio (field : RectangularField) : ℤ × ℤ :=
  let n := field.length.num * field.width.den
  let d := field.width.num * field.length.den
  let gcd := Int.gcd n d
  (n / gcd, d / gcd)

/-- Theorem stating the ratio of length to width for a specific rectangular field -/
theorem field_ratio_is_seven_to_five :
  ∃ (field : RectangularField),
    field.width = 60 ∧
    perimeter field = 288 ∧
    lengthToWidthRatio field = (7, 5) := by
  let field : RectangularField := ⟨60, 84⟩
  existsi field
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_field_ratio_is_seven_to_five_l974_97493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l974_97440

/-- Represents a four-digit number as an ordered quadruple of digits -/
def FourDigitNumber := (Nat × Nat × Nat × Nat)

/-- Converts a four-digit number to its decimal representation -/
def toDecimal (n : FourDigitNumber) : Nat :=
  let (a, b, c, d) := n
  1000 * a + 100 * b + 10 * c + d

/-- Reverses the digits of a four-digit number -/
def reverse (n : FourDigitNumber) : FourDigitNumber :=
  let (a, b, c, d) := n
  (d, c, b, a)

/-- Rearranges the digits of a four-digit number in BDAC order -/
def rearrange (n : FourDigitNumber) : FourDigitNumber :=
  let (a, b, c, d) := n
  (b, d, a, c)

theorem unique_solution :
  ∃! (n : FourDigitNumber),
    let (a, b, c, d) := n
    a > b ∧ b > c ∧ c > d
    ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10
    ∧ toDecimal n - toDecimal (reverse n) = toDecimal (rearrange n)
    ∧ n = (7, 6, 4, 1) := by
  sorry

#eval toDecimal (7, 6, 4, 1) - toDecimal (reverse (7, 6, 4, 1)) -- Should output 3825
#eval toDecimal (rearrange (7, 6, 4, 1)) -- Should also output 3825

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l974_97440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_discount_is_twenty_percent_l974_97489

/-- Calculates the second discount percentage given the initial price, first discount rate, and final price -/
noncomputable def second_discount_percentage (initial_price first_discount_rate final_price : ℝ) : ℝ :=
  let price_after_first_discount := initial_price * (1 - first_discount_rate)
  let discount_amount := price_after_first_discount - final_price
  (discount_amount / price_after_first_discount) * 100

/-- Theorem stating that the second discount percentage is 20% given the problem conditions -/
theorem second_discount_is_twenty_percent :
  second_discount_percentage 49.99 0.1 36 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_discount_is_twenty_percent_l974_97489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_in_cones_l974_97403

/-- Represents a right circular cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

/-- Calculates the volume of a sphere -/
noncomputable def sphereVolume (s : Sphere) : ℝ := (4/3) * Real.pi * s.radius^3

/-- Calculates the rise in liquid level when a sphere is submerged in a cone -/
noncomputable def liquidRise (c : Cone) (s : Sphere) : ℝ :=
  sphereVolume s / (Real.pi * c.radius^2)

theorem liquid_rise_ratio_in_cones (c1 c2 : Cone) (s : Sphere) :
  c1.radius = 4 →
  c2.radius = 8 →
  s.radius = 2 →
  coneVolume c1 = coneVolume c2 →
  liquidRise c1 s / liquidRise c2 s = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_in_cones_l974_97403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l974_97496

-- Define the nabla operation for positive real numbers
noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- Theorem statement
theorem nabla_calculation : 
  nabla (nabla (1/2) (1/3)) (1/4) = 9/11 := by
  -- Expand the definition of nabla
  unfold nabla
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l974_97496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_subset_l974_97497

def A : Finset Nat := {2, 5, 8, 11, 14, 17, 20, 23, 26, 29}

def is_valid_set (S : Finset Nat) : Prop :=
  ∀ a b : Nat, ∀ k : Int, a ∈ S → b ∈ S →
    ¬∃ n : Int, (a : Int) + (b : Int) + 30 * k = n * (n + 1)

def is_subset_of_range (S : Finset Nat) : Prop :=
  ∀ x, x ∈ S → x ≤ 29

theorem max_valid_subset :
  is_valid_set A ∧
  is_subset_of_range A ∧
  ∀ S : Finset Nat, is_valid_set S → is_subset_of_range S →
    Finset.card S ≤ Finset.card A :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_subset_l974_97497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_theorem_l974_97473

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculates the centroid of a triangle -/
def centroid (t : Triangle) : Point :=
  { x := (t.A.x + t.B.x + t.C.x) / 3,
    y := (t.A.y + t.B.y + t.C.y) / 3 }

/-- Theorem: In triangle ABC with A(1,2), B(3,1), and centroid G(3,2), the coordinates of point C are (5,3) -/
theorem triangle_centroid_theorem : 
  ∀ (t : Triangle), 
    t.A = { x := 1, y := 2 } →
    t.B = { x := 3, y := 1 } →
    centroid t = { x := 3, y := 2 } →
    t.C = { x := 5, y := 3 } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_theorem_l974_97473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_angle_of_inclination_l974_97445

-- Define the function
noncomputable def f (x : ℝ) := x^3 / 3 - x^2 + 1

-- Define the derivative of the function
noncomputable def f' (x : ℝ) := x^2 - 2*x

-- Define the angle of inclination
noncomputable def α (x : ℝ) := Real.arctan (f' x)

-- Theorem statement
theorem min_angle_of_inclination :
  ∀ x : ℝ, 0 < x → x < 2 → α x ≥ 3*π/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_angle_of_inclination_l974_97445
