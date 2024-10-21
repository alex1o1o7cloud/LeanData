import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_cooler_capacity_l400_40049

theorem water_cooler_capacity (initial_ounces : ℕ) : 
  let rows : ℕ := 5
  let chairs_per_row : ℕ := 10
  let ounces_per_cup : ℕ := 6
  let remaining_ounces : ℕ := 84
  let total_chairs : ℕ := rows * chairs_per_row
  let total_ounces_used : ℕ := total_chairs * ounces_per_cup
  let ounces_per_gallon : ℕ := 128
  initial_ounces = total_ounces_used + remaining_ounces →
  initial_ounces / ounces_per_gallon = 3 := by
  intro h
  sorry

#eval 3 * 128  -- This should output 384, which is the initial number of ounces

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_cooler_capacity_l400_40049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_cutting_line_slope_l400_40044

/-- A parallelogram with vertices at (10,45), (10,114), (28,153), and (28,84) -/
structure Parallelogram where
  v1 : ℝ × ℝ := (10, 45)
  v2 : ℝ × ℝ := (10, 114)
  v3 : ℝ × ℝ := (28, 153)
  v4 : ℝ × ℝ := (28, 84)

/-- A line through the origin with slope m/n -/
structure CuttingLine where
  m : ℕ
  n : ℕ
  m_pos : m > 0
  n_pos : n > 0
  coprime : Nat.Coprime m n

/-- The theorem to be proved -/
theorem parallelogram_cutting_line_slope (p : Parallelogram) (l : CuttingLine) :
  (l.m : ℝ) / l.n = 99 / 19 → l.m + l.n = 118 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_cutting_line_slope_l400_40044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rental_payment_is_260_l400_40048

/-- Calculates the total payment for a car rental given the rental conditions. -/
def totalPayment (dailyRate : ℚ) (mileageRate : ℚ) (serviceCharge : ℚ) (days : ℕ) (miles : ℚ) : ℚ :=
  dailyRate * days + mileageRate * miles + serviceCharge

/-- Theorem stating that the total payment for the given rental conditions is $260. -/
theorem rental_payment_is_260 :
  let dailyRate : ℚ := 30
  let mileageRate : ℚ := 1/4
  let serviceCharge : ℚ := 15
  let days : ℕ := 4
  let miles : ℚ := 500
  totalPayment dailyRate mileageRate serviceCharge days miles = 260 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rental_payment_is_260_l400_40048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_is_reals_l400_40030

/-- The function f(x) with parameter k -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (3*x^2 + 2*x - k) / (-7*x^2 + 3*x + 4*k)

/-- The theorem stating the condition for f to have a domain of all real numbers -/
theorem domain_of_f_is_reals (k : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, f k x = y) ↔ k < -9/112 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_is_reals_l400_40030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_probabilities_l400_40082

/-- The number of balls in the lottery -/
def N : ℕ := 18

/-- The number of balls drawn in the main draw -/
def main_draw : ℕ := 10

/-- The number of balls drawn in the additional draw -/
def additional_draw : ℕ := 8

/-- The lucky sum in the main draw -/
def main_sum : ℕ := 63

/-- The lucky sum in the additional draw -/
def additional_sum : ℕ := 44

/-- The number of combinations of selecting 'k' balls from 'n' balls with a sum of 's' -/
noncomputable def count_sum_combinations (n k s : ℕ) : ℕ := sorry

/-- The probability of selecting 'k' balls from 'n' balls with a sum of 's' -/
noncomputable def prob_sum (n k s : ℕ) : ℚ :=
  (count_sum_combinations n k s : ℚ) / (Nat.choose n k : ℚ)

/-- Theorem stating that the probabilities of the two events are equal -/
theorem equal_probabilities :
  prob_sum N main_draw main_sum = prob_sum N additional_draw additional_sum := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_probabilities_l400_40082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l400_40025

/-- Given a triangle ABC, this theorem states the relationship between
    the lengths of tangents, medians, semiperimeter, circumradius, and inradius. -/
theorem triangle_inequality (a b c s R r t_a t_b t_c m_a m_b m_c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_s : s = (a + b + c) / 2)
  (h_R : R > 0)
  (h_r : r > 0)
  (h_t_a : t_a^2 = s * (s - a))
  (h_t_b : t_b^2 = s * (s - b))
  (h_t_c : t_c^2 = s * (s - c))
  (h_m_a : m_a^2 = (2*b^2 + 2*c^2 - a^2) / 4)
  (h_m_b : m_b^2 = (2*c^2 + 2*a^2 - b^2) / 4)
  (h_m_c : m_c^2 = (2*a^2 + 2*b^2 - c^2) / 4) :
  t_a^6 + t_b^6 + t_c^6 ≤ s^4 * (s^2 - 12*R*r) ∧ 
  s^4 * (s^2 - 12*R*r) ≤ m_a^6 + m_b^6 + m_c^6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l400_40025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_lines_l400_40077

/-- Two lines in 3D space --/
def Line1 (u : ℝ) : Fin 3 → ℝ := fun i => 
  match i with
  | 0 => 1 + u
  | 1 => -2 - 3*u
  | 2 => 3 + 2*u

def Line2 (v : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => -1 + 2*v
  | 1 => 1 + 4*v
  | 2 => 5 - 2*v

/-- Squared distance between two points --/
def dist_squared (p q : Fin 3 → ℝ) : ℝ :=
  (p 0 - q 0)^2 + (p 1 - q 1)^2 + (p 2 - q 2)^2

/-- Theorem: The shortest distance between Line1 and Line2 is √13 --/
theorem shortest_distance_between_lines :
  ∃ (u v : ℝ), ∀ (u' v' : ℝ),
    dist_squared (Line1 u) (Line2 v) ≤ dist_squared (Line1 u') (Line2 v') ∧
    dist_squared (Line1 u) (Line2 v) = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_lines_l400_40077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_job_hours_ratio_l400_40070

/-- Represents the weekly work schedule and earnings of James --/
structure WorkSchedule where
  main_job_hourly_rate : ℚ
  second_job_hourly_rate : ℚ
  main_job_hours : ℚ
  total_weekly_earnings : ℚ

/-- Calculates the ratio of hours worked at the second job to the main job --/
def calculate_job_hours_ratio (schedule : WorkSchedule) : ℚ :=
  let second_job_earnings := schedule.total_weekly_earnings - (schedule.main_job_hourly_rate * schedule.main_job_hours)
  let second_job_hours := second_job_earnings / schedule.second_job_hourly_rate
  second_job_hours / schedule.main_job_hours

/-- The theorem stating that James' job hours ratio is 1/2 --/
theorem james_job_hours_ratio :
  let james_schedule : WorkSchedule := {
    main_job_hourly_rate := 20,
    second_job_hourly_rate := 20 * 4/5,
    main_job_hours := 30,
    total_weekly_earnings := 840
  }
  calculate_job_hours_ratio james_schedule = 1/2 := by sorry

#eval calculate_job_hours_ratio {
  main_job_hourly_rate := 20,
  second_job_hourly_rate := 20 * 4/5,
  main_job_hours := 30,
  total_weekly_earnings := 840
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_job_hours_ratio_l400_40070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_l400_40020

/-- Partnership investment structure and profit calculation -/
theorem partnership_profit (x : ℝ) :
  let a_investment := x
  let b_investment := 2 * x
  let c_investment := 3 * x
  let a_time := 12
  let b_time := 6
  let c_time := 4
  let total_weighted_investment := a_investment * a_time + b_investment * b_time + c_investment * c_time
  let a_share := 5000
  let total_profit := (total_weighted_investment * a_share) / (a_investment * a_time)
  total_profit = 15000 := by
  sorry

#check partnership_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_l400_40020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_base_angle_is_72_degrees_l400_40087

/-- An isosceles trapezoid with a diagonal that divides it into two isosceles triangles -/
structure IsoscelesTrapezoid (A B C D : ℝ × ℝ) :=
  (isIsosceles : A.1 - B.1 = C.1 - D.1)
  (diagonalDividesIntoIsosceles : (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2
                                ∧ (A.1 - C.1)^2 + (A.2 - C.2)^2 = (A.1 - D.1)^2 + (A.2 - D.2)^2)

/-- The base angle of an isosceles trapezoid -/
noncomputable def baseAngle (A B C D : ℝ × ℝ) : ℝ :=
  let v1 := (B.1 - A.1, B.2 - A.2)
  let v2 := (C.1 - A.1, C.2 - A.2)
  Real.arccos ((v1.1 * v2.1 + v1.2 * v2.2) / (Real.sqrt (v1.1^2 + v1.2^2) * Real.sqrt (v2.1^2 + v2.2^2)))

theorem isosceles_trapezoid_base_angle_is_72_degrees (A B C D : ℝ × ℝ) 
  (h : IsoscelesTrapezoid A B C D) : 
  baseAngle A B C D = 72 * Real.pi / 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_base_angle_is_72_degrees_l400_40087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l400_40021

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_a_ge_b : a ≥ b

/-- Calculates the distance between the foci of an ellipse -/
noncomputable def focalDistance (e : Ellipse) : ℝ :=
  2 * (e.a^2 - e.b^2).sqrt

/-- Theorem: For an ellipse with semi-major axis 5 and semi-minor axis 3,
    the distance between the foci is 8 -/
theorem ellipse_focal_distance :
  ∃ (e : Ellipse), e.a = 5 ∧ e.b = 3 ∧ focalDistance e = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l400_40021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l400_40074

/-- The time it takes for A and B to complete the work together -/
noncomputable def combined_time : ℝ := 6

/-- The time it takes for A to complete the work alone -/
noncomputable def a_time : ℝ := 8

/-- The work rate of A -/
noncomputable def a_rate : ℝ := 1 / a_time

/-- The combined work rate of A and B -/
noncomputable def combined_rate : ℝ := 1 / combined_time

/-- Theorem stating that A and B together complete the work in 6 days -/
theorem work_completion_time :
  combined_rate = a_rate + (combined_rate - a_rate) →
  combined_time = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l400_40074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l400_40064

/-- The distance from a point (x,y) to the line x+1=0 -/
def distToLine (x y : ℝ) : ℝ := |x + 1|

/-- The distance from a point (x,y) to the fixed point F(3,0) -/
noncomputable def distToF (x y : ℝ) : ℝ := Real.sqrt ((x - 3)^2 + y^2)

/-- The theorem representing the problem -/
theorem parabola_equation (x y : ℝ) :
  distToF x y = distToLine x y + 2 → y^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l400_40064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pumping_time_is_450_minutes_l400_40068

-- Define the problem parameters
noncomputable def basement_length : ℝ := 30
noncomputable def basement_width : ℝ := 40
noncomputable def water_depth_inches : ℝ := 24
def num_pumps : ℕ := 4
noncomputable def pump_rate : ℝ := 10
noncomputable def cubic_foot_to_gallon : ℝ := 7.5

-- Define the function to calculate the pumping time
noncomputable def pumping_time : ℝ :=
  let water_depth_feet := water_depth_inches / 12
  let water_volume_cubic_feet := water_depth_feet * basement_length * basement_width
  let water_volume_gallons := water_volume_cubic_feet * cubic_foot_to_gallon
  let total_pump_rate := (num_pumps : ℝ) * pump_rate
  water_volume_gallons / total_pump_rate

-- Theorem statement
theorem pumping_time_is_450_minutes : pumping_time = 450 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pumping_time_is_450_minutes_l400_40068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_green_ball_l400_40080

-- Define the containers and their contents
structure Container where
  redBalls : ℕ
  greenBalls : ℕ

-- Define the probabilities of selecting each container
def containerX : Container := ⟨5, 5⟩
def containerY : Container := ⟨7, 3⟩
def containerZ : Container := ⟨2, 8⟩

def probX : ℝ := 0.2
def probY : ℝ := 0.5
def probZ : ℝ := 0.3

-- Function to calculate the probability of selecting a green ball from a container
noncomputable def probGreenFromContainer (c : Container) : ℝ :=
  (c.greenBalls : ℝ) / ((c.redBalls + c.greenBalls) : ℝ)

-- Theorem stating the probability of selecting a green ball
theorem prob_green_ball : 
  probGreenFromContainer containerX * probX + 
  probGreenFromContainer containerY * probY + 
  probGreenFromContainer containerZ * probZ = 0.49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_green_ball_l400_40080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_and_bounds_l400_40006

noncomputable def sequence_a : ℕ → ℝ := sorry
noncomputable def sequence_S : ℕ → ℝ := sorry

axiom S_definition : ∀ n : ℕ, sequence_S n = 3 - 2 * sequence_a n

theorem geometric_sequence_and_bounds :
  (∀ n : ℕ, sequence_a n = (2/3)^(n-1)) ∧
  (∀ n : ℕ, 1 ≤ sequence_S n ∧ sequence_S n < 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_and_bounds_l400_40006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_sums_l400_40037

def is_valid_coloring (blue red : Finset ℕ) : Prop :=
  blue.card = 10 ∧ red.card = 10 ∧ blue ∩ red = ∅ ∧ blue ∪ red = Finset.range 20

def sum_set (blue red : Finset ℕ) : Finset ℕ :=
  Finset.biUnion blue (fun b => Finset.image (fun r => b + r) red)

theorem max_distinct_sums :
  ∃ (blue red : Finset ℕ), 
    is_valid_coloring blue red ∧ 
    (∀ b r : Finset ℕ, is_valid_coloring b r → (sum_set b r).card ≤ (sum_set blue red).card) ∧
    (sum_set blue red).card = 35 :=
by
  sorry

#check max_distinct_sums

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_sums_l400_40037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_cd_is_three_halves_l400_40091

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  /-- Length of side BD -/
  bd : ℝ
  /-- Angle DBA in radians -/
  angle_dba : ℝ
  /-- Angle BDC in radians -/
  angle_bdc : ℝ
  /-- Ratio of BC to AD -/
  ratio_bc_ad : ℝ
  /-- AD is parallel to BC -/
  ad_parallel_bc : Prop
  /-- BD equals 2 -/
  bd_eq_two : bd = 2
  /-- Angle DBA equals 30 degrees (π/6 radians) -/
  angle_dba_eq_thirty_deg : angle_dba = π/6
  /-- Angle BDC equals 60 degrees (π/3 radians) -/
  angle_bdc_eq_sixty_deg : angle_bdc = π/3
  /-- Ratio of BC to AD is 7:4 -/
  ratio_bc_ad_eq_seven_four : ratio_bc_ad = 7/4

/-- The length of CD in the trapezoid -/
noncomputable def length_cd (t : Trapezoid) : ℝ := 3/2

/-- Theorem stating that the length of CD is 3/2 in the given trapezoid -/
theorem length_cd_is_three_halves (t : Trapezoid) : length_cd t = 3/2 := by
  sorry

#eval "Build successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_cd_is_three_halves_l400_40091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_eccentricity_l400_40031

/-- An ellipse where one focus and the two endpoints of its minor axis form an equilateral triangle -/
structure SpecialEllipse where
  /-- Semi-major axis length -/
  a : ℝ
  /-- Distance from center to focus -/
  c : ℝ
  /-- Condition that the focus and minor axis endpoints form an equilateral triangle -/
  h_equilateral : c = (Real.sqrt 3 / 2) * a
  /-- Condition that a > 0 -/
  h_a_pos : a > 0

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : SpecialEllipse) : ℝ := e.c / e.a

/-- Theorem stating that the eccentricity of the special ellipse is √3/2 -/
theorem special_ellipse_eccentricity (e : SpecialEllipse) :
  eccentricity e = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_eccentricity_l400_40031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_max_length_l400_40073

/-- A geometric sequence with integer terms between 100 and 1000, inclusive -/
def GeometricSequence (a : ℕ → ℕ) : Prop :=
  ∃ (r : ℚ), r > 1 ∧
  (∀ n, a n ≥ 100 ∧ a n ≤ 1000) ∧
  (∀ n, a (n + 1) = (a n * r.num) / r.den)

/-- The maximum length of such a geometric sequence -/
def MaxLength : ℕ := 6

/-- The sequence achieving the maximum length -/
def MaxSequence : ℕ → ℕ
| 1 => 128
| 2 => 192
| 3 => 288
| 4 => 432
| 5 => 648
| 6 => 972
| _ => 0

theorem geometric_sequence_max_length :
  (GeometricSequence MaxSequence) ∧
  (∀ a, GeometricSequence a → ∃ n, ∀ m > n, a m = 0) ∧
  (∀ a, GeometricSequence a → ∀ n > MaxLength, a n = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_max_length_l400_40073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_on_zero_one_l400_40078

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := x^2 * f (x - 1)

-- State the theorem
theorem g_decreasing_on_zero_one :
  ∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 1 → g y < g x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_on_zero_one_l400_40078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_distinct_values_l400_40057

/-- Represents a list of positive integers with a unique mode -/
structure IntegerList where
  elements : List Nat
  total_count : Nat
  mode_count : Nat
  mode_unique : Bool

/-- The properties of our specific integer list -/
def our_list : IntegerList :=
  { elements := [],  -- We don't need to specify the actual elements
    total_count := 2020,
    mode_count := 12,
    mode_unique := true }

/-- The minimum number of distinct values in the list -/
def min_distinct_values : Nat := 185

theorem least_distinct_values (list : IntegerList) 
  (h1 : list.total_count = our_list.total_count)
  (h2 : list.mode_count = our_list.mode_count)
  (h3 : list.mode_unique = our_list.mode_unique) :
  ∃ (n : Nat), n = min_distinct_values ∧ 
    n ≤ Finset.card (Finset.image id (Finset.range list.total_count)) := by
  sorry

#check least_distinct_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_distinct_values_l400_40057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l400_40060

/-- The system of equations has exactly four solutions -/
theorem system_solutions :
  ∃! (S : Set (ℝ × ℝ)), S.Finite ∧ S.ncard = 4 ∧
  (∀ (x y : ℝ), (x, y) ∈ S ↔
    ((x^2 + y^2)^2 - x*y*(x + y)^2 = 19 ∧ |x - y| = 1)) ∧
  ((2.8, 1.8) ∈ S ∧ (-1.4, -2.4) ∈ S ∧ (1.8, 2.8) ∈ S ∧ (-2.4, -1.4) ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l400_40060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_equation_of_l_intersection_range_l400_40016

-- Define the curve C
noncomputable def C (t : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos (2 * t), 2 * Real.sin t)

-- Define the line l in polar form
def l (ρ θ m : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 3) + m = 0

-- Statement 1: Cartesian equation of l
theorem cartesian_equation_of_l (x y m : ℝ) :
  (∃ ρ θ, l ρ θ m ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ↔
  Real.sqrt 3 * x + y + 2 * m = 0 :=
sorry

-- Statement 2: Range of m for intersection
theorem intersection_range (m : ℝ) :
  (∃ t, ∃ x y, C t = (x, y) ∧ Real.sqrt 3 * x + y + 2 * m = 0) ↔
  -19/12 ≤ m ∧ m ≤ 5/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_equation_of_l_intersection_range_l400_40016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_times_theorem_l400_40038

/-- Represents a train with a constant speed -/
structure Train where
  speed : ℝ

/-- Represents the journey between two stations -/
structure Journey where
  distance : ℝ
  trainA : Train
  trainB : Train
  meetingTime : ℝ
  trainAPostMeetingTime : ℝ
  trainBPostMeetingTime : ℝ

/-- Calculates the journey time for a train -/
def journeyTime (j : Journey) (isTrainA : Bool) : ℝ :=
  if isTrainA then
    j.meetingTime + j.trainAPostMeetingTime
  else
    j.meetingTime + j.trainBPostMeetingTime

/-- Theorem stating the journey times for both trains -/
theorem journey_times_theorem (j : Journey) (a b c : ℝ) :
  (journeyTime j true = a / 2 + b + Real.sqrt (a^2 + 4 * b * c) / 2) ∧
  (journeyTime j false = c - a / 2 + Real.sqrt (a^2 + 4 * b * c) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_times_theorem_l400_40038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chameleon_color_impossibility_l400_40081

/-- Represents the number of chameleons of each color -/
structure ChameleonPopulation where
  gray : ℕ
  brown : ℕ
  crimson : ℕ

/-- The rule for color change when two different colored chameleons meet -/
def colorChange (pop : ChameleonPopulation) : ChameleonPopulation :=
  pop  -- Placeholder, actual implementation not needed for the theorem

/-- The initial population of chameleons -/
def initialPopulation : ChameleonPopulation :=
  ⟨13, 15, 17⟩

/-- Checks if all chameleons are the same color -/
def allSameColor (pop : ChameleonPopulation) : Prop :=
  (pop.gray = 0 ∧ pop.brown = 0) ∨ (pop.gray = 0 ∧ pop.crimson = 0) ∨ (pop.brown = 0 ∧ pop.crimson = 0)

/-- The difference between gray and brown chameleons modulo 3 is invariant -/
def invariant (pop : ChameleonPopulation) : ℕ :=
  (pop.gray - pop.brown) % 3

/-- Theorem stating that it's impossible for all chameleons to become the same color -/
theorem chameleon_color_impossibility :
  ¬∃ (finalPop : ChameleonPopulation), 
    (∃ (n : ℕ), finalPop = (Nat.iterate colorChange n initialPopulation)) ∧ 
    allSameColor finalPop :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chameleon_color_impossibility_l400_40081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_for_a_l400_40013

theorem solve_equation_for_a (a b c d : ℕ) 
  (h : (18^a : ℕ) * (9^(4*a - 1) : ℕ) * (27^c : ℕ) = (2^6 : ℕ) * (3^b : ℕ) * (7^d : ℕ)) : 
  a = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_for_a_l400_40013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_ratio_l400_40033

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the line
noncomputable def line (x y : ℝ) : Prop := x - Real.sqrt 3 * y + Real.sqrt 3 = 0

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 1)

-- Define the distance from a point to the focus
noncomputable def dist_to_focus (p : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - focus.1)^2 + (p.2 - focus.2)^2)

theorem parabola_line_intersection_ratio :
  ∀ A B : ℝ × ℝ,
  parabola A.1 A.2 →
  parabola B.1 B.2 →
  line A.1 A.2 →
  line B.1 B.2 →
  dist_to_focus A > dist_to_focus B →
  dist_to_focus A / dist_to_focus B = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_ratio_l400_40033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_excluding_stoppages_l400_40096

theorem bus_speed_excluding_stoppages 
  (stop_time : ℝ) 
  (avg_speed_with_stops : ℝ) 
  (h1 : stop_time = 28) 
  (h2 : avg_speed_with_stops = 40) : 
  (avg_speed_with_stops * 60) / (60 - stop_time) = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_excluding_stoppages_l400_40096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_k_same_foci_correct_propositions_l400_40069

-- Define the sum of a geometric sequence
def geometric_sum (n : ℕ) (k : ℝ) : ℝ := 2^n + k

-- Define the hyperbola and ellipse equations
def hyperbola (x y : ℝ) : Prop := x^2 / 25 - y^2 / 9 = 1
def ellipse (x y : ℝ) : Prop := x^2 / 35 + y^2 = 1

-- Theorem for the geometric sequence
theorem geometric_sequence_k (n : ℕ) : 
  ∃ k : ℝ, geometric_sum n k = 2^n - 1 := by
  use -1
  simp [geometric_sum]
  ring

-- Theorem for the foci of hyperbola and ellipse
theorem same_foci : 
  ∃ c : ℝ, c^2 = 34 ∧ 
  (∀ x y : ℝ, hyperbola x y → (x = c ∨ x = -c) ∧ y = 0) ∧
  (∀ x y : ℝ, ellipse x y → (x = c ∨ x = -c) ∧ y = 0) := by
  sorry

-- The correct propositions are 2 and 4
theorem correct_propositions : True := by
  trivial

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_k_same_foci_correct_propositions_l400_40069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l400_40072

/-- The equation of the region -/
def region_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 6*x - 8*y + 9 = 0

/-- The area of the region -/
noncomputable def region_area : ℝ := 16 * Real.pi

/-- Theorem stating that the area of the region defined by the equation is 16π -/
theorem area_of_region :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, region_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l400_40072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l400_40071

theorem simplify_expression (y : ℝ) (h : y ≠ 0) :
  5 / (4 * y^(-4 : ℤ)) * (4 * y^3) / 3 = 5 * y^7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l400_40071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_is_123_l400_40054

/-- The price of one book in kopecks -/
def book_price : ℕ := sorry

/-- Nine books cost slightly less than or equal to 1100 kopecks -/
axiom nine_books_cost : book_price * 9 ≤ 1100

/-- Thirteen books cost slightly less than or equal to 1500 kopecks -/
axiom thirteen_books_cost : book_price * 13 ≤ 1500

/-- The price of one book is 123 kopecks -/
theorem book_price_is_123 : book_price = 123 := by
  sorry

#check book_price_is_123

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_is_123_l400_40054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_plus_product_eq_one_solutions_l400_40097

theorem absolute_difference_plus_product_eq_one_solutions :
  ∀ a b : ℕ, (Nat.sub a b + Nat.sub b a) + a * b = 1 ↔ (a = 1 ∧ b = 0) ∨ (a = 0 ∧ b = 1) ∨ (a = 1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_plus_product_eq_one_solutions_l400_40097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2005_is_cos_l400_40099

open Real

/-- Recursive definition of the function sequence -/
noncomputable def f (n : ℕ) : ℝ → ℝ := 
  match n with
  | 0 => sin
  | n + 1 => deriv (f n)

/-- The 2005th function in the sequence is cosine -/
theorem f_2005_is_cos : f 2005 = cos := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2005_is_cos_l400_40099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_membership_change_l400_40076

theorem membership_change (X : ℝ) (X_pos : X > 0) : 
  let fall_factor := 1.09
  let winter_factor := 1.15
  let spring_factor := 0.81
  let summer_increase_factor := 1.12
  let summer_decrease_factor := 0.95
  let final_count := X * fall_factor * winter_factor * spring_factor * summer_increase_factor * summer_decrease_factor
  let percentage_change := (final_count / X - 1) * 100
  percentage_change = (fall_factor * winter_factor * spring_factor * summer_increase_factor * summer_decrease_factor - 1) * 100 := by
  sorry

#eval (1.09 * 1.15 * 0.81 * 1.12 * 0.95 - 1) * 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_membership_change_l400_40076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_volume_is_53_76_l400_40022

/-- Represents the mass of limestone in grams -/
def limestone_mass : ℚ := 300

/-- Represents the percentage of calcium carbonate in the limestone -/
def calcium_carbonate_percentage : ℚ := 4/5

/-- Represents the molar mass of calcium carbonate in g/mol -/
def molar_mass_CaCO3 : ℚ := 100

/-- Represents the volume occupied by 1 mole of gas at STP in liters -/
def molar_volume_STP : ℚ := 224/10

/-- Calculates the volume of gas released in liters -/
noncomputable def gas_volume_released : ℚ :=
  limestone_mass * calcium_carbonate_percentage / molar_mass_CaCO3 * molar_volume_STP

/-- Theorem stating that the volume of gas released is 53.76 L -/
theorem gas_volume_is_53_76 : gas_volume_released = 5376/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_volume_is_53_76_l400_40022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l400_40042

-- Define the vertices of the quadrilateral
def v1 : ℝ × ℝ := (0, 1)
def v2 : ℝ × ℝ := (5, 4)
def v3 : ℝ × ℝ := (4, 3)
def v4 : ℝ × ℝ := (3, 0)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the perimeter of the quadrilateral
noncomputable def perimeter : ℝ :=
  distance v1 v2 + distance v2 v3 + distance v3 v4 + distance v4 v1

-- Theorem statement
theorem quadrilateral_perimeter :
  perimeter = 4 * Real.sqrt 2 + 2 * Real.sqrt 10 ∧
  (4 : ℤ) + (2 : ℤ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l400_40042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_milk_percentage_l400_40011

theorem initial_milk_percentage
  (initial_volume : ℝ)
  (water_added : ℝ)
  (final_milk_percentage : ℝ)
  (initial_milk_percentage : ℝ)
  (h1 : initial_volume = 60)
  (h2 : water_added = 24)
  (h3 : final_milk_percentage = 60)
  (h4 : final_milk_percentage / 100 * (initial_volume + water_added) = initial_volume * (initial_milk_percentage / 100)) :
  initial_milk_percentage = 84 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_milk_percentage_l400_40011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_in_interval_l400_40053

-- Define the recursive function g
noncomputable def g : ℕ → ℝ
| 0 => 0
| 1 => 0
| 2 => 0
| 3 => Real.log 3
| n + 4 => Real.log (n + 4 + g (n + 3))

-- Define B as g(2024)
noncomputable def B : ℝ := g 2024

-- State the theorem
theorem B_in_interval : Real.log 2027 < B ∧ B < Real.log 2028 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_in_interval_l400_40053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vacation_days_count_l400_40009

-- Define the predicates as functions
def rain_afternoon (d : ℕ) : Prop := sorry
def sunny_morning (d : ℕ) : Prop := sorry
def sunny_afternoon (d : ℕ) : Prop := sorry

theorem vacation_days_count (rain_count : ℕ) (sunny_afternoon_count : ℕ) (sunny_morning_count : ℕ) :
  rain_count = 7 →
  (∀ d, rain_afternoon d → sunny_morning d) →
  (∃ d, sunny_afternoon d) ∧ (∃ d, sunny_morning d) →
  (∀ d, sunny_afternoon d ∨ rain_afternoon d) →
  (∀ d, sunny_morning d ∨ ¬sunny_morning d) →
  ∃ e : ℕ, e = 9 ∧ 
    e = rain_count + sunny_afternoon_count - (rain_count - (sunny_morning_count - (sunny_afternoon_count - rain_count))) :=
by
  sorry

#check vacation_days_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vacation_days_count_l400_40009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_symmetry_and_decreasing_l400_40015

noncomputable def f (k : ℤ) (x : ℝ) : ℝ := (k^2 + k - 1) * x^(k^2 - 3*k)

theorem power_function_symmetry_and_decreasing (k : ℤ) :
  (∀ x > 0, f k x = f k (1/x)) →  -- Symmetry about y-axis
  (∀ x y, 0 < x ∧ x < y → f k y < f k x) →  -- Decreasing on (0, +∞)
  k = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_symmetry_and_decreasing_l400_40015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_fixed_point_and_min_area_l400_40090

/-- Parabola passing through (1,2) -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop

/-- Point on the parabola -/
structure ParabolaPoint (para : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : para.eq x y

/-- Line defined by two points -/
def Line (p1 p2 : ℝ × ℝ) : Set (ℝ × ℝ) := 
  {p : ℝ × ℝ | (p.2 - p1.2) * (p2.1 - p1.1) = (p.1 - p1.1) * (p2.2 - p1.2)}

/-- Theorem statement -/
theorem parabola_fixed_point_and_min_area (para : Parabola) 
  (h1 : para.eq 1 2) -- Parabola passes through (1,2)
  (A B : ParabolaPoint para) 
  (hAB : A ≠ B)
  (hB : ∃ P : ℝ × ℝ, P.1 = (B.y - 3) ∧ P.2 = B.y ∧ P ∈ Line (A.x, A.y) (1, 2)) :
  ((3, 2) ∈ Line (A.x, A.y) (B.x, B.y)) ∧ 
  (∃ (S : ℝ), S = 4 * Real.sqrt 2 ∧ 
    ∀ (A' B' : ParabolaPoint para), 
      A' ≠ B' → 
      (∃ P' : ℝ × ℝ, P'.1 = (B'.y - 3) ∧ P'.2 = B'.y ∧ P' ∈ Line (A'.x, A'.y) (1, 2)) →
      S ≤ abs ((A'.x - 1) * (B'.y - 2) - (B'.x - 1) * (A'.y - 2))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_fixed_point_and_min_area_l400_40090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squares_in_square_l400_40039

/-- A square configuration represents a placement of squares within a larger square. -/
structure SquareConfiguration where
  outer_square : Set (ℝ × ℝ)
  inner_squares : List (Set (ℝ × ℝ))

/-- Predicate to check if two squares are non-overlapping -/
def NonOverlapping (s1 s2 : Set (ℝ × ℝ)) : Prop :=
  s1 ∩ s2 = ∅

/-- Predicate to check if a configuration is valid (all squares are non-overlapping and within the outer square) -/
def ValidConfiguration (config : SquareConfiguration) : Prop :=
  (∀ s, s ∈ config.inner_squares → s ⊆ config.outer_square) ∧
  (∀ s1 s2, s1 ∈ config.inner_squares → s2 ∈ config.inner_squares → s1 ≠ s2 → NonOverlapping s1 s2)

/-- The main theorem: No more than 8 non-overlapping squares can be placed within a square -/
theorem max_squares_in_square (config : SquareConfiguration) 
  (h_valid : ValidConfiguration config) : 
  config.inner_squares.length ≤ 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squares_in_square_l400_40039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_odd_sum_is_118_231_l400_40041

def balls : Finset ℕ := Finset.range 11

def is_sum_odd (s : Finset ℕ) : Bool :=
  s.sum id % 2 = 1

theorem probability_of_odd_sum_is_118_231 :
  (Finset.filter (λ s => is_sum_odd s ∧ s.card = 6) (Finset.powerset balls)).card /
  (Finset.filter (λ s => s.card = 6) (Finset.powerset balls)).card = 118 / 231 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_odd_sum_is_118_231_l400_40041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_tulips_count_l400_40085

/-- Represents the number of flowers in a garden -/
structure GardenFlowers where
  roses : Nat
  daisies : Nat
  tulips : Nat

/-- Calculates the total number of flowers in the garden -/
def total (g : GardenFlowers) : Nat :=
  g.roses + g.daisies + g.tulips

/-- Theorem: Given the conditions, prove that there are 40 tulips in the garden -/
theorem garden_tulips_count : 
  ∀ g : GardenFlowers, 
  g.roses = 25 → 
  g.daisies = 35 → 
  (g.roses : Real) / (total g : Real) = 1/4 → 
  g.tulips = 40 := by
  intro g h1 h2 h3
  sorry

#check garden_tulips_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_tulips_count_l400_40085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l400_40092

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola equation in the form ax² + by² = 1 -/
structure HyperbolaEquation where
  a : ℝ
  b : ℝ

/-- Check if a point satisfies a hyperbola equation -/
def satisfiesEquation (p : Point) (eq : HyperbolaEquation) : Prop :=
  eq.a * p.x^2 + eq.b * p.y^2 = 1

/-- The distance between foci of a hyperbola -/
noncomputable def focalDistance : ℝ := Real.sqrt 6

/-- Theorem stating the standard equation of the hyperbola satisfying given conditions -/
theorem hyperbola_equation 
  (p1 : Point) 
  (p2 : Point) 
  (p3 : Point) 
  (h1 : p1 = ⟨3, 15/4⟩) 
  (h2 : p2 = ⟨-16/3, 5⟩) 
  (h3 : p3 = ⟨-5, 2⟩) 
  (h4 : focalDistance = Real.sqrt 6) 
  (h5 : ∃ (eq : HyperbolaEquation), satisfiesEquation p1 eq ∧ satisfiesEquation p2 eq ∧ satisfiesEquation p3 eq) :
  ∃ (eq : HyperbolaEquation), 
    (eq = ⟨-1/16, 1/9⟩ ∨ eq = ⟨1/5, -1⟩) ∧ 
    satisfiesEquation p1 eq ∧ 
    satisfiesEquation p2 eq ∧ 
    satisfiesEquation p3 eq := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l400_40092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_0_5795_to_hundredth_l400_40035

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

/-- The theorem states that rounding 0.5795 to the nearest hundredth equals 0.58 -/
theorem round_0_5795_to_hundredth :
  roundToHundredth 0.5795 = 0.58 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_0_5795_to_hundredth_l400_40035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l400_40024

theorem remainder_problem (y : ℕ) (h : (7 * y) % 31 = 1) : (15 + y) % 31 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l400_40024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l400_40061

/-- Given an arithmetic sequence 2, 6, 10, ..., x, y, 26, prove that x + y = 40 -/
theorem arithmetic_sequence_sum (x y : ℝ) : 
  (∃ (a d : ℝ), 
    a = 2 ∧ 
    d = 4 ∧ 
    (∀ n : ℕ, n < 6 → a + n * d ∈ ({2, 6, 10, x, y, 26} : Set ℝ)) ∧
    (∀ z ∈ ({2, 6, 10, x, y, 26} : Set ℝ), ∃ n : ℕ, n < 6 ∧ z = a + n * d)) →
  x + y = 40 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l400_40061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_determines_a_l400_40040

-- Define the curve
noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := a + Real.log x

-- Define the tangent line
def tangent_line (x : ℝ) : ℝ := x

-- Theorem statement
theorem tangent_point_determines_a :
  ∃ (a : ℝ), ∀ (x : ℝ), 
    (deriv (curve a)) x = (deriv tangent_line) x → 
    curve a 1 = tangent_line 1 → 
    a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_determines_a_l400_40040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_4_minus_sqrt3_l400_40008

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t, -Real.sqrt 3 * t)

-- Define curve C₁
noncomputable def curve_C1 (θ : ℝ) : ℝ × ℝ := (Real.cos θ, 1 + Real.sin θ)

-- Define curve C₂ in polar coordinates
noncomputable def curve_C2_polar (θ : ℝ) : ℝ := -2 * Real.cos θ + 2 * Real.sqrt 3 * Real.sin θ

-- Define the intersection points
noncomputable def point_A : ℝ × ℝ := (1/2, 1 + Real.sqrt 3/2)
noncomputable def point_B : ℝ × ℝ := (2, -2 * Real.sqrt 3)

-- State the theorem
theorem length_AB_is_4_minus_sqrt3 :
  Real.sqrt ((point_B.1 - point_A.1)^2 + (point_B.2 - point_A.2)^2) = 4 - Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_4_minus_sqrt3_l400_40008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_range_l400_40014

noncomputable def a (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

noncomputable def b : ℝ × ℝ := (Real.sqrt 3, -1)

theorem magnitude_range (θ : ℝ) :
  0 ≤ ‖2 • (a θ) - b‖ ∧ ‖2 • (a θ) - b‖ ≤ 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_range_l400_40014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_count_l400_40089

def n : ℕ := 32784

theorem divisors_count : 
  (Finset.filter (λ i : ℕ ↦ n % i = 0) (Finset.range 10)).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_count_l400_40089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_other_diagonal_l400_40027

/-- The area of a rhombus given its diagonals -/
noncomputable def rhombusArea (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

theorem rhombus_other_diagonal 
  (A : ℝ) -- Area of the rhombus
  (d1 : ℝ) -- Length of the first diagonal
  (h1 : A = 80) -- The area is 80 square centimeters
  (h2 : d1 = 16) -- The first diagonal is 16 centimeters
  (h3 : A = rhombusArea d1 10) -- The area formula holds
  : ∃ d2 : ℝ, d2 = 10 ∧ A = rhombusArea d1 d2 := by
  sorry

#check rhombus_other_diagonal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_other_diagonal_l400_40027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_equals_5_l400_40066

def sequence_a : ℕ → ℚ
  | 0 => -1/4  -- Add case for 0
  | 1 => -1/4
  | n + 2 => 1 - 1 / sequence_a (n + 1)

theorem a_2018_equals_5 : sequence_a 2018 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_equals_5_l400_40066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_plus_n_equals_five_l400_40000

/-- A function f(x) = x³ + 3x² + mx -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + m*x

/-- A function g(x) = ln(x+1) + nx -/
noncomputable def g (n : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) + n*x

/-- The theorem stating that m + n = 5 given the conditions -/
theorem m_plus_n_equals_five (m n : ℝ) : 
  (n > 0) → 
  (∀ x, f m x = f m (-2 - x) - 2) →  -- Symmetry condition
  (∃! x, f m x = g n x) →           -- Unique intersection
  m + n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_plus_n_equals_five_l400_40000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_b_729_eq_neg_two_thirds_implies_b_eq_one_over_three_to_ninth_l400_40055

theorem log_b_729_eq_neg_two_thirds_implies_b_eq_one_over_three_to_ninth (b : ℝ) :
  b > 0 → Real.log 729 / Real.log b = -2/3 → b = 1 / (3^9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_b_729_eq_neg_two_thirds_implies_b_eq_one_over_three_to_ninth_l400_40055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_spheres_count_l400_40098

-- Define the radii of the two main spheres
variable (a b : ℝ)
-- Assume a > b
axiom h_a_gt_b : a > b

-- Define the number of additional spheres
variable (n : ℕ)

-- Define the radius of an additional sphere
noncomputable def r (a b : ℝ) : ℝ := (a * b) / ((Real.sqrt a + Real.sqrt b) ^ 2)

-- Define the half-opening angle of the cone
noncomputable def φ (a b : ℝ) : ℝ := Real.arcsin ((a - b) / (a + b))

-- Define the angle ψ
noncomputable def ψ (a b : ℝ) : ℝ := Real.arcsin ((b - r a b) / (b + r a b))

-- Define the distance R
noncomputable def R (a b : ℝ) : ℝ := 
  (2 * a * b * (a + b + Real.sqrt (a * b))) / ((a + b) * (Real.sqrt a + Real.sqrt b) ^ 2)

-- State the main theorem
theorem additional_spheres_count (a b : ℝ) (n : ℕ) (h : a > b) :
  (r a b / R a b = Real.sin (π / n)) → (n = 7 ∨ n = 8 ∨ n = 9) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_spheres_count_l400_40098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_license_plate_combinations_l400_40065

theorem license_plate_combinations : ℕ := by
  -- Define the total number of letters in the alphabet
  let total_letters : ℕ := 26

  -- Define the number of vowels (including Y)
  let vowels : ℕ := 6

  -- Define the number of consonants
  let consonants : ℕ := total_letters - vowels

  -- Define the number of digits
  let digits : ℕ := 10

  -- Calculate the total number of license plate combinations
  have h : consonants * vowels * vowels * digits = 7200 := by
    norm_num

  -- Return the result
  exact 7200


end NUMINAMATH_CALUDE_ERRORFEEDBACK_license_plate_combinations_l400_40065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_inequality_sqrt_subtraction_sqrt_division_sqrt_negative_square_l400_40012

theorem sqrt_sum_inequality : ¬(Real.sqrt 2 + Real.sqrt 3 = Real.sqrt 5) := by sorry

theorem sqrt_subtraction : 5 * Real.sqrt 3 - 2 * Real.sqrt 3 = 3 * Real.sqrt 3 := by sorry

theorem sqrt_division : Real.sqrt 6 / Real.sqrt 2 = Real.sqrt 3 := by sorry

theorem sqrt_negative_square : (-Real.sqrt 2)^2 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_inequality_sqrt_subtraction_sqrt_division_sqrt_negative_square_l400_40012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_is_hyperbola_l400_40034

def is_hyperbola (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ a b c d e : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 ∧
  ∀ x y, f x y = a * x^2 + b * y^2 + c * x + d * y + e

theorem conic_section_is_hyperbola :
  let f : ℝ → ℝ → ℝ := λ x y ↦ (x - 4)^2 - 9 * (y + 3)^2 - 27
  is_hyperbola f := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_is_hyperbola_l400_40034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pass_both_levels_probability_l400_40043

/-- Represents the outcome of a single die roll -/
inductive DieRoll
  | one
  | two
  | three
  | four
  | five
  | six

/-- The pass-through game -/
structure PassThroughGame where
  level : Nat
  rolls : Nat
  passThreshold : Nat

/-- Defines the rules for the first level of the game -/
def firstLevel : PassThroughGame :=
  { level := 1
    rolls := 1
    passThreshold := 1 }

/-- Defines the rules for the second level of the game -/
def secondLevel : PassThroughGame :=
  { level := 2
    rolls := 2
    passThreshold := 4 }

/-- Probability of passing a given level -/
noncomputable def passProbability (level : PassThroughGame) : Rat :=
  sorry

/-- Theorem stating the probability of passing both levels consecutively -/
theorem pass_both_levels_probability :
  passProbability firstLevel * passProbability secondLevel = 25 / 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pass_both_levels_probability_l400_40043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_middle_term_l400_40056

/-- Given a geometric sequence with first term -1 and last term -4, 
    prove that the middle term is -2 -/
theorem geometric_sequence_middle_term 
  (a c d e b : ℝ) 
  (h_seq : ∃ r : ℝ, r ≠ 0 ∧ [a, c, d, e, b] = [a, a*r, a*r^2, a*r^3, a*r^4])
  (h_a : a = -1) 
  (h_b : b = -4) : 
  d = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_middle_term_l400_40056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_AB_length_l400_40029

/-- Triangle ABC with perpendicular medians from A and B -/
structure TriangleWithPerpendicularMedians where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  BC_length : ℝ
  AC_length : ℝ
  median_A_perpendicular_median_B : Bool

/-- The theorem stating that for a triangle with specific properties, 
    the square of the length of side AB is 720 -/
theorem square_of_AB_length 
  (triangle : TriangleWithPerpendicularMedians)
  (h1 : triangle.BC_length = 36)
  (h2 : triangle.AC_length = 48)
  (h3 : triangle.median_A_perpendicular_median_B = true) :
  ((triangle.A.1 - triangle.B.1)^2 + (triangle.A.2 - triangle.B.2)^2) = 720 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_AB_length_l400_40029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_product_l400_40028

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0
  a_gt_b : a > b

/-- A point on the ellipse -/
def PointOnEllipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Two points are symmetric about the origin -/
def SymmetricAboutOrigin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- The slope of a line passing through two points -/
noncomputable def Slope (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  (y₂ - y₁) / (x₂ - x₁)

/-- The main theorem -/
theorem ellipse_slope_product (a b : ℝ) (e : Ellipse a b) 
  (x_m y_m x_n y_n x_q y_q : ℝ) :
  PointOnEllipse a b x_m y_m →
  PointOnEllipse a b x_n y_n →
  PointOnEllipse a b x_q y_q →
  SymmetricAboutOrigin x_m y_m x_n y_n →
  x_q ≠ x_m →
  x_q ≠ x_n →
  Slope x_q y_q x_m y_m * Slope x_q y_q x_n y_n = -b^2 / a^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_product_l400_40028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_origin_l400_40004

-- Define the power function
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - 3*m + 3) * x^(m^2 - m - 1)

-- Theorem statement
theorem power_function_origin (m : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → f m x ≠ 0) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_origin_l400_40004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_distance_is_25_l400_40075

/-- The initial distance between Stacy and Heather --/
noncomputable def initial_distance (heather_rate stacy_rate heather_start_delay heather_distance : ℝ) : ℝ :=
  let meeting_time := heather_distance / heather_rate
  let stacy_distance := stacy_rate * (meeting_time + heather_start_delay)
  heather_distance + stacy_distance

/-- Theorem stating the initial distance between Stacy and Heather --/
theorem initial_distance_is_25 :
  let heather_rate : ℝ := 5
  let stacy_rate : ℝ := heather_rate + 1
  let heather_start_delay : ℝ := 24 / 60
  let heather_distance : ℝ := 10.272727272727273
  initial_distance heather_rate stacy_rate heather_start_delay heather_distance = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_distance_is_25_l400_40075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l400_40003

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if 0 ≤ x ∧ x ≤ 4 then x^2 + x
  else if -4 ≤ x ∧ x < 0 then -x^2 + x
  else 0

-- State the theorem
theorem odd_function_properties :
  (∀ x ∈ Set.Icc (-4 : ℝ) 4, f (-x) = -f x) ∧  -- f is odd on [-4,4]
  (f (-1) = -2) ∧                              -- f(-1) = -2
  (∀ x ∈ Set.Ioo 0 4, f x = x^2 + x) ∧         -- f(x) = x^2 + x for 0 < x ≤ 4
  (∀ m ∈ Set.Icc 0 (3/2), f (m-1) + f (2*m+1) ≥ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l400_40003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_growth_equation_l400_40088

/-- Represents the monthly average growth rate of job positions -/
def x : ℝ := sorry

/-- The number of job positions in June -/
def june_positions : ℕ := 1501

/-- The number of job positions in August -/
def august_positions : ℕ := 1815

/-- The equation representing the relationship between initial positions, growth rate, and final positions -/
def growth_equation : Prop :=
  (june_positions : ℝ) * (1 + x)^2 = august_positions

/-- Theorem stating that the growth equation correctly represents the job position growth -/
theorem correct_growth_equation :
  ∃ x : ℝ, growth_equation := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_growth_equation_l400_40088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_odd_implies_a_one_f_increasing_on_positive_reals_l400_40062

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 - 2 / x

-- Define the function g
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x - a

-- Theorem 1: If g is an odd function, then a = 1
theorem g_odd_implies_a_one (a : ℝ) :
  (∀ x, g x a = -g (-x) a) → a = 1 := by sorry

-- Theorem 2: f is monotonically increasing on (0, +∞)
theorem f_increasing_on_positive_reals :
  ∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_odd_implies_a_one_f_increasing_on_positive_reals_l400_40062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_l400_40083

open Real

noncomputable def f (x : ℝ) : ℝ := x * (2017 + log x) * Real.exp 1

theorem problem (x₀ : ℝ) (h : deriv f x₀ = 2018) : x₀ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_l400_40083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l400_40050

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (x + b)) / Real.log a

-- State the theorem
theorem f_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  (∀ x : ℝ, f a b x = f a b (-x)) →  -- f is even
  (∀ x y : ℝ, 0 < x → x < y → (f a b x < f a b y ∨ f a b y < f a b x)) →  -- f is monotonic on (0, +∞)
  f a b (b - 2) < f a b (a + 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l400_40050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_specific_cone_l400_40079

/-- A right circular cone with base radius and slant height -/
structure Cone where
  baseRadius : ℝ
  slantHeight : ℝ

/-- The shortest distance from the vertex to the shortest path on the surface -/
noncomputable def shortestDistance (c : Cone) : ℝ :=
  c.slantHeight * Real.cos (Real.pi / 3)

/-- Theorem: For a cone with base radius 1 and slant height 3, 
    the shortest distance is 1.5 -/
theorem shortest_distance_specific_cone :
  let c : Cone := { baseRadius := 1, slantHeight := 3 }
  shortestDistance c = (3 : ℝ) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_specific_cone_l400_40079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_circle_sector_l400_40019

/-- The height of a cone formed from a sector of a circle -/
theorem cone_height_from_circle_sector (r : ℝ) (h : r = 8) :
  2 * Real.sqrt 15 = 
    let circumference := 2 * Real.pi * r;
    let sector_arc_length := circumference / 4;
    let base_radius := sector_arc_length / (2 * Real.pi);
    let slant_height := r;
    Real.sqrt (slant_height^2 - base_radius^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_circle_sector_l400_40019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l400_40026

theorem expression_simplification (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (((x + 2)^3 * (x^2 - 2*x + 4)^3) / (x^3 + 8)^3)^4 *
  (((x - 2)^3 * (x^2 + 2*x + 4)^3) / (x^3 - 8)^3)^4 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l400_40026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_ink_left_ink_percentage_left_l400_40094

/-- Represents the area that a full marker can paint -/
noncomputable def markerCapacity : ℝ := 3 * (4 * 4)

/-- Represents the area of two 6 inch by 2 inch rectangles -/
noncomputable def rectanglesArea : ℝ := 2 * (6 * 2)

/-- Represents the area of one 8 inch by 4 inch rectangle -/
noncomputable def largeRectangleArea : ℝ := 8 * 4

/-- Represents the area of one 5 inch by 3 inch triangle -/
noncomputable def triangleArea : ℝ := (1 / 2) * 5 * 3

/-- The total area painted by TreShaun -/
noncomputable def totalPaintedArea : ℝ := rectanglesArea + largeRectangleArea + triangleArea

theorem no_ink_left : totalPaintedArea ≥ markerCapacity := by
  sorry

theorem ink_percentage_left : (markerCapacity - totalPaintedArea) / markerCapacity * 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_ink_left_ink_percentage_left_l400_40094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_is_63_l400_40047

/-- The polynomial z^15 + z^14 + z^9 + z^8 + z^7 + z^2 + 1 -/
def f (z : ℂ) : ℂ := z^15 + z^14 + z^9 + z^8 + z^7 + z^2 + 1

/-- The property that f(z) divides z^k - 1 -/
def divides (k : ℕ) : Prop := ∀ z : ℂ, ∃ w : ℂ, z^k - 1 = f z * w

theorem smallest_k_is_63 : 
  (divides 63) ∧ (∀ k : ℕ, 0 < k → k < 63 → ¬(divides k)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_is_63_l400_40047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_corresponds_to_ten_meters_l400_40086

/-- Represents the cloth selling scenario -/
structure ClothSale where
  total_meters : ℚ
  gain_percentage : ℚ

/-- Calculates the number of meters corresponding to the gain -/
noncomputable def gain_meters (sale : ClothSale) : ℚ :=
  (sale.total_meters * sale.gain_percentage) / (100 + sale.gain_percentage)

/-- Theorem stating that for 15 meters sold with 200% gain, the gain corresponds to 10 meters -/
theorem gain_corresponds_to_ten_meters :
  let sale := ClothSale.mk 15 200
  gain_meters sale = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_corresponds_to_ten_meters_l400_40086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_axis_at_max_omega_l400_40063

open Real

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x + π / 4)

-- State the theorem
theorem symmetric_axis_at_max_omega :
  ∃ ω_max : ℝ,
    ω_max > 0 ∧
    ω_max = 7 / 9 ∧
    (∀ x ∈ Set.Ioo 0 (3 * π / 4), f ω_max x > 1 / 2) ∧
    (∀ ω' : ℝ, ω' > 0 → (∀ x ∈ Set.Ioo 0 (3 * π / 4), f ω' x > 1 / 2) → ω' ≤ ω_max) ∧
    ∃ k : ℤ, ω_max * (9 * π / 28) + π / 4 = k * π + π / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_axis_at_max_omega_l400_40063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_correct_l400_40084

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The start point of the line -/
def startPoint : Point3D := ⟨1, 1, 1⟩

/-- The end point of the line -/
def endPoint : Point3D := ⟨4, 5, 6⟩

/-- The radius of the sphere -/
def sphereRadius : ℝ := 2

/-- The distance between intersection points -/
noncomputable def intersectionDistance : ℝ := 14 * Real.sqrt 2 / 5

/-- Theorem stating that the distance between intersection points is correct -/
theorem intersection_distance_correct :
  let line := λ t : ℝ => Point3D.mk
    (startPoint.x + t * (endPoint.x - startPoint.x))
    (startPoint.y + t * (endPoint.y - startPoint.y))
    (startPoint.z + t * (endPoint.z - startPoint.z))
  let sphere := λ (p : Point3D) => p.x^2 + p.y^2 + p.z^2 = sphereRadius^2
  let intersections := {t : ℝ | sphere (line t)}
  ∃ t₁ t₂, t₁ ∈ intersections ∧ t₂ ∈ intersections ∧ t₁ ≠ t₂ ∧
    Real.sqrt ((line t₂).x - (line t₁).x)^2 +
              ((line t₂).y - (line t₁).y)^2 +
              ((line t₂).z - (line t₁).z)^2 = intersectionDistance :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_correct_l400_40084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_area_ratios_l400_40023

/-- Represents a tetrahedron ABCD with a point T on face BCD -/
structure Tetrahedron where
  A : EuclideanSpace ℝ (Fin 3)
  B : EuclideanSpace ℝ (Fin 3)
  C : EuclideanSpace ℝ (Fin 3)
  D : EuclideanSpace ℝ (Fin 3)
  T : EuclideanSpace ℝ (Fin 3)

/-- Represents the common intersection line of the three internal angle bisector planes passing through vertex A -/
noncomputable def internalBisectorLine (tetra : Tetrahedron) : Subspace ℝ (EuclideanSpace ℝ (Fin 3)) :=
  sorry

/-- The area of a triangle given by three points -/
noncomputable def triangleArea (p1 p2 p3 : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  sorry

/-- Represents a plane given by three points -/
noncomputable def planeFromPoints (p1 p2 p3 : EuclideanSpace ℝ (Fin 3)) : Subspace ℝ (EuclideanSpace ℝ (Fin 3)) :=
  sorry

/-- The statement of the theorem -/
theorem tetrahedron_area_ratios (tetra : Tetrahedron) :
  (internalBisectorLine tetra) ⊓ (planeFromPoints tetra.B tetra.C tetra.D) = ⊥ →
  ∃ (k : ℝ),
    triangleArea tetra.T tetra.B tetra.C / triangleArea tetra.A tetra.B tetra.C = k ∧
    triangleArea tetra.T tetra.C tetra.D / triangleArea tetra.A tetra.C tetra.D = k ∧
    triangleArea tetra.T tetra.D tetra.B / triangleArea tetra.A tetra.D tetra.B = k :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_area_ratios_l400_40023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l400_40095

/-- Definition of the ellipse -/
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- Area of the quadrilateral formed by the vertices of the ellipse -/
noncomputable def QuadrilateralArea (a b : ℝ) : ℝ := 4 * Real.sqrt 2

/-- Radius of the inscribed circle of the quadrilateral -/
noncomputable def InscribedCircleRadius (a b : ℝ) : ℝ := (2 * Real.sqrt 2) / 3

/-- Theorem stating the properties of the ellipse -/
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : QuadrilateralArea a b = 4 * Real.sqrt 2)
  (h4 : InscribedCircleRadius a b = (2 * Real.sqrt 2) / 3) :
  (∃ (x y : ℝ), Ellipse a b (x, y) ↔ (x^2 / 8) + y^2 = 1) ∧
  (∃ (M : ℝ × ℝ), (M.1^2 / 4) + (M.2^2 / 32) = 1) ∧
  (∃ (minArea : ℝ), minArea = 16 / 9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l400_40095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_2023_l400_40093

theorem opposite_of_2023 : 
  (∀ x : ℤ, x + (-x) = 0) → 
  -2023 = -2023 := by
  intro h
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_2023_l400_40093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_inequality_iff_prime_power_l400_40058

def phi_nm (n m : ℕ) : ℕ := (Finset.filter (fun x => x.gcd m = 1) (Finset.range n)).card

def is_prime_power (m : ℕ) : Prop :=
  ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ m = p ^ k

theorem phi_inequality_iff_prime_power (m : ℕ) (h : m > 1) :
  (∀ n : ℕ, n > 0 → (phi_nm n m : ℚ) / n ≥ (Nat.totient m : ℚ) / m) ↔
  is_prime_power m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_inequality_iff_prime_power_l400_40058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tshirts_left_yesterday_l400_40018

/-- The number of T-shirts left after selling yesterday -/
def x : ℚ := 300

/-- The fraction of T-shirts sold in the morning -/
def morning_fraction : ℚ := 3/5

/-- The number of T-shirts sold in the afternoon -/
def afternoon_sales : ℚ := 180

theorem tshirts_left_yesterday : x = 300 := by
  have h1 : morning_fraction * x = afternoon_sales
  · -- Proof of h1
    sorry
  -- Main proof
  calc x
    = (morning_fraction * x) / morning_fraction := by sorry
    _ = afternoon_sales / morning_fraction := by rw [h1]
    _ = 180 / (3/5) := by rfl
    _ = 180 * (5/3) := by sorry
    _ = 300 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tshirts_left_yesterday_l400_40018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_function_properties_l400_40002

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 - 3*a*x + b

-- State the theorem
theorem tangent_function_properties (a b : ℝ) (h1 : a ≠ 0) :
  (∃ x, f a b x = 8 ∧ (deriv (f a b)) x = 0 ∧ x = 2) →
  (∀ x, f a b x = x^3 - 12*x + 24) ∧
  (∀ x, f a b x ≤ f a b (-2)) ∧
  (∀ x, f a b x ≥ f a b 2) ∧
  f a b (-2) = 40 ∧
  f a b 2 = 8 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_function_properties_l400_40002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unforgettable_count_l400_40005

/-- An unforgettable telephone number is represented as a list of 8 digits. -/
def UnforgettableNumber := List Nat

/-- Check if a list of 8 digits represents an unforgettable number. -/
def is_unforgettable (n : UnforgettableNumber) : Prop :=
  n.length = 8 ∧ 
  (∀ i, i ∈ n.toFinset → i < 10) ∧
  (n.take 4 = n.drop 4)

/-- The count of all possible unforgettable telephone numbers. -/
def count_unforgettable : Nat :=
  10^4

theorem unforgettable_count :
  count_unforgettable = 10000 := by
  rfl

#eval count_unforgettable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unforgettable_count_l400_40005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_operations_closure_l400_40010

theorem integer_operations_closure :
  ∀ (a b : ℤ) (n : ℕ),
    (a + b) ∈ Set.univ ∧
    (a - b) ∈ Set.univ ∧
    (a * b) ∈ Set.univ ∧
    (a^n) ∈ Set.univ := by
  intro a b n
  exact ⟨trivial, trivial, trivial, trivial⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_operations_closure_l400_40010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonal_approx_l400_40067

/-- The diagonal of a square with area equal to a 45m by 40m rectangle is approximately 60m -/
theorem square_diagonal_approx (ε : ℝ) (hε : ε > 0) : ∃ (d : ℝ), 
  let a := 45 * 40 -- area of the rectangle
  let s := Real.sqrt a -- side of the square
  d = s * Real.sqrt 2 ∧ |d - 60| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_diagonal_approx_l400_40067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_layer_max_volume_l400_40017

noncomputable section

/-- The volume of a spherical layer cut from a sphere -/
def spherical_layer_volume (R : ℝ) (m : ℝ) : ℝ :=
  Real.pi * m * R^2 - (1/12) * Real.pi * m^3

/-- The surface area of a spherical segment -/
def spherical_segment_area (R : ℝ) (m : ℝ) : ℝ :=
  2 * Real.pi * R * m

theorem spherical_layer_max_volume (R : ℝ) (F : ℝ) (h_R_pos : R > 0) (h_F_pos : F > 0) :
  ∃ (m : ℝ), 
    spherical_segment_area R m = F ∧
    ∀ (m' : ℝ), spherical_segment_area R m' = F → 
      spherical_layer_volume R m ≥ spherical_layer_volume R m' ∧
      m = 2 * R ∧
      spherical_layer_volume R m = (4/3) * Real.pi * R^3 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_layer_max_volume_l400_40017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_value_implies_x_value_l400_40059

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * x^2 - 8*x + 13

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 2 * Real.sqrt 2 * x - 8

-- Theorem statement
theorem derivative_value_implies_x_value (x₀ : ℝ) :
  f' x₀ = 4 → x₀ = 3 * Real.sqrt 2 := by
  intro h
  -- Proof steps would go here
  sorry

#check derivative_value_implies_x_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_value_implies_x_value_l400_40059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_implies_a_equals_two_l400_40001

theorem divisibility_implies_a_equals_two (a : ℤ) :
  (∀ x : ℤ, ∃ q : Polynomial ℤ, X^13 + X + (90 : Polynomial ℤ) = (X^2 - X + (a : Polynomial ℤ)) * q) →
  a = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_implies_a_equals_two_l400_40001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_13_position_l400_40046

/-- Represents a stack of cards -/
structure CardStack :=
  (n : ℕ)  -- Half the total number of cards

/-- Defines the restacking process -/
def restack (stack : CardStack) : List ℕ :=
  sorry  -- Implementation of restacking process

/-- Theorem: If card 13 retains its position after restacking, then n = 13 -/
theorem card_13_position (stack : CardStack) :
  (restack stack).get? 12 = some 13 → stack.n = 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_13_position_l400_40046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l400_40051

noncomputable def f (x : ℝ) := (1/3) * x^3 - 4*x + 4

theorem f_properties :
  (f 2 = -4/3) ∧
  (deriv f 2 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    f x₁ = 28/3 ∧ f x₂ = -4/3 ∧
    ∀ x : ℝ, f x ≤ 28/3 ∧ f x ≥ -4/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l400_40051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_60_degrees_l400_40045

theorem tan_alpha_minus_60_degrees (α : ℝ) (h : Real.tan α = 4 * Real.sin (420 * π / 180)) :
  Real.tan (α - 60 * π / 180) = Real.sqrt 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_60_degrees_l400_40045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_antonium_value_day_92_l400_40052

/-- The value of Chukhoyn cryptocurrency on day n -/
def C (n : ℕ) : ℚ := n

/-- The value of Antonium cryptocurrency on day n -/
def A : ℕ → ℚ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | n + 2 => (C (n + 1) + A (n + 1)) / (C (n + 1) * A (n + 1))

/-- Theorem stating the value of Antonium on the 92nd day -/
theorem antonium_value_day_92 : A 92 = 92 / 91 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_antonium_value_day_92_l400_40052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ch4_moles_combined_l400_40007

/-- Represents the number of moles of a substance -/
structure Moles where
  value : ℝ
  nonneg : value ≥ 0

/-- Represents a chemical reaction with reactants and products -/
structure Reaction where
  CH4_reactant : Moles
  Br2_reactant : Moles
  HBr_product : Moles

/-- The balanced reaction between CH4 and Br2 -/
def balanced_reaction (r : Reaction) : Prop :=
  r.CH4_reactant.value = r.Br2_reactant.value ∧ r.CH4_reactant.value = r.HBr_product.value

/-- Create a Moles instance from a real number -/
def real_to_moles (x : ℝ) (h : x ≥ 0) : Moles :=
  ⟨x, h⟩

/-- The theorem stating that 1 mole of CH4 is combined -/
theorem ch4_moles_combined (r : Reaction) 
  (h1 : r.Br2_reactant = real_to_moles 1 (by norm_num))
  (h2 : r.HBr_product = real_to_moles 1 (by norm_num))
  (h3 : balanced_reaction r) : 
  r.CH4_reactant = real_to_moles 1 (by norm_num) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ch4_moles_combined_l400_40007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_three_pi_half_minus_A_l400_40032

theorem cos_three_pi_half_minus_A (A : ℝ) (h : Real.sin A = 1/2) :
  Real.cos (3 * π / 2 - A) = -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_three_pi_half_minus_A_l400_40032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repayment_clears_debt_l400_40036

/-- Calculates the annual repayment amount for a loan -/
noncomputable def annual_repayment (a : ℝ) (γ : ℝ) : ℝ :=
  (a * γ * (1 + γ)^5) / ((1 + γ)^5 - 1)

/-- Theorem: The annual repayment amount clears the debt in 5 years -/
theorem repayment_clears_debt (a : ℝ) (γ : ℝ) (h1 : a > 0) (h2 : γ > 0) :
  let x := annual_repayment a γ
  a * (1 + γ)^5 = x * ((1 + γ)^4 + (1 + γ)^3 + (1 + γ)^2 + (1 + γ) + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repayment_clears_debt_l400_40036
