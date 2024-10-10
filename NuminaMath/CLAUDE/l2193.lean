import Mathlib

namespace tens_digit_of_7_pow_2005_l2193_219369

/-- The last two digits of 7^n follow a cycle of length 4 -/
def last_two_digits_cycle : List (Fin 100) := [7, 49, 43, 1]

/-- The tens digit of a number -/
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem tens_digit_of_7_pow_2005 :
  tens_digit (7^2005 % 100) = 0 :=
sorry

end tens_digit_of_7_pow_2005_l2193_219369


namespace conversation_on_thursday_l2193_219329

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents the library schedule -/
def LibrarySchedule := DayOfWeek → Bool

/-- Represents a boy's visiting schedule -/
def VisitSchedule := ℕ → DayOfWeek

/-- The number of days between visits for each boy -/
def visitIntervals : List ℕ := [2, 3, 4]

/-- The library is closed on Wednesdays -/
def libraryClosedWednesday (schedule : LibrarySchedule) : Prop :=
  schedule DayOfWeek.Wednesday = false

/-- All boys met again on a Monday -/
def metAgainOnMonday (schedules : List VisitSchedule) : Prop :=
  ∀ s ∈ schedules, ∃ n : ℕ, s n = DayOfWeek.Monday

/-- The conversation day is the same for all boys -/
def conversationDay (day : DayOfWeek) (schedules : List VisitSchedule) : Prop :=
  ∀ s ∈ schedules, ∃ n : ℕ, s n = day

/-- Main theorem: The conversation occurred on a Thursday -/
theorem conversation_on_thursday
  (library_schedule : LibrarySchedule)
  (boy_schedules : List VisitSchedule)
  (h_closed : libraryClosedWednesday library_schedule)
  (h_intervals : boy_schedules.length = visitIntervals.length)
  (h_monday : metAgainOnMonday boy_schedules)
  (h_adjust : ∀ s ∈ boy_schedules, ∀ n : ℕ,
    library_schedule (s n) = false → s (n + 1) = DayOfWeek.Thursday) :
  conversationDay DayOfWeek.Thursday boy_schedules :=
sorry

end conversation_on_thursday_l2193_219329


namespace min_value_reciprocal_sum_l2193_219389

theorem min_value_reciprocal_sum (m n : ℝ) (h1 : m + n = 2) (h2 : m * n > 0) :
  1 / m + 1 / n ≥ 2 ∧ (1 / m + 1 / n = 2 ↔ m = 1 ∧ n = 1) := by
  sorry

end min_value_reciprocal_sum_l2193_219389


namespace min_value_of_f_l2193_219319

theorem min_value_of_f (x : ℝ) (h : x ≥ 5/2) : (x^2 - 4*x + 5) / (2*x - 4) ≥ 1 := by
  sorry

end min_value_of_f_l2193_219319


namespace triangle_area_change_l2193_219343

theorem triangle_area_change (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (htri : a + b > c ∧ b + c > a ∧ c + a > b) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let a' := 2 * a
  let b' := 1.5 * b
  let c' := c
  let s' := (a' + b' + c') / 2
  let area' := Real.sqrt (s' * (s' - a') * (s' - b') * (s' - c'))
  2 * area < area' ∧ area' < 3 * area :=
by sorry

end triangle_area_change_l2193_219343


namespace intersection_of_lines_l2193_219349

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Represents a circle in 2D space -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- ABC is an acute-angled scalene triangle -/
def Triangle (A B C : Point) : Prop := sorry

/-- AH is an altitude of triangle ABC -/
def IsAltitude (H : Point) (A B C : Point) : Prop := sorry

/-- AM is a median of triangle ABC -/
def IsMedian (M : Point) (A B C : Point) : Prop := sorry

/-- O is the center of the circumscribed circle ω of triangle ABC -/
def IsCircumcenter (O : Point) (A B C : Point) (ω : Circle) : Prop := sorry

/-- Two lines intersect at a point -/
def Intersect (l1 l2 : Line) (P : Point) : Prop := sorry

/-- A line intersects a circle at a point -/
def IntersectCircle (l : Line) (c : Circle) (P : Point) : Prop := sorry

theorem intersection_of_lines 
  (A B C H M O D E F X Y : Point) 
  (ω : Circle) :
  Triangle A B C →
  IsAltitude H A B C →
  IsMedian M A B C →
  IsCircumcenter O A B C ω →
  Intersect (Line.mk 0 0 0) (Line.mk 0 0 0) D →  -- OH and AM
  Intersect (Line.mk 0 0 0) (Line.mk 0 0 0) E →  -- AB and CD
  Intersect (Line.mk 0 0 0) (Line.mk 0 0 0) F →  -- BD and AC
  IntersectCircle (Line.mk 0 0 0) ω X →  -- EH and ω
  IntersectCircle (Line.mk 0 0 0) ω Y →  -- FH and ω
  ∃ P, Intersect (Line.mk 0 0 0) (Line.mk 0 0 0) P ∧  -- BY and CX
      Intersect (Line.mk 0 0 0) (Line.mk 0 0 0) P ∧  -- CX and AH
      Intersect (Line.mk 0 0 0) (Line.mk 0 0 0) P    -- AH and BY
:= by sorry

end intersection_of_lines_l2193_219349


namespace midpoint_octagon_area_ratio_l2193_219359

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry -- Additional conditions to ensure the octagon is regular

/-- The octagon formed by joining the midpoints of a regular octagon's sides -/
def midpointOctagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

theorem midpoint_octagon_area_ratio (o : RegularOctagon) :
  area (midpointOctagon o) = (1 : ℝ) / 4 * area o := by
  sorry

end midpoint_octagon_area_ratio_l2193_219359


namespace equation_solution_l2193_219399

theorem equation_solution :
  ∃! x : ℝ, x ≠ 1 ∧ (3 * x) / (x - 1) = 2 + 1 / (x - 1) ∧ x = -1 := by
  sorry

end equation_solution_l2193_219399


namespace average_speed_two_hours_car_average_speed_l2193_219378

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) :
  speed1 > 0 →
  speed2 > 0 →
  (speed1 + speed2) / 2 = (speed1 * 1 + speed2 * 1) / (1 + 1) := by
  sorry

/-- The average speed of a car traveling 90 km/h for the first hour and 40 km/h for the second hour is 65 km/h -/
theorem car_average_speed :
  let speed1 : ℝ := 90
  let speed2 : ℝ := 40
  (speed1 + speed2) / 2 = 65 := by
  sorry

end average_speed_two_hours_car_average_speed_l2193_219378


namespace gcd_of_sums_of_squares_l2193_219345

theorem gcd_of_sums_of_squares : 
  Nat.gcd (118^2 + 227^2 + 341^2) (119^2 + 226^2 + 340^2) = 3 := by
  sorry

end gcd_of_sums_of_squares_l2193_219345


namespace fundraising_contribution_l2193_219346

theorem fundraising_contribution (total_amount : ℕ) (num_participants : ℕ) 
  (h1 : total_amount = 2400) (h2 : num_participants = 9) : 
  (total_amount + num_participants - 1) / num_participants = 267 :=
by
  sorry

#check fundraising_contribution

end fundraising_contribution_l2193_219346


namespace largest_integer_with_remainder_ninety_four_satisfies_conditions_ninety_four_is_largest_l2193_219326

theorem largest_integer_with_remainder (n : ℕ) : 
  n < 100 ∧ n % 6 = 4 → n ≤ 94 :=
by
  sorry

theorem ninety_four_satisfies_conditions : 
  94 < 100 ∧ 94 % 6 = 4 :=
by
  sorry

theorem ninety_four_is_largest : 
  ∀ m : ℕ, m < 100 ∧ m % 6 = 4 → m ≤ 94 :=
by
  sorry

end largest_integer_with_remainder_ninety_four_satisfies_conditions_ninety_four_is_largest_l2193_219326


namespace sachins_age_l2193_219379

theorem sachins_age (sachin_age rahul_age : ℕ) : 
  rahul_age = sachin_age + 14 →
  sachin_age * 9 = rahul_age * 7 →
  sachin_age = 49 := by
sorry

end sachins_age_l2193_219379


namespace no_real_roots_l2193_219311

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the quadratic equation
def quadratic_equation (t : Triangle) (x : ℝ) : Prop :=
  t.a^2 * x^2 + (t.b^2 - t.a^2 - t.c^2) * x + t.c^2 = 0

-- Theorem statement
theorem no_real_roots (t : Triangle) : ¬∃ x : ℝ, quadratic_equation t x := by
  sorry

end no_real_roots_l2193_219311


namespace tv_price_changes_l2193_219309

theorem tv_price_changes (P : ℝ) (P_positive : P > 0) :
  let price_after_changes := P * 1.30 * 1.20 * 0.90 * 1.15
  let single_increase := 1.6146
  price_after_changes = P * single_increase :=
by sorry

end tv_price_changes_l2193_219309


namespace painters_work_days_theorem_l2193_219364

/-- The number of work-days required for a given number of painters to complete a job,
    assuming the product of painters and work-days is constant. -/
def work_days (painters : ℕ) (total_work : ℚ) : ℚ :=
  total_work / painters

theorem painters_work_days_theorem (total_work : ℚ) :
  let five_painters_days : ℚ := 3/2
  let four_painters_days : ℚ := work_days 4 (5 * five_painters_days)
  four_painters_days = 15/8 := by sorry

end painters_work_days_theorem_l2193_219364


namespace first_day_rain_l2193_219315

/-- The amount of rain Greg experienced while camping, given the known conditions -/
def camping_rain (first_day : ℝ) : ℝ := first_day + 6 + 5

/-- The amount of rain at Greg's house during the same week -/
def house_rain : ℝ := 26

/-- The difference in rain between Greg's house and his camping experience -/
def rain_difference : ℝ := 12

theorem first_day_rain : 
  ∃ (x : ℝ), camping_rain x = house_rain - rain_difference ∧ x = 3 :=
sorry

end first_day_rain_l2193_219315


namespace geometric_sequence_fourth_term_l2193_219318

theorem geometric_sequence_fourth_term 
  (a : ℝ) -- first term
  (h1 : a ≠ 0) -- ensure first term is non-zero for division
  (h2 : (3*a + 3) / a = (6*a + 6) / (3*a + 3)) -- condition for geometric sequence
  (h3 : 3*a + 3 = a * ((3*a + 3) / a)) -- second term definition
  (h4 : 6*a + 6 = (3*a + 3) * ((3*a + 3) / a)) -- third term definition
  : a * ((3*a + 3) / a)^3 = -24 := by
  sorry


end geometric_sequence_fourth_term_l2193_219318


namespace shane_photos_l2193_219393

theorem shane_photos (total_photos : ℕ) (jan_photos_per_day : ℕ) (jan_days : ℕ) (feb_weeks : ℕ)
  (h1 : total_photos = 146)
  (h2 : jan_photos_per_day = 2)
  (h3 : jan_days = 31)
  (h4 : feb_weeks = 4) :
  (total_photos - jan_photos_per_day * jan_days) / feb_weeks = 21 := by
  sorry

end shane_photos_l2193_219393


namespace barbie_earrings_problem_l2193_219382

theorem barbie_earrings_problem (barbie_earrings : ℕ) 
  (h1 : barbie_earrings % 2 = 0)  -- Ensures barbie_earrings is even
  (h2 : ∃ (alissa_given : ℕ), alissa_given = barbie_earrings / 2)
  (h3 : ∃ (alissa_total : ℕ), alissa_total = 3 * (barbie_earrings / 2))
  (h4 : 3 * (barbie_earrings / 2) = 36) :
  barbie_earrings / 2 = 12 := by
  sorry

end barbie_earrings_problem_l2193_219382


namespace S_min_at_24_l2193_219335

/-- The sequence term a_n -/
def a (n : ℕ) : ℤ := 2 * n - 49

/-- The sum S_n of the first n terms of the sequence -/
def S (n : ℕ) : ℤ := n ^ 2 - 48 * n

/-- Theorem stating that S_n is minimized when n = 24 -/
theorem S_min_at_24 : ∀ n : ℕ, S 24 ≤ S n := by sorry

end S_min_at_24_l2193_219335


namespace sqrt_neg_x_squared_meaningful_l2193_219385

theorem sqrt_neg_x_squared_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = -x^2) ↔ x = 0 := by sorry

end sqrt_neg_x_squared_meaningful_l2193_219385


namespace line_l_standard_equation_l2193_219320

/-- A line in 2D space defined by parametric equations. -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The given parametric line. -/
def line_l : ParametricLine where
  x := fun t => 1 + t
  y := fun t => -1 + t

/-- The standard form of a line equation: ax + by + c = 0 -/
structure StandardLineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The proposed standard equation of the line. -/
def proposed_equation : StandardLineEquation where
  a := 1
  b := -1
  c := -2

theorem line_l_standard_equation :
  ∀ t : ℝ, proposed_equation.a * (line_l.x t) + proposed_equation.b * (line_l.y t) + proposed_equation.c = 0 := by
  sorry

end line_l_standard_equation_l2193_219320


namespace S_at_one_l2193_219365

def R (x : ℝ) : ℝ := 3 * x^3 - 5 * x + 4

def S (x : ℝ) : ℝ := (3 + 2) * x^3 + (-5 + 2) * x + (4 + 2)

theorem S_at_one : S 1 = 8 := by sorry

end S_at_one_l2193_219365


namespace cinema_renovation_unique_solution_l2193_219372

theorem cinema_renovation_unique_solution :
  ∃! (x y : ℕ), 
    x > 0 ∧ 
    y > 20 ∧ 
    y * (2 * x + y - 1) = 4008 := by
  sorry

end cinema_renovation_unique_solution_l2193_219372


namespace problem_statement_l2193_219338

theorem problem_statement (a b : ℝ) (h : a + b - 3 = 0) :
  2 * a^2 + 4 * a * b + 2 * b^2 - 6 = 12 := by
  sorry

end problem_statement_l2193_219338


namespace difference_in_combined_area_l2193_219322

/-- The difference in combined area (front and back) between two rectangular sheets of paper -/
theorem difference_in_combined_area : 
  let sheet1_length : ℝ := 11
  let sheet1_width : ℝ := 13
  let sheet2_length : ℝ := 6.5
  let sheet2_width : ℝ := 11
  let combined_area (l w : ℝ) := 2 * l * w
  combined_area sheet1_length sheet1_width - combined_area sheet2_length sheet2_width = 143
  := by sorry

end difference_in_combined_area_l2193_219322


namespace altitude_segment_length_l2193_219333

/-- An acute triangle with two altitudes dividing the sides -/
structure AcuteTriangleWithAltitudes where
  /-- The triangle is acute -/
  is_acute : Bool
  /-- Lengths of segments created by altitudes -/
  segment1 : ℝ
  segment2 : ℝ
  segment3 : ℝ
  segment4 : ℝ
  /-- Conditions on segment lengths -/
  h1 : segment1 = 6
  h2 : segment2 = 4
  h3 : segment3 = 3

/-- The theorem stating that the fourth segment length is 9/7 -/
theorem altitude_segment_length (t : AcuteTriangleWithAltitudes) : t.segment4 = 9/7 := by
  sorry

end altitude_segment_length_l2193_219333


namespace possible_m_values_l2193_219330

def A (m : ℝ) : Set ℝ := {1, 3, Real.sqrt m}
def B (m : ℝ) : Set ℝ := {1, m}

theorem possible_m_values (m : ℝ) : B m ⊆ A m → m = 0 ∨ m = 3 := by
  sorry

end possible_m_values_l2193_219330


namespace westward_movement_negative_l2193_219323

/-- Represents the direction of movement --/
inductive Direction
| East
| West

/-- Represents a movement with magnitude and direction --/
structure Movement where
  magnitude : ℝ
  direction : Direction

/-- Converts a movement to a signed real number --/
def Movement.toSignedReal (m : Movement) : ℝ :=
  match m.direction with
  | Direction.East => m.magnitude
  | Direction.West => -m.magnitude

theorem westward_movement_negative 
  (east_convention : Movement.toSignedReal { magnitude := 2, direction := Direction.East } = 2) :
  Movement.toSignedReal { magnitude := 3, direction := Direction.West } = -3 := by
  sorry

end westward_movement_negative_l2193_219323


namespace sand_remaining_proof_l2193_219377

/-- The amount of sand remaining on a truck after transit -/
def sand_remaining (initial : ℝ) (lost : ℝ) : ℝ :=
  initial - lost

/-- Theorem: The amount of sand remaining on the truck is 1.7 pounds -/
theorem sand_remaining_proof (initial : ℝ) (lost : ℝ) 
    (h1 : initial = 4.1)
    (h2 : lost = 2.4) : 
  sand_remaining initial lost = 1.7 := by
  sorry

end sand_remaining_proof_l2193_219377


namespace vampire_survival_l2193_219310

/-- The number of people a vampire needs to suck blood from each day to survive -/
def vampire_daily_victims : ℕ :=
  let gallons_per_week : ℕ := 7
  let pints_per_gallon : ℕ := 8
  let pints_per_person : ℕ := 2
  let days_per_week : ℕ := 7
  (gallons_per_week * pints_per_gallon) / (pints_per_person * days_per_week)

theorem vampire_survival : vampire_daily_victims = 4 := by
  sorry

end vampire_survival_l2193_219310


namespace total_points_theorem_l2193_219368

/-- The total number of participating teams -/
def num_teams : ℕ := 70

/-- The total number of points earned on question 33 -/
def points_q33 : ℕ := 3

/-- The total number of points earned on question 34 -/
def points_q34 : ℕ := 6

/-- The total number of points earned on question 35 -/
def points_q35 : ℕ := 4

/-- The total number of points A earned over all participating teams on questions 33, 34, and 35 -/
def A : ℕ := points_q33 + points_q34 + points_q35

theorem total_points_theorem : A = 13 := by sorry

end total_points_theorem_l2193_219368


namespace sqrt_6_simplest_l2193_219352

def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → y^2 = x → ¬∃ z : ℝ, z > 0 ∧ z < y ∧ z^2 = x

theorem sqrt_6_simplest :
  is_simplest_sqrt 6 ∧
  ¬is_simplest_sqrt (1/6) ∧
  ¬is_simplest_sqrt 0.6 ∧
  ¬is_simplest_sqrt 60 :=
sorry

end sqrt_6_simplest_l2193_219352


namespace equal_roots_quadratic_l2193_219375

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k = 0 ∧ (∀ y : ℝ, y^2 - 2*y + k = 0 → y = x)) → k = 1 := by
  sorry

end equal_roots_quadratic_l2193_219375


namespace integral_proof_l2193_219358

open Real

noncomputable def f (x : ℝ) : ℝ := 
  (1/2) * log (abs (x^2 + x + 1)) + 
  (1/sqrt 3) * arctan ((2*x + 1)/sqrt 3) + 
  (1/2) * log (abs (x^2 + 1))

theorem integral_proof (x : ℝ) : 
  deriv f x = (2*x^3 + 2*x^2 + 2*x + 1) / ((x^2 + x + 1) * (x^2 + 1)) :=
by sorry

end integral_proof_l2193_219358


namespace min_value_F_l2193_219316

/-- The function F(x, y) -/
def F (x y : ℝ) : ℝ := x^2 + 8*y + y^2 + 14*x - 6

/-- The constraint equation -/
def constraint (x y : ℝ) : Prop := x^2 + y^2 + 25 = 10*(x + y)

/-- Theorem stating that the minimum value of F(x, y) is 29 under the given constraint -/
theorem min_value_F :
  ∃ (m : ℝ), m = 29 ∧
  (∀ x y : ℝ, constraint x y → F x y ≥ m) ∧
  (∃ x y : ℝ, constraint x y ∧ F x y = m) :=
sorry

end min_value_F_l2193_219316


namespace expression_simplification_l2193_219327

theorem expression_simplification (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (m^2 - 1/n^2)^m * (n + 1/m)^(n-m) / ((n^2 - 1/m^2)^n * (m - 1/n)^(m-n)) = (m/n)^(m+n) := by
  sorry

end expression_simplification_l2193_219327


namespace function_inequality_implies_m_range_l2193_219304

/-- Given f(x) = |x+a| + |x-1/a| where a ≠ 0, if for all x ∈ ℝ, f(x) ≥ |m-1|, then m ∈ [-1, 3] -/
theorem function_inequality_implies_m_range (a m : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, |x + a| + |x - 1/a| ≥ |m - 1|) → m ∈ Set.Icc (-1 : ℝ) 3 :=
by sorry

end function_inequality_implies_m_range_l2193_219304


namespace min_dot_product_plane_vectors_l2193_219332

theorem min_dot_product_plane_vectors (a b : ℝ × ℝ) :
  ‖(2 • a) - b‖ ≤ 3 →
  ∀ (c d : ℝ × ℝ), ‖(2 • c) - d‖ ≤ 3 →
  a • b ≥ -9/8 ∧ a • b ≤ c • d :=
by sorry

end min_dot_product_plane_vectors_l2193_219332


namespace negation_equivalence_l2193_219371

theorem negation_equivalence :
  (¬ ∀ x : ℝ, |x - 2| + |x - 4| > 3) ↔ (∃ x₀ : ℝ, |x₀ - 2| + |x₀ - 4| ≤ 3) :=
by sorry

end negation_equivalence_l2193_219371


namespace delta_max_success_ratio_l2193_219306

/-- Represents a participant's score for a single day -/
structure DayScore where
  scored : ℕ
  attempted : ℕ

/-- Represents a participant's scores for two days -/
structure TwoDayScore where
  day1 : DayScore
  day2 : DayScore

def success_ratio (score : DayScore) : ℚ :=
  score.scored / score.attempted

def overall_success_ratio (score : TwoDayScore) : ℚ :=
  (score.day1.scored + score.day2.scored) / (score.day1.attempted + score.day2.attempted)

theorem delta_max_success_ratio 
  (gamma : TwoDayScore)
  (delta : TwoDayScore)
  (h1 : gamma.day1 = ⟨210, 350⟩)
  (h2 : gamma.day2 = ⟨150, 250⟩)
  (h3 : delta.day1.attempted + delta.day2.attempted = 600)
  (h4 : delta.day1.attempted ≠ 350)
  (h5 : delta.day1.scored > 0 ∧ delta.day2.scored > 0)
  (h6 : success_ratio delta.day1 < success_ratio gamma.day1)
  (h7 : success_ratio delta.day2 < success_ratio gamma.day2)
  (h8 : overall_success_ratio gamma = 3/5) :
  overall_success_ratio delta ≤ 359/600 := by
sorry

end delta_max_success_ratio_l2193_219306


namespace function_inequality_l2193_219308

/-- Given a function f(x) = axe^x where a ≠ 0 and a ≥ 4/e^2, 
    prove that f(x)/(x+1) - (x+1)ln(x) > 0 for x > 0 -/
theorem function_inequality (a : ℝ) (h1 : a ≠ 0) (h2 : a ≥ 4 / Real.exp 2) :
  ∀ x > 0, (a * x * Real.exp x) / (x + 1) - (x + 1) * Real.log x > 0 := by
  sorry

end function_inequality_l2193_219308


namespace initial_student_count_l2193_219381

theorem initial_student_count (initial_avg : ℝ) (new_avg : ℝ) (new_student_weight : ℝ) :
  initial_avg = 15 →
  new_avg = 14.9 →
  new_student_weight = 13 →
  ∃ n : ℕ, n * initial_avg + new_student_weight = (n + 1) * new_avg ∧ n = 19 :=
by
  sorry

end initial_student_count_l2193_219381


namespace max_t_value_l2193_219356

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) * m * x^2 - 2*x + Real.log (x + 1)

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f m x - Real.log (x + 1) + x^3

theorem max_t_value (m : ℝ) (t : ℝ) :
  m ∈ Set.Icc (-4) (-1) →
  (∀ x ∈ Set.Icc 1 t, g m x ≤ g m 1) →
  t ≤ (1 + Real.sqrt 13) / 2 :=
by sorry

end max_t_value_l2193_219356


namespace quadratic_value_at_zero_l2193_219363

-- Define the quadratic function
def f (h : ℝ) (x : ℝ) : ℝ := -(x + h)^2

-- Define the axis of symmetry
def axis_of_symmetry (h : ℝ) : ℝ := -3

-- Theorem statement
theorem quadratic_value_at_zero (h : ℝ) : 
  axis_of_symmetry h = -3 → f h 0 = -9 := by
  sorry

#check quadratic_value_at_zero

end quadratic_value_at_zero_l2193_219363


namespace condition_neither_sufficient_nor_necessary_l2193_219362

theorem condition_neither_sufficient_nor_necessary :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * b ≥ ((a + b) / 2)^2) ∧
  (∃ a b : ℝ, a * b < ((a + b) / 2)^2 ∧ (a ≤ 0 ∨ b ≤ 0)) := by sorry

end condition_neither_sufficient_nor_necessary_l2193_219362


namespace average_of_x_and_y_l2193_219317

theorem average_of_x_and_y (x y : ℝ) : 
  (2 + 6 + 10 + x + y) / 5 = 18 → (x + y) / 2 = 36 := by
  sorry

end average_of_x_and_y_l2193_219317


namespace sally_bought_three_frames_l2193_219374

/-- The number of photograph frames Sally bought -/
def frames_bought (frame_cost change_received total_paid : ℕ) : ℕ :=
  (total_paid - change_received) / frame_cost

/-- Theorem stating that Sally bought 3 photograph frames -/
theorem sally_bought_three_frames :
  frames_bought 3 11 20 = 3 := by
  sorry

end sally_bought_three_frames_l2193_219374


namespace function_symmetry_l2193_219331

-- Define the function f and constant T
variable (f : ℝ → ℝ) (T : ℝ)

-- Define the conditions
def periodic : Prop := ∀ x, f (x + 2 * T) = f x
def symmetry_1 : Prop := ∀ x, T / 2 ≤ x → x ≤ T → f x = f (T - x)
def antisymmetry : Prop := ∀ x, T ≤ x → x ≤ 3 * T / 2 → f x = -f (x - T)
def symmetry_2 : Prop := ∀ x, 3 * T / 2 ≤ x → x ≤ 2 * T → f x = -f (2 * T - x)

-- State the theorem
theorem function_symmetry 
  (h1 : periodic f T) 
  (h2 : symmetry_1 f T) 
  (h3 : antisymmetry f T) 
  (h4 : symmetry_2 f T) : 
  ∀ x, f x = f (T - x) := by
  sorry

end function_symmetry_l2193_219331


namespace simplify_expression_l2193_219386

theorem simplify_expression (a : ℝ) (h : a ≤ (1/2 : ℝ)) :
  Real.sqrt (1 - 4*a + 4*a^2) + |2*a - 1| = 2 - 4*a := by
  sorry

end simplify_expression_l2193_219386


namespace cubic_function_extrema_l2193_219351

def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 27

def f_derivative (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_function_extrema (a b : ℝ) :
  f_derivative a b (-1) = 0 ∧ f_derivative a b 3 = 0 → a = -3 ∧ b = -9 := by
  sorry

end cubic_function_extrema_l2193_219351


namespace gcd_2352_1560_l2193_219354

theorem gcd_2352_1560 : Nat.gcd 2352 1560 = 24 := by
  sorry

end gcd_2352_1560_l2193_219354


namespace complex_number_location_l2193_219366

theorem complex_number_location :
  let z : ℂ := Complex.I * (1 + Complex.I)
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end complex_number_location_l2193_219366


namespace money_distribution_l2193_219301

/-- Given a distribution of money in the ratio 3 : 5 : 7 among three people,
    where the second person's share is 1500, 
    the difference between the first and third person's shares is 1200. -/
theorem money_distribution (total : ℕ) (f v r : ℕ) : 
  (f + v + r = total) →
  (3 * v = 5 * f) →
  (5 * r = 7 * v) →
  (v = 1500) →
  (r - f = 1200) :=
by sorry

end money_distribution_l2193_219301


namespace rectangular_to_polar_conversion_l2193_219314

theorem rectangular_to_polar_conversion :
  let x : ℝ := Real.sqrt 3
  let y : ℝ := -1
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 2 * Real.pi - Real.arctan (1 / Real.sqrt 3)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ r = 2 ∧ θ = 11 * Real.pi / 6 :=
by
  sorry

#check rectangular_to_polar_conversion

end rectangular_to_polar_conversion_l2193_219314


namespace inequality_proof_l2193_219337

theorem inequality_proof (x : ℝ) (h : x > 0) : x^8 - x^5 - 1/x + 1/x^4 ≥ 0 := by
  sorry

end inequality_proof_l2193_219337


namespace four_sacks_filled_l2193_219303

/-- Calculates the number of sacks filled given the total pieces of wood and capacity per sack -/
def sacks_filled (total_wood : ℕ) (wood_per_sack : ℕ) : ℕ :=
  total_wood / wood_per_sack

/-- Theorem: Given 80 pieces of wood and sacks that can hold 20 pieces each, 4 sacks will be filled -/
theorem four_sacks_filled : sacks_filled 80 20 = 4 := by
  sorry

end four_sacks_filled_l2193_219303


namespace remaining_savings_l2193_219347

def initial_savings : ℕ := 80
def earrings_cost : ℕ := 23
def necklace_cost : ℕ := 48

theorem remaining_savings : 
  initial_savings - (earrings_cost + necklace_cost) = 9 := by sorry

end remaining_savings_l2193_219347


namespace difference_of_squares_l2193_219390

theorem difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by sorry

end difference_of_squares_l2193_219390


namespace sheila_attend_probability_l2193_219355

/-- The probability of rain -/
def prob_rain : ℝ := 0.3

/-- The probability of cloudy weather -/
def prob_cloudy : ℝ := 0.4

/-- The probability of sunshine -/
def prob_sunny : ℝ := 0.3

/-- The probability Sheila attends if it rains -/
def prob_attend_rain : ℝ := 0.25

/-- The probability Sheila attends if it's cloudy -/
def prob_attend_cloudy : ℝ := 0.5

/-- The probability Sheila attends if it's sunny -/
def prob_attend_sunny : ℝ := 0.75

/-- The theorem stating the probability of Sheila attending the picnic -/
theorem sheila_attend_probability : 
  prob_rain * prob_attend_rain + prob_cloudy * prob_attend_cloudy + prob_sunny * prob_attend_sunny = 0.5 := by
  sorry

end sheila_attend_probability_l2193_219355


namespace thirteen_sided_polygon_diagonals_l2193_219348

/-- The number of diagonals in a polygon with n sides. -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals connected to a single vertex in a polygon with n sides. -/
def diagonals_per_vertex (n : ℕ) : ℕ := n - 3

/-- The number of diagonals in a polygon with n sides where one vertex is not connected to any diagonal. -/
def diagonals_with_disconnected_vertex (n : ℕ) : ℕ :=
  diagonals n - diagonals_per_vertex n

theorem thirteen_sided_polygon_diagonals :
  diagonals_with_disconnected_vertex 13 = 55 := by
  sorry

#eval diagonals_with_disconnected_vertex 13

end thirteen_sided_polygon_diagonals_l2193_219348


namespace non_monotonic_function_parameter_range_l2193_219367

/-- The function f(x) = (1/3)x^3 - x^2 + ax - 5 is not monotonic in the interval [-1, 2] -/
theorem non_monotonic_function_parameter_range (a : ℝ) : 
  (∃ x y, x ∈ Set.Icc (-1 : ℝ) 2 ∧ y ∈ Set.Icc (-1 : ℝ) 2 ∧ x < y ∧ 
    ((1/3)*x^3 - x^2 + a*x) > ((1/3)*y^3 - y^2 + a*y)) ↔ 
  a ∈ Set.Ioo (-3 : ℝ) 1 :=
by sorry

end non_monotonic_function_parameter_range_l2193_219367


namespace max_value_DEABC_l2193_219398

/-- Represents a single-digit number -/
def SingleDigit := {n : ℕ // n < 10}

/-- Converts a three-digit number represented by its digits to a natural number -/
def threeDigitToNat (a b c : SingleDigit) : ℕ := 100 * a.val + 10 * b.val + c.val

/-- Converts a two-digit number represented by its digits to a natural number -/
def twoDigitToNat (d e : SingleDigit) : ℕ := 10 * d.val + e.val

/-- Converts a five-digit number represented by its digits to a natural number -/
def fiveDigitToNat (d e a b c : SingleDigit) : ℕ := 
  10000 * d.val + 1000 * e.val + 100 * a.val + 10 * b.val + c.val

theorem max_value_DEABC 
  (A B C D E : SingleDigit)
  (h1 : twoDigitToNat D E = A.val + B.val + C.val)
  (h2 : threeDigitToNat A B C + threeDigitToNat B C A + threeDigitToNat C A B + twoDigitToNat D E = 2016) :
  (∀ A' B' C' D' E', 
    twoDigitToNat D' E' = A'.val + B'.val + C'.val →
    threeDigitToNat A' B' C' + threeDigitToNat B' C' A' + threeDigitToNat C' A' B' + twoDigitToNat D' E' = 2016 →
    fiveDigitToNat D' E' A' B' C' ≤ fiveDigitToNat D E A B C) →
  fiveDigitToNat D E A B C = 18783 :=
sorry

end max_value_DEABC_l2193_219398


namespace hyperbola_eccentricity_l2193_219387

/-- The eccentricity of a hyperbola with equation x^2 - y^2/4 = 1 is √5 -/
theorem hyperbola_eccentricity :
  let a : ℝ := 1  -- semi-major axis
  let b : ℝ := 2  -- semi-minor axis
  let c : ℝ := Real.sqrt (a^2 + b^2)  -- distance from center to focus
  let e : ℝ := c / a  -- eccentricity
  e = Real.sqrt 5 := by sorry

end hyperbola_eccentricity_l2193_219387


namespace examination_student_count_l2193_219357

/-- The total number of students who appeared for the examination -/
def total_students : ℕ := 740

/-- The number of students who failed the examination -/
def failed_students : ℕ := 481

/-- The proportion of students who passed the examination -/
def pass_rate : ℚ := 35 / 100

theorem examination_student_count : 
  total_students = failed_students / (1 - pass_rate) := by
  sorry

end examination_student_count_l2193_219357


namespace positive_sum_and_product_iff_both_positive_l2193_219342

theorem positive_sum_and_product_iff_both_positive (a b : ℝ) :
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := by
  sorry

end positive_sum_and_product_iff_both_positive_l2193_219342


namespace remainder_sum_mod_seven_l2193_219397

theorem remainder_sum_mod_seven : (9^6 + 8^8 + 7^9) % 7 = 2 := by
  sorry

end remainder_sum_mod_seven_l2193_219397


namespace russian_football_championship_l2193_219305

/-- Represents a football championship. -/
structure Championship where
  teams : Nat
  matches_per_pair : Nat

/-- Calculate the number of matches a single team plays in a season. -/
def matches_per_team (c : Championship) : Nat :=
  (c.teams - 1) * c.matches_per_pair

/-- Calculate the total number of matches in a season. -/
def total_matches (c : Championship) : Nat :=
  (c.teams * (c.teams - 1) * c.matches_per_pair) / 2

/-- Theorem stating the number of matches for a single team and total matches in the championship. -/
theorem russian_football_championship 
  (c : Championship) 
  (h1 : c.teams = 16) 
  (h2 : c.matches_per_pair = 2) : 
  matches_per_team c = 30 ∧ total_matches c = 240 := by
  sorry

#eval matches_per_team ⟨16, 2⟩
#eval total_matches ⟨16, 2⟩

end russian_football_championship_l2193_219305


namespace horner_v3_equals_16_l2193_219353

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 3x^6 - 5x^4 + 2x^3 - x^2 + 2x + 1 -/
def f (x : ℝ) : ℝ :=
  3 * x^6 - 5 * x^4 + 2 * x^3 - x^2 + 2 * x + 1

/-- v3 in Horner's method for f(x) -/
def v3 (x : ℝ) : ℝ :=
  (((3 * x - 0) * x - 5) * x + 2)

theorem horner_v3_equals_16 :
  v3 2 = 16 := by sorry

end horner_v3_equals_16_l2193_219353


namespace exactly_one_divisible_l2193_219373

theorem exactly_one_divisible (p a b c d : ℕ) : 
  Prime p → 
  p % 2 = 1 →
  0 < a → a < p →
  0 < b → b < p →
  0 < c → c < p →
  0 < d → d < p →
  p ∣ (a^2 + b^2) →
  p ∣ (c^2 + d^2) →
  (p ∣ (a*c + b*d) ∧ ¬(p ∣ (a*d + b*c))) ∨ (¬(p ∣ (a*c + b*d)) ∧ p ∣ (a*d + b*c)) :=
by sorry

end exactly_one_divisible_l2193_219373


namespace tan_neg_alpha_problem_l2193_219324

theorem tan_neg_alpha_problem (α : Real) (h : Real.tan (-α) = -2) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 ∧ Real.sin (2 * α) = 4/5 := by
  sorry

end tan_neg_alpha_problem_l2193_219324


namespace cone_ratio_l2193_219383

theorem cone_ratio (circumference : ℝ) (volume : ℝ) :
  circumference = 28 * Real.pi →
  volume = 441 * Real.pi →
  ∃ (radius height : ℝ),
    circumference = 2 * Real.pi * radius ∧
    volume = (1/3) * Real.pi * radius^2 * height ∧
    radius / height = 14 / 9 :=
by sorry

end cone_ratio_l2193_219383


namespace red_ball_probability_l2193_219336

/-- The probability of drawing a red ball from a bag with 1 red ball and 4 white balls is 0.2 -/
theorem red_ball_probability (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  total_balls = red_balls + white_balls →
  red_balls = 1 →
  white_balls = 4 →
  (red_balls : ℚ) / total_balls = 1 / 5 := by
  sorry

end red_ball_probability_l2193_219336


namespace yard_length_with_32_trees_l2193_219325

/-- The length of a yard with equally spaced trees -/
def yardLength (numTrees : ℕ) (distanceBetweenTrees : ℕ) : ℕ :=
  (numTrees - 1) * distanceBetweenTrees

/-- Theorem: The length of a yard with 32 equally spaced trees and 14 meters between consecutive trees is 434 meters -/
theorem yard_length_with_32_trees : yardLength 32 14 = 434 := by
  sorry

end yard_length_with_32_trees_l2193_219325


namespace cube_cut_surface_area_l2193_219350

/-- Calculates the total surface area of small blocks after cutting a cube -/
def total_surface_area (edge_length : ℝ) (horizontal_cuts : ℕ) (vertical_cuts : ℕ) : ℝ :=
  let original_surface_area := 6 * edge_length^2
  let horizontal_new_area := 2 * edge_length^2 * (2 * horizontal_cuts)
  let vertical_new_area := 2 * edge_length^2 * (2 * vertical_cuts)
  original_surface_area + horizontal_new_area + vertical_new_area

/-- Theorem: The total surface area of all small blocks after cutting a cube with edge length 2,
    4 horizontal cuts, and 5 vertical cuts, is equal to 96 square units -/
theorem cube_cut_surface_area :
  total_surface_area 2 4 5 = 96 := by sorry

end cube_cut_surface_area_l2193_219350


namespace product_abcde_l2193_219380

theorem product_abcde (a b c d e : ℚ) : 
  3 * a + 4 * b + 6 * c + 8 * d + 10 * e = 55 →
  4 * (d + c + e) = b →
  4 * b + 2 * c = a →
  c - 2 = d →
  d + 1 = e →
  a * b * c * d * e = -1912397372 / 78364164096 := by
sorry

end product_abcde_l2193_219380


namespace parabola_equation_l2193_219334

-- Define a parabola
def Parabola (a b c : ℝ) := {(x, y) : ℝ × ℝ | y = a * x^2 + b * x + c}

-- Define the properties of our specific parabola
def ParabolaProperties (p : Set (ℝ × ℝ)) :=
  ∃ a : ℝ, a ≠ 0 ∧ 
  p = Parabola 0 0 0 ∧  -- vertex at origin
  (∀ x y : ℝ, (x, y) ∈ p → (x, y) ∈ p) ∧  -- y-axis symmetry
  (-4, -2) ∈ p  -- passes through (-4, -2)

-- Theorem statement
theorem parabola_equation :
  ∃ p : Set (ℝ × ℝ), ParabolaProperties p ∧ p = {(x, y) : ℝ × ℝ | x^2 = -8*y} :=
sorry

end parabola_equation_l2193_219334


namespace xy_max_value_l2193_219340

theorem xy_max_value (x y : ℝ) (h : x^2 + 2*y^2 - 2*x*y = 4) :
  x*y ≤ 2*Real.sqrt 2 + 2 := by
sorry

end xy_max_value_l2193_219340


namespace kitae_pencils_l2193_219370

def total_pens : ℕ := 12
def pencil_cost : ℕ := 1000
def pen_cost : ℕ := 1300
def total_spent : ℕ := 15000

theorem kitae_pencils (pencils : ℕ) (pens : ℕ) 
  (h1 : pencils + pens = total_pens)
  (h2 : pencil_cost * pencils + pen_cost * pens = total_spent) :
  pencils = 2 := by
  sorry

end kitae_pencils_l2193_219370


namespace min_cards_for_even_product_l2193_219302

def is_even (n : Nat) : Bool := n % 2 = 0

theorem min_cards_for_even_product :
  ∀ (S : Finset Nat),
  (∀ n ∈ S, 1 ≤ n ∧ n ≤ 16) →
  (Finset.card S = 16) →
  (∃ (T : Finset Nat), T ⊆ S ∧ Finset.card T = 9 ∧ ∃ n ∈ T, is_even n) ∧
  (∀ (U : Finset Nat), U ⊆ S → Finset.card U < 9 → ∀ n ∈ U, ¬is_even n) :=
by sorry

end min_cards_for_even_product_l2193_219302


namespace sum_of_number_and_reverse_is_99_l2193_219312

/-- Definition of a two-digit number -/
def TwoDigitNumber (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9

/-- The property that the difference between the number and its reverse
    is 7 times the sum of its digits -/
def SatisfiesEquation (a b : ℕ) : Prop :=
  (10 * a + b) - (10 * b + a) = 7 * (a + b)

/-- Theorem stating that for a two-digit number satisfying the given equation,
    the sum of the number and its reverse is 99 -/
theorem sum_of_number_and_reverse_is_99 (a b : ℕ) 
  (h1 : TwoDigitNumber a b) (h2 : SatisfiesEquation a b) : 
  (10 * a + b) + (10 * b + a) = 99 := by
  sorry

#check sum_of_number_and_reverse_is_99

end sum_of_number_and_reverse_is_99_l2193_219312


namespace renovation_profit_threshold_l2193_219395

/-- Annual profit without renovation (in millions of yuan) -/
def a (n : ℕ) : ℚ := 500 - 20 * n

/-- Annual profit with renovation (in millions of yuan) -/
def b (n : ℕ) : ℚ := 1000 - 1000 / (2^n)

/-- Cumulative profit without renovation (in millions of yuan) -/
def A (n : ℕ) : ℚ := 500 * n - 10 * n * (n + 1)

/-- Cumulative profit with renovation (in millions of yuan) -/
def B (n : ℕ) : ℚ := 1000 * n - 2600 + 2000 / (2^n)

/-- The minimum number of years for cumulative profit with renovation to exceed that without renovation -/
theorem renovation_profit_threshold : 
  ∀ n : ℕ, n ≥ 5 ↔ B n > A n :=
by sorry

end renovation_profit_threshold_l2193_219395


namespace mary_book_count_l2193_219341

def book_count (initial : ℕ) (book_club : ℕ) (lent_jane : ℕ) (returned_alice : ℕ)
  (bought_5th_month : ℕ) (bought_yard_sales : ℕ) (birthday_daughter : ℕ)
  (birthday_mother : ℕ) (from_sister : ℕ) (buy_one_get_one : ℕ)
  (donated_charity : ℕ) (borrowed_neighbor : ℕ) (sold_used : ℕ) : ℕ :=
  initial + book_club - lent_jane + returned_alice + bought_5th_month +
  bought_yard_sales + birthday_daughter + birthday_mother + from_sister +
  buy_one_get_one - donated_charity - borrowed_neighbor - sold_used

theorem mary_book_count :
  book_count 200 (2 * 12) 10 5 15 8 1 8 6 4 30 5 7 = 219 := by
  sorry

end mary_book_count_l2193_219341


namespace fuel_station_problem_l2193_219394

/-- Fuel station problem -/
theorem fuel_station_problem 
  (service_cost : ℝ) 
  (fuel_cost_per_liter : ℝ) 
  (total_cost : ℝ) 
  (minivan_tank : ℝ) 
  (truck_tank_ratio : ℝ) 
  (num_trucks : ℕ) 
  (h1 : service_cost = 2.30)
  (h2 : fuel_cost_per_liter = 0.70)
  (h3 : total_cost = 396)
  (h4 : minivan_tank = 65)
  (h5 : truck_tank_ratio = 2.20)
  (h6 : num_trucks = 2) :
  ∃ (num_minivans : ℕ), 
    (num_minivans : ℝ) * (service_cost + minivan_tank * fuel_cost_per_liter) + 
    (num_trucks : ℝ) * (service_cost + truck_tank_ratio * minivan_tank * fuel_cost_per_liter) = 
    total_cost ∧ num_minivans = 4 := by
  sorry

end fuel_station_problem_l2193_219394


namespace second_number_calculation_l2193_219300

theorem second_number_calculation (A B : ℝ) : 
  A = 3200 → 
  0.1 * A = 0.2 * B + 190 → 
  B = 650 := by
sorry

end second_number_calculation_l2193_219300


namespace cubic_equation_transformation_l2193_219388

theorem cubic_equation_transformation (p q r : ℝ) : 
  (p^3 - 5*p^2 + 6*p - 7 = 0) → 
  (q^3 - 5*q^2 + 6*q - 7 = 0) → 
  (r^3 - 5*r^2 + 6*r - 7 = 0) → 
  (∀ x : ℝ, x^3 - 10*x^2 + 25*x + 105 = 0 ↔ 
    (x = (p + q + r)/(p - 1) ∨ x = (p + q + r)/(q - 1) ∨ x = (p + q + r)/(r - 1))) :=
by sorry

end cubic_equation_transformation_l2193_219388


namespace middle_group_frequency_l2193_219396

/-- Represents a frequency distribution histogram -/
structure FrequencyHistogram where
  num_rectangles : ℕ
  sample_size : ℕ
  middle_area : ℝ
  other_areas : ℝ

/-- Theorem: The frequency of the middle group in a specific histogram -/
theorem middle_group_frequency (h : FrequencyHistogram) 
  (h_num_rectangles : h.num_rectangles = 11)
  (h_area_equality : h.middle_area = h.other_areas)
  (h_sample_size : h.sample_size = 160) :
  (h.middle_area / (h.middle_area + h.other_areas)) * h.sample_size = 80 := by
  sorry

#check middle_group_frequency

end middle_group_frequency_l2193_219396


namespace average_marks_abcd_l2193_219339

theorem average_marks_abcd (a b c d e : ℝ) : 
  ((a + b + c) / 3 = 48) →
  ((b + c + d + e) / 4 = 48) →
  (e = d + 3) →
  (a = 43) →
  ((a + b + c + d) / 4 = 47) :=
by sorry

end average_marks_abcd_l2193_219339


namespace roots_sum_square_l2193_219307

/-- Given that α and β are the two roots of the equation x^2 - 7x + 3 = 0 and α > β,
    prove that α^2 + 7β = 46 -/
theorem roots_sum_square (α β : ℝ) : 
  α^2 - 7*α + 3 = 0 → 
  β^2 - 7*β + 3 = 0 → 
  α > β →
  α^2 + 7*β = 46 := by
sorry

end roots_sum_square_l2193_219307


namespace curve_intersection_arithmetic_sequence_l2193_219361

/-- Given a curve C: y = 1/x (x > 0) and points A₁(x₁, 0) and A₂(x₂, 0) where x₂ > x₁ > 0,
    perpendicular lines to the x-axis from A₁ and A₂ intersect C at B₁ and B₂.
    The line B₁B₂ intersects the x-axis at A₃(x₃, 0).
    This theorem proves that x₁, x₃/2, x₂ form an arithmetic sequence. -/
theorem curve_intersection_arithmetic_sequence
  (x₁ x₂ : ℝ)
  (h₁ : 0 < x₁)
  (h₂ : x₁ < x₂)
  (x₃ : ℝ)
  (h₃ : x₃ = x₁ + x₂) :
  x₂ - x₃/2 = x₃/2 - x₁ :=
by sorry

end curve_intersection_arithmetic_sequence_l2193_219361


namespace no_special_polyhedron_l2193_219392

-- Define a polyhedron structure
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangle_faces : ℕ
  pentagon_faces : ℕ
  even_degree_vertices : ℕ

-- Define the conditions for our specific polyhedron
def SpecialPolyhedron (p : Polyhedron) : Prop :=
  p.faces = p.triangle_faces + p.pentagon_faces ∧
  p.pentagon_faces = 1 ∧
  p.vertices = p.even_degree_vertices ∧
  p.vertices - p.edges + p.faces = 2 ∧  -- Euler's formula
  3 * p.triangle_faces + 5 * p.pentagon_faces = 2 * p.edges

-- Theorem stating that such a polyhedron does not exist
theorem no_special_polyhedron :
  ¬ ∃ (p : Polyhedron), SpecialPolyhedron p :=
sorry

end no_special_polyhedron_l2193_219392


namespace lg_sum_equals_two_l2193_219344

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_sum_equals_two : lg 4 + lg 25 = 2 := by sorry

end lg_sum_equals_two_l2193_219344


namespace x1_range_l2193_219384

/-- The function f as defined in the problem -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp (x + 1) - Real.exp x + x^2 + 2 * m * (x - 1)

/-- The theorem stating the range of x1 -/
theorem x1_range (m : ℝ) (hm : m > 0) :
  {x1 : ℝ | ∀ x2, x1 + x2 = 1 → f m x1 ≥ f m x2} = Set.Ici (1/2) :=
sorry

end x1_range_l2193_219384


namespace a3_greater_b3_l2193_219321

/-- Two sequences satisfying the given conditions -/
def sequences (a b : ℕ+ → ℝ) : Prop :=
  (∀ n, a n + b n = 700) ∧
  (∀ n, a (n + 1) = (7/10) * a n + (2/5) * b n) ∧
  (a 6 = 400)

/-- Theorem stating that a_3 > b_3 for sequences satisfying the given conditions -/
theorem a3_greater_b3 (a b : ℕ+ → ℝ) (h : sequences a b) : a 3 > b 3 := by
  sorry

end a3_greater_b3_l2193_219321


namespace triangle_side_sum_range_l2193_219328

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if (a-b)(sin A + sin B) = (c-b)sin C and a = √3, then 5 < b² + c² ≤ 6. -/
theorem triangle_side_sum_range (a b c A B C : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ -- Acute triangle
  A + B + C = π ∧ -- Sum of angles in a triangle
  (a - b) * (Real.sin A + Real.sin B) = (c - b) * Real.sin C ∧ -- Given condition
  a = Real.sqrt 3 → -- Given condition
  5 < b^2 + c^2 ∧ b^2 + c^2 ≤ 6 := by
sorry

end triangle_side_sum_range_l2193_219328


namespace sum_remainder_zero_l2193_219360

theorem sum_remainder_zero : (((7283 + 7284 + 7285 + 7286 + 7287) * 2) % 9) = 0 := by
  sorry

end sum_remainder_zero_l2193_219360


namespace teacher_arrangement_count_l2193_219376

/-- The number of female teachers -/
def num_female : ℕ := 2

/-- The number of male teachers -/
def num_male : ℕ := 4

/-- The number of female teachers per group -/
def female_per_group : ℕ := 1

/-- The number of male teachers per group -/
def male_per_group : ℕ := 2

/-- The total number of groups -/
def num_groups : ℕ := 2

theorem teacher_arrangement_count :
  (num_female.choose female_per_group) * (num_male.choose male_per_group) = 12 := by
  sorry

end teacher_arrangement_count_l2193_219376


namespace sarah_apple_ratio_l2193_219313

theorem sarah_apple_ratio : 
  let sarah_apples : ℕ := 45
  let brother_apples : ℕ := 9
  (sarah_apples : ℚ) / brother_apples = 5 := by sorry

end sarah_apple_ratio_l2193_219313


namespace bisecting_line_sum_l2193_219391

/-- Triangle PQR with vertices P(0, 10), Q(3, 0), and R(9, 0) -/
structure Triangle where
  P : ℝ × ℝ := (0, 10)
  Q : ℝ × ℝ := (3, 0)
  R : ℝ × ℝ := (9, 0)

/-- A line represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The line through Q that bisects the area of the triangle -/
def bisectingLine (t : Triangle) : Line :=
  sorry

theorem bisecting_line_sum (t : Triangle) :
  let l := bisectingLine t
  l.slope + l.yIntercept = -20/3 := by
  sorry

end bisecting_line_sum_l2193_219391
