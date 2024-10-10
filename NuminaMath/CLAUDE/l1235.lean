import Mathlib

namespace no_additional_savings_when_purchasing_together_l1235_123557

/-- Represents the store's window offer -/
structure WindowOffer where
  price : ℕ  -- Price per window
  buy : ℕ    -- Number of windows to buy
  free : ℕ   -- Number of free windows

/-- Calculates the cost for a given number of windows under the offer -/
def calculateCost (offer : WindowOffer) (windowsNeeded : ℕ) : ℕ :=
  let fullSets := windowsNeeded / (offer.buy + offer.free)
  let remainingWindows := windowsNeeded % (offer.buy + offer.free)
  fullSets * (offer.price * offer.buy) + min remainingWindows offer.buy * offer.price

/-- Theorem stating that there's no additional savings when purchasing together -/
theorem no_additional_savings_when_purchasing_together 
  (offer : WindowOffer)
  (daveWindows : ℕ)
  (dougWindows : ℕ) :
  offer.price = 150 ∧ 
  offer.buy = 6 ∧ 
  offer.free = 2 ∧
  daveWindows = 9 ∧
  dougWindows = 10 →
  (calculateCost offer daveWindows + calculateCost offer dougWindows) - 
  calculateCost offer (daveWindows + dougWindows) = 0 :=
by
  sorry

end no_additional_savings_when_purchasing_together_l1235_123557


namespace camp_attendance_l1235_123581

/-- The total number of kids in Lawrence county -/
def total_kids : ℕ := 1363293

/-- The number of kids who stay home -/
def kids_at_home : ℕ := 907611

/-- The number of kids who go to camp -/
def kids_at_camp : ℕ := total_kids - kids_at_home

theorem camp_attendance : kids_at_camp = 455682 := by
  sorry

end camp_attendance_l1235_123581


namespace inclination_angle_of_line_l1235_123564

/-- The inclination angle of a line with equation ax + by + c = 0 is the angle between the positive x-axis and the line. -/
def InclinationAngle (a b c : ℝ) : ℝ := sorry

/-- The line equation sqrt(3)x + y - 1 = 0 -/
def LineEquation (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 1 = 0

theorem inclination_angle_of_line :
  InclinationAngle (Real.sqrt 3) 1 (-1) = 2 * Real.pi / 3 := by sorry

end inclination_angle_of_line_l1235_123564


namespace inequality_solution_l1235_123529

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 8) ≥ 4 / 5) ↔ (x ≤ -8 ∨ (-2 ≤ x ∧ x ≤ 2)) :=
by sorry

end inequality_solution_l1235_123529


namespace problem_statement_l1235_123514

theorem problem_statement (x : ℝ) (h : x + 1/x = 7) : 
  (x - 3)^2 + 49/(x - 3)^2 = 23 :=
by sorry

end problem_statement_l1235_123514


namespace eva_marks_total_l1235_123534

/-- Represents Eva's marks in a single semester -/
structure SemesterMarks where
  maths : ℕ
  arts : ℕ
  science : ℕ

/-- Calculates the total marks for a semester -/
def totalMarks (s : SemesterMarks) : ℕ :=
  s.maths + s.arts + s.science

/-- Represents Eva's marks for the entire year -/
structure YearMarks where
  first : SemesterMarks
  second : SemesterMarks

/-- Calculates the total marks for the year -/
def yearTotal (y : YearMarks) : ℕ :=
  totalMarks y.first + totalMarks y.second

theorem eva_marks_total (eva : YearMarks) 
  (h1 : eva.first.maths = eva.second.maths + 10)
  (h2 : eva.first.arts = eva.second.arts - 15)
  (h3 : eva.first.science = eva.second.science - eva.second.science / 3)
  (h4 : eva.second.maths = 80)
  (h5 : eva.second.arts = 90)
  (h6 : eva.second.science = 90) :
  yearTotal eva = 485 := by
  sorry

end eva_marks_total_l1235_123534


namespace percentage_of_120_to_40_l1235_123579

theorem percentage_of_120_to_40 : ∃ (p : ℝ), p = (120 : ℝ) / 40 * 100 ∧ p = 300 := by
  sorry

end percentage_of_120_to_40_l1235_123579


namespace cubic_sum_implies_square_sum_less_than_one_l1235_123580

theorem cubic_sum_implies_square_sum_less_than_one 
  (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : x^3 + y^3 = x - y) : 
  x^2 + y^2 < 1 := by
sorry

end cubic_sum_implies_square_sum_less_than_one_l1235_123580


namespace prob_at_least_three_babies_speak_l1235_123548

/-- The probability that at least 3 out of 6 babies will speak tomorrow, 
    given that each baby has a 1/3 probability of speaking. -/
theorem prob_at_least_three_babies_speak (n : ℕ) (p : ℝ) : 
  n = 6 → p = 1/3 → 
  (1 : ℝ) - (Nat.choose n 0 * (1 - p)^n + 
             Nat.choose n 1 * p * (1 - p)^(n-1) + 
             Nat.choose n 2 * p^2 * (1 - p)^(n-2)) = 353/729 := by
  sorry

end prob_at_least_three_babies_speak_l1235_123548


namespace min_disks_needed_l1235_123576

def total_files : ℕ := 40
def disk_capacity : ℚ := 1.44
def large_files : ℕ := 5
def medium_files : ℕ := 15
def small_files : ℕ := total_files - large_files - medium_files
def large_file_size : ℚ := 0.9
def medium_file_size : ℚ := 0.75
def small_file_size : ℚ := 0.5

theorem min_disks_needed :
  let total_size := large_files * large_file_size + medium_files * medium_file_size + small_files * small_file_size
  ∃ (n : ℕ), n * disk_capacity ≥ total_size ∧
             ∀ (m : ℕ), m * disk_capacity ≥ total_size → n ≤ m ∧
             n = 20 := by
  sorry

end min_disks_needed_l1235_123576


namespace rowing_time_ratio_l1235_123570

/-- Proves that the ratio of time taken to row upstream to downstream is 2:1 
    given the man's rowing speed in still water and the current speed. -/
theorem rowing_time_ratio 
  (man_speed : ℝ) 
  (current_speed : ℝ) 
  (h1 : man_speed = 3.9)
  (h2 : current_speed = 1.3) :
  (man_speed - current_speed) / (man_speed + current_speed) = 1 / 2 := by
  sorry

end rowing_time_ratio_l1235_123570


namespace jet_flight_time_l1235_123586

theorem jet_flight_time (distance : ℝ) (time_with_wind : ℝ) (wind_speed : ℝ) 
  (h1 : distance = 2000)
  (h2 : time_with_wind = 4)
  (h3 : wind_speed = 50)
  : ∃ (jet_speed : ℝ), 
    (jet_speed + wind_speed) * time_with_wind = distance ∧
    distance / (jet_speed - wind_speed) = 5 := by
  sorry

end jet_flight_time_l1235_123586


namespace birds_total_distance_l1235_123544

/-- Calculates the total distance flown by six birds given their speeds and flight times -/
def total_distance_flown (eagle_speed falcon_speed pelican_speed hummingbird_speed hawk_speed swallow_speed : ℝ)
  (eagle_time falcon_time pelican_time hummingbird_time hawk_time swallow_time : ℝ) : ℝ :=
  eagle_speed * eagle_time +
  falcon_speed * falcon_time +
  pelican_speed * pelican_time +
  hummingbird_speed * hummingbird_time +
  hawk_speed * hawk_time +
  swallow_speed * swallow_time

/-- The total distance flown by all birds is 482.5 miles -/
theorem birds_total_distance :
  total_distance_flown 15 46 33 30 45 25 2.5 2.5 2.5 2.5 3 1.5 = 482.5 := by
  sorry

end birds_total_distance_l1235_123544


namespace interest_rate_calculation_l1235_123519

theorem interest_rate_calculation (P : ℝ) (t : ℝ) (diff : ℝ) (r : ℝ) : 
  P = 5100 → 
  t = 2 → 
  P * ((1 + r) ^ t - 1) - P * r * t = diff → 
  diff = 51 → 
  r = 0.1 := by
sorry

end interest_rate_calculation_l1235_123519


namespace inequality_proof_l1235_123590

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  1 ≤ ((x + y) * (x^3 + y^3)) / ((x^2 + y^2)^2) ∧
  ((x + y) * (x^3 + y^3)) / ((x^2 + y^2)^2) ≤ 9/8 := by
  sorry

end inequality_proof_l1235_123590


namespace pages_difference_l1235_123568

theorem pages_difference (beatrix_pages cristobal_pages : ℕ) : 
  beatrix_pages = 704 →
  cristobal_pages = 3 * beatrix_pages + 15 →
  cristobal_pages - beatrix_pages = 1423 := by
sorry

end pages_difference_l1235_123568


namespace angle_triple_complement_measure_l1235_123598

theorem angle_triple_complement_measure :
  ∀ x : ℝ, 
    (x = 3 * (90 - x)) → 
    x = 67.5 := by
  sorry

end angle_triple_complement_measure_l1235_123598


namespace evaluate_expression_l1235_123510

theorem evaluate_expression : -(18 / 3 * 8 - 48 + 4 * 6) = -24 := by
  sorry

end evaluate_expression_l1235_123510


namespace red_ball_probability_l1235_123553

/-- Represents the number of balls of each color in the bag -/
structure BallCounts where
  red : ℕ
  yellow : ℕ
  white : ℕ

/-- Calculates the total number of balls in the bag -/
def totalBalls (counts : BallCounts) : ℕ :=
  counts.red + counts.yellow + counts.white

/-- Calculates the probability of drawing a ball of a specific color -/
def drawProbability (counts : BallCounts) (color : ℕ) : ℚ :=
  color / (totalBalls counts)

/-- Theorem: The probability of drawing a red ball from a bag with 3 red, 5 yellow, and 2 white balls is 3/10 -/
theorem red_ball_probability :
  let bag := BallCounts.mk 3 5 2
  drawProbability bag bag.red = 3 / 10 := by
  sorry

end red_ball_probability_l1235_123553


namespace number_classification_l1235_123542

-- Define the set of given numbers
def givenNumbers : Set ℝ := {-3, -1, 0, 20, 1/4, -6.5, 17/100, -8.5, 7, Real.pi, 16, -3.14}

-- Define the classification sets
def positiveNumbers : Set ℝ := {x | x > 0}
def integers : Set ℝ := {x | ∃ n : ℤ, x = n}
def fractions : Set ℝ := {x | ∃ a b : ℤ, b ≠ 0 ∧ x = a / b}
def positiveIntegers : Set ℝ := {x | ∃ n : ℕ, x = n ∧ n > 0}
def nonNegativeRationals : Set ℝ := {x | x ≥ 0 ∧ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b}

-- Theorem statement
theorem number_classification :
  (givenNumbers ∩ positiveNumbers = {20, 1/4, 17/100, 7, 16, Real.pi}) ∧
  (givenNumbers ∩ integers = {-3, -1, 0, 20, 7, 16}) ∧
  (givenNumbers ∩ fractions = {1/4, -6.5, 17/100, -8.5, -3.14}) ∧
  (givenNumbers ∩ positiveIntegers = {20, 7, 16}) ∧
  (givenNumbers ∩ nonNegativeRationals = {0, 20, 1/4, 17/100, 7, 16}) := by
  sorry

end number_classification_l1235_123542


namespace framed_picture_perimeter_is_six_feet_l1235_123591

/-- Calculates the perimeter of a framed picture given original dimensions and scaling factor. -/
def framedPicturePerimeter (width height scale border : ℚ) : ℚ :=
  2 * (width * scale + height * scale + 2 * border)

/-- Converts inches to feet -/
def inchesToFeet (inches : ℚ) : ℚ :=
  inches / 12

theorem framed_picture_perimeter_is_six_feet :
  let originalWidth : ℚ := 3
  let originalHeight : ℚ := 5
  let scaleFactor : ℚ := 3
  let borderWidth : ℚ := 3
  
  inchesToFeet (framedPicturePerimeter originalWidth originalHeight scaleFactor borderWidth) = 6 := by
  sorry

end framed_picture_perimeter_is_six_feet_l1235_123591


namespace corresponding_angles_random_l1235_123508

-- Define the concept of an event
def Event : Type := Unit

-- Define the concept of a random event
def RandomEvent (e : Event) : Prop := sorry

-- Define the given events
def sunRisesWest : Event := sorry
def triangleAngleSum : Event := sorry
def correspondingAngles : Event := sorry
def drawRedBall : Event := sorry

-- State the theorem
theorem corresponding_angles_random : RandomEvent correspondingAngles := by sorry

end corresponding_angles_random_l1235_123508


namespace union_of_sets_l1235_123513

theorem union_of_sets : 
  let P : Set Int := {-2, 2}
  let Q : Set Int := {-1, 0, 2, 3}
  P ∪ Q = {-2, -1, 0, 2, 3} := by
sorry

end union_of_sets_l1235_123513


namespace min_teams_non_negative_balance_l1235_123565

/-- Represents the number of wins in a series --/
inductive SeriesScore
| Four_Zero
| Four_One
| Four_Two
| Four_Three

/-- Represents a team's performance in the tournament --/
structure TeamPerformance where
  wins : ℕ
  losses : ℕ

/-- Represents the NHL playoff tournament --/
structure NHLPlayoffs where
  num_teams : ℕ
  num_rounds : ℕ
  series_scores : List SeriesScore

/-- Defines a non-negative balance of wins --/
def has_non_negative_balance (team : TeamPerformance) : Prop :=
  team.wins ≥ team.losses

/-- Theorem stating the minimum number of teams with non-negative balance --/
theorem min_teams_non_negative_balance (playoffs : NHLPlayoffs) 
  (h1 : playoffs.num_teams = 16)
  (h2 : playoffs.num_rounds = 4)
  (h3 : ∀ s ∈ playoffs.series_scores, s ∈ [SeriesScore.Four_Zero, SeriesScore.Four_One, SeriesScore.Four_Two, SeriesScore.Four_Three]) :
  ∃ (teams : List TeamPerformance), 
    (∀ team ∈ teams, has_non_negative_balance team) ∧ 
    (teams.length = 2) ∧
    (∀ (n : ℕ), n < 2 → ¬∃ (teams' : List TeamPerformance), 
      (∀ team ∈ teams', has_non_negative_balance team) ∧ 
      (teams'.length = n)) :=
by sorry

end min_teams_non_negative_balance_l1235_123565


namespace middle_school_soccer_league_l1235_123523

theorem middle_school_soccer_league (n : ℕ) : n = 9 :=
  by
  have total_games : n * (n - 1) / 2 = 36 := by sorry
  have min_games_per_team : n - 1 ≥ 8 := by sorry
  sorry

#check middle_school_soccer_league

end middle_school_soccer_league_l1235_123523


namespace complex_magnitude_problem_l1235_123593

theorem complex_magnitude_problem (z : ℂ) : z = (3 + I) / (2 - I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_problem_l1235_123593


namespace integral_shift_reciprocal_l1235_123578

open MeasureTheory

-- Define the function f and the integral L
variable (f : ℝ → ℝ)
variable (L : ℝ)

-- State the theorem
theorem integral_shift_reciprocal (hf : Continuous f) 
  (hL : ∫ (x : ℝ), f x = L) :
  ∫ (x : ℝ), f (x - 1/x) = L := by
  sorry

end integral_shift_reciprocal_l1235_123578


namespace bb_tileable_iff_2b_divides_l1235_123505

/-- A rectangle is (b,b)-tileable if it can be covered by b×b square tiles --/
def is_bb_tileable (m n b : ℕ) : Prop :=
  ∃ (k l : ℕ), m = k * b ∧ n = l * b

/-- Main theorem: An m×n rectangle is (b,b)-tileable iff 2b divides both m and n --/
theorem bb_tileable_iff_2b_divides (m n b : ℕ) (hm : m > 0) (hn : n > 0) (hb : b > 0) :
  is_bb_tileable m n b ↔ (2 * b ∣ m) ∧ (2 * b ∣ n) :=
sorry

end bb_tileable_iff_2b_divides_l1235_123505


namespace correct_schedule_count_l1235_123507

/-- Represents a club with members and scheduling constraints -/
structure Club where
  totalMembers : Nat
  daysToSchedule : Nat
  membersPerDay : Nat

/-- Represents the scheduling constraints for specific members -/
structure SchedulingConstraints where
  mustBeTogetherPair : Fin 2 → Nat
  cannotBeTogether : Fin 2 → Nat

/-- Calculates the total number of possible schedules given the club and constraints -/
def totalPossibleSchedules (club : Club) (constraints : SchedulingConstraints) : Nat :=
  sorry

/-- The main theorem stating the correct number of schedules -/
theorem correct_schedule_count :
  let club := Club.mk 10 5 2
  let constraints := SchedulingConstraints.mk
    (fun i => if i.val = 0 then 0 else 1)  -- A and B (represented as 0 and 1)
    (fun i => if i.val = 0 then 2 else 3)  -- C and D (represented as 2 and 3)
  totalPossibleSchedules club constraints = 5400 := by sorry

end correct_schedule_count_l1235_123507


namespace quadratic_equation_transform_l1235_123585

theorem quadratic_equation_transform (x : ℝ) :
  25 * x^2 - 10 * x - 1000 = 0 →
  ∃ (r : ℝ), (x + r)^2 = 40.04 :=
by sorry

end quadratic_equation_transform_l1235_123585


namespace football_games_per_month_l1235_123582

theorem football_games_per_month 
  (total_games : ℕ) 
  (num_months : ℕ) 
  (h1 : total_games = 323) 
  (h2 : num_months = 17) 
  (h3 : total_games % num_months = 0) : 
  total_games / num_months = 19 := by
sorry

end football_games_per_month_l1235_123582


namespace ball_probabilities_l1235_123518

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  total : ℕ
  red : ℕ
  black : ℕ
  white : ℕ
  green : ℕ

/-- The given ball counts in the problem -/
def problemCounts : BallCounts := {
  total := 12,
  red := 5,
  black := 4,
  white := 2,
  green := 1
}

/-- Calculates the probability of drawing a red or black ball -/
def probRedOrBlack (counts : BallCounts) : ℚ :=
  (counts.red + counts.black : ℚ) / counts.total

/-- Calculates the probability of drawing at least one red ball when two balls are drawn -/
def probAtLeastOneRed (counts : BallCounts) : ℚ :=
  let totalWays := counts.total * (counts.total - 1) / 2
  let oneRedWays := counts.red * (counts.total - counts.red)
  let twoRedWays := counts.red * (counts.red - 1) / 2
  (oneRedWays + twoRedWays : ℚ) / totalWays

theorem ball_probabilities (counts : BallCounts) 
    (h_total : counts.total = 12)
    (h_red : counts.red = 5)
    (h_black : counts.black = 4)
    (h_white : counts.white = 2)
    (h_green : counts.green = 1) :
    probRedOrBlack counts = 3/4 ∧ probAtLeastOneRed counts = 15/22 := by
  sorry

#eval probRedOrBlack problemCounts
#eval probAtLeastOneRed problemCounts

end ball_probabilities_l1235_123518


namespace smallest_prime_factor_of_expression_l1235_123575

theorem smallest_prime_factor_of_expression : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (12^3 + 15^4 - 6^6) ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ (12^3 + 15^4 - 6^6) → p ≤ q :=
by sorry

end smallest_prime_factor_of_expression_l1235_123575


namespace value_of_a_l1235_123515

theorem value_of_a (a b d : ℤ) 
  (h1 : a + b = d) 
  (h2 : b + d = 7) 
  (h3 : d = 4) : 
  a = 1 := by
sorry

end value_of_a_l1235_123515


namespace smallest_q_for_inequality_l1235_123560

theorem smallest_q_for_inequality : ∃ (q : ℕ+), 
  (q = 2015) ∧ 
  (∀ (q' : ℕ+), q' < q → 
    ∃ (m : ℕ), 1 ≤ m ∧ m ≤ 1006 ∧ 
      ∀ (n : ℤ), (↑m / 1007 : ℚ) * ↑q' ≥ ↑n ∨ ↑n ≥ (↑(m + 1) / 1008 : ℚ) * ↑q') ∧
  (∀ (m : ℕ), 1 ≤ m → m ≤ 1006 → 
    ∃ (n : ℤ), (↑m / 1007 : ℚ) * ↑q < ↑n ∧ ↑n < (↑(m + 1) / 1008 : ℚ) * ↑q) :=
by sorry

end smallest_q_for_inequality_l1235_123560


namespace geometric_series_first_term_l1235_123563

theorem geometric_series_first_term 
  (a r : ℝ) 
  (h1 : a / (1 - r) = 20) 
  (h2 : a^2 / (1 - r^2) = 80) : 
  a = 20 / 3 := by
sorry

end geometric_series_first_term_l1235_123563


namespace clock_correction_theorem_l1235_123599

/-- The number of days between March 1st at noon and March 10th at 6 P.M. -/
def days_passed : ℚ := 9 + 6/24

/-- The rate at which the clock loses time, in minutes per day -/
def loss_rate : ℚ := 15

/-- The function to calculate the positive correction in minutes -/
def correction (d : ℚ) (r : ℚ) : ℚ := d * r

/-- Theorem stating that the positive correction needed is 138.75 minutes -/
theorem clock_correction_theorem :
  correction days_passed loss_rate = 138.75 := by sorry

end clock_correction_theorem_l1235_123599


namespace line_perpendicular_parallel_implies_planes_perpendicular_l1235_123501

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem line_perpendicular_parallel_implies_planes_perpendicular
  (m : Line) (α β : Plane) :
  perpendicular m β → parallel m α → perpendicularPlanes α β := by
  sorry

end line_perpendicular_parallel_implies_planes_perpendicular_l1235_123501


namespace cubic_root_sum_l1235_123521

theorem cubic_root_sum (a b c : ℝ) : 
  (45 * a^3 - 70 * a^2 + 28 * a - 2 = 0) →
  (45 * b^3 - 70 * b^2 + 28 * b - 2 = 0) →
  (45 * c^3 - 70 * c^2 + 28 * c - 2 = 0) →
  a ≠ b → b ≠ c → a ≠ c →
  -1 < a → a < 1 →
  -1 < b → b < 1 →
  -1 < c → c < 1 →
  1/(1-a) + 1/(1-b) + 1/(1-c) = 13/9 := by
sorry

end cubic_root_sum_l1235_123521


namespace factorization_equality_l1235_123543

theorem factorization_equality (a b : ℝ) : a * b^2 - 8 * a * b + 16 * a = a * (b - 4)^2 := by
  sorry

end factorization_equality_l1235_123543


namespace lines_without_common_point_are_parallel_or_skew_l1235_123559

-- Define a type for straight lines in 3D space
structure Line3D where
  -- You might represent a line using a point and a direction vector
  -- or any other suitable representation
  -- This is just a placeholder structure
  mk :: (dummy : Unit)

-- Define the property of two lines not having a common point
def noCommonPoint (a b : Line3D) : Prop :=
  -- The actual implementation would depend on how you define Line3D
  sorry

-- Define the property of two lines being parallel
def parallel (a b : Line3D) : Prop :=
  -- The actual implementation would depend on how you define Line3D
  sorry

-- Define the property of two lines being skew
def skew (a b : Line3D) : Prop :=
  -- The actual implementation would depend on how you define Line3D
  sorry

-- The theorem statement
theorem lines_without_common_point_are_parallel_or_skew 
  (a b : Line3D) (h : noCommonPoint a b) : 
  parallel a b ∨ skew a b :=
sorry

end lines_without_common_point_are_parallel_or_skew_l1235_123559


namespace angle_B_value_side_b_value_l1235_123541

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2

/-- The sine law holds for the triangle -/
axiom sine_law (t : AcuteTriangle) : t.a / Real.sin t.A = t.b / Real.sin t.B

/-- The cosine law holds for the triangle -/
axiom cosine_law (t : AcuteTriangle) : t.b^2 = t.a^2 + t.c^2 - 2*t.a*t.c*Real.cos t.B

/-- The given condition a = 2b sin A -/
def condition (t : AcuteTriangle) : Prop := t.a = 2*t.b*Real.sin t.A

theorem angle_B_value (t : AcuteTriangle) (h : condition t) : t.B = π/6 := by sorry

theorem side_b_value (t : AcuteTriangle) (h1 : t.a = 3*Real.sqrt 3) (h2 : t.c = 5) (h3 : t.B = π/6) : 
  t.b = Real.sqrt 7 := by sorry

end angle_B_value_side_b_value_l1235_123541


namespace magnitude_v_l1235_123516

theorem magnitude_v (u v : ℂ) (h1 : u * v = 20 - 15 * I) (h2 : Complex.abs u = Real.sqrt 34) : 
  Complex.abs v = (25 * Real.sqrt 34) / 34 := by
  sorry

end magnitude_v_l1235_123516


namespace survey_result_l1235_123594

def survey (total : ℕ) (neither : ℕ) (enjoyed : ℕ) (understood : ℕ) : Prop :=
  total = 600 ∧ 
  neither = 150 ∧
  enjoyed = understood ∧
  enjoyed + neither = total

theorem survey_result (total neither enjoyed understood : ℕ) 
  (h : survey total neither enjoyed understood) : 
  (enjoyed : ℚ) / total = 3 / 4 := by
sorry

end survey_result_l1235_123594


namespace triangle_properties_l1235_123589

noncomputable section

-- Define the triangle ABC
variable (A B C : Real) -- Angles
variable (a b c : Real) -- Side lengths

-- Define the conditions
axiom angle_side_relation : 2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c
axiom c_value : c = Real.sqrt 7
axiom triangle_area : 1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2

-- Theorem to prove
theorem triangle_properties : C = π/3 ∧ a + b + c = 5 + Real.sqrt 7 := by
  sorry

end triangle_properties_l1235_123589


namespace middle_digit_is_six_l1235_123522

/-- Represents a three-digit number in a given base -/
structure ThreeDigitNumber (base : ℕ) where
  hundreds : ℕ
  tens : ℕ
  ones : ℕ
  valid_digits : hundreds < base ∧ tens < base ∧ ones < base

/-- Converts a ThreeDigitNumber to its numerical value -/
def to_nat {base : ℕ} (n : ThreeDigitNumber base) : ℕ :=
  n.hundreds * base^2 + n.tens * base + n.ones

/-- Theorem: For a number M that is a three-digit number in base 8 and
    has its digits reversed in base 10, the middle digit of M in base 8 is 6 -/
theorem middle_digit_is_six :
  ∀ (M_base8 : ThreeDigitNumber 8) (M_base10 : ThreeDigitNumber 10),
    to_nat M_base8 = to_nat M_base10 →
    M_base8.hundreds = M_base10.ones →
    M_base8.tens = M_base10.tens →
    M_base8.ones = M_base10.hundreds →
    M_base8.tens = 6 := by
  sorry

end middle_digit_is_six_l1235_123522


namespace quadratic_roots_sum_l1235_123500

theorem quadratic_roots_sum (a b : ℝ) : 
  (∀ x, ax^2 + bx + 2 = 0 ↔ x = -1/2 ∨ x = 1/3) → 
  a + b = -14 := by
sorry

end quadratic_roots_sum_l1235_123500


namespace parallel_segments_length_l1235_123537

/-- In a triangle with sides a, b, and c, if three segments parallel to the sides
    pass through one point and have equal length x, then x = (2abc) / (ab + ac + bc). -/
theorem parallel_segments_length (a b c x : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  x = (2 * a * b * c) / (a * b + a * c + b * c) := by
  sorry

#check parallel_segments_length

end parallel_segments_length_l1235_123537


namespace min_rooks_correct_min_rooks_minimal_l1235_123592

/-- A function that returns the minimum number of rooks needed on an n × n board
    to guarantee k non-attacking rooks can be selected. -/
def min_rooks (n k : ℕ) : ℕ :=
  n * (k - 1) + 1

/-- Theorem stating that min_rooks gives the correct minimum number of rooks. -/
theorem min_rooks_correct (n k : ℕ) (h1 : 1 < k) (h2 : k ≤ n) :
  ∀ (m : ℕ), m ≥ min_rooks n k →
    ∀ (placement : Fin m → Fin n × Fin n),
      ∃ (selected : Fin k → Fin m),
        ∀ (i j : Fin k), i ≠ j →
          (placement (selected i)).1 ≠ (placement (selected j)).1 ∧
          (placement (selected i)).2 ≠ (placement (selected j)).2 :=
by
  sorry

/-- Theorem stating that min_rooks gives the smallest such number. -/
theorem min_rooks_minimal (n k : ℕ) (h1 : 1 < k) (h2 : k ≤ n) :
  ∀ (m : ℕ), m < min_rooks n k →
    ∃ (placement : Fin m → Fin n × Fin n),
      ∀ (selected : Fin k → Fin m),
        ∃ (i j : Fin k), i ≠ j ∧
          ((placement (selected i)).1 = (placement (selected j)).1 ∨
           (placement (selected i)).2 = (placement (selected j)).2) :=
by
  sorry

end min_rooks_correct_min_rooks_minimal_l1235_123592


namespace no_integer_solutions_l1235_123533

theorem no_integer_solutions :
  ¬ ∃ (m n : ℤ), m^2 - 11*m*n - 8*n^2 = 88 := by
  sorry

end no_integer_solutions_l1235_123533


namespace highland_baseball_club_members_l1235_123538

/-- The cost of a pair of socks in dollars -/
def sockCost : ℕ := 6

/-- The additional cost of a T-shirt compared to a pair of socks in dollars -/
def tShirtAdditionalCost : ℕ := 7

/-- The total expenditure for all members in dollars -/
def totalExpenditure : ℕ := 5112

/-- Calculates the number of members in the Highland Baseball Club -/
def calculateMembers (sockCost tShirtAdditionalCost totalExpenditure : ℕ) : ℕ :=
  let tShirtCost := sockCost + tShirtAdditionalCost
  let capCost := sockCost
  let costPerMember := (sockCost + tShirtCost) + (sockCost + tShirtCost + capCost)
  totalExpenditure / costPerMember

theorem highland_baseball_club_members :
  calculateMembers sockCost tShirtAdditionalCost totalExpenditure = 116 := by
  sorry

end highland_baseball_club_members_l1235_123538


namespace exists_subset_with_common_gcd_l1235_123596

/-- A function that checks if a number is the product of at most 1987 prime factors -/
def is_valid_element (n : ℕ) : Prop := ∃ (factors : List ℕ), n = factors.prod ∧ factors.all Nat.Prime ∧ factors.length ≤ 1987

/-- The set A of integers, each being a product of at most 1987 prime factors -/
def A : Set ℕ := {n | is_valid_element n}

/-- The theorem to be proved -/
theorem exists_subset_with_common_gcd (h : Set.Infinite A) :
  ∃ (B : Set ℕ) (b : ℕ), Set.Infinite B ∧ B ⊆ A ∧ b > 0 ∧
  ∀ (x y : ℕ), x ∈ B → y ∈ B → Nat.gcd x y = b :=
sorry

end exists_subset_with_common_gcd_l1235_123596


namespace arithmetic_mean_problem_l1235_123583

theorem arithmetic_mean_problem : 
  let a := 9/16
  let b := 3/4
  let c := 5/8
  c = (a + b) / 2 :=
by sorry

end arithmetic_mean_problem_l1235_123583


namespace no_valid_arrangement_l1235_123574

-- Define the set of people
inductive Person : Type
| Alice : Person
| Bob : Person
| Carla : Person
| Derek : Person
| Eric : Person

-- Define a seating arrangement as a function from Person to ℕ (seat number)
def SeatingArrangement := Person → Fin 5

-- Define the adjacency relation for a circular table
def adjacent (s : SeatingArrangement) (p1 p2 : Person) : Prop :=
  (s p1 - s p2 = 1) ∨ (s p2 - s p1 = 1) ∨ (s p1 = 4 ∧ s p2 = 0) ∨ (s p1 = 0 ∧ s p2 = 4)

-- Define the seating restrictions
def validArrangement (s : SeatingArrangement) : Prop :=
  (¬ adjacent s Person.Alice Person.Bob) ∧
  (¬ adjacent s Person.Alice Person.Carla) ∧
  (¬ adjacent s Person.Derek Person.Eric) ∧
  (¬ adjacent s Person.Carla Person.Derek) ∧
  Function.Injective s

-- Theorem stating that no valid seating arrangement exists
theorem no_valid_arrangement : ¬ ∃ s : SeatingArrangement, validArrangement s := by
  sorry


end no_valid_arrangement_l1235_123574


namespace perfect_squares_implications_l1235_123532

theorem perfect_squares_implications (n : ℕ+) 
  (h1 : ∃ a : ℕ, 3 * n + 1 = a^2) 
  (h2 : ∃ b : ℕ, 5 * n - 1 = b^2) :
  (∃ p q : ℕ, p > 1 ∧ q > 1 ∧ 7 * n + 13 = p * q) ∧ 
  (∃ x y : ℕ, 8 * (17 * n^2 + 3 * n) = x^2 + y^2) := by
sorry

end perfect_squares_implications_l1235_123532


namespace sum_of_ages_l1235_123573

theorem sum_of_ages (age1 age2 : ℕ) : 
  age2 = age1 + 1 → age1 = 13 → age2 = 14 → age1 + age2 = 27 := by
sorry

end sum_of_ages_l1235_123573


namespace hundred_with_five_twos_l1235_123577

theorem hundred_with_five_twos :
  (222 / 2) - (22 / 2) = 100 :=
by sorry

end hundred_with_five_twos_l1235_123577


namespace solve_equation_l1235_123561

theorem solve_equation (y : ℚ) (h : 3 * y - 9 = -6 * y + 3) : y = 4 / 3 := by
  sorry

end solve_equation_l1235_123561


namespace course_selection_theorem_l1235_123584

def total_course_selection_plans (n : ℕ) (k₁ k₂ : ℕ) : ℕ :=
  (n.choose k₁) * (n.choose k₂) * (n.choose k₂)

theorem course_selection_theorem :
  total_course_selection_plans 4 2 3 = 96 := by
  sorry

end course_selection_theorem_l1235_123584


namespace union_covers_reals_l1235_123540

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B (a : ℝ) : Set ℝ := {x | |x - a| < 3}

-- State the theorem
theorem union_covers_reals (a : ℝ) : 
  (A ∪ B a = Set.univ) → a ∈ Set.Ioo (-1) 2 := by sorry

end union_covers_reals_l1235_123540


namespace value_of_x_l1235_123572

theorem value_of_x (x y z : ℝ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 96) : x = 8 := by
  sorry

end value_of_x_l1235_123572


namespace intersection_of_A_and_B_l1235_123547

def A : Set ℕ := {x : ℕ | x^2 - 5*x ≤ 0}
def B : Set ℕ := {0, 2, 5, 7}

theorem intersection_of_A_and_B : A ∩ B = {0, 2, 5} := by
  sorry

end intersection_of_A_and_B_l1235_123547


namespace hyperbola_equation_l1235_123535

/-- A hyperbola with given asymptotes and passing through a specific point -/
theorem hyperbola_equation (x y : ℝ) : 
  (∀ (k : ℝ), (2*x = 3*y ∨ 2*x = -3*y) → k*(2*x) = k*(3*y)) →  -- Asymptotes condition
  (4*(1:ℝ)^2 - 9*(2:ℝ)^2 = -32) →                              -- Point (1,2) satisfies the equation
  (4*x^2 - 9*y^2 = -32)                                         -- Resulting hyperbola equation
  := by sorry

end hyperbola_equation_l1235_123535


namespace infinite_series_solution_l1235_123595

theorem infinite_series_solution (x : ℝ) : 
  (∑' n, (2*n + 1) * x^n) = 16 → x = 5/8 := by sorry

end infinite_series_solution_l1235_123595


namespace snowboard_final_price_l1235_123556

/-- Calculates the final price of an item after applying two discounts and a sales tax. -/
def finalPrice (originalPrice : ℝ) (discount1 : ℝ) (discount2 : ℝ) (salesTax : ℝ) : ℝ :=
  let priceAfterDiscount1 := originalPrice * (1 - discount1)
  let priceAfterDiscount2 := priceAfterDiscount1 * (1 - discount2)
  priceAfterDiscount2 * (1 + salesTax)

/-- Theorem stating that the final price of a $200 snowboard after 40% and 20% discounts
    and 5% sales tax is $100.80. -/
theorem snowboard_final_price :
  finalPrice 200 0.4 0.2 0.05 = 100.80 := by
  sorry

end snowboard_final_price_l1235_123556


namespace min_value_theorem_l1235_123511

theorem min_value_theorem (a b : ℝ) (h1 : a + b = 2) (h2 : b > 0) :
  (∀ x y : ℝ, x + y = 2 → y > 0 → (1 / (2 * |x|) + |x| / y) ≥ 3/4) ∧
  (∃ x y : ℝ, x + y = 2 ∧ y > 0 ∧ 1 / (2 * |x|) + |x| / y = 3/4) :=
by sorry

end min_value_theorem_l1235_123511


namespace ceiling_bounds_l1235_123539

-- Define the ceiling function
def ceiling (x : ℚ) : ℤ :=
  Int.ceil x

-- State the theorem
theorem ceiling_bounds (m : ℚ) : m ≤ ceiling m ∧ (ceiling m : ℚ) < m + 1 := by
  sorry

-- Define the property of ceiling function
axiom ceiling_property (a : ℚ) : ∃ b : ℚ, 0 ≤ b ∧ b < 1 ∧ a = ceiling a - b

end ceiling_bounds_l1235_123539


namespace delta_computation_l1235_123588

-- Define the new operation
def delta (a b : ℕ) : ℕ := a^3 - b

-- State the theorem
theorem delta_computation :
  delta (5^(delta 6 8)) (4^(delta 2 7)) = 5^624 - 4 := by
  sorry

end delta_computation_l1235_123588


namespace seven_power_minus_three_times_two_power_eq_one_solutions_l1235_123567

theorem seven_power_minus_three_times_two_power_eq_one_solutions :
  ∀ x y : ℕ, 7^x - 3 * 2^y = 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) := by
  sorry

end seven_power_minus_three_times_two_power_eq_one_solutions_l1235_123567


namespace shekars_social_studies_score_l1235_123566

/-- Given Shekar's scores in four subjects and his average marks across all five subjects,
    prove that his marks in social studies must be 82. -/
theorem shekars_social_studies_score
  (math_score : ℕ)
  (science_score : ℕ)
  (english_score : ℕ)
  (biology_score : ℕ)
  (average_marks : ℕ)
  (num_subjects : ℕ)
  (h1 : math_score = 76)
  (h2 : science_score = 65)
  (h3 : english_score = 47)
  (h4 : biology_score = 85)
  (h5 : average_marks = 71)
  (h6 : num_subjects = 5)
  : ∃ (social_studies_score : ℕ),
    social_studies_score = 82 ∧
    (math_score + science_score + english_score + biology_score + social_studies_score) / num_subjects = average_marks :=
by
  sorry


end shekars_social_studies_score_l1235_123566


namespace compare_large_exponents_l1235_123504

theorem compare_large_exponents : 1997^(1998^1999) > 1999^(1998^1997) := by
  sorry

end compare_large_exponents_l1235_123504


namespace min_value_reciprocal_sum_l1235_123509

theorem min_value_reciprocal_sum (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_eq : 2 * x + y = 1) :
  (2 / x + 1 / y) ≥ 9 :=
sorry

end min_value_reciprocal_sum_l1235_123509


namespace bookstore_sales_l1235_123503

theorem bookstore_sales (wednesday_sales : ℕ) (thursday_sales : ℕ) (friday_sales : ℕ) : 
  wednesday_sales = 15 →
  thursday_sales = 3 * wednesday_sales →
  friday_sales = thursday_sales / 5 →
  wednesday_sales + thursday_sales + friday_sales = 69 := by
sorry

end bookstore_sales_l1235_123503


namespace worker_efficiency_l1235_123569

/-- Given two workers A and B, where A is half as efficient as B,
    this theorem proves that if they together complete a job in 13 days,
    then B alone can complete the job in 19.5 days. -/
theorem worker_efficiency (A B : ℝ) (h1 : A = (1/2) * B) (h2 : (A + B) * 13 = 1) :
  (1 / B) = 19.5 := by
  sorry

end worker_efficiency_l1235_123569


namespace triangle_area_with_ratio_l1235_123502

/-- Given a triangle ABC with sides a, b, c and corresponding angles A, B, C,
    if (b+c):(c+a):(a+b) = 4:5:6 and b+c = 8, then the area of triangle ABC is 15√3/4 -/
theorem triangle_area_with_ratio (a b c : ℝ) (A B C : ℝ) :
  (b + c) / (c + a) = 4 / 5 →
  (c + a) / (a + b) = 5 / 6 →
  b + c = 8 →
  (a + b + c) / 2 > 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a * a = b * b + c * c - 2 * b * c * Real.cos A →
  b * b = a * a + c * c - 2 * a * c * Real.cos B →
  c * c = a * a + b * b - 2 * a * b * Real.cos C →
  (1 / 2) * b * c * Real.sin A = 15 * Real.sqrt 3 / 4 := by
  sorry

end triangle_area_with_ratio_l1235_123502


namespace divisibility_relations_l1235_123531

theorem divisibility_relations (a b : ℤ) (ha : a ≥ 1) (hb : b ≥ 1) :
  (¬ ((a ∣ b^2) ↔ (a ∣ b))) ∧ ((a^2 ∣ b^2) ↔ (a ∣ b)) := by
  sorry

end divisibility_relations_l1235_123531


namespace rectangle_not_always_similar_l1235_123549

-- Define the shapes
structure Square :=
  (side : ℝ)

structure IsoscelesRightTriangle :=
  (leg : ℝ)

structure Rectangle :=
  (length width : ℝ)

structure EquilateralTriangle :=
  (side : ℝ)

-- Define similarity for each shape
def similar_squares (s1 s2 : Square) : Prop :=
  true

def similar_isosceles_right_triangles (t1 t2 : IsoscelesRightTriangle) : Prop :=
  true

def similar_rectangles (r1 r2 : Rectangle) : Prop :=
  r1.length / r1.width = r2.length / r2.width

def similar_equilateral_triangles (e1 e2 : EquilateralTriangle) : Prop :=
  true

-- Theorem statement
theorem rectangle_not_always_similar :
  ∃ r1 r2 : Rectangle, ¬(similar_rectangles r1 r2) ∧
  (∀ s1 s2 : Square, similar_squares s1 s2) ∧
  (∀ t1 t2 : IsoscelesRightTriangle, similar_isosceles_right_triangles t1 t2) ∧
  (∀ e1 e2 : EquilateralTriangle, similar_equilateral_triangles e1 e2) :=
sorry

end rectangle_not_always_similar_l1235_123549


namespace range_of_cubic_sum_l1235_123528

theorem range_of_cubic_sum (a b : ℝ) (h : a^2 + b^2 = a + b) :
  0 ≤ a^3 + b^3 ∧ a^3 + b^3 ≤ 2 := by sorry

end range_of_cubic_sum_l1235_123528


namespace roots_cube_equality_l1235_123506

/-- Given a polynomial P(x) = 3x² + 3mx + m² - 1 where m is a real number,
    and x₁, x₂ are the roots of P(x), prove that P(x₁³) = P(x₂³) -/
theorem roots_cube_equality (m : ℝ) (x₁ x₂ : ℝ) : 
  let P := fun x : ℝ => 3 * x^2 + 3 * m * x + m^2 - 1
  (P x₁ = 0 ∧ P x₂ = 0) → P (x₁^3) = P (x₂^3) := by
  sorry

end roots_cube_equality_l1235_123506


namespace f_minimum_and_tangents_l1235_123597

noncomputable def f (x : ℝ) : ℝ := x * (Real.log x + 1)

theorem f_minimum_and_tangents 
  (a b : ℝ) 
  (h1 : 0 < b) (h2 : b < a * Real.log a + a) :
  (∃ (min : ℝ), min = -Real.exp (-2) ∧ ∀ x > 0, f x ≥ min) ∧
  (∃ (x₁ x₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    x₁ > Real.exp (-2) ∧ 
    x₂ > Real.exp (-2) ∧
    b - f x₁ = (Real.log x₁ + 2) * (a - x₁) ∧
    b - f x₂ = (Real.log x₂ + 2) * (a - x₂)) :=
sorry

end f_minimum_and_tangents_l1235_123597


namespace part_one_part_two_l1235_123545

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 3*x ≤ 10}
def N (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a + 1}

-- Part 1
theorem part_one : 
  M ∩ (Set.univ \ N 2) = {x : ℝ | -2 ≤ x ∧ x < 3} := by sorry

-- Part 2
theorem part_two : 
  ∀ a : ℝ, M ∪ N a = M → a ≤ 2 := by sorry

end part_one_part_two_l1235_123545


namespace parry_prob_secretary_or_treasurer_l1235_123551

-- Define the number of club members
def total_members : ℕ := 10

-- Define the probability of being chosen as secretary
def prob_secretary : ℚ := 1 / 9

-- Define the probability of being chosen as treasurer
def prob_treasurer : ℚ := 1 / 10

-- Theorem statement
theorem parry_prob_secretary_or_treasurer :
  let prob_either := prob_secretary + prob_treasurer
  prob_either = 19 / 90 := by
  sorry

end parry_prob_secretary_or_treasurer_l1235_123551


namespace fifth_valid_number_is_443_l1235_123555

/-- Represents a random number table --/
def RandomNumberTable := List (List Nat)

/-- Checks if a number is valid (less than or equal to 600) --/
def isValidNumber (n : Nat) : Bool :=
  n ≤ 600

/-- Finds the nth valid number in a list --/
def findNthValidNumber (numbers : List Nat) (n : Nat) : Option Nat :=
  let validNumbers := numbers.filter isValidNumber
  validNumbers.get? (n - 1)

/-- The given random number table (partial) --/
def givenTable : RandomNumberTable :=
  [[84, 42, 17, 53, 31, 57, 24, 55, 6, 88, 77, 4, 74, 47, 67, 21, 76, 33, 50, 25, 83, 92, 12, 6, 76, 63, 1, 63],
   [78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 7, 44, 39, 52, 38, 79, 33, 21, 12, 34, 29, 78],
   [64, 56, 7, 82, 52, 42, 7, 44, 38, 15, 51, 0, 13, 42, 99, 66, 2, 79, 54]]

/-- The main theorem --/
theorem fifth_valid_number_is_443 :
  let numbers := (givenTable.get! 1).drop 7 ++ (givenTable.get! 2) ++ (givenTable.get! 3)
  findNthValidNumber numbers 5 = some 443 := by
  sorry

end fifth_valid_number_is_443_l1235_123555


namespace cow_husk_consumption_l1235_123536

/-- Given that 50 cows eat 50 bags of husk in 50 days, prove that one cow will eat one bag of husk in 50 days -/
theorem cow_husk_consumption (cows bags days : ℕ) (h : cows = 50 ∧ bags = 50 ∧ days = 50) :
  (1 : ℕ) * bags * days = cows * (1 : ℕ) * days :=
by sorry

end cow_husk_consumption_l1235_123536


namespace max_books_borrowed_l1235_123546

theorem max_books_borrowed (total_students : ℕ) 
  (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (three_plus_books : ℕ) (five_plus_books : ℕ) 
  (average_books : ℝ) :
  total_students = 100 ∧ 
  zero_books = 5 ∧ 
  one_book = 20 ∧ 
  two_books = 25 ∧ 
  three_plus_books = 30 ∧ 
  five_plus_books = 20 ∧ 
  average_books = 3 →
  ∃ (max_books : ℕ), 
    max_books = 50 ∧ 
    ∀ (student_books : ℕ), 
      student_books ≤ max_books :=
by
  sorry

#check max_books_borrowed

end max_books_borrowed_l1235_123546


namespace fourth_root_equivalence_l1235_123571

theorem fourth_root_equivalence (x : ℝ) (hx : x > 0) :
  (x * x^(1/3))^(1/4) = x^(1/3) := by
  sorry

end fourth_root_equivalence_l1235_123571


namespace toonies_count_l1235_123562

/-- Represents the number of toonies in a set of coins --/
def num_toonies (total_coins : ℕ) (total_value : ℕ) : ℕ :=
  total_coins - (2 * total_coins - total_value)

/-- Theorem stating that given 10 coins with a total value of $14, 
    the number of $2 coins (toonies) is 4 --/
theorem toonies_count : num_toonies 10 14 = 4 := by
  sorry

#eval num_toonies 10 14  -- Should output 4

end toonies_count_l1235_123562


namespace divisor_sum_theorem_l1235_123512

def sum_of_geometric_series (a r : ℕ) (n : ℕ) : ℕ := (a * (r^(n+1) - 1)) / (r - 1)

theorem divisor_sum_theorem (i j k : ℕ) : 
  (sum_of_geometric_series 1 2 i) * (sum_of_geometric_series 1 3 j) * (sum_of_geometric_series 1 5 k) = 3600 → 
  i + j + k = 7 := by
  sorry

end divisor_sum_theorem_l1235_123512


namespace max_digits_product_5_4_l1235_123558

theorem max_digits_product_5_4 : ∀ a b : ℕ, 
  10000 ≤ a ∧ a < 100000 → 1000 ≤ b ∧ b < 10000 → 
  a * b < 1000000000 := by
  sorry

end max_digits_product_5_4_l1235_123558


namespace eventual_habitable_fraction_l1235_123527

-- Define the fraction of earth's surface not covered by water
def landFraction : ℚ := 1 / 3

-- Define the fraction of exposed land initially inhabitable
def initialInhabitableFraction : ℚ := 1 / 3

-- Define the additional fraction of non-inhabitable land made viable by technology
def techAdvancementFraction : ℚ := 1 / 2

-- Theorem statement
theorem eventual_habitable_fraction :
  let initialHabitableLand := landFraction * initialInhabitableFraction
  let additionalHabitableLand := landFraction * (1 - initialInhabitableFraction) * techAdvancementFraction
  initialHabitableLand + additionalHabitableLand = 2 / 9 :=
by sorry

end eventual_habitable_fraction_l1235_123527


namespace divisors_of_60_l1235_123552

/-- The number of positive divisors of 60 is 12. -/
theorem divisors_of_60 : Nat.card (Nat.divisors 60) = 12 := by sorry

end divisors_of_60_l1235_123552


namespace quadratic_equation_solution_l1235_123550

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 2 * x - 1
  ∀ x : ℝ, f x = 0 ↔ x = -1/3 ∨ x = 1 := by
sorry

end quadratic_equation_solution_l1235_123550


namespace michaels_trophy_increase_l1235_123526

theorem michaels_trophy_increase :
  let michael_current : ℕ := 30
  let jack_future : ℕ := 10 * michael_current
  let total_future : ℕ := 430
  let michael_increase : ℕ := total_future - (michael_current + jack_future)
  michael_increase = 100 := by
sorry

end michaels_trophy_increase_l1235_123526


namespace sum_of_A_and_B_l1235_123524

/-- Represents a 3x3 grid with numbers 1, 2, and 3 -/
def Grid := Fin 3 → Fin 3 → Fin 3

/-- Check if a row contains 1, 2, and 3 -/
def valid_row (g : Grid) (row : Fin 3) : Prop :=
  ∀ n : Fin 3, ∃ col : Fin 3, g row col = n

/-- Check if a column contains 1, 2, and 3 -/
def valid_column (g : Grid) (col : Fin 3) : Prop :=
  ∀ n : Fin 3, ∃ row : Fin 3, g row col = n

/-- Check if the main diagonal contains 1, 2, and 3 -/
def valid_main_diagonal (g : Grid) : Prop :=
  ∀ n : Fin 3, ∃ i : Fin 3, g i i = n

/-- Check if the anti-diagonal contains 1, 2, and 3 -/
def valid_anti_diagonal (g : Grid) : Prop :=
  ∀ n : Fin 3, ∃ i : Fin 3, g i (2 - i) = n

/-- A grid is valid if all rows, columns, and diagonals contain 1, 2, and 3 -/
def valid_grid (g : Grid) : Prop :=
  (∀ row : Fin 3, valid_row g row) ∧
  (∀ col : Fin 3, valid_column g col) ∧
  valid_main_diagonal g ∧
  valid_anti_diagonal g

theorem sum_of_A_and_B (g : Grid) (h : valid_grid g) 
  (h1 : g 0 0 = 2) (h2 : g 1 2 = 3) : 
  g 1 1 + g 2 0 = 3 := by
  sorry


end sum_of_A_and_B_l1235_123524


namespace probability_second_red_given_first_red_for_given_numbers_l1235_123525

/-- Represents the probability of drawing a red ball on the second draw,
    given that the first ball drawn was red. -/
def probability_second_red_given_first_red (total : ℕ) (red : ℕ) (white : ℕ) : ℚ :=
  if total = red + white ∧ red > 0 ∧ white ≥ 0 then
    (red - 1) / (total - 1)
  else
    0

theorem probability_second_red_given_first_red_for_given_numbers :
  probability_second_red_given_first_red 10 6 4 = 5/9 := by
  sorry

end probability_second_red_given_first_red_for_given_numbers_l1235_123525


namespace P3_is_one_fourth_P4_is_three_fourths_l1235_123530

/-- The probability that the center of a circle is in the interior of the convex hull of n points
    selected independently with uniform distribution on the circle. -/
def P (n : ℕ) : ℝ := sorry

/-- Theorem: The probability P3 is 1/4 -/
theorem P3_is_one_fourth : P 3 = 1/4 := by sorry

/-- Theorem: The probability P4 is 3/4 -/
theorem P4_is_three_fourths : P 4 = 3/4 := by sorry

end P3_is_one_fourth_P4_is_three_fourths_l1235_123530


namespace cos_four_pi_thirds_minus_alpha_l1235_123517

theorem cos_four_pi_thirds_minus_alpha (α : ℝ) 
  (h : Real.sin (π / 6 + α) = 3 / 5) : 
  Real.cos (4 * π / 3 - α) = -3 / 5 := by
  sorry

end cos_four_pi_thirds_minus_alpha_l1235_123517


namespace min_value_of_expression_l1235_123520

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : -4*a - b + 1 = 0) :
  (1/a + 4/b) ≥ 16 := by
sorry

end min_value_of_expression_l1235_123520


namespace complex_exp_thirteen_pi_over_two_l1235_123587

theorem complex_exp_thirteen_pi_over_two (z : ℂ) : z = Complex.exp (13 * Real.pi * Complex.I / 2) → z = Complex.I := by
  sorry

end complex_exp_thirteen_pi_over_two_l1235_123587


namespace binomial_seven_one_l1235_123554

theorem binomial_seven_one : (7 : ℕ).choose 1 = 7 := by sorry

end binomial_seven_one_l1235_123554
