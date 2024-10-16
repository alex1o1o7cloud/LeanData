import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l291_29148

def large_number : ℕ := 3 * 10^500 - 2022 * 10^497 - 2022

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_large_number : sum_of_digits large_number = 4491 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l291_29148


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_absolute_value_equation_solution_difference_is_30_l291_29134

theorem absolute_value_equation_solution_difference : ℝ → Prop :=
  fun d => ∃ x₁ x₂ : ℝ,
    (|x₁ - 3| = 15) ∧
    (|x₂ - 3| = 15) ∧
    (x₁ ≠ x₂) ∧
    (d = |x₁ - x₂|) ∧
    (d = 30)

-- The proof is omitted
theorem absolute_value_equation_solution_difference_is_30 :
  absolute_value_equation_solution_difference 30 := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_absolute_value_equation_solution_difference_is_30_l291_29134


namespace NUMINAMATH_CALUDE_select_five_from_eight_l291_29170

theorem select_five_from_eight (n m : ℕ) (h1 : n = 8) (h2 : m = 5) :
  Nat.choose n m = 56 := by
  sorry

end NUMINAMATH_CALUDE_select_five_from_eight_l291_29170


namespace NUMINAMATH_CALUDE_sequence_squared_l291_29176

theorem sequence_squared (a : ℕ → ℝ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n > 0 → 4 * a n * a (n + 1) = (a n + a (n + 1) - 1)^2) ∧
  (∀ n : ℕ, n > 1 → a n > a (n - 1)) →
  ∀ n : ℕ, n > 0 → a n = n^2 := by
sorry

end NUMINAMATH_CALUDE_sequence_squared_l291_29176


namespace NUMINAMATH_CALUDE_all_propositions_false_l291_29194

-- Define a type for lines in 3D space
variable (Line : Type)

-- Define relationships between lines
variable (parallel : Line → Line → Prop)
variable (coplanar : Line → Line → Prop)
variable (intersect : Line → Line → Prop)

-- State the theorem
theorem all_propositions_false :
  (∀ a b c : Line,
    (parallel a b ∧ ¬coplanar a c) → ¬coplanar b c) ∧
  (∀ a b c : Line,
    (coplanar a b ∧ ¬coplanar b c) → ¬coplanar a c) ∧
  (∀ a b c : Line,
    (¬coplanar a b ∧ coplanar a c) → ¬coplanar b c) ∧
  (∀ a b c : Line,
    (¬coplanar a b ∧ ¬intersect b c) → ¬intersect a c) →
  False :=
sorry

end NUMINAMATH_CALUDE_all_propositions_false_l291_29194


namespace NUMINAMATH_CALUDE_f_neg_two_equals_nineteen_l291_29147

/-- Given a function f(x) = 2x^2 - 4x + 3, prove that f(-2) = 19 -/
theorem f_neg_two_equals_nineteen : 
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 - 4 * x + 3
  f (-2) = 19 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_equals_nineteen_l291_29147


namespace NUMINAMATH_CALUDE_sunny_lead_second_race_l291_29171

/-- Represents the race scenario between Sunny and Windy -/
structure RaceScenario where
  race_distance : ℝ
  first_race_lead : ℝ
  second_race_handicap : ℝ
  sunny_speed_reduction : ℝ

/-- Calculates Sunny's lead in the second race -/
def second_race_lead (scenario : RaceScenario) : ℝ :=
  sorry

/-- Theorem stating that Sunny finishes 12.5 meters ahead in the second race -/
theorem sunny_lead_second_race (scenario : RaceScenario) 
    (h1 : scenario.race_distance = 400)
    (h2 : scenario.first_race_lead = 50)
    (h3 : scenario.second_race_handicap = 50)
    (h4 : scenario.sunny_speed_reduction = 0.1) :
  second_race_lead scenario = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_sunny_lead_second_race_l291_29171


namespace NUMINAMATH_CALUDE_crayon_factory_output_l291_29155

/-- Calculates the number of boxes filled per hour in a crayon factory --/
def boxes_per_hour (num_colors : ℕ) (crayons_per_color_per_box : ℕ) (total_crayons_in_4_hours : ℕ) : ℕ :=
  let crayons_per_hour := total_crayons_in_4_hours / 4
  let crayons_per_box := num_colors * crayons_per_color_per_box
  crayons_per_hour / crayons_per_box

/-- Theorem stating that under given conditions, the factory fills 5 boxes per hour --/
theorem crayon_factory_output : 
  boxes_per_hour 4 2 160 = 5 := by
  sorry

end NUMINAMATH_CALUDE_crayon_factory_output_l291_29155


namespace NUMINAMATH_CALUDE_number_exceeding_percentage_l291_29118

theorem number_exceeding_percentage (x : ℝ) : x = 0.16 * x + 42 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_percentage_l291_29118


namespace NUMINAMATH_CALUDE_jane_crayons_l291_29167

/-- The number of crayons Jane ends up with after a hippopotamus eats some. -/
def crayons_left (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

/-- Theorem: If Jane starts with 87 crayons and 7 are eaten by a hippopotamus,
    she will end up with 80 crayons. -/
theorem jane_crayons : crayons_left 87 7 = 80 := by
  sorry

end NUMINAMATH_CALUDE_jane_crayons_l291_29167


namespace NUMINAMATH_CALUDE_min_value_2a_plus_b_l291_29106

theorem min_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (heq : 3 * a + b = a^2 + a * b) :
  ∀ x y : ℝ, x > 0 → y > 0 → 3 * x + y = x^2 + x * y → 2 * x + y ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_2a_plus_b_l291_29106


namespace NUMINAMATH_CALUDE_tulip_fraction_l291_29102

theorem tulip_fraction (total : ℕ) (yellow_ratio red_ratio pink_ratio : ℚ) : 
  total = 60 ∧
  yellow_ratio = 1/2 ∧
  red_ratio = 1/3 ∧
  pink_ratio = 1/4 →
  (total - (yellow_ratio * total) - 
   (red_ratio * (total - yellow_ratio * total)) - 
   (pink_ratio * (total - yellow_ratio * total - red_ratio * (total - yellow_ratio * total)))) / total = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_tulip_fraction_l291_29102


namespace NUMINAMATH_CALUDE_chess_pieces_arrangement_l291_29189

theorem chess_pieces_arrangement (total : ℕ) 
  (h1 : ∃ inner : ℕ, total = inner + 60)
  (h2 : ∃ outer : ℕ, 60 = outer + 32) : 
  total = 80 := by
sorry

end NUMINAMATH_CALUDE_chess_pieces_arrangement_l291_29189


namespace NUMINAMATH_CALUDE_cross_spectral_density_symmetry_l291_29159

/-- Cross-spectral density of two random functions -/
noncomputable def cross_spectral_density (X Y : ℝ → ℂ) (ω : ℝ) : ℂ := sorry

/-- Stationarity property for a random function -/
def stationary (X : ℝ → ℂ) : Prop := sorry

/-- Joint stationarity property for two random functions -/
def jointly_stationary (X Y : ℝ → ℂ) : Prop := sorry

/-- Theorem: For stationary and jointly stationary random functions, 
    the cross-spectral densities satisfy s_xy(-ω) = s_yx(ω) -/
theorem cross_spectral_density_symmetry 
  (X Y : ℝ → ℂ) (ω : ℝ) 
  (h1 : stationary X) (h2 : stationary Y) (h3 : jointly_stationary X Y) : 
  cross_spectral_density X Y (-ω) = cross_spectral_density Y X ω := by
  sorry

end NUMINAMATH_CALUDE_cross_spectral_density_symmetry_l291_29159


namespace NUMINAMATH_CALUDE_min_participants_is_100_l291_29120

/-- Represents the number of correct answers for each question in the quiz. -/
structure QuizResults where
  q1 : Nat
  q2 : Nat
  q3 : Nat
  q4 : Nat

/-- Calculates the minimum number of participants given quiz results. -/
def minParticipants (results : QuizResults) : Nat :=
  ((results.q1 + results.q2 + results.q3 + results.q4 + 1) / 2)

/-- Theorem: The minimum number of participants in the quiz is 100. -/
theorem min_participants_is_100 (results : QuizResults) 
  (h1 : results.q1 = 90)
  (h2 : results.q2 = 50)
  (h3 : results.q3 = 40)
  (h4 : results.q4 = 20)
  (h5 : ∀ n : Nat, n ≤ minParticipants results → 
       2 * n ≥ results.q1 + results.q2 + results.q3 + results.q4) :
  minParticipants results = 100 := by
  sorry

#eval minParticipants ⟨90, 50, 40, 20⟩

end NUMINAMATH_CALUDE_min_participants_is_100_l291_29120


namespace NUMINAMATH_CALUDE_prob_same_color_diff_foot_value_l291_29178

def total_pairs : ℕ := 15
def black_pairs : ℕ := 8
def brown_pairs : ℕ := 4
def gray_pairs : ℕ := 3

def total_shoes : ℕ := total_pairs * 2

def prob_same_color_diff_foot : ℚ :=
  (black_pairs * 2 * black_pairs) / (total_shoes * (total_shoes - 1)) +
  (brown_pairs * 2 * brown_pairs) / (total_shoes * (total_shoes - 1)) +
  (gray_pairs * 2 * gray_pairs) / (total_shoes * (total_shoes - 1))

theorem prob_same_color_diff_foot_value :
  prob_same_color_diff_foot = 89 / 435 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_diff_foot_value_l291_29178


namespace NUMINAMATH_CALUDE_complex_number_modulus_l291_29154

theorem complex_number_modulus (a : ℝ) (h1 : a < 0) :
  let z : ℂ := (3 * a * Complex.I) / (1 - 2 * Complex.I)
  Complex.abs z = Real.sqrt 5 → a = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l291_29154


namespace NUMINAMATH_CALUDE_walking_speed_problem_l291_29139

/-- Proves that given a circular track of 640 m, two people walking in opposite directions 
    from the same starting point, meeting after 4.8 minutes, with one person walking at 3.8 km/hr, 
    the other person's speed is 4.2 km/hr. -/
theorem walking_speed_problem (track_length : ℝ) (meeting_time : ℝ) (geeta_speed : ℝ) :
  track_length = 640 →
  meeting_time = 4.8 →
  geeta_speed = 3.8 →
  ∃ lata_speed : ℝ,
    lata_speed = 4.2 ∧
    (lata_speed + geeta_speed) * meeting_time / 60 = track_length / 1000 :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_problem_l291_29139


namespace NUMINAMATH_CALUDE_beth_class_students_left_l291_29199

/-- The number of students who left Beth's class in the final year -/
def students_left (initial : ℕ) (joined : ℕ) (final : ℕ) : ℕ :=
  initial + joined - final

theorem beth_class_students_left : 
  students_left 150 30 165 = 15 := by sorry

end NUMINAMATH_CALUDE_beth_class_students_left_l291_29199


namespace NUMINAMATH_CALUDE_sin_75_degrees_l291_29127

theorem sin_75_degrees : 
  let sin75 := Real.sin (75 * Real.pi / 180)
  let sin45 := Real.sin (45 * Real.pi / 180)
  let cos45 := Real.cos (45 * Real.pi / 180)
  let sin30 := Real.sin (30 * Real.pi / 180)
  let cos30 := Real.cos (30 * Real.pi / 180)
  sin75 = (Real.sqrt 6 + Real.sqrt 2) / 4 ∧
  sin45 = Real.sqrt 2 / 2 ∧
  cos45 = Real.sqrt 2 / 2 ∧
  sin30 = 1 / 2 ∧
  cos30 = Real.sqrt 3 / 2 ∧
  sin75 = sin45 * cos30 + cos45 * sin30 :=
by sorry


end NUMINAMATH_CALUDE_sin_75_degrees_l291_29127


namespace NUMINAMATH_CALUDE_dice_probability_l291_29197

-- Define a die
def Die := Fin 6

-- Define the sum of three dice rolls
def diceSum (d1 d2 d3 : Die) : ℕ := d1.val + d2.val + d3.val + 3

-- Define the condition for the sum to be even and greater than 15
def validRoll (d1 d2 d3 : Die) : Prop :=
  Even (diceSum d1 d2 d3) ∧ diceSum d1 d2 d3 > 15

-- Define the total number of possible outcomes
def totalOutcomes : ℕ := 216

-- Define the number of favorable outcomes
def favorableOutcomes : ℕ := 10

-- Theorem statement
theorem dice_probability :
  (favorableOutcomes : ℚ) / totalOutcomes = 5 / 108 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l291_29197


namespace NUMINAMATH_CALUDE_jefferson_high_club_overlap_l291_29175

/-- Represents the number of students in both robotics and science clubs -/
def students_in_both_clubs (total : ℕ) (robotics : ℕ) (science : ℕ) (either : ℕ) : ℕ :=
  robotics + science - either

/-- Theorem: Given the conditions from Jefferson High School, 
    prove that there are 20 students in both robotics and science clubs -/
theorem jefferson_high_club_overlap :
  students_in_both_clubs 300 80 130 190 = 20 := by
  sorry

end NUMINAMATH_CALUDE_jefferson_high_club_overlap_l291_29175


namespace NUMINAMATH_CALUDE_opposite_reciprocal_absolute_value_l291_29137

theorem opposite_reciprocal_absolute_value (a b c d m : ℝ) : 
  (a = -b) →  -- a and b are opposite numbers
  (c * d = 1) →  -- c and d are reciprocals
  (abs m = 5) →  -- absolute value of m is 5
  (-a - m * c * d - b = 5 ∨ -a - m * c * d - b = -5) :=
by
  sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_absolute_value_l291_29137


namespace NUMINAMATH_CALUDE_conference_handshakes_l291_29104

/-- The number of unique handshakes in a circular seating arrangement --/
def unique_handshakes (n : ℕ) : ℕ := n

/-- Theorem: In a circular seating arrangement with 30 people, 
    where each person shakes hands only with their immediate neighbors, 
    the number of unique handshakes is equal to 30. --/
theorem conference_handshakes : 
  unique_handshakes 30 = 30 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l291_29104


namespace NUMINAMATH_CALUDE_quadratic_increasing_implies_a_bound_l291_29136

/-- A function f is increasing on an interval [a, +∞) if for all x₁, x₂ in the interval with x₁ < x₂, f(x₁) < f(x₂) --/
def IncreasingOn (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x₁ x₂, a ≤ x₁ → x₁ < x₂ → f x₁ < f x₂

/-- The quadratic function f(x) = x^2 + 2ax + 1 --/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 1

theorem quadratic_increasing_implies_a_bound (a : ℝ) :
  IncreasingOn (f a) 2 → a ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_increasing_implies_a_bound_l291_29136


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l291_29179

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let parabola (x y : ℝ) := y^2 = 20*x
  let hyperbola (x y : ℝ) := x^2/a^2 - y^2/b^2 = 1
  let focus_parabola : ℝ × ℝ := (5, 0)
  let asymptote (x y : ℝ) := b*x + a*y = 0
  let distance_focus_asymptote := 4
  let eccentricity := (Real.sqrt (a^2 + b^2)) / a
  (∀ x y, parabola x y → hyperbola x y) →
  (distance_focus_asymptote = 4) →
  eccentricity = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l291_29179


namespace NUMINAMATH_CALUDE_total_catch_l291_29141

def johnny_catch : ℕ := 8

def sony_catch (johnny : ℕ) : ℕ := 4 * johnny

theorem total_catch (johnny : ℕ) (sony : ℕ → ℕ) 
  (h1 : johnny = johnny_catch) 
  (h2 : sony = sony_catch) : 
  sony johnny + johnny = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_catch_l291_29141


namespace NUMINAMATH_CALUDE_travelers_checks_theorem_l291_29160

/-- Represents the number of travelers checks of each denomination -/
structure TravelersChecks where
  fifty : ℕ
  hundred : ℕ

/-- The problem setup for the travelers checks -/
def travelersProblem (tc : TravelersChecks) : Prop :=
  tc.fifty + tc.hundred = 30 ∧
  50 * tc.fifty + 100 * tc.hundred = 1800

/-- The result of spending some $50 checks -/
def spendFiftyChecks (tc : TravelersChecks) (spent : ℕ) : TravelersChecks :=
  { fifty := tc.fifty - spent, hundred := tc.hundred }

/-- Calculate the average value of the remaining checks -/
def averageValue (tc : TravelersChecks) : ℚ :=
  (50 * tc.fifty + 100 * tc.hundred) / (tc.fifty + tc.hundred)

/-- The main theorem to prove -/
theorem travelers_checks_theorem (tc : TravelersChecks) :
  travelersProblem tc →
  averageValue (spendFiftyChecks tc 15) = 70 := by
  sorry


end NUMINAMATH_CALUDE_travelers_checks_theorem_l291_29160


namespace NUMINAMATH_CALUDE_triangle_perimeter_increase_l291_29144

/-- Given an initial equilateral triangle and four subsequent triangles with increasing side lengths,
    calculate the percent increase in perimeter from the first to the fifth triangle. -/
theorem triangle_perimeter_increase (initial_side : ℝ) (scale_factor : ℝ) (num_triangles : ℕ) :
  initial_side = 3 →
  scale_factor = 2 →
  num_triangles = 5 →
  let first_perimeter := 3 * initial_side
  let last_side := initial_side * scale_factor ^ (num_triangles - 1)
  let last_perimeter := 3 * last_side
  (last_perimeter - first_perimeter) / first_perimeter * 100 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_increase_l291_29144


namespace NUMINAMATH_CALUDE_sheila_hourly_rate_l291_29122

/-- Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_long_day : ℕ
  days_long : ℕ
  hours_short_day : ℕ
  days_short : ℕ
  weekly_earnings : ℕ

/-- Calculate hourly rate given a work schedule --/
def hourly_rate (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (schedule.hours_long_day * schedule.days_long + 
                              schedule.hours_short_day * schedule.days_short)

/-- Sheila's specific work schedule --/
def sheila_schedule : WorkSchedule := {
  hours_long_day := 8,
  days_long := 3,
  hours_short_day := 6,
  days_short := 2,
  weekly_earnings := 252
}

/-- Theorem: Sheila's hourly rate is $7 --/
theorem sheila_hourly_rate : hourly_rate sheila_schedule = 7 := by
  sorry


end NUMINAMATH_CALUDE_sheila_hourly_rate_l291_29122


namespace NUMINAMATH_CALUDE_horse_net_earnings_zero_l291_29132

/-- Represents the chessboard and the horse's movement rules --/
structure ChessboardGame where
  /-- The number of black squares the horse lands on --/
  black_squares : ℕ
  /-- The number of white squares the horse lands on --/
  white_squares : ℕ
  /-- Ensures the horse starts and ends on a white square --/
  start_end_white : white_squares > 0
  /-- Ensures the number of black and white squares are equal --/
  equal_squares : black_squares = white_squares
  /-- Represents the rule that the horse earns 2 carrots for each black square --/
  carrots_earned : ℕ := 2 * black_squares
  /-- Represents the rule that the horse pays 1 carrot for each move --/
  carrots_paid : ℕ := black_squares + white_squares

/-- The theorem stating that the net earnings of the horse is always 0 --/
theorem horse_net_earnings_zero (game : ChessboardGame) :
  game.carrots_earned - game.carrots_paid = 0 := by
  sorry

#check horse_net_earnings_zero

end NUMINAMATH_CALUDE_horse_net_earnings_zero_l291_29132


namespace NUMINAMATH_CALUDE_angle_a1fb1_is_right_angle_l291_29187

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- Point on a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Theorem: Angle A1FB1 is 90 degrees in a parabola -/
theorem angle_a1fb1_is_right_angle (parab : Parabola) 
  (focus : Point) 
  (directrix : ℝ) 
  (line : Line) 
  (a b : Point) 
  (a1 b1 : Point) :
  focus.x = parab.p / 2 →
  focus.y = 0 →
  directrix = -parab.p / 2 →
  parab.equation a.x a.y →
  parab.equation b.x b.y →
  line.p1 = focus →
  (line.p2 = a ∨ line.p2 = b) →
  a1.x = directrix →
  b1.x = directrix →
  a1.y = a.y →
  b1.y = b.y →
  -- The conclusion: ∠A1FB1 = 90°
  ∃ (angle : ℝ), angle = Real.pi / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_a1fb1_is_right_angle_l291_29187


namespace NUMINAMATH_CALUDE_max_final_number_l291_29121

/-- The game function that takes a list of integers and returns the largest prime divisor of their sum -/
def game (pair : List Nat) : Nat :=
  sorry

/-- Function to perform one round of the game on a list of numbers -/
def gameRound (numbers : List Nat) : List Nat :=
  sorry

/-- Function to play the game until only one number remains -/
def playUntilOne (numbers : List Nat) : Nat :=
  sorry

theorem max_final_number : 
  ∃ (finalPairing : List (List Nat)), 
    (finalPairing.join = List.range 32) ∧ 
    (∀ pair ∈ finalPairing, pair.length = 2) ∧
    (playUntilOne (finalPairing.map game) = 11) ∧
    (∀ otherPairing : List (List Nat), 
      (otherPairing.join = List.range 32) → 
      (∀ pair ∈ otherPairing, pair.length = 2) →
      playUntilOne (otherPairing.map game) ≤ 11) :=
sorry

end NUMINAMATH_CALUDE_max_final_number_l291_29121


namespace NUMINAMATH_CALUDE_spinning_class_duration_l291_29169

/-- Calculates the number of hours worked out in each spinning class. -/
def hours_per_class (classes_per_week : ℕ) (calories_per_minute : ℕ) (total_calories_per_week : ℕ) : ℚ :=
  (total_calories_per_week / classes_per_week) / (calories_per_minute * 60)

/-- Proves that given the specified conditions, James works out for 1.5 hours in each spinning class. -/
theorem spinning_class_duration :
  let classes_per_week : ℕ := 3
  let calories_per_minute : ℕ := 7
  let total_calories_per_week : ℕ := 1890
  hours_per_class classes_per_week calories_per_minute total_calories_per_week = 3/2 := by
  sorry

#eval hours_per_class 3 7 1890

end NUMINAMATH_CALUDE_spinning_class_duration_l291_29169


namespace NUMINAMATH_CALUDE_home_to_school_distance_proof_l291_29129

/-- The distance from Xiao Hong's home to her school -/
def home_to_school_distance : ℝ := 12000

/-- The distance the father drives Xiao Hong -/
def father_driving_distance : ℝ := 1000

/-- The time it takes Xiao Hong to get from home to school by car and walking -/
def car_and_walking_time : ℝ := 22.5

/-- The time it takes Xiao Hong to ride her bike from home to school -/
def bike_riding_time : ℝ := 40

/-- Xiao Hong's walking speed in meters per minute -/
def walking_speed : ℝ := 80

/-- The difference between father's driving speed and Xiao Hong's bike speed -/
def speed_difference : ℝ := 800

theorem home_to_school_distance_proof :
  home_to_school_distance = 12000 :=
sorry

end NUMINAMATH_CALUDE_home_to_school_distance_proof_l291_29129


namespace NUMINAMATH_CALUDE_croissant_count_is_two_l291_29157

/-- Represents the number of items bought at each price point -/
structure ItemCounts where
  expensive : ℕ
  cheap : ℕ

/-- Calculates the total cost given the number of items at each price point -/
def totalCost (counts : ItemCounts) : ℚ :=
  1.5 * counts.expensive + 1.2 * counts.cheap

/-- Checks if a rational number is a whole number -/
def isWholeNumber (q : ℚ) : Prop :=
  ∃ n : ℤ, q = n

/-- The main theorem to be proved -/
theorem croissant_count_is_two :
  ∀ counts : ItemCounts,
    counts.expensive + counts.cheap = 7 →
    isWholeNumber (totalCost counts) →
    counts.expensive = 2 :=
by sorry

end NUMINAMATH_CALUDE_croissant_count_is_two_l291_29157


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l291_29103

/-- The trajectory of the midpoint of a line segment between a point on a circle and a fixed point -/
theorem midpoint_trajectory (x₀ y₀ x y : ℝ) : 
  x₀^2 + y₀^2 = 4 →  -- P is on the circle x^2 + y^2 = 4
  x = (x₀ + 8) / 2 →  -- x-coordinate of midpoint M
  y = y₀ / 2 →  -- y-coordinate of midpoint M
  (x - 4)^2 + y^2 = 1 :=  -- Trajectory equation
by sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l291_29103


namespace NUMINAMATH_CALUDE_interest_calculation_l291_29192

/-- Calculates the total interest earned on an investment --/
def total_interest_earned (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

/-- Proves that the total interest earned is approximately $563.16 --/
theorem interest_calculation : 
  let principal := 1200
  let rate := 0.08
  let time := 5
  abs (total_interest_earned principal rate time - 563.16) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_interest_calculation_l291_29192


namespace NUMINAMATH_CALUDE_anna_ham_sandwich_problem_l291_29161

/-- The number of additional ham slices Anna needs to make a certain number of sandwiches -/
def additional_slices (slices_per_sandwich : ℕ) (current_slices : ℕ) (desired_sandwiches : ℕ) : ℕ :=
  slices_per_sandwich * desired_sandwiches - current_slices

theorem anna_ham_sandwich_problem : 
  additional_slices 3 31 50 = 119 := by
  sorry

end NUMINAMATH_CALUDE_anna_ham_sandwich_problem_l291_29161


namespace NUMINAMATH_CALUDE_distinct_scores_l291_29133

def goals : ℕ := 7

def possible_scores (n : ℕ) : Finset ℕ :=
  Finset.image (λ y : ℕ => y + n) (Finset.range (n + 1))

theorem distinct_scores : Finset.card (possible_scores goals) = 8 := by
  sorry

end NUMINAMATH_CALUDE_distinct_scores_l291_29133


namespace NUMINAMATH_CALUDE_derivative_of_y_l291_29138

noncomputable def y (x : ℝ) : ℝ := Real.sin (2 * x) - Real.cos (2 * x)

theorem derivative_of_y (x : ℝ) :
  deriv y x = 2 * Real.sqrt 2 * Real.cos (2 * x - Real.pi / 4) := by sorry

end NUMINAMATH_CALUDE_derivative_of_y_l291_29138


namespace NUMINAMATH_CALUDE_church_female_adults_l291_29153

/-- Calculates the number of female adults in a church given the total number of people,
    number of children, and number of male adults. -/
def female_adults (total : ℕ) (children : ℕ) (male_adults : ℕ) : ℕ :=
  total - (children + male_adults)

/-- Theorem stating that the number of female adults in the church is 60. -/
theorem church_female_adults :
  female_adults 200 80 60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_church_female_adults_l291_29153


namespace NUMINAMATH_CALUDE_father_son_age_difference_l291_29182

theorem father_son_age_difference : ∀ (father_age son_age : ℕ),
  son_age = 33 →
  father_age + 2 = 2 * (son_age + 2) →
  father_age - son_age = 35 := by
  sorry

end NUMINAMATH_CALUDE_father_son_age_difference_l291_29182


namespace NUMINAMATH_CALUDE_pencil_length_after_sharpening_l291_29174

def initial_length : ℕ := 100
def monday_sharpening : ℕ := 3
def tuesday_sharpening : ℕ := 5
def wednesday_sharpening : ℕ := 7
def thursday_sharpening : ℕ := 11
def friday_sharpening : ℕ := 13

theorem pencil_length_after_sharpening : 
  initial_length - (monday_sharpening + tuesday_sharpening + wednesday_sharpening + thursday_sharpening + friday_sharpening) = 61 := by
  sorry

end NUMINAMATH_CALUDE_pencil_length_after_sharpening_l291_29174


namespace NUMINAMATH_CALUDE_right_triangle_arithmetic_sequence_l291_29143

theorem right_triangle_arithmetic_sequence (b k : ℝ) (h_k_pos : k > 0) : 
  (b - k) > 0 ∧ (b + k)^2 = (b - k)^2 + b^2 → b = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_arithmetic_sequence_l291_29143


namespace NUMINAMATH_CALUDE_variance_invariant_under_translation_mutually_exclusive_events_l291_29123

-- Define a dataset as a list of real numbers
def Dataset := List Real

-- Define variance function
noncomputable def variance (data : Dataset) : Real := sorry

-- Define a function to add a constant to each element of a dataset
def addConstant (data : Dataset) (c : Real) : Dataset := sorry

-- Statement 1: Variance remains unchanged after adding a constant
theorem variance_invariant_under_translation (data : Dataset) (c : Real) :
  variance (addConstant data c) = variance data := by sorry

-- Define a type for students
inductive Student
| Boy
| Girl

-- Define a function to create a group of students
def createGroup (numBoys numGirls : Nat) : List Student := sorry

-- Define a function to select n students from a group
def selectStudents (group : List Student) (n : Nat) : List (List Student) := sorry

-- Define predicates for the events
def atLeastOneGirl (selection : List Student) : Prop := sorry
def allBoys (selection : List Student) : Prop := sorry

-- Statement 2: "At least 1 girl" and "all boys" are mutually exclusive when selecting 2 from 3 boys and 2 girls
theorem mutually_exclusive_events :
  let group := createGroup 3 2
  let selections := selectStudents group 2
  ∀ selection ∈ selections, ¬(atLeastOneGirl selection ∧ allBoys selection) := by sorry

end NUMINAMATH_CALUDE_variance_invariant_under_translation_mutually_exclusive_events_l291_29123


namespace NUMINAMATH_CALUDE_average_speed_not_necessarily_five_l291_29184

/-- A pedestrian's walk with varying speeds over 2.5 hours -/
structure PedestrianWalk where
  duration : ℝ
  hourly_distance : ℝ
  average_speed : ℝ

/-- Axiom: The pedestrian walks for 2.5 hours -/
axiom walk_duration : ∀ (w : PedestrianWalk), w.duration = 2.5

/-- Axiom: The pedestrian covers 5 km in any one-hour interval -/
axiom hourly_distance : ∀ (w : PedestrianWalk), w.hourly_distance = 5

/-- Theorem: The average speed for the entire journey is not necessarily 5 km per hour -/
theorem average_speed_not_necessarily_five :
  ∃ (w : PedestrianWalk), w.average_speed ≠ 5 := by
  sorry


end NUMINAMATH_CALUDE_average_speed_not_necessarily_five_l291_29184


namespace NUMINAMATH_CALUDE_ellipse_condition_l291_29163

def is_ellipse_equation (m : ℝ) : Prop :=
  (m - 2 > 0) ∧ (6 - m > 0) ∧ (m - 2 ≠ 6 - m)

theorem ellipse_condition (m : ℝ) :
  (is_ellipse_equation m → m ∈ Set.Ioo 2 6) ∧
  (∃ m ∈ Set.Ioo 2 6, ¬is_ellipse_equation m) :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l291_29163


namespace NUMINAMATH_CALUDE_coordinate_sum_theorem_l291_29191

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

-- Theorem statement
theorem coordinate_sum_theorem (h : 3 * (f 2) = 5) :
  2 * (f_inv (5/3)) = 4 ∧ 5/3 + 4 = 17/3 := by
  sorry

end NUMINAMATH_CALUDE_coordinate_sum_theorem_l291_29191


namespace NUMINAMATH_CALUDE_complement_N_subset_M_l291_29112

open Set

-- Define the sets M and N
def M : Set ℝ := {x | x * (x - 3) < 0}
def N : Set ℝ := {x | x < 1 ∨ x ≥ 3}

-- State the theorem
theorem complement_N_subset_M : (𝒰 \ N) ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_complement_N_subset_M_l291_29112


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l291_29183

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  arithmetic_sequence a →
  a 4 = 3 →
  a 5 = 7 →
  a 6 = 11 →
  (a 1) + (a 2) + (a 3) + (a 4) + (a 5) = -5 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l291_29183


namespace NUMINAMATH_CALUDE_joe_running_speed_l291_29196

/-- Proves that Joe's running speed is 16 km/h given the problem conditions -/
theorem joe_running_speed (pete_speed : ℝ) : 
  pete_speed > 0 →
  (2 * pete_speed * (40 / 60) + pete_speed * (40 / 60) = 16) →
  2 * pete_speed = 16 := by
  sorry

#check joe_running_speed

end NUMINAMATH_CALUDE_joe_running_speed_l291_29196


namespace NUMINAMATH_CALUDE_basketball_shooting_improvement_l291_29198

theorem basketball_shooting_improvement (initial_shots : ℕ) (initial_made : ℕ) (next_game_shots : ℕ) (new_average : ℚ) : 
  initial_shots = 35 → 
  initial_made = 15 → 
  next_game_shots = 15 → 
  new_average = 11/20 → 
  ∃ (next_game_made : ℕ), 
    next_game_made = 13 ∧ 
    (initial_made + next_game_made : ℚ) / (initial_shots + next_game_shots : ℚ) = new_average :=
by sorry

#check basketball_shooting_improvement

end NUMINAMATH_CALUDE_basketball_shooting_improvement_l291_29198


namespace NUMINAMATH_CALUDE_log_sum_equality_l291_29146

theorem log_sum_equality : Real.log 50 + Real.log 30 = 3 + Real.log 1.5 := by sorry

end NUMINAMATH_CALUDE_log_sum_equality_l291_29146


namespace NUMINAMATH_CALUDE_cricket_team_captain_age_l291_29149

theorem cricket_team_captain_age
  (team_size : ℕ)
  (captain_age : ℕ)
  (wicket_keeper_age : ℕ)
  (team_average_age : ℕ)
  (h1 : team_size = 11)
  (h2 : wicket_keeper_age = captain_age + 3)
  (h3 : (team_size - 2) * (team_average_age - 1) = team_size * team_average_age - captain_age - wicket_keeper_age)
  (h4 : team_average_age = 23) :
  captain_age = 26 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_captain_age_l291_29149


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l291_29108

/-- The volume of a rectangular parallelepiped with given diagonal, angle, and base perimeter. -/
theorem parallelepiped_volume (l P α : ℝ) (hl : l > 0) (hP : P > 0) (hα : 0 < α ∧ α < π / 2) :
  ∃ V : ℝ, V = (l * (P^2 - 4 * l^2 * Real.sin α ^ 2) * Real.cos α) / 8 ∧
    V > 0 ∧
    ∀ (x y h : ℝ),
      x > 0 → y > 0 → h > 0 →
      x + y = P / 2 →
      x^2 + y^2 = l^2 * Real.sin α ^ 2 →
      h = l * Real.cos α →
      V = x * y * h :=
by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_l291_29108


namespace NUMINAMATH_CALUDE_quadratic_transform_l291_29117

theorem quadratic_transform (p q r : ℝ) :
  (∃ m l : ℝ, ∀ x : ℝ, px^2 + qx + r = 5*(x - 3)^2 + 15 ∧ 2*px^2 + 2*qx + 2*r = m*(x - 3)^2 + l) →
  (∃ m l : ℝ, ∀ x : ℝ, 2*px^2 + 2*qx + 2*r = m*(x - 3)^2 + l) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transform_l291_29117


namespace NUMINAMATH_CALUDE_scientific_notation_of_small_number_l291_29140

theorem scientific_notation_of_small_number :
  ∃ (a : ℝ) (n : ℤ), 0.000815 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ n = -4 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_small_number_l291_29140


namespace NUMINAMATH_CALUDE_science_quiz_bowl_participation_l291_29177

/-- The Science Quiz Bowl Participation Problem -/
theorem science_quiz_bowl_participation (participants_2018 : ℕ) : 
  participants_2018 = 150 → 
  ∃ (participants_2019 participants_2020 : ℕ),
    participants_2019 = 2 * participants_2018 + 20 ∧
    participants_2020 = participants_2019 / 2 - 40 ∧
    participants_2019 - participants_2020 = 200 := by
  sorry

end NUMINAMATH_CALUDE_science_quiz_bowl_participation_l291_29177


namespace NUMINAMATH_CALUDE_derivative_at_one_l291_29195

open Real

noncomputable def f (x : ℝ) (f'1 : ℝ) : ℝ := 2 * x * f'1 + log x

theorem derivative_at_one (f'1 : ℝ) :
  (∀ x > 0, f x f'1 = 2 * x * f'1 + log x) →
  deriv (f · f'1) 1 = -1 :=
by sorry

end NUMINAMATH_CALUDE_derivative_at_one_l291_29195


namespace NUMINAMATH_CALUDE_unique_solution_condition_l291_29142

theorem unique_solution_condition (j : ℝ) : 
  (∃! x : ℝ, (3 * x + 4) * (x - 6) = -52 + j * x) ↔ 
  (j = -14 + 4 * Real.sqrt 21 ∨ j = -14 - 4 * Real.sqrt 21) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l291_29142


namespace NUMINAMATH_CALUDE_crow_eating_quarter_l291_29128

/-- Represents the time it takes for a crow to eat a certain fraction of nuts -/
def crow_eating_time (fraction_eaten : ℚ) (time : ℚ) : Prop :=
  fraction_eaten * time = (1 : ℚ) / 5 * 6

/-- Proves that it takes 7.5 hours for a crow to eat 1/4 of the nuts, 
    given that it eats 1/5 of the nuts in 6 hours -/
theorem crow_eating_quarter : 
  crow_eating_time (1 / 4) (15 / 2) :=
sorry

end NUMINAMATH_CALUDE_crow_eating_quarter_l291_29128


namespace NUMINAMATH_CALUDE_math_homework_pages_l291_29125

theorem math_homework_pages 
  (total_pages : ℕ) 
  (math_pages : ℕ) 
  (reading_pages : ℕ) 
  (problems_per_page : ℕ) 
  (total_problems : ℕ) :
  total_pages = math_pages + reading_pages →
  reading_pages = 6 →
  problems_per_page = 4 →
  total_problems = 40 →
  math_pages = 4 := by
sorry

end NUMINAMATH_CALUDE_math_homework_pages_l291_29125


namespace NUMINAMATH_CALUDE_cats_asleep_l291_29126

theorem cats_asleep (total : ℕ) (awake : ℕ) (h1 : total = 98) (h2 : awake = 6) :
  total - awake = 92 := by
  sorry

end NUMINAMATH_CALUDE_cats_asleep_l291_29126


namespace NUMINAMATH_CALUDE_some_number_value_l291_29162

theorem some_number_value (x : ℝ) : 65 + 5 * 12 / (x / 3) = 66 → x = 180 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l291_29162


namespace NUMINAMATH_CALUDE_prob_different_suits_no_jokers_l291_29193

/-- Extended deck with 54 cards including 2 jokers -/
def extendedDeck : ℕ := 54

/-- Number of jokers in the extended deck -/
def jokers : ℕ := 2

/-- Number of suits in a standard deck -/
def numSuits : ℕ := 4

/-- Number of cards per suit in a standard deck -/
def cardsPerSuit : ℕ := 13

/-- Probability of picking two cards of different suits given no jokers are picked -/
theorem prob_different_suits_no_jokers :
  let nonJokerCards := extendedDeck - jokers
  let firstPickOptions := nonJokerCards
  let secondPickOptions := nonJokerCards - 1
  let differentSuitOptions := (numSuits - 1) * cardsPerSuit
  (differentSuitOptions : ℚ) / secondPickOptions = 13 / 17 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_suits_no_jokers_l291_29193


namespace NUMINAMATH_CALUDE_periodic_sine_function_l291_29168

theorem periodic_sine_function (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = Real.sin (2 * x - π / 4)) →
  a ∈ Set.Ioo 0 π →
  (∀ x, f (x + a) = f (x + 3 * a)) →
  a = π / 2 := by
sorry

end NUMINAMATH_CALUDE_periodic_sine_function_l291_29168


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l291_29110

theorem magnitude_of_complex_fraction (z : ℂ) : z = (2 + Complex.I) / Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l291_29110


namespace NUMINAMATH_CALUDE_cloth_selling_price_l291_29114

/-- Calculates the total selling price of cloth given the length, profit per meter, and cost price per meter. -/
def total_selling_price (length : ℕ) (profit_per_meter : ℕ) (cost_per_meter : ℕ) : ℕ :=
  length * (profit_per_meter + cost_per_meter)

/-- Proves that the total selling price of 45 meters of cloth is 4500 rupees,
    given a profit of 12 rupees per meter and a cost price of 88 rupees per meter. -/
theorem cloth_selling_price :
  total_selling_price 45 12 88 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l291_29114


namespace NUMINAMATH_CALUDE_inscribed_circle_path_length_l291_29173

theorem inscribed_circle_path_length (a b c : ℝ) (h_triangle : a = 10 ∧ b = 8 ∧ c = 12) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r := area / s
  (a + b + c) - 2 * r = 15 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_path_length_l291_29173


namespace NUMINAMATH_CALUDE_arcsin_arccos_equation_solutions_l291_29100

theorem arcsin_arccos_equation_solutions :
  ∀ x : ℝ, (x = 0 ∨ x = 1/2 ∨ x = -1/2) →
  Real.arcsin (2*x) + Real.arcsin (1 - 2*x) = Real.arccos (2*x) := by
sorry

end NUMINAMATH_CALUDE_arcsin_arccos_equation_solutions_l291_29100


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l291_29111

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 12 ≥ 0} = {x : ℝ | x ≤ -4 ∨ x ≥ 3} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l291_29111


namespace NUMINAMATH_CALUDE_fibFactorial_characterization_l291_29165

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Set of positive integers n for which n! is the product of two Fibonacci numbers -/
def fibFactorialSet : Set ℕ :=
  {n : ℕ | n > 0 ∧ ∃ k m : ℕ, n.factorial = fib k * fib m}

/-- Theorem stating that fibFactorialSet contains exactly 1, 2, 3, 4, and 6 -/
theorem fibFactorial_characterization :
    fibFactorialSet = {1, 2, 3, 4, 6} := by
  sorry

end NUMINAMATH_CALUDE_fibFactorial_characterization_l291_29165


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l291_29151

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line with 60° inclination passing through the focus
def line (x y : ℝ) : Prop := y = Real.sqrt 3 * (x - 1)

-- Define a point in the first quadrant
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Theorem statement
theorem parabola_line_intersection :
  ∀ (x y : ℝ),
  parabola x y →
  line x y →
  first_quadrant x y →
  Real.sqrt ((x - 1)^2 + y^2) = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l291_29151


namespace NUMINAMATH_CALUDE_oranges_picked_sum_l291_29172

/-- Given the number of oranges picked by Joan and Sara, prove that their sum
    equals the total number of oranges picked. -/
theorem oranges_picked_sum (joan_oranges sara_oranges total_oranges : ℕ)
  (h1 : joan_oranges = 37)
  (h2 : sara_oranges = 10)
  (h3 : total_oranges = 47) :
  joan_oranges + sara_oranges = total_oranges :=
by sorry

end NUMINAMATH_CALUDE_oranges_picked_sum_l291_29172


namespace NUMINAMATH_CALUDE_condition_equiv_range_l291_29185

/-- The set A in the real numbers -/
def A : Set ℝ := {x | -5 < x ∧ x < 4}

/-- The set B in the real numbers -/
def B : Set ℝ := {x | x < -6 ∨ x > 1}

/-- The set C in the real numbers, parameterized by m -/
def C (m : ℝ) : Set ℝ := {x | x < m}

/-- The theorem stating the equivalence of the conditions and the range of m -/
theorem condition_equiv_range :
  ∀ m : ℝ,
  (C m ⊇ (A ∩ B) ∧ C m ⊇ (Aᶜ ∩ Bᶜ)) ↔ m ∈ Set.Ici 4 := by
  sorry

end NUMINAMATH_CALUDE_condition_equiv_range_l291_29185


namespace NUMINAMATH_CALUDE_tan_product_pi_eighths_l291_29130

theorem tan_product_pi_eighths : 
  Real.tan (π / 8) * Real.tan (3 * π / 8) * Real.tan (5 * π / 8) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_pi_eighths_l291_29130


namespace NUMINAMATH_CALUDE_ariana_flowers_l291_29113

theorem ariana_flowers (total : ℕ) 
  (h1 : 2 * total = 5 * (total - 10 - 14)) -- 2/5 of flowers were roses
  (h2 : 10 ≤ total) -- 10 flowers were tulips
  (h3 : 14 ≤ total - 10) -- 14 flowers were carnations
  : total = 40 := by sorry

end NUMINAMATH_CALUDE_ariana_flowers_l291_29113


namespace NUMINAMATH_CALUDE_sixth_quiz_score_achieves_target_mean_l291_29152

def quiz_scores : List ℕ := [75, 80, 85, 90, 100]
def target_mean : ℕ := 95
def num_quizzes : ℕ := 6
def sixth_score : ℕ := 140

theorem sixth_quiz_score_achieves_target_mean :
  (List.sum quiz_scores + sixth_score) / num_quizzes = target_mean := by
  sorry

end NUMINAMATH_CALUDE_sixth_quiz_score_achieves_target_mean_l291_29152


namespace NUMINAMATH_CALUDE_right_triangle_345_l291_29109

theorem right_triangle_345 : ∃ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2 :=
  sorry

end NUMINAMATH_CALUDE_right_triangle_345_l291_29109


namespace NUMINAMATH_CALUDE_distance_between_A_and_B_l291_29190

-- Define the position of point A
def A : ℝ := 3

-- Define the possible positions of point B
def B : Set ℝ := {-9, 9}

-- Define the distance function
def distance (x y : ℝ) : ℝ := |x - y|

-- Theorem statement
theorem distance_between_A_and_B :
  ∀ b ∈ B, distance A b = 6 ∨ distance A b = 12 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_A_and_B_l291_29190


namespace NUMINAMATH_CALUDE_min_p_plus_q_l291_29166

theorem min_p_plus_q (p q : ℕ+) (h : 98 * p = q^3) : 
  ∀ (p' q' : ℕ+), 98 * p' = q'^3 → p + q ≤ p' + q' :=
by
  sorry

#check min_p_plus_q

end NUMINAMATH_CALUDE_min_p_plus_q_l291_29166


namespace NUMINAMATH_CALUDE_equation_solution_l291_29186

theorem equation_solution : 
  {x : ℝ | (x ≠ 0 ∧ x + 2 ≠ 0 ∧ x + 4 ≠ 0 ∧ x + 6 ≠ 0 ∧ x + 8 ≠ 0) ∧ 
           (1/x + 1/(x+2) - 1/(x+4) - 1/(x+6) + 1/(x+8) = 0)} = 
  {-4 - 2 * Real.sqrt 3, 2 - 2 * Real.sqrt 3} := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l291_29186


namespace NUMINAMATH_CALUDE_ball_distribution_theorem_l291_29116

/-- Represents the distribution of painted balls among different colors. -/
structure BallDistribution where
  totalBalls : ℕ
  numColors : ℕ
  equalColorCount : ℕ
  doubleColorCount : ℕ
  ballsPerEqualColor : ℕ
  ballsPerDoubleColor : ℕ

/-- Theorem stating the correct distribution of balls among colors. -/
theorem ball_distribution_theorem (d : BallDistribution) : 
  d.totalBalls = 600 ∧ 
  d.numColors = 15 ∧ 
  d.equalColorCount = 10 ∧ 
  d.doubleColorCount = 5 ∧ 
  d.ballsPerDoubleColor = 2 * d.ballsPerEqualColor →
  d.ballsPerEqualColor = 30 ∧ 
  d.ballsPerDoubleColor = 60 ∧
  d.totalBalls = d.equalColorCount * d.ballsPerEqualColor + d.doubleColorCount * d.ballsPerDoubleColor :=
by sorry

end NUMINAMATH_CALUDE_ball_distribution_theorem_l291_29116


namespace NUMINAMATH_CALUDE_unique_three_digit_number_with_digit_property_l291_29131

/-- Calculate the total number of digits used to write all integers from 1 to n -/
def totalDigits (n : ℕ) : ℕ :=
  if n < 10 then n
  else if n < 100 then 9 + 2 * (n - 9)
  else 189 + 3 * (n - 99)

/-- The property that a number, when doubled, equals the total digits required to write all numbers up to itself -/
def hasDigitProperty (n : ℕ) : Prop :=
  2 * n = totalDigits n

theorem unique_three_digit_number_with_digit_property :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ hasDigitProperty n ∧ n = 108 :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_with_digit_property_l291_29131


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l291_29164

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the problem statement
theorem complex_modulus_problem (z : ℂ) (h : (1 + i) * z = 2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l291_29164


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l291_29181

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def ends_in (n : ℕ) (d : ℕ) : Prop := n % 10 = d

theorem unique_number_satisfying_conditions : 
  ∃! n : ℕ, is_two_digit n ∧ 
    ((ends_in n 6 ∨ n % 7 = 0) ∧ ¬(ends_in n 6 ∧ n % 7 = 0)) ∧
    ((n > 26 ∨ ends_in n 8) ∧ ¬(n > 26 ∧ ends_in n 8)) ∧
    ((n % 13 = 0 ∨ n < 27) ∧ ¬(n % 13 = 0 ∧ n < 27)) ∧
    n = 91 := by
  sorry

#check unique_number_satisfying_conditions

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l291_29181


namespace NUMINAMATH_CALUDE_probability_same_gender_l291_29105

def total_volunteers : ℕ := 5
def male_volunteers : ℕ := 3
def female_volunteers : ℕ := 2
def volunteers_needed : ℕ := 2

def same_gender_combinations : ℕ := (male_volunteers.choose volunteers_needed) + (female_volunteers.choose volunteers_needed)
def total_combinations : ℕ := total_volunteers.choose volunteers_needed

theorem probability_same_gender :
  (same_gender_combinations : ℚ) / total_combinations = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_same_gender_l291_29105


namespace NUMINAMATH_CALUDE_project_completion_days_l291_29150

/-- Calculates the number of days required to complete a project given normal work hours, 
    extra work hours, and total project hours. -/
theorem project_completion_days 
  (normal_hours : ℕ) 
  (extra_hours : ℕ) 
  (total_project_hours : ℕ) 
  (h1 : normal_hours = 10)
  (h2 : extra_hours = 5)
  (h3 : total_project_hours = 1500) : 
  total_project_hours / (normal_hours + extra_hours) = 100 := by
  sorry

end NUMINAMATH_CALUDE_project_completion_days_l291_29150


namespace NUMINAMATH_CALUDE_beths_marbles_l291_29101

/-- Proves that given the conditions of Beth's marble problem, she initially had 72 marbles. -/
theorem beths_marbles (initial_per_color : ℕ) : 
  (3 * initial_per_color) - (5 + 10 + 15) = 42 → 
  3 * initial_per_color = 72 := by
  sorry

end NUMINAMATH_CALUDE_beths_marbles_l291_29101


namespace NUMINAMATH_CALUDE_incircle_circumcircle_ratio_bound_incircle_circumcircle_ratio_bound_tight_l291_29188

/-- The ratio of the incircle radius to the circumcircle radius of a right triangle is at most √2 - 1 -/
theorem incircle_circumcircle_ratio_bound (a b c : ℝ) (h_right : a^2 + b^2 = c^2) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  (a + b - c) / c ≤ Real.sqrt 2 - 1 :=
sorry

/-- The upper bound √2 - 1 is achievable for the ratio of incircle to circumcircle radius in a right triangle -/
theorem incircle_circumcircle_ratio_bound_tight :
  ∃ (a b c : ℝ), a^2 + b^2 = c^2 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ (a + b - c) / c = Real.sqrt 2 - 1 :=
sorry

end NUMINAMATH_CALUDE_incircle_circumcircle_ratio_bound_incircle_circumcircle_ratio_bound_tight_l291_29188


namespace NUMINAMATH_CALUDE_inequality_range_l291_29180

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x - 2 < 0) ↔ -8 < m ∧ m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_inequality_range_l291_29180


namespace NUMINAMATH_CALUDE_point_with_specific_rate_of_change_l291_29145

/-- The curve function -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 5

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 2*x - 3

theorem point_with_specific_rate_of_change :
  ∃ (x y : ℝ), f x = y ∧ f' x = 5 ∧ x = 4 ∧ y = 9 := by sorry

end NUMINAMATH_CALUDE_point_with_specific_rate_of_change_l291_29145


namespace NUMINAMATH_CALUDE_f_properties_l291_29119

def f (x : ℝ) : ℝ := |x + 3| + |x - 2|

theorem f_properties :
  (∀ x, f x > 7 ↔ x < -4 ∨ x > 3) ∧
  (∀ m, m > 1 → ∃ x, f x = 4 / (m - 1) + m) := by sorry

end NUMINAMATH_CALUDE_f_properties_l291_29119


namespace NUMINAMATH_CALUDE_print_shop_cost_difference_l291_29107

/-- The cost difference between two print shops for a given number of copies -/
def cost_difference (price_x price_y : ℚ) (num_copies : ℕ) : ℚ :=
  (price_y - price_x) * num_copies

/-- Theorem stating the cost difference between print shops Y and X for 40 copies -/
theorem print_shop_cost_difference :
  cost_difference (120/100) (170/100) 40 = 20 := by
  sorry

end NUMINAMATH_CALUDE_print_shop_cost_difference_l291_29107


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l291_29124

/-- The number of cards --/
def n : ℕ := 7

/-- The special card that must be at the beginning or end --/
def special_card : ℕ := 7

/-- The number of cards that will remain after removal --/
def remaining_cards : ℕ := 5

/-- The number of possible positions for the special card --/
def special_card_positions : ℕ := 2

/-- The number of ways to choose a card to remove from the non-special cards --/
def removal_choices : ℕ := n - 1

/-- The number of permutations of the remaining cards --/
def remaining_permutations : ℕ := remaining_cards.factorial

/-- The number of possible orderings (ascending or descending) --/
def possible_orderings : ℕ := 2

/-- The total number of valid arrangements --/
def valid_arrangements : ℕ := 
  special_card_positions * removal_choices * remaining_permutations * possible_orderings

theorem valid_arrangements_count : valid_arrangements = 2880 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l291_29124


namespace NUMINAMATH_CALUDE_chess_tournament_games_l291_29115

theorem chess_tournament_games (n : ℕ) (h : n = 9) : 
  (n * (n - 1)) / 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l291_29115


namespace NUMINAMATH_CALUDE_no_natural_number_power_of_two_l291_29135

theorem no_natural_number_power_of_two : 
  ∀ n : ℕ, ¬∃ k : ℕ, n^2012 - 1 = 2^k := by
  sorry

end NUMINAMATH_CALUDE_no_natural_number_power_of_two_l291_29135


namespace NUMINAMATH_CALUDE_seashell_sale_theorem_l291_29158

/-- Calculates the total money earned from selling items collected over two days -/
def total_money (day1_items : ℕ) (price_per_item : ℚ) : ℚ :=
  let day2_items := day1_items / 2
  let total_items := day1_items + day2_items
  total_items * price_per_item

/-- Proves that collecting 30 items on day 1, half as many on day 2, 
    and selling each for $1.20 results in $54 total -/
theorem seashell_sale_theorem :
  total_money 30 (6/5) = 54 := by
  sorry

end NUMINAMATH_CALUDE_seashell_sale_theorem_l291_29158


namespace NUMINAMATH_CALUDE_simplify_expression_l291_29156

theorem simplify_expression (z : ℝ) : (3 - 5*z^2) - (4*z^2 + 2*z - 5) = 8 - 9*z^2 - 2*z := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l291_29156
