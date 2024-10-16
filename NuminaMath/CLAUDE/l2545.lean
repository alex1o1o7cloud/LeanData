import Mathlib

namespace NUMINAMATH_CALUDE_josh_bought_four_cookies_l2545_254526

/-- Calculates the number of cookies Josh bought given his initial money,
    the cost of other items, the cost per cookie, and the remaining money. -/
def cookies_bought (initial_money : ℚ) (hat_cost : ℚ) (pencil_cost : ℚ)
                   (cookie_cost : ℚ) (remaining_money : ℚ) : ℚ :=
  ((initial_money - hat_cost - pencil_cost - remaining_money) / cookie_cost)

/-- Proves that Josh bought 4 cookies given the problem conditions. -/
theorem josh_bought_four_cookies :
  cookies_bought 20 10 2 1.25 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_josh_bought_four_cookies_l2545_254526


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l2545_254532

/-- Given a jar with blue, red, and yellow marbles, this theorem proves
    the number of yellow marbles, given the number of blue and red marbles
    and the probability of picking a yellow marble. -/
theorem yellow_marbles_count
  (blue : ℕ) (red : ℕ) (prob_yellow : ℚ)
  (h_blue : blue = 7)
  (h_red : red = 11)
  (h_prob : prob_yellow = 1/4) :
  ∃ (yellow : ℕ), yellow = 6 ∧
    prob_yellow = yellow / (blue + red + yellow) :=
by sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l2545_254532


namespace NUMINAMATH_CALUDE_p2023_coordinates_l2545_254536

/-- Transformation function that maps a point (x, y) to (-y+1, x+2) -/
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2 + 1, p.1 + 2)

/-- Function to apply the transformation n times -/
def iterate_transform (p : ℝ × ℝ) (n : ℕ) : ℝ × ℝ :=
  match n with
  | 0 => p
  | n + 1 => transform (iterate_transform p n)

/-- The starting point P1 -/
def P1 : ℝ × ℝ := (2, 0)

theorem p2023_coordinates :
  iterate_transform P1 2023 = (-3, 3) := by
  sorry

end NUMINAMATH_CALUDE_p2023_coordinates_l2545_254536


namespace NUMINAMATH_CALUDE_total_pears_l2545_254535

/-- Given 4 boxes of pears with 16 pears in each box, the total number of pears is 64. -/
theorem total_pears (num_boxes : ℕ) (pears_per_box : ℕ) 
  (h1 : num_boxes = 4) 
  (h2 : pears_per_box = 16) : 
  num_boxes * pears_per_box = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_l2545_254535


namespace NUMINAMATH_CALUDE_min_value_when_k_is_one_l2545_254596

/-- The function for which we want to find the minimum value -/
def f (x k : ℝ) : ℝ := x^2 - (2*k + 3)*x + 2*k^2 - k - 3

/-- The theorem stating the minimum value of the function when k = 1 -/
theorem min_value_when_k_is_one :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x 1 ≥ f x_min 1 ∧ f x_min 1 = -33/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_when_k_is_one_l2545_254596


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l2545_254565

theorem sqrt_sum_fractions : 
  Real.sqrt ((9 : ℝ) / 16 + 25 / 9) = Real.sqrt 481 / 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l2545_254565


namespace NUMINAMATH_CALUDE_equation_solution_l2545_254574

theorem equation_solution :
  ∃ x : ℚ, (x + 2 ≠ 0 ∧ 3 - x ≠ 0) ∧
  ((3 * x - 5) / (x + 2) + (3 * x - 9) / (3 - x) = 2) ∧
  x = -15 / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2545_254574


namespace NUMINAMATH_CALUDE_sum_of_digits_is_23_l2545_254584

/-- A structure representing a four-digit number with unique digits -/
structure FourDigitNumber where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  d4 : Nat
  d1_pos : d1 > 0
  d2_pos : d2 > 0
  d3_pos : d3 > 0
  d4_pos : d4 > 0
  d1_lt_10 : d1 < 10
  d2_lt_10 : d2 < 10
  d3_lt_10 : d3 < 10
  d4_lt_10 : d4 < 10
  unique : d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4

/-- Theorem stating that for a four-digit number with product of digits 810 and unique digits, the sum of digits is 23 -/
theorem sum_of_digits_is_23 (n : FourDigitNumber) (h : n.d1 * n.d2 * n.d3 * n.d4 = 810) :
  n.d1 + n.d2 + n.d3 + n.d4 = 23 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_is_23_l2545_254584


namespace NUMINAMATH_CALUDE_gcd_5280_12155_l2545_254594

theorem gcd_5280_12155 : Nat.gcd 5280 12155 = 55 := by
  sorry

end NUMINAMATH_CALUDE_gcd_5280_12155_l2545_254594


namespace NUMINAMATH_CALUDE_percentage_same_grade_is_42_5_l2545_254558

/-- The total number of students in the class -/
def total_students : ℕ := 40

/-- The number of students who received an 'A' on both tests -/
def same_grade_A : ℕ := 3

/-- The number of students who received a 'B' on both tests -/
def same_grade_B : ℕ := 5

/-- The number of students who received a 'C' on both tests -/
def same_grade_C : ℕ := 6

/-- The number of students who received a 'D' on both tests -/
def same_grade_D : ℕ := 2

/-- The number of students who received an 'E' on both tests -/
def same_grade_E : ℕ := 1

/-- The total number of students who received the same grade on both tests -/
def total_same_grade : ℕ := same_grade_A + same_grade_B + same_grade_C + same_grade_D + same_grade_E

/-- The percentage of students who received the same grade on both tests -/
def percentage_same_grade : ℚ := (total_same_grade : ℚ) / (total_students : ℚ) * 100

theorem percentage_same_grade_is_42_5 : percentage_same_grade = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_same_grade_is_42_5_l2545_254558


namespace NUMINAMATH_CALUDE_power_of_two_equals_quadratic_plus_one_l2545_254587

theorem power_of_two_equals_quadratic_plus_one (x y : ℕ) :
  2^x = y^2 + y + 1 → x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equals_quadratic_plus_one_l2545_254587


namespace NUMINAMATH_CALUDE_log_simplification_l2545_254578

theorem log_simplification (p q r s z u : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (hz : z > 0) (hu : u > 0) : 
  Real.log (p / q) + Real.log (q / r) + Real.log (r / s) - Real.log (p * z / (s * u)) = Real.log (u / z) := by
  sorry

end NUMINAMATH_CALUDE_log_simplification_l2545_254578


namespace NUMINAMATH_CALUDE_basketball_game_scores_l2545_254528

/-- Represents the quarterly scores of a team -/
structure QuarterlyScores :=
  (q1 q2 q3 q4 : ℕ)

/-- Checks if the scores form an arithmetic sequence -/
def is_arithmetic_sequence (s : QuarterlyScores) : Prop :=
  ∃ d : ℕ, d > 0 ∧ 
    s.q2 = s.q1 + d ∧
    s.q3 = s.q2 + d ∧
    s.q4 = s.q3 + d

/-- Checks if the scores form a geometric sequence -/
def is_geometric_sequence (s : QuarterlyScores) : Prop :=
  ∃ r : ℚ, r > 1 ∧
    s.q2 = s.q1 * r ∧
    s.q3 = s.q2 * r ∧
    s.q4 = s.q3 * r

/-- The main theorem -/
theorem basketball_game_scores 
  (tigers lions : QuarterlyScores)
  (h1 : tigers.q1 = lions.q1)  -- Tied at the end of first quarter
  (h2 : is_arithmetic_sequence tigers)
  (h3 : is_geometric_sequence lions)
  (h4 : (tigers.q1 + tigers.q2 + tigers.q3 + tigers.q4) + 2 = 
        (lions.q1 + lions.q2 + lions.q3 + lions.q4))  -- Lions won by 2 points
  (h5 : tigers.q1 + tigers.q2 + tigers.q3 + tigers.q4 ≤ 100)
  (h6 : lions.q1 + lions.q2 + lions.q3 + lions.q4 ≤ 100)
  : tigers.q1 + tigers.q2 + lions.q1 + lions.q2 = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_game_scores_l2545_254528


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l2545_254537

theorem smallest_dual_base_representation : ∃ (a b : ℕ), 
  a > 3 ∧ b > 3 ∧ 
  13 = 1 * a + 3 ∧
  13 = 3 * b + 1 ∧
  (∀ (x y : ℕ), x > 3 → y > 3 → 1 * x + 3 = 3 * y + 1 → 1 * x + 3 ≥ 13) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l2545_254537


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2545_254593

-- Define the triangle ABC
theorem triangle_abc_properties (A B C : ℝ) (AB BC : ℝ) :
  AB = Real.sqrt 3 →
  BC = 2 →
  -- Part I
  (Real.cos B = -1/2 → Real.sin C = Real.sqrt 3 / 2) ∧
  -- Part II
  (∃ (lower upper : ℝ), lower = 0 ∧ upper = 2 * Real.pi / 3 ∧
    ∀ (x : ℝ), lower < C ∧ C ≤ upper) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2545_254593


namespace NUMINAMATH_CALUDE_matt_cookies_left_l2545_254563

/-- Represents the cookie-making scenario -/
structure CookieScenario where
  flour_per_batch : ℕ        -- pounds of flour per batch
  cookies_per_batch : ℕ      -- number of cookies per batch
  flour_bags : ℕ             -- number of flour bags used
  flour_per_bag : ℕ          -- pounds of flour per bag
  cookies_eaten : ℕ          -- number of cookies eaten

/-- Calculates the number of cookies left after baking and eating -/
def cookies_left (scenario : CookieScenario) : ℕ :=
  let total_flour := scenario.flour_bags * scenario.flour_per_bag
  let total_batches := total_flour / scenario.flour_per_batch
  let total_cookies := total_batches * scenario.cookies_per_batch
  total_cookies - scenario.cookies_eaten

/-- Theorem stating the number of cookies left in Matt's scenario -/
theorem matt_cookies_left :
  let matt_scenario : CookieScenario := {
    flour_per_batch := 2,
    cookies_per_batch := 12,
    flour_bags := 4,
    flour_per_bag := 5,
    cookies_eaten := 15
  }
  cookies_left matt_scenario = 105 := by
  sorry


end NUMINAMATH_CALUDE_matt_cookies_left_l2545_254563


namespace NUMINAMATH_CALUDE_division_problem_l2545_254595

theorem division_problem : (72 : ℚ) / ((6 : ℚ) / 3) = 36 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2545_254595


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_l2545_254540

theorem least_integer_absolute_value (x : ℤ) : (∀ y : ℤ, y < x → |3 * y + 10| > 25) ∧ |3 * x + 10| ≤ 25 ↔ x = -11 := by
  sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_l2545_254540


namespace NUMINAMATH_CALUDE_clay_pot_earnings_l2545_254520

/-- Calculate the money earned from selling clay pots --/
theorem clay_pot_earnings (total_pots : ℕ) (cracked_fraction : ℚ) (price_per_pot : ℕ) : 
  total_pots = 80 →
  cracked_fraction = 2 / 5 →
  price_per_pot = 40 →
  (total_pots : ℚ) * (1 - cracked_fraction) * price_per_pot = 1920 := by
  sorry

end NUMINAMATH_CALUDE_clay_pot_earnings_l2545_254520


namespace NUMINAMATH_CALUDE_min_value_xy_over_x2_plus_2y2_l2545_254523

theorem min_value_xy_over_x2_plus_2y2 (x y : ℝ) 
  (hx : 0.4 ≤ x ∧ x ≤ 0.6) (hy : 0.3 ≤ y ∧ y ≤ 0.5) :
  (∃ m : ℝ, m = (x * y) / (x^2 + 2 * y^2) ∧ 
    (∀ x' y' : ℝ, 0.4 ≤ x' ∧ x' ≤ 0.6 → 0.3 ≤ y' ∧ y' ≤ 0.5 → 
      m ≤ (x' * y') / (x'^2 + 2 * y'^2)) ∧
    m = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_xy_over_x2_plus_2y2_l2545_254523


namespace NUMINAMATH_CALUDE_cos_power_negative_set_l2545_254512

open Set Real

theorem cos_power_negative_set (M : Set ℝ) : 
  M = {x : ℝ | ∀ n : ℕ, cos (2^n * x) < 0} ↔ 
  M = {x : ℝ | ∃ k : ℤ, x = 2*k*π + 2*π/3 ∨ x = 2*k*π - 2*π/3} :=
by sorry

end NUMINAMATH_CALUDE_cos_power_negative_set_l2545_254512


namespace NUMINAMATH_CALUDE_teal_color_survey_l2545_254545

theorem teal_color_survey (total : ℕ) (green : ℕ) (both : ℕ) (neither : ℕ) 
  (h_total : total = 120)
  (h_green : green = 70)
  (h_both : both = 35)
  (h_neither : neither = 20) :
  ∃ blue : ℕ, blue = 65 ∧ 
    blue + green - both + neither = total :=
by sorry

end NUMINAMATH_CALUDE_teal_color_survey_l2545_254545


namespace NUMINAMATH_CALUDE_four_valid_orders_l2545_254518

/-- Represents a runner in the relay team -/
inductive Runner : Type
| Jordan : Runner
| Friend1 : Runner  -- The fastest friend
| Friend2 : Runner
| Friend3 : Runner

/-- Represents a lap in the relay race -/
inductive Lap : Type
| First : Lap
| Second : Lap
| Third : Lap
| Fourth : Lap

/-- A valid running order for the relay team -/
def RunningOrder : Type := Lap → Runner

/-- Checks if a running order is valid according to the given conditions -/
def isValidOrder (order : RunningOrder) : Prop :=
  (order Lap.First = Runner.Friend1) ∧  -- Fastest friend starts
  ((order Lap.Third = Runner.Jordan) ∨ (order Lap.Fourth = Runner.Jordan)) ∧  -- Jordan runs 3rd or 4th
  (∀ l : Lap, ∃! r : Runner, order l = r)  -- Each lap has exactly one runner

/-- The main theorem: there are exactly 4 valid running orders -/
theorem four_valid_orders :
  ∃ (orders : Finset RunningOrder),
    (∀ o ∈ orders, isValidOrder o) ∧
    (∀ o : RunningOrder, isValidOrder o → o ∈ orders) ∧
    (Finset.card orders = 4) :=
sorry

end NUMINAMATH_CALUDE_four_valid_orders_l2545_254518


namespace NUMINAMATH_CALUDE_dana_marcus_difference_l2545_254533

/-- The number of pencils Jayden has -/
def jayden_pencils : ℕ := 20

/-- The number of pencils Dana has -/
def dana_pencils : ℕ := jayden_pencils + 15

/-- The number of pencils Marcus has -/
def marcus_pencils : ℕ := jayden_pencils / 2

/-- Theorem stating that Dana has 25 more pencils than Marcus -/
theorem dana_marcus_difference : dana_pencils - marcus_pencils = 25 := by
  sorry

end NUMINAMATH_CALUDE_dana_marcus_difference_l2545_254533


namespace NUMINAMATH_CALUDE_frog_climb_time_l2545_254546

/-- Represents the frog's climbing problem -/
structure FrogClimb where
  well_depth : ℝ
  climb_distance : ℝ
  slip_distance : ℝ
  slip_time_ratio : ℝ
  time_to_near_top : ℝ

/-- Calculates the total time for the frog to climb to the top of the well -/
def total_climb_time (f : FrogClimb) : ℝ :=
  sorry

/-- Theorem stating that the total climb time is 20 minutes -/
theorem frog_climb_time (f : FrogClimb) 
  (h1 : f.well_depth = 12)
  (h2 : f.climb_distance = 3)
  (h3 : f.slip_distance = 1)
  (h4 : f.slip_time_ratio = 1/3)
  (h5 : f.time_to_near_top = 17) :
  total_climb_time f = 20 := by
  sorry

end NUMINAMATH_CALUDE_frog_climb_time_l2545_254546


namespace NUMINAMATH_CALUDE_cakes_sold_l2545_254564

theorem cakes_sold (initial_cakes : ℕ) (additional_cakes : ℕ) (remaining_cakes : ℕ) :
  initial_cakes = 62 →
  additional_cakes = 149 →
  remaining_cakes = 67 →
  initial_cakes + additional_cakes - remaining_cakes = 144 :=
by sorry

end NUMINAMATH_CALUDE_cakes_sold_l2545_254564


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2545_254586

theorem fraction_evaluation : (4 - 3/5) / (3 - 2/7) = 119/95 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2545_254586


namespace NUMINAMATH_CALUDE_division_fraction_proof_l2545_254521

theorem division_fraction_proof : (5 : ℚ) / ((8 : ℚ) / 13) = 65 / 8 := by
  sorry

end NUMINAMATH_CALUDE_division_fraction_proof_l2545_254521


namespace NUMINAMATH_CALUDE_football_team_right_handed_players_l2545_254553

theorem football_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 70) 
  (h2 : throwers = 34) 
  (h3 : (total_players - throwers) % 3 = 0) -- Ensures non-throwers can be divided into thirds
  (h4 : throwers ≤ total_players) : -- Ensures there are not more throwers than total players
  throwers + ((total_players - throwers) - (total_players - throwers) / 3) = 58 := by
  sorry

end NUMINAMATH_CALUDE_football_team_right_handed_players_l2545_254553


namespace NUMINAMATH_CALUDE_questions_per_day_l2545_254597

/-- Given a mathematician who needs to write a certain number of questions for two projects in one week,
    this theorem proves the number of questions he should complete each day. -/
theorem questions_per_day
  (project1_questions : ℕ)
  (project2_questions : ℕ)
  (days_in_week : ℕ)
  (h1 : project1_questions = 518)
  (h2 : project2_questions = 476)
  (h3 : days_in_week = 7) :
  (project1_questions + project2_questions) / days_in_week = 142 := by
  sorry

end NUMINAMATH_CALUDE_questions_per_day_l2545_254597


namespace NUMINAMATH_CALUDE_babysitting_cost_difference_l2545_254508

/-- Represents the babysitting scenario with given rates and conditions -/
structure BabysittingScenario where
  current_rate : ℕ -- Rate of current babysitter in dollars per hour
  new_base_rate : ℕ -- Base rate of new babysitter in dollars per hour
  new_scream_charge : ℕ -- Extra charge for each scream by new babysitter
  hours : ℕ -- Number of hours of babysitting
  screams : ℕ -- Number of times kids scream during babysitting

/-- Calculates the cost difference between current and new babysitter -/
def costDifference (scenario : BabysittingScenario) : ℕ :=
  scenario.current_rate * scenario.hours - 
  (scenario.new_base_rate * scenario.hours + scenario.new_scream_charge * scenario.screams)

/-- Theorem stating the cost difference for the given scenario -/
theorem babysitting_cost_difference :
  ∃ (scenario : BabysittingScenario),
    scenario.current_rate = 16 ∧
    scenario.new_base_rate = 12 ∧
    scenario.new_scream_charge = 3 ∧
    scenario.hours = 6 ∧
    scenario.screams = 2 ∧
    costDifference scenario = 18 := by
  sorry

end NUMINAMATH_CALUDE_babysitting_cost_difference_l2545_254508


namespace NUMINAMATH_CALUDE_total_shaded_area_l2545_254541

/-- Calculates the total shaded area of a floor tiled with patterned square tiles. -/
theorem total_shaded_area (floor_length floor_width tile_size circle_radius : ℝ) : 
  floor_length = 8 ∧ 
  floor_width = 10 ∧ 
  tile_size = 2 ∧ 
  circle_radius = 1 →
  (floor_length * floor_width / (tile_size * tile_size)) * (tile_size * tile_size - π * circle_radius ^ 2) = 80 - 20 * π :=
by sorry

end NUMINAMATH_CALUDE_total_shaded_area_l2545_254541


namespace NUMINAMATH_CALUDE_original_price_calculation_l2545_254588

theorem original_price_calculation (reduced_price : ℝ) (reduction_percent : ℝ) 
  (h1 : reduced_price = 620) 
  (h2 : reduction_percent = 20) : 
  reduced_price / (1 - reduction_percent / 100) = 775 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l2545_254588


namespace NUMINAMATH_CALUDE_cosine_ratio_equals_one_l2545_254562

theorem cosine_ratio_equals_one (c : ℝ) (h : c = 2 * Real.pi / 7) :
  (Real.cos (3 * c) * Real.cos (5 * c) * Real.cos (6 * c)) /
  (Real.cos c * Real.cos (2 * c) * Real.cos (3 * c)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_ratio_equals_one_l2545_254562


namespace NUMINAMATH_CALUDE_goat_is_guilty_l2545_254503

/-- Represents the three defendants in the court case -/
inductive Defendant
  | Goat
  | Beetle
  | Mosquito

/-- Represents the possible outcomes of the court case -/
inductive Verdict
  | Guilty
  | NotGuilty
  | Inconclusive

/-- Represents an accusation made by one defendant against another -/
structure Accusation where
  accuser : Defendant
  accused : Defendant

/-- Represents the testimonies given in the court case -/
structure Testimonies where
  goatAccusation : Accusation
  beetleAccusation : Accusation
  mosquitoAccusation : Accusation

/-- Determines if a given testimony is truthful based on the actual guilty party -/
def isTestimonyTruthful (testimony : Accusation) (guiltyParty : Defendant) : Prop :=
  testimony.accused = guiltyParty

/-- The main theorem stating that given the conditions, the Goat must be guilty -/
theorem goat_is_guilty (testimonies : Testimonies) 
  (h1 : ¬isTestimonyTruthful testimonies.goatAccusation Defendant.Goat)
  (h2 : isTestimonyTruthful testimonies.beetleAccusation Defendant.Goat)
  (h3 : isTestimonyTruthful testimonies.mosquitoAccusation Defendant.Goat)
  (h4 : testimonies.goatAccusation.accused ≠ testimonies.goatAccusation.accuser)
  (h5 : testimonies.beetleAccusation.accused ≠ testimonies.beetleAccusation.accuser)
  (h6 : testimonies.mosquitoAccusation.accused ≠ testimonies.mosquitoAccusation.accuser) :
  Verdict.Guilty = Verdict.Guilty := by
  sorry


end NUMINAMATH_CALUDE_goat_is_guilty_l2545_254503


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l2545_254531

theorem simplify_and_evaluate_expression :
  let x : ℝ := Real.sqrt 3 + 1
  (x + 1) / x / (x - 1 / x) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l2545_254531


namespace NUMINAMATH_CALUDE_thompson_children_ages_l2545_254548

/-- Represents the ages of Miss Thompson's children -/
def ChildrenAges : Type := Fin 5 → Nat

/-- Represents a 3-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_range : hundreds ≥ 1 ∧ hundreds ≤ 9
  t_range : tens ≥ 0 ∧ tens ≤ 9
  o_range : ones ≥ 0 ∧ ones ≤ 9
  different : hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones

theorem thompson_children_ages
  (ages : ChildrenAges)
  (number : ThreeDigitNumber)
  (h_oldest : ages 0 = 11)
  (h_middle : ages 2 = 7)
  (h_different : ∀ i j, i ≠ j → ages i ≠ ages j)
  (h_divisible_oldest : (number.hundreds * 100 + number.tens * 10 + number.ones) % 11 = 0)
  (h_divisible_middle : (number.hundreds * 100 + number.tens * 10 + number.ones) % 7 = 0)
  (h_youngest : ∃ i, ages i = number.ones)
  : ¬(∃ i, ages i = 6) :=
by sorry

end NUMINAMATH_CALUDE_thompson_children_ages_l2545_254548


namespace NUMINAMATH_CALUDE_hotel_pricing_l2545_254510

/-- The hotel pricing problem -/
theorem hotel_pricing
  (night_rate : ℝ)
  (night_hours : ℝ)
  (morning_hours : ℝ)
  (initial_money : ℝ)
  (remaining_money : ℝ)
  (h1 : night_rate = 1.5)
  (h2 : night_hours = 6)
  (h3 : morning_hours = 4)
  (h4 : initial_money = 80)
  (h5 : remaining_money = 63)
  : ∃ (morning_rate : ℝ), 
    night_rate * night_hours + morning_rate * morning_hours = initial_money - remaining_money ∧
    morning_rate = 2 := by
  sorry

end NUMINAMATH_CALUDE_hotel_pricing_l2545_254510


namespace NUMINAMATH_CALUDE_enrollment_difference_l2545_254539

/-- Represents the enrollment of a school --/
structure School where
  name : String
  enrollment : Nat

/-- Theorem: The positive difference between the maximum and minimum enrollments is 700 --/
theorem enrollment_difference (schools : List School) 
    (h1 : schools = [
      ⟨"Varsity", 1150⟩, 
      ⟨"Northwest", 1530⟩, 
      ⟨"Central", 1850⟩, 
      ⟨"Greenbriar", 1680⟩, 
      ⟨"Riverside", 1320⟩
    ]) : 
    (List.maximum (schools.map School.enrollment)).getD 0 - 
    (List.minimum (schools.map School.enrollment)).getD 0 = 700 := by
  sorry


end NUMINAMATH_CALUDE_enrollment_difference_l2545_254539


namespace NUMINAMATH_CALUDE_binomial_sum_l2545_254547

theorem binomial_sum : 
  let p := Nat.choose 20 6
  let q := Nat.choose 20 5
  p + q = 62016 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_l2545_254547


namespace NUMINAMATH_CALUDE_intersection_M_N_l2545_254577

def M : Set ℝ := {x | x / (x - 1) ≥ 0}

def N : Set ℝ := {y | ∃ x, y = 3 * x^2 + 1}

theorem intersection_M_N : M ∩ N = {x | x > 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2545_254577


namespace NUMINAMATH_CALUDE_probability_odd_and_multiple_of_3_l2545_254571

/-- Represents a fair die with n sides -/
structure Die (n : ℕ) where
  sides : Finset (Fin n)
  fair : sides.card = n

/-- The event of rolling an odd number on a die -/
def oddEvent (d : Die n) : Finset (Fin n) :=
  d.sides.filter (λ x => x.val % 2 = 1)

/-- The event of rolling a multiple of 3 on a die -/
def multipleOf3Event (d : Die n) : Finset (Fin n) :=
  d.sides.filter (λ x => x.val % 3 = 0)

/-- The probability of an event occurring on a fair die -/
def probability (d : Die n) (event : Finset (Fin n)) : ℚ :=
  event.card / d.sides.card

theorem probability_odd_and_multiple_of_3 
  (d8 : Die 8) 
  (d12 : Die 12) : 
  probability d8 (oddEvent d8) * probability d12 (multipleOf3Event d12) = 1/6 := by
sorry

end NUMINAMATH_CALUDE_probability_odd_and_multiple_of_3_l2545_254571


namespace NUMINAMATH_CALUDE_casper_candies_proof_l2545_254519

/-- The number of candies Casper initially had -/
def initial_candies : ℕ := 622

/-- The number of candies Casper gave to his brother on day 1 -/
def brother_candies : ℕ := 3

/-- The number of candies Casper gave to his sister on day 2 -/
def sister_candies : ℕ := 5

/-- The number of candies Casper gave to his friend on day 3 -/
def friend_candies : ℕ := 2

/-- The number of candies Casper had left on day 4 -/
def final_candies : ℕ := 10

theorem casper_candies_proof :
  (1 / 48 : ℚ) * initial_candies - 71 / 24 = final_candies := by
  sorry

end NUMINAMATH_CALUDE_casper_candies_proof_l2545_254519


namespace NUMINAMATH_CALUDE_converse_and_inverse_false_l2545_254530

-- Define the types
variable (Quadrilateral : Type)
variable (isRhombus : Quadrilateral → Prop)
variable (isParallelogram : Quadrilateral → Prop)

-- Define the original statement
axiom original_statement : ∀ q : Quadrilateral, isRhombus q → isParallelogram q

-- State the theorem to be proved
theorem converse_and_inverse_false :
  (∀ q : Quadrilateral, isParallelogram q → isRhombus q) = False ∧
  (∀ q : Quadrilateral, ¬isRhombus q → ¬isParallelogram q) = False :=
by sorry

end NUMINAMATH_CALUDE_converse_and_inverse_false_l2545_254530


namespace NUMINAMATH_CALUDE_math_score_calculation_l2545_254580

theorem math_score_calculation (total_subjects : ℕ) (avg_without_math : ℝ) (avg_with_math : ℝ) :
  total_subjects = 5 →
  avg_without_math = 88 →
  avg_with_math = 92 →
  (total_subjects - 1) * avg_without_math + (avg_with_math * total_subjects - (total_subjects - 1) * avg_without_math) = 108 :=
by sorry

end NUMINAMATH_CALUDE_math_score_calculation_l2545_254580


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l2545_254550

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distributeBalls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 41 ways to distribute 5 distinguishable balls into 3 indistinguishable boxes -/
theorem five_balls_three_boxes : distributeBalls 5 3 = 41 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l2545_254550


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l2545_254570

-- Problem 1
theorem equation_one_solution (x : ℝ) :
  (2 / x + 1 / (x * (x - 2)) = 5 / (2 * x)) ↔ x = 4 :=
sorry

-- Problem 2
theorem equation_two_no_solution :
  ¬∃ (x : ℝ), (5 * x - 4) / (x - 2) = (4 * x + 10) / (3 * x - 6) - 1 :=
sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l2545_254570


namespace NUMINAMATH_CALUDE_constant_term_product_l2545_254542

-- Define polynomials p, q, r, and s
variable (p q r s : ℝ[X])

-- Define the relationship between s, p, q, and r
axiom h1 : s = p * q * r

-- Define the constant term of p as 2
axiom h2 : p.coeff 0 = 2

-- Define the constant term of s as 6
axiom h3 : s.coeff 0 = 6

-- Theorem to prove
theorem constant_term_product : q.coeff 0 * r.coeff 0 = 3 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_product_l2545_254542


namespace NUMINAMATH_CALUDE_division_result_l2545_254501

theorem division_result : (4 : ℚ) / (8 / 13) = 13 / 2 := by sorry

end NUMINAMATH_CALUDE_division_result_l2545_254501


namespace NUMINAMATH_CALUDE_min_side_difference_l2545_254579

theorem min_side_difference (PQ PR QR : ℕ) : 
  PQ + PR + QR = 3010 →
  PQ < PR →
  PR ≤ QR →
  PQ + PR > QR →
  PQ + QR > PR →
  PR + QR > PQ →
  ∀ PQ' PR' QR' : ℕ, 
    PQ' + PR' + QR' = 3010 →
    PQ' < PR' →
    PR' ≤ QR' →
    PQ' + PR' > QR' →
    PQ' + QR' > PR' →
    PR' + QR' > PQ' →
    QR - PQ ≤ QR' - PQ' :=
by sorry

end NUMINAMATH_CALUDE_min_side_difference_l2545_254579


namespace NUMINAMATH_CALUDE_gcd_6a_8b_lower_bound_l2545_254514

theorem gcd_6a_8b_lower_bound (a b : ℕ+) (h : Nat.gcd a b = 10) :
  (∃ (a' b' : ℕ+), Nat.gcd a' b' = 10 ∧ Nat.gcd (6 * a') (8 * b') = 20) ∧
  (Nat.gcd (6 * a) (8 * b) ≥ 20) :=
sorry

end NUMINAMATH_CALUDE_gcd_6a_8b_lower_bound_l2545_254514


namespace NUMINAMATH_CALUDE_both_reunions_count_l2545_254591

/-- The number of people attending both the Oates and Yellow reunions -/
def both_reunions (total_guests oates_guests yellow_guests : ℕ) : ℕ :=
  oates_guests + yellow_guests - total_guests

theorem both_reunions_count :
  both_reunions 100 42 65 = 7 := by
  sorry

end NUMINAMATH_CALUDE_both_reunions_count_l2545_254591


namespace NUMINAMATH_CALUDE_power_fraction_plus_two_l2545_254581

theorem power_fraction_plus_two : (5 / 3 : ℚ)^7 + 2 = 82499 / 2187 := by sorry

end NUMINAMATH_CALUDE_power_fraction_plus_two_l2545_254581


namespace NUMINAMATH_CALUDE_probability_same_color_l2545_254534

def green_balls : ℕ := 8
def white_balls : ℕ := 6
def red_balls : ℕ := 5
def blue_balls : ℕ := 4

def total_balls : ℕ := green_balls + white_balls + red_balls + blue_balls

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

def total_combinations : ℕ := choose total_balls 3

def same_color_combinations : ℕ := 
  choose green_balls 3 + choose white_balls 3 + choose red_balls 3 + choose blue_balls 3

theorem probability_same_color :
  (same_color_combinations : ℚ) / total_combinations = 90 / 1771 := by sorry

end NUMINAMATH_CALUDE_probability_same_color_l2545_254534


namespace NUMINAMATH_CALUDE_seashell_collection_l2545_254511

theorem seashell_collection (joan_daily : ℕ) (jessica_daily : ℕ) (days : ℕ) : 
  joan_daily = 6 → jessica_daily = 8 → days = 7 → 
  (joan_daily + jessica_daily) * days = 98 := by
  sorry

end NUMINAMATH_CALUDE_seashell_collection_l2545_254511


namespace NUMINAMATH_CALUDE_total_games_l2545_254556

def games_this_year : ℕ := 36
def games_last_year : ℕ := 11

theorem total_games : games_this_year + games_last_year = 47 := by
  sorry

end NUMINAMATH_CALUDE_total_games_l2545_254556


namespace NUMINAMATH_CALUDE_real_roots_of_quartic_equation_l2545_254509

theorem real_roots_of_quartic_equation :
  let f : ℝ → ℝ := λ x => 2 * x^4 + 4 * x^3 + 3 * x^2 + x - 1
  let x₁ : ℝ := (-1 + Real.sqrt 3) / 2
  let x₂ : ℝ := (-1 - Real.sqrt 3) / 2
  (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) ∧ (f x₁ = 0 ∧ f x₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_real_roots_of_quartic_equation_l2545_254509


namespace NUMINAMATH_CALUDE_elaines_rent_percentage_l2545_254529

/-- Proves that given the conditions in the problem, Elaine spent 20% of her annual earnings on rent last year. -/
theorem elaines_rent_percentage (E : ℝ) (P : ℝ) : 
  E > 0 → -- Elaine's earnings last year (assumed positive)
  0.30 * (1.35 * E) = 2.025 * (P / 100 * E) → -- Condition relating this year's and last year's rent
  P = 20 := by sorry

end NUMINAMATH_CALUDE_elaines_rent_percentage_l2545_254529


namespace NUMINAMATH_CALUDE_money_sharing_l2545_254555

theorem money_sharing (amanda ben carlos diana total : ℕ) : 
  amanda = 45 →
  amanda + ben + carlos + diana = total →
  3 * ben = 5 * amanda →
  3 * carlos = 6 * amanda →
  3 * diana = 8 * amanda →
  total = 330 := by
sorry

end NUMINAMATH_CALUDE_money_sharing_l2545_254555


namespace NUMINAMATH_CALUDE_correct_fill_times_l2545_254557

/-- Represents the time taken to fill a vessel using three taps -/
structure VesselFillTime where
  tap1 : ℝ  -- Time for first tap
  tap2 : ℝ  -- Time for second tap
  tap3 : ℝ  -- Time for third tap

/-- Checks if the given fill times satisfy the problem conditions -/
def satisfies_conditions (t : VesselFillTime) : Prop :=
  1 / t.tap1 + 1 / t.tap2 + 1 / t.tap3 = 1 / 6 ∧
  t.tap2 = 0.75 * t.tap1 ∧
  t.tap3 = t.tap2 + 10

/-- The theorem stating the correct fill times for each tap -/
theorem correct_fill_times :
  ∃ (t : VesselFillTime),
    satisfies_conditions t ∧
    t.tap1 = 56 / 3 ∧
    t.tap2 = 14 ∧
    t.tap3 = 24 :=
by sorry

end NUMINAMATH_CALUDE_correct_fill_times_l2545_254557


namespace NUMINAMATH_CALUDE_filter_price_theorem_l2545_254516

-- Define the number of filters and their prices
def total_filters : ℕ := 5
def kit_price : ℚ := 87.50
def price_filter_1 : ℚ := 16.45
def price_filter_2 : ℚ := 19.50
def num_filter_1 : ℕ := 2
def num_filter_2 : ℕ := 1
def num_unknown_price : ℕ := 2
def savings_percentage : ℚ := 0.08

-- Define the function to calculate the total individual price
def total_individual_price (x : ℚ) : ℚ :=
  num_filter_1 * price_filter_1 + num_unknown_price * x + num_filter_2 * price_filter_2

-- Define the theorem
theorem filter_price_theorem (x : ℚ) :
  (savings_percentage * total_individual_price x = total_individual_price x - kit_price) →
  x = 21.36 := by
  sorry

end NUMINAMATH_CALUDE_filter_price_theorem_l2545_254516


namespace NUMINAMATH_CALUDE_divisibility_problem_l2545_254590

theorem divisibility_problem (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 45)
  (h2 : Nat.gcd q r = 75)
  (h3 : Nat.gcd r s = 90)
  (h4 : 150 < Nat.gcd s p ∧ Nat.gcd s p < 200) :
  10 ∣ p.val := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2545_254590


namespace NUMINAMATH_CALUDE_pure_imaginary_square_root_l2545_254599

theorem pure_imaginary_square_root (a : ℝ) : 
  (∃ (b : ℝ), (1 + a * Complex.I)^2 = b * Complex.I) → (a = 1 ∨ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_square_root_l2545_254599


namespace NUMINAMATH_CALUDE_intersection_sum_l2545_254583

theorem intersection_sum (c d : ℝ) : 
  (2 * 4 + c = 6) →  -- First line passes through (4, 6)
  (5 * 4 + d = 6) →  -- Second line passes through (4, 6)
  c + d = -16 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l2545_254583


namespace NUMINAMATH_CALUDE_right_triangle_area_and_perimeter_l2545_254585

theorem right_triangle_area_and_perimeter :
  ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  c = 13 →
  a = 5 →
  b > a →
  (1/2 * a * b = 30 ∧ a + b + c = 30) :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_and_perimeter_l2545_254585


namespace NUMINAMATH_CALUDE_factorization_equality_l2545_254566

theorem factorization_equality (x y : ℝ) : 2 * x^2 - 8 * y^2 = 2 * (x + 2*y) * (x - 2*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2545_254566


namespace NUMINAMATH_CALUDE_subset_implies_x_value_l2545_254549

theorem subset_implies_x_value (A B : Set ℝ) (x : ℝ) : 
  A = {-2, 1} → 
  B = {0, 1, x + 1} → 
  A ⊆ B → 
  x = -3 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_x_value_l2545_254549


namespace NUMINAMATH_CALUDE_platform_length_l2545_254543

/-- The length of a platform given a train's speed, crossing time, and length -/
theorem platform_length 
  (train_speed : Real) 
  (crossing_time : Real) 
  (train_length : Real) : 
  train_speed = 72 * (5/18) → 
  crossing_time = 36 → 
  train_length = 470.06 → 
  (train_speed * crossing_time) - train_length = 249.94 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l2545_254543


namespace NUMINAMATH_CALUDE_allowance_multiple_l2545_254575

theorem allowance_multiple (middle_school_allowance senior_year_allowance x : ℝ) :
  middle_school_allowance = 8 + 2 →
  senior_year_allowance = middle_school_allowance * x + 5 →
  (senior_year_allowance - middle_school_allowance) / middle_school_allowance = 1.5 →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_allowance_multiple_l2545_254575


namespace NUMINAMATH_CALUDE_six_digit_multiple_of_nine_l2545_254552

theorem six_digit_multiple_of_nine :
  ∃ (d : ℕ), d < 10 ∧ (567890 + d) % 9 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_six_digit_multiple_of_nine_l2545_254552


namespace NUMINAMATH_CALUDE_road_project_completion_time_l2545_254502

/-- Represents a road construction project -/
structure RoadProject where
  totalLength : ℝ
  initialWorkers : ℕ
  daysWorked : ℝ
  completedLength : ℝ
  extraWorkers : ℕ

/-- Calculates the total number of days required to complete the road project -/
def totalDaysRequired (project : RoadProject) : ℝ :=
  sorry

/-- Theorem stating that given the project conditions, it will be completed in 15 days -/
theorem road_project_completion_time (project : RoadProject)
  (h1 : project.totalLength = 10)
  (h2 : project.initialWorkers = 30)
  (h3 : project.daysWorked = 5)
  (h4 : project.completedLength = 2)
  (h5 : project.extraWorkers = 30) :
  totalDaysRequired project = 15 :=
sorry

end NUMINAMATH_CALUDE_road_project_completion_time_l2545_254502


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2545_254589

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  (1 / x + 3 / y) ≥ 1 + Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2545_254589


namespace NUMINAMATH_CALUDE_intersection_of_lines_l2545_254527

theorem intersection_of_lines :
  ∃! (x y : ℚ), (8 * x - 5 * y = 10) ∧ (3 * x + 2 * y = 16) ∧ 
  (x = 100 / 31) ∧ (y = 98 / 31) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l2545_254527


namespace NUMINAMATH_CALUDE_gcf_of_32_and_12_l2545_254572

theorem gcf_of_32_and_12 (n : ℕ) (h1 : n = 32) (h2 : Nat.lcm n 12 = 48) :
  Nat.gcd n 12 = 8 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_32_and_12_l2545_254572


namespace NUMINAMATH_CALUDE_hexagon_walk_distance_l2545_254554

def regular_hexagon_side_length : ℝ := 3
def walk_distance : ℝ := 10

theorem hexagon_walk_distance (start_point end_point : ℝ × ℝ) : 
  start_point = (0, 0) →
  end_point = (0.5, -Real.sqrt 3 / 2) →
  Real.sqrt ((end_point.1 - start_point.1)^2 + (end_point.2 - start_point.2)^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_walk_distance_l2545_254554


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l2545_254524

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_y_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (4, -1 + y)
  are_parallel a b → y = 7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l2545_254524


namespace NUMINAMATH_CALUDE_vector_parallel_implies_x_eq_half_l2545_254506

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![1, 2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 1]

-- Define the parallel condition
def are_parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v 0 * w 1 = k * v 1 * w 0

-- State the theorem
theorem vector_parallel_implies_x_eq_half :
  ∀ x : ℝ, are_parallel (a + 2 • b x) (2 • a - 2 • b x) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_implies_x_eq_half_l2545_254506


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2545_254569

theorem complex_equation_solution (z : ℂ) 
  (h : 12 * Complex.abs z ^ 2 = 2 * Complex.abs (z + 2) ^ 2 + Complex.abs (z ^ 2 + 1) ^ 2 + 31) :
  z + 6 / z = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2545_254569


namespace NUMINAMATH_CALUDE_ternary_121_equals_16_l2545_254598

def ternary_to_decimal (d₂ d₁ d₀ : ℕ) : ℕ :=
  d₂ * 3^2 + d₁ * 3^1 + d₀ * 3^0

theorem ternary_121_equals_16 : ternary_to_decimal 1 2 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ternary_121_equals_16_l2545_254598


namespace NUMINAMATH_CALUDE_insect_count_l2545_254513

theorem insect_count (total_legs : ℕ) (legs_per_insect : ℕ) (h1 : total_legs = 36) (h2 : legs_per_insect = 6) :
  total_legs / legs_per_insect = 6 := by
  sorry

end NUMINAMATH_CALUDE_insect_count_l2545_254513


namespace NUMINAMATH_CALUDE_parabola_equation_l2545_254582

/-- A parabola is defined by its directrix. -/
structure Parabola where
  directrix : ℝ

/-- The standard equation of a parabola. -/
def standard_equation (p : Parabola) : Prop :=
  ∀ x y : ℝ, y^2 = 28 * x

/-- Theorem: If the directrix of a parabola is x = -7, then its standard equation is y² = 28x. -/
theorem parabola_equation (p : Parabola) (h : p.directrix = -7) : standard_equation p := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l2545_254582


namespace NUMINAMATH_CALUDE_sandy_fish_problem_l2545_254504

/-- The number of pet fish Sandy has after buying more -/
def sandys_final_fish_count (initial : ℕ) (bought : ℕ) : ℕ :=
  initial + bought

/-- Theorem: Sandy's final fish count is 32 given the initial conditions -/
theorem sandy_fish_problem :
  sandys_final_fish_count 26 6 = 32 := by
  sorry

end NUMINAMATH_CALUDE_sandy_fish_problem_l2545_254504


namespace NUMINAMATH_CALUDE_prob_divisible_by_eight_l2545_254559

/-- The number of dice rolled -/
def num_dice : ℕ := 8

/-- The number of sides on each die -/
def die_sides : ℕ := 6

/-- The probability of rolling an odd number on a single die -/
def prob_odd : ℚ := 1 / 2

/-- The probability of rolling a 2 on a single die -/
def prob_two : ℚ := 1 / 6

/-- The probability of rolling a 4 on a single die -/
def prob_four : ℚ := 1 / 6

/-- The probability that the product of the rolls is divisible by 8 -/
theorem prob_divisible_by_eight : 
  (1 : ℚ) - (prob_odd ^ num_dice + 
    (num_dice.choose 1 : ℚ) * prob_two * prob_odd ^ (num_dice - 1) +
    (num_dice.choose 2 : ℚ) * prob_two ^ 2 * prob_odd ^ (num_dice - 2) +
    (num_dice.choose 1 : ℚ) * prob_four * prob_odd ^ (num_dice - 1)) = 65 / 72 := by
  sorry


end NUMINAMATH_CALUDE_prob_divisible_by_eight_l2545_254559


namespace NUMINAMATH_CALUDE_smallest_sum_is_four_ninths_l2545_254551

theorem smallest_sum_is_four_ninths :
  let sums : List ℚ := [1/3 + 1/4, 1/3 + 1/5, 1/3 + 1/6, 1/3 + 1/7, 1/3 + 1/9]
  (∀ s ∈ sums, 1/3 + 1/9 ≤ s) ∧ (1/3 + 1/9 = 4/9) := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_is_four_ninths_l2545_254551


namespace NUMINAMATH_CALUDE_inscribed_squares_max_distance_l2545_254517

def inner_square_perimeter : ℝ := 20
def outer_square_perimeter : ℝ := 28

theorem inscribed_squares_max_distance :
  let inner_side := inner_square_perimeter / 4
  let outer_side := outer_square_perimeter / 4
  ∃ (x y : ℝ),
    x + y = outer_side ∧
    x^2 + y^2 = inner_side^2 ∧
    Real.sqrt (x^2 + (x + y)^2) = Real.sqrt 65 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_squares_max_distance_l2545_254517


namespace NUMINAMATH_CALUDE_energy_conservation_train_ball_system_energy_changes_specific_scenario_l2545_254561

/-- Represents the velocity of an object -/
structure Velocity where
  value : ℝ
  unit : String

/-- Represents the kinetic energy of an object -/
structure KineticEnergy where
  value : ℝ
  unit : String

/-- Represents a physical system consisting of a train and a ball -/
structure TrainBallSystem where
  trainVelocity : Velocity
  ballMass : ℝ
  ballThrowingVelocity : Velocity

/-- Calculates the kinetic energy of an object given its mass and velocity -/
def calculateKineticEnergy (mass : ℝ) (velocity : Velocity) : KineticEnergy :=
  { value := 0.5 * mass * velocity.value ^ 2, unit := "J" }

/-- Theorem: Energy conservation in the train-ball system -/
theorem energy_conservation_train_ball_system
  (system : TrainBallSystem)
  (initial_train_energy : KineticEnergy)
  (initial_ball_energy : KineticEnergy)
  (final_ball_energy_forward : KineticEnergy)
  (final_ball_energy_backward : KineticEnergy) :
  (initial_train_energy.value + initial_ball_energy.value =
   initial_train_energy.value + final_ball_energy_forward.value) ∧
  (initial_train_energy.value + initial_ball_energy.value =
   initial_train_energy.value + final_ball_energy_backward.value) :=
by sorry

/-- Corollary: Specific energy changes for the given scenario -/
theorem energy_changes_specific_scenario
  (system : TrainBallSystem)
  (h_train_velocity : system.trainVelocity.value = 60 ∧ system.trainVelocity.unit = "km/hour")
  (h_ball_velocity : system.ballThrowingVelocity.value = 60 ∧ system.ballThrowingVelocity.unit = "km/hour")
  (initial_ball_energy : KineticEnergy)
  (h_forward : calculateKineticEnergy system.ballMass
    { value := system.trainVelocity.value + system.ballThrowingVelocity.value, unit := "km/hour" } =
    { value := 4 * initial_ball_energy.value, unit := initial_ball_energy.unit })
  (h_backward : calculateKineticEnergy system.ballMass
    { value := system.trainVelocity.value - system.ballThrowingVelocity.value, unit := "km/hour" } =
    { value := 0, unit := initial_ball_energy.unit }) :
  ∃ (compensating_energy : KineticEnergy),
    compensating_energy.value = 3 * initial_ball_energy.value ∧
    compensating_energy.value = initial_ball_energy.value :=
by sorry

end NUMINAMATH_CALUDE_energy_conservation_train_ball_system_energy_changes_specific_scenario_l2545_254561


namespace NUMINAMATH_CALUDE_max_gingerbread_production_l2545_254560

/-- The gingerbread production function -/
def gingerbread_production (k : ℝ) (t : ℝ) : ℝ := k * t * (24 - t)

/-- Theorem stating that gingerbread production is maximized at 16 hours of work -/
theorem max_gingerbread_production (k : ℝ) (h : k > 0) :
  ∃ (t : ℝ), t = 16 ∧ ∀ (s : ℝ), 0 ≤ s ∧ s ≤ 24 → gingerbread_production k s ≤ gingerbread_production k t :=
by
  sorry

#check max_gingerbread_production

end NUMINAMATH_CALUDE_max_gingerbread_production_l2545_254560


namespace NUMINAMATH_CALUDE_chrome_parts_total_l2545_254573

/-- Represents the number of machines of type A -/
def a : ℕ := sorry

/-- Represents the number of machines of type B -/
def b : ℕ := sorry

/-- The total number of machines -/
def total_machines : ℕ := 21

/-- The total number of steel parts -/
def total_steel : ℕ := 50

/-- The number of steel parts in a type A machine -/
def steel_parts_A : ℕ := 3

/-- The number of steel parts in a type B machine -/
def steel_parts_B : ℕ := 2

/-- The number of chrome parts in a type A machine -/
def chrome_parts_A : ℕ := 2

/-- The number of chrome parts in a type B machine -/
def chrome_parts_B : ℕ := 4

theorem chrome_parts_total : 
  a + b = total_machines ∧ 
  steel_parts_A * a + steel_parts_B * b = total_steel →
  chrome_parts_A * a + chrome_parts_B * b = 68 := by
  sorry

end NUMINAMATH_CALUDE_chrome_parts_total_l2545_254573


namespace NUMINAMATH_CALUDE_sheridan_fish_problem_l2545_254538

/-- The number of fish Mrs. Sheridan's sister gave her -/
def fish_given (initial : ℕ) (final : ℕ) : ℕ := final - initial

theorem sheridan_fish_problem : fish_given 22 69 = 47 := by sorry

end NUMINAMATH_CALUDE_sheridan_fish_problem_l2545_254538


namespace NUMINAMATH_CALUDE_winning_candidate_votes_l2545_254525

/-- Proves that the winning candidate received 11628 votes in the described election scenario -/
theorem winning_candidate_votes :
  let total_votes : ℝ := (4136 + 7636) / (1 - 0.4969230769230769)
  let winning_votes : ℝ := 0.4969230769230769 * total_votes
  ⌊winning_votes⌋ = 11628 := by
  sorry

end NUMINAMATH_CALUDE_winning_candidate_votes_l2545_254525


namespace NUMINAMATH_CALUDE_password_probability_l2545_254507

def even_two_digit_numbers : ℕ := 45
def vowels : ℕ := 5
def total_letters : ℕ := 26
def prime_two_digit_numbers : ℕ := 21
def total_two_digit_numbers : ℕ := 90

theorem password_probability :
  (even_two_digit_numbers / total_two_digit_numbers) *
  (vowels / total_letters) *
  (prime_two_digit_numbers / total_two_digit_numbers) =
  7 / 312 := by sorry

end NUMINAMATH_CALUDE_password_probability_l2545_254507


namespace NUMINAMATH_CALUDE_calculation_result_l2545_254576

theorem calculation_result : 
  let initial := 180
  let percentage := 35 / 100
  let first_calc := initial * percentage
  let one_third_less := first_calc - (1 / 3 * first_calc)
  let remaining := initial - one_third_less
  let three_fifths := 3 / 5 * remaining
  (three_fifths ^ 2) = 6857.84 := by sorry

end NUMINAMATH_CALUDE_calculation_result_l2545_254576


namespace NUMINAMATH_CALUDE_lindas_trip_length_l2545_254544

theorem lindas_trip_length :
  ∀ (total_length : ℚ),
  (1 / 4 : ℚ) * total_length + 30 + (1 / 6 : ℚ) * total_length = total_length →
  total_length = 360 / 7 := by
  sorry

end NUMINAMATH_CALUDE_lindas_trip_length_l2545_254544


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2545_254568

theorem inequality_system_solution : 
  ∀ x : ℤ, (2 * (x - 1) < x + 1 ∧ 1 - (2 * x + 5) / 3 ≤ x ∧ x > 0) ↔ (x = 1 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2545_254568


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l2545_254505

theorem sum_of_fractions_equals_one
  (a b c x y z : ℝ)
  (eq1 : 15 * x + b * y + c * z = 0)
  (eq2 : a * x + 25 * y + c * z = 0)
  (eq3 : a * x + b * y + 45 * z = 0)
  (ha : a ≠ 15)
  (hb : b ≠ 25)
  (hx : x ≠ 0) :
  a / (a - 15) + b / (b - 25) + c / (c - 45) = 1 :=
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l2545_254505


namespace NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_q_l2545_254522

-- Define the conditions p and q
def p (a : ℝ) : Prop := a < 0
def q (a : ℝ) : Prop := a^2 > a

-- Theorem statement
theorem not_p_necessary_not_sufficient_for_not_q :
  (∀ a, ¬(q a) → ¬(p a)) ∧ 
  (∃ a, ¬(p a) ∧ q a) :=
by sorry

end NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_q_l2545_254522


namespace NUMINAMATH_CALUDE_min_value_theorem_l2545_254515

theorem min_value_theorem (p q r s t u v w : ℝ) 
  (h1 : p * q * r * s = 16) 
  (h2 : t * u * v * w = 25) :
  (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 80 ∧
  ∃ (p' q' r' s' t' u' v' w' : ℝ),
    p' * q' * r' * s' = 16 ∧
    t' * u' * v' * w' = 25 ∧
    (p' * t')^2 + (q' * u')^2 + (r' * v')^2 + (s' * w')^2 = 80 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2545_254515


namespace NUMINAMATH_CALUDE_dinner_bill_proof_l2545_254592

/-- The number of friends in the group -/
def num_friends : ℕ := 10

/-- The additional amount each paying friend contributes to cover the non-paying friend -/
def extra_payment : ℚ := 4

/-- The total bill for the group dinner -/
def total_bill : ℚ := 360

theorem dinner_bill_proof :
  ∃ (individual_share : ℚ),
    (num_friends - 1 : ℚ) * (individual_share + extra_payment) = total_bill :=
by sorry

end NUMINAMATH_CALUDE_dinner_bill_proof_l2545_254592


namespace NUMINAMATH_CALUDE_no_obtuse_angles_l2545_254500

-- Define an isosceles triangle with two 70-degree angles
structure IsoscelesTriangle70 where
  angle_a : ℝ
  angle_b : ℝ
  angle_c : ℝ
  is_isosceles : angle_a = angle_b
  angles_70 : angle_a = 70 ∧ angle_b = 70
  sum_180 : angle_a + angle_b + angle_c = 180

-- Define what an obtuse angle is
def is_obtuse (angle : ℝ) : Prop := angle > 90

-- Theorem statement
theorem no_obtuse_angles (t : IsoscelesTriangle70) :
  ¬ (is_obtuse t.angle_a ∨ is_obtuse t.angle_b ∨ is_obtuse t.angle_c) :=
by sorry

end NUMINAMATH_CALUDE_no_obtuse_angles_l2545_254500


namespace NUMINAMATH_CALUDE_unique_product_l2545_254567

theorem unique_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * b * c = 8 * (a + b + c))
  (h2 : c = a + b)
  (h3 : b = 2 * a) :
  a * b * c = 96 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_unique_product_l2545_254567
