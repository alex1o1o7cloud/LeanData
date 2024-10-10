import Mathlib

namespace power_sum_integer_l1989_198947

theorem power_sum_integer (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/x^n = m := by
  sorry

end power_sum_integer_l1989_198947


namespace min_sum_of_product_144_l1989_198966

theorem min_sum_of_product_144 (a b : ℤ) (h : a * b = 144) :
  ∀ (x y : ℤ), x * y = 144 → a + b ≤ x + y ∧ ∃ (a₀ b₀ : ℤ), a₀ * b₀ = 144 ∧ a₀ + b₀ = -145 :=
sorry

end min_sum_of_product_144_l1989_198966


namespace gcd_power_two_minus_one_l1989_198941

theorem gcd_power_two_minus_one : 
  Nat.gcd (2^2004 - 1) (2^1995 - 1) = 2^9 - 1 := by
  sorry

end gcd_power_two_minus_one_l1989_198941


namespace prudence_sleep_weeks_l1989_198912

/-- Represents Prudence's sleep schedule and total sleep time --/
structure SleepSchedule where
  weekdayNights : Nat  -- Number of weekday nights (Sun-Thurs)
  weekendNights : Nat  -- Number of weekend nights (Fri-Sat)
  napDays : Nat        -- Number of days with naps
  weekdaySleep : Nat   -- Hours of sleep on weekday nights
  weekendSleep : Nat   -- Hours of sleep on weekend nights
  napDuration : Nat    -- Duration of naps in hours
  totalSleep : Nat     -- Total hours of sleep

/-- Calculates the number of weeks required to reach the total sleep time --/
def weeksToReachSleep (schedule : SleepSchedule) : Nat :=
  let weeklySleeep := 
    schedule.weekdayNights * schedule.weekdaySleep +
    schedule.weekendNights * schedule.weekendSleep +
    schedule.napDays * schedule.napDuration
  schedule.totalSleep / weeklySleeep

/-- Theorem: Given Prudence's sleep schedule, it takes 4 weeks to reach 200 hours of sleep --/
theorem prudence_sleep_weeks : 
  weeksToReachSleep {
    weekdayNights := 5,
    weekendNights := 2,
    napDays := 2,
    weekdaySleep := 6,
    weekendSleep := 9,
    napDuration := 1,
    totalSleep := 200
  } = 4 := by
  sorry


end prudence_sleep_weeks_l1989_198912


namespace integer_fraction_condition_l1989_198951

theorem integer_fraction_condition (n : ℤ) : 
  (∃ k : ℤ, 16 * (n^2 - n - 1)^2 = k * (2*n - 1)) ↔ 
  n = -12 ∨ n = -2 ∨ n = 0 ∨ n = 1 ∨ n = 3 ∨ n = 13 :=
sorry

end integer_fraction_condition_l1989_198951


namespace largest_constant_inequality_l1989_198995

theorem largest_constant_inequality (x y : ℝ) :
  ∃ (C : ℝ), C = Real.sqrt 5 ∧
  (∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + 10 ≥ C*(x + y + 2)) ∧
  (∀ (D : ℝ), D > C → ∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + 10 < D*(x + y + 2)) :=
by sorry

end largest_constant_inequality_l1989_198995


namespace some_number_value_l1989_198993

theorem some_number_value (x : ℝ) : 60 + 5 * 12 / (180 / x) = 61 → x = 3 := by
  sorry

end some_number_value_l1989_198993


namespace double_pieces_count_l1989_198981

/-- Represents the number of circles on top of a Lego piece -/
inductive PieceType
| Single
| Double
| Triple
| Quadruple

/-- The cost of a Lego piece in cents -/
def cost (p : PieceType) : ℕ :=
  match p with
  | .Single => 1
  | .Double => 2
  | .Triple => 3
  | .Quadruple => 4

/-- The total revenue in cents -/
def total_revenue : ℕ := 1000

/-- The number of single pieces sold -/
def single_count : ℕ := 100

/-- The number of triple pieces sold -/
def triple_count : ℕ := 50

/-- The number of quadruple pieces sold -/
def quadruple_count : ℕ := 165

theorem double_pieces_count :
  ∃ (double_count : ℕ),
    double_count * cost PieceType.Double =
      total_revenue
        - (single_count * cost PieceType.Single
          + triple_count * cost PieceType.Triple
          + quadruple_count * cost PieceType.Quadruple)
    ∧ double_count = 45 := by
  sorry


end double_pieces_count_l1989_198981


namespace sam_remaining_seashells_l1989_198985

def initial_seashells : ℕ := 35
def seashells_given_away : ℕ := 18

theorem sam_remaining_seashells : 
  initial_seashells - seashells_given_away = 17 := by sorry

end sam_remaining_seashells_l1989_198985


namespace angle_ratio_not_sufficient_for_right_triangle_l1989_198905

theorem angle_ratio_not_sufficient_for_right_triangle 
  (A B C : ℝ) (h_sum : A + B + C = 180) (h_ratio : A / 9 = B / 12 ∧ B / 12 = C / 15) :
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) :=
sorry

end angle_ratio_not_sufficient_for_right_triangle_l1989_198905


namespace jacks_paycheck_l1989_198900

theorem jacks_paycheck (paycheck : ℝ) : 
  (paycheck * 0.8 * 0.2 = 20) → paycheck = 125 := by
  sorry

end jacks_paycheck_l1989_198900


namespace correct_product_l1989_198935

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem correct_product (a b : ℕ) : 
  (a ≥ 10 ∧ a ≤ 99) →
  (reverse_digits a * b + 5 = 266) →
  (a * b = 828) :=
by sorry

end correct_product_l1989_198935


namespace triangle_area_l1989_198921

/-- Given a triangle ABC with the following properties:
  * sinB = √2 * sinA
  * ∠C = 105°
  * c = √3 + 1
  Prove that the area of triangle ABC is (√3 + 1) / 2 -/
theorem triangle_area (A B C : ℝ) (h1 : Real.sin B = Real.sqrt 2 * Real.sin A)
  (h2 : C = 105 * π / 180) (h3 : Real.sqrt 3 + 1 = 2 * Real.sin (C / 2) * Real.sin ((A + B) / 2)) :
  (Real.sqrt 3 + 1) / 2 = (Real.sin C) * (Real.sin A) * (Real.sin B) / (Real.sin (A + B)) := by
  sorry

end triangle_area_l1989_198921


namespace xiao_ming_reading_problem_l1989_198953

/-- Represents the problem of finding the minimum number of pages to read per day -/
def min_pages_per_day (total_pages : ℕ) (total_days : ℕ) (initial_days : ℕ) (initial_pages_per_day : ℕ) : ℕ :=
  let remaining_days := total_days - initial_days
  let remaining_pages := total_pages - (initial_days * initial_pages_per_day)
  (remaining_pages + remaining_days - 1) / remaining_days

/-- Theorem stating the solution to Xiao Ming's reading problem -/
theorem xiao_ming_reading_problem :
  min_pages_per_day 72 10 2 5 = 8 :=
by sorry

end xiao_ming_reading_problem_l1989_198953


namespace no_divisor_square_sum_l1989_198915

theorem no_divisor_square_sum (n : ℕ+) :
  ¬∃ d : ℕ+, (d ∣ 2 * n^2) ∧ ∃ x : ℕ, d^2 * n^2 + d^3 = x^2 := by
  sorry

end no_divisor_square_sum_l1989_198915


namespace trivia_contest_probability_l1989_198903

/-- The number of questions in the trivia contest -/
def num_questions : ℕ := 5

/-- The number of possible answers for each question -/
def num_answers : ℕ := 5

/-- The probability of guessing a single question correctly -/
def p_correct : ℚ := 1 / num_answers

/-- The probability of guessing a single question incorrectly -/
def p_incorrect : ℚ := 1 - p_correct

/-- The probability of guessing all questions incorrectly -/
def p_all_incorrect : ℚ := p_incorrect ^ num_questions

/-- The probability of guessing at least one question correctly -/
def p_at_least_one_correct : ℚ := 1 - p_all_incorrect

theorem trivia_contest_probability :
  p_at_least_one_correct = 2101 / 3125 := by
  sorry

end trivia_contest_probability_l1989_198903


namespace ratio_chain_l1989_198999

theorem ratio_chain (a b c d : ℚ) 
  (hab : a / b = 3 / 4)
  (hbc : b / c = 7 / 9)
  (hcd : c / d = 5 / 7) :
  a / d = 1 / 12 := by
sorry

end ratio_chain_l1989_198999


namespace problem_statement_l1989_198965

theorem problem_statement : 
  (∃ x : ℝ, x - 2 > 0) ∧ ¬(∀ x : ℝ, Real.sqrt x < x) := by
  sorry

end problem_statement_l1989_198965


namespace lcm_of_48_and_64_l1989_198955

theorem lcm_of_48_and_64 :
  let a := 48
  let b := 64
  let hcf := 16
  lcm a b = 192 :=
by
  sorry

end lcm_of_48_and_64_l1989_198955


namespace total_fish_l1989_198917

-- Define the number of gold fish and blue fish
def gold_fish : ℕ := 15
def blue_fish : ℕ := 7

-- State the theorem
theorem total_fish : gold_fish + blue_fish = 22 := by
  sorry

end total_fish_l1989_198917


namespace different_color_probability_l1989_198925

/-- The probability of drawing two balls of different colors from a box containing 2 red balls and 3 black balls -/
theorem different_color_probability (total_balls : ℕ) (red_balls : ℕ) (black_balls : ℕ) : 
  total_balls = 5 →
  red_balls = 2 →
  black_balls = 3 →
  (red_balls * black_balls : ℚ) / ((total_balls * (total_balls - 1)) / 2) = 3/5 := by
sorry

end different_color_probability_l1989_198925


namespace boat_stream_speed_ratio_l1989_198918

/-- If rowing against a stream takes twice as long as rowing with the stream for the same distance,
    then the ratio of the boat's speed in still water to the stream's speed is 3:1. -/
theorem boat_stream_speed_ratio 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (distance : ℝ) 
  (h1 : boat_speed > stream_speed) 
  (h2 : stream_speed > 0) 
  (h3 : distance > 0) 
  (h4 : distance / (boat_speed - stream_speed) = 2 * (distance / (boat_speed + stream_speed))) : 
  boat_speed / stream_speed = 3 := by
sorry


end boat_stream_speed_ratio_l1989_198918


namespace decagon_adjacent_vertex_probability_l1989_198960

/-- A decagon is a polygon with 10 vertices -/
def Decagon : Nat := 10

/-- The number of adjacent vertices for each vertex in a decagon -/
def AdjacentVertices : Nat := 2

/-- The probability of choosing two distinct adjacent vertices in a decagon -/
theorem decagon_adjacent_vertex_probability : 
  (AdjacentVertices : ℚ) / (Decagon - 1 : ℚ) = 2 / 9 := by
  sorry

end decagon_adjacent_vertex_probability_l1989_198960


namespace intersection_limit_l1989_198974

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 - 4

-- Define the horizontal line function
def g (m : ℝ) (x : ℝ) : ℝ := m

-- Define L(m) as the x-coordinate of the left endpoint of intersection
noncomputable def L (m : ℝ) : ℝ := -Real.sqrt (m + 4)

-- Define r as a function of m
noncomputable def r (m : ℝ) : ℝ := (L (-m) - L m) / m

-- Theorem statement
theorem intersection_limit :
  ∀ ε > 0, ∃ δ > 0, ∀ m : ℝ, 0 < |m| ∧ |m| < δ ∧ -4 < m ∧ m < 4 →
    |r m - (1/2)| < ε :=
sorry

end intersection_limit_l1989_198974


namespace extra_flowers_count_l1989_198992

def tulips : ℕ := 5
def roses : ℕ := 10
def daisies : ℕ := 8
def lilies : ℕ := 4
def used_flowers : ℕ := 19

def total_picked : ℕ := tulips + roses + daisies + lilies

theorem extra_flowers_count : total_picked - used_flowers = 8 := by
  sorry

end extra_flowers_count_l1989_198992


namespace dance_team_members_l1989_198913

theorem dance_team_members :
  ∀ (track_members choir_members dance_members : ℕ),
    track_members + choir_members + dance_members = 100 →
    choir_members = 2 * track_members →
    dance_members = choir_members + 10 →
    dance_members = 46 := by
  sorry

end dance_team_members_l1989_198913


namespace tank_width_is_six_l1989_198948

/-- Represents the properties of a rectangular tank being filled with water. -/
structure Tank where
  fill_rate : ℝ  -- Cubic feet per hour
  fill_time : ℝ  -- Hours
  length : ℝ     -- Feet
  depth : ℝ      -- Feet

/-- Calculates the volume of a rectangular tank. -/
def tank_volume (t : Tank) (width : ℝ) : ℝ :=
  t.length * width * t.depth

/-- Calculates the volume of water filled in the tank. -/
def filled_volume (t : Tank) : ℝ :=
  t.fill_rate * t.fill_time

/-- Theorem stating that the width of the tank is 6 feet. -/
theorem tank_width_is_six (t : Tank) 
  (h1 : t.fill_rate = 5)
  (h2 : t.fill_time = 60)
  (h3 : t.length = 10)
  (h4 : t.depth = 5) :
  ∃ (w : ℝ), w = 6 ∧ tank_volume t w = filled_volume t :=
sorry

end tank_width_is_six_l1989_198948


namespace invertible_elements_and_inverses_l1989_198978

-- Define the invertible elements and their inverses for modulo 8
def invertible_mod_8 : Set ℤ := {1, 3, 5, 7}
def inverse_mod_8 : ℤ → ℤ
  | 1 => 1
  | 3 => 3
  | 5 => 5
  | 7 => 7
  | _ => 0  -- Default case for non-invertible elements

-- Define the invertible elements and their inverses for modulo 9
def invertible_mod_9 : Set ℤ := {1, 2, 4, 5, 7, 8}
def inverse_mod_9 : ℤ → ℤ
  | 1 => 1
  | 2 => 5
  | 4 => 7
  | 5 => 2
  | 7 => 4
  | 8 => 8
  | _ => 0  -- Default case for non-invertible elements

theorem invertible_elements_and_inverses :
  (∀ x ∈ invertible_mod_8, (x * inverse_mod_8 x) % 8 = 1) ∧
  (∀ x ∈ invertible_mod_9, (x * inverse_mod_9 x) % 9 = 1) :=
by sorry

end invertible_elements_and_inverses_l1989_198978


namespace puppies_adoption_time_l1989_198942

theorem puppies_adoption_time (initial_puppies : ℕ) (additional_puppies : ℕ) (adoption_rate : ℕ) :
  initial_puppies = 10 →
  additional_puppies = 15 →
  adoption_rate = 7 →
  (∃ (days : ℕ), days = 4 ∧ days * adoption_rate ≥ initial_puppies + additional_puppies ∧
   (days - 1) * adoption_rate < initial_puppies + additional_puppies) :=
by sorry

end puppies_adoption_time_l1989_198942


namespace iggy_thursday_miles_l1989_198950

/-- Represents Iggy's running schedule for a week --/
structure RunningSchedule where
  monday : Nat
  tuesday : Nat
  wednesday : Nat
  thursday : Nat
  friday : Nat

/-- Calculates the total miles run in a week --/
def totalMiles (schedule : RunningSchedule) : Nat :=
  schedule.monday + schedule.tuesday + schedule.wednesday + schedule.thursday + schedule.friday

/-- Calculates the total minutes run in a week given a pace in minutes per mile --/
def totalMinutes (schedule : RunningSchedule) (pace : Nat) : Nat :=
  (totalMiles schedule) * pace

/-- Theorem stating that Iggy ran 8 miles on Thursday --/
theorem iggy_thursday_miles :
  ∀ (schedule : RunningSchedule) (pace : Nat),
    schedule.monday = 3 →
    schedule.tuesday = 4 →
    schedule.wednesday = 6 →
    schedule.friday = 3 →
    pace = 10 →
    totalMinutes schedule pace = 4 * 60 →
    schedule.thursday = 8 := by
  sorry


end iggy_thursday_miles_l1989_198950


namespace problem_solution_l1989_198986

theorem problem_solution (x y : ℝ) (h1 : x^(2*y) = 9) (h2 : x = 3) : y = 1 := by
  sorry

end problem_solution_l1989_198986


namespace student_calculation_l1989_198908

theorem student_calculation (x : ℕ) (h : x = 121) : 2 * x - 140 = 102 := by
  sorry

end student_calculation_l1989_198908


namespace negation_of_all_even_divisible_by_two_l1989_198962

theorem negation_of_all_even_divisible_by_two :
  (¬ ∀ n : ℤ, 2 ∣ n → Even n) ↔ (∃ n : ℤ, ¬(2 ∣ n) ∧ ¬(Even n)) :=
by sorry

end negation_of_all_even_divisible_by_two_l1989_198962


namespace coin_diameter_is_14_l1989_198969

/-- The diameter of a coin given its radius -/
def coin_diameter (radius : ℝ) : ℝ := 2 * radius

/-- Theorem: The diameter of a coin with radius 7 cm is 14 cm -/
theorem coin_diameter_is_14 : coin_diameter 7 = 14 := by
  sorry

end coin_diameter_is_14_l1989_198969


namespace monotonic_cubic_function_l1989_198980

/-- A function f: ℝ → ℝ is monotonic if it is either monotonically increasing or monotonically decreasing -/
def IsMonotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y → f x ≤ f y) ∨ (∀ x y, x ≤ y → f y ≤ f x)

/-- The function f(x) = x³ + x² + mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + x^2 + m*x + 1

theorem monotonic_cubic_function (m : ℝ) :
  IsMonotonic (f m) ↔ m ∈ Set.Ici (1/3) :=
sorry

end monotonic_cubic_function_l1989_198980


namespace system_of_equations_l1989_198952

/-- Given a system of equations, prove the values of x, y, and z. -/
theorem system_of_equations : 
  let x := 80 * (1 + 0.11)
  let y := 120 * (1 - 0.15)
  let z := (0.4 * (x + y)) * (1 + 0.2)
  (x = 88.8) ∧ (y = 102) ∧ (z = 91.584) := by
  sorry

end system_of_equations_l1989_198952


namespace environmental_policy_support_l1989_198904

theorem environmental_policy_support (men_support_rate : ℚ) (women_support_rate : ℚ)
  (men_count : ℕ) (women_count : ℕ) 
  (h1 : men_support_rate = 75 / 100)
  (h2 : women_support_rate = 70 / 100)
  (h3 : men_count = 200)
  (h4 : women_count = 800) :
  (men_support_rate * men_count + women_support_rate * women_count) / (men_count + women_count) = 71 / 100 :=
by sorry

end environmental_policy_support_l1989_198904


namespace absolute_value_half_l1989_198922

theorem absolute_value_half (a : ℝ) : 
  |a| = 1/2 → (a = 1/2 ∨ a = -1/2) := by sorry

end absolute_value_half_l1989_198922


namespace range_of_G_l1989_198924

/-- The function G(x) defined as |x+1|-|x-1| for all real x -/
def G (x : ℝ) : ℝ := |x + 1| - |x - 1|

/-- The range of G(x) is [-2,2] -/
theorem range_of_G : Set.range G = Set.Icc (-2) 2 := by sorry

end range_of_G_l1989_198924


namespace banana_profit_calculation_l1989_198938

/-- Calculates the profit from selling bananas given the purchase and selling rates and the total quantity purchased. -/
theorem banana_profit_calculation 
  (purchase_rate_pounds : ℚ) 
  (purchase_rate_dollars : ℚ) 
  (sell_rate_pounds : ℚ) 
  (sell_rate_dollars : ℚ) 
  (total_pounds : ℚ) : 
  purchase_rate_pounds = 3 →
  purchase_rate_dollars = 1/2 →
  sell_rate_pounds = 4 →
  sell_rate_dollars = 1 →
  total_pounds = 72 →
  (sell_rate_dollars / sell_rate_pounds * total_pounds) - 
  (purchase_rate_dollars / purchase_rate_pounds * total_pounds) = 6 := by
sorry

end banana_profit_calculation_l1989_198938


namespace next_simultaneous_occurrence_l1989_198902

def factory_whistle_period : ℕ := 18
def train_bell_period : ℕ := 30
def foghorn_period : ℕ := 45

def start_time : ℕ := 360  -- 6:00 a.m. in minutes since midnight

theorem next_simultaneous_occurrence :
  ∃ (t : ℕ), t > start_time ∧
  t % factory_whistle_period = 0 ∧
  t % train_bell_period = 0 ∧
  t % foghorn_period = 0 ∧
  t - start_time = 90 :=
sorry

end next_simultaneous_occurrence_l1989_198902


namespace unique_cube_difference_l1989_198968

theorem unique_cube_difference (m n : ℕ+) : 
  (∃ k : ℕ+, 2^n.val - 13^m.val = k^3) ↔ m = 2 ∧ n = 9 := by
sorry

end unique_cube_difference_l1989_198968


namespace inequality_solution_l1989_198928

theorem inequality_solution (x : ℝ) : 
  (2 / (x - 2) - 5 / (x - 3) + 3 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔ 
  (x < 2 ∨ (3 < x ∧ x < 4) ∨ 5 < x) :=
by sorry

end inequality_solution_l1989_198928


namespace polar_to_rectangular_conversion_l1989_198943

theorem polar_to_rectangular_conversion :
  let r : ℝ := 6
  let θ : ℝ := 5 * π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = -3 * Real.sqrt 2) ∧ (y = -3 * Real.sqrt 2) := by sorry

end polar_to_rectangular_conversion_l1989_198943


namespace peter_erasers_l1989_198971

theorem peter_erasers (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 8 → received = 3 → total = initial + received → total = 11 := by
  sorry

end peter_erasers_l1989_198971


namespace max_dominoes_9x10_board_l1989_198997

/-- Represents a chessboard with given dimensions -/
structure Chessboard where
  rows : ℕ
  cols : ℕ

/-- Represents a domino with given dimensions -/
structure Domino where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of dominoes that can be placed on a chessboard -/
def max_dominoes (board : Chessboard) (domino : Domino) : ℕ :=
  sorry

/-- Theorem stating the maximum number of 6x1 dominoes on a 9x10 chessboard -/
theorem max_dominoes_9x10_board :
  let board := Chessboard.mk 9 10
  let domino := Domino.mk 6 1
  max_dominoes board domino = 14 := by
  sorry

end max_dominoes_9x10_board_l1989_198997


namespace platform_length_l1989_198963

/-- Calculates the length of a platform given train speed and crossing times -/
theorem platform_length (train_speed : ℝ) (platform_cross_time : ℝ) (man_cross_time : ℝ) :
  train_speed = 72 * (1000 / 3600) →
  platform_cross_time = 30 →
  man_cross_time = 17 →
  (train_speed * platform_cross_time) - (train_speed * man_cross_time) = 260 := by
  sorry

#check platform_length

end platform_length_l1989_198963


namespace three_digit_number_puzzle_l1989_198994

theorem three_digit_number_puzzle :
  ∃ (x y z : ℕ),
    0 ≤ x ∧ x ≤ 9 ∧
    0 ≤ y ∧ y ≤ 9 ∧
    0 ≤ z ∧ z ≤ 9 ∧
    (100 * x + 10 * y + z) + (100 * z + 10 * y + x) = 1252 ∧
    x + y + z = 14 ∧
    x^2 + y^2 + z^2 = 84 ∧
    100 * x + 10 * y + z = 824 ∧
    100 * z + 10 * y + x = 428 :=
by
  sorry

end three_digit_number_puzzle_l1989_198994


namespace f_30_value_l1989_198906

def is_valid_f (f : ℕ+ → ℕ+) : Prop :=
  (∀ n : ℕ+, f (n + 1) > f n) ∧ 
  (∀ m n : ℕ+, f (m * n) = f m * f n) ∧
  (∀ m n : ℕ+, m ≠ n → m ^ (n : ℕ) = n ^ (m : ℕ) → (f m = n ∨ f n = m))

theorem f_30_value (f : ℕ+ → ℕ+) (h : is_valid_f f) : f 30 = 900 := by
  sorry

end f_30_value_l1989_198906


namespace quarters_needed_for_final_soda_l1989_198961

theorem quarters_needed_for_final_soda (total_quarters : ℕ) (soda_cost : ℕ) : 
  total_quarters = 855 → soda_cost = 7 → 
  (soda_cost - (total_quarters % soda_cost)) = 6 := by
sorry

end quarters_needed_for_final_soda_l1989_198961


namespace triangle_angle_B_l1989_198946

theorem triangle_angle_B (a b c A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  3 * a * Real.cos C = 2 * c * Real.cos A →
  Real.tan A = 1 / 3 →
  B = 3 * π / 4 := by
sorry

end triangle_angle_B_l1989_198946


namespace blisters_on_rest_eq_80_l1989_198939

/-- Represents the number of blisters on one arm -/
def blisters_per_arm : ℕ := 60

/-- Represents the total number of blisters -/
def total_blisters : ℕ := 200

/-- Calculates the number of blisters on the rest of the body -/
def blisters_on_rest : ℕ := total_blisters - 2 * blisters_per_arm

theorem blisters_on_rest_eq_80 : blisters_on_rest = 80 := by
  sorry

end blisters_on_rest_eq_80_l1989_198939


namespace f_min_value_a_value_l1989_198977

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2| + 2 * |x - 1|

-- Theorem for the minimum value of f(x)
theorem f_min_value : ∃ (x : ℝ), ∀ (y : ℝ), f x ≤ f y ∧ f x = 3 := by sorry

-- Define the solution set condition
def solution_set_condition (a : ℝ) (m n : ℝ) : Prop :=
  (∀ x, m < x ∧ x < n ↔ f x + x - a < 0) ∧ n - m = 6

-- Theorem for the value of a
theorem a_value : ∀ (m n : ℝ), solution_set_condition 8 m n := by sorry

end f_min_value_a_value_l1989_198977


namespace work_completion_time_l1989_198988

theorem work_completion_time (work : ℝ) (time_renu : ℝ) (time_suma : ℝ) 
  (h1 : time_renu = 8) 
  (h2 : time_suma = 8) 
  (h3 : work > 0) :
  let rate_renu := work / time_renu
  let rate_suma := work / time_suma
  let combined_rate := rate_renu + rate_suma
  work / combined_rate = 4 := by
sorry


end work_completion_time_l1989_198988


namespace arcsin_cos_4pi_over_7_l1989_198909

theorem arcsin_cos_4pi_over_7 : 
  Real.arcsin (Real.cos (4 * π / 7)) = -π / 14 := by
  sorry

end arcsin_cos_4pi_over_7_l1989_198909


namespace odd_scripts_in_final_state_l1989_198972

/-- Represents the state of the box of scripts -/
structure ScriptBox where
  total : Nat
  odd : Nat
  even : Nat

/-- The procedure of selecting and manipulating scripts -/
def select_and_manipulate (box : ScriptBox) : ScriptBox :=
  sorry

/-- Represents the final state of the box -/
def final_state (initial : ScriptBox) : ScriptBox :=
  sorry

theorem odd_scripts_in_final_state :
  ∀ (initial : ScriptBox),
    initial.total = 4032 →
    initial.odd = initial.total / 2 →
    initial.even = initial.total / 2 →
    let final := final_state initial
    final.total = 3 →
    final.odd > 0 →
    final.even > 0 →
    final.odd = 2 := by
  sorry

end odd_scripts_in_final_state_l1989_198972


namespace polynomial_value_l1989_198998

theorem polynomial_value : (3 : ℝ)^6 - 7 * 3 = 708 := by
  sorry

end polynomial_value_l1989_198998


namespace initial_value_theorem_l1989_198901

theorem initial_value_theorem (y : ℕ) (h : y > 0) :
  ∃ x : ℤ, (x : ℤ) + 49 = y^2 ∧ x = y^2 - 49 :=
by sorry

end initial_value_theorem_l1989_198901


namespace smallest_side_range_l1989_198970

theorem smallest_side_range (c : ℝ) (a b d : ℝ) (h1 : c > 0) (h2 : a > 0) (h3 : b > 0) (h4 : d > 0) 
  (h5 : a + b + d = c) (h6 : d = 2 * a) (h7 : a ≤ b) (h8 : a ≤ d) : 
  c / 6 < a ∧ a < c / 4 := by
sorry

end smallest_side_range_l1989_198970


namespace f_max_min_range_l1989_198907

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

/-- The condition for f to have both a maximum and a minimum -/
def has_max_and_min (a : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  (∀ x, f a x ≤ f a x₁) ∧
  (∀ x, f a x ≥ f a x₂)

/-- The theorem stating the range of a for which f has both a maximum and a minimum -/
theorem f_max_min_range :
  ∀ a : ℝ, has_max_and_min a ↔ (a < -3 ∨ a > 6) :=
sorry

end f_max_min_range_l1989_198907


namespace line_mb_value_l1989_198911

/-- Given a line y = mx + b passing through the points (0, -3) and (1, -1), prove that mb = 6 -/
theorem line_mb_value (m b : ℝ) : 
  (0 : ℝ) = m * 0 + b →  -- The line passes through (0, -3)
  (-3 : ℝ) = m * 0 + b →  -- The line passes through (0, -3)
  (-1 : ℝ) = m * 1 + b →  -- The line passes through (1, -1)
  m * b = 6 := by
sorry

end line_mb_value_l1989_198911


namespace no_perfect_square_solution_l1989_198929

theorem no_perfect_square_solution : 
  ¬ ∃ (n : ℕ+) (m : ℕ), n^2 + 12*n - 2006 = m^2 := by
sorry

end no_perfect_square_solution_l1989_198929


namespace inverse_sum_reciprocals_l1989_198982

theorem inverse_sum_reciprocals (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (2 * a⁻¹ + 3 * b⁻¹)⁻¹ = a * b / (2 * b + 3 * a) :=
by sorry

end inverse_sum_reciprocals_l1989_198982


namespace ratio_after_adding_water_l1989_198987

/-- Represents a mixture of alcohol and water -/
structure Mixture where
  alcohol : ℝ
  water : ℝ

/-- Calculates the ratio of alcohol to water in a mixture -/
def ratio (m : Mixture) : ℝ × ℝ :=
  (m.alcohol, m.water)

/-- Adds water to a mixture -/
def add_water (m : Mixture) (amount : ℝ) : Mixture :=
  { alcohol := m.alcohol, water := m.water + amount }

/-- The initial mixture -/
def initial_mixture : Mixture :=
  { alcohol := 4, water := 3 }

/-- The amount of water added -/
def water_added : ℝ := 8

/-- Theorem stating that adding water changes the ratio to 4:11 -/
theorem ratio_after_adding_water :
  ratio (add_water initial_mixture water_added) = (4, 11) := by
  sorry

end ratio_after_adding_water_l1989_198987


namespace fish_sold_correct_l1989_198931

/-- The number of fish initially in stock -/
def initial_stock : ℕ := 200

/-- The number of fish in the new stock -/
def new_stock : ℕ := 200

/-- The final number of fish in stock -/
def final_stock : ℕ := 300

/-- The fraction of remaining fish that become spoiled -/
def spoilage_rate : ℚ := 1/3

/-- The number of fish sold -/
def fish_sold : ℕ := 50

theorem fish_sold_correct :
  (initial_stock - fish_sold - (initial_stock - fish_sold) * spoilage_rate + new_stock : ℚ) = final_stock :=
sorry

end fish_sold_correct_l1989_198931


namespace stationery_cost_theorem_l1989_198996

/-- Calculates the total cost of stationery given the number of boxes of pencils,
    pencils per box, cost per pencil, and cost per pen. -/
def total_stationery_cost (boxes : ℕ) (pencils_per_box : ℕ) (pencil_cost : ℕ) (pen_cost : ℕ) : ℕ :=
  let total_pencils := boxes * pencils_per_box
  let total_pens := 2 * total_pencils + 300
  let pencil_total_cost := total_pencils * pencil_cost
  let pen_total_cost := total_pens * pen_cost
  pencil_total_cost + pen_total_cost

/-- Theorem stating that the total cost of stationery under the given conditions is $18,300. -/
theorem stationery_cost_theorem :
  total_stationery_cost 15 80 4 5 = 18300 := by
  sorry

end stationery_cost_theorem_l1989_198996


namespace insect_legs_l1989_198975

theorem insect_legs (num_insects : ℕ) (total_legs : ℕ) (h1 : num_insects = 8) (h2 : total_legs = 48) :
  total_legs / num_insects = 6 := by
  sorry

end insect_legs_l1989_198975


namespace power_product_exponent_l1989_198934

theorem power_product_exponent (a b : ℝ) : (a^2 * b^3)^2 = a^4 * b^6 := by
  sorry

end power_product_exponent_l1989_198934


namespace circle_sequence_theorem_circle_sequence_theorem_proof_l1989_198932

-- Define a structure for a point in 2D space
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a structure for a circle
structure Circle :=
  (center : Point) (radius : ℝ)

-- Define a structure for a triangle
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

-- Define a function to check if a circle passes through two points
def passesThrough (c : Circle) (p1 p2 : Point) : Prop :=
  (c.center.x - p1.x)^2 + (c.center.y - p1.y)^2 = c.radius^2 ∧
  (c.center.x - p2.x)^2 + (c.center.y - p2.y)^2 = c.radius^2

-- Define a function to check if two circles are tangent
def areTangent (c1 c2 : Circle) : Prop :=
  (c1.center.x - c2.center.x)^2 + (c1.center.y - c2.center.y)^2 = (c1.radius + c2.radius)^2

-- Define the main theorem
theorem circle_sequence_theorem (t : Triangle) 
  (C1 C2 C3 C4 C5 C6 C7 : Circle) : Prop :=
  passesThrough C1 t.A t.B ∧
  passesThrough C2 t.B t.C ∧ areTangent C1 C2 ∧
  passesThrough C3 t.C t.A ∧ areTangent C2 C3 ∧
  passesThrough C4 t.A t.B ∧ areTangent C3 C4 ∧
  passesThrough C5 t.B t.C ∧ areTangent C4 C5 ∧
  passesThrough C6 t.C t.A ∧ areTangent C5 C6 ∧
  passesThrough C7 t.A t.B ∧ areTangent C6 C7
  →
  C7 = C1

-- The proof would go here
theorem circle_sequence_theorem_proof : ∀ t C1 C2 C3 C4 C5 C6 C7, 
  circle_sequence_theorem t C1 C2 C3 C4 C5 C6 C7 :=
sorry

end circle_sequence_theorem_circle_sequence_theorem_proof_l1989_198932


namespace phone_number_proof_l1989_198984

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def transform (n : ℕ) : ℕ :=
  2 * 10000000 + (n / 100000) * 1000000 + 800000 + (n % 100000)

theorem phone_number_proof (x : ℕ) (h1 : is_six_digit x) (h2 : transform x = 81 * x) :
  x = 260000 := by
  sorry

end phone_number_proof_l1989_198984


namespace company_handshakes_l1989_198944

/-- Represents the number of handshakes between employees of different departments -/
def handshakes (total_employees : ℕ) (dept_x_employees : ℕ) (dept_y_employees : ℕ) : ℕ :=
  dept_x_employees * dept_y_employees

/-- Theorem stating the number of handshakes between employees of different departments -/
theorem company_handshakes :
  ∃ (total_employees dept_x_employees dept_y_employees : ℕ),
    total_employees = 50 ∧
    dept_x_employees = 30 ∧
    dept_y_employees = 20 ∧
    total_employees = dept_x_employees + dept_y_employees ∧
    handshakes total_employees dept_x_employees dept_y_employees = 600 := by
  sorry

end company_handshakes_l1989_198944


namespace janets_freelance_rate_janets_freelance_rate_is_33_75_l1989_198976

/-- Calculates Janet's hourly rate as a freelancer given her current job details and additional costs --/
theorem janets_freelance_rate (current_hourly_rate : ℝ) 
  (weekly_hours : ℝ) (weeks_per_month : ℝ) (extra_fica_per_week : ℝ) 
  (healthcare_premium : ℝ) (additional_monthly_income : ℝ) : ℝ :=
  let current_monthly_income := current_hourly_rate * weekly_hours * weeks_per_month
  let additional_costs := extra_fica_per_week * weeks_per_month + healthcare_premium
  let freelance_income := current_monthly_income + additional_monthly_income
  let net_freelance_income := freelance_income - additional_costs
  let monthly_hours := weekly_hours * weeks_per_month
  net_freelance_income / monthly_hours

/-- Proves that Janet's freelance hourly rate is $33.75 given the specified conditions --/
theorem janets_freelance_rate_is_33_75 : 
  janets_freelance_rate 30 40 4 25 400 1100 = 33.75 := by
  sorry

end janets_freelance_rate_janets_freelance_rate_is_33_75_l1989_198976


namespace point_on_bisector_coordinates_l1989_198983

/-- A point on the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The bisector of the first and third quadrants -/
def firstThirdQuadrantBisector (p : Point) : Prop :=
  p.x = p.y

/-- Point P with coordinates (a, 2a-1) -/
def P (a : ℝ) : Point :=
  { x := a, y := 2 * a - 1 }

/-- Theorem stating that if P(a) is on the bisector, its coordinates are (1, 1) -/
theorem point_on_bisector_coordinates :
  ∀ a : ℝ, firstThirdQuadrantBisector (P a) → P a = { x := 1, y := 1 } :=
by
  sorry

end point_on_bisector_coordinates_l1989_198983


namespace pat_earned_stickers_l1989_198973

/-- The number of stickers Pat had at the beginning of the week -/
def initial_stickers : ℕ := 39

/-- The number of stickers Pat had at the end of the week -/
def final_stickers : ℕ := 61

/-- The number of stickers Pat earned during the week -/
def earned_stickers : ℕ := final_stickers - initial_stickers

theorem pat_earned_stickers : earned_stickers = 22 := by sorry

end pat_earned_stickers_l1989_198973


namespace find_numbers_with_difference_and_quotient_equal_l1989_198957

theorem find_numbers_with_difference_and_quotient_equal (x y : ℚ) :
  x - y = 5 ∧ x / y = 5 → x = 25 / 4 ∧ y = 5 / 4 := by
  sorry

end find_numbers_with_difference_and_quotient_equal_l1989_198957


namespace max_value_2x_plus_y_l1989_198959

theorem max_value_2x_plus_y (x y : ℝ) (h1 : x + 2*y ≤ 3) (h2 : y ≥ 0) :
  (∀ x' y', x' + 2*y' ≤ 3 → y' ≥ 0 → 2*x' + y' ≤ 6) ∧ 
  (∃ x₀ y₀, x₀ + 2*y₀ ≤ 3 ∧ y₀ ≥ 0 ∧ 2*x₀ + y₀ = 6) :=
by sorry

end max_value_2x_plus_y_l1989_198959


namespace bowling_team_size_l1989_198990

/-- The number of original players in a bowling team -/
def original_players : ℕ := 7

/-- The original average weight of the team in kg -/
def original_avg : ℚ := 94

/-- The weight of the first new player in kg -/
def new_player1 : ℚ := 110

/-- The weight of the second new player in kg -/
def new_player2 : ℚ := 60

/-- The new average weight of the team after adding two players, in kg -/
def new_avg : ℚ := 92

theorem bowling_team_size :
  (original_avg * original_players + new_player1 + new_player2) / (original_players + 2) = new_avg :=
sorry

end bowling_team_size_l1989_198990


namespace band_members_formation_l1989_198958

theorem band_members_formation :
  ∃! n : ℕ, 200 < n ∧ n < 300 ∧
  (∃ k : ℕ, n = 10 * k + 4) ∧
  (∃ m : ℕ, n = 12 * m + 6) := by
  sorry

end band_members_formation_l1989_198958


namespace angle_of_inclination_at_max_area_l1989_198979

-- Define the line equation
def line_equation (k x y : ℝ) : Prop := y = (k - 1) * x + 2

-- Define the circle equation
def circle_equation (k x y : ℝ) : Prop := x^2 + y^2 + k*x + 2*y + k^2 = 0

-- Define the condition for maximum area of the circle
def max_area_condition (k : ℝ) : Prop := k = 0

-- Theorem statement
theorem angle_of_inclination_at_max_area (k : ℝ) :
  max_area_condition k →
  ∃ (x y : ℝ), line_equation k x y ∧ circle_equation k x y →
  Real.arctan (-1) = 3 * Real.pi / 4 :=
sorry

end angle_of_inclination_at_max_area_l1989_198979


namespace line_parabola_single_intersection_l1989_198940

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line passing through (-3, 1) with slope k
def line (k x y : ℝ) : Prop := y - 1 = k * (x + 3)

-- Define the condition for the line to intersect the parabola at exactly one point
def single_intersection (k : ℝ) : Prop :=
  (k = 0 ∨ k = -1 ∨ k = 2/3) ∧
  (∃! p : ℝ × ℝ, parabola p.1 p.2 ∧ line k p.1 p.2)

-- Theorem statement
theorem line_parabola_single_intersection (k : ℝ) :
  single_intersection k ↔ k = 0 ∨ k = -1 ∨ k = 2/3 :=
sorry

end line_parabola_single_intersection_l1989_198940


namespace negation_equivalence_l1989_198967

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 0 ∧ x^2 - x ≤ 0) ↔ (∀ x : ℝ, x ≤ 0 → x^2 - x > 0) := by
  sorry

end negation_equivalence_l1989_198967


namespace three_zeros_a_range_l1989_198956

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 1/4

noncomputable def g (x : ℝ) : ℝ := -Real.log x

noncomputable def h (a : ℝ) (x : ℝ) : ℝ := min (f a x) (g x)

theorem three_zeros_a_range (a : ℝ) :
  (∃ x y z : ℝ, x < y ∧ y < z ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧
    h a x = 0 ∧ h a y = 0 ∧ h a z = 0 ∧
    (∀ w : ℝ, w > 0 → h a w = 0 → w = x ∨ w = y ∨ w = z)) →
  -5/4 < a ∧ a < -3/4 :=
sorry

end three_zeros_a_range_l1989_198956


namespace polynomial_factor_d_value_l1989_198919

theorem polynomial_factor_d_value :
  ∀ d : ℚ,
  (∀ x : ℚ, (3 * x + 4 = 0) → (5 * x^3 + 17 * x^2 + d * x + 28 = 0)) →
  d = 233 / 9 := by
sorry

end polynomial_factor_d_value_l1989_198919


namespace book_arrangement_count_book_arrangement_proof_l1989_198916

theorem book_arrangement_count : ℕ :=
  let math_books : ℕ := 4
  let english_books : ℕ := 5
  let math_arrangements : ℕ := Nat.factorial (math_books - 1)
  let english_arrangements : ℕ := Nat.factorial (english_books - 1)
  math_arrangements * english_arrangements

theorem book_arrangement_proof :
  book_arrangement_count = 144 :=
by
  sorry

end book_arrangement_count_book_arrangement_proof_l1989_198916


namespace maries_trip_l1989_198989

theorem maries_trip (total_distance : ℚ) 
  (h1 : total_distance / 4 + 15 + total_distance / 6 = total_distance) : 
  total_distance = 180 / 7 := by
  sorry

end maries_trip_l1989_198989


namespace angle_measure_l1989_198964

-- Define a type for angles
structure Angle where
  degrees : ℕ
  minutes : ℕ

-- Define vertical angles
def vertical_angles (a1 a2 : Angle) : Prop := a1 = a2

-- Define complementary angle
def complementary_angle (a : Angle) : Angle :=
  ⟨90 - a.degrees, 60 - a.minutes⟩

-- Theorem statement
theorem angle_measure :
  ∀ (angle1 angle2 : Angle),
  vertical_angles angle1 angle2 →
  complementary_angle angle1 = ⟨79, 32⟩ →
  angle2 = ⟨100, 28⟩ :=
by
  sorry

end angle_measure_l1989_198964


namespace solve_equation_l1989_198937

theorem solve_equation (x : ℝ) (h : 5 + 7 / x = 6 - 5 / x) : x = 12 := by
  sorry

end solve_equation_l1989_198937


namespace train_passing_time_l1989_198920

/-- Proves that a train of given length and speed takes a specific time to pass a stationary point -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 280 →
  train_speed_kmh = 63 →
  passing_time = 16 →
  train_length / (train_speed_kmh * 1000 / 3600) = passing_time := by
  sorry

#check train_passing_time

end train_passing_time_l1989_198920


namespace right_triangle_hypotenuse_l1989_198914

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 80 → 
  b = 150 → 
  c^2 = a^2 + b^2 → 
  c = 170 := by
sorry

end right_triangle_hypotenuse_l1989_198914


namespace negation_of_universal_proposition_l1989_198949

theorem negation_of_universal_proposition :
  (¬ ∀ (x : ℕ+), (1/2 : ℝ)^(x : ℝ) ≤ 1/2) ↔ (∃ (x : ℕ+), (1/2 : ℝ)^(x : ℝ) > 1/2) :=
by sorry

end negation_of_universal_proposition_l1989_198949


namespace expression_value_l1989_198910

theorem expression_value (a b c : ℝ) (h : a * b + b * c + c * a = 3) :
  (a * (b^2 + 3)) / (a + b) + (b * (c^2 + 3)) / (b + c) + (c * (a^2 + 3)) / (c + a) = 6 :=
by sorry

end expression_value_l1989_198910


namespace unique_solution_linear_system_l1989_198926

theorem unique_solution_linear_system :
  ∃! (x y z : ℝ), 
    2*x - 3*y + z = -4 ∧
    5*x - 2*y - 3*z = 7 ∧
    x + y - 4*z = -6 := by
  sorry

end unique_solution_linear_system_l1989_198926


namespace largest_multiple_of_45_with_nine_and_zero_m_div_45_l1989_198991

/-- A function that checks if a natural number consists only of digits 9 and 0 -/
def only_nine_and_zero (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 9 ∨ d = 0

/-- The largest positive integer that is a multiple of 45 and consists only of digits 9 and 0 -/
def m : ℕ := 99990

theorem largest_multiple_of_45_with_nine_and_zero :
  m % 45 = 0 ∧
  only_nine_and_zero m ∧
  ∀ n : ℕ, n % 45 = 0 → only_nine_and_zero n → n ≤ m :=
sorry

theorem m_div_45 : m / 45 = 2222 :=
sorry

end largest_multiple_of_45_with_nine_and_zero_m_div_45_l1989_198991


namespace house_rent_fraction_l1989_198933

theorem house_rent_fraction (salary : ℝ) (house_rent food conveyance left : ℝ) : 
  food = (3/10) * salary →
  conveyance = (1/8) * salary →
  food + conveyance = 3400 →
  left = 1400 →
  house_rent = salary - (food + conveyance + left) →
  house_rent / salary = 2/5 := by
sorry

end house_rent_fraction_l1989_198933


namespace candy_bar_cost_candy_bar_cost_is_7_l1989_198930

def chocolate_cost : ℕ := 3
def extra_cost : ℕ := 4

theorem candy_bar_cost : ℕ :=
  chocolate_cost + extra_cost

#check candy_bar_cost

theorem candy_bar_cost_is_7 : candy_bar_cost = 7 := by
  sorry

end candy_bar_cost_candy_bar_cost_is_7_l1989_198930


namespace allison_bought_28_items_l1989_198923

/-- The number of glue sticks Marie bought -/
def marie_glue_sticks : ℕ := 15

/-- The number of construction paper packs Marie bought -/
def marie_paper_packs : ℕ := 30

/-- The difference in glue sticks between Allison and Marie -/
def glue_stick_difference : ℕ := 8

/-- The ratio of construction paper packs between Marie and Allison -/
def paper_pack_ratio : ℕ := 6

/-- The total number of craft supply items Allison bought -/
def allison_total_items : ℕ := marie_glue_sticks + glue_stick_difference + marie_paper_packs / paper_pack_ratio

theorem allison_bought_28_items : allison_total_items = 28 := by
  sorry

end allison_bought_28_items_l1989_198923


namespace prob_second_science_example_l1989_198927

/-- Represents a set of questions with science and humanities subjects -/
structure QuestionSet where
  total : Nat
  science : Nat
  humanities : Nat
  h_total : total = science + humanities

/-- Calculates the probability of drawing a science question on the second draw,
    given that the first drawn question was a science question -/
def prob_second_science (qs : QuestionSet) : Rat :=
  if qs.science > 0 then
    (qs.science - 1) / (qs.total - 1)
  else
    0

theorem prob_second_science_example :
  let qs : QuestionSet := ⟨5, 3, 2, rfl⟩
  prob_second_science qs = 1/2 := by sorry

end prob_second_science_example_l1989_198927


namespace quadratic_factorization_l1989_198945

theorem quadratic_factorization (c d : ℕ) (hc : c > 0) (hd : d > 0) (hcd : c > d) :
  (∀ x, x^2 - 18*x + 72 = (x - c)*(x - d)) →
  4*d - c = 12 := by
sorry

end quadratic_factorization_l1989_198945


namespace hyperbola_ellipse_same_foci_l1989_198936

/-- Given a hyperbola and an ellipse with the same foci, prove that m = 1/11 -/
theorem hyperbola_ellipse_same_foci (m : ℝ) : 
  (∃ (c : ℝ), c^2 = 2*m ∧ c^2 = (m+1)/6) → m = 1/11 := by
  sorry

end hyperbola_ellipse_same_foci_l1989_198936


namespace reflect_x_three_two_l1989_198954

/-- Reflects a point across the x-axis in a 2D Cartesian coordinate system. -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The coordinates of (3,2) with respect to the x-axis are (3,-2). -/
theorem reflect_x_three_two :
  reflect_x (3, 2) = (3, -2) := by
  sorry

end reflect_x_three_two_l1989_198954
