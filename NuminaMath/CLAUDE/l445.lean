import Mathlib

namespace butternut_figurines_eq_four_l445_44577

/-- The number of figurines that can be created from a block of basswood -/
def basswood_figurines : ℕ := 3

/-- The number of figurines that can be created from a block of Aspen wood -/
def aspen_figurines : ℕ := 2 * basswood_figurines

/-- The total number of figurines Adam can make -/
def total_figurines : ℕ := 245

/-- The number of blocks of basswood Adam has -/
def basswood_blocks : ℕ := 15

/-- The number of blocks of butternut wood Adam has -/
def butternut_blocks : ℕ := 20

/-- The number of blocks of Aspen wood Adam has -/
def aspen_blocks : ℕ := 20

/-- The number of figurines that can be created from a block of butternut wood -/
def butternut_figurines : ℕ := (total_figurines - basswood_blocks * basswood_figurines - aspen_blocks * aspen_figurines) / butternut_blocks

theorem butternut_figurines_eq_four : butternut_figurines = 4 := by
  sorry

end butternut_figurines_eq_four_l445_44577


namespace stewart_farm_ratio_l445_44580

theorem stewart_farm_ratio : 
  ∀ (horse_food_per_day : ℕ) (total_horse_food : ℕ) (num_sheep : ℕ),
    horse_food_per_day = 230 →
    total_horse_food = 12880 →
    num_sheep = 32 →
    let num_horses := total_horse_food / horse_food_per_day
    (num_sheep : ℚ) / num_horses = 4 / 7 := by
  sorry

end stewart_farm_ratio_l445_44580


namespace max_value_theorem_l445_44505

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 1) :
  2*a*b + 2*b*c*Real.sqrt 2 ≤ Real.sqrt 3 ∧ 
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ 
    a'^2 + b'^2 + c'^2 = 1 ∧ 
    2*a'*b' + 2*b'*c'*Real.sqrt 2 = Real.sqrt 3 :=
by sorry

end max_value_theorem_l445_44505


namespace sqrt_inequality_solution_set_l445_44537

theorem sqrt_inequality_solution_set (x : ℝ) :
  Real.sqrt (2 * x + 2) > x - 1 ↔ -1 ≤ x ∧ x ≤ 2 + Real.sqrt 5 := by
  sorry

end sqrt_inequality_solution_set_l445_44537


namespace rectangle_area_l445_44543

/-- Proves that the area of a rectangle is 432 square meters, given that its length is thrice its breadth and its perimeter is 96 meters. -/
theorem rectangle_area (b : ℝ) (l : ℝ) : 
  l = 3 * b →                  -- Length is thrice the breadth
  2 * (l + b) = 96 →           -- Perimeter is 96 meters
  l * b = 432 := by            -- Area is 432 square meters
sorry

end rectangle_area_l445_44543


namespace hike_length_is_48_l445_44535

/-- Represents the length of a multi-day hike --/
structure HikeLength where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ

/-- The conditions of the hike as described in the problem --/
def hike_conditions (h : HikeLength) : Prop :=
  h.day1 + h.day2 + h.day3 = 34 ∧
  (h.day2 + h.day3) / 2 = 12 ∧
  h.day3 + h.day4 + h.day5 = 40 ∧
  h.day1 + h.day3 + h.day5 = 38 ∧
  h.day4 = 14

/-- The theorem stating that given the conditions, the total length of the trail is 48 miles --/
theorem hike_length_is_48 (h : HikeLength) (hc : hike_conditions h) :
  h.day1 + h.day2 + h.day3 + h.day4 + h.day5 = 48 := by
  sorry


end hike_length_is_48_l445_44535


namespace total_distance_not_unique_l445_44568

/-- Represents a part of a journey with a specific speed -/
structure JourneyPart where
  speed : ℝ
  time : ℝ

/-- Represents a complete journey -/
structure Journey where
  parts : List JourneyPart
  totalTime : ℝ

/-- Calculates the distance of a journey part -/
def distanceOfPart (part : JourneyPart) : ℝ :=
  part.speed * part.time

/-- Calculates the total distance of a journey -/
def totalDistance (journey : Journey) : ℝ :=
  (journey.parts.map distanceOfPart).sum

/-- Theorem stating that the total distance cannot be uniquely determined -/
theorem total_distance_not_unique (totalTime : ℝ) (speeds : List ℝ) :
  ∃ (j1 j2 : Journey), 
    j1.totalTime = totalTime ∧ 
    j2.totalTime = totalTime ∧ 
    (j1.parts.map (·.speed)) = speeds ∧ 
    (j2.parts.map (·.speed)) = speeds ∧ 
    totalDistance j1 ≠ totalDistance j2 := by
  sorry

#check total_distance_not_unique

end total_distance_not_unique_l445_44568


namespace smaller_integer_is_49_l445_44573

theorem smaller_integer_is_49 (m n : ℕ) : 
  10 ≤ m ∧ m < 100 ∧  -- m is a 2-digit positive integer
  10 ≤ n ∧ n < 100 ∧  -- n is a 2-digit positive integer
  ∃ k : ℕ, n = 25 * k ∧ -- n is a multiple of 25
  m < n ∧  -- n is larger than m
  (m + n) / 2 = m + n / 100  -- their average equals the decimal number
  → m = 49 := by sorry

end smaller_integer_is_49_l445_44573


namespace catfish_weight_l445_44564

theorem catfish_weight (trout_count : ℕ) (catfish_count : ℕ) (bluegill_count : ℕ)
  (trout_weight : ℝ) (bluegill_weight : ℝ) (total_weight : ℝ)
  (h1 : trout_count = 4)
  (h2 : catfish_count = 3)
  (h3 : bluegill_count = 5)
  (h4 : trout_weight = 2)
  (h5 : bluegill_weight = 2.5)
  (h6 : total_weight = 25)
  (h7 : total_weight = trout_count * trout_weight + catfish_count * (total_weight - trout_count * trout_weight - bluegill_count * bluegill_weight) / catfish_count + bluegill_count * bluegill_weight) :
  (total_weight - trout_count * trout_weight - bluegill_count * bluegill_weight) / catfish_count = 1.5 := by
sorry

end catfish_weight_l445_44564


namespace relatively_prime_dates_february_leap_year_count_l445_44540

/-- The number of days in February during a leap year -/
def leap_year_february_days : ℕ := 29

/-- The month number for February -/
def february_number : ℕ := 2

/-- A function that returns the number of relatively prime dates in February of a leap year -/
def relatively_prime_dates_february_leap_year : ℕ := 
  leap_year_february_days - (leap_year_february_days / february_number)

/-- Theorem stating that the number of relatively prime dates in February of a leap year is 15 -/
theorem relatively_prime_dates_february_leap_year_count : 
  relatively_prime_dates_february_leap_year = 15 := by sorry

end relatively_prime_dates_february_leap_year_count_l445_44540


namespace arithmetic_computation_l445_44528

theorem arithmetic_computation : 8 + 6 * (3 - 8)^2 = 158 := by
  sorry

end arithmetic_computation_l445_44528


namespace rooms_with_two_windows_l445_44567

/-- Represents a building with rooms and windows. -/
structure Building where
  total_windows : ℕ
  rooms_with_four : ℕ
  rooms_with_three : ℕ
  rooms_with_two : ℕ

/-- Conditions for the building. -/
def building_conditions (b : Building) : Prop :=
  b.total_windows = 122 ∧
  b.rooms_with_four = 5 ∧
  b.rooms_with_three = 8 ∧
  b.total_windows = 4 * b.rooms_with_four + 3 * b.rooms_with_three + 2 * b.rooms_with_two

/-- Theorem stating the number of rooms with two windows. -/
theorem rooms_with_two_windows (b : Building) :
  building_conditions b → b.rooms_with_two = 39 := by
  sorry

end rooms_with_two_windows_l445_44567


namespace h_zero_iff_b_eq_seven_fifths_l445_44503

/-- The function h(x) = 5x - 7 -/
def h (x : ℝ) : ℝ := 5 * x - 7

/-- Theorem stating that h(b) = 0 if and only if b = 7/5 -/
theorem h_zero_iff_b_eq_seven_fifths :
  ∀ b : ℝ, h b = 0 ↔ b = 7 / 5 := by sorry

end h_zero_iff_b_eq_seven_fifths_l445_44503


namespace arrange_plates_eq_365240_l445_44545

/-- Number of ways to arrange plates around a circular table with constraints -/
def arrange_plates : ℕ :=
  let total_plates : ℕ := 13
  let blue_plates : ℕ := 6
  let red_plates : ℕ := 3
  let green_plates : ℕ := 3
  let orange_plates : ℕ := 1
  let total_arrangements : ℕ := (Nat.factorial (total_plates - 1)) / (Nat.factorial blue_plates * Nat.factorial red_plates * Nat.factorial green_plates)
  let green_adjacent : ℕ := (Nat.factorial (total_plates - green_plates)) / (Nat.factorial blue_plates * Nat.factorial red_plates)
  let red_adjacent : ℕ := (Nat.factorial (total_plates - red_plates)) / (Nat.factorial blue_plates * Nat.factorial green_plates)
  total_arrangements - green_adjacent - red_adjacent

theorem arrange_plates_eq_365240 : arrange_plates = 365240 := by
  sorry

end arrange_plates_eq_365240_l445_44545


namespace intersection_characterization_l445_44592

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period_two (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x

def matches_x_squared_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x^2

def has_two_distinct_intersections (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = x₁ + a ∧ f x₂ = x₂ + a

theorem intersection_characterization (f : ℝ → ℝ) (a : ℝ) :
  is_even_function f ∧ has_period_two f ∧ matches_x_squared_on_unit_interval f →
  has_two_distinct_intersections f a ↔ ∃ n : ℤ, a = 2 * n ∨ a = 2 * n - 1/4 :=
sorry

end intersection_characterization_l445_44592


namespace whittlesworth_band_size_l445_44547

theorem whittlesworth_band_size (n : ℕ) : 
  (20 * n % 28 = 6) →
  (20 * n % 19 = 5) →
  (20 * n < 1200) →
  (∀ m : ℕ, (20 * m % 28 = 6) → (20 * m % 19 = 5) → (20 * m < 1200) → m ≤ n) →
  20 * n = 2000 :=
by sorry

end whittlesworth_band_size_l445_44547


namespace line_segment_intersection_condition_l445_44579

/-- A line in 2D space defined by the equation ax + y + 2 = 0 -/
structure Line2D where
  a : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Checks if a line intersects a line segment -/
def intersects (l : Line2D) (p q : Point2D) : Prop :=
  sorry

/-- The theorem to be proved -/
theorem line_segment_intersection_condition (l : Line2D) (p q : Point2D) :
  p = Point2D.mk (-2) 1 →
  q = Point2D.mk 3 2 →
  intersects l p q →
  l.a ∈ Set.Ici (3/2) ∪ Set.Iic (-4/3) :=
sorry

end line_segment_intersection_condition_l445_44579


namespace function_inequality_l445_44536

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition f''(x) < f(x) for all x ∈ ℝ
variable (h : ∀ x : ℝ, (deriv (deriv f)) x < f x)

-- State the theorem
theorem function_inequality (f : ℝ → ℝ) (h : ∀ x : ℝ, (deriv (deriv f)) x < f x) :
  f 2 < Real.exp 2 * f 0 ∧ f 2001 < Real.exp 2001 * f 0 := by
  sorry

end function_inequality_l445_44536


namespace player_A_wins_l445_44593

/-- Represents a pile of matches -/
structure Pile :=
  (count : Nat)

/-- Represents the game state -/
structure GameState :=
  (piles : List Pile)

/-- Represents a player's move -/
structure Move :=
  (take : Nat)
  (split : Nat)
  (into : Nat × Nat)

/-- Checks if a move is valid -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  move.take ∈ state.piles.map Pile.count ∧
  move.split ∈ state.piles.map Pile.count ∧
  move.split ≠ move.take ∧
  move.into.1 > 0 ∧ move.into.2 > 0 ∧
  move.into.1 + move.into.2 = move.split

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  { piles := (state.piles.filter (λ p => p.count ≠ move.take ∧ p.count ≠ move.split)) ++
              [Pile.mk move.into.1, Pile.mk move.into.2] }

/-- Checks if the game is over -/
def isGameOver (state : GameState) : Prop :=
  ∀ move, ¬isValidMove state move

/-- Represents the optimal strategy for a player -/
def OptimalStrategy := GameState → Option Move

/-- Theorem: Player A has a winning strategy -/
theorem player_A_wins (initialState : GameState)
  (h : initialState.piles = [Pile.mk 100, Pile.mk 200, Pile.mk 300]) :
  ∃ (strategyA : OptimalStrategy),
    ∀ (strategyB : OptimalStrategy),
      ∃ (finalState : GameState),
        isGameOver finalState ∧
        -- The last move was made by Player B (meaning A wins)
        (∃ (moves : List Move),
          moves.length % 2 = 1 ∧
          finalState = moves.foldl applyMove initialState) :=
sorry

end player_A_wins_l445_44593


namespace solve_lollipop_problem_l445_44578

def lollipop_problem (alison henry diane emily : ℕ) (daily_rate : ℝ) : Prop :=
  henry = alison + 30 ∧
  alison = 60 ∧
  alison * 2 = diane ∧
  emily = 50 ∧
  emily + 10 = diane ∧
  daily_rate = 1.5 ∧
  ∃ (days : ℕ), days = 4 ∧
    (let total := alison + henry + diane + emily
     let first_day := 45
     let rec consumed (n : ℕ) : ℝ :=
       if n = 0 then 0
       else if n = 1 then first_day
       else consumed (n - 1) * daily_rate
     consumed days > total ∧ consumed (days - 1) ≤ total)

theorem solve_lollipop_problem :
  ∃ (alison henry diane emily : ℕ) (daily_rate : ℝ),
    lollipop_problem alison henry diane emily daily_rate :=
by
  sorry

end solve_lollipop_problem_l445_44578


namespace cos_arcsin_plus_arccos_l445_44526

theorem cos_arcsin_plus_arccos : 
  Real.cos (Real.arcsin (3/5) + Real.arccos (-5/13)) = -56/65 := by sorry

end cos_arcsin_plus_arccos_l445_44526


namespace eggs_per_cake_l445_44569

def total_eggs : ℕ := 60
def fridge_eggs : ℕ := 10
def num_cakes : ℕ := 10

theorem eggs_per_cake :
  (total_eggs - fridge_eggs) / num_cakes = 5 := by
  sorry

end eggs_per_cake_l445_44569


namespace amanda_ticket_sales_l445_44561

/-- The number of tickets Amanda sells on the first day -/
def day1_tickets : ℕ := 5 * 4

/-- The number of tickets Amanda sells on the second day -/
def day2_tickets : ℕ := 32

/-- The number of tickets Amanda needs to sell on the third day -/
def day3_tickets : ℕ := 28

/-- The total number of tickets Amanda needs to sell -/
def total_tickets : ℕ := day1_tickets + day2_tickets + day3_tickets

/-- Theorem stating that the total number of tickets Amanda needs to sell is 80 -/
theorem amanda_ticket_sales : total_tickets = 80 := by
  sorry

end amanda_ticket_sales_l445_44561


namespace disjunction_true_l445_44597

theorem disjunction_true (p q : Prop) (hp : p) (hq : ¬q) : p ∨ q := by
  sorry

end disjunction_true_l445_44597


namespace ten_point_circle_chords_l445_44557

/-- The number of chords between non-adjacent points on a circle with n points -/
def non_adjacent_chords (n : ℕ) : ℕ :=
  Nat.choose n 2 - n

/-- Theorem: Given 10 points on a circle, there are 35 chords connecting non-adjacent points -/
theorem ten_point_circle_chords :
  non_adjacent_chords 10 = 35 := by
  sorry

#eval non_adjacent_chords 10  -- This should output 35

end ten_point_circle_chords_l445_44557


namespace quadratic_inequality_solution_l445_44595

-- Define the quadratic function
def f (x : ℝ) := 2 * x^2 + 4 * x - 6

-- Define the solution set
def solution_set := {x : ℝ | f x < 0}

-- State the theorem
theorem quadratic_inequality_solution :
  solution_set = Set.Ioo (-3 : ℝ) 1 := by
  sorry

end quadratic_inequality_solution_l445_44595


namespace completing_square_quadratic_l445_44512

theorem completing_square_quadratic (x : ℝ) : 
  (∃ c, (x^2 + 4*x + 2 = 0) ↔ ((x + 2)^2 = c)) → 
  (∃ c, ((x + 2)^2 = c) ∧ c = 2) :=
by sorry

end completing_square_quadratic_l445_44512


namespace john_cannot_achieve_goal_l445_44532

/-- Represents John's quiz scores throughout the year -/
structure QuizScores where
  total : Nat
  goal_percentage : Rat
  taken : Nat
  high_scores : Nat

/-- Checks if it's possible to achieve the goal given the current scores -/
def can_achieve_goal (scores : QuizScores) : Prop :=
  ∃ (remaining_high_scores : Nat),
    remaining_high_scores ≤ scores.total - scores.taken ∧
    (scores.high_scores + remaining_high_scores : Rat) / scores.total ≥ scores.goal_percentage

/-- John's actual quiz scores -/
def john_scores : QuizScores :=
  { total := 60
  , goal_percentage := 9/10
  , taken := 40
  , high_scores := 32 }

/-- Theorem stating that John cannot achieve his goal -/
theorem john_cannot_achieve_goal :
  ¬(can_achieve_goal john_scores) := by
  sorry

end john_cannot_achieve_goal_l445_44532


namespace brooke_has_eight_customers_l445_44521

/-- Represents Brooke's milk and butter business --/
structure MilkBusiness where
  num_cows : ℕ
  milk_price : ℚ
  butter_price : ℚ
  milk_per_cow : ℕ
  milk_per_customer : ℕ
  total_revenue : ℚ

/-- Calculates the number of customers in Brooke's milk business --/
def calculate_customers (business : MilkBusiness) : ℕ :=
  let total_milk := business.num_cows * business.milk_per_cow
  total_milk / business.milk_per_customer

/-- Theorem stating that Brooke has 8 customers --/
theorem brooke_has_eight_customers :
  let brooke_business : MilkBusiness := {
    num_cows := 12,
    milk_price := 3,
    butter_price := 3/2,
    milk_per_cow := 4,
    milk_per_customer := 6,
    total_revenue := 144
  }
  calculate_customers brooke_business = 8 := by
  sorry

end brooke_has_eight_customers_l445_44521


namespace football_league_analysis_l445_44599

structure Team :=
  (avg_goals_conceded : ℝ)
  (std_dev_goals : ℝ)

def team1 : Team := ⟨1.5, 1.1⟩
def team2 : Team := ⟨2.1, 0.4⟩

def better_defense (t1 t2 : Team) : Prop :=
  t1.avg_goals_conceded < t2.avg_goals_conceded

def more_stable_defense (t1 t2 : Team) : Prop :=
  t1.std_dev_goals < t2.std_dev_goals

def inconsistent_defense (t : Team) : Prop :=
  t.std_dev_goals > 1.0

def rarely_concedes_no_goals (t : Team) : Prop :=
  t.avg_goals_conceded > 2.0 ∧ t.std_dev_goals < 0.5

theorem football_league_analysis :
  (better_defense team1 team2) ∧
  (more_stable_defense team2 team1) ∧
  (inconsistent_defense team1) ∧
  ¬(rarely_concedes_no_goals team2) :=
by sorry

end football_league_analysis_l445_44599


namespace prob_A_wins_sixth_game_l445_44558

/-- Represents a player in the coin tossing game -/
inductive Player : Type
| A : Player
| B : Player

/-- Represents the outcome of a single game -/
inductive GameOutcome : Type
| Win : Player → GameOutcome
| Lose : Player → GameOutcome

/-- Represents the state of the game after a certain number of rounds -/
structure GameState :=
  (round : ℕ)
  (last_loser : Player)

/-- The probability of winning a single coin toss -/
def coin_toss_prob : ℚ := 1/2

/-- The probability of a player winning a game given they start first -/
def win_prob_starting (p : Player) : ℚ := coin_toss_prob

/-- The probability of a player winning a game given they start second -/
def win_prob_second (p : Player) : ℚ := 1 - coin_toss_prob

/-- The probability of player A winning the nth game given the initial state -/
def prob_A_wins_nth_game (n : ℕ) (initial_state : GameState) : ℚ :=
  sorry

theorem prob_A_wins_sixth_game :
  prob_A_wins_nth_game 6 ⟨0, Player.B⟩ = 7/30 :=
sorry

end prob_A_wins_sixth_game_l445_44558


namespace smallest_fourth_lucky_number_l445_44542

/-- Represents a two-digit positive integer -/
structure TwoDigitNumber where
  value : Nat
  is_two_digit : 10 ≤ value ∧ value ≤ 99

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- The main theorem -/
theorem smallest_fourth_lucky_number :
  ∀ (n : TwoDigitNumber),
    (sumOfDigits 46 + sumOfDigits 24 + sumOfDigits 85 + sumOfDigits n.value = 
     (46 + 24 + 85 + n.value) / 4) →
    n.value ≥ 59 := by
  sorry

#eval sumOfDigits 59  -- Expected output: 14
#eval (46 + 24 + 85 + 59) / 4  -- Expected output: 53
#eval sumOfDigits 46 + sumOfDigits 24 + sumOfDigits 85 + sumOfDigits 59  -- Expected output: 53

end smallest_fourth_lucky_number_l445_44542


namespace triangle_max_sin2A_tan2C_l445_44520

theorem triangle_max_sin2A_tan2C (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- a, b, c are positive sides of the triangle
  0 < a ∧ 0 < b ∧ 0 < c →
  -- -c cosB is the arithmetic mean of √2a cosB and √2b cosA
  -c * Real.cos B = (Real.sqrt 2 * a * Real.cos B + Real.sqrt 2 * b * Real.cos A) / 2 →
  -- Maximum value of sin2A•tan²C
  ∃ (max : Real), ∀ (A' B' C' : Real) (a' b' c' : Real),
    0 < A' ∧ 0 < B' ∧ 0 < C' ∧ A' + B' + C' = π →
    0 < a' ∧ 0 < b' ∧ 0 < c' →
    -c' * Real.cos B' = (Real.sqrt 2 * a' * Real.cos B' + Real.sqrt 2 * b' * Real.cos A') / 2 →
    Real.sin (2 * A') * (Real.tan C')^2 ≤ max ∧
    max = 3 - 2 * Real.sqrt 2 :=
by sorry

end triangle_max_sin2A_tan2C_l445_44520


namespace rainfall_difference_l445_44544

theorem rainfall_difference (monday_rain tuesday_rain : Real) 
  (h1 : monday_rain = 0.9)
  (h2 : tuesday_rain = 0.2) :
  monday_rain - tuesday_rain = 0.7 := by
sorry

end rainfall_difference_l445_44544


namespace quadratic_equation_solution_l445_44511

theorem quadratic_equation_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + a*a + b = 0) ∧ (b^2 + a*b + b = 0) → a = 1 ∧ b = -2 := by
  sorry

end quadratic_equation_solution_l445_44511


namespace all_stones_equal_weight_l445_44507

/-- A type representing a stone with an integer weight -/
structure Stone where
  weight : ℤ

/-- A function that checks if a list of 12 stones can be split into two groups of 6 with equal weight -/
def canBalanceAny12 (stones : List Stone) : Prop :=
  stones.length = 13 ∧
  ∀ (subset : List Stone), subset.length = 12 ∧ subset.Sublist stones →
    ∃ (group1 group2 : List Stone),
      group1.length = 6 ∧ group2.length = 6 ∧
      group1.Sublist subset ∧ group2.Sublist subset ∧
      (group1.map Stone.weight).sum = (group2.map Stone.weight).sum

/-- The main theorem -/
theorem all_stones_equal_weight (stones : List Stone) :
  canBalanceAny12 stones →
  ∀ (s1 s2 : Stone), s1 ∈ stones → s2 ∈ stones → s1.weight = s2.weight :=
by sorry

end all_stones_equal_weight_l445_44507


namespace muffin_division_l445_44518

theorem muffin_division (num_friends : ℕ) (total_muffins : ℕ) : 
  num_friends = 4 → total_muffins = 20 → (total_muffins / (num_friends + 1) : ℚ) = 4 := by
  sorry

end muffin_division_l445_44518


namespace training_completion_time_l445_44549

/-- Calculates the number of days required to complete a training regimen. -/
def trainingDays (totalHours : ℕ) (multiplicationMinutes : ℕ) (divisionMinutes : ℕ) : ℕ :=
  let totalMinutes := totalHours * 60
  let dailyMinutes := multiplicationMinutes + divisionMinutes
  totalMinutes / dailyMinutes

/-- Proves that given the specified training schedule, it takes 10 days to complete the training. -/
theorem training_completion_time :
  trainingDays 5 10 20 = 10 := by
  sorry

end training_completion_time_l445_44549


namespace total_days_is_210_l445_44506

/-- Calculates the total number of days spent on two islands given the durations of expeditions. -/
def total_days_on_islands (island_a_first : ℕ) (island_b_first : ℕ) : ℕ :=
  let island_a_second := island_a_first + 2
  let island_a_third := island_a_second * 2
  let island_b_second := island_b_first - 3
  let island_b_third := island_b_first
  let total_weeks := (island_a_first + island_a_second + island_a_third) +
                     (island_b_first + island_b_second + island_b_third)
  total_weeks * 7

/-- Theorem stating that the total number of days spent on both islands is 210. -/
theorem total_days_is_210 : total_days_on_islands 3 5 = 210 := by
  sorry

end total_days_is_210_l445_44506


namespace harmonic_mean_inequality_l445_44594

theorem harmonic_mean_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1 / a + 1 / b > 1 / (a + b) := by
  sorry

end harmonic_mean_inequality_l445_44594


namespace converse_of_negative_square_positive_l445_44504

theorem converse_of_negative_square_positive :
  (∀ x : ℝ, x < 0 → x^2 > 0) →
  (∀ x : ℝ, x^2 > 0 → x < 0) :=
sorry

end converse_of_negative_square_positive_l445_44504


namespace biker_problem_l445_44525

/-- Two bikers on a circular path problem -/
theorem biker_problem (t1 t2 meet_time : ℕ) : 
  t1 = 12 →  -- First rider completes a round in 12 minutes
  meet_time = 36 →  -- They meet again at the starting point after 36 minutes
  meet_time % t1 = 0 →  -- First rider completes whole number of rounds
  meet_time % t2 = 0 →  -- Second rider completes whole number of rounds
  t2 > t1 →  -- Second rider is slower than the first
  t2 = 36  -- Second rider takes 36 minutes to complete a round
  := by sorry

end biker_problem_l445_44525


namespace inequality_solution_inequality_proof_l445_44501

-- Problem 1
def solution_set (x : ℝ) : Prop := x < -7 ∨ x > 5/3

theorem inequality_solution : 
  ∀ x : ℝ, |2*x + 1| - |x - 4| > 2 ↔ solution_set x := by sorry

-- Problem 2
theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by sorry

end inequality_solution_inequality_proof_l445_44501


namespace line_parabola_intersection_l445_44581

/-- The line x = k intersects the parabola x = -3y^2 - 2y + 7 at exactly one point if and only if k = 22/3 -/
theorem line_parabola_intersection (k : ℝ) : 
  (∃! y : ℝ, k = -3 * y^2 - 2 * y + 7) ↔ k = 22 / 3 := by
  sorry

end line_parabola_intersection_l445_44581


namespace unique_linear_function_l445_44571

/-- A linear function passing through two points -/
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

theorem unique_linear_function :
  ∃! (k b : ℝ), linear_function k b 3 = 4 ∧ linear_function k b 4 = 5 ∧
  ∀ x, linear_function k b x = x + 1 :=
sorry

end unique_linear_function_l445_44571


namespace min_digit_ratio_l445_44587

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  hundreds_nonzero : hundreds ≠ 0
  digits_bound : hundreds < 10 ∧ tens < 10 ∧ ones < 10

/-- The value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- The sum of digits of a three-digit number -/
def ThreeDigitNumber.digitSum (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.ones

/-- The ratio of a number to the sum of its digits -/
def digitRatio (n : ThreeDigitNumber) : Rat :=
  n.value / n.digitSum

/-- The condition that the difference between hundreds and tens digit is 8 -/
def diffEight (n : ThreeDigitNumber) : Prop :=
  n.hundreds - n.tens = 8 ∨ n.tens - n.hundreds = 8

theorem min_digit_ratio :
  ∀ k : ThreeDigitNumber,
    diffEight k →
    ∀ m : ThreeDigitNumber,
      diffEight m →
      digitRatio k ≤ digitRatio m →
      k.value = 190 :=
sorry

end min_digit_ratio_l445_44587


namespace point_outside_circle_l445_44516

theorem point_outside_circle (a b : ℝ) 
  (h_intersect : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    a * x₁ + b * y₁ = 1 ∧ 
    a * x₂ + b * y₂ = 1 ∧ 
    x₁^2 + y₁^2 = 1 ∧ 
    x₂^2 + y₂^2 = 1) : 
  a^2 + b^2 > 1 := by
sorry

end point_outside_circle_l445_44516


namespace james_truck_trip_distance_l445_44527

/-- 
Given:
- James gets paid $0.50 per mile to drive a truck.
- Gas costs $4.00 per gallon.
- The truck gets 20 miles per gallon.
- James made a profit of $180 from a trip.

Prove: The length of the trip was 600 miles.
-/
theorem james_truck_trip_distance : 
  let pay_rate : ℝ := 0.50  -- pay rate in dollars per mile
  let gas_price : ℝ := 4.00  -- gas price in dollars per gallon
  let fuel_efficiency : ℝ := 20  -- miles per gallon
  let profit : ℝ := 180  -- profit in dollars
  ∃ distance : ℝ, 
    distance * pay_rate - (distance / fuel_efficiency) * gas_price = profit ∧ 
    distance = 600 := by
  sorry

end james_truck_trip_distance_l445_44527


namespace rectangular_plot_breadth_l445_44541

theorem rectangular_plot_breadth (length width area : ℝ) : 
  length = 3 * width → 
  area = length * width → 
  area = 2028 → 
  width = 26 := by
sorry

end rectangular_plot_breadth_l445_44541


namespace angle_B_measure_l445_44548

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b*cos(C) + (2a+c)*cos(B) = 0, then the measure of angle B is 2π/3 -/
theorem angle_B_measure (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  b * Real.cos C + (2 * a + c) * Real.cos B = 0 →
  B = 2 * π / 3 := by
  sorry

end angle_B_measure_l445_44548


namespace octagon_area_reduction_l445_44517

theorem octagon_area_reduction (x : ℝ) : 
  x > 0 ∧ x < 1 →  -- The smaller square's side length is positive and less than the original square
  4 + 2*x = 1.4 * 4 →  -- Perimeter condition
  (1 - x^2) / 1 = 0.36 :=  -- Area reduction
by sorry

end octagon_area_reduction_l445_44517


namespace whole_number_between_fractions_l445_44596

theorem whole_number_between_fractions (N : ℤ) :
  (3.5 < (N : ℚ) / 5 ∧ (N : ℚ) / 5 < 4.5) ↔ (N = 18 ∨ N = 19 ∨ N = 20 ∨ N = 21 ∨ N = 22) :=
by sorry

end whole_number_between_fractions_l445_44596


namespace function_inequality_range_l445_44590

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |x - a| + |x + 3*a - 2|
def g (a x : ℝ) : ℝ := -x^2 + 2*a*x + 1

-- State the theorem
theorem function_inequality_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, f a x₁ > g a x₂) ↔ 
  (a ∈ Set.Ioo (-2 - Real.sqrt 5) (-2 + Real.sqrt 5) ∪ Set.Ioo 1 3) :=
sorry

end function_inequality_range_l445_44590


namespace gcd_lcm_sum_l445_44559

/-- The greatest common factor of 18, 30, and 45 -/
def C : ℕ := Nat.gcd 18 (Nat.gcd 30 45)

/-- The least common multiple of 18, 30, and 45 -/
def D : ℕ := Nat.lcm 18 (Nat.lcm 30 45)

/-- The sum of the greatest common factor and the least common multiple of 18, 30, and 45 is 93 -/
theorem gcd_lcm_sum : C + D = 93 := by
  sorry

end gcd_lcm_sum_l445_44559


namespace negative_integer_sum_and_square_equals_neg_twelve_l445_44552

theorem negative_integer_sum_and_square_equals_neg_twelve (N : ℤ) :
  N < 0 → N^2 + N = -12 → N = -3 ∨ N = -4 := by
  sorry

end negative_integer_sum_and_square_equals_neg_twelve_l445_44552


namespace exponential_function_extrema_l445_44546

theorem exponential_function_extrema (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x
  let max_val := max (f 1) (f 2)
  let min_val := min (f 1) (f 2)
  max_val + min_val = 12 → a = 3 := by
  sorry

end exponential_function_extrema_l445_44546


namespace solution_and_minimum_value_l445_44574

def A (a : ℕ) : Set ℝ := {x : ℝ | |x - 2| < a}

theorem solution_and_minimum_value (a : ℕ) (h1 : a > 0) 
  (h2 : (3/2 : ℝ) ∈ A a) (h3 : (1/2 : ℝ) ∉ A a) :
  (a = 1) ∧ 
  (∀ x : ℝ, |x + a| + |x - 2| ≥ 3) ∧ 
  (∃ x : ℝ, |x + a| + |x - 2| = 3) := by
sorry

end solution_and_minimum_value_l445_44574


namespace stratified_sampling_calculation_l445_44508

/-- Stratified sampling calculation -/
theorem stratified_sampling_calculation 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (stratum_size : ℕ) 
  (h1 : total_population = 2000) 
  (h2 : sample_size = 200) 
  (h3 : stratum_size = 250) : 
  (stratum_size : ℚ) / total_population * sample_size = 25 := by
  sorry

end stratified_sampling_calculation_l445_44508


namespace lunch_combo_options_count_l445_44510

/-- The number of lunch combo options for Terry at the salad bar -/
def lunch_combo_options : ℕ :=
  let lettuce_types : ℕ := 2
  let tomato_types : ℕ := 3
  let olive_types : ℕ := 4
  let soup_types : ℕ := 2
  lettuce_types * tomato_types * olive_types * soup_types

theorem lunch_combo_options_count : lunch_combo_options = 48 := by
  sorry

end lunch_combo_options_count_l445_44510


namespace square_sum_value_l445_44560

theorem square_sum_value (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) :
  x^2 + y^2 = 20 := by
  sorry

end square_sum_value_l445_44560


namespace complement_P_wrt_U_l445_44598

-- Define the sets U and P
def U : Set ℝ := Set.univ
def P : Set ℝ := Set.Ioo 0 (1/2)

-- State the theorem
theorem complement_P_wrt_U :
  (U \ P) = Set.Iic 0 ∪ Set.Ici (1/2) := by
  sorry

end complement_P_wrt_U_l445_44598


namespace absolute_value_of_S_eq_121380_l445_44554

/-- The sum of all integers b for which x^2 + bx + 2023b can be factored over the integers -/
def S : ℤ := sorry

/-- The polynomial x^2 + bx + 2023b -/
def polynomial (x b : ℤ) : ℤ := x^2 + b*x + 2023*b

/-- Predicate to check if a polynomial can be factored over the integers -/
def is_factorable (b : ℤ) : Prop := ∃ (p q : ℤ → ℤ), ∀ x, polynomial x b = p x * q x

theorem absolute_value_of_S_eq_121380 : |S| = 121380 := by sorry

end absolute_value_of_S_eq_121380_l445_44554


namespace stevens_height_l445_44572

/-- Given a pole's height and shadow length, and a person's shadow length,
    calculate the person's height in centimeters. -/
def calculate_height (pole_height pole_shadow person_shadow : ℚ) : ℚ :=
  let ratio := pole_height / pole_shadow
  let person_height_feet := ratio * (person_shadow / 12)
  person_height_feet * 30.48

/-- Theorem stating that under the given conditions, Steven's height is 190.5 cm. -/
theorem stevens_height :
  let pole_height : ℚ := 60
  let pole_shadow : ℚ := 20
  let steven_shadow_inches : ℚ := 25
  calculate_height pole_height pole_shadow (steven_shadow_inches / 12) = 190.5 := by
  sorry

end stevens_height_l445_44572


namespace even_function_implies_A_equals_one_l445_44538

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = (x + 1)(x - A) -/
def f (A : ℝ) : ℝ → ℝ := λ x ↦ (x + 1) * (x - A)

/-- If f(x) = (x + 1)(x - A) is an even function, then A = 1 -/
theorem even_function_implies_A_equals_one :
  IsEven (f A) → A = 1 := by sorry

end even_function_implies_A_equals_one_l445_44538


namespace base10_512_to_base5_l445_44534

/-- Converts a base-10 number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

theorem base10_512_to_base5 :
  toBase5 512 = [4, 0, 2, 2] :=
sorry

end base10_512_to_base5_l445_44534


namespace arithmetic_mean_square_difference_l445_44509

theorem arithmetic_mean_square_difference (p u v : ℕ) : 
  Nat.Prime p → 
  u ≠ v → 
  u > 0 → 
  v > 0 → 
  p * p = (u * u + v * v) / 2 → 
  ∃ (x : ℕ), (2 * p - u - v = x * x) ∨ (2 * p - u - v = 2 * x * x) := by
sorry

end arithmetic_mean_square_difference_l445_44509


namespace gcd_problem_l445_44588

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2142 * k) : 
  Nat.gcd (Int.natAbs (b^2 + 11*b + 28)) (Int.natAbs (b + 6)) = 2 := by
  sorry

end gcd_problem_l445_44588


namespace no_perfect_square_9999xxxx_l445_44555

theorem no_perfect_square_9999xxxx : ¬∃ x : ℕ, 
  99990000 ≤ x ∧ x ≤ 99999999 ∧ ∃ y : ℕ, x = y^2 := by
  sorry

end no_perfect_square_9999xxxx_l445_44555


namespace equilateral_triangle_area_perimeter_ratio_l445_44576

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 4 -/
theorem equilateral_triangle_area_perimeter_ratio : 
  let side_length : ℝ := 4
  let area : ℝ := side_length^2 * Real.sqrt 3 / 4
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 / 3 := by
  sorry

end equilateral_triangle_area_perimeter_ratio_l445_44576


namespace cyclist_speed_l445_44591

-- Define the parameters of the problem
def first_distance : ℝ := 8
def second_distance : ℝ := 10
def second_speed : ℝ := 8
def total_average_speed : ℝ := 8.78

-- Define the theorem
theorem cyclist_speed (v : ℝ) (h : v > 0) :
  (first_distance + second_distance) / ((first_distance / v) + (second_distance / second_speed)) = total_average_speed →
  v = 10 := by
  sorry

end cyclist_speed_l445_44591


namespace quadrilateral_area_is_76_l445_44583

-- Define the vertices of the quadrilateral
def v1 : ℝ × ℝ := (4, -3)
def v2 : ℝ × ℝ := (4, 7)
def v3 : ℝ × ℝ := (12, 2)
def v4 : ℝ × ℝ := (12, -7)

-- Define the function to calculate the area of the quadrilateral
def quadrilateralArea (v1 v2 v3 v4 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area_is_76 : 
  quadrilateralArea v1 v2 v3 v4 = 76 := by sorry

end quadrilateral_area_is_76_l445_44583


namespace number_satisfies_equation_l445_44553

theorem number_satisfies_equation : ∃ (n : ℕ), n = 14 ∧ 2^n - 2^(n-2) = 3 * 2^12 :=
by
  sorry

end number_satisfies_equation_l445_44553


namespace point_in_third_quadrant_l445_44585

/-- A point P with coordinates (m, 4+2m) is in the third quadrant if and only if m < -2 -/
theorem point_in_third_quadrant (m : ℝ) :
  (m < 0 ∧ 4 + 2 * m < 0) ↔ m < -2 := by
sorry

end point_in_third_quadrant_l445_44585


namespace percentage_increase_is_30_percent_l445_44500

-- Define the initial weight James can lift for 20 meters
def initial_weight : ℝ := 300

-- Define the weight increase for 20 meters
def weight_increase : ℝ := 50

-- Define the weight with straps for 10 meters
def weight_with_straps : ℝ := 546

-- Define the strap increase percentage
def strap_increase : ℝ := 0.20

-- Define the function to calculate the weight for 10 meters with a given percentage increase
def weight_for_10m (p : ℝ) : ℝ := (initial_weight + weight_increase) * (1 + p)

-- Define the function to calculate the weight for 10 meters with straps
def weight_for_10m_with_straps (p : ℝ) : ℝ := weight_for_10m p * (1 + strap_increase)

-- Theorem to prove
theorem percentage_increase_is_30_percent :
  ∃ p : ℝ, p = 0.3 ∧ weight_for_10m_with_straps p = weight_with_straps :=
sorry

end percentage_increase_is_30_percent_l445_44500


namespace correct_sticker_distribution_l445_44524

/-- Represents the number of stickers Miss Walter has and distributes -/
structure StickerDistribution where
  gold : Nat
  silver : Nat
  bronze : Nat
  students : Nat

/-- Calculates the number of stickers each student receives -/
def stickersPerStudent (sd : StickerDistribution) : Nat :=
  (sd.gold + sd.silver + sd.bronze) / sd.students

/-- Theorem stating the correct number of stickers each student receives -/
theorem correct_sticker_distribution :
  ∀ sd : StickerDistribution,
    sd.gold = 50 →
    sd.silver = 2 * sd.gold →
    sd.bronze = sd.silver - 20 →
    sd.students = 5 →
    stickersPerStudent sd = 46 := by
  sorry

end correct_sticker_distribution_l445_44524


namespace abs_z2_minus_z1_equals_sqrt2_l445_44515

theorem abs_z2_minus_z1_equals_sqrt2 : ∀ (z₁ z₂ : ℂ), 
  z₁ = 1 + 2*Complex.I → z₂ = 2 + Complex.I → Complex.abs (z₂ - z₁) = Real.sqrt 2 := by
  sorry

end abs_z2_minus_z1_equals_sqrt2_l445_44515


namespace final_value_of_A_l445_44570

theorem final_value_of_A (A : Int) : A = 20 → -A + 10 = -10 := by
  sorry

end final_value_of_A_l445_44570


namespace permutation_problem_l445_44531

theorem permutation_problem (n : ℕ) : (n * (n - 1) = 132) ↔ (n = 12) := by sorry

end permutation_problem_l445_44531


namespace parallel_vectors_sum_l445_44589

/-- Two vectors in ℝ² are parallel if the ratio of their components is constant -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 / b.1 = a.2 / b.2

/-- The sum of two vectors in ℝ² -/
def vec_sum (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 + b.1, a.2 + b.2)

theorem parallel_vectors_sum :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → vec_sum a b = (-2, -1) := by
sorry

end parallel_vectors_sum_l445_44589


namespace common_root_equations_l445_44551

theorem common_root_equations (k : ℝ) :
  (∃ x : ℝ, x^2 - k*x - 7 = 0 ∧ x^2 - 6*x - (k + 1) = 0) →
  (k = -6 ∧
   (∃ x : ℝ, x^2 + 6*x - 7 = 0 ∧ x^2 - 6*x + 5 = 0 ∧ x = 1) ∧
   (∃ y z : ℝ, y^2 + 6*y - 7 = 0 ∧ z^2 - 6*z + 5 = 0 ∧ y = -7 ∧ z = 5)) :=
by sorry

end common_root_equations_l445_44551


namespace cube_labeling_impossibility_cube_labeling_with_13_l445_44565

/-- The number of edges in a cube -/
def num_edges : ℕ := 12

/-- The number of vertices in a cube -/
def num_vertices : ℕ := 8

/-- The number of edges connected to each vertex in a cube -/
def edges_per_vertex : ℕ := 3

/-- A labeling of a cube's edges -/
def Labeling := Fin num_edges → ℕ

/-- The sum of labels at a vertex for a given labeling -/
def vertex_sum (l : Labeling) : ℕ := sorry

/-- Predicate for a valid labeling with values 1 to 12 -/
def valid_labeling (l : Labeling) : Prop :=
  ∀ e : Fin num_edges, l e ∈ Finset.range num_edges

/-- Predicate for a constant sum labeling -/
def constant_sum_labeling (l : Labeling) : Prop :=
  ∃ s : ℕ, ∀ v : Fin num_vertices, vertex_sum l = s

/-- Predicate for a valid labeling with one value replaced by 13 -/
def valid_labeling_with_13 (l : Labeling) : Prop :=
  ∃ e : Fin num_edges, l e = 13 ∧
    ∀ e' : Fin num_edges, e' ≠ e → l e' ∈ Finset.range num_edges

theorem cube_labeling_impossibility :
  ¬∃ l : Labeling, valid_labeling l ∧ constant_sum_labeling l :=
sorry

theorem cube_labeling_with_13 :
  ∃ l : Labeling, valid_labeling_with_13 l ∧ constant_sum_labeling l ↔
    ∃ i ∈ ({3, 7, 11} : Finset ℕ), ∃ l : Labeling,
      valid_labeling_with_13 l ∧ constant_sum_labeling l ∧
      ∃ e : Fin num_edges, l e = 13 ∧ (∀ e' : Fin num_edges, e' ≠ e → l e' ≠ i) :=
sorry

end cube_labeling_impossibility_cube_labeling_with_13_l445_44565


namespace double_root_condition_l445_44530

theorem double_root_condition (m : ℝ) :
  (∃! x : ℝ, (x - 3) / (x - 1) = m / (x - 1) ∧ x ≠ 1) ↔ m = -2 := by
  sorry

end double_root_condition_l445_44530


namespace initial_snatch_weight_l445_44519

/-- Represents John's weightlifting progress --/
structure Weightlifter where
  initialCleanAndJerk : ℝ
  initialSnatch : ℝ
  newCleanAndJerk : ℝ
  newSnatch : ℝ
  newTotal : ℝ

/-- Theorem stating that given the conditions, John's initial Snatch weight was 50 kg --/
theorem initial_snatch_weight (john : Weightlifter) :
  john.initialCleanAndJerk = 80 ∧
  john.newCleanAndJerk = 2 * john.initialCleanAndJerk ∧
  john.newSnatch = 1.8 * john.initialSnatch ∧
  john.newTotal = 250 ∧
  john.newTotal = john.newCleanAndJerk + john.newSnatch →
  john.initialSnatch = 50 := by
  sorry

#check initial_snatch_weight

end initial_snatch_weight_l445_44519


namespace opposite_of_negative_six_l445_44529

theorem opposite_of_negative_six : ∃ x : ℤ, ((-6 : ℤ) + x = 0) ∧ x = 6 := by
  sorry

end opposite_of_negative_six_l445_44529


namespace fraction_equality_l445_44550

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 2) 
  (h2 : r / t = 7 / 15) : 
  (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -3 / 5 := by
  sorry

end fraction_equality_l445_44550


namespace min_value_fraction_l445_44582

theorem min_value_fraction (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 5) :
  ∃ (m : ℝ), m = 3 ∧ ∀ z, z = (x * y) / x → m ≤ z :=
by sorry

end min_value_fraction_l445_44582


namespace box_volume_increase_l445_44502

/-- 
A rectangular box with length l, width w, and height h.
Given the conditions:
1. Volume is 5400 cubic inches
2. Surface area is 2352 square inches
3. Sum of the lengths of its 12 edges is 240 inches
Prove that increasing each dimension by 1 inch results in a volume of 6637 cubic inches
-/
theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 5400)
  (surface_area : 2 * l * w + 2 * w * h + 2 * h * l = 2352)
  (edge_sum : 4 * l + 4 * w + 4 * h = 240) :
  (l + 1) * (w + 1) * (h + 1) = 6637 := by
  sorry

end box_volume_increase_l445_44502


namespace black_squares_in_29th_row_l445_44575

/-- Represents the number of squares in a row of the pattern -/
def squaresInRow (n : ℕ) : ℕ := 1 + 2 * (n - 1)

/-- Represents the number of black squares in a row of the pattern -/
def blackSquaresInRow (n : ℕ) : ℕ := (squaresInRow n - 1) / 2

/-- Theorem stating that the 29th row contains 28 black squares -/
theorem black_squares_in_29th_row : blackSquaresInRow 29 = 28 := by
  sorry

end black_squares_in_29th_row_l445_44575


namespace fraction_equality_l445_44523

theorem fraction_equality : (7 + 21) / (14 + 42) = 1 / 2 := by
  sorry

end fraction_equality_l445_44523


namespace min_value_of_sum_l445_44539

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y + 2 * x + y = 8) :
  x + y ≥ 2 * Real.sqrt 10 - 3 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ * y₀ + 2 * x₀ + y₀ = 8 ∧ x₀ + y₀ = 2 * Real.sqrt 10 - 3 :=
by sorry

end min_value_of_sum_l445_44539


namespace no_roots_of_composite_l445_44566

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem no_roots_of_composite (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x, f a b c x ≠ 2 * x) →
  (∀ x, f a b c (f a b c x) ≠ 4 * x) :=
sorry

end no_roots_of_composite_l445_44566


namespace sprinkles_remaining_l445_44522

theorem sprinkles_remaining (initial_cans : ℕ) (remaining_cans : ℕ) : 
  initial_cans = 12 → 
  remaining_cans = initial_cans / 2 - 3 → 
  remaining_cans = 3 := by
sorry

end sprinkles_remaining_l445_44522


namespace previous_average_production_l445_44563

theorem previous_average_production (n : ℕ) (today_production : ℕ) (new_average : ℚ) :
  n = 9 →
  today_production = 100 →
  new_average = 55 →
  let previous_total := n * (((n + 1) : ℚ) * new_average - today_production) / n
  previous_total / n = 50 := by sorry

end previous_average_production_l445_44563


namespace solution_inequality_l445_44533

theorem solution_inequality (x : ℝ) (h : x = 1.8) : x < 2 := by
  sorry

end solution_inequality_l445_44533


namespace no_fraternity_member_is_club_member_l445_44513

-- Define the universe
variable (U : Type)

-- Define predicates
variable (Student : U → Prop)
variable (ClubMember : U → Prop)
variable (FraternityMember : U → Prop)
variable (Honest : U → Prop)

-- State the theorem
theorem no_fraternity_member_is_club_member
  (h1 : ∀ x, ClubMember x → Honest x)
  (h2 : ∃ x, Student x ∧ ¬Honest x)
  (h3 : ∀ x, Student x → FraternityMember x → ¬ClubMember x) :
  ∀ x, FraternityMember x → ¬ClubMember x :=
by
  sorry

end no_fraternity_member_is_club_member_l445_44513


namespace smallest_third_term_of_gp_l445_44586

theorem smallest_third_term_of_gp (a b c : ℝ) : 
  (∃ d : ℝ, a = 5 ∧ b = 5 + d ∧ c = 5 + 2*d) →  -- arithmetic progression
  (∃ r : ℝ, 5 * (20 + 2*c - 10) = (8 + b - 5)^2) →  -- geometric progression after modification
  20 + 2*c - 10 ≥ -4 :=
by sorry

end smallest_third_term_of_gp_l445_44586


namespace largest_integer_problem_l445_44556

theorem largest_integer_problem (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} : Finset ℕ) = {57, 70, 83} →
  max a (max b (max c (max d e))) = 48 := by
sorry

end largest_integer_problem_l445_44556


namespace mom_gets_eighteen_strawberries_l445_44562

def strawberries_for_mom (dozen_picked : ℕ) (eaten : ℕ) : ℕ :=
  dozen_picked * 12 - eaten

theorem mom_gets_eighteen_strawberries :
  strawberries_for_mom 2 6 = 18 := by
  sorry

end mom_gets_eighteen_strawberries_l445_44562


namespace line_ellipse_intersection_slopes_l445_44584

/-- The set of possible slopes for a line with y-intercept (0, -3) that intersects the ellipse 4x^2 + 25y^2 = 100 -/
def possible_slopes : Set ℝ :=
  {m : ℝ | m ≤ -Real.sqrt (2/110) ∨ m ≥ Real.sqrt (2/110)}

/-- Theorem stating the possible slopes of the line -/
theorem line_ellipse_intersection_slopes :
  ∀ (m : ℝ), (∃ (x y : ℝ), 4*x^2 + 25*y^2 = 100 ∧ y = m*x - 3) ↔ m ∈ possible_slopes := by
  sorry

end line_ellipse_intersection_slopes_l445_44584


namespace correct_calculation_l445_44514

theorem correct_calculation : (-2)^3 + 6 / ((1/2) - (1/3)) = 28 := by
  sorry

end correct_calculation_l445_44514
