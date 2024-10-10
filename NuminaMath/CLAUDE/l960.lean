import Mathlib

namespace coffee_stock_percentage_l960_96086

theorem coffee_stock_percentage (initial_stock : ℝ) (initial_decaf_percent : ℝ)
  (additional_stock : ℝ) (additional_decaf_percent : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 30)
  (h3 : additional_stock = 100)
  (h4 : additional_decaf_percent = 60) :
  let total_stock := initial_stock + additional_stock
  let total_decaf := (initial_stock * initial_decaf_percent / 100) +
                     (additional_stock * additional_decaf_percent / 100)
  total_decaf / total_stock * 100 = 36 := by
sorry

end coffee_stock_percentage_l960_96086


namespace stating_thirty_cents_combinations_l960_96038

/-- The value of a penny in cents -/
def pennyValue : ℕ := 1

/-- The value of a nickel in cents -/
def nickelValue : ℕ := 5

/-- The value of a dime in cents -/
def dimeValue : ℕ := 10

/-- The total value we want to achieve in cents -/
def totalValue : ℕ := 30

/-- 
A function that calculates the number of ways to make a given amount of cents
using pennies, nickels, and dimes.
-/
def countCombinations (cents : ℕ) : ℕ := sorry

/-- 
Theorem stating that the number of combinations to make 30 cents
using pennies, nickels, and dimes is 20.
-/
theorem thirty_cents_combinations : 
  countCombinations totalValue = 20 := by sorry

end stating_thirty_cents_combinations_l960_96038


namespace record_storage_space_theorem_l960_96061

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

theorem record_storage_space_theorem 
  (box_dims : BoxDimensions)
  (storage_cost_per_box : ℝ)
  (total_monthly_payment : ℝ)
  (h1 : box_dims.length = 15)
  (h2 : box_dims.width = 12)
  (h3 : box_dims.height = 10)
  (h4 : storage_cost_per_box = 0.2)
  (h5 : total_monthly_payment = 120) :
  (total_monthly_payment / storage_cost_per_box) * boxVolume box_dims = 1080000 := by
  sorry

#check record_storage_space_theorem

end record_storage_space_theorem_l960_96061


namespace circles_externally_tangent_l960_96073

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Determines if two circles are externally tangent --/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2 = (c1.radius + c2.radius) ^ 2

theorem circles_externally_tangent : 
  let c1 : Circle := { center := (4, 2), radius := 3 }
  let c2 : Circle := { center := (0, -1), radius := 2 }
  are_externally_tangent c1 c2 := by sorry

end circles_externally_tangent_l960_96073


namespace divisor_coloring_game_strategy_l960_96059

/-- A player in the divisor coloring game -/
inductive Player
| A
| B

/-- The result of the divisor coloring game -/
inductive GameResult
| AWins
| BWins

/-- The divisor coloring game for a positive integer n -/
def divisorColoringGame (n : ℕ+) : GameResult := sorry

/-- Check if a number is a perfect square -/
def isPerfectSquare (n : ℕ+) : Prop := ∃ m : ℕ+, n = m * m

/-- Theorem: Player A wins if and only if n is a perfect square or prime -/
theorem divisor_coloring_game_strategy (n : ℕ+) :
  divisorColoringGame n = GameResult.AWins ↔ isPerfectSquare n ∨ Nat.Prime n.val := by sorry

end divisor_coloring_game_strategy_l960_96059


namespace flour_per_pizza_l960_96040

def carnival_time : ℕ := 7 * 60 -- 7 hours in minutes
def flour_amount : ℚ := 22 -- 22 kg of flour
def pizza_time : ℕ := 10 -- 10 minutes per pizza
def extra_pizzas : ℕ := 2 -- 2 additional pizzas from leftover flour

theorem flour_per_pizza :
  let total_pizzas := carnival_time / pizza_time + extra_pizzas
  flour_amount / total_pizzas = 1/2 := by sorry

end flour_per_pizza_l960_96040


namespace same_color_marble_probability_l960_96046

/-- The probability of drawing three marbles of the same color from a bag containing
    6 red marbles, 4 white marbles, and 8 blue marbles, without replacement. -/
theorem same_color_marble_probability :
  let red : ℕ := 6
  let white : ℕ := 4
  let blue : ℕ := 8
  let total : ℕ := red + white + blue
  let prob_same_color : ℚ := (Nat.choose red 3 + Nat.choose white 3 + Nat.choose blue 3 : ℚ) / Nat.choose total 3
  prob_same_color = 5 / 51 :=
by sorry

end same_color_marble_probability_l960_96046


namespace musician_earnings_per_song_l960_96041

/-- Represents the earnings of a musician over a period of time --/
structure MusicianEarnings where
  songs_per_month : ℕ
  total_earnings : ℕ
  years : ℕ

/-- Calculates the earnings per song for a musician --/
def earnings_per_song (m : MusicianEarnings) : ℚ :=
  m.total_earnings / (m.songs_per_month * 12 * m.years)

/-- Theorem: A musician releasing 3 songs per month and earning $216,000 in 3 years makes $2,000 per song --/
theorem musician_earnings_per_song :
  let m : MusicianEarnings := { songs_per_month := 3, total_earnings := 216000, years := 3 }
  earnings_per_song m = 2000 := by
  sorry


end musician_earnings_per_song_l960_96041


namespace distance_not_half_radius_l960_96090

/-- Two circles with radii p and p/2, whose centers are a non-zero distance d apart -/
structure TwoCircles (p : ℝ) where
  d : ℝ
  d_pos : d > 0

/-- Theorem: The distance between the centers cannot be p/2 -/
theorem distance_not_half_radius (p : ℝ) (circles : TwoCircles p) :
  circles.d ≠ p / 2 := by
  sorry

end distance_not_half_radius_l960_96090


namespace melissa_points_per_game_l960_96005

-- Define the total points scored
def total_points : ℕ := 1200

-- Define the number of games played
def num_games : ℕ := 10

-- Define the points per game
def points_per_game : ℕ := total_points / num_games

-- Theorem statement
theorem melissa_points_per_game : points_per_game = 120 := by
  sorry

end melissa_points_per_game_l960_96005


namespace cos_pi_minus_2theta_l960_96001

theorem cos_pi_minus_2theta (θ : Real) (h : ∃ (x y : Real), x = 3 ∧ y = -4 ∧ x = Real.cos θ * Real.sqrt (x^2 + y^2) ∧ y = Real.sin θ * Real.sqrt (x^2 + y^2)) :
  Real.cos (π - 2*θ) = 7/25 := by
sorry

end cos_pi_minus_2theta_l960_96001


namespace square_difference_plus_double_l960_96003

theorem square_difference_plus_double (x y : ℝ) (h : x + y = 1) : x^2 - y^2 + 2*y = 1 := by
  sorry

end square_difference_plus_double_l960_96003


namespace train_crossing_time_l960_96017

/-- Time taken for two trains to cross each other -/
theorem train_crossing_time (train_length : ℝ) (fast_speed : ℝ) : 
  train_length = 100 →
  fast_speed = 24 →
  (50 : ℝ) / 9 = (2 * train_length) / (fast_speed + fast_speed / 2) := by
  sorry

#eval (50 : ℚ) / 9

end train_crossing_time_l960_96017


namespace ivan_total_distance_l960_96098

/-- Represents the distances Ivan ran on each day of the week -/
structure WeeklyRun where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ

/-- The conditions of Ivan's running schedule -/
def validWeeklyRun (run : WeeklyRun) : Prop :=
  run.tuesday = 2 * run.monday ∧
  run.wednesday = run.tuesday / 2 ∧
  run.thursday = run.wednesday / 2 ∧
  run.friday = 2 * run.thursday ∧
  min run.monday (min run.tuesday (min run.wednesday (min run.thursday run.friday))) = 5

/-- The theorem stating that the total distance Ivan ran is 55 km -/
theorem ivan_total_distance (run : WeeklyRun) (h : validWeeklyRun run) :
  run.monday + run.tuesday + run.wednesday + run.thursday + run.friday = 55 := by
  sorry


end ivan_total_distance_l960_96098


namespace quadratic_inequality_l960_96075

theorem quadratic_inequality (x : ℝ) : -9 * x^2 + 6 * x + 8 > 0 ↔ -2/3 < x ∧ x < 4/3 := by
  sorry

end quadratic_inequality_l960_96075


namespace marbles_distribution_l960_96052

theorem marbles_distribution (total_marbles : ℕ) (num_boys : ℕ) (marbles_per_boy : ℕ) :
  total_marbles = 99 →
  num_boys = 11 →
  total_marbles = num_boys * marbles_per_boy →
  marbles_per_boy = 9 := by
  sorry

end marbles_distribution_l960_96052


namespace swimmers_speed_l960_96095

/-- Swimmer's speed in still water given time, distance, and current speed -/
theorem swimmers_speed (time : ℝ) (distance : ℝ) (current_speed : ℝ) 
  (h1 : time = 2.5)
  (h2 : distance = 5)
  (h3 : current_speed = 2) :
  ∃ v : ℝ, v = 4 ∧ time = distance / (v - current_speed) :=
by sorry

end swimmers_speed_l960_96095


namespace baker_cakes_theorem_l960_96096

def total_cakes (initial : ℕ) (extra : ℕ) : ℕ :=
  initial + extra

theorem baker_cakes_theorem (initial : ℕ) (extra : ℕ) :
  total_cakes initial extra = initial + extra := by
  sorry

end baker_cakes_theorem_l960_96096


namespace parallel_vectors_m_value_l960_96089

/-- Given two parallel vectors a and b in R², prove that m = -3 --/
theorem parallel_vectors_m_value (m : ℝ) :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (1 + m, 1 - m)
  (∃ (k : ℝ), k ≠ 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2) →
  m = -3 := by
sorry


end parallel_vectors_m_value_l960_96089


namespace unique_three_digit_number_divisible_by_11_l960_96023

theorem unique_three_digit_number_divisible_by_11 :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
  n % 10 = 7 ∧ 
  (n / 100) % 10 = 8 ∧ 
  n % 11 = 0 ∧
  n = 847 :=
by sorry

end unique_three_digit_number_divisible_by_11_l960_96023


namespace total_nailcutter_sounds_l960_96034

/-- The number of nails per person (fingers and toes combined) -/
def nails_per_person : ℕ := 20

/-- The number of customers -/
def num_customers : ℕ := 3

/-- The number of sounds produced when trimming one nail -/
def sounds_per_nail : ℕ := 1

/-- Theorem: The total number of nailcutter sounds for 3 customers is 60 -/
theorem total_nailcutter_sounds :
  nails_per_person * num_customers * sounds_per_nail = 60 := by
  sorry

end total_nailcutter_sounds_l960_96034


namespace smallest_value_w4_plus_z4_l960_96081

theorem smallest_value_w4_plus_z4 (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2)
  (h2 : Complex.abs (w^2 + z^2) = 10) :
  Complex.abs (w^4 + z^4) = 82 := by
  sorry

end smallest_value_w4_plus_z4_l960_96081


namespace election_winner_votes_l960_96065

theorem election_winner_votes 
  (total_votes : ℕ) 
  (winner_percentage : ℚ) 
  (vote_difference : ℕ) 
  (h1 : winner_percentage = 62 / 100) 
  (h2 : winner_percentage * total_votes - (1 - winner_percentage) * total_votes = vote_difference) 
  (h3 : vote_difference = 348) : 
  ⌊winner_percentage * total_votes⌋ = 899 :=
sorry

end election_winner_votes_l960_96065


namespace line_through_p_equally_divided_l960_96050

/-- The line passing through P(3,0) and equally divided by P in the segment AB
    between lines 2x - y - 2 = 0 and x + y + 3 = 0 has the equation 4x - 5y = 12 -/
theorem line_through_p_equally_divided : 
  let P : ℝ × ℝ := (3, 0)
  let line1 : ℝ → ℝ → Prop := λ x y => 2 * x - y - 2 = 0
  let line2 : ℝ → ℝ → Prop := λ x y => x + y + 3 = 0
  let sought_line : ℝ → ℝ → Prop := λ x y => 4 * x - 5 * y = 12
  ∃ A B : ℝ × ℝ,
    (line1 A.1 A.2 ∧ sought_line A.1 A.2) ∧ 
    (line2 B.1 B.2 ∧ sought_line B.1 B.2) ∧
    (P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2) ∧
    sought_line P.1 P.2 :=
by sorry

end line_through_p_equally_divided_l960_96050


namespace power_function_value_l960_96068

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- Define the theorem
theorem power_function_value (f : ℝ → ℝ) :
  isPowerFunction f → f 2 = 4 → f (-3) = 9 := by
  sorry

end power_function_value_l960_96068


namespace manuscript_cost_theorem_l960_96021

/-- Represents the cost of typing and revising a manuscript. -/
def manuscript_cost (
  total_pages : ℕ
  ) (
  first_type_cost : ℕ
  ) (
  first_revision_cost : ℕ
  ) (
  second_revision_cost : ℕ
  ) (
  third_plus_revision_cost : ℕ
  ) (
  pages_revised_once : ℕ
  ) (
  pages_revised_twice : ℕ
  ) (
  pages_revised_thrice : ℕ
  ) (
  pages_revised_four_times : ℕ
  ) : ℕ :=
  total_pages * first_type_cost +
  pages_revised_once * first_revision_cost +
  pages_revised_twice * (first_revision_cost + second_revision_cost) +
  pages_revised_thrice * (first_revision_cost + second_revision_cost + third_plus_revision_cost) +
  pages_revised_four_times * (first_revision_cost + second_revision_cost + 2 * third_plus_revision_cost)

/-- Theorem: The total cost of typing and revising the manuscript is $2240. -/
theorem manuscript_cost_theorem :
  manuscript_cost 270 5 3 2 1 90 60 30 20 = 2240 := by
  sorry


end manuscript_cost_theorem_l960_96021


namespace accommodation_arrangements_count_l960_96039

/-- Represents the types of rooms in the hotel -/
inductive RoomType
  | Triple
  | Double
  | Single

/-- Represents a person staying in the hotel -/
inductive Person
  | Adult
  | Child

/-- Calculates the number of ways to arrange accommodation for 3 adults and 2 children
    in a hotel with one triple room, one double room, and one single room,
    where children must be accompanied by an adult -/
def accommodationArrangements (rooms : List RoomType) (people : List Person) : Nat :=
  sorry

/-- The main theorem stating that there are 27 different ways to arrange the accommodation -/
theorem accommodation_arrangements_count :
  accommodationArrangements
    [RoomType.Triple, RoomType.Double, RoomType.Single]
    [Person.Adult, Person.Adult, Person.Adult, Person.Child, Person.Child] = 27 :=
by sorry

end accommodation_arrangements_count_l960_96039


namespace success_arrangements_l960_96062

-- Define the total number of letters
def total_letters : ℕ := 7

-- Define the repetitions of each letter
def s_count : ℕ := 3
def c_count : ℕ := 2
def u_count : ℕ := 1
def e_count : ℕ := 1

-- Define the function to calculate the number of arrangements
def arrangements : ℕ := total_letters.factorial / (s_count.factorial * c_count.factorial * u_count.factorial * e_count.factorial)

-- State the theorem
theorem success_arrangements : arrangements = 420 := by
  sorry

end success_arrangements_l960_96062


namespace altitude_intersection_property_l960_96076

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Checks if a triangle is acute -/
def isAcute (t : Triangle) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Checks if a line is perpendicular to another line -/
def isPerpendicular (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Finds the intersection point of two lines -/
def intersectionPoint (p1 p2 p3 p4 : Point) : Point := sorry

/-- Main theorem -/
theorem altitude_intersection_property (t : Triangle) (P Q H : Point) :
  isAcute t →
  isPerpendicular t.B t.C t.A P →
  isPerpendicular t.A t.C t.B Q →
  H = intersectionPoint t.A P t.B Q →
  distance H P = 4 →
  distance H Q = 3 →
  let B' := intersectionPoint t.A t.C t.B P
  let C' := intersectionPoint t.A t.B t.C P
  let A' := intersectionPoint t.B t.C t.A Q
  let C'' := intersectionPoint t.A t.B t.C Q
  (distance t.B P * distance P C') - (distance t.A Q * distance Q C'') = 7 := by
  sorry

end altitude_intersection_property_l960_96076


namespace sqrt_x_minus_one_real_l960_96055

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) → x ≥ 1 := by
  sorry

end sqrt_x_minus_one_real_l960_96055


namespace min_perimeter_sum_l960_96035

/-- Represents a chessboard configuration -/
structure ChessboardConfig (m : ℕ) where
  size : Fin (2^m) → Fin (2^m) → Bool
  diagonal_unit : ∀ i : Fin (2^m), size i i = true

/-- Calculates the sum of perimeters for a given chessboard configuration -/
def sumPerimeters (m : ℕ) (config : ChessboardConfig m) : ℕ :=
  sorry

/-- Theorem: The minimum sum of perimeters for a 2^m × 2^m chessboard configuration -/
theorem min_perimeter_sum (m : ℕ) : 
  (∃ (config : ChessboardConfig m), 
    ∀ (other_config : ChessboardConfig m), 
      sumPerimeters m config ≤ sumPerimeters m other_config) ∧
  (∃ (config : ChessboardConfig m), sumPerimeters m config = 2^(m+2) * (m+1)) := by
  sorry

end min_perimeter_sum_l960_96035


namespace sin_graph_transformation_l960_96053

theorem sin_graph_transformation :
  ∀ (x y : ℝ),
  (y = Real.sin x) →
  (∃ (x' y' : ℝ),
    x' = 2 * (x + π / 10) ∧
    y' = y ∧
    y' = Real.sin (x' / 2 - π / 10)) :=
by sorry

end sin_graph_transformation_l960_96053


namespace progression_to_floor_pushups_l960_96067

/-- The number of weeks it takes to progress to floor push-ups -/
def weeks_to_floor_pushups (days_per_week : ℕ) (levels_before_floor : ℕ) (days_per_level : ℕ) : ℕ :=
  (levels_before_floor * days_per_level) / days_per_week

/-- Theorem stating that it takes 9 weeks to progress to floor push-ups under given conditions -/
theorem progression_to_floor_pushups :
  weeks_to_floor_pushups 5 3 15 = 9 := by
  sorry

end progression_to_floor_pushups_l960_96067


namespace fixed_internet_charge_is_4_l960_96079

/-- Represents Elvin's monthly telephone bill structure -/
structure MonthlyBill where
  callCharge : ℝ
  internetCharge : ℝ

/-- Calculates the total bill amount -/
def totalBill (bill : MonthlyBill) : ℝ :=
  bill.callCharge + bill.internetCharge

theorem fixed_internet_charge_is_4
  (january : MonthlyBill)
  (february : MonthlyBill)
  (h1 : totalBill january = 40)
  (h2 : totalBill february = 76)
  (h3 : february.callCharge = 2 * january.callCharge)
  (h4 : january.internetCharge = february.internetCharge) :
  january.internetCharge = 4 := by
  sorry

#check fixed_internet_charge_is_4

end fixed_internet_charge_is_4_l960_96079


namespace simplify_sqrt_expression_l960_96074

theorem simplify_sqrt_expression : Real.sqrt (68 - 28 * Real.sqrt 2) = 6 - 4 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_expression_l960_96074


namespace prob_exactly_two_correct_prob_at_least_two_correct_prob_all_incorrect_l960_96008

/-- The number of students and backpacks -/
def n : ℕ := 4

/-- The total number of ways to pick up backpacks -/
def total_outcomes : ℕ := 24

/-- The number of outcomes where exactly two students pick up their correct backpacks -/
def exactly_two_correct : ℕ := 6

/-- The number of outcomes where at least two students pick up their correct backpacks -/
def at_least_two_correct : ℕ := 7

/-- The number of outcomes where all backpacks are picked up incorrectly -/
def all_incorrect : ℕ := 9

/-- The probability of exactly two students picking up the correct backpacks -/
theorem prob_exactly_two_correct : 
  exactly_two_correct / total_outcomes = 1 / 4 := by sorry

/-- The probability of at least two students picking up the correct backpacks -/
theorem prob_at_least_two_correct : 
  at_least_two_correct / total_outcomes = 7 / 24 := by sorry

/-- The probability of all backpacks being picked up incorrectly -/
theorem prob_all_incorrect : 
  all_incorrect / total_outcomes = 3 / 8 := by sorry

end prob_exactly_two_correct_prob_at_least_two_correct_prob_all_incorrect_l960_96008


namespace circle_line_intersection_l960_96031

/-- Given a circle (x-a)^2 + y^2 = 4 and a line x - y = 2, 
    if the chord length intercepted by the circle on the line is 2√2, 
    then a = 0 or a = 4 -/
theorem circle_line_intersection (a : ℝ) : 
  (∃ x y : ℝ, (x - a)^2 + y^2 = 4 ∧ x - y = 2) →  -- circle and line intersect
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ - a)^2 + y₁^2 = 4 ∧ x₁ - y₁ = 2 ∧  -- first intersection point
    (x₂ - a)^2 + y₂^2 = 4 ∧ x₂ - y₂ = 2 ∧  -- second intersection point
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8) →      -- chord length is 2√2
  a = 0 ∨ a = 4 := by
sorry

end circle_line_intersection_l960_96031


namespace expected_value_Y_l960_96099

-- Define a random variable X
variable (X : ℝ → ℝ)

-- Define Y as a function of X
def Y (X : ℝ → ℝ) : ℝ → ℝ := λ ω => 2 * (X ω) + 7

-- Define the expectation operator M
def M (Z : ℝ → ℝ) : ℝ := sorry

-- State the theorem
theorem expected_value_Y (hX : M X = 4) : M (Y X) = 15 := by
  sorry

end expected_value_Y_l960_96099


namespace defective_bulb_probability_l960_96042

/-- The probability of randomly picking a defective bulb from a box with a given pass rate -/
theorem defective_bulb_probability (pass_rate : ℝ) (h : pass_rate = 0.875) :
  1 - pass_rate = 0.125 := by
  sorry

#check defective_bulb_probability

end defective_bulb_probability_l960_96042


namespace select_two_with_boy_l960_96071

/-- The number of ways to select 2 people from 4 boys and 2 girls, with at least one boy -/
def select_with_boy (total : ℕ) (boys : ℕ) (girls : ℕ) (to_select : ℕ) : ℕ :=
  Nat.choose total to_select - Nat.choose girls to_select

theorem select_two_with_boy :
  select_with_boy 6 4 2 2 = 14 :=
by sorry

end select_two_with_boy_l960_96071


namespace greatest_n_squared_l960_96013

theorem greatest_n_squared (n : ℤ) (V : ℝ) : 
  (∀ m : ℤ, 102 * m^2 ≤ V → m ≤ 8) →
  (102 * 8^2 ≤ V) →
  V = 6528 := by
  sorry

end greatest_n_squared_l960_96013


namespace hcf_of_48_and_99_l960_96093

theorem hcf_of_48_and_99 : Nat.gcd 48 99 = 3 := by
  sorry

end hcf_of_48_and_99_l960_96093


namespace cassidy_poster_count_l960_96087

/-- The number of posters Cassidy had two years ago -/
def posters_two_years_ago : ℕ := 14

/-- The number of posters Cassidy will add this summer -/
def posters_to_add : ℕ := 6

/-- Cassidy's current number of posters -/
def current_posters : ℕ := 22

theorem cassidy_poster_count :
  current_posters + posters_to_add = 2 * posters_two_years_ago :=
by sorry

end cassidy_poster_count_l960_96087


namespace fraction_to_decimal_l960_96028

theorem fraction_to_decimal : (7 : ℚ) / 16 = (4375 : ℚ) / 10000 := by sorry

end fraction_to_decimal_l960_96028


namespace pizza_combinations_l960_96078

theorem pizza_combinations (n_toppings : ℕ) (n_crusts : ℕ) (k_toppings : ℕ) : 
  n_toppings = 8 → n_crusts = 2 → k_toppings = 5 → 
  (Nat.choose n_toppings k_toppings) * n_crusts = 112 := by
  sorry

end pizza_combinations_l960_96078


namespace john_good_games_l960_96091

/-- The number of good games John ended up with -/
def good_games (games_from_friend : ℕ) (games_from_garage_sale : ℕ) (broken_games : ℕ) : ℕ :=
  games_from_friend + games_from_garage_sale - broken_games

/-- Theorem stating that John ended up with 6 good games -/
theorem john_good_games :
  good_games 21 8 23 = 6 := by
  sorry

end john_good_games_l960_96091


namespace smallest_c_for_inequality_l960_96097

theorem smallest_c_for_inequality : ∃ c : ℕ, (∀ k : ℕ, 27^k > 3^24 → c ≤ k) ∧ 27^c > 3^24 := by
  sorry

end smallest_c_for_inequality_l960_96097


namespace initial_amount_proof_l960_96083

theorem initial_amount_proof (P : ℝ) : 
  (P * (1 + 1/8) * (1 + 1/8) = 2025) → P = 1600 := by
  sorry

end initial_amount_proof_l960_96083


namespace janet_pill_count_l960_96026

def pills_per_day_first_two_weeks : ℕ := 2 + 3
def pills_per_day_last_two_weeks : ℕ := 2 + 1
def days_per_week : ℕ := 7
def weeks_in_month : ℕ := 4

theorem janet_pill_count :
  (pills_per_day_first_two_weeks * days_per_week * (weeks_in_month / 2)) +
  (pills_per_day_last_two_weeks * days_per_week * (weeks_in_month / 2)) = 112 :=
by sorry

end janet_pill_count_l960_96026


namespace union_of_A_and_B_range_of_a_l960_96006

-- Define sets A and B
def A (a : ℝ) := {x : ℝ | 0 < a * x - 1 ∧ a * x - 1 ≤ 5}
def B := {x : ℝ | -1/2 < x ∧ x ≤ 2}

-- Part I
theorem union_of_A_and_B (a : ℝ) (h : a = 1) :
  A a ∪ B = {x : ℝ | -1/2 < x ∧ x ≤ 6} := by sorry

-- Part II
theorem range_of_a (a : ℝ) (h1 : A a ∩ B = ∅) (h2 : a > 0) :
  0 < a ∧ a ≤ 1/2 := by sorry

end union_of_A_and_B_range_of_a_l960_96006


namespace man_swimming_speed_l960_96020

/-- The speed of a man in still water given his downstream and upstream swimming times and distances -/
theorem man_swimming_speed
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (upstream_distance : ℝ)
  (upstream_time : ℝ)
  (h1 : downstream_distance = 50)
  (h2 : downstream_time = 4)
  (h3 : upstream_distance = 30)
  (h4 : upstream_time = 6)
  : ∃ (v_m : ℝ), v_m = 8.75 ∧ 
    downstream_distance / downstream_time = v_m + (downstream_distance / downstream_time - v_m) ∧
    upstream_distance / upstream_time = v_m - (downstream_distance / downstream_time - v_m) :=
by
  sorry

#check man_swimming_speed

end man_swimming_speed_l960_96020


namespace p_sufficient_not_necessary_q_l960_96014

theorem p_sufficient_not_necessary_q :
  (∀ x : ℝ, 0 < x ∧ x < 2 → -1 < x ∧ x < 3) ∧
  (∃ x : ℝ, -1 < x ∧ x < 3 ∧ ¬(0 < x ∧ x < 2)) := by
  sorry

end p_sufficient_not_necessary_q_l960_96014


namespace worksheets_graded_l960_96011

theorem worksheets_graded (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (problems_left : ℕ) : 
  total_worksheets = 9 →
  problems_per_worksheet = 4 →
  problems_left = 16 →
  total_worksheets - (problems_left / problems_per_worksheet) = 5 :=
by sorry

end worksheets_graded_l960_96011


namespace nested_subtraction_simplification_l960_96063

theorem nested_subtraction_simplification (x : ℝ) : 1 - (2 - (3 - (4 - (5 - x)))) = 3 - x := by
  sorry

end nested_subtraction_simplification_l960_96063


namespace arcsin_equation_solution_l960_96069

theorem arcsin_equation_solution :
  ∃ x : ℝ, x = 1 ∧ Real.arcsin x + Real.arcsin (x - 1) = Real.arccos (1 - x) :=
by sorry

end arcsin_equation_solution_l960_96069


namespace no_solutions_for_divisor_sum_equation_l960_96051

/-- Sum of positive divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: No integers between 1 and 10000 satisfy f(i) = 1 + 2√i + i -/
theorem no_solutions_for_divisor_sum_equation :
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ 10000 →
    sum_of_divisors i ≠ 1 + 2 * (Int.sqrt i).toNat + i := by
  sorry

end no_solutions_for_divisor_sum_equation_l960_96051


namespace fraction_equals_d_minus_one_l960_96077

theorem fraction_equals_d_minus_one (n d : ℕ) (h : d ∣ n) :
  ∃ k : ℕ, k < n ∧ (k : ℚ) / (n - k : ℚ) = d - 1 := by
  sorry

end fraction_equals_d_minus_one_l960_96077


namespace f_derivative_at_pi_over_4_l960_96088

noncomputable def f (x : ℝ) : ℝ := Real.sin x / (Real.sin x + Real.cos x)

theorem f_derivative_at_pi_over_4 :
  deriv f (π/4) = 1/2 := by sorry

end f_derivative_at_pi_over_4_l960_96088


namespace area_enclosed_by_graph_l960_96044

/-- The area enclosed by the graph of |x| + |3y| = 12 -/
def areaEnclosedByGraph : ℝ := 96

/-- The equation of the graph -/
def graphEquation (x y : ℝ) : Prop := (abs x) + (abs (3 * y)) = 12

theorem area_enclosed_by_graph :
  areaEnclosedByGraph = 96 := by sorry

end area_enclosed_by_graph_l960_96044


namespace y_axis_intersection_x_axis_intersections_l960_96032

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 - 3*x + 2

-- Theorem for y-axis intersection
theorem y_axis_intersection : f 0 = 2 := by sorry

-- Theorem for x-axis intersections
theorem x_axis_intersections :
  (f 2 = 0 ∧ f 1 = 0) ∧ ∀ x : ℝ, f x = 0 → (x = 2 ∨ x = 1) := by sorry

end y_axis_intersection_x_axis_intersections_l960_96032


namespace cyclists_circular_track_l960_96094

/-- Given two cyclists on a circular track starting from the same point in opposite directions
    with speeds of 7 m/s and 8 m/s, meeting at the starting point after 20 seconds,
    the circumference of the track is 300 meters. -/
theorem cyclists_circular_track (speed1 speed2 time : ℝ) (circumference : ℝ) : 
  speed1 = 7 → 
  speed2 = 8 → 
  time = 20 → 
  circumference = (speed1 + speed2) * time → 
  circumference = 300 := by sorry

end cyclists_circular_track_l960_96094


namespace distance_after_three_minutes_l960_96066

/-- The distance between two vehicles after a given time -/
def distance_between_vehicles (v1 v2 : ℝ) (t : ℝ) : ℝ :=
  (v2 - v1) * t

/-- Theorem: The distance between two vehicles moving at 65 km/h and 85 km/h after 3 minutes is 1 km -/
theorem distance_after_three_minutes :
  let v1 : ℝ := 65  -- Speed of the truck in km/h
  let v2 : ℝ := 85  -- Speed of the car in km/h
  let t : ℝ := 3 / 60  -- 3 minutes converted to hours
  distance_between_vehicles v1 v2 t = 1 := by
  sorry


end distance_after_three_minutes_l960_96066


namespace base8_cube_c_is_zero_l960_96029

/-- Represents a number in base 8 of the form 4c3 --/
def base8Number (c : ℕ) : ℕ := 4 * 8^2 + c * 8 + 3

/-- Checks if a number is a perfect cube --/
def isPerfectCube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n

theorem base8_cube_c_is_zero :
  ∃ c : ℕ, isPerfectCube (base8Number c) → c = 0 :=
sorry

end base8_cube_c_is_zero_l960_96029


namespace only_D_correct_l960_96009

/-- Represents the contestants in the singing competition -/
inductive Contestant : Type
  | one | two | three | four | five | six

/-- Represents the students who made guesses -/
inductive Student : Type
  | A | B | C | D

/-- The guess made by each student -/
def studentGuess (s : Student) : Contestant → Prop :=
  match s with
  | Student.A => λ c => c = Contestant.four ∨ c = Contestant.five
  | Student.B => λ c => c ≠ Contestant.three
  | Student.C => λ c => c = Contestant.one ∨ c = Contestant.two ∨ c = Contestant.six
  | Student.D => λ c => c ≠ Contestant.four ∧ c ≠ Contestant.five ∧ c ≠ Contestant.six

/-- The theorem to be proved -/
theorem only_D_correct :
  ∃ (winner : Contestant),
    (∀ (s : Student), s ≠ Student.D → ¬(studentGuess s winner)) ∧
    (studentGuess Student.D winner) :=
  sorry

end only_D_correct_l960_96009


namespace recreation_spending_comparison_l960_96037

theorem recreation_spending_comparison (last_week_wages : ℝ) : 
  let last_week_recreation := 0.20 * last_week_wages
  let this_week_wages := 0.80 * last_week_wages
  let this_week_recreation := 0.40 * this_week_wages
  (this_week_recreation / last_week_recreation) * 100 = 160 := by
sorry

end recreation_spending_comparison_l960_96037


namespace fixed_point_exponential_function_l960_96092

theorem fixed_point_exponential_function :
  ∀ (a : ℝ), a > 0 → ((-2 : ℝ)^((-2 : ℝ) + 2) - 3 = -2) := by
  sorry

end fixed_point_exponential_function_l960_96092


namespace xy_value_l960_96007

theorem xy_value (x y : ℤ) (h : (30 : ℚ) / 2 * (x * y) = 21 * x + 20 * y - 13) : x * y = 6 := by
  sorry

end xy_value_l960_96007


namespace lillys_fish_l960_96060

theorem lillys_fish (rosys_fish : ℕ) (total_fish : ℕ) (h1 : rosys_fish = 14) (h2 : total_fish = 24) :
  total_fish - rosys_fish = 10 := by
sorry

end lillys_fish_l960_96060


namespace merchant_pricing_strategy_l960_96048

theorem merchant_pricing_strategy (list_price : ℝ) (list_price_pos : list_price > 0) :
  let purchase_price := list_price * 0.7
  let marked_price := list_price * 1.25
  let selling_price := marked_price * 0.8
  let profit := selling_price - purchase_price
  profit = selling_price * 0.3 := by sorry

end merchant_pricing_strategy_l960_96048


namespace championship_assignments_l960_96002

theorem championship_assignments (n_students : ℕ) (n_titles : ℕ) :
  n_students = 4 → n_titles = 3 →
  (n_students ^ n_titles : ℕ) = 64 := by
  sorry

end championship_assignments_l960_96002


namespace sum_of_integers_l960_96045

theorem sum_of_integers (x y : ℕ+) (h1 : x.val - y.val = 18) (h2 : x.val * y.val = 98) :
  x.val + y.val = 2 * Real.sqrt 179 := by
  sorry

end sum_of_integers_l960_96045


namespace algebraic_expression_value_l960_96022

theorem algebraic_expression_value (x : ℝ) (h : x * (x + 2) = 2023) :
  2 * (x + 3) * (x - 1) - 2018 = 2022 := by
  sorry

end algebraic_expression_value_l960_96022


namespace sin_alpha_value_l960_96054

theorem sin_alpha_value (α : Real) 
  (h1 : α > -π/2 ∧ α < π/2)
  (h2 : Real.tan α = Real.sin (76 * π / 180) * Real.cos (46 * π / 180) - 
                     Real.cos (76 * π / 180) * Real.sin (46 * π / 180)) : 
  Real.sin α = Real.sqrt 5 / 5 := by
sorry

end sin_alpha_value_l960_96054


namespace solve_eggs_problem_l960_96015

def eggs_problem (total_cost : ℝ) (price_per_egg : ℝ) (remaining_eggs : ℕ) : Prop :=
  let eggs_sold := total_cost / price_per_egg
  let initial_eggs := eggs_sold + remaining_eggs
  initial_eggs = 30

theorem solve_eggs_problem :
  eggs_problem 5 0.20 5 :=
sorry

end solve_eggs_problem_l960_96015


namespace library_book_distribution_l960_96043

/-- The number of ways to distribute books between the library and checked-out status -/
def distribution_count (total : ℕ) (min_in_library : ℕ) (min_checked_out : ℕ) : ℕ :=
  (total - min_in_library - min_checked_out + 1)

/-- Theorem: For 8 identical books with at least 2 in the library and 2 checked out,
    there are 5 different ways to distribute the books -/
theorem library_book_distribution :
  distribution_count 8 2 2 = 5 := by
  sorry

end library_book_distribution_l960_96043


namespace shoe_price_calculation_l960_96072

theorem shoe_price_calculation (initial_price : ℝ) (increase_rate : ℝ) (discount_rate : ℝ) : 
  initial_price = 50 →
  increase_rate = 0.2 →
  discount_rate = 0.15 →
  initial_price * (1 + increase_rate) * (1 - discount_rate) = 51 :=
by sorry

end shoe_price_calculation_l960_96072


namespace correct_registration_sequence_l960_96070

-- Define the registration steps
inductive RegistrationStep
  | collectTicket
  | register
  | takeTests
  | takePhoto

-- Define a type for sequences of registration steps
def RegistrationSequence := List RegistrationStep

-- Define the given sequence of steps
def givenSequence : RegistrationSequence := 
  [RegistrationStep.register, RegistrationStep.takePhoto, 
   RegistrationStep.collectTicket, RegistrationStep.takeTests]

-- Define a function to check if a sequence is correct
def isCorrectSequence (seq : RegistrationSequence) : Prop :=
  seq = givenSequence

-- Theorem stating that the given sequence is correct
theorem correct_registration_sequence :
  isCorrectSequence givenSequence := by
  sorry

end correct_registration_sequence_l960_96070


namespace coffee_shop_sales_l960_96058

/-- Represents the number of lattes sold by the coffee shop. -/
def lattes : ℕ := sorry

/-- Represents the number of teas sold by the coffee shop. -/
def teas : ℕ := 6

/-- The relationship between lattes and teas sold. -/
axiom latte_tea_relation : lattes = 4 * teas + 8

theorem coffee_shop_sales : lattes = 32 := by sorry

end coffee_shop_sales_l960_96058


namespace no_solution_for_pair_C_solutions_for_other_pairs_roots_of_original_equation_l960_96025

theorem no_solution_for_pair_C (x y : ℝ) : ¬(y = x ∧ y = x + 1) := by sorry

theorem solutions_for_other_pairs :
  (∃ x y : ℝ, y = x^2 ∧ y = 5*x - 6 ∧ (x = 2 ∨ x = 3)) ∧
  (∃ x : ℝ, x^2 - 5*x + 6 = 0 ∧ (x = 2 ∨ x = 3)) ∧
  (∃ x y : ℝ, y = x^2 - 5*x + 7 ∧ y = 1 ∧ (x = 2 ∨ x = 3)) ∧
  (∃ x y : ℝ, y = x^2 - 1 ∧ y = 5*x - 7 ∧ (x = 2 ∨ x = 3)) := by sorry

theorem roots_of_original_equation (x : ℝ) : x^2 - 5*x + 6 = 0 ↔ (x = 2 ∨ x = 3) := by sorry

end no_solution_for_pair_C_solutions_for_other_pairs_roots_of_original_equation_l960_96025


namespace percent_relation_l960_96057

theorem percent_relation (x y z P : ℝ) 
  (hy : y = 0.75 * x) 
  (hz : z = 0.65 * x) 
  (hP : P / 100 * z = 0.39 * y) : 
  P = 45 := by
sorry

end percent_relation_l960_96057


namespace binary_rep_of_23_l960_96082

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec go (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
    go n

/-- Theorem: The binary representation of 23 is [true, true, true, false, true] -/
theorem binary_rep_of_23 : toBinary 23 = [true, true, true, false, true] := by
  sorry

end binary_rep_of_23_l960_96082


namespace exists_distinct_power_sum_l960_96064

/-- Represents a sum of distinct powers of 3, 4, and 7 -/
structure DistinctPowerSum where
  powers_of_3 : List Nat
  powers_of_4 : List Nat
  powers_of_7 : List Nat
  distinct : powers_of_3.Nodup ∧ powers_of_4.Nodup ∧ powers_of_7.Nodup

/-- Calculates the sum of the powers in a DistinctPowerSum -/
def sumPowers (dps : DistinctPowerSum) : Nat :=
  (dps.powers_of_3.map (fun x => 3^x)).sum +
  (dps.powers_of_4.map (fun x => 4^x)).sum +
  (dps.powers_of_7.map (fun x => 7^x)).sum

/-- Theorem: Every positive integer can be represented as a sum of distinct powers of 3, 4, and 7 -/
theorem exists_distinct_power_sum (n : Nat) (h : n > 0) :
  ∃ (dps : DistinctPowerSum), sumPowers dps = n := by
  sorry

end exists_distinct_power_sum_l960_96064


namespace log_equation_solution_l960_96080

theorem log_equation_solution (a b c x : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.log x = Real.log a + 3 * Real.log b - 5 * Real.log c →
  x = a * b^3 / c^5 := by
  sorry

end log_equation_solution_l960_96080


namespace spinner_final_direction_l960_96030

-- Define the direction type
inductive Direction
  | North
  | East
  | South
  | West

-- Define the rotation type
inductive Rotation
  | Clockwise
  | Counterclockwise

-- Define a function to calculate the final direction after a rotation
def rotateSpinner (initialDir : Direction) (rotation : Rotation) (revolutions : ℚ) : Direction :=
  sorry

-- Theorem statement
theorem spinner_final_direction :
  let initialDir := Direction.North
  let clockwiseRot := 7/2
  let counterclockwiseRot := 21/4
  let finalDir := rotateSpinner (rotateSpinner initialDir Rotation.Clockwise clockwiseRot) Rotation.Counterclockwise counterclockwiseRot
  finalDir = Direction.East := by sorry

end spinner_final_direction_l960_96030


namespace isabel_initial_candy_l960_96010

/- Given conditions -/
def initial_candy : ℕ → Prop := λ x => True  -- Initial amount of candy (unknown)
def friend_gave : ℕ := 25                    -- Amount of candy given by friend
def total_candy : ℕ := 93                    -- Total amount of candy after receiving from friend

/- Theorem to prove -/
theorem isabel_initial_candy :
  ∃ x : ℕ, initial_candy x ∧ x + friend_gave = total_candy ∧ x = 68 :=
by sorry

end isabel_initial_candy_l960_96010


namespace total_birds_is_148_l960_96018

/-- The number of birds seen on Monday -/
def monday_birds : ℕ := 70

/-- The number of birds seen on Tuesday -/
def tuesday_birds : ℕ := monday_birds / 2

/-- The number of birds seen on Wednesday -/
def wednesday_birds : ℕ := tuesday_birds + 8

/-- The total number of birds seen from Monday to Wednesday -/
def total_birds : ℕ := monday_birds + tuesday_birds + wednesday_birds

/-- Theorem stating that the total number of birds seen is 148 -/
theorem total_birds_is_148 : total_birds = 148 := by
  sorry

end total_birds_is_148_l960_96018


namespace f_of_3_eq_neg_1_l960_96019

-- Define the function f
def f (x : ℝ) : ℝ := 
  let t := 2 * x + 1
  x^2 - 2*x

-- Theorem statement
theorem f_of_3_eq_neg_1 : f 1 = -1 := by
  sorry

end f_of_3_eq_neg_1_l960_96019


namespace odd_root_symmetry_l960_96027

theorem odd_root_symmetry (x : ℝ) (n : ℕ) : 
  (x ^ (1 / (2 * n + 1 : ℝ))) = -((-x) ^ (1 / (2 * n + 1 : ℝ))) := by
  sorry

end odd_root_symmetry_l960_96027


namespace ordering_of_a_b_c_l960_96024

theorem ordering_of_a_b_c :
  let a : ℝ := (2 : ℝ) ^ (4/3)
  let b : ℝ := (4 : ℝ) ^ (2/5)
  let c : ℝ := (5 : ℝ) ^ (2/3)
  c > a ∧ a > b := by sorry

end ordering_of_a_b_c_l960_96024


namespace proposition_and_variants_true_l960_96004

theorem proposition_and_variants_true :
  (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0) ∧
  (∀ x y : ℝ, x = 0 ∧ y = 0 → x^2 + y^2 = 0) ∧
  (∀ x y : ℝ, ¬(x = 0 ∧ y = 0) → x^2 + y^2 ≠ 0) ∧
  (∀ x y : ℝ, x^2 + y^2 ≠ 0 → ¬(x = 0 ∧ y = 0)) :=
by sorry

end proposition_and_variants_true_l960_96004


namespace f_has_unique_zero_in_interval_l960_96016

/-- The function f(x) = -x³ + x² + x - 2 -/
def f (x : ℝ) := -x^3 + x^2 + x - 2

/-- The theorem stating that f has exactly one zero in (-∞, -1/3) -/
theorem f_has_unique_zero_in_interval :
  ∃! x, x < -1/3 ∧ f x = 0 :=
sorry

end f_has_unique_zero_in_interval_l960_96016


namespace correct_stratified_sample_l960_96047

/-- Represents the number of people in each age group -/
structure Population :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- Calculates the total population -/
def totalPopulation (p : Population) : ℕ :=
  p.elderly + p.middleAged + p.young

/-- Calculates the number of people to be sampled from each stratum -/
def stratifiedSample (p : Population) (sampleSize : ℕ) : Population :=
  { elderly := (p.elderly * sampleSize) / totalPopulation p,
    middleAged := (p.middleAged * sampleSize) / totalPopulation p,
    young := (p.young * sampleSize) / totalPopulation p }

/-- The theorem to be proved -/
theorem correct_stratified_sample :
  let p : Population := { elderly := 27, middleAged := 54, young := 81 }
  let sample := stratifiedSample p 36
  sample.elderly = 6 ∧ sample.middleAged = 12 ∧ sample.young = 18 := by
  sorry


end correct_stratified_sample_l960_96047


namespace max_value_of_function_l960_96056

theorem max_value_of_function (x : ℝ) : 
  1 / (x^2 + x + 1) ≤ 4/3 ∧ ∃ y : ℝ, 1 / (y^2 + y + 1) = 4/3 :=
by sorry

end max_value_of_function_l960_96056


namespace max_basketballs_part1_max_basketballs_part2_l960_96084

/-- Represents the prices and quantities of basketballs and soccer balls -/
structure BallPurchase where
  basketball_price : ℕ
  soccer_ball_price : ℕ
  basketball_quantity : ℕ
  soccer_ball_quantity : ℕ

/-- Calculates the total cost of the purchase -/
def total_cost (purchase : BallPurchase) : ℕ :=
  purchase.basketball_price * purchase.basketball_quantity +
  purchase.soccer_ball_price * purchase.soccer_ball_quantity

/-- Calculates the total quantity of balls purchased -/
def total_quantity (purchase : BallPurchase) : ℕ :=
  purchase.basketball_quantity + purchase.soccer_ball_quantity

/-- Theorem for part 1 of the problem -/
theorem max_basketballs_part1 (purchase : BallPurchase) 
  (h1 : purchase.basketball_price = 100)
  (h2 : purchase.soccer_ball_price = 80)
  (h3 : total_cost purchase = 5600)
  (h4 : total_quantity purchase = 60) :
  purchase.basketball_quantity = 40 ∧ purchase.soccer_ball_quantity = 20 := by
  sorry

/-- Theorem for part 2 of the problem -/
theorem max_basketballs_part2 (purchase : BallPurchase) 
  (h1 : purchase.basketball_price = 100)
  (h2 : purchase.soccer_ball_price = 80)
  (h3 : total_cost purchase ≤ 6890)
  (h4 : total_quantity purchase = 80) :
  purchase.basketball_quantity ≤ 24 := by
  sorry

end max_basketballs_part1_max_basketballs_part2_l960_96084


namespace system_solvability_l960_96000

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  (x - a)^2 = 4*(y - x + a - 1) ∧
  x ≠ 1 ∧ x > 0 ∧
  (Real.sqrt y - 1) / (Real.sqrt x - 1) = 1

-- Define the solution set for a
def solution_set (a : ℝ) : Prop :=
  a > 1 ∧ a ≠ 5

-- Theorem statement
theorem system_solvability (a : ℝ) :
  (∃ x y, system x y a) ↔ solution_set a :=
sorry

end system_solvability_l960_96000


namespace no_solution_of_double_composition_l960_96049

theorem no_solution_of_double_composition
  (P : ℝ → ℝ)
  (h_continuous : Continuous P)
  (h_no_solution : ∀ x : ℝ, P x ≠ x) :
  ∀ x : ℝ, P (P x) ≠ x :=
by sorry

end no_solution_of_double_composition_l960_96049


namespace ali_fish_weight_l960_96012

/-- Proves that Ali caught 12 kg of fish given the conditions of the fishing problem -/
theorem ali_fish_weight (peter_weight : ℝ) 
  (h1 : peter_weight + 2 * peter_weight + (peter_weight + 1) = 25) : 
  2 * peter_weight = 12 := by
  sorry

end ali_fish_weight_l960_96012


namespace amusement_park_spending_l960_96085

theorem amusement_park_spending (admission_cost food_cost total_cost : ℕ) : 
  food_cost = admission_cost - 13 →
  total_cost = admission_cost + food_cost →
  total_cost = 77 →
  admission_cost = 45 := by
sorry

end amusement_park_spending_l960_96085


namespace hyperbola_eccentricity_range_l960_96033

theorem hyperbola_eccentricity_range (a : ℝ) (h : a > 1) :
  let e := Real.sqrt (1 + 1 / a^2)
  1 < e ∧ e < Real.sqrt 2 := by sorry

end hyperbola_eccentricity_range_l960_96033


namespace abigail_cans_collected_l960_96036

/-- Given:
  - The total number of cans needed is 100
  - Alyssa has collected 30 cans
  - They still need to collect 27 more cans
  Prove that Abigail has collected 43 cans -/
theorem abigail_cans_collected 
  (total_cans : ℕ) 
  (alyssa_cans : ℕ) 
  (more_cans_needed : ℕ) 
  (h1 : total_cans = 100)
  (h2 : alyssa_cans = 30)
  (h3 : more_cans_needed = 27) :
  total_cans - (alyssa_cans + more_cans_needed) = 43 := by
  sorry

end abigail_cans_collected_l960_96036
