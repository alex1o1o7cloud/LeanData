import Mathlib

namespace ab_value_l379_37981

theorem ab_value (a b c : ℝ) 
  (eq1 : a - b = 5)
  (eq2 : a^2 + b^2 = 34)
  (eq3 : a^3 - b^3 = 30)
  (eq4 : a^2 + b^2 - c^2 = 50) : 
  a * b = 4.5 := by
sorry

end ab_value_l379_37981


namespace max_divisible_integers_l379_37951

theorem max_divisible_integers (n : ℕ) : ℕ := by
  -- Let S be the set of 2n consecutive integers
  -- Let D be the set of divisors {n+1, n+2, ..., 2n}
  -- max_divisible is the maximum number of integers in S divisible by at least one number in D
  -- We want to prove that max_divisible = n + ⌊n/2⌋
  sorry

#check max_divisible_integers

end max_divisible_integers_l379_37951


namespace value_of_Y_l379_37988

theorem value_of_Y : ∀ P Q Y : ℚ,
  P = 6036 / 2 →
  Q = P / 4 →
  Y = P - 3 * Q →
  Y = 754.5 := by
sorry

end value_of_Y_l379_37988


namespace triple_equality_l379_37990

theorem triple_equality (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h1 : x * y * (x + y) = y * z * (y + z)) 
  (h2 : y * z * (y + z) = z * x * (z + x)) : 
  (x = y ∧ y = z) ∨ x + y + z = 0 := by
sorry

end triple_equality_l379_37990


namespace largest_common_divisor_408_340_l379_37919

theorem largest_common_divisor_408_340 : ∃ (n : ℕ), n = 68 ∧ 
  n ∣ 408 ∧ n ∣ 340 ∧ ∀ (m : ℕ), m ∣ 408 ∧ m ∣ 340 → m ≤ n :=
by sorry

end largest_common_divisor_408_340_l379_37919


namespace fourth_power_sum_l379_37966

theorem fourth_power_sum (a b c : ℝ) 
  (sum_condition : a + b + c = 3)
  (sum_squares : a^2 + b^2 + c^2 = 5)
  (sum_cubes : a^3 + b^3 + c^3 = 15) :
  a^4 + b^4 + c^4 = 15 := by
  sorry

end fourth_power_sum_l379_37966


namespace total_balls_bought_l379_37989

/-- Represents the total amount of money Mr. Li had --/
def total_money : ℚ := 1

/-- The cost of a plastic ball --/
def plastic_ball_cost : ℚ := 1 / 60

/-- The cost of a glass ball --/
def glass_ball_cost : ℚ := 1 / 36

/-- The cost of a wooden ball --/
def wooden_ball_cost : ℚ := 1 / 45

/-- The number of plastic balls Mr. Li bought --/
def plastic_balls_bought : ℕ := 10

/-- The number of glass balls Mr. Li bought --/
def glass_balls_bought : ℕ := 10

theorem total_balls_bought : ℕ := by
  -- The total number of balls Mr. Li bought is 45
  sorry

end total_balls_bought_l379_37989


namespace cherry_pie_count_l379_37909

theorem cherry_pie_count (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) 
  (h1 : total_pies = 36)
  (h2 : apple_ratio = 2)
  (h3 : blueberry_ratio = 5)
  (h4 : cherry_ratio = 4) :
  (cherry_ratio : ℚ) * total_pies / (apple_ratio + blueberry_ratio + cherry_ratio) = 144 / 11 := by
  sorry

end cherry_pie_count_l379_37909


namespace mn_m_plus_n_is_even_l379_37944

theorem mn_m_plus_n_is_even (m n : ℤ) : 2 ∣ (m * n * (m + n)) := by
  sorry

end mn_m_plus_n_is_even_l379_37944


namespace exists_isosceles_right_triangle_same_color_l379_37999

/-- A color type with three possible values -/
inductive Color
  | Red
  | Green
  | Blue

/-- A point in the 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A coloring function that assigns a color to each point in the grid -/
def ColoringFunction := GridPoint → Color

/-- An isosceles right triangle in the grid -/
structure IsoscelesRightTriangle where
  a : GridPoint
  b : GridPoint
  c : GridPoint
  is_isosceles : (a.x - b.x)^2 + (a.y - b.y)^2 = (a.x - c.x)^2 + (a.y - c.y)^2
  is_right : (b.x - c.x) * (a.x - c.x) + (b.y - c.y) * (a.y - c.y) = 0

/-- The main theorem: There exists an isosceles right triangle with vertices of the same color -/
theorem exists_isosceles_right_triangle_same_color (coloring : ColoringFunction) :
  ∃ (t : IsoscelesRightTriangle), coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c :=
sorry

end exists_isosceles_right_triangle_same_color_l379_37999


namespace black_shirts_per_pack_l379_37915

/-- Given:
  * 3 packs of black shirts and 3 packs of yellow shirts were bought
  * Yellow shirts come in packs of 2
  * Total number of shirts is 21
Prove that the number of black shirts in each pack is 5 -/
theorem black_shirts_per_pack (black_packs yellow_packs : ℕ) 
  (yellow_per_pack total_shirts : ℕ) (black_per_pack : ℕ) :
  black_packs = 3 →
  yellow_packs = 3 →
  yellow_per_pack = 2 →
  total_shirts = 21 →
  black_packs * black_per_pack + yellow_packs * yellow_per_pack = total_shirts →
  black_per_pack = 5 := by
  sorry

#check black_shirts_per_pack

end black_shirts_per_pack_l379_37915


namespace cos_240_deg_l379_37945

/-- Cosine of 240 degrees is equal to -1/2 -/
theorem cos_240_deg : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end cos_240_deg_l379_37945


namespace systematic_sampling_result_l379_37971

def systematic_sampling (population : ℕ) (sample_size : ℕ) (first_drawn : ℕ) (range_start : ℕ) (range_end : ℕ) : ℕ := 
  let interval := population / sample_size
  let sequence := fun n => first_drawn + (n - 1) * interval
  let n := (range_start - first_drawn + interval - 1) / interval
  sequence n

theorem systematic_sampling_result :
  systematic_sampling 960 32 9 401 430 = 429 := by
  sorry

end systematic_sampling_result_l379_37971


namespace total_routes_is_seven_l379_37918

/-- The number of routes from A to C -/
def total_routes (highways_AB : ℕ) (paths_BC : ℕ) (direct_waterway : ℕ) : ℕ :=
  highways_AB * paths_BC + direct_waterway

/-- Theorem: Given the specified number of routes, the total number of routes from A to C is 7 -/
theorem total_routes_is_seven :
  total_routes 2 3 1 = 7 := by
  sorry

end total_routes_is_seven_l379_37918


namespace greatest_number_l379_37969

theorem greatest_number : ∀ (a b c : ℝ), 
  a = 43.23 ∧ b = 2/5 ∧ c = 21.23 →
  a > b ∧ a > c :=
by sorry

end greatest_number_l379_37969


namespace gary_money_calculation_l379_37997

/-- Calculates Gary's final amount of money after a series of transactions -/
def gary_final_amount (initial_amount snake_sale_price hamster_cost supplies_cost : ℝ) : ℝ :=
  initial_amount + snake_sale_price - hamster_cost - supplies_cost

/-- Theorem stating that Gary's final amount is 90.60 dollars -/
theorem gary_money_calculation :
  gary_final_amount 73.25 55.50 25.75 12.40 = 90.60 := by
  sorry

end gary_money_calculation_l379_37997


namespace fractional_equation_simplification_l379_37995

theorem fractional_equation_simplification (x : ℝ) (h : x ≠ 2) : 
  (x / (x - 2) - 2 = 3 / (2 - x)) ↔ (x - 2 * (x - 2) = -3) :=
sorry

end fractional_equation_simplification_l379_37995


namespace paint_cost_per_kg_l379_37923

/-- The cost of paint per kg given specific conditions -/
theorem paint_cost_per_kg (coverage : Real) (total_cost : Real) (side_length : Real) :
  coverage = 15 →
  total_cost = 200 →
  side_length = 5 →
  (total_cost / (6 * side_length^2 / coverage)) = 20 := by
  sorry

end paint_cost_per_kg_l379_37923


namespace derek_age_is_20_l379_37943

-- Define the ages as natural numbers
def aunt_beatrice_age : ℕ := 54

-- Define Emily's age in terms of Aunt Beatrice's age
def emily_age : ℕ := aunt_beatrice_age / 2

-- Define Derek's age in terms of Emily's age
def derek_age : ℕ := emily_age - 7

-- Theorem statement
theorem derek_age_is_20 : derek_age = 20 := by
  sorry

end derek_age_is_20_l379_37943


namespace book_pages_digits_l379_37931

/-- Given a book with n pages, calculate the total number of digits used to number all pages. -/
def totalDigits (n : ℕ) : ℕ :=
  let singleDigits := min n 9
  let doubleDigits := max (min n 99 - 9) 0
  let tripleDigits := max (n - 99) 0
  singleDigits + 2 * doubleDigits + 3 * tripleDigits

/-- Theorem stating that a book with 360 pages requires exactly 972 digits to number all its pages. -/
theorem book_pages_digits : totalDigits 360 = 972 := by
  sorry

end book_pages_digits_l379_37931


namespace function_q_polynomial_l379_37961

/-- Given a function q(x) satisfying the equation
    q(x) + (2x^6 + 4x^4 + 5x^2 + 7) = (3x^4 + 18x^3 + 15x^2 + 8x + 3),
    prove that q(x) = -2x^6 - x^4 + 18x^3 + 10x^2 + 8x - 4 -/
theorem function_q_polynomial (q : ℝ → ℝ) :
  (∀ x, q x + (2*x^6 + 4*x^4 + 5*x^2 + 7) = (3*x^4 + 18*x^3 + 15*x^2 + 8*x + 3)) →
  (∀ x, q x = -2*x^6 - x^4 + 18*x^3 + 10*x^2 + 8*x - 4) := by
  sorry

end function_q_polynomial_l379_37961


namespace next_simultaneous_activation_l379_37920

/-- Represents the time interval in minutes for each location's signal -/
structure SignalIntervals :=
  (fire : ℕ)
  (police : ℕ)
  (hospital : ℕ)

/-- Calculates the time in minutes until the next simultaneous activation -/
def timeUntilNextSimultaneous (intervals : SignalIntervals) : ℕ :=
  Nat.lcm (Nat.lcm intervals.fire intervals.police) intervals.hospital

/-- Theorem stating that for the given intervals, the next simultaneous activation occurs after 180 minutes -/
theorem next_simultaneous_activation (intervals : SignalIntervals)
  (h1 : intervals.fire = 12)
  (h2 : intervals.police = 18)
  (h3 : intervals.hospital = 30) :
  timeUntilNextSimultaneous intervals = 180 := by
  sorry

#eval timeUntilNextSimultaneous ⟨12, 18, 30⟩

end next_simultaneous_activation_l379_37920


namespace ribbon_cost_comparison_l379_37928

/-- Represents the cost and quantity of ribbons --/
structure RibbonPurchase where
  cost : ℕ
  quantity : ℕ

/-- Determines if one ribbon is cheaper than another --/
def isCheaper (r1 r2 : RibbonPurchase) : Prop :=
  r1.cost * r2.quantity < r2.cost * r1.quantity

theorem ribbon_cost_comparison 
  (yellow blue : RibbonPurchase)
  (h_yellow : yellow.cost = 24)
  (h_blue : blue.cost = 36) :
  (∃ y b, isCheaper {cost := 24, quantity := y} {cost := 36, quantity := b}) ∧
  (∃ y b, isCheaper {cost := 36, quantity := b} {cost := 24, quantity := y}) ∧
  (∃ y b, yellow.cost * b = blue.cost * y) :=
sorry

end ribbon_cost_comparison_l379_37928


namespace bus_speed_is_40_l379_37986

/-- Represents the scenario of a bus and cyclist traveling between points A, B, C, and D. -/
structure TravelScenario where
  distance_AB : ℝ
  time_to_C : ℝ
  distance_CD : ℝ
  bus_speed : ℝ
  cyclist_speed : ℝ

/-- The travel scenario satisfies the given conditions. -/
def satisfies_conditions (s : TravelScenario) : Prop :=
  s.distance_AB = 4 ∧
  s.time_to_C = 1/6 ∧
  s.distance_CD = 2/3 ∧
  s.bus_speed > 0 ∧
  s.cyclist_speed > 0 ∧
  s.bus_speed > s.cyclist_speed

/-- The theorem stating that under the given conditions, the bus speed is 40 km/h. -/
theorem bus_speed_is_40 (s : TravelScenario) (h : satisfies_conditions s) : 
  s.bus_speed = 40 := by
  sorry

#check bus_speed_is_40

end bus_speed_is_40_l379_37986


namespace min_sum_factors_l379_37958

def S (n : ℕ) : ℕ := (3 + 7 + 13 + (2*n + 2*n - 1))

theorem min_sum_factors (a b c : ℕ+) (h : S 10 = a * b * c) :
  ∃ (x y z : ℕ+), S 10 = x * y * z ∧ x + y + z ≤ a + b + c ∧ x + y + z = 68 :=
sorry

end min_sum_factors_l379_37958


namespace cousin_calls_l379_37959

/-- Represents the number of days in a leap year -/
def leapYearDays : ℕ := 366

/-- Represents the calling frequencies of the four cousins -/
def callingFrequencies : List ℕ := [2, 3, 4, 6]

/-- Calculates the number of days with at least one call in a leap year -/
def daysWithCalls (frequencies : List ℕ) (totalDays : ℕ) : ℕ :=
  sorry

theorem cousin_calls :
  daysWithCalls callingFrequencies leapYearDays = 244 :=
sorry

end cousin_calls_l379_37959


namespace max_value_fraction_sum_l379_37984

theorem max_value_fraction_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (2 * x + y)) + (y / (x + 2 * y)) ≤ 2 / 3 := by
  sorry

end max_value_fraction_sum_l379_37984


namespace complex_magnitude_problem_l379_37956

theorem complex_magnitude_problem (z : ℂ) (h : (1 - Complex.I) * z = 1 + Complex.I) : 
  Complex.abs z = 1 := by
  sorry

end complex_magnitude_problem_l379_37956


namespace polynomial_remainder_l379_37947

def f (x : ℝ) : ℝ := x^5 + 2*x^3 + x^2 + 4

theorem polynomial_remainder : 
  ∃ (q : ℝ → ℝ), f = λ x => (x - 2) * q x + 56 := by
  sorry

end polynomial_remainder_l379_37947


namespace greatest_number_neither_swimming_nor_soccer_l379_37942

theorem greatest_number_neither_swimming_nor_soccer 
  (total_students : ℕ) 
  (swimming_fans : ℕ) 
  (soccer_fans : ℕ) 
  (h1 : total_students = 1460) 
  (h2 : swimming_fans = 33) 
  (h3 : soccer_fans = 36) : 
  ∃ (neither_fans : ℕ), 
    neither_fans ≤ total_students - (swimming_fans + soccer_fans) ∧ 
    neither_fans = 1391 :=
sorry

end greatest_number_neither_swimming_nor_soccer_l379_37942


namespace second_equation_result_l379_37949

theorem second_equation_result (x y : ℤ) 
  (eq1 : 3 * x + y = 40) 
  (eq2 : 3 * y^2 = 48) : 
  2 * x - y = 20 := by
sorry

end second_equation_result_l379_37949


namespace average_percent_change_population_l379_37948

-- Define the initial and final population
def initial_population : ℕ := 175000
def final_population : ℕ := 297500

-- Define the time period in years
def years : ℕ := 10

-- Define the theorem
theorem average_percent_change_population (initial_pop : ℕ) (final_pop : ℕ) (time : ℕ) :
  initial_pop = initial_population →
  final_pop = final_population →
  time = years →
  (((final_pop - initial_pop : ℝ) / initial_pop) * 100) / time = 7 :=
by sorry

end average_percent_change_population_l379_37948


namespace min_buttons_for_adjacency_l379_37907

/-- Represents a color of a button -/
inductive Color
| A | B | C | D | E | F

/-- Represents a sequence of buttons -/
def ButtonSequence := List Color

/-- Checks if two colors are adjacent in a button sequence -/
def areColorsAdjacent (seq : ButtonSequence) (c1 c2 : Color) : Prop :=
  ∃ i, (seq.get? i = some c1 ∧ seq.get? (i+1) = some c2) ∨
       (seq.get? i = some c2 ∧ seq.get? (i+1) = some c1)

/-- Checks if a button sequence satisfies the adjacency condition for all color pairs -/
def satisfiesCondition (seq : ButtonSequence) : Prop :=
  ∀ c1 c2, c1 ≠ c2 → areColorsAdjacent seq c1 c2

/-- The main theorem stating the minimum number of buttons required -/
theorem min_buttons_for_adjacency :
  ∃ (seq : ButtonSequence),
    seq.length = 18 ∧
    satisfiesCondition seq ∧
    ∀ (seq' : ButtonSequence), satisfiesCondition seq' → seq'.length ≥ 18 :=
sorry

end min_buttons_for_adjacency_l379_37907


namespace first_day_is_sunday_l379_37908

/-- Enumeration of days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to get the day of week after a given number of days -/
def afterDays (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | Nat.succ m => nextDay (afterDays start m)

/-- Theorem: If the 21st day of a month is a Saturday, then the 1st day of that month is a Sunday -/
theorem first_day_is_sunday (d : DayOfWeek) :
  afterDays d 20 = DayOfWeek.Saturday → d = DayOfWeek.Sunday :=
by
  sorry


end first_day_is_sunday_l379_37908


namespace complex_equation_solution_l379_37974

theorem complex_equation_solution :
  ∃ z : ℂ, (4 : ℂ) - 2 * Complex.I * z = 3 + 5 * Complex.I * z ∧ z = (1 / 7 : ℂ) * Complex.I :=
by sorry

end complex_equation_solution_l379_37974


namespace dodecahedron_interior_diagonals_l379_37913

/-- Represents a dodecahedron -/
structure Dodecahedron where
  vertices : Nat
  edges_per_vertex : Nat

/-- Calculates the number of interior diagonals in a dodecahedron -/
def interior_diagonals (d : Dodecahedron) : Nat :=
  (d.vertices * (d.vertices - 1 - d.edges_per_vertex)) / 2

/-- Theorem stating that a dodecahedron has 160 interior diagonals -/
theorem dodecahedron_interior_diagonals :
  ∃ d : Dodecahedron, d.vertices = 20 ∧ d.edges_per_vertex = 3 ∧ interior_diagonals d = 160 := by
  sorry

end dodecahedron_interior_diagonals_l379_37913


namespace dixon_passing_students_l379_37925

theorem dixon_passing_students (collins_total : ℕ) (collins_passed : ℕ) (dixon_total : ℕ) 
  (h1 : collins_total = 30) 
  (h2 : collins_passed = 18) 
  (h3 : dixon_total = 45) :
  (dixon_total * collins_passed) / collins_total = 27 := by
  sorry

end dixon_passing_students_l379_37925


namespace pet_store_puppies_l379_37976

/-- The number of puppies sold -/
def puppies_sold : ℕ := 3

/-- The number of cages used -/
def cages_used : ℕ := 3

/-- The number of puppies in each cage -/
def puppies_per_cage : ℕ := 5

/-- The initial number of puppies in the pet store -/
def initial_puppies : ℕ := puppies_sold + cages_used * puppies_per_cage

theorem pet_store_puppies : initial_puppies = 18 := by
  sorry

end pet_store_puppies_l379_37976


namespace initial_state_is_losing_l379_37904

/-- Represents a game state with two piles of matches -/
structure GameState :=
  (pile1 : Nat) (pile2 : Nat)

/-- Checks if a move is valid according to the game rules -/
def isValidMove (state : GameState) (move : Nat) (fromPile : Bool) : Prop :=
  if fromPile then
    move > 0 ∧ move ≤ state.pile1 ∧ state.pile2 % move = 0
  else
    move > 0 ∧ move ≤ state.pile2 ∧ state.pile1 % move = 0

/-- Defines a losing position in the game -/
def isLosingPosition (state : GameState) : Prop :=
  ∃ (k m n : Nat),
    state.pile1 = 2^k * (2*m + 1) ∧
    state.pile2 = 2^k * (2*n + 1)

/-- The main theorem stating that the initial position (100, 252) is a losing position -/
theorem initial_state_is_losing :
  isLosingPosition (GameState.mk 100 252) :=
sorry

#check initial_state_is_losing

end initial_state_is_losing_l379_37904


namespace player_field_time_l379_37917

/-- Given a sports tournament with the following conditions:
  * The team has 10 players
  * 8 players are always on the field
  * The match lasts 45 minutes
  * All players must play the same amount of time
This theorem proves that each player will be on the field for 36 minutes. -/
theorem player_field_time 
  (total_players : ℕ) 
  (field_players : ℕ) 
  (match_duration : ℕ) 
  (h1 : total_players = 10)
  (h2 : field_players = 8)
  (h3 : match_duration = 45) :
  (field_players * match_duration) / total_players = 36 := by
  sorry

end player_field_time_l379_37917


namespace solution_set_for_a_neg_one_range_of_a_l379_37932

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |3*x - 1|

-- Define the set M
def M (a : ℝ) : Set ℝ := {x | f a x ≤ |3*x + 1|}

-- Statement for part 1
theorem solution_set_for_a_neg_one :
  {x : ℝ | f (-1) x ≤ 1} = {x : ℝ | 1/4 ≤ x ∧ x ≤ 1/2} :=
sorry

-- Statement for part 2
theorem range_of_a (a : ℝ) :
  (Set.Icc (1/4 : ℝ) 1 ⊆ M a) → -7/3 ≤ a ∧ a ≤ 1 :=
sorry

end solution_set_for_a_neg_one_range_of_a_l379_37932


namespace parabola_intersection_theorem_l379_37967

/-- Represents a parabola of the form y = x^2 + bx + c -/
structure Parabola where
  b : ℝ
  c : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem stating the conditions and conclusion about the parabola -/
theorem parabola_intersection_theorem (p : Parabola) 
  (A B C : Point) :
  (A.x = 0) →  -- A is on y-axis
  (B.x > 0 ∧ C.x > 0) →  -- B and C are on positive x-axis
  (B.y = 0 ∧ C.y = 0) →  -- B and C are on x-axis
  (A.y = p.c) →  -- A is the y-intercept
  (C.x - B.x = 2) →  -- BC = 2
  (1/2 * A.y * (C.x - B.x) = 3) →  -- Area of triangle ABC is 3
  (p.b = -4) := by
  sorry

end parabola_intersection_theorem_l379_37967


namespace batsman_average_after_17th_innings_l379_37972

/-- Represents a batsman's performance over multiple innings -/
structure BatsmanPerformance where
  innings : Nat
  totalScore : Nat
  averageIncrease : Nat
  lastInningsScore : Nat

/-- Calculates the average score of a batsman -/
def calculateAverage (performance : BatsmanPerformance) : Rat :=
  performance.totalScore / performance.innings

theorem batsman_average_after_17th_innings
  (performance : BatsmanPerformance)
  (h1 : performance.innings = 17)
  (h2 : performance.lastInningsScore = 85)
  (h3 : calculateAverage performance - calculateAverage { performance with
    innings := performance.innings - 1
    totalScore := performance.totalScore - performance.lastInningsScore
  } = performance.averageIncrease)
  (h4 : performance.averageIncrease = 3) :
  calculateAverage performance = 37 := by
  sorry

#eval calculateAverage {
  innings := 17,
  totalScore := 17 * 37,
  averageIncrease := 3,
  lastInningsScore := 85
}

end batsman_average_after_17th_innings_l379_37972


namespace all_statements_correct_l379_37911

/-- The volume of a rectangle with sides a and b, considered as a 3D object of unit height -/
def volume (a b : ℝ) : ℝ := a * b

theorem all_statements_correct (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (volume (2 * a) b = 2 * volume a b) ∧
  (volume a (3 * b) = 3 * volume a b) ∧
  (volume (2 * a) (3 * b) = 6 * volume a b) ∧
  (volume (a / 2) (2 * b) = volume a b) ∧
  (volume (3 * a) (b / 2) = (3 / 2) * volume a b) :=
by sorry

end all_statements_correct_l379_37911


namespace oranges_from_third_tree_l379_37970

/-- The number of oranges picked from the third tree -/
def oranges_third_tree (total : ℕ) (first : ℕ) (second : ℕ) : ℕ :=
  total - (first + second)

/-- Theorem stating that the number of oranges picked from the third tree is 120 -/
theorem oranges_from_third_tree :
  oranges_third_tree 260 80 60 = 120 := by
  sorry

end oranges_from_third_tree_l379_37970


namespace edric_work_hours_l379_37979

/-- Calculates the number of hours worked per day given monthly salary, days worked per week, and hourly rate -/
def hours_per_day (monthly_salary : ℕ) (days_per_week : ℕ) (hourly_rate : ℕ) : ℕ :=
  let days_per_month := days_per_week * 4
  let total_hours := monthly_salary / hourly_rate
  total_hours / days_per_month

theorem edric_work_hours :
  hours_per_day 576 6 3 = 8 := by
  sorry

#eval hours_per_day 576 6 3

end edric_work_hours_l379_37979


namespace taxi_speed_is_60_l379_37916

/-- The speed of the taxi in mph -/
def taxi_speed : ℝ := 60

/-- The speed of the bus in mph -/
def bus_speed : ℝ := taxi_speed - 30

/-- The time difference between the bus and taxi departure in hours -/
def time_difference : ℝ := 3

/-- The time it takes for the taxi to overtake the bus in hours -/
def overtake_time : ℝ := 3

theorem taxi_speed_is_60 :
  (taxi_speed * overtake_time = bus_speed * (time_difference + overtake_time)) →
  taxi_speed = 60 := by
  sorry

#check taxi_speed_is_60

end taxi_speed_is_60_l379_37916


namespace intersection_complement_when_m_2_intersection_complement_empty_iff_l379_37994

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B (m : ℝ) : Set ℝ := {x | x ≤ 3*m - 4 ∨ x ≥ 8 + m}

-- Theorem for part 1
theorem intersection_complement_when_m_2 :
  A ∩ (Set.univ \ B 2) = {x | 2 < x ∧ x < 4} := by sorry

-- Theorem for part 2
theorem intersection_complement_empty_iff (m : ℝ) :
  m < 6 →
  (A ∩ (Set.univ \ B m) = ∅ ↔ m ≤ -7 ∨ (8/3 ≤ m ∧ m < 6)) := by sorry

end intersection_complement_when_m_2_intersection_complement_empty_iff_l379_37994


namespace investment_rate_proof_l379_37987

def total_investment : ℝ := 17000
def investment_at_4_percent : ℝ := 12000
def total_interest : ℝ := 1380
def known_rate : ℝ := 0.04

theorem investment_rate_proof :
  let remaining_investment := total_investment - investment_at_4_percent
  let interest_at_4_percent := investment_at_4_percent * known_rate
  let remaining_interest := total_interest - interest_at_4_percent
  let unknown_rate := remaining_interest / remaining_investment
  unknown_rate = 0.18 := by sorry

end investment_rate_proof_l379_37987


namespace congruence_problem_l379_37980

theorem congruence_problem (y : ℤ) 
  (h1 : (4 + y) % (2^4) = 3^2 % (2^4))
  (h2 : (6 + y) % (3^4) = 2^3 % (3^4))
  (h3 : (8 + y) % (5^4) = 7^2 % (5^4)) :
  y % 360 = 317 := by
  sorry

end congruence_problem_l379_37980


namespace certain_number_problem_l379_37975

theorem certain_number_problem (x : ℝ) : 45 * 7 = 0.35 * x → x = 900 := by
  sorry

end certain_number_problem_l379_37975


namespace expected_smallest_seven_from_sixtythree_l379_37935

/-- The expected value of the smallest number when randomly selecting r numbers from a set of n numbers. -/
def expected_smallest (n : ℕ) (r : ℕ) : ℚ :=
  (n + 1 : ℚ) / (r + 1 : ℚ)

/-- The set size -/
def n : ℕ := 63

/-- The sample size -/
def r : ℕ := 7

theorem expected_smallest_seven_from_sixtythree :
  expected_smallest n r = 8 := by
  sorry

end expected_smallest_seven_from_sixtythree_l379_37935


namespace toms_remaining_balloons_l379_37998

/-- Theorem: Tom's remaining violet balloons -/
theorem toms_remaining_balloons (initial_balloons : ℕ) (given_balloons : ℕ) 
  (h1 : initial_balloons = 30)
  (h2 : given_balloons = 16) :
  initial_balloons - given_balloons = 14 := by
  sorry

end toms_remaining_balloons_l379_37998


namespace smallest_side_of_triangle_l379_37940

/-- Given a triangle ABC with sides a, b, and c satisfying b^2 + c^2 ≥ 5a^2, 
    BC is the smallest side of the triangle. -/
theorem smallest_side_of_triangle (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0) 
    (h_condition : b^2 + c^2 ≥ 5*a^2) : 
    c ≤ a ∧ c ≤ b := by
  sorry

end smallest_side_of_triangle_l379_37940


namespace hash_difference_l379_37965

-- Define the # operation
def hash (x y : ℤ) : ℤ := x * y - 3 * x

-- Theorem statement
theorem hash_difference : (hash 7 4) - (hash 4 7) = -9 := by
  sorry

end hash_difference_l379_37965


namespace x_equals_five_l379_37938

theorem x_equals_five (x : ℝ) (h : x - 2 = 3) : x = 5 := by
  sorry

end x_equals_five_l379_37938


namespace gecko_count_l379_37992

theorem gecko_count : 
  ∀ (gecko_count : ℕ) (lizard_count : ℕ) (insects_per_gecko : ℕ) (total_insects : ℕ),
    lizard_count = 3 →
    insects_per_gecko = 6 →
    total_insects = 66 →
    total_insects = gecko_count * insects_per_gecko + lizard_count * (2 * insects_per_gecko) →
    gecko_count = 5 := by
  sorry

end gecko_count_l379_37992


namespace imaginary_difference_condition_l379_37993

def is_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem imaginary_difference_condition (z₁ z₂ : ℂ) :
  (is_imaginary (z₁ - z₂) → (is_imaginary z₁ ∨ is_imaginary z₂)) ∧
  ∃ z₁ z₂ : ℂ, (is_imaginary z₁ ∨ is_imaginary z₂) ∧ ¬is_imaginary (z₁ - z₂) :=
sorry

end imaginary_difference_condition_l379_37993


namespace certain_number_problem_l379_37954

theorem certain_number_problem (a : ℕ) (certain_number : ℕ) 
  (h1 : a = 105)
  (h2 : a^3 = 21 * 25 * 45 * certain_number) :
  certain_number = 49 := by
  sorry

end certain_number_problem_l379_37954


namespace geometric_sum_first_eight_l379_37926

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n - 1)

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  if r = 1 then n * a else a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_eight :
  let a : ℚ := 1/3
  let r : ℚ := 1/3
  let n : ℕ := 8
  geometric_sum a r n = 3280/6561 := by sorry

end geometric_sum_first_eight_l379_37926


namespace complex_magnitude_problem_l379_37906

theorem complex_magnitude_problem (z : ℂ) (h : (1 + Complex.I) * z = 1 - Complex.I) :
  Complex.abs (1 + z) = Real.sqrt 2 := by
  sorry

end complex_magnitude_problem_l379_37906


namespace car_speed_problem_l379_37924

/-- The speed of Car A in miles per hour -/
def speed_A : ℝ := 58

/-- The speed of Car B in miles per hour -/
def speed_B : ℝ := 50

/-- The initial distance between Car A and Car B in miles -/
def initial_distance : ℝ := 16

/-- The final distance between Car A and Car B in miles -/
def final_distance : ℝ := 8

/-- The time taken for Car A to overtake Car B in hours -/
def time : ℝ := 3

theorem car_speed_problem :
  speed_A * time = speed_B * time + initial_distance + final_distance := by
  sorry

end car_speed_problem_l379_37924


namespace range_of_expression_l379_37901

open Real

theorem range_of_expression (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 ≤ β ∧ β ≤ π/2) : 
  -π/6 < 2*α - β/3 ∧ 2*α - β/3 < π := by
sorry

end range_of_expression_l379_37901


namespace derivative_f_at_1_l379_37900

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1)^2 * (x - 1)

-- State the theorem
theorem derivative_f_at_1 : 
  deriv f 1 = 4 := by sorry

end derivative_f_at_1_l379_37900


namespace star_example_l379_37910

def star (x y : ℝ) : ℝ := 5 * x - 2 * y

theorem star_example : (star 3 4) + (star 2 2) = 13 := by
  sorry

end star_example_l379_37910


namespace meal_price_calculation_meal_price_correct_l379_37950

/-- Calculate the entire price of a meal given individual costs, tax rate, and tip rate -/
theorem meal_price_calculation (appetizer : ℚ) (buffy_entree : ℚ) (oz_entree : ℚ) 
  (side1 : ℚ) (side2 : ℚ) (dessert : ℚ) (drink_price : ℚ) 
  (tax_rate : ℚ) (tip_rate : ℚ) : ℚ :=
  let total_before_tax := appetizer + buffy_entree + oz_entree + side1 + side2 + dessert + 2 * drink_price
  let tax := total_before_tax * tax_rate
  let total_with_tax := total_before_tax + tax
  let tip := total_with_tax * tip_rate
  let total_price := total_with_tax + tip
  total_price

/-- The entire price of the meal is $120.66 -/
theorem meal_price_correct : 
  meal_price_calculation 9 20 25 6 8 11 (13/2) (3/40) (11/50) = 12066/100 := by
  sorry

end meal_price_calculation_meal_price_correct_l379_37950


namespace train_speed_problem_l379_37934

/-- Prove that given two trains of equal length 62.5 meters, where the faster train
    travels at 46 km/hr and passes the slower train in 45 seconds, the speed of
    the slower train is 36 km/hr. -/
theorem train_speed_problem (train_length : ℝ) (faster_speed : ℝ) (passing_time : ℝ) :
  train_length = 62.5 →
  faster_speed = 46 →
  passing_time = 45 →
  ∃ (slower_speed : ℝ),
    slower_speed = 36 ∧
    (faster_speed - slower_speed) * (1000 / 3600) * passing_time = 2 * train_length :=
by sorry

end train_speed_problem_l379_37934


namespace households_B_and_C_eq_22_l379_37927

/-- A residential building where each household subscribes to exactly two different newspapers. -/
structure Building where
  /-- The number of subscriptions for newspaper A -/
  subscriptions_A : ℕ
  /-- The number of subscriptions for newspaper B -/
  subscriptions_B : ℕ
  /-- The number of subscriptions for newspaper C -/
  subscriptions_C : ℕ
  /-- The total number of households in the building -/
  total_households : ℕ
  /-- Each household subscribes to exactly two different newspapers -/
  two_subscriptions : subscriptions_A + subscriptions_B + subscriptions_C = 2 * total_households

/-- The number of households subscribing to both newspaper B and C in a given building -/
def households_B_and_C (b : Building) : ℕ :=
  b.total_households - b.subscriptions_A

theorem households_B_and_C_eq_22 (b : Building) 
  (h_A : b.subscriptions_A = 30)
  (h_B : b.subscriptions_B = 34)
  (h_C : b.subscriptions_C = 40) :
  households_B_and_C b = 22 := by
  sorry

#eval households_B_and_C ⟨30, 34, 40, 52, by norm_num⟩

end households_B_and_C_eq_22_l379_37927


namespace triangle_side_length_l379_37982

noncomputable section

/-- Given a triangle ABC with BC = 1, if sin(A/2) * cos(B/2) = sin(B/2) * cos(A/2), then AC = sin(A) / sin(C) -/
theorem triangle_side_length (A B C : Real) (BC : Real) (h1 : BC = 1) 
  (h2 : Real.sin (A / 2) * Real.cos (B / 2) = Real.sin (B / 2) * Real.cos (A / 2)) :
  ∃ (AC : Real), AC = Real.sin A / Real.sin C :=
by sorry

end triangle_side_length_l379_37982


namespace quadratic_one_solution_l379_37968

theorem quadratic_one_solution (k : ℝ) : 
  (k > 0) → (∃! x, 4 * x^2 + k * x + 4 = 0) ↔ k = 8 := by
  sorry

end quadratic_one_solution_l379_37968


namespace select_five_from_eight_l379_37902

theorem select_five_from_eight : Nat.choose 8 5 = 56 := by
  sorry

end select_five_from_eight_l379_37902


namespace remainder_problem_l379_37939

theorem remainder_problem (d : ℕ) (r : ℕ) (h1 : d > 1) 
  (h2 : 1083 % d = r) (h3 : 1455 % d = r) (h4 : 2345 % d = r) : 
  d - r = 1 := by
  sorry

end remainder_problem_l379_37939


namespace probability_two_red_balls_l379_37946

def total_balls : ℕ := 7 + 5 + 4

def red_balls : ℕ := 7

theorem probability_two_red_balls :
  (Nat.choose red_balls 2 : ℚ) / (Nat.choose total_balls 2) = 7 / 40 :=
by sorry

end probability_two_red_balls_l379_37946


namespace chocolate_cake_price_is_12_l379_37977

/-- The price of a chocolate cake given the order details and total payment -/
def chocolate_cake_price (num_chocolate : ℕ) (num_strawberry : ℕ) (strawberry_price : ℕ) (total_payment : ℕ) : ℕ :=
  (total_payment - num_strawberry * strawberry_price) / num_chocolate

theorem chocolate_cake_price_is_12 :
  chocolate_cake_price 3 6 22 168 = 12 := by
  sorry

end chocolate_cake_price_is_12_l379_37977


namespace geometric_sequence_sum_l379_37955

/-- Given a geometric sequence {a_n} with a₁ = 3 and a₁ + a₃ + a₅ = 21, 
    prove that a₃ + a₅ + a₇ = 42 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) : 
  (∀ n, a (n + 1) = a n * q) →  -- Geometric sequence definition
  a 1 = 3 →                     -- First term condition
  a 1 + a 3 + a 5 = 21 →        -- Sum of odd terms condition
  a 3 + a 5 + a 7 = 42 :=
by sorry

end geometric_sequence_sum_l379_37955


namespace differential_at_zero_l379_37973

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 + 3)

theorem differential_at_zero (x : ℝ) : 
  deriv f 0 = 3 := by sorry

end differential_at_zero_l379_37973


namespace queen_high_school_teachers_queen_high_school_teachers_correct_l379_37929

theorem queen_high_school_teachers (num_students : ℕ) (classes_per_student : ℕ) 
  (classes_per_teacher : ℕ) (students_per_class : ℕ) : ℕ :=
  let total_classes := num_students * classes_per_student
  let unique_classes := total_classes / students_per_class
  unique_classes / classes_per_teacher

theorem queen_high_school_teachers_correct : 
  queen_high_school_teachers 1500 6 5 25 = 72 := by
  sorry

end queen_high_school_teachers_queen_high_school_teachers_correct_l379_37929


namespace sqrt_pattern_l379_37983

theorem sqrt_pattern (n : ℕ) (hn : n > 0) : 
  Real.sqrt (1 + 1 / (n^2 : ℝ) + 1 / ((n+1)^2 : ℝ)) = (n^2 + n + 1 : ℝ) / (n * (n+1)) := by
  sorry

end sqrt_pattern_l379_37983


namespace vector_to_point_coordinates_l379_37952

/-- Given a vector AB = (-2, 4), if point A is at the origin (0, 0), 
    then the coordinates of point B are (-2, 4). -/
theorem vector_to_point_coordinates (A B : ℝ × ℝ) : 
  (A.1 - B.1 = 2 ∧ A.2 - B.2 = -4) → 
  (A = (0, 0) → B = (-2, 4)) := by
  sorry

end vector_to_point_coordinates_l379_37952


namespace greatest_integer_inequality_unique_greatest_integer_l379_37964

theorem greatest_integer_inequality (x : ℤ) : (7 : ℚ) / 9 > (x : ℚ) / 15 ↔ x ≤ 11 := by sorry

theorem unique_greatest_integer : ∃! x : ℤ, x = (Nat.floor ((7 : ℚ) / 9 * 15) : ℤ) ∧ x = 11 := by sorry

end greatest_integer_inequality_unique_greatest_integer_l379_37964


namespace colombian_coffee_amount_l379_37903

/-- Proves the amount of Colombian coffee in a specific coffee mix -/
theorem colombian_coffee_amount
  (total_mix : ℝ)
  (colombian_price : ℝ)
  (brazilian_price : ℝ)
  (mix_price : ℝ)
  (h1 : total_mix = 100)
  (h2 : colombian_price = 8.75)
  (h3 : brazilian_price = 3.75)
  (h4 : mix_price = 6.35) :
  ∃ (colombian_amount : ℝ),
    colombian_amount = 52 ∧
    colombian_amount ≥ 0 ∧
    colombian_amount ≤ total_mix ∧
    ∃ (brazilian_amount : ℝ),
      brazilian_amount = total_mix - colombian_amount ∧
      colombian_price * colombian_amount + brazilian_price * brazilian_amount = mix_price * total_mix :=
by sorry

end colombian_coffee_amount_l379_37903


namespace arithmetic_sequence_sum_l379_37941

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Theorem statement
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 2 + a 3 + a 7 + a 8 = 8 →
  a 4 + a 6 = 4 :=
by
  sorry

end arithmetic_sequence_sum_l379_37941


namespace fraction_equals_zero_l379_37930

theorem fraction_equals_zero (x : ℝ) (h : x ≠ 0) :
  (x - 3) / (4 * x) = 0 ↔ x = 3 :=
by sorry

end fraction_equals_zero_l379_37930


namespace intersection_M_N_l379_37963

def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def N : Set ℝ := {x | |x| < 2}

theorem intersection_M_N : M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} := by sorry

end intersection_M_N_l379_37963


namespace nabla_problem_l379_37905

-- Define the ∇ operation
def nabla (a b : ℕ) : ℕ := 3 + b^(2*a)

-- Theorem statement
theorem nabla_problem : nabla (nabla 2 1) 2 = 259 := by
  sorry

end nabla_problem_l379_37905


namespace polynomial_factorization_l379_37912

theorem polynomial_factorization :
  (∀ x : ℝ, 2 * x^4 - 2 = 2 * (x^2 + 1) * (x + 1) * (x - 1)) ∧
  (∀ x : ℝ, x^4 - 18 * x^2 + 81 = (x + 3)^2 * (x - 3)^2) ∧
  (∀ y : ℝ, (y^2 - 1)^2 + 11 * (1 - y^2) + 24 = (y + 2) * (y - 2) * (y + 3) * (y - 3)) :=
by sorry

end polynomial_factorization_l379_37912


namespace positive_quadratic_expression_l379_37921

theorem positive_quadratic_expression (x y : ℝ) : x^2 + 2*y^2 + 2*x*y + 6*y + 10 > 0 := by
  sorry

end positive_quadratic_expression_l379_37921


namespace smaller_number_problem_l379_37914

theorem smaller_number_problem (a b : ℝ) (h1 : a + b = 18) (h2 : a * b = 45) :
  min a b = 3 := by sorry

end smaller_number_problem_l379_37914


namespace square_perimeter_l379_37933

theorem square_perimeter (area : ℝ) (perimeter : ℝ) : 
  area = 468 → perimeter = 24 * Real.sqrt 13 := by
  sorry

end square_perimeter_l379_37933


namespace yellow_two_days_ago_white_tomorrow_dandelion_counts_l379_37936

/-- Represents the state of dandelions in the meadow on a given day -/
structure DandelionState where
  yellow : ℕ
  white : ℕ

/-- The lifecycle of a dandelion -/
def dandelionLifecycle : ℕ := 3

/-- The state of dandelions yesterday -/
def yesterdayState : DandelionState := { yellow := 20, white := 14 }

/-- The state of dandelions today -/
def todayState : DandelionState := { yellow := 15, white := 11 }

/-- Theorem: The number of yellow dandelions the day before yesterday -/
theorem yellow_two_days_ago : ℕ := 25

/-- Theorem: The number of white dandelions tomorrow -/
theorem white_tomorrow : ℕ := 9

/-- Main theorem combining both results -/
theorem dandelion_counts : 
  (yellow_two_days_ago = yesterdayState.white + todayState.white) ∧
  (white_tomorrow = yesterdayState.yellow - todayState.white) := by
  sorry

end yellow_two_days_ago_white_tomorrow_dandelion_counts_l379_37936


namespace f_max_value_f_min_value_f_touches_x_axis_l379_37962

/-- A cubic function that touches the x-axis at (1,0) -/
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + x

/-- The maximum value of f(x) is 4/27 -/
theorem f_max_value : ∃ (x : ℝ), f x = 4/27 ∧ ∀ (y : ℝ), f y ≤ 4/27 :=
sorry

/-- The minimum value of f(x) is 0 -/
theorem f_min_value : ∃ (x : ℝ), f x = 0 ∧ ∀ (y : ℝ), f y ≥ 0 :=
sorry

/-- The function f(x) touches the x-axis at (1,0) -/
theorem f_touches_x_axis : f 1 = 0 ∧ ∀ (x : ℝ), x ≠ 1 → f x ≠ 0 :=
sorry

end f_max_value_f_min_value_f_touches_x_axis_l379_37962


namespace intersection_point_of_perpendicular_lines_l379_37953

/-- Given a line l: 2x + y = 10 and a point (-10, 0), this theorem proves that the 
    intersection point of l and the line l' passing through (-10, 0) and perpendicular 
    to l is (2, 6). -/
theorem intersection_point_of_perpendicular_lines 
  (l : Set (ℝ × ℝ)) 
  (h_l : l = {(x, y) | 2 * x + y = 10}) 
  (p : ℝ × ℝ) 
  (h_p : p = (-10, 0)) :
  ∃ (q : ℝ × ℝ), q ∈ l ∧ 
    (∃ (l' : Set (ℝ × ℝ)), p ∈ l' ∧ 
      (∀ (x y : ℝ), (x, y) ∈ l' ↔ (x - p.1) * 2 + (y - p.2) = 0) ∧
      q ∈ l' ∧
      q = (2, 6)) :=
sorry

end intersection_point_of_perpendicular_lines_l379_37953


namespace geometric_sequence_inequality_l379_37960

/-- 
Given a geometric sequence {b_n} where b_n > 0 for all n and the common ratio q > 1,
prove that b₄ + b₈ > b₅ + b₇.
-/
theorem geometric_sequence_inequality (b : ℕ → ℝ) (q : ℝ) 
  (h_positive : ∀ n, b n > 0)
  (h_geometric : ∀ n, b (n + 1) = q * b n)
  (h_q_gt_one : q > 1) :
  b 4 + b 8 > b 5 + b 7 := by
sorry

end geometric_sequence_inequality_l379_37960


namespace apple_trees_count_l379_37985

theorem apple_trees_count (total_trees orange_trees : ℕ) 
  (h1 : total_trees = 74)
  (h2 : orange_trees = 27) :
  total_trees - orange_trees = 47 := by
  sorry

end apple_trees_count_l379_37985


namespace find_number_l379_37957

theorem find_number : ∃ x : ℚ, (4 * x) / 7 + 12 = 36 ∧ x = 42 := by
  sorry

end find_number_l379_37957


namespace collinear_iff_sqrt_two_l379_37991

def a (k : ℝ) : ℝ × ℝ := (k, 2)
def b (k : ℝ) : ℝ × ℝ := (1, k)

def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v = (t • w.1, t • w.2)

theorem collinear_iff_sqrt_two (k : ℝ) :
  collinear (a k) (b k) ↔ k = Real.sqrt 2 := by
  sorry

end collinear_iff_sqrt_two_l379_37991


namespace area_common_to_translated_triangles_l379_37996

theorem area_common_to_translated_triangles : 
  let hypotenuse : ℝ := 10
  let translation : ℝ := 2
  let short_leg : ℝ := hypotenuse / 2
  let long_leg : ℝ := short_leg * Real.sqrt 3
  let overlap_height : ℝ := long_leg - translation
  let common_area : ℝ := (1 / 2) * hypotenuse * overlap_height
  common_area = 25 * Real.sqrt 3 - 10 := by
sorry

end area_common_to_translated_triangles_l379_37996


namespace square_side_length_l379_37978

theorem square_side_length (r s : ℕ) : 
  (2*r + s = 2000) →
  (2*r + 5*s = 3030) →
  s = 258 := by
sorry

end square_side_length_l379_37978


namespace yimin_orchard_tree_count_l379_37937

/-- The number of trees in Yimin Orchard -/
theorem yimin_orchard_tree_count : 
  let pear_rows : ℕ := 15
  let apple_rows : ℕ := 34
  let trees_per_row : ℕ := 21
  (pear_rows + apple_rows) * trees_per_row = 1029 := by
sorry

end yimin_orchard_tree_count_l379_37937


namespace factorization_1_factorization_2_factorization_3_factorization_4_l379_37922

-- 1. 2x^2 + 2x = 2x(x+1)
theorem factorization_1 (x : ℝ) : 2*x^2 + 2*x = 2*x*(x+1) := by sorry

-- 2. a^3 - a = a(a+1)(a-1)
theorem factorization_2 (a : ℝ) : a^3 - a = a*(a+1)*(a-1) := by sorry

-- 3. (x-y)^2 - 4(x-y) + 4 = (x-y-2)^2
theorem factorization_3 (x y : ℝ) : (x-y)^2 - 4*(x-y) + 4 = (x-y-2)^2 := by sorry

-- 4. x^2 + 2xy + y^2 - 9 = (x+y+3)(x+y-3)
theorem factorization_4 (x y : ℝ) : x^2 + 2*x*y + y^2 - 9 = (x+y+3)*(x+y-3) := by sorry

end factorization_1_factorization_2_factorization_3_factorization_4_l379_37922
