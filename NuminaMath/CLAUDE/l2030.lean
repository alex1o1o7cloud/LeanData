import Mathlib

namespace find_m_l2030_203014

theorem find_m (w x y z m : ℝ) 
  (h : 9 / (w + x + y) = m / (w + z) ∧ m / (w + z) = 15 / (z - x - y)) : 
  m = 24 := by
sorry

end find_m_l2030_203014


namespace player_a_wins_l2030_203048

/-- Represents a point on the coordinate plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a move in the game -/
inductive Move
  | right : Move
  | up : Move

/-- Represents the game state -/
structure GameState where
  piecePosition : Point
  markedPoints : Set Point
  movesLeft : ℕ

/-- The game rules -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  match move with
  | Move.right => 
    let newPos := Point.mk (state.piecePosition.x + 1) state.piecePosition.y
    newPos ∉ state.markedPoints
  | Move.up => 
    let newPos := Point.mk state.piecePosition.x (state.piecePosition.y + 1)
    newPos ∉ state.markedPoints

/-- Player A's strategy -/
def strategyA (k : ℕ) (state : GameState) : Point := sorry

/-- Theorem: Player A has a winning strategy for any positive k -/
theorem player_a_wins (k : ℕ) (h : k > 0) : 
  ∃ (strategy : GameState → Point), 
    ∀ (initialState : GameState),
      (∀ (move : Move), ¬isValidMove initialState move) ∨
      (∃ (finalState : GameState), 
        finalState.markedPoints = insert (strategy initialState) initialState.markedPoints ∧
        ∀ (move : Move), ¬isValidMove finalState move) := by
  sorry

end player_a_wins_l2030_203048


namespace arithmetic_sum_1000_l2030_203061

theorem arithmetic_sum_1000 : 
  ∀ m n : ℕ+, 
    (Finset.sum (Finset.range (m + 1)) (λ i => n + i) = 1000) ↔ 
    ((m = 4 ∧ n = 198) ∨ (m = 24 ∧ n = 28)) := by
  sorry

end arithmetic_sum_1000_l2030_203061


namespace first_car_right_turn_distance_l2030_203065

/-- The distance between two cars on a road --/
def distance_between_cars (initial_distance : ℝ) (car1_distance : ℝ) (car2_distance : ℝ) : ℝ :=
  initial_distance - car1_distance - car2_distance

/-- The total distance traveled by the first car --/
def car1_total_distance (x : ℝ) : ℝ := 25 + x + 25

theorem first_car_right_turn_distance (initial_distance : ℝ) (car2_distance : ℝ) (final_distance : ℝ) :
  initial_distance = 113 ∧ 
  car2_distance = 35 ∧ 
  final_distance = 28 →
  ∃ x : ℝ, 
    car1_total_distance x + car2_distance = 
    distance_between_cars initial_distance 25 car2_distance + final_distance ∧
    x = 21 := by
  sorry

end first_car_right_turn_distance_l2030_203065


namespace expected_value_of_product_l2030_203055

def marbles : Finset ℕ := {1, 2, 3, 4, 5, 6}

def product_sum : ℕ := (marbles.powerset.filter (fun s => s.card = 2)).sum (fun s => s.prod id)

def total_combinations : ℕ := Nat.choose 6 2

theorem expected_value_of_product :
  (product_sum : ℚ) / total_combinations = 35 / 3 :=
sorry

end expected_value_of_product_l2030_203055


namespace freshmen_psych_majors_percentage_l2030_203096

/-- The percentage of freshmen psychology majors in the School of Liberal Arts
    among all students at a certain college. -/
theorem freshmen_psych_majors_percentage
  (total_students : ℕ)
  (freshmen_percentage : ℚ)
  (liberal_arts_percentage : ℚ)
  (psychology_percentage : ℚ)
  (h1 : freshmen_percentage = 2/5)
  (h2 : liberal_arts_percentage = 1/2)
  (h3 : psychology_percentage = 1/2)
  : (freshmen_percentage * liberal_arts_percentage * psychology_percentage : ℚ) = 1/10 := by
  sorry

#check freshmen_psych_majors_percentage

end freshmen_psych_majors_percentage_l2030_203096


namespace triangle_property_l2030_203085

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
theorem triangle_property (A B C : Real) (a b c : Real) :
  -- Triangle existence conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Law of sines
  a / Real.sin A = b / Real.sin B →
  -- Given condition
  Real.cos A / (1 + Real.sin A) = Real.sin B / (1 + Real.cos B) →
  -- Conclusions
  C = π / 2 ∧ 
  1 < (a * b + b * c + c * a) / (c^2) ∧ 
  (a * b + b * c + c * a) / (c^2) ≤ (1 + 2 * Real.sqrt 2) / 2 := by
  sorry

end triangle_property_l2030_203085


namespace punch_bowl_theorem_l2030_203079

/-- The capacity of the punch bowl in gallons -/
def bowl_capacity : ℝ := 16

/-- The amount of punch Mark adds in the second refill -/
def second_refill : ℝ := 4

/-- The amount of punch Sally drinks -/
def sally_drinks : ℝ := 2

/-- The amount of punch Mark adds to completely fill the bowl at the end -/
def final_addition : ℝ := 12

/-- The initial amount of punch Mark added to the bowl -/
def initial_amount : ℝ := 4

theorem punch_bowl_theorem :
  let after_cousin := initial_amount / 2
  let after_second_refill := after_cousin + second_refill
  let after_sally := after_second_refill - sally_drinks
  after_sally + final_addition = bowl_capacity :=
by sorry

end punch_bowl_theorem_l2030_203079


namespace no_integer_tangent_length_l2030_203056

theorem no_integer_tangent_length (t₁ m : ℕ) : 
  (∃ (m : ℕ), m % 2 = 1 ∧ m < 24 ∧ t₁^2 = m * (24 - m)) → False :=
sorry

end no_integer_tangent_length_l2030_203056


namespace difference_of_squares_l2030_203036

theorem difference_of_squares (n : ℕ) : 
  (n = 105 → ∃! k : ℕ, k = 4 ∧ ∃ s : Finset (ℕ × ℕ), s.card = k ∧ ∀ (x y : ℕ), (x, y) ∈ s ↔ x^2 - y^2 = n) ∧
  (n = 106 → ¬∃ (x y : ℕ), x^2 - y^2 = n) :=
by sorry

end difference_of_squares_l2030_203036


namespace necklace_cost_calculation_l2030_203024

/-- The cost of a single necklace -/
def necklace_cost : ℕ := sorry

/-- The number of necklaces sold -/
def necklaces_sold : ℕ := 4

/-- The number of rings sold -/
def rings_sold : ℕ := 8

/-- The cost of a single ring -/
def ring_cost : ℕ := 4

/-- The total sales amount -/
def total_sales : ℕ := 80

theorem necklace_cost_calculation :
  necklace_cost = 12 :=
by
  sorry

#check necklace_cost_calculation

end necklace_cost_calculation_l2030_203024


namespace f_composition_at_two_l2030_203004

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -Real.sqrt x
  else (x - 1/x)^4

theorem f_composition_at_two : f (f 2) = 1/4 := by
  sorry

end f_composition_at_two_l2030_203004


namespace min_value_C_over_D_l2030_203090

theorem min_value_C_over_D (x C D : ℝ) (hx : x ≠ 0) 
  (hC : x^2 + 1/x^2 = C) (hD : x + 1/x = D) (hCpos : C > 0) (hDpos : D > 0) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 2 ∧ ∀ y, y = C / D → y ≥ m :=
sorry

end min_value_C_over_D_l2030_203090


namespace sin_alpha_value_l2030_203015

theorem sin_alpha_value (α : Real) 
  (h1 : 0 < α) 
  (h2 : α < Real.pi / 4) 
  (h3 : Real.sin α * Real.cos α = 3 * Real.sqrt 7 / 16) : 
  Real.sin α = Real.sqrt 7 / 4 := by
  sorry

end sin_alpha_value_l2030_203015


namespace cost_in_usd_l2030_203073

/-- The cost of coffee and snack in USD given their prices in yen and the exchange rate -/
theorem cost_in_usd (coffee_yen : ℕ) (snack_yen : ℕ) (exchange_rate : ℚ) : 
  coffee_yen = 250 → snack_yen = 150 → exchange_rate = 1 / 100 →
  (coffee_yen + snack_yen : ℚ) * exchange_rate = 4 := by
  sorry

end cost_in_usd_l2030_203073


namespace markup_rate_l2030_203031

theorem markup_rate (S : ℝ) (h1 : S > 0) : 
  let profit := 0.20 * S
  let expenses := 0.20 * S
  let cost := S - profit - expenses
  (S - cost) / cost * 100 = 200 / 3 := by sorry

end markup_rate_l2030_203031


namespace apollo_wheel_replacement_ratio_l2030_203066

/-- Represents the chariot wheel replacement scenario -/
structure WheelReplacement where
  initial_rate : ℕ  -- Initial rate in golden apples
  months : ℕ        -- Total number of months
  half_year : ℕ     -- Number of months before rate change
  total_payment : ℕ -- Total payment for the year

/-- Calculates the ratio of new rate to old rate -/
def rate_ratio (w : WheelReplacement) : ℚ :=
  let first_half_payment := w.initial_rate * w.half_year
  let second_half_payment := w.total_payment - first_half_payment
  (second_half_payment : ℚ) / (w.initial_rate * (w.months - w.half_year))

/-- Theorem stating that the rate ratio is 2 for the given scenario -/
theorem apollo_wheel_replacement_ratio :
  let w : WheelReplacement := ⟨3, 12, 6, 54⟩
  rate_ratio w = 2 := by
  sorry

end apollo_wheel_replacement_ratio_l2030_203066


namespace figure_100_squares_l2030_203003

def f (n : ℕ) : ℕ := n^3 + 2*n^2 + 2*n + 1

theorem figure_100_squares :
  f 0 = 1 ∧ f 1 = 6 ∧ f 2 = 20 ∧ f 3 = 50 → f 100 = 1020201 := by
  sorry

end figure_100_squares_l2030_203003


namespace henri_reads_1800_words_l2030_203052

/-- Calculates the number of words read given total free time, movie durations, and reading rate. -/
def words_read (total_time : ℝ) (movie1_duration : ℝ) (movie2_duration : ℝ) (reading_rate : ℝ) : ℝ :=
  (total_time - movie1_duration - movie2_duration) * reading_rate * 60

/-- Proves that Henri reads 1800 words given the specified conditions. -/
theorem henri_reads_1800_words :
  words_read 8 3.5 1.5 10 = 1800 := by
  sorry

end henri_reads_1800_words_l2030_203052


namespace solution_set_equiv_solution_values_l2030_203013

-- Part I
def solution_set (x : ℝ) : Prop := |x + 3| < 2*x + 1

theorem solution_set_equiv : ∀ x : ℝ, solution_set x ↔ x > 2 := by sorry

-- Part II
def has_solution (t : ℝ) : Prop := 
  t ≠ 0 ∧ ∃ x : ℝ, |x - t| + |x + 1/t| = 2

theorem solution_values : ∀ t : ℝ, has_solution t → t = 1 ∨ t = -1 := by sorry

end solution_set_equiv_solution_values_l2030_203013


namespace exotic_courses_divisibility_l2030_203076

/-- Represents a country with airports -/
structure Country where
  airports : ℕ

/-- Represents the flight system between two countries -/
structure FlightSystem where
  countryA : Country
  countryB : Country
  flightsPerAirport : ℕ
  noInternalFlights : Bool

/-- Represents an exotic traveling course -/
structure ExoticTravelingCourse where
  flightSystem : FlightSystem
  courseLength : ℕ

/-- The number of all exotic traveling courses -/
def numberOfExoticCourses (f : FlightSystem) : ℕ :=
  sorry

theorem exotic_courses_divisibility (f : FlightSystem) 
  (h1 : f.countryA.airports = f.countryB.airports)
  (h2 : f.countryA.airports ≥ 2)
  (h3 : f.flightsPerAirport = 3)
  (h4 : f.noInternalFlights = true) :
  ∃ k : ℕ, numberOfExoticCourses f = 8 * f.countryA.airports * k ∧ Even k :=
sorry

end exotic_courses_divisibility_l2030_203076


namespace correct_average_l2030_203050

theorem correct_average (n : ℕ) (initial_avg : ℚ) (wrong_num correct_num : ℕ) :
  n = 10 ∧ initial_avg = 14 ∧ wrong_num = 26 ∧ correct_num = 36 →
  (n * initial_avg + (correct_num - wrong_num)) / n = 15 := by
  sorry

end correct_average_l2030_203050


namespace zach_savings_l2030_203038

/-- Represents the financial situation of Zach saving for a bike --/
structure BikeSavings where
  bikeCost : ℕ
  weeklyAllowance : ℕ
  lawnMowingPay : ℕ
  babysittingRate : ℕ
  babysittingHours : ℕ
  additionalNeeded : ℕ

/-- Calculates the amount Zach has already saved --/
def amountSaved (s : BikeSavings) : ℕ :=
  s.bikeCost - (s.weeklyAllowance + s.lawnMowingPay + s.babysittingRate * s.babysittingHours) - s.additionalNeeded

/-- Theorem stating that for Zach's specific situation, he has already saved $65 --/
theorem zach_savings : 
  let s : BikeSavings := {
    bikeCost := 100,
    weeklyAllowance := 5,
    lawnMowingPay := 10,
    babysittingRate := 7,
    babysittingHours := 2,
    additionalNeeded := 6
  }
  amountSaved s = 65 := by sorry

end zach_savings_l2030_203038


namespace lineups_count_is_sixty_l2030_203074

/-- The number of ways to arrange r items out of n items -/
def permutations (n : ℕ) (r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

/-- The number of possible lineups for 3 games with 3 athletes selected from 5 -/
def lineups_count : ℕ := permutations 5 3

theorem lineups_count_is_sixty : lineups_count = 60 := by sorry

end lineups_count_is_sixty_l2030_203074


namespace cupcake_cost_split_l2030_203022

theorem cupcake_cost_split (num_cupcakes : ℕ) (cost_per_cupcake : ℚ) (num_people : ℕ) :
  num_cupcakes = 12 →
  cost_per_cupcake = 3/2 →
  num_people = 2 →
  (num_cupcakes : ℚ) * cost_per_cupcake / (num_people : ℚ) = 9 :=
by sorry

end cupcake_cost_split_l2030_203022


namespace smallest_square_perimeter_l2030_203072

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ

/-- Calculates the perimeter of a square -/
def Square.perimeter (s : Square) : ℝ := 4 * s.sideLength

/-- Represents three concentric squares -/
structure ConcentricSquares where
  largest : Square
  middle : Square
  smallest : Square
  distanceBetweenSides : ℝ

/-- The theorem stating the perimeter of the smallest square in the given configuration -/
theorem smallest_square_perimeter (cs : ConcentricSquares)
    (h1 : cs.largest.sideLength = 22)
    (h2 : cs.distanceBetweenSides = 3)
    (h3 : cs.middle.sideLength = cs.largest.sideLength - 2 * cs.distanceBetweenSides)
    (h4 : cs.smallest.sideLength = cs.middle.sideLength - 2 * cs.distanceBetweenSides) :
    cs.smallest.perimeter = 40 := by
  sorry

end smallest_square_perimeter_l2030_203072


namespace abhay_speed_l2030_203009

theorem abhay_speed (distance : ℝ) (a s : ℝ → ℝ) :
  distance = 30 →
  (∀ x, a x > 0 ∧ s x > 0) →
  (∀ x, distance / (a x) = distance / (s x) + 2) →
  (∀ x, distance / (2 * a x) = distance / (s x) - 1) →
  (∃ x, a x = 5 * Real.sqrt 6) :=
sorry

end abhay_speed_l2030_203009


namespace small_semicircle_radius_l2030_203069

/-- Configuration of tangent semicircles and circle -/
structure TangentConfiguration where
  R : ℝ  -- Radius of the large semicircle
  r : ℝ  -- Radius of the circle
  x : ℝ  -- Radius of the small semicircle
  tangent : R > 0 ∧ r > 0 ∧ x > 0  -- All radii are positive
  large_semicircle : R = 12  -- Large semicircle has radius 12
  circle : r = 6  -- Circle has radius 6
  pythagorean : r^2 + (R - x)^2 = (r + x)^2  -- Pythagorean theorem for tangent configuration

/-- The radius of the small semicircle in the tangent configuration is 4 -/
theorem small_semicircle_radius (config : TangentConfiguration) : config.x = 4 :=
  sorry

end small_semicircle_radius_l2030_203069


namespace shelving_orders_eq_1280_l2030_203033

/-- The number of books in total -/
def total_books : ℕ := 10

/-- The label of the book that has already been shelved -/
def shelved_book : ℕ := 9

/-- Calculate the number of different possible orders for shelving the remaining books -/
def shelving_orders : ℕ :=
  (Finset.range (total_books - 1)).sum (fun k =>
    (Nat.choose (total_books - 2) k) * (k + 2))

/-- Theorem stating that the number of different possible orders for shelving the remaining books is 1280 -/
theorem shelving_orders_eq_1280 : shelving_orders = 1280 := by
  sorry

end shelving_orders_eq_1280_l2030_203033


namespace circle_radius_values_l2030_203086

/-- Given a circle and its tangent line, prove the possible values of its radius -/
theorem circle_radius_values (r : ℝ) (k : ℝ) : 
  r > 0 → 
  (∀ x y, (x - 1)^2 + (y - 3 * Real.sqrt 3)^2 = r^2) →
  (∃ x y, y = k * x + Real.sqrt 3) →
  (k = Real.sqrt 3 ∨ k = -Real.sqrt 3) →
  (r = Real.sqrt 3 / 2 ∨ r = 3 * Real.sqrt 3 / 2) := by
  sorry

end circle_radius_values_l2030_203086


namespace shopkeeper_gain_percentage_l2030_203092

theorem shopkeeper_gain_percentage (marked_price cost_price : ℝ) :
  marked_price > 0 ∧ cost_price > 0 ∧
  0.9 * marked_price = 1.17 * cost_price →
  (marked_price - cost_price) / cost_price = 0.3 := by
  sorry

end shopkeeper_gain_percentage_l2030_203092


namespace problem_solution_l2030_203035

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Define the theorem
theorem problem_solution :
  -- Given conditions
  ∃ m : ℝ,
    m > 0 ∧
    (∀ x : ℝ, f (x + 5) ≤ 3 * m ↔ -7 ≤ x ∧ x ≤ -1) ∧
    -- Part 1: The value of m is 1
    m = 1 ∧
    -- Part 2: Maximum value of 2a√(1+b²) is 2√2
    (∀ a b : ℝ, a > 0 → b > 0 → 2 * a^2 + b^2 = 3 * m →
      2 * a * Real.sqrt (1 + b^2) ≤ 2 * Real.sqrt 2) ∧
    (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2 * a^2 + b^2 = 3 * m ∧
      2 * a * Real.sqrt (1 + b^2) = 2 * Real.sqrt 2) :=
by sorry

end problem_solution_l2030_203035


namespace infinite_series_sum_l2030_203007

theorem infinite_series_sum : 
  ∑' (n : ℕ), (1 : ℝ) / (n * (n + 3)) = 1/3 + 1/6 + 1/9 := by
  sorry

end infinite_series_sum_l2030_203007


namespace laura_triathlon_speed_l2030_203091

theorem laura_triathlon_speed :
  ∃ x : ℝ, x > 0 ∧ (20 / (2 * x + 1)) + (5 / x) + (5 / 60) = 110 / 60 := by
  sorry

end laura_triathlon_speed_l2030_203091


namespace discount_percentage_l2030_203099

theorem discount_percentage (original_price sale_price : ℝ) 
  (h1 : original_price = 150)
  (h2 : sale_price = 135) :
  (original_price - sale_price) / original_price * 100 = 10 := by
  sorry

end discount_percentage_l2030_203099


namespace complex_equation_solution_l2030_203062

theorem complex_equation_solution (z : ℂ) (b : ℝ) :
  z * (1 + Complex.I) = 1 - b * Complex.I →
  Complex.abs z = Real.sqrt 2 →
  b = Real.sqrt 3 ∨ b = -Real.sqrt 3 :=
by sorry

end complex_equation_solution_l2030_203062


namespace marys_max_earnings_l2030_203053

/-- Calculates the maximum weekly earnings for a worker with the given parameters. -/
def max_weekly_earnings (max_hours : ℕ) (regular_hours : ℕ) (regular_rate : ℚ) (overtime_rate_increase : ℚ) : ℚ :=
  let overtime_hours := max_hours - regular_hours
  let overtime_rate := regular_rate * (1 + overtime_rate_increase)
  regular_rate * regular_hours + overtime_rate * overtime_hours

/-- Theorem stating that Mary's maximum weekly earnings are $410 -/
theorem marys_max_earnings :
  max_weekly_earnings 45 20 8 (1/4) = 410 := by
  sorry

#eval max_weekly_earnings 45 20 8 (1/4)

end marys_max_earnings_l2030_203053


namespace ellipse_sum_l2030_203064

-- Define the ellipse
def Ellipse (F₁ F₂ : ℝ × ℝ) (d : ℝ) :=
  {P : ℝ × ℝ | dist P F₁ + dist P F₂ = d}

-- Define the foci
def F₁ : ℝ × ℝ := (0, 0)
def F₂ : ℝ × ℝ := (6, 0)

-- Define the distance sum
def d : ℝ := 10

-- Theorem statement
theorem ellipse_sum (h k a b : ℝ) :
  Ellipse F₁ F₂ d →
  (∀ (x y : ℝ), (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1 ↔ (x, y) ∈ Ellipse F₁ F₂ d) →
  h + k + a + b = 12 := by sorry

end ellipse_sum_l2030_203064


namespace sum_of_squares_of_roots_l2030_203049

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (6 * x₁^2 - 9 * x₁ + 5 = 0) → 
  (6 * x₂^2 - 9 * x₂ + 5 = 0) → 
  x₁^2 + x₂^2 = 7/12 := by
  sorry

end sum_of_squares_of_roots_l2030_203049


namespace twentieth_term_is_220_l2030_203045

def a (n : ℕ) : ℚ := (1/2) * n * (n + 2)

theorem twentieth_term_is_220 : a 20 = 220 := by
  sorry

end twentieth_term_is_220_l2030_203045


namespace difference_of_squares_special_case_l2030_203017

theorem difference_of_squares_special_case : (831 : ℤ) * 831 - 830 * 832 = 1 := by
  sorry

end difference_of_squares_special_case_l2030_203017


namespace max_cookies_andy_l2030_203027

/-- The number of cookies baked by the siblings -/
def total_cookies : ℕ := 36

/-- Andy's cookies -/
def andy_cookies : ℕ → ℕ := λ x => x

/-- Aaron's cookies -/
def aaron_cookies : ℕ → ℕ := λ x => 2 * x

/-- Alexa's cookies -/
def alexa_cookies : ℕ → ℕ := λ x => total_cookies - x - 2 * x

/-- The maximum number of cookies Andy could have eaten -/
def max_andy_cookies : ℕ := 12

theorem max_cookies_andy :
  ∀ x : ℕ,
  x ≤ max_andy_cookies ∧
  andy_cookies x + aaron_cookies x + alexa_cookies x = total_cookies ∧
  alexa_cookies x ≥ 0 :=
by
  sorry

end max_cookies_andy_l2030_203027


namespace quadratic_function_properties_l2030_203000

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_properties (a b c : ℝ) 
  (h1 : QuadraticFunction a b c 1 = -a/2)
  (h2 : a > 0)
  (h3 : ∀ x, QuadraticFunction a b c x < 1 ↔ 0 < x ∧ x < 3) :
  (QuadraticFunction a b c = fun x ↦ (2/3) * x^2 - 2 * x + 1) ∧
  (∃ x, 0 < x ∧ x < 2 ∧ QuadraticFunction a b c x = 0) := by
  sorry

end quadratic_function_properties_l2030_203000


namespace inverse_proportion_problem_l2030_203044

/-- Given that a and b are inversely proportional and a = 3b when a + b = 60,
    prove that b = -67.5 when a = -10 -/
theorem inverse_proportion_problem (a b : ℝ) (k : ℝ) : 
  (∀ x y, x * y = k → (x = a ∧ y = b) ∨ (x = b ∧ y = a)) →  -- inverse proportion
  (∃ a' b', a' + b' = 60 ∧ a' = 3 * b') →                   -- condition when sum is 60
  (a = -10) →                                               -- given a value
  (b = -67.5) :=                                            -- to prove
by sorry

end inverse_proportion_problem_l2030_203044


namespace binomial_150_150_l2030_203077

theorem binomial_150_150 : Nat.choose 150 150 = 1 := by sorry

end binomial_150_150_l2030_203077


namespace sample_size_is_twenty_l2030_203037

/-- Represents the total number of employees in the company -/
def total_employees : ℕ := 1000

/-- Represents the number of middle-aged workers in the company -/
def middle_aged_workers : ℕ := 350

/-- Represents the number of middle-aged workers in the sample -/
def sample_middle_aged : ℕ := 7

/-- Theorem stating that the sample size is 20 given the conditions -/
theorem sample_size_is_twenty :
  ∃ (sample_size : ℕ),
    (sample_middle_aged : ℚ) / sample_size = middle_aged_workers / total_employees ∧
    sample_size = 20 := by
  sorry

end sample_size_is_twenty_l2030_203037


namespace quadratic_a_value_main_quadratic_theorem_l2030_203078

/-- A quadratic function with vertex form y = a(x - h)^2 + k, where (h, k) is the vertex -/
structure QuadraticFunction where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The theorem stating the value of 'a' for a quadratic function with given properties -/
theorem quadratic_a_value (f : QuadraticFunction) 
  (vertex_condition : f.h = -3 ∧ f.k = 0)
  (point_condition : f.a * (2 - f.h)^2 + f.k = -36) :
  f.a = -36/25 := by
  sorry

/-- The main theorem proving the value of 'a' for the given quadratic function -/
theorem main_quadratic_theorem :
  ∃ f : QuadraticFunction, 
    f.h = -3 ∧ 
    f.k = 0 ∧ 
    f.a * (2 - f.h)^2 + f.k = -36 ∧
    f.a = -36/25 := by
  sorry

end quadratic_a_value_main_quadratic_theorem_l2030_203078


namespace wall_length_is_850_l2030_203083

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a wall in centimeters -/
structure WallDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a brick given its dimensions -/
def brickVolume (b : BrickDimensions) : ℝ :=
  b.length * b.width * b.height

/-- Calculates the volume of a wall given its dimensions -/
def wallVolume (w : WallDimensions) : ℝ :=
  w.length * w.width * w.height

/-- The main theorem stating that under given conditions, the wall length is 850 cm -/
theorem wall_length_is_850 (brick : BrickDimensions)
    (wall : WallDimensions) (num_bricks : ℕ) :
    brick.length = 25 →
    brick.width = 11.25 →
    brick.height = 6 →
    wall.width = 600 →
    wall.height = 22.5 →
    num_bricks = 6800 →
    brickVolume brick * num_bricks = wallVolume wall →
    wall.length = 850 := by
  sorry


end wall_length_is_850_l2030_203083


namespace steven_arrangement_count_l2030_203043

/-- The number of letters in "STEVEN" excluding one "E" -/
def n : ℕ := 5

/-- The number of permutations of "STEVEN" with one "E" fixed at the end -/
def steven_permutations : ℕ := n.factorial

theorem steven_arrangement_count : steven_permutations = 120 := by
  sorry

end steven_arrangement_count_l2030_203043


namespace triangle_theorem_l2030_203028

theorem triangle_theorem (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π → -- Triangle condition
  a > 0 ∧ b > 0 ∧ c > 0 → -- Positive side lengths
  a * (Real.sin A - Real.sin B) + b * Real.sin B = c * Real.sin C → -- Line condition
  2 * (Real.cos (A / 2))^2 - 2 * (Real.sin (B / 2))^2 = Real.sqrt 3 / 2 → -- Given equation
  A < B → -- Given inequality
  C = π / 3 ∧ c / a = Real.sqrt 3 := by
sorry

end triangle_theorem_l2030_203028


namespace parallel_lines_a_equals_3_l2030_203020

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {m1 m2 b1 b2 : ℝ} :
  (∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The first line: 3y - a = 9x + 1 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := 3 * y - a = 9 * x + 1

/-- The second line: y - 2 = (2a - 3)x -/
def line2 (a : ℝ) (x y : ℝ) : Prop := y - 2 = (2 * a - 3) * x

theorem parallel_lines_a_equals_3 :
  ∀ a : ℝ, (∀ x y : ℝ, line1 a x y ↔ line2 a x y) → a = 3 := by
  sorry

end parallel_lines_a_equals_3_l2030_203020


namespace tan_three_implies_sum_l2030_203012

theorem tan_three_implies_sum (θ : Real) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ + Real.sin θ / (1 + Real.cos θ) = 2 * (Real.sqrt 10 - 1) / 3 := by
  sorry

end tan_three_implies_sum_l2030_203012


namespace hexagonal_pyramid_volume_l2030_203051

/-- The volume of a regular hexagonal pyramid with base side length a and lateral surface area 10 times larger than the base area -/
theorem hexagonal_pyramid_volume (a : ℝ) (h : a > 0) : 
  let base_area := (3 * Real.sqrt 3 / 2) * a^2
  let lateral_area := 10 * base_area
  let height := (3 * a * Real.sqrt 33) / 2
  let volume := (1 / 3) * base_area * height
  volume = (9 * a^3 * Real.sqrt 11) / 4 := by
sorry

end hexagonal_pyramid_volume_l2030_203051


namespace lawn_width_l2030_203094

theorem lawn_width (area : ℝ) (length : ℝ) (width : ℝ) 
  (h1 : area = 20)
  (h2 : length = 4)
  (h3 : area = length * width) : 
  width = 5 := by
sorry

end lawn_width_l2030_203094


namespace quadratic_roots_relation_l2030_203025

theorem quadratic_roots_relation (a b n r s : ℝ) : 
  (a^2 - n*a + 3 = 0) →
  (b^2 - n*b + 3 = 0) →
  ((a + 1/b)^2 - r*(a + 1/b) + s = 0) →
  ((b + 1/a)^2 - r*(b + 1/a) + s = 0) →
  s = 16/3 := by
sorry

end quadratic_roots_relation_l2030_203025


namespace expand_and_simplify_l2030_203054

theorem expand_and_simplify (x : ℝ) (h : x ≠ 0) :
  3 / 7 * (7 / x + 14 * x^3) = 3 / x + 6 * x^3 := by
  sorry

end expand_and_simplify_l2030_203054


namespace quadratic_inequality_properties_l2030_203081

/-- Given a quadratic inequality with specific properties, prove certain statements about its coefficients and solutions. -/
theorem quadratic_inequality_properties
  (a b : ℝ) (d : ℝ)
  (h_a_pos : a > 0)
  (h_solution_set : ∀ x : ℝ, x^2 + a*x + b > 0 ↔ x ≠ d) :
  (a^2 = 4*b) ∧
  (a^2 + 1/b ≥ 4) ∧
  (∀ c x₁ x₂ : ℝ, (∀ x : ℝ, x^2 + a*x + b < c ↔ x₁ < x ∧ x < x₂) →
    |x₁ - x₂| = 4 → c = 4) :=
by sorry

end quadratic_inequality_properties_l2030_203081


namespace correct_num_technicians_l2030_203059

/-- Represents the workshop scenario with workers and salaries -/
structure Workshop where
  total_workers : ℕ
  avg_salary : ℚ
  technician_salary : ℚ
  other_salary : ℚ

/-- The number of technicians in the workshop -/
def num_technicians (w : Workshop) : ℕ :=
  7  -- We'll prove this is correct

/-- The given workshop scenario -/
def given_workshop : Workshop :=
  { total_workers := 56
    avg_salary := 6750
    technician_salary := 12000
    other_salary := 6000 }

/-- Theorem stating that the number of technicians in the given workshop is correct -/
theorem correct_num_technicians :
    let n := num_technicians given_workshop
    let m := given_workshop.total_workers - n
    n + m = given_workshop.total_workers ∧
    (n * given_workshop.technician_salary + m * given_workshop.other_salary) / given_workshop.total_workers = given_workshop.avg_salary :=
  sorry


end correct_num_technicians_l2030_203059


namespace annulus_area_l2030_203008

/-- An annulus is the region between two concentric circles. -/
structure Annulus where
  b : ℝ
  c : ℝ
  a : ℝ
  h1 : b > c
  h2 : a^2 + c^2 = b^2

/-- The area of an annulus is πa², where a is the length of a line segment
    tangent to the inner circle and extending from the outer circle to the
    point of tangency. -/
theorem annulus_area (ann : Annulus) : Real.pi * ann.a^2 = Real.pi * (ann.b^2 - ann.c^2) := by
  sorry

end annulus_area_l2030_203008


namespace rectangle_count_l2030_203001

def horizontal_lines : ℕ := 5
def vertical_lines : ℕ := 5

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem rectangle_count : 
  choose horizontal_lines 2 * choose vertical_lines 2 = 100 := by sorry

end rectangle_count_l2030_203001


namespace percentage_calculation_l2030_203002

theorem percentage_calculation : (1 / 8 / 100 * 160) + 0.5 = 0.7 := by
  sorry

end percentage_calculation_l2030_203002


namespace walking_rate_ratio_l2030_203010

theorem walking_rate_ratio (usual_time faster_time : ℝ) (h1 : usual_time = 28) 
  (h2 : faster_time = usual_time - 4) : 
  (usual_time / faster_time) = 7 / 6 := by
  sorry

end walking_rate_ratio_l2030_203010


namespace max_value_of_expression_l2030_203080

theorem max_value_of_expression (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x + y + z = 3) : 
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) ≤ 729/108 := by
  sorry

end max_value_of_expression_l2030_203080


namespace range_of_a_l2030_203040

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | x^2 + 5*a*x + 6*a^2 ≤ 0}

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (A ∪ B a = A) → a < 0 → -1/2 ≥ a ∧ a > -4/3 :=
by sorry

end range_of_a_l2030_203040


namespace book_price_decrease_l2030_203034

theorem book_price_decrease (P : ℝ) (x : ℝ) : 
  P - ((P - (x / 100) * P) * 1.2) = 10.000000000000014 → 
  x = 50 / 3 := by
sorry

end book_price_decrease_l2030_203034


namespace max_value_quadratic_l2030_203082

theorem max_value_quadratic (x y : ℝ) : 
  4 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 10 ≤ -13 :=
sorry

end max_value_quadratic_l2030_203082


namespace x_value_proof_l2030_203070

theorem x_value_proof (x : ℕ) 
  (h1 : x > 0) 
  (h2 : (x * x) / 100 = 16) 
  (h3 : 4 ∣ x) : 
  x = 40 := by
sorry

end x_value_proof_l2030_203070


namespace simplify_and_evaluate_l2030_203039

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 3 + 1) :
  (a + 1) / (a^2 - 2*a + 1) / (1 + 2 / (a - 1)) = Real.sqrt 3 / 3 := by
  sorry

end simplify_and_evaluate_l2030_203039


namespace contingency_and_sampling_theorem_l2030_203068

/-- Represents the contingency table --/
structure ContingencyTable :=
  (male_running : ℕ)
  (male_not_running : ℕ)
  (female_running : ℕ)
  (female_not_running : ℕ)

/-- Calculates the K^2 value for the contingency table --/
def calculate_k_squared (table : ContingencyTable) : ℚ :=
  let n := table.male_running + table.male_not_running + table.female_running + table.female_not_running
  let a := table.male_running
  let b := table.male_not_running
  let c := table.female_running
  let d := table.female_not_running
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Calculates the expected value of females selected in the sampling process --/
def expected_females_selected (male_count female_count : ℕ) : ℚ :=
  (0 * (male_count * (male_count - 1)) + 
   1 * (2 * male_count * female_count) + 
   2 * (female_count * (female_count - 1))) / 
  ((male_count + female_count) * (male_count + female_count - 1))

/-- Main theorem to prove --/
theorem contingency_and_sampling_theorem 
  (table : ContingencyTable) 
  (h_total : table.male_running + table.male_not_running + table.female_running + table.female_not_running = 80)
  (h_male_running : table.male_running = 20)
  (h_male_not_running : table.male_not_running = 20)
  (h_female_not_running : table.female_not_running = 30) :
  calculate_k_squared table < (6635 : ℚ) / 1000 ∧ 
  expected_females_selected 2 3 = 6 / 5 := by
  sorry

end contingency_and_sampling_theorem_l2030_203068


namespace imaginary_part_of_complex_fraction_l2030_203047

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (Complex.I - 1)^2 + 4 / (Complex.I + 1)
  (z.im = -3) := by sorry

end imaginary_part_of_complex_fraction_l2030_203047


namespace original_integer_is_45_l2030_203018

theorem original_integer_is_45 (a b c d : ℤ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (eq1 : (b + c + d) / 3 + 10 = 37)
  (eq2 : (a + c + d) / 3 + 10 = 31)
  (eq3 : (a + b + d) / 3 + 10 = 25)
  (eq4 : (a + b + c) / 3 + 10 = 19) :
  a = 45 ∨ b = 45 ∨ c = 45 ∨ d = 45 :=
sorry

end original_integer_is_45_l2030_203018


namespace shelter_cats_count_l2030_203029

theorem shelter_cats_count (total animals : ℕ) (cats dogs : ℕ) : 
  total = 60 →
  cats = dogs + 20 →
  cats + dogs = total →
  cats = 40 :=
by sorry

end shelter_cats_count_l2030_203029


namespace burn_represents_8615_l2030_203093

/-- Represents a mapping from characters to digits -/
def DigitMapping := Char → Fin 10

/-- The sequence of characters used in the code -/
def codeSequence : List Char := ['G', 'R', 'E', 'A', 'T', 'N', 'U', 'M', 'B', 'S']

/-- Creates a mapping from the code sequence to digits 0-9 -/
def createMapping (seq : List Char) : DigitMapping :=
  fun c => match seq.indexOf? c with
    | some i => ⟨i, by sorry⟩
    | none => 0

/-- The mapping for our specific code -/
def mapping : DigitMapping := createMapping codeSequence

/-- Converts a string to a number using the given mapping -/
def stringToNumber (s : String) (m : DigitMapping) : Nat :=
  s.foldr (fun c acc => acc * 10 + m c) 0

theorem burn_represents_8615 :
  stringToNumber "BURN" mapping = 8615 := by sorry

end burn_represents_8615_l2030_203093


namespace right_triangle_arctan_sum_l2030_203058

theorem right_triangle_arctan_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  a^2 = b^2 + c^2 → Real.arctan (b / (c + a)) + Real.arctan (c / (b + a)) = π / 4 := by
  sorry

end right_triangle_arctan_sum_l2030_203058


namespace coupon_savings_difference_l2030_203071

/-- Represents the savings from Coupon A (20% discount) -/
def savingsA (price : ℝ) : ℝ := 0.2 * price

/-- Represents the savings from Coupon B ($50 flat discount) -/
def savingsB : ℝ := 50

/-- Represents the savings from Coupon C (30% discount on amount over $200) -/
def savingsC (price : ℝ) : ℝ := 0.3 * (price - 200)

/-- The minimum price where Coupon A saves at least as much as Coupons B and C -/
def minPrice : ℝ := 250

/-- The maximum price where Coupon A saves at least as much as Coupons B and C -/
def maxPrice : ℝ := 600

theorem coupon_savings_difference :
  ∀ price : ℝ, price > 200 →
  (savingsA price ≥ savingsB ∧ savingsA price ≥ savingsC price) →
  minPrice ≤ price ∧ price ≤ maxPrice →
  maxPrice - minPrice = 350 :=
by
  sorry

end coupon_savings_difference_l2030_203071


namespace not_necessarily_divisible_by_44_l2030_203042

theorem not_necessarily_divisible_by_44 (k : ℤ) (n : ℤ) : 
  n = k * (k + 1) * (k + 2) → 
  11 ∣ n → 
  ¬ (∀ m : ℤ, n = k * (k + 1) * (k + 2) ∧ 11 ∣ n → 44 ∣ n) :=
by sorry

end not_necessarily_divisible_by_44_l2030_203042


namespace inequality_range_l2030_203087

theorem inequality_range (a : ℝ) : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π / 2 → 
    Real.sin (2 * θ) - (2 * Real.sqrt 2 + Real.sqrt 2 * a) * Real.sin (θ + π / 4) - 
    (2 * Real.sqrt 2 / Real.cos (θ - π / 4)) > -3 - 2 * a) → 
  a > 3 := by
sorry

end inequality_range_l2030_203087


namespace norm_took_110_photos_l2030_203063

/-- The number of photos taken by Norm given the conditions of the problem -/
def norm_photos (lisa mike norm : ℕ) : Prop :=
  (lisa + mike = mike + norm - 60) ∧ 
  (norm = 2 * lisa + 10) ∧
  (norm = 110)

/-- Theorem stating that Norm took 110 photos given the problem conditions -/
theorem norm_took_110_photos :
  ∃ (lisa mike norm : ℕ), norm_photos lisa mike norm :=
by
  sorry

end norm_took_110_photos_l2030_203063


namespace pizza_order_l2030_203032

theorem pizza_order (people : ℕ) (slices_per_person : ℕ) (slices_per_pizza : ℕ) 
  (h1 : people = 18) 
  (h2 : slices_per_person = 3) 
  (h3 : slices_per_pizza = 9) : 
  (people * slices_per_person) / slices_per_pizza = 6 := by
  sorry

end pizza_order_l2030_203032


namespace sea_glass_ratio_l2030_203041

/-- Sea glass collection problem -/
theorem sea_glass_ratio : 
  ∀ (blanche_green blanche_red rose_red rose_blue dorothy_total : ℕ),
  blanche_green = 12 →
  blanche_red = 3 →
  rose_red = 9 →
  rose_blue = 11 →
  dorothy_total = 57 →
  ∃ (dorothy_red dorothy_blue : ℕ),
    dorothy_blue = 3 * rose_blue ∧
    dorothy_red + dorothy_blue = dorothy_total ∧
    2 * (blanche_red + rose_red) = dorothy_red :=
by sorry

end sea_glass_ratio_l2030_203041


namespace investment_plans_count_l2030_203084

/-- The number of ways to distribute projects across cities -/
def distribute_projects (num_projects : ℕ) (num_cities : ℕ) (max_per_city : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of investment plans -/
theorem investment_plans_count : distribute_projects 3 4 2 = 16 := by
  sorry

end investment_plans_count_l2030_203084


namespace three_equal_numbers_sum_300_l2030_203060

theorem three_equal_numbers_sum_300 :
  ∃ (x : ℕ), x + x + x = 300 ∧ x = 100 := by
  sorry

end three_equal_numbers_sum_300_l2030_203060


namespace harry_age_l2030_203016

/-- Represents the ages of the people in the problem -/
structure Ages where
  kiarra : ℕ
  bea : ℕ
  job : ℕ
  figaro : ℕ
  harry : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.kiarra = 2 * ages.bea ∧
  ages.job = 3 * ages.bea ∧
  ages.figaro = ages.job + 7 ∧
  2 * ages.harry = ages.figaro ∧
  ages.kiarra = 30

/-- The theorem stating that under the given conditions, Harry's age is 26 -/
theorem harry_age (ages : Ages) :
  problem_conditions ages → ages.harry = 26 := by
  sorry

end harry_age_l2030_203016


namespace two_white_balls_probability_l2030_203030

/-- The probability of drawing two white balls without replacement from a box containing 
    8 white balls and 10 black balls is 28/153. -/
theorem two_white_balls_probability (white_balls black_balls : ℕ) 
    (h1 : white_balls = 8) (h2 : black_balls = 10) :
  let total_balls := white_balls + black_balls
  let prob_first_white := white_balls / total_balls
  let prob_second_white := (white_balls - 1) / (total_balls - 1)
  prob_first_white * prob_second_white = 28 / 153 := by
  sorry

end two_white_balls_probability_l2030_203030


namespace geometric_sequence_common_ratio_l2030_203019

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_incr : is_increasing_sequence a)
  (h_pos : a 1 > 0)
  (h_eq : ∀ n : ℕ, 2 * (a n + a (n + 2)) = 5 * a (n + 1)) :
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 2 := by
  sorry

end geometric_sequence_common_ratio_l2030_203019


namespace regular_star_points_l2030_203006

/-- An n-pointed regular star polygon -/
structure RegularStar where
  n : ℕ
  angle_A : ℝ
  angle_B : ℝ
  h1 : angle_A = angle_B - 15
  h2 : n * angle_A = n * angle_B - 180

theorem regular_star_points (star : RegularStar) : star.n = 12 := by
  sorry

end regular_star_points_l2030_203006


namespace probability_white_or_red_l2030_203097

def white_balls : ℕ := 7
def black_balls : ℕ := 8
def red_balls : ℕ := 5

def total_balls : ℕ := white_balls + black_balls + red_balls

def favorable_outcomes : ℕ := white_balls + red_balls

theorem probability_white_or_red :
  (favorable_outcomes : ℚ) / total_balls = 3 / 5 := by
  sorry

end probability_white_or_red_l2030_203097


namespace function_extrema_l2030_203088

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 4*x + 4

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 4

-- Theorem statement
theorem function_extrema (a : ℝ) : 
  (f' a 1 = -3) → 
  (a = 1/3) ∧ 
  (∀ x, f (1/3) x ≤ 28/3) ∧ 
  (∀ x, f (1/3) x ≥ -4/3) ∧
  (∃ x, f (1/3) x = 28/3) ∧ 
  (∃ x, f (1/3) x = -4/3) :=
sorry

end function_extrema_l2030_203088


namespace problem_solution_l2030_203057

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

theorem problem_solution :
  ∀ a b c : ℝ,
  (∀ x : ℝ, f a b c (-x) = -(f a b c x)) →
  f a b c 1 = 3 →
  f a b c 2 = 12 →
  (a = 1 ∧ b = 0 ∧ c = 2) ∧
  (∀ x y : ℝ, x < y → f a b c x < f a b c y) ∧
  (∀ m n : ℝ, m^3 - 3*m^2 + 5*m = 5 → n^3 - 3*n^2 + 5*n = 1 → m + n = 2) ∧
  (∀ k : ℝ, (∀ x : ℝ, 0 < x ∧ x < 1 → f a b c (x^2 - 4) + f a b c (k*x + 2*k) < 0) → k ≤ 1) :=
by sorry

end problem_solution_l2030_203057


namespace loes_speed_l2030_203026

/-- Proves that Loe's speed is 50 mph given the conditions of the problem -/
theorem loes_speed (teena_speed : ℝ) (initial_distance : ℝ) (time : ℝ) (final_distance : ℝ) :
  teena_speed = 55 →
  initial_distance = 7.5 →
  time = 1.5 →
  final_distance = 15 →
  ∃ (loe_speed : ℝ), loe_speed = 50 ∧
    teena_speed * time - loe_speed * time = final_distance + initial_distance :=
by sorry

end loes_speed_l2030_203026


namespace geometric_sequence_sum_l2030_203023

/-- Given a geometric sequence {a_n} with common ratio q and sum of first n terms S_n,
    prove that if q = 2 and S_5 = 1, then S_10 = 33. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- sum formula
  q = 2 →
  S 5 = 1 →
  S 10 = 33 := by
sorry

end geometric_sequence_sum_l2030_203023


namespace katya_age_l2030_203095

/-- Represents the ages of the children in the family -/
structure FamilyAges where
  anya : ℕ
  katya : ℕ
  vasya : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  ages.anya + ages.katya = 19 ∧
  ages.anya + ages.vasya = 14 ∧
  ages.katya + ages.vasya = 7

/-- The theorem to prove Katya's age -/
theorem katya_age (ages : FamilyAges) (h : satisfiesConditions ages) : ages.katya = 6 := by
  sorry


end katya_age_l2030_203095


namespace quadratic_form_h_l2030_203075

theorem quadratic_form_h (a k h : ℝ) : 
  (∀ x, 8 * x^2 + 12 * x + 7 = a * (x - h)^2 + k) → h = -3/4 := by
sorry

end quadratic_form_h_l2030_203075


namespace intersection_point_on_diagonal_l2030_203067

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Represents a plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- Check if a point lies on a line -/
def pointOnLine (p : Point3D) (l : Line3D) : Prop :=
  sorry

/-- Check if a point lies on a plane -/
def pointOnPlane (p : Point3D) (plane : Plane3D) : Prop :=
  sorry

/-- Check if two lines intersect -/
def linesIntersect (l1 l2 : Line3D) : Prop :=
  sorry

theorem intersection_point_on_diagonal (A B C D E F G H P : Point3D)
  (AB : Line3D) (BC : Line3D) (CD : Line3D) (DA : Line3D)
  (EF : Line3D) (GH : Line3D) (AC : Line3D)
  (ABC : Plane3D) (ADC : Plane3D) :
  pointOnLine E AB →
  pointOnLine F BC →
  pointOnLine G CD →
  pointOnLine H DA →
  linesIntersect EF GH →
  pointOnLine P EF →
  pointOnLine P GH →
  pointOnPlane E ABC →
  pointOnPlane F ABC →
  pointOnPlane G ADC →
  pointOnPlane H ADC →
  pointOnLine P AC :=
by sorry

end intersection_point_on_diagonal_l2030_203067


namespace specific_polyhedron_volume_l2030_203021

/-- A polyhedron formed by a unit square base and four points above its vertices -/
structure UnitSquarePolyhedron where
  -- Heights of the points above the unit square vertices
  h1 : ℝ
  h2 : ℝ
  h3 : ℝ
  h4 : ℝ

/-- The volume of the UnitSquarePolyhedron -/
def volume (p : UnitSquarePolyhedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific polyhedron is 4.5 -/
theorem specific_polyhedron_volume :
  ∃ (p : UnitSquarePolyhedron),
    p.h1 = 3 ∧ p.h2 = 4 ∧ p.h3 = 6 ∧ p.h4 = 5 ∧
    volume p = 4.5 :=
  sorry

end specific_polyhedron_volume_l2030_203021


namespace ellipse_constant_product_l2030_203089

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l
def l (k m x y : ℝ) : Prop := y = k*x + m

-- Define the intersection point Q
def Q (k m : ℝ) : ℝ × ℝ := (-4, k*(-4) + m)

-- Define the left focus F
def F : ℝ × ℝ := (-1, 0)

-- State the theorem
theorem ellipse_constant_product (k m : ℝ) (A B P : ℝ × ℝ) :
  E A.1 A.2 →
  E B.1 B.2 →
  E P.1 P.2 →
  l k m A.1 A.2 →
  l k m B.1 B.2 →
  P = (A.1 + B.1, A.2 + B.2) →
  (P.1 - F.1) * (Q k m).1 + (P.2 - F.2) * (Q k m).2 = 3/2 :=
sorry

end ellipse_constant_product_l2030_203089


namespace quadratic_completing_square_l2030_203098

/-- Given a quadratic equation x^2 - 6x + 4 = 0, 
    its equivalent form using the completing the square method is (x - 3)^2 = 5 -/
theorem quadratic_completing_square : 
  ∀ x : ℝ, x^2 - 6*x + 4 = 0 ↔ (x - 3)^2 = 5 := by sorry

end quadratic_completing_square_l2030_203098


namespace solution_set_part1_range_of_a_part2_l2030_203046

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end solution_set_part1_range_of_a_part2_l2030_203046


namespace homework_problem_ratio_l2030_203005

theorem homework_problem_ratio : 
  ∀ (total_problems : ℕ) 
    (martha_problems : ℕ) 
    (angela_problems : ℕ) 
    (jenna_problems : ℕ),
  total_problems = 20 →
  martha_problems = 2 →
  angela_problems = 9 →
  jenna_problems + martha_problems + (jenna_problems / 2) + angela_problems = total_problems →
  (jenna_problems : ℚ) / martha_problems = 3 := by
  sorry

end homework_problem_ratio_l2030_203005


namespace average_hours_worked_per_month_l2030_203011

def hours_per_day_april : ℕ := 6
def hours_per_day_june : ℕ := 5
def hours_per_day_september : ℕ := 8
def days_per_month : ℕ := 30
def num_months : ℕ := 3

theorem average_hours_worked_per_month :
  (hours_per_day_april * days_per_month +
   hours_per_day_june * days_per_month +
   hours_per_day_september * days_per_month) / num_months = 190 := by
  sorry

end average_hours_worked_per_month_l2030_203011
