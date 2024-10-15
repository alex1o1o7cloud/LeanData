import Mathlib

namespace NUMINAMATH_CALUDE_customers_not_buying_coffee_l3431_343193

theorem customers_not_buying_coffee (total_customers : ℕ) (coffee_fraction : ℚ) : 
  total_customers = 25 → coffee_fraction = 3/5 → 
  total_customers - (coffee_fraction * total_customers).floor = 10 :=
by sorry

end NUMINAMATH_CALUDE_customers_not_buying_coffee_l3431_343193


namespace NUMINAMATH_CALUDE_max_bing_games_and_wins_l3431_343105

/-- Represents a player in the table tennis game -/
inductive Player : Type
| jia : Player
| yi : Player
| bing : Player

/-- The game state, tracking the number of games played by each player -/
structure GameState :=
  (jia_games : ℕ)
  (yi_games : ℕ)
  (bing_games : ℕ)
  (bing_wins : ℕ)

/-- Checks if the game state is valid according to the rules -/
def is_valid_state (state : GameState) : Prop :=
  state.jia_games = 10 ∧ 
  state.yi_games = 7 ∧ 
  state.bing_games ≤ state.jia_games ∧ 
  state.bing_games ≤ state.yi_games + state.bing_wins ∧
  state.bing_wins ≤ state.bing_games

/-- The main theorem to prove -/
theorem max_bing_games_and_wins :
  ∃ (state : GameState), 
    is_valid_state state ∧ 
    ∀ (other_state : GameState), 
      is_valid_state other_state → 
      other_state.bing_games ≤ state.bing_games ∧
      other_state.bing_wins ≤ state.bing_wins ∧
      state.bing_games = 13 ∧
      state.bing_wins = 10 := by
  sorry

end NUMINAMATH_CALUDE_max_bing_games_and_wins_l3431_343105


namespace NUMINAMATH_CALUDE_pool_ground_area_l3431_343167

theorem pool_ground_area (length width : ℝ) (h1 : length = 5) (h2 : width = 4) :
  length * width = 20 := by
  sorry

end NUMINAMATH_CALUDE_pool_ground_area_l3431_343167


namespace NUMINAMATH_CALUDE_female_employees_with_advanced_degrees_l3431_343116

theorem female_employees_with_advanced_degrees
  (total_employees : ℕ)
  (total_females : ℕ)
  (total_advanced_degrees : ℕ)
  (males_college_only : ℕ)
  (h1 : total_employees = 200)
  (h2 : total_females = 120)
  (h3 : total_advanced_degrees = 100)
  (h4 : males_college_only = 40) :
  total_advanced_degrees - (total_employees - total_females - males_college_only) = 60 :=
by sorry

end NUMINAMATH_CALUDE_female_employees_with_advanced_degrees_l3431_343116


namespace NUMINAMATH_CALUDE_hat_price_after_discounts_l3431_343109

/-- The final price of an item after two successive discounts --/
def finalPrice (originalPrice : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  originalPrice * (1 - discount1) * (1 - discount2)

/-- Theorem stating that a $20 item with 20% and 25% successive discounts results in a $12 final price --/
theorem hat_price_after_discounts :
  finalPrice 20 0.2 0.25 = 12 := by
  sorry

end NUMINAMATH_CALUDE_hat_price_after_discounts_l3431_343109


namespace NUMINAMATH_CALUDE_apple_crate_weight_l3431_343117

/-- The weight of one original box of apples in kilograms. -/
def original_box_weight : ℝ := 35

/-- The number of crates in the original set. -/
def num_crates : ℕ := 7

/-- The amount of apples removed from each crate in kilograms. -/
def removed_weight : ℝ := 20

/-- The number of original crates that equal the weight of all crates after removal. -/
def equivalent_crates : ℕ := 3

theorem apple_crate_weight :
  num_crates * (original_box_weight - removed_weight) = equivalent_crates * original_box_weight :=
sorry

end NUMINAMATH_CALUDE_apple_crate_weight_l3431_343117


namespace NUMINAMATH_CALUDE_eggs_solution_l3431_343188

def eggs_problem (dozen_count : ℕ) (price_per_egg : ℚ) (tax_rate : ℚ) (discount_rate : ℚ) : ℚ :=
  let total_eggs := dozen_count * 12
  let original_cost := total_eggs * price_per_egg
  let discounted_cost := original_cost * (1 - discount_rate)
  let tax_amount := discounted_cost * tax_rate
  discounted_cost + tax_amount

theorem eggs_solution :
  eggs_problem 3 (1/2) (5/100) (10/100) = 1701/100 := by
  sorry

end NUMINAMATH_CALUDE_eggs_solution_l3431_343188


namespace NUMINAMATH_CALUDE_total_tires_changed_is_304_l3431_343175

/-- Represents the number of tires changed by Mike in a day -/
def total_tires_changed : ℕ :=
  let motorcycles := 12
  let cars := 10
  let bicycles := 8
  let trucks := 5
  let atvs := 7
  let dual_axle_trailers := 4
  let triple_axle_boat_trailers := 3
  let unicycles := 2
  let dually_pickup_trucks := 6

  let motorcycle_tires := 2
  let car_tires := 4
  let bicycle_tires := 2
  let truck_tires := 18
  let atv_tires := 4
  let dual_axle_trailer_tires := 8
  let triple_axle_boat_trailer_tires := 12
  let unicycle_tires := 1
  let dually_pickup_truck_tires := 6

  motorcycles * motorcycle_tires +
  cars * car_tires +
  bicycles * bicycle_tires +
  trucks * truck_tires +
  atvs * atv_tires +
  dual_axle_trailers * dual_axle_trailer_tires +
  triple_axle_boat_trailers * triple_axle_boat_trailer_tires +
  unicycles * unicycle_tires +
  dually_pickup_trucks * dually_pickup_truck_tires

/-- Theorem stating that the total number of tires changed by Mike in a day is 304 -/
theorem total_tires_changed_is_304 : total_tires_changed = 304 := by
  sorry

end NUMINAMATH_CALUDE_total_tires_changed_is_304_l3431_343175


namespace NUMINAMATH_CALUDE_new_cards_for_500_l3431_343190

/-- Given a total number of cards, calculate the number of new cards received
    when trading one-fifth of the duplicate cards, where duplicates are one-fourth
    of the total. -/
def new_cards_received (total : ℕ) : ℕ :=
  (total / 4) / 5

/-- Theorem stating that given 500 total cards, the number of new cards
    received is 25. -/
theorem new_cards_for_500 : new_cards_received 500 = 25 := by
  sorry

end NUMINAMATH_CALUDE_new_cards_for_500_l3431_343190


namespace NUMINAMATH_CALUDE_max_area_rectangular_pen_l3431_343112

/-- The maximum area of a rectangular pen with a perimeter of 60 feet -/
theorem max_area_rectangular_pen :
  let perimeter : ℝ := 60
  let area (x : ℝ) : ℝ := x * (perimeter / 2 - x)
  ∀ x, 0 < x → x < perimeter / 2 → area x ≤ 225 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangular_pen_l3431_343112


namespace NUMINAMATH_CALUDE_weekly_payment_problem_l3431_343181

/-- The weekly payment problem -/
theorem weekly_payment_problem (payment_B : ℝ) (payment_ratio : ℝ) : 
  payment_B = 180 →
  payment_ratio = 1.5 →
  payment_B + payment_ratio * payment_B = 450 := by
  sorry

end NUMINAMATH_CALUDE_weekly_payment_problem_l3431_343181


namespace NUMINAMATH_CALUDE_benjamins_dinner_cost_l3431_343124

-- Define the prices of items
def burger_price : ℕ := 5
def fries_price : ℕ := 2
def salad_price : ℕ := 3 * fries_price

-- Define the quantities of items
def burger_quantity : ℕ := 1
def fries_quantity : ℕ := 2
def salad_quantity : ℕ := 1

-- Define the total cost function
def total_cost : ℕ := 
  burger_price * burger_quantity + 
  fries_price * fries_quantity + 
  salad_price * salad_quantity

-- Theorem statement
theorem benjamins_dinner_cost : total_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_benjamins_dinner_cost_l3431_343124


namespace NUMINAMATH_CALUDE_subtraction_difference_l3431_343140

theorem subtraction_difference : 
  let total : ℝ := 7000
  let one_tenth : ℝ := 1 / 10
  let one_tenth_percent : ℝ := 1 / 1000
  (one_tenth * total) - (one_tenth_percent * total) = 693 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_difference_l3431_343140


namespace NUMINAMATH_CALUDE_min_sticks_for_13_triangles_l3431_343198

/-- The minimum number of sticks needed to form n equilateral triangles -/
def min_sticks (n : ℕ) : ℕ := 2 * n + 1

/-- Theorem: Given the conditions for forming 1, 2, and 3 equilateral triangles,
    the minimum number of sticks required to form 13 equilateral triangles is 27 -/
theorem min_sticks_for_13_triangles :
  (min_sticks 1 = 3) →
  (min_sticks 2 = 5) →
  (min_sticks 3 = 7) →
  min_sticks 13 = 27 := by
  sorry

end NUMINAMATH_CALUDE_min_sticks_for_13_triangles_l3431_343198


namespace NUMINAMATH_CALUDE_cube_zero_of_fourth_power_zero_l3431_343138

theorem cube_zero_of_fourth_power_zero (A : Matrix (Fin 3) (Fin 3) ℝ) 
  (h : A ^ 4 = 0) : A ^ 3 = 0 := by sorry

end NUMINAMATH_CALUDE_cube_zero_of_fourth_power_zero_l3431_343138


namespace NUMINAMATH_CALUDE_cafe_meal_cost_l3431_343122

theorem cafe_meal_cost (s c k : ℝ) : 
  (2 * s + 5 * c + 2 * k = 6.50) → 
  (3 * s + 8 * c + 3 * k = 10.20) → 
  (s + c + k = 1.90) :=
by
  sorry

end NUMINAMATH_CALUDE_cafe_meal_cost_l3431_343122


namespace NUMINAMATH_CALUDE_three_valid_pairs_l3431_343137

/-- The number of ordered pairs (a, b) satisfying the floor painting conditions -/
def num_valid_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    let a := p.1
    let b := p.2
    b > a ∧ (a - 4) * (b - 4) = a * b / 3
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- Theorem stating that there are exactly 3 valid pairs -/
theorem three_valid_pairs : num_valid_pairs = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_valid_pairs_l3431_343137


namespace NUMINAMATH_CALUDE_sphere_to_great_circle_area_ratio_l3431_343155

/-- The ratio of the area of a sphere to the area of its great circle is 4 -/
theorem sphere_to_great_circle_area_ratio :
  ∀ (R : ℝ), R > 0 →
  (4 * π * R^2) / (π * R^2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_sphere_to_great_circle_area_ratio_l3431_343155


namespace NUMINAMATH_CALUDE_total_weekly_revenue_l3431_343153

def normal_price : ℝ := 5

def monday_sales : ℕ := 9
def tuesday_sales : ℕ := 12
def wednesday_sales : ℕ := 18
def thursday_sales : ℕ := 14
def friday_sales : ℕ := 16
def saturday_sales : ℕ := 20
def sunday_sales : ℕ := 11

def wednesday_discount : ℝ := 0.1
def friday_discount : ℝ := 0.05

def daily_revenue (sales : ℕ) (discount : ℝ) : ℝ :=
  (sales : ℝ) * normal_price * (1 - discount)

theorem total_weekly_revenue :
  daily_revenue monday_sales 0 +
  daily_revenue tuesday_sales 0 +
  daily_revenue wednesday_sales wednesday_discount +
  daily_revenue thursday_sales 0 +
  daily_revenue friday_sales friday_discount +
  daily_revenue saturday_sales 0 +
  daily_revenue sunday_sales 0 = 487 := by
  sorry

end NUMINAMATH_CALUDE_total_weekly_revenue_l3431_343153


namespace NUMINAMATH_CALUDE_infinitely_many_friendly_squares_l3431_343182

/-- A number is friendly if the set {1, 2, ..., N} can be partitioned into disjoint pairs 
    where the sum of each pair is a perfect square -/
def is_friendly (N : ℕ) : Prop :=
  ∃ (partition : List (ℕ × ℕ)), 
    (∀ (pair : ℕ × ℕ), pair ∈ partition → pair.1 ∈ Finset.range N ∧ pair.2 ∈ Finset.range N) ∧
    (∀ n ∈ Finset.range N, ∃ (pair : ℕ × ℕ), pair ∈ partition ∧ (n = pair.1 ∨ n = pair.2)) ∧
    (∀ (pair : ℕ × ℕ), pair ∈ partition → ∃ k : ℕ, pair.1 + pair.2 = k^2)

/-- There are infinitely many friendly perfect squares -/
theorem infinitely_many_friendly_squares :
  ∀ (p : ℕ), p ≥ 2 → ∃ (N : ℕ), N = 2^(2*p - 3) ∧ is_friendly N ∧ ∃ (k : ℕ), N = k^2 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_friendly_squares_l3431_343182


namespace NUMINAMATH_CALUDE_freds_basketball_games_l3431_343101

theorem freds_basketball_games 
  (missed_this_year : ℕ) 
  (attended_last_year : ℕ) 
  (total_attended : ℕ) 
  (h1 : missed_this_year = 35)
  (h2 : attended_last_year = 11)
  (h3 : total_attended = 47) :
  total_attended - attended_last_year = 36 :=
by sorry

end NUMINAMATH_CALUDE_freds_basketball_games_l3431_343101


namespace NUMINAMATH_CALUDE_race_distance_l3431_343165

theorem race_distance (d : ℝ) 
  (h1 : ∃ x y : ℝ, x > y ∧ d / x = (d - 25) / y)
  (h2 : ∃ y z : ℝ, y > z ∧ d / y = (d - 15) / z)
  (h3 : ∃ x z : ℝ, x > z ∧ d / x = (d - 35) / z)
  : d = 75 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_l3431_343165


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l3431_343125

theorem solution_set_equivalence (x : ℝ) :
  (Real.log (|x - π/3|) / Real.log (1/2) ≥ Real.log (π/2) / Real.log (1/2)) ↔
  (-π/6 ≤ x ∧ x ≤ 5*π/6 ∧ x ≠ π/3) :=
sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l3431_343125


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3431_343118

/-- Given that the solution set of ax² - bx + c < 0 is (-2, 3), 
    prove that the solution set of bx² + ax + c < 0 is (-3, 2) -/
theorem solution_set_inequality (a b c : ℝ) : 
  (∀ x, ax^2 - b*x + c < 0 ↔ -2 < x ∧ x < 3) →
  (∀ x, b*x^2 + a*x + c < 0 ↔ -3 < x ∧ x < 2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3431_343118


namespace NUMINAMATH_CALUDE_class_size_problem_l3431_343144

theorem class_size_problem (n : ℕ) 
  (h1 : 20 ≤ n ∧ n ≤ 30) 
  (h2 : ∃ x y : ℕ, x < n ∧ y < n ∧ x ≠ y ∧ 2 * x + 1 = n - x ∧ 3 * y + 1 = n - y) :
  n = 25 := by
sorry

end NUMINAMATH_CALUDE_class_size_problem_l3431_343144


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3431_343199

theorem tan_alpha_value (α : Real) 
  (h : (Real.sin α - 2 * Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = -5) :
  Real.tan α = -23/16 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3431_343199


namespace NUMINAMATH_CALUDE_cone_volume_from_sector_l3431_343163

/-- Given a cone whose lateral surface develops into a sector with central angle 4π/3 and area 6π,
    the volume of the cone is (4√5/3)π. -/
theorem cone_volume_from_sector (θ r l h V : ℝ) : 
  θ = (4 / 3) * Real.pi →  -- Central angle of the sector
  (1 / 2) * l^2 * θ = 6 * Real.pi →  -- Area of the sector
  2 * Real.pi * r = θ * l →  -- Circumference of base equals arc length of sector
  h^2 + r^2 = l^2 →  -- Pythagorean theorem for cone dimensions
  V = (1 / 3) * Real.pi * r^2 * h →  -- Volume formula for cone
  V = (4 * Real.sqrt 5 / 3) * Real.pi := by
sorry

end NUMINAMATH_CALUDE_cone_volume_from_sector_l3431_343163


namespace NUMINAMATH_CALUDE_trapezoid_area_l3431_343154

-- Define the trapezoid ABCD
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  height : ℝ

-- Define the circle γ
structure Circle where
  radius : ℝ
  center_in_trapezoid : Bool
  tangent_to_AB_BC_DA : Bool
  arc_angle : ℝ

-- Define the problem
def trapezoid_circle_problem (ABCD : Trapezoid) (γ : Circle) : Prop :=
  ABCD.AB = 10 ∧
  ABCD.CD = 15 ∧
  γ.radius = 6 ∧
  γ.center_in_trapezoid = true ∧
  γ.tangent_to_AB_BC_DA = true ∧
  γ.arc_angle = 120

-- Theorem statement
theorem trapezoid_area (ABCD : Trapezoid) (γ : Circle) :
  trapezoid_circle_problem ABCD γ →
  (ABCD.AB + ABCD.CD) * ABCD.height / 2 = 225 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l3431_343154


namespace NUMINAMATH_CALUDE_casper_candies_proof_l3431_343189

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

end NUMINAMATH_CALUDE_casper_candies_proof_l3431_343189


namespace NUMINAMATH_CALUDE_largest_n_for_square_sum_l3431_343106

theorem largest_n_for_square_sum : ∃ (m : ℕ), 
  (4^995 + 4^1500 + 4^2004 = m^2) ∧ 
  (∀ (k : ℕ), k > 2004 → ¬∃ (l : ℕ), 4^995 + 4^1500 + 4^k = l^2) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_square_sum_l3431_343106


namespace NUMINAMATH_CALUDE_trains_meet_time_l3431_343173

/-- Two trains moving towards each other on a straight track meet at 10 a.m. -/
theorem trains_meet_time :
  -- Define the distance between stations P and Q
  let distance_PQ : ℝ := 110

  -- Define the speed of the first train
  let speed_train1 : ℝ := 20

  -- Define the speed of the second train
  let speed_train2 : ℝ := 25

  -- Define the time difference between the starts of the two trains (in hours)
  let time_diff : ℝ := 1

  -- Define the start time of the second train
  let start_time_train2 : ℝ := 8

  -- The time when the trains meet (in hours after midnight)
  let meet_time : ℝ := start_time_train2 + 
    (distance_PQ - speed_train1 * time_diff) / (speed_train1 + speed_train2)

  -- Prove that the meet time is 10 a.m.
  meet_time = 10 := by sorry

end NUMINAMATH_CALUDE_trains_meet_time_l3431_343173


namespace NUMINAMATH_CALUDE_x_power_minus_reciprocal_l3431_343127

theorem x_power_minus_reciprocal (φ : ℝ) (x : ℂ) (n : ℕ) 
  (h1 : 0 < φ) (h2 : φ < π) 
  (h3 : x - 1 / x = (2 * Complex.I * Real.sin φ))
  (h4 : Odd n) :
  x^n - 1 / x^n = 2 * Complex.I^n * (Real.sin φ)^n :=
by sorry

end NUMINAMATH_CALUDE_x_power_minus_reciprocal_l3431_343127


namespace NUMINAMATH_CALUDE_max_value_3m_4n_l3431_343133

theorem max_value_3m_4n (m n : ℕ) (h_sum : m * (m + 1) + n^2 ≤ 1987) (h_n_odd : Odd n) :
  3 * m + 4 * n ≤ 221 :=
sorry

end NUMINAMATH_CALUDE_max_value_3m_4n_l3431_343133


namespace NUMINAMATH_CALUDE_rectangle_packing_l3431_343195

/-- Represents the maximum number of non-overlapping 2-by-3 rectangles
    that can be placed in an m-by-n rectangle -/
def max_rectangles (m n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of 2-by-3 rectangles
    that can be placed in an m-by-n rectangle is at least ⌊mn/6⌋ -/
theorem rectangle_packing (m n : ℕ) (hm : m > 1) (hn : n > 1) :
  max_rectangles m n ≥ (m * n) / 6 :=
sorry

end NUMINAMATH_CALUDE_rectangle_packing_l3431_343195


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l3431_343185

theorem geometric_sequence_second_term (a₁ a₃ : ℝ) (h₁ : a₁ = 180) (h₃ : a₃ = 75 / 32) :
  ∃ b : ℝ, b > 0 ∧ b^2 = 421.875 ∧ ∃ r : ℝ, a₁ * r = b ∧ b * r = a₃ :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l3431_343185


namespace NUMINAMATH_CALUDE_vakha_always_wins_l3431_343197

/-- Represents a point on the circle -/
structure Point where
  index : Fin 99

/-- Represents a color (Red or Blue) -/
inductive Color
  | Red
  | Blue

/-- Represents the game state -/
structure GameState where
  coloredPoints : Fin 99 → Option Color

/-- Represents a player (Bjorn or Vakha) -/
inductive Player
  | Bjorn
  | Vakha

/-- Checks if three points form an equilateral triangle -/
def isEquilateralTriangle (p1 p2 p3 : Point) : Prop :=
  (p2.index - p1.index) % 33 = 0 ∧
  (p3.index - p2.index) % 33 = 0 ∧
  (p1.index - p3.index) % 33 = 0

/-- Checks if a monochromatic equilateral triangle exists in the game state -/
def existsMonochromaticTriangle (state : GameState) : Prop :=
  ∃ (p1 p2 p3 : Point) (c : Color),
    isEquilateralTriangle p1 p2 p3 ∧
    state.coloredPoints p1.index = some c ∧
    state.coloredPoints p2.index = some c ∧
    state.coloredPoints p3.index = some c

/-- Represents a valid move in the game -/
def validMove (state : GameState) (p : Point) (c : Color) : Prop :=
  state.coloredPoints p.index = none ∧
  (∃ (q : Point), state.coloredPoints q.index ≠ none ∧ (q.index + 1 = p.index ∨ q.index = p.index + 1))

/-- Represents a winning strategy for Vakha -/
def hasWinningStrategy (player : Player) : Prop :=
  ∀ (initialState : GameState),
    ∃ (finalState : GameState),
      (∀ (p : Point) (c : Color), validMove initialState p c → 
        ∃ (nextState : GameState), validMove nextState p c) ∧
      existsMonochromaticTriangle finalState

/-- The main theorem: Vakha always has a winning strategy -/
theorem vakha_always_wins : hasWinningStrategy Player.Vakha := by
  sorry

end NUMINAMATH_CALUDE_vakha_always_wins_l3431_343197


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3431_343146

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 - 2*y + m = 0 → y = x) → 
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3431_343146


namespace NUMINAMATH_CALUDE_milk_packet_content_l3431_343141

theorem milk_packet_content 
  (num_packets : ℕ) 
  (oz_to_ml : ℝ) 
  (total_oz : ℝ) 
  (h1 : num_packets = 150)
  (h2 : oz_to_ml = 30)
  (h3 : total_oz = 1250) :
  (total_oz * oz_to_ml) / num_packets = 250 := by
sorry

end NUMINAMATH_CALUDE_milk_packet_content_l3431_343141


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l3431_343111

theorem complex_fraction_sum (a b : ℝ) : 
  (1 + 2*I) / (1 + I) = a + b*I → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l3431_343111


namespace NUMINAMATH_CALUDE_expression_factorization_l3431_343179

theorem expression_factorization (x : ℝ) :
  (8 * x^6 + 36 * x^4 - 5) - (2 * x^6 - 6 * x^4 + 5) = 2 * (3 * x^6 + 21 * x^4 - 5) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3431_343179


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3431_343110

/-- Given a geometric sequence with positive terms and common ratio q,
    where S_n denotes the sum of the first n terms, prove that
    if 2^10 * S_30 + S_10 = (2^10 + 1) * S_20, then q = 1/2 -/
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_geometric : ∀ n, a (n + 1) = q * a n)
  (S : ℕ → ℝ)
  (h_sum : ∀ n, S n = (a 0) * (1 - q^n) / (1 - q))
  (h_equation : 2^10 * S 30 + S 10 = (2^10 + 1) * S 20) :
  q = 1/2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3431_343110


namespace NUMINAMATH_CALUDE_no_solution_iff_m_eq_seven_l3431_343129

theorem no_solution_iff_m_eq_seven (m : ℝ) : 
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) ≠ (x - m) / (x - 8)) ↔ m = 7 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_eq_seven_l3431_343129


namespace NUMINAMATH_CALUDE_vector_decomposition_l3431_343183

/-- Prove that the given vector x can be decomposed in terms of vectors p, q, and r -/
theorem vector_decomposition (x p q r : ℝ × ℝ × ℝ) : 
  x = (15, -20, -1) → 
  p = (0, 2, 1) → 
  q = (0, 1, -1) → 
  r = (5, -3, 2) → 
  x = (-6 : ℝ) • p + (1 : ℝ) • q + (3 : ℝ) • r :=
by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l3431_343183


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l3431_343169

theorem ratio_x_to_y (x y : ℚ) (h : (12 * x - 5 * y) / (16 * x - 3 * y) = 4 / 7) :
  x / y = 23 / 20 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l3431_343169


namespace NUMINAMATH_CALUDE_product_of_numbers_l3431_343158

theorem product_of_numbers (x y : ℚ) : 
  (- x = 3 / 4) → (y = x - 1 / 2) → (x * y = 15 / 16) := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3431_343158


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3431_343108

/-- An isosceles triangle with sides of 4cm and 3cm has a perimeter of either 10cm or 11cm. -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Triangle inequality
  (a = 4 ∧ b = 3) ∨ (a = 3 ∧ b = 4) →  -- Given side lengths
  ((a = b ∧ c = 3) ∨ (a = c ∧ b = 3)) →  -- Isosceles condition
  a + b + c = 10 ∨ a + b + c = 11 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3431_343108


namespace NUMINAMATH_CALUDE_grape_juice_mixture_proof_l3431_343168

/-- Proves that adding 10 gallons of grape juice to 40 gallons of a mixture 
    containing 10% grape juice results in a new mixture with 28.000000000000004% grape juice. -/
theorem grape_juice_mixture_proof : 
  let initial_mixture : ℝ := 40
  let initial_concentration : ℝ := 0.1
  let added_juice : ℝ := 10
  let final_concentration : ℝ := 0.28000000000000004
  (initial_mixture * initial_concentration + added_juice) / (initial_mixture + added_juice) = final_concentration := by
  sorry

end NUMINAMATH_CALUDE_grape_juice_mixture_proof_l3431_343168


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l3431_343176

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

theorem vector_difference_magnitude : ‖a - b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l3431_343176


namespace NUMINAMATH_CALUDE_gorilla_exhibit_visitors_l3431_343121

def visitors_per_hour : ℕ := 50
def open_hours : ℕ := 8
def gorilla_exhibit_percentage : ℚ := 4/5

theorem gorilla_exhibit_visitors :
  (visitors_per_hour * open_hours : ℚ) * gorilla_exhibit_percentage = 320 := by
  sorry

end NUMINAMATH_CALUDE_gorilla_exhibit_visitors_l3431_343121


namespace NUMINAMATH_CALUDE_sum_reciprocals_lower_bound_l3431_343196

theorem sum_reciprocals_lower_bound 
  (a b c d m : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hm : m > 0)
  (eq1 : 1/a = (a + b + c + d + m)/a)
  (eq2 : 1/b = (a + b + c + d + m)/b)
  (eq3 : 1/c = (a + b + c + d + m)/c)
  (eq4 : 1/d = (a + b + c + d + m)/d)
  (eq5 : 1/m = (a + b + c + d + m)/m) :
  1/a + 1/b + 1/c + 1/d + 1/m ≥ 25 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_lower_bound_l3431_343196


namespace NUMINAMATH_CALUDE_f_max_values_l3431_343145

noncomputable def f (x θ : Real) : Real :=
  Real.sin x ^ 2 + Real.sqrt 3 * Real.tan θ * Real.cos x + (Real.sqrt 3 / 8) * Real.tan θ - 3/2

theorem f_max_values (θ : Real) (h : θ ∈ Set.Icc 0 (Real.pi / 3)) :
  (∃ (x : Real), f x (Real.pi / 3) ≤ f x (Real.pi / 3) ∧ f x (Real.pi / 3) = 15/8) ∧
  (∃ (θ' : Real) (h' : θ' ∈ Set.Icc 0 (Real.pi / 3)), 
    (∃ (x : Real), ∀ (y : Real), f y θ' ≤ f x θ' ∧ f x θ' = -1/8) ∧ 
    θ' = Real.pi / 6) :=
by sorry

end NUMINAMATH_CALUDE_f_max_values_l3431_343145


namespace NUMINAMATH_CALUDE_angle_between_c_and_a_plus_b_is_zero_l3431_343161

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem angle_between_c_and_a_plus_b_is_zero
  (a b c : V)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hab : ‖a‖ = ‖b‖)
  (habgt : ‖a‖ > ‖a + b‖)
  (hc_eq : ‖c‖ = ‖a + b‖) :
  Real.arccos (inner c (a + b) / (‖c‖ * ‖a + b‖)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_c_and_a_plus_b_is_zero_l3431_343161


namespace NUMINAMATH_CALUDE_perpendicular_lines_and_intersection_l3431_343103

-- Define the four lines
def line1 (x y : ℚ) : Prop := 4 * y - 3 * x = 15
def line2 (x y : ℚ) : Prop := -3 * x - 4 * y = 15
def line3 (x y : ℚ) : Prop := 4 * y + 3 * x = 15
def line4 (x y : ℚ) : Prop := 3 * y + 4 * x = 15

-- Define perpendicularity
def perpendicular (f g : ℚ → ℚ → Prop) : Prop :=
  ∃ m1 m2 : ℚ, (∀ x y, f x y ↔ y = m1 * x + (15 / 4)) ∧
             (∀ x y, g x y ↔ y = m2 * x + 5) ∧
             m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines_and_intersection :
  perpendicular line1 line4 ∧
  line1 (15/32) (35/8) ∧
  line4 (15/32) (35/8) := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_and_intersection_l3431_343103


namespace NUMINAMATH_CALUDE_range_of_k_equation_of_l_when_OB_twice_OA_l3431_343156

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 6)^2 + y^2 = 20

-- Define the line l
def line_l (k x y : ℝ) : Prop := y = k * x

-- Define the condition that line l intersects circle C at two distinct points
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ line_l k x₁ y₁ ∧ line_l k x₂ y₂

-- Define the condition OB = 2OA
def OB_twice_OA (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₂^2 + y₂^2 = 4 * (x₁^2 + y₁^2)

-- Theorem for the range of k
theorem range_of_k (k : ℝ) :
  intersects_at_two_points k ↔ -Real.sqrt 5 / 2 < k ∧ k < Real.sqrt 5 / 2 :=
sorry

-- Theorem for the equation of line l when OB = 2OA
theorem equation_of_l_when_OB_twice_OA (k : ℝ) :
  (intersects_at_two_points k ∧
   ∃ (x₁ y₁ x₂ y₂ : ℝ), circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧ OB_twice_OA x₁ y₁ x₂ y₂)
  → k = 1 ∨ k = -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_k_equation_of_l_when_OB_twice_OA_l3431_343156


namespace NUMINAMATH_CALUDE_circle_equation_proof_l3431_343135

-- Define a circle in R^2
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define when a circle is tangent to the x-axis
def TangentToXAxis (c : Set (ℝ × ℝ)) : Prop :=
  ∃ x : ℝ, (x, 0) ∈ c ∧ ∀ y : ℝ, y ≠ 0 → (x, y) ∉ c

theorem circle_equation_proof :
  let c : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + (p.2 - 2)^2 = 4}
  c = Circle (0, 2) 2 ∧ TangentToXAxis c := by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l3431_343135


namespace NUMINAMATH_CALUDE_bob_improvement_percentage_l3431_343151

def bob_time : ℝ := 640
def sister_time : ℝ := 557

theorem bob_improvement_percentage :
  let time_difference := bob_time - sister_time
  let percentage_improvement := (time_difference / bob_time) * 100
  ∃ ε > 0, abs (percentage_improvement - 12.97) < ε :=
by sorry

end NUMINAMATH_CALUDE_bob_improvement_percentage_l3431_343151


namespace NUMINAMATH_CALUDE_jackson_money_l3431_343152

/-- Proves that given two people where one has 5 times more money than the other, 
    and together they have $150, the person with more money has $125. -/
theorem jackson_money (williams_money : ℝ) 
  (h1 : williams_money + 5 * williams_money = 150) : 
  5 * williams_money = 125 := by
  sorry

end NUMINAMATH_CALUDE_jackson_money_l3431_343152


namespace NUMINAMATH_CALUDE_largest_number_with_same_quotient_and_remainder_l3431_343164

theorem largest_number_with_same_quotient_and_remainder : ∃ (n : ℕ), n = 90 ∧
  (∀ m : ℕ, m > n →
    ¬(∃ (q r : ℕ), m = 13 * q + r ∧ m = 15 * q + r ∧ r < 13 ∧ r < 15)) ∧
  (∃ (q r : ℕ), n = 13 * q + r ∧ n = 15 * q + r ∧ r < 13 ∧ r < 15) :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_same_quotient_and_remainder_l3431_343164


namespace NUMINAMATH_CALUDE_simplify_fraction_l3431_343148

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3431_343148


namespace NUMINAMATH_CALUDE_square_equals_product_solution_l3431_343160

theorem square_equals_product_solution :
  ∀ a b : ℕ, a^2 = b * (b + 7) → (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) := by
  sorry

end NUMINAMATH_CALUDE_square_equals_product_solution_l3431_343160


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l3431_343134

/-- The sampling interval for systematic sampling -/
def sampling_interval (population_size : ℕ) (sample_size : ℕ) : ℕ :=
  population_size / sample_size

/-- Theorem: The sampling interval for a population of 1000 and sample size of 20 is 50 -/
theorem systematic_sampling_interval :
  sampling_interval 1000 20 = 50 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l3431_343134


namespace NUMINAMATH_CALUDE_negation_proposition_l3431_343132

theorem negation_proposition :
  (∀ x : ℝ, x < 0 → x^2 ≤ 0) ↔ ¬(∃ x₀ : ℝ, x₀ < 0 ∧ x₀^2 > 0) :=
sorry

end NUMINAMATH_CALUDE_negation_proposition_l3431_343132


namespace NUMINAMATH_CALUDE_polynomial_equation_solution_l3431_343100

theorem polynomial_equation_solution (p : Polynomial ℝ) :
  (∀ x : ℝ, x ≠ 0 → p.eval x ^ 2 + p.eval (1 / x) ^ 2 = p.eval (x ^ 2) * p.eval (1 / x ^ 2)) →
  p = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equation_solution_l3431_343100


namespace NUMINAMATH_CALUDE_program_requires_eight_sessions_l3431_343113

/-- Calculates the number of seating sessions required for a group -/
def sessionsRequired (groupSize : ℕ) (capacity : ℕ) : ℕ :=
  (groupSize + capacity - 1) / capacity

/-- Represents the seating program -/
structure SeatingProgram where
  totalParents : ℕ
  totalPupils : ℕ
  capacity : ℕ
  parentsMorning : ℕ
  parentsAfternoon : ℕ
  pupilsMorning : ℕ
  pupilsMidDay : ℕ
  pupilsEvening : ℕ

/-- Calculates the total number of seating sessions required -/
def totalSessions (program : SeatingProgram) : ℕ :=
  sessionsRequired program.parentsMorning program.capacity +
  sessionsRequired program.parentsAfternoon program.capacity +
  sessionsRequired program.pupilsMorning program.capacity +
  sessionsRequired program.pupilsMidDay program.capacity +
  sessionsRequired program.pupilsEvening program.capacity

/-- Theorem stating that the given program requires 8 seating sessions -/
theorem program_requires_eight_sessions (program : SeatingProgram)
  (h1 : program.totalParents = 61)
  (h2 : program.totalPupils = 177)
  (h3 : program.capacity = 44)
  (h4 : program.parentsMorning = 35)
  (h5 : program.parentsAfternoon = 26)
  (h6 : program.pupilsMorning = 65)
  (h7 : program.pupilsMidDay = 57)
  (h8 : program.pupilsEvening = 55)
  : totalSessions program = 8 := by
  sorry


end NUMINAMATH_CALUDE_program_requires_eight_sessions_l3431_343113


namespace NUMINAMATH_CALUDE_dolls_count_l3431_343136

/-- Given that Hannah has 5 times as many dolls as her sister, and her sister has 8 dolls,
    prove that they have 48 dolls altogether. -/
theorem dolls_count (hannah_dolls : ℕ) (sister_dolls : ℕ) : 
  hannah_dolls = 5 * sister_dolls → sister_dolls = 8 → hannah_dolls + sister_dolls = 48 := by
  sorry

end NUMINAMATH_CALUDE_dolls_count_l3431_343136


namespace NUMINAMATH_CALUDE_city_distance_min_city_distance_l3431_343119

def is_valid_distance (S : ℕ) : Prop :=
  (∀ x : ℕ, x ≤ S → (Nat.gcd x (S - x) = 1 ∨ Nat.gcd x (S - x) = 3 ∨ Nat.gcd x (S - x) = 13)) ∧
  (∃ x : ℕ, x ≤ S ∧ Nat.gcd x (S - x) = 1) ∧
  (∃ x : ℕ, x ≤ S ∧ Nat.gcd x (S - x) = 3) ∧
  (∃ x : ℕ, x ≤ S ∧ Nat.gcd x (S - x) = 13)

theorem city_distance : 
  ∀ S : ℕ, is_valid_distance S → S ≥ 39 :=
by sorry

theorem min_city_distance :
  is_valid_distance 39 :=
by sorry

end NUMINAMATH_CALUDE_city_distance_min_city_distance_l3431_343119


namespace NUMINAMATH_CALUDE_function_inequality_l3431_343171

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, x * (deriv f x) + f x > 0) (a b : ℝ) (hab : a > b) : 
  a * f a > b * f b := by sorry

end NUMINAMATH_CALUDE_function_inequality_l3431_343171


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l3431_343174

-- Define the matrix expression evaluation rule
def matrix_value (a b c d : ℝ) : ℝ := a * b - c * d

-- Define the equation to solve
def equation (x : ℝ) : Prop :=
  matrix_value (3 * x) (x + 2) (x + 1) (2 * x) = 6

-- State the theorem
theorem matrix_equation_solution :
  ∃ x₁ x₂ : ℝ, x₁ = -2 + Real.sqrt 10 ∧ x₂ = -2 - Real.sqrt 10 ∧
  ∀ x : ℝ, equation x ↔ (x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l3431_343174


namespace NUMINAMATH_CALUDE_weighted_average_closer_to_larger_set_l3431_343107

theorem weighted_average_closer_to_larger_set 
  (set1 set2 : Finset ℝ) 
  (mean1 mean2 : ℝ) 
  (h_size : set1.card > set2.card) 
  (h_mean1 : mean1 = (set1.sum id) / set1.card) 
  (h_mean2 : mean2 = (set2.sum id) / set2.card) 
  (h_total_mean : (set1.sum id + set2.sum id) / (set1.card + set2.card) = 80) :
  |80 - mean1| < |80 - mean2| :=
sorry

end NUMINAMATH_CALUDE_weighted_average_closer_to_larger_set_l3431_343107


namespace NUMINAMATH_CALUDE_kenya_peanuts_l3431_343142

theorem kenya_peanuts (jose_peanuts : ℕ) (kenya_additional_peanuts : ℕ) 
  (h1 : jose_peanuts = 85)
  (h2 : kenya_additional_peanuts = 48) :
  jose_peanuts + kenya_additional_peanuts = 133 :=
by sorry

end NUMINAMATH_CALUDE_kenya_peanuts_l3431_343142


namespace NUMINAMATH_CALUDE_infinite_solutions_l3431_343150

/-- The equation (x-1)^2 + (x+1)^2 = y^2 + 1 -/
def is_solution (x y : ℕ) : Prop :=
  (x - 1)^2 + (x + 1)^2 = y^2 + 1

/-- The transformation function -/
def transform (x y : ℕ) : ℕ × ℕ :=
  (3*x + 2*y, 4*x + 3*y)

theorem infinite_solutions :
  (is_solution 0 1) ∧
  (is_solution 2 3) ∧
  (∀ x y : ℕ, is_solution x y → is_solution (transform x y).1 (transform x y).2) →
  ∃ f : ℕ → ℕ × ℕ, ∀ n : ℕ, is_solution (f n).1 (f n).2 ∧ f n ≠ f (n+1) :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_l3431_343150


namespace NUMINAMATH_CALUDE_triangle_side_length_l3431_343143

theorem triangle_side_length (B C BDC : Real) (BD : Real) :
  B = π/6 → -- 30°
  C = π/4 → -- 45°
  BDC = 5*π/6 → -- 150°
  BD = 5 →
  ∃ (AB : Real), AB = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3431_343143


namespace NUMINAMATH_CALUDE_max_m_value_l3431_343130

/-- Given a > 0, proves that the maximum value of m is e^(1/2) when the tangents of 
    y = x²/2 + ax and y = 2a²ln(x) + m coincide at their intersection point. -/
theorem max_m_value (a : ℝ) (h_a : a > 0) : 
  let C₁ : ℝ → ℝ := λ x => x^2 / 2 + a * x
  let C₂ : ℝ → ℝ → ℝ := λ x m => 2 * a^2 * Real.log x + m
  let tangent_C₁ : ℝ → ℝ := λ x => x + a
  let tangent_C₂ : ℝ → ℝ := λ x => 2 * a^2 / x
  ∃ x₀ m, C₁ x₀ = C₂ x₀ m ∧ tangent_C₁ x₀ = tangent_C₂ x₀ ∧ 
    (∀ m', C₁ x₀ = C₂ x₀ m' ∧ tangent_C₁ x₀ = tangent_C₂ x₀ → m' ≤ m) ∧
    m = Real.exp (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_max_m_value_l3431_343130


namespace NUMINAMATH_CALUDE_infinite_special_numbers_l3431_343180

theorem infinite_special_numbers (k : ℕ) :
  let n := 250 * 3^(6*k)
  ∃ (a b c d : ℕ), 
    n = a^2 + b^2 ∧ 
    n = c^3 + d^3 ∧ 
    ¬∃ (x y : ℕ), n = x^6 + y^6 := by
  sorry

end NUMINAMATH_CALUDE_infinite_special_numbers_l3431_343180


namespace NUMINAMATH_CALUDE_set_equality_l3431_343131

theorem set_equality : {p : ℝ × ℝ | p.1 + p.2 = 5 ∧ 2 * p.1 - p.2 = 1} = {(2, 3)} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l3431_343131


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l3431_343191

/-- 
Given two lines in the form of linear equations:
  3y - 2x - 6 = 0 and 4y + bx - 5 = 0
If these lines are perpendicular, then b = 6.
-/
theorem perpendicular_lines_b_value (b : ℝ) : 
  (∀ x y, 3 * y - 2 * x - 6 = 0 → 
           4 * y + b * x - 5 = 0 → 
           (2 / 3) * (-b / 4) = -1) → 
  b = 6 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l3431_343191


namespace NUMINAMATH_CALUDE_square_sequence_theorem_l3431_343104

/-- The number of nonoverlapping unit squares in the nth figure -/
def f (n : ℕ) : ℕ := 2 * n^2 + 4 * n + 3

/-- The theorem stating the properties of the sequence and the value for the 100th figure -/
theorem square_sequence_theorem :
  (f 0 = 3) ∧ (f 1 = 9) ∧ (f 2 = 19) ∧ (f 3 = 33) → f 100 = 20403 :=
by
  sorry

end NUMINAMATH_CALUDE_square_sequence_theorem_l3431_343104


namespace NUMINAMATH_CALUDE_am_gm_for_even_sum_l3431_343184

theorem am_gm_for_even_sum (a b : ℕ) (ha : a > 0) (hb : b > 0) (hsum : Even (a + b)) :
  (a + b : ℝ) / 2 ≥ Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_am_gm_for_even_sum_l3431_343184


namespace NUMINAMATH_CALUDE_field_trip_buses_l3431_343186

/-- Given a field trip scenario with vans and buses, calculate the number of buses required. -/
theorem field_trip_buses (total_people : ℕ) (num_vans : ℕ) (people_per_van : ℕ) (people_per_bus : ℕ)
  (h1 : total_people = 180)
  (h2 : num_vans = 6)
  (h3 : people_per_van = 6)
  (h4 : people_per_bus = 18) :
  (total_people - num_vans * people_per_van) / people_per_bus = 8 :=
by sorry

end NUMINAMATH_CALUDE_field_trip_buses_l3431_343186


namespace NUMINAMATH_CALUDE_sum_of_digits_after_addition_l3431_343157

def sum_of_digits (n : ℕ) : ℕ := sorry

def number_of_carries (a b : ℕ) : ℕ := sorry

theorem sum_of_digits_after_addition (A B : ℕ) 
  (hA : A > 0) 
  (hB : B > 0) 
  (hSumA : sum_of_digits A = 19) 
  (hSumB : sum_of_digits B = 20) 
  (hCarries : number_of_carries A B = 2) : 
  sum_of_digits (A + B) = 21 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_after_addition_l3431_343157


namespace NUMINAMATH_CALUDE_river_crossing_trips_l3431_343187

/-- Represents the number of trips required to transport one adult across the river -/
def trips_per_adult : ℕ := 4

/-- Represents the total number of adults to be transported -/
def total_adults : ℕ := 358

/-- Calculates the total number of trips required to transport all adults -/
def total_trips : ℕ := trips_per_adult * total_adults

/-- Theorem stating that the total number of trips is 1432 -/
theorem river_crossing_trips : total_trips = 1432 := by
  sorry

end NUMINAMATH_CALUDE_river_crossing_trips_l3431_343187


namespace NUMINAMATH_CALUDE_parabola_focus_l3431_343147

/-- The focus of a parabola y = ax^2 (a ≠ 0) is at (0, 1/(4a)) -/
theorem parabola_focus (a : ℝ) (h : a ≠ 0) :
  let parabola := {(x, y) : ℝ × ℝ | y = a * x^2}
  ∃ (focus : ℝ × ℝ), focus ∈ parabola ∧ focus = (0, 1 / (4 * a)) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l3431_343147


namespace NUMINAMATH_CALUDE_password_digit_l3431_343194

theorem password_digit (n : ℕ) : 
  n = 5678 * 6789 → 
  ∃ (a b c d e f g h i : ℕ),
    n = a * 10^8 + b * 10^7 + c * 10^6 + d * 10^5 + e * 10^4 + f * 10^3 + g * 10^2 + h * 10 + i ∧
    a = 3 ∧ b = 8 ∧ c = 5 ∧ d = 4 ∧ f = 9 ∧ g = 4 ∧ h = 2 ∧
    e = 7 :=
sorry

end NUMINAMATH_CALUDE_password_digit_l3431_343194


namespace NUMINAMATH_CALUDE_square_diff_equality_l3431_343139

theorem square_diff_equality (x y A : ℝ) : 
  (2*x - y)^2 + A = (2*x + y)^2 → A = 8*x*y := by
  sorry

end NUMINAMATH_CALUDE_square_diff_equality_l3431_343139


namespace NUMINAMATH_CALUDE_operational_probability_independent_of_root_l3431_343102

/-- Represents a computer network -/
structure ComputerNetwork where
  servers : Type
  channels : servers → servers → Prop
  failure_prob : ℝ
  failure_prob_nonneg : 0 ≤ failure_prob
  failure_prob_le_one : failure_prob ≤ 1

/-- Predicate to check if a server can reach another server using operating channels -/
def can_reach (G : ComputerNetwork) (s t : G.servers) : Prop :=
  sorry

/-- Predicate to check if a network is operational with respect to a root server -/
def is_operational (G : ComputerNetwork) (r : G.servers) : Prop :=
  ∀ s : G.servers, can_reach G s r

/-- The probability that a network is operational -/
noncomputable def operational_probability (G : ComputerNetwork) (r : G.servers) : ℝ :=
  sorry

/-- Theorem stating that the operational probability is independent of the choice of root server -/
theorem operational_probability_independent_of_root (G : ComputerNetwork) 
  (r₁ r₂ : G.servers) (h : r₁ ≠ r₂) : 
  operational_probability G r₁ = operational_probability G r₂ :=
sorry

end NUMINAMATH_CALUDE_operational_probability_independent_of_root_l3431_343102


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3431_343123

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x | -1 ≤ x ∧ x ≤ 1}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3431_343123


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3431_343178

/-- Given a principal amount and an unknown interest rate, 
    if increasing the rate by 8% for 15 years results in 2,750 more interest,
    then the principal amount is 2,291.67 -/
theorem simple_interest_problem (P R : ℝ) : 
  (P * R * 15 / 100 + 2750 = P * (R + 8) * 15 / 100) → 
  P = 2291.67 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3431_343178


namespace NUMINAMATH_CALUDE_nested_sqrt_equality_l3431_343126

theorem nested_sqrt_equality (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = x ^ (15/16) := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_equality_l3431_343126


namespace NUMINAMATH_CALUDE_truncated_pyramid_lateral_area_l3431_343166

/-- Represents a regular quadrangular pyramid -/
structure RegularQuadPyramid where
  base_side : ℝ
  height : ℝ

/-- Represents a truncated regular quadrangular pyramid -/
structure TruncatedRegularQuadPyramid where
  base_side : ℝ
  height : ℝ
  cut_height : ℝ

/-- Calculates the lateral surface area of a truncated regular quadrangular pyramid -/
def lateral_surface_area (t : TruncatedRegularQuadPyramid) : ℝ :=
  sorry

theorem truncated_pyramid_lateral_area :
  let p : RegularQuadPyramid := { base_side := 6, height := 4 }
  let t : TruncatedRegularQuadPyramid := { base_side := 6, height := 4, cut_height := 1 }
  lateral_surface_area t = 26.25 := by
  sorry

end NUMINAMATH_CALUDE_truncated_pyramid_lateral_area_l3431_343166


namespace NUMINAMATH_CALUDE_arithmetic_progression_of_primes_l3431_343115

theorem arithmetic_progression_of_primes (a : ℕ → ℕ) (d : ℕ) :
  (∀ i ∈ Finset.range 15, Nat.Prime (a i)) →
  (∀ i ∈ Finset.range 14, a (i + 1) = a i + d) →
  d > 0 →
  a 0 > 15 →
  d > 30000 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_of_primes_l3431_343115


namespace NUMINAMATH_CALUDE_cos_two_thirds_pi_plus_two_alpha_l3431_343170

theorem cos_two_thirds_pi_plus_two_alpha (α : Real) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  Real.cos (2 * π / 3 + 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_thirds_pi_plus_two_alpha_l3431_343170


namespace NUMINAMATH_CALUDE_bookshop_revenue_l3431_343120

-- Define book types and their prices
structure BookType where
  name : String
  price : Nat

-- Define a day's transactions
structure DayTransactions where
  novels_sold : Nat
  comics_sold : Nat
  biographies_sold : Nat
  novels_returned : Nat
  comics_returned : Nat
  biographies_returned : Nat
  discount : Nat  -- Discount percentage (0 for no discount)

def calculate_revenue (novel : BookType) (comic : BookType) (biography : BookType) 
                      (monday : DayTransactions) (tuesday : DayTransactions) 
                      (wednesday : DayTransactions) (thursday : DayTransactions) 
                      (friday : DayTransactions) : Nat :=
  sorry  -- Proof to be implemented

theorem bookshop_revenue : 
  let novel : BookType := { name := "Novel", price := 10 }
  let comic : BookType := { name := "Comic", price := 5 }
  let biography : BookType := { name := "Biography", price := 15 }
  
  let monday : DayTransactions := {
    novels_sold := 30, comics_sold := 20, biographies_sold := 25,
    novels_returned := 1, comics_returned := 5, biographies_returned := 0,
    discount := 0
  }
  
  let tuesday : DayTransactions := {
    novels_sold := 20, comics_sold := 10, biographies_sold := 20,
    novels_returned := 0, comics_returned := 0, biographies_returned := 0,
    discount := 20
  }
  
  let wednesday : DayTransactions := {
    novels_sold := 30, comics_sold := 20, biographies_sold := 14,
    novels_returned := 5, comics_returned := 0, biographies_returned := 3,
    discount := 0
  }
  
  let thursday : DayTransactions := {
    novels_sold := 40, comics_sold := 25, biographies_sold := 13,
    novels_returned := 0, comics_returned := 0, biographies_returned := 0,
    discount := 10
  }
  
  let friday : DayTransactions := {
    novels_sold := 55, comics_sold := 40, biographies_sold := 40,
    novels_returned := 2, comics_returned := 5, biographies_returned := 3,
    discount := 0
  }
  
  calculate_revenue novel comic biography monday tuesday wednesday thursday friday = 3603 :=
by sorry


end NUMINAMATH_CALUDE_bookshop_revenue_l3431_343120


namespace NUMINAMATH_CALUDE_complement_P_union_Q_l3431_343114

def U : Set ℝ := Set.univ
def P : Set ℝ := {x | x > 1}
def Q : Set ℝ := {x | x * (x - 2) < 0}

theorem complement_P_union_Q : 
  (U \ (P ∪ Q)) = {x : ℝ | x ≤ 0} := by sorry

end NUMINAMATH_CALUDE_complement_P_union_Q_l3431_343114


namespace NUMINAMATH_CALUDE_log_equation_holds_l3431_343172

theorem log_equation_holds (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 3) * (Real.log 5 / Real.log x) = Real.log 5 / Real.log 3 := by
  sorry


end NUMINAMATH_CALUDE_log_equation_holds_l3431_343172


namespace NUMINAMATH_CALUDE_average_daily_attendance_l3431_343159

def monday_attendance : ℕ := 10
def tuesday_attendance : ℕ := 15
def wednesday_to_friday_attendance : ℕ := 10
def total_days : ℕ := 5

def total_attendance : ℕ := 
  monday_attendance + tuesday_attendance + 3 * wednesday_to_friday_attendance

theorem average_daily_attendance : 
  total_attendance / total_days = 11 := by
  sorry

end NUMINAMATH_CALUDE_average_daily_attendance_l3431_343159


namespace NUMINAMATH_CALUDE_circle_equation_proof_l3431_343177

/-- The equation of a circle with center (h, k) and radius r is (x - h)² + (y - k)² = r² -/
def is_circle_equation (h k r : ℝ) (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f x y = (x - h)^2 + (y - k)^2 - r^2

/-- A point (x, y) is on a line ax + by + c = 0 if it satisfies the equation -/
def point_on_line (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

/-- A point (x, y) is on a circle if it satisfies the circle's equation -/
def point_on_circle (f : ℝ → ℝ → ℝ) (x y : ℝ) : Prop :=
  f x y = 0

theorem circle_equation_proof (f : ℝ → ℝ → ℝ) :
  is_circle_equation 1 1 2 f →
  (∀ x y, point_on_line 1 1 (-2) x y → point_on_circle f x y) →
  point_on_circle f 1 (-1) →
  point_on_circle f (-1) 1 →
  ∀ x y, f x y = (x - 1)^2 + (y - 1)^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l3431_343177


namespace NUMINAMATH_CALUDE_jumping_rooks_remainder_l3431_343149

/-- The number of ways to place 2n jumping rooks on an n×n chessboard 
    such that each rook attacks exactly two other rooks. -/
def f (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 0
  | 2 => 1
  | 3 => 6
  | n + 1 => n.choose 2 * (2 * f n + n * f (n - 1))

/-- The main theorem stating that the number of ways to place 16 jumping rooks
    on an 8×8 chessboard, with each rook attacking exactly two others,
    when divided by 1000, gives a remainder of 530. -/
theorem jumping_rooks_remainder : f 8 % 1000 = 530 := by
  sorry


end NUMINAMATH_CALUDE_jumping_rooks_remainder_l3431_343149


namespace NUMINAMATH_CALUDE_product_evaluation_l3431_343192

theorem product_evaluation (n : ℕ) (h : n = 3) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) = 720 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l3431_343192


namespace NUMINAMATH_CALUDE_perpendicular_lines_l3431_343162

theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, 2 * y + x + 3 = 0 ∧ 3 * y + a * x + 2 = 0 → 
    ((-1/2) * (-a/3) = -1)) → 
  a = -6 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l3431_343162


namespace NUMINAMATH_CALUDE_shares_distribution_l3431_343128

/-- Proves that if 120 rs are divided among three people (a, b, c) such that a's share is 20 rs more than b's and 20 rs less than c's, then b's share is 20 rs. -/
theorem shares_distribution (a b c : ℕ) : 
  (a + b + c = 120) →  -- Total amount is 120 rs
  (a = b + 20) →       -- a's share is 20 rs more than b's
  (c = a + 20) →       -- c's share is 20 rs more than a's
  b = 20 :=            -- b's share is 20 rs
by sorry


end NUMINAMATH_CALUDE_shares_distribution_l3431_343128
