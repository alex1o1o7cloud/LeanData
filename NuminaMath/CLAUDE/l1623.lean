import Mathlib

namespace arithmetic_sequence_common_ratio_l1623_162333

theorem arithmetic_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum1 : a 1 + a 3 = 10) 
  (h_sum2 : a 4 + a 6 = 5/4) : 
  ∃ q : ℚ, q = 1/2 ∧ ∀ n : ℕ, a (n + 1) = a n + q := by
  sorry

end arithmetic_sequence_common_ratio_l1623_162333


namespace store_earnings_theorem_l1623_162344

/-- Represents the earnings from selling bottled drinks in a country store. -/
def store_earnings (cola_price juice_price water_price : ℚ) 
                   (cola_sold juice_sold water_sold : ℕ) : ℚ :=
  cola_price * cola_sold + juice_price * juice_sold + water_price * water_sold

/-- Theorem stating that the store earned $88 from selling bottled drinks. -/
theorem store_earnings_theorem : 
  store_earnings 3 1.5 1 15 12 25 = 88 := by
  sorry

end store_earnings_theorem_l1623_162344


namespace quadratic_expansion_sum_l1623_162391

theorem quadratic_expansion_sum (d : ℝ) (h : d ≠ 0) : 
  ∃ (a b c : ℤ), (15 * d^2 + 15 + 7 * d) + (3 * d + 9)^2 = a * d^2 + b * d + c ∧ a + b + c = 181 := by
  sorry

end quadratic_expansion_sum_l1623_162391


namespace ellipse_midpoint_theorem_l1623_162323

/-- Defines an ellipse with semi-major axis a and semi-minor axis b -/
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2) + (p.2 ^ 2 / b ^ 2) = 1}

/-- Defines a line with slope m passing through point (x₀, y₀) -/
def Line (m x₀ y₀ : ℝ) := {p : ℝ × ℝ | p.2 = m * (p.1 - x₀) + y₀}

theorem ellipse_midpoint_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let C := Ellipse a b
  let L := Line (4/5) 3 0
  (0, 4) ∈ C ∧ 
  (a^2 - b^2) / a^2 = 9/25 →
  ∃ p q : ℝ × ℝ, p ∈ C ∧ p ∈ L ∧ q ∈ C ∧ q ∈ L ∧ 
  (p.1 + q.1) / 2 = 3/2 ∧ (p.2 + q.2) / 2 = -6/5 := by
sorry

end ellipse_midpoint_theorem_l1623_162323


namespace division_simplification_l1623_162340

theorem division_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (12 * x^2 * y) / (-6 * x * y) = -2 * x :=
by sorry

end division_simplification_l1623_162340


namespace largest_four_digit_congruence_l1623_162321

theorem largest_four_digit_congruence :
  ∃ (n : ℕ), 
    n ≤ 9999 ∧ 
    n ≥ 1000 ∧ 
    45 * n ≡ 180 [MOD 315] ∧
    ∀ (m : ℕ), m ≤ 9999 ∧ m ≥ 1000 ∧ 45 * m ≡ 180 [MOD 315] → m ≤ n ∧
    n = 9993 :=
by sorry

end largest_four_digit_congruence_l1623_162321


namespace brian_stones_l1623_162368

theorem brian_stones (total : ℕ) (white black grey green : ℕ) : 
  total = 100 → 
  white + black = total → 
  grey = 40 → 
  green = 60 → 
  white * green = black * grey → 
  white > black → 
  white = 60 := by sorry

end brian_stones_l1623_162368


namespace min_value_theorem_l1623_162386

/-- A line that bisects a circle -/
structure BisectingLine where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : ∀ x y : ℝ, a * x + b * y - 2 = 0 → 
    (x - 3)^2 + (y - 2)^2 = 25 → (x - 3)^2 + (y - 2)^2 ≤ 25

/-- The theorem stating the minimum value of 3/a + 2/b -/
theorem min_value_theorem (l : BisectingLine) : 
  (∀ k : BisectingLine, 3 / l.a + 2 / l.b ≤ 3 / k.a + 2 / k.b) → 
  3 / l.a + 2 / l.b = 25 / 2 :=
sorry

end min_value_theorem_l1623_162386


namespace book_reading_ratio_l1623_162312

theorem book_reading_ratio (total_pages : ℕ) (pages_day1 : ℕ) (pages_left : ℕ) :
  total_pages = 360 →
  pages_day1 = 50 →
  pages_left = 210 →
  ∃ (pages_day2 : ℕ),
    pages_day2 + pages_day1 + pages_left = total_pages ∧
    pages_day2 = 2 * pages_day1 :=
by sorry

end book_reading_ratio_l1623_162312


namespace fifteen_team_league_games_l1623_162357

/-- The number of games played in a league where each team plays every other team once -/
def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a league with 15 teams, where each team plays every other team once,
    the total number of games played is 105 -/
theorem fifteen_team_league_games :
  games_played 15 = 105 := by
  sorry

end fifteen_team_league_games_l1623_162357


namespace complex_sum_equality_l1623_162352

theorem complex_sum_equality : 
  let z₁ : ℂ := -1/2 + 3/4 * I
  let z₂ : ℂ := 7/3 - 5/6 * I
  z₁ + z₂ = 11/6 - 1/12 * I := by
sorry

end complex_sum_equality_l1623_162352


namespace water_sip_calculation_l1623_162313

/-- Proves that given a 2-liter bottle of water consumed in 250 minutes with sips taken every 5 minutes, each sip is 40 ml. -/
theorem water_sip_calculation (bottle_volume : ℕ) (total_time : ℕ) (sip_interval : ℕ) :
  bottle_volume = 2000 →
  total_time = 250 →
  sip_interval = 5 →
  (bottle_volume / (total_time / sip_interval) : ℚ) = 40 := by
  sorry

#check water_sip_calculation

end water_sip_calculation_l1623_162313


namespace cosine_sum_simplification_l1623_162316

theorem cosine_sum_simplification :
  Real.cos (π / 15) + Real.cos (4 * π / 15) + Real.cos (14 * π / 15) = (Real.sqrt 21 - 1) / 4 := by
  sorry

end cosine_sum_simplification_l1623_162316


namespace gcd_7163_209_l1623_162346

theorem gcd_7163_209 :
  let a := 7163
  let b := 209
  let c := 57
  let d := 38
  let e := 19
  a = b * 34 + c →
  b = c * 3 + d →
  c = d * 1 + e →
  d = e * 2 →
  Nat.gcd a b = e :=
by sorry

end gcd_7163_209_l1623_162346


namespace min_sum_a_b_l1623_162394

theorem min_sum_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∃ x : ℝ, x^2 + a*x + 2*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 2*b*x + a = 0) :
  ∀ c d : ℝ, c > 0 → d > 0 →
  (∃ x : ℝ, x^2 + c*x + 2*d = 0) →
  (∃ x : ℝ, x^2 + 2*d*x + c = 0) →
  c + d ≥ 6 :=
by sorry

end min_sum_a_b_l1623_162394


namespace parabola_vertex_l1623_162327

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = -9 * (x - 7)^2

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (7, 0)

/-- Theorem: The vertex of the parabola y = -9(x-7)^2 is at the point (7, 0) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
by
  sorry

end parabola_vertex_l1623_162327


namespace distribute_five_balls_three_boxes_l1623_162345

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes,
    with each box containing at least one ball. -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 150 ways to distribute 5 distinguishable balls into 3 indistinguishable boxes,
    with each box containing at least one ball. -/
theorem distribute_five_balls_three_boxes :
  distribute_balls 5 3 = 150 := by sorry

end distribute_five_balls_three_boxes_l1623_162345


namespace max_small_squares_in_large_square_l1623_162307

/-- The side length of the large square -/
def large_square_side : ℕ := 8

/-- The side length of the small squares -/
def small_square_side : ℕ := 2

/-- The maximum number of non-overlapping small squares that can fit inside the large square -/
def max_small_squares : ℕ := (large_square_side / small_square_side) ^ 2

theorem max_small_squares_in_large_square :
  max_small_squares = 16 :=
sorry

end max_small_squares_in_large_square_l1623_162307


namespace alexanders_pictures_l1623_162332

theorem alexanders_pictures (total_pencils : ℕ) 
  (new_galleries : ℕ) (pictures_per_new_gallery : ℕ) 
  (pencils_per_picture : ℕ) (pencils_per_exhibition : ℕ) : 
  total_pencils = 88 →
  new_galleries = 5 →
  pictures_per_new_gallery = 2 →
  pencils_per_picture = 4 →
  pencils_per_exhibition = 2 →
  (total_pencils - 
    (new_galleries * pictures_per_new_gallery * pencils_per_picture) - 
    ((new_galleries + 1) * pencils_per_exhibition)) / pencils_per_picture = 9 :=
by sorry

end alexanders_pictures_l1623_162332


namespace prob_red_then_white_l1623_162366

/-- The probability of drawing a red marble first and a white marble second without replacement
    from a bag containing 3 red marbles and 5 white marbles is 15/56. -/
theorem prob_red_then_white (red : ℕ) (white : ℕ) (total : ℕ) (h1 : red = 3) (h2 : white = 5) 
  (h3 : total = red + white) :
  (red / total) * (white / (total - 1)) = 15 / 56 := by
  sorry

end prob_red_then_white_l1623_162366


namespace mary_saw_256_snakes_l1623_162363

/-- The number of breeding balls -/
def num_breeding_balls : Nat := 7

/-- The number of snakes in each breeding ball -/
def snakes_in_balls : List Nat := [15, 20, 25, 30, 35, 40, 45]

/-- The number of extra pairs of snakes -/
def extra_pairs : Nat := 23

/-- The total number of snakes Mary saw -/
def total_snakes : Nat := (List.sum snakes_in_balls) + (2 * extra_pairs)

theorem mary_saw_256_snakes :
  total_snakes = 256 := by sorry

end mary_saw_256_snakes_l1623_162363


namespace smallest_abs_z_l1623_162337

theorem smallest_abs_z (z : ℂ) (h : Complex.abs (z - 9) + Complex.abs (z - 6*I) = 15) : 
  ∃ (min_abs_z : ℝ), min_abs_z = 3.6 ∧ ∀ w : ℂ, Complex.abs (w - 9) + Complex.abs (w - 6*I) = 15 → Complex.abs w ≥ min_abs_z :=
sorry

end smallest_abs_z_l1623_162337


namespace students_in_both_band_and_chorus_l1623_162355

theorem students_in_both_band_and_chorus 
  (total : ℕ) 
  (band : ℕ) 
  (chorus : ℕ) 
  (band_or_chorus : ℕ) 
  (h1 : total = 300)
  (h2 : band = 100)
  (h3 : chorus = 120)
  (h4 : band_or_chorus = 195) :
  band + chorus - band_or_chorus = 25 :=
by sorry

end students_in_both_band_and_chorus_l1623_162355


namespace people_on_boats_l1623_162328

theorem people_on_boats (num_boats : ℕ) (people_per_boat : ℕ) :
  num_boats = 5 → people_per_boat = 3 → num_boats * people_per_boat = 15 := by
  sorry

end people_on_boats_l1623_162328


namespace unique_solution_fifth_power_equation_l1623_162385

theorem unique_solution_fifth_power_equation :
  ∃! (x : ℝ), x ≠ 0 ∧ (5 * x)^10 = (10 * x)^5 :=
by
  -- The unique solution is x = 2/5
  use 2/5
  sorry

end unique_solution_fifth_power_equation_l1623_162385


namespace round_table_seating_arrangements_l1623_162364

def num_people : ℕ := 6
def num_specific_people : ℕ := 2

theorem round_table_seating_arrangements :
  let num_units := num_people - num_specific_people + 1
  (num_specific_people.factorial) * ((num_units - 1).factorial) = 48 := by
  sorry

end round_table_seating_arrangements_l1623_162364


namespace max_pencils_buyable_l1623_162301

/-- Represents the number of pencils in a set -/
inductive PencilSet
| Large : PencilSet  -- 20 pencils
| Small : PencilSet  -- 5 pencils

/-- Represents the rebate percentage for a given set -/
def rebate_percentage (s : PencilSet) : ℚ :=
  match s with
  | PencilSet.Large => 25 / 100
  | PencilSet.Small => 10 / 100

/-- Represents the number of pencils in a given set -/
def pencils_in_set (s : PencilSet) : ℕ :=
  match s with
  | PencilSet.Large => 20
  | PencilSet.Small => 5

/-- The initial number of pencils Vasya can afford -/
def initial_pencils : ℕ := 30

/-- Theorem stating the maximum number of pencils Vasya can buy -/
theorem max_pencils_buyable :
  ∃ (large_sets small_sets : ℕ),
    large_sets * pencils_in_set PencilSet.Large +
    small_sets * pencils_in_set PencilSet.Small +
    ⌊large_sets * (rebate_percentage PencilSet.Large * pencils_in_set PencilSet.Large : ℚ)⌋ +
    ⌊small_sets * (rebate_percentage PencilSet.Small * pencils_in_set PencilSet.Small : ℚ)⌋ = 41 ∧
    large_sets * pencils_in_set PencilSet.Large +
    small_sets * pencils_in_set PencilSet.Small ≤ initial_pencils :=
by sorry

end max_pencils_buyable_l1623_162301


namespace reflected_ray_equation_l1623_162395

/-- The equation of a line passing through two given points is correct. -/
theorem reflected_ray_equation (x y : ℝ) :
  let p1 : ℝ × ℝ := (-1, -3)  -- Symmetric point of (-1, 3) with respect to x-axis
  let p2 : ℝ × ℝ := (4, 6)    -- Given point that the reflected ray passes through
  9 * x - 5 * y - 6 = 0 ↔ (y - p1.2) * (p2.1 - p1.1) = (x - p1.1) * (p2.2 - p1.2) :=
by sorry

end reflected_ray_equation_l1623_162395


namespace books_left_after_donation_l1623_162315

/-- Calculates the total number of books left after donation --/
def booksLeftAfterDonation (
  mysteryShelvesCount : ℕ)
  (mysteryBooksPerShelf : ℕ)
  (pictureBooksShelvesCount : ℕ)
  (pictureBooksPerShelf : ℕ)
  (autobiographyShelvesCount : ℕ)
  (autobiographyBooksPerShelf : ℝ)
  (cookbookShelvesCount : ℕ)
  (cookbookBooksPerShelf : ℝ)
  (mysteryBooksDonated : ℕ)
  (pictureBooksdonated : ℕ)
  (autobiographiesDonated : ℕ)
  (cookbooksDonated : ℕ) : ℝ :=
  let totalBooksBeforeDonation :=
    (mysteryShelvesCount * mysteryBooksPerShelf : ℝ) +
    (pictureBooksShelvesCount * pictureBooksPerShelf : ℝ) +
    (autobiographyShelvesCount : ℝ) * autobiographyBooksPerShelf +
    (cookbookShelvesCount : ℝ) * cookbookBooksPerShelf
  let totalBooksDonated :=
    (mysteryBooksDonated + pictureBooksdonated + autobiographiesDonated + cookbooksDonated : ℝ)
  totalBooksBeforeDonation - totalBooksDonated

theorem books_left_after_donation :
  booksLeftAfterDonation 3 9 5 12 4 8.5 2 11.5 7 8 3 5 = 121 := by
  sorry

end books_left_after_donation_l1623_162315


namespace line_intercepts_minimum_sum_l1623_162384

theorem line_intercepts_minimum_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_point : a + b = a * b) : 
  (a / b + b / a) ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = a₀ * b₀ ∧ a₀ / b₀ + b₀ / a₀ = 4 :=
by sorry

end line_intercepts_minimum_sum_l1623_162384


namespace rajan_investment_is_20000_l1623_162343

/-- Represents the investment scenario with Rajan, Rakesh, and Mukesh --/
structure InvestmentScenario where
  rajan_investment : ℕ
  rakesh_investment : ℕ
  mukesh_investment : ℕ
  total_profit : ℕ
  rajan_profit : ℕ

/-- The investment scenario satisfies the given conditions --/
def satisfies_conditions (scenario : InvestmentScenario) : Prop :=
  scenario.rakesh_investment = 25000 ∧
  scenario.mukesh_investment = 15000 ∧
  scenario.total_profit = 4600 ∧
  scenario.rajan_profit = 2400 ∧
  (scenario.rajan_investment * 12 : ℚ) / 
    (scenario.rajan_investment * 12 + scenario.rakesh_investment * 4 + scenario.mukesh_investment * 8) = 
    (scenario.rajan_profit : ℚ) / scenario.total_profit

/-- Theorem stating that if the scenario satisfies the conditions, Rajan's investment is 20000 --/
theorem rajan_investment_is_20000 (scenario : InvestmentScenario) :
  satisfies_conditions scenario → scenario.rajan_investment = 20000 := by
  sorry

#check rajan_investment_is_20000

end rajan_investment_is_20000_l1623_162343


namespace arithmetic_series_sum_10_70_1_7th_l1623_162387

def arithmeticSeriesSum (a₁ aₙ d : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_series_sum_10_70_1_7th : 
  arithmeticSeriesSum 10 70 (1/7) = 16840 := by
  sorry

end arithmetic_series_sum_10_70_1_7th_l1623_162387


namespace profit_maximization_l1623_162326

/-- Represents the daily sales volume as a function of the selling price. -/
def sales_volume (x : ℝ) : ℝ := -20 * x + 1600

/-- Represents the daily profit as a function of the selling price. -/
def profit (x : ℝ) : ℝ := (x - 40) * (sales_volume x)

theorem profit_maximization (x : ℝ) (h1 : x ≥ 45) (h2 : x < 80) :
  profit x ≤ 8000 ∧ profit 60 = 8000 :=
by sorry

#check profit_maximization

end profit_maximization_l1623_162326


namespace decimal_difference_l1623_162330

-- Define the repeating decimal 0.727272...
def repeating_decimal : ℚ := 72 / 99

-- Define the terminating decimal 0.72
def terminating_decimal : ℚ := 72 / 100

-- Theorem statement
theorem decimal_difference :
  repeating_decimal - terminating_decimal = 2 / 275 := by
  sorry

end decimal_difference_l1623_162330


namespace cubic_less_than_square_l1623_162310

theorem cubic_less_than_square (x : ℚ) : 
  (x = 3/4 → x^3 < x^2) ∧ 
  (x = 5/3 → x^3 ≥ x^2) ∧ 
  (x = 1 → x^3 ≥ x^2) ∧ 
  (x = 3/2 → x^3 ≥ x^2) ∧ 
  (x = 21/20 → x^3 ≥ x^2) := by
  sorry

end cubic_less_than_square_l1623_162310


namespace probability_two_ties_l1623_162324

/-- The probability of selecting 2 ties from a boutique with shirts, pants, and ties -/
theorem probability_two_ties (shirts pants ties : ℕ) : 
  shirts = 4 → pants = 8 → ties = 18 → 
  (ties : ℚ) / (shirts + pants + ties) * ((ties - 1) : ℚ) / (shirts + pants + ties - 1) = 51 / 145 := by
  sorry

end probability_two_ties_l1623_162324


namespace quadratic_function_theorem_l1623_162362

/-- A quadratic function with specific properties -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- The derivative of the function -/
def HasDerivative (f : ℝ → ℝ) : Prop :=
  ∀ x, deriv f x = 2 * x + 2

/-- The function has two equal real roots -/
def HasEqualRoots (f : ℝ → ℝ) : Prop :=
  ∃ r : ℝ, (∀ x, f x = 0 ↔ x = r) ∧ (deriv f r = 0)

/-- The main theorem -/
theorem quadratic_function_theorem (f : ℝ → ℝ) 
  (h1 : QuadraticFunction f) 
  (h2 : HasDerivative f) 
  (h3 : HasEqualRoots f) : 
  ∀ x, f x = x^2 + 2*x + 1 := by
  sorry

end quadratic_function_theorem_l1623_162362


namespace park_hikers_l1623_162397

theorem park_hikers (total : ℕ) (difference : ℕ) (hikers : ℕ) (bikers : ℕ) : 
  total = 676 → 
  difference = 178 → 
  total = hikers + bikers → 
  hikers = bikers + difference → 
  hikers = 427 := by
sorry

end park_hikers_l1623_162397


namespace leftSideSeats_l1623_162341

/-- Represents the seating arrangement in a bus -/
structure BusSeats where
  leftSeats : ℕ
  rightSeats : ℕ
  backSeat : ℕ
  peoplePerSeat : ℕ
  totalCapacity : ℕ

/-- The bus seating arrangement satisfies the given conditions -/
def validBusSeats (bus : BusSeats) : Prop :=
  bus.rightSeats = bus.leftSeats - 3 ∧
  bus.peoplePerSeat = 3 ∧
  bus.backSeat = 9 ∧
  bus.totalCapacity = 90

/-- The theorem stating that the number of seats on the left side is 15 -/
theorem leftSideSeats (bus : BusSeats) (h : validBusSeats bus) : 
  bus.leftSeats = 15 := by
  sorry

#check leftSideSeats

end leftSideSeats_l1623_162341


namespace order_of_logarithmic_expressions_l1623_162314

theorem order_of_logarithmic_expressions (x : ℝ) 
  (h1 : x ∈ Set.Ioo (Real.exp (-1)) 1)
  (a b c : ℝ) 
  (ha : a = Real.log x)
  (hb : b = 2 * Real.log x)
  (hc : c = (Real.log x) ^ 3) : 
  b < a ∧ a < c := by
sorry

end order_of_logarithmic_expressions_l1623_162314


namespace condition_sufficient_not_necessary_l1623_162371

-- Define a triangle with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality_ab : a < b + c
  triangle_inequality_bc : b < a + c
  triangle_inequality_ca : c < a + b

-- Define what it means for an angle to be acute in terms of side lengths
def is_angle_A_acute (t : Triangle) : Prop :=
  t.b ^ 2 + t.c ^ 2 > t.a ^ 2

-- Define the condition a ≤ (b + c) / 2
def condition (t : Triangle) : Prop :=
  t.a ≤ (t.b + t.c) / 2

-- Theorem statement
theorem condition_sufficient_not_necessary :
  (∀ t : Triangle, condition t → is_angle_A_acute t) ∧
  ¬(∀ t : Triangle, is_angle_A_acute t → condition t) :=
sorry

end condition_sufficient_not_necessary_l1623_162371


namespace greatest_power_of_two_factor_l1623_162380

theorem greatest_power_of_two_factor (n : ℕ) : 
  ∃ (k : ℕ), 2^k ∣ (10^1000 - 4^500) ∧ 
  ∀ (m : ℕ), 2^m ∣ (10^1000 - 4^500) → m ≤ k := by
  sorry

end greatest_power_of_two_factor_l1623_162380


namespace units_digit_of_A_is_1_l1623_162382

-- Define the sequence of powers of 3
def powerOf3 : ℕ → ℕ
| 0 => 1
| n + 1 => 3 * powerOf3 n

-- Define A
def A : ℕ := 2 * (3 + 1) * (powerOf3 2 + 1) * (powerOf3 4 + 1) + 1

-- Theorem statement
theorem units_digit_of_A_is_1 : A % 10 = 1 := by
  sorry


end units_digit_of_A_is_1_l1623_162382


namespace b_share_correct_l1623_162373

/-- The share of the total payment for worker b -/
def b_share (a_days b_days c_days d_days total_payment : ℚ) : ℚ :=
  (1 / b_days) / ((1 / a_days) + (1 / b_days) + (1 / c_days) + (1 / d_days)) * total_payment

/-- Theorem stating that b's share is correct given the problem conditions -/
theorem b_share_correct :
  b_share 6 8 12 15 2400 = (1 / 8) / (53 / 120) * 2400 := by
  sorry

#eval b_share 6 8 12 15 2400

end b_share_correct_l1623_162373


namespace arithmetic_mean_problem_l1623_162318

theorem arithmetic_mean_problem (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 26)
  (h3 : r - p = 32) :
  (p + q) / 2 = 10 := by
sorry

end arithmetic_mean_problem_l1623_162318


namespace equal_triangle_areas_l1623_162351

-- Define the trapezoid ABCD
structure Trapezoid (A B C D : ℝ × ℝ) : Prop where
  parallel : (A.2 - D.2) / (A.1 - D.1) = (B.2 - C.2) / (B.1 - C.1)

-- Define a point inside a polygon
def PointInside (P : ℝ × ℝ) (polygon : List (ℝ × ℝ)) : Prop := sorry

-- Define parallel lines
def Parallel (P₁ Q₁ P₂ Q₂ : ℝ × ℝ) : Prop :=
  (P₁.2 - Q₁.2) / (P₁.1 - Q₁.1) = (P₂.2 - Q₂.2) / (P₂.1 - Q₂.1)

-- Define the area of a triangle
def TriangleArea (P Q R : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem equal_triangle_areas 
  (A B C D M N : ℝ × ℝ) 
  (trap : Trapezoid A B C D)
  (m_inside : PointInside M [A, B, C, D])
  (n_inside : PointInside N [B, M, C])
  (am_cn_parallel : Parallel A M C N)
  (bm_dn_parallel : Parallel B M D N) :
  TriangleArea A B N = TriangleArea C D M := by
  sorry

end equal_triangle_areas_l1623_162351


namespace max_value_inequality_l1623_162331

theorem max_value_inequality (a b c : ℝ) 
  (sum_eq : a + b + c = 3)
  (a_ge : a ≥ -1/2)
  (b_ge : b ≥ -3/2)
  (c_ge : c ≥ -2) :
  Real.sqrt (4*a + 2) + Real.sqrt (4*b + 6) + Real.sqrt (4*c + 8) ≤ 2 * Real.sqrt 21 := by
  sorry

end max_value_inequality_l1623_162331


namespace max_greece_value_l1623_162377

/-- Represents a mapping from letters to digits -/
def LetterMap := Char → Nat

/-- Check if a LetterMap is valid according to the problem conditions -/
def isValidMapping (m : LetterMap) : Prop :=
  (∀ c₁ c₂, c₁ ≠ c₂ → m c₁ ≠ m c₂) ∧ 
  (∀ c, m c ≤ 9) ∧
  m 'G' ≠ 0 ∧ m 'E' ≠ 0 ∧ m 'V' ≠ 0 ∧ m 'I' ≠ 0

/-- Convert a string of letters to a number using the given mapping -/
def stringToNumber (m : LetterMap) (s : String) : Nat :=
  s.foldl (fun acc c => acc * 10 + m c) 0

/-- Check if the equation holds for a given mapping -/
def equationHolds (m : LetterMap) : Prop :=
  (stringToNumber m "VER" - stringToNumber m "IA") = 
  (m 'G')^((m 'R')^(m 'E')) * (stringToNumber m "GRE" + stringToNumber m "ECE")

/-- The main theorem to be proved -/
theorem max_greece_value (m : LetterMap) :
  isValidMapping m →
  equationHolds m →
  (∀ m', isValidMapping m' → equationHolds m' → 
    stringToNumber m' "GREECE" ≤ stringToNumber m "GREECE") →
  stringToNumber m "GREECE" = 196646 := by
  sorry

end max_greece_value_l1623_162377


namespace max_value_of_expression_l1623_162342

theorem max_value_of_expression (x : ℝ) (h : 0 ≤ x ∧ x ≤ 25) :
  Real.sqrt (x + 64) + Real.sqrt (25 - x) + 2 * Real.sqrt x ≤ 19 := by
  sorry

end max_value_of_expression_l1623_162342


namespace debbie_number_l1623_162353

def alice_skips (n : ℕ) : Bool :=
  n % 4 = 3

def barbara_says (n : ℕ) : Bool :=
  alice_skips n ∧ ¬(n % 12 = 7)

def candice_says (n : ℕ) : Bool :=
  alice_skips n ∧ barbara_says n ∧ ¬(n % 24 = 11)

def debbie_says (n : ℕ) : Bool :=
  1 ≤ n ∧ n ≤ 1200 ∧ ¬(alice_skips n) ∧ ¬(barbara_says n) ∧ ¬(candice_says n)

theorem debbie_number : ∃! n : ℕ, debbie_says n ∧ n = 1187 := by
  sorry

end debbie_number_l1623_162353


namespace george_and_hannah_win_l1623_162304

-- Define the set of students
inductive Student : Type
  | Elaine : Student
  | Frank : Student
  | George : Student
  | Hannah : Student

-- Define a function to represent winning a prize
def wins_prize (s : Student) : Prop := sorry

-- Define the conditions
axiom only_two_winners :
  ∃ (a b : Student), a ≠ b ∧
    (∀ s : Student, wins_prize s ↔ (s = a ∨ s = b))

axiom elaine_implies_frank :
  wins_prize Student.Elaine → wins_prize Student.Frank

axiom frank_implies_george :
  wins_prize Student.Frank → wins_prize Student.George

axiom george_implies_hannah :
  wins_prize Student.George → wins_prize Student.Hannah

-- Theorem to prove
theorem george_and_hannah_win :
  wins_prize Student.George ∧ wins_prize Student.Hannah ∧
  ¬wins_prize Student.Elaine ∧ ¬wins_prize Student.Frank :=
sorry

end george_and_hannah_win_l1623_162304


namespace smallest_2016_div_2017_correct_l1623_162390

/-- The smallest natural number that starts with 2016 and is divisible by 2017 -/
def smallest_2016_div_2017 : ℕ := 20162001

/-- A number starts with 2016 if it's greater than or equal to 2016 * 10^4 and less than 2017 * 10^4 -/
def starts_with_2016 (n : ℕ) : Prop :=
  2016 * 10^4 ≤ n ∧ n < 2017 * 10^4

theorem smallest_2016_div_2017_correct :
  starts_with_2016 smallest_2016_div_2017 ∧
  smallest_2016_div_2017 % 2017 = 0 ∧
  ∀ n : ℕ, n < smallest_2016_div_2017 →
    ¬(starts_with_2016 n ∧ n % 2017 = 0) :=
by sorry

end smallest_2016_div_2017_correct_l1623_162390


namespace gcd_power_two_minus_one_l1623_162350

theorem gcd_power_two_minus_one (a b : ℕ+) :
  Nat.gcd (2^a.val - 1) (2^b.val - 1) = 2^(Nat.gcd a.val b.val) - 1 := by
  sorry

end gcd_power_two_minus_one_l1623_162350


namespace range_of_a_satisfying_condition_l1623_162354

/-- The universal set U is the set of real numbers. -/
def U : Set ℝ := Set.univ

/-- Set A is defined as {x | (x - 2)(x - 9) < 0}. -/
def A : Set ℝ := {x | (x - 2) * (x - 9) < 0}

/-- Set B is defined as {x | -2 - x ≤ 0 ≤ 5 - x}. -/
def B : Set ℝ := {x | -2 - x ≤ 0 ∧ 0 ≤ 5 - x}

/-- Set C is defined as {x | a ≤ x ≤ 2 - a}, where a is a real number. -/
def C (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 2 - a}

/-- The theorem states that given the conditions, the range of values for a that satisfies C ∪ (∁ₘB) = R is (-∞, -3]. -/
theorem range_of_a_satisfying_condition :
  ∀ a : ℝ, (C a ∪ (U \ B) = U) ↔ a ≤ -3 :=
by sorry

end range_of_a_satisfying_condition_l1623_162354


namespace cone_shape_l1623_162347

/-- Represents a point in spherical coordinates -/
structure SphericalPoint where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Defines the set of points satisfying φ ≤ c -/
def ConeSet (c : ℝ) : Set SphericalPoint :=
  {p : SphericalPoint | p.φ ≤ c}

/-- Theorem: The set of points satisfying φ ≤ c forms a cone -/
theorem cone_shape (c : ℝ) (h : 0 ≤ c ∧ c ≤ π) :
  ∃ (cone : Set SphericalPoint), ConeSet c = cone :=
sorry

end cone_shape_l1623_162347


namespace arithmetic_sequence_middle_term_l1623_162348

theorem arithmetic_sequence_middle_term (a b c : ℤ) : 
  (a = 3^2 ∧ c = 3^4 ∧ b - a = c - b) → b = 45 := by
  sorry

end arithmetic_sequence_middle_term_l1623_162348


namespace sum_of_reciprocals_of_roots_l1623_162370

theorem sum_of_reciprocals_of_roots (x : ℝ) : 
  x^2 - 15*x + 6 = 0 → 
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ x^2 - 15*x + 6 = (x - r₁) * (x - r₂) ∧ 
  (1 / r₁ + 1 / r₂ = 5 / 2) := by
sorry

end sum_of_reciprocals_of_roots_l1623_162370


namespace exam_pass_count_l1623_162306

theorem exam_pass_count :
  let total_candidates : ℕ := 120
  let overall_average : ℚ := 35
  let pass_average : ℚ := 39
  let fail_average : ℚ := 15
  ∃ pass_count : ℕ,
    pass_count ≤ total_candidates ∧
    (pass_count : ℚ) * pass_average + (total_candidates - pass_count : ℚ) * fail_average = 
      (total_candidates : ℚ) * overall_average ∧
    pass_count = 100 := by
sorry

end exam_pass_count_l1623_162306


namespace M_mod_49_l1623_162361

/-- M is the 92-digit number formed by concatenating integers from 1 to 50 -/
def M : ℕ := sorry

/-- The sum of digits from 1 to 50 -/
def sum_digits : ℕ := (50 * (1 + 50)) / 2

theorem M_mod_49 : M % 49 = 18 := by sorry

end M_mod_49_l1623_162361


namespace square_root_three_expansion_special_case_square_root_three_simplify_square_root_expression_l1623_162378

-- Part 1
theorem square_root_three_expansion (a b m n : ℕ+) :
  a + b * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2 →
  a = m ^ 2 + 3 * n ^ 2 ∧ b = 2 * m * n :=
sorry

-- Part 2
theorem special_case_square_root_three (a m n : ℕ+) :
  a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2 →
  a = 13 ∨ a = 7 :=
sorry

-- Part 3
theorem simplify_square_root_expression :
  Real.sqrt (25 + 4 * Real.sqrt 6) = 5 + 2 * Real.sqrt 6 :=
sorry

end square_root_three_expansion_special_case_square_root_three_simplify_square_root_expression_l1623_162378


namespace sum_of_same_sign_values_l1623_162399

theorem sum_of_same_sign_values (a b : ℝ) : 
  (abs a = 3) → (abs b = 1) → (a * b > 0) → (a + b = 4 ∨ a + b = -4) := by
  sorry

end sum_of_same_sign_values_l1623_162399


namespace intersection_and_union_for_negative_one_intersection_equals_B_iff_l1623_162392

def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 5}
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a+2}

theorem intersection_and_union_for_negative_one :
  (A ∩ B (-1) = {x | -2 ≤ x ∧ x ≤ -1}) ∧
  (A ∪ B (-1) = {x | x ≤ 1 ∨ x ≥ 5}) := by sorry

theorem intersection_equals_B_iff :
  ∀ a : ℝ, A ∩ B a = B a ↔ a ≤ -3 ∨ a > 2 := by sorry

end intersection_and_union_for_negative_one_intersection_equals_B_iff_l1623_162392


namespace imaginary_unit_cubed_l1623_162338

theorem imaginary_unit_cubed (i : ℂ) (h : i^2 = -1) : i^3 = -i := by
  sorry

end imaginary_unit_cubed_l1623_162338


namespace poplar_tree_count_l1623_162379

theorem poplar_tree_count : ∃ (poplar willow : ℕ),
  poplar + willow = 120 ∧ poplar + 10 = willow ∧ poplar = 55 := by
  sorry

end poplar_tree_count_l1623_162379


namespace student_count_l1623_162388

theorem student_count (average_decrease : ℝ) (weight_difference : ℝ) : 
  average_decrease = 8 → weight_difference = 32 → (weight_difference / average_decrease : ℝ) = 4 := by
  sorry

end student_count_l1623_162388


namespace monotonic_quadratic_range_l1623_162381

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

theorem monotonic_quadratic_range (a : ℝ) :
  (∀ x ∈ Set.Icc 2 3, Monotone (fun x => f a x)) →
  a ∈ Set.Iic 2 ∪ Set.Ici 3 :=
by sorry

end monotonic_quadratic_range_l1623_162381


namespace additional_apples_needed_l1623_162311

def apples_needed (pies : ℕ) (apples_per_pie : ℕ) (available_apples : ℕ) : ℕ :=
  pies * apples_per_pie - available_apples

theorem additional_apples_needed : 
  apples_needed 10 8 50 = 30 := by
  sorry

end additional_apples_needed_l1623_162311


namespace no_natural_pair_satisfies_condition_l1623_162393

theorem no_natural_pair_satisfies_condition : 
  ¬∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ (b^a ∣ a^b - 1) :=
by sorry

end no_natural_pair_satisfies_condition_l1623_162393


namespace unique_n_congruence_l1623_162320

theorem unique_n_congruence : ∃! n : ℤ, 3 ≤ n ∧ n ≤ 9 ∧ n ≡ 12473 [ZMOD 7] ∧ n = 6 := by
  sorry

end unique_n_congruence_l1623_162320


namespace adult_meal_cost_l1623_162303

def restaurant_problem (total_people : ℕ) (num_kids : ℕ) (total_cost : ℚ) : Prop :=
  let num_adults := total_people - num_kids
  let cost_per_adult := total_cost / num_adults
  cost_per_adult = 7

theorem adult_meal_cost :
  restaurant_problem 13 9 28 := by sorry

end adult_meal_cost_l1623_162303


namespace impossible_tiling_after_replacement_l1623_162360

/-- Represents a tile type -/
inductive Tile
| TwoByTwo
| OneByFour

/-- Represents a tiling of a rectangular grid -/
def Tiling := List Tile

/-- Represents a rectangular grid -/
structure Grid :=
(rows : Nat)
(cols : Nat)

/-- Checks if a tiling is valid for a given grid -/
def isValidTiling (g : Grid) (t : Tiling) : Prop :=
  -- Definition omitted
  sorry

/-- Checks if a grid can be tiled with 2x2 and 1x4 tiles -/
def canBeTiled (g : Grid) : Prop :=
  ∃ t : Tiling, isValidTiling g t

/-- Represents the operation of replacing one 2x2 tile with a 1x4 tile -/
def replaceTile (t : Tiling) : Tiling :=
  -- Definition omitted
  sorry

/-- Main theorem: If a grid can be tiled, replacing one 2x2 tile with a 1x4 tile makes it impossible to tile -/
theorem impossible_tiling_after_replacement (g : Grid) :
  canBeTiled g → ¬(∃ t : Tiling, isValidTiling g (replaceTile t)) :=
by
  sorry

end impossible_tiling_after_replacement_l1623_162360


namespace polynomial_factorization_l1623_162334

/-- A polynomial with integer coefficients -/
def IntPolynomial (n : ℕ) := Fin n → ℤ

/-- The degree of a polynomial -/
def degree (p : IntPolynomial n) : ℕ := n

/-- Irreducibility of a polynomial -/
def irreducible (p : IntPolynomial n) : Prop := sorry

/-- The modulus of a complex number is not greater than 1 -/
def modulusNotGreaterThanOne (z : ℂ) : Prop := Complex.abs z ≤ 1

/-- The roots of a polynomial -/
def roots (p : IntPolynomial n) : Set ℂ := sorry

/-- The statement of the theorem -/
theorem polynomial_factorization 
  (n : ℕ+) 
  (f : IntPolynomial n.val) 
  (h_irred : irreducible f) 
  (h_an : f (Fin.last n.val) ≠ 0)
  (h_roots : ∀ z ∈ roots f, modulusNotGreaterThanOne z) :
  ∃ (m : ℕ+) (g : IntPolynomial m.val), 
    ∃ (h : IntPolynomial (n.val + m.val)), 
      h = sorry ∧ 
      (∀ i, h i = if i.val < n.val then f i else if i.val < n.val + m.val then g (i - n.val) else 0) ∧
      h = λ i => if i.val = n.val + m.val - 1 then 1 else if i.val = n.val + m.val then -1 else 0 :=
sorry

end polynomial_factorization_l1623_162334


namespace r_eq_m_times_phi_l1623_162365

/-- The algorithm for writing numbers on intersecting circles -/
def writeNumbers (m : ℕ) (n : ℕ) : Set (ℕ × ℕ) := sorry

/-- The number of appearances of a number on the circles -/
def r (n : ℕ) (m : ℕ) : ℕ := sorry

/-- Euler's totient function -/
def φ : ℕ → ℕ := sorry

/-- Theorem stating the relationship between r(n,m) and φ(n) -/
theorem r_eq_m_times_phi (n : ℕ) (m : ℕ) :
  r n m = m * φ n := by sorry

end r_eq_m_times_phi_l1623_162365


namespace simple_interest_rate_l1623_162300

/-- Calculate the rate of simple interest given principal, time, and interest amount -/
theorem simple_interest_rate 
  (principal : ℝ) 
  (time : ℝ) 
  (interest : ℝ) 
  (h1 : principal = 20000)
  (h2 : time = 3)
  (h3 : interest = 7200) : 
  (interest * 100) / (principal * time) = 12 := by
  sorry

#check simple_interest_rate

end simple_interest_rate_l1623_162300


namespace factorization_correctness_l1623_162359

theorem factorization_correctness (x : ℝ) : 3 * x^2 - 2*x - 1 = (3*x + 1) * (x - 1) := by
  sorry

end factorization_correctness_l1623_162359


namespace compound_composition_l1623_162305

/-- Atomic weight of hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1

/-- Atomic weight of chlorine in g/mol -/
def atomic_weight_Cl : ℝ := 35.5

/-- Atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16

/-- Molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 68

/-- Number of oxygen atoms in the compound -/
def n : ℕ := 2

theorem compound_composition :
  molecular_weight = atomic_weight_H + atomic_weight_Cl + n * atomic_weight_O :=
sorry

end compound_composition_l1623_162305


namespace prism_volume_l1623_162369

theorem prism_volume (x y z : ℝ) 
  (h1 : x * y = 24) 
  (h2 : y * z = 8) 
  (h3 : x * z = 3) : 
  x * y * z = 24 := by
  sorry

end prism_volume_l1623_162369


namespace range_of_f_l1623_162396

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 3 * Real.sin (ω * x)
noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (2 * x + φ) + 1

theorem range_of_f (ω : ℝ) (h_ω : ω > 0) (φ : ℝ) 
  (h_symmetry : ∀ x : ℝ, ∃ c : ℝ, f ω (c - x) = f ω (c + x) ∧ g φ (c - x) = g φ (c + x)) :
  Set.range (f ω) = Set.Icc (-3) 3 :=
by sorry

end range_of_f_l1623_162396


namespace grape_juice_mixture_l1623_162335

/-- Given an initial mixture with 10% grape juice, adding 20 gallons of pure grape juice
    to create a new mixture with 40% grape juice, prove that the initial mixture
    must have been 40 gallons. -/
theorem grape_juice_mixture (initial_volume : ℝ) : 
  (0.1 * initial_volume + 20) / (initial_volume + 20) = 0.4 → initial_volume = 40 := by
  sorry

end grape_juice_mixture_l1623_162335


namespace positive_expression_l1623_162336

theorem positive_expression (x y z : ℝ) 
  (hx : 0 < x ∧ x < 2) 
  (hy : -1 < y ∧ y < 0) 
  (hz : 0 < z ∧ z < 1) : 
  0 < y + x^2 := by
  sorry

end positive_expression_l1623_162336


namespace average_of_shifted_data_l1623_162375

/-- Given four positive real numbers with a specific variance, prove that the average of these numbers plus 3 is 5 -/
theorem average_of_shifted_data (x₁ x₂ x₃ x₄ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0)
  (h_var : (x₁^2 + x₂^2 + x₃^2 + x₄^2 - 16) / 4 = (x₁^2 + x₂^2 + x₃^2 + x₄^2) / 4 - ((x₁ + x₂ + x₃ + x₄) / 4)^2) :
  ((x₁ + 3) + (x₂ + 3) + (x₃ + 3) + (x₄ + 3)) / 4 = 5 := by
  sorry

end average_of_shifted_data_l1623_162375


namespace students_pets_difference_fourth_grade_classrooms_difference_l1623_162319

theorem students_pets_difference : ℕ → ℕ → ℕ → ℕ → ℕ
  | num_classrooms, students_per_class, rabbits_per_class, hamsters_per_class =>
    let total_students := num_classrooms * students_per_class
    let total_rabbits := num_classrooms * rabbits_per_class
    let total_hamsters := num_classrooms * hamsters_per_class
    let total_pets := total_rabbits + total_hamsters
    total_students - total_pets

theorem fourth_grade_classrooms_difference :
  students_pets_difference 5 20 2 1 = 85 := by
  sorry

end students_pets_difference_fourth_grade_classrooms_difference_l1623_162319


namespace least_boxes_for_candy_packing_l1623_162308

/-- Given that N is a non-zero perfect cube and 45 is a factor of N,
    prove that the least number of boxes needed to pack N pieces of candy,
    with 45 pieces per box, is 75. -/
theorem least_boxes_for_candy_packing (N : ℕ) : 
  N ≠ 0 ∧ 
  (∃ k : ℕ, N = k^3) ∧ 
  (∃ m : ℕ, N = 45 * m) ∧
  (∀ M : ℕ, M ≠ 0 ∧ (∃ j : ℕ, M = j^3) ∧ (∃ n : ℕ, M = 45 * n) → N ≤ M) →
  N / 45 = 75 := by
sorry

end least_boxes_for_candy_packing_l1623_162308


namespace debate_panel_probability_l1623_162358

def total_members : ℕ := 20
def boys : ℕ := 8
def girls : ℕ := 12
def panel_size : ℕ := 4

theorem debate_panel_probability :
  let total_combinations := Nat.choose total_members panel_size
  let all_boys := Nat.choose boys panel_size
  let all_girls := Nat.choose girls panel_size
  let prob_complement := (all_boys + all_girls : ℚ) / total_combinations
  1 - prob_complement = 856 / 969 := by sorry

end debate_panel_probability_l1623_162358


namespace inequality_sequence_properties_l1623_162398

/-- Definition of the nth inequality in the sequence -/
def nth_inequality (n : ℕ+) (x : ℝ) : Prop :=
  x + (2*n*(2*n-1))/x < 4*n - 1

/-- Definition of the solution set for the nth inequality -/
def nth_solution_set (n : ℕ+) (x : ℝ) : Prop :=
  (2*n - 1 : ℝ) < x ∧ x < 2*n

/-- Definition of the special inequality with parameter a -/
def special_inequality (a : ℕ+) (x : ℝ) : Prop :=
  x + (12*a)/(x+1) < 4*a + 2

/-- Definition of the solution set for the special inequality -/
def special_solution_set (a : ℕ+) (x : ℝ) : Prop :=
  2 < x ∧ x < 4*a - 1

/-- Main theorem statement -/
theorem inequality_sequence_properties :
  ∀ (n : ℕ+),
    (∀ (x : ℝ), nth_inequality n x ↔ nth_solution_set n x) ∧
    (∀ (a : ℕ+) (x : ℝ), special_inequality a x ↔ special_solution_set a x) := by
  sorry

end inequality_sequence_properties_l1623_162398


namespace weight_replacement_l1623_162309

theorem weight_replacement (n : ℕ) (avg_increase w_new : ℝ) :
  n = 7 →
  avg_increase = 6.2 →
  w_new = 119.4 →
  ∃ w_old : ℝ, w_old = w_new - n * avg_increase ∧ w_old = 76 :=
by sorry

end weight_replacement_l1623_162309


namespace m_range_l1623_162325

-- Define the propositions
def p (m : ℝ) : Prop := ∀ x, |x| + |x - 1| > m
def q (m : ℝ) : Prop := ∀ x y, x < y → (-(5 - 2*m)^x) > (-(5 - 2*m)^y)

-- Define the theorem
theorem m_range :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ∈ Set.Icc 1 2 ∧ m ≠ 2 :=
sorry

end m_range_l1623_162325


namespace square_starts_with_self_l1623_162329

def starts_with (a b : ℕ) : Prop :=
  ∃ k, a = b * 10^k + (a % 10^k)

theorem square_starts_with_self (N : ℕ) :
  (N > 0) → (starts_with (N^2) N) → ∃ k, N = 10^(k-1) :=
sorry

end square_starts_with_self_l1623_162329


namespace problem_1_problem_2_problem_3_problem_4_l1623_162383

-- Problem 1
theorem problem_1 : -4 * 9 = -36 := by sorry

-- Problem 2
theorem problem_2 : 10 - 14 - (-5) = 1 := by sorry

-- Problem 3
theorem problem_3 : -3 * (-1/3)^3 = 1/9 := by sorry

-- Problem 4
theorem problem_4 : -56 + (-8) * (1/8) = -57 := by sorry

end problem_1_problem_2_problem_3_problem_4_l1623_162383


namespace complement_P_intersect_Q_range_of_a_when_P_subset_Q_l1623_162374

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a + 1}
def Q : Set ℝ := {x | x^2 - 3*x ≤ 10}

-- Statement for the first part of the problem
theorem complement_P_intersect_Q :
  (Set.univ \ P 3) ∩ Q = Set.Icc (-2) 4 := by sorry

-- Statement for the second part of the problem
theorem range_of_a_when_P_subset_Q :
  {a : ℝ | P a ∩ Q = P a} = Set.Iic 2 := by sorry

end complement_P_intersect_Q_range_of_a_when_P_subset_Q_l1623_162374


namespace number_puzzle_l1623_162367

theorem number_puzzle (x : ℝ) : (1/2 : ℝ) * x + (1/3 : ℝ) * x = (1/4 : ℝ) * x + 7 → x = 12 := by
  sorry

end number_puzzle_l1623_162367


namespace minimal_disks_is_16_l1623_162389

-- Define the problem parameters
def total_files : ℕ := 42
def disk_capacity : ℚ := 2.88
def large_files : ℕ := 8
def medium_files : ℕ := 16
def large_file_size : ℚ := 1.6
def medium_file_size : ℚ := 1
def small_file_size : ℚ := 0.5

-- Define the function to calculate the minimal number of disks
def minimal_disks : ℕ := sorry

-- State the theorem
theorem minimal_disks_is_16 : minimal_disks = 16 := by sorry

end minimal_disks_is_16_l1623_162389


namespace tenth_largest_number_l1623_162317

/-- Given a list of digits, generate all possible three-digit numbers -/
def generateThreeDigitNumbers (digits : List Nat) : List Nat :=
  sorry

/-- Sort a list of numbers in descending order -/
def sortDescending (numbers : List Nat) : List Nat :=
  sorry

theorem tenth_largest_number : 
  let digits : List Nat := [5, 3, 1, 9]
  let threeDigitNumbers := generateThreeDigitNumbers digits
  let sortedNumbers := sortDescending threeDigitNumbers
  List.get! sortedNumbers 9 = 531 := by
  sorry

end tenth_largest_number_l1623_162317


namespace weight_of_replaced_person_l1623_162339

/-- Given 4 persons, if replacing one person with a new person weighing 129 kg
    increases the average weight by 8.5 kg, then the weight of the replaced person was 95 kg. -/
theorem weight_of_replaced_person
  (initial_count : Nat)
  (weight_increase : ℝ)
  (new_person_weight : ℝ)
  (h1 : initial_count = 4)
  (h2 : weight_increase = 8.5)
  (h3 : new_person_weight = 129) :
  new_person_weight - (initial_count : ℝ) * weight_increase = 95 := by
sorry

end weight_of_replaced_person_l1623_162339


namespace pairwise_sum_product_inequality_l1623_162356

theorem pairwise_sum_product_inequality 
  (x : Fin 64 → ℝ) 
  (h_pos : ∀ i, x i > 0) 
  (h_strict_mono : StrictMono x) : 
  (x 63 * x 64) / (x 0 * x 1) > (x 63 + x 64) / (x 0 + x 1) := by
  sorry

end pairwise_sum_product_inequality_l1623_162356


namespace complement_union_theorem_l1623_162322

def U : Set Int := {-1, 0, 1, 2, 3}
def P : Set Int := {0, 1, 2}
def Q : Set Int := {-1, 0}

theorem complement_union_theorem :
  (U \ P) ∪ Q = {-1, 0, 3} := by
  sorry

end complement_union_theorem_l1623_162322


namespace point_not_on_line_l1623_162372

theorem point_not_on_line (a c : ℝ) (h : a * c > 0) :
  ¬(∃ (x y : ℝ), x = 2500 ∧ y = 0 ∧ y = a * x + c) :=
by sorry

end point_not_on_line_l1623_162372


namespace inequality_proof_l1623_162349

theorem inequality_proof (a b c x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x / (y + z)) * (b + c) + (y / (z + x)) * (c + a) + (z / (x + y)) * (a + b) ≥ 
  Real.sqrt (3 * (a * b + b * c + c * a)) := by
  sorry

end inequality_proof_l1623_162349


namespace true_proposition_l1623_162302

-- Define proposition p
def p : Prop := ∀ x : ℝ, (3 : ℝ) ^ x > 0

-- Define proposition q
def q : Prop := (∀ x : ℝ, x > 0 → x > 1) ∧ ¬(∀ x : ℝ, x > 1 → x > 0)

-- Theorem statement
theorem true_proposition : p ∧ ¬q := by sorry

end true_proposition_l1623_162302


namespace equation_solution_l1623_162376

theorem equation_solution : ∃ c : ℝ, (c - 15) / 3 = (2 * c - 3) / 5 ∧ c = -66 := by
  sorry

end equation_solution_l1623_162376
