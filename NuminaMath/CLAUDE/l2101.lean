import Mathlib

namespace NUMINAMATH_CALUDE_fill_time_without_leakage_l2101_210160

/-- Represents the time to fill a tank with leakage -/
def fill_time_with_leakage : ℝ := 18

/-- Represents the time to empty the tank due to leakage -/
def empty_time_leakage : ℝ := 36

/-- Represents the volume of the tank -/
def tank_volume : ℝ := 1

/-- Theorem stating the time to fill the tank without leakage -/
theorem fill_time_without_leakage :
  let fill_rate := tank_volume / fill_time_with_leakage + tank_volume / empty_time_leakage
  tank_volume / fill_rate = 12 := by
  sorry

end NUMINAMATH_CALUDE_fill_time_without_leakage_l2101_210160


namespace NUMINAMATH_CALUDE_distinct_triangles_count_l2101_210137

/-- Represents a triangle with sides divided into segments -/
structure DividedTriangle where
  sides : ℕ  -- number of segments each side is divided into

/-- Counts the number of distinct triangles formed from division points -/
def count_distinct_triangles (t : DividedTriangle) : ℕ :=
  let total_points := (t.sides - 1) * 3
  let total_triangles := (total_points.choose 3)
  let parallel_sided := 3 * (t.sides - 1)^2
  let double_parallel := 3 * (t.sides - 1)
  let triple_parallel := 1
  total_triangles - parallel_sided + double_parallel - triple_parallel

/-- The main theorem stating the number of distinct triangles -/
theorem distinct_triangles_count (t : DividedTriangle) (h : t.sides = 8) :
  count_distinct_triangles t = 216 := by
  sorry

#eval count_distinct_triangles ⟨8⟩

end NUMINAMATH_CALUDE_distinct_triangles_count_l2101_210137


namespace NUMINAMATH_CALUDE_trig_equality_l2101_210101

theorem trig_equality (α β γ : Real) 
  (h : (1 - Real.sin α) * (1 - Real.sin β) * (1 - Real.sin γ) = 
       (1 + Real.sin α) * (1 + Real.sin β) * (1 + Real.sin γ)) : 
  (1 - Real.sin α) * (1 - Real.sin β) * (1 - Real.sin γ) = 
  |Real.cos α * Real.cos β * Real.cos γ| := by
  sorry

end NUMINAMATH_CALUDE_trig_equality_l2101_210101


namespace NUMINAMATH_CALUDE_binomial_expansion_103_minus_2_pow_5_l2101_210189

theorem binomial_expansion_103_minus_2_pow_5 :
  (103 - 2)^5 = 10510100501 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_103_minus_2_pow_5_l2101_210189


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2101_210182

/-- The sum of the infinite series ∑(k=1 to ∞) k^2 / 3^k is equal to 4.5 -/
theorem infinite_series_sum : 
  (∑' k : ℕ, (k : ℝ)^2 / 3^k) = (9/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2101_210182


namespace NUMINAMATH_CALUDE_regression_line_equation_l2101_210144

/-- Given a regression line with slope 1.23 passing through (4,5), prove its equation is y = 1.23x + 0.08 -/
theorem regression_line_equation (slope : ℝ) (center_x center_y : ℝ) :
  slope = 1.23 →
  center_x = 4 →
  center_y = 5 →
  ∃ (b : ℝ), b = 0.08 ∧ ∀ (x y : ℝ), y = slope * x + b ↔ y - center_y = slope * (x - center_x) :=
sorry

end NUMINAMATH_CALUDE_regression_line_equation_l2101_210144


namespace NUMINAMATH_CALUDE_signals_coincide_l2101_210105

def town_hall_period : ℕ := 18
def library_period : ℕ := 24
def fire_station_period : ℕ := 36

def coincidence_time : ℕ := 72

theorem signals_coincide :
  coincidence_time = Nat.lcm town_hall_period (Nat.lcm library_period fire_station_period) :=
by sorry

end NUMINAMATH_CALUDE_signals_coincide_l2101_210105


namespace NUMINAMATH_CALUDE_circular_fountain_area_l2101_210112

theorem circular_fountain_area (AB DC : ℝ) (h1 : AB = 20) (h2 : DC = 12) : 
  let AD := AB / 2
  let R := Real.sqrt (AD ^ 2 + DC ^ 2)
  π * R ^ 2 = 244 * π := by sorry

end NUMINAMATH_CALUDE_circular_fountain_area_l2101_210112


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2101_210156

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - x^2

-- Define the point of tangency
def P : ℝ × ℝ := (2, 4)

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x

-- Theorem statement
theorem tangent_line_equation :
  let slope := f' P.1
  let tangent_eq (x y : ℝ) := slope * (x - P.1) - (y - P.2)
  tangent_eq = λ x y => 8*x - y - 12 := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2101_210156


namespace NUMINAMATH_CALUDE_spinner_probability_l2101_210190

theorem spinner_probability (pA pB pC pD : ℚ) : 
  pA = 1/4 →
  pB = 1/3 →
  pA + pB + pC + pD = 1 →
  pD = 1/4 := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_l2101_210190


namespace NUMINAMATH_CALUDE_widgets_sold_is_3125_l2101_210121

/-- Represents Jenna's wholesale business --/
structure WholesaleBusiness where
  buy_price : ℝ
  sell_price : ℝ
  rent : ℝ
  tax_rate : ℝ
  worker_salary : ℝ
  num_workers : ℕ
  profit_after_tax : ℝ

/-- Calculates the number of widgets sold given the business parameters --/
def widgets_sold (b : WholesaleBusiness) : ℕ :=
  sorry

/-- Theorem stating that the number of widgets sold is 3125 --/
theorem widgets_sold_is_3125 (jenna : WholesaleBusiness) 
  (h1 : jenna.buy_price = 3)
  (h2 : jenna.sell_price = 8)
  (h3 : jenna.rent = 10000)
  (h4 : jenna.tax_rate = 0.2)
  (h5 : jenna.worker_salary = 2500)
  (h6 : jenna.num_workers = 4)
  (h7 : jenna.profit_after_tax = 4000) :
  widgets_sold jenna = 3125 :=
sorry

end NUMINAMATH_CALUDE_widgets_sold_is_3125_l2101_210121


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l2101_210176

theorem sqrt_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (45 * q) * Real.sqrt (15 * q) * Real.sqrt (10 * q) = 30 * q * Real.sqrt (15 * q) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l2101_210176


namespace NUMINAMATH_CALUDE_prob_three_unused_correct_expected_hits_correct_l2101_210111

-- Define the probability of hitting a target with a single shot
variable (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1)

-- Define the number of rockets and targets
def num_rockets : ℕ := 10
def num_targets_a : ℕ := 5
def num_targets_b : ℕ := 9

-- Part (a): Probability of exactly three unused rockets
def prob_three_unused : ℝ := 10 * p^3 * (1-p)^2

-- Part (b): Expected number of targets hit
def expected_hits : ℝ := 10*p - p^10

-- Theorem statements
theorem prob_three_unused_correct :
  prob_three_unused p = 10 * p^3 * (1-p)^2 :=
sorry

theorem expected_hits_correct :
  expected_hits p = 10*p - p^10 :=
sorry

end NUMINAMATH_CALUDE_prob_three_unused_correct_expected_hits_correct_l2101_210111


namespace NUMINAMATH_CALUDE_constraint_extrema_l2101_210167

def constraint (x y : ℝ) : Prop :=
  Real.sqrt (x - 3) + Real.sqrt (y - 4) = 4

def objective (x y : ℝ) : ℝ :=
  2 * x + 3 * y

theorem constraint_extrema :
  ∃ (x_min y_min x_max y_max : ℝ),
    constraint x_min y_min ∧
    constraint x_max y_max ∧
    (∀ x y, constraint x y → objective x y ≥ objective x_min y_min) ∧
    (∀ x y, constraint x y → objective x y ≤ objective x_max y_max) ∧
    x_min = 219 / 25 ∧
    y_min = 264 / 25 ∧
    x_max = 3 ∧
    y_max = 20 ∧
    objective x_min y_min = 37.2 ∧
    objective x_max y_max = 66 :=
  sorry

#check constraint_extrema

end NUMINAMATH_CALUDE_constraint_extrema_l2101_210167


namespace NUMINAMATH_CALUDE_unique_factorial_sum_power_l2101_210163

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def factorial_sum (m : ℕ) : ℕ := (List.range m).map factorial |>.sum

theorem unique_factorial_sum_power : 
  ∃! (m n k : ℕ), 
    m > 1 ∧ 
    n^k > 1 ∧ 
    factorial_sum m = n^k ∧
    m = 3 ∧ 
    n = 3 ∧ 
    k = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_factorial_sum_power_l2101_210163


namespace NUMINAMATH_CALUDE_wall_photo_area_l2101_210194

theorem wall_photo_area (paper_width paper_length frame_width : ℕ) 
  (hw : paper_width = 8)
  (hl : paper_length = 12)
  (hf : frame_width = 2) : 
  (paper_width + 2 * frame_width) * (paper_length + 2 * frame_width) = 192 := by
  sorry

end NUMINAMATH_CALUDE_wall_photo_area_l2101_210194


namespace NUMINAMATH_CALUDE_car_mileage_comparison_l2101_210131

/-- Calculates the weighted average miles per gallon and compares it to the advertised mileage -/
theorem car_mileage_comparison 
  (advertised_mpg : ℝ)
  (tank_capacity : ℝ)
  (regular_amount premium_amount diesel_amount : ℝ)
  (regular_mpg premium_mpg diesel_mpg : ℝ)
  (h1 : advertised_mpg = 35)
  (h2 : tank_capacity = 12)
  (h3 : regular_amount = 4)
  (h4 : premium_amount = 4)
  (h5 : diesel_amount = 4)
  (h6 : regular_mpg = 30)
  (h7 : premium_mpg = 40)
  (h8 : diesel_mpg = 32)
  (h9 : regular_amount + premium_amount + diesel_amount = tank_capacity) :
  advertised_mpg - (regular_amount * regular_mpg + premium_amount * premium_mpg + diesel_amount * diesel_mpg) / tank_capacity = 1 := by
  sorry

#check car_mileage_comparison

end NUMINAMATH_CALUDE_car_mileage_comparison_l2101_210131


namespace NUMINAMATH_CALUDE_triangle_count_proof_l2101_210136

/-- The number of triangles formed by 9 distinct lines in a plane -/
def num_triangles : ℕ := 23

/-- The total number of ways to choose 3 lines from 9 lines -/
def total_combinations : ℕ := Nat.choose 9 3

/-- The number of intersections where exactly three lines meet -/
def num_intersections : ℕ := 61

theorem triangle_count_proof :
  num_triangles = total_combinations - num_intersections :=
by sorry

end NUMINAMATH_CALUDE_triangle_count_proof_l2101_210136


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_6_mod_19_l2101_210145

theorem least_five_digit_congruent_to_6_mod_19 :
  ∃ n : ℕ, 
    n ≥ 10000 ∧ 
    n < 100000 ∧ 
    n % 19 = 6 ∧ 
    ∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 19 = 6 → m ≥ n :=
by
  use 10011
  sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_6_mod_19_l2101_210145


namespace NUMINAMATH_CALUDE_fixed_point_sets_l2101_210169

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- Define set A
def A (a b : ℝ) : Set ℝ := {x | f a b x = x}

-- Define set B
def B (a b : ℝ) : Set ℝ := {x | f a b (f a b x) = x}

-- Theorem statement
theorem fixed_point_sets (a b : ℝ) :
  A a b = {-1, 3} →
  B a b = {-Real.sqrt 3, -1, Real.sqrt 3, 3} ∧ A a b ⊆ B a b := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_sets_l2101_210169


namespace NUMINAMATH_CALUDE_calculate_expression_l2101_210192

theorem calculate_expression : 
  Real.sqrt 4 - abs (-1/4 : ℝ) + (π - 2)^0 + 2^(-2 : ℝ) = 3 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_l2101_210192


namespace NUMINAMATH_CALUDE_stool_height_l2101_210158

-- Define the constants
def ceiling_height : ℝ := 300  -- in cm
def bulb_below_ceiling : ℝ := 15  -- in cm
def alice_height : ℝ := 160  -- in cm
def alice_reach : ℝ := 50  -- in cm

-- Define the theorem
theorem stool_height : 
  ∃ (h : ℝ), 
    h = ceiling_height - bulb_below_ceiling - (alice_height + alice_reach) ∧ 
    h = 75 :=
by sorry

end NUMINAMATH_CALUDE_stool_height_l2101_210158


namespace NUMINAMATH_CALUDE_median_is_55_l2101_210166

/-- A set of consecutive integers with a specific property --/
structure ConsecutiveIntegerSet where
  first : ℤ  -- The first integer in the set
  count : ℕ  -- The number of integers in the set
  sum_property : ∀ n : ℕ, n ≤ count → first + (n - 1) + (first + (count - n)) = 110

/-- The median of a set of consecutive integers --/
def median (s : ConsecutiveIntegerSet) : ℚ :=
  (s.first + (s.first + (s.count - 1))) / 2

/-- Theorem: The median of the ConsecutiveIntegerSet is always 55 --/
theorem median_is_55 (s : ConsecutiveIntegerSet) : median s = 55 := by
  sorry


end NUMINAMATH_CALUDE_median_is_55_l2101_210166


namespace NUMINAMATH_CALUDE_multiply_99_105_l2101_210155

theorem multiply_99_105 : 99 * 105 = 10395 := by
  sorry

end NUMINAMATH_CALUDE_multiply_99_105_l2101_210155


namespace NUMINAMATH_CALUDE_april_earnings_l2101_210138

/-- The price of a rose in dollars -/
def rose_price : ℕ := 7

/-- The price of a lily in dollars -/
def lily_price : ℕ := 5

/-- The initial number of roses -/
def initial_roses : ℕ := 9

/-- The initial number of lilies -/
def initial_lilies : ℕ := 6

/-- The remaining number of roses -/
def remaining_roses : ℕ := 4

/-- The remaining number of lilies -/
def remaining_lilies : ℕ := 2

/-- The total earnings from the sale -/
def total_earnings : ℕ := 55

theorem april_earnings : 
  (initial_roses - remaining_roses) * rose_price + 
  (initial_lilies - remaining_lilies) * lily_price = total_earnings := by
  sorry

end NUMINAMATH_CALUDE_april_earnings_l2101_210138


namespace NUMINAMATH_CALUDE_qz_length_l2101_210191

/-- A quadrilateral ABZY with a point Q on the intersection of AZ and BY -/
structure Quadrilateral :=
  (A B Y Z Q : ℝ × ℝ)
  (AB_parallel_YZ : (A.2 - B.2) / (A.1 - B.1) = (Y.2 - Z.2) / (Y.1 - Z.1))
  (Q_on_AZ : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • A + t • Z)
  (Q_on_BY : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ Q = (1 - s) • B + s • Y)
  (AZ_length : Real.sqrt ((A.1 - Z.1)^2 + (A.2 - Z.2)^2) = 42)
  (BQ_length : Real.sqrt ((B.1 - Q.1)^2 + (B.2 - Q.2)^2) = 12)
  (QY_length : Real.sqrt ((Q.1 - Y.1)^2 + (Q.2 - Y.2)^2) = 24)

/-- The length of QZ in the given quadrilateral is 28 -/
theorem qz_length (quad : Quadrilateral) :
  Real.sqrt ((quad.Q.1 - quad.Z.1)^2 + (quad.Q.2 - quad.Z.2)^2) = 28 := by
  sorry

end NUMINAMATH_CALUDE_qz_length_l2101_210191


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l2101_210159

theorem arithmetic_expression_evaluation : 3 + 2 * (8 - 3) = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l2101_210159


namespace NUMINAMATH_CALUDE_ratio_problem_l2101_210150

theorem ratio_problem (x y a : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x / y = 5 / a) 
  (h4 : (x + 12) / (y + 12) = 3 / 4) : y - x = 9 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2101_210150


namespace NUMINAMATH_CALUDE_train_speed_with_stoppages_l2101_210149

/-- Given a train that travels at 80 km/h without stoppages and stops for 15 minutes every hour,
    its average speed with stoppages is 60 km/h. -/
theorem train_speed_with_stoppages :
  let speed_without_stoppages : ℝ := 80
  let stop_time_per_hour : ℝ := 15/60
  let speed_with_stoppages : ℝ := speed_without_stoppages * (1 - stop_time_per_hour)
  speed_with_stoppages = 60 := by sorry

end NUMINAMATH_CALUDE_train_speed_with_stoppages_l2101_210149


namespace NUMINAMATH_CALUDE_polynomial_inequality_l2101_210130

theorem polynomial_inequality (x : ℝ) : (x + 2) * (x - 8) * (x - 3) > 0 ↔ x ∈ Set.Ioo (-2 : ℝ) 3 ∪ Set.Ioi 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l2101_210130


namespace NUMINAMATH_CALUDE_clinton_shoes_count_l2101_210125

theorem clinton_shoes_count (hats belts shoes : ℕ) : 
  hats = 5 →
  belts = hats + 2 →
  shoes = 2 * belts →
  shoes = 14 := by
sorry

end NUMINAMATH_CALUDE_clinton_shoes_count_l2101_210125


namespace NUMINAMATH_CALUDE_potatoes_for_mashed_l2101_210127

theorem potatoes_for_mashed (initial : ℕ) (salad : ℕ) (remaining : ℕ) : 
  initial = 52 → salad = 15 → remaining = 13 → initial - salad - remaining = 24 := by
  sorry

end NUMINAMATH_CALUDE_potatoes_for_mashed_l2101_210127


namespace NUMINAMATH_CALUDE_chloe_trivia_score_l2101_210141

/-- The total points scored in a trivia game with three rounds -/
def trivia_game_score (round1 : Int) (round2 : Int) (round3 : Int) : Int :=
  round1 + round2 + round3

/-- Theorem: The total points at the end of Chloe's trivia game is 86 -/
theorem chloe_trivia_score : trivia_game_score 40 50 (-4) = 86 := by
  sorry

end NUMINAMATH_CALUDE_chloe_trivia_score_l2101_210141


namespace NUMINAMATH_CALUDE_root_cubic_expression_l2101_210104

theorem root_cubic_expression (m : ℝ) : 
  m^2 + 3*m - 2022 = 0 → m^3 + 4*m^2 - 2019*m - 2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_root_cubic_expression_l2101_210104


namespace NUMINAMATH_CALUDE_probability_is_one_over_432_l2101_210153

/-- Represents a fair die with 6 faces -/
def Die := Fin 6

/-- Represents the outcome of tossing four dice -/
def FourDiceOutcome := (Die × Die × Die × Die)

/-- Checks if a sequence of four numbers forms an arithmetic progression with common difference 1 -/
def isArithmeticProgression (a b c d : ℕ) : Prop :=
  b - a = 1 ∧ c - b = 1 ∧ d - c = 1

/-- The set of all possible outcomes when tossing four fair dice -/
def allOutcomes : Finset FourDiceOutcome := sorry

/-- The set of favorable outcomes (forming an arithmetic progression) -/
def favorableOutcomes : Finset FourDiceOutcome := sorry

/-- The probability of getting an arithmetic progression when tossing four fair dice -/
def probabilityOfArithmeticProgression : ℚ :=
  (favorableOutcomes.card : ℚ) / (allOutcomes.card : ℚ)

/-- Theorem stating that the probability of getting an arithmetic progression is 1/432 -/
theorem probability_is_one_over_432 :
  probabilityOfArithmeticProgression = 1 / 432 := by sorry

end NUMINAMATH_CALUDE_probability_is_one_over_432_l2101_210153


namespace NUMINAMATH_CALUDE_sum_of_reversed_square_digits_l2101_210177

/-- The number to be squared -/
def n : ℕ := 11111

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (m : ℕ) : ℕ := sorry

/-- Function to reverse the digits of a natural number -/
def reverse_digits (m : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of the digits of the reversed square of 11111 is 25 -/
theorem sum_of_reversed_square_digits : sum_of_digits (reverse_digits (n^2)) = 25 := by sorry

end NUMINAMATH_CALUDE_sum_of_reversed_square_digits_l2101_210177


namespace NUMINAMATH_CALUDE_square_triangle_area_ratio_l2101_210186

/-- Given a square with side length s, where R is the midpoint of one side,
    S is the midpoint of a diagonal, and V is a vertex,
    prove that the area of triangle RSV is √2/16 of the square's area. -/
theorem square_triangle_area_ratio (s : ℝ) (h : s > 0) :
  let square_area := s^2
  let r_to_s := s / 2
  let s_to_v := s * Real.sqrt 2 / 2
  let r_to_v := s
  let triangle_height := s * Real.sqrt 2 / 4
  let triangle_area := 1 / 2 * r_to_s * triangle_height
  triangle_area / square_area = Real.sqrt 2 / 16 := by
sorry

end NUMINAMATH_CALUDE_square_triangle_area_ratio_l2101_210186


namespace NUMINAMATH_CALUDE_starters_count_l2101_210100

-- Define the total number of players
def total_players : ℕ := 16

-- Define the number of triplets
def num_triplets : ℕ := 3

-- Define the number of twins
def num_twins : ℕ := 2

-- Define the number of starters to be chosen
def num_starters : ℕ := 6

-- Define the function to calculate the number of ways to choose starters
def choose_starters (total : ℕ) (triplets : ℕ) (twins : ℕ) (starters : ℕ) : ℕ :=
  -- No triplets and no twins
  Nat.choose (total - triplets - twins) starters +
  -- One triplet and no twins
  triplets * Nat.choose (total - triplets - twins) (starters - 1) +
  -- No triplets and one twin
  twins * Nat.choose (total - triplets - twins) (starters - 1) +
  -- One triplet and one twin
  triplets * twins * Nat.choose (total - triplets - twins) (starters - 2)

-- Theorem statement
theorem starters_count :
  choose_starters total_players num_triplets num_twins num_starters = 4752 :=
by sorry

end NUMINAMATH_CALUDE_starters_count_l2101_210100


namespace NUMINAMATH_CALUDE_monkey_hop_distance_l2101_210199

/-- Represents the climbing problem of a monkey on a tree. -/
def monkey_climb (tree_height : ℝ) (total_hours : ℕ) (slip_distance : ℝ) (hop_distance : ℝ) : Prop :=
  let net_climb_per_hour := hop_distance - slip_distance
  (total_hours - 1 : ℝ) * net_climb_per_hour + hop_distance = tree_height

/-- Theorem stating that for the given conditions, the monkey must hop 3 feet each hour. -/
theorem monkey_hop_distance :
  monkey_climb 20 18 2 3 :=
sorry

end NUMINAMATH_CALUDE_monkey_hop_distance_l2101_210199


namespace NUMINAMATH_CALUDE_interview_segment_ratio_l2101_210134

/-- Represents the lengths of three interview segments in a radio show. -/
structure InterviewSegments where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Theorem stating the ratio of the third segment to the second segment is 1:2
    given the conditions of the radio show. -/
theorem interview_segment_ratio
  (segments : InterviewSegments)
  (total_time : segments.first + segments.second + segments.third = 90)
  (first_twice_others : segments.first = 2 * (segments.second + segments.third))
  (third_length : segments.third = 10) :
  segments.third / segments.second = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_interview_segment_ratio_l2101_210134


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2101_210113

theorem sum_of_coefficients (b₀ b₁ b₂ b₃ b₄ b₅ b₆ : ℝ) :
  (∀ x : ℝ, (2*x + 3)^6 = b₆*x^6 + b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₆ + b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 15625 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2101_210113


namespace NUMINAMATH_CALUDE_line_vector_at_zero_l2101_210154

def line_vector (t : ℝ) : ℝ × ℝ × ℝ := sorry

theorem line_vector_at_zero : 
  (∀ (t : ℝ), line_vector t = line_vector 0 + t • (line_vector 1 - line_vector 0)) →
  line_vector (-2) = (2, 4, 10) →
  line_vector 1 = (-1, -3, -5) →
  line_vector 0 = (0, -2/3, 0) := by sorry

end NUMINAMATH_CALUDE_line_vector_at_zero_l2101_210154


namespace NUMINAMATH_CALUDE_pond_to_field_area_ratio_l2101_210122

/-- Proves that the ratio of a square pond's area to a rectangular field's area is 1:8,
    given specific dimensions of the field and pond. -/
theorem pond_to_field_area_ratio :
  ∀ (field_length field_width pond_side : ℝ),
    field_length = 36 →
    field_length = 2 * field_width →
    pond_side = 9 →
    (pond_side^2) / (field_length * field_width) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_pond_to_field_area_ratio_l2101_210122


namespace NUMINAMATH_CALUDE_min_questions_to_determine_order_l2101_210195

/-- Represents a question that reveals the relative order of 50 numbers -/
def Question := Fin 100 → Prop

/-- The set of all possible permutations of numbers from 1 to 100 -/
def Permutations := Fin 100 → Fin 100

/-- A function that determines if a given permutation is consistent with the answers to all questions -/
def IsConsistent (p : Permutations) (qs : List Question) : Prop := sorry

/-- The minimum number of questions needed to determine the order of 100 integers -/
def MinQuestions : ℕ := 5

theorem min_questions_to_determine_order :
  ∀ (qs : List Question),
    (∀ (p₁ p₂ : Permutations), IsConsistent p₁ qs ∧ IsConsistent p₂ qs → p₁ = p₂) →
    qs.length ≥ MinQuestions :=
sorry

end NUMINAMATH_CALUDE_min_questions_to_determine_order_l2101_210195


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2101_210183

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)) →  -- Definition of S_n
  a 1 = -2011 →                                            -- Given a_1
  (S 2010 / 2010) - (S 2008 / 2008) = 2 →                  -- Given condition
  S 2011 = -2011 :=                                        -- Conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2101_210183


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l2101_210126

theorem parallel_vectors_sum (x y : ℝ) 
  (a : ℝ × ℝ × ℝ) (b : ℝ × ℝ × ℝ) 
  (ha : a = (2, 1, x)) (hb : b = (4, y, -1)) 
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ a = k • b) : 
  2 * x + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l2101_210126


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l2101_210179

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (Complex.I : ℂ) / (1 + a * Complex.I) = b * Complex.I) → a = 0 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l2101_210179


namespace NUMINAMATH_CALUDE_equation_solutions_l2101_210114

noncomputable def floor (x : ℝ) : ℤ :=
  ⌊x⌋

def is_solution (x : ℝ) : Prop :=
  x ≠ 0.5 ∧ (floor x : ℝ) - Real.sqrt ((floor x : ℝ) / (x - 0.5)) - 6 / (x - 0.5) = 0

theorem equation_solutions :
  ∀ x : ℝ, is_solution x ↔ (x = -1.5 ∨ x = 3.5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2101_210114


namespace NUMINAMATH_CALUDE_line_symmetry_theorem_l2101_210106

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Check if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

/-- Two lines are symmetric about the y-axis -/
def symmetric_about_y_axis (l₁ l₂ : Line) : Prop :=
  l₁.slope = -l₂.slope ∧ l₁.intercept = l₂.intercept

theorem line_symmetry_theorem (b : ℝ) :
  let l₁ : Line := { slope := -2, intercept := b }
  let l₂ : Line := { slope := 2, intercept := 4 }
  symmetric_about_y_axis l₁ l₂ ∧ l₂.contains 1 6 → b = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_symmetry_theorem_l2101_210106


namespace NUMINAMATH_CALUDE_delores_money_left_l2101_210107

/-- Calculates the money left after purchases given initial amount and costs --/
def money_left (initial_amount : ℕ) (computer_cost : ℕ) (printer_cost : ℕ) : ℕ :=
  initial_amount - (computer_cost + printer_cost)

theorem delores_money_left :
  money_left 450 400 40 = 10 := by
  sorry

end NUMINAMATH_CALUDE_delores_money_left_l2101_210107


namespace NUMINAMATH_CALUDE_echo_earnings_l2101_210110

-- Define the schools and their parameters
structure School where
  name : String
  students : ℕ
  days : ℕ
  rate_multiplier : ℚ

-- Define the problem parameters
def delta : School := { name := "Delta", students := 8, days := 4, rate_multiplier := 1 }
def echo : School := { name := "Echo", students := 6, days := 6, rate_multiplier := 3/2 }
def foxtrot : School := { name := "Foxtrot", students := 7, days := 7, rate_multiplier := 1 }

def total_payment : ℚ := 1284

-- Function to calculate effective student-days
def effective_student_days (s : School) : ℚ :=
  ↑s.students * ↑s.days * s.rate_multiplier

-- Theorem statement
theorem echo_earnings :
  let total_effective_days := effective_student_days delta + effective_student_days echo + effective_student_days foxtrot
  let daily_wage := total_payment / total_effective_days
  effective_student_days echo * daily_wage = 513.6 := by
sorry

end NUMINAMATH_CALUDE_echo_earnings_l2101_210110


namespace NUMINAMATH_CALUDE_six_hundred_million_scientific_notation_l2101_210171

-- Define 600 million
def six_hundred_million : ℝ := 600000000

-- Theorem statement
theorem six_hundred_million_scientific_notation :
  six_hundred_million = 6 * 10^8 := by
  sorry

end NUMINAMATH_CALUDE_six_hundred_million_scientific_notation_l2101_210171


namespace NUMINAMATH_CALUDE_g_limit_pos_infinity_g_limit_neg_infinity_l2101_210147

/-- The function g(x) = -3x^4 + 5x^3 - 6 -/
def g (x : ℝ) : ℝ := -3 * x^4 + 5 * x^3 - 6

/-- The limit of g(x) approaches negative infinity as x approaches positive infinity -/
theorem g_limit_pos_infinity :
  ∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M :=
sorry

/-- The limit of g(x) approaches negative infinity as x approaches negative infinity -/
theorem g_limit_neg_infinity :
  ∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x < M :=
sorry

end NUMINAMATH_CALUDE_g_limit_pos_infinity_g_limit_neg_infinity_l2101_210147


namespace NUMINAMATH_CALUDE_martin_answered_fewer_than_43_l2101_210132

/-- The number of questions answered correctly by each person -/
structure QuizResults where
  campbell : ℕ
  kelsey : ℕ
  martin : ℕ

/-- The conditions of the quiz results -/
def QuizConditions (results : QuizResults) : Prop :=
  results.campbell = 35 ∧
  results.kelsey = results.campbell + 8 ∧
  results.martin < results.kelsey

theorem martin_answered_fewer_than_43 (results : QuizResults) 
  (h : QuizConditions results) : results.martin < 43 := by
  sorry

end NUMINAMATH_CALUDE_martin_answered_fewer_than_43_l2101_210132


namespace NUMINAMATH_CALUDE_square_area_difference_l2101_210193

theorem square_area_difference : 
  ∀ (smaller_length greater_length : ℝ),
    greater_length = 7 →
    greater_length = smaller_length + 2 →
    (greater_length ^ 2 - smaller_length ^ 2 : ℝ) = 24 := by
  sorry

end NUMINAMATH_CALUDE_square_area_difference_l2101_210193


namespace NUMINAMATH_CALUDE_no_valid_permutation_1986_l2101_210142

/-- Represents a permutation of the sequence 1,1,2,2,...,n,n -/
def Permutation (n : ℕ) := Fin (2*n) → Fin n

/-- The separation between pairs in a permutation -/
def separation (n : ℕ) (p : Permutation n) (i : Fin n) : ℕ := sorry

/-- A permutation satisfies the separation condition if for each i,
    there are exactly i numbers between the two occurrences of i -/
def satisfies_separation (n : ℕ) (p : Permutation n) : Prop :=
  ∀ i : Fin n, separation n p i = i.val

/-- The main theorem: there is no permutation of 1,1,2,2,...,1986,1986
    that satisfies the separation condition -/
theorem no_valid_permutation_1986 :
  ¬ ∃ (p : Permutation 1986), satisfies_separation 1986 p :=
sorry

end NUMINAMATH_CALUDE_no_valid_permutation_1986_l2101_210142


namespace NUMINAMATH_CALUDE_employee_salary_problem_l2101_210115

/-- Given two employees M and N with a total weekly salary of $605,
    where M's salary is 120% of N's, prove that N's salary is $275 per week. -/
theorem employee_salary_problem (total_salary m_salary n_salary : ℝ) :
  total_salary = 605 →
  m_salary = 1.2 * n_salary →
  total_salary = m_salary + n_salary →
  n_salary = 275 := by
sorry

end NUMINAMATH_CALUDE_employee_salary_problem_l2101_210115


namespace NUMINAMATH_CALUDE_subtraction_result_l2101_210118

theorem subtraction_result : 888888888888 - 111111111111 = 777777777777 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l2101_210118


namespace NUMINAMATH_CALUDE_line_through_point_equal_intercepts_l2101_210170

/-- A line passing through (2,3) with equal absolute intercepts -/
theorem line_through_point_equal_intercepts :
  ∃ (m c : ℝ), 
    (3 = 2 * m + c) ∧ 
    (|c| = |c / m|) ∧
    (∀ x y : ℝ, y = m * x + c ↔ y = x + 1) := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_equal_intercepts_l2101_210170


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l2101_210185

/-- Proves that the percentage increase resulting in a $324 weekly salary is 8%,
    given that a 10% increase results in a $330 weekly salary. -/
theorem salary_increase_percentage (current_salary : ℝ) : 
  (current_salary * 1.1 = 330) →
  (current_salary * (1 + 0.08) = 324) := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l2101_210185


namespace NUMINAMATH_CALUDE_right_triangle_sine_inequality_l2101_210128

/-- Given a right-angled triangle with hypotenuse parallel to plane α, and angles θ₁ and θ₂
    between the lines containing the two legs of the triangle and plane α,
    prove that sin²θ₁ + sin²θ₂ ≤ 1 -/
theorem right_triangle_sine_inequality (θ₁ θ₂ : Real) 
    (h₁ : 0 ≤ θ₁ ∧ θ₁ ≤ π / 2) 
    (h₂ : 0 ≤ θ₂ ∧ θ₂ ≤ π / 2) 
    (h_right_angle : θ₁ + θ₂ ≤ π / 2) : 
    Real.sin θ₁ ^ 2 + Real.sin θ₂ ^ 2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sine_inequality_l2101_210128


namespace NUMINAMATH_CALUDE_lily_typing_break_l2101_210178

-- Define Lily's typing parameters
def typing_rate : ℝ := 15  -- words per minute
def break_duration : ℝ := 2  -- minutes
def total_words : ℝ := 255
def total_time : ℝ := 19  -- minutes

-- Define the function to calculate typing time before break
def typing_time_before_break : ℝ → ℝ := λ x => x

-- Theorem statement
theorem lily_typing_break (typing_time : ℝ) : 
  typing_time_before_break typing_time = 8 ↔ 
  (typing_rate * typing_time * 2 = total_words ∧ 
   typing_time * 2 + break_duration = total_time) :=
by sorry

end NUMINAMATH_CALUDE_lily_typing_break_l2101_210178


namespace NUMINAMATH_CALUDE_f_pi_sixth_value_l2101_210187

/-- Given a function f(x) = 2sin(ωx + φ) where for all x, f(π/3 + x) = f(-x),
    prove that f(π/6) is either -2 or 2. -/
theorem f_pi_sixth_value (ω φ : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sin (ω * x + φ)
  (∀ x, f (π / 3 + x) = f (-x)) →
  f (π / 6) = -2 ∨ f (π / 6) = 2 :=
by sorry

end NUMINAMATH_CALUDE_f_pi_sixth_value_l2101_210187


namespace NUMINAMATH_CALUDE_pedal_triangles_common_circumcircle_l2101_210172

/-- Triangle type -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Isotomic conjugates with respect to a triangle -/
def IsotomicConjugates (P₁ P₂ : Point) (T : Triangle) : Prop := sorry

/-- Pedal triangle of a point with respect to a triangle, given an angle -/
def PedalTriangle (P : Point) (T : Triangle) (angle : ℝ) : Triangle := sorry

/-- Circumcircle of a triangle -/
def Circumcircle (T : Triangle) : Circle := sorry

/-- Center of a circle -/
def Center (C : Circle) : Point := sorry

/-- Midpoint of a segment -/
def Midpoint (A B : Point) : Point := sorry

theorem pedal_triangles_common_circumcircle 
  (T : Triangle) (P₁ P₂ : Point) (angle : ℝ) :
  IsotomicConjugates P₁ P₂ T →
  ∃ (C : Circle), 
    Circumcircle (PedalTriangle P₁ T angle) = C ∧
    Circumcircle (PedalTriangle P₂ T angle) = C ∧
    Center C = Midpoint P₁ P₂ := by
  sorry

end NUMINAMATH_CALUDE_pedal_triangles_common_circumcircle_l2101_210172


namespace NUMINAMATH_CALUDE_power_of_three_mod_five_l2101_210146

theorem power_of_three_mod_five : 3^2000 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_five_l2101_210146


namespace NUMINAMATH_CALUDE_log_difference_cube_l2101_210184

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem log_difference_cube (x y a : ℝ) (h : x > 0) (h' : y > 0) :
  lg x - lg y = a → lg ((x / 2) ^ 3) - lg ((y / 2) ^ 3) = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_log_difference_cube_l2101_210184


namespace NUMINAMATH_CALUDE_complex_number_equal_parts_l2101_210161

theorem complex_number_equal_parts (a : ℝ) : 
  let z : ℂ := (a + Complex.I) * Complex.I
  (z.re = z.im) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equal_parts_l2101_210161


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2101_210123

theorem pure_imaginary_complex_number (θ : Real) : 
  θ ∈ Set.Icc 0 (2 * Real.pi) →
  (∃ (y : Real), (Complex.cos θ + Complex.I) * (2 * Complex.sin θ - Complex.I) = Complex.I * y) →
  θ = 3 * Real.pi / 4 ∨ θ = 7 * Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2101_210123


namespace NUMINAMATH_CALUDE_factorization_equality_l2101_210164

theorem factorization_equality (x y : ℝ) : x + x^2 - y - y^2 = (x + y + 1) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2101_210164


namespace NUMINAMATH_CALUDE_infinitely_many_primary_triplets_l2101_210168

/-- A primary triplet is a triplet of positive integers (x, y, z) satisfying
    x, y, z > 1 and x^3 - yz^3 = 2021, where at least two of x, y, z are prime numbers. -/
def PrimaryTriplet (x y z : ℕ) : Prop :=
  x > 1 ∧ y > 1 ∧ z > 1 ∧
  x^3 - y*z^3 = 2021 ∧
  (Nat.Prime x ∧ Nat.Prime y) ∨ (Nat.Prime x ∧ Nat.Prime z) ∨ (Nat.Prime y ∧ Nat.Prime z)

/-- There exist infinitely many primary triplets. -/
theorem infinitely_many_primary_triplets :
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ ∃ x y z : ℕ, PrimaryTriplet x y z :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primary_triplets_l2101_210168


namespace NUMINAMATH_CALUDE_f_properties_l2101_210116

def f (x : ℝ) : ℝ := x * (x + 1) * (x + 2) * (x + 3)

theorem f_properties :
  (∀ x : ℝ, f x ≥ -1) ∧
  (∃ x : ℝ, f x = -1) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-2) (-1) → f x ≤ -9/16) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-2) (-1) ∧ f x = -9/16) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2101_210116


namespace NUMINAMATH_CALUDE_infinitely_many_coprime_pairs_l2101_210175

theorem infinitely_many_coprime_pairs (m : ℤ) :
  ∃ f : ℕ → ℤ × ℤ, ∀ n : ℕ,
    let (x, y) := f n
    -- Condition 1: x and y are coprime
    Int.gcd x y = 1 ∧
    -- Condition 2: y divides x^2 + m
    (x^2 + m) % y = 0 ∧
    -- Condition 3: x divides y^2 + m
    (y^2 + m) % x = 0 ∧
    -- Ensure infinitely many distinct pairs
    (∀ k < n, f k ≠ f n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_coprime_pairs_l2101_210175


namespace NUMINAMATH_CALUDE_expansion_coefficient_a_l2101_210197

theorem expansion_coefficient_a (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x + 1)^5 = a + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + a₄*(x - 1)^4 + a₅*(x - 1)^5) →
  a = 32 := by
sorry

end NUMINAMATH_CALUDE_expansion_coefficient_a_l2101_210197


namespace NUMINAMATH_CALUDE_four_line_segment_lengths_exists_distinct_positive_integer_lengths_l2101_210135

-- Define a type for lines in a plane
def Line : Type := ℝ → ℝ → Prop

-- Define a type for points in a plane
def Point : Type := ℝ × ℝ

-- Define a function to check if three lines are concurrent
def areConcurrent (l₁ l₂ l₃ : Line) : Prop := sorry

-- Define a function to check if two lines intersect
def intersect (l₁ l₂ : Line) : Prop := sorry

-- Define a function to get the length of a line segment
def segmentLength (p₁ p₂ : Point) : ℝ := sorry

-- Define the configuration of four lines
structure FourLineConfiguration :=
  (lines : Fin 4 → Line)
  (intersectionPoints : Fin 6 → Point)
  (segmentLengths : Fin 8 → ℝ)
  (twoLinesIntersect : ∀ i j, i ≠ j → intersect (lines i) (lines j))
  (noThreeConcurrent : ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬areConcurrent (lines i) (lines j) (lines k))
  (eightSegments : ∀ i, segmentLengths i > 0)
  (distinctSegments : ∀ i j, i ≠ j → segmentLengths i ≠ segmentLengths j)

theorem four_line_segment_lengths 
  (config : FourLineConfiguration) : 
  (∀ i : Fin 8, config.segmentLengths i = i.val + 1) → False :=
sorry

theorem exists_distinct_positive_integer_lengths 
  (config : FourLineConfiguration) :
  ∃ (lengths : Fin 8 → ℕ), ∀ i : Fin 8, config.segmentLengths i = lengths i ∧ lengths i > 0 :=
sorry

end NUMINAMATH_CALUDE_four_line_segment_lengths_exists_distinct_positive_integer_lengths_l2101_210135


namespace NUMINAMATH_CALUDE_inequality_proof_l2101_210140

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^2 + y^4 + z^6 ≥ x*y^2 + y^2*z^3 + x*z^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2101_210140


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2101_210152

/-- Given constants a, b, c, where P(a,c) is in the fourth quadrant,
    prove that ax^2 + bx + c = 0 has two distinct real roots -/
theorem quadratic_equation_roots (a b c : ℝ) 
  (h1 : a > 0) (h2 : c < 0) : 
  ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2101_210152


namespace NUMINAMATH_CALUDE_intersection_line_slope_l2101_210117

-- Define the two circle equations
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 10*x + 8*y + 24 = 0

-- Define the slope of the line passing through the intersection points
def slope_of_intersection_line (circle1 circle2 : ℝ → ℝ → Prop) : ℝ := 1

-- Theorem statement
theorem intersection_line_slope :
  slope_of_intersection_line circle1 circle2 = 1 := by sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l2101_210117


namespace NUMINAMATH_CALUDE_cubic_difference_999_l2101_210165

theorem cubic_difference_999 : 
  ∀ m n : ℕ+, m^3 - n^3 = 999 ↔ (m = 10 ∧ n = 1) ∨ (m = 12 ∧ n = 9) := by
sorry

end NUMINAMATH_CALUDE_cubic_difference_999_l2101_210165


namespace NUMINAMATH_CALUDE_delta_eight_four_l2101_210173

/-- The Δ operation for non-zero integers -/
def delta (a b : ℤ) : ℚ :=
  a - a / b

/-- Theorem stating that 8 Δ 4 = 6 -/
theorem delta_eight_four : delta 8 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_delta_eight_four_l2101_210173


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2101_210124

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) / a n = a 2 / a 1

-- State the theorem
theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a → a 2 * a 4 * a 12 = 64 → a 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2101_210124


namespace NUMINAMATH_CALUDE_parabola_point_ordering_l2101_210188

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Define the points A, B, and C
def A : ℝ × ℝ := (0, f 0)
def B : ℝ × ℝ := (1, f 1)
def C : ℝ × ℝ := (-2, f (-2))

-- Define y₁, y₂, and y₃
def y₁ : ℝ := A.2
def y₂ : ℝ := B.2
def y₃ : ℝ := C.2

-- Theorem statement
theorem parabola_point_ordering : y₃ > y₁ ∧ y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_ordering_l2101_210188


namespace NUMINAMATH_CALUDE_james_writing_hours_l2101_210196

/-- Calculates the number of hours James spends writing per week -/
def writing_hours_per_week (pages_per_hour : ℕ) (pages_per_person_per_day : ℕ) (num_people : ℕ) (days_per_week : ℕ) : ℕ :=
  (pages_per_person_per_day * num_people * days_per_week) / pages_per_hour

theorem james_writing_hours (pages_per_hour : ℕ) (pages_per_person_per_day : ℕ) (num_people : ℕ) (days_per_week : ℕ)
  (h1 : pages_per_hour = 10)
  (h2 : pages_per_person_per_day = 5)
  (h3 : num_people = 2)
  (h4 : days_per_week = 7) :
  writing_hours_per_week pages_per_hour pages_per_person_per_day num_people days_per_week = 7 := by
  sorry

end NUMINAMATH_CALUDE_james_writing_hours_l2101_210196


namespace NUMINAMATH_CALUDE_fraction_simplification_l2101_210180

theorem fraction_simplification (m : ℝ) (h : m ≠ 3 ∧ m ≠ -3) : 
  12 / (m^2 - 9) + 2 / (3 - m) = -2 / (m + 3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2101_210180


namespace NUMINAMATH_CALUDE_least_comic_books_l2101_210103

theorem least_comic_books (n : ℕ) : n > 0 ∧ n % 7 = 3 ∧ n % 4 = 1 → n ≥ 17 :=
by sorry

end NUMINAMATH_CALUDE_least_comic_books_l2101_210103


namespace NUMINAMATH_CALUDE_inequality_not_true_l2101_210143

theorem inequality_not_true (x y : ℝ) (h : x > y) : ¬(-3*x + 6 > -3*y + 6) := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_true_l2101_210143


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l2101_210151

theorem ratio_x_to_y (x y : ℝ) (h : 0.8 * x = 0.2 * y) : x / y = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l2101_210151


namespace NUMINAMATH_CALUDE_pressure_area_relation_l2101_210133

/-- Proves that given pressure P = F/S, force F = 50N, and P > 500Pa, the area S < 0.1m² -/
theorem pressure_area_relation (F S P : ℝ) : 
  F = 50 → P = F / S → P > 500 → S < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_pressure_area_relation_l2101_210133


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_l2101_210174

/-- The value of a cow in dollars -/
def cow_value : ℕ := 500

/-- The value of a sheep in dollars -/
def sheep_value : ℕ := 350

/-- The smallest positive debt that can be resolved using cows and sheep -/
def smallest_debt : ℕ := 50

/-- Theorem stating that the smallest_debt is the smallest positive value that can be expressed as a linear combination of cow_value and sheep_value with integer coefficients -/
theorem smallest_resolvable_debt : 
  smallest_debt = Nat.gcd cow_value sheep_value ∧
  ∃ (c s : ℤ), smallest_debt = c * cow_value + s * sheep_value ∧
  ∀ (d : ℕ), d > 0 → (∃ (x y : ℤ), d = x * cow_value + y * sheep_value) → d ≥ smallest_debt :=
sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_l2101_210174


namespace NUMINAMATH_CALUDE_frank_first_half_correct_l2101_210102

/-- Represents the trivia game scenario -/
structure TriviaGame where
  points_per_question : ℕ
  final_score : ℕ
  second_half_correct : ℕ

/-- Calculates the number of questions answered correctly in the first half -/
def first_half_correct (game : TriviaGame) : ℕ :=
  (game.final_score - game.second_half_correct * game.points_per_question) / game.points_per_question

/-- Theorem stating that Frank answered 3 questions correctly in the first half -/
theorem frank_first_half_correct :
  let game : TriviaGame := {
    points_per_question := 3,
    final_score := 15,
    second_half_correct := 2
  }
  first_half_correct game = 3 := by
  sorry

end NUMINAMATH_CALUDE_frank_first_half_correct_l2101_210102


namespace NUMINAMATH_CALUDE_prob_second_draw_l2101_210119

structure Bag where
  red : ℕ
  blue : ℕ

def initial_bag : Bag := ⟨5, 4⟩

def P_A2 (b : Bag) : ℚ :=
  (b.red : ℚ) / (b.red + b.blue)

def P_B2 (b : Bag) : ℚ :=
  (b.blue : ℚ) / (b.red + b.blue)

def P_A2_given_A1 (b : Bag) : ℚ :=
  ((b.red - 1) : ℚ) / (b.red + b.blue - 1)

def P_B2_given_A1 (b : Bag) : ℚ :=
  (b.blue : ℚ) / (b.red + b.blue - 1)

theorem prob_second_draw (b : Bag) :
  P_A2 b = 5/9 ∧
  P_A2 b + P_B2 b = 1 ∧
  P_A2_given_A1 b + P_B2_given_A1 b = 1 :=
by sorry

end NUMINAMATH_CALUDE_prob_second_draw_l2101_210119


namespace NUMINAMATH_CALUDE_ninety_eight_squared_l2101_210148

theorem ninety_eight_squared : 98 * 98 = 9604 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ninety_eight_squared_l2101_210148


namespace NUMINAMATH_CALUDE_eight_dice_probability_l2101_210139

theorem eight_dice_probability : 
  let n : ℕ := 8  -- number of dice
  let k : ℕ := 4  -- number of dice showing even numbers
  let p : ℚ := 1/2  -- probability of a single die showing an even number
  Nat.choose n k * p^n = 35/128 := by
  sorry

end NUMINAMATH_CALUDE_eight_dice_probability_l2101_210139


namespace NUMINAMATH_CALUDE_percentage_selected_state_A_l2101_210181

theorem percentage_selected_state_A (
  total_A : ℕ) (total_B : ℕ) (percent_B : ℚ) (diff : ℕ) :
  total_A = 8000 →
  total_B = 8000 →
  percent_B = 7 / 100 →
  (total_B : ℚ) * percent_B = (total_A : ℚ) * (7 / 100 : ℚ) + diff →
  diff = 80 →
  (7 / 100 : ℚ) * total_A = (total_A : ℚ) * (7 / 100 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_percentage_selected_state_A_l2101_210181


namespace NUMINAMATH_CALUDE_monomial_sum_equality_l2101_210162

/-- Given that the sum of two monomials is a monomial, prove the exponents are equal -/
theorem monomial_sum_equality (x y : ℝ) (m n : ℕ) : 
  (∃ (a : ℝ), ∀ (x y : ℝ), -x^m * y^(2+3*n) + 5 * x^(2*n-3) * y^8 = a * x^m * y^(2+3*n)) → 
  (m = 1 ∧ n = 2) :=
sorry

end NUMINAMATH_CALUDE_monomial_sum_equality_l2101_210162


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2101_210198

/-- A hyperbola with center at the origin, one focus at (-√5, 0), and a point P such that
    the midpoint of PF₁ is at (0, 2) has the equation x² - y²/4 = 1 -/
theorem hyperbola_equation (P : ℝ × ℝ) : 
  (∃ (x y : ℝ), P = (x, y) ∧ x^2 - y^2/4 = 1) ↔ 
  (∃ (x y : ℝ), P = (x, y) ∧ 
    -- P is on the hyperbola
    (x - (-Real.sqrt 5))^2 + y^2 = (x - Real.sqrt 5)^2 + y^2 ∧ 
    -- Midpoint of PF₁ is (0, 2)
    ((x + (-Real.sqrt 5))/2 = 0 ∧ (y + 0)/2 = 2)) :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_equation_l2101_210198


namespace NUMINAMATH_CALUDE_doughnuts_per_box_l2101_210157

theorem doughnuts_per_box (total_doughnuts : ℕ) (num_boxes : ℕ) 
  (h1 : total_doughnuts = 48)
  (h2 : num_boxes = 4)
  (h3 : total_doughnuts % num_boxes = 0) : 
  total_doughnuts / num_boxes = 12 := by
sorry

end NUMINAMATH_CALUDE_doughnuts_per_box_l2101_210157


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l2101_210108

-- Define set A
def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}

-- Define set B
def B : Set ℝ := {x | Real.rpow 2022 x > Real.sqrt 2022}

-- Theorem statement
theorem A_intersect_B_eq_open_interval :
  A ∩ B = Set.Ioo (1/2 : ℝ) 6 := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l2101_210108


namespace NUMINAMATH_CALUDE_largest_subset_size_150_l2101_210109

/-- A function that returns the size of the largest subset of integers from 1 to n 
    where no member is 4 times another member -/
def largest_subset_size (n : ℕ) : ℕ := 
  sorry

/-- The theorem to be proved -/
theorem largest_subset_size_150 : largest_subset_size 150 = 142 := by
  sorry

end NUMINAMATH_CALUDE_largest_subset_size_150_l2101_210109


namespace NUMINAMATH_CALUDE_no_solution_for_inequality_l2101_210120

theorem no_solution_for_inequality :
  ¬ ∃ x : ℝ, |x| + |2023 - x| < 2023 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_inequality_l2101_210120


namespace NUMINAMATH_CALUDE_unfair_coin_prob_theorem_l2101_210129

/-- An unfair coin with probabilities of heads and tails -/
structure UnfairCoin where
  pH : ℝ  -- Probability of heads
  pT : ℝ  -- Probability of tails
  sum_one : pH + pT = 1
  unfair : pH ≠ pT

/-- The probability of getting one head and one tail in two tosses -/
def prob_one_head_one_tail (c : UnfairCoin) : ℝ :=
  2 * c.pH * c.pT

/-- The probability of getting two heads and two tails in four tosses -/
def prob_two_heads_two_tails (c : UnfairCoin) : ℝ :=
  6 * c.pH * c.pH * c.pT * c.pT

/-- Theorem: For an unfair coin where the probability of getting one head and one tail
    in two tosses is 1/2, the probability of getting two heads and two tails in four tosses is 3/8 -/
theorem unfair_coin_prob_theorem (c : UnfairCoin) 
    (h : prob_one_head_one_tail c = 1/2) : 
    prob_two_heads_two_tails c = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_unfair_coin_prob_theorem_l2101_210129
