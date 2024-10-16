import Mathlib

namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3843_384327

/-- The coefficient of x^r in the expansion of (1 + ax)^n -/
def binomialCoefficient (n : ℕ) (a : ℝ) (r : ℕ) : ℝ :=
  a^r * (n.choose r)

theorem binomial_expansion_coefficient (n : ℕ) :
  binomialCoefficient n 3 2 = 54 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3843_384327


namespace NUMINAMATH_CALUDE_number_of_rattlesnakes_rattlesnakes_count_l3843_384319

/-- The number of rattlesnakes in a park with given conditions -/
theorem number_of_rattlesnakes (total_snakes : ℕ) (boa_constrictors : ℕ) : ℕ :=
  let pythons := 3 * boa_constrictors
  let rattlesnakes := total_snakes - (boa_constrictors + pythons)
  rattlesnakes

/-- Proof that the number of rattlesnakes is 40 given the conditions -/
theorem rattlesnakes_count :
  number_of_rattlesnakes 200 40 = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_of_rattlesnakes_rattlesnakes_count_l3843_384319


namespace NUMINAMATH_CALUDE_root_difference_l3843_384337

-- Define the equation
def equation (r : ℝ) : Prop :=
  (r^2 - 5*r - 20) / (r - 2) = 2*r + 7

-- Define the roots of the equation
def roots : Set ℝ :=
  {r : ℝ | equation r}

-- Theorem statement
theorem root_difference : ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ |r₁ - r₂| = 4 :=
sorry

end NUMINAMATH_CALUDE_root_difference_l3843_384337


namespace NUMINAMATH_CALUDE_sample_size_is_selected_size_l3843_384386

/-- Represents the total number of first-year high school students -/
def population_size : ℕ := 1320

/-- Represents the number of students selected for measurement -/
def selected_size : ℕ := 220

/-- Theorem stating that the sample size is equal to the number of selected students -/
theorem sample_size_is_selected_size : 
  selected_size = 220 := by sorry

end NUMINAMATH_CALUDE_sample_size_is_selected_size_l3843_384386


namespace NUMINAMATH_CALUDE_cost_prices_calculation_l3843_384353

/-- Represents the cost price of an item -/
structure CostPrice where
  value : ℝ
  positive : value > 0

/-- Represents the selling price of an item -/
structure SellingPrice where
  value : ℝ
  positive : value > 0

/-- Calculates the selling price given a cost price and a percentage change -/
def calculateSellingPrice (cp : CostPrice) (percentageChange : ℝ) : SellingPrice :=
  { value := cp.value * (1 + percentageChange),
    positive := sorry }

/-- Determines if two real numbers are approximately equal within a small tolerance -/
def approximatelyEqual (x y : ℝ) : Prop :=
  |x - y| < 0.01

theorem cost_prices_calculation
  (diningSet : CostPrice)
  (chandelier : CostPrice)
  (sofaSet : CostPrice)
  (diningSetSelling : SellingPrice)
  (chandelierSelling : SellingPrice)
  (sofaSetSelling : SellingPrice) :
  (diningSetSelling = calculateSellingPrice diningSet (-0.18)) →
  (calculateSellingPrice diningSet 0.15).value = diningSetSelling.value + 2500 →
  (chandelierSelling = calculateSellingPrice chandelier 0.20) →
  (calculateSellingPrice chandelier (-0.20)).value = chandelierSelling.value - 3000 →
  (sofaSetSelling = calculateSellingPrice sofaSet (-0.10)) →
  (calculateSellingPrice sofaSet 0.25).value = sofaSetSelling.value + 4000 →
  approximatelyEqual diningSet.value 7576 ∧
  chandelier.value = 7500 ∧
  approximatelyEqual sofaSet.value 11429 := by
  sorry

#check cost_prices_calculation

end NUMINAMATH_CALUDE_cost_prices_calculation_l3843_384353


namespace NUMINAMATH_CALUDE_sqrt_two_thirds_is_quadratic_radical_l3843_384394

/-- A number is a quadratic radical if it can be expressed as the square root of a non-negative real number. -/
def is_quadratic_radical (x : ℝ) : Prop :=
  ∃ (y : ℝ), y ≥ 0 ∧ x = Real.sqrt y

/-- The square root of 2/3 is a quadratic radical. -/
theorem sqrt_two_thirds_is_quadratic_radical :
  is_quadratic_radical (Real.sqrt (2/3)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_thirds_is_quadratic_radical_l3843_384394


namespace NUMINAMATH_CALUDE_max_cut_length_30x30_225pieces_l3843_384369

/-- Represents a square board with side length and number of pieces it's cut into -/
structure Board where
  side_length : ℕ
  num_pieces : ℕ

/-- Calculates the maximum possible total length of cuts for a given board -/
def max_cut_length (b : Board) : ℕ :=
  sorry

/-- The theorem stating the maximum cut length for a 30x30 board cut into 225 pieces -/
theorem max_cut_length_30x30_225pieces :
  let b : Board := { side_length := 30, num_pieces := 225 }
  max_cut_length b = 1065 := by
  sorry

end NUMINAMATH_CALUDE_max_cut_length_30x30_225pieces_l3843_384369


namespace NUMINAMATH_CALUDE_prove_movie_theatre_seats_l3843_384354

def movie_theatre_seats (adult_price child_price : ℕ) (num_children total_revenue : ℕ) : Prop :=
  let total_seats := num_children + (total_revenue - num_children * child_price) / adult_price
  total_seats = 250 ∧
  adult_price * (total_seats - num_children) + child_price * num_children = total_revenue

theorem prove_movie_theatre_seats :
  movie_theatre_seats 6 4 188 1124 := by
  sorry

end NUMINAMATH_CALUDE_prove_movie_theatre_seats_l3843_384354


namespace NUMINAMATH_CALUDE_track_completion_time_l3843_384324

/-- Time to complete a circular track -/
def complete_track_time (half_track_time : ℝ) : ℝ :=
  2 * half_track_time

/-- Theorem: The time to complete the circular track is 6 minutes -/
theorem track_completion_time :
  let half_track_time : ℝ := 3
  complete_track_time half_track_time = 6 := by
  sorry


end NUMINAMATH_CALUDE_track_completion_time_l3843_384324


namespace NUMINAMATH_CALUDE_scientific_notation_of_nine_billion_l3843_384357

theorem scientific_notation_of_nine_billion :
  9000000000 = 9 * (10 ^ 9) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_nine_billion_l3843_384357


namespace NUMINAMATH_CALUDE_binomial_square_constant_l3843_384315

theorem binomial_square_constant (c : ℝ) : 
  (∃ a b : ℝ, ∀ x, 9*x^2 + 30*x + c = (a*x + b)^2) → c = 25 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l3843_384315


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l3843_384379

/-- Given a hyperbola and a parabola with specific properties, 
    prove that the focal length of the hyperbola is 2√5 -/
theorem hyperbola_focal_length 
  (a b p : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hp : p > 0) 
  (h_vertex_focus : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2/a^2 - y₁^2/b^2 = 1 ∧ 
    y₂^2 = 2*p*x₂ ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16) 
  (h_asymptote_directrix : ∃ (k : ℝ), 
    (-2)^2/a^2 - (-1)^2/b^2 = k^2 ∧ 
    -2 = -p/2) : 
  2 * (a^2 + b^2).sqrt = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l3843_384379


namespace NUMINAMATH_CALUDE_mixed_oil_cost_theorem_l3843_384368

/-- Calculates the cost per litre of a mixed oil blend --/
def cost_per_litre_mixed_oil (volume_A volume_B volume_C : ℚ) 
                             (price_A price_B price_C : ℚ) : ℚ :=
  let total_cost := volume_A * price_A + volume_B * price_B + volume_C * price_C
  let total_volume := volume_A + volume_B + volume_C
  total_cost / total_volume

/-- The cost per litre of the mixed oil is approximately 54.52 --/
theorem mixed_oil_cost_theorem :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
  abs (cost_per_litre_mixed_oil 10 5 8 54 66 48 - 54.52) < ε :=
sorry

end NUMINAMATH_CALUDE_mixed_oil_cost_theorem_l3843_384368


namespace NUMINAMATH_CALUDE_factorial_difference_l3843_384314

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 8 = 3588480 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l3843_384314


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3843_384382

theorem inequality_solution_set : 
  {x : ℝ | x^2 - 2*x - 5 > 2*x} = {x : ℝ | x > 5 ∨ x < -1} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3843_384382


namespace NUMINAMATH_CALUDE_tom_seashells_l3843_384311

def seashells_remaining (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

theorem tom_seashells : seashells_remaining 5 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tom_seashells_l3843_384311


namespace NUMINAMATH_CALUDE_system_solution_l3843_384350

/-- Given a system of equations:
    0.05x + 0.07(30 + x) = 14.9
    0.03y - 5.6 = 0.07x
    Prove that x = 106.67 and y = 435.567 are the solutions. -/
theorem system_solution : ∃ (x y : ℝ), 
  (0.05 * x + 0.07 * (30 + x) = 14.9) ∧ 
  (0.03 * y - 5.6 = 0.07 * x) ∧ 
  (x = 106.67) ∧ 
  (y = 435.567) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3843_384350


namespace NUMINAMATH_CALUDE_complex_subtraction_l3843_384322

theorem complex_subtraction (z₁ z₂ : ℂ) (h1 : z₁ = 7 - 6*I) (h2 : z₂ = 4 - 7*I) :
  z₁ - z₂ = 3 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l3843_384322


namespace NUMINAMATH_CALUDE_unique_divisible_sum_l3843_384325

theorem unique_divisible_sum (p : ℕ) (h_prime : Nat.Prime p) :
  ∃! n : ℕ, (p * n) % (p + n) = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_divisible_sum_l3843_384325


namespace NUMINAMATH_CALUDE_pollywogs_disappearance_l3843_384352

/-- The number of pollywogs that mature into toads and leave the pond per day -/
def maturation_rate : ℕ := 50

/-- The number of pollywogs Melvin catches per day for the first 20 days -/
def melvin_catch_rate : ℕ := 10

/-- The number of days Melvin catches pollywogs -/
def melvin_catch_days : ℕ := 20

/-- The total number of days it took for all pollywogs to disappear -/
def total_days : ℕ := 44

/-- The initial number of pollywogs in the pond -/
def initial_pollywogs : ℕ := 2400

theorem pollywogs_disappearance :
  initial_pollywogs = 
    (maturation_rate + melvin_catch_rate) * melvin_catch_days + 
    maturation_rate * (total_days - melvin_catch_days) := by
  sorry

end NUMINAMATH_CALUDE_pollywogs_disappearance_l3843_384352


namespace NUMINAMATH_CALUDE_number42_does_not_contain_5_l3843_384348

/-- Represents a five-digit rising number -/
structure RisingNumber :=
  (d1 d2 d3 d4 d5 : Nat)
  (h1 : d1 < d2)
  (h2 : d2 < d3)
  (h3 : d3 < d4)
  (h4 : d4 < d5)
  (h5 : 1 ≤ d1 ∧ d5 ≤ 8)

/-- The list of all valid rising numbers -/
def risingNumbers : List RisingNumber := sorry

/-- The 42nd number in the sorted list of rising numbers -/
def number42 : RisingNumber := sorry

/-- Theorem stating that the 42nd rising number does not contain 5 -/
theorem number42_does_not_contain_5 : 
  number42.d1 ≠ 5 ∧ number42.d2 ≠ 5 ∧ number42.d3 ≠ 5 ∧ number42.d4 ≠ 5 ∧ number42.d5 ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_number42_does_not_contain_5_l3843_384348


namespace NUMINAMATH_CALUDE_g_of_seven_l3843_384340

/-- Given a function g(x) = (2x + 3) / (4x - 5), prove that g(7) = 17/23 -/
theorem g_of_seven (g : ℝ → ℝ) (h : ∀ x, g x = (2 * x + 3) / (4 * x - 5)) : 
  g 7 = 17 / 23 := by
  sorry

end NUMINAMATH_CALUDE_g_of_seven_l3843_384340


namespace NUMINAMATH_CALUDE_stream_speed_l3843_384305

/-- Proves that the speed of a stream is 8 kmph given the conditions of the boat's travel -/
theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (upstream_distance : ℝ) 
  (h1 : boat_speed = 24)
  (h2 : downstream_distance = 64)
  (h3 : upstream_distance = 32)
  (h4 : downstream_distance / (boat_speed + x) = upstream_distance / (boat_speed - x)) :
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l3843_384305


namespace NUMINAMATH_CALUDE_profit_calculation_l3843_384328

/-- Represents the profit made from commercial farming -/
def profit_from_farming 
  (total_land : ℝ)           -- Total land area in hectares
  (num_sons : ℕ)             -- Number of sons
  (profit_per_son : ℝ)       -- Annual profit per son
  (land_unit : ℝ)            -- Land unit for profit calculation in m^2
  : ℝ :=
  -- The function body is left empty as we only need the statement
  sorry

/-- Theorem stating the profit from farming under given conditions -/
theorem profit_calculation :
  let total_land : ℝ := 3                    -- 3 hectares
  let num_sons : ℕ := 8                      -- 8 sons
  let profit_per_son : ℝ := 10000            -- $10,000 per year per son
  let land_unit : ℝ := 750                   -- 750 m^2 unit
  let hectare_to_sqm : ℝ := 10000            -- 1 hectare = 10,000 m^2
  profit_from_farming total_land num_sons profit_per_son land_unit = 500 :=
by
  sorry  -- The proof is omitted as per instructions

end NUMINAMATH_CALUDE_profit_calculation_l3843_384328


namespace NUMINAMATH_CALUDE_bedroom_set_final_price_l3843_384362

/-- Calculates the final price of a bedroom set after discounts and gift card application --/
def final_price (initial_price gift_card first_discount second_discount : ℚ) : ℚ :=
  let price_after_first_discount := initial_price * (1 - first_discount)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount)
  price_after_second_discount - gift_card

/-- Theorem: The final price of the bedroom set is $1330 --/
theorem bedroom_set_final_price :
  final_price 2000 200 0.15 0.10 = 1330 := by
  sorry

end NUMINAMATH_CALUDE_bedroom_set_final_price_l3843_384362


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l3843_384377

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflection of a point across the x-axis -/
def reflect_x (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

theorem reflection_across_x_axis : 
  let P : Point2D := { x := -2, y := 5 }
  reflect_x P = { x := -2, y := -5 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l3843_384377


namespace NUMINAMATH_CALUDE_color_tv_price_l3843_384381

/-- The original price of a color TV -/
def original_price : ℝ := 1200

/-- The price after 40% increase -/
def increased_price (x : ℝ) : ℝ := x * (1 + 0.4)

/-- The final price after 20% discount -/
def final_price (x : ℝ) : ℝ := increased_price x * 0.8

theorem color_tv_price :
  final_price original_price - original_price = 144 := by sorry

end NUMINAMATH_CALUDE_color_tv_price_l3843_384381


namespace NUMINAMATH_CALUDE_min_sum_squares_l3843_384336

theorem min_sum_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 2 * a + 3 * b + 5 * c = 100) : 
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → 2 * x + 3 * y + 5 * z = 100 → 
  a^2 + b^2 + c^2 ≤ x^2 + y^2 + z^2 ∧ 
  a^2 + b^2 + c^2 = 5000 / 19 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3843_384336


namespace NUMINAMATH_CALUDE_sandy_spending_percentage_l3843_384396

def initial_amount : ℝ := 300
def remaining_amount : ℝ := 210

theorem sandy_spending_percentage :
  (initial_amount - remaining_amount) / initial_amount * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sandy_spending_percentage_l3843_384396


namespace NUMINAMATH_CALUDE_race_conditions_satisfied_l3843_384329

/-- The speed of Xiao Ying in meters per second -/
def xiao_ying_speed : ℝ := 4

/-- The speed of Xiao Liang in meters per second -/
def xiao_liang_speed : ℝ := 6

/-- Theorem stating that the given speeds satisfy the race conditions -/
theorem race_conditions_satisfied : 
  (5 * xiao_ying_speed + 10 = 5 * xiao_liang_speed) ∧ 
  (6 * xiao_ying_speed = 4 * xiao_liang_speed) := by
  sorry

end NUMINAMATH_CALUDE_race_conditions_satisfied_l3843_384329


namespace NUMINAMATH_CALUDE_largest_x_floor_ratio_l3843_384384

theorem largest_x_floor_ratio : 
  ∀ x : ℝ, (↑(Int.floor x) / x = 8 / 9) → x ≤ 63 / 8 :=
by sorry

end NUMINAMATH_CALUDE_largest_x_floor_ratio_l3843_384384


namespace NUMINAMATH_CALUDE_product_digit_sum_l3843_384345

/-- The number of digits in the second factor of the product (9)(999...9) -/
def k : ℕ := sorry

/-- The sum of digits in the resulting integer -/
def digit_sum : ℕ := 1009

/-- The resulting integer from the product (9)(999...9) -/
def result : ℕ := 10^k - 1

theorem product_digit_sum : 
  (∀ n : ℕ, n ≤ k → (result / 10^n) % 10 = 9) ∧ 
  (result % 10 = 9) ∧
  (digit_sum = 9 * k) ∧
  (k = 112) :=
sorry

end NUMINAMATH_CALUDE_product_digit_sum_l3843_384345


namespace NUMINAMATH_CALUDE_fiftieth_term_of_sequence_l3843_384301

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem fiftieth_term_of_sequence : arithmetic_sequence 2 4 50 = 198 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_of_sequence_l3843_384301


namespace NUMINAMATH_CALUDE_some_mystical_enchanted_l3843_384393

-- Define the sets
variable (U : Type) -- Universe set
variable (D : Set U) -- Set of dragons
variable (M : Set U) -- Set of mystical creatures
variable (E : Set U) -- Set of enchanted beings

-- Define the conditions
axiom all_dragons_mystical : D ⊆ M
axiom some_enchanted_dragons : ∃ x, x ∈ E ∩ D

-- State the theorem to be proved
theorem some_mystical_enchanted : ∃ x, x ∈ M ∩ E := by sorry

end NUMINAMATH_CALUDE_some_mystical_enchanted_l3843_384393


namespace NUMINAMATH_CALUDE_complex_simplification_l3843_384304

theorem complex_simplification (i : ℂ) (h : i^2 = -1) :
  6 * (4 - 2*i) + 2*i * (7 - 3*i) = 30 + 2*i := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l3843_384304


namespace NUMINAMATH_CALUDE_smallest_geometric_sequence_number_l3843_384391

/-- A function that checks if a three-digit number's digits form a geometric sequence -/
def is_geometric_sequence (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ b * b = a * c

/-- A function that checks if a three-digit number has distinct digits -/
def has_distinct_digits (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_geometric_sequence_number :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 →
    is_geometric_sequence n ∧ has_distinct_digits n →
    124 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_geometric_sequence_number_l3843_384391


namespace NUMINAMATH_CALUDE_total_games_won_l3843_384342

-- Define the number of games won by Betsy
def betsy_games : ℕ := 5

-- Define Helen's games in terms of Betsy's
def helen_games : ℕ := 2 * betsy_games

-- Define Susan's games in terms of Betsy's
def susan_games : ℕ := 3 * betsy_games

-- Theorem to prove the total number of games won
theorem total_games_won : betsy_games + helen_games + susan_games = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_games_won_l3843_384342


namespace NUMINAMATH_CALUDE_weight_replacement_l3843_384355

theorem weight_replacement (initial_count : ℕ) (weight_increase : ℚ) (new_weight : ℚ) :
  initial_count = 8 →
  weight_increase = 5/2 →
  new_weight = 40 →
  ∃ (old_weight : ℚ),
    old_weight = new_weight - (initial_count * weight_increase) ∧
    old_weight = 20 := by
  sorry

end NUMINAMATH_CALUDE_weight_replacement_l3843_384355


namespace NUMINAMATH_CALUDE_largest_fraction_equal_digit_sums_l3843_384359

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if a number is four-digit -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- The largest fraction with equal digit sums -/
theorem largest_fraction_equal_digit_sums :
  ∀ m n : ℕ, 
    is_four_digit m → 
    is_four_digit n → 
    digit_sum m = digit_sum n → 
    (m : ℚ) / n ≤ 9900 / 1089 := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_equal_digit_sums_l3843_384359


namespace NUMINAMATH_CALUDE_min_socks_for_different_pairs_l3843_384326

/-- Represents a sock with a size and a color -/
structure Sock :=
  (size : Nat)
  (color : Nat)

/-- Represents the total number of socks -/
def totalSocks : Nat := 8

/-- Represents the number of different sizes -/
def numSizes : Nat := 2

/-- Represents the number of different colors -/
def numColors : Nat := 2

/-- Theorem stating the minimum number of socks needed to guarantee two pairs of different sizes and colors -/
theorem min_socks_for_different_pairs :
  ∀ (socks : Finset Sock),
    (Finset.card socks = totalSocks) →
    (∀ s ∈ socks, s.size < numSizes ∧ s.color < numColors) →
    (∃ (n : Nat),
      ∀ (subset : Finset Sock),
        (Finset.card subset = n) →
        (subset ⊆ socks) →
        (∃ (s1 s2 s3 s4 : Sock),
          s1 ∈ subset ∧ s2 ∈ subset ∧ s3 ∈ subset ∧ s4 ∈ subset ∧
          s1.size ≠ s2.size ∧ s1.color ≠ s2.color ∧
          s3.size ≠ s4.size ∧ s3.color ≠ s4.color)) →
    n = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_socks_for_different_pairs_l3843_384326


namespace NUMINAMATH_CALUDE_distance_specific_point_to_line_l3843_384372

/-- The distance from a point to a line in 3D space --/
def distance_point_to_line (point : ℝ × ℝ × ℝ) (line_point1 line_point2 : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- Theorem: The distance from (2, 3, -1) to the line passing through (3, -1, 4) and (5, 0, 1) is √3667/14 --/
theorem distance_specific_point_to_line :
  let point : ℝ × ℝ × ℝ := (2, 3, -1)
  let line_point1 : ℝ × ℝ × ℝ := (3, -1, 4)
  let line_point2 : ℝ × ℝ × ℝ := (5, 0, 1)
  distance_point_to_line point line_point1 line_point2 = Real.sqrt 3667 / 14 := by
  sorry

end NUMINAMATH_CALUDE_distance_specific_point_to_line_l3843_384372


namespace NUMINAMATH_CALUDE_right_triangle_area_l3843_384309

theorem right_triangle_area (a b c d : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 → 
  a + b = 24 → 
  c = 24 → 
  d = 24 → 
  a^2 + b^2 = c^2 → 
  (1/2) * a * d = 216 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3843_384309


namespace NUMINAMATH_CALUDE_third_quadrant_trig_expression_l3843_384366

theorem third_quadrant_trig_expression (α : Real) : 
  (α > π ∧ α < 3*π/2) →  -- α is in the third quadrant
  (2 * Real.sin α) / Real.sqrt (1 - Real.cos α ^ 2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_third_quadrant_trig_expression_l3843_384366


namespace NUMINAMATH_CALUDE_ice_water_masses_l3843_384378

/-- Given a cylindrical vessel with ice and water, calculate the initial masses. -/
theorem ice_water_masses (S : ℝ) (ρw ρi : ℝ) (Δh hf : ℝ) 
  (hS : S = 15) 
  (hρw : ρw = 1) 
  (hρi : ρi = 0.92) 
  (hΔh : Δh = 5) 
  (hhf : hf = 115) :
  ∃ (mi mw : ℝ), 
    mi = 862.5 ∧ 
    mw = 1050 ∧ 
    mi / ρi - mi / ρw = S * Δh ∧ 
    mw + mi = ρw * S * hf := by
  sorry

#check ice_water_masses

end NUMINAMATH_CALUDE_ice_water_masses_l3843_384378


namespace NUMINAMATH_CALUDE_price_decrease_l3843_384323

/-- Given a 24% decrease in price resulting in a cost of Rs. 684, prove that the original price was Rs. 900. -/
theorem price_decrease (original_price : ℝ) : 
  (original_price * (1 - 0.24) = 684) → original_price = 900 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_l3843_384323


namespace NUMINAMATH_CALUDE_cosine_sum_simplification_l3843_384398

theorem cosine_sum_simplification :
  Real.cos ((2 * Real.pi) / 17) + Real.cos ((6 * Real.pi) / 17) + Real.cos ((8 * Real.pi) / 17) = (Real.sqrt 13 - 1) / 4 :=
by sorry

end NUMINAMATH_CALUDE_cosine_sum_simplification_l3843_384398


namespace NUMINAMATH_CALUDE_dads_dimes_count_l3843_384385

/-- The number of dimes Tom's dad gave him -/
def dimes_from_dad (initial_dimes final_dimes : ℕ) : ℕ :=
  final_dimes - initial_dimes

/-- Proof that Tom's dad gave him 33 dimes -/
theorem dads_dimes_count : dimes_from_dad 15 48 = 33 := by
  sorry

end NUMINAMATH_CALUDE_dads_dimes_count_l3843_384385


namespace NUMINAMATH_CALUDE_unique_sum_of_eight_only_36_37_l3843_384308

/-- A function that returns true if there exists exactly one set of 8 different positive integers that sum to n -/
def unique_sum_of_eight (n : ℕ) : Prop :=
  ∃! (s : Finset ℕ), s.card = 8 ∧ (∀ x ∈ s, x > 0) ∧ s.sum id = n

/-- Theorem stating that 36 and 37 are the only natural numbers with a unique sum of eight different positive integers -/
theorem unique_sum_of_eight_only_36_37 :
  ∀ n : ℕ, unique_sum_of_eight n ↔ n = 36 ∨ n = 37 := by
  sorry

#check unique_sum_of_eight_only_36_37

end NUMINAMATH_CALUDE_unique_sum_of_eight_only_36_37_l3843_384308


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3843_384331

theorem trigonometric_identity (a b x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (Real.sin x)^4 / a^2 + (Real.cos x)^4 / b^2 = 1 / (a^2 + b^2)) :
  (Real.sin x)^100 / a^100 + (Real.cos x)^100 / b^100 = 2 / (a^2 + b^2)^100 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3843_384331


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l3843_384389

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℕ)

-- Define the condition for integer side lengths and no two sides being equal
def validTriangle (t : Triangle) : Prop :=
  t.a ≠ t.b ∧ t.b ≠ t.c ∧ t.c ≠ t.a

-- Define the semiperimeter
def semiperimeter (t : Triangle) : ℚ :=
  (t.a + t.b + t.c) / 2

-- Define the inradius
def inradius (t : Triangle) : ℚ :=
  let s := semiperimeter t
  (s - t.a) * (s - t.b) * (s - t.c) / s

-- Define the excircle radii
def excircleRadius (t : Triangle) (side : ℕ) : ℚ :=
  let s := semiperimeter t
  let r := inradius t
  r * s / (s - side)

-- Define the tangency conditions
def tangencyConditions (t : Triangle) : Prop :=
  let r := inradius t
  let rA := excircleRadius t t.a
  let rB := excircleRadius t t.b
  let rC := excircleRadius t t.c
  r + rA = rB ∧ r + rA = rC

-- Theorem statement
theorem min_perimeter_triangle (t : Triangle) :
  validTriangle t → tangencyConditions t → t.a + t.b + t.c ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l3843_384389


namespace NUMINAMATH_CALUDE_f_eight_equals_zero_l3843_384387

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_eight_equals_zero
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_period : ∀ x, f (x + 2) = -f x) :
  f 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_eight_equals_zero_l3843_384387


namespace NUMINAMATH_CALUDE_spiral_stripe_length_l3843_384312

/-- The length of a spiral stripe on a right circular cylinder -/
theorem spiral_stripe_length (base_circumference height : ℝ) (h1 : base_circumference = 18) (h2 : height = 8) :
  Real.sqrt (height^2 + (2 * base_circumference)^2) = Real.sqrt 1360 := by
  sorry

end NUMINAMATH_CALUDE_spiral_stripe_length_l3843_384312


namespace NUMINAMATH_CALUDE_merchant_profit_l3843_384351

theorem merchant_profit (C S : ℝ) (h : 22 * C = 16 * S) : 
  (S - C) / C * 100 = 37.5 := by sorry

end NUMINAMATH_CALUDE_merchant_profit_l3843_384351


namespace NUMINAMATH_CALUDE_mothers_carrots_count_l3843_384333

/-- The number of carrots Olivia picked -/
def olivias_carrots : ℕ := 20

/-- The total number of good carrots -/
def good_carrots : ℕ := 19

/-- The total number of bad carrots -/
def bad_carrots : ℕ := 15

/-- The number of carrots Olivia's mother picked -/
def mothers_carrots : ℕ := (good_carrots + bad_carrots) - olivias_carrots

theorem mothers_carrots_count : mothers_carrots = 14 := by
  sorry

end NUMINAMATH_CALUDE_mothers_carrots_count_l3843_384333


namespace NUMINAMATH_CALUDE_johns_total_income_this_year_l3843_384334

/-- Calculates the total income (salary + bonus) for the current year given the previous year's salary and bonus, and the current year's salary. -/
def totalIncomeCurrentYear (prevSalary prevBonus currSalary : ℕ) : ℕ :=
  let bonusRate := prevBonus / prevSalary
  let currBonus := currSalary * bonusRate
  currSalary + currBonus

/-- Theorem stating that John's total income this year is $220,000 -/
theorem johns_total_income_this_year :
  totalIncomeCurrentYear 100000 10000 200000 = 220000 := by
  sorry

end NUMINAMATH_CALUDE_johns_total_income_this_year_l3843_384334


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3843_384361

/-- Given a hyperbola with equation y²/16 - x²/m = 1 and eccentricity e = 2, prove that m = 48 -/
theorem hyperbola_eccentricity (m : ℝ) (e : ℝ) :
  (∀ x y : ℝ, y^2 / 16 - x^2 / m = 1) →
  e = 2 →
  m = 48 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3843_384361


namespace NUMINAMATH_CALUDE_cell_phone_company_customers_l3843_384344

theorem cell_phone_company_customers (us_customers other_customers : ℕ) 
  (h1 : us_customers = 723)
  (h2 : other_customers = 6699) :
  us_customers + other_customers = 7422 := by
  sorry

end NUMINAMATH_CALUDE_cell_phone_company_customers_l3843_384344


namespace NUMINAMATH_CALUDE_randy_initial_biscuits_l3843_384339

/-- The number of biscuits Randy's father gave him -/
def father_gift : ℕ := 13

/-- The number of biscuits Randy's mother gave him -/
def mother_gift : ℕ := 15

/-- The number of biscuits Randy's brother ate -/
def brother_ate : ℕ := 20

/-- The number of biscuits Randy is left with -/
def remaining_biscuits : ℕ := 40

/-- Randy's initial number of biscuits -/
def initial_biscuits : ℕ := 32

theorem randy_initial_biscuits :
  initial_biscuits + father_gift + mother_gift - brother_ate = remaining_biscuits :=
by sorry

end NUMINAMATH_CALUDE_randy_initial_biscuits_l3843_384339


namespace NUMINAMATH_CALUDE_simplify_expression_l3843_384347

theorem simplify_expression (a b x y : ℝ) (h : b*x + a*y ≠ 0) :
  (b*x*(a^2*x^2 + 2*a^2*y^2 + b^2*y^2)) / (b*x + a*y) +
  (a*y*(a^2*x^2 + 2*b^2*x^2 + b^2*y^2)) / (b*x + a*y) =
  (a*x + b*y)^2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3843_384347


namespace NUMINAMATH_CALUDE_cinema_visitors_l3843_384320

theorem cinema_visitors (female_visitors : ℕ) (female_office_workers : ℕ) 
  (male_excess : ℕ) (male_non_workers : ℕ) 
  (h1 : female_visitors = 1518)
  (h2 : female_office_workers = 536)
  (h3 : male_excess = 525)
  (h4 : male_non_workers = 1257) :
  female_office_workers + (female_visitors + male_excess - male_non_workers) = 1322 := by
  sorry

end NUMINAMATH_CALUDE_cinema_visitors_l3843_384320


namespace NUMINAMATH_CALUDE_diegos_stamp_collection_cost_l3843_384380

def brazil_stamps : ℕ := 6 + 9
def peru_stamps : ℕ := 8 + 5
def colombia_stamps : ℕ := 7 + 6

def brazil_cost : ℚ := 0.07
def peru_cost : ℚ := 0.05
def colombia_cost : ℚ := 0.07

def total_cost : ℚ := 
  brazil_stamps * brazil_cost + 
  peru_stamps * peru_cost + 
  colombia_stamps * colombia_cost

theorem diegos_stamp_collection_cost : total_cost = 2.61 := by
  sorry

end NUMINAMATH_CALUDE_diegos_stamp_collection_cost_l3843_384380


namespace NUMINAMATH_CALUDE_empty_solution_set_non_empty_solution_set_l3843_384317

-- Define the inequality
def inequality (x a : ℝ) : Prop := |x - 4| + |3 - x| < a

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | inequality x a}

-- Theorem for the empty solution set case
theorem empty_solution_set (a : ℝ) :
  solution_set a = ∅ ↔ a ≤ 1 :=
sorry

-- Theorem for the non-empty solution set case
theorem non_empty_solution_set (a : ℝ) :
  solution_set a ≠ ∅ ↔ a > 1 :=
sorry

end NUMINAMATH_CALUDE_empty_solution_set_non_empty_solution_set_l3843_384317


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l3843_384310

theorem reciprocal_inequality (a b : ℝ) (ha : a < 0) (hb : b > 0) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l3843_384310


namespace NUMINAMATH_CALUDE_typing_contest_orders_l3843_384363

/-- The number of different possible orders for a given number of participants to finish a contest without ties. -/
def numberOfOrders (n : ℕ) : ℕ := Nat.factorial n

/-- The number of participants in the typing contest. -/
def numberOfParticipants : ℕ := 4

theorem typing_contest_orders :
  numberOfOrders numberOfParticipants = 24 := by
  sorry

end NUMINAMATH_CALUDE_typing_contest_orders_l3843_384363


namespace NUMINAMATH_CALUDE_toms_dog_age_l3843_384390

/-- Given the ages of Tom's pets, prove the age of his dog. -/
theorem toms_dog_age (cat_age : ℕ) (rabbit_age : ℕ) (dog_age : ℕ)
  (h1 : cat_age = 8)
  (h2 : rabbit_age = cat_age / 2)
  (h3 : dog_age = rabbit_age * 3) :
  dog_age = 12 :=
by sorry

end NUMINAMATH_CALUDE_toms_dog_age_l3843_384390


namespace NUMINAMATH_CALUDE_probability_less_than_4_is_7_9_l3843_384356

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  sideLength : ℝ

/-- The probability that a randomly chosen point in the square satisfies x + y < 4 --/
def probabilityLessThan4 (s : Square) : ℝ :=
  sorry

/-- Our specific square with vertices (0,0), (0,3), (3,3), and (3,0) --/
def specificSquare : Square :=
  { bottomLeft := (0, 0), sideLength := 3 }

theorem probability_less_than_4_is_7_9 :
  probabilityLessThan4 specificSquare = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_probability_less_than_4_is_7_9_l3843_384356


namespace NUMINAMATH_CALUDE_bakery_rolls_combinations_l3843_384343

theorem bakery_rolls_combinations : 
  let n : ℕ := 4  -- number of remaining rolls to distribute
  let k : ℕ := 4  -- number of kinds of rolls
  Nat.choose (n + k - 1) (k - 1) = 35 := by
sorry

end NUMINAMATH_CALUDE_bakery_rolls_combinations_l3843_384343


namespace NUMINAMATH_CALUDE_malt_shop_syrup_usage_l3843_384332

/-- Calculates the total syrup used in a malt shop given specific conditions -/
theorem malt_shop_syrup_usage
  (syrup_per_shake : ℝ)
  (syrup_per_cone : ℝ)
  (syrup_per_sundae : ℝ)
  (extra_syrup : ℝ)
  (extra_syrup_percentage : ℝ)
  (num_shakes : ℕ)
  (num_cones : ℕ)
  (num_sundaes : ℕ)
  (h1 : syrup_per_shake = 5.5)
  (h2 : syrup_per_cone = 8)
  (h3 : syrup_per_sundae = 4.2)
  (h4 : extra_syrup = 0.3)
  (h5 : extra_syrup_percentage = 0.1)
  (h6 : num_shakes = 5)
  (h7 : num_cones = 4)
  (h8 : num_sundaes = 3) :
  ∃ total_syrup : ℝ,
    total_syrup = num_shakes * syrup_per_shake +
                  num_cones * syrup_per_cone +
                  num_sundaes * syrup_per_sundae +
                  (↑(round ((num_shakes + num_cones) * extra_syrup_percentage)) * extra_syrup) ∧
    total_syrup = 72.4 := by
  sorry

end NUMINAMATH_CALUDE_malt_shop_syrup_usage_l3843_384332


namespace NUMINAMATH_CALUDE_product_sum_in_base_l3843_384307

/-- Converts a number from base b to base 10 -/
def to_base_10 (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base b -/
def from_base_10 (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Checks if a number is expressed in a given base -/
def is_in_base (n : ℕ) (b : ℕ) : Prop := sorry

theorem product_sum_in_base (b : ℕ) :
  (b > 1) →
  (is_in_base 13 b) →
  (is_in_base 14 b) →
  (is_in_base 17 b) →
  (is_in_base 5167 b) →
  (to_base_10 13 b * to_base_10 14 b * to_base_10 17 b = to_base_10 5167 b) →
  (from_base_10 (to_base_10 13 b + to_base_10 14 b + to_base_10 17 b) 7 = 50) :=
by sorry

end NUMINAMATH_CALUDE_product_sum_in_base_l3843_384307


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_value_l3843_384364

theorem unique_solution_implies_a_value (a : ℝ) :
  (∃! x : ℝ, |x^2 + 2*a*x + 3*a| ≤ 2) → (a = 1 ∨ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_value_l3843_384364


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3843_384302

theorem tan_alpha_value (α : Real) (h1 : π/2 < α ∧ α < π) 
  (h2 : Real.sin (α + π/4) = Real.sqrt 2 / 10) : Real.tan α = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3843_384302


namespace NUMINAMATH_CALUDE_animals_per_aquarium_l3843_384318

/-- Given that Tyler has 56 saltwater aquariums and 2184 saltwater animals,
    prove that there are 39 animals in each saltwater aquarium. -/
theorem animals_per_aquarium (saltwater_aquariums : ℕ) (saltwater_animals : ℕ)
    (h1 : saltwater_aquariums = 56)
    (h2 : saltwater_animals = 2184) :
    saltwater_animals / saltwater_aquariums = 39 := by
  sorry

end NUMINAMATH_CALUDE_animals_per_aquarium_l3843_384318


namespace NUMINAMATH_CALUDE_solve_x_equation_l3843_384346

theorem solve_x_equation (x y : ℝ) (hx : x ≠ 0) (h1 : x / 3 = y^2 + 1) (h2 : x / 5 = 5 * y) :
  x = (625 + 25 * Real.sqrt 589) / 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_x_equation_l3843_384346


namespace NUMINAMATH_CALUDE_roots_product_equation_l3843_384303

theorem roots_product_equation (p q : ℝ) (α β γ δ : ℂ) 
  (h1 : α^2 + p*α + 4 = 0) 
  (h2 : β^2 + p*β + 4 = 0)
  (h3 : γ^2 + q*γ + 4 = 0)
  (h4 : δ^2 + q*δ + 4 = 0) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = -4 * (p^2 - q^2) := by
  sorry

end NUMINAMATH_CALUDE_roots_product_equation_l3843_384303


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_relation_l3843_384392

/-- Proves that for a cube with volume 3x cubic units and surface area x square units, x = 5832 -/
theorem cube_volume_surface_area_relation : 
  ∃ (x : ℝ), (∃ (s : ℝ), s > 0 ∧ s^3 = 3*x ∧ 6*s^2 = x) → x = 5832 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_relation_l3843_384392


namespace NUMINAMATH_CALUDE_parallelepiped_net_removal_l3843_384371

/-- Represents a parallelepiped with integer dimensions -/
structure Parallelepiped where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a net of a parallelepiped -/
structure Net where
  squares : ℕ

/-- Represents the number of possible positions to remove a square from a net -/
def possible_removals (n : Net) : ℕ := sorry

theorem parallelepiped_net_removal 
  (p : Parallelepiped) 
  (n : Net) :
  p.length = 2 ∧ p.width = 1 ∧ p.height = 1 →
  n.squares = 10 →
  possible_removals { squares := n.squares - 1 } = 5 :=
sorry

end NUMINAMATH_CALUDE_parallelepiped_net_removal_l3843_384371


namespace NUMINAMATH_CALUDE_angle_expression_equality_l3843_384341

theorem angle_expression_equality (θ : Real) 
  (h1 : 0 < θ ∧ θ < π) 
  (h2 : Real.sin θ * Real.cos θ = -1/8) : 
  Real.sin (2*π + θ) - Real.sin (π/2 - θ) = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_expression_equality_l3843_384341


namespace NUMINAMATH_CALUDE_third_file_size_l3843_384313

theorem third_file_size 
  (internet_speed : ℝ) 
  (download_time : ℝ) 
  (file1_size : ℝ) 
  (file2_size : ℝ) 
  (h1 : internet_speed = 2) 
  (h2 : download_time = 2 * 60) 
  (h3 : file1_size = 80) 
  (h4 : file2_size = 90) : 
  ∃ (file3_size : ℝ), 
    file3_size = internet_speed * download_time - (file1_size + file2_size) ∧ 
    file3_size = 70 := by
  sorry

end NUMINAMATH_CALUDE_third_file_size_l3843_384313


namespace NUMINAMATH_CALUDE_battle_station_staffing_l3843_384335

/-- The number of job openings --/
def num_openings : ℕ := 6

/-- The total number of resumes received --/
def total_resumes : ℕ := 36

/-- The number of suitable candidates after removing one-third --/
def suitable_candidates : ℕ := total_resumes - (total_resumes / 3)

/-- The number of ways to staff the battle station --/
def staffing_ways : ℕ := 255024240

theorem battle_station_staffing :
  (suitable_candidates.factorial) / ((suitable_candidates - num_openings).factorial) = staffing_ways := by
  sorry

end NUMINAMATH_CALUDE_battle_station_staffing_l3843_384335


namespace NUMINAMATH_CALUDE_polynomial_value_at_n_plus_one_l3843_384358

theorem polynomial_value_at_n_plus_one (n : ℕ) (p : ℝ → ℝ) :
  (∀ k : ℕ, k ≤ n → p k = k / (k + 1)) →
  p (n + 1) = if n % 2 = 1 then 1 else n / (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_n_plus_one_l3843_384358


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3843_384360

theorem trigonometric_identity (x y z a : ℝ) 
  (h1 : (Real.cos x + Real.cos y + Real.cos z) / Real.cos (x + y + z) = a)
  (h2 : (Real.sin x + Real.sin y + Real.sin z) / Real.sin (x + y + z) = a) :
  Real.cos (x + y) + Real.cos (y + z) + Real.cos (z + x) = a :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3843_384360


namespace NUMINAMATH_CALUDE_no_such_function_l3843_384373

theorem no_such_function : ¬∃ f : ℝ → ℝ, f 0 > 0 ∧ ∀ x y : ℝ, f (x + y) ≥ f x + y * f (f x) := by
  sorry

end NUMINAMATH_CALUDE_no_such_function_l3843_384373


namespace NUMINAMATH_CALUDE_total_trolls_l3843_384365

/-- The number of trolls in different locations --/
structure TrollCounts where
  forest : ℕ
  bridge : ℕ
  plains : ℕ
  mountain : ℕ

/-- The conditions of the troll counting problem --/
def troll_conditions (t : TrollCounts) : Prop :=
  t.forest = 8 ∧
  t.forest = 2 * t.bridge - 4 ∧
  t.plains = t.bridge / 2 ∧
  t.mountain = t.plains + 3 ∧
  t.forest - t.mountain = 2 * t.bridge

/-- The theorem stating that given the conditions, the total number of trolls is 23 --/
theorem total_trolls (t : TrollCounts) (h : troll_conditions t) : 
  t.forest + t.bridge + t.plains + t.mountain = 23 := by
  sorry


end NUMINAMATH_CALUDE_total_trolls_l3843_384365


namespace NUMINAMATH_CALUDE_regression_difference_is_residual_sum_of_squares_l3843_384399

/-- In regression analysis, the term representing the difference between a data point
    and its corresponding position on the regression line -/
def regression_difference_term : String := "residual sum of squares"

/-- The residual sum of squares represents the difference between data points
    and their corresponding positions on the regression line -/
axiom residual_sum_of_squares_def :
  regression_difference_term = "residual sum of squares"

theorem regression_difference_is_residual_sum_of_squares :
  regression_difference_term = "residual sum of squares" := by
  sorry

end NUMINAMATH_CALUDE_regression_difference_is_residual_sum_of_squares_l3843_384399


namespace NUMINAMATH_CALUDE_power_boat_travel_time_l3843_384316

/-- Represents the scenario of a power boat and raft traveling on a river -/
structure RiverTravel where
  boatSpeed : ℝ  -- Speed of the power boat relative to the river
  riverSpeed : ℝ  -- Speed of the river current
  totalTime : ℝ  -- Total time until the boat meets the raft after returning
  travelTime : ℝ  -- Time taken by the boat to travel from A to B

/-- The conditions of the river travel scenario -/
def riverTravelConditions (rt : RiverTravel) : Prop :=
  rt.riverSpeed = rt.boatSpeed / 2 ∧
  rt.totalTime = 12 ∧
  (rt.boatSpeed + rt.riverSpeed) * rt.travelTime + 
    (rt.boatSpeed - rt.riverSpeed) * (rt.totalTime - rt.travelTime) = 
    rt.riverSpeed * rt.totalTime

/-- The theorem stating that under the given conditions, 
    the travel time from A to B is 6 hours -/
theorem power_boat_travel_time 
  (rt : RiverTravel) 
  (h : riverTravelConditions rt) : 
  rt.travelTime = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_boat_travel_time_l3843_384316


namespace NUMINAMATH_CALUDE_fifth_day_income_correct_l3843_384321

/-- Calculates the income for the fifth day given the income for four days and the average income for five days. -/
def fifth_day_income (day1 day2 day3 day4 average : ℝ) : ℝ :=
  5 * average - (day1 + day2 + day3 + day4)

/-- Theorem stating that the calculated fifth day income is correct. -/
theorem fifth_day_income_correct (day1 day2 day3 day4 day5 average : ℝ) 
  (h_average : average = (day1 + day2 + day3 + day4 + day5) / 5) :
  fifth_day_income day1 day2 day3 day4 average = day5 := by
  sorry

#eval fifth_day_income 250 400 750 400 460

end NUMINAMATH_CALUDE_fifth_day_income_correct_l3843_384321


namespace NUMINAMATH_CALUDE_substitution_result_l3843_384397

theorem substitution_result (x y : ℝ) :
  (y = x - 1) ∧ (x - 2*y = 7) → (x - 2*x + 2 = 7) := by sorry

end NUMINAMATH_CALUDE_substitution_result_l3843_384397


namespace NUMINAMATH_CALUDE_opposite_numbers_cube_root_l3843_384388

theorem opposite_numbers_cube_root (x y : ℝ) : 
  y = -x → 3 * x - 4 * y = 7 → (x * y) ^ (1/3 : ℝ) = -1 := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_cube_root_l3843_384388


namespace NUMINAMATH_CALUDE_camping_site_campers_l3843_384374

theorem camping_site_campers (total : ℕ) (last_week : ℕ) : 
  total = 150 → last_week = 80 → ∃ (three_weeks_ago two_weeks_ago : ℕ), 
    two_weeks_ago = three_weeks_ago + 10 ∧ 
    total = three_weeks_ago + two_weeks_ago + last_week ∧
    two_weeks_ago = 40 := by sorry

end NUMINAMATH_CALUDE_camping_site_campers_l3843_384374


namespace NUMINAMATH_CALUDE_daniel_goats_count_l3843_384338

/-- The number of goats Daniel has -/
def num_goats : ℕ := 1

/-- The number of horses Daniel has -/
def num_horses : ℕ := 2

/-- The number of dogs Daniel has -/
def num_dogs : ℕ := 5

/-- The number of cats Daniel has -/
def num_cats : ℕ := 7

/-- The number of turtles Daniel has -/
def num_turtles : ℕ := 3

/-- The total number of legs of all animals -/
def total_legs : ℕ := 72

/-- The number of legs each animal has -/
def legs_per_animal : ℕ := 4

theorem daniel_goats_count :
  num_goats * legs_per_animal + 
  num_horses * legs_per_animal + 
  num_dogs * legs_per_animal + 
  num_cats * legs_per_animal + 
  num_turtles * legs_per_animal = total_legs :=
by sorry

end NUMINAMATH_CALUDE_daniel_goats_count_l3843_384338


namespace NUMINAMATH_CALUDE_quadratic_roots_distance_l3843_384375

/-- Given a quadratic function y = ax² + bx + c satisfying specific conditions,
    prove that the distance between its roots is √17/2 -/
theorem quadratic_roots_distance (a b c : ℝ) : 
  (a*(-1)^2 + b*(-1) + c = -1) →
  (a*0^2 + b*0 + c = -2) →
  (a*1^2 + b*1 + c = 1) →
  let f := fun x => a*x^2 + b*x + c
  let roots := {x : ℝ | f x = 0}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ roots ∧ x₂ ∈ roots ∧ x₁ ≠ x₂ ∧ |x₁ - x₂| = Real.sqrt 17 / 2 :=
by sorry


end NUMINAMATH_CALUDE_quadratic_roots_distance_l3843_384375


namespace NUMINAMATH_CALUDE_a_power_b_is_one_fourth_l3843_384395

theorem a_power_b_is_one_fourth (a b : ℝ) (h : (a + b)^2 + |b + 2| = 0) : a^b = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_a_power_b_is_one_fourth_l3843_384395


namespace NUMINAMATH_CALUDE_ob_value_l3843_384306

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the points
variable (O F₁ F₂ A B : ℝ × ℝ)

-- State the conditions
variable (h1 : O = (0, 0))
variable (h2 : ellipse (F₁.1) (F₁.2))
variable (h3 : ellipse (F₂.1) (F₂.2))
variable (h4 : F₁.1 < 0 ∧ F₂.1 > 0)
variable (h5 : ellipse A.1 A.2)
variable (h6 : (A.1 - F₂.1) * (F₂.1 - F₁.1) + (A.2 - F₂.2) * (F₂.2 - F₁.2) = 0)
variable (h7 : B.1 = 0)
variable (h8 : ∃ t : ℝ, B = t • F₁ + (1 - t) • A)

-- State the theorem
theorem ob_value : abs B.2 = 3/4 := by sorry

end NUMINAMATH_CALUDE_ob_value_l3843_384306


namespace NUMINAMATH_CALUDE_right_triangle_345_l3843_384349

/-- A triangle with side lengths 3, 4, and 5 is a right triangle. -/
theorem right_triangle_345 :
  ∀ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5 →
  a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_345_l3843_384349


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l3843_384370

-- Define the points
def O : ℝ × ℝ := (0, 0)
def M : ℝ × ℝ := (1, 1)
def N : ℝ × ℝ := (4, 2)

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 6*y = 0

-- Theorem statement
theorem circle_passes_through_points :
  circle_equation O.1 O.2 ∧
  circle_equation M.1 M.2 ∧
  circle_equation N.1 N.2 := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l3843_384370


namespace NUMINAMATH_CALUDE_female_to_male_ratio_l3843_384367

/-- Represents a dog breed with the number of female and male puppies -/
structure Breed where
  name : String
  females : Nat
  males : Nat

/-- The litter of puppies -/
def litter : List Breed := [
  { name := "Golden Retriever", females := 2, males := 4 },
  { name := "Labrador", females := 1, males := 3 },
  { name := "Poodle", females := 3, males := 2 },
  { name := "Beagle", females := 1, males := 2 }
]

/-- The total number of female puppies in the litter -/
def totalFemales : Nat := litter.foldl (fun acc breed => acc + breed.females) 0

/-- The total number of male puppies in the litter -/
def totalMales : Nat := litter.foldl (fun acc breed => acc + breed.males) 0

/-- Theorem stating that the ratio of female to male puppies is 7:11 -/
theorem female_to_male_ratio :
  totalFemales = 7 ∧ totalMales = 11 := by sorry

end NUMINAMATH_CALUDE_female_to_male_ratio_l3843_384367


namespace NUMINAMATH_CALUDE_julie_count_correct_l3843_384300

/-- Represents the number of people with a given name in the crowd -/
structure NameCount where
  barry : ℕ
  kevin : ℕ
  julie : ℕ
  joe : ℕ

/-- Represents the proportion of nice people for each name -/
structure NiceProportion where
  barry : ℚ
  kevin : ℚ
  julie : ℚ
  joe : ℚ

/-- The total number of nice people in the crowd -/
def totalNicePeople : ℕ := 99

/-- The actual count of people with each name -/
def actualCount : NameCount where
  barry := 24
  kevin := 20
  julie := 80  -- This is what we want to prove
  joe := 50

/-- The proportion of nice people for each name -/
def niceProportion : NiceProportion where
  barry := 1
  kevin := 1/2
  julie := 3/4
  joe := 1/10

/-- Calculates the number of nice people for a given name -/
def niceCount (count : ℕ) (proportion : ℚ) : ℚ :=
  (count : ℚ) * proportion

/-- Theorem stating that the number of people named Julie is correct -/
theorem julie_count_correct :
  actualCount.julie = 80 ∧
  (niceCount actualCount.barry niceProportion.barry +
   niceCount actualCount.kevin niceProportion.kevin +
   niceCount actualCount.julie niceProportion.julie +
   niceCount actualCount.joe niceProportion.joe : ℚ) = totalNicePeople :=
by sorry

end NUMINAMATH_CALUDE_julie_count_correct_l3843_384300


namespace NUMINAMATH_CALUDE_smallest_n_boxes_l3843_384383

theorem smallest_n_boxes : ∃ n : ℕ, n > 0 ∧ 
  (∀ m : ℕ, m > 0 → m < n → ¬(17 * m - 3) % 11 = 0) ∧ 
  (17 * n - 3) % 11 = 0 ∧ 
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_boxes_l3843_384383


namespace NUMINAMATH_CALUDE_log_inequality_l3843_384376

theorem log_inequality : Real.log 2 / Real.log 3 < Real.log 3 / Real.log 2 ∧ 
                         Real.log 3 / Real.log 2 < Real.log 5 / Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l3843_384376


namespace NUMINAMATH_CALUDE_worker_c_left_days_l3843_384330

def work_rate (days : ℕ) : ℚ := 1 / days

theorem worker_c_left_days 
  (rate_a rate_b rate_c : ℚ)
  (total_days : ℕ)
  (h1 : rate_a = work_rate 30)
  (h2 : rate_b = work_rate 30)
  (h3 : rate_c = work_rate 40)
  (h4 : total_days = 12)
  : ∃ (x : ℕ), 
    (total_days - x) * (rate_a + rate_b + rate_c) + x * (rate_a + rate_b) = 1 ∧ 
    x = 8 := by
  sorry

end NUMINAMATH_CALUDE_worker_c_left_days_l3843_384330
