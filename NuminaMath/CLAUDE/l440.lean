import Mathlib

namespace NUMINAMATH_CALUDE_complex_modulus_problem_l440_44018

theorem complex_modulus_problem : Complex.abs ((Complex.I + 1) / Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l440_44018


namespace NUMINAMATH_CALUDE_tan_alpha_value_l440_44002

theorem tan_alpha_value (α : ℝ) (h : (Real.sin α - 2 * Real.cos α) / (2 * Real.sin α + 3 * Real.cos α) = 2) :
  Real.tan α = -8/3 := by sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l440_44002


namespace NUMINAMATH_CALUDE_problem_solution_l440_44041

theorem problem_solution (p q : ℝ) 
  (h1 : 1 < p) (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) 
  (h4 : p * q = 6) : 
  q = 3 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l440_44041


namespace NUMINAMATH_CALUDE_cleaning_hourly_rate_l440_44039

/-- Calculates the hourly rate for cleaning rooms in a building -/
theorem cleaning_hourly_rate
  (floors : ℕ)
  (rooms_per_floor : ℕ)
  (hours_per_room : ℕ)
  (total_earnings : ℕ)
  (h1 : floors = 4)
  (h2 : rooms_per_floor = 10)
  (h3 : hours_per_room = 6)
  (h4 : total_earnings = 3600) :
  total_earnings / (floors * rooms_per_floor * hours_per_room) = 15 := by
  sorry

#check cleaning_hourly_rate

end NUMINAMATH_CALUDE_cleaning_hourly_rate_l440_44039


namespace NUMINAMATH_CALUDE_negative_fraction_identification_l440_44026

-- Define a predicate for negative fractions
def is_negative_fraction (x : ℝ) : Prop :=
  ∃ (a b : ℤ), b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ) ∧ x < 0

-- Theorem statement
theorem negative_fraction_identification :
  is_negative_fraction (-0.7) ∧
  ¬is_negative_fraction (1/2) ∧
  ¬is_negative_fraction (-π) ∧
  ¬is_negative_fraction (-3/3) :=
by sorry

end NUMINAMATH_CALUDE_negative_fraction_identification_l440_44026


namespace NUMINAMATH_CALUDE_emily_candy_consumption_l440_44093

/-- Emily's Halloween candy problem -/
theorem emily_candy_consumption (neighbor_candy : ℕ) (sister_candy : ℕ) (days : ℕ) 
  (h1 : neighbor_candy = 5)
  (h2 : sister_candy = 13)
  (h3 : days = 2) :
  (neighbor_candy + sister_candy) / days = 9 := by
  sorry

end NUMINAMATH_CALUDE_emily_candy_consumption_l440_44093


namespace NUMINAMATH_CALUDE_angle_between_given_lines_l440_44067

-- Define the lines
def line1 (x y : ℝ) : Prop := x - 3 * y + 3 = 0
def line2 (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the angle between two lines
def angle_between_lines (l1 l2 : (ℝ → ℝ → Prop)) : ℝ := sorry

-- Theorem statement
theorem angle_between_given_lines :
  angle_between_lines line1 line2 = Real.arctan (1 / 2) := by sorry

end NUMINAMATH_CALUDE_angle_between_given_lines_l440_44067


namespace NUMINAMATH_CALUDE_empty_solution_set_iff_a_in_range_l440_44083

-- Define the quadratic function
def f (a x : ℝ) : ℝ := (a - 1) * x^2 + 2 * (a - 1) * x - 4

-- Define the property of empty solution set
def has_empty_solution_set (a : ℝ) : Prop :=
  ∀ x : ℝ, f a x < 0

-- State the theorem
theorem empty_solution_set_iff_a_in_range :
  ∀ a : ℝ, has_empty_solution_set a ↔ -3 < a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_empty_solution_set_iff_a_in_range_l440_44083


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l440_44027

/-- Given that i² = -1, prove that (1 - i) / (2 + 3i) = -1/13 - 5/13 * i -/
theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) :
  (1 - i) / (2 + 3*i) = -1/13 - 5/13 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l440_44027


namespace NUMINAMATH_CALUDE_peaches_per_basket_l440_44098

/-- The number of red peaches in each basket -/
def red_peaches : ℕ := 7

/-- The number of green peaches in each basket -/
def green_peaches : ℕ := 3

/-- The total number of peaches in each basket -/
def total_peaches : ℕ := red_peaches + green_peaches

theorem peaches_per_basket : total_peaches = 10 := by
  sorry

end NUMINAMATH_CALUDE_peaches_per_basket_l440_44098


namespace NUMINAMATH_CALUDE_black_water_bottles_l440_44024

theorem black_water_bottles (red : ℕ) (blue : ℕ) (taken_out : ℕ) (left : ℕ) :
  red = 2 →
  blue = 4 →
  taken_out = 5 →
  left = 4 →
  ∃ black : ℕ, red + black + blue = taken_out + left ∧ black = 3 :=
by sorry

end NUMINAMATH_CALUDE_black_water_bottles_l440_44024


namespace NUMINAMATH_CALUDE_bus_children_count_l440_44085

theorem bus_children_count (initial : ℕ) (joined : ℕ) (total : ℕ) : 
  initial = 64 → joined = 14 → total = initial + joined → total = 78 := by
  sorry

end NUMINAMATH_CALUDE_bus_children_count_l440_44085


namespace NUMINAMATH_CALUDE_sandwich_cost_is_two_dollars_l440_44043

/-- Represents the cost of ingredients for sandwiches -/
structure SandwichCost where
  bread : ℚ
  meat : ℚ
  cheese : ℚ
  meatPacks : ℕ
  cheesePacks : ℕ
  sandwiches : ℕ
  meatCoupon : ℚ
  cheeseCoupon : ℚ

/-- Calculates the cost per sandwich given the ingredients and coupons -/
def costPerSandwich (c : SandwichCost) : ℚ :=
  (c.bread + c.meat * c.meatPacks + c.cheese * c.cheesePacks - c.meatCoupon - c.cheeseCoupon) / c.sandwiches

/-- Theorem stating that the cost per sandwich is $2.00 given the specified conditions -/
theorem sandwich_cost_is_two_dollars :
  let c : SandwichCost := {
    bread := 4,
    meat := 5,
    cheese := 4,
    meatPacks := 2,
    cheesePacks := 2,
    sandwiches := 10,
    meatCoupon := 1,
    cheeseCoupon := 1
  }
  costPerSandwich c = 2 := by
  sorry


end NUMINAMATH_CALUDE_sandwich_cost_is_two_dollars_l440_44043


namespace NUMINAMATH_CALUDE_tangent_line_speed_l440_44012

theorem tangent_line_speed 
  (a T R L x : ℝ) 
  (h_pos : a > 0 ∧ T > 0 ∧ R > 0 ∧ L > 0)
  (h_eq : (a * T) / (a * T - R) = (L + x) / x) :
  x / T = a * L / R :=
sorry

end NUMINAMATH_CALUDE_tangent_line_speed_l440_44012


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l440_44050

-- Define the sets M and N
def M : Set ℝ := {x | -3 < x ∧ x ≤ 5}
def N : Set ℝ := {x | x < -5 ∨ x > 5}

-- State the theorem
theorem union_of_M_and_N : 
  M ∪ N = {x : ℝ | x < -5 ∨ x > -3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l440_44050


namespace NUMINAMATH_CALUDE_m_range_proof_l440_44079

theorem m_range_proof (m : ℝ) : 
  (∀ x : ℝ, (4 * x - m < 0 → -1 ≤ x ∧ x ≤ 2) ∧ 
  (∃ y : ℝ, -1 ≤ y ∧ y ≤ 2 ∧ ¬(4 * y - m < 0))) → 
  m > 8 := by
sorry

end NUMINAMATH_CALUDE_m_range_proof_l440_44079


namespace NUMINAMATH_CALUDE_rectangle_area_l440_44030

/-- Given a rectangle with perimeter 14 cm and diagonal 5 cm, its area is 12 square centimeters. -/
theorem rectangle_area (l w : ℝ) (h_perimeter : 2 * l + 2 * w = 14) 
  (h_diagonal : l^2 + w^2 = 5^2) : l * w = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l440_44030


namespace NUMINAMATH_CALUDE_logarithm_properties_l440_44007

-- Define the logarithm function for any base
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define lg as log base 10
noncomputable def lg (x : ℝ) : ℝ := log 10 x

theorem logarithm_properties :
  (lg 2 + lg 5 = 1) ∧ (log 3 9 = 2) := by sorry

end NUMINAMATH_CALUDE_logarithm_properties_l440_44007


namespace NUMINAMATH_CALUDE_dice_probability_l440_44078

/-- The number of sides on each die -/
def num_sides : ℕ := 12

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The number of one-digit outcomes on each die -/
def one_digit_outcomes : ℕ := 9

/-- The number of two-digit outcomes on each die -/
def two_digit_outcomes : ℕ := num_sides - one_digit_outcomes

/-- The probability of rolling a one-digit number on a single die -/
def prob_one_digit : ℚ := one_digit_outcomes / num_sides

/-- The probability of rolling a two-digit number on a single die -/
def prob_two_digit : ℚ := two_digit_outcomes / num_sides

/-- The number of dice showing one-digit numbers -/
def num_one_digit : ℕ := 3

/-- The number of dice showing two-digit numbers -/
def num_two_digit : ℕ := num_dice - num_one_digit

theorem dice_probability :
  (Nat.choose num_dice num_one_digit : ℚ) *
  (prob_one_digit ^ num_one_digit) *
  (prob_two_digit ^ num_two_digit) =
  135 / 512 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l440_44078


namespace NUMINAMATH_CALUDE_f_properties_l440_44074

/-- The function f(x) = mx² + 1 + ln x -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + 1 + Real.log x

/-- The derivative of f(x) -/
noncomputable def f_deriv (m : ℝ) (x : ℝ) : ℝ := 2 * m * x + 1 / x

/-- Theorem stating the main properties to be proved -/
theorem f_properties (m : ℝ) (n : ℝ) (a b : ℝ) :
  (∃ (t : ℝ), t = f_deriv m 1 ∧ 2 = f m 1 + t * (-2)) →  -- Tangent line condition
  (f m a = n ∧ f m b = n ∧ a < b) →                      -- Roots condition
  (∀ x > 0, f m x ≤ 1 - x) ∧                             -- Property 1
  (b - a < 1 - 2 * n)                                    -- Property 2
  := by sorry

end NUMINAMATH_CALUDE_f_properties_l440_44074


namespace NUMINAMATH_CALUDE_kendras_cookies_l440_44071

/-- Kendra's cookie problem -/
theorem kendras_cookies (cookies_per_batch : ℕ) (family_members : ℕ) (batches : ℕ) (chips_per_cookie : ℕ)
  (h1 : cookies_per_batch = 12)
  (h2 : family_members = 4)
  (h3 : batches = 3)
  (h4 : chips_per_cookie = 2) :
  (batches * cookies_per_batch / family_members) * chips_per_cookie = 18 := by
  sorry

end NUMINAMATH_CALUDE_kendras_cookies_l440_44071


namespace NUMINAMATH_CALUDE_select_blocks_count_l440_44089

/-- The number of ways to select 4 blocks from a 6x6 grid such that no two blocks are in the same row or column -/
def select_blocks : ℕ := (Nat.choose 6 4) * (Nat.choose 6 4) * (Nat.factorial 4)

/-- Theorem stating that the number of ways to select 4 blocks from a 6x6 grid
    such that no two blocks are in the same row or column is 5400 -/
theorem select_blocks_count : select_blocks = 5400 := by
  sorry

end NUMINAMATH_CALUDE_select_blocks_count_l440_44089


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l440_44025

/-- A quadratic function f(x) = x^2 + 2x + a has no real roots if and only if a > 1 -/
theorem quadratic_no_real_roots (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + a ≠ 0) ↔ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l440_44025


namespace NUMINAMATH_CALUDE_min_stamps_for_50_cents_l440_44045

/-- Represents the number of stamps needed to make a certain value -/
def StampCombination := ℕ × ℕ

/-- Calculates the total value of a stamp combination -/
def value (c : StampCombination) : ℕ := 3 * c.1 + 4 * c.2

/-- Calculates the total number of stamps in a combination -/
def total_stamps (c : StampCombination) : ℕ := c.1 + c.2

/-- Checks if a stamp combination is valid (equals 50 cents) -/
def is_valid (c : StampCombination) : Prop := value c = 50

/-- Theorem: The minimum number of stamps needed to make 50 cents is 13 -/
theorem min_stamps_for_50_cents :
  ∃ (c : StampCombination), is_valid c ∧
    total_stamps c = 13 ∧
    ∀ (d : StampCombination), is_valid d → total_stamps c ≤ total_stamps d :=
by
  sorry

#check min_stamps_for_50_cents

end NUMINAMATH_CALUDE_min_stamps_for_50_cents_l440_44045


namespace NUMINAMATH_CALUDE_round_robin_28_games_8_teams_l440_44075

/-- The number of games in a single round-robin tournament with n teams -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: A single round-robin tournament with 28 games requires 8 teams -/
theorem round_robin_28_games_8_teams :
  ∃ (n : ℕ), n > 0 ∧ num_games n = 28 ∧ n = 8 := by sorry

end NUMINAMATH_CALUDE_round_robin_28_games_8_teams_l440_44075


namespace NUMINAMATH_CALUDE_ivanov_petrov_probability_l440_44048

/-- The number of people in the group -/
def n : ℕ := 11

/-- The number of people that should be between Ivanov and Petrov -/
def k : ℕ := 3

/-- The probability of exactly k people sitting between two specific people
    in a random circular arrangement of n people -/
def probability (n k : ℕ) : ℚ :=
  if n > k + 1 then 1 / (n - 1) else 0

theorem ivanov_petrov_probability :
  probability n k = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_ivanov_petrov_probability_l440_44048


namespace NUMINAMATH_CALUDE_palindrome_product_sum_l440_44029

/-- A function that checks if a number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

/-- The theorem stating the sum of the two three-digit palindromes whose product is 522729 -/
theorem palindrome_product_sum : 
  ∃ (a b : ℕ), isThreeDigitPalindrome a ∧ 
                isThreeDigitPalindrome b ∧ 
                a * b = 522729 ∧ 
                a + b = 1366 := by
  sorry

end NUMINAMATH_CALUDE_palindrome_product_sum_l440_44029


namespace NUMINAMATH_CALUDE_simplify_fraction_l440_44095

theorem simplify_fraction (a b : ℝ) (h1 : a ≠ -b) (h2 : a ≠ 2*b) (h3 : a ≠ b) :
  (a + 2*b) / (a + b) - (a - b) / (a - 2*b) / ((a^2 - b^2) / (a^2 - 4*a*b + 4*b^2)) = 4*b / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l440_44095


namespace NUMINAMATH_CALUDE_angle_triple_supplement_l440_44013

theorem angle_triple_supplement (x : ℝ) : x = 3 * (180 - x) → x = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_supplement_l440_44013


namespace NUMINAMATH_CALUDE_trishas_walk_distance_l440_44034

theorem trishas_walk_distance :
  let distance_hotel_to_postcard : ℚ := 0.1111111111111111
  let distance_postcard_to_tshirt : ℚ := 0.1111111111111111
  let distance_tshirt_to_hotel : ℚ := 0.6666666666666666
  distance_hotel_to_postcard + distance_postcard_to_tshirt + distance_tshirt_to_hotel = 0.8888888888888888 := by
  sorry

end NUMINAMATH_CALUDE_trishas_walk_distance_l440_44034


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l440_44011

theorem rectangular_solid_volume
  (a b c : ℕ+)
  (h1 : a * b - c * a - b * c = 1)
  (h2 : c * a = b * c + 1) :
  a * b * c = 6 :=
sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l440_44011


namespace NUMINAMATH_CALUDE_temperature_conversion_l440_44009

theorem temperature_conversion (t k : ℚ) (f : ℚ) : 
  t = f * (k - 32) → t = 50 → k = 122 → f = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l440_44009


namespace NUMINAMATH_CALUDE_checkerboard_exists_l440_44035

/-- Represents the color of a cell -/
inductive Color
| Black
| White

/-- Represents a 100x100 board -/
def Board := Fin 100 → Fin 100 → Color

/-- Checks if a cell is adjacent to the boundary -/
def isAdjacentToBoundary (i j : Fin 100) : Prop :=
  i = 0 ∨ i = 99 ∨ j = 0 ∨ j = 99

/-- Checks if a 2x2 square is monochromatic -/
def isMonochromatic2x2 (board : Board) (i j : Fin 100) : Prop :=
  ∃ c : Color, 
    board i j = c ∧ 
    board (i+1) j = c ∧ 
    board i (j+1) = c ∧ 
    board (i+1) (j+1) = c

/-- Checks if a 2x2 square has a checkerboard pattern -/
def isCheckerboard2x2 (board : Board) (i j : Fin 100) : Prop :=
  (board i j = Color.Black ∧ board (i+1) (j+1) = Color.Black ∧
   board (i+1) j = Color.White ∧ board i (j+1) = Color.White) ∨
  (board i j = Color.White ∧ board (i+1) (j+1) = Color.White ∧
   board (i+1) j = Color.Black ∧ board i (j+1) = Color.Black)

theorem checkerboard_exists (board : Board) 
  (boundary_black : ∀ i j : Fin 100, isAdjacentToBoundary i j → board i j = Color.Black)
  (no_monochromatic : ∀ i j : Fin 100, ¬isMonochromatic2x2 board i j) :
  ∃ i j : Fin 100, isCheckerboard2x2 board i j :=
sorry

end NUMINAMATH_CALUDE_checkerboard_exists_l440_44035


namespace NUMINAMATH_CALUDE_sammys_homework_l440_44001

theorem sammys_homework (total : ℕ) (completed : ℕ) (h1 : total = 9) (h2 : completed = 2) :
  total - completed = 7 := by sorry

end NUMINAMATH_CALUDE_sammys_homework_l440_44001


namespace NUMINAMATH_CALUDE_men_to_women_percentage_l440_44051

/-- If the population of women is 50% of the population of men,
    then the population of men is 200% of the population of women. -/
theorem men_to_women_percentage (men women : ℝ) (h : women = 0.5 * men) :
  men / women * 100 = 200 := by
  sorry

end NUMINAMATH_CALUDE_men_to_women_percentage_l440_44051


namespace NUMINAMATH_CALUDE_max_consecutive_sum_45_l440_44016

/-- The sum of consecutive integers starting from a given integer -/
def sum_consecutive (start : ℤ) (count : ℕ) : ℤ :=
  (count : ℤ) * (2 * start + count - 1) / 2

/-- The property that a sequence of consecutive integers sums to 45 -/
def sums_to_45 (start : ℤ) (count : ℕ) : Prop :=
  sum_consecutive start count = 45

/-- The theorem stating that 90 is the maximum number of consecutive integers that sum to 45 -/
theorem max_consecutive_sum_45 :
  (∃ start : ℤ, sums_to_45 start 90) ∧
  (∀ count : ℕ, count > 90 → ∀ start : ℤ, ¬ sums_to_45 start count) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_45_l440_44016


namespace NUMINAMATH_CALUDE_complex_absolute_value_l440_44047

theorem complex_absolute_value : 
  Complex.abs (7/4 - 3*Complex.I + Real.sqrt 3) = 
  (Real.sqrt (241 + 56*Real.sqrt 3))/4 := by
sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l440_44047


namespace NUMINAMATH_CALUDE_class_size_l440_44046

theorem class_size (total_budget : ℕ) (souvenir_cost : ℕ) (remaining : ℕ) : 
  total_budget = 730 →
  souvenir_cost = 17 →
  remaining = 16 →
  (total_budget - remaining) / souvenir_cost = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_class_size_l440_44046


namespace NUMINAMATH_CALUDE_jacket_sale_price_l440_44090

/-- Proves that the price of each jacket after noon was $18.95 given the sale conditions --/
theorem jacket_sale_price (total_jackets : ℕ) (price_before_noon : ℚ) (total_receipts : ℚ) (jackets_sold_after_noon : ℕ) :
  total_jackets = 214 →
  price_before_noon = 31.95 →
  total_receipts = 5108.30 →
  jackets_sold_after_noon = 133 →
  (total_receipts - (total_jackets - jackets_sold_after_noon : ℚ) * price_before_noon) / jackets_sold_after_noon = 18.95 := by
  sorry

end NUMINAMATH_CALUDE_jacket_sale_price_l440_44090


namespace NUMINAMATH_CALUDE_square_division_exists_l440_44054

-- Define a rectangle
structure Rectangle where
  width : ℚ
  height : ℚ

-- Define a function to check if all numbers in a list are distinct
def allDistinct (list : List ℚ) : Prop :=
  ∀ i j, i ≠ j → list.get! i ≠ list.get! j

-- State the theorem
theorem square_division_exists : ∃ (rectangles : List Rectangle),
  -- There are 5 rectangles
  rectangles.length = 5 ∧
  -- The sum of areas equals 1
  (rectangles.map (λ r => r.width * r.height)).sum = 1 ∧
  -- All widths and heights are distinct
  allDistinct (rectangles.map (λ r => r.width) ++ rectangles.map (λ r => r.height)) :=
sorry

end NUMINAMATH_CALUDE_square_division_exists_l440_44054


namespace NUMINAMATH_CALUDE_correct_combined_average_l440_44040

def num_students : ℕ := 100
def math_avg : ℚ := 85
def science_avg : ℚ := 89
def num_incorrect : ℕ := 5

def incorrect_math_marks : List ℕ := [76, 80, 95, 70, 90]
def correct_math_marks : List ℕ := [86, 70, 75, 90, 100]
def incorrect_science_marks : List ℕ := [105, 60, 80, 92, 78]
def correct_science_marks : List ℕ := [95, 70, 90, 82, 88]

theorem correct_combined_average :
  let math_total := num_students * math_avg + (correct_math_marks.sum - incorrect_math_marks.sum)
  let science_total := num_students * science_avg + (correct_science_marks.sum - incorrect_science_marks.sum)
  let combined_total := math_total + science_total
  let combined_avg := combined_total / (2 * num_students)
  combined_avg = 87.1 := by sorry

end NUMINAMATH_CALUDE_correct_combined_average_l440_44040


namespace NUMINAMATH_CALUDE_scientific_notation_of_34000_l440_44037

theorem scientific_notation_of_34000 :
  (34000 : ℝ) = 3.4 * (10 : ℝ)^4 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_34000_l440_44037


namespace NUMINAMATH_CALUDE_algebraic_simplification_l440_44033

theorem algebraic_simplification (b : ℝ) : 3*b*(3*b^2 + 2*b - 1) - 2*b^2 = 9*b^3 + 4*b^2 - 3*b := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l440_44033


namespace NUMINAMATH_CALUDE_tangent_circles_area_sum_l440_44008

/-- A right triangle with sides 6, 8, and 10, where each vertex is the center of a circle
    and each circle is externally tangent to the other two. -/
structure TangentCirclesTriangle where
  -- The sides of the triangle
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  -- The radii of the circles
  radius1 : ℝ
  radius2 : ℝ
  radius3 : ℝ
  -- Conditions
  is_right_triangle : side1^2 + side2^2 = side3^2
  side_lengths : side1 = 6 ∧ side2 = 8 ∧ side3 = 10
  tangency1 : radius1 + radius2 = side3
  tangency2 : radius2 + radius3 = side1
  tangency3 : radius1 + radius3 = side2

/-- The sum of the areas of the circles in a TangentCirclesTriangle is 56π. -/
theorem tangent_circles_area_sum (t : TangentCirclesTriangle) :
  π * (t.radius1^2 + t.radius2^2 + t.radius3^2) = 56 * π := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_area_sum_l440_44008


namespace NUMINAMATH_CALUDE_digit_sum_unbounded_l440_44097

/-- Sum of digits of a positive integer -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sequence of sum of digits of a^n -/
def digitSumSequence (a : ℕ) (n : ℕ) : ℕ := sumOfDigits (a^n)

theorem digit_sum_unbounded (a : ℕ) (h1 : Even a) (h2 : ¬(5 ∣ a)) :
  ∀ M : ℕ, ∃ N : ℕ, ∀ n ≥ N, digitSumSequence a n > M :=
sorry

end NUMINAMATH_CALUDE_digit_sum_unbounded_l440_44097


namespace NUMINAMATH_CALUDE_problem_statement_l440_44044

theorem problem_statement (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = -6) :
  a^3 * b - 2 * a^2 * b^2 + a * b^3 = -150 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l440_44044


namespace NUMINAMATH_CALUDE_stars_per_bottle_l440_44081

/-- Given that Shiela prepared 45 paper stars and has 9 classmates,
    prove that the number of stars per bottle is 5. -/
theorem stars_per_bottle (total_stars : ℕ) (num_classmates : ℕ) 
  (h1 : total_stars = 45) (h2 : num_classmates = 9) :
  total_stars / num_classmates = 5 := by
  sorry

end NUMINAMATH_CALUDE_stars_per_bottle_l440_44081


namespace NUMINAMATH_CALUDE_revenue_calculation_l440_44038

/-- The revenue from a single sold-out performance for Steve's circus production -/
def revenue_per_performance : ℕ := sorry

/-- The overhead cost for Steve's circus production -/
def overhead_cost : ℕ := 81000

/-- The production cost per performance for Steve's circus production -/
def production_cost_per_performance : ℕ := 7000

/-- The number of sold-out performances needed to break even -/
def performances_to_break_even : ℕ := 9

/-- Theorem stating that the revenue from a single sold-out performance is $16,000 -/
theorem revenue_calculation :
  revenue_per_performance = 16000 :=
by
  sorry

#check revenue_calculation

end NUMINAMATH_CALUDE_revenue_calculation_l440_44038


namespace NUMINAMATH_CALUDE_equation_proof_l440_44060

theorem equation_proof : 16 * 0.2 * 5 * 0.5 / 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l440_44060


namespace NUMINAMATH_CALUDE_partnership_profit_theorem_l440_44061

/-- Represents the profit distribution in a partnership business -/
structure PartnershipProfit where
  investment_A : ℕ
  investment_B : ℕ
  investment_C : ℕ
  profit_share_C : ℕ

/-- Calculates the total profit of the partnership -/
def total_profit (p : PartnershipProfit) : ℕ :=
  (p.profit_share_C * (p.investment_A + p.investment_B + p.investment_C)) / p.investment_C

/-- Theorem stating that given the investments and C's profit share, the total profit is 80000 -/
theorem partnership_profit_theorem (p : PartnershipProfit) 
  (h1 : p.investment_A = 27000)
  (h2 : p.investment_B = 72000)
  (h3 : p.investment_C = 81000)
  (h4 : p.profit_share_C = 36000) :
  total_profit p = 80000 := by
  sorry

#eval total_profit { investment_A := 27000, investment_B := 72000, investment_C := 81000, profit_share_C := 36000 }

end NUMINAMATH_CALUDE_partnership_profit_theorem_l440_44061


namespace NUMINAMATH_CALUDE_root_implies_k_value_l440_44022

theorem root_implies_k_value (k : ℝ) : 
  (2 * (4 : ℝ)^2 + 3 * 4 - k = 0) → k = 44 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_k_value_l440_44022


namespace NUMINAMATH_CALUDE_unique_1x5x_divisible_by_36_l440_44087

def is_form_1x5x (n : ℕ) : Prop :=
  ∃ x : ℕ, x < 10 ∧ n = 1000 + 100 * x + 50 + x

theorem unique_1x5x_divisible_by_36 :
  ∃! n : ℕ, is_form_1x5x n ∧ n % 36 = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_1x5x_divisible_by_36_l440_44087


namespace NUMINAMATH_CALUDE_partnership_profit_calculation_l440_44057

/-- Represents the profit distribution in a partnership business -/
structure ProfitDistribution where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  c_profit : ℕ

/-- Calculates the total profit given a profit distribution -/
def total_profit (pd : ProfitDistribution) : ℕ :=
  let total_investment := pd.a_investment + pd.b_investment + pd.c_investment
  let c_ratio := pd.c_investment / (total_investment / 20)
  (pd.c_profit * 20) / c_ratio

/-- Theorem stating that given the specific investments and c's profit, 
    the total profit is $60,000 -/
theorem partnership_profit_calculation (pd : ProfitDistribution) 
  (h1 : pd.a_investment = 45000)
  (h2 : pd.b_investment = 63000)
  (h3 : pd.c_investment = 72000)
  (h4 : pd.c_profit = 24000) :
  total_profit pd = 60000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_calculation_l440_44057


namespace NUMINAMATH_CALUDE_initial_group_size_l440_44092

theorem initial_group_size (total_groups : Nat) (students_left : Nat) (remaining_students : Nat) :
  total_groups = 3 →
  students_left = 2 →
  remaining_students = 22 →
  ∃ initial_group_size : Nat, 
    initial_group_size * total_groups - students_left = remaining_students ∧
    initial_group_size = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_group_size_l440_44092


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l440_44096

theorem imaginary_part_of_z (z : ℂ) (h : 1 + 2*I = I * z) : z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l440_44096


namespace NUMINAMATH_CALUDE_two_distinct_roots_implies_b_value_l440_44072

-- Define the polynomial function
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - b*x^2 + 1

-- State the theorem
theorem two_distinct_roots_implies_b_value (b : ℝ) :
  (∃ (x y : ℝ), x ≠ y ∧ f b x = 0 ∧ f b y = 0 ∧ 
   ∀ (z : ℝ), f b z = 0 → (z = x ∨ z = y)) →
  b = (3/2) * Real.rpow 2 (1/3) :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_implies_b_value_l440_44072


namespace NUMINAMATH_CALUDE_min_value_of_f_on_interval_l440_44015

def f (x : ℝ) := -x^2 + 4*x - 2

theorem min_value_of_f_on_interval :
  ∃ (m : ℝ), m = -2 ∧ ∀ x ∈ Set.Icc 1 4, f x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_on_interval_l440_44015


namespace NUMINAMATH_CALUDE_cirrus_cloud_count_l440_44042

/-- The number of cumulonimbus clouds -/
def cumulonimbus : ℕ := 3

/-- The number of cumulus clouds -/
def cumulus : ℕ := 12 * cumulonimbus

/-- The number of cirrus clouds -/
def cirrus : ℕ := 4 * cumulus

theorem cirrus_cloud_count : cirrus = 144 := by
  sorry

end NUMINAMATH_CALUDE_cirrus_cloud_count_l440_44042


namespace NUMINAMATH_CALUDE_remainder_sum_factorials_l440_44077

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem remainder_sum_factorials (n : ℕ) (h : n ≥ 50) :
  sum_factorials n % 24 = (factorial 1 + factorial 2 + factorial 3 + factorial 4) % 24 :=
sorry

end NUMINAMATH_CALUDE_remainder_sum_factorials_l440_44077


namespace NUMINAMATH_CALUDE_line_through_P_parallel_to_AB_circle_equation_is_circumcircle_OAB_l440_44099

-- Define the points
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (0, 2)
def P : ℝ × ℝ := (2, 3)
def O : ℝ × ℝ := (0, 0)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + 2*y - 8 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 5

-- Theorem for the line equation
theorem line_through_P_parallel_to_AB :
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | line_equation p.1 p.2} ↔
  ∃ t : ℝ, x = 2 + t ∧ y = 3 - t/2 :=
sorry

-- Theorem for the circle equation
theorem circle_equation_is_circumcircle_OAB :
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | circle_equation p.1 p.2} ↔
  (x - O.1)^2 + (y - O.2)^2 = (x - A.1)^2 + (y - A.2)^2 ∧
  (x - O.1)^2 + (y - O.2)^2 = (x - B.1)^2 + (y - B.2)^2 :=
sorry

end NUMINAMATH_CALUDE_line_through_P_parallel_to_AB_circle_equation_is_circumcircle_OAB_l440_44099


namespace NUMINAMATH_CALUDE_rebus_solution_l440_44055

/-- Represents a two-digit number (or single-digit for YA) -/
def TwoDigitNum := {n : ℕ // n < 100}

/-- The rebus equation -/
def rebusEquation (ya oh my : TwoDigitNum) : Prop :=
  ya.val + 8 * oh.val = my.val

/-- All digits in the equation are different -/
def differentDigits (ya oh my : TwoDigitNum) : Prop :=
  ya.val ≠ oh.val ∧ ya.val ≠ my.val ∧ oh.val ≠ my.val

theorem rebus_solution :
  ∃! (ya oh my : TwoDigitNum),
    rebusEquation ya oh my ∧
    differentDigits ya oh my ∧
    ya.val = 0 ∧
    oh.val = 12 ∧
    my.val = 96 :=
sorry

end NUMINAMATH_CALUDE_rebus_solution_l440_44055


namespace NUMINAMATH_CALUDE_max_groups_is_two_l440_44069

/-- Represents the number of boys in the class -/
def num_boys : ℕ := 20

/-- Represents the number of girls in the class -/
def num_girls : ℕ := 24

/-- Represents the total number of students in the class -/
def total_students : ℕ := num_boys + num_girls

/-- Represents the number of age groups -/
def num_age_groups : ℕ := 3

/-- Represents the number of skill levels -/
def num_skill_levels : ℕ := 3

/-- Represents the maximum number of groups that can be formed -/
def max_groups : ℕ := 2

/-- Theorem stating that the maximum number of groups is 2 -/
theorem max_groups_is_two :
  (num_boys % max_groups = 0) ∧
  (num_girls % max_groups = 0) ∧
  (total_students % max_groups = 0) ∧
  (max_groups % num_age_groups = 0) ∧
  (max_groups % num_skill_levels = 0) ∧
  (∀ n : ℕ, n > max_groups →
    (num_boys % n ≠ 0) ∨
    (num_girls % n ≠ 0) ∨
    (total_students % n ≠ 0) ∨
    (n % num_age_groups ≠ 0) ∨
    (n % num_skill_levels ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_max_groups_is_two_l440_44069


namespace NUMINAMATH_CALUDE_camp_children_count_l440_44020

/-- The initial number of children in the camp -/
def initial_children : ℕ := 50

/-- The fraction of boys in the initial group -/
def boys_fraction : ℚ := 4/5

/-- The number of boys added -/
def boys_added : ℕ := 50

/-- The fraction of girls in the final group -/
def final_girls_fraction : ℚ := 1/10

theorem camp_children_count :
  (initial_children : ℚ) * (1 - boys_fraction) = 
    final_girls_fraction * (initial_children + boys_added) := by
  sorry

end NUMINAMATH_CALUDE_camp_children_count_l440_44020


namespace NUMINAMATH_CALUDE_min_cubes_is_60_l440_44053

/-- The dimensions of the box in centimeters -/
def box_dimensions : Fin 3 → ℕ
| 0 => 30
| 1 => 40
| 2 => 50
| _ => 0

/-- The function to calculate the minimum number of cubes -/
def min_cubes (dimensions : Fin 3 → ℕ) : ℕ :=
  let cube_side := Nat.gcd (dimensions 0) (Nat.gcd (dimensions 1) (dimensions 2))
  (dimensions 0 / cube_side) * (dimensions 1 / cube_side) * (dimensions 2 / cube_side)

/-- Theorem stating that the minimum number of cubes is 60 -/
theorem min_cubes_is_60 : min_cubes box_dimensions = 60 := by
  sorry

#eval min_cubes box_dimensions

end NUMINAMATH_CALUDE_min_cubes_is_60_l440_44053


namespace NUMINAMATH_CALUDE_equal_marked_cells_exist_l440_44065

/-- Represents an L-shaped triomino -/
structure Triomino where
  cells : Fin 3 → (Fin 2010 × Fin 2010)

/-- Represents a marking of cells in the grid -/
def Marking := Fin 2010 → Fin 2010 → Bool

/-- Checks if a marking is valid (one cell per triomino) -/
def isValidMarking (grid : List Triomino) (m : Marking) : Prop := sorry

/-- Counts marked cells in a given row -/
def countMarkedInRow (m : Marking) (row : Fin 2010) : Nat := sorry

/-- Counts marked cells in a given column -/
def countMarkedInColumn (m : Marking) (col : Fin 2010) : Nat := sorry

/-- Main theorem statement -/
theorem equal_marked_cells_exist (grid : List Triomino) 
  (h : grid.length = 2010 * 2010 / 3) : 
  ∃ m : Marking, 
    isValidMarking grid m ∧ 
    (∀ r₁ r₂ : Fin 2010, countMarkedInRow m r₁ = countMarkedInRow m r₂) ∧
    (∀ c₁ c₂ : Fin 2010, countMarkedInColumn m c₁ = countMarkedInColumn m c₂) := by
  sorry

end NUMINAMATH_CALUDE_equal_marked_cells_exist_l440_44065


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l440_44080

theorem quadratic_equation_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h1 : c^2 + c*c + d = 0) (h2 : (2*d)^2 + c*(2*d) + d = 0) : 
  c = (1 : ℝ) / 2 ∧ d = -(1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l440_44080


namespace NUMINAMATH_CALUDE_roses_cut_correct_l440_44031

/-- The number of roses Mary cut from her garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Theorem stating that the number of roses Mary cut is correct -/
theorem roses_cut_correct (initial_roses final_roses : ℕ) 
  (h : initial_roses ≤ final_roses) : 
  roses_cut initial_roses final_roses = final_roses - initial_roses :=
by
  sorry

#eval roses_cut 6 16

end NUMINAMATH_CALUDE_roses_cut_correct_l440_44031


namespace NUMINAMATH_CALUDE_sheridan_fish_count_l440_44064

/-- Calculate the number of fish Mrs. Sheridan has left -/
def fish_remaining (initial : ℕ) (received : ℕ) (given_away : ℕ) (sold : ℕ) : ℕ :=
  initial + received - given_away - sold

/-- Theorem stating that Mrs. Sheridan has 46 fish left -/
theorem sheridan_fish_count : fish_remaining 22 47 15 8 = 46 := by
  sorry

end NUMINAMATH_CALUDE_sheridan_fish_count_l440_44064


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_k_nonzero_l440_44059

/-- Given a quadratic equation kx^2 - 2x + 1/2 = 0, if it has two distinct real roots, then k ≠ 0 -/
theorem quadratic_distinct_roots_k_nonzero (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 2*x + 1/2 = 0 ∧ k * y^2 - 2*y + 1/2 = 0) → k ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_k_nonzero_l440_44059


namespace NUMINAMATH_CALUDE_trig_problem_l440_44049

theorem trig_problem (α : Real) (h1 : 0 < α) (h2 : α < Real.pi / 2) 
  (h3 : Real.sin (α + Real.pi / 6) = 3 / 5) : 
  Real.cos (2 * α - Real.pi / 6) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_trig_problem_l440_44049


namespace NUMINAMATH_CALUDE_equation_solution_l440_44032

theorem equation_solution (x : Real) :
  x ∈ Set.Ioo (-π / 2) 0 →
  (Real.sqrt 3 / Real.sin x) + (1 / Real.cos x) = 4 →
  x = -4 * π / 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l440_44032


namespace NUMINAMATH_CALUDE_inequality_solution_l440_44017

theorem inequality_solution (x : ℝ) :
  x ≥ 0 →
  (2021 * (x^2020)^(1/202) - 1 ≥ 2020 * x) ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l440_44017


namespace NUMINAMATH_CALUDE_tan_double_special_angle_l440_44076

/-- An angle with vertex at the origin, initial side on positive x-axis, and terminal side on y = 2x -/
structure SpecialAngle where
  θ : Real
  terminal_side : (x : Real) → y = 2 * x

theorem tan_double_special_angle (α : SpecialAngle) : Real.tan (2 * α.θ) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_special_angle_l440_44076


namespace NUMINAMATH_CALUDE_largest_d_for_g_range_contains_one_l440_44052

/-- The quadratic function g(x) defined as 2x^2 - 8x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 8 * x + d

/-- Theorem stating that the largest value of d such that 1 is in the range of g(x) is 9 -/
theorem largest_d_for_g_range_contains_one :
  (∃ (d : ℝ), ∀ (d' : ℝ), (∃ (x : ℝ), g d' x = 1) → d' ≤ d) ∧
  (∃ (x : ℝ), g 9 x = 1) :=
sorry

end NUMINAMATH_CALUDE_largest_d_for_g_range_contains_one_l440_44052


namespace NUMINAMATH_CALUDE_commission_rate_proof_l440_44063

/-- The commission rate for an agent who earned a commission of 12.50 on sales of 250. -/
theorem commission_rate_proof (commission : ℝ) (sales : ℝ) 
  (h1 : commission = 12.50) (h2 : sales = 250) :
  (commission / sales) * 100 = 5 := by
sorry

end NUMINAMATH_CALUDE_commission_rate_proof_l440_44063


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l440_44005

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 4) 
  (h2 : a^2 + b^2 = 30) : 
  a * b = 32 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l440_44005


namespace NUMINAMATH_CALUDE_special_circle_equation_l440_44021

/-- A circle passing through the origin with center on the negative x-axis and radius 2 -/
structure SpecialCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_negative_x_axis : center.1 < 0 ∧ center.2 = 0
  passes_through_origin : (center.1 ^ 2 + center.2 ^ 2) = radius ^ 2
  radius_is_two : radius = 2

/-- The equation of the special circle is (x+2)^2 + y^2 = 4 -/
theorem special_circle_equation (c : SpecialCircle) :
  ∀ (x y : ℝ), ((x + 2) ^ 2 + y ^ 2 = 4) ↔ 
  ((x - c.center.1) ^ 2 + (y - c.center.2) ^ 2 = c.radius ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_special_circle_equation_l440_44021


namespace NUMINAMATH_CALUDE_tan_theta_three_expression_l440_44028

theorem tan_theta_three_expression (θ : Real) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ ^ 2) = (11 * Real.sqrt 10 - 101) / 33 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_three_expression_l440_44028


namespace NUMINAMATH_CALUDE_profit_sharing_ratio_l440_44006

/-- Represents the capital contribution and duration for a business partner -/
structure Partner where
  capital : ℕ
  duration : ℕ

/-- Calculates the effective capital contribution of a partner -/
def effectiveCapital (p : Partner) : ℕ := p.capital * p.duration

/-- Represents the business scenario with two partners -/
structure Business where
  partnerA : Partner
  partnerB : Partner

/-- The given business scenario -/
def givenBusiness : Business :=
  { partnerA := { capital := 3500, duration := 12 }
  , partnerB := { capital := 21000, duration := 3 }
  }

/-- Theorem stating that the profit sharing ratio is 2:3 for the given business -/
theorem profit_sharing_ratio (b : Business := givenBusiness) :
  (effectiveCapital b.partnerA) * 3 = (effectiveCapital b.partnerB) * 2 := by
  sorry

end NUMINAMATH_CALUDE_profit_sharing_ratio_l440_44006


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l440_44014

/-- Represents the number of advertisements --/
def total_ads : ℕ := 5

/-- Represents the number of commercial advertisements --/
def commercial_ads : ℕ := 3

/-- Represents the number of public service advertisements --/
def public_service_ads : ℕ := 2

/-- Calculates the number of ways to arrange the advertisements --/
def arrangement_count : ℕ := 36

/-- Theorem stating that the number of valid arrangements is 36 --/
theorem valid_arrangements_count :
  (total_ads = commercial_ads + public_service_ads) →
  (public_service_ads > 0) →
  (arrangement_count = 36) := by
  sorry

#check valid_arrangements_count

end NUMINAMATH_CALUDE_valid_arrangements_count_l440_44014


namespace NUMINAMATH_CALUDE_cos_300_degrees_l440_44086

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l440_44086


namespace NUMINAMATH_CALUDE_sqrt_x_plus_y_equals_two_l440_44070

theorem sqrt_x_plus_y_equals_two (x y : ℝ) (h : Real.sqrt (3 - x) + Real.sqrt (x - 3) + 1 = y) :
  Real.sqrt (x + y) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_y_equals_two_l440_44070


namespace NUMINAMATH_CALUDE_lily_to_rose_ratio_l440_44088

def number_of_roses : ℕ := 20
def cost_of_rose : ℕ := 5
def total_spent : ℕ := 250

theorem lily_to_rose_ratio :
  let cost_of_lily : ℕ := 2 * cost_of_rose
  let total_spent_on_roses : ℕ := number_of_roses * cost_of_rose
  let total_spent_on_lilies : ℕ := total_spent - total_spent_on_roses
  let number_of_lilies : ℕ := total_spent_on_lilies / cost_of_lily
  (number_of_lilies : ℚ) / (number_of_roses : ℚ) = 3 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_lily_to_rose_ratio_l440_44088


namespace NUMINAMATH_CALUDE_least_divisor_for_perfect_square_l440_44023

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y * y

theorem least_divisor_for_perfect_square :
  ∃! n : ℕ, n > 0 ∧ is_perfect_square (16800 / n) ∧ 
  ∀ m : ℕ, m > 0 → is_perfect_square (16800 / m) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_divisor_for_perfect_square_l440_44023


namespace NUMINAMATH_CALUDE_company_employees_l440_44019

theorem company_employees (total : ℕ) 
  (h1 : (60 : ℚ) / 100 * total = (total : ℚ).floor)
  (h2 : (20 : ℚ) / 100 * total = ((40 : ℚ) / 100 * total).floor / 2)
  (h3 : (60 : ℚ) / 100 * total = (20 : ℚ) / 100 * total + 40) :
  total = 100 := by
sorry

end NUMINAMATH_CALUDE_company_employees_l440_44019


namespace NUMINAMATH_CALUDE_sum_of_parts_l440_44068

/-- Given a number 24 divided into two parts, where the first part is 13.0,
    prove that the sum of 7 times the first part and 5 times the second part is 146. -/
theorem sum_of_parts (first_part second_part : ℝ) : 
  first_part + second_part = 24 →
  first_part = 13 →
  7 * first_part + 5 * second_part = 146 := by
sorry

end NUMINAMATH_CALUDE_sum_of_parts_l440_44068


namespace NUMINAMATH_CALUDE_tangent_line_to_unit_circle_l440_44084

/-- The equation of the tangent line to the unit circle at point (a, b) -/
theorem tangent_line_to_unit_circle (a b : ℝ) (h : a^2 + b^2 = 1) :
  ∀ x y : ℝ, (x - a)^2 + (y - b)^2 = (a*x + b*y - 1)^2 / (a^2 + b^2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_unit_circle_l440_44084


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l440_44073

/-- A parabola with focus at (2, 0) that opens to the right has the standard equation y² = 8x -/
theorem parabola_standard_equation (f : ℝ × ℝ) (opens_right : Bool) :
  f = (2, 0) → opens_right = true → ∃ (x y : ℝ), y^2 = 8*x := by
  sorry

end NUMINAMATH_CALUDE_parabola_standard_equation_l440_44073


namespace NUMINAMATH_CALUDE_final_sum_theorem_l440_44010

theorem final_sum_theorem (x y R : ℝ) (h : x + y = R) :
  3 * (x + 5) + 3 * (y + 5) = 3 * R + 30 :=
by sorry

end NUMINAMATH_CALUDE_final_sum_theorem_l440_44010


namespace NUMINAMATH_CALUDE_k_domain_l440_44058

-- Define the function h
def h : ℝ → ℝ := sorry

-- Define the domain of h
def h_domain : Set ℝ := Set.Icc (-8) 4

-- Define the function k in terms of h
def k (x : ℝ) : ℝ := h (3 * x + 1)

-- State the theorem
theorem k_domain :
  {x : ℝ | k x ∈ Set.range h} = Set.Icc (-3) 1 := by sorry

end NUMINAMATH_CALUDE_k_domain_l440_44058


namespace NUMINAMATH_CALUDE_viggo_payment_l440_44091

/-- Represents the denomination of the other bills used by Viggo --/
def other_denomination : ℕ := sorry

/-- The total amount spent on the shirt --/
def total_spent : ℕ := 80

/-- The number of other denomination bills used --/
def num_other_bills : ℕ := 2

/-- The denomination of the $20 bills --/
def twenty_bill : ℕ := 20

/-- The number of $20 bills used --/
def num_twenty_bills : ℕ := num_other_bills + 1

theorem viggo_payment :
  (num_twenty_bills * twenty_bill) + (num_other_bills * other_denomination) = total_spent ∧
  other_denomination = 10 := by sorry

end NUMINAMATH_CALUDE_viggo_payment_l440_44091


namespace NUMINAMATH_CALUDE_b₁_value_l440_44082

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 8 + 32*x - 12*x^2 - 4*x^3 + x^4

-- Define the set of roots of f(x)
def roots_f : Set ℝ := {x | f x = 0}

-- Define the polynomial g(x)
def g (b₀ b₁ b₂ b₃ : ℝ) (x : ℝ) : ℝ := b₀ + b₁*x + b₂*x^2 + b₃*x^3 + x^4

-- Define the set of roots of g(x)
def roots_g (b₀ b₁ b₂ b₃ : ℝ) : Set ℝ := {x | g b₀ b₁ b₂ b₃ x = 0}

theorem b₁_value (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : roots_f = {x₁, x₂, x₃, x₄})
  (h₂ : x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄)
  (h₃ : ∃ b₀ b₁ b₂ b₃, roots_g b₀ b₁ b₂ b₃ = {x₁^2, x₂^2, x₃^2, x₄^2}) :
  ∃ b₀ b₂ b₃, g b₀ (-1024) b₂ b₃ = g b₀ b₁ b₂ b₃ := by sorry

end NUMINAMATH_CALUDE_b₁_value_l440_44082


namespace NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_5_fifteenth_odd_multiple_of_5_is_145_l440_44094

/-- The nth positive odd multiple of 5 -/
def nthOddMultipleOf5 (n : ℕ) : ℕ := 10 * n - 5

theorem fifteenth_odd_multiple_of_5 : nthOddMultipleOf5 15 = 145 := by
  sorry

/-- The 15th positive integer that is both odd and a multiple of 5 -/
def fifteenthOddMultipleOf5 : ℕ := nthOddMultipleOf5 15

theorem fifteenth_odd_multiple_of_5_is_145 : fifteenthOddMultipleOf5 = 145 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_5_fifteenth_odd_multiple_of_5_is_145_l440_44094


namespace NUMINAMATH_CALUDE_quadratic_inequality_l440_44066

theorem quadratic_inequality (x : ℝ) : x^2 - 6*x - 16 > 0 ↔ x < -2 ∨ x > 8 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l440_44066


namespace NUMINAMATH_CALUDE_vector_BC_coordinates_l440_44062

/-- Given points A and B, and vector AC, prove that vector BC has specific coordinates -/
theorem vector_BC_coordinates (A B C : ℝ × ℝ) : 
  A = (0, 1) → B = (3, 2) → C - A = (-4, -3) → C - B = (-7, -4) := by
  sorry

end NUMINAMATH_CALUDE_vector_BC_coordinates_l440_44062


namespace NUMINAMATH_CALUDE_total_pens_equals_sum_l440_44056

/-- The number of pens given to friends -/
def pens_given : ℕ := 22

/-- The number of pens kept for herself -/
def pens_kept : ℕ := 34

/-- The total number of pens bought by her parents -/
def total_pens : ℕ := pens_given + pens_kept

/-- Theorem stating that the total number of pens is the sum of pens given and pens kept -/
theorem total_pens_equals_sum : total_pens = pens_given + pens_kept := by sorry

end NUMINAMATH_CALUDE_total_pens_equals_sum_l440_44056


namespace NUMINAMATH_CALUDE_points_four_units_away_l440_44000

theorem points_four_units_away (P : ℝ) : 
  P = -3 → {x : ℝ | |x - P| = 4} = {1, -7} := by sorry

end NUMINAMATH_CALUDE_points_four_units_away_l440_44000


namespace NUMINAMATH_CALUDE_profit_percentage_previous_year_l440_44003

theorem profit_percentage_previous_year
  (revenue_prev : ℝ)
  (revenue_2009 : ℝ)
  (profit_prev : ℝ)
  (profit_2009 : ℝ)
  (h1 : revenue_2009 = 0.8 * revenue_prev)
  (h2 : profit_2009 = 0.11 * revenue_2009)
  (h3 : profit_2009 = 0.8800000000000001 * profit_prev) :
  profit_prev = 0.1 * revenue_prev :=
sorry

end NUMINAMATH_CALUDE_profit_percentage_previous_year_l440_44003


namespace NUMINAMATH_CALUDE_segment_length_l440_44004

theorem segment_length : Real.sqrt 157 = Real.sqrt ((8 - 2)^2 + (18 - 7)^2) := by sorry

end NUMINAMATH_CALUDE_segment_length_l440_44004


namespace NUMINAMATH_CALUDE_tenth_angle_measure_l440_44036

/-- The sum of interior angles of a decagon -/
def decagon_angle_sum : ℝ := 1440

/-- The number of angles in a decagon that are 150° -/
def num_150_angles : ℕ := 9

/-- The measure of each of the known angles -/
def known_angle_measure : ℝ := 150

theorem tenth_angle_measure (decagon_sum : ℝ) (num_known : ℕ) (known_measure : ℝ) 
  (h1 : decagon_sum = decagon_angle_sum) 
  (h2 : num_known = num_150_angles) 
  (h3 : known_measure = known_angle_measure) :
  decagon_sum - num_known * known_measure = 90 := by
  sorry

end NUMINAMATH_CALUDE_tenth_angle_measure_l440_44036
