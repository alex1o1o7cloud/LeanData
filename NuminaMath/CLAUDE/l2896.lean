import Mathlib

namespace NUMINAMATH_CALUDE_map_distance_theorem_l2896_289607

/-- Represents the scale of a map as a ratio -/
def MapScale : ℚ := 1 / 250000

/-- Converts kilometers to centimeters -/
def kmToCm (km : ℚ) : ℚ := km * 100000

/-- Calculates the distance on a map given the actual distance and map scale -/
def mapDistance (actualDistance : ℚ) (scale : ℚ) : ℚ :=
  actualDistance * scale

theorem map_distance_theorem (actualDistanceKm : ℚ) 
  (h : actualDistanceKm = 5) : 
  mapDistance (kmToCm actualDistanceKm) MapScale = 2 := by
  sorry

#check map_distance_theorem

end NUMINAMATH_CALUDE_map_distance_theorem_l2896_289607


namespace NUMINAMATH_CALUDE_rectangle_shading_l2896_289699

theorem rectangle_shading (total_rectangles : ℕ) 
  (h1 : total_rectangles = 12) : 
  (2 : ℚ) / 3 * (3 : ℚ) / 4 * total_rectangles = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_shading_l2896_289699


namespace NUMINAMATH_CALUDE_lizas_peanut_butter_cookies_l2896_289635

/-- Given the conditions of Liza's cookie-making scenario, prove that she used 2/5 of the remaining butter for peanut butter cookies. -/
theorem lizas_peanut_butter_cookies (total_butter : ℝ) (remaining_butter : ℝ) (peanut_butter_fraction : ℝ) :
  total_butter = 10 →
  remaining_butter = total_butter / 2 →
  2 = remaining_butter - peanut_butter_fraction * remaining_butter - (1 / 3) * (remaining_butter - peanut_butter_fraction * remaining_butter) →
  peanut_butter_fraction = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_lizas_peanut_butter_cookies_l2896_289635


namespace NUMINAMATH_CALUDE_expression_simplification_l2896_289646

theorem expression_simplification (a : ℕ) (h : a = 2021) :
  let expr := (a + 1 : ℚ) / a + 1 / (a + 1) - a / (a + 1)
  expr = (a^2 + a + 2 : ℚ) / (a * (a + 1)) ∧
  (a^2 + a + 2 = 4094865) := by
  sorry

#check expression_simplification

end NUMINAMATH_CALUDE_expression_simplification_l2896_289646


namespace NUMINAMATH_CALUDE_pond_a_twice_pond_b_total_frogs_is_48_l2896_289695

/-- The number of frogs in Pond A -/
def frogs_in_pond_a : ℕ := 32

/-- The number of frogs in Pond B -/
def frogs_in_pond_b : ℕ := frogs_in_pond_a / 2

/-- Pond A has twice as many frogs as Pond B -/
theorem pond_a_twice_pond_b : frogs_in_pond_a = 2 * frogs_in_pond_b := by sorry

/-- The total number of frogs in both ponds -/
def total_frogs : ℕ := frogs_in_pond_a + frogs_in_pond_b

/-- Theorem: The total number of frogs in both ponds is 48 -/
theorem total_frogs_is_48 : total_frogs = 48 := by sorry

end NUMINAMATH_CALUDE_pond_a_twice_pond_b_total_frogs_is_48_l2896_289695


namespace NUMINAMATH_CALUDE_pyramid_stone_count_l2896_289677

/-- Calculates the total number of stones in a pyramid -/
def pyramid_stones (bottom : ℕ) (top : ℕ) (diff : ℕ) : ℕ :=
  let n := (bottom - top) / diff + 1
  n * (bottom + top) / 2

/-- Theorem: A pyramid with 10 stones on the bottom row, 2 stones on the top row,
    and each row having 2 fewer stones than the row beneath it, contains 30 stones in total. -/
theorem pyramid_stone_count : pyramid_stones 10 2 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_stone_count_l2896_289677


namespace NUMINAMATH_CALUDE_prime_power_sum_l2896_289600

theorem prime_power_sum (w x y z : ℕ) :
  2^w * 3^x * 5^y * 7^z = 588 → 2*w + 3*x + 5*y + 7*z = 21 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_sum_l2896_289600


namespace NUMINAMATH_CALUDE_equation_solutions_l2896_289616

theorem equation_solutions :
  (∀ x, x^2 - 8*x + 1 = 0 ↔ x = 4 + Real.sqrt 15 ∨ x = 4 - Real.sqrt 15) ∧
  (∀ x, 3*x*(x - 1) = 2 - 2*x ↔ x = 1 ∨ x = -2/3) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2896_289616


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l2896_289679

theorem decimal_sum_to_fraction :
  (0.1 : ℚ) + 0.02 + 0.003 + 0.0004 + 0.00005 + 0.000006 + 0.0000007 = 1234567 / 10000000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l2896_289679


namespace NUMINAMATH_CALUDE_upper_bound_of_prime_set_l2896_289606

theorem upper_bound_of_prime_set (A : Set ℕ) : 
  (∀ x ∈ A, Nat.Prime x) →   -- A contains only prime numbers
  (∃ a ∈ A, a > 62) →        -- Lower bound is greater than 62
  (∀ a ∈ A, a > 62) →        -- All elements are greater than 62
  (∃ max min : ℕ, max ∈ A ∧ min ∈ A ∧ max - min = 16 ∧
    ∀ a ∈ A, min ≤ a ∧ a ≤ max) →  -- Range of A is 16
  (∃ x ∈ A, ∀ y ∈ A, y ≤ x) →  -- A has a maximum element
  (∃ x ∈ A, x = 83 ∧ ∀ y ∈ A, y ≤ x) :=  -- The upper bound (maximum) is 83
by sorry

end NUMINAMATH_CALUDE_upper_bound_of_prime_set_l2896_289606


namespace NUMINAMATH_CALUDE_chess_competition_games_l2896_289620

theorem chess_competition_games (W M : ℕ) 
  (h1 : W * (W - 1) / 2 = 45)
  (h2 : M * (M - 1) / 2 = 190) :
  W * M = 200 := by
  sorry

end NUMINAMATH_CALUDE_chess_competition_games_l2896_289620


namespace NUMINAMATH_CALUDE_gravel_path_cost_is_360_l2896_289619

/-- Calculates the cost of gravelling a path around a rectangular plot -/
def gravel_path_cost (plot_length plot_width path_width : ℝ) (cost_per_sqm : ℝ) : ℝ :=
  let outer_length := plot_length + 2 * path_width
  let outer_width := plot_width + 2 * path_width
  let path_area := outer_length * outer_width - plot_length * plot_width
  path_area * cost_per_sqm

/-- Theorem: The cost of gravelling the path is 360 rupees -/
theorem gravel_path_cost_is_360 :
  gravel_path_cost 110 65 2.5 0.4 = 360 := by
  sorry

end NUMINAMATH_CALUDE_gravel_path_cost_is_360_l2896_289619


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l2896_289612

-- Define a function to convert a number from any base to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base^i) 0

-- Define the given numbers in their respective bases
def num1 : List Nat := [5, 2, 3]  -- 325 in base 6 (least significant digit first)
def num2 : List Nat := [1, 3, 2]  -- 231 in base 5 (least significant digit first)

-- State the theorem
theorem base_conversion_subtraction :
  to_base_10 num1 6 - to_base_10 num2 5 = 59 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l2896_289612


namespace NUMINAMATH_CALUDE_count_integers_between_cubes_l2896_289663

theorem count_integers_between_cubes : 
  ∃ (n : ℕ), n = 37 ∧ 
  (∀ k : ℤ, (11.1 : ℝ)^3 < k ∧ k < (11.2 : ℝ)^3 ↔ 
   (⌊(11.1 : ℝ)^3⌋ + 1 : ℤ) ≤ k ∧ k ≤ (⌊(11.2 : ℝ)^3⌋ : ℤ)) ∧
  n = ⌊(11.2 : ℝ)^3⌋ - ⌊(11.1 : ℝ)^3⌋ :=
by sorry

end NUMINAMATH_CALUDE_count_integers_between_cubes_l2896_289663


namespace NUMINAMATH_CALUDE_product_digit_sum_l2896_289656

def number1 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707

def number2 : ℕ := 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909

def product : ℕ := number1 * number2

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

def units_digit (n : ℕ) : ℕ := n % 10

theorem product_digit_sum :
  hundreds_digit product + units_digit product = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_sum_l2896_289656


namespace NUMINAMATH_CALUDE_hershel_goldfish_count_l2896_289631

/-- The number of goldfish Hershel had initially -/
def initial_goldfish : ℕ := 15

theorem hershel_goldfish_count :
  let initial_betta : ℕ := 10
  let bexley_betta : ℕ := (2 * initial_betta) / 5
  let bexley_goldfish : ℕ := initial_goldfish / 3
  let total_fish : ℕ := initial_betta + bexley_betta + initial_goldfish + bexley_goldfish
  let remaining_fish : ℕ := total_fish / 2
  remaining_fish = 17 :=
by sorry

end NUMINAMATH_CALUDE_hershel_goldfish_count_l2896_289631


namespace NUMINAMATH_CALUDE_fractional_inequality_solution_set_l2896_289697

theorem fractional_inequality_solution_set (x : ℝ) :
  (x + 5) / (1 - x) ≤ 0 ↔ (x ≤ -5 ∨ x > 1) ∧ x ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_fractional_inequality_solution_set_l2896_289697


namespace NUMINAMATH_CALUDE_total_rain_time_l2896_289626

/-- Given rain durations over three days, prove the total rain time -/
theorem total_rain_time (first_day_start : Nat) (first_day_end : Nat)
  (h1 : first_day_end - first_day_start = 10)
  (h2 : ∃ second_day_duration : Nat, second_day_duration = (first_day_end - first_day_start) + 2)
  (h3 : ∃ third_day_duration : Nat, third_day_duration = 2 * (first_day_end - first_day_start + 2)) :
  ∃ total_duration : Nat, total_duration = 46 := by
  sorry

end NUMINAMATH_CALUDE_total_rain_time_l2896_289626


namespace NUMINAMATH_CALUDE_difference_of_squares_l2896_289640

theorem difference_of_squares : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2896_289640


namespace NUMINAMATH_CALUDE_a_positive_sufficient_not_necessary_for_abs_a_positive_l2896_289604

theorem a_positive_sufficient_not_necessary_for_abs_a_positive :
  (∃ a : ℝ, (a > 0 → abs a > 0) ∧ ¬(abs a > 0 → a > 0)) := by
  sorry

end NUMINAMATH_CALUDE_a_positive_sufficient_not_necessary_for_abs_a_positive_l2896_289604


namespace NUMINAMATH_CALUDE_smallest_number_with_property_l2896_289621

theorem smallest_number_with_property : ∃! n : ℕ, 
  n > 0 ∧
  n % 2 = 1 ∧
  n % 3 = 2 ∧
  n % 4 = 3 ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  n % 8 = 7 ∧
  n % 9 = 8 ∧
  n % 10 = 9 ∧
  (∀ m : ℕ, m > 0 ∧
    m % 2 = 1 ∧
    m % 3 = 2 ∧
    m % 4 = 3 ∧
    m % 5 = 4 ∧
    m % 6 = 5 ∧
    m % 7 = 6 ∧
    m % 8 = 7 ∧
    m % 9 = 8 ∧
    m % 10 = 9 → m ≥ n) ∧
  n = 2519 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_property_l2896_289621


namespace NUMINAMATH_CALUDE_expression_value_l2896_289622

theorem expression_value : (1/3 * 9 * 1/27 * 81 * 1/243 * 729)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2896_289622


namespace NUMINAMATH_CALUDE_monomial_exponents_sum_l2896_289642

/-- Two monomials are like terms if their variables have the same exponents -/
def are_like_terms (a b c d : ℕ) : Prop :=
  a = c ∧ b = d

theorem monomial_exponents_sum (m n : ℕ) : 
  are_like_terms 5 (2*n) m 4 → m + n = 7 := by
  sorry

end NUMINAMATH_CALUDE_monomial_exponents_sum_l2896_289642


namespace NUMINAMATH_CALUDE_all_pairs_successful_probability_expected_successful_pairs_gt_half_l2896_289685

-- Define the number of sock pairs
variable (n : ℕ)

-- Define a successful pair
def successful_pair (pair : ℕ × ℕ) : Prop := pair.1 = pair.2

-- Define the probability of all pairs being successful
def all_pairs_successful_prob : ℚ := (2^n * n.factorial) / (2*n).factorial

-- Define the expected number of successful pairs
def expected_successful_pairs : ℚ := n / (2*n - 1)

-- Theorem 1: Probability of all pairs being successful
theorem all_pairs_successful_probability :
  all_pairs_successful_prob n = (2^n * n.factorial) / (2*n).factorial :=
sorry

-- Theorem 2: Expected number of successful pairs is greater than 0.5
theorem expected_successful_pairs_gt_half :
  expected_successful_pairs n > 1/2 :=
sorry

end NUMINAMATH_CALUDE_all_pairs_successful_probability_expected_successful_pairs_gt_half_l2896_289685


namespace NUMINAMATH_CALUDE_combination_equality_l2896_289681

theorem combination_equality (n : ℕ) : 
  (Nat.choose (n + 1) 7 - Nat.choose n 7 = Nat.choose n 8) → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_l2896_289681


namespace NUMINAMATH_CALUDE_exp_greater_equal_linear_l2896_289688

theorem exp_greater_equal_linear : ∀ x : ℝ, Real.exp x ≥ Real.exp 1 * x := by sorry

end NUMINAMATH_CALUDE_exp_greater_equal_linear_l2896_289688


namespace NUMINAMATH_CALUDE_marble_ratio_l2896_289605

/-- Represents the number of marbles of each color in a box -/
structure MarbleBox where
  red : ℕ
  green : ℕ
  yellow : ℕ
  other : ℕ

/-- Conditions for the marble box problem -/
def MarbleBoxConditions (box : MarbleBox) : Prop :=
  box.red = 20 ∧
  box.yellow = box.green / 5 ∧
  box.red + box.green + box.yellow + box.other = 3 * box.green ∧
  box.other = 88

theorem marble_ratio (box : MarbleBox) 
  (h : MarbleBoxConditions box) : 
  box.green = 3 * box.red := by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_l2896_289605


namespace NUMINAMATH_CALUDE_exponential_inequality_l2896_289625

open Real

theorem exponential_inequality (f : ℝ → ℝ) (h : ∀ x, f x = exp x) :
  (∀ a, (∀ x, f x ≥ exp 1 * x + a) ↔ a ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l2896_289625


namespace NUMINAMATH_CALUDE_min_benches_for_equal_seating_l2896_289673

/-- Represents the seating capacity of a bench section -/
structure BenchCapacity where
  adults : Nat
  children : Nat

/-- Defines the standard bench capacity -/
def standard_bench : BenchCapacity := ⟨8, 12⟩

/-- Defines the extended bench capacity -/
def extended_bench : BenchCapacity := ⟨8, 16⟩

/-- Theorem stating the minimum number of benches required -/
theorem min_benches_for_equal_seating :
  ∃ (n : Nat), n > 0 ∧
    n * standard_bench.adults + n * extended_bench.adults =
    n * standard_bench.children + n * extended_bench.children ∧
    ∀ (m : Nat), m > 0 →
      m * standard_bench.adults + m * extended_bench.adults =
      m * standard_bench.children + m * extended_bench.children →
      m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_min_benches_for_equal_seating_l2896_289673


namespace NUMINAMATH_CALUDE_allison_craft_items_l2896_289655

/-- Represents the number of craft items bought by a person -/
structure CraftItems where
  glueSticks : ℕ
  constructionPaper : ℕ

/-- Calculates the total number of craft items -/
def totalItems (items : CraftItems) : ℕ :=
  items.glueSticks + items.constructionPaper

theorem allison_craft_items (marie : CraftItems) 
    (marie_glue : marie.glueSticks = 15)
    (marie_paper : marie.constructionPaper = 30)
    (allison : CraftItems)
    (glue_diff : allison.glueSticks = marie.glueSticks + 8)
    (paper_ratio : marie.constructionPaper = 6 * allison.constructionPaper) :
    totalItems allison = 28 := by
  sorry

end NUMINAMATH_CALUDE_allison_craft_items_l2896_289655


namespace NUMINAMATH_CALUDE_product_cost_reduction_l2896_289633

theorem product_cost_reduction (original_selling_price : ℝ) 
  (original_profit_rate : ℝ) (new_profit_rate : ℝ) (additional_profit : ℝ) :
  original_selling_price = 659.9999999999994 →
  original_profit_rate = 0.1 →
  new_profit_rate = 0.3 →
  additional_profit = 42 →
  let original_cost := original_selling_price / (1 + original_profit_rate)
  let new_cost := (original_selling_price + additional_profit) / (1 + new_profit_rate)
  (original_cost - new_cost) / original_cost = 0.1 := by
sorry

end NUMINAMATH_CALUDE_product_cost_reduction_l2896_289633


namespace NUMINAMATH_CALUDE_tuesday_is_valid_start_day_l2896_289602

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def advanceDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDays d m)

def isValidRedemptionSchedule (startDay : DayOfWeek) : Prop :=
  ∀ i : Fin 7, advanceDays startDay (i.val * 12) ≠ DayOfWeek.Saturday

theorem tuesday_is_valid_start_day :
  isValidRedemptionSchedule DayOfWeek.Tuesday ∧
  ∀ d : DayOfWeek, d ≠ DayOfWeek.Tuesday → ¬ isValidRedemptionSchedule d :=
sorry

end NUMINAMATH_CALUDE_tuesday_is_valid_start_day_l2896_289602


namespace NUMINAMATH_CALUDE_car_repair_cost_proof_l2896_289628

/-- Calculates the total cost for a car repair given the hourly rate, hours worked per day,
    number of days worked, and cost of parts. -/
def total_repair_cost (hourly_rate : ℕ) (hours_per_day : ℕ) (days_worked : ℕ) (parts_cost : ℕ) : ℕ :=
  hourly_rate * hours_per_day * days_worked + parts_cost

/-- Proves that given the specified conditions, the total cost for the car's owner is $9220. -/
theorem car_repair_cost_proof :
  total_repair_cost 60 8 14 2500 = 9220 := by
  sorry

end NUMINAMATH_CALUDE_car_repair_cost_proof_l2896_289628


namespace NUMINAMATH_CALUDE_amoeba_problem_l2896_289653

/-- The number of amoebas after n days, given an initial population and split factor --/
def amoeba_population (initial : ℕ) (split_factor : ℕ) (days : ℕ) : ℕ :=
  initial * split_factor ^ days

/-- Theorem: Given 2 initial amoebas that split into 3 each day, after 5 days there will be 486 amoebas --/
theorem amoeba_problem :
  amoeba_population 2 3 5 = 486 := by
  sorry

#eval amoeba_population 2 3 5

end NUMINAMATH_CALUDE_amoeba_problem_l2896_289653


namespace NUMINAMATH_CALUDE_domain_of_f_l2896_289670

-- Define the function f
def f (x : ℝ) : ℝ := (x^2 - 2*x + 1)^(1/3) + (9 - x^2)^(1/3)

-- State the theorem about the domain of f
theorem domain_of_f :
  ∀ x : ℝ, ∃ y : ℝ, f x = y :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_l2896_289670


namespace NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_unit_cube_l2896_289639

/-- The surface area of a sphere that circumscribes a cube with edge length 1 is 3π. -/
theorem sphere_surface_area_circumscribing_unit_cube (π : ℝ) : 
  (∃ (S : ℝ), S = 3 * π ∧ 
    S = 4 * π * (((1 : ℝ)^2 + (1 : ℝ)^2 + (1 : ℝ)^2).sqrt / 2)^2) :=
by sorry


end NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_unit_cube_l2896_289639


namespace NUMINAMATH_CALUDE_emma_room_coverage_l2896_289682

/-- Represents the dimensions of Emma's room --/
structure RoomDimensions where
  rectangleLength : ℝ
  rectangleWidth : ℝ
  triangleBase : ℝ
  triangleHeight : ℝ

/-- Represents the tiles used to cover the room --/
structure Tiles where
  squareTiles : ℕ
  triangularTiles : ℕ
  squareTileArea : ℝ
  triangularTileBase : ℝ
  triangularTileHeight : ℝ

/-- Calculates the fraction of the room covered by tiles --/
def fractionalCoverage (room : RoomDimensions) (tiles : Tiles) : ℚ :=
  sorry

/-- Theorem stating that the fractional coverage of Emma's room is 3/20 --/
theorem emma_room_coverage :
  let room : RoomDimensions := {
    rectangleLength := 12,
    rectangleWidth := 20,
    triangleBase := 10,
    triangleHeight := 8
  }
  let tiles : Tiles := {
    squareTiles := 40,
    triangularTiles := 4,
    squareTileArea := 1,
    triangularTileBase := 1,
    triangularTileHeight := 1
  }
  fractionalCoverage room tiles = 3/20 := by
  sorry

end NUMINAMATH_CALUDE_emma_room_coverage_l2896_289682


namespace NUMINAMATH_CALUDE_probability_point_between_F_and_G_l2896_289664

/-- Given a line segment AB with points A, E, F, G, B placed consecutively,
    where AB = 4AE and AB = 8BF, the probability that a randomly selected
    point on AB lies between F and G is 1/2. -/
theorem probability_point_between_F_and_G (AB AE BF FG : ℝ) : 
  AB > 0 → AB = 4 * AE → AB = 8 * BF → FG = AB / 2 → FG / AB = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_point_between_F_and_G_l2896_289664


namespace NUMINAMATH_CALUDE_expression_simplification_l2896_289624

theorem expression_simplification (x y : ℚ) (hx : x = -2) (hy : y = -1) :
  (2 * (x - 2*y) * (2*x + y) - (x + 2*y)^2 + x * (8*y - 3*x)) / (6*y) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2896_289624


namespace NUMINAMATH_CALUDE_prime_equation_solution_l2896_289636

/-- Given that x and y are prime numbers, prove that x^y - y^x = xy^2 - 19 if and only if (x, y) = (2, 3) or (x, y) = (2, 7) -/
theorem prime_equation_solution (x y : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y) :
  x^y - y^x = x*y^2 - 19 ↔ (x = 2 ∧ y = 3) ∨ (x = 2 ∧ y = 7) := by
  sorry

end NUMINAMATH_CALUDE_prime_equation_solution_l2896_289636


namespace NUMINAMATH_CALUDE_john_works_fifty_weeks_l2896_289654

/-- Represents the number of weeks John works in a year -/
def weeks_worked (patients_hospital1 : ℕ) (patients_hospital2_increase : ℚ) 
  (days_per_week : ℕ) (total_patients_per_year : ℕ) : ℚ :=
  let patients_hospital2 := patients_hospital1 * (1 + patients_hospital2_increase)
  let patients_per_week := (patients_hospital1 + patients_hospital2) * days_per_week
  total_patients_per_year / patients_per_week

/-- Theorem stating that John works 50 weeks a year given the problem conditions -/
theorem john_works_fifty_weeks :
  weeks_worked 20 (1/5 : ℚ) 5 11000 = 50 := by sorry

end NUMINAMATH_CALUDE_john_works_fifty_weeks_l2896_289654


namespace NUMINAMATH_CALUDE_carries_payment_l2896_289611

def clothes_shopping (shirt_quantity : ℕ) (pants_quantity : ℕ) (jacket_quantity : ℕ)
                     (shirt_price : ℕ) (pants_price : ℕ) (jacket_price : ℕ) : ℕ :=
  let total_cost := shirt_quantity * shirt_price + pants_quantity * pants_price + jacket_quantity * jacket_price
  total_cost / 2

theorem carries_payment :
  clothes_shopping 4 2 2 8 18 60 = 94 :=
by
  sorry

end NUMINAMATH_CALUDE_carries_payment_l2896_289611


namespace NUMINAMATH_CALUDE_water_mixture_percentage_l2896_289630

theorem water_mixture_percentage 
  (initial_mixture : ℝ) 
  (water_added : ℝ) 
  (final_percentage : ℝ) :
  initial_mixture = 20 →
  water_added = 4 →
  final_percentage = 25 →
  (initial_mixture * (initial_percentage / 100) + water_added) / (initial_mixture + water_added) * 100 = final_percentage →
  initial_percentage = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_water_mixture_percentage_l2896_289630


namespace NUMINAMATH_CALUDE_probability_at_least_one_female_l2896_289694

def total_students : ℕ := 5
def male_students : ℕ := 3
def female_students : ℕ := 2
def students_to_select : ℕ := 2

theorem probability_at_least_one_female :
  (1 : ℚ) - (Nat.choose male_students students_to_select : ℚ) / (Nat.choose total_students students_to_select : ℚ) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_female_l2896_289694


namespace NUMINAMATH_CALUDE_square_fraction_count_l2896_289617

theorem square_fraction_count : 
  ∃! (S : Finset ℤ), 
    (∀ n ∈ S, 0 ≤ n ∧ n ≤ 23 ∧ ∃ k : ℤ, (n : ℚ) / (24 - n) = k^2) ∧ 
    Finset.card S = 2 :=
by sorry

end NUMINAMATH_CALUDE_square_fraction_count_l2896_289617


namespace NUMINAMATH_CALUDE_max_obtuse_angles_in_quadrilateral_with_120_degree_angle_l2896_289659

/-- A quadrilateral with one angle of 120 degrees can have at most 3 obtuse angles. -/
theorem max_obtuse_angles_in_quadrilateral_with_120_degree_angle :
  ∀ (a b c d : ℝ),
  a = 120 →
  a + b + c + d = 360 →
  a > 90 ∧ b > 90 ∧ c > 90 ∧ d > 90 →
  False :=
by
  sorry

end NUMINAMATH_CALUDE_max_obtuse_angles_in_quadrilateral_with_120_degree_angle_l2896_289659


namespace NUMINAMATH_CALUDE_sin_pi_over_six_l2896_289660

theorem sin_pi_over_six : Real.sin (π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_over_six_l2896_289660


namespace NUMINAMATH_CALUDE_number_of_dimes_l2896_289651

/-- Given a total of 11 coins, including 2 nickels and 7 quarters, prove that the number of dimes is 2 -/
theorem number_of_dimes (total : ℕ) (nickels : ℕ) (quarters : ℕ) (h1 : total = 11) (h2 : nickels = 2) (h3 : quarters = 7) :
  total - nickels - quarters = 2 := by
sorry

end NUMINAMATH_CALUDE_number_of_dimes_l2896_289651


namespace NUMINAMATH_CALUDE_even_heads_probability_l2896_289696

def coin_flips : ℕ := 8

theorem even_heads_probability : 
  (Finset.filter (fun n => Even n) (Finset.range (coin_flips + 1))).card / 2^coin_flips = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_even_heads_probability_l2896_289696


namespace NUMINAMATH_CALUDE_bird_nest_difference_l2896_289603

def number_of_birds : ℕ := 6
def number_of_nests : ℕ := 3

theorem bird_nest_difference :
  number_of_birds - number_of_nests = 3 := by
  sorry

end NUMINAMATH_CALUDE_bird_nest_difference_l2896_289603


namespace NUMINAMATH_CALUDE_stating_min_gloves_for_matching_pair_l2896_289692

/-- Represents the number of different glove patterns -/
def num_patterns : ℕ := 4

/-- Represents the number of pairs for each pattern -/
def pairs_per_pattern : ℕ := 3

/-- Represents the total number of gloves in the wardrobe -/
def total_gloves : ℕ := num_patterns * pairs_per_pattern * 2

/-- 
Theorem stating the minimum number of gloves needed to ensure a matching pair
-/
theorem min_gloves_for_matching_pair : 
  ∃ (n : ℕ), n = num_patterns * pairs_per_pattern + 1 ∧ 
  (∀ (m : ℕ), m < n → ∃ (pattern : Fin num_patterns), 
    (m.choose 2 : ℕ) < pairs_per_pattern) ∧
  n ≤ total_gloves := by
  sorry

end NUMINAMATH_CALUDE_stating_min_gloves_for_matching_pair_l2896_289692


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l2896_289627

theorem opposite_of_negative_two :
  ∀ x : ℤ, x + (-2) = 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l2896_289627


namespace NUMINAMATH_CALUDE_cyclists_speed_problem_l2896_289632

theorem cyclists_speed_problem (total_distance : ℝ) (speed_difference : ℝ) :
  total_distance = 270 →
  speed_difference = 1.5 →
  ∃ (speed1 speed2 time : ℝ),
    speed1 > 0 ∧
    speed2 > 0 ∧
    speed1 = speed2 + speed_difference ∧
    time = speed1 ∧
    speed1 * time + speed2 * time = total_distance ∧
    speed1 = 12 ∧
    speed2 = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_cyclists_speed_problem_l2896_289632


namespace NUMINAMATH_CALUDE_sum_of_roots_l2896_289652

theorem sum_of_roots (c d : ℝ) 
  (hc : c^3 - 12*c^2 + 15*c - 36 = 0) 
  (hd : 6*d^3 - 36*d^2 - 150*d + 1350 = 0) : 
  c + d = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2896_289652


namespace NUMINAMATH_CALUDE_circle_delta_area_l2896_289661

/-- Circle δ with points A and B -/
structure Circle_delta where
  center : ℝ × ℝ
  radius : ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- Conditions for the circle δ -/
def circle_conditions (δ : Circle_delta) : Prop :=
  δ.A = (2, 9) ∧ 
  δ.B = (10, 5) ∧
  (δ.A.1 - δ.center.1)^2 + (δ.A.2 - δ.center.2)^2 = δ.radius^2 ∧
  (δ.B.1 - δ.center.1)^2 + (δ.B.2 - δ.center.2)^2 = δ.radius^2

/-- Tangent lines intersection condition -/
def tangent_intersection (δ : Circle_delta) : Prop :=
  ∃ x : ℝ, 
    let slope_AB := (δ.B.2 - δ.A.2) / (δ.B.1 - δ.A.1)
    let perp_slope := -1 / slope_AB
    let midpoint := ((δ.A.1 + δ.B.1) / 2, (δ.A.2 + δ.B.2) / 2)
    perp_slope * (x - midpoint.1) + midpoint.2 = 0

/-- Theorem stating the area of circle δ -/
theorem circle_delta_area (δ : Circle_delta) 
  (h1 : circle_conditions δ) (h2 : tangent_intersection δ) : 
  π * δ.radius^2 = 83.44 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_delta_area_l2896_289661


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2896_289648

theorem quadratic_equations_solutions :
  let eq1 : ℝ → Prop := λ x ↦ 2 * x^2 + 4 * x + 1 = 0
  let eq2 : ℝ → Prop := λ x ↦ x^2 + 6 * x = 5
  let sol1_1 : ℝ := -1 + Real.sqrt 2 / 2
  let sol1_2 : ℝ := -1 - Real.sqrt 2 / 2
  let sol2_1 : ℝ := -3 + Real.sqrt 14
  let sol2_2 : ℝ := -3 - Real.sqrt 14
  (eq1 sol1_1 ∧ eq1 sol1_2) ∧ (eq2 sol2_1 ∧ eq2 sol2_2) := by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2896_289648


namespace NUMINAMATH_CALUDE_athlete_C_is_best_l2896_289658

structure Athlete where
  name : String
  average_score : ℝ
  variance : ℝ

def athletes : List Athlete := [
  ⟨"A", 7, 0.9⟩,
  ⟨"B", 8, 1.1⟩,
  ⟨"C", 8, 0.9⟩,
  ⟨"D", 7, 1.0⟩
]

def has_best_performance_and_stability (a : Athlete) (athletes : List Athlete) : Prop :=
  ∀ b ∈ athletes, 
    (a.average_score > b.average_score) ∨ 
    (a.average_score = b.average_score ∧ a.variance ≤ b.variance)

theorem athlete_C_is_best : 
  ∃ a ∈ athletes, a.name = "C" ∧ has_best_performance_and_stability a athletes := by
  sorry

end NUMINAMATH_CALUDE_athlete_C_is_best_l2896_289658


namespace NUMINAMATH_CALUDE_candy_making_time_l2896_289683

/-- Candy-making process time calculation -/
theorem candy_making_time
  (initial_temp : ℝ)
  (target_temp : ℝ)
  (final_temp : ℝ)
  (heating_rate : ℝ)
  (cooling_rate : ℝ)
  (h1 : initial_temp = 60)
  (h2 : target_temp = 240)
  (h3 : final_temp = 170)
  (h4 : heating_rate = 5)
  (h5 : cooling_rate = 7) :
  (target_temp - initial_temp) / heating_rate + (target_temp - final_temp) / cooling_rate = 46 :=
by sorry

end NUMINAMATH_CALUDE_candy_making_time_l2896_289683


namespace NUMINAMATH_CALUDE_quadratic_root_m_value_l2896_289676

theorem quadratic_root_m_value :
  ∀ m : ℝ, (2 : ℝ)^2 + m * 2 + 2 = 0 → m = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_m_value_l2896_289676


namespace NUMINAMATH_CALUDE_set_operations_l2896_289645

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 12 ≤ 0}
def B : Set ℝ := {x | x^2 - 4*x - 5 > 0}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Theorem statement
theorem set_operations :
  (A ∩ B = {x | -3 ≤ x ∧ x < -1}) ∧
  (A ∪ B = {x | x ≤ 4 ∨ x > 5}) ∧
  ((Set.compl A) ∩ (Set.compl B) = {x | 4 < x ∧ x ≤ 5}) :=
sorry

end NUMINAMATH_CALUDE_set_operations_l2896_289645


namespace NUMINAMATH_CALUDE_circle_center_x_coordinate_l2896_289672

theorem circle_center_x_coordinate (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - a*x = 0 → (x - 1)^2 + y^2 = 1) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_x_coordinate_l2896_289672


namespace NUMINAMATH_CALUDE_ceiling_bounds_range_of_x_solutions_of_equation_l2896_289693

-- Define the ceiling function for rational numbers
noncomputable def ceiling (a : ℚ) : ℤ :=
  Int.natAbs (Int.floor a + 1) - 1

-- State the theorem
theorem ceiling_bounds (m : ℚ) : m ≤ ↑(ceiling m) ∧ ↑(ceiling m) < m + 1 := by
  sorry

-- Define the range of x when {3x+2} = 8
theorem range_of_x (x : ℚ) : ceiling (3 * x + 2) = 8 → 5/3 < x ∧ x ≤ 2 := by
  sorry

-- Define the solutions for {3x-2} = 2x + 1/2
theorem solutions_of_equation (x : ℚ) : 
  ceiling (3 * x - 2) = ↑⌊2 * x + 1/2⌋ → x = 7/4 ∨ x = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_bounds_range_of_x_solutions_of_equation_l2896_289693


namespace NUMINAMATH_CALUDE_woodburning_cost_l2896_289678

def woodburning_problem (num_sold : ℕ) (price_per_item : ℚ) (profit : ℚ) : Prop :=
  let total_revenue := num_sold * price_per_item
  let cost_of_wood := total_revenue - profit
  cost_of_wood = 100

theorem woodburning_cost :
  woodburning_problem 20 15 200 := by
  sorry

end NUMINAMATH_CALUDE_woodburning_cost_l2896_289678


namespace NUMINAMATH_CALUDE_bob_always_has_valid_move_l2896_289689

-- Define the game board
def GameBoard (n : ℕ) := ℤ × ℤ

-- Define the possible moves for Bob and Alice
def BobMove (p : ℤ × ℤ) : Set (ℤ × ℤ) :=
  {(p.1 + 2, p.2 + 1), (p.1 + 2, p.2 - 1), (p.1 - 2, p.2 + 1), (p.1 - 2, p.2 - 1)}

def AliceMove (p : ℤ × ℤ) : Set (ℤ × ℤ) :=
  {(p.1 + 1, p.2 + 2), (p.1 + 1, p.2 - 2), (p.1 - 1, p.2 + 2), (p.1 - 1, p.2 - 2)}

-- Define the modulo condition
def ModuloCondition (n : ℕ) (a b c d : ℤ) : Prop :=
  c % n = a % n ∧ d % n = b % n

-- Define a valid move
def ValidMove (n : ℕ) (occupied : Set (ℤ × ℤ)) (p : ℤ × ℤ) : Prop :=
  ∀ (a b : ℤ), (a, b) ∈ occupied → ¬(ModuloCondition n a b p.1 p.2)

-- Theorem: Bob always has a valid move
theorem bob_always_has_valid_move (n : ℕ) (h : n = 2018 ∨ n = 2019) 
  (occupied : Set (ℤ × ℤ)) (last_move : ℤ × ℤ) :
  ∃ (next_move : ℤ × ℤ), next_move ∈ BobMove last_move ∧ ValidMove n occupied next_move :=
sorry

end NUMINAMATH_CALUDE_bob_always_has_valid_move_l2896_289689


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l2896_289623

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1 / b ≥ 2) ∨ (b + 1 / c ≥ 2) ∨ (c + 1 / a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l2896_289623


namespace NUMINAMATH_CALUDE_expression_simplification_l2896_289657

theorem expression_simplification (x : ℝ) (h : x = 8) :
  (2 * x) / (x + 1) - ((2 * x + 4) / (x^2 - 1)) / ((x + 2) / (x^2 - 2*x + 1)) = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2896_289657


namespace NUMINAMATH_CALUDE_f_negative_a_eq_zero_l2896_289637

noncomputable def f (x : ℝ) : ℝ := x^3 * (Real.exp x + Real.exp (-x)) + 2

theorem f_negative_a_eq_zero (a : ℝ) (h : f a = 4) : f (-a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_a_eq_zero_l2896_289637


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_fourth_l2896_289669

theorem arctan_sum_equals_pi_fourth (m : ℕ+) : 
  (Real.arctan (1/7) + Real.arctan (1/8) + Real.arctan (1/9) + Real.arctan (1/m.val : ℝ) = π/4) → m = 133 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_fourth_l2896_289669


namespace NUMINAMATH_CALUDE_parking_methods_count_l2896_289641

/-- Represents the number of parking spaces --/
def parking_spaces : ℕ := 6

/-- Represents the number of cars to be parked --/
def cars : ℕ := 3

/-- Calculates the number of available slots for parking --/
def available_slots : ℕ := parking_spaces - cars + 1

/-- Calculates the number of ways to park cars --/
def parking_methods : ℕ := available_slots * (available_slots - 1) * (available_slots - 2)

/-- Theorem stating that the number of parking methods is 24 --/
theorem parking_methods_count : parking_methods = 24 := by sorry

end NUMINAMATH_CALUDE_parking_methods_count_l2896_289641


namespace NUMINAMATH_CALUDE_compressed_music_space_l2896_289671

-- Define the parameters of the problem
def total_days : ℕ := 20
def total_space : ℕ := 25000
def compression_rate : ℚ := 1/10

-- Define the function to calculate the average space per hour
def avg_space_per_hour (days : ℕ) (space : ℕ) (rate : ℚ) : ℚ :=
  let total_hours : ℕ := days * 24
  let compressed_space : ℚ := space * (1 - rate)
  compressed_space / total_hours

-- Theorem statement
theorem compressed_music_space :
  round (avg_space_per_hour total_days total_space compression_rate) = 47 := by
  sorry

end NUMINAMATH_CALUDE_compressed_music_space_l2896_289671


namespace NUMINAMATH_CALUDE_arcsin_plus_arcsin_2x_eq_arccos_l2896_289610

theorem arcsin_plus_arcsin_2x_eq_arccos (x : ℝ) : 
  (Real.arcsin x + Real.arcsin (2*x) = Real.arccos x) ↔ 
  (x = 0 ∨ x = 2/Real.sqrt 5 ∨ x = -2/Real.sqrt 5) := by
sorry

end NUMINAMATH_CALUDE_arcsin_plus_arcsin_2x_eq_arccos_l2896_289610


namespace NUMINAMATH_CALUDE_complex_magnitude_l2896_289643

theorem complex_magnitude (z : ℂ) (h : z * (1 - 2*I) = 3 + 4*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2896_289643


namespace NUMINAMATH_CALUDE_sum_base6_series_l2896_289662

/-- Converts a number from base 6 to base 10 -/
def base6To10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 6 -/
def base10To6 (n : ℕ) : ℕ := sorry

/-- Sum of arithmetic series in base 10 -/
def sumArithmeticSeries (a l n : ℕ) : ℕ := n * (a + l) / 2

theorem sum_base6_series :
  let first := base6To10 3
  let last := base6To10 100
  let n := last - first + 1
  base10To6 (sumArithmeticSeries first last n) = 3023 :=
by sorry

end NUMINAMATH_CALUDE_sum_base6_series_l2896_289662


namespace NUMINAMATH_CALUDE_distance_between_points_l2896_289666

theorem distance_between_points (a : ℝ) : 
  let A : ℝ × ℝ := (a, -2)
  let B : ℝ × ℝ := (0, 3)
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 7^2) → (a = 2 * Real.sqrt 6 ∨ a = -2 * Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_l2896_289666


namespace NUMINAMATH_CALUDE_percentage_of_sikh_boys_l2896_289629

theorem percentage_of_sikh_boys 
  (total_boys : ℕ) 
  (muslim_percentage : ℚ) 
  (hindu_percentage : ℚ) 
  (other_boys : ℕ) 
  (h1 : total_boys = 300)
  (h2 : muslim_percentage = 44 / 100)
  (h3 : hindu_percentage = 28 / 100)
  (h4 : other_boys = 54) :
  (total_boys - (muslim_percentage * total_boys + hindu_percentage * total_boys + other_boys)) / total_boys = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_sikh_boys_l2896_289629


namespace NUMINAMATH_CALUDE_hajis_mother_sales_l2896_289665

/-- Haji's mother's sales problem -/
theorem hajis_mother_sales (tough_week_sales : ℕ) (total_sales : ℕ) :
  tough_week_sales = 800 →
  total_sales = 10400 →
  ∃ (good_weeks : ℕ),
    good_weeks * (2 * tough_week_sales) + 3 * tough_week_sales = total_sales ∧
    good_weeks = 5 :=
by sorry

end NUMINAMATH_CALUDE_hajis_mother_sales_l2896_289665


namespace NUMINAMATH_CALUDE_matrix_commute_result_l2896_289618

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 3, 4]

def B (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]

theorem matrix_commute_result (a b c d : ℝ) (h1 : A * B a b c d = B a b c d * A) 
  (h2 : 4 * b ≠ c) : (a - d) / (c - 4 * b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_commute_result_l2896_289618


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l2896_289680

/-- The surface area of a sphere circumscribing a right square prism -/
theorem circumscribed_sphere_surface_area (a h : ℝ) (ha : a = 2) (hh : h = 3) :
  let R := (1 / 2 : ℝ) * Real.sqrt (h^2 + 2 * a^2)
  4 * Real.pi * R^2 = 17 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l2896_289680


namespace NUMINAMATH_CALUDE_triangle_side_length_l2896_289668

theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_area : (1/2) * a * c * Real.sin B = Real.sqrt 3)
  (h_angle : B = Real.pi / 3)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2896_289668


namespace NUMINAMATH_CALUDE_least_four_digit_number_with_conditions_l2896_289650

/-- A function that checks if a number has all different digits -/
def hasDifferentDigits (n : ℕ) : Prop := sorry

/-- A function that checks if a number includes the digit 5 -/
def includesFive (n : ℕ) : Prop := sorry

/-- A function that checks if a number is divisible by all of its digits -/
def divisibleByAllDigits (n : ℕ) : Prop := sorry

theorem least_four_digit_number_with_conditions :
  ∀ n : ℕ,
    1000 ≤ n ∧ n < 10000 ∧
    hasDifferentDigits n ∧
    includesFive n ∧
    divisibleByAllDigits n →
    1536 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_number_with_conditions_l2896_289650


namespace NUMINAMATH_CALUDE_determinant_property_l2896_289638

theorem determinant_property (a b c d : ℝ) :
  Matrix.det ![![a, b], ![c, d]] = 4 →
  Matrix.det ![![a + 2*c, b + 2*d], ![c, d]] = 4 := by
  sorry

end NUMINAMATH_CALUDE_determinant_property_l2896_289638


namespace NUMINAMATH_CALUDE_two_digit_divisor_with_remainder_l2896_289691

theorem two_digit_divisor_with_remainder (x y : ℕ) : ∃! n : ℕ, 
  (0 < x ∧ x ≤ 9) ∧ 
  (0 ≤ y ∧ y ≤ 9) ∧
  (n = 10 * x + y) ∧
  (∃ q : ℕ, 491 = n * q + 59) ∧
  (n = 72) := by
sorry

end NUMINAMATH_CALUDE_two_digit_divisor_with_remainder_l2896_289691


namespace NUMINAMATH_CALUDE_fifteenth_prime_l2896_289686

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def nth_prime (n : ℕ) : ℕ := sorry

theorem fifteenth_prime :
  (nth_prime 6 = 13) → (nth_prime 15 = 47) :=
sorry

end NUMINAMATH_CALUDE_fifteenth_prime_l2896_289686


namespace NUMINAMATH_CALUDE_max_large_chips_l2896_289667

theorem max_large_chips (total : ℕ) (small large : ℕ) (h1 : total = 100) 
  (h2 : total = small + large) (h3 : ∃ p : ℕ, Prime p ∧ Even p ∧ small = large + p) : 
  large ≤ 49 := by
  sorry

end NUMINAMATH_CALUDE_max_large_chips_l2896_289667


namespace NUMINAMATH_CALUDE_forgotten_digit_probability_l2896_289634

theorem forgotten_digit_probability : 
  let total_digits : ℕ := 10
  let max_attempts : ℕ := 2
  let favorable_outcomes : ℕ := (total_digits - 1) + (total_digits - 1)
  let total_outcomes : ℕ := total_digits * (total_digits - 1)
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_forgotten_digit_probability_l2896_289634


namespace NUMINAMATH_CALUDE_thirteen_pow_2023_mod_1000_l2896_289613

theorem thirteen_pow_2023_mod_1000 : 13^2023 % 1000 = 99 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_pow_2023_mod_1000_l2896_289613


namespace NUMINAMATH_CALUDE_intersection_point_k_value_l2896_289684

theorem intersection_point_k_value (x y k : ℝ) : 
  x = -6.3 →
  3 * x + y = k →
  -0.75 * x + y = 25 →
  k = 1.375 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_k_value_l2896_289684


namespace NUMINAMATH_CALUDE_degenerate_ellipse_max_y_coordinate_l2896_289687

theorem degenerate_ellipse_max_y_coordinate :
  ∀ x y : ℝ, (x^2 / 49 + (y - 3)^2 / 25 = 0) → y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_degenerate_ellipse_max_y_coordinate_l2896_289687


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l2896_289690

theorem ceiling_floor_difference : 
  ⌈(15 : ℝ) / 8 * (-34 : ℝ) / 4⌉ - ⌊(15 : ℝ) / 8 * ⌊(-34 : ℝ) / 4⌋⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l2896_289690


namespace NUMINAMATH_CALUDE_monthly_income_calculation_l2896_289608

theorem monthly_income_calculation (income : ℝ) : 
  (income / 2 - 20 = 100) → income = 240 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_calculation_l2896_289608


namespace NUMINAMATH_CALUDE_erics_score_l2896_289675

theorem erics_score (total_students : ℕ) (students_before : ℕ) (avg_before : ℚ) (avg_after : ℚ) :
  total_students = 22 →
  students_before = 21 →
  avg_before = 84 →
  avg_after = 85 →
  (students_before * avg_before + (total_students - students_before) * 106) / total_students = avg_after :=
by sorry

end NUMINAMATH_CALUDE_erics_score_l2896_289675


namespace NUMINAMATH_CALUDE_ellipse_minimum_value_l2896_289698

theorem ellipse_minimum_value (x y : ℝ) :
  x > 0 → y > 0 → x^2 / 16 + y^2 / 12 = 1 →
  x / (4 - x) + 3 * y / (6 - y) ≥ 4 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀^2 / 16 + y₀^2 / 12 = 1 ∧
    x₀ / (4 - x₀) + 3 * y₀ / (6 - y₀) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_minimum_value_l2896_289698


namespace NUMINAMATH_CALUDE_max_min_values_l2896_289615

theorem max_min_values (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 5 * y = 20) :
  (∃ (u : ℝ), u = Real.log x / Real.log 10 + Real.log y / Real.log 10 ∧
    u ≤ 1 ∧
    ∀ (v : ℝ), v = Real.log x / Real.log 10 + Real.log y / Real.log 10 → v ≤ u) ∧
  (∃ (w : ℝ), w = 1 / x + 1 / y ∧
    w ≥ (7 + 2 * Real.sqrt 10) / 20 ∧
    ∀ (z : ℝ), z = 1 / x + 1 / y → z ≥ w) := by
  sorry

end NUMINAMATH_CALUDE_max_min_values_l2896_289615


namespace NUMINAMATH_CALUDE_function_equation_implies_zero_l2896_289647

/-- A function satisfying f(x + |y|) = f(|x|) + f(y) for all real x and y is identically zero. -/
theorem function_equation_implies_zero (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, f (x + |y|) = f (|x|) + f y) : 
    ∀ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_equation_implies_zero_l2896_289647


namespace NUMINAMATH_CALUDE_complex_expressions_equality_l2896_289601

theorem complex_expressions_equality : 
  let z₁ : ℂ := (-2 * Real.sqrt 3 * I + 1) / (1 + 2 * Real.sqrt 3 * I) + ((Real.sqrt 2) / (1 + I)) ^ 2000 + (1 + I) / (3 - I)
  let z₂ : ℂ := (5 * (4 + I)^2) / (I * (2 + I)) + 2 / (1 - I)^2
  z₁ = 6/65 + (39/65) * I ∧ z₂ = -1 + 39 * I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_expressions_equality_l2896_289601


namespace NUMINAMATH_CALUDE_defective_probability_l2896_289649

/-- The probability of a randomly chosen unit being defective in a factory with two machines --/
theorem defective_probability (total_output : ℝ) (machine_a_output : ℝ) (machine_b_output : ℝ)
  (machine_a_defective_rate : ℝ) (machine_b_defective_rate : ℝ) :
  machine_a_output = 0.4 * total_output →
  machine_b_output = 0.6 * total_output →
  machine_a_defective_rate = 9 / 1000 →
  machine_b_defective_rate = 1 / 50 →
  (machine_a_output / total_output) * machine_a_defective_rate +
  (machine_b_output / total_output) * machine_b_defective_rate = 0.0156 := by
  sorry


end NUMINAMATH_CALUDE_defective_probability_l2896_289649


namespace NUMINAMATH_CALUDE_disputed_food_weight_l2896_289674

/-- 
Given a piece of food disputed by a dog and a cat:
- x is the total weight of the piece
- d is the difference in the amount the dog wants to take compared to the cat
- The cat takes (x - d) grams
- The dog takes (x + d) grams
- We know that (x - d) = 300 and (x + d) = 500

This theorem proves that the total weight of the disputed piece is 400 grams.
-/
theorem disputed_food_weight (x d : ℝ) 
  (h1 : x - d = 300) 
  (h2 : x + d = 500) : 
  x = 400 := by
sorry


end NUMINAMATH_CALUDE_disputed_food_weight_l2896_289674


namespace NUMINAMATH_CALUDE_cone_volume_approximation_l2896_289609

theorem cone_volume_approximation (L h : ℝ) (h1 : L > 0) (h2 : h > 0) :
  (7 / 264 : ℝ) * L^2 * h = (1 / 3 : ℝ) * ((22 / 7 : ℝ) / 4) * L^2 * h := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_approximation_l2896_289609


namespace NUMINAMATH_CALUDE_correct_meeting_turns_l2896_289644

/-- The number of points on the circle. -/
def n : ℕ := 12

/-- Alice's clockwise movement per turn. -/
def a : ℕ := 7

/-- Bob's counterclockwise movement per turn. -/
def b : ℕ := 4

/-- The function to find the smallest positive integer k such that Alice and Bob meet. -/
def meetingTurns : ℕ := sorry

theorem correct_meeting_turns : meetingTurns = 12 := by sorry

end NUMINAMATH_CALUDE_correct_meeting_turns_l2896_289644


namespace NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l2896_289614

/-- Given a cube with space diagonal 5√3, prove its volume is 125 -/
theorem cube_volume_from_space_diagonal :
  ∀ s : ℝ,
  s > 0 →
  (s * s * s = 5 * 5 * 5) →
  (s * s + s * s + s * s = (5 * Real.sqrt 3) * (5 * Real.sqrt 3)) →
  s * s * s = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l2896_289614
