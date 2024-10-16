import Mathlib

namespace NUMINAMATH_CALUDE_expected_winnings_is_five_thirds_l3361_336126

/-- A coin with three possible outcomes -/
inductive CoinOutcome
  | Heads
  | Tails
  | Edge

/-- The probability of each outcome -/
def probability (outcome : CoinOutcome) : ℚ :=
  match outcome with
  | .Heads => 1/3
  | .Tails => 1/2
  | .Edge => 1/6

/-- The payoff for each outcome in dollars -/
def payoff (outcome : CoinOutcome) : ℤ :=
  match outcome with
  | .Heads => 2
  | .Tails => 4
  | .Edge => -6

/-- The expected winnings from flipping the coin -/
def expectedWinnings : ℚ :=
  (probability CoinOutcome.Heads * payoff CoinOutcome.Heads) +
  (probability CoinOutcome.Tails * payoff CoinOutcome.Tails) +
  (probability CoinOutcome.Edge * payoff CoinOutcome.Edge)

theorem expected_winnings_is_five_thirds :
  expectedWinnings = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_winnings_is_five_thirds_l3361_336126


namespace NUMINAMATH_CALUDE_cost_exceeds_fifty_l3361_336164

/-- Calculates the total cost of items after discount and tax --/
def total_cost (pizza_price : ℝ) (juice_price : ℝ) (chips_price : ℝ) (chocolate_price : ℝ)
  (pizza_count : ℕ) (juice_count : ℕ) (chips_count : ℕ) (chocolate_count : ℕ)
  (pizza_discount : ℝ) (sales_tax : ℝ) : ℝ :=
  let pizza_cost := pizza_price * pizza_count
  let juice_cost := juice_price * juice_count
  let chips_cost := chips_price * chips_count
  let chocolate_cost := chocolate_price * chocolate_count
  let discounted_pizza_cost := pizza_cost * (1 - pizza_discount)
  let subtotal := discounted_pizza_cost + juice_cost + chips_cost + chocolate_cost
  subtotal * (1 + sales_tax)

/-- Theorem: The total cost exceeds $50 --/
theorem cost_exceeds_fifty :
  total_cost 15 4 3.5 1.25 3 4 2 5 0.1 0.05 > 50 := by
  sorry

end NUMINAMATH_CALUDE_cost_exceeds_fifty_l3361_336164


namespace NUMINAMATH_CALUDE_probability_at_least_one_boy_l3361_336199

/-- The probability of selecting at least one boy from a group of 5 girls and 2 boys,
    given that girl A is already selected and a total of 3 people are to be selected. -/
theorem probability_at_least_one_boy (total_girls : ℕ) (total_boys : ℕ) 
  (h_girls : total_girls = 5) (h_boys : total_boys = 2) (selection_size : ℕ) 
  (h_selection : selection_size = 3) :
  (Nat.choose (total_boys + total_girls - 1) (selection_size - 1) - 
   Nat.choose (total_girls - 1) (selection_size - 1)) / 
  Nat.choose (total_boys + total_girls - 1) (selection_size - 1) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_boy_l3361_336199


namespace NUMINAMATH_CALUDE_units_digit_of_M_M15_l3361_336115

-- Define the Modified Lucas sequence
def M : ℕ → ℕ
  | 0 => 3
  | 1 => 2
  | (n + 2) => M (n + 1) + M n

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_M_M15 : unitsDigit (M (M 15)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_M_M15_l3361_336115


namespace NUMINAMATH_CALUDE_box_office_growth_l3361_336161

theorem box_office_growth (x : ℝ) : 
  (∃ (initial final : ℝ), 
    initial = 2 ∧ 
    final = 4 ∧ 
    final = initial * (1 + x)^2) ↔ 
  2 * (1 + x)^2 = 4 := by sorry

end NUMINAMATH_CALUDE_box_office_growth_l3361_336161


namespace NUMINAMATH_CALUDE_odd_special_function_sum_l3361_336185

/-- An odd function f: ℝ → ℝ satisfying f(1+x) = f(1-x) for all x and f(1) = 2 -/
def OddSpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (1 + x) = f (1 - x)) ∧
  (f 1 = 2)

/-- Theorem stating that for an OddSpecialFunction f, f(2010) + f(2011) = -2 -/
theorem odd_special_function_sum (f : ℝ → ℝ) (hf : OddSpecialFunction f) : 
  f 2010 + f 2011 = -2 := by
  sorry

end NUMINAMATH_CALUDE_odd_special_function_sum_l3361_336185


namespace NUMINAMATH_CALUDE_thabo_hardcover_books_l3361_336159

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ

/-- Thabo's book collection satisfies the given conditions -/
def is_valid_collection (books : BookCollection) : Prop :=
  books.hardcover_nonfiction + books.paperback_nonfiction + books.paperback_fiction = 200 ∧
  books.paperback_nonfiction = books.hardcover_nonfiction + 20 ∧
  books.paperback_fiction = 2 * books.paperback_nonfiction

theorem thabo_hardcover_books :
  ∀ (books : BookCollection), is_valid_collection books → books.hardcover_nonfiction = 35 := by
  sorry

end NUMINAMATH_CALUDE_thabo_hardcover_books_l3361_336159


namespace NUMINAMATH_CALUDE_cube_difference_l3361_336186

theorem cube_difference (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) :
  x^3 - y^3 = 108 := by sorry

end NUMINAMATH_CALUDE_cube_difference_l3361_336186


namespace NUMINAMATH_CALUDE_cos_300_degrees_l3361_336104

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l3361_336104


namespace NUMINAMATH_CALUDE_cloth_cost_calculation_l3361_336169

/-- The total cost of cloth given its length and price per meter -/
def totalCost (length : ℝ) (pricePerMeter : ℝ) : ℝ :=
  length * pricePerMeter

/-- Theorem: The total cost of 9.25 meters of cloth at $47 per meter is $434.75 -/
theorem cloth_cost_calculation :
  totalCost 9.25 47 = 434.75 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_calculation_l3361_336169


namespace NUMINAMATH_CALUDE_line_slope_thirty_degrees_l3361_336179

theorem line_slope_thirty_degrees (m : ℝ) : 
  (∃ (x y : ℝ), x + m * y - 3 = 0) →
  (Real.tan (30 * π / 180) = -1 / m) →
  m = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_line_slope_thirty_degrees_l3361_336179


namespace NUMINAMATH_CALUDE_proportion_solution_l3361_336180

theorem proportion_solution (x : ℝ) :
  (0.75 : ℝ) / x = (4.5 : ℝ) / (7/3 : ℝ) →
  x = (0.75 : ℝ) * (7/3 : ℝ) / (4.5 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_proportion_solution_l3361_336180


namespace NUMINAMATH_CALUDE_nested_radical_equality_l3361_336146

theorem nested_radical_equality (a b : ℕ) (h1 : a < b) (h2 : a > 0) (h3 : b > 0) :
  Real.sqrt (1 + Real.sqrt (24 + 15 * Real.sqrt 3)) = Real.sqrt a + Real.sqrt b →
  a = 2 ∧ b = 3 := by
sorry

end NUMINAMATH_CALUDE_nested_radical_equality_l3361_336146


namespace NUMINAMATH_CALUDE_greatest_base12_divisible_by_7_l3361_336149

/-- Converts a base 12 number to decimal --/
def base12ToDecimal (a b c : Nat) : Nat :=
  a * 12^2 + b * 12 + c

/-- Checks if a number is divisible by 7 --/
def isDivisibleBy7 (n : Nat) : Prop :=
  n % 7 = 0

/-- Theorem: BB6₁₂ is the greatest 3-digit base 12 positive integer divisible by 7 --/
theorem greatest_base12_divisible_by_7 :
  let bb6 := base12ToDecimal 11 11 6
  isDivisibleBy7 bb6 ∧
  ∀ n, n > bb6 → n ≤ base12ToDecimal 11 11 11 →
    ¬(isDivisibleBy7 n) :=
by sorry

end NUMINAMATH_CALUDE_greatest_base12_divisible_by_7_l3361_336149


namespace NUMINAMATH_CALUDE_megacorp_oil_refining_earnings_l3361_336157

/-- MegaCorp's financial data and fine calculation --/
theorem megacorp_oil_refining_earnings 
  (daily_mining_earnings : ℝ)
  (monthly_expenses : ℝ)
  (fine_amount : ℝ)
  (fine_rate : ℝ)
  (days_per_month : ℕ)
  (months_per_year : ℕ)
  (h1 : daily_mining_earnings = 3000000)
  (h2 : monthly_expenses = 30000000)
  (h3 : fine_amount = 25600000)
  (h4 : fine_rate = 0.01)
  (h5 : days_per_month = 30)
  (h6 : months_per_year = 12) :
  ∃ daily_oil_earnings : ℝ,
    daily_oil_earnings = 5111111.11 ∧
    fine_amount = fine_rate * months_per_year * 
      (days_per_month * (daily_mining_earnings + daily_oil_earnings) - monthly_expenses) :=
by sorry

end NUMINAMATH_CALUDE_megacorp_oil_refining_earnings_l3361_336157


namespace NUMINAMATH_CALUDE_negation_of_existential_negation_of_specific_existential_l3361_336102

theorem negation_of_existential (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by sorry

theorem negation_of_specific_existential :
  (¬ ∃ x : ℝ, x^2 - 2 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existential_negation_of_specific_existential_l3361_336102


namespace NUMINAMATH_CALUDE_probability_at_least_two_same_l3361_336171

def num_dice : ℕ := 8
def sides_per_die : ℕ := 8

theorem probability_at_least_two_same (num_dice : ℕ) (sides_per_die : ℕ) :
  num_dice = 8 ∧ sides_per_die = 8 →
  (1 : ℚ) - (Nat.factorial num_dice : ℚ) / (sides_per_die ^ num_dice : ℚ) = 1291 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_same_l3361_336171


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_difference_l3361_336194

theorem cube_root_equation_solution_difference : ∃ x₁ x₂ : ℝ,
  (x₁ ≠ x₂) ∧
  ((9 - x₁^2 / 4)^(1/3 : ℝ) = -3) ∧
  ((9 - x₂^2 / 4)^(1/3 : ℝ) = -3) ∧
  (abs (x₁ - x₂) = 24) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_difference_l3361_336194


namespace NUMINAMATH_CALUDE_pqr_value_l3361_336100

theorem pqr_value (p q r : ℤ) 
  (h1 : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h2 : p + q + r = 30)
  (h3 : 1 / p + 1 / q + 1 / r + 390 / (p * q * r) = 1) :
  p * q * r = 1680 := by
sorry

end NUMINAMATH_CALUDE_pqr_value_l3361_336100


namespace NUMINAMATH_CALUDE_equation_solution_l3361_336167

theorem equation_solution :
  ∃ x : ℝ, (8 : ℝ) ^ (2 * x - 9) = 2 ^ (-2 * x - 3) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3361_336167


namespace NUMINAMATH_CALUDE_farmer_land_area_l3361_336109

theorem farmer_land_area (A : ℝ) 
  (h1 : 0.9 * A * 0.1 = 360) 
  (h2 : 0.9 * A * 0.6 + 0.9 * A * 0.3 + 360 = 0.9 * A) : 
  A = 4000 := by
  sorry

end NUMINAMATH_CALUDE_farmer_land_area_l3361_336109


namespace NUMINAMATH_CALUDE_equation_solution_l3361_336145

theorem equation_solution (x : ℝ) :
  x^2 + x + 1 = 1 / (x^2 - x + 1) ∧ x^2 - x + 1 ≠ 0 → x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3361_336145


namespace NUMINAMATH_CALUDE_grade_difference_l3361_336101

theorem grade_difference (x y : ℕ) (h : 3 * y = 4 * x) :
  y - x = 3 ∧ y - x = 4 :=
sorry

end NUMINAMATH_CALUDE_grade_difference_l3361_336101


namespace NUMINAMATH_CALUDE_sqrt_two_squared_l3361_336181

theorem sqrt_two_squared : Real.sqrt 2 * Real.sqrt 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_squared_l3361_336181


namespace NUMINAMATH_CALUDE_percentage_difference_l3361_336138

theorem percentage_difference : 
  (80 / 100 * 40) - (4 / 5 * 15) = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3361_336138


namespace NUMINAMATH_CALUDE_math_club_smallest_size_l3361_336170

theorem math_club_smallest_size :
  ∀ (total boys girls : ℕ),
    total = boys + girls →
    girls ≥ 2 →
    boys > (91 : ℝ) / 100 * total →
    total ≥ 23 ∧ ∃ (t b g : ℕ), t = 23 ∧ b + g = t ∧ g ≥ 2 ∧ b > (91 : ℝ) / 100 * t :=
by
  sorry

end NUMINAMATH_CALUDE_math_club_smallest_size_l3361_336170


namespace NUMINAMATH_CALUDE_flag_stripes_l3361_336132

theorem flag_stripes :
  ∀ (S : ℕ), 
    S > 0 →
    (10 * (1 + (S - 1) / 2 : ℚ) = 70) →
    S = 13 := by
  sorry

end NUMINAMATH_CALUDE_flag_stripes_l3361_336132


namespace NUMINAMATH_CALUDE_tower_height_l3361_336141

/-- The height of a tower given specific angles and distance -/
theorem tower_height (distance : ℝ) (elevation_angle depression_angle : ℝ) 
  (h1 : distance = 20)
  (h2 : elevation_angle = 30 * π / 180)
  (h3 : depression_angle = 45 * π / 180) :
  ∃ (height : ℝ), height = 20 * (1 + Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_tower_height_l3361_336141


namespace NUMINAMATH_CALUDE_mismatching_socks_l3361_336105

theorem mismatching_socks (total_socks : ℕ) (matching_pairs : ℕ) 
  (h1 : total_socks = 25) (h2 : matching_pairs = 4) :
  total_socks - 2 * matching_pairs = 17 := by
  sorry

end NUMINAMATH_CALUDE_mismatching_socks_l3361_336105


namespace NUMINAMATH_CALUDE_tank_filling_time_l3361_336168

/-- Represents the time taken to fill a tank using different pipe configurations -/
structure TankFilling where
  /-- Time taken by pipe B alone to fill the tank -/
  time_B : ℝ
  /-- Time taken by all three pipes together to fill the tank -/
  time_ABC : ℝ

/-- Proves that given the conditions, the time taken for all three pipes to fill the tank is 10 hours -/
theorem tank_filling_time (t : TankFilling) (h1 : t.time_B = 35) : t.time_ABC = 10 := by
  sorry

#check tank_filling_time

end NUMINAMATH_CALUDE_tank_filling_time_l3361_336168


namespace NUMINAMATH_CALUDE_expression_evaluation_l3361_336155

theorem expression_evaluation : 
  2 * 3 + 4 - 5 / 6 = 37 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3361_336155


namespace NUMINAMATH_CALUDE_mama_bird_stolen_worms_l3361_336176

/-- The number of worms stolen from Mama bird -/
def stolen_worms : ℕ := by sorry

theorem mama_bird_stolen_worms :
  let babies : ℕ := 6
  let worms_per_baby_per_day : ℕ := 3
  let days : ℕ := 3
  let papa_worms : ℕ := 9
  let mama_worms : ℕ := 13
  let additional_worms_needed : ℕ := 34
  
  stolen_worms = 2 := by sorry

end NUMINAMATH_CALUDE_mama_bird_stolen_worms_l3361_336176


namespace NUMINAMATH_CALUDE_inequality_solution_l3361_336114

theorem inequality_solution (x : ℝ) (h : x ≠ -1) :
  (x - 2) / (x + 1) ≤ 2 ↔ x ≤ -4 ∨ x > -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3361_336114


namespace NUMINAMATH_CALUDE_opposite_of_2022_l3361_336129

theorem opposite_of_2022 : -(2022 : ℝ) = -2022 := by sorry

end NUMINAMATH_CALUDE_opposite_of_2022_l3361_336129


namespace NUMINAMATH_CALUDE_rachel_homework_l3361_336153

theorem rachel_homework (reading_pages math_pages : ℕ) : 
  reading_pages = math_pages + 6 →
  reading_pages = 14 →
  math_pages = 8 := by sorry

end NUMINAMATH_CALUDE_rachel_homework_l3361_336153


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_zero_l3361_336150

/-- A quadratic function g and its inverse g⁻¹ -/
structure QuadraticWithInverse where
  a : ℝ
  b : ℝ
  c : ℝ
  g : ℝ → ℝ
  g_inv : ℝ → ℝ
  g_eq : g = fun x ↦ a * x^2 + b * x + c
  g_inv_eq : g_inv = fun x ↦ c * x^2 + b * x + a
  inverse : ∀ x, g (g_inv x) = x

/-- The sum of coefficients of a quadratic function with inverse is zero -/
theorem sum_of_coefficients_is_zero (q : QuadraticWithInverse) : q.a + q.b + q.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_zero_l3361_336150


namespace NUMINAMATH_CALUDE_inequality_proof_l3361_336130

theorem inequality_proof (a b : ℝ) (h : 1/a > 1/b ∧ 1/b > 0) : 
  a^3 < b^3 ∧ Real.sqrt b - Real.sqrt a < Real.sqrt (b - a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3361_336130


namespace NUMINAMATH_CALUDE_liquid_rise_ratio_l3361_336196

-- Define the cones and marbles
def narrow_cone_radius : ℝ := 5
def wide_cone_radius : ℝ := 10
def narrow_marble_radius : ℝ := 2
def wide_marble_radius : ℝ := 3

-- Define the volume ratio
def volume_ratio : ℝ := 4

-- Theorem statement
theorem liquid_rise_ratio :
  let narrow_cone_volume := (1/3) * Real.pi * narrow_cone_radius^2
  let wide_cone_volume := (1/3) * Real.pi * wide_cone_radius^2
  let narrow_marble_volume := (4/3) * Real.pi * narrow_marble_radius^3
  let wide_marble_volume := (4/3) * Real.pi * wide_marble_radius^3
  let narrow_cone_rise := narrow_marble_volume / (Real.pi * narrow_cone_radius^2)
  let wide_cone_rise := wide_marble_volume / (Real.pi * wide_cone_radius^2)
  wide_cone_volume = volume_ratio * narrow_cone_volume →
  narrow_cone_rise / wide_cone_rise = 8 := by
  sorry

end NUMINAMATH_CALUDE_liquid_rise_ratio_l3361_336196


namespace NUMINAMATH_CALUDE_addilynns_broken_eggs_l3361_336190

/-- The number of eggs in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of eggs Addilynn bought -/
def dozens_bought : ℕ := 6

/-- The number of eggs left on the shelf -/
def eggs_left : ℕ := 21

/-- The number of eggs Addilynn accidentally broke -/
def eggs_broken : ℕ := dozens_bought * dozen / 2 - eggs_left

theorem addilynns_broken_eggs :
  eggs_broken = 15 :=
sorry

end NUMINAMATH_CALUDE_addilynns_broken_eggs_l3361_336190


namespace NUMINAMATH_CALUDE_f_of_5_l3361_336106

def f (x : ℝ) : ℝ := x^2 - x

theorem f_of_5 : f 5 = 20 := by sorry

end NUMINAMATH_CALUDE_f_of_5_l3361_336106


namespace NUMINAMATH_CALUDE_remainder_evaluation_l3361_336124

-- Define the remainder function
def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

-- State the theorem
theorem remainder_evaluation :
  rem (-1/7 : ℚ) (1/3 : ℚ) = 4/21 := by
  sorry

end NUMINAMATH_CALUDE_remainder_evaluation_l3361_336124


namespace NUMINAMATH_CALUDE_homogeneous_de_solution_l3361_336144

/-- The homogeneous differential equation -/
def homogeneous_de (x y : ℝ) (dx dy : ℝ) : Prop :=
  (x^2 - y^2) * dy - 2 * y * x * dx = 0

/-- The general solution to the homogeneous differential equation -/
def general_solution (x y C : ℝ) : Prop :=
  x^2 + y^2 = C * y

/-- Theorem stating that the general solution satisfies the homogeneous differential equation -/
theorem homogeneous_de_solution (x y C : ℝ) :
  general_solution x y C →
  ∃ (dx dy : ℝ), homogeneous_de x y dx dy :=
sorry

end NUMINAMATH_CALUDE_homogeneous_de_solution_l3361_336144


namespace NUMINAMATH_CALUDE_girls_without_notebooks_l3361_336172

theorem girls_without_notebooks (total_girls : ℕ) (total_boys : ℕ) 
  (notebooks_brought : ℕ) (boys_with_notebooks : ℕ) (girls_with_notebooks : ℕ) 
  (h1 : total_girls = 18)
  (h2 : total_boys = 20)
  (h3 : notebooks_brought = 30)
  (h4 : boys_with_notebooks = 17)
  (h5 : girls_with_notebooks = 11) :
  total_girls - girls_with_notebooks = 7 := by
sorry

end NUMINAMATH_CALUDE_girls_without_notebooks_l3361_336172


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3361_336166

theorem decimal_to_fraction : (2.35 : ℚ) = 47 / 20 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3361_336166


namespace NUMINAMATH_CALUDE_intersection_M_N_l3361_336122

def M : Set ℕ := {0, 2, 4, 8}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_M_N : M ∩ N = {0, 4, 8} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3361_336122


namespace NUMINAMATH_CALUDE_max_safe_daily_dose_l3361_336147

/-- Represents the amount of medication in the body after n doses -/
def medication_amount (m : ℝ) : ℕ → ℝ
| 0 => 0
| n + 1 => m + 0.2 * medication_amount m n

/-- The maximum safe amount of medication in the body -/
def max_safe_amount : ℝ := 25

/-- Theorem stating the maximum safe daily dose -/
theorem max_safe_daily_dose :
  ∃ (max_m : ℝ), max_m = 20 ∧
  ∀ (m : ℝ), m ≤ max_m →
  ∀ (n : ℕ), medication_amount m n ≤ max_safe_amount :=
sorry

end NUMINAMATH_CALUDE_max_safe_daily_dose_l3361_336147


namespace NUMINAMATH_CALUDE_problem_statement_l3361_336158

-- Define the function f
noncomputable def f (a m x : ℝ) : ℝ := m * x^a + (Real.log (1 + x))^a - a * Real.log (1 - x) - 2

-- State the theorem
theorem problem_statement (a : ℝ) (h1 : a^(1/2) ≤ 3) (h2 : Real.log 3 / Real.log a ≤ 1/2) :
  ((0 < a ∧ a < 1) ∨ a = 9) ∧
  (a > 1 → ∃ m : ℝ, f a m (1/2) = a → f a m (-1/2) = -13) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l3361_336158


namespace NUMINAMATH_CALUDE_initial_marbles_count_l3361_336142

/-- The number of marbles Carla bought -/
def marbles_bought : ℕ := 134

/-- The total number of marbles Carla has now -/
def total_marbles_now : ℕ := 187

/-- The initial number of marbles Carla had -/
def initial_marbles : ℕ := total_marbles_now - marbles_bought

theorem initial_marbles_count : initial_marbles = 53 := by
  sorry

end NUMINAMATH_CALUDE_initial_marbles_count_l3361_336142


namespace NUMINAMATH_CALUDE_trig_expression_equality_l3361_336188

theorem trig_expression_equality : 
  (Real.sin (20 * π / 180) * Real.sqrt (1 + Real.cos (40 * π / 180))) / 
  (Real.cos (50 * π / 180)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l3361_336188


namespace NUMINAMATH_CALUDE_expression_equals_one_l3361_336131

theorem expression_equals_one (x y z : ℝ) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) :
  (x^2 / ((x-y)*(x-z))) + (y^2 / ((y-x)*(y-z))) + (z^2 / ((z-x)*(z-y))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l3361_336131


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l3361_336197

theorem line_segment_endpoint (x y : ℝ) :
  let start : ℝ × ℝ := (2, 2)
  let length : ℝ := 8
  let slope : ℝ := 3/4
  y > 0 ∧
  (y - start.2) / (x - start.1) = slope ∧
  Real.sqrt ((x - start.1)^2 + (y - start.2)^2) = length →
  ((x = 2 + 4 * Real.sqrt 5475 / 25 ∧ y = 3/4 * (2 + 4 * Real.sqrt 5475 / 25) + 1/2) ∨
   (x = 2 - 4 * Real.sqrt 5475 / 25 ∧ y = 3/4 * (2 - 4 * Real.sqrt 5475 / 25) + 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l3361_336197


namespace NUMINAMATH_CALUDE_room_area_l3361_336103

/-- The area of a room given the costs of floor replacement -/
theorem room_area (removal_cost : ℝ) (per_sqft_cost : ℝ) (total_cost : ℝ) : 
  removal_cost = 50 →
  per_sqft_cost = 1.25 →
  total_cost = 120 →
  (total_cost - removal_cost) / per_sqft_cost = 56 := by
sorry

end NUMINAMATH_CALUDE_room_area_l3361_336103


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3361_336156

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Represents a line with slope m and y-intercept c -/
structure Line where
  m : ℝ
  c : ℝ

/-- States that a line passes through a focus of the hyperbola -/
def passes_through_focus (l : Line) (h : Hyperbola) : Prop :=
  ∃ x y, y = l.m * x + l.c ∧ x^2 + y^2 = h.a^2 + h.b^2

/-- States that a line is parallel to an asymptote of the hyperbola -/
def parallel_to_asymptote (l : Line) (h : Hyperbola) : Prop :=
  l.m = h.b / h.a ∨ l.m = -h.b / h.a

theorem hyperbola_equation (h : Hyperbola) (l : Line) 
  (h_focus : passes_through_focus l h)
  (h_parallel : parallel_to_asymptote l h)
  (h_line : l.m = 2 ∧ l.c = 10) :
  h.a^2 = 5 ∧ h.b^2 = 20 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3361_336156


namespace NUMINAMATH_CALUDE_alloy_composition_proof_l3361_336174

/-- The percentage of chromium in the first alloy -/
def chromium_percent_1 : ℝ := 0.12

/-- The percentage of chromium in the second alloy -/
def chromium_percent_2 : ℝ := 0.08

/-- The amount of the first alloy used in kg -/
def amount_1 : ℝ := 15

/-- The percentage of chromium in the new alloy -/
def chromium_percent_new : ℝ := 0.092

/-- The amount of the second alloy used in kg -/
def amount_2 : ℝ := 35

theorem alloy_composition_proof :
  chromium_percent_1 * amount_1 + chromium_percent_2 * amount_2 =
  chromium_percent_new * (amount_1 + amount_2) := by
  sorry

end NUMINAMATH_CALUDE_alloy_composition_proof_l3361_336174


namespace NUMINAMATH_CALUDE_lateral_surface_area_is_four_l3361_336183

/-- A regular quadrilateral pyramid inscribed in a unit sphere -/
structure RegularQuadPyramid where
  /-- The radius of the sphere in which the pyramid is inscribed -/
  radius : ℝ
  /-- The dihedral angle at the apex of the pyramid in radians -/
  dihedral_angle : ℝ
  /-- Assertion that the radius is 1 -/
  radius_is_one : radius = 1
  /-- Assertion that the dihedral angle is π/4 (45 degrees) -/
  angle_is_45 : dihedral_angle = Real.pi / 4

/-- The lateral surface area of a regular quadrilateral pyramid -/
def lateral_surface_area (p : RegularQuadPyramid) : ℝ :=
  sorry

/-- Theorem: The lateral surface area of the specified pyramid is 4 -/
theorem lateral_surface_area_is_four (p : RegularQuadPyramid) :
  lateral_surface_area p = 4 := by
  sorry

end NUMINAMATH_CALUDE_lateral_surface_area_is_four_l3361_336183


namespace NUMINAMATH_CALUDE_max_m_value_l3361_336139

theorem max_m_value (m : ℝ) : 
  (¬ ∃ x : ℝ, x ≥ 3 ∧ 2*x - 1 < m) → m ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l3361_336139


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_l3361_336148

theorem isosceles_right_triangle (a b c h_a h_b h_c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  h_a > 0 → h_b > 0 → h_c > 0 →
  a * h_a = 2 * area → b * h_b = 2 * area → c * h_c = 2 * area →
  a ≤ h_a → b ≤ h_b →
  a = b ∧ c = a * Real.sqrt 2 :=
sorry


end NUMINAMATH_CALUDE_isosceles_right_triangle_l3361_336148


namespace NUMINAMATH_CALUDE_vicente_meat_purchase_l3361_336193

theorem vicente_meat_purchase
  (rice_kg : ℕ)
  (rice_price : ℚ)
  (meat_price : ℚ)
  (total_spent : ℚ)
  (h1 : rice_kg = 5)
  (h2 : rice_price = 2)
  (h3 : meat_price = 5)
  (h4 : total_spent = 25)
  : (total_spent - rice_kg * rice_price) / meat_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_vicente_meat_purchase_l3361_336193


namespace NUMINAMATH_CALUDE_zoe_drank_bottles_l3361_336117

def initial_bottles : ℕ := 42
def bought_bottles : ℕ := 30
def final_bottles : ℕ := 47

theorem zoe_drank_bottles :
  ∃ (drank_bottles : ℕ), initial_bottles - drank_bottles + bought_bottles = final_bottles ∧ drank_bottles = 25 := by
  sorry

end NUMINAMATH_CALUDE_zoe_drank_bottles_l3361_336117


namespace NUMINAMATH_CALUDE_cookies_per_tray_is_12_l3361_336184

/-- The number of cookies per tray that Frank bakes -/
def cookies_per_tray : ℕ := 12

/-- The number of days Frank bakes cookies -/
def baking_days : ℕ := 6

/-- The number of trays Frank bakes per day -/
def trays_per_day : ℕ := 2

/-- The number of cookies Frank eats per day -/
def frank_cookies_per_day : ℕ := 1

/-- The number of cookies Ted eats on the sixth day -/
def ted_cookies : ℕ := 4

/-- The number of cookies left after Ted leaves -/
def cookies_left : ℕ := 134

theorem cookies_per_tray_is_12 :
  cookies_per_tray * trays_per_day * baking_days - 
  (frank_cookies_per_day * baking_days + ted_cookies) = cookies_left :=
by sorry

end NUMINAMATH_CALUDE_cookies_per_tray_is_12_l3361_336184


namespace NUMINAMATH_CALUDE_unique_function_is_zero_l3361_336112

-- Define the property that f must satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f (x - y) + f (f (1 - x * y))

-- Theorem statement
theorem unique_function_is_zero :
  ∃! f : ℝ → ℝ, SatisfiesProperty f ∧ ∀ x : ℝ, f x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_function_is_zero_l3361_336112


namespace NUMINAMATH_CALUDE_arithmetic_progression_ratio_l3361_336128

theorem arithmetic_progression_ratio (a d : ℝ) : 
  (7 * a + (7 * (7 - 1) / 2) * d = 3 * a + (3 * (3 - 1) / 2) * d + 20) → 
  a / d = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_ratio_l3361_336128


namespace NUMINAMATH_CALUDE_cement_mixture_weight_l3361_336162

/-- Given a cement mixture composed of sand, water, and gravel, where:
    - 1/4 of the mixture is sand (by weight)
    - 2/5 of the mixture is water (by weight)
    - 14 pounds of the mixture is gravel
    Prove that the total weight of the mixture is 40 pounds. -/
theorem cement_mixture_weight :
  ∀ (total_weight : ℝ),
  (1/4 : ℝ) * total_weight +     -- Weight of sand
  (2/5 : ℝ) * total_weight +     -- Weight of water
  14 = total_weight →            -- Weight of gravel
  total_weight = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_cement_mixture_weight_l3361_336162


namespace NUMINAMATH_CALUDE_mortgage_payment_proof_l3361_336198

/-- Calculates the monthly mortgage payment -/
def calculate_monthly_payment (house_price : ℕ) (deposit : ℕ) (years : ℕ) : ℚ :=
  let mortgage := house_price - deposit
  let annual_payment := mortgage / years
  annual_payment / 12

/-- Proves that the monthly payment for the given mortgage scenario is 2 thousand dollars -/
theorem mortgage_payment_proof (house_price deposit years : ℕ) 
  (h1 : house_price = 280000)
  (h2 : deposit = 40000)
  (h3 : years = 10) :
  calculate_monthly_payment house_price deposit years = 2000 := by
  sorry

#eval calculate_monthly_payment 280000 40000 10

end NUMINAMATH_CALUDE_mortgage_payment_proof_l3361_336198


namespace NUMINAMATH_CALUDE_deposit_calculation_l3361_336192

theorem deposit_calculation (total_cost : ℝ) (deposit : ℝ) : 
  deposit = 0.1 * total_cost ∧ 
  total_cost - deposit = 1080 → 
  deposit = 120 := by
sorry

end NUMINAMATH_CALUDE_deposit_calculation_l3361_336192


namespace NUMINAMATH_CALUDE_student_permutations_l3361_336173

/-- Represents the number of students --/
def n : ℕ := 5

/-- The factorial function --/
def factorial (m : ℕ) : ℕ := 
  match m with
  | 0 => 1
  | m + 1 => (m + 1) * factorial m

/-- The number of permutations of n elements --/
def permutations (n : ℕ) : ℕ := factorial n

/-- The number of permutations not in alphabetical order --/
def permutations_not_alphabetical (n : ℕ) : ℕ := permutations n - 1

/-- The number of permutations where two specific elements are consecutive --/
def permutations_consecutive_pair (n : ℕ) : ℕ := 2 * factorial (n - 1)

theorem student_permutations :
  (permutations n = 120) ∧
  (permutations_not_alphabetical n = 119) ∧
  (permutations_consecutive_pair n = 48) := by
  sorry

end NUMINAMATH_CALUDE_student_permutations_l3361_336173


namespace NUMINAMATH_CALUDE_cylinder_volume_equals_cube_surface_l3361_336154

theorem cylinder_volume_equals_cube_surface (side : ℝ) (h r V : ℝ) : 
  side = 3 → 
  6 * side^2 = 2 * π * r^2 + 2 * π * r * h → 
  h = r → 
  V = π * r^2 * h → 
  V = (81 * Real.sqrt 3 / 2) * Real.sqrt 5 / Real.sqrt π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_equals_cube_surface_l3361_336154


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3361_336107

theorem quadratic_expression_value (a : ℝ) (h : a^2 + 4*a - 5 = 0) : 3*a^2 + 12*a = 15 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3361_336107


namespace NUMINAMATH_CALUDE_sum_and_transformations_l3361_336120

theorem sum_and_transformations (x y z M : ℚ) : 
  x + y + z = 72 ∧ 
  x - 9 = M ∧ 
  y + 9 = M ∧ 
  9 * z = M → 
  M = 34 := by
sorry

end NUMINAMATH_CALUDE_sum_and_transformations_l3361_336120


namespace NUMINAMATH_CALUDE_water_tank_problem_l3361_336177

/-- Calculates the water volume in a tank after a given number of hours, 
    with specified initial volume, loss rate, and water additions. -/
def water_volume (initial_volume : ℝ) (loss_rate : ℝ) (additions : List ℝ) : ℝ :=
  initial_volume - loss_rate * additions.length + additions.sum

/-- The water volume problem -/
theorem water_tank_problem : 
  let initial_volume : ℝ := 40
  let loss_rate : ℝ := 2
  let additions : List ℝ := [0, 0, 1, 3]
  water_volume initial_volume loss_rate additions = 36 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_problem_l3361_336177


namespace NUMINAMATH_CALUDE_sector_area_l3361_336110

/-- The area of a sector of a circle with radius 4 cm and arc length 3.5 cm is 7 cm² -/
theorem sector_area (r : ℝ) (arc_length : ℝ) (h1 : r = 4) (h2 : arc_length = 3.5) :
  (arc_length / (2 * π * r)) * (π * r^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3361_336110


namespace NUMINAMATH_CALUDE_infinite_intersection_points_l3361_336151

/-- The first curve equation -/
def curve1 (x y : ℝ) : Prop :=
  2 * x^2 - x * y - y^2 - x - 2 * y - 1 = 0

/-- The second curve equation -/
def curve2 (x y : ℝ) : Prop :=
  3 * x^2 - 4 * x * y + y^2 - 3 * x + y = 0

/-- The set of points that satisfy both curve equations -/
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | curve1 p.1 p.2 ∧ curve2 p.1 p.2}

/-- Theorem stating that the intersection of the two curves contains infinitely many points -/
theorem infinite_intersection_points : Set.Infinite intersection_points :=
sorry

end NUMINAMATH_CALUDE_infinite_intersection_points_l3361_336151


namespace NUMINAMATH_CALUDE_factor_implies_d_value_l3361_336140

theorem factor_implies_d_value (d : ℝ) : 
  (∀ x : ℝ, (2 * x + 5) ∣ (8 * x^3 + 27 * x^2 + d * x + 55)) → d = 39.5 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_d_value_l3361_336140


namespace NUMINAMATH_CALUDE_geometry_test_passing_l3361_336137

theorem geometry_test_passing (total_problems : ℕ) (passing_percentage : ℚ) 
  (hp : passing_percentage = 85 / 100) (ht : total_problems = 50) : 
  ∃ (max_missed : ℕ), 
    (((total_problems - max_missed : ℚ) / total_problems) ≥ passing_percentage ∧ 
     ∀ (n : ℕ), n > max_missed → 
       ((total_problems - n : ℚ) / total_problems) < passing_percentage) ∧
    max_missed = 7 :=
sorry

end NUMINAMATH_CALUDE_geometry_test_passing_l3361_336137


namespace NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l3361_336136

/-- Given a line in vector form, prove its equivalence to slope-intercept form -/
theorem line_vector_to_slope_intercept :
  let vector_line : ℝ × ℝ → Prop :=
    λ p => (3 : ℝ) * (p.1 + 2) + (7 : ℝ) * (p.2 - 8) = 0
  let slope_intercept_line : ℝ × ℝ → Prop :=
    λ p => p.2 = (-3/7 : ℝ) * p.1 + 50/7
  ∀ p : ℝ × ℝ, vector_line p ↔ slope_intercept_line p :=
by sorry

end NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l3361_336136


namespace NUMINAMATH_CALUDE_return_journey_speed_l3361_336195

/-- Calculates the average speed of a return journey given the conditions of the problem -/
theorem return_journey_speed 
  (morning_time : ℝ) 
  (evening_time : ℝ) 
  (morning_speed : ℝ) 
  (h1 : morning_time = 1) 
  (h2 : evening_time = 1.5) 
  (h3 : morning_speed = 30) : 
  (morning_speed * morning_time) / evening_time = 20 :=
by
  sorry

#check return_journey_speed

end NUMINAMATH_CALUDE_return_journey_speed_l3361_336195


namespace NUMINAMATH_CALUDE_minimum_nickels_needed_l3361_336116

def jacket_cost : ℚ := 45
def ten_dollar_bills : ℕ := 4
def quarters : ℕ := 10
def nickel_value : ℚ := 0.05

theorem minimum_nickels_needed : 
  ∀ n : ℕ, (ten_dollar_bills * 10 + quarters * 0.25 + n * nickel_value ≥ jacket_cost) → n ≥ 50 := by
  sorry

end NUMINAMATH_CALUDE_minimum_nickels_needed_l3361_336116


namespace NUMINAMATH_CALUDE_total_pencils_l3361_336118

/-- Given an initial number of pencils and a number of pencils added,
    the total number of pencils is equal to the sum of the initial number and the added number. -/
theorem total_pencils (initial : ℕ) (added : ℕ) :
  initial + added = initial + added :=
by sorry

end NUMINAMATH_CALUDE_total_pencils_l3361_336118


namespace NUMINAMATH_CALUDE_max_value_a_sqrt_1_plus_b_sq_l3361_336125

theorem max_value_a_sqrt_1_plus_b_sq (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (heq : a^2 / 2 + b^2 = 4) :
  ∃ (max : ℝ), max = (5 * Real.sqrt 2) / 2 ∧ 
  ∀ (x y : ℝ), x > 0 → y > 0 → x^2 / 2 + y^2 = 4 → 
  x * Real.sqrt (1 + y^2) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_a_sqrt_1_plus_b_sq_l3361_336125


namespace NUMINAMATH_CALUDE_determinant_implies_fraction_value_l3361_336123

-- Define the determinant operation
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Theorem statement
theorem determinant_implies_fraction_value (θ : ℝ) :
  det (Real.sin θ) 2 (Real.cos θ) 1 = 0 →
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 3 :=
by sorry

end NUMINAMATH_CALUDE_determinant_implies_fraction_value_l3361_336123


namespace NUMINAMATH_CALUDE_arccos_cos_three_pi_half_l3361_336135

theorem arccos_cos_three_pi_half : Real.arccos (Real.cos (3 * π / 2)) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_three_pi_half_l3361_336135


namespace NUMINAMATH_CALUDE_remaining_money_l3361_336127

def savings : ℕ := 5555 -- in base 8
def ticket_cost : ℕ := 1200 -- in base 10

def base_8_to_10 (n : ℕ) : ℕ :=
  (n / 1000) * 512 + ((n / 100) % 10) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem remaining_money :
  base_8_to_10 savings - ticket_cost = 1725 := by sorry

end NUMINAMATH_CALUDE_remaining_money_l3361_336127


namespace NUMINAMATH_CALUDE_monster_family_eyes_l3361_336187

/-- Represents the number of eyes for each family member -/
structure MonsterEyes where
  mom : Nat
  dad : Nat
  child : Nat
  num_children : Nat

/-- Calculates the total number of eyes in the monster family -/
def total_eyes (m : MonsterEyes) : Nat :=
  m.mom + m.dad + m.child * m.num_children

/-- Theorem stating that the total number of eyes in the given monster family is 16 -/
theorem monster_family_eyes :
  ∃ m : MonsterEyes, m.mom = 1 ∧ m.dad = 3 ∧ m.child = 4 ∧ m.num_children = 3 ∧ total_eyes m = 16 := by
  sorry

end NUMINAMATH_CALUDE_monster_family_eyes_l3361_336187


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3361_336175

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- Define the point of tangency
def tangent_point : ℝ × ℝ := (1, f 1)

-- Theorem statement
theorem tangent_line_equation :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (x = tangent_point.1 ∧ y = tangent_point.2) ∨
    (y - tangent_point.2 = m * (x - tangent_point.1)) ↔
    (2 * x - y - 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3361_336175


namespace NUMINAMATH_CALUDE_lee_proposal_time_l3361_336178

/-- Calculates the number of months needed to save for an engagement ring based on annual salary and monthly savings. -/
def months_to_save_for_ring (annual_salary : ℕ) (monthly_savings : ℕ) : ℕ :=
  let monthly_salary := annual_salary / 12
  let ring_cost := 2 * monthly_salary
  ring_cost / monthly_savings

/-- Proves that given the specified conditions, it takes 10 months to save for the ring. -/
theorem lee_proposal_time : months_to_save_for_ring 60000 1000 = 10 := by
  sorry

end NUMINAMATH_CALUDE_lee_proposal_time_l3361_336178


namespace NUMINAMATH_CALUDE_connie_blue_markers_l3361_336113

/-- Given the total number of markers and the number of red markers,
    calculate the number of blue markers. -/
def blue_markers (total : ℕ) (red : ℕ) : ℕ := total - red

/-- Theorem stating that Connie has 64 blue markers -/
theorem connie_blue_markers :
  let total_markers : ℕ := 105
  let red_markers : ℕ := 41
  blue_markers total_markers red_markers = 64 := by
  sorry

end NUMINAMATH_CALUDE_connie_blue_markers_l3361_336113


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l3361_336160

theorem ratio_x_to_y (x y : ℚ) (h : (15 * x - 4 * y) / (18 * x - 3 * y) = 2 / 3) :
  x / y = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l3361_336160


namespace NUMINAMATH_CALUDE_cricket_team_size_l3361_336108

/-- Represents the number of players on a cricket team -/
def total_players : ℕ := 61

/-- Represents the number of throwers on the team -/
def throwers : ℕ := 37

/-- Represents the number of right-handed players on the team -/
def right_handed : ℕ := 53

/-- Theorem stating that the total number of players is 61 -/
theorem cricket_team_size :
  total_players = throwers + (right_handed - throwers) * 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_size_l3361_336108


namespace NUMINAMATH_CALUDE_problem_statement_l3361_336143

/-- Given a function f(x) = ax^5 + bx^3 + cx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem problem_statement (a b c : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x^5 + b * x^3 + c * x - 8)
    (h2 : f (-2) = 10) : 
  f 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3361_336143


namespace NUMINAMATH_CALUDE_max_gcd_abb_aba_l3361_336152

def abb (a b : Nat) : Nat := 100 * a + 11 * b

def aba (a b : Nat) : Nat := 101 * a + 10 * b

theorem max_gcd_abb_aba :
  ∃ (a b : Nat), a ≠ b ∧ a < 10 ∧ b < 10 ∧
  (∀ (c d : Nat), c ≠ d → c < 10 → d < 10 →
    Nat.gcd (abb c d) (aba c d) ≤ Nat.gcd (abb a b) (aba a b)) ∧
  Nat.gcd (abb a b) (aba a b) = 18 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_abb_aba_l3361_336152


namespace NUMINAMATH_CALUDE_domain_of_f_l3361_336189

-- Define the function f
def f (x : ℝ) : ℝ := (2 * x - 3) ^ (1/3) + (5 - 2 * x) ^ (1/3)

-- State the theorem
theorem domain_of_f :
  ∀ x : ℝ, ∃ y : ℝ, f x = y :=
by
  sorry

end NUMINAMATH_CALUDE_domain_of_f_l3361_336189


namespace NUMINAMATH_CALUDE_milk_problem_l3361_336134

theorem milk_problem (initial_milk : ℚ) (rachel_fraction : ℚ) (jack_fraction : ℚ) :
  initial_milk = 3/4 →
  rachel_fraction = 5/8 →
  jack_fraction = 1/2 →
  (initial_milk - rachel_fraction * initial_milk) * jack_fraction = 9/64 := by
  sorry

end NUMINAMATH_CALUDE_milk_problem_l3361_336134


namespace NUMINAMATH_CALUDE_painted_cubes_l3361_336111

theorem painted_cubes (n : ℕ) (h : n = 10) : 
  n^3 - (n - 2)^3 = 488 := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_l3361_336111


namespace NUMINAMATH_CALUDE_consecutive_integers_squares_minus_product_l3361_336163

theorem consecutive_integers_squares_minus_product (n : ℕ) :
  n = 9 → (n^2 + (n+1)^2) - (n * (n+1)) = 91 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_squares_minus_product_l3361_336163


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l3361_336121

theorem bernoulli_inequality (n : ℕ+) (x : ℝ) (h : x > -1) :
  (1 + x)^(n : ℝ) ≥ 1 + n * x := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l3361_336121


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l3361_336165

/-- Sum of first n terms of an arithmetic sequence -/
def S (a : ℚ) (n : ℕ) : ℚ := n * (2 * a + (n - 1) * 5) / 2

/-- The theorem statement -/
theorem arithmetic_sequence_first_term (a : ℚ) :
  (∃ c : ℚ, ∀ n : ℕ, n > 0 → S a (2 * n) / S a n = c) →
  a = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l3361_336165


namespace NUMINAMATH_CALUDE_money_left_over_l3361_336191

theorem money_left_over (hourly_rate : ℕ) (hours_worked : ℕ) (game_cost : ℕ) (candy_cost : ℕ) : 
  hourly_rate = 8 → 
  hours_worked = 9 → 
  game_cost = 60 → 
  candy_cost = 5 → 
  hourly_rate * hours_worked - (game_cost + candy_cost) = 7 := by
  sorry

end NUMINAMATH_CALUDE_money_left_over_l3361_336191


namespace NUMINAMATH_CALUDE_twelve_sided_die_expected_value_l3361_336119

-- Define the number of sides on the die
def n : ℕ := 12

-- Define the expected value function for a fair die with n sides
def expected_value (n : ℕ) : ℚ :=
  (↑n + 1) / 2

-- Theorem statement
theorem twelve_sided_die_expected_value :
  expected_value n = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_twelve_sided_die_expected_value_l3361_336119


namespace NUMINAMATH_CALUDE_square_diagonal_l3361_336133

theorem square_diagonal (s : Real) (h : s > 0) (area_eq : s * s = 8) :
  Real.sqrt (2 * s * s) = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_l3361_336133


namespace NUMINAMATH_CALUDE_product_plus_one_is_perfect_square_l3361_336182

theorem product_plus_one_is_perfect_square (n m : ℤ) : 
  m - n = 2 → ∃ k : ℤ, n * m + 1 = k^2 := by sorry

end NUMINAMATH_CALUDE_product_plus_one_is_perfect_square_l3361_336182
