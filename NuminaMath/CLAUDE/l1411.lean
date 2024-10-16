import Mathlib

namespace NUMINAMATH_CALUDE_angle_from_terminal_point_l1411_141127

theorem angle_from_terminal_point (α : Real) :
  (∃ (x y : Real), x = Real.sin (π / 5) ∧ y = -Real.cos (π / 5) ∧ 
   x = Real.sin α ∧ y = Real.cos α) →
  ∃ (k : ℤ), α = -3 * π / 10 + 2 * π * (k : Real) :=
by sorry

end NUMINAMATH_CALUDE_angle_from_terminal_point_l1411_141127


namespace NUMINAMATH_CALUDE_radium_decay_heat_equivalence_l1411_141197

/-- The amount of radium in the Earth's crust in kilograms -/
def radium_in_crust : ℝ := 10000000000

/-- The amount of coal in kilograms that releases equivalent heat to 1 kg of radium decay -/
def coal_equivalent : ℝ := 375000

/-- The amount of coal in kilograms that releases equivalent heat to the complete decay of radium in Earth's crust -/
def total_coal_equivalent : ℝ := radium_in_crust * coal_equivalent

theorem radium_decay_heat_equivalence :
  total_coal_equivalent = 3.75 * (10 ^ 15) := by
  sorry

end NUMINAMATH_CALUDE_radium_decay_heat_equivalence_l1411_141197


namespace NUMINAMATH_CALUDE_ratio_as_percent_l1411_141188

theorem ratio_as_percent (first_part second_part : ℕ) (h1 : first_part = 25) (h2 : second_part = 50) :
  ∃ (p : ℚ), abs (p - 100 * (first_part : ℚ) / (first_part + second_part)) < 0.01 ∧ p = 33.33 := by
  sorry

end NUMINAMATH_CALUDE_ratio_as_percent_l1411_141188


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1411_141140

-- Define the geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 1 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_sequence_ratio
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_prod : a 3 * a 7 = 6)
  (h_sum : a 2 + a 8 = 5) :
  a 10 / a 4 = 3/2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1411_141140


namespace NUMINAMATH_CALUDE_vacation_days_l1411_141105

theorem vacation_days (rainy_days clear_mornings clear_afternoons : ℕ) 
  (h1 : rainy_days = 13)
  (h2 : clear_mornings = 11)
  (h3 : clear_afternoons = 12)
  (h4 : ∀ d, (d ≤ rainy_days ↔ (d ≤ rainy_days - clear_mornings ∨ d ≤ rainy_days - clear_afternoons) ∧
                               ¬(d ≤ rainy_days - clear_mornings ∧ d ≤ rainy_days - clear_afternoons))) :
  rainy_days + clear_mornings = 18 :=
by sorry

end NUMINAMATH_CALUDE_vacation_days_l1411_141105


namespace NUMINAMATH_CALUDE_sqrt_of_nine_equals_three_l1411_141190

theorem sqrt_of_nine_equals_three : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_nine_equals_three_l1411_141190


namespace NUMINAMATH_CALUDE_intersection_nonempty_condition_l1411_141164

theorem intersection_nonempty_condition (k : ℝ) : 
  let A : Set ℝ := {x | -2 < x ∧ x ≤ 1}
  let B : Set ℝ := {x | x - k ≥ 0}
  (A ∩ B).Nonempty → k ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_nonempty_condition_l1411_141164


namespace NUMINAMATH_CALUDE_parabola_c_value_l1411_141156

/-- A parabola with equation x = ay^2 + by + c, vertex at (4, 1), and passing through (1, 3) -/
def Parabola (a b c : ℝ) : Prop :=
  ∀ y : ℝ, 4 = a * 1^2 + b * 1 + c ∧
            1 = a * 3^2 + b * 3 + c

theorem parabola_c_value :
  ∀ a b c : ℝ, Parabola a b c → c = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l1411_141156


namespace NUMINAMATH_CALUDE_quadratic_roots_coefficients_l1411_141153

theorem quadratic_roots_coefficients (p q : ℝ) :
  (∀ x, x^2 + p*x + q = 0 ↔ x = p ∨ x = q) →
  ((p = 0 ∧ q = 0) ∨ (p = 1 ∧ q = -2)) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_coefficients_l1411_141153


namespace NUMINAMATH_CALUDE_swimming_pool_volume_l1411_141114

/-- A swimming pool with trapezoidal cross-section -/
structure SwimmingPool where
  width : ℝ
  length : ℝ
  shallow_depth : ℝ
  deep_depth : ℝ

/-- Calculate the volume of a swimming pool with trapezoidal cross-section -/
def pool_volume (pool : SwimmingPool) : ℝ :=
  0.5 * (pool.shallow_depth + pool.deep_depth) * pool.width * pool.length

/-- Theorem stating that the volume of the given swimming pool is 270 cubic meters -/
theorem swimming_pool_volume :
  let pool : SwimmingPool := {
    width := 9,
    length := 12,
    shallow_depth := 1,
    deep_depth := 4
  }
  pool_volume pool = 270 := by sorry

end NUMINAMATH_CALUDE_swimming_pool_volume_l1411_141114


namespace NUMINAMATH_CALUDE_smallest_n_value_l1411_141171

theorem smallest_n_value : ∃ (n : ℕ+), 
  (∀ (m : ℕ+), m < n → ¬(∃ (r g b : ℕ+), 10*r = 16*g ∧ 16*g = 18*b ∧ 18*b = 24*m ∧ 24*m = 30*22)) ∧ 
  (∃ (r g b : ℕ+), 10*r = 16*g ∧ 16*g = 18*b ∧ 18*b = 24*n ∧ 24*n = 30*22) ∧
  n = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_value_l1411_141171


namespace NUMINAMATH_CALUDE_repeating_decimal_ratio_l1411_141112

/-- Represents a repeating decimal with a given numerator and period length. -/
def RepeatingDecimal (numerator : ℕ) (period : ℕ) : ℚ :=
  numerator / (10^period - 1)

/-- The numerator of 0.8571 repeating -/
def num1 : ℕ := 8571

/-- The period length of 0.8571 repeating -/
def period1 : ℕ := 4

/-- The numerator of 0.142857 repeating -/
def num2 : ℕ := 142857

/-- The period length of 0.142857 repeating -/
def period2 : ℕ := 6

/-- Theorem stating that the ratio of the given repeating decimals equals 1/2 -/
theorem repeating_decimal_ratio :
  (RepeatingDecimal num1 period1) / (2 + RepeatingDecimal num2 period2) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_ratio_l1411_141112


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1411_141132

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (180 - 360 / n : ℝ) = 160 → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1411_141132


namespace NUMINAMATH_CALUDE_hot_dogs_dinner_l1411_141154

def hot_dogs_today : ℕ := 11
def hot_dogs_lunch : ℕ := 9

theorem hot_dogs_dinner : hot_dogs_today - hot_dogs_lunch = 2 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_dinner_l1411_141154


namespace NUMINAMATH_CALUDE_no_real_solutions_l1411_141148

theorem no_real_solutions :
  ¬∃ (x : ℝ), x + 2 * Real.sqrt (x - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1411_141148


namespace NUMINAMATH_CALUDE_prism_division_theorem_l1411_141163

/-- Represents a rectangular prism -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c

/-- Represents the division of a rectangular prism by three planes -/
structure PrismDivision (T : RectangularPrism) where
  x : ℝ
  y : ℝ
  z : ℝ
  x_bounds : 0 < x ∧ x < T.a
  y_bounds : 0 < y ∧ y < T.b
  z_bounds : 0 < z ∧ z < T.c

/-- The theorem to be proved -/
theorem prism_division_theorem (T : RectangularPrism) (div : PrismDivision T) :
  let vol_black := div.x * div.y * div.z + 
                   div.x * (T.b - div.y) * (T.c - div.z) + 
                   (T.a - div.x) * div.y * (T.c - div.z) + 
                   (T.a - div.x) * (T.b - div.y) * div.z
  let vol_white := (T.a - div.x) * (T.b - div.y) * (T.c - div.z) + 
                   (T.a - div.x) * div.y * div.z + 
                   div.x * (T.b - div.y) * div.z + 
                   div.x * div.y * (T.c - div.z)
  vol_black = vol_white → 
  div.x = T.a / 2 ∨ div.y = T.b / 2 ∨ div.z = T.c / 2 := by
  sorry

end NUMINAMATH_CALUDE_prism_division_theorem_l1411_141163


namespace NUMINAMATH_CALUDE_wage_increase_l1411_141183

theorem wage_increase (original_wage new_wage : ℝ) (increase_percentage : ℝ) :
  new_wage = original_wage * (1 + increase_percentage / 100) →
  increase_percentage = 30 →
  new_wage = 78 →
  original_wage = 60 := by
sorry

end NUMINAMATH_CALUDE_wage_increase_l1411_141183


namespace NUMINAMATH_CALUDE_min_speed_to_arrive_earlier_l1411_141116

/-- Proves the minimum speed required for the second person to arrive earlier -/
theorem min_speed_to_arrive_earlier
  (distance : ℝ)
  (speed_A : ℝ)
  (delay : ℝ)
  (h_distance : distance = 180)
  (h_speed_A : speed_A = 30)
  (h_delay : delay = 2) :
  ∀ speed_B : ℝ, speed_B > 45 →
    distance / speed_B + delay < distance / speed_A :=
by sorry

end NUMINAMATH_CALUDE_min_speed_to_arrive_earlier_l1411_141116


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1411_141101

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + (m-1)*x + 9 = (a*x + b)^2) → 
  (m = 7 ∨ m = -5) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1411_141101


namespace NUMINAMATH_CALUDE_three_doors_two_colors_l1411_141180

/-- The number of ways to paint a given number of doors with a given number of colors -/
def paintingWays (doors : ℕ) (colors : ℕ) : ℕ := colors ^ doors

/-- Theorem: The number of ways to paint 3 doors with 2 colors is 8 -/
theorem three_doors_two_colors : paintingWays 3 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_three_doors_two_colors_l1411_141180


namespace NUMINAMATH_CALUDE_balloon_expenses_l1411_141123

/-- The problem of calculating the total money Harry and Kevin brought to the store -/
theorem balloon_expenses (sheet_cost rope_cost propane_cost : ℕ)
  (helium_cost_per_oz : ℚ)
  (height_per_oz : ℕ)
  (max_height : ℕ) :
  sheet_cost = 42 →
  rope_cost = 18 →
  propane_cost = 14 →
  helium_cost_per_oz = 3/2 →
  height_per_oz = 113 →
  max_height = 9492 →
  ∃ (total_money : ℕ), total_money = 200 := by
  sorry

end NUMINAMATH_CALUDE_balloon_expenses_l1411_141123


namespace NUMINAMATH_CALUDE_gcd_lcm_multiple_relation_l1411_141170

theorem gcd_lcm_multiple_relation (x y z : ℤ) (h1 : y ≠ 0) (h2 : x / y = z) : 
  Int.gcd x y = y ∧ Int.lcm x y = x := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_multiple_relation_l1411_141170


namespace NUMINAMATH_CALUDE_max_sum_with_digit_constraints_l1411_141182

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a number is within a specific digit range -/
def is_n_digit (n : ℕ) (lower : ℕ) (upper : ℕ) : Prop :=
  lower ≤ n ∧ n ≤ upper

theorem max_sum_with_digit_constraints :
  ∃ (a b c : ℕ),
    is_n_digit a 10 99 ∧
    is_n_digit b 100 999 ∧
    is_n_digit c 1000 9999 ∧
    sum_of_digits (a + b) = 2 ∧
    sum_of_digits (b + c) = 2 ∧
    ∀ (x y z : ℕ),
      is_n_digit x 10 99 →
      is_n_digit y 100 999 →
      is_n_digit z 1000 9999 →
      sum_of_digits (x + y) = 2 →
      sum_of_digits (y + z) = 2 →
      x + y + z ≤ a + b + c ∧
      a + b + c = 10199 :=
sorry

end NUMINAMATH_CALUDE_max_sum_with_digit_constraints_l1411_141182


namespace NUMINAMATH_CALUDE_rectangular_field_equation_l1411_141150

theorem rectangular_field_equation (x : ℝ) : 
  (((60 - x) / 2) * ((60 + x) / 2) = 864) ↔ 
  (∃ (length width : ℝ), 
    length * width = 864 ∧ 
    length + width = 60 ∧ 
    length = width + x) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_equation_l1411_141150


namespace NUMINAMATH_CALUDE_square_plus_self_divisible_by_two_l1411_141191

theorem square_plus_self_divisible_by_two (n : ℤ) : ∃ k : ℤ, n^2 + n = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_square_plus_self_divisible_by_two_l1411_141191


namespace NUMINAMATH_CALUDE_fixed_points_for_specific_values_condition_for_two_fixed_points_minimum_b_value_l1411_141178

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 1

-- Define what it means for x to be a fixed point of f
def is_fixed_point (a b x : ℝ) : Prop := f a b x = x

-- Statement 1
theorem fixed_points_for_specific_values :
  is_fixed_point 1 3 (-2) ∧ is_fixed_point 1 3 (-1) :=
sorry

-- Statement 2
theorem condition_for_two_fixed_points :
  (∀ b : ℝ, ∃ x y : ℝ, x ≠ y ∧ is_fixed_point a b x ∧ is_fixed_point a b y) ↔
  (0 < a ∧ a < 1) :=
sorry

-- Statement 3
theorem minimum_b_value (a : ℝ) (h : 0 < a ∧ a < 1) :
  let g (x : ℝ) := -x + (2 * a) / (5 * a^2 - 4 * a + 1)
  ∃ b x y : ℝ, x ≠ y ∧ 
    is_fixed_point a b x ∧ 
    is_fixed_point a b y ∧ 
    g ((x + y) / 2) = (x + y) / 2 ∧
    (∀ b' : ℝ, b' ≥ b) ∧
    b = -2 :=
sorry

end NUMINAMATH_CALUDE_fixed_points_for_specific_values_condition_for_two_fixed_points_minimum_b_value_l1411_141178


namespace NUMINAMATH_CALUDE_total_medals_1996_l1411_141115

def gold_medals : ℕ := 16
def silver_medals : ℕ := 22
def bronze_medals : ℕ := 12

theorem total_medals_1996 : gold_medals + silver_medals + bronze_medals = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_medals_1996_l1411_141115


namespace NUMINAMATH_CALUDE_books_left_l1411_141195

theorem books_left (initial_books sold_books : ℝ) 
  (h1 : initial_books = 51.5)
  (h2 : sold_books = 45.75) : 
  initial_books - sold_books = 5.75 := by
  sorry

end NUMINAMATH_CALUDE_books_left_l1411_141195


namespace NUMINAMATH_CALUDE_position_of_2015_l1411_141104

/-- Represents a digit in the base-6 number system -/
inductive Digit : Type
| zero : Digit
| one : Digit
| two : Digit
| three : Digit
| four : Digit
| five : Digit

/-- Converts a base-6 number to its decimal equivalent -/
def toDecimal (n : List Digit) : Nat :=
  sorry

/-- Checks if a number is representable in base-6 using digits 0-5 -/
def isValidBase6 (n : Nat) : Prop :=
  sorry

/-- The sequence of numbers formed by digits 0-5 in ascending order -/
def base6Sequence : List Nat :=
  sorry

/-- The position of a number in the base6Sequence -/
def positionInSequence (n : Nat) : Nat :=
  sorry

/-- Theorem: The position of 2015 in the base-6 sequence is 443 -/
theorem position_of_2015 : positionInSequence 2015 = 443 :=
  sorry

end NUMINAMATH_CALUDE_position_of_2015_l1411_141104


namespace NUMINAMATH_CALUDE_sequence_with_least_period_l1411_141147

theorem sequence_with_least_period (p : ℕ) (h : p ≥ 2) :
  ∃ (x : ℕ → ℝ), 
    (∀ n, x (n + p) = x n) ∧ 
    (∀ n, x (n + 1) = x n - 1 / x n) ∧
    (∀ k, k < p → ¬(∀ n, x (n + k) = x n)) := by
  sorry

end NUMINAMATH_CALUDE_sequence_with_least_period_l1411_141147


namespace NUMINAMATH_CALUDE_factors_of_81_l1411_141139

theorem factors_of_81 : Finset.card (Nat.divisors 81) = 5 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_81_l1411_141139


namespace NUMINAMATH_CALUDE_find_k_value_l1411_141128

/-- Given two functions f and g, prove that k = 27/25 when f(5) - g(5) = 45 -/
theorem find_k_value (f g : ℝ → ℝ) (k : ℝ) 
    (hf : ∀ x, f x = 2*x^3 - 5*x^2 + 3*x + 7)
    (hg : ∀ x, g x = 3*x^3 - k*x^2 + 4)
    (h_diff : f 5 - g 5 = 45) : 
  k = 27/25 := by
sorry

end NUMINAMATH_CALUDE_find_k_value_l1411_141128


namespace NUMINAMATH_CALUDE_pants_price_satisfies_conditions_l1411_141185

/-- The original price of pants that satisfies the given conditions -/
def original_pants_price : ℝ := 110

/-- The number of pairs of pants purchased -/
def num_pants : ℕ := 4

/-- The number of pairs of socks purchased -/
def num_socks : ℕ := 2

/-- The original price of socks -/
def original_socks_price : ℝ := 60

/-- The discount rate applied to all items -/
def discount_rate : ℝ := 0.3

/-- The total cost after discount -/
def total_cost_after_discount : ℝ := 392

/-- Theorem stating that the original pants price satisfies the given conditions -/
theorem pants_price_satisfies_conditions :
  (num_pants : ℝ) * original_pants_price * (1 - discount_rate) +
  (num_socks : ℝ) * original_socks_price * (1 - discount_rate) =
  total_cost_after_discount := by sorry

end NUMINAMATH_CALUDE_pants_price_satisfies_conditions_l1411_141185


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l1411_141186

theorem max_value_trig_expression (a b : ℝ) :
  (∀ θ : ℝ, a * Real.cos (2 * θ) + b * Real.sin (2 * θ) ≤ Real.sqrt (a^2 + b^2)) ∧
  (∃ θ : ℝ, a * Real.cos (2 * θ) + b * Real.sin (2 * θ) = Real.sqrt (a^2 + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l1411_141186


namespace NUMINAMATH_CALUDE_intersection_with_complement_l1411_141160

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3, 4}

theorem intersection_with_complement :
  A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1411_141160


namespace NUMINAMATH_CALUDE_base_k_equals_seven_l1411_141144

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- Converts a number from base k to base 10 -/
def baseKToBase10 (n : Nat) (k : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * k^2 + tens * k^1 + ones * k^0

theorem base_k_equals_seven :
  ∃ k : Nat, base8ToBase10 524 = baseKToBase10 664 k ∧ k = 7 := by sorry

end NUMINAMATH_CALUDE_base_k_equals_seven_l1411_141144


namespace NUMINAMATH_CALUDE_first_nonzero_digit_of_1_over_143_l1411_141168

theorem first_nonzero_digit_of_1_over_143 : ∃ (n : ℕ) (d : ℕ), 
  (1 : ℚ) / 143 = (n : ℚ) / 10^d ∧ 
  n % 10 = 7 ∧ 
  ∀ (m : ℕ), m < d → (1 : ℚ) / 143 * 10^m < 1 :=
sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_of_1_over_143_l1411_141168


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l1411_141109

theorem value_of_a_minus_b (a b : ℝ) : (a - 5)^2 + |b^3 - 27| = 0 → a - b = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l1411_141109


namespace NUMINAMATH_CALUDE_six_people_lineup_permutations_l1411_141149

theorem six_people_lineup_permutations : Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_six_people_lineup_permutations_l1411_141149


namespace NUMINAMATH_CALUDE_no_real_graph_l1411_141102

/-- The equation x^2 + y^2 + 2x + 4y + 6 = 0 does not represent any real graph in the xy-plane. -/
theorem no_real_graph : ¬∃ (x y : ℝ), x^2 + y^2 + 2*x + 4*y + 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_graph_l1411_141102


namespace NUMINAMATH_CALUDE_paint_needed_paint_problem_l1411_141167

theorem paint_needed (initial_paint : ℚ) (day1_fraction : ℚ) (day2_fraction : ℚ) (additional_needed : ℚ) : ℚ :=
  let remaining_after_day1 := initial_paint - day1_fraction * initial_paint
  let remaining_after_day2 := remaining_after_day1 - day2_fraction * remaining_after_day1
  let total_needed := remaining_after_day2 + additional_needed
  total_needed - remaining_after_day2

theorem paint_problem : paint_needed 2 (1/4) (1/2) (1/2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_paint_needed_paint_problem_l1411_141167


namespace NUMINAMATH_CALUDE_lara_flowers_in_vase_l1411_141107

def flowers_in_vase (total_flowers mom_flowers grandma_extra : ℕ) : ℕ :=
  total_flowers - (mom_flowers + (mom_flowers + grandma_extra))

theorem lara_flowers_in_vase :
  flowers_in_vase 52 15 6 = 16 := by
  sorry

end NUMINAMATH_CALUDE_lara_flowers_in_vase_l1411_141107


namespace NUMINAMATH_CALUDE_perfect_power_l1411_141192

theorem perfect_power (M a b r : ℕ) (f : ℤ → ℤ) 
  (h_a : a ≥ 2) 
  (h_r : r ≥ 2) 
  (h_comp : ∀ n : ℤ, (f^[r]) n = a * n + b) 
  (h_nonneg : ∀ n : ℤ, n ≥ M → f n ≥ 0) 
  (h_div : ∀ n m : ℤ, n > m → m > M → (n - m) ∣ (f n - f m)) :
  ∃ k : ℕ, a = k^r := by
sorry

end NUMINAMATH_CALUDE_perfect_power_l1411_141192


namespace NUMINAMATH_CALUDE_combined_weight_equals_3655_574_l1411_141146

-- Define molar masses of elements
def mass_C : ℝ := 12.01
def mass_H : ℝ := 1.008
def mass_O : ℝ := 16.00
def mass_Na : ℝ := 22.99

-- Define molar masses of compounds
def mass_citric_acid : ℝ := 6 * mass_C + 8 * mass_H + 7 * mass_O
def mass_sodium_carbonate : ℝ := 2 * mass_Na + mass_C + 3 * mass_O
def mass_sodium_citrate : ℝ := 3 * mass_Na + 6 * mass_C + 5 * mass_H + 7 * mass_O
def mass_carbon_dioxide : ℝ := mass_C + 2 * mass_O
def mass_water : ℝ := 2 * mass_H + mass_O

-- Define number of moles for each substance
def moles_citric_acid : ℝ := 3
def moles_sodium_carbonate : ℝ := 4.5
def moles_sodium_citrate : ℝ := 9
def moles_carbon_dioxide : ℝ := 4.5
def moles_water : ℝ := 4.5

-- Theorem statement
theorem combined_weight_equals_3655_574 :
  moles_citric_acid * mass_citric_acid +
  moles_sodium_carbonate * mass_sodium_carbonate +
  moles_sodium_citrate * mass_sodium_citrate +
  moles_carbon_dioxide * mass_carbon_dioxide +
  moles_water * mass_water = 3655.574 := by
  sorry

end NUMINAMATH_CALUDE_combined_weight_equals_3655_574_l1411_141146


namespace NUMINAMATH_CALUDE_book_organizing_group_size_l1411_141120

/-- Represents the number of hours of work for one person to complete the task -/
def total_hours : ℕ := 40

/-- Represents the number of hours worked by the initial group -/
def initial_hours : ℕ := 2

/-- Represents the number of hours worked by the remaining group -/
def remaining_hours : ℕ := 4

/-- Represents the number of people who left the group -/
def people_left : ℕ := 2

theorem book_organizing_group_size :
  ∃ (initial_group : ℕ),
    (initial_hours : ℚ) / total_hours * initial_group + 
    (remaining_hours : ℚ) / total_hours * (initial_group - people_left) = 1 ∧
    initial_group = 8 := by
  sorry

end NUMINAMATH_CALUDE_book_organizing_group_size_l1411_141120


namespace NUMINAMATH_CALUDE_two_thirds_of_45_minus_7_l1411_141165

theorem two_thirds_of_45_minus_7 : (2 / 3 : ℚ) * 45 - 7 = 23 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_of_45_minus_7_l1411_141165


namespace NUMINAMATH_CALUDE_min_bailing_rate_is_14_l1411_141198

/-- Represents the scenario of Amy and Boris in the leaking boat --/
structure BoatScenario where
  distance_to_shore : Real
  water_intake_rate : Real
  sinking_threshold : Real
  initial_speed : Real
  speed_increase : Real
  speed_increase_interval : Real

/-- Calculates the time taken to reach the shore --/
def time_to_shore (scenario : BoatScenario) : Real :=
  sorry

/-- Calculates the total potential water intake --/
def total_water_intake (scenario : BoatScenario) (time : Real) : Real :=
  scenario.water_intake_rate * time

/-- Calculates the minimum bailing rate required --/
def min_bailing_rate (scenario : BoatScenario) : Real :=
  sorry

/-- The main theorem stating the minimum bailing rate for the given scenario --/
theorem min_bailing_rate_is_14 (scenario : BoatScenario) 
  (h1 : scenario.distance_to_shore = 2)
  (h2 : scenario.water_intake_rate = 15)
  (h3 : scenario.sinking_threshold = 50)
  (h4 : scenario.initial_speed = 2)
  (h5 : scenario.speed_increase = 1)
  (h6 : scenario.speed_increase_interval = 0.5) :
  min_bailing_rate scenario = 14 := by
  sorry

end NUMINAMATH_CALUDE_min_bailing_rate_is_14_l1411_141198


namespace NUMINAMATH_CALUDE_problem_solution_l1411_141187

theorem problem_solution (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : x*y + x + y = 5) : 
  x^2*y + x*y^2 + x^2 + y^2 = 18 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1411_141187


namespace NUMINAMATH_CALUDE_vector_not_parallel_implies_x_value_l1411_141184

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b (x : ℝ) : ℝ × ℝ := (4, x)

-- Define the condition that vectors are not parallel
def not_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 ≠ v.2 * w.1

-- Theorem statement
theorem vector_not_parallel_implies_x_value :
  ∃ x : ℝ, not_parallel a (b x) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_not_parallel_implies_x_value_l1411_141184


namespace NUMINAMATH_CALUDE_thirteen_percent_problem_l1411_141179

theorem thirteen_percent_problem : ∃ x : ℝ, 
  (13 / 100) * x = 85 ∧ 
  Int.floor (x + 0.5) = 654 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_percent_problem_l1411_141179


namespace NUMINAMATH_CALUDE_circle_center_and_equation_l1411_141119

/-- A circle passing through two points with a given radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  passesThrough : (ℝ × ℝ) → Prop

/-- The line passing through two points -/
def Line (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r : ℝ × ℝ | ∃ t : ℝ, r = (1 - t) • p + t • q}

theorem circle_center_and_equation 
  (C : Circle) 
  (h1 : C.passesThrough (1, 0)) 
  (h2 : C.passesThrough (0, 1)) 
  (h3 : C.radius = 1) : 
  (∃ t : ℝ, C.center = (t, t)) ∧ 
  (∀ x y : ℝ, C.passesThrough (x, y) ↔ x^2 + y^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_circle_center_and_equation_l1411_141119


namespace NUMINAMATH_CALUDE_largest_710_triple_l1411_141125

/-- Converts a natural number to its base-7 representation as a list of digits -/
def toBase7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 7) :: aux (m / 7)
  aux n |>.reverse

/-- Interprets a list of digits as a base-10 number -/
def fromDigits (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 10 * acc + d) 0

/-- Checks if a number is a 7-10 triple -/
def is710Triple (n : ℕ) : Prop :=
  fromDigits (toBase7 n) = 3 * n

/-- States that 1422 is the largest 7-10 triple -/
theorem largest_710_triple :
  is710Triple 1422 ∧ ∀ m : ℕ, m > 1422 → ¬is710Triple m :=
sorry

end NUMINAMATH_CALUDE_largest_710_triple_l1411_141125


namespace NUMINAMATH_CALUDE_sum_of_absolute_values_zero_l1411_141111

theorem sum_of_absolute_values_zero (a b : ℝ) :
  |a - 5| + |b + 8| = 0 → a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_values_zero_l1411_141111


namespace NUMINAMATH_CALUDE_quadratic_intersection_and_sum_of_y_l1411_141189

/-- Quadratic function -/
def f (a x : ℝ) : ℝ := a * x^2 - (2*a - 2) * x - 3*a - 1

theorem quadratic_intersection_and_sum_of_y (a : ℝ) (h1 : a > 0) :
  (∃! x, f a x = -3*a - 2) →
  a^2 + 1/a^2 = 7 ∧
  ∀ m n y1 y2 : ℝ, m ≠ n → m + n = -2 → 
    f a m = y1 → f a n = y2 → y1 + y2 > -6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_and_sum_of_y_l1411_141189


namespace NUMINAMATH_CALUDE_intercepted_arc_measure_l1411_141145

/-- An equilateral triangle with a circle rolling along its side -/
structure TriangleWithCircle where
  /-- Side length of the equilateral triangle -/
  side : ℝ
  /-- Radius of the circle (equal to the height of the triangle) -/
  radius : ℝ
  /-- The radius is equal to the height of the equilateral triangle -/
  height_eq_radius : radius = side * Real.sqrt 3 / 2

/-- The theorem stating that the intercepted arc measure is 60° -/
theorem intercepted_arc_measure (tc : TriangleWithCircle) :
  let arc_measure := Real.pi / 3  -- 60° in radians
  ∃ (center : ℝ × ℝ) (point_on_side : ℝ × ℝ),
    arc_measure = Real.arccos ((point_on_side.1 - center.1) / tc.radius) :=
sorry

end NUMINAMATH_CALUDE_intercepted_arc_measure_l1411_141145


namespace NUMINAMATH_CALUDE_bedbug_growth_proof_l1411_141100

def bedbug_population (initial_population : ℕ) (growth_factor : ℕ) (days : ℕ) : ℕ :=
  initial_population * growth_factor ^ days

theorem bedbug_growth_proof :
  bedbug_population 30 3 4 = 2430 := by
  sorry

end NUMINAMATH_CALUDE_bedbug_growth_proof_l1411_141100


namespace NUMINAMATH_CALUDE_ten_point_circle_triangles_l1411_141113

/-- Represents a circle with points and chords -/
structure CircleWithChords where
  numPoints : ℕ
  noTripleIntersection : Bool

/-- Calculates the number of triangles formed inside the circle -/
def trianglesInsideCircle (c : CircleWithChords) : ℕ :=
  sorry

/-- The main theorem stating that for 10 points on a circle with the given conditions,
    the number of triangles formed inside is 105 -/
theorem ten_point_circle_triangles :
  ∀ (c : CircleWithChords),
    c.numPoints = 10 →
    c.noTripleIntersection = true →
    trianglesInsideCircle c = 105 :=
by sorry

end NUMINAMATH_CALUDE_ten_point_circle_triangles_l1411_141113


namespace NUMINAMATH_CALUDE_three_lines_cannot_form_triangle_l1411_141137

/-- Three lines in the plane -/
structure ThreeLines where
  l1 : ℝ → ℝ → Prop
  l2 : ℝ → ℝ → Prop
  l3 : ℝ → ℝ → ℝ → Prop

/-- The condition that three lines cannot form a triangle -/
def cannotFormTriangle (lines : ThreeLines) (m : ℝ) : Prop :=
  (∃ (x y : ℝ), lines.l1 x y ∧ lines.l2 x y ∧ lines.l3 m x y) ∨
  (∃ (a b : ℝ), ∀ (x y : ℝ), (lines.l1 x y ↔ y = a*x + b) ∧ 
                              (lines.l3 m x y ↔ y = a*x + (1 - a*m)/m)) ∨
  (∃ (a b : ℝ), ∀ (x y : ℝ), (lines.l2 x y ↔ y = a*x + b) ∧ 
                              (lines.l3 m x y ↔ y = a*x + (1 + a*m)/m))

/-- The given lines -/
def givenLines : ThreeLines :=
  { l1 := λ x y => 2*x - 3*y + 1 = 0
  , l2 := λ x y => 4*x + 3*y + 5 = 0
  , l3 := λ m x y => m*x - y - 1 = 0 }

theorem three_lines_cannot_form_triangle :
  {m : ℝ | cannotFormTriangle givenLines m} = {-4/3, 2/3, 4/3} := by sorry

end NUMINAMATH_CALUDE_three_lines_cannot_form_triangle_l1411_141137


namespace NUMINAMATH_CALUDE_divisible_by_45_digits_l1411_141138

theorem divisible_by_45_digits (a b : Nat) : 
  a < 10 → b < 10 → (72000 + 100 * a + 30 + b) % 45 = 0 → 
  ((a = 6 ∧ b = 0) ∨ (a = 1 ∧ b = 5)) := by
sorry

end NUMINAMATH_CALUDE_divisible_by_45_digits_l1411_141138


namespace NUMINAMATH_CALUDE_intersection_M_N_l1411_141175

def M : Set ℤ := {-2, 0, 2}
def N : Set ℤ := {x | x^2 = x}

theorem intersection_M_N : M ∩ N = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1411_141175


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l1411_141169

/-- A rectangle inscribed in a semicircle -/
structure InscribedRectangle where
  /-- The length of side PR of the rectangle -/
  pr : ℝ
  /-- The length of PG and SH, which are equal -/
  pg : ℝ
  /-- Assumption that PR is positive -/
  pr_pos : pr > 0
  /-- Assumption that PG is positive -/
  pg_pos : pg > 0

/-- The theorem stating that the area of the inscribed rectangle is 160√6 -/
theorem inscribed_rectangle_area (rect : InscribedRectangle) 
  (h1 : rect.pr = 20) (h2 : rect.pg = 12) : 
  ∃ (area : ℝ), area = rect.pr * Real.sqrt (rect.pg * (rect.pr + 2 * rect.pg - rect.pg)) ∧ 
  area = 160 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l1411_141169


namespace NUMINAMATH_CALUDE_half_squared_is_quarter_l1411_141161

theorem half_squared_is_quarter : (1/2)^2 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_half_squared_is_quarter_l1411_141161


namespace NUMINAMATH_CALUDE_order_of_powers_l1411_141152

theorem order_of_powers : 
  let a : ℝ := (2/5: ℝ)^(3/5: ℝ)
  let b : ℝ := (2/5: ℝ)^(2/5: ℝ)
  let c : ℝ := (3/5: ℝ)^(2/5: ℝ)
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_order_of_powers_l1411_141152


namespace NUMINAMATH_CALUDE_remainder_theorem_example_l1411_141177

theorem remainder_theorem_example (x : ℤ) :
  (Polynomial.X ^ 9 + 3 : Polynomial ℤ).eval 2 = 515 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_example_l1411_141177


namespace NUMINAMATH_CALUDE_largest_four_digit_with_property_l1411_141181

/-- Represents a four-digit number as a tuple of its digits -/
def FourDigitNumber := (Nat × Nat × Nat × Nat)

/-- Checks if a four-digit number satisfies the property that each of the last two digits
    is equal to the sum of the two preceding digits -/
def satisfiesProperty (n : FourDigitNumber) : Prop :=
  let (a, b, c, d) := n
  c = a + b ∧ d = b + c

/-- Converts a four-digit number tuple to its numerical value -/
def toNumber (n : FourDigitNumber) : Nat :=
  let (a, b, c, d) := n
  1000 * a + 100 * b + 10 * c + d

/-- The target number 9099 -/
def target : FourDigitNumber := (9, 0, 9, 9)

theorem largest_four_digit_with_property :
  satisfiesProperty target ∧
  ∀ n : FourDigitNumber, satisfiesProperty n → toNumber n ≤ toNumber target :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_with_property_l1411_141181


namespace NUMINAMATH_CALUDE_ben_is_25_l1411_141166

/-- Ben's age -/
def ben_age : ℕ := sorry

/-- Dan's age -/
def dan_age : ℕ := sorry

/-- Ben is 3 years younger than Dan -/
axiom age_difference : ben_age = dan_age - 3

/-- The sum of their ages is 53 -/
axiom age_sum : ben_age + dan_age = 53

theorem ben_is_25 : ben_age = 25 := by sorry

end NUMINAMATH_CALUDE_ben_is_25_l1411_141166


namespace NUMINAMATH_CALUDE_pet_store_cats_l1411_141135

theorem pet_store_cats (siamese_cats : ℕ) (cats_sold : ℕ) (cats_left : ℕ) (house_cats : ℕ) : 
  siamese_cats = 38 → 
  cats_sold = 45 → 
  cats_left = 18 → 
  siamese_cats + house_cats - cats_sold = cats_left → 
  house_cats = 25 := by
sorry

end NUMINAMATH_CALUDE_pet_store_cats_l1411_141135


namespace NUMINAMATH_CALUDE_min_value_of_sum_l1411_141103

theorem min_value_of_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2*x + 8*y - x*y = 0) :
  x + y ≥ 18 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 8*y₀ - x₀*y₀ = 0 ∧ x₀ + y₀ = 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l1411_141103


namespace NUMINAMATH_CALUDE_sons_age_l1411_141194

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 29 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 27 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l1411_141194


namespace NUMINAMATH_CALUDE_injective_function_characterization_l1411_141151

theorem injective_function_characterization (f : ℤ → ℤ) :
  Function.Injective f ∧ (∀ x y : ℤ, |f x - f y| ≤ |x - y|) →
  ∃ a : ℤ, (∀ x : ℤ, f x = a + x) ∨ (∀ x : ℤ, f x = a - x) :=
by sorry

end NUMINAMATH_CALUDE_injective_function_characterization_l1411_141151


namespace NUMINAMATH_CALUDE_negation_of_all_divisible_by_two_are_even_l1411_141158

theorem negation_of_all_divisible_by_two_are_even :
  (¬ ∀ n : ℤ, n % 2 = 0 → Even n) ↔ (∃ n : ℤ, n % 2 = 0 ∧ ¬ Even n) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_divisible_by_two_are_even_l1411_141158


namespace NUMINAMATH_CALUDE_field_trip_total_l1411_141126

theorem field_trip_total (num_vans num_buses students_per_van students_per_bus teachers_per_van teachers_per_bus : ℕ) 
  (h1 : num_vans = 6)
  (h2 : num_buses = 8)
  (h3 : students_per_van = 6)
  (h4 : students_per_bus = 18)
  (h5 : teachers_per_van = 1)
  (h6 : teachers_per_bus = 2) :
  num_vans * students_per_van + num_buses * students_per_bus + 
  num_vans * teachers_per_van + num_buses * teachers_per_bus = 202 :=
by sorry

end NUMINAMATH_CALUDE_field_trip_total_l1411_141126


namespace NUMINAMATH_CALUDE_prize_distribution_l1411_141196

theorem prize_distribution (total_winners : ℕ) (min_award : ℚ) (max_award : ℚ) :
  total_winners = 20 →
  min_award = 20 →
  max_award = 160 →
  ∃ (total_prize : ℚ),
    total_prize > 0 ∧
    (2 / 5 : ℚ) * total_prize = max_award ∧
    (∀ (winner : ℕ), winner ≤ total_winners → ∃ (award : ℚ), min_award ≤ award ∧ award ≤ max_award) ∧
    total_prize = 1000 :=
by sorry

end NUMINAMATH_CALUDE_prize_distribution_l1411_141196


namespace NUMINAMATH_CALUDE_equal_after_adjustments_l1411_141110

/-- The number of adjustments needed to equalize the number of boys and girls -/
def num_adjustments : ℕ := 8

/-- The initial number of boys -/
def initial_boys : ℕ := 40

/-- The initial number of girls -/
def initial_girls : ℕ := 0

/-- The number of boys reduced in each adjustment -/
def boys_reduction : ℕ := 3

/-- The number of girls increased in each adjustment -/
def girls_increase : ℕ := 2

/-- Calculates the number of boys after a given number of adjustments -/
def boys_after (n : ℕ) : ℤ :=
  initial_boys - n * boys_reduction

/-- Calculates the number of girls after a given number of adjustments -/
def girls_after (n : ℕ) : ℤ :=
  initial_girls + n * girls_increase

theorem equal_after_adjustments :
  boys_after num_adjustments = girls_after num_adjustments := by
  sorry

end NUMINAMATH_CALUDE_equal_after_adjustments_l1411_141110


namespace NUMINAMATH_CALUDE_unique_a_value_l1411_141157

def U : Set ℤ := {-5, -3, 1, 2, 3, 4, 5, 6}

def A : Set ℤ := {x | x^2 - 7*x + 12 = 0}

def B (a : ℤ) : Set ℤ := {a^2, 2*a - 1, 6}

theorem unique_a_value : 
  ∃! a : ℤ, A ∩ B a = {4} ∧ B a ⊆ U ∧ a = -2 :=
sorry

end NUMINAMATH_CALUDE_unique_a_value_l1411_141157


namespace NUMINAMATH_CALUDE_simplify_fraction_l1411_141159

theorem simplify_fraction : (75 : ℚ) / 225 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1411_141159


namespace NUMINAMATH_CALUDE_hexagon_parallelogram_theorem_l1411_141155

-- Define a point in 2D space
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a hexagon as a collection of 6 points
structure Hexagon :=
  (A B C D E F : Point)

-- Define a property for convex hexagon
def is_convex (h : Hexagon) : Prop := sorry

-- Define a quadrilateral as a collection of 4 points
structure Quadrilateral :=
  (P Q R S : Point)

-- Define a property for parallelogram
def is_parallelogram (q : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem hexagon_parallelogram_theorem (h : Hexagon) 
  (convex_h : is_convex h)
  (para_ABDE : is_parallelogram ⟨h.A, h.B, h.D, h.E⟩)
  (para_ACDF : is_parallelogram ⟨h.A, h.C, h.D, h.F⟩) :
  is_parallelogram ⟨h.B, h.C, h.E, h.F⟩ := by
  sorry

end NUMINAMATH_CALUDE_hexagon_parallelogram_theorem_l1411_141155


namespace NUMINAMATH_CALUDE_cheerleaders_size2_l1411_141193

/-- The number of cheerleaders needing size 2 uniforms -/
def size2 (total size6 : ℕ) : ℕ :=
  total - (size6 + size6 / 2)

/-- Theorem stating that 4 cheerleaders need size 2 uniforms -/
theorem cheerleaders_size2 :
  size2 19 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_cheerleaders_size2_l1411_141193


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l1411_141172

theorem quadratic_solution_sum (x y : ℝ) : 
  x + y = 5 → x * y = 7/5 →
  ∃ (a b c d : ℝ), x = (a + b * Real.sqrt c) / d ∧ 
                    x = (a - b * Real.sqrt c) / d ∧
                    a + b + c + d = 521 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l1411_141172


namespace NUMINAMATH_CALUDE_no_solution_condition_l1411_141173

theorem no_solution_condition (a : ℝ) : 
  (∀ x : ℝ, 8 * |x - 4*a| + |x - a^2| + 7*x - 2*a ≠ 0) ↔ (a < -22 ∨ a > 0) :=
sorry

end NUMINAMATH_CALUDE_no_solution_condition_l1411_141173


namespace NUMINAMATH_CALUDE_local_min_condition_l1411_141199

/-- The function f(x) defined in terms of parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := (x - 2*a) * (x^2 + a^2*x + 2*a^3)

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*(a^2 - 2*a)*x

/-- Theorem stating the condition for x = 0 to be a local minimum of f -/
theorem local_min_condition (a : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs x < δ → f a x ≥ f a 0) ↔ a < 0 ∨ a > 2 :=
sorry

end NUMINAMATH_CALUDE_local_min_condition_l1411_141199


namespace NUMINAMATH_CALUDE_mans_speed_in_still_water_l1411_141134

/-- Proves that given a man who swims downstream 36 km in 6 hours and upstream 18 km in 6 hours, his speed in still water is 4.5 km/h. -/
theorem mans_speed_in_still_water 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (time : ℝ) 
  (h1 : downstream_distance = 36) 
  (h2 : upstream_distance = 18) 
  (h3 : time = 6) : 
  ∃ (speed_still_water : ℝ) (stream_speed : ℝ),
    speed_still_water + stream_speed = downstream_distance / time ∧ 
    speed_still_water - stream_speed = upstream_distance / time ∧
    speed_still_water = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_mans_speed_in_still_water_l1411_141134


namespace NUMINAMATH_CALUDE_pool_filling_time_l1411_141117

theorem pool_filling_time (faster_pipe_rate : ℝ) (slower_pipe_rate : ℝ) :
  faster_pipe_rate = 1 / 9 →
  slower_pipe_rate = faster_pipe_rate / 1.25 →
  1 / (faster_pipe_rate + slower_pipe_rate) = 5 := by
  sorry

end NUMINAMATH_CALUDE_pool_filling_time_l1411_141117


namespace NUMINAMATH_CALUDE_contrapositive_x_squared_greater_than_one_l1411_141129

theorem contrapositive_x_squared_greater_than_one (x : ℝ) : 
  x ≤ 1 → x^2 ≤ 1 := by sorry

end NUMINAMATH_CALUDE_contrapositive_x_squared_greater_than_one_l1411_141129


namespace NUMINAMATH_CALUDE_ellipse_slope_bound_l1411_141141

/-- Given an ellipse with equation x²/a² + y²/b² = 1 where a > b > 0,
    and points A(-a, 0), B(a, 0), and P(x, y) on the ellipse such that
    P ≠ A, P ≠ B, and |AP| = |OA|, prove that the absolute value of
    the slope of line OP is greater than √3. -/
theorem ellipse_slope_bound (a b x y : ℝ) :
  a > b ∧ b > 0 ∧
  x^2 / a^2 + y^2 / b^2 = 1 ∧
  (x ≠ -a ∨ y ≠ 0) ∧ (x ≠ a ∨ y ≠ 0) ∧
  (x + a)^2 + y^2 = 4 * a^2 →
  abs (y / x) > Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_slope_bound_l1411_141141


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_36_l1411_141162

def is_divisible_by_36 (n : ℕ) : Prop := n % 36 = 0

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def form_number (a b : ℕ) : ℕ := 90000 + 1000 * a + 650 + b

theorem five_digit_divisible_by_36 :
  ∀ a b : ℕ,
    is_single_digit a →
    is_single_digit b →
    is_divisible_by_36 (form_number a b) →
    ((a = 5 ∧ b = 2) ∨ (a = 1 ∧ b = 6)) :=
sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_36_l1411_141162


namespace NUMINAMATH_CALUDE_inequality_solution_l1411_141143

theorem inequality_solution (p q r : ℝ) 
  (h1 : ∀ x : ℝ, (x - p) * (x - q) / (x - r) ≤ 0 ↔ x < -1 ∨ |x - 30| ≤ 2)
  (h2 : p < q) : 
  p + 2*q + 3*r = 89 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1411_141143


namespace NUMINAMATH_CALUDE_content_paths_count_l1411_141176

/-- Represents the grid structure of the "CONTENT" word pattern --/
def ContentGrid : Type := Unit  -- Placeholder for the grid structure

/-- Represents a valid path in the ContentGrid --/
def ValidPath (grid : ContentGrid) : Type := Unit  -- Placeholder for path representation

/-- Counts the number of valid paths in the ContentGrid --/
def countValidPaths (grid : ContentGrid) : ℕ := sorry

/-- The main theorem stating that the number of valid paths is 127 --/
theorem content_paths_count (grid : ContentGrid) : countValidPaths grid = 127 := by
  sorry

end NUMINAMATH_CALUDE_content_paths_count_l1411_141176


namespace NUMINAMATH_CALUDE_sqrt_expression_simplification_l1411_141118

theorem sqrt_expression_simplification :
  (Real.sqrt 48 + Real.sqrt 20) - (Real.sqrt 12 - Real.sqrt 5) = 2 * Real.sqrt 3 + 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_simplification_l1411_141118


namespace NUMINAMATH_CALUDE_smallest_solution_for_floor_equation_l1411_141106

theorem smallest_solution_for_floor_equation :
  ∃ (x : ℝ), x > 0 ∧ 
  (⌊x^2⌋ : ℝ) - x * (⌊x⌋ : ℝ) = 10 ∧
  ∀ (y : ℝ), y > 0 → (⌊y^2⌋ : ℝ) - y * (⌊y⌋ : ℝ) = 10 → y ≥ x :=
by
  use 131 / 11
  sorry

end NUMINAMATH_CALUDE_smallest_solution_for_floor_equation_l1411_141106


namespace NUMINAMATH_CALUDE_increasing_geometric_sequence_formula_l1411_141174

/-- An increasing geometric sequence with specific properties -/
def IncreasingGeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧
  (a 5)^2 = a 10 ∧
  (∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1))

/-- The general term formula for the sequence -/
def GeneralTermFormula (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = 2^n

/-- Theorem stating that an increasing geometric sequence with the given properties
    has the general term formula a_n = 2^n -/
theorem increasing_geometric_sequence_formula (a : ℕ → ℝ) :
  IncreasingGeometricSequence a → GeneralTermFormula a := by
  sorry

end NUMINAMATH_CALUDE_increasing_geometric_sequence_formula_l1411_141174


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1411_141136

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = Complex.abs (1 - Complex.I)) :
  z.im = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1411_141136


namespace NUMINAMATH_CALUDE_problem_solution_l1411_141122

theorem problem_solution (x y : ℝ) (some_number : ℝ) 
  (h1 : x + 3 * y = some_number) 
  (h2 : y = 10) 
  (h3 : x = 3) : 
  some_number = 33 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1411_141122


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1411_141108

/-- Given three mutually externally tangent circles with radii a, b, and c,
    the radius r of the inscribed circle satisfies the equation:
    1/r = 1/a + 1/b + 1/c + 2 * sqrt(1/(a*b) + 1/(a*c) + 1/(b*c)) -/
theorem inscribed_circle_radius (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let r := (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))⁻¹
  a = 5 → b = 10 → c = 20 → r = 20 / (3.5 + 2 * Real.sqrt 14) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1411_141108


namespace NUMINAMATH_CALUDE_min_value_and_integer_solutions_l1411_141142

theorem min_value_and_integer_solutions (x y : ℝ) : 
  x + y + 2*x*y = 5 →
  (∀ (x y : ℝ), x > 0 ∧ y > 0 → x + y ≥ Real.sqrt 11 - 1) ∧
  (∃ (x y : ℤ), x + y + 2*x*y = 5 ∧ (x + y = 5 ∨ x + y = -7)) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_integer_solutions_l1411_141142


namespace NUMINAMATH_CALUDE_smallest_n_for_252_terms_l1411_141131

def count_terms (n : ℕ) : ℕ := Nat.choose n 5

theorem smallest_n_for_252_terms : 
  (∀ k < 10, count_terms k ≠ 252) ∧ count_terms 10 = 252 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_252_terms_l1411_141131


namespace NUMINAMATH_CALUDE_complex_power_sum_l1411_141121

theorem complex_power_sum (z : ℂ) (h : z^2 - z + 1 = 0) : 
  z^98 + z^99 + z^100 + z^101 + z^102 = -z := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1411_141121


namespace NUMINAMATH_CALUDE_man_walking_rest_distance_l1411_141130

/-- Proves that a man walking at 10 mph, resting for 8 minutes after every d miles,
    and taking 332 minutes to walk 50 miles, rests after every 10 miles. -/
theorem man_walking_rest_distance (d : ℝ) : 
  (10 : ℝ) = d → -- walking speed in mph
  (8 : ℝ) = 8 → -- rest duration in minutes
  (332 : ℝ) = 332 → -- total time in minutes
  (50 : ℝ) = 50 → -- total distance in miles
  (300 : ℝ) + (50 / d - 1) * 8 = 332 → -- time equation
  d = 10 := by
sorry

end NUMINAMATH_CALUDE_man_walking_rest_distance_l1411_141130


namespace NUMINAMATH_CALUDE_parabola_intersection_l1411_141133

-- Define the two parabolas
def f (x : ℝ) : ℝ := 3 * x^2 - 9 * x - 18
def g (x : ℝ) : ℝ := x^2 - 2 * x + 4

-- Define the intersection points
def p₁ : ℝ × ℝ := (-2, 12)
def p₂ : ℝ × ℝ := (5.5, 23.25)

-- Theorem statement
theorem parabola_intersection :
  (∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x, y) = p₁ ∨ (x, y) = p₂) := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1411_141133


namespace NUMINAMATH_CALUDE_subset_M_proof_l1411_141124

def M : Set ℝ := {x | x ≤ 2 * Real.sqrt 3}

theorem subset_M_proof (b : ℝ) (hb : b ∈ Set.Ioo 0 1) :
  {Real.sqrt (11 + b)} ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_subset_M_proof_l1411_141124
