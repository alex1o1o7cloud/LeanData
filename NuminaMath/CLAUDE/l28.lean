import Mathlib

namespace NUMINAMATH_CALUDE_dealership_sales_forecast_l28_2863

theorem dealership_sales_forecast (sports_cars sedan_cars : ℕ) : 
  (5 : ℚ) / 8 = sports_cars / sedan_cars →
  sports_cars = 35 →
  sedan_cars = 56 := by
sorry

end NUMINAMATH_CALUDE_dealership_sales_forecast_l28_2863


namespace NUMINAMATH_CALUDE_quadratic_function_range_l28_2865

/-- Given a quadratic function f(x) = ax² - c satisfying certain conditions,
    prove that f(3) is within a specific range. -/
theorem quadratic_function_range (a c : ℝ) (f : ℝ → ℝ) 
    (h_def : ∀ x, f x = a * x^2 - c)
    (h_1 : -4 ≤ f 1 ∧ f 1 ≤ -1)
    (h_2 : -1 ≤ f 2 ∧ f 2 ≤ 5) :
  -1 ≤ f 3 ∧ f 3 ≤ 20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l28_2865


namespace NUMINAMATH_CALUDE_percentage_relation_l28_2845

theorem percentage_relation (T S F : ℝ) 
  (h1 : F = 0.06 * T) 
  (h2 : F = (1/3) * S) : 
  S = 0.18 * T := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l28_2845


namespace NUMINAMATH_CALUDE_nail_container_problem_l28_2888

theorem nail_container_problem (N : ℝ) : 
  (N > 0) →
  (0.7 * N - 0.7 * (0.7 * N) = 84) →
  N = 400 := by
sorry

end NUMINAMATH_CALUDE_nail_container_problem_l28_2888


namespace NUMINAMATH_CALUDE_sequence_increasing_l28_2802

def a (n : ℕ+) : ℚ := (2 * n) / (2 * n + 1)

theorem sequence_increasing (n : ℕ+) : a n < a (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_increasing_l28_2802


namespace NUMINAMATH_CALUDE_solve_for_q_l28_2810

theorem solve_for_q (x y q : ℚ) 
  (h1 : (7 : ℚ) / 8 = x / 96)
  (h2 : (7 : ℚ) / 8 = (x + y) / 104)
  (h3 : (7 : ℚ) / 8 = (q - y) / 144) :
  q = 133 := by sorry

end NUMINAMATH_CALUDE_solve_for_q_l28_2810


namespace NUMINAMATH_CALUDE_polygon_angle_sum_l28_2884

theorem polygon_angle_sum (n : ℕ) : 
  (n - 2) * 180 = 2 * 360 ↔ n = 6 :=
sorry

end NUMINAMATH_CALUDE_polygon_angle_sum_l28_2884


namespace NUMINAMATH_CALUDE_power_of_product_l28_2885

theorem power_of_product (a b : ℝ) : (a^2 * b)^3 = a^6 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l28_2885


namespace NUMINAMATH_CALUDE_min_product_of_three_numbers_l28_2860

theorem min_product_of_three_numbers (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x + y + z = 1 → 
  x ≤ 3*y ∧ x ≤ 3*z ∧ y ≤ 3*x ∧ y ≤ 3*z ∧ z ≤ 3*x ∧ z ≤ 3*y → 
  x * y * z ≥ 1/18 := by
  sorry

end NUMINAMATH_CALUDE_min_product_of_three_numbers_l28_2860


namespace NUMINAMATH_CALUDE_cricket_average_score_l28_2819

theorem cricket_average_score 
  (avg_2_matches : ℝ) 
  (avg_5_matches : ℝ) 
  (num_matches : ℕ) 
  (h1 : avg_2_matches = 20) 
  (h2 : avg_5_matches = 26) 
  (h3 : num_matches = 5) :
  let remaining_matches := num_matches - 2
  let total_score_5 := avg_5_matches * num_matches
  let total_score_2 := avg_2_matches * 2
  let remaining_score := total_score_5 - total_score_2
  remaining_score / remaining_matches = 30 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_score_l28_2819


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l28_2868

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + (1/2 : ℝ) ≤ 0) ↔ -1 < a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l28_2868


namespace NUMINAMATH_CALUDE_dance_camp_rabbits_l28_2814

theorem dance_camp_rabbits :
  ∀ (R S : ℕ),
  R + S = 50 →
  4 * R + 8 * S = 2 * R + 16 * S →
  R = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_dance_camp_rabbits_l28_2814


namespace NUMINAMATH_CALUDE_laran_weekly_profit_l28_2837

/-- Calculates the profit for Laran's poster business over a 5-day school week --/
def calculate_profit (
  total_posters_per_day : ℕ)
  (large_posters_per_day : ℕ)
  (large_poster_price : ℚ)
  (large_poster_tax_rate : ℚ)
  (large_poster_cost : ℚ)
  (small_poster_price : ℚ)
  (small_poster_tax_rate : ℚ)
  (small_poster_cost : ℚ)
  (fixed_weekly_expense : ℚ)
  (days_per_week : ℕ) : ℚ :=
  sorry

/-- Theorem stating that Laran's weekly profit is $98.50 --/
theorem laran_weekly_profit :
  calculate_profit 5 2 10 (1/10) 5 6 (3/20) 3 20 5 = 197/2 :=
  sorry

end NUMINAMATH_CALUDE_laran_weekly_profit_l28_2837


namespace NUMINAMATH_CALUDE_largest_five_digit_with_given_product_l28_2867

/-- The product of the digits of a natural number -/
def digit_product (n : ℕ) : ℕ := sorry

/-- Check if a number is a five-digit integer -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem largest_five_digit_with_given_product :
  (∀ n : ℕ, is_five_digit n ∧ digit_product n = 40320 → n ≤ 98752) ∧
  is_five_digit 98752 ∧
  digit_product 98752 = 40320 := by sorry

end NUMINAMATH_CALUDE_largest_five_digit_with_given_product_l28_2867


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l28_2841

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => geometric_sequence a q n * q

theorem geometric_sequence_sum (a q : ℝ) :
  let seq := geometric_sequence a q
  (seq 0 + seq 1 = 2) →
  (seq 4 + seq 5 = 4) →
  (seq 8 + seq 9 = 8) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l28_2841


namespace NUMINAMATH_CALUDE_gcd_of_42_77_105_l28_2852

theorem gcd_of_42_77_105 : Nat.gcd 42 (Nat.gcd 77 105) = 7 := by sorry

end NUMINAMATH_CALUDE_gcd_of_42_77_105_l28_2852


namespace NUMINAMATH_CALUDE_john_paid_21_dollars_l28_2850

/-- Calculates the amount John paid for candy bars -/
def john_payment (total_bars : ℕ) (dave_bars : ℕ) (cost_per_bar : ℚ) : ℚ :=
  (total_bars - dave_bars) * cost_per_bar

/-- Proves that John paid $21 for the candy bars -/
theorem john_paid_21_dollars (total_bars : ℕ) (dave_bars : ℕ) (cost_per_bar : ℚ)
  (h1 : total_bars = 20)
  (h2 : dave_bars = 6)
  (h3 : cost_per_bar = 3/2) :
  john_payment total_bars dave_bars cost_per_bar = 21 := by
  sorry

end NUMINAMATH_CALUDE_john_paid_21_dollars_l28_2850


namespace NUMINAMATH_CALUDE_problem_solution_l28_2828

theorem problem_solution : ∃ x : ℕ, x = 13 ∧ (4 * x) / 8 = 6 ∧ (4 * x) % 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l28_2828


namespace NUMINAMATH_CALUDE_quadratic_solution_positive_l28_2843

theorem quadratic_solution_positive (x : ℝ) : 
  x > 0 ∧ 4 * x^2 + 8 * x - 20 = 0 ↔ x = Real.sqrt 6 - 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_positive_l28_2843


namespace NUMINAMATH_CALUDE_amoeba_count_after_week_l28_2895

/-- The number of amoebas after n days, given an initial population of 1 and each amoeba splitting into two every day. -/
def amoeba_count (n : ℕ) : ℕ := 2^n

/-- Theorem stating that the number of amoebas after 7 days is 128. -/
theorem amoeba_count_after_week : amoeba_count 7 = 128 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_count_after_week_l28_2895


namespace NUMINAMATH_CALUDE_pencil_count_l28_2838

/-- The total number of pencils in the drawer after Sarah's addition -/
def total_pencils (initial : ℕ) (mike_added : ℕ) (sarah_added : ℕ) : ℕ :=
  initial + mike_added + sarah_added

/-- Theorem stating the total number of pencils after all additions -/
theorem pencil_count (x : ℕ) :
  total_pencils 41 30 x = 71 + x := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l28_2838


namespace NUMINAMATH_CALUDE_library_visitors_average_l28_2813

theorem library_visitors_average (sunday_visitors : ℕ) (other_day_visitors : ℕ) 
  (days_in_month : ℕ) (h1 : sunday_visitors = 140) (h2 : other_day_visitors = 80) 
  (h3 : days_in_month = 30) :
  let sundays : ℕ := (days_in_month + 6) / 7
  let other_days : ℕ := days_in_month - sundays
  let total_visitors : ℕ := sundays * sunday_visitors + other_days * other_day_visitors
  (total_visitors : ℚ) / days_in_month = 88 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_average_l28_2813


namespace NUMINAMATH_CALUDE_cube_root_function_l28_2848

theorem cube_root_function (k : ℝ) :
  (∀ x, x > 0 → ∃ y, y = k * x^(1/3)) →
  (4 * Real.sqrt 3 = k * 64^(1/3)) →
  (2 * Real.sqrt 3 = k * 8^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_function_l28_2848


namespace NUMINAMATH_CALUDE_probability_three_heads_in_eight_tosses_l28_2816

theorem probability_three_heads_in_eight_tosses (n : Nat) (k : Nat) :
  n = 8 → k = 3 →
  (Nat.choose n k : Rat) / (2 ^ n : Rat) = 7 / 32 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_heads_in_eight_tosses_l28_2816


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l28_2873

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 5 * x - 2 * y + 2 * x * y

-- State the theorem
theorem diamond_equation_solution :
  ∃! y : ℝ, diamond 4 y = 30 ∧ y = 5/3 := by sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l28_2873


namespace NUMINAMATH_CALUDE_inequality_solution_set_l28_2830

theorem inequality_solution_set (a : ℝ) (x₁ x₂ : ℝ) : 
  a > 0 → 
  (∀ x, x^2 - 2*a*x - 3*a^2 < 0 ↔ x₁ < x ∧ x < x₂) → 
  |x₁ - x₂| = 8 → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l28_2830


namespace NUMINAMATH_CALUDE_sin_40_minus_sin_80_l28_2880

theorem sin_40_minus_sin_80 : 
  Real.sin (40 * π / 180) - Real.sin (80 * π / 180) = 
    Real.sin (40 * π / 180) * (1 - 2 * Real.sqrt (1 - Real.sin (40 * π / 180) ^ 2)) := by
  sorry

end NUMINAMATH_CALUDE_sin_40_minus_sin_80_l28_2880


namespace NUMINAMATH_CALUDE_initial_ratio_is_11_to_9_l28_2870

/-- Represents the contents of a can with milk and water -/
structure CanContents where
  milk : ℝ
  water : ℝ

/-- Proves that the initial ratio of milk to water is 11:9 given the conditions -/
theorem initial_ratio_is_11_to_9 (can : CanContents) : 
  can.milk + can.water = 20 → -- Initial contents
  can.milk + can.water + 10 = 30 → -- Adding 10L fills the can
  (can.milk + 10) / can.water = 5 / 2 → -- Resulting ratio is 5:2
  can.milk / can.water = 11 / 9 := by
  sorry

/-- Verify the solution satisfies the conditions -/
example : 
  let can : CanContents := { milk := 11, water := 9 }
  can.milk + can.water = 20 ∧
  can.milk + can.water + 10 = 30 ∧
  (can.milk + 10) / can.water = 5 / 2 ∧
  can.milk / can.water = 11 / 9 := by
  sorry

end NUMINAMATH_CALUDE_initial_ratio_is_11_to_9_l28_2870


namespace NUMINAMATH_CALUDE_fifteen_point_five_minutes_in_hours_l28_2818

/-- Converts minutes to hours -/
def minutes_to_hours (minutes : ℚ) : ℚ :=
  minutes * (1 / 60)

theorem fifteen_point_five_minutes_in_hours : 
  minutes_to_hours 15.5 = 930 / 3600 := by
sorry

end NUMINAMATH_CALUDE_fifteen_point_five_minutes_in_hours_l28_2818


namespace NUMINAMATH_CALUDE_consecutive_composites_under_40_l28_2887

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem consecutive_composites_under_40 :
  ∃ (a : ℕ),
    (∀ i : Fin 6, isTwoDigit (a + i) ∧ a + i < 40) ∧
    (∀ i : Fin 6, ¬ isPrime (a + i)) ∧
    (∀ n : ℕ, n > a + 5 →
      ¬(∀ i : Fin 6, isTwoDigit (n - i) ∧ n - i < 40 ∧ ¬ isPrime (n - i))) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_composites_under_40_l28_2887


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l28_2815

theorem root_exists_in_interval : ∃ x : ℝ, x ∈ Set.Ioo (-2) (-1) ∧ 2^x - x - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l28_2815


namespace NUMINAMATH_CALUDE_arctan_gt_arcsin_iff_in_open_interval_l28_2827

theorem arctan_gt_arcsin_iff_in_open_interval (x : ℝ) :
  Real.arctan x > Real.arcsin x ↔ x ∈ Set.Ioo (-1 : ℝ) 0 :=
by
  sorry

end NUMINAMATH_CALUDE_arctan_gt_arcsin_iff_in_open_interval_l28_2827


namespace NUMINAMATH_CALUDE_ratio_d_b_is_negative_four_l28_2891

def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem ratio_d_b_is_negative_four
  (a b c d : ℝ)
  (h_even : ∀ x, f a b c d x = f a b c d (-x))
  (h_solution : ∀ x, f a b c d x < 0 ↔ -2 < x ∧ x < 2) :
  d / b = -4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_d_b_is_negative_four_l28_2891


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l28_2824

theorem complex_modulus_problem (z₁ z₂ : ℂ) : 
  (z₁ - 2) * Complex.I = 1 + Complex.I →
  z₂.im = 2 →
  ∃ (r : ℝ), z₁ * z₂ = r →
  Complex.abs z₂ = 2 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l28_2824


namespace NUMINAMATH_CALUDE_unique_prime_pair_l28_2825

theorem unique_prime_pair : ∃! (p q : ℕ), 
  Prime p ∧ Prime q ∧ 
  ∃ r : ℕ, Prime r ∧ 
  (1 : ℚ) + (p^q - q^p : ℚ) / (p + q : ℚ) = r := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_pair_l28_2825


namespace NUMINAMATH_CALUDE_bakery_inventory_theorem_l28_2844

/-- Represents the inventory and sales of a bakery --/
structure BakeryInventory where
  cheesecakes_display : ℕ
  cheesecakes_fridge : ℕ
  cherry_pies_ready : ℕ
  cherry_pies_oven : ℕ
  chocolate_eclairs_counter : ℕ
  chocolate_eclairs_pantry : ℕ
  cheesecakes_sold : ℕ
  cherry_pies_sold : ℕ
  chocolate_eclairs_sold : ℕ

/-- Calculates the total number of desserts left to sell --/
def desserts_left_to_sell (inventory : BakeryInventory) : ℕ :=
  (inventory.cheesecakes_display + inventory.cheesecakes_fridge - inventory.cheesecakes_sold) +
  (inventory.cherry_pies_ready + inventory.cherry_pies_oven - inventory.cherry_pies_sold) +
  (inventory.chocolate_eclairs_counter + inventory.chocolate_eclairs_pantry - inventory.chocolate_eclairs_sold)

/-- Theorem stating that given the specific inventory and sales, there are 62 desserts left to sell --/
theorem bakery_inventory_theorem (inventory : BakeryInventory) 
  (h1 : inventory.cheesecakes_display = 10)
  (h2 : inventory.cheesecakes_fridge = 15)
  (h3 : inventory.cherry_pies_ready = 12)
  (h4 : inventory.cherry_pies_oven = 20)
  (h5 : inventory.chocolate_eclairs_counter = 20)
  (h6 : inventory.chocolate_eclairs_pantry = 10)
  (h7 : inventory.cheesecakes_sold = 7)
  (h8 : inventory.cherry_pies_sold = 8)
  (h9 : inventory.chocolate_eclairs_sold = 10) :
  desserts_left_to_sell inventory = 62 := by
  sorry

end NUMINAMATH_CALUDE_bakery_inventory_theorem_l28_2844


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_square_roots_l28_2876

theorem max_value_of_sum_of_square_roots (a b c : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c)
  (sum_condition : a + b + c = 1) :
  Real.sqrt (4 * a + 1) + Real.sqrt (4 * b + 1) + Real.sqrt (4 * c + 1) ≤ Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_square_roots_l28_2876


namespace NUMINAMATH_CALUDE_equation_solution_l28_2875

theorem equation_solution (x : ℚ) : 
  (1 : ℚ) / 3 + 1 / x = (3 : ℚ) / 4 → x = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l28_2875


namespace NUMINAMATH_CALUDE_smallest_stairs_l28_2820

theorem smallest_stairs (n : ℕ) : 
  n > 15 ∧ 
  n % 6 = 4 ∧ 
  n % 7 = 3 → 
  n ≥ 52 :=
by sorry

end NUMINAMATH_CALUDE_smallest_stairs_l28_2820


namespace NUMINAMATH_CALUDE_basketball_lineup_count_l28_2897

def total_players : ℕ := 16
def twins : ℕ := 2
def seniors : ℕ := 5
def lineup_size : ℕ := 7

/-- The number of ways to choose a lineup of 7 players from a team of 16 players,
    including a set of twins and 5 seniors, where exactly one twin must be in the lineup
    and at least two seniors must be selected. -/
theorem basketball_lineup_count : 
  (Nat.choose twins 1) *
  (Nat.choose seniors 2 * Nat.choose (total_players - twins - seniors) 4 +
   Nat.choose seniors 3 * Nat.choose (total_players - twins - seniors) 3) = 4200 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_count_l28_2897


namespace NUMINAMATH_CALUDE_total_tax_percentage_l28_2831

/-- Calculates the total tax percentage given spending percentages and tax rates -/
theorem total_tax_percentage
  (clothing_percent : ℝ)
  (food_percent : ℝ)
  (other_percent : ℝ)
  (clothing_tax_rate : ℝ)
  (food_tax_rate : ℝ)
  (other_tax_rate : ℝ)
  (h1 : clothing_percent = 0.5)
  (h2 : food_percent = 0.2)
  (h3 : other_percent = 0.3)
  (h4 : clothing_percent + food_percent + other_percent = 1)
  (h5 : clothing_tax_rate = 0.04)
  (h6 : food_tax_rate = 0)
  (h7 : other_tax_rate = 0.1) :
  clothing_percent * clothing_tax_rate +
  food_percent * food_tax_rate +
  other_percent * other_tax_rate = 0.05 := by
sorry


end NUMINAMATH_CALUDE_total_tax_percentage_l28_2831


namespace NUMINAMATH_CALUDE_tiaorizhi_approximation_of_pi_l28_2853

def tiaorizhi (a b c d : ℕ) : ℚ := (b + d) / (a + c)

theorem tiaorizhi_approximation_of_pi :
  let initial_lower : ℚ := 3 / 1
  let initial_upper : ℚ := 7 / 2
  let step1 : ℚ := tiaorizhi 1 3 2 7
  let step2 : ℚ := tiaorizhi 1 3 4 13
  let step3 : ℚ := tiaorizhi 1 3 5 16
  initial_lower < Real.pi ∧ Real.pi < initial_upper →
  step3 - Real.pi < 0.1 ∧ Real.pi < step3 := by
  sorry

end NUMINAMATH_CALUDE_tiaorizhi_approximation_of_pi_l28_2853


namespace NUMINAMATH_CALUDE_isosceles_angle_B_l28_2889

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)

-- Define the property of an isosceles triangle
def isIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.A = t.C

-- Define the exterior angle of A
def exteriorAngleA (t : Triangle) : ℝ :=
  180 - t.A

-- Theorem statement
theorem isosceles_angle_B (t : Triangle) 
  (h_ext : exteriorAngleA t = 110) :
  isIsosceles t → t.B = 70 ∨ t.B = 55 ∨ t.B = 40 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_angle_B_l28_2889


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l28_2804

/-- An ellipse with parametric equations x = 3cos(θ) and y = 2sin(θ) -/
structure Ellipse where
  x : ℝ → ℝ
  y : ℝ → ℝ
  h_x : ∀ θ, x θ = 3 * Real.cos θ
  h_y : ∀ θ, y θ = 2 * Real.sin θ

/-- The length of the major axis of an ellipse -/
def major_axis_length (e : Ellipse) : ℝ := 6

/-- Theorem: The length of the major axis of the given ellipse is 6 -/
theorem ellipse_major_axis_length (e : Ellipse) : 
  major_axis_length e = 6 := by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l28_2804


namespace NUMINAMATH_CALUDE_unique_right_triangle_with_2021_leg_l28_2839

theorem unique_right_triangle_with_2021_leg : 
  ∃! (a b c : ℕ+), (a = 2021 ∨ b = 2021) ∧ a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_right_triangle_with_2021_leg_l28_2839


namespace NUMINAMATH_CALUDE_series_sum_is_zero_l28_2881

open Real
open Topology
open Tendsto

noncomputable def series_sum : ℝ := ∑' n, (3 * n + 4) / ((n + 1) * (n + 2) * (n + 3))

theorem series_sum_is_zero : series_sum = 0 := by sorry

end NUMINAMATH_CALUDE_series_sum_is_zero_l28_2881


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_sum_l28_2882

theorem arithmetic_geometric_sequence_sum (a : ℕ → ℤ) :
  (∀ n, a (n + 1) = a n + 2) →  -- arithmetic sequence with common difference 2
  (a 3)^2 = a 1 * a 4 →         -- a_1, a_3, a_4 form a geometric sequence
  a 2 + a 3 = -10 :=            -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_sum_l28_2882


namespace NUMINAMATH_CALUDE_max_value_of_expression_l28_2855

theorem max_value_of_expression (x : ℝ) : 
  (4 * x^2 + 8 * x + 21) / (4 * x^2 + 8 * x + 5) ≤ 17 ∧ 
  ∃ (y : ℝ), (4 * y^2 + 8 * y + 21) / (4 * y^2 + 8 * y + 5) = 17 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l28_2855


namespace NUMINAMATH_CALUDE_sarahs_stamp_collection_value_l28_2871

/-- The value of a stamp collection given the total number of stamps,
    the number of stamps in a subset, and the value of that subset. -/
def stamp_collection_value (total_stamps : ℕ) (subset_stamps : ℕ) (subset_value : ℚ) : ℚ :=
  (total_stamps : ℚ) * subset_value / (subset_stamps : ℚ)

/-- Theorem stating that Sarah's stamp collection is worth 60 dollars -/
theorem sarahs_stamp_collection_value :
  stamp_collection_value 24 8 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_stamp_collection_value_l28_2871


namespace NUMINAMATH_CALUDE_tan_a_values_l28_2857

theorem tan_a_values (a : ℝ) (h : Real.sin (2 * a) = 2 - 2 * Real.cos (2 * a)) :
  Real.tan a = 0 ∨ Real.tan a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_tan_a_values_l28_2857


namespace NUMINAMATH_CALUDE_power_of_negative_cube_l28_2808

theorem power_of_negative_cube (a : ℝ) : (-2 * a^3)^2 = 4 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_cube_l28_2808


namespace NUMINAMATH_CALUDE_sector_angle_l28_2847

theorem sector_angle (area : Real) (radius : Real) (h1 : area = 3 * Real.pi / 16) (h2 : radius = 1) :
  (2 * area) / (radius ^ 2) = 3 * Real.pi / 8 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l28_2847


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_constant_l28_2822

theorem partial_fraction_decomposition_constant (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 5 ∧ x ≠ -3 → 
    1 / (x^3 - 7*x^2 + 11*x + 45) = A / (x - 5) + B / (x + 3) + C / (x + 3)^2) →
  B = -1 / 64 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_constant_l28_2822


namespace NUMINAMATH_CALUDE_entrepreneur_raised_12000_l28_2807

/-- Represents the crowdfunding levels and backers for an entrepreneur's business effort -/
structure CrowdfundingCampaign where
  highest_level : ℕ
  second_level : ℕ
  lowest_level : ℕ
  highest_backers : ℕ
  second_backers : ℕ
  lowest_backers : ℕ

/-- Calculates the total amount raised in a crowdfunding campaign -/
def total_raised (campaign : CrowdfundingCampaign) : ℕ :=
  campaign.highest_level * campaign.highest_backers +
  campaign.second_level * campaign.second_backers +
  campaign.lowest_level * campaign.lowest_backers

/-- Theorem stating that the entrepreneur raised $12000 -/
theorem entrepreneur_raised_12000 :
  ∀ (campaign : CrowdfundingCampaign),
  campaign.highest_level = 5000 ∧
  campaign.second_level = campaign.highest_level / 10 ∧
  campaign.lowest_level = campaign.second_level / 10 ∧
  campaign.highest_backers = 2 ∧
  campaign.second_backers = 3 ∧
  campaign.lowest_backers = 10 →
  total_raised campaign = 12000 :=
sorry

end NUMINAMATH_CALUDE_entrepreneur_raised_12000_l28_2807


namespace NUMINAMATH_CALUDE_final_lives_correct_l28_2800

/-- Given a player's initial lives, lost lives, and gained lives (before bonus),
    calculate the final number of lives after a secret bonus is applied. -/
def final_lives (initial_lives lost_lives gained_lives : ℕ) : ℕ :=
  initial_lives - lost_lives + 3 * gained_lives

/-- Theorem stating that the final_lives function correctly calculates
    the number of lives after the secret bonus is applied. -/
theorem final_lives_correct (X Y Z : ℕ) (h : Y ≤ X) :
  final_lives X Y Z = X - Y + 3 * Z :=
by sorry

end NUMINAMATH_CALUDE_final_lives_correct_l28_2800


namespace NUMINAMATH_CALUDE_exists_winning_strategy_l28_2849

/-- The set of numbers from which the hidden numbers are chosen -/
def S : Set ℕ := Finset.range 250

/-- A strategy is a function that takes the player's number and the history of announcements,
    and returns the next announcement -/
def Strategy := ℕ → List ℕ → ℕ

/-- The game state consists of both players' numbers and the history of announcements -/
structure GameState :=
  (player_a_number : ℕ)
  (player_b_number : ℕ)
  (announcements : List ℕ)

/-- A game is valid if both players' numbers are in S and the sum of announcements is 20 -/
def valid_game (g : GameState) : Prop :=
  g.player_a_number ∈ S ∧ g.player_b_number ∈ S ∧ g.announcements.sum = 20

/-- A strategy is winning if it allows both players to determine each other's number -/
def winning_strategy (strat_a strat_b : Strategy) : Prop :=
  ∀ (g : GameState), valid_game g →
    ∃ (n : ℕ), strat_a g.player_a_number (g.announcements.take n) = g.player_b_number ∧
               strat_b g.player_b_number (g.announcements.take n) = g.player_a_number

/-- There exists a winning strategy for the game -/
theorem exists_winning_strategy : ∃ (strat_a strat_b : Strategy), winning_strategy strat_a strat_b :=
sorry

end NUMINAMATH_CALUDE_exists_winning_strategy_l28_2849


namespace NUMINAMATH_CALUDE_intersection_bounds_l28_2840

theorem intersection_bounds (m : ℝ) : 
  let A : Set ℝ := {x | -2 < x ∧ x < 8}
  let B : Set ℝ := {x | 2*m - 1 < x ∧ x < m + 3}
  let U : Set ℝ := Set.univ
  ∃ (a b : ℝ), A ∩ B = {x | a < x ∧ x < b} ∧ b - a = 3 → m = -2 ∨ m = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_bounds_l28_2840


namespace NUMINAMATH_CALUDE_maximize_product_l28_2842

theorem maximize_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 100) :
  x^4 * y^6 ≤ 40^4 * 60^6 ∧ 
  (x^4 * y^6 = 40^4 * 60^6 ↔ x = 40 ∧ y = 60) :=
sorry

end NUMINAMATH_CALUDE_maximize_product_l28_2842


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l28_2826

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |2 * x - 6| = 3 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l28_2826


namespace NUMINAMATH_CALUDE_power_of_two_l28_2856

theorem power_of_two : (1 : ℕ) * 2^6 = 64 := by sorry

end NUMINAMATH_CALUDE_power_of_two_l28_2856


namespace NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l28_2883

/-- The ratio of the side length of a regular pentagon to the width of a rectangle with the same perimeter -/
theorem pentagon_rectangle_ratio : 
  ∀ (pentagon_side rectangle_width : ℝ),
  pentagon_side > 0 → 
  rectangle_width > 0 →
  5 * pentagon_side = 20 →
  6 * rectangle_width = 20 →
  pentagon_side / rectangle_width = 6 / 5 := by
sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l28_2883


namespace NUMINAMATH_CALUDE_todds_initial_gum_l28_2878

theorem todds_initial_gum (x : ℕ) : x + 16 = 54 → x = 38 := by
  sorry

end NUMINAMATH_CALUDE_todds_initial_gum_l28_2878


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l28_2879

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l28_2879


namespace NUMINAMATH_CALUDE_point_distance_l28_2899

-- Define the points as real numbers representing their positions on a line
variable (A B C D : ℝ)

-- Define the conditions
variable (h_order : A < B ∧ B < C ∧ C < D)
variable (h_ratio : (B - A) / (C - B) = (D - A) / (D - C))
variable (h_AC : C - A = 3)
variable (h_BD : D - B = 4)

-- State the theorem
theorem point_distance (A B C D : ℝ) 
  (h_order : A < B ∧ B < C ∧ C < D)
  (h_ratio : (B - A) / (C - B) = (D - A) / (D - C))
  (h_AC : C - A = 3)
  (h_BD : D - B = 4) : 
  D - A = 6 := by sorry

end NUMINAMATH_CALUDE_point_distance_l28_2899


namespace NUMINAMATH_CALUDE_paperboy_delivery_ways_l28_2864

/-- Represents the number of ways to deliver newspapers to n houses without missing four consecutive houses. -/
def E : ℕ → ℕ
  | 0 => 0  -- Define E(0) as 0 for completeness
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | 4 => 15
  | n + 5 => E (n + 4) + E (n + 3) + E (n + 2) + E (n + 1)

/-- Theorem stating that there are 2872 ways for a paperboy to deliver newspapers to 12 houses without missing four consecutive houses. -/
theorem paperboy_delivery_ways : E 12 = 2872 := by
  sorry

end NUMINAMATH_CALUDE_paperboy_delivery_ways_l28_2864


namespace NUMINAMATH_CALUDE_journey_possible_l28_2832

/-- Represents the journey parameters and conditions -/
structure JourneyParams where
  total_distance : ℝ
  motorcycle_speed : ℝ
  baldwin_speed : ℝ
  clark_speed : ℝ
  (total_distance_positive : total_distance > 0)
  (speeds_positive : motorcycle_speed > 0 ∧ baldwin_speed > 0 ∧ clark_speed > 0)
  (motorcycle_fastest : motorcycle_speed > baldwin_speed ∧ motorcycle_speed > clark_speed)

/-- Represents a valid journey plan -/
structure JourneyPlan where
  params : JourneyParams
  baldwin_pickup : ℝ
  clark_pickup : ℝ
  (valid_pickups : 0 ≤ baldwin_pickup ∧ baldwin_pickup ≤ params.total_distance ∧
                   0 ≤ clark_pickup ∧ clark_pickup ≤ params.total_distance)

/-- Calculates the total time for a given journey plan -/
def totalTime (plan : JourneyPlan) : ℝ :=
  sorry

/-- Theorem stating that there exists a journey plan where everyone arrives in 5 hours -/
theorem journey_possible (params : JourneyParams) 
  (h1 : params.total_distance = 52)
  (h2 : params.motorcycle_speed = 20)
  (h3 : params.baldwin_speed = 5)
  (h4 : params.clark_speed = 4) :
  ∃ (plan : JourneyPlan), totalTime plan = 5 :=
sorry

end NUMINAMATH_CALUDE_journey_possible_l28_2832


namespace NUMINAMATH_CALUDE_sqrt_1_0201_l28_2893

theorem sqrt_1_0201 (h : Real.sqrt 102.01 = 10.1) : Real.sqrt 1.0201 = 1.01 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_1_0201_l28_2893


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l28_2866

/-- A rhombus with side length 35 units and shorter diagonal 42 units has a longer diagonal of 56 units. -/
theorem rhombus_longer_diagonal (s d_short : ℝ) (h1 : s = 35) (h2 : d_short = 42) :
  let d_long := Real.sqrt (4 * s^2 - d_short^2)
  d_long = 56 := by sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l28_2866


namespace NUMINAMATH_CALUDE_sqrt_inequality_sum_reciprocal_inequality_l28_2874

-- Problem 1
theorem sqrt_inequality : Real.sqrt 3 + Real.sqrt 8 < 2 + Real.sqrt 7 := by sorry

-- Problem 2
theorem sum_reciprocal_inequality {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 1/a + 1/b + 1/c ≥ 9 := by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_sum_reciprocal_inequality_l28_2874


namespace NUMINAMATH_CALUDE_angle_sum_theorem_l28_2829

-- Define the sum of angles around a point
def sum_of_angles : ℝ := 360

-- Define the four angles as functions of x
def angle1 (x : ℝ) : ℝ := 5 * x
def angle2 (x : ℝ) : ℝ := 4 * x
def angle3 (x : ℝ) : ℝ := x
def angle4 (x : ℝ) : ℝ := 2 * x

-- Theorem statement
theorem angle_sum_theorem :
  ∃ x : ℝ, angle1 x + angle2 x + angle3 x + angle4 x = sum_of_angles ∧ x = 30 :=
by sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l28_2829


namespace NUMINAMATH_CALUDE_square_of_linear_expression_l28_2812

theorem square_of_linear_expression (p : ℝ) (m : ℝ) : p ≠ 0 →
  (∃ a b : ℝ, ∀ x : ℝ, (9 * x^2 + 21 * x + 4 * m) / 9 = (a * x + b)^2) ∧
  (∃ a b : ℝ, (9 * (p - 1)^2 + 21 * (p - 1) + 4 * m) / 9 = (a * (p - 1) + b)^2) →
  m = 49 / 16 := by
sorry

end NUMINAMATH_CALUDE_square_of_linear_expression_l28_2812


namespace NUMINAMATH_CALUDE_arithmetic_progression_x_value_l28_2890

/-- An arithmetic progression of positive integers -/
def arithmetic_progression (s : ℕ → ℕ) : Prop :=
  ∃ a d : ℕ, ∀ n : ℕ, s n = a + (n - 1) * d

theorem arithmetic_progression_x_value
  (s : ℕ → ℕ) (x : ℝ)
  (h_arithmetic : arithmetic_progression s)
  (h_s1 : s (s 1) = x + 2)
  (h_s2 : s (s 2) = x^2 + 18)
  (h_s3 : s (s 3) = 2*x^2 + 18) :
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_x_value_l28_2890


namespace NUMINAMATH_CALUDE_equation_solution_l28_2898

theorem equation_solution (x : ℝ) (h : x ≠ -2/3) :
  (3*x + 2) / (3*x^2 - 7*x - 6) = (2*x + 1) / (3*x - 2) ↔
  x = (13 + Real.sqrt 109) / 6 ∨ x = (13 - Real.sqrt 109) / 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l28_2898


namespace NUMINAMATH_CALUDE_largest_five_digit_with_product_180_l28_2801

/-- A function that returns true if a number is a five-digit number -/
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

/-- A function that returns the product of digits of a natural number -/
def digit_product (n : ℕ) : ℕ :=
  sorry

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  sorry

/-- The theorem to be proved -/
theorem largest_five_digit_with_product_180 :
  ∃ M : ℕ, is_five_digit M ∧
           digit_product M = 180 ∧
           (∀ n : ℕ, is_five_digit n → digit_product n = 180 → n ≤ M) ∧
           digit_sum M = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_with_product_180_l28_2801


namespace NUMINAMATH_CALUDE_dance_class_boys_count_l28_2823

theorem dance_class_boys_count :
  ∀ (girls boys : ℕ),
  girls + boys = 35 →
  4 * girls = 3 * boys →
  boys = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_dance_class_boys_count_l28_2823


namespace NUMINAMATH_CALUDE_jack_marbles_shared_l28_2894

/-- Calculates the number of marbles shared given initial and remaining marbles -/
def marblesShared (initial remaining : ℕ) : ℕ := initial - remaining

/-- Proves that the number of marbles shared is correct for Jack's scenario -/
theorem jack_marbles_shared :
  marblesShared 62 29 = 33 := by
  sorry

end NUMINAMATH_CALUDE_jack_marbles_shared_l28_2894


namespace NUMINAMATH_CALUDE_stephanies_internet_bill_l28_2896

/-- Stephanie's household budget problem -/
theorem stephanies_internet_bill :
  let electricity_bill : ℕ := 60
  let gas_bill : ℕ := 40
  let water_bill : ℕ := 40
  let gas_paid : ℚ := 3/4 * gas_bill + 5
  let water_paid : ℚ := 1/2 * water_bill
  let internet_payments : ℕ := 4
  let internet_payment_amount : ℕ := 5
  let total_remaining : ℕ := 30
  
  ∃ (internet_bill : ℕ),
    internet_bill = internet_payments * internet_payment_amount + 
      (total_remaining - (gas_bill - gas_paid) - (water_bill - water_paid)) :=
by
  sorry


end NUMINAMATH_CALUDE_stephanies_internet_bill_l28_2896


namespace NUMINAMATH_CALUDE_marcus_earnings_theorem_l28_2805

/-- Calculates the after-tax earnings for Marcus over two weeks -/
def marcusEarnings (hoursWeek1 hoursWeek2 : ℕ) (extraEarnings : ℚ) (taxRate : ℚ) : ℚ :=
  let hourlyWage := extraEarnings / (hoursWeek2 - hoursWeek1)
  let totalHours := hoursWeek1 + hoursWeek2
  let totalEarnings := hourlyWage * totalHours
  totalEarnings * (1 - taxRate)

/-- Theorem stating that Marcus's earnings after tax for the two weeks is $293.40 -/
theorem marcus_earnings_theorem :
  marcusEarnings 20 30 65.20 0.1 = 293.40 := by
  sorry

end NUMINAMATH_CALUDE_marcus_earnings_theorem_l28_2805


namespace NUMINAMATH_CALUDE_function_value_at_five_l28_2809

theorem function_value_at_five (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f x + 2 * f (1 - x) = 2 * x^2 - 1) : 
  f 5 = 13/3 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_five_l28_2809


namespace NUMINAMATH_CALUDE_partner_p_investment_time_l28_2834

/-- The investment and profit scenario for two partners -/
structure InvestmentScenario where
  /-- The ratio of investments for partners p and q -/
  investment_ratio : Rat × Rat
  /-- The ratio of profits for partners p and q -/
  profit_ratio : Rat × Rat
  /-- The number of months partner q invested -/
  q_months : ℕ

/-- The theorem stating the investment time for partner p -/
theorem partner_p_investment_time (scenario : InvestmentScenario) 
  (h1 : scenario.investment_ratio = (7, 5))
  (h2 : scenario.profit_ratio = (7, 13))
  (h3 : scenario.q_months = 13) :
  ∃ (p_months : ℕ), p_months = 7 ∧ 
  (scenario.investment_ratio.1 * p_months) / (scenario.investment_ratio.2 * scenario.q_months) = 
  scenario.profit_ratio.1 / scenario.profit_ratio.2 :=
sorry

end NUMINAMATH_CALUDE_partner_p_investment_time_l28_2834


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l28_2806

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (1 + x, 1 - 3*x)
  let b : ℝ × ℝ := (2, -1)
  are_parallel a b → x = 3/5 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l28_2806


namespace NUMINAMATH_CALUDE_competition_participants_l28_2869

theorem competition_participants : ℕ :=
  let initial_participants : ℕ := sorry
  let first_round_survival_rate : ℚ := 1 / 3
  let second_round_survival_rate : ℚ := 1 / 4
  let final_participants : ℕ := 18

  have h1 : (initial_participants : ℚ) * first_round_survival_rate * second_round_survival_rate = final_participants := by sorry

  initial_participants

end NUMINAMATH_CALUDE_competition_participants_l28_2869


namespace NUMINAMATH_CALUDE_imaginary_part_of_x_l28_2836

theorem imaginary_part_of_x (x : ℂ) (h : (3 + 4*I)*x = Complex.abs (4 + 3*I)) : 
  x.im = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_x_l28_2836


namespace NUMINAMATH_CALUDE_min_distance_sum_parabola_l28_2862

/-- The minimum distance sum from a point on the parabola y^2 = 8x to two fixed points -/
theorem min_distance_sum_parabola :
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (6, 5)
  let parabola := {P : ℝ × ℝ | P.2^2 = 8 * P.1}
  ∃ (min_dist : ℝ), min_dist = 8 ∧ 
    ∀ P ∈ parabola, Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) + 
                     Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_distance_sum_parabola_l28_2862


namespace NUMINAMATH_CALUDE_cos_alpha_value_l28_2861

def point_on_terminal_side (α : Real) (x y : Real) : Prop :=
  ∃ (r : Real), r > 0 ∧ x = r * Real.cos α ∧ y = r * Real.sin α

theorem cos_alpha_value (α : Real) :
  point_on_terminal_side α 1 3 → Real.cos α = 1 / Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l28_2861


namespace NUMINAMATH_CALUDE_root_range_l28_2821

theorem root_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁ ∈ Set.Icc (k - 1) (k + 1) ∧ 
    x₂ ∈ Set.Icc (k - 1) (k + 1) ∧
    Real.sqrt 2 * |x₁ - k| = k * Real.sqrt x₁ ∧
    Real.sqrt 2 * |x₂ - k| = k * Real.sqrt x₂) 
  ↔ 
  (0 < k ∧ k ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_root_range_l28_2821


namespace NUMINAMATH_CALUDE_cos_300_degrees_l28_2851

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l28_2851


namespace NUMINAMATH_CALUDE_sqrt_221_range_l28_2858

theorem sqrt_221_range : 14 < Real.sqrt 221 ∧ Real.sqrt 221 < 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_221_range_l28_2858


namespace NUMINAMATH_CALUDE_no_solution_range_l28_2859

theorem no_solution_range (m : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 5| > m) ↔ m < 6 := by sorry

end NUMINAMATH_CALUDE_no_solution_range_l28_2859


namespace NUMINAMATH_CALUDE_quadratic_function_property_l28_2846

def f (b c x : ℝ) : ℝ := -x^2 + b*x + c

theorem quadratic_function_property (b c : ℝ) :
  f b c 2 + f b c 4 = 12138 →
  3*b + c = 6079 →
  f b c 3 = 6070 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l28_2846


namespace NUMINAMATH_CALUDE_only_zhong_symmetrical_l28_2811

/-- Represents a Chinese character --/
inductive ChineseCharacter
| ai    -- 爱
| wo    -- 我
| zhong -- 中
| guo   -- 国

/-- Determines if a Chinese character is symmetrical --/
def is_symmetrical (c : ChineseCharacter) : Prop :=
  match c with
  | ChineseCharacter.zhong => True
  | _ => False

/-- Theorem stating that among the given characters, only 中 (zhong) is symmetrical --/
theorem only_zhong_symmetrical :
  ∀ c : ChineseCharacter,
    is_symmetrical c ↔ c = ChineseCharacter.zhong :=
by sorry

end NUMINAMATH_CALUDE_only_zhong_symmetrical_l28_2811


namespace NUMINAMATH_CALUDE_gift_price_proof_l28_2892

def gift_price_calculation (lisa_savings : ℝ) (mother_fraction : ℝ) (brother_multiplier : ℝ) (price_difference : ℝ) : Prop :=
  let mother_contribution := mother_fraction * lisa_savings
  let brother_contribution := brother_multiplier * mother_contribution
  let total_amount := lisa_savings + mother_contribution + brother_contribution
  let gift_price := total_amount + price_difference
  gift_price = 3760

theorem gift_price_proof :
  gift_price_calculation 1200 (3/5) 2 400 := by
  sorry

end NUMINAMATH_CALUDE_gift_price_proof_l28_2892


namespace NUMINAMATH_CALUDE_value_of_y_l28_2803

theorem value_of_y (x y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x = 21) : y = 84 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l28_2803


namespace NUMINAMATH_CALUDE_base_subtraction_proof_l28_2877

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

theorem base_subtraction_proof :
  let base5_num := [4, 2, 3]  -- 324 in base 5 (least significant digit first)
  let base6_num := [3, 5, 1]  -- 153 in base 6 (least significant digit first)
  toBase10 base5_num 5 - toBase10 base6_num 6 = 20 := by
  sorry

end NUMINAMATH_CALUDE_base_subtraction_proof_l28_2877


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_is_one_l28_2886

/-- A geometric progression is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricProgression (a : ℝ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a n = a 0 * r ^ n

/-- The product of any two terms in the progression is also a term in the progression. -/
def ProductIsInProgression (a : ℝ → ℝ) : Prop :=
  ∀ i j k : ℕ, ∃ k : ℕ, a i * a j = a k

/-- In a geometric progression where the product of any two terms is also a term in the progression,
    the first term of the progression must be 1. -/
theorem geometric_progression_first_term_is_one
  (a : ℝ → ℝ) (r : ℝ)
  (h1 : IsGeometricProgression a r)
  (h2 : ProductIsInProgression a) :
  a 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_is_one_l28_2886


namespace NUMINAMATH_CALUDE_complex_equation_proof_l28_2872

def complex_i : ℂ := Complex.I

theorem complex_equation_proof (z : ℂ) (h : z = 1 + complex_i) : 
  2 / z + z^2 = 1 + complex_i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_proof_l28_2872


namespace NUMINAMATH_CALUDE_milburg_population_l28_2854

theorem milburg_population :
  let grown_ups : ℕ := 5256
  let children : ℕ := 2987
  grown_ups + children = 8243 := by
  sorry

end NUMINAMATH_CALUDE_milburg_population_l28_2854


namespace NUMINAMATH_CALUDE_wendy_total_profit_l28_2835

/-- Represents a fruit sale --/
structure FruitSale where
  price : Float
  quantity : Nat
  profit_margin : Float
  discount : Float

/-- Represents a day's sales --/
structure DaySales where
  morning_apples : FruitSale
  morning_oranges : FruitSale
  morning_bananas : FruitSale
  afternoon_apples : FruitSale
  afternoon_oranges : FruitSale
  afternoon_bananas : FruitSale

/-- Represents unsold fruits --/
structure UnsoldFruits where
  banana_quantity : Nat
  banana_price : Float
  banana_discount : Float
  banana_profit_margin : Float
  orange_quantity : Nat
  orange_price : Float
  orange_discount : Float
  orange_profit_margin : Float

/-- Calculate profit for a single fruit sale --/
def calculate_profit (sale : FruitSale) : Float :=
  sale.price * sale.quantity.toFloat * (1 - sale.discount) * sale.profit_margin

/-- Calculate total profit for a day --/
def calculate_day_profit (day : DaySales) : Float :=
  calculate_profit day.morning_apples +
  calculate_profit day.morning_oranges +
  calculate_profit day.morning_bananas +
  calculate_profit day.afternoon_apples +
  calculate_profit day.afternoon_oranges +
  calculate_profit day.afternoon_bananas

/-- Calculate profit from unsold fruits --/
def calculate_unsold_profit (unsold : UnsoldFruits) : Float :=
  unsold.banana_quantity.toFloat * unsold.banana_price * (1 - unsold.banana_discount) * unsold.banana_profit_margin +
  unsold.orange_quantity.toFloat * unsold.orange_price * (1 - unsold.orange_discount) * unsold.orange_profit_margin

/-- Main theorem: Wendy's total profit for the week --/
theorem wendy_total_profit (day1 day2 : DaySales) (unsold : UnsoldFruits) :
  calculate_day_profit day1 + calculate_day_profit day2 + calculate_unsold_profit unsold = 84.07 := by
  sorry

end NUMINAMATH_CALUDE_wendy_total_profit_l28_2835


namespace NUMINAMATH_CALUDE_equation_solution_l28_2833

theorem equation_solution : ∃ x : ℚ, (1/6 : ℚ) + 2/x = 3/x + (1/15 : ℚ) ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l28_2833


namespace NUMINAMATH_CALUDE_root_range_implies_a_range_l28_2817

theorem root_range_implies_a_range (a : ℝ) :
  (∃ α β : ℝ, 5 * α^2 - 7 * α - a = 0 ∧
              5 * β^2 - 7 * β - a = 0 ∧
              -1 < α ∧ α < 0 ∧
              1 < β ∧ β < 2) →
  (0 < a ∧ a < 6) := by
sorry

end NUMINAMATH_CALUDE_root_range_implies_a_range_l28_2817
