import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2582_258262

/-- Given an arithmetic sequence {aₙ} with sum of first n terms Sₙ,
    common difference d = 2, and a₅ = 10, prove that S₁₀ = 110 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = 2) →  -- arithmetic sequence with common difference 2
  (∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * 2)) →  -- sum formula
  a 5 = 10 →  -- given condition
  S 10 = 110 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2582_258262


namespace NUMINAMATH_CALUDE_apple_cost_graph_properties_l2582_258284

def apple_cost (n : ℕ) : ℚ :=
  if n ≤ 10 then 20 * n else 18 * n

theorem apple_cost_graph_properties :
  ∃ (f : ℕ → ℚ),
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → f n = apple_cost n) ∧
    (∀ n : ℕ, 1 ≤ n ∧ n < 10 → f (n + 1) - f n = 20) ∧
    (∀ n : ℕ, 10 < n ∧ n < 20 → f (n + 1) - f n = 18) ∧
    (f 11 - f 10 ≠ 20 ∧ f 11 - f 10 ≠ 18) :=
by sorry

end NUMINAMATH_CALUDE_apple_cost_graph_properties_l2582_258284


namespace NUMINAMATH_CALUDE_total_cost_is_100_l2582_258235

/-- Calculates the total cost in dollars for using whiteboards in all classes for one day -/
def whiteboard_cost (num_classes : ℕ) (boards_per_class : ℕ) (ink_per_board : ℕ) (cost_per_ml : ℚ) : ℚ :=
  num_classes * boards_per_class * ink_per_board * cost_per_ml

/-- Proves that the total cost for using whiteboards in all classes for one day is $100 -/
theorem total_cost_is_100 : 
  whiteboard_cost 5 2 20 (1/2) = 100 := by
  sorry

#eval whiteboard_cost 5 2 20 (1/2)

end NUMINAMATH_CALUDE_total_cost_is_100_l2582_258235


namespace NUMINAMATH_CALUDE_a_7_equals_neg_3_l2582_258277

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem stating that a₇ = -3 in the given geometric sequence -/
theorem a_7_equals_neg_3 (a : ℕ → ℝ) :
  geometric_sequence a →
  a 5 ^ 2 + 2016 * a 5 + 9 = 0 →
  a 9 ^ 2 + 2016 * a 9 + 9 = 0 →
  a 7 = -3 := by
  sorry

end NUMINAMATH_CALUDE_a_7_equals_neg_3_l2582_258277


namespace NUMINAMATH_CALUDE_binomial_n_value_l2582_258298

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial random variable -/
def variance (ξ : BinomialRV) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_n_value (ξ : BinomialRV) 
  (h_exp : expectation ξ = 6)
  (h_var : variance ξ = 3) : 
  ξ.n = 12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_n_value_l2582_258298


namespace NUMINAMATH_CALUDE_g_equals_power_of_two_times_power_of_three_num_divisors_g_2023_equals_4096576_l2582_258229

/-- Function g(n) returns the smallest positive integer k such that 1/k has exactly n digits after the decimal point in base 6 notation -/
def g (n : ℕ+) : ℕ+ :=
  sorry

/-- Theorem stating that g(n) = 2^n * 3^n for all positive integers n -/
theorem g_equals_power_of_two_times_power_of_three (n : ℕ+) :
  g n = 2^(n : ℕ) * 3^(n : ℕ) :=
sorry

/-- The number of positive integer divisors of g(2023) -/
def num_divisors_g_2023 : ℕ :=
  (2023 + 1)^2

/-- Theorem stating that the number of positive integer divisors of g(2023) is 4096576 -/
theorem num_divisors_g_2023_equals_4096576 :
  num_divisors_g_2023 = 4096576 :=
sorry

end NUMINAMATH_CALUDE_g_equals_power_of_two_times_power_of_three_num_divisors_g_2023_equals_4096576_l2582_258229


namespace NUMINAMATH_CALUDE_georgie_guacamole_servings_l2582_258226

/-- The number of avocados needed for one serving of guacamole -/
def avocados_per_serving : ℕ := 3

/-- The number of avocados Georgie initially has -/
def initial_avocados : ℕ := 5

/-- The number of avocados Georgie's sister buys -/
def sister_bought_avocados : ℕ := 4

/-- The total number of avocados Georgie has -/
def total_avocados : ℕ := initial_avocados + sister_bought_avocados

/-- The number of servings of guacamole Georgie can make -/
def servings_of_guacamole : ℕ := total_avocados / avocados_per_serving

theorem georgie_guacamole_servings : servings_of_guacamole = 3 := by
  sorry

end NUMINAMATH_CALUDE_georgie_guacamole_servings_l2582_258226


namespace NUMINAMATH_CALUDE_least_sum_with_conditions_l2582_258213

theorem least_sum_with_conditions (m n : ℕ+) 
  (h1 : Nat.gcd (m + n) 210 = 1)
  (h2 : ∃ k : ℕ, m^m.val = k * n^n.val)
  (h3 : ¬∃ k : ℕ, m = k * n) :
  (∀ p q : ℕ+, 
    Nat.gcd (p + q) 210 = 1 → 
    (∃ k : ℕ, p^p.val = k * q^q.val) → 
    (¬∃ k : ℕ, p = k * q) → 
    m + n ≤ p + q) →
  m + n = 407 := by
sorry

end NUMINAMATH_CALUDE_least_sum_with_conditions_l2582_258213


namespace NUMINAMATH_CALUDE_video_game_collection_cost_l2582_258297

theorem video_game_collection_cost (total_games : ℕ) 
  (games_at_12 : ℕ) (price_12 : ℕ) (price_7 : ℕ) (price_3 : ℕ) :
  total_games = 346 →
  games_at_12 = 80 →
  price_12 = 12 →
  price_7 = 7 →
  price_3 = 3 →
  (games_at_12 * price_12 + 
   ((total_games - games_at_12) / 2) * price_7 + 
   ((total_games - games_at_12) - ((total_games - games_at_12) / 2)) * price_3) = 2290 := by
sorry

#eval 80 * 12 + ((346 - 80) / 2) * 7 + ((346 - 80) - ((346 - 80) / 2)) * 3

end NUMINAMATH_CALUDE_video_game_collection_cost_l2582_258297


namespace NUMINAMATH_CALUDE_total_potatoes_l2582_258211

-- Define the number of people sharing the potatoes
def num_people : Nat := 3

-- Define the number of potatoes each person received
def potatoes_per_person : Nat := 8

-- Theorem to prove the total number of potatoes
theorem total_potatoes : num_people * potatoes_per_person = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_potatoes_l2582_258211


namespace NUMINAMATH_CALUDE_garden_width_is_eleven_l2582_258273

/-- Represents a rectangular garden with specific dimensions. -/
structure RectangularGarden where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  length_width_relation : length = width + 2
  perimeter_formula : perimeter = 2 * (length + width)

/-- Theorem: The width of a rectangular garden with perimeter 48m and length 2m more than width is 11m. -/
theorem garden_width_is_eleven (garden : RectangularGarden) 
    (h_perimeter : garden.perimeter = 48) : garden.width = 11 := by
  sorry

#check garden_width_is_eleven

end NUMINAMATH_CALUDE_garden_width_is_eleven_l2582_258273


namespace NUMINAMATH_CALUDE_excess_value_proof_l2582_258279

def two_digit_number : ℕ := 57

def tens_digit (n : ℕ) : ℕ := n / 10
def ones_digit (n : ℕ) : ℕ := n % 10

def sum_of_digits (n : ℕ) : ℕ := tens_digit n + ones_digit n

def reversed_number (n : ℕ) : ℕ := 10 * (ones_digit n) + tens_digit n

theorem excess_value_proof :
  ∃ (v : ℕ), two_digit_number = 4 * (sum_of_digits two_digit_number) + v ∧
  two_digit_number + 18 = reversed_number two_digit_number ∧
  v = 9 := by
  sorry

end NUMINAMATH_CALUDE_excess_value_proof_l2582_258279


namespace NUMINAMATH_CALUDE_burger_lovers_l2582_258299

theorem burger_lovers (total : ℕ) (pizza_lovers : ℕ) (both_lovers : ℕ) 
    (h1 : total = 200)
    (h2 : pizza_lovers = 125)
    (h3 : both_lovers = 40)
    (h4 : both_lovers ≤ pizza_lovers)
    (h5 : pizza_lovers ≤ total) :
  total - (pizza_lovers - both_lovers) - both_lovers = 115 := by
  sorry

end NUMINAMATH_CALUDE_burger_lovers_l2582_258299


namespace NUMINAMATH_CALUDE_sportswear_problem_l2582_258215

/-- Sportswear Problem -/
theorem sportswear_problem 
  (first_batch_cost : ℝ) 
  (second_batch_cost : ℝ) 
  (selling_price : ℝ) 
  (h1 : first_batch_cost = 12000)
  (h2 : second_batch_cost = 26400)
  (h3 : selling_price = 150) :
  ∃ (first_batch_quantity second_batch_quantity : ℕ),
    (second_batch_quantity = 2 * first_batch_quantity) ∧
    (second_batch_cost / second_batch_quantity = first_batch_cost / first_batch_quantity + 10) ∧
    (second_batch_quantity = 240) ∧
    (first_batch_quantity * (selling_price - first_batch_cost / first_batch_quantity) +
     second_batch_quantity * (selling_price - second_batch_cost / second_batch_quantity) = 15600) := by
  sorry

end NUMINAMATH_CALUDE_sportswear_problem_l2582_258215


namespace NUMINAMATH_CALUDE_percentage_not_liking_basketball_is_52_percent_l2582_258259

/-- Represents the school population and basketball preferences --/
structure School where
  total_students : ℕ
  male_ratio : ℚ
  female_ratio : ℚ
  male_basketball_ratio : ℚ
  female_basketball_ratio : ℚ

/-- Calculates the percentage of students who don't like basketball --/
def percentage_not_liking_basketball (s : School) : ℚ :=
  let male_count := s.total_students * s.male_ratio / (s.male_ratio + s.female_ratio)
  let female_count := s.total_students * s.female_ratio / (s.male_ratio + s.female_ratio)
  let male_playing := male_count * s.male_basketball_ratio
  let female_playing := female_count * s.female_basketball_ratio
  let total_not_playing := s.total_students - (male_playing + female_playing)
  total_not_playing / s.total_students * 100

/-- The main theorem to prove --/
theorem percentage_not_liking_basketball_is_52_percent :
  let s : School := {
    total_students := 1000,
    male_ratio := 3/5,
    female_ratio := 2/5,
    male_basketball_ratio := 2/3,
    female_basketball_ratio := 1/5
  }
  percentage_not_liking_basketball s = 52 := by
  sorry

end NUMINAMATH_CALUDE_percentage_not_liking_basketball_is_52_percent_l2582_258259


namespace NUMINAMATH_CALUDE_sqrt_of_four_l2582_258278

theorem sqrt_of_four : Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_four_l2582_258278


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l2582_258258

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + |x| + 1

-- State the theorem
theorem f_satisfies_conditions :
  (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0) ∧
  (∀ x : ℝ, f x = f (-x)) := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l2582_258258


namespace NUMINAMATH_CALUDE_average_temperature_proof_l2582_258237

def daily_temperatures : List ℝ := [40, 50, 65, 36, 82, 72, 26]
def days_in_week : ℕ := 7

theorem average_temperature_proof :
  (daily_temperatures.sum / days_in_week : ℝ) = 53 := by
  sorry

end NUMINAMATH_CALUDE_average_temperature_proof_l2582_258237


namespace NUMINAMATH_CALUDE_factory_weekly_production_l2582_258227

/-- Calculates the weekly toy production of a factory -/
def weekly_production (days_worked : ℕ) (daily_production : ℕ) : ℕ :=
  days_worked * daily_production

/-- Proves that the factory produces 4340 toys per week -/
theorem factory_weekly_production :
  let days_worked : ℕ := 2
  let daily_production : ℕ := 2170
  weekly_production days_worked daily_production = 4340 := by
sorry

end NUMINAMATH_CALUDE_factory_weekly_production_l2582_258227


namespace NUMINAMATH_CALUDE_trig_expression_value_l2582_258252

theorem trig_expression_value (α : ℝ) (h : Real.tan α = -2) :
  (Real.sin (2 * α) - Real.cos α ^ 2) / Real.sin α ^ 2 = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_value_l2582_258252


namespace NUMINAMATH_CALUDE_remainder_of_3n_mod_7_l2582_258267

theorem remainder_of_3n_mod_7 (n : ℤ) (h : n % 7 = 1) : (3 * n) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_3n_mod_7_l2582_258267


namespace NUMINAMATH_CALUDE_net_amount_calculation_l2582_258295

/-- Calculates the net amount received after selling a stock and deducting brokerage -/
def net_amount_after_brokerage (sale_amount : ℚ) (brokerage_rate : ℚ) : ℚ :=
  sale_amount - (sale_amount * brokerage_rate)

/-- Theorem stating that the net amount received after selling a stock for Rs. 108.25 
    with a 1/4% brokerage rate is Rs. 107.98 -/
theorem net_amount_calculation :
  let sale_amount : ℚ := 108.25
  let brokerage_rate : ℚ := 1 / 400  -- 1/4% expressed as a fraction
  net_amount_after_brokerage sale_amount brokerage_rate = 107.98 := by
  sorry

#eval net_amount_after_brokerage 108.25 (1 / 400)

end NUMINAMATH_CALUDE_net_amount_calculation_l2582_258295


namespace NUMINAMATH_CALUDE_train_length_l2582_258285

/-- The length of a train given its speed, platform length, and crossing time -/
theorem train_length (speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  speed = 72 * 1000 / 3600 → 
  platform_length = 300 → 
  crossing_time = 26 → 
  speed * crossing_time - platform_length = 220 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l2582_258285


namespace NUMINAMATH_CALUDE_max_triangle_area_l2582_258208

/-- Parabola with focus at (0,1) and equation x^2 = 4y -/
def Parabola : Set (ℝ × ℝ) :=
  {p | p.1^2 = 4 * p.2}

/-- The focus of the parabola -/
def F : ℝ × ℝ := (0, 1)

/-- Vector from F to a point -/
def vec (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - F.1, p.2 - F.2)

/-- Condition that A, B, C are on the parabola and FA + FB + FC = 0 -/
def PointsCondition (A B C : ℝ × ℝ) : Prop :=
  A ∈ Parabola ∧ B ∈ Parabola ∧ C ∈ Parabola ∧
  vec A + vec B + vec C = (0, 0)

/-- Area of a triangle given three points -/
noncomputable def TriangleArea (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

/-- The theorem to be proved -/
theorem max_triangle_area :
  ∀ A B C : ℝ × ℝ,
  PointsCondition A B C →
  TriangleArea A B C ≤ (3 * Real.sqrt 6) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l2582_258208


namespace NUMINAMATH_CALUDE_eight_solutions_l2582_258293

/-- The function f(x) = x^2 - 2 -/
def f (x : ℝ) : ℝ := x^2 - 2

/-- The theorem stating that f(f(f(x))) = x has exactly eight distinct real solutions -/
theorem eight_solutions :
  ∃! (s : Finset ℝ), s.card = 8 ∧ (∀ x ∈ s, f (f (f x)) = x) ∧
    (∀ y : ℝ, f (f (f y)) = y → y ∈ s) :=
sorry

end NUMINAMATH_CALUDE_eight_solutions_l2582_258293


namespace NUMINAMATH_CALUDE_additional_toothpicks_3_to_5_l2582_258233

/-- The number of toothpicks needed for a staircase of n steps -/
def toothpicks (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 4
  else if n = 2 then 10
  else if n = 3 then 18
  else toothpicks (n - 1) + 2 * n + 2

theorem additional_toothpicks_3_to_5 :
  toothpicks 5 - toothpicks 3 = 22 :=
sorry

end NUMINAMATH_CALUDE_additional_toothpicks_3_to_5_l2582_258233


namespace NUMINAMATH_CALUDE_total_fruits_in_baskets_total_fruits_proof_l2582_258253

/-- Given a group of 4 fruit baskets, where the first three baskets contain 9 apples, 
    15 oranges, and 14 bananas each, and the fourth basket contains 2 less of each fruit 
    compared to the other baskets, prove that the total number of fruits is 70. -/
theorem total_fruits_in_baskets : ℕ :=
  let apples_per_basket : ℕ := 9
  let oranges_per_basket : ℕ := 15
  let bananas_per_basket : ℕ := 14
  let num_regular_baskets : ℕ := 3
  let fruits_per_regular_basket : ℕ := apples_per_basket + oranges_per_basket + bananas_per_basket
  let fruits_in_regular_baskets : ℕ := fruits_per_regular_basket * num_regular_baskets
  let reduction_in_last_basket : ℕ := 2
  let fruits_in_last_basket : ℕ := fruits_per_regular_basket - (3 * reduction_in_last_basket)
  let total_fruits : ℕ := fruits_in_regular_baskets + fruits_in_last_basket
  70

theorem total_fruits_proof : total_fruits_in_baskets = 70 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_in_baskets_total_fruits_proof_l2582_258253


namespace NUMINAMATH_CALUDE_decimal_point_problem_l2582_258217

theorem decimal_point_problem :
  ∃! (x : ℝ), x > 0 ∧ 100000 * x = 5 * (1 / x) ∧ x = Real.sqrt 2 / 200 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l2582_258217


namespace NUMINAMATH_CALUDE_equation_solutions_l2582_258251

theorem equation_solutions :
  {x : ℝ | x * (2 * x + 1) = 2 * x + 1} = {-1/2, 1} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2582_258251


namespace NUMINAMATH_CALUDE_system_solution_l2582_258200

theorem system_solution : ∃ (X Y : ℝ), 
  (X + (X + 2*Y) / (X^2 + Y^2) = 2 ∧ 
   Y + (2*X - Y) / (X^2 + Y^2) = 0) ↔ 
  ((X = 0 ∧ Y = 1) ∨ (X = 2 ∧ Y = -1)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2582_258200


namespace NUMINAMATH_CALUDE_triangle_height_and_median_l2582_258205

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def triangle : Triangle := {
  A := (4, 0)
  B := (6, 6)
  C := (0, 2)
}

def is_height_equation (t : Triangle) (eq : ℝ → ℝ → ℝ) : Prop :=
  let (x₁, y₁) := t.A
  ∀ x y, eq x y = 0 ↔ 3 * x + 2 * y - 12 = 0

def is_median_equation (t : Triangle) (eq : ℝ → ℝ → ℝ) : Prop :=
  let (x₁, y₁) := t.B
  ∀ x y, eq x y = 0 ↔ x + 2 * y - 18 = 0

theorem triangle_height_and_median :
  ∃ (height_eq median_eq : ℝ → ℝ → ℝ),
    is_height_equation triangle height_eq ∧
    is_median_equation triangle median_eq :=
  sorry

end NUMINAMATH_CALUDE_triangle_height_and_median_l2582_258205


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a6_l2582_258268

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_a6 (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 4 = 1 → a 7 = 16 → a 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a6_l2582_258268


namespace NUMINAMATH_CALUDE_thirteen_pow_seven_mod_eleven_l2582_258221

theorem thirteen_pow_seven_mod_eleven : 13^7 % 11 = 7 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_pow_seven_mod_eleven_l2582_258221


namespace NUMINAMATH_CALUDE_arithmetic_geometric_harmonic_mean_sum_of_squares_l2582_258257

theorem arithmetic_geometric_harmonic_mean_sum_of_squares
  (a b c : ℝ)
  (h_arithmetic : (a + b + c) / 3 = 8)
  (h_geometric : (a * b * c) ^ (1/3 : ℝ) = 5)
  (h_harmonic : 3 / (1/a + 1/b + 1/c) = 3) :
  a^2 + b^2 + c^2 = 326 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_harmonic_mean_sum_of_squares_l2582_258257


namespace NUMINAMATH_CALUDE_dvd_cost_l2582_258239

/-- Given that two identical DVDs cost $36, prove that six of these DVDs cost $108. -/
theorem dvd_cost (two_dvd_cost : ℕ) (h : two_dvd_cost = 36) : 
  (6 * two_dvd_cost / 2 : ℚ) = 108 := by sorry

end NUMINAMATH_CALUDE_dvd_cost_l2582_258239


namespace NUMINAMATH_CALUDE_total_protein_consumed_l2582_258288

-- Define the protein content for each food item
def collagen_protein_per_2_scoops : ℚ := 18
def protein_powder_per_scoop : ℚ := 21
def steak_protein : ℚ := 56
def greek_yogurt_protein : ℚ := 15
def almond_protein_per_quarter_cup : ℚ := 6

-- Define the consumption quantities
def collagen_scoops : ℚ := 1
def protein_powder_scoops : ℚ := 2
def steak_quantity : ℚ := 1
def greek_yogurt_servings : ℚ := 1
def almond_cups : ℚ := 1/2

-- Theorem statement
theorem total_protein_consumed :
  (collagen_scoops / 2 * collagen_protein_per_2_scoops) +
  (protein_powder_scoops * protein_powder_per_scoop) +
  (steak_quantity * steak_protein) +
  (greek_yogurt_servings * greek_yogurt_protein) +
  (almond_cups / (1/4) * almond_protein_per_quarter_cup) = 134 := by
  sorry

end NUMINAMATH_CALUDE_total_protein_consumed_l2582_258288


namespace NUMINAMATH_CALUDE_michael_pet_sitting_cost_l2582_258266

/-- Calculates the total cost of pet sitting for one night -/
def pet_sitting_cost (num_cats num_dogs num_parrots num_fish : ℕ) 
                     (cost_per_cat cost_per_dog cost_per_parrot cost_per_fish : ℕ) : ℕ :=
  num_cats * cost_per_cat + 
  num_dogs * cost_per_dog + 
  num_parrots * cost_per_parrot + 
  num_fish * cost_per_fish

/-- Theorem: The total cost of pet sitting for Michael's pets for one night is $106 -/
theorem michael_pet_sitting_cost : 
  pet_sitting_cost 2 3 1 4 13 18 10 4 = 106 := by
  sorry

end NUMINAMATH_CALUDE_michael_pet_sitting_cost_l2582_258266


namespace NUMINAMATH_CALUDE_P_necessary_not_sufficient_for_Q_l2582_258275

theorem P_necessary_not_sufficient_for_Q :
  (∀ x : ℝ, (x + 2) * (x - 1) < 0 → x < 1) ∧
  (∃ x : ℝ, x < 1 ∧ (x + 2) * (x - 1) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_P_necessary_not_sufficient_for_Q_l2582_258275


namespace NUMINAMATH_CALUDE_shopkeeper_visits_l2582_258249

theorem shopkeeper_visits (initial_amount : ℚ) (spent_per_shop : ℚ) : initial_amount = 8.75 ∧ spent_per_shop = 10 →
  ∃ n : ℕ, n = 3 ∧ 2^n * initial_amount - spent_per_shop * (2^n - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_visits_l2582_258249


namespace NUMINAMATH_CALUDE_line_parallel_plane_not_all_lines_l2582_258255

/-- A plane in 3D space -/
structure Plane3D where
  -- Define plane properties here
  mk :: -- Constructor

/-- A line in 3D space -/
structure Line3D where
  -- Define line properties here
  mk :: -- Constructor

/-- Defines when a line is parallel to a plane -/
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines when two lines are parallel -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Defines when a line is in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

theorem line_parallel_plane_not_all_lines 
  (p : Plane3D) : 
  ∃ (l : Line3D), parallel_line_plane l p ∧ 
  ∃ (m : Line3D), line_in_plane m p ∧ ¬parallel_lines l m :=
sorry

end NUMINAMATH_CALUDE_line_parallel_plane_not_all_lines_l2582_258255


namespace NUMINAMATH_CALUDE_jessica_cut_orchids_l2582_258216

/-- The number of orchids initially in the vase -/
def initial_orchids : ℕ := 2

/-- The number of orchids in the vase after cutting -/
def final_orchids : ℕ := 21

/-- The number of orchids Jessica cut -/
def orchids_cut : ℕ := final_orchids - initial_orchids

theorem jessica_cut_orchids : orchids_cut = 19 := by
  sorry

end NUMINAMATH_CALUDE_jessica_cut_orchids_l2582_258216


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_l2582_258250

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the inequality function
def inequality_function (t : Triangle) : ℝ :=
  t.a^2 * t.b * (t.a - t.b) + t.b^2 * t.c * (t.b - t.c) + t.c^2 * t.a * (t.c - t.a)

-- State the theorem
theorem triangle_inequality_theorem (t : Triangle) :
  inequality_function t ≥ 0 ∧
  (inequality_function t = 0 ↔ t.a = t.b ∧ t.b = t.c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_l2582_258250


namespace NUMINAMATH_CALUDE_triangle_area_squared_l2582_258219

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circle
def Circle := {p : ℝ × ℝ | (p.1)^2 + (p.2)^2 = 16}

-- Define the conditions
def isInscribed (t : Triangle) : Prop :=
  t.A ∈ Circle ∧ t.B ∈ Circle ∧ t.C ∈ Circle

def angleA (t : Triangle) : ℝ := sorry

def sideDifference (t : Triangle) : ℝ := sorry

def area (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_area_squared (t : Triangle) 
  (h1 : isInscribed t)
  (h2 : angleA t = π / 3)  -- 60 degrees in radians
  (h3 : sideDifference t = 4)
  : (area t)^2 = 192 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_squared_l2582_258219


namespace NUMINAMATH_CALUDE_exam_max_score_l2582_258207

/-- The maximum score awarded in an exam given the following conditions:
    1. Gibi scored 59 percent
    2. Jigi scored 55 percent
    3. Mike scored 99 percent
    4. Lizzy scored 67 percent
    5. The average mark scored by all 4 students is 490 -/
theorem exam_max_score :
  let gibi_percent : ℚ := 59 / 100
  let jigi_percent : ℚ := 55 / 100
  let mike_percent : ℚ := 99 / 100
  let lizzy_percent : ℚ := 67 / 100
  let num_students : ℕ := 4
  let average_score : ℚ := 490
  let total_score : ℚ := average_score * num_students
  let sum_percentages : ℚ := gibi_percent + jigi_percent + mike_percent + lizzy_percent
  max_score * sum_percentages = total_score →
  max_score = 700 := by
sorry


end NUMINAMATH_CALUDE_exam_max_score_l2582_258207


namespace NUMINAMATH_CALUDE_equation_solution_l2582_258286

theorem equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ - 3) * (x₁ + 1) = 5 ∧ 
  (x₂ - 3) * (x₂ + 1) = 5 ∧ 
  x₁ = 4 ∧ 
  x₂ = -2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2582_258286


namespace NUMINAMATH_CALUDE_steps_down_empire_state_proof_l2582_258240

/-- The number of steps taken to get down the Empire State Building -/
def steps_down_empire_state : ℕ := sorry

/-- The number of steps taken from the Empire State Building to Madison Square Garden -/
def steps_to_madison_square : ℕ := 315

/-- The total number of steps taken to get to Madison Square Garden -/
def total_steps : ℕ := 991

/-- Theorem stating that the number of steps taken to get down the Empire State Building is 676 -/
theorem steps_down_empire_state_proof : 
  steps_down_empire_state = total_steps - steps_to_madison_square := by sorry

end NUMINAMATH_CALUDE_steps_down_empire_state_proof_l2582_258240


namespace NUMINAMATH_CALUDE_plane_perpendicular_from_line_l2582_258261

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- State the theorem
theorem plane_perpendicular_from_line
  (α β γ : Plane) (l : Line)
  (distinct : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)
  (h1 : perpendicular_line_plane l α)
  (h2 : parallel l β) :
  perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicular_from_line_l2582_258261


namespace NUMINAMATH_CALUDE_number_count_proof_l2582_258225

theorem number_count_proof (total_avg : ℝ) (group1_avg : ℝ) (group2_avg : ℝ) (group3_avg : ℝ) :
  total_avg = 3.95 →
  group1_avg = 4.2 →
  group2_avg = 3.85 →
  group3_avg = 3.8000000000000007 →
  (2 * group1_avg + 2 * group2_avg + 2 * group3_avg) / total_avg = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_number_count_proof_l2582_258225


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2582_258247

theorem min_value_expression (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) ≥ 2 * Real.sqrt 5 :=
by sorry

theorem min_value_achievable : 
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2582_258247


namespace NUMINAMATH_CALUDE_inequality_proof_l2582_258218

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b * c * (a + b + c) = 3) : 
  (a + b) * (b + c) * (c + a) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2582_258218


namespace NUMINAMATH_CALUDE_common_sum_is_negative_fifteen_l2582_258290

def is_valid_arrangement (arr : Matrix (Fin 5) (Fin 5) ℤ) : Prop :=
  ∀ i j, -15 ≤ arr i j ∧ arr i j ≤ 9

def row_sum (arr : Matrix (Fin 5) (Fin 5) ℤ) (i : Fin 5) : ℤ :=
  (Finset.range 5).sum (λ j => arr i j)

def col_sum (arr : Matrix (Fin 5) (Fin 5) ℤ) (j : Fin 5) : ℤ :=
  (Finset.range 5).sum (λ i => arr i j)

def main_diagonal_sum (arr : Matrix (Fin 5) (Fin 5) ℤ) : ℤ :=
  (Finset.range 5).sum (λ i => arr i i)

def anti_diagonal_sum (arr : Matrix (Fin 5) (Fin 5) ℤ) : ℤ :=
  (Finset.range 5).sum (λ i => arr i (4 - i))

def all_sums_equal (arr : Matrix (Fin 5) (Fin 5) ℤ) : Prop :=
  ∃ s, (∀ i, row_sum arr i = s) ∧
       (∀ j, col_sum arr j = s) ∧
       (main_diagonal_sum arr = s) ∧
       (anti_diagonal_sum arr = s)

theorem common_sum_is_negative_fifteen
  (arr : Matrix (Fin 5) (Fin 5) ℤ)
  (h1 : is_valid_arrangement arr)
  (h2 : all_sums_equal arr) :
  ∃ s, s = -15 ∧ all_sums_equal arr ∧ (∀ i j, row_sum arr i = s ∧ col_sum arr j = s) :=
sorry

end NUMINAMATH_CALUDE_common_sum_is_negative_fifteen_l2582_258290


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2582_258210

theorem inequality_equivalence (a b c : ℕ+) :
  (∀ x y z : ℝ, (x - y) ^ a.val * (x - z) ^ b.val * (y - z) ^ c.val ≥ 0) ↔ 
  (∃ m n p : ℕ, a.val = 2 * m ∧ b.val = 2 * n ∧ c.val = 2 * p) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2582_258210


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2582_258222

open Real

theorem trigonometric_identities (α : ℝ) (h : tan α = 2) :
  (sin α + 2 * cos α) / (4 * cos α - sin α) = 2 ∧
  sin α * cos α + cos α ^ 2 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2582_258222


namespace NUMINAMATH_CALUDE_product_45_sum_5_l2582_258214

theorem product_45_sum_5 (v w x y z : ℤ) : 
  v ≠ w ∧ v ≠ x ∧ v ≠ y ∧ v ≠ z ∧
  w ≠ x ∧ w ≠ y ∧ w ≠ z ∧
  x ≠ y ∧ x ≠ z ∧
  y ≠ z ∧
  v * w * x * y * z = 45 →
  v + w + x + y + z = 5 := by
sorry

end NUMINAMATH_CALUDE_product_45_sum_5_l2582_258214


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2582_258230

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < -2) :
  {x : ℝ | a * x^2 + (a - 2) * x - 2 ≥ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 2/a} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2582_258230


namespace NUMINAMATH_CALUDE_feb_2_is_tuesday_l2582_258241

-- Define the days of the week
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

-- Define a function to get the day of the week given a number of days before Sunday
def daysBefore (n : Nat) : DayOfWeek :=
  match n % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Saturday
  | 2 => DayOfWeek.Friday
  | 3 => DayOfWeek.Thursday
  | 4 => DayOfWeek.Wednesday
  | 5 => DayOfWeek.Tuesday
  | _ => DayOfWeek.Monday

-- Theorem statement
theorem feb_2_is_tuesday (h : DayOfWeek.Sunday = daysBefore 0) :
  DayOfWeek.Tuesday = daysBefore 12 := by
  sorry


end NUMINAMATH_CALUDE_feb_2_is_tuesday_l2582_258241


namespace NUMINAMATH_CALUDE_sum_of_products_l2582_258203

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 108)
  (eq2 : y^2 + y*z + z^2 = 16)
  (eq3 : z^2 + x*z + x^2 = 124) :
  x*y + y*z + x*z = 48 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l2582_258203


namespace NUMINAMATH_CALUDE_negation_equivalence_l2582_258202

theorem negation_equivalence :
  (¬ ∃ x : ℤ, 7 ∣ x ∧ ¬ Odd x) ↔ (∀ x : ℤ, 7 ∣ x → Odd x) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2582_258202


namespace NUMINAMATH_CALUDE_factors_of_x4_plus_16_l2582_258243

theorem factors_of_x4_plus_16 (x : ℝ) : 
  (x^4 + 16 : ℝ) = (x^2 - 4*x + 4) * (x^2 + 4*x + 4) := by
  sorry

end NUMINAMATH_CALUDE_factors_of_x4_plus_16_l2582_258243


namespace NUMINAMATH_CALUDE_A_3_1_l2582_258264

def A : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_1 : A 3 1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_A_3_1_l2582_258264


namespace NUMINAMATH_CALUDE_probability_one_common_number_l2582_258201

/-- The number of numbers in the lottery -/
def totalNumbers : ℕ := 45

/-- The number of numbers each participant chooses -/
def chosenNumbers : ℕ := 6

/-- The probability of exactly one common number between two independently chosen combinations -/
def probabilityOneCommon : ℚ :=
  (chosenNumbers : ℚ) * (Nat.choose (totalNumbers - chosenNumbers) (chosenNumbers - 1) : ℚ) /
  (Nat.choose totalNumbers chosenNumbers : ℚ)

/-- Theorem stating the probability of exactly one common number -/
theorem probability_one_common_number :
  probabilityOneCommon = (6 : ℚ) * (Nat.choose 39 5 : ℚ) / (Nat.choose 45 6 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_probability_one_common_number_l2582_258201


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2582_258281

/-- Represents a hyperbola with equation x²/a² - y²/b² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- Represents a line with equation x - √3y + 2 = 0 -/
def special_line (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 2 = 0

theorem hyperbola_real_axis_length
  (h : Hyperbola)
  (focus_on_line : ∃ (x y : ℝ), special_line x y ∧ x^2 / h.a^2 - y^2 / h.b^2 = 1)
  (perpendicular_to_asymptote : h.b / h.a = Real.sqrt 3) :
  2 * h.a = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2582_258281


namespace NUMINAMATH_CALUDE_tangent_chord_distance_l2582_258232

theorem tangent_chord_distance (R a : ℝ) (h : R > 0) :
  let x := R
  let m := 2 * R
  16 * R^2 * x^4 - 16 * R^2 * x^2 * (a^2 + R^2) + 16 * a^4 * R^4 - a^2 * (4 * R^2 - m^2)^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_chord_distance_l2582_258232


namespace NUMINAMATH_CALUDE_complex_imaginary_solution_l2582_258256

/-- Given that z = m^2 - (1-i)m is an imaginary number, prove that m = 1 -/
theorem complex_imaginary_solution (m : ℂ) : 
  let z := m^2 - (1 - Complex.I) * m
  (∃ (y : ℝ), z = Complex.I * y) ∧ z ≠ 0 → m = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_imaginary_solution_l2582_258256


namespace NUMINAMATH_CALUDE_fourth_number_in_sequence_l2582_258223

theorem fourth_number_in_sequence (s : Fin 7 → ℝ) 
  (h1 : (s 0 + s 1 + s 2 + s 3) / 4 = 4)
  (h2 : (s 3 + s 4 + s 5 + s 6) / 4 = 4)
  (h3 : (s 0 + s 1 + s 2 + s 3 + s 4 + s 5 + s 6) / 7 = 3) :
  s 3 = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_in_sequence_l2582_258223


namespace NUMINAMATH_CALUDE_inequality_proof_l2582_258291

theorem inequality_proof (a b c : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ 1) (h3 : 1 ≥ c) (h4 : c ≥ 0) 
  (h5 : a + b + c = 3) : 
  2 ≤ a * b + b * c + c * a ∧ 
  a * b + b * c + c * a ≤ 3 ∧
  24 / (a^3 + b^3 + c^3) + 25 / (a * b + b * c + c * a) ≥ 14 := by
  sorry

#check inequality_proof

end NUMINAMATH_CALUDE_inequality_proof_l2582_258291


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2582_258269

/-- Given a line mx + ny + 2 = 0 intersecting a circle (x+3)^2 + (y+1)^2 = 1 at a chord of length 2,
    the minimum value of 1/m + 3/n is 6, where m > 0 and n > 0 -/
theorem min_value_of_expression (m n : ℝ) : 
  m > 0 → n > 0 → 
  (∃ (x y : ℝ), m*x + n*y + 2 = 0 ∧ (x+3)^2 + (y+1)^2 = 1) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), m*x₁ + n*y₁ + 2 = 0 ∧ m*x₂ + n*y₂ + 2 = 0 ∧
                         (x₁+3)^2 + (y₁+1)^2 = 1 ∧ (x₂+3)^2 + (y₂+1)^2 = 1 ∧
                         (x₁-x₂)^2 + (y₁-y₂)^2 = 4) →
  (1/m + 3/n ≥ 6) ∧ (∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 1/m₀ + 3/n₀ = 6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2582_258269


namespace NUMINAMATH_CALUDE_min_guesses_correct_l2582_258228

/-- The minimum number of guesses required to determine a binary string -/
def minGuesses (n k : ℕ) : ℕ :=
  if n = 2 * k then 2 else 1

theorem min_guesses_correct (n k : ℕ) (h1 : n > k) (h2 : k > 0) :
  minGuesses n k = (if n = 2 * k then 2 else 1) :=
by sorry

end NUMINAMATH_CALUDE_min_guesses_correct_l2582_258228


namespace NUMINAMATH_CALUDE_complex_power_difference_l2582_258283

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference :
  (1 + 2*i)^8 - (1 - 2*i)^8 = 672*i :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l2582_258283


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l2582_258234

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def passesThrough (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

def hasEqualIntercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = l.c / l.b

theorem line_through_point_with_equal_intercepts :
  ∃ (l₁ l₂ : Line),
    (passesThrough l₁ ⟨1, 2⟩ ∧ hasEqualIntercepts l₁) ∧
    (passesThrough l₂ ⟨1, 2⟩ ∧ hasEqualIntercepts l₂) ∧
    ((l₁.a = 2 ∧ l₁.b = -1 ∧ l₁.c = 0) ∨
     (l₂.a = 1 ∧ l₂.b = 1 ∧ l₂.c = -3)) :=
  sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l2582_258234


namespace NUMINAMATH_CALUDE_rectangular_hall_dimension_difference_l2582_258282

/-- Proves that for a rectangular hall with width half the length and area 128 sq. m,
    the difference between length and width is 8 meters. -/
theorem rectangular_hall_dimension_difference :
  ∀ (length width : ℝ),
    width = length / 2 →
    length * width = 128 →
    length - width = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_hall_dimension_difference_l2582_258282


namespace NUMINAMATH_CALUDE_right_distance_is_73_l2582_258244

/-- Represents a square table with a centered round plate -/
structure TableWithPlate where
  /-- Length of the square table's side -/
  table_side : ℝ
  /-- Diameter of the round plate -/
  plate_diameter : ℝ
  /-- Distance from plate edge to left table edge -/
  left_distance : ℝ
  /-- Distance from plate edge to top table edge -/
  top_distance : ℝ
  /-- Distance from plate edge to bottom table edge -/
  bottom_distance : ℝ
  /-- The plate is centered on the table -/
  centered : left_distance + plate_diameter + (table_side - left_distance - plate_diameter) = top_distance + plate_diameter + bottom_distance

/-- Theorem stating the distance from plate edge to right table edge -/
theorem right_distance_is_73 (t : TableWithPlate) (h1 : t.left_distance = 10) (h2 : t.top_distance = 63) (h3 : t.bottom_distance = 20) : 
  t.table_side - t.left_distance - t.plate_diameter = 73 := by
  sorry

end NUMINAMATH_CALUDE_right_distance_is_73_l2582_258244


namespace NUMINAMATH_CALUDE_count_pairs_theorem_l2582_258265

/-- The number of integer pairs (m, n) satisfying the given inequality -/
def count_pairs : ℕ := 1000

/-- The lower bound for m -/
def m_lower_bound : ℕ := 1

/-- The upper bound for m -/
def m_upper_bound : ℕ := 3000

/-- Predicate to check if a pair (m, n) satisfies the inequality -/
def satisfies_inequality (m n : ℕ) : Prop :=
  (5 : ℝ)^n < (3 : ℝ)^m ∧ (3 : ℝ)^m < (3 : ℝ)^(m+1) ∧ (3 : ℝ)^(m+1) < (5 : ℝ)^(n+1)

theorem count_pairs_theorem :
  ∃ S : Finset (ℕ × ℕ),
    S.card = count_pairs ∧
    (∀ (m n : ℕ), (m, n) ∈ S ↔ 
      m_lower_bound ≤ m ∧ m ≤ m_upper_bound ∧ satisfies_inequality m n) :=
sorry

end NUMINAMATH_CALUDE_count_pairs_theorem_l2582_258265


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2582_258271

/-- An arithmetic sequence {a_n} with the given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 3 + a 4 + a 5 = 12 →
  a 6 = 2 →
  a 2 + a 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2582_258271


namespace NUMINAMATH_CALUDE_can_repair_propeller_l2582_258280

/-- The cost of a blade in tugriks -/
def blade_cost : ℕ := 120

/-- The cost of a screw in tugriks -/
def screw_cost : ℕ := 9

/-- The discount rate applied after spending 250 tugriks -/
def discount_rate : ℚ := 0.2

/-- The threshold for applying the discount -/
def discount_threshold : ℕ := 250

/-- Karlsson's budget in tugriks -/
def budget : ℕ := 360

/-- The number of blades needed -/
def blades_needed : ℕ := 3

/-- The number of screws needed -/
def screws_needed : ℕ := 1

/-- Function to calculate the total cost with discount -/
def total_cost_with_discount (blade_cost screw_cost : ℕ) (discount_rate : ℚ) 
  (discount_threshold blades_needed screws_needed : ℕ) : ℚ :=
  let initial_purchase := 2 * blade_cost + 2 * screw_cost
  let remaining_purchase := blade_cost
  if initial_purchase ≥ discount_threshold
  then initial_purchase + remaining_purchase * (1 - discount_rate)
  else initial_purchase + remaining_purchase

/-- Theorem stating that Karlsson can afford to repair his propeller -/
theorem can_repair_propeller : 
  total_cost_with_discount blade_cost screw_cost discount_rate 
    discount_threshold blades_needed screws_needed ≤ budget := by
  sorry

end NUMINAMATH_CALUDE_can_repair_propeller_l2582_258280


namespace NUMINAMATH_CALUDE_purchase_price_calculation_l2582_258296

/-- Given a markup that includes overhead and net profit, calculate the purchase price. -/
theorem purchase_price_calculation (markup : ℝ) (overhead_rate : ℝ) (net_profit : ℝ) : 
  markup = 35 ∧ overhead_rate = 0.1 ∧ net_profit = 12 →
  ∃ (price : ℝ), price = 230 ∧ markup = overhead_rate * price + net_profit :=
by sorry

end NUMINAMATH_CALUDE_purchase_price_calculation_l2582_258296


namespace NUMINAMATH_CALUDE_sum_of_two_positive_integers_greater_than_one_l2582_258242

theorem sum_of_two_positive_integers_greater_than_one (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  a + b > 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_two_positive_integers_greater_than_one_l2582_258242


namespace NUMINAMATH_CALUDE_remainder_a52_mod_52_l2582_258276

def concatenate_integers (n : ℕ) : ℕ :=
  -- Definition of a_n as the concatenation of integers from 1 to n
  sorry

theorem remainder_a52_mod_52 : concatenate_integers 52 % 52 = 28 := by
  sorry

end NUMINAMATH_CALUDE_remainder_a52_mod_52_l2582_258276


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2582_258254

def f (m n x : ℝ) : ℝ := x^2 + m*x + n

theorem quadratic_function_property (m n : ℝ) :
  (∀ x ∈ Set.Icc 1 5, |f m n x| ≤ 2) →
  (f m n 1 - 2*(f m n 3) + f m n 5 = 8) ∧ (m = -6 ∧ n = 7) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2582_258254


namespace NUMINAMATH_CALUDE_average_increase_l2582_258272

theorem average_increase (x₁ x₂ x₃ : ℝ) :
  (x₁ + x₂ + x₃) / 3 = 5 →
  ((x₁ + 2) + (x₂ + 2) + (x₃ + 2)) / 3 = 7 := by
sorry

end NUMINAMATH_CALUDE_average_increase_l2582_258272


namespace NUMINAMATH_CALUDE_cos_pi_minus_alpha_l2582_258224

theorem cos_pi_minus_alpha (α : Real) 
  (h1 : 0 < α) 
  (h2 : α < π) 
  (h3 : 3 * Real.sin (2 * α) = Real.sin α) : 
  Real.cos (π - α) = -1/6 := by
sorry

end NUMINAMATH_CALUDE_cos_pi_minus_alpha_l2582_258224


namespace NUMINAMATH_CALUDE_cartesian_to_polar_equivalence_l2582_258294

theorem cartesian_to_polar_equivalence :
  ∀ (x y ρ θ : ℝ),
  x = 2 ∧ y = -2 →
  ρ = Real.sqrt (x^2 + y^2) →
  θ = Real.arctan (y / x) →
  (ρ = 2 * Real.sqrt 2 ∧ θ = -π/4) := by
  sorry

end NUMINAMATH_CALUDE_cartesian_to_polar_equivalence_l2582_258294


namespace NUMINAMATH_CALUDE_logic_statement_l2582_258270

theorem logic_statement :
  let p : Prop := 2 + 3 = 5
  let q : Prop := 5 < 4
  (p ∨ q) ∧ ¬(p ∧ q) ∧ p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_logic_statement_l2582_258270


namespace NUMINAMATH_CALUDE_range_of_t_l2582_258263

theorem range_of_t (t : ℝ) : 
  (∀ x : ℝ, (|x - t| < 1 → 1 < x ∧ x ≤ 4)) →
  (2 ≤ t ∧ t ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_t_l2582_258263


namespace NUMINAMATH_CALUDE_colored_paper_count_l2582_258209

theorem colored_paper_count (used left : ℕ) (h1 : used = 9) (h2 : left = 12) :
  used + left = 21 := by
  sorry

end NUMINAMATH_CALUDE_colored_paper_count_l2582_258209


namespace NUMINAMATH_CALUDE_guest_speaker_payment_l2582_258206

theorem guest_speaker_payment (n : ℕ) : 
  (n ≥ 200 ∧ n < 300 ∧ n % 100 ≥ 40 ∧ n % 10 = 4 ∧ n % 13 = 0) → n = 274 :=
by sorry

end NUMINAMATH_CALUDE_guest_speaker_payment_l2582_258206


namespace NUMINAMATH_CALUDE_cost_per_block_l2582_258238

/-- Proves that the cost per piece of concrete block is $2 -/
theorem cost_per_block (blocks_per_section : ℕ) (num_sections : ℕ) (total_cost : ℚ) :
  blocks_per_section = 30 →
  num_sections = 8 →
  total_cost = 480 →
  total_cost / (blocks_per_section * num_sections) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_block_l2582_258238


namespace NUMINAMATH_CALUDE_sixth_score_for_target_mean_l2582_258236

def emily_scores : List ℕ := [88, 90, 85, 92, 97]

def target_mean : ℚ := 91

theorem sixth_score_for_target_mean :
  let all_scores := emily_scores ++ [94]
  (all_scores.sum : ℚ) / all_scores.length = target_mean := by sorry

end NUMINAMATH_CALUDE_sixth_score_for_target_mean_l2582_258236


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2582_258289

/-- The total surface area of a rectangular solid -/
def total_surface_area (length width depth : ℝ) : ℝ :=
  2 * (length * width + length * depth + width * depth)

/-- Theorem: The total surface area of a rectangular solid with length 5 meters, width 4 meters, and depth 1 meter is 58 square meters -/
theorem rectangular_solid_surface_area :
  total_surface_area 5 4 1 = 58 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2582_258289


namespace NUMINAMATH_CALUDE_decimal_ratio_is_half_l2582_258204

/-- The decimal representation of 0.8571 repeating -/
def decimal_8571 : ℚ := 8571 / 9999

/-- The decimal representation of 2.142857 repeating -/
def decimal_2142857 : ℚ := 2142857 / 999999

/-- The main theorem stating that the ratio of the two decimals is 1/2 -/
theorem decimal_ratio_is_half : decimal_8571 / (2 + decimal_2142857) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_decimal_ratio_is_half_l2582_258204


namespace NUMINAMATH_CALUDE_unique_solution_l2582_258274

def cross_product (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.2.1 * w.2.2 - v.2.2 * w.2.1, v.2.2 * w.1 - v.1 * w.2.2, v.1 * w.2.1 - v.2.1 * w.1)

theorem unique_solution (a b c d e f : ℝ) :
  cross_product (3, a, c) (6, b, d) = (0, 0, 0) ∧
  cross_product (4, b, f) (8, e, d) = (0, 0, 0) →
  (a, b, c, d, e, f) = (1, 2, 1, 2, 4, 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l2582_258274


namespace NUMINAMATH_CALUDE_double_force_quadruple_power_l2582_258212

/-- Represents the scenario of tugboats pushing a barge -/
structure TugboatScenario where
  /-- Initial force applied by a single tugboat -/
  F : ℝ
  /-- Coefficient of water resistance -/
  k : ℝ
  /-- Initial speed of the barge -/
  v : ℝ
  /-- Water resistance is proportional to speed -/
  resistance_prop : F = k * v

/-- Theorem stating that doubling the force quadruples the power when water resistance is proportional to speed -/
theorem double_force_quadruple_power (scenario : TugboatScenario) :
  let v' := 2 * scenario.v  -- New speed after doubling force
  let P := scenario.F * scenario.v  -- Initial power
  let P' := (2 * scenario.F) * v'  -- New power after doubling force
  P' = 4 * P := by sorry

end NUMINAMATH_CALUDE_double_force_quadruple_power_l2582_258212


namespace NUMINAMATH_CALUDE_initial_children_count_l2582_258260

/-- The number of children who got off the bus -/
def children_off : ℕ := 22

/-- The number of children left on the bus after some got off -/
def children_left : ℕ := 21

/-- The initial number of children on the bus -/
def initial_children : ℕ := children_off + children_left

theorem initial_children_count : initial_children = 43 := by
  sorry

end NUMINAMATH_CALUDE_initial_children_count_l2582_258260


namespace NUMINAMATH_CALUDE_smallest_n_with_square_and_fifth_power_l2582_258248

theorem smallest_n_with_square_and_fifth_power : 
  (∃ (n : ℕ), n > 0 ∧ 
    (∃ (x : ℕ), 2 * n = x^2) ∧ 
    (∃ (y : ℕ), 3 * n = y^5)) →
  (∀ (m : ℕ), m > 0 → 
    (∃ (x : ℕ), 2 * m = x^2) → 
    (∃ (y : ℕ), 3 * m = y^5) → 
    m ≥ 2592) ∧
  (∃ (x : ℕ), 2 * 2592 = x^2) ∧ 
  (∃ (y : ℕ), 3 * 2592 = y^5) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_square_and_fifth_power_l2582_258248


namespace NUMINAMATH_CALUDE_sandwiches_needed_l2582_258245

theorem sandwiches_needed (total_people children adults : ℕ) 
  (h1 : total_people = 219)
  (h2 : children = 125)
  (h3 : adults = 94)
  (h4 : total_people = children + adults)
  (h5 : children * 4 + adults * 3 = 782) : 
  children * 4 + adults * 3 = 782 := by
  sorry

end NUMINAMATH_CALUDE_sandwiches_needed_l2582_258245


namespace NUMINAMATH_CALUDE_farm_animals_l2582_258246

theorem farm_animals (total_animals : ℕ) (num_ducks : ℕ) (total_legs : ℕ) 
  (h1 : total_animals = 11)
  (h2 : num_ducks = 6)
  (h3 : total_legs = 32) :
  total_animals - num_ducks = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l2582_258246


namespace NUMINAMATH_CALUDE_water_formed_ethanol_combustion_l2582_258231

/-- Represents a chemical equation with reactants and products -/
structure ChemicalEquation :=
  (reactants : List (String × ℚ))
  (products : List (String × ℚ))

/-- Represents the available moles of reactants -/
structure AvailableReactants :=
  (ethanol : ℚ)
  (oxygen : ℚ)

/-- The balanced chemical equation for ethanol combustion -/
def ethanolCombustion : ChemicalEquation :=
  { reactants := [("C2H5OH", 1), ("O2", 3)],
    products := [("CO2", 2), ("H2O", 3)] }

/-- Calculates the amount of H2O formed in the ethanol combustion reaction -/
def waterFormed (available : AvailableReactants) (equation : ChemicalEquation) : ℚ :=
  sorry

/-- Theorem stating that 2 moles of H2O are formed when 2 moles of ethanol react with 2 moles of oxygen -/
theorem water_formed_ethanol_combustion :
  waterFormed { ethanol := 2, oxygen := 2 } ethanolCombustion = 2 := by
  sorry

end NUMINAMATH_CALUDE_water_formed_ethanol_combustion_l2582_258231


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2582_258292

/-- A quadratic function with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_inequality (a b c : ℝ) (h_a : a > 0)
  (h_solution : ∀ x, QuadraticFunction a b c x > 0 ↔ x < -2 ∨ x > 4) :
  QuadraticFunction a b c 2 < QuadraticFunction a b c (-1) ∧
  QuadraticFunction a b c (-1) < QuadraticFunction a b c 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2582_258292


namespace NUMINAMATH_CALUDE_count_perfect_squares_l2582_258287

/-- The number of positive perfect square factors of (2^12)(3^15)(5^18)(7^8) -/
def num_perfect_square_factors : ℕ := 2800

/-- The product in question -/
def product : ℕ := 2^12 * 3^15 * 5^18 * 7^8

/-- A function that counts the number of positive perfect square factors of a natural number -/
def count_perfect_square_factors (n : ℕ) : ℕ := sorry

theorem count_perfect_squares :
  count_perfect_square_factors product = num_perfect_square_factors := by sorry

end NUMINAMATH_CALUDE_count_perfect_squares_l2582_258287


namespace NUMINAMATH_CALUDE_non_basketball_theater_percentage_l2582_258220

/-- Represents the student body of Maple Town High School -/
structure School where
  total : ℝ
  basketball : ℝ
  theater : ℝ
  both : ℝ

/-- The conditions given in the problem -/
def school_conditions (s : School) : Prop :=
  s.basketball = 0.7 * s.total ∧
  s.theater = 0.4 * s.total ∧
  s.both = 0.2 * s.basketball ∧
  (s.basketball - s.both) = 0.6 * (s.total - s.theater)

/-- The theorem to be proved -/
theorem non_basketball_theater_percentage (s : School) 
  (h : school_conditions s) : 
  (s.theater - s.both) / (s.total - s.basketball) = 0.87 := by
  sorry

end NUMINAMATH_CALUDE_non_basketball_theater_percentage_l2582_258220
