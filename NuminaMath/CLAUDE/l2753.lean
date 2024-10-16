import Mathlib

namespace NUMINAMATH_CALUDE_function_inequality_condition_l2753_275326

theorem function_inequality_condition (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = 2 * x + 3) →
  a > 0 →
  b > 0 →
  (∀ x, |x + 3| < b → |f x + 5| < a) ↔
  b ≤ a / 2 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l2753_275326


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2753_275360

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 5}

-- State the theorem
theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | -2 < x ∧ x < 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2753_275360


namespace NUMINAMATH_CALUDE_sin_300_degrees_l2753_275381

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  -- Define the properties of sine function
  have sin_periodic : ∀ θ k, Real.sin θ = Real.sin (θ + k * 2 * π) := by sorry
  have sin_symmetry : ∀ θ, Real.sin θ = Real.sin (-θ) := by sorry
  have sin_odd : ∀ θ, Real.sin (-θ) = -Real.sin θ := by sorry
  have sin_60_degrees : Real.sin (60 * π / 180) = Real.sqrt 3 / 2 := by sorry

  -- Proof steps would go here, but we're skipping them as per instructions
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l2753_275381


namespace NUMINAMATH_CALUDE_max_arithmetic_progressions_l2753_275357

/-- A strictly increasing sequence of 101 real numbers -/
def StrictlyIncreasingSeq (a : Fin 101 → ℝ) : Prop :=
  ∀ i j, i < j → a i < a j

/-- Three terms form an arithmetic progression -/
def IsArithmeticProgression (x y z : ℝ) : Prop :=
  y = (x + z) / 2

/-- Count of arithmetic progressions in a sequence -/
def CountArithmeticProgressions (a : Fin 101 → ℝ) : ℕ :=
  (Finset.range 50).sum (fun i => i + 1) +
  (Finset.range 49).sum (fun i => i + 1)

/-- The main theorem -/
theorem max_arithmetic_progressions (a : Fin 101 → ℝ) 
  (h : StrictlyIncreasingSeq a) :
  CountArithmeticProgressions a = 2500 :=
sorry

end NUMINAMATH_CALUDE_max_arithmetic_progressions_l2753_275357


namespace NUMINAMATH_CALUDE_football_games_total_cost_l2753_275390

/-- Calculates the total cost of attending football games over three months -/
theorem football_games_total_cost 
  (this_month_games : ℕ) (this_month_price : ℕ)
  (last_month_games : ℕ) (last_month_price : ℕ)
  (next_month_games : ℕ) (next_month_price : ℕ) :
  this_month_games = 9 →
  this_month_price = 5 →
  last_month_games = 8 →
  last_month_price = 4 →
  next_month_games = 7 →
  next_month_price = 6 →
  this_month_games * this_month_price +
  last_month_games * last_month_price +
  next_month_games * next_month_price = 119 := by
sorry

end NUMINAMATH_CALUDE_football_games_total_cost_l2753_275390


namespace NUMINAMATH_CALUDE_imaginary_number_on_real_axis_l2753_275317

theorem imaginary_number_on_real_axis (z : ℂ) :
  (∃ b : ℝ, z = b * I) →  -- z is a pure imaginary number
  (∃ r : ℝ, (z + 2) / (1 - I) = r) →  -- point lies on real axis
  z = -2 * I :=
by sorry

end NUMINAMATH_CALUDE_imaginary_number_on_real_axis_l2753_275317


namespace NUMINAMATH_CALUDE_line_circle_intersection_range_l2753_275341

theorem line_circle_intersection_range (m : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    y₁ = x₁ + m ∧ 
    y₂ = x₂ + m ∧ 
    x₁^2 + y₁^2 + 4*x₁ + 2 = 0 ∧ 
    x₂^2 + y₂^2 + 4*x₂ + 2 = 0) → 
  0 < m ∧ m < 4 :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_range_l2753_275341


namespace NUMINAMATH_CALUDE_first_car_speed_l2753_275340

theorem first_car_speed (v : ℝ) (h1 : v > 0) : 
  v * 2.25 * 4 = 720 → v * 1.25 = 100 := by
  sorry

end NUMINAMATH_CALUDE_first_car_speed_l2753_275340


namespace NUMINAMATH_CALUDE_equation_solution_l2753_275331

theorem equation_solution : ∃ (x : ℝ), x > 0 ∧ 5 * Real.sqrt (1 + x) + 5 * Real.sqrt (1 - x) = 7 * Real.sqrt 2 ∧ x = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2753_275331


namespace NUMINAMATH_CALUDE_circle_C_properties_l2753_275371

-- Define the circle C
def circle_C (x y k : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - k = 0

-- Define the center of the circle
def center_of_circle (h k : ℝ) : Prop := 
  ∀ x y k, circle_C x y k ↔ (x - h)^2 + (y - k)^2 = k + 5

-- Define the radius of the circle
def radius_of_circle (r k : ℝ) : Prop := 
  ∀ x y, circle_C x y k ↔ (x - 1)^2 + (y + 2)^2 = r^2

-- Theorem statements
theorem circle_C_properties :
  (∀ k, (∃ x y, circle_C x y k) → k > -5) ∧
  center_of_circle 1 (-2) ∧
  radius_of_circle 3 4 ∧
  (∀ k, (∃ x, circle_C x 0 k) ∧ (∀ y, y ≠ 0 → ¬circle_C x y k) → k = -1) :=
sorry

end NUMINAMATH_CALUDE_circle_C_properties_l2753_275371


namespace NUMINAMATH_CALUDE_system_solutions_l2753_275399

theorem system_solutions :
  ∃! (S : Set (ℝ × ℝ × ℝ)), 
    S = {(1, 1, 1), (-2, -2, -2)} ∧
    ∀ (x y z : ℝ), (x, y, z) ∈ S ↔ 
      (x + y * z = 2 ∧ y + z * x = 2 ∧ z + x * y = 2) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l2753_275399


namespace NUMINAMATH_CALUDE_roberto_outfits_l2753_275367

/-- The number of different outfits Roberto can put together -/
def number_of_outfits : ℕ := 180

/-- The number of pairs of trousers Roberto has -/
def number_of_trousers : ℕ := 6

/-- The number of shirts Roberto has -/
def number_of_shirts : ℕ := 8

/-- The number of jackets Roberto has -/
def number_of_jackets : ℕ := 4

/-- The number of shirts that cannot be worn with Jacket 1 -/
def number_of_restricted_shirts : ℕ := 2

theorem roberto_outfits :
  number_of_outfits = 
    number_of_trousers * number_of_shirts * number_of_jackets - 
    number_of_trousers * number_of_restricted_shirts := by
  sorry

end NUMINAMATH_CALUDE_roberto_outfits_l2753_275367


namespace NUMINAMATH_CALUDE_solve_cubic_equation_l2753_275347

theorem solve_cubic_equation :
  ∃ x : ℝ, x = -15.625 ∧ 3 * x^(1/3) - 5 * (x / x^(2/3)) = 10 + 2 * x^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_solve_cubic_equation_l2753_275347


namespace NUMINAMATH_CALUDE_isosceles_obtuse_triangle_smallest_angle_l2753_275338

/-- 
Theorem: In an isosceles, obtuse triangle where one angle is 60% larger than a right angle, 
each of the two smallest angles measures 18°.
-/
theorem isosceles_obtuse_triangle_smallest_angle :
  ∀ (a b c : ℝ), 
  -- The triangle is isosceles
  a = b →
  -- The triangle is obtuse (one angle > 90°)
  c > 90 →
  -- One angle (c) is 60% larger than a right angle
  c = 90 * 1.6 →
  -- The sum of angles in a triangle is 180°
  a + b + c = 180 →
  -- Each of the two smallest angles (a and b) measures 18°
  a = 18 ∧ b = 18 := by
sorry

end NUMINAMATH_CALUDE_isosceles_obtuse_triangle_smallest_angle_l2753_275338


namespace NUMINAMATH_CALUDE_divisors_between_squares_l2753_275321

theorem divisors_between_squares (m a b d : ℕ) : 
  1 ≤ m → 
  m^2 < a → a < m^2 + m → 
  m^2 < b → b < m^2 + m → 
  a ≠ b → 
  m^2 < d → d < m^2 + m → 
  d ∣ (a * b) → 
  d = a ∨ d = b :=
by sorry

end NUMINAMATH_CALUDE_divisors_between_squares_l2753_275321


namespace NUMINAMATH_CALUDE_vector_subtraction_l2753_275351

theorem vector_subtraction :
  let v₁ : Fin 3 → ℝ := ![-2, 5, -1]
  let v₂ : Fin 3 → ℝ := ![7, -3, 6]
  v₁ - v₂ = ![-9, 8, -7] := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l2753_275351


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2753_275358

theorem simplify_sqrt_expression :
  (Real.sqrt 500 / Real.sqrt 180) + (Real.sqrt 128 / Real.sqrt 32) = 11 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2753_275358


namespace NUMINAMATH_CALUDE_tile_problem_l2753_275378

theorem tile_problem (total_tiles : ℕ) (total_edges : ℕ) (triangular_tiles : ℕ) (square_tiles : ℕ) : 
  total_tiles = 25 →
  total_edges = 84 →
  total_tiles = triangular_tiles + square_tiles →
  total_edges = 3 * triangular_tiles + 4 * square_tiles →
  square_tiles = 9 := by
sorry

end NUMINAMATH_CALUDE_tile_problem_l2753_275378


namespace NUMINAMATH_CALUDE_volume_depends_on_length_l2753_275329

/-- Represents a rectangular prism with variable length -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  volume : ℝ
  length_positive : length > 2
  width_is_two : width = 2
  height_is_one : height = 1
  volume_formula : volume = length * width * height

/-- The volume of a rectangular prism is dependent on its length -/
theorem volume_depends_on_length (prism : RectangularPrism) :
  ∃ f : ℝ → ℝ, prism.volume = f prism.length :=
by sorry

end NUMINAMATH_CALUDE_volume_depends_on_length_l2753_275329


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2753_275301

/-- Given a geometric sequence {a_n} with common ratio q, 
    if a_1 + a_4 = 10 and a_2 + a_5 = 20, then q = 2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = q * a n) 
  (h_sum1 : a 1 + a 4 = 10) 
  (h_sum2 : a 2 + a 5 = 20) : 
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2753_275301


namespace NUMINAMATH_CALUDE_g_max_value_l2753_275310

/-- The function g(x) defined for x > 0 -/
noncomputable def g (x : ℝ) : ℝ := x * Real.log (1 + 1/x) + (1/x) * Real.log (1 + x)

/-- Theorem stating that the maximum value of g(x) for x > 0 is 2ln2 -/
theorem g_max_value : ∃ (M : ℝ), M = 2 * Real.log 2 ∧ ∀ x > 0, g x ≤ M :=
sorry

end NUMINAMATH_CALUDE_g_max_value_l2753_275310


namespace NUMINAMATH_CALUDE_divisor_problem_l2753_275393

theorem divisor_problem (original : Nat) (subtracted : Nat) (divisor : Nat) : 
  original = 427398 →
  subtracted = 8 →
  divisor = 10 →
  (original - subtracted) % divisor = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l2753_275393


namespace NUMINAMATH_CALUDE_g_equals_power_of_two_times_power_of_three_num_divisors_g_2023_equals_4096576_l2753_275350

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

end NUMINAMATH_CALUDE_g_equals_power_of_two_times_power_of_three_num_divisors_g_2023_equals_4096576_l2753_275350


namespace NUMINAMATH_CALUDE_function_property_l2753_275315

def evenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def decreasingOnNegative (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ ∧ x₂ ≤ 0 → (f x₂ - f x₁) / (x₂ - x₁) < 0

theorem function_property (f : ℝ → ℝ) (heven : evenFunction f) (hdec : decreasingOnNegative f) :
  (∀ a : ℝ, f (1 - a) > f (2 * a - 1) ↔ 0 < a ∧ a < 2/3) :=
sorry

end NUMINAMATH_CALUDE_function_property_l2753_275315


namespace NUMINAMATH_CALUDE_dog_grouping_theorem_l2753_275394

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The total number of dogs -/
def total_dogs : ℕ := 12

/-- The number of dogs in Fluffy's group -/
def fluffy_group_size : ℕ := 3

/-- The number of dogs in Nipper's group -/
def nipper_group_size : ℕ := 5

/-- The number of dogs in the third group -/
def third_group_size : ℕ := 4

theorem dog_grouping_theorem :
  choose (total_dogs - 2) (fluffy_group_size - 1) *
  choose (total_dogs - fluffy_group_size - 1) (nipper_group_size - 1) = 3150 := by
  sorry

end NUMINAMATH_CALUDE_dog_grouping_theorem_l2753_275394


namespace NUMINAMATH_CALUDE_claire_gift_card_balance_l2753_275325

/-- Calculates the remaining balance on a gift card after a week of purchases --/
def remaining_balance (gift_card_amount : ℚ) (latte_cost : ℚ) (croissant_cost : ℚ) 
  (days : ℕ) (cookie_cost : ℚ) (cookie_count : ℕ) : ℚ :=
  gift_card_amount - (latte_cost + croissant_cost) * days - cookie_cost * cookie_count

/-- Proves that the remaining balance on Claire's gift card is $43 --/
theorem claire_gift_card_balance : 
  remaining_balance 100 3.75 3.50 7 1.25 5 = 43 := by
  sorry

end NUMINAMATH_CALUDE_claire_gift_card_balance_l2753_275325


namespace NUMINAMATH_CALUDE_factory_weekly_production_l2753_275348

/-- Calculates the weekly toy production of a factory -/
def weekly_production (days_worked : ℕ) (daily_production : ℕ) : ℕ :=
  days_worked * daily_production

/-- Proves that the factory produces 4340 toys per week -/
theorem factory_weekly_production :
  let days_worked : ℕ := 2
  let daily_production : ℕ := 2170
  weekly_production days_worked daily_production = 4340 := by
sorry

end NUMINAMATH_CALUDE_factory_weekly_production_l2753_275348


namespace NUMINAMATH_CALUDE_fraction_equality_l2753_275395

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + y) / (x - 4 * y) = 3) : 
  (x + 4 * y) / (4 * x - y) = 9 / 53 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2753_275395


namespace NUMINAMATH_CALUDE_cos_315_degrees_l2753_275300

theorem cos_315_degrees : Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_315_degrees_l2753_275300


namespace NUMINAMATH_CALUDE_survivor_quitters_probability_l2753_275377

def total_participants : ℕ := 18
def tribe1_size : ℕ := 10
def tribe2_size : ℕ := 8
def quitters : ℕ := 3

theorem survivor_quitters_probability :
  let total_ways := Nat.choose total_participants quitters
  let ways_from_tribe1 := Nat.choose tribe1_size quitters
  let ways_from_tribe2 := Nat.choose tribe2_size quitters
  (ways_from_tribe1 + ways_from_tribe2 : ℚ) / total_ways = 11 / 51 := by
sorry

end NUMINAMATH_CALUDE_survivor_quitters_probability_l2753_275377


namespace NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l2753_275374

theorem smallest_integer_with_given_remainders : ∃ b : ℕ+, 
  (b : ℕ) % 4 = 3 ∧ 
  (b : ℕ) % 6 = 5 ∧ 
  ∀ k : ℕ+, (k : ℕ) % 4 = 3 ∧ (k : ℕ) % 6 = 5 → k ≥ b :=
by
  use 23
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l2753_275374


namespace NUMINAMATH_CALUDE_cube_coloring_count_l2753_275398

/-- The number of distinct orientations of a cube -/
def cubeOrientations : ℕ := 24

/-- The number of ways to permute 6 colors -/
def colorPermutations : ℕ := 720

/-- The number of distinct ways to paint a cube's faces with 6 different colors,
    where each color appears exactly once and rotations are considered identical -/
def distinctCubeColorings : ℕ := colorPermutations / cubeOrientations

theorem cube_coloring_count :
  distinctCubeColorings = 30 := by sorry

end NUMINAMATH_CALUDE_cube_coloring_count_l2753_275398


namespace NUMINAMATH_CALUDE_vector_collinearity_l2753_275383

theorem vector_collinearity (x : ℝ) : 
  let a : Fin 2 → ℝ := ![2, -1]
  let b : Fin 2 → ℝ := ![x, 1]
  (∃ (k : ℝ), k ≠ 0 ∧ (2 • a + b) = k • b) → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l2753_275383


namespace NUMINAMATH_CALUDE_max_expression_value_l2753_275312

-- Define a digit as a natural number between 0 and 9
def Digit := {n : ℕ // n ≤ 9}

-- Define the expression as a function of three digits
def Expression (x y z : Digit) : ℕ := 
  100 * x.val + 10 * y.val + z.val + 
  10 * x.val + z.val + 
  x.val

-- Theorem statement
theorem max_expression_value :
  ∃ (x y z : Digit), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    Expression x y z = 992 ∧
    ∀ (a b c : Digit), a ≠ b ∧ b ≠ c ∧ a ≠ c →
      Expression a b c ≤ 992 :=
sorry

end NUMINAMATH_CALUDE_max_expression_value_l2753_275312


namespace NUMINAMATH_CALUDE_unique_positive_integers_sum_l2753_275333

theorem unique_positive_integers_sum (y : ℝ) : 
  y = Real.sqrt ((Real.sqrt 61) / 2 + 5 / 2) →
  ∃! (p q r : ℕ+), 
    y^100 = 2*y^98 + 16*y^96 + 13*y^94 - y^50 + (p : ℝ)*y^46 + (q : ℝ)*y^44 + (r : ℝ)*y^40 ∧
    p + q + r = 382 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_integers_sum_l2753_275333


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2753_275354

theorem rectangle_perimeter (L W : ℝ) 
  (h1 : L - 4 = W + 3) 
  (h2 : (L - 4) * (W + 3) = L * W) : 
  2 * L + 2 * W = 50 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2753_275354


namespace NUMINAMATH_CALUDE_f_properties_l2753_275355

noncomputable section

/-- The function f(x) = (ax+b)e^x -/
def f (a b x : ℝ) : ℝ := (a * x + b) * Real.exp x

/-- The condition that f has an extremum at x = -1 -/
def has_extremum_at_neg_one (a b : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f a b (-1) ≥ f a b x

/-- The condition that f(x) ≥ x^2 + 2x - 1 for x ≥ -1 -/
def satisfies_inequality (a b : ℝ) : Prop :=
  ∀ x ≥ -1, f a b x ≥ x^2 + 2*x - 1

/-- The main theorem -/
theorem f_properties (a b : ℝ) 
  (h1 : has_extremum_at_neg_one a b)
  (h2 : satisfies_inequality a b) :
  b = 0 ∧ 2 / Real.exp 1 ≤ a ∧ a ≤ 2 * Real.exp 1 :=
sorry

end

end NUMINAMATH_CALUDE_f_properties_l2753_275355


namespace NUMINAMATH_CALUDE_total_stickers_l2753_275392

def folders : Nat := 3
def sheets_per_folder : Nat := 10
def stickers_red : Nat := 3
def stickers_green : Nat := 2
def stickers_blue : Nat := 1

theorem total_stickers :
  folders * sheets_per_folder * stickers_red +
  folders * sheets_per_folder * stickers_green +
  folders * sheets_per_folder * stickers_blue = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_stickers_l2753_275392


namespace NUMINAMATH_CALUDE_parking_probability_probability_equals_actual_l2753_275302

/-- The probability of finding 3 consecutive empty spaces in a row of 18 spaces 
    where 14 spaces are randomly occupied -/
theorem parking_probability : ℝ := by
  -- Define the total number of spaces
  let total_spaces : ℕ := 18
  -- Define the number of occupied spaces
  let occupied_spaces : ℕ := 14
  -- Define the number of consecutive empty spaces needed
  let required_empty_spaces : ℕ := 3
  
  -- Calculate the probability
  -- We're not providing the actual calculation here, just the structure
  sorry

-- The actual probability value
def actual_probability : ℚ := 171 / 204

-- Prove that the calculated probability equals the actual probability
theorem probability_equals_actual : parking_probability = actual_probability := by
  sorry

end NUMINAMATH_CALUDE_parking_probability_probability_equals_actual_l2753_275302


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l2753_275368

theorem at_least_one_greater_than_one (a b : ℝ) :
  a + b > 2 → max a b > 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l2753_275368


namespace NUMINAMATH_CALUDE_f_derivative_and_value_l2753_275306

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 4 + Real.cos x ^ 4

theorem f_derivative_and_value :
  (∀ x, deriv f x = -Real.sin (4 * x)) ∧
  (deriv f (π / 6) = -Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_and_value_l2753_275306


namespace NUMINAMATH_CALUDE_parallel_plane_intersection_theorem_l2753_275356

/-- A plane in 3D space -/
structure Plane where
  -- Add necessary fields for a plane

/-- A line in 3D space -/
structure Line where
  -- Add necessary fields for a line

/-- Two planes are parallel -/
def parallel_planes (α β : Plane) : Prop :=
  sorry

/-- A plane intersects another plane along a line -/
def plane_intersect (α γ : Plane) (l : Line) : Prop :=
  sorry

/-- Two lines are parallel -/
def parallel_lines (a b : Line) : Prop :=
  sorry

/-- Theorem: If two parallel planes are intersected by a third plane, 
    the lines of intersection are parallel -/
theorem parallel_plane_intersection_theorem 
  (α β γ : Plane) (a b : Line) 
  (h1 : parallel_planes α β) 
  (h2 : plane_intersect α γ a) 
  (h3 : plane_intersect β γ b) : 
  parallel_lines a b :=
sorry

end NUMINAMATH_CALUDE_parallel_plane_intersection_theorem_l2753_275356


namespace NUMINAMATH_CALUDE_right_triangle_PQR_area_l2753_275330

/-- A right triangle PQR in the xy-plane with specific properties -/
structure RightTrianglePQR where
  /-- Point P of the triangle -/
  P : ℝ × ℝ
  /-- Point Q of the triangle -/
  Q : ℝ × ℝ
  /-- Point R of the triangle (right angle) -/
  R : ℝ × ℝ
  /-- The triangle has a right angle at R -/
  right_angle_at_R : (P.1 - R.1) * (Q.1 - R.1) + (P.2 - R.2) * (Q.2 - R.2) = 0
  /-- The length of hypotenuse PQ is 50 -/
  hypotenuse_length : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 50^2
  /-- The median through P lies on the line y = x + 5 -/
  median_P : ∃ t : ℝ, (P.1 + Q.1 + R.1) / 3 = t ∧ (P.2 + Q.2 + R.2) / 3 = t + 5
  /-- The median through Q lies on the line y = 3x + 6 -/
  median_Q : ∃ t : ℝ, (P.1 + Q.1 + R.1) / 3 = t ∧ (P.2 + Q.2 + R.2) / 3 = 3 * t + 6

/-- The area of the right triangle PQR is 104.1667 -/
theorem right_triangle_PQR_area (t : RightTrianglePQR) : 
  abs ((t.P.1 - t.R.1) * (t.Q.2 - t.R.2) - (t.Q.1 - t.R.1) * (t.P.2 - t.R.2)) / 2 = 104.1667 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_PQR_area_l2753_275330


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_geometric_sequence_problem_l2753_275396

-- Arithmetic Sequence
theorem arithmetic_sequence_problem (d n a_n : ℤ) (h1 : d = 2) (h2 : n = 15) (h3 : a_n = -10) :
  let a_1 := a_n - (n - 1) * d
  let S_n := n * (a_1 + a_n) / 2
  a_1 = -38 ∧ S_n = -360 := by sorry

-- Geometric Sequence
theorem geometric_sequence_problem (a_2 a_3 a_4 : ℚ) (h1 : a_2 + a_3 = 6) (h2 : a_3 + a_4 = 12) :
  let q := a_3 / a_2
  let a_1 := a_2 / q
  let S_10 := a_1 * (1 - q^10) / (1 - q)
  q = 2 ∧ S_10 = 1023 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_geometric_sequence_problem_l2753_275396


namespace NUMINAMATH_CALUDE_f_decreasing_iff_a_range_l2753_275365

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then -x + 3*a else -(x+1)^2 + 2

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem f_decreasing_iff_a_range (a : ℝ) :
  (is_decreasing (f a)) ↔ a ≥ 1/3 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_iff_a_range_l2753_275365


namespace NUMINAMATH_CALUDE_guest_speaker_payment_l2753_275391

theorem guest_speaker_payment (n : ℕ) : 
  (n ≥ 200 ∧ n < 300 ∧ n % 100 ≥ 40 ∧ n % 10 = 4 ∧ n % 13 = 0) → n = 274 :=
by sorry

end NUMINAMATH_CALUDE_guest_speaker_payment_l2753_275391


namespace NUMINAMATH_CALUDE_min_guesses_correct_l2753_275349

/-- The minimum number of guesses required to determine a binary string -/
def minGuesses (n k : ℕ) : ℕ :=
  if n = 2 * k then 2 else 1

theorem min_guesses_correct (n k : ℕ) (h1 : n > k) (h2 : k > 0) :
  minGuesses n k = (if n = 2 * k then 2 else 1) :=
by sorry

end NUMINAMATH_CALUDE_min_guesses_correct_l2753_275349


namespace NUMINAMATH_CALUDE_triangle_problem_l2753_275305

theorem triangle_problem (A B C : Real) (a b c : Real) :
  (Real.sqrt 3 * Real.sin C - 2 * Real.cos A) * Real.sin B = (2 * Real.sin A - Real.sin C) * Real.cos B →
  a^2 + c^2 = 4 + Real.sqrt 3 →
  (1/2) * a * c * Real.sin B = (3 + Real.sqrt 3) / 4 →
  B = π / 3 ∧ a + b + c = (Real.sqrt 6 + 2 * Real.sqrt 3 + 3 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2753_275305


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2753_275313

def U : Set ℝ := Set.univ

def A : Set ℝ := {-1, 0, 1, 2, 3}

def B : Set ℝ := {x | x ≥ 2}

theorem intersection_complement_equality : A ∩ (U \ B) = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2753_275313


namespace NUMINAMATH_CALUDE_range_of_a_l2753_275322

-- Define propositions p and q
def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def q (x a : ℝ) : Prop := x > a

-- Define the theorem
theorem range_of_a :
  (∀ x a : ℝ, (¬(p x) → ¬(q x a)) ∧ (∃ x : ℝ, ¬(p x) ∧ (q x a))) →
  (∀ a : ℝ, a ≥ 1) ∧ (∀ b : ℝ, b ≥ 1 → ∃ a : ℝ, a = b) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2753_275322


namespace NUMINAMATH_CALUDE_partnership_investment_period_ratio_l2753_275318

/-- Proves that the ratio of investment periods is 2:1 given the partnership conditions --/
theorem partnership_investment_period_ratio :
  ∀ (investment_A investment_B period_A period_B profit_A profit_B : ℚ),
    investment_A = 3 * investment_B →
    ∃ k : ℚ, period_A = k * period_B →
    profit_B = 4500 →
    profit_A + profit_B = 31500 →
    profit_A / profit_B = (investment_A * period_A) / (investment_B * period_B) →
    period_A / period_B = 2 := by
  sorry

end NUMINAMATH_CALUDE_partnership_investment_period_ratio_l2753_275318


namespace NUMINAMATH_CALUDE_joan_bought_72_eggs_l2753_275319

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The number of dozens Joan bought -/
def dozens_bought : ℕ := 6

/-- The total number of eggs Joan bought -/
def total_eggs : ℕ := dozens_bought * eggs_per_dozen

theorem joan_bought_72_eggs : total_eggs = 72 := by
  sorry

end NUMINAMATH_CALUDE_joan_bought_72_eggs_l2753_275319


namespace NUMINAMATH_CALUDE_no_natural_square_difference_2014_l2753_275345

theorem no_natural_square_difference_2014 :
  ∀ m n : ℕ, m^2 ≠ n^2 + 2014 := by
sorry

end NUMINAMATH_CALUDE_no_natural_square_difference_2014_l2753_275345


namespace NUMINAMATH_CALUDE_m_range_theorem_l2753_275344

-- Define the conditions
def p (x : ℝ) : Prop := -2 < x ∧ x < 10
def q (x m : ℝ) : Prop := (x - 1)^2 - m^2 ≤ 0

-- Define the theorem
theorem m_range_theorem :
  (∀ x, p x → q x m) ∧ 
  (∃ x, q x m ∧ ¬p x) ∧ 
  (m > 0) →
  m ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l2753_275344


namespace NUMINAMATH_CALUDE_sin_2y_plus_x_l2753_275307

theorem sin_2y_plus_x (x y : Real) 
  (h1 : Real.sin x = 1/3) 
  (h2 : Real.sin (x + y) = 1) : 
  Real.sin (2*y + x) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_sin_2y_plus_x_l2753_275307


namespace NUMINAMATH_CALUDE_simplify_sum_of_surds_l2753_275332

theorem simplify_sum_of_surds : 
  Real.sqrt (9 + 6 * Real.sqrt 2) + Real.sqrt (9 - 6 * Real.sqrt 2) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sum_of_surds_l2753_275332


namespace NUMINAMATH_CALUDE_intersection_M_N_l2753_275388

def M : Set ℝ := {-2, -1, 0, 1, 2}

def N : Set ℝ := {x | x < 0 ∨ x > 3}

theorem intersection_M_N : M ∩ N = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2753_275388


namespace NUMINAMATH_CALUDE_intersection_with_ratio_l2753_275363

/-- Represents a point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line in a plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Checks if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- Theorem: Given parallel lines and points, there exists a line through B intersecting the parallel lines at X and Y with the given ratio -/
theorem intersection_with_ratio 
  (a c : Line) 
  (A B C : Point) 
  (m n : ℝ) 
  (h_parallel : Line.parallel a c)
  (h_A_on_a : A.on_line a)
  (h_C_on_c : C.on_line c)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0) :
  ∃ (X Y : Point) (l : Line),
    X.on_line a ∧
    Y.on_line c ∧
    B.on_line l ∧
    X.on_line l ∧
    Y.on_line l ∧
    ∃ (k : ℝ), k > 0 ∧ 
      (X.x - A.x)^2 + (X.y - A.y)^2 = k * m^2 ∧
      (Y.x - C.x)^2 + (Y.y - C.y)^2 = k * n^2 :=
sorry

end NUMINAMATH_CALUDE_intersection_with_ratio_l2753_275363


namespace NUMINAMATH_CALUDE_both_runners_in_picture_probability_zero_l2753_275384

/-- Represents a runner on the circular track -/
structure Runner where
  direction : Bool  -- true for counterclockwise, false for clockwise
  lap_time : ℕ      -- time to complete one lap in seconds

/-- Represents the photographer's picture -/
structure Picture where
  coverage : ℝ      -- fraction of the track covered by the picture
  center : ℝ        -- position of the center of the picture on the track (0 ≤ center < 1)

/-- Calculates the probability of both runners being in the picture -/
def probability_both_in_picture (lydia : Runner) (lucas : Runner) (pic : Picture) : ℝ :=
  sorry

/-- Theorem stating that the probability of both runners being in the picture is 0 -/
theorem both_runners_in_picture_probability_zero 
  (lydia : Runner) 
  (lucas : Runner) 
  (pic : Picture) 
  (h1 : lydia.direction = true) 
  (h2 : lydia.lap_time = 120) 
  (h3 : lucas.direction = false) 
  (h4 : lucas.lap_time = 100) 
  (h5 : pic.coverage = 1/3) 
  (h6 : pic.center = 0) : 
  probability_both_in_picture lydia lucas pic = 0 :=
sorry

end NUMINAMATH_CALUDE_both_runners_in_picture_probability_zero_l2753_275384


namespace NUMINAMATH_CALUDE_statue_weight_calculation_l2753_275387

/-- The weight of a marble statue after three weeks of carving --/
def final_statue_weight (initial_weight : ℝ) : ℝ :=
  let week1_remainder := initial_weight * (1 - 0.28)
  let week2_remainder := week1_remainder * (1 - 0.18)
  let week3_remainder := week2_remainder * (1 - 0.20)
  week3_remainder

/-- Theorem stating the final weight of the statue --/
theorem statue_weight_calculation :
  final_statue_weight 180 = 85.0176 := by
  sorry

end NUMINAMATH_CALUDE_statue_weight_calculation_l2753_275387


namespace NUMINAMATH_CALUDE_expression_value_l2753_275397

theorem expression_value : 
  (128^2 - 5^2) / (72^2 - 13^2) * ((72-13)*(72+13)) / ((128-5)*(128+5)) * (128+5) / (72+13) = 133/85 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2753_275397


namespace NUMINAMATH_CALUDE_max_k_value_l2753_275314

/-- Given positive real numbers x, y, and k satisfying the equation
    5 = k³(x²/y² + y²/x²) + k(x/y + y/x) + 2k²,
    the maximum possible value of k is approximately 0.8. -/
theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (heq : 5 = k^3 * ((x^2 / y^2) + (y^2 / x^2)) + k * (x / y + y / x) + 2 * k^2) :
  k ≤ 0.8 := by sorry

end NUMINAMATH_CALUDE_max_k_value_l2753_275314


namespace NUMINAMATH_CALUDE_square_area_error_l2753_275320

theorem square_area_error (s : ℝ) (h : s > 0) : 
  let measured_side := s * (1 + 0.02)
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 4.04 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l2753_275320


namespace NUMINAMATH_CALUDE_number_order_l2753_275382

-- Define the numbers in their respective bases
def a : ℕ := 33
def b : ℕ := 5 * 6 + 2
def c : ℕ := 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0

-- Theorem statement
theorem number_order : a > b ∧ b > c :=
sorry

end NUMINAMATH_CALUDE_number_order_l2753_275382


namespace NUMINAMATH_CALUDE_pyramid_height_l2753_275366

/-- The height of a pyramid with a rectangular base and isosceles triangular faces -/
theorem pyramid_height (ab bc : ℝ) (volume : ℝ) (h_ab : ab = 15 * Real.sqrt 3) 
  (h_bc : bc = 14 * Real.sqrt 3) (h_volume : volume = 750) : ℝ := 
  let base_area := ab * bc
  let height := 3 * volume / base_area
  by
    -- Proof goes here
    sorry

#check pyramid_height

end NUMINAMATH_CALUDE_pyramid_height_l2753_275366


namespace NUMINAMATH_CALUDE_subset_implies_a_values_l2753_275337

def M : Set ℝ := {x | x^2 + 6*x - 16 = 0}
def N (a : ℝ) : Set ℝ := {x | x*a - 3 = 0}

theorem subset_implies_a_values (h : N a ⊆ M) : a = -3/8 ∨ a = 0 ∨ a = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_values_l2753_275337


namespace NUMINAMATH_CALUDE_complex_power_one_minus_i_six_l2753_275343

theorem complex_power_one_minus_i_six :
  (1 - Complex.I) ^ 6 = 8 * Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_power_one_minus_i_six_l2753_275343


namespace NUMINAMATH_CALUDE_max_min_difference_is_16_l2753_275352

def f (x : ℝ) := |x - 1| + |x - 2| + |x - 3|

theorem max_min_difference_is_16 :
  ∃ (x_max x_min : ℝ), x_max ∈ Set.Icc (-4 : ℝ) 4 ∧ x_min ∈ Set.Icc (-4 : ℝ) 4 ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) 4, f x ≤ f x_max) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) 4, f x_min ≤ f x) ∧
  f x_max - f x_min = 16 :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_is_16_l2753_275352


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l2753_275389

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₂ = -x₁ ∧ y₂ = -y₁

/-- Given that point A(1,a) and point B(b,-2) are symmetric with respect to the origin O, prove that a+b = 1 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_wrt_origin 1 a b (-2)) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l2753_275389


namespace NUMINAMATH_CALUDE_power_sum_fourth_l2753_275324

theorem power_sum_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_fourth_l2753_275324


namespace NUMINAMATH_CALUDE_P_in_fourth_quadrant_l2753_275361

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point P -/
def P : Point :=
  { x := 2023, y := -2024 }

/-- Theorem stating that P is in the fourth quadrant -/
theorem P_in_fourth_quadrant : fourth_quadrant P := by
  sorry

end NUMINAMATH_CALUDE_P_in_fourth_quadrant_l2753_275361


namespace NUMINAMATH_CALUDE_triangle_coverage_theorem_l2753_275376

-- Define the original equilateral triangle
structure EquilateralTriangle where
  area : ℝ
  isEquilateral : Bool

-- Define a point inside the triangle
structure Point where
  x : ℝ
  y : ℝ
  insideTriangle : Bool

-- Define the smaller equilateral triangles
structure SmallerTriangle where
  area : ℝ
  sidesParallel : Bool
  containsPoint : Point → Bool

-- Main theorem
theorem triangle_coverage_theorem 
  (original : EquilateralTriangle)
  (points : Finset Point)
  (h1 : original.area = 1)
  (h2 : original.isEquilateral = true)
  (h3 : points.card = 5)
  (h4 : ∀ p ∈ points, p.insideTriangle = true) :
  ∃ (t1 t2 t3 : SmallerTriangle),
    (t1.sidesParallel = true ∧ t2.sidesParallel = true ∧ t3.sidesParallel = true) ∧
    (t1.area + t2.area + t3.area ≤ 0.64) ∧
    (∀ p ∈ points, t1.containsPoint p = true ∨ t2.containsPoint p = true ∨ t3.containsPoint p = true) :=
by sorry

end NUMINAMATH_CALUDE_triangle_coverage_theorem_l2753_275376


namespace NUMINAMATH_CALUDE_sale_price_calculation_l2753_275386

theorem sale_price_calculation (original_price : ℝ) :
  let increased_price := original_price * 1.3
  let sale_price := increased_price * 0.9
  sale_price = original_price * 1.17 :=
by sorry

end NUMINAMATH_CALUDE_sale_price_calculation_l2753_275386


namespace NUMINAMATH_CALUDE_number_with_specific_remainders_l2753_275342

theorem number_with_specific_remainders : ∃! (N : ℕ), N < 221 ∧ N % 13 = 11 ∧ N % 17 = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_with_specific_remainders_l2753_275342


namespace NUMINAMATH_CALUDE_dolly_additional_tickets_l2753_275327

/-- The number of additional tickets Dolly needs to buy for amusement park rides -/
theorem dolly_additional_tickets : ℕ := by
  -- Define the number of rides Dolly wants for each attraction
  let ferris_wheel_rides : ℕ := 2
  let roller_coaster_rides : ℕ := 3
  let log_ride_rides : ℕ := 7

  -- Define the cost in tickets for each attraction
  let ferris_wheel_cost : ℕ := 2
  let roller_coaster_cost : ℕ := 5
  let log_ride_cost : ℕ := 1

  -- Define the number of tickets Dolly currently has
  let current_tickets : ℕ := 20

  -- Calculate the total number of tickets needed
  let total_tickets_needed : ℕ := 
    ferris_wheel_rides * ferris_wheel_cost +
    roller_coaster_rides * roller_coaster_cost +
    log_ride_rides * log_ride_cost

  -- Calculate the additional tickets needed
  let additional_tickets : ℕ := total_tickets_needed - current_tickets

  -- Prove that the additional tickets needed is 6
  have h : additional_tickets = 6 := by sorry

  exact 6

end NUMINAMATH_CALUDE_dolly_additional_tickets_l2753_275327


namespace NUMINAMATH_CALUDE_triangle_side_length_l2753_275362

/-- Represents a triangle with side lengths x, y, z and angles X, Y, Z --/
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ
  X : ℝ
  Y : ℝ
  Z : ℝ

/-- The theorem stating the properties of the specific triangle and its side length y --/
theorem triangle_side_length (t : Triangle) 
  (h1 : t.Z = 4 * t.X) 
  (h2 : t.x = 36) 
  (h3 : t.z = 72) : 
  t.y = 72 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2753_275362


namespace NUMINAMATH_CALUDE_ice_water_volume_change_l2753_275372

theorem ice_water_volume_change (v : ℝ) (h : v > 0) :
  let ice_volume := v * (1 + 1/11)
  (ice_volume - v) / ice_volume = 1/12 := by
sorry

end NUMINAMATH_CALUDE_ice_water_volume_change_l2753_275372


namespace NUMINAMATH_CALUDE_min_value_of_f_l2753_275336

def f (x : ℝ) : ℝ := x^4 + 2*x^2 - 1

theorem min_value_of_f :
  ∃ (m : ℝ), m = -1 ∧ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2753_275336


namespace NUMINAMATH_CALUDE_smallest_n_for_2012_terms_l2753_275346

theorem smallest_n_for_2012_terms (n : ℕ) : (∀ m : ℕ, (m + 1)^2 ≥ 2012 → n ≤ m) ↔ n = 44 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_2012_terms_l2753_275346


namespace NUMINAMATH_CALUDE_characters_with_initial_D_l2753_275364

-- Define the total number of characters
def total_characters : ℕ := 60

-- Define the number of characters with initial A
def characters_A : ℕ := total_characters / 2

-- Define the number of characters with initial C
def characters_C : ℕ := characters_A / 2

-- Define the remaining characters (D and E)
def remaining_characters : ℕ := total_characters - characters_A - characters_C

-- Theorem stating the number of characters with initial D
theorem characters_with_initial_D : 
  ∃ (d e : ℕ), d = 2 * e ∧ d + e = remaining_characters ∧ d = 10 :=
sorry

end NUMINAMATH_CALUDE_characters_with_initial_D_l2753_275364


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2753_275316

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 * y * (4*x + 3*y) = 3) : 
  ∃ (min : ℝ), min = 2 * Real.sqrt 3 ∧ ∀ (x' y' : ℝ), x' > 0 → y' > 0 → 
  x'^2 * y' * (4*x' + 3*y') = 3 → 2*x' + 3*y' ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2753_275316


namespace NUMINAMATH_CALUDE_stratified_sampling_l2753_275385

theorem stratified_sampling (total_students : ℕ) (sample_size : ℕ) (first_grade : ℕ) (second_grade : ℕ) :
  total_students = 2000 →
  sample_size = 100 →
  first_grade = 30 →
  second_grade = 30 →
  sample_size = first_grade + second_grade + (sample_size - first_grade - second_grade) →
  (sample_size - first_grade - second_grade) = 40 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_l2753_275385


namespace NUMINAMATH_CALUDE_interval_constraint_l2753_275369

theorem interval_constraint (x : ℝ) : (1 < 2*x ∧ 2*x < 2) ∧ (1 < 3*x ∧ 3*x < 2) ↔ 1/2 < x ∧ x < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_interval_constraint_l2753_275369


namespace NUMINAMATH_CALUDE_expected_balls_theorem_l2753_275328

/-- Represents a system of balls arranged in a circle -/
structure BallSystem :=
  (n : ℕ)  -- number of balls

/-- Represents a swap operation on the ball system -/
structure SwapOperation :=
  (isAdjacent : Bool)  -- whether the swap is between adjacent balls only

/-- Calculates the probability of a ball remaining in its original position after a swap -/
def probabilityAfterSwap (sys : BallSystem) (op : SwapOperation) : ℚ :=
  if op.isAdjacent then
    (sys.n - 2 : ℚ) / sys.n * 2 / 3 + 2 / sys.n
  else
    (sys.n - 2 : ℚ) / sys.n

/-- Calculates the expected number of balls in their original positions after two swaps -/
def expectedBallsInOriginalPosition (sys : BallSystem) (op1 op2 : SwapOperation) : ℚ :=
  sys.n * probabilityAfterSwap sys op1 * probabilityAfterSwap sys op2

theorem expected_balls_theorem (sys : BallSystem) (op1 op2 : SwapOperation) :
  sys.n = 8 ∧ ¬op1.isAdjacent ∧ op2.isAdjacent →
  expectedBallsInOriginalPosition sys op1 op2 = 2 := by
  sorry

#eval expectedBallsInOriginalPosition ⟨8⟩ ⟨false⟩ ⟨true⟩

end NUMINAMATH_CALUDE_expected_balls_theorem_l2753_275328


namespace NUMINAMATH_CALUDE_unique_solution_iff_a_in_open_interval_l2753_275359

-- Define the system of equations
def system (a x y z : ℝ) : Prop :=
  x + y + z = 0 ∧ x*y + y*z + a*z*x = 0

-- Define the condition for exactly one solution
def has_unique_solution (a : ℝ) : Prop :=
  ∃! (x y z : ℝ), system a x y z

-- Theorem statement
theorem unique_solution_iff_a_in_open_interval :
  ∀ a : ℝ, has_unique_solution a ↔ 0 < a ∧ a < 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_iff_a_in_open_interval_l2753_275359


namespace NUMINAMATH_CALUDE_rectangles_in_4x4_grid_l2753_275323

/-- The number of rows in the grid -/
def n : ℕ := 4

/-- The number of columns in the grid -/
def m : ℕ := 4

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The total number of rectangles in an n × m grid -/
def num_rectangles (n m : ℕ) : ℕ := choose_two n * choose_two m

theorem rectangles_in_4x4_grid : 
  num_rectangles n m = 36 :=
sorry

end NUMINAMATH_CALUDE_rectangles_in_4x4_grid_l2753_275323


namespace NUMINAMATH_CALUDE_baby_age_at_weight_7200_l2753_275380

/-- The relationship between a baby's weight and age -/
def weight_age_relation (a : ℝ) (x : ℝ) : ℝ := a + 800 * x

/-- The theorem stating the age of the baby when their weight is 7200 grams -/
theorem baby_age_at_weight_7200 (a : ℝ) (x : ℝ) 
  (h1 : a = 3200) -- The baby's weight at birth is 3200 grams
  (h2 : weight_age_relation a x = 7200) -- The baby's weight is 7200 grams
  : x = 5 := by
  sorry

#check baby_age_at_weight_7200

end NUMINAMATH_CALUDE_baby_age_at_weight_7200_l2753_275380


namespace NUMINAMATH_CALUDE_max_value_quadratic_max_value_sum_products_l2753_275303

-- Part 1
theorem max_value_quadratic (a : ℝ) (h : a > 0) :
  ∃ (max : ℝ), ∀ (x : ℝ), x > 0 → a > 2*x → x*(a - 2*x) ≤ max ∧
  ∃ (x_max : ℝ), x_max > 0 ∧ a > 2*x_max ∧ x_max*(a - 2*x_max) = max :=
sorry

-- Part 2
theorem max_value_sum_products (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum_squares : a^2 + b^2 + c^2 = 4) :
  a*b + b*c + a*c ≤ 4 ∧ ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
  a'^2 + b'^2 + c'^2 = 4 ∧ a'*b' + b'*c' + a'*c' = 4 :=
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_max_value_sum_products_l2753_275303


namespace NUMINAMATH_CALUDE_min_value_of_f_l2753_275311

-- Define the function f
def f (x : ℝ) : ℝ := (x - 2024)^2

-- State the theorem
theorem min_value_of_f :
  (∀ x : ℝ, f (x + 2023) = x^2 - 2*x + 1) →
  (∃ x₀ : ℝ, f x₀ = 0 ∧ ∀ x : ℝ, f x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2753_275311


namespace NUMINAMATH_CALUDE_length_of_BC_l2753_275379

-- Define the parabola
def parabola (x : ℝ) : ℝ := 4 * x^2

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.A = (2, 16) ∧
  parabola 2 = 16 ∧
  t.B.1 = -t.C.1 ∧
  t.B.2 = t.C.2 ∧
  (1/2 : ℝ) * |t.B.1 - t.C.1| * |t.A.2 - t.B.2| = 128

-- Theorem statement
theorem length_of_BC (t : Triangle) (h : satisfies_conditions t) : 
  |t.B.1 - t.C.1| = 8 := by sorry

end NUMINAMATH_CALUDE_length_of_BC_l2753_275379


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2753_275304

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 3, x^2 + (a - 4)*x + 4 > 0) ↔ a > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2753_275304


namespace NUMINAMATH_CALUDE_inequality_proof_l2753_275308

theorem inequality_proof (a b c : ℝ) (ha : a ≥ c) (hb : b ≥ c) (hc : c > 0) :
  Real.sqrt (c * (a - c)) + Real.sqrt (c * (b - c)) ≤ Real.sqrt (a * b) ∧
  (Real.sqrt (c * (a - c)) + Real.sqrt (c * (b - c)) = Real.sqrt (a * b) ↔ a * b = c * (a + b)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2753_275308


namespace NUMINAMATH_CALUDE_gcd_14m_21n_l2753_275353

theorem gcd_14m_21n (m n : ℕ+) (h : Nat.gcd m n = 18) : Nat.gcd (14 * m) (21 * n) = 126 := by
  sorry

end NUMINAMATH_CALUDE_gcd_14m_21n_l2753_275353


namespace NUMINAMATH_CALUDE_books_count_l2753_275334

/-- The number of books Jason has -/
def jason_books : ℕ := 18

/-- The number of books Mary has -/
def mary_books : ℕ := 42

/-- The total number of books Jason and Mary have together -/
def total_books : ℕ := jason_books + mary_books

theorem books_count : total_books = 60 := by
  sorry

end NUMINAMATH_CALUDE_books_count_l2753_275334


namespace NUMINAMATH_CALUDE_vector_magnitude_direction_comparison_l2753_275375

theorem vector_magnitude_direction_comparison
  (a b : ℝ × ℝ)
  (h1 : a ≠ (0, 0))
  (h2 : b ≠ (0, 0))
  (h3 : ∃ (k : ℝ), k > 0 ∧ a = k • b)
  (h4 : ‖a‖ > ‖b‖) :
  ¬ (∀ (x y : ℝ × ℝ), (∃ (k : ℝ), k > 0 ∧ x = k • y) → x > y) :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_direction_comparison_l2753_275375


namespace NUMINAMATH_CALUDE_polygon_area_l2753_275370

-- Define the polygon
structure Polygon :=
  (sides : ℕ)
  (perimeter : ℝ)
  (num_squares : ℕ)
  (congruent_sides : Bool)
  (perpendicular_sides : Bool)

-- Define the properties of our specific polygon
def special_polygon : Polygon :=
  { sides := 28,
    perimeter := 56,
    num_squares := 25,
    congruent_sides := true,
    perpendicular_sides := true }

-- Theorem statement
theorem polygon_area (p : Polygon) (h1 : p = special_polygon) : 
  (p.perimeter / p.sides)^2 * p.num_squares = 100 := by
  sorry

end NUMINAMATH_CALUDE_polygon_area_l2753_275370


namespace NUMINAMATH_CALUDE_vector_sum_simplification_l2753_275339

variable {V : Type*} [AddCommGroup V]
variable (A B C D : V)

theorem vector_sum_simplification :
  (B - A) + (A - C) + (D - B) = D - C :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_simplification_l2753_275339


namespace NUMINAMATH_CALUDE_cubic_foot_to_cubic_inches_l2753_275335

theorem cubic_foot_to_cubic_inches :
  ∀ (foot inch : ℝ), 
    foot > 0 →
    inch > 0 →
    foot = 12 * inch →
    foot^3 = 1728 * inch^3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_foot_to_cubic_inches_l2753_275335


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2753_275309

theorem quadratic_equation_solution :
  ∀ x : ℝ, x^2 = 5*x ↔ x = 0 ∨ x = 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2753_275309


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l2753_275373

/-- Evaluates |3-7i| + |3+7i| - arg(3+7i) -/
theorem complex_expression_evaluation :
  let z₁ : ℂ := 3 - 7*I
  let z₂ : ℂ := 3 + 7*I
  Complex.abs z₁ + Complex.abs z₂ - Complex.arg z₂ = 2 * Real.sqrt 58 - Real.arctan (7/3) := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l2753_275373
