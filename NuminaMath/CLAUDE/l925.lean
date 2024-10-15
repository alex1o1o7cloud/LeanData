import Mathlib

namespace NUMINAMATH_CALUDE_first_term_value_l925_92522

/-- Given a sequence {aₙ} with sum Sₙ, prove that a₁ = 1/2 -/
theorem first_term_value (a : ℕ → ℚ) (S : ℕ → ℚ) : 
  (∀ n, S n = (a 1 * (4^n - 1)) / 3) →   -- Condition 1
  a 4 = 32 →                             -- Condition 2
  a 1 = 1/2 :=                           -- Conclusion
by sorry

end NUMINAMATH_CALUDE_first_term_value_l925_92522


namespace NUMINAMATH_CALUDE_adrian_water_needed_l925_92566

/-- Represents the recipe ratios and amount of orange juice used --/
structure Recipe where
  water_sugar_ratio : ℚ
  sugar_juice_ratio : ℚ
  orange_juice_cups : ℚ

/-- Calculates the amount of water needed for the punch recipe --/
def water_needed (r : Recipe) : ℚ :=
  r.water_sugar_ratio * r.sugar_juice_ratio * r.orange_juice_cups

/-- Theorem stating that Adrian needs 60 cups of water --/
theorem adrian_water_needed :
  let recipe := Recipe.mk 5 3 4
  water_needed recipe = 60 := by
  sorry

end NUMINAMATH_CALUDE_adrian_water_needed_l925_92566


namespace NUMINAMATH_CALUDE_tangent_line_at_negative_one_l925_92539

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2

-- Define the point of tangency
def x₀ : ℝ := -1

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := 3*x - y + 3 = 0

-- Theorem statement
theorem tangent_line_at_negative_one :
  tangent_line x₀ (f x₀) ∧
  ∀ x : ℝ, tangent_line x (f x₀ + f' x₀ * (x - x₀)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_negative_one_l925_92539


namespace NUMINAMATH_CALUDE_map_distance_calculation_l925_92511

/-- Calculates the distance on a map given travel time, speed, and map scale -/
theorem map_distance_calculation (travel_time : ℝ) (average_speed : ℝ) (map_scale : ℝ) :
  travel_time = 6.5 →
  average_speed = 60 →
  map_scale = 0.01282051282051282 →
  travel_time * average_speed * map_scale = 5 := by
  sorry

end NUMINAMATH_CALUDE_map_distance_calculation_l925_92511


namespace NUMINAMATH_CALUDE_expected_value_of_coin_flips_l925_92530

def penny : ℚ := 1
def nickel : ℚ := 5
def dime : ℚ := 10
def quarter : ℚ := 25
def half_dollar : ℚ := 50
def dollar : ℚ := 100

def coin_flip_probability : ℚ := 1/2

theorem expected_value_of_coin_flips :
  coin_flip_probability * (penny + nickel + dime + quarter + half_dollar + dollar) = 95.5 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_of_coin_flips_l925_92530


namespace NUMINAMATH_CALUDE_circle_radius_from_parabola_tangency_l925_92571

/-- The radius of a circle given specific tangency conditions of a parabola -/
theorem circle_radius_from_parabola_tangency : ∃ (r : ℝ), 
  (∀ x y : ℝ, y = x^2 + r → y ≤ x) ∧ 
  (∃ x : ℝ, x^2 + r = x) ∧
  r = (1 : ℝ) / 4 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_from_parabola_tangency_l925_92571


namespace NUMINAMATH_CALUDE_field_length_calculation_l925_92533

/-- Given a rectangular field wrapped with tape, calculate its length. -/
theorem field_length_calculation (total_tape : ℕ) (width : ℕ) (leftover_tape : ℕ) :
  total_tape = 250 →
  width = 20 →
  leftover_tape = 90 →
  2 * (width + (total_tape - leftover_tape) / 2) = total_tape - leftover_tape →
  (total_tape - leftover_tape) / 2 - width = 60 := by
  sorry

end NUMINAMATH_CALUDE_field_length_calculation_l925_92533


namespace NUMINAMATH_CALUDE_range_of_k_for_two_roots_l925_92502

open Real

theorem range_of_k_for_two_roots (g : ℝ → ℝ) (k : ℝ) :
  (∀ x, g x = 2 * sin (2 * x - π / 6)) →
  (∀ x ∈ Set.Icc 0 (π / 2), (g x - k = 0 → ∃ y ∈ Set.Icc 0 (π / 2), x ≠ y ∧ g y - k = 0)) ↔
  k ∈ Set.Icc 1 2 ∧ k ≠ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_k_for_two_roots_l925_92502


namespace NUMINAMATH_CALUDE_jacks_total_money_l925_92507

/-- Calculates the total amount of money in dollars given an amount in dollars and euros, with a fixed exchange rate. -/
def total_money_in_dollars (dollars : ℕ) (euros : ℕ) (exchange_rate : ℕ) : ℕ :=
  dollars + euros * exchange_rate

/-- Theorem stating that Jack's total money in dollars is 117 given the problem conditions. -/
theorem jacks_total_money :
  total_money_in_dollars 45 36 2 = 117 := by
  sorry

end NUMINAMATH_CALUDE_jacks_total_money_l925_92507


namespace NUMINAMATH_CALUDE_equation_solution_l925_92597

theorem equation_solution :
  ∃! x : ℚ, 2 * x + 3 = 500 - (4 * x + 5 * x) + 7 ∧ x = 504 / 11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l925_92597


namespace NUMINAMATH_CALUDE_trishas_walk_l925_92567

/-- Proves that given a total distance and two equal segments, the remaining distance is as expected. -/
theorem trishas_walk (total : ℝ) (segment : ℝ) (h1 : total = 0.8888888888888888) 
  (h2 : segment = 0.1111111111111111) : 
  total - 2 * segment = 0.6666666666666666 := by
  sorry

end NUMINAMATH_CALUDE_trishas_walk_l925_92567


namespace NUMINAMATH_CALUDE_sequence_equality_l925_92528

def x : ℕ → ℚ
  | 0 => 1
  | n + 1 => x n / (2 + x n)

def y : ℕ → ℚ
  | 0 => 1
  | n + 1 => y n ^ 2 / (1 + 2 * y n)

theorem sequence_equality (n : ℕ) : y n = x (2^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_equality_l925_92528


namespace NUMINAMATH_CALUDE_percent_increase_revenue_l925_92529

/-- Given two positive real numbers M and N representing revenues in millions for two consecutive years,
    this theorem states that the percent increase in revenue relative to the sum of the revenues of both years
    is equal to 100 * (M - N) / (M + N). -/
theorem percent_increase_revenue (M N : ℝ) (hM : M > 0) (hN : N > 0) :
  (M - N) / (M + N) * 100 = 100 * (M - N) / (M + N) := by sorry

end NUMINAMATH_CALUDE_percent_increase_revenue_l925_92529


namespace NUMINAMATH_CALUDE_count_integers_with_repeated_digits_is_140_l925_92583

/-- A function that counts the number of positive three-digit integers 
    between 500 and 999 with at least two identical digits -/
def count_integers_with_repeated_digits : ℕ :=
  let range_start := 500
  let range_end := 999
  let digits := 3
  -- Count of integers where last two digits are the same
  let case1 := 10 * (range_end.div 100 - range_start.div 100 + 1)
  -- Count of integers where first two digits are the same (and different from third)
  let case2 := (range_end.div 100 - range_start.div 100 + 1) * (digits - 1)
  -- Count of integers where first and third digits are the same (and different from second)
  let case3 := (range_end.div 100 - range_start.div 100 + 1) * (digits - 1)
  case1 + case2 + case3

/-- Theorem stating that the count of integers with repeated digits is 140 -/
theorem count_integers_with_repeated_digits_is_140 :
  count_integers_with_repeated_digits = 140 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_with_repeated_digits_is_140_l925_92583


namespace NUMINAMATH_CALUDE_factorization_of_x2y_minus_4y_l925_92532

theorem factorization_of_x2y_minus_4y (x y : ℝ) : x^2 * y - 4 * y = y * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x2y_minus_4y_l925_92532


namespace NUMINAMATH_CALUDE_chess_tournament_games_l925_92547

/-- Calculate the number of games in a chess tournament -/
def tournament_games (n : ℕ) : ℕ :=
  n * (n - 1)

/-- The number of players in the tournament -/
def num_players : ℕ := 10

/-- Theorem: In a chess tournament with 10 players, where each player plays twice 
    with every other player, the total number of games played is 180. -/
theorem chess_tournament_games : 
  2 * tournament_games num_players = 180 := by
sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l925_92547


namespace NUMINAMATH_CALUDE_smallest_special_number_l925_92552

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def has_no_prime_factor_below (n k : ℕ) : Prop := ∀ p : ℕ, p < k → is_prime p → n % p ≠ 0

theorem smallest_special_number : 
  ∀ n : ℕ, n > 0 → n < 4091 → 
  (¬ is_prime n ∧ ¬ is_perfect_square n ∧ has_no_prime_factor_below n 60) → False :=
sorry

#check smallest_special_number

end NUMINAMATH_CALUDE_smallest_special_number_l925_92552


namespace NUMINAMATH_CALUDE_quadratic_other_x_intercept_l925_92577

/-- Given a quadratic function f(x) with vertex (5,10) and one x-intercept at (1,0),
    the x-coordinate of the other x-intercept is 9. -/
theorem quadratic_other_x_intercept 
  (f : ℝ → ℝ) 
  (h1 : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) 
  (h2 : f 1 = 0) 
  (h3 : ∃ y, f 5 = y ∧ y = 10) :
  ∃ x, f x = 0 ∧ x = 9 :=
sorry

end NUMINAMATH_CALUDE_quadratic_other_x_intercept_l925_92577


namespace NUMINAMATH_CALUDE_walnut_weight_in_mixture_l925_92595

/-- Given a mixture of nuts with a specific ratio and total weight, 
    calculate the weight of walnuts -/
theorem walnut_weight_in_mixture 
  (ratio_almonds : ℕ) 
  (ratio_walnuts : ℕ) 
  (ratio_peanuts : ℕ) 
  (ratio_cashews : ℕ) 
  (total_weight : ℕ) 
  (h1 : ratio_almonds = 5) 
  (h2 : ratio_walnuts = 3) 
  (h3 : ratio_peanuts = 2) 
  (h4 : ratio_cashews = 4) 
  (h5 : total_weight = 420) : 
  (ratio_walnuts * total_weight) / (ratio_almonds + ratio_walnuts + ratio_peanuts + ratio_cashews) = 90 := by
  sorry


end NUMINAMATH_CALUDE_walnut_weight_in_mixture_l925_92595


namespace NUMINAMATH_CALUDE_smallest_nonprime_with_large_factors_l925_92587

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < k → ¬(n % p = 0)

def is_nonprime (n : ℕ) : Prop := n > 1 ∧ ¬(is_prime n)

theorem smallest_nonprime_with_large_factors :
  ∃ n : ℕ, is_nonprime n ∧
            has_no_prime_factors_less_than n 20 ∧
            (∀ m : ℕ, m < n → ¬(is_nonprime m ∧ has_no_prime_factors_less_than m 20)) ∧
            500 < n ∧ n ≤ 550 :=
sorry

end NUMINAMATH_CALUDE_smallest_nonprime_with_large_factors_l925_92587


namespace NUMINAMATH_CALUDE_painted_cube_probability_l925_92570

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : ℕ
  paintedFaces : ℕ

/-- The number of unit cubes in a larger cube -/
def totalUnitCubes (size : ℕ) : ℕ := size ^ 3

/-- The number of ways to choose 2 cubes from a set -/
def waysToChooseTwoCubes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of unit cubes with exactly three painted faces -/
def cubesWithThreePaintedFaces (size : ℕ) : ℕ := 8

/-- The number of unit cubes with no painted faces -/
def cubesWithNoPaintedFaces (size : ℕ) : ℕ := (size - 2) ^ 3

/-- The probability of selecting one cube with three painted faces and one with no painted faces -/
def probabilityOfSelection (size : ℕ) : ℚ :=
  let total := totalUnitCubes size
  let ways := waysToChooseTwoCubes total
  let threePainted := cubesWithThreePaintedFaces size
  let noPainted := cubesWithNoPaintedFaces size
  (threePainted * noPainted : ℚ) / ways

theorem painted_cube_probability :
  probabilityOfSelection 5 = 72 / 2583 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_probability_l925_92570


namespace NUMINAMATH_CALUDE_repairs_count_l925_92584

/-- Represents the mechanic shop scenario --/
structure MechanicShop where
  oil_change_price : ℕ
  repair_price : ℕ
  car_wash_price : ℕ
  oil_changes : ℕ
  car_washes : ℕ
  total_earnings : ℕ

/-- Calculates the number of repairs given the shop's data --/
def calculate_repairs (shop : MechanicShop) : ℕ :=
  (shop.total_earnings - (shop.oil_change_price * shop.oil_changes + shop.car_wash_price * shop.car_washes)) / shop.repair_price

/-- Theorem stating that given the specific conditions, the number of repairs is 10 --/
theorem repairs_count (shop : MechanicShop) 
  (h1 : shop.oil_change_price = 20)
  (h2 : shop.repair_price = 30)
  (h3 : shop.car_wash_price = 5)
  (h4 : shop.oil_changes = 5)
  (h5 : shop.car_washes = 15)
  (h6 : shop.total_earnings = 475) :
  calculate_repairs shop = 10 := by
  sorry

#eval calculate_repairs { 
  oil_change_price := 20, 
  repair_price := 30, 
  car_wash_price := 5, 
  oil_changes := 5, 
  car_washes := 15, 
  total_earnings := 475 
}

end NUMINAMATH_CALUDE_repairs_count_l925_92584


namespace NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l925_92557

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A parallelogram defined by four vertices -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Check if four points form a valid parallelogram -/
def isValidParallelogram (p : Parallelogram) : Prop :=
  (p.v1.x + p.v3.x = p.v2.x + p.v4.x) ∧ 
  (p.v1.y + p.v3.y = p.v2.y + p.v4.y)

theorem parallelogram_fourth_vertex 
  (p : Parallelogram)
  (h1 : p.v1 = Point.mk (-1) 0)
  (h2 : p.v2 = Point.mk 3 0)
  (h3 : p.v3 = Point.mk 1 (-5)) :
  isValidParallelogram p →
  (p.v4 = Point.mk 5 (-5) ∨ p.v4 = Point.mk (-3) (-5) ∨ p.v4 = Point.mk 1 5) :=
by sorry


end NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l925_92557


namespace NUMINAMATH_CALUDE_correct_average_l925_92527

theorem correct_average (n : ℕ) (initial_avg : ℚ) (incorrect_readings : List (ℚ × ℚ)) : 
  n = 20 ∧ 
  initial_avg = 15 ∧ 
  incorrect_readings = [(42, 52), (68, 78), (85, 95)] →
  (n : ℚ) * initial_avg + (incorrect_readings.map (λ p => p.2 - p.1)).sum = n * (16.5 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l925_92527


namespace NUMINAMATH_CALUDE_gcd_lcm_equation_solutions_l925_92512

theorem gcd_lcm_equation_solutions :
  let S : Set (ℕ × ℕ) := {(8, 513), (513, 8), (215, 2838), (2838, 215),
                          (258, 1505), (1505, 258), (235, 2961), (2961, 235)}
  ∀ α β : ℕ, (Nat.gcd α β + Nat.lcm α β = 4 * (α + β) + 2021) ↔ (α, β) ∈ S :=
by sorry

end NUMINAMATH_CALUDE_gcd_lcm_equation_solutions_l925_92512


namespace NUMINAMATH_CALUDE_product_of_roots_l925_92575

theorem product_of_roots (x : ℂ) : 
  (2 * x^3 - 3 * x^2 - 8 * x + 10 = 0) → 
  (∃ r₁ r₂ r₃ : ℂ, (x - r₁) * (x - r₂) * (x - r₃) = 2 * x^3 - 3 * x^2 - 8 * x + 10 ∧ r₁ * r₂ * r₃ = -5) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l925_92575


namespace NUMINAMATH_CALUDE_function_difference_inequality_l925_92516

theorem function_difference_inequality
  (f g : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (hg : Differentiable ℝ g)
  (h_deriv : ∀ x, deriv f x > deriv g x)
  {a b : ℝ}
  (hab : a > b) :
  f a - f b > g a - g b :=
sorry

end NUMINAMATH_CALUDE_function_difference_inequality_l925_92516


namespace NUMINAMATH_CALUDE_senior_employee_bonus_l925_92519

/-- Proves that the senior employee receives $3,100 given the conditions of the bonus distribution -/
theorem senior_employee_bonus (total_bonus : ℕ) (difference : ℕ) (senior_share : ℕ) : 
  total_bonus = 5000 →
  difference = 1200 →
  senior_share = total_bonus - difference →
  2 * senior_share = total_bonus + difference →
  senior_share = 3100 := by
sorry

end NUMINAMATH_CALUDE_senior_employee_bonus_l925_92519


namespace NUMINAMATH_CALUDE_sales_price_calculation_l925_92596

theorem sales_price_calculation (C S : ℝ) : 
  (1.20 * C = 24) →  -- Gross profit is $24
  (S = C + 1.20 * C) →  -- Sales price is cost plus gross profit
  (S = 44) :=  -- Prove that sales price is $44
by
  sorry

end NUMINAMATH_CALUDE_sales_price_calculation_l925_92596


namespace NUMINAMATH_CALUDE_competitive_exam_candidates_l925_92580

theorem competitive_exam_candidates (candidates : ℕ) : 
  (candidates * 8 / 100 : ℚ) + 220 = (candidates * 12 / 100 : ℚ) →
  candidates = 5500 := by
sorry

end NUMINAMATH_CALUDE_competitive_exam_candidates_l925_92580


namespace NUMINAMATH_CALUDE_inequality_proof_l925_92548

def M : Set ℝ := {x | |x + 1| + |x - 1| ≤ 2}

theorem inequality_proof (x y z : ℝ) (hx : x ∈ M) (hy : |y| ≤ 1/6) (hz : |z| ≤ 1/9) :
  |x + 2*y - 3*z| ≤ 5/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l925_92548


namespace NUMINAMATH_CALUDE_y1_greater_y2_l925_92500

/-- A line in the 2D plane represented by y = mx + b -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

theorem y1_greater_y2 (l : Line) (p1 p2 : Point) :
  l.m = -1 →
  l.b = 1 →
  p1.x = -2 →
  p2.x = 3 →
  p1.liesOn l →
  p2.liesOn l →
  p1.y > p2.y := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_y2_l925_92500


namespace NUMINAMATH_CALUDE_integral_f_zero_to_one_l925_92542

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x + 2

-- State the theorem
theorem integral_f_zero_to_one :
  ∫ x in (0:ℝ)..(1:ℝ), f x = 11/6 := by sorry

end NUMINAMATH_CALUDE_integral_f_zero_to_one_l925_92542


namespace NUMINAMATH_CALUDE_equation_solution_range_l925_92586

theorem equation_solution_range (k : ℝ) :
  (∃ x : ℝ, (4 * (2015^x) - 2015^(-x)) / (2015^x - 3 * (2015^(-x))) = k) ↔ 
  (k < 1/3 ∨ k > 4) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_range_l925_92586


namespace NUMINAMATH_CALUDE_cube_paint_puzzle_l925_92558

theorem cube_paint_puzzle (n : ℕ) : 
  n > 0 → 
  (4 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 → 
  n = 2 :=
by sorry

end NUMINAMATH_CALUDE_cube_paint_puzzle_l925_92558


namespace NUMINAMATH_CALUDE_tetrahedron_inequality_l925_92568

theorem tetrahedron_inequality (a b c d h_a h_b h_c h_d V : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (hV : V > 0) 
  (h1 : V = (1/3) * a * h_a) 
  (h2 : V = (1/3) * b * h_b) 
  (h3 : V = (1/3) * c * h_c) 
  (h4 : V = (1/3) * d * h_d) : 
  (a + b + c + d) * (h_a + h_b + h_c + h_d) ≥ 48 * V := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_inequality_l925_92568


namespace NUMINAMATH_CALUDE_fruit_seller_inventory_l925_92517

theorem fruit_seller_inventory (apples oranges bananas pears grapes : ℕ) : 
  (apples - apples / 2 + 20 = 370) →
  (oranges - oranges * 35 / 100 = 195) →
  (bananas - bananas * 3 / 5 + 15 = 95) →
  (pears - pears * 45 / 100 = 50) →
  (grapes - grapes * 3 / 10 = 140) →
  (apples = 700 ∧ oranges = 300 ∧ bananas = 200 ∧ pears = 91 ∧ grapes = 200) :=
by sorry

end NUMINAMATH_CALUDE_fruit_seller_inventory_l925_92517


namespace NUMINAMATH_CALUDE_inequality_region_l925_92541

theorem inequality_region (x y : ℝ) : 
  Real.sqrt (x * y) ≥ x - 2 * y ↔ 
  ((x ≥ 0 ∧ y ≥ 0 ∧ y ≥ x / 2) ∨ 
   (x ≤ 0 ∧ y ≤ 0 ∧ y ≥ x / 2) ∨ 
   (x = 0 ∧ y ≥ 0) ∨ 
   (x ≥ 0 ∧ y = 0)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_region_l925_92541


namespace NUMINAMATH_CALUDE_kickball_difference_l925_92554

theorem kickball_difference (wednesday : ℕ) (total : ℕ) : 
  wednesday = 37 →
  total = 65 →
  wednesday - (total - wednesday) = 9 := by
sorry

end NUMINAMATH_CALUDE_kickball_difference_l925_92554


namespace NUMINAMATH_CALUDE_bus_total_capacity_l925_92590

/-- Represents the seating capacity of a bus with specified seat arrangements. -/
def bus_capacity (left_seats : ℕ) (right_seats_difference : ℕ) (people_per_seat : ℕ) (back_seat_capacity : ℕ) : ℕ :=
  let right_seats := left_seats - right_seats_difference
  let left_capacity := left_seats * people_per_seat
  let right_capacity := right_seats * people_per_seat
  left_capacity + right_capacity + back_seat_capacity

/-- Theorem stating the total seating capacity of the bus under given conditions. -/
theorem bus_total_capacity : bus_capacity 15 3 3 8 = 89 := by
  sorry

end NUMINAMATH_CALUDE_bus_total_capacity_l925_92590


namespace NUMINAMATH_CALUDE_unique_positive_solution_l925_92592

/-- The polynomial function f(x) = x^8 + 5x^7 + 10x^6 + 1728x^5 - 1380x^4 -/
def f (x : ℝ) : ℝ := x^8 + 5*x^7 + 10*x^6 + 1728*x^5 - 1380*x^4

/-- The statement that f(x) = 0 has exactly one positive real solution -/
theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ f x = 0 := by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l925_92592


namespace NUMINAMATH_CALUDE_sum_of_variables_l925_92565

theorem sum_of_variables (a b c d : ℚ) 
  (h1 : 2*a + 3 = 2*b + 5)
  (h2 : 2*b + 5 = 2*c + 7)
  (h3 : 2*c + 7 = 2*d + 9)
  (h4 : 2*d + 9 = 2*(a + b + c + d) + 13) :
  a + b + c + d = -14/3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_variables_l925_92565


namespace NUMINAMATH_CALUDE_rational_root_of_polynomial_l925_92536

def f (x : ℚ) : ℚ := 3 * x^3 - 7 * x^2 - 8 * x + 4

theorem rational_root_of_polynomial :
  ∀ x : ℚ, f x = 0 ↔ x = 1/3 := by sorry

end NUMINAMATH_CALUDE_rational_root_of_polynomial_l925_92536


namespace NUMINAMATH_CALUDE_expand_product_l925_92513

theorem expand_product (x : ℝ) : 4 * (x + 3) * (x + 6) = 4 * x^2 + 36 * x + 72 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l925_92513


namespace NUMINAMATH_CALUDE_cone_angle_bisecting_volume_l925_92524

/-- 
Given a cone with the following properties:
- A perpendicular is dropped from the center of the base to the slant height
- This perpendicular rotates about the cone's axis
- The surface of rotation divides the cone's volume in half

The angle between the slant height and the axis is arccos(1 / 2^(1/4))
-/
theorem cone_angle_bisecting_volume (R h : ℝ) (hR : R > 0) (hh : h > 0) : 
  let α := Real.arccos ((1 : ℝ) / 2^(1/4))
  let V := (1/3) * π * R^2 * h
  let V_rotated := (1/3) * π * (R * (Real.cos α)^2)^2 * h
  V_rotated = (1/2) * V :=
by sorry

end NUMINAMATH_CALUDE_cone_angle_bisecting_volume_l925_92524


namespace NUMINAMATH_CALUDE_max_matches_theorem_l925_92508

/-- The maximum number of matches that cannot form a triangle with any two sides differing by at least 10 matches -/
def max_matches : ℕ := 62

/-- A function that checks if three numbers can form a triangle -/
def is_triangle (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if any two sides differ by at least 10 -/
def sides_differ_by_10 (a b c : ℕ) : Prop :=
  (a ≥ b + 10 ∨ b ≥ a + 10) ∧ (b ≥ c + 10 ∨ c ≥ b + 10) ∧ (c ≥ a + 10 ∨ a ≥ c + 10)

theorem max_matches_theorem :
  ∀ n : ℕ, n > max_matches →
    ∃ a b c : ℕ, a + b + c = n ∧ is_triangle a b c ∧ sides_differ_by_10 a b c :=
sorry

end NUMINAMATH_CALUDE_max_matches_theorem_l925_92508


namespace NUMINAMATH_CALUDE_alex_ate_six_ounces_l925_92579

/-- The amount of jelly beans Alex ate -/
def jelly_beans_eaten (initial : ℕ) (num_piles : ℕ) (weight_per_pile : ℕ) : ℕ :=
  initial - (num_piles * weight_per_pile)

/-- Theorem stating that Alex ate 6 ounces of jelly beans -/
theorem alex_ate_six_ounces : 
  jelly_beans_eaten 36 3 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_alex_ate_six_ounces_l925_92579


namespace NUMINAMATH_CALUDE_first_three_digits_of_large_number_l925_92599

-- Define the expression
def large_number : ℝ := (10^100 + 1)^(5/3)

-- Define a function to extract the first three decimal digits
def first_three_decimal_digits (x : ℝ) : ℕ × ℕ × ℕ := sorry

-- State the theorem
theorem first_three_digits_of_large_number :
  first_three_decimal_digits large_number = (6, 6, 6) := by sorry

end NUMINAMATH_CALUDE_first_three_digits_of_large_number_l925_92599


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l925_92520

theorem sqrt_equation_solutions : 
  {x : ℝ | Real.sqrt (4 * x - 3) + 12 / Real.sqrt (4 * x - 3) = 8} = {39/4, 7/4} := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l925_92520


namespace NUMINAMATH_CALUDE_exp_7pi_i_div_3_rectangular_form_l925_92589

theorem exp_7pi_i_div_3_rectangular_form :
  Complex.exp (7 * Real.pi * Complex.I / 3) = (1 / 2 : ℂ) + Complex.I * (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_exp_7pi_i_div_3_rectangular_form_l925_92589


namespace NUMINAMATH_CALUDE_linked_rings_height_l925_92598

/-- Represents the properties of a sequence of linked rings -/
structure LinkedRings where
  thickness : ℝ
  topOutsideDiameter : ℝ
  diameterDecrease : ℝ
  bottomOutsideDiameter : ℝ

/-- Calculates the total height of the linked rings -/
def totalHeight (rings : LinkedRings) : ℝ :=
  sorry

/-- Theorem stating that the total height of the linked rings with given properties is 273 cm -/
theorem linked_rings_height :
  let rings : LinkedRings := {
    thickness := 2,
    topOutsideDiameter := 20,
    diameterDecrease := 0.5,
    bottomOutsideDiameter := 10
  }
  totalHeight rings = 273 := by sorry

end NUMINAMATH_CALUDE_linked_rings_height_l925_92598


namespace NUMINAMATH_CALUDE_worker_payment_l925_92588

theorem worker_payment (total_days : ℕ) (days_not_worked : ℕ) (return_amount : ℕ) 
  (h1 : total_days = 30)
  (h2 : days_not_worked = 24)
  (h3 : return_amount = 25)
  : ∃ x : ℕ, 
    (total_days - days_not_worked) * x = days_not_worked * return_amount ∧ 
    x = 100 := by
  sorry

end NUMINAMATH_CALUDE_worker_payment_l925_92588


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_l925_92510

theorem min_value_sqrt_sum (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : a * b + b * c + c * a = a + b + c) (h5 : 0 < a + b + c) :
  2 ≤ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) ∧
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧
    a' * b' + b' * c' + c' * a' = a' + b' + c' ∧ 0 < a' + b' + c' ∧
    Real.sqrt (a' * b') + Real.sqrt (b' * c') + Real.sqrt (c' * a') = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_l925_92510


namespace NUMINAMATH_CALUDE_points_on_line_value_at_2_l925_92523

/-- A linear function passing through given points -/
def linear_function (x : ℝ) : ℝ := x - 1

/-- The given points satisfy the linear function -/
theorem points_on_line : 
  linear_function (-1) = -2 ∧ 
  linear_function 0 = -1 ∧ 
  linear_function 1 = 0 := by sorry

/-- The y-value corresponding to x = 2 is 1 -/
theorem value_at_2 : linear_function 2 = 1 := by sorry

end NUMINAMATH_CALUDE_points_on_line_value_at_2_l925_92523


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l925_92503

def vector1 : Fin 3 → ℝ := ![3, 4, 5]
def vector2 (k : ℝ) : Fin 3 → ℝ := ![2, k, 3]
def vector3 (k : ℝ) : Fin 3 → ℝ := ![2, 3, k]

def matrix (k : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.of (λ i j => match i, j with
    | 0, 0 => 3 | 0, 1 => 2 | 0, 2 => 2
    | 1, 0 => 4 | 1, 1 => k | 1, 2 => 3
    | 2, 0 => 5 | 2, 1 => 3 | 2, 2 => k
    | _, _ => 0)

theorem parallelepiped_volume (k : ℝ) :
  k > 0 ∧ |Matrix.det (matrix k)| = 30 → k = 3 + Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_l925_92503


namespace NUMINAMATH_CALUDE_six_digit_integers_count_is_60_l925_92509

/-- The number of different six-digit integers that can be formed using the digits 1, 1, 3, 3, 3, and 5 -/
def sixDigitIntegersCount : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of different six-digit integers 
    that can be formed using the digits 1, 1, 3, 3, 3, and 5 is equal to 60 -/
theorem six_digit_integers_count_is_60 : sixDigitIntegersCount = 60 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_integers_count_is_60_l925_92509


namespace NUMINAMATH_CALUDE_expression_value_l925_92550

theorem expression_value (a b : ℤ) (h1 : a = -4) (h2 : b = 3) :
  -2*a - b^3 + 2*a*b = -43 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l925_92550


namespace NUMINAMATH_CALUDE_prove_average_speed_l925_92582

-- Define the distances traveled on each day
def distance_day1 : ℝ := 160
def distance_day2 : ℝ := 280

-- Define the time difference between the two trips
def time_difference : ℝ := 3

-- Define the average speed
def average_speed : ℝ := 40

-- Theorem statement
theorem prove_average_speed :
  (distance_day2 / average_speed) - (distance_day1 / average_speed) = time_difference :=
by
  sorry

end NUMINAMATH_CALUDE_prove_average_speed_l925_92582


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l925_92506

theorem triangle_angle_sum (x : ℝ) : 
  36 + 90 + x = 180 → x = 54 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l925_92506


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l925_92549

/-- Prove that the discount percentage is 10% given the conditions of the sale --/
theorem discount_percentage_proof (actual_sp cost_price : ℝ) (profit_rate : ℝ) : 
  actual_sp = 21000 ∧ 
  cost_price = 17500 ∧ 
  profit_rate = 0.08 → 
  (actual_sp - (cost_price * (1 + profit_rate))) / actual_sp = 0.1 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l925_92549


namespace NUMINAMATH_CALUDE_school_distribution_l925_92537

theorem school_distribution (a b : ℝ) : 
  a + b = 100 →
  0.3 * a + 0.4 * b = 34 →
  a = 60 :=
by sorry

end NUMINAMATH_CALUDE_school_distribution_l925_92537


namespace NUMINAMATH_CALUDE_total_seashells_l925_92573

theorem total_seashells (day1 day2 day3 : ℕ) 
  (h1 : day1 = 27) 
  (h2 : day2 = 46) 
  (h3 : day3 = 19) : 
  day1 + day2 + day3 = 92 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l925_92573


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l925_92578

/-- Given a square banner with side length 12 feet, one large shaded square
    with side length S, and twelve smaller congruent shaded squares with
    side length T, where 12:S = S:T = 4, the total shaded area is 15.75 square feet. -/
theorem shaded_area_calculation (S T : ℝ) : 
  S = 12 / 4 →
  T = S / 4 →
  S^2 + 12 * T^2 = 15.75 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l925_92578


namespace NUMINAMATH_CALUDE_two_digit_powers_of_three_l925_92531

theorem two_digit_powers_of_three : 
  (∃! (s : Finset ℕ), ∀ n : ℕ, n ∈ s ↔ (10 ≤ 3^n ∧ 3^n ≤ 99)) ∧ 
  (∃ (s : Finset ℕ), (∀ n : ℕ, n ∈ s ↔ (10 ≤ 3^n ∧ 3^n ≤ 99)) ∧ s.card = 2) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_powers_of_three_l925_92531


namespace NUMINAMATH_CALUDE_recycling_program_earnings_l925_92574

/-- Calculates the total money earned by Katrina and her friends in the recycling program -/
def total_money_earned (initial_signup : ℕ) (referral_bonus : ℕ) (friends_day1 : ℕ) (friends_week : ℕ) : ℕ :=
  let katrina_earnings := initial_signup + referral_bonus * (friends_day1 + friends_week)
  let friends_earnings := referral_bonus * (friends_day1 + friends_week)
  katrina_earnings + friends_earnings

/-- Theorem stating that the total money earned by Katrina and her friends is $125.00 -/
theorem recycling_program_earnings : 
  total_money_earned 5 5 5 7 = 125 := by
  sorry

#eval total_money_earned 5 5 5 7

end NUMINAMATH_CALUDE_recycling_program_earnings_l925_92574


namespace NUMINAMATH_CALUDE_f_of_x_minus_3_l925_92544

theorem f_of_x_minus_3 (x : ℝ) : (fun (x : ℝ) => x^2) (x - 3) = x^2 - 6*x + 9 := by
  sorry

end NUMINAMATH_CALUDE_f_of_x_minus_3_l925_92544


namespace NUMINAMATH_CALUDE_half_plus_five_equals_thirteen_l925_92525

theorem half_plus_five_equals_thirteen (n : ℝ) : (1/2) * n + 5 = 13 → n = 16 := by
  sorry

end NUMINAMATH_CALUDE_half_plus_five_equals_thirteen_l925_92525


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l925_92515

theorem absolute_value_inequality (x : ℝ) : |x + 3| - |x - 2| ≥ 3 ↔ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l925_92515


namespace NUMINAMATH_CALUDE_number_problem_l925_92535

theorem number_problem : ∃ x : ℝ, x * 0.007 = 0.0063 ∧ x = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l925_92535


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_l925_92563

theorem absolute_value_inequality_solution (a : ℝ) : 
  (∀ x, |x - a| ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_l925_92563


namespace NUMINAMATH_CALUDE_solve_equation_l925_92559

theorem solve_equation : ∃ x : ℚ, 25 * x = 675 ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l925_92559


namespace NUMINAMATH_CALUDE_sin_negative_31pi_over_6_l925_92521

theorem sin_negative_31pi_over_6 : Real.sin (-31 * π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_31pi_over_6_l925_92521


namespace NUMINAMATH_CALUDE_initial_members_family_e_l925_92555

/-- The number of families in Indira Nagar -/
def num_families : ℕ := 6

/-- The initial number of members in family a -/
def family_a : ℕ := 7

/-- The initial number of members in family b -/
def family_b : ℕ := 8

/-- The initial number of members in family c -/
def family_c : ℕ := 10

/-- The initial number of members in family d -/
def family_d : ℕ := 13

/-- The initial number of members in family f -/
def family_f : ℕ := 10

/-- The number of members that left each family -/
def members_left : ℕ := 1

/-- The average number of members in each family after some left -/
def new_average : ℕ := 8

/-- The initial number of members in family e -/
def family_e : ℕ := 6

theorem initial_members_family_e :
  family_a + family_b + family_c + family_d + family_e + family_f - 
  (num_families * members_left) = num_families * new_average := by
  sorry

end NUMINAMATH_CALUDE_initial_members_family_e_l925_92555


namespace NUMINAMATH_CALUDE_fraction_of_one_third_is_one_fifth_l925_92556

theorem fraction_of_one_third_is_one_fifth : (1 : ℚ) / 5 / ((1 : ℚ) / 3) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_one_third_is_one_fifth_l925_92556


namespace NUMINAMATH_CALUDE_extended_quadrilateral_area_l925_92540

/-- A quadrilateral with extended sides -/
structure ExtendedQuadrilateral where
  /-- Side length EF -/
  ef : ℝ
  /-- Side length FG -/
  fg : ℝ
  /-- Side length GH -/
  gh : ℝ
  /-- Side length HE -/
  he : ℝ
  /-- Area of EFGH -/
  area : ℝ
  /-- Extension ratio for EF -/
  ef_ratio : ℝ
  /-- Extension ratio for FG -/
  fg_ratio : ℝ
  /-- Extension ratio for GH -/
  gh_ratio : ℝ
  /-- Extension ratio for HE -/
  he_ratio : ℝ

/-- The area of the extended quadrilateral E'F'G'H' -/
def extended_area (q : ExtendedQuadrilateral) : ℝ := sorry

/-- Theorem stating the area of E'F'G'H' given specific conditions -/
theorem extended_quadrilateral_area 
  (q : ExtendedQuadrilateral)
  (h1 : q.ef = 5)
  (h2 : q.fg = 6)
  (h3 : q.gh = 7)
  (h4 : q.he = 8)
  (h5 : q.area = 12)
  (h6 : q.ef_ratio = 2)
  (h7 : q.fg_ratio = 3/2)
  (h8 : q.gh_ratio = 4/3)
  (h9 : q.he_ratio = 5/4) :
  extended_area q = 84 := by sorry

end NUMINAMATH_CALUDE_extended_quadrilateral_area_l925_92540


namespace NUMINAMATH_CALUDE_largest_n_for_square_sum_l925_92569

theorem largest_n_for_square_sum : ∃ (n : ℕ), n = 1490 ∧ 
  (∀ m : ℕ, m > n → ¬ ∃ k : ℕ, 4^995 + 4^1500 + 4^m = k^2) ∧
  (∃ k : ℕ, 4^995 + 4^1500 + 4^n = k^2) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_square_sum_l925_92569


namespace NUMINAMATH_CALUDE_number_problem_l925_92505

theorem number_problem (x : ℝ) : (10 * x = x + 34.65) → x = 3.85 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l925_92505


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l925_92564

theorem consecutive_even_numbers_sum (n : ℤ) : 
  (∃ (a b c d : ℤ), 
    a = n ∧ b = n + 2 ∧ c = n + 4 ∧ d = n + 6 ∧  -- four consecutive even numbers
    a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2 = 344) →        -- sum of squares is 344
  (n + (n + 2) + (n + 4) + (n + 6) = 36) :=       -- sum of the numbers is 36
by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l925_92564


namespace NUMINAMATH_CALUDE_problem_solution_l925_92594

def problem (a b : ℝ × ℝ) : Prop :=
  let angle := 2 * Real.pi / 3
  let magnitude_b := Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))
  (a = (2, 0)) ∧ 
  (magnitude_b = 1) ∧
  (a.1 * b.1 + a.2 * b.2 = Real.cos angle * magnitude_b * 2) →
  Real.sqrt (((a.1 + 2 * b.1) ^ 2) + ((a.2 + 2 * b.2) ^ 2)) = 2

theorem problem_solution : ∃ (a b : ℝ × ℝ), problem a b := by sorry

end NUMINAMATH_CALUDE_problem_solution_l925_92594


namespace NUMINAMATH_CALUDE_isosceles_triangle_n_count_l925_92562

/-- The number of valid positive integer values for n in the isosceles triangle problem -/
def valid_n_count : ℕ := 7

/-- Checks if a given n satisfies the triangle inequality and angle conditions -/
def is_valid_n (n : ℕ) : Prop :=
  let ab := n + 10
  let bc := 4 * n + 2
  (ab + ab > bc) ∧ 
  (ab + bc > ab) ∧ 
  (bc + ab > ab) ∧
  (bc < ab)  -- This ensures ∠A > ∠B > ∠C in the isosceles triangle

theorem isosceles_triangle_n_count :
  (∃ (S : Finset ℕ), S.card = valid_n_count ∧ 
    (∀ n, n ∈ S ↔ (n > 0 ∧ is_valid_n n)) ∧
    (∀ n, n ∉ S → (n = 0 ∨ ¬is_valid_n n))) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_n_count_l925_92562


namespace NUMINAMATH_CALUDE_stratified_sampling_l925_92501

theorem stratified_sampling (total_capacity : ℕ) (sample_capacity : ℕ) 
  (ratio_A ratio_B ratio_C : ℕ) :
  total_capacity = 56 →
  sample_capacity = 14 →
  ratio_A = 1 →
  ratio_B = 2 →
  ratio_C = 4 →
  ∃ (sample_A sample_B sample_C : ℕ),
    sample_A = 2 ∧
    sample_B = 4 ∧
    sample_C = 8 ∧
    sample_A + sample_B + sample_C = sample_capacity ∧
    sample_A * (ratio_A + ratio_B + ratio_C) = sample_capacity * ratio_A ∧
    sample_B * (ratio_A + ratio_B + ratio_C) = sample_capacity * ratio_B ∧
    sample_C * (ratio_A + ratio_B + ratio_C) = sample_capacity * ratio_C :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_l925_92501


namespace NUMINAMATH_CALUDE_proportion_problem_l925_92560

/-- Given four real numbers a, b, c, d in proportion, where a = 2, b = 3, and d = 6, prove that c = 4. -/
theorem proportion_problem (a b c d : ℝ) 
  (h_prop : a / b = c / d) 
  (h_a : a = 2) 
  (h_b : b = 3) 
  (h_d : d = 6) : 
  c = 4 := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l925_92560


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l925_92572

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = 4 ∧ 
  (x₁^2 - 6*x₁ + 8 = 0) ∧ (x₂^2 - 6*x₂ + 8 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l925_92572


namespace NUMINAMATH_CALUDE_highest_power_of_three_dividing_N_l925_92591

def N : ℕ := sorry

theorem highest_power_of_three_dividing_N : 
  (∃ m : ℕ, N = 3^3 * m) ∧ (∀ k > 3, ¬∃ m : ℕ, N = 3^k * m) := by sorry

end NUMINAMATH_CALUDE_highest_power_of_three_dividing_N_l925_92591


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l925_92546

theorem min_sum_of_squares (x y : ℝ) (h : (x + 8) * (y - 8) = 0) :
  ∃ (min : ℝ), min = 64 ∧ ∀ (a b : ℝ), (a + 8) * (b - 8) = 0 → a^2 + b^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l925_92546


namespace NUMINAMATH_CALUDE_power_difference_theorem_l925_92534

def solution_set : Set (ℕ × ℕ) := {(0, 1), (2, 1), (2, 2), (1, 2)}

theorem power_difference_theorem :
  {(m, n) : ℕ × ℕ | (3:ℤ)^m - (2:ℤ)^n ∈ ({-1, 5, 7} : Set ℤ)} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_power_difference_theorem_l925_92534


namespace NUMINAMATH_CALUDE_josh_marbles_problem_l925_92551

/-- The number of marbles Josh lost -/
def marbles_lost : ℕ := 16

/-- The number of marbles Josh found -/
def marbles_found : ℕ := 8

/-- The initial number of marbles Josh had -/
def initial_marbles : ℕ := 4

theorem josh_marbles_problem :
  marbles_lost = marbles_found + 8 :=
by sorry

end NUMINAMATH_CALUDE_josh_marbles_problem_l925_92551


namespace NUMINAMATH_CALUDE_parking_arrangements_count_l925_92518

-- Define the number of parking spaces
def num_spaces : ℕ := 7

-- Define the number of trucks
def num_trucks : ℕ := 2

-- Define the number of buses
def num_buses : ℕ := 2

-- Define a function to calculate the number of parking arrangements
def num_parking_arrangements (spaces : ℕ) (trucks : ℕ) (buses : ℕ) : ℕ :=
  (spaces.choose trucks) * ((spaces - trucks).choose buses) * (trucks.factorial) * (buses.factorial)

-- Theorem statement
theorem parking_arrangements_count :
  num_parking_arrangements num_spaces num_trucks num_buses = 840 := by
  sorry


end NUMINAMATH_CALUDE_parking_arrangements_count_l925_92518


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l925_92514

-- Define the sets corresponding to ¬p and ¬q
def not_p (x : ℝ) : Prop := x ≤ 0 ∨ x ≥ 2
def not_q (x : ℝ) : Prop := x ≤ 0 ∨ x > 1

-- Define the original conditions p and q
def p (x : ℝ) : Prop := 0 < x ∧ x < 2
def q (x : ℝ) : Prop := 1 / x ≥ 1

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, not_p x → not_q x) ∧ 
  (∃ x, not_q x ∧ ¬(not_p x)) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l925_92514


namespace NUMINAMATH_CALUDE_same_solution_implies_m_value_l925_92593

theorem same_solution_implies_m_value : 
  ∀ (x m : ℝ), 
  (2 * x - m = 1 ∧ 3 * x = 2 * (x - 1)) → 
  m = -5 := by
sorry

end NUMINAMATH_CALUDE_same_solution_implies_m_value_l925_92593


namespace NUMINAMATH_CALUDE_calculation_proofs_l925_92504

theorem calculation_proofs :
  (4.4 * 25 = 110) ∧
  (13.2 * 1.1 - 8.45 = 6.07) ∧
  (76.84 * 103 - 7.684 * 30 = 7684) ∧
  ((2.8 + 3.85 / 3.5) / 3 = 1.3) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proofs_l925_92504


namespace NUMINAMATH_CALUDE_ned_candy_boxes_l925_92543

/-- The number of candy pieces Ned gave to his little brother -/
def pieces_given : ℝ := 7.0

/-- The number of candy pieces in each box -/
def pieces_per_box : ℝ := 6.0

/-- The number of candy pieces Ned still has -/
def pieces_left : ℕ := 42

/-- The number of boxes Ned bought initially -/
def boxes_bought : ℕ := 8

theorem ned_candy_boxes : 
  ⌊(pieces_given + pieces_left : ℝ) / pieces_per_box⌋ = boxes_bought := by
  sorry

end NUMINAMATH_CALUDE_ned_candy_boxes_l925_92543


namespace NUMINAMATH_CALUDE_valid_student_totals_l925_92581

/-- Represents the distribution of students in groups -/
structure StudentDistribution where
  total_groups : Nat
  groups_with_13 : Nat
  total_students : Nat

/-- Checks if a given distribution is valid according to the problem conditions -/
def is_valid_distribution (d : StudentDistribution) : Prop :=
  d.total_groups = 6 ∧
  d.groups_with_13 = 4 ∧
  (d.total_students = 76 ∨ d.total_students = 80)

/-- Theorem stating that the only valid total numbers of students are 76 and 80 -/
theorem valid_student_totals :
  ∀ d : StudentDistribution,
    is_valid_distribution d →
    (d.total_students = 76 ∨ d.total_students = 80) :=
by
  sorry

#check valid_student_totals

end NUMINAMATH_CALUDE_valid_student_totals_l925_92581


namespace NUMINAMATH_CALUDE_divisors_of_72_l925_92585

def divisors (n : ℕ) : Set ℕ := {d | d ∣ n ∧ d > 0}

theorem divisors_of_72 : 
  divisors 72 = {1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72} := by sorry

end NUMINAMATH_CALUDE_divisors_of_72_l925_92585


namespace NUMINAMATH_CALUDE_computer_preference_ratio_l925_92538

theorem computer_preference_ratio (total : ℕ) (mac_preference : ℕ) (no_preference : ℕ) 
  (h1 : total = 210)
  (h2 : mac_preference = 60)
  (h3 : no_preference = 90) :
  (total - (mac_preference + no_preference)) = mac_preference :=
by sorry

end NUMINAMATH_CALUDE_computer_preference_ratio_l925_92538


namespace NUMINAMATH_CALUDE_difference_of_squares_l925_92545

theorem difference_of_squares (a : ℝ) : (a + 2) * (a - 2) = a^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l925_92545


namespace NUMINAMATH_CALUDE_option1_cheaper_at_30_l925_92553

/-- Represents the cost calculation for two shopping options -/
def shopping_options (x : ℕ) : Prop :=
  let shoe_price : ℕ := 200
  let sock_price : ℕ := 40
  let num_shoes : ℕ := 20
  let option1_cost : ℕ := sock_price * x + num_shoes * shoe_price
  let option2_cost : ℕ := (sock_price * x * 9 + num_shoes * shoe_price * 9) / 10
  x > num_shoes ∧ option1_cost < option2_cost

/-- Theorem stating that Option 1 is cheaper when buying 30 pairs of socks -/
theorem option1_cheaper_at_30 : shopping_options 30 := by
  sorry

#check option1_cheaper_at_30

end NUMINAMATH_CALUDE_option1_cheaper_at_30_l925_92553


namespace NUMINAMATH_CALUDE_gcd_bound_l925_92526

theorem gcd_bound (a b : ℕ) (h : ℕ) (h_int : (a + 1) / b + (b + 1) / a = h) :
  Nat.gcd a b ≤ Nat.sqrt (a + b) := by
  sorry

end NUMINAMATH_CALUDE_gcd_bound_l925_92526


namespace NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l925_92576

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_eq1 : a 2 + a 4 + a 5 = a 3 + a 6)
  (h_eq2 : a 9 + a 10 = 3) :
  a 7 = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l925_92576


namespace NUMINAMATH_CALUDE_prime_power_sum_l925_92561

theorem prime_power_sum (p a n : ℕ) : 
  Prime p → 
  a > 0 → 
  n > 0 → 
  2^p + 3^p = a^n → 
  n = 1 := by
sorry

end NUMINAMATH_CALUDE_prime_power_sum_l925_92561
