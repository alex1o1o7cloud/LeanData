import Mathlib

namespace NUMINAMATH_CALUDE_min_value_theorem_l1802_180227

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1/a + 4/b ≥ 9/2 := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1802_180227


namespace NUMINAMATH_CALUDE_jim_ate_15_cookies_l1802_180214

def cookies_problem (cookies_per_batch : ℕ) (flour_per_batch : ℕ) 
  (num_flour_bags : ℕ) (flour_bag_weight : ℕ) (cookies_left : ℕ) : Prop :=
  let total_flour := num_flour_bags * flour_bag_weight
  let num_batches := total_flour / flour_per_batch
  let total_cookies := num_batches * cookies_per_batch
  let cookies_eaten := total_cookies - cookies_left
  cookies_eaten = 15

theorem jim_ate_15_cookies :
  cookies_problem 12 2 4 5 105 := by
  sorry

end NUMINAMATH_CALUDE_jim_ate_15_cookies_l1802_180214


namespace NUMINAMATH_CALUDE_inequality_holds_for_n_2_and_8_l1802_180287

theorem inequality_holds_for_n_2_and_8 (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  ((Real.exp (2 * x)) / (Real.log y)^2 > (x / y)^2) ∧
  ((Real.exp (2 * x)) / (Real.log y)^2 > (x / y)^8) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_for_n_2_and_8_l1802_180287


namespace NUMINAMATH_CALUDE_john_park_distance_l1802_180298

/-- John's journey to the park -/
theorem john_park_distance (speed : ℝ) (time_minutes : ℝ) (h1 : speed = 9) (h2 : time_minutes = 2) :
  speed * (time_minutes / 60) = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_john_park_distance_l1802_180298


namespace NUMINAMATH_CALUDE_triangle_side_length_l1802_180294

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  -- Sides form an arithmetic sequence
  2 * b = a + c →
  -- Angle B is 30°
  B = π / 6 →
  -- Area of triangle is 3/2
  (1 / 2) * a * c * Real.sin B = 3 / 2 →
  -- Side b has length √3 + 1
  b = Real.sqrt 3 + 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1802_180294


namespace NUMINAMATH_CALUDE_right_triangle_side_ratio_l1802_180252

theorem right_triangle_side_ratio (a d : ℝ) (ha : a > 0) (hd : d > 0) :
  (a^2 + (a + d)^2 = (a + 2*d)^2) → (a = 3*d) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_ratio_l1802_180252


namespace NUMINAMATH_CALUDE_prime_iff_factorial_congruence_l1802_180264

theorem prime_iff_factorial_congruence (p : ℕ) (hp : p > 1) : 
  Nat.Prime p ↔ (Nat.factorial (p - 1)) % p = p - 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_iff_factorial_congruence_l1802_180264


namespace NUMINAMATH_CALUDE_trig_inequality_l1802_180267

theorem trig_inequality : 2 * Real.sin (160 * π / 180) < Real.tan (50 * π / 180) ∧
                          Real.tan (50 * π / 180) < 1 + Real.cos (20 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trig_inequality_l1802_180267


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_values_l1802_180263

theorem perpendicular_lines_m_values (m : ℝ) :
  (∀ x y : ℝ, x + 2 * m * y - 1 = 0 ∧ (3 * m - 1) * x - m * y - 1 = 0 →
    ((-1 / (2 * m)) * ((3 * m - 1) / m) = -1)) →
  m = 1 ∨ m = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_values_l1802_180263


namespace NUMINAMATH_CALUDE_product_of_polynomials_l1802_180221

theorem product_of_polynomials (p q : ℤ) : 
  (∀ d : ℤ, (5 * d^2 - 4 * d + p) * (4 * d^2 + q * d - 5) = 20 * d^4 + 11 * d^3 - 45 * d^2 - 20 * d + 25) →
  p = -5 ∧ q = 8 := by
sorry

end NUMINAMATH_CALUDE_product_of_polynomials_l1802_180221


namespace NUMINAMATH_CALUDE_smallest_positive_difference_l1802_180258

theorem smallest_positive_difference (a b : ℤ) (h : 17 * a + 6 * b = 13) :
  ∃ (k : ℤ), a - b = 17 + 23 * k ∧ ∀ (m : ℤ), m > 0 → m = a - b → m ≥ 17 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_difference_l1802_180258


namespace NUMINAMATH_CALUDE_principal_calculation_l1802_180226

/-- Calculates the principal given simple interest, rate, and time -/
def calculate_principal (simple_interest : ℚ) (rate : ℚ) (time : ℕ) : ℚ :=
  simple_interest / (rate * time)

/-- Theorem: Given the specified conditions, the principal is 44625 -/
theorem principal_calculation :
  let simple_interest : ℚ := 4016.25
  let rate : ℚ := 1 / 100
  let time : ℕ := 9
  calculate_principal simple_interest rate time = 44625 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l1802_180226


namespace NUMINAMATH_CALUDE_binomial_plus_ten_l1802_180285

theorem binomial_plus_ten : (Nat.choose 15 12) + 10 = 465 := by
  sorry

end NUMINAMATH_CALUDE_binomial_plus_ten_l1802_180285


namespace NUMINAMATH_CALUDE_find_heaviest_coin_l1802_180231

/-- Represents a weighing scale that may be faulty -/
structure Scale :=
  (isDefective : Bool)

/-- Represents a coin with a certain mass -/
structure Coin :=
  (mass : ℕ)

/-- The minimum number of weighings needed to find the heaviest coin -/
def minWeighings (n : ℕ) : ℕ := 2 * n - 1

theorem find_heaviest_coin (n : ℕ) (h : n > 2) 
  (coins : Fin n → Coin) 
  (scales : Fin n → Scale) 
  (all_different_masses : ∀ i j, i ≠ j → (coins i).mass ≠ (coins j).mass)
  (one_faulty_scale : ∃ i, (scales i).isDefective) :
  ∃ (num_weighings : ℕ), 
    num_weighings = minWeighings n ∧ 
    (∃ heaviest : Fin n, ∀ i, (coins heaviest).mass ≥ (coins i).mass) :=
sorry

end NUMINAMATH_CALUDE_find_heaviest_coin_l1802_180231


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1802_180237

theorem polynomial_simplification (a b c x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
  (((x^2 + a)^2) / ((a - b)*(a - c)) + ((x^2 + b)^2) / ((b - a)*(b - c)) + ((x^2 + c)^2) / ((c - a)*(c - b))) =
  x^4 + x^2*(a + b + c) + (a^2 + b^2 + c^2) := by
  sorry

#check polynomial_simplification

end NUMINAMATH_CALUDE_polynomial_simplification_l1802_180237


namespace NUMINAMATH_CALUDE_towels_per_load_l1802_180268

theorem towels_per_load (total_towels : ℕ) (num_loads : ℕ) (h1 : total_towels = 42) (h2 : num_loads = 6) :
  total_towels / num_loads = 7 := by
  sorry

end NUMINAMATH_CALUDE_towels_per_load_l1802_180268


namespace NUMINAMATH_CALUDE_no_solutions_absolute_value_equation_l1802_180215

theorem no_solutions_absolute_value_equation :
  ¬ ∃ x : ℝ, |x - 5| = |x + 3| + 2 := by
sorry

end NUMINAMATH_CALUDE_no_solutions_absolute_value_equation_l1802_180215


namespace NUMINAMATH_CALUDE_linear_systems_solution_and_expression_l1802_180243

theorem linear_systems_solution_and_expression (a b : ℝ) : 
  (∃ x y : ℝ, (3 * x - 5 * y = 36 ∧ b * x + a * y = -8) ∧
              (2 * x + 5 * y = -26 ∧ a * x - b * y = -4)) →
  (∃ x y : ℝ, x = 2 ∧ y = -6 ∧
              (3 * x - 5 * y = 36 ∧ b * x + a * y = -8) ∧
              (2 * x + 5 * y = -26 ∧ a * x - b * y = -4)) ∧
  (2 * a + b)^2023 = 1 :=
by sorry

end NUMINAMATH_CALUDE_linear_systems_solution_and_expression_l1802_180243


namespace NUMINAMATH_CALUDE_equation_real_root_implies_a_range_l1802_180299

theorem equation_real_root_implies_a_range (a : ℝ) :
  (∃ x : ℝ, 2^(2*x) + 2^x * a + a + 1 = 0) →
  a ∈ Set.Iic (2 - 2 * Real.sqrt 2) ∪ Set.Ici (2 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_real_root_implies_a_range_l1802_180299


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_l1802_180219

theorem quadrilateral_diagonal (sides : Finset ℝ) 
  (h_sides : sides = {1, 2, 2.8, 5, 7.5}) : 
  ∃ (diagonal : ℝ), diagonal ∈ sides ∧
  (∀ (a b c : ℝ), a ∈ sides → b ∈ sides → c ∈ sides → 
   a ≠ diagonal → b ≠ diagonal → c ≠ diagonal → 
   a + b > diagonal ∧ b + c > diagonal ∧ a + c > diagonal) ∧
  diagonal = 2.8 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_l1802_180219


namespace NUMINAMATH_CALUDE_arrangement_theorem_l1802_180228

/-- The number of ways to arrange n distinct objects --/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n distinct objects with k specific objects always adjacent --/
def permutations_with_adjacent (n k : ℕ) : ℕ :=
  permutations (n - k + 1) * permutations k

/-- The number of ways to arrange n distinct objects with k specific objects always adjacent
    and m specific objects never adjacent --/
def permutations_with_adjacent_and_not_adjacent (n k m : ℕ) : ℕ :=
  permutations_with_adjacent n k - permutations_with_adjacent (n - m + 1) (k + m - 1)

theorem arrangement_theorem :
  (permutations_with_adjacent 5 2 = 48) ∧
  (permutations_with_adjacent_and_not_adjacent 5 2 1 = 36) := by
  sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l1802_180228


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1802_180262

def is_solution (x y z : ℕ+) : Prop :=
  x^4 + y^4 + z^4 = 2*x^2*y^2 + 2*y^2*z^2 + 2*z^2*x^2 - 63

def solution_set : Set (ℕ+ × ℕ+ × ℕ+) :=
  {(1, 4, 4), (4, 1, 4), (4, 4, 1), (2, 2, 3), (2, 3, 2), (3, 2, 2)}

theorem diophantine_equation_solution :
  ∀ x y z : ℕ+, is_solution x y z ↔ (x, y, z) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1802_180262


namespace NUMINAMATH_CALUDE_next_door_neighbor_subscriptions_l1802_180283

/-- Represents the number of subscriptions sold to the next-door neighbor -/
def next_door_subscriptions : ℕ := sorry

/-- Represents the total number of subscriptions sold -/
def total_subscriptions : ℕ := sorry

/-- The amount earned per subscription -/
def amount_per_subscription : ℕ := 5

/-- The total amount earned -/
def total_amount_earned : ℕ := 55

/-- Subscriptions sold to parents -/
def parent_subscriptions : ℕ := 4

/-- Subscriptions sold to grandfather -/
def grandfather_subscriptions : ℕ := 1

theorem next_door_neighbor_subscriptions :
  (next_door_subscriptions * amount_per_subscription +
   2 * next_door_subscriptions * amount_per_subscription +
   parent_subscriptions * amount_per_subscription +
   grandfather_subscriptions * amount_per_subscription = total_amount_earned) →
  (total_subscriptions = total_amount_earned / amount_per_subscription) →
  (next_door_subscriptions = 2) := by
  sorry

end NUMINAMATH_CALUDE_next_door_neighbor_subscriptions_l1802_180283


namespace NUMINAMATH_CALUDE_sandcastle_height_difference_l1802_180223

-- Define the heights of the sandcastles
def miki_height : ℝ := 0.8333333333333334
def sister_height : ℝ := 0.5

-- Theorem to prove
theorem sandcastle_height_difference :
  miki_height - sister_height = 0.3333333333333334 := by
  sorry

end NUMINAMATH_CALUDE_sandcastle_height_difference_l1802_180223


namespace NUMINAMATH_CALUDE_square_expansion_area_increase_l1802_180208

theorem square_expansion_area_increase (a : ℝ) : 
  (a + 2)^2 - a^2 = 4*a + 4 := by
  sorry

end NUMINAMATH_CALUDE_square_expansion_area_increase_l1802_180208


namespace NUMINAMATH_CALUDE_quadratic_coefficient_sum_l1802_180216

/-- A quadratic function passing through (2, 5) with vertex at (1, 3) has a - b + c = 11 -/
theorem quadratic_coefficient_sum (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = a * (x - 1)^2 + 3) →  -- vertex form
  a * 2^2 + b * 2 + c = 5 →                         -- passes through (2, 5)
  a - b + c = 11 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_sum_l1802_180216


namespace NUMINAMATH_CALUDE_complex_point_coordinates_l1802_180236

theorem complex_point_coordinates (Z : ℂ) : Z = Complex.I * (1 + Complex.I) → Z.re = -1 ∧ Z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_point_coordinates_l1802_180236


namespace NUMINAMATH_CALUDE_calculate_overall_profit_specific_profit_l1802_180277

/-- Calculate the overall profit or loss from selling a refrigerator and a mobile phone -/
theorem calculate_overall_profit (refrigerator_cost mobile_cost : ℝ) 
  (refrigerator_loss_percent mobile_profit_percent : ℝ) : ℝ :=
  let refrigerator_loss := refrigerator_cost * (refrigerator_loss_percent / 100)
  let refrigerator_sell := refrigerator_cost - refrigerator_loss
  let mobile_profit := mobile_cost * (mobile_profit_percent / 100)
  let mobile_sell := mobile_cost + mobile_profit
  let total_cost := refrigerator_cost + mobile_cost
  let total_sell := refrigerator_sell + mobile_sell
  total_sell - total_cost

/-- Prove that the overall profit is 120 Rs given the specific conditions -/
theorem specific_profit : calculate_overall_profit 15000 8000 4 9 = 120 := by
  sorry

end NUMINAMATH_CALUDE_calculate_overall_profit_specific_profit_l1802_180277


namespace NUMINAMATH_CALUDE_temperature_reaches_target_l1802_180209

/-- The temperature model as a function of time -/
def temperature (t : ℝ) : ℝ := -t^2 + 14*t + 40

/-- The target temperature -/
def target_temp : ℝ := 80

/-- The latest time when the temperature reaches the target -/
def latest_time : ℝ := 10

theorem temperature_reaches_target :
  (∃ t : ℝ, temperature t = target_temp) ∧
  (∀ t : ℝ, temperature t = target_temp → t ≤ latest_time) ∧
  (temperature latest_time = target_temp) := by
  sorry

end NUMINAMATH_CALUDE_temperature_reaches_target_l1802_180209


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l1802_180255

theorem largest_integer_with_remainder (n : ℕ) : 
  n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n → n = 93 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l1802_180255


namespace NUMINAMATH_CALUDE_tiffany_bag_collection_l1802_180210

/-- Represents the number of bags of cans Tiffany collected over three days -/
structure BagCollection where
  monday : Nat
  nextDay : Nat
  dayAfter : Nat
  total : Nat

/-- Theorem stating that given the conditions from the problem, 
    the number of bags collected on the next day must be 3 -/
theorem tiffany_bag_collection (bc : BagCollection) 
  (h1 : bc.monday = 10)
  (h2 : bc.dayAfter = 7)
  (h3 : bc.total = 20)
  (h4 : bc.monday + bc.nextDay + bc.dayAfter = bc.total) :
  bc.nextDay = 3 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_bag_collection_l1802_180210


namespace NUMINAMATH_CALUDE_dog_turns_four_in_two_years_l1802_180249

/-- The number of years until a dog turns 4, given the owner's current age and age when the dog was born. -/
def years_until_dog_turns_four (owner_current_age : ℕ) (owner_age_when_dog_born : ℕ) : ℕ :=
  4 - (owner_current_age - owner_age_when_dog_born)

/-- Theorem: Given that the dog was born when the owner was 15 and the owner is now 17,
    the dog will turn 4 in 2 years. -/
theorem dog_turns_four_in_two_years :
  years_until_dog_turns_four 17 15 = 2 := by
  sorry

end NUMINAMATH_CALUDE_dog_turns_four_in_two_years_l1802_180249


namespace NUMINAMATH_CALUDE_a_range_theorem_l1802_180260

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def q (x a : ℝ) : Prop := x > a

-- Define the range of a
def a_range (a : ℝ) : Prop := a ≥ 1

-- State the theorem
theorem a_range_theorem :
  (∀ x a : ℝ, (¬(q x a) → ¬(p x)) ∧ (∃ x : ℝ, ¬(p x) ∧ q x a)) →
  (∀ a : ℝ, a_range a ↔ ∀ x : ℝ, p x → q x a) :=
sorry

end NUMINAMATH_CALUDE_a_range_theorem_l1802_180260


namespace NUMINAMATH_CALUDE_stock_price_change_l1802_180297

theorem stock_price_change (initial_price : ℝ) (h : initial_price > 0) :
  let price_after_decrease := initial_price * (1 - 0.05)
  let final_price := price_after_decrease * (1 + 0.10)
  let net_change_percentage := (final_price - initial_price) / initial_price * 100
  net_change_percentage = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_change_l1802_180297


namespace NUMINAMATH_CALUDE_sqrt_inequality_fraction_product_inequality_l1802_180201

-- Part 1
theorem sqrt_inequality : Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 := by
  sorry

-- Part 2
theorem fraction_product_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) : 
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_fraction_product_inequality_l1802_180201


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_sixth_l1802_180203

theorem tan_alpha_plus_pi_sixth (α : Real) (h : α > 0) (h' : α < π / 2) 
  (h_eq : Real.sqrt 3 * Real.sin α + Real.cos α = 8 / 5) : 
  Real.tan (α + π / 6) = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_sixth_l1802_180203


namespace NUMINAMATH_CALUDE_balloon_difference_l1802_180254

theorem balloon_difference (your_balloons friend_balloons : ℕ) 
  (h1 : your_balloons = 7) 
  (h2 : friend_balloons = 5) : 
  your_balloons - friend_balloons = 2 := by
  sorry

end NUMINAMATH_CALUDE_balloon_difference_l1802_180254


namespace NUMINAMATH_CALUDE_min_ratio_two_digit_integers_l1802_180200

theorem min_ratio_two_digit_integers (x y : ℕ) : 
  10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 → -- x and y are two-digit positive integers
  (x + y) / 2 = 55 → -- mean of x and y is 55
  ∀ a b : ℕ, 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ (a + b) / 2 = 55 →
  x / y ≤ a / b →
  x / y = 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_min_ratio_two_digit_integers_l1802_180200


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_for_purely_imaginary_l1802_180278

-- Define a complex number
def complex (a b : ℝ) := a + b * Complex.I

-- Define what it means for a complex number to be purely imaginary
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Theorem statement
theorem necessary_not_sufficient_condition_for_purely_imaginary (a b : ℝ) :
  (isPurelyImaginary (complex a b) → a = 0) ∧
  ¬(a = 0 → isPurelyImaginary (complex a b)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_for_purely_imaginary_l1802_180278


namespace NUMINAMATH_CALUDE_orchard_problem_l1802_180257

theorem orchard_problem (total_trees : ℕ) (pure_fuji : ℕ) (pure_gala : ℕ) :
  (pure_fuji : ℚ) = 3 / 4 * total_trees →
  (pure_fuji : ℚ) + 1 / 10 * total_trees = 221 →
  pure_gala = 39 := by
  sorry

end NUMINAMATH_CALUDE_orchard_problem_l1802_180257


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_neg_one_a_range_when_f_bounded_l1802_180244

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - |x - 4|

-- Theorem for part I
theorem solution_set_when_a_is_neg_one :
  {x : ℝ | f (-1) x ≥ 4} = {x : ℝ | x ≥ 7/2} := by sorry

-- Theorem for part II
theorem a_range_when_f_bounded :
  (∀ x : ℝ, |f a x| ≤ 2) → a ∈ Set.Icc 2 6 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_neg_one_a_range_when_f_bounded_l1802_180244


namespace NUMINAMATH_CALUDE_parallelogram_vertex_sum_l1802_180256

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The sum of coordinates of a point -/
def sumCoordinates (p : Point) : ℝ := p.x + p.y

/-- Theorem: Sum of coordinates of vertex C in the given parallelogram is 7 -/
theorem parallelogram_vertex_sum : 
  ∀ (ABCD : Parallelogram),
    ABCD.A = ⟨2, 3⟩ →
    ABCD.B = ⟨-1, 0⟩ →
    ABCD.D = ⟨5, -4⟩ →
    sumCoordinates ABCD.C = 7 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_vertex_sum_l1802_180256


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1802_180246

theorem infinite_series_sum : 
  let r : ℝ := (1 : ℝ) / 1000
  let series_sum := ∑' n, (n : ℝ)^2 * r^(n - 1)
  series_sum = (r + 1) / ((1 - r)^3) := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1802_180246


namespace NUMINAMATH_CALUDE_x_geq_1_necessary_not_sufficient_for_lg_x_geq_1_l1802_180266

-- Define the base-10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem x_geq_1_necessary_not_sufficient_for_lg_x_geq_1 :
  (∀ x : ℝ, lg x ≥ 1 → x ≥ 1) ∧
  ¬(∀ x : ℝ, x ≥ 1 → lg x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_x_geq_1_necessary_not_sufficient_for_lg_x_geq_1_l1802_180266


namespace NUMINAMATH_CALUDE_value_at_2023_l1802_180289

/-- An even function satisfying the given functional equation -/
def EvenFunctionWithProperty (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x, f (x + 2) = 3 - Real.sqrt (6 * f x - f x ^ 2))

/-- The main theorem stating the value of f(2023) -/
theorem value_at_2023 (f : ℝ → ℝ) (h : EvenFunctionWithProperty f) : 
  f 2023 = 3 - (3 / 2) * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_value_at_2023_l1802_180289


namespace NUMINAMATH_CALUDE_oliver_final_amount_l1802_180232

def oliver_money (initial : ℕ) (spent : ℕ) (received : ℕ) : ℕ :=
  initial - spent + received

theorem oliver_final_amount :
  oliver_money 33 4 32 = 61 := by sorry

end NUMINAMATH_CALUDE_oliver_final_amount_l1802_180232


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_l1802_180233

theorem trajectory_is_ellipse (x y : ℝ) 
  (h1 : (2*y)^2 = (1+x)*(1-x)) 
  (h2 : y ≠ 0) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_l1802_180233


namespace NUMINAMATH_CALUDE_students_just_passed_l1802_180211

/-- The number of students who just passed an examination -/
theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ) 
  (h_total : total = 300)
  (h_first : first_div_percent = 26 / 100)
  (h_second : second_div_percent = 54 / 100)
  (h_all_passed : first_div_percent + second_div_percent < 1) :
  total - (first_div_percent * total).floor - (second_div_percent * total).floor = 60 := by
  sorry

end NUMINAMATH_CALUDE_students_just_passed_l1802_180211


namespace NUMINAMATH_CALUDE_equation_solutions_l1802_180218

theorem equation_solutions :
  (∀ x : ℝ, (x - 5)^2 ≠ -1) ∧
  (∀ x : ℝ, |(-2 * x)| + 7 ≠ 0) ∧
  (∃ x : ℝ, Real.sqrt (2 - x) - 3 = 0) ∧
  (∃ x : ℝ, Real.sqrt (2 * x + 6) - 5 = 0) ∧
  (∃ x : ℝ, |(-2 * x)| - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1802_180218


namespace NUMINAMATH_CALUDE_rainwater_chickens_l1802_180222

/-- Mr. Rainwater's farm animals -/
structure Farm where
  cows : ℕ
  goats : ℕ
  chickens : ℕ

/-- The conditions of Mr. Rainwater's farm -/
def rainwater_farm (f : Farm) : Prop :=
  f.cows = 9 ∧ f.goats = 4 * f.cows ∧ f.goats = 2 * f.chickens

/-- Theorem: Mr. Rainwater has 18 chickens -/
theorem rainwater_chickens (f : Farm) (h : rainwater_farm f) : f.chickens = 18 := by
  sorry

end NUMINAMATH_CALUDE_rainwater_chickens_l1802_180222


namespace NUMINAMATH_CALUDE_union_equals_B_implies_a_range_l1802_180238

-- Define the sets A, B, and C
def A : Set ℝ := {x | |x - 1| < 2}
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x - 6 < 0}
def C : Set ℝ := {x | x^2 - 2*x - 15 < 0}

-- State the theorem
theorem union_equals_B_implies_a_range (a : ℝ) :
  A ∪ B a = B a → a ∈ Set.Icc (-5) (-1) :=
by sorry

-- Note: Set.Icc represents a closed interval [a, b]

end NUMINAMATH_CALUDE_union_equals_B_implies_a_range_l1802_180238


namespace NUMINAMATH_CALUDE_tangent_lines_theorem_l1802_180240

-- Define the function f(x)
def f (t : ℝ) (x : ℝ) : ℝ := x^3 + (t - 1) * x^2 - 1

-- Define the derivative of f(x)
def f_deriv (t : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * (t - 1) * x

theorem tangent_lines_theorem (t k : ℝ) (hk : k ≠ 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    f_deriv t x₁ = k ∧
    f_deriv t x₂ = k ∧
    f t x₁ = 2 * x₁ - 1 ∧
    f t x₂ = 2 * x₂ - 1) →
  t + k = 7 := by
sorry

end NUMINAMATH_CALUDE_tangent_lines_theorem_l1802_180240


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_7_l1802_180281

theorem smallest_lcm_with_gcd_7 :
  ∃ (m n : ℕ), 
    1000 ≤ m ∧ m < 10000 ∧
    1000 ≤ n ∧ n < 10000 ∧
    Nat.gcd m n = 7 ∧
    Nat.lcm m n = 144001 ∧
    ∀ (a b : ℕ), 
      1000 ≤ a ∧ a < 10000 ∧
      1000 ≤ b ∧ b < 10000 ∧
      Nat.gcd a b = 7 →
      Nat.lcm a b ≥ 144001 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_7_l1802_180281


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1802_180247

/-- The number of games played in a chess tournament --/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 7 players, where each player plays every other player once,
    the total number of games played is 21. --/
theorem chess_tournament_games :
  num_games 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l1802_180247


namespace NUMINAMATH_CALUDE_expression_evaluation_l1802_180235

theorem expression_evaluation : 200 * (200 - 7) - (200 * 200 - 7) = -1393 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1802_180235


namespace NUMINAMATH_CALUDE_divisibility_by_six_l1802_180276

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number_875a (a : ℕ) : ℕ := 8750 + a

theorem divisibility_by_six (a : ℕ) (h : is_single_digit a) : 
  (number_875a a) % 6 = 0 ↔ a = 4 := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_six_l1802_180276


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1802_180284

theorem solution_set_of_inequality (x : ℝ) : 
  (x-1)/(x^2-x-6) ≥ 0 ↔ x ∈ Set.Ioc (-2) 1 ∪ Set.Ioi 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1802_180284


namespace NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_inequality_l1802_180280

theorem smallest_positive_integer_satisfying_inequality :
  ∀ x : ℕ, x > 0 → (x + 3 < 2 * x - 7) → x ≥ 11 ∧
  (11 + 3 < 2 * 11 - 7) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_inequality_l1802_180280


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l1802_180282

theorem scientific_notation_equivalence : 
  ∃ (a : ℝ) (n : ℤ), 0.0000907 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 9.07 ∧ n = -5 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l1802_180282


namespace NUMINAMATH_CALUDE_five_integer_chords_l1802_180217

/-- A circle with a point P inside it -/
structure CircleWithPoint where
  radius : ℝ
  distance_to_p : ℝ

/-- Count of integer length chords passing through P -/
def count_integer_chords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The specific circle and point from the problem -/
def problem_circle : CircleWithPoint :=
  { radius := 17,
    distance_to_p := 8 }

/-- The theorem stating that there are 5 integer length chords -/
theorem five_integer_chords :
  count_integer_chords problem_circle = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_integer_chords_l1802_180217


namespace NUMINAMATH_CALUDE_arithmetic_progression_difference_l1802_180250

/-- An arithmetic progression with first term a₁, last term aₙ, common difference d, and sum Sₙ. -/
structure ArithmeticProgression (α : Type*) [Field α] where
  a₁ : α
  aₙ : α
  d : α
  n : ℕ
  Sₙ : α
  h₁ : aₙ = a₁ + (n - 1) * d
  h₂ : Sₙ = n / 2 * (a₁ + aₙ)

/-- The common difference of an arithmetic progression can be expressed in terms of its first term, 
last term, and sum. -/
theorem arithmetic_progression_difference (α : Type*) [Field α] (ap : ArithmeticProgression α) :
  ap.d = (ap.aₙ^2 - ap.a₁^2) / (2 * ap.Sₙ - (ap.a₁ + ap.aₙ)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_difference_l1802_180250


namespace NUMINAMATH_CALUDE_x_range_l1802_180291

theorem x_range (x : ℝ) (h1 : 1 / x < 4) (h2 : 1 / x > -2) : x > -1/2 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l1802_180291


namespace NUMINAMATH_CALUDE_course_selection_theorem_l1802_180286

def physical_education_courses : ℕ := 4
def art_courses : ℕ := 4
def total_courses : ℕ := physical_education_courses + art_courses

def choose (n k : ℕ) : ℕ := Nat.choose n k

def two_course_selections : ℕ := choose physical_education_courses 1 * choose art_courses 1

def three_course_selections : ℕ := 
  choose physical_education_courses 2 * choose art_courses 1 + 
  choose physical_education_courses 1 * choose art_courses 2

def total_selections : ℕ := two_course_selections + three_course_selections

theorem course_selection_theorem : total_selections = 64 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_theorem_l1802_180286


namespace NUMINAMATH_CALUDE_find_a_l1802_180265

-- Define the universal set U
def U (a : ℝ) : Set ℝ := {3, 7, a^2 - 2*a - 3}

-- Define set A
def A (a : ℝ) : Set ℝ := {7, |a - 7|}

-- Define the complement of A in U
def complement_A (a : ℝ) : Set ℝ := U a \ A a

-- Theorem statement
theorem find_a : ∃ (a : ℝ), 
  (U a = {3, 7, a^2 - 2*a - 3}) ∧ 
  (A a = {7, |a - 7|}) ∧ 
  (complement_A a = {5}) → 
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_find_a_l1802_180265


namespace NUMINAMATH_CALUDE_maple_trees_remaining_l1802_180296

theorem maple_trees_remaining (initial_maples : Real) (cut_maples : Real) (remaining_maples : Real) : 
  initial_maples = 9.0 → cut_maples = 2.0 → remaining_maples = initial_maples - cut_maples → remaining_maples = 7.0 := by
  sorry

end NUMINAMATH_CALUDE_maple_trees_remaining_l1802_180296


namespace NUMINAMATH_CALUDE_tank_length_proof_l1802_180229

/-- Proves that a rectangular tank with given dimensions and plastering cost has a specific length -/
theorem tank_length_proof (width depth cost_per_sqm total_cost : ℝ) 
  (h_width : width = 12)
  (h_depth : depth = 6)
  (h_cost_per_sqm : cost_per_sqm = 0.70)
  (h_total_cost : total_cost = 520.8)
  : ∃ length : ℝ, 
    length = 25 ∧ 
    total_cost = (2 * width * depth + 2 * length * depth + width * length) * cost_per_sqm :=
by sorry

end NUMINAMATH_CALUDE_tank_length_proof_l1802_180229


namespace NUMINAMATH_CALUDE_vision_data_median_l1802_180212

structure VisionData where
  values : List Float
  frequencies : List Nat
  total_students : Nat

def median (data : VisionData) : Float :=
  sorry

theorem vision_data_median :
  let data : VisionData := {
    values := [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0],
    frequencies := [1, 2, 6, 3, 3, 4, 1, 2, 5, 7, 5],
    total_students := 39
  }
  median data = 4.6 := by sorry

end NUMINAMATH_CALUDE_vision_data_median_l1802_180212


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l1802_180230

-- Define the parabola function
def parabola (x : ℝ) : ℝ := x^2 - 12*x + 32

-- Define the rectangle
structure Rectangle where
  base : ℝ
  height : ℝ

-- Define the conditions of the problem
def inscribedRectangle (r : Rectangle) : Prop :=
  ∃ t : ℝ,
    r.base = 2*t ∧
    r.height = (2*t)/3 ∧
    parabola (6 - t) = r.height ∧
    t > 0

-- The theorem to prove
theorem inscribed_rectangle_area :
  ∀ r : Rectangle, inscribedRectangle r →
    r.base * r.height = 91 + 25 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l1802_180230


namespace NUMINAMATH_CALUDE_intersection_and_union_when_a_is_two_union_with_complement_equals_reals_iff_l1802_180270

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | x < a}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- Theorem for part (1)
theorem intersection_and_union_when_a_is_two :
  (A 2 ∩ B = {x : ℝ | 1 < x ∧ x < 2}) ∧
  (A 2 ∪ B = {x : ℝ | x < 3}) := by
sorry

-- Theorem for part (2)
theorem union_with_complement_equals_reals_iff (a : ℝ) :
  (A a ∪ (Set.univ \ B) = Set.univ) ↔ a ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_a_is_two_union_with_complement_equals_reals_iff_l1802_180270


namespace NUMINAMATH_CALUDE_no_adjacent_standing_probability_l1802_180220

def num_people : ℕ := 10

-- Define a recursive function to calculate the number of favorable arrangements
def favorable_arrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | n + 3 => favorable_arrangements (n + 1) + favorable_arrangements (n + 2)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^num_people

-- Define the probability
def probability : ℚ := favorable_arrangements num_people / total_outcomes

-- Theorem statement
theorem no_adjacent_standing_probability :
  probability = 123 / 1024 := by sorry

end NUMINAMATH_CALUDE_no_adjacent_standing_probability_l1802_180220


namespace NUMINAMATH_CALUDE_circle_tangent_y_intercept_l1802_180245

/-- Two circles with given centers and radii have a common external tangent with y-intercept 135/28 -/
theorem circle_tangent_y_intercept :
  ∃ (m b : ℝ),
    m > 0 ∧
    b = 135 / 28 ∧
    ∀ (x y : ℝ),
      (y = m * x + b) →
      ((x - 1)^2 + (y - 3)^2 = 3^2 ∨ (x - 10)^2 + (y - 8)^2 = 6^2) →
      ∀ (x' y' : ℝ),
        ((x' - 1)^2 + (y' - 3)^2 < 3^2 ∧ (x' - 10)^2 + (y' - 8)^2 < 6^2) →
        (y' ≠ m * x' + b) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_y_intercept_l1802_180245


namespace NUMINAMATH_CALUDE_prob_rain_weekend_is_correct_l1802_180239

-- Define the probabilities of rain for each day
def prob_rain_friday : ℝ := 0.30
def prob_rain_saturday : ℝ := 0.60
def prob_rain_sunday : ℝ := 0.40

-- Define the probability of rain on at least one day during the weekend
def prob_rain_weekend : ℝ := 1 - (1 - prob_rain_friday) * (1 - prob_rain_saturday) * (1 - prob_rain_sunday)

-- Theorem statement
theorem prob_rain_weekend_is_correct : 
  prob_rain_weekend = 0.832 := by sorry

end NUMINAMATH_CALUDE_prob_rain_weekend_is_correct_l1802_180239


namespace NUMINAMATH_CALUDE_ava_activities_duration_l1802_180290

/-- Converts hours to minutes -/
def hours_to_minutes (h : ℕ) : ℕ := h * 60

/-- Represents a duration in hours and minutes -/
structure Duration :=
  (hours : ℕ)
  (minutes : ℕ)

/-- Converts a Duration to total minutes -/
def duration_to_minutes (d : Duration) : ℕ :=
  hours_to_minutes d.hours + d.minutes

/-- The total duration of Ava's activities in minutes -/
def total_duration : ℕ :=
  hours_to_minutes 4 +  -- TV watching
  duration_to_minutes { hours := 2, minutes := 30 } +  -- Video game playing
  duration_to_minutes { hours := 1, minutes := 45 }  -- Walking

theorem ava_activities_duration :
  total_duration = 495 := by sorry

end NUMINAMATH_CALUDE_ava_activities_duration_l1802_180290


namespace NUMINAMATH_CALUDE_square_root_of_nine_l1802_180261

theorem square_root_of_nine : 
  {x : ℝ | x^2 = 9} = {3, -3} := by sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l1802_180261


namespace NUMINAMATH_CALUDE_prob_B_wins_value_l1802_180248

/-- The probability of player B winning in a chess game -/
def prob_B_wins (prob_A_wins : ℝ) (prob_draw : ℝ) : ℝ :=
  1 - prob_A_wins - prob_draw

/-- Theorem: The probability of player B winning is 0.3 -/
theorem prob_B_wins_value :
  prob_B_wins 0.3 0.4 = 0.3 := by
sorry

end NUMINAMATH_CALUDE_prob_B_wins_value_l1802_180248


namespace NUMINAMATH_CALUDE_gus_egg_consumption_l1802_180292

/-- The number of eggs Gus ate for breakfast -/
def breakfast_eggs : ℕ := 2

/-- The number of eggs Gus ate for lunch -/
def lunch_eggs : ℕ := 3

/-- The number of eggs Gus ate for dinner -/
def dinner_eggs : ℕ := 1

/-- The total number of eggs Gus ate -/
def total_eggs : ℕ := breakfast_eggs + lunch_eggs + dinner_eggs

theorem gus_egg_consumption : total_eggs = 6 := by
  sorry

end NUMINAMATH_CALUDE_gus_egg_consumption_l1802_180292


namespace NUMINAMATH_CALUDE_potato_division_l1802_180204

theorem potato_division (total_potatoes : ℕ) (num_people : ℕ) (potatoes_per_person : ℕ) :
  total_potatoes = 24 →
  num_people = 3 →
  total_potatoes = num_people * potatoes_per_person →
  potatoes_per_person = 8 := by
  sorry

end NUMINAMATH_CALUDE_potato_division_l1802_180204


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l1802_180225

theorem smallest_number_of_eggs :
  ∀ (total_eggs : ℕ) (num_containers : ℕ),
    total_eggs > 150 →
    total_eggs = 15 * num_containers - 3 →
    (∀ smaller_total : ℕ, smaller_total > 150 → smaller_total = 15 * (smaller_total / 15) - 3 → smaller_total ≥ total_eggs) →
    total_eggs = 162 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l1802_180225


namespace NUMINAMATH_CALUDE_cubic_polynomial_integer_root_l1802_180224

theorem cubic_polynomial_integer_root 
  (b c : ℚ) 
  (h1 : (3 - Real.sqrt 5)^3 + b*(3 - Real.sqrt 5) + c = 0) 
  (h2 : ∃ (n : ℤ), n^3 + b*n + c = 0) :
  ∃ (n : ℤ), n^3 + b*n + c = 0 ∧ n = -6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_integer_root_l1802_180224


namespace NUMINAMATH_CALUDE_polygon_internal_angle_sum_l1802_180259

theorem polygon_internal_angle_sum (n : ℕ) (h : n > 2) :
  let external_angle : ℚ := 40
  let internal_angle_sum : ℚ := (n - 2) * 180
  external_angle * n = 360 → internal_angle_sum = 1260 := by
  sorry

end NUMINAMATH_CALUDE_polygon_internal_angle_sum_l1802_180259


namespace NUMINAMATH_CALUDE_m_range_l1802_180274

def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem m_range (m : ℝ) : (¬(p m ∨ q m)) → m ≥ 2 := by sorry

end NUMINAMATH_CALUDE_m_range_l1802_180274


namespace NUMINAMATH_CALUDE_bob_homework_time_l1802_180272

theorem bob_homework_time (alice_time bob_time : ℕ) : 
  alice_time = 40 → bob_time = (3 * alice_time) / 8 → bob_time = 15 := by
  sorry

end NUMINAMATH_CALUDE_bob_homework_time_l1802_180272


namespace NUMINAMATH_CALUDE_first_month_sale_is_6435_l1802_180241

/-- Represents the sales data for a grocery shop over 6 months -/
structure GrocerySales where
  average_target : ℕ
  month2 : ℕ
  month3 : ℕ
  month4 : ℕ
  month5 : ℕ
  month6 : ℕ

/-- Calculates the sale in the first month given the sales data -/
def first_month_sale (s : GrocerySales) : ℕ :=
  6 * s.average_target - (s.month2 + s.month3 + s.month4 + s.month5 + s.month6)

/-- Theorem stating that the first month's sale is 6435 given the specific sales data -/
theorem first_month_sale_is_6435 :
  let s : GrocerySales := {
    average_target := 6500,
    month2 := 6927,
    month3 := 6855,
    month4 := 7230,
    month5 := 6562,
    month6 := 4991
  }
  first_month_sale s = 6435 := by
  sorry

end NUMINAMATH_CALUDE_first_month_sale_is_6435_l1802_180241


namespace NUMINAMATH_CALUDE_new_boarders_count_l1802_180295

/-- The number of new boarders that joined the school -/
def new_boarders : ℕ := 30

/-- The initial number of boarders -/
def initial_boarders : ℕ := 150

/-- The initial ratio of boarders to day students -/
def initial_ratio : ℚ := 5 / 12

/-- The final ratio of boarders to day students -/
def final_ratio : ℚ := 1 / 2

theorem new_boarders_count :
  ∃ (initial_day_students : ℕ),
    (initial_boarders : ℚ) / initial_day_students = initial_ratio ∧
    (initial_boarders + new_boarders : ℚ) / initial_day_students = final_ratio :=
by sorry

end NUMINAMATH_CALUDE_new_boarders_count_l1802_180295


namespace NUMINAMATH_CALUDE_sum_remainder_l1802_180273

theorem sum_remainder (a b c d e : ℕ) : 
  a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
  a % 13 = 3 → b % 13 = 5 → c % 13 = 7 → d % 13 = 9 → e % 13 = 12 →
  (a + b + c + d + e) % 13 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_l1802_180273


namespace NUMINAMATH_CALUDE_probability_at_least_one_female_l1802_180293

def total_students : ℕ := 5
def male_students : ℕ := 3
def female_students : ℕ := 2
def selected_students : ℕ := 3

theorem probability_at_least_one_female :
  let total_combinations := Nat.choose total_students selected_students
  let all_male_combinations := Nat.choose male_students selected_students
  (1 : ℚ) - (all_male_combinations : ℚ) / (total_combinations : ℚ) = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_female_l1802_180293


namespace NUMINAMATH_CALUDE_min_cuts_for_eleven_sided_polygons_l1802_180213

/-- Represents a straight-line cut on a piece of paper -/
structure Cut where
  -- Add necessary fields

/-- Represents a polygon on the table -/
structure Polygon where
  sides : ℕ

/-- Represents the state of the paper after a series of cuts -/
structure PaperState where
  polygons : List Polygon

/-- Function to apply a cut to a paper state -/
def applyCut (state : PaperState) (cut : Cut) : PaperState :=
  sorry

/-- Function to count the number of eleven-sided polygons in a paper state -/
def countElevenSidedPolygons (state : PaperState) : ℕ :=
  sorry

/-- Theorem stating the minimum number of cuts required -/
theorem min_cuts_for_eleven_sided_polygons :
  ∀ (initial : PaperState),
    (∃ (cuts : List Cut),
      cuts.length = 2015 ∧
      countElevenSidedPolygons (cuts.foldl applyCut initial) ≥ 252) ∧
    (∀ (cuts : List Cut),
      cuts.length < 2015 →
      countElevenSidedPolygons (cuts.foldl applyCut initial) < 252) :=
by
  sorry

end NUMINAMATH_CALUDE_min_cuts_for_eleven_sided_polygons_l1802_180213


namespace NUMINAMATH_CALUDE_seating_arrangements_with_restrictions_l1802_180205

/-- The number of ways to seat n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to seat n people in a row where two specific people must sit together -/
def arrangementsWithPairTogether (n : ℕ) : ℕ := (Nat.factorial (n - 1)) * (Nat.factorial 2)

/-- The number of ways to seat n people in a row where three specific people must sit together -/
def arrangementsWithTrioTogether (n : ℕ) : ℕ := (Nat.factorial (n - 2)) * (Nat.factorial 3)

/-- The number of ways to seat 7 people in a row where 3 specific people cannot sit next to each other -/
theorem seating_arrangements_with_restrictions : 
  totalArrangements 7 - 3 * arrangementsWithPairTogether 7 + arrangementsWithTrioTogether 7 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_with_restrictions_l1802_180205


namespace NUMINAMATH_CALUDE_triangle_inequality_l1802_180253

theorem triangle_inequality (a b c : ℝ) (h : a + b + c = 1) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) :
  5 * (a^2 + b^2 + c^2) + 18 * a * b * c ≥ 7/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1802_180253


namespace NUMINAMATH_CALUDE_prime_factorization_problem_l1802_180269

theorem prime_factorization_problem :
  2006^2 * 2262 - 669^2 * 3599 + 1593^2 * 1337 = 2 * 3 * 7 * 13 * 29 * 59 * 61 * 191 := by
  sorry

end NUMINAMATH_CALUDE_prime_factorization_problem_l1802_180269


namespace NUMINAMATH_CALUDE_subtracted_number_l1802_180202

theorem subtracted_number (t k x : ℝ) : 
  t = 5/9 * (k - x) → 
  t = 20 → 
  k = 68 → 
  x = 32 := by
sorry

end NUMINAMATH_CALUDE_subtracted_number_l1802_180202


namespace NUMINAMATH_CALUDE_triangle_side_expression_l1802_180251

theorem triangle_side_expression (m : ℝ) : 
  (2 : ℝ) > 0 ∧ 5 > 0 ∧ m > 0 ∧ 
  2 + 5 > m ∧ 2 + m > 5 ∧ 5 + m > 2 →
  Real.sqrt ((m - 3)^2) + Real.sqrt ((m - 7)^2) = 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_expression_l1802_180251


namespace NUMINAMATH_CALUDE_min_y_value_l1802_180207

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 16*x + 40*y) :
  ∃ (y_min : ℝ), y_min = 20 - Real.sqrt 464 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 = 16*x' + 40*y' → y' ≥ y_min := by
sorry

end NUMINAMATH_CALUDE_min_y_value_l1802_180207


namespace NUMINAMATH_CALUDE_morning_routine_duration_l1802_180279

def coffee_bagel_time : ℕ := 15

def paper_eating_time : ℕ := 2 * coffee_bagel_time

def total_routine_time : ℕ := coffee_bagel_time + paper_eating_time

theorem morning_routine_duration :
  total_routine_time = 45 :=
by sorry

end NUMINAMATH_CALUDE_morning_routine_duration_l1802_180279


namespace NUMINAMATH_CALUDE_max_value_polynomial_l1802_180288

/-- Given real numbers x and y such that x + y = 5, 
    the maximum value of x^5*y + x^4*y^2 + x^3*y^3 + x^2*y^4 + x*y^5 is 30625/44 -/
theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  (∃ (z w : ℝ), z + w = 5 ∧ 
    ∀ (a b : ℝ), a + b = 5 → 
      z^5*w + z^4*w^2 + z^3*w^3 + z^2*w^4 + z*w^5 ≥ a^5*b + a^4*b^2 + a^3*b^3 + a^2*b^4 + a*b^5) ∧
  (∀ (a b : ℝ), a + b = 5 → 
    x^5*y + x^4*y^2 + x^3*y^3 + x^2*y^4 + x*y^5 ≤ 30625/44) :=
by sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l1802_180288


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1802_180206

theorem fractional_equation_solution :
  ∃ x : ℝ, (3 / (x - 1) = 5 + 3 * x / (1 - x)) ∧ (x = 4) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1802_180206


namespace NUMINAMATH_CALUDE_num_distinct_representations_eq_six_l1802_180271

/-- Represents a digit configuration using matchsticks -/
def DigitConfig := Nat

/-- The maximum number of matchsticks in the original configuration -/
def max_sticks : Nat := 7

/-- The set of all possible digit configurations -/
def all_configs : Finset DigitConfig := sorry

/-- The number of distinct digit representations -/
def num_distinct_representations : Nat := Finset.card all_configs

/-- Theorem stating that the number of distinct representations is 6 -/
theorem num_distinct_representations_eq_six :
  num_distinct_representations = 6 := by sorry

end NUMINAMATH_CALUDE_num_distinct_representations_eq_six_l1802_180271


namespace NUMINAMATH_CALUDE_unique_solution_l1802_180242

theorem unique_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : 2*x - 2*y + 1/z = 1/2014)
  (eq2 : 2*y - 2*z + 1/x = 1/2014)
  (eq3 : 2*z - 2*x + 1/y = 1/2014) :
  x = 2014 ∧ y = 2014 ∧ z = 2014 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l1802_180242


namespace NUMINAMATH_CALUDE_smallest_AAAB_l1802_180234

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def AB (a b : ℕ) : ℕ := 10 * a + b

def AAAB (a b : ℕ) : ℕ := 1000 * a + 100 * a + 10 * a + b

theorem smallest_AAAB :
  ∀ a b : ℕ,
    a ≠ b →
    a < 10 →
    b < 10 →
    is_two_digit (AB a b) →
    is_four_digit (AAAB a b) →
    7 * (AB a b) = AAAB a b →
    ∀ a' b' : ℕ,
      a' ≠ b' →
      a' < 10 →
      b' < 10 →
      is_two_digit (AB a' b') →
      is_four_digit (AAAB a' b') →
      7 * (AB a' b') = AAAB a' b' →
      AAAB a b ≤ AAAB a' b' →
    AAAB a b = 6661 :=
by sorry

end NUMINAMATH_CALUDE_smallest_AAAB_l1802_180234


namespace NUMINAMATH_CALUDE_problem_statement_l1802_180275

theorem problem_statement (x y : ℝ) (h1 : x + 3*y = 5) (h2 : 2*x - y = 2) :
  2*x^2 + 5*x*y - 3*y^2 = 10 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1802_180275
