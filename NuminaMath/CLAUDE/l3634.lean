import Mathlib

namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3634_363479

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3634_363479


namespace NUMINAMATH_CALUDE_phi_value_l3634_363407

theorem phi_value : ∃! (Φ : ℕ), Φ < 10 ∧ 504 / Φ = 40 + 3 * Φ :=
  sorry

end NUMINAMATH_CALUDE_phi_value_l3634_363407


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3634_363449

/-- The sum of the repeating decimals 0.4̄ and 0.26̄ is equal to 70/99 -/
theorem sum_of_repeating_decimals : 
  (∃ (x y : ℚ), x = 4/9 ∧ y = 26/99 ∧ x + y = 70/99) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3634_363449


namespace NUMINAMATH_CALUDE_article_cost_l3634_363420

/-- Proves that the cost of an article is 50 Rs given the profit conditions -/
theorem article_cost (original_profit : Real) (reduced_cost_percentage : Real) 
  (price_reduction : Real) (new_profit : Real) :
  original_profit = 0.25 →
  reduced_cost_percentage = 0.20 →
  price_reduction = 10.50 →
  new_profit = 0.30 →
  ∃ (cost : Real), cost = 50 ∧
    (cost + original_profit * cost) - price_reduction = 
    (cost - reduced_cost_percentage * cost) + new_profit * (cost - reduced_cost_percentage * cost) :=
by sorry

end NUMINAMATH_CALUDE_article_cost_l3634_363420


namespace NUMINAMATH_CALUDE_bowling_ball_weight_proof_l3634_363415

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 18.75

/-- The weight of one kayak in pounds -/
def kayak_weight : ℝ := 30

theorem bowling_ball_weight_proof :
  (∀ (b k : ℝ), 8 * b = 5 * k ∧ 4 * k = 120 → b = bowling_ball_weight) :=
by sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_proof_l3634_363415


namespace NUMINAMATH_CALUDE_triangle_angle_A_l3634_363488

theorem triangle_angle_A (a b : ℝ) (A B : ℝ) (hb : b = 2 * Real.sqrt 3) (ha : a = 2) (hB : B = π / 3) :
  A = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l3634_363488


namespace NUMINAMATH_CALUDE_both_arithmetic_and_geometric_is_geometric_with_ratio_one_l3634_363401

/-- A sequence that is both arithmetic and geometric -/
def BothArithmeticAndGeometric (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r)

/-- Theorem: A sequence that is both arithmetic and geometric is a geometric sequence with common ratio 1 -/
theorem both_arithmetic_and_geometric_is_geometric_with_ratio_one 
  (a : ℕ → ℝ) (h : BothArithmeticAndGeometric a) : 
  ∃ r : ℝ, r = 1 ∧ ∀ n : ℕ, a (n + 1) = a n * r :=
by sorry

end NUMINAMATH_CALUDE_both_arithmetic_and_geometric_is_geometric_with_ratio_one_l3634_363401


namespace NUMINAMATH_CALUDE_angelina_walking_speed_l3634_363464

/-- Angelina's walking problem -/
theorem angelina_walking_speed 
  (home_to_grocery : ℝ) 
  (grocery_to_gym : ℝ) 
  (initial_speed : ℝ) 
  (h1 : home_to_grocery = 100)
  (h2 : grocery_to_gym = 180)
  (h3 : home_to_grocery / initial_speed - grocery_to_gym / (2 * initial_speed) = 40) :
  2 * initial_speed = 1/2 := by
sorry

end NUMINAMATH_CALUDE_angelina_walking_speed_l3634_363464


namespace NUMINAMATH_CALUDE_trajectory_of_moving_circle_l3634_363445

/-- The trajectory of the center of a moving circle -/
def trajectory_equation (x y : ℝ) : Prop :=
  (x - 5)^2 + (y + 7)^2 = 25

/-- The equation of the stationary circle -/
def stationary_circle (x y : ℝ) : Prop :=
  (x - 5)^2 + (y + 7)^2 = 16

/-- The radius of the moving circle -/
def moving_circle_radius : ℝ := 1

theorem trajectory_of_moving_circle :
  ∀ x y : ℝ,
  (∃ x₀ y₀ : ℝ, stationary_circle x₀ y₀ ∧ 
    ((x - x₀)^2 + (y - y₀)^2 = (moving_circle_radius + 4)^2)) →
  trajectory_equation x y :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_moving_circle_l3634_363445


namespace NUMINAMATH_CALUDE_power_function_property_l3634_363482

/-- A power function is a function of the form f(x) = x^a for some real number a -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ a

theorem power_function_property (f : ℝ → ℝ) (h1 : IsPowerFunction f) (h2 : f 4 = 2) :
  f (1/4) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_property_l3634_363482


namespace NUMINAMATH_CALUDE_arithmetic_sum_proof_l3634_363403

/-- 
Given an arithmetic sequence with:
- first term a₁ = k² + 1
- common difference d = 1
- number of terms n = 2k + 1

Prove that the sum of the first 2k + 1 terms is k³ + (k + 1)³
-/
theorem arithmetic_sum_proof (k : ℕ) : 
  let a₁ : ℕ := k^2 + 1
  let d : ℕ := 1
  let n : ℕ := 2 * k + 1
  let S := n * (2 * a₁ + (n - 1) * d) / 2
  S = k^3 + (k + 1)^3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_proof_l3634_363403


namespace NUMINAMATH_CALUDE_smallest_number_of_purple_marbles_l3634_363417

theorem smallest_number_of_purple_marbles :
  ∀ (n : ℕ),
  (n ≥ 10) →  -- Ensuring n is at least 10 to satisfy all conditions
  (n % 10 = 0) →  -- n must be a multiple of 10
  (n / 2 : ℕ) + (n / 5 : ℕ) + 7 < n →  -- Ensuring there's at least one purple marble
  (∃ (blue red green purple : ℕ),
    blue = n / 2 ∧
    red = n / 5 ∧
    green = 7 ∧
    purple = n - (blue + red + green) ∧
    purple > 0) →
  (∀ (m : ℕ),
    m < n →
    ¬(∃ (blue red green purple : ℕ),
      blue = m / 2 ∧
      red = m / 5 ∧
      green = 7 ∧
      purple = m - (blue + red + green) ∧
      purple > 0)) →
  (n - (n / 2 + n / 5 + 7) = 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_purple_marbles_l3634_363417


namespace NUMINAMATH_CALUDE_system_solution_l3634_363446

theorem system_solution (a b c x y z : ℝ) : 
  (a * x + (a - b) * y + (a - c) * z = a^2 + (b - c)^2) ∧
  ((b - a) * x + b * y + (b - c) * z = b^2 + (c - a)^2) ∧
  ((c - a) * x + (c - b) * y + c * z = c^2 + (a - b)^2) →
  (x = b + c - a ∧ y = c + a - b ∧ z = a + b - c) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3634_363446


namespace NUMINAMATH_CALUDE_min_expression_l3634_363491

theorem min_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x * y / 2 + 18 / (x * y) ≥ 6) ∧ 
  ((x * y / 2 + 18 / (x * y) = 6) → (y / 2 + x / 3 ≥ 2)) ∧
  ((x * y / 2 + 18 / (x * y) = 6) ∧ (y / 2 + x / 3 = 2) → (x = 3 ∧ y = 2)) := by
sorry

end NUMINAMATH_CALUDE_min_expression_l3634_363491


namespace NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l3634_363444

theorem infinite_geometric_series_ratio
  (a : ℝ)  -- first term
  (S : ℝ)  -- sum of the series
  (h1 : a = 328)
  (h2 : S = 2009)
  (h3 : S = a / (1 - r))  -- formula for sum of infinite geometric series
  : r = 41 / 49 :=
by sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l3634_363444


namespace NUMINAMATH_CALUDE_length_a_prime_b_prime_l3634_363499

/-- Given points A, B, and C, where A' and B' are the intersections of lines AC and BC with the line y = x respectively, the length of A'B' is (3√2)/10. -/
theorem length_a_prime_b_prime (A B C A' B' : ℝ × ℝ) : 
  A = (0, 7) →
  B = (0, 14) →
  C = (3, 5) →
  (A'.1 = A'.2) →  -- A' is on y = x
  (B'.1 = B'.2) →  -- B' is on y = x
  (C.2 - A.2) / (C.1 - A.1) = (A'.2 - A.2) / (A'.1 - A.1) →  -- A' is on line AC
  (C.2 - B.2) / (C.1 - B.1) = (B'.2 - B.2) / (B'.1 - B.1) →  -- B' is on line BC
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = (3 * Real.sqrt 2) / 10 := by
sorry

end NUMINAMATH_CALUDE_length_a_prime_b_prime_l3634_363499


namespace NUMINAMATH_CALUDE_sin_alpha_plus_pi_sixth_l3634_363465

theorem sin_alpha_plus_pi_sixth (α : Real) 
  (h1 : 0 < α) (h2 : α < π / 2) 
  (h3 : Real.sin (2 * α - π / 6) = -1 / 3) : 
  Real.sin (α + π / 6) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_pi_sixth_l3634_363465


namespace NUMINAMATH_CALUDE_binomial_seven_two_l3634_363406

theorem binomial_seven_two : Nat.choose 7 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_seven_two_l3634_363406


namespace NUMINAMATH_CALUDE_recipe_total_l3634_363466

theorem recipe_total (eggs : ℕ) (flour : ℕ) : 
  eggs = 60 → flour = eggs / 2 → eggs + flour = 90 := by
  sorry

end NUMINAMATH_CALUDE_recipe_total_l3634_363466


namespace NUMINAMATH_CALUDE_perpendicular_line_implies_parallel_planes_skew_lines_parallel_to_planes_implies_parallel_planes_l3634_363418

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (skew : Line → Line → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the planes α and β
variable (α β : Plane)

-- Theorem 1: Condition ①
theorem perpendicular_line_implies_parallel_planes 
  (a : Line) 
  (h1 : perpendicular a α) 
  (h2 : perpendicular a β) : 
  parallel α β := by sorry

-- Theorem 2: Condition ④
theorem skew_lines_parallel_to_planes_implies_parallel_planes 
  (a b : Line) 
  (h1 : contains α a) 
  (h2 : contains β b) 
  (h3 : line_parallel_plane a β) 
  (h4 : line_parallel_plane b α) 
  (h5 : skew a b) : 
  parallel α β := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_implies_parallel_planes_skew_lines_parallel_to_planes_implies_parallel_planes_l3634_363418


namespace NUMINAMATH_CALUDE_march_and_may_greatest_drop_l3634_363455

/-- Represents the months of the year --/
inductive Month
| January | February | March | April | May | June | July | August

/-- Price change for each month --/
def price_change : Month → ℝ
| Month.January  => -1.00
| Month.February => 1.50
| Month.March    => -3.00
| Month.April    => 2.00
| Month.May      => -3.00
| Month.June     => 0.50
| Month.July     => -2.50
| Month.August   => -1.50

/-- Predicate to check if a month has the greatest price drop --/
def has_greatest_drop (m : Month) : Prop :=
  ∀ n : Month, price_change m ≤ price_change n

/-- Theorem stating that March and May have the greatest monthly drop in price --/
theorem march_and_may_greatest_drop :
  has_greatest_drop Month.March ∧ has_greatest_drop Month.May :=
sorry

end NUMINAMATH_CALUDE_march_and_may_greatest_drop_l3634_363455


namespace NUMINAMATH_CALUDE_arrangements_count_l3634_363471

/-- The number of ways to arrange four people in a row with one person not at the ends -/
def arrangements_with_restriction : ℕ :=
  let total_people : ℕ := 4
  let restricted_person : ℕ := 1
  let unrestricted_people : ℕ := total_people - restricted_person
  let unrestricted_arrangements : ℕ := Nat.factorial unrestricted_people
  let valid_positions : ℕ := unrestricted_people - 1
  unrestricted_arrangements * valid_positions

theorem arrangements_count :
  arrangements_with_restriction = 12 :=
sorry

end NUMINAMATH_CALUDE_arrangements_count_l3634_363471


namespace NUMINAMATH_CALUDE_inequality_proof_l3634_363490

theorem inequality_proof (w x y z : ℝ) (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  12 / (w + x + y + z) ≤ 1/(w + x) + 1/(w + y) + 1/(w + z) + 1/(x + y) + 1/(x + z) + 1/(y + z) ∧
  1/(w + x) + 1/(w + y) + 1/(w + z) + 1/(x + y) + 1/(x + z) + 1/(y + z) ≤ 3/4 * (1/w + 1/x + 1/y + 1/z) :=
by sorry


end NUMINAMATH_CALUDE_inequality_proof_l3634_363490


namespace NUMINAMATH_CALUDE_prime_product_660_l3634_363441

theorem prime_product_660 (w x y z a b c d : ℕ) : 
  (w.Prime ∧ x.Prime ∧ y.Prime ∧ z.Prime) →
  (w < x ∧ x < y ∧ y < z) →
  ((w^a) * (x^b) * (y^c) * (z^d) = 660) →
  ((a + b) - (c + d) = 1) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_product_660_l3634_363441


namespace NUMINAMATH_CALUDE_proposition_truth_l3634_363486

theorem proposition_truth : 
  -- Proposition A
  (∃ a b m : ℝ, a < b ∧ ¬(a * m^2 < b * m^2)) ∧
  -- Proposition B
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a * b > 1) ∧
  -- Proposition C
  (∃ x : ℝ, x^2 = 9 ∧ x ≠ 3) ∧
  -- Proposition D
  ((∀ x : ℝ, x > 1 → 1/x < 1) ∧ (∃ x : ℝ, 1/x < 1 ∧ ¬(x > 1))) :=
by sorry

end NUMINAMATH_CALUDE_proposition_truth_l3634_363486


namespace NUMINAMATH_CALUDE_absolute_difference_l3634_363447

theorem absolute_difference (a x : ℝ) (h1 : a < 0) (h2 : |a| * x ≤ a) : 
  |x + 1| - |x - 2| = -3 := by
sorry

end NUMINAMATH_CALUDE_absolute_difference_l3634_363447


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_seven_sqrt_two_over_six_l3634_363452

theorem sqrt_difference_equals_seven_sqrt_two_over_six :
  Real.sqrt (9 / 2) - Real.sqrt (2 / 9) = 7 * Real.sqrt 2 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_seven_sqrt_two_over_six_l3634_363452


namespace NUMINAMATH_CALUDE_not_product_of_consecutive_numbers_l3634_363463

theorem not_product_of_consecutive_numbers (n : ℕ) :
  ¬ ∃ k : ℕ, 2 * (6^n + 1) = k * (k + 1) := by
sorry

end NUMINAMATH_CALUDE_not_product_of_consecutive_numbers_l3634_363463


namespace NUMINAMATH_CALUDE_range_of_m2_plus_n2_l3634_363410

-- Define a decreasing function on ℝ
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- Define the theorem
theorem range_of_m2_plus_n2
  (f : ℝ → ℝ)
  (h_decreasing : DecreasingFunction f)
  (h_inequality : ∀ m n : ℝ, f (n^2 - 10*n - 15) ≥ f (12 - m^2 + 24*m)) :
  ∀ m n : ℝ, 0 ≤ m^2 + n^2 ∧ m^2 + n^2 ≤ 729 :=
sorry

end NUMINAMATH_CALUDE_range_of_m2_plus_n2_l3634_363410


namespace NUMINAMATH_CALUDE_hayley_stickers_l3634_363493

/-- The number of Hayley's close friends who like stickers. -/
def num_friends : ℕ := 9

/-- The number of stickers each friend would receive if distributed equally. -/
def stickers_per_friend : ℕ := 8

/-- The total number of stickers Hayley has. -/
def total_stickers : ℕ := num_friends * stickers_per_friend

theorem hayley_stickers : total_stickers = 72 := by
  sorry

end NUMINAMATH_CALUDE_hayley_stickers_l3634_363493


namespace NUMINAMATH_CALUDE_exists_valid_arrangement_l3634_363480

/-- A permutation of integers from 1 to n -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- Check if a number is divisible by either 4 or 7 -/
def isDivisibleBy4Or7 (n : ℕ) : Prop := n % 4 = 0 ∨ n % 7 = 0

/-- Check if a permutation satisfies the adjacency condition when arranged in a circle -/
def isValidCircularArrangement (p : Permutation 2015) : Prop :=
  ∀ i : Fin 2015, isDivisibleBy4Or7 ((p i).val + (p (i + 1)).val)

theorem exists_valid_arrangement : ∃ p : Permutation 2015, isValidCircularArrangement p := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_arrangement_l3634_363480


namespace NUMINAMATH_CALUDE_investment_interest_rate_proof_l3634_363400

theorem investment_interest_rate_proof 
  (total_investment : ℝ)
  (first_part : ℝ)
  (first_rate : ℝ)
  (total_interest : ℝ)
  (h1 : total_investment = 3500)
  (h2 : first_part = 1549.9999999999998)
  (h3 : first_rate = 3)
  (h4 : total_interest = 144)
  (h5 : first_part * (first_rate / 100) + (total_investment - first_part) * (second_rate / 100) = total_interest) :
  second_rate = 5 := by
  sorry


end NUMINAMATH_CALUDE_investment_interest_rate_proof_l3634_363400


namespace NUMINAMATH_CALUDE_wendy_furniture_assembly_time_l3634_363460

/-- Calculates the total time spent assembling furniture --/
def total_assembly_time (chair_count : ℕ) (table_count : ℕ) (bookshelf_count : ℕ)
                        (chair_time : ℕ) (table_time : ℕ) (bookshelf_time : ℕ) : ℕ :=
  chair_count * chair_time + table_count * table_time + bookshelf_count * bookshelf_time

/-- Theorem stating that the total assembly time for Wendy's furniture is 84 minutes --/
theorem wendy_furniture_assembly_time :
  total_assembly_time 4 3 2 6 10 15 = 84 := by
  sorry

#eval total_assembly_time 4 3 2 6 10 15

end NUMINAMATH_CALUDE_wendy_furniture_assembly_time_l3634_363460


namespace NUMINAMATH_CALUDE_dalton_needs_sixteen_more_l3634_363461

theorem dalton_needs_sixteen_more : ∀ (jump_rope board_game ball puzzle saved uncle_gift : ℕ),
  jump_rope = 9 →
  board_game = 15 →
  ball = 5 →
  puzzle = 8 →
  saved = 7 →
  uncle_gift = 14 →
  jump_rope + board_game + ball + puzzle - (saved + uncle_gift) = 16 := by
  sorry

end NUMINAMATH_CALUDE_dalton_needs_sixteen_more_l3634_363461


namespace NUMINAMATH_CALUDE_largest_valid_number_l3634_363421

def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ i, 1 < i ∧ i < digits.length - 1 →
    digits[i]! < (digits[i-1]! + digits[i+1]!) / 2

theorem largest_valid_number :
  (96433469 : ℕ).digits 10 = [9, 6, 4, 3, 3, 4, 6, 9] ∧
  is_valid_number 96433469 ∧
  ∀ m : ℕ, m > 96433469 → ¬ is_valid_number m :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l3634_363421


namespace NUMINAMATH_CALUDE_floor_times_self_equals_108_l3634_363496

theorem floor_times_self_equals_108 :
  ∃! (x : ℝ), (⌊x⌋ : ℝ) * x = 108 ∧ x = 10.8 := by sorry

end NUMINAMATH_CALUDE_floor_times_self_equals_108_l3634_363496


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l3634_363458

theorem adult_ticket_cost (child_cost : ℕ) (total_tickets : ℕ) (total_revenue : ℕ) (adult_count : ℕ)
  (h1 : child_cost = 6)
  (h2 : total_tickets = 225)
  (h3 : total_revenue = 1875)
  (h4 : adult_count = 175) :
  (total_revenue - child_cost * (total_tickets - adult_count)) / adult_count = 9 := by
  sorry

#eval (1875 - 6 * (225 - 175)) / 175  -- Should output 9

end NUMINAMATH_CALUDE_adult_ticket_cost_l3634_363458


namespace NUMINAMATH_CALUDE_uncle_wang_flower_pots_l3634_363472

theorem uncle_wang_flower_pots :
  ∃! x : ℕ,
    ∃ a : ℕ,
      x / 2 + x / 4 + x / 7 + a = x ∧
      1 ≤ a ∧ a < 6 :=
by
  sorry

end NUMINAMATH_CALUDE_uncle_wang_flower_pots_l3634_363472


namespace NUMINAMATH_CALUDE_equation_solution_l3634_363470

theorem equation_solution : ∃ (S : Set ℝ), S = {x : ℝ | (3*x + 6) / (x^2 + 5*x + 6) = (3 - x) / (x - 2) ∧ x ≠ 2 ∧ x ≠ -2} ∧ S = {3, -3} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3634_363470


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3634_363485

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∃ (min : ℝ), min = 4 ∧ ∀ (x : ℝ), x = 1/(a-1) + 4/(b-1) → x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3634_363485


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l3634_363433

/-- An ellipse with one focus at (0,1) and eccentricity 1/2 has the standard equation x²/3 + y²/4 = 1 -/
theorem ellipse_standard_equation (x y : ℝ) : 
  let e : ℝ := 1/2
  let f : ℝ × ℝ := (0, 1)
  x^2/3 + y^2/4 = 1 ↔ 
    ∃ (a b c : ℝ), 
      a > 0 ∧ b > 0 ∧
      c = 1 ∧
      e = c/a ∧
      a^2 = b^2 + c^2 ∧
      x^2/a^2 + y^2/b^2 = 1 :=
by sorry


end NUMINAMATH_CALUDE_ellipse_standard_equation_l3634_363433


namespace NUMINAMATH_CALUDE_sector_area_l3634_363468

/-- The area of a circular sector with central angle 3/4π and radius 4 is 6π. -/
theorem sector_area : 
  let central_angle : Real := 3/4 * Real.pi
  let radius : Real := 4
  let sector_area : Real := 1/2 * central_angle * radius^2
  sector_area = 6 * Real.pi := by sorry

end NUMINAMATH_CALUDE_sector_area_l3634_363468


namespace NUMINAMATH_CALUDE_new_average_age_l3634_363435

theorem new_average_age
  (initial_students : ℕ)
  (initial_average : ℚ)
  (new_student_age : ℕ)
  (h1 : initial_students = 8)
  (h2 : initial_average = 15)
  (h3 : new_student_age = 17) :
  let total_age : ℚ := initial_students * initial_average + new_student_age
  let new_total_students : ℕ := initial_students + 1
  total_age / new_total_students = 137 / 9 :=
by sorry

end NUMINAMATH_CALUDE_new_average_age_l3634_363435


namespace NUMINAMATH_CALUDE_unique_solution_l3634_363483

/-- The number of communications between any n-2 people -/
def communications (n : ℕ) : ℕ := 3^(Nat.succ 0)

/-- The theorem stating that 5 is the only solution -/
theorem unique_solution :
  ∀ n : ℕ,
  (n > 0) →
  (∀ m : ℕ, m = communications n) →
  (∀ i j : Fin n, i ≠ j → (∃! x : ℕ, x ≤ 1)) →
  n = 5 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3634_363483


namespace NUMINAMATH_CALUDE_gain_percent_problem_l3634_363419

/-- Calculate the gain percent given the gain in paise and the cost price in rupees. -/
def gain_percent (gain_paise : ℕ) (cost_price_rupees : ℕ) : ℚ :=
  (gain_paise : ℚ) / (cost_price_rupees * 100 : ℚ) * 100

/-- Theorem stating that the gain percent is 1% when the gain is 70 paise on a cost price of Rs. 70. -/
theorem gain_percent_problem : gain_percent 70 70 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_problem_l3634_363419


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3634_363457

-- Define sets A and B
def A : Set ℝ := {x | 2 * x + 1 < 3}
def B : Set ℝ := {x | -3 < x ∧ x < 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -3 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3634_363457


namespace NUMINAMATH_CALUDE_nested_function_evaluation_l3634_363430

-- Define P and Q functions
def P (x : ℝ) : ℝ := 3 * (x ^ (1/3))
def Q (x : ℝ) : ℝ := x ^ 3

-- State the theorem
theorem nested_function_evaluation :
  P (Q (P (Q (P (Q 4))))) = 108 :=
sorry

end NUMINAMATH_CALUDE_nested_function_evaluation_l3634_363430


namespace NUMINAMATH_CALUDE_value_of_x_l3634_363439

theorem value_of_x (z y x : ℝ) (hz : z = 90) (hy : y = 1/3 * z) (hx : x = 1/2 * y) :
  x = 15 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l3634_363439


namespace NUMINAMATH_CALUDE_complex_number_location_l3634_363429

theorem complex_number_location :
  ∀ (z : ℂ), (2 + I) * z = -I →
  (z.re < 0 ∧ z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l3634_363429


namespace NUMINAMATH_CALUDE_tan_fifteen_ratio_equals_sqrt_three_l3634_363434

theorem tan_fifteen_ratio_equals_sqrt_three :
  (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_ratio_equals_sqrt_three_l3634_363434


namespace NUMINAMATH_CALUDE_calvin_collection_total_l3634_363487

def insect_collection (roaches scorpions : ℕ) : ℕ :=
  let crickets := roaches / 2
  let caterpillars := 2 * scorpions
  roaches + scorpions + crickets + caterpillars

theorem calvin_collection_total :
  insect_collection 12 3 = 27 :=
by sorry

end NUMINAMATH_CALUDE_calvin_collection_total_l3634_363487


namespace NUMINAMATH_CALUDE_fraction_equality_l3634_363428

theorem fraction_equality (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) : 
  (p - r) * (q - s) / ((p - q) * (r - s)) = -3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3634_363428


namespace NUMINAMATH_CALUDE_rod_and_rope_problem_l3634_363438

theorem rod_and_rope_problem (x y : ℝ) : 
  (x = y + 5 ∧ x / 2 = y - 5) ↔ 
  (x - y = 5 ∧ y - x / 2 = 5) := by sorry

end NUMINAMATH_CALUDE_rod_and_rope_problem_l3634_363438


namespace NUMINAMATH_CALUDE_d_eq_l_l3634_363478

/-- The number of partitions of n into distinct summands -/
def d (n : ℕ) : ℕ := sorry

/-- The number of partitions of n into odd summands -/
def l (n : ℕ) : ℕ := sorry

/-- The generating function for d(n) -/
noncomputable def d_gen_fun (x : ℝ) : ℝ := ∑' n, d n * x^n

/-- The generating function for l(n) -/
noncomputable def l_gen_fun (x : ℝ) : ℝ := ∑' n, l n * x^n

/-- The product representation of d_gen_fun -/
noncomputable def d_prod (x : ℝ) : ℝ := ∏' k, (1 + x^k)

/-- The product representation of l_gen_fun -/
noncomputable def l_prod (x : ℝ) : ℝ := ∏' k, (1 - x^(2*k+1))⁻¹

/-- The main theorem: d(n) = l(n) for all n -/
theorem d_eq_l : ∀ n : ℕ, d n = l n := by sorry

/-- d(0) = l(0) = 1 -/
axiom d_zero : d 0 = 1
axiom l_zero : l 0 = 1

/-- The generating functions are equal to their product representations -/
axiom d_gen_fun_eq_prod : d_gen_fun = d_prod
axiom l_gen_fun_eq_prod : l_gen_fun = l_prod

end NUMINAMATH_CALUDE_d_eq_l_l3634_363478


namespace NUMINAMATH_CALUDE_juice_cans_bought_l3634_363495

-- Define the original price of ice cream
def original_ice_cream_price : ℚ := 12

-- Define the discount on ice cream
def ice_cream_discount : ℚ := 2

-- Define the price of juice
def juice_price : ℚ := 2

-- Define the number of cans in a set of juice
def cans_per_set : ℕ := 5

-- Define the total cost
def total_cost : ℚ := 24

-- Define the number of ice cream tubs bought
def ice_cream_tubs : ℕ := 2

-- Theorem to prove
theorem juice_cans_bought : ℕ := by
  -- The proof goes here
  sorry

#check juice_cans_bought

end NUMINAMATH_CALUDE_juice_cans_bought_l3634_363495


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l3634_363459

theorem range_of_a_minus_b (a b : ℝ) (ha : -1 < a ∧ a < 1) (hb : 1 < b ∧ b < 3) : 
  -4 < a - b ∧ a - b < 0 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l3634_363459


namespace NUMINAMATH_CALUDE_beth_coin_sale_l3634_363462

theorem beth_coin_sale (initial_coins : ℕ) (gift_coins : ℕ) : 
  initial_coins = 125 → gift_coins = 35 → 
  (initial_coins + gift_coins) / 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_beth_coin_sale_l3634_363462


namespace NUMINAMATH_CALUDE_rick_ironing_time_l3634_363416

/-- Represents the rate at which Rick irons dress shirts per hour -/
def shirts_per_hour : ℕ := 4

/-- Represents the rate at which Rick irons dress pants per hour -/
def pants_per_hour : ℕ := 3

/-- Represents the number of hours Rick spent ironing dress pants -/
def hours_ironing_pants : ℕ := 5

/-- Represents the total number of pieces of clothing Rick ironed -/
def total_pieces : ℕ := 27

/-- Proves that Rick spent 3 hours ironing dress shirts given the conditions -/
theorem rick_ironing_time :
  ∃ (h : ℕ), h * shirts_per_hour + hours_ironing_pants * pants_per_hour = total_pieces ∧ h = 3 :=
by sorry

end NUMINAMATH_CALUDE_rick_ironing_time_l3634_363416


namespace NUMINAMATH_CALUDE_square_difference_305_295_l3634_363497

theorem square_difference_305_295 : (305 : ℤ)^2 - (295 : ℤ)^2 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_305_295_l3634_363497


namespace NUMINAMATH_CALUDE_remainder_theorem_l3634_363425

theorem remainder_theorem : ∃ q : ℕ, 2^404 + 404 = (2^203 + 2^101 + 1) * q + 403 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3634_363425


namespace NUMINAMATH_CALUDE_city_distance_min_city_distance_l3634_363411

def is_valid_distance (S : ℕ) : Prop :=
  (∀ x : ℕ, x ≤ S → (Nat.gcd x (S - x) = 1 ∨ Nat.gcd x (S - x) = 3 ∨ Nat.gcd x (S - x) = 13)) ∧
  (∃ x : ℕ, x ≤ S ∧ Nat.gcd x (S - x) = 1) ∧
  (∃ x : ℕ, x ≤ S ∧ Nat.gcd x (S - x) = 3) ∧
  (∃ x : ℕ, x ≤ S ∧ Nat.gcd x (S - x) = 13)

theorem city_distance : 
  ∀ S : ℕ, is_valid_distance S → S ≥ 39 :=
by sorry

theorem min_city_distance :
  is_valid_distance 39 :=
by sorry

end NUMINAMATH_CALUDE_city_distance_min_city_distance_l3634_363411


namespace NUMINAMATH_CALUDE_f_derivative_at_pi_over_4_l3634_363432

noncomputable def f (x : ℝ) : ℝ := Real.sin x / (Real.sin x + Real.cos x)

theorem f_derivative_at_pi_over_4 :
  deriv f (π/4) = 1/2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_pi_over_4_l3634_363432


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3634_363469

theorem rationalize_denominator :
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3634_363469


namespace NUMINAMATH_CALUDE_sum_min_max_cubic_quartic_l3634_363494

theorem sum_min_max_cubic_quartic (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 8)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 18) : 
  let f := fun (x y z w : ℝ) => 3 * (x^3 + y^3 + z^3 + w^3) - 2 * (x^4 + y^4 + z^4 + w^4)
  ∃ (m M : ℝ), (∀ (x y z w : ℝ), (x + y + z + w = 8 ∧ x^2 + y^2 + z^2 + w^2 = 18) → 
    m ≤ f x y z w ∧ f x y z w ≤ M) ∧ m + M = 29 :=
sorry

end NUMINAMATH_CALUDE_sum_min_max_cubic_quartic_l3634_363494


namespace NUMINAMATH_CALUDE_existence_of_critical_point_and_positive_function_l3634_363484

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp (m * x) - Real.log x - 2

theorem existence_of_critical_point_and_positive_function :
  (∃ t : ℝ, t ∈ Set.Ioo (1/2) 1 ∧ (deriv (f 1)) t = 0 ∧
    ∀ t' : ℝ, t' ∈ Set.Ioo (1/2) 1 ∧ (deriv (f 1)) t' = 0 → t' = t) ∧
  (∃ m : ℝ, m ∈ Set.Ioo 0 1 ∧ ∀ x : ℝ, x > 0 → f m x > 0) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_critical_point_and_positive_function_l3634_363484


namespace NUMINAMATH_CALUDE_red_surface_fraction_is_one_l3634_363498

/-- Represents a cube with its edge length and number of smaller cubes -/
structure Cube where
  edge_length : ℕ
  num_small_cubes : ℕ

/-- Represents the composition of the cube in terms of colored smaller cubes -/
structure CubeComposition where
  total_cubes : Cube
  red_cubes : ℕ
  blue_cubes : ℕ

/-- The fraction of the surface area of the larger cube that is red -/
def red_surface_fraction (c : CubeComposition) : ℚ :=
  sorry

/-- The theorem stating the fraction of red surface area -/
theorem red_surface_fraction_is_one (c : CubeComposition) 
  (h1 : c.total_cubes.edge_length = 4)
  (h2 : c.total_cubes.num_small_cubes = 64)
  (h3 : c.red_cubes = 40)
  (h4 : c.blue_cubes = 24)
  (h5 : c.red_cubes + c.blue_cubes = c.total_cubes.num_small_cubes) :
  red_surface_fraction c = 1 := by
  sorry

end NUMINAMATH_CALUDE_red_surface_fraction_is_one_l3634_363498


namespace NUMINAMATH_CALUDE_distance_calculation_l3634_363473

/-- The distance between Maxwell's and Brad's homes -/
def distance_between_homes : ℝ := 54

/-- Maxwell's walking speed in km/h -/
def maxwell_speed : ℝ := 4

/-- Brad's running speed in km/h -/
def brad_speed : ℝ := 6

/-- The time Maxwell walks before meeting Brad, in hours -/
def maxwell_time : ℝ := 6

/-- The time Brad runs before meeting Maxwell, in hours -/
def brad_time : ℝ := maxwell_time - 1

theorem distance_calculation :
  distance_between_homes = maxwell_speed * maxwell_time + brad_speed * brad_time :=
by sorry

end NUMINAMATH_CALUDE_distance_calculation_l3634_363473


namespace NUMINAMATH_CALUDE_divisibility_of_binomial_coefficient_l3634_363489

theorem divisibility_of_binomial_coefficient (m n d : ℕ) : 
  0 < m → 0 < n → m ≤ n → d = Nat.gcd m n → 
  ∃ k : ℤ, (d : ℤ) * (Nat.choose n m : ℤ) = k * (n : ℤ) :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_binomial_coefficient_l3634_363489


namespace NUMINAMATH_CALUDE_tan_alpha_3_implies_fraction_eq_5_6_l3634_363474

theorem tan_alpha_3_implies_fraction_eq_5_6 (α : ℝ) (h : Real.tan α = 3) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 3 * Real.cos α) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_3_implies_fraction_eq_5_6_l3634_363474


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l3634_363456

theorem min_sum_of_squares (x y : ℝ) (h : (x + 4) * (y - 4) = 0) :
  ∃ (m : ℝ), m = 16 ∧ ∀ (a b : ℝ), (a + 4) * (b - 4) = 0 → a^2 + b^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l3634_363456


namespace NUMINAMATH_CALUDE_probability_sum_16_three_dice_rolls_l3634_363476

theorem probability_sum_16_three_dice_rolls :
  let die_faces : ℕ := 6
  let total_outcomes : ℕ := die_faces ^ 3
  let favorable_outcomes : ℕ := 6
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 36 :=
by sorry

end NUMINAMATH_CALUDE_probability_sum_16_three_dice_rolls_l3634_363476


namespace NUMINAMATH_CALUDE_range_of_a_l3634_363423

def prop_p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x

def prop_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 4*x + a = 0

theorem range_of_a (a : ℝ) (hp : prop_p a) (hq : prop_q a) :
  a ∈ Set.Icc (Real.exp 1) 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3634_363423


namespace NUMINAMATH_CALUDE_statistical_relationships_properties_l3634_363443

-- Define the basic concepts
def FunctionalRelationship : Type := Unit
def DeterministicRelationship : Type := Unit
def Correlation : Type := Unit
def NonDeterministicRelationship : Type := Unit
def RegressionAnalysis : Type := Unit
def StatisticalAnalysisMethod : Type := Unit
def TwoVariables : Type := Unit

-- Define the properties
def isDeterministic (r : FunctionalRelationship) : Prop := sorry
def isNonDeterministic (c : Correlation) : Prop := sorry
def isUsedFor (m : StatisticalAnalysisMethod) (v : TwoVariables) (c : Correlation) : Prop := sorry

-- Theorem to prove
theorem statistical_relationships_properties :
  (∀ (r : FunctionalRelationship), isDeterministic r) ∧
  (∀ (c : Correlation), isNonDeterministic c) ∧
  (∃ (m : RegressionAnalysis) (v : TwoVariables) (c : Correlation), 
    isUsedFor m v c) :=
by sorry

end NUMINAMATH_CALUDE_statistical_relationships_properties_l3634_363443


namespace NUMINAMATH_CALUDE_jerrys_action_figures_l3634_363436

theorem jerrys_action_figures (initial : ℕ) : 
  (initial + 2 - 7 = 10) → initial = 15 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_action_figures_l3634_363436


namespace NUMINAMATH_CALUDE_specific_prism_volume_l3634_363440

/-- A rectangular prism with given edge length sum and proportions -/
structure RectangularPrism where
  edgeSum : ℝ
  width : ℝ
  height : ℝ
  length : ℝ
  edgeSum_eq : edgeSum = 4 * (width + height + length)
  height_prop : height = 2 * width
  length_prop : length = 4 * width

/-- The volume of a rectangular prism -/
def volume (p : RectangularPrism) : ℝ := p.width * p.height * p.length

/-- Theorem: The volume of the specific rectangular prism is 85184/343 -/
theorem specific_prism_volume :
  ∃ (p : RectangularPrism), p.edgeSum = 88 ∧ volume p = 85184 / 343 := by
  sorry

end NUMINAMATH_CALUDE_specific_prism_volume_l3634_363440


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l3634_363450

-- Define the function
def f (x : ℝ) : ℝ := 4 * x^3 - 5 * x + 6

-- State the theorem
theorem root_exists_in_interval :
  ∃ c ∈ Set.Ioo (-2 : ℝ) (-1 : ℝ), f c = 0 :=
by
  have h1 : Continuous f := sorry
  have h2 : f (-2) < 0 := sorry
  have h3 : f (-1) > 0 := sorry
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_root_exists_in_interval_l3634_363450


namespace NUMINAMATH_CALUDE_f_properties_l3634_363414

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

def monotonic_intervals (a : ℝ) : Prop :=
  (a ≤ 0 → ∀ x y, 0 < x ∧ x < y → f a x < f a y) ∧
  (a > 0 → (∀ x y, 0 < x ∧ x < y ∧ y < 1/a → f a x < f a y) ∧
           (∀ x y, 1/a < x ∧ x < y → f a y < f a x))

def minimum_value (a : ℝ) : ℝ :=
  if a ≥ 1 then f a 2
  else if 0 < a ∧ a < 1/2 then f a 1
  else min (f a 1) (f a 2)

theorem f_properties (a : ℝ) :
  monotonic_intervals a ∧
  (a > 0 → ∀ x, x ∈ Set.Icc 1 2 → f a x ≥ minimum_value a) :=
sorry

end

end NUMINAMATH_CALUDE_f_properties_l3634_363414


namespace NUMINAMATH_CALUDE_constant_function_invariant_l3634_363424

-- Define the function g
def g : ℝ → ℝ := λ x => -3

-- State the theorem
theorem constant_function_invariant (x : ℝ) : g (3 * x - 1) = -3 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_invariant_l3634_363424


namespace NUMINAMATH_CALUDE_system_solution_proof_single_equation_solution_proof_l3634_363492

-- System of equations
theorem system_solution_proof (x y : ℝ) : 
  x = 1 ∧ y = 2 → 2*x + 3*y = 8 ∧ 3*x - 5*y = -7 := by sorry

-- Single equation
theorem single_equation_solution_proof (x : ℝ) :
  x = -1 → (x-2)/(x+2) - 12/(x^2-4) = 1 := by sorry

end NUMINAMATH_CALUDE_system_solution_proof_single_equation_solution_proof_l3634_363492


namespace NUMINAMATH_CALUDE_common_volume_for_ratios_l3634_363409

/-- The volume of the common part of two identical triangular pyramids -/
noncomputable def common_volume (V : ℝ) (r : ℝ) : ℝ := sorry

/-- Theorem stating the volume of the common part for different ratios -/
theorem common_volume_for_ratios (V : ℝ) (V_pos : V > 0) :
  (common_volume V (1/2) = 2/3 * V) ∧
  (common_volume V (3/4) = 1/2 * V) ∧
  (common_volume V (2/3) = 110/243 * V) ∧
  (common_volume V (4/5) = 12/25 * V) := by sorry

end NUMINAMATH_CALUDE_common_volume_for_ratios_l3634_363409


namespace NUMINAMATH_CALUDE_equation_solution_l3634_363481

theorem equation_solution (x y : ℚ) 
  (eq1 : 3 * x + y = 6) 
  (eq2 : x + 3 * y = 8) : 
  9 * x^2 + 15 * x * y + 9 * y^2 = 1629 / 16 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3634_363481


namespace NUMINAMATH_CALUDE_crazy_silly_school_movies_l3634_363427

/-- The number of movies in the 'crazy silly school' series -/
def num_movies : ℕ := 9

/-- The number of different books in the series -/
def num_books : ℕ := 10

/-- The number of books read -/
def books_read : ℕ := 14

theorem crazy_silly_school_movies :
  (books_read = num_movies + 5) →
  (books_read ≤ num_books) →
  (num_movies = 9) := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_movies_l3634_363427


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3634_363412

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the focal length
def focal_length (c : ℝ) : Prop := c = 2

-- Define the eccentricity
def eccentricity (e a c : ℝ) : Prop := e = c / a

theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : focal_length 2) 
  (hp : hyperbola a b 2 3) : 
  ∃ e, eccentricity e a 2 ∧ e = 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3634_363412


namespace NUMINAMATH_CALUDE_cassidy_poster_count_l3634_363431

/-- The number of posters Cassidy had two years ago -/
def posters_two_years_ago : ℕ := 14

/-- The number of posters Cassidy will add this summer -/
def posters_to_add : ℕ := 6

/-- Cassidy's current number of posters -/
def current_posters : ℕ := 22

theorem cassidy_poster_count :
  current_posters + posters_to_add = 2 * posters_two_years_ago :=
by sorry

end NUMINAMATH_CALUDE_cassidy_poster_count_l3634_363431


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l3634_363477

theorem sum_of_reciprocals_squared (a b c d : ℝ) : 
  a = Real.sqrt 2 + 2 * Real.sqrt 3 + Real.sqrt 6 →
  b = -Real.sqrt 2 + 2 * Real.sqrt 3 + Real.sqrt 6 →
  c = Real.sqrt 2 - 2 * Real.sqrt 3 + Real.sqrt 6 →
  d = -Real.sqrt 2 - 2 * Real.sqrt 3 + Real.sqrt 6 →
  (1/a + 1/b + 1/c + 1/d)^2 = 3/50 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l3634_363477


namespace NUMINAMATH_CALUDE_interest_rate_proof_l3634_363402

/-- Given a principal amount, time, and the difference between compound and simple interest,
    prove that the interest rate is 25%. -/
theorem interest_rate_proof (P t : ℝ) (diff : ℝ) : 
  P = 3600 → t = 2 → diff = 225 →
  ∃ r : ℝ, r = 25 ∧ 
    P * ((1 + r / 100) ^ t - 1) - (P * r * t / 100) = diff :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l3634_363402


namespace NUMINAMATH_CALUDE_score_sum_theorem_l3634_363426

def total_score (keith larry danny emma fiona : ℝ) : ℝ :=
  keith + larry + danny + emma + fiona

theorem score_sum_theorem (keith larry danny emma fiona : ℝ) 
  (h1 : keith = 3.5)
  (h2 : larry = 3.2 * keith)
  (h3 : danny = larry + 5.7)
  (h4 : emma = 2 * danny - 1.2)
  (h5 : fiona = (keith + larry + danny + emma) / 4) :
  total_score keith larry danny emma fiona = 80.25 := by
  sorry

end NUMINAMATH_CALUDE_score_sum_theorem_l3634_363426


namespace NUMINAMATH_CALUDE_inequality_range_l3634_363454

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 8*x + 20) / (m*x^2 - m*x - 1) < 0) ↔ 
  (-4 < m ∧ m ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l3634_363454


namespace NUMINAMATH_CALUDE_nonnegative_solutions_system_l3634_363413

theorem nonnegative_solutions_system (x y z : ℝ) :
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 →
  Real.sqrt (x + y) + Real.sqrt z = 7 →
  Real.sqrt (x + z) + Real.sqrt y = 7 →
  Real.sqrt (y + z) + Real.sqrt x = 5 →
  ((x = 1 ∧ y = 4 ∧ z = 4) ∨ (x = 1 ∧ y = 9 ∧ z = 9)) :=
by sorry

end NUMINAMATH_CALUDE_nonnegative_solutions_system_l3634_363413


namespace NUMINAMATH_CALUDE_sequence_sum_l3634_363422

theorem sequence_sum : 
  let S := 1 / (4 - Real.sqrt 15) - 1 / (Real.sqrt 15 - Real.sqrt 14) + 
           1 / (Real.sqrt 14 - Real.sqrt 13) - 1 / (Real.sqrt 13 - 3)
  S = 13 / 4 + (3 / 4) * Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l3634_363422


namespace NUMINAMATH_CALUDE_liter_milliliter_comparison_l3634_363448

theorem liter_milliliter_comparison : ¬(1000 < 9000 / 1000) := by
  sorry

end NUMINAMATH_CALUDE_liter_milliliter_comparison_l3634_363448


namespace NUMINAMATH_CALUDE_min_value_expression_l3634_363442

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (p + q + r) * (1 / (p + q) + 1 / (p + r) + 1 / (q + r) + 1 / (p + q + r)) ≥ 5 ∧
  (∃ t : ℝ, t > 0 ∧ (t + t + t) * (1 / (t + t) + 1 / (t + t) + 1 / (t + t) + 1 / (t + t + t)) = 5) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3634_363442


namespace NUMINAMATH_CALUDE_expression_evaluation_l3634_363405

theorem expression_evaluation : -4 / (4 / 9) * (9 / 4) = -81 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3634_363405


namespace NUMINAMATH_CALUDE_square_of_negative_sqrt_two_equals_two_l3634_363475

theorem square_of_negative_sqrt_two_equals_two :
  ((-Real.sqrt 2) ^ 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_sqrt_two_equals_two_l3634_363475


namespace NUMINAMATH_CALUDE_tyson_race_time_l3634_363408

/-- Calculates the total time Tyson spent in races given his swimming speeds and race details. -/
theorem tyson_race_time (lake_speed ocean_speed : ℝ) (total_races : ℕ) (race_distance : ℝ) : 
  lake_speed = 3 →
  ocean_speed = 2.5 →
  total_races = 10 →
  race_distance = 3 →
  (total_races / 2 : ℝ) * race_distance / lake_speed + 
  (total_races / 2 : ℝ) * race_distance / ocean_speed = 11 := by
  sorry


end NUMINAMATH_CALUDE_tyson_race_time_l3634_363408


namespace NUMINAMATH_CALUDE_midpoint_distance_to_origin_l3634_363437

theorem midpoint_distance_to_origin : 
  let p1 : ℝ × ℝ := (-6, 8)
  let p2 : ℝ × ℝ := (6, -8)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  (midpoint.1^2 + midpoint.2^2).sqrt = 0 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_distance_to_origin_l3634_363437


namespace NUMINAMATH_CALUDE_circle_region_area_l3634_363467

/-- Given a circle with radius 36 and two chords of length 90 intersecting at a point 12 units from the center,
    the area of one of the regions formed can be expressed as 216π, which is equivalent to aπ - b√c
    where a + b + c = 216 and a, b, c are positive integers with c not divisible by the square of any prime. -/
theorem circle_region_area (r : ℝ) (chord_length : ℝ) (intersection_distance : ℝ)
  (h_radius : r = 36)
  (h_chord : chord_length = 90)
  (h_intersection : intersection_distance = 12) :
  ∃ (a b c : ℕ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (∀ (p : ℕ), Prime p → c % (p^2) ≠ 0) ∧
    (a + b + c = 216) ∧
    (Real.pi * (a : ℝ) - (b : ℝ) * Real.sqrt (c : ℝ) = 216 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_circle_region_area_l3634_363467


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_x_squared_minus_x_geq_zero_l3634_363404

theorem negation_of_forall_positive_x_squared_minus_x_geq_zero :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≥ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_x_squared_minus_x_geq_zero_l3634_363404


namespace NUMINAMATH_CALUDE_half_month_days_l3634_363451

/-- Prove that given a 30-day month with specific mean profits, each half of the month contains 15 days -/
theorem half_month_days (total_days : ℕ) (mean_profit : ℚ) (first_half_mean : ℚ) (second_half_mean : ℚ) :
  total_days = 30 ∧ 
  mean_profit = 350 ∧ 
  first_half_mean = 275 ∧ 
  second_half_mean = 425 →
  ∃ (half_days : ℕ), half_days = 15 ∧ total_days = 2 * half_days :=
by sorry

end NUMINAMATH_CALUDE_half_month_days_l3634_363451


namespace NUMINAMATH_CALUDE_percentage_commutation_l3634_363453

theorem percentage_commutation (n : ℝ) (h : 0.3 * (0.4 * n) = 36) :
  0.4 * (0.3 * n) = 0.3 * (0.4 * n) := by
  sorry

end NUMINAMATH_CALUDE_percentage_commutation_l3634_363453
