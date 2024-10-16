import Mathlib

namespace NUMINAMATH_CALUDE_billy_sodas_l944_94414

/-- The number of sodas in Billy's pack -/
def sodas_in_pack (sisters : ℕ) (brothers : ℕ) (sodas_per_sibling : ℕ) : ℕ :=
  (sisters + brothers) * sodas_per_sibling

/-- Theorem: The number of sodas in Billy's pack is 12 -/
theorem billy_sodas :
  ∀ (sisters brothers sodas_per_sibling : ℕ),
    brothers = 2 * sisters →
    sisters = 2 →
    sodas_per_sibling = 2 →
    sodas_in_pack sisters brothers sodas_per_sibling = 12 := by
  sorry

end NUMINAMATH_CALUDE_billy_sodas_l944_94414


namespace NUMINAMATH_CALUDE_triangle_area_l944_94434

theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b = 6 → 
  a = 2 * c → 
  B = π / 3 → 
  (1 / 2) * a * c * Real.sin B = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l944_94434


namespace NUMINAMATH_CALUDE_bus_capacity_is_90_l944_94473

/-- The number of people that can sit in a bus with given seat arrangements -/
def bus_capacity (left_seats : ℕ) (right_seat_difference : ℕ) (people_per_seat : ℕ) (back_seat_capacity : ℕ) : ℕ :=
  let right_seats := left_seats - right_seat_difference
  let total_regular_seats := left_seats + right_seats
  let total_regular_capacity := total_regular_seats * people_per_seat
  total_regular_capacity + back_seat_capacity

/-- Theorem stating that the bus capacity is 90 given the specific conditions -/
theorem bus_capacity_is_90 : 
  bus_capacity 15 3 3 9 = 90 := by
  sorry

end NUMINAMATH_CALUDE_bus_capacity_is_90_l944_94473


namespace NUMINAMATH_CALUDE_problem_solution_l944_94459

theorem problem_solution : (2021^2 - 2021) / 2021 = 2020 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l944_94459


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l944_94477

theorem sine_cosine_inequality (x : ℝ) (n : ℕ) :
  (Real.sin (2 * x))^n + ((Real.sin x)^n - (Real.cos x)^n)^2 ≤ 1 := by sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l944_94477


namespace NUMINAMATH_CALUDE_solve_linear_equations_l944_94484

theorem solve_linear_equations :
  (∃ y : ℚ, 8 * y - 4 * (3 * y + 2) = 6 ∧ y = -7/2) ∧
  (∃ x : ℚ, 2 - (x + 2) / 3 = x - (x - 1) / 6 ∧ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_solve_linear_equations_l944_94484


namespace NUMINAMATH_CALUDE_otimes_four_eight_l944_94485

-- Define the operation ⊗
def otimes (a b : ℚ) : ℚ := a / b + b / a

-- Theorem statement
theorem otimes_four_eight : otimes 4 8 = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_otimes_four_eight_l944_94485


namespace NUMINAMATH_CALUDE_scenic_spot_assignment_l944_94418

/-- The number of scenic spots -/
def num_spots : ℕ := 3

/-- The number of people -/
def num_people : ℕ := 4

/-- The total number of possible assignments without restrictions -/
def total_assignments : ℕ := num_spots ^ num_people

/-- The number of assignments where A and B are in the same spot -/
def restricted_assignments : ℕ := num_spots * (num_spots ^ (num_people - 2))

/-- The number of valid assignments where A and B are not in the same spot -/
def valid_assignments : ℕ := total_assignments - restricted_assignments

theorem scenic_spot_assignment :
  valid_assignments = 54 := by sorry

end NUMINAMATH_CALUDE_scenic_spot_assignment_l944_94418


namespace NUMINAMATH_CALUDE_divisibility_property_l944_94444

theorem divisibility_property (a b c : ℤ) (h : 13 ∣ (a + b + c)) :
  13 ∣ (a^2007 + b^2007 + c^2007 + 2 * 2007 * a * b * c) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_property_l944_94444


namespace NUMINAMATH_CALUDE_two_even_dice_probability_l944_94443

/-- The probability of rolling an even number on a fair 8-sided die -/
def prob_even : ℚ := 1/2

/-- The number of ways to choose 2 dice out of 3 -/
def ways_to_choose : ℕ := 3

/-- The probability of exactly two dice showing even numbers when rolling three fair 8-sided dice -/
def prob_two_even : ℚ := ways_to_choose * (prob_even^2 * (1 - prob_even))

theorem two_even_dice_probability : prob_two_even = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_two_even_dice_probability_l944_94443


namespace NUMINAMATH_CALUDE_pizza_slices_per_pizza_l944_94424

theorem pizza_slices_per_pizza (total_pizzas : ℕ) (total_slices : ℕ) 
  (h1 : total_pizzas = 2) 
  (h2 : total_slices = 16) : 
  total_slices / total_pizzas = 8 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_per_pizza_l944_94424


namespace NUMINAMATH_CALUDE_partnership_profit_calculation_l944_94432

/-- Calculates the total profit of a partnership given investments and one partner's profit share -/
def calculate_total_profit (investment_a investment_b investment_c c_profit : ℚ) : ℚ :=
  let ratio_sum := (investment_a / 1000) + (investment_b / 1000) + (investment_c / 1000)
  let c_ratio := investment_c / 1000
  (c_profit * ratio_sum) / c_ratio

/-- Theorem stating that given the specified investments and C's profit share, the total profit is approximately 97777.78 -/
theorem partnership_profit_calculation :
  let investment_a : ℚ := 5000
  let investment_b : ℚ := 8000
  let investment_c : ℚ := 9000
  let c_profit : ℚ := 36000
  let total_profit := calculate_total_profit investment_a investment_b investment_c c_profit
  ∃ ε > 0, |total_profit - 97777.78| < ε :=
sorry

end NUMINAMATH_CALUDE_partnership_profit_calculation_l944_94432


namespace NUMINAMATH_CALUDE_ellipse_max_b_l944_94419

/-- Given an ellipse x^2 + y^2/b^2 = 1 where 0 < b < 1, with foci F1 and F2 at distance 2c apart,
    if there exists a point P on the ellipse such that the distance from P to the line x = 1/c
    is the arithmetic mean of |PF1| and |PF2|, then the maximum value of b is √3/2. -/
theorem ellipse_max_b (b c : ℝ) (h1 : 0 < b) (h2 : b < 1) :
  (∃ (x y : ℝ), x^2 + y^2/b^2 = 1 ∧
    ∃ (PF1 PF2 : ℝ), |x - 1/c| = (PF1 + PF2)/2 ∧
      ∃ (c_foci : ℝ), c_foci = 2*c) →
  b ≤ Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_max_b_l944_94419


namespace NUMINAMATH_CALUDE_constant_term_is_70_l944_94408

/-- 
Given a natural number n, this function represents the coefficient 
of the r-th term in the expansion of (x + 1/x)^(2n)
-/
def binomialCoeff (n : ℕ) (r : ℕ) : ℕ := Nat.choose (2 * n) r

/-- 
This theorem states that if the coefficients of the fourth and sixth terms 
in the expansion of (x + 1/x)^(2n) are equal, then the constant term 
in the expansion is 70
-/
theorem constant_term_is_70 (n : ℕ) 
  (h : binomialCoeff n 3 = binomialCoeff n 5) : 
  binomialCoeff n 4 = 70 := by
  sorry


end NUMINAMATH_CALUDE_constant_term_is_70_l944_94408


namespace NUMINAMATH_CALUDE_division_remainder_problem_l944_94436

theorem division_remainder_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) 
  (h1 : dividend = 127)
  (h2 : divisor = 14)
  (h3 : quotient = 9)
  (h4 : dividend = divisor * quotient + (dividend % divisor)) :
  dividend % divisor = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l944_94436


namespace NUMINAMATH_CALUDE_pigeonhole_disks_l944_94435

/-- The number of distinct labels -/
def n : ℕ := 50

/-- The function that maps a label to the number of disks with that label -/
def f (i : ℕ) : ℕ := i

/-- The total number of disks -/
def total_disks : ℕ := n * (n + 1) / 2

/-- The minimum number of disks to guarantee at least 10 of the same label -/
def min_disks : ℕ := 415

theorem pigeonhole_disks :
  ∀ (S : Finset ℕ), S.card = min_disks →
  ∃ (i : ℕ), i ∈ Finset.range n ∧ (S.filter (λ x => x = i)).card ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_pigeonhole_disks_l944_94435


namespace NUMINAMATH_CALUDE_train_platform_problem_l944_94449

/-- Represents the speed of the train in km/hr -/
def train_speed : ℝ := 72

/-- Represents the time taken to cross the platform in minutes -/
def crossing_time : ℝ := 1

/-- Calculates the length of the train in meters -/
def train_length : ℝ := 600

theorem train_platform_problem :
  ∀ (train_length platform_length : ℝ),
  train_length = platform_length →
  train_length = train_speed * (1000 / 3600) * (crossing_time * 60) / 2 →
  train_length = 600 := by
  sorry

end NUMINAMATH_CALUDE_train_platform_problem_l944_94449


namespace NUMINAMATH_CALUDE_sum_of_digits_five_pow_eq_two_pow_l944_94492

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The only natural number n for which the sum of digits of 5^n equals 2^n is 3 -/
theorem sum_of_digits_five_pow_eq_two_pow :
  ∃! n : ℕ, sum_of_digits (5^n) = 2^n ∧ n = 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_five_pow_eq_two_pow_l944_94492


namespace NUMINAMATH_CALUDE_expression_simplification_l944_94463

theorem expression_simplification (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) (h3 : x ≠ -1) : 
  (2*x + 4) / (x^2 - 1) / ((x + 2) / (x^2 - 2*x + 1)) - 2*x / (x + 1) = -2 / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l944_94463


namespace NUMINAMATH_CALUDE_corn_height_after_three_weeks_l944_94496

def corn_growth (first_week_growth : ℝ) (second_week_multiplier : ℝ) (third_week_multiplier : ℝ) : ℝ :=
  let second_week_growth := first_week_growth * second_week_multiplier
  let third_week_growth := second_week_growth * third_week_multiplier
  first_week_growth + second_week_growth + third_week_growth

theorem corn_height_after_three_weeks :
  corn_growth 2 2 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_corn_height_after_three_weeks_l944_94496


namespace NUMINAMATH_CALUDE_smallest_label_same_as_1993_solution_l944_94467

/-- The number of points on the circle -/
def num_points : ℕ := 2000

/-- The highest label used in the problem -/
def max_label : ℕ := 1993

/-- Function to calculate the position of a label -/
def label_position (n : ℕ) : ℕ :=
  (n * (n + 1) / 2 - 1) % num_points

/-- Theorem stating that 118 is the smallest positive integer that labels the same point as 1993 -/
theorem smallest_label_same_as_1993 :
  ∀ k : ℕ, 0 < k → k < 118 → label_position k ≠ label_position max_label ∧
  label_position 118 = label_position max_label := by
  sorry

/-- Main theorem proving the solution -/
theorem solution : ∃ n : ℕ, n > 0 ∧ 
  (∀ k : ℕ, 0 < k → k < n → label_position k ≠ label_position max_label) ∧
  label_position n = label_position max_label ∧ n = 118 := by
  sorry

end NUMINAMATH_CALUDE_smallest_label_same_as_1993_solution_l944_94467


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_eq_sum_l944_94497

theorem sqrt_sum_squares_eq_sum (a b : ℝ) :
  Real.sqrt (a^2 + b^2) = a + b ↔ a * b = 0 ∧ a + b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_eq_sum_l944_94497


namespace NUMINAMATH_CALUDE_longest_common_length_l944_94423

theorem longest_common_length (wood_lengths : List Nat) : 
  wood_lengths = [90, 72, 120, 150, 108] → 
  Nat.gcd 90 (Nat.gcd 72 (Nat.gcd 120 (Nat.gcd 150 108))) = 6 := by
  sorry

end NUMINAMATH_CALUDE_longest_common_length_l944_94423


namespace NUMINAMATH_CALUDE_stating_failed_both_percentage_l944_94495

/-- Represents the percentage of students in various categories -/
structure ExamResults where
  failed_hindi : ℝ
  failed_english : ℝ
  passed_both : ℝ

/-- 
Calculates the percentage of students who failed in both Hindi and English
given the exam results.
-/
def percentage_failed_both (results : ExamResults) : ℝ :=
  results.failed_hindi + results.failed_english - (100 - results.passed_both)

/-- 
Theorem stating that given the specific exam results, 
the percentage of students who failed in both subjects is 27%.
-/
theorem failed_both_percentage 
  (results : ExamResults)
  (h1 : results.failed_hindi = 25)
  (h2 : results.failed_english = 48)
  (h3 : results.passed_both = 54) :
  percentage_failed_both results = 27 := by
  sorry

#eval percentage_failed_both ⟨25, 48, 54⟩

end NUMINAMATH_CALUDE_stating_failed_both_percentage_l944_94495


namespace NUMINAMATH_CALUDE_calculator_game_sum_l944_94411

def iterate_calculator (n : ℕ) (initial : ℤ) (f : ℤ → ℤ) : ℤ :=
  match n with
  | 0 => initial
  | m + 1 => f (iterate_calculator m initial f)

theorem calculator_game_sum (n : ℕ) : 
  iterate_calculator n 1 (λ x => x^3) + 
  iterate_calculator n 0 (λ x => x^2) + 
  iterate_calculator n (-1) (λ x => -x) = 0 :=
by sorry

end NUMINAMATH_CALUDE_calculator_game_sum_l944_94411


namespace NUMINAMATH_CALUDE_total_amount_distributed_l944_94464

/-- Given an equal distribution of money among 22 persons, where each person receives Rs 1950,
    prove that the total amount distributed is Rs 42900. -/
theorem total_amount_distributed (num_persons : ℕ) (amount_per_person : ℕ) 
  (h1 : num_persons = 22)
  (h2 : amount_per_person = 1950) : 
  num_persons * amount_per_person = 42900 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_distributed_l944_94464


namespace NUMINAMATH_CALUDE_book_profit_rate_l944_94428

/-- Calculates the overall rate of profit for three books --/
def overall_rate_of_profit (cost_a cost_b cost_c sell_a sell_b sell_c : ℚ) : ℚ :=
  let total_cost := cost_a + cost_b + cost_c
  let total_sell := sell_a + sell_b + sell_c
  (total_sell - total_cost) / total_cost * 100

/-- Theorem: The overall rate of profit for the given book prices is approximately 42.86% --/
theorem book_profit_rate :
  let cost_a : ℚ := 50
  let cost_b : ℚ := 120
  let cost_c : ℚ := 75
  let sell_a : ℚ := 90
  let sell_b : ℚ := 150
  let sell_c : ℚ := 110
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/10000 ∧ 
  |overall_rate_of_profit cost_a cost_b cost_c sell_a sell_b sell_c - 42.86| < ε :=
by sorry

end NUMINAMATH_CALUDE_book_profit_rate_l944_94428


namespace NUMINAMATH_CALUDE_true_proposition_l944_94474

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

/-- Proposition p: There exists a φ ∈ ℝ such that f(x) = sin(x + φ) is an even function -/
def p : Prop := ∃ φ : ℝ, IsEven (fun x ↦ Real.sin (x + φ))

/-- Proposition q: For all x ∈ ℝ, cos(2x) + 4sin(x) - 3 < 0 -/
def q : Prop := ∀ x : ℝ, Real.cos (2 * x) + 4 * Real.sin x - 3 < 0

/-- The true proposition is p ∨ (¬q) -/
theorem true_proposition : p ∨ ¬q := by
  sorry

end NUMINAMATH_CALUDE_true_proposition_l944_94474


namespace NUMINAMATH_CALUDE_train_crossing_time_l944_94478

/-- Proves that a train with length 1050 meters, traveling at 126 km/hr, 
    takes 60 seconds to cross a platform of equal length. -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (h1 : train_length = 1050) 
  (h2 : train_speed_kmh = 126) : 
  (2 * train_length) / (train_speed_kmh * 1000 / 3600) = 60 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l944_94478


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l944_94466

-- Define the function f implicitly
def f : ℝ → ℝ := sorry

-- Define the solution set of f(x) > 0
def solution_set_f (x : ℝ) : Prop := x < -1 ∨ x > 1/2

-- Define the solution set of f(10^x) > 0
def solution_set_f_exp (x : ℝ) : Prop := x > -Real.log 2 / Real.log 10

-- Theorem statement
theorem solution_set_equivalence :
  (∀ x, f x > 0 ↔ solution_set_f x) →
  (∀ x, f (10^x) > 0 ↔ solution_set_f_exp x) :=
sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l944_94466


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l944_94469

theorem quadratic_roots_condition (p q r : ℝ) : 
  (p^4 * (q - r)^2 + 2 * p^2 * (q + r) + 1 = p^4) ↔ 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    (x₁^2 + p*x₁ + q = 0) ∧ 
    (x₂^2 + p*x₂ + q = 0) ∧ 
    (y₁^2 - p*y₁ + r = 0) ∧ 
    (y₂^2 - p*y₂ + r = 0) ∧ 
    (x₁*y₁ - x₂*y₂ = 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l944_94469


namespace NUMINAMATH_CALUDE_jared_popcorn_theorem_l944_94422

/-- The number of pieces of popcorn in a serving -/
def popcorn_per_serving : ℕ := 30

/-- The number of pieces of popcorn each of Jared's friends can eat -/
def friend_popcorn_consumption : ℕ := 60

/-- The number of Jared's friends -/
def number_of_friends : ℕ := 3

/-- The number of servings Jared should order -/
def servings_ordered : ℕ := 9

/-- The number of pieces of popcorn Jared can eat -/
def jared_popcorn_consumption : ℕ := 
  servings_ordered * popcorn_per_serving - number_of_friends * friend_popcorn_consumption

theorem jared_popcorn_theorem : jared_popcorn_consumption = 90 := by
  sorry

end NUMINAMATH_CALUDE_jared_popcorn_theorem_l944_94422


namespace NUMINAMATH_CALUDE_same_parity_min_max_l944_94440

/-- A set with elements related to positioning in a function or polynomial -/
def A_P : Set ℤ := sorry

/-- The smallest element of A_P -/
def smallest (A : Set ℤ) : ℤ := sorry

/-- The largest element of A_P -/
def largest (A : Set ℤ) : ℤ := sorry

/-- A function to determine if a number is even -/
def isEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem same_parity_min_max : isEven (smallest A_P) ↔ isEven (largest A_P) := by sorry

end NUMINAMATH_CALUDE_same_parity_min_max_l944_94440


namespace NUMINAMATH_CALUDE_pirate_total_distance_l944_94401

def island1_distances : List ℝ := [10, 15, 20]
def island1_increase : ℝ := 1.1
def island2_distance : ℝ := 40
def island2_increase : ℝ := 1.15
def island3_morning : ℝ := 25
def island3_afternoon : ℝ := 20
def island3_days : ℕ := 2
def island3_increase : ℝ := 1.2
def island4_distance : ℝ := 35
def island4_increase : ℝ := 1.25

theorem pirate_total_distance :
  let island1_total := (island1_distances.map (· * island1_increase)).sum
  let island2_total := island2_distance * island2_increase
  let island3_total := (island3_morning + island3_afternoon) * island3_increase * island3_days
  let island4_total := island4_distance * island4_increase
  island1_total + island2_total + island3_total + island4_total = 247.25 := by
  sorry

end NUMINAMATH_CALUDE_pirate_total_distance_l944_94401


namespace NUMINAMATH_CALUDE_triangle_problem_l944_94489

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) (T : ℝ) :
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  (2 * a - c) * Real.cos B = b * Real.cos C ∧
  b = Real.sqrt 3 ∧
  T = (1 / 2) * a * c * Real.sin B →
  B = π / 3 ∧ a + c = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l944_94489


namespace NUMINAMATH_CALUDE_max_discount_percentage_l944_94499

/-- The maximum discount percentage that can be applied to a product while maintaining a minimum profit margin. -/
theorem max_discount_percentage
  (cost : ℝ)              -- Cost price in yuan
  (price : ℝ)             -- Selling price in yuan
  (min_margin : ℝ)        -- Minimum profit margin as a decimal
  (h_cost : cost = 100)   -- Cost is 100 yuan
  (h_price : price = 150) -- Price is 150 yuan
  (h_margin : min_margin = 0.2) -- Minimum margin is 20%
  : ∃ (max_discount : ℝ),
    max_discount = 20 ∧
    ∀ (discount : ℝ),
      0 ≤ discount ∧ discount ≤ max_discount →
      (price * (1 - discount / 100) - cost) / cost ≥ min_margin :=
by sorry

end NUMINAMATH_CALUDE_max_discount_percentage_l944_94499


namespace NUMINAMATH_CALUDE_max_value_theorem_l944_94441

theorem max_value_theorem (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  2 * x * y * Real.sqrt 3 + 3 * y * z * Real.sqrt 2 + 3 * z * x ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l944_94441


namespace NUMINAMATH_CALUDE_common_tangent_sum_l944_94460

-- Define the parabolas
def Q₁ (x y : ℝ) : Prop := y = x^2 + 53/50
def Q₂ (x y : ℝ) : Prop := x = y^2 + 91/8

-- Define the common tangent line
def M (p q r : ℕ) (x y : ℝ) : Prop := p * x + q * y = r

-- Main theorem
theorem common_tangent_sum (p q r : ℕ) :
  (p > 0) →
  (q > 0) →
  (r > 0) →
  (Nat.gcd p q = 1) →
  (Nat.gcd p r = 1) →
  (Nat.gcd q r = 1) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    Q₁ x₁ y₁ ∧ 
    Q₂ x₂ y₂ ∧ 
    M p q r x₁ y₁ ∧ 
    M p q r x₂ y₂ ∧
    (∃ (m : ℚ), q = m * p)) →
  p + q + r = 9 := by
  sorry

end NUMINAMATH_CALUDE_common_tangent_sum_l944_94460


namespace NUMINAMATH_CALUDE_gnomes_in_fifth_house_l944_94494

theorem gnomes_in_fifth_house 
  (total_houses : ℕ)
  (gnomes_per_house : ℕ)
  (houses_with_known_gnomes : ℕ)
  (total_gnomes : ℕ)
  (h1 : total_houses = 5)
  (h2 : gnomes_per_house = 3)
  (h3 : houses_with_known_gnomes = 4)
  (h4 : total_gnomes = 20) :
  total_gnomes - (houses_with_known_gnomes * gnomes_per_house) = 8 :=
by
  sorry

#check gnomes_in_fifth_house

end NUMINAMATH_CALUDE_gnomes_in_fifth_house_l944_94494


namespace NUMINAMATH_CALUDE_f_inequality_l944_94493

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

theorem f_inequality : f (Real.sqrt 3 / 2) > f (Real.sqrt 6 / 2) ∧ f (Real.sqrt 6 / 2) > f (Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l944_94493


namespace NUMINAMATH_CALUDE_f_negation_property_l944_94482

theorem f_negation_property (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = 2 * Real.sin x + x^3 + 1) →
  f a = 3 →
  f (-a) = -1 := by
sorry

end NUMINAMATH_CALUDE_f_negation_property_l944_94482


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l944_94426

theorem complex_modulus_problem (x y : ℝ) (z : ℂ) :
  z = x + y * Complex.I →
  x / (1 - Complex.I) = 1 + y * Complex.I →
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l944_94426


namespace NUMINAMATH_CALUDE_total_cows_l944_94430

/-- The number of cows owned by four men given specific conditions -/
theorem total_cows (matthews tyron aaron marovich : ℕ) : 
  matthews = 60 ∧ 
  aaron = 4 * matthews ∧ 
  tyron = matthews - 20 ∧ 
  aaron + matthews + tyron = marovich + 30 → 
  matthews + tyron + aaron + marovich = 650 := by
sorry

end NUMINAMATH_CALUDE_total_cows_l944_94430


namespace NUMINAMATH_CALUDE_bus_rental_plans_l944_94452

theorem bus_rental_plans (total_people : Nat) (large_bus_capacity : Nat) (medium_bus_capacity : Nat)
  (h1 : total_people = 1511)
  (h2 : large_bus_capacity = 42)
  (h3 : medium_bus_capacity = 25) :
  (∃! n : Nat, n = (Finset.filter
    (fun p : Nat × Nat => p.1 * large_bus_capacity + p.2 * medium_bus_capacity = total_people ∧
                          p.1 ≤ large_bus_capacity ∧ p.2 ≤ medium_bus_capacity)
    (Finset.product (Finset.range (large_bus_capacity + 1)) (Finset.range (medium_bus_capacity + 1)))).card) ∧
  n = 2 := by
sorry

end NUMINAMATH_CALUDE_bus_rental_plans_l944_94452


namespace NUMINAMATH_CALUDE_incorrect_inequality_l944_94479

theorem incorrect_inequality (x y : ℝ) (h : x > y) : ¬(-3 * x > -3 * y) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l944_94479


namespace NUMINAMATH_CALUDE_dunk_a_clown_tickets_l944_94488

/-- Proves the number of tickets spent at the 'dunk a clown' booth -/
theorem dunk_a_clown_tickets (total_tickets : ℕ) (num_rides : ℕ) (tickets_per_ride : ℕ) :
  total_tickets - (num_rides * tickets_per_ride) =
  total_tickets - num_rides * tickets_per_ride :=
by sorry

/-- Calculates the number of tickets spent at the 'dunk a clown' booth -/
def tickets_at_dunk_a_clown (total_tickets : ℕ) (num_rides : ℕ) (tickets_per_ride : ℕ) : ℕ :=
  total_tickets - (num_rides * tickets_per_ride)

#eval tickets_at_dunk_a_clown 79 8 7

end NUMINAMATH_CALUDE_dunk_a_clown_tickets_l944_94488


namespace NUMINAMATH_CALUDE_twentieth_15gonal_number_l944_94417

/-- The n-th k-gonal number -/
def N (n k : ℕ) : ℕ := ((k - 2) * n^2 + (4 - k) * n) / 2

/-- Theorem: The 20th 15-gonal number is 2490 -/
theorem twentieth_15gonal_number : N 20 15 = 2490 := by sorry

end NUMINAMATH_CALUDE_twentieth_15gonal_number_l944_94417


namespace NUMINAMATH_CALUDE_ratio_equality_l944_94487

theorem ratio_equality (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_eq1 : (x + z) / (2 * z - x) = x / y)
  (h_eq2 : (z + 2 * y) / (2 * x - z) = x / y) : 
  x / y = 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l944_94487


namespace NUMINAMATH_CALUDE_annabelle_savings_l944_94415

/-- Calculates the amount saved from a weekly allowance after spending on junk food and sweets -/
def calculate_savings (weekly_allowance : ℚ) (junk_food_fraction : ℚ) (sweets_cost : ℚ) : ℚ :=
  weekly_allowance - (weekly_allowance * junk_food_fraction + sweets_cost)

/-- Proves that given a weekly allowance of $30, spending 1/3 of it on junk food and an additional $8 on sweets, the remaining amount saved is $12 -/
theorem annabelle_savings :
  calculate_savings 30 (1/3) 8 = 12 := by
  sorry

end NUMINAMATH_CALUDE_annabelle_savings_l944_94415


namespace NUMINAMATH_CALUDE_two_pedestrians_problem_l944_94465

/-- Two pedestrians problem -/
theorem two_pedestrians_problem (meet_time : ℝ) (time_difference : ℝ) :
  meet_time = 2 ∧ time_difference = 5/3 →
  ∃ (distance_AB : ℝ) (speed_A : ℝ) (speed_B : ℝ),
    distance_AB = 18 ∧
    speed_A = 5 ∧
    speed_B = 4 ∧
    distance_AB = speed_A * meet_time + speed_B * meet_time ∧
    distance_AB / speed_A = meet_time + time_difference ∧
    distance_AB / speed_B = meet_time + (meet_time + time_difference) :=
by sorry

end NUMINAMATH_CALUDE_two_pedestrians_problem_l944_94465


namespace NUMINAMATH_CALUDE_hyperbola_range_solution_range_not_p_or_q_range_l944_94425

def is_hyperbola (m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (1 - 2*m) + y^2 / (m + 2) = 1 → 
    (1 - 2*m) * (m + 2) < 0

def has_solution (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*m*x + 2 - m = 0

theorem hyperbola_range (m : ℝ) :
  is_hyperbola m ↔ (m < -2 ∨ m > 1/2) :=
sorry

theorem solution_range (m : ℝ) :
  has_solution m ↔ (m ≤ -2 ∨ m ≥ 1) :=
sorry

theorem not_p_or_q_range (m : ℝ) :
  (¬is_hyperbola m ∧ ¬has_solution m) ↔ (-2 < m ∧ m ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_range_solution_range_not_p_or_q_range_l944_94425


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l944_94491

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h1 : a 3 - 3 * a 2 = 2) 
  (h2 : 5 * a 4 = (12 * a 3 + 2 * a 5) / 2) : 
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l944_94491


namespace NUMINAMATH_CALUDE_good_numbers_up_to_17_and_18_not_good_l944_94437

/-- The number of positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- A number m is "good" if there exists a positive integer n such that m = n / d(n) -/
def is_good (m : ℕ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ m = n / num_divisors n

theorem good_numbers_up_to_17_and_18_not_good :
  (∀ m : ℕ, m > 0 ∧ m ≤ 17 → is_good m) ∧ ¬ is_good 18 := by
  sorry

end NUMINAMATH_CALUDE_good_numbers_up_to_17_and_18_not_good_l944_94437


namespace NUMINAMATH_CALUDE_zoo_animals_count_l944_94454

theorem zoo_animals_count (zebras camels monkeys giraffes : ℕ) : 
  zebras = 12 →
  camels = zebras / 2 →
  monkeys = 4 * camels →
  monkeys = giraffes + 22 →
  giraffes = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_zoo_animals_count_l944_94454


namespace NUMINAMATH_CALUDE_smallest_6digit_binary_palindrome_4digit_other_base_l944_94420

/-- Checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a number from one base to another -/
def baseConvert (n : ℕ) (fromBase toBase : ℕ) : ℕ := sorry

/-- Counts the number of digits in a number in a given base -/
def digitCount (n : ℕ) (base : ℕ) : ℕ := sorry

theorem smallest_6digit_binary_palindrome_4digit_other_base :
  ∀ n : ℕ,
  isPalindrome n 2 →
  digitCount n 2 = 6 →
  (∃ b : ℕ, b > 2 ∧ isPalindrome (baseConvert n 2 b) b ∧ digitCount (baseConvert n 2 b) b = 4) →
  n ≥ 33 := by sorry

end NUMINAMATH_CALUDE_smallest_6digit_binary_palindrome_4digit_other_base_l944_94420


namespace NUMINAMATH_CALUDE_initial_population_size_l944_94446

/-- Given a population with specific birth and death rates, prove the initial population size. -/
theorem initial_population_size 
  (P : ℕ) 
  (birth_rate : ℕ) 
  (death_rate : ℕ) 
  (net_growth_rate : ℚ) 
  (h1 : birth_rate = 52) 
  (h2 : death_rate = 16) 
  (h3 : net_growth_rate = 12/1000) 
  (h4 : (birth_rate - death_rate : ℚ) / P = net_growth_rate) : 
  P = 3000 := by
  sorry

end NUMINAMATH_CALUDE_initial_population_size_l944_94446


namespace NUMINAMATH_CALUDE_expression_evaluation_l944_94412

theorem expression_evaluation :
  let x : ℚ := -2
  (1 - 1 / (1 - x)) / (x^2 / (x^2 - 1)) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l944_94412


namespace NUMINAMATH_CALUDE_lawn_care_supplies_cost_l944_94410

/-- The total cost of supplies for a lawn care company -/
theorem lawn_care_supplies_cost 
  (num_blades : ℕ) 
  (blade_cost : ℕ) 
  (string_cost : ℕ) :
  num_blades = 4 →
  blade_cost = 8 →
  string_cost = 7 →
  num_blades * blade_cost + string_cost = 39 :=
by
  sorry

end NUMINAMATH_CALUDE_lawn_care_supplies_cost_l944_94410


namespace NUMINAMATH_CALUDE_function_property_l944_94450

theorem function_property (f : ℤ → ℤ) 
  (h1 : ∀ x y : ℤ, f (x + y) = f x + f y + 2 * x * y + 1) 
  (h2 : f (-2) = 1) :
  ∀ n : ℕ+, f (2 * n) = 4 * n^2 + 2 * n - 1 :=
by sorry

end NUMINAMATH_CALUDE_function_property_l944_94450


namespace NUMINAMATH_CALUDE_janet_snowball_percentage_l944_94445

/-- Given that Janet makes 50 snowballs and her brother makes 150 snowballs,
    prove that Janet made 25% of the total snowballs. -/
theorem janet_snowball_percentage
  (janet_snowballs : ℕ)
  (brother_snowballs : ℕ)
  (h1 : janet_snowballs = 50)
  (h2 : brother_snowballs = 150) :
  (janet_snowballs : ℚ) / (janet_snowballs + brother_snowballs) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_janet_snowball_percentage_l944_94445


namespace NUMINAMATH_CALUDE_fraction_comparison_l944_94451

theorem fraction_comparison : (17 : ℚ) / 14 > (31 : ℚ) / 11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l944_94451


namespace NUMINAMATH_CALUDE_translator_assignment_count_l944_94461

def total_translators : ℕ := 9
def english_only_translators : ℕ := 6
def korean_only_translators : ℕ := 2
def bilingual_translators : ℕ := 1
def groups_needing_korean : ℕ := 2
def groups_needing_english : ℕ := 3

def assignment_ways : ℕ := sorry

theorem translator_assignment_count : 
  assignment_ways = 900 := by sorry

end NUMINAMATH_CALUDE_translator_assignment_count_l944_94461


namespace NUMINAMATH_CALUDE_x_value_l944_94448

theorem x_value (x y : ℝ) (hx : x ≠ 0) (h1 : x/3 = y^2) (h2 : x/6 = 3*y) : x = 108 :=
by sorry

end NUMINAMATH_CALUDE_x_value_l944_94448


namespace NUMINAMATH_CALUDE_edward_final_earnings_l944_94481

/-- Edward's lawn mowing business earnings and expenses --/
def edward_business (spring_earnings summer_earnings supplies_cost : ℕ) : ℕ :=
  spring_earnings + summer_earnings - supplies_cost

/-- Theorem: Edward's final earnings --/
theorem edward_final_earnings :
  edward_business 2 27 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_edward_final_earnings_l944_94481


namespace NUMINAMATH_CALUDE_add_zero_or_nine_divisible_by_nine_l944_94490

/-- Represents a ten-digit number with different digits -/
def TenDigitNumber := {n : Fin 10 → Fin 10 // Function.Injective n}

/-- The sum of digits in a ten-digit number -/
def digitSum (n : TenDigitNumber) : ℕ :=
  (Finset.univ.sum fun i => (n.val i).val)

/-- The theorem stating that adding 0 or 9 to a ten-digit number with different digits 
    results in a number divisible by 9 -/
theorem add_zero_or_nine_divisible_by_nine (n : TenDigitNumber) :
  (∃ x : Fin 10, x = 0 ∨ x = 9) ∧ 
  (∃ m : ℕ, (digitSum n + x) = 9 * m) := by
  sorry


end NUMINAMATH_CALUDE_add_zero_or_nine_divisible_by_nine_l944_94490


namespace NUMINAMATH_CALUDE_diagonals_from_vertex_l944_94407

/-- For a polygon with interior angles summing to 540°, 
    the number of diagonals that can be drawn from one vertex is 2. -/
theorem diagonals_from_vertex (n : ℕ) : 
  (n - 2) * 180 = 540 → (n - 3 : ℕ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_from_vertex_l944_94407


namespace NUMINAMATH_CALUDE_jihoon_calculation_mistake_l944_94400

theorem jihoon_calculation_mistake (x : ℝ) : 
  x - 7 = 0.45 → x * 7 = 52.15 := by
sorry

end NUMINAMATH_CALUDE_jihoon_calculation_mistake_l944_94400


namespace NUMINAMATH_CALUDE_correct_calculation_l944_94409

theorem correct_calculation : (-36 : ℚ) / (-1/2 + 1/6 - 1/3) = 54 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l944_94409


namespace NUMINAMATH_CALUDE_intersection_point_x_coord_l944_94421

/-- Hyperbola C with given properties -/
structure Hyperbola where
  center : ℝ × ℝ
  left_focus : ℝ × ℝ
  eccentricity : ℝ
  left_vertex : ℝ × ℝ
  right_vertex : ℝ × ℝ

/-- Line intersecting the left branch of the hyperbola -/
structure IntersectingLine where
  point : ℝ × ℝ
  intersection1 : ℝ × ℝ
  intersection2 : ℝ × ℝ

/-- Point P is the intersection of lines MA₁ and NA₂ -/
def intersection_point (h : Hyperbola) (l : IntersectingLine) : ℝ × ℝ := sorry

/-- Main theorem: The x-coordinate of point P is always -1 -/
theorem intersection_point_x_coord (h : Hyperbola) (l : IntersectingLine) :
  h.center = (0, 0) →
  h.left_focus = (-2 * Real.sqrt 5, 0) →
  h.eccentricity = Real.sqrt 5 →
  h.left_vertex = (-2, 0) →
  h.right_vertex = (2, 0) →
  l.point = (-4, 0) →
  l.intersection1.1 < 0 ∧ l.intersection1.2 > 0 →  -- M is in the second quadrant
  (intersection_point h l).1 = -1 := by sorry

end NUMINAMATH_CALUDE_intersection_point_x_coord_l944_94421


namespace NUMINAMATH_CALUDE_quadratic_root_implies_coefficients_l944_94470

theorem quadratic_root_implies_coefficients
  (a b : ℚ)
  (h : (1 + Real.sqrt 3)^2 + a * (1 + Real.sqrt 3) + b = 0) :
  a = -2 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_coefficients_l944_94470


namespace NUMINAMATH_CALUDE_dining_sales_tax_percentage_l944_94416

/-- Proves that the sales tax percentage is 10% given the conditions of the dining problem -/
theorem dining_sales_tax_percentage : 
  ∀ (total_spent food_price tip_percentage sales_tax_percentage : ℝ),
  total_spent = 132 →
  food_price = 100 →
  tip_percentage = 20 →
  total_spent = food_price * (1 + sales_tax_percentage / 100) * (1 + tip_percentage / 100) →
  sales_tax_percentage = 10 := by
sorry


end NUMINAMATH_CALUDE_dining_sales_tax_percentage_l944_94416


namespace NUMINAMATH_CALUDE_problem_statement_l944_94413

theorem problem_statement (m n : ℝ) (a b : ℝ) 
  (h1 : m + n = 9)
  (h2 : 0 < a ∧ 0 < b)
  (h3 : a^2 + b^2 = 9) : 
  (∀ x : ℝ, |x - m| + |x + n| ≥ 9) ∧ 
  (a + b) * (a^3 + b^3) ≥ 81 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l944_94413


namespace NUMINAMATH_CALUDE_framing_for_enlarged_picture_l944_94457

/-- Calculates the minimum number of linear feet of framing needed for an enlarged and bordered picture. -/
def min_framing_feet (original_width original_height enlarge_factor border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlarge_factor
  let enlarged_height := original_height * enlarge_factor
  let total_width := enlarged_width + 2 * border_width
  let total_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (total_width + total_height)
  ((perimeter_inches + 11) / 12 : ℕ)

/-- Theorem stating the minimum number of linear feet of framing needed for the given picture specifications. -/
theorem framing_for_enlarged_picture :
  min_framing_feet 4 6 4 3 = 9 :=
by sorry

end NUMINAMATH_CALUDE_framing_for_enlarged_picture_l944_94457


namespace NUMINAMATH_CALUDE_divisibility_properties_l944_94439

theorem divisibility_properties (a : ℤ) : 
  (∃ k : ℤ, a^5 - a = 30 * k) ∧
  (∃ l : ℤ, a^17 - a = 510 * l) ∧
  (∃ m : ℤ, a^11 - a = 66 * m) ∧
  (∃ n : ℤ, a^73 - a = (2 * 3 * 5 * 7 * 13 * 19 * 37 * 73) * n) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_properties_l944_94439


namespace NUMINAMATH_CALUDE_triangle_dimensions_l944_94456

theorem triangle_dimensions (a m : ℝ) (h1 : a = m + 4) (h2 : (a + 12) * (m + 12) = 5 * a * m) : 
  a = 12 ∧ m = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_dimensions_l944_94456


namespace NUMINAMATH_CALUDE_mass_of_man_is_60kg_l944_94462

/-- The mass of a man causing a boat to sink in water -/
def mass_of_man (boat_length boat_breadth boat_sinking water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sinking * water_density

/-- Theorem stating the mass of the man is 60 kg -/
theorem mass_of_man_is_60kg : 
  mass_of_man 3 2 0.01 1000 = 60 := by sorry

end NUMINAMATH_CALUDE_mass_of_man_is_60kg_l944_94462


namespace NUMINAMATH_CALUDE_ball_placement_theorem_l944_94455

/-- Represents the number of balls and boxes -/
def n : ℕ := 5

/-- Calculates the number of ways to place n balls into n boxes with one empty box -/
def ways_one_empty (n : ℕ) : ℕ := sorry

/-- Calculates the number of ways to place n balls into n boxes with no empty box and not all numbers matching -/
def ways_no_empty_not_all_match (n : ℕ) : ℕ := sorry

/-- Calculates the number of ways to place n balls into n boxes with one ball in each box and at least two balls matching their box numbers -/
def ways_at_least_two_match (n : ℕ) : ℕ := sorry

/-- Theorem stating the correct number of ways for each scenario with 5 balls and 5 boxes -/
theorem ball_placement_theorem :
  ways_one_empty n = 1200 ∧
  ways_no_empty_not_all_match n = 119 ∧
  ways_at_least_two_match n = 31 := by sorry

end NUMINAMATH_CALUDE_ball_placement_theorem_l944_94455


namespace NUMINAMATH_CALUDE_olympiad_problem_l944_94472

theorem olympiad_problem (total_students : ℕ) 
  (solved_at_least_1 solved_at_least_2 solved_at_least_3 solved_at_least_4 solved_at_least_5 solved_all_6 : ℕ) : 
  total_students = 2006 →
  solved_at_least_1 = 4 * solved_at_least_2 →
  solved_at_least_2 = 4 * solved_at_least_3 →
  solved_at_least_3 = 4 * solved_at_least_4 →
  solved_at_least_4 = 4 * solved_at_least_5 →
  solved_at_least_5 = 4 * solved_all_6 →
  total_students - solved_at_least_1 = 982 :=
by sorry

end NUMINAMATH_CALUDE_olympiad_problem_l944_94472


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_shared_foci_l944_94458

theorem hyperbola_ellipse_shared_foci (m : ℝ) : 
  (∃ (c : ℝ), c > 0 ∧ 
    c^2 = 12 - 4 ∧ 
    c^2 = m + 1) → 
  m = 7 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_shared_foci_l944_94458


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l944_94471

theorem solution_set_of_inequality (x : ℝ) :
  (-x^2 + 2*x + 15 ≥ 0) ↔ (-3 ≤ x ∧ x ≤ 5) := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l944_94471


namespace NUMINAMATH_CALUDE_words_with_consonant_count_l944_94468

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def consonants : Finset Char := {'B', 'C', 'D', 'F'}
def vowels : Finset Char := alphabet \ consonants

def word_length : Nat := 5

def total_words : Nat := alphabet.card ^ word_length
def vowel_only_words : Nat := vowels.card ^ word_length

theorem words_with_consonant_count :
  total_words - vowel_only_words = 7744 :=
sorry

end NUMINAMATH_CALUDE_words_with_consonant_count_l944_94468


namespace NUMINAMATH_CALUDE_inequality_proof_l944_94429

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z ≥ 1) :
  (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + z^2 + x^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l944_94429


namespace NUMINAMATH_CALUDE_perpendicular_condition_l944_94433

-- Define the lines
def line1 (m : ℝ) (x y : ℝ) : Prop := m * x + (2 * m - 1) * y + 1 = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := 3 * x + m * y + 3 = 0

-- Define perpendicularity of two lines
def perpendicular (m : ℝ) : Prop :=
  ∀ x1 y1 x2 y2 : ℝ, line1 m x1 y1 → line2 m x2 y2 →
    (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ≠ 0 →
    (x2 - x1) * (y2 - y1) = 0

-- State the theorem
theorem perpendicular_condition (m : ℝ) :
  (m = -1 → perpendicular m) ∧ ¬(perpendicular m → m = -1) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l944_94433


namespace NUMINAMATH_CALUDE_exercise_book_distribution_l944_94406

theorem exercise_book_distribution (m n : ℕ) : 
  (3 * n + 8 = m) →  -- If each student receives 3 books, there will be 8 books left over
  (0 < m - 5 * (n - 1)) →  -- The last student receives some books
  (m - 5 * (n - 1) < 5) →  -- The last student receives less than 5 books
  (n = 5 ∨ n = 6) := by
sorry

end NUMINAMATH_CALUDE_exercise_book_distribution_l944_94406


namespace NUMINAMATH_CALUDE_prob_even_first_odd_second_l944_94442

/-- The number of sides on a standard die -/
def sides : ℕ := 6

/-- The number of even outcomes on a standard die -/
def evenOutcomes : ℕ := 3

/-- The number of odd outcomes on a standard die -/
def oddOutcomes : ℕ := 3

/-- The probability of rolling an even number on one die -/
def probEven : ℚ := evenOutcomes / sides

/-- The probability of rolling an odd number on one die -/
def probOdd : ℚ := oddOutcomes / sides

theorem prob_even_first_odd_second : probEven * probOdd = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_first_odd_second_l944_94442


namespace NUMINAMATH_CALUDE_first_day_exceeding_threshold_l944_94475

def initial_bacteria : ℕ := 5
def growth_rate : ℕ := 3
def threshold : ℕ := 200

def bacteria_count (n : ℕ) : ℕ := initial_bacteria * growth_rate ^ n

theorem first_day_exceeding_threshold :
  ∃ n : ℕ, bacteria_count n > threshold ∧ ∀ m : ℕ, m < n → bacteria_count m ≤ threshold :=
by sorry

end NUMINAMATH_CALUDE_first_day_exceeding_threshold_l944_94475


namespace NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l944_94483

/-- A dodecahedron is a 3D figure with 20 vertices and 3 faces meeting at each vertex. -/
structure Dodecahedron where
  vertices : ℕ
  faces_per_vertex : ℕ
  h_vertices : vertices = 20
  h_faces_per_vertex : faces_per_vertex = 3

/-- The number of interior diagonals in a dodecahedron. -/
def interior_diagonals (d : Dodecahedron) : ℕ :=
  (d.vertices * (d.vertices - 1 - 2 * d.faces_per_vertex)) / 2

/-- Theorem: The number of interior diagonals in a dodecahedron is 160. -/
theorem dodecahedron_interior_diagonals (d : Dodecahedron) :
  interior_diagonals d = 160 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l944_94483


namespace NUMINAMATH_CALUDE_tree_age_conversion_l944_94405

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The given number in base 7 -/
def treeAgeBase7 : List Nat := [7, 4, 5, 2]

theorem tree_age_conversion :
  base7ToBase10 treeAgeBase7 = 966 := by
  sorry

end NUMINAMATH_CALUDE_tree_age_conversion_l944_94405


namespace NUMINAMATH_CALUDE_sean_has_more_whistles_l944_94447

/-- The number of whistles Sean has -/
def sean_whistles : ℕ := 45

/-- The number of whistles Charles has -/
def charles_whistles : ℕ := 13

/-- The difference in whistle count between Sean and Charles -/
def whistle_difference : ℕ := sean_whistles - charles_whistles

theorem sean_has_more_whistles : whistle_difference = 32 := by
  sorry

end NUMINAMATH_CALUDE_sean_has_more_whistles_l944_94447


namespace NUMINAMATH_CALUDE_equation_solution_l944_94431

theorem equation_solution : 
  ∃! x : ℚ, (x + 2) / 4 - (2 * x - 3) / 6 = 2 ∧ x = -12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l944_94431


namespace NUMINAMATH_CALUDE_A_inter_B_l944_94453

def A : Set ℤ := {-1, 0, 1}

def B : Set ℤ := {y | ∃ x ∈ A, y = x^2}

theorem A_inter_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_A_inter_B_l944_94453


namespace NUMINAMATH_CALUDE_complex_equation_implies_real_equation_l944_94403

theorem complex_equation_implies_real_equation (a b : ℝ) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (3 + 4 * Complex.I) * (a + b * Complex.I) = 10 * Complex.I →
  3 * a - 4 * b = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_implies_real_equation_l944_94403


namespace NUMINAMATH_CALUDE_parking_lot_cars_l944_94427

/-- Given a parking lot with large and small cars, prove the number of each type. -/
theorem parking_lot_cars (total_vehicles : ℕ) (total_wheels : ℕ) 
  (large_car_wheels : ℕ) (small_car_wheels : ℕ) 
  (h_total_vehicles : total_vehicles = 6)
  (h_total_wheels : total_wheels = 32)
  (h_large_car_wheels : large_car_wheels = 6)
  (h_small_car_wheels : small_car_wheels = 4) :
  ∃ (large_cars small_cars : ℕ),
    large_cars + small_cars = total_vehicles ∧
    large_cars * large_car_wheels + small_cars * small_car_wheels = total_wheels ∧
    large_cars = 4 ∧
    small_cars = 2 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_cars_l944_94427


namespace NUMINAMATH_CALUDE_triangle_problem_l944_94438

theorem triangle_problem (A B C : Real) (a b c : Real) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  a = Real.sqrt 7 →
  b = 3 →
  Real.sqrt 7 * Real.sin B + Real.sin A = 2 * Real.sqrt 3 →
  a / Real.sin A = b / Real.sin B →
  c / Real.sin C = a / Real.sin A →
  A + B + C = π →
  (A = π/3 ∧ Real.sin (2*B + π/6) = -1/7) := by
sorry


end NUMINAMATH_CALUDE_triangle_problem_l944_94438


namespace NUMINAMATH_CALUDE_angle_property_equivalence_l944_94498

theorem angle_property_equivalence (θ : Real) (h : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) ↔
  (Real.pi / 12 < θ ∧ θ < 5 * Real.pi / 12) :=
by sorry

end NUMINAMATH_CALUDE_angle_property_equivalence_l944_94498


namespace NUMINAMATH_CALUDE_prob_units_digit_8_is_3_16_l944_94486

def S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 100}

def units_digit (n : ℕ) : ℕ := n % 10

def prob_units_digit_8 : ℚ :=
  (Finset.filter (fun (a, b) => units_digit (3^a + 7^b) = 8) (Finset.product (Finset.range 100) (Finset.range 100))).card /
  (Finset.product (Finset.range 100) (Finset.range 100)).card

theorem prob_units_digit_8_is_3_16 : prob_units_digit_8 = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_units_digit_8_is_3_16_l944_94486


namespace NUMINAMATH_CALUDE_set_operations_l944_94402

-- Define the sets A and B
def A : Set ℝ := {x | x < 1 ∨ x > 2}
def B : Set ℝ := {x | x < -3 ∨ x ≥ 1}

-- State the theorem
theorem set_operations :
  (Set.univ \ A = {x | 1 ≤ x ∧ x ≤ 2}) ∧
  (Set.univ \ B = {x | -3 ≤ x ∧ x < 1}) ∧
  (A ∩ B = {x | x < -3 ∨ x > 2}) ∧
  (A ∪ B = Set.univ) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l944_94402


namespace NUMINAMATH_CALUDE_bus_journey_speed_l944_94404

/-- Calculates the average speed for the remaining distance of a bus journey -/
theorem bus_journey_speed 
  (total_distance : ℝ) 
  (partial_distance : ℝ) 
  (partial_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : total_distance = 250)
  (h2 : partial_distance = 100)
  (h3 : partial_speed = 40)
  (h4 : total_time = 5)
  : (total_distance - partial_distance) / (total_time - partial_distance / partial_speed) = 60 := by
  sorry

end NUMINAMATH_CALUDE_bus_journey_speed_l944_94404


namespace NUMINAMATH_CALUDE_julia_total_kids_l944_94480

/-- The total number of kids Julia played or interacted with during the week -/
def total_kids : ℕ :=
  let monday_tag := 7
  let tuesday_tag := 13
  let thursday_tag := 18
  let wednesday_cards := 20
  let wednesday_hide_seek := 11
  let wednesday_puzzle := 9
  let friday_board_game := 15
  let friday_drawing := 12
  monday_tag + tuesday_tag + thursday_tag + wednesday_cards + wednesday_hide_seek + wednesday_puzzle + friday_board_game + friday_drawing

theorem julia_total_kids : total_kids = 105 := by
  sorry

end NUMINAMATH_CALUDE_julia_total_kids_l944_94480


namespace NUMINAMATH_CALUDE_circle_packing_line_division_l944_94476

/-- A circle in the coordinate plane --/
structure Circle where
  center : ℝ × ℝ
  diameter : ℝ

/-- The region formed by the union of circular regions --/
def Region (circles : List Circle) : Set (ℝ × ℝ) := sorry

/-- A line in the coordinate plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Check if a line divides a region into two equal areas --/
def dividesEquallyArea (l : Line) (r : Set (ℝ × ℝ)) : Prop := sorry

/-- Express a line in the form ax = by + c --/
def lineToStandardForm (l : Line) : ℕ × ℕ × ℕ := sorry

/-- The greatest common divisor of three natural numbers --/
def gcd3 (a b c : ℕ) : ℕ := sorry

theorem circle_packing_line_division :
  ∀ (circles : List Circle) (l : Line),
    circles.length = 6 ∧
    (∀ c ∈ circles, c.diameter = 2 ∧ c.center.1 > 0 ∧ c.center.2 > 0) ∧
    l.slope = 2 ∧
    dividesEquallyArea l (Region circles) →
    let (a, b, c) := lineToStandardForm l
    gcd3 a b c = 1 →
    a^2 + b^2 + c^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_packing_line_division_l944_94476
