import Mathlib

namespace NUMINAMATH_CALUDE_no_consecutive_beeches_probability_l91_9158

/-- The number of oaks to be planted -/
def num_oaks : ℕ := 3

/-- The number of holm oaks to be planted -/
def num_holm_oaks : ℕ := 4

/-- The number of beeches to be planted -/
def num_beeches : ℕ := 5

/-- The total number of trees to be planted -/
def total_trees : ℕ := num_oaks + num_holm_oaks + num_beeches

/-- The probability of no two beeches being consecutive when planted randomly -/
def prob_no_consecutive_beeches : ℚ := 7 / 99

theorem no_consecutive_beeches_probability :
  let total_arrangements := (total_trees.factorial) / (num_oaks.factorial * num_holm_oaks.factorial * num_beeches.factorial)
  let favorable_arrangements := (Nat.choose 8 5) * ((num_oaks + num_holm_oaks).factorial / (num_oaks.factorial * num_holm_oaks.factorial))
  (favorable_arrangements : ℚ) / total_arrangements = prob_no_consecutive_beeches := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_beeches_probability_l91_9158


namespace NUMINAMATH_CALUDE_washing_machine_cost_l91_9114

theorem washing_machine_cost (down_payment : ℝ) (down_payment_percentage : ℝ) (total_cost : ℝ) : 
  down_payment = 200 →
  down_payment_percentage = 25 →
  down_payment = (down_payment_percentage / 100) * total_cost →
  total_cost = 800 := by
sorry

end NUMINAMATH_CALUDE_washing_machine_cost_l91_9114


namespace NUMINAMATH_CALUDE_min_socks_for_different_pairs_l91_9126

/-- Represents a sock with a size and a color -/
structure Sock :=
  (size : Nat)
  (color : Nat)

/-- Represents the total number of socks -/
def totalSocks : Nat := 8

/-- Represents the number of different sizes -/
def numSizes : Nat := 2

/-- Represents the number of different colors -/
def numColors : Nat := 2

/-- Theorem stating the minimum number of socks needed to guarantee two pairs of different sizes and colors -/
theorem min_socks_for_different_pairs :
  ∀ (socks : Finset Sock),
    (Finset.card socks = totalSocks) →
    (∀ s ∈ socks, s.size < numSizes ∧ s.color < numColors) →
    (∃ (n : Nat),
      ∀ (subset : Finset Sock),
        (Finset.card subset = n) →
        (subset ⊆ socks) →
        (∃ (s1 s2 s3 s4 : Sock),
          s1 ∈ subset ∧ s2 ∈ subset ∧ s3 ∈ subset ∧ s4 ∈ subset ∧
          s1.size ≠ s2.size ∧ s1.color ≠ s2.color ∧
          s3.size ≠ s4.size ∧ s3.color ≠ s4.color)) →
    n = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_socks_for_different_pairs_l91_9126


namespace NUMINAMATH_CALUDE_sticker_distribution_l91_9188

/-- The number of ways to partition n identical objects into at most k parts -/
def partitions (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 30 ways to partition 10 identical objects into at most 5 parts -/
theorem sticker_distribution : partitions 10 5 = 30 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l91_9188


namespace NUMINAMATH_CALUDE_students_not_finding_parents_funny_l91_9105

theorem students_not_finding_parents_funny 
  (total : ℕ) 
  (funny_dad : ℕ) 
  (funny_mom : ℕ) 
  (funny_both : ℕ) 
  (h1 : total = 50) 
  (h2 : funny_dad = 25) 
  (h3 : funny_mom = 30) 
  (h4 : funny_both = 18) : 
  total - (funny_dad + funny_mom - funny_both) = 13 := by
  sorry

end NUMINAMATH_CALUDE_students_not_finding_parents_funny_l91_9105


namespace NUMINAMATH_CALUDE_rectangle_circle_mass_ratio_l91_9112

/-- Represents the mass of an object -/
structure Mass where
  value : ℝ
  nonneg : 0 ≤ value

/-- Represents an equal-arm scale -/
structure EqualArmScale where
  left : Mass
  right : Mass
  balanced : left.value = right.value

/-- The mass of a rectangle -/
def rectangle_mass : Mass := sorry

/-- The mass of a circle -/
def circle_mass : Mass := sorry

/-- The theorem statement -/
theorem rectangle_circle_mass_ratio 
  (scale : EqualArmScale)
  (h1 : scale.left = Mass.mk (2 * rectangle_mass.value) (by sorry))
  (h2 : scale.right = Mass.mk (6 * circle_mass.value) (by sorry)) :
  rectangle_mass.value = 3 * circle_mass.value :=
sorry

end NUMINAMATH_CALUDE_rectangle_circle_mass_ratio_l91_9112


namespace NUMINAMATH_CALUDE_expression_simplification_l91_9146

theorem expression_simplification (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 2) (h3 : a ≠ -2) :
  (a - 1) / (a + 2) / ((a^2 - 2*a) / (a^2 - 4)) - (a + 1) / a = -2 / a := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l91_9146


namespace NUMINAMATH_CALUDE_prime_cube_difference_equation_l91_9189

theorem prime_cube_difference_equation :
  ∀ p q r : ℕ,
    Prime p → Prime q → Prime r →
    5 * p = q^3 - r^3 →
    p = 67 ∧ q = 7 ∧ r = 2 :=
by sorry

end NUMINAMATH_CALUDE_prime_cube_difference_equation_l91_9189


namespace NUMINAMATH_CALUDE_linear_function_shift_l91_9104

/-- A linear function y = 2x + b shifted down by 2 units passing through (-1, 0) implies b = 4 -/
theorem linear_function_shift (b : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + b - 2) →  -- shifted function
  (0 = 2 * (-1) + b - 2) →          -- passes through (-1, 0)
  b = 4 := by
sorry

end NUMINAMATH_CALUDE_linear_function_shift_l91_9104


namespace NUMINAMATH_CALUDE_function_transformation_l91_9141

theorem function_transformation (f : ℝ → ℝ) : 
  (∀ x, f (x - 1) = 19 * x^2 + 55 * x - 44) → 
  (∀ x, f x = 19 * x^2 + 93 * x + 30) :=
by sorry

end NUMINAMATH_CALUDE_function_transformation_l91_9141


namespace NUMINAMATH_CALUDE_product_of_three_consecutive_integers_divisible_by_six_l91_9101

theorem product_of_three_consecutive_integers_divisible_by_six (k : ℤ) :
  ∃ m : ℤ, k * (k + 1) * (k + 2) = 6 * m :=
by sorry

end NUMINAMATH_CALUDE_product_of_three_consecutive_integers_divisible_by_six_l91_9101


namespace NUMINAMATH_CALUDE_food_drive_mark_cans_l91_9142

/-- Represents the number of cans brought by each person -/
structure Cans where
  rachel : ℕ
  jaydon : ℕ
  mark : ℕ
  sophie : ℕ

/-- Conditions for the food drive -/
def FoodDriveConditions (c : Cans) : Prop :=
  c.mark = 4 * c.jaydon ∧
  c.jaydon = 2 * c.rachel + 5 ∧
  4 * c.sophie = 3 * c.jaydon ∧
  c.rachel ≥ 5 ∧ c.jaydon ≥ 5 ∧ c.mark ≥ 5 ∧ c.sophie ≥ 5 ∧
  Odd (c.rachel + c.jaydon + c.mark + c.sophie) ∧
  c.rachel + c.jaydon + c.mark + c.sophie ≥ 250

theorem food_drive_mark_cans (c : Cans) (h : FoodDriveConditions c) : c.mark = 148 := by
  sorry

end NUMINAMATH_CALUDE_food_drive_mark_cans_l91_9142


namespace NUMINAMATH_CALUDE_least_coins_coins_exist_l91_9111

theorem least_coins (n : ℕ) : 
  (n % 6 = 3) ∧ (n % 4 = 1) ∧ (n % 7 = 2) → n ≥ 9 :=
by sorry

theorem coins_exist : 
  ∃ n : ℕ, (n % 6 = 3) ∧ (n % 4 = 1) ∧ (n % 7 = 2) ∧ n = 9 :=
by sorry

end NUMINAMATH_CALUDE_least_coins_coins_exist_l91_9111


namespace NUMINAMATH_CALUDE_union_A_B_when_a_4_intersection_A_B_equals_A_iff_l91_9162

open Set
open Real

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}
def B : Set ℝ := {x | (4 - x) * (x - 1) ≤ 0}

-- Theorem 1: When a = 4, A ∪ B = {x | x ≥ 3 ∨ x ≤ 1}
theorem union_A_B_when_a_4 : 
  A 4 ∪ B = {x : ℝ | x ≥ 3 ∨ x ≤ 1} := by sorry

-- Theorem 2: A ∩ B = A if and only if a ≥ 5 or a ≤ 0
theorem intersection_A_B_equals_A_iff (a : ℝ) : 
  A a ∩ B = A a ↔ a ≥ 5 ∨ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_a_4_intersection_A_B_equals_A_iff_l91_9162


namespace NUMINAMATH_CALUDE_sqrt_ln_relation_l91_9197

theorem sqrt_ln_relation (a b : ℝ) :
  (∀ a b, (Real.log a > Real.log b) → (Real.sqrt a > Real.sqrt b)) ∧
  (∃ a b, (Real.sqrt a > Real.sqrt b) ∧ ¬(Real.log a > Real.log b)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_ln_relation_l91_9197


namespace NUMINAMATH_CALUDE_vector_expression_in_quadrilateral_l91_9156

/-- Given a quadrilateral OABC in space, prove that MN = -1/2 * a + 1/2 * b + 1/2 * c -/
theorem vector_expression_in_quadrilateral
  (O A B C M N : EuclideanSpace ℝ (Fin 3))
  (a b c : EuclideanSpace ℝ (Fin 3))
  (h1 : A - O = a)
  (h2 : B - O = b)
  (h3 : C - O = c)
  (h4 : M - O = (1/2) • (A - O))
  (h5 : N - B = (1/2) • (C - B)) :
  N - M = (-1/2) • a + (1/2) • b + (1/2) • c := by
  sorry

end NUMINAMATH_CALUDE_vector_expression_in_quadrilateral_l91_9156


namespace NUMINAMATH_CALUDE_mask_digit_assignment_l91_9154

/-- Represents the four masks in the problem -/
inductive Mask
| elephant
| mouse
| pig
| panda

/-- Assigns a digit to each mask -/
def digit_assignment : Mask → Nat
| Mask.elephant => 6
| Mask.mouse => 4
| Mask.pig => 8
| Mask.panda => 1

/-- Checks if a number is two digits -/
def is_two_digit (n : Nat) : Prop := n ≥ 10 ∧ n < 100

/-- The main theorem statement -/
theorem mask_digit_assignment :
  (∀ m : Mask, digit_assignment m ≤ 9) ∧ 
  (∀ m1 m2 : Mask, m1 ≠ m2 → digit_assignment m1 ≠ digit_assignment m2) ∧
  (∀ m : Mask, is_two_digit ((digit_assignment m) * (digit_assignment m))) ∧
  (∀ m : Mask, (digit_assignment m) * (digit_assignment m) % 10 ≠ digit_assignment m) ∧
  ((digit_assignment Mask.mouse) * (digit_assignment Mask.mouse) % 10 = digit_assignment Mask.elephant) :=
by sorry

end NUMINAMATH_CALUDE_mask_digit_assignment_l91_9154


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_2_range_of_t_l91_9102

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 2| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x > 2/3 ∨ x < -6} := by sorry

-- Theorem for the range of t
theorem range_of_t :
  {t : ℝ | ∀ x, f x ≥ t^2 - (7/2)*t} = {t : ℝ | 3/2 ≤ t ∧ t ≤ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_2_range_of_t_l91_9102


namespace NUMINAMATH_CALUDE_inverse_of_exponential_function_l91_9129

noncomputable def f (x : ℝ) : ℝ := 3^x

theorem inverse_of_exponential_function (x : ℝ) (h : x > 0) : 
  f⁻¹ x = Real.log x / Real.log 3 := by
sorry

end NUMINAMATH_CALUDE_inverse_of_exponential_function_l91_9129


namespace NUMINAMATH_CALUDE_rain_probability_tel_aviv_l91_9107

theorem rain_probability_tel_aviv : 
  let n : ℕ := 6  -- number of days
  let k : ℕ := 4  -- number of rainy days
  let p : ℚ := 1/2  -- probability of rain on any given day
  Nat.choose n k * p^k * (1-p)^(n-k) = 15/64 :=
by sorry

end NUMINAMATH_CALUDE_rain_probability_tel_aviv_l91_9107


namespace NUMINAMATH_CALUDE_jose_investment_is_45000_l91_9135

/-- Represents the investment problem with Tom and Jose --/
structure InvestmentProblem where
  tom_investment : ℕ
  jose_join_delay : ℕ
  total_profit : ℕ
  jose_profit_share : ℕ

/-- Calculates Jose's investment amount based on the given parameters --/
def calculate_jose_investment (problem : InvestmentProblem) : ℕ :=
  let tom_investment_months : ℕ := problem.tom_investment * 12
  let jose_investment_months : ℕ := (12 - problem.jose_join_delay) * (problem.jose_profit_share * tom_investment_months) / (problem.total_profit - problem.jose_profit_share)
  jose_investment_months / (12 - problem.jose_join_delay)

/-- Theorem stating that Jose's investment is 45000 given the problem conditions --/
theorem jose_investment_is_45000 (problem : InvestmentProblem) 
  (h1 : problem.tom_investment = 30000)
  (h2 : problem.jose_join_delay = 2)
  (h3 : problem.total_profit = 54000)
  (h4 : problem.jose_profit_share = 30000) :
  calculate_jose_investment problem = 45000 := by
  sorry

end NUMINAMATH_CALUDE_jose_investment_is_45000_l91_9135


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l91_9136

theorem smallest_integer_with_remainders : ∃! x : ℕ+, 
  (x : ℕ) % 4 = 3 ∧ 
  (x : ℕ) % 3 = 2 ∧ 
  ∀ y : ℕ+, y < x → (y : ℕ) % 4 ≠ 3 ∨ (y : ℕ) % 3 ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l91_9136


namespace NUMINAMATH_CALUDE_pennsylvania_quarter_percentage_l91_9144

theorem pennsylvania_quarter_percentage 
  (total_quarters : ℕ) 
  (state_quarter_ratio : ℚ) 
  (pennsylvania_quarters : ℕ) 
  (h1 : total_quarters = 35)
  (h2 : state_quarter_ratio = 2 / 5)
  (h3 : pennsylvania_quarters = 7) :
  (pennsylvania_quarters : ℚ) / ((state_quarter_ratio * total_quarters) : ℚ) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_pennsylvania_quarter_percentage_l91_9144


namespace NUMINAMATH_CALUDE_f_has_root_in_interval_l91_9139

-- Define the function f(x) = x^3 - 3x - 3
def f (x : ℝ) : ℝ := x^3 - 3*x - 3

-- State the theorem
theorem f_has_root_in_interval : 
  ∃ x ∈ Set.Ioo 2 3, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_has_root_in_interval_l91_9139


namespace NUMINAMATH_CALUDE_running_time_ratio_l91_9196

theorem running_time_ratio :
  ∀ (danny_time steve_time : ℝ),
    danny_time = 27 →
    steve_time / 2 = danny_time / 2 + 13.5 →
    danny_time / steve_time = 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_running_time_ratio_l91_9196


namespace NUMINAMATH_CALUDE_log_expression_simplification_l91_9122

theorem log_expression_simplification :
  Real.log 16 / Real.log 4 / (Real.log (1/16) / Real.log 4) + Real.log 32 / Real.log 4 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_simplification_l91_9122


namespace NUMINAMATH_CALUDE_octal_26_is_decimal_22_l91_9194

def octal_to_decimal (octal : ℕ) : ℕ :=
  let ones_digit := octal % 10
  let eights_digit := octal / 10
  eights_digit * 8 + ones_digit

theorem octal_26_is_decimal_22 : octal_to_decimal 26 = 22 := by
  sorry

end NUMINAMATH_CALUDE_octal_26_is_decimal_22_l91_9194


namespace NUMINAMATH_CALUDE_xy_value_from_inequality_l91_9176

theorem xy_value_from_inequality (x y : ℝ) :
  2 * x - 3 ≤ Real.log (x + y + 1) + Real.log (x - y - 2) →
  x * y = -9/4 := by
sorry

end NUMINAMATH_CALUDE_xy_value_from_inequality_l91_9176


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l91_9175

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - x - 1 < 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l91_9175


namespace NUMINAMATH_CALUDE_passes_through_origin_symmetric_about_y_axis_symmetric_expression_inequality_condition_l91_9181

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*(m-1)*x - 2*m + m^2

-- Theorem 1: The graph passes through the origin when m = 0 or m = 2
theorem passes_through_origin (m : ℝ) : 
  f m 0 = 0 ↔ m = 0 ∨ m = 2 := by sorry

-- Theorem 2: The graph is symmetric about the y-axis when m = 1
theorem symmetric_about_y_axis (m : ℝ) :
  (∀ x, f m x = f m (-x)) ↔ m = 1 := by sorry

-- Theorem 3: Expression when symmetric about y-axis
theorem symmetric_expression (x : ℝ) :
  f 1 x = x^2 - 1 := by sorry

-- Theorem 4: Condition for f(x) ≥ 3 in the interval [1, 3]
theorem inequality_condition (m : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f m x ≥ 3) ↔ m ≤ 0 ∨ m ≥ 6 := by sorry

end NUMINAMATH_CALUDE_passes_through_origin_symmetric_about_y_axis_symmetric_expression_inequality_condition_l91_9181


namespace NUMINAMATH_CALUDE_cube_side_length_l91_9171

/-- The side length of a cube given paint cost and coverage -/
theorem cube_side_length 
  (paint_cost : ℝ)  -- Cost of paint per kg
  (paint_coverage : ℝ)  -- Area covered by 1 kg of paint in sq. ft
  (total_cost : ℝ)  -- Total cost to paint the cube
  (h1 : paint_cost = 40)  -- Paint costs Rs. 40 per kg
  (h2 : paint_coverage = 20)  -- 1 kg of paint covers 20 sq. ft
  (h3 : total_cost = 10800)  -- Total cost is Rs. 10800
  : ∃ (side_length : ℝ), side_length = 30 ∧ 
    total_cost = 6 * side_length^2 * paint_cost / paint_coverage :=
by sorry

end NUMINAMATH_CALUDE_cube_side_length_l91_9171


namespace NUMINAMATH_CALUDE_mark_chicken_nuggets_cost_l91_9192

-- Define the number of chicken nuggets Mark orders
def total_nuggets : ℕ := 100

-- Define the number of nuggets in a box
def nuggets_per_box : ℕ := 20

-- Define the cost of one box
def cost_per_box : ℕ := 4

-- Theorem to prove
theorem mark_chicken_nuggets_cost :
  (total_nuggets / nuggets_per_box) * cost_per_box = 20 := by
  sorry

end NUMINAMATH_CALUDE_mark_chicken_nuggets_cost_l91_9192


namespace NUMINAMATH_CALUDE_data_average_is_three_l91_9100

def data : List ℝ := [2, 3, 2, 2, 3, 6]

def is_mode (x : ℝ) (l : List ℝ) : Prop :=
  ∀ y ∈ l, (l.count x ≥ l.count y)

theorem data_average_is_three :
  is_mode 2 data →
  (data.sum / data.length : ℝ) = 3 := by
sorry

end NUMINAMATH_CALUDE_data_average_is_three_l91_9100


namespace NUMINAMATH_CALUDE_correct_selling_prices_l91_9117

-- Define the types of items
inductive Item
| Pencil
| Eraser
| Sharpener

-- Define the cost price function in A-coins
def costPriceA (item : Item) : ℝ :=
  match item with
  | Item.Pencil => 15
  | Item.Eraser => 25
  | Item.Sharpener => 35

-- Define the exchange rate
def exchangeRate : ℝ := 2

-- Define the profit percentage function
def profitPercentage (item : Item) : ℝ :=
  match item with
  | Item.Pencil => 0.20
  | Item.Eraser => 0.25
  | Item.Sharpener => 0.30

-- Define the selling price function in B-coins
def sellingPriceB (item : Item) : ℝ :=
  let costB := costPriceA item * exchangeRate
  costB + (costB * profitPercentage item)

-- Theorem to prove the selling prices are correct
theorem correct_selling_prices :
  sellingPriceB Item.Pencil = 36 ∧
  sellingPriceB Item.Eraser = 62.5 ∧
  sellingPriceB Item.Sharpener = 91 := by
  sorry

end NUMINAMATH_CALUDE_correct_selling_prices_l91_9117


namespace NUMINAMATH_CALUDE_basketball_reach_theorem_l91_9152

/-- Represents the height a basketball player can reach above their head using their arms -/
def reachAboveHead (playerHeight rimHeight jumpHeight : ℕ) : ℕ :=
  rimHeight * 12 + 6 - (playerHeight * 12 + jumpHeight)

/-- Theorem stating that a 6-foot tall player who can jump 32 inches high needs to reach 22 inches above their head to dunk on a 10-foot rim -/
theorem basketball_reach_theorem :
  reachAboveHead 6 10 32 = 22 := by
  sorry

end NUMINAMATH_CALUDE_basketball_reach_theorem_l91_9152


namespace NUMINAMATH_CALUDE_fraction_equality_l91_9159

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (3 * x + y) / (x - 3 * y) = -2) : 
  (x + 3 * y) / (3 * x - y) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l91_9159


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l91_9184

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_same_line 
  (a : Line) (α β : Plane) :
  perpendicular a α → perpendicular a β → parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l91_9184


namespace NUMINAMATH_CALUDE_correct_proposition_l91_9174

theorem correct_proposition :
  (∀ a b : ℝ, a > |b| → a^2 > b^2) ∧
  (∃ a b c : ℝ, a > b ∧ ¬(a*c^2 > b*c^2)) ∧
  (∃ a b : ℝ, a > b ∧ ¬(a^2 > b^2)) ∧
  (∃ a b : ℝ, |a| > b ∧ ¬(a^2 > b^2)) :=
by sorry

end NUMINAMATH_CALUDE_correct_proposition_l91_9174


namespace NUMINAMATH_CALUDE_octagon_diagonals_l91_9195

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

theorem octagon_diagonals :
  num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l91_9195


namespace NUMINAMATH_CALUDE_same_terminal_side_angle_l91_9199

theorem same_terminal_side_angle :
  ∃ α : ℝ, 0 ≤ α ∧ α < 360 ∧ ∃ k : ℤ, α = k * 360 - 30 ∧ α = 330 := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_angle_l91_9199


namespace NUMINAMATH_CALUDE_tangent_line_problem_l91_9116

theorem tangent_line_problem (a : ℝ) :
  (∃ (m : ℝ), 
    (∀ x y : ℝ, y = x^3 → (y - 0 = m * (x - 1) → x = 1 ∨ (y - x^3) = 3 * x^2 * (x - x))) ∧
    (∀ x y : ℝ, y = a * x^2 + (15/4) * x - 9 → (y - 0 = m * (x - 1) → x = 1 ∨ (y - (a * x^2 + (15/4) * x - 9)) = (2 * a * x + 15/4) * (x - x))))
  → a = -1 ∨ a = -25/64 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l91_9116


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l91_9155

theorem inequality_system_solution_set (x : ℝ) : 
  (2/3 * (2*x + 5) > 2 ∧ x - 2 < 0) ↔ (-1 < x ∧ x < 2) := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l91_9155


namespace NUMINAMATH_CALUDE_sum_of_coordinates_D_l91_9119

/-- Given a line segment CD with midpoint M(6,6) and endpoint C(2,10),
    prove that the sum of coordinates of the other endpoint D is 12. -/
theorem sum_of_coordinates_D (C D M : ℝ × ℝ) : 
  C = (2, 10) →
  M = (6, 6) →
  M.1 = (C.1 + D.1) / 2 →
  M.2 = (C.2 + D.2) / 2 →
  D.1 + D.2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_D_l91_9119


namespace NUMINAMATH_CALUDE_distance_between_points_l91_9179

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (-3, 4)
  let p2 : ℝ × ℝ := (4, -5)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 130 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l91_9179


namespace NUMINAMATH_CALUDE_absolute_value_equation_l91_9145

theorem absolute_value_equation (x : ℝ) : 
  |2*x - 1| = Real.sqrt 2 - 1 → x = Real.sqrt 2 / 2 ∨ x = (2 - Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l91_9145


namespace NUMINAMATH_CALUDE_jellybeans_left_in_jar_l91_9168

theorem jellybeans_left_in_jar
  (total_jellybeans : ℕ)
  (total_kids : ℕ)
  (absent_kids : ℕ)
  (jellybeans_per_kid : ℕ)
  (h1 : total_jellybeans = 100)
  (h2 : total_kids = 24)
  (h3 : absent_kids = 2)
  (h4 : jellybeans_per_kid = 3) :
  total_jellybeans - (total_kids - absent_kids) * jellybeans_per_kid = 34 :=
by
  sorry


end NUMINAMATH_CALUDE_jellybeans_left_in_jar_l91_9168


namespace NUMINAMATH_CALUDE_onions_left_on_scale_l91_9124

/-- Represents the problem of calculating the number of onions left on a scale. -/
def OnionProblem (initial_count : ℕ) (total_weight : ℝ) (removed_count : ℕ) (remaining_avg_weight : ℝ) (removed_avg_weight : ℝ) : Prop :=
  let remaining_count := initial_count - removed_count
  let total_weight_grams := total_weight * 1000
  let remaining_weight := remaining_count * remaining_avg_weight
  let removed_weight := removed_count * removed_avg_weight
  (remaining_weight + removed_weight = total_weight_grams) ∧
  (remaining_count = 35)

/-- Theorem stating that given the problem conditions, 35 onions are left on the scale. -/
theorem onions_left_on_scale :
  OnionProblem 40 7.68 5 190 206 :=
by
  sorry

end NUMINAMATH_CALUDE_onions_left_on_scale_l91_9124


namespace NUMINAMATH_CALUDE_optimal_distribution_theorem_l91_9180

/-- Represents the total value of the estate in talents -/
def estate_value : ℚ := 210

/-- Represents the fraction of the estate allocated to the son if only a son is born -/
def son_fraction : ℚ := 2/3

/-- Represents the fraction of the estate allocated to the daughter if only a daughter is born -/
def daughter_fraction : ℚ := 1/3

/-- Represents the optimal fraction of the estate allocated to the son when twins are born -/
def optimal_son_fraction : ℚ := 4/7

/-- Represents the optimal fraction of the estate allocated to the daughter when twins are born -/
def optimal_daughter_fraction : ℚ := 1/7

/-- Represents the optimal fraction of the estate allocated to the mother when twins are born -/
def optimal_mother_fraction : ℚ := 2/7

/-- Theorem stating that the optimal distribution is the best approximation of the will's conditions -/
theorem optimal_distribution_theorem :
  optimal_son_fraction + optimal_daughter_fraction + optimal_mother_fraction = 1 ∧
  optimal_son_fraction * estate_value + 
  optimal_daughter_fraction * estate_value + 
  optimal_mother_fraction * estate_value = estate_value ∧
  optimal_son_fraction > optimal_daughter_fraction ∧
  optimal_son_fraction < son_fraction ∧
  optimal_daughter_fraction < daughter_fraction :=
sorry

end NUMINAMATH_CALUDE_optimal_distribution_theorem_l91_9180


namespace NUMINAMATH_CALUDE_regular_quad_pyramid_angle_relation_l91_9172

/-- Regular quadrilateral pyramid -/
structure RegularQuadPyramid where
  /-- Dihedral angle between a lateral face and the base -/
  α : ℝ
  /-- Dihedral angle between two adjacent lateral faces -/
  β : ℝ

/-- Theorem: In a regular quadrilateral pyramid, 2cosβ + cos2α = -1 -/
theorem regular_quad_pyramid_angle_relation (p : RegularQuadPyramid) :
  2 * Real.cos p.β + Real.cos (2 * p.α) = -1 := by
  sorry

end NUMINAMATH_CALUDE_regular_quad_pyramid_angle_relation_l91_9172


namespace NUMINAMATH_CALUDE_x_intercept_of_specific_line_l91_9182

/-- The x-intercept of a line is a point on the x-axis where the line intersects it. -/
def x_intercept (a b c : ℚ) : ℚ × ℚ :=
  (c / a, 0)

/-- The line equation is of the form ax + by = c, where a, b, and c are rational numbers. -/
structure LineEquation where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Theorem: The x-intercept of the line 4x - 3y = 24 is the point (6, 0). -/
theorem x_intercept_of_specific_line :
  let line : LineEquation := { a := 4, b := -3, c := 24 }
  x_intercept line.a line.b line.c = (6, 0) := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_specific_line_l91_9182


namespace NUMINAMATH_CALUDE_boat_current_rate_l91_9130

/-- Proves that given a boat with a speed of 20 km/hr in still water,
    traveling 9.6 km downstream in 24 minutes, the rate of the current is 4 km/hr. -/
theorem boat_current_rate (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 20 →
  downstream_distance = 9.6 →
  downstream_time = 24 / 60 →
  ∃ (current_rate : ℝ),
    current_rate = 4 ∧
    downstream_distance = (boat_speed + current_rate) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_boat_current_rate_l91_9130


namespace NUMINAMATH_CALUDE_sum_and_count_theorem_l91_9150

def sum_of_range (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_in_range (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_theorem :
  sum_of_range 30 50 + count_even_in_range 30 50 = 851 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_count_theorem_l91_9150


namespace NUMINAMATH_CALUDE_ball_motion_problem_l91_9148

/-- Ball motion problem -/
theorem ball_motion_problem 
  (dist_A_to_wall : ℝ) 
  (dist_wall_to_B : ℝ) 
  (dist_AB : ℝ) 
  (initial_velocity : ℝ) 
  (acceleration : ℝ) 
  (h1 : dist_A_to_wall = 5)
  (h2 : dist_wall_to_B = 2)
  (h3 : dist_AB = 9)
  (h4 : initial_velocity = 5)
  (h5 : acceleration = -0.4) :
  ∃ (return_speed : ℝ) (required_initial_speed : ℝ),
    return_speed = 3 ∧ required_initial_speed = 4 := by
  sorry


end NUMINAMATH_CALUDE_ball_motion_problem_l91_9148


namespace NUMINAMATH_CALUDE_opposite_numbers_iff_differ_in_sign_l91_9134

/-- Two real numbers are opposite if and only if they differ only in their sign -/
theorem opposite_numbers_iff_differ_in_sign (a b : ℝ) : 
  (a = -b) ↔ (abs a = abs b) := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_iff_differ_in_sign_l91_9134


namespace NUMINAMATH_CALUDE_sin_45_degrees_l91_9198

theorem sin_45_degrees : Real.sin (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l91_9198


namespace NUMINAMATH_CALUDE_simplify_expression_l91_9173

theorem simplify_expression (t : ℝ) (h : t ≠ 0) : (t^5 * t^7) / t^3 = t^9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l91_9173


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l91_9133

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  (∀ n, a (n + 1) = a n * q) →
  a 3 + a 4 + a 5 = 84 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l91_9133


namespace NUMINAMATH_CALUDE_notebook_length_l91_9163

/-- Given a rectangular notebook with area 1.77 cm² and width 3 cm, prove its length is 0.59 cm -/
theorem notebook_length (area : ℝ) (width : ℝ) (length : ℝ) :
  area = 1.77 ∧ width = 3 ∧ area = length * width → length = 0.59 := by
  sorry

end NUMINAMATH_CALUDE_notebook_length_l91_9163


namespace NUMINAMATH_CALUDE_farm_tax_calculation_l91_9161

/-- Represents the farm tax calculation for a village and an individual landowner. -/
theorem farm_tax_calculation 
  (total_tax : ℝ) 
  (individual_land_ratio : ℝ) 
  (h1 : total_tax = 3840) 
  (h2 : individual_land_ratio = 0.5) : 
  individual_land_ratio * total_tax = 1920 := by
  sorry


end NUMINAMATH_CALUDE_farm_tax_calculation_l91_9161


namespace NUMINAMATH_CALUDE_range_of_a_l91_9123

-- Define the inequality
def inequality (x a : ℝ) : Prop := x^2 - 2*x + 3 ≤ a^2 - 2*a - 1

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | inequality x a}

-- Theorem statement
theorem range_of_a : 
  (∀ a : ℝ, solution_set a = ∅) ↔ (∀ a : ℝ, -1 < a ∧ a < 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l91_9123


namespace NUMINAMATH_CALUDE_odd_numbers_mean_contradiction_l91_9137

theorem odd_numbers_mean_contradiction (a b c d e f g : ℤ) :
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧ f < g ∧  -- Ordered
  Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ Odd e ∧ Odd f ∧ Odd g ∧  -- All odd
  (a + b + c + d + e + f + g) / 7 - d = 3 / 7  -- Mean minus middle equals 3/7
  → False := by sorry

end NUMINAMATH_CALUDE_odd_numbers_mean_contradiction_l91_9137


namespace NUMINAMATH_CALUDE_sum_of_numeric_values_l91_9118

/-- The numeric value assigned to a letter based on its position in the alphabet. -/
def letterValue (n : ℕ) : ℤ :=
  match n % 8 with
  | 1 => 1
  | 2 => 2
  | 3 => 1
  | 4 => 0
  | 5 => -1
  | 6 => -2
  | 7 => -1
  | 0 => 0
  | _ => 0  -- This case should never occur, but Lean requires it for completeness

/-- The positions of the letters in "numeric" in the alphabet. -/
def numericPositions : List ℕ := [14, 21, 13, 5, 18, 9, 3]

/-- The theorem stating that the sum of the numeric values of the letters in "numeric" is -1. -/
theorem sum_of_numeric_values :
  (numericPositions.map letterValue).sum = -1 := by
  sorry

#eval (numericPositions.map letterValue).sum

end NUMINAMATH_CALUDE_sum_of_numeric_values_l91_9118


namespace NUMINAMATH_CALUDE_parallelogram_existence_l91_9167

/-- Represents a cell in the table -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents a table with marked cells -/
structure Table where
  size : Nat
  markedCells : Finset Cell

/-- Represents a parallelogram in the table -/
structure Parallelogram where
  v1 : Cell
  v2 : Cell
  v3 : Cell
  v4 : Cell

/-- Checks if a cell is within the table bounds -/
def Cell.isValid (c : Cell) (n : Nat) : Prop :=
  c.row < n ∧ c.col < n

/-- Checks if a parallelogram is valid (all vertices are marked and form a parallelogram) -/
def Parallelogram.isValid (p : Parallelogram) (t : Table) : Prop :=
  p.v1 ∈ t.markedCells ∧ p.v2 ∈ t.markedCells ∧ p.v3 ∈ t.markedCells ∧ p.v4 ∈ t.markedCells ∧
  (p.v1.row - p.v2.row = p.v4.row - p.v3.row) ∧
  (p.v1.col - p.v2.col = p.v4.col - p.v3.col)

/-- Main theorem: In an n × n table with 2n marked cells, there exists a valid parallelogram -/
theorem parallelogram_existence (t : Table) (h1 : t.markedCells.card = 2 * t.size) :
  ∃ p : Parallelogram, p.isValid t :=
sorry

end NUMINAMATH_CALUDE_parallelogram_existence_l91_9167


namespace NUMINAMATH_CALUDE_corner_sum_is_164_l91_9178

/-- Represents a 9x9 grid filled with numbers 1 to 81 in row-major order -/
def Grid := Fin 9 → Fin 9 → Nat

/-- The value at position (i, j) in the grid -/
def gridValue (i j : Fin 9) : Nat :=
  9 * i.val + j.val + 1

/-- The grid filled with numbers 1 to 81 -/
def numberGrid : Grid :=
  λ i j => gridValue i j

/-- The sum of the numbers in the four corners of the grid -/
def cornerSum (g : Grid) : Nat :=
  g 0 0 + g 0 8 + g 8 0 + g 8 8

theorem corner_sum_is_164 :
  cornerSum numberGrid = 164 := by sorry

end NUMINAMATH_CALUDE_corner_sum_is_164_l91_9178


namespace NUMINAMATH_CALUDE_coursework_materials_theorem_l91_9128

def total_budget : ℝ := 1000

def food_percentage : ℝ := 0.30
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.25

def coursework_materials_spending : ℝ := 
  total_budget * (1 - (food_percentage + accommodation_percentage + entertainment_percentage))

theorem coursework_materials_theorem : 
  coursework_materials_spending = 300 := by sorry

end NUMINAMATH_CALUDE_coursework_materials_theorem_l91_9128


namespace NUMINAMATH_CALUDE_trigonometric_identity_l91_9103

theorem trigonometric_identity (α : ℝ) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) : 
  Real.sin (α - π / 6) ^ 2 - Real.cos (5 * π / 6 + α) = (2 + Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l91_9103


namespace NUMINAMATH_CALUDE_triangle_OAB_area_and_point_C_l91_9191

-- Define points in 2D space
def O : Fin 2 → ℝ := ![0, 0]
def A : Fin 2 → ℝ := ![2, 4]
def B : Fin 2 → ℝ := ![6, -2]

-- Define the area of a triangle given three points
def triangleArea (p1 p2 p3 : Fin 2 → ℝ) : ℝ := sorry

-- Define a function to check if two line segments are parallel
def isParallel (p1 p2 p3 p4 : Fin 2 → ℝ) : Prop := sorry

-- Define a function to calculate the length of a line segment
def segmentLength (p1 p2 : Fin 2 → ℝ) : ℝ := sorry

theorem triangle_OAB_area_and_point_C :
  (triangleArea O A B = 14) ∧
  (∃ (C : Fin 2 → ℝ), (C = ![4, -6] ∨ C = ![8, 2]) ∧
                      isParallel O A B C ∧
                      segmentLength O A = segmentLength B C) := by
  sorry

end NUMINAMATH_CALUDE_triangle_OAB_area_and_point_C_l91_9191


namespace NUMINAMATH_CALUDE_parabola_decreasing_range_l91_9169

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem parabola_decreasing_range :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → (f x₁ > f x₂ ↔ x₁ < 1 ∧ x₂ < 1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_decreasing_range_l91_9169


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l91_9106

-- Define the polynomial
def P (x : ℝ) : ℝ := (4 * x^2 - 4 * x + 3)^4 * (4 + 3 * x - 3 * x^2)^2

-- Theorem statement
theorem sum_of_coefficients :
  (P 1) = 1296 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l91_9106


namespace NUMINAMATH_CALUDE_chromosome_replication_not_in_prophase_i_l91_9166

-- Define the events that can occur during cell division
inductive CellDivisionEvent
  | ChromosomeReplication
  | ChromosomeShortening
  | HomologousPairing
  | CrossingOver

-- Define the phases of meiosis
inductive MeiosisPhase
  | Interphase
  | ProphaseI
  | OtherPhases

-- Define a function that determines if an event occurs in a given phase
def occurs_in (event : CellDivisionEvent) (phase : MeiosisPhase) : Prop := sorry

-- State the theorem
theorem chromosome_replication_not_in_prophase_i :
  occurs_in CellDivisionEvent.ChromosomeReplication MeiosisPhase.Interphase →
  occurs_in CellDivisionEvent.ChromosomeShortening MeiosisPhase.ProphaseI →
  occurs_in CellDivisionEvent.HomologousPairing MeiosisPhase.ProphaseI →
  occurs_in CellDivisionEvent.CrossingOver MeiosisPhase.ProphaseI →
  ¬ occurs_in CellDivisionEvent.ChromosomeReplication MeiosisPhase.ProphaseI :=
by
  sorry

end NUMINAMATH_CALUDE_chromosome_replication_not_in_prophase_i_l91_9166


namespace NUMINAMATH_CALUDE_arithmetic_sequence_13th_term_l91_9121

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence (α : Type*) [Field α] where
  a : ℕ → α
  d : α
  h_nonzero : d ≠ 0
  h_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

/-- The 13th term of the arithmetic sequence is 28 -/
theorem arithmetic_sequence_13th_term
  {α : Type*} [Field α] (seq : ArithmeticSequence α)
  (h_geometric : seq.a 9 * seq.a 1 = (seq.a 5)^2)
  (h_sum : seq.a 1 + 3 * seq.a 5 + seq.a 9 = 20) :
  seq.a 13 = 28 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_13th_term_l91_9121


namespace NUMINAMATH_CALUDE_min_distance_squared_l91_9177

/-- Given real numbers a, b, c, and d satisfying certain conditions,
    the minimum value of (a-c)^2 + (b-d)^2 is 1. -/
theorem min_distance_squared (a b c d : ℝ) 
  (h1 : Real.log (b + 1) + a - 3 * b = 0)
  (h2 : 2 * d - c + Real.sqrt 5 = 0) : 
  ∃ (min_val : ℝ), min_val = 1 ∧ 
  ∀ (a' b' c' d' : ℝ), 
    Real.log (b' + 1) + a' - 3 * b' = 0 → 
    2 * d' - c' + Real.sqrt 5 = 0 → 
    (a' - c')^2 + (b' - d')^2 ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_distance_squared_l91_9177


namespace NUMINAMATH_CALUDE_joey_study_time_l91_9143

/-- Calculates the total study time for Joey's SAT exam preparation --/
def total_study_time (weekday_hours_per_night : ℕ) (weekday_nights : ℕ) 
  (weekend_hours_per_day : ℕ) (weekend_days : ℕ) (weeks_until_exam : ℕ) : ℕ :=
  ((weekday_hours_per_night * weekday_nights + weekend_hours_per_day * weekend_days) * weeks_until_exam)

/-- Proves that Joey will spend 96 hours studying for his SAT exam --/
theorem joey_study_time : 
  total_study_time 2 5 3 2 6 = 96 := by
  sorry

end NUMINAMATH_CALUDE_joey_study_time_l91_9143


namespace NUMINAMATH_CALUDE_divisible_by_five_l91_9160

theorem divisible_by_five (B : ℕ) : 
  B < 10 → (5270 + B) % 5 = 0 ↔ B = 0 ∨ B = 5 := by sorry

end NUMINAMATH_CALUDE_divisible_by_five_l91_9160


namespace NUMINAMATH_CALUDE_intersection_point_k_value_l91_9151

/-- Given two lines that intersect at x = -12, prove that k = 65 -/
theorem intersection_point_k_value :
  ∀ (y : ℝ),
  -3 * (-12) + y = k →
  0.75 * (-12) + y = 20 →
  k = 65 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_k_value_l91_9151


namespace NUMINAMATH_CALUDE_rate_of_interest_l91_9153

/-- Simple interest formula -/
def simple_interest (principal rate time : ℝ) : ℝ := principal * rate * time

/-- Given conditions -/
def principal : ℝ := 400
def interest : ℝ := 160
def time : ℝ := 2

/-- Theorem: The rate of interest is 0.2 given the conditions -/
theorem rate_of_interest :
  ∃ (rate : ℝ), simple_interest principal rate time = interest ∧ rate = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_rate_of_interest_l91_9153


namespace NUMINAMATH_CALUDE_no_quadratic_polynomials_with_special_roots_l91_9109

theorem no_quadratic_polynomials_with_special_roots : 
  ¬ ∃ (a b c : ℝ), a ≠ 0 ∧ 
  ∃ (x y : ℝ), x + y = -b / a ∧ x * y = c / a ∧
  ((x = a + b + c ∧ y = a * b * c) ∨ (y = a + b + c ∧ x = a * b * c)) :=
sorry

end NUMINAMATH_CALUDE_no_quadratic_polynomials_with_special_roots_l91_9109


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l91_9132

theorem solution_set_of_inequality (x : ℝ) : 
  (3 * x + 1) * (1 - 2 * x) > 0 ↔ -1/3 < x ∧ x < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l91_9132


namespace NUMINAMATH_CALUDE_hyperbola_midpoint_existence_l91_9120

theorem hyperbola_midpoint_existence : ∃ (x₁ y₁ x₂ y₂ : ℝ),
  (x₁^2 - y₁^2/9 = 1) ∧
  (x₂^2 - y₂^2/9 = 1) ∧
  ((x₁ + x₂)/2 = -1) ∧
  ((y₁ + y₂)/2 = -4) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_midpoint_existence_l91_9120


namespace NUMINAMATH_CALUDE_seventh_power_sum_l91_9115

theorem seventh_power_sum (α₁ α₂ α₃ : ℂ) 
  (h1 : α₁ + α₂ + α₃ = 2)
  (h2 : α₁^2 + α₂^2 + α₃^2 = 6)
  (h3 : α₁^3 + α₂^3 + α₃^3 = 14) :
  α₁^7 + α₂^7 + α₃^7 = 478 := by
  sorry

end NUMINAMATH_CALUDE_seventh_power_sum_l91_9115


namespace NUMINAMATH_CALUDE_alcohol_solution_volume_l91_9164

/-- Given an initial solution with volume V and 5% alcohol concentration,
    adding 5.5 liters of alcohol and 4.5 liters of water results in
    a new solution with 15% alcohol concentration if and only if
    the initial volume V is 40 liters. -/
theorem alcohol_solution_volume (V : ℝ) : 
  (0.15 * (V + 10) = 0.05 * V + 5.5) ↔ V = 40 := by sorry

end NUMINAMATH_CALUDE_alcohol_solution_volume_l91_9164


namespace NUMINAMATH_CALUDE_pancake_theorem_l91_9193

/-- The fraction of pancakes that could be flipped -/
def flipped_fraction : ℚ := 4 / 5

/-- The fraction of flipped pancakes that didn't burn -/
def not_burnt_fraction : ℚ := 51 / 100

/-- The fraction of edible pancakes that weren't dropped -/
def not_dropped_fraction : ℚ := 5 / 6

/-- The percentage of pancakes Anya could offer her family -/
def offered_percentage : ℚ := flipped_fraction * not_burnt_fraction * not_dropped_fraction * 100

theorem pancake_theorem : 
  ∃ (ε : ℚ), abs (offered_percentage - 34) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_pancake_theorem_l91_9193


namespace NUMINAMATH_CALUDE_stuart_initial_marbles_l91_9110

def betty_initial_marbles : ℕ := 150
def tom_initial_marbles : ℕ := 30
def susan_initial_marbles : ℕ := 20
def stuart_final_marbles : ℕ := 80

def marbles_to_tom : ℕ := (betty_initial_marbles * 20) / 100
def marbles_to_susan : ℕ := (betty_initial_marbles * 10) / 100
def marbles_to_stuart : ℕ := (betty_initial_marbles * 40) / 100

theorem stuart_initial_marbles :
  stuart_final_marbles - marbles_to_stuart = 20 :=
by sorry

end NUMINAMATH_CALUDE_stuart_initial_marbles_l91_9110


namespace NUMINAMATH_CALUDE_smallest_debate_club_size_l91_9125

/-- Represents the number of students in each grade --/
structure GradeCount where
  eighth : ℕ
  sixth : ℕ
  seventh : ℕ
  ninth : ℕ

/-- Checks if the given counts satisfy the ratio conditions --/
def satisfiesRatios (counts : GradeCount) : Prop :=
  7 * counts.sixth = 4 * counts.eighth ∧
  6 * counts.seventh = 5 * counts.eighth ∧
  9 * counts.ninth = 2 * counts.eighth

/-- Calculates the total number of students --/
def totalStudents (counts : GradeCount) : ℕ :=
  counts.eighth + counts.sixth + counts.seventh + counts.ninth

/-- Theorem stating that the smallest number of students satisfying the ratios is 331 --/
theorem smallest_debate_club_size :
  ∀ counts : GradeCount,
    satisfiesRatios counts →
    totalStudents counts ≥ 331 ∧
    ∃ counts' : GradeCount, satisfiesRatios counts' ∧ totalStudents counts' = 331 :=
by sorry

end NUMINAMATH_CALUDE_smallest_debate_club_size_l91_9125


namespace NUMINAMATH_CALUDE_D_72_l91_9170

/-- D(n) represents the number of ways of expressing the positive integer n 
    as a product of integers greater than 1, where the order matters. -/
def D (n : ℕ) : ℕ := sorry

/-- Theorem stating that D(72) is equal to 103 -/
theorem D_72 : D 72 = 103 := by sorry

end NUMINAMATH_CALUDE_D_72_l91_9170


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l91_9140

theorem simplify_trig_expression (x : ℝ) :
  (Real.sqrt 2 / 4) * Real.sin (π / 4 - x) + (Real.sqrt 6 / 4) * Real.cos (π / 4 - x) =
  (Real.sqrt 2 / 2) * Real.sin (7 * π / 12 - x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l91_9140


namespace NUMINAMATH_CALUDE_volume_maximized_at_10cm_l91_9157

/-- The volume of a lidless container made from a rectangular sheet -/
def containerVolume (sheetLength sheetWidth height : ℝ) : ℝ :=
  (sheetLength - 2 * height) * (sheetWidth - 2 * height) * height

/-- The statement that the volume is maximized at a specific height -/
theorem volume_maximized_at_10cm (sheetLength sheetWidth : ℝ) 
  (hLength : sheetLength = 90) 
  (hWidth : sheetWidth = 48) :
  ∃ (maxHeight : ℝ), maxHeight = 10 ∧ 
  ∀ (h : ℝ), 0 < h → h < 24 → 
  containerVolume sheetLength sheetWidth h ≤ containerVolume sheetLength sheetWidth maxHeight :=
sorry

end NUMINAMATH_CALUDE_volume_maximized_at_10cm_l91_9157


namespace NUMINAMATH_CALUDE_derivative_f_at_3_l91_9185

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem derivative_f_at_3 : 
  deriv f 3 = 6 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_3_l91_9185


namespace NUMINAMATH_CALUDE_three_lines_plane_count_l91_9149

/-- Represents a line in 3D space -/
structure Line3D where
  -- We don't need to define the internal structure of a line for this problem

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- We don't need to define the internal structure of a plane for this problem

/-- Predicate to check if a line intersects another line -/
def intersects (l1 l2 : Line3D) : Prop :=
  sorry

/-- Function to determine the number of planes formed by three lines -/
def num_planes_formed (l1 l2 l3 : Line3D) : ℕ :=
  sorry

/-- Theorem stating that three lines, where one intersects the other two,
    can form either 1, 2, or 3 planes -/
theorem three_lines_plane_count 
  (l1 l2 l3 : Line3D) 
  (h1 : intersects l1 l2) 
  (h2 : intersects l1 l3) : 
  let n := num_planes_formed l1 l2 l3
  n = 1 ∨ n = 2 ∨ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_three_lines_plane_count_l91_9149


namespace NUMINAMATH_CALUDE_triangle_right_angled_l91_9127

theorem triangle_right_angled (A B C : ℝ) (h : A - C = B) : A = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_right_angled_l91_9127


namespace NUMINAMATH_CALUDE_area_of_triangle_range_of_sum_a_c_l91_9186

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 2 * Real.sqrt 3 ∧
  Real.sqrt 3 * Real.cos t.B = t.b * Real.sin t.C

-- Theorem 1: Area of triangle ABC
theorem area_of_triangle (t : Triangle) (h : triangle_conditions t) (ha : t.a = 2) :
  (1/2) * t.a * t.c * Real.sin t.B = 2 * Real.sqrt 3 :=
sorry

-- Theorem 2: Range of a + c
theorem range_of_sum_a_c (t : Triangle) (h : triangle_conditions t) (acute : t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧ t.A + t.B + t.C = π) :
  2 * Real.sqrt 3 < t.a + t.c ∧ t.a + t.c ≤ 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_area_of_triangle_range_of_sum_a_c_l91_9186


namespace NUMINAMATH_CALUDE_definite_integral_2x_minus_3x_squared_l91_9183

theorem definite_integral_2x_minus_3x_squared : 
  ∫ x in (0 : ℝ)..2, (2 * x - 3 * x^2) = -4 := by sorry

end NUMINAMATH_CALUDE_definite_integral_2x_minus_3x_squared_l91_9183


namespace NUMINAMATH_CALUDE_multiple_condition_l91_9138

theorem multiple_condition (n : ℕ+) : 
  (∃ k : ℕ, 3^n.val + 5^n.val = k * (3^(n.val - 1) + 5^(n.val - 1))) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_multiple_condition_l91_9138


namespace NUMINAMATH_CALUDE_soldier_arrangement_l91_9165

theorem soldier_arrangement (x : ℕ) 
  (h1 : x % 2 = 1)
  (h2 : x % 3 = 2)
  (h3 : x % 5 = 3) :
  x % 30 = 23 := by
  sorry

end NUMINAMATH_CALUDE_soldier_arrangement_l91_9165


namespace NUMINAMATH_CALUDE_stating_two_thousandth_hit_on_second_string_l91_9187

/-- Represents the number of strings on the guitar. -/
def num_strings : ℕ := 6

/-- Represents the total number of hits we're interested in. -/
def total_hits : ℕ := 2000

/-- 
Represents the string number for a given hit in the sequence.
n: The hit number
-/
def string_number (n : ℕ) : ℕ :=
  let cycle_length := 2 * num_strings - 2
  let position_in_cycle := n % cycle_length
  if position_in_cycle ≤ num_strings
  then position_in_cycle
  else 2 * num_strings - position_in_cycle

/-- 
Theorem stating that the 2000th hit lands on string number 2.
-/
theorem two_thousandth_hit_on_second_string : 
  string_number total_hits = 2 := by
  sorry

end NUMINAMATH_CALUDE_stating_two_thousandth_hit_on_second_string_l91_9187


namespace NUMINAMATH_CALUDE_motorcycle_sales_decrease_l91_9131

/-- Represents the pricing and sales of motorcycles before and after a price increase --/
structure MotorcycleSales where
  original_price : ℝ
  new_price : ℝ
  original_quantity : ℕ
  new_quantity : ℕ
  original_revenue : ℝ
  new_revenue : ℝ

/-- The theorem stating the decrease in motorcycle sales after the price increase --/
theorem motorcycle_sales_decrease (sales : MotorcycleSales) : 
  sales.new_price = sales.original_price + 1000 →
  sales.new_revenue = sales.original_revenue + 26000 →
  sales.new_revenue = 594000 →
  sales.new_quantity = 63 →
  sales.original_quantity - sales.new_quantity = 4 := by
  sorry

#check motorcycle_sales_decrease

end NUMINAMATH_CALUDE_motorcycle_sales_decrease_l91_9131


namespace NUMINAMATH_CALUDE_tangent_roots_sum_l91_9108

theorem tangent_roots_sum (α β : Real) :
  (∃ (x y : Real), x^2 + 3 * Real.sqrt 3 * x + 4 = 0 ∧ y^2 + 3 * Real.sqrt 3 * y + 4 = 0 ∧ 
   x = Real.tan α ∧ y = Real.tan β) →
  α > -π/2 ∧ α < π/2 ∧ β > -π/2 ∧ β < π/2 →
  α + β = -2*π/3 := by
sorry

end NUMINAMATH_CALUDE_tangent_roots_sum_l91_9108


namespace NUMINAMATH_CALUDE_geometric_progression_a5_l91_9147

-- Define a geometric progression
def isGeometricProgression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_progression_a5 (a : ℕ → ℝ) :
  isGeometricProgression a →
  (a 3) ^ 2 - 5 * (a 3) + 4 = 0 →
  (a 7) ^ 2 - 5 * (a 7) + 4 = 0 →
  a 5 = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_geometric_progression_a5_l91_9147


namespace NUMINAMATH_CALUDE_tan_value_from_trig_equation_l91_9190

theorem tan_value_from_trig_equation (α : ℝ) 
  (h : (Real.sin α + 2 * Real.cos α) / (5 * Real.cos α - Real.sin α) = 5/16) :
  Real.tan α = -1/3 := by sorry

end NUMINAMATH_CALUDE_tan_value_from_trig_equation_l91_9190


namespace NUMINAMATH_CALUDE_absolute_value_sin_sqrt_calculation_l91_9113

theorem absolute_value_sin_sqrt_calculation :
  |(-3 : ℝ)| + 2 * Real.sin (30 * π / 180) - Real.sqrt 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sin_sqrt_calculation_l91_9113
