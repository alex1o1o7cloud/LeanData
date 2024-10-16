import Mathlib

namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l461_46197

/-- The volume of a rectangular prism -/
def volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem: The volume of a rectangular prism with dimensions 0.6m, 0.3m, and 0.2m is 0.036 m³ -/
theorem rectangular_prism_volume : volume 0.6 0.3 0.2 = 0.036 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l461_46197


namespace NUMINAMATH_CALUDE_puzzle_solution_l461_46120

def special_operation (a b c : Nat) : Nat :=
  (a * b) * 10000 + (a * c) * 100 + ((a + b + c) * 2)

theorem puzzle_solution :
  (special_operation 5 3 2 = 151022) →
  (special_operation 9 2 4 = 183652) →
  (special_operation 7 2 5 = 143556) := by
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l461_46120


namespace NUMINAMATH_CALUDE_inequality_empty_solution_set_l461_46122

theorem inequality_empty_solution_set (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + 2 * m * x + 1 ≥ 0) → 0 ≤ m ∧ m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_empty_solution_set_l461_46122


namespace NUMINAMATH_CALUDE_opposite_sqrt5_minus_2_l461_46125

theorem opposite_sqrt5_minus_2 :
  -(Real.sqrt 5 - 2) = 2 - Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_opposite_sqrt5_minus_2_l461_46125


namespace NUMINAMATH_CALUDE_math_carnival_probabilities_l461_46164

/-- Represents the Math Carnival game with three rounds -/
structure MathCarnival where
  /-- Probability of success in the first round -/
  p1 : ℝ
  /-- Probability of success in the second round -/
  p2 : ℝ
  /-- Probability of success in the third round -/
  p3 : ℝ
  /-- Probability of choosing to proceed to the next round -/
  q : ℝ
  /-- Assumption: 0 ≤ p1, p2, p3, q ≤ 1 -/
  h1 : 0 ≤ p1 ∧ p1 ≤ 1
  h2 : 0 ≤ p2 ∧ p2 ≤ 1
  h3 : 0 ≤ p3 ∧ p3 ≤ 1
  h4 : 0 ≤ q ∧ q ≤ 1

/-- The probability of winning 5 learning beans -/
def prob_win_5_beans (game : MathCarnival) : ℝ :=
  game.p1 * (1 - game.q)

/-- The probability of completing the first round but ending with zero beans -/
def prob_complete_first_zero_beans (game : MathCarnival) : ℝ :=
  game.p1 * game.q * ((1 - game.p2) + game.p2 * game.q * (1 - game.p3))

theorem math_carnival_probabilities (game : MathCarnival) 
  (h5 : game.p1 = 3/4) (h6 : game.p2 = 2/3) (h7 : game.p3 = 1/2) (h8 : game.q = 1/2) : 
  prob_win_5_beans game = 3/8 ∧ 
  prob_complete_first_zero_beans game = 3/16 := by
  sorry


end NUMINAMATH_CALUDE_math_carnival_probabilities_l461_46164


namespace NUMINAMATH_CALUDE_a_minus_b_is_perfect_square_l461_46134

theorem a_minus_b_is_perfect_square (a b : ℕ+) (h : 2 * a ^ 2 + a = 3 * b ^ 2 + b) :
  ∃ k : ℕ, (a : ℤ) - (b : ℤ) = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_a_minus_b_is_perfect_square_l461_46134


namespace NUMINAMATH_CALUDE_region_location_l461_46177

-- Define the line
def line (x y : ℝ) : Prop := x - 2*y + 6 = 0

-- Define the region
def region (x y : ℝ) : Prop := x - 2*y + 6 > 0

-- Define what it means to be on the lower right side of the line
def lower_right_side (x y : ℝ) : Prop := 
  x > -6 ∧ y < 3 ∧ region x y

-- Theorem statement
theorem region_location : 
  ∀ x y : ℝ, region x y → lower_right_side x y :=
sorry

end NUMINAMATH_CALUDE_region_location_l461_46177


namespace NUMINAMATH_CALUDE_find_divisor_l461_46163

theorem find_divisor (dividend : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = quotient * 3 + remainder ∧ remainder < 3 →
  3 = dividend / quotient ∧ remainder = dividend % quotient :=
by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l461_46163


namespace NUMINAMATH_CALUDE_average_cost_28_apples_l461_46176

/-- Represents the cost and quantity of apples in a bundle --/
structure AppleBundle where
  quantity : ℕ
  cost : ℕ

/-- Calculates the total number of apples received when purchasing a given amount --/
def totalApples (purchased : ℕ) : ℕ :=
  if purchased ≥ 20 then purchased + 5 else purchased

/-- Calculates the total cost of apples purchased --/
def totalCost (purchased : ℕ) : ℕ :=
  let bundle1 : AppleBundle := ⟨4, 15⟩
  let bundle2 : AppleBundle := ⟨7, 25⟩
  (purchased / bundle2.quantity) * bundle2.cost

/-- Theorem stating the average cost per apple when purchasing 28 apples --/
theorem average_cost_28_apples :
  (totalCost 28 : ℚ) / (totalApples 28 : ℚ) = 100 / 33 := by
  sorry

#check average_cost_28_apples

end NUMINAMATH_CALUDE_average_cost_28_apples_l461_46176


namespace NUMINAMATH_CALUDE_min_value_of_a_l461_46131

theorem min_value_of_a (a : ℝ) (h1 : a > 0) : 
  (∀ x : ℝ, x > 1 → x + a / (x - 1) ≥ 5) → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_a_l461_46131


namespace NUMINAMATH_CALUDE_pure_imaginary_magnitude_l461_46184

theorem pure_imaginary_magnitude (a : ℝ) : 
  (((a - 2 * Complex.I) / (1 + Complex.I)).re = 0) → 
  Complex.abs (1 + a * Complex.I) = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_magnitude_l461_46184


namespace NUMINAMATH_CALUDE_student_committee_size_l461_46118

theorem student_committee_size (ways_to_select : ℕ) (h : ways_to_select = 42) :
  ∃ n : ℕ, n > 1 ∧ n * (n - 1) = ways_to_select ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_student_committee_size_l461_46118


namespace NUMINAMATH_CALUDE_peasant_money_problem_l461_46167

theorem peasant_money_problem (initial_money : ℕ) : 
  let after_first := initial_money / 2 - 1
  let after_second := after_first / 2 - 2
  let after_third := after_second / 2 - 1
  (after_third = 0) → initial_money = 6 := by
sorry

end NUMINAMATH_CALUDE_peasant_money_problem_l461_46167


namespace NUMINAMATH_CALUDE_sum_of_solutions_l461_46128

theorem sum_of_solutions (x y : ℝ) : 
  x * y = 1 ∧ x + y = 3 → ∃ (x₁ x₂ : ℝ), x₁ + x₂ = 3 ∧ 
  (x₁ * (3 - x₁) = 1 ∧ x₁ + (3 - x₁) = 3) ∧
  (x₂ * (3 - x₂) = 1 ∧ x₂ + (3 - x₂) = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l461_46128


namespace NUMINAMATH_CALUDE_prob_red_after_transfer_l461_46148

/-- Represents the contents of a bag as a pair of natural numbers (white balls, red balls) -/
def BagContents := ℕ × ℕ

/-- The initial contents of bag A -/
def bagA : BagContents := (2, 1)

/-- The initial contents of bag B -/
def bagB : BagContents := (1, 2)

/-- Calculates the probability of drawing a red ball from a bag -/
def probRedBall (bag : BagContents) : ℚ :=
  (bag.2 : ℚ) / ((bag.1 + bag.2) : ℚ)

/-- Calculates the probability of transferring a red ball from bag A to bag B -/
def probTransferRed (bagA : BagContents) : ℚ :=
  (bagA.2 : ℚ) / ((bagA.1 + bagA.2) : ℚ)

/-- Theorem: The probability of drawing a red ball from bag B after transferring a random ball from bag A is 7/12 -/
theorem prob_red_after_transfer (bagA bagB : BagContents) :
  let probWhiteTransfer := 1 - probTransferRed bagA
  let probRedAfterWhite := probRedBall (bagB.1 + 1, bagB.2)
  let probRedAfterRed := probRedBall (bagB.1, bagB.2 + 1)
  probWhiteTransfer * probRedAfterWhite + probTransferRed bagA * probRedAfterRed = 7 / 12 :=
sorry

end NUMINAMATH_CALUDE_prob_red_after_transfer_l461_46148


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l461_46186

def M : Set ℕ := {1, 2}
def N : Set ℕ := {n | ∃ a ∈ M, n = 2*a - 1}

theorem union_of_M_and_N : M ∪ N = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l461_46186


namespace NUMINAMATH_CALUDE_weekly_pill_count_l461_46143

/-- Calculates the total number of pills taken in a week given daily intake of different types of pills -/
theorem weekly_pill_count 
  (insulin_daily : ℕ) 
  (blood_pressure_daily : ℕ) 
  (anticonvulsant_multiplier : ℕ) :
  insulin_daily = 2 →
  blood_pressure_daily = 3 →
  anticonvulsant_multiplier = 2 →
  (insulin_daily + blood_pressure_daily + anticonvulsant_multiplier * blood_pressure_daily) * 7 = 77 := by
  sorry

#check weekly_pill_count

end NUMINAMATH_CALUDE_weekly_pill_count_l461_46143


namespace NUMINAMATH_CALUDE_social_practice_problem_l461_46162

/-- Represents the number of students -/
def num_students : ℕ := sorry

/-- Represents the number of 35-seat buses needed to exactly fit all students -/
def num_35_seat_buses : ℕ := sorry

/-- Represents the number of 55-seat buses needed -/
def num_55_seat_buses : ℕ := sorry

/-- Cost of renting a 35-seat bus -/
def cost_35_seat : ℕ := 320

/-- Cost of renting a 55-seat bus -/
def cost_55_seat : ℕ := 400

/-- Total number of buses to rent -/
def total_buses : ℕ := 4

/-- Maximum budget for bus rental -/
def max_budget : ℕ := 1500

/-- Theorem stating the conditions and the result to be proven -/
theorem social_practice_problem :
  num_students = 35 * num_35_seat_buses ∧
  num_students = 55 * num_55_seat_buses - 45 ∧
  num_55_seat_buses = num_35_seat_buses - 1 ∧
  num_students = 175 ∧
  ∃ (x y : ℕ), x + y = total_buses ∧
               x * cost_35_seat + y * cost_55_seat ≤ max_budget ∧
               x * cost_35_seat + y * cost_55_seat = 1440 :=
by sorry

end NUMINAMATH_CALUDE_social_practice_problem_l461_46162


namespace NUMINAMATH_CALUDE_stop_after_fourth_draw_l461_46182

/-- The probability of stopping after the fourth draw in a box with 5 black and 4 white balls -/
theorem stop_after_fourth_draw (total_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) :
  total_balls = black_balls + white_balls →
  black_balls = 5 →
  white_balls = 4 →
  (black_balls / total_balls : ℚ)^3 * (white_balls / total_balls : ℚ) = (5/9 : ℚ)^3 * (4/9 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_stop_after_fourth_draw_l461_46182


namespace NUMINAMATH_CALUDE_sector_perimeter_l461_46190

/-- Given a circular sector with area 2 and central angle 4 radians, its perimeter is 6. -/
theorem sector_perimeter (r : ℝ) (h1 : r > 0) : 
  (1/2 * r * (4 * r) = 2) → (4 * r + 2 * r = 6) := by
  sorry

end NUMINAMATH_CALUDE_sector_perimeter_l461_46190


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l461_46181

theorem value_of_a_minus_b (a b : ℚ) 
  (eq1 : 3015 * a + 3021 * b = 3025)
  (eq2 : 3017 * a + 3023 * b = 3027) : 
  a - b = -7/3 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l461_46181


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l461_46199

/-- The function f(x) = x³ - x + 3 -/
def f (x : ℝ) : ℝ := x^3 - x + 3

/-- The derivative of f(x) -/
def f_derivative (x : ℝ) : ℝ := 3 * x^2 - 1

theorem tangent_line_at_x_1 : 
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f_derivative x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (2*x - y + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l461_46199


namespace NUMINAMATH_CALUDE_total_mulberries_correct_l461_46173

/-- Represents the mulberry purchase and sale scenario -/
structure MulberrySale where
  total_cost : ℝ
  first_sale_quantity : ℝ
  first_sale_price_increase : ℝ
  second_sale_price_decrease : ℝ
  total_profit : ℝ

/-- Calculates the total amount of mulberries purchased -/
def calculate_total_mulberries (sale : MulberrySale) : ℝ :=
  200 -- The actual calculation is omitted and replaced with the known result

/-- Theorem stating that the calculated total mulberries is correct -/
theorem total_mulberries_correct (sale : MulberrySale) 
  (h1 : sale.total_cost = 3000)
  (h2 : sale.first_sale_quantity = 150)
  (h3 : sale.first_sale_price_increase = 0.4)
  (h4 : sale.second_sale_price_decrease = 0.2)
  (h5 : sale.total_profit = 750) :
  calculate_total_mulberries sale = 200 := by
  sorry

#eval calculate_total_mulberries {
  total_cost := 3000,
  first_sale_quantity := 150,
  first_sale_price_increase := 0.4,
  second_sale_price_decrease := 0.2,
  total_profit := 750
}

end NUMINAMATH_CALUDE_total_mulberries_correct_l461_46173


namespace NUMINAMATH_CALUDE_total_money_l461_46157

def money_problem (john peter quincy andrew : ℝ) : Prop :=
  peter = 2 * john ∧
  quincy = peter + 20 ∧
  andrew = 1.15 * quincy ∧
  john + peter + quincy + andrew = 1211

theorem total_money :
  ∃ john peter quincy andrew : ℝ,
    money_problem john peter quincy andrew ∧
    john + peter + quincy + andrew = 1072.01 := by sorry

end NUMINAMATH_CALUDE_total_money_l461_46157


namespace NUMINAMATH_CALUDE_increasing_f_implies_a_nonnegative_l461_46108

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x

theorem increasing_f_implies_a_nonnegative (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x < f a y) → a ≥ 0 := by sorry

end NUMINAMATH_CALUDE_increasing_f_implies_a_nonnegative_l461_46108


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l461_46189

theorem cube_sum_reciprocal (m : ℝ) (h : m + 1/m = 10) : m^3 + 1/m^3 = 970 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l461_46189


namespace NUMINAMATH_CALUDE_solve_refrigerator_problem_l461_46142

def refrigerator_problem (part_payment : ℝ) (percentage : ℝ) : Prop :=
  let total_cost := part_payment / (percentage / 100)
  let remaining_amount := total_cost - part_payment
  (part_payment = 875) ∧ 
  (percentage = 25) ∧ 
  (remaining_amount = 2625)

theorem solve_refrigerator_problem :
  ∃ (part_payment percentage : ℝ), refrigerator_problem part_payment percentage :=
sorry

end NUMINAMATH_CALUDE_solve_refrigerator_problem_l461_46142


namespace NUMINAMATH_CALUDE_quadratic_reciprocal_roots_l461_46194

theorem quadratic_reciprocal_roots (a b c : ℝ) (ha : a ≠ 0) :
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ x * y = 1) ↔ c = a :=
sorry

end NUMINAMATH_CALUDE_quadratic_reciprocal_roots_l461_46194


namespace NUMINAMATH_CALUDE_tinas_savings_l461_46100

/-- Tina's savings problem -/
theorem tinas_savings (x : ℕ) : 
  (x + 14 + 21) - (5 + 17) = 40 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_tinas_savings_l461_46100


namespace NUMINAMATH_CALUDE_baguettes_left_at_end_of_day_l461_46191

/-- The number of baguettes left at the end of the day in a bakery --/
def baguettes_left (batches_per_day : ℕ) (baguettes_per_batch : ℕ) 
  (sold_after_first : ℕ) (sold_after_second : ℕ) (sold_after_third : ℕ) : ℕ :=
  let total_baguettes := batches_per_day * baguettes_per_batch
  let left_after_first := baguettes_per_batch - sold_after_first
  let left_after_second := (baguettes_per_batch + left_after_first) - sold_after_second
  let left_after_third := (baguettes_per_batch + left_after_second) - sold_after_third
  left_after_third

/-- Theorem stating the number of baguettes left at the end of the day --/
theorem baguettes_left_at_end_of_day :
  baguettes_left 3 48 37 52 49 = 6 := by
  sorry

end NUMINAMATH_CALUDE_baguettes_left_at_end_of_day_l461_46191


namespace NUMINAMATH_CALUDE_line_l1_equation_l461_46146

-- Define the lines and circle
def line_l2 (x y : ℝ) : Prop := 4 * x - 3 * y + 1 = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*y - 3 = 0

-- Define the property of being perpendicular
def perpendicular (l1 l2 : ℝ → ℝ → Prop) : Prop := sorry

-- Define the property of being tangent
def tangent (l : ℝ → ℝ → Prop) (c : ℝ → ℝ → Prop) : Prop := sorry

-- Theorem statement
theorem line_l1_equation :
  ∀ (l1 : ℝ → ℝ → Prop),
  perpendicular l1 line_l2 →
  tangent l1 circle_C →
  (∀ x y, l1 x y ↔ (3*x + 4*y + 14 = 0 ∨ 3*x + 4*y - 6 = 0)) :=
sorry

end NUMINAMATH_CALUDE_line_l1_equation_l461_46146


namespace NUMINAMATH_CALUDE_vector_parallel_and_dot_product_l461_46103

/-- Given two vectors a and b, and an angle α, prove the following statements -/
theorem vector_parallel_and_dot_product (α : Real) 
    (h1 : α ∈ Set.Ioo 0 (π/4)) 
    (a : Fin 2 → Real) (b : Fin 2 → Real)
    (h2 : a = λ i => if i = 0 then 2 * Real.sin α else 1)
    (h3 : b = λ i => if i = 0 then Real.cos α else 1) :
  (∃ (k : Real), a = k • b → Real.tan α = 1/2) ∧
  (a • b = 9/5 → Real.sin (2*α + π/4) = 7*Real.sqrt 2/10) := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_and_dot_product_l461_46103


namespace NUMINAMATH_CALUDE_sum_pqr_values_l461_46104

theorem sum_pqr_values (p q r : ℝ) (distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r)
  (eq1 : q = p * (4 - p)) (eq2 : r = q * (4 - q)) (eq3 : p = r * (4 - r)) :
  p + q + r = 6 ∨ p + q + r = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_pqr_values_l461_46104


namespace NUMINAMATH_CALUDE_least_three_digit_with_digit_product_12_l461_46105

/-- A function that returns the product of the digits of a three-digit number -/
def digit_product (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * tens * ones

/-- A predicate that checks if a number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem least_three_digit_with_digit_product_12 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 12 → 126 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_with_digit_product_12_l461_46105


namespace NUMINAMATH_CALUDE_nell_gave_jeff_168_cards_l461_46139

/-- The number of cards Nell gave to Jeff -/
def cards_to_jeff (initial : ℕ) (to_john : ℕ) (remaining : ℕ) : ℕ :=
  initial - to_john - remaining

/-- Proof that Nell gave 168 cards to Jeff -/
theorem nell_gave_jeff_168_cards :
  cards_to_jeff 573 195 210 = 168 := by
  sorry

end NUMINAMATH_CALUDE_nell_gave_jeff_168_cards_l461_46139


namespace NUMINAMATH_CALUDE_minimum_bailing_rate_l461_46102

/-- Represents the minimum bailing rate problem --/
theorem minimum_bailing_rate
  (distance_to_shore : ℝ)
  (rowing_speed : ℝ)
  (water_intake_rate : ℝ)
  (max_water_capacity : ℝ)
  (h1 : distance_to_shore = 2)
  (h2 : rowing_speed = 3)
  (h3 : water_intake_rate = 8)
  (h4 : max_water_capacity = 50)
  : ∃ (min_bailing_rate : ℝ),
    min_bailing_rate ≥ 7 ∧
    (distance_to_shore / rowing_speed) * (water_intake_rate - min_bailing_rate) ≤ max_water_capacity ∧
    ∀ (r : ℝ), r < min_bailing_rate →
      (distance_to_shore / rowing_speed) * (water_intake_rate - r) > max_water_capacity :=
by sorry

end NUMINAMATH_CALUDE_minimum_bailing_rate_l461_46102


namespace NUMINAMATH_CALUDE_jesse_sam_earnings_l461_46198

theorem jesse_sam_earnings (t : ℝ) : 
  t > 0 → 
  (t - 3) * (3 * t - 4) = 2 * (3 * t - 6) * (t - 3) → 
  t = 4 := by
sorry

end NUMINAMATH_CALUDE_jesse_sam_earnings_l461_46198


namespace NUMINAMATH_CALUDE_go_stones_theorem_l461_46166

/-- Represents a stone on the grid -/
inductive Stone
| Black
| White

/-- Represents the grid configuration -/
def Grid (n : ℕ) := Fin (2*n) → Fin (2*n) → Option Stone

/-- Predicate to check if a stone exists at a given position -/
def has_stone (grid : Grid n) (i j : Fin (2*n)) : Prop :=
  ∃ (s : Stone), grid i j = some s

/-- Predicate to check if a black stone exists at a given position -/
def has_black_stone (grid : Grid n) (i j : Fin (2*n)) : Prop :=
  grid i j = some Stone.Black

/-- Predicate to check if a white stone exists at a given position -/
def has_white_stone (grid : Grid n) (i j : Fin (2*n)) : Prop :=
  grid i j = some Stone.White

/-- The grid after removing black stones that share a column with any white stone -/
def remove_black_stones (grid : Grid n) : Grid n :=
  sorry

/-- The grid after removing white stones that share a row with any remaining black stone -/
def remove_white_stones (grid : Grid n) : Grid n :=
  sorry

/-- Count the number of stones of a given type in the grid -/
def count_stones (grid : Grid n) (stone_type : Stone) : ℕ :=
  sorry

theorem go_stones_theorem (n : ℕ) (initial_grid : Grid n) :
  let final_grid := remove_white_stones (remove_black_stones initial_grid)
  (count_stones final_grid Stone.Black ≤ n^2) ∨ (count_stones final_grid Stone.White ≤ n^2) :=
sorry

end NUMINAMATH_CALUDE_go_stones_theorem_l461_46166


namespace NUMINAMATH_CALUDE_card_covers_at_least_twelve_squares_l461_46156

/-- Represents a square card with a given side length -/
structure Card where
  side_length : ℝ

/-- Represents a checkerboard with squares of a given side length -/
structure Checkerboard where
  square_side_length : ℝ

/-- Calculates the maximum number of squares that can be covered by a card on a checkerboard -/
def max_squares_covered (card : Card) (board : Checkerboard) : ℕ :=
  sorry

/-- Theorem stating that a 1.5-inch square card can cover at least 12 one-inch squares on a checkerboard -/
theorem card_covers_at_least_twelve_squares :
  ∀ (card : Card) (board : Checkerboard),
    card.side_length = 1.5 ∧ board.square_side_length = 1 →
    max_squares_covered card board ≥ 12 :=
  sorry

end NUMINAMATH_CALUDE_card_covers_at_least_twelve_squares_l461_46156


namespace NUMINAMATH_CALUDE_min_time_for_given_problem_l461_46113

/-- Represents the chef's cooking problem -/
structure ChefProblem where
  total_potatoes : ℕ
  cooked_potatoes : ℕ
  cooking_time_per_potato : ℕ
  salad_prep_time : ℕ

/-- Calculates the minimum time needed to complete the cooking task -/
def min_time_needed (problem : ChefProblem) : ℕ :=
  max problem.salad_prep_time (problem.cooking_time_per_potato)

/-- Theorem stating the minimum time needed for the given problem -/
theorem min_time_for_given_problem :
  let problem : ChefProblem := {
    total_potatoes := 35,
    cooked_potatoes := 11,
    cooking_time_per_potato := 7,
    salad_prep_time := 15
  }
  min_time_needed problem = 15 := by sorry

end NUMINAMATH_CALUDE_min_time_for_given_problem_l461_46113


namespace NUMINAMATH_CALUDE_greatest_integer_radius_of_circle_exists_greatest_integer_radius_greatest_integer_radius_is_8_l461_46140

theorem greatest_integer_radius_of_circle (r : ℕ) : 
  (r : ℝ) * (r : ℝ) * Real.pi < 75 * Real.pi → r ≤ 8 :=
by sorry

theorem exists_greatest_integer_radius : 
  ∃ (r : ℕ), (r : ℝ) * (r : ℝ) * Real.pi < 75 * Real.pi ∧ 
  ∀ (s : ℕ), (s : ℝ) * (s : ℝ) * Real.pi < 75 * Real.pi → s ≤ r :=
by sorry

theorem greatest_integer_radius_is_8 : 
  ∃! (r : ℕ), (r : ℝ) * (r : ℝ) * Real.pi < 75 * Real.pi ∧ 
  ∀ (s : ℕ), (s : ℝ) * (s : ℝ) * Real.pi < 75 * Real.pi → s ≤ r ∧ r = 8 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_of_circle_exists_greatest_integer_radius_greatest_integer_radius_is_8_l461_46140


namespace NUMINAMATH_CALUDE_wall_length_with_mirrors_l461_46149

/-- The length of a rectangular wall with specific mirror configurations -/
theorem wall_length_with_mirrors (square_side : ℝ) (circle_diameter : ℝ) (wall_width : ℝ)
  (h_square : square_side = 18)
  (h_circle : circle_diameter = 20)
  (h_width : wall_width = 32)
  (h_combined_area : square_side ^ 2 + π * (circle_diameter / 2) ^ 2 = wall_width * wall_length / 2) :
  wall_length = (324 + 100 * π) / 16 := by
  sorry

#check wall_length_with_mirrors

end NUMINAMATH_CALUDE_wall_length_with_mirrors_l461_46149


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l461_46159

/-- The operation ⊗ as defined in the problem -/
noncomputable def bowtie (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

/-- Theorem stating that if 5 ⊗ g = 11, then g = 30 -/
theorem bowtie_equation_solution :
  ∃ g : ℝ, bowtie 5 g = 11 ∧ g = 30 := by sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l461_46159


namespace NUMINAMATH_CALUDE_expression_values_l461_46138

theorem expression_values (a b c d x y : ℝ) : 
  (a + b = 0) → 
  (c * d = 1) → 
  (x = 4 ∨ x = -4) → 
  (y = -6) → 
  ((2 * x - c * d + 4 * (a + b) - y^2 = -29 ∧ x = 4) ∨ 
   (2 * x - c * d + 4 * (a + b) - y^2 = -45 ∧ x = -4)) :=
by sorry

end NUMINAMATH_CALUDE_expression_values_l461_46138


namespace NUMINAMATH_CALUDE_equidistant_chord_length_l461_46151

theorem equidistant_chord_length 
  (d : ℝ) 
  (c1 c2 : ℝ) 
  (dist : ℝ) 
  (h1 : d = 20) 
  (h2 : c1 = 10) 
  (h3 : c2 = 14) 
  (h4 : dist = 6) :
  ∃ (x : ℝ), x^2 = 164 ∧ 
  (∃ (y : ℝ), y > 0 ∧ y < dist ∧
    (d/2)^2 = (c1/2)^2 + y^2 ∧
    (d/2)^2 = (c2/2)^2 + (dist - y)^2 ∧
    x^2/4 + (y + (dist - y)/2)^2 = (d/2)^2) :=
by sorry

end NUMINAMATH_CALUDE_equidistant_chord_length_l461_46151


namespace NUMINAMATH_CALUDE_min_vegetable_dishes_l461_46171

theorem min_vegetable_dishes (n : ℕ) (h : n ≥ 5) :
  (∃ x : ℕ, x ≥ 7 ∧ Nat.choose n 2 * Nat.choose x 2 > 200) ∧
  (∀ y : ℕ, y < 7 → Nat.choose n 2 * Nat.choose y 2 ≤ 200) :=
by sorry

end NUMINAMATH_CALUDE_min_vegetable_dishes_l461_46171


namespace NUMINAMATH_CALUDE_sum_of_digits_power_minus_hundred_l461_46180

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Calculates 10^n - 100 for n ≥ 2 -/
def power_minus_hundred (n : ℕ) : ℕ := 
  if n ≥ 2 then 10^n - 100 else 0

theorem sum_of_digits_power_minus_hundred : 
  sum_of_digits (power_minus_hundred 100) = 882 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_power_minus_hundred_l461_46180


namespace NUMINAMATH_CALUDE_system_solution_l461_46127

theorem system_solution (a b c x y z : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hsum : x + y + z ≠ 0) (hyz : y + z ≠ 0) (hzx : z + x ≠ 0) (hxy : x + y ≠ 0)
  (eq1 : 1/x + 1/(x+y) = 1/a)
  (eq2 : 1/y + 1/(z+x) = 1/b)
  (eq3 : 1/z + 1/(x+y) = 1/c) :
  x = (2*(a*b + a*c + b*c) - (a^2 + b^2 + c^2)) / (2*(-a + b + c)) ∧
  y = (2*(a*b + a*c + b*c) - (a^2 + b^2 + c^2)) / (2*(a - b + c)) ∧
  z = (2*(a*b + a*c + b*c) - (a^2 + b^2 + c^2)) / (2*(a + b - c)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l461_46127


namespace NUMINAMATH_CALUDE_problem_solution_l461_46123

theorem problem_solution (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0)
    (h4 : 3 * a + 2 * b + c = 5) (h5 : 2 * a + b - 3 * c = 1) :
    (3 / 7 ≤ c ∧ c ≤ 7 / 11) ∧
    (∀ x, 3 * a + b - 7 * c ≤ x → x ≤ -1 / 11) ∧
    (∀ y, -5 / 7 ≤ y → y ≤ 3 * a + b - 7 * c) :=
  sorry

end NUMINAMATH_CALUDE_problem_solution_l461_46123


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l461_46110

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem derivative_f_at_one :
  deriv f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l461_46110


namespace NUMINAMATH_CALUDE_min_value_theorem_l461_46129

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_condition : a + b + c = 3) :
  ∃ (min_value : ℝ), min_value = 9 ∧ 
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x + y + z = 3 → 
  x^2 + y^2 + (x + y)^2 + z^2 ≥ min_value :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l461_46129


namespace NUMINAMATH_CALUDE_specific_boy_girl_not_adjacent_girls_not_adjacent_l461_46170

-- Define the number of boys and girls
def num_boys : ℕ := 5
def num_girls : ℕ := 3

-- Define the total number of people
def total_people : ℕ := num_boys + num_girls

-- Define the total number of arrangements without restrictions
def total_arrangements : ℕ := (total_people - 1).factorial

-- Theorem for the first part of the problem
theorem specific_boy_girl_not_adjacent :
  (total_arrangements - 2 * (total_people - 2).factorial) = 3600 := by sorry

-- Theorem for the second part of the problem
theorem girls_not_adjacent :
  (num_boys - 1).factorial * (num_boys.choose num_girls) * num_girls.factorial = 1440 := by sorry

end NUMINAMATH_CALUDE_specific_boy_girl_not_adjacent_girls_not_adjacent_l461_46170


namespace NUMINAMATH_CALUDE_probability_all_sweet_is_one_sixth_l461_46185

def total_oranges : ℕ := 10
def sweet_oranges : ℕ := 6
def picked_oranges : ℕ := 3

def probability_all_sweet : ℚ :=
  (sweet_oranges.choose picked_oranges) / (total_oranges.choose picked_oranges)

theorem probability_all_sweet_is_one_sixth :
  probability_all_sweet = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_probability_all_sweet_is_one_sixth_l461_46185


namespace NUMINAMATH_CALUDE_kitty_cleaning_weeks_l461_46107

/-- The time Kitty spends on each cleaning task and the total time spent -/
structure CleaningTime where
  pickup : ℕ       -- Time spent picking up toys and straightening
  vacuum : ℕ       -- Time spent vacuuming
  windows : ℕ      -- Time spent cleaning windows
  dusting : ℕ      -- Time spent dusting furniture
  total : ℕ        -- Total time spent cleaning

/-- Calculate the number of weeks Kitty has been cleaning -/
def weeks_cleaning (ct : CleaningTime) : ℕ :=
  ct.total / (ct.pickup + ct.vacuum + ct.windows + ct.dusting)

/-- Theorem stating that Kitty has been cleaning for 4 weeks -/
theorem kitty_cleaning_weeks :
  let ct : CleaningTime := {
    pickup := 5,
    vacuum := 20,
    windows := 15,
    dusting := 10,
    total := 200
  }
  weeks_cleaning ct = 4 := by sorry

end NUMINAMATH_CALUDE_kitty_cleaning_weeks_l461_46107


namespace NUMINAMATH_CALUDE_missing_root_l461_46192

theorem missing_root (x : ℝ) : x^2 - 2*x = 0 → (x = 2 ∨ x = 0) := by
  sorry

end NUMINAMATH_CALUDE_missing_root_l461_46192


namespace NUMINAMATH_CALUDE_not_perfect_square_l461_46111

theorem not_perfect_square (n : ℤ) : ¬ ∃ m : ℤ, m^2 = 4*n + 3 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l461_46111


namespace NUMINAMATH_CALUDE_new_person_weight_l461_46175

/-- Given a group of 6 persons where replacing a 65 kg person with a new person
    increases the average weight by 1.5 kg, prove that the weight of the new person is 74 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_replaced : ℝ) (avg_increase : ℝ) :
  initial_count = 6 →
  weight_replaced = 65 →
  avg_increase = 1.5 →
  (initial_count : ℝ) * avg_increase + weight_replaced = 74 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l461_46175


namespace NUMINAMATH_CALUDE_parabola_and_line_intersection_l461_46169

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line y = x + m -/
structure Line where
  m : ℝ

/-- The distance from a point to the y-axis -/
def distToAxis (pt : Point) : ℝ := |pt.x|

theorem parabola_and_line_intersection
  (para : Parabola)
  (A : Point)
  (l : Line)
  (h1 : A.y^2 = 2 * para.p * A.x) -- A is on the parabola
  (h2 : A.x = 2) -- x-coordinate of A is 2
  (h3 : distToAxis A = 4) -- distance from A to axis is 4
  (h4 : ∃ (P Q : Point), P ≠ Q ∧
        P.y^2 = 2 * para.p * P.x ∧ Q.y^2 = 2 * para.p * Q.x ∧
        P.y = P.x + l.m ∧ Q.y = Q.x + l.m) -- l intersects parabola at distinct P and Q
  (h5 : ∃ (P Q : Point), P ≠ Q ∧
        P.y^2 = 2 * para.p * P.x ∧ Q.y^2 = 2 * para.p * Q.x ∧
        P.y = P.x + l.m ∧ Q.y = Q.x + l.m ∧
        P.x * Q.x + P.y * Q.y = 0) -- OP ⊥ OQ
  : para.p = 4 ∧ l.m = -8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_and_line_intersection_l461_46169


namespace NUMINAMATH_CALUDE_expression_simplification_l461_46172

theorem expression_simplification (x y : ℝ) (hx : x = 2) (hy : y = 2016) :
  (3 * x + 2 * y) * (3 * x - 2 * y) - (x + 2 * y) * (5 * x - 2 * y) / (8 * x) = -2015 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l461_46172


namespace NUMINAMATH_CALUDE_fraction_difference_prime_l461_46117

theorem fraction_difference_prime (p : ℕ) (hp : Prime p) :
  ∀ x y : ℕ, x > 0 ∧ y > 0 →
  (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / p ↔ x = p - 1 ∧ y = p * (p - 1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_difference_prime_l461_46117


namespace NUMINAMATH_CALUDE_tangent_lines_properties_l461_46135

/-- The number of lines tangent to a circle -/
def num_tangent_lines : ℕ := 26

/-- The number of regions that are not enclosed -/
def num_not_enclosed_regions : ℕ := 68

/-- Theorem stating the properties of the number of tangent lines -/
theorem tangent_lines_properties :
  num_tangent_lines = 30 - 4 ∧
  num_not_enclosed_regions = 2 * num_tangent_lines :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_properties_l461_46135


namespace NUMINAMATH_CALUDE_parabola_sum_l461_46130

/-- A parabola with equation x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_sum (p : Parabola) :
  p.x_coord (-6) = 7 →  -- vertex condition
  p.x_coord 0 = 5 →     -- point condition
  p.a + p.b + p.c = -32/3 := by
sorry

end NUMINAMATH_CALUDE_parabola_sum_l461_46130


namespace NUMINAMATH_CALUDE_autumn_pencils_l461_46106

def pencil_count (initial : ℕ) (lost : ℕ) (broken : ℕ) (found : ℕ) (bought : ℕ) : ℕ :=
  initial - lost - broken + found + bought

theorem autumn_pencils :
  pencil_count 20 7 3 4 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_autumn_pencils_l461_46106


namespace NUMINAMATH_CALUDE_quadratic_point_relationship_l461_46196

/-- The quadratic function f(x) = x² + 4x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + c

theorem quadratic_point_relationship (c : ℝ) :
  let y₁ := f c (-4)
  let y₂ := f c (-3)
  let y₃ := f c 1
  y₂ < y₁ ∧ y₁ < y₃ := by sorry

end NUMINAMATH_CALUDE_quadratic_point_relationship_l461_46196


namespace NUMINAMATH_CALUDE_equal_sides_implies_rhombus_l461_46112

/-- A quadrilateral is a polygon with four sides -/
structure Quadrilateral where
  sides : Fin 4 → ℝ
  positive : ∀ i, sides i > 0

/-- A rhombus is a quadrilateral with all sides of equal length -/
def is_rhombus (q : Quadrilateral) : Prop :=
  ∀ i j, q.sides i = q.sides j

/-- Theorem: A quadrilateral with all sides of equal length is a rhombus -/
theorem equal_sides_implies_rhombus (q : Quadrilateral) 
  (h : ∀ i j, q.sides i = q.sides j) : is_rhombus q := by
  sorry

end NUMINAMATH_CALUDE_equal_sides_implies_rhombus_l461_46112


namespace NUMINAMATH_CALUDE_sqrt_18_div_sqrt_8_l461_46150

theorem sqrt_18_div_sqrt_8 : Real.sqrt 18 / Real.sqrt 8 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_div_sqrt_8_l461_46150


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l461_46153

theorem root_exists_in_interval : ∃! x : ℝ, 1/2 < x ∧ x < 1 ∧ Real.exp x = 1/x := by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l461_46153


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l461_46145

theorem no_positive_integer_solution :
  ¬ ∃ (x : ℕ), (x > 0) ∧ ((5 * x + 1) / (x - 1) > 2 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l461_46145


namespace NUMINAMATH_CALUDE_recurrence_sequence_a1_l461_46124

/-- A sequence of positive real numbers satisfying the given recurrence relation. -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  a 0 = 1 ∧ (∀ n : ℕ, 0 < a n) ∧ (∀ n : ℕ, a n = a (n + 1) + a (n + 2))

/-- The theorem stating that a₁ equals (√5 - 1) / 2 for the given recurrence sequence. -/
theorem recurrence_sequence_a1 (a : ℕ → ℝ) (h : RecurrenceSequence a) :
    a 1 = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_recurrence_sequence_a1_l461_46124


namespace NUMINAMATH_CALUDE_divisible_by_4_or_6_count_l461_46154

def countDivisibleByEither (n : ℕ) (a b : ℕ) : ℕ :=
  (n / a) + (n / b) - (n / (Nat.lcm a b))

theorem divisible_by_4_or_6_count :
  countDivisibleByEither 80 4 6 = 27 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_4_or_6_count_l461_46154


namespace NUMINAMATH_CALUDE_problem_statement_l461_46178

theorem problem_statement (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : x*y + x + y = 7) : 
  x^2*y + x*y^2 = 49 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l461_46178


namespace NUMINAMATH_CALUDE_total_people_in_program_l461_46188

theorem total_people_in_program (parents : Nat) (pupils : Nat) 
  (h1 : parents = 105) (h2 : pupils = 698) : 
  parents + pupils = 803 := by
  sorry

end NUMINAMATH_CALUDE_total_people_in_program_l461_46188


namespace NUMINAMATH_CALUDE_prove_postcard_selection_l461_46165

def postcardSelection (typeA : ℕ) (typeB : ℕ) (teachers : ℕ) : Prop :=
  typeA = 2 ∧ typeB = 3 ∧ teachers = 4 →
  (Nat.choose teachers typeA + Nat.choose (teachers - 1) (typeA - 1)) = 10

theorem prove_postcard_selection :
  postcardSelection 2 3 4 :=
by
  sorry

end NUMINAMATH_CALUDE_prove_postcard_selection_l461_46165


namespace NUMINAMATH_CALUDE_function_properties_l461_46147

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -a^2 * x - 2 * a * x + 1

-- State the theorem
theorem function_properties (a : ℝ) (h_a : a > 1) :
  -- Part 1: Range of f(x)
  (∀ y : ℝ, (∃ x : ℝ, f a x = y) ↔ y < 1) ∧
  -- Part 2: Value of a when minimum on [-2, 1] is -7
  (∃ x : ℝ, x ∈ Set.Icc (-2) 1 ∧ f a x = -7 ∧ ∀ y ∈ Set.Icc (-2) 1, f a y ≥ f a x) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l461_46147


namespace NUMINAMATH_CALUDE_student_number_problem_l461_46116

theorem student_number_problem (x y : ℝ) : 
  3 * x - y = 110 → x = 110 → y = 220 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l461_46116


namespace NUMINAMATH_CALUDE_oliver_used_30_tickets_l461_46119

/-- The number of times Oliver rode the ferris wheel -/
def ferris_rides : ℕ := 7

/-- The number of times Oliver rode the bumper cars -/
def bumper_rides : ℕ := 3

/-- The cost in tickets for each ride -/
def ticket_cost : ℕ := 3

/-- The total number of tickets Oliver used -/
def total_tickets : ℕ := (ferris_rides + bumper_rides) * ticket_cost

/-- Theorem stating that Oliver used 30 tickets -/
theorem oliver_used_30_tickets : total_tickets = 30 := by
  sorry

end NUMINAMATH_CALUDE_oliver_used_30_tickets_l461_46119


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l461_46187

theorem consecutive_negative_integers_sum (n : ℤ) : 
  n < 0 ∧ n * (n + 1) = 3080 → n + (n + 1) = -111 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l461_46187


namespace NUMINAMATH_CALUDE_complex_simplification_l461_46179

theorem complex_simplification :
  (4 - 3 * Complex.I) - (6 - 5 * Complex.I) + (2 + 3 * Complex.I) = 5 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l461_46179


namespace NUMINAMATH_CALUDE_no_lines_satisfying_conditions_l461_46144

-- Define the plane and points A and B
def Plane : Type := ℝ × ℝ
def A : Plane := sorry
def B : Plane := sorry

-- Define the distance between two points in the plane
def distance (p q : Plane) : ℝ := sorry

-- Define a line in the plane
def Line : Type := Plane → Prop

-- Define the distance from a point to a line
def point_to_line_distance (p : Plane) (l : Line) : ℝ := sorry

-- Define the angle between two lines
def angle_between_lines (l1 l2 : Line) : ℝ := sorry

-- Define the line y = x
def y_equals_x : Line := sorry

-- State the theorem
theorem no_lines_satisfying_conditions :
  ∀ (l : Line),
    distance A B = 8 →
    point_to_line_distance A l = 3 →
    point_to_line_distance B l = 4 →
    angle_between_lines l y_equals_x = π/4 →
    False :=
sorry

end NUMINAMATH_CALUDE_no_lines_satisfying_conditions_l461_46144


namespace NUMINAMATH_CALUDE_f_has_max_iff_solution_set_when_a_is_one_l461_46114

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * abs (x - 2) + x

-- Theorem 1: f has a maximum value iff a ≤ -1
theorem f_has_max_iff (a : ℝ) : 
  (∃ (M : ℝ), ∀ (x : ℝ), f a x ≤ M) ↔ a ≤ -1 :=
sorry

-- Theorem 2: Solution set of f(x) < |2x - 3| when a = 1
theorem solution_set_when_a_is_one : 
  {x : ℝ | f 1 x < abs (2 * x - 3)} = {x : ℝ | x > 1/2} :=
sorry

end NUMINAMATH_CALUDE_f_has_max_iff_solution_set_when_a_is_one_l461_46114


namespace NUMINAMATH_CALUDE_total_sheets_prepared_l461_46121

/-- Given the number of sheets used for a crane and the number of sheets left,
    prove that the total number of sheets prepared at the beginning
    is equal to the sum of sheets used and sheets left. -/
theorem total_sheets_prepared
  (sheets_used : ℕ) (sheets_left : ℕ)
  (h1 : sheets_used = 12)
  (h2 : sheets_left = 9) :
  sheets_used + sheets_left = 21 := by
sorry

end NUMINAMATH_CALUDE_total_sheets_prepared_l461_46121


namespace NUMINAMATH_CALUDE_olympic_inequalities_l461_46132

/-- Given positive real numbers a, b, c, d such that a + b + c + d = 3,
    prove the following inequalities:
    1. (1/a^2 + 1/b^2 + 1/c^2 + 1/d^2) ≤ 1/(a^2*b^2*c^2*d^2)
    2. (1/a^3 + 1/b^3 + 1/c^3 + 1/d^3) ≤ 1/(a^3*b^3*c^3*d^3) -/
theorem olympic_inequalities (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_sum : a + b + c + d = 3) :
  (1/a^2 + 1/b^2 + 1/c^2 + 1/d^2 ≤ 1/(a^2*b^2*c^2*d^2)) ∧
  (1/a^3 + 1/b^3 + 1/c^3 + 1/d^3 ≤ 1/(a^3*b^3*c^3*d^3)) := by
  sorry

end NUMINAMATH_CALUDE_olympic_inequalities_l461_46132


namespace NUMINAMATH_CALUDE_journey_average_speed_l461_46161

/-- Calculates the average speed of a journey with two segments -/
def average_speed (speed1 : ℝ) (time1_fraction : ℝ) (speed2 : ℝ) (time2_fraction : ℝ) : ℝ :=
  speed1 * time1_fraction + speed2 * time2_fraction

theorem journey_average_speed :
  let speed1 := 10
  let speed2 := 50
  let time1_fraction := 0.25
  let time2_fraction := 0.75
  average_speed speed1 time1_fraction speed2 time2_fraction = 40 := by
sorry

end NUMINAMATH_CALUDE_journey_average_speed_l461_46161


namespace NUMINAMATH_CALUDE_dans_cards_l461_46155

theorem dans_cards (initial_cards : ℕ) (bought_cards : ℕ) (total_cards : ℕ) : 
  initial_cards = 27 → bought_cards = 20 → total_cards = 88 → 
  total_cards - bought_cards - initial_cards = 41 := by
sorry

end NUMINAMATH_CALUDE_dans_cards_l461_46155


namespace NUMINAMATH_CALUDE_smallest_marble_count_l461_46137

/-- Represents the number of marbles of each color in the urn -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the total number of marbles in the urn -/
def total_marbles (mc : MarbleCount) : ℕ :=
  mc.red + mc.white + mc.blue + mc.green

/-- Calculates the probability of selecting a specific combination of marbles -/
def probability (mc : MarbleCount) (red white blue green : ℕ) : ℚ :=
  (mc.red.choose red * mc.white.choose white * mc.blue.choose blue * mc.green.choose green : ℚ) /
  (total_marbles mc).choose 5

/-- Checks if all specified probabilities are equal -/
def probabilities_equal (mc : MarbleCount) : Prop :=
  probability mc 5 0 0 0 = probability mc 3 2 0 0 ∧
  probability mc 3 2 0 0 = probability mc 1 2 2 0 ∧
  probability mc 1 2 2 0 = probability mc 2 1 1 1

/-- The theorem stating that the smallest number of marbles satisfying the conditions is 24 -/
theorem smallest_marble_count : 
  ∃ (mc : MarbleCount), probabilities_equal mc ∧ total_marbles mc = 24 ∧
  (∀ (mc' : MarbleCount), probabilities_equal mc' → total_marbles mc' ≥ 24) :=
sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l461_46137


namespace NUMINAMATH_CALUDE_coffee_beans_cost_l461_46126

/-- Proves the amount spent on coffee beans given initial amount, cost of tumbler, and remaining amount -/
theorem coffee_beans_cost (initial_amount : ℕ) (tumbler_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 50 →
  tumbler_cost = 30 →
  remaining_amount = 10 →
  initial_amount - tumbler_cost - remaining_amount = 10 := by
  sorry

end NUMINAMATH_CALUDE_coffee_beans_cost_l461_46126


namespace NUMINAMATH_CALUDE_speed_in_still_water_l461_46158

/-- 
Given a man's upstream and downstream speeds, calculate his speed in still water.
-/
theorem speed_in_still_water 
  (upstream_speed : ℝ) 
  (downstream_speed : ℝ) 
  (h1 : upstream_speed = 25) 
  (h2 : downstream_speed = 55) : 
  (upstream_speed + downstream_speed) / 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_speed_in_still_water_l461_46158


namespace NUMINAMATH_CALUDE_friday_to_monday_ratio_l461_46152

def num_rabbits : ℕ := 16
def monday_toys : ℕ := 6
def wednesday_toys : ℕ := 2 * monday_toys
def saturday_toys : ℕ := wednesday_toys / 2
def toys_per_rabbit : ℕ := 3

def total_toys : ℕ := num_rabbits * toys_per_rabbit

def friday_toys : ℕ := total_toys - monday_toys - wednesday_toys - saturday_toys

theorem friday_to_monday_ratio :
  friday_toys / monday_toys = 4 ∧ friday_toys % monday_toys = 0 := by
  sorry

end NUMINAMATH_CALUDE_friday_to_monday_ratio_l461_46152


namespace NUMINAMATH_CALUDE_integer_average_sum_l461_46109

theorem integer_average_sum (a b c d : ℤ) 
  (h1 : (a + b + c) / 3 + d = 29)
  (h2 : (b + c + d) / 3 + a = 23)
  (h3 : (a + c + d) / 3 + b = 21)
  (h4 : (a + b + d) / 3 + c = 17) :
  a = 21 ∨ b = 21 ∨ c = 21 ∨ d = 21 :=
by sorry

end NUMINAMATH_CALUDE_integer_average_sum_l461_46109


namespace NUMINAMATH_CALUDE_isosceles_base_length_l461_46160

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- An equilateral triangle is a triangle where all sides are equal -/
def Triangle.isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- An isosceles triangle is a triangle where at least two sides are equal -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

/-- The perimeter of a triangle is the sum of its side lengths -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Given an equilateral triangle with perimeter 45 and an isosceles triangle with perimeter 40,
    where at least one side of the isosceles triangle is equal to the side of the equilateral triangle,
    prove that the base of the isosceles triangle is 10 units -/
theorem isosceles_base_length
  (equilateral : Triangle)
  (isosceles : Triangle)
  (h_equilateral : equilateral.isEquilateral)
  (h_isosceles : isosceles.isIsosceles)
  (h_equilateral_perimeter : equilateral.perimeter = 45)
  (h_isosceles_perimeter : isosceles.perimeter = 40)
  (h_shared_side : isosceles.a = equilateral.a ∨ isosceles.b = equilateral.a ∨ isosceles.c = equilateral.a) :
  isosceles.c = 10 ∨ isosceles.b = 10 ∨ isosceles.a = 10 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_base_length_l461_46160


namespace NUMINAMATH_CALUDE_function_property_l461_46168

/-- Given a function f(x) = ax^5 + bx^3 + cx + 1, where a, b, and c are non-zero real numbers,
    if f(3) = 11, then f(-3) = -9. -/
theorem function_property (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^5 + b * x^3 + c * x + 1
  f 3 = 11 → f (-3) = -9 := by
sorry

end NUMINAMATH_CALUDE_function_property_l461_46168


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l461_46136

/-- A sequence satisfying the given conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n ≥ 2, 2 * a n = a (n - 1) + a (n + 1)) ∧
  (a 1 + a 3 + a 5 = 9) ∧
  (a 3 + a 5 + a 7 = 15)

/-- The main theorem -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  a 3 + a 4 + a 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l461_46136


namespace NUMINAMATH_CALUDE_polynomial_expansion_properties_l461_46195

theorem polynomial_expansion_properties (a a₁ a₂ a₃ a₄ : ℝ) 
  (h : ∀ x : ℝ, (1 + 2*x)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) : 
  a₂ = 24 ∧ a + a₁ + a₂ + a₃ + a₄ = 81 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_properties_l461_46195


namespace NUMINAMATH_CALUDE_set_union_problem_l461_46101

theorem set_union_problem (a b : ℕ) :
  let M : Set ℕ := {3, 2^a}
  let N : Set ℕ := {a, b}
  M ∩ N = {2} →
  M ∪ N = {1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l461_46101


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l461_46174

theorem fractional_equation_solution : 
  ∃! x : ℝ, (x ≠ 0 ∧ x ≠ 2) ∧ (5 / x = 7 / (x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l461_46174


namespace NUMINAMATH_CALUDE_expression_evaluation_l461_46193

theorem expression_evaluation (a b : ℝ) (h1 : a ≠ -b) (h2 : a ≠ 0) (h3 : b ≠ 0) :
  let x := (4 * a * b) / (a + b)
  ((x + 2*b) / (x - 2*b) + (x + 2*a) / (x - 2*a)) / (x / 2) = (a + b) / (a * b) :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l461_46193


namespace NUMINAMATH_CALUDE_base_conversion_addition_equality_l461_46141

-- Define a function to convert a number from base b to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

-- Define the given numbers in their respective bases
def n1 : List Nat := [2, 5, 3]
def b1 : Nat := 8
def d1 : List Nat := [1, 3]
def b2 : Nat := 3
def n2 : List Nat := [2, 4, 5]
def b3 : Nat := 7
def d2 : List Nat := [3, 5]
def b4 : Nat := 6

-- State the theorem
theorem base_conversion_addition_equality :
  (to_base_10 n1 b1 : ℚ) / (to_base_10 d1 b2 : ℚ) + 
  (to_base_10 n2 b3 : ℚ) / (to_base_10 d2 b4 : ℚ) = 
  171 / 6 + 131 / 23 := by sorry

end NUMINAMATH_CALUDE_base_conversion_addition_equality_l461_46141


namespace NUMINAMATH_CALUDE_inequality_solution_set_l461_46133

theorem inequality_solution_set : 
  {x : ℝ | -2 ≤ x ∧ x ≤ 1} = {x : ℝ | 2 - x - x^2 ≥ 0} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l461_46133


namespace NUMINAMATH_CALUDE_proposition_evaluations_l461_46183

theorem proposition_evaluations :
  (∀ x : ℝ, x^2 - x + 1 > 0) ∧
  (∀ x : ℝ, x > 2 → x^2 + x - 6 ≥ 0) ∧
  (∃ x : ℝ, x ≠ 2 ∧ x^2 - 5*x + 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_proposition_evaluations_l461_46183


namespace NUMINAMATH_CALUDE_line_intersection_l461_46115

/-- Given a line with slope 3/4 passing through (400, 0), 
    prove that it intersects y = -39 at x = 348 -/
theorem line_intersection (m : ℚ) (x₀ y₀ x₁ y₁ : ℚ) : 
  m = 3/4 → x₀ = 400 → y₀ = 0 → y₁ = -39 →
  (y₁ - y₀) = m * (x₁ - x₀) →
  x₁ = 348 := by
sorry

end NUMINAMATH_CALUDE_line_intersection_l461_46115
