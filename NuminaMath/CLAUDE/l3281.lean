import Mathlib

namespace NUMINAMATH_CALUDE_value_of_y_l3281_328157

theorem value_of_y : ∃ y : ℚ, (3 * y) / 7 = 15 ∧ y = 35 := by sorry

end NUMINAMATH_CALUDE_value_of_y_l3281_328157


namespace NUMINAMATH_CALUDE_end_zeros_imply_n_greater_than_seven_l3281_328159

theorem end_zeros_imply_n_greater_than_seven (m n : ℕ+) 
  (h1 : m > n) 
  (h2 : (22220038 ^ m.val - 22220038 ^ n.val) % (10 ^ 8) = 0) : 
  n > 7 := by
  sorry

end NUMINAMATH_CALUDE_end_zeros_imply_n_greater_than_seven_l3281_328159


namespace NUMINAMATH_CALUDE_panda_increase_l3281_328163

/-- Represents the number of animals in the zoo -/
structure ZooPopulation where
  cheetahs : ℕ
  pandas : ℕ

/-- The ratio of cheetahs to pandas is 1:3 -/
def valid_ratio (pop : ZooPopulation) : Prop :=
  3 * pop.cheetahs = pop.pandas

theorem panda_increase (old_pop new_pop : ZooPopulation) :
  valid_ratio old_pop →
  valid_ratio new_pop →
  new_pop.cheetahs = old_pop.cheetahs + 2 →
  new_pop.pandas = old_pop.pandas + 6 := by
  sorry

end NUMINAMATH_CALUDE_panda_increase_l3281_328163


namespace NUMINAMATH_CALUDE_sum_of_tenth_powers_l3281_328141

theorem sum_of_tenth_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_tenth_powers_l3281_328141


namespace NUMINAMATH_CALUDE_cricket_run_rate_theorem_l3281_328172

/-- Represents a cricket game scenario --/
structure CricketGame where
  totalOvers : ℕ
  firstPartOvers : ℕ
  firstPartRunRate : ℚ
  target : ℕ

/-- Calculates the required run rate for the remaining overs --/
def requiredRunRate (game : CricketGame) : ℚ :=
  let remainingOvers := game.totalOvers - game.firstPartOvers
  let firstPartRuns := game.firstPartRunRate * game.firstPartOvers
  let remainingRuns := game.target - firstPartRuns
  remainingRuns / remainingOvers

/-- Theorem stating the required run rate for the given scenario --/
theorem cricket_run_rate_theorem (game : CricketGame) 
    (h1 : game.totalOvers = 50)
    (h2 : game.firstPartOvers = 10)
    (h3 : game.firstPartRunRate = 3.4)
    (h4 : game.target = 282) :
  requiredRunRate game = 6.2 := by
  sorry

#eval requiredRunRate {
  totalOvers := 50,
  firstPartOvers := 10,
  firstPartRunRate := 3.4,
  target := 282
}

end NUMINAMATH_CALUDE_cricket_run_rate_theorem_l3281_328172


namespace NUMINAMATH_CALUDE_blue_candy_count_l3281_328162

theorem blue_candy_count (total red : ℕ) (h1 : total = 3409) (h2 : red = 145) :
  total - red = 3264 := by
  sorry

end NUMINAMATH_CALUDE_blue_candy_count_l3281_328162


namespace NUMINAMATH_CALUDE_burning_time_3x5_rectangle_l3281_328180

/-- Represents a rectangle of toothpicks -/
structure ToothpickRectangle where
  rows : Nat
  cols : Nat
  burnTime : Nat  -- Time to burn one toothpick

/-- Calculates the burning time for a ToothpickRectangle -/
def burningTime (rect : ToothpickRectangle) : Nat :=
  let maxDim := max rect.rows rect.cols
  (maxDim - 1) * rect.burnTime + 5

theorem burning_time_3x5_rectangle :
  let rect : ToothpickRectangle := {
    rows := 3,
    cols := 5,
    burnTime := 10
  }
  burningTime rect = 65 := by sorry

end NUMINAMATH_CALUDE_burning_time_3x5_rectangle_l3281_328180


namespace NUMINAMATH_CALUDE_total_vehicles_l3281_328107

/-- Proves that the total number of vehicles on a lot is 400, given the specified conditions -/
theorem total_vehicles (total dodge hyundai kia : ℕ) : 
  dodge = total / 2 →
  hyundai = dodge / 2 →
  kia = 100 →
  total = dodge + hyundai + kia →
  total = 400 := by
sorry

end NUMINAMATH_CALUDE_total_vehicles_l3281_328107


namespace NUMINAMATH_CALUDE_wilcoxon_rank_sum_test_result_l3281_328135

def sample1 : List ℝ := [3, 4, 6, 10, 13, 17]
def sample2 : List ℝ := [1, 2, 5, 7, 16, 20, 22]

def significanceLevel : ℝ := 0.01

def calculateRankSum (sample : List ℝ) (allValues : List ℝ) : ℕ :=
  sorry

def wilcoxonRankSumTest (sample1 sample2 : List ℝ) (significanceLevel : ℝ) : Bool :=
  sorry

theorem wilcoxon_rank_sum_test_result :
  let n1 := sample1.length
  let n2 := sample2.length
  let allValues := sample1 ++ sample2
  let W1 := calculateRankSum sample1 allValues
  let Wlower := 24  -- Critical value from Wilcoxon rank-sum test table
  let Wupper := (n1 + n2 + 1) * n1 - Wlower
  Wlower < W1 ∧ W1 < Wupper ∧ wilcoxonRankSumTest sample1 sample2 significanceLevel = false :=
by
  sorry

end NUMINAMATH_CALUDE_wilcoxon_rank_sum_test_result_l3281_328135


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l3281_328183

theorem unique_quadratic_solution (a : ℝ) :
  (∃! x, a * x^2 + 2 * x - 1 = 0) → a = 0 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l3281_328183


namespace NUMINAMATH_CALUDE_product_a4_a5_a6_l3281_328110

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem product_a4_a5_a6 (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 * a 9 = 16 →
  a 1 + a 9 = 10 →
  a 4 * a 5 * a 6 = 64 := by
sorry

end NUMINAMATH_CALUDE_product_a4_a5_a6_l3281_328110


namespace NUMINAMATH_CALUDE_line_segment_length_l3281_328190

structure Line where
  points : Fin 5 → ℝ
  consecutive : ∀ i : Fin 4, points i < points (Fin.succ i)

def Line.segment (l : Line) (i j : Fin 5) : ℝ :=
  |l.points j - l.points i|

theorem line_segment_length (l : Line) 
  (h1 : l.segment 1 2 = 3 * l.segment 2 3)
  (h2 : l.segment 3 4 = 7)
  (h3 : l.segment 0 1 = 5)
  (h4 : l.segment 0 2 = 11) :
  l.segment 0 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_length_l3281_328190


namespace NUMINAMATH_CALUDE_taylor_books_l3281_328168

theorem taylor_books (candice amanda kara patricia taylor : ℕ) : 
  candice = 3 * amanda →
  kara = amanda / 2 →
  patricia = 7 * kara →
  taylor = (candice + amanda + kara + patricia) / 4 →
  candice = 18 →
  taylor = 12 := by
sorry

end NUMINAMATH_CALUDE_taylor_books_l3281_328168


namespace NUMINAMATH_CALUDE_julia_tag_total_l3281_328155

/-- The number of kids Julia played tag with on Monday -/
def monday_kids : ℕ := 16

/-- The number of kids Julia played tag with on Tuesday -/
def tuesday_kids : ℕ := 14

/-- The total number of kids Julia played tag with over two days -/
def total_kids : ℕ := monday_kids + tuesday_kids

theorem julia_tag_total : total_kids = 30 := by sorry

end NUMINAMATH_CALUDE_julia_tag_total_l3281_328155


namespace NUMINAMATH_CALUDE_correct_quantities_correct_min_discount_l3281_328165

-- Define the problem parameters
def total_cost : ℝ := 25000
def total_profit : ℝ := 11700
def ornament_cost : ℝ := 40
def ornament_price : ℝ := 58
def pendant_cost : ℝ := 30
def pendant_price : ℝ := 45
def second_profit_goal : ℝ := 10800

-- Define the quantities of ornaments and pendants
def ornaments : ℕ := 400
def pendants : ℕ := 300

-- Define the theorem for part 1
theorem correct_quantities :
  ornament_cost * ornaments + pendant_cost * pendants = total_cost ∧
  (ornament_price - ornament_cost) * ornaments + (pendant_price - pendant_cost) * pendants = total_profit :=
sorry

-- Define the minimum discount percentage
def min_discount_percentage : ℝ := 20

-- Define the theorem for part 2
theorem correct_min_discount :
  let new_pendant_price := pendant_price * (1 - min_discount_percentage / 100)
  (ornament_price - ornament_cost) * ornaments + (new_pendant_price - pendant_cost) * (2 * pendants) ≥ second_profit_goal ∧
  ∀ d : ℝ, d < min_discount_percentage →
    let price := pendant_price * (1 - d / 100)
    (ornament_price - ornament_cost) * ornaments + (price - pendant_cost) * (2 * pendants) < second_profit_goal :=
sorry

end NUMINAMATH_CALUDE_correct_quantities_correct_min_discount_l3281_328165


namespace NUMINAMATH_CALUDE_prob_white_second_given_red_first_l3281_328100

/-- The probability of drawing a white ball on the second draw, given that the first ball drawn is red -/
theorem prob_white_second_given_red_first
  (total_balls : ℕ)
  (red_balls : ℕ)
  (white_balls : ℕ)
  (h_total : total_balls = red_balls + white_balls)
  (h_red : red_balls = 5)
  (h_white : white_balls = 3) :
  (white_balls : ℚ) / (total_balls - 1) = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_white_second_given_red_first_l3281_328100


namespace NUMINAMATH_CALUDE_expression_evaluation_l3281_328112

theorem expression_evaluation :
  let x : ℚ := 2
  let y : ℚ := 5
  let z : ℚ := 3
  (3 * x^5 + 4 * y^3 + z^2) / 7 = 605 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3281_328112


namespace NUMINAMATH_CALUDE_imon_disentanglement_l3281_328130

-- Define a graph structure to represent imons and their entanglements
structure ImonGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)

-- Define the two operations
def destroyOddDegreeImon (g : ImonGraph) (v : Nat) : ImonGraph :=
  sorry

def doubleImonFamily (g : ImonGraph) : ImonGraph :=
  sorry

-- Define a predicate to check if a graph has no entanglements
def noEntanglements (g : ImonGraph) : Prop :=
  ∀ v ∈ g.vertices, ∀ w ∈ g.vertices, v ≠ w → (v, w) ∉ g.edges

-- Main theorem
theorem imon_disentanglement (g : ImonGraph) :
  ∃ (ops : List (ImonGraph → ImonGraph)), noEntanglements ((ops.foldl (· ∘ ·) id) g) :=
  sorry

end NUMINAMATH_CALUDE_imon_disentanglement_l3281_328130


namespace NUMINAMATH_CALUDE_ellipse_y_axis_iff_m_greater_n_l3281_328174

/-- The equation of an ellipse with foci on the y-axis -/
def is_ellipse_y_axis (m n : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ 
  ∀ (x y : ℝ), m * x^2 + n * y^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1

/-- The condition for m and n -/
def m_greater_n (m n : ℝ) : Prop :=
  m > n ∧ n > 0

theorem ellipse_y_axis_iff_m_greater_n (m n : ℝ) :
  is_ellipse_y_axis m n ↔ m_greater_n m n :=
sorry

end NUMINAMATH_CALUDE_ellipse_y_axis_iff_m_greater_n_l3281_328174


namespace NUMINAMATH_CALUDE_power_of_three_mod_eight_l3281_328128

theorem power_of_three_mod_eight : 3^2028 % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_eight_l3281_328128


namespace NUMINAMATH_CALUDE_p_plus_q_value_l3281_328188

theorem p_plus_q_value (p q : ℝ) 
  (hp : p^3 - 18*p^2 + 27*p - 72 = 0) 
  (hq : 10*q^3 - 75*q^2 + 50*q - 625 = 0) : 
  p + q = 2 * (180 : ℝ)^(1/3) + 43/3 := by
  sorry

end NUMINAMATH_CALUDE_p_plus_q_value_l3281_328188


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l3281_328138

-- Define the list of ages
def euler_family_ages : List ℕ := [6, 6, 9, 11, 13, 16]

-- Theorem statement
theorem euler_family_mean_age :
  let ages := euler_family_ages
  let sum_ages := ages.sum
  let num_children := ages.length
  (sum_ages : ℚ) / num_children = 61 / 6 := by sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l3281_328138


namespace NUMINAMATH_CALUDE_smallest_square_enclosing_circle_area_l3281_328104

-- Define the radius of the circle
def radius : ℝ := 5

-- Define the area of the smallest enclosing square
def smallest_enclosing_square_area (r : ℝ) : ℝ := (2 * r) ^ 2

-- Theorem statement
theorem smallest_square_enclosing_circle_area :
  smallest_enclosing_square_area radius = 100 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_enclosing_circle_area_l3281_328104


namespace NUMINAMATH_CALUDE_milk_cartons_per_stack_l3281_328147

theorem milk_cartons_per_stack (total_cartons : ℕ) (num_stacks : ℕ) 
  (h1 : total_cartons = 799)
  (h2 : num_stacks = 133)
  (h3 : total_cartons % num_stacks = 0) :
  total_cartons / num_stacks = 6 := by
  sorry

end NUMINAMATH_CALUDE_milk_cartons_per_stack_l3281_328147


namespace NUMINAMATH_CALUDE_cos_seven_pi_sixths_l3281_328115

theorem cos_seven_pi_sixths : Real.cos (7 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_sixths_l3281_328115


namespace NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l3281_328145

/-- Calculates the profit percentage given the sale price including tax, tax rate, and cost price. -/
def profit_percentage (sale_price_with_tax : ℚ) (tax_rate : ℚ) (cost_price : ℚ) : ℚ :=
  let sale_price := sale_price_with_tax / (1 + tax_rate)
  let profit := sale_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that under the given conditions, the profit percentage is approximately 4.54%. -/
theorem shopkeeper_profit_percentage :
  let sale_price_with_tax : ℚ := 616
  let tax_rate : ℚ := 1/10
  let cost_price : ℚ := 535.65
  abs (profit_percentage sale_price_with_tax tax_rate cost_price - 454/100) < 1/100 := by
  sorry

#eval profit_percentage 616 (1/10) 535.65

end NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l3281_328145


namespace NUMINAMATH_CALUDE_min_disks_for_files_l3281_328137

theorem min_disks_for_files : 
  let total_files : ℕ := 35
  let disk_capacity : ℚ := 1.44
  let files_0_6MB : ℕ := 5
  let files_0_5MB : ℕ := 18
  let files_0_3MB : ℕ := total_files - files_0_6MB - files_0_5MB
  let size_0_6MB : ℚ := 0.6
  let size_0_5MB : ℚ := 0.5
  let size_0_3MB : ℚ := 0.3
  ∀ n : ℕ, 
    (n * disk_capacity ≥ 
      files_0_6MB * size_0_6MB + 
      files_0_5MB * size_0_5MB + 
      files_0_3MB * size_0_3MB) →
    n ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_min_disks_for_files_l3281_328137


namespace NUMINAMATH_CALUDE_divisor_inequality_l3281_328134

theorem divisor_inequality (d d' n : ℕ) (h1 : d' > d) (h2 : d ∣ n) (h3 : d' ∣ n) :
  d' > d + d^2 / n :=
by sorry

end NUMINAMATH_CALUDE_divisor_inequality_l3281_328134


namespace NUMINAMATH_CALUDE_compound_interest_rate_l3281_328101

/-- Given an initial investment of $7000, invested for 2 years with annual compounding,
    resulting in a final amount of $8470, prove that the annual interest rate is 0.1 (10%). -/
theorem compound_interest_rate (P A : ℝ) (t n : ℕ) (r : ℝ) : 
  P = 7000 → A = 8470 → t = 2 → n = 1 → 
  A = P * (1 + r / n) ^ (n * t) →
  r = 0.1 := by
  sorry


end NUMINAMATH_CALUDE_compound_interest_rate_l3281_328101


namespace NUMINAMATH_CALUDE_apex_to_center_distance_for_specific_pyramid_l3281_328192

/-- Represents a rectangular pyramid with a parallel cut -/
structure CutPyramid where
  ab : ℝ
  bc : ℝ
  height : ℝ
  volumeRatio : ℝ

/-- Calculate the distance between the apex and the center of the circumsphere of the frustum -/
def apexToCenterDistance (p : CutPyramid) : ℝ :=
  sorry

/-- Theorem statement -/
theorem apex_to_center_distance_for_specific_pyramid :
  let p : CutPyramid := {
    ab := 15,
    bc := 20,
    height := 30,
    volumeRatio := 9
  }
  apexToCenterDistance p = 250 / 9 := by sorry

end NUMINAMATH_CALUDE_apex_to_center_distance_for_specific_pyramid_l3281_328192


namespace NUMINAMATH_CALUDE_round_to_nearest_integer_l3281_328193

def number : ℝ := 7293847.2635142

theorem round_to_nearest_integer : 
  Int.floor (number + 0.5) = 7293847 := by sorry

end NUMINAMATH_CALUDE_round_to_nearest_integer_l3281_328193


namespace NUMINAMATH_CALUDE_complex_power_2017_l3281_328196

theorem complex_power_2017 : 
  let z : ℂ := (1 - Complex.I) / (1 + Complex.I)
  z^2017 = -Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_power_2017_l3281_328196


namespace NUMINAMATH_CALUDE_dice_sum_multiple_of_5_prob_correct_l3281_328113

/-- The probability that the sum of n rolls of a 6-sided die is a multiple of 5 -/
def dice_sum_multiple_of_5_prob (n : ℕ) : ℚ :=
  if 5 ∣ n then
    (6^n + 4) / (5 * 6^n)
  else
    (6^n - 1) / (5 * 6^n)

/-- Theorem: The probability that the sum of n rolls of a 6-sided die is a multiple of 5
    is (6^n - 1) / (5 * 6^n) if 5 doesn't divide n, and (6^n + 4) / (5 * 6^n) if 5 divides n -/
theorem dice_sum_multiple_of_5_prob_correct (n : ℕ) :
  dice_sum_multiple_of_5_prob n =
    if 5 ∣ n then
      (6^n + 4) / (5 * 6^n)
    else
      (6^n - 1) / (5 * 6^n) := by
  sorry

end NUMINAMATH_CALUDE_dice_sum_multiple_of_5_prob_correct_l3281_328113


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l3281_328164

theorem quadratic_root_difference (C : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > x₂ ∧ x₁ - x₂ = 5.5 ∧ 2 * x₁^2 + 5 * x₁ = C ∧ 2 * x₂^2 + 5 * x₂ = C) → 
  C = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l3281_328164


namespace NUMINAMATH_CALUDE_original_price_calculation_l3281_328177

theorem original_price_calculation (original_price new_price : ℝ) : 
  new_price = 0.8 * original_price ∧ new_price = 80 → original_price = 100 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l3281_328177


namespace NUMINAMATH_CALUDE_pizza_combinations_l3281_328158

theorem pizza_combinations (n k : ℕ) (h1 : n = 8) (h2 : k = 5) : 
  Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l3281_328158


namespace NUMINAMATH_CALUDE_marc_total_spent_l3281_328143

/-- Calculates the total amount Marc spent on his purchases --/
def total_spent (model_car_price : ℝ) (paint_price : ℝ) (paintbrush_price : ℝ)
                (display_case_price : ℝ) (model_car_discount : ℝ) (paint_coupon : ℝ)
                (gift_card : ℝ) (first_tax_rate : ℝ) (second_tax_rate : ℝ) : ℝ :=
  let model_cars_cost := 5 * model_car_price * (1 - model_car_discount)
  let paint_cost := 5 * paint_price - paint_coupon
  let paintbrushes_cost := 7 * paintbrush_price
  let first_subtotal := model_cars_cost + paint_cost + paintbrushes_cost - gift_card
  let first_transaction := first_subtotal * (1 + first_tax_rate)
  let display_cases_cost := 3 * display_case_price
  let second_transaction := display_cases_cost * (1 + second_tax_rate)
  first_transaction + second_transaction

/-- Theorem stating that Marc's total spent is $187.02 --/
theorem marc_total_spent :
  total_spent 20 10 2 15 0.1 5 20 0.08 0.06 = 187.02 := by
  sorry

end NUMINAMATH_CALUDE_marc_total_spent_l3281_328143


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3281_328161

theorem arithmetic_calculation : 3521 + 480 / 60 * 3 - 521 = 3024 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3281_328161


namespace NUMINAMATH_CALUDE_bicycle_shop_period_l3281_328132

/-- Proves that the number of weeks that passed is 4, given the conditions of the bicycle shop problem. -/
theorem bicycle_shop_period (initial_stock : ℕ) (weekly_addition : ℕ) (sold : ℕ) (final_stock : ℕ)
  (h1 : initial_stock = 51)
  (h2 : weekly_addition = 3)
  (h3 : sold = 18)
  (h4 : final_stock = 45) :
  ∃ weeks : ℕ, weeks = 4 ∧ initial_stock + weekly_addition * weeks - sold = final_stock :=
by
  sorry

#check bicycle_shop_period

end NUMINAMATH_CALUDE_bicycle_shop_period_l3281_328132


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3281_328184

/-- A function that represents the cubic polynomial f(x) = x³ + 2x² + mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + m*x + 1

/-- The condition p: ∀x ∈ ℝ, x²-4x+3m > 0 -/
def condition_p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 4*x + 3*m > 0

/-- The condition q: f(x) is strictly increasing on (-∞,+∞) -/
def condition_q (m : ℝ) : Prop := StrictMono (f m)

/-- Theorem stating that p is a sufficient but not necessary condition for q -/
theorem p_sufficient_not_necessary_for_q :
  (∃ m : ℝ, condition_p m → condition_q m) ∧
  (∃ m : ℝ, condition_q m ∧ ¬condition_p m) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3281_328184


namespace NUMINAMATH_CALUDE_quadratic_inequality_region_l3281_328167

theorem quadratic_inequality_region (x y : ℝ) :
  (∀ t : ℝ, t^2 ≤ 1 → t^2 + y*t + x ≥ 0) →
  (y ≤ x + 1 ∧ y ≥ -x - 1 ∧ x ≥ y^2/4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_region_l3281_328167


namespace NUMINAMATH_CALUDE_scholarship_difference_l3281_328169

theorem scholarship_difference (nina kelly wendy : ℕ) : 
  nina < kelly →
  kelly = 2 * wendy →
  wendy = 20000 →
  nina + kelly + wendy = 92000 →
  kelly - nina = 8000 := by
sorry

end NUMINAMATH_CALUDE_scholarship_difference_l3281_328169


namespace NUMINAMATH_CALUDE_transformed_area_doubled_l3281_328108

-- Define a function representing the area under a curve
noncomputable def areaUnderCurve (f : ℝ → ℝ) (a b : ℝ) : ℝ := sorry

-- Define the original function g
variable (g : ℝ → ℝ)

-- Define the interval [a, b] over which we're measuring the area
variable (a b : ℝ)

-- Theorem statement
theorem transformed_area_doubled 
  (h : areaUnderCurve g a b = 15) : 
  areaUnderCurve (fun x ↦ 2 * g (x + 3)) a b = 30 := by
  sorry

end NUMINAMATH_CALUDE_transformed_area_doubled_l3281_328108


namespace NUMINAMATH_CALUDE_cookies_received_l3281_328109

theorem cookies_received (brother sister cousin self : ℕ) 
  (h1 : brother = 12)
  (h2 : sister = 9)
  (h3 : cousin = 7)
  (h4 : self = 17) :
  brother + sister + cousin + self = 45 := by
  sorry

end NUMINAMATH_CALUDE_cookies_received_l3281_328109


namespace NUMINAMATH_CALUDE_complex_norm_squared_not_equal_square_l3281_328191

theorem complex_norm_squared_not_equal_square : 
  ¬ ∀ (z : ℂ), (Complex.abs z)^2 = z^2 := by
  sorry

end NUMINAMATH_CALUDE_complex_norm_squared_not_equal_square_l3281_328191


namespace NUMINAMATH_CALUDE_no_divisible_by_five_append_l3281_328150

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Appends a digit to the left of 864 to form a four-digit number -/
def appendDigit (d : Digit) : Nat := d.val * 1000 + 864

/-- Theorem: There are no digits that can be appended to the left of 864
    to create a four-digit number divisible by 5 -/
theorem no_divisible_by_five_append :
  ∀ d : Digit, ¬(appendDigit d % 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_divisible_by_five_append_l3281_328150


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3281_328182

theorem smallest_n_congruence : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → (7 ^ m.val) % 3 ≠ (m.val ^ 4) % 3) ∧ 
  (7 ^ n.val) % 3 = (n.val ^ 4) % 3 ∧
  n = 1 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3281_328182


namespace NUMINAMATH_CALUDE_johns_share_l3281_328179

theorem johns_share (total_amount : ℕ) (john_ratio jose_ratio binoy_ratio : ℕ) 
  (h1 : total_amount = 6000)
  (h2 : john_ratio = 2)
  (h3 : jose_ratio = 4)
  (h4 : binoy_ratio = 6) :
  (john_ratio : ℚ) / (john_ratio + jose_ratio + binoy_ratio : ℚ) * total_amount = 1000 :=
by sorry

end NUMINAMATH_CALUDE_johns_share_l3281_328179


namespace NUMINAMATH_CALUDE_set_inclusion_implies_m_values_l3281_328136

def A (m : ℝ) : Set ℝ := {1, 3, 2*m+3}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem set_inclusion_implies_m_values (m : ℝ) : B m ⊆ A m → m = 1 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_set_inclusion_implies_m_values_l3281_328136


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3281_328126

def polynomial (x : ℝ) : ℝ :=
  3 * (2 * x^6 - x^5 + 4 * x^3 - 7) - 5 * (x^4 - 2 * x^3 + 3 * x^2 + 1) + 6 * (x^7 - 5)

theorem sum_of_coefficients :
  polynomial 1 = 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3281_328126


namespace NUMINAMATH_CALUDE_volleyball_lineup_theorem_l3281_328117

def volleyball_lineup_count (n : ℕ) (k : ℕ) (mvp_count : ℕ) (trio_count : ℕ) : ℕ :=
  Nat.choose (n - mvp_count - trio_count) (k - mvp_count - 1) * trio_count +
  Nat.choose (n - mvp_count - trio_count) (k - mvp_count - 2) * Nat.choose trio_count 2 +
  Nat.choose (n - mvp_count - trio_count) (k - mvp_count - 3) * Nat.choose trio_count 3

theorem volleyball_lineup_theorem :
  volleyball_lineup_count 15 7 2 3 = 1035 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_lineup_theorem_l3281_328117


namespace NUMINAMATH_CALUDE_special_isosceles_triangle_base_l3281_328123

/-- An isosceles triangle with a specific configuration of inscribed circles -/
structure SpecialIsoscelesTriangle where
  /-- The radius of the incircle of the triangle -/
  r₁ : ℝ
  /-- The radius of the smaller circle tangent to the incircle and congruent sides -/
  r₂ : ℝ
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- Condition that r₁ = 3 -/
  h₁ : r₁ = 3
  /-- Condition that r₂ = 2 -/
  h₂ : r₂ = 2

/-- The theorem stating that the base of the special isosceles triangle is 3√6 -/
theorem special_isosceles_triangle_base (t : SpecialIsoscelesTriangle) : t.base = 3 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_special_isosceles_triangle_base_l3281_328123


namespace NUMINAMATH_CALUDE_black_lambs_count_l3281_328189

/-- The total number of lambs -/
def total_lambs : ℕ := 6048

/-- The number of white lambs -/
def white_lambs : ℕ := 193

/-- Theorem: The number of black lambs is 5855 -/
theorem black_lambs_count : total_lambs - white_lambs = 5855 := by
  sorry

end NUMINAMATH_CALUDE_black_lambs_count_l3281_328189


namespace NUMINAMATH_CALUDE_inequality_theorem_l3281_328151

theorem inequality_theorem (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n ≥ 1) :
  x + (n^n : ℝ) / x^n ≥ n + 1 := by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3281_328151


namespace NUMINAMATH_CALUDE_translation_theorem_l3281_328125

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translation of a point in 2D space -/
def translate (p : Point) (dx dy : ℝ) : Point :=
  { x := p.x + dx, y := p.y + dy }

theorem translation_theorem :
  let A : Point := { x := -3, y := 2 }
  let A' : Point := translate (translate A 4 0) 0 (-3)
  A'.x = 1 ∧ A'.y = -1 := by sorry

end NUMINAMATH_CALUDE_translation_theorem_l3281_328125


namespace NUMINAMATH_CALUDE_symmetry_line_equation_l3281_328140

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two points are symmetric about a line -/
def symmetric_about (P Q : Point) (l : Line) : Prop :=
  -- Definition of symmetry (to be implemented)
  sorry

/-- The problem statement -/
theorem symmetry_line_equation :
  let P : Point := ⟨3, 2⟩
  let Q : Point := ⟨1, 4⟩
  let l : Line := ⟨1, -1, 1⟩  -- Represents x - y + 1 = 0
  symmetric_about P Q l → l = ⟨1, -1, 1⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetry_line_equation_l3281_328140


namespace NUMINAMATH_CALUDE_racketCostProof_l3281_328178

/-- Calculates the total cost of two rackets under a specific promotion --/
def totalCostOfRackets (fullPrice : ℚ) : ℚ :=
  fullPrice + (fullPrice / 2)

/-- Proves that the total cost of two rackets is $90 under the given conditions --/
theorem racketCostProof : totalCostOfRackets 60 = 90 := by
  sorry

end NUMINAMATH_CALUDE_racketCostProof_l3281_328178


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3281_328121

theorem sufficient_not_necessary : 
  (∃ x : ℝ, |x - 2| < 3 ∧ ¬(0 < x ∧ x < 5)) ∧
  (∀ x : ℝ, (0 < x ∧ x < 5) → |x - 2| < 3) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3281_328121


namespace NUMINAMATH_CALUDE_expression_value_l3281_328118

theorem expression_value (a b c : ℚ) (ha : a = 5) (hb : b = -3) (hc : c = 2) :
  (3 * c) / (a + b) + c = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3281_328118


namespace NUMINAMATH_CALUDE_chaperones_count_l3281_328146

/-- Calculates the number of volunteer chaperones given the number of children,
    additional lunches, cost per lunch, and total cost. -/
def calculate_chaperones (children : ℕ) (additional : ℕ) (cost_per_lunch : ℕ) (total_cost : ℕ) : ℕ :=
  (total_cost / cost_per_lunch) - children - additional - 1

/-- Theorem stating that the number of volunteer chaperones is 6 given the problem conditions. -/
theorem chaperones_count :
  let children : ℕ := 35
  let additional : ℕ := 3
  let cost_per_lunch : ℕ := 7
  let total_cost : ℕ := 308
  calculate_chaperones children additional cost_per_lunch total_cost = 6 := by
  sorry

#eval calculate_chaperones 35 3 7 308

end NUMINAMATH_CALUDE_chaperones_count_l3281_328146


namespace NUMINAMATH_CALUDE_log_equation_solution_l3281_328194

theorem log_equation_solution (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  p * (q + 1) = q →
  (Real.log p + Real.log q = Real.log (p + q) ↔ p = q / (q + 1)) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3281_328194


namespace NUMINAMATH_CALUDE_notebook_spending_l3281_328153

/-- Calculates the total amount spent on notebooks --/
def total_spent (total_notebooks : ℕ) (red_notebooks : ℕ) (green_notebooks : ℕ) 
  (red_price : ℕ) (green_price : ℕ) (blue_price : ℕ) : ℕ :=
  let blue_notebooks := total_notebooks - red_notebooks - green_notebooks
  red_notebooks * red_price + green_notebooks * green_price + blue_notebooks * blue_price

/-- Proves that the total amount spent on notebooks is $37 --/
theorem notebook_spending : 
  total_spent 12 3 2 4 2 3 = 37 := by
  sorry

end NUMINAMATH_CALUDE_notebook_spending_l3281_328153


namespace NUMINAMATH_CALUDE_final_sign_is_minus_l3281_328133

/-- Represents the two types of signs on the board -/
inductive Sign
| Plus
| Minus

/-- Represents the state of the board -/
structure Board :=
  (plus_count : ℕ)
  (minus_count : ℕ)

/-- Performs one operation on the board -/
def perform_operation (b : Board) : Board :=
  sorry

/-- Performs n operations on the board -/
def perform_n_operations (b : Board) (n : ℕ) : Board :=
  sorry

/-- The main theorem to prove -/
theorem final_sign_is_minus :
  let initial_board : Board := ⟨10, 15⟩
  let final_board := perform_n_operations initial_board 24
  final_board.plus_count = 0 ∧ final_board.minus_count = 1 :=
sorry

end NUMINAMATH_CALUDE_final_sign_is_minus_l3281_328133


namespace NUMINAMATH_CALUDE_sum_of_A_and_B_l3281_328127

theorem sum_of_A_and_B (A B : ℕ) (h1 : 3 * 7 = 7 * A) (h2 : 7 * A = B) : A + B = 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_A_and_B_l3281_328127


namespace NUMINAMATH_CALUDE_least_value_quadratic_inequality_l3281_328186

theorem least_value_quadratic_inequality :
  ∃ (x : ℝ), x = 4 ∧ (∀ y : ℝ, -y^2 + 9*y - 20 ≤ 0 → y ≥ x) := by
  sorry

end NUMINAMATH_CALUDE_least_value_quadratic_inequality_l3281_328186


namespace NUMINAMATH_CALUDE_prob_king_queen_is_16_2862_l3281_328195

/-- Represents a standard deck of cards with Jokers -/
structure Deck :=
  (total_cards : ℕ)
  (num_kings : ℕ)
  (num_queens : ℕ)
  (num_jokers : ℕ)

/-- The probability of drawing a King then a Queen from the deck -/
def prob_king_then_queen (d : Deck) : ℚ :=
  (d.num_kings * d.num_queens : ℚ) / ((d.total_cards * (d.total_cards - 1)) : ℚ)

/-- Our specific deck -/
def our_deck : Deck :=
  { total_cards := 54
  , num_kings := 4
  , num_queens := 4
  , num_jokers := 2 }

theorem prob_king_queen_is_16_2862 :
  prob_king_then_queen our_deck = 16 / 2862 := by
  sorry

end NUMINAMATH_CALUDE_prob_king_queen_is_16_2862_l3281_328195


namespace NUMINAMATH_CALUDE_solution_absolute_value_equation_l3281_328122

theorem solution_absolute_value_equation :
  ∀ x : ℝ, 2 * |x - 5| = 6 ↔ x = 2 ∨ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_solution_absolute_value_equation_l3281_328122


namespace NUMINAMATH_CALUDE_cube_volume_problem_l3281_328197

theorem cube_volume_problem (a : ℝ) (h : a > 0) :
  (a - 2) * a * (a + 2) = a^3 - 12 → a^3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l3281_328197


namespace NUMINAMATH_CALUDE_at_least_one_negative_l3281_328148

theorem at_least_one_negative (a b c d : ℝ) 
  (sum1 : a + b = 1) 
  (sum2 : c + d = 1) 
  (prod_sum : a * c + b * d > 1) : 
  (a < 0) ∨ (b < 0) ∨ (c < 0) ∨ (d < 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_negative_l3281_328148


namespace NUMINAMATH_CALUDE_farmer_milk_production_l3281_328198

/-- Calculates the total milk production for a farmer in a week -/
def total_milk_production (num_cows : ℕ) (milk_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  num_cows * milk_per_day * days_in_week

/-- Proves that a farmer with 52 cows, each producing 5 liters of milk per day,
    will get 1820 liters of milk in a week (7 days) -/
theorem farmer_milk_production :
  total_milk_production 52 5 7 = 1820 := by
  sorry

end NUMINAMATH_CALUDE_farmer_milk_production_l3281_328198


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3281_328185

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 1 + a 8 = 9) 
  (h_a4 : a 4 = 3) : 
  a 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3281_328185


namespace NUMINAMATH_CALUDE_first_number_is_five_l3281_328160

/-- A sequence where each term is obtained by adding 9 to the previous term -/
def arithmeticSequence (a₁ : ℕ) : ℕ → ℕ
  | 0 => a₁
  | n + 1 => arithmeticSequence a₁ n + 9

/-- The property that 2012 is in the sequence -/
def contains2012 (a₁ : ℕ) : Prop :=
  ∃ n : ℕ, arithmeticSequence a₁ n = 2012

theorem first_number_is_five :
  ∃ a₁ : ℕ, a₁ < 10 ∧ contains2012 a₁ ∧ a₁ = 5 :=
by sorry

end NUMINAMATH_CALUDE_first_number_is_five_l3281_328160


namespace NUMINAMATH_CALUDE_percent_equality_l3281_328106

theorem percent_equality (x : ℝ) : (60 / 100 : ℝ) * 600 = (50 / 100 : ℝ) * x → x = 720 := by
  sorry

end NUMINAMATH_CALUDE_percent_equality_l3281_328106


namespace NUMINAMATH_CALUDE_sequence_ratio_theorem_l3281_328120

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- Property of the sequence: a_{n+1} - 2a_n = 0 for all n -/
def HasConstantRatio (a : Sequence) : Prop :=
  ∀ n : ℕ, a (n + 1) - 2 * a n = 0

/-- Property of the sequence: a_n ≠ 0 for all n -/
def IsNonZero (a : Sequence) : Prop :=
  ∀ n : ℕ, a n ≠ 0

/-- The main theorem -/
theorem sequence_ratio_theorem (a : Sequence) 
  (h1 : HasConstantRatio a) (h2 : IsNonZero a) : 
  (2 * a 1 + a 2) / (a 3 + a 5) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sequence_ratio_theorem_l3281_328120


namespace NUMINAMATH_CALUDE_ellipse_slope_product_l3281_328103

theorem ellipse_slope_product (a b m n x y : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (x^2 / a^2 + y^2 / b^2 = 1) →
  (m^2 / a^2 + n^2 / b^2 = 1) →
  ((y - n) / (x - m)) * ((y + n) / (x + m)) = -b^2 / a^2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_slope_product_l3281_328103


namespace NUMINAMATH_CALUDE_min_sales_to_break_even_l3281_328102

def current_salary : ℕ := 90000
def new_base_salary : ℕ := 45000
def sale_value : ℕ := 1500
def commission_rate : ℚ := 15 / 100

theorem min_sales_to_break_even : 
  ∃ (n : ℕ), n = 200 ∧ 
  (n : ℚ) * commission_rate * sale_value + new_base_salary = current_salary ∧
  ∀ (m : ℕ), m < n → (m : ℚ) * commission_rate * sale_value + new_base_salary < current_salary :=
sorry

end NUMINAMATH_CALUDE_min_sales_to_break_even_l3281_328102


namespace NUMINAMATH_CALUDE_square_difference_identity_nine_point_five_squared_l3281_328105

theorem square_difference_identity (x : ℝ) : (10 - x)^2 = 10^2 - 2 * 10 * x + x^2 := by sorry

theorem nine_point_five_squared :
  (9.5 : ℝ)^2 = 10^2 - 2 * 10 * 0.5 + 0.5^2 := by sorry

end NUMINAMATH_CALUDE_square_difference_identity_nine_point_five_squared_l3281_328105


namespace NUMINAMATH_CALUDE_tan_negative_two_fraction_l3281_328154

theorem tan_negative_two_fraction (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_two_fraction_l3281_328154


namespace NUMINAMATH_CALUDE_additional_money_needed_l3281_328142

def water_bottles : ℕ := 5 * 12
def original_price : ℚ := 2
def reduced_price : ℚ := 185 / 100

theorem additional_money_needed :
  water_bottles * original_price - water_bottles * reduced_price = 9 := by
  sorry

end NUMINAMATH_CALUDE_additional_money_needed_l3281_328142


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l3281_328173

theorem quadratic_roots_relation (a b c : ℚ) : 
  (∃ (r s : ℚ), (4 * r^2 + 2 * r - 9 = 0) ∧ 
                 (4 * s^2 + 2 * s - 9 = 0) ∧ 
                 (a * (r - 3)^2 + b * (r - 3) + c = 0) ∧
                 (a * (s - 3)^2 + b * (s - 3) + c = 0)) →
  c = 51 / 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l3281_328173


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l3281_328114

theorem scientific_notation_equality : 0.0000012 = 1.2 * 10^(-6) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l3281_328114


namespace NUMINAMATH_CALUDE_remainder_equality_l3281_328156

/-- Represents a natural number as a list of its digits in reverse order -/
def DigitList := List Nat

/-- Converts a natural number to its digit list representation -/
def toDigitList (n : Nat) : DigitList :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : DigitList) : DigitList :=
      if m = 0 then acc
      else aux (m / 10) ((m % 10) :: acc)
    aux n []

/-- Pairs digits from right to left, allowing the leftmost pair to be a single digit -/
def pairDigits (dl : DigitList) : List Nat :=
  match dl with
  | [] => []
  | [x] => [x]
  | x :: y :: rest => (x + 10 * y) :: pairDigits rest

/-- Sums a list of natural numbers -/
def sumList (l : List Nat) : Nat := l.foldl (·+·) 0

/-- The main theorem statement -/
theorem remainder_equality (n : Nat) :
  n % 99 = (sumList (pairDigits (toDigitList n))) % 99 := by
  sorry

end NUMINAMATH_CALUDE_remainder_equality_l3281_328156


namespace NUMINAMATH_CALUDE_intercepted_segment_length_l3281_328124

-- Define the polar equations
def line_equation (p θ : ℝ) : Prop := p * Real.cos θ = 1
def circle_equation (p θ : ℝ) : Prop := p = 4 * Real.cos θ

-- Define the theorem
theorem intercepted_segment_length :
  ∃ (p₁ θ₁ p₂ θ₂ : ℝ),
    line_equation p₁ θ₁ ∧
    line_equation p₂ θ₂ ∧
    circle_equation p₁ θ₁ ∧
    circle_equation p₂ θ₂ ∧
    (p₁ * Real.cos θ₁ - p₂ * Real.cos θ₂)^2 + (p₁ * Real.sin θ₁ - p₂ * Real.sin θ₂)^2 = 12 :=
sorry

end NUMINAMATH_CALUDE_intercepted_segment_length_l3281_328124


namespace NUMINAMATH_CALUDE_tax_free_items_cost_l3281_328175

theorem tax_free_items_cost 
  (total_paid : ℝ) 
  (sales_tax : ℝ) 
  (tax_rate : ℝ) 
  (h1 : total_paid = 30)
  (h2 : sales_tax = 1.28)
  (h3 : tax_rate = 0.08) : 
  total_paid - sales_tax / tax_rate = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_tax_free_items_cost_l3281_328175


namespace NUMINAMATH_CALUDE_remainder_theorem_l3281_328131

theorem remainder_theorem : (7 * 10^23 + 3^25) % 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3281_328131


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l3281_328166

def dress_shirt_price : ℝ := 25
def pants_price : ℝ := 35
def socks_price : ℝ := 10
def dress_shirt_quantity : ℕ := 4
def pants_quantity : ℕ := 2
def socks_quantity : ℕ := 3
def dress_shirt_discount : ℝ := 0.15
def pants_discount : ℝ := 0.20
def socks_discount : ℝ := 0.10
def tax_rate : ℝ := 0.10
def shipping_fee : ℝ := 12.50

def total_cost : ℝ :=
  let dress_shirts_total := dress_shirt_price * dress_shirt_quantity * (1 - dress_shirt_discount)
  let pants_total := pants_price * pants_quantity * (1 - pants_discount)
  let socks_total := socks_price * socks_quantity * (1 - socks_discount)
  let subtotal := dress_shirts_total + pants_total + socks_total
  let tax := subtotal * tax_rate
  subtotal + tax + shipping_fee

theorem total_cost_is_correct : total_cost = 197.30 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l3281_328166


namespace NUMINAMATH_CALUDE_quadruple_pieces_sold_l3281_328144

/-- Represents the number of pieces sold for each type --/
structure PiecesSold where
  single : Nat
  double : Nat
  triple : Nat
  quadruple : Nat

/-- Calculates the total earnings in cents --/
def totalEarnings (pieces : PiecesSold) : Nat :=
  pieces.single + 2 * pieces.double + 3 * pieces.triple + 4 * pieces.quadruple

/-- The main theorem to prove --/
theorem quadruple_pieces_sold (pieces : PiecesSold) :
  pieces.single = 100 ∧ 
  pieces.double = 45 ∧ 
  pieces.triple = 50 ∧ 
  totalEarnings pieces = 1000 →
  pieces.quadruple = 165 := by
  sorry

#eval totalEarnings { single := 100, double := 45, triple := 50, quadruple := 165 }

end NUMINAMATH_CALUDE_quadruple_pieces_sold_l3281_328144


namespace NUMINAMATH_CALUDE_division_decomposition_l3281_328170

theorem division_decomposition : (36 : ℕ) / 3 = (30 / 3) + (6 / 3) := by sorry

end NUMINAMATH_CALUDE_division_decomposition_l3281_328170


namespace NUMINAMATH_CALUDE_periodic_double_period_l3281_328111

open Real

/-- A function f is a-periodic if it satisfies the given functional equation. -/
def IsPeriodic (f : ℝ → ℝ) (a : ℝ) : Prop :=
  a > 0 ∧ ∀ x, f (x + a) = 1/2 + Real.sqrt (f x - (f x)^2)

/-- If a function is a-periodic, then it is also 2a-periodic. -/
theorem periodic_double_period (f : ℝ → ℝ) (a : ℝ) (h : IsPeriodic f a) :
  ∀ x, f (x + 2*a) = f x := by
  sorry

end NUMINAMATH_CALUDE_periodic_double_period_l3281_328111


namespace NUMINAMATH_CALUDE_equation_solution_l3281_328199

theorem equation_solution : ∃ y : ℝ, (4 * y - 2) / (5 * y - 5) = 3 / 4 ∧ y = -7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3281_328199


namespace NUMINAMATH_CALUDE_point_locations_l3281_328176

def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

def is_in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem point_locations (x y : ℝ) (h : |3*x + 2| + |2*y - 1| = 0) :
  is_in_second_quadrant x y ∧ is_in_fourth_quadrant (x + 1) (y - 2) := by
  sorry

end NUMINAMATH_CALUDE_point_locations_l3281_328176


namespace NUMINAMATH_CALUDE_james_and_louise_ages_l3281_328187

/-- 
Given:
- James is five years older than Louise
- In six years, James will be three times as old as Louise was three years ago

Prove that the sum of their current ages is 25.
-/
theorem james_and_louise_ages :
  ∀ (james louise : ℕ),
  james = louise + 5 →
  james + 6 = 3 * (louise - 3) →
  james + louise = 25 :=
by sorry

end NUMINAMATH_CALUDE_james_and_louise_ages_l3281_328187


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3281_328152

/-- The fixed point of the function f(x) = 2a^(x+1) - 3, where a > 0 and a ≠ 1, is (-1, -1). -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 2 * a^(x + 1) - 3
  f (-1) = -1 ∧ ∀ x : ℝ, f x = x → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3281_328152


namespace NUMINAMATH_CALUDE_larger_number_proof_l3281_328116

theorem larger_number_proof (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3281_328116


namespace NUMINAMATH_CALUDE_flowers_per_bouquet_l3281_328119

theorem flowers_per_bouquet 
  (total_flowers : ℕ) 
  (wilted_flowers : ℕ) 
  (num_bouquets : ℕ) 
  (h1 : total_flowers = 45) 
  (h2 : wilted_flowers = 35) 
  (h3 : num_bouquets = 2) : 
  (total_flowers - wilted_flowers) / num_bouquets = 5 := by
sorry

end NUMINAMATH_CALUDE_flowers_per_bouquet_l3281_328119


namespace NUMINAMATH_CALUDE_square_root_problem_l3281_328139

theorem square_root_problem (x y : ℝ) (h : Real.sqrt (2 * x - 16) + |x - 2 * y + 2| = 0) :
  Real.sqrt (x - 4 / 5 * y) = 2 ∨ Real.sqrt (x - 4 / 5 * y) = -2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l3281_328139


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3281_328149

/-- An isosceles triangle with congruent sides of length 7 cm and perimeter 23 cm has a base of length 9 cm. -/
theorem isosceles_triangle_base_length : 
  ∀ (base congruent_side perimeter : ℝ),
  congruent_side = 7 →
  perimeter = 23 →
  perimeter = 2 * congruent_side + base →
  base = 9 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3281_328149


namespace NUMINAMATH_CALUDE_critical_point_and_zeros_l3281_328129

noncomputable section

def f (a : ℝ) (x : ℝ) := Real.exp x * Real.sin x - a * Real.log (x + 1)

def f_derivative (a : ℝ) (x : ℝ) := Real.exp x * (Real.sin x + Real.cos x) - a / (x + 1)

theorem critical_point_and_zeros (a : ℝ) :
  (f_derivative a 0 = 0 → a = 1) ∧
  ((∃ x₁ ∈ Set.Ioo (-1 : ℝ) 0, f a x₁ = 0) ∧
   (∃ x₂ ∈ Set.Ioo (Real.pi / 4) Real.pi, f a x₂ = 0) →
   0 < a ∧ a < 1) :=
by sorry

-- Given condition
axiom given_inequality : Real.sqrt 2 / 2 * Real.exp (Real.pi / 4) > 1

end NUMINAMATH_CALUDE_critical_point_and_zeros_l3281_328129


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l3281_328181

/-- A function satisfying the given functional equation. -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x * f y) + f (x + y) = f (x * y)

/-- The theorem stating that any function satisfying the functional equation
    must be one of the three specified functions. -/
theorem functional_equation_solutions (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) : 
  (∀ x, f x = 0) ∨ (∀ x, f x = x - 1) ∨ (∀ x, f x = 1 - x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l3281_328181


namespace NUMINAMATH_CALUDE_markup_discount_profit_l3281_328171

/-- Given a markup percentage and a discount percentage, calculate the profit percentage -/
def profit_percentage (markup : ℝ) (discount : ℝ) : ℝ :=
  let marked_price := 1 + markup
  let selling_price := marked_price * (1 - discount)
  (selling_price - 1) * 100

/-- Theorem stating that a 75% markup followed by a 30% discount results in a 22.5% profit -/
theorem markup_discount_profit : profit_percentage 0.75 0.3 = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_markup_discount_profit_l3281_328171
