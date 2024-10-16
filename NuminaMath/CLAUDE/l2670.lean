import Mathlib

namespace NUMINAMATH_CALUDE_mayor_harvey_flowers_l2670_267059

/-- Represents the quantities of flowers for an institution -/
structure FlowerQuantities :=
  (roses : ℕ)
  (tulips : ℕ)
  (lilies : ℕ)

/-- Calculates the total number of flowers for given quantities -/
def totalFlowers (quantities : FlowerQuantities) : ℕ :=
  quantities.roses + quantities.tulips + quantities.lilies

/-- Theorem: The total number of flowers Mayor Harvey needs to buy is 855 -/
theorem mayor_harvey_flowers :
  let nursing_home : FlowerQuantities := ⟨90, 80, 100⟩
  let shelter : FlowerQuantities := ⟨120, 75, 95⟩
  let maternity_ward : FlowerQuantities := ⟨100, 110, 85⟩
  totalFlowers nursing_home + totalFlowers shelter + totalFlowers maternity_ward = 855 :=
by
  sorry

#eval let nursing_home : FlowerQuantities := ⟨90, 80, 100⟩
      let shelter : FlowerQuantities := ⟨120, 75, 95⟩
      let maternity_ward : FlowerQuantities := ⟨100, 110, 85⟩
      totalFlowers nursing_home + totalFlowers shelter + totalFlowers maternity_ward

end NUMINAMATH_CALUDE_mayor_harvey_flowers_l2670_267059


namespace NUMINAMATH_CALUDE_chocolate_boxes_given_away_l2670_267010

theorem chocolate_boxes_given_away (total_boxes : ℕ) (pieces_per_box : ℕ) (remaining_pieces : ℕ) : 
  total_boxes = 14 → pieces_per_box = 6 → remaining_pieces = 54 → 
  (total_boxes * pieces_per_box - remaining_pieces) / pieces_per_box = 5 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_boxes_given_away_l2670_267010


namespace NUMINAMATH_CALUDE_ellipse_x_intercept_l2670_267018

/-- Definition of an ellipse with given foci and a point on it -/
def is_ellipse (f1 f2 p : ℝ × ℝ) : Prop :=
  let d1 := Real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2)
  let d2 := Real.sqrt ((p.2 - f2.1)^2 + (p.2 - f2.2)^2)
  let c := Real.sqrt ((f1.1 - f2.1)^2 + (f1.2 - f2.2)^2)
  d1 + d2 = 2 * Real.sqrt ((c/2)^2 + ((d1 - d2)/2)^2)

/-- The ellipse intersects the x-axis at (0, 0) -/
def intersects_origin (f1 f2 : ℝ × ℝ) : Prop :=
  is_ellipse f1 f2 (0, 0)

/-- Theorem: For an ellipse with foci at (0, 3) and (4, 0) that intersects
    the x-axis at (0, 0), the other x-intercept is at (56/11, 0) -/
theorem ellipse_x_intercept :
  let f1 : ℝ × ℝ := (0, 3)
  let f2 : ℝ × ℝ := (4, 0)
  intersects_origin f1 f2 →
  is_ellipse f1 f2 (56/11, 0) ∧
  ∀ x : ℝ, x ≠ 0 ∧ x ≠ 56/11 → ¬is_ellipse f1 f2 (x, 0) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_x_intercept_l2670_267018


namespace NUMINAMATH_CALUDE_tank_emptying_time_l2670_267012

/-- Represents the state and properties of a water tank system -/
structure WaterTank where
  initialFullness : ℚ
  pipeARatePerMinute : ℚ
  pipeBRatePerMinute : ℚ
  pipeCRatePerMinute : ℚ

/-- Calculates the time to empty or fill the tank given its properties -/
def timeToEmptyOrFill (tank : WaterTank) : ℚ :=
  tank.initialFullness / (tank.pipeARatePerMinute + tank.pipeBRatePerMinute + tank.pipeCRatePerMinute)

/-- Theorem stating the time to empty the specific tank configuration -/
theorem tank_emptying_time :
  let tank : WaterTank := {
    initialFullness := 7/11,
    pipeARatePerMinute := 1/15,
    pipeBRatePerMinute := -1/8,
    pipeCRatePerMinute := 1/20
  }
  timeToEmptyOrFill tank = 840/11 := by
  sorry

end NUMINAMATH_CALUDE_tank_emptying_time_l2670_267012


namespace NUMINAMATH_CALUDE_sharon_supplies_theorem_l2670_267025

def angela_pots : ℕ → ℕ := λ p => p

def angela_plates : ℕ → ℕ := λ p => 3 * p + 6

def angela_cutlery : ℕ → ℕ := λ p => (3 * p + 6) / 2

def sharon_pots : ℕ → ℕ := λ p => p / 2

def sharon_plates : ℕ → ℕ := λ p => 3 * (3 * p + 6) - 20

def sharon_cutlery : ℕ → ℕ := λ p => 3 * p + 6

def sharon_total_supplies : ℕ → ℕ := λ p => 
  sharon_pots p + sharon_plates p + sharon_cutlery p

theorem sharon_supplies_theorem (p : ℕ) : 
  p = 20 → sharon_total_supplies p = 254 := by
  sorry

end NUMINAMATH_CALUDE_sharon_supplies_theorem_l2670_267025


namespace NUMINAMATH_CALUDE_meaningful_expression_l2670_267056

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 2)) ↔ x > 2 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l2670_267056


namespace NUMINAMATH_CALUDE_smallest_sum_of_coefficients_l2670_267007

theorem smallest_sum_of_coefficients (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x : ℝ, x^2 + 2*a*x + 3*b = 0) →
  (∃ y : ℝ, y^2 + 3*b*y + 2*a = 0) →
  a + b ≥ 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_coefficients_l2670_267007


namespace NUMINAMATH_CALUDE_largest_digit_change_l2670_267099

def original_sum : ℕ := 2570
def correct_sum : ℕ := 2580
def num1 : ℕ := 725
def num2 : ℕ := 864
def num3 : ℕ := 991

theorem largest_digit_change :
  ∃ (d : ℕ), d ≤ 9 ∧ 
  (num1 + num2 + (num3 - 10) = correct_sum) ∧
  (∀ (d' : ℕ), d' > d → 
    (num1 + num2 + num3 - d' * 10 ≠ correct_sum ∧ 
     num1 + (num2 - d' * 10) + num3 ≠ correct_sum ∧
     (num1 - d' * 10) + num2 + num3 ≠ correct_sum)) :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_change_l2670_267099


namespace NUMINAMATH_CALUDE_greatest_five_digit_multiple_of_6_l2670_267044

def is_multiple_of_6 (n : ℕ) : Prop := n % 6 = 0

def uses_digits_once (n : ℕ) (digits : List ℕ) : Prop :=
  let digit_list := n.digits 10
  digit_list.length = digits.length ∧ 
  ∀ d, d ∈ digit_list ↔ d ∈ digits

theorem greatest_five_digit_multiple_of_6 :
  let digits : List ℕ := [4, 5, 7, 8, 9]
  ∀ n : ℕ, 
    n ≤ 99999 ∧ 
    10000 ≤ n ∧
    is_multiple_of_6 n ∧ 
    uses_digits_once n digits →
    n ≤ 97548 :=
by sorry

end NUMINAMATH_CALUDE_greatest_five_digit_multiple_of_6_l2670_267044


namespace NUMINAMATH_CALUDE_haleys_trees_l2670_267054

theorem haleys_trees (dead : ℕ) (survived : ℕ) : 
  dead = 6 → 
  survived = dead + 1 → 
  dead + survived = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_haleys_trees_l2670_267054


namespace NUMINAMATH_CALUDE_fence_painting_fraction_l2670_267003

theorem fence_painting_fraction (total_time : ℝ) (part_time : ℝ) 
  (h1 : total_time = 60) 
  (h2 : part_time = 12) : 
  (part_time / total_time) = (1 : ℝ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_fence_painting_fraction_l2670_267003


namespace NUMINAMATH_CALUDE_max_distance_on_C_l2670_267057

noncomputable section

open Real

-- Define the curve in polar coordinates
def C (θ : ℝ) : ℝ := 4 * sin θ

-- Define a point on the curve
def point_on_C (θ : ℝ) : ℝ × ℝ := (C θ * cos θ, C θ * sin θ)

-- Define the distance between two points on the curve
def distance_on_C (θ₁ θ₂ : ℝ) : ℝ :=
  let (x₁, y₁) := point_on_C θ₁
  let (x₂, y₂) := point_on_C θ₂
  sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Theorem statement
theorem max_distance_on_C :
  ∃ (M : ℝ), M = 4 ∧ ∀ (θ₁ θ₂ : ℝ), distance_on_C θ₁ θ₂ ≤ M :=
sorry

end

end NUMINAMATH_CALUDE_max_distance_on_C_l2670_267057


namespace NUMINAMATH_CALUDE_bug_position_after_1995_jumps_l2670_267046

/-- Represents the five points on the circle -/
inductive Point
| one
| two
| three
| four
| five

/-- Determines if a point is odd -/
def is_odd (p : Point) : Bool :=
  match p with
  | Point.one => true
  | Point.two => false
  | Point.three => true
  | Point.four => false
  | Point.five => true

/-- Moves the bug according to the rules -/
def move (p : Point) : Point :=
  match p with
  | Point.one => Point.two
  | Point.two => Point.five
  | Point.three => Point.four
  | Point.four => Point.two
  | Point.five => Point.one

/-- Performs n jumps starting from a given point -/
def jump (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | n + 1 => jump (move start) n

theorem bug_position_after_1995_jumps :
  jump Point.three 1995 = Point.one := by sorry

end NUMINAMATH_CALUDE_bug_position_after_1995_jumps_l2670_267046


namespace NUMINAMATH_CALUDE_min_distance_ellipse_to_line_l2670_267092

/-- The minimum distance from any point on the ellipse x² + y²/3 = 1 to the line x + y = 4 is √2. -/
theorem min_distance_ellipse_to_line :
  ∀ (x y : ℝ), x^2 + y^2/3 = 1 →
  (∃ (x' y' : ℝ), x' + y' = 4 ∧ (x - x')^2 + (y - y')^2 ≥ 2) ∧
  (∃ (x₀ y₀ : ℝ), x₀ + y₀ = 4 ∧ (x - x₀)^2 + (y - y₀)^2 = 2) :=
by sorry


end NUMINAMATH_CALUDE_min_distance_ellipse_to_line_l2670_267092


namespace NUMINAMATH_CALUDE_min_panels_for_intensity_reduction_l2670_267053

/-- Represents the reduction factor of light intensity when passing through a glass panel -/
def reduction_factor : ℝ := 0.9

/-- Calculates the light intensity after passing through a number of panels -/
def intensity_after_panels (a : ℝ) (x : ℕ) : ℝ := a * reduction_factor ^ x

/-- Theorem stating the minimum number of panels required to reduce light intensity to less than 1/11 of original -/
theorem min_panels_for_intensity_reduction (a : ℝ) (h : a > 0) :
  ∃ x : ℕ, (∀ y : ℕ, y < x → intensity_after_panels a y ≥ a / 11) ∧
           intensity_after_panels a x < a / 11 :=
by sorry

end NUMINAMATH_CALUDE_min_panels_for_intensity_reduction_l2670_267053


namespace NUMINAMATH_CALUDE_jenny_change_calculation_l2670_267029

/-- Calculate Jenny's change after her purchase -/
theorem jenny_change_calculation :
  let printing_discount : Float := 0.05
  let gift_card_balance : Float := 8.00
  let single_sided_cost : Float := 0.10
  let double_sided_cost : Float := 0.17
  let total_copies : Nat := 7
  let pages_per_essay : Nat := 25
  let single_sided_copies : Nat := 5
  let double_sided_copies : Nat := total_copies - single_sided_copies
  let pen_cost : Float := 1.50
  let pen_count : Nat := 7
  let sales_tax : Float := 0.10
  let cash_payment : Float := 2 * 20.00

  let single_sided_total : Float := single_sided_cost * (single_sided_copies.toFloat * pages_per_essay.toFloat)
  let double_sided_total : Float := double_sided_cost * (double_sided_copies.toFloat * pages_per_essay.toFloat)
  let printing_total : Float := single_sided_total + double_sided_total
  let printing_discounted : Float := printing_total * (1 - printing_discount)
  let pens_total : Float := pen_cost * pen_count.toFloat
  let pens_with_tax : Float := pens_total * (1 + sales_tax)
  let total_cost : Float := printing_discounted + pens_with_tax
  let remaining_cost : Float := total_cost - gift_card_balance
  let change : Float := cash_payment - remaining_cost

  change = 16.50 := by sorry

end NUMINAMATH_CALUDE_jenny_change_calculation_l2670_267029


namespace NUMINAMATH_CALUDE_other_interest_rate_is_sixteen_percent_l2670_267011

/-- Proves that given the investment conditions, the other interest rate is 16% -/
theorem other_interest_rate_is_sixteen_percent
  (investment_difference : ℝ)
  (higher_rate_investment : ℝ)
  (higher_rate : ℝ)
  (h1 : investment_difference = 1260)
  (h2 : higher_rate_investment = 2520)
  (h3 : higher_rate = 0.08)
  (h4 : higher_rate_investment = (higher_rate_investment - investment_difference) + investment_difference)
  (h5 : higher_rate_investment * higher_rate = (higher_rate_investment - investment_difference) * (16 / 100)) :
  ∃ (other_rate : ℝ), other_rate = 16 / 100 :=
by
  sorry

#check other_interest_rate_is_sixteen_percent

end NUMINAMATH_CALUDE_other_interest_rate_is_sixteen_percent_l2670_267011


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_l2670_267062

theorem cubic_sum_over_product (x y z : ℂ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 30)
  (h_eq : (x - y)^2 + (y - z)^2 + (z - x)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 33 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_l2670_267062


namespace NUMINAMATH_CALUDE_cafeteria_shirts_l2670_267027

theorem cafeteria_shirts (total : Nat) (vertical : Nat) 
  (h1 : total = 40)
  (h2 : vertical = 5)
  (h3 : ∃ (checkered : Nat), total = checkered + 4 * checkered + vertical) :
  ∃ (checkered : Nat), checkered = 7 ∧ 
    total = checkered + 4 * checkered + vertical := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_shirts_l2670_267027


namespace NUMINAMATH_CALUDE_anna_candy_distribution_l2670_267034

/-- Given a number of candies and friends, returns the minimum number of candies
    to remove for equal distribution -/
def min_candies_to_remove (candies : ℕ) (friends : ℕ) : ℕ :=
  candies % friends

theorem anna_candy_distribution :
  let total_candies : ℕ := 30
  let num_friends : ℕ := 4
  min_candies_to_remove total_candies num_friends = 2 := by
sorry

end NUMINAMATH_CALUDE_anna_candy_distribution_l2670_267034


namespace NUMINAMATH_CALUDE_equation_system_proof_l2670_267033

theorem equation_system_proof (x y m : ℝ) 
  (eq1 : x + m = 4) 
  (eq2 : y - 3 = m) : 
  x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_system_proof_l2670_267033


namespace NUMINAMATH_CALUDE_one_third_squared_times_one_eighth_l2670_267061

theorem one_third_squared_times_one_eighth : (1 / 3 : ℚ)^2 * (1 / 8 : ℚ) = 1 / 72 := by
  sorry

end NUMINAMATH_CALUDE_one_third_squared_times_one_eighth_l2670_267061


namespace NUMINAMATH_CALUDE_georgia_has_24_students_l2670_267085

/-- Represents the number of students Georgia has, given her muffin-making habits. -/
def georgia_students : ℕ :=
  let batches : ℕ := 36
  let muffins_per_batch : ℕ := 6
  let months : ℕ := 9
  let total_muffins : ℕ := batches * muffins_per_batch
  total_muffins / months

/-- Proves that Georgia has 24 students based on her muffin-making habits. -/
theorem georgia_has_24_students : georgia_students = 24 := by
  sorry

end NUMINAMATH_CALUDE_georgia_has_24_students_l2670_267085


namespace NUMINAMATH_CALUDE_amanda_ticket_sales_l2670_267036

/-- The number of tickets Amanda needs to sell on the third day -/
def tickets_to_sell_day3 (total_goal : ℕ) (sold_day1 : ℕ) (sold_day2 : ℕ) : ℕ :=
  total_goal - (sold_day1 + sold_day2)

/-- Theorem stating that Amanda needs to sell 28 tickets on the third day -/
theorem amanda_ticket_sales : tickets_to_sell_day3 80 20 32 = 28 := by
  sorry

end NUMINAMATH_CALUDE_amanda_ticket_sales_l2670_267036


namespace NUMINAMATH_CALUDE_largest_valid_number_l2670_267022

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (∀ d, d ∈ n.digits 10 → d ≠ 0 → n % d = 0) ∧
  (n.digits 10).sum % 6 = 0

theorem largest_valid_number : 
  is_valid_number 936 ∧ ∀ m, is_valid_number m → m ≤ 936 :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l2670_267022


namespace NUMINAMATH_CALUDE_valerie_skipping_rate_l2670_267016

/-- Roberto's skipping rate in skips per hour -/
def roberto_rate : ℕ := 4200

/-- Total skips for Roberto and Valerie in 15 minutes -/
def total_skips : ℕ := 2250

/-- Duration of skipping in minutes -/
def duration : ℕ := 15

/-- Valerie's skipping rate in skips per minute -/
def valerie_rate : ℕ := 80

theorem valerie_skipping_rate :
  (roberto_rate * duration / 60 + valerie_rate * duration = total_skips) ∧
  (valerie_rate = (total_skips - roberto_rate * duration / 60) / duration) :=
sorry

end NUMINAMATH_CALUDE_valerie_skipping_rate_l2670_267016


namespace NUMINAMATH_CALUDE_movie_of_the_year_fraction_l2670_267045

/-- The fraction of top-10 lists a film must appear on to be considered for "movie of the year" -/
def required_fraction (total_members : ℕ) (min_lists : ℚ) : ℚ :=
  min_lists / total_members

/-- Proof that the required fraction is 0.25 given the specific conditions -/
theorem movie_of_the_year_fraction :
  let total_members : ℕ := 775
  let min_lists : ℚ := 193.75
  required_fraction total_members min_lists = 0.25 := by
sorry

end NUMINAMATH_CALUDE_movie_of_the_year_fraction_l2670_267045


namespace NUMINAMATH_CALUDE_max_value_expression_l2670_267028

theorem max_value_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 12) :
  a * b + b * c + a * c + a * b * c ≤ 112 ∧ 
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a' + b' + c' = 12 ∧ 
    a' * b' + b' * c' + a' * c' + a' * b' * c' = 112 :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l2670_267028


namespace NUMINAMATH_CALUDE_max_sum_of_rolls_l2670_267083

def is_valid_roll_set (rolls : List Nat) : Prop :=
  rolls.length = 24 ∧
  (∀ n : Nat, n ≥ 1 ∧ n ≤ 6 → n ∈ rolls) ∧
  (∀ n : Nat, n ≥ 2 ∧ n ≤ 6 → rolls.count 1 > rolls.count n)

def sum_of_rolls (rolls : List Nat) : Nat :=
  rolls.sum

theorem max_sum_of_rolls :
  ∀ rolls : List Nat,
    is_valid_roll_set rolls →
    sum_of_rolls rolls ≤ 90 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_rolls_l2670_267083


namespace NUMINAMATH_CALUDE_table_color_change_l2670_267055

/-- Represents the color of a cell in the table -/
inductive CellColor
| White
| Black
| Orange

/-- Represents a 3n × 3n table with the given coloring pattern -/
def Table (n : ℕ) := Fin (3*n) → Fin (3*n) → CellColor

/-- Predicate to check if a given 2×2 square can be chosen for color change -/
def CanChangeSquare (t : Table n) (i j : Fin (3*n-1)) : Prop := True

/-- Predicate to check if the table has all white cells turned to black and all black cells turned to white -/
def IsTargetState (t : Table n) : Prop := True

/-- Predicate to check if it's possible to reach the target state in a finite number of steps -/
def CanReachTargetState (n : ℕ) : Prop := 
  ∃ (t : Table n), IsTargetState t

theorem table_color_change (n : ℕ) : 
  CanReachTargetState n ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_table_color_change_l2670_267055


namespace NUMINAMATH_CALUDE_not_all_problems_solvable_by_algorithm_l2670_267074

/-- Represents a problem that can be solved computationally -/
def Problem : Type := Unit

/-- Represents an algorithm -/
def Algorithm : Type := Unit

/-- Represents the characteristic that an algorithm is executed step by step -/
def stepwise (a : Algorithm) : Prop := sorry

/-- Represents the characteristic that each step of an algorithm yields a unique result -/
def uniqueStepResult (a : Algorithm) : Prop := sorry

/-- Represents the characteristic that algorithms are effective for a class of problems -/
def effectiveForClass (a : Algorithm) : Prop := sorry

/-- Represents the characteristic that algorithms are mechanical -/
def mechanical (a : Algorithm) : Prop := sorry

/-- Represents the characteristic that algorithms can require repetitive calculation -/
def repetitiveCalculation (a : Algorithm) : Prop := sorry

/-- Represents the characteristic that algorithms are a universal method -/
def universalMethod (a : Algorithm) : Prop := sorry

/-- Theorem stating that not all problems can be solved by algorithms -/
theorem not_all_problems_solvable_by_algorithm : 
  ¬ (∀ (p : Problem), ∃ (a : Algorithm), 
    stepwise a ∧ 
    uniqueStepResult a ∧ 
    effectiveForClass a ∧ 
    mechanical a ∧ 
    repetitiveCalculation a ∧ 
    universalMethod a) := by sorry


end NUMINAMATH_CALUDE_not_all_problems_solvable_by_algorithm_l2670_267074


namespace NUMINAMATH_CALUDE_group_purchase_equation_l2670_267038

/-- Represents a group purchase scenario where:
    - x is the number of people
    - p is the price of the item in coins
    - If each person contributes 8 coins, there's an excess of 3 coins
    - If each person contributes 7 coins, there's a shortage of 4 coins -/
structure GroupPurchase where
  x : ℕ  -- number of people
  p : ℕ  -- price of the item in coins
  excess_condition : 8 * x = p + 3
  shortage_condition : 7 * x + 4 = p

/-- Theorem stating that in a valid GroupPurchase scenario, 
    the number of people satisfies the equation 8x - 3 = 7x + 4 -/
theorem group_purchase_equation (gp : GroupPurchase) : 8 * gp.x - 3 = 7 * gp.x + 4 := by
  sorry


end NUMINAMATH_CALUDE_group_purchase_equation_l2670_267038


namespace NUMINAMATH_CALUDE_smallest_4digit_divisible_by_5_6_2_l2670_267006

def is_divisible (n m : ℕ) : Prop := ∃ k, n = m * k

theorem smallest_4digit_divisible_by_5_6_2 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 →
  (is_divisible n 5 ∧ is_divisible n 6 ∧ is_divisible n 2) →
  1020 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_4digit_divisible_by_5_6_2_l2670_267006


namespace NUMINAMATH_CALUDE_square_area_ratio_l2670_267051

theorem square_area_ratio (R : ℝ) (R_pos : R > 0) : 
  let x := Real.sqrt ((4 / 5) * R^2)
  let y := R * Real.sqrt 2
  x^2 / y^2 = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2670_267051


namespace NUMINAMATH_CALUDE_shirt_double_discount_l2670_267042

theorem shirt_double_discount (original_price : ℝ) (discount_rate : ℝ) : 
  original_price = 32 → 
  discount_rate = 0.25 → 
  (1 - discount_rate) * (1 - discount_rate) * original_price = 18 := by
sorry

end NUMINAMATH_CALUDE_shirt_double_discount_l2670_267042


namespace NUMINAMATH_CALUDE_map_to_actual_distance_l2670_267073

/-- Given a map distance and scale, calculate the actual distance between two cities. -/
theorem map_to_actual_distance 
  (map_distance : ℝ) 
  (scale : ℝ) 
  (h1 : map_distance = 88) 
  (h2 : scale = 15) : 
  map_distance * scale = 1320 := by
  sorry

end NUMINAMATH_CALUDE_map_to_actual_distance_l2670_267073


namespace NUMINAMATH_CALUDE_triangle_inequality_check_l2670_267097

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_inequality_check :
  can_form_triangle 9 6 13 ∧
  ¬(can_form_triangle 6 8 16) ∧
  ¬(can_form_triangle 18 9 8) ∧
  ¬(can_form_triangle 3 5 9) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_check_l2670_267097


namespace NUMINAMATH_CALUDE_Q_sufficient_not_necessary_for_P_l2670_267043

-- Define the property P(x) as x^2 - 1 > 0
def P (x : ℝ) : Prop := x^2 - 1 > 0

-- Define the condition Q(x) as x < -1
def Q (x : ℝ) : Prop := x < -1

-- Theorem stating that Q is sufficient but not necessary for P
theorem Q_sufficient_not_necessary_for_P :
  (∀ x : ℝ, Q x → P x) ∧ ¬(∀ x : ℝ, P x → Q x) :=
sorry

end NUMINAMATH_CALUDE_Q_sufficient_not_necessary_for_P_l2670_267043


namespace NUMINAMATH_CALUDE_norm_photos_difference_l2670_267070

-- Define the number of photos taken by each photographer
variable (L M N : ℕ)

-- Define the conditions from the problem
def condition1 (L M N : ℕ) : Prop := L + M = M + N - 60
def condition2 (L N : ℕ) : Prop := ∃ X, N = 2 * L + X
def condition3 (N : ℕ) : Prop := N = 110

-- State the theorem
theorem norm_photos_difference (L M N : ℕ) 
  (h1 : condition1 L M N) (h2 : condition2 L N) (h3 : condition3 N) : 
  ∃ X, N = 2 * L + X ∧ X = 110 - 2 * L :=
sorry

end NUMINAMATH_CALUDE_norm_photos_difference_l2670_267070


namespace NUMINAMATH_CALUDE_sum_of_circle_areas_l2670_267001

/-- Given a right triangle with sides 5, 12, and 13, where the vertices are centers of
    three mutually externally tangent circles, the sum of the areas of these circles is 113π. -/
theorem sum_of_circle_areas (a b c : ℝ) (r s t : ℝ) : 
  a = 5 → b = 12 → c = 13 →
  a^2 + b^2 = c^2 →
  r + s = b →
  s + t = a →
  r + t = c →
  π * (r^2 + s^2 + t^2) = 113 * π := by
sorry

end NUMINAMATH_CALUDE_sum_of_circle_areas_l2670_267001


namespace NUMINAMATH_CALUDE_integer_root_pairs_l2670_267078

/-- A function that checks if all roots of a quadratic polynomial ax^2 + bx + c are integers -/
def allRootsInteger (a b c : ℤ) : Prop :=
  ∃ x y : ℤ, a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ x ≠ y

/-- The main theorem stating the only valid pairs (p,q) -/
theorem integer_root_pairs :
  ∀ p q : ℤ,
    (allRootsInteger 1 p q ∧ allRootsInteger 1 q p) ↔
    ((p = 4 ∧ q = 4) ∨ (p = 9 ∧ q = 8) ∨ (p = 8 ∧ q = 9)) :=
by sorry

end NUMINAMATH_CALUDE_integer_root_pairs_l2670_267078


namespace NUMINAMATH_CALUDE_probability_of_one_out_of_four_l2670_267076

theorem probability_of_one_out_of_four (S : Finset α) (h : S.card = 4) :
  ∀ a ∈ S, (1 : ℝ) / S.card = (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_one_out_of_four_l2670_267076


namespace NUMINAMATH_CALUDE_oranges_returned_l2670_267031

def oranges_problem (initial_oranges : ℕ) (eaten_oranges : ℕ) (final_oranges : ℕ) : ℕ :=
  let remaining_after_eating := initial_oranges - eaten_oranges
  let stolen_oranges := remaining_after_eating / 2
  let remaining_after_theft := remaining_after_eating - stolen_oranges
  final_oranges - remaining_after_theft

theorem oranges_returned (initial_oranges eaten_oranges final_oranges : ℕ) 
  (h1 : initial_oranges = 60)
  (h2 : eaten_oranges = 10)
  (h3 : final_oranges = 30) : 
  oranges_problem initial_oranges eaten_oranges final_oranges = 5 := by
  sorry

#eval oranges_problem 60 10 30

end NUMINAMATH_CALUDE_oranges_returned_l2670_267031


namespace NUMINAMATH_CALUDE_largest_term_binomial_expansion_l2670_267026

theorem largest_term_binomial_expansion (k : ℕ) :
  k ≠ 64 →
  Nat.choose 100 64 * (Real.sqrt 3) ^ 64 > Nat.choose 100 k * (Real.sqrt 3) ^ k :=
sorry

end NUMINAMATH_CALUDE_largest_term_binomial_expansion_l2670_267026


namespace NUMINAMATH_CALUDE_negation_equivalence_l2670_267081

theorem negation_equivalence :
  (¬ ∃ (x y : ℝ), x^2 + y^2 - 1 ≤ 0) ↔ (∀ (x y : ℝ), x^2 + y^2 - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2670_267081


namespace NUMINAMATH_CALUDE_square_difference_given_sum_and_product_l2670_267015

theorem square_difference_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = -5) (h2 : x * y = 6) : (x - y)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_given_sum_and_product_l2670_267015


namespace NUMINAMATH_CALUDE_alyssa_grapes_cost_l2670_267040

/-- The amount Alyssa paid for grapes -/
def grapesCost (totalSpent refund : ℚ) : ℚ := totalSpent + refund

/-- Proof that Alyssa paid $12.08 for grapes -/
theorem alyssa_grapes_cost : 
  let totalSpent : ℚ := 223/100
  let cherryRefund : ℚ := 985/100
  grapesCost totalSpent cherryRefund = 1208/100 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_grapes_cost_l2670_267040


namespace NUMINAMATH_CALUDE_house_transaction_loss_l2670_267094

def house_transaction (initial_value : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) : ℝ :=
  let first_sale := initial_value * (1 - loss_percent)
  let second_sale := first_sale * (1 + gain_percent)
  initial_value - second_sale

theorem house_transaction_loss :
  house_transaction 9000 0.1 0.1 = 810 := by
  sorry

end NUMINAMATH_CALUDE_house_transaction_loss_l2670_267094


namespace NUMINAMATH_CALUDE_lisas_large_spoons_lisas_large_spoons_is_ten_l2670_267020

/-- Calculates the number of large spoons in Lisa's new cutlery set -/
theorem lisas_large_spoons (num_children : ℕ) (baby_spoons_per_child : ℕ) 
  (decorative_spoons : ℕ) (new_teaspoons : ℕ) (total_spoons : ℕ) : ℕ :=
  let kept_spoons := num_children * baby_spoons_per_child + decorative_spoons
  let known_spoons := kept_spoons + new_teaspoons
  total_spoons - known_spoons

/-- Proves that the number of large spoons in Lisa's new cutlery set is 10 -/
theorem lisas_large_spoons_is_ten :
  lisas_large_spoons 4 3 2 15 39 = 10 := by
  sorry

end NUMINAMATH_CALUDE_lisas_large_spoons_lisas_large_spoons_is_ten_l2670_267020


namespace NUMINAMATH_CALUDE_inequality_proof_l2670_267091

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let K := a^4*(b^2*c + b*c^2) + a^3*(b^3*c + b*c^3) + a^2*(b^3*c^2 + b^2*c^3 + b^2*c + b*c^2) + a*(b^3*c + b*c^3) + (b^3*c^2 + b^2*c^3)
  K ≥ 12*a^2*b^2*c^2 ∧ (K = 12*a^2*b^2*c^2 ↔ a = 1 ∧ b = 1 ∧ c = 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2670_267091


namespace NUMINAMATH_CALUDE_abs_comparison_negative_numbers_l2670_267023

theorem abs_comparison_negative_numbers (x y : ℝ) 
  (hx_neg : x < 0) (hy_neg : y < 0) (hxy : x < y) : 
  |x| > |y| := by
  sorry

end NUMINAMATH_CALUDE_abs_comparison_negative_numbers_l2670_267023


namespace NUMINAMATH_CALUDE_square_divisors_count_l2670_267082

-- Define a function to count divisors
def count_divisors (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem square_divisors_count (n : ℕ) : 
  count_divisors n = 4 → count_divisors (n^2) = 7 := by sorry

end NUMINAMATH_CALUDE_square_divisors_count_l2670_267082


namespace NUMINAMATH_CALUDE_fish_count_proof_l2670_267067

/-- The number of fish Jerk Tuna has -/
def jerk_tuna_fish : ℕ := 144

/-- The number of fish Tall Tuna has -/
def tall_tuna_fish : ℕ := 2 * jerk_tuna_fish

/-- The total number of fish Jerk Tuna and Tall Tuna have together -/
def total_fish : ℕ := jerk_tuna_fish + tall_tuna_fish

theorem fish_count_proof : total_fish = 432 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_proof_l2670_267067


namespace NUMINAMATH_CALUDE_machine_work_time_l2670_267098

theorem machine_work_time (x : ℝ) (h1 : x > 0) 
  (h2 : 1/x + 1/2 + 1/6 = 11/12) : x = 4 := by
  sorry

#check machine_work_time

end NUMINAMATH_CALUDE_machine_work_time_l2670_267098


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2670_267090

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π/4) = 3) : Real.tan α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2670_267090


namespace NUMINAMATH_CALUDE_exists_interior_points_l2670_267004

/-- A point in the plane represented by its coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - q.x) = (r.y - q.y) * (q.x - p.x)

/-- Check if a point is in the interior of a triangle -/
def interior_point (p : Point) (a b c : Point) : Prop :=
  ∃ (u v w : ℝ), u > 0 ∧ v > 0 ∧ w > 0 ∧ u + v + w = 1 ∧
  p.x = u * a.x + v * b.x + w * c.x ∧
  p.y = u * a.y + v * b.y + w * c.y

/-- The main theorem -/
theorem exists_interior_points (n : ℕ) (S : Finset Point) :
  S.card = n →
  (∀ (a b c : Point), a ∈ S → b ∈ S → c ∈ S → ¬collinear a b c) →
  ∃ (P : Finset Point), P.card = 2 * n - 5 ∧
    ∀ (a b c : Point), a ∈ S → b ∈ S → c ∈ S →
      ∃ (p : Point), p ∈ P ∧ interior_point p a b c :=
sorry

end NUMINAMATH_CALUDE_exists_interior_points_l2670_267004


namespace NUMINAMATH_CALUDE_overlap_range_l2670_267030

theorem overlap_range (total : ℕ) (math : ℕ) (chem : ℕ) (x : ℕ) 
  (h_total : total = 45)
  (h_math : math = 28)
  (h_chem : chem = 21)
  (h_overlap : x ≤ math ∧ x ≤ chem)
  (h_inclusion : math + chem - x ≤ total) :
  4 ≤ x ∧ x ≤ 21 := by
sorry

end NUMINAMATH_CALUDE_overlap_range_l2670_267030


namespace NUMINAMATH_CALUDE_range_of_m_l2670_267050

def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m (m : ℝ) : 
  (p m ∨ q m) → m < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2670_267050


namespace NUMINAMATH_CALUDE_count_x_values_l2670_267068

theorem count_x_values (x y z w : ℕ+) 
  (h1 : x > y ∧ y > z ∧ z > w)
  (h2 : x + y + z + w = 4020)
  (h3 : x^2 - y^2 + z^2 - w^2 = 4020) :
  ∃ (S : Finset ℕ+), (∀ a ∈ S, ∃ y z w : ℕ+, 
    x = a ∧ 
    a > y ∧ y > z ∧ z > w ∧
    a + y + z + w = 4020 ∧
    a^2 - y^2 + z^2 - w^2 = 4020) ∧ 
  S.card = 1003 :=
sorry

end NUMINAMATH_CALUDE_count_x_values_l2670_267068


namespace NUMINAMATH_CALUDE_geometric_sequence_min_value_l2670_267077

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_min_value (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  (2 * a 4 + a 3 - 2 * a 2 - a 1 = 8) →
  (∃ m : ℝ, m = 54 ∧ ∀ x : ℝ, 2 * a 8 + a 7 ≥ m) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_min_value_l2670_267077


namespace NUMINAMATH_CALUDE_xy_squared_l2670_267080

theorem xy_squared (x y : ℝ) (h1 : 2 * x * (x + y) = 36) (h2 : 3 * y * (x + y) = 81) :
  (x + y)^2 = 117 / 5 := by
  sorry

end NUMINAMATH_CALUDE_xy_squared_l2670_267080


namespace NUMINAMATH_CALUDE_vector_operation_result_l2670_267058

theorem vector_operation_result : 
  let v1 : Fin 3 → ℝ := ![(-3 : ℝ), 2, -5]
  let v2 : Fin 3 → ℝ := ![4, 10, -6]
  3 • v1 + v2 = ![-5, 16, -21] := by
sorry

end NUMINAMATH_CALUDE_vector_operation_result_l2670_267058


namespace NUMINAMATH_CALUDE_divisible_by_six_l2670_267005

theorem divisible_by_six (m : ℕ) : ∃ k : ℤ, (m : ℤ)^3 + 11*(m : ℤ) = 6*k := by sorry

end NUMINAMATH_CALUDE_divisible_by_six_l2670_267005


namespace NUMINAMATH_CALUDE_sin_alpha_value_l2670_267032

theorem sin_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) : 
  Real.sin α = Real.sqrt 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l2670_267032


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l2670_267087

theorem ratio_of_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) 
  (hsum : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l2670_267087


namespace NUMINAMATH_CALUDE_female_officers_count_l2670_267086

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) 
  (female_ratio : ℚ) (female_on_duty_percent : ℚ) :
  total_on_duty = 160 →
  female_ratio = 1/2 →
  female_on_duty_percent = 16/100 →
  (female_on_duty_ratio * ↑total_on_duty : ℚ) = (female_ratio * ↑total_on_duty : ℚ) →
  (female_on_duty_percent * (female_on_duty_ratio * ↑total_on_duty / female_on_duty_percent : ℚ) : ℚ) = 500 := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l2670_267086


namespace NUMINAMATH_CALUDE_fraction_equality_implies_sum_l2670_267019

theorem fraction_equality_implies_sum (A B : ℚ) :
  (∀ x : ℚ, x ≠ 4 ∧ x ≠ 5 →
    (B * x - 23) / (x^2 - 9*x + 20) = A / (x - 4) + 5 / (x - 5)) →
  A + B = 11/9 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_sum_l2670_267019


namespace NUMINAMATH_CALUDE_fraction_simplification_l2670_267069

theorem fraction_simplification (y b : ℝ) : 
  (y + 2) / 4 + (5 - 4*y + b) / 3 = (-13*y + 4*b + 26) / 12 := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2670_267069


namespace NUMINAMATH_CALUDE_runners_passing_count_l2670_267079

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ  -- speed in meters per minute
  radius : ℝ  -- radius of the track in meters
  direction : ℤ  -- 1 for clockwise, -1 for counterclockwise

/-- Calculates the number of times two runners pass each other -/
def passingCount (r1 r2 : Runner) (duration : ℝ) : ℕ :=
  sorry

theorem runners_passing_count :
  let odell : Runner := { speed := 260, radius := 55, direction := 1 }
  let kershaw : Runner := { speed := 280, radius := 65, direction := -1 }
  passingCount odell kershaw 30 = 126 :=
sorry

end NUMINAMATH_CALUDE_runners_passing_count_l2670_267079


namespace NUMINAMATH_CALUDE_base8_to_base7_conversion_l2670_267093

-- Define a function to convert from base 8 to decimal
def base8ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (8 ^ i)) 0

-- Define a function to convert from decimal to base 7
def decimalToBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 7) ((m % 7) :: acc)
  aux n []

-- Theorem statement
theorem base8_to_base7_conversion :
  decimalToBase7 (base8ToDecimal [3, 6, 5]) = [1, 0, 4, 0] :=
sorry

end NUMINAMATH_CALUDE_base8_to_base7_conversion_l2670_267093


namespace NUMINAMATH_CALUDE_tower_arrangements_l2670_267075

def num_red_cubes : ℕ := 2
def num_blue_cubes : ℕ := 4
def num_green_cubes : ℕ := 3
def tower_height : ℕ := 8

def remaining_cubes : ℕ := tower_height - 1
def remaining_blue_cubes : ℕ := num_blue_cubes - 1
def remaining_red_cubes : ℕ := num_red_cubes
def remaining_green_cubes : ℕ := num_green_cubes - 1

theorem tower_arrangements :
  (remaining_cubes.factorial) / (remaining_blue_cubes.factorial * remaining_red_cubes.factorial * remaining_green_cubes.factorial) = 210 := by
  sorry

end NUMINAMATH_CALUDE_tower_arrangements_l2670_267075


namespace NUMINAMATH_CALUDE_courier_cost_formula_l2670_267008

/-- The cost function for sending a small item via courier service -/
def courier_cost (P : ℕ) : ℕ :=
  5 * P + 12

theorem courier_cost_formula (P : ℕ) (h : P ≥ 1) :
  courier_cost P =
    5 + -- flat service fee
    12 + -- cost for the first kilogram
    (5 * (P - 1)) -- cost for additional kilograms
  := by sorry

end NUMINAMATH_CALUDE_courier_cost_formula_l2670_267008


namespace NUMINAMATH_CALUDE_largest_non_expressible_l2670_267060

/-- A positive integer is composite if it has a proper divisor greater than 1. -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- A function that checks if a number can be expressed as 42k + c, 
    where k is a positive integer and c is a positive composite integer. -/
def CanBeExpressed (n : ℕ) : Prop :=
  ∃ (k c : ℕ), k > 0 ∧ IsComposite c ∧ n = 42 * k + c

/-- The theorem stating that 215 is the largest positive integer that cannot be expressed
    as the sum of a positive integral multiple of 42 and a positive composite integer. -/
theorem largest_non_expressible : 
  (∀ n : ℕ, n > 215 → CanBeExpressed n) ∧ 
  (¬ CanBeExpressed 215) := by
  sorry

#check largest_non_expressible

end NUMINAMATH_CALUDE_largest_non_expressible_l2670_267060


namespace NUMINAMATH_CALUDE_proposition_false_iff_m_in_range_l2670_267096

/-- The proposition is false for all real x when m is in [2,6) -/
theorem proposition_false_iff_m_in_range (m : ℝ) :
  (∀ x : ℝ, (m - 2) * x^2 + (m - 2) * x + 1 > 0) ↔ (2 ≤ m ∧ m < 6) :=
by sorry

end NUMINAMATH_CALUDE_proposition_false_iff_m_in_range_l2670_267096


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2670_267072

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2670_267072


namespace NUMINAMATH_CALUDE_same_solution_k_value_l2670_267064

theorem same_solution_k_value : ∃ (k : ℝ), 
  (∀ (x : ℝ), (2 * x + 4 = 4 * (x - 2)) ↔ (-x + k = 2 * x - 1)) → k = 17 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_k_value_l2670_267064


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2670_267002

theorem quadratic_roots_property (α β : ℝ) : 
  α ≠ β →
  α^2 + 3*α - 1 = 0 →
  β^2 + 3*β - 1 = 0 →
  α^2 + 4*α + β = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2670_267002


namespace NUMINAMATH_CALUDE_orange_fraction_l2670_267088

theorem orange_fraction (total_fruit : ℕ) (oranges peaches apples : ℕ) :
  total_fruit = 56 →
  peaches = oranges / 2 →
  apples = 5 * peaches →
  apples = 35 →
  oranges = total_fruit / 4 := by
  sorry

end NUMINAMATH_CALUDE_orange_fraction_l2670_267088


namespace NUMINAMATH_CALUDE_p_iff_q_l2670_267063

theorem p_iff_q (a b : ℝ) :
  (a > 2 ∧ b > 3) ↔ (a + b > 5 ∧ (a - 2) * (b - 3) > 0) := by
  sorry

end NUMINAMATH_CALUDE_p_iff_q_l2670_267063


namespace NUMINAMATH_CALUDE_isosceles_triangle_from_wire_isosceles_triangle_with_side_6_l2670_267037

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Represents an isosceles triangle -/
def IsoscelesTriangle (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

theorem isosceles_triangle_from_wire (wire_length : ℝ) 
  (h_wire : wire_length = 24) :
  ∃ (t : Triangle), IsoscelesTriangle t ∧ 
    t.a + t.b + t.c = wire_length ∧
    t.a = t.b ∧ t.a = 2 * t.c ∧
    t.a = 48 / 5 := by
  sorry

theorem isosceles_triangle_with_side_6 (wire_length : ℝ) 
  (h_wire : wire_length = 24) :
  ∃ (t : Triangle), IsoscelesTriangle t ∧ 
    t.a + t.b + t.c = wire_length ∧
    (t.a = 6 ∨ t.b = 6 ∨ t.c = 6) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_from_wire_isosceles_triangle_with_side_6_l2670_267037


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l2670_267065

def U : Set Int := {0, -1, -2, -3, -4}
def M : Set Int := {0, -1, -2}

theorem complement_of_M_in_U :
  (U \ M) = {-3, -4} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l2670_267065


namespace NUMINAMATH_CALUDE_jerrys_action_figures_l2670_267000

theorem jerrys_action_figures :
  ∀ (initial_figures initial_books added_figures : ℕ),
    initial_figures = 2 →
    initial_books = 10 →
    initial_books = (initial_figures + added_figures) + 4 →
    added_figures = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_jerrys_action_figures_l2670_267000


namespace NUMINAMATH_CALUDE_positive_root_of_equation_l2670_267021

theorem positive_root_of_equation (x : ℝ) : 
  x > 0 ∧ (1/3) * (4*x^2 - 2) = (x^2 - 35*x - 7) * (x^2 + 20*x + 4) → 
  x = (35 + Real.sqrt 1257) / 2 := by
sorry

end NUMINAMATH_CALUDE_positive_root_of_equation_l2670_267021


namespace NUMINAMATH_CALUDE_wrapping_paper_area_l2670_267052

/-- The area of a square sheet of wrapping paper required to wrap a rectangular box -/
theorem wrapping_paper_area (l w h : ℝ) (h_positive : l > 0 ∧ w > 0 ∧ h > 0) 
  (h_different : h ≠ l ∧ h ≠ w) : 
  let side_length := l + w
  (side_length ^ 2 : ℝ) = (l + w) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_l2670_267052


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2670_267039

theorem polynomial_division_theorem (x : ℝ) :
  (x - 3) * (x^5 - 2*x^4 + 18*x^3 + 42*x^2 + 135*x + 387) + 1221 =
  x^6 - 5*x^5 + 24*x^4 - 12*x^3 + 9*x^2 - 18*x + 15 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2670_267039


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_p_l2670_267095

/-- Given a quadratic equation x^2 - px + 2q = 0 where p and q are its roots and both non-zero,
    the sum of the roots is equal to p. -/
theorem sum_of_roots_equals_p (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) 
    (h : ∀ x, x^2 - p*x + 2*q = 0 ↔ x = p ∨ x = q) : 
  p + q = p := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_p_l2670_267095


namespace NUMINAMATH_CALUDE_find_y_l2670_267084

theorem find_y : ∃ y : ℝ, (Real.sqrt (1 + Real.sqrt (4 * y - 5)) = Real.sqrt 8) ∧ y = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l2670_267084


namespace NUMINAMATH_CALUDE_janet_waterpark_cost_l2670_267035

/-- Calculates the total cost for a group visiting a waterpark with a discount -/
def waterpark_cost (adult_price : ℚ) (num_adults num_children : ℕ) (discount_percent : ℚ) (soda_price : ℚ) : ℚ :=
  let child_price := adult_price / 2
  let total_admission := adult_price * num_adults + child_price * num_children
  let discounted_admission := total_admission * (1 - discount_percent / 100)
  discounted_admission + soda_price

/-- The total cost for Janet's group visit to the waterpark -/
theorem janet_waterpark_cost :
  waterpark_cost 30 6 4 20 5 = 197 := by
  sorry

end NUMINAMATH_CALUDE_janet_waterpark_cost_l2670_267035


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2670_267049

theorem contrapositive_equivalence :
  (∀ (a b : ℝ), ab > 0 → a > 0) ↔ (∀ (a b : ℝ), a ≤ 0 → ab ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2670_267049


namespace NUMINAMATH_CALUDE_train_passing_time_l2670_267048

/-- The length of the high-speed train in meters -/
def high_speed_train_length : ℝ := 400

/-- The length of the regular train in meters -/
def regular_train_length : ℝ := 600

/-- The time in seconds it takes for a passenger on the high-speed train to see the regular train pass -/
def high_speed_observation_time : ℝ := 3

/-- The time in seconds it takes for a passenger on the regular train to see the high-speed train pass -/
def regular_observation_time : ℝ := 2

theorem train_passing_time :
  (regular_train_length / high_speed_observation_time) * regular_observation_time = high_speed_train_length :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l2670_267048


namespace NUMINAMATH_CALUDE_equation_solution_l2670_267013

theorem equation_solution (x : ℝ) :
  x > 6 →
  (Real.sqrt (x - 6 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 9)) - 3) ↔
  x ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2670_267013


namespace NUMINAMATH_CALUDE_rectangular_hall_dimensions_l2670_267066

theorem rectangular_hall_dimensions (length width : ℝ) : 
  width = length / 2 → 
  length * width = 450 → 
  length - width = 15 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_hall_dimensions_l2670_267066


namespace NUMINAMATH_CALUDE_subtraction_problem_l2670_267047

theorem subtraction_problem :
  572 - 275 = 297 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l2670_267047


namespace NUMINAMATH_CALUDE_room_height_proof_l2670_267041

theorem room_height_proof (l b h : ℝ) : 
  l = 12 → b = 8 → (l^2 + b^2 + h^2 = 17^2) → h = 9 := by sorry

end NUMINAMATH_CALUDE_room_height_proof_l2670_267041


namespace NUMINAMATH_CALUDE_linear_equation_condition_l2670_267009

/-- If mx^(m+2) + m - 2 = 0 is a linear equation with respect to x, then m = -1 -/
theorem linear_equation_condition (m : ℝ) : 
  (∃ a b, ∀ x, m * x^(m + 2) + m - 2 = a * x + b) → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l2670_267009


namespace NUMINAMATH_CALUDE_min_production_volume_for_break_even_l2670_267017

/-- The total cost function -/
def total_cost (x : ℝ) : ℝ := 3000 + 20 * x - 0.1 * x^2

/-- The revenue function -/
def revenue (x : ℝ) : ℝ := 25 * x

/-- The break-even condition -/
def break_even (x : ℝ) : Prop := revenue x ≥ total_cost x

theorem min_production_volume_for_break_even :
  ∃ (x : ℝ), x = 150 ∧ 0 < x ∧ x < 240 ∧ break_even x ∧
  ∀ (y : ℝ), 0 < y ∧ y < x → ¬(break_even y) := by
  sorry

end NUMINAMATH_CALUDE_min_production_volume_for_break_even_l2670_267017


namespace NUMINAMATH_CALUDE_green_blue_tile_difference_l2670_267014

/-- Proves that the difference between green and blue tiles after adding two borders is 29 -/
theorem green_blue_tile_difference : 
  let initial_blue : ℕ := 13
  let initial_green : ℕ := 6
  let tiles_per_border : ℕ := 18
  let borders_added : ℕ := 2
  let final_green : ℕ := initial_green + borders_added * tiles_per_border
  let final_blue : ℕ := initial_blue
  final_green - final_blue = 29 := by
sorry


end NUMINAMATH_CALUDE_green_blue_tile_difference_l2670_267014


namespace NUMINAMATH_CALUDE_logarithm_sum_equals_three_logarithm_base_25_144_l2670_267024

-- Part 1
theorem logarithm_sum_equals_three :
  (Real.log 2) ^ 2 + (Real.log 20 + 2) * Real.log 5 + Real.log 4 = 3 := by sorry

-- Part 2
theorem logarithm_base_25_144 (a b : ℝ) (h1 : Real.log 3 / Real.log 5 = a) (h2 : Real.log 4 / Real.log 5 = b) :
  Real.log 144 / Real.log 25 = a + b := by sorry

end NUMINAMATH_CALUDE_logarithm_sum_equals_three_logarithm_base_25_144_l2670_267024


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l2670_267089

theorem quadratic_form_sum (a b c : ℝ) : 
  (∀ x, 8 * x^2 - 48 * x - 320 = a * (x + b)^2 + c) → 
  a + b + c = -387 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l2670_267089


namespace NUMINAMATH_CALUDE_job_completion_time_l2670_267071

/-- The time taken for three workers to complete a job together, given their individual work rates -/
theorem job_completion_time 
  (rate_a rate_b rate_c : ℚ) 
  (h_a : rate_a = 1 / 8) 
  (h_b : rate_b = 1 / 16) 
  (h_c : rate_c = 1 / 16) : 
  1 / (rate_a + rate_b + rate_c) = 4 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l2670_267071
