import Mathlib

namespace quadratic_function_sum_l2555_255567

/-- Given two quadratic functions f and g, prove that A + B = 0 under certain conditions -/
theorem quadratic_function_sum (A B : ℝ) (f g : ℝ → ℝ) : 
  A ≠ B →
  (∀ x, f x = A * x^2 + B) →
  (∀ x, g x = B * x^2 + A) →
  (∀ x, f (g x) - g (f x) = -A^2 + B^2) →
  A + B = 0 := by
  sorry


end quadratic_function_sum_l2555_255567


namespace set_intersection_problem_l2555_255553

def M : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def N : Set ℝ := {-2, -1, 1, 2}

theorem set_intersection_problem : M ∩ N = {1, 2} := by
  sorry

end set_intersection_problem_l2555_255553


namespace jan_paid_288_dollars_l2555_255582

def roses_per_dozen : ℕ := 12
def dozen_bought : ℕ := 5
def cost_per_rose : ℚ := 6
def discount_percentage : ℚ := 80

def total_roses : ℕ := dozen_bought * roses_per_dozen

def full_price : ℚ := (total_roses : ℚ) * cost_per_rose

def discounted_price : ℚ := full_price * (discount_percentage / 100)

theorem jan_paid_288_dollars : discounted_price = 288 := by
  sorry

end jan_paid_288_dollars_l2555_255582


namespace quadratic_inequality_range_l2555_255537

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, x^2 - k*x + 1 > 0) ↔ -2 < k ∧ k < 2 := by
  sorry

end quadratic_inequality_range_l2555_255537


namespace emma_skateboard_time_l2555_255546

/-- The time taken for Emma to skateboard along a looping path on a highway --/
theorem emma_skateboard_time : ∀ (highway_length highway_width emma_speed : ℝ),
  highway_length = 2 * 5280 →
  highway_width = 50 →
  emma_speed = 4 →
  ∃ (time : ℝ), time = π / 2 ∧ time * emma_speed = 2 * π :=
by
  sorry

end emma_skateboard_time_l2555_255546


namespace factorial_divisibility_l2555_255538

theorem factorial_divisibility :
  ¬(57 ∣ Nat.factorial 18) ∧ (57 ∣ Nat.factorial 19) := by
  sorry

end factorial_divisibility_l2555_255538


namespace units_digit_sum_base8_l2555_255552

-- Define a function to get the units digit in base 8
def units_digit_base8 (n : Nat) : Nat :=
  n % 8

-- Define the addition operation in base 8
def add_base8 (a b : Nat) : Nat :=
  (a + b) % 8

-- Theorem statement
theorem units_digit_sum_base8 :
  units_digit_base8 (add_base8 65 75) = 2 := by
  sorry

end units_digit_sum_base8_l2555_255552


namespace volume_of_regular_triangular_truncated_pyramid_l2555_255587

/-- A regular triangular truncated pyramid -/
structure RegularTriangularTruncatedPyramid where
  /-- Height of the pyramid -/
  H : ℝ
  /-- Angle between lateral edge and base -/
  α : ℝ
  /-- H is positive -/
  H_pos : 0 < H
  /-- α is between 0 and π/2 -/
  α_range : 0 < α ∧ α < Real.pi / 2
  /-- H is the geometric mean between the sides of the bases -/
  H_is_geometric_mean : ∃ a b : ℝ, 0 < b ∧ b < a ∧ H^2 = a * b

/-- Volume of a regular triangular truncated pyramid -/
noncomputable def volume (p : RegularTriangularTruncatedPyramid) : ℝ :=
  (p.H^3 * Real.sqrt 3) / (4 * Real.sin p.α ^ 2)

/-- Theorem stating the volume of a regular triangular truncated pyramid -/
theorem volume_of_regular_triangular_truncated_pyramid (p : RegularTriangularTruncatedPyramid) :
  volume p = (p.H^3 * Real.sqrt 3) / (4 * Real.sin p.α ^ 2) := by sorry

end volume_of_regular_triangular_truncated_pyramid_l2555_255587


namespace area_ratio_dodecagon_quadrilateral_l2555_255580

/-- A regular dodecagon -/
structure RegularDodecagon where
  -- We don't need to define the vertices explicitly
  area : ℝ

/-- A quadrilateral formed by connecting every third vertex of a regular dodecagon -/
structure Quadrilateral where
  area : ℝ

/-- The theorem stating the ratio of areas -/
theorem area_ratio_dodecagon_quadrilateral 
  (d : RegularDodecagon) 
  (q : Quadrilateral) : 
  q.area / d.area = Real.sqrt 3 / 6 := by
  sorry

end area_ratio_dodecagon_quadrilateral_l2555_255580


namespace odd_function_values_and_monotonicity_and_inequality_l2555_255560

noncomputable def f (a b x : ℝ) : ℝ := (b - 2^x) / (2^(x+1) + a)

theorem odd_function_values_and_monotonicity_and_inequality (a b : ℝ) :
  (∀ x, f a b x = -f a b (-x)) →
  (a = 2 ∧ b = 1) ∧
  (∀ x y, x < y → f 2 1 x > f 2 1 y) ∧
  (∀ k, (∀ x ≥ 1, f 2 1 (k * 3^x) + f 2 1 (3^x - 9^x + 2) > 0) ↔ k < 4/3) :=
by sorry

end odd_function_values_and_monotonicity_and_inequality_l2555_255560


namespace jasmine_solution_concentration_l2555_255577

theorem jasmine_solution_concentration 
  (initial_volume : ℝ) 
  (initial_concentration : ℝ) 
  (added_jasmine : ℝ) 
  (added_water : ℝ) 
  (final_concentration : ℝ) 
  (h1 : initial_volume = 100)
  (h2 : initial_concentration = 0.1)
  (h3 : added_jasmine = 5)
  (h4 : added_water = 67.5)
  (h5 : final_concentration = 0.08695652173913043) :
  let initial_jasmine := initial_volume * initial_concentration
  let total_jasmine := initial_jasmine + added_jasmine
  let final_volume := initial_volume + added_jasmine + added_water
  total_jasmine / final_volume = final_concentration :=
sorry

end jasmine_solution_concentration_l2555_255577


namespace multiply_213_by_16_l2555_255503

theorem multiply_213_by_16 (h : 213 * 1.6 = 340.8) : 213 * 16 = 3408 := by
  sorry

end multiply_213_by_16_l2555_255503


namespace continuity_at_two_l2555_255551

noncomputable def f (x : ℝ) : ℝ := (x^3 - 8) / (x^2 - 4)

theorem continuity_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 2| ∧ |x - 2| < δ → |f x - 3| < ε :=
sorry

end continuity_at_two_l2555_255551


namespace tiling_cost_difference_l2555_255571

/-- Represents a tiling option with its cost per tile and labor cost per square foot -/
structure TilingOption where
  tileCost : ℕ
  laborCost : ℕ

/-- Calculates the total cost for a tiling option -/
def totalCost (option : TilingOption) (totalArea : ℕ) (tilesPerSqFt : ℕ) : ℕ :=
  option.tileCost * totalArea * tilesPerSqFt + option.laborCost * totalArea

theorem tiling_cost_difference :
  let turquoise := TilingOption.mk 13 6
  let purple := TilingOption.mk 11 8
  let orange := TilingOption.mk 15 5
  let totalArea := 5 * 8 + 7 * 8 + 6 * 9
  let tilesPerSqFt := 4
  let turquoiseCost := totalCost turquoise totalArea tilesPerSqFt
  let purpleCost := totalCost purple totalArea tilesPerSqFt
  let orangeCost := totalCost orange totalArea tilesPerSqFt
  max turquoiseCost (max purpleCost orangeCost) - min turquoiseCost (min purpleCost orangeCost) = 1950 := by
  sorry

end tiling_cost_difference_l2555_255571


namespace solve_for_c_l2555_255558

theorem solve_for_c : ∃ C : ℝ, (4 * C + 5 = 25) ∧ (C = 5) := by sorry

end solve_for_c_l2555_255558


namespace rotated_angle_measure_l2555_255555

/-- Given an initial angle of 50 degrees that is rotated 580 degrees clockwise,
    the resulting acute angle is 90 degrees. -/
theorem rotated_angle_measure (initial_angle rotation : ℕ) : 
  initial_angle = 50 → 
  rotation = 580 → 
  (initial_angle + rotation) % 360 = 270 → 
  360 - ((initial_angle + rotation) % 360) = 90 :=
by sorry

end rotated_angle_measure_l2555_255555


namespace greatest_five_digit_multiple_of_6_l2555_255543

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

end greatest_five_digit_multiple_of_6_l2555_255543


namespace integer_list_mean_mode_l2555_255565

theorem integer_list_mean_mode (x : ℕ) : 
  x ≤ 120 →
  x > 0 →
  let list := [45, 76, 110, x, x]
  (list.sum / list.length : ℚ) = 2 * x →
  x = 29 := by
sorry

end integer_list_mean_mode_l2555_255565


namespace new_years_day_in_big_month_l2555_255519

-- Define the set of months
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

-- Define the set of holidays
inductive Holiday
| NewYearsDay
| ChildrensDay
| TeachersDay

-- Define a function to get the month of a holiday
def holiday_month (h : Holiday) : Month :=
  match h with
  | Holiday.NewYearsDay => Month.January
  | Holiday.ChildrensDay => Month.June
  | Holiday.TeachersDay => Month.September

-- Define the set of big months
def is_big_month (m : Month) : Prop :=
  m = Month.January ∨ m = Month.March ∨ m = Month.May ∨
  m = Month.July ∨ m = Month.August ∨ m = Month.October ∨
  m = Month.December

-- Theorem: New Year's Day falls in a big month
theorem new_years_day_in_big_month :
  is_big_month (holiday_month Holiday.NewYearsDay) :=
by sorry

end new_years_day_in_big_month_l2555_255519


namespace waysToSum1800_eq_45651_l2555_255557

/-- The number of ways to write 1800 as the sum of ones, twos, and threes, ignoring order -/
def waysToSum1800 : ℕ := sorry

/-- The target number we're considering -/
def targetNumber : ℕ := 1800

theorem waysToSum1800_eq_45651 : waysToSum1800 = 45651 := by sorry

end waysToSum1800_eq_45651_l2555_255557


namespace triangle_area_l2555_255517

/-- Given a triangle with perimeter 28 cm, inradius 2.5 cm, one angle of 75 degrees,
    and side lengths in the ratio 3:4:5, prove that its area is 35 cm². -/
theorem triangle_area (p : ℝ) (r : ℝ) (angle : ℝ) (a b c : ℝ)
  (h_perimeter : p = 28)
  (h_inradius : r = 2.5)
  (h_angle : angle = 75)
  (h_ratio : ∃ k : ℝ, a = 3 * k ∧ b = 4 * k ∧ c = 5 * k)
  (h_sides : a + b + c = p) :
  r * p / 2 = 35 := by
  sorry

end triangle_area_l2555_255517


namespace rosas_phone_book_calling_l2555_255513

/-- Rosa's phone book calling problem -/
theorem rosas_phone_book_calling (pages_last_week pages_total : ℝ) 
  (h1 : pages_last_week = 10.2)
  (h2 : pages_total = 18.8) :
  pages_total - pages_last_week = 8.6 := by
  sorry

end rosas_phone_book_calling_l2555_255513


namespace prob_end_multiple_3_l2555_255529

/-- The number of cards --/
def num_cards : ℕ := 15

/-- The probability of moving left on the spinner --/
def prob_left : ℚ := 1/4

/-- The probability of moving right on the spinner --/
def prob_right : ℚ := 3/4

/-- The probability of starting at a multiple of 3 --/
def prob_start_multiple_3 : ℚ := 1/3

/-- The probability of starting one more than a multiple of 3 --/
def prob_start_one_more : ℚ := 4/15

/-- The probability of starting one less than a multiple of 3 --/
def prob_start_one_less : ℚ := 1/3

/-- The probability of ending at a multiple of 3 after two spins --/
theorem prob_end_multiple_3 : 
  prob_start_multiple_3 * prob_left * prob_left +
  prob_start_one_more * prob_right * prob_right +
  prob_start_one_less * prob_left * prob_left = 7/30 := by sorry

end prob_end_multiple_3_l2555_255529


namespace sqrt_equality_l2555_255547

theorem sqrt_equality : ∃ (a b : ℕ+), a < b ∧ Real.sqrt (1 + Real.sqrt (45 + 18 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b := by
  sorry

end sqrt_equality_l2555_255547


namespace range_of_a_when_p_and_q_false_l2555_255596

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - 2

-- Define proposition p
def p (a : ℝ) : Prop :=
  ∃! x : ℝ, x ∈ Set.Icc (-1) 1 ∧ f a x = 0

-- Define proposition q
def q (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc (1/2) (3/2) → x^2 + 3*(a+1)*x + 2 ≤ 0

-- Theorem statement
theorem range_of_a_when_p_and_q_false :
  ∀ a : ℝ, ¬(p a ∧ q a) → a > -5/2 :=
sorry

end range_of_a_when_p_and_q_false_l2555_255596


namespace total_cost_is_correct_l2555_255566

def running_shoes_original_price : ℝ := 80
def casual_shoes_original_price : ℝ := 60
def running_shoes_discount : ℝ := 0.25
def casual_shoes_discount : ℝ := 0.40
def sales_tax_rate : ℝ := 0.08
def num_running_shoes : ℕ := 2
def num_casual_shoes : ℕ := 3

def total_cost : ℝ :=
  let running_shoes_discounted_price := running_shoes_original_price * (1 - running_shoes_discount)
  let casual_shoes_discounted_price := casual_shoes_original_price * (1 - casual_shoes_discount)
  let subtotal := num_running_shoes * running_shoes_discounted_price + num_casual_shoes * casual_shoes_discounted_price
  subtotal * (1 + sales_tax_rate)

theorem total_cost_is_correct : total_cost = 246.24 := by
  sorry

end total_cost_is_correct_l2555_255566


namespace sue_chewing_gums_l2555_255527

theorem sue_chewing_gums (mary_gums sam_gums total_gums : ℕ) (sue_gums : ℕ) :
  mary_gums = 5 →
  sam_gums = 10 →
  total_gums = 30 →
  total_gums = mary_gums + sam_gums + sue_gums →
  sue_gums = 15 := by
  sorry

end sue_chewing_gums_l2555_255527


namespace remainder_problem_l2555_255548

theorem remainder_problem (x : ℤ) (h : (x + 13) % 41 = 18) : x % 82 = 5 := by
  sorry

end remainder_problem_l2555_255548


namespace triangle_side_relation_l2555_255586

/-- Given a triangle ABC where the angles satisfy the equation 3α + 2β = 180°,
    prove that a^2 + bc = c^2, where a, b, and c are the lengths of the sides
    opposite to angles α, β, and γ respectively. -/
theorem triangle_side_relation (α β γ a b c : ℝ) : 
  0 < α ∧ 0 < β ∧ 0 < γ ∧ 
  α + β + γ = Real.pi ∧
  3 * α + 2 * β = Real.pi →
  a^2 + b * c = c^2 := by
  sorry

end triangle_side_relation_l2555_255586


namespace unique_remainder_mod_10_l2555_255534

theorem unique_remainder_mod_10 : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 10] := by sorry

end unique_remainder_mod_10_l2555_255534


namespace fraction_product_l2555_255599

theorem fraction_product : (2 : ℚ) / 9 * (-4 : ℚ) / 5 = (-8 : ℚ) / 45 := by
  sorry

end fraction_product_l2555_255599


namespace expression_simplification_l2555_255590

theorem expression_simplification (x y : ℝ) : 
  x^2*y - 3*x*y^2 + 2*y*x^2 - y^2*x = 3*x^2*y - 4*x*y^2 := by
  sorry

end expression_simplification_l2555_255590


namespace max_shapes_8x14_l2555_255564

/-- The number of grid points in an m × n rectangle --/
def gridPoints (m n : ℕ) : ℕ := (m + 1) * (n + 1)

/-- The number of grid points covered by each shape --/
def pointsPerShape : ℕ := 8

/-- The maximum number of shapes that can be placed in the grid --/
def maxShapes (m n : ℕ) : ℕ := (gridPoints m n) / pointsPerShape

theorem max_shapes_8x14 :
  maxShapes 8 14 = 16 := by sorry

end max_shapes_8x14_l2555_255564


namespace jenny_change_calculation_l2555_255531

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

end jenny_change_calculation_l2555_255531


namespace topsoil_cost_l2555_255528

/-- The cost of premium topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The number of cubic yards of topsoil to be purchased -/
def cubic_yards_to_purchase : ℝ := 7

/-- Theorem: The total cost of purchasing 7 cubic yards of premium topsoil is 1512 dollars -/
theorem topsoil_cost : 
  cost_per_cubic_foot * cubic_feet_per_cubic_yard * cubic_yards_to_purchase = 1512 := by
  sorry

end topsoil_cost_l2555_255528


namespace hyperbola_eccentricity_l2555_255536

noncomputable section

-- Define the hyperbola and its properties
def Hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

-- Define the foci
def LeftFocus (a c : ℝ) : ℝ × ℝ := (-c, 0)
def RightFocus (a c : ℝ) : ℝ × ℝ := (c, 0)

-- Define the point P on the right branch of the hyperbola
def P (a b : ℝ) : ℝ × ℝ := sorry

-- Define the perpendicular bisector of PF₁
def PerpendicularBisectorPF₁ (a b c : ℝ) : Set (ℝ × ℝ) := sorry

-- Define the distance from origin to line PF₁
def DistanceOriginToPF₁ (a b c : ℝ) : ℝ := sorry

-- Theorem statement
theorem hyperbola_eccentricity (a b c : ℝ) :
  (P a b ∈ Hyperbola a b) →
  (RightFocus a c ∈ PerpendicularBisectorPF₁ a b c) →
  (DistanceOriginToPF₁ a b c = a) →
  (c / a = 5 / 3) := by
  sorry

end

end hyperbola_eccentricity_l2555_255536


namespace hash_difference_seven_four_l2555_255541

-- Define the # operation
def hash (x y : ℤ) : ℤ := 2*x*y - 3*x - y

-- Theorem statement
theorem hash_difference_seven_four : hash 7 4 - hash 4 7 = -6 := by
  sorry

end hash_difference_seven_four_l2555_255541


namespace abc_value_l2555_255578

theorem abc_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 30 * Real.sqrt 5)
  (hac : a * c = 45 * Real.sqrt 5)
  (hbc : b * c = 40 * Real.sqrt 5) :
  a * b * c = 300 * Real.sqrt 3 * (5 : ℝ) ^ (1/4) := by
  sorry

end abc_value_l2555_255578


namespace banana_permutations_eq_60_l2555_255591

/-- The number of distinct permutations of the letters in "BANANA" -/
def banana_permutations : ℕ := 
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)

theorem banana_permutations_eq_60 : banana_permutations = 60 := by
  sorry

end banana_permutations_eq_60_l2555_255591


namespace min_board_sum_with_hundred_ones_l2555_255581

/-- Represents the state of the board --/
structure BoardState where
  ones : ℕ
  tens : ℕ
  twentyFives : ℕ

/-- Defines the allowed operations on the board --/
inductive Operation
  | replaceOneWithTen
  | replaceTenWithOneAndTwentyFive
  | replaceTwentyFiveWithTwoTens

/-- Applies an operation to the board state --/
def applyOperation (state : BoardState) (op : Operation) : BoardState :=
  match op with
  | Operation.replaceOneWithTen => 
    { ones := state.ones - 1, tens := state.tens + 1, twentyFives := state.twentyFives }
  | Operation.replaceTenWithOneAndTwentyFive => 
    { ones := state.ones + 1, tens := state.tens - 1, twentyFives := state.twentyFives + 1 }
  | Operation.replaceTwentyFiveWithTwoTens => 
    { ones := state.ones, tens := state.tens + 2, twentyFives := state.twentyFives - 1 }

/-- Calculates the sum of all numbers on the board --/
def boardSum (state : BoardState) : ℕ :=
  state.ones + 10 * state.tens + 25 * state.twentyFives

/-- The main theorem to prove --/
theorem min_board_sum_with_hundred_ones : 
  ∃ (final : BoardState) (ops : List Operation),
    final.ones = 100 ∧
    (∀ (state : BoardState), 
      state.ones = 100 → boardSum state ≥ boardSum final) ∧
    boardSum final = 1370 := by
  sorry


end min_board_sum_with_hundred_ones_l2555_255581


namespace no_prime_interior_angles_l2555_255502

def interior_angle (n : ℕ) : ℚ := (180 * (n - 2)) / n

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem no_prime_interior_angles :
  ∀ n : ℕ, 10 ≤ n → n < 20 →
    ¬(∃ k : ℕ, interior_angle n = k ∧ is_prime k) :=
by sorry

end no_prime_interior_angles_l2555_255502


namespace min_mutual_greetings_school_l2555_255516

/-- Represents a school with students and their greetings. -/
structure School :=
  (num_students : Nat)
  (greetings_per_student : Nat)
  (h_students : num_students = 400)
  (h_greetings : greetings_per_student = 200)

/-- The minimum number of pairs of students who have mutually greeted each other. -/
def min_mutual_greetings (s : School) : Nat :=
  s.greetings_per_student * s.num_students - Nat.choose s.num_students 2

/-- Theorem stating the minimum number of mutual greetings in the given school. -/
theorem min_mutual_greetings_school :
    ∀ s : School, min_mutual_greetings s = 200 :=
  sorry

end min_mutual_greetings_school_l2555_255516


namespace jordana_age_proof_l2555_255533

/-- Jennifer's current age -/
def jennifer_current_age : ℕ := 30 - 10

/-- Jennifer's age in 10 years -/
def jennifer_future_age : ℕ := 30

/-- Jordana's age in 10 years -/
def jordana_future_age : ℕ := 3 * jennifer_future_age

/-- Jordana's current age -/
def jordana_current_age : ℕ := jordana_future_age - 10

theorem jordana_age_proof : jordana_current_age = 80 := by
  sorry

end jordana_age_proof_l2555_255533


namespace planting_methods_eq_120_l2555_255532

/-- The number of ways to select and plant vegetables -/
def plantingMethods (totalVarieties : ℕ) (selectedVarieties : ℕ) (plots : ℕ) : ℕ :=
  Nat.choose totalVarieties selectedVarieties * Nat.factorial plots

/-- Theorem stating the number of planting methods for the given scenario -/
theorem planting_methods_eq_120 :
  plantingMethods 5 4 4 = 120 := by
  sorry

end planting_methods_eq_120_l2555_255532


namespace fractional_inequality_solution_set_l2555_255540

def solution_set : Set ℝ := Set.union (Set.Icc (-1/2) 1) (Set.Ico 1 3)

theorem fractional_inequality_solution_set :
  {x : ℝ | (x + 5) / ((x - 1)^2) ≥ 2 ∧ x ≠ 1} = solution_set := by
  sorry

end fractional_inequality_solution_set_l2555_255540


namespace circle_ratio_l2555_255501

theorem circle_ratio (r R c d : ℝ) (h1 : 0 < r) (h2 : r < R) (h3 : 0 < c) (h4 : c < d) :
  (π * R^2) = (c / d) * (π * R^2 - π * r^2) →
  R / r = (Real.sqrt c) / (Real.sqrt (d - c)) :=
by sorry

end circle_ratio_l2555_255501


namespace shaded_area_is_36_l2555_255592

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a square -/
structure Square :=
  (bottomLeft : Point)
  (sideLength : ℝ)

/-- Represents a right triangle -/
structure RightTriangle :=
  (bottomLeft : Point)
  (base : ℝ)
  (height : ℝ)

/-- Calculates the area of the shaded region -/
def shadedArea (square : Square) (triangle : RightTriangle) : ℝ :=
  sorry

/-- Theorem stating the area of the shaded region is 36 square units -/
theorem shaded_area_is_36 (square : Square) (triangle : RightTriangle) :
  square.bottomLeft = Point.mk 0 0 →
  square.sideLength = 12 →
  triangle.bottomLeft = Point.mk 12 0 →
  triangle.base = 12 →
  triangle.height = 12 →
  shadedArea square triangle = 36 :=
  sorry

end shaded_area_is_36_l2555_255592


namespace movie_of_the_year_fraction_l2555_255544

/-- The fraction of top-10 lists a film must appear on to be considered for "movie of the year" -/
def required_fraction (total_members : ℕ) (min_lists : ℚ) : ℚ :=
  min_lists / total_members

/-- Proof that the required fraction is 0.25 given the specific conditions -/
theorem movie_of_the_year_fraction :
  let total_members : ℕ := 775
  let min_lists : ℚ := 193.75
  required_fraction total_members min_lists = 0.25 := by
sorry

end movie_of_the_year_fraction_l2555_255544


namespace sin_theta_value_l2555_255595

theorem sin_theta_value (θ : Real) (h1 : θ ∈ Set.Icc (π/4) (π/2)) (h2 : Real.sin (2*θ) = 3*Real.sqrt 7/8) : 
  Real.sin θ = 3/4 := by
sorry

end sin_theta_value_l2555_255595


namespace ruth_math_class_time_l2555_255511

/-- Represents Ruth's school schedule and math class time --/
structure RuthSchedule where
  hours_per_day : ℝ
  days_per_week : ℝ
  math_class_percentage : ℝ

/-- Calculates the number of hours Ruth spends in math class per week --/
def math_class_hours (schedule : RuthSchedule) : ℝ :=
  schedule.hours_per_day * schedule.days_per_week * schedule.math_class_percentage

/-- Theorem stating that Ruth spends 10 hours per week in math class --/
theorem ruth_math_class_time :
  ∃ (schedule : RuthSchedule),
    schedule.hours_per_day = 8 ∧
    schedule.days_per_week = 5 ∧
    schedule.math_class_percentage = 0.25 ∧
    math_class_hours schedule = 10 := by
  sorry

end ruth_math_class_time_l2555_255511


namespace fraction_equality_l2555_255597

theorem fraction_equality (P Q : ℤ) : 
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 4 ∧ x ≠ -4 → 
    P / (x^2 - 5*x) + Q / (x + 4) = (x^2 - 3*x + 8) / (x^3 - 5*x^2 + 4*x)) →
  (Q : ℚ) / (P : ℚ) = 7 / 2 := by
sorry

end fraction_equality_l2555_255597


namespace stating_num_distributions_eq_16_l2555_255593

/-- Represents the number of classes -/
def num_classes : ℕ := 4

/-- Represents the number of "Outstanding Class" spots -/
def num_outstanding_class : ℕ := 4

/-- Represents the number of "Outstanding Group Branch" spots -/
def num_outstanding_group : ℕ := 1

/-- Represents the total number of spots to be distributed -/
def total_spots : ℕ := num_outstanding_class + num_outstanding_group

/-- 
  Theorem stating that the number of ways to distribute the spots among classes,
  with each class receiving at least one spot, is equal to 16
-/
theorem num_distributions_eq_16 : 
  (Finset.univ.filter (fun f : Fin num_classes → Fin (total_spots + 1) => 
    (∀ i, f i > 0) ∧ (Finset.sum Finset.univ f = total_spots))).card = 16 := by
  sorry


end stating_num_distributions_eq_16_l2555_255593


namespace occupancy_theorem_hundred_mathematicians_l2555_255598

/-- The number of ways k mathematicians can occupy k rooms under the given conditions -/
def occupancy_ways (k : ℕ) : ℕ :=
  2^(k - 1)

/-- Theorem stating that the number of ways k mathematicians can occupy k rooms is 2^(k-1) -/
theorem occupancy_theorem (k : ℕ) (h : k > 0) :
  occupancy_ways k = 2^(k - 1) :=
by sorry

/-- Corollary for the specific case of 100 mathematicians -/
theorem hundred_mathematicians :
  occupancy_ways 100 = 2^99 :=
by sorry

end occupancy_theorem_hundred_mathematicians_l2555_255598


namespace geometric_sequence_sum_l2555_255518

/-- Given a geometric sequence {a_n} where a_2 + a_3 = 1 and a_3 + a_4 = -2,
    prove that a_5 + a_6 + a_7 = 24 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- {a_n} is a geometric sequence
  a 2 + a 3 = 1 →                           -- a_2 + a_3 = 1
  a 3 + a 4 = -2 →                          -- a_3 + a_4 = -2
  a 5 + a 6 + a 7 = 24 :=                   -- Conclusion to prove
by sorry

end geometric_sequence_sum_l2555_255518


namespace quadratic_factorization_l2555_255526

theorem quadratic_factorization (x : ℝ) : 15 * x^2 + 10 * x - 20 = 5 * (x - 1) * (3 * x + 4) := by
  sorry

end quadratic_factorization_l2555_255526


namespace simplify_and_rationalize_l2555_255500

theorem simplify_and_rationalize :
  (Real.sqrt 2 / Real.sqrt 3) * (Real.sqrt 4 / Real.sqrt 5) *
  (Real.sqrt 6 / Real.sqrt 7) * (Real.sqrt 8 / Real.sqrt 9) =
  16 * Real.sqrt 105 / 315 := by sorry

end simplify_and_rationalize_l2555_255500


namespace log_ratio_squared_l2555_255506

theorem log_ratio_squared (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (hx1 : x ≠ 1) (hy1 : y ≠ 1) 
  (hlog : Real.log x / Real.log 3 = Real.log 81 / Real.log y) 
  (hprod : x * y = 243) : 
  (Real.log (x / y) / Real.log 3)^2 = 9 := by
  sorry

end log_ratio_squared_l2555_255506


namespace quadratic_equation_roots_l2555_255575

theorem quadratic_equation_roots (m : ℝ) :
  ((-1 : ℝ)^2 + m * (-1) - 5 = 0) →
  (m = -4 ∧ ∃ x₂ : ℝ, x₂ = 5 ∧ x₂^2 + m * x₂ - 5 = 0) :=
by sorry

end quadratic_equation_roots_l2555_255575


namespace shortest_chain_no_self_intersections_l2555_255585

/-- A polygonal chain in a plane -/
structure PolygonalChain (n : ℕ) where
  points : Fin n → ℝ × ℝ
  
/-- The length of a polygonal chain -/
def length (chain : PolygonalChain n) : ℝ := sorry

/-- A polygonal chain has self-intersections -/
def has_self_intersections (chain : PolygonalChain n) : Prop := sorry

/-- A polygonal chain is the shortest among all chains connecting the same points -/
def is_shortest (chain : PolygonalChain n) : Prop := 
  ∀ other : PolygonalChain n, chain.points = other.points → length chain ≤ length other

/-- The shortest polygonal chain connecting n points in a plane has no self-intersections -/
theorem shortest_chain_no_self_intersections (n : ℕ) (chain : PolygonalChain n) :
  is_shortest chain → ¬ has_self_intersections chain :=
sorry

end shortest_chain_no_self_intersections_l2555_255585


namespace not_p_and_not_q_implies_not_p_and_not_q_l2555_255525

theorem not_p_and_not_q_implies_not_p_and_not_q (p q : Prop) :
  (¬p ∧ ¬q) → (¬p ∧ ¬q) := by
  sorry

end not_p_and_not_q_implies_not_p_and_not_q_l2555_255525


namespace remainder_divisibility_l2555_255539

theorem remainder_divisibility (n : ℕ) : 
  (∃ k : ℕ, n = 3 * k + 2) ∧ 
  (∃ m : ℕ, k = 4 * m + 3) → 
  n % 6 = 5 := by
sorry

end remainder_divisibility_l2555_255539


namespace probability_sum_10_three_dice_l2555_255594

-- Define a die as having 6 faces
def die : Finset ℕ := Finset.range 6

-- Define the total number of outcomes when rolling three dice
def total_outcomes : ℕ := die.card ^ 3

-- Define the favorable outcomes (sum of 10)
def favorable_outcomes : ℕ := 27

-- Theorem statement
theorem probability_sum_10_three_dice :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 8 := by
  sorry

end probability_sum_10_three_dice_l2555_255594


namespace toms_age_ratio_l2555_255588

/-- Tom's age problem -/
theorem toms_age_ratio (T N : ℚ) : T > 0 → N > 0 →
  (∃ (a b c d : ℚ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ T = a + b + c + d) →
  (T - N = 3 * (T - 4 * N)) →
  T / N = 11 / 2 := by
sorry

end toms_age_ratio_l2555_255588


namespace proportional_relationship_l2555_255584

/-- Given that x is directly proportional to y^4, y is inversely proportional to z^2,
    and x = 4 when z = 3, prove that x = 3/192 when z = 6. -/
theorem proportional_relationship (x y z : ℝ) (k : ℝ) 
    (h1 : ∃ m : ℝ, x = m * y^4)
    (h2 : ∃ n : ℝ, y = n / z^2)
    (h3 : x = 4 ∧ z = 3 → x * z^8 = k)
    (h4 : x * z^8 = k) :
    z = 6 → x = 3 / 192 := by
  sorry

end proportional_relationship_l2555_255584


namespace smallest_solution_floor_equation_sqrt_152_is_solution_sqrt_152_is_smallest_solution_l2555_255523

theorem smallest_solution_floor_equation :
  ∀ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 23) → x ≥ Real.sqrt 152 :=
by sorry

theorem sqrt_152_is_solution :
  ⌊(Real.sqrt 152)^2⌋ - ⌊Real.sqrt 152⌋^2 = 23 :=
by sorry

theorem sqrt_152_is_smallest_solution :
  ∀ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 23) → x ≥ Real.sqrt 152 ∧
  ⌊(Real.sqrt 152)^2⌋ - ⌊Real.sqrt 152⌋^2 = 23 :=
by sorry

end smallest_solution_floor_equation_sqrt_152_is_solution_sqrt_152_is_smallest_solution_l2555_255523


namespace min_value_of_reciprocal_sum_l2555_255514

/-- Given a line ax - by + 2 = 0 (a > 0, b > 0) intercepted by the circle x^2 + y^2 + 2x - 4y + 1 = 0
    with a chord length of 4, prove that the minimum value of 1/a + 1/b is 3/2 + √2 -/
theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), a * x - b * y + 2 = 0 ∧ x^2 + y^2 + 2*x - 4*y + 1 = 0) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    a * x₁ - b * y₁ + 2 = 0 ∧ x₁^2 + y₁^2 + 2*x₁ - 4*y₁ + 1 = 0 ∧
    a * x₂ - b * y₂ + 2 = 0 ∧ x₂^2 + y₂^2 + 2*x₂ - 4*y₂ + 1 = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16) →
  (1 / a + 1 / b) ≥ 3/2 + Real.sqrt 2 :=
sorry

end min_value_of_reciprocal_sum_l2555_255514


namespace gage_received_fraction_l2555_255505

-- Define the initial numbers of cubes
def grady_red : ℕ := 20
def grady_blue : ℕ := 15
def gage_initial_red : ℕ := 10
def gage_initial_blue : ℕ := 12

-- Define the fraction of blue cubes Gage received
def blue_fraction : ℚ := 1/3

-- Define the total number of cubes Gage has after receiving some from Grady
def gage_total : ℕ := 35

-- Define the fraction of red cubes Gage received as a rational number
def red_fraction : ℚ := 2/5

-- Theorem statement
theorem gage_received_fraction :
  (gage_initial_red : ℚ) + red_fraction * grady_red + 
  (gage_initial_blue : ℚ) + blue_fraction * grady_blue = gage_total :=
sorry

end gage_received_fraction_l2555_255505


namespace regular_hexagon_diagonal_l2555_255572

theorem regular_hexagon_diagonal (side_length : ℝ) (h : side_length = 10) :
  let diagonal_length := side_length * Real.sqrt 3
  diagonal_length = 10 * Real.sqrt 3 := by
sorry

end regular_hexagon_diagonal_l2555_255572


namespace simplify_cube_roots_l2555_255556

theorem simplify_cube_roots (h1 : 343 = 7^3) (h2 : 125 = 5^3) :
  (343 : ℝ)^(1/3) * (125 : ℝ)^(1/3) = 35 := by
  sorry

end simplify_cube_roots_l2555_255556


namespace cosine_irrationality_l2555_255507

theorem cosine_irrationality (n : ℕ) (h : n ≥ 2) : Irrational (Real.cos (π / 2^n)) := by
  sorry

end cosine_irrationality_l2555_255507


namespace roots_sum_of_squares_l2555_255510

theorem roots_sum_of_squares (p q : ℝ) : 
  (p^2 - 5*p + 6 = 0) → (q^2 - 5*q + 6 = 0) → p ≠ q → p^2 + q^2 = 13 := by
  sorry

end roots_sum_of_squares_l2555_255510


namespace baseball_cards_packs_l2555_255570

/-- The number of people who bought baseball cards -/
def num_people : ℕ := 4

/-- The number of baseball cards each person bought -/
def cards_per_person : ℕ := 540

/-- The number of cards in each pack -/
def cards_per_pack : ℕ := 20

/-- The total number of packs of baseball cards for all people -/
def total_packs : ℕ := (num_people * cards_per_person) / cards_per_pack

theorem baseball_cards_packs : total_packs = 108 := by
  sorry

end baseball_cards_packs_l2555_255570


namespace rachel_reading_homework_l2555_255573

/-- The number of pages of reading homework Rachel had to complete -/
def reading_pages : ℕ := 15 - 2 - 10

/-- The total number of pages Rachel had to complete -/
def total_pages : ℕ := 15

/-- The number of pages of math homework Rachel had to complete -/
def math_pages : ℕ := 2

/-- The number of pages of biology homework Rachel had to complete -/
def biology_pages : ℕ := 10

theorem rachel_reading_homework :
  reading_pages = 3 ∧
  total_pages = math_pages + reading_pages + biology_pages :=
sorry

end rachel_reading_homework_l2555_255573


namespace toms_profit_l2555_255574

/-- Calculate Tom's profit from the world's largest dough ball event -/
theorem toms_profit (flour_needed : ℕ) (flour_bag_size : ℕ) (flour_bag_cost : ℕ)
                    (salt_needed : ℕ) (salt_cost_per_pound : ℚ)
                    (promotion_cost : ℕ) (ticket_price : ℕ) (tickets_sold : ℕ) :
  flour_needed = 500 →
  flour_bag_size = 50 →
  flour_bag_cost = 20 →
  salt_needed = 10 →
  salt_cost_per_pound = 1/5 →
  promotion_cost = 1000 →
  ticket_price = 20 →
  tickets_sold = 500 →
  (tickets_sold * ticket_price : ℤ) - 
  (((flour_needed / flour_bag_size) * flour_bag_cost : ℕ) + 
   (salt_needed * salt_cost_per_pound).num + 
   promotion_cost : ℤ) = 8798 :=
by sorry

end toms_profit_l2555_255574


namespace distribution_difference_l2555_255583

theorem distribution_difference (total_amount : ℕ) (group1 : ℕ) (group2 : ℕ)
  (h1 : total_amount = 5040)
  (h2 : group1 = 14)
  (h3 : group2 = 18)
  (h4 : group1 < group2) :
  (total_amount / group1) - (total_amount / group2) = 80 := by
  sorry

end distribution_difference_l2555_255583


namespace joes_speed_to_petes_speed_ratio_l2555_255542

/-- Prove that the ratio of Joe's speed to Pete's speed is 2:1 -/
theorem joes_speed_to_petes_speed_ratio (
  time : ℝ)
  (total_distance : ℝ)
  (joes_speed : ℝ)
  (h1 : time = 40)
  (h2 : total_distance = 16)
  (h3 : joes_speed = 0.266666666667)
  : joes_speed / ((total_distance - joes_speed * time) / time) = 2 := by
  sorry

end joes_speed_to_petes_speed_ratio_l2555_255542


namespace probability_adjacent_vertices_decagon_l2555_255512

/-- A decagon is a polygon with 10 vertices -/
def Decagon := Fin 10

/-- Two vertices in a decagon are adjacent if their indices differ by 1 (mod 10) -/
def adjacent (a b : Decagon) : Prop :=
  (a.val + 1) % 10 = b.val ∨ (b.val + 1) % 10 = a.val

/-- The total number of ways to choose 2 distinct vertices from a decagon -/
def total_choices : ℕ := 10 * 9 / 2

/-- The number of ways to choose 2 adjacent vertices from a decagon -/
def adjacent_choices : ℕ := 10

theorem probability_adjacent_vertices_decagon :
  (adjacent_choices : ℚ) / total_choices = 2 / 9 := by
  sorry

#eval (adjacent_choices : ℚ) / total_choices

end probability_adjacent_vertices_decagon_l2555_255512


namespace quadruple_solutions_l2555_255563

theorem quadruple_solutions : 
  ∀ (a b c d : ℕ+), 
    (a * b + 2 * a - b = 58 ∧ 
     b * c + 4 * b + 2 * c = 300 ∧ 
     c * d - 6 * c + 4 * d = 101) → 
    ((a = 3 ∧ b = 26 ∧ c = 7 ∧ d = 13) ∨ 
     (a = 15 ∧ b = 2 ∧ c = 73 ∧ d = 7)) := by
  sorry

end quadruple_solutions_l2555_255563


namespace probability_two_red_shoes_l2555_255520

-- Define the number of red and green shoes
def num_red_shoes : ℕ := 7
def num_green_shoes : ℕ := 3

-- Define the total number of shoes
def total_shoes : ℕ := num_red_shoes + num_green_shoes

-- Define the number of shoes to be drawn
def shoes_drawn : ℕ := 2

-- Define the probability of drawing two red shoes
def prob_two_red_shoes : ℚ := 7 / 15

-- Theorem statement
theorem probability_two_red_shoes :
  (Nat.choose num_red_shoes shoes_drawn : ℚ) / (Nat.choose total_shoes shoes_drawn : ℚ) = prob_two_red_shoes :=
sorry

end probability_two_red_shoes_l2555_255520


namespace x_value_proof_l2555_255562

theorem x_value_proof (x y : ℝ) : 
  x / (x + 1) = (y^2 + 3*y + 1) / (y^2 + 3*y + 2) → x = y^2 + 3*y + 1 := by
  sorry

end x_value_proof_l2555_255562


namespace equation_one_solutions_l2555_255530

theorem equation_one_solutions (x : ℝ) : (x - 1)^2 = 4 ↔ x = 3 ∨ x = -1 := by
  sorry

end equation_one_solutions_l2555_255530


namespace ice_cream_sundaes_l2555_255509

theorem ice_cream_sundaes (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 3) :
  Nat.choose n k = 56 := by
  sorry

end ice_cream_sundaes_l2555_255509


namespace modulus_of_z_l2555_255561

def i : ℂ := Complex.I

def z : ℂ := (1 + i) * (1 + 2*i)

theorem modulus_of_z : Complex.abs z = Real.sqrt 10 := by sorry

end modulus_of_z_l2555_255561


namespace cubic_polynomial_unique_solution_l2555_255545

def cubic_polynomial (P : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, P x = a + b * x + c * x^2 + d * x^3

theorem cubic_polynomial_unique_solution 
  (P : ℝ → ℝ) 
  (h_cubic : cubic_polynomial P) 
  (h_neg_one : P (-1) = 2)
  (h_zero : P 0 = 3)
  (h_one : P 1 = 1)
  (h_two : P 2 = 15) :
  ∀ x, P x = 3 + x - 2 * x^2 - x^3 := by
sorry

end cubic_polynomial_unique_solution_l2555_255545


namespace greatest_common_divisor_546_126_under_30_l2555_255559

theorem greatest_common_divisor_546_126_under_30 : 
  ∃ (n : ℕ), n = 21 ∧ 
  n ∣ 546 ∧ 
  n < 30 ∧ 
  n ∣ 126 ∧
  ∀ (m : ℕ), m ∣ 546 → m < 30 → m ∣ 126 → m ≤ n :=
by sorry

end greatest_common_divisor_546_126_under_30_l2555_255559


namespace flush_probability_l2555_255522

/-- Represents the number of players in the card game -/
def num_players : ℕ := 4

/-- Represents the total number of cards in the deck -/
def total_cards : ℕ := 20

/-- Represents the number of cards per suit -/
def cards_per_suit : ℕ := 5

/-- Represents the number of cards dealt to each player -/
def cards_per_player : ℕ := 5

/-- Calculates the probability of at least one player having a flush after card exchange -/
def probability_of_flush : ℚ := 8 / 969

/-- Theorem stating the probability of at least one player having a flush after card exchange -/
theorem flush_probability : 
  probability_of_flush = 8 / 969 :=
sorry

end flush_probability_l2555_255522


namespace sum_of_divisors_900_prime_factors_l2555_255521

theorem sum_of_divisors_900_prime_factors : 
  let n := 900
  let sum_of_divisors := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id
  (Nat.factors sum_of_divisors).toFinset.card = 3 := by
  sorry

end sum_of_divisors_900_prime_factors_l2555_255521


namespace mississippi_permutations_count_l2555_255569

def mississippi_permutations : ℕ :=
  let total_letters := 11
  let m_count := 1
  let i_count := 4
  let s_count := 4
  let p_count := 2
  Nat.factorial total_letters / (Nat.factorial m_count * Nat.factorial i_count * Nat.factorial s_count * Nat.factorial p_count)

theorem mississippi_permutations_count : mississippi_permutations = 34650 := by
  sorry

end mississippi_permutations_count_l2555_255569


namespace school_classes_count_l2555_255549

/-- Represents a school with classes -/
structure School where
  total_students : ℕ
  largest_class : ℕ
  class_difference : ℕ

/-- Calculates the number of classes in a school -/
def number_of_classes (s : School) : ℕ :=
  sorry

/-- Theorem stating that for a school with 120 students, largest class of 28, 
    and class difference of 2, the number of classes is 5 -/
theorem school_classes_count (s : School) 
  (h1 : s.total_students = 120) 
  (h2 : s.largest_class = 28) 
  (h3 : s.class_difference = 2) : 
  number_of_classes s = 5 := by
  sorry

end school_classes_count_l2555_255549


namespace distribute_five_balls_two_boxes_l2555_255579

/-- The number of ways to distribute n distinct objects into two boxes,
    where box 1 contains at least k objects and box 2 contains at least m objects. -/
def distribute (n k m : ℕ) : ℕ :=
  (Finset.range (n - k - m + 1)).sum (λ i => Nat.choose n (k + i))

/-- Theorem: There are 25 ways to distribute 5 distinct objects into two boxes,
    where box 1 contains at least 1 object and box 2 contains at least 2 objects. -/
theorem distribute_five_balls_two_boxes : distribute 5 1 2 = 25 := by
  sorry

end distribute_five_balls_two_boxes_l2555_255579


namespace remainder_eleven_pow_2023_mod_8_l2555_255508

theorem remainder_eleven_pow_2023_mod_8 : 11^2023 % 8 = 3 := by
  sorry

end remainder_eleven_pow_2023_mod_8_l2555_255508


namespace equation_solution_l2555_255589

theorem equation_solution :
  ∃! x : ℝ, ∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 4.5 = 0 :=
by
  -- The unique value of x that satisfies the equation for all y is 3/2
  use 3/2
  sorry

end equation_solution_l2555_255589


namespace paper_length_calculation_l2555_255524

/-- Calculates the length of paper wrapped around a tube -/
theorem paper_length_calculation 
  (paper_width : ℝ) 
  (initial_diameter : ℝ) 
  (final_diameter : ℝ) 
  (num_layers : ℕ) 
  (h1 : paper_width = 4)
  (h2 : initial_diameter = 4)
  (h3 : final_diameter = 16)
  (h4 : num_layers = 500) :
  (π * num_layers * (initial_diameter + final_diameter) / 2) / 100 = 50 * π := by
sorry

end paper_length_calculation_l2555_255524


namespace sum_of_coefficients_equals_one_l2555_255554

theorem sum_of_coefficients_equals_one (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^11 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                           a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = 1 :=
by
  sorry

end sum_of_coefficients_equals_one_l2555_255554


namespace age_problem_l2555_255576

/-- Given three people a, b, and c, where:
  * a is two years older than b
  * b is twice as old as c
  * The sum of their ages is 22
  Prove that b is 8 years old. -/
theorem age_problem (a b c : ℕ) 
  (h1 : a = b + 2) 
  (h2 : b = 2 * c) 
  (h3 : a + b + c = 22) : 
  b = 8 := by
  sorry

end age_problem_l2555_255576


namespace olivias_quarters_l2555_255568

theorem olivias_quarters (spent : ℕ) (left : ℕ) : spent = 4 → left = 7 → spent + left = 11 := by
  sorry

end olivias_quarters_l2555_255568


namespace constant_z_is_plane_l2555_255504

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- The set of points satisfying z = c in cylindrical coordinates -/
def ConstantZSet (c : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.z = c}

/-- Definition of a plane in cylindrical coordinates -/
def IsPlane (S : Set CylindricalPoint) : Prop :=
  ∃ (a b c d : ℝ), c ≠ 0 ∧ ∀ p ∈ S, a * p.r * (Real.cos p.θ) + b * p.r * (Real.sin p.θ) + c * p.z = d

theorem constant_z_is_plane (c : ℝ) : IsPlane (ConstantZSet c) := by
  sorry

end constant_z_is_plane_l2555_255504


namespace quadratic_roots_relation_l2555_255550

theorem quadratic_roots_relation (u v : ℝ) (m n : ℝ) : 
  (3 * u^2 + 4 * u + 5 = 0) →
  (3 * v^2 + 4 * v + 5 = 0) →
  ((u^2 + 1)^2 + m * (u^2 + 1) + n = 0) →
  ((v^2 + 1)^2 + m * (v^2 + 1) + n = 0) →
  m = -4/9 := by
sorry

end quadratic_roots_relation_l2555_255550


namespace distance_to_larger_cross_section_l2555_255535

/-- A right octagonal pyramid with two parallel cross sections -/
structure OctagonalPyramid where
  /-- Area of the smaller cross section in square feet -/
  area_small : ℝ
  /-- Area of the larger cross section in square feet -/
  area_large : ℝ
  /-- Distance between the two cross sections in feet -/
  distance_between : ℝ

/-- The distance from the apex to the larger cross section -/
def distance_to_larger (p : OctagonalPyramid) : ℝ := sorry

theorem distance_to_larger_cross_section
  (p : OctagonalPyramid)
  (h1 : p.area_small = 400 * Real.sqrt 2)
  (h2 : p.area_large = 900 * Real.sqrt 2)
  (h3 : p.distance_between = 10) :
  distance_to_larger p = 30 := by sorry

end distance_to_larger_cross_section_l2555_255535


namespace smallest_sum_of_squares_and_cubes_infinitely_many_coprime_sums_l2555_255515

-- Define a function to check if a number is the sum of two squares
def isSumOfTwoSquares (n : ℕ) : Prop :=
  ∃ a b : ℕ, a^2 + b^2 = n ∧ a > 0 ∧ b > 0

-- Define a function to check if a number is the sum of two cubes
def isSumOfTwoCubes (n : ℕ) : Prop :=
  ∃ a b : ℕ, a^3 + b^3 = n ∧ a > 0 ∧ b > 0

-- Define a function to check if two numbers are coprime
def areCoprime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

theorem smallest_sum_of_squares_and_cubes :
  (∀ n : ℕ, n > 2 ∧ n < 65 → ¬(isSumOfTwoSquares n ∧ isSumOfTwoCubes n)) ∧
  (isSumOfTwoSquares 65 ∧ isSumOfTwoCubes 65) :=
sorry

theorem infinitely_many_coprime_sums :
  ∀ k : ℕ, ∃ n : ℕ,
    (∃ a b : ℕ, n = a^2 + b^2 ∧ areCoprime a b) ∧
    (∃ c d : ℕ, n = c^3 + d^3 ∧ areCoprime c d) ∧
    n > k :=
sorry

end smallest_sum_of_squares_and_cubes_infinitely_many_coprime_sums_l2555_255515
