import Mathlib

namespace quadratic_equation_one_solution_positive_n_value_l3286_328677

theorem quadratic_equation_one_solution (n : ℝ) : 
  (∃! x : ℝ, 4 * x^2 + n * x + 16 = 0) → n = 16 ∨ n = -16 :=
by sorry

theorem positive_n_value (n : ℝ) :
  (∃! x : ℝ, 4 * x^2 + n * x + 16 = 0) ∧ n > 0 → n = 16 :=
by sorry

end quadratic_equation_one_solution_positive_n_value_l3286_328677


namespace point_A_coordinates_l3286_328635

-- Define the Cartesian coordinate system
structure Point where
  x : ℝ
  y : ℝ

-- Define the translation operations
def translateLeft (p : Point) (d : ℝ) : Point :=
  { x := p.x - d, y := p.y }

def translateUp (p : Point) (d : ℝ) : Point :=
  { x := p.x, y := p.y + d }

-- Theorem statement
theorem point_A_coordinates 
  (A : Point) 
  (B : Point)
  (C : Point)
  (hB : ∃ d : ℝ, translateLeft A d = B)
  (hC : ∃ d : ℝ, translateUp A d = C)
  (hBcoord : B.x = 1 ∧ B.y = 2)
  (hCcoord : C.x = 3 ∧ C.y = 4) :
  A.x = 3 ∧ A.y = 2 := by
  sorry

end point_A_coordinates_l3286_328635


namespace complement_of_A_in_U_l3286_328650

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | |x - 1| > 1}

-- State the theorem
theorem complement_of_A_in_U : 
  (Set.univ \ A) = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by
  sorry

end complement_of_A_in_U_l3286_328650


namespace percent_relation_l3286_328630

theorem percent_relation (a b : ℝ) (h : a = 1.2 * b) : 
  2 * b / a = 5 / 3 := by sorry

end percent_relation_l3286_328630


namespace average_hamburgers_per_day_l3286_328695

-- Define the total number of hamburgers sold
def total_hamburgers : ℕ := 49

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the average number of hamburgers sold per day
def average_hamburgers : ℚ := total_hamburgers / days_in_week

-- Theorem statement
theorem average_hamburgers_per_day :
  average_hamburgers = 7 := by sorry

end average_hamburgers_per_day_l3286_328695


namespace total_amount_proof_l3286_328625

/-- Given an amount divided into two parts, where one part is invested at 3% and the other at 5%,
    this theorem proves that the total amount is 4000 when the first part is 2800 and
    the total annual interest is 144. -/
theorem total_amount_proof (T A : ℝ) : 
  A = 2800 → 
  0.03 * A + 0.05 * (T - A) = 144 → 
  T = 4000 := by
sorry

end total_amount_proof_l3286_328625


namespace line_intercepts_sum_l3286_328668

theorem line_intercepts_sum (c : ℝ) : 
  (∃ (x y : ℝ), 6*x + 9*y + c = 0 ∧ x + y = 30) → c = -108 := by
  sorry

end line_intercepts_sum_l3286_328668


namespace sarahs_trip_distance_l3286_328607

theorem sarahs_trip_distance :
  ∀ y : ℚ, (y / 4 + 25 + y / 6 = y) → y = 300 / 7 := by
  sorry

end sarahs_trip_distance_l3286_328607


namespace horner_rule_operations_l3286_328641

/-- Horner's Rule representation of a polynomial -/
def horner_representation (coeffs : List ℤ) : ℤ → ℤ :=
  fun x => coeffs.foldl (fun acc a => acc * x + a) 0

/-- Count of operations in Horner's Rule evaluation -/
def horner_operation_count (coeffs : List ℤ) : ℕ × ℕ :=
  (coeffs.length - 1, coeffs.length - 1)

/-- The polynomial f(x) = 4x^5 - 3x^4 + 6x - 9 -/
def f : List ℤ := [4, -3, 0, 0, 6, -9]

theorem horner_rule_operations :
  horner_operation_count f = (5, 3) := by sorry

end horner_rule_operations_l3286_328641


namespace chess_match_max_ab_l3286_328651

theorem chess_match_max_ab (a b c : ℝ) : 
  0 ≤ a ∧ a < 1 ∧
  0 ≤ b ∧ b < 1 ∧
  0 ≤ c ∧ c < 1 ∧
  a + b + c = 1 ∧
  3*a + b = 1 →
  a * b ≤ 1/12 := by
sorry

end chess_match_max_ab_l3286_328651


namespace jenny_investment_l3286_328698

theorem jenny_investment (total : ℝ) (ratio : ℝ) (real_estate : ℝ) : 
  total = 220000 →
  ratio = 7 →
  real_estate = ratio * (total / (ratio + 1)) →
  real_estate = 192500 := by
sorry

end jenny_investment_l3286_328698


namespace cow_manure_plant_height_l3286_328655

/-- The height of the cow manure plant given the heights of control and bone meal plants -/
theorem cow_manure_plant_height
  (control_height : ℝ)
  (bone_meal_percentage : ℝ)
  (cow_manure_percentage : ℝ)
  (h1 : control_height = 36)
  (h2 : bone_meal_percentage = 1.25)
  (h3 : cow_manure_percentage = 2) :
  control_height * bone_meal_percentage * cow_manure_percentage = 90 := by
  sorry

#check cow_manure_plant_height

end cow_manure_plant_height_l3286_328655


namespace total_money_division_l3286_328690

theorem total_money_division (b c : ℕ) (total : ℕ) : 
  (b : ℚ) / c = 4 / 16 →
  c * 100 = 1600 →
  total = b * 100 + c * 100 →
  total = 2000 := by
sorry

end total_money_division_l3286_328690


namespace larger_cube_surface_area_l3286_328606

theorem larger_cube_surface_area (small_cube_surface_area : ℝ) (num_small_cubes : ℕ) :
  small_cube_surface_area = 24 →
  num_small_cubes = 125 →
  ∃ (larger_cube_surface_area : ℝ), larger_cube_surface_area = 600 := by
  sorry

end larger_cube_surface_area_l3286_328606


namespace perpendicular_line_x_intercept_l3286_328623

/-- Given a line L1: 2x + 3y = 12, and a perpendicular line L2 with y-intercept -1,
    the x-intercept of L2 is 2/3. -/
theorem perpendicular_line_x_intercept :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 2 * x + 3 * y = 12
  let m1 : ℝ := -2 / 3  -- slope of L1
  let m2 : ℝ := 3 / 2   -- slope of L2 (perpendicular to L1)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ y = m2 * x - 1  -- equation of L2
  (∀ x y, L2 x y → (x = 0 → y = -1)) →  -- y-intercept of L2 is -1
  (∀ x, L2 x 0 → x = 2/3) :=  -- x-intercept of L2 is 2/3
by sorry

end perpendicular_line_x_intercept_l3286_328623


namespace modified_lottery_win_probability_l3286_328685

/-- The number of balls for the MegaBall drawing -/
def megaBallCount : ℕ := 30

/-- The number of balls for the WinnerBalls drawing -/
def winnerBallCount : ℕ := 46

/-- The number of WinnerBalls picked -/
def pickedWinnerBallCount : ℕ := 5

/-- The probability of winning the modified lottery game -/
def winProbability : ℚ := 1 / 34321980

theorem modified_lottery_win_probability :
  winProbability = 1 / (megaBallCount * (Nat.choose winnerBallCount pickedWinnerBallCount)) :=
by sorry

end modified_lottery_win_probability_l3286_328685


namespace trig_expression_equals_seven_l3286_328616

theorem trig_expression_equals_seven :
  2 * Real.sin (390 * π / 180) - Real.tan (-45 * π / 180) + 5 * Real.cos (360 * π / 180) = 7 := by
  sorry

end trig_expression_equals_seven_l3286_328616


namespace triangle_angle_ratio_l3286_328624

-- Define a triangle with side lengths a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_angle_ratio (t : Triangle) 
  (h1 : t.a^2 = t.b * (t.b + t.c)) -- Given condition
  (h2 : t.A > 0 ∧ t.B > 0 ∧ t.C > 0) -- Angles are positive
  (h3 : t.A + t.B + t.C = π) -- Sum of angles in a triangle
  (h4 : t.a > 0 ∧ t.b > 0 ∧ t.c > 0) -- Side lengths are positive
  : t.B / t.A = 1 / 2 := by
  sorry


end triangle_angle_ratio_l3286_328624


namespace A_intersect_B_l3286_328600

def A : Set ℕ := {1, 2, 4, 6, 8}

def B : Set ℕ := {x | ∃ k ∈ A, x = 2 * k}

theorem A_intersect_B : A ∩ B = {2, 4, 8} := by sorry

end A_intersect_B_l3286_328600


namespace waiter_customers_l3286_328688

/-- The number of customers a waiter served before the lunch rush -/
def customers_before_rush : ℕ := 29

/-- The number of additional customers during the lunch rush -/
def additional_customers : ℕ := 20

/-- The number of customers who didn't leave a tip -/
def customers_no_tip : ℕ := 34

/-- The number of customers who left a tip -/
def customers_with_tip : ℕ := 15

theorem waiter_customers :
  customers_before_rush + additional_customers =
  customers_no_tip + customers_with_tip :=
by sorry

end waiter_customers_l3286_328688


namespace person_2019_chooses_left_l3286_328670

def chocolate_distribution (L M R : ℕ+) (n : ℕ) : ℕ :=
  let total := L + M + R
  let full_rounds := n / total
  let remainder := n % total
  let left_count := full_rounds * L.val + min remainder L.val
  let middle_count := full_rounds * M.val + min (remainder - left_count) M.val
  let right_count := full_rounds * R.val + (remainder - left_count - middle_count)
  if (L.val : ℚ) / (left_count + 1) ≥ max ((M.val : ℚ) / (middle_count + 1)) ((R.val : ℚ) / (right_count + 1))
  then 0  -- Left table
  else if (M.val : ℚ) / (middle_count + 1) > (R.val : ℚ) / (right_count + 1)
  then 1  -- Middle table
  else 2  -- Right table

theorem person_2019_chooses_left (L M R : ℕ+) (h1 : L = 9) (h2 : M = 19) (h3 : R = 25) :
  chocolate_distribution L M R 2019 = 0 :=
sorry

end person_2019_chooses_left_l3286_328670


namespace largest_value_l3286_328649

theorem largest_value (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  b > 1/2 ∧ b > a^2 + b^2 ∧ b > 2*a*b := by
  sorry

end largest_value_l3286_328649


namespace optimal_price_and_profit_l3286_328654

/-- Represents the profit function for a product with given pricing conditions -/
def profit_function (x : ℝ) : ℝ := -x^2 + 140*x - 4000

/-- Represents the valid range for the selling price -/
def valid_price_range (x : ℝ) : Prop := 50 ≤ x ∧ x ≤ 100

theorem optimal_price_and_profit :
  ∃ (x : ℝ), 
    valid_price_range x ∧ 
    (∀ y, valid_price_range y → profit_function y ≤ profit_function x) ∧
    x = 70 ∧ 
    profit_function x = 900 :=
sorry

end optimal_price_and_profit_l3286_328654


namespace quadratic_roots_sum_l3286_328622

theorem quadratic_roots_sum (m n : ℝ) : 
  m^2 + 2*m - 2022 = 0 → 
  n^2 + 2*n - 2022 = 0 → 
  m^2 + 3*m + n = 2020 :=
by
  sorry

end quadratic_roots_sum_l3286_328622


namespace odd_function_fourth_composition_even_l3286_328662

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Theorem statement
theorem odd_function_fourth_composition_even (f : ℝ → ℝ) (h : OddFunction f) : 
  EvenFunction (fun x ↦ f (f (f (f x)))) :=
sorry

end odd_function_fourth_composition_even_l3286_328662


namespace neg_half_pow_4_mul_6_three_squared_mul_neg_three_cubed_two_cubed_sum_four_times_find_p_in_equation_l3286_328691

-- Operation rule of multiplication of powers with the same base
axiom pow_mul_rule {α : Type*} [Monoid α] (a : α) (m n : ℕ) : a^m * a^n = a^(m+n)

-- Statement 1
theorem neg_half_pow_4_mul_6 : (-1/2 : ℚ)^4 * (-1/2 : ℚ)^6 = (-1/2 : ℚ)^10 := by sorry

-- Statement 2
theorem three_squared_mul_neg_three_cubed : (3 : ℤ)^2 * (-3 : ℤ)^3 = -243 := by sorry

-- Statement 3
theorem two_cubed_sum_four_times : (2 : ℕ)^3 + (2 : ℕ)^3 + (2 : ℕ)^3 + (2 : ℕ)^3 = (2 : ℕ)^5 := by sorry

-- Statement 4
theorem find_p_in_equation (x y : ℝ) :
  ∃ p : ℕ, (x - y)^2 * (x - y)^p * (x - y)^5 = (x - y)^2023 ∧ p = 2016 := by sorry

end neg_half_pow_4_mul_6_three_squared_mul_neg_three_cubed_two_cubed_sum_four_times_find_p_in_equation_l3286_328691


namespace max_radius_of_circle_l3286_328658

-- Define a circle in 2D space
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the theorem
theorem max_radius_of_circle (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ)
  (h1 : C = Circle center radius)
  (h2 : (4, 0) ∈ C)
  (h3 : (-4, 0) ∈ C)
  (h4 : ∃ (x y : ℝ), (x, y) ∈ C) :
  radius ≤ 4 := by
sorry

end max_radius_of_circle_l3286_328658


namespace inequality_system_solution_l3286_328684

-- Define the inequality system
def inequality_system (x k : ℝ) : Prop :=
  (2 * x + 9 > 6 * x + 1) ∧ (x - k < 1)

-- Define the solution set
def solution_set (x : ℝ) : Prop := x < 2

-- Theorem statement
theorem inequality_system_solution (k : ℝ) :
  (∀ x, inequality_system x k ↔ solution_set x) → k ≥ 1 := by
  sorry

end inequality_system_solution_l3286_328684


namespace combine_like_terms_l3286_328632

theorem combine_like_terms (a b : ℝ) : 
  2 * a^3 * b - (1/2) * a^3 * b - a^2 * b + (1/2) * a^2 * b - a * b^2 = 
  (3/2) * a^3 * b - (1/2) * a^2 * b - a * b^2 := by
  sorry

end combine_like_terms_l3286_328632


namespace is_solution_l3286_328689

-- Define the function f(x) = x^2 + x + C
def f (C : ℝ) (x : ℝ) : ℝ := x^2 + x + C

-- State the theorem
theorem is_solution (C : ℝ) : 
  ∀ x : ℝ, deriv (f C) x = 2 * x + 1 := by
  sorry

end is_solution_l3286_328689


namespace weight_loss_duration_l3286_328687

/-- Represents the weight loss pattern over a 5-month cycle -/
structure WeightLossPattern :=
  (month1 : Int)
  (month2 : Int)
  (month3 : Int)
  (month4and5 : Int)

/-- Calculates the time needed to reach the target weight -/
def timeToReachTarget (initialWeight : Int) (pattern : WeightLossPattern) (targetWeight : Int) : Int :=
  sorry

/-- The theorem statement -/
theorem weight_loss_duration :
  let initialWeight := 222
  let pattern := WeightLossPattern.mk (-12) (-6) 2 (-8)
  let targetWeight := 170
  timeToReachTarget initialWeight pattern targetWeight = 6 :=
sorry

end weight_loss_duration_l3286_328687


namespace carriage_equation_correct_l3286_328633

/-- Represents the scenario of people and carriages as described in the ancient Chinese problem --/
def carriage_problem (x : ℕ) : Prop :=
  -- Three people sharing a carriage leaves two carriages empty
  (3 * (x - 2) : ℤ) = (3 * x - 6 : ℤ) ∧
  -- Two people sharing a carriage leaves nine people walking
  (2 * x + 9 : ℤ) = (3 * x - 6 : ℤ)

/-- The equation 3(x-2) = 2x + 9 correctly represents the carriage problem --/
theorem carriage_equation_correct (x : ℕ) :
  carriage_problem x ↔ (3 * (x - 2) : ℤ) = (2 * x + 9 : ℤ) :=
sorry

end carriage_equation_correct_l3286_328633


namespace kendalls_quarters_l3286_328682

/-- Represents the number of coins of each type -/
structure CoinCounts where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ

/-- Calculates the total value of coins in dollars -/
def totalValue (c : CoinCounts) : ℚ :=
  c.quarters * (1/4) + c.dimes * (1/10) + c.nickels * (1/20)

theorem kendalls_quarters :
  ∃ (c : CoinCounts), c.dimes = 12 ∧ c.nickels = 6 ∧ totalValue c = 4 ∧ c.quarters = 10 := by
  sorry

end kendalls_quarters_l3286_328682


namespace geometric_sequence_fourth_term_l3286_328643

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_sum : a 1 + 2 * a 2 = 3)
  (h_prod : a 3 ^ 2 = 4 * a 2 * a 6)
  (h_geo : GeometricSequence a) :
  a 4 = 3 / 16 := by
sorry

end geometric_sequence_fourth_term_l3286_328643


namespace min_value_when_a_is_one_range_of_a_l3286_328648

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + |x - 4| - a

-- Theorem for the minimum value of f when a = 1
theorem min_value_when_a_is_one :
  ∃ (m : ℝ), m = 4 ∧ ∀ (x : ℝ), f 1 x ≥ m :=
sorry

-- Theorem for the range of a
theorem range_of_a :
  ∀ (a : ℝ), (∀ (x : ℝ), f a x ≥ 4/a + 1) ↔ (a < 0 ∨ a = 2) :=
sorry

end min_value_when_a_is_one_range_of_a_l3286_328648


namespace tan_equality_proof_l3286_328608

theorem tan_equality_proof (n : Int) :
  -180 < n ∧ n < 180 → Real.tan (n * π / 180) = Real.tan (210 * π / 180) → n = 30 := by
  sorry

end tan_equality_proof_l3286_328608


namespace perpendicular_vectors_x_value_l3286_328601

/-- Given two vectors a and b in ℝ², where a = (4, 8) and b = (x, 4),
    if a is perpendicular to b, then x = -8. -/
theorem perpendicular_vectors_x_value :
  let a : ℝ × ℝ := (4, 8)
  let b : ℝ × ℝ := (x, 4)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = -8 :=
by sorry

end perpendicular_vectors_x_value_l3286_328601


namespace three_W_four_l3286_328629

-- Define the operation W
def W (a b : ℤ) : ℤ := b + 5*a - 3*a^2

-- Theorem statement
theorem three_W_four : W 3 4 = -8 := by sorry

end three_W_four_l3286_328629


namespace outfits_count_l3286_328645

/-- Number of red shirts -/
def red_shirts : ℕ := 4

/-- Number of green shirts -/
def green_shirts : ℕ := 4

/-- Number of blue shirts -/
def blue_shirts : ℕ := 4

/-- Number of pants -/
def pants : ℕ := 7

/-- Number of green hats -/
def green_hats : ℕ := 6

/-- Number of red hats -/
def red_hats : ℕ := 6

/-- Number of blue hats -/
def blue_hats : ℕ := 6

/-- Calculate the number of outfits with different colored shirts and hats -/
def outfits : ℕ := 
  (red_shirts * pants * (green_hats + blue_hats)) +
  (green_shirts * pants * (red_hats + blue_hats)) +
  (blue_shirts * pants * (red_hats + green_hats))

theorem outfits_count : outfits = 1008 := by
  sorry

end outfits_count_l3286_328645


namespace trapezoid_division_l3286_328619

/-- Represents a trapezoid with the given side lengths -/
structure Trapezoid where
  short_base : ℝ
  long_base : ℝ
  side1 : ℝ
  side2 : ℝ

/-- Represents a point that divides a line segment -/
structure DivisionPoint where
  ratio : ℝ

/-- 
Given a trapezoid with parallel sides of length 3 and 9, and non-parallel sides of length 4 and 6,
if a line parallel to the bases divides the trapezoid into two trapezoids of equal perimeters,
then this line divides each of the non-parallel sides in the ratio 3:2.
-/
theorem trapezoid_division (t : Trapezoid) (d : DivisionPoint) : 
  t.short_base = 3 ∧ t.long_base = 9 ∧ t.side1 = 4 ∧ t.side2 = 6 →
  (t.long_base - t.short_base) * d.ratio + t.short_base = 
    (t.side1 * d.ratio + t.side2 * d.ratio) / 2 →
  d.ratio = 3 / 5 := by
  sorry

end trapezoid_division_l3286_328619


namespace catherine_bottle_caps_l3286_328659

def number_of_friends : ℕ := 6
def bottle_caps_per_friend : ℕ := 3

theorem catherine_bottle_caps : 
  number_of_friends * bottle_caps_per_friend = 18 := by
  sorry

end catherine_bottle_caps_l3286_328659


namespace fraction_equality_implies_x_value_l3286_328642

theorem fraction_equality_implies_x_value :
  ∀ x : ℝ, (4 + 2*x) / (6 + 3*x) = (3 + 2*x) / (5 + 3*x) → x = -2 := by
sorry

end fraction_equality_implies_x_value_l3286_328642


namespace semicircle_pattern_area_l3286_328644

/-- The area of shaded region formed by semicircles in a pattern --/
theorem semicircle_pattern_area (d : ℝ) (l : ℝ) (h1 : d = 4) (h2 : l = 24) : 
  (l / d) / 2 * (π * (d / 2)^2) = 12 * π := by
  sorry

end semicircle_pattern_area_l3286_328644


namespace compounded_growth_rate_l3286_328692

/-- Given an initial investment P that grows by k% in the first year and m% in the second year,
    the compounded rate of growth R after two years is equal to k + m + (km/100). -/
theorem compounded_growth_rate (P k m : ℝ) (hP : P > 0) (hk : k ≥ 0) (hm : m ≥ 0) :
  let R := k + m + (k * m) / 100
  let growth_factor := (1 + k / 100) * (1 + m / 100)
  R = (growth_factor - 1) * 100 :=
by sorry

end compounded_growth_rate_l3286_328692


namespace revenue_change_l3286_328665

/-- Proves that when the price increases by 50% and the quantity sold decreases by 20%, the revenue increases by 20% -/
theorem revenue_change 
  (P Q : ℝ) 
  (P' : ℝ) (hP' : P' = 1.5 * P) 
  (Q' : ℝ) (hQ' : Q' = 0.8 * Q) : 
  P' * Q' = 1.2 * (P * Q) := by
  sorry

#check revenue_change

end revenue_change_l3286_328665


namespace alice_savings_l3286_328646

/-- Alice's savings problem -/
theorem alice_savings (B : ℕ) : 
  let month1 : ℕ := 10
  let month2 : ℕ := month1 + 30 + B
  let month3 : ℕ := month1 + 60
  month1 + month2 + month3 = 120 + B := by
  sorry

end alice_savings_l3286_328646


namespace three_sequence_inequality_l3286_328693

theorem three_sequence_inequality (a b c : ℕ → ℕ) :
  ∃ p q : ℕ, p ≠ q ∧ a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
by sorry

end three_sequence_inequality_l3286_328693


namespace probability_sum_nine_l3286_328638

/-- The number of sides on a standard die -/
def sides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The target sum we're looking for -/
def targetSum : ℕ := 9

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := sides ^ numDice

/-- The number of favorable outcomes (ways to get a sum of 9) -/
def favorableOutcomes : ℕ := 19

/-- The probability of rolling a sum of 9 with three fair, standard six-sided dice -/
theorem probability_sum_nine :
  (favorableOutcomes : ℚ) / totalOutcomes = 19 / 216 := by
  sorry

end probability_sum_nine_l3286_328638


namespace prime_divisor_greater_than_exponent_l3286_328647

theorem prime_divisor_greater_than_exponent (p q : ℕ) : 
  Prime p → Prime q → q > 5 → q ∣ (2^p + 3^p) → q > p := by
  sorry

end prime_divisor_greater_than_exponent_l3286_328647


namespace group_size_proof_l3286_328697

theorem group_size_proof (average_increase : ℝ) (new_weight : ℝ) (old_weight : ℝ) :
  average_increase = 6 →
  new_weight = 88 →
  old_weight = 40 →
  (average_increase * (new_weight - old_weight) / average_increase : ℝ) = 8 := by
  sorry

end group_size_proof_l3286_328697


namespace town_businesses_town_businesses_proof_l3286_328611

theorem town_businesses : ℕ → Prop :=
  fun total_businesses =>
    let fired := total_businesses / 2
    let quit := total_businesses / 3
    let can_apply := 12
    fired + quit + can_apply = total_businesses ∧ total_businesses = 72

-- Proof
theorem town_businesses_proof : ∃ n : ℕ, town_businesses n := by
  sorry

end town_businesses_town_businesses_proof_l3286_328611


namespace largest_integer_satisfying_inequality_l3286_328617

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, x ≤ 2 ↔ 3 * x + 4 > 5 * x - 1 :=
by sorry

end largest_integer_satisfying_inequality_l3286_328617


namespace correct_oranges_put_back_l3286_328674

/-- Represents the fruit selection problem with given prices and quantities -/
structure FruitSelection where
  apple_price : ℚ
  orange_price : ℚ
  total_fruits : ℕ
  initial_avg_price : ℚ
  desired_avg_price : ℚ

/-- Calculates the number of oranges to put back to achieve the desired average price -/
def oranges_to_put_back (fs : FruitSelection) : ℕ :=
  2

/-- Theorem stating that putting back the calculated number of oranges achieves the desired average price -/
theorem correct_oranges_put_back (fs : FruitSelection)
  (h1 : fs.apple_price = 40/100)
  (h2 : fs.orange_price = 60/100)
  (h3 : fs.total_fruits = 10)
  (h4 : fs.initial_avg_price = 48/100)
  (h5 : fs.desired_avg_price = 45/100) :
  let num_oranges_back := oranges_to_put_back fs
  let remaining_fruits := fs.total_fruits - num_oranges_back
  let num_apples := 6  -- Derived from the problem's solution
  let num_oranges := 4 -- Derived from the problem's solution
  fs.apple_price * num_apples + fs.orange_price * (num_oranges - num_oranges_back) =
    fs.desired_avg_price * remaining_fruits :=
by
  sorry

end correct_oranges_put_back_l3286_328674


namespace dimes_count_l3286_328604

/-- Represents the number of coins of each type --/
structure CoinCounts where
  quarters : Nat
  dimes : Nat
  nickels : Nat
  pennies : Nat

/-- Calculates the total value in cents for a given set of coin counts --/
def totalValue (coins : CoinCounts) : Nat :=
  coins.quarters * 25 + coins.dimes * 10 + coins.nickels * 5 + coins.pennies

/-- Theorem: Given the total amount and the number of other coins, the number of dimes is 3 --/
theorem dimes_count (coins : CoinCounts) :
  coins.quarters = 10 ∧ coins.nickels = 3 ∧ coins.pennies = 5 ∧ totalValue coins = 300 →
  coins.dimes = 3 := by
  sorry


end dimes_count_l3286_328604


namespace f_non_monotonic_l3286_328683

/-- A piecewise function f defined on ℝ with a parameter a and a split point t -/
noncomputable def f (a t : ℝ) (x : ℝ) : ℝ :=
  if x ≤ t then (2*a - 1)*x + 3*a - 4 else x^3 - x

/-- The theorem stating the condition for non-monotonicity of f -/
theorem f_non_monotonic (a : ℝ) :
  (∀ t : ℝ, ¬ Monotone (f a t)) ↔ a ≤ (1/2 : ℝ) :=
sorry

end f_non_monotonic_l3286_328683


namespace largest_common_divisor_360_315_l3286_328637

theorem largest_common_divisor_360_315 : Nat.gcd 360 315 = 45 := by
  sorry

end largest_common_divisor_360_315_l3286_328637


namespace sine_three_fourths_pi_minus_alpha_l3286_328666

theorem sine_three_fourths_pi_minus_alpha (α : ℝ) 
  (h : Real.sin (π / 4 + α) = 3 / 5) : 
  Real.sin (3 * π / 4 - α) = 3 / 5 := by
  sorry

end sine_three_fourths_pi_minus_alpha_l3286_328666


namespace total_spent_is_23_88_l3286_328628

def green_grape_price : ℝ := 2.79
def red_grape_price : ℝ := 3.25
def regular_cherry_price : ℝ := 4.90
def organic_cherry_price : ℝ := 5.75

def green_grape_weight : ℝ := 2.5
def red_grape_weight : ℝ := 1.8
def regular_cherry_weight : ℝ := 1.2
def organic_cherry_weight : ℝ := 0.9

def total_spent : ℝ :=
  green_grape_price * green_grape_weight +
  red_grape_price * red_grape_weight +
  regular_cherry_price * regular_cherry_weight +
  organic_cherry_price * organic_cherry_weight

theorem total_spent_is_23_88 : total_spent = 23.88 := by
  sorry

end total_spent_is_23_88_l3286_328628


namespace scale_length_theorem_l3286_328656

/-- Given a scale divided into equal parts, this function calculates its total length -/
def scaleLength (numParts : ℕ) (partLength : ℕ) : ℕ :=
  numParts * partLength

/-- Theorem stating that a scale with 4 parts of 20 inches each has a total length of 80 inches -/
theorem scale_length_theorem :
  scaleLength 4 20 = 80 := by
  sorry

end scale_length_theorem_l3286_328656


namespace a_four_plus_b_four_l3286_328612

theorem a_four_plus_b_four (a b : ℝ) (h1 : a^2 - b^2 = 10) (h2 : a * b = 8) : a^4 + b^4 = 548 := by
  sorry

end a_four_plus_b_four_l3286_328612


namespace strawberries_eaten_l3286_328680

-- Define the initial number of strawberries
def initial_strawberries : ℕ := 35

-- Define the remaining number of strawberries
def remaining_strawberries : ℕ := 33

-- Theorem to prove
theorem strawberries_eaten : initial_strawberries - remaining_strawberries = 2 := by
  sorry

end strawberries_eaten_l3286_328680


namespace arithmetic_progression_rth_term_l3286_328671

/-- Given an arithmetic progression where the sum of n terms is 2n^2 + 3n for every n,
    this function represents the r-th term of the progression. -/
def arithmeticProgressionTerm (r : ℕ) : ℕ := 4 * r + 1

/-- The sum of the first n terms of the arithmetic progression. -/
def arithmeticProgressionSum (n : ℕ) : ℕ := 2 * n^2 + 3 * n

/-- Theorem stating that the r-th term of the arithmetic progression is 4r + 1,
    given that the sum of n terms is 2n^2 + 3n for every n. -/
theorem arithmetic_progression_rth_term (r : ℕ) :
  arithmeticProgressionTerm r = arithmeticProgressionSum r - arithmeticProgressionSum (r - 1) :=
sorry

end arithmetic_progression_rth_term_l3286_328671


namespace monochromatic_rectangle_exists_l3286_328627

/-- A color type representing red, white, and blue -/
inductive Color
  | Red
  | White
  | Blue

/-- A point in the grid -/
structure Point where
  x : Nat
  y : Nat

/-- A coloring function that assigns a color to each point in the grid -/
def Coloring := Point → Color

/-- A rectangle in the grid -/
structure Rectangle where
  topLeft : Point
  bottomRight : Point

/-- Theorem stating the existence of a monochromatic rectangle in a 12x12 grid -/
theorem monochromatic_rectangle_exists (coloring : Coloring) :
  ∃ (rect : Rectangle) (c : Color),
    rect.topLeft.x ≤ 12 ∧ rect.topLeft.y ≤ 12 ∧
    rect.bottomRight.x ≤ 12 ∧ rect.bottomRight.y ≤ 12 ∧
    coloring rect.topLeft = c ∧
    coloring { x := rect.topLeft.x, y := rect.bottomRight.y } = c ∧
    coloring { x := rect.bottomRight.x, y := rect.topLeft.y } = c ∧
    coloring rect.bottomRight = c := by
  sorry

end monochromatic_rectangle_exists_l3286_328627


namespace intersection_point_k_value_l3286_328664

/-- Given two lines that intersect at x = -10, prove the value of k -/
theorem intersection_point_k_value :
  let line1 : ℝ → ℝ → ℝ := λ x y => -3 * x + y
  let line2 : ℝ → ℝ → ℝ := λ x y => 0.75 * x + y
  let k : ℝ := line1 (-10) (line2 (-10) 20)
  k = 57.5 := by
  sorry

end intersection_point_k_value_l3286_328664


namespace smallest_sum_of_five_consecutive_primes_l3286_328675

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- A function that returns true if five consecutive primes starting from the nth prime sum to a multiple of 3, false otherwise -/
def sumDivisibleByThree (n : ℕ) : Prop :=
  (nthPrime n + nthPrime (n+1) + nthPrime (n+2) + nthPrime (n+3) + nthPrime (n+4)) % 3 = 0

/-- The index of the first prime in the sequence of five consecutive primes that sum to 39 -/
def firstPrimeIndex : ℕ := sorry

theorem smallest_sum_of_five_consecutive_primes :
  (∀ k < firstPrimeIndex, ¬sumDivisibleByThree k) ∧
  sumDivisibleByThree firstPrimeIndex ∧
  nthPrime firstPrimeIndex + nthPrime (firstPrimeIndex+1) + nthPrime (firstPrimeIndex+2) +
  nthPrime (firstPrimeIndex+3) + nthPrime (firstPrimeIndex+4) = 39 := by sorry

end smallest_sum_of_five_consecutive_primes_l3286_328675


namespace stadium_length_yards_l3286_328636

-- Define the length of the stadium in feet
def stadium_length_feet : ℕ := 186

-- Define the number of feet in a yard
def feet_per_yard : ℕ := 3

-- Theorem to prove the length of the stadium in yards
theorem stadium_length_yards : 
  stadium_length_feet / feet_per_yard = 62 := by
  sorry

end stadium_length_yards_l3286_328636


namespace custom_op_result_l3286_328694

-- Define the custom operation
def customOp (a b : ℚ) : ℚ := (a^2 + b^2) / (a - b)

-- State the theorem
theorem custom_op_result : customOp (customOp 7 5) 4 = 42 + 1/33 := by
  sorry

end custom_op_result_l3286_328694


namespace phase_shift_sine_function_l3286_328609

/-- The phase shift of the function y = 4 sin(3x - π/4) is π/12 to the right -/
theorem phase_shift_sine_function :
  let f : ℝ → ℝ := λ x => 4 * Real.sin (3 * x - π / 4)
  ∃ (shift : ℝ), shift = π / 12 ∧ 
    ∀ x, f (x + shift) = 4 * Real.sin (3 * x) :=
by sorry

end phase_shift_sine_function_l3286_328609


namespace movie_concessions_cost_l3286_328672

/-- Calculates the amount spent on concessions given the total cost of a movie trip and ticket prices. -/
theorem movie_concessions_cost 
  (total_cost : ℝ) 
  (num_adults : ℕ) 
  (num_children : ℕ) 
  (adult_ticket_price : ℝ) 
  (child_ticket_price : ℝ) 
  (h1 : total_cost = 76) 
  (h2 : num_adults = 5) 
  (h3 : num_children = 2) 
  (h4 : adult_ticket_price = 10) 
  (h5 : child_ticket_price = 7) : 
  total_cost - (num_adults * adult_ticket_price + num_children * child_ticket_price) = 12 := by
sorry


end movie_concessions_cost_l3286_328672


namespace triangle_angle_from_sides_l3286_328681

theorem triangle_angle_from_sides : 
  ∀ (a b c : ℝ), 
    a = 1 → 
    b = Real.sqrt 7 → 
    c = Real.sqrt 3 → 
    ∃ (A B C : ℝ), 
      A + B + C = π ∧ 
      0 < A ∧ A < π ∧ 
      0 < B ∧ B < π ∧ 
      0 < C ∧ C < π ∧ 
      b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) ∧ 
      B = 5*π/6 :=
by sorry

end triangle_angle_from_sides_l3286_328681


namespace extremum_values_l3286_328699

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- Theorem stating that if f(x) has an extremum of 10 at x = 1, then a = 4 and b = -11 -/
theorem extremum_values (a b : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b 1 ≥ f a b x) ∧ 
  (f a b 1 = 10) →
  a = 4 ∧ b = -11 := by
sorry

end extremum_values_l3286_328699


namespace vertical_asymptote_at_three_halves_l3286_328686

-- Define the rational function
def f (x : ℚ) : ℚ := (2 * x + 3) / (6 * x - 9)

-- Theorem statement
theorem vertical_asymptote_at_three_halves :
  ∃ (ε : ℚ), ∀ (δ : ℚ), δ > 0 → ε > 0 → 
    ∀ (x : ℚ), 0 < |x - (3/2)| ∧ |x - (3/2)| < δ → |f x| > ε :=
sorry

end vertical_asymptote_at_three_halves_l3286_328686


namespace quadratic_roots_sum_bound_l3286_328660

theorem quadratic_roots_sum_bound (p : ℝ) (r₁ r₂ : ℝ) : 
  (r₁ ≠ r₂) →
  (r₁^2 + p*r₁ + 7 = 0) →
  (r₂^2 + p*r₂ + 7 = 0) →
  |r₁ + r₂| > 2 * Real.sqrt 7 :=
by sorry

end quadratic_roots_sum_bound_l3286_328660


namespace triangle_properties_l3286_328610

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.sin t.A + Real.sqrt 3 * Real.cos t.A = 0)
  (h2 : t.a = 2 * Real.sqrt 7)
  (h3 : t.b = 2) :
  t.A = 2 * Real.pi / 3 ∧ 
  t.c = 4 ∧ 
  1/2 * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 3 := by
sorry

end triangle_properties_l3286_328610


namespace simplify_expression_find_k_l3286_328621

-- Problem 1
theorem simplify_expression (x : ℝ) :
  (2*x + 1)^2 - (2*x + 1)*(2*x - 1) + (x + 1)*(x - 3) = x^2 + 2*x - 1 := by
  sorry

-- Problem 2
theorem find_k (x y k : ℝ) 
  (eq1 : x + y = 1)
  (eq2 : k*x + (k - 1)*y = 7)
  (eq3 : 3*x - 2*y = 5) :
  k = 33/5 := by
  sorry

end simplify_expression_find_k_l3286_328621


namespace table_tennis_match_probability_l3286_328679

/-- The probability of Player A winning a single game -/
def p_A : ℝ := 0.6

/-- The probability of Player B winning a single game -/
def p_B : ℝ := 0.4

/-- The probability of Player A winning the match in a best-of-three format -/
def p_A_wins_match : ℝ := p_A * p_A + p_A * p_B * p_A + p_B * p_A * p_A

theorem table_tennis_match_probability :
  p_A + p_B = 1 →
  p_A_wins_match = 0.648 := by
  sorry

end table_tennis_match_probability_l3286_328679


namespace hari_join_time_l3286_328661

/-- Represents the number of months in a year -/
def monthsInYear : ℕ := 12

/-- Praveen's initial investment in rupees -/
def praveenInvestment : ℚ := 3500

/-- Hari's investment in rupees -/
def hariInvestment : ℚ := 9000.000000000002

/-- Profit sharing ratio for Praveen -/
def praveenShare : ℚ := 2

/-- Profit sharing ratio for Hari -/
def hariShare : ℚ := 3

/-- Theorem stating when Hari joined the business -/
theorem hari_join_time : 
  ∃ (x : ℕ), x < monthsInYear ∧ 
  (praveenInvestment * monthsInYear) / (hariInvestment * (monthsInYear - x)) = praveenShare / hariShare ∧
  x = 5 := by sorry

end hari_join_time_l3286_328661


namespace opposite_roots_iff_ab_eq_c_l3286_328673

-- Define the cubic polynomial f(x) = x^3 + a x^2 + b x + c
def f (a b c x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

-- Define a predicate for when two roots are opposite numbers
def has_opposite_roots (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, f a b c x = 0 ∧ f a b c y = 0 ∧ y = -x

-- State the theorem
theorem opposite_roots_iff_ab_eq_c (a b c : ℝ) (h : b ≤ 0) :
  has_opposite_roots a b c ↔ a * b = c :=
sorry

end opposite_roots_iff_ab_eq_c_l3286_328673


namespace ellipse_minor_axis_length_l3286_328603

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with axes parallel to coordinate axes -/
structure Ellipse where
  center : Point
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Check if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  ((p.x - e.center.x)^2 / e.semi_major_axis^2) + ((p.y - e.center.y)^2 / e.semi_minor_axis^2) = 1

theorem ellipse_minor_axis_length : 
  ∀ (e : Ellipse),
    let p1 : Point := ⟨-2, 1⟩
    let p2 : Point := ⟨0, 0⟩
    let p3 : Point := ⟨0, 3⟩
    let p4 : Point := ⟨4, 0⟩
    let p5 : Point := ⟨4, 3⟩
    (¬ collinear p1 p2 p3 ∧ ¬ collinear p1 p2 p4 ∧ ¬ collinear p1 p2 p5 ∧ 
     ¬ collinear p1 p3 p4 ∧ ¬ collinear p1 p3 p5 ∧ ¬ collinear p1 p4 p5 ∧ 
     ¬ collinear p2 p3 p4 ∧ ¬ collinear p2 p3 p5 ∧ ¬ collinear p2 p4 p5 ∧ 
     ¬ collinear p3 p4 p5) →
    (pointOnEllipse p1 e ∧ pointOnEllipse p2 e ∧ pointOnEllipse p3 e ∧ 
     pointOnEllipse p4 e ∧ pointOnEllipse p5 e) →
    2 * e.semi_minor_axis = 2 * Real.sqrt 3 := by
  sorry

end ellipse_minor_axis_length_l3286_328603


namespace race_outcome_l3286_328613

/-- Represents the distance traveled by an animal at a given time --/
structure DistanceTime where
  distance : ℝ
  time : ℝ

/-- Represents the race between a tortoise and a hare --/
structure Race where
  tortoise : List DistanceTime
  hare : List DistanceTime

/-- Checks if a list of DistanceTime points represents a steady pace --/
def isSteadyPace (points : List DistanceTime) : Prop := sorry

/-- Checks if a list of DistanceTime points has exactly two stops --/
def hasTwoStops (points : List DistanceTime) : Prop := sorry

/-- Checks if the first point in a list finishes before the first point in another list --/
def finishesFirst (winner loser : List DistanceTime) : Prop := sorry

/-- Theorem representing the race conditions and outcome --/
theorem race_outcome (race : Race) : 
  isSteadyPace race.tortoise ∧ 
  hasTwoStops race.hare ∧ 
  finishesFirst race.tortoise race.hare := by
  sorry

#check race_outcome

end race_outcome_l3286_328613


namespace book_purchase_equation_l3286_328634

/-- Represents a book purchase scenario with two purchases -/
structure BookPurchase where
  first_cost : ℝ
  second_cost : ℝ
  quantity_difference : ℕ
  first_quantity : ℝ

/-- The equation correctly represents the book purchase scenario -/
def correct_equation (bp : BookPurchase) : Prop :=
  bp.first_cost / bp.first_quantity = bp.second_cost / (bp.first_quantity + bp.quantity_difference)

/-- Theorem stating that the given equation correctly represents the book purchase scenario -/
theorem book_purchase_equation (bp : BookPurchase) 
  (h1 : bp.first_cost = 7000)
  (h2 : bp.second_cost = 9000)
  (h3 : bp.quantity_difference = 60)
  (h4 : bp.first_quantity > 0) :
  correct_equation bp := by
  sorry

end book_purchase_equation_l3286_328634


namespace unique_number_with_three_prime_divisors_l3286_328663

theorem unique_number_with_three_prime_divisors (x n : ℕ) : 
  x = 9^n - 1 →
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧
    (∀ d : ℕ, d ∣ x → d = 1 ∨ d = p ∨ d = q ∨ d = 11 ∨ d = p*q ∨ d = p*11 ∨ d = q*11 ∨ d = p*q*11)) →
  11 ∣ x →
  x = 59048 :=
by sorry

end unique_number_with_three_prime_divisors_l3286_328663


namespace smallest_n_with_constant_term_l3286_328657

theorem smallest_n_with_constant_term : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → k < n → 
    ¬ ∃ (r : ℕ), r ≤ k ∧ 3 * k = (7 * r) / 2) ∧
  (∃ (r : ℕ), r ≤ n ∧ 3 * n = (7 * r) / 2) ∧
  n = 7 :=
sorry

end smallest_n_with_constant_term_l3286_328657


namespace quadratic_two_distinct_roots_l3286_328631

/-- The quadratic equation x^2 - 2x - 6 = 0 has two distinct real roots -/
theorem quadratic_two_distinct_roots : ∃ (x₁ x₂ : ℝ), 
  x₁ ≠ x₂ ∧ 
  x₁^2 - 2*x₁ - 6 = 0 ∧ 
  x₂^2 - 2*x₂ - 6 = 0 :=
by
  sorry

end quadratic_two_distinct_roots_l3286_328631


namespace floor_equality_condition_l3286_328626

theorem floor_equality_condition (a b : ℝ) :
  (∀ n : ℕ+, a * ⌊b * n⌋ = b * ⌊a * n⌋) ↔ (a = b ∨ a = 0 ∨ b = 0) := by
  sorry

end floor_equality_condition_l3286_328626


namespace yellow_parrots_count_l3286_328676

theorem yellow_parrots_count (total : ℕ) (red_fraction : ℚ) : 
  total = 108 → red_fraction = 5/6 → (1 - red_fraction) * total = 18 := by
  sorry

end yellow_parrots_count_l3286_328676


namespace arcsin_one_half_equals_pi_over_six_l3286_328602

theorem arcsin_one_half_equals_pi_over_six : 
  Real.arcsin (1/2) = π/6 := by
  sorry

end arcsin_one_half_equals_pi_over_six_l3286_328602


namespace polynomial_sum_equality_l3286_328667

theorem polynomial_sum_equality : 
  let p (x : ℝ) := 4 * x^2 - 2 * x + 1
  let q (x : ℝ) := -3 * x^2 + x - 5
  let r (x : ℝ) := 2 * x^2 - 4 * x + 3
  ∀ x, p x + q x + r x = 3 * x^2 - 5 * x - 1 := by
    sorry

end polynomial_sum_equality_l3286_328667


namespace fourth_root_simplification_l3286_328615

theorem fourth_root_simplification (x : ℝ) (hx : x > 0) :
  (x^3 * (x^5)^(1/2))^(1/4) = x^(11/8) := by
  sorry

end fourth_root_simplification_l3286_328615


namespace streetlight_shadow_indeterminate_l3286_328653

-- Define persons A and B
def Person : Type := String

-- Define shadow length under sunlight
def sunShadowLength (p : Person) : ℝ := sorry

-- Define shadow length under streetlight
def streetShadowLength (p : Person) (distance : ℝ) : ℝ := sorry

-- Define the problem conditions
axiom longer_sun_shadow (A B : Person) : sunShadowLength A > sunShadowLength B

-- Theorem stating that the relative shadow lengths under streetlight cannot be determined
theorem streetlight_shadow_indeterminate (A B : Person) :
  ∃ (d1 d2 : ℝ), 
    (streetShadowLength A d1 > streetShadowLength B d2) ∧
    (∃ (d3 d4 : ℝ), streetShadowLength A d3 < streetShadowLength B d4) ∧
    (∃ (d5 d6 : ℝ), streetShadowLength A d5 = streetShadowLength B d6) :=
sorry

end streetlight_shadow_indeterminate_l3286_328653


namespace inequality_proof_l3286_328620

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end inequality_proof_l3286_328620


namespace ben_hours_per_shift_l3286_328618

/-- Represents the time it takes Ben to build one rocking chair -/
def time_per_chair : ℕ := 5

/-- Represents the number of chairs Ben builds in 10 days -/
def chairs_in_ten_days : ℕ := 16

/-- Represents the number of days Ben works -/
def work_days : ℕ := 10

/-- Represents the number of shifts Ben works per day -/
def shifts_per_day : ℕ := 1

/-- Theorem stating that Ben works 8 hours per shift -/
theorem ben_hours_per_shift : 
  (chairs_in_ten_days * time_per_chair) / work_days = 8 := by
  sorry

end ben_hours_per_shift_l3286_328618


namespace M_congruent_to_1_mod_47_l3286_328696

def M : ℕ := sorry -- Definition of M as the 81-digit number

theorem M_congruent_to_1_mod_47 :
  M % 47 = 1 := by sorry

end M_congruent_to_1_mod_47_l3286_328696


namespace grant_total_sales_l3286_328669

def baseball_cards_price : ℝ := 25
def baseball_bat_price : ℝ := 10
def baseball_glove_original_price : ℝ := 30
def baseball_glove_discount : ℝ := 0.2
def baseball_cleats_price : ℝ := 10
def baseball_cleats_count : ℕ := 2

def total_sales : ℝ :=
  baseball_cards_price +
  baseball_bat_price +
  (baseball_glove_original_price * (1 - baseball_glove_discount)) +
  (baseball_cleats_price * baseball_cleats_count)

theorem grant_total_sales :
  total_sales = 79 := by sorry

end grant_total_sales_l3286_328669


namespace part1_part2_l3286_328678

-- Define propositions p, q, and r as functions of a
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + (a - 1) * x + a^2 > 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (2 * a^2 - a)^x < (2 * a^2 - a)^y

def r (a : ℝ) : Prop := (2 * a - 1) / (a - 2) ≤ 1

-- Define the range of a for part 1
def range_a (a : ℝ) : Prop := (a ≥ -1 ∧ a < -1/2) ∨ (a > 1/3 ∧ a ≤ 1)

-- Theorem for part 1
theorem part1 (a : ℝ) : (p a ∨ q a) ∧ ¬(p a ∧ q a) → range_a a := by sorry

-- Theorem for part 2
theorem part2 : (∀ a : ℝ, ¬(p a) → r a) ∧ ¬(∀ a : ℝ, r a → ¬(p a)) := by sorry

end part1_part2_l3286_328678


namespace not_cube_of_integer_l3286_328652

theorem not_cube_of_integer : ¬ ∃ (k : ℤ), 10^202 + 5 * 10^101 + 1 = k^3 := by sorry

end not_cube_of_integer_l3286_328652


namespace problem_solution_l3286_328640

theorem problem_solution (x : ℝ) : (20 / 100 * 30 = 25 / 100 * x + 2) → x = 16 := by
  sorry

end problem_solution_l3286_328640


namespace binary_polynomial_form_l3286_328614

/-- A binary homogeneous polynomial of degree n -/
def BinaryHomogeneousPolynomial (n : ℕ) := ℝ → ℝ → ℝ

/-- The polynomial condition for all real numbers a, b, c -/
def SatisfiesCondition (P : BinaryHomogeneousPolynomial n) : Prop :=
  ∀ a b c : ℝ, P (a + b) c + P (b + c) a + P (c + a) b = 0

/-- The theorem stating the form of the polynomial P -/
theorem binary_polynomial_form (n : ℕ) (P : BinaryHomogeneousPolynomial n)
  (h1 : SatisfiesCondition P) (h2 : P 1 0 = 1) :
  ∃ f : ℝ → ℝ → ℝ, (∀ x y : ℝ, P x y = f x y * (x - 2*y)) ∧
                    (∀ x y : ℝ, f x y = (x + y)^(n-1)) :=
sorry

end binary_polynomial_form_l3286_328614


namespace area_of_special_parallelogram_l3286_328639

/-- Represents a parallelogram with base and altitude. -/
structure Parallelogram where
  base : ℝ
  altitude : ℝ

/-- The area of a parallelogram. -/
def area (p : Parallelogram) : ℝ := p.base * p.altitude

/-- A parallelogram with altitude twice the base and base length 12. -/
def special_parallelogram : Parallelogram where
  base := 12
  altitude := 2 * 12

theorem area_of_special_parallelogram :
  area special_parallelogram = 288 := by
  sorry

end area_of_special_parallelogram_l3286_328639


namespace krishan_money_l3286_328605

/-- Represents the money ratios and changes for Ram, Gopal, Shyam, and Krishan --/
structure MoneyProblem where
  ram_initial : ℚ
  gopal_initial : ℚ
  shyam_initial : ℚ
  krishan_ratio : ℚ
  ram_increase_percent : ℚ
  shyam_decrease_percent : ℚ
  ram_final : ℚ
  shyam_final : ℚ

/-- Theorem stating that given the conditions, Krishan's money is 3400 --/
theorem krishan_money (p : MoneyProblem)
  (h1 : p.ram_initial = 7)
  (h2 : p.gopal_initial = 17)
  (h3 : p.shyam_initial = 10)
  (h4 : p.krishan_ratio = 16)
  (h5 : p.ram_increase_percent = 18.5)
  (h6 : p.shyam_decrease_percent = 20)
  (h7 : p.ram_final = 699.8)
  (h8 : p.shyam_final = 800)
  (h9 : p.gopal_initial / p.ram_initial = 8 / p.krishan_ratio)
  (h10 : p.gopal_initial / p.shyam_initial = 8 / 9) :
  ∃ (x : ℚ), x * p.krishan_ratio = 3400 := by
  sorry


end krishan_money_l3286_328605
