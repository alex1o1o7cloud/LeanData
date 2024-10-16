import Mathlib

namespace NUMINAMATH_CALUDE_value_of_a_satisfying_equation_l908_90840

-- Define the function F
def F (a b c : ℝ) : ℝ := a * b^3 + c

-- State the theorem
theorem value_of_a_satisfying_equation :
  ∃ a : ℝ, F a 3 8 = F a 5 12 ∧ a = -2/49 := by sorry

end NUMINAMATH_CALUDE_value_of_a_satisfying_equation_l908_90840


namespace NUMINAMATH_CALUDE_milk_selling_price_l908_90879

/-- Calculates the selling price of a milk-water mixture given the initial milk price, water percentage, and desired gain percentage. -/
def calculate_selling_price (milk_price : ℚ) (water_percentage : ℚ) (gain_percentage : ℚ) : ℚ :=
  let total_volume : ℚ := 1 + water_percentage
  let cost_price : ℚ := milk_price
  let selling_price : ℚ := cost_price * (1 + gain_percentage)
  selling_price / total_volume

/-- Proves that the selling price of the milk-water mixture is 15 rs per liter under the given conditions. -/
theorem milk_selling_price :
  calculate_selling_price 12 (20/100) (50/100) = 15 := by
  sorry

end NUMINAMATH_CALUDE_milk_selling_price_l908_90879


namespace NUMINAMATH_CALUDE_fixed_points_subset_stable_points_l908_90807

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the set of fixed points
def FixedPoints (f : RealFunction) : Set ℝ :=
  {x : ℝ | f x = x}

-- Define the set of stable points
def StablePoints (f : RealFunction) : Set ℝ :=
  {x : ℝ | f (f x) = x}

-- Theorem statement
theorem fixed_points_subset_stable_points (f : RealFunction) :
  FixedPoints f ⊆ StablePoints f := by
  sorry


end NUMINAMATH_CALUDE_fixed_points_subset_stable_points_l908_90807


namespace NUMINAMATH_CALUDE_school_weeks_l908_90863

/-- Proves that the number of school weeks is 36 given the conditions --/
theorem school_weeks (sandwiches_per_week : ℕ) (missed_days : ℕ) (total_sandwiches : ℕ) 
  (h1 : sandwiches_per_week = 2)
  (h2 : missed_days = 3)
  (h3 : total_sandwiches = 69) : 
  (total_sandwiches + missed_days) / sandwiches_per_week = 36 := by
  sorry

#check school_weeks

end NUMINAMATH_CALUDE_school_weeks_l908_90863


namespace NUMINAMATH_CALUDE_complex_power_4_l908_90808

theorem complex_power_4 : 
  (3 * (Complex.cos (30 * π / 180) + Complex.I * Complex.sin (30 * π / 180)))^4 = 
  Complex.ofReal (-40.5) + Complex.I * Complex.ofReal (40.5 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_complex_power_4_l908_90808


namespace NUMINAMATH_CALUDE_angle_pairs_same_terminal_side_l908_90897

def same_terminal_side (a b : Int) : Prop :=
  ∃ k : Int, b - a = k * 360

theorem angle_pairs_same_terminal_side :
  ¬ same_terminal_side 390 690 ∧
  same_terminal_side (-330) 750 ∧
  ¬ same_terminal_side 480 (-420) ∧
  ¬ same_terminal_side 3000 (-840) :=
by sorry

end NUMINAMATH_CALUDE_angle_pairs_same_terminal_side_l908_90897


namespace NUMINAMATH_CALUDE_fruit_arrangement_unique_l908_90838

-- Define the fruits
inductive Fruit
| Apple
| Pear
| Orange
| Banana

-- Define a type for box numbers
inductive BoxNumber
| One
| Two
| Three
| Four

-- Define a function type for fruit arrangements
def Arrangement := BoxNumber → Fruit

-- Define a predicate for the correctness of labels
def LabelIncorrect (arr : Arrangement) : Prop :=
  arr BoxNumber.One ≠ Fruit.Orange ∧
  arr BoxNumber.Two ≠ Fruit.Pear ∧
  (arr BoxNumber.One = Fruit.Banana → arr BoxNumber.Three ≠ Fruit.Apple ∧ arr BoxNumber.Three ≠ Fruit.Pear) ∧
  arr BoxNumber.Four ≠ Fruit.Apple

-- Define the correct arrangement
def CorrectArrangement : Arrangement :=
  fun b => match b with
  | BoxNumber.One => Fruit.Banana
  | BoxNumber.Two => Fruit.Apple
  | BoxNumber.Three => Fruit.Orange
  | BoxNumber.Four => Fruit.Pear

-- Theorem statement
theorem fruit_arrangement_unique :
  ∀ (arr : Arrangement),
    (∀ (b : BoxNumber), ∃! (f : Fruit), arr b = f) →
    LabelIncorrect arr →
    arr = CorrectArrangement :=
sorry

end NUMINAMATH_CALUDE_fruit_arrangement_unique_l908_90838


namespace NUMINAMATH_CALUDE_alice_winning_strategy_l908_90843

theorem alice_winning_strategy (x : ℕ) (h : x ≤ 2020) :
  ∃ k : ℤ, (2021 - x)^2 - x^2 = 2021 * k := by
  sorry

end NUMINAMATH_CALUDE_alice_winning_strategy_l908_90843


namespace NUMINAMATH_CALUDE_quadratic_root_implies_coefficients_l908_90805

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def is_root (x a b : ℂ) : Prop := x^2 + a*x + b = 0

-- State the theorem
theorem quadratic_root_implies_coefficients :
  ∀ (a b : ℝ), is_root (1 - i) a b → a = -2 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_coefficients_l908_90805


namespace NUMINAMATH_CALUDE_stacked_squares_area_l908_90861

/-- Represents a square sheet of paper -/
structure Square where
  side_length : ℝ

/-- Represents the configuration of four stacked squares -/
structure StackedSquares where
  base : Square
  rotated45 : Square
  middle : Square
  rotated90 : Square

/-- The area of the polygon formed by the stacked squares -/
def polygon_area (s : StackedSquares) : ℝ := sorry

theorem stacked_squares_area :
  ∀ (s : StackedSquares),
    s.base.side_length = 8 ∧
    s.rotated45.side_length = 8 ∧
    s.middle.side_length = 8 ∧
    s.rotated90.side_length = 8 →
    polygon_area s = 192 - 128 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_stacked_squares_area_l908_90861


namespace NUMINAMATH_CALUDE_jonas_current_socks_l908_90848

-- Define the wardrobe items
def shoes : ℕ := 5
def pants : ℕ := 10
def tshirts : ℕ := 10
def socks_to_buy : ℕ := 35

-- Define the function to calculate individual items
def individual_items (socks : ℕ) : ℕ :=
  2 * shoes + 2 * pants + tshirts + 2 * socks

-- Theorem to prove
theorem jonas_current_socks :
  ∃ current_socks : ℕ,
    individual_items (current_socks + socks_to_buy) = 2 * individual_items current_socks ∧
    current_socks = 15 := by
  sorry


end NUMINAMATH_CALUDE_jonas_current_socks_l908_90848


namespace NUMINAMATH_CALUDE_factorial_300_trailing_zeros_l908_90810

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 300! has 74 trailing zeros -/
theorem factorial_300_trailing_zeros :
  trailingZeros 300 = 74 := by
  sorry

end NUMINAMATH_CALUDE_factorial_300_trailing_zeros_l908_90810


namespace NUMINAMATH_CALUDE_largest_possible_value_l908_90841

theorem largest_possible_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x * y + y * z + z * x = 1) :
  let M := 3 / (Real.sqrt 3 + 1)
  (x / (1 + y * z / x)) + (y / (1 + z * x / y)) + (z / (1 + x * y / z)) ≥ M ∧ 
  ∀ N > M, ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b + b * c + c * a = 1 ∧
    (a / (1 + b * c / a)) + (b / (1 + c * a / b)) + (c / (1 + a * b / c)) < N :=
by sorry

end NUMINAMATH_CALUDE_largest_possible_value_l908_90841


namespace NUMINAMATH_CALUDE_line_perp_condition_l908_90851

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perpPlane : Line → Plane → Prop)
variable (para : Line → Plane → Prop)

-- State the theorem
theorem line_perp_condition 
  (a b : Line) (α : Plane) :
  perpPlane a α → para b α → perp a b :=
sorry

end NUMINAMATH_CALUDE_line_perp_condition_l908_90851


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l908_90827

theorem rectangle_area (square_area : ℝ) (rectangle_length : ℝ) : ℝ :=
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_breath := (3 / 5) * circle_radius
  rectangle_length * rectangle_breath

theorem rectangle_area_proof :
  rectangle_area 2025 10 = 270 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l908_90827


namespace NUMINAMATH_CALUDE_chairs_remaining_l908_90839

theorem chairs_remaining (initial_chairs : ℕ) (difference : ℕ) (remaining_chairs : ℕ) : 
  initial_chairs = 15 → 
  difference = 12 → 
  initial_chairs - remaining_chairs = difference →
  remaining_chairs = 3 := by
  sorry

end NUMINAMATH_CALUDE_chairs_remaining_l908_90839


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_6_proof_l908_90826

/-- The greatest four-digit number divisible by 6 -/
def greatest_four_digit_divisible_by_6 : ℕ := 9996

/-- A number is a four-digit number if it's between 1000 and 9999 inclusive -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem greatest_four_digit_divisible_by_6_proof :
  (is_four_digit greatest_four_digit_divisible_by_6) ∧ 
  (greatest_four_digit_divisible_by_6 % 6 = 0) ∧
  (∀ n : ℕ, is_four_digit n → n % 6 = 0 → n ≤ greatest_four_digit_divisible_by_6) :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_6_proof_l908_90826


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l908_90801

/-- A quadratic function symmetric about x = 1 and passing through the origin -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

/-- The function is symmetric about x = 1 -/
def symmetric_about_one (a b : ℝ) : Prop := ∀ x, f a b (1 + x) = f a b (1 - x)

/-- The function passes through the origin -/
def passes_through_origin (a b : ℝ) : Prop := f a b 0 = 0

theorem quadratic_function_properties (a b : ℝ) 
  (h1 : symmetric_about_one a b) (h2 : passes_through_origin a b) :
  (∀ x, f a b x = x^2 - 2*x) ∧ 
  Set.Icc (-1) 3 = Set.range (fun x => f a b x) ∩ Set.Ioo 0 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l908_90801


namespace NUMINAMATH_CALUDE_largest_equilateral_triangle_l908_90893

/-- Represents a square piece of paper -/
structure Square where
  side : ℝ
  side_positive : side > 0

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

/-- The folding process that creates the largest equilateral triangle from a square -/
noncomputable def foldLargestTriangle (s : Square) : EquilateralTriangle :=
  sorry

/-- Theorem stating that the triangle produced by foldLargestTriangle is the largest possible -/
theorem largest_equilateral_triangle (s : Square) :
  ∀ t : EquilateralTriangle, t.side ≤ (foldLargestTriangle s).side :=
  sorry

end NUMINAMATH_CALUDE_largest_equilateral_triangle_l908_90893


namespace NUMINAMATH_CALUDE_total_students_count_l908_90857

/-- The number of students per team -/
def students_per_team : ℕ := 18

/-- The number of teams -/
def number_of_teams : ℕ := 9

/-- The total number of students -/
def total_students : ℕ := students_per_team * number_of_teams

theorem total_students_count : total_students = 162 := by
  sorry

end NUMINAMATH_CALUDE_total_students_count_l908_90857


namespace NUMINAMATH_CALUDE_local_minimum_implies_b_range_l908_90821

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 3*b*x + 3*b

-- State the theorem
theorem local_minimum_implies_b_range :
  ∀ b : ℝ, (∃ c ∈ Set.Ioo 0 1, IsLocalMin (f b) c) → 0 < b ∧ b < 1 := by
  sorry

end NUMINAMATH_CALUDE_local_minimum_implies_b_range_l908_90821


namespace NUMINAMATH_CALUDE_original_average_marks_l908_90833

/-- Given 12 students in a class, if doubling each student's marks results in
    a new average of 100, then the original average of marks is 50. -/
theorem original_average_marks (n : ℕ) (original_avg new_avg : ℝ) : 
  n = 12 → 
  new_avg = 100 → 
  new_avg = 2 * original_avg → 
  original_avg = 50 := by
  sorry

end NUMINAMATH_CALUDE_original_average_marks_l908_90833


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l908_90872

theorem regular_polygon_sides (D : ℕ) (h : D = 20) : 
  ∃ (n : ℕ), n > 2 ∧ D = n * (n - 3) / 2 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l908_90872


namespace NUMINAMATH_CALUDE_negative_abs_two_squared_equals_two_l908_90844

theorem negative_abs_two_squared_equals_two : (-|2|)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_two_squared_equals_two_l908_90844


namespace NUMINAMATH_CALUDE_contest_possible_orders_l908_90885

/-- The number of questions in the contest -/
def num_questions : ℕ := 10

/-- The number of possible orders to answer the questions -/
def num_possible_orders : ℕ := 512

/-- Theorem stating that the number of possible orders is correct -/
theorem contest_possible_orders :
  (2 ^ (num_questions - 1) : ℕ) = num_possible_orders := by
  sorry

end NUMINAMATH_CALUDE_contest_possible_orders_l908_90885


namespace NUMINAMATH_CALUDE_no_all_power_of_five_l908_90813

theorem no_all_power_of_five : ¬∃ (a : Fin 2018 → ℕ), ∀ i : Fin 2018, 
  ∃ k : ℕ, (a i)^2018 + a (i.succ) = 5^k := by
  sorry

end NUMINAMATH_CALUDE_no_all_power_of_five_l908_90813


namespace NUMINAMATH_CALUDE_cube_root_sum_of_eighth_powers_l908_90853

theorem cube_root_sum_of_eighth_powers (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a^3 - 3*a + 1 = 0 →
  b^3 - 3*b + 1 = 0 →
  c^3 - 3*c + 1 = 0 →
  a^8 + b^8 + c^8 = 186 := by
sorry

end NUMINAMATH_CALUDE_cube_root_sum_of_eighth_powers_l908_90853


namespace NUMINAMATH_CALUDE_tetrahedron_volume_formula_l908_90898

/-- A tetrahedron with an inscribed sphere. -/
structure TetrahedronWithInscribedSphere where
  R : ℝ  -- Radius of the inscribed sphere
  S₁ : ℝ  -- Area of face 1
  S₂ : ℝ  -- Area of face 2
  S₃ : ℝ  -- Area of face 3
  S₄ : ℝ  -- Area of face 4

/-- The volume of a tetrahedron with an inscribed sphere. -/
def volume (t : TetrahedronWithInscribedSphere) : ℝ :=
  t.R * (t.S₁ + t.S₂ + t.S₃ + t.S₄)

/-- Theorem: The volume of a tetrahedron with an inscribed sphere
    is equal to the radius of the inscribed sphere multiplied by
    the sum of the areas of its four faces. -/
theorem tetrahedron_volume_formula (t : TetrahedronWithInscribedSphere) :
  volume t = t.R * (t.S₁ + t.S₂ + t.S₃ + t.S₄) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_formula_l908_90898


namespace NUMINAMATH_CALUDE_pauls_money_duration_l908_90899

/-- 
Given Paul's earnings and weekly spending, prove how long the money will last.
-/
theorem pauls_money_duration (lawn_mowing : ℕ) (weed_eating : ℕ) (weekly_spending : ℕ) 
  (h1 : lawn_mowing = 44)
  (h2 : weed_eating = 28)
  (h3 : weekly_spending = 9) :
  (lawn_mowing + weed_eating) / weekly_spending = 8 := by
  sorry

end NUMINAMATH_CALUDE_pauls_money_duration_l908_90899


namespace NUMINAMATH_CALUDE_infection_spread_theorem_l908_90850

/-- Represents the infection spread in a cube grid -/
structure InfectionSpread where
  edge : Nat
  t : Nat
  h_edge : edge = 2015
  h_t_range : 1 ≤ t ∧ t ≤ edge

/-- The minimum number of initially infected cells for possible complete infection -/
def min_cells_possible (is : InfectionSpread) : Nat :=
  is.t ^ 3

/-- The minimum number of initially infected cells for certain complete infection -/
def min_cells_certain (is : InfectionSpread) : Nat :=
  is.edge ^ 3 - (is.edge - (is.t - 1)) ^ 3 + 1

/-- Theorem stating the minimum number of cells for possible and certain infection -/
theorem infection_spread_theorem (is : InfectionSpread) :
  (min_cells_possible is = is.t ^ 3) ∧
  (min_cells_certain is = is.edge ^ 3 - (is.edge - (is.t - 1)) ^ 3 + 1) := by
  sorry

end NUMINAMATH_CALUDE_infection_spread_theorem_l908_90850


namespace NUMINAMATH_CALUDE_power_function_value_l908_90892

-- Define a power function that passes through (1/2, √2/2)
def f (x : ℝ) : ℝ := x^(1/2)

-- State the theorem
theorem power_function_value : f (1/4) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_value_l908_90892


namespace NUMINAMATH_CALUDE_range_of_a_l908_90822

def linear_function (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x + a - 8

def fractional_equation (a : ℝ) (y : ℝ) : Prop :=
  (y - 5) / (1 - y) + 3 = a / (y - 1)

theorem range_of_a (a : ℝ) :
  (∀ x y, y = linear_function a x → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) ∨ (x > 0 ∧ y < 0)) →
  (∀ y, fractional_equation a y → y > -3) →
  1 < a ∧ a < 8 ∧ a ≠ 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l908_90822


namespace NUMINAMATH_CALUDE_inequality_solution_set_l908_90819

theorem inequality_solution_set (x : ℝ) :
  (∀ x, -x^2 - 3*x + 4 > 0 ↔ -4 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l908_90819


namespace NUMINAMATH_CALUDE_pen_price_correct_max_pens_correct_l908_90812

-- Define the original price of a pen
def original_pen_price : ℝ := 4

-- Define the discount rate for pens in the first part
def discount_rate : ℝ := 0.1

-- Define the total budget
def budget : ℝ := 360

-- Define the number of additional pens that can be bought after discount
def additional_pens : ℕ := 10

-- Define the total number of items to be purchased
def total_items : ℕ := 80

-- Define the original price of a pencil case
def pencil_case_price : ℝ := 10

-- Define the discount rate for both items in the second part
def discount_rate_2 : ℝ := 0.2

-- Define the minimum total purchase amount
def min_purchase_amount : ℝ := 400

theorem pen_price_correct :
  budget / original_pen_price + additional_pens = budget / (original_pen_price * (1 - discount_rate)) :=
sorry

theorem max_pens_correct :
  ∀ y : ℕ, y ≤ 50 →
  y ≤ total_items →
  min_purchase_amount ≤ original_pen_price * (1 - discount_rate_2) * y + pencil_case_price * (1 - discount_rate_2) * (total_items - y) :=
sorry

#check pen_price_correct
#check max_pens_correct

end NUMINAMATH_CALUDE_pen_price_correct_max_pens_correct_l908_90812


namespace NUMINAMATH_CALUDE_rancher_cows_count_l908_90829

theorem rancher_cows_count (horses : ℕ) (cows : ℕ) : 
  cows = 5 * horses →
  cows + horses = 168 →
  cows = 140 := by
sorry

end NUMINAMATH_CALUDE_rancher_cows_count_l908_90829


namespace NUMINAMATH_CALUDE_added_number_proof_l908_90883

theorem added_number_proof : 
  let n : ℝ := 90
  let x : ℝ := 3
  (1/2 : ℝ) * (1/3 : ℝ) * (1/5 : ℝ) * n + x = (1/15 : ℝ) * n := by
  sorry

end NUMINAMATH_CALUDE_added_number_proof_l908_90883


namespace NUMINAMATH_CALUDE_subscription_period_l908_90809

/-- Proves that the subscription period is 18 months given the promotion conditions -/
theorem subscription_period (normal_price : ℚ) (discount_per_issue : ℚ) (total_discount : ℚ) :
  normal_price = 34 →
  discount_per_issue = 0.25 →
  total_discount = 9 →
  ∃ (period : ℕ), period * 2 * discount_per_issue = total_discount ∧ period = 18 :=
by sorry

end NUMINAMATH_CALUDE_subscription_period_l908_90809


namespace NUMINAMATH_CALUDE_expression_factorization_l908_90832

theorem expression_factorization (a b c : ℝ) (h : (a - b) + (b - c) + (c - a) ≠ 0) :
  ((a - b)^2 + (b - c)^2 + (c - a)^2) / ((a - b) + (b - c) + (c - a)) = a - b + b - c + c - a :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l908_90832


namespace NUMINAMATH_CALUDE_sum_not_prime_l908_90825

theorem sum_not_prime (a b c d : ℕ+) (h : a * b = c * d) : 
  ¬ Nat.Prime (a.val + b.val + c.val + d.val) := by
sorry

end NUMINAMATH_CALUDE_sum_not_prime_l908_90825


namespace NUMINAMATH_CALUDE_moon_arrangements_count_l908_90896

/-- The number of letters in the word "MOON" -/
def word_length : ℕ := 4

/-- The number of repeated letters (O's) in the word "MOON" -/
def repeated_letters : ℕ := 2

/-- The number of unique arrangements of the letters in "MOON" -/
def moon_arrangements : ℕ := Nat.factorial word_length / Nat.factorial repeated_letters

theorem moon_arrangements_count : moon_arrangements = 12 := by
  sorry

end NUMINAMATH_CALUDE_moon_arrangements_count_l908_90896


namespace NUMINAMATH_CALUDE_point_plane_configuration_exists_l908_90816

-- Define a type for points in space
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a type for planes in space
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define a function to check if a point lies on a plane
def pointOnPlane (p : Point) (pl : Plane) : Prop :=
  pl.a * p.x + pl.b * p.y + pl.c * p.z + pl.d = 0

-- Define a function to check if a set of points is collinear
def collinear (points : Set Point) : Prop :=
  ∃ (a b c : ℝ), ∀ p ∈ points, a * p.x + b * p.y + c * p.z = 0

-- State the theorem
theorem point_plane_configuration_exists :
  ∃ (points : Set Point) (planes : Set Plane),
    -- There are several points and planes
    (points.Nonempty ∧ planes.Nonempty) ∧
    -- Through any two points, exactly two planes pass
    (∀ p q : Point, p ∈ points → q ∈ points → p ≠ q →
      ∃! (pl1 pl2 : Plane), pl1 ∈ planes ∧ pl2 ∈ planes ∧ pl1 ≠ pl2 ∧
        pointOnPlane p pl1 ∧ pointOnPlane q pl1 ∧
        pointOnPlane p pl2 ∧ pointOnPlane q pl2) ∧
    -- Each plane contains at least four points
    (∀ pl : Plane, pl ∈ planes →
      ∃ (p1 p2 p3 p4 : Point), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ p4 ∈ points ∧
        p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
        pointOnPlane p1 pl ∧ pointOnPlane p2 pl ∧ pointOnPlane p3 pl ∧ pointOnPlane p4 pl) ∧
    -- Not all points lie on a single line
    ¬collinear points :=
by
  sorry

end NUMINAMATH_CALUDE_point_plane_configuration_exists_l908_90816


namespace NUMINAMATH_CALUDE_sum_maximum_l908_90823

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  first_positive : a 1 > 0
  relation : 8 * a 5 = 13 * a 11

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (List.range n).map seq.a |>.sum

/-- The theorem stating when the sum reaches its maximum -/
theorem sum_maximum (seq : ArithmeticSequence) :
  ∃ (n : ℕ), ∀ (m : ℕ), sum_n seq n ≥ sum_n seq m ∧ n = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_maximum_l908_90823


namespace NUMINAMATH_CALUDE_fourth_ball_black_prob_l908_90865

/-- A box containing red and black balls -/
structure Box where
  red_balls : ℕ
  black_balls : ℕ

/-- The probability of selecting a black ball from a box -/
def prob_black_ball (box : Box) : ℚ :=
  box.black_balls / (box.red_balls + box.black_balls)

/-- The theorem stating the probability of the fourth ball being black -/
theorem fourth_ball_black_prob (box : Box) 
  (h1 : box.red_balls = 3) 
  (h2 : box.black_balls = 3) : 
  prob_black_ball box = 1/2 := by
  sorry

#eval prob_black_ball { red_balls := 3, black_balls := 3 }

end NUMINAMATH_CALUDE_fourth_ball_black_prob_l908_90865


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l908_90887

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (d : ℝ) :
  A = 4 * Real.pi →
  A = Real.pi * r^2 →
  d = 2 * r →
  d = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l908_90887


namespace NUMINAMATH_CALUDE_beatles_collection_theorem_l908_90880

/-- The number of albums in either Andrew's or John's collection, but not both -/
def unique_albums (shared : ℕ) (andrew_total : ℕ) (john_unique : ℕ) : ℕ :=
  (andrew_total - shared) + john_unique

theorem beatles_collection_theorem :
  unique_albums 9 17 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_beatles_collection_theorem_l908_90880


namespace NUMINAMATH_CALUDE_writing_time_for_three_books_l908_90824

/-- Calculates the number of days required to write multiple books given the daily writing rate and book length. -/
def days_to_write_books (pages_per_day : ℕ) (pages_per_book : ℕ) (num_books : ℕ) : ℕ :=
  (pages_per_book * num_books) / pages_per_day

/-- Theorem stating that it takes 60 days to write 3 books of 400 pages each at a rate of 20 pages per day. -/
theorem writing_time_for_three_books :
  days_to_write_books 20 400 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_writing_time_for_three_books_l908_90824


namespace NUMINAMATH_CALUDE_max_abs_z_given_condition_l908_90855

theorem max_abs_z_given_condition (z : ℂ) (h : Complex.abs (z + Complex.I) + Complex.abs (z - Complex.I) = 2) : 
  Complex.abs z ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_z_given_condition_l908_90855


namespace NUMINAMATH_CALUDE_rectangle_x_value_l908_90852

/-- A rectangle in a 2D plane --/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The area of a rectangle --/
def rectangleArea (r : Rectangle) : ℝ :=
  (r.C.1 - r.A.1) * (r.B.2 - r.A.2)

theorem rectangle_x_value 
  (x : ℝ) 
  (h_pos : x > 0) 
  (rect : Rectangle) 
  (h_vertices : rect = { 
    A := (0, 0), 
    B := (0, 4), 
    C := (x, 4), 
    D := (x, 0) 
  }) 
  (h_area : rectangleArea rect = 28) : 
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_rectangle_x_value_l908_90852


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_l908_90876

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and lines
variable (perp : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_transitivity 
  (α β : Plane) (m n : Line) 
  (h1 : perp_line_plane n α) 
  (h2 : perp_line_plane n β) 
  (h3 : perp_line_plane m α) : 
  perp_line_plane m β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_l908_90876


namespace NUMINAMATH_CALUDE_square_diagonal_perimeter_ratio_l908_90870

theorem square_diagonal_perimeter_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (a * Real.sqrt 2) / (b * Real.sqrt 2) = 5/2 → (4 * a) / (4 * b) = 5/2 := by
sorry

end NUMINAMATH_CALUDE_square_diagonal_perimeter_ratio_l908_90870


namespace NUMINAMATH_CALUDE_edwards_remaining_money_l908_90860

/-- Calculates the remaining money after a purchase -/
def remainingMoney (initialAmount spentAmount : ℕ) : ℕ :=
  initialAmount - spentAmount

/-- Theorem: Edward's remaining money is $6 -/
theorem edwards_remaining_money :
  remainingMoney 22 16 = 6 := by
  sorry

end NUMINAMATH_CALUDE_edwards_remaining_money_l908_90860


namespace NUMINAMATH_CALUDE_B_max_at_50_l908_90830

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Definition of B_k -/
def B (k : ℕ) : ℝ := (binomial 500 k) * (0.1 ^ k)

/-- Statement: B_k is largest when k = 50 -/
theorem B_max_at_50 : ∀ k : ℕ, k ≤ 500 → B k ≤ B 50 := by sorry

end NUMINAMATH_CALUDE_B_max_at_50_l908_90830


namespace NUMINAMATH_CALUDE_friend_total_time_l908_90828

def my_reading_time : ℝ := 3 * 60 -- 3 hours in minutes
def my_writing_time : ℝ := 60 -- 1 hour in minutes
def friend_reading_speed_ratio : ℝ := 4 -- friend reads 4 times as fast

theorem friend_total_time (friend_reading_time friend_writing_time : ℝ) :
  friend_reading_time = my_reading_time / friend_reading_speed_ratio →
  friend_writing_time = my_writing_time →
  friend_reading_time + friend_writing_time = 105 := by
sorry

end NUMINAMATH_CALUDE_friend_total_time_l908_90828


namespace NUMINAMATH_CALUDE_unique_cut_l908_90858

/-- Represents a cut of the original number -/
structure Cut where
  pos1 : Nat
  pos2 : Nat
  valid : pos1 < pos2 ∧ pos2 < 5

/-- Checks if a given cut produces the required difference -/
def isValidCut (c : Cut) : Prop :=
  let part1 := 12345 / (10^(5 - c.pos1))
  let part2 := (12345 / (10^(5 - c.pos2))) % (10^(c.pos2 - c.pos1))
  let part3 := 12345 % (10^(5 - c.pos2))
  (part1 * 10^4 + part2 * 10^(5 - c.pos2) + part3) -
  (part2 * 10^4 + part3 * 10^c.pos1 + part1) = 28926

theorem unique_cut : 
  ∃! c : Cut, isValidCut c ∧ c.pos1 = 1 ∧ c.pos2 = 4 :=
sorry

end NUMINAMATH_CALUDE_unique_cut_l908_90858


namespace NUMINAMATH_CALUDE_complex_equality_implies_a_equals_three_l908_90834

theorem complex_equality_implies_a_equals_three (a : ℝ) : 
  Complex.re ((1 + a * Complex.I) * (2 - Complex.I)) = Complex.im ((1 + a * Complex.I) * (2 - Complex.I)) →
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_equality_implies_a_equals_three_l908_90834


namespace NUMINAMATH_CALUDE_cube_surface_area_l908_90888

/-- The surface area of a cube with edge length 5 cm is 150 cm². -/
theorem cube_surface_area (edge_length : ℝ) (h : edge_length = 5) :
  6 * edge_length ^ 2 = 150 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l908_90888


namespace NUMINAMATH_CALUDE_divisible_by_27_l908_90869

theorem divisible_by_27 (n : ℕ) : ∃ k : ℤ, 2^(2*n - 1) - 9*n^2 + 21*n - 14 = 27*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_27_l908_90869


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l908_90862

/-- A triangle with altitudes 9, 12, and 18 -/
structure TriangleWithAltitudes where
  a : ℝ
  b : ℝ
  c : ℝ
  altitude_a : ℝ
  altitude_b : ℝ
  altitude_c : ℝ
  ha : altitude_a = 9
  hb : altitude_b = 12
  hc : altitude_c = 18
  area_eq1 : a * altitude_a = b * altitude_b
  area_eq2 : b * altitude_b = c * altitude_c
  triangle_ineq1 : a + b > c
  triangle_ineq2 : b + c > a
  triangle_ineq3 : c + a > b

/-- The largest angle in a triangle with altitudes 9, 12, and 18 is arccos(-1/4) -/
theorem largest_angle_in_special_triangle (t : TriangleWithAltitudes) :
  ∃ θ : ℝ, θ = Real.arccos (-1/4) ∧ 
  θ = max (Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)))
         (max (Real.arccos ((a^2 + c^2 - b^2) / (2*a*c)))
              (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)))) :=
by sorry


end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l908_90862


namespace NUMINAMATH_CALUDE_wire_cutting_l908_90877

theorem wire_cutting (total_length : ℝ) (piece1 piece2 piece3 : ℝ) : 
  total_length = 95 →
  piece2 = 1.5 * piece1 →
  piece3 = 1.5 * piece2 →
  piece1 + piece2 + piece3 = total_length →
  piece3 = 45 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l908_90877


namespace NUMINAMATH_CALUDE_binary_to_octal_conversion_l908_90867

/-- Convert a binary number represented as a list of bits to decimal --/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Convert a decimal number to octal --/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem binary_to_octal_conversion :
  let binary := [false, false, true, true, true, false, true, true]
  let decimal := binary_to_decimal binary.reverse
  let octal := decimal_to_octal decimal
  decimal = 220 ∧ octal = [3, 3, 4] := by
  sorry

#eval binary_to_decimal [false, false, true, true, true, false, true, true].reverse
#eval decimal_to_octal 220

end NUMINAMATH_CALUDE_binary_to_octal_conversion_l908_90867


namespace NUMINAMATH_CALUDE_order_of_x_l908_90820

theorem order_of_x (x₁ x₂ x₃ x₄ x₅ a₁ a₂ a₃ a₄ a₅ : ℝ)
  (eq1 : x₁ + x₂ + x₃ = a₁)
  (eq2 : x₂ + x₃ + x₁ = a₂)
  (eq3 : x₃ + x₄ + x₅ = a₃)
  (eq4 : x₄ + x₅ + x₁ = a₄)
  (eq5 : x₅ + x₁ + x₂ = a₅)
  (h : a₁ > a₂ ∧ a₂ > a₃ ∧ a₃ > a₄ ∧ a₄ > a₅) :
  x₃ > x₁ ∧ x₁ > x₄ ∧ x₄ > x₂ ∧ x₂ > x₅ := by
  sorry

end NUMINAMATH_CALUDE_order_of_x_l908_90820


namespace NUMINAMATH_CALUDE_gretchen_scuba_trips_l908_90864

/-- The minimum number of trips required to transport a given number of objects,
    where each trip can carry a fixed number of objects. -/
def min_trips (total_objects : ℕ) (objects_per_trip : ℕ) : ℕ :=
  (total_objects + objects_per_trip - 1) / objects_per_trip

/-- Theorem stating that 6 trips are required to transport 17 objects
    when carrying 3 objects per trip. -/
theorem gretchen_scuba_trips :
  min_trips 17 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gretchen_scuba_trips_l908_90864


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l908_90868

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let roots := {x : ℝ | a * x^2 + b * x + c = 0}
  (∃ x y : ℝ, roots = {x, y}) →
  (∃ s : ℝ, ∀ z ∈ roots, ∃ w ∈ roots, z + w = s) →
  (∃ s : ℝ, ∀ z ∈ roots, ∃ w ∈ roots, z + w = s) → s = -b / a :=
by sorry

theorem sum_of_roots_specific_quadratic :
  let roots := {x : ℝ | x^2 - 6*x + 8 = 0}
  (∃ x y : ℝ, roots = {x, y}) →
  (∃ s : ℝ, ∀ z ∈ roots, ∃ w ∈ roots, z + w = s) →
  s = 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l908_90868


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l908_90818

/-- Given real numbers x, y, and z, if -1, x, y, z, -3 form a geometric sequence,
    then the product of x and z equals 3. -/
theorem geometric_sequence_product (x y z : ℝ) :
  (∃ r : ℝ, r ≠ 0 ∧ x = -1 * r ∧ y = x * r ∧ z = y * r ∧ -3 = z * r) →
  x * z = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l908_90818


namespace NUMINAMATH_CALUDE_digit_2000th_position_l908_90845

/-- The sequence of digits formed by concatenating consecutive positive integers starting from 1 -/
def concatenatedSequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => (concatenatedSequence n) * 10 + ((n + 1) % 10)

/-- The digit at a given position in the concatenated sequence -/
def digitAtPosition (pos : ℕ) : ℕ :=
  (concatenatedSequence pos) % 10

theorem digit_2000th_position :
  digitAtPosition 1999 = 0 := by
  sorry

end NUMINAMATH_CALUDE_digit_2000th_position_l908_90845


namespace NUMINAMATH_CALUDE_range_of_a_l908_90815

theorem range_of_a (a : ℝ) : (¬ ∃ x, x < 2023 ∧ x > a) → a ≥ 2023 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l908_90815


namespace NUMINAMATH_CALUDE_square_field_area_l908_90875

/-- Given a square field with side length s, prove that the area is 27889 square meters
    when the cost of barbed wire at 1.20 per meter for (4s - 2) meters equals 799.20. -/
theorem square_field_area (s : ℝ) : 
  (4 * s - 2) * 1.20 = 799.20 → s^2 = 27889 := by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l908_90875


namespace NUMINAMATH_CALUDE_sequence_sum_l908_90802

theorem sequence_sum (A B C D E F G H I J : ℤ) : 
  E = 8 ∧ 
  A + B + C = 27 ∧ 
  B + C + D = 27 ∧ 
  C + D + E = 27 ∧ 
  D + E + F = 27 ∧ 
  E + F + G = 27 ∧ 
  F + G + H = 27 ∧ 
  G + H + I = 27 ∧ 
  H + I + J = 27 
  → A + J = -27 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l908_90802


namespace NUMINAMATH_CALUDE_test_results_l908_90889

/-- The probability of exactly two people meeting the standard in a test where
    A, B, and C have independent probabilities of 2/5, 3/4, and 1/2 respectively. -/
def prob_two_meet_standard : ℚ := 17/40

/-- The most likely number of people to meet the standard in the test. -/
def most_likely_number : ℕ := 2

/-- Probabilities of A, B, and C meeting the standard -/
def prob_A : ℚ := 2/5
def prob_B : ℚ := 3/4
def prob_C : ℚ := 1/2

theorem test_results :
  (prob_two_meet_standard = prob_A * prob_B * (1 - prob_C) +
                            prob_A * (1 - prob_B) * prob_C +
                            (1 - prob_A) * prob_B * prob_C) ∧
  (most_likely_number = 2) := by
  sorry

end NUMINAMATH_CALUDE_test_results_l908_90889


namespace NUMINAMATH_CALUDE_existence_of_special_multiple_l908_90894

theorem existence_of_special_multiple (p : ℕ) (hp : p > 1) (hgcd : Nat.gcd p 10 = 1) :
  ∃ n : ℕ, 
    (Nat.digits 10 n).length = p - 2 ∧ 
    (∀ d ∈ Nat.digits 10 n, d = 1 ∨ d = 3) ∧
    p ∣ n :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_multiple_l908_90894


namespace NUMINAMATH_CALUDE_unique_solution_values_l908_90846

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : Prop := a * x^2 - 2 * x + 1 = 0

-- Define the property of having exactly one solution
def has_unique_solution (a : ℝ) : Prop :=
  ∃! x, quadratic_equation a x

-- Theorem statement
theorem unique_solution_values :
  ∀ a : ℝ, has_unique_solution a ↔ (a = 0 ∨ a = 1) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_values_l908_90846


namespace NUMINAMATH_CALUDE_number_problem_l908_90800

theorem number_problem (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 14) : 
  (40/100) * N = 168 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l908_90800


namespace NUMINAMATH_CALUDE_sin_to_cos_transformation_l908_90874

theorem sin_to_cos_transformation (x : ℝ) :
  Real.sqrt 2 * Real.sin (2 * x + Real.pi / 4) =
  Real.sqrt 2 * Real.cos (2 * x - Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_sin_to_cos_transformation_l908_90874


namespace NUMINAMATH_CALUDE_new_figure_length_is_32_l908_90817

/-- Represents the dimensions of the original polygon --/
structure PolygonDimensions where
  vertical_side : ℝ
  top_first_horizontal : ℝ
  top_second_horizontal : ℝ
  remaining_horizontal : ℝ
  last_vertical_drop : ℝ

/-- Calculates the total length of segments in the new figure after removing four sides --/
def newFigureLength (d : PolygonDimensions) : ℝ :=
  d.vertical_side + (d.top_first_horizontal + d.top_second_horizontal + d.remaining_horizontal) +
  (d.vertical_side - d.last_vertical_drop) + d.last_vertical_drop

/-- Theorem stating that for the given dimensions, the new figure length is 32 units --/
theorem new_figure_length_is_32 (d : PolygonDimensions)
    (h1 : d.vertical_side = 10)
    (h2 : d.top_first_horizontal = 3)
    (h3 : d.top_second_horizontal = 4)
    (h4 : d.remaining_horizontal = 5)
    (h5 : d.last_vertical_drop = 2) :
    newFigureLength d = 32 := by
  sorry

end NUMINAMATH_CALUDE_new_figure_length_is_32_l908_90817


namespace NUMINAMATH_CALUDE_squirrel_count_l908_90831

theorem squirrel_count (first_count : ℕ) (second_count : ℕ) : 
  first_count = 12 → 
  second_count = first_count + first_count / 3 → 
  first_count + second_count = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_squirrel_count_l908_90831


namespace NUMINAMATH_CALUDE_quadrilaterals_on_circle_l908_90847

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of points on the circle -/
def num_points : ℕ := 12

/-- The number of vertices in a quadrilateral -/
def vertices_per_quad : ℕ := 4

/-- The number of points to choose from after fixing two points -/
def remaining_points : ℕ := num_points - 2

/-- The number of additional vertices needed after fixing two points -/
def additional_vertices : ℕ := vertices_per_quad - 2

theorem quadrilaterals_on_circle :
  choose num_points vertices_per_quad - choose remaining_points additional_vertices = 450 := by
  sorry

end NUMINAMATH_CALUDE_quadrilaterals_on_circle_l908_90847


namespace NUMINAMATH_CALUDE_exam_time_ratio_l908_90806

theorem exam_time_ratio :
  let total_time : ℕ := 3 * 60  -- 3 hours in minutes
  let time_type_a : ℕ := 120    -- Time spent on type A problems in minutes
  let time_type_b : ℕ := total_time - time_type_a  -- Time spent on type B problems
  (time_type_a : ℚ) / time_type_b = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_exam_time_ratio_l908_90806


namespace NUMINAMATH_CALUDE_line_disjoint_from_circle_l908_90804

/-- Given a point M(a,b) inside the unit circle, prove that the line ax + by = 1 is disjoint from the circle -/
theorem line_disjoint_from_circle (a b : ℝ) (h : a^2 + b^2 < 1) :
  ∀ x y : ℝ, x^2 + y^2 = 1 → a*x + b*y ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_line_disjoint_from_circle_l908_90804


namespace NUMINAMATH_CALUDE_wire_cutting_l908_90878

theorem wire_cutting (total_length : ℝ) (difference : ℝ) 
  (h1 : total_length = 30)
  (h2 : difference = 2) :
  ∃ (shorter longer : ℝ),
    shorter + longer = total_length ∧
    longer = shorter + difference ∧
    shorter = 14 ∧
    longer = 16 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l908_90878


namespace NUMINAMATH_CALUDE_fn_equals_de_l908_90849

-- Define the circle
variable (O : Point) (A B : Point)
variable (circle : Circle O)

-- Define other points
variable (C D E F M N : Point)

-- Define the conditions
variable (h1 : C ∈ circle)
variable (h2 : Diameter circle A B)
variable (h3 : Perpendicular CD AB D)
variable (h4 : E ∈ Segment B D)
variable (h5 : AE = AC)
variable (h6 : Square D E F M)
variable (h7 : N ∈ circle ∩ Line A M)

-- State the theorem
theorem fn_equals_de : FN = DE := by
  sorry

end NUMINAMATH_CALUDE_fn_equals_de_l908_90849


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l908_90835

theorem quadratic_roots_expression (a b : ℝ) : 
  (a^2 - a - 1 = 0) → (b^2 - b - 1 = 0) → (3*a^2 + 2*b^2 - 3*a - 2*b = 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l908_90835


namespace NUMINAMATH_CALUDE_at_least_one_leq_neg_two_l908_90842

theorem at_least_one_leq_neg_two (a b c : ℝ) 
  (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  (a + 1/b ≤ -2) ∨ (b + 1/c ≤ -2) ∨ (c + 1/a ≤ -2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_leq_neg_two_l908_90842


namespace NUMINAMATH_CALUDE_remainder_theorem_l908_90837

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^5 - 2*x^3 + 4*x^2 + x + 5

-- Theorem statement
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, p = λ x => (x + 2) * q x + 3 :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l908_90837


namespace NUMINAMATH_CALUDE_christinas_walking_speed_l908_90854

/-- The problem of finding Christina's walking speed -/
theorem christinas_walking_speed 
  (initial_distance : ℝ) 
  (jacks_speed : ℝ) 
  (lindys_speed : ℝ) 
  (lindys_total_distance : ℝ) 
  (h1 : initial_distance = 270) 
  (h2 : jacks_speed = 4) 
  (h3 : lindys_speed = 8) 
  (h4 : lindys_total_distance = 240) : 
  ∃ (christinas_speed : ℝ), christinas_speed = 5 := by
  sorry

#check christinas_walking_speed

end NUMINAMATH_CALUDE_christinas_walking_speed_l908_90854


namespace NUMINAMATH_CALUDE_parabola_intersection_l908_90866

/-- First parabola equation -/
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

/-- Second parabola equation -/
def g (x : ℝ) : ℝ := -x^2 + 6 * x + 8

/-- Theorem stating that (-0.5, 4.75) and (3, 17) are the only intersection points -/
theorem parabola_intersection :
  (∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x = -0.5 ∧ y = 4.75) ∨ (x = 3 ∧ y = 17)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_l908_90866


namespace NUMINAMATH_CALUDE_repeating_decimal_ratio_l908_90859

/-- Represents a repeating decimal with a repeating part of two digits -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b) / 99

theorem repeating_decimal_ratio : 
  (RepeatingDecimal 5 4) / (RepeatingDecimal 1 8) = 3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_ratio_l908_90859


namespace NUMINAMATH_CALUDE_difference_of_squares_l908_90884

theorem difference_of_squares : 75^2 - 25^2 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l908_90884


namespace NUMINAMATH_CALUDE_point_p_coordinates_l908_90890

/-- Given points A, B, C in ℝ³ and a point P such that vector AP is half of vector CB,
    prove that P has the specified coordinates. -/
theorem point_p_coordinates (A B C P : ℝ × ℝ × ℝ) : 
  A = (2, -1, 2) → 
  B = (4, 5, -1) → 
  C = (-2, 2, 3) → 
  P - A = (1/2 : ℝ) • (B - C) → 
  P = (5, 1/2, 0) := by
sorry


end NUMINAMATH_CALUDE_point_p_coordinates_l908_90890


namespace NUMINAMATH_CALUDE_gamma_value_l908_90886

theorem gamma_value (γ δ : ℂ) 
  (h1 : (γ + δ).re > 0)
  (h2 : (Complex.I * (γ - δ)).re > 0)
  (h3 : δ = 2 + 3 * Complex.I) : 
  γ = 2 - 3 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_gamma_value_l908_90886


namespace NUMINAMATH_CALUDE_sin_2alpha_values_l908_90856

theorem sin_2alpha_values (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : 3 * Real.cos (2 * α) = Real.sin (π / 4 - α)) :
  Real.sin (2 * α) = 1 ∨ Real.sin (2 * α) = -17/18 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_values_l908_90856


namespace NUMINAMATH_CALUDE_sum_always_positive_l908_90836

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f (x + 4)) ∧
  (∀ x ≥ 2, Monotone (fun y ↦ f y))

/-- Theorem statement -/
theorem sum_always_positive
  (f : ℝ → ℝ)
  (hf : special_function f)
  (x₁ x₂ : ℝ)
  (h_sum : x₁ + x₂ > 4)
  (h_prod : (x₁ - 2) * (x₂ - 2) < 0) :
  f x₁ + f x₂ > 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_always_positive_l908_90836


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l908_90882

theorem sufficient_not_necessary_condition (a b c d : ℝ) :
  (∀ a b c d : ℝ, a > b ∧ c > d → a + c > b + d) ∧
  (∃ a b c d : ℝ, a + c > b + d ∧ ¬(a > b ∧ c > d)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l908_90882


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l908_90811

/-- A line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Check if a point lies on a line -/
def pointOnLine (x y : ℝ) (l : Line) : Prop :=
  y = l.slope * x + l.intercept

/-- The given line y = -2x + 1 -/
def givenLine : Line :=
  { slope := -2, intercept := 1 }

theorem parallel_line_through_point :
  ∃ (l : Line), parallel l givenLine ∧ pointOnLine (-1) 2 l ∧ l.slope * x + l.intercept = -2 * x :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l908_90811


namespace NUMINAMATH_CALUDE_intercept_sum_l908_90881

theorem intercept_sum : ∃ (x₀ y₀ : ℕ), 
  x₀ < 25 ∧ y₀ < 25 ∧
  (4 * x₀) % 25 = 2 % 25 ∧
  (5 * y₀ + 2) % 25 = 0 ∧
  x₀ + y₀ = 28 := by
sorry

end NUMINAMATH_CALUDE_intercept_sum_l908_90881


namespace NUMINAMATH_CALUDE_zhonghuan_cup_exam_l908_90873

theorem zhonghuan_cup_exam (total : ℕ) (english : ℕ) (chinese : ℕ) (both : ℕ) 
  (h1 : total = 45)
  (h2 : english = 35)
  (h3 : chinese = 31)
  (h4 : both = 24) :
  total - (english + chinese - both) = 3 := by
  sorry

end NUMINAMATH_CALUDE_zhonghuan_cup_exam_l908_90873


namespace NUMINAMATH_CALUDE_farmers_animal_purchase_l908_90803

/-- The farmer's animal purchase problem -/
theorem farmers_animal_purchase
  (total : ℕ) (goat_pig_sheep : ℕ) (cow_pig_sheep : ℕ) (goat_pig : ℕ)
  (h1 : total = 1325)
  (h2 : goat_pig_sheep = 425)
  (h3 : cow_pig_sheep = 1225)
  (h4 : goat_pig = 275) :
  ∃ (cow goat sheep pig : ℕ),
    cow + goat + sheep + pig = total ∧
    goat + sheep + pig = goat_pig_sheep ∧
    cow + sheep + pig = cow_pig_sheep ∧
    goat + pig = goat_pig ∧
    cow = 900 ∧ goat = 100 ∧ sheep = 150 ∧ pig = 175 := by
  sorry


end NUMINAMATH_CALUDE_farmers_animal_purchase_l908_90803


namespace NUMINAMATH_CALUDE_function_inequality_solution_l908_90814

theorem function_inequality_solution (f : ℕ → ℝ) 
  (h1 : ∀ n ≥ 2, n * f n - (n - 1) * f (n + 1) ≥ 1)
  (h2 : f 2 = 3) :
  ∃ g : ℕ → ℝ, 
    (∀ n ≥ 2, f n = 1 + (n - 1) * g n) ∧ 
    (∀ n ≥ 2, g n ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_solution_l908_90814


namespace NUMINAMATH_CALUDE_union_complement_equals_set_l908_90891

def U : Finset Nat := {0, 1, 2, 4, 6, 8}
def M : Finset Nat := {0, 4, 6}
def N : Finset Nat := {0, 1, 6}

theorem union_complement_equals_set : M ∪ (U \ N) = {0, 2, 4, 6, 8} := by sorry

end NUMINAMATH_CALUDE_union_complement_equals_set_l908_90891


namespace NUMINAMATH_CALUDE_expression_simplification_l908_90871

theorem expression_simplification (a b x y : ℝ) 
  (h1 : a * b * (x^2 - y^2) + x * y * (a^2 - b^2) ≠ 0)
  (h2 : x ≠ -a * y / b)
  (h3 : x ≠ b * y / a) :
  ((a * b * (x^2 + y^2) + x * y * (a^2 + b^2)) * ((a * x + b * y)^2 - 4 * a * b * x * y)) /
  (a * b * (x^2 - y^2) + x * y * (a^2 - b^2)) = a^2 * x^2 - b^2 * y^2 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l908_90871


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_fifty_l908_90895

theorem at_least_one_greater_than_fifty (a₁ a₂ : ℝ) (h : a₁ + a₂ > 100) :
  a₁ > 50 ∨ a₂ > 50 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_fifty_l908_90895
