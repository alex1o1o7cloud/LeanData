import Mathlib

namespace NUMINAMATH_CALUDE_arrange_five_and_three_books_l962_96226

/-- The number of ways to arrange two types of indistinguishable objects in a row -/
def arrange_books (n m : ℕ) : ℕ := Nat.choose (n + m) n

/-- Theorem: Arranging 5 copies of one book and 3 copies of another book results in 56 ways -/
theorem arrange_five_and_three_books : arrange_books 5 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_arrange_five_and_three_books_l962_96226


namespace NUMINAMATH_CALUDE_house_distances_l962_96200

-- Define the positions of houses on a straight line
variable (A B V G : ℝ)

-- Define the distances between houses
def AB := |A - B|
def VG := |V - G|
def AG := |A - G|
def BV := |B - V|

-- State the theorem
theorem house_distances (h1 : AB = 600) (h2 : VG = 600) (h3 : AG = 3 * BV) :
  AG = 900 ∨ AG = 1800 := by
  sorry

end NUMINAMATH_CALUDE_house_distances_l962_96200


namespace NUMINAMATH_CALUDE_divisibility_product_l962_96209

theorem divisibility_product (a b c d : ℤ) : a ∣ b → c ∣ d → (a * c) ∣ (b * d) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_product_l962_96209


namespace NUMINAMATH_CALUDE_least_months_to_triple_debt_l962_96235

theorem least_months_to_triple_debt (interest_rate : ℝ) (compound_frequency : ℕ) : 
  interest_rate = 0.06 → compound_frequency = 1 →
  ∃ t : ℕ, t = 19 ∧ (∀ n : ℕ, n < t → (1 + interest_rate) ^ n ≤ 3) ∧ (1 + interest_rate) ^ t > 3 :=
by sorry

end NUMINAMATH_CALUDE_least_months_to_triple_debt_l962_96235


namespace NUMINAMATH_CALUDE_function_properties_l962_96290

-- Define the function f(x) = -2x + 1
def f (x : ℝ) : ℝ := -2 * x + 1

-- State the theorem
theorem function_properties :
  (f 1 = -1) ∧ 
  (∃ (x y z : ℝ), x > 0 ∧ y < 0 ∧ z > 0 ∧ f x > 0 ∧ f y < 0 ∧ f z < 0) ∧
  (∀ (x y : ℝ), x < y → f x > f y) ∧
  (∃ (x : ℝ), x > 0 ∧ f x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l962_96290


namespace NUMINAMATH_CALUDE_f_max_value_inequality_proof_l962_96274

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| - |2*x + 4|

-- Statement 1: The maximum value of f(x) is 3
theorem f_max_value : ∃ (M : ℝ), M = 3 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

-- Statement 2: For positive real numbers x, y, z such that x + y + z = 3, y²/x + z²/y + x²/z ≥ 3
theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 3) :
  y^2 / x + z^2 / y + x^2 / z ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_f_max_value_inequality_proof_l962_96274


namespace NUMINAMATH_CALUDE_complex_square_i_positive_l962_96264

theorem complex_square_i_positive (a : ℝ) : 
  (((a + Complex.I) ^ 2) * Complex.I).re > 0 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_i_positive_l962_96264


namespace NUMINAMATH_CALUDE_total_spent_equals_20_l962_96248

def bracelet_price : ℕ := 4
def keychain_price : ℕ := 5
def coloring_book_price : ℕ := 3

def paula_bracelets : ℕ := 2
def paula_keychains : ℕ := 1
def olive_coloring_books : ℕ := 1
def olive_bracelets : ℕ := 1

def paula_total : ℕ := paula_bracelets * bracelet_price + paula_keychains * keychain_price
def olive_total : ℕ := olive_coloring_books * coloring_book_price + olive_bracelets * bracelet_price

theorem total_spent_equals_20 : paula_total + olive_total = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_equals_20_l962_96248


namespace NUMINAMATH_CALUDE_rectangle_length_fraction_l962_96266

theorem rectangle_length_fraction (square_area : ℝ) (rectangle_area : ℝ) (rectangle_breadth : ℝ)
  (h1 : square_area = 4761)
  (h2 : rectangle_area = 598)
  (h3 : rectangle_breadth = 13) :
  let circle_radius := Real.sqrt square_area
  let rectangle_length := rectangle_area / rectangle_breadth
  rectangle_length / circle_radius = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_rectangle_length_fraction_l962_96266


namespace NUMINAMATH_CALUDE_function_form_l962_96257

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + 2 * y) = 6 * x + f (f y - x)

/-- The theorem stating the form of the function satisfying the equation -/
theorem function_form (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∃ a : ℝ, ∀ x : ℝ, f x = 2 * x + a := by
  sorry

end NUMINAMATH_CALUDE_function_form_l962_96257


namespace NUMINAMATH_CALUDE_g_property_S_sum_S_difference_l962_96232

def g (k : ℕ+) : ℕ+ :=
  sorry

def S (n : ℕ) : ℕ :=
  sorry

theorem g_property (m : ℕ+) : g (2 * m) = g m :=
  sorry

theorem S_sum : S 1 + S 2 + S 3 = 30 :=
  sorry

theorem S_difference (n : ℕ) (h : n ≥ 2) : S n - S (n - 1) = 4^(n - 1) :=
  sorry

end NUMINAMATH_CALUDE_g_property_S_sum_S_difference_l962_96232


namespace NUMINAMATH_CALUDE_xiao_gang_steps_for_one_kilocalorie_l962_96225

/-- The number of steps Xiao Gang walks for 1 kilocalorie of energy -/
def xiao_gang_steps : ℕ := 30

/-- The number of steps Xiao Qiong walks for 1 kilocalorie of energy -/
def xiao_qiong_steps : ℕ := xiao_gang_steps + 15

/-- The total steps Xiao Gang walks for a certain amount of energy -/
def xiao_gang_total_steps : ℕ := 9000

/-- The total steps Xiao Qiong walks for the same amount of energy as Xiao Gang -/
def xiao_qiong_total_steps : ℕ := 13500

theorem xiao_gang_steps_for_one_kilocalorie :
  xiao_gang_steps = 30 ∧
  xiao_qiong_steps = xiao_gang_steps + 15 ∧
  xiao_gang_total_steps * xiao_qiong_steps = xiao_qiong_total_steps * xiao_gang_steps :=
by sorry

end NUMINAMATH_CALUDE_xiao_gang_steps_for_one_kilocalorie_l962_96225


namespace NUMINAMATH_CALUDE_f_increasing_after_one_l962_96293

def f (x : ℝ) : ℝ := (x - 1)^2 + 5

theorem f_increasing_after_one :
  ∀ x₁ x₂ : ℝ, x₁ > 1 → x₂ > x₁ → f x₂ > f x₁ := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_after_one_l962_96293


namespace NUMINAMATH_CALUDE_three_digit_difference_times_second_largest_l962_96228

def largest_three_digit (a b c : Nat) : Nat :=
  100 * max a (max b c) + 10 * max (min (max a b) (max b c)) (min a (min b c)) + min a (min b c)

def smallest_three_digit (a b c : Nat) : Nat :=
  100 * min a (min b c) + 10 * min (max (min a b) (min b c)) (max a (max b c)) + max a (max b c)

def second_largest_three_digit (a b c : Nat) : Nat :=
  let max_digit := max a (max b c)
  let min_digit := min a (min b c)
  let mid_digit := a + b + c - max_digit - min_digit
  100 * max_digit + 10 * mid_digit + min_digit

theorem three_digit_difference_times_second_largest (a b c : Nat) 
  (ha : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (hd : a ∈ [2, 5, 8] ∧ b ∈ [2, 5, 8] ∧ c ∈ [2, 5, 8]) : 
  (largest_three_digit a b c - smallest_three_digit a b c) * second_largest_three_digit a b c = 490050 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_difference_times_second_largest_l962_96228


namespace NUMINAMATH_CALUDE_inverse_propositions_l962_96250

-- Definitions for geometric concepts
def Line : Type := sorry
def Angle : Type := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def corresponding_angles_equal (l1 l2 : Line) : Prop := sorry

-- Definition for last digit
def last_digit (n : ℕ) : ℕ := n % 10

theorem inverse_propositions :
  -- 1. If two lines are parallel, then the corresponding angles are equal
  (∀ (l1 l2 : Line), parallel l1 l2 → corresponding_angles_equal l1 l2) ∧
  -- 2. There exist a and b such that a² = b² but a ≠ b
  (∃ (a b : ℝ), a^2 = b^2 ∧ a ≠ b) ∧
  -- 3. There exists a number divisible by 5 whose last digit is not 0
  (∃ (n : ℕ), n % 5 = 0 ∧ last_digit n ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_inverse_propositions_l962_96250


namespace NUMINAMATH_CALUDE_maria_average_balance_l962_96219

def maria_balance : List ℝ := [50, 250, 100, 200, 150, 250]

theorem maria_average_balance :
  (maria_balance.sum / maria_balance.length : ℝ) = 1000 / 6 := by sorry

end NUMINAMATH_CALUDE_maria_average_balance_l962_96219


namespace NUMINAMATH_CALUDE_y₁_less_than_y₂_l962_96237

/-- Linear function f(x) = 2x + 1 -/
def f (x : ℝ) : ℝ := 2 * x + 1

/-- y₁ is the value of f at x = -5 -/
def y₁ : ℝ := f (-5)

/-- y₂ is the value of f at x = 3 -/
def y₂ : ℝ := f 3

theorem y₁_less_than_y₂ : y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_less_than_y₂_l962_96237


namespace NUMINAMATH_CALUDE_triangle_existence_l962_96281

theorem triangle_existence (x : ℕ) : 
  (∃ (a b c : ℝ), a = 8 ∧ b = 12 ∧ c = x^3 + 1 ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b) ↔ (x = 2 ∨ x = 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_l962_96281


namespace NUMINAMATH_CALUDE_stating_pyramid_base_is_isosceles_l962_96211

/-- Represents a triangular pyramid -/
structure TriangularPyramid where
  /-- The length of each lateral edge -/
  edge_length : ℝ
  /-- The area of each lateral face -/
  face_area : ℝ
  /-- Assumption that all lateral edges have the same length -/
  equal_edges : edge_length > 0
  /-- Assumption that all lateral faces have the same area -/
  equal_faces : face_area > 0

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  /-- The length of the two equal sides -/
  equal_side : ℝ
  /-- The length of the base -/
  base : ℝ
  /-- Assumption that the equal sides are positive -/
  positive_equal_side : equal_side > 0
  /-- Assumption that the base is positive -/
  positive_base : base > 0

/-- 
Theorem stating that the base of a triangular pyramid with equal lateral edges 
and equal lateral face areas is an isosceles triangle 
-/
theorem pyramid_base_is_isosceles (p : TriangularPyramid) : 
  ∃ (t : IsoscelesTriangle), True :=
sorry

end NUMINAMATH_CALUDE_stating_pyramid_base_is_isosceles_l962_96211


namespace NUMINAMATH_CALUDE_victoria_worked_five_weeks_l962_96243

/-- Calculates the number of weeks worked given the total hours and daily hours. -/
def weeksWorked (totalHours : ℕ) (dailyHours : ℕ) : ℚ :=
  (totalHours : ℚ) / (dailyHours * 7 : ℚ)

/-- Theorem: Victoria worked for 5 weeks -/
theorem victoria_worked_five_weeks :
  weeksWorked 315 9 = 5 := by sorry

end NUMINAMATH_CALUDE_victoria_worked_five_weeks_l962_96243


namespace NUMINAMATH_CALUDE_punch_bowl_ratio_l962_96270

/-- Proves that the ratio of punch the cousin drank to the initial amount is 1:1 -/
theorem punch_bowl_ratio : 
  ∀ (initial_amount cousin_drink : ℚ),
  initial_amount > 0 →
  cousin_drink > 0 →
  initial_amount - cousin_drink + 4 - 2 + 12 = 16 →
  initial_amount + 14 = 16 →
  cousin_drink / initial_amount = 1 := by
sorry

end NUMINAMATH_CALUDE_punch_bowl_ratio_l962_96270


namespace NUMINAMATH_CALUDE_possible_numbers_correct_l962_96238

/-- Represents a digit on a seven-segment display -/
inductive Digit
| Zero | One | Two | Three | Four | Five | Six | Seven | Eight | Nine

/-- Represents a three-digit number -/
structure ThreeDigitNumber :=
(hundreds : Digit)
(tens : Digit)
(ones : Digit)

/-- Converts a ThreeDigitNumber to a natural number -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  match n.hundreds, n.tens, n.ones with
  | Digit.Three, Digit.Five, Digit.One => 351
  | Digit.Three, Digit.Five, Digit.Four => 354
  | Digit.Three, Digit.Five, Digit.Seven => 357
  | Digit.Three, Digit.Six, Digit.One => 361
  | Digit.Three, Digit.Six, Digit.Seven => 367
  | Digit.Three, Digit.Eight, Digit.One => 381
  | Digit.Three, Digit.Nine, Digit.One => 391
  | Digit.Three, Digit.Nine, Digit.Seven => 397
  | Digit.Eight, Digit.Five, Digit.One => 851
  | Digit.Nine, Digit.Five, Digit.One => 951
  | Digit.Nine, Digit.Five, Digit.Seven => 957
  | Digit.Nine, Digit.Six, Digit.One => 961
  | Digit.Nine, Digit.Nine, Digit.One => 991
  | _, _, _ => 0

/-- The set of all possible original numbers -/
def possibleNumbers : Set Nat :=
  {351, 354, 357, 361, 367, 381, 391, 397, 851, 951, 957, 961, 991}

/-- Function to check if a number can be displayed as 351 with two malfunctioning segments -/
def canBeDisplayedAs351WithTwoMalfunctions (n : ThreeDigitNumber) : Prop :=
  ∃ (seg1 seg2 : Nat), seg1 ≠ seg2 ∧ seg1 < 7 ∧ seg2 < 7 ∧
    (n.toNat ∈ possibleNumbers)

/-- Theorem stating that the set of possible numbers is correct -/
theorem possible_numbers_correct :
  ∀ n : ThreeDigitNumber, canBeDisplayedAs351WithTwoMalfunctions n ↔ n.toNat ∈ possibleNumbers :=
sorry

end NUMINAMATH_CALUDE_possible_numbers_correct_l962_96238


namespace NUMINAMATH_CALUDE_parallel_tangent_implies_a_le_one_l962_96229

open Real

/-- The function f(x) = ln x + (1/2)x^2 + ax has a tangent line parallel to 3x - y = 0 for some x > 0 -/
def has_parallel_tangent (a : ℝ) : Prop :=
  ∃ x > 0, (1 / x) + x + a = 3

/-- Theorem: If f(x) has a tangent line parallel to 3x - y = 0, then a ≤ 1 -/
theorem parallel_tangent_implies_a_le_one (a : ℝ) (h : has_parallel_tangent a) : a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_tangent_implies_a_le_one_l962_96229


namespace NUMINAMATH_CALUDE_equation_one_l962_96223

theorem equation_one (x : ℝ) : (3 - x)^2 + x^2 = 5 ↔ x = 1 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_equation_one_l962_96223


namespace NUMINAMATH_CALUDE_rectangle_width_25_l962_96285

/-- A rectangle with given area and perimeter -/
structure Rectangle where
  area : ℝ
  perimeter : ℝ

/-- The width of a rectangle -/
def width (r : Rectangle) : ℝ :=
  sorry

/-- The length of a rectangle -/
def length (r : Rectangle) : ℝ :=
  sorry

theorem rectangle_width_25 (r : Rectangle) 
  (h_area : r.area = 750)
  (h_perimeter : r.perimeter = 110) :
  width r = 25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_25_l962_96285


namespace NUMINAMATH_CALUDE_sum_a_b_equals_one_l962_96224

theorem sum_a_b_equals_one (a b : ℝ) (h : (a + 1)^2 + |b - 2| = 0) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_one_l962_96224


namespace NUMINAMATH_CALUDE_min_angle_function_l962_96275

/-- For any triangle with internal angles α, β, and γ in radians, 
    the minimum value of 4/α + 1/(β + γ) is 9/π. -/
theorem min_angle_function (α β γ : ℝ) (h1 : 0 < α) (h2 : 0 < β) (h3 : 0 < γ) 
    (h4 : α + β + γ = π) : 
  (∀ α' β' γ' : ℝ, 0 < α' ∧ 0 < β' ∧ 0 < γ' ∧ α' + β' + γ' = π → 
    4 / α + 1 / (β + γ) ≤ 4 / α' + 1 / (β' + γ')) → 
  4 / α + 1 / (β + γ) = 9 / π := by
sorry

end NUMINAMATH_CALUDE_min_angle_function_l962_96275


namespace NUMINAMATH_CALUDE_average_of_sequence_l962_96279

theorem average_of_sequence (z : ℝ) : (0 + 3*z + 6*z + 12*z + 24*z) / 5 = 9*z := by
  sorry

end NUMINAMATH_CALUDE_average_of_sequence_l962_96279


namespace NUMINAMATH_CALUDE_trapezium_marked_length_l962_96294

/-- Represents an isosceles triangle ABC with base AC and equal sides AB and BC -/
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ

/-- Represents a trapezium AMNC formed from an isosceles triangle ABC -/
structure Trapezium (triangle : IsoscelesTriangle) where
  markedLength : ℝ
  perimeter : ℝ

/-- Theorem: In an isosceles triangle with base 12 and side 18, 
    if a trapezium is formed with perimeter 40, 
    then the marked length on each side is 6 -/
theorem trapezium_marked_length 
  (triangle : IsoscelesTriangle) 
  (trap : Trapezium triangle) : 
  triangle.base = 12 → 
  triangle.side = 18 → 
  trap.perimeter = 40 → 
  trap.markedLength = 6 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_marked_length_l962_96294


namespace NUMINAMATH_CALUDE_fib_8_and_sum_2016_l962_96278

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Sum of first n terms of Fibonacci sequence -/
def fib_sum (n : ℕ) : ℕ :=
  (List.range n).map fib |>.sum

theorem fib_8_and_sum_2016 :
  fib 7 = 21 ∧
  ∀ m : ℕ, fib 2017 = m^2 + 1 → fib_sum 2016 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_fib_8_and_sum_2016_l962_96278


namespace NUMINAMATH_CALUDE_arccos_negative_one_equals_pi_l962_96230

theorem arccos_negative_one_equals_pi : Real.arccos (-1) = π := by
  sorry

end NUMINAMATH_CALUDE_arccos_negative_one_equals_pi_l962_96230


namespace NUMINAMATH_CALUDE_house_coloring_l962_96231

/-- A type representing the colors of houses -/
inductive Color
| Blue
| Green
| Red

/-- A function representing the move of residents between houses -/
def move (n : ℕ) : ℕ → ℕ :=
  sorry

/-- A function representing the coloring of houses -/
def color (n : ℕ) : ℕ → Color :=
  sorry

/-- The main theorem -/
theorem house_coloring (n : ℕ) (h_pos : 0 < n) :
  ∃ (move : ℕ → ℕ) (color : ℕ → Color),
    (∀ i : ℕ, i < n → move i < n) ∧  -- Each person moves to a valid house
    (∀ i j : ℕ, i < n → j < n → i ≠ j → move i ≠ move j) ∧  -- No two people move to the same house
    (∀ i : ℕ, i < n → move (move i) ≠ i) ∧  -- No person returns to their original house
    (∀ i : ℕ, i < n → color i ≠ color (move i)) :=  -- No person's new house has the same color as their old house
  sorry

#check house_coloring 1000

end NUMINAMATH_CALUDE_house_coloring_l962_96231


namespace NUMINAMATH_CALUDE_complex_expression_equals_25_1_l962_96206

theorem complex_expression_equals_25_1 :
  (50 + 5 * (12 / (180 / 3))^2) * Real.sin (30 * π / 180) = 25.1 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_25_1_l962_96206


namespace NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l962_96269

theorem sum_of_numbers_with_lcm_and_ratio (a b : ℕ+) : 
  Nat.lcm a b = 108 → 
  (a : ℚ) / (b : ℚ) = 3 / 7 → 
  (a : ℕ) + b = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l962_96269


namespace NUMINAMATH_CALUDE_exists_N_average_fifteen_l962_96260

theorem exists_N_average_fifteen : 
  ∃ N : ℝ, 15 < N ∧ N < 25 ∧ (8 + 14 + N) / 3 = 15 := by
sorry

end NUMINAMATH_CALUDE_exists_N_average_fifteen_l962_96260


namespace NUMINAMATH_CALUDE_sum_of_cubes_l962_96233

theorem sum_of_cubes (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a * b + a * c + b * c = 2) 
  (h3 : a * b * c = 5) : 
  a^3 + b^3 + c^3 = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l962_96233


namespace NUMINAMATH_CALUDE_pizza_bill_theorem_l962_96218

/-- The total bill amount for a group of people dividing equally -/
def total_bill (num_people : ℕ) (amount_per_person : ℕ) : ℕ :=
  num_people * amount_per_person

/-- Theorem: For a group of 5 people paying $8 each, the total bill is $40 -/
theorem pizza_bill_theorem :
  total_bill 5 8 = 40 := by
  sorry

end NUMINAMATH_CALUDE_pizza_bill_theorem_l962_96218


namespace NUMINAMATH_CALUDE_valid_combinations_count_l962_96202

def digits : List Nat := [1, 1, 2, 2, 3, 3, 3, 3]

def is_valid_price (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ 9999

def count_valid_combinations (digits : List Nat) : Nat :=
  sorry

theorem valid_combinations_count :
  count_valid_combinations digits = 14700 := by sorry

end NUMINAMATH_CALUDE_valid_combinations_count_l962_96202


namespace NUMINAMATH_CALUDE_lillian_candy_count_l962_96284

/-- The number of candies Lillian has after receiving candies from her father -/
def total_candies (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Lillian has 93 candies after receiving candies from her father -/
theorem lillian_candy_count :
  total_candies 88 5 = 93 := by
  sorry

end NUMINAMATH_CALUDE_lillian_candy_count_l962_96284


namespace NUMINAMATH_CALUDE_triangles_not_always_congruent_l962_96220

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_A : ℝ
  angle_B : ℝ
  angle_C : ℝ

-- Define the condition for the theorem
def satisfies_condition (t1 t2 : Triangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ 
   ((t1.a < t1.b ∧ t1.angle_A = t2.angle_A) ∨ 
    (t1.b < t1.a ∧ t1.angle_B = t2.angle_B)))

-- Define triangle congruence
def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c ∧
  t1.angle_A = t2.angle_A ∧ t1.angle_B = t2.angle_B ∧ t1.angle_C = t2.angle_C

-- Theorem statement
theorem triangles_not_always_congruent :
  ∃ (t1 t2 : Triangle), satisfies_condition t1 t2 ∧ ¬(congruent t1 t2) :=
sorry

end NUMINAMATH_CALUDE_triangles_not_always_congruent_l962_96220


namespace NUMINAMATH_CALUDE_carter_to_dog_height_ratio_l962_96259

-- Define the heights in inches
def dog_height : ℕ := 24
def betty_height_feet : ℕ := 3
def height_difference : ℕ := 12

-- Theorem to prove
theorem carter_to_dog_height_ratio :
  let betty_height_inches : ℕ := betty_height_feet * 12
  let carter_height : ℕ := betty_height_inches + height_difference
  carter_height / dog_height = 2 := by
sorry

end NUMINAMATH_CALUDE_carter_to_dog_height_ratio_l962_96259


namespace NUMINAMATH_CALUDE_amy_candy_difference_l962_96288

/-- Amy's candy problem -/
theorem amy_candy_difference (initial : ℕ) (given_away : ℕ) (left : ℕ) 
  (h1 : given_away = 6)
  (h2 : left = 5)
  (h3 : initial = given_away + left) :
  given_away - left = 1 := by
  sorry

end NUMINAMATH_CALUDE_amy_candy_difference_l962_96288


namespace NUMINAMATH_CALUDE_projection_symmetry_l962_96212

/-- For any non-right triangle with sides a, b, c, the equation a² = b² + c² + 2bc' 
    holds true regardless of which side is projected onto which, 
    where c' is the projection of c onto b. -/
theorem projection_symmetry (a b c : ℝ) (θ : ℝ) 
    (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
    (h_triangle : a < b + c ∧ b < a + c ∧ c < a + b)
    (h_not_right : θ ≠ π / 2)
    (h_angle : 0 < θ ∧ θ < π) : 
  a^2 = b^2 + c^2 + 2 * b * (c * Real.cos θ) ↔ 
  a^2 = c^2 + b^2 + 2 * c * (b * Real.cos θ) :=
sorry

end NUMINAMATH_CALUDE_projection_symmetry_l962_96212


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l962_96262

theorem quadratic_complete_square (a b c : ℝ) (h : 4 * a^2 - 8 * a - 320 = 0) :
  ∃ s : ℝ, s = 81 ∧ ∃ k : ℝ, (a - k)^2 = s :=
sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l962_96262


namespace NUMINAMATH_CALUDE_integer_solutions_system_l962_96297

theorem integer_solutions_system : 
  {(x, y, z) : ℤ × ℤ × ℤ | x + y - z = 6 ∧ x^3 + y^3 - z^3 = 414} = 
  {(3, 8, 5), (8, 3, 5), (3, -5, -8), (-5, 8, -3), (-5, 3, -8), (8, -5, -3)} :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_system_l962_96297


namespace NUMINAMATH_CALUDE_water_flow_speed_equation_l962_96271

/-- The speed of water flow in a river where two boats meet under specific conditions -/
def water_flow_speed : ℝ → Prop := λ V =>
  -- Speed of boat A in still water
  let speed_A : ℝ := 44
  -- Speed of boat B in still water
  let speed_B : ℝ := V^2
  -- Normal meeting time
  let normal_time : ℝ := 11
  -- Delayed meeting time
  let delayed_time : ℝ := 11.25
  -- Delay of boat B
  let delay : ℝ := 2/3
  -- Equation representing the scenario
  5 * V^2 - 8 * V - 132 = 0

theorem water_flow_speed_equation : ∃ V : ℝ, water_flow_speed V := by
  sorry

end NUMINAMATH_CALUDE_water_flow_speed_equation_l962_96271


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l962_96282

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) → (∃ r₁ r₂ : ℝ, (r₁ + r₂ = 6) ∧ (x = r₁ ∨ x = r₂)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l962_96282


namespace NUMINAMATH_CALUDE_perfect_square_condition_l962_96273

theorem perfect_square_condition (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 18*x + k = (a*x + b)^2) ↔ k = 81 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l962_96273


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l962_96207

/-- The eccentricity of a hyperbola with equation x²/4 - y²/5 = 1 is 3/2 -/
theorem hyperbola_eccentricity : ∃ (e : ℝ), e = 3/2 ∧ ∀ (x y : ℝ), x^2/4 - y^2/5 = 1 → 
  ∃ (a b c : ℝ), a^2 = 4 ∧ b^2 = 5 ∧ c^2 = a^2 + b^2 ∧ e = c/a := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l962_96207


namespace NUMINAMATH_CALUDE_mrs_hilt_bugs_l962_96295

theorem mrs_hilt_bugs (total_flowers : ℝ) (flowers_per_bug : ℝ) (h1 : total_flowers = 3.0) (h2 : flowers_per_bug = 1.5) :
  total_flowers / flowers_per_bug = 2 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_bugs_l962_96295


namespace NUMINAMATH_CALUDE_graces_pool_capacity_l962_96265

/-- Represents the capacity of Grace's pool in gallons -/
def C : ℝ := sorry

/-- Represents the unknown initial drain rate in gallons per hour -/
def x : ℝ := sorry

/-- The rate of the first hose in gallons per hour -/
def hose1_rate : ℝ := 50

/-- The rate of the second hose in gallons per hour -/
def hose2_rate : ℝ := 70

/-- The duration of the first filling period in hours -/
def time1 : ℝ := 3

/-- The duration of the second filling period in hours -/
def time2 : ℝ := 2

/-- The increase in drain rate during the second period in gallons per hour -/
def drain_rate_increase : ℝ := 10

theorem graces_pool_capacity :
  C = (hose1_rate - x) * time1 + (hose1_rate + hose2_rate - (x + drain_rate_increase)) * time2 ∧
  C = 390 - 5 * x := by sorry

end NUMINAMATH_CALUDE_graces_pool_capacity_l962_96265


namespace NUMINAMATH_CALUDE_twentyByFifteenGridToothpicks_l962_96210

/-- Represents a grid of toothpicks with alternating crossbars -/
structure ToothpickGrid where
  height : ℕ
  width : ℕ

/-- Calculates the total number of toothpicks used in the grid -/
def totalToothpicks (grid : ToothpickGrid) : ℕ :=
  let horizontalToothpicks := (grid.height + 1) * grid.width
  let verticalToothpicks := (grid.width + 1) * grid.height
  let totalSquares := grid.height * grid.width
  let crossbarToothpicks := (totalSquares / 2) * 2
  horizontalToothpicks + verticalToothpicks + crossbarToothpicks

/-- Theorem stating that a 20x15 grid uses 935 toothpicks -/
theorem twentyByFifteenGridToothpicks :
  totalToothpicks { height := 20, width := 15 } = 935 := by
  sorry


end NUMINAMATH_CALUDE_twentyByFifteenGridToothpicks_l962_96210


namespace NUMINAMATH_CALUDE_inequality_problem_l962_96216

theorem inequality_problem (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  (∀ x y z : ℝ, x < y ∧ y < z ∧ x * z < 0 → x * y^2 < x * z^2 → False) ∧
  (a * b > a * c) ∧
  (c * (b - a) > 0) ∧
  (a * c * (a - c) < 0) :=
sorry

end NUMINAMATH_CALUDE_inequality_problem_l962_96216


namespace NUMINAMATH_CALUDE_bisecting_line_theorem_l962_96242

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Calculates the area of a quadrilateral given its vertices -/
def quadrilateralArea (a b c d : Point) : ℝ := sorry

/-- Checks if two lines are parallel -/
def isParallel (l1 l2 : Line) : Prop := sorry

/-- Calculates the area of the part of the quadrilateral below a given line -/
def areaBelow (a b c d : Point) (l : Line) : ℝ := sorry

/-- The main theorem to be proved -/
theorem bisecting_line_theorem (a b c d : Point) (l : Line) : 
  a = Point.mk 0 0 →
  b = Point.mk 16 0 →
  c = Point.mk 8 8 →
  d = Point.mk 0 8 →
  l = Line.mk 1 (-4) →
  isParallel l (Line.mk 1 0) ∧ 
  areaBelow a b c d l = (quadrilateralArea a b c d) / 2 := by
  sorry

end NUMINAMATH_CALUDE_bisecting_line_theorem_l962_96242


namespace NUMINAMATH_CALUDE_expression_evaluation_l962_96217

theorem expression_evaluation : 5 + 7 * (2 - 9)^2 = 348 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l962_96217


namespace NUMINAMATH_CALUDE_equation_solutions_l962_96252

theorem equation_solutions :
  (∃ x : ℚ, 2 * x - 1 = 5 * x + 2 ↔ x = -1) ∧
  (∃ x : ℚ, (5 * x + 1) / 2 - (2 * x - 1) / 4 = 1 ↔ x = 1 / 8) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l962_96252


namespace NUMINAMATH_CALUDE_binomial_representation_existence_and_uniqueness_l962_96213

theorem binomial_representation_existence_and_uniqueness 
  (t l : ℕ) : 
  ∃! (m : ℕ) (a : ℕ → ℕ), 
    m ≤ l ∧ 
    (∀ i ∈ Finset.range (l - m + 1), a (m + i) ≥ m + i) ∧
    (∀ i ∈ Finset.range (l - m), a (m + i + 1) > a (m + i)) ∧
    t = (Finset.range (l - m + 1)).sum (λ i => Nat.choose (a (m + i)) (m + i)) :=
by sorry

end NUMINAMATH_CALUDE_binomial_representation_existence_and_uniqueness_l962_96213


namespace NUMINAMATH_CALUDE_calculate_expression_l962_96255

theorem calculate_expression : (-2 + 3) * 2 + (-2)^3 / 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l962_96255


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l962_96205

theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let sphere_diameter := outer_cube_edge
  let inner_cube_diagonal := sphere_diameter
  let inner_cube_edge := inner_cube_diagonal / Real.sqrt 3
  inner_cube_edge ^ 3 = 192 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l962_96205


namespace NUMINAMATH_CALUDE_average_speed_two_hours_car_average_speed_l962_96201

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) : 
  speed1 > 0 → speed2 > 0 → (speed1 + speed2) / 2 = (speed1 * 1 + speed2 * 1) / (1 + 1) := by
  sorry

/-- The average speed of a car traveling 90 km in the first hour and 30 km in the second hour is 60 km/h -/
theorem car_average_speed : 
  let speed1 : ℝ := 90
  let speed2 : ℝ := 30
  (speed1 + speed2) / 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_car_average_speed_l962_96201


namespace NUMINAMATH_CALUDE_function_equation_solution_l962_96240

theorem function_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f x * f y + f (x + y) = x * y) →
  (∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = -x - 1) :=
by sorry

end NUMINAMATH_CALUDE_function_equation_solution_l962_96240


namespace NUMINAMATH_CALUDE_new_player_weight_l962_96221

/-- Represents a basketball team --/
structure BasketballTeam where
  players : ℕ
  averageWeight : ℝ
  totalWeight : ℝ

/-- Calculates the total weight of a team --/
def totalWeight (team : BasketballTeam) : ℝ :=
  team.players * team.averageWeight

/-- Represents the change in team composition --/
structure TeamChange where
  oldTeam : BasketballTeam
  newTeam : BasketballTeam
  replacedWeight1 : ℝ
  replacedWeight2 : ℝ
  newPlayerWeight : ℝ

/-- Theorem stating the weight of the new player --/
theorem new_player_weight (change : TeamChange) 
  (h1 : change.oldTeam.players = 12)
  (h2 : change.oldTeam.averageWeight = 80)
  (h3 : change.newTeam.players = change.oldTeam.players)
  (h4 : change.newTeam.averageWeight = change.oldTeam.averageWeight + 2.5)
  (h5 : change.replacedWeight1 = 65)
  (h6 : change.replacedWeight2 = 75) :
  change.newPlayerWeight = 170 := by
  sorry

end NUMINAMATH_CALUDE_new_player_weight_l962_96221


namespace NUMINAMATH_CALUDE_inventory_difference_l962_96268

/-- Inventory problem -/
theorem inventory_difference (ties belts black_shirts white_shirts : ℕ) 
  (h_ties : ties = 34)
  (h_belts : belts = 40)
  (h_black_shirts : black_shirts = 63)
  (h_white_shirts : white_shirts = 42)
  : (2 * (black_shirts + white_shirts) / 3) - ((ties + belts) / 2) = 33 := by
  sorry

end NUMINAMATH_CALUDE_inventory_difference_l962_96268


namespace NUMINAMATH_CALUDE_complex_number_magnitude_squared_l962_96280

theorem complex_number_magnitude_squared (z : ℂ) : z + Complex.abs z = 2 + 8*I → Complex.abs z ^ 2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_squared_l962_96280


namespace NUMINAMATH_CALUDE_part_one_part_two_l962_96234

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 1

-- Part 1
theorem part_one (m n : ℝ) :
  (∀ x, f m x < 0 ↔ -2 < x ∧ x < n) →
  m = 5/2 ∧ n = 1/2 := by sorry

-- Part 2
theorem part_two (m : ℝ) :
  (∀ x ∈ Set.Icc m (m+1), f m x < 0) →
  m > -Real.sqrt 2 / 2 ∧ m < 0 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l962_96234


namespace NUMINAMATH_CALUDE_shell_collection_sum_l962_96214

/-- The sum of an arithmetic sequence with first term 2, common difference 3, and 15 terms -/
def shell_sum : ℕ := 
  let a₁ : ℕ := 2  -- first term
  let d : ℕ := 3   -- common difference
  let n : ℕ := 15  -- number of terms
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem stating that the sum of shells collected is 345 -/
theorem shell_collection_sum : shell_sum = 345 := by
  sorry

end NUMINAMATH_CALUDE_shell_collection_sum_l962_96214


namespace NUMINAMATH_CALUDE_x_in_terms_of_abc_l962_96277

theorem x_in_terms_of_abc (x y z a b c : ℝ) 
  (h1 : x * y / (x + y + 1) = a)
  (h2 : x * z / (x + z + 1) = b)
  (h3 : y * z / (y + z + 1) = c)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hab : a * b + a * c - b * c ≠ 0) :
  x = 2 * a * b * c / (a * b + a * c - b * c) :=
sorry

end NUMINAMATH_CALUDE_x_in_terms_of_abc_l962_96277


namespace NUMINAMATH_CALUDE_real_part_of_pure_imaginary_l962_96256

-- Define a pure imaginary number
def PureImaginary (z : ℂ) : Prop := ∃ b : ℝ, b ≠ 0 ∧ z = Complex.I * b

-- The theorem statement
theorem real_part_of_pure_imaginary (a : ℝ) (i : ℂ) (h : PureImaginary i) : a = 0 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_pure_imaginary_l962_96256


namespace NUMINAMATH_CALUDE_triangle_circle_square_sum_l962_96298

/-- Given a system of equations representing triangles, circles, and squares,
    prove that the sum of one triangle, two circles, and one square equals 35. -/
theorem triangle_circle_square_sum : 
  ∀ (x y z : ℝ),
  (2 * x + 3 * y + z = 45) →
  (x + 5 * y + 2 * z = 58) →
  (3 * x + y + 3 * z = 62) →
  (x + 2 * y + z = 35) := by
  sorry

end NUMINAMATH_CALUDE_triangle_circle_square_sum_l962_96298


namespace NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l962_96283

theorem quadratic_roots_to_coefficients :
  ∀ (b c : ℝ),
  (∀ x : ℝ, 2 * x^2 + b * x + c = 0 ↔ x = -1 ∨ x = 3) →
  b = -4 ∧ c = -6 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l962_96283


namespace NUMINAMATH_CALUDE_lucky_in_thirteen_l962_96241

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- A number is lucky if the sum of its digits is divisible by 7 -/
def is_lucky (n : ℕ) : Prop :=
  sum_of_digits n % 7 = 0

/-- Main theorem: Any sequence of 13 consecutive natural numbers contains a lucky number -/
theorem lucky_in_thirteen (start : ℕ) : ∃ k : ℕ, k ∈ Finset.range 13 ∧ is_lucky (start + k) := by
  sorry

end NUMINAMATH_CALUDE_lucky_in_thirteen_l962_96241


namespace NUMINAMATH_CALUDE_chords_intersection_theorem_l962_96203

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define a chord
structure Chord (c : Circle) where
  p1 : Point
  p2 : Point

-- Define the length of a segment
def segmentLength (p1 p2 : Point) : ℝ := sorry

-- Define a right angle
def isRightAngle (p1 p2 p3 : Point) : Prop := sorry

-- Theorem statement
theorem chords_intersection_theorem (c : Circle) (ab cd : Chord c) (e : Point) :
  isRightAngle ab.p1 e cd.p1 →
  (segmentLength ab.p1 e)^2 + (segmentLength ab.p2 e)^2 + 
  (segmentLength cd.p1 e)^2 + (segmentLength cd.p2 e)^2 = 
  (2 * c.radius)^2 := by
  sorry

end NUMINAMATH_CALUDE_chords_intersection_theorem_l962_96203


namespace NUMINAMATH_CALUDE_half_square_identity_l962_96215

theorem half_square_identity (a : ℤ) : (a + 1/2)^2 = a * (a + 1) + 1/4 := by
  sorry

end NUMINAMATH_CALUDE_half_square_identity_l962_96215


namespace NUMINAMATH_CALUDE_salt_solution_problem_l962_96296

theorem salt_solution_problem (initial_weight : ℝ) (added_salt : ℝ) (final_percentage : ℝ) :
  initial_weight = 60 →
  added_salt = 3 →
  final_percentage = 25 →
  let final_weight := initial_weight + added_salt
  let final_salt := (final_percentage / 100) * final_weight
  let initial_salt := final_salt - added_salt
  initial_salt / initial_weight * 100 = 21.25 := by
sorry

end NUMINAMATH_CALUDE_salt_solution_problem_l962_96296


namespace NUMINAMATH_CALUDE_sin_810_plus_cos_neg_60_l962_96254

theorem sin_810_plus_cos_neg_60 : 
  Real.sin (810 * π / 180) + Real.cos (-60 * π / 180) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_810_plus_cos_neg_60_l962_96254


namespace NUMINAMATH_CALUDE_range_of_a_l962_96246

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a + Real.cos (2 * x) < 5 - 4 * Real.sin x + Real.sqrt (5 * a - 4)) →
  (4/5 ≤ a ∧ a < 8) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l962_96246


namespace NUMINAMATH_CALUDE_min_guests_football_banquet_l962_96267

theorem min_guests_football_banquet (total_food : ℕ) (max_per_guest : ℕ) 
  (h1 : total_food = 325)
  (h2 : max_per_guest = 2) :
  (total_food + max_per_guest - 1) / max_per_guest = 163 := by
  sorry

end NUMINAMATH_CALUDE_min_guests_football_banquet_l962_96267


namespace NUMINAMATH_CALUDE_condition_2_not_implies_right_triangle_l962_96291

/-- A triangle ABC --/
structure Triangle :=
  (A B C : ℝ)
  (sum_angles : A + B + C = 180)

/-- Definition of a right triangle --/
def is_right_triangle (t : Triangle) : Prop :=
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

/-- The condition ∠A = ∠B - ∠C --/
def condition_2 (t : Triangle) : Prop :=
  t.A = t.B - t.C

/-- Theorem: The condition ∠A = ∠B - ∠C does not necessarily imply a right triangle --/
theorem condition_2_not_implies_right_triangle :
  ∃ t : Triangle, condition_2 t ∧ ¬is_right_triangle t :=
sorry

end NUMINAMATH_CALUDE_condition_2_not_implies_right_triangle_l962_96291


namespace NUMINAMATH_CALUDE_circle_op_calculation_l962_96287

-- Define the ⊗ operation
def circle_op (a b : ℚ) : ℚ := (a^2 + b^2) / (a + b)

-- State the theorem
theorem circle_op_calculation : circle_op (circle_op 5 2) 4 = 11375 / 2793 := by
  sorry

end NUMINAMATH_CALUDE_circle_op_calculation_l962_96287


namespace NUMINAMATH_CALUDE_bank_savings_exceed_target_l962_96272

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

def initial_deposit := 5
def daily_ratio := 2
def target_amount := 5000  -- 50 dollars in cents

theorem bank_savings_exceed_target :
  ∃ n : ℕ, 
    n = 10 ∧ 
    geometric_sum initial_deposit daily_ratio n ≥ target_amount ∧
    ∀ m : ℕ, m < n → geometric_sum initial_deposit daily_ratio m < target_amount :=
by sorry

end NUMINAMATH_CALUDE_bank_savings_exceed_target_l962_96272


namespace NUMINAMATH_CALUDE_power_product_simplification_l962_96227

theorem power_product_simplification :
  (-4/5 : ℚ)^2022 * (5/4 : ℚ)^2021 = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_simplification_l962_96227


namespace NUMINAMATH_CALUDE_max_digit_sum_l962_96292

theorem max_digit_sum (a b c : ℕ) (y : ℕ) : 
  a < 10 → b < 10 → c < 10 →  -- a, b, c are digits
  (a * 100 + b * 10 + c : ℚ) / 900 = 1 / y →  -- 0.abc = 1/y = abc/900
  0 < y → y ≤ 7 →  -- conditions on y
  a + b + c ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_digit_sum_l962_96292


namespace NUMINAMATH_CALUDE_line_point_k_value_l962_96222

/-- A line contains the points (8,10), (0,k), and (-8,3). This theorem proves that k = 13/2. -/
theorem line_point_k_value : 
  ∀ (k : ℚ), 
  (∃ (line : Set (ℚ × ℚ)), 
    (8, 10) ∈ line ∧ 
    (0, k) ∈ line ∧ 
    (-8, 3) ∈ line ∧ 
    (∀ (x y z : ℚ × ℚ), x ∈ line → y ∈ line → z ∈ line → 
      (x.2 - y.2) * (y.1 - z.1) = (y.2 - z.2) * (x.1 - y.1))) → 
  k = 13 / 2 := by
sorry

end NUMINAMATH_CALUDE_line_point_k_value_l962_96222


namespace NUMINAMATH_CALUDE_sum_reciprocal_squares_cubic_l962_96253

/-- Given a cubic equation x^3 - 12x^2 + 17x + 4 = 0 with real roots a, b, and c,
    prove that the sum of reciprocals of squares of roots equals 385/16 -/
theorem sum_reciprocal_squares_cubic (a b c : ℝ) : 
  a^3 - 12*a^2 + 17*a + 4 = 0 → 
  b^3 - 12*b^2 + 17*b + 4 = 0 → 
  c^3 - 12*c^2 + 17*c + 4 = 0 → 
  (1/a^2) + (1/b^2) + (1/c^2) = 385/16 :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocal_squares_cubic_l962_96253


namespace NUMINAMATH_CALUDE_opposite_of_2023_l962_96244

theorem opposite_of_2023 : -(2023 : ℤ) = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l962_96244


namespace NUMINAMATH_CALUDE_triangle_area_l962_96276

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its area is 9√3/14 when a = 3, b = 2c, and A = 2π/3 -/
theorem triangle_area (a b c : ℝ) (A : ℝ) :
  a = 3 →
  b = 2 * c →
  A = 2 * Real.pi / 3 →
  (1 / 2 : ℝ) * b * c * Real.sin A = 9 * Real.sqrt 3 / 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l962_96276


namespace NUMINAMATH_CALUDE_min_tiles_for_2014_area_l962_96299

/-- Represents the side length of a square tile in centimeters -/
inductive TileSize
  | Small : TileSize  -- 3 cm
  | Large : TileSize  -- 5 cm

/-- Calculates the area of a square tile given its size -/
def tileArea (size : TileSize) : ℕ :=
  match size with
  | TileSize.Small => 9   -- 3² = 9
  | TileSize.Large => 25  -- 5² = 25

/-- Represents a collection of tiles -/
structure TileCollection where
  smallCount : ℕ
  largeCount : ℕ

/-- Calculates the total area covered by a collection of tiles -/
def totalArea (tiles : TileCollection) : ℕ :=
  tiles.smallCount * tileArea TileSize.Small + tiles.largeCount * tileArea TileSize.Large

/-- Calculates the total number of tiles in a collection -/
def totalTiles (tiles : TileCollection) : ℕ :=
  tiles.smallCount + tiles.largeCount

theorem min_tiles_for_2014_area :
  ∃ (tiles : TileCollection),
    totalArea tiles = 2014 ∧
    (∀ (other : TileCollection), totalArea other = 2014 → totalTiles tiles ≤ totalTiles other) ∧
    totalTiles tiles = 94 :=
  sorry

end NUMINAMATH_CALUDE_min_tiles_for_2014_area_l962_96299


namespace NUMINAMATH_CALUDE_jims_weight_l962_96236

theorem jims_weight (jim steve stan : ℕ) 
  (h1 : stan = steve + 5)
  (h2 : steve = jim - 8)
  (h3 : jim + steve + stan = 319) :
  jim = 110 := by
sorry

end NUMINAMATH_CALUDE_jims_weight_l962_96236


namespace NUMINAMATH_CALUDE_sum_of_fractions_l962_96204

theorem sum_of_fractions : (2 : ℚ) / 20 + (4 : ℚ) / 40 + (5 : ℚ) / 50 = (3 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l962_96204


namespace NUMINAMATH_CALUDE_average_increase_is_eight_l962_96249

/-- Represents a cricketer's batting statistics -/
structure CricketerStats where
  initialInnings : Nat
  initialTotalRuns : Nat
  newInningScore : Nat
  newAverage : Nat

/-- Calculates the increase in average for a cricketer -/
def averageIncrease (stats : CricketerStats) : Nat :=
  stats.newAverage - (stats.initialTotalRuns / stats.initialInnings)

/-- Theorem: Given the specific conditions, the average increase is 8 runs -/
theorem average_increase_is_eight (stats : CricketerStats) 
  (h1 : stats.initialInnings = 9)
  (h2 : stats.newInningScore = 200)
  (h3 : stats.newAverage = 128)
  (h4 : stats.initialTotalRuns + stats.newInningScore = (stats.initialInnings + 1) * stats.newAverage) :
  averageIncrease stats = 8 := by
  sorry

#eval averageIncrease { initialInnings := 9, initialTotalRuns := 1080, newInningScore := 200, newAverage := 128 }

end NUMINAMATH_CALUDE_average_increase_is_eight_l962_96249


namespace NUMINAMATH_CALUDE_factorization_ax2_minus_4a_l962_96289

theorem factorization_ax2_minus_4a (a x : ℝ) : a * x^2 - 4 * a = a * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_ax2_minus_4a_l962_96289


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l962_96245

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l962_96245


namespace NUMINAMATH_CALUDE_lina_sticker_collection_l962_96239

/-- The sum of an arithmetic sequence with first term a, common difference d, and n terms -/
def arithmeticSequenceSum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Lina's sticker collection problem -/
theorem lina_sticker_collection :
  arithmeticSequenceSum 3 2 10 = 120 := by
  sorry

end NUMINAMATH_CALUDE_lina_sticker_collection_l962_96239


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l962_96251

def A : Set ℝ := {x | x^2 ≤ 1}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l962_96251


namespace NUMINAMATH_CALUDE_polynomial_evaluation_gcd_of_three_numbers_l962_96247

-- Problem 1: Polynomial evaluation
def f (x : ℝ) : ℝ := 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x - 6

theorem polynomial_evaluation : f 1 = 9 := by sorry

-- Problem 2: GCD of three numbers
theorem gcd_of_three_numbers : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_gcd_of_three_numbers_l962_96247


namespace NUMINAMATH_CALUDE_function_minimum_implies_inequality_l962_96208

open Real

/-- Given a function f(x) = ax^2 + bx - 2ln(x) where a > 0 and b is real,
    if f(x) ≥ f(2) for all x > 0, then ln(a) < -b - 1 -/
theorem function_minimum_implies_inequality (a b : ℝ) (ha : a > 0) :
  (∀ x > 0, a * x^2 + b * x - 2 * log x ≥ a * 2^2 + b * 2 - 2 * log 2) →
  log a < -b - 1 :=
by sorry

end NUMINAMATH_CALUDE_function_minimum_implies_inequality_l962_96208


namespace NUMINAMATH_CALUDE_arrangements_theorem_l962_96263

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 2

/-- The total number of students in the group -/
def total_students : ℕ := num_boys + num_girls

/-- Function to calculate the number of arrangements where girls are not next to each other -/
def arrangements_girls_not_adjacent : ℕ := sorry

/-- Function to calculate the number of arrangements with girl A not at left end and girl B not at right end -/
def arrangements_girl_A_B_restricted : ℕ := sorry

/-- Function to calculate the number of arrangements where all boys stand next to each other -/
def arrangements_boys_together : ℕ := sorry

/-- Function to calculate the number of arrangements where A, B, C stand in height order -/
def arrangements_ABC_height_order : ℕ := sorry

theorem arrangements_theorem :
  arrangements_girls_not_adjacent = 480 ∧
  arrangements_girl_A_B_restricted = 504 ∧
  arrangements_boys_together = 144 ∧
  arrangements_ABC_height_order = 120 := by sorry

end NUMINAMATH_CALUDE_arrangements_theorem_l962_96263


namespace NUMINAMATH_CALUDE_interior_angles_increase_l962_96258

/-- The sum of interior angles of a convex polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

theorem interior_angles_increase (n : ℕ) :
  sum_interior_angles n = 2340 → sum_interior_angles (n + 4) = 3060 := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_increase_l962_96258


namespace NUMINAMATH_CALUDE_panel_discussion_selection_l962_96286

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

theorem panel_discussion_selection (num_boys num_girls : ℕ) 
  (hb : num_boys = 5) (hg : num_girls = 4) : 
  -- I. Number of ways to select 2 boys and 2 girls
  (choose num_boys 2) * (choose num_girls 2) = 60 ∧ 
  -- II. Number of ways to select 4 people including at least one of boy A or girl B
  (choose (num_boys + num_girls) 4) - (choose (num_boys + num_girls - 2) 4) = 91 ∧
  -- III. Number of ways to select 4 people containing both boys and girls
  (choose (num_boys + num_girls) 4) - (choose num_boys 4) - (choose num_girls 4) = 120 :=
by sorry

end NUMINAMATH_CALUDE_panel_discussion_selection_l962_96286


namespace NUMINAMATH_CALUDE_function_value_2010_l962_96261

theorem function_value_2010 (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 1) = -f x) 
  (h2 : f 1 = 4) : 
  f 2010 = -4 := by sorry

end NUMINAMATH_CALUDE_function_value_2010_l962_96261
