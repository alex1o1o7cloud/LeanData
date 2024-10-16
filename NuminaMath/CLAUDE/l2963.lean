import Mathlib

namespace NUMINAMATH_CALUDE_monotonic_cubic_function_parameter_range_l2963_296333

/-- Given that f(x) = -x^3 + 2ax^2 - x - 3 is a monotonic function on ℝ, 
    prove that a ∈ [-√3/2, √3/2] -/
theorem monotonic_cubic_function_parameter_range (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => -x^3 + 2*a*x^2 - x - 3)) →
  a ∈ Set.Icc (-Real.sqrt 3 / 2) (Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_CALUDE_monotonic_cubic_function_parameter_range_l2963_296333


namespace NUMINAMATH_CALUDE_quadratic_no_solution_l2963_296386

theorem quadratic_no_solution (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * x - 1 ≠ 0) → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_solution_l2963_296386


namespace NUMINAMATH_CALUDE_systematic_sampling_selection_l2963_296319

theorem systematic_sampling_selection (total_rooms : Nat) (sample_size : Nat) (first_room : Nat) : 
  total_rooms = 64 → 
  sample_size = 8 → 
  first_room = 5 → 
  ∃ k : Nat, k < sample_size ∧ (first_room + k * (total_rooms / sample_size)) % total_rooms = 53 :=
by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_selection_l2963_296319


namespace NUMINAMATH_CALUDE_area_of_smaller_circle_l2963_296360

/-- Given two externally tangent circles with common tangent lines,
    where the tangent segment length is 6 and the radius of the larger circle
    is 3 times that of the smaller circle, prove that the area of the
    smaller circle is 12π/5. -/
theorem area_of_smaller_circle (r : ℝ) : 
  r > 0 →  -- radius of smaller circle is positive
  6^2 + r^2 = (4*r)^2 →  -- Pythagorean theorem applied to the tangent-radius triangle
  π * r^2 = 12*π/5 :=
by sorry

end NUMINAMATH_CALUDE_area_of_smaller_circle_l2963_296360


namespace NUMINAMATH_CALUDE_regression_lines_intersection_l2963_296308

-- Define the regression line type
def RegressionLine := ℝ → ℝ

-- Define the property that a regression line passes through a point
def passes_through (l : RegressionLine) (p : ℝ × ℝ) : Prop :=
  l p.1 = p.2

-- Theorem statement
theorem regression_lines_intersection
  (l₁ l₂ : RegressionLine)
  (s t : ℝ)
  (h₁ : passes_through l₁ (s, t))
  (h₂ : passes_through l₂ (s, t)) :
  ∃ p : ℝ × ℝ, p = (s, t) ∧ passes_through l₁ p ∧ passes_through l₂ p :=
sorry

end NUMINAMATH_CALUDE_regression_lines_intersection_l2963_296308


namespace NUMINAMATH_CALUDE_set_equalities_l2963_296378

-- Define the sets A, B, and C
def A : Set ℝ := {x | -3 < x ∧ x < 1}
def B : Set ℝ := {x | x ≤ -1}
def C : Set ℝ := {x | -2 < x ∧ x ≤ 2}

-- Theorem stating the equalities
theorem set_equalities :
  (A = A ∩ (B ∪ C)) ∧
  (A = A ∪ (B ∩ C)) ∧
  (A = (A ∩ B) ∪ (A ∩ C)) := by
  sorry

end NUMINAMATH_CALUDE_set_equalities_l2963_296378


namespace NUMINAMATH_CALUDE_angle_bisector_implies_line_equation_l2963_296382

/-- Given points A and B, and a line representing the angle bisector of ∠ACB,
    prove that the line AC has the equation x - 2y - 1 = 0 -/
theorem angle_bisector_implies_line_equation 
  (A B : ℝ × ℝ)
  (angle_bisector : ℝ → ℝ)
  (h1 : A = (3, 1))
  (h2 : B = (-1, 2))
  (h3 : ∀ x y, y = angle_bisector x ↔ y = x + 1)
  (h4 : ∃ C : ℝ × ℝ, (angle_bisector (C.1) = C.2) ∧ 
       (∃ t : ℝ, C = (1 - t) • A + t • B) ∧
       (∃ s : ℝ, C = (1 - s) • A' + s • B)) 
  (A' : ℝ × ℝ)
  (h5 : A'.2 - 1 = -(A'.1 - 3))  -- Reflection condition
  (h6 : (A'.2 + 1) / 2 = (A'.1 + 3) / 2 + 1)  -- Reflection condition
  : ∀ x y, x - 2*y - 1 = 0 ↔ ∃ t : ℝ, (x, y) = (1 - t) • A + t • C :=
sorry


end NUMINAMATH_CALUDE_angle_bisector_implies_line_equation_l2963_296382


namespace NUMINAMATH_CALUDE_unique_five_digit_number_l2963_296384

def is_valid_number (n : ℕ) : Prop :=
  ∃ (x y : ℕ),
    n = 10 * x + y ∧
    0 ≤ y ∧ y ≤ 9 ∧
    10000 ≤ n ∧ n ≤ 99999 ∧
    1000 ≤ x ∧ x ≤ 9999 ∧
    n - x = 54321

theorem unique_five_digit_number : 
  ∃! (n : ℕ), is_valid_number n ∧ n = 60356 :=
sorry

end NUMINAMATH_CALUDE_unique_five_digit_number_l2963_296384


namespace NUMINAMATH_CALUDE_max_median_value_l2963_296310

theorem max_median_value (k m r s t : ℕ) : 
  k > 0 → m > 0 → r > 0 → s > 0 → t > 0 →
  (k + m + r + s + t) / 5 = 10 →
  k < m → m < r → r < s → s < t →
  t = 20 →
  r ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_max_median_value_l2963_296310


namespace NUMINAMATH_CALUDE_expected_value_of_heads_l2963_296340

/-- Represents the different types of coins -/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- Returns the value of a coin in cents -/
def coinValue (c : Coin) : ℚ :=
  match c with
  | .Penny => 1
  | .Nickel => 5
  | .Dime => 10
  | .Quarter => 25
  | .HalfDollar => 50

/-- Returns the probability of a coin landing heads -/
def headsProbability (c : Coin) : ℚ :=
  match c with
  | .HalfDollar => 1/3
  | _ => 1/2

/-- The set of all coins -/
def coinSet : List Coin := [Coin.Penny, Coin.Nickel, Coin.Dime, Coin.Quarter, Coin.HalfDollar]

/-- Calculates the expected value for a single coin -/
def expectedValue (c : Coin) : ℚ := (headsProbability c) * (coinValue c)

/-- Theorem: The expected value of the amount of money from coins that come up heads is 223/6 cents -/
theorem expected_value_of_heads : 
  (coinSet.map expectedValue).sum = 223/6 := by sorry

end NUMINAMATH_CALUDE_expected_value_of_heads_l2963_296340


namespace NUMINAMATH_CALUDE_triangle_ratio_sqrt_two_l2963_296348

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a * sin(A) * sin(B) + b * cos²(A) = √2 * a, then b/a = √2 -/
theorem triangle_ratio_sqrt_two (a b c : ℝ) (A B C : ℝ) 
    (h : a * Real.sin A * Real.sin B + b * Real.cos A ^ 2 = Real.sqrt 2 * a) 
    (h_positive : a > 0) : b / a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_sqrt_two_l2963_296348


namespace NUMINAMATH_CALUDE_ellipse_equation_and_max_slope_product_l2963_296393

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a line in the form x = my + 1 -/
structure Line where
  m : ℝ

/-- Calculates the product of slopes of the three sides of a triangle formed by
    two points on an ellipse and a fixed point -/
def slope_product (e : Ellipse) (l : Line) (p : ℝ × ℝ) : ℝ :=
  sorry

theorem ellipse_equation_and_max_slope_product 
  (e : Ellipse) (p : ℝ × ℝ) (h_p_on_ellipse : p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1)
  (h_p : p = (1, 3/2)) (h_eccentricity : Real.sqrt (e.a^2 - e.b^2) / e.a = 1/2) :
  (∃ (e' : Ellipse), e'.a = 2 ∧ e'.b = Real.sqrt 3 ∧
    (∀ (x y : ℝ), x^2 / 4 + y^2 / 3 = 1 ↔ x^2 / e'.a^2 + y^2 / e'.b^2 = 1)) ∧
  (∃ (max_t : ℝ), max_t = 9/64 ∧
    ∀ (l : Line), slope_product e' l p ≤ max_t) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_and_max_slope_product_l2963_296393


namespace NUMINAMATH_CALUDE_sum_of_roots_lower_bound_l2963_296390

theorem sum_of_roots_lower_bound (k : ℝ) (α β : ℝ) : 
  (∃ x : ℝ, x^2 - 2*(1-k)*x + k^2 = 0) →
  (α^2 - 2*(1-k)*α + k^2 = 0) →
  (β^2 - 2*(1-k)*β + k^2 = 0) →
  α + β ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_lower_bound_l2963_296390


namespace NUMINAMATH_CALUDE_set_equality_l2963_296399

open Set

-- Define the universal set U as the set of real numbers
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | x < 1}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem statement
theorem set_equality : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by sorry

end NUMINAMATH_CALUDE_set_equality_l2963_296399


namespace NUMINAMATH_CALUDE_max_volume_at_10cm_l2963_296335

/-- The length of the original sheet in centimeters -/
def sheet_length : ℝ := 90

/-- The width of the original sheet in centimeters -/
def sheet_width : ℝ := 48

/-- The side length of the cut-out squares in centimeters -/
def cut_length : ℝ := 10

/-- The volume of the container as a function of the cut length -/
def container_volume (x : ℝ) : ℝ := (sheet_length - 2*x) * (sheet_width - 2*x) * x

theorem max_volume_at_10cm :
  ∀ x, 0 < x → x < sheet_width/2 → x < sheet_length/2 →
  container_volume x ≤ container_volume cut_length :=
sorry

end NUMINAMATH_CALUDE_max_volume_at_10cm_l2963_296335


namespace NUMINAMATH_CALUDE_difference_between_squares_l2963_296303

theorem difference_between_squares : (50 : ℕ)^2 - (49 : ℕ)^2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_squares_l2963_296303


namespace NUMINAMATH_CALUDE_sqrt_500_simplification_l2963_296375

theorem sqrt_500_simplification : Real.sqrt 500 = 10 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_500_simplification_l2963_296375


namespace NUMINAMATH_CALUDE_solution_set_when_m_is_neg_three_m_range_when_subset_condition_holds_l2963_296343

-- Define the function f
def f (x m : ℝ) : ℝ := |x + 1| + |2*x + m|

-- Part I
theorem solution_set_when_m_is_neg_three :
  {x : ℝ | f x (-3) ≤ 6} = {x : ℝ | -4/3 ≤ x ∧ x ≤ 8/3} := by sorry

-- Part II
theorem m_range_when_subset_condition_holds :
  (∀ x ∈ Set.Icc (-1 : ℝ) (1/2 : ℝ), f x m ≤ |2*x - 4|) →
  m ∈ Set.Icc (-5/2 : ℝ) (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_is_neg_three_m_range_when_subset_condition_holds_l2963_296343


namespace NUMINAMATH_CALUDE_tileable_rectangle_divisibility_l2963_296389

/-- A rectangle is (a, b)-tileable if it can be covered by non-overlapping a × b tiles -/
def is_tileable (m n a b : ℕ) : Prop := sorry

/-- Main theorem: If k divides a and b, and an m × n rectangle is (a, b)-tileable, 
    then 2k divides m or 2k divides n -/
theorem tileable_rectangle_divisibility 
  (k a b m n : ℕ) 
  (h1 : k ∣ a) 
  (h2 : k ∣ b) 
  (h3 : is_tileable m n a b) : 
  (2 * k) ∣ m ∨ (2 * k) ∣ n :=
sorry

end NUMINAMATH_CALUDE_tileable_rectangle_divisibility_l2963_296389


namespace NUMINAMATH_CALUDE_raised_beds_planks_l2963_296325

/-- Calculates the number of planks needed for raised beds -/
def planks_needed (num_beds : ℕ) (bed_height : ℕ) (bed_width : ℕ) (bed_length : ℕ) (plank_width : ℕ) : ℕ :=
  let long_sides := 2 * bed_height
  let short_sides := 2 * bed_height
  let planks_per_bed := long_sides + short_sides
  num_beds * planks_per_bed

/-- Proves that 60 planks are needed for 10 raised beds with given dimensions -/
theorem raised_beds_planks :
  planks_needed 10 2 2 8 1 = 60 := by
  sorry

end NUMINAMATH_CALUDE_raised_beds_planks_l2963_296325


namespace NUMINAMATH_CALUDE_megan_carrots_count_l2963_296317

/-- Calculates the total number of carrots Megan has after picking, throwing out, and picking again. -/
def total_carrots (initial : ℕ) (thrown_out : ℕ) (new_picked : ℕ) : ℕ :=
  initial - thrown_out + new_picked

/-- Proves that Megan's total carrots is 61 given the specific numbers in the problem. -/
theorem megan_carrots_count :
  total_carrots 19 4 46 = 61 := by
  sorry

end NUMINAMATH_CALUDE_megan_carrots_count_l2963_296317


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l2963_296350

theorem difference_of_squares_special_case : (723 : ℤ) * 723 - 722 * 724 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l2963_296350


namespace NUMINAMATH_CALUDE_correct_product_is_5810_l2963_296381

/-- Reverses the digits of a three-digit number -/
def reverse_digits (n : Nat) : Nat :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

/-- Checks if a number is three-digit -/
def is_three_digit (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem correct_product_is_5810 (a b : Nat) :
  a > 0 ∧ b > 0 ∧ is_three_digit a ∧ (reverse_digits a - 3) * b = 245 →
  a * b = 5810 := by
  sorry

end NUMINAMATH_CALUDE_correct_product_is_5810_l2963_296381


namespace NUMINAMATH_CALUDE_circle_m_range_l2963_296322

-- Define the equation of the circle
def circle_equation (x y m : ℝ) : Prop := x^2 + y^2 - x + y + m = 0

-- Define the condition for the equation to represent a circle
def is_circle (m : ℝ) : Prop := ∃ (x y : ℝ), circle_equation x y m

-- Theorem stating the range of m for which the equation represents a circle
theorem circle_m_range :
  ∀ m : ℝ, is_circle m ↔ m < (1/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_circle_m_range_l2963_296322


namespace NUMINAMATH_CALUDE_quadratic_polynomial_property_l2963_296300

/-- A quadratic polynomial -/
structure QuadraticPolynomial where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Evaluation of a quadratic polynomial at a point -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℚ) : ℚ :=
  p.a * x^2 + p.b * x + p.c

/-- Condition for divisibility by (x - 2)(x + 2)(x - 9) -/
def isDivisibleByFactors (p : QuadraticPolynomial) : Prop :=
  (p.eval 2)^3 = 2 ∧ (p.eval (-2))^3 = -2 ∧ (p.eval 9)^3 = 9

theorem quadratic_polynomial_property (p : QuadraticPolynomial) 
  (h : isDivisibleByFactors p) : p.eval 14 = -230/11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_property_l2963_296300


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l2963_296342

-- First expression
theorem factorization_1 (a b : ℝ) :
  -6 * a * b + 3 * a^2 + 3 * b^2 = 3 * (a - b)^2 := by sorry

-- Second expression
theorem factorization_2 (x y m : ℝ) :
  y^2 * (2 - m) + x^2 * (m - 2) = (m - 2) * (x + y) * (x - y) := by sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l2963_296342


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l2963_296327

-- Define the sets A and B
def A (p : ℝ) : Set ℝ := {x : ℝ | x^2 - p*x + 15 = 0}
def B (q : ℝ) : Set ℝ := {x : ℝ | x^2 - 5*x + q = 0}

-- State the theorem
theorem intersection_implies_sum (p q : ℝ) : A p ∩ B q = {3} → p + q = 14 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l2963_296327


namespace NUMINAMATH_CALUDE_permutations_not_divisible_by_three_l2963_296321

/-- The number of permutations of 1 to n where 1 is fixed and each number differs from its neighbors by at most 2 -/
def p (n : ℕ) : ℕ :=
  if n ≤ 2 then 1
  else if n = 3 then 2
  else p (n - 1) + p (n - 3) + 1

/-- The theorem stating that the number of permutations for 1996 is not divisible by 3 -/
theorem permutations_not_divisible_by_three :
  ¬ (3 ∣ p 1996) :=
sorry

end NUMINAMATH_CALUDE_permutations_not_divisible_by_three_l2963_296321


namespace NUMINAMATH_CALUDE_apple_harvest_l2963_296392

theorem apple_harvest (apples peaches : ℕ) 
  (h1 : peaches = 3 * apples) 
  (h2 : peaches - apples = 120) : 
  apples = 60 := by
sorry

end NUMINAMATH_CALUDE_apple_harvest_l2963_296392


namespace NUMINAMATH_CALUDE_files_remaining_l2963_296302

theorem files_remaining (music_files video_files deleted_files : ℕ) :
  music_files = 4 →
  video_files = 21 →
  deleted_files = 23 →
  music_files + video_files - deleted_files = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_files_remaining_l2963_296302


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l2963_296372

theorem quadratic_equation_solutions : {x : ℝ | x^2 = x} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l2963_296372


namespace NUMINAMATH_CALUDE_three_digit_number_proof_l2963_296330

theorem three_digit_number_proof (a : Nat) (h1 : a < 10) : 
  (100 * a + 10 * a + 5) % 9 = 8 → 100 * a + 10 * a + 5 = 665 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_proof_l2963_296330


namespace NUMINAMATH_CALUDE_range_of_a_l2963_296306

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 5*a| + |2*x + 1|
def g (x : ℝ) : ℝ := |x - 1| + 3

theorem range_of_a (a : ℝ) :
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) →
  a ≥ 0.4 ∨ a ≤ -0.8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2963_296306


namespace NUMINAMATH_CALUDE_circle_radius_is_three_l2963_296398

/-- Given a circle where the product of three inches and its circumference
    is twice its area, prove that its radius is 3 inches. -/
theorem circle_radius_is_three (r : ℝ) (h : 3 * (2 * π * r) = 2 * (π * r^2)) : r = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_three_l2963_296398


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_l2963_296347

/-- Perimeter of a quadrilateral EFGH with specific properties -/
theorem quadrilateral_perimeter (EF HG FG : ℝ) (h1 : EF = 15) (h2 : HG = 6) (h3 : FG = 20) :
  ∃ (EH : ℝ), EF + FG + HG + EH = 41 + Real.sqrt 481 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_l2963_296347


namespace NUMINAMATH_CALUDE_shipment_box_count_l2963_296356

theorem shipment_box_count :
  ∀ (x y : ℕ),
  (10 * x + 20 * y) / (x + y) = 18 →
  (10 * x + 20 * (y - 15)) / (x + y - 15) = 16 →
  x + y = 30 := by
sorry

end NUMINAMATH_CALUDE_shipment_box_count_l2963_296356


namespace NUMINAMATH_CALUDE_first_half_speed_l2963_296374

theorem first_half_speed
  (total_distance : ℝ)
  (total_time : ℝ)
  (second_half_speed : ℝ)
  (h1 : total_distance = 300)
  (h2 : total_time = 11)
  (h3 : second_half_speed = 25)
  : ∃ (first_half_speed : ℝ),
    first_half_speed = 30 ∧
    total_distance / 2 / first_half_speed +
    total_distance / 2 / second_half_speed = total_time :=
by sorry

end NUMINAMATH_CALUDE_first_half_speed_l2963_296374


namespace NUMINAMATH_CALUDE_problem_solution_l2963_296307

theorem problem_solution (m n : ℝ) (h1 : m + 1/m = -4) (h2 : n + 1/n = -4) (h3 : m ≠ n) : 
  m * (n + 1) + n = -3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2963_296307


namespace NUMINAMATH_CALUDE_intersection_point_sum_squares_l2963_296362

-- Define the lines
def line1 (x y : ℝ) : Prop := 323 * x + 457 * y = 1103
def line2 (x y : ℝ) : Prop := 177 * x + 543 * y = 897

-- Define the intersection point
def intersection_point (a b : ℝ) : Prop := line1 a b ∧ line2 a b

-- Theorem statement
theorem intersection_point_sum_squares :
  ∀ a b : ℝ, intersection_point a b → a^2 + 2004 * b^2 = 2008 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_sum_squares_l2963_296362


namespace NUMINAMATH_CALUDE_sum_of_digits_0_to_2012_l2963_296332

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits of all numbers in a range -/
def sumOfDigitsInRange (start finish : ℕ) : ℕ :=
  (List.range (finish - start + 1)).map (fun i => sumOfDigits (start + i))
    |> List.sum

/-- The sum of the digits of all numbers from 0 to 2012 is 28077 -/
theorem sum_of_digits_0_to_2012 :
    sumOfDigitsInRange 0 2012 = 28077 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_0_to_2012_l2963_296332


namespace NUMINAMATH_CALUDE_quadratic_negative_root_l2963_296380

/-- The quadratic equation ax^2 + 2x + 1 = 0 has at least one negative root
    if and only if a < 0 or 0 < a ≤ 1 -/
theorem quadratic_negative_root (a : ℝ) (ha : a ≠ 0) :
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0) ↔ (a < 0 ∨ (0 < a ∧ a ≤ 1)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_negative_root_l2963_296380


namespace NUMINAMATH_CALUDE_line_equation_proof_l2963_296334

def P : ℝ × ℝ := (1, 2)
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (4, -5)

def Line (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

def DistancePointToLine (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ :=
  sorry -- Definition of distance from a point to a line

theorem line_equation_proof :
  ∃ (l : Set (ℝ × ℝ)),
    P ∈ l ∧
    DistancePointToLine A l = DistancePointToLine B l ∧
    (l = Line 4 1 (-6) ∨ l = Line 3 2 (-7)) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2963_296334


namespace NUMINAMATH_CALUDE_square_sum_equals_sixteen_l2963_296305

theorem square_sum_equals_sixteen (a b : ℝ) (h : a + b = 4) : a^2 + 2*a*b + b^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_sixteen_l2963_296305


namespace NUMINAMATH_CALUDE_multiplication_proof_l2963_296349

theorem multiplication_proof (m : ℕ) : m = 32505 → m * 9999 = 325027405 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_proof_l2963_296349


namespace NUMINAMATH_CALUDE_circle_C_equation_l2963_296364

-- Define the circles
def circle_C (r : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = r^2}
def circle_other : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 - 3*p.1 = 0}

-- Define the line passing through (5, -2)
def common_chord_line (r : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + 2*p.2 - 5 + r^2 = 0}

-- Theorem statement
theorem circle_C_equation :
  ∃ (r : ℝ), 
    (∀ (x y : ℝ), (x, y) ∈ circle_C r ∩ circle_other → (x, y) ∈ common_chord_line r) ∧
    (5, -2) ∈ common_chord_line r →
    r = 2 :=
sorry

end NUMINAMATH_CALUDE_circle_C_equation_l2963_296364


namespace NUMINAMATH_CALUDE_ice_cream_arrangement_count_l2963_296301

theorem ice_cream_arrangement_count : Nat.factorial 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_arrangement_count_l2963_296301


namespace NUMINAMATH_CALUDE_value_of_a_l2963_296371

theorem value_of_a (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a^2 / b = 1) (h2 : b^2 / c = 4) (h3 : c^2 / a = 4) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2963_296371


namespace NUMINAMATH_CALUDE_other_number_proof_l2963_296355

/-- Prove that given two positive integers, 24 and x, if their HCF (h) is 17 and their LCM (l) is 312, then x = 221. -/
theorem other_number_proof (x : ℕ) (h l : ℕ) : 
  x > 0 ∧ h > 0 ∧ l > 0 ∧ 
  h = Nat.gcd 24 x ∧ 
  l = Nat.lcm 24 x ∧ 
  h = 17 ∧ 
  l = 312 → 
  x = 221 := by
sorry

end NUMINAMATH_CALUDE_other_number_proof_l2963_296355


namespace NUMINAMATH_CALUDE_chord_length_l2963_296320

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by sorry

end NUMINAMATH_CALUDE_chord_length_l2963_296320


namespace NUMINAMATH_CALUDE_surface_area_unchanged_l2963_296367

/-- Represents the dimensions of a rectangular solid -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a cube -/
structure Cube where
  side : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (r : RectangularSolid) : ℝ :=
  2 * (r.length * r.width + r.length * r.height + r.width * r.height)

/-- Calculates the exposed surface area of a cube when it touches two faces of the solid -/
def exposedCubeArea (c : Cube) : ℝ :=
  2 * c.side * c.side

/-- Theorem: The surface area remains unchanged after cube removal -/
theorem surface_area_unchanged 
  (original : RectangularSolid)
  (removed : Cube)
  (h1 : original.length = 5)
  (h2 : original.width = 3)
  (h3 : original.height = 4)
  (h4 : removed.side = 2)
  (h5 : exposedCubeArea removed = exposedCubeArea removed) :
  surfaceArea original = surfaceArea original :=
by sorry

end NUMINAMATH_CALUDE_surface_area_unchanged_l2963_296367


namespace NUMINAMATH_CALUDE_prob_male_monday_female_tuesday_is_one_third_l2963_296359

/-- Represents the number of male volunteers -/
def num_men : ℕ := 2

/-- Represents the number of female volunteers -/
def num_women : ℕ := 2

/-- Represents the total number of volunteers -/
def total_volunteers : ℕ := num_men + num_women

/-- Represents the number of days for which volunteers are selected -/
def num_days : ℕ := 2

/-- Calculates the probability of selecting a male volunteer for Monday
    and a female volunteer for Tuesday -/
def prob_male_monday_female_tuesday : ℚ :=
  (num_men * num_women) / (total_volunteers * (total_volunteers - 1))

/-- Proves that the probability of selecting a male volunteer for Monday
    and a female volunteer for Tuesday is 1/3 -/
theorem prob_male_monday_female_tuesday_is_one_third :
  prob_male_monday_female_tuesday = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_prob_male_monday_female_tuesday_is_one_third_l2963_296359


namespace NUMINAMATH_CALUDE_three_fourths_to_fifth_power_l2963_296354

theorem three_fourths_to_fifth_power :
  (3 / 4 : ℚ) ^ 5 = 243 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_three_fourths_to_fifth_power_l2963_296354


namespace NUMINAMATH_CALUDE_no_real_solutions_l2963_296344

theorem no_real_solutions : ¬∃ y : ℝ, (y - 4*y + 10)^2 + 4 = -2*abs y := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2963_296344


namespace NUMINAMATH_CALUDE_old_card_sale_amount_l2963_296312

def initial_cost : ℕ := 1200
def new_card_cost : ℕ := 500
def total_spent : ℕ := 1400

theorem old_card_sale_amount : 
  initial_cost + new_card_cost - total_spent = 300 :=
by sorry

end NUMINAMATH_CALUDE_old_card_sale_amount_l2963_296312


namespace NUMINAMATH_CALUDE_emily_minimum_grade_to_beat_ahmed_l2963_296366

/-- Represents a student's grade -/
structure StudentGrade where
  current_grade : ℕ
  final_grade : ℕ

/-- Calculates the final average grade given current grade and final assignment grade -/
def finalAverageGrade (s : StudentGrade) : ℚ :=
  (9 * s.current_grade + s.final_grade) / 10

theorem emily_minimum_grade_to_beat_ahmed :
  ∀ (ahmed emily : StudentGrade),
    ahmed.current_grade = 91 →
    emily.current_grade = 92 →
    ahmed.final_grade = 100 →
    (∀ g : ℕ, g < 92 → finalAverageGrade emily < finalAverageGrade ahmed) ∧
    finalAverageGrade { current_grade := 92, final_grade := 92 } > finalAverageGrade ahmed :=
by sorry

end NUMINAMATH_CALUDE_emily_minimum_grade_to_beat_ahmed_l2963_296366


namespace NUMINAMATH_CALUDE_book_pair_count_l2963_296370

theorem book_pair_count :
  let num_genres : ℕ := 4
  let books_per_genre : ℕ := 4
  let choose_genres : ℕ := 2
  num_genres.choose choose_genres * books_per_genre^choose_genres = 96 :=
by sorry

end NUMINAMATH_CALUDE_book_pair_count_l2963_296370


namespace NUMINAMATH_CALUDE_max_attendance_l2963_296363

-- Define the days of the week
inductive Day
| monday
| tuesday
| wednesday
| thursday
| friday

-- Define the team members
inductive Member
| alice
| bob
| charlie
| diana
| edward

-- Define a function that returns whether a member is available on a given day
def isAvailable (m : Member) (d : Day) : Bool :=
  match m, d with
  | Member.alice, Day.monday => false
  | Member.alice, Day.thursday => false
  | Member.bob, Day.tuesday => false
  | Member.bob, Day.friday => false
  | Member.charlie, Day.monday => false
  | Member.charlie, Day.tuesday => false
  | Member.charlie, Day.thursday => false
  | Member.charlie, Day.friday => false
  | Member.diana, Day.wednesday => false
  | Member.diana, Day.thursday => false
  | Member.edward, Day.wednesday => false
  | _, _ => true

-- Define a function that counts the number of available members on a given day
def countAvailable (d : Day) : Nat :=
  List.length (List.filter (λ m => isAvailable m d) [Member.alice, Member.bob, Member.charlie, Member.diana, Member.edward])

-- State the theorem
theorem max_attendance :
  (∀ d : Day, countAvailable d ≤ 3) ∧
  (countAvailable Day.monday = 3) ∧
  (countAvailable Day.wednesday = 3) ∧
  (countAvailable Day.friday = 3) :=
sorry

end NUMINAMATH_CALUDE_max_attendance_l2963_296363


namespace NUMINAMATH_CALUDE_monochromatic_triangle_exists_l2963_296376

/-- A complete graph with 6 vertices where each edge is colored either black or red -/
def ColoredGraph6 := Fin 6 → Fin 6 → Bool

/-- A triangle in the graph is represented by three distinct vertices -/
def Triangle (G : ColoredGraph6) (a b c : Fin 6) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- A monochromatic triangle has all edges of the same color -/
def MonochromaticTriangle (G : ColoredGraph6) (a b c : Fin 6) : Prop :=
  Triangle G a b c ∧
  ((G a b = G b c ∧ G b c = G a c) ∨
   (G a b ≠ G b c ∧ G b c ≠ G a c ∧ G a c ≠ G a b))

/-- The main theorem: every 2-coloring of K6 contains a monochromatic triangle -/
theorem monochromatic_triangle_exists (G : ColoredGraph6) :
  ∃ (a b c : Fin 6), MonochromaticTriangle G a b c := by
  sorry


end NUMINAMATH_CALUDE_monochromatic_triangle_exists_l2963_296376


namespace NUMINAMATH_CALUDE_pairing_fraction_l2963_296331

/-- Represents the number of students in each grade --/
structure Students where
  seventh : ℕ
  tenth : ℕ

/-- Represents the pairing between seventh and tenth graders --/
def Pairing (s : Students) :=
  (s.tenth / 4 : ℚ) = (s.seventh / 3 : ℚ)

/-- Calculates the fraction of students with partners --/
def fractionWithPartners (s : Students) : ℚ :=
  (s.tenth / 4 + s.seventh / 3) / (s.tenth + s.seventh)

theorem pairing_fraction (s : Students) (h : Pairing s) :
  fractionWithPartners s = 2 / 7 := by
  sorry


end NUMINAMATH_CALUDE_pairing_fraction_l2963_296331


namespace NUMINAMATH_CALUDE_red_bottle_caps_l2963_296318

theorem red_bottle_caps (total : ℕ) (green_percentage : ℚ) : 
  total = 125 → green_percentage = 60 / 100 → 
  (total : ℚ) * (1 - green_percentage) = 50 := by
sorry

end NUMINAMATH_CALUDE_red_bottle_caps_l2963_296318


namespace NUMINAMATH_CALUDE_circle_properties_l2963_296351

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 5 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (3, 0)

-- Define the radius of the circle
def circle_radius : ℝ := 2

-- Theorem statement
theorem circle_properties :
  ∀ x y : ℝ, circle_equation x y ↔ 
    ((x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l2963_296351


namespace NUMINAMATH_CALUDE_instant_noodle_price_reduction_l2963_296353

theorem instant_noodle_price_reduction 
  (original_weight : ℝ) 
  (original_price : ℝ) 
  (weight_increase_percentage : ℝ) 
  (h1 : weight_increase_percentage = 0.25) 
  (h2 : original_weight > 0) 
  (h3 : original_price > 0) : 
  let new_weight := original_weight * (1 + weight_increase_percentage)
  let original_price_per_unit := original_price / original_weight
  let new_price_per_unit := original_price / new_weight
  (original_price_per_unit - new_price_per_unit) / original_price_per_unit = 0.2
  := by sorry

end NUMINAMATH_CALUDE_instant_noodle_price_reduction_l2963_296353


namespace NUMINAMATH_CALUDE_golf_distance_l2963_296326

/-- 
Given a golf scenario where:
1. The distance from the starting tee to the hole is 250 yards.
2. On the second turn, the ball traveled half as far as it did on the first turn.
3. After the second turn, the ball landed 20 yards beyond the hole.
This theorem proves that the distance the ball traveled on the first turn is 180 yards.
-/
theorem golf_distance (first_turn : ℝ) (second_turn : ℝ) : 
  (first_turn + second_turn = 250 + 20) →  -- Total distance is to the hole plus 20 yards beyond
  (second_turn = first_turn / 2) →         -- Second turn is half of the first turn
  (first_turn = 180) :=                    -- The distance of the first turn is 180 yards
by sorry

end NUMINAMATH_CALUDE_golf_distance_l2963_296326


namespace NUMINAMATH_CALUDE_teaspoon_knife_ratio_l2963_296394

/-- Proves that the ratio of initial teaspoons to initial knives is 2:1 --/
theorem teaspoon_knife_ratio : 
  ∀ (initial_teaspoons : ℕ),
  let initial_knives : ℕ := 24
  let additional_knives : ℕ := initial_knives / 3
  let additional_teaspoons : ℕ := (2 * initial_teaspoons) / 3
  let total_cutlery : ℕ := 112
  (initial_knives + initial_teaspoons + additional_knives + additional_teaspoons = total_cutlery) →
  (initial_teaspoons : ℚ) / initial_knives = 2 := by
  sorry

end NUMINAMATH_CALUDE_teaspoon_knife_ratio_l2963_296394


namespace NUMINAMATH_CALUDE_sunset_colors_proof_l2963_296395

/-- The number of colors the sky turns during a sunset --/
def sunset_colors (sunset_duration : ℕ) (color_change_interval : ℕ) : ℕ :=
  sunset_duration / color_change_interval

theorem sunset_colors_proof (hours : ℕ) (minutes_per_hour : ℕ) (color_change_interval : ℕ) :
  hours = 2 →
  minutes_per_hour = 60 →
  color_change_interval = 10 →
  sunset_colors (hours * minutes_per_hour) color_change_interval = 12 := by
  sorry

end NUMINAMATH_CALUDE_sunset_colors_proof_l2963_296395


namespace NUMINAMATH_CALUDE_fraction_say_dislike_actually_like_l2963_296361

def TotalStudents : ℝ := 100

def LikeDancing : ℝ := 0.6 * TotalStudents
def DislikeDancing : ℝ := 0.4 * TotalStudents

def SayLikeActuallyLike : ℝ := 0.8 * LikeDancing
def SayDislikeActuallyLike : ℝ := 0.2 * LikeDancing
def SayDislikeActuallyDislike : ℝ := 0.9 * DislikeDancing
def SayLikeActuallyDislike : ℝ := 0.1 * DislikeDancing

def TotalSayDislike : ℝ := SayDislikeActuallyLike + SayDislikeActuallyDislike

theorem fraction_say_dislike_actually_like : 
  SayDislikeActuallyLike / TotalSayDislike = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_say_dislike_actually_like_l2963_296361


namespace NUMINAMATH_CALUDE_reading_time_theorem_l2963_296304

/-- Calculates the number of days needed to read a book given the total pages and reading speeds for each half. -/
def days_to_read_book (total_pages : ℕ) (first_half_speed : ℕ) (second_half_speed : ℕ) : ℕ :=
  let half_pages := total_pages / 2
  let first_half_days := half_pages / first_half_speed
  let second_half_days := half_pages / second_half_speed
  first_half_days + second_half_days

/-- Theorem stating that reading a 500-page book with given speeds takes 75 days. -/
theorem reading_time_theorem :
  days_to_read_book 500 10 5 = 75 := by
  sorry

#eval days_to_read_book 500 10 5

end NUMINAMATH_CALUDE_reading_time_theorem_l2963_296304


namespace NUMINAMATH_CALUDE_no_rational_solution_for_odd_coefficient_quadratic_l2963_296368

theorem no_rational_solution_for_odd_coefficient_quadratic
  (a b c : ℕ) (ha : Odd a) (hb : Odd b) (hc : Odd c) :
  ¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_for_odd_coefficient_quadratic_l2963_296368


namespace NUMINAMATH_CALUDE_largest_angle_convex_pentagon_l2963_296379

theorem largest_angle_convex_pentagon (x : ℝ) : 
  (x + 2) + (2*x + 3) + (3*x - 5) + (4*x + 1) + (5*x - 1) = 540 →
  max (x + 2) (max (2*x + 3) (max (3*x - 5) (max (4*x + 1) (5*x - 1)))) = 179 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_convex_pentagon_l2963_296379


namespace NUMINAMATH_CALUDE_marks_remaining_money_l2963_296338

def initial_money : ℕ := 85
def num_books : ℕ := 10
def book_cost : ℕ := 5

theorem marks_remaining_money :
  initial_money - (num_books * book_cost) = 35 := by
  sorry

end NUMINAMATH_CALUDE_marks_remaining_money_l2963_296338


namespace NUMINAMATH_CALUDE_sum_is_positive_l2963_296346

theorem sum_is_positive (x y : ℝ) (h1 : x * y < 0) (h2 : x > abs y) : x + y > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_is_positive_l2963_296346


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2963_296314

/-- The number of games in a chess tournament -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1)

/-- The theorem stating the number of games in the specific tournament -/
theorem chess_tournament_games :
  tournament_games 19 * 2 = 684 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2963_296314


namespace NUMINAMATH_CALUDE_min_group_size_l2963_296377

/-- Represents the number of men in a group with various attributes -/
structure MenGroup where
  total : ℕ
  married : ℕ
  hasTV : ℕ
  hasRadio : ℕ
  hasAC : ℕ
  hasAll : ℕ

/-- The minimum number of men in the group is at least the maximum of any single category -/
theorem min_group_size (g : MenGroup) 
  (h1 : g.married = 81)
  (h2 : g.hasTV = 75)
  (h3 : g.hasRadio = 85)
  (h4 : g.hasAC = 70)
  (h5 : g.hasAll = 11)
  : g.total ≥ 85 := by
  sorry

#check min_group_size

end NUMINAMATH_CALUDE_min_group_size_l2963_296377


namespace NUMINAMATH_CALUDE_four_numbers_product_sum_l2963_296369

theorem four_numbers_product_sum (x₁ x₂ x₃ x₄ : ℝ) : 
  (x₁ + x₂ * x₃ * x₄ = 2 ∧
   x₂ + x₁ * x₃ * x₄ = 2 ∧
   x₃ + x₁ * x₂ * x₄ = 2 ∧
   x₄ + x₁ * x₂ * x₃ = 2) ↔
  ((x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1 ∧ x₄ = 1) ∨
   (x₁ = -1 ∧ x₂ = -1 ∧ x₃ = -1 ∧ x₄ = 3) ∨
   (x₁ = -1 ∧ x₂ = -1 ∧ x₃ = 3 ∧ x₄ = -1) ∨
   (x₁ = -1 ∧ x₂ = 3 ∧ x₃ = -1 ∧ x₄ = -1) ∨
   (x₁ = 3 ∧ x₂ = -1 ∧ x₃ = -1 ∧ x₄ = -1)) := by
sorry


end NUMINAMATH_CALUDE_four_numbers_product_sum_l2963_296369


namespace NUMINAMATH_CALUDE_lucas_addition_example_l2963_296315

/-- Lucas's notation for integers -/
def lucas_notation (n : ℤ) : ℕ :=
  if n ≥ 0 then n.natAbs else n.natAbs + 1

/-- Addition in Lucas's notation -/
def lucas_add (a b : ℕ) : ℕ :=
  lucas_notation (-(a : ℤ) + -(b : ℤ))

/-- Theorem: 000 + 0000 = 000000 in Lucas's notation -/
theorem lucas_addition_example : lucas_add 3 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_lucas_addition_example_l2963_296315


namespace NUMINAMATH_CALUDE_closest_point_is_A_l2963_296324

-- Define the points as real numbers
variable (A B C D E : ℝ)

-- Define the conditions
axiom A_range : 0 < A ∧ A < 1
axiom B_range : 0 < B ∧ B < 1
axiom C_range : 0 < C ∧ C < 1
axiom D_range : 0 < D ∧ D < 1
axiom E_range : 1 < E ∧ E < 2

-- Define the order of points
axiom point_order : A < B ∧ B < C ∧ C < D

-- Define a function to calculate the distance between two real numbers
def distance (x y : ℝ) : ℝ := |x - y|

-- State the theorem
theorem closest_point_is_A :
  distance (B * C) A < distance (B * C) B ∧
  distance (B * C) A < distance (B * C) C ∧
  distance (B * C) A < distance (B * C) D ∧
  distance (B * C) A < distance (B * C) E :=
sorry

end NUMINAMATH_CALUDE_closest_point_is_A_l2963_296324


namespace NUMINAMATH_CALUDE_y_intercept_distance_of_intersecting_lines_l2963_296396

/-- A line in 2D space represented by its slope and a point it passes through -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Calculate the y-intercept of a line -/
def y_intercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- The distance between two real numbers -/
def distance (a b : ℝ) : ℝ :=
  |a - b|

theorem y_intercept_distance_of_intersecting_lines :
  let l1 : Line := { slope := -2, point := (8, 20) }
  let l2 : Line := { slope := 4, point := (8, 20) }
  distance (y_intercept l1) (y_intercept l2) = 68 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_distance_of_intersecting_lines_l2963_296396


namespace NUMINAMATH_CALUDE_arithmetic_sequence_from_equation_l2963_296341

theorem arithmetic_sequence_from_equation (a b c : ℝ) :
  (2*b - a)^2 + (2*b - c)^2 = 2*(2*b^2 - a*c) →
  b = (a + c) / 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_from_equation_l2963_296341


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l2963_296391

theorem opposite_of_negative_fraction :
  -(-(1 : ℚ) / 2023) = 1 / 2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l2963_296391


namespace NUMINAMATH_CALUDE_fraction_transformation_l2963_296345

theorem fraction_transformation (x : ℚ) : 
  (3 + x) / (11 + x) = 5 / 9 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_l2963_296345


namespace NUMINAMATH_CALUDE_probability_at_least_one_correct_l2963_296383

theorem probability_at_least_one_correct (n m : ℕ) (h : n > 0 ∧ m > 0) :
  let p := 1 - (1 - 1 / n) ^ m
  n = 6 ∧ m = 6 → p = 31031 / 46656 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_correct_l2963_296383


namespace NUMINAMATH_CALUDE_smallest_divisible_integer_l2963_296329

theorem smallest_divisible_integer : ∃ (M : ℕ), 
  (M = 362) ∧ 
  (∀ (k : ℕ), k < M → ¬(
    (∃ (i : Fin 3), 2^2 ∣ (k + i)) ∧
    (∃ (i : Fin 3), 3^2 ∣ (k + i)) ∧
    (∃ (i : Fin 3), 7^2 ∣ (k + i)) ∧
    (∃ (i : Fin 3), 11^2 ∣ (k + i))
  )) ∧
  (∃ (i : Fin 3), 2^2 ∣ (M + i)) ∧
  (∃ (i : Fin 3), 3^2 ∣ (M + i)) ∧
  (∃ (i : Fin 3), 7^2 ∣ (M + i)) ∧
  (∃ (i : Fin 3), 11^2 ∣ (M + i)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_integer_l2963_296329


namespace NUMINAMATH_CALUDE_balance_colored_balls_l2963_296337

/-- Given balance relationships between colored balls, prove the number of blue balls needed to balance a specific combination. -/
theorem balance_colored_balls (r b o p : ℚ) 
  (h1 : 4 * r = 8 * b) 
  (h2 : 3 * o = 7 * b) 
  (h3 : 8 * b = 6 * p) : 
  5 * r + 3 * o + 4 * p = 67/3 * b := by
  sorry

end NUMINAMATH_CALUDE_balance_colored_balls_l2963_296337


namespace NUMINAMATH_CALUDE_investment_with_interest_l2963_296385

def total_investment : ℝ := 1000
def amount_at_3_percent : ℝ := 199.99999999999983
def interest_rate_3_percent : ℝ := 0.03
def interest_rate_5_percent : ℝ := 0.05

theorem investment_with_interest :
  let amount_at_5_percent := total_investment - amount_at_3_percent
  let interest_at_3_percent := amount_at_3_percent * interest_rate_3_percent
  let interest_at_5_percent := amount_at_5_percent * interest_rate_5_percent
  let total_with_interest := total_investment + interest_at_3_percent + interest_at_5_percent
  total_with_interest = 1046 := by sorry

end NUMINAMATH_CALUDE_investment_with_interest_l2963_296385


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_300_l2963_296311

def mersenne_number (n : ℕ) : ℕ := 2^n - 1

def is_mersenne_prime (m : ℕ) : Prop :=
  ∃ n : ℕ, Prime n ∧ m = mersenne_number n ∧ Prime m

theorem largest_mersenne_prime_under_300 :
  ∀ m : ℕ, is_mersenne_prime m → m < 300 → m ≤ 127 :=
by sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_300_l2963_296311


namespace NUMINAMATH_CALUDE_cafeteria_bill_theorem_l2963_296357

/-- The total cost of a cafeteria order for three people -/
def cafeteria_cost (coffee_price ice_cream_price cake_price : ℕ) : ℕ :=
  let mell_order := 2 * coffee_price + cake_price
  let friend_order := 2 * coffee_price + cake_price + ice_cream_price
  mell_order + 2 * friend_order

/-- Theorem stating the total cost for Mell and her friends' cafeteria order -/
theorem cafeteria_bill_theorem :
  cafeteria_cost 4 3 7 = 51 :=
by
  sorry

end NUMINAMATH_CALUDE_cafeteria_bill_theorem_l2963_296357


namespace NUMINAMATH_CALUDE_cubic_extreme_values_l2963_296388

/-- Given a cubic function f(x) = x^3 - px^2 - qx that passes through (1,0),
    prove that its maximum value is 4/27 and its minimum value is 0. -/
theorem cubic_extreme_values (p q : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 - p*x^2 - q*x
  (f 1 = 0) →
  (∃ x, f x = 4/27) ∧ (∀ y, f y ≤ 4/27) ∧ (∃ z, f z = 0) ∧ (∀ w, f w ≥ 0) :=
by sorry


end NUMINAMATH_CALUDE_cubic_extreme_values_l2963_296388


namespace NUMINAMATH_CALUDE_negative_one_and_half_equality_l2963_296365

theorem negative_one_and_half_equality : -1 - (1/2 : ℚ) = -(3/2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_negative_one_and_half_equality_l2963_296365


namespace NUMINAMATH_CALUDE_student_pairs_l2963_296328

theorem student_pairs (n : ℕ) (h : n = 12) : Nat.choose n 2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_student_pairs_l2963_296328


namespace NUMINAMATH_CALUDE_steves_pool_filling_time_l2963_296373

/-- The time required to fill Steve's pool -/
theorem steves_pool_filling_time :
  let pool_capacity : ℝ := 30000  -- gallons
  let num_hoses : ℕ := 5
  let flow_rate_per_hose : ℝ := 3  -- gallons per minute
  let minutes_per_hour : ℕ := 60
  
  let total_flow_rate : ℝ := num_hoses * flow_rate_per_hose  -- gallons per minute
  let hourly_flow_rate : ℝ := total_flow_rate * minutes_per_hour  -- gallons per hour
  let filling_time : ℝ := pool_capacity / hourly_flow_rate  -- hours
  
  ⌈filling_time⌉ = 34 := by
  sorry

end NUMINAMATH_CALUDE_steves_pool_filling_time_l2963_296373


namespace NUMINAMATH_CALUDE_original_cost_price_satisfies_conditions_l2963_296336

/-- The original cost price of a computer satisfying given conditions -/
def original_cost_price : ℝ := 40

/-- The selling price of the computer -/
def selling_price : ℝ := 48

/-- The decrease rate of the cost price -/
def cost_decrease_rate : ℝ := 0.04

/-- The increase rate of the profit margin -/
def profit_margin_increase_rate : ℝ := 0.05

/-- Theorem stating that the original cost price satisfies all given conditions -/
theorem original_cost_price_satisfies_conditions :
  let new_cost_price := original_cost_price * (1 - cost_decrease_rate)
  let original_profit_margin := (selling_price - original_cost_price) / original_cost_price
  let new_profit_margin := (selling_price - new_cost_price) / new_cost_price
  new_profit_margin = original_profit_margin + profit_margin_increase_rate := by
  sorry


end NUMINAMATH_CALUDE_original_cost_price_satisfies_conditions_l2963_296336


namespace NUMINAMATH_CALUDE_quadratic_equation_general_form_l2963_296323

/-- A quadratic equation in one variable -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  x : ℝ
  h_quadratic : a ≠ 0

/-- The general form of a quadratic equation -/
def general_form (eq : QuadraticEquation) : Prop :=
  eq.a * eq.x^2 + eq.b * eq.x + eq.c = 0

/-- Theorem: The general form of a quadratic equation in one variable is ax^2 + bx + c = 0 where a ≠ 0 -/
theorem quadratic_equation_general_form (eq : QuadraticEquation) :
  general_form eq :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_general_form_l2963_296323


namespace NUMINAMATH_CALUDE_existence_of_c_l2963_296339

theorem existence_of_c (a b : ℝ) : ∃ c ∈ Set.Icc 0 1, |a * c + b + 1 / (c + 1)| ≥ 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_c_l2963_296339


namespace NUMINAMATH_CALUDE_batsman_average_l2963_296387

/-- Calculates the average runs for a batsman given two sets of matches --/
def calculate_average (matches1 : ℕ) (average1 : ℕ) (matches2 : ℕ) (average2 : ℕ) : ℚ :=
  let total_runs := matches1 * average1 + matches2 * average2
  let total_matches := matches1 + matches2
  (total_runs : ℚ) / total_matches

/-- Proves that the batsman's average for 30 matches is 31 runs --/
theorem batsman_average : calculate_average 20 40 10 13 = 31 := by
  sorry

#eval calculate_average 20 40 10 13

end NUMINAMATH_CALUDE_batsman_average_l2963_296387


namespace NUMINAMATH_CALUDE_career_d_degrees_l2963_296309

/-- Represents the ratio of male to female students -/
def maleToFemaleRatio : Rat := 2 / 3

/-- Represents the percentage of males preferring each career -/
def malePreference : Fin 6 → Rat
| 0 => 25 / 100  -- Career A
| 1 => 15 / 100  -- Career B
| 2 => 30 / 100  -- Career C
| 3 => 40 / 100  -- Career D
| 4 => 20 / 100  -- Career E
| 5 => 35 / 100  -- Career F

/-- Represents the percentage of females preferring each career -/
def femalePreference : Fin 6 → Rat
| 0 => 50 / 100  -- Career A
| 1 => 40 / 100  -- Career B
| 2 => 10 / 100  -- Career C
| 3 => 20 / 100  -- Career D
| 4 => 30 / 100  -- Career E
| 5 => 25 / 100  -- Career F

/-- Calculates the degrees in a circle graph for a given career -/
def careerDegrees (careerIndex : Fin 6) : ℚ :=
  let totalStudents := maleToFemaleRatio + 1
  let maleStudents := maleToFemaleRatio
  let femaleStudents := 1
  let studentsPreferringCareer := 
    maleStudents * malePreference careerIndex + femaleStudents * femalePreference careerIndex
  (studentsPreferringCareer / totalStudents) * 360

/-- Theorem stating that Career D should be represented by 100.8 degrees in the circle graph -/
theorem career_d_degrees : careerDegrees 3 = 100.8 := by sorry

end NUMINAMATH_CALUDE_career_d_degrees_l2963_296309


namespace NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l2963_296313

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 9| = |x + 3| + 2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l2963_296313


namespace NUMINAMATH_CALUDE_right_triangle_min_std_dev_l2963_296352

theorem right_triangle_min_std_dev (a b c : ℝ) : 
  a > 0 → b > 0 → c = 3 → a^2 + b^2 = c^2 →
  let s := Real.sqrt ((a^2 + b^2 + c^2) / 3 - ((a + b + c) / 3)^2)
  s ≥ Real.sqrt 2 - 1 ∧ 
  (s = Real.sqrt 2 - 1 ↔ a = 3 * Real.sqrt 2 / 2 ∧ b = 3 * Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_min_std_dev_l2963_296352


namespace NUMINAMATH_CALUDE_stream_speed_l2963_296316

/-- Proves that the speed of the stream is 8 kmph given the conditions of the problem -/
theorem stream_speed (boat_speed : ℝ) (stream_speed : ℝ) : 
  boat_speed = 24 →
  (1 / (boat_speed - stream_speed)) = (2 / (boat_speed + stream_speed)) →
  stream_speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l2963_296316


namespace NUMINAMATH_CALUDE_frustum_volume_specific_l2963_296397

/-- Frustum properties -/
structure Frustum where
  height : ℝ
  slant_height : ℝ
  lateral_area : ℝ

/-- Calculate the volume of a frustum -/
def frustum_volume (f : Frustum) : ℝ :=
  sorry

/-- Theorem: The volume of a frustum with height 4, slant height 5, and lateral area 45π is 84π -/
theorem frustum_volume_specific (f : Frustum) 
  (h_height : f.height = 4)
  (h_slant : f.slant_height = 5)
  (h_lateral : f.lateral_area = 45 * Real.pi) : 
  frustum_volume f = 84 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_frustum_volume_specific_l2963_296397


namespace NUMINAMATH_CALUDE_remainder_17_pow_53_mod_7_l2963_296358

theorem remainder_17_pow_53_mod_7 : 17^53 % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_17_pow_53_mod_7_l2963_296358
