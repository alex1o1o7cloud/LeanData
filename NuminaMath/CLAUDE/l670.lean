import Mathlib

namespace equilibrium_constant_temperature_relation_l670_67016

-- Define the chemical equilibrium constant
variable (K : ℝ)

-- Define temperature
variable (T : ℝ)

-- Define a relation between K and T
def related_to_temperature (K T : ℝ) : Prop := sorry

-- Theorem stating that K is related to temperature
theorem equilibrium_constant_temperature_relation :
  related_to_temperature K T :=
sorry

end equilibrium_constant_temperature_relation_l670_67016


namespace probability_of_two_red_balls_l670_67026

-- Define the number of balls of each color
def red_balls : ℕ := 5
def blue_balls : ℕ := 4
def green_balls : ℕ := 3

-- Define the total number of balls
def total_balls : ℕ := red_balls + blue_balls + green_balls

-- Define the number of balls to be picked
def balls_picked : ℕ := 2

-- Theorem statement
theorem probability_of_two_red_balls :
  (Nat.choose red_balls balls_picked : ℚ) / (Nat.choose total_balls balls_picked) = 5 / 33 :=
sorry

end probability_of_two_red_balls_l670_67026


namespace potion_price_l670_67025

theorem potion_price (current_price : ℚ) (original_price : ℚ) : 
  current_price = 9 → current_price = (1 / 15) * original_price → original_price = 135 := by
  sorry

end potion_price_l670_67025


namespace equation_solution_l670_67010

theorem equation_solution (x : ℝ) : 
  (x^3 - 3*x^2) / (x^2 - 4*x + 3) + 2*x = 0 → x = 0 ∨ x = 2/3 := by
  sorry

end equation_solution_l670_67010


namespace min_value_expression_l670_67017

theorem min_value_expression (a b c : ℝ) (h1 : c > b) (h2 : b > a) (h3 : c ≠ 0) :
  ((a + b)^2 + (b - c)^2 + (c - b)^2) / c^2 ≥ 0 ∧
  ∃ a b, ((a + b)^2 + (b - c)^2 + (c - b)^2) / c^2 = 0 := by
  sorry

end min_value_expression_l670_67017


namespace pharmacist_weights_exist_l670_67020

theorem pharmacist_weights_exist : ∃ (w₁ w₂ w₃ : ℝ),
  w₁ < 90 ∧ w₂ < 90 ∧ w₃ < 90 ∧
  w₁ + w₂ + w₃ = 100 ∧
  w₁ + w₂ + (w₃ + 1) = 101 ∧
  w₂ + w₃ + (w₃ + 1) = 102 :=
by sorry

end pharmacist_weights_exist_l670_67020


namespace alice_bob_meeting_l670_67089

/-- The number of points on the circle -/
def n : ℕ := 12

/-- Alice's clockwise movement per turn -/
def alice_move : ℕ := 7

/-- Bob's counterclockwise movement per turn -/
def bob_move : ℕ := 4

/-- The relative movement of Alice to Bob per turn -/
def relative_move : ℤ := alice_move - (n - bob_move)

/-- The number of turns required for Alice and Bob to meet -/
def meeting_turns : ℕ := n

theorem alice_bob_meeting :
  (relative_move * meeting_turns) % n = 0 ∧
  ∀ k : ℕ, k < meeting_turns → (relative_move * k) % n ≠ 0 :=
sorry

end alice_bob_meeting_l670_67089


namespace interest_rate_difference_l670_67050

/-- Given a principal amount, time period, and difference in interest earned,
    calculate the difference between two interest rates. -/
theorem interest_rate_difference
  (principal : ℝ)
  (time : ℝ)
  (interest_diff : ℝ)
  (h1 : principal = 200)
  (h2 : time = 10)
  (h3 : interest_diff = 100) :
  (interest_diff / (principal * time)) * 100 = 5 := by
  sorry

end interest_rate_difference_l670_67050


namespace smallest_four_digit_arithmetic_sequence_l670_67045

def is_arithmetic_sequence (a b c d : ℕ) : Prop :=
  ∃ r : ℤ, b = a + r ∧ c = b + r ∧ d = c + r

def digits_are_distinct (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 4 ∧ digits.toFinset.card = 4

theorem smallest_four_digit_arithmetic_sequence :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 
    digits_are_distinct n ∧
    is_arithmetic_sequence (n / 1000 % 10) (n / 100 % 10) (n / 10 % 10) (n % 10) →
  1234 ≤ n :=
sorry

end smallest_four_digit_arithmetic_sequence_l670_67045


namespace fixed_point_on_all_lines_l670_67085

/-- The fixed point through which all lines of a certain form pass -/
def fixed_point : ℝ × ℝ := (2, 1)

/-- The line equation parameterized by k -/
def line_equation (k : ℝ) (x y : ℝ) : Prop :=
  k * x - y + 1 - 2 * k = 0

/-- Theorem stating that the fixed point lies on all lines of the given form -/
theorem fixed_point_on_all_lines :
  ∀ k : ℝ, line_equation k (fixed_point.1) (fixed_point.2) :=
by
  sorry

#check fixed_point_on_all_lines

end fixed_point_on_all_lines_l670_67085


namespace candy_distribution_l670_67074

theorem candy_distribution (total_candy : ℕ) (total_bags : ℕ) (heart_bags : ℕ) (kiss_bags : ℕ) (jelly_bags : ℕ) :
  total_candy = 260 →
  total_bags = 13 →
  heart_bags = 4 →
  kiss_bags = 5 →
  jelly_bags = 3 →
  total_candy % total_bags = 0 →
  let pieces_per_bag := total_candy / total_bags
  let chew_bags := total_bags - heart_bags - kiss_bags - jelly_bags
  heart_bags * pieces_per_bag + chew_bags * pieces_per_bag + jelly_bags * pieces_per_bag = total_candy :=
by sorry

#check candy_distribution

end candy_distribution_l670_67074


namespace expression_simplification_l670_67081

theorem expression_simplification (a b : ℝ) 
  (ha : a = Real.sqrt 3 - Real.sqrt 11) 
  (hb : b = Real.sqrt 3 + Real.sqrt 11) : 
  (a^2 - b^2) / (a^2 * b - a * b^2) / (1 + (a^2 + b^2) / (2 * a * b)) = Real.sqrt 3 / 6 := by
  sorry

end expression_simplification_l670_67081


namespace prime_pairs_sum_50_l670_67009

/-- A function that returns the number of unordered pairs of prime numbers that sum to a given natural number. -/
def count_prime_pairs (n : ℕ) : ℕ :=
  (Finset.filter (fun p => Nat.Prime p ∧ Nat.Prime (n - p) ∧ 2 * p ≤ n) (Finset.range (n / 2 + 1))).card

/-- The theorem stating that there are exactly 4 unordered pairs of prime numbers that sum to 50. -/
theorem prime_pairs_sum_50 : count_prime_pairs 50 = 4 := by
  sorry

end prime_pairs_sum_50_l670_67009


namespace velvet_for_hats_and_cloaks_l670_67042

/-- The amount of velvet needed for hats and cloaks -/
def velvet_needed (hats_per_yard : ℚ) (yards_per_cloak : ℚ) (num_hats : ℚ) (num_cloaks : ℚ) : ℚ :=
  (num_hats / hats_per_yard) + (num_cloaks * yards_per_cloak)

/-- Theorem stating the total amount of velvet needed for 6 cloaks and 12 hats -/
theorem velvet_for_hats_and_cloaks :
  velvet_needed 4 3 12 6 = 21 := by
  sorry

end velvet_for_hats_and_cloaks_l670_67042


namespace f_increasing_iff_a_in_range_l670_67049

/-- Piecewise function f(x) defined by a parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (6 - a) * x - 2 * a else a^x

/-- Theorem stating the range of a for which f is increasing on ℝ -/
theorem f_increasing_iff_a_in_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ (3/2 ≤ a ∧ a < 6) :=
sorry

end f_increasing_iff_a_in_range_l670_67049


namespace floor_square_minus_floor_product_l670_67099

theorem floor_square_minus_floor_product (x : ℝ) : x = 13.2 →
  ⌊x^2⌋ - ⌊x⌋ * ⌊x⌋ = 5 := by sorry

end floor_square_minus_floor_product_l670_67099


namespace smallest_consecutive_sum_divisible_by_17_l670_67093

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Check if two natural numbers are consecutive -/
def areConsecutive (a b : ℕ) : Prop := b = a + 1

/-- Check if there exist smaller consecutive numbers satisfying the condition -/
def existSmallerPair (a b : ℕ) : Prop :=
  ∃ (x y : ℕ), x < a ∧ areConsecutive x y ∧ 
    (sumOfDigits x % 17 = 0) ∧ (sumOfDigits y % 17 = 0)

theorem smallest_consecutive_sum_divisible_by_17 :
  areConsecutive 8899 8900 ∧
  (sumOfDigits 8899 % 17 = 0) ∧
  (sumOfDigits 8900 % 17 = 0) ∧
  ¬(existSmallerPair 8899 8900) :=
sorry

end smallest_consecutive_sum_divisible_by_17_l670_67093


namespace ExistsFourDigitNumberDivisibleBy11WithDigitSum10_l670_67063

-- Define a four-digit number
def FourDigitNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

-- Define the sum of digits
def SumOfDigits (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

-- Theorem statement
theorem ExistsFourDigitNumberDivisibleBy11WithDigitSum10 :
  ∃ n : ℕ, FourDigitNumber n ∧ SumOfDigits n = 10 ∧ n % 11 = 0 :=
by sorry

end ExistsFourDigitNumberDivisibleBy11WithDigitSum10_l670_67063


namespace identity_function_divisibility_l670_67058

def is_divisible (a b : ℕ) : Prop := ∃ k, b = a * k

theorem identity_function_divisibility :
  ∀ f : ℕ+ → ℕ+, 
  (∀ x y : ℕ+, is_divisible (x.val * f x + y.val * f y) ((x.val^2 + y.val^2)^2022)) → 
  (∀ x : ℕ+, f x = x) := by sorry

end identity_function_divisibility_l670_67058


namespace simplify_and_evaluate_simplify_with_condition_l670_67037

-- Part 1
theorem simplify_and_evaluate (x y : ℤ) (h1 : x = -2) (h2 : y = -3) :
  x^2 - 2*(x^2 - 3*y) - 3*(2*x^2 + 5*y) = -1 := by sorry

-- Part 2
theorem simplify_with_condition (a b : ℝ) (h : a - b = 2*b^2) :
  2*(a^3 - 2*b^2) - (2*b - a) + a - 2*a^3 = 0 := by sorry

end simplify_and_evaluate_simplify_with_condition_l670_67037


namespace sanitizer_dilution_l670_67006

/-- Proves that adding 6 ounces of water to 12 ounces of hand sanitizer with 60% alcohol
    concentration results in a solution with 40% alcohol concentration. -/
theorem sanitizer_dilution (initial_volume : ℝ) (initial_concentration : ℝ)
    (water_added : ℝ) (final_concentration : ℝ)
    (h1 : initial_volume = 12)
    (h2 : initial_concentration = 0.6)
    (h3 : water_added = 6)
    (h4 : final_concentration = 0.4) :
  initial_concentration * initial_volume =
    final_concentration * (initial_volume + water_added) :=
by sorry

end sanitizer_dilution_l670_67006


namespace sum_of_abs_and_square_zero_l670_67008

theorem sum_of_abs_and_square_zero (x y : ℝ) :
  |x + 3| + (2 * y - 5)^2 = 0 → x + 2 * y = 2 := by
  sorry

end sum_of_abs_and_square_zero_l670_67008


namespace simplify_expression_l670_67013

theorem simplify_expression (a b : ℝ) (h : a > 0 ∧ b > 0) :
  a^(2/3) * b^(1/2) * (-3 * a^(1/2) * b^(1/3)) / (1/3 * a^(1/6) * b^(5/6)) = -9 * a :=
by sorry

end simplify_expression_l670_67013


namespace system_solution_ratio_l670_67078

/-- The system of equations has a nontrivial solution with the given ratio -/
theorem system_solution_ratio :
  ∃ (x y z : ℚ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
  x + (95/9)*y + 4*z = 0 ∧
  4*x + (95/9)*y - 3*z = 0 ∧
  3*x + 5*y - 4*z = 0 ∧
  x*z / (y^2) = 175/81 := by
sorry

end system_solution_ratio_l670_67078


namespace horizontal_cut_length_l670_67094

/-- An isosceles triangle with given properties -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  area : ℝ

/-- A horizontal cut in an isosceles triangle -/
structure HorizontalCut where
  triangle : IsoscelesTriangle
  trapezoidArea : ℝ
  cutLength : ℝ

/-- The main theorem -/
theorem horizontal_cut_length 
  (triangle : IsoscelesTriangle)
  (cut : HorizontalCut)
  (h1 : triangle.area = 144)
  (h2 : triangle.height = 24)
  (h3 : cut.triangle = triangle)
  (h4 : cut.trapezoidArea = 108) :
  cut.cutLength = 6 := by
  sorry

end horizontal_cut_length_l670_67094


namespace probability_negative_product_l670_67036

def S : Finset Int := {-6, -3, -1, 2, 5, 8}

def negative_product_pairs (S : Finset Int) : Finset (Int × Int) :=
  S.product S |>.filter (fun (a, b) => a ≠ b ∧ a * b < 0)

def total_pairs (S : Finset Int) : Finset (Int × Int) :=
  S.product S |>.filter (fun (a, b) => a ≠ b)

theorem probability_negative_product :
  (negative_product_pairs S).card / (total_pairs S).card = 3 / 5 := by
  sorry

end probability_negative_product_l670_67036


namespace nala_seashells_l670_67002

/-- The number of seashells Nala found on the first day -/
def first_day : ℕ := 5

/-- The number of seashells Nala found on the second day -/
def second_day : ℕ := 7

/-- The number of seashells Nala found on the third day is twice the sum of the first two days -/
def third_day : ℕ := 2 * (first_day + second_day)

/-- The total number of seashells Nala has -/
def total_seashells : ℕ := first_day + second_day + third_day

theorem nala_seashells : total_seashells = 36 := by
  sorry

end nala_seashells_l670_67002


namespace roots_of_unity_quadratic_equation_l670_67047

theorem roots_of_unity_quadratic_equation :
  ∃! (S : Finset ℂ),
    (∀ z ∈ S, (Complex.abs z = 1) ∧
      (∃ a : ℤ, z ^ 2 + a * z + 1 = 0 ∧
        -2 ≤ a ∧ a ≤ 2 ∧
        ∃ k : ℤ, a = k * Real.cos (k * π / 6))) ∧
    Finset.card S = 8 := by
  sorry

end roots_of_unity_quadratic_equation_l670_67047


namespace cos_graph_transformation_l670_67001

theorem cos_graph_transformation (x : ℝ) :
  let original_point := (x, Real.cos x)
  let transformed_point := (4 * x, Real.cos (x / 4))
  transformed_point.2 = original_point.2 := by
sorry

end cos_graph_transformation_l670_67001


namespace cubic_equation_solution_l670_67071

theorem cubic_equation_solution (p q : ℝ) : 
  (3 * p^2 - 5 * p - 8 = 0) → 
  (3 * q^2 - 5 * q - 8 = 0) → 
  p ≠ q →
  (9 * p^3 - 9 * q^3) * (p - q)⁻¹ = 49 := by
  sorry

end cubic_equation_solution_l670_67071


namespace power_of_ten_zeros_l670_67098

theorem power_of_ten_zeros (n : ℕ) : ∃ k : ℕ, (5000^50) * 100^2 = k * 10^154 ∧ 10^154 ≤ k ∧ k < 10^155 := by
  sorry

end power_of_ten_zeros_l670_67098


namespace flagpole_break_height_l670_67086

/-- Proves that a 6-meter flagpole breaking and touching the ground 2 meters away
    from its base breaks at a height of 3 meters. -/
theorem flagpole_break_height :
  ∀ (h x : ℝ),
  h = 6 →                            -- Total height of flagpole
  x > 0 →                            -- Breaking point is above ground
  x < h →                            -- Breaking point is below the top
  x^2 + 2^2 = (h - x)^2 →            -- Pythagorean theorem
  x = 3 := by
sorry

end flagpole_break_height_l670_67086


namespace extended_segment_coordinates_l670_67070

/-- Given two points A and B on a plane, and a point C such that BC = 2/3 * AB,
    this theorem proves that the coordinates of C can be determined. -/
theorem extended_segment_coordinates
  (A B : ℝ × ℝ)
  (hA : A = (-1, 3))
  (hB : B = (11, 7))
  (hC : ∃ C : ℝ × ℝ, (C.1 - B.1)^2 + (C.2 - B.2)^2 = (2/3)^2 * ((B.1 - A.1)^2 + (B.2 - A.2)^2)) :
  ∃ C : ℝ × ℝ, C = (19, 29/3) :=
sorry

end extended_segment_coordinates_l670_67070


namespace triangle_properties_l670_67024

-- Define the triangle ABC
def A : ℝ × ℝ := (0, 4)
def B : ℝ × ℝ := (-2, 6)
def C : ℝ × ℝ := (8, 2)

-- Define the median from A to BC
def median_A_BC (x y : ℝ) : Prop :=
  y = 4

-- Define the perpendicular bisector of AC
def perp_bisector_AC (x y : ℝ) : Prop :=
  y = 4 * x - 13

-- Theorem statement
theorem triangle_properties :
  (∀ x y, median_A_BC x y ↔ y = 4) ∧
  (∀ x y, perp_bisector_AC x y ↔ y = 4 * x - 13) := by
  sorry

end triangle_properties_l670_67024


namespace intersection_line_slope_l670_67021

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 6*y + 12 = 0

-- Define the intersection points
def intersection_points (x y : ℝ) : Prop := circle1 x y ∧ circle2 x y

-- Theorem stating that the slope of the line connecting intersection points is 1
theorem intersection_line_slope :
  ∃ (x1 y1 x2 y2 : ℝ),
    intersection_points x1 y1 ∧
    intersection_points x2 y2 ∧
    x1 ≠ x2 →
    (y2 - y1) / (x2 - x1) = 1 := by sorry

end intersection_line_slope_l670_67021


namespace a_between_3_and_5_necessary_not_sufficient_l670_67062

/-- The equation of a potential ellipse -/
def ellipse_equation (a x y : ℝ) : Prop :=
  x^2 / (a - 3) + y^2 / (5 - a) = 1

/-- The condition that a is between 3 and 5 -/
def a_between_3_and_5 (a : ℝ) : Prop :=
  3 < a ∧ a < 5

/-- The statement that the equation represents an ellipse -/
def is_ellipse (a : ℝ) : Prop :=
  ∃ (x y : ℝ), ellipse_equation a x y ∧ (x ≠ 0 ∨ y ≠ 0)

/-- The main theorem: a_between_3_and_5 is necessary but not sufficient for is_ellipse -/
theorem a_between_3_and_5_necessary_not_sufficient :
  (∀ a : ℝ, is_ellipse a → a_between_3_and_5 a) ∧
  ¬(∀ a : ℝ, a_between_3_and_5 a → is_ellipse a) :=
sorry

end a_between_3_and_5_necessary_not_sufficient_l670_67062


namespace arithmetic_sequence_common_difference_l670_67067

def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def sum_arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_common_difference 
  (a₁ d : ℚ) :
  arithmetic_sequence a₁ d 5 + arithmetic_sequence a₁ d 6 = -10 ∧
  sum_arithmetic_sequence a₁ d 14 = -14 →
  d = 2 := by
sorry

end arithmetic_sequence_common_difference_l670_67067


namespace product_of_solutions_abs_y_eq_3_abs_y_minus_2_l670_67069

theorem product_of_solutions_abs_y_eq_3_abs_y_minus_2 :
  ∃ (y₁ y₂ : ℝ), (|y₁| = 3 * (|y₁| - 2)) ∧ (|y₂| = 3 * (|y₂| - 2)) ∧ y₁ ≠ y₂ ∧ y₁ * y₂ = -9 :=
by sorry

end product_of_solutions_abs_y_eq_3_abs_y_minus_2_l670_67069


namespace base_eight_sum_l670_67096

theorem base_eight_sum (A B C : ℕ) : 
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
  A < 8 ∧ B < 8 ∧ C < 8 ∧
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  (A * 8^2 + B * 8 + C) + (B * 8^2 + C * 8 + A) + (C * 8^2 + A * 8 + B) = A * (8^3 + 8^2 + 8) →
  B + C = 7 :=
by sorry

end base_eight_sum_l670_67096


namespace safe_access_theorem_access_conditions_l670_67022

/-- Represents the number of members in the commission -/
def commission_size : ℕ := 11

/-- Represents the minimum number of members needed for access -/
def min_access : ℕ := 6

/-- Calculates the number of locks needed -/
def num_locks : ℕ := Nat.choose commission_size (min_access - 1)

/-- Calculates the number of keys each member should have -/
def keys_per_member : ℕ := num_locks * min_access / commission_size

/-- Theorem stating the correct number of locks and keys per member -/
theorem safe_access_theorem :
  num_locks = 462 ∧ keys_per_member = 252 :=
sorry

/-- Theorem proving that the arrangement satisfies the access conditions -/
theorem access_conditions (members : Finset (Fin commission_size)) :
  (members.card ≥ min_access → ∃ (lock : Fin num_locks), ∀ k ∈ members, k.val < keys_per_member) ∧
  (members.card < min_access → ∃ (lock : Fin num_locks), ∀ k ∈ members, k.val ≥ keys_per_member) :=
sorry

end safe_access_theorem_access_conditions_l670_67022


namespace sqrt_720_equals_12_sqrt_5_l670_67033

theorem sqrt_720_equals_12_sqrt_5 : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  sorry

end sqrt_720_equals_12_sqrt_5_l670_67033


namespace symmetric_difference_A_B_l670_67023

def A : Set ℤ := {1, 2}
def B : Set ℤ := {x : ℤ | |x| < 2}

def set_difference (X Y : Set ℤ) : Set ℤ := {x : ℤ | x ∈ X ∧ x ∉ Y}
def symmetric_difference (X Y : Set ℤ) : Set ℤ := (set_difference X Y) ∪ (set_difference Y X)

theorem symmetric_difference_A_B :
  symmetric_difference A B = {-1, 0, 2} := by
  sorry

end symmetric_difference_A_B_l670_67023


namespace power_multiplication_l670_67066

theorem power_multiplication (x : ℝ) : x^3 * x^2 = x^5 := by
  sorry

end power_multiplication_l670_67066


namespace river_width_l670_67035

/-- Two ferries traveling between opposite banks of a river -/
structure FerrySystem where
  /-- Width of the river -/
  width : ℝ
  /-- Distance from one bank where ferries first meet -/
  first_meeting : ℝ
  /-- Distance from the other bank where ferries second meet -/
  second_meeting : ℝ

/-- Theorem stating the width of the river given the meeting points -/
theorem river_width (fs : FerrySystem) 
    (h1 : fs.first_meeting = 700)
    (h2 : fs.second_meeting = 400) : 
    fs.width = 1700 := by
  sorry

#check river_width

end river_width_l670_67035


namespace franks_money_l670_67012

theorem franks_money (initial_money : ℚ) : 
  (3/4 : ℚ) * ((4/5 : ℚ) * initial_money) = 360 → initial_money = 600 := by
  sorry

end franks_money_l670_67012


namespace max_sum_with_lcm_gcd_l670_67015

theorem max_sum_with_lcm_gcd (a b : ℕ) 
  (h_lcm : Nat.lcm a b = 140) 
  (h_gcd : Nat.gcd a b = 5) : 
  a + b ≤ 145 := by
  sorry

end max_sum_with_lcm_gcd_l670_67015


namespace haley_trees_l670_67082

/-- The number of trees that died after the typhoon -/
def dead_trees : ℕ := 2

/-- The difference between survived trees and dead trees -/
def survival_difference : ℕ := 7

/-- The total number of trees Haley initially grew -/
def total_trees : ℕ := dead_trees + (dead_trees + survival_difference)

theorem haley_trees : total_trees = 11 := by
  sorry

end haley_trees_l670_67082


namespace unique_magnitude_quadratic_l670_67032

theorem unique_magnitude_quadratic :
  ∃! m : ℝ, ∃ w : ℂ, w^2 - 6*w + 40 = 0 ∧ Complex.abs w = m := by sorry

end unique_magnitude_quadratic_l670_67032


namespace linda_college_applications_l670_67019

def number_of_colleges (hourly_rate : ℚ) (application_fee : ℚ) (hours_worked : ℚ) : ℚ :=
  (hourly_rate * hours_worked) / application_fee

theorem linda_college_applications : number_of_colleges 10 25 15 = 6 := by
  sorry

end linda_college_applications_l670_67019


namespace count_ordered_pairs_l670_67034

theorem count_ordered_pairs : ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 = 128) (Finset.product (Finset.range 129) (Finset.range 129))).card ∧ n = 8 := by
  sorry

end count_ordered_pairs_l670_67034


namespace dice_sum_product_l670_67083

theorem dice_sum_product (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 ∧
  1 ≤ b ∧ b ≤ 6 ∧
  1 ≤ c ∧ c ≤ 6 ∧
  1 ≤ d ∧ d ≤ 6 ∧
  a * b * c * d = 180 →
  a + b + c + d ≠ 14 ∧ a + b + c + d ≠ 17 := by
sorry

end dice_sum_product_l670_67083


namespace sqrt_equation_solution_l670_67061

theorem sqrt_equation_solution :
  ∃! x : ℝ, Real.sqrt (x + 4) + Real.sqrt (x + 6) = 12 ∧ x = 4465 / 144 := by
  sorry

end sqrt_equation_solution_l670_67061


namespace evelyn_lost_bottle_caps_l670_67027

/-- The number of bottle caps Evelyn lost -/
def bottle_caps_lost (initial : ℝ) (final : ℝ) : ℝ :=
  initial - final

/-- Proof that Evelyn lost 18.0 bottle caps -/
theorem evelyn_lost_bottle_caps :
  bottle_caps_lost 63.0 45 = 18.0 := by
  sorry

end evelyn_lost_bottle_caps_l670_67027


namespace evaluate_expression_l670_67007

theorem evaluate_expression : (1023 : ℕ) * 1023 - 1022 * 1024 = 1 := by
  sorry

end evaluate_expression_l670_67007


namespace mark_can_bench_press_55_pounds_l670_67051

/-- The weight that Mark can bench press -/
def marks_bench_press (daves_weight : ℝ) : ℝ :=
  let daves_bench_press := 3 * daves_weight
  let craigs_bench_press := 0.2 * daves_bench_press
  craigs_bench_press - 50

/-- Proof that Mark can bench press 55 pounds -/
theorem mark_can_bench_press_55_pounds :
  marks_bench_press 175 = 55 := by
  sorry

end mark_can_bench_press_55_pounds_l670_67051


namespace speed_limit_violation_percentage_l670_67072

theorem speed_limit_violation_percentage
  (total_motorists : ℝ)
  (h1 : total_motorists > 0)
  (ticketed_percentage : ℝ)
  (h2 : ticketed_percentage = 40)
  (unticketed_speeders_percentage : ℝ)
  (h3 : unticketed_speeders_percentage = 20)
  : (ticketed_percentage / (100 - unticketed_speeders_percentage)) * 100 = 50 := by
  sorry

#check speed_limit_violation_percentage

end speed_limit_violation_percentage_l670_67072


namespace min_portraits_theorem_l670_67039

def min_year := 1600
def max_year := 2008
def max_age := 80

def ScientistData := {birth : ℕ // min_year ≤ birth ∧ birth ≤ max_year}

def death_year (s : ScientistData) : ℕ := s.val + (Nat.min max_age (max_year - s.val))

def product_ratio (scientists : List ScientistData) : ℚ :=
  (scientists.map death_year).prod / (scientists.map (λ s => s.val)).prod

theorem min_portraits_theorem :
  ∃ (scientists : List ScientistData),
    scientists.length = 5 ∧
    product_ratio scientists = 5/4 ∧
    ∀ (smaller_list : List ScientistData),
      smaller_list.length < 5 →
      product_ratio smaller_list < 5/4 :=
sorry

end min_portraits_theorem_l670_67039


namespace smallest_perimeter_is_nine_l670_67091

/-- A triangle with consecutive integer side lengths where the smallest side is even. -/
structure ConsecutiveIntegerTriangle where
  a : ℕ
  is_even : Even a
  satisfies_triangle_inequality : a + (a + 1) > (a + 2) ∧ a + (a + 2) > (a + 1) ∧ (a + 1) + (a + 2) > a

/-- The perimeter of a ConsecutiveIntegerTriangle. -/
def perimeter (t : ConsecutiveIntegerTriangle) : ℕ := t.a + (t.a + 1) + (t.a + 2)

/-- The smallest possible perimeter of a ConsecutiveIntegerTriangle is 9. -/
theorem smallest_perimeter_is_nine :
  ∀ t : ConsecutiveIntegerTriangle, perimeter t ≥ 9 :=
sorry

end smallest_perimeter_is_nine_l670_67091


namespace geometric_sequence_seventh_term_l670_67048

theorem geometric_sequence_seventh_term
  (a : ℝ) (r : ℝ)
  (positive_sequence : ∀ n : ℕ, a * r ^ (n - 1) > 0)
  (fourth_term : a * r^3 = 16)
  (tenth_term : a * r^9 = 2) :
  a * r^6 = 4 := by
sorry

end geometric_sequence_seventh_term_l670_67048


namespace richsWalkDistance_l670_67031

/-- Calculates the total distance Rich walks given his walking pattern --/
def richsWalk (houseToSidewalk : ℕ) (sidewalkToRoadEnd : ℕ) : ℕ :=
  let initialDistance := houseToSidewalk + sidewalkToRoadEnd
  let toIntersection := initialDistance * 2
  let toEndOfRoute := (initialDistance + toIntersection) / 2
  let oneWayDistance := initialDistance + toIntersection + toEndOfRoute
  oneWayDistance * 2

theorem richsWalkDistance :
  richsWalk 20 200 = 1980 := by
  sorry

end richsWalkDistance_l670_67031


namespace inequality_solution_l670_67004

def linear_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

def negative_for_positive (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → f x < 0

theorem inequality_solution 
  (f : ℝ → ℝ) 
  (h_linear : linear_function f) 
  (h_neg : negative_for_positive f) 
  (n : ℕ) 
  (hn : n > 0) 
  (a : ℝ) 
  (ha : a < 0) :
  ∀ x : ℝ, 
    (1 / n : ℝ) * f (a * x^2) - f x > (1 / n : ℝ) * f (a^2 * x) - f a ↔ 
      (a < -Real.sqrt n ∧ (x > n / a ∨ x < a)) ∨ 
      (a = -Real.sqrt n ∧ x ≠ -Real.sqrt n) ∨ 
      (-Real.sqrt n < a ∧ (x > a ∨ x < n / a)) :=
by sorry

end inequality_solution_l670_67004


namespace left_handed_jazz_lovers_l670_67046

/-- Represents a club with members of different handedness and music preferences -/
structure Club where
  total : ℕ
  leftHanded : ℕ
  ambidextrous : ℕ
  rightHanded : ℕ
  jazzLovers : ℕ
  rightHandedJazzDislikers : ℕ

/-- Theorem stating the number of left-handed jazz lovers in the club -/
theorem left_handed_jazz_lovers (c : Club) 
  (h1 : c.total = 30)
  (h2 : c.leftHanded = 12)
  (h3 : c.ambidextrous = 3)
  (h4 : c.rightHanded = c.total - c.leftHanded - c.ambidextrous)
  (h5 : c.jazzLovers = 20)
  (h6 : c.rightHandedJazzDislikers = 4) :
  ∃ x : ℕ, x = 6 ∧ 
    x ≤ c.leftHanded ∧ 
    x + (c.rightHanded - c.rightHandedJazzDislikers) + c.ambidextrous = c.jazzLovers :=
  sorry


end left_handed_jazz_lovers_l670_67046


namespace tangent_perpendicular_and_minimum_l670_67060

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 1 / (2 * x) + (3 / 2) * x + 1

theorem tangent_perpendicular_and_minimum (a : ℝ) :
  (∀ x, x > 0 → HasDerivAt (f a) ((a / x) - 1 / (2 * x^2) + 3 / 2) x) →
  (HasDerivAt (f a) 0 1) →
  a = -1 ∧
  ∀ x > 0, f (-1) x ≥ 3 ∧ f (-1) 1 = 3 :=
by sorry

end tangent_perpendicular_and_minimum_l670_67060


namespace square_area_perimeter_ratio_l670_67059

theorem square_area_perimeter_ratio : 
  ∀ (a b : ℝ), a > 0 → b > 0 → (a^2 / b^2 = 49 / 64) → (4*a / (4*b) = 7 / 8) := by
  sorry

end square_area_perimeter_ratio_l670_67059


namespace largest_gcd_of_ten_numbers_summing_to_1001_l670_67038

theorem largest_gcd_of_ten_numbers_summing_to_1001 :
  ∃ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℕ),
    a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = 1001 ∧
    (∀ d : ℕ, d > 0 → d ∣ a₁ ∧ d ∣ a₂ ∧ d ∣ a₃ ∧ d ∣ a₄ ∧ d ∣ a₅ ∧
                      d ∣ a₆ ∧ d ∣ a₇ ∧ d ∣ a₈ ∧ d ∣ a₉ ∧ d ∣ a₁₀ → d ≤ 91) ∧
    91 ∣ a₁ ∧ 91 ∣ a₂ ∧ 91 ∣ a₃ ∧ 91 ∣ a₄ ∧ 91 ∣ a₅ ∧
    91 ∣ a₆ ∧ 91 ∣ a₇ ∧ 91 ∣ a₈ ∧ 91 ∣ a₉ ∧ 91 ∣ a₁₀ := by
  sorry

end largest_gcd_of_ten_numbers_summing_to_1001_l670_67038


namespace candy_distribution_totals_l670_67087

/-- Represents the number of candies each child has -/
structure CandyDistribution where
  vitya : Nat
  masha : Nat
  sasha : Nat

/-- Checks if a candy distribution satisfies the given conditions -/
def isValidDistribution (d : CandyDistribution) : Prop :=
  d.vitya = 5 ∧ d.masha < d.vitya ∧ d.sasha = d.vitya + d.masha

/-- Calculates the total number of candies for a distribution -/
def totalCandies (d : CandyDistribution) : Nat :=
  d.vitya + d.masha + d.sasha

/-- The set of possible total numbers of candies -/
def possibleTotals : Set Nat := {18, 16, 14, 12}

/-- Theorem stating that the possible total numbers of candies are 18, 16, 14, and 12 -/
theorem candy_distribution_totals :
  ∀ d : CandyDistribution, isValidDistribution d →
    totalCandies d ∈ possibleTotals := by
  sorry

end candy_distribution_totals_l670_67087


namespace cone_volume_from_cylinder_l670_67043

/-- Given a cylinder with volume 72π cm³ and a cone with the same radius as the cylinder 
    and half its height, the volume of the cone is 12π cm³ -/
theorem cone_volume_from_cylinder (r h : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  π * r^2 * h = 72 * π → (1/3) * π * r^2 * (h/2) = 12 * π := by
  sorry

end cone_volume_from_cylinder_l670_67043


namespace students_exceed_pets_l670_67075

/-- The number of third-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 22

/-- The number of rabbits in each classroom -/
def rabbits_per_classroom : ℕ := 3

/-- The number of hamsters in each classroom -/
def hamsters_per_classroom : ℕ := 5

/-- The theorem stating the difference between students and pets -/
theorem students_exceed_pets : 
  (num_classrooms * students_per_classroom) - 
  (num_classrooms * (rabbits_per_classroom + hamsters_per_classroom)) = 70 := by
  sorry

end students_exceed_pets_l670_67075


namespace right_triangle_hypotenuse_l670_67077

theorem right_triangle_hypotenuse (leg : ℝ) (angle : ℝ) : 
  leg = 15 → angle = 45 → ∃ (hypotenuse : ℝ), hypotenuse = 15 * Real.sqrt 2 :=
by
  sorry

end right_triangle_hypotenuse_l670_67077


namespace integer_set_property_l670_67052

theorem integer_set_property (n : ℕ+) :
  ∃ (S : Finset ℤ), Finset.card S = n ∧
    ∀ (a b : ℤ), a ∈ S → b ∈ S → a ≠ b →
      ∃ (k : ℤ), a * b = k * (a - b)^2 :=
sorry

end integer_set_property_l670_67052


namespace log_sum_equality_l670_67076

theorem log_sum_equality : Real.log 0.01 / Real.log 10 + Real.log 16 / Real.log 2 = 2 := by
  sorry

end log_sum_equality_l670_67076


namespace equation_has_solution_in_interval_l670_67095

theorem equation_has_solution_in_interval : 
  ∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^3 = 2^x := by sorry

end equation_has_solution_in_interval_l670_67095


namespace prime_factor_difference_l670_67003

theorem prime_factor_difference (a b : ℕ) : 
  Prime a → Prime b → b > a → 
  456456 = 2^3 * a * 7 * 11 * 13 * b → 
  b - a = 16 := by
sorry

end prime_factor_difference_l670_67003


namespace range_of_t_l670_67080

theorem range_of_t (x y a : ℝ) 
  (eq1 : x + 3 * y + a = 4)
  (eq2 : x - y - 3 * a = 0)
  (bounds : -1 ≤ a ∧ a ≤ 1) :
  let t := x + y
  1 ≤ t ∧ t ≤ 3 := by sorry

end range_of_t_l670_67080


namespace nonnegative_inequality_l670_67040

theorem nonnegative_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a * (a - b) * (a - 2 * b) + b * (b - c) * (b - 2 * c) + c * (c - a) * (c - 2 * a) ≥ 0 := by
  sorry

end nonnegative_inequality_l670_67040


namespace difference_of_squares_303_297_l670_67092

theorem difference_of_squares_303_297 : 303^2 - 297^2 = 3600 := by
  sorry

end difference_of_squares_303_297_l670_67092


namespace intersection_distance_l670_67055

/-- The distance between the intersection points of y = -2 and y = 3x^2 + 2x - 5 -/
theorem intersection_distance : 
  let f (x : ℝ) := 3 * x^2 + 2 * x - 5
  let y := -2
  let roots := {x : ℝ | f x = y}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ roots ∧ x₂ ∈ roots ∧ x₁ ≠ x₂ ∧ |x₁ - x₂| = 2 * Real.sqrt 10 / 3 :=
by sorry

end intersection_distance_l670_67055


namespace theta_range_l670_67054

theorem theta_range (θ : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, x^2 * Real.cos θ - x * (1 - x) + (1 - x^2) * Real.sin θ > 0) →
  ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 6 < θ ∧ θ < 2 * k * Real.pi + Real.pi / 2 :=
by sorry

end theta_range_l670_67054


namespace remainder_problem_l670_67028

theorem remainder_problem (x : ℤ) : x % 95 = 31 → x % 19 = 12 := by
  sorry

end remainder_problem_l670_67028


namespace dave_derek_money_difference_l670_67056

/-- Calculates the difference between Dave's and Derek's remaining money after expenses -/
def moneyDifference (derekInitial : ℕ) (derekExpenses : List ℕ) (daveInitial : ℕ) (daveExpenses : List ℕ) : ℕ :=
  let derekRemaining := derekInitial - derekExpenses.sum
  let daveRemaining := daveInitial - daveExpenses.sum
  daveRemaining - derekRemaining

/-- Proves that Dave has $20 more left than Derek after expenses -/
theorem dave_derek_money_difference :
  moneyDifference 40 [14, 11, 5, 8] 50 [7, 12, 9] = 20 := by
  sorry

#eval moneyDifference 40 [14, 11, 5, 8] 50 [7, 12, 9]

end dave_derek_money_difference_l670_67056


namespace max_points_top_four_l670_67079

/-- Represents a tournament with 8 teams -/
structure Tournament :=
  (teams : Fin 8)
  (games : Fin 8 → Fin 8 → Nat)
  (points : Fin 8 → Nat)

/-- The scoring system for the tournament -/
def score (result : Nat) : Nat :=
  match result with
  | 0 => 3  -- win
  | 1 => 1  -- draw
  | _ => 0  -- loss

/-- The theorem stating the maximum possible points for the top four teams -/
theorem max_points_top_four (t : Tournament) : 
  ∃ (a b c d : Fin 8), 
    (∀ i : Fin 8, t.points i ≤ t.points a) ∧
    (t.points a = t.points b) ∧
    (t.points b = t.points c) ∧
    (t.points c = t.points d) ∧
    (t.points a ≤ 33) :=
sorry

end max_points_top_four_l670_67079


namespace race_head_start_l670_67097

/-- Proves that if A's speed is 32/27 times B's speed, then A needs to give B
    a head start of 5/32 of the race length for the race to end in a dead heat. -/
theorem race_head_start (L : ℝ) (Va Vb : ℝ) (h : Va = (32/27) * Vb) :
  (L / Va = (L - (5/32) * L) / Vb) := by
  sorry

end race_head_start_l670_67097


namespace braden_final_amount_l670_67029

/-- Calculates the final amount in Braden's money box after winning a bet -/
def final_amount (initial_amount : ℕ) (bet_multiplier : ℕ) : ℕ :=
  initial_amount + bet_multiplier * initial_amount

/-- Theorem stating that given the initial conditions, Braden's final amount is $1200 -/
theorem braden_final_amount :
  let initial_amount : ℕ := 400
  let bet_multiplier : ℕ := 2
  final_amount initial_amount bet_multiplier = 1200 := by sorry

end braden_final_amount_l670_67029


namespace fair_haired_employees_percentage_l670_67030

theorem fair_haired_employees_percentage :
  -- Define the total number of employees
  ∀ (total_employees : ℕ),
  total_employees > 0 →
  -- Define the number of women with fair hair
  ∀ (women_fair_hair : ℕ),
  women_fair_hair = (28 * total_employees) / 100 →
  -- Define the number of fair-haired employees
  ∀ (fair_haired_employees : ℕ),
  women_fair_hair = (40 * fair_haired_employees) / 100 →
  -- The percentage of employees with fair hair is 70%
  (fair_haired_employees : ℚ) / total_employees = 70 / 100 :=
by
  sorry

end fair_haired_employees_percentage_l670_67030


namespace min_value_C_squared_minus_D_squared_l670_67005

theorem min_value_C_squared_minus_D_squared
  (x y z : ℝ)
  (hx : x ≥ 0)
  (hy : y ≥ 0)
  (hz : z ≥ 0)
  (C : ℝ := Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 11))
  (D : ℝ := Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2)) :
  C^2 - D^2 ≥ 36 :=
by sorry

end min_value_C_squared_minus_D_squared_l670_67005


namespace existence_of_divalent_radical_with_bounded_growth_l670_67044

/-- A set of positive integers is a divalent radical if any sufficiently large positive integer
    can be expressed as the sum of two elements in the set. -/
def IsDivalentRadical (A : Set ℕ+) : Prop :=
  ∃ N : ℕ+, ∀ n : ℕ+, n ≥ N → ∃ a b : ℕ+, a ∈ A ∧ b ∈ A ∧ (a : ℕ) + b = n

/-- A(x) is the set of all elements in A that do not exceed x -/
def ASubset (A : Set ℕ+) (x : ℝ) : Set ℕ+ :=
  {a ∈ A | (a : ℝ) ≤ x}

theorem existence_of_divalent_radical_with_bounded_growth :
  ∃ (A : Set ℕ+) (C : ℝ), A.Nonempty ∧ IsDivalentRadical A ∧
    ∀ x : ℝ, x ≥ 1 → (ASubset A x).ncard ≤ C * Real.sqrt x :=
sorry

end existence_of_divalent_radical_with_bounded_growth_l670_67044


namespace acute_triangle_sine_cosine_inequality_l670_67073

theorem acute_triangle_sine_cosine_inequality 
  (A B C : Real) 
  (h_acute : A ∈ Set.Ioo 0 (π/2) ∧ B ∈ Set.Ioo 0 (π/2) ∧ C ∈ Set.Ioo 0 (π/2)) 
  (h_sum : A + B + C = π) : 
  Real.sin A + Real.sin B > Real.cos A + Real.cos B + Real.cos C := by
  sorry

end acute_triangle_sine_cosine_inequality_l670_67073


namespace fewer_baseball_cards_l670_67084

theorem fewer_baseball_cards (hockey football baseball : ℕ) 
  (h1 : baseball < football)
  (h2 : football = 4 * hockey)
  (h3 : hockey = 200)
  (h4 : baseball + football + hockey = 1750) :
  football - baseball = 50 := by
sorry

end fewer_baseball_cards_l670_67084


namespace even_function_implies_a_zero_l670_67000

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3

theorem even_function_implies_a_zero :
  (∀ x, f a x = f a (-x)) → a = 0 := by
  sorry

end even_function_implies_a_zero_l670_67000


namespace daycare_peas_preference_l670_67053

theorem daycare_peas_preference (total : ℕ) (peas carrots corn : ℕ) : 
  total > 0 ∧
  carrots = 9 ∧
  corn = 5 ∧
  corn = (25 : ℕ) * total / 100 ∧
  total = peas + carrots + corn →
  peas = 6 := by
  sorry

end daycare_peas_preference_l670_67053


namespace vacation_photos_count_l670_67065

/-- The number of photos Alyssa took on vacation -/
def total_photos : ℕ := 100

/-- The total number of pages in the album -/
def total_pages : ℕ := 30

/-- The number of photos that can be placed on each of the first 10 pages -/
def photos_per_page_first : ℕ := 3

/-- The number of photos that can be placed on each of the next 10 pages -/
def photos_per_page_second : ℕ := 4

/-- The number of photos that can be placed on each of the remaining pages -/
def photos_per_page_last : ℕ := 3

/-- The number of pages in the first section -/
def pages_first_section : ℕ := 10

/-- The number of pages in the second section -/
def pages_second_section : ℕ := 10

theorem vacation_photos_count : 
  total_photos = 
    photos_per_page_first * pages_first_section + 
    photos_per_page_second * pages_second_section + 
    photos_per_page_last * (total_pages - pages_first_section - pages_second_section) :=
by sorry

end vacation_photos_count_l670_67065


namespace log_difference_equals_negative_one_l670_67068

theorem log_difference_equals_negative_one :
  ∀ (log : ℝ → ℝ → ℝ),
    (∀ (a N : ℝ), a > 0 → a ≠ 1 → ∃ b, N = a^b → log a N = b) →
    9 = 3^2 →
    125 = 5^3 →
    log 3 9 - log 5 125 = -1 := by
  sorry

end log_difference_equals_negative_one_l670_67068


namespace john_light_bulbs_left_l670_67018

/-- The number of light bulbs John has left after using some and giving away half of the remainder --/
def lightBulbsLeft (initial : ℕ) (used : ℕ) : ℕ :=
  let remaining := initial - used
  remaining - remaining / 2

/-- Theorem stating that John has 12 light bulbs left --/
theorem john_light_bulbs_left :
  lightBulbsLeft 40 16 = 12 := by
  sorry

end john_light_bulbs_left_l670_67018


namespace intersection_condition_l670_67014

-- Define the sets M and N
def M : Set ℝ := {x | 2 * x + 1 < 3}
def N (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem intersection_condition (a : ℝ) : M ∩ N a = N a → a ≤ 1 := by
  sorry

end intersection_condition_l670_67014


namespace factorization_of_quadratic_l670_67057

theorem factorization_of_quadratic (x : ℝ) : 2 * x^2 - 4 * x = 2 * x * (x - 2) := by
  sorry

end factorization_of_quadratic_l670_67057


namespace girls_in_class_l670_67064

theorem girls_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (num_girls : ℕ) :
  total = 35 →
  ratio_girls = 3 →
  ratio_boys = 4 →
  ratio_girls + ratio_boys = num_girls + (total - num_girls) →
  num_girls * ratio_boys = (total - num_girls) * ratio_girls →
  num_girls = 15 := by
sorry

end girls_in_class_l670_67064


namespace pool_attendance_difference_l670_67088

theorem pool_attendance_difference (total : ℕ) (day1 : ℕ) (day3 : ℕ) 
  (h1 : total = 246)
  (h2 : day1 = 79)
  (h3 : day3 = 120)
  (h4 : total = day1 + day3 + (total - day1 - day3)) :
  (total - day1 - day3) - day3 = 47 := by
  sorry

end pool_attendance_difference_l670_67088


namespace max_valid_sequence_length_l670_67011

/-- A sequence of integers satisfying the given property -/
def ValidSequence (x : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ i j, 1 ≤ i → i ≤ j → j ≤ n → i + j ≤ n →
    (3 ∣ x i - x j) → (3 ∣ x (i + j) + x i + x j + 1)

/-- The maximum length of a valid sequence -/
def MaxValidSequenceLength : ℕ := 8

/-- Theorem stating that the maximum length of a valid sequence is 8 -/
theorem max_valid_sequence_length :
  (∃ x, ValidSequence x MaxValidSequenceLength) ∧
  (∀ n > MaxValidSequenceLength, ¬∃ x, ValidSequence x n) :=
sorry

end max_valid_sequence_length_l670_67011


namespace fifth_match_goals_is_five_l670_67090

/-- The number of goals scored in the fifth match -/
def fifth_match_goals : ℕ := 5

/-- The total number of matches played -/
def total_matches : ℕ := 5

/-- The increase in average goals after the fifth match -/
def average_increase : ℚ := 1/5

/-- The total number of goals in all matches -/
def total_goals : ℕ := 21

/-- Theorem stating that the number of goals scored in the fifth match is 5 -/
theorem fifth_match_goals_is_five :
  fifth_match_goals = 5 ∧
  (fifth_match_goals : ℚ) + (total_goals - fifth_match_goals) = total_goals ∧
  (total_goals : ℚ) / total_matches = 
    ((total_goals - fifth_match_goals) : ℚ) / (total_matches - 1) + average_increase :=
by sorry

end fifth_match_goals_is_five_l670_67090


namespace total_vegetables_collected_schoolchildren_vegetable_collection_l670_67041

/-- Represents the amount of vegetables collected by each grade -/
structure VegetableCollection where
  fourth_cabbage : ℕ
  fourth_carrots : ℕ
  fifth_cucumbers : ℕ
  sixth_cucumbers : ℕ
  sixth_onions : ℕ

/-- Theorem stating the total amount of vegetables collected -/
theorem total_vegetables_collected (vc : VegetableCollection) : ℕ :=
  vc.fourth_cabbage + vc.fourth_carrots + vc.fifth_cucumbers + vc.sixth_cucumbers + vc.sixth_onions

/-- Main theorem proving the total amount of vegetables collected is 49 centners -/
theorem schoolchildren_vegetable_collection : 
  ∃ (vc : VegetableCollection), 
    vc.fourth_cabbage = 18 ∧ 
    vc.fourth_carrots = vc.sixth_onions ∧
    vc.fifth_cucumbers < vc.sixth_cucumbers ∧
    vc.fifth_cucumbers > vc.fourth_carrots ∧
    vc.sixth_onions = 7 ∧
    vc.sixth_cucumbers = vc.fourth_cabbage / 2 ∧
    total_vegetables_collected vc = 49 := by
  sorry

end total_vegetables_collected_schoolchildren_vegetable_collection_l670_67041
