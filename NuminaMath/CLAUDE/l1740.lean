import Mathlib

namespace NUMINAMATH_CALUDE_gcf_of_lcms_l1740_174094

def GCF (a b : ℕ) : ℕ := Nat.gcd a b

def LCM (c d : ℕ) : ℕ := Nat.lcm c d

theorem gcf_of_lcms : GCF (LCM 18 30) (LCM 10 45) = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_lcms_l1740_174094


namespace NUMINAMATH_CALUDE_probability_at_least_one_of_each_color_l1740_174084

theorem probability_at_least_one_of_each_color (white red yellow drawn : ℕ) 
  (hw : white = 5) (hr : red = 4) (hy : yellow = 3) (hd : drawn = 4) :
  let total := white + red + yellow
  let total_ways := Nat.choose total drawn
  let favorable_ways := 
    Nat.choose white 2 * Nat.choose red 1 * Nat.choose yellow 1 +
    Nat.choose white 1 * Nat.choose red 2 * Nat.choose yellow 1 +
    Nat.choose white 1 * Nat.choose red 1 * Nat.choose yellow 2
  (favorable_ways : ℚ) / total_ways = 6 / 11 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_of_each_color_l1740_174084


namespace NUMINAMATH_CALUDE_cathys_total_money_l1740_174062

def cathys_money (initial_balance dad_contribution : ℕ) : ℕ :=
  initial_balance + dad_contribution + 2 * dad_contribution

theorem cathys_total_money :
  cathys_money 12 25 = 87 := by
  sorry

end NUMINAMATH_CALUDE_cathys_total_money_l1740_174062


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_1_to_18_not_19_20_l1740_174010

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def divisible_up_to (n m : ℕ) : Prop := ∀ i : ℕ, 1 ≤ i → i ≤ m → is_divisible n i

theorem smallest_number_divisible_by_1_to_18_not_19_20 :
  ∃ n : ℕ, 
    n > 0 ∧
    divisible_up_to n 18 ∧
    ¬(is_divisible n 19) ∧
    ¬(is_divisible n 20) ∧
    ∀ m : ℕ, m > 0 → divisible_up_to m 18 → ¬(is_divisible m 19) → ¬(is_divisible m 20) → n ≤ m :=
by
  sorry

#eval 12252240

end NUMINAMATH_CALUDE_smallest_number_divisible_by_1_to_18_not_19_20_l1740_174010


namespace NUMINAMATH_CALUDE_maximum_mark_calculation_l1740_174056

def passing_threshold (max_mark : ℝ) : ℝ := 0.33 * max_mark

theorem maximum_mark_calculation (student_marks : ℝ) (failed_by : ℝ) 
  (h1 : student_marks = 125)
  (h2 : failed_by = 40)
  (h3 : passing_threshold (student_marks + failed_by) = student_marks + failed_by) :
  student_marks + failed_by = 500 := by
  sorry

end NUMINAMATH_CALUDE_maximum_mark_calculation_l1740_174056


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l1740_174043

theorem floor_ceil_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(30.95 : ℝ)⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l1740_174043


namespace NUMINAMATH_CALUDE_problem_solution_l1740_174067

theorem problem_solution (y₁ y₂ y₃ y₄ y₅ y₆ y₇ : ℝ) 
  (h₁ : y₁ + 3*y₂ + 5*y₃ + 7*y₄ + 9*y₅ + 11*y₆ + 13*y₇ = 0)
  (h₂ : 3*y₁ + 5*y₂ + 7*y₃ + 9*y₄ + 11*y₅ + 13*y₆ + 15*y₇ = 10)
  (h₃ : 5*y₁ + 7*y₂ + 9*y₃ + 11*y₄ + 13*y₅ + 15*y₆ + 17*y₇ = 104) :
  7*y₁ + 9*y₂ + 11*y₃ + 13*y₄ + 15*y₅ + 17*y₆ + 19*y₇ = 282 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1740_174067


namespace NUMINAMATH_CALUDE_reciprocal_of_sum_divided_l1740_174026

theorem reciprocal_of_sum_divided : 
  (((1 : ℚ) / 4 + (1 : ℚ) / 5) / ((1 : ℚ) / 3))⁻¹ = 20 / 27 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_sum_divided_l1740_174026


namespace NUMINAMATH_CALUDE_keychain_arrangements_l1740_174009

def number_of_keychains : ℕ := 5

def total_permutations (n : ℕ) : ℕ := n.factorial

def adjacent_permutations (n : ℕ) : ℕ := 2 * (n - 1).factorial

theorem keychain_arrangements :
  total_permutations number_of_keychains - adjacent_permutations number_of_keychains = 72 :=
sorry

end NUMINAMATH_CALUDE_keychain_arrangements_l1740_174009


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1740_174023

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {0, 1, 2}

theorem intersection_of_M_and_N : M ∩ N = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1740_174023


namespace NUMINAMATH_CALUDE_spurs_rockets_basketballs_l1740_174021

/-- The number of basketballs for two teams given their player counts and basketballs per player -/
def combined_basketballs (x y z : ℕ) : ℕ := x * z + y * z

/-- Theorem: The combined number of basketballs for the Spurs and Rockets is 440 -/
theorem spurs_rockets_basketballs :
  let x : ℕ := 22  -- number of Spurs players
  let y : ℕ := 18  -- number of Rockets players
  let z : ℕ := 11  -- number of basketballs per player
  combined_basketballs x y z = 440 := by
  sorry

end NUMINAMATH_CALUDE_spurs_rockets_basketballs_l1740_174021


namespace NUMINAMATH_CALUDE_binary_conversion_and_subtraction_l1740_174044

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binaryToDecimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 101101₂ -/
def binaryNumber : List Bool := [true, false, true, true, false, true]

/-- The main theorem to prove -/
theorem binary_conversion_and_subtraction :
  (binaryToDecimal binaryNumber) - 5 = 40 := by sorry

end NUMINAMATH_CALUDE_binary_conversion_and_subtraction_l1740_174044


namespace NUMINAMATH_CALUDE_can_identification_theorem_l1740_174053

-- Define the type for our weighing results
inductive WeighResult
| Heavy
| Medium
| Light

def WeighSequence := List WeighResult

theorem can_identification_theorem (n : ℕ) (weights : Fin n → ℝ) 
  (h_n : n = 80) (h_distinct : ∀ i j : Fin n, i ≠ j → weights i ≠ weights j) :
  (∃ (f : Fin n → WeighSequence), 
    (∀ seq, (∃ i, f i = seq) → seq.length ≤ 4) ∧ 
    (∀ i j : Fin n, i ≠ j → f i ≠ f j)) ∧ 
  (¬ ∃ (f : Fin n → WeighSequence), 
    (∀ seq, (∃ i, f i = seq) → seq.length ≤ 3) ∧ 
    (∀ i j : Fin n, i ≠ j → f i ≠ f j)) := by
  sorry


end NUMINAMATH_CALUDE_can_identification_theorem_l1740_174053


namespace NUMINAMATH_CALUDE_n_div_f_n_equals_5_for_625_n_div_f_n_equals_1_solutions_l1740_174088

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Defines the function f as described in the problem -/
def f (n : ThreeDigitNumber) : Nat :=
  let a := n.hundreds
  let b := n.tens
  let c := n.ones
  a + b + c + a * b + b * c + c * a + a * b * c

theorem n_div_f_n_equals_5_for_625 :
  let n : ThreeDigitNumber := ⟨6, 2, 5, by simp, by simp, by simp⟩
  (n.toNat : ℚ) / f n = 5 := by sorry

theorem n_div_f_n_equals_1_solutions :
  {n : ThreeDigitNumber | (n.toNat : ℚ) / f n = 1} =
  {⟨1, 9, 9, by simp, by simp, by simp⟩,
   ⟨2, 9, 9, by simp, by simp, by simp⟩,
   ⟨3, 9, 9, by simp, by simp, by simp⟩,
   ⟨4, 9, 9, by simp, by simp, by simp⟩,
   ⟨5, 9, 9, by simp, by simp, by simp⟩,
   ⟨6, 9, 9, by simp, by simp, by simp⟩,
   ⟨7, 9, 9, by simp, by simp, by simp⟩,
   ⟨8, 9, 9, by simp, by simp, by simp⟩,
   ⟨9, 9, 9, by simp, by simp, by simp⟩} := by sorry

end NUMINAMATH_CALUDE_n_div_f_n_equals_5_for_625_n_div_f_n_equals_1_solutions_l1740_174088


namespace NUMINAMATH_CALUDE_circle_chord_distance_l1740_174041

theorem circle_chord_distance (r : ℝ) (AB AC BC : ℝ) : 
  r = 10 →
  AB = 2 * r →
  AC = 12 →
  AB^2 = AC^2 + BC^2 →
  BC = 16 := by
sorry

end NUMINAMATH_CALUDE_circle_chord_distance_l1740_174041


namespace NUMINAMATH_CALUDE_apple_count_in_second_group_l1740_174032

/-- The cost of an apple in dollars -/
def apple_cost : ℚ := 21/100

/-- The cost of an orange in dollars -/
def orange_cost : ℚ := 17/100

/-- The number of apples in the second group -/
def x : ℕ := 2

theorem apple_count_in_second_group :
  (6 * apple_cost + 3 * orange_cost = 177/100) →
  (↑x * apple_cost + 5 * orange_cost = 127/100) →
  (apple_cost = 21/100) →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_apple_count_in_second_group_l1740_174032


namespace NUMINAMATH_CALUDE_line_not_in_fourth_quadrant_l1740_174052

/-- A line in the 2D plane represented by the equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The fourth quadrant of the 2D plane -/
def fourth_quadrant : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 < 0}

/-- Theorem: A line Ax + By + C = 0 where AB < 0 and BC < 0 does not pass through the fourth quadrant -/
theorem line_not_in_fourth_quadrant (l : Line) 
    (h1 : l.A * l.B < 0) 
    (h2 : l.B * l.C < 0) : 
    ∀ p ∈ fourth_quadrant, l.A * p.1 + l.B * p.2 + l.C ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_line_not_in_fourth_quadrant_l1740_174052


namespace NUMINAMATH_CALUDE_polynomial_division_l1740_174090

theorem polynomial_division (x : ℝ) :
  x^6 + 3 = (x - 2) * (x^5 + 2*x^4 + 4*x^3 + 8*x^2 + 16*x + 32) + 67 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l1740_174090


namespace NUMINAMATH_CALUDE_new_person_weight_l1740_174099

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem new_person_weight (initial_people : ℕ) (replaced_weight : ℕ) (avg_increase : ℕ) (total_weight : ℕ) :
  initial_people = 4 →
  replaced_weight = 70 →
  avg_increase = 3 →
  total_weight = 390 →
  ∃ (new_weight : ℕ), 
    is_prime new_weight ∧
    new_weight = total_weight - (initial_people * replaced_weight + initial_people * avg_increase) :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1740_174099


namespace NUMINAMATH_CALUDE_investment_problem_l1740_174060

/-- Investment problem -/
theorem investment_problem (x y : ℕ) (profit_ratio : Rat) (y_investment : ℕ) : 
  profit_ratio = 2 / 6 →
  y_investment = 15000 →
  x = 5000 :=
by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l1740_174060


namespace NUMINAMATH_CALUDE_remainder_problem_l1740_174015

theorem remainder_problem (k : ℕ) 
  (h1 : k > 0) 
  (h2 : k < 41) 
  (h3 : k % 5 = 2) 
  (h4 : k % 6 = 5) : 
  k % 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1740_174015


namespace NUMINAMATH_CALUDE_equality_condition_l1740_174020

theorem equality_condition (x y z : ℝ) : 
  x + y * z = (x + y) * (x + z) ↔ x + y + z = 0 := by
  sorry

end NUMINAMATH_CALUDE_equality_condition_l1740_174020


namespace NUMINAMATH_CALUDE_max_value_of_objective_function_l1740_174070

-- Define the constraint set
def ConstraintSet (x y : ℝ) : Prop :=
  y ≥ x ∧ x + 3 * y ≤ 4 ∧ x ≥ -2

-- Define the objective function
def ObjectiveFunction (x y : ℝ) : ℝ :=
  |x - 3 * y|

-- Theorem statement
theorem max_value_of_objective_function :
  ∃ (max : ℝ), max = 4 ∧
  ∀ (x y : ℝ), ConstraintSet x y →
  ObjectiveFunction x y ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_objective_function_l1740_174070


namespace NUMINAMATH_CALUDE_min_sum_of_digits_of_sum_l1740_174071

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define the theorem
theorem min_sum_of_digits_of_sum (A B : ℕ) 
  (hA : sumOfDigits A = 59) 
  (hB : sumOfDigits B = 77) : 
  ∃ (C : ℕ), C = A + B ∧ sumOfDigits C = 1 ∧ 
  ∀ (D : ℕ), D = A + B → sumOfDigits D ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_digits_of_sum_l1740_174071


namespace NUMINAMATH_CALUDE_valid_outfit_choices_l1740_174083

def num_items : ℕ := 4
def num_colors : ℕ := 8

def total_combinations : ℕ := num_colors ^ num_items

def same_color_two_items : ℕ := (num_items.choose 2) * num_colors * (num_colors - 1) * (num_colors - 2)

def same_color_three_items : ℕ := (num_items.choose 3) * num_colors * (num_colors - 1)

def same_color_four_items : ℕ := num_colors

def two_pairs_same_color : ℕ := (num_items.choose 2) * 1 * num_colors * (num_colors - 1)

def invalid_combinations : ℕ := same_color_two_items + same_color_three_items + same_color_four_items + two_pairs_same_color

theorem valid_outfit_choices : 
  total_combinations - invalid_combinations = 1512 :=
sorry

end NUMINAMATH_CALUDE_valid_outfit_choices_l1740_174083


namespace NUMINAMATH_CALUDE_number_puzzle_l1740_174042

theorem number_puzzle : ∃! x : ℝ, 3 * (2 * x + 9) = 69 := by sorry

end NUMINAMATH_CALUDE_number_puzzle_l1740_174042


namespace NUMINAMATH_CALUDE_abs_neg_sqrt_16_plus_9_l1740_174037

theorem abs_neg_sqrt_16_plus_9 : |-(Real.sqrt 16) + 9| = 5 := by sorry

end NUMINAMATH_CALUDE_abs_neg_sqrt_16_plus_9_l1740_174037


namespace NUMINAMATH_CALUDE_mango_purchase_l1740_174059

/-- The amount of grapes purchased in kg -/
def grapes : ℕ := 8

/-- The price of grapes per kg -/
def grape_price : ℕ := 70

/-- The price of mangoes per kg -/
def mango_price : ℕ := 60

/-- The total amount paid to the shopkeeper -/
def total_paid : ℕ := 1100

/-- The amount of mangoes purchased in kg -/
def mangoes : ℕ := (total_paid - grapes * grape_price) / mango_price

theorem mango_purchase : mangoes = 9 := by
  sorry

end NUMINAMATH_CALUDE_mango_purchase_l1740_174059


namespace NUMINAMATH_CALUDE_prob_one_defective_is_half_l1740_174012

/-- Represents the total number of items -/
def total_items : ℕ := 4

/-- Represents the number of genuine items -/
def genuine_items : ℕ := 3

/-- Represents the number of defective items -/
def defective_items : ℕ := 1

/-- Represents the number of items selected -/
def items_selected : ℕ := 2

/-- Calculates the number of ways to select k items from n items -/
def combinations (n k : ℕ) : ℕ := Nat.choose n k

/-- Calculates the probability of selecting exactly one defective item -/
def prob_one_defective : ℚ :=
  (combinations defective_items 1 * combinations genuine_items 1) /
  combinations total_items items_selected

theorem prob_one_defective_is_half :
  prob_one_defective = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_prob_one_defective_is_half_l1740_174012


namespace NUMINAMATH_CALUDE_negation_equivalence_l1740_174058

theorem negation_equivalence :
  (¬ ∃ x : ℝ, (|x| + |x - 1| < 2)) ↔ (∀ x : ℝ, |x| + |x - 1| ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1740_174058


namespace NUMINAMATH_CALUDE_cups_per_serving_l1740_174031

/-- Given a recipe that requires 18.0 servings of cereal and 36 cups in total,
    prove that each serving consists of 2 cups. -/
theorem cups_per_serving (servings : Real) (total_cups : Nat) 
    (h1 : servings = 18.0) (h2 : total_cups = 36) : 
    (total_cups : Real) / servings = 2 := by
  sorry

end NUMINAMATH_CALUDE_cups_per_serving_l1740_174031


namespace NUMINAMATH_CALUDE_solution_in_interval_l1740_174038

theorem solution_in_interval (x₀ : ℝ) (h : Real.exp x₀ + x₀ = 2) : 0 < x₀ ∧ x₀ < 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_in_interval_l1740_174038


namespace NUMINAMATH_CALUDE_total_paintable_area_l1740_174074

def num_bedrooms : ℕ := 4
def room_length : ℝ := 14
def room_width : ℝ := 11
def room_height : ℝ := 9
def unpaintable_area : ℝ := 70

def wall_area (length width height : ℝ) : ℝ :=
  2 * (length * height + width * height)

def paintable_area (total_area unpaintable_area : ℝ) : ℝ :=
  total_area - unpaintable_area

theorem total_paintable_area :
  (num_bedrooms : ℝ) * paintable_area (wall_area room_length room_width room_height) unpaintable_area = 1520 := by
  sorry

end NUMINAMATH_CALUDE_total_paintable_area_l1740_174074


namespace NUMINAMATH_CALUDE_max_value_complex_l1740_174089

theorem max_value_complex (z : ℂ) (h : Complex.abs z = 2) :
  Complex.abs ((z - 2)^3 * (z + 1)) ≤ 8 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_max_value_complex_l1740_174089


namespace NUMINAMATH_CALUDE_expression_evaluation_l1740_174035

theorem expression_evaluation : 200 * (200 - 8) - (200 * 200 + 8) = -1608 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1740_174035


namespace NUMINAMATH_CALUDE_train_length_calculation_l1740_174051

/-- Calculates the length of a train given its speed, platform length, and time to cross the platform. -/
theorem train_length_calculation (speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  speed = 72 → platform_length = 250 → crossing_time = 30 →
  (speed * (5/18) * crossing_time) - platform_length = 350 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1740_174051


namespace NUMINAMATH_CALUDE_multiply_whole_and_mixed_number_l1740_174072

theorem multiply_whole_and_mixed_number : 8 * (9 + 2/5) = 75 + 1/5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_whole_and_mixed_number_l1740_174072


namespace NUMINAMATH_CALUDE_inequality_range_l1740_174030

theorem inequality_range (a : ℝ) : 
  (∀ x ∈ Set.Icc (-5 : ℝ) 0, x^2 + 2*x - 3 + a ≤ 0) ↔ a ∈ Set.Iic (-12 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l1740_174030


namespace NUMINAMATH_CALUDE_min_value_of_squares_l1740_174061

theorem min_value_of_squares (a b t : ℝ) (h : 2 * a + b = 2 * t) :
  ∃ (min : ℝ), min = (4 * t^2) / 5 ∧ ∀ (x y : ℝ), 2 * x + y = 2 * t → x^2 + y^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_squares_l1740_174061


namespace NUMINAMATH_CALUDE_alpha_beta_sum_l1740_174079

theorem alpha_beta_sum (α β : ℝ) : 
  (∀ x : ℝ, x ≠ 30 → (x - α) / (x + β) = (x^2 - 120*x + 3600) / (x^2 + 70*x - 2300)) →
  α + β = 137 := by
sorry

end NUMINAMATH_CALUDE_alpha_beta_sum_l1740_174079


namespace NUMINAMATH_CALUDE_special_line_properties_l1740_174095

/-- A line passing through (5, 2) with x-intercept twice the y-intercept -/
def special_line (x y : ℝ) : Prop :=
  x + 2 * y - 9 = 0

theorem special_line_properties :
  (special_line 5 2) ∧ 
  (∃ (a : ℝ), a ≠ 0 ∧ special_line (2*a) 0 ∧ special_line 0 a) :=
by sorry

end NUMINAMATH_CALUDE_special_line_properties_l1740_174095


namespace NUMINAMATH_CALUDE_point_between_l1740_174049

theorem point_between (a b c : ℚ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) 
  (h4 : |a - b| + |b - c| = |a - c|) : 
  (a < b ∧ b < c) ∨ (c < b ∧ b < a) :=
sorry

end NUMINAMATH_CALUDE_point_between_l1740_174049


namespace NUMINAMATH_CALUDE_vectors_perpendicular_l1740_174014

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (1, 3)

theorem vectors_perpendicular : 
  let c := (a.1 - b.1, a.2 - b.2)
  (a.1 * c.1 + a.2 * c.2 = 0) := by sorry

end NUMINAMATH_CALUDE_vectors_perpendicular_l1740_174014


namespace NUMINAMATH_CALUDE_inverse_function_problem_l1740_174063

-- Define the function f and its inverse
def f : ℝ → ℝ := sorry

-- State the theorem
theorem inverse_function_problem (h : ∀ x > 0, f⁻¹ x = x^2) : f 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_problem_l1740_174063


namespace NUMINAMATH_CALUDE_six_digit_number_concatenation_divisibility_l1740_174055

theorem six_digit_number_concatenation_divisibility :
  ∃ (A B : ℕ), 
    A ≠ B ∧
    100000 ≤ A ∧ A < 1000000 ∧
    100000 ≤ B ∧ B < 1000000 ∧
    (10^6 * B + A) % (A * B) = 0 :=
by sorry

end NUMINAMATH_CALUDE_six_digit_number_concatenation_divisibility_l1740_174055


namespace NUMINAMATH_CALUDE_theater_seats_count_l1740_174039

/-- Represents the theater ticket sales scenario -/
structure TheaterSales where
  adultTicketPrice : ℕ
  childTicketPrice : ℕ
  totalRevenue : ℕ
  childTicketsSold : ℕ

/-- Calculates the total number of seats in the theater -/
def totalSeats (sales : TheaterSales) : ℕ :=
  let adultTicketsSold := (sales.totalRevenue - sales.childTicketPrice * sales.childTicketsSold) / sales.adultTicketPrice
  adultTicketsSold + sales.childTicketsSold

/-- Theorem stating that given the specific conditions, the theater has 80 seats -/
theorem theater_seats_count (sales : TheaterSales) 
  (h1 : sales.adultTicketPrice = 12)
  (h2 : sales.childTicketPrice = 5)
  (h3 : sales.totalRevenue = 519)
  (h4 : sales.childTicketsSold = 63) :
  totalSeats sales = 80 := by
  sorry

end NUMINAMATH_CALUDE_theater_seats_count_l1740_174039


namespace NUMINAMATH_CALUDE_opinion_change_difference_l1740_174087

theorem opinion_change_difference (initial_yes initial_no final_yes final_no : ℝ) :
  initial_yes = 30 →
  initial_no = 70 →
  final_yes = 60 →
  final_no = 40 →
  initial_yes + initial_no = 100 →
  final_yes + final_no = 100 →
  ∃ (min_change max_change : ℝ),
    (min_change ≤ max_change) ∧
    (∀ (change : ℝ), change ≥ min_change ∧ change ≤ max_change →
      ∃ (yes_to_no no_to_yes : ℝ),
        yes_to_no ≥ 0 ∧
        no_to_yes ≥ 0 ∧
        yes_to_no + no_to_yes = change ∧
        initial_yes - yes_to_no + no_to_yes = final_yes) ∧
    (max_change - min_change = 30) :=
by sorry

end NUMINAMATH_CALUDE_opinion_change_difference_l1740_174087


namespace NUMINAMATH_CALUDE_smallest_number_l1740_174024

/-- Converts a number from base b to decimal --/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

/-- The binary representation of 1111 --/
def binary_1111 : List Nat := [1, 1, 1, 1]

/-- The base-6 representation of 210 --/
def base6_210 : List Nat := [2, 1, 0]

/-- The base-4 representation of 1000 --/
def base4_1000 : List Nat := [1, 0, 0, 0]

/-- The octal representation of 101 --/
def octal_101 : List Nat := [1, 0, 1]

theorem smallest_number :
  to_decimal binary_1111 2 < to_decimal base6_210 6 ∧
  to_decimal binary_1111 2 < to_decimal base4_1000 4 ∧
  to_decimal binary_1111 2 < to_decimal octal_101 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l1740_174024


namespace NUMINAMATH_CALUDE_twenty_one_numbers_inequality_l1740_174006

theorem twenty_one_numbers_inequality (S : Finset ℕ) : 
  S ⊆ Finset.range 2047 →
  S.card = 21 →
  ∃ a b c : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (b * c : ℝ) < 2 * (a^2 : ℝ) ∧ 2 * (a^2 : ℝ) < 4 * (b * c : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_twenty_one_numbers_inequality_l1740_174006


namespace NUMINAMATH_CALUDE_distribute_six_balls_two_boxes_limit_four_l1740_174013

/-- The number of ways to distribute n distinguishable balls into 2 indistinguishable boxes,
    where no box can hold more than m balls. -/
def distributeWithLimit (n : ℕ) (m : ℕ) : ℕ := sorry

/-- The theorem stating that there are 25 ways to distribute 6 distinguishable balls
    into 2 indistinguishable boxes, where no box can hold more than 4 balls. -/
theorem distribute_six_balls_two_boxes_limit_four :
  distributeWithLimit 6 4 = 25 := by sorry

end NUMINAMATH_CALUDE_distribute_six_balls_two_boxes_limit_four_l1740_174013


namespace NUMINAMATH_CALUDE_midpoint_coordinates_l1740_174027

/-- Given two points P and Q in a plane, where Q is the midpoint of PR, 
    prove that R has specific coordinates. -/
theorem midpoint_coordinates (P Q : ℝ × ℝ) (h1 : P = (1, 3)) (h2 : Q = (4, 7)) 
    (h3 : Q = ((P.1 + R.1) / 2, (P.2 + R.2) / 2)) : R = (7, 11) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_coordinates_l1740_174027


namespace NUMINAMATH_CALUDE_power_of_three_plus_five_mod_ten_l1740_174002

theorem power_of_three_plus_five_mod_ten : (3^108 + 5) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_plus_five_mod_ten_l1740_174002


namespace NUMINAMATH_CALUDE_unique_function_theorem_l1740_174085

open Real

-- Define the function type
def FunctionType := (x : ℝ) → x > 0 → ℝ

-- State the theorem
theorem unique_function_theorem (f : FunctionType) 
  (h1 : f 2009 (by norm_num) = 1)
  (h2 : ∀ (x y : ℝ) (hx : x > 0) (hy : y > 0), 
    f x hx * f y hy + f (2009 / x) (by positivity) * f (2009 / y) (by positivity) = 2 * f (x * y) (by positivity)) :
  ∀ (x : ℝ) (hx : x > 0), f x hx = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_function_theorem_l1740_174085


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1740_174028

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 4) * (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 7 / Real.sqrt 8) = Real.sqrt 70 / 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1740_174028


namespace NUMINAMATH_CALUDE_square_of_difference_l1740_174047

theorem square_of_difference (y : ℝ) (h : y^2 ≥ 49) :
  (7 - Real.sqrt (y^2 - 49))^2 = y^2 - 14 * Real.sqrt (y^2 - 49) := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l1740_174047


namespace NUMINAMATH_CALUDE_percent_of_360_l1740_174005

theorem percent_of_360 : (35 / 100) * 360 = 126 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_360_l1740_174005


namespace NUMINAMATH_CALUDE_line_vector_proof_l1740_174007

def line_vector (t : ℝ) : ℝ × ℝ × ℝ := sorry

theorem line_vector_proof :
  (line_vector 0 = (2, 5, 9)) →
  (line_vector 1 = (3, 3, 5)) →
  (line_vector (-1) = (1, 7, 13)) := by sorry

end NUMINAMATH_CALUDE_line_vector_proof_l1740_174007


namespace NUMINAMATH_CALUDE_mark_and_carolyn_money_l1740_174092

theorem mark_and_carolyn_money (mark_money : ℚ) (carolyn_money : ℚ) : 
  mark_money = 5/8 → carolyn_money = 2/5 → mark_money + carolyn_money = 1.025 := by
  sorry

end NUMINAMATH_CALUDE_mark_and_carolyn_money_l1740_174092


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_l1740_174019

def prime_factors : List (Nat × Nat) := [(2, 12), (3, 16), (7, 18), (11, 7)]

def count_square_factors (p : Nat) (e : Nat) : Nat :=
  (e / 2) + 1

theorem count_perfect_square_factors :
  (prime_factors.map (fun (p, e) => count_square_factors p e)).prod = 2520 := by
  sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_l1740_174019


namespace NUMINAMATH_CALUDE_walter_exceptional_days_l1740_174093

/-- Represents the number of days Walter performed his chores in each category -/
structure ChorePerformance where
  poor : ℕ
  adequate : ℕ
  exceptional : ℕ

/-- Theorem stating that given the conditions, Walter performed exceptionally well for 6 days -/
theorem walter_exceptional_days :
  ∃ (perf : ChorePerformance),
    perf.poor + perf.adequate + perf.exceptional = 15 ∧
    2 * perf.poor + 4 * perf.adequate + 7 * perf.exceptional = 70 ∧
    perf.exceptional = 6 := by
  sorry


end NUMINAMATH_CALUDE_walter_exceptional_days_l1740_174093


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l1740_174045

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem tenth_term_of_sequence (a : ℤ) (d : ℤ) :
  arithmetic_sequence a d 4 = 23 →
  arithmetic_sequence a d 8 = 55 →
  arithmetic_sequence a d 10 = 71 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l1740_174045


namespace NUMINAMATH_CALUDE_baking_cookies_theorem_l1740_174078

/-- The number of pans of cookies that can be baked in a given time -/
def pans_of_cookies (total_time minutes_per_pan : ℕ) : ℕ :=
  total_time / minutes_per_pan

theorem baking_cookies_theorem (total_time minutes_per_pan : ℕ) 
  (h1 : total_time = 28) (h2 : minutes_per_pan = 7) : 
  pans_of_cookies total_time minutes_per_pan = 4 := by
  sorry

end NUMINAMATH_CALUDE_baking_cookies_theorem_l1740_174078


namespace NUMINAMATH_CALUDE_simplify_fraction_l1740_174034

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  (1 - x / (x - 1)) / (1 / (x^2 - x)) = -x := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1740_174034


namespace NUMINAMATH_CALUDE_mary_added_four_peanuts_l1740_174054

/-- The number of peanuts Mary added to the box -/
def peanuts_added (initial final : ℕ) : ℕ := final - initial

/-- Theorem stating that Mary added 4 peanuts to the box -/
theorem mary_added_four_peanuts :
  peanuts_added 4 8 = 4 := by sorry

end NUMINAMATH_CALUDE_mary_added_four_peanuts_l1740_174054


namespace NUMINAMATH_CALUDE_cake_price_problem_l1740_174025

theorem cake_price_problem (original_price : ℝ) : 
  (8 * original_price = 320) → 
  (10 * (0.8 * original_price) = 320) → 
  original_price = 40 := by
sorry

end NUMINAMATH_CALUDE_cake_price_problem_l1740_174025


namespace NUMINAMATH_CALUDE_buddy_cards_l1740_174057

/-- Calculates the number of baseball cards Buddy has on Saturday --/
def saturday_cards (initial : ℕ) : ℕ :=
  let tuesday := initial - (initial * 30 / 100)
  let wednesday := tuesday + (tuesday * 20 / 100)
  let thursday := wednesday - (wednesday / 4)
  let friday := thursday + (thursday / 3)
  friday + (friday * 2)

/-- Theorem stating that Buddy will have 252 cards on Saturday --/
theorem buddy_cards : saturday_cards 100 = 252 := by
  sorry

end NUMINAMATH_CALUDE_buddy_cards_l1740_174057


namespace NUMINAMATH_CALUDE_square_sum_problem_l1740_174068

theorem square_sum_problem (x y : ℕ+) 
  (h1 : x.val * y.val + x.val + y.val = 35)
  (h2 : x.val * y.val * (x.val + y.val) = 360) :
  x.val^2 + y.val^2 = 185 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_problem_l1740_174068


namespace NUMINAMATH_CALUDE_square_root_squared_specific_square_root_squared_l1740_174004

theorem square_root_squared (n : ℝ) (h : 0 ≤ n) : (Real.sqrt n) ^ 2 = n := by sorry

theorem specific_square_root_squared : (Real.sqrt 529441) ^ 2 = 529441 := by
  apply square_root_squared
  norm_num

end NUMINAMATH_CALUDE_square_root_squared_specific_square_root_squared_l1740_174004


namespace NUMINAMATH_CALUDE_root_difference_quadratic_equation_l1740_174011

theorem root_difference_quadratic_equation :
  let a : ℝ := 2
  let b : ℝ := 5
  let c : ℝ := -12
  let larger_root : ℝ := (-b + (b^2 - 4*a*c).sqrt) / (2*a)
  let smaller_root : ℝ := (-b - (b^2 - 4*a*c).sqrt) / (2*a)
  larger_root - smaller_root = 5.5 :=
by sorry

end NUMINAMATH_CALUDE_root_difference_quadratic_equation_l1740_174011


namespace NUMINAMATH_CALUDE_rachels_father_age_rachels_father_age_at_25_l1740_174098

/-- Rachel's age problem -/
theorem rachels_father_age (rachel_age : ℕ) (grandfather_age_multiplier : ℕ) 
  (father_age_difference : ℕ) (rachel_future_age : ℕ) : ℕ :=
  let grandfather_age := rachel_age * grandfather_age_multiplier
  let mother_age := grandfather_age / 2
  let father_age := mother_age + father_age_difference
  let years_passed := rachel_future_age - rachel_age
  father_age + years_passed

/-- Proof of Rachel's father's age when she is 25 -/
theorem rachels_father_age_at_25 : 
  rachels_father_age 12 7 5 25 = 60 := by
  sorry

end NUMINAMATH_CALUDE_rachels_father_age_rachels_father_age_at_25_l1740_174098


namespace NUMINAMATH_CALUDE_negation_equivalence_l1740_174016

theorem negation_equivalence :
  (¬ ∀ n : ℕ, 3^n > 500^n) ↔ (∃ n₀ : ℕ, 3^n₀ ≤ 500) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1740_174016


namespace NUMINAMATH_CALUDE_twenty_five_percent_less_than_80_l1740_174065

theorem twenty_five_percent_less_than_80 (x : ℝ) : x + (1/4) * x = 80 - (1/4) * 80 → x = 48 := by
  sorry

end NUMINAMATH_CALUDE_twenty_five_percent_less_than_80_l1740_174065


namespace NUMINAMATH_CALUDE_draw_all_red_probability_l1740_174046

-- Define the number of red and green chips
def num_red : ℕ := 3
def num_green : ℕ := 2

-- Define the total number of chips
def total_chips : ℕ := num_red + num_green

-- Define the probability of drawing all red chips before both green chips
def prob_all_red : ℚ := 3 / 10

-- Theorem statement
theorem draw_all_red_probability :
  prob_all_red = (num_red * (num_red - 1) * (num_red - 2)) / 
    (total_chips * (total_chips - 1) * (total_chips - 2)) :=
by sorry

end NUMINAMATH_CALUDE_draw_all_red_probability_l1740_174046


namespace NUMINAMATH_CALUDE_sum_seventh_eighth_l1740_174076

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  sum_first_two : a 1 + a 2 = 16
  sum_third_fourth : a 3 + a 4 = 32

/-- The sum of the 7th and 8th terms is 128 -/
theorem sum_seventh_eighth (seq : GeometricSequence) : seq.a 7 + seq.a 8 = 128 := by
  sorry

end NUMINAMATH_CALUDE_sum_seventh_eighth_l1740_174076


namespace NUMINAMATH_CALUDE_trig_inequality_l1740_174017

theorem trig_inequality (α β : Real) (h_α : 0 < α ∧ α < π / 2) (h_β : 0 < β ∧ β < π / 2) :
  Real.sin α ^ 3 * Real.cos β ^ 3 + Real.sin α ^ 3 * Real.sin β ^ 3 + Real.cos α ^ 3 ≥ Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_inequality_l1740_174017


namespace NUMINAMATH_CALUDE_d_must_be_positive_l1740_174096

theorem d_must_be_positive
  (a b c d e f : ℤ)
  (h1 : a * b + c * d * e * f < 0)
  (h2 : a < 0)
  (h3 : b < 0)
  (h4 : c < 0)
  (h5 : e < 0)
  (h6 : f < 0) :
  d > 0 := by
sorry

end NUMINAMATH_CALUDE_d_must_be_positive_l1740_174096


namespace NUMINAMATH_CALUDE_divide_by_fraction_twelve_divided_by_one_sixth_l1740_174040

theorem divide_by_fraction (a b : ℚ) (hb : b ≠ 0) : a / (1 / b) = a * b := by sorry

theorem twelve_divided_by_one_sixth : 12 / (1 / 6) = 72 := by sorry

end NUMINAMATH_CALUDE_divide_by_fraction_twelve_divided_by_one_sixth_l1740_174040


namespace NUMINAMATH_CALUDE_f_value_at_sqrt3_over_2_main_theorem_l1740_174048

-- Define the function f
def f (x : ℝ) : ℝ := 1 - 2 * x^2

-- Theorem statement
theorem f_value_at_sqrt3_over_2 : f (Real.sqrt 3 / 2) = -1/2 := by
  sorry

-- The main theorem that corresponds to the original problem
theorem main_theorem : 
  (∀ x, f (Real.sin x) = 1 - 2 * (Real.sin x)^2) → f (Real.sqrt 3 / 2) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_sqrt3_over_2_main_theorem_l1740_174048


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l1740_174075

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The product of 45 and 800 -/
def product : ℕ := 45 * 800

theorem product_trailing_zeros :
  trailingZeros product = 3 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l1740_174075


namespace NUMINAMATH_CALUDE_expectation_of_linear_combination_l1740_174001

variable (ξ η : ℝ → ℝ)
variable (E : (ℝ → ℝ) → ℝ)

axiom linearity_of_expectation : ∀ (a b : ℝ) (X Y : ℝ → ℝ), E (λ ω => a * X ω + b * Y ω) = a * E X + b * E Y

theorem expectation_of_linear_combination
  (h1 : E ξ = 10)
  (h2 : E η = 3) :
  E (λ ω => 3 * ξ ω + 5 * η ω) = 45 := by
sorry

end NUMINAMATH_CALUDE_expectation_of_linear_combination_l1740_174001


namespace NUMINAMATH_CALUDE_power_five_hundred_mod_eighteen_l1740_174018

theorem power_five_hundred_mod_eighteen : 
  (5 : ℤ) ^ 100 % 18 = 13 := by
  sorry

end NUMINAMATH_CALUDE_power_five_hundred_mod_eighteen_l1740_174018


namespace NUMINAMATH_CALUDE_f_properties_l1740_174097

noncomputable def f (x : ℝ) := Real.exp x - Real.exp (-x)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x, deriv f x > 0) ∧
  (∀ k, (∀ x, f (x^2) + f (k*x + 1) > 0) ↔ -2 < k ∧ k < 2) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1740_174097


namespace NUMINAMATH_CALUDE_price_reduction_equivalence_l1740_174086

theorem price_reduction_equivalence : 
  let first_reduction : ℝ := 0.25
  let second_reduction : ℝ := 0.20
  let equivalent_reduction : ℝ := 1 - (1 - first_reduction) * (1 - second_reduction)
  equivalent_reduction = 0.40
  := by sorry

end NUMINAMATH_CALUDE_price_reduction_equivalence_l1740_174086


namespace NUMINAMATH_CALUDE_roots_of_g_l1740_174091

def f (a b x : ℝ) : ℝ := a * x - b

def g (a b x : ℝ) : ℝ := b * x^2 + 3 * a * x

theorem roots_of_g (a b : ℝ) (h : f a b 3 = 0) :
  {x : ℝ | g a b x = 0} = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_roots_of_g_l1740_174091


namespace NUMINAMATH_CALUDE_water_tank_capacity_l1740_174073

theorem water_tank_capacity (c : ℝ) (h1 : c > 0) : 
  (c / 4 : ℝ) / c = 1 / 4 ∧ 
  ((c / 4 + 5) : ℝ) / c = 1 / 3 → 
  c = 60 := by
sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l1740_174073


namespace NUMINAMATH_CALUDE_quadratic_roots_conditions_l1740_174050

theorem quadratic_roots_conditions (k : ℝ) :
  let f : ℝ → ℝ := λ x => 2 * x^2 - 4 * k * x + x - (1 - 2 * k^2)
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) ↔ k ≤ 9/8 ∧
  (∀ x : ℝ, f x ≠ 0) ↔ k > 9/8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_conditions_l1740_174050


namespace NUMINAMATH_CALUDE_non_multiples_count_is_546_l1740_174080

/-- The count of three-digit numbers that are not multiples of 3 or 11 -/
def non_multiples_count : ℕ :=
  let total_three_digit := 999 - 100 + 1
  let multiples_of_3 := (999 - 100) / 3 + 1
  let multiples_of_11 := (990 - 110) / 11 + 1
  let multiples_of_33 := (990 - 132) / 33 + 1
  total_three_digit - (multiples_of_3 + multiples_of_11 - multiples_of_33)

theorem non_multiples_count_is_546 : non_multiples_count = 546 := by
  sorry

end NUMINAMATH_CALUDE_non_multiples_count_is_546_l1740_174080


namespace NUMINAMATH_CALUDE_reciprocal_equals_self_l1740_174000

theorem reciprocal_equals_self (q : ℚ) : q⁻¹ = q → q = 1 ∨ q = -1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equals_self_l1740_174000


namespace NUMINAMATH_CALUDE_roots_sum_product_l1740_174081

theorem roots_sum_product (α' β' : ℝ) : 
  (α' + β' = 5) → (α' * β' = 6) → 3 * α'^3 + 4 * β'^2 = 271 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_product_l1740_174081


namespace NUMINAMATH_CALUDE_f_increasing_iff_l1740_174003

/-- Definition of the piecewise function f(x) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < -1 then (-a + 4) * x - 3 * a
  else x^2 + a * x - 8

/-- Theorem stating the condition for f to be increasing --/
theorem f_increasing_iff (a : ℝ) :
  (∀ x y, x < y → f a x < f a y) ↔ 3 ≤ a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_iff_l1740_174003


namespace NUMINAMATH_CALUDE_gcd_204_85_f_at_2_l1740_174008

-- Part 1: GCD of 204 and 85
theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by sorry

-- Part 2: Value of polynomial at x = 2
def f (x : ℝ) : ℝ := 2*x^4 + 3*x^3 + 5*x - 4

theorem f_at_2 : f 2 = 62 := by sorry

end NUMINAMATH_CALUDE_gcd_204_85_f_at_2_l1740_174008


namespace NUMINAMATH_CALUDE_simplify_fraction_1_l1740_174029

theorem simplify_fraction_1 (x : ℝ) (h : x ≠ 0 ∧ x ≠ 1) :
  1 / x + 1 / (x * (x - 1)) = 1 / (x - 1) := by
  sorry


end NUMINAMATH_CALUDE_simplify_fraction_1_l1740_174029


namespace NUMINAMATH_CALUDE_digit_sum_theorem_l1740_174033

theorem digit_sum_theorem (A B C D : ℕ) : 
  A ≠ 0 →
  A < 10 → B < 10 → C < 10 → D < 10 →
  1000 * A + 100 * B + 10 * C + D = (10 * C + D)^2 - (10 * A + B)^2 →
  A + B + C + D = 21 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_theorem_l1740_174033


namespace NUMINAMATH_CALUDE_christmas_to_january_10_l1740_174022

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDay (nextDay d) n

-- Theorem statement
theorem christmas_to_january_10 :
  advanceDay DayOfWeek.Wednesday 16 = DayOfWeek.Friday := by
  sorry

end NUMINAMATH_CALUDE_christmas_to_january_10_l1740_174022


namespace NUMINAMATH_CALUDE_lcm_from_hcf_and_product_l1740_174066

theorem lcm_from_hcf_and_product (a b : ℕ+) : 
  Nat.gcd a b = 21 → a * b = 138567 → Nat.lcm a b = 6603 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_hcf_and_product_l1740_174066


namespace NUMINAMATH_CALUDE_positive_difference_of_solutions_l1740_174077

-- Define the equation
def equation (x : ℝ) : Prop := (9 - x^2 / 3)^(1/3) = 3

-- Define the set of solutions
def solutions : Set ℝ := {x : ℝ | equation x}

-- Theorem statement
theorem positive_difference_of_solutions :
  ∃ (x y : ℝ), x ∈ solutions ∧ y ∈ solutions ∧ x ≠ y ∧ |x - y| = 18 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_positive_difference_of_solutions_l1740_174077


namespace NUMINAMATH_CALUDE_lassis_from_twenty_fruit_l1740_174036

/-- The number of lassis that can be made given a certain number of fruit units -/
def lassis_from_fruit (fruit_units : ℕ) : ℚ :=
  (9 : ℚ) / 4 * fruit_units

/-- Theorem stating that 45 lassis can be made from 20 fruit units -/
theorem lassis_from_twenty_fruit : lassis_from_fruit 20 = 45 := by
  sorry

end NUMINAMATH_CALUDE_lassis_from_twenty_fruit_l1740_174036


namespace NUMINAMATH_CALUDE_derivative_of_sin_minus_cos_l1740_174069

theorem derivative_of_sin_minus_cos (α : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin α - Real.cos x
  (deriv f) α = Real.sin α :=
by
  sorry

end NUMINAMATH_CALUDE_derivative_of_sin_minus_cos_l1740_174069


namespace NUMINAMATH_CALUDE_area_of_right_triangle_abc_l1740_174082

/-- Right triangle ABC with specific properties -/
structure RightTriangleABC where
  -- A, B, C are points in the plane
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- ABC is a right triangle with right angle at C
  is_right_triangle : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
  -- Length of AB is 50
  hypotenuse_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 50^2
  -- Median through A lies on y = x + 2
  median_A : ∃ t : ℝ, (A.1 + C.1) / 2 = t ∧ (A.2 + C.2) / 2 = t + 2
  -- Median through B lies on y = 3x + 1
  median_B : ∃ t : ℝ, (B.1 + C.1) / 2 = t ∧ (B.2 + C.2) / 2 = 3 * t + 1

/-- The area of the right triangle ABC is 250/3 -/
theorem area_of_right_triangle_abc (t : RightTriangleABC) : 
  abs ((t.A.1 - t.C.1) * (t.B.2 - t.C.2) - (t.B.1 - t.C.1) * (t.A.2 - t.C.2)) / 2 = 250 / 3 := by
  sorry

end NUMINAMATH_CALUDE_area_of_right_triangle_abc_l1740_174082


namespace NUMINAMATH_CALUDE_summer_discount_is_fifty_percent_l1740_174064

def original_price : ℝ := 49
def final_price : ℝ := 14.50
def additional_discount : ℝ := 10

def summer_discount_percentage (d : ℝ) : Prop :=
  original_price * (1 - d / 100) - additional_discount = final_price

theorem summer_discount_is_fifty_percent : 
  summer_discount_percentage 50 := by sorry

end NUMINAMATH_CALUDE_summer_discount_is_fifty_percent_l1740_174064
