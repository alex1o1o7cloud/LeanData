import Mathlib

namespace NUMINAMATH_CALUDE_orange_gumdrops_after_replacement_l1047_104724

/-- Represents the number of gumdrops of each color in a jar -/
structure GumdropJar where
  blue : ℕ
  brown : ℕ
  red : ℕ
  yellow : ℕ
  green : ℕ
  orange : ℕ

/-- Calculates the total number of gumdrops in the jar -/
def GumdropJar.total (jar : GumdropJar) : ℕ :=
  jar.blue + jar.brown + jar.red + jar.yellow + jar.green + jar.orange

/-- Represents the initial distribution of gumdrops -/
def initial_jar : GumdropJar :=
  { blue := 40
    brown := 15
    red := 10
    yellow := 5
    green := 20
    orange := 10 }

/-- Theorem stating that after replacing a third of blue gumdrops with orange,
    the number of orange gumdrops will be 23 -/
theorem orange_gumdrops_after_replacement (jar : GumdropJar)
    (h1 : jar = initial_jar)
    (h2 : jar.total = 100)
    (h3 : jar.blue / 3 = 13) :
  (⟨jar.blue - 13, jar.brown, jar.red, jar.yellow, jar.green, jar.orange + 13⟩ : GumdropJar).orange = 23 := by
  sorry

end NUMINAMATH_CALUDE_orange_gumdrops_after_replacement_l1047_104724


namespace NUMINAMATH_CALUDE_chord_length_l1047_104779

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := 3*x + 4*y - 11 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ line_l A.1 A.2 ∧ line_l B.1 B.2

-- Theorem statement
theorem chord_length (A B : ℝ × ℝ) :
  intersection_points A B → abs (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l1047_104779


namespace NUMINAMATH_CALUDE_circle_radius_is_6_sqrt_2_l1047_104761

/-- A right triangle with squares constructed on two sides --/
structure RightTriangleWithSquares where
  -- The lengths of the two sides of the right triangle
  PQ : ℝ
  QR : ℝ
  -- Assertion that the squares are constructed on these sides
  square_PQ_constructed : Bool
  square_QR_constructed : Bool
  -- Assertion that the corners of the squares lie on a circle
  corners_on_circle : Bool

/-- The radius of the circle passing through the corners of the squares --/
def circle_radius (t : RightTriangleWithSquares) : ℝ :=
  sorry

/-- Theorem stating that for a right triangle with PQ = 9 and QR = 12,
    and squares constructed on these sides, if the corners of the squares
    lie on a circle, then the radius of this circle is 6√2 --/
theorem circle_radius_is_6_sqrt_2 (t : RightTriangleWithSquares)
    (h1 : t.PQ = 9)
    (h2 : t.QR = 12)
    (h3 : t.square_PQ_constructed = true)
    (h4 : t.square_QR_constructed = true)
    (h5 : t.corners_on_circle = true) :
  circle_radius t = 6 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_is_6_sqrt_2_l1047_104761


namespace NUMINAMATH_CALUDE_teal_color_survey_l1047_104798

theorem teal_color_survey (total : ℕ) (more_blue : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 150)
  (h2 : more_blue = 85)
  (h3 : both = 47)
  (h4 : neither = 22) :
  total - (more_blue - both + both + neither) = 90 := by
  sorry

end NUMINAMATH_CALUDE_teal_color_survey_l1047_104798


namespace NUMINAMATH_CALUDE_investment_growth_l1047_104705

/-- Compound interest function -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Problem statement -/
theorem investment_growth :
  let principal : ℝ := 8000
  let rate : ℝ := 0.04
  let time : ℕ := 10
  abs (compound_interest principal rate time - 11841.92) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_investment_growth_l1047_104705


namespace NUMINAMATH_CALUDE_count_integer_ratios_eq_five_l1047_104769

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  sorry

/-- Given two arithmetic sequences and a property of their sums,
    counts the number of positive integers that make the ratio of their terms an integer -/
def count_integer_ratios (a b : ArithmeticSequence) : ℕ :=
  sorry

theorem count_integer_ratios_eq_five
  (a b : ArithmeticSequence)
  (h : ∀ n : ℕ+, sum_n a n / sum_n b n = (7 * n + 45) / (n + 3)) :
  count_integer_ratios a b = 5 :=
sorry

end NUMINAMATH_CALUDE_count_integer_ratios_eq_five_l1047_104769


namespace NUMINAMATH_CALUDE_slacks_percentage_is_25_percent_l1047_104763

/-- Represents the clothing items and their quantities -/
structure Wardrobe where
  blouses : ℕ
  skirts : ℕ
  slacks : ℕ

/-- Represents the percentages of clothing items in the hamper -/
structure HamperPercentages where
  blouses : ℚ
  skirts : ℚ
  slacks : ℚ

/-- Calculates the percentage of slacks in the hamper -/
def calculate_slacks_percentage (w : Wardrobe) (h : HamperPercentages) (total_in_washer : ℕ) : ℚ :=
  let blouses_in_hamper := (w.blouses : ℚ) * h.blouses
  let skirts_in_hamper := (w.skirts : ℚ) * h.skirts
  let slacks_in_hamper := (total_in_washer : ℚ) - blouses_in_hamper - skirts_in_hamper
  slacks_in_hamper / (w.slacks : ℚ)

/-- Theorem stating that the percentage of slacks in the hamper is 25% -/
theorem slacks_percentage_is_25_percent (w : Wardrobe) (h : HamperPercentages) :
  w.blouses = 12 →
  w.skirts = 6 →
  w.slacks = 8 →
  h.blouses = 3/4 →
  h.skirts = 1/2 →
  calculate_slacks_percentage w h 14 = 1/4 := by
  sorry

#eval (1 : ℚ) / 4  -- To verify that 1/4 is indeed 25%

end NUMINAMATH_CALUDE_slacks_percentage_is_25_percent_l1047_104763


namespace NUMINAMATH_CALUDE_units_digit_factorial_product_l1047_104700

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_factorial_product :
  units_digit (factorial 1 * factorial 2 * factorial 3 * factorial 4) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_factorial_product_l1047_104700


namespace NUMINAMATH_CALUDE_gcd_of_factorials_l1047_104742

theorem gcd_of_factorials : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_factorials_l1047_104742


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1047_104785

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + x - 1 < 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1047_104785


namespace NUMINAMATH_CALUDE_square_side_length_l1047_104773

theorem square_side_length (rectangle_length : ℝ) (rectangle_width : ℝ) (square_side : ℝ) : 
  rectangle_length = 7 → 
  rectangle_width = 5 → 
  4 * square_side = 2 * (rectangle_length + rectangle_width) → 
  square_side = 6 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l1047_104773


namespace NUMINAMATH_CALUDE_factory_defect_rate_l1047_104736

theorem factory_defect_rate (total_output : ℝ) : 
  let machine_a_output := 0.4 * total_output
  let machine_b_output := 0.6 * total_output
  let machine_a_defect_rate := 9 / 1000
  let total_defect_rate := 0.0156
  ∃ (machine_b_defect_rate : ℝ),
    0.4 * machine_a_defect_rate + 0.6 * machine_b_defect_rate = total_defect_rate ∧
    1 / machine_b_defect_rate = 50 := by
  sorry

end NUMINAMATH_CALUDE_factory_defect_rate_l1047_104736


namespace NUMINAMATH_CALUDE_travel_time_difference_l1047_104776

/-- Given a set of 5 numbers (x, y, 10, 11, 9) with an average of 10 and a variance of 2, |x-y| = 4 -/
theorem travel_time_difference (x y : ℝ) : 
  (x + y + 10 + 11 + 9) / 5 = 10 ∧ 
  ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2 →
  |x - y| = 4 := by
sorry


end NUMINAMATH_CALUDE_travel_time_difference_l1047_104776


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_24_l1047_104706

theorem modular_inverse_of_5_mod_24 :
  ∃ a : ℕ, a < 24 ∧ (5 * a) % 24 = 1 ∧ a = 5 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_24_l1047_104706


namespace NUMINAMATH_CALUDE_cards_difference_l1047_104795

theorem cards_difference (ann_cards : ℕ) (ann_heike_ratio : ℕ) (anton_heike_ratio : ℕ) :
  ann_cards = 60 →
  ann_heike_ratio = 6 →
  anton_heike_ratio = 3 →
  ann_cards - (anton_heike_ratio * (ann_cards / ann_heike_ratio)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_cards_difference_l1047_104795


namespace NUMINAMATH_CALUDE_square_field_area_l1047_104715

/-- Proves that a square field with specific barbed wire conditions has an area of 27889 square meters -/
theorem square_field_area (wire_cost_per_meter : ℝ) (total_cost : ℝ) (gate_width : ℝ) (num_gates : ℕ) :
  wire_cost_per_meter = 2.0 →
  total_cost = 1332 →
  gate_width = 1 →
  num_gates = 2 →
  ∃ (side_length : ℝ),
    side_length > 0 ∧
    total_cost = wire_cost_per_meter * (4 * side_length - (↑num_gates * gate_width)) ∧
    side_length^2 = 27889 :=
by sorry

end NUMINAMATH_CALUDE_square_field_area_l1047_104715


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l1047_104797

theorem tangent_line_intersection (x₀ : ℝ) (m : ℝ) : 
  (0 < m) → (m < 1) →
  (2 * x₀ = 1 / m) →
  (x₀^2 - Real.log (2 * x₀) - 1 = 0) →
  (Real.sqrt 2 < x₀) ∧ (x₀ < Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l1047_104797


namespace NUMINAMATH_CALUDE_triangle_area_l1047_104727

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the area of the triangle is √2 under the following conditions:
    1. b = a*cos(C) + c*cos(B)
    2. CA · CB = 1 (dot product)
    3. c = 2 -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b = a * Real.cos C + c * Real.cos B →
  a * c * Real.cos B = 1 →
  c = 2 →
  (1/2) * a * b * Real.sin C = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1047_104727


namespace NUMINAMATH_CALUDE_elevator_movement_l1047_104716

/-- Represents the number of floors in a building --/
def TotalFloors : ℕ := 13

/-- Represents the initial floor of the elevator --/
def InitialFloor : ℕ := 9

/-- Represents the first upward movement of the elevator --/
def FirstUpwardMovement : ℕ := 3

/-- Represents the second upward movement of the elevator --/
def SecondUpwardMovement : ℕ := 8

/-- Represents the final floor of the elevator (top floor) --/
def FinalFloor : ℕ := 13

theorem elevator_movement (x : ℕ) : 
  InitialFloor - x + FirstUpwardMovement + SecondUpwardMovement = FinalFloor → 
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_elevator_movement_l1047_104716


namespace NUMINAMATH_CALUDE_product_sum_6545_l1047_104709

theorem product_sum_6545 : ∃ (a b : ℕ), 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 6545 ∧ 
  a + b = 162 := by
sorry

end NUMINAMATH_CALUDE_product_sum_6545_l1047_104709


namespace NUMINAMATH_CALUDE_sum_leq_fourth_powers_over_product_l1047_104789

theorem sum_leq_fourth_powers_over_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≤ (a^4 + b^4 + c^4) / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_sum_leq_fourth_powers_over_product_l1047_104789


namespace NUMINAMATH_CALUDE_lunch_spending_solution_l1047_104720

def lunch_spending (your_spending : ℚ) : ℚ × ℚ × ℚ × ℚ :=
  (your_spending, 
   your_spending + 15, 
   your_spending - 20, 
   2 * your_spending)

theorem lunch_spending_solution : 
  ∃! (your_spending : ℚ), 
    let (you, friend1, friend2, friend3) := lunch_spending your_spending
    you + friend1 + friend2 + friend3 = 150 ∧
    friend1 = you + 15 ∧
    friend2 = you - 20 ∧
    friend3 = 2 * you :=
by
  sorry

#eval lunch_spending 31

end NUMINAMATH_CALUDE_lunch_spending_solution_l1047_104720


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1047_104794

theorem arithmetic_mean_problem (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 79 → a = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1047_104794


namespace NUMINAMATH_CALUDE_largest_and_smallest_A_l1047_104756

/-- Given a nine-digit number B, returns the number A obtained by moving the last digit of B to the first place -/
def getA (B : ℕ) : ℕ :=
  (B % 10) * 10^8 + B / 10

/-- Checks if two natural numbers are coprime -/
def isCoprime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

theorem largest_and_smallest_A :
  ∃ (A_max A_min : ℕ),
    (∀ A B : ℕ,
      B > 22222222 →
      isCoprime B 18 →
      A = getA B →
      A ≤ A_max ∧ A ≥ A_min) ∧
    A_max = 999999998 ∧
    A_min = 122222224 := by
  sorry

end NUMINAMATH_CALUDE_largest_and_smallest_A_l1047_104756


namespace NUMINAMATH_CALUDE_biology_quiz_probability_l1047_104735

theorem biology_quiz_probability : 
  let n : ℕ := 6  -- number of guessed questions
  let k : ℕ := 4  -- number of possible answers per question
  let p : ℚ := 1 / k  -- probability of guessing correctly on a single question
  1 - (1 - p) ^ n = 3367 / 4096 :=
by
  sorry

end NUMINAMATH_CALUDE_biology_quiz_probability_l1047_104735


namespace NUMINAMATH_CALUDE_fraction_difference_l1047_104787

theorem fraction_difference (a b : ℝ) (h : b / a = 2) :
  b / (a + b) - a / (a + b) = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_fraction_difference_l1047_104787


namespace NUMINAMATH_CALUDE_shiela_animal_drawings_l1047_104744

/-- Proves that each neighbor receives 8 animal drawings when Shiela distributes
    96 drawings equally among 12 neighbors. -/
theorem shiela_animal_drawings (neighbors : ℕ) (drawings : ℕ) (h1 : neighbors = 12) (h2 : drawings = 96) :
  drawings / neighbors = 8 := by
  sorry

end NUMINAMATH_CALUDE_shiela_animal_drawings_l1047_104744


namespace NUMINAMATH_CALUDE_square_root_expression_equals_256_l1047_104754

theorem square_root_expression_equals_256 :
  Real.sqrt ((16^12 + 2^36) / (16^5 + 2^42)) = 256 := by
  sorry

end NUMINAMATH_CALUDE_square_root_expression_equals_256_l1047_104754


namespace NUMINAMATH_CALUDE_lawn_mowing_time_l1047_104746

/-- Calculates the time required to mow a rectangular lawn -/
theorem lawn_mowing_time (lawn_length lawn_width swath_width overlap mowing_rate : ℝ) :
  lawn_length = 120 →
  lawn_width = 180 →
  swath_width = 30 / 12 →
  overlap = 6 / 12 →
  mowing_rate = 4000 →
  (lawn_width / (swath_width - overlap) * lawn_length) / mowing_rate = 2.7 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_time_l1047_104746


namespace NUMINAMATH_CALUDE_f_has_maximum_for_negative_x_l1047_104719

/-- The function f(x) = 2x + 1/x - 1 has a maximum value when x < 0 -/
theorem f_has_maximum_for_negative_x :
  ∃ (M : ℝ), ∀ (x : ℝ), x < 0 → (2 * x + 1 / x - 1 : ℝ) ≤ M := by
  sorry

end NUMINAMATH_CALUDE_f_has_maximum_for_negative_x_l1047_104719


namespace NUMINAMATH_CALUDE_min_sum_and_inequality_range_l1047_104701

-- Define the conditions
def conditions (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ 4 * a + b = a * b

-- Define the minimum value of a + b
def min_sum (a b : ℝ) : ℝ := a + b

-- Define the inequality condition
def inequality_condition (a b t : ℝ) : Prop :=
  ∀ x : ℝ, |x - a| + |x - b| ≥ t^2 - 2*t

-- Theorem statement
theorem min_sum_and_inequality_range :
  ∃ a b : ℝ, conditions a b ∧
    (∀ a' b' : ℝ, conditions a' b' → min_sum a b ≤ min_sum a' b') ∧
    min_sum a b = 9 ∧
    (∀ t : ℝ, inequality_condition a b t ↔ -1 ≤ t ∧ t ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_min_sum_and_inequality_range_l1047_104701


namespace NUMINAMATH_CALUDE_expression_evaluation_l1047_104772

theorem expression_evaluation : 72 + (120 / 15) + (15 * 12) - 250 - (480 / 8) = -50 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1047_104772


namespace NUMINAMATH_CALUDE_error_probability_theorem_l1047_104749

-- Define the probability of error
def probability_of_error : ℝ := 0.01

-- Define the observed value of K²
def observed_k_squared : ℝ := 6.635

-- Define the relationship between variables
def relationship_exists : Prop := True

-- Define the conclusion of the statistical test
def statistical_conclusion (p : ℝ) (relationship : Prop) : Prop :=
  p ≤ probability_of_error ∧ relationship

-- Theorem statement
theorem error_probability_theorem 
  (h : statistical_conclusion probability_of_error relationship_exists) :
  probability_of_error = 0.01 := by sorry

end NUMINAMATH_CALUDE_error_probability_theorem_l1047_104749


namespace NUMINAMATH_CALUDE_calculation_proof_l1047_104714

theorem calculation_proof : 1.23 * 67 + 8.2 * 12.3 - 90 * 0.123 = 172.2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1047_104714


namespace NUMINAMATH_CALUDE_opposite_reciprocal_abs_sum_l1047_104775

theorem opposite_reciprocal_abs_sum (x y m n a : ℝ) : 
  (x + y = 0) →  -- x and y are opposite numbers
  (m * n = 1) →  -- m and n are reciprocals
  (|a| = 3) →    -- absolute value of a is 3
  (a / (m * n) + 2018 * (x + y) = a) ∧ (a = 3 ∨ a = -3) := by
    sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_abs_sum_l1047_104775


namespace NUMINAMATH_CALUDE_remainder_of_3_pow_19_mod_10_l1047_104713

theorem remainder_of_3_pow_19_mod_10 : 3^19 % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_3_pow_19_mod_10_l1047_104713


namespace NUMINAMATH_CALUDE_decimal_comparisons_l1047_104790

theorem decimal_comparisons : 
  (3 > 2.95) ∧ (0.08 < 0.21) ∧ (0.6 = 0.60) := by
  sorry

end NUMINAMATH_CALUDE_decimal_comparisons_l1047_104790


namespace NUMINAMATH_CALUDE_correct_average_weight_l1047_104704

theorem correct_average_weight 
  (n : ℕ) 
  (initial_average : ℝ) 
  (misread_weight : ℝ) 
  (correct_weight : ℝ) :
  n = 20 →
  initial_average = 58.4 →
  misread_weight = 56 →
  correct_weight = 66 →
  (n * initial_average + (correct_weight - misread_weight)) / n = 58.9 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_weight_l1047_104704


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1047_104770

theorem polynomial_factorization (x : ℝ) : 
  x^6 - 4*x^4 + 6*x^2 - 4 = (x^2 - 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1047_104770


namespace NUMINAMATH_CALUDE_fifth_element_row_20_value_l1047_104703

/-- Pascal's triangle element -/
def pascal_triangle_element (n k : ℕ) : ℕ := Nat.choose n k

/-- The fifth element in Row 20 of Pascal's triangle -/
def fifth_element_row_20 : ℕ := pascal_triangle_element 20 4

theorem fifth_element_row_20_value : fifth_element_row_20 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_fifth_element_row_20_value_l1047_104703


namespace NUMINAMATH_CALUDE_symmetric_line_l1047_104708

/-- Given a line l with equation x - y + 1 = 0, prove that its symmetric line l' 
    with respect to x = 2 has the equation x + y - 5 = 0 -/
theorem symmetric_line (x y : ℝ) : 
  (x - y + 1 = 0) → 
  (∃ x' y', x' + y' - 5 = 0 ∧ x' = 4 - x ∧ y' = y) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_l1047_104708


namespace NUMINAMATH_CALUDE_fixed_point_on_all_lines_fixed_point_unique_l1047_104743

/-- The fixed point through which all lines of the form ax + y + 1 = 0 pass -/
def fixed_point : ℝ × ℝ := (0, -1)

/-- The equation of the line ax + y + 1 = 0 -/
def line_equation (a x y : ℝ) : Prop := a * x + y + 1 = 0

theorem fixed_point_on_all_lines :
  ∀ a : ℝ, line_equation a (fixed_point.1) (fixed_point.2) :=
by sorry

theorem fixed_point_unique :
  ∀ x y : ℝ, (∀ a : ℝ, line_equation a x y) → (x, y) = fixed_point :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_on_all_lines_fixed_point_unique_l1047_104743


namespace NUMINAMATH_CALUDE_tan_alpha_3_implies_fraction_l1047_104758

theorem tan_alpha_3_implies_fraction (α : Real) (h : Real.tan α = 3) :
  (Real.sin α + 3 * Real.cos α) / (2 * Real.sin α + 5 * Real.cos α) = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_3_implies_fraction_l1047_104758


namespace NUMINAMATH_CALUDE_horner_method_v₃_l1047_104788

def f (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 + 3*x + 2

def horner_v₃ (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  let v₀ := 1
  let v₁ := x + (-5)
  let v₂ := v₁ * x + 6
  v₂ * x + 0

theorem horner_method_v₃ :
  horner_v₃ f (-2) = -40 := by sorry

end NUMINAMATH_CALUDE_horner_method_v₃_l1047_104788


namespace NUMINAMATH_CALUDE_seminar_ratio_l1047_104757

theorem seminar_ratio (total_attendees : ℕ) (avg_age_all : ℚ) (avg_age_doctors : ℚ) (avg_age_lawyers : ℚ)
  (h_total : total_attendees = 20)
  (h_avg_all : avg_age_all = 45)
  (h_avg_doctors : avg_age_doctors = 40)
  (h_avg_lawyers : avg_age_lawyers = 55) :
  ∃ (num_doctors num_lawyers : ℚ),
    num_doctors + num_lawyers = total_attendees ∧
    (num_doctors * avg_age_doctors + num_lawyers * avg_age_lawyers) / total_attendees = avg_age_all ∧
    num_doctors / num_lawyers = 2 := by
  sorry


end NUMINAMATH_CALUDE_seminar_ratio_l1047_104757


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1047_104750

/-- Given an arithmetic sequence {1/aₙ} where a₁ = 1 and a₄ = 4, prove that a₁₀ = -4/5 -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) :
  (∃ d : ℚ, ∀ n : ℕ, 1 / a (n + 1) - 1 / a n = d) →
  a 1 = 1 →
  a 4 = 4 →
  a 10 = -4/5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1047_104750


namespace NUMINAMATH_CALUDE_isabellas_paintable_area_l1047_104752

/-- Calculates the total paintable area of walls in multiple bedrooms. -/
def total_paintable_area (num_bedrooms : ℕ) (length width height : ℝ) (unpaintable_area : ℝ) : ℝ :=
  let wall_area := 2 * (length * height + width * height)
  let paintable_area := wall_area - unpaintable_area
  num_bedrooms * paintable_area

/-- Proves that the total paintable area for Isabella's bedrooms is 1552 square feet. -/
theorem isabellas_paintable_area :
  total_paintable_area 4 14 12 9 80 = 1552 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_paintable_area_l1047_104752


namespace NUMINAMATH_CALUDE_ed_lighter_than_al_l1047_104745

/-- Prove that Ed is 38 pounds lighter than Al given the following conditions:
  * Al is 25 pounds heavier than Ben
  * Ben is 16 pounds lighter than Carl
  * Ed weighs 146 pounds
  * Carl weighs 175 pounds
-/
theorem ed_lighter_than_al (carl_weight ben_weight al_weight ed_weight : ℕ) : 
  carl_weight = 175 →
  ben_weight = carl_weight - 16 →
  al_weight = ben_weight + 25 →
  ed_weight = 146 →
  al_weight - ed_weight = 38 := by
  sorry

#check ed_lighter_than_al

end NUMINAMATH_CALUDE_ed_lighter_than_al_l1047_104745


namespace NUMINAMATH_CALUDE_sum_of_cubes_nonnegative_l1047_104731

theorem sum_of_cubes_nonnegative (n : ℤ) (a b : ℚ) 
  (h1 : n > 1) 
  (h2 : n = a^3 + b^3) : 
  ∃ (x y : ℚ), x ≥ 0 ∧ y ≥ 0 ∧ n = x^3 + y^3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_nonnegative_l1047_104731


namespace NUMINAMATH_CALUDE_expansion_coefficient_remainder_counts_l1047_104755

/-- 
Given a natural number n, Tᵣ(n) represents the number of coefficients in the expansion of (1+x)ⁿ 
that give a remainder of r when divided by 3, where r ∈ {0,1,2}.
-/
def T (r n : ℕ) : ℕ := sorry

/-- The theorem states the values of T₀(2006), T₁(2006), and T₂(2006) for the expansion of (1+x)²⁰⁰⁶. -/
theorem expansion_coefficient_remainder_counts : 
  T 0 2006 = 1764 ∧ T 1 2006 = 122 ∧ T 2 2006 = 121 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_remainder_counts_l1047_104755


namespace NUMINAMATH_CALUDE_max_sphere_radius_l1047_104733

-- Define the glass shape function
def glass_shape (x : ℝ) : ℝ := x^4

-- Define the circle equation
def circle_equation (x y r : ℝ) : Prop := x^2 + (y - r)^2 = r^2

-- Define the condition that the circle contains the origin
def contains_origin (r : ℝ) : Prop := circle_equation 0 0 r

-- Define the condition that the circle lies above or on the glass shape
def above_glass_shape (x y r : ℝ) : Prop := 
  circle_equation x y r → y ≥ glass_shape x

-- State the theorem
theorem max_sphere_radius : 
  ∃ (r : ℝ), r = (3 * 2^(1/3)) / 4 ∧ 
  (∀ (x y : ℝ), above_glass_shape x y r) ∧
  contains_origin r ∧
  (∀ (r' : ℝ), r' > r → ¬(∀ (x y : ℝ), above_glass_shape x y r') ∨ ¬(contains_origin r')) :=
sorry

end NUMINAMATH_CALUDE_max_sphere_radius_l1047_104733


namespace NUMINAMATH_CALUDE_exists_fib_divisible_by_2007_l1047_104760

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem exists_fib_divisible_by_2007 : ∃ n : ℕ, n > 0 ∧ 2007 ∣ fib n := by
  sorry

end NUMINAMATH_CALUDE_exists_fib_divisible_by_2007_l1047_104760


namespace NUMINAMATH_CALUDE_probability_two_white_balls_l1047_104723

def total_balls : ℕ := 7 + 8
def white_balls : ℕ := 7
def black_balls : ℕ := 8

theorem probability_two_white_balls :
  (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1)) = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_white_balls_l1047_104723


namespace NUMINAMATH_CALUDE_police_catches_thief_l1047_104781

-- Define the square courtyard
def side_length : ℝ := 340

-- Define speeds
def police_speed : ℝ := 85
def thief_speed : ℝ := 75

-- Define the time to catch
def time_to_catch : ℝ := 44

-- Theorem statement
theorem police_catches_thief :
  let time_to_sight : ℝ := (4 * side_length) / (police_speed - thief_speed)
  let police_distance : ℝ := police_speed * time_to_sight
  let thief_distance : ℝ := thief_speed * time_to_sight
  let remaining_side : ℝ := side_length - (thief_distance % side_length)
  let chase_time : ℝ := Real.sqrt ((remaining_side^2) / (police_speed^2 - thief_speed^2))
  time_to_sight + chase_time = time_to_catch :=
by sorry

end NUMINAMATH_CALUDE_police_catches_thief_l1047_104781


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1047_104762

theorem algebraic_expression_value (m n : ℝ) (h : -2*m + 3*n^2 = -7) : 
  12*n^2 - 8*m + 4 = -24 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1047_104762


namespace NUMINAMATH_CALUDE_dairy_farm_husk_consumption_l1047_104710

/-- Given a dairy farm scenario where multiple cows eat multiple bags of husk over multiple days,
    this theorem proves that the number of days for one cow to eat one bag is the same as the
    total number of days for all cows to eat all bags. -/
theorem dairy_farm_husk_consumption
  (num_cows : ℕ)
  (num_bags : ℕ)
  (num_days : ℕ)
  (h_cows : num_cows = 46)
  (h_bags : num_bags = 46)
  (h_days : num_days = 46)
  : num_days = (num_days * num_cows) / num_cows :=
by
  sorry

#check dairy_farm_husk_consumption

end NUMINAMATH_CALUDE_dairy_farm_husk_consumption_l1047_104710


namespace NUMINAMATH_CALUDE_coat_price_calculation_l1047_104759

def calculate_final_price (initial_price : ℝ) (initial_tax_rate : ℝ) 
                          (discount_rate : ℝ) (additional_discount : ℝ) 
                          (final_tax_rate : ℝ) : ℝ :=
  let price_after_initial_tax := initial_price * (1 + initial_tax_rate)
  let price_after_discount := price_after_initial_tax * (1 - discount_rate)
  let price_after_additional_discount := price_after_discount - additional_discount
  price_after_additional_discount * (1 + final_tax_rate)

theorem coat_price_calculation :
  calculate_final_price 200 0.10 0.25 10 0.05 = 162.75 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_calculation_l1047_104759


namespace NUMINAMATH_CALUDE_camping_trip_percentage_l1047_104739

theorem camping_trip_percentage (total_students : ℝ) (students_over_100 : ℝ) (students_100_or_less : ℝ) :
  students_over_100 = 0.16 * total_students →
  students_over_100 + students_100_or_less = 0.64 * total_students →
  (students_over_100 + students_100_or_less) / total_students = 0.64 :=
by
  sorry

end NUMINAMATH_CALUDE_camping_trip_percentage_l1047_104739


namespace NUMINAMATH_CALUDE_beanie_babies_per_stocking_l1047_104771

theorem beanie_babies_per_stocking : 
  ∀ (candy_canes_per_stocking : ℕ) 
    (books_per_stocking : ℕ) 
    (num_stockings : ℕ) 
    (total_stuffers : ℕ),
  candy_canes_per_stocking = 4 →
  books_per_stocking = 1 →
  num_stockings = 3 →
  total_stuffers = 21 →
  (total_stuffers - (candy_canes_per_stocking + books_per_stocking) * num_stockings) / num_stockings = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_beanie_babies_per_stocking_l1047_104771


namespace NUMINAMATH_CALUDE_crease_length_l1047_104782

theorem crease_length (width : Real) (θ : Real) : width = 10 → 
  let crease_length := width / 2 * Real.tan θ
  crease_length = 5 * Real.tan θ := by sorry

end NUMINAMATH_CALUDE_crease_length_l1047_104782


namespace NUMINAMATH_CALUDE_pizza_slices_count_l1047_104737

-- Define the number of pizzas
def num_pizzas : Nat := 4

-- Define the number of slices for each type of pizza
def slices_first_two : Nat := 8
def slices_third : Nat := 10
def slices_fourth : Nat := 12

-- Define the total number of slices
def total_slices : Nat := 2 * slices_first_two + slices_third + slices_fourth

-- Theorem to prove
theorem pizza_slices_count : total_slices = 38 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_count_l1047_104737


namespace NUMINAMATH_CALUDE_max_sum_with_lcm_gcd_constraint_l1047_104753

theorem max_sum_with_lcm_gcd_constraint (m n : ℕ) : 
  m + 3*n - 5 = 2*(Nat.lcm m n) - 11*(Nat.gcd m n) → 
  m + n ≤ 70 ∧ ∃ (m₀ n₀ : ℕ), m₀ + 3*n₀ - 5 = 2*(Nat.lcm m₀ n₀) - 11*(Nat.gcd m₀ n₀) ∧ m₀ + n₀ = 70 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_with_lcm_gcd_constraint_l1047_104753


namespace NUMINAMATH_CALUDE_merchant_profit_percentage_l1047_104729

theorem merchant_profit_percentage
  (markup_rate : ℝ)
  (discount_rate : ℝ)
  (h_markup : markup_rate = 0.40)
  (h_discount : discount_rate = 0.15) :
  let marked_price := 1 + markup_rate
  let selling_price := marked_price * (1 - discount_rate)
  let profit_percentage := (selling_price - 1) * 100
  profit_percentage = 19 := by
sorry

end NUMINAMATH_CALUDE_merchant_profit_percentage_l1047_104729


namespace NUMINAMATH_CALUDE_initial_value_theorem_l1047_104774

theorem initial_value_theorem :
  ∃ (n : ℤ) (initial_value : ℤ), 
    initial_value = 136 * n - 21 ∧ 
    ∃ (added_value : ℤ), initial_value + added_value = 136 * n ∧ 
    (added_value ≥ 20 ∧ added_value ≤ 22) := by
  sorry

end NUMINAMATH_CALUDE_initial_value_theorem_l1047_104774


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l1047_104780

theorem power_fraction_simplification :
  ((2^5) * (9^2)) / ((8^2) * (3^5)) = 1/6 := by sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l1047_104780


namespace NUMINAMATH_CALUDE_cube_opposite_faces_l1047_104741

/-- Represents a face of a cube --/
inductive Face : Type
| G | H | I | J | S | K

/-- Represents the adjacency relation between faces --/
def adjacent : Face → Face → Prop := sorry

/-- Represents the opposite relation between faces --/
def opposite : Face → Face → Prop := sorry

/-- Theorem: If H and I are adjacent, G is adjacent to both H and I, 
    and J is adjacent to H and I, then J is opposite to G --/
theorem cube_opposite_faces 
  (adj_H_I : adjacent Face.H Face.I)
  (adj_G_H : adjacent Face.G Face.H)
  (adj_G_I : adjacent Face.G Face.I)
  (adj_J_H : adjacent Face.J Face.H)
  (adj_J_I : adjacent Face.J Face.I) :
  opposite Face.G Face.J := by sorry

end NUMINAMATH_CALUDE_cube_opposite_faces_l1047_104741


namespace NUMINAMATH_CALUDE_travel_time_calculation_l1047_104717

/-- Travel time calculation given distance and average speed -/
theorem travel_time_calculation 
  (distance : ℝ) 
  (average_speed : ℝ) 
  (h1 : distance = 790) 
  (h2 : average_speed = 50) :
  distance / average_speed = 15.8 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l1047_104717


namespace NUMINAMATH_CALUDE_vote_difference_is_40_l1047_104783

-- Define the committee and voting scenario
def CommitteeVoting (total_members : ℕ) (initial_for initial_against revote_for revote_against : ℕ) : Prop :=
  -- Total members condition
  total_members = initial_for + initial_against ∧
  total_members = revote_for + revote_against ∧
  -- Initially rejected condition
  initial_against > initial_for ∧
  -- Re-vote margin condition
  (revote_for - revote_against) = 3 * (initial_against - initial_for) ∧
  -- Re-vote for vs initial against condition
  revote_for * 12 = initial_against * 13

-- Theorem statement
theorem vote_difference_is_40 :
  ∀ (initial_for initial_against revote_for revote_against : ℕ),
    CommitteeVoting 500 initial_for initial_against revote_for revote_against →
    revote_for - initial_for = 40 := by
  sorry

end NUMINAMATH_CALUDE_vote_difference_is_40_l1047_104783


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l1047_104799

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem billion_to_scientific_notation :
  toScientificNotation 1673000000 = ScientificNotation.mk 1.673 9 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l1047_104799


namespace NUMINAMATH_CALUDE_curve_and_intersection_l1047_104791

-- Define the curve C
def C (x y : ℝ) : Prop :=
  Real.sqrt ((x - 0)^2 + (y - (-Real.sqrt 3))^2) +
  Real.sqrt ((x - 0)^2 + (y - Real.sqrt 3)^2) = 4

-- Define the line that intersects C
def intersecting_line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 1

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

-- Theorem statement
theorem curve_and_intersection :
  ∃ (k : ℝ),
    (∀ x y, C x y ↔ x^2 + y^2/4 = 1) ∧
    (∃ x₁ y₁ x₂ y₂,
      C x₁ y₁ ∧ C x₂ y₂ ∧
      intersecting_line k x₁ y₁ ∧
      intersecting_line k x₂ y₂ ∧
      perpendicular x₁ y₁ x₂ y₂ ∧
      (k = 1/2 ∨ k = -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_curve_and_intersection_l1047_104791


namespace NUMINAMATH_CALUDE_rectangle_to_total_height_ratio_l1047_104765

/-- Represents an octagon with specific properties -/
structure Octagon :=
  (area : ℝ)
  (rectangle_width : ℝ)
  (triangle_base : ℝ)

/-- Properties of the octagon -/
axiom octagon_properties (o : Octagon) :
  o.area = 12 ∧
  o.rectangle_width = 3 ∧
  o.triangle_base = 3

/-- The diagonal bisects the area of the octagon -/
axiom diagonal_bisects (o : Octagon) (rectangle_height : ℝ) :
  o.rectangle_width * rectangle_height = o.area / 2

/-- The total height of the octagon -/
def total_height (o : Octagon) (rectangle_height : ℝ) : ℝ :=
  2 * rectangle_height

/-- Theorem: The ratio of rectangle height to total height is 1/2 -/
theorem rectangle_to_total_height_ratio (o : Octagon) (rectangle_height : ℝ) :
  rectangle_height / (total_height o rectangle_height) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_total_height_ratio_l1047_104765


namespace NUMINAMATH_CALUDE_aunt_gemma_dog_food_duration_l1047_104767

/-- Calculates the number of days dog food will last given the number of dogs, 
    feeding frequency, food consumption per meal, and amount of food bought. -/
def dogFoodDuration (numDogs : ℕ) (feedingsPerDay : ℕ) (gramsPerMeal : ℕ) 
                    (numSacks : ℕ) (kgPerSack : ℕ) : ℕ :=
  let dailyConsumptionGrams := numDogs * feedingsPerDay * gramsPerMeal
  let totalFoodKg := numSacks * kgPerSack
  totalFoodKg * 1000 / dailyConsumptionGrams

theorem aunt_gemma_dog_food_duration :
  dogFoodDuration 4 2 250 2 50 = 50 := by
  sorry

#eval dogFoodDuration 4 2 250 2 50

end NUMINAMATH_CALUDE_aunt_gemma_dog_food_duration_l1047_104767


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_m_value_l1047_104712

/-- Given vectors a and b in R², prove that if 2a + b is parallel to a - 2b, then m = -1/2 --/
theorem parallel_vectors_imply_m_value (m : ℝ) :
  let a : ℝ × ℝ := (1, m)
  let b : ℝ × ℝ := (2, -1)
  (∃ (k : ℝ), k ≠ 0 ∧ (2 • a + b) = k • (a - 2 • b)) →
  m = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_m_value_l1047_104712


namespace NUMINAMATH_CALUDE_min_value_triangle_sides_l1047_104748

/-- 
Given a triangle with side lengths x+10, x+5, and 4x, where the angle opposite to side 4x
is the largest angle, the minimum value of 4x - (x+5) is 5.
-/
theorem min_value_triangle_sides (x : ℝ) : 
  (x + 5 + 4*x > x + 10) ∧ 
  (x + 5 + x + 10 > 4*x) ∧ 
  (4*x + x + 10 > x + 5) ∧
  (4*x > x + 5) ∧ 
  (4*x > x + 10) →
  ∃ (y : ℝ), y ≥ x ∧ ∀ (z : ℝ), z ≥ x → 4*z - (z + 5) ≥ 4*y - (y + 5) ∧ 4*y - (y + 5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_triangle_sides_l1047_104748


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1047_104726

theorem fraction_multiplication : (-1/6 + 3/4 - 5/12) * 48 = 8 := by sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1047_104726


namespace NUMINAMATH_CALUDE_andre_gave_23_flowers_l1047_104738

/-- The number of flowers Rosa initially had -/
def initial_flowers : ℕ := 67

/-- The number of flowers Rosa has now -/
def final_flowers : ℕ := 90

/-- The number of flowers Andre gave to Rosa -/
def andre_flowers : ℕ := final_flowers - initial_flowers

theorem andre_gave_23_flowers : andre_flowers = 23 := by
  sorry

end NUMINAMATH_CALUDE_andre_gave_23_flowers_l1047_104738


namespace NUMINAMATH_CALUDE_square_sum_equality_l1047_104732

theorem square_sum_equality (x y : ℝ) (h : x + y = -2) : x^2 + y^2 + 2*x*y = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l1047_104732


namespace NUMINAMATH_CALUDE_parabola_directrix_l1047_104786

/-- The directrix of the parabola y = (x^2 - 8x + 12) / 16 is y = -17/64 -/
theorem parabola_directrix :
  let f : ℝ → ℝ := λ x => (x^2 - 8*x + 12) / 16
  ∃ (a b c : ℝ), (∀ x, f x = a * (x - b)^2 + c) ∧
                 (a ≠ 0) ∧
                 (c - 1 / (4 * a) = -17/64) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1047_104786


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1047_104766

/-- Simple interest calculation -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Problem statement -/
theorem interest_rate_calculation (principal time interest : ℝ) 
  (h1 : principal = 500)
  (h2 : time = 4)
  (h3 : interest = 90) :
  ∃ (rate : ℝ), simple_interest principal rate time = interest ∧ rate = 0.045 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l1047_104766


namespace NUMINAMATH_CALUDE_k_not_determined_l1047_104725

theorem k_not_determined (k r : ℝ) (a : ℝ → ℝ) :
  (∀ r, a r = (k * r)^3) →
  (a (r / 2) = 0.125 * a r) →
  True
:= by sorry

end NUMINAMATH_CALUDE_k_not_determined_l1047_104725


namespace NUMINAMATH_CALUDE_roots_are_cosines_of_triangle_angles_l1047_104702

-- Define the polynomial p(x)
def p (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the condition
def condition (a b c : ℝ) : Prop := a^2 - 2*b - 2*c = 1

-- Theorem statement
theorem roots_are_cosines_of_triangle_angles 
  (a b c : ℝ) 
  (h_positive_roots : ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    ∀ t : ℝ, p a b c t = 0 ↔ t = x ∨ t = y ∨ t = z) :
  condition a b c ↔ 
  ∃ A B C : ℝ, 
    0 < A ∧ A < π/2 ∧
    0 < B ∧ B < π/2 ∧
    0 < C ∧ C < π/2 ∧
    A + B + C = π ∧
    (∀ t : ℝ, p a b c t = 0 ↔ t = Real.cos A ∨ t = Real.cos B ∨ t = Real.cos C) :=
by sorry

end NUMINAMATH_CALUDE_roots_are_cosines_of_triangle_angles_l1047_104702


namespace NUMINAMATH_CALUDE_min_value_divisible_by_72_l1047_104718

def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = b * k

theorem min_value_divisible_by_72 (x y : ℕ) (h1 : x ≥ 4) 
  (h2 : is_divisible_by (98348 * 10 + x * 10 + y) 72) : y = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_divisible_by_72_l1047_104718


namespace NUMINAMATH_CALUDE_water_pressure_force_on_trapezoidal_dam_l1047_104711

/-- The force of water pressure on a trapezoidal dam -/
theorem water_pressure_force_on_trapezoidal_dam 
  (ρ : Real) (g : Real) (a b h : Real) : 
  ρ = 1000 →
  g = 10 →
  a = 6.9 →
  b = 11.4 →
  h = 5.0 →
  ρ * g * h^2 * (b / 2 - (b - a) * h / (6 * h)) = 1050000 := by
  sorry

end NUMINAMATH_CALUDE_water_pressure_force_on_trapezoidal_dam_l1047_104711


namespace NUMINAMATH_CALUDE_x_power_4095_minus_reciprocal_l1047_104793

theorem x_power_4095_minus_reciprocal (x : ℝ) (h : x - 1/x = Real.sqrt 2) :
  x^4095 - 1/x^4095 = 20 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_x_power_4095_minus_reciprocal_l1047_104793


namespace NUMINAMATH_CALUDE_remainder_theorem_l1047_104722

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 25 * k - 1) :
  (n^2 + 3*n + 5) % 25 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1047_104722


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_not_necessary_condition_l1047_104707

/-- Proposition P: segments of lengths a, b, c can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Proposition Q: a² + b² + c² < 2(ab + bc + ca) -/
def inequality_holds (a b c : ℝ) : Prop :=
  a^2 + b^2 + c^2 < 2*(a*b + b*c + c*a)

theorem triangle_inequality_theorem (a b c : ℝ) :
  can_form_triangle a b c → inequality_holds a b c :=
sorry

theorem not_necessary_condition :
  ∃ a b c : ℝ, inequality_holds a b c ∧ ¬can_form_triangle a b c :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_not_necessary_condition_l1047_104707


namespace NUMINAMATH_CALUDE_polynomial_coefficients_l1047_104768

theorem polynomial_coefficients 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) : 
  (∀ x : ℝ, (x + 2) * (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  (a₁ - a₂ + a₃ - a₄ + a₅ - a₆ = 241 ∧ a₂ = -70) := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficients_l1047_104768


namespace NUMINAMATH_CALUDE_m_value_l1047_104747

theorem m_value (m : ℝ) (M : Set ℝ) : M = {3, m + 1} → 4 ∈ M → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_m_value_l1047_104747


namespace NUMINAMATH_CALUDE_computer_price_decrease_l1047_104721

/-- The price of a computer after a certain number of years, given an initial price and a rate of decrease every 3 years. -/
def price_after_years (initial_price : ℝ) (decrease_rate : ℝ) (years : ℕ) : ℝ :=
  initial_price * (1 - decrease_rate) ^ (years / 3)

/-- Theorem stating that the price of a computer initially priced at 8100 yuan,
    decreasing by 1/3 every 3 years, will be 2400 yuan after 9 years. -/
theorem computer_price_decrease :
  price_after_years 8100 (1/3) 9 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_decrease_l1047_104721


namespace NUMINAMATH_CALUDE_max_area_rectangle_fixed_perimeter_l1047_104796

/-- The maximum area of a rectangle with perimeter 30 meters is 56.25 square meters. -/
theorem max_area_rectangle_fixed_perimeter :
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ 2 * (l + w) = 30 ∧
  (∀ (l' w' : ℝ), l' > 0 → w' > 0 → 2 * (l' + w') = 30 → l' * w' ≤ l * w) ∧
  l * w = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_fixed_perimeter_l1047_104796


namespace NUMINAMATH_CALUDE_m_greater_than_n_l1047_104784

theorem m_greater_than_n : ∀ a : ℝ, 2 * a^2 - 4 * a > a^2 - 2 * a - 3 := by
  sorry

end NUMINAMATH_CALUDE_m_greater_than_n_l1047_104784


namespace NUMINAMATH_CALUDE_function_inequality_l1047_104740

theorem function_inequality (a x : ℝ) : 
  let f := fun (t : ℝ) => t^2 - t + 13
  |x - a| < 1 → |f x - f a| < 2 * (|a| + 1) := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l1047_104740


namespace NUMINAMATH_CALUDE_average_price_per_book_l1047_104777

-- Define the problem parameters
def books_shop1 : ℕ := 32
def cost_shop1 : ℕ := 1500
def books_shop2 : ℕ := 60
def cost_shop2 : ℕ := 340

-- Theorem to prove
theorem average_price_per_book :
  (cost_shop1 + cost_shop2) / (books_shop1 + books_shop2) = 20 := by
  sorry


end NUMINAMATH_CALUDE_average_price_per_book_l1047_104777


namespace NUMINAMATH_CALUDE_solve_cube_equation_l1047_104764

theorem solve_cube_equation : ∃ x : ℝ, (x - 3)^3 = (1/27)⁻¹ ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_cube_equation_l1047_104764


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1047_104734

theorem polynomial_division_theorem (x : ℝ) : 
  x^6 + 2*x^4 - 5*x^3 + 9 = 
  (x - 2) * (x^5 + 2*x^4 + 6*x^3 + 7*x^2 + 14*x + 28) + R :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1047_104734


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1047_104778

theorem polynomial_division_remainder 
  (x : ℝ) : 
  ∃ (q : ℝ → ℝ), 
  x^4 - 8*x^3 + 18*x^2 - 27*x + 15 = 
  (x^2 - 3*x + 14/3) * q x + (2*x + 205/9) :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1047_104778


namespace NUMINAMATH_CALUDE_expansion_equality_l1047_104728

theorem expansion_equality (a b : ℝ) : (a - b) * (-a - b) = b^2 - a^2 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equality_l1047_104728


namespace NUMINAMATH_CALUDE_find_number_l1047_104751

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 9) = 63 :=
  sorry

end NUMINAMATH_CALUDE_find_number_l1047_104751


namespace NUMINAMATH_CALUDE_quadratic_trinomial_decomposition_l1047_104792

theorem quadratic_trinomial_decomposition (a b c : ℝ) :
  ∃ (p q r s t u : ℝ), 
    (∀ x : ℝ, a * x^2 + b * x + c = (p * x^2 + q * x + r) + (s * x^2 + t * x + u)) ∧
    (q^2 - 4*p*r = 0) ∧
    (t^2 - 4*s*u = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_decomposition_l1047_104792


namespace NUMINAMATH_CALUDE_absolute_value_squared_l1047_104730

theorem absolute_value_squared (a b : ℝ) : |a| > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_squared_l1047_104730
