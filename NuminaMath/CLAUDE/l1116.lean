import Mathlib

namespace NUMINAMATH_CALUDE_circular_center_ratio_l1116_111637

/-- Represents a square flag with a symmetric cross design -/
structure SymmetricCrossFlag where
  side : ℝ
  cross_area_ratio : ℝ
  (cross_area_valid : cross_area_ratio = 1/4)

/-- The area of the circular center of the cross -/
noncomputable def circular_center_area (flag : SymmetricCrossFlag) : ℝ :=
  (flag.cross_area_ratio * flag.side^2) / 4

theorem circular_center_ratio (flag : SymmetricCrossFlag) :
  circular_center_area flag / flag.side^2 = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_circular_center_ratio_l1116_111637


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_divisibility_l1116_111671

theorem consecutive_odd_numbers_divisibility (a b c : ℤ) : 
  (∃ k : ℤ, b = 2 * k + 1) →  -- b is odd
  (a = b - 2) →              -- a is the previous odd number
  (c = b + 2) →              -- c is the next odd number
  ∃ m : ℤ, a * b * c + 4 * b = m * b^3 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_divisibility_l1116_111671


namespace NUMINAMATH_CALUDE_problem_solution_l1116_111684

noncomputable def f (a x : ℝ) : ℝ :=
  if x < a then 2 * a - (x + 4 / x) else x - 4 / x

theorem problem_solution (a : ℝ) :
  (a = 1 → ∃! x, f a x = 3 ∧ x = 4) ∧
  (a ≤ -1 →
    (∃ x₁ x₂ x₃, x₁ < x₂ ∧ x₂ < x₃ ∧
      f a x₁ = 3 ∧ f a x₂ = 3 ∧ f a x₃ = 3 ∧
      x₃ - x₂ = x₂ - x₁) →
    a = -11/6) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1116_111684


namespace NUMINAMATH_CALUDE_equation_solution_l1116_111649

theorem equation_solution (k : ℝ) : 
  (7 * (-1)^3 - 3 * (-1)^2 + k * (-1) + 5 = 0) → 
  (k^3 + 2 * k^2 - 11 * k - 85 = -105) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1116_111649


namespace NUMINAMATH_CALUDE_water_jars_problem_l1116_111690

theorem water_jars_problem (total_volume : ℚ) (x : ℕ) : 
  total_volume = 42 →
  (x : ℚ) * (1/4 + 1/2 + 1) = total_volume →
  3 * x = 72 :=
by sorry

end NUMINAMATH_CALUDE_water_jars_problem_l1116_111690


namespace NUMINAMATH_CALUDE_rectangle_cylinder_max_volume_l1116_111632

theorem rectangle_cylinder_max_volume (x y : Real) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 9) :
  let V := π * x * y^2
  (∀ x' y' : Real, x' > 0 → y' > 0 → x' + y' = 9 → π * x' * y'^2 ≤ π * x * y^2) →
  x = 6 ∧ V = 108 * π :=
by sorry

end NUMINAMATH_CALUDE_rectangle_cylinder_max_volume_l1116_111632


namespace NUMINAMATH_CALUDE_original_equals_scientific_l1116_111647

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coefficientInRange : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def originalNumber : ℕ := 44300000

/-- The scientific notation representation of the original number -/
def scientificForm : ScientificNotation := {
  coefficient := 4.43
  exponent := 7
  coefficientInRange := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific : 
  (originalNumber : ℝ) = scientificForm.coefficient * (10 : ℝ) ^ scientificForm.exponent := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l1116_111647


namespace NUMINAMATH_CALUDE_parabola_intersection_sum_l1116_111612

/-- The parabola P with equation y = x^2 -/
def P : Set (ℝ × ℝ) := {p | p.2 = p.1^2}

/-- The point R -/
def R : ℝ × ℝ := (10, -6)

/-- The line through R with slope n -/
def line_through_R (n : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 + 6 = n * (p.1 - 10)}

/-- The condition for non-intersection -/
def no_intersection (n : ℝ) : Prop :=
  line_through_R n ∩ P = ∅

theorem parabola_intersection_sum (a b : ℝ) :
  (∀ n, no_intersection n ↔ a < n ∧ n < b) →
  a + b = 40 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_sum_l1116_111612


namespace NUMINAMATH_CALUDE_triangle_side_sum_l1116_111658

theorem triangle_side_sum (A B C : ℝ) (a b c : ℝ) :
  B = 2 * A →
  Real.cos A = 4 / 5 →
  (1 / 2) * a * b * Real.sin C = 468 / 25 →
  a + b = 13 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l1116_111658


namespace NUMINAMATH_CALUDE_washington_dc_july_4th_avg_temp_l1116_111648

def washington_dc_july_4th_temps : List ℝ := [90, 90, 90, 79, 71]

theorem washington_dc_july_4th_avg_temp :
  (washington_dc_july_4th_temps.sum / washington_dc_july_4th_temps.length : ℝ) = 84 := by
  sorry

end NUMINAMATH_CALUDE_washington_dc_july_4th_avg_temp_l1116_111648


namespace NUMINAMATH_CALUDE_final_payment_calculation_l1116_111696

/-- Calculates the final amount John will pay for three articles with given costs and discounts, including sales tax. -/
theorem final_payment_calculation (cost_A cost_B cost_C : ℝ)
  (discount_A discount_B discount_C : ℝ) (sales_tax_rate : ℝ)
  (h_cost_A : cost_A = 200)
  (h_cost_B : cost_B = 300)
  (h_cost_C : cost_C = 400)
  (h_discount_A : discount_A = 0.5)
  (h_discount_B : discount_B = 0.3)
  (h_discount_C : discount_C = 0.4)
  (h_sales_tax : sales_tax_rate = 0.05) :
  let discounted_A := cost_A * (1 - discount_A)
  let discounted_B := cost_B * (1 - discount_B)
  let discounted_C := cost_C * (1 - discount_C)
  let total_discounted := discounted_A + discounted_B + discounted_C
  let final_amount := total_discounted * (1 + sales_tax_rate)
  final_amount = 577.5 := by sorry


end NUMINAMATH_CALUDE_final_payment_calculation_l1116_111696


namespace NUMINAMATH_CALUDE_fifteen_sided_polygon_diagonals_l1116_111633

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 15 sides has 90 diagonals -/
theorem fifteen_sided_polygon_diagonals :
  num_diagonals 15 = 90 := by sorry

end NUMINAMATH_CALUDE_fifteen_sided_polygon_diagonals_l1116_111633


namespace NUMINAMATH_CALUDE_intersection_subsets_count_l1116_111642

def M : Finset ℕ := {0, 1, 2, 3, 4}
def N : Finset ℕ := {1, 3, 5}

theorem intersection_subsets_count :
  Finset.card (Finset.powerset (M ∩ N)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_subsets_count_l1116_111642


namespace NUMINAMATH_CALUDE_cube_of_102_l1116_111600

theorem cube_of_102 : (100 + 2)^3 = 1061208 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_102_l1116_111600


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1116_111662

theorem negation_of_universal_proposition :
  (¬ ∀ n : ℕ, 3^n ≥ n^2 + 1) ↔ (∃ n₀ : ℕ, 3^n₀ < n₀^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1116_111662


namespace NUMINAMATH_CALUDE_final_S_value_l1116_111654

def sequence_A : ℕ → ℕ
  | 0 => 1
  | n + 1 => sequence_A n + 1

def sequence_S : ℕ → ℕ
  | 0 => 0
  | n + 1 => sequence_S n + sequence_A (n + 1)

theorem final_S_value :
  ∃ n : ℕ, sequence_S n ≤ 36 ∧ sequence_S (n + 1) > 36 ∧ sequence_S (n + 1) = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_final_S_value_l1116_111654


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l1116_111693

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the y-axis -/
def symmetricYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

theorem symmetric_point_coordinates :
  let A : Point2D := { x := 2, y := -8 }
  let B : Point2D := symmetricYAxis A
  B.x = -2 ∧ B.y = -8 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l1116_111693


namespace NUMINAMATH_CALUDE_ship_grain_calculation_l1116_111668

/-- The amount of grain (in tons) that spilled into the water -/
def spilled_grain : ℕ := 49952

/-- The amount of grain (in tons) that remained onboard -/
def remaining_grain : ℕ := 918

/-- The original amount of grain (in tons) on the ship -/
def original_grain : ℕ := spilled_grain + remaining_grain

theorem ship_grain_calculation : original_grain = 50870 := by
  sorry

end NUMINAMATH_CALUDE_ship_grain_calculation_l1116_111668


namespace NUMINAMATH_CALUDE_bridge_length_l1116_111604

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 150 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 225 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l1116_111604


namespace NUMINAMATH_CALUDE_max_pieces_20x24_cake_l1116_111685

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Represents a piece of cake -/
structure CakePiece where
  size : Dimensions

/-- Represents the whole cake -/
structure Cake where
  size : Dimensions

/-- Calculates the maximum number of pieces that can be cut from a cake -/
def maxPieces (cake : Cake) (piece : CakePiece) : ℕ :=
  let horizontal := (cake.size.length / piece.size.length) * (cake.size.width / piece.size.width)
  let vertical := (cake.size.length / piece.size.width) * (cake.size.width / piece.size.length)
  max horizontal vertical

theorem max_pieces_20x24_cake (cake : Cake) (piece : CakePiece) :
  cake.size = Dimensions.mk 20 24 →
  piece.size = Dimensions.mk 4 4 →
  maxPieces cake piece = 30 := by
  sorry

#eval maxPieces (Cake.mk (Dimensions.mk 20 24)) (CakePiece.mk (Dimensions.mk 4 4))

end NUMINAMATH_CALUDE_max_pieces_20x24_cake_l1116_111685


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1116_111672

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_properties :
  ∀ d : ℤ,
  (arithmetic_sequence 23 d 6 > 0) →
  (arithmetic_sequence 23 d 7 < 0) →
  (d = -4) ∧
  (∀ n : ℕ, sum_arithmetic_sequence 23 d n ≤ 78) ∧
  (∀ n : ℕ, n ≤ 12 ↔ sum_arithmetic_sequence 23 d n > 0) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1116_111672


namespace NUMINAMATH_CALUDE_married_couples_children_l1116_111622

/-- The fraction of married couples with more than one child -/
def fraction_more_than_one_child : ℚ := 3/5

/-- The fraction of married couples with more than 3 children -/
def fraction_more_than_three_children : ℚ := 2/5

/-- The fraction of married couples with 2 or 3 children -/
def fraction_two_or_three_children : ℚ := 1/5

theorem married_couples_children :
  fraction_more_than_one_child = 
    fraction_more_than_three_children + fraction_two_or_three_children :=
by sorry

end NUMINAMATH_CALUDE_married_couples_children_l1116_111622


namespace NUMINAMATH_CALUDE_stratified_sampling_sophomores_l1116_111691

theorem stratified_sampling_sophomores (total_students : ℕ) (sophomore_students : ℕ) (sample_size : ℕ)
  (h1 : total_students = 4500)
  (h2 : sophomore_students = 1500)
  (h3 : sample_size = 600) :
  (sophomore_students : ℚ) / total_students * sample_size = 200 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_sophomores_l1116_111691


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l1116_111650

/-- Given vectors a and b, if (k*a + b) is parallel to (a - 3*b), then k = -1/3 --/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
    (ha : a = (1, 2))
    (hb : b = (-3, 2))
    (h_parallel : ∃ (t : ℝ), t • (k • a + b) = (a - 3 • b)) :
  k = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l1116_111650


namespace NUMINAMATH_CALUDE_satisfying_function_is_identity_l1116_111646

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℕ → ℕ) : Prop :=
  (∀ k : ℕ, k > 0 → ∃ S : Set ℕ, S.Infinite ∧ ∀ p ∈ S, Prime p ∧ ∃ c : ℕ, c > 0 ∧ f c = p^k) ∧
  (∀ m n : ℕ, m > 0 ∧ n > 0 → (f m + f n) ∣ f (m + n))

/-- The main theorem stating that any function satisfying the conditions is the identity function -/
theorem satisfying_function_is_identity (f : ℕ → ℕ) (h : SatisfyingFunction f) : 
  ∀ n : ℕ, f n = n := by
  sorry

end NUMINAMATH_CALUDE_satisfying_function_is_identity_l1116_111646


namespace NUMINAMATH_CALUDE_always_positive_expression_l1116_111620

theorem always_positive_expression (a : ℝ) : |a| + 2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_always_positive_expression_l1116_111620


namespace NUMINAMATH_CALUDE_function_properties_l1116_111681

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

theorem function_properties :
  ∀ (a : ℝ),
  (∀ (x : ℝ), -5 ≤ x ∧ x ≤ 5 → 
    (a = -1 → 
      (∀ (y : ℝ), -5 ≤ y ∧ y ≤ 5 → f a y ≤ 37) ∧
      (∃ (y : ℝ), -5 ≤ y ∧ y ≤ 5 ∧ f a y = 37) ∧
      (∀ (y : ℝ), -5 ≤ y ∧ y ≤ 5 → f a y ≥ 1) ∧
      (∃ (y : ℝ), -5 ≤ y ∧ y ≤ 5 ∧ f a y = 1)) ∧
    ((-5 < a ∧ a < 5) ↔ 
      (∃ (y z : ℝ), -5 ≤ y ∧ y < z ∧ z ≤ 5 ∧ f a y > f a z)) ∧
    ((-5 < a ∧ a < 0) → 
      (∀ (y : ℝ), -5 ≤ y ∧ y ≤ 5 → f a y ≤ 27 - 10*a) ∧
      (∃ (y : ℝ), -5 ≤ y ∧ y ≤ 5 ∧ f a y = 27 - 10*a)) ∧
    ((0 ≤ a ∧ a < 5) → 
      (∀ (y : ℝ), -5 ≤ y ∧ y ≤ 5 → f a y ≤ 27 + 10*a) ∧
      (∃ (y : ℝ), -5 ≤ y ∧ y ≤ 5 ∧ f a y = 27 + 10*a))) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1116_111681


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1116_111638

theorem sqrt_equation_solution :
  let f (x : ℝ) := Real.sqrt (3 * x - 5) + 14 / Real.sqrt (3 * x - 5)
  ∀ x : ℝ, f x = 8 ↔ x = (23 + 8 * Real.sqrt 2) / 3 ∨ x = (23 - 8 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1116_111638


namespace NUMINAMATH_CALUDE_stratified_sampling_11th_grade_l1116_111652

theorem stratified_sampling_11th_grade (total_students : ℕ) (eleventh_grade_students : ℕ) (sample_size : ℕ) :
  total_students = 5000 →
  eleventh_grade_students = 1500 →
  sample_size = 30 →
  (eleventh_grade_students : ℚ) / (total_students : ℚ) * (sample_size : ℚ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_11th_grade_l1116_111652


namespace NUMINAMATH_CALUDE_distinct_paintings_l1116_111618

/-- The number of disks in the circle -/
def n : ℕ := 12

/-- The number of disks to be painted blue -/
def blue : ℕ := 4

/-- The number of disks to be painted red -/
def red : ℕ := 3

/-- The number of disks to be painted green -/
def green : ℕ := 2

/-- The total number of disks to be painted -/
def painted : ℕ := blue + red + green

/-- The number of rotational symmetries of the circle -/
def symmetries : ℕ := n

/-- The number of ways to color the disks without considering symmetry -/
def total_colorings : ℕ := Nat.choose n blue * Nat.choose (n - blue) red * Nat.choose (n - blue - red) green

/-- The number of distinct paintings considering rotational symmetry -/
theorem distinct_paintings : (total_colorings / symmetries : ℚ) = 23100 := by
  sorry

end NUMINAMATH_CALUDE_distinct_paintings_l1116_111618


namespace NUMINAMATH_CALUDE_smallest_n_not_divisible_by_ten_l1116_111624

theorem smallest_n_not_divisible_by_ten (n : ℕ) :
  (n > 2016 ∧ ¬(10 ∣ (1^n + 2^n + 3^n + 4^n)) ∧
   ∀ m, 2016 < m ∧ m < n → (10 ∣ (1^m + 2^m + 3^m + 4^m))) →
  n = 2020 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_not_divisible_by_ten_l1116_111624


namespace NUMINAMATH_CALUDE_negative_plus_abs_neg_l1116_111695

theorem negative_plus_abs_neg (a : ℝ) (h : a < 0) : a + |-a| = 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_plus_abs_neg_l1116_111695


namespace NUMINAMATH_CALUDE_certain_number_calculation_l1116_111651

theorem certain_number_calculation : 5 * 3 + 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_calculation_l1116_111651


namespace NUMINAMATH_CALUDE_ribbon_length_difference_equals_side_length_specific_box_ribbon_difference_l1116_111669

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the ribbon length for the first method -/
def ribbonLength1 (box : BoxDimensions) (bowLength : ℝ) : ℝ :=
  2 * box.length + 2 * box.width + 4 * box.height + bowLength

/-- Calculates the ribbon length for the second method -/
def ribbonLength2 (box : BoxDimensions) (bowLength : ℝ) : ℝ :=
  2 * box.length + 4 * box.width + 2 * box.height + bowLength

/-- The main theorem to prove -/
theorem ribbon_length_difference_equals_side_length 
  (box : BoxDimensions) (bowLength : ℝ) : 
  ribbonLength2 box bowLength - ribbonLength1 box bowLength = box.length :=
by
  sorry

/-- The specific case with given dimensions -/
theorem specific_box_ribbon_difference :
  let box : BoxDimensions := ⟨22, 22, 11⟩
  let bowLength : ℝ := 24
  ribbonLength2 box bowLength - ribbonLength1 box bowLength = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_ribbon_length_difference_equals_side_length_specific_box_ribbon_difference_l1116_111669


namespace NUMINAMATH_CALUDE_four_digit_number_theorem_l1116_111607

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def first_two_digits (n : ℕ) : ℕ := n / 100

def last_two_digits (n : ℕ) : ℕ := n % 100

def satisfies_conditions (n : ℕ) : Prop :=
  is_four_digit n ∧
  n % 3 = 0 ∧
  first_two_digits n - last_two_digits n = 11

def solution_set : Set ℕ := {1302, 1605, 1908, 2211, 2514, 2817, 3120, 3423, 3726, 4029, 4332, 4635, 4938, 5241, 5544, 5847, 6150, 6453, 6756, 7059, 7362, 7665, 7968, 8271, 8574, 8877, 9180, 9483, 9786, 10089, 10392, 10695, 10998}

theorem four_digit_number_theorem :
  {n : ℕ | satisfies_conditions n} = solution_set := by sorry

end NUMINAMATH_CALUDE_four_digit_number_theorem_l1116_111607


namespace NUMINAMATH_CALUDE_correct_product_after_decimal_error_l1116_111640

theorem correct_product_after_decimal_error (incorrect_product : ℝ) 
  (h1 : incorrect_product = 12.04) : 
  ∃ (factor1 factor2 : ℝ), 
    (0.01 ≤ factor1 ∧ factor1 < 1) ∧ 
    (factor1 * 100 * factor2 = incorrect_product) ∧
    (factor1 * factor2 = 0.1204) := by
  sorry

end NUMINAMATH_CALUDE_correct_product_after_decimal_error_l1116_111640


namespace NUMINAMATH_CALUDE_units_digit_problem_l1116_111602

theorem units_digit_problem :
  ∃ n : ℕ, (15 + Real.sqrt 221)^19 + 3 * (15 + Real.sqrt 221)^83 = 10 * n + 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l1116_111602


namespace NUMINAMATH_CALUDE_smallest_a_value_l1116_111621

theorem smallest_a_value (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) 
  (h : ∀ x : ℤ, Real.cos (a * ↑x + b) = Real.sin (36 * ↑x)) :
  ∀ a' ≥ 0, (∀ x : ℤ, ∃ b' ≥ 0, Real.cos (a' * ↑x + b') = Real.sin (36 * ↑x)) → a' ≥ 36 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l1116_111621


namespace NUMINAMATH_CALUDE_proper_subsets_of_m_n_l1116_111645

def S : Set (Set Char) := {{}, {'m'}, {'n'}}

theorem proper_subsets_of_m_n :
  {A : Set Char | A ⊂ {'m', 'n'}} = S := by sorry

end NUMINAMATH_CALUDE_proper_subsets_of_m_n_l1116_111645


namespace NUMINAMATH_CALUDE_original_recipe_butter_l1116_111641

/-- Represents a bread recipe with butter and flour quantities -/
structure BreadRecipe where
  butter : ℝ  -- Amount of butter in ounces
  flour : ℝ   -- Amount of flour in cups

/-- The original bread recipe -/
def original_recipe : BreadRecipe := { butter := 0, flour := 5 }

/-- The scaled up recipe -/
def scaled_recipe : BreadRecipe := { butter := 12, flour := 20 }

/-- The scale factor between the original and scaled recipe -/
def scale_factor : ℝ := 4

theorem original_recipe_butter :
  original_recipe.butter = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_original_recipe_butter_l1116_111641


namespace NUMINAMATH_CALUDE_current_velocity_is_two_l1116_111611

-- Define the rowing speed in still water
def still_water_speed : ℝ := 10

-- Define the total time for the round trip
def total_time : ℝ := 15

-- Define the distance to the place
def distance : ℝ := 72

-- Define the velocity of the current as a variable
def current_velocity : ℝ → ℝ := λ v => v

-- Define the equation for the total time of the round trip
def time_equation (v : ℝ) : Prop :=
  distance / (still_water_speed - v) + distance / (still_water_speed + v) = total_time

-- Theorem statement
theorem current_velocity_is_two :
  ∃ v : ℝ, time_equation v ∧ current_velocity v = 2 :=
sorry

end NUMINAMATH_CALUDE_current_velocity_is_two_l1116_111611


namespace NUMINAMATH_CALUDE_reading_order_l1116_111661

variable (a b c d : ℝ)

theorem reading_order (h1 : a + c = b + d) (h2 : a + b > c + d) (h3 : d > b + c) :
  a > d ∧ d > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_reading_order_l1116_111661


namespace NUMINAMATH_CALUDE_rose_orchid_difference_l1116_111605

/-- Given the initial and final counts of roses and orchids in a vase, 
    prove that there are 10 more roses than orchids in the final state. -/
theorem rose_orchid_difference :
  let initial_roses : ℕ := 5
  let initial_orchids : ℕ := 3
  let final_roses : ℕ := 12
  let final_orchids : ℕ := 2
  final_roses - final_orchids = 10 := by
  sorry

end NUMINAMATH_CALUDE_rose_orchid_difference_l1116_111605


namespace NUMINAMATH_CALUDE_infinite_powers_of_two_l1116_111689

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The sequence a_n -/
noncomputable def a (n : ℕ) : ℤ :=
  floor (n * Real.sqrt 2)

/-- Statement: There are infinitely many n such that a_n is a power of 2 -/
theorem infinite_powers_of_two : ∀ k : ℕ, ∃ n > k, ∃ m : ℕ, a n = 2^m :=
sorry

end NUMINAMATH_CALUDE_infinite_powers_of_two_l1116_111689


namespace NUMINAMATH_CALUDE_black_ball_probability_l1116_111692

theorem black_ball_probability 
  (p_red : ℝ) 
  (p_white : ℝ) 
  (h_red : p_red = 0.42) 
  (h_white : p_white = 0.28) :
  1 - p_red - p_white = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_black_ball_probability_l1116_111692


namespace NUMINAMATH_CALUDE_range_of_a_l1116_111664

def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + a = 0}

theorem range_of_a : ∀ a : ℝ, (A ∪ B a = A) ↔ (0 ≤ a ∧ a < 4) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1116_111664


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1116_111631

/-- A line that passes through (3, 4) and is tangent to the circle x^2 + y^2 = 25 has the equation 3x + 4y - 25 = 0 -/
theorem tangent_line_equation :
  ∃! (a b c : ℝ), 
    (∀ x y : ℝ, a * x + b * y + c = 0 → a * 3 + b * 4 + c = 0) ∧ 
    (∀ x y : ℝ, x^2 + y^2 = 25 → (a * x + b * y + c)^2 = (a^2 + b^2) * 25) ∧
    a = 3 ∧ b = 4 ∧ c = -25 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1116_111631


namespace NUMINAMATH_CALUDE_ratio_from_linear_equation_l1116_111609

theorem ratio_from_linear_equation (x y : ℝ) (h : 2 * y - 5 * x = 0) :
  ∃ (k : ℝ), k > 0 ∧ x = 2 * k ∧ y = 5 * k :=
sorry

end NUMINAMATH_CALUDE_ratio_from_linear_equation_l1116_111609


namespace NUMINAMATH_CALUDE_square_ratio_side_length_l1116_111663

theorem square_ratio_side_length (area_ratio : ℚ) :
  area_ratio = 270 / 125 →
  ∃ (a b c : ℕ), 
    (a = 3 ∧ b = 30 ∧ c = 25) ∧
    (Real.sqrt area_ratio = a * Real.sqrt b / c) ∧
    (a + b + c = 58) := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_side_length_l1116_111663


namespace NUMINAMATH_CALUDE_smallest_integer_y_smallest_integer_y_is_six_l1116_111657

theorem smallest_integer_y (y : ℤ) : (7 - 3 * y < -8) ↔ (y ≥ 6) := by sorry

theorem smallest_integer_y_is_six : ∃ (y : ℤ), (7 - 3 * y < -8) ∧ (∀ (z : ℤ), (7 - 3 * z < -8) → z ≥ y) ∧ y = 6 := by sorry

end NUMINAMATH_CALUDE_smallest_integer_y_smallest_integer_y_is_six_l1116_111657


namespace NUMINAMATH_CALUDE_total_ants_l1116_111686

theorem total_ants (red_ants : ℕ) (black_ants : ℕ) 
  (h1 : red_ants = 413) (h2 : black_ants = 487) : 
  red_ants + black_ants = 900 := by
  sorry

end NUMINAMATH_CALUDE_total_ants_l1116_111686


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1116_111644

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℚ
  -- Common difference of the sequence
  d : ℚ
  -- Sum of first 5 terms is 10
  sum_5 : (5 : ℚ) / 2 * (2 * a + 4 * d) = 10
  -- Sum of first 50 terms is 150
  sum_50 : (50 : ℚ) / 2 * (2 * a + 49 * d) = 150

/-- Properties of the 55th term and sum of first 55 terms -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  let sum_55 := (55 : ℚ) / 2 * (2 * seq.a + 54 * seq.d)
  let term_55 := seq.a + 54 * seq.d
  sum_55 = 171 ∧ term_55 = 4.31 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1116_111644


namespace NUMINAMATH_CALUDE_min_value_of_f_l1116_111634

/-- The quadratic function f(x) = x^2 - 8x + 15 -/
def f (x : ℝ) : ℝ := x^2 - 8*x + 15

/-- The theorem stating that the minimum value of f(x) is -1 -/
theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1116_111634


namespace NUMINAMATH_CALUDE_right_triangle_cosine_l1116_111619

/-- In a right triangle DEF where angle D is 90 degrees and sin E = 3/5, cos F = 3/5 -/
theorem right_triangle_cosine (D E F : ℝ) : 
  D = Real.pi / 2 → 
  Real.sin E = 3 / 5 → 
  Real.cos F = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cosine_l1116_111619


namespace NUMINAMATH_CALUDE_jury_deliberation_theorem_l1116_111601

/-- Calculates the equivalent full days spent in jury deliberation --/
def jury_deliberation_days (total_days : ℕ) (selection_days : ℕ) (trial_multiplier : ℕ) 
  (deliberation_hours_per_day : ℕ) (hours_per_day : ℕ) : ℕ :=
  let trial_days := selection_days * trial_multiplier
  let deliberation_days := total_days - selection_days - trial_days
  let total_deliberation_hours := deliberation_days * deliberation_hours_per_day
  total_deliberation_hours / hours_per_day

theorem jury_deliberation_theorem :
  jury_deliberation_days 19 2 4 16 24 = 6 := by
  sorry

end NUMINAMATH_CALUDE_jury_deliberation_theorem_l1116_111601


namespace NUMINAMATH_CALUDE_increasing_function_m_range_l1116_111674

def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * m * x^2 + 6 * x

theorem increasing_function_m_range :
  ∀ m : ℝ, (∀ x > 2, (∀ h > 0, f m (x + h) > f m x)) ↔ m < 5/2 :=
sorry

end NUMINAMATH_CALUDE_increasing_function_m_range_l1116_111674


namespace NUMINAMATH_CALUDE_resettlement_threshold_year_consecutive_equal_proportion_l1116_111613

/-- The area of new housing constructed in the first year (2015) in millions of square meters. -/
def initial_new_housing : ℝ := 5

/-- The area of resettlement housing in the first year (2015) in millions of square meters. -/
def initial_resettlement : ℝ := 2

/-- The annual growth rate of new housing area. -/
def new_housing_growth_rate : ℝ := 0.1

/-- The annual increase in resettlement housing area in millions of square meters. -/
def resettlement_increase : ℝ := 0.5

/-- The cumulative area of resettlement housing after n years. -/
def cumulative_resettlement (n : ℕ) : ℝ :=
  25 * n^2 + 175 * n

/-- The area of new housing in the nth year. -/
def new_housing (n : ℕ) : ℝ :=
  initial_new_housing * (1 + new_housing_growth_rate)^(n - 1)

/-- The area of resettlement housing in the nth year. -/
def resettlement (n : ℕ) : ℝ :=
  initial_resettlement + resettlement_increase * (n - 1)

theorem resettlement_threshold_year :
  ∃ n : ℕ, cumulative_resettlement n ≥ 30 ∧ ∀ m < n, cumulative_resettlement m < 30 :=
sorry

theorem consecutive_equal_proportion :
  ∃ n : ℕ, resettlement n / new_housing n = resettlement (n + 1) / new_housing (n + 1) :=
sorry

end NUMINAMATH_CALUDE_resettlement_threshold_year_consecutive_equal_proportion_l1116_111613


namespace NUMINAMATH_CALUDE_millet_majority_on_fourth_day_l1116_111629

/-- Represents the proportion of millet remaining after birds consume 40% --/
def milletRemainingRatio : ℝ := 0.6

/-- Represents the proportion of millet in the daily seed addition --/
def dailyMilletAddition : ℝ := 0.4

/-- Calculates the total proportion of millet in the feeder after n days --/
def milletProportion (n : ℕ) : ℝ :=
  1 - milletRemainingRatio ^ n

/-- Theorem stating that on the fourth day, the proportion of millet exceeds 50% for the first time --/
theorem millet_majority_on_fourth_day :
  (milletProportion 4 > 1/2) ∧ 
  (∀ k : ℕ, k < 4 → milletProportion k ≤ 1/2) := by
  sorry


end NUMINAMATH_CALUDE_millet_majority_on_fourth_day_l1116_111629


namespace NUMINAMATH_CALUDE_common_solution_y_value_l1116_111676

theorem common_solution_y_value : ∃ (x y : ℝ), 
  (x^2 + y^2 - 16 = 0) ∧ 
  (x^2 - 3*y + 12 = 0) → 
  y = 4 := by sorry

end NUMINAMATH_CALUDE_common_solution_y_value_l1116_111676


namespace NUMINAMATH_CALUDE_f_composition_value_l1116_111626

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp (-x) else Real.log x

theorem f_composition_value : f (f (1/3)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l1116_111626


namespace NUMINAMATH_CALUDE_statement_A_statement_D_l1116_111655

-- Statement A
theorem statement_A (a b c : ℝ) : 
  a / (c^2 + 1) > b / (c^2 + 1) → a > b :=
by sorry

-- Statement D
theorem statement_D (a b : ℝ) :
  -1 < 2*a + b ∧ 2*a + b < 1 ∧ -1 < a - b ∧ a - b < 2 →
  -3 < 4*a - b ∧ 4*a - b < 5 :=
by sorry

end NUMINAMATH_CALUDE_statement_A_statement_D_l1116_111655


namespace NUMINAMATH_CALUDE_sandwich_slices_count_l1116_111666

/-- Given the total number of sandwiches and the total number of bread slices,
    calculate the number of slices per sandwich. -/
def slices_per_sandwich (total_sandwiches : ℕ) (total_slices : ℕ) : ℚ :=
  total_slices / total_sandwiches

/-- Theorem stating that for 5 sandwiches and 15 slices, each sandwich consists of 3 slices. -/
theorem sandwich_slices_count :
  slices_per_sandwich 5 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_slices_count_l1116_111666


namespace NUMINAMATH_CALUDE_quiz_scores_l1116_111679

theorem quiz_scores (nicole kim cherry : ℕ) 
  (h1 : nicole = kim - 3)
  (h2 : kim = cherry + 8)
  (h3 : nicole = 22) : 
  cherry = 17 := by sorry

end NUMINAMATH_CALUDE_quiz_scores_l1116_111679


namespace NUMINAMATH_CALUDE_range_of_a_l1116_111608

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 2*a - 4}
def B : Set ℝ := {x | x^2 - 5*x - 6 < 0}

-- State the theorem
theorem range_of_a (a : ℝ) : A a ∩ B = A a → a < 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1116_111608


namespace NUMINAMATH_CALUDE_more_polygons_without_A1_l1116_111687

-- Define the number of points on the circle
def n : ℕ := 16

-- Define the function to calculate the number of polygons including A1
def polygons_with_A1 (n : ℕ) : ℕ :=
  (2^(n-1) : ℕ) - (n : ℕ)

-- Define the function to calculate the number of polygons not including A1
def polygons_without_A1 (n : ℕ) : ℕ :=
  (2^(n-1) : ℕ) - (n : ℕ) - ((n-1).choose 2)

-- State the theorem
theorem more_polygons_without_A1 :
  polygons_without_A1 n > polygons_with_A1 n :=
by sorry

end NUMINAMATH_CALUDE_more_polygons_without_A1_l1116_111687


namespace NUMINAMATH_CALUDE_monomial_replacement_four_terms_l1116_111675

/-- Given an expression (x^4 - 3)^2 + (x^3 + *)^2, where * is to be replaced by a monomial,
    prove that replacing * with (x^3 + 3x) results in an expression with exactly four terms
    after squaring and combining like terms. -/
theorem monomial_replacement_four_terms (x : ℝ) : 
  let original_expr := (x^4 - 3)^2 + (x^3 + (x^3 + 3*x))^2
  ∃ (a b c d : ℝ) (n₁ n₂ n₃ n₄ : ℕ), 
    original_expr = a * x^n₁ + b * x^n₂ + c * x^n₃ + d * x^n₄ ∧
    n₁ > n₂ ∧ n₂ > n₃ ∧ n₃ > n₄ ∧
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_monomial_replacement_four_terms_l1116_111675


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l1116_111688

theorem solution_set_implies_a_value (a : ℝ) :
  ({x : ℝ | |x - a| < 1} = {x : ℝ | 2 < x ∧ x < 4}) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l1116_111688


namespace NUMINAMATH_CALUDE_least_three_digit_with_digit_product_8_l1116_111627

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * tens * ones

theorem least_three_digit_with_digit_product_8 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 8 → 118 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_with_digit_product_8_l1116_111627


namespace NUMINAMATH_CALUDE_equation_equality_l1116_111610

theorem equation_equality : 5 + (-6) - (-7) = 5 - 6 + 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l1116_111610


namespace NUMINAMATH_CALUDE_tesseract_simplex_ratio_l1116_111635

-- Define the vertices of the 4-simplex
def v₀ : Fin 4 → ℝ := λ _ => 0
def v₁ : Fin 4 → ℝ := λ i => if i.val < 2 then 1 else 0
def v₂ : Fin 4 → ℝ := λ i => if i.val = 0 ∨ i.val = 2 then 1 else 0
def v₃ : Fin 4 → ℝ := λ i => if i.val = 0 ∨ i.val = 3 then 1 else 0
def v₄ : Fin 4 → ℝ := λ i => if i.val > 0 then 1 else 0

-- Define the 4-simplex
def simplex : Fin 5 → (Fin 4 → ℝ) := λ i =>
  match i with
  | 0 => v₀
  | 1 => v₁
  | 2 => v₂
  | 3 => v₃
  | 4 => v₄

-- Define the hypervolume of a unit tesseract
def tesseract_hypervolume : ℝ := 1

-- Define the function to calculate the hypervolume of the 4-simplex
noncomputable def simplex_hypervolume : ℝ := sorry

-- State the theorem
theorem tesseract_simplex_ratio :
  tesseract_hypervolume / simplex_hypervolume = 24 / Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_tesseract_simplex_ratio_l1116_111635


namespace NUMINAMATH_CALUDE_terminating_decimal_count_l1116_111625

/-- A fraction a/b has a terminating decimal representation if and only if
    the denominator b can be factored as 2^m * 5^n * d, where d is coprime to 10 -/
def has_terminating_decimal (a b : ℕ) : Prop := sorry

/-- Count of integers in a given range satisfying a property -/
def count_satisfying (lower upper : ℕ) (P : ℕ → Prop) : ℕ := sorry

theorem terminating_decimal_count :
  count_satisfying 1 508 (λ k => has_terminating_decimal k 425) = 29 := by sorry

end NUMINAMATH_CALUDE_terminating_decimal_count_l1116_111625


namespace NUMINAMATH_CALUDE_complex_sum_of_parts_l1116_111677

theorem complex_sum_of_parts (z : ℂ) (h : z * Complex.I = -1 + Complex.I) : 
  z.re + z.im = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_parts_l1116_111677


namespace NUMINAMATH_CALUDE_cody_discount_l1116_111615

/-- The discount Cody got after taxes --/
def discount_after_taxes (initial_price tax_rate discount final_price : ℝ) : ℝ :=
  initial_price * (1 + tax_rate) - final_price

/-- Theorem stating the discount Cody got after taxes --/
theorem cody_discount :
  ∃ (discount : ℝ),
    discount_after_taxes 40 0.05 discount (2 * 17) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_cody_discount_l1116_111615


namespace NUMINAMATH_CALUDE_arthur_muffins_l1116_111680

theorem arthur_muffins (james_muffins : ℕ) (arthur_muffins : ℕ) 
  (h1 : james_muffins = 12 * arthur_muffins) 
  (h2 : james_muffins = 1380) : 
  arthur_muffins = 115 := by
sorry

end NUMINAMATH_CALUDE_arthur_muffins_l1116_111680


namespace NUMINAMATH_CALUDE_special_collection_books_l1116_111667

/-- The number of books in a special collection at the beginning of a month,
    given the number of books loaned, returned, and remaining at the end. -/
theorem special_collection_books
  (loaned : ℕ)
  (return_rate : ℚ)
  (end_count : ℕ)
  (h1 : loaned = 40)
  (h2 : return_rate = 7/10)
  (h3 : end_count = 63) :
  loaned * (1 - return_rate) + end_count = 47 :=
sorry

end NUMINAMATH_CALUDE_special_collection_books_l1116_111667


namespace NUMINAMATH_CALUDE_reciprocal_nonexistence_l1116_111614

theorem reciprocal_nonexistence (a : ℝ) : (¬∃x : ℝ, x * a = 1) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_nonexistence_l1116_111614


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l1116_111643

theorem right_triangle_side_length 
  (X Y Z : ℝ) 
  (h_right_angle : X^2 + Y^2 = Z^2)  -- Y is the right angle
  (h_cos : Real.cos X = 3/5)
  (h_hypotenuse : Z = 10) :
  Y = 8 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l1116_111643


namespace NUMINAMATH_CALUDE_forty_fifth_even_positive_integer_l1116_111636

theorem forty_fifth_even_positive_integer :
  (fun n : ℕ => 2 * n) 45 = 90 := by sorry

end NUMINAMATH_CALUDE_forty_fifth_even_positive_integer_l1116_111636


namespace NUMINAMATH_CALUDE_sum_3x_4y_equals_60_l1116_111653

theorem sum_3x_4y_equals_60 
  (x y N : ℝ) 
  (h1 : 3 * x + 4 * y = N) 
  (h2 : 6 * x - 4 * y = 12) 
  (h3 : x * y = 72) : 
  3 * x + 4 * y = 60 := by
sorry

end NUMINAMATH_CALUDE_sum_3x_4y_equals_60_l1116_111653


namespace NUMINAMATH_CALUDE_mean_score_problem_l1116_111665

theorem mean_score_problem (m_mean a_mean : ℝ) (m a : ℕ) 
  (h1 : m_mean = 75)
  (h2 : a_mean = 65)
  (h3 : m = 2 * a / 3) :
  (m_mean * m + a_mean * a) / (m + a) = 69 := by
sorry

end NUMINAMATH_CALUDE_mean_score_problem_l1116_111665


namespace NUMINAMATH_CALUDE_f_domain_l1116_111659

noncomputable def f (x : ℝ) : ℝ := Real.tan (Real.arcsin (x^2))

theorem f_domain : Set.Icc (-1 : ℝ) 1 = {x : ℝ | ∃ y, f x = y} :=
sorry

end NUMINAMATH_CALUDE_f_domain_l1116_111659


namespace NUMINAMATH_CALUDE_trig_identity_l1116_111682

theorem trig_identity (α β : Real) 
  (h : (Real.cos α)^6 / (Real.cos β)^3 + (Real.sin α)^6 / (Real.sin β)^3 = 1) :
  (Real.sin β)^6 / (Real.sin α)^3 + (Real.cos β)^6 / (Real.cos α)^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1116_111682


namespace NUMINAMATH_CALUDE_only_prop3_true_l1116_111699

-- Define a sequence as a function from ℕ to ℝ
def Sequence := ℕ → ℝ

-- Define the limit of a sequence
def LimitOf (a : Sequence) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - L| < ε

-- Define the four propositions
def Prop1 (a : Sequence) (A : ℝ) : Prop :=
  LimitOf (fun n => (a n)^2) (A^2) → LimitOf a A

def Prop2 (a : Sequence) (A : ℝ) : Prop :=
  (∀ n, a n > 0) → LimitOf a A → A > 0

def Prop3 (a : Sequence) (A : ℝ) : Prop :=
  LimitOf a A → LimitOf (fun n => (a n)^2) (A^2)

def Prop4 (a b : Sequence) : Prop :=
  LimitOf (fun n => a n - b n) 0 → 
  (∃ L, LimitOf a L ∧ LimitOf b L)

-- Theorem stating that only Prop3 is always true
theorem only_prop3_true : 
  (∃ a A, ¬ Prop1 a A) ∧
  (∃ a A, ¬ Prop2 a A) ∧
  (∀ a A, Prop3 a A) ∧
  (∃ a b, ¬ Prop4 a b) := by
  sorry

end NUMINAMATH_CALUDE_only_prop3_true_l1116_111699


namespace NUMINAMATH_CALUDE_smallest_natural_with_congruences_l1116_111656

theorem smallest_natural_with_congruences (m : ℕ) : 
  (∀ k : ℕ, k < m → (k % 3 ≠ 1 ∨ k % 7 ≠ 5 ∨ k % 11 ≠ 4)) → 
  m % 3 = 1 → 
  m % 7 = 5 → 
  m % 11 = 4 → 
  m % 4 = 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_natural_with_congruences_l1116_111656


namespace NUMINAMATH_CALUDE_g_13_l1116_111673

def g (n : ℕ) : ℕ := n^2 + 2*n + 41

theorem g_13 : g 13 = 236 := by
  sorry

end NUMINAMATH_CALUDE_g_13_l1116_111673


namespace NUMINAMATH_CALUDE_least_b_is_five_l1116_111678

/-- A triangle with angles a, b, c in degrees, where a, b, c are prime numbers and a > b > c -/
structure PrimeAngleTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  a_prime : Nat.Prime a
  b_prime : Nat.Prime b
  c_prime : Nat.Prime c
  angle_sum : a + b + c = 180
  a_gt_b : a > b
  b_gt_c : b > c
  not_right : a ≠ 90 ∧ b ≠ 90 ∧ c ≠ 90

/-- The least possible value of b in a PrimeAngleTriangle is 5 -/
theorem least_b_is_five (t : PrimeAngleTriangle) : t.b ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_least_b_is_five_l1116_111678


namespace NUMINAMATH_CALUDE_max_students_equal_distribution_l1116_111694

theorem max_students_equal_distribution (pens pencils : ℕ) 
  (h_pens : pens = 1001) (h_pencils : pencils = 910) : 
  (∃ (students : ℕ), 
    students > 0 ∧ 
    pens % students = 0 ∧ 
    pencils % students = 0 ∧ 
    ∀ (n : ℕ), n > students → (pens % n ≠ 0 ∨ pencils % n ≠ 0)) ↔ 
  (∃ (max_students : ℕ), max_students = Nat.gcd pens pencils) :=
sorry

end NUMINAMATH_CALUDE_max_students_equal_distribution_l1116_111694


namespace NUMINAMATH_CALUDE_school_demographics_l1116_111628

theorem school_demographics (total_students : ℕ) (boys_avg_age girls_avg_age school_avg_age : ℚ) : 
  total_students = 632 →
  boys_avg_age = 12 →
  girls_avg_age = 11 →
  school_avg_age = 47/4 →
  ∃ (num_girls : ℕ), num_girls = 156 ∧ num_girls ≤ total_students := by
  sorry

end NUMINAMATH_CALUDE_school_demographics_l1116_111628


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l1116_111660

/-- A line passing through point (2,3) and parallel to 2x+4y-3=0 has equation x + 2y - 8 = 0 -/
theorem line_through_point_parallel_to_line :
  let line1 : ℝ → ℝ → Prop := λ x y => x + 2*y - 8 = 0
  let line2 : ℝ → ℝ → Prop := λ x y => 2*x + 4*y - 3 = 0
  (line1 2 3) ∧ 
  (∀ (x y : ℝ), line1 x y ↔ ∃ (k : ℝ), y = (-1/2)*x + k) ∧
  (∀ (x y : ℝ), line2 x y ↔ ∃ (k : ℝ), y = (-1/2)*x + k) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l1116_111660


namespace NUMINAMATH_CALUDE_thalassa_population_estimate_l1116_111697

-- Define the initial population in 2020
def initial_population : ℕ := 500

-- Define the doubling period in years
def doubling_period : ℕ := 30

-- Define the target year
def target_year : ℕ := 2075

-- Define the base year
def base_year : ℕ := 2020

-- Function to calculate the number of complete doubling periods
def complete_doubling_periods (start_year end_year doubling_period : ℕ) : ℕ :=
  (end_year - start_year) / doubling_period

-- Function to estimate population after a number of complete doubling periods
def population_after_doubling (initial_pop doubling_periods : ℕ) : ℕ :=
  initial_pop * (2 ^ doubling_periods)

-- Theorem statement
theorem thalassa_population_estimate :
  let complete_periods := complete_doubling_periods base_year target_year doubling_period
  let pop_at_last_complete_period := population_after_doubling initial_population complete_periods
  let pop_at_next_complete_period := pop_at_last_complete_period * 2
  (pop_at_last_complete_period + pop_at_next_complete_period) / 2 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_thalassa_population_estimate_l1116_111697


namespace NUMINAMATH_CALUDE_total_food_consumed_theorem_l1116_111639

/-- Represents the amount of food a dog eats per meal -/
structure MealPortion where
  dry : Float
  wet : Float

/-- Represents the feeding schedule for a dog -/
structure FeedingSchedule where
  portion : MealPortion
  mealsPerDay : Nat

/-- Conversion rates for dry and wet food -/
def dryFoodConversion : Float := 3.2  -- cups per pound
def wetFoodConversion : Float := 2.8  -- cups per pound

/-- Feeding schedules for each dog -/
def momoSchedule : FeedingSchedule := { portion := { dry := 1.3, wet := 0.7 }, mealsPerDay := 2 }
def fifiSchedule : FeedingSchedule := { portion := { dry := 1.6, wet := 0.5 }, mealsPerDay := 2 }
def gigiSchedule : FeedingSchedule := { portion := { dry := 2.0, wet := 1.0 }, mealsPerDay := 3 }

/-- Calculate total food consumed by all dogs in pounds -/
def totalFoodConsumed (momo fifi gigi : FeedingSchedule) : Float :=
  let totalDry := (momo.portion.dry * momo.mealsPerDay.toFloat +
                   fifi.portion.dry * fifi.mealsPerDay.toFloat +
                   gigi.portion.dry * gigi.mealsPerDay.toFloat) / dryFoodConversion
  let totalWet := (momo.portion.wet * momo.mealsPerDay.toFloat +
                   fifi.portion.wet * fifi.mealsPerDay.toFloat +
                   gigi.portion.wet * gigi.mealsPerDay.toFloat) / wetFoodConversion
  totalDry + totalWet

/-- Theorem: The total amount of food consumed by all three dogs in a day is approximately 5.6161 pounds -/
theorem total_food_consumed_theorem :
  Float.abs (totalFoodConsumed momoSchedule fifiSchedule gigiSchedule - 5.6161) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_total_food_consumed_theorem_l1116_111639


namespace NUMINAMATH_CALUDE_horner_method_proof_l1116_111616

def horner_polynomial (x : ℝ) : ℝ := x * (x * (x * (x * (2 * x + 0) + 4) + 3) + 1)

theorem horner_method_proof :
  let f (x : ℝ) := 3 * x^2 + 2 * x^5 + 4 * x^3 + x
  f 3 = horner_polynomial 3 ∧ horner_polynomial 3 = 624 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_proof_l1116_111616


namespace NUMINAMATH_CALUDE_rabbit_speed_l1116_111623

def rabbit_speed_equation (x : ℝ) : Prop :=
  2 * (2 * x + 4) = 188

theorem rabbit_speed : ∃ x : ℝ, rabbit_speed_equation x ∧ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_speed_l1116_111623


namespace NUMINAMATH_CALUDE_simplified_win_ratio_l1116_111603

def chloe_wins : ℕ := 24
def max_wins : ℕ := 9

theorem simplified_win_ratio : 
  ∃ (a b : ℕ), a = 8 ∧ b = 3 ∧ chloe_wins * b = max_wins * a := by
  sorry

end NUMINAMATH_CALUDE_simplified_win_ratio_l1116_111603


namespace NUMINAMATH_CALUDE_last_person_coins_l1116_111630

/-- Represents the amount of coins each person receives in an arithmetic sequence. -/
structure CoinDistribution where
  a : ℚ
  d : ℚ

/-- Calculates the total number of coins distributed. -/
def totalCoins (dist : CoinDistribution) : ℚ :=
  5 * dist.a

/-- Checks if the sum of the first two equals the sum of the last three. -/
def sumCondition (dist : CoinDistribution) : Prop :=
  (dist.a - 2*dist.d) + (dist.a - dist.d) = dist.a + (dist.a + dist.d) + (dist.a + 2*dist.d)

/-- The main theorem stating the amount the last person receives. -/
theorem last_person_coins (dist : CoinDistribution) 
  (h1 : totalCoins dist = 5)
  (h2 : sumCondition dist) :
  dist.a + 2*dist.d = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_last_person_coins_l1116_111630


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l1116_111606

-- Part 1
theorem factorization_1 (x y : ℝ) : 
  (x - y)^2 - 4*(x - y) + 4 = (x - y - 2)^2 := by sorry

-- Part 2
theorem factorization_2 (a b : ℝ) : 
  (a^2 + b^2)^2 - 4*a^2*b^2 = (a - b)^2 * (a + b)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l1116_111606


namespace NUMINAMATH_CALUDE_seventh_root_unity_sum_l1116_111698

theorem seventh_root_unity_sum (q : ℂ) (h : q^7 = 1) :
  q / (1 + q^2) + q^2 / (1 + q^4) + q^3 / (1 + q^6) = 
    if q = 1 then (3 : ℂ) / 2 else -2 := by sorry

end NUMINAMATH_CALUDE_seventh_root_unity_sum_l1116_111698


namespace NUMINAMATH_CALUDE_min_value_theorem_l1116_111617

def vector_a (x : ℝ) : ℝ × ℝ := (x - 1, 2)
def vector_b (y : ℝ) : ℝ × ℝ := (4, y)

def perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

theorem min_value_theorem (x y : ℝ) :
  perpendicular (vector_a x) (vector_b y) →
  ∃ (min : ℝ), min = 6 ∧ ∀ (z : ℝ), 9^x + 3^y ≥ z := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1116_111617


namespace NUMINAMATH_CALUDE_QR_length_l1116_111683

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define the point N on QR
def N_on_QR (Q R N : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ N = (1 - t) • Q + t • R

-- Define the ratio condition for N on QR
def N_divides_QR_in_ratio (Q R N : ℝ × ℝ) : Prop :=
  ∃ x : ℝ, dist Q N = 2 * x ∧ dist N R = 3 * x

-- Main theorem
theorem QR_length 
  (P Q R N : ℝ × ℝ) 
  (triangle : Triangle P Q R)
  (pr_length : dist P R = 5)
  (pq_length : dist P Q = 7)
  (n_on_qr : N_on_QR Q R N)
  (n_divides_qr : N_divides_QR_in_ratio Q R N)
  (pn_length : dist P N = 4) :
  dist Q R = 5 * Real.sqrt 3.9 := by
  sorry


end NUMINAMATH_CALUDE_QR_length_l1116_111683


namespace NUMINAMATH_CALUDE_kiarra_age_l1116_111670

/-- Given the ages of several people and their relationships, prove Kiarra's age --/
theorem kiarra_age (bea job figaro harry kiarra : ℕ) 
  (h1 : kiarra = 2 * bea)
  (h2 : job = 3 * bea)
  (h3 : figaro = job + 7)
  (h4 : harry * 2 = figaro)
  (h5 : harry = 26) :
  kiarra = 30 := by
  sorry

end NUMINAMATH_CALUDE_kiarra_age_l1116_111670
