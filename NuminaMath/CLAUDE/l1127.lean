import Mathlib

namespace NUMINAMATH_CALUDE_bridge_length_l1127_112780

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 140 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 235 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l1127_112780


namespace NUMINAMATH_CALUDE_alice_most_dogs_l1127_112781

-- Define the number of cats and dogs for each person
variable (Kc Ac Bc Kd Ad Bd : ℕ)

-- Define the conditions
axiom kathy_more_cats : Kc > Ac
axiom kathy_more_dogs : Kd > Bd
axiom alice_more_dogs : Ad > Kd
axiom bruce_more_cats : Bc > Ac

-- Theorem to prove
theorem alice_most_dogs : Ad > Kd ∧ Ad > Bd :=
sorry

end NUMINAMATH_CALUDE_alice_most_dogs_l1127_112781


namespace NUMINAMATH_CALUDE_incorrect_number_calculation_l1127_112787

theorem incorrect_number_calculation (n : ℕ) (initial_avg correct_avg correct_num : ℝ) :
  n = 10 ∧ initial_avg = 19 ∧ correct_avg = 24 ∧ correct_num = 76 →
  ∃ (incorrect_num : ℝ),
    n * initial_avg + (correct_num - incorrect_num) = n * correct_avg ∧
    incorrect_num = 26 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_number_calculation_l1127_112787


namespace NUMINAMATH_CALUDE_M_mod_1000_l1127_112757

/-- The number of 8-digit positive integers with strictly increasing digits -/
def M : ℕ := Nat.choose 9 8

/-- Theorem stating that M modulo 1000 equals 9 -/
theorem M_mod_1000 : M % 1000 = 9 := by
  sorry

end NUMINAMATH_CALUDE_M_mod_1000_l1127_112757


namespace NUMINAMATH_CALUDE_equation_solution_l1127_112755

theorem equation_solution : ∃ x : ℝ, 4 * x - 7 = 5 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1127_112755


namespace NUMINAMATH_CALUDE_decimal_524_to_octal_l1127_112765

-- Define a function to convert decimal to octal
def decimalToOctal (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec helper (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else helper (m / 8) ((m % 8) :: acc)
    helper n []

-- Theorem statement
theorem decimal_524_to_octal :
  decimalToOctal 524 = [1, 0, 1, 4] := by sorry

end NUMINAMATH_CALUDE_decimal_524_to_octal_l1127_112765


namespace NUMINAMATH_CALUDE_milk_cost_verify_milk_cost_l1127_112772

/-- Proves that the cost of a gallon of milk is $4 given the conditions about coffee consumption and costs --/
theorem milk_cost (cups_per_day : ℕ) (oz_per_cup : ℚ) (bag_cost : ℚ) (oz_per_bag : ℚ) 
  (milk_usage : ℚ) (total_cost : ℚ) : ℚ :=
by
  -- Define the conditions
  have h1 : cups_per_day = 2 := by sorry
  have h2 : oz_per_cup = 3/2 := by sorry
  have h3 : bag_cost = 8 := by sorry
  have h4 : oz_per_bag = 21/2 := by sorry
  have h5 : milk_usage = 1/2 := by sorry
  have h6 : total_cost = 18 := by sorry

  -- Calculate the cost of a gallon of milk
  sorry

/-- The cost of a gallon of milk --/
def gallon_milk_cost : ℚ := 4

/-- Proves that the calculated cost matches the expected cost --/
theorem verify_milk_cost : 
  milk_cost 2 (3/2) 8 (21/2) (1/2) 18 = gallon_milk_cost := by sorry

end NUMINAMATH_CALUDE_milk_cost_verify_milk_cost_l1127_112772


namespace NUMINAMATH_CALUDE_arithmetic_progression_x_value_l1127_112746

/-- An arithmetic progression with first three terms 2x - 3, 3x - 1, and 5x + 1 has x = 0 --/
theorem arithmetic_progression_x_value (x : ℝ) : 
  let a₁ : ℝ := 2*x - 3
  let a₂ : ℝ := 3*x - 1
  let a₃ : ℝ := 5*x + 1
  (a₂ - a₁ = a₃ - a₂) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_x_value_l1127_112746


namespace NUMINAMATH_CALUDE_factorial_ratio_l1127_112778

theorem factorial_ratio : (12 : ℕ).factorial / (11 : ℕ).factorial = 12 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l1127_112778


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l1127_112705

theorem binomial_coefficient_equality (m : ℕ) : 
  (Nat.choose 15 m = Nat.choose 15 (m - 3)) ↔ m = 9 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l1127_112705


namespace NUMINAMATH_CALUDE_arrive_at_beths_house_time_l1127_112790

/-- The time it takes for Tom and Beth to meet and return to Beth's house -/
def meeting_and_return_time (tom_speed beth_speed : ℚ) : ℚ :=
  let meeting_time := 1 / (tom_speed + beth_speed)
  let return_time := (1 / 2) / beth_speed
  meeting_time + return_time

/-- Theorem stating that Tom and Beth will arrive at Beth's house 78 minutes after noon -/
theorem arrive_at_beths_house_time :
  let tom_speed : ℚ := 1 / 63
  let beth_speed : ℚ := 1 / 84
  meeting_and_return_time tom_speed beth_speed = 78 / 1 := by
  sorry

#eval meeting_and_return_time (1 / 63) (1 / 84)

end NUMINAMATH_CALUDE_arrive_at_beths_house_time_l1127_112790


namespace NUMINAMATH_CALUDE_exclusive_movies_count_l1127_112786

/-- Given two movie collections belonging to Andrew and John, this theorem proves
    the number of movies that are in either collection but not both. -/
theorem exclusive_movies_count
  (total_andrew : ℕ)
  (shared : ℕ)
  (john_exclusive : ℕ)
  (h1 : total_andrew = 25)
  (h2 : shared = 15)
  (h3 : john_exclusive = 8) :
  total_andrew - shared + john_exclusive = 18 :=
by sorry

end NUMINAMATH_CALUDE_exclusive_movies_count_l1127_112786


namespace NUMINAMATH_CALUDE_sqrt_expressions_l1127_112753

theorem sqrt_expressions :
  (∀ x y z : ℝ, x = 27 ∧ y = 1/3 ∧ z = 3 → 
    Real.sqrt x - Real.sqrt y + Real.sqrt z = (11 * Real.sqrt 3) / 3) ∧
  (∀ a b c : ℝ, a = 32 ∧ b = 18 ∧ c = 2 → 
    (Real.sqrt a + Real.sqrt b) / Real.sqrt c - 8 = -1) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_expressions_l1127_112753


namespace NUMINAMATH_CALUDE_octal_123_equals_decimal_83_l1127_112702

/-- Converts an octal number to decimal --/
def octal_to_decimal (d₂ d₁ d₀ : ℕ) : ℕ :=
  d₂ * 8^2 + d₁ * 8^1 + d₀ * 8^0

/-- Proves that the octal number 123₈ is equal to the decimal number 83 --/
theorem octal_123_equals_decimal_83 : octal_to_decimal 1 2 3 = 83 := by
  sorry

end NUMINAMATH_CALUDE_octal_123_equals_decimal_83_l1127_112702


namespace NUMINAMATH_CALUDE_opposite_of_reciprocal_l1127_112710

theorem opposite_of_reciprocal : -(1 / (-1/3)) = 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_reciprocal_l1127_112710


namespace NUMINAMATH_CALUDE_derivative_from_second_derivative_l1127_112736

open Real

theorem derivative_from_second_derivative
  (f : ℝ → ℝ)
  (h : ∀ x, deriv^[2] f x = 3) :
  ∀ x, deriv f x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_derivative_from_second_derivative_l1127_112736


namespace NUMINAMATH_CALUDE_parent_gift_cost_is_30_l1127_112751

/-- The amount spent on each sibling's gift -/
def sibling_gift_cost : ℕ := 30

/-- The number of siblings -/
def num_siblings : ℕ := 3

/-- The total amount spent on all gifts -/
def total_spent : ℕ := 150

/-- The amount spent on each parent's gift -/
def parent_gift_cost : ℕ := (total_spent - sibling_gift_cost * num_siblings) / 2

/-- Theorem stating that the amount spent on each parent's gift is $30 -/
theorem parent_gift_cost_is_30 : parent_gift_cost = 30 := by
  sorry

end NUMINAMATH_CALUDE_parent_gift_cost_is_30_l1127_112751


namespace NUMINAMATH_CALUDE_min_packs_for_126_cans_l1127_112767

/-- Represents the number of cans in each pack size --/
inductive PackSize
| small : PackSize
| medium : PackSize
| large : PackSize

/-- Returns the number of cans for a given pack size --/
def cansInPack (p : PackSize) : ℕ :=
  match p with
  | PackSize.small => 15
  | PackSize.medium => 18
  | PackSize.large => 36

/-- Represents a combination of packs --/
structure PackCombination where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total number of cans in a pack combination --/
def totalCans (c : PackCombination) : ℕ :=
  c.small * cansInPack PackSize.small +
  c.medium * cansInPack PackSize.medium +
  c.large * cansInPack PackSize.large

/-- Calculates the total number of packs in a combination --/
def totalPacks (c : PackCombination) : ℕ :=
  c.small + c.medium + c.large

/-- Defines what it means for a pack combination to be valid --/
def isValidCombination (c : PackCombination) : Prop :=
  totalCans c = 126

/-- Theorem: The minimum number of packs needed to buy exactly 126 cans is 4 --/
theorem min_packs_for_126_cans :
  ∃ (c : PackCombination), isValidCombination c ∧
    totalPacks c = 4 ∧
    (∀ (c' : PackCombination), isValidCombination c' → totalPacks c ≤ totalPacks c') :=
sorry

end NUMINAMATH_CALUDE_min_packs_for_126_cans_l1127_112767


namespace NUMINAMATH_CALUDE_second_question_correct_percentage_l1127_112726

-- Define the percentages as real numbers between 0 and 100
def first_correct : ℝ := 80
def neither_correct : ℝ := 5
def both_correct : ℝ := 60

-- Define the function to calculate the percentage who answered the second question correctly
def second_correct : ℝ := 100 - neither_correct - first_correct + both_correct

-- Theorem statement
theorem second_question_correct_percentage :
  second_correct = 75 :=
sorry

end NUMINAMATH_CALUDE_second_question_correct_percentage_l1127_112726


namespace NUMINAMATH_CALUDE_max_missed_problems_l1127_112713

theorem max_missed_problems (total_problems : ℕ) (passing_percentage : ℚ) : 
  total_problems = 50 → 
  passing_percentage = 75/100 → 
  ∃ (max_missed : ℕ), max_missed = 12 ∧ 
    (∀ (missed : ℕ), missed ≤ max_missed → 
      (total_problems - missed) / total_problems ≥ passing_percentage) ∧
    (∀ (missed : ℕ), missed > max_missed → 
      (total_problems - missed) / total_problems < passing_percentage) :=
by sorry

end NUMINAMATH_CALUDE_max_missed_problems_l1127_112713


namespace NUMINAMATH_CALUDE_books_purchased_with_grant_l1127_112717

/-- The number of books purchased by Silvergrove Public Library using a grant --/
theorem books_purchased_with_grant 
  (total_books : Nat) 
  (books_before_grant : Nat) 
  (h1 : total_books = 8582)
  (h2 : books_before_grant = 5935) :
  total_books - books_before_grant = 2647 := by
  sorry

end NUMINAMATH_CALUDE_books_purchased_with_grant_l1127_112717


namespace NUMINAMATH_CALUDE_smallest_solution_abs_quadratic_l1127_112734

theorem smallest_solution_abs_quadratic (x : ℝ) :
  (|2 * x^2 + 3 * x - 1| = 33) →
  x ≥ ((-3 - Real.sqrt 281) / 4) ∧
  (|2 * (((-3 - Real.sqrt 281) / 4)^2) + 3 * ((-3 - Real.sqrt 281) / 4) - 1| = 33) :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_quadratic_l1127_112734


namespace NUMINAMATH_CALUDE_undefined_expression_l1127_112750

theorem undefined_expression (x : ℝ) : 
  (x - 1) / (x^2 - 5*x + 6) = 0⁻¹ ↔ x = 2 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_undefined_expression_l1127_112750


namespace NUMINAMATH_CALUDE_sum_of_areas_decomposition_l1127_112706

/-- Represents a 1 by 1 by 1 cube -/
structure UnitCube where
  side : ℝ
  is_unit : side = 1

/-- Represents a triangle with vertices on the cube -/
structure CubeTriangle where
  vertices : Fin 3 → Fin 8

/-- The area of a triangle on the cube -/
noncomputable def triangle_area (t : CubeTriangle) : ℝ := sorry

/-- The sum of areas of all triangles on the cube -/
noncomputable def sum_of_triangle_areas (cube : UnitCube) : ℝ := sorry

/-- The theorem to be proved -/
theorem sum_of_areas_decomposition (cube : UnitCube) :
  ∃ (m n p : ℕ), sum_of_triangle_areas cube = m + Real.sqrt n + Real.sqrt p ∧ m + n + p = 348 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_areas_decomposition_l1127_112706


namespace NUMINAMATH_CALUDE_smallest_lcm_for_four_digit_gcd_five_l1127_112797

theorem smallest_lcm_for_four_digit_gcd_five (m n : ℕ) : 
  m ≥ 1000 ∧ m ≤ 9999 ∧ n ≥ 1000 ∧ n ≤ 9999 ∧ Nat.gcd m n = 5 →
  Nat.lcm m n ≥ 203010 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_for_four_digit_gcd_five_l1127_112797


namespace NUMINAMATH_CALUDE_solution_x_fourth_plus_81_l1127_112745

theorem solution_x_fourth_plus_81 :
  let solutions : List ℂ := [
    Complex.mk ((3 * Real.sqrt 2) / 2) ((3 * Real.sqrt 2) / 2),
    Complex.mk (-(3 * Real.sqrt 2) / 2) (-(3 * Real.sqrt 2) / 2),
    Complex.mk (-(3 * Real.sqrt 2) / 2) ((3 * Real.sqrt 2) / 2),
    Complex.mk ((3 * Real.sqrt 2) / 2) (-(3 * Real.sqrt 2) / 2)
  ]
  ∀ z : ℂ, z^4 + 81 = 0 ↔ z ∈ solutions := by
sorry

end NUMINAMATH_CALUDE_solution_x_fourth_plus_81_l1127_112745


namespace NUMINAMATH_CALUDE_geometric_series_proof_l1127_112712

def geometric_series (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_proof 
  (a : ℚ) 
  (h_a : a = 4/7) 
  (r : ℚ) 
  (h_r : r = 4/7) :
  r = 4/7 ∧ geometric_series a r 3 = 372/343 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_proof_l1127_112712


namespace NUMINAMATH_CALUDE_unique_solutions_l1127_112737

def is_solution (m n p : ℕ) : Prop :=
  p.Prime ∧ m > 0 ∧ n > 0 ∧ p^n + 3600 = m^2

theorem unique_solutions :
  ∀ m n p : ℕ,
    is_solution m n p ↔
      (m = 61 ∧ n = 2 ∧ p = 11) ∨
      (m = 65 ∧ n = 4 ∧ p = 5) ∨
      (m = 68 ∧ n = 10 ∧ p = 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solutions_l1127_112737


namespace NUMINAMATH_CALUDE_sign_sum_theorem_l1127_112708

theorem sign_sum_theorem (x y z w : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) :
  let sign_sum := x / |x| + y / |y| + z / |z| + w / |w| + (x * y * z * w) / |x * y * z * w|
  sign_sum = 5 ∨ sign_sum = 1 ∨ sign_sum = -1 ∨ sign_sum = -5 := by
  sorry

end NUMINAMATH_CALUDE_sign_sum_theorem_l1127_112708


namespace NUMINAMATH_CALUDE_prove_ball_size_ratio_l1127_112776

def ball_size_ratio (first_ball : ℝ) (second_ball : ℝ) (third_ball : ℝ) : Prop :=
  first_ball = second_ball / 2 ∧ 
  second_ball = 18 ∧ 
  third_ball = 27 ∧ 
  third_ball / first_ball = 3

theorem prove_ball_size_ratio : 
  ∃ (first_ball second_ball third_ball : ℝ), 
    ball_size_ratio first_ball second_ball third_ball :=
sorry

end NUMINAMATH_CALUDE_prove_ball_size_ratio_l1127_112776


namespace NUMINAMATH_CALUDE_hexagon_triangle_perimeter_ratio_l1127_112718

theorem hexagon_triangle_perimeter_ratio :
  ∀ (s_h s_t : ℝ),
  s_h > 0 → s_t > 0 →
  (s_t^2 * Real.sqrt 3) / 4 = 2 * ((3 * s_h^2 * Real.sqrt 3) / 2) →
  (3 * s_t) / (6 * s_h) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_hexagon_triangle_perimeter_ratio_l1127_112718


namespace NUMINAMATH_CALUDE_symmetric_through_swaps_l1127_112709

/-- A binary digit (0 or 1) -/
inductive BinaryDigit : Type
| zero : BinaryDigit
| one : BinaryDigit

/-- A sequence of binary digits -/
def BinarySequence := List BinaryDigit

/-- Swap operation that exchanges two elements in a list at given indices -/
def swap (seq : BinarySequence) (i j : Nat) : BinarySequence :=
  sorry

/-- Check if a sequence is symmetric -/
def isSymmetric (seq : BinarySequence) : Prop :=
  sorry

/-- The main theorem stating that any binary sequence of length 1999 can be made symmetric through swaps -/
theorem symmetric_through_swaps (seq : BinarySequence) (h : seq.length = 1999) :
  ∃ (swapSequence : List (Nat × Nat)), 
    isSymmetric (swapSequence.foldl (λ s (i, j) => swap s i j) seq) :=
  sorry

end NUMINAMATH_CALUDE_symmetric_through_swaps_l1127_112709


namespace NUMINAMATH_CALUDE_coplanar_vectors_lambda_l1127_112775

/-- Given three vectors a, b, and c in R³, if they are coplanar and have specific coordinates,
    then the third coordinate of c is 65/7. -/
theorem coplanar_vectors_lambda (a b c : ℝ × ℝ × ℝ) :
  a = (2, -1, 3) →
  b = (-1, 4, -2) →
  c.1 = 7 ∧ c.2.1 = 5 →
  (∃ (p q : ℝ), c = p • a + q • b) →
  c.2.2 = 65 / 7 := by
  sorry

end NUMINAMATH_CALUDE_coplanar_vectors_lambda_l1127_112775


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l1127_112722

theorem quadratic_root_sum (a b : ℤ) : 
  (∃ x : ℝ, x^2 + a*x + b = 0 ∧ x = Real.sqrt (7 - 4 * Real.sqrt 3)) →
  a + b = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l1127_112722


namespace NUMINAMATH_CALUDE_expression_value_l1127_112761

theorem expression_value (x y : ℤ) (hx : x = -2) (hy : y = -4) :
  5 * (x - y)^2 - x * y = 28 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1127_112761


namespace NUMINAMATH_CALUDE_lcm_of_12_and_16_l1127_112701

theorem lcm_of_12_and_16 :
  let n : ℕ := 12
  let m : ℕ := 16
  let gcf : ℕ := 4
  Nat.gcd n m = gcf →
  Nat.lcm n m = 48 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_12_and_16_l1127_112701


namespace NUMINAMATH_CALUDE_arithmetic_vector_sequence_sum_parallel_l1127_112762

/-- An arithmetic vector sequence in 2D space -/
def ArithmeticVectorSequence (a : ℕ → ℝ × ℝ) : Prop :=
  ∃ d : ℝ × ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first n vectors in a sequence -/
def VectorSum (a : ℕ → ℝ × ℝ) (n : ℕ) : ℝ × ℝ :=
  (List.range n).map a |>.sum

/-- Two vectors are parallel -/
def Parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = k • w

theorem arithmetic_vector_sequence_sum_parallel
  (a : ℕ → ℝ × ℝ) (h : ArithmeticVectorSequence a) :
  Parallel (VectorSum a 21) (a 11) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_vector_sequence_sum_parallel_l1127_112762


namespace NUMINAMATH_CALUDE_incircle_tangent_smaller_triangle_perimeter_l1127_112729

/-- Given a triangle with sides a, b, c and an inscribed incircle, 
    the perimeter of the smaller triangle formed by a tangent to the incircle 
    intersecting the two longer sides is equal to 2 * (semiperimeter - shortest_side) -/
theorem incircle_tangent_smaller_triangle_perimeter 
  (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_sides : a = 6 ∧ b = 10 ∧ c = 12) : 
  let p := (a + b + c) / 2
  2 * (p - min a (min b c)) = 28 := by sorry

end NUMINAMATH_CALUDE_incircle_tangent_smaller_triangle_perimeter_l1127_112729


namespace NUMINAMATH_CALUDE_total_weight_equals_sum_l1127_112719

/-- The weight of the blue ball in pounds -/
def blue_ball_weight : ℝ := 6

/-- The weight of the brown ball in pounds -/
def brown_ball_weight : ℝ := 3.12

/-- The total weight of both balls in pounds -/
def total_weight : ℝ := blue_ball_weight + brown_ball_weight

/-- Theorem: The total weight is equal to the sum of individual weights -/
theorem total_weight_equals_sum : total_weight = 9.12 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_equals_sum_l1127_112719


namespace NUMINAMATH_CALUDE_profit_at_8750_max_profit_price_l1127_112727

-- Define constants
def cost_price : ℝ := 40
def initial_selling_price : ℝ := 50
def initial_monthly_sales : ℝ := 500
def price_increase_step : ℝ := 1
def sales_decrease_step : ℝ := 10

-- Define functions
def selling_price (x : ℝ) : ℝ := initial_selling_price + x
def monthly_sales (x : ℝ) : ℝ := initial_monthly_sales - sales_decrease_step * x
def monthly_profit (x : ℝ) : ℝ := (monthly_sales x) * (selling_price x - cost_price)

-- Theorem statements
theorem profit_at_8750 (x : ℝ) : 
  monthly_profit x = 8750 → (x = 25 ∨ x = 15) := by sorry

theorem max_profit_price : 
  ∃ x : ℝ, ∀ y : ℝ, monthly_profit x ≥ monthly_profit y ∧ selling_price x = 70 := by sorry

end NUMINAMATH_CALUDE_profit_at_8750_max_profit_price_l1127_112727


namespace NUMINAMATH_CALUDE_number_division_problem_l1127_112770

theorem number_division_problem (x : ℚ) : x / 5 = 80 + x / 6 → x = 2400 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l1127_112770


namespace NUMINAMATH_CALUDE_jake_initial_bitcoins_l1127_112728

def initial_bitcoins : ℕ → Prop
| b => let after_first_donation := b - 20
       let after_giving_half := (after_first_donation) / 2
       let after_tripling := 3 * after_giving_half
       let final_amount := after_tripling - 10
       final_amount = 80

theorem jake_initial_bitcoins : initial_bitcoins 80 := by
  sorry

end NUMINAMATH_CALUDE_jake_initial_bitcoins_l1127_112728


namespace NUMINAMATH_CALUDE_amanda_purchase_cost_l1127_112766

def dress_price : ℚ := 50
def shoes_price : ℚ := 75
def dress_discount : ℚ := 0.30
def shoes_discount : ℚ := 0.25
def tax_rate : ℚ := 0.05

def total_cost : ℚ :=
  let dress_discounted := dress_price * (1 - dress_discount)
  let shoes_discounted := shoes_price * (1 - shoes_discount)
  let subtotal := dress_discounted + shoes_discounted
  let tax := subtotal * tax_rate
  subtotal + tax

theorem amanda_purchase_cost : total_cost = 95.81 := by
  sorry

end NUMINAMATH_CALUDE_amanda_purchase_cost_l1127_112766


namespace NUMINAMATH_CALUDE_sum_reciprocals_l1127_112723

/-- Given two positive integers m and n with sum 60, HCF 6, and LCM 210, prove that 1/m + 1/n = 1/21 -/
theorem sum_reciprocals (m n : ℕ+) 
  (h_sum : m + n = 60)
  (h_hcf : Nat.gcd m.val n.val = 6)
  (h_lcm : Nat.lcm m.val n.val = 210) : 
  1 / (m : ℚ) + 1 / (n : ℚ) = 1 / 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l1127_112723


namespace NUMINAMATH_CALUDE_ratio_equality_l1127_112795

theorem ratio_equality (x : ℝ) : (0.6 / x = 5 / 8) → x = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1127_112795


namespace NUMINAMATH_CALUDE_anika_pencils_excess_l1127_112760

theorem anika_pencils_excess (reeta_pencils : ℕ) (total_pencils : ℕ) (anika_pencils : ℕ) : 
  reeta_pencils = 20 →
  anika_pencils + reeta_pencils = total_pencils →
  total_pencils = 64 →
  ∃ m : ℕ, anika_pencils = 2 * reeta_pencils + m →
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_anika_pencils_excess_l1127_112760


namespace NUMINAMATH_CALUDE_unique_n_with_divisor_sum_property_l1127_112732

def isDivisor (d n : ℕ) : Prop := n % d = 0

theorem unique_n_with_divisor_sum_property :
  ∃! n : ℕ+, 
    (∃ (d₁ d₂ d₃ d₄ : ℕ+),
      isDivisor d₁ n ∧ isDivisor d₂ n ∧ isDivisor d₃ n ∧ isDivisor d₄ n ∧
      d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄ ∧
      d₁ = 1 ∧
      n = d₁^2 + d₂^2 + d₃^2 + d₄^2) ∧
    (∀ d : ℕ+, isDivisor d n → d = 1 ∨ d ≥ d₂) ∧
    n = 130 :=
by sorry

end NUMINAMATH_CALUDE_unique_n_with_divisor_sum_property_l1127_112732


namespace NUMINAMATH_CALUDE_vector_sum_closed_polygon_l1127_112716

variable {V : Type*} [AddCommGroup V]

/-- Given vectors AB, CF, BC, and FA in a vector space V, 
    their sum is equal to the zero vector. -/
theorem vector_sum_closed_polygon (AB CF BC FA : V) :
  AB + CF + BC + FA = (0 : V) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_closed_polygon_l1127_112716


namespace NUMINAMATH_CALUDE_original_triangle_area_l1127_112721

/-- Given a triangle whose dimensions are quadrupled to form a new triangle with an area of 256 square feet, 
    the area of the original triangle is 16 square feet. -/
theorem original_triangle_area (original : ℝ) (new : ℝ) : 
  new = 4 * original →  -- The dimensions are quadrupled
  new^2 = 256 →         -- The area of the new triangle is 256 square feet
  original^2 = 16 :=    -- The area of the original triangle is 16 square feet
by sorry

end NUMINAMATH_CALUDE_original_triangle_area_l1127_112721


namespace NUMINAMATH_CALUDE_commonly_used_charts_characterization_l1127_112789

/-- A type representing different types of charts -/
inductive Chart
  | ContingencyTable
  | ThreeDimensionalBarChart
  | TwoDimensionalBarChart
  | OtherChart

/-- The set of charts commonly used for analyzing relationships between two categorical variables -/
def commonly_used_charts : Set Chart := sorry

/-- The theorem stating that the commonly used charts are exactly the contingency tables,
    three-dimensional bar charts, and two-dimensional bar charts -/
theorem commonly_used_charts_characterization :
  commonly_used_charts = {Chart.ContingencyTable, Chart.ThreeDimensionalBarChart, Chart.TwoDimensionalBarChart} := by sorry

end NUMINAMATH_CALUDE_commonly_used_charts_characterization_l1127_112789


namespace NUMINAMATH_CALUDE_flora_initial_daily_milk_l1127_112724

def total_milk : ℕ := 105
def weeks : ℕ := 3
def days_per_week : ℕ := 7
def brother_additional : ℕ := 2

theorem flora_initial_daily_milk :
  let total_days : ℕ := weeks * days_per_week
  let flora_initial_think : ℕ := total_milk / total_days
  flora_initial_think = 5 := by sorry

end NUMINAMATH_CALUDE_flora_initial_daily_milk_l1127_112724


namespace NUMINAMATH_CALUDE_contribution_increase_l1127_112791

theorem contribution_increase (initial_contributions : ℕ) (initial_average : ℚ) (new_contribution : ℚ) :
  initial_contributions = 3 →
  initial_average = 75 →
  new_contribution = 150 →
  let total_initial := initial_contributions * initial_average
  let new_total := total_initial + new_contribution
  let new_average := new_total / (initial_contributions + 1)
  let increase := new_average - initial_average
  let percentage_increase := (increase / initial_average) * 100
  percentage_increase = 25 := by
  sorry

end NUMINAMATH_CALUDE_contribution_increase_l1127_112791


namespace NUMINAMATH_CALUDE_club_members_count_l1127_112774

/-- The number of members in the club -/
def n : ℕ := sorry

/-- The age of the old (replaced) member -/
def O : ℕ := sorry

/-- The age of the new member -/
def N : ℕ := sorry

/-- The average age remains unchanged after replacement and 3 years -/
axiom avg_unchanged : (n * O + 3 * n) / n = (n * N + 3 * n) / n

/-- The difference between the ages of the replaced and new member is 15 -/
axiom age_difference : O - N = 15

/-- Theorem: The number of members in the club is 5 -/
theorem club_members_count : n = 5 := by sorry

end NUMINAMATH_CALUDE_club_members_count_l1127_112774


namespace NUMINAMATH_CALUDE_jelly_cost_l1127_112731

theorem jelly_cost (N C J : ℕ) (h1 : N > 1) (h2 : 3 * N * C + 6 * N * J = 312) : 
  (6 * N * J : ℚ) / 100 = 0.72 := by sorry

end NUMINAMATH_CALUDE_jelly_cost_l1127_112731


namespace NUMINAMATH_CALUDE_summer_program_undergrads_l1127_112742

theorem summer_program_undergrads (total_students : ℕ) 
  (coding_team_ugrad_percent : ℚ) (coding_team_grad_percent : ℚ) :
  total_students = 36 →
  coding_team_ugrad_percent = 1/5 →
  coding_team_grad_percent = 1/4 →
  ∃ (undergrads grads coding_team_size : ℕ),
    undergrads + grads = total_students ∧
    coding_team_size * 2 = coding_team_ugrad_percent * undergrads + coding_team_grad_percent * grads ∧
    undergrads = 20 := by
  sorry

end NUMINAMATH_CALUDE_summer_program_undergrads_l1127_112742


namespace NUMINAMATH_CALUDE_shooting_competition_l1127_112785

theorem shooting_competition (hit_rate_A hit_rate_B prob_total_2 : ℚ) : 
  hit_rate_A = 3/5 →
  prob_total_2 = 9/20 →
  hit_rate_A * (1 - hit_rate_B) + (1 - hit_rate_A) * hit_rate_B = prob_total_2 →
  hit_rate_B = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_shooting_competition_l1127_112785


namespace NUMINAMATH_CALUDE_system_of_equations_solution_system_of_inequalities_solution_l1127_112703

-- System of equations
theorem system_of_equations_solution :
  ∃! (x y : ℝ), x + 2*y = 7 ∧ 3*x + y = 6 ∧ x = 1 ∧ y = 3 := by sorry

-- System of inequalities
theorem system_of_inequalities_solution :
  ∀ x : ℝ, (2*(x - 1) + 1 > -3 ∧ x - 1 ≤ (1 + x) / 3) ↔ (-1 < x ∧ x ≤ 2) := by sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_system_of_inequalities_solution_l1127_112703


namespace NUMINAMATH_CALUDE_average_age_campo_verde_l1127_112740

/-- Proves that the average age of a population is 40 years, given the specified conditions -/
theorem average_age_campo_verde (H : ℕ) (h_positive : H > 0) : 
  let M := (3 / 2 : ℚ) * H
  let total_population := H + M
  let men_age_sum := 37 * H
  let women_age_sum := 42 * M
  let total_age_sum := men_age_sum + women_age_sum
  (total_age_sum / total_population : ℚ) = 40 := by
sorry


end NUMINAMATH_CALUDE_average_age_campo_verde_l1127_112740


namespace NUMINAMATH_CALUDE_initial_quarters_l1127_112798

def quarters_after_events (initial : ℕ) : ℕ :=
  let after_doubling := initial * 2
  let after_second_year := after_doubling + 3 * 12
  let after_third_year := after_second_year + 4
  let before_loss := after_third_year
  (before_loss * 3) / 4

theorem initial_quarters (initial : ℕ) : 
  quarters_after_events initial = 105 ↔ initial = 50 := by
  sorry

#eval quarters_after_events 50  -- Should output 105

end NUMINAMATH_CALUDE_initial_quarters_l1127_112798


namespace NUMINAMATH_CALUDE_factory_output_growth_rate_l1127_112758

theorem factory_output_growth_rate (x : ℝ) : 
  (∀ y : ℝ, y > 0 → (1 + x)^2 * y = 1.2 * y) → 
  x < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_factory_output_growth_rate_l1127_112758


namespace NUMINAMATH_CALUDE_water_source_distance_l1127_112738

-- Define the actual distance to the water source
def d : ℝ := sorry

-- Alice's statement is false
axiom alice_false : ¬(d ≥ 8)

-- Bob's statement is false
axiom bob_false : ¬(d ≤ 6)

-- Charlie's statement is false
axiom charlie_false : ¬(d = 7)

-- Theorem to prove
theorem water_source_distance :
  d ∈ Set.union (Set.Ioo 6 7) (Set.Ioo 7 8) :=
sorry

end NUMINAMATH_CALUDE_water_source_distance_l1127_112738


namespace NUMINAMATH_CALUDE_M_is_graph_of_square_function_l1127_112771

def M : Set (ℝ × ℝ) := {p | p.2 = p.1^2}

theorem M_is_graph_of_square_function :
  M = {p : ℝ × ℝ | p.2 = p.1^2} := by sorry

end NUMINAMATH_CALUDE_M_is_graph_of_square_function_l1127_112771


namespace NUMINAMATH_CALUDE_red_light_runners_estimate_l1127_112707

/-- Represents the result of a survey on traffic law compliance -/
structure SurveyResult where
  total_students : ℕ
  yes_answers : ℕ
  id_range : Finset ℕ
  odd_ids : Finset ℕ

/-- Calculates the estimated number of students who have run a red light -/
def estimate_red_light_runners (result : SurveyResult) : ℕ :=
  2 * (result.yes_answers - result.odd_ids.card / 2)

/-- Theorem stating the estimated number of red light runners based on the survey -/
theorem red_light_runners_estimate 
  (result : SurveyResult)
  (h1 : result.total_students = 300)
  (h2 : result.yes_answers = 90)
  (h3 : result.id_range = Finset.range 300)
  (h4 : result.odd_ids = result.id_range.filter (fun n => n % 2 = 1)) :
  estimate_red_light_runners result = 30 := by
  sorry

end NUMINAMATH_CALUDE_red_light_runners_estimate_l1127_112707


namespace NUMINAMATH_CALUDE_computer_on_time_l1127_112782

/-- Represents days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

/-- Represents time of day in hours and minutes -/
structure Time where
  hour : ℕ
  minute : ℕ
  h_valid : hour < 24
  m_valid : minute < 60

/-- Represents a specific moment (day and time) -/
structure Moment where
  day : Day
  time : Time

def computer_on_duration : ℕ := 100

def computer_off_moment : Moment :=
  { day := Day.Friday
  , time := { hour := 17, minute := 0, h_valid := by norm_num, m_valid := by norm_num } }

theorem computer_on_time (on_moment off_moment : Moment) 
  (h : off_moment = computer_off_moment) 
  (duration : ℕ) (h_duration : duration = computer_on_duration) :
  on_moment = 
    { day := Day.Monday
    , time := { hour := 13, minute := 0, h_valid := by norm_num, m_valid := by norm_num } } :=
  sorry

end NUMINAMATH_CALUDE_computer_on_time_l1127_112782


namespace NUMINAMATH_CALUDE_alyssa_grape_cost_l1127_112792

/-- The amount Alyssa paid for cherries in dollars -/
def cherry_cost : ℚ := 9.85

/-- The total amount Alyssa spent in dollars -/
def total_spent : ℚ := 21.93

/-- The amount Alyssa paid for grapes in dollars -/
def grape_cost : ℚ := total_spent - cherry_cost

theorem alyssa_grape_cost : grape_cost = 12.08 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_grape_cost_l1127_112792


namespace NUMINAMATH_CALUDE_plot_length_is_61_l1127_112788

/-- Proves that the length of a rectangular plot is 61 meters given the specified conditions. -/
theorem plot_length_is_61 (breadth : ℝ) (length : ℝ) (fencing_cost_per_meter : ℝ) (total_fencing_cost : ℝ) :
  length = breadth + 22 →
  fencing_cost_per_meter = 26.5 →
  total_fencing_cost = 5300 →
  fencing_cost_per_meter * (2 * length + 2 * breadth) = total_fencing_cost →
  length = 61 := by
  sorry

end NUMINAMATH_CALUDE_plot_length_is_61_l1127_112788


namespace NUMINAMATH_CALUDE_line_through_parabola_focus_l1127_112799

/-- The value of 'a' for a line ax - y + 1 = 0 passing through the focus of the parabola y^2 = 4x -/
theorem line_through_parabola_focus (a : ℝ) : 
  (∃ x y : ℝ, y^2 = 4*x ∧ a*x - y + 1 = 0 ∧ x = 1 ∧ y = 0) → a = -1 :=
by sorry

end NUMINAMATH_CALUDE_line_through_parabola_focus_l1127_112799


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l1127_112756

theorem simultaneous_equations_solution (m : ℝ) : 
  ∃ (x y : ℝ), y = 3 * m * x + 2 ∧ y = (3 * m - 2) * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l1127_112756


namespace NUMINAMATH_CALUDE_bryans_book_collection_l1127_112764

theorem bryans_book_collection (books_per_continent : ℕ) (total_books : ℕ) 
  (h1 : books_per_continent = 122) 
  (h2 : total_books = 488) : 
  total_books / books_per_continent = 4 := by
sorry

end NUMINAMATH_CALUDE_bryans_book_collection_l1127_112764


namespace NUMINAMATH_CALUDE_ducks_arrived_later_l1127_112743

theorem ducks_arrived_later (initial_ducks : ℕ) (initial_geese : ℕ) (final_ducks : ℕ) (final_geese : ℕ) : 
  initial_ducks = 25 →
  initial_geese = 2 * initial_ducks - 10 →
  final_geese = initial_geese - (15 - 5) →
  final_geese = final_ducks + 1 →
  final_ducks - initial_ducks = 4 :=
by sorry

end NUMINAMATH_CALUDE_ducks_arrived_later_l1127_112743


namespace NUMINAMATH_CALUDE_distance_between_points_l1127_112735

theorem distance_between_points (A B : ℝ) : A = 3 ∧ B = -7 → |A - B| = 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1127_112735


namespace NUMINAMATH_CALUDE_sum_four_characterization_l1127_112796

/-- Represents the outcome of rolling a single die -/
def DieOutcome := Fin 6

/-- Represents the outcome of rolling two dice -/
def TwoDiceOutcome := DieOutcome × DieOutcome

/-- The sum of points obtained when rolling two dice -/
def sumPoints (outcome : TwoDiceOutcome) : Nat :=
  outcome.1.val + 1 + outcome.2.val + 1

/-- The event where the sum of points is 4 -/
def sumIsFour (outcome : TwoDiceOutcome) : Prop :=
  sumPoints outcome = 4

/-- The event where one die shows 3 and the other shows 1 -/
def threeAndOne (outcome : TwoDiceOutcome) : Prop :=
  (outcome.1.val = 2 ∧ outcome.2.val = 0) ∨ (outcome.1.val = 0 ∧ outcome.2.val = 2)

/-- The event where both dice show 2 -/
def bothTwo (outcome : TwoDiceOutcome) : Prop :=
  outcome.1.val = 1 ∧ outcome.2.val = 1

theorem sum_four_characterization (outcome : TwoDiceOutcome) :
  sumIsFour outcome ↔ threeAndOne outcome ∨ bothTwo outcome := by
  sorry

end NUMINAMATH_CALUDE_sum_four_characterization_l1127_112796


namespace NUMINAMATH_CALUDE_black_pens_removed_l1127_112768

/-- Proves that 7 black pens were removed from a jar given the initial and final conditions -/
theorem black_pens_removed (initial_blue : ℕ) (initial_black : ℕ) (initial_red : ℕ)
  (blue_removed : ℕ) (final_count : ℕ)
  (h1 : initial_blue = 9)
  (h2 : initial_black = 21)
  (h3 : initial_red = 6)
  (h4 : blue_removed = 4)
  (h5 : final_count = 25) :
  initial_black - (initial_blue + initial_black + initial_red - blue_removed - final_count) = 7 := by
  sorry

#check black_pens_removed

end NUMINAMATH_CALUDE_black_pens_removed_l1127_112768


namespace NUMINAMATH_CALUDE_sum_of_digits_9ab_l1127_112777

def a : ℕ := 10^2023 - 1

def b : ℕ := 7 * (10^2023 - 1) / 9

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_9ab : sum_of_digits (9 * a * b) = 36410 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_9ab_l1127_112777


namespace NUMINAMATH_CALUDE_grid_path_count_l1127_112725

/-- The number of paths on a grid from (0,0) to (m,n) using only right and up moves -/
def gridPaths (m n : ℕ) : ℕ := Nat.choose (m + n) n

/-- The dimensions of our grid -/
def gridWidth : ℕ := 6
def gridHeight : ℕ := 5

/-- The total number of steps required -/
def totalSteps : ℕ := gridWidth + gridHeight

theorem grid_path_count :
  gridPaths gridWidth gridHeight = 462 := by
  sorry

#eval gridPaths gridWidth gridHeight

end NUMINAMATH_CALUDE_grid_path_count_l1127_112725


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l1127_112752

/-- 
Given a parallelogram with adjacent sides of lengths s and 2s forming a 60-degree angle,
if the area is 12√3 square units, then s = √6.
-/
theorem parallelogram_side_length (s : ℝ) : 
  s > 0 →  -- Assume s is positive
  (2 * s * s * Real.sqrt 3 = 12 * Real.sqrt 3) →  -- Area formula
  s = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l1127_112752


namespace NUMINAMATH_CALUDE_decreasing_quadratic_range_l1127_112754

/-- A quadratic function f(x) with parameter a. -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The theorem stating that if f(x) is decreasing on (-∞, 4], then a ≤ -3. -/
theorem decreasing_quadratic_range (a : ℝ) :
  (∀ x ≤ 4, ∀ y ≤ 4, x < y → f a x > f a y) →
  a ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_range_l1127_112754


namespace NUMINAMATH_CALUDE_percent_relationship_l1127_112720

theorem percent_relationship (a b : ℝ) (h : a = 1.25 * b) : 4 * b = 3.2 * a := by
  sorry

end NUMINAMATH_CALUDE_percent_relationship_l1127_112720


namespace NUMINAMATH_CALUDE_parallel_lines_in_special_triangle_l1127_112714

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : Point)
  (b : Point)

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := sorry

/-- Constructs an equilateral triangle given three points -/
def equilateral_triangle (a b c : Point) : Prop := sorry

theorem parallel_lines_in_special_triangle 
  (A B C M K : Point) 
  (h1 : equilateral_triangle A B C)
  (h2 : M.x ≥ A.x ∧ M.x ≤ B.x ∧ M.y = A.y)  -- M is on side AB
  (h3 : equilateral_triangle M K C) :
  parallel (Line.mk A C) (Line.mk B K) := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_in_special_triangle_l1127_112714


namespace NUMINAMATH_CALUDE_toys_sold_l1127_112763

def initial_toys : ℕ := 7
def remaining_toys : ℕ := 4

theorem toys_sold : initial_toys - remaining_toys = 3 := by
  sorry

end NUMINAMATH_CALUDE_toys_sold_l1127_112763


namespace NUMINAMATH_CALUDE_salesman_commission_percentage_l1127_112730

/-- Proves that the flat commission percentage in the previous scheme is 5% --/
theorem salesman_commission_percentage :
  ∀ (previous_commission_percentage : ℝ),
    -- New scheme: Rs. 1000 fixed salary + 2.5% commission on sales exceeding Rs. 4,000
    let new_scheme_fixed_salary : ℝ := 1000
    let new_scheme_commission_rate : ℝ := 2.5 / 100
    let sales_threshold : ℝ := 4000
    -- Total sales
    let total_sales : ℝ := 12000
    -- Calculate new scheme remuneration
    let new_scheme_commission : ℝ := new_scheme_commission_rate * (total_sales - sales_threshold)
    let new_scheme_remuneration : ℝ := new_scheme_fixed_salary + new_scheme_commission
    -- Previous scheme remuneration
    let previous_scheme_remuneration : ℝ := previous_commission_percentage / 100 * total_sales
    -- New scheme remuneration is Rs. 600 more than the previous scheme
    new_scheme_remuneration = previous_scheme_remuneration + 600
    →
    previous_commission_percentage = 5 := by
  sorry

end NUMINAMATH_CALUDE_salesman_commission_percentage_l1127_112730


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l1127_112700

/-- The diagonal of a rectangle with width 16 and length 12 is 20. -/
theorem rectangle_diagonal : ∃ (d : ℝ), d = 20 ∧ d^2 = 16^2 + 12^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l1127_112700


namespace NUMINAMATH_CALUDE_function_value_theorem_l1127_112711

/-- Given functions f and g, prove that f(2) = 2 under certain conditions -/
theorem function_value_theorem (a b c : ℝ) (h_abc : a * b * c ≠ 0) :
  let f := fun (x : ℝ) ↦ a * x^2 + b * Real.cos x
  let g := fun (x : ℝ) ↦ c * Real.sin x
  (f 2 + g 2 = 3) → (f (-2) + g (-2) = 1) → f 2 = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_function_value_theorem_l1127_112711


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l1127_112784

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.exp x

theorem f_derivative_at_zero : 
  deriv f 0 = 2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l1127_112784


namespace NUMINAMATH_CALUDE_multiples_of_2_are_even_is_universal_l1127_112741

/-- A predicate representing a property of natural numbers -/
def P (n : ℕ) : Prop := Even n

/-- Definition of a universal proposition -/
def UniversalProposition (P : α → Prop) : Prop :=
  ∀ x, P x

/-- The statement "All multiples of 2 are even" -/
def AllMultiplesOf2AreEven : Prop :=
  ∀ n : ℕ, 2 ∣ n → Even n

/-- Theorem stating that "All multiples of 2 are even" is a universal proposition -/
theorem multiples_of_2_are_even_is_universal :
  UniversalProposition (λ n => 2 ∣ n → Even n) :=
sorry

end NUMINAMATH_CALUDE_multiples_of_2_are_even_is_universal_l1127_112741


namespace NUMINAMATH_CALUDE_edge_probability_is_three_nineteenths_l1127_112773

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  vertices : Finset (Fin 20)
  edges : Finset (Fin 20 × Fin 20)
  vertex_degree : ∀ v : Fin 20, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- The probability of selecting two vertices that form an edge in a regular dodecahedron -/
def edge_probability (d : RegularDodecahedron) : ℚ :=
  d.edges.card / Nat.choose 20 2

/-- Theorem stating the probability of selecting two vertices that form an edge -/
theorem edge_probability_is_three_nineteenths (d : RegularDodecahedron) :
  edge_probability d = 3 / 19 := by
  sorry

end NUMINAMATH_CALUDE_edge_probability_is_three_nineteenths_l1127_112773


namespace NUMINAMATH_CALUDE_max_partition_product_l1127_112748

def partition_product (p : List Nat) : Nat :=
  p.prod

def is_valid_partition (p : List Nat) : Prop :=
  p.sum = 25 ∧ p.all (· > 0) ∧ p.length ≤ 25

theorem max_partition_product :
  ∃ (max_p : List Nat), 
    is_valid_partition max_p ∧ 
    partition_product max_p = 8748 ∧
    ∀ (p : List Nat), is_valid_partition p → partition_product p ≤ 8748 := by
  sorry

end NUMINAMATH_CALUDE_max_partition_product_l1127_112748


namespace NUMINAMATH_CALUDE_complex_number_magnitude_l1127_112749

theorem complex_number_magnitude (a : ℝ) (z : ℂ) : 
  z = (1 + a * Complex.I) / Complex.I → 
  z.re = z.im →
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_l1127_112749


namespace NUMINAMATH_CALUDE_f_always_above_y_l1127_112779

/-- The function f(x) = mx^2 - 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 3

/-- The line y = mx - m -/
def y (m : ℝ) (x : ℝ) : ℝ := m * x - m

/-- Theorem stating that f(x) is always above y for all real x if and only if m > 4 -/
theorem f_always_above_y (m : ℝ) : 
  (∀ x : ℝ, f m x > y m x) ↔ m > 4 := by
  sorry

end NUMINAMATH_CALUDE_f_always_above_y_l1127_112779


namespace NUMINAMATH_CALUDE_flowerbed_fraction_is_five_thirty_sixths_l1127_112747

/-- Represents the dimensions and properties of a rectangular yard with flower beds. -/
structure YardWithFlowerBeds where
  length : ℝ
  width : ℝ
  trapezoid_side1 : ℝ
  trapezoid_side2 : ℝ
  
/-- Calculates the fraction of the yard occupied by flower beds. -/
def flowerbed_fraction (yard : YardWithFlowerBeds) : ℚ :=
  sorry

/-- Theorem stating that the fraction of the yard occupied by flower beds is 5/36. -/
theorem flowerbed_fraction_is_five_thirty_sixths 
  (yard : YardWithFlowerBeds) 
  (h1 : yard.length = 30)
  (h2 : yard.width = 6)
  (h3 : yard.trapezoid_side1 = 20)
  (h4 : yard.trapezoid_side2 = 30) :
  flowerbed_fraction yard = 5 / 36 :=
sorry

end NUMINAMATH_CALUDE_flowerbed_fraction_is_five_thirty_sixths_l1127_112747


namespace NUMINAMATH_CALUDE_corrected_mean_problem_l1127_112794

/-- Calculates the corrected mean of observations after fixing an error --/
def corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n * original_mean + (correct_value - incorrect_value)) / n

/-- Theorem stating the corrected mean for the given problem --/
theorem corrected_mean_problem :
  corrected_mean 50 36 23 34 = 36.22 := by
  sorry

#eval corrected_mean 50 36 23 34

end NUMINAMATH_CALUDE_corrected_mean_problem_l1127_112794


namespace NUMINAMATH_CALUDE_y_coordinate_difference_zero_l1127_112715

/-- Given two points (m, n) and (m + 3, n + q) on the line x = (y / 7) - (2 / 5),
    the difference between their y-coordinates is 0. -/
theorem y_coordinate_difference_zero
  (m n q : ℚ) : 
  (m = n / 7 - 2 / 5) →
  (m + 3 = (n + q) / 7 - 2 / 5) →
  q = 0 :=
by sorry

end NUMINAMATH_CALUDE_y_coordinate_difference_zero_l1127_112715


namespace NUMINAMATH_CALUDE_odot_ten_five_l1127_112769

-- Define the ⊙ operation
def odot (a b : ℚ) : ℚ := a + (4 * a) / (3 * b)

-- Theorem statement
theorem odot_ten_five : odot 10 5 = 38 / 3 := by
  sorry

end NUMINAMATH_CALUDE_odot_ten_five_l1127_112769


namespace NUMINAMATH_CALUDE_unknown_number_proof_l1127_112704

theorem unknown_number_proof (x : ℝ) : 1.75 * x = 63 → x = 36 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l1127_112704


namespace NUMINAMATH_CALUDE_pirate_treasure_division_l1127_112793

theorem pirate_treasure_division (N : ℕ) (h : 3000 ≤ N ∧ N ≤ 4000) :
  let remaining1 := (3 * N - 6) / 4
  let remaining2 := (9 * N - 42) / 16
  let remaining3 := (108 * N - 888) / 256
  let remaining4 := (82944 * N - 876400) / 262144
  let share1 := (N + 6) / 4
  let share2 := (3 * N + 18) / 16
  let share3 := (9 * N + 54) / 64
  let share4 := (108 * N + 648) / 1024
  let final_share := remaining4 / 4
  (share1 + final_share = 1178) ∧
  (share2 + final_share = 954) ∧
  (share3 + final_share = 786) ∧
  (share4 + final_share = 660) := by
sorry

end NUMINAMATH_CALUDE_pirate_treasure_division_l1127_112793


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l1127_112759

theorem inequality_not_always_true (x y w : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x^2 > y^2) (hw : w ≠ 0) :
  ¬ (∀ w, x^2 * w > y^2 * w) :=
by sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l1127_112759


namespace NUMINAMATH_CALUDE_solve_for_a_l1127_112733

theorem solve_for_a : ∃ a : ℝ, 
  (∀ x y : ℝ, x = 1 ∧ y = -3 → a * x - y = 1) ∧ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1127_112733


namespace NUMINAMATH_CALUDE_min_value_ratio_l1127_112783

-- Define the arithmetic and geometric sequence properties
def is_arithmetic_sequence (x a b y : ℝ) : Prop :=
  a + b = x + y

def is_geometric_sequence (x c d y : ℝ) : Prop :=
  c * d = x * y

-- State the theorem
theorem min_value_ratio (x y a b c d : ℝ) 
  (hx : x > 0) (hy : y > 0)
  (ha : is_arithmetic_sequence x a b y)
  (hg : is_geometric_sequence x c d y) :
  (a + b)^2 / (c * d) ≥ 4 ∧ 
  ∃ (a b c d : ℝ), (a + b)^2 / (c * d) = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_ratio_l1127_112783


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1127_112739

theorem polynomial_divisibility (a b c d : ℤ) :
  (∀ x : ℤ, ∃ k : ℤ, a * x^3 + b * x^2 + c * x + d = 5 * k) →
  (∃ ka kb kc kd : ℤ, a = 5 * ka ∧ b = 5 * kb ∧ c = 5 * kc ∧ d = 5 * kd) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1127_112739


namespace NUMINAMATH_CALUDE_hannah_easter_eggs_l1127_112744

theorem hannah_easter_eggs :
  ∀ (total helen hannah : ℕ),
  total = 63 →
  hannah = 2 * helen →
  total = helen + hannah →
  hannah = 42 := by
sorry

end NUMINAMATH_CALUDE_hannah_easter_eggs_l1127_112744
