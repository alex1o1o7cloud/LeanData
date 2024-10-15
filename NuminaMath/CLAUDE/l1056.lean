import Mathlib

namespace NUMINAMATH_CALUDE_age_ratio_proof_l1056_105642

/-- Proves that the ratio of Saras's age to the combined age of Kul and Ani is 1:2 -/
theorem age_ratio_proof (kul_age saras_age ani_age : ℕ) 
  (h1 : kul_age = 22)
  (h2 : saras_age = 33)
  (h3 : ani_age = 44) : 
  (saras_age : ℚ) / (kul_age + ani_age : ℚ) = 1 / 2 := by
  sorry

#check age_ratio_proof

end NUMINAMATH_CALUDE_age_ratio_proof_l1056_105642


namespace NUMINAMATH_CALUDE_sum_equals_16x_l1056_105604

/-- Given real numbers x, y, z, and w, where y = 2x, z = 3y, and w = z + x,
    prove that x + y + z + w = 16x -/
theorem sum_equals_16x (x y z w : ℝ) 
    (h1 : y = 2 * x) 
    (h2 : z = 3 * y) 
    (h3 : w = z + x) : 
  x + y + z + w = 16 * x := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_16x_l1056_105604


namespace NUMINAMATH_CALUDE_log_inequality_range_l1056_105630

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_inequality_range (x : ℝ) :
  lg (x + 1) < lg (3 - x) ↔ -1 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_range_l1056_105630


namespace NUMINAMATH_CALUDE_wall_height_is_ten_l1056_105631

-- Define the dimensions of the rooms
def livingRoomSide : ℝ := 40
def bedroomLength : ℝ := 12
def bedroomWidth : ℝ := 10

-- Define the number of walls to be painted in each room
def livingRoomWalls : ℕ := 3
def bedroomWalls : ℕ := 4

-- Define the total area to be painted
def totalAreaToPaint : ℝ := 1640

-- Theorem statement
theorem wall_height_is_ten :
  let livingRoomPerimeter := livingRoomSide * 4
  let livingRoomPaintPerimeter := livingRoomPerimeter - livingRoomSide
  let bedroomPerimeter := 2 * (bedroomLength + bedroomWidth)
  let totalPerimeterToPaint := livingRoomPaintPerimeter + bedroomPerimeter
  totalAreaToPaint / totalPerimeterToPaint = 10 := by
  sorry

end NUMINAMATH_CALUDE_wall_height_is_ten_l1056_105631


namespace NUMINAMATH_CALUDE_stephanies_remaining_payment_l1056_105699

/-- The total amount Stephanie still needs to pay to finish her bills -/
def remaining_payment (electricity gas water internet : ℝ) 
                      (gas_paid_fraction : ℝ) 
                      (gas_additional_payment : ℝ) 
                      (water_paid_fraction : ℝ) 
                      (internet_payments : ℕ) 
                      (internet_payment_amount : ℝ) : ℝ :=
  (gas - gas_paid_fraction * gas - gas_additional_payment) +
  (water - water_paid_fraction * water) +
  (internet - internet_payments * internet_payment_amount)

/-- Stephanie's remaining bill payment theorem -/
theorem stephanies_remaining_payment :
  remaining_payment 60 40 40 25 (3/4) 5 (1/2) 4 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_stephanies_remaining_payment_l1056_105699


namespace NUMINAMATH_CALUDE_max_value_problem_l1056_105676

theorem max_value_problem (X Y Z : ℕ) (h : 2 * X + 3 * Y + Z = 18) :
  (∀ X' Y' Z' : ℕ, 2 * X' + 3 * Y' + Z' = 18 →
    X' * Y' * Z' + X' * Y' + Y' * Z' + Z' * X' ≤ X * Y * Z + X * Y + Y * Z + Z * X) →
  X * Y * Z + X * Y + Y * Z + Z * X = 24 :=
sorry

end NUMINAMATH_CALUDE_max_value_problem_l1056_105676


namespace NUMINAMATH_CALUDE_radical_simplification_l1056_105632

theorem radical_simplification (p : ℝ) (hp : p > 0) :
  Real.sqrt (12 * p) * Real.sqrt (7 * p^3) * Real.sqrt (15 * p^5) = 6 * p^4 * Real.sqrt (35 * p) := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l1056_105632


namespace NUMINAMATH_CALUDE_box_calories_l1056_105668

/-- Calculates the total calories in a box of cookies -/
def total_calories (cookies_per_bag : ℕ) (bags_per_box : ℕ) (calories_per_cookie : ℕ) : ℕ :=
  cookies_per_bag * bags_per_box * calories_per_cookie

/-- Theorem: The total calories in a box of cookies is 1600 -/
theorem box_calories :
  total_calories 20 4 20 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_box_calories_l1056_105668


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1056_105683

theorem triangle_perimeter : ∀ (a b c : ℝ),
  a = 3 ∧ b = 5 ∧ c^2 - 3*c = c - 3 ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b →
  a + b + c = 11 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1056_105683


namespace NUMINAMATH_CALUDE_derivative_equals_function_implies_zero_at_two_l1056_105600

open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem derivative_equals_function_implies_zero_at_two 
  (h : ∀ x, deriv f x = f x) : 
  deriv f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_derivative_equals_function_implies_zero_at_two_l1056_105600


namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_l1056_105612

/-- A sequence is arithmetic if the difference between consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The general term of the sequence. -/
def a (n : ℕ) (p q : ℝ) : ℝ := p * n + q

theorem sequence_is_arithmetic (p q : ℝ) :
  IsArithmeticSequence (a · p q) := by
  sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_l1056_105612


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l1056_105694

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {0, 3, 4}

theorem complement_M_intersect_N :
  (Set.compl M ∩ N) = {0, 4} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l1056_105694


namespace NUMINAMATH_CALUDE_smallest_angle_in_triangle_with_ratio_l1056_105609

theorem smallest_angle_in_triangle_with_ratio (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- angles are positive
  a + b + c = 180 →  -- sum of angles is 180°
  b = 2 * a →  -- ratio condition
  c = 3 * a →  -- ratio condition
  a = 30 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_in_triangle_with_ratio_l1056_105609


namespace NUMINAMATH_CALUDE_chessboard_inner_square_probability_l1056_105650

/-- Represents a square chessboard -/
structure Chessboard :=
  (size : ℕ)

/-- Calculates the total number of squares on the chessboard -/
def total_squares (board : Chessboard) : ℕ :=
  board.size * board.size

/-- Calculates the number of squares in the outermost two rows and columns -/
def outer_squares (board : Chessboard) : ℕ :=
  4 * board.size - 4

/-- Calculates the number of inner squares not touching the outermost two rows or columns -/
def inner_squares (board : Chessboard) : ℕ :=
  total_squares board - outer_squares board

/-- The probability of choosing an inner square -/
def inner_square_probability (board : Chessboard) : ℚ :=
  inner_squares board / total_squares board

theorem chessboard_inner_square_probability :
  ∃ (board : Chessboard), board.size = 10 ∧ inner_square_probability board = 17 / 25 := by
  sorry

end NUMINAMATH_CALUDE_chessboard_inner_square_probability_l1056_105650


namespace NUMINAMATH_CALUDE_exam_question_distribution_l1056_105677

theorem exam_question_distribution (total_questions : ℕ) 
  (group_a_marks group_b_marks group_c_marks : ℕ) 
  (group_b_questions : ℕ) :
  total_questions = 100 →
  group_a_marks = 1 →
  group_b_marks = 2 →
  group_c_marks = 3 →
  group_b_questions = 23 →
  (∀ a b c : ℕ, a + b + c = total_questions → a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1) →
  (∀ a b c : ℕ, a + b + c = total_questions → 
    a * group_a_marks ≥ (6 * (a * group_a_marks + b * group_b_marks + c * group_c_marks)) / 10) →
  ∃! c : ℕ, c = 1 ∧ ∃ a : ℕ, a + group_b_questions + c = total_questions :=
by sorry

end NUMINAMATH_CALUDE_exam_question_distribution_l1056_105677


namespace NUMINAMATH_CALUDE_exponent_simplification_l1056_105635

theorem exponent_simplification : ((-2 : ℝ) ^ 3) ^ (1/3) - (-1 : ℝ) ^ 0 = -3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_simplification_l1056_105635


namespace NUMINAMATH_CALUDE_a2_4_sufficient_not_necessary_for_a3_16_l1056_105639

/-- A geometric sequence with first term 1 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The property that a₂ = 4 is sufficient but not necessary for a₃ = 16 -/
theorem a2_4_sufficient_not_necessary_for_a3_16 :
  ∀ a : ℕ → ℝ, GeometricSequence a →
    (∀ a : ℕ → ℝ, GeometricSequence a → a 2 = 4 → a 3 = 16) ∧
    ¬(∀ a : ℕ → ℝ, GeometricSequence a → a 3 = 16 → a 2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_a2_4_sufficient_not_necessary_for_a3_16_l1056_105639


namespace NUMINAMATH_CALUDE_isosceles_triangle_condition_l1056_105661

/-- Proves that if in a triangle with sides a, b, c and opposite angles α, β, γ,
    the equation a + b = tan(γ/2) * (a * tan(α) + b * tan(β)) holds, then α = β. -/
theorem isosceles_triangle_condition 
  (a b c : ℝ) 
  (α β γ : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < α ∧ 0 < β ∧ 0 < γ)
  (h_angles : α + β + γ = Real.pi)
  (h_condition : a + b = Real.tan (γ / 2) * (a * Real.tan α + b * Real.tan β)) :
  α = β := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_condition_l1056_105661


namespace NUMINAMATH_CALUDE_max_factors_x8_minus_1_l1056_105637

theorem max_factors_x8_minus_1 : 
  ∃ (k : ℕ), k = 5 ∧ 
  (∀ (p : List (Polynomial ℝ)), 
    (∀ q ∈ p, q.degree > 0) → -- Each factor is non-constant
    (List.prod p = Polynomial.X ^ 8 - 1) → -- The product of factors equals x^8 - 1
    List.length p ≤ k) ∧ -- The number of factors is at most k
  (∃ (p : List (Polynomial ℝ)), 
    (∀ q ∈ p, q.degree > 0) ∧ 
    (List.prod p = Polynomial.X ^ 8 - 1) ∧ 
    List.length p = k) -- There exists a factorization with exactly k factors
  := by sorry

end NUMINAMATH_CALUDE_max_factors_x8_minus_1_l1056_105637


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_23_19_l1056_105665

theorem half_abs_diff_squares_23_19 : (1 / 2 : ℝ) * |23^2 - 19^2| = 84 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_23_19_l1056_105665


namespace NUMINAMATH_CALUDE_problem_solution_l1056_105671

theorem problem_solution (x y : ℝ) 
  (h1 : |x| + x + y = 8) 
  (h2 : x + |y| - y = 10) : 
  x + y = 14/5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1056_105671


namespace NUMINAMATH_CALUDE_candy_distribution_l1056_105622

theorem candy_distribution (total : ℕ) (a b c d : ℕ) : 
  total = 2013 →
  a + b + c + d = total →
  a = 2 * b + 10 →
  a = 3 * c + 18 →
  a = 5 * d - 55 →
  a = 990 :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l1056_105622


namespace NUMINAMATH_CALUDE_bond_face_value_l1056_105623

/-- The face value of a bond -/
def face_value : ℝ := 5000

/-- The interest rate as a percentage of face value -/
def interest_rate : ℝ := 0.05

/-- The selling price of the bond -/
def selling_price : ℝ := 3846.153846153846

/-- The interest amount as a percentage of selling price -/
def interest_percentage : ℝ := 0.065

theorem bond_face_value :
  face_value = selling_price * interest_percentage / interest_rate :=
by sorry

end NUMINAMATH_CALUDE_bond_face_value_l1056_105623


namespace NUMINAMATH_CALUDE_terrell_lifting_equivalence_l1056_105669

/-- The number of times Terrell lifts the weights in the initial setup -/
def initial_lifts : ℕ := 10

/-- The weight of each dumbbell in the initial setup (in pounds) -/
def initial_weight : ℕ := 25

/-- The number of dumbbells used in the initial setup -/
def initial_dumbbells : ℕ := 2

/-- The weight of the single dumbbell in the new setup (in pounds) -/
def new_weight : ℕ := 20

/-- The total weight lifted in the initial setup (in pounds) -/
def total_weight : ℕ := initial_dumbbells * initial_weight * initial_lifts

/-- The number of times Terrell must lift the new weight to achieve the same total weight -/
def required_lifts : ℕ := total_weight / new_weight

theorem terrell_lifting_equivalence :
  required_lifts = 25 := by sorry

end NUMINAMATH_CALUDE_terrell_lifting_equivalence_l1056_105669


namespace NUMINAMATH_CALUDE_expression_equality_l1056_105615

theorem expression_equality : 
  Real.sqrt 12 + 2⁻¹ + Real.cos (60 * π / 180) - 3 * Real.tan (30 * π / 180) = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1056_105615


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1056_105684

theorem unique_solution_condition (a b : ℝ) :
  (∃! x : ℝ, 3 * x - 5 + a = b * x + 1) ↔ b ≠ 3 := by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1056_105684


namespace NUMINAMATH_CALUDE_matching_probability_theorem_l1056_105606

/-- Represents the distribution of shoe pairs by color -/
structure ShoeDistribution where
  black : Nat
  brown : Nat
  gray : Nat
  red : Nat

/-- Calculates the total number of individual shoes -/
def totalShoes (d : ShoeDistribution) : Nat :=
  2 * (d.black + d.brown + d.gray + d.red)

/-- Calculates the probability of selecting a matching pair -/
def matchingProbability (d : ShoeDistribution) : Rat :=
  let total := totalShoes d
  let numerator := 
    d.black * (d.black - 1) + 
    d.brown * (d.brown - 1) + 
    d.gray * (d.gray - 1) + 
    d.red * (d.red - 1)
  (numerator : Rat) / (total * (total - 1))

/-- John's shoe distribution -/
def johnsShoes : ShoeDistribution :=
  { black := 8, brown := 4, gray := 3, red := 1 }

theorem matching_probability_theorem : 
  matchingProbability johnsShoes = 45 / 248 := by
  sorry

end NUMINAMATH_CALUDE_matching_probability_theorem_l1056_105606


namespace NUMINAMATH_CALUDE_interest_group_signup_ways_l1056_105680

theorem interest_group_signup_ways (num_students : ℕ) (num_groups : ℕ) : 
  num_students = 4 → num_groups = 3 → (num_groups ^ num_students : ℕ) = 81 := by
  sorry

end NUMINAMATH_CALUDE_interest_group_signup_ways_l1056_105680


namespace NUMINAMATH_CALUDE_greatest_power_of_four_dividing_16_factorial_l1056_105601

theorem greatest_power_of_four_dividing_16_factorial :
  (∃ k : ℕ+, k.val = 7 ∧ 
   ∀ m : ℕ+, (4 ^ m.val ∣ Nat.factorial 16) → m.val ≤ 7) :=
by sorry

end NUMINAMATH_CALUDE_greatest_power_of_four_dividing_16_factorial_l1056_105601


namespace NUMINAMATH_CALUDE_abs_neg_two_eq_two_l1056_105689

theorem abs_neg_two_eq_two : |(-2 : ℝ)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_eq_two_l1056_105689


namespace NUMINAMATH_CALUDE_keiths_cds_l1056_105628

/-- Calculates the number of CDs Keith wanted to buy based on his total spending and the price per CD -/
theorem keiths_cds (speakers_cost cd_player_cost tires_cost total_spent cd_price : ℝ) :
  speakers_cost = 136.01 →
  cd_player_cost = 139.38 →
  tires_cost = 112.46 →
  total_spent = 387.85 →
  cd_price = 6.16 →
  speakers_cost + cd_player_cost + tires_cost = total_spent →
  ⌊total_spent / cd_price⌋ = 62 :=
by sorry

end NUMINAMATH_CALUDE_keiths_cds_l1056_105628


namespace NUMINAMATH_CALUDE_jungkook_persimmons_jungkook_picked_8_persimmons_l1056_105690

theorem jungkook_persimmons : ℕ → Prop :=
  fun j : ℕ =>
    let h := 35  -- Hoseok's persimmons
    h = 4 * j + 3 → j = 8

-- Proof
theorem jungkook_picked_8_persimmons : jungkook_persimmons 8 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_persimmons_jungkook_picked_8_persimmons_l1056_105690


namespace NUMINAMATH_CALUDE_m_range_theorem_l1056_105660

/-- The range of m values satisfying the given conditions -/
def m_range : Set ℝ :=
  Set.Ioc 1 2 ∪ Set.Ici 3

/-- Condition for the first equation to have two distinct negative roots -/
def has_two_negative_roots (m : ℝ) : Prop :=
  ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

/-- Condition for the second equation to have no real roots -/
def has_no_real_roots (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

/-- Main theorem statement -/
theorem m_range_theorem (m : ℝ) :
  (has_two_negative_roots m ∨ has_no_real_roots m) ∧
  ¬(has_two_negative_roots m ∧ has_no_real_roots m) ↔
  m ∈ m_range :=
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l1056_105660


namespace NUMINAMATH_CALUDE_brother_father_age_ratio_l1056_105651

/-- Represents the ages of family members and total family age --/
structure FamilyAges where
  total : ℕ
  father : ℕ
  mother : ℕ
  sister : ℕ
  kaydence : ℕ

/-- Theorem stating the ratio of brother's age to father's age --/
theorem brother_father_age_ratio (f : FamilyAges) 
  (h1 : f.total = 200)
  (h2 : f.father = 60)
  (h3 : f.mother = f.father - 2)
  (h4 : f.sister = 40)
  (h5 : f.kaydence = 12) :
  ∃ (brother_age : ℕ), 
    brother_age = f.total - (f.father + f.mother + f.sister + f.kaydence) ∧
    2 * brother_age = f.father :=
by
  sorry

end NUMINAMATH_CALUDE_brother_father_age_ratio_l1056_105651


namespace NUMINAMATH_CALUDE_window_area_calculation_l1056_105685

/-- Calculates the area of a window given its length in meters and width in feet -/
def windowArea (lengthMeters : ℝ) (widthFeet : ℝ) : ℝ :=
  let meterToFeet : ℝ := 3.28084
  let lengthFeet : ℝ := lengthMeters * meterToFeet
  lengthFeet * widthFeet

theorem window_area_calculation :
  windowArea 2 15 = 98.4252 := by
  sorry

end NUMINAMATH_CALUDE_window_area_calculation_l1056_105685


namespace NUMINAMATH_CALUDE_dvd_average_price_l1056_105681

/-- Calculates the average price of DVDs bought from two boxes with different prices -/
theorem dvd_average_price (box1_count : ℕ) (box1_price : ℚ) (box2_count : ℕ) (box2_price : ℚ) :
  box1_count = 10 →
  box1_price = 2 →
  box2_count = 5 →
  box2_price = 5 →
  (box1_count * box1_price + box2_count * box2_price) / (box1_count + box2_count : ℚ) = 3 := by
  sorry

#check dvd_average_price

end NUMINAMATH_CALUDE_dvd_average_price_l1056_105681


namespace NUMINAMATH_CALUDE_smallest_X_value_l1056_105618

/-- A function that checks if a natural number consists only of 0s and 1s in its decimal representation -/
def onlyZerosAndOnes (n : ℕ) : Prop := sorry

/-- The smallest positive integer T that satisfies the given conditions -/
def T : ℕ := sorry

theorem smallest_X_value :
  T > 0 ∧
  onlyZerosAndOnes T ∧
  T % 15 = 0 ∧
  (∀ t : ℕ, t > 0 → onlyZerosAndOnes t → t % 15 = 0 → t ≥ T) →
  T / 15 = 74 := by sorry

end NUMINAMATH_CALUDE_smallest_X_value_l1056_105618


namespace NUMINAMATH_CALUDE_asymptotes_sum_l1056_105696

theorem asymptotes_sum (A B C : ℤ) : 
  (∀ x, x^3 + A*x^2 + B*x + C = (x + 1)*(x - 3)*(x - 4)) → 
  A + B + C = 11 := by
sorry

end NUMINAMATH_CALUDE_asymptotes_sum_l1056_105696


namespace NUMINAMATH_CALUDE_pyramid_volume_change_specific_pyramid_volume_l1056_105621

/-- Given a pyramid with a triangular base and initial volume V, 
    if the base height is doubled, base dimensions are tripled, 
    and the pyramid's height is increased by 40%, 
    then the new volume is 8.4 * V. -/
theorem pyramid_volume_change (V : ℝ) : 
  V > 0 → 
  (2 * 3 * 3 * 1.4) * V = 8.4 * V :=
by sorry

/-- The new volume of the specific pyramid is 604.8 cubic inches. -/
theorem specific_pyramid_volume : 
  (8.4 : ℝ) * 72 = 604.8 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_change_specific_pyramid_volume_l1056_105621


namespace NUMINAMATH_CALUDE_average_and_square_multiple_l1056_105647

theorem average_and_square_multiple (n : ℝ) (m : ℝ) (h1 : n ≠ 0) (h2 : n = 9) 
  (h3 : (n + n^2) / 2 = m * n) : m = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_and_square_multiple_l1056_105647


namespace NUMINAMATH_CALUDE_stratified_sampling_most_suitable_l1056_105682

/-- Represents the age groups in the population -/
inductive AgeGroup
  | Elderly
  | MiddleAged
  | Young

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | Other

/-- Represents the population structure -/
structure Population where
  total : Nat
  elderly : Nat
  middleAged : Nat
  young : Nat

/-- Determines the most suitable sampling method given a population and sample size -/
def mostSuitableSamplingMethod (pop : Population) (sampleSize : Nat) : SamplingMethod :=
  sorry

/-- The theorem stating that stratified sampling is the most suitable method for the given population and sample size -/
theorem stratified_sampling_most_suitable :
  let pop : Population := { total := 163, elderly := 28, middleAged := 54, young := 81 }
  let sampleSize : Nat := 36
  mostSuitableSamplingMethod pop sampleSize = SamplingMethod.Stratified :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_suitable_l1056_105682


namespace NUMINAMATH_CALUDE_total_monthly_cost_l1056_105679

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ := d.length * d.width * d.height

/-- Represents the storage details -/
structure StorageDetails where
  boxDim : BoxDimensions
  totalVolume : ℝ
  costPerBox : ℝ

/-- Theorem stating that the total monthly cost for record storage is $480 -/
theorem total_monthly_cost (s : StorageDetails)
  (h1 : s.boxDim = ⟨15, 12, 10⟩)
  (h2 : s.totalVolume = 1080000)
  (h3 : s.costPerBox = 0.8) :
  (s.totalVolume / boxVolume s.boxDim) * s.costPerBox = 480 := by
  sorry


end NUMINAMATH_CALUDE_total_monthly_cost_l1056_105679


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1056_105697

def M : Set ℤ := {0, 1, 2, -1}
def N : Set ℤ := {0, 1, 2, 3}

theorem intersection_of_M_and_N : M ∩ N = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1056_105697


namespace NUMINAMATH_CALUDE_outfits_count_l1056_105656

/-- The number of different outfits that can be made -/
def num_outfits (num_shirts : ℕ) (num_ties : ℕ) (num_pants : ℕ) (num_belts : ℕ) : ℕ :=
  num_shirts * num_pants * (num_ties + 1) * (num_belts + 1)

/-- Theorem stating that the number of outfits is 360 given the specific conditions -/
theorem outfits_count :
  num_outfits 5 5 4 2 = 360 :=
by sorry

end NUMINAMATH_CALUDE_outfits_count_l1056_105656


namespace NUMINAMATH_CALUDE_prob_first_class_is_072_l1056_105675

/-- Represents a batch of products -/
structure ProductBatch where
  defectiveRate : ℝ
  firstClassRateAmongQualified : ℝ

/-- Calculates the probability of selecting a first-class product from a batch -/
def probabilityFirstClass (batch : ProductBatch) : ℝ :=
  (1 - batch.defectiveRate) * batch.firstClassRateAmongQualified

/-- Theorem: The probability of selecting a first-class product from the given batch is 0.72 -/
theorem prob_first_class_is_072 (batch : ProductBatch) 
    (h1 : batch.defectiveRate = 0.04)
    (h2 : batch.firstClassRateAmongQualified = 0.75) : 
    probabilityFirstClass batch = 0.72 := by
  sorry

#eval probabilityFirstClass { defectiveRate := 0.04, firstClassRateAmongQualified := 0.75 }

end NUMINAMATH_CALUDE_prob_first_class_is_072_l1056_105675


namespace NUMINAMATH_CALUDE_complex_multiplication_l1056_105687

theorem complex_multiplication :
  (1 + Complex.I) * (2 - Complex.I) = 3 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1056_105687


namespace NUMINAMATH_CALUDE_f_expression_range_f_transformed_is_l1056_105638

/-- A quadratic function satisfying certain conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The property that f(0) = 0 -/
axiom f_zero : f 0 = 0

/-- The property that f(x+1) = f(x) + x + 1 for all x -/
axiom f_next (x : ℝ) : f (x + 1) = f x + x + 1

/-- f is a quadratic function -/
axiom f_quadratic : ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c

/-- Theorem: f(x) = (1/2)x^2 + (1/2)x -/
theorem f_expression : ∀ x, f x = (1/2) * x^2 + (1/2) * x := sorry

/-- The range of y = f(x^2 - 2) -/
def range_f_transformed : Set ℝ := {y | ∃ x, y = f (x^2 - 2)}

/-- Theorem: The range of y = f(x^2 - 2) is [-1/8, +∞) -/
theorem range_f_transformed_is : range_f_transformed = {y | y ≥ -1/8} := sorry

end NUMINAMATH_CALUDE_f_expression_range_f_transformed_is_l1056_105638


namespace NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_3_or_neg_1_l1056_105617

-- Define the lines
def line1 (m : ℝ) (x y : ℝ) : Prop := x + m * y + 7 = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := (m - 2) * x + 3 * y + 2 * m = 0

-- Define parallel lines
def parallel (m : ℝ) : Prop := ∀ x y, line1 m x y ↔ ∃ k, line2 m (x + k) (y + k)

-- Theorem statement
theorem lines_parallel_iff_m_eq_3_or_neg_1 :
  ∀ m : ℝ, parallel m ↔ m = 3 ∨ m = -1 := by sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_3_or_neg_1_l1056_105617


namespace NUMINAMATH_CALUDE_min_stamps_for_40_cents_l1056_105627

theorem min_stamps_for_40_cents :
  let stamp_values : List Nat := [5, 7]
  let target_value : Nat := 40
  ∃ (c f : Nat),
    c * stamp_values[0]! + f * stamp_values[1]! = target_value ∧
    ∀ (c' f' : Nat),
      c' * stamp_values[0]! + f' * stamp_values[1]! = target_value →
      c + f ≤ c' + f' ∧
    c + f = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_stamps_for_40_cents_l1056_105627


namespace NUMINAMATH_CALUDE_john_light_bulbs_l1056_105626

/-- Calculates the number of light bulbs left after using some and giving away half of the remainder -/
def lightBulbsLeft (initial : ℕ) (used : ℕ) : ℕ :=
  let remaining := initial - used
  remaining - remaining / 2

/-- Proves that starting with 40 light bulbs, using 16, and giving away half of the remainder leaves 12 bulbs -/
theorem john_light_bulbs : lightBulbsLeft 40 16 = 12 := by
  sorry

end NUMINAMATH_CALUDE_john_light_bulbs_l1056_105626


namespace NUMINAMATH_CALUDE_chromium_percentage_calculation_l1056_105695

/-- The percentage of chromium in the first alloy -/
def chromium_percentage_first : ℝ := 10

/-- The percentage of chromium in the second alloy -/
def chromium_percentage_second : ℝ := 6

/-- The weight of the first alloy in kg -/
def weight_first : ℝ := 15

/-- The weight of the second alloy in kg -/
def weight_second : ℝ := 35

/-- The percentage of chromium in the resulting alloy -/
def chromium_percentage_result : ℝ := 7.2

theorem chromium_percentage_calculation :
  (chromium_percentage_first * weight_first + chromium_percentage_second * weight_second) / (weight_first + weight_second) = chromium_percentage_result :=
by sorry

end NUMINAMATH_CALUDE_chromium_percentage_calculation_l1056_105695


namespace NUMINAMATH_CALUDE_one_tails_after_flips_l1056_105678

/-- Represents the state of a coin (heads or tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents a circular arrangement of coins -/
def CoinCircle (n : ℕ) := Fin (2*n+1) → CoinState

/-- The initial state of the coin circle where all coins show heads -/
def initialState (n : ℕ) : CoinCircle n :=
  λ _ => CoinState.Heads

/-- The position of the k-th flip in the circle -/
def flipPosition (n k : ℕ) : Fin (2*n+1) :=
  ⟨k * (k + 1) / 2, sorry⟩

/-- Applies a single flip to a coin state -/
def flipCoin : CoinState → CoinState
| CoinState.Heads => CoinState.Tails
| CoinState.Tails => CoinState.Heads

/-- Applies the flipping process to the coin circle -/
def applyFlips (n : ℕ) (state : CoinCircle n) : CoinCircle n :=
  sorry

/-- Counts the number of tails in the final state -/
def countTails (n : ℕ) (state : CoinCircle n) : ℕ :=
  sorry

/-- The main theorem stating that exactly one coin shows tails after the process -/
theorem one_tails_after_flips (n : ℕ) :
  countTails n (applyFlips n (initialState n)) = 1 :=
sorry

end NUMINAMATH_CALUDE_one_tails_after_flips_l1056_105678


namespace NUMINAMATH_CALUDE_waiter_customers_l1056_105653

/-- The number of customers who didn't leave a tip -/
def no_tip : ℕ := 34

/-- The number of customers who left a tip -/
def left_tip : ℕ := 15

/-- The number of customers added during the lunch rush -/
def added_customers : ℕ := 20

/-- The number of customers before the lunch rush -/
def customers_before : ℕ := 29

theorem waiter_customers :
  customers_before = (no_tip + left_tip) - added_customers :=
by sorry

end NUMINAMATH_CALUDE_waiter_customers_l1056_105653


namespace NUMINAMATH_CALUDE_student_calculation_l1056_105634

theorem student_calculation (x : ℕ) (h : x = 120) : 2 * x - 138 = 102 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_l1056_105634


namespace NUMINAMATH_CALUDE_complex_quadratic_roots_l1056_105641

theorem complex_quadratic_roots : ∃ (z₁ z₂ : ℂ), 
  z₁ = Complex.I * 2 ∧ 
  z₂ = -2 - Complex.I * 2 ∧ 
  z₁^2 + 2*z₁ = -3 + Complex.I * 4 ∧ 
  z₂^2 + 2*z₂ = -3 + Complex.I * 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_quadratic_roots_l1056_105641


namespace NUMINAMATH_CALUDE_rider_distances_l1056_105649

/-- The possible distances between two riders after one hour, given their initial distance and speeds -/
theorem rider_distances (initial_distance : ℝ) (speed_athos : ℝ) (speed_aramis : ℝ) :
  initial_distance = 20 ∧ speed_athos = 4 ∧ speed_aramis = 5 →
  ∃ (d₁ d₂ d₃ d₄ : ℝ),
    d₁ = 11 ∧ d₂ = 29 ∧ d₃ = 19 ∧ d₄ = 21 ∧
    ({d₁, d₂, d₃, d₄} : Set ℝ) = {
      initial_distance - (speed_athos + speed_aramis),
      initial_distance + (speed_athos + speed_aramis),
      initial_distance - (speed_aramis - speed_athos),
      initial_distance + (speed_aramis - speed_athos)
    } := by sorry

end NUMINAMATH_CALUDE_rider_distances_l1056_105649


namespace NUMINAMATH_CALUDE_parallel_vectors_trig_identity_l1056_105667

/-- Given two parallel vectors a and b, prove that cos(2α) + sin(2α) = -7/5 -/
theorem parallel_vectors_trig_identity (α : ℝ) :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (Real.sin α - Real.cos α, Real.sin α + Real.cos α)
  (∃ (k : ℝ), a.1 * b.2 = k * a.2 * b.1) →  -- parallel vectors condition
  Real.cos (2 * α) + Real.sin (2 * α) = -7/5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_trig_identity_l1056_105667


namespace NUMINAMATH_CALUDE_senate_subcommittee_count_l1056_105624

theorem senate_subcommittee_count :
  let total_republicans : ℕ := 10
  let total_democrats : ℕ := 8
  let subcommittee_republicans : ℕ := 4
  let subcommittee_democrats : ℕ := 3
  (Nat.choose total_republicans subcommittee_republicans) *
  (Nat.choose total_democrats subcommittee_democrats) = 11760 := by
  sorry

end NUMINAMATH_CALUDE_senate_subcommittee_count_l1056_105624


namespace NUMINAMATH_CALUDE_mindys_tax_rate_mindys_tax_rate_is_15_percent_l1056_105663

/-- Calculates Mindy's tax rate given the conditions of the problem -/
theorem mindys_tax_rate (morks_income : ℝ) (morks_tax_rate : ℝ) (mindys_income_multiplier : ℝ) (combined_tax_rate : ℝ) : ℝ :=
  let mindys_income := mindys_income_multiplier * morks_income
  let total_income := morks_income + mindys_income
  let mindys_tax_rate := (combined_tax_rate * total_income - morks_tax_rate * morks_income) / mindys_income
  mindys_tax_rate

/-- Proves that Mindy's tax rate is 15% given the problem conditions -/
theorem mindys_tax_rate_is_15_percent :
  mindys_tax_rate 1 0.45 4 0.21 = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_mindys_tax_rate_mindys_tax_rate_is_15_percent_l1056_105663


namespace NUMINAMATH_CALUDE_student_calculation_error_l1056_105640

def correct_calculation : ℚ := (3/4 * 16 - 7/8 * 8) / (3/10 - 1/8)

def incorrect_calculation : ℚ := (3/4 * 16 - 7/8 * 8) * (3/5)

def percentage_error (correct incorrect : ℚ) : ℚ :=
  abs (correct - incorrect) / correct * 100

theorem student_calculation_error :
  abs (percentage_error correct_calculation incorrect_calculation - 89.47) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_error_l1056_105640


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l1056_105692

theorem necessary_not_sufficient (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → (a / b + b / a ≥ 2) → (a^2 + b^2 ≥ 2*a*b)) ∧
  (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ (a^2 + b^2 ≥ 2*a*b) ∧ (a / b + b / a < 2)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l1056_105692


namespace NUMINAMATH_CALUDE_simplify_complex_expression_l1056_105674

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem simplify_complex_expression :
  3 * (4 - 2*i) + 2*i * (3 + 2*i) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_expression_l1056_105674


namespace NUMINAMATH_CALUDE_isabellas_hair_growth_l1056_105654

/-- Given Isabella's initial and final hair lengths, prove the amount of hair growth. -/
theorem isabellas_hair_growth 
  (initial_length : ℝ) 
  (final_length : ℝ) 
  (h1 : initial_length = 18) 
  (h2 : final_length = 24) : 
  final_length - initial_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_hair_growth_l1056_105654


namespace NUMINAMATH_CALUDE_additional_hour_rate_is_ten_l1056_105625

/-- Represents the rental cost structure for a power tool -/
structure RentalCost where
  firstHourRate : ℝ
  additionalHourRate : ℝ
  totalHours : ℕ
  totalCost : ℝ

/-- Theorem stating that given the rental conditions, the additional hour rate is $10 -/
theorem additional_hour_rate_is_ten
  (rental : RentalCost)
  (h1 : rental.firstHourRate = 25)
  (h2 : rental.totalHours = 11)
  (h3 : rental.totalCost = 125)
  : rental.additionalHourRate = 10 := by
  sorry

#check additional_hour_rate_is_ten

end NUMINAMATH_CALUDE_additional_hour_rate_is_ten_l1056_105625


namespace NUMINAMATH_CALUDE_mad_hatter_march_hare_meeting_time_difference_l1056_105658

/-- Represents a clock with a specific rate of time change per hour -/
structure Clock where
  rate : ℚ

/-- Calculates the actual time passed for a given clock time -/
def actual_time (c : Clock) (clock_time : ℚ) : ℚ :=
  clock_time * c.rate

theorem mad_hatter_march_hare_meeting_time_difference : 
  let mad_hatter_clock : Clock := ⟨60 / 75⟩
  let march_hare_clock : Clock := ⟨60 / 50⟩
  let meeting_time : ℚ := 5

  actual_time march_hare_clock meeting_time - actual_time mad_hatter_clock meeting_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_mad_hatter_march_hare_meeting_time_difference_l1056_105658


namespace NUMINAMATH_CALUDE_peter_bought_five_large_glasses_l1056_105666

/-- Represents the purchase of glasses by Peter -/
structure GlassesPurchase where
  small_cost : ℕ             -- Cost of a small glass
  large_cost : ℕ             -- Cost of a large glass
  total_money : ℕ            -- Total money Peter has
  small_bought : ℕ           -- Number of small glasses bought
  change : ℕ                 -- Money left as change

/-- Calculates the number of large glasses Peter bought -/
def large_glasses_bought (purchase : GlassesPurchase) : ℕ :=
  (purchase.total_money - purchase.small_cost * purchase.small_bought - purchase.change) / purchase.large_cost

/-- Theorem stating that Peter bought 5 large glasses -/
theorem peter_bought_five_large_glasses :
  ∀ (purchase : GlassesPurchase),
    purchase.small_cost = 3 →
    purchase.large_cost = 5 →
    purchase.total_money = 50 →
    purchase.small_bought = 8 →
    purchase.change = 1 →
    large_glasses_bought purchase = 5 := by
  sorry


end NUMINAMATH_CALUDE_peter_bought_five_large_glasses_l1056_105666


namespace NUMINAMATH_CALUDE_haley_shirts_l1056_105652

/-- The number of shirts Haley bought -/
def shirts_bought : ℕ := 11

/-- The number of shirts Haley returned -/
def shirts_returned : ℕ := 6

/-- The number of shirts Haley ended up with -/
def shirts_remaining : ℕ := shirts_bought - shirts_returned

theorem haley_shirts : shirts_remaining = 5 := by
  sorry

end NUMINAMATH_CALUDE_haley_shirts_l1056_105652


namespace NUMINAMATH_CALUDE_blueberry_muffin_probability_l1056_105670

theorem blueberry_muffin_probability :
  let n : ℕ := 7
  let k : ℕ := 5
  let p : ℚ := 3/4
  let q : ℚ := 1 - p
  Nat.choose n k * p^k * q^(n-k) = 5103/16384 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_muffin_probability_l1056_105670


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1056_105657

-- Define the given line
def given_line (x y : ℝ) : Prop := x - 2*y + 3 = 0

-- Define the point through which the perpendicular line passes
def point : ℝ × ℝ := (-1, 3)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 2*x + y - 1 = 0

-- Theorem statement
theorem perpendicular_line_equation :
  (∀ x y : ℝ, perpendicular_line x y ↔ 
    (∃ m b : ℝ, y = m*x + b ∧ 
      (perpendicular_line point.1 point.2) ∧
      (m * (1/2) = -1))) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1056_105657


namespace NUMINAMATH_CALUDE_cake_cutting_theorem_l1056_105607

/-- Represents a rectangular cake -/
structure Cake where
  length : ℕ
  width : ℕ
  pieces : ℕ

/-- The maximum number of pieces obtainable with one straight cut -/
def max_pieces_one_cut (c : Cake) : ℕ := sorry

/-- The minimum number of cuts required to ensure each piece is cut -/
def min_cuts_all_pieces (c : Cake) : ℕ := sorry

/-- Theorem for the cake cutting problem -/
theorem cake_cutting_theorem (c : Cake) 
  (h1 : c.length = 5 ∧ c.width = 2) 
  (h2 : c.pieces = 10) : 
  max_pieces_one_cut c = 16 ∧ min_cuts_all_pieces c = 2 := by sorry

end NUMINAMATH_CALUDE_cake_cutting_theorem_l1056_105607


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l1056_105686

theorem fraction_equals_zero (x : ℝ) : (x - 2) / (x + 3) = 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l1056_105686


namespace NUMINAMATH_CALUDE_dog_to_cats_ratio_is_two_to_one_l1056_105605

/-- The weight of Christine's first cat in pounds -/
def cat1_weight : ℕ := 7

/-- The weight of Christine's second cat in pounds -/
def cat2_weight : ℕ := 10

/-- The combined weight of Christine's cats in pounds -/
def cats_combined_weight : ℕ := cat1_weight + cat2_weight

/-- The weight of Christine's dog in pounds -/
def dog_weight : ℕ := 34

/-- The ratio of the dog's weight to the combined weight of the cats -/
def dog_to_cats_ratio : ℚ := dog_weight / cats_combined_weight

theorem dog_to_cats_ratio_is_two_to_one :
  dog_to_cats_ratio = 2 := by sorry

end NUMINAMATH_CALUDE_dog_to_cats_ratio_is_two_to_one_l1056_105605


namespace NUMINAMATH_CALUDE_probability_other_side_red_l1056_105629

structure Card where
  side1 : String
  side2 : String

def Box : List Card := [
  {side1 := "black", side2 := "black"},
  {side1 := "black", side2 := "black"},
  {side1 := "black", side2 := "black"},
  {side1 := "black", side2 := "black"},
  {side1 := "black", side2 := "red"},
  {side1 := "black", side2 := "red"},
  {side1 := "black", side2 := "red"},
  {side1 := "red", side2 := "red"},
  {side1 := "red", side2 := "red"},
  {side1 := "blue", side2 := "blue"}
]

def isRed (s : String) : Bool := s == "red"

def countRedSides (cards : List Card) : Nat :=
  cards.foldl (fun acc card => acc + (if isRed card.side1 then 1 else 0) + (if isRed card.side2 then 1 else 0)) 0

def countBothRedCards (cards : List Card) : Nat :=
  cards.foldl (fun acc card => acc + (if isRed card.side1 && isRed card.side2 then 1 else 0)) 0

theorem probability_other_side_red (box : List Card := Box) :
  let totalRedSides := countRedSides box
  let bothRedCards := countBothRedCards box
  (2 * bothRedCards : Rat) / totalRedSides = 4 / 7 := by sorry

end NUMINAMATH_CALUDE_probability_other_side_red_l1056_105629


namespace NUMINAMATH_CALUDE_toms_age_ratio_l1056_105608

/-- Tom's age problem -/
theorem toms_age_ratio (T N : ℝ) : T > 0 → N > 0 → 
  (T = T - 4*N + T - 4*N + T - 4*N + T - 4*N) → -- Sum of children's ages
  (T - N = 3 * (T - 4*N)) →                     -- Relation N years ago
  T / N = 11 / 2 := by
sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l1056_105608


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1056_105619

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15 → 
  ∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 → 
  (x : ℤ) + y ≤ (a : ℤ) + b :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1056_105619


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1056_105602

theorem least_subtraction_for_divisibility : 
  ∃! x : ℕ, x ≤ 29 ∧ (10154 - x) % 30 = 0 ∧ ∀ y : ℕ, y < x → (10154 - y) % 30 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1056_105602


namespace NUMINAMATH_CALUDE_min_h_10_l1056_105645

/-- A function is expansive if f(x) + f(y) > x^2 + y^2 for all positive integers x and y -/
def Expansive (f : ℕ+ → ℤ) : Prop :=
  ∀ x y : ℕ+, f x + f y > (x.val : ℤ)^2 + (y.val : ℤ)^2

/-- The sum of h(1) to h(15) -/
def SumH (h : ℕ+ → ℤ) : ℤ :=
  (Finset.range 15).sum (λ i => h ⟨i + 1, by linarith⟩)

/-- The theorem statement -/
theorem min_h_10 (h : ℕ+ → ℤ) (hExpansive : Expansive h) (hMinSum : ∀ g : ℕ+ → ℤ, Expansive g → SumH g ≥ SumH h) :
  h ⟨10, by norm_num⟩ ≥ 125 := by
  sorry

end NUMINAMATH_CALUDE_min_h_10_l1056_105645


namespace NUMINAMATH_CALUDE_cost_effectiveness_l1056_105610

/-- Represents the cost-effective choice between two malls --/
inductive Choice
  | MallA
  | MallB
  | Either

/-- Calculates the price per unit for Mall A based on the number of items --/
def mall_a_price (n : ℕ) : ℚ :=
  if n * 4 ≤ 40 then 80 - n * 4
  else 40

/-- Calculates the price per unit for Mall B --/
def mall_b_price : ℚ := 80 * (1 - 0.3)

/-- Determines the cost-effective choice based on the number of employees --/
def cost_effective_choice (num_employees : ℕ) : Choice :=
  if num_employees < 6 then Choice.MallB
  else if num_employees = 6 then Choice.Either
  else Choice.MallA

theorem cost_effectiveness 
  (num_employees : ℕ) : 
  (cost_effective_choice num_employees = Choice.MallB ↔ num_employees < 6) ∧
  (cost_effective_choice num_employees = Choice.Either ↔ num_employees = 6) ∧
  (cost_effective_choice num_employees = Choice.MallA ↔ num_employees > 6) :=
sorry

end NUMINAMATH_CALUDE_cost_effectiveness_l1056_105610


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1056_105614

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The sum of specific terms in the sequence equals 200 -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 3 + a 5 + a 7 + a 9 + a 11 = 200

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : SumCondition a) : 
  4 * a 5 - 2 * a 3 = 80 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1056_105614


namespace NUMINAMATH_CALUDE_percentage_failed_hindi_l1056_105698

theorem percentage_failed_hindi (failed_english : ℝ) (failed_both : ℝ) (passed_both : ℝ) :
  failed_english = 44 →
  failed_both = 22 →
  passed_both = 44 →
  ∃ failed_hindi : ℝ, failed_hindi = 34 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_failed_hindi_l1056_105698


namespace NUMINAMATH_CALUDE_chad_bbq_ice_cost_l1056_105691

/-- The cost of ice for Chad's BBQ -/
def bbq_ice_cost (total_people : ℕ) (ice_per_person : ℕ) (package_size : ℕ) (cost_per_package : ℚ) : ℚ :=
  let total_ice := total_people * ice_per_person
  let packages_needed := (total_ice + package_size - 1) / package_size
  packages_needed * cost_per_package

/-- Theorem: The cost of ice for Chad's BBQ is $27 -/
theorem chad_bbq_ice_cost :
  bbq_ice_cost 20 3 10 (4.5 : ℚ) = 27 := by
  sorry

end NUMINAMATH_CALUDE_chad_bbq_ice_cost_l1056_105691


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1056_105633

/-- The lateral surface area of a cone with base radius 3 cm and lateral surface forming a semicircle when unfolded -/
theorem cone_lateral_surface_area : 
  ∀ (r l : ℝ), 
    r = 3 → -- base radius is 3 cm
    l = 6 → -- slant height is 6 cm (derived from the semicircle condition)
    (1/2 : ℝ) * Real.pi * l^2 = 18 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1056_105633


namespace NUMINAMATH_CALUDE_minimum_at_two_l1056_105613

/-- The function f(x) with parameter t -/
def f (t : ℝ) (x : ℝ) : ℝ := x^3 - 2*t*x^2 + t^2*x

/-- The derivative of f(x) with respect to x -/
def f' (t : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*t*x + t^2

theorem minimum_at_two (t : ℝ) : 
  (∀ x : ℝ, f t x ≥ f t 2) ↔ t = 2 := by sorry

end NUMINAMATH_CALUDE_minimum_at_two_l1056_105613


namespace NUMINAMATH_CALUDE_iguana_feed_cost_l1056_105620

/-- The monthly cost to feed each iguana, given the number of pets, 
    the cost to feed geckos and snakes, and the total annual cost for all pets. -/
theorem iguana_feed_cost 
  (num_geckos num_iguanas num_snakes : ℕ)
  (gecko_cost snake_cost : ℚ)
  (total_annual_cost : ℚ)
  (h1 : num_geckos = 3)
  (h2 : num_iguanas = 2)
  (h3 : num_snakes = 4)
  (h4 : gecko_cost = 15)
  (h5 : snake_cost = 10)
  (h6 : total_annual_cost = 1140)
  : ∃ (iguana_cost : ℚ), 
    iguana_cost = 5 ∧
    (num_geckos : ℚ) * gecko_cost + 
    (num_iguanas : ℚ) * iguana_cost + 
    (num_snakes : ℚ) * snake_cost = 
    total_annual_cost / 12 :=
sorry

end NUMINAMATH_CALUDE_iguana_feed_cost_l1056_105620


namespace NUMINAMATH_CALUDE_harvest_season_duration_l1056_105648

theorem harvest_season_duration (regular_earnings overtime_earnings total_earnings : ℕ) 
  (h1 : regular_earnings = 28)
  (h2 : overtime_earnings = 939)
  (h3 : total_earnings = 1054997) : 
  total_earnings / (regular_earnings + overtime_earnings) = 1091 := by
  sorry

end NUMINAMATH_CALUDE_harvest_season_duration_l1056_105648


namespace NUMINAMATH_CALUDE_triangle_altitude_and_median_equations_l1056_105673

/-- Triangle ABC with vertices A(4,0), B(6,7), and C(0,3) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The given triangle -/
def triangle : Triangle := { A := (4, 0), B := (6, 7), C := (0, 3) }

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The equation of the altitude from B to AC -/
def altitudeEquation : LineEquation := { a := 3, b := 2, c := -12 }

/-- The equation of the median from B to AC -/
def medianEquation : LineEquation := { a := 5, b := 1, c := -20 }

theorem triangle_altitude_and_median_equations :
  let t := triangle
  let alt := altitudeEquation
  let med := medianEquation
  (∀ x y : ℝ, alt.a * x + alt.b * y + alt.c = 0 ↔ 
    (x - t.B.1) * (t.A.1 - t.C.1) + (y - t.B.2) * (t.A.2 - t.C.2) = 0) ∧
  (∀ x y : ℝ, med.a * x + med.b * y + med.c = 0 ↔ 
    2 * (x - t.B.1) = t.A.1 - t.C.1 ∧ 2 * (y - t.B.2) = t.A.2 - t.C.2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_and_median_equations_l1056_105673


namespace NUMINAMATH_CALUDE_mabel_shark_count_l1056_105688

-- Define the percentage of sharks and other fish
def shark_percentage : ℚ := 25 / 100
def other_fish_percentage : ℚ := 75 / 100

-- Define the number of fish counted on day one
def day_one_count : ℕ := 15

-- Define the multiplier for day two
def day_two_multiplier : ℕ := 3

-- Theorem statement
theorem mabel_shark_count :
  let day_two_count := day_one_count * day_two_multiplier
  let total_fish := day_one_count + day_two_count
  let shark_count := (total_fish : ℚ) * shark_percentage
  shark_count = 15 := by sorry

end NUMINAMATH_CALUDE_mabel_shark_count_l1056_105688


namespace NUMINAMATH_CALUDE_systematic_sampling_missiles_l1056_105664

/-- Represents a systematic sampling sequence -/
def SystematicSample (total : ℕ) (sampleSize : ℕ) (start : ℕ) : List ℕ :=
  List.range sampleSize |>.map (fun i => start + i * (total / sampleSize))

/-- The problem statement -/
theorem systematic_sampling_missiles :
  let total := 50
  let sampleSize := 5
  let start := 3
  SystematicSample total sampleSize start = [3, 13, 23, 33, 43] := by
  sorry

#eval SystematicSample 50 5 3

end NUMINAMATH_CALUDE_systematic_sampling_missiles_l1056_105664


namespace NUMINAMATH_CALUDE_race_distance_l1056_105644

theorem race_distance (d : ℝ) (vA vB vC : ℝ) 
  (h1 : d / vA = (d - 20) / vB)
  (h2 : d / vB = (d - 10) / vC)
  (h3 : d / vA = (d - 28) / vC)
  (h4 : d > 0) (h5 : vA > 0) (h6 : vB > 0) (h7 : vC > 0) : d = 100 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_l1056_105644


namespace NUMINAMATH_CALUDE_x_zero_value_l1056_105646

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem x_zero_value (x₀ : ℝ) (h : (deriv f) x₀ = 3) :
  x₀ = 1 ∨ x₀ = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_zero_value_l1056_105646


namespace NUMINAMATH_CALUDE_real_part_of_complex_fraction_l1056_105643

theorem real_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  (2 * i / (1 + i)).re = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_fraction_l1056_105643


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_square_plus_one_l1056_105603

theorem negation_of_forall_positive_square_plus_one (P : Real → Prop) : 
  (¬ ∀ x > 1, x^2 + 1 ≥ 0) ↔ (∃ x > 1, x^2 + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_square_plus_one_l1056_105603


namespace NUMINAMATH_CALUDE_original_fraction_value_l1056_105693

theorem original_fraction_value (x : ℚ) : 
  (x + 1) / (x + 8) = 11 / 17 → x / (x + 7) = 71 / 113 := by
  sorry

end NUMINAMATH_CALUDE_original_fraction_value_l1056_105693


namespace NUMINAMATH_CALUDE_wendy_bouquets_l1056_105636

/-- Given the initial number of flowers, flowers per bouquet, and number of wilted flowers,
    calculate the number of bouquets that can be made. -/
def bouquets_remaining (initial_flowers : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) : ℕ :=
  (initial_flowers - wilted_flowers) / flowers_per_bouquet

/-- Prove that Wendy can make 2 bouquets with the remaining flowers. -/
theorem wendy_bouquets :
  bouquets_remaining 45 5 35 = 2 := by
  sorry

end NUMINAMATH_CALUDE_wendy_bouquets_l1056_105636


namespace NUMINAMATH_CALUDE_no_solution_to_equation_l1056_105659

theorem no_solution_to_equation :
  ¬ ∃ (x : ℝ), x ≠ 2 ∧ (1 / (x - 2) = (1 - x) / (2 - x) - 3) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_equation_l1056_105659


namespace NUMINAMATH_CALUDE_counterexamples_exist_l1056_105611

theorem counterexamples_exist : ∃ (a b c : ℝ),
  -- Statement A is not always true
  (a * b ≠ 0 ∧ a < b ∧ (1 / a) ≤ (1 / b)) ∧
  -- Statement C is not always true
  (a > b ∧ b > 0 ∧ ((b + 1) / (a + 1)) ≥ (b / a)) ∧
  -- Statement D is not always true
  (c < b ∧ b < a ∧ a * c < 0 ∧ c * b^2 ≥ a * b^2) :=
sorry

end NUMINAMATH_CALUDE_counterexamples_exist_l1056_105611


namespace NUMINAMATH_CALUDE_area_of_graph_l1056_105662

/-- The area enclosed by the graph of |x| + |3y| = 12 -/
def rhombus_area : ℝ := 384

/-- The equation defining the graph -/
def graph_equation (x y : ℝ) : Prop := |x| + |3 * y| = 12

theorem area_of_graph :
  ∃ (x_intercept y_intercept : ℝ),
    x_intercept > 0 ∧
    y_intercept > 0 ∧
    graph_equation x_intercept 0 ∧
    graph_equation 0 y_intercept ∧
    rhombus_area = 4 * (x_intercept * y_intercept) :=
sorry

end NUMINAMATH_CALUDE_area_of_graph_l1056_105662


namespace NUMINAMATH_CALUDE_bus_speeds_l1056_105672

theorem bus_speeds (distance : ℝ) (time_difference : ℝ) (speed_difference : ℝ)
  (h1 : distance = 48)
  (h2 : time_difference = 1/6)
  (h3 : speed_difference = 4) :
  ∃ (speed1 speed2 : ℝ),
    speed1 = 36 ∧
    speed2 = 32 ∧
    distance / speed1 + time_difference = distance / speed2 ∧
    speed1 = speed2 + speed_difference :=
by sorry

end NUMINAMATH_CALUDE_bus_speeds_l1056_105672


namespace NUMINAMATH_CALUDE_last_digit_of_77_in_binary_l1056_105655

theorem last_digit_of_77_in_binary (n : Nat) : n = 77 → n % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_77_in_binary_l1056_105655


namespace NUMINAMATH_CALUDE_gcd_987654_123456_l1056_105616

theorem gcd_987654_123456 : Nat.gcd 987654 123456 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_987654_123456_l1056_105616
