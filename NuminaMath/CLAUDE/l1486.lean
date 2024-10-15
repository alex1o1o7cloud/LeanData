import Mathlib

namespace NUMINAMATH_CALUDE_fixed_point_satisfies_line_fixed_point_unique_l1486_148601

/-- A line that passes through a fixed point for all values of m -/
def line (m x y : ℝ) : Prop :=
  (3*m + 4)*x + (5 - 2*m)*y + 7*m - 6 = 0

/-- The fixed point through which the line always passes -/
def fixed_point : ℝ × ℝ := (-1, 2)

/-- Theorem stating that the fixed point satisfies the line equation for all m -/
theorem fixed_point_satisfies_line :
  ∀ m : ℝ, line m (fixed_point.1) (fixed_point.2) :=
by sorry

/-- Theorem stating that the fixed point is unique -/
theorem fixed_point_unique :
  ∀ x y : ℝ, (∀ m : ℝ, line m x y) → (x, y) = fixed_point :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_satisfies_line_fixed_point_unique_l1486_148601


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l1486_148655

theorem bridge_length_calculation (train_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) :
  train_length = 250 →
  crossing_time = 25 →
  train_speed_kmh = 57.6 →
  ∃ (bridge_length : ℝ),
    bridge_length = 150 ∧
    bridge_length + train_length = (train_speed_kmh * 1000 / 3600) * crossing_time :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l1486_148655


namespace NUMINAMATH_CALUDE_greatest_power_of_200_dividing_100_factorial_l1486_148621

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem greatest_power_of_200_dividing_100_factorial :
  (∃ k : ℕ, 200^k ∣ factorial 100 ∧ ∀ m : ℕ, m > k → ¬(200^m ∣ factorial 100)) ∧
  (∀ k : ℕ, 200^k ∣ factorial 100 → k ≤ 12) ∧
  (200^12 ∣ factorial 100) :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_200_dividing_100_factorial_l1486_148621


namespace NUMINAMATH_CALUDE_fathers_age_l1486_148605

theorem fathers_age (man_age father_age : ℕ) : 
  man_age = (2 * father_age) / 5 →
  man_age + 8 = (father_age + 8) / 2 →
  father_age = 40 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_l1486_148605


namespace NUMINAMATH_CALUDE_coin_flip_problem_l1486_148650

theorem coin_flip_problem (p_heads : ℝ) (p_event : ℝ) (n : ℕ) :
  p_heads = 1/2 →
  p_event = 0.03125 →
  p_event = p_heads * (1 - p_heads)^4 →
  n = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_coin_flip_problem_l1486_148650


namespace NUMINAMATH_CALUDE_set_equality_implies_values_l1486_148689

theorem set_equality_implies_values (a b : ℝ) : 
  ({1, a, b} : Set ℝ) = {a, a^2, a*b} → a = -1 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_values_l1486_148689


namespace NUMINAMATH_CALUDE_tan_cos_expression_equals_negative_one_l1486_148683

theorem tan_cos_expression_equals_negative_one :
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_cos_expression_equals_negative_one_l1486_148683


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_two_to_ten_l1486_148629

theorem last_digit_of_one_over_two_to_ten (n : ℕ) : 
  n = 10 → (1 : ℚ) / (2^n : ℚ) * 10^n % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_two_to_ten_l1486_148629


namespace NUMINAMATH_CALUDE_expansion_theorem_l1486_148641

-- Define the sum of coefficients for (3x + √x)^n
def sumCoefficients (n : ℕ) : ℝ := 4^n

-- Define the sum of binomial coefficients
def sumBinomialCoefficients (n : ℕ) : ℝ := 2^n

-- Define the condition M - N = 240
def conditionSatisfied (n : ℕ) : Prop :=
  sumCoefficients n - sumBinomialCoefficients n = 240

-- Define the rational terms in the expansion
def rationalTerms (n : ℕ) : List (ℝ × ℕ) :=
  [(81, 4), (54, 3), (1, 2)]

theorem expansion_theorem :
  ∃ n : ℕ, conditionSatisfied n ∧ 
  n = 4 ∧
  rationalTerms n = [(81, 4), (54, 3), (1, 2)] :=
sorry

end NUMINAMATH_CALUDE_expansion_theorem_l1486_148641


namespace NUMINAMATH_CALUDE_oil_in_barrels_l1486_148624

theorem oil_in_barrels (barrel_a barrel_b : ℚ) : 
  barrel_a = 3/4 → 
  barrel_b = barrel_a + 1/10 → 
  barrel_a + barrel_b = 8/5 := by
sorry

end NUMINAMATH_CALUDE_oil_in_barrels_l1486_148624


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1486_148676

theorem cubic_equation_solution (a b y : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 25 * y^3) 
  (h3 : a - b = y) : 
  b = -(1 - Real.sqrt 33) / 2 * y ∨ b = -(1 + Real.sqrt 33) / 2 * y := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1486_148676


namespace NUMINAMATH_CALUDE_greatest_b_value_l1486_148625

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 8*x - 15 ≥ 0 → x ≤ 5) ∧ (-5^2 + 8*5 - 15 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_greatest_b_value_l1486_148625


namespace NUMINAMATH_CALUDE_leftover_fraction_l1486_148617

def fractions : List ℚ := [5/4, 17/6, -5/4, 10/7, 2/3, 14/8, -1/3, 5/3, -3/2]

def has_sum_5_2 (a b : ℚ) : Prop := a + b = 5/2
def has_diff_5_2 (a b : ℚ) : Prop := a - b = 5/2
def has_prod_5_2 (a b : ℚ) : Prop := a * b = 5/2
def has_quot_5_2 (a b : ℚ) : Prop := a / b = 5/2

def is_in_pair (x : ℚ) : Prop :=
  ∃ y ∈ fractions, x ≠ y ∧ (has_sum_5_2 x y ∨ has_diff_5_2 x y ∨ has_prod_5_2 x y ∨ has_quot_5_2 x y)

theorem leftover_fraction :
  ∀ x ∈ fractions, x ≠ -3/2 → is_in_pair x :=
sorry

end NUMINAMATH_CALUDE_leftover_fraction_l1486_148617


namespace NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l1486_148638

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The number of ways to distribute 5 balls into 3 boxes with one box always empty -/
theorem distribute_five_balls_three_boxes :
  distribute_balls 5 3 = 3 :=
sorry

end NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l1486_148638


namespace NUMINAMATH_CALUDE_no_valid_sequence_for_arrangement_D_l1486_148648

/-- Represents a cell in the 2x4 grid -/
inductive Cell
| topLeft | topMidLeft | topMidRight | topRight
| bottomLeft | bottomMidLeft | bottomMidRight | bottomRight

/-- Checks if two cells are adjacent (share a common vertex) -/
def adjacent (c1 c2 : Cell) : Prop :=
  match c1, c2 with
  | Cell.topLeft, Cell.topMidLeft | Cell.topLeft, Cell.bottomLeft | Cell.topLeft, Cell.bottomMidLeft => True
  | Cell.topMidLeft, Cell.topLeft | Cell.topMidLeft, Cell.topMidRight | Cell.topMidLeft, Cell.bottomLeft | Cell.topMidLeft, Cell.bottomMidLeft | Cell.topMidLeft, Cell.bottomMidRight => True
  | Cell.topMidRight, Cell.topMidLeft | Cell.topMidRight, Cell.topRight | Cell.topMidRight, Cell.bottomMidLeft | Cell.topMidRight, Cell.bottomMidRight | Cell.topMidRight, Cell.bottomRight => True
  | Cell.topRight, Cell.topMidRight | Cell.topRight, Cell.bottomMidRight | Cell.topRight, Cell.bottomRight => True
  | Cell.bottomLeft, Cell.topLeft | Cell.bottomLeft, Cell.topMidLeft | Cell.bottomLeft, Cell.bottomMidLeft => True
  | Cell.bottomMidLeft, Cell.topLeft | Cell.bottomMidLeft, Cell.topMidLeft | Cell.bottomMidLeft, Cell.topMidRight | Cell.bottomMidLeft, Cell.bottomLeft | Cell.bottomMidLeft, Cell.bottomMidRight => True
  | Cell.bottomMidRight, Cell.topMidLeft | Cell.bottomMidRight, Cell.topMidRight | Cell.bottomMidRight, Cell.topRight | Cell.bottomMidRight, Cell.bottomMidLeft | Cell.bottomMidRight, Cell.bottomRight => True
  | Cell.bottomRight, Cell.topMidRight | Cell.bottomRight, Cell.topRight | Cell.bottomRight, Cell.bottomMidRight => True
  | _, _ => False

/-- Represents a sequence of cell selections -/
def CellSequence := List Cell

/-- Checks if a cell sequence is valid according to the rules -/
def validSequence (seq : CellSequence) : Prop :=
  match seq with
  | [] => True
  | [_] => True
  | c1 :: c2 :: rest => adjacent c1 c2 ∧ validSequence (c2 :: rest)

/-- Represents the arrangement D -/
def arrangementD : List Cell :=
  [Cell.topLeft, Cell.topMidLeft, Cell.topMidRight, Cell.topRight,
   Cell.bottomLeft, Cell.bottomMidRight, Cell.bottomMidLeft, Cell.bottomRight]

/-- Theorem stating that no valid sequence can produce arrangement D -/
theorem no_valid_sequence_for_arrangement_D :
  ¬∃ (seq : CellSequence), validSequence seq ∧ seq.map (λ c => c) = arrangementD := by
  sorry


end NUMINAMATH_CALUDE_no_valid_sequence_for_arrangement_D_l1486_148648


namespace NUMINAMATH_CALUDE_expression_evaluation_l1486_148626

theorem expression_evaluation :
  let expr := 3 * 15 + 20 / 4 + 1
  let max_expr := 3 * (15 + 20 / 4 + 1)
  let min_expr := (3 * 15 + 20) / (4 + 1)
  (expr = 51) ∧ 
  (max_expr = 63) ∧ 
  (min_expr = 13) ∧
  (∀ x : ℤ, (∃ e : ℤ → ℤ, e expr = x) → (x ≤ max_expr ∧ x ≥ min_expr)) :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1486_148626


namespace NUMINAMATH_CALUDE_inequality_proof_l1486_148649

theorem inequality_proof (a b c : ℝ) : 
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|)) ≤ 1 + (1/3) * (a + b + c)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1486_148649


namespace NUMINAMATH_CALUDE_joyce_gave_three_oranges_l1486_148634

/-- The number of oranges Joyce gave to Clarence -/
def oranges_from_joyce (initial_oranges final_oranges : ℕ) : ℕ :=
  final_oranges - initial_oranges

theorem joyce_gave_three_oranges :
  oranges_from_joyce 5 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_joyce_gave_three_oranges_l1486_148634


namespace NUMINAMATH_CALUDE_test_score_result_l1486_148613

/-- Represents the score calculation for a test with specific conditions -/
def test_score (total_questions : ℕ) 
               (single_answer_questions : ℕ) 
               (multiple_answer_questions : ℕ) 
               (single_answer_marks : ℕ) 
               (multiple_answer_marks : ℕ) 
               (single_answer_penalty : ℕ) 
               (multiple_answer_penalty : ℕ) 
               (jose_wrong_single : ℕ) 
               (jose_wrong_multiple : ℕ) 
               (meghan_diff : ℕ) 
               (alisson_diff : ℕ) : ℕ := 
  sorry

theorem test_score_result : 
  test_score 70 50 20 2 4 1 2 10 5 30 50 = 280 :=
sorry

end NUMINAMATH_CALUDE_test_score_result_l1486_148613


namespace NUMINAMATH_CALUDE_cards_traded_count_l1486_148623

/-- The total number of cards traded between Padma and Robert -/
def total_cards_traded (padma_initial : ℕ) (robert_initial : ℕ) 
  (padma_first_trade : ℕ) (robert_first_trade : ℕ)
  (robert_second_trade : ℕ) (padma_second_trade : ℕ) : ℕ :=
  (padma_first_trade + robert_first_trade) + (robert_second_trade + padma_second_trade)

/-- Theorem stating the total number of cards traded between Padma and Robert -/
theorem cards_traded_count :
  total_cards_traded 75 88 2 10 8 15 = 35 := by
  sorry

end NUMINAMATH_CALUDE_cards_traded_count_l1486_148623


namespace NUMINAMATH_CALUDE_c_worked_four_days_l1486_148692

/-- Represents the number of days worked by person a -/
def days_a : ℕ := 6

/-- Represents the number of days worked by person b -/
def days_b : ℕ := 9

/-- Represents the daily wage of person c -/
def wage_c : ℕ := 125

/-- Represents the total earnings of all three people -/
def total_earnings : ℕ := 1850

/-- Represents the ratio of daily wages for a, b, and c -/
def wage_ratio : Fin 3 → ℕ
  | 0 => 3  -- a's ratio
  | 1 => 4  -- b's ratio
  | 2 => 5  -- c's ratio

/-- Calculates the daily wage for a given person based on c's wage and the ratio -/
def daily_wage (person : Fin 3) : ℕ :=
  wage_c * wage_ratio person / wage_ratio 2

/-- Theorem stating that person c worked for 4 days -/
theorem c_worked_four_days :
  ∃ (days_c : ℕ), 
    days_c * daily_wage 2 + 
    days_a * daily_wage 0 + 
    days_b * daily_wage 1 = total_earnings ∧
    days_c = 4 := by
  sorry

end NUMINAMATH_CALUDE_c_worked_four_days_l1486_148692


namespace NUMINAMATH_CALUDE_age_of_25th_student_l1486_148656

/-- The age of the 25th student in a class with specific age distributions -/
theorem age_of_25th_student (total_students : ℕ) (avg_age : ℝ) 
  (group1_count : ℕ) (group1_avg : ℝ) (group2_count : ℕ) (group2_avg : ℝ) :
  total_students = 25 →
  avg_age = 25 →
  group1_count = 10 →
  group1_avg = 22 →
  group2_count = 14 →
  group2_avg = 28 →
  (total_students * avg_age) - (group1_count * group1_avg + group2_count * group2_avg) = 13 :=
by sorry

end NUMINAMATH_CALUDE_age_of_25th_student_l1486_148656


namespace NUMINAMATH_CALUDE_quinary_444_equals_octal_174_l1486_148697

/-- Converts a quinary (base-5) number to decimal (base-10) --/
def quinary_to_decimal (q : ℕ) : ℕ := 
  4 * 5^2 + 4 * 5^1 + 4 * 5^0

/-- Converts a decimal (base-10) number to octal (base-8) --/
def decimal_to_octal (d : ℕ) : ℕ := 
  1 * 8^2 + 7 * 8^1 + 4 * 8^0

/-- Theorem stating that 444₅ in quinary is equal to 174₈ in octal --/
theorem quinary_444_equals_octal_174 : 
  quinary_to_decimal 444 = decimal_to_octal 174 := by
  sorry

end NUMINAMATH_CALUDE_quinary_444_equals_octal_174_l1486_148697


namespace NUMINAMATH_CALUDE_decreasing_function_implies_a_less_than_one_l1486_148627

/-- A function f: ℝ → ℝ is decreasing if for all x y, x < y implies f x > f y -/
def Decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The function f(x) = (a-1)x + 1 -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ (a - 1) * x + 1

theorem decreasing_function_implies_a_less_than_one (a : ℝ) :
  Decreasing (f a) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_implies_a_less_than_one_l1486_148627


namespace NUMINAMATH_CALUDE_special_pentagon_theorem_l1486_148608

/-- A pentagon with two right angles and three known angles -/
structure SpecialPentagon where
  -- The measures of the three known angles
  angle_P : ℝ
  angle_Q : ℝ
  angle_R : ℝ
  -- The measures of the two unknown angles
  angle_U : ℝ
  angle_V : ℝ
  -- Conditions
  angle_P_eq : angle_P = 42
  angle_Q_eq : angle_Q = 60
  angle_R_eq : angle_R = 38
  -- The pentagon has two right angles
  has_two_right_angles : True
  -- The sum of all interior angles of a pentagon is 540°
  sum_of_angles : angle_P + angle_Q + angle_R + angle_U + angle_V + 180 = 540

theorem special_pentagon_theorem (p : SpecialPentagon) : p.angle_U + p.angle_V = 40 := by
  sorry

end NUMINAMATH_CALUDE_special_pentagon_theorem_l1486_148608


namespace NUMINAMATH_CALUDE_megan_country_albums_l1486_148688

/-- The number of country albums Megan bought -/
def num_country_albums : ℕ := 2

/-- The number of pop albums Megan bought -/
def num_pop_albums : ℕ := 8

/-- The number of songs per album -/
def songs_per_album : ℕ := 7

/-- The total number of songs Megan bought -/
def total_songs : ℕ := 70

/-- Proof that Megan bought 2 country albums -/
theorem megan_country_albums :
  num_country_albums * songs_per_album + num_pop_albums * songs_per_album = total_songs :=
by sorry

end NUMINAMATH_CALUDE_megan_country_albums_l1486_148688


namespace NUMINAMATH_CALUDE_range_of_a_l1486_148652

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := a * x^3 - x^2 + 4*x + 3

-- State the theorem
theorem range_of_a : 
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-2 : ℝ) 1 → f x a ≥ 0) → 
  a ∈ Set.Icc (-6 : ℝ) (-2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1486_148652


namespace NUMINAMATH_CALUDE_largest_valid_number_sum_of_digits_l1486_148670

def is_valid_remainder (r : ℕ) (m : ℕ) : Prop :=
  r > 1 ∧ r < m

def form_geometric_progression (r1 r2 r3 : ℕ) : Prop :=
  (r2 * r2 = r1 * r3) ∧ r1 ≠ r2

def satisfies_conditions (n : ℕ) : Prop :=
  ∃ (r1 r2 r3 : ℕ),
    is_valid_remainder r1 9 ∧
    is_valid_remainder r2 10 ∧
    is_valid_remainder r3 11 ∧
    form_geometric_progression r1 r2 r3 ∧
    n % 9 = r1 ∧
    n % 10 = r2 ∧
    n % 11 = r3

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem largest_valid_number_sum_of_digits :
  ∃ (N : ℕ), N < 990 ∧ satisfies_conditions N ∧
  (∀ (m : ℕ), m < 990 → satisfies_conditions m → m ≤ N) ∧
  sum_of_digits N = 13 :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_sum_of_digits_l1486_148670


namespace NUMINAMATH_CALUDE_movie_tickets_difference_l1486_148651

theorem movie_tickets_difference (x y : ℕ) : 
  x + y = 30 →
  10 * x + 20 * y = 500 →
  y > x →
  y - x = 10 :=
by sorry

end NUMINAMATH_CALUDE_movie_tickets_difference_l1486_148651


namespace NUMINAMATH_CALUDE_cubic_expansion_sum_l1486_148687

theorem cubic_expansion_sum (a a₁ a₂ a₃ : ℝ) 
  (h : ∀ x : ℝ, x^3 = a + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) : 
  a₁ + a₂ + a₃ = 19 := by
sorry

end NUMINAMATH_CALUDE_cubic_expansion_sum_l1486_148687


namespace NUMINAMATH_CALUDE_multiples_of_12_around_negative_150_l1486_148647

theorem multiples_of_12_around_negative_150 :
  ∀ n m : ℤ,
  (∀ k : ℤ, 12 * k < -150 → k ≤ n) →
  (∀ j : ℤ, 12 * j > -150 → m ≤ j) →
  12 * n = -156 ∧ 12 * m = -144 :=
by
  sorry

end NUMINAMATH_CALUDE_multiples_of_12_around_negative_150_l1486_148647


namespace NUMINAMATH_CALUDE_quinton_cupcakes_l1486_148668

/-- The number of cupcakes Quinton brought to school -/
def total_cupcakes : ℕ := sorry

/-- The number of students in Ms. Delmont's class -/
def delmont_students : ℕ := 18

/-- The number of students in Mrs. Donnelly's class -/
def donnelly_students : ℕ := 16

/-- The number of staff members who received a cupcake -/
def staff_members : ℕ := 4

/-- The number of cupcakes left over -/
def leftover_cupcakes : ℕ := 2

/-- Theorem stating that the total number of cupcakes Quinton brought to school is 40 -/
theorem quinton_cupcakes : 
  total_cupcakes = delmont_students + donnelly_students + staff_members + leftover_cupcakes :=
by sorry

end NUMINAMATH_CALUDE_quinton_cupcakes_l1486_148668


namespace NUMINAMATH_CALUDE_binary_arithmetic_proof_l1486_148693

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
    aux n

theorem binary_arithmetic_proof :
  let a := [true, true, false, true, true]  -- 11011₂
  let b := [true, false, true]              -- 101₂
  let c := [false, true, false, true]       -- 1010₂
  let product := binary_to_decimal a * binary_to_decimal b
  let result := product - binary_to_decimal c
  decimal_to_binary result = [true, false, true, true, true, true, true] -- 1111101₂
  := by sorry

end NUMINAMATH_CALUDE_binary_arithmetic_proof_l1486_148693


namespace NUMINAMATH_CALUDE_monic_quartic_specific_values_l1486_148635

-- Define a monic quartic polynomial
def is_monic_quartic (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem monic_quartic_specific_values (f : ℝ → ℝ) 
  (h_monic : is_monic_quartic f)
  (h1 : f (-2) = -4)
  (h2 : f 1 = -1)
  (h3 : f (-4) = -16)
  (h4 : f 5 = -25) :
  f 0 = 40 := by sorry

end NUMINAMATH_CALUDE_monic_quartic_specific_values_l1486_148635


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1486_148639

theorem quadratic_inequality_solution (x : ℝ) : 
  -3 * x^2 + 8 * x + 1 < 0 ↔ -1/3 < x ∧ x < 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1486_148639


namespace NUMINAMATH_CALUDE_x_plus_reciprocal_three_l1486_148640

theorem x_plus_reciprocal_three (x : ℝ) (h : x ≠ 0) :
  x + 1/x = 3 →
  (x - 1)^2 + 16/(x - 1)^2 = x + 16/x :=
by
  sorry

end NUMINAMATH_CALUDE_x_plus_reciprocal_three_l1486_148640


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_and_complementary_l1486_148616

/-- Represents the number of boys in the group -/
def num_boys : ℕ := 3

/-- Represents the number of girls in the group -/
def num_girls : ℕ := 2

/-- Represents the number of students to be selected -/
def num_selected : ℕ := 2

/-- Represents the event "at least 1 girl" -/
def at_least_one_girl : Set (Fin num_boys × Fin num_girls) := sorry

/-- Represents the event "all boys" -/
def all_boys : Set (Fin num_boys × Fin num_girls) := sorry

/-- Proves that the events are mutually exclusive and complementary -/
theorem events_mutually_exclusive_and_complementary :
  (at_least_one_girl ∩ all_boys = ∅) ∧
  (at_least_one_girl ∪ all_boys = Set.univ) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_and_complementary_l1486_148616


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1486_148632

theorem sqrt_product_equality : Real.sqrt 72 * Real.sqrt 18 * Real.sqrt 8 = 72 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1486_148632


namespace NUMINAMATH_CALUDE_diagonal_cuboids_count_l1486_148618

def cuboid_count (a b c L : ℕ) : ℕ :=
  L / a + L / b + L / c - L / (a * b) - L / (a * c) - L / (b * c) + L / (a * b * c)

theorem diagonal_cuboids_count : 
  let a : ℕ := 2
  let b : ℕ := 7
  let c : ℕ := 13
  let L : ℕ := 2002
  let lcm : ℕ := a * b * c
  (L / lcm) * cuboid_count a b c lcm = 1210 := by sorry

end NUMINAMATH_CALUDE_diagonal_cuboids_count_l1486_148618


namespace NUMINAMATH_CALUDE_justin_flower_gathering_time_l1486_148645

/-- Calculates the additional time needed for Justin to gather flowers for his classmates -/
def additional_time_needed (
  classmates : ℕ)
  (average_time_per_flower : ℕ)
  (gathering_time_hours : ℕ)
  (lost_flowers : ℕ) : ℕ :=
  let gathering_time_minutes := gathering_time_hours * 60
  let flowers_gathered := gathering_time_minutes / average_time_per_flower
  let flowers_remaining := flowers_gathered - lost_flowers
  let additional_flowers_needed := classmates - flowers_remaining
  additional_flowers_needed * average_time_per_flower

theorem justin_flower_gathering_time :
  additional_time_needed 30 10 2 3 = 210 := by
  sorry

end NUMINAMATH_CALUDE_justin_flower_gathering_time_l1486_148645


namespace NUMINAMATH_CALUDE_complex_power_2019_l1486_148684

theorem complex_power_2019 (i : ℂ) (h : i^2 = -1) : i^2019 = -i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_2019_l1486_148684


namespace NUMINAMATH_CALUDE_integer_area_iff_specific_lengths_l1486_148661

/-- A right triangle with a circumscribed circle -/
structure RightTriangleWithCircle where
  AB : ℝ  -- Length of side AB
  BC : ℝ  -- Length of side BC (diameter of the circle)
  h : AB > 0
  d : BC > 0
  right_angle : AB * BC = AB^2  -- Condition for right angle and tangency

/-- The area of the triangle is an integer -/
def has_integer_area (t : RightTriangleWithCircle) : Prop :=
  ∃ n : ℕ, (1/2) * t.AB * t.BC = n

/-- The main theorem -/
theorem integer_area_iff_specific_lengths (t : RightTriangleWithCircle) :
  has_integer_area t ↔ t.AB ∈ ({4, 8, 12} : Set ℝ) :=
sorry

end NUMINAMATH_CALUDE_integer_area_iff_specific_lengths_l1486_148661


namespace NUMINAMATH_CALUDE_number_problem_l1486_148677

theorem number_problem (x : ℝ) : 2 * x - x / 2 = 45 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1486_148677


namespace NUMINAMATH_CALUDE_coordinate_change_l1486_148667

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def is_basis (v : Fin 3 → V) : Prop :=
  LinearIndependent ℝ v ∧ Submodule.span ℝ (Set.range v) = ⊤

theorem coordinate_change
  (a b c : V)
  (h1 : is_basis (![a, b, c]))
  (h2 : is_basis (![a - b, a + b, c]))
  (p : V)
  (h3 : p = 4 • a + 2 • b + (-1) • c) :
  ∃ (x y z : ℝ), p = x • (a - b) + y • (a + b) + z • c ∧ x = 1 ∧ y = 3 ∧ z = -1 :=
sorry

end NUMINAMATH_CALUDE_coordinate_change_l1486_148667


namespace NUMINAMATH_CALUDE_not_q_sufficient_not_necessary_for_p_l1486_148673

-- Define the propositions p and q
def p (x : ℝ) : Prop := x ≤ 1
def q (x : ℝ) : Prop := 1/x < 1

-- Theorem stating that ¬q is a sufficient but not necessary condition for p
theorem not_q_sufficient_not_necessary_for_p :
  (∀ x : ℝ, ¬(q x) → p x) ∧ 
  (∃ x : ℝ, p x ∧ q x) :=
sorry

end NUMINAMATH_CALUDE_not_q_sufficient_not_necessary_for_p_l1486_148673


namespace NUMINAMATH_CALUDE_log_four_eighteen_l1486_148615

theorem log_four_eighteen (a b : ℝ) (h1 : Real.log 2 / Real.log 10 = a) (h2 : Real.log 3 / Real.log 10 = b) :
  Real.log 18 / Real.log 4 = (a + 2*b) / (2*a) := by sorry

end NUMINAMATH_CALUDE_log_four_eighteen_l1486_148615


namespace NUMINAMATH_CALUDE_project_completion_days_l1486_148682

/-- Represents the time in days for a worker to complete the project alone -/
structure WorkerRate where
  days : ℕ
  days_pos : days > 0

/-- Represents the project completion scenario -/
structure ProjectCompletion where
  worker_a : WorkerRate
  worker_b : WorkerRate
  worker_c : WorkerRate
  a_quit_before_end : ℕ

/-- Calculates the total days to complete the project -/
def total_days (p : ProjectCompletion) : ℕ := 
  sorry

/-- Theorem stating that the project will be completed in 18 days -/
theorem project_completion_days (p : ProjectCompletion) 
  (h1 : p.worker_a.days = 20)
  (h2 : p.worker_b.days = 30)
  (h3 : p.worker_c.days = 40)
  (h4 : p.a_quit_before_end = 18) :
  total_days p = 18 := by
  sorry

end NUMINAMATH_CALUDE_project_completion_days_l1486_148682


namespace NUMINAMATH_CALUDE_max_value_of_complex_distance_l1486_148654

theorem max_value_of_complex_distance (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (max_val : ℝ), max_val = 5 ∧ ∀ (w : ℂ), Complex.abs (w + 2 - 2*I) = 1 → Complex.abs (w - 2 - 2*I) ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_value_of_complex_distance_l1486_148654


namespace NUMINAMATH_CALUDE_ratio_change_l1486_148658

theorem ratio_change (x y : ℕ) (n : ℕ) (h1 : y = 24) (h2 : x / y = 1 / 4) 
  (h3 : (x + n) / y = 1 / 2) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_change_l1486_148658


namespace NUMINAMATH_CALUDE_percent_of_percent_l1486_148609

theorem percent_of_percent (x : ℝ) :
  (20 / 100) * (x / 100) = 80 / 100 → x = 400 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_percent_l1486_148609


namespace NUMINAMATH_CALUDE_oz_language_word_loss_l1486_148602

theorem oz_language_word_loss :
  let total_letters : ℕ := 64
  let forbidden_letters : ℕ := 1
  let max_word_length : ℕ := 2

  let one_letter_words_lost : ℕ := forbidden_letters
  let two_letter_words_lost : ℕ := 
    total_letters * forbidden_letters + 
    forbidden_letters * total_letters - 
    forbidden_letters * forbidden_letters

  one_letter_words_lost + two_letter_words_lost = 128 :=
by sorry

end NUMINAMATH_CALUDE_oz_language_word_loss_l1486_148602


namespace NUMINAMATH_CALUDE_fourth_power_sum_l1486_148698

theorem fourth_power_sum (a b c : ℝ) 
  (sum_condition : a + b + c = 2)
  (sum_squares : a^2 + b^2 + c^2 = 5)
  (sum_cubes : a^3 + b^3 + c^3 = 8) : 
  a^4 + b^4 + c^4 = 18.5 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l1486_148698


namespace NUMINAMATH_CALUDE_gcf_180_270_l1486_148637

theorem gcf_180_270 : Nat.gcd 180 270 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcf_180_270_l1486_148637


namespace NUMINAMATH_CALUDE_first_day_of_month_l1486_148679

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

-- Define a function to get the day after n days
def dayAfter (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => nextDay (dayAfter d n)

-- Theorem statement
theorem first_day_of_month (d : DayOfWeek) :
  dayAfter d 22 = DayOfWeek.Wednesday → d = DayOfWeek.Tuesday :=
by sorry

end NUMINAMATH_CALUDE_first_day_of_month_l1486_148679


namespace NUMINAMATH_CALUDE_compound_statement_falsity_l1486_148672

theorem compound_statement_falsity (p q : Prop) : 
  ¬(p ∧ q) → (¬p ∨ ¬q) := by sorry

end NUMINAMATH_CALUDE_compound_statement_falsity_l1486_148672


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1486_148663

/-- Definition of a hyperbola with given foci and distance property -/
structure Hyperbola where
  f1 : ℝ × ℝ
  f2 : ℝ × ℝ
  dist_diff : ℝ

/-- The standard form of a hyperbola equation -/
def standard_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Theorem: The standard equation of the given hyperbola -/
theorem hyperbola_equation (h : Hyperbola) 
    (h_f1 : h.f1 = (-5, 0))
    (h_f2 : h.f2 = (5, 0))
    (h_dist : h.dist_diff = 8) :
    ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | ‖p - h.f1‖ - ‖p - h.f2‖ = h.dist_diff} →
    standard_equation 4 3 x y := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1486_148663


namespace NUMINAMATH_CALUDE_circle_center_coordinate_sum_l1486_148644

/-- The sum of the coordinates of the center of the circle defined by x^2 + y^2 = -4x - 6y + 5 is -5 -/
theorem circle_center_coordinate_sum :
  ∃ (x y : ℝ), x^2 + y^2 = -4*x - 6*y + 5 ∧ x + y = -5 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_sum_l1486_148644


namespace NUMINAMATH_CALUDE_expression_equals_ten_to_twelve_l1486_148610

theorem expression_equals_ten_to_twelve : (2 * 5 * 10^5) * 10^6 = 10^12 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_ten_to_twelve_l1486_148610


namespace NUMINAMATH_CALUDE_boat_weight_problem_l1486_148600

theorem boat_weight_problem (initial_average : ℝ) (new_person_weight : ℝ) (new_average : ℝ) :
  initial_average = 60 →
  new_person_weight = 45 →
  new_average = 55 →
  ∃ n : ℕ, n * initial_average + new_person_weight = (n + 1) * new_average ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_boat_weight_problem_l1486_148600


namespace NUMINAMATH_CALUDE_average_weight_calculation_l1486_148694

theorem average_weight_calculation (total_boys : ℕ) (group1_boys : ℕ) (group2_boys : ℕ)
  (group2_avg_weight : ℝ) (total_avg_weight : ℝ) :
  total_boys = group1_boys + group2_boys →
  group2_boys = 8 →
  group2_avg_weight = 45.15 →
  total_avg_weight = 48.55 →
  let group1_avg_weight := (total_boys * total_avg_weight - group2_boys * group2_avg_weight) / group1_boys
  group1_avg_weight = 50.25 := by
sorry

end NUMINAMATH_CALUDE_average_weight_calculation_l1486_148694


namespace NUMINAMATH_CALUDE_book_distribution_l1486_148675

theorem book_distribution (total_books : ℕ) (girls boys non_binary : ℕ) 
  (h1 : total_books = 840)
  (h2 : girls = 20)
  (h3 : boys = 15)
  (h4 : non_binary = 5)
  (h5 : ∃ (x : ℕ), 
    girls * (2 * x) + boys * x + non_binary * x = total_books ∧ 
    x > 0) :
  ∃ (books_per_boy : ℕ),
    books_per_boy = 14 ∧
    girls * (2 * books_per_boy) + boys * books_per_boy + non_binary * books_per_boy = total_books :=
by sorry

end NUMINAMATH_CALUDE_book_distribution_l1486_148675


namespace NUMINAMATH_CALUDE_prob_six_odd_in_eight_rolls_l1486_148690

/-- A fair 6-sided die -/
def fair_die : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The probability of rolling an odd number on a fair 6-sided die -/
def prob_odd : ℚ := 1/2

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 8

/-- The number of odd results we're interested in -/
def target_odd : ℕ := 6

/-- The probability of getting exactly 6 odd results in 8 rolls of a fair 6-sided die -/
theorem prob_six_odd_in_eight_rolls :
  (Nat.choose num_rolls target_odd : ℚ) * prob_odd^target_odd * (1 - prob_odd)^(num_rolls - target_odd) = 28/256 := by
  sorry

end NUMINAMATH_CALUDE_prob_six_odd_in_eight_rolls_l1486_148690


namespace NUMINAMATH_CALUDE_tank_flow_rate_l1486_148636

/-- Represents the flow rate problem for a water tank -/
theorem tank_flow_rate 
  (tank_capacity : ℝ) 
  (initial_level : ℝ) 
  (fill_time : ℝ) 
  (drain1_rate : ℝ) 
  (drain2_rate : ℝ) 
  (h1 : tank_capacity = 8000)
  (h2 : initial_level = tank_capacity / 2)
  (h3 : fill_time = 48)
  (h4 : drain1_rate = 1000 / 4)
  (h5 : drain2_rate = 1000 / 6)
  : ∃ (flow_rate : ℝ), 
    flow_rate = 500 ∧ 
    (flow_rate - (drain1_rate + drain2_rate)) * fill_time = tank_capacity - initial_level :=
by sorry


end NUMINAMATH_CALUDE_tank_flow_rate_l1486_148636


namespace NUMINAMATH_CALUDE_find_a_range_of_t_l1486_148611

-- Define the function f
def f (x a : ℝ) := |2 * x - a| + a

-- Theorem 1
theorem find_a : 
  (∀ x, f x 1 ≤ 4 ↔ -1 ≤ x ∧ x ≤ 2) → 
  (∃! a, ∀ x, f x a ≤ 4 ↔ -1 ≤ x ∧ x ≤ 2) ∧ 
  (∀ x, f x 1 ≤ 4 ↔ -1 ≤ x ∧ x ≤ 2) :=
sorry

-- Theorem 2
theorem range_of_t :
  (∀ t : ℝ, (∃ n : ℝ, |2 * n - 1| + 1 ≤ t - (|2 * (-n) - 1| + 1)) ↔ t ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_find_a_range_of_t_l1486_148611


namespace NUMINAMATH_CALUDE_inequality_proof_l1486_148680

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  3 + (a + b + c) + (1/a + 1/b + 1/c) + (a/b + b/c + c/a) ≥ (3*(a+1)*(b+1)*(c+1))/(a*b*c+1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1486_148680


namespace NUMINAMATH_CALUDE_inverse_of_5_mod_34_l1486_148669

theorem inverse_of_5_mod_34 : ∃ x : ℕ, x < 34 ∧ (5 * x) % 34 = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_5_mod_34_l1486_148669


namespace NUMINAMATH_CALUDE_total_money_calculation_l1486_148620

def hundred_bills : ℕ := 2
def fifty_bills : ℕ := 5
def ten_bills : ℕ := 10

def hundred_value : ℕ := 100
def fifty_value : ℕ := 50
def ten_value : ℕ := 10

theorem total_money_calculation : 
  (hundred_bills * hundred_value) + (fifty_bills * fifty_value) + (ten_bills * ten_value) = 550 := by
  sorry

end NUMINAMATH_CALUDE_total_money_calculation_l1486_148620


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1486_148659

/-- Given an arithmetic sequence {aₙ} where Sₙ denotes the sum of its first n terms,
    if a₄ + a₆ + a₈ = 15, then S₁₁ = 55. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  (∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2) →          -- sum formula
  a 4 + a 6 + a 8 = 15 →                                -- given condition
  S 11 = 55 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1486_148659


namespace NUMINAMATH_CALUDE_fraction_of_satisfactory_grades_l1486_148606

-- Define the grades
inductive Grade
| A
| B
| C
| D
| F

-- Define a function to check if a grade is satisfactory
def is_satisfactory (g : Grade) : Prop :=
  g = Grade.B ∨ g = Grade.C ∨ g = Grade.D

-- Define the number of students for each grade
def num_students (g : Grade) : ℕ :=
  match g with
  | Grade.A => 8
  | Grade.B => 6
  | Grade.C => 5
  | Grade.D => 4
  | Grade.F => 7

-- Define the total number of students
def total_students : ℕ :=
  num_students Grade.A + num_students Grade.B + num_students Grade.C +
  num_students Grade.D + num_students Grade.F

-- Define the number of students with satisfactory grades
def satisfactory_students : ℕ :=
  num_students Grade.B + num_students Grade.C + num_students Grade.D

-- Theorem to prove
theorem fraction_of_satisfactory_grades :
  (satisfactory_students : ℚ) / total_students = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_satisfactory_grades_l1486_148606


namespace NUMINAMATH_CALUDE_partition_five_elements_l1486_148664

/-- The number of ways to partition a set of 5 elements into two non-empty subsets, 
    where two specific elements must be in the same subset -/
def partitionWays : ℕ := 6

/-- A function that calculates the number of ways to partition a set of n elements into two non-empty subsets,
    where two specific elements must be in the same subset -/
def partitionFunction (n : ℕ) : ℕ :=
  if n < 3 then 0 else (n - 2)

theorem partition_five_elements :
  partitionWays = partitionFunction 5 :=
by sorry

end NUMINAMATH_CALUDE_partition_five_elements_l1486_148664


namespace NUMINAMATH_CALUDE_smallest_integer_with_divisibility_prove_smallest_integer_l1486_148604

def is_divisible (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def is_divisible_by_range (n : ℕ) (a b : ℕ) : Prop :=
  ∀ i : ℕ, a ≤ i ∧ i ≤ b → is_divisible n i

theorem smallest_integer_with_divisibility (n : ℕ) : Prop :=
  (n = 1225224000) ∧
  (is_divisible_by_range n 1 26) ∧
  (is_divisible_by_range n 30 30) ∧
  (¬ is_divisible n 27) ∧
  (¬ is_divisible n 28) ∧
  (¬ is_divisible n 29) ∧
  (∀ m : ℕ, m < n →
    ¬(is_divisible_by_range m 1 26 ∧
      is_divisible_by_range m 30 30 ∧
      ¬ is_divisible m 27 ∧
      ¬ is_divisible m 28 ∧
      ¬ is_divisible m 29))

theorem prove_smallest_integer : ∃ n : ℕ, smallest_integer_with_divisibility n :=
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_divisibility_prove_smallest_integer_l1486_148604


namespace NUMINAMATH_CALUDE_original_average_score_l1486_148662

/-- Proves that the original average score of a class is 37, given the conditions. -/
theorem original_average_score (num_students : ℕ) (grace_marks : ℕ) (new_average : ℕ) :
  num_students = 35 →
  grace_marks = 3 →
  new_average = 40 →
  (num_students * new_average - num_students * grace_marks) / num_students = 37 :=
by sorry

end NUMINAMATH_CALUDE_original_average_score_l1486_148662


namespace NUMINAMATH_CALUDE_water_bottle_consumption_l1486_148691

/-- Proves that given a 24-pack of bottled water, if 1/3 is consumed on the first day
    and 1/2 of the remainder is consumed on the second day, then 8 bottles remain after 2 days. -/
theorem water_bottle_consumption (initial_bottles : ℕ) 
  (h1 : initial_bottles = 24)
  (first_day_consumption : ℚ) 
  (h2 : first_day_consumption = 1/3)
  (second_day_consumption : ℚ) 
  (h3 : second_day_consumption = 1/2) :
  initial_bottles - 
  (↑initial_bottles * first_day_consumption).floor - 
  ((↑initial_bottles - (↑initial_bottles * first_day_consumption).floor) * second_day_consumption).floor = 8 :=
by sorry

end NUMINAMATH_CALUDE_water_bottle_consumption_l1486_148691


namespace NUMINAMATH_CALUDE_average_gas_mileage_round_trip_l1486_148603

/-- Calculates the average gas mileage for a round trip with different distances and fuel efficiencies -/
theorem average_gas_mileage_round_trip 
  (distance_outgoing : ℝ) 
  (distance_return : ℝ)
  (efficiency_outgoing : ℝ)
  (efficiency_return : ℝ) :
  let total_distance := distance_outgoing + distance_return
  let total_fuel := distance_outgoing / efficiency_outgoing + distance_return / efficiency_return
  let average_mileage := total_distance / total_fuel
  (distance_outgoing = 150 ∧ 
   distance_return = 180 ∧ 
   efficiency_outgoing = 25 ∧ 
   efficiency_return = 50) →
  (34 < average_mileage ∧ average_mileage < 35) :=
by sorry

end NUMINAMATH_CALUDE_average_gas_mileage_round_trip_l1486_148603


namespace NUMINAMATH_CALUDE_quadratic_properties_l1486_148614

-- Define the quadratic function
def quadratic (b c x : ℝ) : ℝ := -x^2 + b*x + c

theorem quadratic_properties :
  ∀ (b c : ℝ),
  -- Part 1
  (quadratic b c (-1) = 0 ∧ quadratic b c 3 = 0 →
    ∃ x, ∀ y, quadratic b c y ≤ quadratic b c x ∧ quadratic b c x = 4) ∧
  -- Part 2
  (c = -5 ∧ (∃! x, quadratic b c x = 1) →
    b = 2 * Real.sqrt 6 ∨ b = -2 * Real.sqrt 6) ∧
  -- Part 3
  (c = b^2 ∧ (∃ x, b ≤ x ∧ x ≤ b + 3 ∧
    ∀ y, b ≤ y ∧ y ≤ b + 3 → quadratic b c y ≤ quadratic b c x) ∧
    quadratic b c x = 20 →
    b = 2 * Real.sqrt 5 ∨ b = -4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1486_148614


namespace NUMINAMATH_CALUDE_selection_theorem_l1486_148642

/-- Represents the number of students with each skill -/
structure StudentGroup where
  total : ℕ
  singers : ℕ
  dancers : ℕ
  both : ℕ

/-- Represents the selection requirements -/
structure SelectionRequirement where
  singersToSelect : ℕ
  dancersToSelect : ℕ

/-- Calculates the number of ways to select students given a student group and selection requirements -/
def numberOfWaysToSelect (group : StudentGroup) (req : SelectionRequirement) : ℕ :=
  sorry

/-- The theorem to be proved -/
theorem selection_theorem (group : StudentGroup) (req : SelectionRequirement) :
  group.total = 6 ∧ 
  group.singers = 3 ∧ 
  group.dancers = 2 ∧ 
  group.both = 1 ∧
  req.singersToSelect = 2 ∧
  req.dancersToSelect = 1 →
  numberOfWaysToSelect group req = 15 :=
by sorry

end NUMINAMATH_CALUDE_selection_theorem_l1486_148642


namespace NUMINAMATH_CALUDE_cyclic_inequality_l1486_148628

theorem cyclic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 / (a^2 + a*b + b^2)) + (b^3 / (b^2 + b*c + c^2)) + (c^3 / (c^2 + c*a + a^2)) ≥ (a + b + c) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l1486_148628


namespace NUMINAMATH_CALUDE_sum_minus_k_equals_ten_l1486_148678

theorem sum_minus_k_equals_ten (n k : ℕ) (a : ℕ) (h1 : 1 < k) (h2 : k < n) 
  (h3 : (n * (n + 1) / 2 - k) / (n - 1) = 10) (h4 : n + k = a) : a = 29 := by
  sorry

end NUMINAMATH_CALUDE_sum_minus_k_equals_ten_l1486_148678


namespace NUMINAMATH_CALUDE_bisection_method_step_next_interval_is_1_5_to_2_l1486_148666

def f (x : ℝ) := x^3 - x - 5

theorem bisection_method_step (a b : ℝ) (hab : a < b) (hf : f a * f b < 0) :
  let m := (a + b) / 2
  (f a * f m < 0 ∧ (m, b) = (1.5, 2)) ∨
  (f m * f b < 0 ∧ (a, m) = (1.5, 2)) :=
sorry

theorem next_interval_is_1_5_to_2 :
  let a := 1
  let b := 2
  let m := (a + b) / 2
  m = 1.5 ∧ f a * f b < 0 →
  (1.5, 2) = (let m := (a + b) / 2; if f a * f m < 0 then (a, m) else (m, b)) :=
sorry

end NUMINAMATH_CALUDE_bisection_method_step_next_interval_is_1_5_to_2_l1486_148666


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1486_148633

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- {an} is a geometric sequence with common ratio q
  q > 0 →                       -- q is positive
  a 2 = 1 →                     -- a2 = 1
  a 4 = 4 →                     -- a4 = 4
  q = 2 :=                      -- prove q = 2
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1486_148633


namespace NUMINAMATH_CALUDE_cube_root_of_negative_64_l1486_148696

theorem cube_root_of_negative_64 : ∃ x : ℝ, x^3 = -64 ∧ x = -4 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_64_l1486_148696


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l1486_148630

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2018 = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l1486_148630


namespace NUMINAMATH_CALUDE_kenneth_remaining_money_l1486_148695

-- Define the initial amount Kenneth has
def initial_amount : ℕ := 50

-- Define the number of baguettes and bottles of water
def num_baguettes : ℕ := 2
def num_water_bottles : ℕ := 2

-- Define the cost of each baguette and bottle of water
def cost_baguette : ℕ := 2
def cost_water : ℕ := 1

-- Define the total cost of purchases
def total_cost : ℕ := num_baguettes * cost_baguette + num_water_bottles * cost_water

-- Define the remaining money after purchases
def remaining_money : ℕ := initial_amount - total_cost

-- Theorem statement
theorem kenneth_remaining_money :
  remaining_money = 44 :=
by sorry

end NUMINAMATH_CALUDE_kenneth_remaining_money_l1486_148695


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l1486_148671

theorem sqrt_sum_equals_seven (x : ℝ) (h : Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4) :
  Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) = 7 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l1486_148671


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1486_148607

def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 2}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1486_148607


namespace NUMINAMATH_CALUDE_one_third_to_fifth_power_l1486_148681

theorem one_third_to_fifth_power : (1 / 3 : ℚ) ^ 5 = 1 / 243 := by sorry

end NUMINAMATH_CALUDE_one_third_to_fifth_power_l1486_148681


namespace NUMINAMATH_CALUDE_solution_to_equation_l1486_148660

theorem solution_to_equation :
  ∃! (x y : ℝ), x^2 + (1 - y)^2 + (x - y)^2 = 1/3 ∧ x = 1/3 ∧ y = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l1486_148660


namespace NUMINAMATH_CALUDE_sports_club_members_l1486_148619

theorem sports_club_members (badminton tennis both neither : ℕ) 
  (h1 : badminton = 17)
  (h2 : tennis = 19)
  (h3 : both = 8)
  (h4 : neither = 2) :
  badminton + tennis - both + neither = 30 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_members_l1486_148619


namespace NUMINAMATH_CALUDE_minimum_houses_with_more_than_five_floors_l1486_148612

theorem minimum_houses_with_more_than_five_floors (n : ℕ) : 
  (n > 0) → 
  (∃ x : ℕ, x < n ∧ (n - x : ℚ) / n > 47/50) → 
  (∀ m : ℕ, m < n → ∃ y : ℕ, y < m ∧ (m - y : ℚ) / m ≤ 47/50) → 
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_minimum_houses_with_more_than_five_floors_l1486_148612


namespace NUMINAMATH_CALUDE_min_even_integers_l1486_148653

theorem min_even_integers (a b c d e f : ℤ) : 
  a + b = 30 →
  a + b + c + d = 50 →
  a + b + c + d + e + f = 70 →
  ∃ (x y z w u v : ℤ), 
    x + y = 30 ∧
    x + y + z + w = 50 ∧
    x + y + z + w + u + v = 70 ∧
    Even x ∧ Even y ∧ Even z ∧ Even w ∧ Even u ∧ Even v :=
by sorry

end NUMINAMATH_CALUDE_min_even_integers_l1486_148653


namespace NUMINAMATH_CALUDE_units_digit_of_quotient_l1486_148699

theorem units_digit_of_quotient (h : 7 ∣ (4^2065 + 6^2065)) :
  (4^2065 + 6^2065) / 7 % 10 = 0 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_quotient_l1486_148699


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1486_148646

theorem diophantine_equation_solution :
  ∀ (p a b c : ℕ),
    p.Prime →
    0 < a ∧ 0 < b ∧ 0 < c →
    73 * p^2 + 6 = 9 * a^2 + 17 * b^2 + 17 * c^2 →
    ((p = 2 ∧ a = 1 ∧ b = 4 ∧ c = 1) ∨ (p = 2 ∧ a = 1 ∧ b = 1 ∧ c = 4)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1486_148646


namespace NUMINAMATH_CALUDE_min_zeros_in_special_set_l1486_148685

theorem min_zeros_in_special_set (n : ℕ) (a : Fin n → ℝ) 
  (h : n = 2011)
  (sum_property : ∀ i j k : Fin n, ∃ l : Fin n, a i + a j + a k = a l) :
  (Finset.filter (fun i => a i = 0) Finset.univ).card ≥ 2009 :=
sorry

end NUMINAMATH_CALUDE_min_zeros_in_special_set_l1486_148685


namespace NUMINAMATH_CALUDE_perpendicular_line_proof_l1486_148622

noncomputable def curve (x : ℝ) : ℝ := 2 * x^2

def point : ℝ × ℝ := (1, 2)

def tangent_slope : ℝ := 4

def perpendicular_line (x y : ℝ) : Prop := x + 4*y - 9 = 0

theorem perpendicular_line_proof :
  perpendicular_line point.1 point.2 ∧
  (∃ k : ℝ, k * tangent_slope = -1 ∧
    ∀ x y : ℝ, perpendicular_line x y ↔ y - point.2 = k * (x - point.1)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_proof_l1486_148622


namespace NUMINAMATH_CALUDE_company_signs_used_l1486_148686

/-- The number of signs in the special sign language --/
def total_signs : ℕ := 124

/-- The number of unused signs --/
def unused_signs : ℕ := 2

/-- The number of additional area codes if all signs were used --/
def additional_codes : ℕ := 488

/-- The number of signs in each area code --/
def signs_per_code : ℕ := 2

/-- The number of signs used fully by the company --/
def signs_used : ℕ := total_signs - unused_signs

theorem company_signs_used : signs_used = 120 := by
  sorry

end NUMINAMATH_CALUDE_company_signs_used_l1486_148686


namespace NUMINAMATH_CALUDE_parabola_vertex_l1486_148665

/-- The vertex of a parabola defined by y^2 + 8y + 4x + 5 = 0 is (11/4, -4) -/
theorem parabola_vertex : 
  let f (x y : ℝ) := y^2 + 8*y + 4*x + 5
  ∃! (vx vy : ℝ), (∀ (x y : ℝ), f x y = 0 → (x - vx)^2 ≥ 0) ∧ vx = 11/4 ∧ vy = -4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1486_148665


namespace NUMINAMATH_CALUDE_rectangle_covers_ellipse_l1486_148674

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of an ellipse -/
structure Ellipse where
  major_axis : ℝ
  minor_axis : ℝ

/-- Checks if a rectangle can cover an ellipse -/
def can_cover (r : Rectangle) (e : Ellipse) : Prop :=
  r.length ≥ e.minor_axis ∧
  r.width ≥ e.minor_axis ∧
  r.length^2 + r.width^2 ≥ e.major_axis^2 + e.minor_axis^2

/-- The specific rectangle and ellipse from the problem -/
def problem_rectangle : Rectangle := ⟨140, 130⟩
def problem_ellipse : Ellipse := ⟨160, 100⟩

/-- Theorem stating that the problem_rectangle can cover the problem_ellipse -/
theorem rectangle_covers_ellipse : can_cover problem_rectangle problem_ellipse :=
  sorry

end NUMINAMATH_CALUDE_rectangle_covers_ellipse_l1486_148674


namespace NUMINAMATH_CALUDE_consecutive_draw_probability_l1486_148631

def num_purple_chips : ℕ := 4
def num_orange_chips : ℕ := 3
def num_green_chips : ℕ := 5
def total_chips : ℕ := num_purple_chips + num_orange_chips + num_green_chips

def probability_consecutive_draw : ℚ :=
  (Nat.factorial 2 * Nat.factorial num_purple_chips * Nat.factorial num_orange_chips * Nat.factorial num_green_chips) /
  Nat.factorial total_chips

theorem consecutive_draw_probability :
  probability_consecutive_draw = 1 / 13860 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_draw_probability_l1486_148631


namespace NUMINAMATH_CALUDE_watermelon_price_in_units_l1486_148657

/-- The price of a watermelon in won -/
def watermelon_price : ℝ := 5000 - 200

/-- The conversion factor from won to units of 1000 won -/
def conversion_factor : ℝ := 1000

theorem watermelon_price_in_units : watermelon_price / conversion_factor = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_price_in_units_l1486_148657


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l1486_148643

/-- An isosceles triangle with perimeter 26 and one side 12 has the other side length either 12 or 7 -/
theorem isosceles_triangle_side_length (a b c : ℝ) : 
  a + b + c = 26 → -- perimeter is 26
  (a = b ∨ b = c ∨ a = c) → -- isosceles condition
  (a = 12 ∨ b = 12 ∨ c = 12) → -- one side is 12
  (a = 7 ∨ b = 7 ∨ c = 7) ∨ (a = 12 ∧ b = 12) ∨ (b = 12 ∧ c = 12) ∨ (a = 12 ∧ c = 12) :=
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l1486_148643
