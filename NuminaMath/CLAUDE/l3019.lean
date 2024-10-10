import Mathlib

namespace arithmetic_sequence_first_term_l3019_301986

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  sum : ℕ → ℝ -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, sum n = n * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem: For an arithmetic sequence, if S₄ = 6 and 2a₃ - a₂ = 6, then a₁ = -3 -/
theorem arithmetic_sequence_first_term
  (seq : ArithmeticSequence)
  (sum_4 : seq.sum 4 = 6)
  (term_relation : 2 * seq.a 3 - seq.a 2 = 6) :
  seq.a 1 = -3 := by
  sorry

end arithmetic_sequence_first_term_l3019_301986


namespace square_area_to_perimeter_ratio_l3019_301983

theorem square_area_to_perimeter_ratio (s₁ s₂ : ℝ) (h : s₁ ^ 2 / s₂ ^ 2 = 25 / 36) :
  (4 * s₁) / (4 * s₂) = 5 / 6 := by
  sorry

end square_area_to_perimeter_ratio_l3019_301983


namespace original_price_of_discounted_dress_l3019_301996

/-- Proves that given a 30% discount on a dress that results in a final price of $35, the original price of the dress was $50. -/
theorem original_price_of_discounted_dress (discount_percentage : ℝ) (final_price : ℝ) : 
  discount_percentage = 30 →
  final_price = 35 →
  (1 - discount_percentage / 100) * 50 = final_price :=
by sorry

end original_price_of_discounted_dress_l3019_301996


namespace odd_prime_expression_factors_l3019_301975

theorem odd_prime_expression_factors (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) 
  (hodd_a : Odd a) (hodd_b : Odd b) (hab : a < b) : 
  (Finset.filter (· ∣ a^3 * b) (Finset.range (a^3 * b + 1))).card = 8 := by
  sorry

end odd_prime_expression_factors_l3019_301975


namespace cards_given_to_mary_problem_l3019_301957

def cards_given_to_mary (initial_cards found_cards final_cards : ℕ) : ℕ :=
  initial_cards + found_cards - final_cards

theorem cards_given_to_mary_problem : cards_given_to_mary 26 40 48 = 18 := by
  sorry

end cards_given_to_mary_problem_l3019_301957


namespace game_show_boxes_l3019_301945

theorem game_show_boxes (n : ℕ) (h1 : n > 0) : 
  (((n - 1 : ℝ) / n) ^ 3 = 0.2962962962962963) → n = 3 := by
  sorry

end game_show_boxes_l3019_301945


namespace point_P_and_min_value_l3019_301985

-- Define the points
def A : ℝ × ℝ := (3, 4)
def C : ℝ × ℝ := (0, 0)
def M : ℝ × ℝ := (0, 1)
def N : ℝ × ℝ := (1, 0)

-- Define vectors
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def AM : ℝ × ℝ := (M.1 - A.1, M.2 - A.2)
def AN : ℝ × ℝ := (N.1 - A.1, N.2 - A.2)

-- Define the vector equation
def vector_equation (x y : ℝ) : Prop :=
  AC = (x * AM.1, x * AM.2) + (y * AN.1, y * AN.2)

-- Theorem statement
theorem point_P_and_min_value :
  ∃ (x y : ℝ), vector_equation x y ∧ 
  x = 2/3 ∧ y = 1/2 ∧ 
  ∀ (a b : ℝ), vector_equation a b → 9*x^2 + 16*y^2 ≤ 9*a^2 + 16*b^2 :=
sorry

end point_P_and_min_value_l3019_301985


namespace school_sample_size_l3019_301938

theorem school_sample_size (n : ℕ) : 
  (6 : ℚ) / 11 * n / 10 - (5 : ℚ) / 11 * n / 10 = 12 → n = 1320 := by
  sorry

end school_sample_size_l3019_301938


namespace factor_expression_l3019_301999

theorem factor_expression (x : ℝ) : 75 * x^12 + 225 * x^24 = 75 * x^12 * (1 + 3 * x^12) := by
  sorry

end factor_expression_l3019_301999


namespace half_abs_diff_squares_15_13_l3019_301926

theorem half_abs_diff_squares_15_13 : 
  (1/2 : ℝ) * |15^2 - 13^2| = 28 := by
  sorry

end half_abs_diff_squares_15_13_l3019_301926


namespace monthly_spending_fraction_l3019_301970

/-- If a person saves a constant fraction of their unchanging monthly salary,
    and their yearly savings are 6 times their monthly spending,
    then they spend 2/3 of their salary each month. -/
theorem monthly_spending_fraction
  (salary : ℝ)
  (savings_fraction : ℝ)
  (h_salary_positive : 0 < salary)
  (h_savings_fraction : 0 ≤ savings_fraction ∧ savings_fraction ≤ 1)
  (h_yearly_savings : 12 * savings_fraction * salary = 6 * (1 - savings_fraction) * salary) :
  1 - savings_fraction = 2 / 3 := by
sorry

end monthly_spending_fraction_l3019_301970


namespace smallest_two_digit_prime_with_composite_reverse_l3019_301958

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem smallest_two_digit_prime_with_composite_reverse : 
  ∃ (p : ℕ), is_two_digit p ∧ Nat.Prime p ∧ 
  ¬(Nat.Prime (reverse_digits p)) ∧
  ∀ (q : ℕ), is_two_digit q → Nat.Prime q → 
  ¬(Nat.Prime (reverse_digits q)) → p ≤ q ∧ p = 23 :=
sorry

end smallest_two_digit_prime_with_composite_reverse_l3019_301958


namespace perpendicular_lines_slope_l3019_301995

theorem perpendicular_lines_slope (a : ℝ) : 
  (∀ x y, a * x + 2 * y + 1 = 0) →
  (∀ x y, x + y - 2 = 0) →
  (∀ x₁ y₁ x₂ y₂, a * x₁ + 2 * y₁ + 1 = 0 ∧ x₂ + y₂ - 2 = 0 → 
    (y₂ - y₁) * (x₂ - x₁) = -(x₂ - x₁) * (y₂ - y₁)) →
  a = -2 :=
by sorry

end perpendicular_lines_slope_l3019_301995


namespace suraya_kayla_difference_l3019_301968

/-- The number of apples picked by each person --/
structure ApplePickers where
  suraya : ℕ
  caleb : ℕ
  kayla : ℕ

/-- The conditions of the apple-picking scenario --/
def apple_picking_scenario (p : ApplePickers) : Prop :=
  p.suraya = p.caleb + 12 ∧
  p.caleb + 5 = p.kayla ∧
  p.kayla = 20

/-- The theorem stating the difference between Suraya's and Kayla's apple count --/
theorem suraya_kayla_difference (p : ApplePickers) 
  (h : apple_picking_scenario p) : p.suraya - p.kayla = 7 := by
  sorry

end suraya_kayla_difference_l3019_301968


namespace unique_function_theorem_l3019_301921

def is_valid_function (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, ∃! k : ℕ, k > 0 ∧ (f^[k] n ≤ n + k + 1)

theorem unique_function_theorem :
  ∀ f : ℕ → ℕ, is_valid_function f → ∀ n : ℕ, f n = n + 2 := by sorry

end unique_function_theorem_l3019_301921


namespace f_comp_f_four_roots_l3019_301928

/-- A quadratic function f(x) = x^2 + 10x + d -/
def f (d : ℝ) (x : ℝ) : ℝ := x^2 + 10*x + d

/-- The composition of f with itself -/
def f_comp_f (d : ℝ) (x : ℝ) : ℝ := f d (f d x)

/-- The theorem stating the condition for f(f(x)) to have exactly 4 distinct real roots -/
theorem f_comp_f_four_roots (d : ℝ) :
  (∃ (a b c e : ℝ), a < b ∧ b < c ∧ c < e ∧
    (∀ x : ℝ, f_comp_f d x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = e)) ↔
  d < 25 :=
sorry

end f_comp_f_four_roots_l3019_301928


namespace third_smallest_number_indeterminate_l3019_301911

theorem third_smallest_number_indeterminate 
  (a b c d : ℕ) 
  (h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d)
  (h_sum1 : a + b + c = 21)
  (h_sum2 : a + b + d = 27)
  (h_sum3 : a + c + d = 30) :
  ¬∃(n : ℕ), ∀(x : ℕ), (x = c) ↔ (x = n) :=
sorry

end third_smallest_number_indeterminate_l3019_301911


namespace three_numbers_sum_l3019_301956

theorem three_numbers_sum (x y z : ℤ) 
  (sum_xy : x + y = 40)
  (sum_yz : y + z = 50)
  (sum_zx : z + x = 70) :
  x = 30 ∧ y = 10 ∧ z = 40 := by
sorry

end three_numbers_sum_l3019_301956


namespace max_value_a_plus_sqrt3b_l3019_301923

theorem max_value_a_plus_sqrt3b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : Real.sqrt 3 * b = Real.sqrt ((1 - a) * (1 + a))) :
  ∃ (x : ℝ), ∀ (y : ℝ), (∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧
    Real.sqrt 3 * b' = Real.sqrt ((1 - a') * (1 + a')) ∧
    y = a' + Real.sqrt 3 * b') →
  y ≤ x ∧ x = Real.sqrt 2 :=
by sorry

end max_value_a_plus_sqrt3b_l3019_301923


namespace seventh_observation_seventh_observation_value_l3019_301998

theorem seventh_observation (initial_count : Nat) (initial_avg : ℝ) (new_avg : ℝ) : ℝ :=
  let total_count : Nat := initial_count + 1
  let initial_sum : ℝ := initial_count * initial_avg
  let new_sum : ℝ := total_count * new_avg
  new_sum - initial_sum

theorem seventh_observation_value :
  seventh_observation 6 12 11 = 5 := by
  sorry

end seventh_observation_seventh_observation_value_l3019_301998


namespace parallel_vectors_implies_m_equals_one_l3019_301910

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given vectors a and b, prove that if they are parallel, then m = 1 -/
theorem parallel_vectors_implies_m_equals_one :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (m, m + 1)
  are_parallel a b → m = 1 :=
by
  sorry

end parallel_vectors_implies_m_equals_one_l3019_301910


namespace vector_equality_sufficient_not_necessary_l3019_301988

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def parallel (a b : E) : Prop := ∃ (k : ℝ), a = k • b

theorem vector_equality_sufficient_not_necessary 
  (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (a = b → (‖a‖ = ‖b‖ ∧ parallel a b)) ∧ 
  ∃ (c d : E), ‖c‖ = ‖d‖ ∧ parallel c d ∧ c ≠ d := by
  sorry

end vector_equality_sufficient_not_necessary_l3019_301988


namespace no_real_roots_l3019_301950

theorem no_real_roots (a b : ℝ) (h1 : b/a > 1/4) (h2 : a > 0) :
  ∀ x : ℝ, x/a + b/x ≠ 1 := by
sorry

end no_real_roots_l3019_301950


namespace event_probability_theorem_l3019_301961

/-- Given an event A with constant probability in three independent trials, 
    if the probability of A occurring at least once is 63/64, 
    then the probability of A occurring exactly once is 9/64. -/
theorem event_probability_theorem (p : ℝ) : 
  (0 ≤ p) ∧ (p ≤ 1) →
  (1 - (1 - p)^3 = 63/64) →
  (3 * p * (1 - p)^2 = 9/64) :=
by sorry

end event_probability_theorem_l3019_301961


namespace calculate_expression_l3019_301920

theorem calculate_expression : 
  Real.sqrt 4 - abs (-1/4 : ℝ) + (π - 2)^0 + 2^(-2 : ℝ) = 3 := by sorry

end calculate_expression_l3019_301920


namespace complex_expression_equality_l3019_301900

/-- Given complex numbers a and b, prove that 2a - 3bi equals 22 - 12i -/
theorem complex_expression_equality (a b : ℂ) (ha : a = 5 - 3*I) (hb : b = 2 + 4*I) :
  2*a - 3*b*I = 22 - 12*I :=
by sorry

end complex_expression_equality_l3019_301900


namespace intersection_point_unique_l3019_301969

/-- The intersection point of two lines in 2D space -/
def intersection_point : ℚ × ℚ := (20/7, -11/7)

/-- First line equation: 5x - 3y = 19 -/
def line1 (x y : ℚ) : Prop := 5 * x - 3 * y = 19

/-- Second line equation: 6x + 2y = 14 -/
def line2 (x y : ℚ) : Prop := 6 * x + 2 * y = 14

theorem intersection_point_unique :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → x' = x ∧ y' = y :=
by sorry

end intersection_point_unique_l3019_301969


namespace rehana_age_l3019_301925

/-- Represents the ages of Rehana, Phoebe, and Jacob -/
structure Ages where
  rehana : ℕ
  phoebe : ℕ
  jacob : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.jacob = 3 ∧
  ages.jacob = (3 * ages.phoebe) / 5 ∧
  ages.rehana + 5 = 3 * (ages.phoebe + 5)

/-- The theorem stating Rehana's current age -/
theorem rehana_age :
  ∃ (ages : Ages), problem_conditions ages ∧ ages.rehana = 25 := by
  sorry

end rehana_age_l3019_301925


namespace garden_perimeter_l3019_301954

/-- The perimeter of a rectangular garden with width 4 meters and the same area as a rectangular playground of length 16 meters and width 12 meters is 104 meters. -/
theorem garden_perimeter : 
  let playground_length : ℝ := 16
  let playground_width : ℝ := 12
  let garden_width : ℝ := 4
  let playground_area := playground_length * playground_width
  let garden_length := playground_area / garden_width
  let garden_perimeter := 2 * (garden_length + garden_width)
  garden_perimeter = 104 := by sorry

end garden_perimeter_l3019_301954


namespace cara_age_difference_l3019_301901

/-- The age difference between Cara and her mom -/
def age_difference (grandmother_age mom_age_difference cara_age : ℕ) : ℕ :=
  grandmother_age - mom_age_difference - cara_age

/-- Proof that Cara is 20 years younger than her mom -/
theorem cara_age_difference :
  age_difference 75 15 40 = 20 := by
  sorry

end cara_age_difference_l3019_301901


namespace meetings_count_l3019_301967

/-- Represents the movement of an individual between two points -/
structure Movement where
  speed : ℝ
  journeys : ℕ

/-- Calculates the number of meetings between two individuals -/
def calculate_meetings (a b : Movement) : ℕ :=
  sorry

theorem meetings_count :
  let a : Movement := { speed := 1, journeys := 2015 }
  let b : Movement := { speed := 2, journeys := 4029 }
  (calculate_meetings a b) = 6044 := by
  sorry

end meetings_count_l3019_301967


namespace child_growth_l3019_301939

theorem child_growth (current_height previous_height : ℝ) 
  (h1 : current_height = 41.5)
  (h2 : previous_height = 38.5) : 
  current_height - previous_height = 3 := by
sorry

end child_growth_l3019_301939


namespace length_AG_l3019_301932

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Right-angled at A
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∧
  -- AB = 3
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 9 ∧
  -- AC = 3√3
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 27

-- Define the altitude AD
def Altitude (A B C D : ℝ × ℝ) : Prop :=
  (D.1 - A.1) * (B.1 - C.1) + (D.2 - A.2) * (B.2 - C.2) = 0

-- Define the median AM
def Median (A B C M : ℝ × ℝ) : Prop :=
  M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the intersection point G
def Intersection (A D M G : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, G = (A.1 + t * (D.1 - A.1), A.2 + t * (D.2 - A.2)) ∧
             ∃ s : ℝ, G = (A.1 + s * (M.1 - A.1), A.2 + s * (M.2 - A.2))

-- Theorem statement
theorem length_AG (A B C D M G : ℝ × ℝ) :
  Triangle A B C →
  Altitude A B C D →
  Median A B C M →
  Intersection A D M G →
  (G.1 - A.1)^2 + (G.2 - A.2)^2 = 243/64 :=
by sorry

end length_AG_l3019_301932


namespace pear_juice_percentage_l3019_301922

def pears_for_juice : ℕ := 4
def oranges_for_juice : ℕ := 3
def pear_juice_yield : ℚ := 12
def orange_juice_yield : ℚ := 6
def pears_in_blend : ℕ := 8
def oranges_in_blend : ℕ := 6

theorem pear_juice_percentage :
  let pear_juice_per_fruit : ℚ := pear_juice_yield / pears_for_juice
  let orange_juice_per_fruit : ℚ := orange_juice_yield / oranges_for_juice
  let total_pear_juice : ℚ := pear_juice_per_fruit * pears_in_blend
  let total_orange_juice : ℚ := orange_juice_per_fruit * oranges_in_blend
  let total_juice : ℚ := total_pear_juice + total_orange_juice
  (total_pear_juice / total_juice) * 100 = 200 / 3 := by sorry

end pear_juice_percentage_l3019_301922


namespace demand_exceeds_50k_july_august_l3019_301966

def S (n : ℕ) : ℚ := n / 27 * (21 * n - n^2 - 5)

def demand_exceeds_50k (n : ℕ) : Prop := S n - S (n-1) > 5

theorem demand_exceeds_50k_july_august :
  demand_exceeds_50k 7 ∧ demand_exceeds_50k 8 ∧
  ∀ m, m < 7 ∨ m > 8 → ¬demand_exceeds_50k m :=
sorry

end demand_exceeds_50k_july_august_l3019_301966


namespace rosie_account_balance_l3019_301974

/-- Represents the total amount in Rosie's account after m deposits -/
def account_balance (initial_amount : ℕ) (deposit_amount : ℕ) (num_deposits : ℕ) : ℕ :=
  initial_amount + deposit_amount * num_deposits

/-- Theorem stating that Rosie's account balance is correctly represented -/
theorem rosie_account_balance (m : ℕ) : 
  account_balance 120 30 m = 120 + 30 * m := by
  sorry

#check rosie_account_balance

end rosie_account_balance_l3019_301974


namespace sum_160_45_base4_l3019_301916

/-- Convert a decimal number to base 4 -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Convert a list of base 4 digits to decimal -/
def fromBase4 (l : List ℕ) : ℕ :=
  sorry

/-- Add two numbers in base 4 -/
def addBase4 (a b : List ℕ) : List ℕ :=
  sorry

theorem sum_160_45_base4 :
  addBase4 (toBase4 160) (toBase4 45) = [2, 4, 3, 1] := by
  sorry

end sum_160_45_base4_l3019_301916


namespace point_on_line_l3019_301976

/-- A point represented by its x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line : 
  let p1 : Point := ⟨0, 4⟩
  let p2 : Point := ⟨-6, 1⟩
  let p3 : Point := ⟨6, 7⟩
  collinear p1 p2 p3 := by
  sorry

end point_on_line_l3019_301976


namespace smallest_w_divisible_by_13_and_w_plus_3_divisible_by_11_l3019_301944

theorem smallest_w_divisible_by_13_and_w_plus_3_divisible_by_11 : 
  ∃ w : ℕ, w > 0 ∧ w % 13 = 0 ∧ (w + 3) % 11 = 0 ∧
  ∀ x : ℕ, x > 0 ∧ x % 13 = 0 ∧ (x + 3) % 11 = 0 → w ≤ x :=
by sorry

end smallest_w_divisible_by_13_and_w_plus_3_divisible_by_11_l3019_301944


namespace lcm_36_100_l3019_301915

theorem lcm_36_100 : Nat.lcm 36 100 = 900 := by
  sorry

end lcm_36_100_l3019_301915


namespace distributive_property_only_true_l3019_301942

open Real

theorem distributive_property_only_true : ∀ b x y : ℝ,
  (b * (x + y) = b * x + b * y) ∧
  (b^(x + y) ≠ b^x + b^y) ∧
  (log (x + y) ≠ log x + log y) ∧
  (log x / log y ≠ log x - log y) ∧
  (b * (x / y) ≠ b * x / (b * y)) :=
by sorry

end distributive_property_only_true_l3019_301942


namespace quadratic_no_roots_l3019_301913

/-- Given a quadratic function f(x) = ax^2 + bx + c where b is the geometric mean of a and c,
    prove that f(x) has no real roots. -/
theorem quadratic_no_roots (a b c : ℝ) (h : b^2 = a*c) (h_a : a ≠ 0) (h_c : c ≠ 0) :
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 :=
sorry

end quadratic_no_roots_l3019_301913


namespace rhombus_side_length_l3019_301903

-- Define a rhombus with area K and diagonals d and 3d
structure Rhombus where
  K : ℝ  -- Area of the rhombus
  d : ℝ  -- Length of the shorter diagonal
  h : K = (3/2) * d^2  -- Area formula for rhombus

-- Theorem: The side length of the rhombus is sqrt(5K/3)
theorem rhombus_side_length (r : Rhombus) : 
  ∃ s : ℝ, s^2 = (5/3) * r.K ∧ s > 0 := by
  sorry

end rhombus_side_length_l3019_301903


namespace power_multiplication_l3019_301927

theorem power_multiplication (x : ℝ) : x^4 * x^2 = x^6 := by
  sorry

end power_multiplication_l3019_301927


namespace sin_2theta_value_l3019_301980

def line1 (θ : ℝ) : ℝ → ℝ → Prop :=
  fun x y ↦ x * Real.cos θ + 2 * y = 0

def line2 (θ : ℝ) : ℝ → ℝ → Prop :=
  fun x y ↦ 3 * x + y * Real.sin θ + 3 = 0

def perpendicular (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ x₁ y₁ x₂ y₂, f x₁ y₁ ∧ g x₂ y₂ ∧ 
    (y₂ - y₁) * (x₂ - x₁) + (x₂ - x₁) * (y₂ - y₁) = 0

theorem sin_2theta_value (θ : ℝ) :
  perpendicular (line1 θ) (line2 θ) → Real.sin (2 * θ) = -12/13 := by
  sorry

end sin_2theta_value_l3019_301980


namespace percent_equality_l3019_301933

theorem percent_equality (x : ℝ) (h : 0.30 * 0.15 * x = 45) : 0.15 * 0.30 * x = 45 := by
  sorry

end percent_equality_l3019_301933


namespace select_with_boys_l3019_301929

theorem select_with_boys (num_boys num_girls : ℕ) : 
  num_boys = 6 → num_girls = 4 → 
  (2^(num_boys + num_girls) - 2^num_girls) = 1008 := by
  sorry

end select_with_boys_l3019_301929


namespace geometric_sequence_a3_value_l3019_301978

def is_geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ r : ℤ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a3_value
  (a : ℕ → ℤ)
  (h_geometric : is_geometric_sequence a)
  (h_product : a 2 * a 5 = -32)
  (h_sum : a 3 + a 4 = 4)
  (h_integer_ratio : ∃ r : ℤ, ∀ n : ℕ, a (n + 1) = a n * r) :
  a 3 = -4 :=
sorry

end geometric_sequence_a3_value_l3019_301978


namespace joe_taller_than_roy_l3019_301984

/-- Given the heights of Sara and Roy, and the difference between Sara and Joe's heights,
    prove that Joe is 3 inches taller than Roy. -/
theorem joe_taller_than_roy (sara_height joe_height roy_height : ℕ)
  (h1 : sara_height = 45)
  (h2 : sara_height = joe_height + 6)
  (h3 : roy_height = 36) :
  joe_height - roy_height = 3 :=
by sorry

end joe_taller_than_roy_l3019_301984


namespace sum_of_roots_equals_one_l3019_301930

theorem sum_of_roots_equals_one :
  ∀ x y : ℝ, (x + 3) * (x - 4) = 18 ∧ (y + 3) * (y - 4) = 18 → x + y = 1 :=
by
  sorry

end sum_of_roots_equals_one_l3019_301930


namespace pink_ratio_theorem_l3019_301981

/-- Given a class with the following properties:
  * There are 30 students in total
  * There are 18 girls in the class
  * Half of the class likes green
  * 9 students like yellow
  * The remaining students like pink (all of whom are girls)
  Then the ratio of girls who like pink to the total number of girls is 1/3 -/
theorem pink_ratio_theorem (total_students : ℕ) (total_girls : ℕ) (yellow_fans : ℕ) :
  total_students = 30 →
  total_girls = 18 →
  yellow_fans = 9 →
  (total_students / 2 + yellow_fans + (total_girls - (total_students - total_students / 2 - yellow_fans)) = total_students) →
  (total_girls - (total_students - total_students / 2 - yellow_fans)) / total_girls = 1 / 3 := by
  sorry

end pink_ratio_theorem_l3019_301981


namespace geometric_sequence_general_term_l3019_301953

/-- Represents a geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_general_term
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_sum1 : a 1 + a 3 = 5/2)
  (h_sum2 : a 2 + a 4 = 5/4) :
  ∀ n : ℕ, a n = 2^(2-n) :=
sorry

end geometric_sequence_general_term_l3019_301953


namespace contractor_male_workers_l3019_301965

/-- Proves that the number of male workers is 20 given the conditions of the problem -/
theorem contractor_male_workers :
  let female_workers : ℕ := 15
  let child_workers : ℕ := 5
  let male_wage : ℚ := 25
  let female_wage : ℚ := 20
  let child_wage : ℚ := 8
  let average_wage : ℚ := 21
  ∃ male_workers : ℕ,
    (male_wage * male_workers + female_wage * female_workers + child_wage * child_workers) /
    (male_workers + female_workers + child_workers) = average_wage ∧
    male_workers = 20 :=
by
  sorry


end contractor_male_workers_l3019_301965


namespace append_nine_to_two_digit_number_l3019_301973

/-- Given a two-digit number with tens digit t and units digit u,
    appending 9 to the end results in the number 100t + 10u + 9 -/
theorem append_nine_to_two_digit_number (t u : ℕ) (h : t ≤ 9 ∧ u ≤ 9) :
  (10 * t + u) * 10 + 9 = 100 * t + 10 * u + 9 := by
  sorry

end append_nine_to_two_digit_number_l3019_301973


namespace quadratic_equation_solutions_l3019_301904

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_equation_solutions 
  (a b c : ℝ) 
  (h1 : f a b c (-2) = 3)
  (h2 : f a b c (-1) = 4)
  (h3 : f a b c 0 = 3)
  (h4 : f a b c 1 = 0)
  (h5 : f a b c 2 = -5) :
  ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = -4 ∧ f a b c x₁ = -5 ∧ f a b c x₂ = -5 :=
by sorry

end quadratic_equation_solutions_l3019_301904


namespace quadratic_equation_and_expression_calculation_l3019_301934

theorem quadratic_equation_and_expression_calculation :
  (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 7 ∧ x₂ = 2 - Real.sqrt 7 ∧
    x₁^2 - 4*x₁ - 3 = 0 ∧ x₂^2 - 4*x₂ - 3 = 0) ∧
  (|-3| - 4 * Real.sin (π/4) + Real.sqrt 8 + (π - 3)^0 = 4) := by
  sorry

end quadratic_equation_and_expression_calculation_l3019_301934


namespace lenny_remaining_amount_l3019_301940

def calculate_remaining_amount (initial_amount : ℝ) 
  (console_price game_price headphones_price : ℝ)
  (book1_price book2_price book3_price : ℝ)
  (tech_discount tech_tax bookstore_fee : ℝ) : ℝ :=
  let tech_total := console_price + 2 * game_price + headphones_price
  let tech_discounted := tech_total * (1 - tech_discount)
  let tech_with_tax := tech_discounted * (1 + tech_tax)
  let book_total := book1_price + book2_price
  let bookstore_total := book_total * (1 + bookstore_fee)
  let total_spent := tech_with_tax + bookstore_total
  initial_amount - total_spent

theorem lenny_remaining_amount :
  calculate_remaining_amount 500 200 50 75 25 30 15 0.2 0.1 0.02 = 113.90 := by
  sorry

end lenny_remaining_amount_l3019_301940


namespace total_marbles_is_72_marble_ratio_is_2_4_6_l3019_301952

/-- Represents the number of marbles of each color in a bag -/
structure MarbleBag where
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Defines the properties of the marble bag based on the given conditions -/
def special_marble_bag : MarbleBag :=
  { red := 12,
    blue := 24,
    yellow := 36 }

/-- Theorem stating that the total number of marbles in the special bag is 72 -/
theorem total_marbles_is_72 :
  special_marble_bag.red + special_marble_bag.blue + special_marble_bag.yellow = 72 := by
  sorry

/-- Theorem stating that the ratio of marbles in the special bag is 2:4:6 -/
theorem marble_ratio_is_2_4_6 :
  2 * special_marble_bag.red = special_marble_bag.blue ∧
  3 * special_marble_bag.red = special_marble_bag.yellow := by
  sorry

end total_marbles_is_72_marble_ratio_is_2_4_6_l3019_301952


namespace prob_all_red_first_is_half_l3019_301906

/-- The number of red chips in the hat -/
def num_red_chips : ℕ := 3

/-- The number of green chips in the hat -/
def num_green_chips : ℕ := 3

/-- The total number of chips in the hat -/
def total_chips : ℕ := num_red_chips + num_green_chips

/-- The probability of drawing all red chips before all green chips -/
def prob_all_red_first : ℚ :=
  (Nat.choose (total_chips - 1) num_green_chips) / (Nat.choose total_chips num_red_chips)

/-- Theorem stating that the probability of drawing all red chips first is 1/2 -/
theorem prob_all_red_first_is_half : prob_all_red_first = 1 / 2 := by
  sorry

end prob_all_red_first_is_half_l3019_301906


namespace system_solutions_l3019_301949

/-- The system of equations has only three real solutions -/
theorem system_solutions (a b c : ℝ) : 
  (2 * a - b = a^2 * b) ∧ 
  (2 * b - c = b^2 * c) ∧ 
  (2 * c - a = c^2 * a) → 
  ((a = -1 ∧ b = -1 ∧ c = -1) ∨ 
   (a = 0 ∧ b = 0 ∧ c = 0) ∨ 
   (a = 1 ∧ b = 1 ∧ c = 1)) :=
by sorry

end system_solutions_l3019_301949


namespace set_operations_l3019_301971

def A : Set ℤ := {x | |x| ≤ 6}
def B : Set ℤ := {1, 2, 3}
def C : Set ℤ := {3, 4, 5, 6}

theorem set_operations :
  (A ∩ (B ∩ C) = {3}) ∧
  (A ∩ (A \ (B ∪ C)) = {-6, -5, -4, -3, -2, -1, 0}) := by sorry

end set_operations_l3019_301971


namespace smaller_box_size_l3019_301947

/-- Represents the size and cost of a box of macaroni and cheese -/
structure MacaroniBox where
  size : Float
  cost : Float

/-- Calculates the price per ounce of a MacaroniBox -/
def pricePerOunce (box : MacaroniBox) : Float :=
  box.cost / box.size

theorem smaller_box_size 
  (larger_box : MacaroniBox)
  (smaller_box : MacaroniBox)
  (better_value_price : Float)
  (h1 : larger_box.size = 30)
  (h2 : larger_box.cost = 4.80)
  (h3 : smaller_box.cost = 3.40)
  (h4 : better_value_price = 0.16)
  (h5 : pricePerOunce larger_box ≤ pricePerOunce smaller_box)
  (h6 : pricePerOunce larger_box = better_value_price) :
  smaller_box.size = 21.25 := by
  sorry

#check smaller_box_size

end smaller_box_size_l3019_301947


namespace carpet_width_l3019_301994

/-- Proves that given a room 15 meters long and 6 meters wide, carpeted at a cost of 30 paise per meter for a total of Rs. 36, the width of the carpet used is 800 centimeters. -/
theorem carpet_width (room_length : ℝ) (room_breadth : ℝ) (carpet_cost_paise : ℝ) (total_cost_rupees : ℝ) :
  room_length = 15 →
  room_breadth = 6 →
  carpet_cost_paise = 30 →
  total_cost_rupees = 36 →
  ∃ (carpet_width : ℝ), carpet_width = 800 := by
  sorry

end carpet_width_l3019_301994


namespace sibling_ages_sum_l3019_301987

theorem sibling_ages_sum (a b c : ℕ+) : 
  a = b ∧ a < c ∧ a * b * c = 72 → a + b + c = 14 :=
by sorry

end sibling_ages_sum_l3019_301987


namespace eighth_term_of_sequence_l3019_301990

theorem eighth_term_of_sequence (x : ℝ) : 
  let nth_term (n : ℕ) := (-1)^(n+1) * ((n^2 + 1) : ℝ) * x^n
  nth_term 8 = -65 * x^8 := by sorry

end eighth_term_of_sequence_l3019_301990


namespace average_difference_l3019_301943

theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 110) 
  (h2 : (b + c) / 2 = 150) : 
  a - c = -80 := by
sorry

end average_difference_l3019_301943


namespace smallest_isosceles_perimeter_square_l3019_301963

/-- A number is composite if it's greater than 1 and not prime -/
def IsComposite (n : ℕ) : Prop := n > 1 ∧ ¬ Nat.Prime n

/-- A natural number is a perfect square if it's equal to some integer squared -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- The perimeter of an isosceles triangle with two sides of length a and one side of length b -/
def IsoscelesPerimeter (a b : ℕ) : ℕ := 2 * a + b

theorem smallest_isosceles_perimeter_square : 
  ∀ a b : ℕ, 
    IsComposite a → 
    IsComposite b → 
    a ≠ b → 
    IsPerfectSquare ((2 * a + b) * (2 * a + b)) → 
    2 * a > b → 
    a + b > a →
    ∀ c d : ℕ, 
      IsComposite c → 
      IsComposite d → 
      c ≠ d → 
      IsPerfectSquare ((2 * c + d) * (2 * c + d)) → 
      2 * c > d → 
      c + d > c →
      (IsoscelesPerimeter a b) * (IsoscelesPerimeter a b) ≤ (IsoscelesPerimeter c d) * (IsoscelesPerimeter c d) → 
      (IsoscelesPerimeter a b) * (IsoscelesPerimeter a b) = 256 :=
by sorry

end smallest_isosceles_perimeter_square_l3019_301963


namespace required_moles_of_reactants_l3019_301908

-- Define the chemical reaction
structure Reaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String

-- Define the molar ratio
def molarRatio : ℚ := 1

-- Define the desired amount of products
def desiredProduct : ℚ := 3

-- Define the chemical equation
def chemicalEquation : Reaction := {
  reactant1 := "AgNO3"
  reactant2 := "NaOH"
  product1 := "AgOH"
  product2 := "NaNO3"
}

-- Theorem statement
theorem required_moles_of_reactants :
  let requiredReactant1 := desiredProduct * molarRatio
  let requiredReactant2 := desiredProduct * molarRatio
  requiredReactant1 = 3 ∧ requiredReactant2 = 3 :=
sorry

end required_moles_of_reactants_l3019_301908


namespace qz_length_l3019_301919

/-- A quadrilateral ABZY with a point Q on the intersection of AZ and BY -/
structure Quadrilateral :=
  (A B Y Z Q : ℝ × ℝ)
  (AB_parallel_YZ : (A.2 - B.2) / (A.1 - B.1) = (Y.2 - Z.2) / (Y.1 - Z.1))
  (Q_on_AZ : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • A + t • Z)
  (Q_on_BY : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ Q = (1 - s) • B + s • Y)
  (AZ_length : Real.sqrt ((A.1 - Z.1)^2 + (A.2 - Z.2)^2) = 42)
  (BQ_length : Real.sqrt ((B.1 - Q.1)^2 + (B.2 - Q.2)^2) = 12)
  (QY_length : Real.sqrt ((Q.1 - Y.1)^2 + (Q.2 - Y.2)^2) = 24)

/-- The length of QZ in the given quadrilateral is 28 -/
theorem qz_length (quad : Quadrilateral) :
  Real.sqrt ((quad.Q.1 - quad.Z.1)^2 + (quad.Q.2 - quad.Z.2)^2) = 28 := by
  sorry

end qz_length_l3019_301919


namespace purchase_price_calculation_l3019_301948

theorem purchase_price_calculation (P : ℝ) : 0.05 * P + 12 = 30 → P = 360 := by
  sorry

end purchase_price_calculation_l3019_301948


namespace spinner_probability_l3019_301918

theorem spinner_probability (pA pB pC pD : ℚ) : 
  pA = 1/4 →
  pB = 1/3 →
  pA + pB + pC + pD = 1 →
  pD = 1/4 := by
sorry

end spinner_probability_l3019_301918


namespace same_solution_implies_a_plus_b_equals_one_l3019_301946

theorem same_solution_implies_a_plus_b_equals_one 
  (x y a b : ℝ) 
  (h1 : 2*x + 4*y = 20) 
  (h2 : a*x + b*y = 1)
  (h3 : 2*x - y = 5)
  (h4 : b*x + a*y = 6)
  (h5 : 2*x + 4*y = 20 ∧ a*x + b*y = 1 ↔ 2*x - y = 5 ∧ b*x + a*y = 6) : 
  a + b = 1 := by
sorry


end same_solution_implies_a_plus_b_equals_one_l3019_301946


namespace min_value_2x_plus_y_l3019_301935

/-- The minimum value of 2x + y given the constraints |y| ≤ 2 - x and x ≥ -1 is -5 -/
theorem min_value_2x_plus_y (x y : ℝ) (h1 : |y| ≤ 2 - x) (h2 : x ≥ -1) : 
  ∃ (m : ℝ), m = -5 ∧ ∀ (x' y' : ℝ), |y'| ≤ 2 - x' → x' ≥ -1 → 2*x' + y' ≥ m :=
by sorry

end min_value_2x_plus_y_l3019_301935


namespace inverse_proportion_problem_l3019_301997

/-- Given that a and b are inversely proportional, their sum is 40, and their modified difference is 10, prove that b equals 75 when a equals 4. -/
theorem inverse_proportion_problem (a b : ℝ) (k : ℝ) (h1 : a * b = k) (h2 : a + b = 40) (h3 : a - 2*b = 10) : 
  a = 4 → b = 75 := by
  sorry

end inverse_proportion_problem_l3019_301997


namespace coin_and_die_probability_l3019_301905

theorem coin_and_die_probability :
  let coin_prob : ℚ := 2/3  -- Probability of heads for the biased coin
  let die_prob : ℚ := 1/6   -- Probability of rolling a 5 on a fair six-sided die
  coin_prob * die_prob = 1/9 := by sorry

end coin_and_die_probability_l3019_301905


namespace decagon_painting_count_l3019_301902

/-- The number of ways to choose 4 colors from 8 available colors -/
def choose_colors : ℕ := Nat.choose 8 4

/-- The number of circular permutations of 4 colors -/
def circular_permutations : ℕ := Nat.factorial 3

/-- The number of distinct colorings of a decagon -/
def decagon_colorings : ℕ := choose_colors * circular_permutations / 2

/-- Theorem stating the number of distinct ways to paint the decagon -/
theorem decagon_painting_count : decagon_colorings = 210 := by
  sorry

#eval decagon_colorings

end decagon_painting_count_l3019_301902


namespace functional_inequality_solution_l3019_301977

/-- A function satisfying the given inequality condition -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x + y + z) + f x ≥ f (x + y) + f (x + z)

/-- The main theorem statement -/
theorem functional_inequality_solution
    (f : ℝ → ℝ)
    (h_diff : Differentiable ℝ f)
    (h_ineq : SatisfiesInequality f) :
    ∃ a b : ℝ, ∀ x, f x = a * x + b :=
  sorry

end functional_inequality_solution_l3019_301977


namespace midpoint_distance_midpoint_path_l3019_301941

/-- Represents a ladder sliding down a wall --/
structure SlidingLadder where
  L : ℝ  -- Length of the ladder
  x : ℝ  -- Horizontal distance from wall to bottom of ladder
  y : ℝ  -- Vertical distance from floor to top of ladder
  h_positive : L > 0  -- Ladder has positive length
  h_pythagorean : x^2 + y^2 = L^2  -- Pythagorean theorem

/-- The midpoint of a sliding ladder is always L/2 distance from the corner --/
theorem midpoint_distance (ladder : SlidingLadder) :
  (ladder.x / 2)^2 + (ladder.y / 2)^2 = (ladder.L / 2)^2 := by
  sorry

/-- The path of the midpoint forms a quarter circle --/
theorem midpoint_path (ladder : SlidingLadder) :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (0, 0) ∧ 
    radius = ladder.L / 2 ∧
    (ladder.x / 2)^2 + (ladder.y / 2)^2 = radius^2 := by
  sorry

end midpoint_distance_midpoint_path_l3019_301941


namespace exists_k_for_A_l3019_301936

theorem exists_k_for_A (M : ℕ) (hM : M > 2) :
  ∃ k : ℕ, ((M + Real.sqrt (M^2 - 4 : ℝ)) / 2)^5 = (k + Real.sqrt (k^2 - 4 : ℝ)) / 2 :=
by sorry

end exists_k_for_A_l3019_301936


namespace lineup_selections_15_l3019_301992

/-- The number of ways to select an ordered lineup of 5 players and 1 substitute from 15 players -/
def lineup_selections (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2) * (n - 3) * (n - 4) * (n - 5)

/-- Theorem stating that the number of lineup selections from 15 players is 3,276,000 -/
theorem lineup_selections_15 :
  lineup_selections 15 = 3276000 := by
  sorry

#eval lineup_selections 15

end lineup_selections_15_l3019_301992


namespace original_speed_correct_l3019_301964

/-- The original speed of the car traveling between two locations. -/
def original_speed : ℝ := 80

/-- The distance between location A and location B in kilometers. -/
def distance : ℝ := 160

/-- The increase in speed as a percentage. -/
def speed_increase : ℝ := 0.25

/-- The time saved due to the increased speed, in hours. -/
def time_saved : ℝ := 0.4

/-- Theorem stating that the original speed satisfies the given conditions. -/
theorem original_speed_correct :
  distance / original_speed - distance / (original_speed * (1 + speed_increase)) = time_saved := by
  sorry

end original_speed_correct_l3019_301964


namespace triangle_side_length_l3019_301972

theorem triangle_side_length (D E F : ℝ) : 
  -- Triangle DEF exists
  (0 < D) → (0 < E) → (0 < F) → 
  (D + E > F) → (D + F > E) → (E + F > D) →
  -- Given conditions
  (E = 45 * π / 180) →  -- Convert 45° to radians
  (D = 100) →
  (F = 100 * Real.sqrt 2) →
  -- Conclusion
  (E = Real.sqrt (30000 + 5000 * (Real.sqrt 6 - Real.sqrt 2))) :=
by sorry

end triangle_side_length_l3019_301972


namespace sqrt_five_exists_and_unique_l3019_301909

theorem sqrt_five_exists_and_unique :
  ∃! (y : ℝ), y > 0 ∧ y^2 = 5 :=
by sorry

end sqrt_five_exists_and_unique_l3019_301909


namespace max_m_value_l3019_301993

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.exp (2*x) - x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (x - m) * f x - (1/4) * Real.exp (2*x) + x^2 + x

theorem max_m_value (m : ℤ) :
  (∀ x > 0, Monotone (g m)) →
  m ≤ 1 ∧ ∃ m' : ℤ, m' = 1 ∧ (∀ x > 0, Monotone (g m')) :=
sorry

end max_m_value_l3019_301993


namespace tangent_line_equation_l3019_301989

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 + x + 1

/-- The derivative of the parabola function -/
def f' (x : ℝ) : ℝ := 2*x + 1

/-- The point through which the tangent line passes -/
def P : ℝ × ℝ := (-1, 0)

/-- Theorem: The tangent line to y = x^2 + x + 1 passing through (-1, 0) is x - y + 1 = 0 -/
theorem tangent_line_equation :
  ∃ (x₀ : ℝ), 
    let y₀ := f x₀
    let m := f' x₀
    (P.1 - x₀) * m = P.2 - y₀ ∧
    ∀ (x y : ℝ), y = m * (x - x₀) + y₀ ↔ x - y + 1 = 0 :=
sorry

end tangent_line_equation_l3019_301989


namespace fraction_product_simplification_l3019_301924

theorem fraction_product_simplification :
  (2 / 3) * (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 2 / 7 := by
  sorry

end fraction_product_simplification_l3019_301924


namespace outfit_combinations_l3019_301907

theorem outfit_combinations (shirts : ℕ) (hats : ℕ) : shirts = 5 → hats = 3 → shirts * hats = 15 := by
  sorry

end outfit_combinations_l3019_301907


namespace hour_hand_angle_for_9_to_1_ratio_l3019_301951

/-- Represents a toy clock with a specific ratio between hour and minute hand rotations -/
structure ToyClock where
  /-- The number of full circles the minute hand makes for each full circle of the hour hand -/
  minuteToHourRatio : ℕ
  /-- Assumption that the ratio is greater than 1 -/
  ratioGtOne : minuteToHourRatio > 1

/-- Calculates the angle turned by the hour hand when it next coincides with the minute hand -/
def hourHandAngleAtNextCoincidence (clock : ToyClock) : ℚ :=
  360 / (clock.minuteToHourRatio - 1)

/-- Theorem stating that for a toy clock where the minute hand makes 9 circles 
    for each full circle of the hour hand, the hour hand turns 45° at the next coincidence -/
theorem hour_hand_angle_for_9_to_1_ratio :
  let clock : ToyClock := ⟨9, by norm_num⟩
  hourHandAngleAtNextCoincidence clock = 45 := by
  sorry

end hour_hand_angle_for_9_to_1_ratio_l3019_301951


namespace oranges_per_crate_l3019_301912

theorem oranges_per_crate :
  ∀ (num_crates num_boxes nectarines_per_box total_fruit : ℕ),
    num_crates = 12 →
    num_boxes = 16 →
    nectarines_per_box = 30 →
    total_fruit = 2280 →
    total_fruit = num_boxes * nectarines_per_box + num_crates * (total_fruit - num_boxes * nectarines_per_box) / num_crates →
    (total_fruit - num_boxes * nectarines_per_box) / num_crates = 150 :=
by
  sorry

end oranges_per_crate_l3019_301912


namespace comparison_theorem_l3019_301955

theorem comparison_theorem :
  (-7/8 : ℚ) < (-6/7 : ℚ) ∧ |(-0.1 : ℝ)| > (-0.2 : ℝ) := by
  sorry

end comparison_theorem_l3019_301955


namespace race_distance_proof_l3019_301979

/-- The distance of a dogsled race course in Wyoming --/
def race_distance : ℝ := 300

/-- The average speed of Team R in mph --/
def team_r_speed : ℝ := 20

/-- The time difference between Team A and Team R in hours --/
def time_difference : ℝ := 3

/-- The speed difference between Team A and Team R in mph --/
def speed_difference : ℝ := 5

/-- Theorem stating the race distance given the conditions --/
theorem race_distance_proof :
  ∃ (t : ℝ), 
    t > 0 ∧
    race_distance = team_r_speed * t ∧
    race_distance = (team_r_speed + speed_difference) * (t - time_difference) :=
by
  sorry

#check race_distance_proof

end race_distance_proof_l3019_301979


namespace factor_expression_l3019_301931

theorem factor_expression (x : ℝ) : 2*x*(x+3) + (x+3) = (2*x+1)*(x+3) := by
  sorry

end factor_expression_l3019_301931


namespace expression_simplification_l3019_301917

theorem expression_simplification (x : ℝ) (h : x = 1 + Real.sqrt 3) :
  (x + 3) / (x^2 - 2*x + 1) * (x - 1) / (x^2 + 3*x) + 1 / x = Real.sqrt 3 / 3 := by
  sorry

end expression_simplification_l3019_301917


namespace pi_arrangement_face_dots_l3019_301959

/-- Represents a cube with dots on its faces -/
structure Cube where
  face1 : Nat
  face2 : Nat
  face3 : Nat
  face4 : Nat
  face5 : Nat
  face6 : Nat
  three_dot_face : face1 = 3 ∨ face2 = 3 ∨ face3 = 3 ∨ face4 = 3 ∨ face5 = 3 ∨ face6 = 3
  two_dot_faces : (face1 = 2 ∧ face2 = 2) ∨ (face1 = 2 ∧ face3 = 2) ∨ (face1 = 2 ∧ face4 = 2) ∨
                  (face1 = 2 ∧ face5 = 2) ∨ (face1 = 2 ∧ face6 = 2) ∨ (face2 = 2 ∧ face3 = 2) ∨
                  (face2 = 2 ∧ face4 = 2) ∨ (face2 = 2 ∧ face5 = 2) ∨ (face2 = 2 ∧ face6 = 2) ∨
                  (face3 = 2 ∧ face4 = 2) ∨ (face3 = 2 ∧ face5 = 2) ∨ (face3 = 2 ∧ face6 = 2) ∨
                  (face4 = 2 ∧ face5 = 2) ∨ (face4 = 2 ∧ face6 = 2) ∨ (face5 = 2 ∧ face6 = 2)
  one_dot_faces : face1 + face2 + face3 + face4 + face5 + face6 = 9

/-- Represents the "П" shape arrangement of cubes -/
structure PiArrangement where
  cubes : Fin 7 → Cube
  contacting_faces_same : ∀ i j, i ≠ j → (cubes i).face1 = (cubes j).face2

/-- The theorem to be proved -/
theorem pi_arrangement_face_dots (arr : PiArrangement) :
  ∃ (a b c : Cube), (a.face1 = 2 ∧ b.face1 = 2 ∧ c.face1 = 3) :=
sorry

end pi_arrangement_face_dots_l3019_301959


namespace jeremy_age_l3019_301937

theorem jeremy_age (amy jeremy chris : ℕ) 
  (h1 : amy + jeremy + chris = 132)
  (h2 : amy = jeremy / 3)
  (h3 : chris = 2 * amy) :
  jeremy = 66 := by
sorry

end jeremy_age_l3019_301937


namespace complement_M_U_characterization_l3019_301960

-- Define the universal set U
def U : Set Int := {x | ∃ k, x = 2 * k}

-- Define the set M
def M : Set Int := {x | ∃ k, x = 4 * k}

-- Define the complement of M with respect to U
def complement_M_U : Set Int := {x ∈ U | x ∉ M}

-- Theorem statement
theorem complement_M_U_characterization :
  complement_M_U = {x | ∃ k, x = 4 * k - 2} := by sorry

end complement_M_U_characterization_l3019_301960


namespace equivalent_statements_l3019_301914

variable (P Q : Prop)

theorem equivalent_statements : 
  ((P → Q) ↔ (¬Q → ¬P)) ∧ ((P → Q) ↔ (¬P ∨ Q)) := by sorry

end equivalent_statements_l3019_301914


namespace quadratic_root_value_l3019_301962

theorem quadratic_root_value (r s : ℝ) : 
  (∃ x : ℂ, 2 * x^2 + r * x + s = 0 ∧ x = 3 + 2*I) → s = 26 := by
  sorry

end quadratic_root_value_l3019_301962


namespace extra_calories_burned_l3019_301991

def calories_per_hour : ℕ := 30

def calories_burned (hours : ℕ) : ℕ := hours * calories_per_hour

theorem extra_calories_burned : calories_burned 5 - calories_burned 2 = 90 := by
  sorry

end extra_calories_burned_l3019_301991


namespace point_on_linear_graph_l3019_301982

/-- For any point (a, b) on the graph of y = 2x - 1, 2a - b + 1 = 2 -/
theorem point_on_linear_graph (a b : ℝ) (h : b = 2 * a - 1) : 2 * a - b + 1 = 2 := by
  sorry

end point_on_linear_graph_l3019_301982
