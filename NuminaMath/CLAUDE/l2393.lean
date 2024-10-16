import Mathlib

namespace NUMINAMATH_CALUDE_special_polygon_interior_sum_special_polygon_exists_l2393_239394

/-- A polygon where each interior angle is 7.5 times its corresponding exterior angle -/
structure SpecialPolygon where
  n : ℕ  -- number of sides
  interior_angle : ℝ  -- measure of each interior angle
  h_interior_exterior : interior_angle = 7.5 * (360 / n)  -- relation between interior and exterior angles

/-- The sum of interior angles of a SpecialPolygon is 2700° -/
theorem special_polygon_interior_sum (P : SpecialPolygon) : 
  P.n * P.interior_angle = 2700 := by
  sorry

/-- A SpecialPolygon with 17 sides exists -/
theorem special_polygon_exists : 
  ∃ P : SpecialPolygon, P.n = 17 := by
  sorry

end NUMINAMATH_CALUDE_special_polygon_interior_sum_special_polygon_exists_l2393_239394


namespace NUMINAMATH_CALUDE_carpet_cost_proof_l2393_239345

theorem carpet_cost_proof (floor_length floor_width carpet_side_length carpet_cost : ℝ) 
  (h1 : floor_length = 24)
  (h2 : floor_width = 64)
  (h3 : carpet_side_length = 8)
  (h4 : carpet_cost = 24) : 
  (floor_length * floor_width) / (carpet_side_length * carpet_side_length) * carpet_cost = 576 := by
  sorry

end NUMINAMATH_CALUDE_carpet_cost_proof_l2393_239345


namespace NUMINAMATH_CALUDE_rectangle_area_l2393_239305

theorem rectangle_area (L W : ℝ) (h1 : 2 * L + W = 34) (h2 : L + 2 * W = 38) :
  L * W = 140 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2393_239305


namespace NUMINAMATH_CALUDE_custom_op_two_three_custom_op_nested_l2393_239387

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := a^2 - b + a*b

-- Theorem 1: 2 * 3 = 7
theorem custom_op_two_three : custom_op 2 3 = 7 := by sorry

-- Theorem 2: (-2) * [2 * (-3)] = 1
theorem custom_op_nested : custom_op (-2) (custom_op 2 (-3)) = 1 := by sorry

end NUMINAMATH_CALUDE_custom_op_two_three_custom_op_nested_l2393_239387


namespace NUMINAMATH_CALUDE_inequality_proof_l2393_239344

theorem inequality_proof (x y k : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x ≠ y) 
  (h4 : k > 0) 
  (h5 : k < 2) : 
  ((x + y) / 2) ^ k > (Real.sqrt (x * y)) ^ k ∧ 
  (Real.sqrt (x * y)) ^ k > (2 * x * y / (x + y)) ^ k := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2393_239344


namespace NUMINAMATH_CALUDE_monkey_peaches_l2393_239337

/-- Represents the number of peaches each monkey gets -/
structure MonkeyShares :=
  (eldest : ℕ)
  (second : ℕ)
  (third : ℕ)

/-- The problem statement -/
theorem monkey_peaches (total : ℕ) (shares : MonkeyShares) : shares.second = 20 :=
  sorry

/-- Conditions of the problem -/
axiom divide_ratio (n m : ℕ) : n / (n + m) = 5 / 9
axiom eldest_share (total : ℕ) (shares : MonkeyShares) : shares.eldest = (total * 5) / 9
axiom second_share (total : ℕ) (shares : MonkeyShares) : 
  shares.second = ((total - shares.eldest) * 5) / 9
axiom third_share (total : ℕ) (shares : MonkeyShares) : 
  shares.third = total - shares.eldest - shares.second
axiom eldest_third_difference (shares : MonkeyShares) : shares.eldest - shares.third = 29

end NUMINAMATH_CALUDE_monkey_peaches_l2393_239337


namespace NUMINAMATH_CALUDE_harmonious_point_in_third_quadrant_l2393_239341

/-- A point (x, y) is harmonious if 3x = 2y + 5 -/
def IsHarmonious (x y : ℝ) : Prop := 3 * x = 2 * y + 5

/-- The x-coordinate of point M -/
def Mx (m : ℝ) : ℝ := m - 1

/-- The y-coordinate of point M -/
def My (m : ℝ) : ℝ := 3 * m + 2

theorem harmonious_point_in_third_quadrant :
  ∀ m : ℝ, IsHarmonious (Mx m) (My m) → Mx m < 0 ∧ My m < 0 := by
  sorry

end NUMINAMATH_CALUDE_harmonious_point_in_third_quadrant_l2393_239341


namespace NUMINAMATH_CALUDE_cubic_function_properties_l2393_239307

-- Define the function f
def f (m n x : ℝ) : ℝ := m * x^3 + n * x^2

-- Define the derivative of f
def f' (m n x : ℝ) : ℝ := 3 * m * x^2 + 2 * n * x

-- Theorem statement
theorem cubic_function_properties (m : ℝ) (h : m ≠ 0) :
  ∃ n : ℝ,
    f' m n 2 = 0 ∧
    n = -3 * m ∧
    (∀ x : ℝ, m > 0 → (x < 0 ∨ x > 2) → (f' m n x > 0)) ∧
    (∀ x : ℝ, m < 0 → (x > 0 ∧ x < 2) → (f' m n x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l2393_239307


namespace NUMINAMATH_CALUDE_smallest_coin_set_l2393_239342

/-- Represents the different types of coins available --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- The value of each coin in cents --/
def coinValue : Coin → ℕ
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25
  | Coin.HalfDollar => 50

/-- A function that checks if a given set of coins can pay any amount from 1 to n cents --/
def canPayAllAmounts (coins : List Coin) (n : ℕ) : Prop :=
  ∀ (amount : ℕ), 1 ≤ amount ∧ amount ≤ n →
    ∃ (subset : List Coin), subset ⊆ coins ∧ (subset.map coinValue).sum = amount

/-- The main theorem stating that 10 is the smallest number of coins needed --/
theorem smallest_coin_set :
  ∃ (coins : List Coin),
    coins.length = 10 ∧
    canPayAllAmounts coins 149 ∧
    ∀ (other_coins : List Coin),
      canPayAllAmounts other_coins 149 →
      other_coins.length ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_coin_set_l2393_239342


namespace NUMINAMATH_CALUDE_number_division_problem_l2393_239371

theorem number_division_problem (n : ℕ) : 
  n % 37 = 26 ∧ n / 37 = 2 → 48 - n / 4 = 23 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l2393_239371


namespace NUMINAMATH_CALUDE_parabola_properties_l2393_239302

-- Define the parabola equation
def parabola (x k : ℝ) : ℝ := (x - 2)^2 + k

-- Theorem statement
theorem parabola_properties :
  ∃ k : ℝ, 
    (parabola 4 k = 12) ∧ 
    (k = 8) ∧ 
    (parabola 1 k = 9) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l2393_239302


namespace NUMINAMATH_CALUDE_ellipse_minor_axis_length_l2393_239310

/-- The length of the minor axis of an ellipse with semi-focal distance 2 and eccentricity 1/2 is 2√3. -/
theorem ellipse_minor_axis_length : 
  ∀ (c a b : ℝ), 
  c = 2 → -- semi-focal distance
  a / c = 2 → -- derived from eccentricity e = 1/2
  b ^ 2 = a ^ 2 - c ^ 2 → -- relationship between a, b, and c in an ellipse
  b = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_minor_axis_length_l2393_239310


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l2393_239383

noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1 / x

theorem tangent_slope_at_one :
  HasDerivAt f (Real.exp 1 + 1) 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l2393_239383


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l2393_239353

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃! k : ℕ, k < d ∧ (n - k) % d = 0 :=
by
  sorry

theorem problem_solution :
  let n := 13294
  let d := 97
  ∃! k : ℕ, k < d ∧ (n - k) % d = 0 ∧ k = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l2393_239353


namespace NUMINAMATH_CALUDE_valid_m_set_l2393_239348

def is_valid_m (m : ℕ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ 
    ∃ k : ℕ, m * n = k * k ∧
    ∃ p : ℕ, Nat.Prime p ∧ m - n = p

theorem valid_m_set :
  {m : ℕ | 1000 ≤ m ∧ m ≤ 2021 ∧ is_valid_m m} =
  {1156, 1296, 1369, 1600, 1764} :=
by sorry

end NUMINAMATH_CALUDE_valid_m_set_l2393_239348


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2393_239330

/-- A right triangle with perimeter 40 and area 24 has a hypotenuse of length 18.8 -/
theorem right_triangle_hypotenuse : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  a + b + c = 40 →
  (1/2) * a * b = 24 →
  a^2 + b^2 = c^2 →
  c = 18.8 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2393_239330


namespace NUMINAMATH_CALUDE_angle_bisector_theorem_application_l2393_239350

theorem angle_bisector_theorem_application (DE DF EF D₁F D₁E XY XZ YZ X₁Z X₁Y XX₁ : ℝ) : 
  DE = 13 →
  DF = 5 →
  EF = (DE^2 - DF^2).sqrt →
  D₁F / D₁E = DF / EF →
  D₁F + D₁E = EF →
  XY = D₁E →
  XZ = D₁F →
  YZ = (XY^2 - XZ^2).sqrt →
  X₁Z / X₁Y = XZ / XY →
  X₁Z + X₁Y = YZ →
  XX₁ = XZ - X₁Z →
  XX₁ = 0 := by
sorry

#eval "QED"

end NUMINAMATH_CALUDE_angle_bisector_theorem_application_l2393_239350


namespace NUMINAMATH_CALUDE_water_storage_calculation_l2393_239359

/-- Calculates the total volume of water stored in jars of different sizes -/
theorem water_storage_calculation (total_jars : ℕ) (h1 : total_jars = 24) :
  let jars_per_size := total_jars / 3
  let quart_volume := jars_per_size * (1 / 4 : ℚ)
  let half_gallon_volume := jars_per_size * (1 / 2 : ℚ)
  let gallon_volume := jars_per_size * 1
  quart_volume + half_gallon_volume + gallon_volume = 14 := by
  sorry

#check water_storage_calculation

end NUMINAMATH_CALUDE_water_storage_calculation_l2393_239359


namespace NUMINAMATH_CALUDE_work_completion_time_l2393_239393

/-- The time it takes to complete a work with two workers working sequentially -/
def total_work_time (mahesh_full_time : ℕ) (mahesh_work_time : ℕ) (rajesh_finish_time : ℕ) : ℕ :=
  mahesh_work_time + rajesh_finish_time

/-- Theorem stating that under given conditions, the total work time is 50 days -/
theorem work_completion_time :
  total_work_time 45 20 30 = 50 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2393_239393


namespace NUMINAMATH_CALUDE_center_is_seven_l2393_239303

/-- Represents a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Check if two positions in the grid are adjacent or diagonal -/
def adjacent_or_diagonal (i j i' j' : Fin 3) : Prop :=
  (i = i' ∧ |j - j'| = 1) ∨ (j = j' ∧ |i - i'| = 1) ∨ (|i - i'| = 1 ∧ |j - j'| = 1)

/-- The main theorem -/
theorem center_is_seven (g : Grid) : 
  (∀ n : ℕ, n ∈ Finset.range 9 → ∃ i j : Fin 3, g i j = n + 1) →
  (g 0 0 + g 0 2 + g 2 0 + g 2 2 = 20) →
  (∀ n : ℕ, n ∈ Finset.range 8 → 
    ∃ i j i' j' : Fin 3, g i j = n + 1 ∧ g i' j' = n + 2 ∧ adjacent_or_diagonal i j i' j') →
  g 1 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_center_is_seven_l2393_239303


namespace NUMINAMATH_CALUDE_equation1_solution_equation2_no_solution_l2393_239356

-- Define the equations
def equation1 (x : ℝ) : Prop := (3 / (x + 1)) = (2 / (x - 1))
def equation2 (x : ℝ) : Prop := (2 * x + 9) / (3 * x - 9) = (4 * x - 7) / (x - 3) + 2

-- Theorem for equation 1
theorem equation1_solution : 
  ∃! x : ℝ, equation1 x ∧ x ≠ -1 ∧ x ≠ 1 := by sorry

-- Theorem for equation 2
theorem equation2_no_solution : 
  ∀ x : ℝ, ¬(equation2 x ∧ x ≠ 3) := by sorry

end NUMINAMATH_CALUDE_equation1_solution_equation2_no_solution_l2393_239356


namespace NUMINAMATH_CALUDE_increasing_sequence_condition_l2393_239340

def sequence_term (n : ℕ) (b : ℝ) : ℝ := n^2 + b*n

theorem increasing_sequence_condition (b : ℝ) : 
  (∀ n : ℕ, sequence_term (n + 1) b > sequence_term n b) → b > -3 :=
by
  sorry

#check increasing_sequence_condition

end NUMINAMATH_CALUDE_increasing_sequence_condition_l2393_239340


namespace NUMINAMATH_CALUDE_safe_game_probabilities_l2393_239376

/-- The probability of opening all safes given the number of safes and initially opened safes. -/
def P (m n : ℕ) : ℚ :=
  sorry

theorem safe_game_probabilities (n : ℕ) (h : n ≥ 2) :
  P 2 3 = 2/3 ∧
  (∀ k, P 1 k = 1/k) ∧
  (∀ k ≥ 2, P 2 k = (2/k) * P 1 (k-1) + ((k-2)/k) * P 2 (k-1)) ∧
  (∀ k ≥ 2, P 2 k = 2/k) :=
sorry

end NUMINAMATH_CALUDE_safe_game_probabilities_l2393_239376


namespace NUMINAMATH_CALUDE_min_throws_for_repeated_sum_l2393_239396

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The number of dice being thrown -/
def numDice : ℕ := 4

/-- The minimum possible sum when rolling the dice -/
def minSum : ℕ := numDice

/-- The maximum possible sum when rolling the dice -/
def maxSum : ℕ := numDice * sides

/-- The number of possible unique sums -/
def uniqueSums : ℕ := maxSum - minSum + 1

/-- The minimum number of throws needed to guarantee a repeated sum -/
def minThrows : ℕ := uniqueSums + 1

theorem min_throws_for_repeated_sum :
  minThrows = 22 := by sorry

end NUMINAMATH_CALUDE_min_throws_for_repeated_sum_l2393_239396


namespace NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l2393_239338

/-- 
Given an arithmetic sequence where:
- The first term is 3x - 4
- The second term is 7x - 15
- The third term is 4x + 2
- The nth term is 4018

Prove that n = 803
-/
theorem arithmetic_sequence_nth_term (x : ℚ) (n : ℕ) :
  (3 * x - 4 : ℚ) = (7 * x - 15 : ℚ) - (3 * x - 4 : ℚ) ∧
  (7 * x - 15 : ℚ) = (4 * x + 2 : ℚ) - (7 * x - 15 : ℚ) ∧
  (8 : ℚ) + (n - 1 : ℚ) * 5 = 4018 →
  n = 803 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l2393_239338


namespace NUMINAMATH_CALUDE_min_value_sqrt_and_reciprocal_equality_condition_l2393_239399

theorem min_value_sqrt_and_reciprocal (x : ℝ) (hx : x > 0) :
  3 * Real.sqrt x + 4 / x ≥ 4 * Real.sqrt 2 :=
sorry

theorem equality_condition (x : ℝ) (hx : x > 0) :
  ∃ x > 0, 3 * Real.sqrt x + 4 / x = 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_sqrt_and_reciprocal_equality_condition_l2393_239399


namespace NUMINAMATH_CALUDE_pats_picnic_candy_l2393_239373

/-- Pat's picnic dessert distribution problem -/
theorem pats_picnic_candy (cookies : ℕ) (brownies : ℕ) (family_members : ℕ) (dessert_per_person : ℕ) (candy : ℕ) : 
  cookies = 42 → 
  brownies = 21 → 
  family_members = 7 → 
  dessert_per_person = 18 → 
  candy + cookies + brownies = family_members * dessert_per_person → 
  candy = 63 := by
sorry

end NUMINAMATH_CALUDE_pats_picnic_candy_l2393_239373


namespace NUMINAMATH_CALUDE_lucy_liam_family_theorem_l2393_239319

/-- Represents a family with siblings -/
structure Family where
  girls : Nat
  boys : Nat

/-- Calculates the number of sisters and brothers for a sibling in the family -/
def sibling_count (f : Family) : Nat × Nat :=
  (f.girls, f.boys - 1)

/-- The main theorem about Lucy and Liam's family -/
theorem lucy_liam_family_theorem : 
  ∀ (f : Family), 
  f.girls = 5 → f.boys = 7 → 
  let (s, b) := sibling_count f
  s * b = 25 := by
  sorry

#check lucy_liam_family_theorem

end NUMINAMATH_CALUDE_lucy_liam_family_theorem_l2393_239319


namespace NUMINAMATH_CALUDE_parabola_intersection_difference_l2393_239377

def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 6
def parabola2 (x : ℝ) : ℝ := -2 * x^2 - 4 * x + 6

def intersection_points : Set ℝ := {x : ℝ | parabola1 x = parabola2 x}

theorem parabola_intersection_difference :
  ∃ (a c : ℝ), a ∈ intersection_points ∧ c ∈ intersection_points ∧ c ≥ a ∧ c - a = 2/5 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_difference_l2393_239377


namespace NUMINAMATH_CALUDE_balloon_arrangements_l2393_239313

-- Define the word length and repeated letter counts
def word_length : ℕ := 7
def l_count : ℕ := 2
def o_count : ℕ := 2

-- Theorem statement
theorem balloon_arrangements : 
  (Nat.factorial word_length) / (Nat.factorial l_count * Nat.factorial o_count) = 1260 :=
by sorry

end NUMINAMATH_CALUDE_balloon_arrangements_l2393_239313


namespace NUMINAMATH_CALUDE_prob_even_sum_half_l2393_239306

/-- Represents a die with a specified number of faces -/
structure Die where
  faces : ℕ
  face_range : faces > 0

/-- The probability of getting an even sum when rolling two dice -/
def prob_even_sum (d1 d2 : Die) : ℚ :=
  let even_outcomes := (d1.faces.div 2) * (d2.faces.div 2) + 
                       ((d1.faces + 1).div 2) * ((d2.faces + 1).div 2)
  even_outcomes / (d1.faces * d2.faces)

/-- Theorem stating that the probability of an even sum with the specified dice is 1/2 -/
theorem prob_even_sum_half :
  let d1 : Die := ⟨8, by norm_num⟩
  let d2 : Die := ⟨6, by norm_num⟩
  prob_even_sum d1 d2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_sum_half_l2393_239306


namespace NUMINAMATH_CALUDE_number_problem_l2393_239355

theorem number_problem (x : ℝ) : 0.75 * x = 0.45 * 1500 + 495 → x = 1560 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2393_239355


namespace NUMINAMATH_CALUDE_probability_no_adjacent_standing_l2393_239318

/-- The number of valid arrangements for n people in a circle where no two adjacent people stand -/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => validArrangements (n + 1) + validArrangements n

/-- The total number of possible outcomes when n people flip coins -/
def totalOutcomes (n : ℕ) : ℕ := 2^n

theorem probability_no_adjacent_standing (n : ℕ) : 
  n = 8 → (validArrangements n : ℚ) / totalOutcomes n = 47 / 256 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_adjacent_standing_l2393_239318


namespace NUMINAMATH_CALUDE_ratio_of_numbers_with_given_hcf_lcm_l2393_239327

theorem ratio_of_numbers_with_given_hcf_lcm (a b : ℕ+) : 
  Nat.gcd a b = 84 → Nat.lcm a b = 21 → max a b = 84 → 
  (max a b : ℚ) / (min a b) = 4 / 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_with_given_hcf_lcm_l2393_239327


namespace NUMINAMATH_CALUDE_base_85_subtraction_divisibility_l2393_239384

theorem base_85_subtraction_divisibility (b : ℤ) : 
  (0 ≤ b ∧ b ≤ 20) → 
  (∃ k : ℤ, 346841047 * 85^8 + 4 * 85^7 + 1 * 85^5 + 4 * 85^4 + 8 * 85^3 + 6 * 85^2 + 4 * 85 + 3 - b = 17 * k) → 
  b = 3 := by
sorry

end NUMINAMATH_CALUDE_base_85_subtraction_divisibility_l2393_239384


namespace NUMINAMATH_CALUDE_problem_statement_l2393_239312

theorem problem_statement (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x + (x^3 / y^2) + (y^3 / x^2) + y = 174 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2393_239312


namespace NUMINAMATH_CALUDE_line_equation_60_degrees_l2393_239381

theorem line_equation_60_degrees (x y : ℝ) :
  let angle : ℝ := 60 * π / 180  -- Convert 60° to radians
  let slope : ℝ := Real.tan angle
  let y_intercept : ℝ := -1
  (slope * x - y - y_intercept = 0) ↔ (Real.sqrt 3 * x - y - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_60_degrees_l2393_239381


namespace NUMINAMATH_CALUDE_total_average_donation_l2393_239326

/-- Represents the donation statistics for two units A and B -/
structure DonationStats where
  avg_donation_A : ℝ
  num_people_A : ℕ
  num_people_B : ℕ

/-- The conditions of the donation problem -/
def donation_conditions (stats : DonationStats) : Prop :=
  -- Unit B donated twice as much as unit A
  (stats.avg_donation_A * stats.num_people_A) * 2 = (stats.avg_donation_A - 100) * stats.num_people_B
  -- The average donation per person in unit B is $100 less than the average donation per person in unit A
  ∧ (stats.avg_donation_A - 100) > 0
  -- The number of people in unit A is one-fourth of the number of people in unit B
  ∧ stats.num_people_A * 4 = stats.num_people_B

/-- The theorem stating that the total average donation is $120 -/
theorem total_average_donation (stats : DonationStats) 
  (h : donation_conditions stats) : 
  (stats.avg_donation_A * stats.num_people_A + (stats.avg_donation_A - 100) * stats.num_people_B) / 
  (stats.num_people_A + stats.num_people_B) = 120 := by
  sorry


end NUMINAMATH_CALUDE_total_average_donation_l2393_239326


namespace NUMINAMATH_CALUDE_second_number_is_ninety_l2393_239300

theorem second_number_is_ninety (x y z : ℝ) : 
  z = 4 * y →
  y = 2 * x →
  (x + y + z) / 3 = 165 →
  y = 90 := by
sorry

end NUMINAMATH_CALUDE_second_number_is_ninety_l2393_239300


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l2393_239336

theorem triangle_max_perimeter (a b c : ℝ) (A B C : ℝ) :
  c = Real.sqrt 3 →
  (Real.sqrt 3 + a) * (Real.sin C - Real.sin A) = (a + b) * Real.sin B →
  a > 0 →
  b > 0 →
  (a + b + c : ℝ) ≤ 2 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l2393_239336


namespace NUMINAMATH_CALUDE_parabola_intersection_l2393_239320

theorem parabola_intersection (m : ℝ) : 
  (m > 0) →
  (∃! x : ℝ, -1 < x ∧ x < 4 ∧ -x^2 + 4*x - 2 + m = 0) →
  (2 ≤ m ∧ m < 7) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l2393_239320


namespace NUMINAMATH_CALUDE_expression_equals_one_l2393_239357

theorem expression_equals_one (x : ℝ) (h : x ≠ 2 ∧ x ≠ -2) : 
  ((((x + 2)^3 * (x^2 - 2*x + 2)^3) / (x^3 + 8)^3)^2 * 
   (((x - 2)^3 * (x^2 + 2*x + 2)^3) / (x^3 - 8)^3)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l2393_239357


namespace NUMINAMATH_CALUDE_part_I_part_II_l2393_239395

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x * |x - 2*a| + a^2 - 4*a

-- Part I
theorem part_I :
  let f_neg_one (x : ℝ) := x * |x + 2| + 5
  ∃ (min max : ℝ), min = 2 ∧ max = 5 ∧
    (∀ x ∈ Set.Icc (-3) 0, f_neg_one x ≥ min ∧ f_neg_one x ≤ max) ∧
    (∃ x₁ ∈ Set.Icc (-3) 0, f_neg_one x₁ = min) ∧
    (∃ x₂ ∈ Set.Icc (-3) 0, f_neg_one x₂ = max) :=
sorry

-- Part II
theorem part_II :
  ∀ a : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) →
  (∀ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 →
    (1 + Real.sqrt 2) / 2 < 1 / x₁ + 1 / x₂ + 1 / x₃) :=
sorry

end NUMINAMATH_CALUDE_part_I_part_II_l2393_239395


namespace NUMINAMATH_CALUDE_round_82_367_to_hundredth_l2393_239374

/-- Represents a rational number with a repeating decimal expansion -/
structure RepeatingDecimal where
  integerPart : ℕ
  nonRepeatingPart : ℕ
  repeatingPart : ℕ

/-- Rounds a RepeatingDecimal to the nearest hundredth -/
def roundToHundredth (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The number 82.367367... as a RepeatingDecimal -/
def number : RepeatingDecimal :=
  { integerPart := 82,
    nonRepeatingPart := 3,
    repeatingPart := 67 }

theorem round_82_367_to_hundredth :
  roundToHundredth number = 82.37 := by sorry

end NUMINAMATH_CALUDE_round_82_367_to_hundredth_l2393_239374


namespace NUMINAMATH_CALUDE_alexa_lemonade_profit_l2393_239308

/-- Calculates the profit from a lemonade stand given the price per cup,
    cost of ingredients, and number of cups sold. -/
def lemonade_profit (price_per_cup : ℕ) (ingredient_cost : ℕ) (cups_sold : ℕ) : ℕ :=
  price_per_cup * cups_sold - ingredient_cost

/-- Proves that given the specific conditions of Alexa's lemonade stand,
    her desired profit is $80. -/
theorem alexa_lemonade_profit :
  lemonade_profit 2 20 50 = 80 := by
  sorry

end NUMINAMATH_CALUDE_alexa_lemonade_profit_l2393_239308


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l2393_239390

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x - 4

-- State the theorem
theorem quadratic_roots_range (a b : ℝ) : 
  a > 0 → 
  (∃ x y : ℝ, x ≠ y ∧ f a b x = 0 ∧ f a b y = 0) →
  (∃ r : ℝ, 1 < r ∧ r < 2 ∧ f a b r = 0) →
  ∀ s : ℝ, s < 4 ↔ ∃ t : ℝ, a + b = t ∧ t < 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l2393_239390


namespace NUMINAMATH_CALUDE_tangent_sum_l2393_239365

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition that f is tangent to y = -x + 8 at (5, f(5))
def is_tangent_at_5 (f : ℝ → ℝ) : Prop :=
  f 5 = -5 + 8 ∧ deriv f 5 = -1

-- State the theorem
theorem tangent_sum (f : ℝ → ℝ) (h : is_tangent_at_5 f) :
  f 5 + deriv f 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_l2393_239365


namespace NUMINAMATH_CALUDE_height_relation_l2393_239354

/-- Two right circular cylinders with equal volume and related radii -/
structure TwoCylinders where
  r1 : ℝ  -- radius of the first cylinder
  h1 : ℝ  -- height of the first cylinder
  r2 : ℝ  -- radius of the second cylinder
  h2 : ℝ  -- height of the second cylinder
  r1_pos : 0 < r1  -- r1 is positive
  h1_pos : 0 < h1  -- h1 is positive
  r2_pos : 0 < r2  -- r2 is positive
  h2_pos : 0 < h2  -- h2 is positive
  equal_volume : r1^2 * h1 = r2^2 * h2  -- cylinders have equal volume
  radius_relation : r2 = 1.2 * r1  -- second radius is 20% more than the first

/-- The height of the first cylinder is 44% more than the height of the second cylinder -/
theorem height_relation (c : TwoCylinders) : c.h1 = 1.44 * c.h2 := by
  sorry

end NUMINAMATH_CALUDE_height_relation_l2393_239354


namespace NUMINAMATH_CALUDE_integer_sum_of_fourth_powers_l2393_239380

theorem integer_sum_of_fourth_powers (a b c : ℤ) (h : a = b + c) :
  a^4 + b^4 + c^4 = 2 * (a^2 - b*c)^2 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_of_fourth_powers_l2393_239380


namespace NUMINAMATH_CALUDE_percentage_soccer_players_is_12_5_l2393_239346

/-- The percentage of students who play sports that also play soccer -/
def percentage_soccer_players (total_students : ℕ) (sports_percentage : ℚ) (soccer_players : ℕ) : ℚ :=
  (soccer_players : ℚ) / (sports_percentage * total_students) * 100

/-- Theorem: The percentage of students who play sports that also play soccer is 12.5% -/
theorem percentage_soccer_players_is_12_5 :
  percentage_soccer_players 400 (52 / 100) 26 = 25 / 2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_soccer_players_is_12_5_l2393_239346


namespace NUMINAMATH_CALUDE_initial_water_percentage_l2393_239366

theorem initial_water_percentage (container_capacity : ℝ) (added_water : ℝ) (final_fraction : ℝ) :
  container_capacity = 40 →
  added_water = 18 →
  final_fraction = 3/4 →
  (container_capacity * final_fraction - added_water) / container_capacity * 100 = 30 :=
by sorry

end NUMINAMATH_CALUDE_initial_water_percentage_l2393_239366


namespace NUMINAMATH_CALUDE_robot_fifth_minute_distance_l2393_239316

def robot_distance (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 2
  | k + 1 => 2 * robot_distance k

theorem robot_fifth_minute_distance :
  robot_distance 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_robot_fifth_minute_distance_l2393_239316


namespace NUMINAMATH_CALUDE_colinear_vector_problem_l2393_239398

/-- Given vector a and b in ℝ², prove that if a = (1, -2), b is colinear with a, and |b| = 4|a|, then b = (4, -8) or b = (-4, 8) -/
theorem colinear_vector_problem (a b : ℝ × ℝ) : 
  a = (1, -2) → 
  (∃ (k : ℝ), b = k • a) → 
  Real.sqrt ((b.1)^2 + (b.2)^2) = 4 * Real.sqrt ((a.1)^2 + (a.2)^2) → 
  b = (4, -8) ∨ b = (-4, 8) := by
sorry

end NUMINAMATH_CALUDE_colinear_vector_problem_l2393_239398


namespace NUMINAMATH_CALUDE_card_selection_count_l2393_239392

def total_cards : ℕ := 12
def red_cards : ℕ := 4
def yellow_cards : ℕ := 4
def blue_cards : ℕ := 4
def cards_to_select : ℕ := 3
def max_red_cards : ℕ := 1

theorem card_selection_count :
  (Nat.choose (yellow_cards + blue_cards) cards_to_select) +
  (Nat.choose red_cards max_red_cards * Nat.choose (yellow_cards + blue_cards) (cards_to_select - max_red_cards)) = 168 := by
  sorry

end NUMINAMATH_CALUDE_card_selection_count_l2393_239392


namespace NUMINAMATH_CALUDE_bobbit_worm_predation_l2393_239322

/-- Calculates the number of fish remaining in an aquarium after a Bobbit worm's predation --/
theorem bobbit_worm_predation 
  (initial_fish : ℕ) 
  (daily_eaten : ℕ) 
  (days_before_adding : ℕ) 
  (added_fish : ℕ) 
  (days_after_adding : ℕ) :
  initial_fish = 60 →
  daily_eaten = 2 →
  days_before_adding = 14 →
  added_fish = 8 →
  days_after_adding = 7 →
  initial_fish + added_fish - (daily_eaten * (days_before_adding + days_after_adding)) = 26 :=
by sorry

end NUMINAMATH_CALUDE_bobbit_worm_predation_l2393_239322


namespace NUMINAMATH_CALUDE_no_valid_base_for_122_square_l2393_239370

theorem no_valid_base_for_122_square : ¬ ∃ (b : ℕ), b > 1 ∧ ∃ (n : ℕ), b^2 + 2*b + 2 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_base_for_122_square_l2393_239370


namespace NUMINAMATH_CALUDE_eight_students_in_neither_l2393_239389

/-- Represents the number of students in various categories of a science club. -/
structure ScienceClub where
  total : ℕ
  biology : ℕ
  chemistry : ℕ
  both : ℕ

/-- Calculates the number of students taking neither biology nor chemistry. -/
def studentsInNeither (club : ScienceClub) : ℕ :=
  club.total - (club.biology + club.chemistry - club.both)

/-- Theorem stating that for the given science club configuration, 
    8 students take neither biology nor chemistry. -/
theorem eight_students_in_neither (club : ScienceClub) 
  (h1 : club.total = 60)
  (h2 : club.biology = 42)
  (h3 : club.chemistry = 35)
  (h4 : club.both = 25) : 
  studentsInNeither club = 8 := by
  sorry

#eval studentsInNeither { total := 60, biology := 42, chemistry := 35, both := 25 }

end NUMINAMATH_CALUDE_eight_students_in_neither_l2393_239389


namespace NUMINAMATH_CALUDE_factor_expression_l2393_239367

theorem factor_expression (b : ℝ) : 29*b^2 + 87*b = 29*b*(b+3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2393_239367


namespace NUMINAMATH_CALUDE_profit_ratio_equals_investment_ratio_l2393_239391

/-- The ratio of profits is proportional to the ratio of investments -/
theorem profit_ratio_equals_investment_ratio (p_investment q_investment : ℚ) 
  (hp : p_investment = 50000)
  (hq : q_investment = 66666.67)
  : ∃ (k : ℚ), k * p_investment = 3 ∧ k * q_investment = 4 := by
  sorry

end NUMINAMATH_CALUDE_profit_ratio_equals_investment_ratio_l2393_239391


namespace NUMINAMATH_CALUDE_sqrt_450_simplified_l2393_239360

theorem sqrt_450_simplified : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplified_l2393_239360


namespace NUMINAMATH_CALUDE_pineapple_shipping_cost_l2393_239311

/-- The shipping cost for a dozen pineapples, given the initial cost and total cost per pineapple. -/
theorem pineapple_shipping_cost 
  (initial_cost : ℚ)  -- Cost of each pineapple before shipping
  (total_cost : ℚ)    -- Total cost of each pineapple including shipping
  (h1 : initial_cost = 1.25)  -- Each pineapple costs $1.25
  (h2 : total_cost = 3)       -- Each pineapple ends up costing $3
  : (12 : ℚ) * (total_cost - initial_cost) = 21 := by
  sorry

#check pineapple_shipping_cost

end NUMINAMATH_CALUDE_pineapple_shipping_cost_l2393_239311


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2393_239372

/-- Given that x varies inversely as square of y, prove that x = 1/9 when y = 6,
    given that y = 2 when x = 1 -/
theorem inverse_variation_problem (x y : ℝ) (k : ℝ) (h1 : x = k / y^2) 
    (h2 : 1 = k / 2^2) : 
  y = 6 → x = 1/9 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2393_239372


namespace NUMINAMATH_CALUDE_condition_type_l2393_239363

theorem condition_type (a : ℝ) : 
  (∀ x : ℝ, x > 2 → x^2 > 2*x) ∧ 
  (∃ y : ℝ, y ≤ 2 ∧ y^2 > 2*y) :=
by sorry

end NUMINAMATH_CALUDE_condition_type_l2393_239363


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2393_239352

theorem arithmetic_sequence_middle_term :
  ∀ (a : ℕ → ℤ), 
    (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence
    a 0 = 3^2 →                                           -- first term is 3^2
    a 2 = 3^3 →                                           -- third term is 3^3
    a 1 = 18 :=                                           -- second term is 18
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2393_239352


namespace NUMINAMATH_CALUDE_alex_score_l2393_239325

theorem alex_score (total_students : ℕ) (graded_students : ℕ) (initial_average : ℚ) (final_average : ℚ) :
  total_students = 20 →
  graded_students = 19 →
  initial_average = 72 →
  final_average = 74 →
  (graded_students * initial_average + (total_students - graded_students) * 
    ((total_students * final_average - graded_students * initial_average) / (total_students - graded_students))) / total_students = final_average →
  (total_students * final_average - graded_students * initial_average) = 112 := by
  sorry

end NUMINAMATH_CALUDE_alex_score_l2393_239325


namespace NUMINAMATH_CALUDE_coin_flip_sequences_l2393_239329

/-- The number of flips performed -/
def num_flips : ℕ := 10

/-- The number of possible outcomes for each flip -/
def outcomes_per_flip : ℕ := 2

/-- The total number of distinct sequences possible -/
def total_sequences : ℕ := outcomes_per_flip ^ num_flips

theorem coin_flip_sequences :
  total_sequences = 1024 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_sequences_l2393_239329


namespace NUMINAMATH_CALUDE_original_light_wattage_l2393_239397

theorem original_light_wattage (W : ℝ) : 
  (W + 0.3 * W = 143) → W = 110 := by
  sorry

end NUMINAMATH_CALUDE_original_light_wattage_l2393_239397


namespace NUMINAMATH_CALUDE_asymptote_sum_l2393_239335

/-- Represents a rational function -/
structure RationalFunction where
  numerator : Polynomial ℝ
  denominator : Polynomial ℝ

/-- Counts the number of holes in the graph of a rational function -/
noncomputable def count_holes (f : RationalFunction) : ℕ := sorry

/-- Counts the number of vertical asymptotes of a rational function -/
noncomputable def count_vertical_asymptotes (f : RationalFunction) : ℕ := sorry

/-- Counts the number of horizontal asymptotes of a rational function -/
noncomputable def count_horizontal_asymptotes (f : RationalFunction) : ℕ := sorry

/-- Counts the number of oblique asymptotes of a rational function -/
noncomputable def count_oblique_asymptotes (f : RationalFunction) : ℕ := sorry

/-- The main theorem -/
theorem asymptote_sum (f : RationalFunction) 
  (h : f.numerator = Polynomial.X^2 + 4*Polynomial.X + 3 ∧ 
       f.denominator = Polynomial.X^3 + 2*Polynomial.X^2 - 3*Polynomial.X) : 
  count_holes f + 2 * count_vertical_asymptotes f + 
  3 * count_horizontal_asymptotes f + 4 * count_oblique_asymptotes f = 8 := by
  sorry

end NUMINAMATH_CALUDE_asymptote_sum_l2393_239335


namespace NUMINAMATH_CALUDE_watson_second_graders_l2393_239321

/-- Represents the number of students in each grade and the total in Ms. Watson's class -/
structure ClassComposition where
  total : Nat
  kindergartners : Nat
  firstGraders : Nat
  thirdGraders : Nat
  absentStudents : Nat

/-- Calculates the number of second graders in the class -/
def secondGraders (c : ClassComposition) : Nat :=
  c.total - (c.kindergartners + c.firstGraders + c.thirdGraders + c.absentStudents)

/-- Theorem stating the number of second graders in Ms. Watson's class -/
theorem watson_second_graders :
  let c : ClassComposition := {
    total := 120,
    kindergartners := 34,
    firstGraders := 48,
    thirdGraders := 5,
    absentStudents := 6
  }
  secondGraders c = 27 := by sorry

end NUMINAMATH_CALUDE_watson_second_graders_l2393_239321


namespace NUMINAMATH_CALUDE_tan_plus_four_sin_twenty_degrees_l2393_239351

theorem tan_plus_four_sin_twenty_degrees :
  Real.tan (20 * π / 180) + 4 * Real.sin (20 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_plus_four_sin_twenty_degrees_l2393_239351


namespace NUMINAMATH_CALUDE_difference_of_prime_squares_can_be_perfect_square_l2393_239382

theorem difference_of_prime_squares_can_be_perfect_square :
  ∃ (p q : ℕ) (n : ℕ), Prime p ∧ Prime q ∧ p^2 - q^2 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_prime_squares_can_be_perfect_square_l2393_239382


namespace NUMINAMATH_CALUDE_weight_difference_l2393_239362

/-- Given Heather's and Emily's weights, prove the weight difference between them. -/
theorem weight_difference (heather_weight emily_weight : ℕ) 
  (h_heather : heather_weight = 87)
  (h_emily : emily_weight = 9) :
  heather_weight - emily_weight = 78 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l2393_239362


namespace NUMINAMATH_CALUDE_final_sum_of_numbers_l2393_239334

theorem final_sum_of_numbers (n : ℕ) (h1 : n = 2013) : 
  ∃ (a b c d : ℕ), 
    (a * b * c * d = 27) ∧ 
    (a + b + c + d ≡ (n * (n + 1) / 2) [MOD 9]) ∧
    (a + b + c + d = 30) := by
  sorry

end NUMINAMATH_CALUDE_final_sum_of_numbers_l2393_239334


namespace NUMINAMATH_CALUDE_train_passing_time_l2393_239324

/-- Given a train of length l traveling at constant velocity v, if the time to pass a platform
    of length 3l is 4 times the time to pass a pole, then the time to pass the pole is l/v. -/
theorem train_passing_time
  (l v : ℝ) -- Length of train and velocity
  (h_pos_l : l > 0)
  (h_pos_v : v > 0)
  (t : ℝ) -- Time to pass the pole
  (T : ℝ) -- Time to pass the platform
  (h_platform_time : T = 4 * t) -- Time to pass platform is 4 times time to pass pole
  (h_platform_length : 4 * l = v * T) -- Distance-velocity-time equation for platform
  : t = l / v := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l2393_239324


namespace NUMINAMATH_CALUDE_trevors_brother_age_l2393_239343

/-- Trevor's age a decade ago -/
def trevors_age_decade_ago : ℕ := 16

/-- Current year -/
def current_year : ℕ := 2023

/-- Trevor's current age -/
def trevors_current_age : ℕ := trevors_age_decade_ago + 10

/-- Trevor's age 20 years ago -/
def trevors_age_20_years_ago : ℕ := trevors_current_age - 20

/-- Trevor's brother's age 20 years ago -/
def brothers_age_20_years_ago : ℕ := 2 * trevors_age_20_years_ago

/-- Trevor's brother's current age -/
def brothers_current_age : ℕ := brothers_age_20_years_ago + 20

theorem trevors_brother_age : brothers_current_age = 32 := by
  sorry

end NUMINAMATH_CALUDE_trevors_brother_age_l2393_239343


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2393_239358

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {-2, 0, 1}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2393_239358


namespace NUMINAMATH_CALUDE_quadratic_equation_at_negative_two_l2393_239314

theorem quadratic_equation_at_negative_two :
  let x : ℤ := -2
  x^2 + 6*x - 10 = -18 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_at_negative_two_l2393_239314


namespace NUMINAMATH_CALUDE_spade_calculation_l2393_239331

-- Define the spade operation
def spade (a b : ℤ) : ℤ := Int.natAbs (a - b)

-- State the theorem
theorem spade_calculation : (spade 8 5) + (spade 3 (spade 6 2)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l2393_239331


namespace NUMINAMATH_CALUDE_mark_remaining_hours_l2393_239315

def sick_days : ℕ := 10
def vacation_days : ℕ := 10
def hours_per_day : ℕ := 8
def used_fraction : ℚ := 1/2

theorem mark_remaining_hours : 
  (sick_days + vacation_days) * (1 - used_fraction) * hours_per_day = 80 := by
  sorry

end NUMINAMATH_CALUDE_mark_remaining_hours_l2393_239315


namespace NUMINAMATH_CALUDE_trader_gain_percentage_l2393_239369

theorem trader_gain_percentage : 
  ∀ (cost_per_pen : ℝ), cost_per_pen > 0 →
  (19 * cost_per_pen) / (95 * cost_per_pen) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_trader_gain_percentage_l2393_239369


namespace NUMINAMATH_CALUDE_g_13_equals_218_l2393_239301

-- Define the function g
def g (n : ℕ) : ℕ := n^2 + 2*n + 23

-- State the theorem
theorem g_13_equals_218 : g 13 = 218 := by
  sorry

end NUMINAMATH_CALUDE_g_13_equals_218_l2393_239301


namespace NUMINAMATH_CALUDE_bernoulli_prob_zero_success_l2393_239347

/-- The number of Bernoulli trials -/
def n : ℕ := 7

/-- The probability of success in each trial -/
def p : ℚ := 2/7

/-- The probability of failure in each trial -/
def q : ℚ := 1 - p

/-- The number of successes we're interested in -/
def k : ℕ := 0

/-- Theorem: The probability of 0 successes in 7 Bernoulli trials 
    with success probability 2/7 is (5/7)^7 -/
theorem bernoulli_prob_zero_success : 
  (n.choose k) * p^k * q^(n-k) = (5/7)^7 := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_prob_zero_success_l2393_239347


namespace NUMINAMATH_CALUDE_sqrt_five_approximation_l2393_239361

theorem sqrt_five_approximation :
  (2^2 < 5 ∧ 5 < 3^2) →
  (2.2^2 < 5 ∧ 5 < 2.3^2) →
  (2.23^2 < 5 ∧ 5 < 2.24^2) →
  (2.236^2 < 5 ∧ 5 < 2.237^2) →
  ∃ (x : ℝ), x^2 = 5 ∧ |x - 2.24| < 0.005 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_five_approximation_l2393_239361


namespace NUMINAMATH_CALUDE_unique_cds_l2393_239364

theorem unique_cds (shared : ℕ) (alice_total : ℕ) (bob_unique : ℕ) 
  (h1 : shared = 12)
  (h2 : alice_total = 23)
  (h3 : bob_unique = 8) :
  alice_total - shared + bob_unique = 19 :=
by sorry

end NUMINAMATH_CALUDE_unique_cds_l2393_239364


namespace NUMINAMATH_CALUDE_slope_angle_range_l2393_239375

/-- Given two lines L1 and L2, where L1 has slope k and y-intercept -b,
    and their intersection point M is in the first quadrant,
    prove that the slope angle α of L1 is between arctan(-2/3) and π/2. -/
theorem slope_angle_range (k b : ℝ) :
  let L1 := λ x y : ℝ => y = k * x - b
  let L2 := λ x y : ℝ => 2 * x + 3 * y - 6 = 0
  let M := (((3 * b + 6) / (2 + 3 * k)), ((6 * k + 2 * b) / (2 + 3 * k)))
  let α := Real.arctan k
  (M.1 > 0 ∧ M.2 > 0) → (α > Real.arctan (-2/3) ∧ α < π/2) := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_range_l2393_239375


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2393_239388

theorem equal_roots_quadratic (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 = 0 ∧ 
   ∀ y : ℝ, y^2 + a*y + 1 = 0 → y = x) → 
  a = -2 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2393_239388


namespace NUMINAMATH_CALUDE_solution_of_equation_l2393_239317

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (x : ℂ) : Prop := 2 + i * x = -2 - 2 * i * x

-- State the theorem
theorem solution_of_equation :
  ∃ (x : ℂ), equation x ∧ x = (4 * i) / 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_of_equation_l2393_239317


namespace NUMINAMATH_CALUDE_mark_fruit_count_l2393_239385

/-- The number of pieces of fruit Mark had at the beginning of the week -/
def total_fruit (kept_for_next_week : ℕ) (brought_to_school : ℕ) (eaten_first_four_days : ℕ) : ℕ :=
  kept_for_next_week + brought_to_school + eaten_first_four_days

/-- Theorem stating that Mark had 10 pieces of fruit at the beginning of the week -/
theorem mark_fruit_count : total_fruit 2 3 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_mark_fruit_count_l2393_239385


namespace NUMINAMATH_CALUDE_system_solution_l2393_239368

theorem system_solution (x y k : ℝ) 
  (eq1 : x - y = k + 2)
  (eq2 : x + 3*y = k)
  (eq3 : x + y = 2) :
  k = 1 := by sorry

end NUMINAMATH_CALUDE_system_solution_l2393_239368


namespace NUMINAMATH_CALUDE_unknown_number_divisor_l2393_239333

theorem unknown_number_divisor : ∃ x : ℕ, 
  x > 0 ∧ 
  100 % x = 16 ∧ 
  200 % x = 4 ∧ 
  ∀ y : ℕ, y > 0 → 100 % y = 16 → 200 % y = 4 → y ≤ x :=
sorry

end NUMINAMATH_CALUDE_unknown_number_divisor_l2393_239333


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_specific_quadratic_roots_l2393_239379

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) : 
  let discriminant := b^2 - 4*a*c
  discriminant > 0 → ∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0 :=
by
  sorry

theorem specific_quadratic_roots : 
  ∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x - 6 = 0 ∧ y^2 - 2*y - 6 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_specific_quadratic_roots_l2393_239379


namespace NUMINAMATH_CALUDE_impossible_equal_sum_arrangement_l2393_239323

theorem impossible_equal_sum_arrangement : ¬∃ (a b c d e f : ℕ),
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
   d ≠ e ∧ d ≠ f ∧
   e ≠ f) ∧
  (∃ (s : ℕ), 
    a + b + c = s ∧
    a + d + e = s ∧
    b + d + f = s ∧
    c + e + f = s) :=
by sorry

end NUMINAMATH_CALUDE_impossible_equal_sum_arrangement_l2393_239323


namespace NUMINAMATH_CALUDE_complex_modulus_product_l2393_239332

theorem complex_modulus_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l2393_239332


namespace NUMINAMATH_CALUDE_sixth_term_value_l2393_239349

def sequence_rule (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n = (a (n-1) + a (n+1)) / 3

theorem sixth_term_value (a : ℕ → ℕ) :
  sequence_rule a →
  a 2 = 7 →
  a 3 = 20 →
  a 6 = 364 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_value_l2393_239349


namespace NUMINAMATH_CALUDE_toothpick_pattern_sum_l2393_239386

def arithmeticSequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

def arithmeticSum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem toothpick_pattern_sum :
  arithmeticSum 6 5 150 = 56775 := by sorry

end NUMINAMATH_CALUDE_toothpick_pattern_sum_l2393_239386


namespace NUMINAMATH_CALUDE_diamond_value_l2393_239339

/-- The diamond operation for non-zero integers -/
def diamond (a b : ℤ) : ℚ := (1 : ℚ) / a + (2 : ℚ) / b

/-- Theorem stating the value of a ◇ b given the conditions -/
theorem diamond_value (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 10) (h4 : a * b = 24) :
  diamond a b = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_diamond_value_l2393_239339


namespace NUMINAMATH_CALUDE_family_change_is_74_l2393_239304

/-- Represents the cost of tickets for a family visit to an amusement park --/
def amusement_park_change (regular_price : ℕ) (child_discount : ℕ) (amount_given : ℕ) : ℕ :=
  let adult_cost := regular_price
  let child_cost := regular_price - child_discount
  let total_cost := 2 * adult_cost + 2 * child_cost
  amount_given - total_cost

/-- Theorem stating that the change received by the family is $74 --/
theorem family_change_is_74 :
  amusement_park_change 109 5 500 = 74 := by
  sorry

end NUMINAMATH_CALUDE_family_change_is_74_l2393_239304


namespace NUMINAMATH_CALUDE_circle_x_intersection_l2393_239328

theorem circle_x_intersection (x : ℝ) : 
  let center_x := (-2 + 6) / 2
  let center_y := (1 + 9) / 2
  let radius := Real.sqrt (((-2 - center_x)^2 + (1 - center_y)^2) : ℝ)
  (x - center_x)^2 + (0 - center_y)^2 = radius^2 →
  x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_circle_x_intersection_l2393_239328


namespace NUMINAMATH_CALUDE_divisibility_problem_l2393_239309

theorem divisibility_problem :
  (∃ n : ℕ, n = 9 ∧ (1100 + n) % 53 = 0 ∧ ∀ k : ℕ, k < n → (1100 + k) % 53 ≠ 0) ∧
  (∃ m : ℕ, m = 0 ∧ (1100 - m) % 71 = 0 ∧ ∀ k : ℕ, k < m → (1100 - k) % 71 ≠ 0) ∧
  (∃ X : ℤ, X = 534 ∧ (1100 + X) % 19 = 0 ∧ (1100 + X) % 43 = 0) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2393_239309


namespace NUMINAMATH_CALUDE_initial_deposit_l2393_239378

theorem initial_deposit (P R : ℝ) : 
  P + (P * R * 3) / 100 = 9200 →
  P + (P * (R + 2.5) * 3) / 100 = 9800 →
  P = 8000 := by
sorry

end NUMINAMATH_CALUDE_initial_deposit_l2393_239378
