import Mathlib

namespace NUMINAMATH_CALUDE_unique_factorial_product_l2154_215406

theorem unique_factorial_product (n : ℕ) : (n + 1) * n.factorial = 5040 ↔ n = 6 := by sorry

end NUMINAMATH_CALUDE_unique_factorial_product_l2154_215406


namespace NUMINAMATH_CALUDE_min_correct_answers_to_pass_l2154_215456

/-- Represents the Fire Safety quiz selection -/
structure FireSafetyQuiz where
  total_questions : Nat
  correct_score : Int
  incorrect_score : Int
  passing_score : Int

/-- Calculates the total score based on the number of correct answers -/
def calculate_score (quiz : FireSafetyQuiz) (correct_answers : Nat) : Int :=
  (quiz.correct_score * correct_answers) + 
  (quiz.incorrect_score * (quiz.total_questions - correct_answers))

/-- Theorem: The minimum number of correct answers needed to pass the Fire Safety quiz is 12 -/
theorem min_correct_answers_to_pass (quiz : FireSafetyQuiz) 
  (h1 : quiz.total_questions = 20)
  (h2 : quiz.correct_score = 10)
  (h3 : quiz.incorrect_score = -5)
  (h4 : quiz.passing_score = 80) :
  ∀ n : Nat, calculate_score quiz n ≥ quiz.passing_score → n ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_min_correct_answers_to_pass_l2154_215456


namespace NUMINAMATH_CALUDE_sin_shift_l2154_215428

theorem sin_shift (x : ℝ) : Real.sin (5 * π / 6 - x) = Real.sin (x + π / 6) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_l2154_215428


namespace NUMINAMATH_CALUDE_sum_of_digits_is_nine_l2154_215468

/-- The sum of the tens digit and the units digit in the decimal representation of 7^1974 -/
def sum_of_digits : ℕ :=
  let n := 7^1974
  let tens_digit := (n / 10) % 10
  let units_digit := n % 10
  tens_digit + units_digit

/-- The sum of the tens digit and the units digit in the decimal representation of 7^1974 is 9 -/
theorem sum_of_digits_is_nine : sum_of_digits = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_is_nine_l2154_215468


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l2154_215433

theorem triangle_sine_inequality (α β γ : Real) (h : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π) :
  (Real.sin α + Real.sin β + Real.sin γ)^2 > 9 * Real.sin α * Real.sin β * Real.sin γ := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l2154_215433


namespace NUMINAMATH_CALUDE_square_root_divided_by_18_equals_4_l2154_215472

theorem square_root_divided_by_18_equals_4 (x : ℝ) : 
  (Real.sqrt x) / 18 = 4 → x = 5184 := by sorry

end NUMINAMATH_CALUDE_square_root_divided_by_18_equals_4_l2154_215472


namespace NUMINAMATH_CALUDE_distance_to_center_is_five_l2154_215469

/-- A square with side length 10 and a circle passing through two opposite vertices
    and tangent to one side -/
structure SquareWithCircle where
  /-- The side length of the square -/
  sideLength : ℝ
  /-- The circle passes through two opposite vertices -/
  circlePassesThroughOppositeVertices : Bool
  /-- The circle is tangent to one side -/
  circleTangentToSide : Bool

/-- The distance from the center of the circle to a vertex of the square -/
def distanceToCenterFromVertex (s : SquareWithCircle) : ℝ := sorry

/-- Theorem stating that the distance from the center of the circle to a vertex is 5 -/
theorem distance_to_center_is_five (s : SquareWithCircle) 
  (h1 : s.sideLength = 10)
  (h2 : s.circlePassesThroughOppositeVertices = true)
  (h3 : s.circleTangentToSide = true) : 
  distanceToCenterFromVertex s = 5 := by sorry

end NUMINAMATH_CALUDE_distance_to_center_is_five_l2154_215469


namespace NUMINAMATH_CALUDE_greatest_t_value_l2154_215429

theorem greatest_t_value (t : ℝ) : 
  (t^2 - t - 56) / (t - 8) = 3 / (t + 5) → t ≤ -4 :=
by sorry

end NUMINAMATH_CALUDE_greatest_t_value_l2154_215429


namespace NUMINAMATH_CALUDE_deepak_age_l2154_215422

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's current age -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 5 / 2 →
  rahul_age + 6 = 26 →
  deepak_age = 8 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l2154_215422


namespace NUMINAMATH_CALUDE_files_deleted_l2154_215436

theorem files_deleted (initial_files remaining_files : ℕ) (h1 : initial_files = 25) (h2 : remaining_files = 2) :
  initial_files - remaining_files = 23 := by
  sorry

end NUMINAMATH_CALUDE_files_deleted_l2154_215436


namespace NUMINAMATH_CALUDE_product_mod_25_l2154_215471

theorem product_mod_25 (n : ℕ) : 
  77 * 88 * 99 ≡ n [ZMOD 25] → 0 ≤ n → n < 25 → n = 24 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_25_l2154_215471


namespace NUMINAMATH_CALUDE_library_books_sold_l2154_215441

theorem library_books_sold (total_books : ℕ) (remaining_fraction : ℚ) (books_sold : ℕ) : 
  total_books = 9900 ∧ remaining_fraction = 4/6 → books_sold = 3300 :=
by sorry

end NUMINAMATH_CALUDE_library_books_sold_l2154_215441


namespace NUMINAMATH_CALUDE_average_weight_decrease_l2154_215482

/-- Proves that replacing a 72 kg student with a 12 kg student in a group of 5 decreases the average weight by 12 kg -/
theorem average_weight_decrease (initial_average : ℝ) : 
  let total_weight := 5 * initial_average
  let new_total_weight := total_weight - 72 + 12
  let new_average := new_total_weight / 5
  initial_average - new_average = 12 := by
sorry

end NUMINAMATH_CALUDE_average_weight_decrease_l2154_215482


namespace NUMINAMATH_CALUDE_inequality_proof_l2154_215400

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  Real.sqrt (a^(1 - a) * b^(1 - b) * c^(1 - c)) ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2154_215400


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2154_215416

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, x > 3 → x ≥ 3) ∧ 
  (∃ x : ℝ, x ≥ 3 ∧ ¬(x > 3)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2154_215416


namespace NUMINAMATH_CALUDE_residue_mod_17_l2154_215407

theorem residue_mod_17 : (243 * 15 - 22 * 8 + 5) % 17 = 5 := by
  sorry

end NUMINAMATH_CALUDE_residue_mod_17_l2154_215407


namespace NUMINAMATH_CALUDE_building_height_l2154_215458

/-- Given a flagpole and a building casting shadows under similar conditions,
    this theorem proves that the height of the building can be determined
    using the principle of similar triangles. -/
theorem building_height
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_shadow : ℝ)
  (flagpole_height_pos : 0 < flagpole_height)
  (flagpole_shadow_pos : 0 < flagpole_shadow)
  (building_shadow_pos : 0 < building_shadow)
  (h_flagpole : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_shadow : building_shadow = 70) :
  (flagpole_height / flagpole_shadow) * building_shadow = 28 :=
by sorry

end NUMINAMATH_CALUDE_building_height_l2154_215458


namespace NUMINAMATH_CALUDE_roots_product_l2154_215450

theorem roots_product (p q : ℝ) : 
  (p - 3) * (3 * p + 8) = p^2 - 17 * p + 56 →
  (q - 3) * (3 * q + 8) = q^2 - 17 * q + 56 →
  p ≠ q →
  (p + 2) * (q + 2) = -60 := by
sorry

end NUMINAMATH_CALUDE_roots_product_l2154_215450


namespace NUMINAMATH_CALUDE_marvelous_divisible_by_five_infinitely_many_marvelous_numbers_l2154_215425

def is_marvelous (n : ℕ+) : Prop :=
  ∃ (a b c d e : ℕ+),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    (n : ℕ) % a = 0 ∧ (n : ℕ) % b = 0 ∧ (n : ℕ) % c = 0 ∧ (n : ℕ) % d = 0 ∧ (n : ℕ) % e = 0 ∧
    n = a^4 + b^4 + c^4 + d^4 + e^4

theorem marvelous_divisible_by_five (n : ℕ+) (h : is_marvelous n) :
  (n : ℕ) % 5 = 0 :=
sorry

theorem infinitely_many_marvelous_numbers :
  ∀ k : ℕ, ∃ n : ℕ+, n > k ∧ is_marvelous n :=
sorry

end NUMINAMATH_CALUDE_marvelous_divisible_by_five_infinitely_many_marvelous_numbers_l2154_215425


namespace NUMINAMATH_CALUDE_three_incorrect_statements_l2154_215444

theorem three_incorrect_statements (a b c : ℕ+) 
  (h1 : Nat.Coprime a.val b.val) 
  (h2 : Nat.Coprime b.val c.val) : 
  ∃ (a b c : ℕ+), 
    (¬(¬(b.val ∣ (a.val + c.val)^2))) ∧ 
    (¬(¬(b.val ∣ a.val^2 + c.val^2))) ∧ 
    (¬(¬(c.val ∣ (a.val + b.val)^2))) :=
sorry

end NUMINAMATH_CALUDE_three_incorrect_statements_l2154_215444


namespace NUMINAMATH_CALUDE_f_properties_l2154_215466

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log (1/2)
  else if x = 0 then 0
  else Real.log (-x) / Real.log (1/2)

-- State the theorem
theorem f_properties :
  (∀ x, f x = f (-x)) ∧  -- f is even
  f 0 = 0 ∧             -- f(0) = 0
  (∀ x > 0, f x = Real.log x / Real.log (1/2)) →  -- f(x) = log₍₁/₂₎(x) for x > 0
  f (-4) = -2 ∧         -- Part 1: f(-4) = -2
  (∀ x, f x = if x > 0 then Real.log x / Real.log (1/2)
              else if x = 0 then 0
              else Real.log (-x) / Real.log (1/2))  -- Part 2: Analytic expression of f
  := by sorry

end NUMINAMATH_CALUDE_f_properties_l2154_215466


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_value_l2154_215483

theorem mean_equality_implies_z_value :
  let mean1 := (8 + 12 + 24) / 3
  let mean2 := (16 + z) / 2
  mean1 = mean2 → z = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_value_l2154_215483


namespace NUMINAMATH_CALUDE_student_distribution_l2154_215478

/-- The number of students standing next to exactly one from club A and one from club B -/
def p : ℕ := 16

/-- The number of students standing between two from club A -/
def q : ℕ := 46

/-- The number of students standing between two from club B -/
def r : ℕ := 38

/-- The total number of students -/
def total : ℕ := 100

/-- The number of students standing next to at least one from club A -/
def next_to_A : ℕ := 62

/-- The number of students standing next to at least one from club B -/
def next_to_B : ℕ := 54

theorem student_distribution :
  p + q + r = total ∧
  p + q = next_to_A ∧
  p + r = next_to_B :=
by sorry

end NUMINAMATH_CALUDE_student_distribution_l2154_215478


namespace NUMINAMATH_CALUDE_cos_120_degrees_l2154_215476

theorem cos_120_degrees : Real.cos (120 * π / 180) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l2154_215476


namespace NUMINAMATH_CALUDE_inequality_condition_l2154_215470

theorem inequality_condition (a : ℝ) : 
  (∀ x > 1, (Real.exp x) / (x^3) - x - a * Real.log x ≥ 1) ↔ a ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l2154_215470


namespace NUMINAMATH_CALUDE_sarah_score_l2154_215412

theorem sarah_score (s g : ℕ) (h1 : s = g + 30) (h2 : (s + g) / 2 = 108) : s = 123 := by
  sorry

end NUMINAMATH_CALUDE_sarah_score_l2154_215412


namespace NUMINAMATH_CALUDE_comparison_inequality_l2154_215477

theorem comparison_inequality (h1 : 0.83 > 0.73) 
  (h2 : Real.log 0.4 / Real.log 0.5 > Real.log 0.6 / Real.log 0.5)
  (h3 : Real.log 1.6 > Real.log 1.4) : 
  0.75 - 0.1 > 0.75 * 0.1 := by
  sorry

end NUMINAMATH_CALUDE_comparison_inequality_l2154_215477


namespace NUMINAMATH_CALUDE_integers_starting_with_6_divisible_by_25_no_integers_divisible_by_35_without_first_digit_l2154_215460

def starts_with_6 (x : ℕ) : Prop :=
  ∃ n : ℕ, x = 6 * 10^n + (x % 10^n)

def divisible_by_25_without_first_digit (x : ℕ) : Prop :=
  ∃ n : ℕ, (x % 10^n) % 25 = 0

def divisible_by_35_without_first_digit (x : ℕ) : Prop :=
  ∃ n : ℕ, (x % 10^n) % 35 = 0

theorem integers_starting_with_6_divisible_by_25 :
  ∀ x : ℕ, starts_with_6 x ∧ divisible_by_25_without_first_digit x →
    ∃ k : ℕ, x = 625 * 10^k :=
sorry

theorem no_integers_divisible_by_35_without_first_digit :
  ¬ ∃ x : ℕ, divisible_by_35_without_first_digit x :=
sorry

end NUMINAMATH_CALUDE_integers_starting_with_6_divisible_by_25_no_integers_divisible_by_35_without_first_digit_l2154_215460


namespace NUMINAMATH_CALUDE_intersection_with_complement_l2154_215448

-- Define the universe
def U : Set Nat := {0, 1, 2, 3}

-- Define sets A and B
def A : Set Nat := {0, 1, 2}
def B : Set Nat := {0, 2, 3}

-- State the theorem
theorem intersection_with_complement :
  A ∩ (U \ B) = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l2154_215448


namespace NUMINAMATH_CALUDE_triangle_height_inequality_l2154_215417

/-- Given a triangle ABC with sides a, b, c and heights h_a, h_b, h_c, 
    the sum of squares of heights divided by squares of sides is at most 9/2. -/
theorem triangle_height_inequality (a b c h_a h_b h_c : ℝ) 
    (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
    (h_pos_ha : h_a > 0) (h_pos_hb : h_b > 0) (h_pos_hc : h_c > 0)
    (h_triangle : a * h_a = b * h_b ∧ b * h_b = c * h_c) : 
    (h_b^2 + h_c^2) / a^2 + (h_c^2 + h_a^2) / b^2 + (h_a^2 + h_b^2) / c^2 ≤ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_inequality_l2154_215417


namespace NUMINAMATH_CALUDE_a_cube_gt_b_cube_l2154_215474

theorem a_cube_gt_b_cube (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a * abs a > b * abs b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_a_cube_gt_b_cube_l2154_215474


namespace NUMINAMATH_CALUDE_base8_4523_equals_2387_l2154_215445

def base8_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base8_4523_equals_2387 :
  base8_to_base10 [3, 2, 5, 4] = 2387 := by
  sorry

end NUMINAMATH_CALUDE_base8_4523_equals_2387_l2154_215445


namespace NUMINAMATH_CALUDE_range_of_increasing_function_l2154_215431

/-- Given an increasing function f: ℝ → ℝ with f(0) = -1 and f(3) = 1,
    the set of x ∈ ℝ such that |f(x+1)| < 1 is equal to [-1, 2] -/
theorem range_of_increasing_function (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_f_0 : f 0 = -1) 
  (h_f_3 : f 3 = 1) : 
  {x : ℝ | |f (x + 1)| < 1} = Set.Icc (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_increasing_function_l2154_215431


namespace NUMINAMATH_CALUDE_evaluate_expression_l2154_215414

theorem evaluate_expression : (-3)^4 + (-3)^3 + (-3)^2 + 3^2 + 3^3 + 3^4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2154_215414


namespace NUMINAMATH_CALUDE_current_average_is_53_l2154_215435

/-- Represents a cricket player's batting statistics -/
structure CricketStats where
  matchesPlayed : ℕ
  totalRuns : ℕ

/-- Calculates the batting average -/
def battingAverage (stats : CricketStats) : ℚ :=
  stats.totalRuns / stats.matchesPlayed

/-- Theorem: If a player's average becomes 58 after scoring 78 in the 5th match,
    then their current average after 4 matches is 53 -/
theorem current_average_is_53
  (player : CricketStats)
  (h1 : player.matchesPlayed = 4)
  (h2 : battingAverage ⟨5, player.totalRuns + 78⟩ = 58) :
  battingAverage player = 53 := by
  sorry

end NUMINAMATH_CALUDE_current_average_is_53_l2154_215435


namespace NUMINAMATH_CALUDE_complement_A_union_B_l2154_215415

-- Define the sets A and B
def A : Set ℝ := {x | x < -1 ∨ (2 ≤ x ∧ x < 3)}
def B : Set ℝ := {x | -2 ≤ x ∧ x < 4}

-- State the theorem
theorem complement_A_union_B :
  (Set.univ \ A) ∪ B = {x : ℝ | x ≥ -2} := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_l2154_215415


namespace NUMINAMATH_CALUDE_trig_expression_equals_three_l2154_215491

theorem trig_expression_equals_three :
  let sin_60 : ℝ := Real.sqrt 3 / 2
  let tan_45 : ℝ := 1
  let tan_60 : ℝ := Real.sqrt 3
  ∀ (sin_25 cos_25 : ℝ), 
    sin_25^2 + cos_25^2 = 1 →
    sin_25^2 + 2 * sin_60 + tan_45 - tan_60 + cos_25^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_three_l2154_215491


namespace NUMINAMATH_CALUDE_store_sale_profit_store_sale_result_l2154_215403

/-- Calculates the money left after a store's inventory sale --/
theorem store_sale_profit (total_items : ℕ) (retail_price : ℚ) (discount_percent : ℚ) 
  (sold_percent : ℚ) (debt : ℚ) : ℚ :=
  let items_sold := total_items * sold_percent
  let discount_amount := retail_price * discount_percent
  let sale_price := retail_price - discount_amount
  let total_revenue := items_sold * sale_price
  let profit := total_revenue - debt
  profit

/-- Proves that the store has $3000 left after the sale --/
theorem store_sale_result : 
  store_sale_profit 2000 50 0.8 0.9 15000 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_store_sale_profit_store_sale_result_l2154_215403


namespace NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l2154_215493

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![0, 4; 0, -2] := by sorry

end NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l2154_215493


namespace NUMINAMATH_CALUDE_base12_addition_theorem_l2154_215490

-- Define a custom type for base-12 digits
inductive Base12Digit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B

-- Define a type for base-12 numbers
def Base12Number := List Base12Digit

-- Define the two numbers we're adding
def num1 : Base12Number := [Base12Digit.D5, Base12Digit.D2, Base12Digit.D8]
def num2 : Base12Number := [Base12Digit.D2, Base12Digit.D7, Base12Digit.D3]

-- Define the expected result
def expected_result : Base12Number := [Base12Digit.D7, Base12Digit.D9, Base12Digit.B]

-- Function to add two base-12 numbers
def add_base12 (a b : Base12Number) : Base12Number :=
  sorry

theorem base12_addition_theorem :
  add_base12 num1 num2 = expected_result :=
sorry

end NUMINAMATH_CALUDE_base12_addition_theorem_l2154_215490


namespace NUMINAMATH_CALUDE_cookies_problem_l2154_215462

theorem cookies_problem (glenn_cookies : ℕ) (h1 : glenn_cookies = 24) 
  (h2 : ∃ kenny_cookies : ℕ, glenn_cookies = 4 * kenny_cookies) 
  (h3 : ∃ chris_cookies : ℕ, chris_cookies * 2 = kenny_cookies) : 
  glenn_cookies + kenny_cookies + chris_cookies = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_cookies_problem_l2154_215462


namespace NUMINAMATH_CALUDE_triangle_side_length_l2154_215421

/-- Given a triangle ABC where angle A is 6 degrees, angle C is 75 degrees, 
    and side BC has length √3, prove that the length of side AC 
    is equal to (√3 * sin 6°) / sin 45° -/
theorem triangle_side_length (A B C : ℝ) (AC BC : ℝ) : 
  A = 6 * π / 180 →  -- Convert 6° to radians
  C = 75 * π / 180 →  -- Convert 75° to radians
  BC = Real.sqrt 3 →
  AC = (Real.sqrt 3 * Real.sin (6 * π / 180)) / Real.sin (45 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2154_215421


namespace NUMINAMATH_CALUDE_cube_sum_minus_product_eq_2003_l2154_215439

/-- The equation x^3 + y^3 + z^3 - 3xyz = 2003 has only three integer solutions. -/
theorem cube_sum_minus_product_eq_2003 :
  {(x, y, z) : ℤ × ℤ × ℤ | x^3 + y^3 + z^3 - 3*x*y*z = 2003} =
  {(667, 668, 668), (668, 667, 668), (668, 668, 667)} := by
  sorry

#check cube_sum_minus_product_eq_2003

end NUMINAMATH_CALUDE_cube_sum_minus_product_eq_2003_l2154_215439


namespace NUMINAMATH_CALUDE_output_for_15_l2154_215461

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 ≤ 40 then
    step1 + 10
  else
    step1 - 7

theorem output_for_15 : function_machine 15 = 38 := by
  sorry

end NUMINAMATH_CALUDE_output_for_15_l2154_215461


namespace NUMINAMATH_CALUDE_money_division_l2154_215424

theorem money_division (p q r : ℕ) (total : ℕ) :
  p + q + r = total →
  p = 3 * (total / 22) →
  q = 7 * (total / 22) →
  r = 12 * (total / 22) →
  r - q = 3000 →
  q - p = 2400 :=
by sorry

end NUMINAMATH_CALUDE_money_division_l2154_215424


namespace NUMINAMATH_CALUDE_pants_gross_profit_l2154_215488

/-- Calculates the gross profit for a store selling pants -/
theorem pants_gross_profit (purchase_price : ℝ) (markup_percent : ℝ) (price_decrease : ℝ) :
  purchase_price = 210 ∧ 
  markup_percent = 0.25 ∧ 
  price_decrease = 0.20 →
  let original_price := purchase_price / (1 - markup_percent)
  let new_price := original_price * (1 - price_decrease)
  new_price - purchase_price = 14 := by
  sorry

end NUMINAMATH_CALUDE_pants_gross_profit_l2154_215488


namespace NUMINAMATH_CALUDE_sqrt_three_minus_one_over_two_gt_one_third_l2154_215405

theorem sqrt_three_minus_one_over_two_gt_one_third : (Real.sqrt 3 - 1) / 2 > 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_minus_one_over_two_gt_one_third_l2154_215405


namespace NUMINAMATH_CALUDE_geometric_sequence_a6_l2154_215457

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a6 (a : ℕ → ℝ) :
  geometric_sequence a → a 3 = -3 → a 4 = 6 → a 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a6_l2154_215457


namespace NUMINAMATH_CALUDE_jake_roll_combinations_l2154_215427

/-- The number of different combinations of rolls Jake could buy -/
def num_combinations : ℕ := 3

/-- The number of types of rolls available -/
def num_roll_types : ℕ := 3

/-- The total number of rolls Jake needs to purchase -/
def total_rolls : ℕ := 7

/-- The minimum number of each type of roll Jake must purchase -/
def min_per_type : ℕ := 2

theorem jake_roll_combinations :
  num_combinations = 3 ∧
  num_roll_types = 3 ∧
  total_rolls = 7 ∧
  min_per_type = 2 ∧
  total_rolls = num_roll_types * min_per_type + 1 →
  num_combinations = num_roll_types :=
by sorry

end NUMINAMATH_CALUDE_jake_roll_combinations_l2154_215427


namespace NUMINAMATH_CALUDE_long_division_puzzle_l2154_215487

theorem long_division_puzzle :
  let dividend : ℕ := 1089708
  let divisor : ℕ := 12
  let quotient : ℕ := 90909
  dividend = divisor * quotient := by sorry

end NUMINAMATH_CALUDE_long_division_puzzle_l2154_215487


namespace NUMINAMATH_CALUDE_four_correct_propositions_l2154_215467

theorem four_correct_propositions (x y : ℝ) :
  (((x ≠ 0 ∨ y ≠ 0) → x^2 + y^2 ≠ 0) ∧
   ((x^2 + y^2 ≠ 0) → (x ≠ 0 ∨ y ≠ 0)) ∧
   (¬((x ≠ 0 ∨ y ≠ 0) → x^2 + y^2 ≠ 0)) ∧
   ((x^2 + y^2 = 0) → (x = 0 ∧ y = 0))) :=
by sorry

end NUMINAMATH_CALUDE_four_correct_propositions_l2154_215467


namespace NUMINAMATH_CALUDE_negative_fractions_in_list_l2154_215451

def given_numbers : List ℚ := [5, -1, 0, -6, 125.73, 0.3, -3.5, -0.72, 5.25]

def is_negative_fraction (x : ℚ) : Prop := x < 0 ∧ x ≠ ⌊x⌋

theorem negative_fractions_in_list :
  ∀ x ∈ given_numbers, is_negative_fraction x ↔ x = -3.5 ∨ x = -0.72 := by
  sorry

end NUMINAMATH_CALUDE_negative_fractions_in_list_l2154_215451


namespace NUMINAMATH_CALUDE_smallest_cube_ending_580_l2154_215413

theorem smallest_cube_ending_580 : 
  ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 580 ∧ ∀ m : ℕ, m > 0 ∧ m^3 % 1000 = 580 → m ≥ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_580_l2154_215413


namespace NUMINAMATH_CALUDE_unsold_books_l2154_215459

theorem unsold_books (initial_stock : ℕ) (mon tue wed thu fri : ℕ) :
  initial_stock = 800 →
  mon = 60 →
  tue = 10 →
  wed = 20 →
  thu = 44 →
  fri = 66 →
  initial_stock - (mon + tue + wed + thu + fri) = 600 :=
by
  sorry

end NUMINAMATH_CALUDE_unsold_books_l2154_215459


namespace NUMINAMATH_CALUDE_twelfth_day_is_monday_l2154_215442

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month with its properties -/
structure Month where
  firstDay : DayOfWeek
  lastDay : DayOfWeek
  fridayCount : Nat
  dayCount : Nat

/-- Axiom: The month has exactly 5 Fridays -/
axiom five_fridays (m : Month) : m.fridayCount = 5

/-- Axiom: The first day of the month is not a Friday -/
axiom first_not_friday (m : Month) : m.firstDay ≠ DayOfWeek.Friday

/-- Axiom: The last day of the month is not a Friday -/
axiom last_not_friday (m : Month) : m.lastDay ≠ DayOfWeek.Friday

/-- Function to get the day of week for a given day number -/
def getDayOfWeek (m : Month) (day : Nat) : DayOfWeek :=
  sorry

/-- Theorem: The 12th day of the month is a Monday -/
theorem twelfth_day_is_monday (m : Month) :
  getDayOfWeek m 12 = DayOfWeek.Monday :=
sorry

end NUMINAMATH_CALUDE_twelfth_day_is_monday_l2154_215442


namespace NUMINAMATH_CALUDE_inequality_proof_l2154_215420

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  (1 + a) * (1 + b) * (1 + c) ≥ 8 * (1 - a) * (1 - b) * (1 - c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2154_215420


namespace NUMINAMATH_CALUDE_unique_number_l2154_215402

def is_valid_increase (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 10 * a + b ∧ a < 10 ∧ b < 10 ∧
  ∃ (c d : ℕ), (c = 2 ∨ c = 4) ∧ (d = 2 ∨ d = 4) ∧
  4 * n = 10 * (a + c) + (b + d)

theorem unique_number : ∀ n : ℕ, is_valid_increase n ↔ n = 14 := by sorry

end NUMINAMATH_CALUDE_unique_number_l2154_215402


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2154_215410

theorem rectangle_perimeter (length width : ℝ) (h_ratio : length / width = 4 / 3) (h_area : length * width = 972) :
  2 * (length + width) = 126 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2154_215410


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l2154_215443

/-- A geometric sequence is defined by its first term and common ratio -/
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

/-- The seventh term of a geometric sequence with first term -4 and second term 8 is -256 -/
theorem seventh_term_of_geometric_sequence :
  let a₁ : ℝ := -4
  let a₂ : ℝ := 8
  let r : ℝ := a₂ / a₁
  geometric_sequence a₁ r 7 = -256 := by
sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l2154_215443


namespace NUMINAMATH_CALUDE_units_digit_of_150_factorial_l2154_215475

theorem units_digit_of_150_factorial (n : ℕ) : n = 150 → n.factorial % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_150_factorial_l2154_215475


namespace NUMINAMATH_CALUDE_circle_center_l2154_215432

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 - 8*y - 16 = 0

-- Define the center of a circle
def is_center (h k : ℝ) (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ r, ∀ x y, eq x y ↔ (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem circle_center :
  is_center 3 4 circle_equation :=
sorry

end NUMINAMATH_CALUDE_circle_center_l2154_215432


namespace NUMINAMATH_CALUDE_joe_market_spend_l2154_215430

/-- Calculates the total cost of Joe's market purchases -/
def market_total_cost (orange_price : ℚ) (juice_price : ℚ) (honey_price : ℚ) (plant_pair_price : ℚ)
  (orange_count : ℕ) (juice_count : ℕ) (honey_count : ℕ) (plant_count : ℕ) : ℚ :=
  orange_price * orange_count +
  juice_price * juice_count +
  honey_price * honey_count +
  plant_pair_price * (plant_count / 2)

/-- Theorem stating that Joe's total market spend is $68 -/
theorem joe_market_spend :
  market_total_cost 4.5 0.5 5 18 3 7 3 4 = 68 := by
  sorry

end NUMINAMATH_CALUDE_joe_market_spend_l2154_215430


namespace NUMINAMATH_CALUDE_smallest_three_digit_square_append_l2154_215464

theorem smallest_three_digit_square_append : ∃ (n : ℕ), 
  (n = 183) ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m < n → ¬(∃ k : ℕ, 1000 * m + (m + 1) = k * k)) ∧
  (∃ k : ℕ, 1000 * n + (n + 1) = k * k) :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_square_append_l2154_215464


namespace NUMINAMATH_CALUDE_line_through_point_unique_l2154_215437

/-- A line passing through a point -/
def line_passes_through_point (k : ℝ) : Prop :=
  2 * k * (-1/2) - 3 = -7 * 3

/-- The value of k that satisfies the line equation -/
def k_value : ℝ := 18

/-- Theorem: k_value is the unique real number that satisfies the line equation -/
theorem line_through_point_unique : 
  line_passes_through_point k_value ∧ 
  ∀ k : ℝ, line_passes_through_point k → k = k_value :=
sorry

end NUMINAMATH_CALUDE_line_through_point_unique_l2154_215437


namespace NUMINAMATH_CALUDE_sum_five_consecutive_integers_l2154_215497

theorem sum_five_consecutive_integers (n : ℤ) : 
  (n - 2) + (n - 1) + n + (n + 1) + (n + 2) = 5 * n := by
  sorry

end NUMINAMATH_CALUDE_sum_five_consecutive_integers_l2154_215497


namespace NUMINAMATH_CALUDE_future_age_relationship_l2154_215496

/-- Represents the current ages and future relationship between Rehana, Jacob, and Phoebe -/
theorem future_age_relationship (x : ℕ) : 
  let rehana_current_age : ℕ := 25
  let jacob_current_age : ℕ := 3
  let phoebe_current_age : ℕ := jacob_current_age * 5 / 3
  x = 5 ↔ 
    rehana_current_age + x = 3 * (phoebe_current_age + x) ∧
    x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_future_age_relationship_l2154_215496


namespace NUMINAMATH_CALUDE_focal_length_determination_l2154_215494

/-- Represents a converging lens with a right isosceles triangle -/
structure LensSystem where
  focalLength : ℝ
  triangleArea : ℝ
  imageArea : ℝ

/-- The conditions of the lens system -/
def validLensSystem (s : LensSystem) : Prop :=
  s.triangleArea = 8 ∧ s.imageArea = s.triangleArea / 2

/-- The theorem statement -/
theorem focal_length_determination (s : LensSystem) 
  (h : validLensSystem s) : s.focalLength = 2 := by
  sorry

end NUMINAMATH_CALUDE_focal_length_determination_l2154_215494


namespace NUMINAMATH_CALUDE_sum_of_exponents_15_factorial_l2154_215453

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largestPerfectSquareDivisor (n : ℕ) : ℕ :=
  sorry

def sumOfExponents (n : ℕ) : ℕ :=
  sorry

theorem sum_of_exponents_15_factorial :
  sumOfExponents (largestPerfectSquareDivisor (factorial 15).sqrt) = 10 :=
sorry

end NUMINAMATH_CALUDE_sum_of_exponents_15_factorial_l2154_215453


namespace NUMINAMATH_CALUDE_mod_inverse_remainder_l2154_215486

theorem mod_inverse_remainder (y : ℕ+) : 
  (7 * y.val) % 29 = 1 → (8 + y.val) % 29 = 4 := by sorry

end NUMINAMATH_CALUDE_mod_inverse_remainder_l2154_215486


namespace NUMINAMATH_CALUDE_ball_placement_count_is_144_l2154_215434

/-- The number of ways to place four different balls into four numbered boxes with exactly one box remaining empty -/
def ballPlacementCount : ℕ := 144

/-- The number of different balls -/
def numBalls : ℕ := 4

/-- The number of boxes -/
def numBoxes : ℕ := 4

/-- Theorem stating that the number of ways to place four different balls into four numbered boxes with exactly one box remaining empty is 144 -/
theorem ball_placement_count_is_144 : ballPlacementCount = 144 := by sorry

end NUMINAMATH_CALUDE_ball_placement_count_is_144_l2154_215434


namespace NUMINAMATH_CALUDE_even_function_quadratic_l2154_215423

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 1

theorem even_function_quadratic 
  (a b : ℝ) 
  (h_even : ∀ x, f a b x = f a b (-x))
  (h_domain : Set.Icc (-1 - a) (2 * a) ⊆ Set.range (f a b)) :
  f a b (2 * a - b) = 5 := by
sorry

end NUMINAMATH_CALUDE_even_function_quadratic_l2154_215423


namespace NUMINAMATH_CALUDE_tuesday_to_monday_ratio_l2154_215419

/-- Represents the number of crates of eggs sold on each day of the week --/
structure EggSales where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- Theorem stating the ratio of Tuesday's sales to Monday's sales --/
theorem tuesday_to_monday_ratio (sales : EggSales) : 
  sales.monday = 5 ∧ 
  sales.wednesday = sales.tuesday - 2 ∧ 
  sales.thursday = sales.tuesday / 2 ∧
  sales.monday + sales.tuesday + sales.wednesday + sales.thursday = 28 →
  sales.tuesday = 2 * sales.monday := by
  sorry

#check tuesday_to_monday_ratio

end NUMINAMATH_CALUDE_tuesday_to_monday_ratio_l2154_215419


namespace NUMINAMATH_CALUDE_polygon_sides_difference_l2154_215401

/-- Number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The problem statement -/
theorem polygon_sides_difference : ∃ (m : ℕ),
  m > 3 ∧
  diagonals m = 3 * diagonals (m - 3) ∧
  ∃ (x : ℕ),
    diagonals (m + x) = 7 * diagonals m ∧
    x = 12 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_difference_l2154_215401


namespace NUMINAMATH_CALUDE_factorization_analysis_l2154_215446

theorem factorization_analysis (x y a b : ℝ) :
  (x^4 - y^4 = (x^2 + y^2) * (x + y) * (x - y)) ∧
  (x^3*y - 2*x^2*y^2 + x*y^3 = x*y*(x - y)^2) ∧
  (4*x^2 - 4*x + 1 = (2*x - 1)^2) ∧
  (4*(a - b)^2 + 1 + 4*(a - b) = (2*a - 2*b + 1)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_factorization_analysis_l2154_215446


namespace NUMINAMATH_CALUDE_goldfish_to_pretzel_ratio_l2154_215449

/-- Proves that the ratio of goldfish to pretzels is 4:1 given the specified conditions --/
theorem goldfish_to_pretzel_ratio :
  let total_pretzels : ℕ := 64
  let total_suckers : ℕ := 32
  let num_kids : ℕ := 16
  let items_per_baggie : ℕ := 22
  let total_items : ℕ := num_kids * items_per_baggie
  let total_goldfish : ℕ := total_items - total_pretzels - total_suckers
  total_goldfish / total_pretzels = 4 :=
by sorry

end NUMINAMATH_CALUDE_goldfish_to_pretzel_ratio_l2154_215449


namespace NUMINAMATH_CALUDE_area_is_14_4_l2154_215404

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoidWithInscribedCircle where
  /-- Distance from circle center to one end of a non-parallel side -/
  d1 : ℝ
  /-- Distance from circle center to the other end of the same non-parallel side -/
  d2 : ℝ
  /-- Assumption that d1 and d2 are positive -/
  d1_pos : d1 > 0
  d2_pos : d2 > 0

/-- The area of the isosceles trapezoid with an inscribed circle -/
def area (t : IsoscelesTrapezoidWithInscribedCircle) : ℝ :=
  14.4

/-- Theorem stating that the area of the isosceles trapezoid with an inscribed circle is 14.4 cm² -/
theorem area_is_14_4 (t : IsoscelesTrapezoidWithInscribedCircle) 
    (h1 : t.d1 = 2) (h2 : t.d2 = 4) : area t = 14.4 := by
  sorry

end NUMINAMATH_CALUDE_area_is_14_4_l2154_215404


namespace NUMINAMATH_CALUDE_adam_change_l2154_215409

-- Define the amount Adam has
def adam_amount : ℚ := 5.00

-- Define the cost of the airplane
def airplane_cost : ℚ := 4.28

-- Theorem stating the change Adam will receive
theorem adam_change : adam_amount - airplane_cost = 0.72 := by
  sorry

end NUMINAMATH_CALUDE_adam_change_l2154_215409


namespace NUMINAMATH_CALUDE_certain_number_problem_l2154_215455

theorem certain_number_problem (x : ℝ) (y : ℝ) : 
  x = y + 0.5 * y → x = 132 → y = 88 := by sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2154_215455


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l2154_215498

open Real

noncomputable def f (x : ℝ) : ℝ := -log (x^2 - 3*x + 2)

theorem f_increasing_on_interval :
  StrictMonoOn f (Set.Iio 1) := by sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l2154_215498


namespace NUMINAMATH_CALUDE_base_8_4512_equals_2378_l2154_215438

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_8_4512_equals_2378 : 
  base_8_to_10 [2, 1, 5, 4] = 2378 := by sorry

end NUMINAMATH_CALUDE_base_8_4512_equals_2378_l2154_215438


namespace NUMINAMATH_CALUDE_derivative_at_three_l2154_215489

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem derivative_at_three : 
  (deriv f) 3 = 6 := by sorry

end NUMINAMATH_CALUDE_derivative_at_three_l2154_215489


namespace NUMINAMATH_CALUDE_range_of_m_for_real_solutions_l2154_215426

theorem range_of_m_for_real_solutions (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, 4 * Real.cos y + Real.sin y ^ 2 + m - 4 = 0) →
  0 ≤ m ∧ m ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_for_real_solutions_l2154_215426


namespace NUMINAMATH_CALUDE_vector_addition_l2154_215485

theorem vector_addition :
  let v1 : Fin 3 → ℝ := ![5, -3, 2]
  let v2 : Fin 3 → ℝ := ![-4, 8, -1]
  v1 + v2 = ![1, 5, 1] := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l2154_215485


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l2154_215479

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The product of 125 and 960 -/
def product : ℕ := 125 * 960

theorem product_trailing_zeros :
  trailingZeros product = 4 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l2154_215479


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2154_215454

/-- The eccentricity of a hyperbola given specific conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let parabola := fun (x y : ℝ) ↦ y^2 = 4 * x
  let directrix := fun (x : ℝ) ↦ x = -1
  let asymptote1 := fun (x y : ℝ) ↦ y = (b / a) * x
  let asymptote2 := fun (x y : ℝ) ↦ y = -(b / a) * x
  let triangle_area := 2 * Real.sqrt 3
  let eccentricity := Real.sqrt ((a^2 + b^2) / a^2)
  (∃ A B : ℝ × ℝ, 
    directrix A.1 ∧ asymptote1 A.1 A.2 ∧
    directrix B.1 ∧ asymptote2 B.1 B.2 ∧
    (1/2) * (A.2 - B.2) = triangle_area) →
  eccentricity = Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2154_215454


namespace NUMINAMATH_CALUDE_special_remainder_property_l2154_215418

theorem special_remainder_property (n : ℕ) : 
  (∀ q : ℕ, q > 0 → n % q^2 < q^(q^2) / 2) ↔ (n = 1 ∨ n = 4) :=
by sorry

end NUMINAMATH_CALUDE_special_remainder_property_l2154_215418


namespace NUMINAMATH_CALUDE_correct_scientific_notation_l2154_215481

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coefficient_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be expressed in scientific notation -/
def number : ℕ := 250000

/-- The scientific notation representation of the number -/
def scientific_representation : ScientificNotation := {
  coefficient := 2.5,
  exponent := 5,
  coefficient_range := by sorry
}

/-- Theorem stating that the scientific notation representation is correct -/
theorem correct_scientific_notation :
  (scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent) = number := by
  sorry

end NUMINAMATH_CALUDE_correct_scientific_notation_l2154_215481


namespace NUMINAMATH_CALUDE_theater_seat_count_l2154_215452

/-- Represents a theater with an arithmetic progression of seats per row -/
structure Theater where
  first_row_seats : ℕ
  seat_increase : ℕ
  last_row_seats : ℕ

/-- Calculates the number of rows in the theater -/
def number_of_rows (t : Theater) : ℕ :=
  (t.last_row_seats - t.first_row_seats) / t.seat_increase + 1

/-- Calculates the total number of seats in the theater -/
def total_seats (t : Theater) : ℕ :=
  let n := number_of_rows t
  n * (t.first_row_seats + t.last_row_seats) / 2

/-- The theater described in the problem -/
def problem_theater : Theater :=
  { first_row_seats := 15
  , seat_increase := 2
  , last_row_seats := 53 }

theorem theater_seat_count :
  total_seats problem_theater = 680 := by
  sorry

end NUMINAMATH_CALUDE_theater_seat_count_l2154_215452


namespace NUMINAMATH_CALUDE_garage_wheels_count_l2154_215408

/-- The number of wheels in Connor's garage --/
def total_wheels (bicycles cars motorcycles : ℕ) : ℕ :=
  2 * bicycles + 4 * cars + 2 * motorcycles

/-- Theorem: The total number of wheels in Connor's garage is 90 --/
theorem garage_wheels_count :
  total_wheels 20 10 5 = 90 := by
  sorry

end NUMINAMATH_CALUDE_garage_wheels_count_l2154_215408


namespace NUMINAMATH_CALUDE_average_speed_calculation_l2154_215484

def distance : ℝ := 360
def time : ℝ := 4.5

theorem average_speed_calculation : distance / time = 80 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l2154_215484


namespace NUMINAMATH_CALUDE_ratio_problem_l2154_215411

theorem ratio_problem (a b c : ℝ) (h1 : a / b = 6 / 5) (h2 : b / c = 8 / 7) :
  (7 * a + 6 * b + 5 * c) / (7 * a - 6 * b + 5 * c) = 751 / 271 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2154_215411


namespace NUMINAMATH_CALUDE_x_142_equals_1995_and_unique_l2154_215473

def p (x : ℕ) : ℕ := sorry

def q (x : ℕ) : ℕ := sorry

def x : ℕ → ℕ
  | 0 => 1
  | n + 1 => (x n * p (x n)) / q (x n)

theorem x_142_equals_1995_and_unique :
  x 142 = 1995 ∧ ∀ n : ℕ, n ≠ 142 → x n ≠ 1995 := by sorry

end NUMINAMATH_CALUDE_x_142_equals_1995_and_unique_l2154_215473


namespace NUMINAMATH_CALUDE_journey_distance_ratio_l2154_215480

/-- Given a journey where:
  - The initial distance traveled is 20 hours at 30 kilometers per hour
  - After a setback, the traveler is one-third of the way to the destination
  Prove that the ratio of the initial distance to the total distance is 1/3 -/
theorem journey_distance_ratio :
  ∀ (initial_speed : ℝ) (initial_time : ℝ) (total_distance : ℝ),
    initial_speed = 30 →
    initial_time = 20 →
    initial_speed * initial_time = (1/3) * total_distance →
    (initial_speed * initial_time) / total_distance = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_ratio_l2154_215480


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l2154_215492

theorem cube_sum_theorem (p q r : ℝ) 
  (sum_eq : p + q + r = 3)
  (sum_prod_eq : p * q + q * r + r * p = 11)
  (prod_eq : p * q * r = -6) :
  p^3 + q^3 + r^3 = -90 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l2154_215492


namespace NUMINAMATH_CALUDE_circumcircle_area_of_triangle_ABP_l2154_215499

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define points
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define the condition |AP⃗|cos<AP⃗, AF₂⃗> = |AF₂⃗|
def condition_AP_AF₂ (P : ℝ × ℝ) : Prop :=
  let AP := (P.1 - A.1, P.2 - A.2)
  let AF₂ := (F₂.1 - A.1, F₂.2 - A.2)
  Real.sqrt (AP.1^2 + AP.2^2) * (AP.1 * AF₂.1 + AP.2 * AF₂.2) / 
    (Real.sqrt (AP.1^2 + AP.2^2) * Real.sqrt (AF₂.1^2 + AF₂.2^2)) = 
    Real.sqrt (AF₂.1^2 + AF₂.2^2)

-- Define the theorem
theorem circumcircle_area_of_triangle_ABP (P : ℝ × ℝ) 
  (h₁ : hyperbola P.1 P.2)
  (h₂ : P.1 > B.1)  -- P is on the right branch
  (h₃ : condition_AP_AF₂ P) :
  ∃ (R : ℝ), R > 0 ∧ π * R^2 = 5 * π := by sorry

end NUMINAMATH_CALUDE_circumcircle_area_of_triangle_ABP_l2154_215499


namespace NUMINAMATH_CALUDE_subtract_negative_two_l2154_215495

theorem subtract_negative_two : 0 - (-2) = 2 := by sorry

end NUMINAMATH_CALUDE_subtract_negative_two_l2154_215495


namespace NUMINAMATH_CALUDE_cube_order_preserving_l2154_215447

theorem cube_order_preserving (a b : ℝ) : a^3 > b^3 → a > b := by
  sorry

end NUMINAMATH_CALUDE_cube_order_preserving_l2154_215447


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l2154_215465

theorem largest_n_divisibility : ∃ (n : ℕ), n = 302 ∧ 
  (∀ m : ℕ, m > 302 → ¬(m + 11 ∣ m^3 + 101)) ∧
  (302 + 11 ∣ 302^3 + 101) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l2154_215465


namespace NUMINAMATH_CALUDE_divisibility_by_four_l2154_215463

theorem divisibility_by_four (n : ℕ+) :
  4 ∣ (n * Nat.choose (2 * n) n) ↔ ¬∃ k : ℕ, n = 2^k := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_four_l2154_215463


namespace NUMINAMATH_CALUDE_probability_five_white_two_red_l2154_215440

def white_balls : ℕ := 7
def black_balls : ℕ := 8
def red_balls : ℕ := 3
def total_balls : ℕ := white_balls + black_balls + red_balls
def drawn_balls : ℕ := 7

theorem probability_five_white_two_red :
  (Nat.choose white_balls 5 * Nat.choose red_balls 2) / Nat.choose total_balls drawn_balls = 63 / 31824 :=
sorry

end NUMINAMATH_CALUDE_probability_five_white_two_red_l2154_215440
