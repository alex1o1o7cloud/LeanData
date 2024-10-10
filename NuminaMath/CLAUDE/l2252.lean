import Mathlib

namespace characterization_of_f_l2252_225234

-- Define the property for the function
def satisfies_property (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, (f x * f y) ∣ ((1 + 2 * x) * f y + (1 + 2 * y) * f x)

-- Define strictly increasing function
def strictly_increasing (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, x < y → f x < f y

-- Main theorem
theorem characterization_of_f :
  ∀ f : ℕ → ℕ, strictly_increasing f → satisfies_property f →
  (∀ x : ℕ, f x = 2 * x + 1) ∨ (∀ x : ℕ, f x = 4 * x + 2) :=
sorry

end characterization_of_f_l2252_225234


namespace rectangle_perimeter_l2252_225264

theorem rectangle_perimeter (L B : ℝ) (h1 : L - B = 23) (h2 : L * B = 2030) :
  2 * (L + B) = 186 := by
  sorry

end rectangle_perimeter_l2252_225264


namespace shaded_area_semicircles_l2252_225285

/-- The area of the shaded region in the given semicircle configuration -/
theorem shaded_area_semicircles (r_ADB r_BEC : ℝ) (h_ADB : r_ADB = 2) (h_BEC : r_BEC = 3) : 
  let r_DFE := (r_ADB + r_BEC) / 2
  (π * r_ADB^2 / 2 + π * r_BEC^2 / 2) - (π * r_DFE^2 / 2) = 3.375 * π := by
  sorry

end shaded_area_semicircles_l2252_225285


namespace least_clock_equivalent_hour_l2252_225251

theorem least_clock_equivalent_hour : 
  ∃ (h : ℕ), h > 6 ∧ 
             h % 12 = (h^2) % 12 ∧ 
             h % 12 = (h^3) % 12 ∧ 
             (∀ (k : ℕ), k > 6 ∧ k < h → 
               (k % 12 ≠ (k^2) % 12 ∨ k % 12 ≠ (k^3) % 12)) :=
by
  -- Proof goes here
  sorry

end least_clock_equivalent_hour_l2252_225251


namespace units_digit_of_product_l2252_225286

def product : ℕ := 1 * 3 * 5 * 79 * 97 * 113

theorem units_digit_of_product :
  (product % 10) = 5 := by sorry

end units_digit_of_product_l2252_225286


namespace common_root_of_three_equations_l2252_225294

theorem common_root_of_three_equations (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h_ab : ∃ x : ℝ, a * x^11 + b * x^4 + c = 0 ∧ b * x^11 + c * x^4 + a = 0)
  (h_bc : ∃ x : ℝ, b * x^11 + c * x^4 + a = 0 ∧ c * x^11 + a * x^4 + b = 0)
  (h_ca : ∃ x : ℝ, c * x^11 + a * x^4 + b = 0 ∧ a * x^11 + b * x^4 + c = 0) :
  a * 1^11 + b * 1^4 + c = 0 ∧ b * 1^11 + c * 1^4 + a = 0 ∧ c * 1^11 + a * 1^4 + b = 0 :=
by sorry

end common_root_of_three_equations_l2252_225294


namespace fedya_deposit_l2252_225238

theorem fedya_deposit (n : ℕ) (X : ℕ) : 
  n < 30 →
  X * (100 - n) = 847 * 100 →
  X = 1100 := by
sorry

end fedya_deposit_l2252_225238


namespace highest_seat_number_is_44_l2252_225230

/-- Calculates the highest seat number in a systematic sample -/
def highest_seat_number (total_students : ℕ) (sample_size : ℕ) (first_student : ℕ) : ℕ :=
  let interval := total_students / sample_size
  first_student + (sample_size - 1) * interval

/-- Theorem: The highest seat number in the sample is 44 -/
theorem highest_seat_number_is_44 :
  highest_seat_number 56 4 2 = 44 := by
  sorry

#eval highest_seat_number 56 4 2

end highest_seat_number_is_44_l2252_225230


namespace evaluate_expression_l2252_225226

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end evaluate_expression_l2252_225226


namespace complementary_angles_l2252_225284

theorem complementary_angles (A B : ℝ) : 
  A + B = 90 →  -- angles are complementary
  A = 5 * B →   -- measure of A is 5 times B
  A = 75 :=     -- measure of A is 75 degrees
by
  sorry

end complementary_angles_l2252_225284


namespace bobbit_worm_aquarium_l2252_225279

def fish_count (initial_fish : ℕ) (daily_consumption : ℕ) (added_fish : ℕ) (days_before_adding : ℕ) (total_days : ℕ) : ℕ :=
  initial_fish - (daily_consumption * total_days) + added_fish

theorem bobbit_worm_aquarium (initial_fish : ℕ) (daily_consumption : ℕ) (added_fish : ℕ) (days_before_adding : ℕ) (total_days : ℕ)
  (h1 : initial_fish = 60)
  (h2 : daily_consumption = 2)
  (h3 : added_fish = 8)
  (h4 : days_before_adding = 14)
  (h5 : total_days = 21) :
  fish_count initial_fish daily_consumption added_fish days_before_adding total_days = 26 := by
  sorry

end bobbit_worm_aquarium_l2252_225279


namespace max_profit_at_50_l2252_225261

/-- Profit function given the price increase x -/
def profit (x : ℕ) : ℤ := -5 * x^2 + 500 * x + 20000

/-- The maximum allowed price increase -/
def max_increase : ℕ := 200

/-- Theorem stating the maximum profit and the price increase that achieves it -/
theorem max_profit_at_50 :
  ∃ (x : ℕ), x ≤ max_increase ∧ 
  profit x = 32500 ∧ 
  ∀ (y : ℕ), y ≤ max_increase → profit y ≤ profit x :=
sorry

end max_profit_at_50_l2252_225261


namespace max_value_of_g_l2252_225263

/-- The function g(x) = 4x - x^4 -/
def g (x : ℝ) : ℝ := 4*x - x^4

/-- The theorem stating that the maximum value of g(x) on [0, 2] is 3 -/
theorem max_value_of_g :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧
  g c = 3 :=
sorry

end max_value_of_g_l2252_225263


namespace fraction_simplification_l2252_225237

theorem fraction_simplification (b c d x y : ℝ) :
  (c * x * (b^2 * x^3 + 3 * b^2 * y^3 + c^3 * y^3) + d * y * (b^2 * x^3 + 3 * c^3 * x^3 + c^3 * y^3)) / (c * x + d * y) =
  b^2 * x^3 + 3 * c^2 * x * y^3 + c^3 * y^3 :=
by sorry

end fraction_simplification_l2252_225237


namespace no_integer_b_with_four_integer_solutions_l2252_225295

theorem no_integer_b_with_four_integer_solutions : 
  ¬ ∃ b : ℤ, ∃ x₁ x₂ x₃ x₄ : ℤ, 
    (∀ x : ℤ, x^2 + b*x + 1 ≤ 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) ∧
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) :=
by sorry

end no_integer_b_with_four_integer_solutions_l2252_225295


namespace total_statues_l2252_225246

/-- The length of the street in meters -/
def street_length : ℕ := 1650

/-- The interval between statues in meters -/
def statue_interval : ℕ := 50

/-- The number of sides of the street with statues -/
def sides : ℕ := 2

theorem total_statues : 
  (street_length / statue_interval + 1) * sides = 68 := by
  sorry

end total_statues_l2252_225246


namespace prob_two_high_temp_is_half_l2252_225207

/-- Represents a 3-digit number where each digit is either 0-5 or 6-9 -/
def ThreeDayPeriod := Fin 1000

/-- The probability of a digit being 0-5 (representing a high temperature warning) -/
def p_high_temp : ℚ := 3/5

/-- The number of random samples generated -/
def num_samples : ℕ := 20

/-- Counts the number of digits in a ThreeDayPeriod that are 0-5 -/
def count_high_temp (n : ThreeDayPeriod) : ℕ := sorry

/-- The event of exactly 2 high temperature warnings in a 3-day period -/
def two_high_temp (n : ThreeDayPeriod) : Prop := count_high_temp n = 2

/-- The probability of the event two_high_temp -/
def prob_two_high_temp : ℚ := sorry

theorem prob_two_high_temp_is_half : prob_two_high_temp = 1/2 := by sorry

end prob_two_high_temp_is_half_l2252_225207


namespace angle_B_measure_l2252_225299

theorem angle_B_measure (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 3 →
  b = Real.sqrt 2 →
  A = 60 * π / 180 →
  -- Sine Rule
  a / Real.sin A = b / Real.sin B →
  -- Triangle inequality (ensuring it's a valid triangle)
  a + b > c ∧ b + c > a ∧ c + a > b →
  -- Sum of angles in a triangle is π
  A + B + C = π →
  B = 45 * π / 180 := by sorry

end angle_B_measure_l2252_225299


namespace seventh_term_value_l2252_225265

def sequence_with_sum_rule (a : ℕ → ℕ) : Prop :=
  a 1 = 5 ∧ a 4 = 13 ∧ a 6 = 40 ∧
  ∀ n ≥ 4, a n = a (n-3) + a (n-2) + a (n-1)

theorem seventh_term_value (a : ℕ → ℕ) (h : sequence_with_sum_rule a) : a 7 = 74 := by
  sorry

end seventh_term_value_l2252_225265


namespace base6_403_greater_than_base8_217_l2252_225240

/-- Converts a number from base 6 to decimal --/
def base6ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a number from base 8 to decimal --/
def base8ToDecimal (n : ℕ) : ℕ := sorry

theorem base6_403_greater_than_base8_217 :
  base6ToDecimal 403 > base8ToDecimal 217 := by sorry

end base6_403_greater_than_base8_217_l2252_225240


namespace tan_value_from_trig_equation_l2252_225276

theorem tan_value_from_trig_equation (x : Real) 
  (h1 : 0 < x) (h2 : x < π/2) 
  (h3 : (Real.sin x)^4 / 9 + (Real.cos x)^4 / 4 = 1/13) : 
  Real.tan x = 3/2 := by
  sorry

end tan_value_from_trig_equation_l2252_225276


namespace extraneous_root_value_l2252_225229

theorem extraneous_root_value (x m : ℝ) : 
  ((x + 7) / (x - 1) + 2 = (m + 5) / (x - 1)) ∧ 
  (x = 1) →
  m = 3 :=
by sorry

end extraneous_root_value_l2252_225229


namespace arithmetic_sequence_sum_l2252_225258

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a → a 3 + a 7 = 37 → a 2 + a 4 + a 6 + a 8 = 74 := by
  sorry

end arithmetic_sequence_sum_l2252_225258


namespace geometric_means_l2252_225223

theorem geometric_means (a b : ℝ) (p : ℕ) (ha : 0 < a) (hb : a < b) :
  let r := (b / a) ^ (1 / (p + 1 : ℝ))
  ∀ k : ℕ, k ≥ 1 → k ≤ p →
    a * r ^ k = a * (b / a) ^ (k / (p + 1 : ℝ)) :=
sorry

end geometric_means_l2252_225223


namespace biology_homework_pages_l2252_225241

/-- The number of pages of math homework -/
def math_pages : ℕ := 8

/-- The total number of pages of math and biology homework -/
def total_math_biology_pages : ℕ := 11

/-- The number of pages of biology homework -/
def biology_pages : ℕ := total_math_biology_pages - math_pages

theorem biology_homework_pages : biology_pages = 3 := by
  sorry

end biology_homework_pages_l2252_225241


namespace min_omega_l2252_225252

theorem min_omega (f : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = 2 * Real.sin (ω * x)) →
  ω > 0 →
  (∀ x ∈ Set.Icc (-π/3) (π/4), f x ≥ -2) →
  (∃ x ∈ Set.Icc (-π/3) (π/4), f x = -2) →
  ω ≥ 3/2 ∧ ∀ ω' ≥ 3/2, ∃ x ∈ Set.Icc (-π/3) (π/4), 2 * Real.sin (ω' * x) = -2 :=
by sorry

end min_omega_l2252_225252


namespace coffee_pastry_budget_l2252_225200

theorem coffee_pastry_budget (B : ℝ) (c p : ℝ) 
  (hc : c = (1/4) * (B - p)) 
  (hp : p = (1/10) * (B - c)) : 
  c + p = (4/13) * B := by
sorry

end coffee_pastry_budget_l2252_225200


namespace equation_real_roots_l2252_225271

theorem equation_real_roots (a : ℝ) : 
  (∃ x : ℝ, 9^(-|x - 2|) - 4 * 3^(-|x - 2|) - a = 0) ↔ -3 ≤ a ∧ a < 0 := by
  sorry

end equation_real_roots_l2252_225271


namespace range_of_a_l2252_225298

theorem range_of_a (p : ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) 
                   (q : ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) :
  a ≤ -2 ∨ a = 1 := by
  sorry

end range_of_a_l2252_225298


namespace a_4_equals_18_l2252_225267

def sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (List.range n).map a |>.sum

theorem a_4_equals_18 (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n : ℕ, S n = sequence_sum a n) →
  a 1 = 1 →
  (∀ n : ℕ+, a (n + 1) = 2 * S n) →
  a 4 = 18 := by
sorry

end a_4_equals_18_l2252_225267


namespace quadratic_inequality_l2252_225274

theorem quadratic_inequality (x : ℝ) (h : x ∈ Set.Icc 0 1) :
  |x^2 - x + 1/8| ≤ 1/8 := by
  sorry

end quadratic_inequality_l2252_225274


namespace digit_2007_in_2003_digit_number_l2252_225253

/-- The sequence of digits formed by concatenating positive integers -/
def digit_sequence : ℕ → ℕ := sorry

/-- The function G(n) that calculates the number of digits preceding 10^n in the sequence -/
def G (n : ℕ) : ℕ := sorry

/-- The function f(n) that returns the number of digits in the number where the 10^n-th digit occurs -/
def f (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 10^2007-th digit occurs in a 2003-digit number -/
theorem digit_2007_in_2003_digit_number : f 2007 = 2003 := by sorry

end digit_2007_in_2003_digit_number_l2252_225253


namespace x_greater_than_half_l2252_225249

theorem x_greater_than_half (x : ℝ) (h : (1/2) * x = 1) : 
  (x - 1/2) / (1/2) * 100 = 300 := by
  sorry

end x_greater_than_half_l2252_225249


namespace trapezoid_division_theorem_l2252_225211

/-- A trapezoid with sides a, b, c, d where b is parallel to d -/
structure Trapezoid (α : Type*) [LinearOrderedField α] :=
  (a b c d : α)
  (parallel : b ≠ d)

/-- The ratio in which a line parallel to the bases divides a trapezoid -/
def divisionRatio {α : Type*} [LinearOrderedField α] (t : Trapezoid α) (z : α) : α :=
  (t.d + t.b) / 2 + (t.d - t.b)^2 / (2 * (t.a + t.c))

/-- The condition that two trapezoids formed by a parallel line have equal perimeters -/
def equalPerimeters {α : Type*} [LinearOrderedField α] (t : Trapezoid α) (z : α) : Prop :=
  t.a + z + t.c + (t.d - z) = t.b + z + t.a + (t.d - z)

theorem trapezoid_division_theorem {α : Type*} [LinearOrderedField α] (t : Trapezoid α) (z : α) :
  equalPerimeters t z → z = divisionRatio t z :=
sorry

end trapezoid_division_theorem_l2252_225211


namespace initial_dimes_equation_l2252_225281

/-- The number of dimes Sam initially had -/
def initial_dimes : ℕ := sorry

/-- The number of dimes Sam gave away -/
def dimes_given_away : ℕ := 7

/-- The number of dimes Sam has left -/
def dimes_left : ℕ := 2

/-- Theorem: The initial number of dimes is equal to the sum of dimes given away and dimes left -/
theorem initial_dimes_equation : initial_dimes = dimes_given_away + dimes_left := by
  sorry

end initial_dimes_equation_l2252_225281


namespace simplify_power_expression_l2252_225277

theorem simplify_power_expression (y : ℝ) : (3 * y^4)^5 = 243 * y^20 := by
  sorry

end simplify_power_expression_l2252_225277


namespace imaginary_part_of_one_plus_i_to_fifth_l2252_225289

theorem imaginary_part_of_one_plus_i_to_fifth (i : ℂ) : i * i = -1 → Complex.im ((1 + i)^5) = -4 := by
  sorry

end imaginary_part_of_one_plus_i_to_fifth_l2252_225289


namespace sqrt_fraction_simplification_l2252_225290

theorem sqrt_fraction_simplification :
  Real.sqrt (25 / 36 + 16 / 9) = Real.sqrt 89 / 6 :=
by sorry

end sqrt_fraction_simplification_l2252_225290


namespace centroid_coordinates_specific_triangle_centroid_l2252_225269

/-- The centroid of a triangle is located at the arithmetic mean of its vertices. -/
theorem centroid_coordinates (A B C : ℝ × ℝ) :
  let G := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
  A = (-1, 3) → B = (1, 2) → C = (2, -5) → G = (2/3, 0) := by
  sorry

/-- The centroid of the specific triangle ABC is at (2/3, 0). -/
theorem specific_triangle_centroid :
  let A : ℝ × ℝ := (-1, 3)
  let B : ℝ × ℝ := (1, 2)
  let C : ℝ × ℝ := (2, -5)
  let G := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
  G = (2/3, 0) := by
  sorry

end centroid_coordinates_specific_triangle_centroid_l2252_225269


namespace stating_rhombus_solutions_count_l2252_225244

/-- Represents the number of solutions for inscribing a rhombus in a square and circumscribing it around a circle -/
inductive NumSolutions
  | two
  | one
  | zero

/-- 
  Given a square and a circle with the same center, determines the number of possible rhombuses 
  that can be inscribed in the square and circumscribed around the circle.
-/
def numRhombusSolutions (squareSide : ℝ) (circleRadius : ℝ) : NumSolutions :=
  sorry

/-- 
  Theorem stating that the number of rhombus solutions is either 2, 1, or 0
-/
theorem rhombus_solutions_count (squareSide : ℝ) (circleRadius : ℝ) :
  ∃ (n : NumSolutions), numRhombusSolutions squareSide circleRadius = n :=
  sorry

end stating_rhombus_solutions_count_l2252_225244


namespace necessary_but_not_sufficient_condition_l2252_225222

theorem necessary_but_not_sufficient_condition (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 3, x^2 - a*x + 3 < 0) → (a > 3 ∧ ∃ b > 3, ¬(∀ x ∈ Set.Icc 1 3, x^2 - b*x + 3 < 0)) :=
by sorry

end necessary_but_not_sufficient_condition_l2252_225222


namespace seven_digit_subtraction_l2252_225209

def is_seven_digit (n : ℕ) : Prop := 1000000 ≤ n ∧ n ≤ 9999999

def digit_sum_except_second (n : ℕ) : ℕ :=
  let digits := (Nat.digits 10 n).reverse
  List.sum (digits.take 1 ++ digits.drop 2)

theorem seven_digit_subtraction (n : ℕ) :
  is_seven_digit n →
  ∃ k, n - k = 9875352 →
  n - digit_sum_except_second n = 9875357 :=
sorry

end seven_digit_subtraction_l2252_225209


namespace event_probability_l2252_225254

theorem event_probability (p : ℝ) :
  (0 ≤ p) ∧ (p ≤ 1) →
  (1 - (1 - p)^4 = 65/81) →
  p = 1/3 := by
  sorry

end event_probability_l2252_225254


namespace white_marbles_count_l2252_225242

-- Define the parameters
def total_marbles : ℕ := 60
def blue_marbles : ℕ := 5
def red_marbles : ℕ := 9
def prob_red_or_white : ℚ := 55 / 60

-- Theorem statement
theorem white_marbles_count :
  ∃ (white_marbles : ℕ),
    white_marbles = total_marbles - blue_marbles - red_marbles ∧
    (red_marbles + white_marbles : ℚ) / total_marbles = prob_red_or_white ∧
    white_marbles = 46 := by
  sorry

end white_marbles_count_l2252_225242


namespace fraction_problem_l2252_225203

theorem fraction_problem : ∃ f : ℚ, f * 1 = (144 : ℚ) / 216 ∧ f = 2 / 3 := by
  sorry

end fraction_problem_l2252_225203


namespace probability_x_plus_y_less_than_4_l2252_225293

/-- A square in the 2D plane -/
structure Square where
  bottomLeft : ℝ × ℝ
  sideLength : ℝ

/-- The probability that a randomly chosen point in the square satisfies a condition -/
def probabilityInSquare (s : Square) (condition : ℝ × ℝ → Prop) : ℝ :=
  sorry

theorem probability_x_plus_y_less_than_4 :
  let s : Square := { bottomLeft := (0, 0), sideLength := 3 }
  probabilityInSquare s (fun (x, y) ↦ x + y < 4) = 7 / 9 := by
  sorry

end probability_x_plus_y_less_than_4_l2252_225293


namespace equation_solutions_l2252_225250

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 6*x + 1 = 0 ↔ x = 3 + 2*Real.sqrt 2 ∨ x = 3 - 2*Real.sqrt 2) ∧
  (∀ x : ℝ, (2*x - 3)^2 = 5*(2*x - 3) ↔ x = 3/2 ∨ x = 4) := by
  sorry

end equation_solutions_l2252_225250


namespace fraction_equality_l2252_225217

theorem fraction_equality (x y : ℝ) (h : (x - y) / (x + y) = 5) :
  (2 * x + 3 * y) / (3 * x - 2 * y) = 0 :=
by sorry

end fraction_equality_l2252_225217


namespace coin_flip_probability_l2252_225218

/-- The number of coins flipped -/
def n : ℕ := 10

/-- The probability of getting heads on a single coin flip -/
def p : ℚ := 1/2

/-- The probability of getting an equal number of heads and tails -/
def prob_equal : ℚ := (n.choose (n/2)) / 2^n

/-- The probability of getting more heads than tails -/
def prob_more_heads : ℚ := (1 - prob_equal) / 2

theorem coin_flip_probability : prob_more_heads = 193/512 := by
  sorry

end coin_flip_probability_l2252_225218


namespace base_4_9_digit_difference_l2252_225235

theorem base_4_9_digit_difference (n : ℕ) (h : n = 1024) : 
  (Nat.log 4 n + 1) - (Nat.log 9 n + 1) = 1 := by
  sorry

end base_4_9_digit_difference_l2252_225235


namespace four_prime_pairs_sum_50_l2252_225221

/-- A function that returns the number of unordered pairs of prime numbers that sum to a given natural number. -/
def count_prime_pairs (n : ℕ) : ℕ :=
  (Finset.filter (fun p => Nat.Prime p ∧ Nat.Prime (n - p)) (Finset.range (n/2 + 1))).card

/-- Theorem stating that there are exactly 4 unordered pairs of prime numbers that sum to 50. -/
theorem four_prime_pairs_sum_50 : count_prime_pairs 50 = 4 := by
  sorry

end four_prime_pairs_sum_50_l2252_225221


namespace arithmetic_mean_difference_l2252_225292

/-- Given that the arithmetic mean of p and q is 10 and the arithmetic mean of q and r is 25,
    prove that r - p = 30 -/
theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 25) : 
  r - p = 30 := by
  sorry

end arithmetic_mean_difference_l2252_225292


namespace parallel_line_through_point_l2252_225225

/-- Given a line L1 with equation 6x - 3y = 9 and a point P (1, -2),
    prove that the line L2 with equation y = 2x - 4 is parallel to L1 and passes through P. -/
theorem parallel_line_through_point (x y : ℝ) :
  (6 * x - 3 * y = 9) →  -- Equation of L1
  (y = 2 * x - 4) →      -- Equation of L2
  (2 = (6 : ℝ) / 3) →    -- Slopes are equal (parallel condition)
  (2 * 1 - 4 = -2) →     -- L2 passes through (1, -2)
  ∃ (m b : ℝ), y = m * x + b ∧ m = 2 ∧ b = -4 := by
sorry


end parallel_line_through_point_l2252_225225


namespace voldemort_cake_calories_l2252_225280

/-- Calculates the calories of a cake given daily calorie limit, consumed calories, and remaining allowed calories. -/
def cake_calories (daily_limit : ℕ) (breakfast : ℕ) (lunch : ℕ) (chips : ℕ) (coke : ℕ) (remaining : ℕ) : ℕ :=
  daily_limit - (breakfast + lunch + chips + coke) - remaining

/-- Proves that the cake has 110 calories given Voldemort's calorie intake information. -/
theorem voldemort_cake_calories :
  cake_calories 2500 560 780 310 215 525 = 110 := by
  sorry

#eval cake_calories 2500 560 780 310 215 525

end voldemort_cake_calories_l2252_225280


namespace product_of_solutions_eq_neg_nine_l2252_225270

theorem product_of_solutions_eq_neg_nine :
  ∃ (z₁ z₂ : ℂ), z₁ ≠ z₂ ∧ 
  (Complex.abs z₁ = 3 * (Complex.abs z₁ - 2)) ∧
  (Complex.abs z₂ = 3 * (Complex.abs z₂ - 2)) ∧
  (z₁ * z₂ = -9) := by
sorry

end product_of_solutions_eq_neg_nine_l2252_225270


namespace inner_triangle_perimeter_is_180_l2252_225233

/-- Triangle DEF with given side lengths -/
structure Triangle :=
  (DE : ℝ)
  (EF : ℝ)
  (FD : ℝ)

/-- Parallel lines intersecting the triangle -/
structure ParallelLines :=
  (m_D : ℝ)
  (m_E : ℝ)
  (m_F : ℝ)

/-- The perimeter of the inner triangle formed by parallel lines -/
def inner_triangle_perimeter (t : Triangle) (p : ParallelLines) : ℝ :=
  p.m_D + p.m_E + p.m_F

/-- Theorem stating the perimeter of the inner triangle -/
theorem inner_triangle_perimeter_is_180 
  (t : Triangle) 
  (p : ParallelLines) 
  (h1 : t.DE = 140) 
  (h2 : t.EF = 260) 
  (h3 : t.FD = 200) 
  (h4 : p.m_D = 65) 
  (h5 : p.m_E = 85) 
  (h6 : p.m_F = 30) : 
  inner_triangle_perimeter t p = 180 := by
  sorry

#check inner_triangle_perimeter_is_180

end inner_triangle_perimeter_is_180_l2252_225233


namespace point_division_theorem_l2252_225224

/-- Given a line segment AB and a point P on it such that AP:PB = 3:5,
    prove that P = (5/8)*A + (3/8)*B --/
theorem point_division_theorem (A B P : ℝ × ℝ) : 
  (∃ t : ℝ, P = A + t • (B - A)) → -- P is on line segment AB
  (dist A P : ℝ) / (dist P B) = 3 / 5 → -- AP:PB = 3:5
  P = (5/8 : ℝ) • A + (3/8 : ℝ) • B := by sorry

end point_division_theorem_l2252_225224


namespace basketball_free_throws_l2252_225232

theorem basketball_free_throws :
  ∀ (a b x : ℚ),
  3 * b = 2 * a →
  x = 2 * a - 2 →
  2 * a + 3 * b + x = 78 →
  x = 74 / 3 := by
sorry

end basketball_free_throws_l2252_225232


namespace book_sale_revenue_l2252_225243

theorem book_sale_revenue (total_books : ℕ) (sold_fraction : ℚ) (price_per_book : ℚ) (unsold_books : ℕ) : 
  sold_fraction = 2 / 3 →
  price_per_book = 2 →
  unsold_books = 36 →
  unsold_books = (1 - sold_fraction) * total_books →
  sold_fraction * total_books * price_per_book = 144 := by
  sorry

#check book_sale_revenue

end book_sale_revenue_l2252_225243


namespace zoo_recovery_time_l2252_225272

/-- The time spent recovering escaped animals from a zoo -/
theorem zoo_recovery_time 
  (lions : ℕ) 
  (rhinos : ℕ) 
  (recovery_time_per_animal : ℕ) 
  (h1 : lions = 3) 
  (h2 : rhinos = 2) 
  (h3 : recovery_time_per_animal = 2) : 
  (lions + rhinos) * recovery_time_per_animal = 10 := by
sorry

end zoo_recovery_time_l2252_225272


namespace days_worked_together_l2252_225239

-- Define the total work as a positive real number
variable (W : ℝ) (hW : W > 0)

-- Define the time taken by a and b together to finish the work
def time_together : ℝ := 40

-- Define the time taken by a alone to finish the work
def time_a_alone : ℝ := 12

-- Define the additional time a worked after b left
def additional_time_a : ℝ := 9

-- Define the function to calculate the work done in a given time at a given rate
def work_done (time : ℝ) (rate : ℝ) : ℝ := time * rate

-- Define the theorem to prove
theorem days_worked_together (W : ℝ) (hW : W > 0) : 
  ∃ x : ℝ, x > 0 ∧ 
    work_done x (W / time_together) + 
    work_done additional_time_a (W / time_a_alone) = W ∧
    x = 10 := by
  sorry

end days_worked_together_l2252_225239


namespace clara_age_problem_l2252_225236

theorem clara_age_problem : ∃! x : ℕ+, 
  (∃ n : ℕ+, (x - 2 : ℤ) = n^2) ∧ 
  (∃ m : ℕ+, (x + 3 : ℤ) = m^3) ∧ 
  x = 123 := by
  sorry

end clara_age_problem_l2252_225236


namespace specific_marathon_distance_l2252_225257

/-- A circular marathon with four checkpoints -/
structure CircularMarathon where
  /-- Number of checkpoints -/
  num_checkpoints : Nat
  /-- Distance from start to first checkpoint -/
  start_to_first : ℝ
  /-- Distance from last checkpoint to finish -/
  last_to_finish : ℝ
  /-- Distance between consecutive checkpoints -/
  checkpoint_distance : ℝ

/-- The total distance of the marathon -/
def marathon_distance (m : CircularMarathon) : ℝ :=
  m.start_to_first + 
  m.last_to_finish + 
  (m.num_checkpoints - 1 : ℝ) * m.checkpoint_distance

/-- Theorem stating the total distance of the specific marathon -/
theorem specific_marathon_distance : 
  ∀ (m : CircularMarathon), 
    m.num_checkpoints = 4 ∧ 
    m.start_to_first = 1 ∧ 
    m.last_to_finish = 1 ∧ 
    m.checkpoint_distance = 6 → 
    marathon_distance m = 20 := by
  sorry

end specific_marathon_distance_l2252_225257


namespace power_function_through_point_l2252_225248

/-- If the point (√3/3, √3/9) lies on the graph of a power function f(x), then f(x) = x³ -/
theorem power_function_through_point (f : ℝ → ℝ) :
  (∃ α : ℝ, ∀ x : ℝ, f x = x^α) →
  f (Real.sqrt 3 / 3) = Real.sqrt 3 / 9 →
  ∀ x : ℝ, f x = x^3 := by sorry

end power_function_through_point_l2252_225248


namespace complex_pure_imaginary_a_equals_one_l2252_225282

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_pure_imaginary_a_equals_one (a : ℝ) :
  is_pure_imaginary ((1 + a * Complex.I) / (1 - Complex.I)) → a = 1 := by
  sorry

end complex_pure_imaginary_a_equals_one_l2252_225282


namespace crayons_per_row_l2252_225260

/-- Given that Faye has 16 rows of crayons and pencils, with a total of 96 crayons,
    prove that there are 6 crayons in each row. -/
theorem crayons_per_row (total_rows : ℕ) (total_crayons : ℕ) (h1 : total_rows = 16) (h2 : total_crayons = 96) :
  total_crayons / total_rows = 6 := by
  sorry

end crayons_per_row_l2252_225260


namespace incorrect_calculation_ratio_l2252_225268

theorem incorrect_calculation_ratio (N : ℝ) (h : N ≠ 0) : 
  (N * 16) / ((N / 16) / 8) = 2048 := by
sorry

end incorrect_calculation_ratio_l2252_225268


namespace square_sum_given_product_and_sum_l2252_225216

theorem square_sum_given_product_and_sum (p q : ℝ) 
  (h1 : p * q = 20) 
  (h2 : p + q = 10) : 
  p^2 + q^2 = 60 := by
sorry

end square_sum_given_product_and_sum_l2252_225216


namespace swimmer_laps_theorem_l2252_225278

/-- Represents the number of laps swum by a person in a given number of weeks -/
def laps_swum (laps_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ) : ℕ :=
  laps_per_day * days_per_week * weeks

theorem swimmer_laps_theorem (x : ℕ) :
  laps_swum 12 5 x = 60 * x :=
by
  sorry

#check swimmer_laps_theorem

end swimmer_laps_theorem_l2252_225278


namespace loss_percentage_l2252_225213

/-- Calculate the percentage of loss given the cost price and selling price -/
theorem loss_percentage (cost_price selling_price : ℝ) : 
  cost_price = 750 → selling_price = 600 → 
  (cost_price - selling_price) / cost_price * 100 = 20 := by
sorry

end loss_percentage_l2252_225213


namespace christine_stickers_l2252_225202

/-- The number of stickers Christine currently has -/
def current_stickers : ℕ := 11

/-- The number of stickers required for a prize -/
def required_stickers : ℕ := 30

/-- The number of additional stickers Christine needs -/
def additional_stickers : ℕ := required_stickers - current_stickers

theorem christine_stickers : additional_stickers = 19 := by
  sorry

end christine_stickers_l2252_225202


namespace parabola_intersection_theorem_l2252_225266

/-- Parabola E with parameter p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Line intersecting the parabola -/
structure IntersectingLine where
  k : ℝ

/-- Point on the parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

/-- Theorem about parabola intersection and slope relations -/
theorem parabola_intersection_theorem 
  (E : Parabola) 
  (L : IntersectingLine) 
  (A B : ParabolaPoint) 
  (h_on_parabola : A.x^2 = 2*E.p*A.y ∧ B.x^2 = 2*E.p*B.y)
  (h_on_line : A.y = L.k*A.x + 2 ∧ B.y = L.k*B.x + 2)
  (h_dot_product : A.x*B.x + A.y*B.y = 2) :
  (∃ (k₁ k₂ : ℝ), 
    k₁ = (A.y + 2) / A.x ∧ 
    k₂ = (B.y + 2) / B.x ∧ 
    k₁^2 + k₂^2 - 2*L.k^2 = 16) ∧
  E.p = 1/2 := by sorry

end parabola_intersection_theorem_l2252_225266


namespace problem_solution_l2252_225287

theorem problem_solution (x z : ℚ) : 
  x = 103 → x^3*z - 3*x^2*z + 2*x*z = 208170 → z = 5/265 := by
  sorry

end problem_solution_l2252_225287


namespace consecutive_integers_product_sum_l2252_225273

theorem consecutive_integers_product_sum (x : ℕ) : 
  x > 0 ∧ x * (x + 1) = 272 → x + (x + 1) = 33 := by
  sorry

end consecutive_integers_product_sum_l2252_225273


namespace train_length_l2252_225231

/-- The length of a train given its speed and time to cross an overbridge -/
theorem train_length (speed : ℝ) (time : ℝ) (bridge_length : ℝ) :
  speed = 36 * 1000 / 3600 →
  time = 70 →
  bridge_length = 100 →
  speed * time - bridge_length = 600 := by
  sorry

end train_length_l2252_225231


namespace number_comparison_l2252_225297

theorem number_comparison : 
  ((-3 : ℝ) < -2) ∧ 
  (|(-4 : ℝ)| ≥ -2) ∧ 
  ((0 : ℝ) ≥ -2) ∧ 
  (-(-2 : ℝ) ≥ -2) := by
  sorry

end number_comparison_l2252_225297


namespace seating_arrangement_probability_l2252_225206

/-- The number of delegates --/
def num_delegates : ℕ := 12

/-- The number of countries --/
def num_countries : ℕ := 3

/-- The number of delegates per country --/
def delegates_per_country : ℕ := 4

/-- The probability that each delegate sits next to at least one delegate from another country --/
def seating_probability : ℚ := 21 / 22

/-- Theorem stating the probability of the seating arrangement --/
theorem seating_arrangement_probability :
  let total_arrangements := (num_delegates.factorial) / (delegates_per_country.factorial ^ num_countries)
  let unwanted_arrangements := num_countries * num_delegates * (num_delegates - delegates_per_country).factorial / 
                               (delegates_per_country.factorial ^ (num_countries - 1)) -
                               (num_countries.choose 2) * num_delegates * delegates_per_country +
                               num_delegates * (num_countries - 1)
  (total_arrangements - unwanted_arrangements) / total_arrangements = seating_probability :=
sorry

end seating_arrangement_probability_l2252_225206


namespace plane_equation_proof_l2252_225228

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane equation in the form Ax + By + Cz + D = 0 -/
structure PlaneEquation where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane -/
def pointLiesOnPlane (p : Point3D) (eq : PlaneEquation) : Prop :=
  eq.A * p.x + eq.B * p.y + eq.C * p.z + eq.D = 0

/-- Check if two planes are perpendicular -/
def planesArePerpendicular (eq1 eq2 : PlaneEquation) : Prop :=
  eq1.A * eq2.A + eq1.B * eq2.B + eq1.C * eq2.C = 0

/-- The greatest common divisor of the absolute values of four integers is 1 -/
def gcdOfFourIntsIsOne (a b c d : ℤ) : Prop :=
  Nat.gcd (Nat.gcd (Int.natAbs a) (Int.natAbs b)) (Nat.gcd (Int.natAbs c) (Int.natAbs d)) = 1

theorem plane_equation_proof (p1 p2 : Point3D) (givenPlane : PlaneEquation) 
    (h1 : p1 = ⟨2, -3, 4⟩) 
    (h2 : p2 = ⟨-1, 3, -2⟩)
    (h3 : givenPlane = ⟨3, -2, 1, -7⟩) :
  ∃ (resultPlane : PlaneEquation), 
    resultPlane.A > 0 ∧ 
    gcdOfFourIntsIsOne resultPlane.A resultPlane.B resultPlane.C resultPlane.D ∧
    pointLiesOnPlane p1 resultPlane ∧
    pointLiesOnPlane p2 resultPlane ∧
    planesArePerpendicular resultPlane givenPlane ∧
    resultPlane = ⟨2, 5, -4, 27⟩ := by
  sorry

end plane_equation_proof_l2252_225228


namespace product_equality_l2252_225215

theorem product_equality (h : 213 * 16 = 3408) : 16 * 21.3 = 340.8 := by
  sorry

end product_equality_l2252_225215


namespace dartboard_central_angle_l2252_225212

/-- The central angle of a region on a circular dartboard, given its probability -/
theorem dartboard_central_angle (probability : ℝ) (h : probability = 1 / 8) :
  probability * 360 = 45 := by
  sorry

end dartboard_central_angle_l2252_225212


namespace smallest_positive_angle_same_terminal_side_l2252_225245

/-- Given an angle α = -3000°, this theorem states that the smallest positive angle
    with the same terminal side as α is 240°. -/
theorem smallest_positive_angle_same_terminal_side :
  let α : ℝ := -3000
  ∃ (k : ℤ), α + k * 360 = 240 ∧
    ∀ (m : ℤ), α + m * 360 > 0 → α + m * 360 ≥ 240 :=
by sorry

end smallest_positive_angle_same_terminal_side_l2252_225245


namespace unique_solution_equation_l2252_225204

theorem unique_solution_equation (b : ℝ) : 
  (b + ⌈b⌉ = 21.6) ∧ (b - ⌊b⌋ = 0.6) → b = 10.6 :=
by sorry

end unique_solution_equation_l2252_225204


namespace worker_y_fraction_l2252_225256

theorem worker_y_fraction (P : ℝ) (Px Py : ℝ) (h1 : P > 0) (h2 : Px ≥ 0) (h3 : Py ≥ 0) :
  Px + Py = P →
  0.005 * Px + 0.008 * Py = 0.007 * P →
  Py / P = 2 / 3 := by
sorry

end worker_y_fraction_l2252_225256


namespace alternatingArithmeticSequenceSum_l2252_225291

def alternatingArithmeticSequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : List ℤ :=
  List.range n |> List.map (λ i => a₁ + i * d * (if i % 2 = 0 then 1 else -1))

theorem alternatingArithmeticSequenceSum :
  let seq := alternatingArithmeticSequence 2 4 26
  seq.sum = -52 := by
  sorry

end alternatingArithmeticSequenceSum_l2252_225291


namespace geometric_series_sum_l2252_225283

/-- Given an infinite geometric series with a specific pattern, prove the value of k that makes the series sum to 10 -/
theorem geometric_series_sum (k : ℝ) : 
  (∑' n : ℕ, (4 + n * k) / 5^n) = 10 → k = 19.2 := by
  sorry

end geometric_series_sum_l2252_225283


namespace hyperbola_equation_l2252_225214

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), y = 2*x + 10 ∧ x^2 + y^2 = (a^2 + b^2)) → 
  (b / a = 2) → 
  (a^2 = 5 ∧ b^2 = 20) := by sorry

end hyperbola_equation_l2252_225214


namespace megan_deleted_files_l2252_225201

/-- Calculates the number of deleted files given the initial number of files,
    number of folders after organizing, and number of files per folder. -/
def deleted_files (initial_files : ℕ) (num_folders : ℕ) (files_per_folder : ℕ) : ℕ :=
  initial_files - (num_folders * files_per_folder)

/-- Proves that Megan deleted 21 files given the problem conditions. -/
theorem megan_deleted_files :
  deleted_files 93 9 8 = 21 := by
  sorry

end megan_deleted_files_l2252_225201


namespace point_coordinates_proof_l2252_225208

/-- Given points A and B, and the relation between vectors AP and AB, 
    prove that P has specific coordinates. -/
theorem point_coordinates_proof (A B P : ℝ × ℝ) : 
  A = (2, 3) → 
  B = (4, -3) → 
  P - A = 3 • (B - A) → 
  P = (8, -15) := by
  sorry

end point_coordinates_proof_l2252_225208


namespace perimeter_bisector_min_value_l2252_225288

/-- A line that always bisects the perimeter of a circle -/
structure PerimeterBisector where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : ∀ (x y : ℝ), a * x + b * y + 1 = 0 → x^2 + y^2 + 8*x + 2*y + 1 = 0 →
       ∃ (c : ℝ), c > 0 ∧ (x + 4)^2 + (y + 1)^2 = c^2 ∧ a * (-4) + b * (-1) + 1 = 0

/-- The minimum value of 1/a + 4/b for a perimeter bisector is 16 -/
theorem perimeter_bisector_min_value (pb : PerimeterBisector) :
  (1 / pb.a + 4 / pb.b) ≥ 16 := by
  sorry

end perimeter_bisector_min_value_l2252_225288


namespace trig_simplification_l2252_225205

open Real

theorem trig_simplification (α : ℝ) :
  (tan (π/4 - α) / (1 - tan (π/4 - α)^2)) * ((sin α * cos α) / (cos α^2 - sin α^2)) = 1/4 := by
  sorry

end trig_simplification_l2252_225205


namespace limit_s_at_zero_is_infinity_l2252_225219

/-- The x coordinate of the left endpoint of the intersection of y = x^3 and y = m -/
noncomputable def P (m : ℝ) : ℝ := -Real.rpow m (1/3)

/-- The function s defined as [P(-m) - P(m)]/m -/
noncomputable def s (m : ℝ) : ℝ := (P (-m) - P m) / m

theorem limit_s_at_zero_is_infinity :
  ∀ ε > 0, ∃ δ > 0, ∀ m : ℝ, 0 < |m| ∧ |m| < δ ∧ -2 < m ∧ m < 2 → |s m| > ε :=
sorry

end limit_s_at_zero_is_infinity_l2252_225219


namespace unique_integer_solution_l2252_225275

theorem unique_integer_solution : ∃! x : ℤ, 
  (((2 * x > 70) ∧ (x < 100)) ∨ 
   ((2 * x > 70) ∧ (4 * x > 25)) ∨ 
   ((2 * x > 70) ∧ (x > 5)) ∨ 
   ((x < 100) ∧ (4 * x > 25)) ∨ 
   ((x < 100) ∧ (x > 5)) ∨ 
   ((4 * x > 25) ∧ (x > 5))) ∧
  (((2 * x ≤ 70) ∧ (x ≥ 100)) ∨ 
   ((2 * x ≤ 70) ∧ (4 * x ≤ 25)) ∨ 
   ((2 * x ≤ 70) ∧ (x ≤ 5)) ∨ 
   ((x ≥ 100) ∧ (4 * x ≤ 25)) ∨ 
   ((x ≥ 100) ∧ (x ≤ 5)) ∨ 
   ((4 * x ≤ 25) ∧ (x ≤ 5))) ∧
  x = 6 := by
sorry

end unique_integer_solution_l2252_225275


namespace sector_area_l2252_225247

/-- The area of a circular sector with central angle π/3 and radius 3 is 3π/2 -/
theorem sector_area (θ : Real) (r : Real) (h1 : θ = π / 3) (h2 : r = 3) :
  (1 / 2) * θ * r^2 = 3 * π / 2 := by
  sorry

end sector_area_l2252_225247


namespace min_value_quadratic_l2252_225262

theorem min_value_quadratic (x y : ℝ) : x^2 + x*y + y^2 + 7 ≥ 7 := by
  sorry

end min_value_quadratic_l2252_225262


namespace cube_of_complex_number_l2252_225296

/-- Given that z = sin(π/3) + i*cos(π/3), prove that z^3 = i -/
theorem cube_of_complex_number (z : ℂ) (h : z = Complex.exp (Complex.I * (π / 3))) :
  z^3 = Complex.I := by sorry

end cube_of_complex_number_l2252_225296


namespace rajesh_work_time_l2252_225220

/-- The problem of determining Rajesh's work time -/
theorem rajesh_work_time (rahul_rate : ℝ) (rajesh_rate : ℝ → ℝ) (combined_rate : ℝ → ℝ) 
  (total_payment : ℝ) (rahul_share : ℝ) (R : ℝ) :
  rahul_rate = 1/3 →
  (∀ x, rajesh_rate x = 1/x) →
  (∀ x, combined_rate x = (x + 3) / (3*x)) →
  total_payment = 150 →
  rahul_share = 60 →
  R = 4.5 := by
  sorry

end rajesh_work_time_l2252_225220


namespace fitness_center_membership_ratio_l2252_225227

theorem fitness_center_membership_ratio :
  ∀ (f m c : ℕ), 
  (f > 0) → (m > 0) → (c > 0) →
  (35 * f + 30 * m + 10 * c : ℝ) / (f + m + c : ℝ) = 25 →
  ∃ (k : ℕ), k > 0 ∧ f = 3 * k ∧ m = 6 * k ∧ c = 2 * k :=
by sorry

end fitness_center_membership_ratio_l2252_225227


namespace factor_polynomial_l2252_225259

theorem factor_polynomial (x : ℝ) : 98 * x^7 - 266 * x^13 = 14 * x^7 * (7 - 19 * x^6) := by
  sorry

end factor_polynomial_l2252_225259


namespace min_seedlings_to_plant_l2252_225255

theorem min_seedlings_to_plant (min_survival : ℝ) (max_survival : ℝ) (target : ℕ) : 
  min_survival = 0.75 →
  max_survival = 0.8 →
  target = 1200 →
  ∃ n : ℕ, n ≥ 1500 ∧ ∀ m : ℕ, m < n → (m : ℝ) * max_survival < target := by
  sorry

end min_seedlings_to_plant_l2252_225255


namespace circle_equation_implies_expression_value_l2252_225210

theorem circle_equation_implies_expression_value (x y : ℝ) : 
  x^2 + y^2 = 1 → 
  Real.sqrt (x^2 - 4*x + 4) + Real.sqrt (x*y - 3*x + y - 3) = 3 := by
  sorry

end circle_equation_implies_expression_value_l2252_225210
