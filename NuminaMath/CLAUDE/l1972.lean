import Mathlib

namespace NUMINAMATH_CALUDE_expenditure_ratio_proof_l1972_197291

/-- Given two persons P1 and P2 with incomes and expenditures, prove their expenditure ratio --/
theorem expenditure_ratio_proof 
  (income_ratio : ℚ) -- Ratio of incomes P1:P2
  (savings : ℕ) -- Amount saved by each person
  (income_p1 : ℕ) -- Income of P1
  (h1 : income_ratio = 5 / 4) -- Income ratio condition
  (h2 : savings = 1600) -- Savings condition
  (h3 : income_p1 = 4000) -- P1's income condition
  : (income_p1 - savings) / ((income_p1 * 4 / 5) - savings) = 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_expenditure_ratio_proof_l1972_197291


namespace NUMINAMATH_CALUDE_diet_soda_sales_l1972_197288

theorem diet_soda_sales (total_sodas : ℕ) (regular_ratio diet_ratio : ℕ) (diet_sodas : ℕ) : 
  total_sodas = 64 →
  regular_ratio = 9 →
  diet_ratio = 7 →
  regular_ratio * diet_sodas = diet_ratio * (total_sodas - diet_sodas) →
  diet_sodas = 28 := by
sorry

end NUMINAMATH_CALUDE_diet_soda_sales_l1972_197288


namespace NUMINAMATH_CALUDE_neil_final_three_prob_l1972_197260

/-- A 3-sided die with numbers 1, 2, and 3 -/
inductive Die : Type
| one : Die
| two : Die
| three : Die

/-- The probability of rolling each number on the die -/
def prob_roll (d : Die) : ℚ := 1/3

/-- The event of Neil's final number being 3 -/
def neil_final_three : Set (Die × Die) := {(j, n) | n = Die.three}

/-- The probability space of all possible outcomes (Jerry's roll, Neil's final roll) -/
def prob_space : Set (Die × Die) := Set.univ

/-- The theorem stating the probability of Neil's final number being 3 -/
theorem neil_final_three_prob :
  ∃ (P : Set (Die × Die) → ℚ),
    P prob_space = 1 ∧
    P neil_final_three = 11/18 :=
sorry

end NUMINAMATH_CALUDE_neil_final_three_prob_l1972_197260


namespace NUMINAMATH_CALUDE_bottle_weight_difference_l1972_197247

/-- The weight difference between a glass bottle and a plastic bottle -/
def weight_difference : ℝ := by sorry

theorem bottle_weight_difference :
  let glass_bottle_weight : ℝ := 600 / 3
  let plastic_bottle_weight : ℝ := (1050 - 4 * glass_bottle_weight) / 5
  weight_difference = glass_bottle_weight - plastic_bottle_weight :=
by sorry

end NUMINAMATH_CALUDE_bottle_weight_difference_l1972_197247


namespace NUMINAMATH_CALUDE_prob_monochromatic_triangle_l1972_197297

/-- A complete graph K6 where each edge is colored red or blue -/
def ColoredK6 := Fin 15 → Bool

/-- The probability of an edge being red (or blue) -/
def p : ℚ := 1/2

/-- The set of all possible colorings of K6 -/
def allColorings : Set ColoredK6 := Set.univ

/-- A triangle in K6 -/
structure Triangle :=
  (a b c : Fin 6)
  (ha : a < b)
  (hb : b < c)

/-- The set of all triangles in K6 -/
def allTriangles : Set Triangle := sorry

/-- A coloring has a monochromatic triangle -/
def hasMonochromaticTriangle (coloring : ColoredK6) : Prop := sorry

/-- The probability of having at least one monochromatic triangle -/
noncomputable def probMonochromaticTriangle : ℚ := sorry

theorem prob_monochromatic_triangle :
  probMonochromaticTriangle = 1048575/1048576 := by sorry

end NUMINAMATH_CALUDE_prob_monochromatic_triangle_l1972_197297


namespace NUMINAMATH_CALUDE_april_savings_l1972_197261

def savings_pattern (month : Nat) : Nat :=
  2^month

theorem april_savings : savings_pattern 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_april_savings_l1972_197261


namespace NUMINAMATH_CALUDE_number_divided_and_subtracted_l1972_197258

theorem number_divided_and_subtracted (x : ℝ) : x = 4.5 → x / 3 = x - 3 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_and_subtracted_l1972_197258


namespace NUMINAMATH_CALUDE_solution_set_when_m_equals_3_range_of_m_for_inequality_l1972_197242

-- Define the function f
def f (x m : ℝ) : ℝ := |x + 6| - |m - x|

-- Theorem for part I
theorem solution_set_when_m_equals_3 :
  {x : ℝ | f x 3 ≥ 5} = {x : ℝ | x ≥ 1} := by sorry

-- Theorem for part II
theorem range_of_m_for_inequality :
  {m : ℝ | ∀ x, f x m ≤ 7} = {m : ℝ | -13 ≤ m ∧ m ≤ 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_equals_3_range_of_m_for_inequality_l1972_197242


namespace NUMINAMATH_CALUDE_expression_bounds_l1972_197262

theorem expression_bounds (a b c d e : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) (he : 0 ≤ e ∧ e ≤ 1) :
  4 * Real.sqrt (2/3) ≤ 
    Real.sqrt (a^2 + (1-b)^2 + e^2) + Real.sqrt (b^2 + (1-c)^2 + e^2) + 
    Real.sqrt (c^2 + (1-d)^2 + e^2) + Real.sqrt (d^2 + (1-a)^2 + e^2) ∧
  Real.sqrt (a^2 + (1-b)^2 + e^2) + Real.sqrt (b^2 + (1-c)^2 + e^2) + 
  Real.sqrt (c^2 + (1-d)^2 + e^2) + Real.sqrt (d^2 + (1-a)^2 + e^2) ≤ 8 ∧
  ∃ (a b c d e : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 
    0 ≤ d ∧ d ≤ 1 ∧ 0 ≤ e ∧ e ≤ 1 ∧
    Real.sqrt (a^2 + (1-b)^2 + e^2) + Real.sqrt (b^2 + (1-c)^2 + e^2) + 
    Real.sqrt (c^2 + (1-d)^2 + e^2) + Real.sqrt (d^2 + (1-a)^2 + e^2) = 4 * Real.sqrt (2/3) ∧
  ∃ (a b c d e : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 
    0 ≤ d ∧ d ≤ 1 ∧ 0 ≤ e ∧ e ≤ 1 ∧
    Real.sqrt (a^2 + (1-b)^2 + e^2) + Real.sqrt (b^2 + (1-c)^2 + e^2) + 
    Real.sqrt (c^2 + (1-d)^2 + e^2) + Real.sqrt (d^2 + (1-a)^2 + e^2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_l1972_197262


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1972_197234

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 486 → volume = (surface_area / 6) ^ (3/2) → volume = 729 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1972_197234


namespace NUMINAMATH_CALUDE_min_value_2a_plus_b_l1972_197206

theorem min_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : 1 / a + 2 / b = 1) : 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 1 / a' + 2 / b' = 1 → 2 * a + b ≤ 2 * a' + b') ∧ 
  (∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 1 / a₀ + 2 / b₀ = 1 ∧ 2 * a₀ + b₀ = 8) := by
  sorry

end NUMINAMATH_CALUDE_min_value_2a_plus_b_l1972_197206


namespace NUMINAMATH_CALUDE_largest_digit_change_l1972_197294

def incorrect_sum : ℕ := 2456
def num1 : ℕ := 641
def num2 : ℕ := 852
def num3 : ℕ := 973

theorem largest_digit_change :
  ∃ (d : ℕ), d ≤ 9 ∧
  (num1 + num2 + (num3 - 10) = incorrect_sum) ∧
  (∀ (d' : ℕ), d' ≤ 9 → 
    (num1 - 10 * d' + num2 + num3 = incorrect_sum ∨
     num1 + (num2 - 10 * d') + num3 = incorrect_sum) → 
    d' ≤ d) ∧
  d = 7 :=
sorry

end NUMINAMATH_CALUDE_largest_digit_change_l1972_197294


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l1972_197252

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := y^2 / 4 - x^2 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop := y = 2*x ∨ y = -2*x

-- Theorem statement
theorem hyperbola_asymptote :
  ∀ x y : ℝ, hyperbola_equation x y → (∃ X Y : ℝ, X ≠ 0 ∧ asymptote_equation X Y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l1972_197252


namespace NUMINAMATH_CALUDE_equation_solution_l1972_197292

def f (x : ℝ) (b : ℝ) : ℝ := 2 * x - b

theorem equation_solution :
  let b : ℝ := 3
  let x : ℝ := 5
  2 * (f x b) - 11 = f (x - 2) b :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1972_197292


namespace NUMINAMATH_CALUDE_simplify_expression_l1972_197231

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^4 + b^4 = a + b) :
  a / b + b / a - 1 / (a * b^2) = -(a + b) / (a * b^2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1972_197231


namespace NUMINAMATH_CALUDE_find_A_in_terms_of_B_l1972_197298

/-- Given functions f and g, prove the value of A in terms of B -/
theorem find_A_in_terms_of_B (B : ℝ) (hB : B ≠ 0) :
  let f := fun x => A * x - 3 * B^2 + B * x^2
  let g := fun x => B * x^2
  let A := (3 - 16 * B^2) / 4
  f (g 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_find_A_in_terms_of_B_l1972_197298


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l1972_197241

theorem profit_percentage_calculation 
  (cost original_profit selling_price : ℝ)
  (h1 : selling_price = cost + original_profit)
  (h2 : selling_price = 1.12 * cost + 0.53333333333333333 * selling_price) :
  original_profit / cost = 1.4 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l1972_197241


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1972_197263

theorem complex_equation_solution (c d : ℂ) (x : ℝ) :
  Complex.abs c = 3 →
  Complex.abs d = 5 →
  c * d = x - 3 * Complex.I →
  x > 0 →
  x = 6 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1972_197263


namespace NUMINAMATH_CALUDE_trivia_contest_probability_l1972_197287

/-- The number of questions in the trivia contest -/
def num_questions : ℕ := 4

/-- The number of choices for each question -/
def num_choices : ℕ := 4

/-- The minimum number of correct answers needed to win -/
def min_correct : ℕ := 3

/-- The probability of guessing one question correctly -/
def prob_correct : ℚ := 1 / num_choices

/-- The probability of guessing one question incorrectly -/
def prob_incorrect : ℚ := 1 - prob_correct

/-- The probability of winning the trivia contest -/
def prob_winning : ℚ := 13 / 256

theorem trivia_contest_probability :
  (prob_correct ^ num_questions) +
  (num_questions.choose min_correct) * (prob_correct ^ min_correct) * (prob_incorrect ^ (num_questions - min_correct)) =
  prob_winning := by sorry

end NUMINAMATH_CALUDE_trivia_contest_probability_l1972_197287


namespace NUMINAMATH_CALUDE_loss_percentage_tables_l1972_197290

theorem loss_percentage_tables (C S : ℝ) (h : 15 * C = 20 * S) : 
  (C - S) / C * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_loss_percentage_tables_l1972_197290


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1972_197244

theorem average_speed_calculation (total_distance : ℝ) (first_part_distance : ℝ) (first_part_speed : ℝ) 
  (second_part_distance : ℝ) (second_part_speed : ℝ) : 
  total_distance = 850 ∧ 
  first_part_distance = 400 ∧ 
  first_part_speed = 20 ∧
  second_part_distance = 450 ∧
  second_part_speed = 15 →
  (total_distance / ((first_part_distance / first_part_speed) + (second_part_distance / second_part_speed))) = 17 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l1972_197244


namespace NUMINAMATH_CALUDE_tenth_row_third_element_l1972_197253

/-- Represents the exponent of 2 for an element in the triangular array --/
def triangularArrayExponent (row : ℕ) (position : ℕ) : ℕ :=
  (row - 1) * row / 2 + position

/-- The theorem stating that the third element from the left in the 10th row is 2^47 --/
theorem tenth_row_third_element :
  triangularArrayExponent 10 2 = 47 := by
  sorry

end NUMINAMATH_CALUDE_tenth_row_third_element_l1972_197253


namespace NUMINAMATH_CALUDE_perpendicular_lines_n_value_l1972_197226

/-- Two perpendicular lines with a given foot of perpendicular -/
structure PerpendicularLines where
  m : ℝ
  n : ℝ
  p : ℝ
  line1_eq : ∀ x y, m * x + 4 * y - 2 = 0
  line2_eq : ∀ x y, 2 * x - 5 * y + n = 0
  perpendicular : m * 2 + 4 * 5 = 0
  foot_on_line1 : m * 1 + 4 * p - 2 = 0
  foot_on_line2 : 2 * 1 - 5 * p + n = 0

/-- The value of n in the given perpendicular lines setup is -12 -/
theorem perpendicular_lines_n_value (pl : PerpendicularLines) : pl.n = -12 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_n_value_l1972_197226


namespace NUMINAMATH_CALUDE_farmer_budget_distribution_l1972_197295

theorem farmer_budget_distribution (g sh : ℕ) : 
  g > 0 ∧ sh > 0 ∧ 24 * g + 27 * sh = 1200 → g = 5 ∧ sh = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_farmer_budget_distribution_l1972_197295


namespace NUMINAMATH_CALUDE_new_persons_weight_l1972_197296

theorem new_persons_weight (W : ℝ) (X Y : ℝ) :
  (∀ (T : ℝ), T = 8 * W) →
  (∀ (new_total : ℝ), new_total = 8 * W - 140 + X + Y) →
  (∀ (new_avg : ℝ), new_avg = W + 5) →
  (∀ (new_total : ℝ), new_total = 8 * new_avg) →
  X + Y = 180 := by
sorry

end NUMINAMATH_CALUDE_new_persons_weight_l1972_197296


namespace NUMINAMATH_CALUDE_fraction_of_students_with_partners_l1972_197200

theorem fraction_of_students_with_partners :
  ∀ (a b : ℕ), 
    a > 0 → b > 0 →
    (b : ℚ) / 4 = (3 : ℚ) * a / 7 →
    ((b : ℚ) / 4 + (3 : ℚ) * a / 7) / ((b : ℚ) + a) = 6 / 19 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_students_with_partners_l1972_197200


namespace NUMINAMATH_CALUDE_trig_identity_l1972_197228

theorem trig_identity (α : Real) (h : Real.sin (π / 4 + α) = 1 / 2) :
  Real.sin (5 * π / 4 + α) / Real.cos (9 * π / 4 + α) * Real.cos (7 * π / 4 - α) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1972_197228


namespace NUMINAMATH_CALUDE_power_of_seven_fraction_l1972_197285

theorem power_of_seven_fraction (a b : ℕ) : 
  (2^a : ℕ) = Nat.gcd (2^a) 196 → 
  (7^b : ℕ) = Nat.gcd (7^b) 196 → 
  (1/7 : ℚ)^(b - a) = 1 := by sorry

end NUMINAMATH_CALUDE_power_of_seven_fraction_l1972_197285


namespace NUMINAMATH_CALUDE_initial_cats_proof_l1972_197240

def total_cats : ℕ := 7
def female_kittens : ℕ := 3
def male_kittens : ℕ := 2

def initial_cats : ℕ := total_cats - (female_kittens + male_kittens)

theorem initial_cats_proof : initial_cats = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_cats_proof_l1972_197240


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l1972_197274

theorem smallest_lcm_with_gcd_five (m n : ℕ) : 
  10000 ≤ m ∧ m < 100000 ∧ 
  10000 ≤ n ∧ n < 100000 ∧ 
  Nat.gcd m n = 5 →
  20030010 ≤ Nat.lcm m n :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l1972_197274


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1972_197265

theorem necessary_but_not_sufficient (p q : Prop) :
  ((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q)) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1972_197265


namespace NUMINAMATH_CALUDE_total_values_count_l1972_197201

theorem total_values_count (initial_mean correct_mean : ℚ) 
  (incorrect_value correct_value : ℚ) (n : ℕ) : 
  initial_mean = 150 → 
  correct_mean = 151 → 
  incorrect_value = 135 → 
  correct_value = 165 → 
  (n : ℚ) * initial_mean + incorrect_value = (n : ℚ) * correct_mean + correct_value → 
  n = 30 := by
  sorry

#check total_values_count

end NUMINAMATH_CALUDE_total_values_count_l1972_197201


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l1972_197210

theorem smallest_four_digit_multiple_of_18 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n → 1008 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l1972_197210


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1972_197221

/-- Represents the repeating decimal 0.37246̄ as a rational number -/
def repeating_decimal : ℚ := 37245 / 99900

/-- Theorem stating that the repeating decimal 0.37246̄ is equal to 37245/99900 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = 37245 / 99900 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1972_197221


namespace NUMINAMATH_CALUDE_jason_has_21_toys_l1972_197236

/-- The number of toys Rachel has -/
def rachel_toys : ℕ := 1

/-- The number of toys John has -/
def john_toys : ℕ := rachel_toys + 6

/-- The number of toys Jason has -/
def jason_toys : ℕ := 3 * john_toys

/-- Theorem: Jason has 21 toys -/
theorem jason_has_21_toys : jason_toys = 21 := by
  sorry

end NUMINAMATH_CALUDE_jason_has_21_toys_l1972_197236


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l1972_197281

-- Define the triangle DEF
def triangle_DEF : Set (ℝ × ℝ) := sorry

-- Define that the triangle is isosceles
def is_isosceles (t : Set (ℝ × ℝ)) : Prop := sorry

-- Define that the triangle is acute
def is_acute (t : Set (ℝ × ℝ)) : Prop := sorry

-- Define the measure of an angle
def angle_measure (t : Set (ℝ × ℝ)) (v : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem largest_angle_in_special_triangle :
  ∀ (DEF : Set (ℝ × ℝ)),
    is_isosceles DEF →
    is_acute DEF →
    angle_measure DEF (0, 0) = 30 →
    (∃ (v : ℝ × ℝ), v ∈ DEF ∧ angle_measure DEF v = 75 ∧
      ∀ (w : ℝ × ℝ), w ∈ DEF → angle_measure DEF w ≤ 75) :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l1972_197281


namespace NUMINAMATH_CALUDE_reflection_of_point_A_l1972_197284

/-- Reflects a point across the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- The initial point A -/
def point_A : ℝ × ℝ := (1, 2)

theorem reflection_of_point_A :
  reflect_y_axis point_A = (-1, 2) := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_point_A_l1972_197284


namespace NUMINAMATH_CALUDE_solve_equation_l1972_197207

theorem solve_equation : ∃ x : ℝ, (75 / 100 * 4500 = (1 / 4) * x + 144) ∧ x = 12924 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1972_197207


namespace NUMINAMATH_CALUDE_exists_primitive_root_mod_2p_alpha_l1972_197248

/-- Given an odd prime p and a natural number α, there exists a primitive root modulo 2p^α -/
theorem exists_primitive_root_mod_2p_alpha (p : Nat) (α : Nat) 
  (h_prime : Nat.Prime p) (h_odd : Odd p) : 
  ∃ x : Nat, IsPrimitiveRoot x (2 * p^α) := by
  sorry

end NUMINAMATH_CALUDE_exists_primitive_root_mod_2p_alpha_l1972_197248


namespace NUMINAMATH_CALUDE_students_neither_sport_l1972_197286

theorem students_neither_sport (total : ℕ) (football : ℕ) (cricket : ℕ) (both : ℕ)
  (h_total : total = 460)
  (h_football : football = 325)
  (h_cricket : cricket = 175)
  (h_both : both = 90) :
  total - (football + cricket - both) = 50 := by
  sorry

end NUMINAMATH_CALUDE_students_neither_sport_l1972_197286


namespace NUMINAMATH_CALUDE_sum_remainder_is_two_l1972_197204

theorem sum_remainder_is_two (n : ℤ) : (8 - n + (n + 4)) % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_is_two_l1972_197204


namespace NUMINAMATH_CALUDE_sneakers_final_price_l1972_197205

/-- Calculates the final price of sneakers after applying a coupon and membership discount -/
theorem sneakers_final_price
  (original_price : ℝ)
  (coupon_discount : ℝ)
  (membership_discount_rate : ℝ)
  (h1 : original_price = 120)
  (h2 : coupon_discount = 10)
  (h3 : membership_discount_rate = 0.1) :
  let price_after_coupon := original_price - coupon_discount
  let membership_discount := price_after_coupon * membership_discount_rate
  let final_price := price_after_coupon - membership_discount
  final_price = 99 := by
sorry

end NUMINAMATH_CALUDE_sneakers_final_price_l1972_197205


namespace NUMINAMATH_CALUDE_battery_current_l1972_197215

/-- Given a battery with voltage 48V, prove that when the resistance R is 12Ω, 
    the current I is 4A, where I is related to R by the function I = 48/R. -/
theorem battery_current (R : ℝ) (I : ℝ) : 
  R = 12 → I = 48 / R → I = 4 := by sorry

end NUMINAMATH_CALUDE_battery_current_l1972_197215


namespace NUMINAMATH_CALUDE_power_of_power_l1972_197259

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1972_197259


namespace NUMINAMATH_CALUDE_dads_final_strawberry_weight_l1972_197225

/-- Given the initial total weight of strawberries collected by Marco and his dad,
    the additional weight Marco's dad found, and Marco's final weight of strawberries,
    prove that Marco's dad's final weight of strawberries is 46 pounds. -/
theorem dads_final_strawberry_weight
  (initial_total : ℕ)
  (dads_additional : ℕ)
  (marcos_final : ℕ)
  (h1 : initial_total = 22)
  (h2 : dads_additional = 30)
  (h3 : marcos_final = 36) :
  initial_total - (marcos_final - (initial_total - marcos_final)) + dads_additional = 46 :=
by sorry

end NUMINAMATH_CALUDE_dads_final_strawberry_weight_l1972_197225


namespace NUMINAMATH_CALUDE_vector_computation_l1972_197267

theorem vector_computation : 
  4 • !![3, -5] - 3 • !![2, -6] + 2 • !![0, 3] = !![6, 4] := by sorry

end NUMINAMATH_CALUDE_vector_computation_l1972_197267


namespace NUMINAMATH_CALUDE_permutation_sum_of_digits_l1972_197229

def digit_sum : ℕ := 1 + 2 + 3 + 4 + 5 + 6 + 7

def geometric_sum : ℕ := (10^7 - 1) / 9

theorem permutation_sum_of_digits (n : ℕ) (h : n = 7) :
  (n.factorial * digit_sum * geometric_sum : ℕ) = 22399997760 := by
  sorry

end NUMINAMATH_CALUDE_permutation_sum_of_digits_l1972_197229


namespace NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l1972_197255

/-- The path length of the center of a quarter-circle when rolled along a straight line -/
theorem quarter_circle_roll_path_length (r : ℝ) (h : r = 3 / π) :
  let path_length := 3 * (π * r / 2)
  path_length = 9 / 2 := by sorry

end NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l1972_197255


namespace NUMINAMATH_CALUDE_star_not_associative_l1972_197280

-- Define the set T of non-zero real numbers
def T : Set ℝ := {x : ℝ | x ≠ 0}

-- Define the binary operation *
def star (a b : ℝ) : ℝ := 3 * a * b

-- Theorem: * is not associative over T
theorem star_not_associative :
  ∃ (a b c : T), star (star a b) c ≠ star a (star b c) := by
  sorry

end NUMINAMATH_CALUDE_star_not_associative_l1972_197280


namespace NUMINAMATH_CALUDE_evaluate_64_to_two_thirds_l1972_197282

theorem evaluate_64_to_two_thirds : (64 : ℝ) ^ (2/3) = 16 := by sorry

end NUMINAMATH_CALUDE_evaluate_64_to_two_thirds_l1972_197282


namespace NUMINAMATH_CALUDE_ten_faucets_fill_time_l1972_197243

/-- The time (in seconds) it takes for a given number of faucets to fill a tub of a given capacity. -/
def fill_time (num_faucets : ℕ) (capacity : ℝ) : ℝ :=
  sorry

/-- The rate at which one faucet fills a tub (in gallons per minute). -/
def faucet_rate : ℝ :=
  sorry

theorem ten_faucets_fill_time :
  -- Condition 1: Five faucets fill a 150-gallon tub in 10 minutes
  fill_time 5 150 = 10 * 60 →
  -- Condition 2: All faucets dispense water at the same rate (implicit in the definition of faucet_rate)
  -- Prove: Ten faucets will fill a 50-gallon tub in 100 seconds
  fill_time 10 50 = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_ten_faucets_fill_time_l1972_197243


namespace NUMINAMATH_CALUDE_distance_home_to_school_l1972_197216

/-- The distance between home and school satisfies the given conditions -/
theorem distance_home_to_school :
  ∃ (d : ℝ), d > 0 ∧
  ∃ (t : ℝ), t > 0 ∧
  (5 * (t + 7/60) = d) ∧
  (10 * (t - 8/60) = d) ∧
  d = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_distance_home_to_school_l1972_197216


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l1972_197254

theorem max_value_trig_expression (x y z : ℝ) :
  (Real.sin (2 * x) + Real.sin y + Real.sin (3 * z)) *
  (Real.cos (2 * x) + Real.cos y + Real.cos (3 * z)) ≤ 9 / 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l1972_197254


namespace NUMINAMATH_CALUDE_next_occurrence_is_august_first_l1972_197249

def initial_date : Nat × Nat × Nat := (5, 1, 1994)
def initial_time : Nat × Nat := (7, 32)

def digits : List Nat := [0, 5, 1, 1, 9, 9, 4, 0, 7, 3]

def is_valid_date (d : Nat) (m : Nat) (y : Nat) : Bool :=
  d > 0 && d ≤ 31 && m > 0 && m ≤ 12 && y == 1994

def is_valid_time (h : Nat) (m : Nat) : Bool :=
  h ≥ 0 && h < 24 && m ≥ 0 && m < 60

def date_time_to_digits (d : Nat) (m : Nat) (y : Nat) (h : Nat) (min : Nat) : List Nat :=
  let date_digits := (d / 10) :: (d % 10) :: (m / 10) :: (m % 10) :: (y / 1000) :: ((y / 100) % 10) :: ((y / 10) % 10) :: (y % 10) :: []
  let time_digits := (h / 10) :: (h % 10) :: (min / 10) :: (min % 10) :: []
  date_digits ++ time_digits

def is_next_occurrence (d : Nat) (m : Nat) (h : Nat) (min : Nat) : Prop :=
  is_valid_date d m 1994 ∧
  is_valid_time h min ∧
  date_time_to_digits d m 1994 h min == digits ∧
  (d, m) > (5, 1) ∧
  ∀ (d' : Nat) (m' : Nat) (h' : Nat) (min' : Nat),
    is_valid_date d' m' 1994 →
    is_valid_time h' min' →
    date_time_to_digits d' m' 1994 h' min' == digits →
    (d', m') > (5, 1) →
    (d', m') ≤ (d, m)

theorem next_occurrence_is_august_first :
  is_next_occurrence 1 8 2 45 := by sorry

end NUMINAMATH_CALUDE_next_occurrence_is_august_first_l1972_197249


namespace NUMINAMATH_CALUDE_average_height_four_people_l1972_197277

/-- The average height of four individuals given their relative heights -/
theorem average_height_four_people (G : ℝ) : 
  G + 2 = 64 →  -- Giselle is 2 inches shorter than Parker
  (G + 2) + 4 = 68 →  -- Parker is 4 inches shorter than Daisy
  68 - 8 = 60 →  -- Daisy is 8 inches taller than Reese
  (G + 64 + 68 + 60) / 4 = (192 + G) / 4 := by sorry

end NUMINAMATH_CALUDE_average_height_four_people_l1972_197277


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l1972_197256

def U : Finset Nat := {1, 3, 5, 7}
def M : Finset Nat := {1, 5}

theorem complement_of_M_in_U :
  (U \ M) = {3, 7} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l1972_197256


namespace NUMINAMATH_CALUDE_hoseok_took_fewest_l1972_197275

/-- Represents the number of cards taken by each person -/
structure CardCount where
  jungkook : ℕ
  hoseok : ℕ
  seokjin : ℕ

/-- Defines the conditions of the problem -/
def problemConditions (cc : CardCount) : Prop :=
  cc.jungkook = 10 ∧
  cc.hoseok = 7 ∧
  cc.seokjin = cc.jungkook - 2

/-- Theorem stating that Hoseok took the fewest cards -/
theorem hoseok_took_fewest (cc : CardCount) 
  (h : problemConditions cc) : 
  cc.hoseok < cc.jungkook ∧ cc.hoseok < cc.seokjin :=
by
  sorry

#check hoseok_took_fewest

end NUMINAMATH_CALUDE_hoseok_took_fewest_l1972_197275


namespace NUMINAMATH_CALUDE_water_polo_team_selection_l1972_197257

def total_team_members : ℕ := 20
def starting_lineup : ℕ := 7
def regular_players : ℕ := 5

def choose_team : ℕ := 
  total_team_members * (total_team_members - 1) * (Nat.choose (total_team_members - 2) regular_players)

theorem water_polo_team_selection :
  choose_team = 3268880 :=
sorry

end NUMINAMATH_CALUDE_water_polo_team_selection_l1972_197257


namespace NUMINAMATH_CALUDE_bob_win_probability_l1972_197218

theorem bob_win_probability (p_lose p_tie : ℚ) 
  (h_lose : p_lose = 5/8)
  (h_tie : p_tie = 1/8) : 
  1 - p_lose - p_tie = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_bob_win_probability_l1972_197218


namespace NUMINAMATH_CALUDE_equation_proof_l1972_197299

theorem equation_proof : 3889 + 12.952 - 47.95 = 3854.002 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1972_197299


namespace NUMINAMATH_CALUDE_sin_2alpha_plus_2pi_3_l1972_197250

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ) + 1

theorem sin_2alpha_plus_2pi_3 (ω φ α : ℝ) :
  ω > 0 →
  0 ≤ φ ∧ φ ≤ Real.pi / 2 →
  (∀ x : ℝ, f ω φ (x + Real.pi / ω) = f ω φ x) →
  f ω φ (Real.pi / 6) = 2 →
  f ω φ α = 9 / 5 →
  Real.pi / 6 < α ∧ α < 2 * Real.pi / 3 →
  Real.sin (2 * α + 2 * Real.pi / 3) = -24 / 25 := by
sorry

end NUMINAMATH_CALUDE_sin_2alpha_plus_2pi_3_l1972_197250


namespace NUMINAMATH_CALUDE_synthetic_method_deduces_result_from_cause_l1972_197219

/-- The synthetic method in mathematics -/
def synthetic_method : Type := Unit

/-- Property of deducing result from cause -/
def deduces_result_from_cause (m : Type) : Prop := sorry

/-- The synthetic method is a way of thinking in mathematics -/
axiom synthetic_method_is_way_of_thinking : synthetic_method = Unit

/-- Theorem: The synthetic method deduces the result from the cause -/
theorem synthetic_method_deduces_result_from_cause : 
  deduces_result_from_cause synthetic_method :=
sorry

end NUMINAMATH_CALUDE_synthetic_method_deduces_result_from_cause_l1972_197219


namespace NUMINAMATH_CALUDE_height_statistics_l1972_197232

/-- Heights of students in Class A -/
def class_a_heights : Finset ℕ := sorry

/-- Heights of students in Class B -/
def class_b_heights : Finset ℕ := sorry

/-- The mode of a finite set of natural numbers -/
def mode (s : Finset ℕ) : ℕ := sorry

/-- The median of a finite set of natural numbers -/
def median (s : Finset ℕ) : ℕ := sorry

/-- Theorem stating the mode of Class A heights and median of Class B heights -/
theorem height_statistics :
  mode class_a_heights = 171 ∧ median class_b_heights = 170 := by sorry

end NUMINAMATH_CALUDE_height_statistics_l1972_197232


namespace NUMINAMATH_CALUDE_bike_price_calculation_l1972_197208

theorem bike_price_calculation (upfront_payment : ℝ) (upfront_percentage : ℝ) 
  (h1 : upfront_payment = 200)
  (h2 : upfront_percentage = 0.20) :
  upfront_payment / upfront_percentage = 1000 := by
  sorry

end NUMINAMATH_CALUDE_bike_price_calculation_l1972_197208


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1972_197239

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1, focal length 10, and point P(2, 1) on its asymptote, prove that the equation of C is x²/20 - y²/5 = 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ c : ℝ, c > 0 ∧ c^2 = a^2 + b^2 ∧ 2*c = 10) →
  (∃ x y : ℝ, x = 2 ∧ y = 1 ∧ y = (b/a) * x) →
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ x^2/20 - y^2/5 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1972_197239


namespace NUMINAMATH_CALUDE_negative_five_times_three_l1972_197264

theorem negative_five_times_three : -5 * 3 = -15 := by
  sorry

end NUMINAMATH_CALUDE_negative_five_times_three_l1972_197264


namespace NUMINAMATH_CALUDE_concert_longest_song_duration_l1972_197233

/-- Represents the duration of the longest song in a concert --/
def longest_song_duration (total_time intermission_time num_songs regular_song_duration : ℕ) : ℕ :=
  total_time - intermission_time - (num_songs - 1) * regular_song_duration

/-- Theorem stating the duration of the longest song in the given concert scenario --/
theorem concert_longest_song_duration :
  longest_song_duration 80 10 13 5 = 10 := by sorry

end NUMINAMATH_CALUDE_concert_longest_song_duration_l1972_197233


namespace NUMINAMATH_CALUDE_cos_product_range_in_triangle_l1972_197202

theorem cos_product_range_in_triangle (A B C : ℝ) : 
  0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C ∧ 
  A + B + C = π ∧ 
  B = π / 3 → 
  -1/2 ≤ Real.cos A * Real.cos C ∧ Real.cos A * Real.cos C ≤ 1/4 :=
sorry

end NUMINAMATH_CALUDE_cos_product_range_in_triangle_l1972_197202


namespace NUMINAMATH_CALUDE_problem_solution_l1972_197268

-- Define x as the solution to the equation x = 1 + √3 / x
noncomputable def x : ℝ := Real.sqrt 3 + 1

-- State the theorem
theorem problem_solution : 
  1 / ((x + 1) * (x - 2)) = -(Real.sqrt 3 + 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1972_197268


namespace NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_cylinder_l1972_197273

/-- The surface area of a sphere circumscribing a right circular cylinder with edge length 6 -/
theorem sphere_surface_area_circumscribing_cylinder (r : ℝ) : r^2 = 21 → 4 * Real.pi * r^2 = 84 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_cylinder_l1972_197273


namespace NUMINAMATH_CALUDE_annual_pension_l1972_197213

/-- The annual pension problem -/
theorem annual_pension (c p q : ℝ) (x : ℝ) (k : ℝ) :
  (k * Real.sqrt (x + c) = k * Real.sqrt x + 3 * p) →
  (k * Real.sqrt (x + 2 * c) = k * Real.sqrt x + 4 * q) →
  (k * Real.sqrt x = (16 * q^2 - 18 * p^2) / (12 * p - 8 * q)) :=
by sorry

end NUMINAMATH_CALUDE_annual_pension_l1972_197213


namespace NUMINAMATH_CALUDE_arithmetic_progression_common_difference_l1972_197203

theorem arithmetic_progression_common_difference 
  (x y : ℤ) 
  (eq : 26 * x^2 + 23 * x * y - 3 * y^2 - 19 = 0) 
  (progression : ∃ (a d : ℤ), x = a + 5 * d ∧ y = a + 10 * d ∧ d < 0) :
  ∃ (a : ℤ), x = a + 5 * (-3) ∧ y = a + 10 * (-3) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_common_difference_l1972_197203


namespace NUMINAMATH_CALUDE_electronics_store_theorem_l1972_197235

theorem electronics_store_theorem (total : ℕ) (tv : ℕ) (computer : ℕ) (both : ℕ) 
  (h1 : total = 15)
  (h2 : tv = 9)
  (h3 : computer = 7)
  (h4 : both = 3)
  : total - (tv + computer - both) = 2 :=
by sorry

end NUMINAMATH_CALUDE_electronics_store_theorem_l1972_197235


namespace NUMINAMATH_CALUDE_diagonal_intersection_probability_l1972_197224

theorem diagonal_intersection_probability (n : ℕ) (h : n > 0) :
  let vertices := 2 * n + 1
  let total_diagonals := (vertices * (vertices - 3)) / 2
  let intersecting_diagonals := vertices.choose 4
  intersecting_diagonals / (total_diagonals.choose 2 : ℚ) = 
    n * (2 * n - 1) / (3 * (2 * n^2 - n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_diagonal_intersection_probability_l1972_197224


namespace NUMINAMATH_CALUDE_square_field_area_l1972_197222

/-- The area of a square field with a diagonal of 26 meters is 338.0625 square meters. -/
theorem square_field_area (d : ℝ) (h : d = 26) : 
  let s := d / Real.sqrt 2
  s^2 = 338.0625 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l1972_197222


namespace NUMINAMATH_CALUDE_trig_problem_l1972_197251

theorem trig_problem (α : Real) 
  (h1 : π / 2 < α) 
  (h2 : α < π) 
  (h3 : Real.tan α - 1 / Real.tan α = -3/2) : 
  Real.tan α = -2 ∧ 
  (Real.cos (3*π/2 + α) - Real.cos (π - α)) / Real.sin (π/2 - α) = -1 := by
  sorry

end NUMINAMATH_CALUDE_trig_problem_l1972_197251


namespace NUMINAMATH_CALUDE_overall_gain_percent_l1972_197289

/-- Calculates the overall gain percent after applying two discounts -/
theorem overall_gain_percent (M : ℝ) (M_pos : M > 0) : 
  let cost_price := 0.64 * M
  let price_after_first_discount := 0.86 * M
  let final_price := 0.9 * price_after_first_discount
  let gain := final_price - cost_price
  let gain_percent := (gain / cost_price) * 100
  ∃ ε > 0, |gain_percent - 20.94| < ε :=
by sorry

end NUMINAMATH_CALUDE_overall_gain_percent_l1972_197289


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l1972_197212

/-- A trapezoid with the given properties -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  diagonal : ℝ
  angle_between_diagonals : ℝ

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := sorry

/-- Theorem stating that a trapezoid with the given properties has a perimeter of 22 -/
theorem trapezoid_perimeter (t : Trapezoid) 
  (h1 : t.base1 = 3)
  (h2 : t.base2 = 5)
  (h3 : t.diagonal = 8)
  (h4 : t.angle_between_diagonals = 60 * π / 180) :
  perimeter t = 22 := by sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l1972_197212


namespace NUMINAMATH_CALUDE_total_earnings_value_l1972_197217

def friday_earnings : ℚ := 147
def saturday_earnings : ℚ := 2 * friday_earnings + 7
def sunday_earnings : ℚ := friday_earnings + 78
def monday_earnings : ℚ := 0.75 * friday_earnings
def tuesday_earnings : ℚ := 1.25 * monday_earnings
def wednesday_earnings : ℚ := 0.8 * tuesday_earnings

def total_earnings : ℚ := friday_earnings + saturday_earnings + sunday_earnings + 
                          monday_earnings + tuesday_earnings + wednesday_earnings

theorem total_earnings_value : total_earnings = 1031.3125 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_value_l1972_197217


namespace NUMINAMATH_CALUDE_min_good_pairs_l1972_197230

/-- A circular arrangement of natural numbers from 1 to 100 -/
def CircularArrangement := Fin 100 → ℕ

/-- Predicate to check if a number at index i satisfies the neighbor condition -/
def satisfies_neighbor_condition (arr : CircularArrangement) (i : Fin 100) : Prop :=
  (arr i > arr ((i + 1) % 100) ∧ arr i > arr ((i + 99) % 100)) ∨
  (arr i < arr ((i + 1) % 100) ∧ arr i < arr ((i + 99) % 100))

/-- Predicate to check if a pair at indices i and j form a "good pair" -/
def is_good_pair (arr : CircularArrangement) (i j : Fin 100) : Prop :=
  arr i > arr j ∧ satisfies_neighbor_condition arr i ∧ satisfies_neighbor_condition arr j

/-- The main theorem stating that any valid arrangement has at least 51 good pairs -/
theorem min_good_pairs (arr : CircularArrangement) 
  (h_valid : ∀ i, satisfies_neighbor_condition arr i)
  (h_distinct : ∀ i j, i ≠ j → arr i ≠ arr j)
  (h_range : ∀ i, arr i ∈ Finset.range 101 \ {0}) :
  ∃ (pairs : Finset (Fin 100 × Fin 100)), pairs.card ≥ 51 ∧ 
    ∀ (p : Fin 100 × Fin 100), p ∈ pairs → is_good_pair arr p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_min_good_pairs_l1972_197230


namespace NUMINAMATH_CALUDE_solution_set_for_a_5_range_of_a_l1972_197279

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 2|

-- Part I
theorem solution_set_for_a_5 :
  {x : ℝ | f 5 x > 9} = {x : ℝ | x < -6 ∨ x > 3} := by sorry

-- Part II
def A (a : ℝ) : Set ℝ := {x : ℝ | f a x ≤ |x - 4|}
def B : Set ℝ := {x : ℝ | |2*x - 1| ≤ 3}

theorem range_of_a :
  ∀ a : ℝ, (A a ∪ B = A a) → a ∈ Set.Icc (-1) 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_5_range_of_a_l1972_197279


namespace NUMINAMATH_CALUDE_complement_A_in_U_equals_union_l1972_197246

-- Define the universal set U
def U : Set ℝ := {x | -3 < x ∧ x < 3}

-- Define set A
def A : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define the complement of A in U
def complement_A_in_U : Set ℝ := {x | x ∈ U ∧ x ∉ A}

-- Theorem statement
theorem complement_A_in_U_equals_union : 
  complement_A_in_U = {x | (-3 < x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x < 3)} :=
by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_equals_union_l1972_197246


namespace NUMINAMATH_CALUDE_equal_cost_guests_correct_l1972_197238

/-- The number of guests for which the costs of renting Caesar's and Venus Hall are equal -/
def equal_cost_guests : ℕ :=
  let caesar_rental : ℕ := 800
  let caesar_per_meal : ℕ := 30
  let venus_rental : ℕ := 500
  let venus_per_meal : ℕ := 35
  60

theorem equal_cost_guests_correct :
  let caesar_rental : ℕ := 800
  let caesar_per_meal : ℕ := 30
  let venus_rental : ℕ := 500
  let venus_per_meal : ℕ := 35
  caesar_rental + caesar_per_meal * equal_cost_guests = venus_rental + venus_per_meal * equal_cost_guests :=
by sorry

end NUMINAMATH_CALUDE_equal_cost_guests_correct_l1972_197238


namespace NUMINAMATH_CALUDE_sprint_competition_correct_l1972_197283

def sprint_competition (total_sprinters : ℕ) (byes : ℕ) (first_round_lanes : ℕ) (subsequent_lanes : ℕ) : ℕ :=
  let first_round := (total_sprinters - byes + first_round_lanes - 1) / first_round_lanes
  let second_round := (first_round + byes + subsequent_lanes - 1) / subsequent_lanes
  let third_round := (second_round + subsequent_lanes - 1) / subsequent_lanes
  let final_round := 1
  first_round + second_round + third_round + final_round

theorem sprint_competition_correct :
  sprint_competition 300 16 8 6 = 48 := by
  sorry

end NUMINAMATH_CALUDE_sprint_competition_correct_l1972_197283


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1972_197245

/-- The sum of an infinite geometric series with first term 1 and common ratio 2/3 is 3 -/
theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := 2/3
  let S := ∑' n, a * r^n
  S = 3 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1972_197245


namespace NUMINAMATH_CALUDE_g_range_l1972_197276

noncomputable def g (x : ℝ) : ℝ := 
  (Real.sin x ^ 3 + 11 * Real.sin x ^ 2 + 3 * Real.sin x + 4 * Real.cos x ^ 2 - 10) / (Real.sin x - 2)

theorem g_range : 
  ∀ y ∈ Set.range g, 1 ≤ y ∧ y ≤ 19 ∧ 
  ∃ x : ℝ, g x = 1 ∧ 
  ∃ x : ℝ, g x = 19 :=
sorry

end NUMINAMATH_CALUDE_g_range_l1972_197276


namespace NUMINAMATH_CALUDE_cookie_cost_l1972_197227

theorem cookie_cost (cheeseburger_cost milkshake_cost coke_cost fries_cost tax : ℚ)
  (toby_initial toby_change : ℚ) (cookie_count : ℕ) :
  cheeseburger_cost = 365/100 ∧ 
  milkshake_cost = 2 ∧ 
  coke_cost = 1 ∧ 
  fries_cost = 4 ∧ 
  tax = 1/5 ∧
  toby_initial = 15 ∧ 
  toby_change = 7 ∧
  cookie_count = 3 →
  let total_before_cookies := 2 * cheeseburger_cost + milkshake_cost + coke_cost + fries_cost + tax
  let total_spent := 2 * (toby_initial - toby_change)
  let cookie_total_cost := total_spent - total_before_cookies
  cookie_total_cost / cookie_count = 1/2 := by
sorry

end NUMINAMATH_CALUDE_cookie_cost_l1972_197227


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1972_197209

theorem inequality_solution_set (x : ℝ) :
  -6 * x^2 - x + 2 ≤ 0 ↔ x ≥ (1/2 : ℝ) ∨ x ≤ -(2/3 : ℝ) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1972_197209


namespace NUMINAMATH_CALUDE_magnitude_of_BC_l1972_197237

/-- Given vectors BA and AC in R², prove that the magnitude of BC is 5 -/
theorem magnitude_of_BC (BA AC : ℝ × ℝ) : 
  BA = (3, -2) → AC = (0, 6) → ‖BA + AC‖ = 5 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_BC_l1972_197237


namespace NUMINAMATH_CALUDE_total_sample_variance_stratified_sampling_l1972_197214

/-- Calculates the total sample variance for stratified sampling of student heights -/
theorem total_sample_variance_stratified_sampling 
  (male_count : ℕ) 
  (female_count : ℕ) 
  (male_mean : ℝ) 
  (female_mean : ℝ) 
  (male_variance : ℝ) 
  (female_variance : ℝ) 
  (h_male_count : male_count = 100)
  (h_female_count : female_count = 60)
  (h_male_mean : male_mean = 172)
  (h_female_mean : female_mean = 164)
  (h_male_variance : male_variance = 18)
  (h_female_variance : female_variance = 30) :
  let total_count := male_count + female_count
  let combined_mean := (male_count * male_mean + female_count * female_mean) / total_count
  let total_variance := 
    (male_count : ℝ) / total_count * (male_variance + (male_mean - combined_mean)^2) +
    (female_count : ℝ) / total_count * (female_variance + (female_mean - combined_mean)^2)
  total_variance = 37.5 := by
sorry


end NUMINAMATH_CALUDE_total_sample_variance_stratified_sampling_l1972_197214


namespace NUMINAMATH_CALUDE_no_integer_roots_l1972_197223

theorem no_integer_roots (a b : ℤ) : 
  ¬ ∃ (x : ℤ), (x^2 + 10*a*x + 5*b + 3 = 0) ∨ (x^2 + 10*a*x + 5*b - 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_roots_l1972_197223


namespace NUMINAMATH_CALUDE_group_a_better_performance_l1972_197278

/-- Represents a group of students with their quiz scores -/
structure StudentGroup where
  scores : List Nat
  mean : Nat
  median : Nat
  mode : Nat
  variance : Nat
  excellent_rate : Rat

/-- Defines what score is considered excellent -/
def excellent_score : Nat := 8

/-- Group A data -/
def group_a : StudentGroup := {
  scores := [5, 7, 8, 8, 8, 8, 8, 9, 9, 10],
  mean := 8,
  median := 8,
  mode := 8,
  variance := 16,
  excellent_rate := 8 / 10
}

/-- Group B data -/
def group_b : StudentGroup := {
  scores := [7, 7, 7, 7, 8, 8, 8, 9, 9, 10],
  mean := 8,
  median := 8,
  mode := 7,
  variance := 1,
  excellent_rate := 6 / 10
}

/-- Theorem stating that Group A has a higher excellent rate than Group B -/
theorem group_a_better_performance (ga : StudentGroup) (gb : StudentGroup) 
  (h1 : ga = group_a) (h2 : gb = group_b) : 
  ga.excellent_rate > gb.excellent_rate := by
  sorry

end NUMINAMATH_CALUDE_group_a_better_performance_l1972_197278


namespace NUMINAMATH_CALUDE_survey_result_l1972_197211

/-- Represents the number of questionnaires collected from each unit -/
structure QuestionnaireData where
  total : ℕ
  sample : ℕ
  sample_b : ℕ

/-- Proves that given the conditions from the survey, the number of questionnaires drawn from unit D is 60 -/
theorem survey_result (data : QuestionnaireData) 
  (h_total : data.total = 1000)
  (h_sample : data.sample = 150)
  (h_sample_b : data.sample_b = 30)
  (h_arithmetic : ∃ (a d : ℚ), ∀ i : Fin 4, a + i * d = (data.total : ℚ) / 4)
  (h_prop_arithmetic : ∃ (b e : ℚ), ∀ i : Fin 4, b + i * e = (data.sample : ℚ) / 4 ∧ b + 1 * e = data.sample_b) :
  ∃ (b e : ℚ), b + 3 * e = 60 := by
sorry

end NUMINAMATH_CALUDE_survey_result_l1972_197211


namespace NUMINAMATH_CALUDE_car_distance_theorem_l1972_197271

/-- Theorem: A car traveling at 208 km/h for 3 hours covers a distance of 624 km. -/
theorem car_distance_theorem (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 208 → time = 3 → distance = speed * time → distance = 624 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l1972_197271


namespace NUMINAMATH_CALUDE_smaller_number_proof_l1972_197269

theorem smaller_number_proof (x y : ℤ) 
  (sum_condition : x + y = 84)
  (ratio_condition : y = 3 * x) :
  x = 21 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l1972_197269


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1972_197293

theorem quadratic_inequality_solution (c : ℝ) : 
  (∃ x ∈ Set.Ioo (-2 : ℝ) 1, x^2 + x - c < 0) → c = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1972_197293


namespace NUMINAMATH_CALUDE_inscribed_triangles_area_relation_l1972_197266

/-- Triangle type -/
structure Triangle where
  area : ℝ

/-- Inscribed triangle relation -/
def inscribed (outer inner : Triangle) : Prop :=
  inner.area < outer.area

/-- Parallel sides relation -/
def parallel_sides (t1 t2 : Triangle) : Prop :=
  true  -- We don't need to define this precisely for the theorem

/-- Theorem statement -/
theorem inscribed_triangles_area_relation (a b c : Triangle)
  (h1 : inscribed a b)
  (h2 : inscribed b c)
  (h3 : parallel_sides a c) :
  b.area = Real.sqrt (a.area * c.area) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangles_area_relation_l1972_197266


namespace NUMINAMATH_CALUDE_ana_charging_time_proof_l1972_197272

def smartphone_full_charge : ℕ := 26
def tablet_full_charge : ℕ := 53

def ana_charging_time : ℕ :=
  tablet_full_charge + (smartphone_full_charge / 2)

theorem ana_charging_time_proof :
  ana_charging_time = 66 := by
  sorry

end NUMINAMATH_CALUDE_ana_charging_time_proof_l1972_197272


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l1972_197220

theorem quadratic_one_solution (n : ℝ) : 
  (∃! x, 4 * x^2 + n * x + 4 = 0) ↔ (n = 8 ∨ n = -8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l1972_197220


namespace NUMINAMATH_CALUDE_shirt_price_markdown_l1972_197270

/-- Given a shirt price that goes through two markdowns, prove that the initial sale price
    was 70% of the original price if the second markdown is 10% and the final price
    is 63% of the original price. -/
theorem shirt_price_markdown (original_price : ℝ) (initial_sale_price : ℝ) :
  initial_sale_price > 0 →
  original_price > 0 →
  initial_sale_price * 0.9 = original_price * 0.63 →
  initial_sale_price / original_price = 0.7 := by
sorry

end NUMINAMATH_CALUDE_shirt_price_markdown_l1972_197270
