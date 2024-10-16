import Mathlib

namespace NUMINAMATH_CALUDE_derivative_sum_at_one_l1292_129261

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State the theorem
theorem derivative_sum_at_one 
  (h1 : ∀ x, f x + x * g x = x^2 - 1) 
  (h2 : f 1 = 1) : 
  deriv f 1 + deriv g 1 = 3 := by
sorry

end NUMINAMATH_CALUDE_derivative_sum_at_one_l1292_129261


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1292_129280

/-- An arithmetic sequence with integer common ratio -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  q : ℤ
  seq_def : ∀ n, a (n + 1) = a n + q

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.a 1 * n + seq.q * (n * (n - 1) / 2)

theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h1 : seq.a 2 - seq.a 3 = -2)
  (h2 : seq.a 1 + seq.a 3 = 10/3) :
  sum_n seq 4 = 40/3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1292_129280


namespace NUMINAMATH_CALUDE_expected_heads_after_turn_l1292_129212

/-- Represents the state of pennies on a table -/
structure PennyState where
  total : ℕ
  heads : ℕ
  tails : ℕ

/-- Represents the action of turning over pennies -/
def turn_pennies (state : PennyState) (num_turn : ℕ) : ℝ :=
  let p_heads := state.heads / state.total
  let p_tails := state.tails / state.total
  let expected_heads_turned := num_turn * p_heads
  let expected_tails_turned := num_turn * p_tails
  state.heads - expected_heads_turned + expected_tails_turned

/-- The main theorem to prove -/
theorem expected_heads_after_turn (initial_state : PennyState) 
  (h1 : initial_state.total = 100)
  (h2 : initial_state.heads = 30)
  (h3 : initial_state.tails = 70)
  (num_turn : ℕ)
  (h4 : num_turn = 40) :
  turn_pennies initial_state num_turn = 46 := by
  sorry


end NUMINAMATH_CALUDE_expected_heads_after_turn_l1292_129212


namespace NUMINAMATH_CALUDE_max_gold_marbles_is_66_l1292_129221

/-- Represents the number of marbles of each color --/
structure MarbleCount where
  red : ℕ
  blue : ℕ
  gold : ℕ

/-- Represents an exchange of marbles --/
inductive Exchange
  | RedToGold : Exchange
  | BlueToGold : Exchange

/-- Applies an exchange to a MarbleCount --/
def applyExchange (mc : MarbleCount) (e : Exchange) : MarbleCount :=
  match e with
  | Exchange.RedToGold => 
      if mc.red ≥ 3 then ⟨mc.red - 3, mc.blue + 2, mc.gold + 1⟩ else mc
  | Exchange.BlueToGold => 
      if mc.blue ≥ 4 then ⟨mc.red + 1, mc.blue - 4, mc.gold + 1⟩ else mc

/-- Checks if any exchange is possible --/
def canExchange (mc : MarbleCount) : Prop :=
  mc.red ≥ 3 ∨ mc.blue ≥ 4

/-- The maximum number of gold marbles obtainable --/
def maxGoldMarbles : ℕ := 66

/-- The theorem to be proved --/
theorem max_gold_marbles_is_66 :
  ∀ (exchanges : List Exchange),
    let finalCount := (exchanges.foldl applyExchange ⟨80, 60, 0⟩)
    ¬(canExchange finalCount) →
    finalCount.gold = maxGoldMarbles :=
  sorry

end NUMINAMATH_CALUDE_max_gold_marbles_is_66_l1292_129221


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l1292_129237

theorem smallest_solution_abs_equation :
  let f := fun x : ℝ => x * |x| - (3 * x - 2)
  ∃ x₀ : ℝ, f x₀ = 0 ∧ ∀ x : ℝ, f x = 0 → x₀ ≤ x ∧ x₀ = (-3 - Real.sqrt 17) / 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l1292_129237


namespace NUMINAMATH_CALUDE_probability_three_two_correct_l1292_129215

/-- The number of slips in the hat -/
def total_slips : ℕ := 50

/-- The number of different numbers on the slips -/
def different_numbers : ℕ := 10

/-- The number of slips for each number -/
def slips_per_number : ℕ := 5

/-- The number of slips drawn -/
def drawn_slips : ℕ := 5

/-- The probability of drawing exactly 3 slips with one number and 2 slips with another number -/
def probability_three_two : ℚ := 75 / 35313

theorem probability_three_two_correct :
  probability_three_two = (different_numbers.choose 2 * slips_per_number.choose 3 * slips_per_number.choose 2) / total_slips.choose drawn_slips :=
by sorry

end NUMINAMATH_CALUDE_probability_three_two_correct_l1292_129215


namespace NUMINAMATH_CALUDE_exists_tetrahedron_all_obtuse_dihedral_angles_l1292_129244

/-- A tetrahedron is represented by its four vertices in 3D space -/
def Tetrahedron := Fin 4 → ℝ × ℝ × ℝ

/-- The dihedral angle between two faces of a tetrahedron -/
def dihedralAngle (t : Tetrahedron) (i j : Fin 4) : ℝ :=
  sorry  -- Definition of dihedral angle calculation

/-- A dihedral angle is obtuse if it's greater than π/2 -/
def isObtuse (angle : ℝ) : Prop := angle > Real.pi / 2

/-- Theorem: There exists a tetrahedron where all dihedral angles are obtuse -/
theorem exists_tetrahedron_all_obtuse_dihedral_angles :
  ∃ t : Tetrahedron, ∀ i j : Fin 4, i ≠ j → isObtuse (dihedralAngle t i j) :=
sorry

end NUMINAMATH_CALUDE_exists_tetrahedron_all_obtuse_dihedral_angles_l1292_129244


namespace NUMINAMATH_CALUDE_compute_fraction_power_l1292_129242

theorem compute_fraction_power : 8 * (2 / 7)^4 = 128 / 2401 := by
  sorry

end NUMINAMATH_CALUDE_compute_fraction_power_l1292_129242


namespace NUMINAMATH_CALUDE_rectangle_length_fraction_of_circle_radius_l1292_129259

theorem rectangle_length_fraction_of_circle_radius 
  (square_area : ℝ) 
  (rectangle_area : ℝ) 
  (rectangle_breadth : ℝ) 
  (h1 : square_area = 3025) 
  (h2 : rectangle_area = 220) 
  (h3 : rectangle_breadth = 10) : 
  (rectangle_area / rectangle_breadth) / Real.sqrt square_area = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_fraction_of_circle_radius_l1292_129259


namespace NUMINAMATH_CALUDE_slope_angle_sqrt3_l1292_129201

/-- The slope angle of a line with slope √3 is 60 degrees. -/
theorem slope_angle_sqrt3 : ∃ θ : Real, 
  0 ≤ θ ∧ θ < Real.pi ∧ 
  Real.tan θ = Real.sqrt 3 ∧ 
  θ = Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_slope_angle_sqrt3_l1292_129201


namespace NUMINAMATH_CALUDE_beth_class_size_l1292_129271

/-- The number of students in Beth's class over three years -/
def final_class_size (initial : ℕ) (joined : ℕ) (left : ℕ) : ℕ :=
  initial + joined - left

/-- Theorem stating the final class size given the initial conditions -/
theorem beth_class_size :
  final_class_size 150 30 15 = 165 := by
  sorry

end NUMINAMATH_CALUDE_beth_class_size_l1292_129271


namespace NUMINAMATH_CALUDE_brocard_angle_inequalities_l1292_129205

/-- The Brocard angle of a triangle -/
def brocard_angle (α β γ : ℝ) : ℝ := sorry

/-- Theorem: Brocard angle inequalities -/
theorem brocard_angle_inequalities (α β γ : ℝ) (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ) 
  (hsum : α + β + γ = π) :
  let φ := brocard_angle α β γ
  (φ^3 ≤ (α - φ) * (β - φ) * (γ - φ)) ∧ (8 * φ^3 ≤ α * β * γ) := by sorry

end NUMINAMATH_CALUDE_brocard_angle_inequalities_l1292_129205


namespace NUMINAMATH_CALUDE_first_two_digits_sum_l1292_129210

/-- The number of integer lattice points (x, y) satisfying 4x^2 + 9y^2 ≤ 1000000000 -/
def N : ℕ := sorry

/-- The first digit of N -/
def a : ℕ := sorry

/-- The second digit of N -/
def b : ℕ := sorry

/-- Theorem stating that 10a + b equals 52 -/
theorem first_two_digits_sum : 10 * a + b = 52 := by sorry

end NUMINAMATH_CALUDE_first_two_digits_sum_l1292_129210


namespace NUMINAMATH_CALUDE_divisible_by_eight_l1292_129290

theorem divisible_by_eight (n : ℕ) : ∃ k : ℤ, 5^n + 2 * 3^(n-1) + 1 = 8*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eight_l1292_129290


namespace NUMINAMATH_CALUDE_square_minus_a_nonpositive_implies_a_geq_four_l1292_129274

theorem square_minus_a_nonpositive_implies_a_geq_four :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_a_nonpositive_implies_a_geq_four_l1292_129274


namespace NUMINAMATH_CALUDE_certain_number_proof_l1292_129253

theorem certain_number_proof (x : ℝ) : 
  0.8 * 170 - 0.35 * x = 31 → x = 300 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1292_129253


namespace NUMINAMATH_CALUDE_pentagon_hexagon_side_difference_l1292_129248

theorem pentagon_hexagon_side_difference (e : ℕ) : 
  (∃ (p h : ℝ), 5 * p - 6 * h = 1240 ∧ p - h = e ∧ 5 * p > 0 ∧ 6 * h > 0) ↔ e > 248 :=
sorry

end NUMINAMATH_CALUDE_pentagon_hexagon_side_difference_l1292_129248


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_f_prime_l1292_129245

def f (x : ℝ) : ℝ := (1 - 2*x)^10

theorem coefficient_x_squared_in_f_prime : 
  ∃ (g : ℝ → ℝ), (∀ x, deriv f x = g x) ∧ 
  (∃ (a b c : ℝ), ∀ x, g x = a*x^2 + b*x + c) ∧
  (∃ (a b c : ℝ), (∀ x, g x = a*x^2 + b*x + c) ∧ a = -2880) :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_f_prime_l1292_129245


namespace NUMINAMATH_CALUDE_sale_price_calculation_l1292_129209

/-- Calculates the sale price including tax given the cost price, profit rate, and tax rate -/
def salePriceWithTax (costPrice : ℝ) (profitRate : ℝ) (taxRate : ℝ) : ℝ :=
  costPrice * (1 + profitRate) * (1 + taxRate)

/-- The sale price including tax is 677.60 given the specified conditions -/
theorem sale_price_calculation :
  let costPrice : ℝ := 535.65
  let profitRate : ℝ := 0.15
  let taxRate : ℝ := 0.10
  ∃ ε > 0, |salePriceWithTax costPrice profitRate taxRate - 677.60| < ε :=
by
  sorry

#eval salePriceWithTax 535.65 0.15 0.10

end NUMINAMATH_CALUDE_sale_price_calculation_l1292_129209


namespace NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l1292_129225

theorem arithmetic_sequence_nth_term (x : ℚ) (n : ℕ) : 
  (3*x - 5 : ℚ) = (7*x - 17) - ((7*x - 17) - (3*x - 5)) → 
  (7*x - 17 : ℚ) = (4*x + 3) - ((4*x + 3) - (7*x - 17)) → 
  (∃ a d : ℚ, a = 3*x - 5 ∧ d = (7*x - 17) - (3*x - 5) ∧ 
    a + (n - 1) * d = 4033) → 
  n = 641 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l1292_129225


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1292_129238

-- First expression
theorem simplify_expression_1 (x : ℝ) : 2 * x - 3 * (x - 1) = 3 - x := by sorry

-- Second expression
theorem simplify_expression_2 (a b : ℝ) : 
  6 * (a * b^2 - a^2 * b) - 2 * (3 * a^2 * b + a * b^2) = 4 * a * b^2 - 12 * a^2 * b := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1292_129238


namespace NUMINAMATH_CALUDE_sqrt_square_nine_l1292_129239

theorem sqrt_square_nine : Real.sqrt 9 ^ 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_nine_l1292_129239


namespace NUMINAMATH_CALUDE_range_of_m_l1292_129298

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B : Set ℝ := {x | 3 - |x| ≥ 0}

-- Define the set C parameterized by m
def C (m : ℝ) : Set ℝ := {x | (x - m + 1) * (x - 2*m - 1) < 0}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (C m ⊆ B) ↔ m ∈ Set.Icc (-2) 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1292_129298


namespace NUMINAMATH_CALUDE_symmetric_point_in_fourth_quadrant_l1292_129203

theorem symmetric_point_in_fourth_quadrant (a : ℝ) (P : ℝ × ℝ) :
  a < 0 →
  P = (-a^2 - 1, -a + 3) →
  (∃ P1 : ℝ × ℝ, P1 = (-P.1, -P.2) ∧ P1.1 > 0 ∧ P1.2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_in_fourth_quadrant_l1292_129203


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_6_l1292_129252

theorem smallest_common_multiple_of_8_and_6 : ∃ n : ℕ+, (∀ m : ℕ+, 8 ∣ m ∧ 6 ∣ m → n ≤ m) ∧ 8 ∣ n ∧ 6 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_6_l1292_129252


namespace NUMINAMATH_CALUDE_monochromatic_four_cycle_exists_l1292_129240

/-- A coloring of edges in a graph using two colors -/
def TwoColoring (V : Type*) := V → V → Bool

/-- A complete graph with 6 vertices -/
def CompleteGraph6 := Fin 6

/-- A 4-cycle in a graph -/
def FourCycle (V : Type*) := 
  (V × V × V × V)

/-- Predicate to check if a 4-cycle is monochromatic under a given coloring -/
def IsMonochromatic (c : TwoColoring CompleteGraph6) (cycle : FourCycle CompleteGraph6) : Prop :=
  let (a, b, d, e) := cycle
  c a b = c b d ∧ c b d = c d e ∧ c d e = c e a

/-- Main theorem: In a complete graph with 6 vertices where each edge is colored 
    with one of two colors, there exists a monochromatic 4-cycle -/
theorem monochromatic_four_cycle_exists :
  ∀ (c : TwoColoring CompleteGraph6),
  ∃ (cycle : FourCycle CompleteGraph6), IsMonochromatic c cycle :=
sorry


end NUMINAMATH_CALUDE_monochromatic_four_cycle_exists_l1292_129240


namespace NUMINAMATH_CALUDE_range_of_g_l1292_129260

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 - 2

-- Define the function g as the composition of f with itself
def g (x : ℝ) : ℝ := f (f x)

-- State the theorem about the range of g
theorem range_of_g :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 1 ≤ g x ∧ g x ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_range_of_g_l1292_129260


namespace NUMINAMATH_CALUDE_terms_before_one_l1292_129234

/-- An arithmetic sequence with first term 100 and common difference -3 -/
def arithmeticSequence : ℕ → ℤ := λ n => 100 - 3 * (n - 1)

/-- The position of 1 in the sequence -/
def positionOfOne : ℕ := 34

theorem terms_before_one :
  (∀ k < positionOfOne, arithmeticSequence k > 1) ∧
  arithmeticSequence positionOfOne = 1 ∧
  positionOfOne - 1 = 33 := by sorry

end NUMINAMATH_CALUDE_terms_before_one_l1292_129234


namespace NUMINAMATH_CALUDE_mean_proportional_problem_l1292_129211

theorem mean_proportional_problem (x : ℝ) : 
  (156 : ℝ) = Real.sqrt (234 * x) → x = 104 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_problem_l1292_129211


namespace NUMINAMATH_CALUDE_quadratic_root_implies_q_value_l1292_129226

theorem quadratic_root_implies_q_value (p q : ℝ) (h : Complex.I ^ 2 = -1) :
  (3 * (1 + 4 * Complex.I) ^ 2 + p * (1 + 4 * Complex.I) + q = 0) → q = 51 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_q_value_l1292_129226


namespace NUMINAMATH_CALUDE_original_denominator_proof_l1292_129286

theorem original_denominator_proof (d : ℕ) : 
  (3 : ℚ) / d ≠ 0 →
  (3 + 6 : ℚ) / (d + 6) = 1 / 3 →
  d = 21 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l1292_129286


namespace NUMINAMATH_CALUDE_incorrect_multiplication_l1292_129264

theorem incorrect_multiplication : 79133 * 111107 ≠ 8794230231 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_multiplication_l1292_129264


namespace NUMINAMATH_CALUDE_time_after_1750_minutes_l1292_129269

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  sorry

/-- Converts a number of minutes to hours and minutes -/
def minutesToTime (m : Nat) : Time :=
  sorry

theorem time_after_1750_minutes :
  let start_time : Time := ⟨8, 0, by sorry, by sorry⟩
  let added_time : Time := minutesToTime 1750
  let end_time : Time := addMinutes start_time 1750
  end_time = ⟨13, 10, by sorry, by sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_time_after_1750_minutes_l1292_129269


namespace NUMINAMATH_CALUDE_fiona_owns_three_hoodies_l1292_129279

/-- The number of hoodies Fiona owns -/
def fiona_hoodies : ℕ := sorry

/-- The number of hoodies Casey owns -/
def casey_hoodies : ℕ := sorry

/-- The total number of hoodies Fiona and Casey own -/
def total_hoodies : ℕ := 8

theorem fiona_owns_three_hoodies :
  fiona_hoodies = 3 ∧ casey_hoodies = fiona_hoodies + 2 ∧ fiona_hoodies + casey_hoodies = total_hoodies :=
sorry

end NUMINAMATH_CALUDE_fiona_owns_three_hoodies_l1292_129279


namespace NUMINAMATH_CALUDE_andrew_work_hours_l1292_129230

/-- The number of days Andrew worked on his Science report -/
def days_worked : ℝ := 3

/-- The number of hours Andrew worked each day -/
def hours_per_day : ℝ := 2.5

/-- The total number of hours Andrew worked on his Science report -/
def total_hours : ℝ := days_worked * hours_per_day

theorem andrew_work_hours : total_hours = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_andrew_work_hours_l1292_129230


namespace NUMINAMATH_CALUDE_marble_fraction_after_tripling_l1292_129282

theorem marble_fraction_after_tripling (total : ℚ) (h : total > 0) :
  let initial_blue : ℚ := (4 / 7) * total
  let initial_red : ℚ := total - initial_blue
  let new_red : ℚ := 3 * initial_red
  let new_total : ℚ := initial_blue + new_red
  new_red / new_total = 9 / 13 :=
by sorry

end NUMINAMATH_CALUDE_marble_fraction_after_tripling_l1292_129282


namespace NUMINAMATH_CALUDE_log_properties_l1292_129283

-- Define approximate values for log₁₀ 2 and log₁₀ 3
def log10_2 : ℝ := 0.3010
def log10_3 : ℝ := 0.4771

-- Define the properties to be proved
theorem log_properties :
  let log10_27 := 3 * log10_3
  let log10_100_div_9 := 2 - 2 * log10_3
  let log10_sqrt_10 := (1 : ℝ) / 2
  (log10_27 = 3 * log10_3) ∧
  (log10_100_div_9 = 2 - 2 * log10_3) ∧
  (log10_sqrt_10 = (1 : ℝ) / 2) := by
  sorry


end NUMINAMATH_CALUDE_log_properties_l1292_129283


namespace NUMINAMATH_CALUDE_tangent_line_triangle_area_l1292_129273

/-- The area of the triangle formed by the tangent line to y = e^x at (2, e^2) and the coordinate axes -/
theorem tangent_line_triangle_area : 
  let f (x : ℝ) := Real.exp x
  let x₀ : ℝ := 2
  let y₀ : ℝ := Real.exp x₀
  let m : ℝ := Real.exp x₀  -- slope of the tangent line
  let b : ℝ := y₀ - m * x₀  -- y-intercept of the tangent line
  let x_intercept : ℝ := -b / m  -- x-intercept of the tangent line
  Real.exp 2 / 2 = (x_intercept * y₀) / 2 := by sorry

end NUMINAMATH_CALUDE_tangent_line_triangle_area_l1292_129273


namespace NUMINAMATH_CALUDE_average_score_is_correct_rounded_average_score_l1292_129263

/-- Represents the score distribution for a class test --/
structure ScoreDistribution where
  score_100 : Nat
  score_95  : Nat
  score_85  : Nat
  score_75  : Nat
  score_65  : Nat
  score_55  : Nat
  score_45  : Nat

/-- Calculates the average score given a score distribution --/
def calculateAverageScore (dist : ScoreDistribution) : Rat :=
  let totalStudents := dist.score_100 + dist.score_95 + dist.score_85 + 
                       dist.score_75 + dist.score_65 + dist.score_55 + dist.score_45
  let totalScore := 100 * dist.score_100 + 95 * dist.score_95 + 85 * dist.score_85 +
                    75 * dist.score_75 + 65 * dist.score_65 + 55 * dist.score_55 +
                    45 * dist.score_45
  totalScore / totalStudents

/-- The main theorem stating that the average score is approximately 76.3333 --/
theorem average_score_is_correct (dist : ScoreDistribution) 
  (h1 : dist.score_100 = 10)
  (h2 : dist.score_95 = 20)
  (h3 : dist.score_85 = 40)
  (h4 : dist.score_75 = 30)
  (h5 : dist.score_65 = 25)
  (h6 : dist.score_55 = 15)
  (h7 : dist.score_45 = 10) :
  calculateAverageScore dist = 11450 / 150 := by
  sorry

/-- The rounded average score is 76 --/
theorem rounded_average_score (dist : ScoreDistribution)
  (h : calculateAverageScore dist = 11450 / 150) :
  Int.floor (calculateAverageScore dist + 1/2) = 76 := by
  sorry

end NUMINAMATH_CALUDE_average_score_is_correct_rounded_average_score_l1292_129263


namespace NUMINAMATH_CALUDE_cos_two_alpha_value_l1292_129255

theorem cos_two_alpha_value (α : ℝ) (h : Real.sin α + Real.cos α = 2/3) :
  Real.cos (2 * α) = 2 * Real.sqrt 14 / 9 ∨ Real.cos (2 * α) = -2 * Real.sqrt 14 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_alpha_value_l1292_129255


namespace NUMINAMATH_CALUDE_infinite_triples_exist_l1292_129213

/-- An infinite, strictly increasing sequence of positive integers -/
def StrictlyIncreasingSeq (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- The condition satisfied by the sequence -/
def SequenceCondition (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (a n) ≤ a n + a (n + 3)

/-- The existence of infinitely many triples satisfying the condition -/
def InfinitelyManyTriples (a : ℕ → ℕ) : Prop :=
  ∀ N : ℕ, ∃ k l m : ℕ, k > N ∧ l > k ∧ m > l ∧ a k + a m = 2 * a l

/-- The main theorem -/
theorem infinite_triples_exist (a : ℕ → ℕ) 
  (h1 : StrictlyIncreasingSeq a) 
  (h2 : SequenceCondition a) : 
  InfinitelyManyTriples a :=
sorry

end NUMINAMATH_CALUDE_infinite_triples_exist_l1292_129213


namespace NUMINAMATH_CALUDE_circle_properties_l1292_129285

/-- Given a circle with radius 6, prove the area of the smallest square containing it and its circumference. -/
theorem circle_properties : 
  let r : ℝ := 6
  let square_area : ℝ := (2 * r) ^ 2
  let circle_circumference : ℝ := 2 * π * r
  square_area = 144 ∧ circle_circumference = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l1292_129285


namespace NUMINAMATH_CALUDE_first_sampled_item_l1292_129229

/-- Systematic sampling function -/
def systematicSample (totalItems : ℕ) (sampleSize : ℕ) (sixteenthItem : ℕ) : ℕ :=
  sixteenthItem - (sampleSize - 1) * ((totalItems / sampleSize) - 1)

/-- Theorem: First sampled item in the given systematic sampling scenario -/
theorem first_sampled_item :
  systematicSample 160 20 125 = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_sampled_item_l1292_129229


namespace NUMINAMATH_CALUDE_decrement_value_proof_l1292_129288

theorem decrement_value_proof (n : ℕ) (original_mean updated_mean : ℚ) 
  (h1 : n = 50)
  (h2 : original_mean = 200)
  (h3 : updated_mean = 153) :
  (n : ℚ) * original_mean - n * updated_mean = n * 47 := by
  sorry

end NUMINAMATH_CALUDE_decrement_value_proof_l1292_129288


namespace NUMINAMATH_CALUDE_no_eulerian_or_hamiltonian_path_l1292_129296

/-- A graph representing the science museum layout. -/
structure MuseumGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  bipartite : Finset Nat × Finset Nat
  degree_three : Finset Nat

/-- Predicate for the existence of an Eulerian path in the graph. -/
def has_eulerian_path (g : MuseumGraph) : Prop :=
  ∃ (path : List (Nat × Nat)), path.Nodup ∧ path.length = g.edges.card

/-- Predicate for the existence of a Hamiltonian path in the graph. -/
def has_hamiltonian_path (g : MuseumGraph) : Prop :=
  ∃ (path : List Nat), path.Nodup ∧ path.length = g.vertices.card

/-- The main theorem stating the non-existence of Eulerian and Hamiltonian paths. -/
theorem no_eulerian_or_hamiltonian_path (g : MuseumGraph)
  (h1 : g.vertices.card = 19)
  (h2 : g.edges.card = 30)
  (h3 : g.bipartite.1.card = 7 ∧ g.bipartite.2.card = 12)
  (h4 : g.degree_three.card ≥ 6) :
  ¬(has_eulerian_path g) ∧ ¬(has_hamiltonian_path g) := by
  sorry

#check no_eulerian_or_hamiltonian_path

end NUMINAMATH_CALUDE_no_eulerian_or_hamiltonian_path_l1292_129296


namespace NUMINAMATH_CALUDE_four_special_numbers_exist_l1292_129233

theorem four_special_numbers_exist : ∃ (a b c d : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  (¬(2 ∣ a) ∧ ¬(3 ∣ a) ∧ ¬(4 ∣ a)) ∧
  (¬(2 ∣ b) ∧ ¬(3 ∣ b) ∧ ¬(4 ∣ b)) ∧
  (¬(2 ∣ c) ∧ ¬(3 ∣ c) ∧ ¬(4 ∣ c)) ∧
  (¬(2 ∣ d) ∧ ¬(3 ∣ d) ∧ ¬(4 ∣ d)) ∧
  (2 ∣ (a + b)) ∧ (2 ∣ (a + c)) ∧ (2 ∣ (a + d)) ∧
  (2 ∣ (b + c)) ∧ (2 ∣ (b + d)) ∧ (2 ∣ (c + d)) ∧
  (3 ∣ (a + b + c)) ∧ (3 ∣ (a + b + d)) ∧
  (3 ∣ (a + c + d)) ∧ (3 ∣ (b + c + d)) ∧
  (4 ∣ (a + b + c + d)) := by
  sorry

#check four_special_numbers_exist

end NUMINAMATH_CALUDE_four_special_numbers_exist_l1292_129233


namespace NUMINAMATH_CALUDE_first_player_wins_98_max_n_first_player_wins_l1292_129220

/-- Represents the game board -/
def Board := Fin 1000 → Bool

/-- Represents a player's move -/
inductive Move
| Place (pos : Fin 1000) (num : Nat)
| Remove (start : Fin 1000) (len : Nat)

/-- Represents a player's strategy -/
def Strategy := Board → Move

/-- Applies a move to the board -/
def applyMove (b : Board) (m : Move) : Board :=
  sorry

/-- Checks if all tokens are placed in a row without gaps -/
def isWinningState (b : Board) : Prop :=
  sorry

/-- The game's rules and win condition -/
def gameRules (n : Nat) (s1 s2 : Strategy) : Prop :=
  sorry

/-- Theorem: First player can always win for n = 98 -/
theorem first_player_wins_98 :
  ∃ (s1 : Strategy), ∀ (s2 : Strategy), gameRules 98 s1 s2 → isWinningState (sorry : Board) :=
  sorry

/-- Theorem: 98 is the maximum n for which first player can always win -/
theorem max_n_first_player_wins :
  (∃ (s1 : Strategy), ∀ (s2 : Strategy), gameRules 98 s1 s2 → isWinningState (sorry : Board)) ∧
  (∀ n > 98, ∃ (s2 : Strategy), ∀ (s1 : Strategy), ¬(gameRules n s1 s2 → isWinningState (sorry : Board))) :=
  sorry

end NUMINAMATH_CALUDE_first_player_wins_98_max_n_first_player_wins_l1292_129220


namespace NUMINAMATH_CALUDE_initial_lambs_correct_l1292_129207

/-- The number of lambs Mary initially had -/
def initial_lambs : ℕ := 6

/-- The number of lambs that had babies -/
def lambs_with_babies : ℕ := 2

/-- The number of babies each lamb had -/
def babies_per_lamb : ℕ := 2

/-- The number of lambs Mary traded -/
def traded_lambs : ℕ := 3

/-- The number of extra lambs Mary found -/
def found_lambs : ℕ := 7

/-- The total number of lambs Mary has now -/
def total_lambs : ℕ := 14

/-- Theorem stating that the initial number of lambs is correct given the conditions -/
theorem initial_lambs_correct : 
  initial_lambs + (lambs_with_babies * babies_per_lamb) - traded_lambs + found_lambs = total_lambs :=
by sorry

end NUMINAMATH_CALUDE_initial_lambs_correct_l1292_129207


namespace NUMINAMATH_CALUDE_investment_rate_proof_l1292_129216

/-- Calculates the simple interest given principal, rate, and time -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem investment_rate_proof (principal : ℝ) (time : ℝ) (rate : ℝ) :
  principal = 7000 →
  time = 2 →
  simpleInterest principal rate time = simpleInterest principal 0.12 time + 420 →
  rate = 0.15 := by
sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l1292_129216


namespace NUMINAMATH_CALUDE_sheet_area_difference_l1292_129293

/-- The combined area (front and back) of a rectangular sheet of paper -/
def combinedArea (length width : ℝ) : ℝ := 2 * length * width

/-- The difference in combined area between two rectangular sheets of paper -/
def areaDifference (l1 w1 l2 w2 : ℝ) : ℝ :=
  combinedArea l1 w1 - combinedArea l2 w2

theorem sheet_area_difference :
  areaDifference 11 17 8.5 11 = 187 := by
  sorry

end NUMINAMATH_CALUDE_sheet_area_difference_l1292_129293


namespace NUMINAMATH_CALUDE_closest_value_is_112_l1292_129236

def original_value : ℝ := 50.5
def increase_percentage : ℝ := 0.05
def additional_value : ℝ := 0.15
def multiplier : ℝ := 2.1

def options : List ℝ := [105, 110, 112, 115, 120]

def calculated_value : ℝ := multiplier * ((original_value * (1 + increase_percentage)) + additional_value)

theorem closest_value_is_112 : 
  (options.argmin (λ x => |x - calculated_value|)) = some 112 := by sorry

end NUMINAMATH_CALUDE_closest_value_is_112_l1292_129236


namespace NUMINAMATH_CALUDE_tim_watch_time_l1292_129228

/-- The number of shows Tim watches -/
def num_shows : ℕ := 2

/-- The duration of the short show in hours -/
def short_show_duration : ℚ := 1/2

/-- The duration of the long show in hours -/
def long_show_duration : ℕ := 1

/-- The number of episodes of the short show -/
def short_show_episodes : ℕ := 24

/-- The number of episodes of the long show -/
def long_show_episodes : ℕ := 12

/-- The total number of hours Tim watched TV -/
def total_watch_time : ℚ := short_show_duration * short_show_episodes + long_show_duration * long_show_episodes

theorem tim_watch_time :
  total_watch_time = 24 := by sorry

end NUMINAMATH_CALUDE_tim_watch_time_l1292_129228


namespace NUMINAMATH_CALUDE_flower_shop_problem_l1292_129267

/-- Flower shop problem -/
theorem flower_shop_problem (roses_per_bouquet : ℕ) 
  (total_bouquets : ℕ) (rose_bouquets : ℕ) (daisy_bouquets : ℕ) 
  (total_flowers : ℕ) : 
  roses_per_bouquet = 12 →
  total_bouquets = 20 →
  rose_bouquets = 10 →
  daisy_bouquets = 10 →
  rose_bouquets + daisy_bouquets = total_bouquets →
  total_flowers = 190 →
  (total_flowers - roses_per_bouquet * rose_bouquets) / daisy_bouquets = 7 :=
by sorry

end NUMINAMATH_CALUDE_flower_shop_problem_l1292_129267


namespace NUMINAMATH_CALUDE_difference_between_x_and_y_l1292_129299

theorem difference_between_x_and_y : 
  ∀ x y : ℤ, x = 10 ∧ y = 5 → x - y = 5 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_x_and_y_l1292_129299


namespace NUMINAMATH_CALUDE_total_wall_length_l1292_129218

/-- Represents the daily work rate of a bricklayer in meters per day -/
def daily_rate : ℕ := 8

/-- Represents the number of working days -/
def working_days : ℕ := 15

/-- Theorem: The total length of wall laid by a bricklayer in 15 days -/
theorem total_wall_length : daily_rate * working_days = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_wall_length_l1292_129218


namespace NUMINAMATH_CALUDE_compare_absolute_values_l1292_129206

theorem compare_absolute_values (m n : ℝ) 
  (h1 : m * n < 0) 
  (h2 : m + n < 0) 
  (h3 : n > 0) : 
  |m| > |n| := by
  sorry

end NUMINAMATH_CALUDE_compare_absolute_values_l1292_129206


namespace NUMINAMATH_CALUDE_exponent_division_equality_l1292_129222

theorem exponent_division_equality (a : ℝ) : a^6 / (-a)^2 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_equality_l1292_129222


namespace NUMINAMATH_CALUDE_negation_of_implication_conjunction_implies_disjunction_disjunction_not_implies_conjunction_negation_of_universal_even_function_condition_l1292_129258

-- Define the propositions p and q
variable (p q : Prop)

-- Define the function f
variable (a b c : ℝ)
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

-- 1. Negation of implication
theorem negation_of_implication : ¬(p → q) ↔ (p ∧ ¬q) := by sorry

-- 2. Relationship between conjunction and disjunction
theorem conjunction_implies_disjunction : (p ∧ q) → (p ∨ q) := by sorry

theorem disjunction_not_implies_conjunction : ¬((p ∨ q) → (p ∧ q)) := by sorry

-- 3. Negation of universal quantifier
theorem negation_of_universal : 
  ¬(∀ x : ℝ, x > 2 → x^2 - 2*x > 0) ↔ (∃ x : ℝ, x > 2 ∧ x^2 - 2*x ≤ 0) := by sorry

-- 4. Even function condition
theorem even_function_condition : 
  (∀ x : ℝ, f x = f (-x)) → b = 0 := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_conjunction_implies_disjunction_disjunction_not_implies_conjunction_negation_of_universal_even_function_condition_l1292_129258


namespace NUMINAMATH_CALUDE_gcd_of_quadratic_and_linear_l1292_129243

theorem gcd_of_quadratic_and_linear (b : ℤ) (h : ∃ k : ℤ, b = 1836 * k) :
  Int.gcd (b^2 + 11*b + 28) (b + 6) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_quadratic_and_linear_l1292_129243


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1292_129272

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 8 / 15) 
  (h2 : x - y = 1 / 45) : 
  x^2 - y^2 = 8 / 675 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1292_129272


namespace NUMINAMATH_CALUDE_max_min_values_l1292_129241

theorem max_min_values (x y : ℝ) (h : |5*x + y| + |5*x - y| = 20) :
  (∃ (a b : ℝ), a^2 - a*b + b^2 = 124 ∧ 
   ∀ (c d : ℝ), |5*c + d| + |5*c - d| = 20 → c^2 - c*d + d^2 ≤ 124) ∧
  (∃ (a b : ℝ), a^2 - a*b + b^2 = 4 ∧ 
   ∀ (c d : ℝ), |5*c + d| + |5*c - d| = 20 → c^2 - c*d + d^2 ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_max_min_values_l1292_129241


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l1292_129200

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 5 / 4)
  (hy : y / z = 4 / 3)
  (hz : z / x = 1 / 8) :
  w / y = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l1292_129200


namespace NUMINAMATH_CALUDE_parabola_point_range_l1292_129227

/-- Parabola type representing y² = 8x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → Prop

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : p.equation x y

/-- Circle type -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a circle intersects a line -/
def circle_intersects_line (c : Circle) (l : ℝ → ℝ → Prop) : Prop :=
  ∃ x y, l x y ∧ ((x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2)

/-- Main theorem -/
theorem parabola_point_range (p : Parabola) (m : PointOnParabola p) :
  let c : Circle := { center := p.focus, radius := Real.sqrt ((m.x - p.focus.1)^2 + (m.y - p.focus.2)^2) }
  circle_intersects_line c p.directrix → m.x > 2 := by
  sorry


end NUMINAMATH_CALUDE_parabola_point_range_l1292_129227


namespace NUMINAMATH_CALUDE_tangent_segment_length_l1292_129297

/-- Given three circles where two touch externally and a common tangent, 
    calculate the length of the tangent segment within the third circle. -/
theorem tangent_segment_length 
  (r₁ r₂ r₃ : ℝ) 
  (h₁ : r₁ = 3) 
  (h₂ : r₂ = 4) 
  (h₃ : r₃ = 5) 
  (h_touch : r₁ + r₂ = 7) : 
  ∃ (y : ℝ), y = (40 * Real.sqrt 3) / 7 ∧ 
  y = 2 * Real.sqrt (r₃^2 - ((r₂ - r₁)^2 / (4 * (r₁ + r₂)^2)) * r₃^2) := by
  sorry


end NUMINAMATH_CALUDE_tangent_segment_length_l1292_129297


namespace NUMINAMATH_CALUDE_expression_value_l1292_129214

theorem expression_value : 
  let x : ℝ := 4
  (x^2 - 2*x - 15) / (x - 5) = 7 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1292_129214


namespace NUMINAMATH_CALUDE_dad_borrowed_quarters_l1292_129246

/-- The number of quarters borrowed by Sara's dad -/
def quarters_borrowed (initial_quarters current_quarters : ℕ) : ℕ :=
  initial_quarters - current_quarters

/-- Proof that Sara's dad borrowed 271 quarters -/
theorem dad_borrowed_quarters : quarters_borrowed 783 512 = 271 := by
  sorry

end NUMINAMATH_CALUDE_dad_borrowed_quarters_l1292_129246


namespace NUMINAMATH_CALUDE_expression_evaluation_l1292_129257

theorem expression_evaluation (a b c : ℝ) (ha : a = 3) (hb : b = 2) (hc : c = 5) :
  2 * ((a^2 + b)^2 - (a^2 - b)^2) * c^2 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1292_129257


namespace NUMINAMATH_CALUDE_isosceles_triangle_with_circles_perimeter_l1292_129266

/-- Represents a triangle with circles inside --/
structure TriangleWithCircles where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  circle_radius : ℝ

/-- Calculates the perimeter of a triangle with circles inside --/
def perimeter_with_circles (t : TriangleWithCircles) : ℝ :=
  t.side1 + t.side2 + t.base - 4 * t.circle_radius

/-- Theorem: The perimeter of the specified isosceles triangle with circles is 24 --/
theorem isosceles_triangle_with_circles_perimeter :
  let t : TriangleWithCircles := {
    side1 := 12,
    side2 := 12,
    base := 8,
    circle_radius := 2
  }
  perimeter_with_circles t = 24 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_with_circles_perimeter_l1292_129266


namespace NUMINAMATH_CALUDE_gaussland_olympics_l1292_129254

theorem gaussland_olympics (total_students : ℕ) (events_per_student : ℕ) (students_per_event : ℕ) (total_coaches : ℕ) 
  (h1 : total_students = 480)
  (h2 : events_per_student = 4)
  (h3 : students_per_event = 20)
  (h4 : total_coaches = 16)
  : (total_students * events_per_student) / (students_per_event * total_coaches) = 6 := by
  sorry

#check gaussland_olympics

end NUMINAMATH_CALUDE_gaussland_olympics_l1292_129254


namespace NUMINAMATH_CALUDE_carl_removed_heads_probability_l1292_129275

/-- Represents the state of a coin (Heads or Tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents the configuration of three coins on the table -/
def CoinConfiguration := (CoinState × CoinState × CoinState)

/-- The initial configuration with Alice's coin -/
def initialConfig : CoinConfiguration := (CoinState.Heads, CoinState.Heads, CoinState.Heads)

/-- The set of all possible configurations after Bill flips two coins -/
def allConfigurations : Set CoinConfiguration := {
  (CoinState.Heads, CoinState.Heads, CoinState.Heads),
  (CoinState.Heads, CoinState.Heads, CoinState.Tails),
  (CoinState.Heads, CoinState.Tails, CoinState.Heads),
  (CoinState.Heads, CoinState.Tails, CoinState.Tails)
}

/-- The set of configurations that result in two heads showing after Carl removes a coin -/
def twoHeadsConfigurations : Set CoinConfiguration := {
  (CoinState.Heads, CoinState.Heads, CoinState.Heads),
  (CoinState.Heads, CoinState.Heads, CoinState.Tails),
  (CoinState.Heads, CoinState.Tails, CoinState.Heads)
}

/-- The probability of Carl removing a heads coin given that two heads are showing -/
def probHeadsRemoved : ℚ := 3 / 5

theorem carl_removed_heads_probability :
  probHeadsRemoved = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_carl_removed_heads_probability_l1292_129275


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1292_129265

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  IsGeometricSequence a →
  (a 1 + a 2 = 1) →
  (a 3 + a 4 = 9) →
  (a 4 + a 5 = 27 ∨ a 4 + a 5 = -27) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1292_129265


namespace NUMINAMATH_CALUDE_museum_ticket_cost_l1292_129217

/-- The cost of tickets at a museum --/
theorem museum_ticket_cost (adult_price : ℝ) : 
  (7 * adult_price + 5 * (adult_price / 2) = 35) →
  (10 * adult_price + 8 * (adult_price / 2) = 51.58) := by
  sorry

end NUMINAMATH_CALUDE_museum_ticket_cost_l1292_129217


namespace NUMINAMATH_CALUDE_distribute_6_balls_4_boxes_l1292_129251

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 9 ways to distribute 6 indistinguishable balls into 4 indistinguishable boxes -/
theorem distribute_6_balls_4_boxes : distribute_balls 6 4 = 9 := by sorry

end NUMINAMATH_CALUDE_distribute_6_balls_4_boxes_l1292_129251


namespace NUMINAMATH_CALUDE_solutions_of_f_of_f_eq_x_l1292_129219

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 5*x + 1

-- State the theorem
theorem solutions_of_f_of_f_eq_x :
  ∀ x : ℝ, f (f x) = x ↔ (x = -2 - Real.sqrt 3 ∨ x = -2 + Real.sqrt 3 ∨ x = -3 - Real.sqrt 2 ∨ x = -3 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_solutions_of_f_of_f_eq_x_l1292_129219


namespace NUMINAMATH_CALUDE_combined_figure_area_l1292_129278

/-- The area of a figure consisting of a twelve-sided polygon and a rhombus -/
theorem combined_figure_area (polygon_area : ℝ) (rhombus_diagonal1 : ℝ) (rhombus_diagonal2 : ℝ) :
  polygon_area = 13 →
  rhombus_diagonal1 = 2 →
  rhombus_diagonal2 = 1 →
  polygon_area + (rhombus_diagonal1 * rhombus_diagonal2) / 2 = 14 :=
by sorry

end NUMINAMATH_CALUDE_combined_figure_area_l1292_129278


namespace NUMINAMATH_CALUDE_wang_speed_inequality_l1292_129208

theorem wang_speed_inequality (a b v : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a < b) 
  (hv : v = 2 * a * b / (a + b)) : a < v ∧ v < Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_wang_speed_inequality_l1292_129208


namespace NUMINAMATH_CALUDE_remaining_laps_is_58_l1292_129204

/-- Represents the number of laps swum on each day --/
structure DailyLaps where
  friday : Nat
  saturday : Nat
  sundayMorning : Nat

/-- Calculates the remaining laps after Sunday morning --/
def remainingLaps (totalRequired : Nat) (daily : DailyLaps) : Nat :=
  totalRequired - (daily.friday + daily.saturday + daily.sundayMorning)

/-- Theorem stating that the remaining laps after Sunday morning is 58 --/
theorem remaining_laps_is_58 (totalRequired : Nat) (daily : DailyLaps) :
  totalRequired = 198 →
  daily.friday = 63 →
  daily.saturday = 62 →
  daily.sundayMorning = 15 →
  remainingLaps totalRequired daily = 58 := by
  sorry

#eval remainingLaps 198 { friday := 63, saturday := 62, sundayMorning := 15 }

end NUMINAMATH_CALUDE_remaining_laps_is_58_l1292_129204


namespace NUMINAMATH_CALUDE_max_servings_jordan_l1292_129250

/-- Represents the recipe for hot chocolate -/
structure Recipe where
  servings : ℚ
  chocolate : ℚ
  sugar : ℚ
  water : ℚ
  milk : ℚ

/-- Represents the available ingredients -/
structure Ingredients where
  chocolate : ℚ
  sugar : ℚ
  milk : ℚ

/-- Calculates the maximum number of servings that can be made -/
def maxServings (recipe : Recipe) (ingredients : Ingredients) : ℚ :=
  min (ingredients.chocolate / recipe.chocolate * recipe.servings)
      (min (ingredients.sugar / recipe.sugar * recipe.servings)
           (ingredients.milk / recipe.milk * recipe.servings))

theorem max_servings_jordan :
  let recipe : Recipe := ⟨5, 2, 1/4, 1, 4⟩
  let ingredients : Ingredients := ⟨5, 2, 7⟩
  maxServings recipe ingredients = 35/4 := by
  sorry

end NUMINAMATH_CALUDE_max_servings_jordan_l1292_129250


namespace NUMINAMATH_CALUDE_water_bottle_drinking_time_l1292_129289

/-- Proves that drinking a 2-liter bottle of water with 40 ml sips every 5 minutes takes 250 minutes -/
theorem water_bottle_drinking_time :
  let bottle_capacity_liters : ℝ := 2
  let ml_per_liter : ℝ := 1000
  let sip_volume_ml : ℝ := 40
  let minutes_per_sip : ℝ := 5
  
  bottle_capacity_liters * ml_per_liter / sip_volume_ml * minutes_per_sip = 250 := by
  sorry


end NUMINAMATH_CALUDE_water_bottle_drinking_time_l1292_129289


namespace NUMINAMATH_CALUDE_bike_five_times_a_week_l1292_129276

/-- Given Onur's daily biking distance, Hanil's additional distance, and their total weekly distance,
    calculate the number of days they bike per week. -/
def biking_days_per_week (onur_daily : ℕ) (hanil_additional : ℕ) (total_weekly : ℕ) : ℕ :=
  total_weekly / (onur_daily + (onur_daily + hanil_additional))

/-- Theorem stating that under the given conditions, Onur and Hanil bike 5 times a week. -/
theorem bike_five_times_a_week :
  biking_days_per_week 250 40 2700 = 5 := by
  sorry

end NUMINAMATH_CALUDE_bike_five_times_a_week_l1292_129276


namespace NUMINAMATH_CALUDE_order_of_numbers_l1292_129249

theorem order_of_numbers : (2 : ℝ)^24 < 10^8 ∧ 10^8 < 5^12 := by sorry

end NUMINAMATH_CALUDE_order_of_numbers_l1292_129249


namespace NUMINAMATH_CALUDE_butter_mixture_profit_percentage_l1292_129270

/-- Calculates the profit percentage for a mixture of butter sold at a certain price -/
theorem butter_mixture_profit_percentage
  (weight1 : ℝ) (price1 : ℝ) (weight2 : ℝ) (price2 : ℝ) (selling_price : ℝ)
  (h1 : weight1 = 34)
  (h2 : price1 = 150)
  (h3 : weight2 = 36)
  (h4 : price2 = 125)
  (h5 : selling_price = 192) :
  let total_cost := weight1 * price1 + weight2 * price2
  let total_weight := weight1 + weight2
  let cost_price_per_kg := total_cost / total_weight
  let profit_percentage := (selling_price - cost_price_per_kg) / cost_price_per_kg * 100
  ∃ ε > 0, abs (profit_percentage - 40) < ε :=
by sorry


end NUMINAMATH_CALUDE_butter_mixture_profit_percentage_l1292_129270


namespace NUMINAMATH_CALUDE_polyhedron_vertices_l1292_129247

/-- A polyhedron is a three-dimensional geometric object with flat polygonal faces, straight edges and sharp corners or vertices. -/
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ

/-- Euler's formula for polyhedra states that V - E + F = 2, where V is the number of vertices,
    E is the number of edges, and F is the number of faces. -/
axiom eulers_formula (p : Polyhedron) : p.vertices - p.edges + p.faces = 2

/-- The theorem states that a polyhedron with 21 edges and 9 faces has 14 vertices. -/
theorem polyhedron_vertices (p : Polyhedron) (h1 : p.edges = 21) (h2 : p.faces = 9) : 
  p.vertices = 14 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_vertices_l1292_129247


namespace NUMINAMATH_CALUDE_donna_bananas_l1292_129202

def total_bananas : ℕ := 350
def lydia_bananas : ℕ := 90
def dawn_extra_bananas : ℕ := 70

theorem donna_bananas :
  total_bananas - (lydia_bananas + (lydia_bananas + dawn_extra_bananas)) = 100 :=
by sorry

end NUMINAMATH_CALUDE_donna_bananas_l1292_129202


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1292_129223

theorem perfect_square_condition (x : ℝ) :
  (∃ a : ℤ, 4 * x^5 - 7 = a^2) ∧ 
  (∃ b : ℤ, 4 * x^13 - 7 = b^2) → 
  x = 2 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1292_129223


namespace NUMINAMATH_CALUDE_tadpoles_kept_l1292_129268

theorem tadpoles_kept (total : ℕ) (release_percentage : ℚ) (kept : ℕ) : 
  total = 180 → 
  release_percentage = 75 / 100 → 
  kept = total - (release_percentage * total).floor → 
  kept = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_tadpoles_kept_l1292_129268


namespace NUMINAMATH_CALUDE_businessmen_neither_coffee_nor_tea_l1292_129224

theorem businessmen_neither_coffee_nor_tea
  (total : ℕ)
  (coffee : ℕ)
  (tea : ℕ)
  (both : ℕ)
  (h1 : total = 30)
  (h2 : coffee = 15)
  (h3 : tea = 12)
  (h4 : both = 6) :
  total - (coffee + tea - both) = 9 :=
by sorry

end NUMINAMATH_CALUDE_businessmen_neither_coffee_nor_tea_l1292_129224


namespace NUMINAMATH_CALUDE_larger_number_proof_l1292_129231

theorem larger_number_proof (L S : ℕ) (h1 : L - S = 1325) (h2 : L = 5 * S + 5) : L = 1655 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1292_129231


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l1292_129287

theorem quadratic_roots_range (θ : Real) (α β : Complex) : 
  (∃ x : Complex, x^2 + 2*(Real.cos θ + 1)*x + (Real.cos θ)^2 = 0 ↔ x = α ∨ x = β) →
  Complex.abs (α - β) ≤ 2 * Real.sqrt 2 →
  ∃ k : ℤ, (θ ∈ Set.Icc (2*k*Real.pi + Real.pi/3) (2*k*Real.pi + 2*Real.pi/3)) ∨
           (θ ∈ Set.Icc (2*k*Real.pi + 4*Real.pi/3) (2*k*Real.pi + 5*Real.pi/3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l1292_129287


namespace NUMINAMATH_CALUDE_vitamin_shop_lcm_l1292_129291

theorem vitamin_shop_lcm : ∃ n : ℕ, n > 0 ∧ n % 7 = 0 ∧ n % 17 = 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 7 = 0 ∧ m % 17 = 0) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_vitamin_shop_lcm_l1292_129291


namespace NUMINAMATH_CALUDE_triangular_number_difference_l1292_129292

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The difference between the 2010th and 2008th triangular numbers is 4019 -/
theorem triangular_number_difference : 
  triangular_number 2010 - triangular_number 2008 = 4019 := by
  sorry

end NUMINAMATH_CALUDE_triangular_number_difference_l1292_129292


namespace NUMINAMATH_CALUDE_inequality_proof_l1292_129284

theorem inequality_proof (a b c A B C u v : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hu : 0 < u) (hv : 0 < v)
  (h1 : a * u^2 - b * u + c ≤ 0)
  (h2 : A * v^2 - B * v + C ≤ 0) :
  (a * u + A * v) * (c / u + C / v) ≤ ((b + B) / 2)^2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1292_129284


namespace NUMINAMATH_CALUDE_system_solution_exists_l1292_129256

theorem system_solution_exists : ∃ (x y : ℝ), 
  0 ≤ x ∧ x ≤ 6 ∧ 
  0 ≤ y ∧ y ≤ 4 ∧ 
  x + 2 * Real.sqrt y = 6 ∧ 
  Real.sqrt x + y = 4 ∧ 
  abs (x - 2.985) < 0.001 ∧ 
  abs (y - 2.272) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_exists_l1292_129256


namespace NUMINAMATH_CALUDE_new_water_height_after_cube_submersion_l1292_129232

/-- Calculates the new water height in a fish tank after submerging a cube -/
theorem new_water_height_after_cube_submersion
  (tank_width : ℝ)
  (tank_length : ℝ)
  (initial_height : ℝ)
  (cube_edge : ℝ)
  (h_width : tank_width = 50)
  (h_length : tank_length = 16)
  (h_initial_height : initial_height = 15)
  (h_cube_edge : cube_edge = 10) :
  let tank_area := tank_width * tank_length
  let cube_volume := cube_edge ^ 3
  let height_increase := cube_volume / tank_area
  let new_height := initial_height + height_increase
  new_height = 16.25 := by sorry

end NUMINAMATH_CALUDE_new_water_height_after_cube_submersion_l1292_129232


namespace NUMINAMATH_CALUDE_remaining_pages_to_read_l1292_129294

/-- Given a book where 83 pages represent 1/3 of the total, 
    the number of remaining pages to read is 166. -/
theorem remaining_pages_to_read (total_pages : ℕ) 
  (h1 : 83 = total_pages / 3) : total_pages - 83 = 166 := by
  sorry

end NUMINAMATH_CALUDE_remaining_pages_to_read_l1292_129294


namespace NUMINAMATH_CALUDE_A_intersect_B_l1292_129295

def A : Set ℤ := {1, 2, 3, 4}
def B : Set ℤ := {x | 2 ≤ x ∧ x ≤ 3}

theorem A_intersect_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l1292_129295


namespace NUMINAMATH_CALUDE_x_squared_less_than_abs_x_plus_two_l1292_129277

theorem x_squared_less_than_abs_x_plus_two (x : ℝ) :
  x^2 < |x| + 2 ↔ -2 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_x_squared_less_than_abs_x_plus_two_l1292_129277


namespace NUMINAMATH_CALUDE_smallest_x_with_remainders_l1292_129235

theorem smallest_x_with_remainders : ∃! x : ℕ+, 
  (x : ℤ) % 3 = 2 ∧ 
  (x : ℤ) % 4 = 3 ∧ 
  (x : ℤ) % 5 = 4 ∧
  ∀ y : ℕ+, 
    (y : ℤ) % 3 = 2 → 
    (y : ℤ) % 4 = 3 → 
    (y : ℤ) % 5 = 4 → 
    x ≤ y :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_x_with_remainders_l1292_129235


namespace NUMINAMATH_CALUDE_rectangle_area_l1292_129262

theorem rectangle_area (L W : ℝ) (h1 : 2 * L + 4 * W = 34) (h2 : 4 * L + 2 * W = 38) :
  L * W = 35 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1292_129262


namespace NUMINAMATH_CALUDE_power_inequality_l1292_129281

theorem power_inequality (x t : ℝ) (hx : x ≥ 3) :
  (0 < t ∧ t < 1 → x^t - (x-1)^t < (x-2)^t - (x-3)^t) ∧
  (t > 1 → x^t - (x-1)^t > (x-2)^t - (x-3)^t) := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l1292_129281
