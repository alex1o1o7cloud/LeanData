import Mathlib

namespace valid_schedules_count_l2823_282305

/-- The number of employees and days -/
def n : ℕ := 7

/-- Calculate the number of valid schedules -/
def validSchedules : ℕ :=
  n.factorial - 2 * (n - 1).factorial

/-- Theorem stating the number of valid schedules -/
theorem valid_schedules_count :
  validSchedules = 3600 := by sorry

end valid_schedules_count_l2823_282305


namespace min_value_of_complex_expression_l2823_282348

theorem min_value_of_complex_expression (Z : ℂ) (h : Complex.abs Z = 1) :
  ∃ (min_val : ℝ), min_val = 0 ∧ ∀ (W : ℂ), Complex.abs W = 1 → Complex.abs (W^2 - 2*W + 1) ≥ min_val :=
sorry

end min_value_of_complex_expression_l2823_282348


namespace both_selected_probability_l2823_282394

theorem both_selected_probability (p_ram p_ravi : ℚ) 
  (h_ram : p_ram = 5 / 7)
  (h_ravi : p_ravi = 1 / 5) :
  p_ram * p_ravi = 1 / 7 := by
  sorry

end both_selected_probability_l2823_282394


namespace chocolate_milk_probability_l2823_282363

theorem chocolate_milk_probability (n : ℕ) (k : ℕ) (p : ℚ) :
  n = 6 →
  k = 5 →
  p = 2/3 →
  Nat.choose n k * p^k * (1 - p)^(n - k) = 64/243 :=
by sorry

end chocolate_milk_probability_l2823_282363


namespace power_sum_equals_six_l2823_282315

theorem power_sum_equals_six (a x : ℝ) (h : a^x - a^(-x) = 2) : 
  a^(2*x) + a^(-2*x) = 6 := by
sorry

end power_sum_equals_six_l2823_282315


namespace kim_pizza_purchase_l2823_282396

/-- Given that Kim buys pizzas where each pizza has 12 slices, 
    the total cost is $72, and 5 slices cost $10, 
    prove that Kim bought 3 pizzas. -/
theorem kim_pizza_purchase : 
  ∀ (slices_per_pizza : ℕ) (total_cost : ℚ) (five_slice_cost : ℚ),
    slices_per_pizza = 12 →
    total_cost = 72 →
    five_slice_cost = 10 →
    (total_cost / (slices_per_pizza * (five_slice_cost / 5))) = 3 := by
  sorry

#check kim_pizza_purchase

end kim_pizza_purchase_l2823_282396


namespace priyas_age_l2823_282321

theorem priyas_age (P F : ℕ) : 
  F = P + 31 →
  (P + 8) + (F + 8) = 69 →
  P = 11 :=
by sorry

end priyas_age_l2823_282321


namespace power_sum_equality_l2823_282367

theorem power_sum_equality : (-1)^51 + 3^(2^3 + 5^2 - 7^2) = -1 + 1 / 43046721 := by
  sorry

end power_sum_equality_l2823_282367


namespace triangle_similarity_l2823_282355

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents an excircle of a triangle -/
structure Excircle where
  center : Point
  radius : ℝ

/-- Defines if a triangle is acute and scalene -/
def isAcuteScalene (t : Triangle) : Prop := sorry

/-- Defines the C-excircle of a triangle -/
def cExcircle (t : Triangle) : Excircle := sorry

/-- Defines the B-excircle of a triangle -/
def bExcircle (t : Triangle) : Excircle := sorry

/-- Defines the intersection point of two lines -/
def lineIntersection (p1 p2 p3 p4 : Point) : Point := sorry

/-- Defines a point symmetric to another point with respect to a third point -/
def symmetricPoint (p center : Point) : Point := sorry

/-- Checks if two triangles are similar -/
def areSimilar (t1 t2 : Triangle) : Prop := sorry

/-- Main theorem -/
theorem triangle_similarity (t : Triangle) (h : isAcuteScalene t) :
  let c_ex := cExcircle t
  let b_ex := bExcircle t
  let M := sorry -- Point where C-excircle is tangent to AB
  let N := sorry -- Point where C-excircle is tangent to extension of BC
  let P := sorry -- Point where B-excircle is tangent to AC
  let Q := sorry -- Point where B-excircle is tangent to extension of BC
  let A1 := lineIntersection M N P Q
  let A2 := symmetricPoint t.A A1
  let B1 := sorry -- Defined analogously to A1
  let B2 := symmetricPoint t.B B1
  let C1 := sorry -- Defined analogously to A1
  let C2 := symmetricPoint t.C C1
  let t2 : Triangle := ⟨A2, B2, C2⟩
  areSimilar t t2 :=
by sorry

end triangle_similarity_l2823_282355


namespace total_cost_calculation_l2823_282309

/-- The cost of mangos per kg -/
def mango_cost : ℝ := sorry

/-- The cost of rice per kg -/
def rice_cost : ℝ := sorry

/-- The cost of flour per kg -/
def flour_cost : ℝ := 22

theorem total_cost_calculation : 
  (10 * mango_cost = 24 * rice_cost) → 
  (flour_cost = 2 * rice_cost) → 
  (4 * mango_cost + 3 * rice_cost + 5 * flour_cost = 248.6) := by
  sorry

end total_cost_calculation_l2823_282309


namespace sin_alpha_value_l2823_282311

theorem sin_alpha_value (α : Real) (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : 4 * (Real.tan α)^2 + Real.tan α - 3 = 0) : Real.sin α = 3/5 := by
  sorry

end sin_alpha_value_l2823_282311


namespace square_plus_n_eq_square_plus_k_implies_m_le_n_l2823_282347

theorem square_plus_n_eq_square_plus_k_implies_m_le_n
  (k m n : ℕ+) 
  (h : m^2 + n = k^2 + k) : 
  m ≤ n := by
sorry

end square_plus_n_eq_square_plus_k_implies_m_le_n_l2823_282347


namespace circle_equation_and_extrema_l2823_282337

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line x + y + 5 = 0
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 + 5 = 0}

theorem circle_equation_and_extrema 
  (C : ℝ × ℝ) 
  (h1 : C ∈ Line) 
  (h2 : (0, 2) ∈ Circle C ((0 - C.1)^2 + (2 - C.2)^2).sqrt) 
  (h3 : (1, 1) ∈ Circle C ((1 - C.1)^2 + (1 - C.2)^2).sqrt) :
  (∃ (r : ℝ), Circle C r = {p : ℝ × ℝ | (p.1 + 3)^2 + (p.2 + 2)^2 = 25}) ∧ 
  (∀ (P : ℝ × ℝ), P ∈ Circle C ((0 - C.1)^2 + (2 - C.2)^2).sqrt → 
    3 * P.1 - 4 * P.2 ≤ 24) ∧
  (∀ (P : ℝ × ℝ), P ∈ Circle C ((0 - C.1)^2 + (2 - C.2)^2).sqrt → 
    3 * P.1 - 4 * P.2 ≥ -26) :=
sorry

end circle_equation_and_extrema_l2823_282337


namespace triangle_side_length_l2823_282393

theorem triangle_side_length (side2 side3 perimeter : ℝ) 
  (h1 : side2 = 10)
  (h2 : side3 = 15)
  (h3 : perimeter = 32) :
  ∃ side1 : ℝ, side1 + side2 + side3 = perimeter ∧ side1 = 7 := by
  sorry

end triangle_side_length_l2823_282393


namespace quadratic_equation_roots_l2823_282333

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + m = 0 ∧ x = -1) → 
  (∃ y : ℝ, y^2 - 4*y + m = 0 ∧ y = 5) :=
by sorry

end quadratic_equation_roots_l2823_282333


namespace max_reciprocal_sum_l2823_282364

theorem max_reciprocal_sum (x y a b : ℝ) 
  (ha : a > 1) (hb : b > 1) 
  (hax : a^x = 6) (hby : b^y = 6) 
  (hab : a + b = 2 * Real.sqrt 6) : 
  (∀ x' y' a' b' : ℝ, 
    a' > 1 → b' > 1 → 
    a'^x' = 6 → b'^y' = 6 → 
    a' + b' = 2 * Real.sqrt 6 → 
    1/x + 1/y ≥ 1/x' + 1/y') ∧ 
  (∃ x₀ y₀ a₀ b₀ : ℝ, 
    a₀ > 1 ∧ b₀ > 1 ∧ 
    a₀^x₀ = 6 ∧ b₀^y₀ = 6 ∧ 
    a₀ + b₀ = 2 * Real.sqrt 6 ∧ 
    1/x₀ + 1/y₀ = 1) :=
by sorry

end max_reciprocal_sum_l2823_282364


namespace marble_probability_difference_l2823_282359

/-- The number of red marbles in the box -/
def red_marbles : ℕ := 501

/-- The number of black marbles in the box -/
def black_marbles : ℕ := 1501

/-- The number of blue marbles in the box -/
def blue_marbles : ℕ := 1000

/-- The total number of marbles in the box -/
def total_marbles : ℕ := red_marbles + black_marbles + blue_marbles

/-- The probability of drawing two marbles of the same color -/
def Ps : ℚ := (red_marbles.choose 2 + black_marbles.choose 2 + blue_marbles.choose 2) / total_marbles.choose 2

/-- The probability of drawing two marbles of different colors -/
def Pd : ℚ := 1 - Ps

/-- The theorem stating that the absolute difference between Ps and Pd is 2/9 -/
theorem marble_probability_difference : |Ps - Pd| = 2/9 := by
  sorry

end marble_probability_difference_l2823_282359


namespace ab_value_is_32_l2823_282374

def is_distinct (a b c d e f : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f

def is_valid_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 6

theorem ab_value_is_32 :
  ∃ (a b c d e f : ℕ),
    is_distinct a b c d e f ∧
    is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧
    is_valid_digit d ∧ is_valid_digit e ∧ is_valid_digit f ∧
    (10 * a + b) * ((10 * c + d) - e) + f = 2021 ∧
    10 * a + b = 32 :=
by sorry

end ab_value_is_32_l2823_282374


namespace quadratic_equation_coefficients_l2823_282304

/-- Given a quadratic equation x^2 + 2 = 3x, prove that the coefficient of x^2 is 1 and the coefficient of x is -3. -/
theorem quadratic_equation_coefficients :
  let eq : ℝ → Prop := λ x => x^2 + 2 = 3*x
  ∃ a b c : ℝ, (∀ x, eq x ↔ a*x^2 + b*x + c = 0) ∧ a = 1 ∧ b = -3 := by
  sorry

end quadratic_equation_coefficients_l2823_282304


namespace arithmetic_sequence_a3_l2823_282383

/-- An arithmetic sequence with common difference 2 where a₂ is the geometric mean of a₁ and a₅ -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = a n + 2) ∧ a 2 ^ 2 = a 1 * a 5

theorem arithmetic_sequence_a3 (a : ℕ → ℝ) (h : ArithmeticSequence a) : a 3 = 5 := by
  sorry

end arithmetic_sequence_a3_l2823_282383


namespace statements_equivalence_l2823_282345

-- Define the propositions
variable (S : Prop) -- Saturn is visible from Earth tonight
variable (M : Prop) -- Mars is visible

-- Define the statements
def statement1 : Prop := S → ¬M
def statement2 : Prop := M → ¬S
def statement3 : Prop := ¬S ∨ ¬M

-- Theorem stating the equivalence of the statements
theorem statements_equivalence : statement1 S M ↔ statement2 S M ∧ statement3 S M := by
  sorry

end statements_equivalence_l2823_282345


namespace ellipse_problem_l2823_282378

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define points A, B, and P
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def P : ℝ × ℝ := (0, 1)

-- Define line AB
def lineAB (x y : ℝ) : Prop := sorry

-- Define line y = -x + 2
def intersectLine (x y : ℝ) : Prop := y = -x + 2

-- Define points C and D
def C : ℝ × ℝ := sorry
def D : ℝ × ℝ := sorry

-- Define slopes
def slopePA : ℝ := sorry
def slopeAB : ℝ := sorry
def slopePB : ℝ := sorry

-- Theorem statement
theorem ellipse_problem :
  ellipse A.1 A.2 ∧ 
  ellipse B.1 B.2 ∧ 
  A ≠ P ∧ 
  B ≠ P ∧ 
  lineAB 0 0 ∧
  intersectLine C.1 C.2 ∧
  intersectLine D.1 D.2 →
  (∃ k : ℝ, slopePA + slopePB = 2 * slopeAB) ∧
  (∃ minArea : ℝ, minArea = Real.sqrt 2 / 3 ∧ 
    ∀ area : ℝ, area ≥ minArea) :=
by sorry

end ellipse_problem_l2823_282378


namespace equivalent_coin_value_l2823_282316

theorem equivalent_coin_value : ∀ (quarter_value dime_value : ℕ),
  quarter_value = 25 →
  dime_value = 10 →
  30 * quarter_value + 20 * dime_value = 15 * quarter_value + 58 * dime_value :=
by
  sorry

end equivalent_coin_value_l2823_282316


namespace system_solution_l2823_282325

theorem system_solution (x y z : ℝ) : 
  (2 * x^2 + 3 * y + 5 = 2 * Real.sqrt (2 * z + 5)) ∧
  (2 * y^2 + 3 * z + 5 = 2 * Real.sqrt (2 * x + 5)) ∧
  (2 * z^2 + 3 * x + 5 = 2 * Real.sqrt (2 * y + 5)) →
  x = -1/2 ∧ y = -1/2 ∧ z = -1/2 := by
sorry

end system_solution_l2823_282325


namespace polynomial_irreducibility_l2823_282320

/-- A polynomial of the form x^n + 5x^(n-1) + 3 where n > 1 is irreducible over the integers -/
theorem polynomial_irreducibility (n : ℕ) (hn : n > 1) :
  Irreducible (Polynomial.monomial n 1 + Polynomial.monomial (n-1) 5 + Polynomial.C 3 : Polynomial ℤ) := by
  sorry

end polynomial_irreducibility_l2823_282320


namespace minutes_from_2222_to_midnight_l2823_282375

def minutes_until_midnight (hour : Nat) (minute : Nat) : Nat :=
  (23 - hour) * 60 + (60 - minute)

theorem minutes_from_2222_to_midnight :
  minutes_until_midnight 22 22 = 98 := by
  sorry

end minutes_from_2222_to_midnight_l2823_282375


namespace tank_length_is_six_l2823_282354

def tank_volume : ℝ := 72
def tank_width : ℝ := 4
def tank_depth : ℝ := 3

theorem tank_length_is_six :
  let length := tank_volume / (tank_width * tank_depth)
  length = 6 := by sorry

end tank_length_is_six_l2823_282354


namespace trigonometric_simplification_l2823_282322

theorem trigonometric_simplification (α : ℝ) :
  (Real.sin (6 * α) / Real.sin (2 * α)) + (Real.cos (6 * α - π) / Real.cos (2 * α)) = 2 := by
  sorry

end trigonometric_simplification_l2823_282322


namespace writing_speed_ratio_l2823_282391

/-- Jacob and Nathan's writing speeds -/
def writing_problem (jacob_speed nathan_speed : ℚ) : Prop :=
  nathan_speed = 25 ∧ 
  jacob_speed + nathan_speed = 75 ∧
  jacob_speed / nathan_speed = 2

theorem writing_speed_ratio : ∃ (jacob_speed nathan_speed : ℚ), 
  writing_problem jacob_speed nathan_speed :=
sorry

end writing_speed_ratio_l2823_282391


namespace arithmetic_sequence_tan_l2823_282389

/-- Given an arithmetic sequence {a_n} where a₁ + a₇ + a₁₃ = π, 
    prove that tan(a₂ + a₁₂) = -√3 -/
theorem arithmetic_sequence_tan (a : ℕ → ℝ) :
  (∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n) →  -- arithmetic sequence
  a 1 + a 7 + a 13 = Real.pi →                      -- given condition
  Real.tan (a 2 + a 12) = -Real.sqrt 3 :=           -- conclusion to prove
by sorry

end arithmetic_sequence_tan_l2823_282389


namespace pool_filling_time_l2823_282362

/-- Proves that filling a pool of given capacity with a specific number of hoses 
    and flow rate takes the calculated number of hours -/
theorem pool_filling_time 
  (pool_capacity : ℕ) 
  (num_hoses : ℕ) 
  (flow_rate_per_hose : ℕ) 
  (hours_to_fill : ℕ) 
  (h1 : pool_capacity = 32000)
  (h2 : num_hoses = 3)
  (h3 : flow_rate_per_hose = 4)
  (h4 : hours_to_fill = 44) : 
  pool_capacity = num_hoses * flow_rate_per_hose * 60 * hours_to_fill :=
by
  sorry

#check pool_filling_time

end pool_filling_time_l2823_282362


namespace expression_simplification_l2823_282324

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 / (x - 1) - 1 / (x + 1)) / (2 / ((x - 1)^2)) = 1 - Real.sqrt 2 := by
  sorry

end expression_simplification_l2823_282324


namespace contrapositive_equivalence_l2823_282368

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a = 0 → a * b = 0)) ↔ (a * b ≠ 0 → a ≠ 0) := by sorry

end contrapositive_equivalence_l2823_282368


namespace inequality_proof_l2823_282310

theorem inequality_proof (a b x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  x / (a * y + b * z) + y / (a * z + b * x) + z / (a * x + b * y) ≥ 3 / (a + b) := by
  sorry

end inequality_proof_l2823_282310


namespace percentage_of_students_passed_l2823_282329

/-- The percentage of students who passed an examination, given the total number of students and the number of students who failed. -/
theorem percentage_of_students_passed (total : ℕ) (failed : ℕ) (h1 : total = 840) (h2 : failed = 546) :
  (((total - failed : ℚ) / total) * 100 : ℚ) = 35 := by
  sorry

end percentage_of_students_passed_l2823_282329


namespace flower_arrangement_daisies_percentage_l2823_282377

theorem flower_arrangement_daisies_percentage
  (total_flowers : ℕ)
  (h1 : total_flowers > 0)
  (yellow_flowers : ℕ)
  (h2 : yellow_flowers = (7 * total_flowers) / 10)
  (white_flowers : ℕ)
  (h3 : white_flowers = total_flowers - yellow_flowers)
  (yellow_tulips : ℕ)
  (h4 : yellow_tulips = yellow_flowers / 2)
  (white_daisies : ℕ)
  (h5 : white_daisies = (2 * white_flowers) / 3)
  (yellow_daisies : ℕ)
  (h6 : yellow_daisies = yellow_flowers - yellow_tulips)
  (total_daisies : ℕ)
  (h7 : total_daisies = yellow_daisies + white_daisies) :
  (total_daisies : ℚ) / total_flowers = 11 / 20 :=
sorry

end flower_arrangement_daisies_percentage_l2823_282377


namespace sum_abc_values_l2823_282390

theorem sum_abc_values (a b c : ℤ) 
  (h1 : a - 2*b = 4) 
  (h2 : a*b + c^2 - 1 = 0) : 
  a + b + c = 5 ∨ a + b + c = 3 ∨ a + b + c = -1 ∨ a + b + c = -3 :=
sorry

end sum_abc_values_l2823_282390


namespace cheerleader_ratio_is_half_l2823_282356

/-- Represents the number of cheerleaders for each uniform size -/
structure CheerleaderCounts where
  total : ℕ
  size2 : ℕ
  size6 : ℕ
  size12 : ℕ

/-- The ratio of cheerleaders needing size 12 to those needing size 6 -/
def size12to6Ratio (counts : CheerleaderCounts) : ℚ :=
  counts.size12 / counts.size6

/-- Theorem stating the ratio of cheerleaders needing size 12 to those needing size 6 -/
theorem cheerleader_ratio_is_half (counts : CheerleaderCounts)
  (h_total : counts.total = 19)
  (h_size2 : counts.size2 = 4)
  (h_size6 : counts.size6 = 10)
  (h_sum : counts.total = counts.size2 + counts.size6 + counts.size12) :
  size12to6Ratio counts = 1/2 := by
  sorry

end cheerleader_ratio_is_half_l2823_282356


namespace equation_solution_l2823_282382

theorem equation_solution : ∃ x : ℝ, 45 * x = 0.4 * 900 ∧ x = 8 := by
  sorry

end equation_solution_l2823_282382


namespace shirt_tie_outfits_l2823_282341

theorem shirt_tie_outfits (shirts : ℕ) (ties : ℕ) (h1 : shirts = 8) (h2 : ties = 6) :
  shirts * ties = 48 := by
  sorry

end shirt_tie_outfits_l2823_282341


namespace simultaneous_inequalities_l2823_282332

theorem simultaneous_inequalities (x : ℝ) :
  x^2 - 12*x + 32 > 0 ∧ x^2 - 13*x + 22 < 0 → 2 < x ∧ x < 4 := by
  sorry

end simultaneous_inequalities_l2823_282332


namespace mean_of_two_numbers_l2823_282370

theorem mean_of_two_numbers (a b c : ℝ) : 
  (a + b + c + 100) / 4 = 90 →
  a = 70 →
  a ≤ b ∧ b ≤ c ∧ c ≤ 100 →
  (b + c) / 2 = 95 := by
sorry

end mean_of_two_numbers_l2823_282370


namespace max_correct_answers_l2823_282392

/-- Represents a test score --/
structure TestScore where
  total_questions : ℕ
  correct : ℕ
  incorrect : ℕ
  unanswered : ℕ
  score : ℤ

/-- Checks if a TestScore is valid according to the given conditions --/
def is_valid_score (ts : TestScore) : Prop :=
  ts.total_questions = 30 ∧
  ts.correct + ts.incorrect + ts.unanswered = ts.total_questions ∧
  ts.score = 4 * ts.correct - ts.incorrect

/-- Theorem stating the maximum number of correct answers --/
theorem max_correct_answers (ts : TestScore) (h : is_valid_score ts) (score_70 : ts.score = 70) :
  ts.correct ≤ 20 ∧ ∃ (ts' : TestScore), is_valid_score ts' ∧ ts'.score = 70 ∧ ts'.correct = 20 :=
sorry

end max_correct_answers_l2823_282392


namespace zeros_before_first_nonzero_of_fraction_l2823_282399

/-- The number of zeros between the decimal point and the first non-zero digit
    in the decimal representation of 7/8000 -/
def zeros_before_first_nonzero : ℕ :=
  3

/-- The fraction we're considering -/
def fraction : ℚ :=
  7 / 8000

theorem zeros_before_first_nonzero_of_fraction :
  zeros_before_first_nonzero = 3 :=
sorry

end zeros_before_first_nonzero_of_fraction_l2823_282399


namespace total_squat_bench_press_l2823_282388

/-- Represents the weight Tony can lift in various exercises --/
structure TonyLift where
  curl : ℝ
  military_press : ℝ
  squat : ℝ
  bench_press : ℝ

/-- Defines Tony's lifting capabilities based on the given conditions --/
def tony_lift : TonyLift where
  curl := 90
  military_press := 2 * 90
  squat := 5 * (2 * 90)
  bench_press := 1.5 * (2 * 90)

/-- Theorem stating the total weight Tony can lift in squat and bench press combined --/
theorem total_squat_bench_press (t : TonyLift) (h : t = tony_lift) : 
  t.squat + t.bench_press = 1170 := by
  sorry

end total_squat_bench_press_l2823_282388


namespace both_brothers_selected_probability_l2823_282366

theorem both_brothers_selected_probability 
  (prob_X : ℚ) 
  (prob_Y : ℚ) 
  (h1 : prob_X = 1 / 3) 
  (h2 : prob_Y = 2 / 7) : 
  prob_X * prob_Y = 2 / 21 := by
  sorry

end both_brothers_selected_probability_l2823_282366


namespace ST_length_l2823_282360

-- Define the points
variable (P Q R S T : ℝ × ℝ)

-- Define the distances
def distance (A B : ℝ × ℝ) : ℝ := sorry

-- State the conditions
axiom PQ_eq_6 : distance P Q = 6
axiom QR_eq_6 : distance Q R = 6
axiom RS_eq_6 : distance R S = 6
axiom SP_eq_6 : distance S P = 6
axiom SQ_eq_6 : distance S Q = 6
axiom PT_eq_14 : distance P T = 14
axiom RT_eq_14 : distance R T = 14

-- PQRS is a rhombus
axiom is_rhombus : distance P Q = distance Q R ∧ distance Q R = distance R S ∧ distance R S = distance S P

-- PQS and RQS are equilateral triangles
axiom PQS_equilateral : distance P Q = distance Q S ∧ distance Q S = distance S P
axiom RQS_equilateral : distance R Q = distance Q S ∧ distance Q S = distance R S

-- The theorem to prove
theorem ST_length : distance S T = 10 :=
sorry

end ST_length_l2823_282360


namespace worker_pay_calculation_l2823_282395

/-- Calculates the total pay for a worker given their regular pay rate, 
    regular hours, overtime hours, and overtime pay rate multiplier. -/
def totalPay (regularRate : ℝ) (regularHours : ℝ) (overtimeHours : ℝ) (overtimeMultiplier : ℝ) : ℝ :=
  regularRate * regularHours + regularRate * overtimeMultiplier * overtimeHours

theorem worker_pay_calculation :
  let regularRate : ℝ := 3
  let regularHours : ℝ := 40
  let overtimeHours : ℝ := 8
  let overtimeMultiplier : ℝ := 2
  totalPay regularRate regularHours overtimeHours overtimeMultiplier = 168 := by
sorry

end worker_pay_calculation_l2823_282395


namespace intersection_parameter_value_l2823_282317

/-- Given two lines that intersect at a specific x-coordinate, 
    prove the value of the parameter m in the first line equation. -/
theorem intersection_parameter_value 
  (x : ℝ) 
  (h1 : x = -7.5) 
  (h2 : ∃ y : ℝ, 3 * x - y = m ∧ -0.4 * x + y = 3) : 
  m = -22.5 := by
  sorry

end intersection_parameter_value_l2823_282317


namespace sixth_is_wednesday_l2823_282307

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculates the day of the week for a given date in a month starting with Friday -/
def dayOfWeek (date : Nat) : DayOfWeek :=
  match (date - 1) % 7 with
  | 0 => DayOfWeek.Friday
  | 1 => DayOfWeek.Saturday
  | 2 => DayOfWeek.Sunday
  | 3 => DayOfWeek.Monday
  | 4 => DayOfWeek.Tuesday
  | 5 => DayOfWeek.Wednesday
  | _ => DayOfWeek.Thursday

theorem sixth_is_wednesday 
  (h1 : ∃ (x : Nat), x + (x + 7) + (x + 14) + (x + 21) + (x + 28) = 75) 
  : dayOfWeek 6 = DayOfWeek.Wednesday := by
  sorry

end sixth_is_wednesday_l2823_282307


namespace f_domain_l2823_282308

noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (3 - Real.tan x ^ 2) + Real.sqrt (x * (Real.pi - x))

def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f x = y}

theorem f_domain :
  domain f = Set.Icc 0 (Real.pi / 3) ∪ Set.Ioc (2 * Real.pi / 3) Real.pi :=
by sorry

end f_domain_l2823_282308


namespace area_bounded_by_parabola_and_x_axis_l2823_282386

/-- The area of the figure bounded by y = 2x - x^2 and y = 0 is 4/3 square units. -/
theorem area_bounded_by_parabola_and_x_axis : 
  let f (x : ℝ) := 2 * x - x^2
  ∫ x in (0)..(2), max 0 (f x) = 4/3 := by sorry

end area_bounded_by_parabola_and_x_axis_l2823_282386


namespace modular_equation_solution_l2823_282330

theorem modular_equation_solution (x : ℤ) : 
  (10 * x + 3 ≡ 7 [ZMOD 15]) → 
  (∃ (a m : ℕ), m ≥ 2 ∧ a < m ∧ x ≡ a [ZMOD m] ∧ a + m = 27) :=
by sorry

end modular_equation_solution_l2823_282330


namespace transfer_increases_averages_l2823_282379

/-- Represents a group of students with their average grade and count -/
structure StudentGroup where
  avg_grade : ℝ
  count : ℕ

/-- Checks if transferring students increases average grades in both groups -/
def increases_averages (group_a group_b : StudentGroup) (grade1 grade2 : ℝ) : Prop :=
  let new_a := StudentGroup.mk
    ((group_a.avg_grade * group_a.count - grade1 - grade2) / (group_a.count - 2))
    (group_a.count - 2)
  let new_b := StudentGroup.mk
    ((group_b.avg_grade * group_b.count + grade1 + grade2) / (group_b.count + 2))
    (group_b.count + 2)
  new_a.avg_grade > group_a.avg_grade ∧ new_b.avg_grade > group_b.avg_grade

theorem transfer_increases_averages :
  let group_a := StudentGroup.mk 44.2 10
  let group_b := StudentGroup.mk 38.8 10
  let grade1 := 41
  let grade2 := 44
  increases_averages group_a group_b grade1 grade2 := by
  sorry

end transfer_increases_averages_l2823_282379


namespace thousandth_digit_is_one_l2823_282339

/-- The number of digits in n -/
def num_digits : ℕ := 1998

/-- The number n as a natural number -/
def n : ℕ := (10^num_digits - 1) / 9

/-- The 1000th digit after the decimal point of √n -/
def thousandth_digit_after_decimal (n : ℕ) : ℕ :=
  -- Definition placeholder, actual implementation would be complex
  sorry

/-- Theorem stating that the 1000th digit after the decimal point of √n is 1 -/
theorem thousandth_digit_is_one :
  thousandth_digit_after_decimal n = 1 := by sorry

end thousandth_digit_is_one_l2823_282339


namespace pencil_price_in_units_l2823_282397

-- Define the base price in won
def base_price : ℝ := 5000

-- Define the additional cost in won
def additional_cost : ℝ := 200

-- Define the conversion factor from won to 10,000 won units
def conversion_factor : ℝ := 10000

-- Theorem statement
theorem pencil_price_in_units (price : ℝ) : 
  price = base_price + additional_cost → 
  price / conversion_factor = 0.52 := by
sorry

end pencil_price_in_units_l2823_282397


namespace car_speed_proof_l2823_282398

/-- Proves that a car's speed is 112.5 km/h if it takes 2 seconds longer to travel 1 km compared to 120 km/h -/
theorem car_speed_proof (v : ℝ) : v > 0 → (1 / v - 1 / 120) * 3600 = 2 ↔ v = 112.5 := by sorry

end car_speed_proof_l2823_282398


namespace two_triangle_range_l2823_282365

theorem two_triangle_range (A B C : ℝ) (a b c : ℝ) :
  A = Real.pi / 3 →  -- 60 degrees in radians
  a = Real.sqrt 3 →
  b = x →
  (∃ (x : ℝ), ∀ B, 
    Real.pi / 3 < B ∧ B < 2 * Real.pi / 3 →  -- 60° < B < 120°
    Real.sin B = x / 2 →
    x > Real.sqrt 3 ∧ x < 2) :=
by sorry

end two_triangle_range_l2823_282365


namespace net_increase_is_86400_l2823_282384

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the birth rate in people per two seconds -/
def birth_rate : ℚ := 8

/-- Represents the death rate in people per two seconds -/
def death_rate : ℚ := 6

/-- Calculates the net population increase in one day -/
def net_population_increase (birth_rate death_rate : ℚ) (seconds_per_day : ℕ) : ℚ :=
  (birth_rate - death_rate) / 2 * seconds_per_day

/-- Theorem stating that the net population increase in one day is 86400 -/
theorem net_increase_is_86400 :
  net_population_increase birth_rate death_rate seconds_per_day = 86400 := by
  sorry

end net_increase_is_86400_l2823_282384


namespace spoke_forms_surface_l2823_282338

/-- Represents a spoke in a bicycle wheel -/
structure Spoke :=
  (length : ℝ)
  (angle : ℝ)

/-- Represents a rotating bicycle wheel -/
structure RotatingWheel :=
  (radius : ℝ)
  (angular_velocity : ℝ)
  (spokes : List Spoke)

/-- Represents the surface formed by rotating spokes -/
def SurfaceFormedBySpokes (wheel : RotatingWheel) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- Theorem stating that a rotating spoke forms a surface -/
theorem spoke_forms_surface (wheel : RotatingWheel) (s : Spoke) 
  (h : s ∈ wheel.spokes) : 
  ∃ (surface : Set (ℝ × ℝ × ℝ)), 
    surface = SurfaceFormedBySpokes wheel ∧ 
    (∀ t : ℝ, ∃ p : ℝ × ℝ × ℝ, p ∈ surface) :=
sorry

end spoke_forms_surface_l2823_282338


namespace unique_solution_system_l2823_282336

theorem unique_solution_system (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x^2 + x*y + y^2) * (y^2 + y*z + z^2) * (z^2 + z*x + x^2) = x*y*z ∧
  (x^4 + x^2*y^2 + y^4) * (y^4 + y^2*z^2 + z^4) * (z^4 + z^2*x^2 + x^4) = x^3*y^3*z^3 →
  x = 1/3 ∧ y = 1/3 ∧ z = 1/3 := by
sorry

end unique_solution_system_l2823_282336


namespace total_money_found_l2823_282350

-- Define the amount each person receives
def individual_share : ℝ := 32.50

-- Define the number of people sharing the money
def number_of_people : ℕ := 2

-- Theorem to prove
theorem total_money_found (even_split : ℝ → ℕ → ℝ) :
  even_split individual_share number_of_people = 65.00 :=
by sorry

end total_money_found_l2823_282350


namespace max_y_coordinate_polar_curve_l2823_282373

theorem max_y_coordinate_polar_curve (θ : Real) :
  let r := Real.sin (2 * θ)
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  |y| ≤ 4 * Real.sqrt 3 / 9 := by
  sorry

end max_y_coordinate_polar_curve_l2823_282373


namespace middle_of_five_consecutive_sum_60_l2823_282328

theorem middle_of_five_consecutive_sum_60 (a b c d e : ℕ) : 
  (a + b + c + d + e = 60) → 
  (b = a + 1) → 
  (c = b + 1) → 
  (d = c + 1) → 
  (e = d + 1) → 
  c = 12 := by
sorry

end middle_of_five_consecutive_sum_60_l2823_282328


namespace initial_price_increase_l2823_282313

theorem initial_price_increase (x : ℝ) : 
  (1 + x / 100) * 1.25 = 1.4375 → x = 15 := by
  sorry

end initial_price_increase_l2823_282313


namespace binomial_square_condition_l2823_282312

/-- If 9x^2 - 24x + a is the square of a binomial, then a = 16 -/
theorem binomial_square_condition (a : ℝ) : 
  (∃ p q : ℝ, ∀ x, 9*x^2 - 24*x + a = (p*x + q)^2) → a = 16 := by
  sorry

end binomial_square_condition_l2823_282312


namespace pipe_stack_total_l2823_282334

/-- Calculates the total number of pipes in a trapezoidal stack -/
def total_pipes (layers : ℕ) (bottom : ℕ) (top : ℕ) : ℕ :=
  (bottom + top) * layers / 2

/-- Proves that a trapezoidal stack of pipes with given parameters contains 88 pipes -/
theorem pipe_stack_total : total_pipes 11 13 3 = 88 := by
  sorry

end pipe_stack_total_l2823_282334


namespace intersection_M_N_l2823_282335

def M : Set ℤ := {-1, 0, 1}

def N : Set ℤ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_M_N : M ∩ N = {0} := by
  sorry

end intersection_M_N_l2823_282335


namespace trig_expression_equality_l2823_282358

theorem trig_expression_equality (α : Real) 
  (h : Real.tan α / (Real.tan α - 1) = -1) : 
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α) = -5/3 := by
  sorry

end trig_expression_equality_l2823_282358


namespace fifth_number_15th_row_l2823_282302

def pascal_triangle (n k : ℕ) : ℕ := Nat.choose n k

theorem fifth_number_15th_row : pascal_triangle 15 4 = 1365 := by
  sorry

end fifth_number_15th_row_l2823_282302


namespace green_pepper_weight_is_half_total_l2823_282369

/-- The weight of green peppers bought by Hannah's Vegetarian Restaurant -/
def green_pepper_weight : ℝ := 0.33333333335

/-- The total weight of peppers bought by Hannah's Vegetarian Restaurant -/
def total_pepper_weight : ℝ := 0.6666666667

/-- Theorem stating that the weight of green peppers is half the total weight -/
theorem green_pepper_weight_is_half_total :
  green_pepper_weight = total_pepper_weight / 2 := by sorry

end green_pepper_weight_is_half_total_l2823_282369


namespace base_of_equation_l2823_282387

theorem base_of_equation (e : ℕ) (h : e = 35) :
  ∃ b : ℚ, b^e * (1/4)^18 = 1/(2*(10^35)) ∧ b = 1/5 := by
  sorry

end base_of_equation_l2823_282387


namespace rhombus_area_l2823_282361

/-- The area of a rhombus with side length 3 cm and one angle measuring 45° is (9√2)/2 square cm. -/
theorem rhombus_area (side : ℝ) (angle : ℝ) :
  side = 3 →
  angle = π / 4 →
  let height : ℝ := side / Real.sqrt 2
  let area : ℝ := side * height
  area = (9 * Real.sqrt 2) / 2 := by
  sorry

end rhombus_area_l2823_282361


namespace prob_select_all_cocaptains_l2823_282326

/-- Represents a math team with a given number of students and co-captains -/
structure MathTeam where
  num_students : ℕ
  num_cocaptains : ℕ

/-- Calculates the probability of selecting all co-captains from a given team -/
def prob_select_cocaptains (team : MathTeam) : ℚ :=
  1 / (team.num_students.choose 3)

/-- The set of math teams in the area -/
def math_teams : List MathTeam := [
  { num_students := 6, num_cocaptains := 3 },
  { num_students := 7, num_cocaptains := 3 },
  { num_students := 8, num_cocaptains := 3 },
  { num_students := 9, num_cocaptains := 3 }
]

/-- Theorem stating the probability of selecting all co-captains -/
theorem prob_select_all_cocaptains : 
  (1 / (math_teams.length : ℚ)) * (math_teams.map prob_select_cocaptains).sum = 91 / 6720 := by
  sorry


end prob_select_all_cocaptains_l2823_282326


namespace milk_carton_volume_l2823_282376

theorem milk_carton_volume (surface_area : ℝ) (h : surface_area = 600) :
  let side_length := Real.sqrt (surface_area / 6)
  side_length ^ 3 = 1000 := by
sorry

end milk_carton_volume_l2823_282376


namespace roll_two_dice_prob_at_least_one_two_l2823_282346

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The probability of rolling at least one 2 with two fair 8-sided dice -/
def prob_at_least_one_two : ℚ := 15 / 64

/-- Theorem stating the probability of rolling at least one 2 with two fair 8-sided dice -/
theorem roll_two_dice_prob_at_least_one_two :
  prob_at_least_one_two = 15 / 64 := by
  sorry

#check roll_two_dice_prob_at_least_one_two

end roll_two_dice_prob_at_least_one_two_l2823_282346


namespace rectangle_product_of_b_values_l2823_282323

theorem rectangle_product_of_b_values :
  ∀ (b₁ b₂ : ℝ),
    (∀ x y : ℝ, (y = 1 ∨ y = 4 ∨ x = 2 ∨ x = b₁) → (y = 1 ∨ y = 4 ∨ x = 2 ∨ x = b₂)) →
    (abs (2 - b₁) = 2 * abs (4 - 1)) →
    (abs (2 - b₂) = 2 * abs (4 - 1)) →
    b₁ * b₂ = -32 :=
by sorry

end rectangle_product_of_b_values_l2823_282323


namespace megacorp_fine_l2823_282380

def daily_mining_revenue : ℝ := 3000000
def daily_oil_revenue : ℝ := 5000000
def monthly_expenses : ℝ := 30000000
def fine_percentage : ℝ := 0.01
def days_in_year : ℕ := 365
def months_in_year : ℕ := 12

theorem megacorp_fine :
  let daily_revenue := daily_mining_revenue + daily_oil_revenue
  let annual_revenue := daily_revenue * days_in_year
  let annual_expenses := monthly_expenses * months_in_year
  let annual_profit := annual_revenue - annual_expenses
  let fine := annual_profit * fine_percentage
  fine = 25600000 := by sorry

end megacorp_fine_l2823_282380


namespace product_equals_fraction_l2823_282340

/-- The decimal representation of the repeating decimal 0.456̅ -/
def repeating_decimal : ℚ := 152 / 333

/-- The product of the repeating decimal 0.456̅ and 7 -/
def product : ℚ := repeating_decimal * 7

/-- Theorem stating that the product of 0.456̅ and 7 is equal to 1064/333 -/
theorem product_equals_fraction : product = 1064 / 333 := by sorry

end product_equals_fraction_l2823_282340


namespace unique_valid_number_l2823_282349

/-- Represents a four-digit integer as a tuple of its digits -/
def FourDigitInt := (Fin 10 × Fin 10 × Fin 10 × Fin 10)

/-- Converts a pair of digits to a two-digit integer -/
def twoDigitInt (a b : Fin 10) : Nat := 10 * a.val + b.val

/-- Checks if three numbers form a geometric sequence -/
def isGeometricSequence (x y z : Nat) : Prop := ∃ r : ℚ, r > 1 ∧ y = r * x ∧ z = r * y

/-- Predicate for a valid four-digit integer satisfying the problem conditions -/
def isValidNumber (n : FourDigitInt) : Prop :=
  let (a, b, c, d) := n
  a ≠ 0 ∧
  isGeometricSequence (twoDigitInt a b) (twoDigitInt b c) (twoDigitInt c d)

theorem unique_valid_number :
  ∃! n : FourDigitInt, isValidNumber n :=
sorry

end unique_valid_number_l2823_282349


namespace residue_5_2023_mod_11_l2823_282385

theorem residue_5_2023_mod_11 : 5^2023 ≡ 4 [ZMOD 11] := by
  sorry

end residue_5_2023_mod_11_l2823_282385


namespace common_chord_equation_l2823_282371

/-- Given two circles with equations x^2 + y^2 - 4x = 0 and x^2 + y^2 - 4y = 0,
    the equation of the line where their common chord lies is x - y = 0. -/
theorem common_chord_equation (x y : ℝ) : 
  (x^2 + y^2 - 4*x = 0 ∧ x^2 + y^2 - 4*y = 0) → x - y = 0 :=
by sorry

end common_chord_equation_l2823_282371


namespace fourth_term_is_negative_24_l2823_282343

-- Define a geometric sequence
def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

-- Define the conditions of our specific sequence
def our_sequence (x : ℝ) (n : ℕ) : ℝ :=
  match n with
  | 1 => x
  | 2 => 3*x + 3
  | 3 => 6*x + 6
  | _ => geometric_sequence x 2 n

-- Theorem statement
theorem fourth_term_is_negative_24 : 
  ∀ x : ℝ, our_sequence x 4 = -24 := by sorry

end fourth_term_is_negative_24_l2823_282343


namespace pascal_triangle_15th_row_4th_number_l2823_282319

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem pascal_triangle_15th_row_4th_number : binomial 15 3 = 455 := by
  sorry

end pascal_triangle_15th_row_4th_number_l2823_282319


namespace win_sector_area_l2823_282301

/-- The area of a WIN sector on a circular spinner --/
theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 12) (h2 : p = 1/3) :
  p * π * r^2 = 48 * π := by
  sorry

end win_sector_area_l2823_282301


namespace smallest_bound_for_cubic_inequality_l2823_282327

theorem smallest_bound_for_cubic_inequality :
  ∃ (M : ℝ), (∀ (a b c : ℝ),
    |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M * (a^2 + b^2 + c^2)^2) ∧
  (∀ (M' : ℝ), (∀ (a b c : ℝ),
    |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M' * (a^2 + b^2 + c^2)^2) → M ≤ M') ∧
  M = (9 * Real.sqrt 2) / 32 :=
sorry

end smallest_bound_for_cubic_inequality_l2823_282327


namespace shaded_area_semicircle_pattern_shaded_area_semicircle_pattern_correct_l2823_282372

/-- The area of the shaded region in a pattern of semicircles -/
theorem shaded_area_semicircle_pattern (pattern_length : Real) (semicircle_diameter : Real) 
  (h1 : pattern_length = 2 * 12) -- 2 feet converted to inches
  (h2 : semicircle_diameter = 3) -- diameter in inches
  : Real :=
  18 * Real.pi

theorem shaded_area_semicircle_pattern_correct 
  (pattern_length : Real) (semicircle_diameter : Real) 
  (h1 : pattern_length = 2 * 12) -- 2 feet converted to inches
  (h2 : semicircle_diameter = 3) -- diameter in inches
  : shaded_area_semicircle_pattern pattern_length semicircle_diameter h1 h2 = 18 * Real.pi := by
  sorry

end shaded_area_semicircle_pattern_shaded_area_semicircle_pattern_correct_l2823_282372


namespace geometric_sequence_first_term_l2823_282352

theorem geometric_sequence_first_term
  (a : ℝ) -- first term
  (r : ℝ) -- common ratio
  (h1 : a * r^2 = 18) -- third term is 18
  (h2 : a * r^4 = 72) -- fifth term is 72
  : a = 4.5 := by
  sorry

end geometric_sequence_first_term_l2823_282352


namespace max_value_quadratic_expression_l2823_282306

theorem max_value_quadratic_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 10) : 
  x^2 + 2*x*y + 3*y^2 ≤ 10*(45 + 42*Real.sqrt 3) :=
by sorry

end max_value_quadratic_expression_l2823_282306


namespace last_remaining_number_l2823_282381

/-- Represents the state of a number in Melanie's list -/
inductive NumberState
  | Unmarked
  | Marked
  | Eliminated

/-- Represents a round in Melanie's process -/
structure Round where
  skipCount : Nat
  startNumber : Nat

/-- The list of numbers Melanie works with -/
def initialList : List Nat := List.range 50

/-- Applies the marking and skipping pattern for a single round -/
def applyRound (list : List (Nat × NumberState)) (round : Round) : List (Nat × NumberState) :=
  sorry

/-- Applies all rounds until only one number remains unmarked -/
def applyAllRounds (list : List (Nat × NumberState)) : Nat :=
  sorry

/-- The main theorem stating that the last remaining number is 47 -/
theorem last_remaining_number :
  applyAllRounds (initialList.map (λ n => (n + 1, NumberState.Unmarked))) = 47 :=
sorry

end last_remaining_number_l2823_282381


namespace power_two_plus_two_gt_square_l2823_282314

theorem power_two_plus_two_gt_square (n : ℕ) (hn : n > 0) : 2^n + 2 > n^2 := by
  sorry

end power_two_plus_two_gt_square_l2823_282314


namespace inequality_empty_solution_set_l2823_282331

theorem inequality_empty_solution_set (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 < 0) ↔ -2 ≤ a ∧ a < 6/5 := by
  sorry

end inequality_empty_solution_set_l2823_282331


namespace log_inequality_l2823_282300

theorem log_inequality (x : ℝ) (h : x > 0) : Real.log (1 + x^2) < x^2 := by
  sorry

end log_inequality_l2823_282300


namespace card_pair_probability_l2823_282342

/-- Represents the number of cards for each value in the deck -/
def cardsPerValue : ℕ := 5

/-- Represents the number of different values in the deck -/
def numValues : ℕ := 10

/-- Represents the total number of cards in the original deck -/
def totalCards : ℕ := cardsPerValue * numValues

/-- Represents the number of pairs removed -/
def pairsRemoved : ℕ := 2

/-- Represents the number of cards remaining after removal -/
def remainingCards : ℕ := totalCards - (2 * pairsRemoved)

/-- Represents the number of values with full sets of cards after removal -/
def fullSets : ℕ := numValues - pairsRemoved

/-- Represents the number of values with reduced sets of cards after removal -/
def reducedSets : ℕ := pairsRemoved

theorem card_pair_probability :
  (fullSets * (cardsPerValue.choose 2) + reducedSets * ((cardsPerValue - 2).choose 2)) /
  (remainingCards.choose 2) = 86 / 1035 := by
  sorry

end card_pair_probability_l2823_282342


namespace line_point_value_l2823_282318

/-- Given a line containing points (2, 9), (15, m), and (35, 4), prove that m = 232/33 -/
theorem line_point_value (m : ℚ) : 
  (∃ (line : ℝ → ℝ), line 2 = 9 ∧ line 15 = m ∧ line 35 = 4) → m = 232/33 := by
  sorry

end line_point_value_l2823_282318


namespace point_in_third_quadrant_iff_m_less_than_one_l2823_282303

/-- A point P(x, y) is in the third quadrant if both x and y are negative -/
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- The x-coordinate of point P as a function of m -/
def x_coord (m : ℝ) : ℝ := m - 1

/-- The y-coordinate of point P as a function of m -/
def y_coord (m : ℝ) : ℝ := 2 * m - 3

/-- Theorem stating that for point P(m-1, 2m-3) to be in the third quadrant, m must be less than 1 -/
theorem point_in_third_quadrant_iff_m_less_than_one (m : ℝ) : 
  in_third_quadrant (x_coord m) (y_coord m) ↔ m < 1 := by
  sorry

end point_in_third_quadrant_iff_m_less_than_one_l2823_282303


namespace two_circles_in_triangle_l2823_282351

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a circle touches two sides of a triangle --/
def touchesTwoSides (c : Circle) (t : Triangle) : Prop := sorry

/-- Predicate to check if two circles touch each other --/
def circlesAreEqual (c1 c2 : Circle) : Prop := c1.radius = c2.radius

/-- Predicate to check if two circles touch each other --/
def circlesAreInscribed (c1 c2 : Circle) (t : Triangle) : Prop :=
  touchesTwoSides c1 t ∧ touchesTwoSides c2 t ∧ circlesAreEqual c1 c2

/-- Theorem stating that two equal circles can be inscribed in a triangle --/
theorem two_circles_in_triangle (t : Triangle) :
  ∃ c1 c2 : Circle, circlesAreInscribed c1 c2 t := by sorry

end two_circles_in_triangle_l2823_282351


namespace oil_ratio_in_first_bottle_l2823_282344

theorem oil_ratio_in_first_bottle 
  (C : ℝ) 
  (h1 : C > 0)
  (oil_in_second : ℝ)
  (h2 : oil_in_second = C / 2)
  (total_content : ℝ)
  (h3 : total_content = 3 * C)
  (total_oil : ℝ)
  (h4 : total_oil = total_content / 3)
  (oil_in_first : ℝ)
  (h5 : oil_in_first + oil_in_second = total_oil) :
  oil_in_first / C = 1 / 2 := by
sorry

end oil_ratio_in_first_bottle_l2823_282344


namespace exists_rational_less_than_neg_half_l2823_282357

theorem exists_rational_less_than_neg_half : ∃ q : ℚ, q < -1/2 := by
  sorry

end exists_rational_less_than_neg_half_l2823_282357


namespace jackson_courtyard_tile_cost_l2823_282353

/-- Calculates the total cost of tiles for a courtyard -/
def total_tile_cost (length width : ℝ) (tiles_per_sqft : ℝ) (green_tile_percent : ℝ) (green_tile_cost red_tile_cost : ℝ) : ℝ :=
  let area := length * width
  let total_tiles := area * tiles_per_sqft
  let green_tiles := green_tile_percent * total_tiles
  let red_tiles := total_tiles - green_tiles
  green_tiles * green_tile_cost + red_tiles * red_tile_cost

/-- Theorem stating the total cost of tiles for Jackson's courtyard -/
theorem jackson_courtyard_tile_cost :
  total_tile_cost 10 25 4 0.4 3 1.5 = 2100 :=
by sorry

end jackson_courtyard_tile_cost_l2823_282353
