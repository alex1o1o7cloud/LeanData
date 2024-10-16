import Mathlib

namespace NUMINAMATH_CALUDE_probability_between_C_and_E_l379_37979

/-- Given points A, B, C, D, and E on a line segment AB, where AB = 4AD = 8BC = 2DE,
    the probability of a randomly selected point on AB being between C and E is 7/8. -/
theorem probability_between_C_and_E (A B C D E : ℝ) : 
  A < B ∧ A ≤ C ∧ C < D ∧ D < E ∧ E ≤ B ∧
  B - A = 4 * (D - A) ∧
  B - A = 8 * (C - B) ∧
  B - A = 2 * (E - D) →
  (E - C) / (B - A) = 7 / 8 := by
sorry

end NUMINAMATH_CALUDE_probability_between_C_and_E_l379_37979


namespace NUMINAMATH_CALUDE_road_repaving_l379_37926

theorem road_repaving (total_repaved : ℕ) (repaved_today : ℕ) 
  (h1 : total_repaved = 4938)
  (h2 : repaved_today = 805) :
  total_repaved - repaved_today = 4133 := by
  sorry

end NUMINAMATH_CALUDE_road_repaving_l379_37926


namespace NUMINAMATH_CALUDE_angle_cde_is_85_l379_37937

-- Define the points
variable (A B C D E : Point)

-- Define the angles
def angle (P Q R : Point) : ℝ := sorry

-- State the conditions
variable (h1 : angle A B C = 90)
variable (h2 : angle B C D = 90)
variable (h3 : angle C D A = 90)
variable (h4 : angle A E B = 50)
variable (h5 : angle B E D = angle B D E)

-- State the theorem
theorem angle_cde_is_85 : angle C D E = 85 := by sorry

end NUMINAMATH_CALUDE_angle_cde_is_85_l379_37937


namespace NUMINAMATH_CALUDE_curve_symmetry_line_dot_product_l379_37987

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 1 = 0

-- Define the symmetry line
def symmetry_line (m : ℝ) (x y : ℝ) : Prop := x + m*y + 4 = 0

-- Define the dot product condition
def dot_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

theorem curve_symmetry_line_dot_product 
  (P Q : ℝ × ℝ) (m : ℝ) 
  (h_curve_P : curve P.1 P.2)
  (h_curve_Q : curve Q.1 Q.2)
  (h_symmetry : ∃ (x y : ℝ), symmetry_line m x y ∧ 
    (x - P.1)^2 + (y - P.2)^2 = (x - Q.1)^2 + (y - Q.2)^2)
  (h_dot_product : dot_product_condition P.1 P.2 Q.1 Q.2) :
  m = -1 ∧ Q.2 = -Q.1 + 1 ∧ P.2 = -P.1 + 1 :=
sorry

end NUMINAMATH_CALUDE_curve_symmetry_line_dot_product_l379_37987


namespace NUMINAMATH_CALUDE_no_solutions_sqrt_1452_l379_37998

theorem no_solutions_sqrt_1452 : 
  ¬ ∃ (x y : ℕ), 0 < x ∧ x < y ∧ Real.sqrt 1452 = Real.sqrt x + Real.sqrt y := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_sqrt_1452_l379_37998


namespace NUMINAMATH_CALUDE_one_match_among_withdrawn_l379_37909

/-- Represents a table tennis singles competition. -/
structure TableTennisCompetition where
  n : ℕ  -- Total number of players excluding the 3 who withdrew
  x : ℕ  -- Number of matches played among the 3 withdrawn players

/-- Conditions for the competition -/
def validCompetition (comp : TableTennisCompetition) : Prop :=
  comp.n * (comp.n - 1) / 2 + (6 - comp.x) = 50

theorem one_match_among_withdrawn (comp : TableTennisCompetition) :
  validCompetition comp → comp.x = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_match_among_withdrawn_l379_37909


namespace NUMINAMATH_CALUDE_jamie_coin_problem_l379_37914

/-- The number of nickels (and dimes and quarters) in Jamie's jar -/
def num_coins : ℕ := 33

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The total value of coins in Jamie's jar in cents -/
def total_value : ℕ := 1320

theorem jamie_coin_problem :
  num_coins * nickel_value + num_coins * dime_value + num_coins * quarter_value = total_value :=
by sorry

end NUMINAMATH_CALUDE_jamie_coin_problem_l379_37914


namespace NUMINAMATH_CALUDE_cauchy_schwarz_inequality_l379_37902

theorem cauchy_schwarz_inequality (x y a b : ℝ) : 
  a * x + b * y ≤ Real.sqrt (a^2 + b^2) * Real.sqrt (x^2 + y^2) := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_inequality_l379_37902


namespace NUMINAMATH_CALUDE_monic_quartic_problem_l379_37938

/-- A monic quartic polynomial is a polynomial of degree 4 with leading coefficient 1 -/
def IsMonicQuartic (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d

/-- The theorem statement -/
theorem monic_quartic_problem (f : ℝ → ℝ) 
  (h_monic : IsMonicQuartic f)
  (h_neg2 : f (-2) = 0)
  (h_3 : f 3 = -9)
  (h_neg4 : f (-4) = -16)
  (h_5 : f 5 = -25) :
  f 0 = 0 := by sorry

end NUMINAMATH_CALUDE_monic_quartic_problem_l379_37938


namespace NUMINAMATH_CALUDE_olivia_wallet_remaining_l379_37947

/-- The amount of money remaining in Olivia's wallet after shopping -/
def remaining_money (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Theorem stating that Olivia has 29 dollars left in her wallet -/
theorem olivia_wallet_remaining : remaining_money 54 25 = 29 := by
  sorry

end NUMINAMATH_CALUDE_olivia_wallet_remaining_l379_37947


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l379_37953

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, x^2 + b*x - a < 0 ↔ -2 < x ∧ x < 3) → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l379_37953


namespace NUMINAMATH_CALUDE_knight_placement_exists_l379_37966

/-- A position on the modified 6x6 board -/
structure Position :=
  (x : Fin 6)
  (y : Fin 6)
  (valid : ¬((x < 2 ∧ y < 2) ∨ (x > 3 ∧ y < 2) ∨ (x < 2 ∧ y > 3) ∨ (x > 3 ∧ y > 3)))

/-- A knight's move -/
def knightMove (p q : Position) : Prop :=
  (abs (p.x - q.x) == 2 ∧ abs (p.y - q.y) == 1) ∨
  (abs (p.x - q.x) == 1 ∧ abs (p.y - q.y) == 2)

/-- A valid knight placement -/
structure KnightPlacement :=
  (positions : Fin 10 → Position × Position)
  (distinct : ∀ i j, i ≠ j → positions i ≠ positions j)
  (canAttack : ∀ i, knightMove (positions i).1 (positions i).2)

/-- The main theorem -/
theorem knight_placement_exists : ∃ (k : KnightPlacement), True :=
sorry

end NUMINAMATH_CALUDE_knight_placement_exists_l379_37966


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l379_37988

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := 2 * x + (m + 1) * y + 4 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := m * x + 3 * y - 2 = 0

-- Define the parallel relation
def parallel (m : ℝ) : Prop := ∀ (x y : ℝ), l₁ m x y ↔ l₂ m x y

-- Theorem statement
theorem parallel_lines_m_value (m : ℝ) :
  parallel m → m = 2 ∨ m = -3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l379_37988


namespace NUMINAMATH_CALUDE_smallest_common_factor_l379_37964

theorem smallest_common_factor (n : ℕ) : 
  (∀ m : ℕ, m < 5 → Nat.gcd (11 * m - 3) (8 * m + 4) = 1) ∧ 
  Nat.gcd (11 * 5 - 3) (8 * 5 + 4) > 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l379_37964


namespace NUMINAMATH_CALUDE_altitude_and_median_equations_l379_37945

/-- Triangle with vertices A(4,0), B(6,7), and C(0,3) -/
structure Triangle where
  A : ℝ × ℝ := (4, 0)
  B : ℝ × ℝ := (6, 7)
  C : ℝ × ℝ := (0, 3)

/-- Line represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Altitude from B to side AC -/
def altitude (t : Triangle) : Line :=
  { a := 3, b := 2, c := -12 }

/-- Median from B to side AC -/
def median (t : Triangle) : Line :=
  { a := 4, b := -6, c := 1 }

/-- Theorem stating that the altitude and median equations are correct -/
theorem altitude_and_median_equations (t : Triangle) : 
  (altitude t = { a := 3, b := 2, c := -12 }) ∧ 
  (median t = { a := 4, b := -6, c := 1 }) := by
  sorry

end NUMINAMATH_CALUDE_altitude_and_median_equations_l379_37945


namespace NUMINAMATH_CALUDE_cds_per_rack_is_eight_l379_37900

/-- The number of racks a shelf can hold -/
def num_racks : ℕ := 4

/-- The total number of CDs a shelf can hold -/
def total_cds : ℕ := 32

/-- The number of CDs each rack can hold -/
def cds_per_rack : ℕ := total_cds / num_racks

theorem cds_per_rack_is_eight : cds_per_rack = 8 := by
  sorry

end NUMINAMATH_CALUDE_cds_per_rack_is_eight_l379_37900


namespace NUMINAMATH_CALUDE_inequality_solution_set_l379_37903

theorem inequality_solution_set (x : ℝ) :
  (4 * x^2 - 9 * x > 5) ↔ (x < -1/4 ∨ x > 5) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l379_37903


namespace NUMINAMATH_CALUDE_student_proportion_is_frequency_rate_l379_37929

/-- Represents the total number of people in the population -/
def total_population : ℕ := 10

/-- Represents the number of students in the population -/
def number_of_students : ℕ := 4

/-- Represents the proportion of students in the population -/
def student_proportion : ℚ := 2 / 5

/-- Defines what a frequency rate is in this context -/
def is_frequency_rate (proportion : ℚ) (num : ℕ) (total : ℕ) : Prop :=
  proportion = num / total

/-- Theorem stating that the given proportion is a frequency rate -/
theorem student_proportion_is_frequency_rate :
  is_frequency_rate student_proportion number_of_students total_population := by
  sorry

end NUMINAMATH_CALUDE_student_proportion_is_frequency_rate_l379_37929


namespace NUMINAMATH_CALUDE_common_root_quadratic_equations_l379_37921

theorem common_root_quadratic_equations (a : ℝ) :
  (∃ x : ℝ, x^2 + x + a = 0 ∧ x^2 + a*x + 1 = 0) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_common_root_quadratic_equations_l379_37921


namespace NUMINAMATH_CALUDE_det_B_is_one_l379_37940

theorem det_B_is_one (b e : ℝ) : 
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![b, 2; -3, e]
  B + B⁻¹ = 1 → Matrix.det B = 1 := by
sorry

end NUMINAMATH_CALUDE_det_B_is_one_l379_37940


namespace NUMINAMATH_CALUDE_selection_methods_five_three_two_l379_37930

/-- The number of ways to select 3 students out of 5 for 3 different language majors,
    where 2 specific students cannot be selected for one particular major -/
def selection_methods (n : ℕ) (k : ℕ) (excluded : ℕ) : ℕ :=
  Nat.choose (n - excluded) 1 * (n - 1).factorial / (n - k).factorial

theorem selection_methods_five_three_two :
  selection_methods 5 3 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_five_three_two_l379_37930


namespace NUMINAMATH_CALUDE_hyperbola_equation_l379_37980

theorem hyperbola_equation (m a b : ℝ) :
  (∀ x y : ℝ, x^2 / (4 + m^2) + y^2 / m^2 = 1 → x^2 / a^2 - y^2 / b^2 = 1) →
  (a^2 + b^2 = 4) →
  (a^2 / b^2 = 4) →
  x^2 - y^2 / 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l379_37980


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l379_37977

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  interior_angle = 160 → n * (180 - interior_angle) = 360 → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l379_37977


namespace NUMINAMATH_CALUDE_gcd_of_720_120_168_l379_37928

theorem gcd_of_720_120_168 : Nat.gcd 720 (Nat.gcd 120 168) = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_720_120_168_l379_37928


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_bound_l379_37946

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if a line passing through the right focus with a slope angle of 60° 
    intersects the right branch of the hyperbola at exactly one point,
    then the eccentricity e of the hyperbola satisfies e ≥ 2. -/
theorem hyperbola_eccentricity_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  let slope := Real.tan (π / 3)
  (b / a ≥ slope) → e ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_bound_l379_37946


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l379_37918

theorem arithmetic_sequence_count (a₁ d : ℤ) (n : ℕ) :
  a₁ = -3 →
  d = 4 →
  a₁ + (n - 1) * d ≤ 40 →
  (∀ k : ℕ, k < n → a₁ + (k - 1) * d ≤ 40) →
  (∀ k : ℕ, k > n → a₁ + (k - 1) * d > 40) →
  n = 11 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_count_l379_37918


namespace NUMINAMATH_CALUDE_total_drawings_l379_37924

/-- The number of neighbors Shiela has -/
def num_neighbors : ℕ := 6

/-- The number of drawings each neighbor receives -/
def drawings_per_neighbor : ℕ := 9

/-- Theorem: The total number of drawings Shiela made is 54 -/
theorem total_drawings : num_neighbors * drawings_per_neighbor = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_drawings_l379_37924


namespace NUMINAMATH_CALUDE_total_profit_calculation_l379_37984

def total_subscription : ℕ := 50000
def a_extra_over_b : ℕ := 4000
def b_extra_over_c : ℕ := 5000
def a_profit : ℕ := 15120

theorem total_profit_calculation :
  ∃ (c_subscription : ℕ) (total_profit : ℕ),
    let b_subscription := c_subscription + b_extra_over_c
    let a_subscription := b_subscription + a_extra_over_b
    a_subscription + b_subscription + c_subscription = total_subscription ∧
    a_subscription * total_profit = a_profit * total_subscription ∧
    total_profit = 36000 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_calculation_l379_37984


namespace NUMINAMATH_CALUDE_contradiction_elements_correct_l379_37970

/-- Elements used in the method of contradiction -/
inductive ContradictionElement
  | assumption
  | originalCondition
  | axiomTheoremDefinition

/-- The set of elements used in the method of contradiction -/
def contradictionElements : Set ContradictionElement :=
  {ContradictionElement.assumption, ContradictionElement.originalCondition, ContradictionElement.axiomTheoremDefinition}

/-- Theorem stating that the set of elements used in the method of contradiction
    is exactly the set containing assumptions, original conditions, and axioms/theorems/definitions -/
theorem contradiction_elements_correct :
  contradictionElements = {ContradictionElement.assumption, ContradictionElement.originalCondition, ContradictionElement.axiomTheoremDefinition} := by
  sorry


end NUMINAMATH_CALUDE_contradiction_elements_correct_l379_37970


namespace NUMINAMATH_CALUDE_tan_sum_specific_angles_l379_37968

theorem tan_sum_specific_angles (α β : Real) 
  (h1 : Real.tan α = 2) 
  (h2 : Real.tan β = 3) : 
  Real.tan (α + β) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_specific_angles_l379_37968


namespace NUMINAMATH_CALUDE_exam_passing_probability_l379_37986

def total_questions : ℕ := 10
def selected_questions : ℕ := 3
def questions_student_can_answer : ℕ := 6
def questions_to_pass : ℕ := 2

def probability_of_passing : ℚ :=
  (Nat.choose questions_student_can_answer selected_questions +
   Nat.choose questions_student_can_answer (selected_questions - 1) *
   Nat.choose (total_questions - questions_student_can_answer) 1) /
  Nat.choose total_questions selected_questions

theorem exam_passing_probability :
  probability_of_passing = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_exam_passing_probability_l379_37986


namespace NUMINAMATH_CALUDE_yvonne_word_count_l379_37996

/-- Proves that Yvonne wrote 400 words given the conditions of the research paper problem -/
theorem yvonne_word_count 
  (total_required : Nat) 
  (janna_extra : Nat) 
  (words_removed : Nat) 
  (words_to_add : Nat) 
  (h1 : total_required = 1000)
  (h2 : janna_extra = 150)
  (h3 : words_removed = 20)
  (h4 : words_to_add = 30) : 
  ∃ (yvonne_words : Nat), 
    yvonne_words + (yvonne_words + janna_extra) - words_removed + 2 * words_removed + words_to_add = total_required ∧ 
    yvonne_words = 400 := by
  sorry

#check yvonne_word_count

end NUMINAMATH_CALUDE_yvonne_word_count_l379_37996


namespace NUMINAMATH_CALUDE_sum_of_i_powers_l379_37969

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers :
  i^12 + i^17 + i^22 + i^27 + i^32 + i^37 = 1 + i :=
by
  sorry

-- Define the property i^4 = 1
axiom i_fourth_power : i^4 = 1

-- Define i as the imaginary unit
axiom i_squared : i^2 = -1

end NUMINAMATH_CALUDE_sum_of_i_powers_l379_37969


namespace NUMINAMATH_CALUDE_log_equality_implies_ratio_l379_37962

-- Define the conditions
theorem log_equality_implies_ratio (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (Real.log p / Real.log 8 = Real.log q / Real.log 18) ∧
  (Real.log p / Real.log 8 = Real.log (p + q) / Real.log 32) →
  q / p = (4 + Real.sqrt 41) / 5 := by
  sorry

-- The proof is omitted as per instructions

end NUMINAMATH_CALUDE_log_equality_implies_ratio_l379_37962


namespace NUMINAMATH_CALUDE_two_true_propositions_l379_37919

def p (x y : ℝ) : Prop := (x > |y|) → (x > y)

def q (x y : ℝ) : Prop := (x + y > 0) → (x^2 > y^2)

theorem two_true_propositions (x y : ℝ) :
  (p x y ∨ q x y) ∧
  ¬(¬(p x y) ∧ ¬(q x y)) ∧
  (p x y ∧ ¬(q x y)) ∧
  ¬(p x y ∧ q x y) :=
sorry

end NUMINAMATH_CALUDE_two_true_propositions_l379_37919


namespace NUMINAMATH_CALUDE_multiple_properties_l379_37976

-- Define the properties of c and d
def is_multiple_of_4 (n : ℤ) : Prop := ∃ k : ℤ, n = 4 * k
def is_multiple_of_8 (n : ℤ) : Prop := ∃ k : ℤ, n = 8 * k

-- Define the theorem
theorem multiple_properties (c d : ℤ) 
  (hc : is_multiple_of_4 c) (hd : is_multiple_of_8 d) : 
  (is_multiple_of_4 d) ∧ 
  (is_multiple_of_4 (c + d)) ∧ 
  (∃ k : ℤ, c + d = 2 * k) := by
  sorry


end NUMINAMATH_CALUDE_multiple_properties_l379_37976


namespace NUMINAMATH_CALUDE_find_first_number_l379_37983

theorem find_first_number (x : ℝ) : 
  let set1 := [20, 40, 60]
  let set2 := [x, 70, 19]
  (set1.sum / set1.length) = (set2.sum / set2.length) + 7 →
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_find_first_number_l379_37983


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l379_37989

def p (x : ℝ) : Prop := x + 2 ≥ 10 ∧ x - 10 ≤ 0

def q (x m : ℝ) : Prop := -m ≤ x ∧ x ≤ 1 + m

theorem necessary_not_sufficient_condition (m : ℝ) :
  (∀ x, ¬(q x m) → ¬(p x)) ∧ (∃ x, ¬(p x) ∧ q x m) → m ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l379_37989


namespace NUMINAMATH_CALUDE_stratified_sample_size_l379_37963

/-- Represents the total number of employees -/
def total_employees : ℕ := 750

/-- Represents the number of young employees -/
def young_employees : ℕ := 350

/-- Represents the number of middle-aged employees -/
def middle_aged_employees : ℕ := 250

/-- Represents the number of elderly employees -/
def elderly_employees : ℕ := 150

/-- Represents the number of young employees in the sample -/
def young_in_sample : ℕ := 7

/-- Theorem stating that the sample size is 15 given the conditions -/
theorem stratified_sample_size :
  ∃ (sample_size : ℕ),
    sample_size * young_employees = young_in_sample * total_employees ∧
    sample_size = 15 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l379_37963


namespace NUMINAMATH_CALUDE_wage_comparison_l379_37995

/-- Proves that given the wage relationships between Erica, Robin, and Charles,
    Charles earns approximately 170% more than Erica. -/
theorem wage_comparison (erica robin charles : ℝ) 
  (h1 : robin = erica * 1.30)
  (h2 : charles = robin * 1.3076923076923077) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0000001 ∧ 
  charles = erica * (2.70 + ε) :=
sorry

end NUMINAMATH_CALUDE_wage_comparison_l379_37995


namespace NUMINAMATH_CALUDE_unique_solution_k_squared_minus_2016_equals_3_to_n_l379_37935

theorem unique_solution_k_squared_minus_2016_equals_3_to_n :
  ∃! (k n : ℕ), k > 0 ∧ n > 0 ∧ k^2 - 2016 = 3^n := by sorry

end NUMINAMATH_CALUDE_unique_solution_k_squared_minus_2016_equals_3_to_n_l379_37935


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_l379_37907

theorem sum_of_roots_cubic (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 3 * x^3 - 4 * x^2 - 9 * x
  (∃ a b c : ℝ, f x = (x - a) * (x - b) * (x - c)) →
  a + b + c = 4/3 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_l379_37907


namespace NUMINAMATH_CALUDE_area_invariant_under_translation_l379_37910

/-- Represents a rectangle in a 2D plane --/
structure Rectangle where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- Represents a quadrilateral formed by intersection points of two rectangles --/
structure IntersectionQuadrilateral where
  points : Fin 4 → ℝ × ℝ

/-- Calculates the area of a quadrilateral given its four vertices --/
def quadrilateralArea (q : IntersectionQuadrilateral) : ℝ :=
  sorry

/-- Translates a rectangle by a given vector --/
def translateRectangle (r : Rectangle) (v : ℝ × ℝ) : Rectangle :=
  sorry

/-- Finds the intersection points of two rectangles --/
def findIntersectionPoints (r1 r2 : Rectangle) : Fin 8 → ℝ × ℝ :=
  sorry

/-- Forms a quadrilateral from alternating intersection points --/
def formQuadrilateral (points : Fin 8 → ℝ × ℝ) : IntersectionQuadrilateral :=
  sorry

/-- The main theorem: area invariance under rectangle translation --/
theorem area_invariant_under_translation 
  (r1 r2 : Rectangle) 
  (v : ℝ × ℝ) : 
  let points := findIntersectionPoints r1 r2
  let q1 := formQuadrilateral points
  let r2_translated := translateRectangle r2 v
  let points_after := findIntersectionPoints r1 r2_translated
  let q2 := formQuadrilateral points_after
  quadrilateralArea q1 = quadrilateralArea q2 := by
  sorry

end NUMINAMATH_CALUDE_area_invariant_under_translation_l379_37910


namespace NUMINAMATH_CALUDE_paco_cookies_theorem_l379_37957

/-- Represents the number of cookies Paco has after all actions --/
def remaining_cookies (initial_salty initial_sweet initial_chocolate : ℕ)
  (eaten_sweet eaten_salty : ℕ) (given_chocolate received_chocolate : ℕ) :
  ℕ × ℕ × ℕ :=
  let remaining_sweet := initial_sweet - eaten_sweet
  let remaining_salty := initial_salty - eaten_salty
  let remaining_chocolate := initial_chocolate - given_chocolate + received_chocolate
  (remaining_sweet, remaining_salty, remaining_chocolate)

/-- Theorem stating the final number of cookies Paco has --/
theorem paco_cookies_theorem :
  remaining_cookies 97 34 45 15 56 22 7 = (19, 41, 30) := by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_theorem_l379_37957


namespace NUMINAMATH_CALUDE_specific_grades_average_l379_37906

/-- The overall average percentage of three subjects -/
def overall_average (math_grade : ℚ) (history_grade : ℚ) (third_subject_grade : ℚ) : ℚ :=
  (math_grade + history_grade + third_subject_grade) / 3

/-- Theorem stating that given specific grades, the overall average is 75% -/
theorem specific_grades_average :
  overall_average 74 84 67 = 75 := by
  sorry

end NUMINAMATH_CALUDE_specific_grades_average_l379_37906


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l379_37934

def i : ℂ := Complex.I

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := 1 / (1 + i)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l379_37934


namespace NUMINAMATH_CALUDE_rectangular_box_area_product_l379_37978

/-- Given a rectangular box with dimensions length, width, and height,
    prove that the product of the areas of its base, side, and front
    is equal to the square of its volume. -/
theorem rectangular_box_area_product (length width height : ℝ) :
  (length * width) * (width * height) * (height * length) = (length * width * height) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_area_product_l379_37978


namespace NUMINAMATH_CALUDE_symmetric_points_difference_l379_37901

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are opposite and their y-coordinates are equal -/
def symmetric_wrt_y_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = p2.2

/-- Given two points M(a, 3) and N(5, b) that are symmetric with respect to the y-axis,
    prove that a - b = -8 -/
theorem symmetric_points_difference (a b : ℝ) 
    (h : symmetric_wrt_y_axis (a, 3) (5, b)) : a - b = -8 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_difference_l379_37901


namespace NUMINAMATH_CALUDE_fourth_player_score_zero_l379_37915

/-- Represents the score of a player in the chess tournament -/
structure PlayerScore :=
  (score : ℕ)

/-- Represents the scores of all players in the tournament -/
structure TournamentScores :=
  (players : Fin 4 → PlayerScore)

/-- The total points awarded in a tournament with 4 players -/
def totalPoints : ℕ := 12

/-- Theorem stating that if three players have scores 6, 4, and 2, the fourth must have 0 -/
theorem fourth_player_score_zero (t : TournamentScores) :
  (∃ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    (t.players i).score = 6 ∧ 
    (t.players j).score = 4 ∧ 
    (t.players k).score = 2) →
  (∃ (l : Fin 4), (∀ m : Fin 4, m ≠ l → 
    (t.players m).score = 6 ∨ 
    (t.players m).score = 4 ∨ 
    (t.players m).score = 2) ∧ 
  (t.players l).score = 0) :=
by sorry

end NUMINAMATH_CALUDE_fourth_player_score_zero_l379_37915


namespace NUMINAMATH_CALUDE_arc_length_sector_l379_37942

/-- The arc length of a sector with central angle 2π/3 and radius 3 is 2π. -/
theorem arc_length_sector (α : Real) (r : Real) (l : Real) : 
  α = 2 * Real.pi / 3 → r = 3 → l = α * r → l = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_arc_length_sector_l379_37942


namespace NUMINAMATH_CALUDE_typing_time_together_l379_37936

-- Define the typing rates for Randy and Candy
def randy_rate : ℚ := 1 / 30
def candy_rate : ℚ := 1 / 45

-- Define the combined typing rate
def combined_rate : ℚ := randy_rate + candy_rate

-- Theorem to prove
theorem typing_time_together : (1 : ℚ) / combined_rate = 18 := by
  sorry

end NUMINAMATH_CALUDE_typing_time_together_l379_37936


namespace NUMINAMATH_CALUDE_cylinder_from_constant_rho_l379_37927

/-- Cylindrical coordinates -/
structure CylindricalCoord where
  ρ : ℝ
  φ : ℝ
  z : ℝ

/-- A set of points in cylindrical coordinates -/
def CylindricalSet (c : ℝ) : Set CylindricalCoord :=
  {p : CylindricalCoord | p.ρ = c}

/-- Definition of a cylinder in cylindrical coordinates -/
def IsCylinder (S : Set CylindricalCoord) : Prop :=
  ∃ c > 0, S = CylindricalSet c

/-- Theorem: The set of points satisfying ρ = c forms a cylinder -/
theorem cylinder_from_constant_rho (c : ℝ) (hc : c > 0) :
  IsCylinder (CylindricalSet c) := by
  sorry


end NUMINAMATH_CALUDE_cylinder_from_constant_rho_l379_37927


namespace NUMINAMATH_CALUDE_parabola_shift_l379_37992

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a, b := -2 * p.a * h + p.b, c := p.a * h^2 - p.b * h + p.c }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + v }

/-- The main theorem stating that shifting y = 3x^2 right by 1 and down by 2 results in y = 3(x-1)^2 - 2 -/
theorem parabola_shift :
  let p := Parabola.mk 3 0 0
  let p_shifted := shift_vertical (shift_horizontal p 1) (-2)
  p_shifted = Parabola.mk 3 (-6) 1 := by sorry

end NUMINAMATH_CALUDE_parabola_shift_l379_37992


namespace NUMINAMATH_CALUDE_b_95_mod_121_l379_37932

/-- Calculate b₉₅ modulo 121 where bₙ = 5ⁿ + 11ⁿ -/
theorem b_95_mod_121 : (5^95 + 11^95) % 121 = 16 := by
  sorry

end NUMINAMATH_CALUDE_b_95_mod_121_l379_37932


namespace NUMINAMATH_CALUDE_pages_per_chapter_l379_37912

/-- Given a book with 555 pages equally distributed over 5 chapters,
    prove that each chapter contains 111 pages. -/
theorem pages_per_chapter (total_pages : ℕ) (num_chapters : ℕ) (pages_per_chapter : ℕ) :
  total_pages = 555 →
  num_chapters = 5 →
  total_pages = num_chapters * pages_per_chapter →
  pages_per_chapter = 111 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_chapter_l379_37912


namespace NUMINAMATH_CALUDE_smallest_among_three_l379_37950

theorem smallest_among_three : ∀ (a b c : ℕ), a = 5 ∧ b = 8 ∧ c = 4 → c ≤ a ∧ c ≤ b := by
  sorry

end NUMINAMATH_CALUDE_smallest_among_three_l379_37950


namespace NUMINAMATH_CALUDE_partitioned_triangle_area_l379_37990

/-- Represents a triangle partitioned into three triangles and a quadrilateral -/
structure PartitionedTriangle where
  /-- Area of the first triangle -/
  area1 : ℝ
  /-- Area of the second triangle -/
  area2 : ℝ
  /-- Area of the third triangle -/
  area3 : ℝ
  /-- Area of the quadrilateral -/
  areaQuad : ℝ

/-- The theorem statement -/
theorem partitioned_triangle_area (t : PartitionedTriangle) 
  (h1 : t.area1 = 5)
  (h2 : t.area2 = 9)
  (h3 : t.area3 = 9) :
  t.areaQuad = 40 := by
  sorry

end NUMINAMATH_CALUDE_partitioned_triangle_area_l379_37990


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l379_37971

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola x^2 = 12y -/
def Parabola := {p : Point | p.x^2 = 12 * p.y}

/-- Represents a line y = kx + m -/
def Line (k m : ℝ) := {p : Point | p.y = k * p.x + m}

/-- The focus of the parabola -/
def focus : Point := ⟨0, 3⟩

theorem parabola_line_intersection (k m : ℝ) (h_k : k > 0) :
  ∃ A B : Point,
    A ∈ Parabola ∧
    B ∈ Parabola ∧
    A ∈ Line k m ∧
    B ∈ Line k m ∧
    focus ∈ Line k m ∧
    (A.x - B.x)^2 + (A.y - B.y)^2 = 36^2 →
    k = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l379_37971


namespace NUMINAMATH_CALUDE_bowling_team_average_weight_l379_37974

theorem bowling_team_average_weight 
  (original_team_size : ℕ) 
  (new_player1_weight : ℝ) 
  (new_player2_weight : ℝ) 
  (new_average_weight : ℝ) 
  (h1 : original_team_size = 7)
  (h2 : new_player1_weight = 110)
  (h3 : new_player2_weight = 60)
  (h4 : new_average_weight = 106) :
  ∃ (original_average_weight : ℝ),
    (original_team_size * original_average_weight + new_player1_weight + new_player2_weight) / 
    (original_team_size + 2) = new_average_weight ∧
    original_average_weight = 112 :=
by sorry

end NUMINAMATH_CALUDE_bowling_team_average_weight_l379_37974


namespace NUMINAMATH_CALUDE_park_short_trees_l379_37943

def initial_short_trees : ℕ := 3
def short_trees_to_plant : ℕ := 9
def final_short_trees : ℕ := 12

theorem park_short_trees :
  initial_short_trees + short_trees_to_plant = final_short_trees :=
by sorry

end NUMINAMATH_CALUDE_park_short_trees_l379_37943


namespace NUMINAMATH_CALUDE_great_pyramid_dimensions_l379_37956

/-- The Great Pyramid of Giza's dimensions and sum of height and width -/
theorem great_pyramid_dimensions :
  let height := 500 + 20
  let width := height + 234
  height + width = 1274 := by sorry

end NUMINAMATH_CALUDE_great_pyramid_dimensions_l379_37956


namespace NUMINAMATH_CALUDE_cannot_compare_greening_areas_l379_37917

/-- Represents a city with its total area and greening coverage rate -/
structure City where
  total_area : ℝ
  greening_rate : ℝ
  greening_rate_nonneg : 0 ≤ greening_rate
  greening_rate_le_one : greening_rate ≤ 1

/-- Calculates the greening coverage area of a city -/
def greening_area (city : City) : ℝ :=
  city.total_area * city.greening_rate

/-- Theorem stating that we cannot determine which city has a larger greening area
    based solely on their greening coverage rates -/
theorem cannot_compare_greening_areas (city_a city_b : City) 
  (ha : city_a.greening_rate = 0.1) (hb : city_b.greening_rate = 0.08) :
  ¬ (∀ a b : City, a.greening_rate > b.greening_rate → greening_area a > greening_area b) :=
by
  sorry


end NUMINAMATH_CALUDE_cannot_compare_greening_areas_l379_37917


namespace NUMINAMATH_CALUDE_right_triangle_area_l379_37999

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) :
  hypotenuse = 10 →
  angle = 45 * π / 180 →
  (1 / 2) * hypotenuse * hypotenuse * Real.sin angle = 25 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l379_37999


namespace NUMINAMATH_CALUDE_concert_ticket_cost_is_181_l379_37954

/-- Calculates the cost of a concert ticket given hourly wage, weekly hours, percentage of monthly salary for outing, drink ticket cost, and number of drink tickets. -/
def concert_ticket_cost (hourly_wage : ℚ) (weekly_hours : ℚ) (outing_percentage : ℚ) (drink_ticket_cost : ℚ) (num_drink_tickets : ℕ) : ℚ :=
  let monthly_salary := hourly_wage * weekly_hours * 4
  let outing_budget := monthly_salary * outing_percentage
  let drink_tickets_cost := drink_ticket_cost * num_drink_tickets
  outing_budget - drink_tickets_cost

/-- Theorem stating that the cost of the concert ticket is $181 given the specified conditions. -/
theorem concert_ticket_cost_is_181 :
  concert_ticket_cost 18 30 (1/10) 7 5 = 181 := by
  sorry

#eval concert_ticket_cost 18 30 (1/10) 7 5

end NUMINAMATH_CALUDE_concert_ticket_cost_is_181_l379_37954


namespace NUMINAMATH_CALUDE_hawks_score_l379_37991

theorem hawks_score (total_points eagles_margin : ℕ) 
  (h1 : total_points = 42)
  (h2 : eagles_margin = 18) : 
  let hawks_score := (total_points - eagles_margin) / 2
  hawks_score = 12 := by
sorry

end NUMINAMATH_CALUDE_hawks_score_l379_37991


namespace NUMINAMATH_CALUDE_circle_transformation_l379_37959

/-- Given a circle and a transformation, prove the equation of the resulting shape -/
theorem circle_transformation (x y x' y' : ℝ) : 
  (x^2 + y^2 = 4) → (x' = 2*x ∧ y' = 3*y) → ((x'^2 / 16) + (y'^2 / 36) = 1) := by
sorry

end NUMINAMATH_CALUDE_circle_transformation_l379_37959


namespace NUMINAMATH_CALUDE_joey_age_l379_37975

def ages : List ℕ := [4, 6, 8, 10, 12]

def is_cinema_pair (a b : ℕ) : Prop := a + b = 18 ∧ a ∈ ages ∧ b ∈ ages ∧ a ≠ b

def is_soccer_pair (a b : ℕ) : Prop := 
  a < 11 ∧ b < 11 ∧ a ∈ ages ∧ b ∈ ages ∧ a ≠ b ∧ 
  ¬(∃ c d, is_cinema_pair c d ∧ (a = c ∨ a = d ∨ b = c ∨ b = d))

theorem joey_age : 
  (∃ a b c d, is_cinema_pair a b ∧ is_soccer_pair c d) →
  (∃! x, x ∈ ages ∧ x ≠ 6 ∧ ¬(∃ y z, (is_cinema_pair x y ∨ is_cinema_pair y x) ∧ 
                                     (is_soccer_pair x z ∨ is_soccer_pair z x))) →
  (∃ x, x ∈ ages ∧ x ≠ 6 ∧ ¬(∃ y z, (is_cinema_pair x y ∨ is_cinema_pair y x) ∧ 
                                    (is_soccer_pair x z ∨ is_soccer_pair z x)) ∧ x = 8) :=
by sorry

end NUMINAMATH_CALUDE_joey_age_l379_37975


namespace NUMINAMATH_CALUDE_hyperbola_focus_distance_l379_37960

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the asymptote
def asymptote (x y : ℝ) : Prop :=
  x + Real.sqrt 2 * y = 0

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = 12 * x

-- Define the focus of the parabola
def parabola_focus (x : ℝ) : Prop :=
  x = 3

-- Define the point M on the hyperbola
def point_M (x y : ℝ) : Prop :=
  x = -3 ∧ y = Real.sqrt 6 / 2

-- Define the line F2M
def line_F2M (x y : ℝ) : Prop :=
  y = -Real.sqrt 6 / 12 * x + Real.sqrt 6 / 4

-- State the theorem
theorem hyperbola_focus_distance (a b x y : ℝ) :
  hyperbola a b x y →
  asymptote x y →
  parabola_focus a →
  point_M x y →
  line_F2M x y →
  (6 : ℝ) / 5 = abs (-Real.sqrt 6 / 12 * (-3) + Real.sqrt 6 / 4) / Real.sqrt (1 + 6 / 144) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_distance_l379_37960


namespace NUMINAMATH_CALUDE_rectangle_length_l379_37994

/-- Given a rectangle where the length is three times the width, and decreasing the length by 5
    while increasing the width by 5 results in a square, prove that the original length is 15. -/
theorem rectangle_length (w : ℝ) (h1 : w > 0) : 
  (∃ l : ℝ, l = 3 * w ∧ l - 5 = w + 5) → 3 * w = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l379_37994


namespace NUMINAMATH_CALUDE_right_triangle_median_geometric_mean_l379_37952

theorem right_triangle_median_geometric_mean (c : ℝ) (h : c > 0) :
  ∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    c^2 = a^2 + b^2 ∧
    (c / 2)^2 = a * b ∧
    a + b = c * Real.sqrt (3 / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_median_geometric_mean_l379_37952


namespace NUMINAMATH_CALUDE_presidency_meeting_arrangements_l379_37922

/-- Represents the number of schools --/
def num_schools : ℕ := 3

/-- Represents the number of members per school --/
def members_per_school : ℕ := 5

/-- Calculates the number of ways to choose r items from n items --/
def choose (n : ℕ) (r : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial r) * (Nat.factorial (n - r)))

/-- Represents the number of ways to choose representatives from the host school --/
def host_school_choices : ℕ := choose members_per_school 2

/-- Represents the number of ways to choose representatives from non-host schools --/
def non_host_school_choices : ℕ := (choose members_per_school 1) ^ 2

/-- Represents the total number of ways to arrange the presidency meeting --/
def total_arrangements : ℕ := num_schools * host_school_choices * non_host_school_choices

theorem presidency_meeting_arrangements :
  total_arrangements = 750 :=
sorry

end NUMINAMATH_CALUDE_presidency_meeting_arrangements_l379_37922


namespace NUMINAMATH_CALUDE_circle_radius_and_diameter_l379_37955

theorem circle_radius_and_diameter 
  (M N : ℝ) 
  (h_area : M = π * r^2) 
  (h_circumference : N = 2 * π * r) 
  (h_ratio : M / N = 15) : 
  r = 30 ∧ 2 * r = 60 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_and_diameter_l379_37955


namespace NUMINAMATH_CALUDE_series_sum_equals_three_fourths_l379_37993

theorem series_sum_equals_three_fourths : 
  ∑' k, (k : ℝ) / (3 : ℝ) ^ k = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_three_fourths_l379_37993


namespace NUMINAMATH_CALUDE_P_subset_M_l379_37939

def M : Set ℕ := {0, 2}

def P : Set ℕ := {x | x ∈ M}

theorem P_subset_M : P ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_P_subset_M_l379_37939


namespace NUMINAMATH_CALUDE_solve_equation_l379_37958

theorem solve_equation : ∃ x : ℝ, 3 * x + 36 = 48 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l379_37958


namespace NUMINAMATH_CALUDE_product_evaluation_l379_37916

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l379_37916


namespace NUMINAMATH_CALUDE_business_loss_l379_37973

/-- Proves that the total loss in a business partnership is 1600 given the specified conditions -/
theorem business_loss (ashok_capital pyarelal_capital pyarelal_loss : ℚ) : 
  ashok_capital = (1 : ℚ) / 9 * pyarelal_capital →
  pyarelal_loss = 1440 →
  ashok_capital / pyarelal_capital * pyarelal_loss + pyarelal_loss = 1600 :=
by sorry

end NUMINAMATH_CALUDE_business_loss_l379_37973


namespace NUMINAMATH_CALUDE_softball_team_savings_l379_37967

/-- Calculates the savings for a softball team buying uniforms with a group discount. -/
theorem softball_team_savings 
  (team_size : ℕ) 
  (regular_shirt_price regular_pants_price regular_socks_price : ℚ)
  (discount_shirt_price discount_pants_price discount_socks_price : ℚ)
  (h1 : team_size = 12)
  (h2 : regular_shirt_price = 7.5)
  (h3 : regular_pants_price = 15)
  (h4 : regular_socks_price = 4.5)
  (h5 : discount_shirt_price = 6.75)
  (h6 : discount_pants_price = 13.5)
  (h7 : discount_socks_price = 3.75) :
  (team_size : ℚ) * ((regular_shirt_price + regular_pants_price + regular_socks_price) - 
  (discount_shirt_price + discount_pants_price + discount_socks_price)) = 36 :=
by sorry

end NUMINAMATH_CALUDE_softball_team_savings_l379_37967


namespace NUMINAMATH_CALUDE_unique_prime_solution_l379_37948

/-- The equation p^2 - 6pq + q^2 + 3q - 1 = 0 has only one solution in prime numbers. -/
theorem unique_prime_solution :
  ∃! (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p^2 - 6*p*q + q^2 + 3*q - 1 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l379_37948


namespace NUMINAMATH_CALUDE_max_e_is_one_l379_37972

/-- The sequence b_n defined as (8^n - 1) / 7 -/
def b (n : ℕ) : ℤ := (8^n - 1) / 7

/-- The greatest common divisor of b_n and b_(n+1) -/
def e (n : ℕ) : ℕ := Nat.gcd (Int.natAbs (b n)) (Int.natAbs (b (n + 1)))

/-- Theorem: The maximum value of e_n is always 1 -/
theorem max_e_is_one : ∀ n : ℕ, e n = 1 := by sorry

end NUMINAMATH_CALUDE_max_e_is_one_l379_37972


namespace NUMINAMATH_CALUDE_min_value_quadratic_l379_37982

theorem min_value_quadratic (x : ℝ) :
  let f : ℝ → ℝ := λ x => 3 * x^2 + 18 * x + 7
  ∃ (min_val : ℝ), (∀ x, f x ≥ min_val) ∧ (min_val = -20) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l379_37982


namespace NUMINAMATH_CALUDE_max_production_years_l379_37981

/-- The cumulative production function after n years -/
def f (n : ℕ) : ℚ := (1/2) * n * (n + 1) * (2 * n + 1)

/-- The annual production function -/
def annual_production (n : ℕ) : ℚ := 
  if n = 1 then f 1 else f n - f (n - 1)

/-- The maximum allowed annual production -/
def max_allowed_production : ℚ := 150

/-- The maximum number of years the production line can operate -/
def max_years : ℕ := 7

theorem max_production_years : 
  (∀ n : ℕ, n ≤ max_years → annual_production n ≤ max_allowed_production) ∧
  (annual_production (max_years + 1) > max_allowed_production) :=
sorry

end NUMINAMATH_CALUDE_max_production_years_l379_37981


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l379_37923

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + 3*c)) + (b / (8*c + 4*a)) + (9*c / (3*a + 2*b)) ≥ 47/48 :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + 3*c)) + (b / (8*c + 4*a)) + (9*c / (3*a + 2*b)) = 47/48 ↔ 
  ∃ (k : ℝ), k > 0 ∧ a = 10*k ∧ b = 21*k ∧ c = k :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l379_37923


namespace NUMINAMATH_CALUDE_florist_fertilizer_usage_l379_37933

/-- A florist's fertilizer usage problem -/
theorem florist_fertilizer_usage 
  (daily_usage : ℝ) 
  (num_days : ℕ) 
  (total_usage : ℝ) 
  (h1 : daily_usage = 2) 
  (h2 : num_days = 9) 
  (h3 : total_usage = 22) : 
  total_usage - (daily_usage * num_days) = 4 := by
sorry

end NUMINAMATH_CALUDE_florist_fertilizer_usage_l379_37933


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l379_37913

theorem quadratic_solution_sum (x : ℝ → Prop) :
  (∀ x, x * (4 * x - 5) = -4) →
  (∃ m n p : ℕ,
    (Nat.gcd m (Nat.gcd n p) = 1) ∧
    (∀ x, x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p)) →
  (∃ m n p : ℕ, m + n + p = 52) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l379_37913


namespace NUMINAMATH_CALUDE_solve_equations_l379_37905

theorem solve_equations :
  (∃ x : ℝ, 1 - 3 * (1 - x) = 2 * x ∧ x = 2) ∧
  (∃ x : ℝ, (3 * x + 1) / 2 - (4 * x - 2) / 5 = 1 ∧ x = 1 / 7) :=
by sorry

end NUMINAMATH_CALUDE_solve_equations_l379_37905


namespace NUMINAMATH_CALUDE_complex_sum_equality_l379_37941

theorem complex_sum_equality : 
  let A : ℂ := 2 + I
  let O : ℂ := -4
  let P : ℂ := -I
  let S : ℂ := 2 + 4*I
  A - O + P + S = 8 + 4*I :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l379_37941


namespace NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l379_37965

theorem cos_arcsin_eight_seventeenths : 
  Real.cos (Real.arcsin (8/17)) = 15/17 := by
  sorry

end NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l379_37965


namespace NUMINAMATH_CALUDE_extra_sodas_l379_37944

/-- Given that Robin bought 11 sodas and drank 3 sodas, prove that the number of extra sodas is 8. -/
theorem extra_sodas (total : ℕ) (drank : ℕ) (h1 : total = 11) (h2 : drank = 3) :
  total - drank = 8 := by
  sorry

end NUMINAMATH_CALUDE_extra_sodas_l379_37944


namespace NUMINAMATH_CALUDE_johnson_class_activity_c_contribution_l379_37925

/-- Calculates the individual student contribution for an activity -/
def individualContribution (totalCost classFunds numStudents : ℚ) : ℚ :=
  (totalCost - classFunds) / numStudents

/-- Proves that Mrs. Johnson's class individual contribution for Activity C is $3.60 -/
theorem johnson_class_activity_c_contribution :
  let totalCost : ℚ := 150
  let classFunds : ℚ := 60
  let numStudents : ℚ := 25
  individualContribution totalCost classFunds numStudents = 3.60 := by
sorry

end NUMINAMATH_CALUDE_johnson_class_activity_c_contribution_l379_37925


namespace NUMINAMATH_CALUDE_spider_journey_l379_37911

theorem spider_journey (r : ℝ) (leg : ℝ) (h1 : r = 65) (h2 : leg = 90) :
  let diameter := 2 * r
  let hypotenuse := diameter
  let other_leg := Real.sqrt (hypotenuse ^ 2 - leg ^ 2)
  hypotenuse + leg + other_leg = 220 + 20 * Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_spider_journey_l379_37911


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l379_37949

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 8/15) (h2 : x - y = 1/45) : x^2 - y^2 = 8/675 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l379_37949


namespace NUMINAMATH_CALUDE_correct_ways_to_spend_l379_37920

/-- Represents the number of magazines costing 2 yuan -/
def magazines_2yuan : ℕ := 8

/-- Represents the number of magazines costing 1 yuan -/
def magazines_1yuan : ℕ := 3

/-- Represents the total budget in yuan -/
def budget : ℕ := 10

/-- Calculates the number of ways to select magazines to spend exactly the budget -/
def ways_to_spend_budget : ℕ := sorry

theorem correct_ways_to_spend : ways_to_spend_budget = 266 := by sorry

end NUMINAMATH_CALUDE_correct_ways_to_spend_l379_37920


namespace NUMINAMATH_CALUDE_k_range_l379_37985

theorem k_range (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k^2 - 1 ≤ 0) ↔ k ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_k_range_l379_37985


namespace NUMINAMATH_CALUDE_modified_cube_surface_area_l379_37961

/-- Represents the structure of the cube after modifications -/
structure ModifiedCube where
  initialSize : Nat
  smallCubeSize : Nat
  removedCubes : Nat
  remainingCubes : Nat

/-- Calculates the surface area of the modified cube structure -/
def surfaceArea (cube : ModifiedCube) : Nat :=
  sorry

/-- Theorem stating that the surface area of the specific modified cube is 2820 -/
theorem modified_cube_surface_area :
  let cube : ModifiedCube := {
    initialSize := 12,
    smallCubeSize := 3,
    removedCubes := 14,
    remainingCubes := 50
  }
  surfaceArea cube = 2820 := by
  sorry

end NUMINAMATH_CALUDE_modified_cube_surface_area_l379_37961


namespace NUMINAMATH_CALUDE_initial_marbles_count_initial_marbles_proof_l379_37951

def marbles_to_juan : ℕ := 1835
def marbles_to_lisa : ℕ := 985
def marbles_left : ℕ := 5930

theorem initial_marbles_count : ℕ :=
  marbles_to_juan + marbles_to_lisa + marbles_left

#check initial_marbles_count

theorem initial_marbles_proof : initial_marbles_count = 8750 := by
  sorry

end NUMINAMATH_CALUDE_initial_marbles_count_initial_marbles_proof_l379_37951


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l379_37931

theorem complex_fraction_simplification : 
  ((12^4 + 324) * (24^4 + 324) * (36^4 + 324) * (48^4 + 324) * (60^4 + 324) * (72^4 + 324)) / 
  ((6^4 + 324) * (18^4 + 324) * (30^4 + 324) * (42^4 + 324) * (54^4 + 324) * (66^4 + 324)) = 313 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l379_37931


namespace NUMINAMATH_CALUDE_triangle_angles_l379_37904

theorem triangle_angles (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = 108 →          -- One angle is 108°
  b = 2 * c →        -- One angle is twice the other
  (b = 48 ∧ c = 24)  -- The two smaller angles are 48° and 24°
  := by sorry

end NUMINAMATH_CALUDE_triangle_angles_l379_37904


namespace NUMINAMATH_CALUDE_discount_equation_proof_l379_37908

theorem discount_equation_proof (a : ℝ) : 
  (200 * (1 - a / 100)^2 = 148) ↔ 
  (∃ (original_price final_price : ℝ),
    original_price = 200 ∧
    final_price = 148 ∧
    final_price = original_price * (1 - a / 100)^2) :=
by sorry

end NUMINAMATH_CALUDE_discount_equation_proof_l379_37908


namespace NUMINAMATH_CALUDE_fraction_simplification_l379_37997

theorem fraction_simplification (m : ℝ) (h : m ≠ 3 ∧ m ≠ -3) :
  (m^2 - 3*m) / (9 - m^2) = -m / (m + 3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l379_37997
