import Mathlib

namespace NUMINAMATH_CALUDE_max_consecutive_sum_l3504_350430

/-- The sum of n consecutive integers starting from k -/
def consecutiveSum (n k : ℕ) : ℕ := n * (2 * k + (n - 1)) / 2

/-- The maximum number of consecutive positive integers starting from 3 
    that can be added together before the sum exceeds 500 -/
theorem max_consecutive_sum : 
  (∀ m : ℕ, m ≤ 29 → consecutiveSum m 3 ≤ 500) ∧ 
  consecutiveSum 30 3 > 500 := by
  sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_l3504_350430


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3504_350498

-- Problem 1
theorem simplify_expression_1 (x : ℝ) : 6 * (2 * x - 1) - 3 * (5 + 2 * x) = 6 * x - 21 := by
  sorry

-- Problem 2
theorem simplify_expression_2 (a : ℝ) : (4 * a^2 - 8 * a - 9) + 3 * (2 * a^2 - 2 * a - 5) = 10 * a^2 - 14 * a - 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3504_350498


namespace NUMINAMATH_CALUDE_union_p_complement_q_l3504_350456

-- Define the set P
def P : Set ℝ := {x | ∃ y, y = Real.sqrt (-x^2 + 4*x - 3)}

-- Define the set Q
def Q : Set ℝ := {x | x^2 < 4}

-- Theorem statement
theorem union_p_complement_q :
  P ∪ (Set.univ \ Q) = Set.Iic (-2) ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_union_p_complement_q_l3504_350456


namespace NUMINAMATH_CALUDE_mod_eight_equivalence_l3504_350466

theorem mod_eight_equivalence (m : ℕ) : 
  13^7 ≡ m [ZMOD 8] → 0 ≤ m → m < 8 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_mod_eight_equivalence_l3504_350466


namespace NUMINAMATH_CALUDE_exam_score_calculation_l3504_350405

theorem exam_score_calculation (total_questions : ℕ) (total_marks : ℕ) (correct_answers : ℕ) 
  (h1 : total_questions = 60)
  (h2 : total_marks = 150)
  (h3 : correct_answers = 42)
  (h4 : total_questions = correct_answers + (total_questions - correct_answers)) :
  ∃ (marks_per_correct : ℕ), 
    marks_per_correct * correct_answers - (total_questions - correct_answers) = total_marks ∧ 
    marks_per_correct = 4 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l3504_350405


namespace NUMINAMATH_CALUDE_initial_amount_was_21_l3504_350404

/-- The initial amount of money in the cookie jar -/
def initial_amount : ℕ := sorry

/-- The amount Doris spent -/
def doris_spent : ℕ := 6

/-- The amount Martha spent -/
def martha_spent : ℕ := doris_spent / 2

/-- The amount left in the cookie jar after spending -/
def amount_left : ℕ := 12

/-- Theorem stating that the initial amount in the cookie jar was 21 dollars -/
theorem initial_amount_was_21 : initial_amount = 21 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_was_21_l3504_350404


namespace NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l3504_350429

theorem square_sum_given_product_and_sum (p q : ℝ) 
  (h1 : p * q = 12) 
  (h2 : p + q = 8) : 
  p^2 + q^2 = 40 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l3504_350429


namespace NUMINAMATH_CALUDE_arithmetic_sequence_60th_term_l3504_350444

/-- An arithmetic sequence is defined by its first term and common difference. -/
structure ArithmeticSequence where
  a₁ : ℚ
  d : ℚ

/-- The nth term of an arithmetic sequence. -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.a₁ + (n - 1 : ℚ) * seq.d

theorem arithmetic_sequence_60th_term
  (seq : ArithmeticSequence)
  (h₁ : seq.a₁ = 6)
  (h₁₃ : seq.nthTerm 13 = 32) :
  seq.nthTerm 60 = 803 / 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_60th_term_l3504_350444


namespace NUMINAMATH_CALUDE_toothpicks_for_2003_base_l3504_350406

def small_triangles (n : ℕ) : ℕ := n * (n + 1) / 2

def toothpicks (base : ℕ) : ℕ :=
  let total_triangles := small_triangles base
  3 * total_triangles / 2

theorem toothpicks_for_2003_base :
  toothpicks 2003 = 3010554 :=
by sorry

end NUMINAMATH_CALUDE_toothpicks_for_2003_base_l3504_350406


namespace NUMINAMATH_CALUDE_egg_grouping_l3504_350464

theorem egg_grouping (total_eggs : ℕ) (eggs_per_group : ℕ) (groups : ℕ) : 
  total_eggs = 8 → eggs_per_group = 2 → groups = total_eggs / eggs_per_group → groups = 4 := by
  sorry

end NUMINAMATH_CALUDE_egg_grouping_l3504_350464


namespace NUMINAMATH_CALUDE_f_value_at_2_l3504_350482

-- Define the function f
def f (x a b : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem f_value_at_2 (a b : ℝ) : 
  (f (-2) a b = 3) → (f 2 a b = -19) := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_2_l3504_350482


namespace NUMINAMATH_CALUDE_fraction_problem_l3504_350407

theorem fraction_problem (N : ℝ) (f : ℝ) : 
  N = 180 → 
  (1/2 * f * 1/5 * N) + 6 = 1/15 * N → 
  f = 1/3 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l3504_350407


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3504_350414

theorem quadratic_factorization :
  ∀ x : ℝ, 2 * x^2 - 10 * x - 12 = 2 * (x - 6) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3504_350414


namespace NUMINAMATH_CALUDE_chord_length_polar_l3504_350499

theorem chord_length_polar (ρ θ : ℝ) : 
  ρ = 4 * Real.sin θ → θ = π / 4 → ρ = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_polar_l3504_350499


namespace NUMINAMATH_CALUDE_parabola_coefficient_l3504_350481

/-- Given a parabola y = ax^2 + bx + c with vertex (p, kp) and y-intercept (0, -kp),
    where p ≠ 0 and k is a non-zero constant, prove that b = 4k/p -/
theorem parabola_coefficient (a b c p k : ℝ) (h1 : p ≠ 0) (h2 : k ≠ 0) : 
  (∀ x, a * x^2 + b * x + c = a * (x - p)^2 + k * p) →
  (a * 0^2 + b * 0 + c = -k * p) →
  b = 4 * k / p := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l3504_350481


namespace NUMINAMATH_CALUDE_smallest_sum_with_same_probability_l3504_350410

/-- Represents a symmetrical die with faces numbered 1 to 6 -/
structure SymmetricalDie :=
  (faces : Fin 6)

/-- Represents a set of symmetrical dice -/
def DiceSet := List SymmetricalDie

/-- The probability of getting a specific sum when throwing the dice -/
def probability (d : DiceSet) (sum : Nat) : ℝ :=
  sorry

/-- The condition that the sum 2022 is possible with a positive probability -/
def sum_2022_possible (d : DiceSet) : Prop :=
  ∃ p : ℝ, p > 0 ∧ probability d 2022 = p

/-- The theorem stating the smallest possible sum with the same probability as 2022 -/
theorem smallest_sum_with_same_probability (d : DiceSet) 
  (h : sum_2022_possible d) : 
  ∃ p : ℝ, p > 0 ∧ 
    probability d 2022 = p ∧
    probability d 337 = p ∧
    ∀ (sum : Nat), sum < 337 → probability d sum < p :=
  sorry

end NUMINAMATH_CALUDE_smallest_sum_with_same_probability_l3504_350410


namespace NUMINAMATH_CALUDE_set_a_equals_set_b_l3504_350425

/-- A positive integer that is not a perfect square -/
structure NonSquare (a : ℕ) : Prop where
  pos : 0 < a
  not_square : ∀ n : ℕ, n^2 ≠ a

/-- The equation k = (x^2 - a) / (x^2 - y^2) has a solution in ℤ^2 -/
def HasSolution (k a : ℕ) : Prop :=
  ∃ x y : ℤ, k = (x^2 - a) / (x^2 - y^2)

/-- The set of positive integers k for which the equation has a solution with x > √a -/
def SetA (a : ℕ) : Set ℕ :=
  {k : ℕ | k > 0 ∧ ∃ x y : ℤ, x^2 > a ∧ HasSolution k a}

/-- The set of positive integers k for which the equation has a solution with 0 ≤ x < √a -/
def SetB (a : ℕ) : Set ℕ :=
  {k : ℕ | k > 0 ∧ ∃ x y : ℤ, 0 ≤ x^2 ∧ x^2 < a ∧ HasSolution k a}

/-- The main theorem: Set A equals Set B for any non-square positive integer a -/
theorem set_a_equals_set_b (a : ℕ) (h : NonSquare a) : SetA a = SetB a := by
  sorry

end NUMINAMATH_CALUDE_set_a_equals_set_b_l3504_350425


namespace NUMINAMATH_CALUDE_half_diamond_four_thirds_l3504_350483

-- Define the diamond operation
noncomputable def diamond (a b : ℝ) : ℝ := sorry

-- Axioms
axiom diamond_def {a b : ℝ} (ha : 0 < a) (hb : 0 < b) : 
  diamond (a * b) b = a * (diamond b b)

axiom diamond_identity {a : ℝ} (ha : 0 < a) : 
  diamond (diamond a 1) a = diamond a 1

axiom diamond_one : diamond 1 1 = 1

-- Theorem to prove
theorem half_diamond_four_thirds : 
  diamond (1/2) (4/3) = 2/3 := by sorry

end NUMINAMATH_CALUDE_half_diamond_four_thirds_l3504_350483


namespace NUMINAMATH_CALUDE_expand_binomials_l3504_350408

theorem expand_binomials (x : ℝ) : (2*x - 3) * (x + 2) = 2*x^2 + x - 6 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomials_l3504_350408


namespace NUMINAMATH_CALUDE_expensive_coat_savings_l3504_350494

/-- Represents a coat with its cost and lifespan. -/
structure Coat where
  cost : ℕ
  lifespan : ℕ

/-- Calculates the total cost of a coat over a given period. -/
def totalCost (coat : Coat) (period : ℕ) : ℕ :=
  (period + coat.lifespan - 1) / coat.lifespan * coat.cost

/-- Proves that buying the more expensive coat saves $120 over 30 years. -/
theorem expensive_coat_savings :
  let expensiveCoat : Coat := { cost := 300, lifespan := 15 }
  let cheapCoat : Coat := { cost := 120, lifespan := 5 }
  let period : ℕ := 30
  totalCost cheapCoat period - totalCost expensiveCoat period = 120 := by
  sorry


end NUMINAMATH_CALUDE_expensive_coat_savings_l3504_350494


namespace NUMINAMATH_CALUDE_trapezoid_area_is_787_5_l3504_350415

/-- Represents a trapezoid ABCD with given measurements -/
structure Trapezoid where
  ab : ℝ
  bc : ℝ
  ad : ℝ
  altitude : ℝ
  slant_height : ℝ

/-- Calculates the area of the trapezoid -/
def trapezoid_area (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating that the area of the given trapezoid is 787.5 -/
theorem trapezoid_area_is_787_5 (t : Trapezoid) 
  (h_ab : t.ab = 40)
  (h_bc : t.bc = 30)
  (h_ad : t.ad = 17)
  (h_altitude : t.altitude = 15)
  (h_slant_height : t.slant_height = 34) :
  trapezoid_area t = 787.5 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_787_5_l3504_350415


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3504_350459

theorem solve_linear_equation (x : ℝ) : 3 * x + 7 = -2 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3504_350459


namespace NUMINAMATH_CALUDE_largest_last_digit_l3504_350419

/-- A string of digits satisfying the given conditions -/
structure DigitString where
  digits : Fin 2050 → Nat
  first_digit_is_two : digits 0 = 2
  divisibility_condition : ∀ i : Fin 2049, 
    (digits i * 10 + digits (i + 1)) % 17 = 0 ∨ 
    (digits i * 10 + digits (i + 1)) % 29 = 0

/-- The theorem stating that the largest possible last digit is 8 -/
theorem largest_last_digit (s : DigitString) : 
  s.digits 2049 ≤ 8 ∧ ∃ s : DigitString, s.digits 2049 = 8 := by
  sorry


end NUMINAMATH_CALUDE_largest_last_digit_l3504_350419


namespace NUMINAMATH_CALUDE_dividend_calculation_l3504_350468

theorem dividend_calculation (divisor quotient remainder : ℕ) : 
  divisor = 20 * quotient →
  divisor = 10 * remainder →
  remainder = 100 →
  divisor * quotient + remainder = 50100 :=
by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3504_350468


namespace NUMINAMATH_CALUDE_problem_statement_l3504_350412

theorem problem_statement (x y : ℝ) 
  (h1 : 2 * x - y = 1) 
  (h2 : x * y = 2) : 
  4 * x^3 * y - 4 * x^2 * y^2 + x * y^3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3504_350412


namespace NUMINAMATH_CALUDE_circle_symmetry_l3504_350433

/-- Given a circle with equation x^2 + y^2 + 2x - 4y + 4 = 0 that is symmetric about the line y = 2x + b, prove that b = 4 -/
theorem circle_symmetry (b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 4 = 0 → 
    ∃ x' y' : ℝ, x'^2 + y'^2 + 2*x' - 4*y' + 4 = 0 ∧ 
    y' = 2*x' + b ∧ 
    (x - x')^2 + (y - y')^2 = (x - x')^2 + ((2*x + b) - (2*x' + b))^2) →
  b = 4 := by
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l3504_350433


namespace NUMINAMATH_CALUDE_quiz_competition_participants_l3504_350402

theorem quiz_competition_participants (initial_participants : ℕ) : 
  (initial_participants * 40 / 100 * 1 / 4 = 30) →
  initial_participants = 300 := by
sorry

end NUMINAMATH_CALUDE_quiz_competition_participants_l3504_350402


namespace NUMINAMATH_CALUDE_quadratic_root_property_l3504_350403

theorem quadratic_root_property (a : ℝ) : 
  a^2 + 3*a - 1010 = 0 → 2*a^2 + 6*a + 4 = 2024 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l3504_350403


namespace NUMINAMATH_CALUDE_vector_dot_product_collinear_l3504_350436

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v = (t * w.1, t * w.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem vector_dot_product_collinear :
  ∀ (k : ℝ),
  let m : ℝ × ℝ := (2 * k - 1, k)
  let n : ℝ × ℝ := (4, 1)
  collinear m n → dot_product m n = -17/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_collinear_l3504_350436


namespace NUMINAMATH_CALUDE_cone_lateral_surface_angle_l3504_350400

theorem cone_lateral_surface_angle (r h : ℝ) (h_positive : r > 0 ∧ h > 0) :
  (π * r * (r + (r^2 + h^2).sqrt) = 3 * π * r^2) →
  (2 * π * r / (r^2 + h^2).sqrt : ℝ) = π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_angle_l3504_350400


namespace NUMINAMATH_CALUDE_johns_investment_l3504_350471

theorem johns_investment (total_investment : ℝ) (rate_a rate_b : ℝ) (investment_a : ℝ) (final_amount : ℝ) :
  total_investment = 1500 →
  rate_a = 0.04 →
  rate_b = 0.06 →
  investment_a = 750 →
  final_amount = 1575 →
  investment_a * (1 + rate_a) + (total_investment - investment_a) * (1 + rate_b) = final_amount :=
by sorry

end NUMINAMATH_CALUDE_johns_investment_l3504_350471


namespace NUMINAMATH_CALUDE_jessica_remaining_money_l3504_350475

/-- Given Jessica's initial amount and spending, calculate the remaining amount -/
theorem jessica_remaining_money (initial : ℚ) (spent : ℚ) (remaining : ℚ) : 
  initial = 11.73 ∧ spent = 10.22 ∧ remaining = initial - spent → remaining = 1.51 := by
  sorry

end NUMINAMATH_CALUDE_jessica_remaining_money_l3504_350475


namespace NUMINAMATH_CALUDE_no_prime_solution_l3504_350438

theorem no_prime_solution : ¬∃ (p q : ℕ), Prime p ∧ Prime q ∧ p > 5 ∧ q > 5 ∧ (p * q ∣ (5^p - 2^p) * (5^q - 2^q)) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_solution_l3504_350438


namespace NUMINAMATH_CALUDE_base_5_reversed_in_base_7_l3504_350460

/-- Converts a base 5 number to base 10 -/
def toBase10FromBase5 (a b c : Nat) : Nat :=
  25 * a + 5 * b + c

/-- Converts a base 7 number to base 10 -/
def toBase10FromBase7 (a b c : Nat) : Nat :=
  49 * c + 7 * b + a

/-- Checks if a number is a valid digit in base 5 -/
def isValidBase5Digit (n : Nat) : Prop :=
  n ≤ 4

theorem base_5_reversed_in_base_7 :
  ∃! (a₁ b₁ c₁ a₂ b₂ c₂ : Nat),
    isValidBase5Digit a₁ ∧ isValidBase5Digit b₁ ∧ isValidBase5Digit c₁ ∧
    isValidBase5Digit a₂ ∧ isValidBase5Digit b₂ ∧ isValidBase5Digit c₂ ∧
    toBase10FromBase5 a₁ b₁ c₁ = toBase10FromBase7 c₁ b₁ a₁ ∧
    toBase10FromBase5 a₂ b₂ c₂ = toBase10FromBase7 c₂ b₂ a₂ ∧
    a₁ ≠ 0 ∧ a₂ ≠ 0 ∧
    toBase10FromBase5 a₁ b₁ c₁ + toBase10FromBase5 a₂ b₂ c₂ = 153 :=
  sorry

end NUMINAMATH_CALUDE_base_5_reversed_in_base_7_l3504_350460


namespace NUMINAMATH_CALUDE_marble_arrangement_l3504_350426

def arrange_marbles (n : ℕ) (restricted_pairs : ℕ) : ℕ :=
  n.factorial - restricted_pairs * (n - 1).factorial

theorem marble_arrangement :
  arrange_marbles 5 1 = 72 := by sorry

end NUMINAMATH_CALUDE_marble_arrangement_l3504_350426


namespace NUMINAMATH_CALUDE_janice_purchase_l3504_350443

theorem janice_purchase (a b c : ℕ) : 
  a + b + c = 50 →
  30 * a + 200 * b + 300 * c = 5000 →
  a = 10 :=
by sorry

end NUMINAMATH_CALUDE_janice_purchase_l3504_350443


namespace NUMINAMATH_CALUDE_triangle_circle_perimeter_triangle_circle_perimeter_proof_l3504_350422

/-- The total perimeter of a right triangle with legs 3 and 4, and its inscribed circle -/
theorem triangle_circle_perimeter : ℝ → Prop :=
  fun total_perimeter =>
    ∃ (hypotenuse radius : ℝ),
      -- Triangle properties
      hypotenuse^2 = 3^2 + 4^2 ∧
      -- Circle properties
      radius > 0 ∧
      -- Area of triangle equals semiperimeter times radius
      (3 * 4 / 2 : ℝ) = ((3 + 4 + hypotenuse) / 2) * radius ∧
      -- Total perimeter calculation
      total_perimeter = (3 + 4 + hypotenuse) + 2 * Real.pi * radius ∧
      total_perimeter = 12 + 2 * Real.pi

/-- Proof of the theorem -/
theorem triangle_circle_perimeter_proof : triangle_circle_perimeter (12 + 2 * Real.pi) := by
  sorry

#check triangle_circle_perimeter_proof

end NUMINAMATH_CALUDE_triangle_circle_perimeter_triangle_circle_perimeter_proof_l3504_350422


namespace NUMINAMATH_CALUDE_article_cost_price_l3504_350437

def cost_price : ℝ → Prop :=
  λ c => 
    ∃ s, 
      (s = 1.25 * c) ∧ 
      (s - 14.70 = 1.04 * c) ∧ 
      (c = 70)

theorem article_cost_price : 
  ∃ c, cost_price c :=
sorry

end NUMINAMATH_CALUDE_article_cost_price_l3504_350437


namespace NUMINAMATH_CALUDE_circle_area_half_radius_l3504_350413

/-- The area of a circle with radius 1/2 is π/4 -/
theorem circle_area_half_radius : 
  let r : ℚ := 1/2
  π * r^2 = π/4 := by sorry

end NUMINAMATH_CALUDE_circle_area_half_radius_l3504_350413


namespace NUMINAMATH_CALUDE_problem_statement_l3504_350490

theorem problem_statement (p : Prop) (q : Prop)
  (hp : p ↔ ∃ x₀ : ℝ, Real.exp x₀ ≤ 0)
  (hq : q ↔ ∀ x : ℝ, 2^x > x^2) :
  (¬p) ∨ q := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3504_350490


namespace NUMINAMATH_CALUDE_solve_ages_l3504_350451

/-- Represents the ages of people in the problem -/
structure Ages where
  rehana : ℕ
  phoebe : ℕ
  jacob : ℕ
  xander : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.rehana = 25 ∧
  ages.rehana + 5 = 3 * (ages.phoebe + 5) ∧
  ages.jacob = 3 * ages.phoebe / 5 ∧
  ages.xander = ages.rehana + ages.jacob - 4

/-- The theorem to prove -/
theorem solve_ages : 
  ∃ (ages : Ages), problem_conditions ages ∧ 
    ages.rehana = 25 ∧ 
    ages.phoebe = 5 ∧ 
    ages.jacob = 3 ∧ 
    ages.xander = 24 := by
  sorry

end NUMINAMATH_CALUDE_solve_ages_l3504_350451


namespace NUMINAMATH_CALUDE_fibonacci_problem_l3504_350440

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the property of arithmetic sequence for Fibonacci numbers
def is_arithmetic_seq (a b c : ℕ) : Prop :=
  fib c - fib b = fib b - fib a

-- Define the main theorem
theorem fibonacci_problem (b : ℕ) :
  is_arithmetic_seq (b - 3) b (b + 3) →
  (b - 3) + b + (b + 3) = 2253 →
  b = 751 := by
  sorry


end NUMINAMATH_CALUDE_fibonacci_problem_l3504_350440


namespace NUMINAMATH_CALUDE_chess_tournament_players_l3504_350479

/-- Represents the possible total scores recorded by the scorers -/
def possible_scores : List ℕ := [1979, 1980, 1984, 1985]

/-- Calculates the total number of games in a tournament with n players -/
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The number of players in the chess tournament is 45 -/
theorem chess_tournament_players : ∃ (n : ℕ), n = 45 ∧ 
  ∃ (score : ℕ), score ∈ possible_scores ∧ score = 2 * total_games n := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_players_l3504_350479


namespace NUMINAMATH_CALUDE_bench_arrangements_l3504_350486

theorem bench_arrangements (n : Nat) (h : n = 9) : Nat.factorial n = 362880 := by
  sorry

end NUMINAMATH_CALUDE_bench_arrangements_l3504_350486


namespace NUMINAMATH_CALUDE_problem_solution_l3504_350472

theorem problem_solution (m n : ℕ) 
  (h1 : m + 8 < n + 3)
  (h2 : (m + (m + 3) + (m + 8) + (n + 3) + (n + 4) + 2*n) / 6 = n)
  (h3 : ((m + 8) + (n + 3)) / 2 = n) : 
  m + n = 53 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3504_350472


namespace NUMINAMATH_CALUDE_babysitting_earnings_l3504_350445

def final_balance (hourly_rate : ℕ) (hours_worked : ℕ) (initial_balance : ℕ) : ℕ :=
  initial_balance + hourly_rate * hours_worked

theorem babysitting_earnings : final_balance 5 7 20 = 55 := by
  sorry

end NUMINAMATH_CALUDE_babysitting_earnings_l3504_350445


namespace NUMINAMATH_CALUDE_A_equals_set_l3504_350491

def A : Set ℕ := {x | 0 ≤ x ∧ x < 3}

theorem A_equals_set : A = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_A_equals_set_l3504_350491


namespace NUMINAMATH_CALUDE_ed_limpet_shells_l3504_350452

/-- The number of limpet shells Ed found -/
def L : ℕ := sorry

/-- The initial number of shells in the collection -/
def initial_shells : ℕ := 2

/-- The number of oyster shells Ed found -/
def ed_oyster_shells : ℕ := 2

/-- The number of conch shells Ed found -/
def ed_conch_shells : ℕ := 4

/-- The total number of shells Ed found -/
def ed_total_shells : ℕ := L + ed_oyster_shells + ed_conch_shells

/-- The total number of shells Jacob found -/
def jacob_total_shells : ℕ := ed_total_shells + 2

/-- The total number of shells in the final collection -/
def total_shells : ℕ := 30

theorem ed_limpet_shells :
  initial_shells + ed_total_shells + jacob_total_shells = total_shells ∧ L = 7 := by
  sorry

end NUMINAMATH_CALUDE_ed_limpet_shells_l3504_350452


namespace NUMINAMATH_CALUDE_mod_equivalence_l3504_350488

theorem mod_equivalence (n : ℕ) (h1 : n < 41) (h2 : (5 * n) % 41 = 1) :
  (((2 ^ n) ^ 3) - 3) % 41 = 6 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_l3504_350488


namespace NUMINAMATH_CALUDE_linear_function_proof_l3504_350493

/-- A linear function passing through points (1, -1) and (-2, 8) -/
def f (x : ℝ) : ℝ := -3 * x + 2

theorem linear_function_proof :
  (f 1 = -1) ∧ 
  (f (-2) = 8) ∧ 
  (∀ x : ℝ, f x = -3 * x + 2) ∧
  (f (-10) = 32) := by
  sorry


end NUMINAMATH_CALUDE_linear_function_proof_l3504_350493


namespace NUMINAMATH_CALUDE_petes_number_l3504_350465

theorem petes_number (x : ℚ) : 3 * (x + 15) - 5 = 125 → x = 85 / 3 := by
  sorry

end NUMINAMATH_CALUDE_petes_number_l3504_350465


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_iff_m_eq_one_l3504_350484

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

theorem complex_purely_imaginary_iff_m_eq_one (m : ℝ) :
  is_purely_imaginary (m^2 - m + m * Complex.I) ↔ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_iff_m_eq_one_l3504_350484


namespace NUMINAMATH_CALUDE_sum_of_first_ten_terms_l3504_350423

/-- Given a sequence {a_n} and its partial sum sequence {S_n}, prove that S_10 = 145 -/
theorem sum_of_first_ten_terms 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h1 : ∀ n, S (n + 1) = S n + a n + 3)
  (h2 : a 5 + a 6 = 29) :
  S 10 = 145 := by
sorry

end NUMINAMATH_CALUDE_sum_of_first_ten_terms_l3504_350423


namespace NUMINAMATH_CALUDE_inequality_proof_l3504_350416

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2018 + b^2018)^2019 > (a^2019 + b^2019)^2018 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3504_350416


namespace NUMINAMATH_CALUDE_hoseoks_number_l3504_350442

theorem hoseoks_number (x : ℤ) : x - 10 = 15 → x + 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_hoseoks_number_l3504_350442


namespace NUMINAMATH_CALUDE_binomial_coefficient_19_13_l3504_350487

theorem binomial_coefficient_19_13 (h1 : Nat.choose 18 11 = 31824)
                                   (h2 : Nat.choose 18 12 = 18564)
                                   (h3 : Nat.choose 20 13 = 77520) :
  Nat.choose 19 13 = 27132 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_19_13_l3504_350487


namespace NUMINAMATH_CALUDE_tara_ice_cream_yoghurt_spending_l3504_350431

theorem tara_ice_cream_yoghurt_spending :
  let ice_cream_cartons : ℕ := 19
  let yoghurt_cartons : ℕ := 4
  let ice_cream_price : ℕ := 7
  let yoghurt_price : ℕ := 1
  let ice_cream_total : ℕ := ice_cream_cartons * ice_cream_price
  let yoghurt_total : ℕ := yoghurt_cartons * yoghurt_price
  ice_cream_total - yoghurt_total = 129 :=
by sorry

end NUMINAMATH_CALUDE_tara_ice_cream_yoghurt_spending_l3504_350431


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l3504_350418

theorem rectangle_dimension_change (L B : ℝ) (h : L > 0 ∧ B > 0) :
  let L' := L * (1 + 30 / 100)
  let B' := B * (1 - 20 / 100)
  L' * B' = (L * B) * (1 + 4.0000000000000036 / 100) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l3504_350418


namespace NUMINAMATH_CALUDE_fraction_transformation_l3504_350495

theorem fraction_transformation (x y : ℝ) : 
  -(x - y) / (x + y) = (-x + y) / (x + y) :=
by sorry

end NUMINAMATH_CALUDE_fraction_transformation_l3504_350495


namespace NUMINAMATH_CALUDE_student_average_less_than_true_average_l3504_350435

theorem student_average_less_than_true_average 
  (w x y z : ℝ) (h : w < x ∧ x < y ∧ y < z) : 
  (2*w + 2*x + y + z) / 6 < (w + x + y + z) / 4 := by
sorry

end NUMINAMATH_CALUDE_student_average_less_than_true_average_l3504_350435


namespace NUMINAMATH_CALUDE_brand_x_pen_price_l3504_350457

/-- The price of a brand X pen given the total number of pens, total cost, number of brand X pens, and price of brand Y pens. -/
theorem brand_x_pen_price
  (total_pens : ℕ)
  (total_cost : ℚ)
  (brand_x_count : ℕ)
  (brand_y_price : ℚ)
  (h1 : total_pens = 12)
  (h2 : total_cost = 42)
  (h3 : brand_x_count = 6)
  (h4 : brand_y_price = 2.2)
  : (total_cost - (total_pens - brand_x_count) * brand_y_price) / brand_x_count = 4.8 := by
  sorry

#check brand_x_pen_price

end NUMINAMATH_CALUDE_brand_x_pen_price_l3504_350457


namespace NUMINAMATH_CALUDE_reciprocal_expression_l3504_350492

theorem reciprocal_expression (a b : ℝ) (h : a * b = 1) :
  a^2 * b - (a - 2023) = 2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_expression_l3504_350492


namespace NUMINAMATH_CALUDE_points_collinear_l3504_350480

/-- Three points are collinear if the slope between any two pairs of points is the same. -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)

theorem points_collinear : collinear (1, 2) (3, -2) (4, -4) := by
  sorry

end NUMINAMATH_CALUDE_points_collinear_l3504_350480


namespace NUMINAMATH_CALUDE_hash_2_3_neg1_l3504_350439

def hash (a b c : ℝ) : ℝ := b^3 - 4*a*c + b

theorem hash_2_3_neg1 : hash 2 3 (-1) = 38 := by
  sorry

end NUMINAMATH_CALUDE_hash_2_3_neg1_l3504_350439


namespace NUMINAMATH_CALUDE_base6_division_l3504_350434

/-- Converts a base 6 number to base 10 --/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 --/
def toBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- The quotient of 2134₆ divided by 14₆ is equal to 81₆ in base 6 --/
theorem base6_division :
  toBase6 (toBase10 [4, 3, 1, 2] / toBase10 [4, 1]) = [1, 8] := by
  sorry

end NUMINAMATH_CALUDE_base6_division_l3504_350434


namespace NUMINAMATH_CALUDE_football_team_progress_l3504_350448

/-- Given a football team's yard changes, calculate their progress -/
def teamProgress (lost : ℤ) (gained : ℤ) : ℤ :=
  gained - lost

theorem football_team_progress :
  let lost : ℤ := 5
  let gained : ℤ := 7
  teamProgress lost gained = 2 := by
  sorry

end NUMINAMATH_CALUDE_football_team_progress_l3504_350448


namespace NUMINAMATH_CALUDE_clock_twelve_strikes_l3504_350424

/-- Represents a grandfather clock with a given strike interval -/
structure GrandfatherClock where
  strike_interval : ℝ

/-- Calculates the time taken for a given number of strikes -/
def time_for_strikes (clock : GrandfatherClock) (num_strikes : ℕ) : ℝ :=
  clock.strike_interval * (num_strikes - 1)

theorem clock_twelve_strikes (clock : GrandfatherClock) 
  (h : time_for_strikes clock 6 = 30) :
  time_for_strikes clock 12 = 66 := by
  sorry


end NUMINAMATH_CALUDE_clock_twelve_strikes_l3504_350424


namespace NUMINAMATH_CALUDE_no_simultaneous_roots_one_and_neg_one_l3504_350455

theorem no_simultaneous_roots_one_and_neg_one :
  ¬ ∃ (a b : ℝ), (1 : ℝ)^3 + a * (1 : ℝ)^2 + b = 0 ∧ (-1 : ℝ)^3 + a * (-1 : ℝ)^2 + b = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_simultaneous_roots_one_and_neg_one_l3504_350455


namespace NUMINAMATH_CALUDE_product_pure_imaginary_solution_l3504_350467

theorem product_pure_imaginary_solution (x : ℝ) : 
  (∃ y : ℝ, (x + 2 * Complex.I) * ((x + 1) + 2 * Complex.I) * ((x + 2) + 2 * Complex.I) = y * Complex.I) ↔ 
  x = 1 :=
by sorry

end NUMINAMATH_CALUDE_product_pure_imaginary_solution_l3504_350467


namespace NUMINAMATH_CALUDE_pint_cost_is_eight_l3504_350477

/-- The cost of a pint of paint given the number of doors, cost of a gallon, and savings -/
def pint_cost (num_doors : ℕ) (gallon_cost : ℚ) (savings : ℚ) : ℚ :=
  (gallon_cost + savings) / num_doors

theorem pint_cost_is_eight :
  pint_cost 8 55 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_pint_cost_is_eight_l3504_350477


namespace NUMINAMATH_CALUDE_tenth_term_is_19_l3504_350441

/-- An arithmetic sequence with specified properties -/
structure ArithmeticSequence where
  /-- The sum of the first n terms -/
  S : ℕ → ℝ
  /-- The nth term of the sequence -/
  a : ℕ → ℝ
  /-- The sum of the first 9 terms is 81 -/
  sum_9 : S 9 = 81
  /-- The second term is 3 -/
  second_term : a 2 = 3
  /-- The sequence follows the arithmetic sequence property -/
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- The 10th term of the specified arithmetic sequence is 19 -/
theorem tenth_term_is_19 (seq : ArithmeticSequence) : seq.a 10 = 19 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_19_l3504_350441


namespace NUMINAMATH_CALUDE_biology_marks_calculation_l3504_350462

def english_marks : ℕ := 73
def math_marks : ℕ := 69
def physics_marks : ℕ := 92
def chemistry_marks : ℕ := 64
def average_marks : ℕ := 76
def total_subjects : ℕ := 5

theorem biology_marks_calculation : 
  (english_marks + math_marks + physics_marks + chemistry_marks + 
   (average_marks * total_subjects - (english_marks + math_marks + physics_marks + chemistry_marks))) 
  / total_subjects = average_marks :=
by sorry

end NUMINAMATH_CALUDE_biology_marks_calculation_l3504_350462


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3504_350489

def i : ℂ := Complex.I

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 7 * i
  let z₂ : ℂ := 4 - 7 * i
  (z₁ / z₂) + (z₂ / z₁) = -66 / 65 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3504_350489


namespace NUMINAMATH_CALUDE_fraction_order_l3504_350446

theorem fraction_order : (21 : ℚ) / 17 < 23 / 18 ∧ 23 / 18 < 25 / 19 := by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l3504_350446


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3504_350420

theorem complex_magnitude_problem (z : ℂ) (h : z * (1 + Complex.I * Real.sqrt 3) = 1) :
  Complex.abs z = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3504_350420


namespace NUMINAMATH_CALUDE_factorization_x4_minus_81_complete_factorization_l3504_350458

theorem factorization_x4_minus_81 (x : ℝ) :
  x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by sorry

theorem complete_factorization (p q r : ℝ → ℝ) :
  (∀ x, x^4 - 81 = p x * q x * r x) →
  (∀ x, p x = x - 3 ∨ p x = x + 3 ∨ p x = x^2 + 9) →
  (∀ x, q x = x - 3 ∨ q x = x + 3 ∨ q x = x^2 + 9) →
  (∀ x, r x = x - 3 ∨ r x = x + 3 ∨ r x = x^2 + 9) →
  (p ≠ q ∧ p ≠ r ∧ q ≠ r) →
  (∀ x, p x * q x * r x = (x - 3) * (x + 3) * (x^2 + 9)) :=
by sorry

end NUMINAMATH_CALUDE_factorization_x4_minus_81_complete_factorization_l3504_350458


namespace NUMINAMATH_CALUDE_hari_contribution_is_2160_l3504_350421

/-- Represents the investment details and profit sharing ratio --/
structure InvestmentDetails where
  praveen_investment : ℕ
  praveen_months : ℕ
  hari_months : ℕ
  total_months : ℕ
  profit_ratio_praveen : ℕ
  profit_ratio_hari : ℕ

/-- Calculates Hari's contribution given the investment details --/
def calculate_hari_contribution (details : InvestmentDetails) : ℕ :=
  (details.praveen_investment * details.praveen_months * details.profit_ratio_hari) /
  (details.profit_ratio_praveen * details.hari_months)

/-- Theorem stating that Hari's contribution is 2160 given the problem conditions --/
theorem hari_contribution_is_2160 :
  let details : InvestmentDetails := {
    praveen_investment := 3360,
    praveen_months := 12,
    hari_months := 7,
    total_months := 12,
    profit_ratio_praveen := 2,
    profit_ratio_hari := 3
  }
  calculate_hari_contribution details = 2160 := by
  sorry

#eval calculate_hari_contribution {
  praveen_investment := 3360,
  praveen_months := 12,
  hari_months := 7,
  total_months := 12,
  profit_ratio_praveen := 2,
  profit_ratio_hari := 3
}

end NUMINAMATH_CALUDE_hari_contribution_is_2160_l3504_350421


namespace NUMINAMATH_CALUDE_negative_abs_comparison_l3504_350476

theorem negative_abs_comparison : -|(-8 : ℤ)| < -6 := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_comparison_l3504_350476


namespace NUMINAMATH_CALUDE_problem_statement_l3504_350401

theorem problem_statement (A : ℤ) (h : A = 43^2011 - 2011^43) : 
  (3 ∣ A) ∧ (A % 11 = 7) ∧ (A % 35 = 6) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3504_350401


namespace NUMINAMATH_CALUDE_jason_lost_three_balloons_l3504_350469

/-- The number of violet balloons Jason lost -/
def lost_balloons (initial current : ℕ) : ℕ := initial - current

/-- Proof that Jason lost 3 violet balloons -/
theorem jason_lost_three_balloons :
  let initial_violet : ℕ := 7
  let current_violet : ℕ := 4
  lost_balloons initial_violet current_violet = 3 := by
  sorry

end NUMINAMATH_CALUDE_jason_lost_three_balloons_l3504_350469


namespace NUMINAMATH_CALUDE_quartic_roots_l3504_350411

/-- The value of N in the quartic equation --/
def N : ℝ := 10^10

/-- The quartic function --/
def f (x : ℝ) : ℝ := x^4 - (2*N + 1)*x^2 - x + N^2 + N - 1

/-- The first approximate root --/
def root1 : ℝ := 99999.9984

/-- The second approximate root --/
def root2 : ℝ := 100000.0016

/-- Theorem stating that the quartic equation has two approximate roots --/
theorem quartic_roots : 
  ∃ (r1 r2 : ℝ), 
    (abs (r1 - root1) < 0.00005) ∧ 
    (abs (r2 - root2) < 0.00005) ∧ 
    f r1 = 0 ∧ 
    f r2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quartic_roots_l3504_350411


namespace NUMINAMATH_CALUDE_chess_tournament_players_l3504_350449

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  n : ℕ  -- Number of players not in the lowest 12
  /-- Total number of players is n + 12 -/
  total_players : ℕ := n + 12
  /-- Each player played exactly one game against every other player -/
  total_games : ℕ := (total_players * (total_players - 1)) / 2
  /-- Points earned by n players not in the lowest 12 -/
  top_points : ℕ := n * (n - 1)
  /-- Points earned by 12 lowest-scoring players among themselves -/
  bottom_points : ℕ := 66
  /-- Total points earned in the tournament -/
  total_points : ℕ := total_games

/-- The theorem stating that the total number of players in the tournament is 34 -/
theorem chess_tournament_players : 
  ∀ t : ChessTournament, t.total_players = 34 := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_players_l3504_350449


namespace NUMINAMATH_CALUDE_bisected_polyhedron_edges_l3504_350496

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ

/-- Represents the new polyhedron after bisection -/
structure BisectedPolyhedron where
  original : ConvexPolyhedron
  planes : ℕ

/-- Calculate the number of edges in the bisected polyhedron -/
def edges_after_bisection (T : BisectedPolyhedron) : ℕ :=
  T.original.edges + 2 * T.original.edges

/-- Theorem stating the number of edges in the bisected polyhedron -/
theorem bisected_polyhedron_edges 
  (P : ConvexPolyhedron) 
  (h_vertices : P.vertices = 0)  -- placeholder for actual number of vertices
  (h_edges : P.edges = 150)
  (T : BisectedPolyhedron)
  (h_T : T.original = P)
  (h_planes : T.planes = P.vertices)
  : edges_after_bisection T = 450 := by
  sorry

#check bisected_polyhedron_edges

end NUMINAMATH_CALUDE_bisected_polyhedron_edges_l3504_350496


namespace NUMINAMATH_CALUDE_championship_probability_l3504_350470

def is_win_for_A (n : ℕ) : Bool :=
  n ≤ 5

def count_wins_A (numbers : List ℕ) : ℕ :=
  (numbers.filter is_win_for_A).length

def estimate_probability (wins : ℕ) (total : ℕ) : ℚ :=
  ↑wins / ↑total

def generated_numbers : List ℕ := [1, 9, 2, 9, 0, 7, 9, 6, 6, 9, 2, 5, 2, 7, 1, 9, 3, 2, 8, 1, 2, 6, 7, 3, 9, 3, 1, 2, 7, 5, 5, 6, 4, 8, 8, 7, 3, 0, 1, 1, 3, 5, 3, 7, 9, 8, 9, 4, 3, 1]

theorem championship_probability :
  estimate_probability (count_wins_A generated_numbers) generated_numbers.length = 13/20 := by
  sorry

end NUMINAMATH_CALUDE_championship_probability_l3504_350470


namespace NUMINAMATH_CALUDE_g_50_solutions_l3504_350461

def g₀ (x : ℝ) : ℝ := x + |x - 50| - |x + 50|

def g (n : ℕ) (x : ℝ) : ℝ := 
  match n with
  | 0 => g₀ x
  | n + 1 => |g n x| - 2

theorem g_50_solutions : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, g 50 x = 0) ∧ (∀ x ∉ S, g 50 x ≠ 0) ∧ Finset.card S = 4 := by
  sorry

end NUMINAMATH_CALUDE_g_50_solutions_l3504_350461


namespace NUMINAMATH_CALUDE_inverse_function_sum_l3504_350497

-- Define the functions g and g_inv
def g (c d : ℝ) (x : ℝ) : ℝ := c * x + d
def g_inv (c d : ℝ) (x : ℝ) : ℝ := d * x + c

-- State the theorem
theorem inverse_function_sum (c d : ℝ) :
  (∀ x : ℝ, g c d (g_inv c d x) = x) →
  (∀ x : ℝ, g_inv c d (g c d x) = x) →
  c + d = -2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_sum_l3504_350497


namespace NUMINAMATH_CALUDE_correct_statements_are_ACD_l3504_350417

-- Define the set of all statements
inductive Statement : Type
| A : Statement
| B : Statement
| C : Statement
| D : Statement

-- Define a function to check if a statement is correct
def is_correct : Statement → Prop
| Statement.A => ∀ (residual_width : ℝ) (fitting_quality : ℝ),
    residual_width < 0 → fitting_quality > 0
| Statement.B => ∀ (r_A r_B : ℝ),
    r_A = 0.97 ∧ r_B = -0.99 → abs r_A > abs r_B
| Statement.C => ∀ (R_squared fitting_quality : ℝ),
    R_squared < 0 → fitting_quality < 0
| Statement.D => ∀ (n k d : ℕ),
    n = 10 ∧ k = 2 ∧ d = 3 →
    (Nat.choose d 1 * Nat.choose (n - d) (k - 1)) / Nat.choose n k = 7 / 15

-- Define the set of correct statements
def correct_statements : Set Statement :=
  {s | is_correct s}

-- Theorem to prove
theorem correct_statements_are_ACD :
  correct_statements = {Statement.A, Statement.C, Statement.D} :=
sorry

end NUMINAMATH_CALUDE_correct_statements_are_ACD_l3504_350417


namespace NUMINAMATH_CALUDE_exam_score_calculation_l3504_350478

theorem exam_score_calculation (total_questions : ℕ) (correct_answers : ℕ) (total_marks : ℕ) 
  (h1 : total_questions = 60)
  (h2 : correct_answers = 40)
  (h3 : total_marks = 140)
  (h4 : correct_answers ≤ total_questions) :
  ∃ (marks_per_correct : ℕ), 
    marks_per_correct * correct_answers - (total_questions - correct_answers) = total_marks ∧ 
    marks_per_correct = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l3504_350478


namespace NUMINAMATH_CALUDE_original_sales_tax_percentage_l3504_350485

theorem original_sales_tax_percentage 
  (market_price : ℝ) 
  (new_tax_rate : ℝ) 
  (savings : ℝ) :
  market_price = 8400 ∧ 
  new_tax_rate = 10 / 3 ∧ 
  savings = 14 → 
  ∃ original_tax_rate : ℝ,
    original_tax_rate = 3.5 ∧
    market_price * (original_tax_rate / 100) = 
      market_price * (new_tax_rate / 100) + savings :=
by sorry

end NUMINAMATH_CALUDE_original_sales_tax_percentage_l3504_350485


namespace NUMINAMATH_CALUDE_original_price_calculation_l3504_350432

/-- Calculates the original price of an article given the profit percentage and profit amount. -/
def calculate_original_price (profit_percentage : ℚ) (profit_amount : ℚ) : ℚ :=
  profit_amount / (profit_percentage / 100)

/-- Theorem: Given an article sold at a 50% profit, where the profit is Rs. 750, 
    the original price of the article was Rs. 1500. -/
theorem original_price_calculation :
  calculate_original_price 50 750 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l3504_350432


namespace NUMINAMATH_CALUDE_expected_composite_count_l3504_350450

/-- The number of elements in the set {1, 2, 3, ..., 100} -/
def setSize : ℕ := 100

/-- The number of composite numbers in the set {1, 2, 3, ..., 100} -/
def compositeCount : ℕ := 74

/-- The number of selections made -/
def selectionCount : ℕ := 5

/-- The probability of selecting a composite number -/
def compositeProbability : ℚ := compositeCount / setSize

/-- Expected number of composite numbers when selecting 5 numbers with replacement from {1, 2, 3, ..., 100} -/
theorem expected_composite_count : 
  (selectionCount : ℚ) * compositeProbability = 37 / 10 := by sorry

end NUMINAMATH_CALUDE_expected_composite_count_l3504_350450


namespace NUMINAMATH_CALUDE_number_of_factors_24_l3504_350454

def factors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x => n % x = 0) (Finset.range (n + 1))

theorem number_of_factors_24 : (factors 24).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_24_l3504_350454


namespace NUMINAMATH_CALUDE_log8_three_point_five_equals_512_l3504_350474

-- Define the logarithm base 8
noncomputable def log8 (x : ℝ) : ℝ := Real.log x / Real.log 8

-- State the theorem
theorem log8_three_point_five_equals_512 :
  ∀ x : ℝ, x > 0 → log8 x = 3.5 → x = 512 := by
  sorry

end NUMINAMATH_CALUDE_log8_three_point_five_equals_512_l3504_350474


namespace NUMINAMATH_CALUDE_inequality_proof_l3504_350463

theorem inequality_proof (p q r x y θ : ℝ) :
  p * x^(q - y) + q * x^(r - y) + r * x^(y - θ) ≥ p + q + r := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3504_350463


namespace NUMINAMATH_CALUDE_fireworks_display_total_l3504_350473

/-- Calculate the total number of fireworks in a New Year's Eve display --/
def totalFireworks : ℕ :=
  let yearDigits : ℕ := 4
  let yearFireworksPerDigit : ℕ := 6
  let happyNewYearLetters : ℕ := 12
  let regularLetterFireworks : ℕ := 5
  let helloFireworks : ℕ := 8 + 7 + 6 + 6 + 9
  let additionalBoxes : ℕ := 100
  let fireworksPerBox : ℕ := 10

  yearDigits * yearFireworksPerDigit +
  happyNewYearLetters * regularLetterFireworks +
  helloFireworks +
  additionalBoxes * fireworksPerBox

theorem fireworks_display_total :
  totalFireworks = 1120 := by sorry

end NUMINAMATH_CALUDE_fireworks_display_total_l3504_350473


namespace NUMINAMATH_CALUDE_cubic_equation_root_l3504_350409

theorem cubic_equation_root (a b : ℚ) : 
  (2 + Real.sqrt 3 : ℝ) ^ 3 + a * (2 + Real.sqrt 3 : ℝ) ^ 2 + b * (2 + Real.sqrt 3 : ℝ) - 20 = 0 → 
  b = -79 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l3504_350409


namespace NUMINAMATH_CALUDE_function_properties_l3504_350427

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem function_properties (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_symmetry : ∀ x, f (4 - x) = f x) : 
  (∀ x, f (x + 8) = f x) ∧ (f 2019 + f 2020 + f 2021 = 0) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l3504_350427


namespace NUMINAMATH_CALUDE_diamond_15_25_l3504_350453

-- Define the ⋄ operation
noncomputable def diamond (x y : ℝ) : ℝ := 
  sorry

-- Define the properties of the ⋄ operation
axiom diamond_prop1 (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  diamond (x * y) y = x * (diamond y y)

axiom diamond_prop2 (x : ℝ) (hx : x > 0) : 
  diamond (diamond x 1) x = diamond x 1

axiom diamond_one : diamond 1 1 = 1

-- State the theorem to be proved
theorem diamond_15_25 : diamond 15 25 = 375 := by
  sorry

end NUMINAMATH_CALUDE_diamond_15_25_l3504_350453


namespace NUMINAMATH_CALUDE_integer_sum_proof_l3504_350447

theorem integer_sum_proof (x y : ℕ+) (h1 : x - y = 8) (h2 : x * y = 180) : x + y = 28 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_proof_l3504_350447


namespace NUMINAMATH_CALUDE_maria_fair_spending_l3504_350428

def fair_spending (initial_amount spent_on_rides discounted_ride_cost discount_percent
                   borrowed won food_cost found_money lent_money final_amount : ℚ) : Prop :=
  let discounted_ride_spending := discounted_ride_cost * (1 - discount_percent / 100)
  let net_amount := initial_amount - spent_on_rides - discounted_ride_spending + borrowed + won - food_cost + found_money - lent_money
  net_amount - final_amount = 41

theorem maria_fair_spending :
  fair_spending 87 25 4 25 15 10 12 5 20 16 := by sorry

end NUMINAMATH_CALUDE_maria_fair_spending_l3504_350428
