import Mathlib

namespace shortest_distance_to_circle_l1646_164687

def circle_equation (x y : ℝ) : Prop := (x - 8)^2 + (y - 7)^2 = 25

def point : ℝ × ℝ := (1, -2)

def center : ℝ × ℝ := (8, 7)

def radius : ℝ := 5

theorem shortest_distance_to_circle :
  let d := Real.sqrt ((point.1 - center.1)^2 + (point.2 - center.2)^2) - radius
  d = Real.sqrt 130 - 5 := by sorry

end shortest_distance_to_circle_l1646_164687


namespace min_stones_to_remove_is_ten_l1646_164628

/-- Represents a chessboard configuration -/
def Chessboard := Fin 7 → Fin 8 → Bool

/-- Checks if there are five adjacent stones in any direction -/
def hasFiveAdjacent (board : Chessboard) : Bool :=
  sorry

/-- Counts the number of stones on the board -/
def stoneCount (board : Chessboard) : Nat :=
  sorry

/-- The minimal number of stones that must be removed -/
def minStonesToRemove : Nat := 10

/-- Theorem stating the minimal number of stones to remove -/
theorem min_stones_to_remove_is_ten :
  ∀ (initial : Chessboard),
    stoneCount initial = 56 →
    ∀ (final : Chessboard),
      (¬ hasFiveAdjacent final) →
      (stoneCount initial - stoneCount final ≥ minStonesToRemove) ∧
      (∃ (optimal : Chessboard),
        (¬ hasFiveAdjacent optimal) ∧
        (stoneCount initial - stoneCount optimal = minStonesToRemove)) :=
  sorry

end min_stones_to_remove_is_ten_l1646_164628


namespace negation_cube_even_number_l1646_164615

theorem negation_cube_even_number (n : ℤ) :
  ¬(∀ n : ℤ, 2 ∣ n → 2 ∣ n^3) ↔ ∃ n : ℤ, 2 ∣ n ∧ ¬(2 ∣ n^3) :=
sorry

end negation_cube_even_number_l1646_164615


namespace factorization_a4_2a3_1_l1646_164691

theorem factorization_a4_2a3_1 (a : ℝ) : 
  a^4 + 2*a^3 + 1 = (a + 1) * (a^3 + a^2 - a + 1) := by sorry

end factorization_a4_2a3_1_l1646_164691


namespace biology_marks_calculation_l1646_164651

def english_marks : ℕ := 96
def math_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def average_marks : ℕ := 79
def total_subjects : ℕ := 5

theorem biology_marks_calculation :
  ∃ (biology_marks : ℕ),
    biology_marks = average_marks * total_subjects - (english_marks + math_marks + physics_marks + chemistry_marks) ∧
    biology_marks = 85 := by
  sorry

end biology_marks_calculation_l1646_164651


namespace vector_perpendicular_to_line_l1646_164639

/-- Given a vector a and a line l, prove that they are perpendicular -/
theorem vector_perpendicular_to_line (a : ℝ × ℝ) (l : ℝ → ℝ → Prop) : 
  a = (2, 3) → 
  (∀ x y, l x y ↔ 2 * x + 3 * y - 1 = 0) → 
  ∃ k, k * a.1 + a.2 = 0 ∧ k * 2 - 3 = 0 :=
by sorry

end vector_perpendicular_to_line_l1646_164639


namespace james_earnings_difference_l1646_164642

theorem james_earnings_difference (january_earnings : ℕ) 
  (february_earnings : ℕ) (march_earnings : ℕ) (total_earnings : ℕ) :
  january_earnings = 4000 →
  february_earnings = 2 * january_earnings →
  march_earnings < february_earnings →
  total_earnings = january_earnings + february_earnings + march_earnings →
  total_earnings = 18000 →
  february_earnings - march_earnings = 2000 := by
sorry

end james_earnings_difference_l1646_164642


namespace copper_zinc_ratio_l1646_164698

/-- Given a mixture of copper and zinc, prove that the ratio of copper to zinc is 77:63 -/
theorem copper_zinc_ratio (total_weight zinc_weight : ℝ)
  (h_total : total_weight = 70)
  (h_zinc : zinc_weight = 31.5)
  : ∃ (a b : ℕ), a = 77 ∧ b = 63 ∧ (total_weight - zinc_weight) / zinc_weight = a / b := by
  sorry

end copper_zinc_ratio_l1646_164698


namespace symmetry_implies_values_and_minimum_l1646_164630

/-- A function f(x) that is symmetric about the line x = -1 -/
def symmetric_about_neg_one (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-(x + 2)) = f x

/-- The function f(x) = (x^2 - 4)(x^2 + ax + b) -/
def f (a b : ℝ) (x : ℝ) : ℝ :=
  (x^2 - 4) * (x^2 + a*x + b)

theorem symmetry_implies_values_and_minimum (a b : ℝ) :
  symmetric_about_neg_one (f a b) →
  (a = 4 ∧ b = 0) ∧
  (∃ m : ℝ, m = -16 ∧ ∀ x : ℝ, f a b x ≥ m) :=
by sorry

end symmetry_implies_values_and_minimum_l1646_164630


namespace complex_modulus_l1646_164607

theorem complex_modulus (z : ℂ) (h : z^2 = -4) : Complex.abs (1 + z) = Real.sqrt 5 := by
  sorry

end complex_modulus_l1646_164607


namespace quarters_addition_theorem_l1646_164647

/-- The number of quarters initially in the jar -/
def initial_quarters : ℕ := 267

/-- The value of one quarter in dollars -/
def quarter_value : ℚ := 1/4

/-- The target total value in dollars -/
def target_value : ℚ := 100

/-- The number of quarters to be added -/
def quarters_to_add : ℕ := 133

theorem quarters_addition_theorem :
  (initial_quarters + quarters_to_add : ℚ) * quarter_value = target_value := by
  sorry

end quarters_addition_theorem_l1646_164647


namespace coin_toss_probability_l1646_164680

/-- The probability of getting exactly k heads in n tosses of a coin with probability r of landing heads -/
def binomial_probability (n k : ℕ) (r : ℚ) : ℚ :=
  (n.choose k : ℚ) * r^k * (1 - r)^(n - k)

/-- The main theorem -/
theorem coin_toss_probability : ∀ r : ℚ,
  0 < r →
  r < 1 →
  binomial_probability 5 1 r = binomial_probability 5 2 r →
  binomial_probability 5 3 r = 40 / 243 := by
  sorry

end coin_toss_probability_l1646_164680


namespace solution_comparison_l1646_164656

theorem solution_comparison (c d e f : ℝ) (hc : c ≠ 0) (he : e ≠ 0) :
  (-d / c > -f / e) ↔ (f / e > d / c) :=
sorry

end solution_comparison_l1646_164656


namespace exactly_two_statements_true_l1646_164620

def M (x : ℝ) : ℝ := 2 - 4*x
def N (x : ℝ) : ℝ := 4*x + 1

def statement1 : Prop := ¬ ∃ x : ℝ, M x + N x = 0
def statement2 : Prop := ∀ x : ℝ, ¬(M x > 0 ∧ N x > 0)
def statement3 : Prop := ∀ a : ℝ, (∀ x : ℝ, (M x + a) * N x = 1 - 16*x^2) → a = -1
def statement4 : Prop := ∀ x : ℝ, M x * N x = -3 → M x^2 + N x^2 = 11

theorem exactly_two_statements_true : 
  ∃! n : Fin 4, (n.val = 2 ∧ 
    (statement1 ∧ statement3) ∨
    (statement1 ∧ statement2) ∨
    (statement1 ∧ statement4) ∨
    (statement2 ∧ statement3) ∨
    (statement2 ∧ statement4) ∨
    (statement3 ∧ statement4)) :=
by sorry

end exactly_two_statements_true_l1646_164620


namespace probability_of_one_in_20_rows_l1646_164636

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : Type := Unit

/-- The number of elements in the first n rows of Pascal's Triangle -/
def totalElements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of 1s in the first n rows of Pascal's Triangle -/
def numberOfOnes (n : ℕ) : ℕ := 2 * n - 1

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle -/
def probabilityOfOne (n : ℕ) : ℚ := (numberOfOnes n : ℚ) / (totalElements n : ℚ)

theorem probability_of_one_in_20_rows :
  probabilityOfOne 20 = 13 / 70 := by sorry

end probability_of_one_in_20_rows_l1646_164636


namespace simplify_expressions_l1646_164685

theorem simplify_expressions (a x : ℝ) :
  (-a^3 + (-4*a^2)*a = -5*a^3) ∧
  (-x^2 * (-x)^2 * (-x^2)^3 - 2*x^10 = -x^10) := by
  sorry

end simplify_expressions_l1646_164685


namespace john_incentive_amount_l1646_164618

/-- Calculates the incentive amount given to an agent based on commission, advance fees, and amount paid. --/
def calculate_incentive (commission : ℕ) (advance_fees : ℕ) (amount_paid : ℕ) : Int :=
  (commission - advance_fees : Int) - amount_paid

/-- Proves that the incentive amount for John is -1780 Rs, indicating an excess payment. --/
theorem john_incentive_amount :
  let commission : ℕ := 25000
  let advance_fees : ℕ := 8280
  let amount_paid : ℕ := 18500
  calculate_incentive commission advance_fees amount_paid = -1780 := by
  sorry

end john_incentive_amount_l1646_164618


namespace two_green_then_red_probability_l1646_164652

/-- The number of traffic checkpoints -/
def num_checkpoints : ℕ := 6

/-- The probability of encountering a red light at each checkpoint -/
def red_light_prob : ℚ := 1/3

/-- The probability of passing exactly two checkpoints before encountering a red light -/
def prob_two_green_then_red : ℚ := 4/27

theorem two_green_then_red_probability :
  (1 - red_light_prob)^2 * red_light_prob = prob_two_green_then_red :=
sorry

end two_green_then_red_probability_l1646_164652


namespace fraction_equality_l1646_164601

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) :
  let x := a / b
  (a^2 + b^2) / (a^2 - b^2) = (x^2 + 1) / (x^2 - 1) := by
  sorry

end fraction_equality_l1646_164601


namespace hyperbola_eccentricity_l1646_164611

/-- The eccentricity of a hyperbola tangent to a specific circle -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), (x - Real.sqrt 3)^2 + (y - 1)^2 = 3 ∧ 
    (x^2 / a^2 - y^2 / b^2 = 1) ∧ 
    ((Real.sqrt 3 * b - a)^2 = 3 * (b^2 + a^2) ∨ (Real.sqrt 3 * b + a)^2 = 3 * (b^2 + a^2))) →
  Real.sqrt (a^2 + b^2) / a = 2 * Real.sqrt 3 / 3 := by
sorry

end hyperbola_eccentricity_l1646_164611


namespace population_percentage_l1646_164604

theorem population_percentage : 
  let total_population : ℕ := 40000
  let part_population : ℕ := 32000
  (part_population : ℚ) / (total_population : ℚ) * 100 = 80 := by
  sorry

end population_percentage_l1646_164604


namespace constant_term_binomial_expansion_l1646_164621

/-- The constant term in the expansion of (x^2 - 2/√x)^5 is 80 -/
theorem constant_term_binomial_expansion :
  (∃ (c : ℝ), c = 80 ∧ 
   ∀ (x : ℝ), x > 0 → 
   ∃ (f : ℝ → ℝ), (λ x => (x^2 - 2/Real.sqrt x)^5) = (λ x => f x + c)) := by
sorry

end constant_term_binomial_expansion_l1646_164621


namespace product_of_sum_of_squares_l1646_164690

theorem product_of_sum_of_squares (a b c d : ℤ) : ∃ x y : ℤ, (a^2 + b^2) * (c^2 + d^2) = x^2 + y^2 := by
  sorry

end product_of_sum_of_squares_l1646_164690


namespace final_price_calculation_l1646_164664

def original_price : ℝ := 10.00
def increase_percent : ℝ := 0.40
def decrease_percent : ℝ := 0.30

def price_after_increase (p : ℝ) (i : ℝ) : ℝ := p * (1 + i)
def price_after_decrease (p : ℝ) (d : ℝ) : ℝ := p * (1 - d)

theorem final_price_calculation : 
  price_after_decrease (price_after_increase original_price increase_percent) decrease_percent = 9.80 := by
  sorry

end final_price_calculation_l1646_164664


namespace contractor_fine_calculation_l1646_164654

/-- Proves that the fine for each day of absence is 7.5 given the contract conditions --/
theorem contractor_fine_calculation (total_days : ℕ) (work_pay : ℝ) (total_earnings : ℝ) (absent_days : ℕ) :
  total_days = 30 →
  work_pay = 25 →
  total_earnings = 360 →
  absent_days = 12 →
  ∃ (fine : ℝ), fine = 7.5 ∧ 
    work_pay * (total_days - absent_days) - fine * absent_days = total_earnings :=
by
  sorry

end contractor_fine_calculation_l1646_164654


namespace mean_median_difference_l1646_164622

-- Define the frequency distribution of days missed
def days_missed : List (Nat × Nat) := [
  (0, 2),  -- 2 students missed 0 days
  (1, 3),  -- 3 students missed 1 day
  (2, 6),  -- 6 students missed 2 days
  (3, 5),  -- 5 students missed 3 days
  (4, 2),  -- 2 students missed 4 days
  (5, 2)   -- 2 students missed 5 days
]

-- Define the total number of students
def total_students : Nat := 20

-- Theorem statement
theorem mean_median_difference :
  let mean := (days_missed.map (λ (d, f) => d * f)).sum / total_students
  let median := 2  -- The median is 2 days (10th and 11th students both missed 2 days)
  mean - median = 2 / 5 := by sorry


end mean_median_difference_l1646_164622


namespace unique_k_for_prime_roots_l1646_164625

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 0 ∧ m < n → n % m ≠ 0

/-- The roots of a quadratic equation ax^2 + bx + c = 0 are given by (-b ± √(b^2 - 4ac)) / (2a) -/
def isRootOf (x : ℝ) (a b c : ℝ) : Prop := a * x^2 + b * x + c = 0

theorem unique_k_for_prime_roots : ∃! k : ℕ, 
  ∃ p q : ℕ, 
    isPrime p ∧ 
    isPrime q ∧ 
    isRootOf p 1 (-63) k ∧ 
    isRootOf q 1 (-63) k :=
sorry

end unique_k_for_prime_roots_l1646_164625


namespace quadratic_factorization_l1646_164679

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 := by
  sorry

end quadratic_factorization_l1646_164679


namespace power_sum_equality_l1646_164670

theorem power_sum_equality : (3^2)^3 + (2^3)^2 = 793 := by sorry

end power_sum_equality_l1646_164670


namespace find_a_solve_inequality_l1646_164633

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 6

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x | x < 1 ∨ x > 2}

-- Theorem 1: Given the solution set, prove a = 1
theorem find_a : ∀ a : ℝ, (∀ x : ℝ, f a x > 4 ↔ x ∈ solution_set a) → a = 1 := by sorry

-- Define the linear function for the second inequality
def g (c : ℝ) (x : ℝ) : ℝ := (c - x) * (x + 2)

-- Theorem 2: Solve the inequality (c-x)(x+2) > 0
theorem solve_inequality :
  ∀ c : ℝ,
  (c = -2 → {x : ℝ | g c x > 0} = ∅) ∧
  (c > -2 → {x : ℝ | g c x > 0} = Set.Ioo (-2) c) ∧
  (c < -2 → {x : ℝ | g c x > 0} = Set.Ioo c (-2)) := by sorry

end find_a_solve_inequality_l1646_164633


namespace a_lt_b_neither_sufficient_nor_necessary_for_a_sq_lt_b_sq_l1646_164606

theorem a_lt_b_neither_sufficient_nor_necessary_for_a_sq_lt_b_sq :
  ∃ (a b c d : ℝ),
    (a < b ∧ ¬(a^2 < b^2)) ∧
    (c^2 < d^2 ∧ ¬(c < d)) :=
sorry

end a_lt_b_neither_sufficient_nor_necessary_for_a_sq_lt_b_sq_l1646_164606


namespace exists_valid_point_distribution_l1646_164655

/-- Represents a convex pentagon --/
structure ConvexPentagon where
  -- Add necessary fields

/-- Represents a point inside the pentagon --/
structure Point where
  -- Add necessary fields

/-- Represents a triangle formed by the vertices of the pentagon --/
structure Triangle where
  -- Add necessary fields

/-- Function to check if a point is inside a triangle --/
def pointInTriangle (p : Point) (t : Triangle) : Bool :=
  sorry

/-- Function to count points inside a triangle --/
def countPointsInTriangle (points : List Point) (t : Triangle) : Nat :=
  sorry

/-- Theorem stating the existence of a valid point distribution --/
theorem exists_valid_point_distribution (pentagon : ConvexPentagon) :
  ∃ (points : List Point),
    points.length = 18 ∧
    ∀ (t1 t2 : Triangle),
      countPointsInTriangle points t1 = countPointsInTriangle points t2 :=
  sorry

end exists_valid_point_distribution_l1646_164655


namespace fraction_division_simplify_fraction_division_l1646_164641

theorem fraction_division (a b c d : ℚ) (h1 : b ≠ 0) (h2 : d ≠ 0) (h3 : c ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem simplify_fraction_division :
  (5 : ℚ) / 6 / ((9 : ℚ) / 10) = 25 / 27 :=
by sorry

end fraction_division_simplify_fraction_division_l1646_164641


namespace final_amount_calculation_l1646_164609

def initial_amount : ℕ := 5
def spent_amount : ℕ := 2
def allowance : ℕ := 26

theorem final_amount_calculation :
  initial_amount - spent_amount + allowance = 29 := by
  sorry

end final_amount_calculation_l1646_164609


namespace seven_swimmer_race_outcomes_l1646_164669

/-- The number of different possible outcomes for 1st-2nd-3rd place in a race with n swimmers and no ties -/
def race_outcomes (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

/-- Theorem: The number of different possible outcomes for 1st-2nd-3rd place in a race with 7 swimmers and no ties is 210 -/
theorem seven_swimmer_race_outcomes : race_outcomes 7 = 210 := by
  sorry

end seven_swimmer_race_outcomes_l1646_164669


namespace symmetric_function_intersection_l1646_164635

/-- Definition of a symmetric function -/
def symmetricFunction (m n : ℝ) : ℝ → ℝ := λ x ↦ n * x + m

/-- The given function -/
def givenFunction : ℝ → ℝ := λ x ↦ -6 * x + 4

/-- Theorem: The intersection point of the symmetric function of y=-6x+4 with the y-axis is (0, -6) -/
theorem symmetric_function_intersection :
  let f := symmetricFunction (-6) 4
  (0, f 0) = (0, -6) := by sorry

end symmetric_function_intersection_l1646_164635


namespace product_lower_bound_l1646_164637

theorem product_lower_bound (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ ≥ 0) (h₂ : x₂ ≥ 0) (h₃ : x₃ ≥ 0) 
  (h₄ : x₁ + x₂ + x₃ ≤ 1/2) : 
  (1 - x₁) * (1 - x₂) * (1 - x₃) ≥ 1/2 := by
  sorry

end product_lower_bound_l1646_164637


namespace both_hit_target_probability_l1646_164602

theorem both_hit_target_probability
  (prob_A : ℝ)
  (prob_B : ℝ)
  (h_A : prob_A = 0.8)
  (h_B : prob_B = 0.6) :
  prob_A * prob_B = 0.48 := by
sorry

end both_hit_target_probability_l1646_164602


namespace relationship_abc_l1646_164605

noncomputable def a : ℝ := Real.rpow 0.6 0.6
noncomputable def b : ℝ := Real.rpow 0.6 1.5
noncomputable def c : ℝ := Real.rpow 1.5 0.6

theorem relationship_abc : b < a ∧ a < c := by
  sorry

end relationship_abc_l1646_164605


namespace min_value_trig_expression_l1646_164681

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ (Real.sqrt 244 - 7)^2 := by
  sorry

end min_value_trig_expression_l1646_164681


namespace zoltan_incorrect_answers_l1646_164697

theorem zoltan_incorrect_answers 
  (total_questions : Nat)
  (answered_questions : Nat)
  (total_score : Int)
  (correct_points : Int)
  (incorrect_points : Int)
  (unanswered_points : Int)
  (h1 : total_questions = 50)
  (h2 : answered_questions = 45)
  (h3 : total_score = 135)
  (h4 : correct_points = 4)
  (h5 : incorrect_points = -1)
  (h6 : unanswered_points = 0) :
  ∃ (incorrect : Nat),
    incorrect = 9 ∧
    (answered_questions - incorrect) * correct_points + 
    incorrect * incorrect_points + 
    (total_questions - answered_questions) * unanswered_points = total_score :=
by sorry

end zoltan_incorrect_answers_l1646_164697


namespace square_dancing_problem_l1646_164665

/-- The number of female students in the first class that satisfies the square dancing conditions --/
def female_students_in_first_class : ℕ := by sorry

theorem square_dancing_problem :
  let males_class1 : ℕ := 17
  let males_class2 : ℕ := 14
  let females_class2 : ℕ := 18
  let males_class3 : ℕ := 15
  let females_class3 : ℕ := 17
  let total_males : ℕ := males_class1 + males_class2 + males_class3
  let total_females : ℕ := female_students_in_first_class + females_class2 + females_class3
  let unpartnered_students : ℕ := 2

  female_students_in_first_class = 9 ∧
  total_males = total_females + unpartnered_students := by sorry

end square_dancing_problem_l1646_164665


namespace curve_composition_l1646_164613

-- Define the curve
def curve (x y : ℝ) : Prop := (3*x - y + 1) * (y - Real.sqrt (1 - x^2)) = 0

-- Define a semicircle
def semicircle (x y : ℝ) : Prop := y = Real.sqrt (1 - x^2) ∧ -1 ≤ x ∧ x ≤ 1

-- Define a line segment
def line_segment (x y : ℝ) : Prop := 3*x - y + 1 = 0 ∧ -1 ≤ x ∧ x ≤ 1

-- Theorem statement
theorem curve_composition :
  ∀ x y : ℝ, curve x y ↔ (semicircle x y ∨ line_segment x y) :=
sorry

end curve_composition_l1646_164613


namespace equation_solution_l1646_164692

theorem equation_solution :
  ∃ x : ℚ, x ≠ 1 ∧ 2*x ≠ 2 ∧ x / (x - 1) = 3 / (2*x - 2) - 2 ∧ x = 7/6 :=
by
  sorry

end equation_solution_l1646_164692


namespace initial_distance_is_40_l1646_164616

/-- The initial distance between two people walking towards each other -/
def initial_distance (speed : ℝ) (distance_walked : ℝ) : ℝ :=
  2 * distance_walked

/-- Theorem: The initial distance between Fred and Sam is 40 miles -/
theorem initial_distance_is_40 :
  let fred_speed : ℝ := 4
  let sam_speed : ℝ := 4
  let sam_distance : ℝ := 20
  initial_distance fred_speed sam_distance = 40 := by
  sorry


end initial_distance_is_40_l1646_164616


namespace base4_arithmetic_theorem_l1646_164650

/-- Converts a number from base 4 to base 10 -/
def base4To10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 -/
def base10To4 (n : ℕ) : ℕ := sorry

/-- Performs arithmetic operations in base 4 -/
def base4Arithmetic (a b c d : ℕ) : ℕ := 
  let a10 := base4To10 a
  let b10 := base4To10 b
  let c10 := base4To10 c
  let d10 := base4To10 d
  base10To4 (a10 + b10 * c10 / d10)

theorem base4_arithmetic_theorem : 
  base4Arithmetic 231 21 12 3 = 333 := by sorry

end base4_arithmetic_theorem_l1646_164650


namespace certain_number_value_l1646_164693

theorem certain_number_value (p q : ℕ) (x : ℚ) 
  (hp : p > 1) 
  (hq : q > 1) 
  (hx : x * (p + 1) = 28 * (q + 1)) 
  (hpq_min : ∀ (p' q' : ℕ), p' > 1 → q' > 1 → p' + q' < p + q → ¬∃ (x' : ℚ), x' * (p' + 1) = 28 * (q' + 1)) :
  x = 392 := by
sorry

end certain_number_value_l1646_164693


namespace zach_ticket_purchase_l1646_164640

/-- The number of tickets Zach needs to buy for both rides -/
def tickets_needed (ferris_wheel_cost roller_coaster_cost multiple_ride_discount coupon : ℝ) : ℝ :=
  ferris_wheel_cost + roller_coaster_cost - multiple_ride_discount - coupon

/-- Theorem stating the number of tickets Zach needs to buy -/
theorem zach_ticket_purchase :
  tickets_needed 2.0 7.0 1.0 1.0 = 7.0 := by
  sorry

#eval tickets_needed 2.0 7.0 1.0 1.0

end zach_ticket_purchase_l1646_164640


namespace binary_to_octal_conversion_l1646_164631

-- Define the binary number
def binary_num : ℕ := 0b101101

-- Define the octal number
def octal_num : ℕ := 0o55

-- Theorem statement
theorem binary_to_octal_conversion :
  binary_num = octal_num := by sorry

end binary_to_octal_conversion_l1646_164631


namespace two_digit_subtraction_equality_l1646_164663

theorem two_digit_subtraction_equality (a b : Nat) : 
  0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ a ≠ b → (70 * a - 7 * a) - (70 * b - 7 * b) = 0 := by
  sorry

end two_digit_subtraction_equality_l1646_164663


namespace no_solution_eq1_unique_solution_eq2_l1646_164666

-- Problem 1
theorem no_solution_eq1 : ¬∃ x : ℝ, (1 / (x - 2) + 3 = (1 - x) / (2 - x)) := by sorry

-- Problem 2
theorem unique_solution_eq2 : ∃! x : ℝ, (x / (x - 1) - 1 = 3 / (x^2 - 1)) ∧ x = 2 := by sorry

end no_solution_eq1_unique_solution_eq2_l1646_164666


namespace existence_of_solution_l1646_164682

theorem existence_of_solution : ∃ (x y : ℤ), 2 * x^2 + 8 * y = 26 ∧ x - y = 26 := by
  sorry

end existence_of_solution_l1646_164682


namespace instantaneous_velocity_at_3s_l1646_164629

-- Define the displacement function
def h (t : ℝ) : ℝ := 1.5 * t - 0.1 * t^2

-- Define the velocity function as the derivative of the displacement function
def v (t : ℝ) : ℝ := 1.5 - 0.2 * t

-- Theorem statement
theorem instantaneous_velocity_at_3s : v 3 = 0.9 := by
  sorry

end instantaneous_velocity_at_3s_l1646_164629


namespace quadratic_equation_solution_l1646_164643

theorem quadratic_equation_solution : 
  ∀ x : ℝ, (2 * x^2 + 10 * x + 12 = -(x + 4) * (x + 6)) ↔ (x = -4 ∨ x = -3) :=
by sorry

end quadratic_equation_solution_l1646_164643


namespace williams_riding_time_l1646_164684

def max_riding_time : ℝ := 6

theorem williams_riding_time (x : ℝ) : 
  (2 * max_riding_time) + (2 * x) + (2 * (max_riding_time / 2)) = 21 → x = 1.5 := by
  sorry

end williams_riding_time_l1646_164684


namespace range_of_a_l1646_164694

def S (a : ℝ) : Set ℝ := {x | 2 * a * x^2 - x ≤ 0}

def T (a : ℝ) : Set ℝ := {x | 4 * a * x^2 - 4 * a * (1 - 2 * a) * x + 1 ≥ 0}

theorem range_of_a (a : ℝ) (h : S a ∪ T a = Set.univ) : 0 ≤ a ∧ a ≤ 1 := by
  sorry

end range_of_a_l1646_164694


namespace train_crossing_bridge_time_l1646_164675

/-- Time taken for a train to cross a bridge -/
theorem train_crossing_bridge_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (total_length : ℝ) 
  (h1 : train_length = 130) 
  (h2 : train_speed_kmh = 45) 
  (h3 : total_length = 245) : 
  (total_length / (train_speed_kmh * 1000 / 3600)) = 9.8 := by
  sorry

end train_crossing_bridge_time_l1646_164675


namespace unique_integer_pairs_l1646_164600

theorem unique_integer_pairs :
  ∀ x y : ℕ+,
  x < y →
  x + y = 667 →
  (Nat.lcm x.val y.val : ℕ) / Nat.gcd x.val y.val = 120 →
  ((x = 145 ∧ y = 522) ∨ (x = 184 ∧ y = 483)) :=
by sorry

end unique_integer_pairs_l1646_164600


namespace complex_magnitude_product_l1646_164658

theorem complex_magnitude_product : 
  Complex.abs ((5 - 3*Complex.I) * (7 + 24*Complex.I)) = 25 * Real.sqrt 34 := by
  sorry

end complex_magnitude_product_l1646_164658


namespace smallest_base_for_perfect_square_l1646_164610

theorem smallest_base_for_perfect_square : 
  ∀ b : ℕ, b > 4 → (∃ n : ℕ, 3 * b + 4 = n^2) → b ≥ 7 :=
by sorry

end smallest_base_for_perfect_square_l1646_164610


namespace jony_stops_at_70_l1646_164626

/-- Represents the walking scenario of Jony along Sunrise Boulevard -/
structure WalkingScenario where
  start_block : ℕ
  turn_block : ℕ
  block_length : ℕ
  walking_speed : ℕ
  walking_time : ℕ

/-- Calculates the block where Jony stops walking -/
def stop_block (scenario : WalkingScenario) : ℕ :=
  sorry

/-- Theorem stating that Jony stops at block 70 given the specific scenario -/
theorem jony_stops_at_70 : 
  let scenario : WalkingScenario := {
    start_block := 10,
    turn_block := 90,
    block_length := 40,
    walking_speed := 100,
    walking_time := 40
  }
  stop_block scenario = 70 := by
  sorry

end jony_stops_at_70_l1646_164626


namespace largest_integer_problem_l1646_164603

theorem largest_integer_problem :
  ∃ (m : ℕ), m < 150 ∧ m > 50 ∧ 
  (∃ (a : ℕ), m = 9 * a - 2) ∧
  (∃ (b : ℕ), m = 6 * b - 4) ∧
  (∀ (n : ℕ), n < 150 ∧ n > 50 ∧ 
    (∃ (c : ℕ), n = 9 * c - 2) ∧
    (∃ (d : ℕ), n = 6 * d - 4) → n ≤ m) ∧
  m = 106 :=
sorry

end largest_integer_problem_l1646_164603


namespace kennel_problem_l1646_164646

/-- Represents the number of dogs with various accessories in a kennel -/
structure KennelData where
  total : ℕ
  tags : ℕ
  flea_collars : ℕ
  harnesses : ℕ
  tags_and_flea : ℕ
  tags_and_harnesses : ℕ
  flea_and_harnesses : ℕ
  all_three : ℕ

/-- Calculates the number of dogs with no accessories given kennel data -/
def dogs_with_no_accessories (data : KennelData) : ℕ :=
  data.total - (data.tags + data.flea_collars + data.harnesses - 
    data.tags_and_flea - data.tags_and_harnesses - data.flea_and_harnesses + data.all_three)

/-- Theorem stating that given the specific kennel data, 25 dogs have no accessories -/
theorem kennel_problem (data : KennelData) 
    (h1 : data.total = 120)
    (h2 : data.tags = 60)
    (h3 : data.flea_collars = 50)
    (h4 : data.harnesses = 30)
    (h5 : data.tags_and_flea = 20)
    (h6 : data.tags_and_harnesses = 15)
    (h7 : data.flea_and_harnesses = 10)
    (h8 : data.all_three = 5) :
  dogs_with_no_accessories data = 25 := by
  sorry

end kennel_problem_l1646_164646


namespace potato_bag_fraction_l1646_164649

theorem potato_bag_fraction (weight : ℝ) (x : ℝ) : 
  weight = 12 → weight / x = 12 → x = 1 := by sorry

end potato_bag_fraction_l1646_164649


namespace complex_magnitude_equality_l1646_164644

theorem complex_magnitude_equality (n : ℝ) (hn : n > 0) :
  Complex.abs (4 + 2 * n * Complex.I) = 4 * Real.sqrt 5 ↔ n = 4 := by sorry

end complex_magnitude_equality_l1646_164644


namespace equation_solution_l1646_164674

theorem equation_solution :
  ∃ x : ℚ, (x - 30) / 3 = (5 - 3 * x) / 4 ∧ x = 135 / 13 := by
sorry

end equation_solution_l1646_164674


namespace number_problem_l1646_164653

theorem number_problem : ∃ x : ℝ, (x - 5) / 3 = 4 ∧ x = 17 := by sorry

end number_problem_l1646_164653


namespace quarter_circles_limit_l1646_164672

/-- The limit of the sum of quarter-circle lengths approaches the original circumference -/
theorem quarter_circles_limit (C : ℝ) (h : C > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |2 * n * (C / (2 * n)) - C| < ε :=
sorry

end quarter_circles_limit_l1646_164672


namespace line_point_k_value_l1646_164614

/-- Given a line containing the points (0, 7), (15, k), and (20, 3), prove that k = 4 -/
theorem line_point_k_value (k : ℝ) : 
  (∀ (x y : ℝ), (x = 0 ∧ y = 7) ∨ (x = 15 ∧ y = k) ∨ (x = 20 ∧ y = 3) → 
    ∃ (m b : ℝ), y = m * x + b) → 
  k = 4 := by
sorry

end line_point_k_value_l1646_164614


namespace parabola_intersection_length_l1646_164676

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the intersecting line
def line (k : ℝ) (x y : ℝ) : Prop := y = k*x - 2

-- Define the midpoint condition
def midpoint_condition (k : ℝ) : Prop := (2*k + 4) / (k^2) = 2

-- Main theorem
theorem parabola_intersection_length :
  ∀ (k : ℝ),
  parabola 2 4 →
  k > -1 →
  k ≠ 0 →
  midpoint_condition k →
  ∃ (x₁ x₂ y₁ y₂ : ℝ),
  parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
  line k x₁ y₁ ∧ line k x₂ y₂ ∧
  ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 60 :=
sorry

end parabola_intersection_length_l1646_164676


namespace circle_tangent_to_line_l1646_164648

theorem circle_tangent_to_line (x y : ℝ) :
  (∀ a b : ℝ, a^2 + b^2 = 2 → (b ≠ 2 - a ∨ (a - 0)^2 + (b - 0)^2 = 2)) ∧
  (∃ c d : ℝ, c^2 + d^2 = 2 ∧ d = 2 - c) :=
sorry

end circle_tangent_to_line_l1646_164648


namespace inequality_system_solution_range_l1646_164667

theorem inequality_system_solution_range (a : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℤ), 
    (∀ x : ℤ, (x + 6 < 2 + 3*x ∧ (a + x) / 4 > x) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) →
  (15 < a ∧ a ≤ 18) :=
by sorry

end inequality_system_solution_range_l1646_164667


namespace fraction_equality_implies_x_equals_three_l1646_164659

theorem fraction_equality_implies_x_equals_three (x : ℝ) :
  (5 / (2 * x - 1) = 3 / x) → x = 3 := by
  sorry

end fraction_equality_implies_x_equals_three_l1646_164659


namespace exactly_one_statement_true_l1646_164688

/-- Polynomials A, B, C, D, and E -/
def A (x : ℝ) : ℝ := 2 * x^2
def B (x : ℝ) : ℝ := x + 1
def C (x : ℝ) : ℝ := -2 * x
def D (y : ℝ) : ℝ := y^2
def E (x y : ℝ) : ℝ := 2 * x - y

/-- Statement 1: For all positive integer y, B*C + A + D + E > 0 -/
def statement1 : Prop :=
  ∀ (x : ℝ) (y : ℕ), (B x * C x + A x + D y + E x y) > 0

/-- Statement 2: There exist real numbers x and y such that A + D + 2E = -2 -/
def statement2 : Prop :=
  ∃ (x y : ℝ), A x + D y + 2 * E x y = -2

/-- Statement 3: For all real x, if 3(A-B) + m*B*C has no linear term in x
    (where m is a constant), then 3(A-B) + m*B*C > -3 -/
def statement3 : Prop :=
  ∀ (x m : ℝ),
    (∃ (k : ℝ), 3 * (A x - B x) + m * B x * C x = k * x^2 + (3 * (A 0 - B 0) + m * B 0 * C 0)) →
    3 * (A x - B x) + m * B x * C x > -3

theorem exactly_one_statement_true :
  (statement1 ∧ ¬statement2 ∧ ¬statement3) ∨
  (¬statement1 ∧ statement2 ∧ ¬statement3) ∨
  (¬statement1 ∧ ¬statement2 ∧ statement3) := by sorry

end exactly_one_statement_true_l1646_164688


namespace units_digit_of_n_l1646_164612

/-- Given two natural numbers m and n, returns true if their product ends with the digit d -/
def product_ends_with (m n d : ℕ) : Prop :=
  (m * n) % 10 = d

/-- Given a natural number x, returns true if its units digit is d -/
def units_digit (x d : ℕ) : Prop :=
  x % 10 = d

theorem units_digit_of_n (m n : ℕ) :
  product_ends_with m n 4 →
  units_digit m 8 →
  units_digit n 3 := by
  sorry

#check units_digit_of_n

end units_digit_of_n_l1646_164612


namespace unique_triple_l1646_164617

theorem unique_triple : 
  ∃! (a b c : ℕ), a ≥ b ∧ b ≥ c ∧ a^3 + 9*b^2 + 9*c + 7 = 1997 ∧ 
  a = 10 ∧ b = 10 ∧ c = 10 := by
  sorry

end unique_triple_l1646_164617


namespace integral_quarter_circle_area_l1646_164673

theorem integral_quarter_circle_area (r : ℝ) (h : r > 0) :
  ∫ x in (0)..(r), Real.sqrt (r^2 - x^2) = (π * r^2) / 4 := by
  sorry

end integral_quarter_circle_area_l1646_164673


namespace corina_calculation_l1646_164645

theorem corina_calculation (P Q : ℤ) 
  (h1 : P + Q = 16) 
  (h2 : P - Q = 4) : 
  P = 10 := by
sorry

end corina_calculation_l1646_164645


namespace right_triangle_area_l1646_164634

theorem right_triangle_area (a b c : ℝ) (h1 : a = 40) (h2 : c = 41) (h3 : a^2 + b^2 = c^2) : 
  (1/2) * a * b = 180 := by
  sorry

end right_triangle_area_l1646_164634


namespace quadratic_root_sum_product_l1646_164677

theorem quadratic_root_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 9 ∧ x * y = 14) →
  p + q = 69 := by
sorry

end quadratic_root_sum_product_l1646_164677


namespace xiao_gao_score_l1646_164683

/-- Represents a test score system with a standard score and a recorded score. -/
structure TestScore where
  standard : ℕ
  recorded : ℤ

/-- Calculates the actual score given a TestScore. -/
def actualScore (ts : TestScore) : ℕ :=
  ts.standard + ts.recorded.toNat

/-- Theorem stating that for a standard score of 80 and a recorded score of 12,
    the actual score is 92. -/
theorem xiao_gao_score :
  let ts : TestScore := { standard := 80, recorded := 12 }
  actualScore ts = 92 := by
  sorry

end xiao_gao_score_l1646_164683


namespace work_earnings_equation_l1646_164624

theorem work_earnings_equation (t : ℚ) : 
  (t + 2) * (3 * t - 2) = (3 * t - 4) * (t + 1) + 5 → t = 5 / 3 := by
  sorry

end work_earnings_equation_l1646_164624


namespace sum_of_cyclic_equations_l1646_164638

theorem sum_of_cyclic_equations (x y z : ℝ) 
  (h1 : x + y = 1) 
  (h2 : y + z = 1) 
  (h3 : z + x = 1) : 
  x + y + z = 3/2 := by
  sorry

end sum_of_cyclic_equations_l1646_164638


namespace customers_without_tip_l1646_164686

theorem customers_without_tip (initial_customers : ℕ) (additional_customers : ℕ) (customers_with_tip : ℕ) : 
  initial_customers = 39 → additional_customers = 12 → customers_with_tip = 2 →
  initial_customers + additional_customers - customers_with_tip = 49 := by
sorry

end customers_without_tip_l1646_164686


namespace hash_composition_l1646_164678

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.75 * N + 2

-- State the theorem
theorem hash_composition : hash (hash (hash 72)) = 35 := by sorry

end hash_composition_l1646_164678


namespace min_distance_complex_circles_l1646_164627

/-- The minimum distance between two complex numbers on specific circles -/
theorem min_distance_complex_circles :
  ∀ (z w : ℂ),
  Complex.abs (z - (2 + Complex.I)) = 2 →
  Complex.abs (w + (3 + 4 * Complex.I)) = 4 →
  (∀ (z' w' : ℂ),
    Complex.abs (z' - (2 + Complex.I)) = 2 →
    Complex.abs (w' + (3 + 4 * Complex.I)) = 4 →
    Complex.abs (z - w) ≤ Complex.abs (z' - w')) →
  Complex.abs (z - w) = 5 * Real.sqrt 2 - 6 :=
by sorry

end min_distance_complex_circles_l1646_164627


namespace rectangle_height_double_area_square_side_double_area_cube_side_double_volume_rectangle_half_width_triple_height_rectangle_double_length_triple_width_geometric_transformations_l1646_164689

-- Define geometric shapes
def Rectangle (w h : ℝ) := w * h
def Square (s : ℝ) := s * s
def Cube (s : ℝ) := s * s * s

-- Theorem for statement (A)
theorem rectangle_height_double_area (w h : ℝ) :
  Rectangle w (2 * h) = 2 * Rectangle w h := by sorry

-- Theorem for statement (B)
theorem square_side_double_area (s : ℝ) :
  Square (2 * s) = 4 * Square s := by sorry

-- Theorem for statement (C)
theorem cube_side_double_volume (s : ℝ) :
  Cube (2 * s) = 8 * Cube s := by sorry

-- Theorem for statement (D)
theorem rectangle_half_width_triple_height (w h : ℝ) :
  Rectangle (w / 2) (3 * h) = (3 / 2) * Rectangle w h := by sorry

-- Theorem for statement (E)
theorem rectangle_double_length_triple_width (l w : ℝ) :
  Rectangle (2 * l) (3 * w) = 6 * Rectangle l w := by sorry

-- Main theorem proving (A) is false and others are true
theorem geometric_transformations :
  (∃ w h : ℝ, Rectangle w (2 * h) ≠ 3 * Rectangle w h) ∧
  (∀ s : ℝ, Square (2 * s) = 4 * Square s) ∧
  (∀ s : ℝ, Cube (2 * s) = 8 * Cube s) ∧
  (∀ w h : ℝ, Rectangle (w / 2) (3 * h) = (3 / 2) * Rectangle w h) ∧
  (∀ l w : ℝ, Rectangle (2 * l) (3 * w) = 6 * Rectangle l w) := by sorry

end rectangle_height_double_area_square_side_double_area_cube_side_double_volume_rectangle_half_width_triple_height_rectangle_double_length_triple_width_geometric_transformations_l1646_164689


namespace sum_of_coefficients_l1646_164662

theorem sum_of_coefficients (a b c d e : ℝ) : 
  (∀ x, 1000 * x^3 + 27 = (a*x + b) * (c*x^2 + d*x + e)) →
  a + b + c + d + e = 92 :=
by sorry

end sum_of_coefficients_l1646_164662


namespace square_of_negative_sqrt_two_l1646_164619

theorem square_of_negative_sqrt_two : (-Real.sqrt 2)^2 = 2 := by
  sorry

end square_of_negative_sqrt_two_l1646_164619


namespace sheet_length_l1646_164671

theorem sheet_length (width : ℝ) (side_margin : ℝ) (top_bottom_margin : ℝ) (typing_percentage : ℝ) :
  width = 20 →
  side_margin = 2 →
  top_bottom_margin = 3 →
  typing_percentage = 0.64 →
  ∃ length : ℝ,
    length = 30 ∧
    (width - 2 * side_margin) * (length - 2 * top_bottom_margin) = typing_percentage * width * length :=
by sorry

end sheet_length_l1646_164671


namespace camilo_kenny_difference_l1646_164668

def paint_house_problem (judson_contribution kenny_contribution camilo_contribution total_cost : ℕ) : Prop :=
  judson_contribution = 500 ∧
  kenny_contribution = judson_contribution + judson_contribution / 5 ∧
  camilo_contribution > kenny_contribution ∧
  total_cost = 1900 ∧
  judson_contribution + kenny_contribution + camilo_contribution = total_cost

theorem camilo_kenny_difference :
  ∀ judson_contribution kenny_contribution camilo_contribution total_cost,
    paint_house_problem judson_contribution kenny_contribution camilo_contribution total_cost →
    camilo_contribution - kenny_contribution = 200 := by
  sorry

end camilo_kenny_difference_l1646_164668


namespace complex_equation_real_solution_l1646_164657

theorem complex_equation_real_solution (a : ℝ) : 
  (((a : ℂ) / (1 + Complex.I) + 1 + Complex.I).im = 0) → a = 2 := by
  sorry

end complex_equation_real_solution_l1646_164657


namespace sequence_property_l1646_164695

/-- Given a sequence {a_n} with sum of first n terms S_n = 2a_n - a_1,
    and a_1, a_2+1, a_3 form an arithmetic sequence, prove a_n = 2^n -/
theorem sequence_property (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = 2 * a n - a 1) → 
  (2 * (a 2 + 1) = a 3 + a 1) →
  ∀ n, a n = 2^n := by sorry

end sequence_property_l1646_164695


namespace negation_of_proposition_l1646_164608

open Real

theorem negation_of_proposition (p : ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) :
  ∃ x₀ : ℝ, x₀ > 0 ∧ (x₀ + 1) * Real.exp x₀ ≤ 1 :=
by sorry

end negation_of_proposition_l1646_164608


namespace polynomial_factorization_l1646_164699

def P (x : ℝ) : ℝ := x^4 + 2*x^3 - 23*x^2 + 12*x + 36

theorem polynomial_factorization :
  ∃ (a b c : ℝ),
    (∀ x, P x = (x^2 + a*x + c) * (x^2 + b*x + c)) ∧
    a + b = 2 ∧
    a * b = -35 ∧
    c = 6 ∧
    (∀ x, P x = 0 ↔ x = 2 ∨ x = 3 ∨ x = -1 ∨ x = -6) :=
by sorry

end polynomial_factorization_l1646_164699


namespace taxi_charge_proof_l1646_164660

/-- Calculates the total charge for a taxi trip -/
def totalCharge (initialFee : ℚ) (additionalChargePerIncrement : ℚ) (incrementDistance : ℚ) (tripDistance : ℚ) : ℚ :=
  initialFee + (tripDistance / incrementDistance).floor * additionalChargePerIncrement

/-- Proves that the total charge for a 3.6-mile trip is $4.50 -/
theorem taxi_charge_proof :
  let initialFee : ℚ := 9/4  -- $2.25
  let additionalChargePerIncrement : ℚ := 1/4  -- $0.25
  let incrementDistance : ℚ := 2/5  -- 2/5 mile
  let tripDistance : ℚ := 18/5  -- 3.6 miles
  totalCharge initialFee additionalChargePerIncrement incrementDistance tripDistance = 9/2  -- $4.50
:= by sorry

end taxi_charge_proof_l1646_164660


namespace ring_stack_height_l1646_164661

/-- Represents a stack of linked rings -/
structure RingStack where
  top_diameter : ℝ
  bottom_diameter : ℝ
  ring_thickness : ℝ

/-- Calculates the total height of the ring stack -/
def stack_height (stack : RingStack) : ℝ :=
  sorry

/-- Theorem: The height of the given ring stack is 72 cm -/
theorem ring_stack_height :
  let stack := RingStack.mk 20 4 2
  stack_height stack = 72 := by
  sorry

end ring_stack_height_l1646_164661


namespace complex_expression_equals_negative_65_l1646_164632

theorem complex_expression_equals_negative_65 :
  -2^3 * (-3)^2 / (9/8) - |1/2 - 3/2| = -65 := by
  sorry

end complex_expression_equals_negative_65_l1646_164632


namespace base_eight_132_equals_90_l1646_164623

def base_eight_to_ten (a b c : Nat) : Nat :=
  a * 8^2 + b * 8^1 + c * 8^0

theorem base_eight_132_equals_90 : base_eight_to_ten 1 3 2 = 90 := by
  sorry

end base_eight_132_equals_90_l1646_164623


namespace probability_theorem_l1646_164696

-- Define the interval
def interval : Set ℝ := Set.Icc (-2) 2

-- Define the condition x^2 < 1
def condition (x : ℝ) : Prop := x^2 < 1

-- Define the measure of the interval [-2, 2]
def totalMeasure : ℝ := 4

-- Define the measure of the solution set (-1, 1)
def solutionMeasure : ℝ := 2

-- State the theorem
theorem probability_theorem :
  (solutionMeasure / totalMeasure) = (1 / 2) := by sorry

end probability_theorem_l1646_164696
