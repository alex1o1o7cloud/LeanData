import Mathlib

namespace NUMINAMATH_CALUDE_only_event1_is_random_l237_23793

/-- Represents an event in a probability space -/
structure Event where
  description : String

/-- Defines what it means for an event to be random -/
def isRandomEvent (e : Event) : Prop :=
  sorry  -- Definition of random event

/-- Event 1: Tossing a coin twice in a row and getting heads both times -/
def event1 : Event := ⟨"Tossing a coin twice in a row and getting heads both times"⟩

/-- Event 2: Opposite charges attract each other -/
def event2 : Event := ⟨"Opposite charges attract each other"⟩

/-- Event 3: Water freezes at 1°C under standard atmospheric pressure -/
def event3 : Event := ⟨"Water freezes at 1°C under standard atmospheric pressure"⟩

/-- Theorem: Only event1 is a random event among the given events -/
theorem only_event1_is_random :
  isRandomEvent event1 ∧ ¬isRandomEvent event2 ∧ ¬isRandomEvent event3 :=
by
  sorry


end NUMINAMATH_CALUDE_only_event1_is_random_l237_23793


namespace NUMINAMATH_CALUDE_uncle_li_parking_duration_l237_23741

/-- Calculates the parking duration given the total amount paid and the fee structure -/
def parking_duration (total_paid : ℚ) (first_hour_fee : ℚ) (additional_half_hour_fee : ℚ) : ℚ :=
  (total_paid - first_hour_fee) / (additional_half_hour_fee / (1/2)) + 1

theorem uncle_li_parking_duration :
  let total_paid : ℚ := 25/2
  let first_hour_fee : ℚ := 5/2
  let additional_half_hour_fee : ℚ := 5/2
  parking_duration total_paid first_hour_fee additional_half_hour_fee = 3 := by
sorry

end NUMINAMATH_CALUDE_uncle_li_parking_duration_l237_23741


namespace NUMINAMATH_CALUDE_constant_term_expansion_l237_23755

theorem constant_term_expansion (k : ℕ+) : k = 1 ↔ k^4 * (Nat.choose 6 4) < 120 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l237_23755


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l237_23746

/-- Proves that the common difference of an arithmetic sequence is 5,
    given the specified conditions. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ) -- First term
  (a_n : ℕ) -- Last term
  (S : ℕ) -- Sum of all terms
  (h_a : a = 5)
  (h_a_n : a_n = 50)
  (h_S : S = 275)
  : ∃ (n : ℕ) (d : ℕ), n > 1 ∧ d = 5 ∧ 
    a_n = a + (n - 1) * d ∧
    S = n * (a + a_n) / 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l237_23746


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l237_23705

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (3 * x + 7) = 10 → x = 31 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l237_23705


namespace NUMINAMATH_CALUDE_ratio_of_q_r_to_p_l237_23728

def p : ℝ := 47.99999999999999

theorem ratio_of_q_r_to_p : ∃ (f : ℝ), f = 1/6 ∧ 2 * f * p = p - 32 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_q_r_to_p_l237_23728


namespace NUMINAMATH_CALUDE_n_accurate_to_hundred_thousandth_l237_23723

/-- The number we're considering -/
def n : ℝ := 5.374e8

/-- Definition of accuracy to the hundred thousandth place -/
def accurate_to_hundred_thousandth (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (k : ℝ) * 1e5

/-- Theorem stating that our number is accurate to the hundred thousandth place -/
theorem n_accurate_to_hundred_thousandth : accurate_to_hundred_thousandth n := by
  sorry

end NUMINAMATH_CALUDE_n_accurate_to_hundred_thousandth_l237_23723


namespace NUMINAMATH_CALUDE_sqrt_14_bounds_l237_23730

theorem sqrt_14_bounds : 3 < Real.sqrt 14 ∧ Real.sqrt 14 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_14_bounds_l237_23730


namespace NUMINAMATH_CALUDE_interesting_number_expected_value_l237_23737

/-- A type representing a 6-digit number with specific properties -/
structure InterestingNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  f : Nat
  a_positive : a > 0
  b_positive : b > 0
  c_positive : c > 0
  d_positive : d > 0
  e_positive : e > 0
  f_positive : f > 0
  a_less_b : a < b
  b_less_c : b < c
  d_ge_e : d ≥ e
  e_ge_f : e ≥ f
  a_le_9 : a ≤ 9
  b_le_9 : b ≤ 9
  c_le_9 : c ≤ 9
  d_le_9 : d ≤ 9
  e_le_9 : e ≤ 9
  f_le_9 : f ≤ 9

/-- The expected value of an interesting number -/
def expectedValue (n : InterestingNumber) : ℝ :=
  100000 * n.a + 10000 * n.b + 1000 * n.c + 100 * n.d + 10 * n.e + n.f

/-- The theorem stating the expected value of all interesting numbers -/
theorem interesting_number_expected_value :
  ∃ (μ : ℝ), ∀ (n : InterestingNumber), μ = 308253 := by
  sorry

end NUMINAMATH_CALUDE_interesting_number_expected_value_l237_23737


namespace NUMINAMATH_CALUDE_distinct_permutations_eq_twelve_l237_23791

/-- The number of distinct permutations of the multiset {2, 3, 3, 9} -/
def distinct_permutations : ℕ :=
  Nat.factorial 4 / Nat.factorial 2

/-- Theorem stating that the number of distinct permutations of the multiset {2, 3, 3, 9} is 12 -/
theorem distinct_permutations_eq_twelve : distinct_permutations = 12 := by
  sorry

end NUMINAMATH_CALUDE_distinct_permutations_eq_twelve_l237_23791


namespace NUMINAMATH_CALUDE_m_range_proof_l237_23739

theorem m_range_proof (h : ∀ x, (|x - m| < 1) ↔ (1/3 < x ∧ x < 1/2)) :
  -1/2 ≤ m ∧ m ≤ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_m_range_proof_l237_23739


namespace NUMINAMATH_CALUDE_average_of_seven_thirteen_and_n_l237_23707

theorem average_of_seven_thirteen_and_n (N : ℝ) (h1 : 15 < N) (h2 : N < 25) :
  (7 + 13 + N) / 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_of_seven_thirteen_and_n_l237_23707


namespace NUMINAMATH_CALUDE_necessary_sufficient_condition_l237_23713

theorem necessary_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0) ↔ a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_necessary_sufficient_condition_l237_23713


namespace NUMINAMATH_CALUDE_teachers_per_grade_l237_23763

theorem teachers_per_grade (fifth_graders sixth_graders seventh_graders : ℕ)
  (parents_per_grade : ℕ) (num_buses seat_per_bus : ℕ) (num_grades : ℕ) :
  fifth_graders = 109 →
  sixth_graders = 115 →
  seventh_graders = 118 →
  parents_per_grade = 2 →
  num_buses = 5 →
  seat_per_bus = 72 →
  num_grades = 3 →
  (num_buses * seat_per_bus - (fifth_graders + sixth_graders + seventh_graders + parents_per_grade * num_grades)) / num_grades = 4 := by
  sorry

end NUMINAMATH_CALUDE_teachers_per_grade_l237_23763


namespace NUMINAMATH_CALUDE_shooter_stability_l237_23721

/-- A shooter's score set -/
structure ScoreSet where
  scores : Finset ℝ
  card_eq : scores.card = 10

/-- Standard deviation of a score set -/
def standardDeviation (s : ScoreSet) : ℝ := sorry

/-- Dispersion of a score set -/
def dispersion (s : ScoreSet) : ℝ := sorry

/-- Larger standard deviation implies greater dispersion -/
axiom std_dev_dispersion_relation (s₁ s₂ : ScoreSet) :
  standardDeviation s₁ > standardDeviation s₂ → dispersion s₁ > dispersion s₂

theorem shooter_stability (A B : ScoreSet) :
  standardDeviation A > standardDeviation B →
  dispersion A > dispersion B :=
by sorry

end NUMINAMATH_CALUDE_shooter_stability_l237_23721


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l237_23738

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l237_23738


namespace NUMINAMATH_CALUDE_quiz_average_after_drop_l237_23797

theorem quiz_average_after_drop (n : ℕ) (initial_avg : ℚ) (dropped_score : ℕ) :
  n = 16 →
  initial_avg = 60.5 →
  dropped_score = 8 →
  let total_score := n * initial_avg
  let remaining_score := total_score - dropped_score
  let new_avg := remaining_score / (n - 1)
  new_avg = 64 := by sorry

end NUMINAMATH_CALUDE_quiz_average_after_drop_l237_23797


namespace NUMINAMATH_CALUDE_student_count_l237_23792

theorem student_count (total_average : ℝ) (group1_count : ℕ) (group1_average : ℝ)
                      (group2_count : ℕ) (group2_average : ℝ) (last_student_age : ℕ) :
  total_average = 15 →
  group1_count = 8 →
  group1_average = 14 →
  group2_count = 6 →
  group2_average = 16 →
  last_student_age = 17 →
  (group1_count : ℝ) * group1_average + (group2_count : ℝ) * group2_average + last_student_age = 15 * 15 :=
by sorry

end NUMINAMATH_CALUDE_student_count_l237_23792


namespace NUMINAMATH_CALUDE_books_combination_l237_23798

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem books_combination : choose 15 3 = 455 := by sorry

end NUMINAMATH_CALUDE_books_combination_l237_23798


namespace NUMINAMATH_CALUDE_conference_children_count_l237_23781

theorem conference_children_count :
  let total_men : ℕ := 700
  let total_women : ℕ := 500
  let indian_men_percentage : ℚ := 20 / 100
  let indian_women_percentage : ℚ := 40 / 100
  let indian_children_percentage : ℚ := 10 / 100
  let non_indian_percentage : ℚ := 79 / 100
  ∃ (total_children : ℕ),
    (indian_men_percentage * total_men +
     indian_women_percentage * total_women +
     indian_children_percentage * total_children : ℚ) =
    ((1 - non_indian_percentage) * (total_men + total_women + total_children) : ℚ) ∧
    total_children = 800 :=
by sorry

end NUMINAMATH_CALUDE_conference_children_count_l237_23781


namespace NUMINAMATH_CALUDE_inequality_implies_sum_nonnegative_l237_23747

theorem inequality_implies_sum_nonnegative (a b : ℝ) :
  Real.exp a + Real.pi ^ b ≥ Real.exp (-b) + Real.pi ^ (-a) → a + b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_sum_nonnegative_l237_23747


namespace NUMINAMATH_CALUDE_triangle_least_perimeter_l237_23714

theorem triangle_least_perimeter (a b x : ℕ) : 
  a = 15 → b = 24 → x > 0 → 
  a + x > b → b + x > a → a + b > x → 
  (∀ y : ℕ, y > 0 → y + a > b → b + y > a → a + b > y → a + b + y ≥ a + b + x) →
  a + b + x = 49 :=
sorry

end NUMINAMATH_CALUDE_triangle_least_perimeter_l237_23714


namespace NUMINAMATH_CALUDE_article_cost_l237_23773

/-- Represents the cost and selling price of an article -/
structure Article where
  cost : ℝ
  sellingPrice : ℝ

/-- The original article with 25% profit -/
def originalArticle : Article → Prop := fun a => 
  a.sellingPrice = 1.25 * a.cost

/-- The new article with reduced cost and selling price -/
def newArticle : Article → Prop := fun a => 
  (0.8 * a.cost) * 1.3 = a.sellingPrice - 16.8

/-- Theorem stating that the cost of the article is 80 -/
theorem article_cost : ∃ a : Article, originalArticle a ∧ newArticle a ∧ a.cost = 80 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_l237_23773


namespace NUMINAMATH_CALUDE_nickel_count_l237_23720

theorem nickel_count (total_value : ℚ) (nickel_value : ℚ) (quarter_value : ℚ) :
  total_value = 12 →
  nickel_value = 0.05 →
  quarter_value = 0.25 →
  ∃ n : ℕ, n * nickel_value + n * quarter_value = total_value ∧ n = 40 :=
by sorry

end NUMINAMATH_CALUDE_nickel_count_l237_23720


namespace NUMINAMATH_CALUDE_total_balls_l237_23711

theorem total_balls (jungkook_red_balls : ℕ) (yoongi_blue_balls : ℕ) 
  (h1 : jungkook_red_balls = 3) (h2 : yoongi_blue_balls = 4) : 
  jungkook_red_balls + yoongi_blue_balls = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_l237_23711


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l237_23796

theorem necessary_not_sufficient_condition (a : ℝ) : 
  (∀ a, (1 / a > 1 → a < 1)) ∧ 
  (∃ a, a < 1 ∧ ¬(1 / a > 1)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l237_23796


namespace NUMINAMATH_CALUDE_expression_equality_l237_23733

theorem expression_equality (y a : ℝ) (h1 : y > 0) 
  (h2 : (a * y) / 20 + (3 * y) / 10 = 0.6 * y) : a = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l237_23733


namespace NUMINAMATH_CALUDE_equation_solution_l237_23702

theorem equation_solution : ∃! y : ℚ, 
  (y ≠ 3 ∧ y ≠ 5/4) ∧ 
  (y^2 - 7*y + 12)/(y - 3) + (4*y^2 + 20*y - 25)/(4*y - 5) = 2 ∧
  y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l237_23702


namespace NUMINAMATH_CALUDE_solution_existence_implies_a_bound_l237_23764

theorem solution_existence_implies_a_bound (a : ℝ) :
  (∃ x : ℝ, |x + a| + |x - 2| + a < 2010) → a < 1006 := by
  sorry

end NUMINAMATH_CALUDE_solution_existence_implies_a_bound_l237_23764


namespace NUMINAMATH_CALUDE_undefined_expression_expression_undefined_iff_x_eq_12_l237_23765

theorem undefined_expression (x : ℝ) : 
  (x^2 - 24*x + 144 = 0) ↔ (x = 12) := by sorry

theorem expression_undefined_iff_x_eq_12 :
  ∀ x : ℝ, (∃ y : ℝ, (3*x^3 - 5*x + 2) / (x^2 - 24*x + 144) = y) ↔ (x ≠ 12) := by sorry

end NUMINAMATH_CALUDE_undefined_expression_expression_undefined_iff_x_eq_12_l237_23765


namespace NUMINAMATH_CALUDE_distance_focus_to_asymptote_l237_23722

-- Define the hyperbola
def hyperbola (m : ℝ) (x y : ℝ) : Prop := y^2 - m*x^2 = 3*m

-- Define the condition that m is positive
def m_positive (m : ℝ) : Prop := m > 0

-- Define a focus of the hyperbola
def is_focus (m : ℝ) (F : ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), a^2 = 3*m ∧ b^2 = 3 ∧ c^2 = a^2 + b^2 ∧ 
  (F.1 = 0 ∧ F.2 = c ∨ F.1 = 0 ∧ F.2 = -c)

-- Define an asymptote of the hyperbola
def is_asymptote (m : ℝ) (l : ℝ → ℝ) : Prop :=
  ∀ x, l x = Real.sqrt m * x ∨ l x = -Real.sqrt m * x

-- Theorem statement
theorem distance_focus_to_asymptote (m : ℝ) (F : ℝ × ℝ) (l : ℝ → ℝ) :
  m_positive m →
  hyperbola m F.1 F.2 →
  is_focus m F →
  is_asymptote m l →
  ∃ (d : ℝ), d = Real.sqrt 3 ∧ 
    d = |F.2 - l F.1| / Real.sqrt (1 + (Real.sqrt m)^2) :=
sorry

end NUMINAMATH_CALUDE_distance_focus_to_asymptote_l237_23722


namespace NUMINAMATH_CALUDE_division_problem_l237_23750

theorem division_problem (dividend quotient remainder divisor : ℕ) 
  (h1 : dividend = 52)
  (h2 : quotient = 16)
  (h3 : remainder = 4)
  (h4 : dividend = divisor * quotient + remainder) :
  divisor = 3 := by sorry

end NUMINAMATH_CALUDE_division_problem_l237_23750


namespace NUMINAMATH_CALUDE_multiple_factor_statement_l237_23734

theorem multiple_factor_statement (h : 8 * 9 = 72) : ¬(∃ k : ℕ, 72 = 8 * k ∧ ∃ m : ℕ, 72 = m * 8) :=
sorry

end NUMINAMATH_CALUDE_multiple_factor_statement_l237_23734


namespace NUMINAMATH_CALUDE_trajectory_equation_l237_23767

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (-1, -2)

-- Define the vector equation
def vector_equation (C : ℝ × ℝ) (s t : ℝ) : Prop :=
  C = (s * A.1 + t * B.1, s * A.2 + t * B.2)

-- Define the constraint
def constraint (s t : ℝ) : Prop := s + t = 1

-- Theorem statement
theorem trajectory_equation :
  ∀ (C : ℝ × ℝ) (s t : ℝ),
  vector_equation C s t → constraint s t →
  C.1 - C.2 - 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l237_23767


namespace NUMINAMATH_CALUDE_derivative_ln_at_e_l237_23762

open Real

theorem derivative_ln_at_e (f : ℝ → ℝ) (h : ∀ x, f x = log x) : 
  deriv f e = 1 / e := by
  sorry

end NUMINAMATH_CALUDE_derivative_ln_at_e_l237_23762


namespace NUMINAMATH_CALUDE_discount_profit_percentage_l237_23717

theorem discount_profit_percentage 
  (discount : Real) 
  (no_discount_profit : Real) 
  (h1 : discount = 0.05) 
  (h2 : no_discount_profit = 0.26) : 
  let marked_price := 1 + no_discount_profit
  let selling_price := marked_price * (1 - discount)
  let profit := selling_price - 1
  profit * 100 = 19.7 := by
sorry

end NUMINAMATH_CALUDE_discount_profit_percentage_l237_23717


namespace NUMINAMATH_CALUDE_parallel_intersection_l237_23732

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Define the intersection operation for planes
variable (intersection : Plane → Plane → Line)

-- State the theorem
theorem parallel_intersection
  (l₁ l₂ l₃ : Line) (α β : Plane)
  (h1 : parallel l₁ l₂)
  (h2 : subset l₁ α)
  (h3 : subset l₂ β)
  (h4 : intersection α β = l₃) :
  parallel l₁ l₃ :=
sorry

end NUMINAMATH_CALUDE_parallel_intersection_l237_23732


namespace NUMINAMATH_CALUDE_triangle_side_sum_l237_23716

theorem triangle_side_sum (a b c : ℝ) (A B C : ℝ) :
  c = 2 →
  C = π / 3 →
  (1 / 2) * a * b * Real.sin C = Real.sqrt 3 →
  c^2 = a^2 + b^2 - 2 * a * b * Real.cos C →
  a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l237_23716


namespace NUMINAMATH_CALUDE_equation_solution_l237_23771

theorem equation_solution :
  ∃! x : ℚ, 7 * (4 * x + 3) - 5 = -3 * (2 - 5 * x) ∧ x = -22 / 13 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l237_23771


namespace NUMINAMATH_CALUDE_alice_winning_strategy_l237_23788

theorem alice_winning_strategy :
  ∀ (n : ℕ), n < 10000000 ∧ n ≥ 1000000 →
  (n % 10 = 1 ∨ n % 10 = 3 ∨ n % 10 = 5 ∨ n % 10 = 7 ∨ n % 10 = 9) →
  ∃ (k : ℕ), k^7 % 10000000 = n := by
sorry

end NUMINAMATH_CALUDE_alice_winning_strategy_l237_23788


namespace NUMINAMATH_CALUDE_two_valid_plans_l237_23756

/-- The number of valid purchasing plans for notebooks and pens -/
def valid_purchasing_plans : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    p.1 > 0 ∧ p.2 > 0 ∧ 3 * p.1 + 5 * p.2 = 35) 
    (Finset.product (Finset.range 36) (Finset.range 36))).card

/-- Theorem stating that there are exactly 2 valid purchasing plans -/
theorem two_valid_plans : valid_purchasing_plans = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_valid_plans_l237_23756


namespace NUMINAMATH_CALUDE_two_digit_primes_with_units_digit_9_l237_23782

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def has_units_digit_9 (n : ℕ) : Prop := n % 10 = 9

theorem two_digit_primes_with_units_digit_9 :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, is_two_digit n ∧ has_units_digit_9 n ∧ Nat.Prime n) ∧ 
    (∀ n, is_two_digit n → has_units_digit_9 n → Nat.Prime n → n ∈ s) ∧
    s.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_primes_with_units_digit_9_l237_23782


namespace NUMINAMATH_CALUDE_simplify_expression_l237_23710

theorem simplify_expression (x : ℝ) : x + 3 - 4*x - 5 + 6*x + 7 - 8*x - 9 = -5*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l237_23710


namespace NUMINAMATH_CALUDE_vacuum_savings_theorem_l237_23736

/-- The number of weeks needed to save for a vacuum cleaner. -/
def weeks_to_save (initial_savings : ℕ) (weekly_savings : ℕ) (vacuum_cost : ℕ) : ℕ :=
  ((vacuum_cost - initial_savings) + weekly_savings - 1) / weekly_savings

/-- Theorem stating that it takes 10 weeks to save for the vacuum cleaner. -/
theorem vacuum_savings_theorem :
  weeks_to_save 20 10 120 = 10 := by
  sorry

end NUMINAMATH_CALUDE_vacuum_savings_theorem_l237_23736


namespace NUMINAMATH_CALUDE_unreachable_y_value_l237_23768

theorem unreachable_y_value (x : ℝ) (h : x ≠ -5/4) :
  ¬∃y : ℝ, y = -3/4 ∧ y = (2 - 3*x) / (4*x + 5) :=
by sorry

end NUMINAMATH_CALUDE_unreachable_y_value_l237_23768


namespace NUMINAMATH_CALUDE_exists_large_number_with_exchangeable_digits_l237_23754

/-- A function that checks if two natural numbers have the same set of prime divisors -/
def samePrimeDivisors (a b : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ a ↔ p ∣ b)

/-- A function that checks if a number can have two distinct non-zero digits exchanged -/
def canExchangeDigits (n : ℕ) : Prop :=
  ∃ (d₁ d₂ : ℕ) (k m : ℕ),
    d₁ ≠ d₂ ∧ d₁ ≠ 0 ∧ d₂ ≠ 0 ∧
    (∃ n₁ n₂ : ℕ,
      n₁ = n + (d₁ - d₂) * 10^k ∧
      n₂ = n + (d₂ - d₁) * 10^m ∧
      samePrimeDivisors n₁ n₂)

/-- The main theorem -/
theorem exists_large_number_with_exchangeable_digits :
  ∃ n : ℕ, n > 10^1000 ∧ ¬(10 ∣ n) ∧ canExchangeDigits n :=
sorry

end NUMINAMATH_CALUDE_exists_large_number_with_exchangeable_digits_l237_23754


namespace NUMINAMATH_CALUDE_shortest_side_length_l237_23770

/-- Triangle with specific properties -/
structure SpecialTriangle where
  -- Base of the triangle
  base : ℝ
  -- One base angle in radians
  baseAngle : ℝ
  -- Sum of the other two sides
  sumOtherSides : ℝ
  -- Conditions
  base_positive : base > 0
  baseAngle_in_range : 0 < baseAngle ∧ baseAngle < π
  sumOtherSides_positive : sumOtherSides > 0

/-- The length of the shortest side in the special triangle -/
def shortestSide (t : SpecialTriangle) : ℝ := sorry

/-- Theorem stating the length of the shortest side in the specific triangle -/
theorem shortest_side_length (t : SpecialTriangle) 
  (h1 : t.base = 80)
  (h2 : t.baseAngle = π / 3)  -- 60° in radians
  (h3 : t.sumOtherSides = 90) :
  shortestSide t = 40 := by sorry

end NUMINAMATH_CALUDE_shortest_side_length_l237_23770


namespace NUMINAMATH_CALUDE_product_evaluation_l237_23772

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l237_23772


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l237_23790

-- Define the right triangle
def rightTriangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

-- Define the inscribed rectangle
def inscribedRectangle (x : ℝ) (a b c : ℝ) : Prop :=
  rightTriangle a b c ∧ x > 0 ∧ 2*x > 0 ∧ x ≤ a ∧ 2*x ≤ b

-- Theorem statement
theorem inscribed_rectangle_area (x : ℝ) :
  rightTriangle 24 (60 - 24) 60 →
  inscribedRectangle x 24 (60 - 24) 60 →
  x * (2*x) = 1440 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l237_23790


namespace NUMINAMATH_CALUDE_afternoon_eggs_calculation_l237_23795

theorem afternoon_eggs_calculation (total_eggs day_eggs morning_eggs : ℕ) 
  (h1 : total_eggs = 1339)
  (h2 : morning_eggs = 816)
  (h3 : day_eggs = total_eggs - morning_eggs) : 
  day_eggs = 523 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_eggs_calculation_l237_23795


namespace NUMINAMATH_CALUDE_solve_movie_problem_l237_23779

def movie_problem (ticket_count : ℕ) (rental_cost bought_cost total_cost : ℚ) : Prop :=
  ticket_count = 2 ∧
  rental_cost = 1.59 ∧
  bought_cost = 13.95 ∧
  total_cost = 36.78 ∧
  ∃ (ticket_price : ℚ),
    ticket_price * ticket_count + rental_cost + bought_cost = total_cost ∧
    ticket_price = 10.62

theorem solve_movie_problem :
  ∃ (ticket_count : ℕ) (rental_cost bought_cost total_cost : ℚ),
    movie_problem ticket_count rental_cost bought_cost total_cost :=
  sorry

end NUMINAMATH_CALUDE_solve_movie_problem_l237_23779


namespace NUMINAMATH_CALUDE_log_xyz_t_equals_three_l237_23775

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_xyz_t_equals_three 
  (t x y z : ℝ) 
  (h1 : log x t = 6)
  (h2 : log y t = 10)
  (h3 : log z t = 15) :
  log (x * y * z) t = 3 :=
by sorry

end NUMINAMATH_CALUDE_log_xyz_t_equals_three_l237_23775


namespace NUMINAMATH_CALUDE_diana_hourly_wage_l237_23774

/-- Diana's work schedule and earnings --/
structure DianaWork where
  monday_hours : ℕ
  tuesday_hours : ℕ
  wednesday_hours : ℕ
  thursday_hours : ℕ
  friday_hours : ℕ
  weekly_earnings : ℕ

/-- Calculate Diana's hourly wage --/
def hourly_wage (d : DianaWork) : ℚ :=
  d.weekly_earnings / (d.monday_hours + d.tuesday_hours + d.wednesday_hours + d.thursday_hours + d.friday_hours)

/-- Theorem: Diana's hourly wage is $30 --/
theorem diana_hourly_wage :
  let d : DianaWork := {
    monday_hours := 10,
    tuesday_hours := 15,
    wednesday_hours := 10,
    thursday_hours := 15,
    friday_hours := 10,
    weekly_earnings := 1800
  }
  hourly_wage d = 30 := by sorry

end NUMINAMATH_CALUDE_diana_hourly_wage_l237_23774


namespace NUMINAMATH_CALUDE_triangle_abc_is_right_angled_l237_23787

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the 2D plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A parabola in the 2D plane -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Theorem: Given point A(1, 2) and a line passing through (5, -2) that intersects
    the parabola y^2 = 4x at points B and C, triangle ABC is right-angled -/
theorem triangle_abc_is_right_angled 
  (A : Point)
  (l : Line)
  (p : Parabola)
  (B C : Point)
  (h1 : A.x = 1 ∧ A.y = 2)
  (h2 : l.slope * (-2) + l.intercept = 5)
  (h3 : p.a = 4 ∧ p.h = 0 ∧ p.k = 0)
  (h4 : B.y^2 = 4 * B.x ∧ B.y = l.slope * B.x + l.intercept)
  (h5 : C.y^2 = 4 * C.x ∧ C.y = l.slope * C.x + l.intercept)
  (h6 : B ≠ C) :
  (B.x - A.x) * (C.x - A.x) + (B.y - A.y) * (C.y - A.y) = 0 := by
  sorry


end NUMINAMATH_CALUDE_triangle_abc_is_right_angled_l237_23787


namespace NUMINAMATH_CALUDE_line_through_points_l237_23740

/-- Given a line y = ax + b passing through points (3,7) and (7,19), prove that a - b = 5 -/
theorem line_through_points (a b : ℝ) : 
  (3 * a + b = 7) → (7 * a + b = 19) → a - b = 5 := by sorry

end NUMINAMATH_CALUDE_line_through_points_l237_23740


namespace NUMINAMATH_CALUDE_two_inscribed_cube_lengths_l237_23760

/-- A regular tetrahedron with unit edge length -/
structure RegularTetrahedron where
  edge_length : ℝ
  is_unit : edge_length = 1

/-- A cube inscribed in a tetrahedron such that each vertex lies on a face of the tetrahedron -/
structure InscribedCube where
  edge_length : ℝ
  vertices_on_faces : True  -- This is a placeholder for the geometric condition

/-- The set of all possible edge lengths for inscribed cubes in a unit regular tetrahedron -/
def inscribed_cube_edge_lengths (t : RegularTetrahedron) : Set ℝ :=
  {l | ∃ c : InscribedCube, c.edge_length = l}

/-- Theorem stating that there are exactly two distinct edge lengths for inscribed cubes -/
theorem two_inscribed_cube_lengths (t : RegularTetrahedron) :
  ∃ l₁ l₂ : ℝ, l₁ ≠ l₂ ∧ inscribed_cube_edge_lengths t = {l₁, l₂} :=
sorry

end NUMINAMATH_CALUDE_two_inscribed_cube_lengths_l237_23760


namespace NUMINAMATH_CALUDE_birthday_party_guests_solve_birthday_party_guests_l237_23718

theorem birthday_party_guests : ℕ → Prop :=
  fun total_guests =>
    -- Define the number of women, men, and children
    let women := total_guests / 2
    let men := 15
    let children := total_guests - women - men

    -- Define the number of people who left
    let men_left := men / 3
    let children_left := 5

    -- Define the number of people who stayed
    let people_stayed := total_guests - men_left - children_left

    -- State the conditions and the conclusion
    women = men ∧
    women + men + children = total_guests ∧
    people_stayed = 50 ∧
    total_guests = 60

-- The proof of the theorem
theorem solve_birthday_party_guests : birthday_party_guests 60 := by
  sorry

#check solve_birthday_party_guests

end NUMINAMATH_CALUDE_birthday_party_guests_solve_birthday_party_guests_l237_23718


namespace NUMINAMATH_CALUDE_bus_remaining_distance_l237_23761

def distance_between_points (z : ℝ) : Prop :=
  ∃ (x : ℝ), x > 0 ∧
  (z / 2) / (z - 19.2) = x ∧
  (z - 12) / (z / 2) = x

theorem bus_remaining_distance (z : ℝ) (h : distance_between_points z) :
  z - z * (4/5) = 6.4 :=
sorry

end NUMINAMATH_CALUDE_bus_remaining_distance_l237_23761


namespace NUMINAMATH_CALUDE_function_inequality_l237_23777

/-- Given functions f and g, prove that g(x) > f(x) + kx - 1 for all x > 0 and a ∈ (0, e^2/2] -/
theorem function_inequality (k : ℝ) :
  ∀ (x a : ℝ), x > 0 → 0 < a → a ≤ Real.exp 2 / 2 →
  (Real.exp x) / (a * x) > Real.log x - k * x + 1 + k * x - 1 := by
  sorry


end NUMINAMATH_CALUDE_function_inequality_l237_23777


namespace NUMINAMATH_CALUDE_roots_count_lower_bound_l237_23753

def count_roots (f : ℝ → ℝ) (a b : ℝ) : ℕ :=
  sorry

theorem roots_count_lower_bound
  (f : ℝ → ℝ)
  (h1 : ∀ x, f (3 + x) = f (3 - x))
  (h2 : ∀ x, f (9 + x) = f (9 - x))
  (h3 : f 1 = 0) :
  count_roots f (-1000) 1000 ≥ 334 := by
  sorry

end NUMINAMATH_CALUDE_roots_count_lower_bound_l237_23753


namespace NUMINAMATH_CALUDE_log_base_is_two_range_of_m_l237_23757

noncomputable section

-- Define the logarithm function with base a
def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := log_base a x

-- Theorem 1: If f(x) = log_a(x), a > 0, a ≠ 1, and f(2) = 1, then f(x) = log_2(x)
theorem log_base_is_two (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 2 = 1) :
  ∀ x > 0, f a x = log_base 2 x :=
sorry

-- Theorem 2: For f(x) = log_2(x), the set of real numbers m satisfying f(m^2 - m) < 1 is (-1,0) ∪ (1,2)
theorem range_of_m (m : ℝ) :
  log_base 2 (m^2 - m) < 1 ↔ (m > -1 ∧ m < 0) ∨ (m > 1 ∧ m < 2) :=
sorry

end

end NUMINAMATH_CALUDE_log_base_is_two_range_of_m_l237_23757


namespace NUMINAMATH_CALUDE_fifteen_times_number_equals_three_hundred_l237_23769

theorem fifteen_times_number_equals_three_hundred :
  ∃ x : ℝ, 15 * x = 300 ∧ x = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_fifteen_times_number_equals_three_hundred_l237_23769


namespace NUMINAMATH_CALUDE_population_decrease_proof_l237_23786

/-- The annual rate of population decrease -/
def annual_decrease_rate : ℝ := 0.1

/-- The population after 2 years -/
def population_after_2_years : ℕ := 6480

/-- The initial population of the town -/
def initial_population : ℕ := 8000

theorem population_decrease_proof :
  (1 - annual_decrease_rate)^2 * initial_population = population_after_2_years :=
by sorry

end NUMINAMATH_CALUDE_population_decrease_proof_l237_23786


namespace NUMINAMATH_CALUDE_escalator_step_count_l237_23748

/-- Represents the number of steps a person counts while descending an escalator -/
def count_steps (escalator_length : ℕ) (walking_count : ℕ) (speed_multiplier : ℕ) : ℕ :=
  let escalator_speed := escalator_length - walking_count
  let speed_ratio := escalator_speed / walking_count
  let new_ratio := speed_ratio / speed_multiplier
  escalator_length / (new_ratio + 1)

/-- Theorem stating that given an escalator of 200 steps, where a person counts 50 steps while 
    walking down, the same person will count 80 steps when running twice as fast -/
theorem escalator_step_count :
  count_steps 200 50 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_escalator_step_count_l237_23748


namespace NUMINAMATH_CALUDE_oil_leak_calculation_l237_23789

theorem oil_leak_calculation (total_leaked : ℕ) (leaked_while_fixing : ℕ) 
  (h1 : total_leaked = 6206)
  (h2 : leaked_while_fixing = 3731) :
  total_leaked - leaked_while_fixing = 2475 := by
sorry

end NUMINAMATH_CALUDE_oil_leak_calculation_l237_23789


namespace NUMINAMATH_CALUDE_trig_identity_l237_23735

theorem trig_identity (α : Real) 
  (h : 3 * Real.sin α + Real.cos α = 0) : 
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l237_23735


namespace NUMINAMATH_CALUDE_min_positive_temperatures_l237_23719

theorem min_positive_temperatures (n : ℕ) (pos_products neg_products : ℕ) :
  n = 11 →
  pos_products = 62 →
  neg_products = 48 →
  ∃ (pos_temps : ℕ), pos_temps ≥ 3 ∧
    pos_temps * (pos_temps - 1) = pos_products ∧
    (n - pos_temps) * (n - 1 - pos_temps) = neg_products ∧
    ∀ (k : ℕ), k < pos_temps →
      k * (k - 1) ≠ pos_products ∨ (n - k) * (n - 1 - k) ≠ neg_products :=
by sorry

end NUMINAMATH_CALUDE_min_positive_temperatures_l237_23719


namespace NUMINAMATH_CALUDE_function_maximum_value_l237_23726

/-- Given a function f(x) = x / (x^2 + a) where a > 0, 
    if its maximum value on [1, +∞) is √3/3, then a = √3 - 1 -/
theorem function_maximum_value (a : ℝ) : 
  a > 0 → 
  (∀ x : ℝ, x ≥ 1 → x / (x^2 + a) ≤ Real.sqrt 3 / 3) →
  (∃ x : ℝ, x ≥ 1 ∧ x / (x^2 + a) = Real.sqrt 3 / 3) →
  a = Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_function_maximum_value_l237_23726


namespace NUMINAMATH_CALUDE_ratio_evaluation_l237_23783

theorem ratio_evaluation : 
  (5^3003 * 2^3005) / 10^3004 = 2/5 := by
sorry

end NUMINAMATH_CALUDE_ratio_evaluation_l237_23783


namespace NUMINAMATH_CALUDE_number_problem_l237_23784

theorem number_problem :
  ∃ (x : ℝ), ∃ (y : ℝ), 0.5 * x = y + 20 ∧ x - 2 * y = 40 ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l237_23784


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_2102012_base7_l237_23743

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- Gets the largest prime divisor of a number -/
def largestPrimeDivisor (n : ℕ) : ℕ := sorry

theorem largest_prime_divisor_of_2102012_base7 :
  largestPrimeDivisor (base7ToBase10 2102012) = 79 := by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_2102012_base7_l237_23743


namespace NUMINAMATH_CALUDE_shooting_competition_probability_l237_23712

theorem shooting_competition_probability 
  (p_single : ℝ) 
  (p_twice : ℝ) 
  (h1 : p_single = 4/5) 
  (h2 : p_twice = 1/2) : 
  p_twice / p_single = 5/8 := by
sorry

end NUMINAMATH_CALUDE_shooting_competition_probability_l237_23712


namespace NUMINAMATH_CALUDE_range_of_a_l237_23704

-- Define the set of real numbers x that satisfy 0 < x < 2
def P : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- Define the set of real numbers x that satisfy a-1 < x ≤ a
def Q (a : ℝ) : Set ℝ := {x : ℝ | a - 1 < x ∧ x ≤ a}

-- State the theorem
theorem range_of_a : 
  (∀ a : ℝ, (Q a ⊆ P) ∧ (Q a ≠ P)) → 
  {a : ℝ | 1 ≤ a ∧ a < 2} = {a : ℝ | ∃ x : ℝ, x ∈ Q a} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l237_23704


namespace NUMINAMATH_CALUDE_base_conversion_equality_l237_23727

theorem base_conversion_equality (b : ℕ) : b > 0 ∧ (4 * 6 + 2 = 1 * b^2 + 2 * b + 1) → b = 3 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_equality_l237_23727


namespace NUMINAMATH_CALUDE_largest_class_has_61_students_l237_23701

/-- Represents a school with a given number of classes and students. -/
structure School where
  num_classes : ℕ
  total_students : ℕ
  class_diff : ℕ

/-- Calculates the number of students in the largest class of a school. -/
def largest_class_size (s : School) : ℕ :=
  (s.total_students + s.class_diff * (s.num_classes - 1) * s.num_classes / 2) / s.num_classes

/-- Theorem stating that for a school with 8 classes, 380 total students,
    and 4 students difference between classes, the largest class has 61 students. -/
theorem largest_class_has_61_students :
  let s : School := { num_classes := 8, total_students := 380, class_diff := 4 }
  largest_class_size s = 61 := by
  sorry

#eval largest_class_size { num_classes := 8, total_students := 380, class_diff := 4 }

end NUMINAMATH_CALUDE_largest_class_has_61_students_l237_23701


namespace NUMINAMATH_CALUDE_december_ear_muff_sales_l237_23706

/-- The number of type B ear muffs sold in December -/
def type_b_count : ℕ := 3258

/-- The price of each type B ear muff -/
def type_b_price : ℚ := 69/10

/-- The number of type C ear muffs sold in December -/
def type_c_count : ℕ := 3186

/-- The price of each type C ear muff -/
def type_c_price : ℚ := 74/10

/-- The total amount spent on ear muffs in December -/
def total_spent : ℚ := type_b_count * type_b_price + type_c_count * type_c_price

theorem december_ear_muff_sales :
  total_spent = 460566/10 := by sorry

end NUMINAMATH_CALUDE_december_ear_muff_sales_l237_23706


namespace NUMINAMATH_CALUDE_hyperbola_equation_l237_23708

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    if its asymptote intersects the circle with its foci as diameter
    at the point (2, 1) in the first quadrant, then a = 2 and b = 1. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (c : ℝ), c > 0 ∧ 
    (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → y = (b / a) * x) ∧
    (2^2 + 1^2 = c^2) ∧
    (a^2 + b^2 = c^2)) →
  a = 2 ∧ b = 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l237_23708


namespace NUMINAMATH_CALUDE_unique_pairs_sum_product_l237_23724

theorem unique_pairs_sum_product (S P : ℝ) (h : S^2 ≥ 4*P) :
  ∃! (x₁ y₁ x₂ y₂ : ℝ),
    (x₁ + y₁ = S ∧ x₁ * y₁ = P) ∧
    (x₂ + y₂ = S ∧ x₂ * y₂ = P) ∧
    x₁ = (S + Real.sqrt (S^2 - 4*P)) / 2 ∧
    y₁ = S - x₁ ∧
    x₂ = (S - Real.sqrt (S^2 - 4*P)) / 2 ∧
    y₂ = S - x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_unique_pairs_sum_product_l237_23724


namespace NUMINAMATH_CALUDE_cookies_packages_bought_l237_23785

def num_children : ℕ := 5
def cookies_per_package : ℕ := 25
def cookies_per_child : ℕ := 15

theorem cookies_packages_bought : 
  (num_children * cookies_per_child) / cookies_per_package = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookies_packages_bought_l237_23785


namespace NUMINAMATH_CALUDE_total_cost_is_10_79_l237_23778

/-- The total cost of peppers purchased by Dale's Vegetarian Restaurant -/
def total_cost_peppers : ℝ :=
  let green_peppers := 2.8333333333333335
  let red_peppers := 3.254
  let yellow_peppers := 1.375
  let orange_peppers := 0.567
  let green_price := 1.20
  let red_price := 1.35
  let yellow_price := 1.50
  let orange_price := 1.65
  green_peppers * green_price +
  red_peppers * red_price +
  yellow_peppers * yellow_price +
  orange_peppers * orange_price

/-- Theorem stating that the total cost of peppers is $10.79 -/
theorem total_cost_is_10_79 : total_cost_peppers = 10.79 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_10_79_l237_23778


namespace NUMINAMATH_CALUDE_cone_angle_l237_23794

/-- Given a cone where the ratio of its lateral surface area to the area of the section through its axis
    is 2√3π/3, the angle between its generatrix and axis is π/6. -/
theorem cone_angle (r h l : ℝ) (θ : ℝ) : 
  r > 0 → h > 0 → l > 0 →
  (π * r * l) / (r * h) = 2 * Real.sqrt 3 * π / 3 →
  l = Real.sqrt ((r^2) + (h^2)) →
  θ = Real.arccos (h / l) →
  θ = π / 6 := by
  sorry

#check cone_angle

end NUMINAMATH_CALUDE_cone_angle_l237_23794


namespace NUMINAMATH_CALUDE_trajectory_of_point_M_l237_23766

/-- The trajectory of point M given the specified conditions -/
theorem trajectory_of_point_M :
  ∀ (M : ℝ × ℝ),
  let A : ℝ × ℝ := (0, -1)
  let B : ℝ × ℝ := (M.1, -3)
  let O : ℝ × ℝ := (0, 0)
  -- MB parallel to OA
  (∃ k : ℝ, B.1 - M.1 = k * A.1 ∧ B.2 - M.2 = k * A.2) →
  -- MA • AB = MB • BA
  ((A.1 - M.1) * (B.1 - A.1) + (A.2 - M.2) * (B.2 - A.2) =
   (B.1 - M.1) * (A.1 - B.1) + (B.2 - M.2) * (A.2 - B.2)) →
  -- Trajectory equation
  M.2 = (1/4) * M.1^2 - 2 := by
sorry

end NUMINAMATH_CALUDE_trajectory_of_point_M_l237_23766


namespace NUMINAMATH_CALUDE_expression_simplification_l237_23703

theorem expression_simplification (a : ℝ) (h : a = 2 + Real.sqrt 2) :
  (a / (a + 2) + 1 / (a^2 - 4)) / ((a - 1) / (a + 2)) + 1 / (a - 2) = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l237_23703


namespace NUMINAMATH_CALUDE_problem_statement_l237_23799

theorem problem_statement (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  (a + b + c = 0 → a*b + b*c + c*a = -1/2) ∧
  ((a + b + c)^2 ≤ 3 ∧ ∃ (x y z : ℝ), x^2 + y^2 + z^2 = 1 ∧ (x + y + z)^2 = 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l237_23799


namespace NUMINAMATH_CALUDE_jacket_price_calculation_l237_23715

/-- Calculates the final price of an item after applying three sequential discounts -/
def final_price (original_price : ℝ) (discount1 discount2 discount3 : ℝ) : ℝ :=
  original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)

/-- Theorem stating that the final price of a $250 jacket after three specific discounts is $94.5 -/
theorem jacket_price_calculation : 
  final_price 250 0.4 0.3 0.1 = 94.5 := by
  sorry

#eval final_price 250 0.4 0.3 0.1

end NUMINAMATH_CALUDE_jacket_price_calculation_l237_23715


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_m_l237_23776

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 3| - 5
def g (x : ℝ) : ℝ := |x + 2| - 2

-- Theorem for the solution set of f(x) ≤ 2
theorem solution_set_f (x : ℝ) : f x ≤ 2 ↔ -4 ≤ x ∧ x ≤ 10 := by sorry

-- Theorem for the range of m
theorem range_of_m : 
  ∀ m : ℝ, (∃ x : ℝ, f x - g x ≥ m - 3) ↔ m ≤ 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_m_l237_23776


namespace NUMINAMATH_CALUDE_davids_windows_l237_23742

/-- Represents the time taken to wash windows -/
def wash_time : ℕ := 160

/-- Represents the number of windows washed in a single set -/
def windows_per_set : ℕ := 4

/-- Represents the time taken to wash one set of windows -/
def time_per_set : ℕ := 10

/-- Theorem stating the number of windows in David's house -/
theorem davids_windows : 
  (wash_time / time_per_set) * windows_per_set = 64 := by
  sorry

end NUMINAMATH_CALUDE_davids_windows_l237_23742


namespace NUMINAMATH_CALUDE_no_real_roots_l237_23744

theorem no_real_roots : ¬ ∃ x : ℝ, Real.sqrt (x + 9) - Real.sqrt (x - 6) + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l237_23744


namespace NUMINAMATH_CALUDE_vector_operations_l237_23759

def a : Fin 2 → ℝ := ![3, 4]
def b : Fin 2 → ℝ := ![2, -6]
def c : Fin 2 → ℝ := ![4, 1]

theorem vector_operations :
  (a • (a + b) = 7) ∧
  (c = a + (1/2 : ℝ) • b) := by sorry

end NUMINAMATH_CALUDE_vector_operations_l237_23759


namespace NUMINAMATH_CALUDE_equation_solution_l237_23725

theorem equation_solution (x : ℝ) : 144 / 0.144 = 14.4 / x → x = 0.0144 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l237_23725


namespace NUMINAMATH_CALUDE_triangle_count_l237_23709

/-- The number of ways to choose k items from n items without replacement and where order doesn't matter -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of points on the circle -/
def num_points : ℕ := 9

/-- The number of points needed to form a triangle -/
def points_per_triangle : ℕ := 3

/-- The number of different triangles that can be formed -/
def num_triangles : ℕ := binomial num_points points_per_triangle

theorem triangle_count : num_triangles = 84 := by sorry

end NUMINAMATH_CALUDE_triangle_count_l237_23709


namespace NUMINAMATH_CALUDE_count_solutions_quadratic_congruence_l237_23729

theorem count_solutions_quadratic_congruence (p : Nat) (a : Int) 
  (h_p : p.Prime ∧ p > 2) :
  let S := {(x, y) : Fin p × Fin p | (x.val^2 + y.val^2) % p = a % p}
  Fintype.card S = p + 1 := by
sorry

end NUMINAMATH_CALUDE_count_solutions_quadratic_congruence_l237_23729


namespace NUMINAMATH_CALUDE_total_bottles_l237_23780

theorem total_bottles (juice : ℕ) (water : ℕ) : 
  juice = 34 → 
  water = (3 * juice) / 2 + 3 → 
  juice + water = 88 := by
sorry

end NUMINAMATH_CALUDE_total_bottles_l237_23780


namespace NUMINAMATH_CALUDE_wire_ratio_l237_23700

/-- Given a wire of total length 60 cm with a shorter piece of 20 cm,
    prove that the ratio of the shorter piece to the longer piece is 1/2. -/
theorem wire_ratio (total_length : ℝ) (shorter_piece : ℝ) 
  (h1 : total_length = 60)
  (h2 : shorter_piece = 20)
  (h3 : shorter_piece < total_length) :
  shorter_piece / (total_length - shorter_piece) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_wire_ratio_l237_23700


namespace NUMINAMATH_CALUDE_intersecting_lines_equality_l237_23751

/-- Given two linear functions y = ax + b and y = cx + d that intersect at (1, 0),
    prove that a^3 + c^2 = d^2 - b^3 -/
theorem intersecting_lines_equality (a b c d : ℝ) 
  (h1 : a * 1 + b = 0)  -- y = ax + b passes through (1, 0)
  (h2 : c * 1 + d = 0)  -- y = cx + d passes through (1, 0)
  : a^3 + c^2 = d^2 - b^3 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_equality_l237_23751


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l237_23731

/-- Given x is inversely proportional to y, prove that y₁/y₂ = 4/3 when x₁/x₂ = 3/4 -/
theorem inverse_proportion_ratio (x y : ℝ → ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (h_inverse : ∀ t : ℝ, t ≠ 0 → x t * y t = x₁ * y₁)
  (h_x₁_nonzero : x₁ ≠ 0)
  (h_x₂_nonzero : x₂ ≠ 0)
  (h_y₁_nonzero : y₁ ≠ 0)
  (h_y₂_nonzero : y₂ ≠ 0)
  (h_x_ratio : x₁ / x₂ = 3 / 4) :
  y₁ / y₂ = 4 / 3 := by
  sorry


end NUMINAMATH_CALUDE_inverse_proportion_ratio_l237_23731


namespace NUMINAMATH_CALUDE_exam_score_calculation_l237_23752

/-- Given an exam with mean score and a score below the mean, calculate the score above the mean -/
theorem exam_score_calculation (mean : ℝ) (below_score : ℝ) (below_sd : ℝ) (above_sd : ℝ)
  (h1 : mean = 76)
  (h2 : below_score = 60)
  (h3 : below_sd = 2)
  (h4 : above_sd = 3)
  (h5 : below_score = mean - below_sd * ((mean - below_score) / below_sd)) :
  mean + above_sd * ((mean - below_score) / below_sd) = 100 := by
sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l237_23752


namespace NUMINAMATH_CALUDE_container_volume_tripled_l237_23758

theorem container_volume_tripled (original_volume : ℝ) (h : original_volume = 2) :
  let new_volume := original_volume * 3 * 3 * 3
  new_volume = 54 := by
sorry

end NUMINAMATH_CALUDE_container_volume_tripled_l237_23758


namespace NUMINAMATH_CALUDE_line_point_x_coordinate_l237_23745

/-- Theorem: For a line passing through points (x₁, -4) and (5, 0.8) with slope 0.8, x₁ = -1 -/
theorem line_point_x_coordinate (x₁ : ℝ) : 
  let y₁ : ℝ := -4
  let x₂ : ℝ := 5
  let y₂ : ℝ := 0.8
  let k : ℝ := 0.8
  (y₂ - y₁) / (x₂ - x₁) = k → x₁ = -1 := by
  sorry

end NUMINAMATH_CALUDE_line_point_x_coordinate_l237_23745


namespace NUMINAMATH_CALUDE_successive_discounts_equivalence_l237_23749

/-- The equivalent single discount rate after applying successive discounts -/
def equivalent_discount (d1 d2 : ℝ) : ℝ :=
  1 - (1 - d1) * (1 - d2)

/-- Theorem stating that the equivalent single discount rate after applying
    successive discounts of 15% and 25% is 36.25% -/
theorem successive_discounts_equivalence :
  equivalent_discount 0.15 0.25 = 0.3625 := by
  sorry

end NUMINAMATH_CALUDE_successive_discounts_equivalence_l237_23749
