import Mathlib

namespace NUMINAMATH_CALUDE_angle_A_is_pi_third_max_area_is_sqrt_three_max_area_achieved_l3246_324608

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.a = 2 ∧ (2 + t.b) * (Real.sin t.A - Real.sin t.B) = (t.c - t.b) * Real.sin t.C

-- Theorem 1: Angle A is π/3
theorem angle_A_is_pi_third (t : Triangle) (h : satisfiesConditions t) : t.A = π / 3 := by
  sorry

-- Theorem 2: Maximum area is √3
theorem max_area_is_sqrt_three (t : Triangle) (h : satisfiesConditions t) : 
  (1/2 * t.b * t.c * Real.sin t.A) ≤ Real.sqrt 3 := by
  sorry

-- Theorem 2 (continued): The maximum area is achieved
theorem max_area_achieved (t : Triangle) : 
  ∃ (t : Triangle), satisfiesConditions t ∧ (1/2 * t.b * t.c * Real.sin t.A) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_third_max_area_is_sqrt_three_max_area_achieved_l3246_324608


namespace NUMINAMATH_CALUDE_range_of_3a_plus_4b_l3246_324692

theorem range_of_3a_plus_4b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : 2 * a + 2 * b ≤ 15) (h2 : 4 / a + 3 / b ≤ 2) :
  ∃ (min max : ℝ), min = 24 ∧ max = 27 ∧
  (∀ x, (∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧
    2 * a' + 2 * b' ≤ 15 ∧ 4 / a' + 3 / b' ≤ 2 ∧
    x = 3 * a' + 4 * b') → min ≤ x ∧ x ≤ max) :=
sorry

end NUMINAMATH_CALUDE_range_of_3a_plus_4b_l3246_324692


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l3246_324649

-- Define the points A and B
def A : ℝ × ℝ := (2, -2)
def B : ℝ × ℝ := (4, 3)

-- Define vector a as a function of k
def a (k : ℝ) : ℝ × ℝ := (2*k - 1, 7)

-- Define vector AB
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Theorem statement
theorem parallel_vectors_k_value : 
  ∃ (c : ℝ), c ≠ 0 ∧ a (19/10) = (c * AB.1, c * AB.2) :=
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l3246_324649


namespace NUMINAMATH_CALUDE_probability_sum_12_probability_sum_12_is_19_216_l3246_324603

/-- The number of faces on a standard die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ 3

/-- The number of ways to roll a sum of 12 with three dice -/
def waysToRoll12 : ℕ := 19

/-- The probability of rolling a sum of 12 with three standard six-faced dice -/
theorem probability_sum_12 : ℚ :=
  waysToRoll12 / totalOutcomes

/-- Proof that the probability of rolling a sum of 12 with three standard six-faced dice is 19/216 -/
theorem probability_sum_12_is_19_216 : probability_sum_12 = 19 / 216 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_12_probability_sum_12_is_19_216_l3246_324603


namespace NUMINAMATH_CALUDE_cone_height_l3246_324695

/-- The height of a cone given its lateral surface properties -/
theorem cone_height (r l : ℝ) (h : r > 0) (h' : l > 0) : 
  (l = 3) → (2 * Real.pi * r = 2 * Real.pi / 3 * 3) → 
  Real.sqrt (l^2 - r^2) = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_cone_height_l3246_324695


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extremum_l3246_324626

noncomputable def f (x : ℝ) := x * Real.exp (-x)

theorem f_monotonicity_and_extremum :
  (∀ x y : ℝ, x < y ∧ y < 1 → f x < f y) ∧
  (∀ x y : ℝ, 1 < x ∧ x < y → f y < f x) ∧
  (∀ x : ℝ, x ≠ 1 → f x < f 1) ∧
  f 1 = Real.exp (-1) := by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_extremum_l3246_324626


namespace NUMINAMATH_CALUDE_sin_cos_pi_twelve_eq_one_fourth_l3246_324662

theorem sin_cos_pi_twelve_eq_one_fourth : 
  Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_sin_cos_pi_twelve_eq_one_fourth_l3246_324662


namespace NUMINAMATH_CALUDE_tangent_line_at_e_l3246_324607

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_at_e :
  let p : ℝ × ℝ := (Real.exp 1, f (Real.exp 1))
  let m : ℝ := deriv f (Real.exp 1)
  let tangent_line (x : ℝ) : ℝ := m * (x - p.1) + p.2
  tangent_line = λ x => 2 * x - Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_e_l3246_324607


namespace NUMINAMATH_CALUDE_executive_committee_formation_l3246_324639

/-- Represents the number of members in each department -/
def membersPerDepartment : ℕ := 10

/-- Represents the total number of departments -/
def totalDepartments : ℕ := 3

/-- Represents the size of the executive committee -/
def committeeSize : ℕ := 5

/-- Represents the total number of club members -/
def totalMembers : ℕ := membersPerDepartment * totalDepartments

/-- Calculates the number of ways to choose the executive committee -/
def waysToChooseCommittee : ℕ := 
  membersPerDepartment ^ totalDepartments * (Nat.choose (totalMembers - totalDepartments) (committeeSize - totalDepartments))

theorem executive_committee_formation :
  waysToChooseCommittee = 351000 := by sorry

end NUMINAMATH_CALUDE_executive_committee_formation_l3246_324639


namespace NUMINAMATH_CALUDE_f_composition_value_l3246_324672

noncomputable section

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x else Real.log x

theorem f_composition_value : f (f (1 / Real.exp 1)) = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l3246_324672


namespace NUMINAMATH_CALUDE_triangle_angle_A_l3246_324680

theorem triangle_angle_A (A : Real) : 
  4 * Real.pi * Real.sin A - 3 * Real.arccos (-1/2) = 0 →
  (A = Real.pi / 6 ∨ A = 5 * Real.pi / 6) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l3246_324680


namespace NUMINAMATH_CALUDE_base_four_representation_of_256_l3246_324669

theorem base_four_representation_of_256 :
  (256 : ℕ).digits 4 = [0, 0, 0, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_base_four_representation_of_256_l3246_324669


namespace NUMINAMATH_CALUDE_total_arrangements_l3246_324688

/-- Represents the three elective math courses -/
inductive Course
| MatrixTransformation
| InfoSecCrypto
| SwitchCircuits

/-- Represents a teacher and their teaching capabilities -/
structure Teacher where
  id : Nat
  canTeach : Course → Bool

/-- The pool of available teachers -/
def teacherPool : Finset Teacher := sorry

/-- Teachers who can teach only Matrix and Transformation -/
def matrixOnlyTeachers : Finset Teacher := sorry

/-- Teachers who can teach only Information Security and Cryptography -/
def cryptoOnlyTeachers : Finset Teacher := sorry

/-- Teachers who can teach only Switch Circuits and Boolean Algebra -/
def switchOnlyTeachers : Finset Teacher := sorry

/-- Teachers who can teach all three courses -/
def versatileTeachers : Finset Teacher := sorry

/-- A valid selection of teachers for the courses -/
def isValidSelection (selection : Finset Teacher) : Prop := sorry

/-- The number of different valid arrangements -/
def numArrangements : Nat := sorry

theorem total_arrangements :
  (Finset.card teacherPool = 10) →
  (Finset.card matrixOnlyTeachers = 3) →
  (Finset.card cryptoOnlyTeachers = 2) →
  (Finset.card switchOnlyTeachers = 3) →
  (Finset.card versatileTeachers = 2) →
  (∀ s : Finset Teacher, isValidSelection s → Finset.card s = 9) →
  (∀ c : Course, ∀ s : Finset Teacher, isValidSelection s →
    Finset.card (s.filter (fun t => t.canTeach c)) = 3) →
  numArrangements = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_arrangements_l3246_324688


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3246_324660

theorem chess_tournament_games (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 7) : 
  (n * k) / 2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3246_324660


namespace NUMINAMATH_CALUDE_parabola_uniqueness_l3246_324691

/-- A tangent line to a parabola -/
structure Tangent where
  line : Line2D

/-- A parabola in 2D space -/
structure Parabola where
  focus : Point2D
  directrix : Line2D

/-- The vertex tangent of a parabola -/
def vertexTangent (p : Parabola) : Tangent :=
  sorry

/-- Determines if a given tangent is valid for a parabola -/
def isValidTangent (p : Parabola) (t : Tangent) : Prop :=
  sorry

theorem parabola_uniqueness 
  (t : Tangent) (t₁ : Tangent) (t₂ : Tangent) : 
  ∃! p : Parabola, 
    (vertexTangent p = t) ∧ 
    (isValidTangent p t₁) ∧ 
    (isValidTangent p t₂) :=
sorry

end NUMINAMATH_CALUDE_parabola_uniqueness_l3246_324691


namespace NUMINAMATH_CALUDE_q_definition_l3246_324693

/-- Given p: x ≤ 1, and ¬p is a sufficient but not necessary condition for q,
    prove that q can be defined as x > 0 -/
theorem q_definition (x : ℝ) :
  (∃ p : Prop, (p ↔ x ≤ 1) ∧ 
   (∃ q : Prop, (¬p → q) ∧ ¬(q → ¬p))) →
  ∃ q : Prop, q ↔ x > 0 :=
by sorry

end NUMINAMATH_CALUDE_q_definition_l3246_324693


namespace NUMINAMATH_CALUDE_first_year_after_2010_with_sum_of_digits_10_l3246_324600

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def isFirstYearAfter2010WithSumOfDigits10 (year : ℕ) : Prop :=
  year > 2010 ∧ 
  sumOfDigits year = 10 ∧
  ∀ y, 2010 < y ∧ y < year → sumOfDigits y ≠ 10

theorem first_year_after_2010_with_sum_of_digits_10 :
  isFirstYearAfter2010WithSumOfDigits10 2017 := by
  sorry

end NUMINAMATH_CALUDE_first_year_after_2010_with_sum_of_digits_10_l3246_324600


namespace NUMINAMATH_CALUDE_distinct_paths_theorem_l3246_324679

/-- The number of distinct paths in a rectangular grid from point C to point D -/
def distinct_paths (right_steps : ℕ) (up_steps : ℕ) : ℕ :=
  Nat.choose (right_steps + up_steps) up_steps

/-- Theorem: The number of distinct paths from C to D is equal to (10 choose 3) -/
theorem distinct_paths_theorem :
  distinct_paths 7 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_distinct_paths_theorem_l3246_324679


namespace NUMINAMATH_CALUDE_red_yellow_difference_l3246_324699

/-- Represents the number of marbles of each color in a bowl. -/
structure MarbleCount where
  total : ℕ
  yellow : ℕ
  blue : ℕ
  red : ℕ

/-- Represents the ratio of blue to red marbles. -/
structure BlueRedRatio where
  blue : ℕ
  red : ℕ

/-- Given the total number of marbles, the number of yellow marbles, and the ratio of blue to red marbles,
    proves that there are 3 more red marbles than yellow marbles. -/
theorem red_yellow_difference (m : MarbleCount) (ratio : BlueRedRatio) : 
  m.total = 19 → 
  m.yellow = 5 → 
  m.blue + m.red = m.total - m.yellow →
  m.blue * ratio.red = m.red * ratio.blue →
  ratio.blue = 3 →
  ratio.red = 4 →
  m.red - m.yellow = 3 := by
  sorry

end NUMINAMATH_CALUDE_red_yellow_difference_l3246_324699


namespace NUMINAMATH_CALUDE_largest_integer_with_conditions_l3246_324664

def digit_sum_of_squares (n : ℕ) : ℕ := sorry

def digits_increasing (n : ℕ) : Prop := sorry

def product_of_digits (n : ℕ) : ℕ := sorry

theorem largest_integer_with_conditions (n : ℕ) :
  (digit_sum_of_squares n = 82) →
  digits_increasing n →
  product_of_digits n ≤ 9 := by sorry

end NUMINAMATH_CALUDE_largest_integer_with_conditions_l3246_324664


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l3246_324628

theorem opposite_of_negative_fraction :
  -(-(1 / 2023)) = 1 / 2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l3246_324628


namespace NUMINAMATH_CALUDE_stock_price_calculation_l3246_324605

/-- Calculates the price of a stock given the investment amount, stock percentage, and annual income. -/
theorem stock_price_calculation (investment : ℝ) (stock_percentage : ℝ) (annual_income : ℝ) :
  investment = 6800 ∧ 
  stock_percentage = 0.6 ∧ 
  annual_income = 3000 →
  ∃ (stock_price : ℝ), stock_price = 136 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l3246_324605


namespace NUMINAMATH_CALUDE_picture_books_count_l3246_324602

theorem picture_books_count (total : ℕ) (fiction : ℕ) 
  (h1 : total = 35)
  (h2 : fiction = 5)
  (h3 : total = fiction + (fiction + 4) + (2 * fiction) + picture_books) :
  picture_books = 11 := by
  sorry

end NUMINAMATH_CALUDE_picture_books_count_l3246_324602


namespace NUMINAMATH_CALUDE_infinite_solutions_cube_fifth_square_l3246_324640

theorem infinite_solutions_cube_fifth_square (x y z : ℕ+) (k : ℕ+) 
  (h : x^3 + y^5 = z^2) :
  (k^10 * x)^3 + (k^6 * y)^5 = (k^15 * z)^2 := by
  sorry

#check infinite_solutions_cube_fifth_square

end NUMINAMATH_CALUDE_infinite_solutions_cube_fifth_square_l3246_324640


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l3246_324659

theorem opposite_of_negative_two : 
  ∃ x : ℤ, x + (-2) = 0 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l3246_324659


namespace NUMINAMATH_CALUDE_smallest_nonprime_with_conditions_l3246_324614

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def has_no_prime_factor_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < k → ¬(n % p = 0)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem smallest_nonprime_with_conditions (n : ℕ) : 
  n = 289 ↔ 
    (¬is_prime n ∧ 
     n > 25 ∧ 
     has_no_prime_factor_less_than n 15 ∧ 
     sum_of_digits n > 10 ∧
     ∀ m : ℕ, m < n → 
       (¬is_prime m → 
        m ≤ 25 ∨ 
        ¬has_no_prime_factor_less_than m 15 ∨ 
        sum_of_digits m ≤ 10)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_nonprime_with_conditions_l3246_324614


namespace NUMINAMATH_CALUDE_parallel_resistor_calculation_l3246_324683

/-- Calculates the resistance of the second resistor in a parallel circuit -/
theorem parallel_resistor_calculation (R1 R_total : ℝ) (h1 : R1 = 9) (h2 : R_total = 4.235294117647059) :
  ∃ R2 : ℝ, R2 = 8 ∧ 1 / R_total = 1 / R1 + 1 / R2 := by
sorry

end NUMINAMATH_CALUDE_parallel_resistor_calculation_l3246_324683


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_count_l3246_324658

/-- A quadrilateral with side lengths 9, 11, 15, and 14 has exactly 17 possible whole number lengths for a diagonal. -/
theorem quadrilateral_diagonal_count : ∃ (possible_lengths : Finset ℕ),
  (∀ d ∈ possible_lengths, 
    -- Triangle inequality for both triangles formed by the diagonal
    9 + d > 11 ∧ d + 11 > 9 ∧ 9 + 11 > d ∧
    14 + d > 15 ∧ d + 15 > 14 ∧ 14 + 15 > d) ∧
  (∀ d : ℕ, 
    (9 + d > 11 ∧ d + 11 > 9 ∧ 9 + 11 > d ∧
     14 + d > 15 ∧ d + 15 > 14 ∧ 14 + 15 > d) → d ∈ possible_lengths) ∧
  Finset.card possible_lengths = 17 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_count_l3246_324658


namespace NUMINAMATH_CALUDE_customers_served_today_l3246_324629

theorem customers_served_today (x : ℕ) 
  (h1 : (65 : ℝ) = (65 * x) / x) 
  (h2 : (90 : ℝ) = (65 * x + C) / (x + 1)) 
  (h3 : x = 1) : C = 115 := by
  sorry

end NUMINAMATH_CALUDE_customers_served_today_l3246_324629


namespace NUMINAMATH_CALUDE_cell_growth_after_12_days_l3246_324673

/-- The number of cells after a given number of periods, where each cell triples every period. -/
def cell_count (initial_cells : ℕ) (periods : ℕ) : ℕ :=
  initial_cells * 3^periods

/-- The problem statement -/
theorem cell_growth_after_12_days :
  let initial_cells := 5
  let days := 12
  let period := 3
  let periods := days / period
  cell_count initial_cells periods = 135 := by
  sorry

end NUMINAMATH_CALUDE_cell_growth_after_12_days_l3246_324673


namespace NUMINAMATH_CALUDE_ms_elizabeth_has_five_investments_l3246_324648

-- Define the variables
def mr_banks_investments : ℕ := 8
def mr_banks_revenue_per_investment : ℕ := 500
def ms_elizabeth_revenue_per_investment : ℕ := 900
def revenue_difference : ℕ := 500

-- Define Ms. Elizabeth's number of investments as a function
def ms_elizabeth_investments : ℕ :=
  let mr_banks_total_revenue := mr_banks_investments * mr_banks_revenue_per_investment
  let ms_elizabeth_total_revenue := mr_banks_total_revenue + revenue_difference
  ms_elizabeth_total_revenue / ms_elizabeth_revenue_per_investment

-- Theorem statement
theorem ms_elizabeth_has_five_investments :
  ms_elizabeth_investments = 5 := by
  sorry

end NUMINAMATH_CALUDE_ms_elizabeth_has_five_investments_l3246_324648


namespace NUMINAMATH_CALUDE_set_intersection_problem_l3246_324623

theorem set_intersection_problem :
  let A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
  let B : Set ℝ := {-1, 0, 2, 3}
  A ∩ B = {-1, 0, 2} := by sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l3246_324623


namespace NUMINAMATH_CALUDE_golden_ratio_exponential_monotonicity_l3246_324620

theorem golden_ratio_exponential_monotonicity 
  (a : ℝ) 
  (f : ℝ → ℝ) 
  (m n : ℝ) 
  (h1 : a = (Real.sqrt 5 - 1) / 2) 
  (h2 : ∀ x, f x = a ^ x) 
  (h3 : f m > f n) : 
  m < n := by
sorry

end NUMINAMATH_CALUDE_golden_ratio_exponential_monotonicity_l3246_324620


namespace NUMINAMATH_CALUDE_time_to_finish_game_l3246_324657

/-- Calculates the time to finish a game given initial and increased play times --/
theorem time_to_finish_game 
  (initial_hours_per_day : ℝ)
  (initial_days : ℝ)
  (completion_percentage : ℝ)
  (increased_hours_per_day : ℝ) :
  initial_hours_per_day = 4 →
  initial_days = 14 →
  completion_percentage = 0.4 →
  increased_hours_per_day = 7 →
  (initial_days * initial_hours_per_day * (1 / completion_percentage) - 
   initial_days * initial_hours_per_day) / increased_hours_per_day = 12 := by
sorry

end NUMINAMATH_CALUDE_time_to_finish_game_l3246_324657


namespace NUMINAMATH_CALUDE_arcade_tickets_l3246_324622

theorem arcade_tickets (initial_tickets spent_tickets additional_tickets : ℕ) :
  initial_tickets ≥ spent_tickets →
  initial_tickets - spent_tickets + additional_tickets =
    initial_tickets + additional_tickets - spent_tickets :=
by sorry

end NUMINAMATH_CALUDE_arcade_tickets_l3246_324622


namespace NUMINAMATH_CALUDE_no_real_solutions_l3246_324653

theorem no_real_solutions : 
  ¬∃ (x : ℝ), (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 8) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3246_324653


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l3246_324631

theorem quadratic_root_sum (m n : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (1 - Complex.I * Real.sqrt 3)^2 + m * (1 - Complex.I * Real.sqrt 3) + n = 0 →
  m + n = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l3246_324631


namespace NUMINAMATH_CALUDE_expression_evaluation_l3246_324668

theorem expression_evaluation (a b c : ℝ) 
  (h1 : c = b - 11)
  (h2 : b = a + 3)
  (h3 : a = 5)
  (h4 : a + 2 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 9) / (c + 7) = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3246_324668


namespace NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l3246_324617

/-- Given a parabola y² = 2px (p > 0) with focus F, if its directrix intersects 
    the hyperbola y²/3 - x² = 1 at points M and N, and MF is perpendicular to NF, 
    then p = 2√3. -/
theorem parabola_hyperbola_intersection (p : ℝ) (F M N : ℝ × ℝ) : 
  p > 0 →  -- p is positive
  (∀ x y, y^2 = 2*p*x) →  -- equation of parabola
  (∀ x y, y^2/3 - x^2 = 1) →  -- equation of hyperbola
  (M.1 = -p/2 ∧ N.1 = -p/2) →  -- M and N are on the directrix
  (M.2^2/3 - M.1^2 = 1 ∧ N.2^2/3 - N.1^2 = 1) →  -- M and N are on the hyperbola
  ((M.1 - F.1) * (N.1 - F.1) + (M.2 - F.2) * (N.2 - F.2) = 0) →  -- MF ⊥ NF
  p = 2 * Real.sqrt 3 := by
    sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l3246_324617


namespace NUMINAMATH_CALUDE_wade_average_points_l3246_324604

/-- Represents a basketball team with Wade and his teammates -/
structure BasketballTeam where
  wade_avg : ℝ
  teammates_avg : ℝ
  total_points : ℝ
  num_games : ℝ

/-- Theorem stating Wade's average points per game -/
theorem wade_average_points (team : BasketballTeam)
  (h1 : team.teammates_avg = 40)
  (h2 : team.total_points = 300)
  (h3 : team.num_games = 5) :
  team.wade_avg = 20 := by
  sorry

#check wade_average_points

end NUMINAMATH_CALUDE_wade_average_points_l3246_324604


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_57_l3246_324624

theorem smallest_four_digit_divisible_by_57 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 57 = 0 → n ≥ 1026 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_57_l3246_324624


namespace NUMINAMATH_CALUDE_remaining_bottles_calculation_l3246_324615

theorem remaining_bottles_calculation (small_initial big_initial medium_initial : ℕ)
  (small_sold_percent small_damaged_percent : ℚ)
  (big_sold_percent big_damaged_percent : ℚ)
  (medium_sold_percent medium_damaged_percent : ℚ)
  (h_small_initial : small_initial = 6000)
  (h_big_initial : big_initial = 15000)
  (h_medium_initial : medium_initial = 5000)
  (h_small_sold : small_sold_percent = 11/100)
  (h_small_damaged : small_damaged_percent = 3/100)
  (h_big_sold : big_sold_percent = 12/100)
  (h_big_damaged : big_damaged_percent = 2/100)
  (h_medium_sold : medium_sold_percent = 8/100)
  (h_medium_damaged : medium_damaged_percent = 4/100) :
  (small_initial - (small_initial * small_sold_percent).floor - (small_initial * small_damaged_percent).floor) +
  (big_initial - (big_initial * big_sold_percent).floor - (big_initial * big_damaged_percent).floor) +
  (medium_initial - (medium_initial * medium_sold_percent).floor - (medium_initial * medium_damaged_percent).floor) = 22560 := by
sorry

end NUMINAMATH_CALUDE_remaining_bottles_calculation_l3246_324615


namespace NUMINAMATH_CALUDE_number_of_benches_l3246_324690

/-- Converts a base 6 number to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

/-- The number of people that can be seated in the shop -/
def totalSeats : ℕ := base6ToBase10 204

/-- The number of people that sit on one bench -/
def peoplePerBench : ℕ := 2

/-- Theorem: The number of benches in the shop is 38 -/
theorem number_of_benches :
  totalSeats / peoplePerBench = 38 := by
  sorry

end NUMINAMATH_CALUDE_number_of_benches_l3246_324690


namespace NUMINAMATH_CALUDE_train_length_specific_train_length_l3246_324635

/-- The length of a train given its speed and time to cross a fixed point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ := by
  sorry

/-- Proof that a train with speed 144 km/hr crossing a point in 9.99920006399488 seconds has length approximately 399.97 meters -/
theorem specific_train_length : 
  ∃ (length : ℝ), abs (length - train_length 144 9.99920006399488) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_specific_train_length_l3246_324635


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l3246_324675

theorem degree_to_radian_conversion :
  ∃ (k : ℤ) (α : ℝ), 
    -885 * (π / 180) = 2 * k * π + α ∧
    0 ≤ α ∧ α ≤ 2 * π ∧
    2 * k * π + α = -6 * π + 13 * π / 12 :=
by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l3246_324675


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3246_324619

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (∀ x, x^2 = 9*x - 20 → x + (9 - x) = 9) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3246_324619


namespace NUMINAMATH_CALUDE_park_route_length_l3246_324674

/-- A bike route in a park -/
structure BikeRoute where
  horizontal_segments : List Float
  vertical_segments : List Float

/-- The total length of a bike route -/
def total_length (route : BikeRoute) : Float :=
  2 * (route.horizontal_segments.sum + route.vertical_segments.sum)

/-- The specific bike route described in the problem -/
def park_route : BikeRoute :=
  { horizontal_segments := [4, 7, 2],
    vertical_segments := [6, 7] }

theorem park_route_length :
  total_length park_route = 52 := by
  sorry

#eval total_length park_route

end NUMINAMATH_CALUDE_park_route_length_l3246_324674


namespace NUMINAMATH_CALUDE_perpendicular_distance_to_plane_l3246_324666

/-- The perpendicular distance from a point to a plane --/
def perpendicularDistance (p : ℝ × ℝ × ℝ) (plane : Set (ℝ × ℝ × ℝ)) : ℝ :=
  sorry

/-- The plane containing three points --/
def planeThroughPoints (a b c : ℝ × ℝ × ℝ) : Set (ℝ × ℝ × ℝ) :=
  sorry

theorem perpendicular_distance_to_plane :
  let a : ℝ × ℝ × ℝ := (0, 0, 0)
  let b : ℝ × ℝ × ℝ := (5, 0, 0)
  let c : ℝ × ℝ × ℝ := (0, 3, 0)
  let d : ℝ × ℝ × ℝ := (0, 0, 6)
  let plane := planeThroughPoints a b c
  perpendicularDistance d plane = 6 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_distance_to_plane_l3246_324666


namespace NUMINAMATH_CALUDE_exponential_equation_sum_of_reciprocals_l3246_324641

theorem exponential_equation_sum_of_reciprocals (x y : ℝ) 
  (h1 : 3^x = Real.sqrt 12) 
  (h2 : 4^y = Real.sqrt 12) : 
  1/x + 1/y = 2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_sum_of_reciprocals_l3246_324641


namespace NUMINAMATH_CALUDE_classroom_children_l3246_324684

theorem classroom_children (total_pencils : ℕ) (pencils_per_student : ℕ) (h1 : total_pencils = 8) (h2 : pencils_per_student = 2) :
  total_pencils / pencils_per_student = 4 :=
by sorry

end NUMINAMATH_CALUDE_classroom_children_l3246_324684


namespace NUMINAMATH_CALUDE_product_greater_than_sum_implies_sum_greater_than_four_l3246_324634

theorem product_greater_than_sum_implies_sum_greater_than_four (x y : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_product : x * y > x + y) : x + y > 4 := by
  sorry

end NUMINAMATH_CALUDE_product_greater_than_sum_implies_sum_greater_than_four_l3246_324634


namespace NUMINAMATH_CALUDE_max_profit_at_8_l3246_324612

noncomputable def C (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 8 then (1/2) * x^2 + 4*x
  else if x ≥ 8 then 11*x + 49/x - 35
  else 0

noncomputable def P (x : ℝ) : ℝ :=
  10*x - C x - 5

theorem max_profit_at_8 :
  ∀ x > 0, P x ≤ P 8 ∧ P 8 = 127/8 :=
by sorry

end NUMINAMATH_CALUDE_max_profit_at_8_l3246_324612


namespace NUMINAMATH_CALUDE_initial_tax_rate_is_42_percent_l3246_324610

-- Define the initial tax rate as a real number between 0 and 1
variable (initial_rate : ℝ) (h_rate : 0 ≤ initial_rate ∧ initial_rate ≤ 1)

-- Define the new tax rate (32%)
def new_rate : ℝ := 0.32

-- Define the annual income
def annual_income : ℝ := 42400

-- Define the differential savings
def differential_savings : ℝ := 4240

-- Theorem statement
theorem initial_tax_rate_is_42_percent :
  (initial_rate * annual_income - new_rate * annual_income = differential_savings) →
  initial_rate = 0.42 := by
  sorry


end NUMINAMATH_CALUDE_initial_tax_rate_is_42_percent_l3246_324610


namespace NUMINAMATH_CALUDE_average_weight_increase_l3246_324663

/-- Proves that replacing a person in a group of 5 increases the average weight by 1.5 kg -/
theorem average_weight_increase (group_size : ℕ) (old_weight new_weight : ℝ) :
  group_size = 5 →
  old_weight = 65 →
  new_weight = 72.5 →
  (new_weight - old_weight) / group_size = 1.5 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l3246_324663


namespace NUMINAMATH_CALUDE_josh_wallet_amount_l3246_324642

def calculate_final_wallet_amount (initial_wallet : ℝ) (investment : ℝ) (debt : ℝ)
  (stock_a_percent : ℝ) (stock_b_percent : ℝ) (stock_c_percent : ℝ)
  (stock_a_change : ℝ) (stock_b_change : ℝ) (stock_c_change : ℝ) : ℝ :=
  let stock_a_value := investment * stock_a_percent * (1 + stock_a_change)
  let stock_b_value := investment * stock_b_percent * (1 + stock_b_change)
  let stock_c_value := investment * stock_c_percent * (1 + stock_c_change)
  let total_stock_value := stock_a_value + stock_b_value + stock_c_value
  let remaining_after_debt := total_stock_value - debt
  initial_wallet + remaining_after_debt

theorem josh_wallet_amount :
  calculate_final_wallet_amount 300 2000 500 0.4 0.3 0.3 0.2 0.3 (-0.1) = 2080 := by
  sorry

end NUMINAMATH_CALUDE_josh_wallet_amount_l3246_324642


namespace NUMINAMATH_CALUDE_age_ratio_problem_l3246_324616

theorem age_ratio_problem (albert mary betty : ℕ) 
  (h1 : ∃ k : ℕ, albert = k * mary)
  (h2 : albert = 4 * betty)
  (h3 : mary = albert - 22)
  (h4 : betty = 11) :
  albert / mary = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l3246_324616


namespace NUMINAMATH_CALUDE_min_a_is_minimum_l3246_324644

/-- The inequality that holds for all x ≥ 0 -/
def inequality (a : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 0 → x * Real.exp x + a * Real.exp x * Real.log (x + 1) + 1 ≥ Real.exp x * (x + 1) ^ a

/-- The minimum value of a that satisfies the inequality -/
def min_a : ℝ := -1

/-- Theorem stating that min_a is the minimum value satisfying the inequality -/
theorem min_a_is_minimum :
  (∀ a : ℝ, inequality a → a ≥ min_a) ∧ inequality min_a := by sorry

end NUMINAMATH_CALUDE_min_a_is_minimum_l3246_324644


namespace NUMINAMATH_CALUDE_farm_animals_feet_count_l3246_324676

theorem farm_animals_feet_count (total_heads : Nat) (hen_count : Nat) : 
  total_heads = 60 → hen_count = 20 → (total_heads - hen_count) * 4 + hen_count * 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_feet_count_l3246_324676


namespace NUMINAMATH_CALUDE_existence_of_square_root_of_minus_one_l3246_324625

theorem existence_of_square_root_of_minus_one (p : ℕ) (hp : Nat.Prime p) :
  (∃ a : ℤ, a^2 ≡ -1 [ZMOD p]) ↔ p ≡ 1 [MOD 4] := by sorry

end NUMINAMATH_CALUDE_existence_of_square_root_of_minus_one_l3246_324625


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l3246_324632

/-- Systematic sampling function -/
def systematic_sample (total : ℕ) (sample_size : ℕ) (start : ℕ) : ℕ → ℕ :=
  fun n => start + (n - 1) * (total / sample_size)

theorem systematic_sampling_theorem :
  let total := 200
  let sample_size := 40
  let group_size := total / sample_size
  let fifth_group_sample := 22
  systematic_sample total sample_size fifth_group_sample 8 = 37 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l3246_324632


namespace NUMINAMATH_CALUDE_smallest_angle_in_pentadecagon_l3246_324618

/-- Represents the number of sides in the polygon -/
def n : ℕ := 15

/-- The sum of internal angles in an n-sided polygon -/
def angleSum : ℝ := (n - 2) * 180

/-- The largest angle in the sequence -/
def largestAngle : ℝ := 176

/-- Theorem: In a convex 15-sided polygon where the angles form an arithmetic sequence 
    and the largest angle is 176°, the smallest angle is 136°. -/
theorem smallest_angle_in_pentadecagon : 
  ∃ (a d : ℝ), 
    (∀ i : ℕ, i < n → a + i * d ≤ a + (n - 1) * d) ∧  -- Convexity condition
    (a + (n - 1) * d = largestAngle) ∧                -- Largest angle condition
    (n * (2 * a + (n - 1) * d) / 2 = angleSum) ∧      -- Sum of angles condition
    (a = 136)                                         -- Conclusion
    := by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_pentadecagon_l3246_324618


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3246_324685

theorem arithmetic_sequence_length :
  ∀ (a₁ aₙ d n : ℤ),
    a₁ = -38 →
    aₙ = 69 →
    d = 6 →
    aₙ = a₁ + (n - 1) * d →
    n = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3246_324685


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_slope_condition_l3246_324697

/-- The slope of a line intersecting an ellipse satisfies a certain condition -/
theorem line_ellipse_intersection_slope_condition 
  (m : ℝ) -- slope of the line
  (h : ∃ (x y : ℝ), y = m * x + 10 ∧ 4 * x^2 + 25 * y^2 = 100) -- line intersects ellipse
  : m^2 ≥ 1/624 := by
  sorry

#check line_ellipse_intersection_slope_condition

end NUMINAMATH_CALUDE_line_ellipse_intersection_slope_condition_l3246_324697


namespace NUMINAMATH_CALUDE_inscribed_sphere_polyhedron_volume_l3246_324643

/-- A polyhedron with an inscribed sphere -/
structure InscribedSpherePolyhedron where
  /-- The volume of the polyhedron -/
  volume : ℝ
  /-- The total surface area of the polyhedron -/
  surface_area : ℝ
  /-- The radius of the inscribed sphere -/
  inscribed_sphere_radius : ℝ
  /-- The radius is positive -/
  radius_pos : 0 < inscribed_sphere_radius

/-- 
Theorem: For a polyhedron with an inscribed sphere, 
the volume of the polyhedron is equal to one-third of the product 
of its total surface area and the radius of the inscribed sphere.
-/
theorem inscribed_sphere_polyhedron_volume 
  (p : InscribedSpherePolyhedron) : 
  p.volume = (1 / 3) * p.surface_area * p.inscribed_sphere_radius := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_polyhedron_volume_l3246_324643


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l3246_324621

theorem largest_integer_satisfying_inequality :
  ∃ (n : ℕ), n = 8 ∧ 
  (∀ (m : ℕ), 3 * (m^2007 : ℝ) < 3^4015 → m ≤ n) ∧
  (3 * ((n : ℝ)^2007) < 3^4015) :=
sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l3246_324621


namespace NUMINAMATH_CALUDE_jaysons_mom_age_at_birth_l3246_324627

theorem jaysons_mom_age_at_birth (jayson_age : ℕ) (dad_age : ℕ) (mom_age : ℕ) 
  (h1 : jayson_age = 10)
  (h2 : dad_age = 4 * jayson_age)
  (h3 : mom_age = dad_age - 2) :
  mom_age - jayson_age = 28 := by
  sorry

end NUMINAMATH_CALUDE_jaysons_mom_age_at_birth_l3246_324627


namespace NUMINAMATH_CALUDE_gcd_count_equals_fourteen_l3246_324670

theorem gcd_count_equals_fourteen : 
  (Finset.filter (fun n : ℕ => Nat.gcd 21 n = 7) (Finset.range 150)).card = 14 := by
  sorry

end NUMINAMATH_CALUDE_gcd_count_equals_fourteen_l3246_324670


namespace NUMINAMATH_CALUDE_marbles_in_container_l3246_324656

/-- Given that a container with volume 24 cm³ holds 75 marbles, 
    prove that a container with volume 72 cm³ holds 225 marbles, 
    assuming the number of marbles is proportional to the volume. -/
theorem marbles_in_container (v₁ v₂ : ℝ) (m₁ m₂ : ℕ) 
  (h₁ : v₁ = 24) (h₂ : v₂ = 72) (h₃ : m₁ = 75) 
  (h₄ : v₁ * m₂ = v₂ * m₁) : m₂ = 225 := by
  sorry

end NUMINAMATH_CALUDE_marbles_in_container_l3246_324656


namespace NUMINAMATH_CALUDE_abcd_inequality_l3246_324601

theorem abcd_inequality (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_eq : a^2/(1+a^2) + b^2/(1+b^2) + c^2/(1+c^2) + d^2/(1+d^2) = 1) : 
  a * b * c * d ≤ 1/9 := by
sorry

end NUMINAMATH_CALUDE_abcd_inequality_l3246_324601


namespace NUMINAMATH_CALUDE_division_problem_l3246_324645

theorem division_problem (x y : ℕ+) : 
  (x : ℝ) / y = 96.15 → 
  ∃ q : ℕ, x = q * y + 9 →
  y = 60 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3246_324645


namespace NUMINAMATH_CALUDE_power_of_two_equality_l3246_324694

theorem power_of_two_equality (a b : ℕ+) (h : 2^(a.val) * 2^(b.val) = 8) : 
  (2^(a.val))^(b.val) = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l3246_324694


namespace NUMINAMATH_CALUDE_smallest_in_row_10_n_squared_minus_n_and_2n_in_row_largest_n_not_including_n_squared_minus_10n_l3246_324638

/-- Predicate defining whether an integer m is in Row n -/
def in_row (n : ℕ) (m : ℕ) : Prop :=
  m % n = 0 ∧ m ≤ n^2 ∧ ∀ k < n, ¬in_row k m

theorem smallest_in_row_10 :
  ∀ m, in_row 10 m → m ≥ 10 :=
sorry

theorem n_squared_minus_n_and_2n_in_row (n : ℕ) (h : n ≥ 3) :
  in_row n (n^2 - n) ∧ in_row n (n^2 - 2*n) :=
sorry

theorem largest_n_not_including_n_squared_minus_10n :
  ∀ n > 9, in_row n (n^2 - 10*n) :=
sorry

end NUMINAMATH_CALUDE_smallest_in_row_10_n_squared_minus_n_and_2n_in_row_largest_n_not_including_n_squared_minus_10n_l3246_324638


namespace NUMINAMATH_CALUDE_larger_number_problem_l3246_324678

theorem larger_number_problem (S L : ℕ) 
  (h1 : L - S = 50000)
  (h2 : L = 13 * S + 317) :
  L = 54140 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3246_324678


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_c_value_l3246_324652

theorem polynomial_factor_implies_c_value : ∀ (c q : ℝ),
  (∃ (a : ℝ), (X^3 + q*X + 1) * (3*X + a) = 3*X^4 + c*X^2 + 8*X + 9) →
  c = 24 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_c_value_l3246_324652


namespace NUMINAMATH_CALUDE_solution_difference_l3246_324650

theorem solution_difference : ∃ (a b : ℝ), 
  (∀ x : ℝ, (3*x - 9) / (x^2 + x - 6) = x + 1 ↔ (x = a ∨ x = b)) ∧ 
  a > b ∧ 
  a - b = 4 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l3246_324650


namespace NUMINAMATH_CALUDE_slope_relation_l3246_324686

theorem slope_relation (k : ℝ) : 
  (∃ α : ℝ, α = (2 : ℝ) ∧ (2 : ℝ) * α = k) → k = -(4 : ℝ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_slope_relation_l3246_324686


namespace NUMINAMATH_CALUDE_top_three_average_score_l3246_324636

theorem top_three_average_score (total_students : ℕ) (top_students : ℕ) 
  (class_average : ℝ) (score_difference : ℝ) : 
  total_students = 12 →
  top_students = 3 →
  class_average = 85 →
  score_difference = 8 →
  let other_students := total_students - top_students
  let top_average := (total_students * class_average - other_students * (class_average - score_difference)) / top_students
  top_average = 91 := by sorry

end NUMINAMATH_CALUDE_top_three_average_score_l3246_324636


namespace NUMINAMATH_CALUDE_rotten_bananas_percentage_l3246_324655

theorem rotten_bananas_percentage
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (rotten_oranges_percentage : ℚ)
  (good_fruits_percentage : ℚ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : rotten_oranges_percentage = 15 / 100)
  (h4 : good_fruits_percentage = 894 / 1000) :
  (total_oranges * (1 - rotten_oranges_percentage) + total_bananas * (1 - (4 / 100 : ℚ))) / (total_oranges + total_bananas) = good_fruits_percentage :=
by sorry

end NUMINAMATH_CALUDE_rotten_bananas_percentage_l3246_324655


namespace NUMINAMATH_CALUDE_B_parity_2021_2022_2023_l3246_324654

def B : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 1
  | n + 3 => B (n + 2) + B (n + 1)

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem B_parity_2021_2022_2023 :
  is_odd (B 2021) ∧ is_odd (B 2022) ∧ ¬is_odd (B 2023) := by sorry

end NUMINAMATH_CALUDE_B_parity_2021_2022_2023_l3246_324654


namespace NUMINAMATH_CALUDE_iris_rose_ratio_l3246_324647

theorem iris_rose_ratio (initial_roses : ℕ) (added_roses : ℕ) : 
  initial_roses = 42 →
  added_roses = 35 →
  (3 : ℚ) / 7 = (irises_needed : ℚ) / (initial_roses + added_roses) →
  irises_needed = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_iris_rose_ratio_l3246_324647


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l3246_324682

theorem rectangular_to_polar_conversion :
  let x : ℝ := Real.sqrt 3
  let y : ℝ := -Real.sqrt 3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := if x > 0 ∧ y < 0
                then 2 * Real.pi - Real.arctan ((-y) / x)
                else 0  -- This else case is just a placeholder
  (r = Real.sqrt 6 ∧ θ = 7 * Real.pi / 4 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l3246_324682


namespace NUMINAMATH_CALUDE_quadratic_roots_l3246_324646

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 5*x + m

theorem quadratic_roots (m : ℝ) (h : f m 1 = 0) : 
  ∃ (r₁ r₂ : ℝ), r₁ = 1 ∧ r₂ = 4 ∧ ∀ x, f m x = 0 ↔ x = r₁ ∨ x = r₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l3246_324646


namespace NUMINAMATH_CALUDE_alcohol_solution_proof_l3246_324687

/-- Proves that adding 2.4 liters of pure alcohol to a 6-liter solution that is 30% alcohol 
    results in a 50% alcohol solution -/
theorem alcohol_solution_proof :
  let initial_volume : ℝ := 6
  let initial_percentage : ℝ := 0.30
  let target_percentage : ℝ := 0.50
  let added_alcohol : ℝ := 2.4
  let final_volume : ℝ := initial_volume + added_alcohol
  let final_alcohol_volume : ℝ := initial_volume * initial_percentage + added_alcohol
  final_alcohol_volume / final_volume = target_percentage :=
by sorry

end NUMINAMATH_CALUDE_alcohol_solution_proof_l3246_324687


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l3246_324677

theorem consecutive_integers_product (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
  a * b * c * d * e = 2520 →
  e = 7 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l3246_324677


namespace NUMINAMATH_CALUDE_rocks_needed_l3246_324651

/-- The number of rocks Mrs. Hilt needs for her garden border -/
def total_rocks_needed : ℕ := 125

/-- The number of rocks Mrs. Hilt currently has -/
def rocks_on_hand : ℕ := 64

/-- Theorem: Mrs. Hilt needs 61 more rocks to complete her garden border -/
theorem rocks_needed : total_rocks_needed - rocks_on_hand = 61 := by
  sorry

end NUMINAMATH_CALUDE_rocks_needed_l3246_324651


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l3246_324611

theorem right_triangle_leg_length 
  (a b c : ℝ) 
  (right_triangle : a^2 + b^2 = c^2) 
  (leg_a : a = 4) 
  (hypotenuse : c = 5) : 
  b = 3 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l3246_324611


namespace NUMINAMATH_CALUDE_point_on_curve_l3246_324613

theorem point_on_curve : (3^2 : ℝ) - 3 * 10 + 2 * 10 + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_point_on_curve_l3246_324613


namespace NUMINAMATH_CALUDE_parabola_equation_l3246_324696

/-- A parabola with vertex at the origin, directrix perpendicular to the x-axis, 
    and passing through the point (1, -√2) has the equation y² = 2x -/
theorem parabola_equation : ∃ (f : ℝ → ℝ),
  (∀ x y : ℝ, f x = y ↔ y^2 = 2*x) ∧ 
  (f 0 = 0) ∧ 
  (∃ a : ℝ, ∀ x : ℝ, (x < a ↔ f x < 0) ∧ (x > a ↔ f x > 0)) ∧
  (f 1 = -Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l3246_324696


namespace NUMINAMATH_CALUDE_expand_expression_l3246_324637

theorem expand_expression (x y : ℝ) : 24 * (3 * x - 4 * y + 6) = 72 * x - 96 * y + 144 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3246_324637


namespace NUMINAMATH_CALUDE_cube_monotone_l3246_324681

theorem cube_monotone (a b : ℝ) : a > b ↔ a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_monotone_l3246_324681


namespace NUMINAMATH_CALUDE_cube_volume_problem_l3246_324661

theorem cube_volume_problem (x : ℝ) (h : x > 0) (eq : x^3 + 6*x^2 = 16*x) :
  27 * x^3 = 216 := by sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l3246_324661


namespace NUMINAMATH_CALUDE_age_problem_l3246_324698

theorem age_problem (a b c d : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  d = c / 2 →
  a + b + c + d = 39 →
  b = 14 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l3246_324698


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l3246_324671

theorem quadratic_roots_sum (x₁ x₂ : ℝ) : 
  x₁^2 + x₁ - 2023 = 0 → x₂^2 + x₂ - 2023 = 0 → x₁^2 + 2*x₁ + x₂ = 2022 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l3246_324671


namespace NUMINAMATH_CALUDE_f_derivative_l3246_324606

noncomputable def f (x : ℝ) : ℝ := (1 - x) / ((1 + x^2) * Real.cos x)

theorem f_derivative :
  deriv f = λ x => ((x^2 - 2*x - 1) * Real.cos x + (1 - x) * (1 + x^2) * Real.sin x) / ((1 + x^2)^2 * (Real.cos x)^2) :=
by sorry

end NUMINAMATH_CALUDE_f_derivative_l3246_324606


namespace NUMINAMATH_CALUDE_percentage_calculation_l3246_324609

theorem percentage_calculation (p : ℝ) : 
  (p / 100) * 170 = 0.20 * 552.50 → p = 65 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3246_324609


namespace NUMINAMATH_CALUDE_linear_function_properties_l3246_324689

def f (x : ℝ) : ℝ := -2 * x - 4

theorem linear_function_properties :
  (f (-1) = -2) ∧
  (f 0 ≠ -2) ∧
  (∀ x, x < -2 → f x > 0) ∧
  (∀ x y, x ≥ 0 → y ≥ 0 → f x ≠ y) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l3246_324689


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3246_324667

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (l : ℕ) (d : ℕ) : ℕ :=
  let n : ℕ := (l - a) / d + 1
  n * (a + l) / 2

theorem arithmetic_sequence_sum : 
  2 * arithmetic_sum 51 99 2 = 3750 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3246_324667


namespace NUMINAMATH_CALUDE_specific_polygon_triangulation_l3246_324633

/-- Represents a convex polygon with additional internal points -/
structure EnhancedPolygon where
  sides : ℕ
  internal_points : ℕ
  no_collinear_triples : Prop

/-- Represents the triangulation of an EnhancedPolygon -/
def triangulation (p : EnhancedPolygon) : ℕ := sorry

/-- The theorem stating the number of triangles in the specific polygon -/
theorem specific_polygon_triangulation :
  ∀ (p : EnhancedPolygon),
    p.sides = 1000 →
    p.internal_points = 500 →
    p.no_collinear_triples →
    triangulation p = 1998 := by sorry

end NUMINAMATH_CALUDE_specific_polygon_triangulation_l3246_324633


namespace NUMINAMATH_CALUDE_travel_time_for_a_l3246_324630

/-- Represents the travel time of a person given their relative speed and time difference from a reference traveler -/
def travelTime (relativeSpeed : ℚ) (timeDiff : ℚ) : ℚ :=
  (4 : ℚ) / 3 * ((3 : ℚ) / 2 + timeDiff)

theorem travel_time_for_a (speedRatio : ℚ) (timeDiffHours : ℚ) 
    (h1 : speedRatio = 3 / 4) 
    (h2 : timeDiffHours = 1 / 2) : 
  travelTime speedRatio timeDiffHours = 2 := by
  sorry

#eval travelTime (3/4) (1/2)

end NUMINAMATH_CALUDE_travel_time_for_a_l3246_324630


namespace NUMINAMATH_CALUDE_cat_roaming_area_l3246_324665

/-- The area accessible to a cat tethered to a circular water tank -/
theorem cat_roaming_area (tank_radius rope_length : ℝ) (h1 : tank_radius = 20) (h2 : rope_length = 10) :
  π * (tank_radius + rope_length)^2 - π * tank_radius^2 = 500 * π :=
by sorry

end NUMINAMATH_CALUDE_cat_roaming_area_l3246_324665
