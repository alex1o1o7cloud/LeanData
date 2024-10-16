import Mathlib

namespace NUMINAMATH_CALUDE_age_problem_l2505_250596

theorem age_problem (A B C : ℕ) : 
  (A + B + C) / 3 = 26 →
  (A + C) / 2 = 29 →
  B = 20 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l2505_250596


namespace NUMINAMATH_CALUDE_inequality_proof_l2505_250506

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x / Real.sqrt y + y / Real.sqrt x ≥ Real.sqrt x + Real.sqrt y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2505_250506


namespace NUMINAMATH_CALUDE_range_of_m_l2505_250530

def has_two_distinct_negative_roots (m : ℝ) : Prop :=
  ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def has_no_real_roots (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

def satisfies_conditions (m : ℝ) : Prop :=
  (has_two_distinct_negative_roots m ∨ has_no_real_roots m) ∧
  ¬(has_two_distinct_negative_roots m ∧ has_no_real_roots m)

theorem range_of_m : 
  {m : ℝ | satisfies_conditions m} = {m : ℝ | 1 < m ∧ m ≤ 2 ∨ 3 ≤ m} :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2505_250530


namespace NUMINAMATH_CALUDE_wall_width_proof_l2505_250556

theorem wall_width_proof (width height length volume : ℝ) : 
  height = 6 * width →
  length = 7 * height →
  volume = length * width * height →
  volume = 16128 →
  width = (384 : ℝ) ^ (1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_wall_width_proof_l2505_250556


namespace NUMINAMATH_CALUDE_fraction_simplification_l2505_250594

theorem fraction_simplification : (3 : ℚ) / (2 - 3 / 4) = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2505_250594


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2505_250585

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∃ d, ∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  a 1 + a 2 = 1 →
  a 3 + a 4 = 4 →
  a 4 + a 5 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2505_250585


namespace NUMINAMATH_CALUDE_right_triangle_existence_unique_non_right_triangle_l2505_250544

theorem right_triangle_existence (a b c : ℝ) : Bool :=
  a * a + b * b = c * c

theorem unique_non_right_triangle : 
  right_triangle_existence 3 4 5 = true ∧
  right_triangle_existence 1 1 (Real.sqrt 2) = true ∧
  right_triangle_existence 8 15 18 = false ∧
  right_triangle_existence 5 12 13 = true ∧
  right_triangle_existence 6 8 10 = true :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_existence_unique_non_right_triangle_l2505_250544


namespace NUMINAMATH_CALUDE_cost_of_one_ring_l2505_250529

/-- The cost of a single ring given the total cost and number of rings. -/
def ring_cost (total_cost : ℕ) (num_rings : ℕ) : ℕ :=
  total_cost / num_rings

/-- Theorem stating that the cost of one ring is $24 given the problem conditions. -/
theorem cost_of_one_ring :
  let total_cost : ℕ := 48
  let num_rings : ℕ := 2
  ring_cost total_cost num_rings = 24 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_one_ring_l2505_250529


namespace NUMINAMATH_CALUDE_power_equation_solution_l2505_250553

theorem power_equation_solution : ∃ x : ℕ, 2^4 + 3 = 5^2 - x ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2505_250553


namespace NUMINAMATH_CALUDE_trip_savings_l2505_250550

/-- The amount Trip can save by going to the earlier movie. -/
def total_savings (evening_ticket_cost : ℚ) (food_combo_cost : ℚ) 
  (ticket_discount_percent : ℚ) (food_discount_percent : ℚ) : ℚ :=
  (ticket_discount_percent / 100) * evening_ticket_cost + 
  (food_discount_percent / 100) * food_combo_cost

/-- Proof that Trip can save $7 by going to the earlier movie. -/
theorem trip_savings : 
  total_savings 10 10 20 50 = 7 := by
  sorry

end NUMINAMATH_CALUDE_trip_savings_l2505_250550


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2505_250508

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + a > 0) ↔ (0 < a ∧ a < 4) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2505_250508


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l2505_250562

theorem simplify_and_evaluate_expression (m : ℚ) (h : m = 5) :
  (m + 2 - 5 / (m - 2)) / ((3 * m - m^2) / (m - 2)) = -8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l2505_250562


namespace NUMINAMATH_CALUDE_pigeonhole_principle_for_library_l2505_250574

/-- The number of different types of books available. -/
def num_book_types : ℕ := 4

/-- The maximum number of books a student can borrow. -/
def max_books_per_student : ℕ := 3

/-- The type representing a borrowing pattern (number and types of books borrowed). -/
def BorrowingPattern := Fin num_book_types → Fin (max_books_per_student + 1)

/-- The minimum number of students required to guarantee a repeated borrowing pattern. -/
def min_students_for_repeat : ℕ := 15

theorem pigeonhole_principle_for_library :
  ∀ (students : Fin min_students_for_repeat → BorrowingPattern),
  ∃ (i j : Fin min_students_for_repeat), i ≠ j ∧ students i = students j :=
sorry

end NUMINAMATH_CALUDE_pigeonhole_principle_for_library_l2505_250574


namespace NUMINAMATH_CALUDE_last_four_average_l2505_250521

theorem last_four_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 65 →
  (list.take 3).sum / 3 = 60 →
  (list.drop 3).sum / 4 = 68.75 := by
  sorry

end NUMINAMATH_CALUDE_last_four_average_l2505_250521


namespace NUMINAMATH_CALUDE_inconsistent_means_l2505_250509

theorem inconsistent_means : ¬ ∃ x : ℝ,
  (x + 42 + 78 + 104) / 4 = 62 ∧
  (48 + 62 + 98 + 124 + x) / 5 = 78 := by
  sorry

end NUMINAMATH_CALUDE_inconsistent_means_l2505_250509


namespace NUMINAMATH_CALUDE_cereal_difference_theorem_l2505_250512

/-- Represents the probability of eating unsweetened cereal -/
def p_unsweetened : ℚ := 3/5

/-- Represents the probability of eating sweetened cereal -/
def p_sweetened : ℚ := 2/5

/-- Number of days in a non-leap year -/
def days_in_year : ℕ := 365

/-- Expected difference between days of eating unsweetened and sweetened cereal -/
def expected_difference : ℚ := days_in_year * (p_unsweetened - p_sweetened)

theorem cereal_difference_theorem : 
  expected_difference = 73 := by sorry

end NUMINAMATH_CALUDE_cereal_difference_theorem_l2505_250512


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l2505_250572

theorem a_equals_one_sufficient_not_necessary :
  (∃ a : ℝ, a = 1 → a^2 = 1) ∧ 
  (∃ a : ℝ, a^2 = 1 ∧ a ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l2505_250572


namespace NUMINAMATH_CALUDE_initial_limes_count_l2505_250560

def limes_given_to_sara : ℕ := 4
def limes_dan_has_now : ℕ := 5

theorem initial_limes_count : 
  limes_given_to_sara + limes_dan_has_now = 9 := by sorry

end NUMINAMATH_CALUDE_initial_limes_count_l2505_250560


namespace NUMINAMATH_CALUDE_division_problem_l2505_250504

theorem division_problem (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 729 ∧ quotient = 19 ∧ remainder = 7 →
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 38 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2505_250504


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l2505_250515

theorem lcm_gcd_problem (a b : ℕ+) 
  (h1 : Nat.lcm a b = 3780)
  (h2 : Nat.gcd a b = 18)
  (h3 : a = 180) :
  b = 378 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l2505_250515


namespace NUMINAMATH_CALUDE_lcm_factor_problem_l2505_250513

theorem lcm_factor_problem (A B : ℕ+) (X : ℕ) (hcf : Nat.gcd A B = 42) 
  (lcm : Nat.lcm A B = 42 * X * 14) (a_val : A = 588) (a_greater : A > B) : X = 1 := by
  sorry

end NUMINAMATH_CALUDE_lcm_factor_problem_l2505_250513


namespace NUMINAMATH_CALUDE_inequality_proof_l2505_250531

theorem inequality_proof (a b c : ℝ) (h : a * b + b * c + c * a = 1) :
  |a - b| / |1 + c^2| + |b - c| / |1 + a^2| ≥ |c - a| / |1 + b^2| := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2505_250531


namespace NUMINAMATH_CALUDE_inequality_and_range_l2505_250571

theorem inequality_and_range (a b c m : ℝ) 
  (h1 : a + b + c + 2 - 2*m = 0)
  (h2 : a^2 + (1/4)*b^2 + (1/9)*c^2 + m - 1 = 0) :
  (a^2 + (1/4)*b^2 + (1/9)*c^2 ≥ (a + b + c)^2 / 14) ∧ 
  (-5/2 ≤ m ∧ m ≤ 1) := by
sorry

end NUMINAMATH_CALUDE_inequality_and_range_l2505_250571


namespace NUMINAMATH_CALUDE_correct_ticket_count_l2505_250576

/-- The number of stations between Ernakulam and Chennai -/
def num_stations : ℕ := 50

/-- The number of different train routes -/
def num_routes : ℕ := 3

/-- The number of second class tickets needed for one route -/
def tickets_per_route : ℕ := num_stations * (num_stations - 1) / 2

/-- The total number of second class tickets needed for all routes -/
def total_tickets : ℕ := num_routes * tickets_per_route

theorem correct_ticket_count : total_tickets = 3675 := by
  sorry

end NUMINAMATH_CALUDE_correct_ticket_count_l2505_250576


namespace NUMINAMATH_CALUDE_lcm_140_225_l2505_250592

theorem lcm_140_225 : Nat.lcm 140 225 = 6300 := by
  sorry

end NUMINAMATH_CALUDE_lcm_140_225_l2505_250592


namespace NUMINAMATH_CALUDE_recurrence_relation_and_generating_function_l2505_250579

def a (n : ℕ) : ℝ := (n^2 + 1) * 3^n

theorem recurrence_relation_and_generating_function :
  (∀ n : ℕ, a n - a (n + 1) + (1/3) * a (n + 2) - (1/27) * a (n + 3) = 0) ∧
  (∀ x : ℝ, abs x < 1/3 → ∑' (n : ℕ), a n * x^n = (1 - 3*x + 18*x^2) / (1 - 9*x + 27*x^2 - 27*x^3)) :=
by sorry

end NUMINAMATH_CALUDE_recurrence_relation_and_generating_function_l2505_250579


namespace NUMINAMATH_CALUDE_dress_count_proof_l2505_250540

def total_dresses (emily melissa debora sophia : ℕ) : ℕ :=
  emily + melissa + debora + sophia

theorem dress_count_proof 
  (emily : ℕ) 
  (h_emily : emily = 16)
  (melissa : ℕ) 
  (h_melissa : melissa = emily / 2)
  (debora : ℕ)
  (h_debora : debora = melissa + 12)
  (sophia : ℕ)
  (h_sophia : sophia = debora * 3 / 4) :
  total_dresses emily melissa debora sophia = 59 := by
sorry

end NUMINAMATH_CALUDE_dress_count_proof_l2505_250540


namespace NUMINAMATH_CALUDE_car_motorcycle_transaction_loss_l2505_250554

theorem car_motorcycle_transaction_loss : 
  ∀ (car_cost motorcycle_cost : ℝ),
  car_cost * (1 - 0.25) = 16000 →
  motorcycle_cost * (1 + 0.25) = 16000 →
  car_cost + motorcycle_cost - 2 * 16000 = 2133.33 := by
sorry

end NUMINAMATH_CALUDE_car_motorcycle_transaction_loss_l2505_250554


namespace NUMINAMATH_CALUDE_arithmetic_grid_solution_l2505_250524

/-- Represents a 7x1 arithmetic sequence -/
def RowSequence := Fin 7 → ℤ

/-- Represents a 4x1 arithmetic sequence -/
def ColumnSequence := Fin 4 → ℤ

/-- The problem setup -/
structure ArithmeticGrid :=
  (row : RowSequence)
  (col1 : ColumnSequence)
  (col2 : ColumnSequence)
  (is_arithmetic_row : ∀ i j k : Fin 7, i.val + 1 = j.val ∧ j.val + 1 = k.val → 
    row j - row i = row k - row j)
  (is_arithmetic_col1 : ∀ i j k : Fin 4, i.val + 1 = j.val ∧ j.val + 1 = k.val → 
    col1 j - col1 i = col1 k - col1 j)
  (is_arithmetic_col2 : ∀ i j k : Fin 4, i.val + 1 = j.val ∧ j.val + 1 = k.val → 
    col2 j - col2 i = col2 k - col2 j)
  (distinct_sequences : 
    (∀ i j : Fin 7, i ≠ j → row i - row j ≠ 0) ∧ 
    (∀ i j : Fin 4, i ≠ j → col1 i - col1 j ≠ 0) ∧ 
    (∀ i j : Fin 4, i ≠ j → col2 i - col2 j ≠ 0))
  (top_left : row 0 = 25)
  (middle_column : col1 1 = 12 ∧ col1 2 = 16)
  (bottom_right : col2 3 = -13)

/-- The main theorem -/
theorem arithmetic_grid_solution (grid : ArithmeticGrid) : grid.col2 0 = -16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_grid_solution_l2505_250524


namespace NUMINAMATH_CALUDE_harmonic_mean_counterexample_l2505_250518

theorem harmonic_mean_counterexample :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (2 / (1/a + 1/b) < Real.sqrt (a * b)) := by
  sorry

end NUMINAMATH_CALUDE_harmonic_mean_counterexample_l2505_250518


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l2505_250591

theorem more_girls_than_boys (total_students : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_students = 30 →
  boys + girls = total_students →
  2 * girls = 3 * boys →
  girls - boys = 6 := by
sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l2505_250591


namespace NUMINAMATH_CALUDE_modulus_of_z_l2505_250548

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition for z
def z_condition (z : ℂ) : Prop := z * (1 + i) = 2

-- State the theorem
theorem modulus_of_z (z : ℂ) (h : z_condition z) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2505_250548


namespace NUMINAMATH_CALUDE_men_to_women_ratio_l2505_250586

theorem men_to_women_ratio (men : ℝ) (women : ℝ) (h : women = 0.9 * men) :
  (men / women) * 100 = (1 / 0.9) * 100 := by
sorry

end NUMINAMATH_CALUDE_men_to_women_ratio_l2505_250586


namespace NUMINAMATH_CALUDE_no_valid_triples_l2505_250537

theorem no_valid_triples :
  ¬ ∃ (x y z : ℕ),
    (1 ≤ x) ∧ (x ≤ y) ∧ (y ≤ z) ∧
    (x * y * z + 2 * (x * y + y * z + z * x) = 2 * (2 * (x * y + y * z + z * x)) + 12) :=
by sorry


end NUMINAMATH_CALUDE_no_valid_triples_l2505_250537


namespace NUMINAMATH_CALUDE_problem_statement_l2505_250549

noncomputable def f (x : ℝ) : ℝ := 3^x + 2 / (1 - x)

theorem problem_statement 
  (x₀ x₁ x₂ : ℝ) 
  (h_root : f x₀ = 0)
  (h_x₁ : 1 < x₁ ∧ x₁ < x₀)
  (h_x₂ : x₀ < x₂) :
  f x₁ < 0 ∧ f x₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2505_250549


namespace NUMINAMATH_CALUDE_coordinate_axes_equiv_product_zero_l2505_250590

/-- The set of points on the coordinate axes in a Cartesian coordinate system -/
def CoordinateAxes : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0}

/-- The set of points where the product of coordinates is zero -/
def ProductZeroSet : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 * p.2 = 0}

theorem coordinate_axes_equiv_product_zero :
  CoordinateAxes = ProductZeroSet :=
sorry

end NUMINAMATH_CALUDE_coordinate_axes_equiv_product_zero_l2505_250590


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2505_250527

/-- Predicate defining when the equation represents an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  m - 1 > 0 ∧ 3 - m > 0 ∧ m - 1 ≠ 3 - m

/-- The condition given in the problem statement -/
def condition (m : ℝ) : Prop :=
  1 < m ∧ m < 3

/-- Theorem stating that the condition is necessary but not sufficient -/
theorem condition_necessary_not_sufficient :
  (∀ m : ℝ, is_ellipse m → condition m) ∧
  ¬(∀ m : ℝ, condition m → is_ellipse m) :=
sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2505_250527


namespace NUMINAMATH_CALUDE_correct_males_in_orchestra_l2505_250526

/-- The number of males in the orchestra -/
def males_in_orchestra : ℕ := 11

/-- The number of females in the orchestra -/
def females_in_orchestra : ℕ := 12

/-- The number of musicians in the orchestra -/
def musicians_in_orchestra : ℕ := males_in_orchestra + females_in_orchestra

/-- The number of musicians in the band -/
def musicians_in_band : ℕ := 2 * musicians_in_orchestra

/-- The number of musicians in the choir -/
def musicians_in_choir : ℕ := 12 + 17

/-- The total number of musicians in all three groups -/
def total_musicians : ℕ := 98

theorem correct_males_in_orchestra :
  musicians_in_orchestra + musicians_in_band + musicians_in_choir = total_musicians :=
sorry

end NUMINAMATH_CALUDE_correct_males_in_orchestra_l2505_250526


namespace NUMINAMATH_CALUDE_percentage_of_male_employees_l2505_250558

theorem percentage_of_male_employees
  (total_employees : ℕ)
  (males_below_50 : ℕ)
  (h_total : total_employees = 5200)
  (h_below_50 : males_below_50 = 1170)
  (h_half_above_50 : males_below_50 = (total_employees * (percentage_males / 100) / 2)) :
  percentage_males = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_of_male_employees_l2505_250558


namespace NUMINAMATH_CALUDE_banana_arrangement_count_l2505_250568

/-- The number of distinct arrangements of the letters in "BANANA" -/
def banana_arrangements : ℕ := 60

/-- The total number of letters in "BANANA" -/
def total_letters : ℕ := 6

/-- The number of occurrences of the letter 'A' in "BANANA" -/
def a_count : ℕ := 3

/-- The number of occurrences of the letter 'N' in "BANANA" -/
def n_count : ℕ := 2

/-- The number of occurrences of the letter 'B' in "BANANA" -/
def b_count : ℕ := 1

theorem banana_arrangement_count :
  banana_arrangements = (Nat.factorial total_letters) / ((Nat.factorial a_count) * (Nat.factorial n_count)) :=
by sorry

end NUMINAMATH_CALUDE_banana_arrangement_count_l2505_250568


namespace NUMINAMATH_CALUDE_specific_pairings_probability_l2505_250569

/-- The probability of two specific pairings occurring simultaneously in a class of 32 students -/
theorem specific_pairings_probability (n : ℕ) (h : n = 32) : 
  (1 : ℚ) / (n - 1) * (1 : ℚ) / (n - 3) = 1 / 899 :=
sorry

end NUMINAMATH_CALUDE_specific_pairings_probability_l2505_250569


namespace NUMINAMATH_CALUDE_stratified_sampling_male_count_l2505_250575

theorem stratified_sampling_male_count :
  let total_students : ℕ := 980
  let male_students : ℕ := 560
  let sample_size : ℕ := 280
  let sample_ratio : ℚ := sample_size / total_students
  sample_ratio * male_students = 160 := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_male_count_l2505_250575


namespace NUMINAMATH_CALUDE_equation_one_real_root_l2505_250563

theorem equation_one_real_root (t : ℝ) : 
  (∃! x : ℝ, 3 * x + 7 * t - 2 + (2 * t * x^2 + 7 * t^2 - 9) / (x - t) = 0) ↔ 
  (t = -3 ∨ t = -7/2 ∨ t = 1) := by sorry

end NUMINAMATH_CALUDE_equation_one_real_root_l2505_250563


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_three_numbers_l2505_250511

theorem arithmetic_mean_of_three_numbers (a b c : ℕ) (h : a = 25 ∧ b = 41 ∧ c = 50) : 
  (a + b + c) / 3 = 116 / 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_three_numbers_l2505_250511


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2505_250577

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |3*x + 1| - |x - 1| < 0} = {x : ℝ | -1 < x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2505_250577


namespace NUMINAMATH_CALUDE_jeanette_juggling_progress_l2505_250543

/-- Calculates the number of objects Jeanette can juggle after a given number of weeks -/
def juggle_objects (initial_objects : ℕ) (weekly_increase : ℕ) (sessions_per_week : ℕ) (session_increase : ℕ) (weeks : ℕ) : ℕ :=
  initial_objects + weeks * (weekly_increase + sessions_per_week * session_increase)

/-- Proves that Jeanette can juggle 21 objects by the end of the 5th week -/
theorem jeanette_juggling_progress : 
  juggle_objects 3 2 3 1 5 = 21 := by
  sorry

#eval juggle_objects 3 2 3 1 5

end NUMINAMATH_CALUDE_jeanette_juggling_progress_l2505_250543


namespace NUMINAMATH_CALUDE_system_solution_l2505_250505

theorem system_solution (x y z : ℝ) 
  (eq1 : y + z = 8 - 2*x)
  (eq2 : x + z = 10 - 2*y)
  (eq3 : x + y = 14 - 2*z) :
  2*x + 2*y + 2*z = 16 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2505_250505


namespace NUMINAMATH_CALUDE_pricing_theorem_l2505_250588

/-- Proves that for an item with a marked price 50% above its cost price,
    a discount of 23.33% on the marked price results in a 15% profit,
    and the final selling price is 115% of the cost price. -/
theorem pricing_theorem (cost_price : ℝ) (cost_price_pos : cost_price > 0) :
  let marked_price := cost_price * 1.5
  let discount_percentage := 23.33 / 100
  let selling_price := marked_price * (1 - discount_percentage)
  selling_price = cost_price * 1.15 ∧ 
  (selling_price - cost_price) / cost_price = 0.15 := by
  sorry

#check pricing_theorem

end NUMINAMATH_CALUDE_pricing_theorem_l2505_250588


namespace NUMINAMATH_CALUDE_ellipse_curve_l2505_250546

-- Define the set of points (x,y) parametrized by t
def ellipse_points : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, p.1 = 2 * Real.cos t ∧ p.2 = 3 * Real.sin t}

-- Define the standard form equation of an ellipse
def is_ellipse (S : Set (ℝ × ℝ)) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ p ∈ S, (p.1 / a)^2 + (p.2 / b)^2 = 1

-- Theorem statement
theorem ellipse_curve : is_ellipse ellipse_points := by
  sorry

end NUMINAMATH_CALUDE_ellipse_curve_l2505_250546


namespace NUMINAMATH_CALUDE_ratio_michael_monica_l2505_250507

-- Define the ages as real numbers
variable (patrick_age michael_age monica_age : ℝ)

-- Define the conditions
axiom ratio_patrick_michael : patrick_age / michael_age = 3 / 5
axiom sum_of_ages : patrick_age + michael_age + monica_age = 245
axiom age_difference : monica_age - patrick_age = 80

-- Theorem to prove
theorem ratio_michael_monica :
  michael_age / monica_age = 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_ratio_michael_monica_l2505_250507


namespace NUMINAMATH_CALUDE_limit_tan_sin_ratio_l2505_250564

open Real

noncomputable def f (x : ℝ) : ℝ := tan (6 * x) / sin (3 * x)

theorem limit_tan_sin_ratio :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → |f x - 2| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_tan_sin_ratio_l2505_250564


namespace NUMINAMATH_CALUDE_board_numbers_theorem_l2505_250597

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def proper_divisors (n : ℕ) : Set ℕ :=
  {d : ℕ | d ∣ n ∧ 1 < d ∧ d < n}

theorem board_numbers_theorem (n : ℕ) (hn : is_composite n) :
  (∃ m : ℕ, proper_divisors m = {d + 1 | d ∈ proper_divisors n}) ↔ n = 4 ∨ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_board_numbers_theorem_l2505_250597


namespace NUMINAMATH_CALUDE_sum_of_powers_of_three_l2505_250551

theorem sum_of_powers_of_three : (-3)^4 + (-3)^3 + (-3)^2 + 3^2 + 3^3 + 3^4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_three_l2505_250551


namespace NUMINAMATH_CALUDE_min_value_K_l2505_250519

theorem min_value_K (α β γ : ℝ) (hα : α > 0) (hβ : β > 0) (hγ : γ > 0) :
  (α + 3*γ)/(α + 2*β + γ) + 4*β/(α + β + 2*γ) - 8*γ/(α + β + 3*γ) ≥ 2/5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_K_l2505_250519


namespace NUMINAMATH_CALUDE_junk_mail_calculation_l2505_250533

/-- Calculates the total number of junk mail pieces per block -/
def total_junk_mail_per_block (houses_per_block : ℕ) (mail_per_house : ℕ) : ℕ :=
  houses_per_block * mail_per_house

/-- Theorem stating that the total junk mail per block is 640 -/
theorem junk_mail_calculation :
  total_junk_mail_per_block 20 32 = 640 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_calculation_l2505_250533


namespace NUMINAMATH_CALUDE_birds_in_marsh_end_of_day_l2505_250510

/-- Calculates the total number of birds in the marsh at the end of the day -/
def total_birds_end_of_day (initial_geese initial_ducks geese_departed swans_arrived herons_arrived : ℕ) : ℕ :=
  (initial_geese - geese_departed) + initial_ducks + swans_arrived + herons_arrived

/-- Theorem stating the total number of birds at the end of the day -/
theorem birds_in_marsh_end_of_day :
  total_birds_end_of_day 58 37 15 22 2 = 104 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_marsh_end_of_day_l2505_250510


namespace NUMINAMATH_CALUDE_square_nine_implies_fourth_power_eightyone_l2505_250542

theorem square_nine_implies_fourth_power_eightyone (a : ℝ) : a^2 = 9 → a^4 = 81 := by
  sorry

end NUMINAMATH_CALUDE_square_nine_implies_fourth_power_eightyone_l2505_250542


namespace NUMINAMATH_CALUDE_square_root_divided_by_15_equals_4_l2505_250581

theorem square_root_divided_by_15_equals_4 (x : ℝ) : 
  (Real.sqrt x) / 15 = 4 → x = 3600 := by
  sorry

end NUMINAMATH_CALUDE_square_root_divided_by_15_equals_4_l2505_250581


namespace NUMINAMATH_CALUDE_polynomial_equality_implies_sum_l2505_250565

theorem polynomial_equality_implies_sum (a b c d e f : ℝ) :
  (∀ x : ℝ, (3*x + 1)^5 = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) →
  a - b + c - d + e - f = 32 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_implies_sum_l2505_250565


namespace NUMINAMATH_CALUDE_geometric_series_properties_l2505_250589

theorem geometric_series_properties (q : ℝ) (b₁ : ℝ) (h_q : |q| < 1) :
  (b₁ / (1 - q) = 16) →
  (b₁^2 / (1 - q^2) = 153.6) →
  (b₁ * q^3 = 32/9 ∧ q = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_properties_l2505_250589


namespace NUMINAMATH_CALUDE_sine_ratio_in_triangle_l2505_250578

theorem sine_ratio_in_triangle (a b c : ℝ) (A B C : ℝ) :
  (b + c) / (c + a) = 4 / 5 ∧
  (c + a) / (a + b) = 5 / 6 ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C →
  Real.sin A / Real.sin B = 7 / 5 ∧
  Real.sin B / Real.sin C = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_sine_ratio_in_triangle_l2505_250578


namespace NUMINAMATH_CALUDE_parabola_tangent_ellipse_l2505_250547

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the tangent line
def tangent_line (x : ℝ) : ℝ := 4*x - 4

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

theorem parabola_tangent_ellipse :
  -- Conditions
  (∀ x, parabola x = x^2) →
  (parabola 2 = 4) →
  (tangent_line 2 = 4) →
  (tangent_line 1 = 0) →
  (tangent_line 0 = -4) →
  -- Conclusion
  ellipse (Real.sqrt 17) 4 1 0 ∧ ellipse (Real.sqrt 17) 4 0 (-4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_ellipse_l2505_250547


namespace NUMINAMATH_CALUDE_domain_transformation_l2505_250525

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the domain of f(x+1)
def domain_f_shifted : Set ℝ := Set.Icc 0 1

-- Define the domain of f(√(2x-1))
def domain_f_sqrt : Set ℝ := Set.Icc 1 (5/2)

-- State the theorem
theorem domain_transformation (h : ∀ x ∈ domain_f_shifted, f (x + 1) = f (x + 1)) :
  ∀ x ∈ domain_f_sqrt, f (Real.sqrt (2 * x - 1)) = f (Real.sqrt (2 * x - 1)) :=
sorry

end NUMINAMATH_CALUDE_domain_transformation_l2505_250525


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2505_250532

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^6 + X^5 + 2*X^3 - X^2 + 3 = (X + 2) * (X - 1) * q + (-X + 5) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2505_250532


namespace NUMINAMATH_CALUDE_sequence_representation_l2505_250580

def is_valid_sequence (q : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → q n < q m ∧ q n < 2 * n

theorem sequence_representation (q : ℕ → ℕ) (h : is_valid_sequence q) :
  ∀ m : ℕ, (∃ i : ℕ, q i = m) ∨ (∃ j k : ℕ, q j - q k = m) :=
by sorry

end NUMINAMATH_CALUDE_sequence_representation_l2505_250580


namespace NUMINAMATH_CALUDE_worker_wage_problem_l2505_250534

/-- Represents the daily wages and work days of three workers -/
structure WorkerData where
  a_wage : ℚ
  b_wage : ℚ
  c_wage : ℚ
  a_days : ℕ
  b_days : ℕ
  c_days : ℕ

/-- The theorem statement for the worker wage problem -/
theorem worker_wage_problem (data : WorkerData) 
  (h_ratio : data.a_wage / 3 = data.b_wage / 4 ∧ data.b_wage / 4 = data.c_wage / 5)
  (h_days : data.a_days = 6 ∧ data.b_days = 9 ∧ data.c_days = 4)
  (h_total : data.a_wage * data.a_days + data.b_wage * data.b_days + data.c_wage * data.c_days = 1850) :
  data.c_wage = 125 := by
  sorry

end NUMINAMATH_CALUDE_worker_wage_problem_l2505_250534


namespace NUMINAMATH_CALUDE_inequality_solution_l2505_250555

theorem inequality_solution (a : ℝ) :
  (a > 0 → (∀ x : ℝ, 6 * x^2 + a * x - a^2 < 0 ↔ -a/2 < x ∧ x < a/3)) ∧
  (a = 0 → ¬∃ x : ℝ, 6 * x^2 + a * x - a^2 < 0) ∧
  (a < 0 → (∀ x : ℝ, 6 * x^2 + a * x - a^2 < 0 ↔ a/3 < x ∧ x < -a/2)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2505_250555


namespace NUMINAMATH_CALUDE_andy_final_position_l2505_250557

/-- Represents a point in 2D space -/
structure Point where
  x : Int
  y : Int

/-- Represents a direction -/
inductive Direction
  | East
  | North
  | West
  | South

/-- Represents the state of Andy the Ant -/
structure AntState where
  position : Point
  direction : Direction
  moveCount : Nat

/-- Performs a single move for Andy the Ant -/
def move (state : AntState) : AntState :=
  sorry

/-- Performs n moves for Andy the Ant -/
def moveN (n : Nat) (state : AntState) : AntState :=
  sorry

/-- The main theorem to prove -/
theorem andy_final_position :
  let initialState : AntState := {
    position := { x := -10, y := 10 },
    direction := Direction.East,
    moveCount := 0
  }
  let finalState := moveN 2030 initialState
  finalState.position = { x := -3054, y := 3053 } :=
sorry

end NUMINAMATH_CALUDE_andy_final_position_l2505_250557


namespace NUMINAMATH_CALUDE_prism_coloring_iff_divisible_by_three_l2505_250598

/-- A prism with an n-gon base -/
structure Prism (n : ℕ) where
  base : Fin n → Fin 3  -- coloring of the base
  top : Fin n → Fin 3   -- coloring of the top

/-- Check if a coloring is valid for a prism -/
def is_valid_coloring (n : ℕ) (p : Prism n) : Prop :=
  ∀ (i : Fin n),
    -- Each vertex is connected to all three colors
    (∃ j, p.base j ≠ p.base i ∧ p.base j ≠ p.top i) ∧
    (∃ j, p.top j ≠ p.base i ∧ p.top j ≠ p.top i) ∧
    p.base i ≠ p.top i

theorem prism_coloring_iff_divisible_by_three (n : ℕ) :
  (∃ p : Prism n, is_valid_coloring n p) ↔ 3 ∣ n :=
sorry

end NUMINAMATH_CALUDE_prism_coloring_iff_divisible_by_three_l2505_250598


namespace NUMINAMATH_CALUDE_expression_value_l2505_250528

/-- The polynomial function p(x) = x^2 - x + 1 -/
def p (x : ℝ) : ℝ := x^2 - x + 1

/-- Theorem: If α is a root of p(p(p(p(x)))), then the given expression equals -1 -/
theorem expression_value (α : ℝ) (h : p (p (p (p α))) = 0) :
  (p α - 1) * p α * p (p α) * p (p (p α)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2505_250528


namespace NUMINAMATH_CALUDE_stating_currency_exchange_problem_l2505_250535

/-- Represents the exchange rate from U.S. dollars to Canadian dollars -/
def exchange_rate : ℚ := 12 / 8

/-- Represents the amount spent in Canadian dollars -/
def amount_spent : ℕ := 72

/-- 
Theorem stating that if a person exchanges m U.S. dollars to Canadian dollars
at the given exchange rate, spends the specified amount, and is left with m
Canadian dollars, then m must equal 144.
-/
theorem currency_exchange_problem (m : ℕ) :
  (m : ℚ) * exchange_rate - amount_spent = m →
  m = 144 :=
by sorry

end NUMINAMATH_CALUDE_stating_currency_exchange_problem_l2505_250535


namespace NUMINAMATH_CALUDE_triangular_arrangement_rows_l2505_250520

/-- The number of cans in a triangular arrangement with n rows -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The proposition to be proved -/
theorem triangular_arrangement_rows : 
  ∃ (n : ℕ), triangular_sum n = 480 - 15 ∧ n = 30 := by
  sorry

end NUMINAMATH_CALUDE_triangular_arrangement_rows_l2505_250520


namespace NUMINAMATH_CALUDE_no_real_solutions_for_f_iteration_l2505_250593

def f (x : ℝ) : ℝ := x^2 + 2*x

theorem no_real_solutions_for_f_iteration :
  ¬ ∃ c : ℝ, f (f (f (f c))) = -4 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_f_iteration_l2505_250593


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l2505_250536

/-- Calculates the length of a bridge given train parameters -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  255 = (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l2505_250536


namespace NUMINAMATH_CALUDE_janes_total_hours_l2505_250583

/-- Jane's exercise routine -/
structure ExerciseRoutine where
  hours_per_day : ℕ
  days_per_week : ℕ
  weeks : ℕ

/-- Calculate total exercise hours -/
def total_hours (routine : ExerciseRoutine) : ℕ :=
  routine.hours_per_day * routine.days_per_week * routine.weeks

/-- Jane's specific routine -/
def janes_routine : ExerciseRoutine :=
  { hours_per_day := 1
    days_per_week := 5
    weeks := 8 }

/-- Theorem: Jane's total exercise hours equal 40 -/
theorem janes_total_hours : total_hours janes_routine = 40 := by
  sorry

end NUMINAMATH_CALUDE_janes_total_hours_l2505_250583


namespace NUMINAMATH_CALUDE_llama_to_goat_ratio_l2505_250584

def goat_cost : ℕ := 400
def num_goats : ℕ := 3
def total_spent : ℕ := 4800

def llama_cost : ℕ := goat_cost + goat_cost / 2

def num_llamas : ℕ := (total_spent - num_goats * goat_cost) / llama_cost

theorem llama_to_goat_ratio :
  num_llamas * 1 = num_goats * 2 :=
by sorry

end NUMINAMATH_CALUDE_llama_to_goat_ratio_l2505_250584


namespace NUMINAMATH_CALUDE_or_true_iff_not_and_not_false_l2505_250503

theorem or_true_iff_not_and_not_false (p q : Prop) :
  (p ∨ q) ↔ ¬(¬p ∧ ¬q) :=
sorry

end NUMINAMATH_CALUDE_or_true_iff_not_and_not_false_l2505_250503


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l2505_250501

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ b : ℝ, (a + 1) * (a - 1 + Complex.I) = Complex.I * b) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l2505_250501


namespace NUMINAMATH_CALUDE_negative_two_in_M_l2505_250573

def M : Set ℝ := {x | x^2 - 4 = 0}

theorem negative_two_in_M : -2 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_negative_two_in_M_l2505_250573


namespace NUMINAMATH_CALUDE_orange_difference_l2505_250561

/-- The number of oranges and apples picked by George and Amelia -/
structure FruitPicking where
  george_oranges : ℕ
  george_apples : ℕ
  amelia_oranges : ℕ
  amelia_apples : ℕ

/-- The conditions of the fruit picking problem -/
def fruit_picking_conditions (fp : FruitPicking) : Prop :=
  fp.george_oranges = 45 ∧
  fp.george_apples = fp.amelia_apples + 5 ∧
  fp.amelia_oranges < fp.george_oranges ∧
  fp.amelia_apples = 15 ∧
  fp.george_oranges + fp.george_apples + fp.amelia_oranges + fp.amelia_apples = 107

/-- The theorem stating the difference in orange count -/
theorem orange_difference (fp : FruitPicking) 
  (h : fruit_picking_conditions fp) : 
  fp.george_oranges - fp.amelia_oranges = 18 := by
  sorry

end NUMINAMATH_CALUDE_orange_difference_l2505_250561


namespace NUMINAMATH_CALUDE_representatives_can_be_paired_l2505_250538

/-- A type representing a representative -/
def Representative : Type := ℕ

/-- A function that determines if a group of representatives can communicate -/
def can_communicate (group : Finset Representative) : Prop :=
  group.card = 3 → ∃ (a b c : Representative), a ∈ group ∧ b ∈ group ∧ c ∈ group ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- The set of all representatives -/
def all_representatives : Finset Representative :=
  Finset.range 1000

/-- The theorem stating that representatives can be paired in communicating rooms -/
theorem representatives_can_be_paired :
  (∀ (group : Finset Representative), group ⊆ all_representatives → can_communicate group) →
  ∃ (pairs : Finset (Finset Representative)),
    pairs.card = 500 ∧
    (∀ pair ∈ pairs, pair.card = 2) ∧
    (∀ pair ∈ pairs, can_communicate pair) ∧
    (∀ (a : Representative), a ∈ all_representatives → ∃! (pair : Finset Representative), pair ∈ pairs ∧ a ∈ pair) :=
by
  sorry

end NUMINAMATH_CALUDE_representatives_can_be_paired_l2505_250538


namespace NUMINAMATH_CALUDE_divisor_totient_sum_theorem_l2505_250523

def divisor_count (n : ℕ) : ℕ := (Nat.divisors n).card

theorem divisor_totient_sum_theorem (n : ℕ) (c : ℕ) :
  (n > 0) →
  (divisor_count n + Nat.totient n = n + c) ↔
  ((c = 1 ∧ (n = 1 ∨ Nat.Prime n ∨ n = 4)) ∨
   (c = 0 ∧ (n = 6 ∨ n = 8 ∨ n = 9))) :=
by sorry

end NUMINAMATH_CALUDE_divisor_totient_sum_theorem_l2505_250523


namespace NUMINAMATH_CALUDE_no_solution_absolute_value_plus_seven_l2505_250516

theorem no_solution_absolute_value_plus_seven :
  (∀ x : ℝ, |x| + 7 ≠ 0) ∧
  (∃ x : ℝ, (x - 5)^2 = 0) ∧
  (∃ x : ℝ, Real.sqrt (x + 9) - 3 = 0) ∧
  (∃ x : ℝ, (x + 4)^(1/3) - 1 = 0) ∧
  (∃ x : ℝ, |x + 6| - 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_absolute_value_plus_seven_l2505_250516


namespace NUMINAMATH_CALUDE_smallest_m_is_13_l2505_250582

/-- The set of complex numbers with real part between 1/2 and √2/2 -/
def S : Set ℂ :=
  {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ Real.sqrt 2 / 2}

/-- Property that for all n ≥ m, there exists a z in S such that z^n = 1 -/
def property (m : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ m → ∃ z : ℂ, z ∈ S ∧ z^n = 1

/-- Theorem stating that 13 is the smallest positive integer satisfying the property -/
theorem smallest_m_is_13 : 
  (property 13 ∧ ∀ m : ℕ, 0 < m → m < 13 → ¬property m) := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_is_13_l2505_250582


namespace NUMINAMATH_CALUDE_dilution_proof_l2505_250559

/-- Proves that adding 7.2 ounces of water to 12 ounces of 40% alcohol shaving lotion 
    results in a solution with 25% alcohol concentration -/
theorem dilution_proof (initial_volume : ℝ) (initial_concentration : ℝ) 
  (target_concentration : ℝ) (water_added : ℝ) : 
  initial_volume = 12 ∧ 
  initial_concentration = 0.4 ∧ 
  target_concentration = 0.25 ∧
  water_added = 7.2 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = target_concentration :=
by sorry

end NUMINAMATH_CALUDE_dilution_proof_l2505_250559


namespace NUMINAMATH_CALUDE_spider_dressing_theorem_l2505_250502

def spider_dressing_orders (n : ℕ) : ℚ :=
  (Nat.factorial (3 * n) * (4 ^ n)) / (6 ^ n)

theorem spider_dressing_theorem (n : ℕ) (hn : n = 8) :
  spider_dressing_orders n = (Nat.factorial (3 * n) * (4 ^ n)) / (6 ^ n) :=
by sorry

end NUMINAMATH_CALUDE_spider_dressing_theorem_l2505_250502


namespace NUMINAMATH_CALUDE_orange_harvest_theorem_l2505_250545

/-- Calculates the total number of orange sacks kept after a given number of harvest days. -/
def total_sacks_kept (daily_harvest : ℕ) (daily_discard : ℕ) (harvest_days : ℕ) : ℕ :=
  (daily_harvest - daily_discard) * harvest_days

/-- Proves that given the specified harvest conditions, the total number of sacks kept is 1425. -/
theorem orange_harvest_theorem :
  total_sacks_kept 150 135 95 = 1425 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_theorem_l2505_250545


namespace NUMINAMATH_CALUDE_flagpole_distance_l2505_250541

/-- Given a street of length 11.5 meters with 6 flagpoles placed at regular intervals,
    including both ends, the distance between adjacent flagpoles is 2.3 meters. -/
theorem flagpole_distance (street_length : ℝ) (num_flagpoles : ℕ) :
  street_length = 11.5 ∧ num_flagpoles = 6 →
  (street_length / (num_flagpoles - 1 : ℝ)) = 2.3 := by
  sorry

end NUMINAMATH_CALUDE_flagpole_distance_l2505_250541


namespace NUMINAMATH_CALUDE_complex_division_simplification_l2505_250567

theorem complex_division_simplification :
  (3 + Complex.I) / (1 + Complex.I) = 2 - Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l2505_250567


namespace NUMINAMATH_CALUDE_problem_statement_inequality_statement_l2505_250599

noncomputable section

def f (x : ℝ) : ℝ := x * Real.log x
def g (a x : ℝ) : ℝ := -x^2 + a*x - 3

theorem problem_statement (a : ℝ) : 
  (∀ x > 0, 2 * f x ≥ g a x) → a ≤ 4 :=
sorry

theorem inequality_statement : 
  ∀ x > 0, Real.log x > 1 / Real.exp x - 2 / (Real.exp 1 * x) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_inequality_statement_l2505_250599


namespace NUMINAMATH_CALUDE_triangle_side_b_l2505_250539

theorem triangle_side_b (a b c : ℝ) (A B C : ℝ) : 
  a = 8 → B = π/3 → C = 5*π/12 → b = 4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_b_l2505_250539


namespace NUMINAMATH_CALUDE_calculate_markup_l2505_250500

/-- Calculates the markup for an article given its purchase price, overhead percentage, and desired net profit. -/
theorem calculate_markup (purchase_price overhead_percent net_profit : ℚ) : 
  purchase_price = 48 →
  overhead_percent = 15 / 100 →
  net_profit = 12 →
  purchase_price + overhead_percent * purchase_price + net_profit - purchase_price = 19.2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_markup_l2505_250500


namespace NUMINAMATH_CALUDE_lcm_factor_proof_l2505_250570

theorem lcm_factor_proof (A B : ℕ) (x : ℕ) (h1 : Nat.gcd A B = 23) (h2 : A = 391) 
  (h3 : Nat.lcm A B = 23 * 17 * x) : x = 17 := by
  sorry

end NUMINAMATH_CALUDE_lcm_factor_proof_l2505_250570


namespace NUMINAMATH_CALUDE_factorial_divisibility_l2505_250514

theorem factorial_divisibility (n : ℕ) : 
  (∃ (p q : ℕ), p ≤ n ∧ q ≤ n ∧ n + 2 = p * q) ∨ 
  (∃ (p : ℕ), p ≥ 3 ∧ Prime p ∧ n + 2 = p^2) ↔ 
  (n + 2) ∣ n! :=
sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l2505_250514


namespace NUMINAMATH_CALUDE_digits_of_2_12_times_5_8_l2505_250517

theorem digits_of_2_12_times_5_8 : 
  (Nat.log 10 (2^12 * 5^8) + 1 : ℕ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_digits_of_2_12_times_5_8_l2505_250517


namespace NUMINAMATH_CALUDE_bicycle_profit_problem_l2505_250587

theorem bicycle_profit_problem (initial_cost final_price : ℝ) : 
  (initial_cost * 1.25 * 1.25 = final_price) →
  (final_price = 225) →
  (initial_cost = 144) := by
sorry

end NUMINAMATH_CALUDE_bicycle_profit_problem_l2505_250587


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2505_250522

def p (x : ℂ) : ℂ := 5 * x^5 + 18 * x^3 - 45 * x^2 + 30 * x

theorem roots_of_polynomial :
  ∀ x : ℂ, p x = 0 ↔ x = 0 ∨ x = 1/5 ∨ x = Complex.I * Real.sqrt 3 ∨ x = -Complex.I * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2505_250522


namespace NUMINAMATH_CALUDE_sales_difference_l2505_250595

-- Define the regular day sales quantities
def regular_croissants : ℕ := 10
def regular_muffins : ℕ := 10
def regular_sourdough : ℕ := 6
def regular_wholewheat : ℕ := 4

-- Define the Monday sales quantities
def monday_croissants : ℕ := 8
def monday_muffins : ℕ := 6
def monday_sourdough : ℕ := 15
def monday_wholewheat : ℕ := 10

-- Define the regular prices
def price_croissant : ℚ := 2.5
def price_muffin : ℚ := 1.75
def price_sourdough : ℚ := 4.25
def price_wholewheat : ℚ := 5

-- Define the discount rate
def discount_rate : ℚ := 0.1

-- Calculate the daily average sales
def daily_average : ℚ :=
  regular_croissants * price_croissant +
  regular_muffins * price_muffin +
  regular_sourdough * price_sourdough +
  regular_wholewheat * price_wholewheat

-- Calculate the Monday sales with discount
def monday_sales : ℚ :=
  monday_croissants * price_croissant * (1 - discount_rate) +
  monday_muffins * price_muffin * (1 - discount_rate) +
  monday_sourdough * price_sourdough * (1 - discount_rate) +
  monday_wholewheat * price_wholewheat * (1 - discount_rate)

-- State the theorem
theorem sales_difference : monday_sales - daily_average = 41.825 := by sorry

end NUMINAMATH_CALUDE_sales_difference_l2505_250595


namespace NUMINAMATH_CALUDE_min_sum_squares_complex_l2505_250552

theorem min_sum_squares_complex (w : ℂ) (h : Complex.abs (w - (3 - 2*I)) = 4) :
  ∃ (min : ℝ), min = 48 ∧ 
  ∀ (z : ℂ), Complex.abs (z - (3 - 2*I)) = 4 →
    Complex.abs (z + (1 + 2*I))^2 + Complex.abs (z - (7 + 2*I))^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_complex_l2505_250552


namespace NUMINAMATH_CALUDE_distance_A_to_B_l2505_250566

def point_A : Fin 3 → ℝ := ![2, 3, 5]
def point_B : Fin 3 → ℝ := ![3, 1, 7]

theorem distance_A_to_B :
  Real.sqrt ((point_B 0 - point_A 0)^2 + (point_B 1 - point_A 1)^2 + (point_B 2 - point_A 2)^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_A_to_B_l2505_250566
