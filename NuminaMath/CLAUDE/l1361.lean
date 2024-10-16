import Mathlib

namespace NUMINAMATH_CALUDE_system_of_inequalities_solution_l1361_136144

theorem system_of_inequalities_solution (x : ℝ) :
  (x < x / 5 + 4 ∧ 4 * x + 1 > 3 * (2 * x - 1)) → x < 2 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_solution_l1361_136144


namespace NUMINAMATH_CALUDE_translation_theorem_l1361_136106

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line horizontally and vertically -/
def translate (l : Line) (dx dy : ℝ) : Line :=
  { slope := l.slope,
    intercept := l.intercept - l.slope * dx + dy }

/-- The original line y = 2x - 3 -/
def original_line : Line :=
  { slope := 2,
    intercept := -3 }

/-- The amount of horizontal translation -/
def right_shift : ℝ := 2

/-- The amount of vertical translation -/
def up_shift : ℝ := 1

/-- The expected resulting line after translation -/
def expected_result : Line :=
  { slope := 2,
    intercept := -6 }

theorem translation_theorem :
  translate original_line right_shift up_shift = expected_result := by
  sorry

end NUMINAMATH_CALUDE_translation_theorem_l1361_136106


namespace NUMINAMATH_CALUDE_compound_weight_l1361_136182

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (carbon_count : ℕ) (hydrogen_count : ℕ) (oxygen_count : ℕ) 
  (carbon_weight : ℝ) (hydrogen_weight : ℝ) (oxygen_weight : ℝ) : ℝ :=
  (carbon_count : ℝ) * carbon_weight + 
  (hydrogen_count : ℝ) * hydrogen_weight + 
  (oxygen_count : ℝ) * oxygen_weight

theorem compound_weight : 
  molecular_weight 2 4 2 12.01 1.008 16.00 = 60.052 := by
  sorry

end NUMINAMATH_CALUDE_compound_weight_l1361_136182


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1361_136145

theorem absolute_value_equation_solution (x : ℝ) : 
  |24 / x + 4| = 4 → x = -3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1361_136145


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l1361_136134

def m : ℕ := 2023^2 + 2^2023

theorem units_digit_of_m_squared_plus_two_to_m (m : ℕ) : 
  m = 2023^2 + 2^2023 → (m^2 + 2^m) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l1361_136134


namespace NUMINAMATH_CALUDE_cubic_trinomial_condition_l1361_136124

/-- 
Given a polynomial of the form 3xy^(|m|) - (1/4)(m-2)xy + 1,
prove that for it to be a cubic trinomial, m must equal -2.
-/
theorem cubic_trinomial_condition (m : ℤ) : 
  (abs m = 2) ∧ ((1/4 : ℚ) * (m - 2) ≠ 0) → m = -2 := by sorry

end NUMINAMATH_CALUDE_cubic_trinomial_condition_l1361_136124


namespace NUMINAMATH_CALUDE_candidate_selection_probability_l1361_136190

/-- Represents the probability distribution of Excel skills among job candidates -/
structure ExcelSkills where
  beginner : ℝ
  intermediate : ℝ
  advanced : ℝ
  none : ℝ
  sum_to_one : beginner + intermediate + advanced + none = 1

/-- Represents the probability distribution of shift preferences among job candidates -/
structure ShiftPreference where
  day : ℝ
  night : ℝ
  sum_to_one : day + night = 1

/-- Represents the probability distribution of weekend work preferences among job candidates -/
structure WeekendPreference where
  willing : ℝ
  not_willing : ℝ
  sum_to_one : willing + not_willing = 1

/-- Theorem stating the probability of selecting a candidate with specific characteristics -/
theorem candidate_selection_probability 
  (excel : ExcelSkills)
  (shift : ShiftPreference)
  (weekend : WeekendPreference)
  (h1 : excel.beginner = 0.35)
  (h2 : excel.intermediate = 0.25)
  (h3 : excel.advanced = 0.2)
  (h4 : excel.none = 0.2)
  (h5 : shift.day = 0.7)
  (h6 : shift.night = 0.3)
  (h7 : weekend.willing = 0.4)
  (h8 : weekend.not_willing = 0.6) :
  (excel.intermediate + excel.advanced) * shift.night * weekend.not_willing = 0.081 := by
  sorry

end NUMINAMATH_CALUDE_candidate_selection_probability_l1361_136190


namespace NUMINAMATH_CALUDE_doughnuts_per_person_l1361_136163

/-- The number of doughnuts each person receives when Samuel and Cathy share their doughnuts with friends -/
theorem doughnuts_per_person :
  ∀ (samuel_dozens cathy_dozens num_friends : ℕ),
  samuel_dozens = 2 →
  cathy_dozens = 3 →
  num_friends = 8 →
  (samuel_dozens * 12 + cathy_dozens * 12) / (num_friends + 2) = 6 :=
by sorry

end NUMINAMATH_CALUDE_doughnuts_per_person_l1361_136163


namespace NUMINAMATH_CALUDE_fedya_deposit_l1361_136141

theorem fedya_deposit (n k : ℕ) : 
  0 < k ∧ k < 30 ∧ 
  n * (100 - k) = 84700 ∧
  ∀ (m l : ℕ), 0 < l ∧ l < 30 ∧ m * (100 - l) = 84700 → m = n ∧ l = k →
  n = 1100 ∧ k = 23 := by
  sorry

end NUMINAMATH_CALUDE_fedya_deposit_l1361_136141


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l1361_136100

theorem square_area_from_diagonal (diagonal : ℝ) (area : ℝ) : 
  diagonal = 12 → area = diagonal^2 / 2 → area = 72 := by
  sorry

#check square_area_from_diagonal

end NUMINAMATH_CALUDE_square_area_from_diagonal_l1361_136100


namespace NUMINAMATH_CALUDE_correlation_coefficient_comparison_l1361_136137

def X : List Float := [10, 11.3, 11.8, 12.5, 13]
def Y : List Float := [1, 2, 3, 4, 5]
def U : List Float := [10, 11.3, 11.8, 12.5, 13]
def V : List Float := [5, 4, 3, 2, 1]

def linear_correlation_coefficient (x : List Float) (y : List Float) : Float :=
  sorry

def r₁ : Float := linear_correlation_coefficient X Y
def r₂ : Float := linear_correlation_coefficient U V

theorem correlation_coefficient_comparison : r₂ < r₁ := by
  sorry

end NUMINAMATH_CALUDE_correlation_coefficient_comparison_l1361_136137


namespace NUMINAMATH_CALUDE_connor_sleep_time_l1361_136162

theorem connor_sleep_time (puppy_sleep : ℕ) (luke_sleep : ℕ) (connor_sleep : ℕ) : 
  puppy_sleep = 16 →
  puppy_sleep = 2 * luke_sleep →
  luke_sleep = connor_sleep + 2 →
  connor_sleep = 6 :=
by sorry

end NUMINAMATH_CALUDE_connor_sleep_time_l1361_136162


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1361_136199

theorem geometric_sequence_property (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- Geometric sequence property
  q > 0 →  -- Common ratio is positive
  (3 * a 1 + 2 * a 2 = a 3) →  -- Arithmetic sequence property
  q = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1361_136199


namespace NUMINAMATH_CALUDE_seans_fraction_of_fritz_money_l1361_136113

theorem seans_fraction_of_fritz_money (fritz_money sean_money rick_money : ℚ) 
  (x : ℚ) : 
  fritz_money = 40 →
  sean_money = x * fritz_money + 4 →
  rick_money = 3 * sean_money →
  rick_money + sean_money = 96 →
  x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_seans_fraction_of_fritz_money_l1361_136113


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_correct_l1361_136102

/-- The area of a quadrilateral with vertices at (2, 2), (2, -1), (3, -1), and (2007, 2008) -/
def quadrilateralArea : ℝ := 2008006.5

/-- The vertices of the quadrilateral -/
def vertices : List (ℝ × ℝ) := [(2, 2), (2, -1), (3, -1), (2007, 2008)]

/-- Theorem stating that the area of the quadrilateral with the given vertices is 2008006.5 -/
theorem quadrilateral_area_is_correct : 
  let area := quadrilateralArea
  ∃ (f : List (ℝ × ℝ) → ℝ), f vertices = area :=
by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_area_is_correct_l1361_136102


namespace NUMINAMATH_CALUDE_no_integer_solution_l1361_136176

theorem no_integer_solution :
  ∀ (x y z : ℤ), x ≠ 0 → 2 * x^4 + 2 * x^2 * y^2 + y^4 ≠ z^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1361_136176


namespace NUMINAMATH_CALUDE_trapezoid_area_l1361_136170

/-- Represents a rectangle PQRS with a trapezoid TURS inside it -/
structure RectangleWithTrapezoid where
  /-- Length of the rectangle PQRS -/
  length : ℝ
  /-- Width of the rectangle PQRS -/
  width : ℝ
  /-- Distance from P to T (same as distance from Q to U) -/
  side_length : ℝ
  /-- Area of rectangle PQRS is 24 -/
  area_eq : length * width = 24
  /-- T and U are on the top side of PQRS -/
  side_constraint : side_length < length

/-- The area of trapezoid TURS is 16 square units -/
theorem trapezoid_area (rect : RectangleWithTrapezoid) : 
  rect.width * (rect.length - 2 * rect.side_length) + 2 * (rect.side_length * rect.width / 2) = 16 := by
  sorry

#check trapezoid_area

end NUMINAMATH_CALUDE_trapezoid_area_l1361_136170


namespace NUMINAMATH_CALUDE_zoo_tickets_problem_l1361_136156

/-- Proves that for a family of seven people buying zoo tickets, where adult tickets 
    cost $21 and children's tickets cost $14, if the total cost is $119, 
    then the number of adult tickets purchased is 3. -/
theorem zoo_tickets_problem (adult_cost children_cost total_cost : ℕ) 
  (family_size : ℕ) (num_adults : ℕ) :
  adult_cost = 21 →
  children_cost = 14 →
  total_cost = 119 →
  family_size = 7 →
  num_adults + (family_size - num_adults) = family_size →
  num_adults * adult_cost + (family_size - num_adults) * children_cost = total_cost →
  num_adults = 3 := by
  sorry

end NUMINAMATH_CALUDE_zoo_tickets_problem_l1361_136156


namespace NUMINAMATH_CALUDE_larger_number_in_sum_and_difference_l1361_136195

theorem larger_number_in_sum_and_difference (x y : ℝ) 
  (sum_eq : x + y = 40) 
  (diff_eq : x - y = 6) : 
  max x y = 23 := by
sorry

end NUMINAMATH_CALUDE_larger_number_in_sum_and_difference_l1361_136195


namespace NUMINAMATH_CALUDE_divide_by_recurring_decimal_l1361_136117

/-- The recurring decimal 0.363636... represented as a rational number -/
def recurring_decimal : ℚ := 4 / 11

/-- The result of dividing 12 by the recurring decimal 0.363636... -/
theorem divide_by_recurring_decimal : 12 / recurring_decimal = 33 := by
  sorry

end NUMINAMATH_CALUDE_divide_by_recurring_decimal_l1361_136117


namespace NUMINAMATH_CALUDE_karen_grooms_one_chihuahua_l1361_136132

/-- The time it takes to groom a Rottweiler -/
def rottweiler_time : ℕ := 20

/-- The time it takes to groom a border collie -/
def border_collie_time : ℕ := 10

/-- The time it takes to groom a chihuahua -/
def chihuahua_time : ℕ := 45

/-- The total time Karen spends grooming -/
def total_time : ℕ := 255

/-- The number of Rottweilers Karen grooms -/
def num_rottweilers : ℕ := 6

/-- The number of border collies Karen grooms -/
def num_border_collies : ℕ := 9

/-- The number of chihuahuas Karen grooms -/
def num_chihuahuas : ℕ := 1

theorem karen_grooms_one_chihuahua :
  num_chihuahuas * chihuahua_time =
  total_time - (num_rottweilers * rottweiler_time + num_border_collies * border_collie_time) :=
by sorry

end NUMINAMATH_CALUDE_karen_grooms_one_chihuahua_l1361_136132


namespace NUMINAMATH_CALUDE_zou_mei_competition_l1361_136181

theorem zou_mei_competition (n : ℕ) : 
  n^2 + 15 + 18 = (n + 1)^2 → n^2 + 15 = 271 := by
  sorry

end NUMINAMATH_CALUDE_zou_mei_competition_l1361_136181


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l1361_136121

theorem equilateral_triangle_area (h : ℝ) (A : ℝ) : 
  h = 3 * Real.sqrt 3 → A = (Real.sqrt 3 / 4) * (2 * h / Real.sqrt 3)^2 → A = 9 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l1361_136121


namespace NUMINAMATH_CALUDE_batsman_total_score_l1361_136183

/-- Represents the score of a batsman in cricket --/
structure BatsmanScore where
  total : ℝ
  boundaries : ℕ
  sixes : ℕ
  runningPercentage : ℝ

/-- The total score of a batsman is 120 runs given the specified conditions --/
theorem batsman_total_score 
  (score : BatsmanScore) 
  (h1 : score.boundaries = 5) 
  (h2 : score.sixes = 5) 
  (h3 : score.runningPercentage = 58.333333333333336) :
  score.total = 120 := by
  sorry

end NUMINAMATH_CALUDE_batsman_total_score_l1361_136183


namespace NUMINAMATH_CALUDE_triangle_max_area_l1361_136107

theorem triangle_max_area (a b : ℝ) (h1 : a + b = 4) (h2 : 0 < a) (h3 : 0 < b) : 
  (1/2 : ℝ) * a * b * Real.sin (π/6) ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1361_136107


namespace NUMINAMATH_CALUDE_expression_evaluation_l1361_136173

theorem expression_evaluation :
  ∀ (a b c d : ℝ),
    d = c + 1 →
    c = b - 8 →
    b = a + 4 →
    a = 7 →
    a + 3 ≠ 0 →
    b - 3 ≠ 0 →
    c + 10 ≠ 0 →
    d + 1 ≠ 0 →
    ((a + 5) / (a + 3)) * ((b - 2) / (b - 3)) * ((c + 7) / (c + 10)) * ((d - 4) / (d + 1)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1361_136173


namespace NUMINAMATH_CALUDE_arithmetic_equation_l1361_136184

theorem arithmetic_equation : 8 + 15 / 3 - 4 * 2 + 2^3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l1361_136184


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l1361_136193

theorem largest_multiple_of_15_under_500 : ∃ (n : ℕ), n * 15 = 495 ∧ 
  495 < 500 ∧ 
  ∀ (m : ℕ), m * 15 < 500 → m * 15 ≤ 495 := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l1361_136193


namespace NUMINAMATH_CALUDE_exists_n_for_root_1000_l1361_136177

theorem exists_n_for_root_1000 : ∃ n : ℕ, (1000 : ℝ) ^ (1 / n) < 1.001 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_for_root_1000_l1361_136177


namespace NUMINAMATH_CALUDE_solution_set_and_range_of_a_l1361_136189

def f (a x : ℝ) : ℝ := |x - a| + x

theorem solution_set_and_range_of_a :
  (∀ x : ℝ, f 3 x ≥ x + 4 ↔ (x ≤ -1 ∨ x ≥ 7)) ∧
  (∀ a : ℝ, a > 0 →
    (∀ x : ℝ, x ∈ Set.Icc 1 3 → f a x ≥ x + 2 * a^2) ↔
    (-1 ≤ a ∧ a ≤ 1/2)) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_and_range_of_a_l1361_136189


namespace NUMINAMATH_CALUDE_process_terminates_with_bound_bound_is_tight_l1361_136147

/-- Represents the state of the queue -/
structure QueueState where
  n : ℕ
  positions : Fin n → Fin n

/-- Represents a single move in the process -/
structure Move where
  i : ℕ

/-- The result of applying a move to a queue state -/
inductive MoveResult
  | Continue (new_state : QueueState) (euros_paid : ℕ)
  | End

/-- Applies a move to a queue state -/
def apply_move (state : QueueState) (move : Move) : MoveResult :=
  sorry

/-- Represents a sequence of moves -/
def MoveSequence := List Move

/-- Applies a sequence of moves to an initial state -/
def apply_moves (initial_state : QueueState) (moves : MoveSequence) : ℕ :=
  sorry

theorem process_terminates_with_bound (n : ℕ) :
  ∀ (initial_state : QueueState),
  ∀ (moves : MoveSequence),
  apply_moves initial_state moves ≤ 2^n - n - 1 :=
sorry

theorem bound_is_tight (n : ℕ) :
  ∃ (initial_state : QueueState),
  ∃ (moves : MoveSequence),
  apply_moves initial_state moves = 2^n - n - 1 :=
sorry

end NUMINAMATH_CALUDE_process_terminates_with_bound_bound_is_tight_l1361_136147


namespace NUMINAMATH_CALUDE_twenty_is_eighty_percent_of_twentyfive_l1361_136111

theorem twenty_is_eighty_percent_of_twentyfive (x : ℝ) : 20 = 0.8 * x → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_twenty_is_eighty_percent_of_twentyfive_l1361_136111


namespace NUMINAMATH_CALUDE_multiples_of_four_l1361_136127

theorem multiples_of_four (n : ℕ) : 
  n ≤ 104 → 
  (∃ (k : ℕ), k = 24 ∧ 
    (∀ (i : ℕ), i ≤ k → 
      ∃ (m : ℕ), m * 4 = n + (i - 1) * 4 ∧ m * 4 ≤ 104)) → 
  n = 88 := by
sorry

end NUMINAMATH_CALUDE_multiples_of_four_l1361_136127


namespace NUMINAMATH_CALUDE_least_n_for_determinant_l1361_136188

theorem least_n_for_determinant (n : ℕ) : n ≥ 1 → (∀ k < n, 2^(k-1) < 2015) → 2^(n-1) ≥ 2015 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_least_n_for_determinant_l1361_136188


namespace NUMINAMATH_CALUDE_bread_slices_left_l1361_136160

/-- The number of slices of bread Tony uses per sandwich -/
def slices_per_sandwich : ℕ := 2

/-- The number of sandwiches Tony made from Monday to Friday -/
def weekday_sandwiches : ℕ := 5

/-- The number of sandwiches Tony made on Saturday -/
def saturday_sandwiches : ℕ := 2

/-- The total number of slices in the loaf Tony started with -/
def initial_slices : ℕ := 22

/-- Theorem stating the number of bread slices left after Tony made sandwiches for the week -/
theorem bread_slices_left : 
  initial_slices - (slices_per_sandwich * (weekday_sandwiches + saturday_sandwiches)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_bread_slices_left_l1361_136160


namespace NUMINAMATH_CALUDE_power_equality_l1361_136125

theorem power_equality : 32^5 * 4^3 = 2^31 := by sorry

end NUMINAMATH_CALUDE_power_equality_l1361_136125


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1361_136139

theorem simplify_and_evaluate (x : ℝ) (h : x = 4) :
  (x - 1 - 3 / (x + 1)) / ((x^2 + 2*x) / (x + 1)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1361_136139


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1361_136109

theorem complex_modulus_problem (z : ℂ) (h : (3 - I) / z = 1 + I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1361_136109


namespace NUMINAMATH_CALUDE_salary_change_l1361_136153

theorem salary_change (initial_salary : ℝ) (h : initial_salary > 0) :
  let increased_salary := initial_salary * (1 + 0.2)
  let final_salary := increased_salary * (1 - 0.2)
  (final_salary - initial_salary) / initial_salary = -0.04 :=
by
  sorry

end NUMINAMATH_CALUDE_salary_change_l1361_136153


namespace NUMINAMATH_CALUDE_weekly_allowance_calculation_l1361_136168

/-- Proves that if a person spends 3/5 of their allowance, then 1/3 of the remainder, 
    and finally $0.96, their original allowance was $3.60. -/
theorem weekly_allowance_calculation (A : ℝ) : 
  (A > 0) →
  ((2/5) * A - (1/3) * ((2/5) * A) = 0.96) →
  A = 3.60 := by
sorry

end NUMINAMATH_CALUDE_weekly_allowance_calculation_l1361_136168


namespace NUMINAMATH_CALUDE_batsman_average_after_15th_inning_l1361_136171

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  innings : ℕ
  totalScore : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (stats : BatsmanStats) (score : ℕ) : ℚ :=
  (stats.totalScore + score : ℚ) / (stats.innings + 1)

/-- Theorem: A batsman's new average after the 15th inning is 33 -/
theorem batsman_average_after_15th_inning
  (stats : BatsmanStats)
  (h1 : stats.innings = 14)
  (h2 : newAverage stats 75 = stats.average + 3)
  : newAverage stats 75 = 33 := by
  sorry


end NUMINAMATH_CALUDE_batsman_average_after_15th_inning_l1361_136171


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1361_136143

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (d : ℝ)
  (h_arithmetic : arithmetic_sequence a d)
  (h_sum1 : a 3 + a 6 = 11)
  (h_sum2 : a 5 + a 8 = 39) :
  d = 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1361_136143


namespace NUMINAMATH_CALUDE_subtracted_value_proof_l1361_136180

theorem subtracted_value_proof (N : ℕ) (h : N = 2976) : ∃ V : ℚ, (N / 12 : ℚ) - V = 8 ∧ V = 240 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_proof_l1361_136180


namespace NUMINAMATH_CALUDE_negative_cube_squared_l1361_136129

theorem negative_cube_squared (a : ℝ) : (-a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l1361_136129


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_ninety_l1361_136185

theorem thirty_percent_less_than_ninety (x : ℝ) : x = 50 ↔ (1 + 1/4) * x = 90 * (1 - 3/10) := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_ninety_l1361_136185


namespace NUMINAMATH_CALUDE_right_triangle_consecutive_sides_l1361_136126

theorem right_triangle_consecutive_sides (a c : ℕ) (b : ℝ) : 
  c = a + 1 → -- c and a are consecutive integers
  a^2 + b^2 = c^2 → -- Pythagorean theorem
  b^2 = c + a := by
sorry

end NUMINAMATH_CALUDE_right_triangle_consecutive_sides_l1361_136126


namespace NUMINAMATH_CALUDE_heartsuit_four_six_l1361_136142

-- Define the ♥ operation
def heartsuit (x y : ℝ) : ℝ := 5*x + 3*y

-- Theorem statement
theorem heartsuit_four_six : heartsuit 4 6 = 38 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_four_six_l1361_136142


namespace NUMINAMATH_CALUDE_smallest_bound_is_two_l1361_136166

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, x ∈ Set.Icc 0 1 → f x ≥ 0) ∧
  (∀ x₁ x₂, x₁ ≥ 0 → x₂ ≥ 0 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂) ∧
  f 0 = 0 ∧ f 1 = 1

/-- The theorem stating that 2 is the smallest positive number c such that f(x) ≤ cx for all x ∈ [0,1] -/
theorem smallest_bound_is_two (f : ℝ → ℝ) (h : SatisfyingFunction f) :
  (∀ c > 0, (∀ x ∈ Set.Icc 0 1, f x ≤ c * x) → c ≥ 2) ∧
  (∀ x ∈ Set.Icc 0 1, f x ≤ 2 * x) :=
sorry

end NUMINAMATH_CALUDE_smallest_bound_is_two_l1361_136166


namespace NUMINAMATH_CALUDE_square_area_ratio_l1361_136175

theorem square_area_ratio (side_C side_D : ℝ) (h1 : side_C = 12.5) (h2 : side_D = 18.5) :
  (side_C^2) / (side_D^2) = 625 / 1369 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1361_136175


namespace NUMINAMATH_CALUDE_can_collection_ratio_l1361_136148

theorem can_collection_ratio : 
  ∀ (total LaDonna Yoki Prikya : ℕ),
    total = 85 →
    LaDonna = 25 →
    Yoki = 10 →
    Prikya = total - LaDonna - Yoki →
    (Prikya : ℚ) / LaDonna = 2 := by
  sorry

end NUMINAMATH_CALUDE_can_collection_ratio_l1361_136148


namespace NUMINAMATH_CALUDE_candy_left_l1361_136135

/-- Represents the number of candy pieces Debby has -/
def candy_count : ℕ := 12

/-- Represents the number of candy pieces Debby ate -/
def eaten_candy : ℕ := 9

/-- Theorem stating how many pieces of candy Debby has left -/
theorem candy_left : candy_count - eaten_candy = 3 := by sorry

end NUMINAMATH_CALUDE_candy_left_l1361_136135


namespace NUMINAMATH_CALUDE_top_number_after_folds_l1361_136187

/-- Represents a 4x4 grid of numbers -/
def Grid := Fin 4 → Fin 4 → Fin 16

/-- The initial configuration of the grid -/
def initial_grid : Grid :=
  fun i j => ⟨i.val * 4 + j.val + 1, by sorry⟩

/-- Fold the right half over the left half -/
def fold_right_left (g : Grid) : Grid :=
  fun i j => g i (Fin.cast (by sorry) (3 - j))

/-- Fold the top half over the bottom half -/
def fold_top_bottom (g : Grid) : Grid :=
  fun i j => g (Fin.cast (by sorry) (3 - i)) j

/-- Fold the bottom half over the top half -/
def fold_bottom_top (g : Grid) : Grid :=
  fun i j => g (Fin.cast (by sorry) (3 - i)) j

/-- Fold the left half over the right half -/
def fold_left_right (g : Grid) : Grid :=
  fun i j => g i (Fin.cast (by sorry) (3 - j))

/-- Apply all folding operations in sequence -/
def apply_all_folds (g : Grid) : Grid :=
  fold_left_right ∘ fold_bottom_top ∘ fold_top_bottom ∘ fold_right_left $ g

theorem top_number_after_folds :
  (apply_all_folds initial_grid 0 0).val = 1 := by sorry

end NUMINAMATH_CALUDE_top_number_after_folds_l1361_136187


namespace NUMINAMATH_CALUDE_unique_root_between_zero_and_e_l1361_136146

/-- The natural logarithm function -/
noncomputable def ln : ℝ → ℝ := Real.log

/-- The mathematical constant e -/
noncomputable def e : ℝ := Real.exp 1

theorem unique_root_between_zero_and_e (a : ℝ) (h1 : 0 < a) (h2 : a < e) :
  ∃! x : ℝ, x = ln (a * x) := by sorry

end NUMINAMATH_CALUDE_unique_root_between_zero_and_e_l1361_136146


namespace NUMINAMATH_CALUDE_katie_packages_l1361_136122

def cupcake_packages (initial_cupcakes : ℕ) (eaten_cupcakes : ℕ) (cupcakes_per_package : ℕ) : ℕ :=
  (initial_cupcakes - eaten_cupcakes) / cupcakes_per_package

theorem katie_packages :
  cupcake_packages 18 8 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_katie_packages_l1361_136122


namespace NUMINAMATH_CALUDE_households_with_bike_only_l1361_136101

/-- Proves that the number of households with only a bike is 35 -/
theorem households_with_bike_only
  (total : ℕ)
  (neither : ℕ)
  (both : ℕ)
  (with_car : ℕ)
  (h_total : total = 90)
  (h_neither : neither = 11)
  (h_both : both = 16)
  (h_with_car : with_car = 44) :
  total - neither - (with_car - both) - both = 35 :=
by sorry

end NUMINAMATH_CALUDE_households_with_bike_only_l1361_136101


namespace NUMINAMATH_CALUDE_sam_and_dan_balloons_l1361_136196

/-- The number of red balloons Sam and Dan have in total -/
def total_balloons (sam_initial : ℝ) (sam_given : ℝ) (dan : ℝ) : ℝ :=
  (sam_initial - sam_given) + dan

/-- Theorem stating the total number of red balloons Sam and Dan have -/
theorem sam_and_dan_balloons :
  total_balloons 46.0 10.0 16.0 = 52.0 := by
  sorry

end NUMINAMATH_CALUDE_sam_and_dan_balloons_l1361_136196


namespace NUMINAMATH_CALUDE_same_color_sock_pairs_l1361_136167

def num_white_socks : ℕ := 5
def num_black_socks : ℕ := 6
def num_red_socks : ℕ := 3
def num_green_socks : ℕ := 2

def choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

theorem same_color_sock_pairs :
  choose_2 num_white_socks +
  choose_2 num_black_socks +
  choose_2 num_red_socks +
  choose_2 num_green_socks = 29 := by sorry

end NUMINAMATH_CALUDE_same_color_sock_pairs_l1361_136167


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l1361_136123

/-- A function satisfying the given functional equation. -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x - f y) = (x - y)^2 * f (x + y)

/-- The theorem stating the possible forms of functions satisfying the equation. -/
theorem functional_equation_solutions :
  ∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f →
    (∀ x, f x = 0) ∨ (∀ x, f x = x^2) ∨ (∀ x, f x = -x^2) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l1361_136123


namespace NUMINAMATH_CALUDE_children_count_proof_l1361_136194

theorem children_count_proof :
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 150 ∧ n % 8 = 5 ∧ n % 10 = 7 ∧ n = 125 := by
  sorry

end NUMINAMATH_CALUDE_children_count_proof_l1361_136194


namespace NUMINAMATH_CALUDE_paint_for_smaller_statues_l1361_136140

-- Define the height of the original statue
def original_height : ℝ := 6

-- Define the height of the smaller statues
def small_height : ℝ := 2

-- Define the number of smaller statues
def num_statues : ℕ := 1080

-- Define the amount of paint needed for the original statue
def paint_for_original : ℝ := 1

-- Theorem statement
theorem paint_for_smaller_statues :
  (paint_for_original * (small_height / original_height)^2 * num_statues : ℝ) = 120 := by
  sorry

end NUMINAMATH_CALUDE_paint_for_smaller_statues_l1361_136140


namespace NUMINAMATH_CALUDE_unique_numbers_satisfying_condition_l1361_136108

theorem unique_numbers_satisfying_condition : ∃! (a b : ℕ), 
  100 ≤ a ∧ a < 1000 ∧ 
  1000 ≤ b ∧ b < 10000 ∧ 
  10000 * a + b = 7 * a * b ∧ 
  a + b = 1458 := by sorry

end NUMINAMATH_CALUDE_unique_numbers_satisfying_condition_l1361_136108


namespace NUMINAMATH_CALUDE_solve_cubic_equation_l1361_136158

theorem solve_cubic_equation :
  ∃ y : ℝ, (y - 5)^3 = (1/27)⁻¹ ∧ y = 8 := by sorry

end NUMINAMATH_CALUDE_solve_cubic_equation_l1361_136158


namespace NUMINAMATH_CALUDE_robin_albums_l1361_136120

theorem robin_albums (total_pictures : ℕ) (pictures_per_album : ℕ) (h1 : total_pictures = 40) (h2 : pictures_per_album = 8) : 
  total_pictures / pictures_per_album = 5 := by
  sorry

end NUMINAMATH_CALUDE_robin_albums_l1361_136120


namespace NUMINAMATH_CALUDE_collinear_points_sum_l1361_136110

/-- Three points in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Collinearity of three points -/
def collinear (p q r : Point3D) : Prop :=
  ∃ (t s : ℝ), q.x - p.x = t * (r.x - p.x) ∧ 
                q.y - p.y = t * (r.y - p.y) ∧
                q.z - p.z = t * (r.z - p.z) ∧
                q.x - p.x = s * (r.x - q.x) ∧
                q.y - p.y = s * (r.y - q.y) ∧
                q.z - p.z = s * (r.z - q.z)

theorem collinear_points_sum (a b : ℝ) :
  collinear (Point3D.mk 2 a b) (Point3D.mk a 3 b) (Point3D.mk a b 4) →
  a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l1361_136110


namespace NUMINAMATH_CALUDE_sum_xy_is_zero_l1361_136154

theorem sum_xy_is_zero (x y : ℝ) 
  (h : (x + Real.sqrt (1 + x^2)) * (y + Real.sqrt (1 + y^2)) = 1) : 
  x + y = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_xy_is_zero_l1361_136154


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_integers_l1361_136128

def is_all_ones (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 1

theorem infinitely_many_divisible_integers :
  ∀ k : ℕ, ∃ n : ℕ, 
    n > k ∧ 
    is_all_ones n ∧ 
    n % 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_integers_l1361_136128


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1361_136138

theorem solution_set_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  {x : ℝ | -b < 1/x ∧ 1/x < a} = {x : ℝ | x < -1/b ∨ x > 1/a} := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1361_136138


namespace NUMINAMATH_CALUDE_f_plus_one_nonnegative_min_a_value_l1361_136191

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * (Real.log x - 1)

-- Theorem 1: f(x) + 1 ≥ 0 for all x > 0
theorem f_plus_one_nonnegative : ∀ x > 0, f x + 1 ≥ 0 := by sorry

-- Theorem 2: The minimum value of a such that 4f'(x) ≤ a(x+1) - 8 for all x > 0 is 4
theorem min_a_value : 
  (∃ a : ℝ, ∀ x > 0, 4 * (Real.log x) ≤ a * (x + 1) - 8) ∧ 
  (∀ a < 4, ∃ x > 0, 4 * (Real.log x) > a * (x + 1) - 8) := by sorry

end NUMINAMATH_CALUDE_f_plus_one_nonnegative_min_a_value_l1361_136191


namespace NUMINAMATH_CALUDE_parallel_vectors_difference_magnitude_l1361_136179

theorem parallel_vectors_difference_magnitude :
  ∀ x : ℝ,
  let a : Fin 2 → ℝ := ![1, x]
  let b : Fin 2 → ℝ := ![2*x + 3, -x]
  (∃ (k : ℝ), a = k • b) →
  ‖a - b‖ = 2 ∨ ‖a - b‖ = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_difference_magnitude_l1361_136179


namespace NUMINAMATH_CALUDE_arrangement_theorem_l1361_136198

/-- The number of permutations of n distinct objects. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose k objects from n distinct objects. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to arrange 7 students with Student A not at either end. -/
def arrangement_count_1 : ℕ := 5 * permutations 6

/-- The number of ways to arrange 7 students with Student A not on the left end
    and Student B not on the right end. -/
def arrangement_count_2 : ℕ := permutations 6 + choose 5 1 * choose 5 1 * permutations 5

theorem arrangement_theorem :
  (arrangement_count_1 = 5 * permutations 6) ∧
  (arrangement_count_2 = permutations 6 + choose 5 1 * choose 5 1 * permutations 5) :=
by sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l1361_136198


namespace NUMINAMATH_CALUDE_smallest_n_square_fourth_power_l1361_136169

/-- The smallest positive integer n such that 5n is a perfect square and 3n is a perfect fourth power is 75. -/
theorem smallest_n_square_fourth_power : 
  (∃ (n : ℕ), n > 0 ∧ 
    (∃ (k : ℕ), 5 * n = k^2) ∧ 
    (∃ (m : ℕ), 3 * n = m^4)) ∧
  (∀ (n : ℕ), n > 0 ∧ 
    (∃ (k : ℕ), 5 * n = k^2) ∧ 
    (∃ (m : ℕ), 3 * n = m^4) → 
    n ≥ 75) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_square_fourth_power_l1361_136169


namespace NUMINAMATH_CALUDE_debby_tickets_spent_l1361_136112

theorem debby_tickets_spent (hat_tickets stuffed_animal_tickets yoyo_tickets : ℕ) 
  (h1 : hat_tickets = 2)
  (h2 : stuffed_animal_tickets = 10)
  (h3 : yoyo_tickets = 2) :
  hat_tickets + stuffed_animal_tickets + yoyo_tickets = 14 :=
by sorry

end NUMINAMATH_CALUDE_debby_tickets_spent_l1361_136112


namespace NUMINAMATH_CALUDE_range_of_f_l1361_136197

def f (x : ℝ) : ℝ := |x + 8| - |x - 3|

theorem range_of_f :
  Set.range f = Set.Icc (-11) 11 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1361_136197


namespace NUMINAMATH_CALUDE_potato_sack_problem_l1361_136159

theorem potato_sack_problem (original_potatoes : ℕ) : 
  original_potatoes - 69 - (2 * 69) - ((2 * 69) / 3) = 47 → 
  original_potatoes = 300 := by
sorry

end NUMINAMATH_CALUDE_potato_sack_problem_l1361_136159


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l1361_136174

theorem right_triangle_shorter_leg : ∀ a b c : ℕ,
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 39 →           -- Hypotenuse is 39 units
  a ≤ b →            -- a is the shorter leg
  a = 15 :=          -- The shorter leg is 15 units
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l1361_136174


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1361_136155

universe u

def U : Finset ℕ := {1, 2, 3, 4, 5}
def M : Finset ℕ := {2, 4}
def N : Finset ℕ := {3, 5}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1361_136155


namespace NUMINAMATH_CALUDE_find_x_l1361_136151

theorem find_x : ∃ x : ℝ, (3 * x + 5) / 5 = 13 ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l1361_136151


namespace NUMINAMATH_CALUDE_complex_exp_13pi_div_2_l1361_136133

theorem complex_exp_13pi_div_2 : Complex.exp (13 * π * Complex.I / 2) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_13pi_div_2_l1361_136133


namespace NUMINAMATH_CALUDE_solve_equations_l1361_136165

theorem solve_equations :
  (∀ x : ℝ, 4 * x^2 = 9 ↔ x = 3/2 ∨ x = -3/2) ∧
  (∀ x : ℝ, (1 - 2*x)^3 = 8 ↔ x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_solve_equations_l1361_136165


namespace NUMINAMATH_CALUDE_cube_root_of_four_fifth_power_l1361_136157

theorem cube_root_of_four_fifth_power : 
  (5^7 + 5^7 + 5^7 + 5^7 : ℝ)^(1/3) = 5^(7/3) * 4^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_of_four_fifth_power_l1361_136157


namespace NUMINAMATH_CALUDE_hamburger_count_l1361_136192

/-- The total number of hamburgers made for lunch -/
def total_hamburgers (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that the total number of hamburgers is the sum of initial and additional -/
theorem hamburger_count (initial : ℕ) (additional : ℕ) :
  total_hamburgers initial additional = initial + additional :=
by sorry

end NUMINAMATH_CALUDE_hamburger_count_l1361_136192


namespace NUMINAMATH_CALUDE_translator_selection_count_l1361_136150

/-- Represents the number of translators for each category -/
structure TranslatorCounts where
  total : Nat
  english : Nat
  japanese : Nat
  both : Nat

/-- Represents the required number of translators for each language -/
structure RequiredTranslators where
  english : Nat
  japanese : Nat

/-- Calculates the number of ways to select translators given the constraints -/
def countTranslatorSelections (counts : TranslatorCounts) (required : RequiredTranslators) : Nat :=
  sorry

/-- Theorem stating that there are 29 different ways to select the translators -/
theorem translator_selection_count :
  let counts : TranslatorCounts := ⟨8, 3, 3, 2⟩
  let required : RequiredTranslators := ⟨3, 2⟩
  countTranslatorSelections counts required = 29 :=
by sorry

end NUMINAMATH_CALUDE_translator_selection_count_l1361_136150


namespace NUMINAMATH_CALUDE_ice_cream_frozen_yoghurt_cost_difference_is_28_31_l1361_136119

/-- Calculates the difference between ice cream and frozen yoghurt costs --/
def ice_cream_frozen_yoghurt_cost_difference : ℝ :=
  let chocolate_ice_cream := 6 * 5 * (1 - 0.10)
  let vanilla_ice_cream := 4 * 4 * (1 - 0.07)
  let strawberry_frozen_yoghurt := 3 * 3 * (1 + 0.05)
  let mango_frozen_yoghurt := 2 * 2 * (1 + 0.03)
  let total_ice_cream := chocolate_ice_cream + vanilla_ice_cream
  let total_frozen_yoghurt := strawberry_frozen_yoghurt + mango_frozen_yoghurt
  total_ice_cream - total_frozen_yoghurt

/-- The difference between ice cream and frozen yoghurt costs is $28.31 --/
theorem ice_cream_frozen_yoghurt_cost_difference_is_28_31 :
  ice_cream_frozen_yoghurt_cost_difference = 28.31 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_frozen_yoghurt_cost_difference_is_28_31_l1361_136119


namespace NUMINAMATH_CALUDE_least_crayons_l1361_136103

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

theorem least_crayons (n : ℕ) : 
  (is_divisible_by n 3 ∧ 
   is_divisible_by n 4 ∧ 
   is_divisible_by n 5 ∧ 
   is_divisible_by n 7 ∧ 
   is_divisible_by n 8) →
  (∀ m : ℕ, m < n → 
    ¬(is_divisible_by m 3 ∧ 
      is_divisible_by m 4 ∧ 
      is_divisible_by m 5 ∧ 
      is_divisible_by m 7 ∧ 
      is_divisible_by m 8)) →
  n = 840 := by
sorry

end NUMINAMATH_CALUDE_least_crayons_l1361_136103


namespace NUMINAMATH_CALUDE_simplify_expression_l1361_136186

theorem simplify_expression (x : ℝ) : (3*x - 4)*(x + 8) - (x + 6)*(3*x - 2) = 4*x - 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1361_136186


namespace NUMINAMATH_CALUDE_warehouse_theorem_l1361_136152

def warehouse_problem (second_floor_space : ℝ) (boxes_space : ℝ) : Prop :=
  let first_floor_space := 2 * second_floor_space
  let total_space := first_floor_space + second_floor_space
  let available_space := total_space - boxes_space
  (boxes_space = 5000) ∧
  (boxes_space = second_floor_space / 4) ∧
  (available_space = 55000)

theorem warehouse_theorem :
  ∃ (second_floor_space : ℝ), warehouse_problem second_floor_space 5000 :=
sorry

end NUMINAMATH_CALUDE_warehouse_theorem_l1361_136152


namespace NUMINAMATH_CALUDE_water_needed_l1361_136178

/-- Represents the recipe for lemonade tea --/
structure LemonadeTea where
  lemonJuice : ℝ
  sugar : ℝ
  water : ℝ
  tea : ℝ

/-- Checks if the recipe satisfies the given conditions --/
def isValidRecipe (recipe : LemonadeTea) : Prop :=
  recipe.water = 3 * recipe.sugar ∧
  recipe.sugar = 1.5 * recipe.lemonJuice ∧
  recipe.tea = (recipe.water + recipe.sugar + recipe.lemonJuice) / 6 ∧
  recipe.lemonJuice = 4

/-- Theorem stating that a valid recipe requires 18 cups of water --/
theorem water_needed (recipe : LemonadeTea) 
  (h : isValidRecipe recipe) : recipe.water = 18 := by
  sorry


end NUMINAMATH_CALUDE_water_needed_l1361_136178


namespace NUMINAMATH_CALUDE_ball_distribution_theorem_l1361_136115

/-- Represents the number of ways to distribute balls into boxes -/
def distribute_balls (total_balls : ℕ) (num_boxes : ℕ) (min_balls : ℕ → ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are 15 ways to distribute 10 balls into 3 boxes -/
theorem ball_distribution_theorem :
  distribute_balls 10 3 (fun i => i) = 15 := by sorry

end NUMINAMATH_CALUDE_ball_distribution_theorem_l1361_136115


namespace NUMINAMATH_CALUDE_function_properties_l1361_136104

noncomputable def f (a b x : ℝ) : ℝ := Real.log (x / 2) - a * x + b / x

theorem function_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x > 0, f a b x + f a b (4 / x) = 0) →
  (b = 4 * a ∧ (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ f a b x₁ = 0 ∧ f a b x₂ = 0 ∧ f a b x₃ = 0) ↔ 0 < a ∧ a < 1/4) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1361_136104


namespace NUMINAMATH_CALUDE_marcus_cookies_count_l1361_136136

/-- The number of peanut butter cookies Marcus brought to the bake sale -/
def marcus_peanut_butter_cookies : ℕ := 30

/-- The number of peanut butter cookies Jenny brought to the bake sale -/
def jenny_peanut_butter_cookies : ℕ := 40

/-- The total number of non-peanut butter cookies at the bake sale -/
def total_non_peanut_butter_cookies : ℕ := 70

/-- The probability of picking a peanut butter cookie -/
def peanut_butter_probability : ℚ := 1/2

theorem marcus_cookies_count :
  marcus_peanut_butter_cookies = 30 ∧
  jenny_peanut_butter_cookies + marcus_peanut_butter_cookies = total_non_peanut_butter_cookies ∧
  (jenny_peanut_butter_cookies + marcus_peanut_butter_cookies : ℚ) /
    (jenny_peanut_butter_cookies + marcus_peanut_butter_cookies + total_non_peanut_butter_cookies) = peanut_butter_probability :=
by sorry

end NUMINAMATH_CALUDE_marcus_cookies_count_l1361_136136


namespace NUMINAMATH_CALUDE_units_digit_of_product_first_four_composites_l1361_136164

def first_four_composites : List Nat := [4, 6, 8, 9]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (· * ·) 1

def units_digit (n : Nat) : Nat :=
  n % 10

theorem units_digit_of_product_first_four_composites :
  units_digit (product_of_list first_four_composites) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_first_four_composites_l1361_136164


namespace NUMINAMATH_CALUDE_dave_has_least_money_l1361_136161

-- Define the set of people
inductive Person : Type
  | Alice : Person
  | Ben : Person
  | Carol : Person
  | Dave : Person
  | Ethan : Person

-- Define a function to represent the amount of money each person has
variable (money : Person → ℕ)

-- Define the conditions
axiom different_amounts : ∀ (p q : Person), p ≠ q → money p ≠ money q
axiom ethan_less_than_alice : money Person.Ethan < money Person.Alice
axiom ben_more_than_dave : money Person.Dave < money Person.Ben
axiom carol_more_than_dave : money Person.Dave < money Person.Carol
axiom alice_between_dave_and_ben : money Person.Dave < money Person.Alice ∧ money Person.Alice < money Person.Ben
axiom carol_between_ethan_and_alice : money Person.Ethan < money Person.Carol ∧ money Person.Carol < money Person.Alice

-- Theorem to prove
theorem dave_has_least_money :
  ∀ (p : Person), p ≠ Person.Dave → money Person.Dave < money p :=
sorry

end NUMINAMATH_CALUDE_dave_has_least_money_l1361_136161


namespace NUMINAMATH_CALUDE_rope_problem_l1361_136105

theorem rope_problem (x : ℝ) :
  (8 : ℝ)^2 + (x - 3)^2 = x^2 :=
by sorry

end NUMINAMATH_CALUDE_rope_problem_l1361_136105


namespace NUMINAMATH_CALUDE_beef_cabbage_cost_comparison_l1361_136131

/-- Represents the cost calculation for beef and spicy cabbage orders --/
theorem beef_cabbage_cost_comparison (a : ℝ) (h : a > 50) :
  (4500 + 27 * a) ≤ (4400 + 30 * a) := by
  sorry

#check beef_cabbage_cost_comparison

end NUMINAMATH_CALUDE_beef_cabbage_cost_comparison_l1361_136131


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1361_136116

/-- The quadratic equation x^2 + (2k+1)x + k^2 + 1 = 0 has two distinct real roots -/
def has_distinct_roots (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + (2*k+1)*x₁ + k^2 + 1 = 0 ∧ x₂^2 + (2*k+1)*x₂ + k^2 + 1 = 0

/-- The product of the roots of the quadratic equation is 5 -/
def roots_product_is_5 (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ * x₂ = 5 ∧ x₁^2 + (2*k+1)*x₁ + k^2 + 1 = 0 ∧ x₂^2 + (2*k+1)*x₂ + k^2 + 1 = 0

theorem quadratic_equation_properties :
  (∀ k : ℝ, has_distinct_roots k ↔ k > 3/4) ∧
  (∀ k : ℝ, roots_product_is_5 k → k = 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1361_136116


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_l1361_136172

theorem complex_sum_magnitude (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1) 
  (h4 : a^2 / (b*c) + b^2 / (a*c) + c^2 / (a*b) = 1) : 
  Complex.abs (a + b + c) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_l1361_136172


namespace NUMINAMATH_CALUDE_chord_length_perpendicular_chord_m_l1361_136149

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equations
def line1_equation (x y : ℝ) : Prop :=
  x + y - 1 = 0

def line2_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Part 1: Chord length
theorem chord_length : 
  ∀ x y : ℝ, circle_equation x y 1 ∧ line1_equation x y → 
  ∃ chord_length : ℝ, chord_length = 2 * Real.sqrt 2 :=
sorry

-- Part 2: Value of m
theorem perpendicular_chord_m :
  ∃ m : ℝ, ∀ x1 y1 x2 y2 : ℝ,
    circle_equation x1 y1 m ∧ circle_equation x2 y2 m ∧
    line2_equation x1 y1 ∧ line2_equation x2 y2 ∧
    x1 * x2 + y1 * y2 = 0 →
    m = 8/5 :=
sorry

end NUMINAMATH_CALUDE_chord_length_perpendicular_chord_m_l1361_136149


namespace NUMINAMATH_CALUDE_area_ratio_square_to_rectangle_l1361_136114

/-- The ratio of the area of a square with side length 48 cm to the area of a rectangle with dimensions 56 cm by 63 cm is 2/3. -/
theorem area_ratio_square_to_rectangle : 
  let square_side : ℝ := 48
  let rect_width : ℝ := 56
  let rect_height : ℝ := 63
  let square_area := square_side ^ 2
  let rect_area := rect_width * rect_height
  square_area / rect_area = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_area_ratio_square_to_rectangle_l1361_136114


namespace NUMINAMATH_CALUDE_larger_integer_value_l1361_136118

theorem larger_integer_value (a b : ℕ+) (h1 : (a : ℚ) / (b : ℚ) = 7 / 3) (h2 : (a : ℕ) * b = 189) : a = 21 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_value_l1361_136118


namespace NUMINAMATH_CALUDE_min_value_sum_l1361_136130

theorem min_value_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 1) : 
  ∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 1/a + 1/b + 1/c = 1 → x + 4*y + 9*z ≤ a + 4*b + 9*c :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_l1361_136130
