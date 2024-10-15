import Mathlib

namespace NUMINAMATH_CALUDE_disjoint_subsets_remainder_l362_36213

def S : Finset Nat := Finset.range 12

def count_disjoint_subsets (S : Finset Nat) : Nat :=
  (3^S.card - 2 * 2^S.card + 1) / 2

theorem disjoint_subsets_remainder (S : Finset Nat) (h : S = Finset.range 12) :
  count_disjoint_subsets S % 1000 = 625 := by
  sorry

end NUMINAMATH_CALUDE_disjoint_subsets_remainder_l362_36213


namespace NUMINAMATH_CALUDE_binomial_square_coeff_l362_36232

/-- If ax^2 + 8x + 16 is the square of a binomial, then a = 1 -/
theorem binomial_square_coeff (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 8 * x + 16 = (r * x + s)^2) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_coeff_l362_36232


namespace NUMINAMATH_CALUDE_paths_through_B_and_C_l362_36299

/-- Represents a point on the square grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Calculates the number of paths between two points on a square grid -/
def num_paths (start finish : GridPoint) : ℕ :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

/-- The points on the grid -/
def A : GridPoint := ⟨0, 0⟩
def B : GridPoint := ⟨2, 3⟩
def C : GridPoint := ⟨6, 4⟩
def D : GridPoint := ⟨9, 6⟩

/-- The theorem to be proved -/
theorem paths_through_B_and_C : 
  num_paths A B * num_paths B C * num_paths C D = 500 := by
  sorry

end NUMINAMATH_CALUDE_paths_through_B_and_C_l362_36299


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l362_36230

theorem negation_of_universal_statement :
  (¬∀ x : ℝ, x > 2 → x^2 - 2*x > 0) ↔ (∃ x : ℝ, x > 2 ∧ x^2 - 2*x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l362_36230


namespace NUMINAMATH_CALUDE_square_area_26_l362_36284

/-- The area of a square with vertices at (0, 0), (-5, -1), (-4, -6), and (1, -5) is 26 square units. -/
theorem square_area_26 : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (-5, -1)
  let C : ℝ × ℝ := (-4, -6)
  let D : ℝ × ℝ := (1, -5)
  let square_area := (B.1 - A.1)^2 + (B.2 - A.2)^2
  square_area = 26 := by
  sorry


end NUMINAMATH_CALUDE_square_area_26_l362_36284


namespace NUMINAMATH_CALUDE_regular_polygon_120_degrees_l362_36225

/-- A regular polygon with interior angles of 120° has 6 sides -/
theorem regular_polygon_120_degrees (n : ℕ) : 
  (n ≥ 3) → 
  ((n - 2) * 180 / n = 120) → 
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_120_degrees_l362_36225


namespace NUMINAMATH_CALUDE_odd_sum_of_squares_implies_odd_sum_l362_36247

theorem odd_sum_of_squares_implies_odd_sum (n m : ℤ) :
  Odd (n^2 + m^2) → Odd (n + m) := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_of_squares_implies_odd_sum_l362_36247


namespace NUMINAMATH_CALUDE_basketball_tryouts_l362_36254

/-- Given the number of girls and boys trying out for a basketball team,
    and the number of students called back, calculate the number of
    students who didn't make the cut. -/
theorem basketball_tryouts (girls boys called_back : ℕ) : 
  girls = 39 → boys = 4 → called_back = 26 → 
  girls + boys - called_back = 17 := by
  sorry

end NUMINAMATH_CALUDE_basketball_tryouts_l362_36254


namespace NUMINAMATH_CALUDE_christmas_discount_problem_l362_36266

/-- Represents the Christmas discount problem for an air-conditioning unit. -/
theorem christmas_discount_problem (original_price : ℝ) (price_increase : ℝ) (final_price : ℝ) 
  (h1 : original_price = 470)
  (h2 : price_increase = 0.12)
  (h3 : final_price = 442.18) :
  ∃ (x : ℝ), 
    x ≥ 0 ∧ 
    x ≤ 100 ∧ 
    abs (x - 1.11) < 0.01 ∧
    original_price * (1 - x / 100) * (1 + price_increase) = final_price :=
sorry

end NUMINAMATH_CALUDE_christmas_discount_problem_l362_36266


namespace NUMINAMATH_CALUDE_denise_age_l362_36219

theorem denise_age (amanda beth carlos denise : ℕ) 
  (h1 : amanda = carlos - 4)
  (h2 : carlos = beth + 5)
  (h3 : denise = beth + 2)
  (h4 : amanda = 16) : 
  denise = 17 := by
  sorry

end NUMINAMATH_CALUDE_denise_age_l362_36219


namespace NUMINAMATH_CALUDE_f_minus_three_equals_six_l362_36222

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_minus_three_equals_six 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period f 4) 
  (h_sum : f 1 + f 2 + f 3 + f 4 + f 5 = 6) : 
  f (-3) = 6 := by
sorry

end NUMINAMATH_CALUDE_f_minus_three_equals_six_l362_36222


namespace NUMINAMATH_CALUDE_gumball_solution_l362_36251

/-- Represents the gumball distribution problem --/
def gumball_problem (total : ℕ) (todd : ℕ) (alisha : ℕ) (bobby : ℕ) : Prop :=
  total = 45 ∧
  todd = 4 ∧
  alisha = 2 * todd ∧
  bobby = 4 * alisha - 5 ∧
  total - (todd + alisha + bobby) = 6

/-- Theorem stating that the gumball problem has a solution --/
theorem gumball_solution : ∃ (total todd alisha bobby : ℕ), gumball_problem total todd alisha bobby :=
sorry

end NUMINAMATH_CALUDE_gumball_solution_l362_36251


namespace NUMINAMATH_CALUDE_min_sum_of_squares_with_diff_l362_36275

theorem min_sum_of_squares_with_diff (x y : ℤ) (h : x^2 - y^2 = 165) :
  ∃ (a b : ℤ), a^2 - b^2 = 165 ∧ a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = 173 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_with_diff_l362_36275


namespace NUMINAMATH_CALUDE_prime_square_mod_360_l362_36262

theorem prime_square_mod_360 (p : Nat) (h_prime : Prime p) (h_gt_5 : p > 5) :
  (p^2 : Nat) % 360 = 1 ∨ (p^2 : Nat) % 360 = 289 := by
  sorry

#check prime_square_mod_360

end NUMINAMATH_CALUDE_prime_square_mod_360_l362_36262


namespace NUMINAMATH_CALUDE_completed_square_form_l362_36215

theorem completed_square_form (x : ℝ) :
  x^2 - 6*x - 5 = 0 ↔ (x - 3)^2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_completed_square_form_l362_36215


namespace NUMINAMATH_CALUDE_max_a_value_l362_36211

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 4

-- Define the property that f(x) ≤ 6 for all x in (0,2]
def property (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x ≤ 2 → f a x ≤ 6

-- State the theorem
theorem max_a_value :
  (∃ a : ℝ, property a) →
  (∃ a_max : ℝ, property a_max ∧ ∀ a : ℝ, property a → a ≤ a_max) →
  (∀ a_max : ℝ, (property a_max ∧ ∀ a : ℝ, property a → a ≤ a_max) → a_max = -1) :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l362_36211


namespace NUMINAMATH_CALUDE_rectangle_area_is_twelve_l362_36273

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle defined by its four vertices -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The area of a rectangle -/
def rectangleArea (rect : Rectangle) : ℝ :=
  (rect.B.x - rect.A.x) * (rect.C.y - rect.B.y)

theorem rectangle_area_is_twelve :
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨3, 0⟩
  let C : Point := ⟨3, 4⟩
  let D : Point := ⟨0, 4⟩
  let rect : Rectangle := ⟨A, B, C, D⟩
  rectangleArea rect = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_twelve_l362_36273


namespace NUMINAMATH_CALUDE_problem_solution_l362_36297

-- Define the properties of functions f and g
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

def is_inverse_proportion (g : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → g x = k / x

-- Define the conditions given in the problem
def problem_conditions (f g : ℝ → ℝ) : Prop :=
  is_direct_proportion f ∧ is_inverse_proportion g ∧ f 1 = 1 ∧ g 1 = 2

-- Define what it means for a function to be odd
def is_odd_function (h : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, h (-x) = -h x

-- Theorem statement
theorem problem_solution (f g : ℝ → ℝ) (h : problem_conditions f g) :
  (∀ x : ℝ, f x = x) ∧
  (∀ x : ℝ, x ≠ 0 → g x = 2 / x) ∧
  is_odd_function (λ x => f x + g x) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l362_36297


namespace NUMINAMATH_CALUDE_polynomial_remainder_l362_36260

theorem polynomial_remainder (x : ℝ) : (x^11 + 2) % (x - 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l362_36260


namespace NUMINAMATH_CALUDE_complex_expression_equals_9980_l362_36285

theorem complex_expression_equals_9980 : 
  3 * 995 + 4 * 996 + 5 * 997 + 6 * 998 + 7 * 999 - 4985 * 3 = 9980 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_9980_l362_36285


namespace NUMINAMATH_CALUDE_miae_closer_estimate_l362_36258

def bowl_volume : ℝ := 1000  -- in milliliters
def miae_estimate : ℝ := 1100  -- in milliliters
def hyori_estimate : ℝ := 850  -- in milliliters

theorem miae_closer_estimate :
  |miae_estimate - bowl_volume| < |hyori_estimate - bowl_volume| := by
  sorry

end NUMINAMATH_CALUDE_miae_closer_estimate_l362_36258


namespace NUMINAMATH_CALUDE_largest_two_twos_l362_36279

def two_twos_operation : ℕ → Prop :=
  λ n => ∃ (op : ℕ → ℕ → ℕ), n = op 2 2 ∨ n = 22

theorem largest_two_twos :
  ∀ n : ℕ, two_twos_operation n → n ≤ 22 :=
by
  sorry

#check largest_two_twos

end NUMINAMATH_CALUDE_largest_two_twos_l362_36279


namespace NUMINAMATH_CALUDE_task_completion_probability_l362_36291

theorem task_completion_probability (p1 p2 : ℚ) (h1 : p1 = 3/8) (h2 : p2 = 3/5) :
  p1 * (1 - p2) = 3/20 := by
  sorry

end NUMINAMATH_CALUDE_task_completion_probability_l362_36291


namespace NUMINAMATH_CALUDE_revenue_calculation_l362_36293

-- Define the initial and final counts of phones
def samsung_start : ℕ := 14
def samsung_end : ℕ := 10
def iphone_start : ℕ := 8
def iphone_end : ℕ := 5

-- Define the number of damaged phones
def samsung_damaged : ℕ := 2
def iphone_damaged : ℕ := 1

-- Define the retail prices
def samsung_price : ℚ := 800
def iphone_price : ℚ := 1000

-- Define the discount and tax rates
def samsung_discount : ℚ := 0.10
def samsung_tax : ℚ := 0.12
def iphone_discount : ℚ := 0.15
def iphone_tax : ℚ := 0.10

-- Calculate the number of phones sold
def samsung_sold : ℕ := samsung_start - samsung_end - samsung_damaged
def iphone_sold : ℕ := iphone_start - iphone_end - iphone_damaged

-- Calculate the final price for each phone type after discount and tax
def samsung_final_price : ℚ := samsung_price * (1 - samsung_discount) * (1 + samsung_tax)
def iphone_final_price : ℚ := iphone_price * (1 - iphone_discount) * (1 + iphone_tax)

-- Calculate the total revenue
def total_revenue : ℚ := samsung_final_price * samsung_sold + iphone_final_price * iphone_sold

-- Theorem to prove
theorem revenue_calculation : total_revenue = 3482.80 := by
  sorry

end NUMINAMATH_CALUDE_revenue_calculation_l362_36293


namespace NUMINAMATH_CALUDE_sliding_window_is_only_translation_l362_36276

/-- Represents a type of movement --/
inductive Movement
  | PingPongBall
  | SlidingWindow
  | Kite
  | Basketball

/-- Predicate to check if a movement is a translation --/
def isTranslation (m : Movement) : Prop :=
  match m with
  | Movement.SlidingWindow => True
  | _ => False

/-- Theorem stating that only the sliding window movement is a translation --/
theorem sliding_window_is_only_translation :
  ∀ m : Movement, isTranslation m ↔ m = Movement.SlidingWindow :=
sorry

#check sliding_window_is_only_translation

end NUMINAMATH_CALUDE_sliding_window_is_only_translation_l362_36276


namespace NUMINAMATH_CALUDE_parallel_linear_functions_min_value_l362_36231

/-- Two linear functions with parallel graphs not parallel to coordinate axes -/
structure ParallelLinearFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  parallel : ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x = a * x + b) ∧ (∀ x, g x = a * x + c)

/-- The minimum value of a quadratic function -/
def quadratic_min (h : ℝ → ℝ) : ℝ := sorry

theorem parallel_linear_functions_min_value 
  (funcs : ParallelLinearFunctions)
  (h_min : quadratic_min (λ x => (funcs.f x)^2 + 5 * funcs.g x) = -17) :
  quadratic_min (λ x => (funcs.g x)^2 + 5 * funcs.f x) = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_linear_functions_min_value_l362_36231


namespace NUMINAMATH_CALUDE_ab_nonpositive_l362_36201

theorem ab_nonpositive (a b : ℚ) (ha : |a| = a) (hb : |b| = -b) : a * b ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_nonpositive_l362_36201


namespace NUMINAMATH_CALUDE_fractional_equation_m_range_l362_36229

theorem fractional_equation_m_range :
  ∀ m x : ℝ,
  (x / (x - 3) = 2 + m / (x - 3)) →
  (x > 0) →
  (m < 6 ∧ m ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_m_range_l362_36229


namespace NUMINAMATH_CALUDE_russian_football_championship_l362_36278

/-- Represents a football championship. -/
structure Championship where
  teams : ℕ
  matches_per_pair : ℕ

/-- Calculate the number of matches a single team plays. -/
def matches_per_team (c : Championship) : ℕ :=
  (c.teams - 1) * c.matches_per_pair

/-- Calculate the total number of matches in the championship. -/
def total_matches (c : Championship) : ℕ :=
  (c.teams * matches_per_team c) / 2

theorem russian_football_championship 
  (c : Championship) 
  (h1 : c.teams = 16) 
  (h2 : c.matches_per_pair = 2) : 
  matches_per_team c = 30 ∧ total_matches c = 240 := by
  sorry

#eval matches_per_team ⟨16, 2⟩
#eval total_matches ⟨16, 2⟩

end NUMINAMATH_CALUDE_russian_football_championship_l362_36278


namespace NUMINAMATH_CALUDE_x_plus_y_values_l362_36204

theorem x_plus_y_values (x y : ℝ) (hx : x = y * (3 - y)^2) (hy : y = x * (3 - x)^2) :
  x + y ∈ ({0, 3, 4, 5, 8} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l362_36204


namespace NUMINAMATH_CALUDE_single_interval_condition_l362_36239

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

/-- Condition for single interval solution of [x]^2 + k[x] + l = 0 -/
theorem single_interval_condition (k l : ℤ) : 
  (∃ (a b : ℝ), ∀ x, (floor x)^2 + k * (floor x) + l = 0 ↔ a ≤ x ∧ x < b) ↔ 
  l = floor ((k^2 : ℝ) / 4) :=
sorry

end NUMINAMATH_CALUDE_single_interval_condition_l362_36239


namespace NUMINAMATH_CALUDE_direct_proportion_iff_m_eq_neg_one_l362_36289

/-- A function f(x) is a direct proportion function if there exists a non-zero constant k such that f(x) = kx for all x. -/
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- The function y = (m-1)x + m^2 - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x + m^2 - 1

theorem direct_proportion_iff_m_eq_neg_one (m : ℝ) :
  is_direct_proportion (f m) ↔ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_iff_m_eq_neg_one_l362_36289


namespace NUMINAMATH_CALUDE_investment_calculation_l362_36272

/-- Given two investors p and q, where p invested 52000 and the profit is divided in the ratio 4:5,
    prove that q invested 65000. -/
theorem investment_calculation (p q : ℕ) : 
  p = 52000 → 
  (4 : ℚ) / 5 = p / q →
  q = 65000 := by
sorry

end NUMINAMATH_CALUDE_investment_calculation_l362_36272


namespace NUMINAMATH_CALUDE_chord_intersections_count_l362_36205

/-- The number of intersection points of chords on a circle -/
def chord_intersections (n : ℕ) : ℕ :=
  Nat.choose n 4

/-- Theorem: The number of intersection points of chords drawn between n vertices 
    on a circle, excluding the vertices themselves, is equal to binom(n, 4), 
    given that no three chords are concurrent except at a vertex. -/
theorem chord_intersections_count (n : ℕ) (h : n ≥ 4) :
  chord_intersections n = Nat.choose n 4 := by
  sorry

end NUMINAMATH_CALUDE_chord_intersections_count_l362_36205


namespace NUMINAMATH_CALUDE_ethans_net_income_l362_36261

/-- Calculates Ethan's net income after deductions for a 5-week period -/
def calculate_net_income (hourly_wage : ℚ) (hours_per_day : ℕ) (days_per_week : ℕ) 
  (total_weeks : ℕ) (tax_rate : ℚ) (health_insurance_per_week : ℚ) (retirement_rate : ℚ) : ℚ :=
  let gross_income := hourly_wage * hours_per_day * days_per_week * total_weeks
  let income_tax := tax_rate * gross_income
  let health_insurance := health_insurance_per_week * total_weeks
  let retirement_contribution := retirement_rate * gross_income
  let total_deductions := income_tax + health_insurance + retirement_contribution
  gross_income - total_deductions

/-- Theorem stating that Ethan's net income after deductions for a 5-week period is $2447 -/
theorem ethans_net_income : 
  calculate_net_income 18 8 5 5 (15/100) 65 (8/100) = 2447 := by
  sorry

end NUMINAMATH_CALUDE_ethans_net_income_l362_36261


namespace NUMINAMATH_CALUDE_percentage_passed_all_topics_percentage_passed_all_topics_proof_l362_36298

/-- The percentage of students who passed in all topics in a practice paper -/
theorem percentage_passed_all_topics : ℝ :=
  let total_students : ℕ := 2500
  let passed_three_topics : ℕ := 500
  let percent_no_pass : ℝ := 10
  let percent_one_topic : ℝ := 20
  let percent_two_topics : ℝ := 25
  let percent_four_topics : ℝ := 24
  let percent_three_topics : ℝ := (passed_three_topics : ℝ) / (total_students : ℝ) * 100

  1 -- This is the percentage we need to prove

theorem percentage_passed_all_topics_proof : percentage_passed_all_topics = 1 := by
  sorry

end NUMINAMATH_CALUDE_percentage_passed_all_topics_percentage_passed_all_topics_proof_l362_36298


namespace NUMINAMATH_CALUDE_paper_towel_savings_l362_36238

theorem paper_towel_savings (package_price : ℚ) (individual_price : ℚ) (rolls : ℕ) : 
  package_price = 9 → individual_price = 1 → rolls = 12 →
  (1 - package_price / (individual_price * rolls)) * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_paper_towel_savings_l362_36238


namespace NUMINAMATH_CALUDE_solve_system_l362_36267

theorem solve_system (x y : ℝ) (h1 : 2 * x + y = 4) (h2 : (x + y) / 3 = 1) :
  x + 2 * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l362_36267


namespace NUMINAMATH_CALUDE_solve_equation_l362_36255

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((3 / x) + 3) = 4 / 3 → x = -27 / 11 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l362_36255


namespace NUMINAMATH_CALUDE_a2_value_l362_36288

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) :=
  b^2 = a * c

theorem a2_value (a : ℕ → ℝ) :
  arithmetic_sequence a 2 →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_a2_value_l362_36288


namespace NUMINAMATH_CALUDE_alpha_necessary_not_sufficient_for_beta_l362_36233

theorem alpha_necessary_not_sufficient_for_beta :
  (∀ a b : ℝ, b ≠ 0 → (a / b ≥ 1 → b * (b - a) ≤ 0)) ∧
  (∃ a b : ℝ, b ≠ 0 ∧ b * (b - a) ≤ 0 ∧ a / b < 1) := by
sorry

end NUMINAMATH_CALUDE_alpha_necessary_not_sufficient_for_beta_l362_36233


namespace NUMINAMATH_CALUDE_martin_failed_by_200_l362_36244

/-- Calculates the number of marks by which a student failed an exam -/
def marksFailedBy (maxMarks passingPercentage studentScore : ℕ) : ℕ :=
  let passingMark := (passingPercentage * maxMarks) / 100
  passingMark - studentScore

/-- Proves that Martin failed the exam by 200 marks -/
theorem martin_failed_by_200 :
  let maxMarks : ℕ := 500
  let passingPercentage : ℕ := 80
  let martinScore : ℕ := 200
  let passingMark := (passingPercentage * maxMarks) / 100
  martinScore < passingMark →
  marksFailedBy maxMarks passingPercentage martinScore = 200 := by
  sorry

end NUMINAMATH_CALUDE_martin_failed_by_200_l362_36244


namespace NUMINAMATH_CALUDE_probability_of_integer_occurrence_l362_36241

theorem probability_of_integer_occurrence (a b : ℤ) (h : a ≤ b) :
  let range := b - a + 1
  (∀ k : ℤ, a ≤ k ∧ k ≤ b → (1 : ℚ) / range = (1 : ℚ) / range) :=
by sorry

end NUMINAMATH_CALUDE_probability_of_integer_occurrence_l362_36241


namespace NUMINAMATH_CALUDE_janice_homework_time_l362_36296

/-- Represents the time (in minutes) it takes Janice to complete various tasks before watching a movie -/
structure JanicesTasks where
  total_time : ℝ
  homework_time : ℝ
  cleaning_time : ℝ
  dog_walking_time : ℝ
  trash_time : ℝ
  remaining_time : ℝ

/-- The theorem stating that Janice's homework time is 30 minutes given the conditions -/
theorem janice_homework_time (tasks : JanicesTasks) :
  tasks.total_time = 120 ∧
  tasks.cleaning_time = tasks.homework_time / 2 ∧
  tasks.dog_walking_time = tasks.homework_time + 5 ∧
  tasks.trash_time = tasks.homework_time / 6 ∧
  tasks.remaining_time = 35 ∧
  tasks.total_time = tasks.homework_time + tasks.cleaning_time + tasks.dog_walking_time + tasks.trash_time + tasks.remaining_time
  →
  tasks.homework_time = 30 :=
by sorry

end NUMINAMATH_CALUDE_janice_homework_time_l362_36296


namespace NUMINAMATH_CALUDE_a_greater_than_b_greater_than_one_l362_36227

theorem a_greater_than_b_greater_than_one
  (n : ℕ) (a b : ℝ) 
  (h_n : n ≥ 2)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_a_eq : a^n = a + 1)
  (h_b_eq : b^(2*n) = b + 3*a) :
  a > b ∧ b > 1 := by
sorry

end NUMINAMATH_CALUDE_a_greater_than_b_greater_than_one_l362_36227


namespace NUMINAMATH_CALUDE_complex_power_equality_l362_36250

theorem complex_power_equality : (((1 + Complex.I) / (1 - Complex.I)) ^ 2016 = 1) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_equality_l362_36250


namespace NUMINAMATH_CALUDE_tv_sales_after_three_years_l362_36263

def initial_sales : ℕ := 327
def yearly_increase : ℕ := 50
def years : ℕ := 3

theorem tv_sales_after_three_years :
  initial_sales + years * yearly_increase = 477 :=
by sorry

end NUMINAMATH_CALUDE_tv_sales_after_three_years_l362_36263


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l362_36265

/-- An isosceles triangle with side lengths 4 and 8 has a perimeter of 20. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 8 → b = 8 → c = 4 →
  (a = b ∨ a = c ∨ b = c) →  -- isosceles condition
  a + b + c = 20 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l362_36265


namespace NUMINAMATH_CALUDE_fence_pole_count_l362_36237

/-- Calculates the number of fence poles needed for a path with a bridge --/
def fence_poles (total_length : ℕ) (bridge_length : ℕ) (pole_spacing : ℕ) : ℕ :=
  2 * ((total_length - bridge_length) / pole_spacing)

/-- Theorem statement for the fence pole problem --/
theorem fence_pole_count : 
  fence_poles 900 42 6 = 286 := by
  sorry

end NUMINAMATH_CALUDE_fence_pole_count_l362_36237


namespace NUMINAMATH_CALUDE_sequence_inequality_l362_36280

theorem sequence_inequality (n : ℕ) (a : ℕ → ℝ) 
  (h0 : a 0 = 1/2)
  (h1 : ∀ k, k < n → a (k + 1) = a k + (1/n) * (a k)^2) :
  1 - 1/n < a n ∧ a n < 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l362_36280


namespace NUMINAMATH_CALUDE_remainder_nine_eight_mod_five_l362_36277

theorem remainder_nine_eight_mod_five : 9^8 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_nine_eight_mod_five_l362_36277


namespace NUMINAMATH_CALUDE_not_diff_of_squares_2022_l362_36268

theorem not_diff_of_squares_2022 : ∀ a b : ℤ, a^2 - b^2 ≠ 2022 := by
  sorry

end NUMINAMATH_CALUDE_not_diff_of_squares_2022_l362_36268


namespace NUMINAMATH_CALUDE_problems_per_worksheet_l362_36242

/-- Given a set of worksheets with some graded and some problems left to grade,
    calculate the number of problems per worksheet. -/
theorem problems_per_worksheet
  (total_worksheets : ℕ)
  (graded_worksheets : ℕ)
  (remaining_problems : ℕ)
  (h1 : total_worksheets = 16)
  (h2 : graded_worksheets = 8)
  (h3 : remaining_problems = 32)
  : (remaining_problems / (total_worksheets - graded_worksheets) : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_problems_per_worksheet_l362_36242


namespace NUMINAMATH_CALUDE_y₁_not_in_third_quadrant_l362_36235

-- Define the linear functions
def y₁ (x : ℝ) (b : ℝ) : ℝ := -x + b
def y₂ (x : ℝ) : ℝ := -x

-- State the theorem
theorem y₁_not_in_third_quadrant :
  ∃ b : ℝ, (∀ x : ℝ, y₁ x b = y₂ x + 2) →
  ∀ x y : ℝ, y = y₁ x b → (x < 0 ∧ y < 0 → False) := by
  sorry

end NUMINAMATH_CALUDE_y₁_not_in_third_quadrant_l362_36235


namespace NUMINAMATH_CALUDE_line_through_circle_center_perpendicular_to_given_line_l362_36271

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + 2*x + y^2 = 0

-- Define the given line equation
def given_line_equation (x y : ℝ) : Prop := x + y = 0

-- Define the resulting line equation
def result_line_equation (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the center of a circle
def circle_center (c : ℝ × ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq x y ↔ (x - c.1)^2 + (y - c.2)^2 = 1

-- Define perpendicularity of two lines
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

theorem line_through_circle_center_perpendicular_to_given_line :
  ∃ c : ℝ × ℝ,
    circle_center c circle_equation ∧
    (∃ m₁ m₂ : ℝ,
      (∀ x y, given_line_equation x y ↔ y = m₁ * x) ∧
      (∀ x y, result_line_equation x y ↔ y = m₂ * x + c.2) ∧
      perpendicular m₁ m₂) :=
sorry

end NUMINAMATH_CALUDE_line_through_circle_center_perpendicular_to_given_line_l362_36271


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_53_and_3_l362_36216

theorem smallest_five_digit_divisible_by_53_and_3 : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit positive integer
  n % 53 = 0 ∧                -- divisible by 53
  n % 3 = 0 ∧                 -- divisible by 3
  n = 10062 ∧                 -- the number is 10062
  ∀ m : ℕ, (m ≥ 10000 ∧ m < 100000 ∧ m % 53 = 0 ∧ m % 3 = 0) → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_53_and_3_l362_36216


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l362_36200

theorem triangle_side_calculation (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a = 10 →
  A = π / 4 →
  B = π / 6 →
  a / Real.sin A = b / Real.sin B →
  b = 5 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l362_36200


namespace NUMINAMATH_CALUDE_problem_statement_l362_36228

theorem problem_statement : 
  let a := ((7 + 4 * Real.sqrt 3)^(1/2) - (7 - 4 * Real.sqrt 3)^(1/2)) / Real.sqrt 3
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l362_36228


namespace NUMINAMATH_CALUDE_ellen_hits_nine_l362_36223

-- Define the set of possible scores
def ScoreSet : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 15}

-- Define a type for the players
inductive Player : Type
| Alice | Ben | Cindy | Dave | Ellen | Frank

-- Define a function that returns the total score for each player
def playerScore (p : Player) : ℕ :=
  match p with
  | Player.Alice => 27
  | Player.Ben => 14
  | Player.Cindy => 20
  | Player.Dave => 22
  | Player.Ellen => 24
  | Player.Frank => 30

-- Define a predicate that checks if a list of scores is valid for a player
def validScores (scores : List ℕ) (p : Player) : Prop :=
  scores.length = 3 ∧
  scores.toFinset.card = 3 ∧
  (∀ s ∈ scores, s ∈ ScoreSet) ∧
  scores.sum = playerScore p

theorem ellen_hits_nine :
  ∃ (scores : List ℕ), validScores scores Player.Ellen ∧ 9 ∈ scores ∧
  (∀ (p : Player), p ≠ Player.Ellen → ∀ (s : List ℕ), validScores s p → 9 ∉ s) :=
sorry

end NUMINAMATH_CALUDE_ellen_hits_nine_l362_36223


namespace NUMINAMATH_CALUDE_remainder_problem_l362_36292

theorem remainder_problem (d r : ℤ) : 
  d > 1 → 
  2024 % d = r → 
  3250 % d = r → 
  4330 % d = r → 
  d - r = 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l362_36292


namespace NUMINAMATH_CALUDE_surjective_function_theorem_l362_36206

def is_surjective (f : ℕ → ℕ) : Prop :=
  ∀ y : ℕ, ∃ x : ℕ, f x = y

theorem surjective_function_theorem (f : ℕ → ℕ) 
  (h_surj : is_surjective f)
  (h_div : ∀ (m n : ℕ) (p : ℕ), Nat.Prime p → (p ∣ f (m + n) ↔ p ∣ f m + f n)) :
  ∀ n : ℕ, f n = n := by sorry

end NUMINAMATH_CALUDE_surjective_function_theorem_l362_36206


namespace NUMINAMATH_CALUDE_rhombus_prism_lateral_area_l362_36269

/-- Given a rectangular quadrilateral prism with a rhombus base, this theorem calculates its lateral surface area. -/
theorem rhombus_prism_lateral_area (side_length : ℝ) (diagonal_length : ℝ) (h1 : side_length = 2) (h2 : diagonal_length = 2 * Real.sqrt 3) :
  let lateral_edge := Real.sqrt (diagonal_length^2 - side_length^2)
  let perimeter := 4 * side_length
  let lateral_area := perimeter * lateral_edge
  lateral_area = 16 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_prism_lateral_area_l362_36269


namespace NUMINAMATH_CALUDE_complex_radical_expression_simplification_l362_36257

theorem complex_radical_expression_simplification :
  3 * Real.sqrt (1/3) + Real.sqrt 2 * (Real.sqrt 3 - Real.sqrt 6) - Real.sqrt 12 / Real.sqrt 2 = - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_radical_expression_simplification_l362_36257


namespace NUMINAMATH_CALUDE_quadratic_completion_square_l362_36282

theorem quadratic_completion_square (a : ℝ) : 
  (a > 0) → 
  (∃ n : ℝ, ∀ x : ℝ, x^2 + a*x + 27 = (x + n)^2 + 3) → 
  a = 4 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_completion_square_l362_36282


namespace NUMINAMATH_CALUDE_total_feed_amount_l362_36207

/-- Proves that the total amount of feed is 27 pounds given the specified conditions --/
theorem total_feed_amount (cheap_cost expensive_cost mix_cost cheap_amount : ℝ) 
  (h1 : cheap_cost = 0.17)
  (h2 : expensive_cost = 0.36)
  (h3 : mix_cost = 0.26)
  (h4 : cheap_amount = 14.2105263158)
  (h5 : cheap_cost * cheap_amount + expensive_cost * (total - cheap_amount) = mix_cost * total)
  : total = 27 :=
by sorry

#check total_feed_amount

end NUMINAMATH_CALUDE_total_feed_amount_l362_36207


namespace NUMINAMATH_CALUDE_fraction_equality_l362_36212

theorem fraction_equality (x y : ℚ) (a b : ℤ) (h1 : y = 40) (h2 : x + 35 = 4 * y) (h3 : 1/5 * x = a/b * y) (h4 : b ≠ 0) : a/b = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l362_36212


namespace NUMINAMATH_CALUDE_factoring_expression_l362_36281

theorem factoring_expression (a b : ℝ) : 6 * a^2 * b + 2 * a = 2 * a * (3 * a * b + 1) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l362_36281


namespace NUMINAMATH_CALUDE_final_sugar_amount_l362_36252

def sugar_calculation (initial : ℕ) (used : ℕ) (bought : ℕ) : ℕ :=
  initial - used + bought

theorem final_sugar_amount :
  sugar_calculation 65 18 50 = 97 := by
  sorry

end NUMINAMATH_CALUDE_final_sugar_amount_l362_36252


namespace NUMINAMATH_CALUDE_kite_altitude_l362_36224

theorem kite_altitude (C D K : ℝ × ℝ) (h1 : D.1 - C.1 = 15) (h2 : C.2 = D.2)
  (h3 : K.1 = C.1) (h4 : Real.tan (45 * π / 180) = (K.2 - C.2) / (K.1 - C.1))
  (h5 : Real.tan (30 * π / 180) = (K.2 - D.2) / (D.1 - K.1)) :
  K.2 - C.2 = 15 * (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_kite_altitude_l362_36224


namespace NUMINAMATH_CALUDE_a_4_equals_zero_l362_36294

def a (n : ℕ+) : ℤ := n^2 - 3*n - 4

theorem a_4_equals_zero : a 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_4_equals_zero_l362_36294


namespace NUMINAMATH_CALUDE_factorial_sum_remainder_l362_36220

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem factorial_sum_remainder (n : ℕ) (h : n ≥ 100) :
  sum_factorials n % 30 = sum_factorials 4 % 30 := by
  sorry

#eval sum_factorials 4 % 30  -- Should output 3

end NUMINAMATH_CALUDE_factorial_sum_remainder_l362_36220


namespace NUMINAMATH_CALUDE_fourth_root_over_sixth_root_of_seven_l362_36218

theorem fourth_root_over_sixth_root_of_seven (x : ℝ) :
  (7 ^ (1/4)) / (7 ^ (1/6)) = 7 ^ (1/12) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_over_sixth_root_of_seven_l362_36218


namespace NUMINAMATH_CALUDE_sin_n_equals_cos_678_l362_36202

theorem sin_n_equals_cos_678 (n : ℤ) (h1 : -120 ≤ n) (h2 : n ≤ 120) :
  Real.sin (n * π / 180) = Real.cos (678 * π / 180) → n = 48 := by
  sorry

end NUMINAMATH_CALUDE_sin_n_equals_cos_678_l362_36202


namespace NUMINAMATH_CALUDE_equal_roots_condition_l362_36203

/-- 
For a quadratic equation ax^2 + bx + c = 0, 
the discriminant is defined as b^2 - 4ac
-/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- 
A quadratic equation has two equal real roots 
if and only if its discriminant is zero
-/
axiom equal_roots_iff_zero_discriminant (a b c : ℝ) : 
  a ≠ 0 → (∃ x : ℝ, a*x^2 + b*x + c = 0 ∧ (∀ y : ℝ, a*y^2 + b*y + c = 0 → y = x)) ↔ 
    discriminant a b c = 0

/-- 
For the quadratic equation x^2 + 6x + m = 0 to have two equal real roots, 
m must equal 9
-/
theorem equal_roots_condition : 
  (∃ x : ℝ, x^2 + 6*x + m = 0 ∧ (∀ y : ℝ, y^2 + 6*y + m = 0 → y = x)) → m = 9 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l362_36203


namespace NUMINAMATH_CALUDE_min_side_length_l362_36236

theorem min_side_length (PQ PR SR SQ : ℝ) (h1 : PQ = 7.5) (h2 : PR = 14.5) (h3 : SR = 9.5) (h4 : SQ = 23.5) :
  ∃ (QR : ℕ), (QR : ℝ) > PR - PQ ∧ (QR : ℝ) > SQ - SR ∧ ∀ (n : ℕ), (n : ℝ) > PR - PQ ∧ (n : ℝ) > SQ - SR → n ≥ QR :=
by
  sorry

#check min_side_length

end NUMINAMATH_CALUDE_min_side_length_l362_36236


namespace NUMINAMATH_CALUDE_sin_pi_six_l362_36246

theorem sin_pi_six : Real.sin (π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_six_l362_36246


namespace NUMINAMATH_CALUDE_lower_price_option2_l362_36287

def initial_value : ℝ := 12000

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def option1_final_price : ℝ :=
  apply_discount (apply_discount (apply_discount initial_value 0.3) 0.1) 0.05

def option2_final_price : ℝ :=
  apply_discount (apply_discount (apply_discount initial_value 0.3) 0.05) 0.15

theorem lower_price_option2 :
  option2_final_price < option1_final_price ∧ option2_final_price = 6783 :=
by sorry

end NUMINAMATH_CALUDE_lower_price_option2_l362_36287


namespace NUMINAMATH_CALUDE_total_wool_is_82_l362_36253

/-- The number of scarves Aaron makes -/
def aaron_scarves : ℕ := 10

/-- The number of sweaters Aaron makes -/
def aaron_sweaters : ℕ := 5

/-- The number of sweaters Enid makes -/
def enid_sweaters : ℕ := 8

/-- The number of balls of wool used for one scarf -/
def wool_per_scarf : ℕ := 3

/-- The number of balls of wool used for one sweater -/
def wool_per_sweater : ℕ := 4

/-- The total number of balls of wool used by Enid and Aaron -/
def total_wool : ℕ := aaron_scarves * wool_per_scarf + aaron_sweaters * wool_per_sweater + enid_sweaters * wool_per_sweater

theorem total_wool_is_82 : total_wool = 82 := by sorry

end NUMINAMATH_CALUDE_total_wool_is_82_l362_36253


namespace NUMINAMATH_CALUDE_problem_one_problem_two_problem_three_problem_four_l362_36243

-- Problem 1
theorem problem_one : 6 + (-8) - (-5) = 3 := by sorry

-- Problem 2
theorem problem_two : 5 + 3/5 + (-5 - 2/3) + 4 + 2/5 + (-1/3) = 4 := by sorry

-- Problem 3
theorem problem_three : (-1/2 + 1/6 - 1/4) * 12 = -7 := by sorry

-- Problem 4
theorem problem_four : -1^2022 + 27 * (-1/3)^2 - |(-5)| = -3 := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_problem_three_problem_four_l362_36243


namespace NUMINAMATH_CALUDE_simplify_and_ratio_l362_36270

theorem simplify_and_ratio (m : ℝ) : ∃ (c d : ℝ), 
  (6 * m + 12) / 3 = c * m + d ∧ c / d = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_ratio_l362_36270


namespace NUMINAMATH_CALUDE_basil_pots_l362_36249

theorem basil_pots (rosemary_pots thyme_pots : ℕ)
  (basil_leaves rosemary_leaves thyme_leaves total_leaves : ℕ) :
  rosemary_pots = 9 →
  thyme_pots = 6 →
  basil_leaves = 4 →
  rosemary_leaves = 18 →
  thyme_leaves = 30 →
  total_leaves = 354 →
  ∃ basil_pots : ℕ,
    basil_pots * basil_leaves +
    rosemary_pots * rosemary_leaves +
    thyme_pots * thyme_leaves = total_leaves ∧
    basil_pots = 3 :=
by sorry

end NUMINAMATH_CALUDE_basil_pots_l362_36249


namespace NUMINAMATH_CALUDE_salary_change_percentage_l362_36234

theorem salary_change_percentage (x : ℝ) : 
  (1 - x / 100) * (1 + x / 100) = 64 / 100 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l362_36234


namespace NUMINAMATH_CALUDE_sum_remainder_is_six_l362_36264

/-- Given positive integers a, b, c less than 7 satisfying certain congruences,
    prove that their sum has a remainder of 6 when divided by 7. -/
theorem sum_remainder_is_six (a b c : ℕ) 
    (ha : a < 7) (hb : b < 7) (hc : c < 7) 
    (ha_pos : a > 0) (hb_pos : b > 0) (hc_pos : c > 0)
    (h1 : a * b * c % 7 = 2)
    (h2 : (3 * c) % 7 = 4)
    (h3 : (4 * b) % 7 = (2 + b) % 7) :
    (a + b + c) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_is_six_l362_36264


namespace NUMINAMATH_CALUDE_coordinates_wrt_origin_l362_36274

-- Define a point in a 2D Cartesian coordinate system
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the origin
def origin : Point2D := ⟨0, 0⟩

-- Define the point P
def P : Point2D := ⟨2, 4⟩

-- Theorem stating that the coordinates of P with respect to the origin are (2,4)
theorem coordinates_wrt_origin :
  (P.x - origin.x = 2) ∧ (P.y - origin.y = 4) := by
  sorry

end NUMINAMATH_CALUDE_coordinates_wrt_origin_l362_36274


namespace NUMINAMATH_CALUDE_intersection_empty_union_equals_A_l362_36295

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -3 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 4*a*x + 3*a^2 = 0}

-- Theorem for part (1)
theorem intersection_empty (a : ℝ) : 
  A ∩ B a = ∅ ↔ a ≤ -3 ∨ a ≥ 4 :=
sorry

-- Theorem for part (2)
theorem union_equals_A (a : ℝ) :
  A ∪ B a = A ↔ -1 < a ∧ a < 4/3 :=
sorry

end NUMINAMATH_CALUDE_intersection_empty_union_equals_A_l362_36295


namespace NUMINAMATH_CALUDE_soap_box_theorem_l362_36208

/-- The number of bars of soap in each box of bars -/
def bars_per_box : ℕ := 5

/-- The smallest number of each type of soap sold -/
def min_sold : ℕ := 95

/-- The number of bottles of soap in each box of bottles -/
def bottles_per_box : ℕ := 19

theorem soap_box_theorem :
  ∃ (bar_boxes bottle_boxes : ℕ),
    bar_boxes * bars_per_box = bottle_boxes * bottles_per_box ∧
    bar_boxes * bars_per_box = min_sold ∧
    bottle_boxes * bottles_per_box = min_sold ∧
    bottles_per_box > 1 ∧
    bottles_per_box < min_sold :=
by sorry

end NUMINAMATH_CALUDE_soap_box_theorem_l362_36208


namespace NUMINAMATH_CALUDE_winnie_lollipop_distribution_l362_36221

/-- The number of lollipops left after equal distribution --/
def lollipops_left (cherry wintergreen grape shrimp_cocktail friends : ℕ) : ℕ :=
  (cherry + wintergreen + grape + shrimp_cocktail) % friends

theorem winnie_lollipop_distribution :
  lollipops_left 55 134 12 265 15 = 1 := by sorry

end NUMINAMATH_CALUDE_winnie_lollipop_distribution_l362_36221


namespace NUMINAMATH_CALUDE_simulation_needed_for_exact_probability_l362_36256

structure Player where
  money : Nat

structure GameState where
  players : List Player

def initial_state : GameState :=
  { players := [{ money := 2 }, { money := 2 }, { money := 2 }] }

def can_give_money (p : Player) : Bool :=
  p.money > 1

def ring_bell (state : GameState) : GameState :=
  sorry

def is_final_state (state : GameState) : Bool :=
  state.players.all (fun p => p.money = 2)

noncomputable def probability_of_final_state (num_rings : Nat) : ℝ :=
  sorry

theorem simulation_needed_for_exact_probability :
  ∀ (analytical_function : Nat → ℝ),
    ∃ (ε : ℝ), ε > 0 ∧
      |probability_of_final_state 2019 - analytical_function 2019| > ε :=
by sorry

end NUMINAMATH_CALUDE_simulation_needed_for_exact_probability_l362_36256


namespace NUMINAMATH_CALUDE_points_collinear_l362_36217

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangular pyramid -/
structure TriangularPyramid where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Check if four points are collinear -/
def are_collinear (p1 p2 p3 p4 : Point3D) : Prop :=
  ∃ (t1 t2 t3 : ℝ), p2 = Point3D.mk (p1.x + t1 * (p4.x - p1.x)) (p1.y + t1 * (p4.y - p1.y)) (p1.z + t1 * (p4.z - p1.z)) ∧
                     p3 = Point3D.mk (p1.x + t2 * (p4.x - p1.x)) (p1.y + t2 * (p4.y - p1.y)) (p1.z + t2 * (p4.z - p1.z)) ∧
                     p4 = Point3D.mk (p1.x + t3 * (p4.x - p1.x)) (p1.y + t3 * (p4.y - p1.y)) (p1.z + t3 * (p4.z - p1.z))

/-- Main theorem -/
theorem points_collinear (pyramid : TriangularPyramid) 
  (M K P H E F Q T : Point3D)
  (h1 : (pyramid.A.x - M.x)^2 + (pyramid.A.y - M.y)^2 + (pyramid.A.z - M.z)^2 = 
        (M.x - K.x)^2 + (M.y - K.y)^2 + (M.z - K.z)^2)
  (h2 : (M.x - K.x)^2 + (M.y - K.y)^2 + (M.z - K.z)^2 = 
        (K.x - pyramid.D.x)^2 + (K.y - pyramid.D.y)^2 + (K.z - pyramid.D.z)^2)
  (h3 : (pyramid.B.x - P.x)^2 + (pyramid.B.y - P.y)^2 + (pyramid.B.z - P.z)^2 = 
        (P.x - H.x)^2 + (P.y - H.y)^2 + (P.z - H.z)^2)
  (h4 : (P.x - H.x)^2 + (P.y - H.y)^2 + (P.z - H.z)^2 = 
        (H.x - pyramid.C.x)^2 + (H.y - pyramid.C.y)^2 + (H.z - pyramid.C.z)^2)
  (h5 : (pyramid.A.x - E.x)^2 + (pyramid.A.y - E.y)^2 + (pyramid.A.z - E.z)^2 = 
        0.25 * ((pyramid.A.x - pyramid.B.x)^2 + (pyramid.A.y - pyramid.B.y)^2 + (pyramid.A.z - pyramid.B.z)^2))
  (h6 : (M.x - F.x)^2 + (M.y - F.y)^2 + (M.z - F.z)^2 = 
        0.25 * ((M.x - P.x)^2 + (M.y - P.y)^2 + (M.z - P.z)^2))
  (h7 : (K.x - Q.x)^2 + (K.y - Q.y)^2 + (K.z - Q.z)^2 = 
        0.25 * ((K.x - H.x)^2 + (K.y - H.y)^2 + (K.z - H.z)^2))
  (h8 : (pyramid.D.x - T.x)^2 + (pyramid.D.y - T.y)^2 + (pyramid.D.z - T.z)^2 = 
        0.25 * ((pyramid.D.x - pyramid.C.x)^2 + (pyramid.D.y - pyramid.C.y)^2 + (pyramid.D.z - pyramid.C.z)^2))
  : are_collinear E F Q T :=
sorry

end NUMINAMATH_CALUDE_points_collinear_l362_36217


namespace NUMINAMATH_CALUDE_percent_to_decimal_three_percent_to_decimal_l362_36214

theorem percent_to_decimal (p : ℝ) : p / 100 = p * 0.01 := by sorry

theorem three_percent_to_decimal : (3 : ℝ) / 100 = 0.03 := by sorry

end NUMINAMATH_CALUDE_percent_to_decimal_three_percent_to_decimal_l362_36214


namespace NUMINAMATH_CALUDE_distance_to_directrix_l362_36226

/-- A parabola C with equation y² = 2px and a point A(1, √5) lying on it -/
structure Parabola where
  p : ℝ
  A : ℝ × ℝ
  h1 : A.1 = 1
  h2 : A.2 = Real.sqrt 5
  h3 : A.2^2 = 2 * p * A.1

/-- The distance from point A to the directrix of parabola C is 9/4 -/
theorem distance_to_directrix (C : Parabola) : 
  C.A.1 + C.p / 2 = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_directrix_l362_36226


namespace NUMINAMATH_CALUDE_circle_area_ratio_l362_36290

theorem circle_area_ratio : 
  ∀ (r1 r2 : ℝ), r1 > 0 → r2 = 3 * r1 → 
  (π * r2^2) / (π * r1^2) = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l362_36290


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l362_36210

/-- Given a circle with circumference 36 cm, its area is 324/π square centimeters. -/
theorem circle_area_from_circumference : 
  ∀ (r : ℝ), 2 * π * r = 36 → π * r^2 = 324 / π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l362_36210


namespace NUMINAMATH_CALUDE_reciprocal_of_2022_l362_36209

theorem reciprocal_of_2022 : (2022⁻¹ : ℚ) = 1 / 2022 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_2022_l362_36209


namespace NUMINAMATH_CALUDE_square_sum_difference_l362_36245

theorem square_sum_difference (n : ℕ) : 
  (2*n + 1)^2 - (2*n - 1)^2 + (2*n - 1)^2 - (2*n - 3)^2 + (2*n - 3)^2 - (2*n - 5)^2 + 
  (2*n - 5)^2 - (2*n - 7)^2 + (2*n - 7)^2 - (2*n - 9)^2 + (2*n - 9)^2 - (2*n - 11)^2 = 288 :=
by sorry

end NUMINAMATH_CALUDE_square_sum_difference_l362_36245


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l362_36259

theorem perfect_square_trinomial (m : ℚ) : 
  (∃ a b : ℚ, ∀ x, 4*x^2 - (2*m+1)*x + 121 = (a*x + b)^2) → 
  (m = 43/2 ∨ m = -45/2) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l362_36259


namespace NUMINAMATH_CALUDE_transistors_in_2000_l362_36286

/-- Moore's law states that the number of transistors doubles every two years -/
def moores_law (t : ℕ) : ℕ := 2^(t/2)

/-- The number of transistors in a typical CPU in 1990 -/
def transistors_1990 : ℕ := 1000000

/-- The year we're calculating for -/
def target_year : ℕ := 2000

/-- The starting year -/
def start_year : ℕ := 1990

theorem transistors_in_2000 : 
  transistors_1990 * moores_law (target_year - start_year) = 32000000 := by
  sorry

end NUMINAMATH_CALUDE_transistors_in_2000_l362_36286


namespace NUMINAMATH_CALUDE_snake_paint_theorem_l362_36240

/-- The amount of paint needed for a single cube -/
def paint_per_cube : ℕ := 60

/-- The number of cubes in the snake -/
def total_cubes : ℕ := 2016

/-- The number of cubes in each periodic fragment -/
def cubes_per_fragment : ℕ := 6

/-- The additional paint needed for adjustments -/
def additional_paint : ℕ := 20

/-- The total amount of paint needed for the snake -/
def total_paint_needed : ℕ :=
  (total_cubes / cubes_per_fragment) * (cubes_per_fragment * paint_per_cube) + additional_paint

theorem snake_paint_theorem :
  total_paint_needed = 120980 := by
  sorry

end NUMINAMATH_CALUDE_snake_paint_theorem_l362_36240


namespace NUMINAMATH_CALUDE_curve_self_intersects_l362_36283

/-- The x-coordinate of a point on the curve given a parameter t -/
def x (t : ℝ) : ℝ := t^2 - 4

/-- The y-coordinate of a point on the curve given a parameter t -/
def y (t : ℝ) : ℝ := t^3 - 6*t + 7

/-- The curve intersects itself if there exist two distinct real numbers that yield the same point -/
def self_intersects : Prop :=
  ∃ a b : ℝ, a ≠ b ∧ x a = x b ∧ y a = y b

/-- The point of self-intersection -/
def intersection_point : ℝ × ℝ := (2, 7)

/-- Theorem stating that the curve intersects itself at (2, 7) -/
theorem curve_self_intersects :
  self_intersects ∧ ∃ a b : ℝ, a ≠ b ∧ x a = (intersection_point.1) ∧ y a = (intersection_point.2) :=
sorry

end NUMINAMATH_CALUDE_curve_self_intersects_l362_36283


namespace NUMINAMATH_CALUDE_number_of_divisors_36_l362_36248

theorem number_of_divisors_36 : Nat.card {d : ℕ | d > 0 ∧ 36 % d = 0} = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_36_l362_36248
