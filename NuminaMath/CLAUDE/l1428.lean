import Mathlib

namespace NUMINAMATH_CALUDE_min_value_fraction_equality_condition_l1428_142866

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / (a + 2*b) + b / (a + b) ≥ (2*Real.sqrt 2 - 1) / (2*Real.sqrt 2) :=
by sorry

theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / (a + 2*b) + b / (a + b) = (2*Real.sqrt 2 - 1) / (2*Real.sqrt 2) ↔ a = Real.sqrt 2 * b :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_equality_condition_l1428_142866


namespace NUMINAMATH_CALUDE_equation_solutions_l1428_142831

theorem equation_solutions :
  (∃ x : ℝ, 0.5 * x + 1.1 = 6.5 - 1.3 * x ∧ x = 3) ∧
  (∃ x : ℝ, (1/6) * (3 * x - 9) = (2/5) * x - 3 ∧ x = -15) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1428_142831


namespace NUMINAMATH_CALUDE_analysis_seeks_sufficient_condition_l1428_142811

/-- Represents a mathematical method for proving inequalities -/
inductive ProofMethod
| Analysis
| Synthesis

/-- Represents types of conditions in mathematical proofs -/
inductive ConditionType
| Sufficient
| Necessary
| NecessaryAndSufficient
| Neither

/-- Represents an inequality to be proved -/
structure Inequality where
  -- We don't need to specify the actual inequality, just that it exists
  dummy : Unit

/-- Function that represents the process of seeking a condition in the analysis method -/
def seekCondition (m : ProofMethod) (i : Inequality) : ConditionType :=
  match m with
  | ProofMethod.Analysis => ConditionType.Sufficient
  | ProofMethod.Synthesis => ConditionType.Neither -- This is arbitrary for non-Analysis methods

/-- Theorem stating that the analysis method seeks a sufficient condition -/
theorem analysis_seeks_sufficient_condition (i : Inequality) :
  seekCondition ProofMethod.Analysis i = ConditionType.Sufficient := by
  sorry

#check analysis_seeks_sufficient_condition

end NUMINAMATH_CALUDE_analysis_seeks_sufficient_condition_l1428_142811


namespace NUMINAMATH_CALUDE_smallest_difference_l1428_142887

/-- Vovochka's sum method for three-digit numbers -/
def vovochka_sum (a b c d e f : ℕ) : ℕ :=
  1000 * (a + d) + 100 * (b + e) + (c + f)

/-- Correct sum method for three-digit numbers -/
def correct_sum (a b c d e f : ℕ) : ℕ :=
  100 * (a + d) + 10 * (b + e) + (c + f)

/-- The difference between Vovochka's sum and the correct sum -/
def sum_difference (a b c d e f : ℕ) : ℕ :=
  vovochka_sum a b c d e f - correct_sum a b c d e f

theorem smallest_difference :
  ∀ a b c d e f : ℕ,
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 →
    sum_difference a b c d e f > 0 →
    sum_difference a b c d e f ≥ 1800 :=
sorry

end NUMINAMATH_CALUDE_smallest_difference_l1428_142887


namespace NUMINAMATH_CALUDE_correct_number_of_men_l1428_142873

/-- The number of men in the first group that completes a job in 15 days,
    given that 25 men can finish the same job in 18 days. -/
def number_of_men : ℕ := 30

/-- The number of days taken by the first group to complete the job. -/
def days_first_group : ℕ := 15

/-- The number of men in the second group. -/
def men_second_group : ℕ := 25

/-- The number of days taken by the second group to complete the job. -/
def days_second_group : ℕ := 18

/-- Theorem stating that the number of men in the first group is correct. -/
theorem correct_number_of_men :
  number_of_men * days_first_group = men_second_group * days_second_group :=
sorry

end NUMINAMATH_CALUDE_correct_number_of_men_l1428_142873


namespace NUMINAMATH_CALUDE_roots_sum_equals_four_l1428_142809

/-- Given that x₁ and x₂ are the roots of ln|x-2| = m for some real m, prove that x₁ + x₂ = 4 -/
theorem roots_sum_equals_four (m : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : Real.log (|x₁ - 2|) = m) 
  (h₂ : Real.log (|x₂ - 2|) = m) : 
  x₁ + x₂ = 4 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_equals_four_l1428_142809


namespace NUMINAMATH_CALUDE_cross_figure_sum_l1428_142852

-- Define the set of digits
def Digits : Set Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the structure
structure CrossFigure where
  vertical : Fin 3 → Nat
  horizontal1 : Fin 3 → Nat
  horizontal2 : Fin 3 → Nat
  all_digits : List Nat
  h_vertical_sum : vertical 0 + vertical 1 + vertical 2 = 17
  h_horizontal1_sum : horizontal1 0 + horizontal1 1 + horizontal1 2 = 18
  h_horizontal2_sum : horizontal2 0 + horizontal2 1 + horizontal2 2 = 13
  h_intersection1 : vertical 0 = horizontal1 0
  h_intersection2 : vertical 2 = horizontal2 0
  h_all_digits : all_digits.length = 7
  h_all_digits_unique : all_digits.Nodup
  h_all_digits_in_set : ∀ d ∈ all_digits, d ∈ Digits
  h_all_digits_cover : (vertical 0 :: vertical 1 :: vertical 2 :: 
                        horizontal1 1 :: horizontal1 2 :: 
                        horizontal2 1 :: horizontal2 2 :: []).toFinset = all_digits.toFinset

theorem cross_figure_sum (cf : CrossFigure) : 
  cf.all_digits.sum = 34 := by
  sorry

end NUMINAMATH_CALUDE_cross_figure_sum_l1428_142852


namespace NUMINAMATH_CALUDE_function_satisfies_equation_l1428_142800

theorem function_satisfies_equation (x b : ℝ) : 
  let y := (b + x) / (1 + b*x)
  let y' := ((1 - b^2) / (1 + b*x)^2)
  y - x * y' = b * (1 + x^2 * y') := by sorry

end NUMINAMATH_CALUDE_function_satisfies_equation_l1428_142800


namespace NUMINAMATH_CALUDE_ratio_p_to_q_l1428_142847

theorem ratio_p_to_q (p q : ℚ) (h : 18 / 7 + (2 * q - p) / (2 * q + p) = 3) : p / q = 29 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ratio_p_to_q_l1428_142847


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1428_142896

theorem min_value_quadratic (x y : ℝ) : x^2 + 4*x*y + 5*y^2 - 8*x - 6*y ≥ -41 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1428_142896


namespace NUMINAMATH_CALUDE_walnut_trees_planted_l1428_142865

/-- The number of walnut trees planted in a park --/
theorem walnut_trees_planted (initial_trees final_trees : ℕ) :
  initial_trees < final_trees →
  final_trees - initial_trees = final_trees - initial_trees :=
by sorry

end NUMINAMATH_CALUDE_walnut_trees_planted_l1428_142865


namespace NUMINAMATH_CALUDE_simple_interest_time_period_l1428_142884

theorem simple_interest_time_period 
  (principal : ℝ)
  (amount1 : ℝ)
  (amount2 : ℝ)
  (rate_increase : ℝ)
  (h1 : principal = 825)
  (h2 : amount1 = 956)
  (h3 : amount2 = 1055)
  (h4 : rate_increase = 4) :
  ∃ (rate : ℝ) (time : ℝ),
    amount1 = principal + (principal * rate * time) / 100 ∧
    amount2 = principal + (principal * (rate + rate_increase) * time) / 100 ∧
    time = 3 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_time_period_l1428_142884


namespace NUMINAMATH_CALUDE_restaurant_bill_proof_l1428_142848

/-- Calculates the total bill given the number of people and the amount each person paid -/
def totalBill (numPeople : ℕ) (amountPerPerson : ℕ) : ℕ :=
  numPeople * amountPerPerson

/-- Proves that if three people divide a bill evenly and each pays $45, then the total bill is $135 -/
theorem restaurant_bill_proof :
  totalBill 3 45 = 135 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_proof_l1428_142848


namespace NUMINAMATH_CALUDE_container_volume_ratio_l1428_142869

theorem container_volume_ratio (container1 container2 : ℝ) : 
  container1 > 0 → container2 > 0 →
  (2/3 : ℝ) * container1 + (1/6 : ℝ) * container1 = (5/6 : ℝ) * container2 →
  container1 = container2 := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l1428_142869


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_implies_a_in_open_interval_l1428_142893

-- Define the complex number z
def z (a : ℝ) : ℂ := (1 + a * Complex.I) * (1 - Complex.I)

-- Define the condition for a point to be in the fourth quadrant
def in_fourth_quadrant (w : ℂ) : Prop := 0 < w.re ∧ w.im < 0

-- Theorem statement
theorem z_in_fourth_quadrant_implies_a_in_open_interval :
  ∀ a : ℝ, in_fourth_quadrant (z a) → -1 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_implies_a_in_open_interval_l1428_142893


namespace NUMINAMATH_CALUDE_initial_overs_played_l1428_142821

/-- Proves that the number of overs played initially is 15, given the target score,
    initial run rate, required run rate for remaining overs, and the number of remaining overs. -/
theorem initial_overs_played (target_score : ℝ) (initial_run_rate : ℝ) (required_run_rate : ℝ) (remaining_overs : ℝ) :
  target_score = 275 →
  initial_run_rate = 3.2 →
  required_run_rate = 6.485714285714286 →
  remaining_overs = 35 →
  ∃ (initial_overs : ℝ), initial_overs = 15 ∧
    target_score = initial_run_rate * initial_overs + required_run_rate * remaining_overs :=
by
  sorry


end NUMINAMATH_CALUDE_initial_overs_played_l1428_142821


namespace NUMINAMATH_CALUDE_circle_plus_k_circle_plus_k_k_l1428_142849

-- Define the ⊕ operation
def circle_plus (x y : ℝ) : ℝ := x^3 + x - y

-- Theorem statement
theorem circle_plus_k_circle_plus_k_k (k : ℝ) : circle_plus k (circle_plus k k) = k := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_k_circle_plus_k_k_l1428_142849


namespace NUMINAMATH_CALUDE_pool_cost_per_person_l1428_142817

theorem pool_cost_per_person
  (total_earnings : ℝ)
  (num_people : ℕ)
  (amount_left : ℝ)
  (h1 : total_earnings = 30)
  (h2 : num_people = 10)
  (h3 : amount_left = 5) :
  (total_earnings - amount_left) / num_people = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_pool_cost_per_person_l1428_142817


namespace NUMINAMATH_CALUDE_diane_poker_loss_l1428_142894

/-- The total amount of money Diane lost in her poker game -/
def total_loss (initial_amount won_amount final_debt : ℝ) : ℝ :=
  initial_amount + won_amount + final_debt

/-- Theorem stating that Diane's total loss is $215 -/
theorem diane_poker_loss :
  let initial_amount : ℝ := 100
  let won_amount : ℝ := 65
  let final_debt : ℝ := 50
  total_loss initial_amount won_amount final_debt = 215 := by
  sorry

end NUMINAMATH_CALUDE_diane_poker_loss_l1428_142894


namespace NUMINAMATH_CALUDE_sin_cos_difference_75_15_l1428_142826

theorem sin_cos_difference_75_15 :
  Real.sin (75 * π / 180) * Real.cos (15 * π / 180) -
  Real.cos (75 * π / 180) * Real.sin (15 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_75_15_l1428_142826


namespace NUMINAMATH_CALUDE_book_pages_proof_l1428_142892

/-- Calculates the total number of pages in a book given the reading schedule --/
def total_pages (pages_per_day_first_four : ℕ) (pages_per_day_next_two : ℕ) (pages_last_day : ℕ) : ℕ :=
  4 * pages_per_day_first_four + 2 * pages_per_day_next_two + pages_last_day

/-- Proves that the total number of pages in the book is 264 --/
theorem book_pages_proof : total_pages 42 38 20 = 264 := by
  sorry

#eval total_pages 42 38 20

end NUMINAMATH_CALUDE_book_pages_proof_l1428_142892


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1428_142874

theorem complex_fraction_equality (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^3 + a*b + b^3 = 0) : 
  (a^10 + b^10) / (a + b)^10 = 1/18 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1428_142874


namespace NUMINAMATH_CALUDE_magistrate_seating_arrangements_l1428_142803

/-- Represents the number of people of each nationality -/
def magistrates : Finset (Nat × Nat) := {(2, 3), (1, 4)}

/-- The total number of magistrate members -/
def total_members : Nat := (magistrates.sum (λ x => x.1))

/-- Calculates the number of valid seating arrangements -/
def valid_arrangements (m : Finset (Nat × Nat)) (total : Nat) : Nat :=
  sorry

theorem magistrate_seating_arrangements :
  valid_arrangements magistrates total_members = 1895040 := by
  sorry

end NUMINAMATH_CALUDE_magistrate_seating_arrangements_l1428_142803


namespace NUMINAMATH_CALUDE_cube_root_equation_sum_l1428_142838

theorem cube_root_equation_sum (x y z : ℕ+) :
  (4 * Real.sqrt (Real.rpow 7 (1/3) - Real.rpow 6 (1/3)) = Real.rpow x.val (1/3) + Real.rpow y.val (1/3) - Real.rpow z.val (1/3)) →
  x.val + y.val + z.val = 79 := by
sorry

end NUMINAMATH_CALUDE_cube_root_equation_sum_l1428_142838


namespace NUMINAMATH_CALUDE_admission_ratio_theorem_l1428_142832

def admission_problem (a c : ℕ+) : Prop :=
  30 * a.val + 15 * c.val = 2550

def ratio_closest_to_one (a c : ℕ+) : Prop :=
  ∀ (x y : ℕ+), admission_problem x y →
    |((a:ℚ) / c) - 1| ≤ |((x:ℚ) / y) - 1|

theorem admission_ratio_theorem :
  ∃ (a c : ℕ+), admission_problem a c ∧ ratio_closest_to_one a c ∧ a.val = 57 ∧ c.val = 56 :=
sorry

end NUMINAMATH_CALUDE_admission_ratio_theorem_l1428_142832


namespace NUMINAMATH_CALUDE_total_hockey_games_l1428_142871

/-- The number of hockey games in a season -/
def hockey_season_games (games_per_month : ℕ) (months_in_season : ℕ) : ℕ :=
  games_per_month * months_in_season

/-- Theorem: The total number of hockey games in the season is 182 -/
theorem total_hockey_games : hockey_season_games 13 14 = 182 := by
  sorry

end NUMINAMATH_CALUDE_total_hockey_games_l1428_142871


namespace NUMINAMATH_CALUDE_johnson_family_has_four_children_l1428_142827

/-- Represents the Johnson family -/
structure JohnsonFamily where
  num_children : ℕ
  father_age : ℕ
  mother_age : ℕ
  children_ages : List ℕ

/-- The conditions of the Johnson family -/
def johnson_family_conditions (family : JohnsonFamily) : Prop :=
  family.father_age = 55 ∧
  family.num_children + 2 = 6 ∧
  (family.father_age + family.mother_age + family.children_ages.sum) / 6 = 25 ∧
  (family.mother_age + family.children_ages.sum) / (family.num_children + 1) = 18

/-- The theorem stating that the Johnson family has 4 children -/
theorem johnson_family_has_four_children (family : JohnsonFamily) 
  (h : johnson_family_conditions family) : family.num_children = 4 := by
  sorry

end NUMINAMATH_CALUDE_johnson_family_has_four_children_l1428_142827


namespace NUMINAMATH_CALUDE_grain_remaining_after_crash_l1428_142836

/-- The amount of grain remaining onboard after a ship crash -/
def remaining_grain (original : ℕ) (spilled : ℕ) : ℕ :=
  original - spilled

/-- Theorem stating the amount of grain remaining onboard after the specific crash -/
theorem grain_remaining_after_crash : 
  remaining_grain 50870 49952 = 918 := by
  sorry

end NUMINAMATH_CALUDE_grain_remaining_after_crash_l1428_142836


namespace NUMINAMATH_CALUDE_casey_savings_l1428_142822

/-- Represents the weekly savings when hiring the cheaper employee --/
def weeklySavings (hourlyRate1 hourlyRate2 subsidy hoursPerWeek : ℝ) : ℝ :=
  (hourlyRate1 * hoursPerWeek) - ((hourlyRate2 - subsidy) * hoursPerWeek)

/-- Proves that Casey saves $160 per week by hiring the cheaper employee --/
theorem casey_savings :
  let hourlyRate1 : ℝ := 20
  let hourlyRate2 : ℝ := 22
  let subsidy : ℝ := 6
  let hoursPerWeek : ℝ := 40
  weeklySavings hourlyRate1 hourlyRate2 subsidy hoursPerWeek = 160 := by
  sorry

end NUMINAMATH_CALUDE_casey_savings_l1428_142822


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1428_142862

def A : Set ℝ := {x | |x| < 1}
def B : Set ℝ := {x | -2 < x ∧ x < 0}

theorem intersection_of_A_and_B : A ∩ B = Set.Ioo (-1) 0 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1428_142862


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1428_142859

theorem polynomial_divisibility (k n : ℕ+) 
  (h : (k : ℝ) + 1 ≤ Real.sqrt ((n : ℝ) + 1 / Real.log (n + 1))) :
  ∃ (P : Polynomial ℤ), 
    (∀ i, Polynomial.coeff P i ∈ ({0, 1, -1} : Set ℤ)) ∧ 
    (Polynomial.degree P = n) ∧
    ((X - 1) ^ k.val ∣ P) :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1428_142859


namespace NUMINAMATH_CALUDE_system_solution_triangle_side_range_l1428_142833

-- Problem 1
theorem system_solution (m : ℤ) : 
  (∃ x y : ℝ, 2*x + y = -3*m + 2 ∧ x + 2*y = 4 ∧ x + y > -3/2) ↔ 
  (m = 1 ∨ m = 2 ∨ m = 3) :=
sorry

-- Problem 2
theorem triangle_side_range (a b c : ℝ) :
  (a^2 + b^2 = 10*a + 8*b - 41) ∧
  (c ≥ a ∧ c ≥ b) →
  (5 ≤ c ∧ c < 9) :=
sorry

end NUMINAMATH_CALUDE_system_solution_triangle_side_range_l1428_142833


namespace NUMINAMATH_CALUDE_function_analysis_l1428_142857

def f (a : ℝ) (x : ℝ) : ℝ := |x^2 - 2*x + a - 1| - a^2 - 2*a

theorem function_analysis (a : ℝ) :
  (a = 3 → {x : ℝ | f 3 x ≥ -10} = {x : ℝ | x ≥ 3 ∨ x ≤ -1}) ∧
  ({a : ℝ | ∀ x, f a x ≥ 0} = {a : ℝ | -2 ≤ a ∧ a ≤ 0}) :=
by sorry

end NUMINAMATH_CALUDE_function_analysis_l1428_142857


namespace NUMINAMATH_CALUDE_existence_of_xy_l1428_142819

theorem existence_of_xy (n : ℕ) (k : ℕ) (h : n = 4 * k + 1) : 
  ∃ (x y : ℤ), (x^n + y^n) ∈ {z : ℤ | ∃ (a b : ℤ), z = a^2 + n * b^2} ∧ 
  (x + y) ∉ {z : ℤ | ∃ (a b : ℤ), z = a^2 + n * b^2} := by
  sorry

end NUMINAMATH_CALUDE_existence_of_xy_l1428_142819


namespace NUMINAMATH_CALUDE_nested_sqrt_equality_l1428_142825

theorem nested_sqrt_equality (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = (x ^ (11 / 8)) :=
sorry

end NUMINAMATH_CALUDE_nested_sqrt_equality_l1428_142825


namespace NUMINAMATH_CALUDE_infinite_solutions_abs_value_equation_l1428_142818

theorem infinite_solutions_abs_value_equation (a : ℝ) :
  (∀ x : ℝ, |x - 2| = a * x - 2) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_abs_value_equation_l1428_142818


namespace NUMINAMATH_CALUDE_simplify_radical_product_l1428_142839

theorem simplify_radical_product (x : ℝ) (h : x > 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x) = 120 * x * Real.sqrt (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l1428_142839


namespace NUMINAMATH_CALUDE_slower_speed_calculation_l1428_142841

/-- Given a person who walks 30 km at a slower speed and could have walked 45 km at 15 km/hr 
    in the same amount of time, prove that the slower speed is 10 km/hr. -/
theorem slower_speed_calculation (x : ℝ) (h1 : x > 0) : 
  (30 / x = 45 / 15) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_slower_speed_calculation_l1428_142841


namespace NUMINAMATH_CALUDE_difference_of_squares_l1428_142879

theorem difference_of_squares (a b : ℝ) : (-a + b) * (-a - b) = b^2 - a^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1428_142879


namespace NUMINAMATH_CALUDE_percentage_not_sold_approx_l1428_142898

def initial_stock : ℕ := 1200

def monday_sold : ℕ := 75
def monday_returned : ℕ := 6

def tuesday_sold : ℕ := 50

def wednesday_sold : ℕ := 64
def wednesday_returned : ℕ := 8

def thursday_sold : ℕ := 78

def friday_sold : ℕ := 135
def friday_returned : ℕ := 5

def total_sold : ℕ := 
  (monday_sold - monday_returned) + 
  tuesday_sold + 
  (wednesday_sold - wednesday_returned) + 
  thursday_sold + 
  (friday_sold - friday_returned)

def books_not_sold : ℕ := initial_stock - total_sold

def percentage_not_sold : ℚ := (books_not_sold : ℚ) / (initial_stock : ℚ) * 100

theorem percentage_not_sold_approx :
  abs (percentage_not_sold - 68.08) < 0.01 := by sorry

end NUMINAMATH_CALUDE_percentage_not_sold_approx_l1428_142898


namespace NUMINAMATH_CALUDE_drawing_pie_satisfies_hunger_is_impossible_l1428_142878

/-- An event that involves drawing a pie and satisfying hunger -/
def drawing_pie_to_satisfy_hunger : Set (Nat × Nat) := sorry

/-- Definition of an impossible event -/
def impossible_event (E : Set (Nat × Nat)) : Prop :=
  E = ∅

/-- Theorem: Drawing a pie to satisfy hunger is an impossible event -/
theorem drawing_pie_satisfies_hunger_is_impossible :
  impossible_event drawing_pie_to_satisfy_hunger := by sorry

end NUMINAMATH_CALUDE_drawing_pie_satisfies_hunger_is_impossible_l1428_142878


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1428_142807

theorem complex_equation_solution :
  ∀ (z : ℂ), z = Complex.I * (2 - z) → z = 1 + Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1428_142807


namespace NUMINAMATH_CALUDE_octal_is_smallest_l1428_142823

-- Define the base conversion function
def toDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

-- Define the given numbers
def binary : Nat := toDecimal [1, 0, 1, 0, 1, 0] 2
def quinary : Nat := toDecimal [1, 1, 1] 5
def octal : Nat := toDecimal [3, 2] 8
def senary : Nat := toDecimal [5, 4] 6

-- Theorem statement
theorem octal_is_smallest : 
  octal ≤ binary ∧ octal ≤ quinary ∧ octal ≤ senary :=
sorry

end NUMINAMATH_CALUDE_octal_is_smallest_l1428_142823


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_rotation_volume_l1428_142851

theorem isosceles_right_triangle_rotation_volume :
  ∀ (r h : ℝ), r = 1 → h = 1 →
  (1 / 3 : ℝ) * Real.pi * r^2 * h = Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_rotation_volume_l1428_142851


namespace NUMINAMATH_CALUDE_cube_sum_expression_l1428_142830

theorem cube_sum_expression (x y z w a b c d : ℝ) 
  (hxy : x * y = a)
  (hxz : x * z = b)
  (hyz : y * z = c)
  (hxw : x * w = d)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hz : z ≠ 0)
  (hw : w ≠ 0)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hd : d ≠ 0) :
  x^3 + y^3 + z^3 + w^3 = (a^3 * d^3 + a^3 * c^3 + b^3 * d^3 + d^3 * b^3) / (a * b * c * d) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_expression_l1428_142830


namespace NUMINAMATH_CALUDE_student_distribution_theorem_l1428_142889

/-- The number of ways to distribute n students to k towns, ensuring each town receives at least one student. -/
def distribute_students (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The number of ways to choose r items from n items. -/
def binomial_coefficient (n : ℕ) (r : ℕ) : ℕ :=
  sorry

/-- The number of permutations of n items. -/
def permutations (n : ℕ) : ℕ :=
  sorry

theorem student_distribution_theorem :
  distribute_students 4 3 = 36 :=
by sorry

end NUMINAMATH_CALUDE_student_distribution_theorem_l1428_142889


namespace NUMINAMATH_CALUDE_arithmetic_progression_squares_products_l1428_142863

/-- If a, b, and c form an arithmetic progression, then a^2 + ab + b^2, a^2 + ac + c^2, and b^2 + bc + c^2 form an arithmetic progression. -/
theorem arithmetic_progression_squares_products (a b c : ℝ) :
  (∃ d : ℝ, b = a + d ∧ c = a + 2*d) →
  ∃ q : ℝ, (a^2 + a*c + c^2) - (a^2 + a*b + b^2) = q ∧
           (b^2 + b*c + c^2) - (a^2 + a*c + c^2) = q :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_squares_products_l1428_142863


namespace NUMINAMATH_CALUDE_soccer_team_lineup_count_l1428_142829

def total_players : ℕ := 18
def goalie_needed : ℕ := 1
def field_players_needed : ℕ := 10

theorem soccer_team_lineup_count :
  (total_players.choose goalie_needed) * ((total_players - goalie_needed).choose field_players_needed) = 349864 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_lineup_count_l1428_142829


namespace NUMINAMATH_CALUDE_river_depth_problem_l1428_142861

theorem river_depth_problem (d k : ℝ) : 
  (d + 0.5 * d + k = 1.5 * (d + 0.5 * d)) →  -- Depth in mid-July is 1.5 times the depth at the end of May
  (1.5 * (d + 0.5 * d) = 45) →               -- Final depth in mid-July is 45 feet
  (d = 15 ∧ k = 11.25) :=                    -- Initial depth is 15 feet and depth increase in June is 11.25 feet
by sorry

end NUMINAMATH_CALUDE_river_depth_problem_l1428_142861


namespace NUMINAMATH_CALUDE_proportional_value_l1428_142868

-- Define the given ratio
def given_ratio : ℚ := 12 / 6

-- Define the conversion factor from minutes to seconds
def minutes_to_seconds : ℕ := 60

-- Define the target time in minutes
def target_time_minutes : ℕ := 8

-- Define the target time in seconds
def target_time_seconds : ℕ := target_time_minutes * minutes_to_seconds

-- State the theorem
theorem proportional_value :
  (given_ratio * target_time_seconds : ℚ) = 960 := by sorry

end NUMINAMATH_CALUDE_proportional_value_l1428_142868


namespace NUMINAMATH_CALUDE_trig_identity_l1428_142864

theorem trig_identity : 
  Real.sin (47 * π / 180) * Real.cos (17 * π / 180) - 
  Real.cos (47 * π / 180) * Real.cos (73 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1428_142864


namespace NUMINAMATH_CALUDE_square_minus_nine_l1428_142834

theorem square_minus_nine (x : ℤ) (h : x^2 = 1681) : (x + 3) * (x - 3) = 1672 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_nine_l1428_142834


namespace NUMINAMATH_CALUDE_expression_evaluation_l1428_142881

theorem expression_evaluation : 
  (-Real.sqrt 27 + Real.cos (30 * π / 180) - (π - Real.sqrt 2) ^ 0 + (-1/2)⁻¹) = 
  -(5 * Real.sqrt 3 + 6) / 2 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1428_142881


namespace NUMINAMATH_CALUDE_smallest_norm_u_l1428_142895

theorem smallest_norm_u (u : ℝ × ℝ) (h : ‖u + (4, 2)‖ = 10) :
  ∃ (v : ℝ × ℝ), ‖v‖ = 10 - 2 * Real.sqrt 5 ∧ ∀ (w : ℝ × ℝ), ‖w + (4, 2)‖ = 10 → ‖v‖ ≤ ‖w‖ := by
  sorry

end NUMINAMATH_CALUDE_smallest_norm_u_l1428_142895


namespace NUMINAMATH_CALUDE_fifteen_students_in_neither_l1428_142858

/-- Represents the number of students in different categories of a robotics club. -/
structure RoboticsClub where
  total : ℕ
  cs : ℕ
  electronics : ℕ
  both : ℕ

/-- Calculates the number of students taking neither computer science nor electronics. -/
def studentsInNeither (club : RoboticsClub) : ℕ :=
  club.total - (club.cs + club.electronics - club.both)

/-- Theorem stating that 15 students take neither computer science nor electronics. -/
theorem fifteen_students_in_neither (club : RoboticsClub)
  (h1 : club.total = 80)
  (h2 : club.cs = 52)
  (h3 : club.electronics = 38)
  (h4 : club.both = 25) :
  studentsInNeither club = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_students_in_neither_l1428_142858


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l1428_142812

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 6 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 6
  let area : ℝ := side_length^2 * Real.sqrt 3 / 4
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 / 2 := by
sorry


end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l1428_142812


namespace NUMINAMATH_CALUDE_average_speed_theorem_l1428_142837

theorem average_speed_theorem (total_distance : ℝ) (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) 
  (h1 : total_distance = 80)
  (h2 : distance1 = 30)
  (h3 : speed1 = 30)
  (h4 : distance2 = 50)
  (h5 : speed2 = 50)
  (h6 : total_distance = distance1 + distance2) :
  (total_distance) / ((distance1 / speed1) + (distance2 / speed2)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_theorem_l1428_142837


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1428_142899

theorem inequality_solution_set (x : ℝ) :
  (x^2 - |x| - 2 < 0) ↔ (-2 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1428_142899


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l1428_142897

theorem rectangular_box_volume (l w h : ℝ) 
  (face1 : l * w = 30)
  (face2 : w * h = 20)
  (face3 : l * h = 12) :
  l * w * h = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l1428_142897


namespace NUMINAMATH_CALUDE_paint_remaining_l1428_142835

theorem paint_remaining (initial_paint : ℚ) : 
  initial_paint > 0 → 
  (initial_paint - (initial_paint / 2) - ((initial_paint - (initial_paint / 2)) / 2)) / initial_paint = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_paint_remaining_l1428_142835


namespace NUMINAMATH_CALUDE_two_cyclists_problem_l1428_142816

/-- Two cyclists problem -/
theorem two_cyclists_problem (MP : ℝ) : 
  (∀ (t : ℝ), t > 0 → 
    (MP / t = 42 / ((420 / (MP + 30)) + 1/3)) ∧
    (MP + 30) / t = 42 / (420 / MP)) →
  MP = 180 := by
sorry

end NUMINAMATH_CALUDE_two_cyclists_problem_l1428_142816


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1428_142875

theorem unique_integer_solution :
  ∃! (a : ℤ), ∃ (d e : ℤ), ∀ (x : ℤ), (x - a) * (x - 8) - 3 = (x + d) * (x + e) ∧ a = 6 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1428_142875


namespace NUMINAMATH_CALUDE_sum_geq_sqrt_three_l1428_142844

theorem sum_geq_sqrt_three (a b c : ℝ) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (h : a * b + b * c + c * a = 1) : a + b + c ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_geq_sqrt_three_l1428_142844


namespace NUMINAMATH_CALUDE_solve_equation_l1428_142840

theorem solve_equation :
  ∃ x : ℚ, 5 * x + 9 * x = 350 - 10 * (x - 5) ∧ x = 50 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1428_142840


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1428_142890

theorem inequality_solution_set (a b m : ℝ) : 
  (∀ x, x^2 - a*x - 2 > 0 ↔ (x < -1 ∨ x > b)) →
  b > -1 →
  m > -1/2 →
  (a = 1 ∧ b = 2) ∧
  (
    (m > 0 → ∀ x, (m*x + a)*(x - b) > 0 ↔ (x < -1/m ∨ x > 2)) ∧
    (m = 0 → ∀ x, (m*x + a)*(x - b) > 0 ↔ x > 2) ∧
    (-1/2 < m ∧ m < 0 → ∀ x, (m*x + a)*(x - b) > 0 ↔ (2 < x ∧ x < -1/m))
  ) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1428_142890


namespace NUMINAMATH_CALUDE_smallest_satisfying_congruences_l1428_142886

theorem smallest_satisfying_congruences : 
  ∃ (x : ℕ), x > 0 ∧ 
    x % 3 = 2 ∧ 
    x % 5 = 3 ∧ 
    x % 7 = 2 ∧
    (∀ (y : ℕ), y > 0 ∧ y % 3 = 2 ∧ y % 5 = 3 ∧ y % 7 = 2 → x ≤ y) ∧
    x = 23 :=
by sorry

end NUMINAMATH_CALUDE_smallest_satisfying_congruences_l1428_142886


namespace NUMINAMATH_CALUDE_unique_solution_3x_4y_5z_l1428_142870

theorem unique_solution_3x_4y_5z :
  ∀ x y z : ℕ+, 3^(x : ℕ) + 4^(y : ℕ) = 5^(z : ℕ) → x = 2 ∧ y = 2 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_3x_4y_5z_l1428_142870


namespace NUMINAMATH_CALUDE_weekend_art_class_earnings_l1428_142801

/-- Calculates the total money earned over a weekend of art classes --/
def weekend_earnings (beginner_cost advanced_cost : ℕ)
  (saturday_beginner saturday_advanced : ℕ)
  (sibling_discount : ℕ) (sibling_pairs : ℕ) : ℕ :=
  let saturday_total := beginner_cost * saturday_beginner + advanced_cost * saturday_advanced
  let sunday_total := beginner_cost * (saturday_beginner / 2) + advanced_cost * (saturday_advanced / 2)
  let total_before_discount := saturday_total + sunday_total
  let total_discount := sibling_discount * (2 * sibling_pairs)
  total_before_discount - total_discount

/-- Theorem stating that the total earnings for the weekend is $720.00 --/
theorem weekend_art_class_earnings :
  weekend_earnings 15 20 20 10 3 5 = 720 := by
  sorry

end NUMINAMATH_CALUDE_weekend_art_class_earnings_l1428_142801


namespace NUMINAMATH_CALUDE_hiker_journey_distance_l1428_142842

/-- Represents the hiker's journey --/
structure HikerJourney where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The conditions of the hiker's journey --/
def journey_conditions (j : HikerJourney) : Prop :=
  j.distance = j.speed * j.time ∧
  j.distance = (j.speed + 1) * (3/4 * j.time) ∧
  j.distance = (j.speed - 1) * (j.time + 3)

/-- The theorem statement --/
theorem hiker_journey_distance :
  ∀ j : HikerJourney, journey_conditions j → j.distance = 90 := by
  sorry

end NUMINAMATH_CALUDE_hiker_journey_distance_l1428_142842


namespace NUMINAMATH_CALUDE_max_value_theorem_l1428_142877

theorem max_value_theorem (x : ℝ) (h : 0 < x ∧ x < 1) :
  ∃ (max_x : ℝ), max_x = 1/2 ∧
  ∀ y, 0 < y ∧ y < 1 → x * (3 - 3 * x) ≤ max_x * (3 - 3 * max_x) :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1428_142877


namespace NUMINAMATH_CALUDE_max_profit_l1428_142824

/-- Represents the shopping mall's helmet purchasing problem --/
structure HelmetProblem where
  costA : ℕ → ℕ  -- Cost function for type A helmets
  costB : ℕ → ℕ  -- Cost function for type B helmets
  sellA : ℕ      -- Selling price of type A helmet
  sellB : ℕ      -- Selling price of type B helmet
  totalHelmets : ℕ  -- Total number of helmets to purchase
  maxCost : ℕ    -- Maximum total cost
  minProfit : ℕ  -- Minimum required profit

/-- Calculates the profit for a given number of type A helmets --/
def profit (p : HelmetProblem) (numA : ℕ) : ℤ :=
  let numB := p.totalHelmets - numA
  (p.sellA - p.costA 1) * numA + (p.sellB - p.costB 1) * numB

/-- Theorem stating the maximum profit configuration --/
theorem max_profit (p : HelmetProblem) : 
  p.costA 8 + p.costB 6 = 630 →
  p.costA 6 + p.costB 8 = 700 →
  p.sellA = 58 →
  p.sellB = 98 →
  p.totalHelmets = 200 →
  p.maxCost = 10200 →
  p.minProfit = 6180 →
  p.costA 1 = 30 →
  p.costB 1 = 65 →
  (∀ n : ℕ, n ≤ p.totalHelmets → 
    p.costA n + p.costB (p.totalHelmets - n) ≤ p.maxCost →
    profit p n ≥ p.minProfit →
    profit p n ≤ profit p 80) ∧
  profit p 80 = 6200 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_l1428_142824


namespace NUMINAMATH_CALUDE_min_remote_uses_l1428_142808

/-- Represents the state of lamps --/
def LampState := Fin 169 → Bool

/-- The remote control operation --/
def remote_control (s : LampState) (switches : Finset (Fin 169)) : LampState :=
  λ i => if i ∈ switches then !s i else s i

/-- All lamps are initially on --/
def initial_state : LampState := λ _ => true

/-- All lamps are off --/
def all_off (s : LampState) : Prop := ∀ i, s i = false

/-- The remote control changes exactly 19 switches --/
def valid_remote_use (switches : Finset (Fin 169)) : Prop :=
  switches.card = 19

theorem min_remote_uses :
  ∃ (sequence : List (Finset (Fin 169))),
    sequence.length = 9 ∧
    (∀ switches ∈ sequence, valid_remote_use switches) ∧
    all_off (sequence.foldl remote_control initial_state) ∧
    (∀ (shorter_sequence : List (Finset (Fin 169))),
      shorter_sequence.length < 9 →
      (∀ switches ∈ shorter_sequence, valid_remote_use switches) →
      ¬ all_off (shorter_sequence.foldl remote_control initial_state)) :=
sorry

end NUMINAMATH_CALUDE_min_remote_uses_l1428_142808


namespace NUMINAMATH_CALUDE_smallest_dual_palindrome_l1428_142853

/-- Checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Bool := sorry

/-- Converts a number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_palindrome : 
  ∀ n : ℕ, n > 6 → (isPalindrome n 3 ∧ isPalindrome n 5) → n ≥ 26 := by
  sorry

#check smallest_dual_palindrome

end NUMINAMATH_CALUDE_smallest_dual_palindrome_l1428_142853


namespace NUMINAMATH_CALUDE_ellipse_k_range_l1428_142860

-- Define the ellipse equation
def is_ellipse (k : ℝ) : Prop :=
  ∃ (x y : ℝ), k * x^2 + 4 * y^2 = 4 * k

-- Define the condition for foci on x-axis
def foci_on_x_axis (k : ℝ) : Prop :=
  4 > k ∧ k > 0

-- Theorem statement
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse k ∧ foci_on_x_axis k ↔ 0 < k ∧ k < 4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l1428_142860


namespace NUMINAMATH_CALUDE_total_annual_interest_l1428_142850

def total_investment : ℕ := 3200
def first_part : ℕ := 800
def first_rate : ℚ := 3 / 100
def second_rate : ℚ := 5 / 100

def second_part : ℕ := total_investment - first_part

def interest_first : ℚ := (first_part : ℚ) * first_rate
def interest_second : ℚ := (second_part : ℚ) * second_rate

theorem total_annual_interest :
  interest_first + interest_second = 144 := by sorry

end NUMINAMATH_CALUDE_total_annual_interest_l1428_142850


namespace NUMINAMATH_CALUDE_parabola_sum_property_l1428_142891

/-- Represents a quadratic function of the form ax^2 + bx + c --/
structure Quadratic where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The function resulting from reflecting a quadratic about the y-axis --/
def reflect (q : Quadratic) : Quadratic :=
  { a := q.a, b := -q.b, c := q.c }

/-- Vertical translation of a quadratic function --/
def translate (q : Quadratic) (d : ℝ) : Quadratic :=
  { a := q.a, b := q.b, c := q.c + d }

/-- The sum of two quadratic functions --/
def add (q1 q2 : Quadratic) : Quadratic :=
  { a := q1.a + q2.a, b := q1.b + q2.b, c := q1.c + q2.c }

theorem parabola_sum_property (q : Quadratic) :
  let f := translate q 4
  let g := translate (reflect q) (-4)
  (add f g).b = 0 := by sorry

end NUMINAMATH_CALUDE_parabola_sum_property_l1428_142891


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l1428_142882

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 2*y^2 = 2

-- Define the line l
def line_l (x y : ℝ) (k₁ : ℝ) : Prop := y = k₁ * (x + 2)

-- Define the point M
def point_M : ℝ × ℝ := (-2, 0)

-- Define the origin O
def point_O : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem ellipse_line_intersection 
  (P₁ P₂ P : ℝ × ℝ) 
  (k₁ k₂ : ℝ) 
  (h₁ : k₁ ≠ 0)
  (h₂ : ellipse P₁.1 P₁.2)
  (h₃ : ellipse P₂.1 P₂.2)
  (h₄ : line_l P₁.1 P₁.2 k₁)
  (h₅ : line_l P₂.1 P₂.2 k₁)
  (h₆ : P = ((P₁.1 + P₂.1) / 2, (P₁.2 + P₂.2) / 2))
  (h₇ : k₂ = P.2 / P.1) :
  k₁ * k₂ = -1/2 := by sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l1428_142882


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l1428_142885

theorem hyperbolas_same_asymptotes (N : ℝ) :
  (∀ x y : ℝ, x^2 / 4 - y^2 / 9 = 1 ↔ y^2 / 18 - x^2 / N = 1) →
  (∀ k : ℝ, (∃ x y : ℝ, y = k * x ∧ x^2 / 4 - y^2 / 9 = 1) ↔
            (∃ x y : ℝ, y = k * x ∧ y^2 / 18 - x^2 / N = 1)) →
  N = 8 :=
by sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l1428_142885


namespace NUMINAMATH_CALUDE_fold_angle_is_36_degrees_l1428_142854

/-- The angle of fold that creates a regular decagon when a piece of paper is folded and cut,
    given that all vertices except one lie on a circle centered at that vertex,
    and the angle between adjacent vertices at the center is 144°. -/
def fold_angle_for_decagon : ℝ := sorry

/-- The internal angle of a regular decagon. -/
def decagon_internal_angle : ℝ := sorry

/-- Theorem stating that the fold angle for creating a regular decagon
    under the given conditions is 36°. -/
theorem fold_angle_is_36_degrees :
  fold_angle_for_decagon = 36 * (π / 180) ∧
  0 < fold_angle_for_decagon ∧
  fold_angle_for_decagon < π / 2 ∧
  decagon_internal_angle = 144 * (π / 180) :=
sorry

end NUMINAMATH_CALUDE_fold_angle_is_36_degrees_l1428_142854


namespace NUMINAMATH_CALUDE_sum_m_n_eq_67_l1428_142828

-- Define the point R
def R : ℝ × ℝ := (8, 6)

-- Define the lines
def line1 (x y : ℝ) : Prop := 8 * y = 15 * x
def line2 (x y : ℝ) : Prop := 10 * y = 3 * x

-- Define points P and Q
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- Define the conditions
axiom P_on_line1 : line1 P.1 P.2
axiom Q_on_line2 : line2 Q.1 Q.2
axiom R_is_midpoint : R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the length of PQ
def PQ_length : ℝ := sorry

-- Define m and n as positive integers
def m : ℕ+ := sorry
def n : ℕ+ := sorry

-- PQ length is equal to m/n
axiom PQ_length_eq_m_div_n : PQ_length = m.val / n.val

-- m and n are coprime
axiom m_n_coprime : Nat.Coprime m.val n.val

-- Theorem to prove
theorem sum_m_n_eq_67 : m.val + n.val = 67 := sorry

end NUMINAMATH_CALUDE_sum_m_n_eq_67_l1428_142828


namespace NUMINAMATH_CALUDE_square_sum_problem_l1428_142856

theorem square_sum_problem (a b c d : ℤ) (h : (a^2 + b^2) * (c^2 + d^2) = 1993) : a^2 + b^2 + c^2 + d^2 = 1994 := by
  sorry

-- Define 1993 as a prime number
def p : ℕ := 1993

axiom p_prime : Nat.Prime p

end NUMINAMATH_CALUDE_square_sum_problem_l1428_142856


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l1428_142805

-- Define the logarithm functions
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10
noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

-- State the theorem
theorem log_sum_equals_two : lg 0.01 + log2 16 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l1428_142805


namespace NUMINAMATH_CALUDE_green_blue_difference_after_border_l1428_142855

/-- Represents the number of tiles in a hexagonal figure --/
structure HexFigure where
  blue : ℕ
  green : ℕ

/-- Adds a double-layer border of green tiles to a hexagonal figure --/
def addDoubleBorder (fig : HexFigure) : HexFigure :=
  { blue := fig.blue,
    green := fig.green + 12 + 18 }

/-- The initial hexagonal figure --/
def initialFigure : HexFigure :=
  { blue := 20, green := 10 }

/-- Theorem stating the difference between green and blue tiles after adding a double border --/
theorem green_blue_difference_after_border :
  let newFigure := addDoubleBorder initialFigure
  (newFigure.green - newFigure.blue) = 20 := by
  sorry

end NUMINAMATH_CALUDE_green_blue_difference_after_border_l1428_142855


namespace NUMINAMATH_CALUDE_rationalize_denominator_seven_sqrt_147_l1428_142888

theorem rationalize_denominator_seven_sqrt_147 :
  ∃ (a b : ℝ) (h : b ≠ 0), (7 / Real.sqrt 147) = (a * Real.sqrt b) / b :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_seven_sqrt_147_l1428_142888


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1428_142883

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  interior_angle = 140 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1428_142883


namespace NUMINAMATH_CALUDE_exercise_minimum_sets_l1428_142845

/-- Represents the exercise routine over 100 days -/
structure ExerciseRoutine where
  pushups_per_set : ℕ
  pullups_per_set : ℕ
  initial_reps : ℕ
  days : ℕ

/-- Calculates the total number of repetitions over the given days -/
def total_reps (routine : ExerciseRoutine) : ℕ :=
  routine.days * (2 * routine.initial_reps + routine.days - 1) / 2

/-- Represents the solution to the exercise problem -/
structure ExerciseSolution where
  pushup_sets : ℕ
  pullup_sets : ℕ

/-- Theorem stating the minimum number of sets for push-ups and pull-ups -/
theorem exercise_minimum_sets (routine : ExerciseRoutine) 
  (h1 : routine.pushups_per_set = 8)
  (h2 : routine.pullups_per_set = 5)
  (h3 : routine.initial_reps = 41)
  (h4 : routine.days = 100) :
  ∃ (solution : ExerciseSolution), 
    solution.pushup_sets ≥ 100 ∧ 
    solution.pullup_sets ≥ 106 ∧
    solution.pushup_sets * routine.pushups_per_set + 
    solution.pullup_sets * routine.pullups_per_set = 
    total_reps routine :=
  sorry

end NUMINAMATH_CALUDE_exercise_minimum_sets_l1428_142845


namespace NUMINAMATH_CALUDE_candy_bar_calories_l1428_142876

theorem candy_bar_calories (total_calories : ℕ) (total_bars : ℕ) (h1 : total_calories = 2016) (h2 : total_bars = 42) :
  (total_calories / total_bars) / 12 = 4 :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_calories_l1428_142876


namespace NUMINAMATH_CALUDE_total_seashells_l1428_142872

theorem total_seashells (yesterday today : ℕ) 
  (h1 : yesterday = 7) 
  (h2 : today = 4) : 
  yesterday + today = 11 := by
sorry

end NUMINAMATH_CALUDE_total_seashells_l1428_142872


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1428_142867

/-- A quadratic equation x^2 + kx + 1 = 0 has two equal real roots if and only if k = ±2 -/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x + 1 = 0 ∧ (∀ y : ℝ, y^2 + k*y + 1 = 0 → y = x)) ↔ 
  k = 2 ∨ k = -2 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1428_142867


namespace NUMINAMATH_CALUDE_probability_of_X_selection_l1428_142880

theorem probability_of_X_selection (p_Y p_both : ℝ) 
  (h1 : p_Y = 2/7)
  (h2 : p_both = 0.09523809523809523)
  (h3 : p_both = p_Y * p_X)
  : p_X = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_X_selection_l1428_142880


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l1428_142820

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals : diagonals dodecagon_sides = 54 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l1428_142820


namespace NUMINAMATH_CALUDE_product_of_decimals_product_of_fractions_l1428_142804

/-- Proves that (-0.4) * (-0.8) * (-1.25) * 2.5 = -1 -/
theorem product_of_decimals : (-0.4) * (-0.8) * (-1.25) * 2.5 = -1 := by
  sorry

/-- Proves that (-5/8) * (3/14) * (-16/5) * (-7/6) = -1/2 -/
theorem product_of_fractions : (-5/8 : ℚ) * (3/14 : ℚ) * (-16/5 : ℚ) * (-7/6 : ℚ) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_decimals_product_of_fractions_l1428_142804


namespace NUMINAMATH_CALUDE_white_washing_cost_is_4530_l1428_142806

/-- Calculates the cost of white washing a room with given dimensions and openings -/
def white_washing_cost (room_length room_width room_height : ℝ)
                       (door_length door_width : ℝ)
                       (window_length window_width : ℝ)
                       (num_windows : ℕ)
                       (cost_per_sqft : ℝ) : ℝ :=
  let wall_area := 2 * (room_length + room_width) * room_height
  let door_area := door_length * door_width
  let window_area := window_length * window_width * num_windows
  let paintable_area := wall_area - door_area - window_area
  paintable_area * cost_per_sqft

/-- The cost of white washing the room is 4530 rupees -/
theorem white_washing_cost_is_4530 :
  white_washing_cost 25 15 12 6 3 4 3 3 5 = 4530 := by
  sorry

end NUMINAMATH_CALUDE_white_washing_cost_is_4530_l1428_142806


namespace NUMINAMATH_CALUDE_transformations_of_g_reflection_about_y_axis_horizontal_stretch_horizontal_shift_l1428_142814

-- Define the original function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the transformed function g
noncomputable def g (x : ℝ) : ℝ := f ((1 - x) / 2)

-- Theorem stating the transformations
theorem transformations_of_g (x : ℝ) :
  g x = f (-(x - 1) / 2) :=
sorry

-- Theorem stating the reflection about y-axis
theorem reflection_about_y_axis (x : ℝ) :
  g (-x + 1) = f x :=
sorry

-- Theorem stating the horizontal stretch
theorem horizontal_stretch (x : ℝ) :
  g (2 * x + 1) = f x :=
sorry

-- Theorem stating the horizontal shift
theorem horizontal_shift (x : ℝ) :
  g (x + 1) = f (x / 2) :=
sorry

end NUMINAMATH_CALUDE_transformations_of_g_reflection_about_y_axis_horizontal_stretch_horizontal_shift_l1428_142814


namespace NUMINAMATH_CALUDE_edward_earnings_l1428_142815

/-- Calculate Edward's earnings for a week --/
theorem edward_earnings (regular_rate : ℝ) (regular_hours overtime_hours : ℕ) : 
  regular_rate = 7 →
  regular_hours = 40 →
  overtime_hours = 5 →
  let overtime_rate := 2 * regular_rate
  let regular_earnings := regular_rate * regular_hours
  let overtime_earnings := overtime_rate * overtime_hours
  regular_earnings + overtime_earnings = 350 := by sorry

end NUMINAMATH_CALUDE_edward_earnings_l1428_142815


namespace NUMINAMATH_CALUDE_eleventh_term_of_geometric_sequence_l1428_142802

def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem eleventh_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_fifth : a 5 = 5)
  (h_eighth : a 8 = 40) :
  a 11 = 320 := by
sorry

end NUMINAMATH_CALUDE_eleventh_term_of_geometric_sequence_l1428_142802


namespace NUMINAMATH_CALUDE_average_headcount_rounded_l1428_142810

def fall_03_04_headcount : ℕ := 11500
def fall_04_05_headcount : ℕ := 11600
def fall_05_06_headcount : ℕ := 11300

def average_headcount : ℚ := (fall_03_04_headcount + fall_04_05_headcount + fall_05_06_headcount) / 3

def round_to_nearest (x : ℚ) : ℕ := 
  (x + 1/2).floor.toNat

theorem average_headcount_rounded : round_to_nearest average_headcount = 11467 := by
  sorry

end NUMINAMATH_CALUDE_average_headcount_rounded_l1428_142810


namespace NUMINAMATH_CALUDE_benny_seashells_l1428_142813

/-- Proves that the initial number of seashells Benny found is equal to the number of seashells he has now plus the number of seashells he gave away. -/
theorem benny_seashells (seashells_now : ℕ) (seashells_given : ℕ) 
  (h1 : seashells_now = 14) 
  (h2 : seashells_given = 52) : 
  seashells_now + seashells_given = 66 := by
  sorry

#check benny_seashells

end NUMINAMATH_CALUDE_benny_seashells_l1428_142813


namespace NUMINAMATH_CALUDE_tired_painting_time_l1428_142843

/-- Represents the time needed to paint houses -/
def paint_time (people : ℕ) (houses : ℕ) (hours : ℝ) (efficiency : ℝ) : Prop :=
  people * hours * efficiency = houses * 32

theorem tired_painting_time :
  paint_time 8 2 4 1 →
  paint_time 5 2 8 0.8 :=
by
  sorry

end NUMINAMATH_CALUDE_tired_painting_time_l1428_142843


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_l1428_142846

theorem sqrt_three_irrational :
  (∃ (a b : ℤ), -2 = a / b) →
  (∃ (c d : ℤ), 0 = c / d) →
  (∃ (e f : ℤ), -1/2 = e / f) →
  ¬(∃ (x y : ℤ), Real.sqrt 3 = x / y) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_l1428_142846
