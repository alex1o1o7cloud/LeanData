import Mathlib

namespace NUMINAMATH_CALUDE_collin_cans_at_home_l2088_208842

/-- The number of cans Collin found at home -/
def cans_at_home : ℕ := sorry

/-- The amount earned per can in cents -/
def cents_per_can : ℕ := 25

/-- The number of cans from the neighbor -/
def cans_from_neighbor : ℕ := 46

/-- The number of cans from dad's office -/
def cans_from_office : ℕ := 250

/-- The amount Collin has to put into savings in cents -/
def savings_amount : ℕ := 4300

theorem collin_cans_at_home :
  cans_at_home = 12 ∧
  cents_per_can * (cans_at_home + 3 * cans_at_home + cans_from_neighbor + cans_from_office) = 2 * savings_amount :=
by sorry

end NUMINAMATH_CALUDE_collin_cans_at_home_l2088_208842


namespace NUMINAMATH_CALUDE_grape_juice_solution_l2088_208832

/-- Represents the problem of adding grape juice to a mixture --/
def GrapeJuiceProblem (initial_volume : ℝ) (initial_concentration : ℝ) (final_concentration : ℝ) (added_juice : ℝ) : Prop :=
  let final_volume := initial_volume + added_juice
  let initial_juice := initial_volume * initial_concentration
  let final_juice := final_volume * final_concentration
  final_juice = initial_juice + added_juice

/-- Theorem stating the solution to the grape juice problem --/
theorem grape_juice_solution :
  GrapeJuiceProblem 30 0.1 0.325 10 := by
  sorry

end NUMINAMATH_CALUDE_grape_juice_solution_l2088_208832


namespace NUMINAMATH_CALUDE_both_correct_percentage_l2088_208806

-- Define the percentages as real numbers between 0 and 1
def first_correct : ℝ := 0.80
def second_correct : ℝ := 0.75
def neither_correct : ℝ := 0.05

-- Theorem to prove
theorem both_correct_percentage :
  first_correct + second_correct - (1 - neither_correct) = 0.60 := by
  sorry

end NUMINAMATH_CALUDE_both_correct_percentage_l2088_208806


namespace NUMINAMATH_CALUDE_rectangular_field_dimensions_l2088_208878

theorem rectangular_field_dimensions (m : ℝ) : 
  m > 3 ∧ (2 * m + 9) * (m - 3) = 55 →
  m = (-3 + Real.sqrt 665) / 4 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_dimensions_l2088_208878


namespace NUMINAMATH_CALUDE_accident_rate_calculation_l2088_208810

theorem accident_rate_calculation (total_vehicles : ℕ) (accident_vehicles : ℕ) 
  (rate_vehicles : ℕ) (rate_accidents : ℕ) :
  total_vehicles = 3000000000 →
  accident_vehicles = 2880 →
  rate_accidents = 96 →
  (rate_accidents : ℚ) / rate_vehicles = (accident_vehicles : ℚ) / total_vehicles →
  rate_vehicles = 100000000 := by
sorry

end NUMINAMATH_CALUDE_accident_rate_calculation_l2088_208810


namespace NUMINAMATH_CALUDE_ferry_distance_ratio_l2088_208804

/-- The ratio of the distance covered by ferry Q to the distance covered by ferry P -/
theorem ferry_distance_ratio :
  let speed_p : ℝ := 8
  let time_p : ℝ := 3
  let speed_q : ℝ := speed_p + 1
  let time_q : ℝ := time_p + 5
  let distance_p : ℝ := speed_p * time_p
  let distance_q : ℝ := speed_q * time_q
  (distance_q / distance_p : ℝ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ferry_distance_ratio_l2088_208804


namespace NUMINAMATH_CALUDE_sin_70_cos_20_plus_cos_70_sin_20_l2088_208846

theorem sin_70_cos_20_plus_cos_70_sin_20 : 
  Real.sin (70 * π / 180) * Real.cos (20 * π / 180) + 
  Real.cos (70 * π / 180) * Real.sin (20 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_70_cos_20_plus_cos_70_sin_20_l2088_208846


namespace NUMINAMATH_CALUDE_unique_solution_iff_sqrt_three_l2088_208863

/-- The function f(x) = x^2 + a|x| + a^2 - 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * |x| + a^2 - 3

/-- The theorem stating that the equation f(x) = 0 has a unique real solution iff a = √3 -/
theorem unique_solution_iff_sqrt_three (a : ℝ) :
  (∃! x : ℝ, f a x = 0) ↔ a = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_unique_solution_iff_sqrt_three_l2088_208863


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l2088_208833

/-- Given a line symmetric to 4x - 3y + 5 = 0 with respect to the y-axis, prove its equation is 4x + 3y - 5 = 0 -/
theorem symmetric_line_equation : 
  ∃ (l : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ l ↔ (-x, y) ∈ {(x, y) | 4*x - 3*y + 5 = 0}) → 
    (∀ (x y : ℝ), (x, y) ∈ l ↔ 4*x + 3*y - 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l2088_208833


namespace NUMINAMATH_CALUDE_davids_math_marks_l2088_208802

theorem davids_math_marks (english : ℕ) (physics : ℕ) (chemistry : ℕ) (biology : ℕ) (average : ℕ) (total_subjects : ℕ) :
  english = 96 →
  physics = 82 →
  chemistry = 97 →
  biology = 95 →
  average = 93 →
  total_subjects = 5 →
  (english + physics + chemistry + biology + (average * total_subjects - (english + physics + chemistry + biology))) / total_subjects = average :=
by sorry

end NUMINAMATH_CALUDE_davids_math_marks_l2088_208802


namespace NUMINAMATH_CALUDE_line_through_points_l2088_208883

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define a line by its equation coefficients (ax + by + c = 0)
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

-- Theorem statement
theorem line_through_points : 
  ∃ (l : Line), 
    point_on_line (-3, 0) l ∧ 
    point_on_line (0, 4) l ∧ 
    l.a = 4 ∧ l.b = -3 ∧ l.c = 12 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l2088_208883


namespace NUMINAMATH_CALUDE_coordinate_points_count_l2088_208849

theorem coordinate_points_count (S : Finset ℕ) (h : S = {1, 2, 3, 4, 5}) :
  Finset.card (Finset.product S S) = 25 := by
  sorry

end NUMINAMATH_CALUDE_coordinate_points_count_l2088_208849


namespace NUMINAMATH_CALUDE_range_of_m_l2088_208813

/-- The function f(x) = -x^2 + 2x + 5 -/
def f (x : ℝ) : ℝ := -x^2 + 2*x + 5

/-- Theorem stating the range of m given the conditions on f -/
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ≤ 6) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 6) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ 5) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 5) →
  m ∈ Set.Icc 1 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2088_208813


namespace NUMINAMATH_CALUDE_exists_sum_of_digits_div_by_11_l2088_208861

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that in any 39 consecutive natural numbers, 
    there is at least one whose sum of digits is divisible by 11 -/
theorem exists_sum_of_digits_div_by_11 (n : ℕ) : 
  ∃ k : ℕ, n ≤ k ∧ k ≤ n + 38 ∧ (sum_of_digits k) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_sum_of_digits_div_by_11_l2088_208861


namespace NUMINAMATH_CALUDE_internet_fee_calculation_l2088_208830

/-- The fixed monthly fee for Anna's internet service -/
def fixed_fee : ℝ := sorry

/-- The variable fee per hour of usage for Anna's internet service -/
def variable_fee : ℝ := sorry

/-- Anna's internet usage in November (in hours) -/
def november_usage : ℝ := sorry

/-- Anna's bill for November -/
def november_bill : ℝ := 20.60

/-- Anna's bill for December -/
def december_bill : ℝ := 33.20

theorem internet_fee_calculation :
  (fixed_fee + variable_fee * november_usage = november_bill) ∧
  (fixed_fee + variable_fee * (3 * november_usage) = december_bill) →
  fixed_fee = 14.30 := by
sorry

end NUMINAMATH_CALUDE_internet_fee_calculation_l2088_208830


namespace NUMINAMATH_CALUDE_spellbook_cost_l2088_208869

/-- Proves that each spellbook costs 5 gold given the conditions of Harry's purchase --/
theorem spellbook_cost (num_spellbooks : ℕ) (num_potion_kits : ℕ) (owl_cost_gold : ℕ) 
  (potion_kit_cost_silver : ℕ) (silver_per_gold : ℕ) (total_cost_silver : ℕ) :
  num_spellbooks = 5 →
  num_potion_kits = 3 →
  owl_cost_gold = 28 →
  potion_kit_cost_silver = 20 →
  silver_per_gold = 9 →
  total_cost_silver = 537 →
  (total_cost_silver - (owl_cost_gold * silver_per_gold + num_potion_kits * potion_kit_cost_silver)) / num_spellbooks / silver_per_gold = 5 := by
  sorry

#check spellbook_cost

end NUMINAMATH_CALUDE_spellbook_cost_l2088_208869


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2088_208809

theorem inequality_equivalence (x : ℝ) : 
  |2*x - 1| < |x| + 1 ↔ 0 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2088_208809


namespace NUMINAMATH_CALUDE_book_reading_ratio_l2088_208837

/-- Given the number of books read by Candice, Amanda, Kara, and Patricia in a Book Tournament, 
    prove the ratio of books read by Kara to Amanda. -/
theorem book_reading_ratio 
  (candice amanda kara patricia : ℕ) 
  (x : ℚ) 
  (h1 : candice = 3 * amanda) 
  (h2 : candice = 18) 
  (h3 : kara = x * amanda) 
  (h4 : patricia = 7 * kara) : 
  (kara : ℚ) / amanda = x := by
  sorry

end NUMINAMATH_CALUDE_book_reading_ratio_l2088_208837


namespace NUMINAMATH_CALUDE_lidia_apps_to_buy_l2088_208812

-- Define the given conditions
def average_app_cost : ℕ := 4
def total_budget : ℕ := 66
def remaining_money : ℕ := 6

-- Define the number of apps to buy
def apps_to_buy : ℕ := (total_budget - remaining_money) / average_app_cost

-- Theorem statement
theorem lidia_apps_to_buy : apps_to_buy = 15 := by
  sorry

end NUMINAMATH_CALUDE_lidia_apps_to_buy_l2088_208812


namespace NUMINAMATH_CALUDE_essay_word_limit_l2088_208807

/-- The word limit for Vinnie's essay --/
def word_limit (saturday_words sunday_words exceeded_words : ℕ) : ℕ :=
  saturday_words + sunday_words - exceeded_words

/-- Theorem: The word limit for Vinnie's essay is 1000 words --/
theorem essay_word_limit :
  word_limit 450 650 100 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_essay_word_limit_l2088_208807


namespace NUMINAMATH_CALUDE_rectangle_ratio_l2088_208811

theorem rectangle_ratio (l w : ℝ) (hl : l = 10) (hp : 2 * l + 2 * w = 36) :
  w / l = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l2088_208811


namespace NUMINAMATH_CALUDE_local_extremum_values_l2088_208808

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 - b*x + a^2

-- State the theorem
theorem local_extremum_values (a b : ℝ) :
  (f a b 1 = 10) ∧ 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b x ≤ f a b 1) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b x ≥ f a b 1) →
  a = -4 ∧ b = 11 := by
  sorry

end NUMINAMATH_CALUDE_local_extremum_values_l2088_208808


namespace NUMINAMATH_CALUDE_binomial_55_3_l2088_208820

theorem binomial_55_3 : Nat.choose 55 3 = 26235 := by
  sorry

end NUMINAMATH_CALUDE_binomial_55_3_l2088_208820


namespace NUMINAMATH_CALUDE_circle_op_eq_power_l2088_208857

noncomputable def circle_op (a : ℚ) (n : ℕ) : ℚ :=
  if n = 0 then 1 else if n = 1 then a else a / (circle_op a (n - 1))

theorem circle_op_eq_power (a : ℚ) (n : ℕ) (h : a ≠ 0) :
  circle_op a n = (1 / a) ^ (n - 2) :=
sorry

end NUMINAMATH_CALUDE_circle_op_eq_power_l2088_208857


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2088_208851

/-- Given that the solution set of ax² + bx + c ≤ 0 is {x | x ≤ -4 ∨ x ≥ 3}, 
    prove that a + b + c > 0 and that bx + c > 0 has the solution set {x | x < 12} -/
theorem quadratic_inequality_solution (a b c : ℝ) 
  (h : ∀ x, ax^2 + b*x + c ≤ 0 ↔ x ≤ -4 ∨ x ≥ 3) : 
  (a + b + c > 0) ∧ (∀ x, b*x + c > 0 ↔ x < 12) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2088_208851


namespace NUMINAMATH_CALUDE_mans_rate_in_still_water_l2088_208884

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate_in_still_water
  (speed_with_stream : ℝ)
  (speed_against_stream : ℝ)
  (h1 : speed_with_stream = 26)
  (h2 : speed_against_stream = 12) :
  (speed_with_stream + speed_against_stream) / 2 = 19 := by
  sorry

end NUMINAMATH_CALUDE_mans_rate_in_still_water_l2088_208884


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2088_208854

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |2*x + 3| - |x - 1| = 4*x - 3 :=
by
  -- The unique solution is 7/3
  use 7/3
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2088_208854


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l2088_208858

theorem simultaneous_equations_solution (m : ℝ) : 
  ∃ (x y : ℝ), y = 3 * m * x + 5 ∧ y = (3 * m - 2) * x + 7 :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l2088_208858


namespace NUMINAMATH_CALUDE_tuesday_extra_minutes_l2088_208885

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of minutes Ayen jogs on a regular weekday -/
def regular_jog : ℕ := 30

/-- The number of extra minutes Ayen jogged on Friday -/
def friday_extra : ℕ := 25

/-- The total number of minutes Ayen jogged this week -/
def total_jog : ℕ := 3 * 60

/-- The number of extra minutes Ayen jogged on Tuesday -/
def tuesday_extra : ℕ := total_jog - (weekdays * regular_jog) - friday_extra

theorem tuesday_extra_minutes : tuesday_extra = 5 := by sorry

end NUMINAMATH_CALUDE_tuesday_extra_minutes_l2088_208885


namespace NUMINAMATH_CALUDE_smallest_solution_floor_square_diff_l2088_208843

theorem smallest_solution_floor_square_diff (x : ℝ) :
  (∀ y : ℝ, y < x → ⌊y^2⌋ - ⌊y⌋^2 ≠ 19) ∧ ⌊x^2⌋ - ⌊x⌋^2 = 19 ↔ x = Real.sqrt 104 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_square_diff_l2088_208843


namespace NUMINAMATH_CALUDE_weekly_average_expenditure_l2088_208826

/-- The average expenditure for a week given the average expenditures for two parts of the week -/
theorem weekly_average_expenditure 
  (first_three_days_avg : ℝ) 
  (next_four_days_avg : ℝ) 
  (h1 : first_three_days_avg = 350)
  (h2 : next_four_days_avg = 420) :
  (3 * first_three_days_avg + 4 * next_four_days_avg) / 7 = 390 := by
sorry

end NUMINAMATH_CALUDE_weekly_average_expenditure_l2088_208826


namespace NUMINAMATH_CALUDE_solve_for_y_l2088_208894

theorem solve_for_y (x y : ℝ) (h1 : 3 * x + 2 = 2) (h2 : y - x = 2) : y = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2088_208894


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2088_208800

theorem functional_equation_solution (f : ℝ → ℝ) (a : ℝ) 
  (h : ∀ x y : ℝ, f x * f y - a * f (x * y) = x + y) : 
  a = 1 ∨ a = -1 := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2088_208800


namespace NUMINAMATH_CALUDE_equation_B_positive_correlation_l2088_208868

/-- Represents an empirical regression equation of the form ŷ = ax + b -/
structure RegressionEquation where
  a : ℝ  -- Coefficient of x
  b : ℝ  -- y-intercept

/-- Defines what it means for a regression equation to show positive correlation -/
def shows_positive_correlation (eq : RegressionEquation) : Prop :=
  eq.a > 0

/-- The specific regression equation we're interested in -/
def equation_B : RegressionEquation :=
  { a := 1.2, b := 1.5 }

/-- Theorem stating that equation B shows a positive correlation -/
theorem equation_B_positive_correlation :
  shows_positive_correlation equation_B := by
  sorry


end NUMINAMATH_CALUDE_equation_B_positive_correlation_l2088_208868


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l2088_208886

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 7 / 5)
  (hdb : d / b = 1 / 9) :
  a / c = 112.5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l2088_208886


namespace NUMINAMATH_CALUDE_opposite_sqrt_81_l2088_208856

theorem opposite_sqrt_81 : -(Real.sqrt 81) = -9 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sqrt_81_l2088_208856


namespace NUMINAMATH_CALUDE_quadratic_not_in_third_quadrant_l2088_208853

/-- A linear function passing through the first, third, and fourth quadrants -/
structure LinearFunction where
  a : ℝ
  b : ℝ
  a_nonzero : a ≠ 0
  passes_first_quadrant : ∃ x > 0, -a * x + b > 0
  passes_third_quadrant : ∃ x < 0, -a * x + b < 0
  passes_fourth_quadrant : ∃ x > 0, -a * x + b < 0

/-- The corresponding quadratic function -/
def quadratic_function (f : LinearFunction) (x : ℝ) : ℝ :=
  -f.a * x^2 + f.b * x

/-- Theorem stating that the quadratic function does not pass through the third quadrant -/
theorem quadratic_not_in_third_quadrant (f : LinearFunction) :
  ¬∃ x < 0, quadratic_function f x < 0 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_not_in_third_quadrant_l2088_208853


namespace NUMINAMATH_CALUDE_range_of_a_l2088_208814

def P (a : ℝ) : Set ℝ := {x | a - 4 < x ∧ x < a + 4}
def Q : Set ℝ := {x | x^2 - 4*x + 3 < 0}

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Q, x ∈ P a) → -1 < a ∧ a < 5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2088_208814


namespace NUMINAMATH_CALUDE_min_triangle_perimeter_l2088_208855

theorem min_triangle_perimeter (a b x : ℕ) (ha : a = 24) (hb : b = 51) : 
  (a + b + x > a + b ∧ a + x > b ∧ b + x > a) → (∀ y : ℕ, (a + b + y > a + b ∧ a + y > b ∧ b + y > a) → x ≤ y) 
  → a + b + x = 103 :=
sorry

end NUMINAMATH_CALUDE_min_triangle_perimeter_l2088_208855


namespace NUMINAMATH_CALUDE_midpoint_distance_squared_l2088_208876

/-- A rectangle ABCD with given dimensions and midpoints -/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  h_rectangle : A.1 = D.1 ∧ B.1 = C.1 ∧ A.2 = B.2 ∧ C.2 = D.2
  h_AB : (B.1 - A.1)^2 + (B.2 - A.2)^2 = 15^2
  h_BC : (C.1 - B.1)^2 + (C.2 - B.2)^2 = 8^2
  h_right_angle : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0
  h_X_midpoint : X = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  h_Y_midpoint : Y = ((D.1 + A.1) / 2, (D.2 + A.2) / 2)

/-- The square of the distance between midpoints X and Y is 64 -/
theorem midpoint_distance_squared (r : Rectangle) : 
  (r.X.1 - r.Y.1)^2 + (r.X.2 - r.Y.2)^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_distance_squared_l2088_208876


namespace NUMINAMATH_CALUDE_forest_tree_ratio_l2088_208852

/-- Proves the ratio of trees after Monday to initial trees is 3:1 --/
theorem forest_tree_ratio : 
  ∀ (initial_trees monday_trees : ℕ),
    initial_trees = 30 →
    monday_trees + (monday_trees / 3) = 80 →
    (initial_trees + monday_trees) / initial_trees = 3 := by
  sorry

end NUMINAMATH_CALUDE_forest_tree_ratio_l2088_208852


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2088_208865

theorem sqrt_inequality (x : ℝ) :
  3 - x ≥ 0 → x + 1 ≥ 0 →
  (Real.sqrt (3 - x) - Real.sqrt (x + 1) > 1 / 2 ↔ -1 ≤ x ∧ x < 1 - Real.sqrt 31 / 8) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2088_208865


namespace NUMINAMATH_CALUDE_min_value_of_a_is_two_l2088_208817

/-- Given an equation with parameter a and two real solutions, 
    prove that the minimum value of a is 2 -/
theorem min_value_of_a_is_two (a : ℝ) (x₁ x₂ : ℝ) : 
  (9 * x₁ - (4 + a) * 3 * x₁ + 4 = 0) ∧ 
  (9 * x₂ - (4 + a) * 3 * x₂ + 4 = 0) ∧ 
  (x₁ ≠ x₂) →
  ∀ b : ℝ, (∃ y₁ y₂ : ℝ, (9 * y₁ - (4 + b) * 3 * y₁ + 4 = 0) ∧ 
                         (9 * y₂ - (4 + b) * 3 * y₂ + 4 = 0) ∧ 
                         (y₁ ≠ y₂)) →
  b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_a_is_two_l2088_208817


namespace NUMINAMATH_CALUDE_dice_remainder_prob_l2088_208891

/-- The probability of getting a specific remainder when the sum of two dice is divided by 4 -/
def remainder_probability (r : Fin 4) : ℚ := sorry

/-- The sum of all probabilities should be 1 -/
axiom prob_sum_one : remainder_probability 0 + remainder_probability 1 + remainder_probability 2 + remainder_probability 3 = 1

/-- The probabilities are non-negative -/
axiom prob_non_negative (r : Fin 4) : remainder_probability r ≥ 0

theorem dice_remainder_prob :
  2 * remainder_probability 3 - 3 * remainder_probability 2 + remainder_probability 1 - remainder_probability 0 = -2/9 := by
  sorry

end NUMINAMATH_CALUDE_dice_remainder_prob_l2088_208891


namespace NUMINAMATH_CALUDE_polygon_sides_l2088_208896

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 = 2 * 360 + 180 → n = 7 :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l2088_208896


namespace NUMINAMATH_CALUDE_positive_expression_l2088_208801

theorem positive_expression (a b : ℝ) (h : a ≠ 0 ∨ b ≠ 0) :
  5 * a^2 - 6 * a * b + 5 * b^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l2088_208801


namespace NUMINAMATH_CALUDE_inequality_proof_l2088_208895

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (2 * x^2) / (y + z) + (2 * y^2) / (z + x) + (2 * z^2) / (x + y) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2088_208895


namespace NUMINAMATH_CALUDE_event_probability_range_l2088_208859

/-- The probability of event A occurring in a single trial -/
def p : ℝ := sorry

/-- The number of independent trials -/
def n : ℕ := 4

/-- The probability of event A occurring exactly k times in n trials -/
def prob_k (k : ℕ) : ℝ := sorry

theorem event_probability_range :
  (0 ≤ p ∧ p ≤ 1) →  -- Probability is between 0 and 1
  (prob_k 1 ≤ prob_k 2) →  -- Probability of occurring once ≤ probability of occurring twice
  (2/5 ≤ p ∧ p ≤ 1) :=  -- The range of probability p is [2/5, 1]
sorry

end NUMINAMATH_CALUDE_event_probability_range_l2088_208859


namespace NUMINAMATH_CALUDE_cubic_solution_sum_l2088_208873

theorem cubic_solution_sum (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a = 12 →
  b^3 - 6*b^2 + 11*b = 12 →
  c^3 - 6*c^2 + 11*c = 12 →
  a * b / c + b * c / a + c * a / b = -23 / 12 :=
by sorry

end NUMINAMATH_CALUDE_cubic_solution_sum_l2088_208873


namespace NUMINAMATH_CALUDE_coffee_order_total_cost_l2088_208824

def drip_coffee_price : ℝ := 2.25
def drip_coffee_quantity : ℕ := 2

def espresso_price : ℝ := 3.50
def espresso_quantity : ℕ := 1

def latte_price : ℝ := 4.00
def latte_quantity : ℕ := 2

def vanilla_syrup_price : ℝ := 0.50
def vanilla_syrup_quantity : ℕ := 1

def cold_brew_price : ℝ := 2.50
def cold_brew_quantity : ℕ := 2

def cappuccino_price : ℝ := 3.50
def cappuccino_quantity : ℕ := 1

theorem coffee_order_total_cost :
  drip_coffee_price * drip_coffee_quantity +
  espresso_price * espresso_quantity +
  latte_price * latte_quantity +
  vanilla_syrup_price * vanilla_syrup_quantity +
  cold_brew_price * cold_brew_quantity +
  cappuccino_price * cappuccino_quantity = 25.00 := by
  sorry

end NUMINAMATH_CALUDE_coffee_order_total_cost_l2088_208824


namespace NUMINAMATH_CALUDE_hard_lens_price_l2088_208888

/-- Represents the price of contact lenses and sales information -/
structure LensSales where
  soft_price : ℕ
  hard_price : ℕ
  soft_count : ℕ
  hard_count : ℕ
  total_sales : ℕ

/-- Theorem stating the price of hard contact lenses -/
theorem hard_lens_price (sales : LensSales) : 
  sales.soft_price = 150 ∧ 
  sales.soft_count = sales.hard_count + 5 ∧
  sales.soft_count + sales.hard_count = 11 ∧
  sales.total_sales = sales.soft_price * sales.soft_count + sales.hard_price * sales.hard_count ∧
  sales.total_sales = 1455 →
  sales.hard_price = 85 := by
sorry

end NUMINAMATH_CALUDE_hard_lens_price_l2088_208888


namespace NUMINAMATH_CALUDE_girls_count_l2088_208882

theorem girls_count (boys girls : ℕ) : 
  (boys : ℚ) / girls = 8 / 5 →
  boys + girls = 351 →
  girls = 135 := by
sorry

end NUMINAMATH_CALUDE_girls_count_l2088_208882


namespace NUMINAMATH_CALUDE_triangle_is_isosceles_triangle_area_l2088_208838

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def isIsosceles (t : Triangle) : Prop :=
  t.b * Real.cos t.C = t.a * (Real.cos t.B)^2 + t.b * Real.cos t.A * Real.cos t.B

def hasSpecificProperties (t : Triangle) : Prop :=
  isIsosceles t ∧ Real.cos t.A = 7/8 ∧ t.a + t.b + t.c = 5

-- State the theorems
theorem triangle_is_isosceles (t : Triangle) (h : isIsosceles t) : 
  t.B = t.C := by sorry

theorem triangle_area (t : Triangle) (h : hasSpecificProperties t) :
  (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 15 / 4 := by sorry

end NUMINAMATH_CALUDE_triangle_is_isosceles_triangle_area_l2088_208838


namespace NUMINAMATH_CALUDE_min_balls_for_five_same_color_l2088_208864

/-- Given a bag with 10 red balls, 10 yellow balls, and 10 white balls,
    the minimum number of balls that must be drawn to ensure
    at least 5 balls of the same color is 13. -/
theorem min_balls_for_five_same_color (red yellow white : ℕ) 
  (h_red : red = 10) (h_yellow : yellow = 10) (h_white : white = 10) :
  ∃ (n : ℕ), n = 13 ∧ 
  ∀ (m : ℕ), m < n → 
  ∃ (r y w : ℕ), r + y + w = m ∧ r < 5 ∧ y < 5 ∧ w < 5 :=
sorry

end NUMINAMATH_CALUDE_min_balls_for_five_same_color_l2088_208864


namespace NUMINAMATH_CALUDE_inequality_solution_l2088_208841

theorem inequality_solution (x : ℝ) : (x^2 - 49) / (x + 7) < 0 ↔ -7 < x ∧ x < 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2088_208841


namespace NUMINAMATH_CALUDE_club_leadership_combinations_l2088_208877

/-- Represents the total number of students in the club -/
def total_students : ℕ := 30

/-- Represents the number of boys in the club -/
def num_boys : ℕ := 18

/-- Represents the number of girls in the club -/
def num_girls : ℕ := 12

/-- Represents the number of boy seniors (equal to boy juniors) -/
def num_boy_seniors : ℕ := num_boys / 2

/-- Represents the number of girl seniors (equal to girl juniors) -/
def num_girl_seniors : ℕ := num_girls / 2

/-- Represents the number of genders (boys and girls) -/
def num_genders : ℕ := 2

/-- Represents the number of class years (senior and junior) -/
def num_class_years : ℕ := 2

theorem club_leadership_combinations : 
  (num_genders * num_class_years * num_boy_seniors * num_boy_seniors) + 
  (num_genders * num_class_years * num_girl_seniors * num_girl_seniors) = 324 := by
  sorry

end NUMINAMATH_CALUDE_club_leadership_combinations_l2088_208877


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l2088_208834

/-- The product of fractions from 10/5 to 2520/2515 -/
def fraction_product : ℕ → ℚ
  | 0 => 2 -- 10/5
  | n + 1 => fraction_product n * ((5 * (n + 2)) / (5 * (n + 1)))

theorem fraction_product_simplification :
  fraction_product 502 = 504 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l2088_208834


namespace NUMINAMATH_CALUDE_T_properties_l2088_208831

/-- T(N) is the number of arrangements of integers 1 to N satisfying specific conditions. -/
def T (N : ℕ) : ℕ := sorry

/-- v₂(n) is the 2-adic valuation of n. -/
def v₂ (n : ℕ) : ℕ := sorry

theorem T_properties :
  (T 7 = 80) ∧
  (∀ n : ℕ, n ≥ 1 → v₂ (T (2^n - 1)) = 2^n - n - 1) ∧
  (∀ n : ℕ, n ≥ 1 → v₂ (T (2^n + 1)) = 2^n - 1) :=
by sorry

end NUMINAMATH_CALUDE_T_properties_l2088_208831


namespace NUMINAMATH_CALUDE_surface_area_specific_cube_l2088_208821

/-- Calculates the surface area of a cube with holes -/
def surface_area_cube_with_holes (cube_edge_length : ℝ) (hole_side_length : ℝ) (num_holes_per_face : ℕ) : ℝ :=
  let original_surface_area := 6 * cube_edge_length^2
  let area_removed_by_holes := 6 * num_holes_per_face * hole_side_length^2
  let area_exposed_by_holes := 6 * num_holes_per_face * 4 * hole_side_length^2
  original_surface_area - area_removed_by_holes + area_exposed_by_holes

/-- Theorem stating the surface area of the specific cube with holes -/
theorem surface_area_specific_cube : surface_area_cube_with_holes 4 1 2 = 132 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_specific_cube_l2088_208821


namespace NUMINAMATH_CALUDE_pentagonal_figure_segment_length_pentagonal_figure_segment_length_proof_l2088_208836

/-- The length of segment AB in a folded pentagonal figure --/
theorem pentagonal_figure_segment_length : ℝ :=
  -- Define the side length of the regular pentagons
  let side_length : ℝ := 1

  -- Define the number of pentagons
  let num_pentagons : ℕ := 4

  -- Define the internal angle of a regular pentagon (in radians)
  let pentagon_angle : ℝ := 3 * Real.pi / 5

  -- Define the angle between the square base and the pentagon face (in radians)
  let folding_angle : ℝ := Real.pi / 2 - pentagon_angle / 2

  -- The length of segment AB
  2

/-- Proof of the pentagonal_figure_segment_length theorem --/
theorem pentagonal_figure_segment_length_proof :
  pentagonal_figure_segment_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_figure_segment_length_pentagonal_figure_segment_length_proof_l2088_208836


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2088_208844

theorem sum_of_coefficients (b₅ b₄ b₃ b₂ b₁ b₀ : ℝ) :
  (∀ x : ℝ, (2*x + 3)^5 = b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 3125 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2088_208844


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2088_208879

-- Define the solution set of x^2 + ax + b > 0
def solution_set (a b : ℝ) : Set ℝ :=
  {x | x < -3 ∨ x > 1}

-- Define the quadratic inequality ax^2 + bx - 2 < 0
def quadratic_inequality (a b : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x - 2 < 0

-- Theorem statement
theorem solution_set_equivalence (a b : ℝ) :
  (∀ x, x ∈ solution_set a b ↔ x^2 + a*x + b > 0) →
  (∀ x, quadratic_inequality a b x ↔ x ∈ Set.Ioo (-1/2) 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2088_208879


namespace NUMINAMATH_CALUDE_armband_break_even_l2088_208805

/-- The cost of an individual ticket in dollars -/
def individual_ticket_cost : ℚ := 3/4

/-- The cost of an armband in dollars -/
def armband_cost : ℚ := 15

/-- The number of rides at which the armband cost equals the individual ticket cost -/
def break_even_rides : ℕ := 20

theorem armband_break_even :
  (individual_ticket_cost * break_even_rides : ℚ) = armband_cost :=
sorry

end NUMINAMATH_CALUDE_armband_break_even_l2088_208805


namespace NUMINAMATH_CALUDE_crowdfunding_highest_level_l2088_208823

/-- Represents the financial backing levels and backers for a crowdfunding campaign -/
structure CrowdfundingCampaign where
  lowest_level : ℕ
  second_level : ℕ
  highest_level : ℕ
  lowest_backers : ℕ
  second_backers : ℕ
  highest_backers : ℕ

/-- Theorem stating the conditions and the result to be proven -/
theorem crowdfunding_highest_level 
  (campaign : CrowdfundingCampaign)
  (level_relation : campaign.second_level = 10 * campaign.lowest_level ∧ 
                    campaign.highest_level = 10 * campaign.second_level)
  (backers : campaign.lowest_backers = 10 ∧ 
             campaign.second_backers = 3 ∧ 
             campaign.highest_backers = 2)
  (total_raised : campaign.lowest_backers * campaign.lowest_level + 
                  campaign.second_backers * campaign.second_level + 
                  campaign.highest_backers * campaign.highest_level = 12000) :
  campaign.highest_level = 5000 := by
  sorry


end NUMINAMATH_CALUDE_crowdfunding_highest_level_l2088_208823


namespace NUMINAMATH_CALUDE_cameron_chase_speed_ratio_l2088_208848

/-- Proves that the ratio of Cameron's speed to Chase's speed is 2:1 given the conditions -/
theorem cameron_chase_speed_ratio 
  (cameron_speed chase_speed danielle_speed : ℝ)
  (danielle_time chase_time : ℝ)
  (h1 : danielle_speed = 3 * cameron_speed)
  (h2 : danielle_time = 30)
  (h3 : chase_time = 180)
  (h4 : danielle_speed * danielle_time = chase_speed * chase_time) :
  cameron_speed / chase_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_cameron_chase_speed_ratio_l2088_208848


namespace NUMINAMATH_CALUDE_power_function_property_l2088_208803

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ a

-- State the theorem
theorem power_function_property (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = Real.sqrt 2) : 
  f 8 = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_power_function_property_l2088_208803


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2088_208872

theorem algebraic_expression_value (x : ℝ) :
  2 * x^2 + 3 * x + 7 = 8 → 4 * x^2 + 6 * x - 9 = -7 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2088_208872


namespace NUMINAMATH_CALUDE_fraction_equality_l2088_208835

theorem fraction_equality (a : ℕ+) :
  (a : ℚ) / (a + 45 : ℚ) = 3 / 4 → a = 135 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2088_208835


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2088_208827

/-- Given two arithmetic sequences {an} and {bn} with the specified conditions, 
    prove that a5 + b5 = 35 -/
theorem arithmetic_sequence_sum (a b : ℕ → ℕ) 
  (h1 : ∀ n, a (n + 1) - a n = a 2 - a 1)  -- a is an arithmetic sequence
  (h2 : ∀ n, b (n + 1) - b n = b 2 - b 1)  -- b is an arithmetic sequence
  (h3 : a 1 + b 1 = 7)                     -- first condition
  (h4 : a 3 + b 3 = 21)                    -- second condition
  : a 5 + b 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2088_208827


namespace NUMINAMATH_CALUDE_max_sequence_length_sequence_of_length_12_exists_l2088_208847

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The property that the sum of any five consecutive terms is negative -/
def SumOfFiveNegative (a : Sequence) :=
  ∀ i, i > 0 → (a i + a (i+1) + a (i+2) + a (i+3) + a (i+4)) < 0

/-- The property that the sum of any nine consecutive terms is positive -/
def SumOfNinePositive (a : Sequence) :=
  ∀ i, i > 0 → (a i + a (i+1) + a (i+2) + a (i+3) + a (i+4) + a (i+5) + a (i+6) + a (i+7) + a (i+8)) > 0

/-- The maximum length of a sequence satisfying both properties is 12 -/
theorem max_sequence_length :
  ∀ n : ℕ, n > 12 →
    ¬∃ a : Sequence, (SumOfFiveNegative a ∧ SumOfNinePositive a ∧ ∀ i > n, a i = 0) :=
by sorry

/-- There exists a sequence of length 12 satisfying both properties -/
theorem sequence_of_length_12_exists :
  ∃ a : Sequence, SumOfFiveNegative a ∧ SumOfNinePositive a ∧ ∀ i > 12, a i = 0 :=
by sorry

end NUMINAMATH_CALUDE_max_sequence_length_sequence_of_length_12_exists_l2088_208847


namespace NUMINAMATH_CALUDE_winning_percentage_correct_l2088_208822

/-- Represents the percentage of votes secured by the winning candidate -/
def winning_percentage : ℝ := 70

/-- Represents the total number of valid votes -/
def total_votes : ℕ := 450

/-- Represents the majority of votes by which the winning candidate won -/
def vote_majority : ℕ := 180

/-- Theorem stating that the winning percentage is correct given the conditions -/
theorem winning_percentage_correct :
  (winning_percentage / 100 * total_votes : ℝ) -
  ((100 - winning_percentage) / 100 * total_votes : ℝ) = vote_majority :=
sorry

end NUMINAMATH_CALUDE_winning_percentage_correct_l2088_208822


namespace NUMINAMATH_CALUDE_quadratic_properties_l2088_208889

-- Define the quadratic function
def f (a x : ℝ) : ℝ := (x + a) * (x - a - 1)

-- State the theorem
theorem quadratic_properties (a : ℝ) (h_a : a > 0) :
  -- 1. Axis of symmetry
  (∃ (x : ℝ), x = 1/2 ∧ ∀ (y : ℝ), f a (x - y) = f a (x + y)) ∧
  -- 2. Vertex coordinates when maximum is 4
  (∃ (x_max : ℝ), x_max ∈ Set.Icc (-1) 3 ∧ 
    (∀ (x : ℝ), x ∈ Set.Icc (-1) 3 → f a x ≤ 4) ∧ 
    f a x_max = 4 →
    f a (1/2) = -9/4) ∧
  -- 3. Range of t
  (∀ (t x₁ x₂ y₁ y₂ : ℝ),
    y₁ ≠ y₂ ∧
    t < x₁ ∧ x₁ < t + 1 ∧
    t + 2 < x₂ ∧ x₂ < t + 3 ∧
    f a x₁ = y₁ ∧ f a x₂ = y₂ →
    t ≥ -1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2088_208889


namespace NUMINAMATH_CALUDE_part_1_part_2_l2088_208825

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x + 1|

-- Define the function g
def g (x : ℝ) : ℝ := f 2 x - |x + 1|

-- Theorem for part 1
theorem part_1 : ∃ (a : ℝ), (∀ x, f a x ≤ 3 ↔ -2 ≤ x ∧ x ≤ 1) ∧ a = 2 := by sorry

-- Theorem for part 2
theorem part_2 : ∃ (min_value : ℝ), (∀ x, g x ≥ min_value) ∧ min_value = -1/2 := by sorry

end NUMINAMATH_CALUDE_part_1_part_2_l2088_208825


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l2088_208862

theorem cricket_team_age_difference 
  (team_size : ℕ) 
  (team_avg_age : ℝ) 
  (wicket_keeper_age_diff : ℝ) 
  (remaining_avg_age : ℝ) 
  (h1 : team_size = 11) 
  (h2 : team_avg_age = 26) 
  (h3 : wicket_keeper_age_diff = 3) 
  (h4 : remaining_avg_age = 23) : 
  team_avg_age - ((team_size * team_avg_age - (team_avg_age + wicket_keeper_age_diff + team_avg_age)) / (team_size - 2)) = 0.33 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l2088_208862


namespace NUMINAMATH_CALUDE_hourly_runoff_is_1000_l2088_208887

/-- The total capacity of the sewers in gallons -/
def sewer_capacity : ℕ := 240000

/-- The number of days the sewers can handle rain before overflowing -/
def days_before_overflow : ℕ := 10

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculates the hourly runoff rate -/
def hourly_runoff_rate : ℕ := sewer_capacity / (days_before_overflow * hours_per_day)

/-- Theorem stating that the hourly runoff rate is 1000 gallons per hour -/
theorem hourly_runoff_is_1000 : hourly_runoff_rate = 1000 := by
  sorry

end NUMINAMATH_CALUDE_hourly_runoff_is_1000_l2088_208887


namespace NUMINAMATH_CALUDE_replace_section_breaks_loop_l2088_208839

/-- Represents a railway section type -/
inductive SectionType
| Type1
| Type2

/-- Represents a railway configuration -/
structure RailwayConfig where
  type1Count : ℕ
  type2Count : ℕ

/-- Checks if a railway configuration forms a valid closed loop -/
def isValidClosedLoop (config : RailwayConfig) : Prop :=
  config.type1Count = config.type2Count

/-- Represents the operation of replacing a type 1 section with a type 2 section -/
def replaceSection (config : RailwayConfig) : RailwayConfig :=
  { type1Count := config.type1Count - 1,
    type2Count := config.type2Count + 1 }

/-- Main theorem: If a configuration forms a valid closed loop, 
    replacing a type 1 section with a type 2 section makes it impossible to form a closed loop -/
theorem replace_section_breaks_loop (config : RailwayConfig) :
  isValidClosedLoop config → ¬isValidClosedLoop (replaceSection config) := by
  sorry

end NUMINAMATH_CALUDE_replace_section_breaks_loop_l2088_208839


namespace NUMINAMATH_CALUDE_four_mat_weaves_four_days_l2088_208890

-- Define the rate of weaving (mats per mat-weave per day)
def weaving_rate (mats : ℕ) (mat_weaves : ℕ) (days : ℕ) : ℚ :=
  (mats : ℚ) / ((mat_weaves : ℚ) * (days : ℚ))

theorem four_mat_weaves_four_days (mats : ℕ) :
  -- Condition: 8 mat-weaves weave 16 mats in 8 days
  weaving_rate 16 8 8 = weaving_rate mats 4 4 →
  -- Conclusion: 4 mat-weaves weave 4 mats in 4 days
  mats = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_mat_weaves_four_days_l2088_208890


namespace NUMINAMATH_CALUDE_prime_factor_puzzle_l2088_208881

theorem prime_factor_puzzle (a b c d w x y z : ℕ) : 
  (Nat.Prime w) → 
  (Nat.Prime x) → 
  (Nat.Prime y) → 
  (Nat.Prime z) → 
  (w < x) → 
  (x < y) → 
  (y < z) → 
  ((w^a) * (x^b) * (y^c) * (z^d) = 660) → 
  ((a + b) - (c + d) = 1) → 
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_prime_factor_puzzle_l2088_208881


namespace NUMINAMATH_CALUDE_highest_power_of_two_dividing_13_4_minus_11_4_l2088_208874

theorem highest_power_of_two_dividing_13_4_minus_11_4 :
  ∃ (n : ℕ), 2^n = (Nat.gcd (13^4 - 11^4) (2^32 : ℕ)) ∧
  ∀ (m : ℕ), 2^m ∣ (13^4 - 11^4) → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_highest_power_of_two_dividing_13_4_minus_11_4_l2088_208874


namespace NUMINAMATH_CALUDE_rectangle_area_y_value_l2088_208899

/-- A rectangle with vertices at (-2, y), (10, y), (-2, 1), and (10, 1) has an area of 108 square units. Prove that y = 10. -/
theorem rectangle_area_y_value (y : ℝ) : 
  y > 0 → -- y is positive
  (10 - (-2)) * (y - 1) = 108 → -- area of the rectangle is 108 square units
  y = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_y_value_l2088_208899


namespace NUMINAMATH_CALUDE_radius_of_special_isosceles_triangle_l2088_208892

/-- Represents an isosceles triangle with a circumscribed circle. -/
structure IsoscelesTriangleWithCircle where
  /-- The length of the base of the isosceles triangle -/
  base : ℝ
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The two equal sides of the triangle are each twice the length of the base -/
  equal_sides_twice_base : base > 0
  /-- The perimeter in inches equals the area of the circumscribed circle in square inches -/
  perimeter_equals_circle_area : 5 * base = π * radius^2

/-- 
The radius of the circumscribed circle of an isosceles triangle is 2√5/π inches,
given that the perimeter in inches equals the area of the circumscribed circle in square inches,
and the two equal sides of the triangle are each twice the length of the base.
-/
theorem radius_of_special_isosceles_triangle (t : IsoscelesTriangleWithCircle) : 
  t.radius = 2 * Real.sqrt 5 / π :=
by sorry

end NUMINAMATH_CALUDE_radius_of_special_isosceles_triangle_l2088_208892


namespace NUMINAMATH_CALUDE_checkerboard_sum_l2088_208898

/-- The number of rectangles in a 7x7 checkerboard -/
def r' : ℕ := 784

/-- The number of squares in a 7x7 checkerboard -/
def s' : ℕ := 140

/-- m' and n' are relatively prime positive integers such that s'/r' = m'/n' -/
def m' : ℕ := 5
def n' : ℕ := 28

theorem checkerboard_sum : m' + n' = 33 := by sorry

end NUMINAMATH_CALUDE_checkerboard_sum_l2088_208898


namespace NUMINAMATH_CALUDE_stratified_sampling_male_count_l2088_208828

theorem stratified_sampling_male_count 
  (total_employees : ℕ) 
  (female_employees : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 120) 
  (h2 : female_employees = 72) 
  (h3 : sample_size = 15) :
  (total_employees - female_employees) * sample_size / total_employees = 6 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_male_count_l2088_208828


namespace NUMINAMATH_CALUDE_two_and_three_digit_sum_l2088_208870

theorem two_and_three_digit_sum : ∃! (x y : ℕ), 
  10 ≤ x ∧ x < 100 ∧ 
  100 ≤ y ∧ y < 1000 ∧ 
  1000 * x + y = 4 * x * y ∧ 
  x + y = 266 := by
sorry

end NUMINAMATH_CALUDE_two_and_three_digit_sum_l2088_208870


namespace NUMINAMATH_CALUDE_product_prs_l2088_208897

theorem product_prs (p r s : ℕ) : 
  4^p + 4^3 = 320 → 
  3^r + 27 = 108 → 
  2^s + 7^4 = 2617 → 
  p * r * s = 112 := by
sorry

end NUMINAMATH_CALUDE_product_prs_l2088_208897


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_20_l2088_208880

def arithmetic_sequence_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_20 :
  arithmetic_sequence_sum (-5) 3 20 = 470 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_20_l2088_208880


namespace NUMINAMATH_CALUDE_diamond_property_false_l2088_208815

/-- The diamond operation for real numbers -/
def diamond (x y : ℝ) : ℝ := |x + y - 1|

/-- The statement that is false -/
theorem diamond_property_false : ∃ x y : ℝ, 2 * (diamond x y) ≠ diamond (2 * x) (2 * y) := by
  sorry

end NUMINAMATH_CALUDE_diamond_property_false_l2088_208815


namespace NUMINAMATH_CALUDE_seaweed_harvest_l2088_208893

theorem seaweed_harvest (total : ℝ) :
  (0.5 * total ≥ 0) →                    -- 50% used for starting fires
  (0.25 * (0.5 * total) ≥ 0) →           -- 25% of remaining for human consumption
  (0.75 * (0.5 * total) = 150) →         -- 75% of remaining (150 pounds) fed to livestock
  (total = 400) :=
by sorry

end NUMINAMATH_CALUDE_seaweed_harvest_l2088_208893


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l2088_208867

theorem geometric_arithmetic_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  (2 * a 3 = a 1 + a 2) →       -- arithmetic sequence condition
  (q = 1 ∨ q = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l2088_208867


namespace NUMINAMATH_CALUDE_correlation_identification_l2088_208860

/-- Represents a relationship between two variables -/
inductive Relationship
| AgeWealth
| CurvePoint
| AppleProduction
| TreeDiameterHeight

/-- Determines if a relationship exhibits correlation -/
def has_correlation (r : Relationship) : Prop :=
  match r with
  | Relationship.AgeWealth => true
  | Relationship.CurvePoint => false
  | Relationship.AppleProduction => true
  | Relationship.TreeDiameterHeight => true

/-- The main theorem stating which relationships have correlation -/
theorem correlation_identification :
  (has_correlation Relationship.AgeWealth) ∧
  (¬has_correlation Relationship.CurvePoint) ∧
  (has_correlation Relationship.AppleProduction) ∧
  (has_correlation Relationship.TreeDiameterHeight) :=
sorry


end NUMINAMATH_CALUDE_correlation_identification_l2088_208860


namespace NUMINAMATH_CALUDE_complement_S_union_T_l2088_208829

def S : Set ℝ := {x | x > -2}
def T : Set ℝ := {x | x^2 + 3*x - 4 ≤ 0}

theorem complement_S_union_T : (Set.univ \ S) ∪ T = {x : ℝ | x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_S_union_T_l2088_208829


namespace NUMINAMATH_CALUDE_fraction_simplification_l2088_208871

theorem fraction_simplification (x y : ℝ) (h : x ≠ 3*y ∧ x ≠ -3*y) : 
  (2*x)/(x^2 - 9*y^2) - 1/(x - 3*y) = 1/(x + 3*y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2088_208871


namespace NUMINAMATH_CALUDE_triangle_third_side_l2088_208840

/-- Given a triangle with sides b, c, and x, where the area S = 0.4bc, 
    prove that the third side x satisfies the equation: x² = b² + c² ± 1.2bc -/
theorem triangle_third_side (b c x : ℝ) (h : b > 0 ∧ c > 0 ∧ x > 0) :
  (0.4 * b * c)^2 = (1/16) * (4 * b^2 * c^2 - (b^2 + c^2 - x^2)^2) →
  x^2 = b^2 + c^2 + 1.2 * b * c ∨ x^2 = b^2 + c^2 - 1.2 * b * c :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_side_l2088_208840


namespace NUMINAMATH_CALUDE_power_of_128_fourths_sevenths_l2088_208875

theorem power_of_128_fourths_sevenths : (128 : ℝ) ^ (4/7) = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_of_128_fourths_sevenths_l2088_208875


namespace NUMINAMATH_CALUDE_dolphin_training_l2088_208819

theorem dolphin_training (total : ℕ) (fully_trained_ratio : ℚ) (semi_trained_ratio : ℚ)
  (beginner_ratio : ℚ) (intermediate_ratio : ℚ)
  (h1 : total = 120)
  (h2 : fully_trained_ratio = 1/4)
  (h3 : semi_trained_ratio = 1/6)
  (h4 : beginner_ratio = 3/8)
  (h5 : intermediate_ratio = 5/9) :
  let fully_trained := (total : ℚ) * fully_trained_ratio
  let remaining_after_fully_trained := total - fully_trained.floor
  let semi_trained := (remaining_after_fully_trained : ℚ) * semi_trained_ratio
  let untrained := remaining_after_fully_trained - semi_trained.floor
  let semi_and_untrained := semi_trained.floor + untrained
  let in_beginner := (semi_and_untrained : ℚ) * beginner_ratio
  let remaining_after_beginner := semi_and_untrained - in_beginner.floor
  let start_intermediate := (remaining_after_beginner : ℚ) * intermediate_ratio
  start_intermediate.floor = 31 :=
by sorry

end NUMINAMATH_CALUDE_dolphin_training_l2088_208819


namespace NUMINAMATH_CALUDE_positive_root_range_l2088_208850

theorem positive_root_range : ∃ x : ℝ, x^2 - 2*x - 1 = 0 ∧ x > 0 ∧ 2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_root_range_l2088_208850


namespace NUMINAMATH_CALUDE_min_sum_of_dimensions_l2088_208866

theorem min_sum_of_dimensions (a b c : ℕ+) : 
  a * b * c = 2310 → a + b + c ≥ 42 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_dimensions_l2088_208866


namespace NUMINAMATH_CALUDE_rectangle_area_increase_rectangle_area_percentage_increase_l2088_208818

theorem rectangle_area_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  (1.3 * l) * (1.2 * w) = 1.56 * (l * w) := by
  sorry

theorem rectangle_area_percentage_increase :
  (1.56 - 1) * 100 = 56 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_rectangle_area_percentage_increase_l2088_208818


namespace NUMINAMATH_CALUDE_complex_number_solution_l2088_208816

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_number_solution (z : ℂ) 
  (h1 : is_purely_imaginary (z - 1))
  (h2 : is_purely_imaginary ((z + 1)^2 - 8*I)) :
  z = 1 - 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_solution_l2088_208816


namespace NUMINAMATH_CALUDE_prove_trip_length_l2088_208845

def trip_length : ℚ := 360 / 7

theorem prove_trip_length :
  let first_part : ℚ := 1 / 4
  let second_part : ℚ := 30
  let third_part : ℚ := 1 / 6
  (first_part + third_part + second_part / trip_length = 1) →
  trip_length = 360 / 7 := by
sorry

end NUMINAMATH_CALUDE_prove_trip_length_l2088_208845
