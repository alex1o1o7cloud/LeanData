import Mathlib

namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l1733_173334

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, x > 0 ∧ x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x > 0 → x^2 - 2*x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l1733_173334


namespace NUMINAMATH_CALUDE_percent_profit_calculation_l1733_173395

theorem percent_profit_calculation (C S : ℝ) (h : 60 * C = 50 * S) :
  (S - C) / C * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percent_profit_calculation_l1733_173395


namespace NUMINAMATH_CALUDE_complex_magnitude_l1733_173384

theorem complex_magnitude (s : ℝ) (w : ℂ) (h1 : |s| < 4) (h2 : w + 4 / w = s) : Complex.abs w = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1733_173384


namespace NUMINAMATH_CALUDE_tom_teaching_years_l1733_173314

theorem tom_teaching_years (tom devin : ℕ) 
  (h1 : tom + devin = 70)
  (h2 : devin = tom / 2 - 5) :
  tom = 50 := by
  sorry

end NUMINAMATH_CALUDE_tom_teaching_years_l1733_173314


namespace NUMINAMATH_CALUDE_y_increases_with_x_on_positive_slope_line_l1733_173385

/-- Given two points on a line with a positive slope, if the x-coordinate of the first point
    is less than the x-coordinate of the second point, then the y-coordinate of the first point
    is less than the y-coordinate of the second point. -/
theorem y_increases_with_x_on_positive_slope_line 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = 3 * x₁ + 4) 
  (h2 : y₂ = 3 * x₂ + 4) 
  (h3 : x₁ < x₂) : 
  y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y_increases_with_x_on_positive_slope_line_l1733_173385


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l1733_173359

theorem largest_angle_in_triangle (y : ℝ) : 
  y + 60 + 70 = 180 → 
  max y (max 60 70) = 70 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l1733_173359


namespace NUMINAMATH_CALUDE_jens_birds_l1733_173353

theorem jens_birds (ducks chickens : ℕ) : 
  ducks > 4 * chickens →
  ducks = 150 →
  ducks + chickens = 185 →
  ducks - 4 * chickens = 10 := by
sorry

end NUMINAMATH_CALUDE_jens_birds_l1733_173353


namespace NUMINAMATH_CALUDE_songs_deleted_l1733_173348

theorem songs_deleted (pictures : ℕ) (text_files : ℕ) (total_files : ℕ) (songs : ℕ) : 
  pictures = 2 → text_files = 7 → total_files = 17 → pictures + songs + text_files = total_files → songs = 8 := by
  sorry

end NUMINAMATH_CALUDE_songs_deleted_l1733_173348


namespace NUMINAMATH_CALUDE_expand_product_l1733_173367

theorem expand_product (x : ℝ) : 4 * (x + 3) * (x + 6) = 4 * x^2 + 36 * x + 72 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1733_173367


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1733_173379

theorem contrapositive_equivalence (x : ℝ) :
  (x ≠ 3 ∧ x ≠ 4 → x^2 - 7*x + 12 ≠ 0) ↔ (x^2 - 7*x + 12 = 0 → x = 3 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1733_173379


namespace NUMINAMATH_CALUDE_equation_solution_l1733_173377

theorem equation_solution (m n k x : ℝ) 
  (hm : m ≠ 0) (hn : n ≠ 0) (hk : k ≠ 0) (hmn : m ≠ n) :
  (x + m)^2 - (x + n)^2 = k * (m - n)^2 → 
  x = ((k - 1) * (m + n) - 2 * k * n) / 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1733_173377


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1733_173327

/-- An isosceles triangle with side lengths 2 and 4 has a perimeter of 10 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  (a = 2 ∧ b = 4 ∧ c = 4) ∨ (a = 4 ∧ b = 2 ∧ c = 4) →
  a + b > c → b + c > a → c + a > b →
  a + b + c = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1733_173327


namespace NUMINAMATH_CALUDE_script_writing_problem_l1733_173362

/-- Represents the number of lines for each character in the script -/
structure ScriptLines where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The conditions of the script writing problem -/
def script_conditions (s : ScriptLines) : Prop :=
  s.first = s.second + 8 ∧
  s.third = 2 ∧
  s.second = 3 * s.third + 6 ∧
  s.first = 20

/-- The theorem stating the solution to the script writing problem -/
theorem script_writing_problem (s : ScriptLines) 
  (h : script_conditions s) : ∃ m : ℕ, s.second = m * s.third + 6 ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_script_writing_problem_l1733_173362


namespace NUMINAMATH_CALUDE_inequality_holds_iff_a_in_range_l1733_173307

theorem inequality_holds_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_a_in_range_l1733_173307


namespace NUMINAMATH_CALUDE_uniform_prices_theorem_l1733_173391

/-- Represents a servant's employment terms and compensation --/
structure Servant where
  annual_salary : ℕ  -- Annual salary in Rupees
  service_months : ℕ  -- Months of service completed
  partial_payment : ℕ  -- Partial payment received in Rupees

/-- Calculates the price of a uniform given a servant's terms and compensation --/
def uniform_price (s : Servant) : ℕ :=
  (s.service_months * s.annual_salary - 12 * s.partial_payment) / (12 - s.service_months)

theorem uniform_prices_theorem (servant_a servant_b servant_c : Servant) 
  (h_a : servant_a = { annual_salary := 500, service_months := 9, partial_payment := 250 })
  (h_b : servant_b = { annual_salary := 800, service_months := 6, partial_payment := 300 })
  (h_c : servant_c = { annual_salary := 1200, service_months := 4, partial_payment := 200 }) :
  uniform_price servant_a = 500 ∧ 
  uniform_price servant_b = 200 ∧ 
  uniform_price servant_c = 300 := by
  sorry

#eval uniform_price { annual_salary := 500, service_months := 9, partial_payment := 250 }
#eval uniform_price { annual_salary := 800, service_months := 6, partial_payment := 300 }
#eval uniform_price { annual_salary := 1200, service_months := 4, partial_payment := 200 }

end NUMINAMATH_CALUDE_uniform_prices_theorem_l1733_173391


namespace NUMINAMATH_CALUDE_derivative_h_at_one_l1733_173374

-- Define a function f
variable (f : ℝ → ℝ)

-- Define g(x) = f(x) - f(2x)
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x - f (2 * x)

-- Define h(x) = f(x) - f(4x)
def h (f : ℝ → ℝ) (x : ℝ) : ℝ := f x - f (4 * x)

-- State the theorem
theorem derivative_h_at_one (f : ℝ → ℝ) 
  (hg1 : deriv (g f) 1 = 5)
  (hg2 : deriv (g f) 2 = 7) :
  deriv (h f) 1 = 19 := by
  sorry

end NUMINAMATH_CALUDE_derivative_h_at_one_l1733_173374


namespace NUMINAMATH_CALUDE_total_books_calculation_l1733_173350

def initial_books : ℕ := 9
def added_books : ℕ := 10

theorem total_books_calculation :
  initial_books + added_books = 19 :=
by sorry

end NUMINAMATH_CALUDE_total_books_calculation_l1733_173350


namespace NUMINAMATH_CALUDE_intersection_A_B_l1733_173329

def U : Set Int := {-1, 3, 5, 7, 9}
def complement_A : Set Int := {-1, 9}
def B : Set Int := {3, 7, 9}

def A : Set Int := U \ complement_A

theorem intersection_A_B :
  A ∩ B = {3, 7} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1733_173329


namespace NUMINAMATH_CALUDE_max_min_sum_of_f_l1733_173392

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) - x) + (3 * Real.exp x + 1) / (Real.exp x + 1)

def domain : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

theorem max_min_sum_of_f :
  ∃ (M N : ℝ), (∀ x ∈ domain, f x ≤ M) ∧
               (∀ x ∈ domain, N ≤ f x) ∧
               (∃ x₁ ∈ domain, f x₁ = M) ∧
               (∃ x₂ ∈ domain, f x₂ = N) ∧
               M + N = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_min_sum_of_f_l1733_173392


namespace NUMINAMATH_CALUDE_paperclip_capacity_l1733_173333

theorem paperclip_capacity (small_volume small_capacity large_volume efficiency : ℝ) 
  (h1 : small_volume = 12)
  (h2 : small_capacity = 40)
  (h3 : large_volume = 60)
  (h4 : efficiency = 0.8)
  : (large_volume * efficiency * small_capacity) / small_volume = 160 := by
  sorry

end NUMINAMATH_CALUDE_paperclip_capacity_l1733_173333


namespace NUMINAMATH_CALUDE_correct_ranking_l1733_173370

/-- Represents a contestant's score -/
structure Score where
  value : ℝ
  positive : value > 0

/-- Represents the scores of the four contestants -/
structure ContestScores where
  ann : Score
  bill : Score
  carol : Score
  dick : Score
  sum_equality : bill.value + dick.value = ann.value + carol.value
  interchange_inequality : carol.value + bill.value > dick.value + ann.value
  carol_exceeds_sum : carol.value > ann.value + bill.value

/-- Represents the ranking of contestants -/
inductive Ranking
  | CDBA : Ranking  -- Carol, Dick, Bill, Ann
  | CDAB : Ranking  -- Carol, Dick, Ann, Bill
  | DCBA : Ranking  -- Dick, Carol, Bill, Ann
  | ACDB : Ranking  -- Ann, Carol, Dick, Bill
  | DCAB : Ranking  -- Dick, Carol, Ann, Bill

/-- The theorem stating that given the contest conditions, the correct ranking is CDBA -/
theorem correct_ranking (scores : ContestScores) : Ranking.CDBA = 
  (match scores with
  | ⟨ann, bill, carol, dick, _, _, _⟩ => 
      if carol.value > dick.value ∧ dick.value > bill.value ∧ bill.value > ann.value
      then Ranking.CDBA
      else Ranking.CDBA) := by
  sorry

end NUMINAMATH_CALUDE_correct_ranking_l1733_173370


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l1733_173366

theorem symmetric_points_sum_power (m n : ℤ) : 
  (m = -6 ∧ n = 5) → (m + n)^2012 = 1 := by sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l1733_173366


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1733_173368

-- Define the sets corresponding to ¬p and ¬q
def not_p (x : ℝ) : Prop := x ≤ 0 ∨ x ≥ 2
def not_q (x : ℝ) : Prop := x ≤ 0 ∨ x > 1

-- Define the original conditions p and q
def p (x : ℝ) : Prop := 0 < x ∧ x < 2
def q (x : ℝ) : Prop := 1 / x ≥ 1

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, not_p x → not_q x) ∧ 
  (∃ x, not_q x ∧ ¬(not_p x)) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1733_173368


namespace NUMINAMATH_CALUDE_unknown_number_theorem_l1733_173305

theorem unknown_number_theorem (X : ℝ) : 30 = 0.50 * X + 10 → X = 40 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_theorem_l1733_173305


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1733_173339

theorem polynomial_simplification (x y : ℝ) :
  (10 * x^12 + 8 * x^9 + 5 * x^7) + (11 * x^9 + 3 * x^7 + 4 * x^3 + 6 * y^2 + 7 * x + 9) =
  10 * x^12 + 19 * x^9 + 8 * x^7 + 4 * x^3 + 6 * y^2 + 7 * x + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1733_173339


namespace NUMINAMATH_CALUDE_exists_permutation_with_many_swaps_l1733_173344

/-- 
Represents a permutation of cards numbered from 1 to n.
-/
def Permutation (n : ℕ) := Fin n → Fin n

/-- 
Counts the number of adjacent swaps needed to sort a permutation into descending order.
-/
def countSwaps (n : ℕ) (p : Permutation n) : ℕ := sorry

/-- 
Theorem: There exists a permutation of n cards that requires at least n(n-1)/2 adjacent swaps
to sort into descending order.
-/
theorem exists_permutation_with_many_swaps (n : ℕ) :
  ∃ (p : Permutation n), countSwaps n p ≥ n * (n - 1) / 2 := by sorry

end NUMINAMATH_CALUDE_exists_permutation_with_many_swaps_l1733_173344


namespace NUMINAMATH_CALUDE_y_value_l1733_173369

theorem y_value : ∃ y : ℝ, (3 * y) / 7 = 15 ∧ y = 35 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l1733_173369


namespace NUMINAMATH_CALUDE_randy_farm_trees_l1733_173347

/-- Calculates the total number of trees on Randy's farm -/
def total_trees (mango_trees : ℕ) (coconut_trees : ℕ) : ℕ :=
  mango_trees + coconut_trees

/-- Theorem: Given Randy's farm conditions, the total number of trees is 85 -/
theorem randy_farm_trees :
  let mango_trees : ℕ := 60
  let coconut_trees : ℕ := mango_trees / 2 - 5
  total_trees mango_trees coconut_trees = 85 := by
  sorry

end NUMINAMATH_CALUDE_randy_farm_trees_l1733_173347


namespace NUMINAMATH_CALUDE_function_range_is_all_reals_l1733_173386

theorem function_range_is_all_reals :
  ∀ y : ℝ, ∃ x : ℝ, y = (x^2 + 3*x + 2) / (x^2 + x + 1) := by
  sorry

end NUMINAMATH_CALUDE_function_range_is_all_reals_l1733_173386


namespace NUMINAMATH_CALUDE_digit_sum_property_l1733_173321

def S (k : ℕ) : ℕ := (k.repr.toList.map (λ c => c.toNat - 48)).sum

theorem digit_sum_property (n : ℕ) : 
  (∃ (a b : ℕ), n = S a ∧ n = S b ∧ n = S (a + b)) ↔ 
  (∃ (k : ℕ+), n = 9 * k) :=
sorry

end NUMINAMATH_CALUDE_digit_sum_property_l1733_173321


namespace NUMINAMATH_CALUDE_system_solution_l1733_173346

theorem system_solution :
  ∀ x y : ℝ,
  x^2 - 3*y - 88 ≥ 0 →
  x + 6*y ≥ 0 →
  (5 * Real.sqrt (x^2 - 3*y - 88) + Real.sqrt (x + 6*y) = 19 ∧
   3 * Real.sqrt (x^2 - 3*y - 88) = 1 + 2 * Real.sqrt (x + 6*y)) →
  ((x = 10 ∧ y = 1) ∨ (x = -21/2 ∧ y = 53/12)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1733_173346


namespace NUMINAMATH_CALUDE_xw_value_l1733_173381

/-- Triangle XYZ with point W on YZ such that XW is perpendicular to YZ -/
structure TriangleXYZW where
  /-- The length of side XY -/
  XY : ℝ
  /-- The length of side XZ -/
  XZ : ℝ
  /-- The length of XW, where W is on YZ and XW ⟂ YZ -/
  XW : ℝ
  /-- The length of YW -/
  YW : ℝ
  /-- The length of ZW -/
  ZW : ℝ
  /-- XY equals 15 -/
  xy_eq : XY = 15
  /-- XZ equals 26 -/
  xz_eq : XZ = 26
  /-- YW:ZW ratio is 3:4 -/
  yw_zw_ratio : YW / ZW = 3 / 4
  /-- Pythagorean theorem for XYW -/
  pythagoras_xyw : YW ^ 2 = XY ^ 2 - XW ^ 2
  /-- Pythagorean theorem for XZW -/
  pythagoras_xzw : ZW ^ 2 = XZ ^ 2 - XW ^ 2

/-- The main theorem: If the conditions are met, then XW = 42/√7 -/
theorem xw_value (t : TriangleXYZW) : t.XW = 42 / Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_xw_value_l1733_173381


namespace NUMINAMATH_CALUDE_max_value_f_l1733_173336

/-- The function f(x) = x(1 - x^2) -/
def f (x : ℝ) : ℝ := x * (1 - x^2)

/-- The maximum value of f(x) on [0, 1] is 2√3/9 -/
theorem max_value_f : ∃ (c : ℝ), c = (2 * Real.sqrt 3) / 9 ∧ 
  (∀ x ∈ Set.Icc 0 1, f x ≤ c) ∧ 
  (∃ x ∈ Set.Icc 0 1, f x = c) := by
  sorry

end NUMINAMATH_CALUDE_max_value_f_l1733_173336


namespace NUMINAMATH_CALUDE_equation_solution_l1733_173383

theorem equation_solution (x : ℂ) (h1 : x ≠ -2) (h2 : x ≠ 3) :
  (3*x - 6) / (x + 2) + (3*x^2 - 12) / (3 - x) = 3 ↔ x = -2 + 2*I ∨ x = -2 - 2*I :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l1733_173383


namespace NUMINAMATH_CALUDE_emma_calculation_l1733_173323

/-- Reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Checks if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem emma_calculation (a b : ℕ) (ha : is_two_digit a) (hb : b > 0) :
  (reverse_digits a * b - 18 = 120) → (a * b = 192) := by
  sorry

end NUMINAMATH_CALUDE_emma_calculation_l1733_173323


namespace NUMINAMATH_CALUDE_dog_year_conversion_l1733_173354

/-- Represents the conversion of dog years to human years -/
structure DogYearConversion where
  first_year : ℕ
  second_year : ℕ
  later_years : ℕ

/-- Calculates the total human years for a given dog age -/
def human_years (c : DogYearConversion) (dog_age : ℕ) : ℕ :=
  if dog_age = 0 then 0
  else if dog_age = 1 then c.first_year
  else if dog_age = 2 then c.first_year + c.second_year
  else c.first_year + c.second_year + (dog_age - 2) * c.later_years

/-- The main theorem to prove -/
theorem dog_year_conversion (c : DogYearConversion) :
  c.first_year = 15 → c.second_year = 9 → human_years c 10 = 64 → c.later_years = 5 := by
  sorry

end NUMINAMATH_CALUDE_dog_year_conversion_l1733_173354


namespace NUMINAMATH_CALUDE_line_slope_l1733_173312

/-- Given a line with equation x/4 + y/3 = 2, its slope is -3/4 -/
theorem line_slope (x y : ℝ) : (x / 4 + y / 3 = 2) → (∃ m b : ℝ, y = m * x + b ∧ m = -3/4) := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l1733_173312


namespace NUMINAMATH_CALUDE_cyclic_difference_fourth_power_sum_l1733_173371

theorem cyclic_difference_fourth_power_sum (a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℕ) 
  (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ a₁ ≠ a₆ ∧ a₁ ≠ a₇ ∧
                a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ a₂ ≠ a₆ ∧ a₂ ≠ a₇ ∧
                a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ a₃ ≠ a₆ ∧ a₃ ≠ a₇ ∧
                a₄ ≠ a₅ ∧ a₄ ≠ a₆ ∧ a₄ ≠ a₇ ∧
                a₅ ≠ a₆ ∧ a₅ ≠ a₇ ∧
                a₆ ≠ a₇) :
  (a₁ - a₂)^4 + (a₂ - a₃)^4 + (a₃ - a₄)^4 + (a₄ - a₅)^4 + 
  (a₅ - a₆)^4 + (a₆ - a₇)^4 + (a₇ - a₁)^4 ≥ 82 :=
by sorry

end NUMINAMATH_CALUDE_cyclic_difference_fourth_power_sum_l1733_173371


namespace NUMINAMATH_CALUDE_probability_theorem_l1733_173342

/-- Represents the contents of the magician's box -/
structure Box :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)

/-- Calculates the probability of drawing all red chips before blue and green chips -/
def probability_all_red_first (b : Box) : ℚ :=
  sorry

/-- The magician's box -/
def magicians_box : Box :=
  { red := 4, green := 3, blue := 1 }

/-- Theorem stating the probability of drawing all red chips first -/
theorem probability_theorem :
  probability_all_red_first magicians_box = 5 / 6720 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l1733_173342


namespace NUMINAMATH_CALUDE_baby_whales_count_l1733_173394

/-- Represents the number of whales observed during Ishmael's monitoring --/
structure WhaleCount where
  first_trip_males : ℕ
  first_trip_females : ℕ
  third_trip_males : ℕ
  third_trip_females : ℕ
  total_whales : ℕ

/-- Theorem stating the number of baby whales observed on the second trip --/
theorem baby_whales_count (w : WhaleCount) 
  (h1 : w.first_trip_males = 28)
  (h2 : w.first_trip_females = 2 * w.first_trip_males)
  (h3 : w.third_trip_males = w.first_trip_males / 2)
  (h4 : w.third_trip_females = w.first_trip_females)
  (h5 : w.total_whales = 178) :
  w.total_whales - (w.first_trip_males + w.first_trip_females + w.third_trip_males + w.third_trip_females) = 24 := by
  sorry

end NUMINAMATH_CALUDE_baby_whales_count_l1733_173394


namespace NUMINAMATH_CALUDE_six_digit_integers_count_is_60_l1733_173399

/-- The number of different six-digit integers that can be formed using the digits 1, 1, 3, 3, 3, and 5 -/
def sixDigitIntegersCount : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of different six-digit integers 
    that can be formed using the digits 1, 1, 3, 3, 3, and 5 is equal to 60 -/
theorem six_digit_integers_count_is_60 : sixDigitIntegersCount = 60 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_integers_count_is_60_l1733_173399


namespace NUMINAMATH_CALUDE_product_sum_theorem_l1733_173326

theorem product_sum_theorem (p q r s t : ℤ) :
  (7 - p) * (7 - q) * (7 - r) * (7 - s) * (7 - t) = -48 →
  p + q + r + s + t = 22 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l1733_173326


namespace NUMINAMATH_CALUDE_megan_folders_l1733_173340

theorem megan_folders (initial_files : ℕ) (deleted_files : ℕ) (files_per_folder : ℕ) : 
  initial_files = 93 → 
  deleted_files = 21 → 
  files_per_folder = 8 → 
  (initial_files - deleted_files) / files_per_folder = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_megan_folders_l1733_173340


namespace NUMINAMATH_CALUDE_inequality_proof_l1733_173315

theorem inequality_proof (a b c d p q : ℝ) 
  (h1 : a * b + c * d = 2 * p * q)
  (h2 : a * c ≥ p ^ 2)
  (h3 : p > 0) : 
  b * d ≤ q ^ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1733_173315


namespace NUMINAMATH_CALUDE_johns_age_l1733_173393

/-- Proves that John's current age is 39 years old given the problem conditions -/
theorem johns_age (john_age : ℕ) (james_age : ℕ) (james_brother_age : ℕ) : 
  james_brother_age = 16 →
  james_brother_age = james_age + 4 →
  john_age - 3 = 2 * (james_age + 6) →
  john_age = 39 := by
  sorry

#check johns_age

end NUMINAMATH_CALUDE_johns_age_l1733_173393


namespace NUMINAMATH_CALUDE_plates_used_l1733_173317

theorem plates_used (guests : ℕ) (meals_per_day : ℕ) (plates_per_meal : ℕ) (days : ℕ) : 
  guests = 5 →
  meals_per_day = 3 →
  plates_per_meal = 2 →
  days = 4 →
  (guests + 1) * meals_per_day * plates_per_meal * days = 144 :=
by sorry

end NUMINAMATH_CALUDE_plates_used_l1733_173317


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1733_173357

theorem quadratic_solution_difference_squared :
  ∀ (α β : ℝ),
    α ≠ β →
    α^2 - 3*α + 2 = 0 →
    β^2 - 3*β + 2 = 0 →
    (α - β)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1733_173357


namespace NUMINAMATH_CALUDE_sum_ratio_equals_four_sevenths_l1733_173343

theorem sum_ratio_equals_four_sevenths 
  (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_squares_abc : a^2 + b^2 + c^2 = 16)
  (sum_squares_xyz : x^2 + y^2 + z^2 = 49)
  (sum_products : a*x + b*y + c*z = 28) :
  (a + b + c) / (x + y + z) = 4/7 := by
sorry

end NUMINAMATH_CALUDE_sum_ratio_equals_four_sevenths_l1733_173343


namespace NUMINAMATH_CALUDE_parallel_vectors_x_equals_9_l1733_173303

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given vectors a and b, prove that if they are parallel, then x = 9 -/
theorem parallel_vectors_x_equals_9 (x : ℝ) :
  let a : ℝ × ℝ := (x, 3)
  let b : ℝ × ℝ := (3, 1)
  parallel a b → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_equals_9_l1733_173303


namespace NUMINAMATH_CALUDE_fruit_seller_inventory_l1733_173387

theorem fruit_seller_inventory (apples oranges bananas pears grapes : ℕ) : 
  (apples - apples / 2 + 20 = 370) →
  (oranges - oranges * 35 / 100 = 195) →
  (bananas - bananas * 3 / 5 + 15 = 95) →
  (pears - pears * 45 / 100 = 50) →
  (grapes - grapes * 3 / 10 = 140) →
  (apples = 700 ∧ oranges = 300 ∧ bananas = 200 ∧ pears = 91 ∧ grapes = 200) :=
by sorry

end NUMINAMATH_CALUDE_fruit_seller_inventory_l1733_173387


namespace NUMINAMATH_CALUDE_line_arrangement_count_l1733_173355

def number_of_students : ℕ := 5
def number_of_boys : ℕ := 2
def number_of_girls : ℕ := 3

theorem line_arrangement_count : 
  (number_of_students = number_of_boys + number_of_girls) →
  (number_of_boys = 2) →
  (number_of_girls = 3) →
  (∃ (arrangement_count : ℕ), 
    arrangement_count = (Nat.factorial number_of_boys) * (Nat.factorial (number_of_girls + 1)) ∧
    arrangement_count = 48) :=
by sorry

end NUMINAMATH_CALUDE_line_arrangement_count_l1733_173355


namespace NUMINAMATH_CALUDE_income_increase_percentage_l1733_173306

theorem income_increase_percentage 
  (initial_income : ℝ) 
  (initial_expenditure_ratio : ℝ) 
  (expenditure_increase_ratio : ℝ) 
  (savings_increase_ratio : ℝ) 
  (income_increase_ratio : ℝ)
  (h1 : initial_expenditure_ratio = 0.75)
  (h2 : expenditure_increase_ratio = 1.1)
  (h3 : savings_increase_ratio = 1.5)
  (h4 : income_increase_ratio > 0)
  : income_increase_ratio = 1.2 := by
  sorry

#check income_increase_percentage

end NUMINAMATH_CALUDE_income_increase_percentage_l1733_173306


namespace NUMINAMATH_CALUDE_AMC9_paths_count_l1733_173311

/-- Represents the layout of the AMC9 puzzle --/
structure AMC9Layout where
  start_A : Nat
  adjacent_Ms : Nat
  adjacent_Cs : Nat
  Cs_with_two_9s : Nat
  Cs_with_one_9 : Nat

/-- Calculates the number of paths in the AMC9 puzzle --/
def count_AMC9_paths (layout : AMC9Layout) : Nat :=
  layout.adjacent_Ms * 
  (layout.Cs_with_two_9s * 2 + layout.Cs_with_one_9 * 1)

/-- Theorem stating that the number of paths in the AMC9 puzzle is 20 --/
theorem AMC9_paths_count :
  ∀ (layout : AMC9Layout),
  layout.start_A = 1 →
  layout.adjacent_Ms = 4 →
  layout.adjacent_Cs = 3 →
  layout.Cs_with_two_9s = 2 →
  layout.Cs_with_one_9 = 1 →
  count_AMC9_paths layout = 20 := by
  sorry

end NUMINAMATH_CALUDE_AMC9_paths_count_l1733_173311


namespace NUMINAMATH_CALUDE_z_squared_minus_one_equals_two_plus_four_i_l1733_173363

def z : ℂ := 2 + Complex.I

theorem z_squared_minus_one_equals_two_plus_four_i :
  z^2 - 1 = 2 + 4*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_z_squared_minus_one_equals_two_plus_four_i_l1733_173363


namespace NUMINAMATH_CALUDE_max_product_other_sides_l1733_173378

/-- Given a triangle with one side of length 4 and the opposite angle of 60°,
    the maximum product of the lengths of the other two sides is 16. -/
theorem max_product_other_sides (a b c : ℝ) (A B C : ℝ) :
  a = 4 →
  A = π / 3 →
  0 < b ∧ 0 < c →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  b * c ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_max_product_other_sides_l1733_173378


namespace NUMINAMATH_CALUDE_jones_wardrobe_count_l1733_173338

/-- Represents the clothing items of Mr. Jones -/
structure Wardrobe where
  pants : ℕ
  shirts : ℕ
  ties : ℕ
  socks : ℕ

/-- Calculates the total number of clothing items -/
def total_clothes (w : Wardrobe) : ℕ :=
  w.pants + w.shirts + w.ties + w.socks

/-- Theorem stating the total number of clothes Mr. Jones owns -/
theorem jones_wardrobe_count :
  ∃ (w : Wardrobe),
    w.pants = 40 ∧
    w.shirts = 6 * w.pants ∧
    w.ties = (3 * w.shirts) / 2 ∧
    w.socks = w.ties ∧
    total_clothes w = 1000 := by
  sorry

#check jones_wardrobe_count

end NUMINAMATH_CALUDE_jones_wardrobe_count_l1733_173338


namespace NUMINAMATH_CALUDE_annual_growth_rate_proof_l1733_173382

-- Define the initial number of students
def initial_students : ℕ := 200

-- Define the final number of students
def final_students : ℕ := 675

-- Define the number of years
def years : ℕ := 3

-- Define the growth rate as a real number between 0 and 1
def growth_rate : ℝ := 0.5

-- Theorem statement
theorem annual_growth_rate_proof :
  (initial_students : ℝ) * (1 + growth_rate)^years = final_students :=
sorry

end NUMINAMATH_CALUDE_annual_growth_rate_proof_l1733_173382


namespace NUMINAMATH_CALUDE_chess_square_exists_l1733_173316

/-- Represents the color of a cell -/
inductive Color
| Black
| White

/-- Represents a 100x100 table of colored cells -/
def Table := Fin 100 → Fin 100 → Color

/-- Checks if a cell is on the border of the table -/
def isBorder (i j : Fin 100) : Prop :=
  i = 0 || i = 99 || j = 0 || j = 99

/-- Checks if a 2x2 square starting at (i,j) contains cells of two colors -/
def hasTwoColors (t : Table) (i j : Fin 100) : Prop :=
  ∃ (c₁ c₂ : Color), c₁ ≠ c₂ ∧
    ((t i j = c₁ ∧ t (i+1) j = c₂) ∨
     (t i j = c₁ ∧ t i (j+1) = c₂) ∨
     (t i j = c₁ ∧ t (i+1) (j+1) = c₂) ∨
     (t (i+1) j = c₁ ∧ t i (j+1) = c₂) ∨
     (t (i+1) j = c₁ ∧ t (i+1) (j+1) = c₂) ∨
     (t i (j+1) = c₁ ∧ t (i+1) (j+1) = c₂))

/-- Checks if a 2x2 square starting at (i,j) is colored in chess order -/
def isChessOrder (t : Table) (i j : Fin 100) : Prop :=
  (t i j = Color.Black ∧ t (i+1) j = Color.White ∧ t i (j+1) = Color.White ∧ t (i+1) (j+1) = Color.Black) ∨
  (t i j = Color.White ∧ t (i+1) j = Color.Black ∧ t i (j+1) = Color.Black ∧ t (i+1) (j+1) = Color.White)

theorem chess_square_exists (t : Table) 
  (border_black : ∀ i j, isBorder i j → t i j = Color.Black)
  (two_colors : ∀ i j, hasTwoColors t i j) :
  ∃ i j, isChessOrder t i j := by
  sorry

end NUMINAMATH_CALUDE_chess_square_exists_l1733_173316


namespace NUMINAMATH_CALUDE_jamie_lost_balls_jamie_lost_six_balls_l1733_173302

theorem jamie_lost_balls (initial_red : ℕ) (blue_multiplier : ℕ) (yellow_bought : ℕ) (final_total : ℕ) : ℕ :=
  let initial_blue := blue_multiplier * initial_red
  let initial_total := initial_red + initial_blue + yellow_bought
  initial_total - final_total

theorem jamie_lost_six_balls : jamie_lost_balls 16 2 32 74 = 6 := by
  sorry

end NUMINAMATH_CALUDE_jamie_lost_balls_jamie_lost_six_balls_l1733_173302


namespace NUMINAMATH_CALUDE_distance_to_line_l1733_173349

/-- The point from which we're measuring the distance -/
def P : ℝ × ℝ × ℝ := (0, 1, 5)

/-- The point on the line -/
def Q : ℝ → ℝ × ℝ × ℝ := λ t => (4 + 3*t, 5 - t, 6 + 2*t)

/-- The direction vector of the line -/
def v : ℝ × ℝ × ℝ := (3, -1, 2)

/-- The distance from a point to a line -/
def distanceToLine (P : ℝ × ℝ × ℝ) (Q : ℝ → ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) : ℝ := 
  sorry

theorem distance_to_line : 
  distanceToLine P Q v = Real.sqrt 1262 / 7 := by sorry

end NUMINAMATH_CALUDE_distance_to_line_l1733_173349


namespace NUMINAMATH_CALUDE_mike_notebooks_count_l1733_173328

theorem mike_notebooks_count :
  ∀ (total_spent blue_cost : ℕ) (red_count green_count : ℕ) (red_cost green_cost : ℕ),
    total_spent = 37 →
    red_count = 3 →
    green_count = 2 →
    red_cost = 4 →
    green_cost = 2 →
    blue_cost = 3 →
    total_spent = red_count * red_cost + green_count * green_cost + 
      ((total_spent - (red_count * red_cost + green_count * green_cost)) / blue_cost) * blue_cost →
    red_count + green_count + (total_spent - (red_count * red_cost + green_count * green_cost)) / blue_cost = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_mike_notebooks_count_l1733_173328


namespace NUMINAMATH_CALUDE_min_value_abs_function_l1733_173309

theorem min_value_abs_function (x : ℝ) :
  ∀ x, |x - 1| + |x - 2| - |x - 3| ≥ -1 ∧ ∃ x, |x - 1| + |x - 2| - |x - 3| = -1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_abs_function_l1733_173309


namespace NUMINAMATH_CALUDE_no_equilateral_right_triangle_l1733_173360

theorem no_equilateral_right_triangle :
  ¬ ∃ (a b c : ℝ) (A B C : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive side lengths
    A > 0 ∧ B > 0 ∧ C > 0 ∧  -- Positive angles
    a = b ∧ b = c ∧          -- Equilateral condition
    A = 90 ∧                 -- Right angle condition
    A + B + C = 180          -- Sum of angles in a triangle
    := by sorry

end NUMINAMATH_CALUDE_no_equilateral_right_triangle_l1733_173360


namespace NUMINAMATH_CALUDE_map_distance_calculation_l1733_173389

/-- Calculates the distance on a map given travel time, speed, and map scale -/
theorem map_distance_calculation (travel_time : ℝ) (average_speed : ℝ) (map_scale : ℝ) :
  travel_time = 6.5 →
  average_speed = 60 →
  map_scale = 0.01282051282051282 →
  travel_time * average_speed * map_scale = 5 := by
  sorry

end NUMINAMATH_CALUDE_map_distance_calculation_l1733_173389


namespace NUMINAMATH_CALUDE_angle_B_is_pi_over_six_l1733_173331

-- Define a triangle ABC
def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the theorem
theorem angle_B_is_pi_over_six 
  (a b c : ℝ) 
  (h_triangle : Triangle a b c) 
  (h_condition : 2 * b * Real.cos (Real.arccos ((b^2 + c^2 - a^2) / (2*b*c))) = 2 * c - Real.sqrt 3 * a) :
  Real.arccos ((c^2 + a^2 - b^2) / (2*a*c)) = π / 6 := by
sorry


end NUMINAMATH_CALUDE_angle_B_is_pi_over_six_l1733_173331


namespace NUMINAMATH_CALUDE_division_scaling_l1733_173335

theorem division_scaling (a b q r k : ℤ) (h1 : a = b * q + r) (h2 : 0 ≤ r ∧ r < b) (h3 : k ≠ 0) :
  (∃ q' r', a * k = (b * k) * q' + r' ∧ q' = q ∧ r' = r * k ∧ 0 ≤ r' ∧ r' < b * k) ∧
  (k ∣ r → ∃ q' r', a / k = (b / k) * q' + r' ∧ q' = q ∧ r' = r / k ∧ 0 ≤ r' ∧ r' < b / k) :=
sorry

end NUMINAMATH_CALUDE_division_scaling_l1733_173335


namespace NUMINAMATH_CALUDE_birdseed_theorem_l1733_173301

/-- Calculates the amount of birdseed Peter needs to buy for a week -/
def birdseed_for_week : ℕ :=
  let parakeet_daily_consumption : ℕ := 2
  let parrot_daily_consumption : ℕ := 14
  let finch_daily_consumption : ℕ := parakeet_daily_consumption / 2
  let num_parakeets : ℕ := 3
  let num_parrots : ℕ := 2
  let num_finches : ℕ := 4
  let days_in_week : ℕ := 7
  
  let total_daily_consumption : ℕ := 
    num_parakeets * parakeet_daily_consumption +
    num_parrots * parrot_daily_consumption +
    num_finches * finch_daily_consumption

  total_daily_consumption * days_in_week

/-- Theorem stating that the amount of birdseed Peter needs to buy for a week is 266 grams -/
theorem birdseed_theorem : birdseed_for_week = 266 := by
  sorry

end NUMINAMATH_CALUDE_birdseed_theorem_l1733_173301


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_54_l1733_173352

theorem gcd_lcm_product_24_54 : Nat.gcd 24 54 * Nat.lcm 24 54 = 1296 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_54_l1733_173352


namespace NUMINAMATH_CALUDE_point_on_segment_coordinates_l1733_173310

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line segment between two other points -/
def lies_on_segment (p q r : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    p.x = q.x + t * (r.x - q.x) ∧
    p.y = q.y + t * (r.y - q.y)

theorem point_on_segment_coordinates :
  let K : Point := ⟨4, 2⟩
  let M : Point := ⟨10, 11⟩
  let L : Point := ⟨6, w⟩
  lies_on_segment L K M → w = 5 := by
sorry

end NUMINAMATH_CALUDE_point_on_segment_coordinates_l1733_173310


namespace NUMINAMATH_CALUDE_function_inequality_l1733_173308

theorem function_inequality (a b c : ℝ) (f : ℝ → ℝ) 
  (h1 : a < b) (h2 : b < c)
  (h3 : ∀ x, f x = |Real.log x|)
  (h4 : f a > f c) (h5 : f c > f b) : 
  (a - 1) * (c - 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1733_173308


namespace NUMINAMATH_CALUDE_factorization_problem1_l1733_173365

theorem factorization_problem1 (a b : ℝ) : -3 * a^2 + 6 * a * b - 3 * b^2 = -3 * (a - b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem1_l1733_173365


namespace NUMINAMATH_CALUDE_range_of_m_chord_length_l1733_173356

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Theorem for the range of m
theorem range_of_m :
  ∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) → m < 5 :=
sorry

-- Theorem for the length of chord MN when m = 4
theorem chord_length :
  let m : ℝ := 4
  ∃ M N : ℝ × ℝ,
    circle_equation M.1 M.2 m ∧
    circle_equation N.1 N.2 m ∧
    line_equation M.1 M.2 ∧
    line_equation N.1 N.2 ∧
    Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 4 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_chord_length_l1733_173356


namespace NUMINAMATH_CALUDE_handshake_count_is_correct_handshakes_per_person_is_correct_l1733_173345

/-- Represents a social gathering with married couples -/
structure SocialGathering where
  couples : ℕ
  people : ℕ
  handshakes_per_person : ℕ

/-- Calculate the total number of unique handshakes in the gathering -/
def total_handshakes (g : SocialGathering) : ℕ :=
  g.people * g.handshakes_per_person / 2

/-- The specific social gathering described in the problem -/
def our_gathering : SocialGathering :=
  { couples := 8
  , people := 16
  , handshakes_per_person := 12 }

theorem handshake_count_is_correct :
  total_handshakes our_gathering = 96 := by
  sorry

/-- Prove that the number of handshakes per person is correct -/
theorem handshakes_per_person_is_correct (g : SocialGathering) :
  g.handshakes_per_person = g.people - 1 - 3 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_is_correct_handshakes_per_person_is_correct_l1733_173345


namespace NUMINAMATH_CALUDE_function_inequality_implies_k_range_l1733_173318

theorem function_inequality_implies_k_range (k : ℝ) : 
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 3, ∃ x₀ ∈ Set.Icc (-1 : ℝ) 3, 
    2 * x₁^2 + x₁ - k ≤ x₀^3 - 3 * x₀) → 
  k ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_k_range_l1733_173318


namespace NUMINAMATH_CALUDE_cost_calculation_l1733_173398

/-- The total cost of buying apples and bananas -/
def total_cost (a b : ℝ) : ℝ := 2 * a + 3 * b

/-- Theorem: The total cost of buying 2 kg of apples at 'a' yuan/kg and 3 kg of bananas at 'b' yuan/kg is (2a + 3b) yuan -/
theorem cost_calculation (a b : ℝ) :
  total_cost a b = 2 * a + 3 * b := by
  sorry

end NUMINAMATH_CALUDE_cost_calculation_l1733_173398


namespace NUMINAMATH_CALUDE_cube_minus_reciprocal_cube_l1733_173313

theorem cube_minus_reciprocal_cube (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 := by
  sorry

end NUMINAMATH_CALUDE_cube_minus_reciprocal_cube_l1733_173313


namespace NUMINAMATH_CALUDE_tv_cost_l1733_173375

def lindas_savings : ℚ := 960

theorem tv_cost (furniture_fraction : ℚ) (h1 : furniture_fraction = 3 / 4) :
  (1 - furniture_fraction) * lindas_savings = 240 := by
  sorry

end NUMINAMATH_CALUDE_tv_cost_l1733_173375


namespace NUMINAMATH_CALUDE_large_lemonhead_doll_cost_l1733_173332

/-- The cost of a large lemonhead doll satisfies the given conditions -/
theorem large_lemonhead_doll_cost :
  ∃ (L : ℝ), 
    (L > 0) ∧ 
    (350 / (L - 2) = 350 / L + 20) ∧ 
    (L = 7) := by
  sorry

end NUMINAMATH_CALUDE_large_lemonhead_doll_cost_l1733_173332


namespace NUMINAMATH_CALUDE_sum_of_products_bound_l1733_173376

theorem sum_of_products_bound (a b c : ℝ) (h : a + b + c = 1) :
  0 ≤ a * b + a * c + b * c ∧ a * b + a * c + b * c ≤ 1/3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_bound_l1733_173376


namespace NUMINAMATH_CALUDE_florist_roses_l1733_173304

theorem florist_roses (initial : ℕ) (sold : ℕ) (picked : ℕ) : 
  initial = 37 → sold = 16 → picked = 19 → initial - sold + picked = 40 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_l1733_173304


namespace NUMINAMATH_CALUDE_water_tank_capacity_l1733_173322

/-- The capacity of a water tank given its filling rate and time to reach 3/4 capacity -/
theorem water_tank_capacity (fill_rate : ℝ) (time_to_three_quarters : ℝ) 
  (h1 : fill_rate = 10) 
  (h2 : time_to_three_quarters = 300) : 
  fill_rate * time_to_three_quarters / (3/4) = 4000 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l1733_173322


namespace NUMINAMATH_CALUDE_f_composition_of_five_l1733_173320

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 5 * x + 1

theorem f_composition_of_five : f (f (f (f (f 5)))) = 166 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_of_five_l1733_173320


namespace NUMINAMATH_CALUDE_no_even_three_digit_sum_27_l1733_173351

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem no_even_three_digit_sum_27 :
  ¬ ∃ n : ℕ, is_three_digit n ∧ digit_sum n = 27 ∧ Even n :=
sorry

end NUMINAMATH_CALUDE_no_even_three_digit_sum_27_l1733_173351


namespace NUMINAMATH_CALUDE_ribbons_given_in_afternoon_l1733_173337

/-- Given the initial number of ribbons, the number given away in the morning,
    and the number left at the end, prove that the number of ribbons given away
    in the afternoon is 16. -/
theorem ribbons_given_in_afternoon
  (initial : ℕ)
  (morning : ℕ)
  (left : ℕ)
  (h1 : initial = 38)
  (h2 : morning = 14)
  (h3 : left = 8) :
  initial - morning - left = 16 := by
  sorry

end NUMINAMATH_CALUDE_ribbons_given_in_afternoon_l1733_173337


namespace NUMINAMATH_CALUDE_base_ten_satisfies_equation_l1733_173380

/-- Given a base b, converts a number in base b to decimal --/
def toDecimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun digit acc => digit + b * acc) 0

/-- Checks if the equation 253_b + 176_b = 431_b holds for a given base b --/
def equationHolds (b : Nat) : Prop :=
  toDecimal [2, 5, 3] b + toDecimal [1, 7, 6] b = toDecimal [4, 3, 1] b

theorem base_ten_satisfies_equation :
  equationHolds 10 ∧ ∀ b : Nat, b ≠ 10 → ¬equationHolds b :=
sorry

end NUMINAMATH_CALUDE_base_ten_satisfies_equation_l1733_173380


namespace NUMINAMATH_CALUDE_house_painting_theorem_l1733_173341

/-- Represents the number of worker-hours required to paint a house -/
def totalWorkerHours : ℕ := 32

/-- Represents the number of people who started painting -/
def initialWorkers : ℕ := 6

/-- Represents the number of hours the initial workers painted -/
def initialHours : ℕ := 2

/-- Represents the total time available to paint the house -/
def totalTime : ℕ := 4

/-- Calculates the number of additional workers needed to complete the painting -/
def additionalWorkersNeeded : ℕ :=
  (totalWorkerHours - initialWorkers * initialHours) / (totalTime - initialHours) - initialWorkers

theorem house_painting_theorem :
  additionalWorkersNeeded = 4 := by
  sorry

#eval additionalWorkersNeeded

end NUMINAMATH_CALUDE_house_painting_theorem_l1733_173341


namespace NUMINAMATH_CALUDE_cube_root_problem_l1733_173361

theorem cube_root_problem (a : ℝ) (h : a^3 = 21 * 25 * 15 * 147) : a = 105 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_problem_l1733_173361


namespace NUMINAMATH_CALUDE_expression_simplification_l1733_173325

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 5 - 1) :
  (x / (x - 1) - 1) / ((x^2 - 1) / (x^2 - 2*x + 1)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1733_173325


namespace NUMINAMATH_CALUDE_p_plus_q_equals_twenty_l1733_173390

theorem p_plus_q_equals_twenty (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 3 → P / (x - 3) + Q * (x + 2) = (-5 * x^2 + 18 * x + 40) / (x - 3)) →
  P + Q = 20 := by
sorry

end NUMINAMATH_CALUDE_p_plus_q_equals_twenty_l1733_173390


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_l1733_173388

theorem min_value_sqrt_sum (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : a * b + b * c + c * a = a + b + c) (h5 : 0 < a + b + c) :
  2 ≤ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) ∧
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧
    a' * b' + b' * c' + c' * a' = a' + b' + c' ∧ 0 < a' + b' + c' ∧
    Real.sqrt (a' * b') + Real.sqrt (b' * c') + Real.sqrt (c' * a') = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_l1733_173388


namespace NUMINAMATH_CALUDE_not_always_swappable_renumbering_l1733_173358

-- Define a type for cities
def City : Type := ℕ

-- Define a type for the connection list
def ConnectionList : Type := List (City × City)

-- Function to check if a list is valid (placeholder)
def isValidList (list : ConnectionList) : Prop := sorry

-- Function to represent renumbering of cities
def renumber (oldNum newNum : City) (list : ConnectionList) : ConnectionList := sorry

-- Theorem statement
theorem not_always_swappable_renumbering :
  ∃ (list : ConnectionList) (M N : City),
    isValidList list ∧
    (∀ X Y : City, isValidList (renumber X Y list)) ∧
    ¬(isValidList (renumber M N (renumber N M list))) :=
sorry

end NUMINAMATH_CALUDE_not_always_swappable_renumbering_l1733_173358


namespace NUMINAMATH_CALUDE_origin_not_in_convex_hull_probability_l1733_173324

/-- The unit circle in the complex plane -/
def S1 : Set ℂ := {z : ℂ | Complex.abs z = 1}

/-- The probability that the origin is not contained in the convex hull of n randomly selected points from S¹ -/
noncomputable def probability (n : ℕ) : ℝ := 1 - (n : ℝ) / 2^(n - 1)

/-- Theorem: The probability that the origin is not contained in the convex hull of seven randomly selected points from S¹ is 57/64 -/
theorem origin_not_in_convex_hull_probability :
  probability 7 = 57 / 64 := by sorry

end NUMINAMATH_CALUDE_origin_not_in_convex_hull_probability_l1733_173324


namespace NUMINAMATH_CALUDE_distance_between_hyperbola_and_ellipse_l1733_173364

theorem distance_between_hyperbola_and_ellipse 
  (x y z w : ℝ) 
  (h1 : x * y = 4) 
  (h2 : z^2 + 4 * w^2 = 4) : 
  (x - z)^2 + (y - w)^2 ≥ 1.6 := by
sorry

end NUMINAMATH_CALUDE_distance_between_hyperbola_and_ellipse_l1733_173364


namespace NUMINAMATH_CALUDE_cubic_identity_l1733_173372

theorem cubic_identity (x : ℝ) : (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_l1733_173372


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1733_173397

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x > 0, x^2 - a*x + 1 > 0) → a ∈ Set.Ioo (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1733_173397


namespace NUMINAMATH_CALUDE_jerry_total_games_l1733_173319

/-- The total number of video games Jerry has after his birthday -/
def total_games (initial_games birthday_games : ℕ) : ℕ :=
  initial_games + birthday_games

/-- Theorem: Jerry has 9 video games in total -/
theorem jerry_total_games :
  total_games 7 2 = 9 := by sorry

end NUMINAMATH_CALUDE_jerry_total_games_l1733_173319


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l1733_173300

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1: Solution set when a = 2
theorem solution_set_when_a_is_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} :=
sorry

-- Part 2: Range of values for a
theorem range_of_a (h : ∀ x : ℝ, f x a ≥ 4) :
  a ≤ -1 ∨ a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l1733_173300


namespace NUMINAMATH_CALUDE_expression_evaluation_l1733_173373

theorem expression_evaluation :
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1733_173373


namespace NUMINAMATH_CALUDE_rosas_total_flowers_l1733_173330

/-- Rosa's total number of flowers after receiving more from Andre -/
theorem rosas_total_flowers (initial : ℝ) (received : ℝ) (total : ℝ)
  (h1 : initial = 67.0)
  (h2 : received = 90.0)
  (h3 : total = initial + received) :
  total = 157.0 := by sorry

end NUMINAMATH_CALUDE_rosas_total_flowers_l1733_173330


namespace NUMINAMATH_CALUDE_minor_axis_length_of_ellipse_l1733_173396

/-- The length of the minor axis of the ellipse x^2/4 + y^2/36 = 1 is 4 -/
theorem minor_axis_length_of_ellipse : 
  let ellipse := (fun (x y : ℝ) => x^2/4 + y^2/36 = 1)
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
    (∀ x y, ellipse x y ↔ x^2/a^2 + y^2/b^2 = 1) ∧
    2 * min a b = 4 :=
by sorry

end NUMINAMATH_CALUDE_minor_axis_length_of_ellipse_l1733_173396
