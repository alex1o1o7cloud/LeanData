import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l2030_203013

theorem equation_solution : 
  {x : ℝ | x^6 + (3 - x)^6 = 730} = {1.5 + Real.sqrt 5, 1.5 - Real.sqrt 5} :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2030_203013


namespace NUMINAMATH_CALUDE_cycling_speed_l2030_203039

/-- The speed of Alice and Bob when cycling under specific conditions -/
theorem cycling_speed : ∃ (x : ℝ),
  (x^2 - 5*x - 14 = (x^2 + x - 20) / (x - 4)) ∧
  (x^2 - 5*x - 14 = 8 + 2*Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_cycling_speed_l2030_203039


namespace NUMINAMATH_CALUDE_initial_overs_played_l2030_203090

/-- Proves that the number of overs played initially is 10, given the specified conditions --/
theorem initial_overs_played (total_target : ℝ) (initial_run_rate : ℝ) (remaining_overs : ℝ) (required_run_rate : ℝ)
  (h1 : total_target = 282)
  (h2 : initial_run_rate = 3.8)
  (h3 : remaining_overs = 40)
  (h4 : required_run_rate = 6.1)
  : ∃ (x : ℝ), x = 10 ∧ initial_run_rate * x + required_run_rate * remaining_overs = total_target :=
by
  sorry

end NUMINAMATH_CALUDE_initial_overs_played_l2030_203090


namespace NUMINAMATH_CALUDE_tangent_line_condition_max_ab_value_l2030_203057

noncomputable section

/-- The function f(x) = ln(ax + b) + x^2 -/
def f (a b x : ℝ) : ℝ := Real.log (a * x + b) + x^2

/-- The derivative of f with respect to x -/
def f_deriv (a b x : ℝ) : ℝ := a / (a * x + b) + 2 * x

theorem tangent_line_condition (a b : ℝ) (h1 : a ≠ 0) :
  (f_deriv a b 1 = 1 ∧ f a b 1 = 1) → (a = -1 ∧ b = 2) :=
sorry

theorem max_ab_value (a b : ℝ) (h1 : a ≠ 0) :
  (∀ x, f a b x ≤ x^2 + x) → (a * b ≤ Real.exp 1 / 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_condition_max_ab_value_l2030_203057


namespace NUMINAMATH_CALUDE_fox_initial_coins_l2030_203017

/-- The number of times Fox crosses the bridge -/
def num_crossings : ℕ := 3

/-- The toll Fox pays after each crossing -/
def toll : ℚ := 50

/-- The final amount Fox wants to have -/
def final_amount : ℚ := 50

/-- The factor by which Fox's money is multiplied each crossing -/
def multiplier : ℚ := 3

theorem fox_initial_coins (x : ℚ) :
  (((x * multiplier - toll) * multiplier - toll) * multiplier - toll = final_amount) →
  (x = 700 / 27) :=
by sorry

end NUMINAMATH_CALUDE_fox_initial_coins_l2030_203017


namespace NUMINAMATH_CALUDE_set_equality_l2030_203016

-- Define the set A
def A : Set ℝ := {x : ℝ | 2 * x^2 + x - 3 = 0}

-- Define the set B
def B : Set ℝ := {i : ℝ | i^2 ≥ 4}

-- Define the complement of set C in real numbers
def compl_C : Set ℝ := {-1, 1, 3/2}

-- Theorem statement
theorem set_equality : A ∩ B ∪ compl_C = {-1, 1, 3/2} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l2030_203016


namespace NUMINAMATH_CALUDE_initial_total_marbles_l2030_203093

/-- Represents the number of parts in the ratio for each person -/
def brittany_ratio : ℕ := 3
def alex_ratio : ℕ := 5
def jamy_ratio : ℕ := 7

/-- Represents the total number of marbles Alex has after receiving half of Brittany's marbles -/
def alex_final_marbles : ℕ := 260

/-- The theorem stating the initial total number of marbles -/
theorem initial_total_marbles :
  ∃ (x : ℕ),
    (brittany_ratio * x + alex_ratio * x + jamy_ratio * x = 600) ∧
    (alex_ratio * x + (brittany_ratio * x) / 2 = alex_final_marbles) :=
by sorry

end NUMINAMATH_CALUDE_initial_total_marbles_l2030_203093


namespace NUMINAMATH_CALUDE_no_four_digit_square_palindromes_l2030_203029

/-- A function that checks if a natural number is a 4-digit number -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that checks if a natural number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- A function that checks if a natural number is a palindrome -/
def is_palindrome (n : ℕ) : Prop := 
  let digits := n.digits 10
  digits = digits.reverse

/-- Theorem stating that there are no 4-digit square numbers that are palindromes -/
theorem no_four_digit_square_palindromes : 
  ¬∃ n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end NUMINAMATH_CALUDE_no_four_digit_square_palindromes_l2030_203029


namespace NUMINAMATH_CALUDE_photo_arrangement_count_l2030_203031

/-- The number of ways to arrange 2 teachers and 4 students in a row -/
def arrangementCount (n : ℕ) (m : ℕ) (k : ℕ) : ℕ :=
  if n = 2 ∧ m = 4 ∧ k = 1 then
    Nat.factorial 2 * 2 * Nat.factorial 3
  else
    0

/-- Theorem stating the correct number of arrangements -/
theorem photo_arrangement_count :
  arrangementCount 2 4 1 = 24 :=
by sorry

end NUMINAMATH_CALUDE_photo_arrangement_count_l2030_203031


namespace NUMINAMATH_CALUDE_quadratic_one_solution_sum_l2030_203018

theorem quadratic_one_solution_sum (b₁ b₂ : ℝ) : 
  (∀ x, 3 * x^2 + b₁ * x + 12 * x + 16 = 0 → (b₁ + 12)^2 = 4 * 3 * 16) ∧
  (∀ x, 3 * x^2 + b₂ * x + 12 * x + 16 = 0 → (b₂ + 12)^2 = 4 * 3 * 16) →
  b₁ + b₂ = -24 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_sum_l2030_203018


namespace NUMINAMATH_CALUDE_polynomial_expansion_theorem_l2030_203056

theorem polynomial_expansion_theorem (N : ℕ) : 
  (Nat.choose N 5 = 2002) ↔ (N = 17) := by sorry

#check polynomial_expansion_theorem

end NUMINAMATH_CALUDE_polynomial_expansion_theorem_l2030_203056


namespace NUMINAMATH_CALUDE_solve_for_q_l2030_203073

theorem solve_for_q (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 20) 
  (eq2 : 6 * p + 5 * q = 29) : 
  q = -25 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_q_l2030_203073


namespace NUMINAMATH_CALUDE_solution_exists_l2030_203054

theorem solution_exists (R₀ : ℝ) : ∃ x₁ x₂ x₃ : ℤ,
  x₁ > ⌊R₀⌋ ∧ x₂ > ⌊R₀⌋ ∧ x₃ > ⌊R₀⌋ ∧ x₁^2 + x₂^2 + x₃^2 = x₁ * x₂ * x₃ := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_l2030_203054


namespace NUMINAMATH_CALUDE_ellipse_with_foci_on_y_axis_range_l2030_203096

/-- The equation of the curve -/
def equation (x y k : ℝ) : Prop := x^2 / (k - 5) + y^2 / (10 - k) = 1

/-- The condition for the equation to represent an ellipse -/
def is_ellipse (k : ℝ) : Prop := k - 5 > 0 ∧ 10 - k > 0

/-- The condition for the foci to be on the y-axis -/
def foci_on_y_axis (k : ℝ) : Prop := 10 - k > k - 5

/-- The theorem stating the range of k for which the equation represents an ellipse with foci on the y-axis -/
theorem ellipse_with_foci_on_y_axis_range (k : ℝ) :
  is_ellipse k ∧ foci_on_y_axis k ↔ k ∈ Set.Ioo 5 7.5 :=
sorry

end NUMINAMATH_CALUDE_ellipse_with_foci_on_y_axis_range_l2030_203096


namespace NUMINAMATH_CALUDE_value_of_b_l2030_203074

theorem value_of_b (a b : ℝ) (h1 : 4 * a^2 + 1 = 1) (h2 : b - a = 3) : b = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l2030_203074


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2030_203027

/-- 
If the quadratic equation x^2 + 6x + c = 0 has two equal real roots,
then c = 9.
-/
theorem equal_roots_quadratic (c : ℝ) : 
  (∃ x : ℝ, x^2 + 6*x + c = 0 ∧ 
   ∀ y : ℝ, y^2 + 6*y + c = 0 → y = x) → 
  c = 9 :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2030_203027


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l2030_203044

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers,
    with at least one object in each container. -/
def distribute_with_minimum (n k : ℕ) : ℕ :=
  Nat.choose (n - k + k - 1) (k - 1)

/-- Theorem stating that there are 6 ways to distribute 5 scoops into 3 flavors
    with at least one scoop of each flavor. -/
theorem ice_cream_flavors :
  distribute_with_minimum 5 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l2030_203044


namespace NUMINAMATH_CALUDE_probability_theorem_l2030_203019

/-- Represents the number of letters in each name -/
def letters_per_name : ℕ := 5

/-- Represents the total number of cards -/
def total_cards : ℕ := 15

/-- Represents the number of cards selected -/
def cards_selected : ℕ := 3

/-- Represents the number of different ways to select one letter from each name -/
def selection_arrangements : ℕ := 6

/-- Calculates the probability of selecting one letter from each of three names -/
def probability_one_from_each : ℚ :=
  selection_arrangements * (letters_per_name : ℚ) / total_cards *
  (letters_per_name : ℚ) / (total_cards - 1) *
  (letters_per_name : ℚ) / (total_cards - 2)

theorem probability_theorem :
  probability_one_from_each = 125 / 455 :=
sorry

end NUMINAMATH_CALUDE_probability_theorem_l2030_203019


namespace NUMINAMATH_CALUDE_display_window_problem_l2030_203077

/-- The number of configurations for two display windows --/
def total_configurations : ℕ := 36

/-- The number of non-fiction books in the right window --/
def non_fiction_books : ℕ := 3

/-- The number of fiction books in the left window --/
def fiction_books : ℕ := 3

theorem display_window_problem :
  fiction_books.factorial * non_fiction_books.factorial = total_configurations :=
sorry

end NUMINAMATH_CALUDE_display_window_problem_l2030_203077


namespace NUMINAMATH_CALUDE_axis_of_symmetry_parabola_axis_of_symmetry_specific_parabola_l2030_203040

/-- The axis of symmetry of a parabola y = ax^2 + bx + c is the line x = -b/(2a) -/
theorem axis_of_symmetry_parabola (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  ∀ x, f (x + (-b / (2 * a))) = f (-b / (2 * a) - x) :=
sorry

/-- The axis of symmetry of the parabola y = -x^2 + 2022 is the line x = 0 -/
theorem axis_of_symmetry_specific_parabola :
  let f : ℝ → ℝ := λ x ↦ -x^2 + 2022
  ∀ x, f (x + 0) = f (0 - x) :=
sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_parabola_axis_of_symmetry_specific_parabola_l2030_203040


namespace NUMINAMATH_CALUDE_sufficient_condition_for_quadratic_inequality_l2030_203025

theorem sufficient_condition_for_quadratic_inequality (a : ℝ) :
  (∀ x : ℝ, x > a → x^2 > 2*x) ∧
  (∃ x : ℝ, x^2 > 2*x ∧ x ≤ a) →
  a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_quadratic_inequality_l2030_203025


namespace NUMINAMATH_CALUDE_contest_result_l2030_203058

/-- The number of times Frannie jumped -/
def frannies_jumps : ℕ := 53

/-- The difference between Meg's and Frannie's jumps -/
def jump_difference : ℕ := 18

/-- Meg's number of jumps -/
def megs_jumps : ℕ := frannies_jumps + jump_difference

theorem contest_result : megs_jumps = 71 := by
  sorry

end NUMINAMATH_CALUDE_contest_result_l2030_203058


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2030_203050

theorem rectangle_perimeter (length width : ℝ) (h1 : length = 3 * width) (h2 : length * width = 147) :
  2 * (length + width) = 56 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2030_203050


namespace NUMINAMATH_CALUDE_min_area_triangle_min_area_is_minimum_l2030_203055

/-- The minimum area of a triangle with vertices (0,0), (30,18), and a third point with integer coordinates -/
theorem min_area_triangle : ℝ :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (30, 18)
  3

/-- The area of the triangle is indeed the minimum possible -/
theorem min_area_is_minimum (p q : ℤ) : 
  let C : ℝ × ℝ := (p, q)
  let area := (1/2 : ℝ) * |18 * p - 30 * q|
  3 ≤ area := by
  sorry

#check min_area_triangle
#check min_area_is_minimum

end NUMINAMATH_CALUDE_min_area_triangle_min_area_is_minimum_l2030_203055


namespace NUMINAMATH_CALUDE_sons_age_l2030_203000

/-- Prove that the son's current age is 24 years given the conditions -/
theorem sons_age (father_age son_age : ℕ) : 
  father_age = 3 * son_age →
  father_age - 8 = 4 * (son_age - 8) →
  son_age = 24 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l2030_203000


namespace NUMINAMATH_CALUDE_bryan_books_and_magazines_l2030_203068

theorem bryan_books_and_magazines (books_per_shelf : ℕ) (magazines_per_shelf : ℕ) (num_shelves : ℕ) :
  books_per_shelf = 23 →
  magazines_per_shelf = 61 →
  num_shelves = 29 →
  books_per_shelf * num_shelves + magazines_per_shelf * num_shelves = 2436 :=
by
  sorry

end NUMINAMATH_CALUDE_bryan_books_and_magazines_l2030_203068


namespace NUMINAMATH_CALUDE_ratio_of_sums_l2030_203001

theorem ratio_of_sums (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_squares_abc : a^2 + b^2 + c^2 = 49)
  (sum_squares_xyz : x^2 + y^2 + z^2 = 64)
  (dot_product : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_sums_l2030_203001


namespace NUMINAMATH_CALUDE_city_college_juniors_seniors_l2030_203080

theorem city_college_juniors_seniors (total : ℕ) (j s : ℕ) : 
  total = 300 →
  j + s = total →
  (1 : ℚ) / 3 * j = (2 : ℚ) / 3 * s →
  j - s = 100 :=
by sorry

end NUMINAMATH_CALUDE_city_college_juniors_seniors_l2030_203080


namespace NUMINAMATH_CALUDE_max_factors_upper_bound_max_factors_achievable_max_factors_is_maximum_l2030_203021

def max_factors (b n : ℕ+) : ℕ :=
  sorry

theorem max_factors_upper_bound (b n : ℕ+) (hb : b ≤ 15) (hn : n ≤ 20) :
  max_factors b n ≤ 861 :=
sorry

theorem max_factors_achievable :
  ∃ (b n : ℕ+), b ≤ 15 ∧ n ≤ 20 ∧ max_factors b n = 861 :=
sorry

theorem max_factors_is_maximum :
  ∀ (b n : ℕ+), b ≤ 15 → n ≤ 20 → max_factors b n ≤ 861 :=
sorry

end NUMINAMATH_CALUDE_max_factors_upper_bound_max_factors_achievable_max_factors_is_maximum_l2030_203021


namespace NUMINAMATH_CALUDE_interest_rates_equality_l2030_203023

theorem interest_rates_equality (initial_savings : ℝ) 
  (simple_interest : ℝ) (compound_interest : ℝ) : 
  initial_savings = 1000 ∧ 
  simple_interest = 100 ∧ 
  compound_interest = 105 →
  ∃ (r : ℝ), 
    simple_interest = (initial_savings / 2) * r * 2 ∧
    compound_interest = (initial_savings / 2) * ((1 + r)^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_interest_rates_equality_l2030_203023


namespace NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_five_l2030_203043

theorem least_positive_integer_to_multiple_of_five : 
  ∃ (n : ℕ), n > 0 ∧ (∀ m : ℕ, m > 0 → (624 + m) % 5 = 0 → m ≥ n) ∧ (624 + n) % 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_five_l2030_203043


namespace NUMINAMATH_CALUDE_green_balls_count_l2030_203063

theorem green_balls_count (total : ℕ) (red blue green : ℕ) : 
  red + blue + green = total →
  red = total / 3 →
  blue = (2 * total) / 7 →
  green = 2 * blue - 8 →
  green = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_green_balls_count_l2030_203063


namespace NUMINAMATH_CALUDE_calculate_required_hours_per_week_l2030_203065

/-- Proves that given an initial work plan and a period of unavailable work time,
    the required hours per week to meet the financial goal can be calculated. -/
theorem calculate_required_hours_per_week 
  (initial_hours_per_week : ℝ)
  (initial_weeks : ℝ)
  (financial_goal : ℝ)
  (unavailable_weeks : ℝ)
  (h1 : initial_hours_per_week = 25)
  (h2 : initial_weeks = 15)
  (h3 : financial_goal = 4500)
  (h4 : unavailable_weeks = 3)
  : (initial_hours_per_week * initial_weeks) / (initial_weeks - unavailable_weeks) = 31.25 := by
  sorry

end NUMINAMATH_CALUDE_calculate_required_hours_per_week_l2030_203065


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l2030_203045

theorem modular_arithmetic_problem : ((367 * 373 * 379 % 53) * 383) % 47 = 0 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l2030_203045


namespace NUMINAMATH_CALUDE_rectangle_strip_problem_l2030_203008

theorem rectangle_strip_problem (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * b + a * c + a * (b - a) + a * a + a * (c - a) = 43) :
  (a = 1 ∧ b + c = 22) ∨ (a = 22 ∧ b + c = 1) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_strip_problem_l2030_203008


namespace NUMINAMATH_CALUDE_race_time_differences_l2030_203048

def runner_A : ℕ := 60
def runner_B : ℕ := 100
def runner_C : ℕ := 80
def runner_D : ℕ := 120

def time_difference (t1 t2 : ℕ) : ℕ := 
  if t1 > t2 then t1 - t2 else t2 - t1

theorem race_time_differences : 
  (time_difference runner_A runner_B = 40) ∧
  (time_difference runner_A runner_C = 20) ∧
  (time_difference runner_A runner_D = 60) ∧
  (time_difference runner_B runner_C = 20) ∧
  (time_difference runner_B runner_D = 20) ∧
  (time_difference runner_C runner_D = 40) :=
by sorry

end NUMINAMATH_CALUDE_race_time_differences_l2030_203048


namespace NUMINAMATH_CALUDE_max_value_sum_l2030_203053

theorem max_value_sum (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0)
  (h_sum : a^2 + b^2 + c^2 + d^2 + e^2 = 504) :
  ∃ (N a_N b_N c_N d_N e_N : ℝ),
    (∀ x y z w v : ℝ, x > 0 → y > 0 → z > 0 → w > 0 → v > 0 → 
      x^2 + y^2 + z^2 + w^2 + v^2 = 504 → 
      x*z + 3*y*z + 4*z*w + 8*z*v ≤ N) ∧
    (a_N*c_N + 3*b_N*c_N + 4*c_N*d_N + 8*c_N*e_N = N) ∧
    (a_N^2 + b_N^2 + c_N^2 + d_N^2 + e_N^2 = 504) ∧
    (N + a_N + b_N + c_N + d_N + e_N = 32 + 756 * Real.sqrt 10 + 6 * Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_l2030_203053


namespace NUMINAMATH_CALUDE_books_per_shelf_l2030_203075

theorem books_per_shelf (total_books : ℕ) (num_shelves : ℕ) 
  (h1 : total_books = 315) (h2 : num_shelves = 7) : 
  total_books / num_shelves = 45 := by
  sorry

end NUMINAMATH_CALUDE_books_per_shelf_l2030_203075


namespace NUMINAMATH_CALUDE_equation_solution_l2030_203024

theorem equation_solution : ∃ x : ℚ, (5*x + 9*x = 450 - 10*(x - 5)) ∧ x = 125/6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2030_203024


namespace NUMINAMATH_CALUDE_mark_age_is_18_l2030_203072

/-- Represents the ages of family members --/
structure FamilyAges where
  mark : ℕ
  john : ℕ
  parents : ℕ

/-- Defines the relationships between family members' ages --/
def validFamilyAges (ages : FamilyAges) : Prop :=
  ages.john = ages.mark - 10 ∧
  ages.parents = 5 * ages.john ∧
  ages.parents - 22 = ages.mark

/-- Theorem stating that Mark's age is 18 given the family age relationships --/
theorem mark_age_is_18 :
  ∀ (ages : FamilyAges), validFamilyAges ages → ages.mark = 18 := by
  sorry

end NUMINAMATH_CALUDE_mark_age_is_18_l2030_203072


namespace NUMINAMATH_CALUDE_factors_of_42_l2030_203015

/-- The number of positive factors of 42 -/
def number_of_factors_42 : ℕ :=
  (Finset.filter (· ∣ 42) (Finset.range 43)).card

/-- Theorem stating that the number of positive factors of 42 is 8 -/
theorem factors_of_42 : number_of_factors_42 = 8 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_42_l2030_203015


namespace NUMINAMATH_CALUDE_existence_of_alpha_l2030_203081

theorem existence_of_alpha (p : Nat) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ α : Nat, 1 ≤ α ∧ α ≤ p - 2 ∧
    ¬(p^2 ∣ α^(p-1) - 1) ∧ ¬(p^2 ∣ (α+1)^(p-1) - 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_alpha_l2030_203081


namespace NUMINAMATH_CALUDE_multiple_remainder_l2030_203087

theorem multiple_remainder (n m : ℤ) (h1 : n % 7 = 1) (h2 : ∃ k, (k * n) % 7 = 3) :
  m % 7 = 3 → (m * n) % 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_multiple_remainder_l2030_203087


namespace NUMINAMATH_CALUDE_fifteen_factorial_base_eight_zeroes_l2030_203026

/-- The number of trailing zeroes in n! when written in base b --/
def trailingZeroes (n : ℕ) (b : ℕ) : ℕ :=
  sorry

/-- The factorial function --/
def factorial (n : ℕ) : ℕ :=
  sorry

theorem fifteen_factorial_base_eight_zeroes :
  trailingZeroes (factorial 15) 8 = 3 :=
sorry

end NUMINAMATH_CALUDE_fifteen_factorial_base_eight_zeroes_l2030_203026


namespace NUMINAMATH_CALUDE_gcd_lcm_45_75_l2030_203061

theorem gcd_lcm_45_75 :
  (Nat.gcd 45 75 = 15) ∧ (Nat.lcm 45 75 = 1125) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_45_75_l2030_203061


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2030_203005

theorem sufficient_not_necessary_condition :
  (∃ a b : ℝ, a < 0 ∧ -1 < b ∧ b < 0 → a + a * b < 0) ∧
  (∃ a b : ℝ, a + a * b < 0 ∧ ¬(a < 0 ∧ -1 < b ∧ b < 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2030_203005


namespace NUMINAMATH_CALUDE_min_value_is_four_l2030_203022

/-- The line passing through points A(3, 0) and B(1, 1) -/
def line_AB (x y : ℝ) : Prop := y = (x - 3) / (-2)

/-- The objective function to be minimized -/
def objective_function (x y : ℝ) : ℝ := 2 * x + 4 * y

/-- Theorem stating that the minimum value of the objective function is 4 -/
theorem min_value_is_four :
  ∀ x y : ℝ, line_AB x y → objective_function x y ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_is_four_l2030_203022


namespace NUMINAMATH_CALUDE_police_force_competition_l2030_203084

theorem police_force_competition (x y : ℕ) : 
  (70 * x + 60 * y = 740) → 
  ((x = 8 ∧ y = 3) ∨ (x = 2 ∧ y = 10)) := by
sorry

end NUMINAMATH_CALUDE_police_force_competition_l2030_203084


namespace NUMINAMATH_CALUDE_profit_sharing_l2030_203052

/-- The profit sharing problem -/
theorem profit_sharing
  (invest_a invest_b invest_c : ℝ)
  (total_profit : ℝ)
  (h1 : invest_a = 3 * invest_b)
  (h2 : invest_a = 2 / 3 * invest_c)
  (h3 : total_profit = 12375) :
  (invest_c / (invest_a + invest_b + invest_c)) * total_profit = (9 / 17) * 12375 := by
sorry

#eval (9 / 17 : ℚ) * 12375

end NUMINAMATH_CALUDE_profit_sharing_l2030_203052


namespace NUMINAMATH_CALUDE_arithmetic_progression_difference_divisibility_l2030_203007

theorem arithmetic_progression_difference_divisibility
  (p : ℕ) (a : ℕ → ℕ) (d : ℕ) 
  (h_p_prime : Nat.Prime p)
  (h_a_prime : ∀ i, i ∈ Finset.range p → Nat.Prime (a i))
  (h_arithmetic_progression : ∀ i, i ∈ Finset.range (p - 1) → a (i + 1) = a i + d)
  (h_increasing : ∀ i j, i < j → j < p → a i < a j)
  (h_a1_gt_p : a 0 > p) :
  p ∣ d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_difference_divisibility_l2030_203007


namespace NUMINAMATH_CALUDE_solve_for_m_l2030_203092

theorem solve_for_m : ∀ m : ℚ, 
  (∃ x y : ℚ, m * x + y = 2 ∧ x = -2 ∧ y = 1) → m = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l2030_203092


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_inequality_l2030_203083

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |a * x - 1|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x > 1} = {x : ℝ | x > 1/2} := by sorry

-- Part 2
theorem range_of_a_for_inequality :
  {a : ℝ | ∀ x ∈ Set.Ioo 0 1, f a x > x} = Set.Ioc 0 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_inequality_l2030_203083


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l2030_203099

/-- Three points are collinear if the slope between any two pairs of points is equal -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℚ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- Theorem: If (1,1), (-1,0), and (2,k) are collinear, then k = 3/2 -/
theorem collinear_points_k_value :
  collinear 1 1 (-1) 0 2 k → k = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l2030_203099


namespace NUMINAMATH_CALUDE_total_presents_equals_58_l2030_203082

/-- The number of presents Ethan has -/
def ethan_presents : ℕ := 31

/-- The number of presents Alissa has -/
def alissa_presents : ℕ := ethan_presents - 22

/-- The number of presents Bella has -/
def bella_presents : ℕ := 2 * alissa_presents

/-- The total number of presents Bella, Ethan, and Alissa have -/
def total_presents : ℕ := ethan_presents + alissa_presents + bella_presents

theorem total_presents_equals_58 : total_presents = 58 := by
  sorry

end NUMINAMATH_CALUDE_total_presents_equals_58_l2030_203082


namespace NUMINAMATH_CALUDE_square_equality_base_is_ten_l2030_203095

/-- The base in which 34 squared equals 1296 -/
def base_b : ℕ := sorry

/-- The representation of 34 in base b -/
def thirty_four_b (b : ℕ) : ℕ := 3 * b + 4

/-- The representation of 1296 in base b -/
def twelve_ninety_six_b (b : ℕ) : ℕ := b^3 + 2*b^2 + 9*b + 6

/-- The theorem stating that the square of 34 in base b equals 1296 in base b -/
theorem square_equality (b : ℕ) : (thirty_four_b b)^2 = twelve_ninety_six_b b := by sorry

/-- The main theorem proving that the base b is 10 -/
theorem base_is_ten : base_b = 10 := by sorry

end NUMINAMATH_CALUDE_square_equality_base_is_ten_l2030_203095


namespace NUMINAMATH_CALUDE_bucket_weight_l2030_203062

/-- Given a bucket where:
    - The weight when half full (including the bucket) is c
    - The weight when completely full (including the bucket) is d
    This theorem proves that the weight when three-quarters full is (1/2)c + (1/2)d -/
theorem bucket_weight (c d : ℝ) : ℝ :=
  let half_full := c
  let full := d
  let three_quarters_full := (1/2 : ℝ) * c + (1/2 : ℝ) * d
  three_quarters_full

#check bucket_weight

end NUMINAMATH_CALUDE_bucket_weight_l2030_203062


namespace NUMINAMATH_CALUDE_mystery_number_l2030_203004

theorem mystery_number : ∃ x : ℤ, x + 45 = 92 ∧ x = 47 := by
  sorry

end NUMINAMATH_CALUDE_mystery_number_l2030_203004


namespace NUMINAMATH_CALUDE_allan_total_balloons_l2030_203047

def initial_balloons : Nat := 5
def additional_balloons : Nat := 3

theorem allan_total_balloons : 
  initial_balloons + additional_balloons = 8 := by
  sorry

end NUMINAMATH_CALUDE_allan_total_balloons_l2030_203047


namespace NUMINAMATH_CALUDE_three_consecutive_heads_sequences_l2030_203037

def coin_flip_sequence (n : ℕ) : ℕ := 2^n

def no_three_consecutive_heads : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | n + 3 => no_three_consecutive_heads (n + 2) + no_three_consecutive_heads (n + 1) + no_three_consecutive_heads n

theorem three_consecutive_heads_sequences (n : ℕ) (h : n = 10) :
  coin_flip_sequence n - no_three_consecutive_heads n = 520 := by
  sorry

end NUMINAMATH_CALUDE_three_consecutive_heads_sequences_l2030_203037


namespace NUMINAMATH_CALUDE_units_digit_of_2189_power_1242_l2030_203069

theorem units_digit_of_2189_power_1242 : ∃ n : ℕ, 2189^1242 ≡ 1 [ZMOD 10] :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_2189_power_1242_l2030_203069


namespace NUMINAMATH_CALUDE_all_triangles_congruent_l2030_203035

/-- Represents a square tablecloth with hanging triangles -/
structure Tablecloth where
  -- Side length of the square tablecloth
  side : ℝ
  -- Heights of the hanging triangles
  hA : ℝ
  hB : ℝ
  hC : ℝ
  hD : ℝ
  -- Condition that all heights are positive
  hA_pos : hA > 0
  hB_pos : hB > 0
  hC_pos : hC > 0
  hD_pos : hD > 0
  -- Condition that △A and △B are congruent (given)
  hA_eq_hB : hA = hB

/-- Theorem stating that if △A and △B are congruent, then all hanging triangles are congruent -/
theorem all_triangles_congruent (t : Tablecloth) :
  t.hA = t.hB ∧ t.hA = t.hC ∧ t.hA = t.hD :=
sorry

end NUMINAMATH_CALUDE_all_triangles_congruent_l2030_203035


namespace NUMINAMATH_CALUDE_isosceles_diagonal_probability_l2030_203033

/-- The probability of selecting two diagonals from a regular pentagon 
    such that they form the two legs of an isosceles triangle -/
theorem isosceles_diagonal_probability (n m : ℕ) : 
  n = 10 → m = 5 → (m : ℚ) / n = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_isosceles_diagonal_probability_l2030_203033


namespace NUMINAMATH_CALUDE_arithmetic_and_geometric_sequence_l2030_203003

theorem arithmetic_and_geometric_sequence (a b c : ℝ) :
  (b - a = c - b) → -- arithmetic sequence condition
  (b / a = c / b) → -- geometric sequence condition
  (a ≠ 0) →         -- non-zero condition for geometric sequence
  (a = b ∧ b = c ∧ a ≠ 0) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_and_geometric_sequence_l2030_203003


namespace NUMINAMATH_CALUDE_gcd_of_324_243_135_l2030_203032

theorem gcd_of_324_243_135 : Nat.gcd 324 (Nat.gcd 243 135) = 27 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_324_243_135_l2030_203032


namespace NUMINAMATH_CALUDE_first_group_students_l2030_203046

theorem first_group_students (total : ℕ) (group2 group3 group4 : ℕ) 
  (h1 : total = 24)
  (h2 : group2 = 8)
  (h3 : group3 = 7)
  (h4 : group4 = 4) :
  total - (group2 + group3 + group4) = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_group_students_l2030_203046


namespace NUMINAMATH_CALUDE_units_digit_sum_base8_l2030_203089

/-- The units digit of a number in a given base -/
def unitsDigit (n : ℕ) (base : ℕ) : ℕ := n % base

/-- Addition in a given base -/
def addInBase (a b base : ℕ) : ℕ := (a + b) % base

theorem units_digit_sum_base8 :
  unitsDigit (addInBase 45 37 8) 8 = 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_base8_l2030_203089


namespace NUMINAMATH_CALUDE_second_week_cut_percentage_sculpture_problem_l2030_203041

/-- Calculates the percentage of marble cut away in the second week of sculpting -/
theorem second_week_cut_percentage (initial_weight : ℝ) (first_week_cut : ℝ) 
  (third_week_cut : ℝ) (final_weight : ℝ) : ℝ :=
  let remaining_after_first := initial_weight * (1 - first_week_cut / 100)
  let second_week_cut := 100 * (1 - (final_weight / (remaining_after_first * (1 - third_week_cut / 100))))
  second_week_cut

/-- The percentage of marble cut away in the second week is 30% -/
theorem sculpture_problem :
  second_week_cut_percentage 300 30 15 124.95 = 30 := by
  sorry

end NUMINAMATH_CALUDE_second_week_cut_percentage_sculpture_problem_l2030_203041


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l2030_203038

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_arithmetic (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_eighth_term
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_sum : sum_arithmetic a 15 = 0) :
  a 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l2030_203038


namespace NUMINAMATH_CALUDE_inequality_solution_implies_k_value_l2030_203006

theorem inequality_solution_implies_k_value (k : ℚ) :
  (∀ x : ℚ, 3 * x - (2 * k - 3) < 4 * x + 3 * k + 6 ↔ x > 1) →
  k = -4/5 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_k_value_l2030_203006


namespace NUMINAMATH_CALUDE_power_function_through_point_l2030_203002

/-- Given a power function that passes through the point (2, 8), prove its equation is x^3 -/
theorem power_function_through_point (n : ℝ) (f : ℝ → ℝ) : 
  (∀ x, f x = x^n) → f 2 = 8 → (∀ x, f x = x^3) := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l2030_203002


namespace NUMINAMATH_CALUDE_trapezoid_area_l2030_203036

/-- Trapezoid ABCD with given properties -/
structure Trapezoid where
  -- Length of AD
  ad : ℝ
  -- Length of BC
  bc : ℝ
  -- Length of CD
  cd : ℝ
  -- BC is parallel to AD
  parallel : True
  -- Ratio of BC to AD is 5:7
  ratio_bc_ad : bc / ad = 5 / 7
  -- AF:FD = 4:3
  ratio_af_fd : (4 / 7 * ad) / (3 / 7 * ad) = 4 / 3
  -- CE:ED = 2:3
  ratio_ce_ed : (2 / 5 * cd) / (3 / 5 * cd) = 2 / 3
  -- Area of ABEF is 123
  area_abef : (ad * cd - (3 / 7 * ad) * (3 / 5 * cd) - bc * (2 / 5 * cd)) / 2 = 123

/-- The area of trapezoid ABCD is 180 -/
theorem trapezoid_area (t : Trapezoid) : (t.ad + t.bc) * t.cd / 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l2030_203036


namespace NUMINAMATH_CALUDE_coordinate_plane_points_theorem_l2030_203088

theorem coordinate_plane_points_theorem (x y : ℝ) :
  (x^2 * y + y^3 = 2 * x^2 + 2 * y^2 → ((x = 0 ∧ y = 0) ∨ y = 2)) ∧
  (x * y + 1 = x + y → (x = 1 ∨ y = 1)) := by
  sorry

end NUMINAMATH_CALUDE_coordinate_plane_points_theorem_l2030_203088


namespace NUMINAMATH_CALUDE_right_triangle_area_l2030_203076

theorem right_triangle_area (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a ≤ b) (h5 : b < c) (h6 : a + b = 13) (h7 : a = 5) (h8 : c^2 = a^2 + b^2) :
  (1/2) * a * b = 20 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2030_203076


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2030_203042

theorem quadratic_roots_property (m n : ℝ) : 
  (m^2 - 2*m - 2025 = 0) → 
  (n^2 - 2*n - 2025 = 0) → 
  (m^2 - 3*m - n = 2023) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2030_203042


namespace NUMINAMATH_CALUDE_lcm_36_45_l2030_203010

theorem lcm_36_45 : Nat.lcm 36 45 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_45_l2030_203010


namespace NUMINAMATH_CALUDE_abs_z_minus_i_equals_sqrt2_over_2_l2030_203085

-- Define the complex number i
def i : ℂ := Complex.I

-- Define z based on the given condition
def z : ℂ := by
  sorry

-- Theorem statement
theorem abs_z_minus_i_equals_sqrt2_over_2 :
  Complex.abs (z - i) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_minus_i_equals_sqrt2_over_2_l2030_203085


namespace NUMINAMATH_CALUDE_series_sum_theorem_l2030_203097

/-- The sum of the infinite series (2n+1)x^n from n=0 to infinity -/
noncomputable def S (x : ℝ) : ℝ := ∑' n, (2 * n + 1) * x^n

/-- Theorem stating that if S(x) = 16, then x = (4 - √2) / 4 -/
theorem series_sum_theorem (x : ℝ) (hx : S x = 16) : x = (4 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_theorem_l2030_203097


namespace NUMINAMATH_CALUDE_incoming_scholars_count_l2030_203078

theorem incoming_scholars_count :
  ∃! n : ℕ, n < 600 ∧ n % 15 = 14 ∧ n % 19 = 13 ∧ n = 509 := by
  sorry

end NUMINAMATH_CALUDE_incoming_scholars_count_l2030_203078


namespace NUMINAMATH_CALUDE_exclusive_or_implications_l2030_203070

theorem exclusive_or_implications (p q : Prop) 
  (h_or : p ∨ q) (h_not_and : ¬(p ∧ q)) : 
  (q ↔ ¬p) ∧ (p ↔ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_exclusive_or_implications_l2030_203070


namespace NUMINAMATH_CALUDE_koolaid_water_increase_factor_l2030_203086

/-- Proves that the water increase factor is 4 given the initial conditions and final percentage --/
theorem koolaid_water_increase_factor : 
  ∀ (initial_koolaid initial_water evaporated_water : ℚ)
    (final_percentage : ℚ),
  initial_koolaid = 2 →
  initial_water = 16 →
  evaporated_water = 4 →
  final_percentage = 4/100 →
  ∃ (increase_factor : ℚ),
    increase_factor = 4 ∧
    initial_koolaid / (initial_koolaid + (initial_water - evaporated_water) * increase_factor) = final_percentage :=
by
  sorry

end NUMINAMATH_CALUDE_koolaid_water_increase_factor_l2030_203086


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2030_203030

def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | x^2 - x - 2 = 0}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2030_203030


namespace NUMINAMATH_CALUDE_marble_product_l2030_203060

theorem marble_product (red blue : ℕ) : 
  (red - blue = 12) →
  (red + blue = red - blue + 40) →
  red * blue = 640 := by
sorry

end NUMINAMATH_CALUDE_marble_product_l2030_203060


namespace NUMINAMATH_CALUDE_sum_of_fractions_simplification_l2030_203066

theorem sum_of_fractions_simplification (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h_sum : a + b + c + d = 0) :
  (1 / (b^2 + c^2 + d^2 - a^2)) + 
  (1 / (a^2 + c^2 + d^2 - b^2)) + 
  (1 / (a^2 + b^2 + d^2 - c^2)) + 
  (1 / (a^2 + b^2 + c^2 - d^2)) = 4 / d^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_simplification_l2030_203066


namespace NUMINAMATH_CALUDE_woman_work_days_value_l2030_203011

/-- The number of days it takes for a woman to complete the work -/
def woman_work_days (total_members : ℕ) (man_work_days : ℕ) (combined_work_days : ℕ) (num_women : ℕ) : ℚ :=
  let num_men := total_members - num_women
  let man_work_rate := 1 / man_work_days
  let total_man_work := (combined_work_days / 2 : ℚ) * man_work_rate * num_men
  let total_woman_work := 1 - total_man_work
  let woman_work_rate := (total_woman_work * 3) / (combined_work_days * num_women)
  1 / woman_work_rate

/-- Theorem stating the number of days it takes for a woman to complete the work -/
theorem woman_work_days_value :
  woman_work_days 15 120 17 3 = 5100 / 83 :=
by sorry

end NUMINAMATH_CALUDE_woman_work_days_value_l2030_203011


namespace NUMINAMATH_CALUDE_non_zero_coeffs_bound_l2030_203009

/-- A polynomial is non-zero if it has at least one non-zero coefficient -/
def NonZeroPoly (p : Polynomial ℝ) : Prop :=
  ∃ (i : ℕ), p.coeff i ≠ 0

/-- The number of non-zero coefficients in a polynomial -/
def NumNonZeroCoeffs (p : Polynomial ℝ) : ℕ :=
  (p.support).card

/-- The statement to be proved -/
theorem non_zero_coeffs_bound (Q : Polynomial ℝ) (n : ℕ) 
  (hQ : NonZeroPoly Q) (hn : n > 0) : 
  NumNonZeroCoeffs ((X - 1)^n * Q) ≥ n + 1 :=
sorry

end NUMINAMATH_CALUDE_non_zero_coeffs_bound_l2030_203009


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2030_203067

theorem trigonometric_simplification (α : ℝ) :
  Real.sin ((5 / 2) * Real.pi + 4 * α) - 
  Real.sin ((5 / 2) * Real.pi + 2 * α) ^ 6 + 
  Real.cos ((7 / 2) * Real.pi - 2 * α) ^ 6 = 
  (1 / 8) * Real.sin (8 * α) * Real.sin (4 * α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2030_203067


namespace NUMINAMATH_CALUDE_area_conversions_l2030_203071

-- Define conversion rates
def sq_dm_to_sq_cm : ℝ := 100
def hectare_to_sq_m : ℝ := 10000
def sq_km_to_hectare : ℝ := 100
def sq_m_to_sq_dm : ℝ := 100

-- Theorem to prove the conversions
theorem area_conversions :
  (7 * sq_dm_to_sq_cm = 700) ∧
  (5 * hectare_to_sq_m = 50000) ∧
  (600 / sq_km_to_hectare = 6) ∧
  (200 / sq_m_to_sq_dm = 2) :=
by sorry

end NUMINAMATH_CALUDE_area_conversions_l2030_203071


namespace NUMINAMATH_CALUDE_reciprocal_of_three_halves_l2030_203079

-- Define the concept of reciprocal
def is_reciprocal (a b : ℚ) : Prop := a * b = 1

-- State the theorem
theorem reciprocal_of_three_halves : 
  is_reciprocal (3/2 : ℚ) (2/3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_three_halves_l2030_203079


namespace NUMINAMATH_CALUDE_vacation_tents_l2030_203051

/-- Given a total number of people, house capacity, and tent capacity, 
    calculate the minimum number of tents needed. -/
def tents_needed (total_people : ℕ) (house_capacity : ℕ) (tent_capacity : ℕ) : ℕ :=
  ((total_people - house_capacity + tent_capacity - 1) / tent_capacity)

/-- Theorem stating that for 13 people, a house capacity of 4, and tents that sleep 2 each, 
    the minimum number of tents needed is 5. -/
theorem vacation_tents : tents_needed 13 4 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_vacation_tents_l2030_203051


namespace NUMINAMATH_CALUDE_gold_coins_count_verify_conditions_l2030_203014

/-- The number of gold coins -/
def n : ℕ := 109

/-- The number of treasure chests -/
def c : ℕ := 13

/-- Theorem stating that the number of gold coins is 109 -/
theorem gold_coins_count : n = 109 :=
  by
  -- Condition 1: When putting 12 gold coins in each chest, 4 chests were left empty
  have h1 : n = 12 * (c - 4) := by sorry
  
  -- Condition 2: When putting 8 gold coins in each chest, 5 gold coins were left over
  have h2 : n = 8 * c + 5 := by sorry
  
  -- Prove that n equals 109
  sorry

/-- Theorem verifying the conditions -/
theorem verify_conditions :
  (n = 12 * (c - 4)) ∧ (n = 8 * c + 5) :=
  by sorry

end NUMINAMATH_CALUDE_gold_coins_count_verify_conditions_l2030_203014


namespace NUMINAMATH_CALUDE_value_of_a_l2030_203091

theorem value_of_a (a : ℝ) (S : Set ℝ) : 
  S = {x : ℝ | 3 * x + a = 0} → 
  (1 : ℝ) ∈ S → 
  a = -3 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l2030_203091


namespace NUMINAMATH_CALUDE_exists_quadratic_with_2n_roots_l2030_203094

/-- Definition of function iteration -/
def iterate (f : ℝ → ℝ) : ℕ → (ℝ → ℝ)
  | 0 => id
  | n + 1 => f ∘ (iterate f n)

/-- A quadratic polynomial -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem stating the existence of a quadratic polynomial with the desired property -/
theorem exists_quadratic_with_2n_roots :
  ∃ (a b c : ℝ), ∀ (n : ℕ), n > 0 →
    (∃ (roots : Finset ℝ), roots.card = 2^n ∧
      (∀ x : ℝ, x ∈ roots ↔ iterate (quadratic a b c) n x = 0) ∧
      (∀ x y : ℝ, x ∈ roots → y ∈ roots → x ≠ y → x ≠ y)) :=
sorry

end NUMINAMATH_CALUDE_exists_quadratic_with_2n_roots_l2030_203094


namespace NUMINAMATH_CALUDE_equation_solution_l2030_203059

theorem equation_solution :
  ∃! x : ℚ, x ≠ -3 ∧ (x^2 + 4*x + 5) / (x + 3) = x + 6 :=
by
  use -13/5
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2030_203059


namespace NUMINAMATH_CALUDE_shipping_cost_invariant_l2030_203098

/-- Represents a settlement with its distance from the city and required goods weight -/
structure Settlement where
  distance : ℝ
  weight : ℝ
  distance_eq_weight : distance = weight

/-- Calculates the shipping cost for a given delivery order -/
def shipping_cost (settlements : List Settlement) : ℝ :=
  settlements.enum.foldl
    (fun acc (i, s) =>
      acc + s.weight * (settlements.take i).foldl (fun sum t => sum + t.distance) 0)
    0

/-- Theorem stating that the shipping cost is invariant under different delivery orders -/
theorem shipping_cost_invariant (settlements : List Settlement) :
  ∀ (perm : List Settlement), settlements.Perm perm →
    shipping_cost settlements = shipping_cost perm :=
  sorry

end NUMINAMATH_CALUDE_shipping_cost_invariant_l2030_203098


namespace NUMINAMATH_CALUDE_value_of_expression_l2030_203049

theorem value_of_expression : 6 * 2017 - 2017 * 4 = 4034 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l2030_203049


namespace NUMINAMATH_CALUDE_min_value_of_f_l2030_203020

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem that the minimum value of f(x) is -2
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2030_203020


namespace NUMINAMATH_CALUDE_area_AEC_is_18_l2030_203064

-- Define the lengths BE and EC
def BE : ℝ := 3
def EC : ℝ := 2

-- Define the area of triangle ABE
def area_ABE : ℝ := 27

-- Theorem statement
theorem area_AEC_is_18 :
  let ratio := BE / EC
  let area_AEC := (EC / BE) * area_ABE
  area_AEC = 18 := by sorry

end NUMINAMATH_CALUDE_area_AEC_is_18_l2030_203064


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2030_203034

theorem solution_set_equivalence (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2030_203034


namespace NUMINAMATH_CALUDE_min_value_expression_l2030_203012

theorem min_value_expression (x y : ℝ) :
  x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 ∧
  ∃ (a b : ℝ), a^2 + b^2 - 8*a + 6*b + 25 = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2030_203012


namespace NUMINAMATH_CALUDE_quadrupled_base_exponent_l2030_203028

theorem quadrupled_base_exponent (c d y : ℝ) (hc : c > 0) (hd : d > 0) (hy : y > 0) :
  (4 * c)^(4 * d) = (c^d * y^d)^2 → y = 16 * c := by
  sorry

end NUMINAMATH_CALUDE_quadrupled_base_exponent_l2030_203028
