import Mathlib

namespace NUMINAMATH_CALUDE_unique_prime_solution_l2260_226066

theorem unique_prime_solution :
  ∃! (p q r : ℕ), 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    3 * p^4 - 5 * q^4 - 4 * r^2 = 26 ∧
    p = 5 ∧ q = 3 ∧ r = 19 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l2260_226066


namespace NUMINAMATH_CALUDE_triangle_angle_tangent_l2260_226074

theorem triangle_angle_tangent (A : Real) :
  (Real.sqrt 3 * Real.cos A + Real.sin A) / (Real.sqrt 3 * Real.sin A - Real.cos A) = Real.tan (-7 * π / 12) →
  Real.tan A = 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_tangent_l2260_226074


namespace NUMINAMATH_CALUDE_min_draws_for_red_specific_l2260_226050

/-- Given a bag with red, white, and black balls, we define the minimum number of draws
    required to guarantee drawing a red ball. -/
def min_draws_for_red (red white black : ℕ) : ℕ :=
  white + black + 1

/-- Theorem stating that for a bag with 10 red, 8 white, and 7 black balls,
    the minimum number of draws to guarantee a red ball is 16. -/
theorem min_draws_for_red_specific : min_draws_for_red 10 8 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_draws_for_red_specific_l2260_226050


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_l2260_226037

/-- The measure of each interior angle in a regular octagon is 135 degrees. -/
theorem regular_octagon_interior_angle : ℝ := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_l2260_226037


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l2260_226089

theorem greatest_integer_inequality (y : ℤ) : 
  (5 : ℚ) / 11 > (y : ℚ) / 17 ↔ y ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l2260_226089


namespace NUMINAMATH_CALUDE_book_arrangement_count_book_arrangement_proof_l2260_226081

theorem book_arrangement_count : Nat :=
  let num_dictionaries : Nat := 3
  let num_novels : Nat := 2
  let dict_arrangements : Nat := Nat.factorial num_dictionaries
  let novel_arrangements : Nat := Nat.factorial num_novels
  let group_arrangements : Nat := Nat.factorial 2
  dict_arrangements * novel_arrangements * group_arrangements

theorem book_arrangement_proof :
  book_arrangement_count = 24 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_book_arrangement_proof_l2260_226081


namespace NUMINAMATH_CALUDE_birthday_cards_l2260_226031

theorem birthday_cards (initial_cards total_cards : ℕ) 
  (h1 : initial_cards = 64)
  (h2 : total_cards = 82) :
  total_cards - initial_cards = 18 := by
  sorry

end NUMINAMATH_CALUDE_birthday_cards_l2260_226031


namespace NUMINAMATH_CALUDE_ellipse_properties_l2260_226021

-- Define the ellipse C
def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the line l
def line (x y m : ℝ) : Prop :=
  y = x + m

-- Define the theorem
theorem ellipse_properties
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0)
  (h_triangle : ∃ A F₁ F₂ : ℝ × ℝ, 
    ellipse A.1 A.2 a b ∧
    ellipse F₁.1 F₁.2 a b ∧
    ellipse F₂.1 F₂.2 a b ∧
    (A.2 - F₁.2)^2 + (A.1 - F₁.1)^2 = (A.2 - F₂.2)^2 + (A.1 - F₂.1)^2 ∧
    (F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2 = 8) :
  (∀ x y, ellipse x y 2 (Real.sqrt 2)) ∧
  (∀ P Q : ℝ × ℝ, 
    ellipse P.1 P.2 2 (Real.sqrt 2) ∧
    ellipse Q.1 Q.2 2 (Real.sqrt 2) ∧
    line P.1 P.2 1 ∧
    line Q.1 Q.2 1 →
    (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 80/9) ∧
  (∀ m : ℝ, 
    (∃ P Q : ℝ × ℝ,
      ellipse P.1 P.2 2 (Real.sqrt 2) ∧
      ellipse Q.1 Q.2 2 (Real.sqrt 2) ∧
      line P.1 P.2 m ∧
      line Q.1 Q.2 m ∧
      P.1 * Q.2 - P.2 * Q.1 = 8/3) ↔
    (m = 2 ∨ m = -2 ∨ m = Real.sqrt 2 ∨ m = -Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2260_226021


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2260_226058

theorem necessary_not_sufficient_condition : 
  (∀ x : ℝ, (abs (x + 1) ≤ 4) → (-6 ≤ x ∧ x ≤ 3)) ∧
  (∃ x : ℝ, (-6 ≤ x ∧ x ≤ 3) ∧ ¬(abs (x + 1) ≤ 4)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2260_226058


namespace NUMINAMATH_CALUDE_expression_equals_negative_one_l2260_226000

theorem expression_equals_negative_one (b y : ℝ) (hb : b ≠ 0) (hy1 : y ≠ b) (hy2 : y ≠ -b) :
  (((b / (b + y)) + (y / (b - y))) / ((y / (b + y)) - (b / (b - y)))) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_negative_one_l2260_226000


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2260_226090

theorem fraction_evaluation (a b : ℝ) (ha : a = 7) (hb : b = 3) :
  5 / (a + b) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2260_226090


namespace NUMINAMATH_CALUDE_prob_non_matching_is_five_sixths_l2260_226029

/-- Represents the possible colors for shorts -/
inductive ShortsColor
| Black
| Gold
| Blue

/-- Represents the possible colors for jerseys -/
inductive JerseyColor
| White
| Gold

/-- The total number of possible color combinations -/
def total_combinations : ℕ := 6

/-- The number of non-matching color combinations -/
def non_matching_combinations : ℕ := 5

/-- Probability of selecting a non-matching color combination -/
def prob_non_matching : ℚ := non_matching_combinations / total_combinations

theorem prob_non_matching_is_five_sixths :
  prob_non_matching = 5 / 6 := by sorry

end NUMINAMATH_CALUDE_prob_non_matching_is_five_sixths_l2260_226029


namespace NUMINAMATH_CALUDE_triangle_dot_product_l2260_226001

/-- Given a triangle ABC with area √3 and angle A = π/3, 
    the dot product of vectors AB and AC is equal to 2. -/
theorem triangle_dot_product (A B C : ℝ × ℝ) : 
  let S := Real.sqrt 3
  let angleA := π / 3
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  let area := Real.sqrt 3
  area = Real.sqrt 3 ∧ 
  angleA = π / 3 →
  AB.1 * AC.1 + AB.2 * AC.2 = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_dot_product_l2260_226001


namespace NUMINAMATH_CALUDE_trash_can_problem_l2260_226005

theorem trash_can_problem (x y a b : ℝ) : 
  3 * x + 4 * y = 580 →
  6 * x + 5 * y = 860 →
  a + b = 200 →
  60 * a + 100 * b ≤ 15000 →
  (x = 60 ∧ y = 100) ∧ a ≥ 125 := by sorry

end NUMINAMATH_CALUDE_trash_can_problem_l2260_226005


namespace NUMINAMATH_CALUDE_largest_x_value_l2260_226012

theorem largest_x_value (x y : ℤ) : 
  (1/4 : ℚ) < (x : ℚ)/7 ∧ (x : ℚ)/7 < 2/3 ∧ x + y = 10 →
  x ≤ 4 ∧ (∃ (z : ℤ), (1/4 : ℚ) < (z : ℚ)/7 ∧ (z : ℚ)/7 < 2/3 ∧ z + (10 - z) = 10 ∧ z = 4) :=
by sorry

end NUMINAMATH_CALUDE_largest_x_value_l2260_226012


namespace NUMINAMATH_CALUDE_units_digit_of_2143_power_752_l2260_226009

theorem units_digit_of_2143_power_752 : (2143^752) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_2143_power_752_l2260_226009


namespace NUMINAMATH_CALUDE_eliza_ironed_17_pieces_l2260_226060

/-- Calculates the total number of clothes Eliza ironed given the time spent on blouses and dresses --/
def total_clothes_ironed (blouse_time : ℕ) (dress_time : ℕ) (blouse_hours : ℕ) (dress_hours : ℕ) : ℕ :=
  let blouses := (blouse_hours * 60) / blouse_time
  let dresses := (dress_hours * 60) / dress_time
  blouses + dresses

/-- Theorem stating that Eliza ironed 17 pieces of clothes --/
theorem eliza_ironed_17_pieces :
  total_clothes_ironed 15 20 2 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_eliza_ironed_17_pieces_l2260_226060


namespace NUMINAMATH_CALUDE_exponent_rule_problem_solution_l2260_226024

theorem exponent_rule (a : ℕ) (m n : ℕ) : a^m * a^n = a^(m + n) := by sorry

theorem problem_solution : 3000 * (3000^2999) = 3000^3000 := by
  have h1 : 3000 * (3000^2999) = 3000^1 * 3000^2999 := by sorry
  have h2 : 3000^1 * 3000^2999 = 3000^(1 + 2999) := by sorry
  have h3 : 1 + 2999 = 3000 := by sorry
  sorry

end NUMINAMATH_CALUDE_exponent_rule_problem_solution_l2260_226024


namespace NUMINAMATH_CALUDE_equation_solution_l2260_226044

theorem equation_solution : ∃ (x : ℝ), x^2 - 2*x - 8 = -(x + 2)*(x - 6) ↔ x = 5 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2260_226044


namespace NUMINAMATH_CALUDE_reactor_rearrangements_count_l2260_226062

/-- The number of distinguishable rearrangements of REACTOR with vowels at the end -/
def rearrangements_reactor : ℕ :=
  let consonants := 4  -- R, C, T, R
  let vowels := 3      -- E, A, O
  let consonant_arrangements := Nat.factorial consonants / Nat.factorial 2  -- 4! / 2! due to repeated R
  let vowel_arrangements := Nat.factorial vowels
  consonant_arrangements * vowel_arrangements

/-- Theorem stating that the number of rearrangements is 72 -/
theorem reactor_rearrangements_count :
  rearrangements_reactor = 72 := by
  sorry

#eval rearrangements_reactor  -- Should output 72

end NUMINAMATH_CALUDE_reactor_rearrangements_count_l2260_226062


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_range_l2260_226025

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*x + a ≥ 0}

theorem intersection_nonempty_implies_a_range :
  (∃ a : ℝ, (A ∩ B a).Nonempty) ↔ {a : ℝ | a > -8} = Set.Ioi (-8) := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_range_l2260_226025


namespace NUMINAMATH_CALUDE_unique_complementary_digit_l2260_226018

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem unique_complementary_digit (N : ℕ) : 
  ∃! d : ℕ, 0 < d ∧ d < 9 ∧ (sum_of_digits N + d) % 9 = 0 := by sorry

end NUMINAMATH_CALUDE_unique_complementary_digit_l2260_226018


namespace NUMINAMATH_CALUDE_odd_numbers_between_300_and_700_l2260_226082

def count_odd_numbers (lower upper : ℕ) : ℕ :=
  (upper - lower - 1 + (lower % 2)) / 2

theorem odd_numbers_between_300_and_700 :
  count_odd_numbers 300 700 = 200 := by
  sorry

end NUMINAMATH_CALUDE_odd_numbers_between_300_and_700_l2260_226082


namespace NUMINAMATH_CALUDE_unique_solution_ceiling_equation_l2260_226047

theorem unique_solution_ceiling_equation :
  ∃! x : ℝ, x > 0 ∧ x + ⌈x⌉ = 21.3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_ceiling_equation_l2260_226047


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2260_226016

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (-2 + 3 * Complex.I) / Complex.I
  (z.re > 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2260_226016


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2260_226011

open Real

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k : ℝ)^2 / 3^k

theorem infinite_series_sum : series_sum = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2260_226011


namespace NUMINAMATH_CALUDE_stone_growth_prevention_l2260_226039

/-- The amount of stone consumed by one warrior per day -/
def warrior_consumption : ℝ := 1

/-- The number of days it takes for the stone to pierce the sky with 14 warriors -/
def days_with_14 : ℕ := 16

/-- The number of days it takes for the stone to pierce the sky with 15 warriors -/
def days_with_15 : ℕ := 24

/-- The daily growth rate of the stone -/
def stone_growth_rate : ℝ := 17 * warrior_consumption

/-- The minimum number of warriors needed to prevent the stone from piercing the sky -/
def min_warriors : ℕ := 17

theorem stone_growth_prevention :
  (↑min_warriors * warrior_consumption = stone_growth_rate) ∧
  (∀ n : ℕ, n < min_warriors → ↑n * warrior_consumption < stone_growth_rate) := by
  sorry

#check stone_growth_prevention

end NUMINAMATH_CALUDE_stone_growth_prevention_l2260_226039


namespace NUMINAMATH_CALUDE_optimal_apps_l2260_226079

/-- The maximum number of apps Roger can have on his phone for optimal function -/
def max_apps : ℕ := 50

/-- The recommended number of apps -/
def recommended_apps : ℕ := 35

/-- The number of apps Roger currently has -/
def rogers_current_apps : ℕ := 2 * recommended_apps

/-- The number of apps Roger needs to delete -/
def apps_to_delete : ℕ := 20

/-- Theorem stating the maximum number of apps Roger can have for optimal function -/
theorem optimal_apps : max_apps = rogers_current_apps - apps_to_delete := by
  sorry

end NUMINAMATH_CALUDE_optimal_apps_l2260_226079


namespace NUMINAMATH_CALUDE_three_intersection_points_l2260_226046

-- Define the four lines
def line1 (x y : ℝ) : Prop := 3 * y - 2 * x = 1
def line2 (x y : ℝ) : Prop := x + 2 * y = 2
def line3 (x y : ℝ) : Prop := 4 * x - 6 * y = 5
def line4 (x y : ℝ) : Prop := 2 * x - 3 * y = 4

-- Define a function to check if a point lies on a line
def point_on_line (x y : ℝ) (line : ℝ → ℝ → Prop) : Prop := line x y

-- Define a function to check if a point is an intersection of at least two lines
def is_intersection (x y : ℝ) : Prop :=
  (point_on_line x y line1 ∧ point_on_line x y line2) ∨
  (point_on_line x y line1 ∧ point_on_line x y line3) ∨
  (point_on_line x y line1 ∧ point_on_line x y line4) ∨
  (point_on_line x y line2 ∧ point_on_line x y line3) ∨
  (point_on_line x y line2 ∧ point_on_line x y line4) ∨
  (point_on_line x y line3 ∧ point_on_line x y line4)

-- Theorem stating that there are exactly 3 distinct intersection points
theorem three_intersection_points :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    is_intersection p1.1 p1.2 ∧
    is_intersection p2.1 p2.2 ∧
    is_intersection p3.1 p3.2 ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    ∀ (x y : ℝ), is_intersection x y → (x, y) = p1 ∨ (x, y) = p2 ∨ (x, y) = p3 :=
by sorry


end NUMINAMATH_CALUDE_three_intersection_points_l2260_226046


namespace NUMINAMATH_CALUDE_wage_increase_calculation_l2260_226088

theorem wage_increase_calculation (W H W' H' : ℝ) : 
  W > 0 → H > 0 → W' > W → -- Initial conditions
  H' = H * (1 - 0.20) → -- 20% reduction in hours
  W * H = W' * H' → -- Total weekly income remains the same
  (W' - W) / W = 0.25 := by -- The wage increase is 25%
  sorry

end NUMINAMATH_CALUDE_wage_increase_calculation_l2260_226088


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2260_226014

theorem sqrt_equation_solution (t : ℝ) : 
  (Real.sqrt (49 - (t - 3)^2) - 7 = 0) ↔ (t = 3) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2260_226014


namespace NUMINAMATH_CALUDE_boys_age_problem_l2260_226086

theorem boys_age_problem (x : ℕ) : x + 4 = 2 * (x - 6) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_boys_age_problem_l2260_226086


namespace NUMINAMATH_CALUDE_min_total_faces_l2260_226003

/-- Represents a fair die with a given number of faces. -/
structure Die where
  faces : ℕ
  faces_gt_6 : faces > 6

/-- Calculates the number of ways to roll a given sum with two dice. -/
def waysToRoll (d1 d2 : Die) (sum : ℕ) : ℕ :=
  sorry

/-- The probability of rolling a given sum with two dice. -/
def probOfSum (d1 d2 : Die) (sum : ℕ) : ℚ :=
  sorry

theorem min_total_faces (d1 d2 : Die) :
  (probOfSum d1 d2 8 = (1 : ℚ) / 2 * probOfSum d1 d2 11) →
  (probOfSum d1 d2 15 = (1 : ℚ) / 30) →
  d1.faces + d2.faces ≥ 18 :=
sorry

end NUMINAMATH_CALUDE_min_total_faces_l2260_226003


namespace NUMINAMATH_CALUDE_f_difference_l2260_226035

/-- The function f(x) = 3x^3 + 2x^2 - 4x - 1 -/
def f (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 - 4 * x - 1

/-- Theorem stating that f(x + h) - f(x) = h(9x^2 + 9xh + 3h^2 + 4x + 2h - 4) for all real x and h -/
theorem f_difference (x h : ℝ) : f (x + h) - f x = h * (9 * x^2 + 9 * x * h + 3 * h^2 + 4 * x + 2 * h - 4) := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l2260_226035


namespace NUMINAMATH_CALUDE_eight_percent_difference_l2260_226095

theorem eight_percent_difference (x y : ℝ) 
  (hx : 8 = 0.25 * x) 
  (hy : 8 = 0.5 * y) : 
  x - y = 16 := by
  sorry

end NUMINAMATH_CALUDE_eight_percent_difference_l2260_226095


namespace NUMINAMATH_CALUDE_salary_comparison_l2260_226061

/-- Proves that given the salaries of A, B, and C are in the ratio of 1 : 2 : 3, 
    and the sum of B and C's salaries is 6000, 
    the percentage by which C's salary exceeds A's salary is 200%. -/
theorem salary_comparison (sa sb sc : ℝ) : 
  sa > 0 → sb > 0 → sc > 0 → 
  sb / sa = 2 → sc / sa = 3 → 
  sb + sc = 6000 → 
  (sc - sa) / sa * 100 = 200 := by
sorry

end NUMINAMATH_CALUDE_salary_comparison_l2260_226061


namespace NUMINAMATH_CALUDE_smallest_upper_bound_l2260_226077

theorem smallest_upper_bound (x : ℤ) 
  (h1 : 0 < x ∧ x < 7)
  (h2 : -1 < x ∧ x < 5)
  (h3 : 0 < x ∧ x < 3)
  (h4 : x + 2 < 4) :
  ∀ y : ℤ, (0 < x ∧ x < y) → y ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_upper_bound_l2260_226077


namespace NUMINAMATH_CALUDE_reciprocal_comparison_reciprocal_comparison_with_condition_l2260_226051

theorem reciprocal_comparison :
  (-3/2 : ℚ) < (-2/3 : ℚ) ∧
  (2/3 : ℚ) < (3/2 : ℚ) ∧
  ¬((-1 : ℚ) < (-1 : ℚ)) ∧
  ¬((1 : ℚ) < (1 : ℚ)) ∧
  ¬((3 : ℚ) < (1/3 : ℚ)) :=
by
  sorry

-- Helper definition for the condition
def less_than_reciprocal (x : ℚ) : Prop :=
  x ≠ 0 ∧ x < 1 ∧ x < 1 / x

-- Theorem using the helper definition
theorem reciprocal_comparison_with_condition :
  less_than_reciprocal (-3/2) ∧
  less_than_reciprocal (2/3) ∧
  ¬less_than_reciprocal (-1) ∧
  ¬less_than_reciprocal 1 ∧
  ¬less_than_reciprocal 3 :=
by
  sorry

end NUMINAMATH_CALUDE_reciprocal_comparison_reciprocal_comparison_with_condition_l2260_226051


namespace NUMINAMATH_CALUDE_cookie_bags_problem_l2260_226019

/-- Given a total number of cookies and the number of cookies per bag,
    calculate the number of bags. -/
def number_of_bags (total_cookies : ℕ) (cookies_per_bag : ℕ) : ℕ :=
  total_cookies / cookies_per_bag

theorem cookie_bags_problem :
  let total_cookies : ℕ := 14
  let cookies_per_bag : ℕ := 2
  number_of_bags total_cookies cookies_per_bag = 7 := by
  sorry


end NUMINAMATH_CALUDE_cookie_bags_problem_l2260_226019


namespace NUMINAMATH_CALUDE_our_ellipse_equation_l2260_226043

-- Define the ellipse
structure Ellipse where
  f1 : ℝ × ℝ  -- Focus 1
  f2 : ℝ × ℝ  -- Focus 2
  min_dist : ℝ -- Shortest distance from a point on the ellipse to F₁

-- Define our specific ellipse
def our_ellipse : Ellipse :=
  { f1 := (0, -4)
  , f2 := (0, 4)
  , min_dist := 2
  }

-- Define the equation of an ellipse
def is_ellipse_equation (e : Ellipse) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq x y ↔ (x, y) ∈ {p : ℝ × ℝ | dist p e.f1 + dist p e.f2 = 2 * (e.f2.1 - e.f1.1)}

-- Theorem statement
theorem our_ellipse_equation :
  is_ellipse_equation our_ellipse (fun x y => x^2/20 + y^2/36 = 1) :=
sorry

end NUMINAMATH_CALUDE_our_ellipse_equation_l2260_226043


namespace NUMINAMATH_CALUDE_intersection_A_B_l2260_226040

def A : Set ℕ := {0}
def B : Set ℕ := {0, 1, 2}

theorem intersection_A_B : A ∩ B = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2260_226040


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2260_226007

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2260_226007


namespace NUMINAMATH_CALUDE_rectangle_side_problem_l2260_226075

theorem rectangle_side_problem (side1 : ℝ) (side2 : ℕ) (unknown_side : ℝ) : 
  side1 = 5 →
  side2 = 12 →
  (side1 * side2 = side1 * unknown_side + 25 ∨ side1 * unknown_side = side1 * side2 + 25) →
  unknown_side = 7 ∨ unknown_side = 17 := by
sorry

end NUMINAMATH_CALUDE_rectangle_side_problem_l2260_226075


namespace NUMINAMATH_CALUDE_susan_ate_six_candies_l2260_226002

/-- The number of candies Susan bought on Tuesday -/
def tuesday_candies : ℕ := 3

/-- The number of candies Susan bought on Thursday -/
def thursday_candies : ℕ := 5

/-- The number of candies Susan bought on Friday -/
def friday_candies : ℕ := 2

/-- The number of candies Susan has left -/
def candies_left : ℕ := 4

/-- The total number of candies Susan bought -/
def total_candies : ℕ := tuesday_candies + thursday_candies + friday_candies

/-- The number of candies Susan ate -/
def candies_eaten : ℕ := total_candies - candies_left

theorem susan_ate_six_candies : candies_eaten = 6 := by
  sorry

end NUMINAMATH_CALUDE_susan_ate_six_candies_l2260_226002


namespace NUMINAMATH_CALUDE_ticket_price_possibilities_l2260_226006

theorem ticket_price_possibilities : ∃ (n : ℕ), n = (Nat.divisors 90 ∩ Nat.divisors 150).card ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ticket_price_possibilities_l2260_226006


namespace NUMINAMATH_CALUDE_system_of_equations_l2260_226063

theorem system_of_equations (x y A : ℝ) : 
  2 * x + y = A → 
  x + 2 * y = 8 → 
  (x + y) / 3 = 1.6666666666666667 → 
  A = 7 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l2260_226063


namespace NUMINAMATH_CALUDE_complement_intersection_equals_singleton_l2260_226056

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2, 4}
def N : Set ℕ := {2, 4, 5}

theorem complement_intersection_equals_singleton :
  (U \ M) ∩ (U \ N) = {3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_singleton_l2260_226056


namespace NUMINAMATH_CALUDE_least_odd_prime_factor_1234_10_plus_1_l2260_226072

theorem least_odd_prime_factor_1234_10_plus_1 : 
  (Nat.minFac (1234^10 + 1)) = 61 := by sorry

end NUMINAMATH_CALUDE_least_odd_prime_factor_1234_10_plus_1_l2260_226072


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2260_226008

theorem smallest_integer_satisfying_inequality :
  ∃ (y : ℤ), (7 - 3 * y ≤ 29) ∧ (∀ (z : ℤ), z < y → 7 - 3 * z > 29) ∧ y = -7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2260_226008


namespace NUMINAMATH_CALUDE_exists_n_for_m_l2260_226041

-- Define the function f(n) as the sum of n and its digits
def f (n : ℕ) : ℕ :=
  n + (Nat.digits 10 n).sum

-- Theorem statement
theorem exists_n_for_m (m : ℕ) :
  m > 0 → ∃ n : ℕ, f n = m ∨ f n = m + 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_for_m_l2260_226041


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l2260_226094

theorem arithmetic_geometric_sequence_ratio
  (a : ℕ → ℝ)  -- arithmetic sequence
  (d : ℝ)      -- common difference
  (h1 : d ≠ 0) -- d is non-zero
  (h2 : ∀ n, a (n + 1) = a n + d)  -- definition of arithmetic sequence
  (h3 : (a 9) ^ 2 = a 5 * a 15)    -- a_5, a_9, a_15 form geometric sequence
  : (a 9) / (a 5) = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l2260_226094


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2260_226004

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 3 = 3 ∧
  a 1 + a 2 = 0

/-- The general term of the sequence -/
def GeneralTerm (n : ℕ) : ℝ := 2 * n - 3

theorem arithmetic_sequence_formula (a : ℕ → ℝ) :
  ArithmeticSequence a → (∀ n : ℕ, a n = GeneralTerm n) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2260_226004


namespace NUMINAMATH_CALUDE_min_toothpicks_to_remove_correct_l2260_226032

/-- Represents a figure made of toothpicks and triangles -/
structure ToothpickFigure where
  total_toothpicks : ℕ
  upward_triangles : ℕ
  downward_triangles : ℕ
  upward_side_length : ℕ
  downward_side_length : ℕ

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (figure : ToothpickFigure) : ℕ :=
  figure.upward_triangles * figure.upward_side_length

/-- Theorem stating the minimum number of toothpicks to remove -/
theorem min_toothpicks_to_remove_correct (figure : ToothpickFigure) 
  (h1 : figure.total_toothpicks = 40)
  (h2 : figure.upward_triangles = 10)
  (h3 : figure.downward_triangles = 8)
  (h4 : figure.upward_side_length = 2)
  (h5 : figure.downward_side_length = 1) :
  min_toothpicks_to_remove figure = 20 := by
  sorry

#eval min_toothpicks_to_remove {
  total_toothpicks := 40,
  upward_triangles := 10,
  downward_triangles := 8,
  upward_side_length := 2,
  downward_side_length := 1
}

end NUMINAMATH_CALUDE_min_toothpicks_to_remove_correct_l2260_226032


namespace NUMINAMATH_CALUDE_min_S_and_max_m_l2260_226042

-- Define the function S
def S (x : ℝ) : ℝ := |x - 2| + |x - 4|

-- State the theorem
theorem min_S_and_max_m :
  (∃ (min_S : ℝ), ∀ x : ℝ, S x ≥ min_S ∧ ∃ x₀ : ℝ, S x₀ = min_S) ∧
  (∃ (max_m : ℝ), (∀ x y : ℝ, S x ≥ max_m * (-y^2 + 2*y)) ∧
    ∀ m : ℝ, (∀ x y : ℝ, S x ≥ m * (-y^2 + 2*y)) → m ≤ max_m) ∧
  (∀ x : ℝ, S x ≥ 2) ∧
  (∀ x y : ℝ, S x ≥ 2 * (-y^2 + 2*y)) :=
by sorry

end NUMINAMATH_CALUDE_min_S_and_max_m_l2260_226042


namespace NUMINAMATH_CALUDE_faster_runner_overtakes_in_five_laps_l2260_226070

/-- The length of the track in meters -/
def track_length : ℝ := 400

/-- The speed ratio of the faster runner to the slower runner -/
def speed_ratio : ℝ := 1.25

/-- The number of laps after which the faster runner overtakes the slower runner -/
def overtake_laps : ℝ := 5

/-- Theorem stating that the faster runner overtakes the slower runner after 5 laps -/
theorem faster_runner_overtakes_in_five_laps :
  ∀ (v : ℝ), v > 0 →
  speed_ratio * v * (overtake_laps * track_length / v) =
  (overtake_laps + 1) * track_length :=
by sorry

end NUMINAMATH_CALUDE_faster_runner_overtakes_in_five_laps_l2260_226070


namespace NUMINAMATH_CALUDE_highest_power_of_seven_in_square_of_factorial_l2260_226034

theorem highest_power_of_seven_in_square_of_factorial (n : ℕ) (h : n = 50) :
  (∃ k : ℕ, (7 : ℕ)^k ∣ (n! : ℕ)^2 ∧ ∀ m : ℕ, (7 : ℕ)^m ∣ (n! : ℕ)^2 → m ≤ k) →
  (∃ k : ℕ, k = 16 ∧ (7 : ℕ)^k ∣ (n! : ℕ)^2 ∧ ∀ m : ℕ, (7 : ℕ)^m ∣ (n! : ℕ)^2 → m ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_highest_power_of_seven_in_square_of_factorial_l2260_226034


namespace NUMINAMATH_CALUDE_oil_barrel_ratio_l2260_226069

theorem oil_barrel_ratio (mass_A mass_B : ℝ) : 
  (mass_A + 10000 : ℝ) / (mass_B + 10000) = 4 / 5 →
  (mass_A + 18000 : ℝ) / (mass_B + 2000) = 8 / 7 →
  mass_A / mass_B = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_oil_barrel_ratio_l2260_226069


namespace NUMINAMATH_CALUDE_girls_in_circle_l2260_226087

/-- The number of girls in a circular arrangement where one girl is
    both the fifth to the left and the eighth to the right of another girl. -/
def num_girls_in_circle : ℕ := 13

/-- Proposition: In a circular arrangement of girls, if one girl is both
    the fifth to the left and the eighth to the right of another girl,
    then the total number of girls in the circle is 13. -/
theorem girls_in_circle :
  ∀ (n : ℕ), n > 0 →
  (∃ (a b : ℕ), a < n ∧ b < n ∧ a ≠ b ∧
   ((a + 5) % n = b) ∧ ((b + 8) % n = a)) →
  n = num_girls_in_circle :=
sorry

end NUMINAMATH_CALUDE_girls_in_circle_l2260_226087


namespace NUMINAMATH_CALUDE_star_five_two_l2260_226054

def star (a b : ℚ) : ℚ := a^2 + a/b

theorem star_five_two : star 5 2 = 55/2 := by
  sorry

end NUMINAMATH_CALUDE_star_five_two_l2260_226054


namespace NUMINAMATH_CALUDE_spread_diluted_ecoli_correct_l2260_226023

/-- Represents different biological experimental procedures -/
inductive ExperimentalProcedure
  | SpreadDilutedEColi
  | IntroduceSterileAir
  | InoculateSoilLeachate
  | UseOpenRoseFlowers

/-- Represents the outcome of an experimental procedure -/
inductive ExperimentOutcome
  | Success
  | Failure

/-- Function that determines the outcome of a given experimental procedure -/
def experimentResult (procedure : ExperimentalProcedure) : ExperimentOutcome :=
  match procedure with
  | ExperimentalProcedure.SpreadDilutedEColi => ExperimentOutcome.Success
  | _ => ExperimentOutcome.Failure

/-- Theorem stating that spreading diluted E. coli culture is the correct method -/
theorem spread_diluted_ecoli_correct :
  ∀ (procedure : ExperimentalProcedure),
    experimentResult procedure = ExperimentOutcome.Success ↔
    procedure = ExperimentalProcedure.SpreadDilutedEColi :=
by
  sorry

#check spread_diluted_ecoli_correct

end NUMINAMATH_CALUDE_spread_diluted_ecoli_correct_l2260_226023


namespace NUMINAMATH_CALUDE_worker_completion_time_l2260_226071

theorem worker_completion_time (worker_b_time worker_ab_time : Real) 
  (hb : worker_b_time = 10)
  (hab : worker_ab_time = 3.333333333333333)
  : ∃ worker_a_time : Real, 
    worker_a_time = 5 ∧ 
    1 / worker_a_time + 1 / worker_b_time = 1 / worker_ab_time :=
by
  sorry

end NUMINAMATH_CALUDE_worker_completion_time_l2260_226071


namespace NUMINAMATH_CALUDE_king_middle_school_teachers_l2260_226015

/-- Represents a school with students and teachers -/
structure School where
  num_students : ℕ
  classes_per_student : ℕ
  classes_per_teacher : ℕ
  students_per_class : ℕ

/-- Calculates the number of teachers in a school -/
def num_teachers (s : School) : ℕ :=
  let total_classes := s.num_students * s.classes_per_student
  let unique_classes := (total_classes + s.students_per_class - 1) / s.students_per_class
  (unique_classes + s.classes_per_teacher - 1) / s.classes_per_teacher

/-- King Middle School -/
def king_middle_school : School :=
  { num_students := 1200
  , classes_per_student := 6
  , classes_per_teacher := 5
  , students_per_class := 35 }

theorem king_middle_school_teachers :
  num_teachers king_middle_school = 42 := by
  sorry

end NUMINAMATH_CALUDE_king_middle_school_teachers_l2260_226015


namespace NUMINAMATH_CALUDE_laser_beam_distance_laser_beam_distance_is_ten_l2260_226045

/-- The total distance traveled by a laser beam with given conditions -/
theorem laser_beam_distance : ℝ :=
  let start : ℝ × ℝ := (2, 3)
  let end_point : ℝ × ℝ := (6, 3)
  let reflected_end : ℝ × ℝ := (-6, -3)
  Real.sqrt ((start.1 - reflected_end.1)^2 + (start.2 - reflected_end.2)^2)

/-- Proof that the laser beam distance is 10 -/
theorem laser_beam_distance_is_ten : laser_beam_distance = 10 := by
  sorry

end NUMINAMATH_CALUDE_laser_beam_distance_laser_beam_distance_is_ten_l2260_226045


namespace NUMINAMATH_CALUDE_rewards_calculation_l2260_226096

def rewards_remaining (total_bowls : ℕ) (total_customers : ℕ) (large_purchase_customers : ℕ) 
  (large_purchase_amount : ℕ) (reward_rate : ℚ) : ℕ :=
  let bowls_purchased := large_purchase_customers * large_purchase_amount
  let reward_bowls := (bowls_purchased / 10) * 2
  total_bowls - reward_bowls

theorem rewards_calculation :
  rewards_remaining 70 20 10 20 (2/10) = 30 := by
  sorry

end NUMINAMATH_CALUDE_rewards_calculation_l2260_226096


namespace NUMINAMATH_CALUDE_moscow_olympiad_1975_l2260_226052

theorem moscow_olympiad_1975 (a b c p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r →
  p = b^c + a →
  q = a^b + c →
  r = c^a + b →
  q = r := by
sorry

end NUMINAMATH_CALUDE_moscow_olympiad_1975_l2260_226052


namespace NUMINAMATH_CALUDE_line_intersection_l2260_226064

theorem line_intersection : 
  let x : ℚ := 27/50
  let y : ℚ := -9/10
  let line1 (x : ℚ) : ℚ := -5/3 * x
  let line2 (x : ℚ) : ℚ := 15*x - 9
  (y = line1 x) ∧ (y = line2 x) :=
by sorry

end NUMINAMATH_CALUDE_line_intersection_l2260_226064


namespace NUMINAMATH_CALUDE_saree_stripe_theorem_l2260_226048

/-- Represents the stripes on a Saree --/
structure SareeStripes where
  brown : ℕ
  gold : ℕ
  blue : ℕ

/-- Represents the properties of the Saree's stripe pattern --/
def SareeProperties (s : SareeStripes) : Prop :=
  s.gold = 3 * s.brown ∧
  s.blue = 5 * s.gold ∧
  s.brown = 4 ∧
  s.brown + s.gold + s.blue = 100

/-- Calculates the number of complete patterns on the Saree --/
def patternCount (s : SareeStripes) : ℕ :=
  (s.brown + s.gold + s.blue) / 3

theorem saree_stripe_theorem (s : SareeStripes) 
  (h : SareeProperties s) : s.blue = 84 ∧ patternCount s = 33 := by
  sorry

#check saree_stripe_theorem

end NUMINAMATH_CALUDE_saree_stripe_theorem_l2260_226048


namespace NUMINAMATH_CALUDE_range_of_ab_l2260_226097

-- Define the polynomial
def P (a b x : ℝ) : ℝ := (x^2 - a*x + 1) * (x^2 - b*x + 1)

-- State the theorem
theorem range_of_ab (a b : ℝ) (q : ℝ) (h_q : q ∈ Set.Icc (1/3) 2) 
  (h_roots : ∃ (r₁ r₂ r₃ r₄ : ℝ), 
    (P a b r₁ = 0 ∧ P a b r₂ = 0 ∧ P a b r₃ = 0 ∧ P a b r₄ = 0) ∧ 
    (∃ (m : ℝ), r₁ = m ∧ r₂ = m*q ∧ r₃ = m*q^2 ∧ r₄ = m*q^3)) :
  a * b ∈ Set.Icc 4 (112/9) :=
sorry

end NUMINAMATH_CALUDE_range_of_ab_l2260_226097


namespace NUMINAMATH_CALUDE_product_of_solutions_abs_y_eq_3_abs_y_minus_2_l2260_226068

theorem product_of_solutions_abs_y_eq_3_abs_y_minus_2 :
  ∃ (y₁ y₂ : ℝ), (|y₁| = 3*(|y₁| - 2)) ∧ (|y₂| = 3*(|y₂| - 2)) ∧ (y₁ ≠ y₂) ∧ (y₁ * y₂ = -9) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_abs_y_eq_3_abs_y_minus_2_l2260_226068


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2260_226028

/-- An arithmetic sequence with first term 13 and fourth term 1 has common difference -4. -/
theorem arithmetic_sequence_common_difference :
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence property
  a 1 = 13 →
  a 4 = 1 →
  a 2 - a 1 = -4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2260_226028


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_train_passes_jogger_in_36_seconds_l2260_226085

/-- Time for a train to pass a jogger given their speeds, train length, and initial distance -/
theorem train_passing_jogger_time (jogger_speed train_speed : Real) 
  (train_length initial_distance : Real) : Real :=
  let jogger_speed_ms := jogger_speed * (1000 / 3600)
  let train_speed_ms := train_speed * (1000 / 3600)
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- Proof that the time for the train to pass the jogger is 36 seconds -/
theorem train_passes_jogger_in_36_seconds :
  train_passing_jogger_time 9 45 120 240 = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_jogger_time_train_passes_jogger_in_36_seconds_l2260_226085


namespace NUMINAMATH_CALUDE_easter_egg_count_l2260_226030

/-- The number of Easter eggs found in the club house -/
def club_house_eggs : ℕ := 40

/-- The number of Easter eggs found in the park -/
def park_eggs : ℕ := 25

/-- The number of Easter eggs found in the town hall -/
def town_hall_eggs : ℕ := 15

/-- The total number of Easter eggs found -/
def total_eggs : ℕ := club_house_eggs + park_eggs + town_hall_eggs

theorem easter_egg_count : total_eggs = 80 := by
  sorry

end NUMINAMATH_CALUDE_easter_egg_count_l2260_226030


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_ten_l2260_226049

-- Define the variables
variable (a b c : ℝ)

-- State the theorem
theorem sqrt_sum_equals_sqrt_ten
  (h1 : (2*a + 2)^(1/3) = 2)
  (h2 : b^(1/2) = 2)
  (h3 : ⌊Real.sqrt 15⌋ = c)
  : Real.sqrt (a + b + c) = Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_ten_l2260_226049


namespace NUMINAMATH_CALUDE_x_current_age_l2260_226057

-- Define variables for current ages
variable (x y : ℕ)

-- Define the conditions
def condition1 : Prop := x - 3 = 2 * (y - 3)
def condition2 : Prop := (x + 7) + (y + 7) = 83

-- Theorem to prove
theorem x_current_age (h1 : condition1 x y) (h2 : condition2 x y) : x = 45 := by
  sorry

end NUMINAMATH_CALUDE_x_current_age_l2260_226057


namespace NUMINAMATH_CALUDE_distance_between_complex_points_l2260_226067

theorem distance_between_complex_points :
  let z₁ : ℂ := 3 - 4*I
  let z₂ : ℂ := -2 - 3*I
  Complex.abs (z₁ - z₂) = Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_complex_points_l2260_226067


namespace NUMINAMATH_CALUDE_circle_equation_correct_l2260_226033

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  center : PolarPoint
  radius : ℝ

/-- The equation of a circle in polar coordinates -/
def circleEquation (c : PolarCircle) (p : PolarPoint) : Prop :=
  p.ρ = 2 * Real.cos (p.θ - c.center.θ)

theorem circle_equation_correct (c : PolarCircle) :
  c.center.ρ = 1 ∧ c.center.θ = 1 ∧ c.radius = 1 →
  ∀ p : PolarPoint, circleEquation c p ↔
    (p.ρ * Real.cos p.θ - c.center.ρ * Real.cos c.center.θ)^2 +
    (p.ρ * Real.sin p.θ - c.center.ρ * Real.sin c.center.θ)^2 = c.radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l2260_226033


namespace NUMINAMATH_CALUDE_sum_of_two_primes_10003_l2260_226055

/-- A function that returns true if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The number of ways 10003 can be written as the sum of two primes -/
theorem sum_of_two_primes_10003 :
  ∃! (p q : ℕ), isPrime p ∧ isPrime q ∧ p + q = 10003 :=
sorry

end NUMINAMATH_CALUDE_sum_of_two_primes_10003_l2260_226055


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_modified_fibonacci_factorial_series_l2260_226084

def fibonacci_factorial_series : List ℕ := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 55]

def last_two_digits (n : ℕ) : ℕ := n % 100

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem sum_of_last_two_digits_of_modified_fibonacci_factorial_series :
  (fibonacci_factorial_series.map (λ n => last_two_digits (factorial n))).sum % 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_modified_fibonacci_factorial_series_l2260_226084


namespace NUMINAMATH_CALUDE_prob_second_odd_given_first_even_l2260_226059

/-- A card is represented by a natural number between 1 and 5 -/
def Card : Type := { n : ℕ // 1 ≤ n ∧ n ≤ 5 }

/-- The set of all cards -/
def allCards : Finset Card := sorry

/-- A card is even if its number is even -/
def isEven (c : Card) : Prop := c.val % 2 = 0

/-- A card is odd if its number is odd -/
def isOdd (c : Card) : Prop := c.val % 2 = 1

/-- The set of even cards -/
def evenCards : Finset Card := sorry

/-- The set of odd cards -/
def oddCards : Finset Card := sorry

theorem prob_second_odd_given_first_even :
  (Finset.card oddCards : ℚ) / (Finset.card allCards - 1 : ℚ) = 3/4 := by sorry

end NUMINAMATH_CALUDE_prob_second_odd_given_first_even_l2260_226059


namespace NUMINAMATH_CALUDE_min_value_of_f_l2260_226078

noncomputable def f (a : ℝ) : ℝ := a/2 - 1/4 + (Real.exp (-2*a))/2

theorem min_value_of_f :
  ∃ (a : ℝ), a > 0 ∧ 
  (∀ (b : ℝ), b > 0 → f a ≤ f b) ∧
  a = Real.log 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2260_226078


namespace NUMINAMATH_CALUDE_first_car_years_earlier_l2260_226099

-- Define the manufacture years of the cars
def first_car_year : ℕ := 1970
def third_car_year : ℕ := 2000

-- Define the time difference between the second and third cars
def years_between_second_and_third : ℕ := 20

-- Define the manufacture year of the second car
def second_car_year : ℕ := third_car_year - years_between_second_and_third

-- Theorem to prove
theorem first_car_years_earlier (h : second_car_year = third_car_year - years_between_second_and_third) :
  second_car_year - first_car_year = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_car_years_earlier_l2260_226099


namespace NUMINAMATH_CALUDE_shirt_cost_problem_l2260_226093

theorem shirt_cost_problem (total_shirts : Nat) (total_cost : ℚ) (same_cost_shirts : Nat) (other_shirt_cost : ℚ) :
  total_shirts = 5 →
  total_cost = 85 →
  same_cost_shirts = 3 →
  other_shirt_cost = 20 →
  ∃ (same_shirt_cost : ℚ),
    same_shirt_cost * same_cost_shirts + other_shirt_cost * (total_shirts - same_cost_shirts) = total_cost ∧
    same_shirt_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_problem_l2260_226093


namespace NUMINAMATH_CALUDE_negation_equivalence_l2260_226065

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 - x + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 - x₀ + 1 ≤ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2260_226065


namespace NUMINAMATH_CALUDE_houses_built_l2260_226092

theorem houses_built (original : ℕ) (current : ℕ) (built : ℕ) : 
  original = 20817 → current = 118558 → built = current - original → built = 97741 := by
sorry

end NUMINAMATH_CALUDE_houses_built_l2260_226092


namespace NUMINAMATH_CALUDE_union_equality_implies_m_value_l2260_226010

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 3, Real.sqrt m}
def B (m : ℝ) : Set ℝ := {1, m}

-- State the theorem
theorem union_equality_implies_m_value (m : ℝ) :
  A m ∪ B m = A m → m = 0 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_m_value_l2260_226010


namespace NUMINAMATH_CALUDE_circles_intersect_at_right_angle_l2260_226036

/-- Two circles intersect at right angles if and only if the sum of the squares of their radii equals the square of the distance between their centers. -/
theorem circles_intersect_at_right_angle (a b c : ℝ) :
  ∃ (x y : ℝ), (x^2 + y^2 - 2*a*x + b^2 = 0 ∧ x^2 + y^2 - 2*c*y - b^2 = 0) →
  (a^2 - b^2) + (b^2 + c^2) = a^2 + c^2 :=
sorry

end NUMINAMATH_CALUDE_circles_intersect_at_right_angle_l2260_226036


namespace NUMINAMATH_CALUDE_opponent_scissors_is_random_event_l2260_226038

/-- Represents the possible choices in the game of rock, paper, scissors -/
inductive Choice
  | Rock
  | Paper
  | Scissors

/-- Represents a game of rock, paper, scissors -/
structure RockPaperScissors where
  opponentChoice : Choice

/-- Defines what it means for an event to be random in this context -/
def isRandomEvent (game : RockPaperScissors → Prop) : Prop :=
  ∀ (c : Choice), ∃ (g : RockPaperScissors), g.opponentChoice = c ∧ game g

/-- The main theorem: opponent choosing scissors is a random event -/
theorem opponent_scissors_is_random_event :
  isRandomEvent (λ g => g.opponentChoice = Choice.Scissors) :=
sorry

end NUMINAMATH_CALUDE_opponent_scissors_is_random_event_l2260_226038


namespace NUMINAMATH_CALUDE_solve_inequality_minimum_a_l2260_226083

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - a * |x - 1|

-- Part 1
theorem solve_inequality (x : ℝ) : 
  f (-2) x > 5 ↔ x < -4/3 ∨ x > 2 := by sorry

-- Part 2
theorem minimum_a : 
  (∃ (a : ℝ), ∀ (x : ℝ), f a x ≤ a * |x + 3|) ∧
  (∀ (a : ℝ), (∀ (x : ℝ), f a x ≤ a * |x + 3|) → a ≥ 1/2) := by sorry

end NUMINAMATH_CALUDE_solve_inequality_minimum_a_l2260_226083


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_negation_of_proposition_l2260_226022

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x < 2, P x) ↔ (∀ x < 2, ¬ P x) := by sorry

theorem negation_of_inequality (x : ℝ) :
  ¬(x^2 - 2*x < 0) ↔ (x^2 - 2*x ≥ 0) := by sorry

theorem negation_of_proposition :
  (¬ ∃ x < 2, x^2 - 2*x < 0) ↔ (∀ x < 2, x^2 - 2*x ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_negation_of_proposition_l2260_226022


namespace NUMINAMATH_CALUDE_twenty_five_is_eighty_percent_of_l2260_226091

theorem twenty_five_is_eighty_percent_of : ∃ y : ℝ, (25 : ℝ) = 0.8 * y ∧ y = 31.25 := by
  sorry

end NUMINAMATH_CALUDE_twenty_five_is_eighty_percent_of_l2260_226091


namespace NUMINAMATH_CALUDE_people_count_is_32_l2260_226017

/-- Given a room with chairs and people, calculate the number of people in the room. -/
def people_in_room (empty_chairs : ℕ) : ℕ :=
  let total_chairs := 3 * empty_chairs
  let seated_people := 2 * empty_chairs
  2 * seated_people

/-- Prove that the number of people in the room is 32, given the problem conditions. -/
theorem people_count_is_32 :
  let empty_chairs := 8
  let total_people := people_in_room empty_chairs
  let total_chairs := 3 * empty_chairs
  let seated_people := 2 * empty_chairs
  (2 * seated_people = total_people) ∧
  (seated_people = total_people / 2) ∧
  (seated_people = 2 * total_chairs / 3) ∧
  (total_people = 32) :=
by
  sorry

#eval people_in_room 8

end NUMINAMATH_CALUDE_people_count_is_32_l2260_226017


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l2260_226013

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (∃ (k : ℤ), (10*x - 4) * (10*x) * (5*x + 15) = 1200 * k) ∧
  (∀ (m : ℤ), m > 1200 → ∃ (y : ℤ), Odd y ∧ ¬(∃ (l : ℤ), (10*y - 4) * (10*y) * (5*y + 15) = m * l)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l2260_226013


namespace NUMINAMATH_CALUDE_min_surface_area_five_cubes_l2260_226020

/-- Represents a shape made of unit cubes -/
structure Shape :=
  (num_cubes : ℕ)
  (num_joins : ℕ)

/-- Calculates the surface area of a shape -/
def surface_area (s : Shape) : ℕ :=
  s.num_cubes * 6 - s.num_joins * 2

/-- Theorem: Among shapes with 5 unit cubes, the one with 5 joins has the smallest surface area -/
theorem min_surface_area_five_cubes (s : Shape) (h1 : s.num_cubes = 5) (h2 : s.num_joins ≤ 5) :
  surface_area s ≥ surface_area { num_cubes := 5, num_joins := 5 } :=
sorry

end NUMINAMATH_CALUDE_min_surface_area_five_cubes_l2260_226020


namespace NUMINAMATH_CALUDE_line_equation_proof_l2260_226073

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point is on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The projection of one point onto a line -/
def Point.projection (p : Point) (l : Line) : Point :=
  sorry

theorem line_equation_proof (A : Point) (P : Point) (l : Line) :
  A.x = 1 ∧ A.y = 2 ∧ P.x = -1 ∧ P.y = 4 ∧ P = A.projection l →
  l.a = 1 ∧ l.b = -1 ∧ l.c = 5 :=
sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2260_226073


namespace NUMINAMATH_CALUDE_max_sum_of_squares_max_sum_of_squares_value_exact_max_sum_of_squares_l2260_226026

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 18 →
  a * b + c + d = 95 →
  a * d + b * c = 180 →
  c * d = 105 →
  ∀ (w x y z : ℝ), 
    w + x = 18 →
    w * x + y + z = 95 →
    w * z + x * y = 180 →
    y * z = 105 →
    a^2 + b^2 + c^2 + d^2 ≥ w^2 + x^2 + y^2 + z^2 :=
by
  sorry

theorem max_sum_of_squares_value (a b c d : ℝ) :
  a + b = 18 →
  a * b + c + d = 95 →
  a * d + b * c = 180 →
  c * d = 105 →
  a^2 + b^2 + c^2 + d^2 ≤ 1486 :=
by
  sorry

theorem exact_max_sum_of_squares (a b c d : ℝ) :
  a + b = 18 →
  a * b + c + d = 95 →
  a * d + b * c = 180 →
  c * d = 105 →
  ∃ (w x y z : ℝ),
    w + x = 18 ∧
    w * x + y + z = 95 ∧
    w * z + x * y = 180 ∧
    y * z = 105 ∧
    w^2 + x^2 + y^2 + z^2 = 1486 :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_max_sum_of_squares_value_exact_max_sum_of_squares_l2260_226026


namespace NUMINAMATH_CALUDE_total_kites_sold_l2260_226098

-- Define the sequence
def kite_sequence (n : ℕ) : ℕ := 2 + 3 * (n - 1)

-- Define the sum of the sequence
def kite_sum (n : ℕ) : ℕ := 
  n * (kite_sequence 1 + kite_sequence n) / 2

-- Theorem statement
theorem total_kites_sold : kite_sum 15 = 345 := by
  sorry

end NUMINAMATH_CALUDE_total_kites_sold_l2260_226098


namespace NUMINAMATH_CALUDE_inequality_and_minimum_l2260_226076

theorem inequality_and_minimum (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum_prod : x + y + z ≥ x * y * z) : 
  (x / (y * z) + y / (z * x) + z / (x * y) ≥ 1 / x + 1 / y + 1 / z) ∧ 
  (∃ (u : ℝ), u = x / (y * z) + y / (z * x) + z / (x * y) ∧ 
              u ≥ Real.sqrt 3 ∧ 
              ∀ (v : ℝ), v = x / (y * z) + y / (z * x) + z / (x * y) → v ≥ u) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_l2260_226076


namespace NUMINAMATH_CALUDE_boys_who_love_marbles_l2260_226053

/-- The number of marbles Haley has -/
def total_marbles : ℕ := 20

/-- The number of marbles each boy receives -/
def marbles_per_boy : ℕ := 10

/-- The number of boys who love to play marbles -/
def num_boys : ℕ := total_marbles / marbles_per_boy

theorem boys_who_love_marbles : num_boys = 2 := by
  sorry

end NUMINAMATH_CALUDE_boys_who_love_marbles_l2260_226053


namespace NUMINAMATH_CALUDE_correct_arrangements_l2260_226027

/-- The number of different arrangements of representatives for 7 subjects -/
def num_arrangements (num_boys num_girls num_subjects : ℕ) : ℕ :=
  num_boys * num_girls * (Nat.factorial (num_subjects - 2))

/-- Theorem stating the correct number of arrangements -/
theorem correct_arrangements :
  num_arrangements 4 3 7 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_correct_arrangements_l2260_226027


namespace NUMINAMATH_CALUDE_island_navigation_time_l2260_226080

/-- The time to navigate around the island once, in minutes -/
def navigation_time : ℕ := 30

/-- The total number of rounds completed over the weekend -/
def total_rounds : ℕ := 26

/-- The total time spent circling the island over the weekend, in minutes -/
def total_time : ℕ := 780

/-- Proof that the navigation time around the island is 30 minutes -/
theorem island_navigation_time :
  navigation_time * total_rounds = total_time :=
by sorry

end NUMINAMATH_CALUDE_island_navigation_time_l2260_226080
