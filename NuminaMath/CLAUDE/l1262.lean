import Mathlib

namespace NUMINAMATH_CALUDE_certain_number_is_fourteen_l1262_126207

/-- A certain number multiplied by d is the square of an integer -/
def is_square_multiple (n : ℕ) (d : ℕ) : Prop :=
  ∃ m : ℕ, n * d = m^2

/-- d is the smallest positive integer satisfying the condition -/
def is_smallest_d (n : ℕ) (d : ℕ) : Prop :=
  is_square_multiple n d ∧ ∀ k < d, ¬(is_square_multiple n k)

theorem certain_number_is_fourteen (d : ℕ) (h1 : d = 14) 
  (h2 : ∃ n : ℕ, is_smallest_d n d) : 
  ∃ n : ℕ, is_smallest_d n d ∧ n = 14 :=
sorry

end NUMINAMATH_CALUDE_certain_number_is_fourteen_l1262_126207


namespace NUMINAMATH_CALUDE_inequality_proof_l1262_126279

theorem inequality_proof (x y z : ℝ) :
  -3/2 * (x^2 + y^2 + 2*z^2) ≤ 3*x*y + y*z + z*x ∧
  3*x*y + y*z + z*x ≤ (3 + Real.sqrt 13)/4 * (x^2 + y^2 + 2*z^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1262_126279


namespace NUMINAMATH_CALUDE_function_property_l1262_126260

theorem function_property (α : ℝ) (hα : α > 0) :
  ∃ (b : ℝ), ∀ (f : ℕ+ → ℝ),
    (∀ (k m : ℕ+), α * m ≤ k ∧ k < (α + 1) * m → f (k + m) = f k + f m) ↔
    (∀ (n : ℕ+), f n = b * n) :=
by sorry

end NUMINAMATH_CALUDE_function_property_l1262_126260


namespace NUMINAMATH_CALUDE_no_m_for_equality_subset_condition_l1262_126229

def P : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}
def S (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

theorem no_m_for_equality : ¬∃ m : ℝ, P = S m := by sorry

theorem subset_condition (m : ℝ) (h : m ≤ 0) : S m ⊆ P := by sorry

end NUMINAMATH_CALUDE_no_m_for_equality_subset_condition_l1262_126229


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1262_126256

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  (a 2 + a 6 = 8) ∧
  (a 3 + a 4 = 3)

/-- The common difference of the arithmetic sequence is 5 -/
theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1262_126256


namespace NUMINAMATH_CALUDE_playground_fence_posts_l1262_126224

/-- Calculates the minimum number of fence posts needed for a rectangular playground. -/
def min_fence_posts (length width post_spacing : ℕ) : ℕ :=
  let long_side := max length width
  let short_side := min length width
  let long_side_posts := long_side / post_spacing + 1
  let short_side_posts := short_side / post_spacing + 1
  long_side_posts + 2 * (short_side_posts - 1)

/-- Theorem stating the minimum number of fence posts needed for the given playground. -/
theorem playground_fence_posts :
  min_fence_posts 100 50 10 = 21 :=
by
  sorry

#eval min_fence_posts 100 50 10

end NUMINAMATH_CALUDE_playground_fence_posts_l1262_126224


namespace NUMINAMATH_CALUDE_estimate_comparison_l1262_126236

theorem estimate_comparison (x y z w : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 1) (hw : w > 0) (hxy : x > y) :
  (x + w) - (y - w) * z > x - y * z := by
  sorry

end NUMINAMATH_CALUDE_estimate_comparison_l1262_126236


namespace NUMINAMATH_CALUDE_peter_notebooks_l1262_126245

def green_notebooks : ℕ := 2
def black_notebooks : ℕ := 1
def pink_notebooks : ℕ := 1

def total_notebooks : ℕ := green_notebooks + black_notebooks + pink_notebooks

theorem peter_notebooks : total_notebooks = 4 := by sorry

end NUMINAMATH_CALUDE_peter_notebooks_l1262_126245


namespace NUMINAMATH_CALUDE_work_completion_men_count_l1262_126258

/-- Given a work that can be completed by M men in 20 days, 
    or by (M - 4) men in 25 days, prove that M = 16. -/
theorem work_completion_men_count :
  ∀ (M : ℕ) (W : ℝ),
  (M : ℝ) * (W / 20) = (M - 4 : ℝ) * (W / 25) →
  M = 16 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_men_count_l1262_126258


namespace NUMINAMATH_CALUDE_four_digit_square_and_cube_l1262_126213

theorem four_digit_square_and_cube (a : ℕ) :
  (1000 ≤ 4 * a^2) ∧ (4 * a^2 < 10000) ∧
  (1000 ≤ (4 / 3) * a^3) ∧ ((4 / 3) * a^3 < 10000) ∧
  (∃ (n : ℕ), (4 / 3) * a^3 = n) →
  a = 18 := by sorry

end NUMINAMATH_CALUDE_four_digit_square_and_cube_l1262_126213


namespace NUMINAMATH_CALUDE_bob_distance_from_start_l1262_126251

/-- Regular hexagon with side length 3 km -/
structure RegularHexagon where
  side_length : ℝ
  is_regular : side_length = 3

/-- Position after walking along the perimeter -/
def position_after_walk (h : RegularHexagon) (distance : ℝ) : ℝ × ℝ :=
  sorry

/-- Distance between two points -/
def distance_between_points (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

theorem bob_distance_from_start (h : RegularHexagon) :
  let start_point := (0, 0)
  let end_point := position_after_walk h 7
  distance_between_points start_point end_point = 2 := by
  sorry

end NUMINAMATH_CALUDE_bob_distance_from_start_l1262_126251


namespace NUMINAMATH_CALUDE_flower_pots_theorem_l1262_126220

/-- Represents the number of pots of each type of flower seedling --/
structure FlowerPots where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if the given FlowerPots satisfies all conditions --/
def isValid (pots : FlowerPots) : Prop :=
  pots.a > 0 ∧ pots.b > 0 ∧ pots.c > 0 ∧
  pots.a + pots.b + pots.c = 16 ∧
  2 * pots.a + 4 * pots.b + 10 * pots.c = 50

/-- The theorem stating that the only valid numbers of pots for type A are 10 and 13 --/
theorem flower_pots_theorem :
  ∀ pots : FlowerPots, isValid pots → pots.a = 10 ∨ pots.a = 13 := by
  sorry

end NUMINAMATH_CALUDE_flower_pots_theorem_l1262_126220


namespace NUMINAMATH_CALUDE_arkansas_game_profit_calculation_l1262_126202

/-- The amount of money made per t-shirt in dollars -/
def profit_per_shirt : ℕ := 98

/-- The total number of t-shirts sold during both games -/
def total_shirts_sold : ℕ := 163

/-- The number of t-shirts sold during the Arkansas game -/
def arkansas_shirts_sold : ℕ := 89

/-- The money made from selling t-shirts during the Arkansas game -/
def arkansas_game_profit : ℕ := arkansas_shirts_sold * profit_per_shirt

theorem arkansas_game_profit_calculation :
  arkansas_game_profit = 8722 :=
sorry

end NUMINAMATH_CALUDE_arkansas_game_profit_calculation_l1262_126202


namespace NUMINAMATH_CALUDE_yellow_highlighters_count_l1262_126248

theorem yellow_highlighters_count (pink : ℕ) (blue : ℕ) (total : ℕ) (yellow : ℕ) : 
  pink = 10 → blue = 8 → total = 33 → yellow = total - (pink + blue) → yellow = 15 := by
  sorry

end NUMINAMATH_CALUDE_yellow_highlighters_count_l1262_126248


namespace NUMINAMATH_CALUDE_problem_solution_l1262_126211

theorem problem_solution (m : ℤ) (a b c : ℝ) 
  (h1 : ∃! (x : ℤ), |2 * (x : ℝ) - m| ≤ 1 ∧ x = 2)
  (h2 : 4 * a^4 + 4 * b^4 + 4 * c^4 = m) :
  m = 4 ∧ (a^2 + b^2 + c^2 ≤ Real.sqrt 3 ∧ ∃ x y z, x^2 + y^2 + z^2 = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1262_126211


namespace NUMINAMATH_CALUDE_unique_remainder_in_range_l1262_126218

theorem unique_remainder_in_range : ∃! n : ℕ, n ≤ 100 ∧ n % 9 = 3 ∧ n % 13 = 5 ∧ n = 57 := by
  sorry

end NUMINAMATH_CALUDE_unique_remainder_in_range_l1262_126218


namespace NUMINAMATH_CALUDE_percent_juniors_in_sports_l1262_126286

theorem percent_juniors_in_sports (total_students : ℕ) (percent_juniors : ℚ) (juniors_in_sports : ℕ) :
  total_students = 500 →
  percent_juniors = 40 / 100 →
  juniors_in_sports = 140 →
  (juniors_in_sports : ℚ) / (percent_juniors * total_students) * 100 = 70 := by
  sorry


end NUMINAMATH_CALUDE_percent_juniors_in_sports_l1262_126286


namespace NUMINAMATH_CALUDE_negation_equivalence_l1262_126278

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - 4*x + 2 > 0) ↔ (∀ x : ℝ, x^2 - 4*x + 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1262_126278


namespace NUMINAMATH_CALUDE_triangle_inequality_l1262_126298

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  (-a^2 + b^2 + c^2) * (a^2 - b^2 + c^2) * (a^2 + b^2 - c^2) ≤ a^2 * b^2 * c^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1262_126298


namespace NUMINAMATH_CALUDE_infinite_pairs_with_2020_diff_l1262_126266

/-- A positive integer is square-free if it is not divisible by any perfect square other than 1. -/
def IsSquareFree (n : ℕ) : Prop :=
  ∀ d : ℕ, d > 1 → d * d ∣ n → d = 1

/-- The sequence of square-free positive integers in ascending order. -/
def SquareFreeSequence : ℕ → ℕ := sorry

/-- The property that all integers between two given numbers are not square-free. -/
def AllBetweenNotSquareFree (m n : ℕ) : Prop :=
  ∀ k : ℕ, m < k → k < n → ¬(IsSquareFree k)

/-- The main theorem stating that there are infinitely many pairs of consecutive
    square-free integers in the sequence with a difference of 2020. -/
theorem infinite_pairs_with_2020_diff :
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧
    IsSquareFree (SquareFreeSequence n) ∧
    IsSquareFree (SquareFreeSequence (n + 1)) ∧
    SquareFreeSequence (n + 1) - SquareFreeSequence n = 2020 ∧
    AllBetweenNotSquareFree (SquareFreeSequence n) (SquareFreeSequence (n + 1)) :=
sorry

end NUMINAMATH_CALUDE_infinite_pairs_with_2020_diff_l1262_126266


namespace NUMINAMATH_CALUDE_positive_real_solution_of_equation_l1262_126239

theorem positive_real_solution_of_equation : 
  ∃! (x : ℝ), x > 0 ∧ (x - 6) / 11 = 6 / (x - 11) ∧ x = 17 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_solution_of_equation_l1262_126239


namespace NUMINAMATH_CALUDE_kellys_grade_is_42_l1262_126217

/-- Calculates Kelly's grade based on the grades of Jenny, Jason, and Bob -/
def kellysGrade (jennysGrade : ℕ) : ℕ :=
  let jasonsGrade := jennysGrade - 25
  let bobsGrade := jasonsGrade / 2
  let kellyGradeIncrease := bobsGrade * 20 / 100
  bobsGrade + kellyGradeIncrease

/-- Theorem stating that Kelly's grade is 42 given the conditions in the problem -/
theorem kellys_grade_is_42 : kellysGrade 95 = 42 := by
  sorry

end NUMINAMATH_CALUDE_kellys_grade_is_42_l1262_126217


namespace NUMINAMATH_CALUDE_walnut_trees_planted_l1262_126210

/-- The number of walnut trees planted in a park -/
def trees_planted (initial final : ℕ) : ℕ := final - initial

/-- Theorem: The number of walnut trees planted is the difference between
    the final number of trees and the initial number of trees -/
theorem walnut_trees_planted (initial final planted : ℕ) 
  (h1 : initial = 107)
  (h2 : final = 211)
  (h3 : planted = trees_planted initial final) :
  planted = 104 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_planted_l1262_126210


namespace NUMINAMATH_CALUDE_sum_of_roots_for_given_equation_l1262_126285

theorem sum_of_roots_for_given_equation :
  ∀ x₁ x₂ : ℝ, (x₁ - 2) * (x₁ + 5) = 28 ∧ (x₂ - 2) * (x₂ + 5) = 28 →
  x₁ + x₂ = -3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_for_given_equation_l1262_126285


namespace NUMINAMATH_CALUDE_percentage_change_equivalence_l1262_126204

theorem percentage_change_equivalence (r s N : ℝ) 
  (hr : r > 0) (hs : s > 0) (hN : N > 0) (hs_bound : s < 50) : 
  N * (1 + r/100) * (1 - s/100) < N ↔ r < 50*s / (100 - s) := by
  sorry

end NUMINAMATH_CALUDE_percentage_change_equivalence_l1262_126204


namespace NUMINAMATH_CALUDE_ribbon_gap_theorem_l1262_126288

theorem ribbon_gap_theorem (R : ℝ) (h : R > 0) :
  let original_length := 2 * Real.pi * R
  let new_length := original_length + 1
  let new_radius := R + (new_length / (2 * Real.pi) - R)
  new_radius - R = 1 / (2 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ribbon_gap_theorem_l1262_126288


namespace NUMINAMATH_CALUDE_weight_of_A_l1262_126282

theorem weight_of_A (a b c d : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  (b + c + d + (d + 6)) / 4 = 79 →
  a = 174 :=
by sorry

end NUMINAMATH_CALUDE_weight_of_A_l1262_126282


namespace NUMINAMATH_CALUDE_max_points_for_top_teams_l1262_126206

/-- Represents a football tournament with the given rules --/
structure FootballTournament where
  num_teams : ℕ
  num_top_teams : ℕ
  points_for_win : ℕ
  points_for_draw : ℕ

/-- The maximum possible points that can be achieved by the top teams --/
def max_points (t : FootballTournament) : ℕ :=
  let internal_games := t.num_top_teams.choose 2
  let external_games := t.num_top_teams * (t.num_teams - t.num_top_teams)
  internal_games * t.points_for_win + external_games * t.points_for_win

/-- The theorem stating the maximum integer N for which at least 6 teams can score N or more points --/
theorem max_points_for_top_teams (t : FootballTournament) 
  (h1 : t.num_teams = 15)
  (h2 : t.num_top_teams = 6)
  (h3 : t.points_for_win = 3)
  (h4 : t.points_for_draw = 1) :
  ∃ (N : ℕ), N = 34 ∧ 
  (∀ (M : ℕ), (M : ℝ) * t.num_top_teams ≤ max_points t → M ≤ N) ∧
  (N : ℝ) * t.num_top_teams ≤ max_points t :=
by sorry

end NUMINAMATH_CALUDE_max_points_for_top_teams_l1262_126206


namespace NUMINAMATH_CALUDE_point_on_line_not_perpendicular_to_y_axis_l1262_126231

-- Define a line l with equation x + my - 2 = 0
def line_l (m : ℝ) (x y : ℝ) : Prop := x + m * y - 2 = 0

-- Theorem stating that (2,0) always lies on line l
theorem point_on_line (m : ℝ) : line_l m 2 0 := by sorry

-- Theorem stating that line l is not perpendicular to the y-axis
theorem not_perpendicular_to_y_axis (m : ℝ) : m ≠ 0 := by sorry

end NUMINAMATH_CALUDE_point_on_line_not_perpendicular_to_y_axis_l1262_126231


namespace NUMINAMATH_CALUDE_combined_population_is_8000_l1262_126268

/-- The total population of five towns -/
def total_population : ℕ := 120000

/-- The population of Gordonia -/
def gordonia_population : ℕ := total_population / 3

/-- The population of Toadon -/
def toadon_population : ℕ := (gordonia_population * 3) / 4

/-- The population of Riverbank -/
def riverbank_population : ℕ := toadon_population + (toadon_population * 2) / 5

/-- The combined population of Lake Bright and Sunshine Hills -/
def lake_bright_sunshine_hills_population : ℕ := 
  total_population - (gordonia_population + toadon_population + riverbank_population)

theorem combined_population_is_8000 : 
  lake_bright_sunshine_hills_population = 8000 := by
  sorry

end NUMINAMATH_CALUDE_combined_population_is_8000_l1262_126268


namespace NUMINAMATH_CALUDE_direct_proportion_k_value_l1262_126201

def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, f x = m * x

theorem direct_proportion_k_value :
  ∀ k : ℝ,
  (∀ x y : ℝ, y = (k - 1) * x + k^2 - 1) →
  (is_direct_proportion (λ x => (k - 1) * x + k^2 - 1)) →
  k = -1 :=
by sorry

end NUMINAMATH_CALUDE_direct_proportion_k_value_l1262_126201


namespace NUMINAMATH_CALUDE_stock_purchase_probabilities_l1262_126289

/-- The number of stocks available for purchase -/
def num_stocks : ℕ := 6

/-- The number of individuals making purchases -/
def num_individuals : ℕ := 4

/-- The probability that all individuals purchase the same stock -/
def prob_all_same : ℚ := 1 / 216

/-- The probability that at most two individuals purchase the same stock -/
def prob_at_most_two_same : ℚ := 65 / 72

/-- Given 6 stocks and 4 individuals randomly selecting one stock each,
    prove the probabilities of certain outcomes -/
theorem stock_purchase_probabilities :
  (prob_all_same = 1 / num_stocks ^ (num_individuals - 1)) ∧
  (prob_at_most_two_same = 
    (num_stocks * (num_stocks - 1) * Nat.choose num_individuals 2 + 
     num_stocks * Nat.factorial num_individuals) / 
    (num_stocks ^ num_individuals)) := by
  sorry

end NUMINAMATH_CALUDE_stock_purchase_probabilities_l1262_126289


namespace NUMINAMATH_CALUDE_maria_anna_age_sum_prove_maria_anna_age_sum_l1262_126205

theorem maria_anna_age_sum : ℕ → ℕ → Prop :=
  fun maria_age anna_age =>
    (maria_age = anna_age + 5) →
    (maria_age + 7 = 3 * (anna_age - 3)) →
    (maria_age + anna_age = 27)

#check maria_anna_age_sum

theorem prove_maria_anna_age_sum :
  ∃ (maria_age anna_age : ℕ), maria_anna_age_sum maria_age anna_age :=
by
  sorry

end NUMINAMATH_CALUDE_maria_anna_age_sum_prove_maria_anna_age_sum_l1262_126205


namespace NUMINAMATH_CALUDE_sixteen_solutions_l1262_126232

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 4*x

-- State the theorem
theorem sixteen_solutions :
  ∃! (s : Finset ℝ), (∀ c ∈ s, g (g (g (g c))) = 2) ∧ Finset.card s = 16 :=
sorry

end NUMINAMATH_CALUDE_sixteen_solutions_l1262_126232


namespace NUMINAMATH_CALUDE_shortest_chord_length_max_triangle_area_l1262_126230

-- Define the circle and point A
def circle_radius : ℝ := 1
def distance_OA (a : ℝ) : Prop := 0 < a ∧ a < 1

-- Theorem for the shortest chord length
theorem shortest_chord_length (a : ℝ) (h : distance_OA a) :
  ∃ (chord_length : ℝ), chord_length = 2 * Real.sqrt (1 - a^2) ∧
  ∀ (other_chord : ℝ), other_chord ≥ chord_length :=
sorry

-- Theorem for the maximum area of triangle OMN
theorem max_triangle_area (a : ℝ) (h : distance_OA a) :
  ∃ (max_area : ℝ),
    (a ≥ Real.sqrt 2 / 2 → max_area = 1 / 2) ∧
    (a < Real.sqrt 2 / 2 → max_area = a * Real.sqrt (1 - a^2)) :=
sorry

end NUMINAMATH_CALUDE_shortest_chord_length_max_triangle_area_l1262_126230


namespace NUMINAMATH_CALUDE_friend_lunch_cost_l1262_126261

theorem friend_lunch_cost (total : ℝ) (difference : ℝ) (friend_cost : ℝ) : 
  total = 11 →
  difference = 3 →
  friend_cost = (total + difference) / 2 →
  friend_cost = 7 :=
by sorry

end NUMINAMATH_CALUDE_friend_lunch_cost_l1262_126261


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l1262_126238

theorem modulus_of_complex_number (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := i * (2 - i)
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l1262_126238


namespace NUMINAMATH_CALUDE_sum_of_cubes_l1262_126200

theorem sum_of_cubes (a b c : ℝ) 
  (sum_eq : a + b + c = 8)
  (sum_products_eq : a * b + a * c + b * c = 10)
  (product_eq : a * b * c = -15) :
  a^3 + b^3 + c^3 = 227 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l1262_126200


namespace NUMINAMATH_CALUDE_multiple_with_all_digits_l1262_126284

theorem multiple_with_all_digits (n : ℤ) : ∃ m : ℤ, 
  (∃ k : ℤ, m = n * k) ∧ 
  (∀ d : ℕ, d < 10 → ∃ p : ℕ, (m.natAbs / 10^p) % 10 = d) :=
sorry

end NUMINAMATH_CALUDE_multiple_with_all_digits_l1262_126284


namespace NUMINAMATH_CALUDE_parabola_sum_l1262_126244

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℚ × ℚ := (3, -2)

/-- The parabola contains the point (0, 5) -/
def contains_point (p : Parabola) : Prop :=
  p.a * 0^2 + p.b * 0 + p.c = 5

/-- The axis of symmetry is vertical -/
def vertical_axis_of_symmetry (p : Parabola) : Prop :=
  ∃ x : ℚ, ∀ y : ℚ, p.a * (x - 3)^2 = y + 2

theorem parabola_sum (p : Parabola) 
  (h1 : vertex p = (3, -2))
  (h2 : contains_point p)
  (h3 : vertical_axis_of_symmetry p) :
  p.a + p.b + p.c = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_l1262_126244


namespace NUMINAMATH_CALUDE_a_10_equals_512_l1262_126203

/-- The sequence {aₙ} where Sₙ = 2aₙ - 1 for all n ∈ ℕ⁺, and Sₙ is the sum of the first n terms of {aₙ} -/
def sequence_a (n : ℕ+) : ℝ :=
  sorry

/-- The sum of the first n terms of the sequence {aₙ} -/
def S (n : ℕ+) : ℝ :=
  sorry

/-- The main theorem stating that a₁₀ = 512 -/
theorem a_10_equals_512 (h : ∀ n : ℕ+, S n = 2 * sequence_a n - 1) : sequence_a 10 = 512 := by
  sorry

end NUMINAMATH_CALUDE_a_10_equals_512_l1262_126203


namespace NUMINAMATH_CALUDE_cookies_eaten_vs_given_l1262_126246

theorem cookies_eaten_vs_given (initial_cookies : ℕ) (eaten_cookies : ℕ) (given_cookies : ℕ) 
  (h1 : initial_cookies = 17) 
  (h2 : eaten_cookies = 14) 
  (h3 : given_cookies = 13) :
  eaten_cookies - given_cookies = 1 := by
  sorry

end NUMINAMATH_CALUDE_cookies_eaten_vs_given_l1262_126246


namespace NUMINAMATH_CALUDE_g_composition_of_three_l1262_126214

def g (x : ℤ) : ℤ :=
  if x % 2 = 0 then x / 2 else 3 * x - 1

theorem g_composition_of_three : g (g (g (g 3))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_three_l1262_126214


namespace NUMINAMATH_CALUDE_difference_of_squares_division_l1262_126223

theorem difference_of_squares_division : (121^2 - 112^2) / 9 = 233 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_division_l1262_126223


namespace NUMINAMATH_CALUDE_function_properties_l1262_126272

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem function_properties (f : ℝ → ℝ) 
  (h_odd : is_odd (fun x => f (2*x + 1)))
  (h_period : has_period (fun x => f (2*x + 1)) 2) :
  (∀ x, f (x + 1) + f (-x + 1) = 0) ∧
  (∀ x, f x = f (x + 4)) := by
sorry

end NUMINAMATH_CALUDE_function_properties_l1262_126272


namespace NUMINAMATH_CALUDE_set_operations_and_subset_l1262_126240

def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem set_operations_and_subset :
  (A ∩ B = {x | 3 ≤ x ∧ x < 6}) ∧
  ((Bᶜ ∪ A) = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ 9 ≤ x}) ∧
  (∀ a : ℝ, C a ⊆ B → (2 ≤ a ∧ a ≤ 8)) :=
sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_l1262_126240


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1262_126233

-- Define the triangle's side lengths
def side1 : ℝ := 7
def side2 : ℝ := 10
def side3 : ℝ := 15

-- Define the perimeter of the triangle
def perimeter : ℝ := side1 + side2 + side3

-- Theorem: The perimeter of the triangle is 32
theorem triangle_perimeter : perimeter = 32 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1262_126233


namespace NUMINAMATH_CALUDE_set_mean_given_median_l1262_126225

theorem set_mean_given_median (n : ℝ) :
  (Finset.range 5).card = 5 →
  n + 8 = 14 →
  let s := {n, n + 6, n + 8, n + 10, n + 18}
  (Finset.filter (λ x => x ≤ n + 8) s).card = 3 →
  (Finset.sum s id) / 5 = 14.4 := by
sorry

end NUMINAMATH_CALUDE_set_mean_given_median_l1262_126225


namespace NUMINAMATH_CALUDE_initial_sets_count_l1262_126297

/-- The number of letters available (A through J) -/
def n : ℕ := 10

/-- The number of letters in each set of initials -/
def k : ℕ := 3

/-- The number of different three-letter sets of initials possible -/
def num_initial_sets : ℕ := n * (n - 1) * (n - 2)

/-- Theorem stating that the number of different three-letter sets of initials
    using letters A through J, with no repeated letters, is equal to 720 -/
theorem initial_sets_count : num_initial_sets = 720 := by
  sorry

end NUMINAMATH_CALUDE_initial_sets_count_l1262_126297


namespace NUMINAMATH_CALUDE_tangent_curve_intersection_l1262_126234

-- Define the curve C
def C (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 - 9 * x^2 + 4

-- Define the point M
def M : ℝ × ℝ := (1, -4)

-- Define the tangent line l
def l (x : ℝ) : ℝ := -12 * (x - 1) - 4

-- Define a function to count common points
def count_common_points (f g : ℝ → ℝ) : ℕ := sorry

-- Theorem statement
theorem tangent_curve_intersection :
  count_common_points C l = 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_curve_intersection_l1262_126234


namespace NUMINAMATH_CALUDE_prove_vector_sum_with_scalar_multiple_l1262_126292

def vector_sum_with_scalar_multiple : Prop :=
  let v1 : Fin 3 → ℝ := ![3, -2, 5]
  let v2 : Fin 3 → ℝ := ![-1, 4, -3]
  let result : Fin 3 → ℝ := ![1, 6, -1]
  v1 + 2 • v2 = result

theorem prove_vector_sum_with_scalar_multiple : vector_sum_with_scalar_multiple := by
  sorry

end NUMINAMATH_CALUDE_prove_vector_sum_with_scalar_multiple_l1262_126292


namespace NUMINAMATH_CALUDE_like_terms_exponent_difference_l1262_126270

/-- Given that 2a^m * b^2 and -a^5 * b^n are like terms, prove that n-m = -3 -/
theorem like_terms_exponent_difference (a b : ℝ) (m n : ℤ) 
  (h : ∃ (k : ℝ), 2 * a^m * b^2 = k * (-a^5 * b^n)) : n - m = -3 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_difference_l1262_126270


namespace NUMINAMATH_CALUDE_intersection_implies_sum_of_translations_l1262_126250

/-- Given two functions f and g that intersect at points (1,7) and (9,1),
    prove that the sum of their x-axis translation parameters is 10 -/
theorem intersection_implies_sum_of_translations (a b c d : ℝ) :
  (∀ x, -2 * |x - a| + b = 2 * |x - c| + d ↔ (x = 1 ∧ -2 * |x - a| + b = 7) ∨ (x = 9 ∧ -2 * |x - a| + b = 1)) →
  a + c = 10 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_of_translations_l1262_126250


namespace NUMINAMATH_CALUDE_square_difference_division_l1262_126227

theorem square_difference_division : (196^2 - 169^2) / 27 = 365 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_division_l1262_126227


namespace NUMINAMATH_CALUDE_clare_milk_cartons_l1262_126221

def prove_milk_cartons (initial_money : ℕ) (num_bread : ℕ) (cost_bread : ℕ) (cost_milk : ℕ) (money_left : ℕ) : Prop :=
  let money_spent : ℕ := initial_money - money_left
  let bread_cost : ℕ := num_bread * cost_bread
  let milk_cost : ℕ := money_spent - bread_cost
  let num_milk_cartons : ℕ := milk_cost / cost_milk
  num_milk_cartons = 2

theorem clare_milk_cartons :
  prove_milk_cartons 47 4 2 2 35 := by
  sorry

end NUMINAMATH_CALUDE_clare_milk_cartons_l1262_126221


namespace NUMINAMATH_CALUDE_emmalyn_earnings_l1262_126262

/-- Calculates the total amount earned from painting fences. -/
def total_amount_earned (price_per_meter : ℚ) (num_fences : ℕ) (fence_length : ℕ) : ℚ :=
  price_per_meter * (num_fences : ℚ) * (fence_length : ℚ)

/-- Proves that Emmalyn earned $5,000 from painting fences. -/
theorem emmalyn_earnings : 
  total_amount_earned (20 / 100) 50 500 = 5000 := by
  sorry

#eval total_amount_earned (20 / 100) 50 500

end NUMINAMATH_CALUDE_emmalyn_earnings_l1262_126262


namespace NUMINAMATH_CALUDE_solution_set_of_equations_l1262_126209

theorem solution_set_of_equations (x y : ℝ) :
  (x^2 - 2*x*y = 1 ∧ 5*x^2 - 2*x*y + 2*y^2 = 5) ↔
  ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0) ∨ (x = 1/3 ∧ y = -4/3) ∨ (x = -1/3 ∧ y = 4/3)) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_equations_l1262_126209


namespace NUMINAMATH_CALUDE_eliminate_quadratic_term_l1262_126299

/-- The polynomial we're working with -/
def polynomial (x n : ℝ) : ℝ := 4*x^2 + 2*(7 + 3*x - 3*x^2) - n*x^2

/-- The coefficient of x^2 in the expanded polynomial -/
def quadratic_coefficient (n : ℝ) : ℝ := 4 - 6 - n

theorem eliminate_quadratic_term :
  ∃ (n : ℝ), ∀ (x : ℝ), polynomial x n = 6*x + 14 ∧ n = -2 :=
sorry

end NUMINAMATH_CALUDE_eliminate_quadratic_term_l1262_126299


namespace NUMINAMATH_CALUDE_triangle_abc_right_angled_l1262_126215

theorem triangle_abc_right_angled (A B C : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : C > 0)
  (h4 : A + B + C = 180) (h5 : A / 2 = B / 3) (h6 : A / 2 = C / 5) : C = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_right_angled_l1262_126215


namespace NUMINAMATH_CALUDE_ascending_order_l1262_126281

theorem ascending_order (a b c : ℝ) (ha : a = 60.7) (hb : b = 0.76) (hc : c = Real.log 0.76) :
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ascending_order_l1262_126281


namespace NUMINAMATH_CALUDE_gift_shop_pricing_l1262_126287

theorem gift_shop_pricing (p : ℝ) (hp : p > 0) : 
  0.7 * (1.1 * p) = 0.77 * p := by sorry

end NUMINAMATH_CALUDE_gift_shop_pricing_l1262_126287


namespace NUMINAMATH_CALUDE_median_in_75_79_interval_l1262_126241

/-- Represents a score interval with its frequency --/
structure ScoreInterval :=
  (lower upper : ℕ)
  (frequency : ℕ)

/-- The list of score intervals for the test --/
def scoreDistribution : List ScoreInterval :=
  [⟨85, 89, 20⟩, ⟨80, 84, 18⟩, ⟨75, 79, 15⟩, ⟨70, 74, 12⟩,
   ⟨65, 69, 10⟩, ⟨60, 64, 8⟩, ⟨55, 59, 10⟩, ⟨50, 54, 7⟩]

/-- The total number of students --/
def totalStudents : ℕ := 100

/-- Function to calculate the cumulative frequency up to a given interval --/
def cumulativeFrequency (intervals : List ScoreInterval) (targetLower : ℕ) : ℕ :=
  (intervals.filter (fun i => i.lower ≥ targetLower)).foldl (fun acc i => acc + i.frequency) 0

/-- Theorem stating that the median is in the 75-79 interval --/
theorem median_in_75_79_interval :
  ∃ (median : ℕ), 75 ≤ median ∧ median ≤ 79 ∧
  cumulativeFrequency scoreDistribution 75 > totalStudents / 2 ∧
  cumulativeFrequency scoreDistribution 80 ≤ totalStudents / 2 :=
sorry

end NUMINAMATH_CALUDE_median_in_75_79_interval_l1262_126241


namespace NUMINAMATH_CALUDE_total_apples_collected_l1262_126267

-- Define the number of green apples
def green_apples : ℕ := 124

-- Define the number of red apples in terms of green apples
def red_apples : ℕ := 3 * green_apples

-- Define the total number of apples
def total_apples : ℕ := red_apples + green_apples

-- Theorem to prove
theorem total_apples_collected : total_apples = 496 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_collected_l1262_126267


namespace NUMINAMATH_CALUDE_board_transformation_impossibility_l1262_126249

/-- Represents a board state as a list of integers -/
def Board := List Int

/-- Performs one move on the board, pairing integers and replacing with their sum and difference -/
def move (b : Board) : Board :=
  sorry

/-- Checks if a board contains 1000 consecutive integers -/
def isConsecutive1000 (b : Board) : Prop :=
  sorry

/-- Calculates the sum of squares of all integers on the board -/
def sumOfSquares (b : Board) : Int :=
  sorry

theorem board_transformation_impossibility (initial : Board) :
  (sumOfSquares initial) % 8 = 0 →
  ∀ n : Nat, ¬(isConsecutive1000 (n.iterate move initial)) :=
sorry

end NUMINAMATH_CALUDE_board_transformation_impossibility_l1262_126249


namespace NUMINAMATH_CALUDE_parabola_y_intercept_l1262_126294

/-- A parabola passing through two given points has a specific y-intercept -/
theorem parabola_y_intercept (b c : ℝ) : 
  ((-1 : ℝ)^2 + b*(-1) + c = -11) → 
  ((3 : ℝ)^2 + b*3 + c = 17) → 
  c = -7 := by
sorry

end NUMINAMATH_CALUDE_parabola_y_intercept_l1262_126294


namespace NUMINAMATH_CALUDE_checkerboard_coverage_three_by_five_uncoverable_l1262_126257

/-- Represents a checkerboard -/
structure Checkerboard where
  rows : ℕ
  cols : ℕ

/-- A domino covers exactly two squares -/
def domino_size : ℕ := 2

/-- The total number of squares on a checkerboard -/
def total_squares (board : Checkerboard) : ℕ :=
  board.rows * board.cols

/-- A checkerboard can be covered by dominoes if its total squares is even -/
def can_be_covered_by_dominoes (board : Checkerboard) : Prop :=
  total_squares board % domino_size = 0

/-- Theorem: A checkerboard can be covered by dominoes iff its total squares is even -/
theorem checkerboard_coverage (board : Checkerboard) :
  can_be_covered_by_dominoes board ↔ Even (total_squares board) := by sorry

/-- The 3x5 checkerboard cannot be covered by dominoes -/
theorem three_by_five_uncoverable :
  ¬ can_be_covered_by_dominoes ⟨3, 5⟩ := by sorry

end NUMINAMATH_CALUDE_checkerboard_coverage_three_by_five_uncoverable_l1262_126257


namespace NUMINAMATH_CALUDE_least_possible_difference_l1262_126259

theorem least_possible_difference (x y z : ℤ) : 
  Even x → Odd y → Odd z → x < y → y < z → y - x > 3 → 
  ∀ w, (∃ a b c : ℤ, Even a ∧ Odd b ∧ Odd c ∧ a < b ∧ b < c ∧ b - a > 3 ∧ c - a = w) → w ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_difference_l1262_126259


namespace NUMINAMATH_CALUDE_max_area_and_optimal_length_l1262_126252

/-- Represents the dimensions and cost of a simple house. -/
structure SimpleHouse where
  x : ℝ  -- Length of front wall
  y : ℝ  -- Length of side wall
  h : ℝ  -- Height of walls
  colorSteelPrice : ℝ  -- Price per meter of color steel
  compositeSteelPrice : ℝ  -- Price per meter of composite steel
  roofPrice : ℝ  -- Price per square meter of roof material
  maxCost : ℝ  -- Maximum allowed cost

/-- Calculates the total material cost of the house. -/
def materialCost (h : SimpleHouse) : ℝ :=
  2 * h.x * h.colorSteelPrice * h.h + 
  2 * h.y * h.compositeSteelPrice * h.h + 
  h.x * h.y * h.roofPrice

/-- Calculates the area of the house. -/
def area (h : SimpleHouse) : ℝ := h.x * h.y

/-- Theorem stating the maximum area and optimal front wall length. -/
theorem max_area_and_optimal_length (h : SimpleHouse) 
    (h_height : h.h = 2.5)
    (h_colorSteel : h.colorSteelPrice = 450)
    (h_compositeSteel : h.compositeSteelPrice = 200)
    (h_roof : h.roofPrice = 200)
    (h_maxCost : h.maxCost = 32000)
    (h_cost_constraint : materialCost h ≤ h.maxCost) :
    ∃ (maxArea : ℝ) (optimalLength : ℝ),
      maxArea = 100 ∧
      optimalLength = 20 / 3 ∧
      ∀ (x y : ℝ), 
        x > 0 → y > 0 → 
        materialCost { h with x := x, y := y } ≤ h.maxCost →
        area { h with x := x, y := y } ≤ maxArea ∧
        (area { h with x := x, y := y } = maxArea → x = optimalLength) :=
  sorry

end NUMINAMATH_CALUDE_max_area_and_optimal_length_l1262_126252


namespace NUMINAMATH_CALUDE_range_of_a_l1262_126296

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x - a > 0
def q (x : ℝ) : Prop := x > 1

-- Define what it means for p to be a sufficient condition for q
def sufficient (a : ℝ) : Prop := ∀ x, p x a → q x

-- Define what it means for p to be not a necessary condition for q
def not_necessary (a : ℝ) : Prop := ∃ x, q x ∧ ¬(p x a)

-- Theorem statement
theorem range_of_a (a : ℝ) : 
  (sufficient a ∧ not_necessary a) → a > 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1262_126296


namespace NUMINAMATH_CALUDE_quadratic_function_m_not_two_l1262_126237

/-- Given a quadratic function y = a(x-m)^2 where a > 0, 
    if it passes through points (-1,p) and (3,q) where p < q, 
    then m ≠ 2 -/
theorem quadratic_function_m_not_two 
  (a m p q : ℝ) 
  (h1 : a > 0)
  (h2 : a * (-1 - m)^2 = p)
  (h3 : a * (3 - m)^2 = q)
  (h4 : p < q) : 
  m ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_m_not_two_l1262_126237


namespace NUMINAMATH_CALUDE_p_minimum_value_l1262_126275

/-- The quadratic function p in terms of a and b -/
def p (a b : ℝ) : ℝ := 2*a^2 - 8*a*b + 17*b^2 - 16*a - 4*b + 2044

/-- The theorem stating the minimum value of p and the values of a and b at which it occurs -/
theorem p_minimum_value :
  ∃ (a b : ℝ), p a b = 1976 ∧ 
  (∀ (x y : ℝ), p x y ≥ 1976) ∧
  a = 2*b + 4 ∧ b = 2 := by sorry

end NUMINAMATH_CALUDE_p_minimum_value_l1262_126275


namespace NUMINAMATH_CALUDE_sixth_roll_sum_l1262_126208

/-- Represents the sum of numbers on a single die over 6 rolls -/
def single_die_sum : ℕ := 21

/-- Represents the number of dice -/
def num_dice : ℕ := 6

/-- Represents the sums of the top faces for the first 5 rolls -/
def first_five_rolls : List ℕ := [21, 19, 20, 18, 25]

/-- Theorem: The sum of the top faces on the 6th roll is 23 -/
theorem sixth_roll_sum :
  (num_dice * single_die_sum) - (first_five_rolls.sum) = 23 := by
  sorry

end NUMINAMATH_CALUDE_sixth_roll_sum_l1262_126208


namespace NUMINAMATH_CALUDE_complement_of_N_in_M_l1262_126253

def M : Set Nat := {0, 1, 2, 3, 4, 5}
def N : Set Nat := {0, 2, 3}

theorem complement_of_N_in_M :
  M \ N = {1, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_N_in_M_l1262_126253


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l1262_126271

/-- Represents a cricket team with its age-related properties -/
structure CricketTeam where
  total_members : ℕ
  team_average_age : ℝ
  wicket_keeper_age_difference : ℝ
  remaining_players_average_age : ℝ

/-- Theorem stating the difference between the team's average age and the remaining players' average age -/
theorem cricket_team_age_difference (team : CricketTeam)
  (h1 : team.total_members = 11)
  (h2 : team.team_average_age = 28)
  (h3 : team.wicket_keeper_age_difference = 3)
  (h4 : team.remaining_players_average_age = 25) :
  team.team_average_age - team.remaining_players_average_age = 3 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l1262_126271


namespace NUMINAMATH_CALUDE_complement_of_M_l1262_126242

-- Define the universal set U
def U : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}

-- Define the set M
def M : Set ℝ := {1}

-- Theorem statement
theorem complement_of_M (x : ℝ) : x ∈ (U \ M) ↔ 1 < x ∧ x ≤ 5 := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l1262_126242


namespace NUMINAMATH_CALUDE_xiaolins_age_l1262_126264

/-- Represents a person's age as a two-digit number -/
structure TwoDigitAge where
  tens : Nat
  units : Nat
  h_tens : tens < 10
  h_units : units < 10

/-- Swaps the digits of a two-digit age -/
def swapDigits (age : TwoDigitAge) : TwoDigitAge :=
  { tens := age.units,
    units := age.tens,
    h_tens := age.h_units,
    h_units := age.h_tens }

/-- Calculates the numeric value of a two-digit age -/
def toNumber (age : TwoDigitAge) : Nat :=
  10 * age.tens + age.units

theorem xiaolins_age :
  ∀ (grandpa : TwoDigitAge),
    let dad := swapDigits grandpa
    toNumber grandpa - toNumber dad = 5 * 9 →
    9 = 9 := by sorry

end NUMINAMATH_CALUDE_xiaolins_age_l1262_126264


namespace NUMINAMATH_CALUDE_largest_rational_less_than_quarter_rank_3_l1262_126295

-- Define the rank of a rational number
def rank (q : ℚ) : ℕ :=
  -- The definition of rank is given in the problem statement
  sorry

-- Define the property of being the largest rational less than 1/4 with rank 3
def is_largest_less_than_quarter_rank_3 (q : ℚ) : Prop :=
  q < 1/4 ∧ rank q = 3 ∧ ∀ r, r < 1/4 ∧ rank r = 3 → r ≤ q

-- State the theorem
theorem largest_rational_less_than_quarter_rank_3 :
  ∃ q : ℚ, is_largest_less_than_quarter_rank_3 q ∧ q = 1/5 + 1/21 + 1/421 :=
sorry

end NUMINAMATH_CALUDE_largest_rational_less_than_quarter_rank_3_l1262_126295


namespace NUMINAMATH_CALUDE_total_jeans_is_five_l1262_126265

/-- The number of Fox jeans purchased -/
def fox_jeans : ℕ := 3

/-- The number of Pony jeans purchased -/
def pony_jeans : ℕ := 2

/-- The total number of jeans purchased -/
def total_jeans : ℕ := fox_jeans + pony_jeans

theorem total_jeans_is_five : total_jeans = 5 := by sorry

end NUMINAMATH_CALUDE_total_jeans_is_five_l1262_126265


namespace NUMINAMATH_CALUDE_min_value_complex_expression_l1262_126269

theorem min_value_complex_expression (p q r : ℤ) (ξ : ℂ) 
  (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r)
  (h_fourth_root : ξ^4 = 1)
  (h_not_one : ξ ≠ 1) :
  ∃ (m : ℝ), m = Real.sqrt 5 ∧ 
    (∀ (p' q' r' : ℤ) (h_distinct' : p' ≠ q' ∧ q' ≠ r' ∧ p' ≠ r'),
      Complex.abs (p' + q' * ξ + r' * ξ^3) ≥ m) ∧
    (∃ (p' q' r' : ℤ) (h_distinct' : p' ≠ q' ∧ q' ≠ r' ∧ p' ≠ r'),
      Complex.abs (p' + q' * ξ + r' * ξ^3) = m) :=
by sorry

end NUMINAMATH_CALUDE_min_value_complex_expression_l1262_126269


namespace NUMINAMATH_CALUDE_initial_horses_l1262_126263

theorem initial_horses (sheep : ℕ) (chickens : ℕ) (goats : ℕ) (male_animals : ℕ) : 
  sheep = 29 → 
  chickens = 9 → 
  goats = 37 → 
  male_animals = 53 → 
  ∃ (horses : ℕ), 
    horses = 100 ∧ 
    (horses + sheep + chickens) / 2 + goats = male_animals * 2 :=
by sorry

end NUMINAMATH_CALUDE_initial_horses_l1262_126263


namespace NUMINAMATH_CALUDE_circle_radius_zero_circle_equation_implies_zero_radius_l1262_126212

theorem circle_radius_zero (x y : ℝ) :
  x^2 + 8*x + y^2 - 4*y + 20 = 0 → (x + 4)^2 + (y - 2)^2 = 0 := by
  sorry

theorem circle_equation_implies_zero_radius :
  ∃ (x y : ℝ), x^2 + 8*x + y^2 - 4*y + 20 = 0 → 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_zero_circle_equation_implies_zero_radius_l1262_126212


namespace NUMINAMATH_CALUDE_roller_coaster_line_length_l1262_126235

theorem roller_coaster_line_length 
  (num_cars : ℕ) 
  (people_per_car : ℕ) 
  (num_runs : ℕ) 
  (h1 : num_cars = 7)
  (h2 : people_per_car = 2)
  (h3 : num_runs = 6) :
  num_cars * people_per_car * num_runs = 84 :=
by sorry

end NUMINAMATH_CALUDE_roller_coaster_line_length_l1262_126235


namespace NUMINAMATH_CALUDE_first_bank_interest_rate_l1262_126255

/-- Proves that the interest rate of the first bank is 4% given the investment conditions --/
theorem first_bank_interest_rate 
  (total_investment : ℝ)
  (first_bank_investment : ℝ)
  (second_bank_rate : ℝ)
  (total_interest : ℝ)
  (h1 : total_investment = 5000)
  (h2 : first_bank_investment = 1700)
  (h3 : second_bank_rate = 0.065)
  (h4 : total_interest = 282.50)
  : ∃ (first_bank_rate : ℝ), 
    first_bank_rate = 0.04 ∧ 
    first_bank_investment * first_bank_rate + 
    (total_investment - first_bank_investment) * second_bank_rate = 
    total_interest := by
  sorry

end NUMINAMATH_CALUDE_first_bank_interest_rate_l1262_126255


namespace NUMINAMATH_CALUDE_ratio_equality_l1262_126247

theorem ratio_equality (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_diff : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_eq : y / (x + z) = (x + 2*y) / (z + 2*y) ∧ (x + 2*y) / (z + 2*y) = x / (2*y)) :
  x / y = 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l1262_126247


namespace NUMINAMATH_CALUDE_workout_schedule_l1262_126276

theorem workout_schedule (x : ℝ) 
  (h1 : x > 0)  -- Workout duration is positive
  (h2 : x + (x - 2) + 2*x + 2*(x - 2) = 18) :  -- Total workout time is 18 hours
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_workout_schedule_l1262_126276


namespace NUMINAMATH_CALUDE_points_collinear_l1262_126254

/-- Three points A, B, and C in the plane are collinear if there exists a real number k such that 
    vector AC = k * vector AB. -/
def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (C.1 - A.1, C.2 - A.2) = (k * (B.1 - A.1), k * (B.2 - A.2))

/-- The points A(-1, -2), B(2, -1), and C(8, 1) are collinear. -/
theorem points_collinear : collinear (-1, -2) (2, -1) (8, 1) := by
  sorry


end NUMINAMATH_CALUDE_points_collinear_l1262_126254


namespace NUMINAMATH_CALUDE_point_difference_on_line_l1262_126290

/-- Given two points (m, n) and (m + v, n + 18) on the line x = (y / 6) - (2 / 5),
    prove that v = 3 -/
theorem point_difference_on_line (m n : ℝ) :
  (m = n / 6 - 2 / 5) →
  (m + 3 = (n + 18) / 6 - 2 / 5) := by
sorry

end NUMINAMATH_CALUDE_point_difference_on_line_l1262_126290


namespace NUMINAMATH_CALUDE_union_complement_problem_l1262_126228

def U : Finset ℕ := {0, 1, 2, 4, 6, 8}
def M : Finset ℕ := {0, 4, 6}
def N : Finset ℕ := {0, 1, 6}

theorem union_complement_problem : M ∪ (U \ N) = {0, 2, 4, 6, 8} := by sorry

end NUMINAMATH_CALUDE_union_complement_problem_l1262_126228


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1262_126243

theorem geometric_sequence_sixth_term 
  (a : ℝ) -- first term
  (a₇ : ℝ) -- 7th term
  (h₁ : a = 1024)
  (h₂ : a₇ = 16)
  : ∃ r : ℝ, r > 0 ∧ a * r^6 = a₇ ∧ a * r^5 = 32 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1262_126243


namespace NUMINAMATH_CALUDE_rotten_eggs_count_l1262_126280

theorem rotten_eggs_count (total : ℕ) (prob : ℚ) (h_total : total = 36) (h_prob : prob = 47619047619047615 / 10000000000000000) :
  ∃ (rotten : ℕ), rotten = 3 ∧
    (rotten : ℚ) / total * ((rotten : ℚ) - 1) / (total - 1) = prob :=
by sorry

end NUMINAMATH_CALUDE_rotten_eggs_count_l1262_126280


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l1262_126283

/-- The number of ways to place n distinct objects into k distinct containers -/
def placement_count (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 1024 ways to place 5 distinct balls into 4 distinct boxes -/
theorem five_balls_four_boxes : placement_count 5 4 = 1024 := by sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l1262_126283


namespace NUMINAMATH_CALUDE_pool_capacity_l1262_126222

theorem pool_capacity (C : ℝ) 
  (h1 : 0.45 * C + 300 = 0.75 * C) : C = 1000 := by
  sorry

end NUMINAMATH_CALUDE_pool_capacity_l1262_126222


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1262_126277

theorem fixed_point_of_exponential_function (a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a^(4-x) + 3
  f 4 = 4 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1262_126277


namespace NUMINAMATH_CALUDE_range_of_a_l1262_126291

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the theorem
theorem range_of_a (a : ℝ) 
  (h1 : ∀ x, -1 < x ∧ x < 1 → ∃ y, f x = y)  -- f is defined on (-1, 1)
  (h2 : ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y)  -- f is decreasing on (-1, 1)
  (h3 : f (a - 1) > f (2 * a))  -- f(a-1) > f(2a)
  (h4 : -1 < a - 1 ∧ a - 1 < 1)  -- -1 < a-1 < 1
  (h5 : -1 < 2 * a ∧ 2 * a < 1)  -- -1 < 2a < 1
  : 0 < a ∧ a < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1262_126291


namespace NUMINAMATH_CALUDE_exists_complete_gear_rotation_l1262_126226

/-- Represents a gear with a certain number of teeth and some removed teeth -/
structure Gear where
  total_teeth : Nat
  removed_teeth : Finset Nat

/-- Represents the system of two gears -/
structure GearSystem where
  gear1 : Gear
  gear2 : Gear
  rotation : Nat

/-- Checks if a given rotation results in a complete gear -/
def is_complete_gear (gs : GearSystem) : Prop :=
  ∀ i : Nat, i < gs.gear1.total_teeth →
    (i ∉ gs.gear1.removed_teeth ∨ ((i + gs.rotation) % gs.gear1.total_teeth) ∉ gs.gear2.removed_teeth)

/-- The main theorem stating that there exists a rotation forming a complete gear -/
theorem exists_complete_gear_rotation (g1 g2 : Gear)
    (h1 : g1.total_teeth = 14)
    (h2 : g2.total_teeth = 14)
    (h3 : g1.removed_teeth.card = 4)
    (h4 : g2.removed_teeth.card = 4) :
    ∃ r : Nat, is_complete_gear ⟨g1, g2, r⟩ := by
  sorry


end NUMINAMATH_CALUDE_exists_complete_gear_rotation_l1262_126226


namespace NUMINAMATH_CALUDE_BEE_has_largest_value_l1262_126293

def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1
  | 'B' => 2
  | 'C' => 3
  | 'D' => 4
  | 'E' => 5
  | _   => 0

def word_value (w : String) : ℕ :=
  w.toList.map letter_value |>.sum

theorem BEE_has_largest_value :
  let BAD := "BAD"
  let CAB := "CAB"
  let DAD := "DAD"
  let BEE := "BEE"
  let BED := "BED"
  (word_value BEE > word_value BAD) ∧
  (word_value BEE > word_value CAB) ∧
  (word_value BEE > word_value DAD) ∧
  (word_value BEE > word_value BED) := by
  sorry

end NUMINAMATH_CALUDE_BEE_has_largest_value_l1262_126293


namespace NUMINAMATH_CALUDE_regular_polygon_with_18_degree_exterior_angles_has_20_sides_l1262_126274

/-- Theorem: A regular polygon with exterior angles measuring 18 degrees has 20 sides. -/
theorem regular_polygon_with_18_degree_exterior_angles_has_20_sides :
  ∀ n : ℕ,
  n > 0 →
  (360 : ℝ) / n = 18 →
  n = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_18_degree_exterior_angles_has_20_sides_l1262_126274


namespace NUMINAMATH_CALUDE_correct_travel_times_l1262_126216

/-- Represents the tourist's travel information -/
structure TravelInfo where
  boat_distance : ℝ
  walk_distance : ℝ
  walk_time : ℝ
  boat_time : ℝ

/-- Checks if the travel information satisfies the given conditions -/
def satisfies_conditions (info : TravelInfo) : Prop :=
  info.boat_distance = 90 ∧
  info.walk_distance = 10 ∧
  info.boat_time = info.walk_time + 4 ∧
  info.walk_distance * info.boat_time = info.boat_distance * info.walk_time

/-- Theorem stating the correct travel times -/
theorem correct_travel_times (info : TravelInfo) 
  (h : satisfies_conditions info) : 
  info.walk_time = 2 ∧ info.boat_time = 6 := by
  sorry

#check correct_travel_times

end NUMINAMATH_CALUDE_correct_travel_times_l1262_126216


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l1262_126219

theorem cylinder_height_relationship (r₁ h₁ r₂ h₂ : ℝ) :
  r₁ > 0 ∧ h₁ > 0 ∧ r₂ > 0 ∧ h₂ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ := by
sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l1262_126219


namespace NUMINAMATH_CALUDE_derivative_sqrt_l1262_126273

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

theorem derivative_sqrt (x : ℝ) (hx : x > 0) :
  deriv f x = 1 / (2 * Real.sqrt x) := by sorry

end NUMINAMATH_CALUDE_derivative_sqrt_l1262_126273
