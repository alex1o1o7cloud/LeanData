import Mathlib

namespace NUMINAMATH_CALUDE_expression_simplification_l3913_391323

theorem expression_simplification (y : ℝ) : 
  3*y - 5*y^2 + 2 + (8 - 5*y + 2*y^2) = -3*y^2 - 2*y + 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3913_391323


namespace NUMINAMATH_CALUDE_determinant_transformation_l3913_391371

theorem determinant_transformation (p q r s : ℝ) :
  Matrix.det !![p, q; r, s] = -3 →
  Matrix.det !![p, 5*p + 2*q; r, 5*r + 2*s] = -6 := by
  sorry

end NUMINAMATH_CALUDE_determinant_transformation_l3913_391371


namespace NUMINAMATH_CALUDE_sum_of_digits_45_40_l3913_391328

def product_45_40 : Nat := 45 * 40

def sum_of_digits (n : Nat) : Nat :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_45_40 : sum_of_digits product_45_40 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_45_40_l3913_391328


namespace NUMINAMATH_CALUDE_max_value_a_l3913_391347

theorem max_value_a (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∀ a : ℝ, a ≤ 1/x + 9/y) → (∃ a : ℝ, a = 16 ∧ ∀ b : ℝ, b ≤ 1/x + 9/y → b ≤ a) :=
by sorry

end NUMINAMATH_CALUDE_max_value_a_l3913_391347


namespace NUMINAMATH_CALUDE_canvas_cost_decrease_canvas_cost_decrease_is_40_l3913_391377

theorem canvas_cost_decrease (paint_decrease : Real) (total_decrease : Real) 
  (paint_canvas_ratio : Real) (canvas_decrease : Real) : Real :=
  if paint_decrease = 60 ∧ 
     total_decrease = 55.99999999999999 ∧ 
     paint_canvas_ratio = 4 ∧ 
     ((1 - paint_decrease / 100) * paint_canvas_ratio + (1 - canvas_decrease / 100)) / 
     (paint_canvas_ratio + 1) = 1 - total_decrease / 100
  then canvas_decrease
  else 0

#check canvas_cost_decrease

theorem canvas_cost_decrease_is_40 :
  canvas_cost_decrease 60 55.99999999999999 4 40 = 40 := by
  sorry

end NUMINAMATH_CALUDE_canvas_cost_decrease_canvas_cost_decrease_is_40_l3913_391377


namespace NUMINAMATH_CALUDE_gcf_three_digit_palindromes_l3913_391310

def three_digit_palindrome (a b : ℕ) : ℕ := 100 * a + 10 * b + a

theorem gcf_three_digit_palindromes :
  ∃ (gcf : ℕ), 
    gcf > 0 ∧
    (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 → gcf ∣ three_digit_palindrome a b) ∧
    (∀ (d : ℕ), d > 0 ∧ (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 → d ∣ three_digit_palindrome a b) → d ≤ gcf) ∧
    gcf = 1 :=
by sorry

end NUMINAMATH_CALUDE_gcf_three_digit_palindromes_l3913_391310


namespace NUMINAMATH_CALUDE_grain_production_scientific_notation_l3913_391329

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) (h : x > 0) : ScientificNotation :=
  sorry

theorem grain_production_scientific_notation :
  toScientificNotation 686530000 (by norm_num) =
    ScientificNotation.mk 6.8653 8 (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_grain_production_scientific_notation_l3913_391329


namespace NUMINAMATH_CALUDE_chloe_trivia_game_score_l3913_391346

/-- Chloe's trivia game score calculation -/
theorem chloe_trivia_game_score (first_round : ℕ) (second_round : ℕ) (final_score : ℕ) 
  (h1 : first_round = 40)
  (h2 : second_round = 50)
  (h3 : final_score = 86) :
  (first_round + second_round) - final_score = 4 := by
  sorry

end NUMINAMATH_CALUDE_chloe_trivia_game_score_l3913_391346


namespace NUMINAMATH_CALUDE_simplify_nested_expression_l3913_391342

theorem simplify_nested_expression (x : ℝ) : 1 - (1 - (1 + (1 - (1 + (1 - x))))) = x := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_expression_l3913_391342


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3913_391376

theorem min_value_of_expression (x y z : ℝ) : 
  (x*y - z)^2 + (x + y + z)^2 ≥ 0 ∧ 
  ∃ (a b c : ℝ), (a*b - c)^2 + (a + b + c)^2 = 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3913_391376


namespace NUMINAMATH_CALUDE_distance_to_school_l3913_391300

/-- Represents the travel conditions to Jeremy's school -/
structure TravelConditions where
  normal_time : ℝ  -- Normal travel time in hours
  fast_time : ℝ    -- Travel time when speed is increased in hours
  slow_time : ℝ    -- Travel time when speed is decreased in hours
  speed_increase : ℝ  -- Speed increase in mph
  speed_decrease : ℝ  -- Speed decrease in mph

/-- Calculates the distance to Jeremy's school given the travel conditions -/
def calculateDistance (tc : TravelConditions) : ℝ :=
  -- Implementation not required for the statement
  sorry

/-- Theorem stating that the distance to Jeremy's school is 15 miles -/
theorem distance_to_school :
  let tc : TravelConditions := {
    normal_time := 1/2,  -- 30 minutes in hours
    fast_time := 3/10,   -- 18 minutes in hours
    slow_time := 2/3,    -- 40 minutes in hours
    speed_increase := 15,
    speed_decrease := 10
  }
  calculateDistance tc = 15 := by
  sorry


end NUMINAMATH_CALUDE_distance_to_school_l3913_391300


namespace NUMINAMATH_CALUDE_number_equation_l3913_391388

theorem number_equation (x : ℝ) : 150 - x = x + 68 ↔ x = 41 := by sorry

end NUMINAMATH_CALUDE_number_equation_l3913_391388


namespace NUMINAMATH_CALUDE_percentage_of_80_equal_to_12_l3913_391321

theorem percentage_of_80_equal_to_12 (p : ℝ) : 
  (p / 100) * 80 = 12 → p = 15 := by sorry

end NUMINAMATH_CALUDE_percentage_of_80_equal_to_12_l3913_391321


namespace NUMINAMATH_CALUDE_complex_number_property_l3913_391373

theorem complex_number_property (b : ℝ) : 
  let z : ℂ := (2 - b * Complex.I) / (1 + 2 * Complex.I)
  (z.re = -z.im) → b = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_property_l3913_391373


namespace NUMINAMATH_CALUDE_reading_assignment_l3913_391383

theorem reading_assignment (total_pages : ℕ) (pages_read : ℕ) (days_left : ℕ) : 
  total_pages = 408 →
  pages_read = 113 →
  days_left = 5 →
  (total_pages - pages_read) / days_left = 59 := by
  sorry

end NUMINAMATH_CALUDE_reading_assignment_l3913_391383


namespace NUMINAMATH_CALUDE_expo_artworks_arrangements_l3913_391392

/-- Represents the number of artworks of each type -/
structure ArtworkCounts where
  calligraphy : Nat
  paintings : Nat
  architectural : Nat

/-- Calculates the number of arrangements for the given artwork counts -/
def arrangeArtworks (counts : ArtworkCounts) : Nat :=
  sorry

/-- The specific artwork counts for the problem -/
def expoArtworks : ArtworkCounts :=
  { calligraphy := 2, paintings := 2, architectural := 1 }

/-- Theorem stating that the number of arrangements for the expo artworks is 36 -/
theorem expo_artworks_arrangements :
  arrangeArtworks expoArtworks = 36 := by
  sorry

end NUMINAMATH_CALUDE_expo_artworks_arrangements_l3913_391392


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l3913_391304

theorem smallest_number_with_remainders : ∃ (x : ℕ), 
  (x % 3 = 2) ∧ (x % 5 = 3) ∧ (x % 7 = 4) ∧
  (∀ y : ℕ, y < x → ¬((y % 3 = 2) ∧ (y % 5 = 3) ∧ (y % 7 = 4))) ∧
  x = 23 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l3913_391304


namespace NUMINAMATH_CALUDE_initial_tickets_l3913_391391

/-- 
Given that a person spends some tickets and has some tickets left,
this theorem proves the total number of tickets they initially had.
-/
theorem initial_tickets (spent : ℕ) (left : ℕ) : 
  spent = 3 → left = 8 → spent + left = 11 := by
  sorry

end NUMINAMATH_CALUDE_initial_tickets_l3913_391391


namespace NUMINAMATH_CALUDE_correct_arrangement_count_l3913_391364

/-- The number of ways to arrange 3 boys and 2 girls in a line, with the girls being adjacent -/
def arrangement_count : ℕ := 48

/-- The number of boys -/
def num_boys : ℕ := 3

/-- The number of girls -/
def num_girls : ℕ := 2

/-- Theorem stating that the number of arrangements is correct -/
theorem correct_arrangement_count : 
  arrangement_count = 
    (Nat.factorial num_boys) * 
    (Nat.choose (num_boys + 1) 1) * 
    (Nat.factorial num_girls) :=
by sorry

end NUMINAMATH_CALUDE_correct_arrangement_count_l3913_391364


namespace NUMINAMATH_CALUDE_dislike_radio_and_music_l3913_391379

theorem dislike_radio_and_music (total_people : ℕ) 
  (radio_dislike_percent : ℚ) (music_dislike_percent : ℚ) :
  total_people = 1500 →
  radio_dislike_percent = 25 / 100 →
  music_dislike_percent = 15 / 100 →
  ⌊(total_people : ℚ) * radio_dislike_percent * music_dislike_percent⌋ = 56 :=
by sorry

end NUMINAMATH_CALUDE_dislike_radio_and_music_l3913_391379


namespace NUMINAMATH_CALUDE_temperature_conversion_l3913_391359

theorem temperature_conversion (t k a : ℝ) : 
  t = 5/9 * (k - 32) + a * k → t = 20 → a = 3 → k = 10.625 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l3913_391359


namespace NUMINAMATH_CALUDE_peters_height_is_96_inches_l3913_391357

/-- Given a tree height, tree shadow length, and Peter's shadow length,
    calculate Peter's height in inches. -/
def peters_height_inches (tree_height foot_to_inch : ℕ) 
                         (tree_shadow peter_shadow : ℚ) : ℚ :=
  (tree_height : ℚ) / tree_shadow * peter_shadow * foot_to_inch

/-- Theorem stating that Peter's height is 96 inches given the problem conditions. -/
theorem peters_height_is_96_inches :
  peters_height_inches 60 12 15 2 = 96 := by
  sorry

#eval peters_height_inches 60 12 15 2

end NUMINAMATH_CALUDE_peters_height_is_96_inches_l3913_391357


namespace NUMINAMATH_CALUDE_hidden_dots_count_l3913_391326

def standard_die_sum : ℕ := 21

def visible_faces : List ℕ := [1, 2, 3, 3, 4, 5, 6, 6, 6]

def total_dice : ℕ := 4

theorem hidden_dots_count :
  (total_dice * standard_die_sum) - (visible_faces.sum) = 48 := by
  sorry

end NUMINAMATH_CALUDE_hidden_dots_count_l3913_391326


namespace NUMINAMATH_CALUDE_percentage_problem_l3913_391324

theorem percentage_problem (N : ℝ) (P : ℝ) : 
  N = 50 → 
  (0.6 * N) = ((P / 100) * 10 + 27) → 
  P = 30 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l3913_391324


namespace NUMINAMATH_CALUDE_S_intersect_T_eq_T_l3913_391338

def S : Set Int := {s | ∃ n : Int, s = 2 * n + 1}
def T : Set Int := {t | ∃ n : Int, t = 4 * n + 1}

theorem S_intersect_T_eq_T : S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_S_intersect_T_eq_T_l3913_391338


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l3913_391343

theorem smallest_number_with_given_remainders :
  ∃ b : ℕ, b ≥ 0 ∧
    b % 6 = 3 ∧
    b % 5 = 2 ∧
    b % 7 = 2 ∧
    (∀ c : ℕ, c ≥ 0 → c % 6 = 3 → c % 5 = 2 → c % 7 = 2 → b ≤ c) ∧
    b = 177 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l3913_391343


namespace NUMINAMATH_CALUDE_unique_solution_is_x_minus_one_l3913_391368

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y + 1) = y + f (f x * f y)

/-- The main theorem stating that the only function satisfying the equation is f(x) = x - 1 -/
theorem unique_solution_is_x_minus_one (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) : 
  ∀ x : ℝ, f x = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_x_minus_one_l3913_391368


namespace NUMINAMATH_CALUDE_charity_savings_interest_l3913_391320

/-- Represents the initial savings amount in dollars -/
def P : ℝ := 2181

/-- Represents the first interest rate (8% per annum) -/
def r1 : ℝ := 0.08

/-- Represents the second interest rate (4% per annum) -/
def r2 : ℝ := 0.04

/-- Represents the time period for each interest rate (3 months = 0.25 years) -/
def t : ℝ := 0.25

/-- Represents the final amount after applying both interest rates -/
def A : ℝ := 2247.50

/-- Theorem stating that the initial amount P results in the final amount A 
    after applying the given interest rates for the specified time periods -/
theorem charity_savings_interest : 
  P * (1 + r1 * t) * (1 + r2 * t) = A := by sorry

end NUMINAMATH_CALUDE_charity_savings_interest_l3913_391320


namespace NUMINAMATH_CALUDE_park_maple_trees_l3913_391311

/-- The number of maple trees in the park after planting -/
def total_maple_trees (initial : ℕ) (planted : ℕ) : ℕ :=
  initial + planted

/-- Theorem: The park will have 11 maple trees after planting -/
theorem park_maple_trees :
  let initial_trees : ℕ := 2
  let trees_to_plant : ℕ := 9
  total_maple_trees initial_trees trees_to_plant = 11 := by
  sorry

end NUMINAMATH_CALUDE_park_maple_trees_l3913_391311


namespace NUMINAMATH_CALUDE_arithmetic_operation_l3913_391367

theorem arithmetic_operation : 5 + 4 - 3 + 2 - 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operation_l3913_391367


namespace NUMINAMATH_CALUDE_f_inequality_l3913_391394

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a - 2)*x - a*Real.log x

theorem f_inequality (x : ℝ) (hx : x > 0) :
  f 3 x ≥ 2*(1 - x) := by sorry

end NUMINAMATH_CALUDE_f_inequality_l3913_391394


namespace NUMINAMATH_CALUDE_remove_all_triangles_no_triangles_remain_l3913_391382

/-- Represents a toothpick figure -/
structure ToothpickFigure where
  total_toothpicks : ℕ
  is_symmetric : Bool
  has_additional_rows : Bool

/-- Represents the number of toothpicks that must be removed to eliminate all triangles -/
def toothpicks_to_remove (figure : ToothpickFigure) : ℕ := 
  if figure.total_toothpicks = 40 ∧ figure.is_symmetric ∧ figure.has_additional_rows
  then 40
  else 0

/-- Theorem stating that for a specific toothpick figure, 40 toothpicks must be removed -/
theorem remove_all_triangles (figure : ToothpickFigure) :
  figure.total_toothpicks = 40 ∧ figure.is_symmetric ∧ figure.has_additional_rows →
  toothpicks_to_remove figure = 40 :=
by
  sorry

/-- Theorem stating that removing 40 toothpicks is sufficient to eliminate all triangles -/
theorem no_triangles_remain (figure : ToothpickFigure) :
  figure.total_toothpicks = 40 ∧ figure.is_symmetric ∧ figure.has_additional_rows →
  toothpicks_to_remove figure = 40 →
  ∀ remaining_triangles, remaining_triangles = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_remove_all_triangles_no_triangles_remain_l3913_391382


namespace NUMINAMATH_CALUDE_perfect_cube_base9_last_digit_l3913_391352

/-- Represents a number in base 9 of the form ab4c -/
structure Base9Number where
  a : ℕ
  b : ℕ
  c : ℕ
  h1 : a ≠ 0
  h2 : c ≤ 8

/-- Converts a Base9Number to its decimal representation -/
def toDecimal (n : Base9Number) : ℕ :=
  729 * n.a + 81 * n.b + 36 + n.c

/-- Predicate to check if a natural number is a perfect cube -/
def isPerfectCube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

theorem perfect_cube_base9_last_digit 
  (n : Base9Number) 
  (h : isPerfectCube (toDecimal n)) : 
  n.c = 1 ∨ n.c = 8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_cube_base9_last_digit_l3913_391352


namespace NUMINAMATH_CALUDE_total_amount_l3913_391353

/-- Represents the division of money among three people -/
structure MoneyDivision where
  x : ℝ  -- X's share
  y : ℝ  -- Y's share
  z : ℝ  -- Z's share

/-- The conditions of the money division problem -/
def problem_conditions (d : MoneyDivision) : Prop :=
  d.y = 0.75 * d.x ∧ 
  d.z = (2/3) * d.x ∧ 
  d.y = 48

/-- The theorem stating the total amount -/
theorem total_amount (d : MoneyDivision) 
  (h : problem_conditions d) : d.x + d.y + d.z = 154.67 := by
  sorry

#check total_amount

end NUMINAMATH_CALUDE_total_amount_l3913_391353


namespace NUMINAMATH_CALUDE_students_liking_sports_l3913_391355

theorem students_liking_sports (B C : Finset Nat) : 
  (B.card = 10) → 
  (C.card = 8) → 
  ((B ∩ C).card = 4) → 
  ((B ∪ C).card = 14) := by
  sorry

end NUMINAMATH_CALUDE_students_liking_sports_l3913_391355


namespace NUMINAMATH_CALUDE_aria_bone_analysis_l3913_391317

/-- The number of hours Aria needs to analyze all bones in a human body. -/
def total_hours : ℕ := 1030

/-- The number of bones in a human body. -/
def total_bones : ℕ := 206

/-- The number of hours spent analyzing each bone. -/
def hours_per_bone : ℚ := total_hours / total_bones

theorem aria_bone_analysis :
  hours_per_bone = 5 := by sorry

end NUMINAMATH_CALUDE_aria_bone_analysis_l3913_391317


namespace NUMINAMATH_CALUDE_simplify_fraction_l3913_391395

theorem simplify_fraction : (54 : ℚ) / 486 = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3913_391395


namespace NUMINAMATH_CALUDE_no_zero_term_l3913_391337

/-- An arithmetic progression is defined by its first term and common difference -/
structure ArithmeticProgression where
  a : ℝ  -- first term
  d : ℝ  -- common difference

/-- The nth term of an arithmetic progression -/
def nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  ap.a + (n - 1 : ℝ) * ap.d

/-- The condition given in the problem -/
def satisfiesCondition (ap : ArithmeticProgression) : Prop :=
  nthTerm ap 5 + nthTerm ap 21 = nthTerm ap 8 + nthTerm ap 15 + nthTerm ap 13

/-- The main theorem -/
theorem no_zero_term (ap : ArithmeticProgression) 
    (h : satisfiesCondition ap) : 
    ¬∃ (n : ℕ), n > 0 ∧ nthTerm ap n = 0 :=
  sorry

end NUMINAMATH_CALUDE_no_zero_term_l3913_391337


namespace NUMINAMATH_CALUDE_polynomial_ratio_condition_l3913_391358

/-- A polynomial f(x) = x^2 - α x + 1 can be expressed as a ratio of two polynomials
    with non-negative coefficients if and only if α < 2. -/
theorem polynomial_ratio_condition (α : ℝ) :
  (∃ (P Q : ℝ → ℝ), (∀ x, P x ≥ 0 ∧ Q x ≥ 0) ∧
    (∀ x, x^2 - α * x + 1 = P x / Q x)) ↔ α < 2 :=
sorry

end NUMINAMATH_CALUDE_polynomial_ratio_condition_l3913_391358


namespace NUMINAMATH_CALUDE_notebooks_last_fifty_days_l3913_391366

/-- The number of days notebooks last given the number of notebooks, pages per notebook, and pages used per day. -/
def notebook_days (num_notebooks : ℕ) (pages_per_notebook : ℕ) (pages_per_day : ℕ) : ℕ :=
  (num_notebooks * pages_per_notebook) / pages_per_day

/-- Theorem stating that 5 notebooks with 40 pages each, used at a rate of 4 pages per day, last for 50 days. -/
theorem notebooks_last_fifty_days :
  notebook_days 5 40 4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_notebooks_last_fifty_days_l3913_391366


namespace NUMINAMATH_CALUDE_percentage_of_wax_used_l3913_391331

def original_wax_20oz : ℕ := 5
def original_wax_5oz : ℕ := 5
def original_wax_1oz : ℕ := 25
def new_candles : ℕ := 3
def new_candle_size : ℕ := 5

def total_original_wax : ℕ := original_wax_20oz * 20 + original_wax_5oz * 5 + original_wax_1oz * 1
def wax_used_for_new_candles : ℕ := new_candles * new_candle_size

theorem percentage_of_wax_used (total_original_wax wax_used_for_new_candles : ℕ) :
  (wax_used_for_new_candles : ℚ) / (total_original_wax : ℚ) * 100 = 10 :=
sorry

end NUMINAMATH_CALUDE_percentage_of_wax_used_l3913_391331


namespace NUMINAMATH_CALUDE_line_intersects_circle_l3913_391302

/-- Theorem: If a point (x₀, y₀) is outside a circle with radius r centered at the origin,
    then the line x₀x + y₀y = r² intersects the circle. -/
theorem line_intersects_circle (x₀ y₀ r : ℝ) (h : x₀^2 + y₀^2 > r^2) :
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ x₀*x + y₀*y = r^2 := by
  sorry

/-- Definition: A point (x₀, y₀) is outside the circle x² + y² = r² -/
def point_outside_circle (x₀ y₀ r : ℝ) : Prop :=
  x₀^2 + y₀^2 > r^2

/-- Definition: The line equation x₀x + y₀y = r² -/
def line_equation (x₀ y₀ r x y : ℝ) : Prop :=
  x₀*x + y₀*y = r^2

/-- Definition: A point (x, y) is on the circle x² + y² = r² -/
def point_on_circle (x y r : ℝ) : Prop :=
  x^2 + y^2 = r^2

end NUMINAMATH_CALUDE_line_intersects_circle_l3913_391302


namespace NUMINAMATH_CALUDE_jumping_contest_l3913_391345

theorem jumping_contest (grasshopper_jump frog_jump mouse_jump : ℕ) : 
  grasshopper_jump = 14 →
  frog_jump = grasshopper_jump + 37 →
  mouse_jump = frog_jump - 16 →
  mouse_jump - grasshopper_jump = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_jumping_contest_l3913_391345


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3913_391350

def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | -Real.sqrt 3 ≤ x ∧ x ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3913_391350


namespace NUMINAMATH_CALUDE_greater_than_theorem_l3913_391322

theorem greater_than_theorem (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : (a - b) * (b - c) * (c - a) > 0) : 
  a > c := by sorry

end NUMINAMATH_CALUDE_greater_than_theorem_l3913_391322


namespace NUMINAMATH_CALUDE_lcm_of_20_45_36_l3913_391340

theorem lcm_of_20_45_36 : Nat.lcm (Nat.lcm 20 45) 36 = 180 := by sorry

end NUMINAMATH_CALUDE_lcm_of_20_45_36_l3913_391340


namespace NUMINAMATH_CALUDE_ratio_of_sums_l3913_391381

theorem ratio_of_sums (p q r u v w : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hu : u > 0) (hv : v > 0) (hw : w > 0)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p*u + q*v + r*w = 56) :
  (p + q + r) / (u + v + w) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_sums_l3913_391381


namespace NUMINAMATH_CALUDE_notebook_distribution_l3913_391356

theorem notebook_distribution (C : ℕ) (N : ℕ) : 
  (N / C = C / 8) → 
  (N = 16 * (C / 2)) → 
  (N = 512) :=
by
  sorry

end NUMINAMATH_CALUDE_notebook_distribution_l3913_391356


namespace NUMINAMATH_CALUDE_inequality_solution_l3913_391363

theorem inequality_solution (x : ℝ) : (2 * x) / 5 ≤ 3 + x ∧ 3 + x < -3 * (1 + x) ↔ x ∈ Set.Ici (-5) ∩ Set.Iio (-3/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3913_391363


namespace NUMINAMATH_CALUDE_g_in_terms_of_f_l3913_391370

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the function g
def g : ℝ → ℝ := sorry

-- State the theorem
theorem g_in_terms_of_f : ∀ x : ℝ, g x = f (6 - x) := by sorry

end NUMINAMATH_CALUDE_g_in_terms_of_f_l3913_391370


namespace NUMINAMATH_CALUDE_x_plus_y_fifth_power_l3913_391305

theorem x_plus_y_fifth_power (x y : ℝ) 
  (sum_eq : x + y = 3)
  (frac_eq : 1 / (x + y^2) + 1 / (x^2 + y) = 1 / 2) :
  x^5 + y^5 = 243 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_fifth_power_l3913_391305


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l3913_391309

theorem angle_sum_around_point (x : ℝ) : 
  150 + 90 + x + 90 = 360 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l3913_391309


namespace NUMINAMATH_CALUDE_two_members_absent_l3913_391325

/-- Represents a trivia team with its total members and game performance -/
structure TriviaTeam where
  totalMembers : Float
  totalPoints : Float
  pointsPerMember : Float

/-- Calculates the number of members who didn't show up for a trivia game -/
def membersAbsent (team : TriviaTeam) : Float :=
  team.totalMembers - (team.totalPoints / team.pointsPerMember)

/-- Theorem stating that for the given trivia team, 2 members didn't show up -/
theorem two_members_absent (team : TriviaTeam) 
  (h1 : team.totalMembers = 5.0)
  (h2 : team.totalPoints = 6.0)
  (h3 : team.pointsPerMember = 2.0) : 
  membersAbsent team = 2 := by
  sorry

#eval membersAbsent { totalMembers := 5.0, totalPoints := 6.0, pointsPerMember := 2.0 }

end NUMINAMATH_CALUDE_two_members_absent_l3913_391325


namespace NUMINAMATH_CALUDE_warehouse_bins_count_l3913_391349

/-- Represents the warehouse with grain storage bins -/
structure Warehouse where
  bins_20ton : ℕ
  bins_15ton : ℕ
  total_capacity : ℕ

/-- The total number of bins in the warehouse -/
def total_bins (w : Warehouse) : ℕ := w.bins_20ton + w.bins_15ton

/-- The total capacity of the warehouse -/
def calculate_capacity (w : Warehouse) : ℕ :=
  w.bins_20ton * 20 + w.bins_15ton * 15

/-- Theorem stating the total number of bins in the warehouse -/
theorem warehouse_bins_count :
  ∃ (w : Warehouse),
    w.bins_20ton = 12 ∧
    calculate_capacity w = 510 ∧
    total_bins w = 30 :=
  sorry


end NUMINAMATH_CALUDE_warehouse_bins_count_l3913_391349


namespace NUMINAMATH_CALUDE_cantor_set_max_operation_l3913_391351

theorem cantor_set_max_operation : 
  ∃ n : ℕ, (∀ k : ℕ, k > n → (2/3 : ℝ)^(k-1) * (1/3) < 1/60) ∧ 
           (2/3 : ℝ)^(n-1) * (1/3) ≥ 1/60 ∧ 
           n = 8 :=
sorry

end NUMINAMATH_CALUDE_cantor_set_max_operation_l3913_391351


namespace NUMINAMATH_CALUDE_rectangle_length_equals_nine_l3913_391315

-- Define the side length of the square
def square_side : ℝ := 6

-- Define the width of the rectangle
def rectangle_width : ℝ := 4

-- Define the area of the square
def square_area : ℝ := square_side * square_side

-- Define the area of the rectangle
def rectangle_area (length : ℝ) : ℝ := length * rectangle_width

-- Theorem statement
theorem rectangle_length_equals_nine :
  ∃ (length : ℝ), rectangle_area length = square_area ∧ length = 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_equals_nine_l3913_391315


namespace NUMINAMATH_CALUDE_complex_number_existence_l3913_391341

theorem complex_number_existence : ∃ (z : ℂ), 
  Complex.abs z = Real.sqrt 7 ∧ 
  z.re < 0 ∧ 
  z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_existence_l3913_391341


namespace NUMINAMATH_CALUDE_average_sleep_time_l3913_391303

def sleep_times : List ℝ := [10, 9, 10, 8, 8]

theorem average_sleep_time :
  (sleep_times.sum / sleep_times.length : ℝ) = 9 := by sorry

end NUMINAMATH_CALUDE_average_sleep_time_l3913_391303


namespace NUMINAMATH_CALUDE_cistern_initial_water_fraction_l3913_391399

theorem cistern_initial_water_fraction 
  (pipe_a_fill_time : ℝ) 
  (pipe_b_fill_time : ℝ) 
  (combined_fill_time : ℝ) 
  (h1 : pipe_a_fill_time = 12) 
  (h2 : pipe_b_fill_time = 8) 
  (h3 : combined_fill_time = 14.4) : 
  ∃ x : ℝ, x = 2/3 ∧ 
    (1 / combined_fill_time = (1 - x) / pipe_a_fill_time + (1 - x) / pipe_b_fill_time) :=
by sorry

end NUMINAMATH_CALUDE_cistern_initial_water_fraction_l3913_391399


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l3913_391348

theorem triangle_abc_proof (a b c : ℝ) (A B C : ℝ) :
  a > c →
  b = 3 →
  (a * c * (1 / 3) = 2) →  -- Equivalent to BA · BC = 2 and cos B = 1/3
  a + c = 5 →              -- From the solution, but derivable from given conditions
  (a = 3 ∧ c = 2) ∧ (Real.cos C = 7 / 9) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l3913_391348


namespace NUMINAMATH_CALUDE_function_transformation_l3913_391361

theorem function_transformation (f : ℝ → ℝ) (h : f 1 = -1) : f (2 - 1) - 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_function_transformation_l3913_391361


namespace NUMINAMATH_CALUDE_sequence_properties_l3913_391318

/-- A geometric sequence with given properties -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ a 4 = 16 ∧ ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (b : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  b 3 = a 3 ∧ b 5 = a 5 ∧ ∀ n : ℕ, b (n + 1) = b n + (b 2 - b 1)

theorem sequence_properties (a : ℕ → ℝ) (b : ℕ → ℝ) 
    (h_geo : geometric_sequence a) (h_arith : arithmetic_sequence b a) :
  (∀ n : ℕ, a n = 2^n) ∧ 
  (∃ k : ℕ, b k = a 9 ∧ k = 45) := by
sorry

end NUMINAMATH_CALUDE_sequence_properties_l3913_391318


namespace NUMINAMATH_CALUDE_cube_root_of_21952_l3913_391365

theorem cube_root_of_21952 : ∃ n : ℕ, n^3 = 21952 ∧ n = 28 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_21952_l3913_391365


namespace NUMINAMATH_CALUDE_extended_segment_endpoint_l3913_391336

/-- Given a segment with endpoints A(-3, 5) and B(9, -1) extended through B to point C,
    where BC = 1/2 * AB, prove that the coordinates of C are (15, -4). -/
theorem extended_segment_endpoint (A B C : ℝ × ℝ) : 
  A = (-3, 5) →
  B = (9, -1) →
  C - B = (1/2 : ℝ) • (B - A) →
  C = (15, -4) := by
  sorry

end NUMINAMATH_CALUDE_extended_segment_endpoint_l3913_391336


namespace NUMINAMATH_CALUDE_starting_lineup_selection_l3913_391306

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The total number of players in the team -/
def total_players : ℕ := 16

/-- The number of quadruplets -/
def num_quadruplets : ℕ := 4

/-- The number of starters to be selected -/
def num_starters : ℕ := 7

/-- The number of quadruplets that must be in the starting lineup -/
def quadruplets_in_lineup : ℕ := 3

/-- The number of ways to select the starting lineup -/
def num_ways : ℕ := 
  binomial num_quadruplets quadruplets_in_lineup * 
  binomial (total_players - num_quadruplets) (num_starters - quadruplets_in_lineup)

theorem starting_lineup_selection :
  num_ways = 1980 := by sorry

end NUMINAMATH_CALUDE_starting_lineup_selection_l3913_391306


namespace NUMINAMATH_CALUDE_problem_statement_l3913_391334

theorem problem_statement (a b : ℕ+) (h : 8 * (a : ℝ)^(a : ℝ) * (b : ℝ)^(b : ℝ) = 27 * (a : ℝ)^(b : ℝ) * (b : ℝ)^(a : ℝ)) : 
  (a : ℝ)^2 + (b : ℝ)^2 = 117 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3913_391334


namespace NUMINAMATH_CALUDE_marble_count_l3913_391389

theorem marble_count (fabian kyle miles : ℕ) 
  (h1 : fabian = 15)
  (h2 : fabian = 3 * kyle)
  (h3 : fabian = 5 * miles) :
  kyle + miles = 8 := by
  sorry

end NUMINAMATH_CALUDE_marble_count_l3913_391389


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3913_391397

theorem cost_price_calculation (selling_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : 
  selling_price = 12000 ∧ 
  discount_rate = 0.1 ∧ 
  profit_rate = 0.08 →
  (selling_price * (1 - discount_rate)) / (1 + profit_rate) = 10000 := by
sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3913_391397


namespace NUMINAMATH_CALUDE_cost_price_from_profit_loss_equality_l3913_391314

/-- The cost price of an article, given specific profit and loss conditions -/
theorem cost_price_from_profit_loss_equality (selling_price_profit selling_price_loss : ℕ) 
  (h : selling_price_profit = 66 ∧ selling_price_loss = 52) :
  ∃ cost_price : ℕ, 
    (selling_price_profit - cost_price = cost_price - selling_price_loss) ∧ 
    cost_price = 59 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_from_profit_loss_equality_l3913_391314


namespace NUMINAMATH_CALUDE_no_integer_roots_l3913_391375

theorem no_integer_roots : ∀ x : ℤ, x^3 - 4*x^2 - 4*x + 24 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l3913_391375


namespace NUMINAMATH_CALUDE_total_shoes_l3913_391369

def shoe_store_problem (brown_shoes : ℕ) (black_shoes : ℕ) : Prop :=
  black_shoes = 2 * brown_shoes ∧
  brown_shoes = 22 ∧
  black_shoes + brown_shoes = 66

theorem total_shoes : ∃ (brown_shoes black_shoes : ℕ), shoe_store_problem brown_shoes black_shoes := by
  sorry

end NUMINAMATH_CALUDE_total_shoes_l3913_391369


namespace NUMINAMATH_CALUDE_expression_evaluation_l3913_391396

theorem expression_evaluation :
  let x : ℚ := 1/2
  let y : ℚ := -2
  ((x + 2*y)^2 - (x + y)*(x - y)) / (2*y) = -4 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3913_391396


namespace NUMINAMATH_CALUDE_school_purchase_options_l3913_391380

theorem school_purchase_options : 
  let valid_purchase := λ (x y : ℕ) => x ≥ 8 ∧ y ≥ 2 ∧ 120 * x + 140 * y ≤ 1500
  ∃! (n : ℕ), ∃ (S : Finset (ℕ × ℕ)), 
    S.card = n ∧ 
    (∀ (p : ℕ × ℕ), p ∈ S ↔ valid_purchase p.1 p.2) ∧
    n = 5 :=
by sorry

end NUMINAMATH_CALUDE_school_purchase_options_l3913_391380


namespace NUMINAMATH_CALUDE_greatest_common_divisor_546_180_under_70_l3913_391344

def is_greatest_common_divisor (n : ℕ) : Prop :=
  n ∣ 546 ∧ n < 70 ∧ n ∣ 180 ∧
  ∀ m : ℕ, m ∣ 546 → m < 70 → m ∣ 180 → m ≤ n

theorem greatest_common_divisor_546_180_under_70 :
  is_greatest_common_divisor 6 := by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_546_180_under_70_l3913_391344


namespace NUMINAMATH_CALUDE_none_always_true_l3913_391335

/-- Given r > 0 and x^2 + y^2 > x^2y^2 for x, y ≠ 0, none of the following statements are true for all x and y -/
theorem none_always_true (r : ℝ) (x y : ℝ) (hr : r > 0) (hxy : x ≠ 0 ∧ y ≠ 0) (h : x^2 + y^2 > x^2 * y^2) :
  ¬(∀ x y : ℝ, -x > -y) ∧
  ¬(∀ x y : ℝ, -x > y) ∧
  ¬(∀ x y : ℝ, 1 > -y/x) ∧
  ¬(∀ x y : ℝ, 1 < x/y) :=
by sorry

end NUMINAMATH_CALUDE_none_always_true_l3913_391335


namespace NUMINAMATH_CALUDE_odd_sum_of_cubes_not_both_even_l3913_391308

theorem odd_sum_of_cubes_not_both_even (n m : ℤ) 
  (h : Odd (n^3 + m^3)) : ¬(Even n ∧ Even m) := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_of_cubes_not_both_even_l3913_391308


namespace NUMINAMATH_CALUDE_largest_x_absolute_value_equation_l3913_391386

theorem largest_x_absolute_value_equation :
  ∀ x : ℝ, |5 - x| = 15 + x → x ≤ -5 ∧ |-5 - 5| = 15 + (-5) := by
  sorry

end NUMINAMATH_CALUDE_largest_x_absolute_value_equation_l3913_391386


namespace NUMINAMATH_CALUDE_ellipse_condition_l3913_391385

-- Define the equation
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (m - 1) + y^2 / (3 - m) = 1

-- Define the condition for an ellipse with foci on the y-axis
def is_ellipse_with_y_foci (m : ℝ) : Prop :=
  m > 1 ∧ m < 3 ∧ (3 - m > m - 1)

-- State the theorem
theorem ellipse_condition (m : ℝ) :
  (1 < m ∧ m < 2) ↔ is_ellipse_with_y_foci m :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l3913_391385


namespace NUMINAMATH_CALUDE_expected_heads_value_l3913_391393

/-- The number of coins --/
def num_coins : ℕ := 80

/-- The probability of a coin showing heads after up to four tosses --/
def prob_heads : ℚ := 15/16

/-- The expected number of coins showing heads after all tosses --/
def expected_heads : ℚ := num_coins * prob_heads

theorem expected_heads_value : expected_heads = 75 := by
  sorry

end NUMINAMATH_CALUDE_expected_heads_value_l3913_391393


namespace NUMINAMATH_CALUDE_polar_line_properties_l3913_391312

/-- A line in polar coordinates passing through (2, π/3) and parallel to the polar axis -/
def polar_line (r θ : ℝ) : Prop :=
  r * Real.sin θ = Real.sqrt 3

theorem polar_line_properties :
  ∀ (r θ : ℝ),
    polar_line r θ →
    (r = 2 ∧ θ = π/3 → polar_line 2 (π/3)) ∧
    (∀ (r' : ℝ), polar_line r' θ → r' * Real.sin θ = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_polar_line_properties_l3913_391312


namespace NUMINAMATH_CALUDE_first_year_after_2020_with_sum_of_digits_10_l3913_391378

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def isFirstYearAfter2020WithSumOfDigits10 (year : ℕ) : Prop :=
  year > 2020 ∧ 
  sumOfDigits year = 10 ∧ 
  ∀ y, 2020 < y ∧ y < year → sumOfDigits y ≠ 10

theorem first_year_after_2020_with_sum_of_digits_10 :
  isFirstYearAfter2020WithSumOfDigits10 2026 := by
  sorry

#eval sumOfDigits 2026  -- Should output 10

end NUMINAMATH_CALUDE_first_year_after_2020_with_sum_of_digits_10_l3913_391378


namespace NUMINAMATH_CALUDE_count_numbers_with_nine_is_1848_l3913_391319

/-- Counts the number of integers between 1000 and 9000 with four distinct digits, including at least one '9' -/
def count_numbers_with_nine : ℕ := 
  let first_digit_nine := 9 * 8 * 7
  let nine_in_other_positions := 3 * 8 * 8 * 7
  first_digit_nine + nine_in_other_positions

/-- Theorem stating that the count of integers between 1000 and 9000 
    with four distinct digits, including at least one '9', is 1848 -/
theorem count_numbers_with_nine_is_1848 : 
  count_numbers_with_nine = 1848 := by sorry

end NUMINAMATH_CALUDE_count_numbers_with_nine_is_1848_l3913_391319


namespace NUMINAMATH_CALUDE_john_completion_time_l3913_391327

/-- Represents the time it takes to complete a task -/
structure TaskTime where
  days : ℝ
  time_positive : days > 0

/-- Represents a person's ability to complete a task -/
structure Worker where
  time_to_complete : TaskTime

/-- Represents two people working together on a task -/
structure TeamWork where
  worker1 : Worker
  worker2 : Worker
  time_to_complete : TaskTime
  jane_leaves_early : ℝ
  jane_leaves_early_positive : jane_leaves_early > 0
  jane_leaves_early_less_than_total : jane_leaves_early < time_to_complete.days

theorem john_completion_time 
  (john : Worker) 
  (jane : Worker) 
  (team : TeamWork) :
  team.worker1 = john →
  team.worker2 = jane →
  jane.time_to_complete.days = 12 →
  team.time_to_complete.days = 10 →
  team.jane_leaves_early = 4 →
  john.time_to_complete.days = 20 := by
  sorry

end NUMINAMATH_CALUDE_john_completion_time_l3913_391327


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l3913_391332

theorem cryptarithm_solution : 
  ∀ y : ℕ, 
    100000 ≤ y ∧ y < 1000000 →
    y * 3 = (y % 100000) * 10 + y / 100000 →
    y = 142857 ∨ y = 285714 :=
by
  sorry

#check cryptarithm_solution

end NUMINAMATH_CALUDE_cryptarithm_solution_l3913_391332


namespace NUMINAMATH_CALUDE_not_divisible_by_n_plus_4_l3913_391330

theorem not_divisible_by_n_plus_4 (n : ℕ+) : ¬ ∃ k : ℤ, (n.val^2 + 8*n.val + 15 : ℤ) = k * (n.val + 4) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_n_plus_4_l3913_391330


namespace NUMINAMATH_CALUDE_red_marbles_fraction_l3913_391360

theorem red_marbles_fraction (total : ℚ) (h : total > 0) : 
  let initial_blue := (2 / 3) * total
  let initial_red := total - initial_blue
  let new_red := 3 * initial_red
  let new_total := initial_blue + new_red
  new_red / new_total = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_red_marbles_fraction_l3913_391360


namespace NUMINAMATH_CALUDE_like_terms_exponents_l3913_391390

theorem like_terms_exponents (a b : ℝ) (m n : ℕ) :
  (∃ (k : ℝ), 3 * a^(2*m) * b^2 = k * (-1/2 * a^2 * b^(n+1))) →
  m + n = 2 := by
sorry

end NUMINAMATH_CALUDE_like_terms_exponents_l3913_391390


namespace NUMINAMATH_CALUDE_workshop_prize_difference_l3913_391387

theorem workshop_prize_difference (total : ℕ) (wolf : ℕ) (both : ℕ) (nobel : ℕ) 
  (h_total : total = 50)
  (h_wolf : wolf = 31)
  (h_both : both = 14)
  (h_nobel : nobel = 25)
  (h_wolf_less : wolf ≤ total)
  (h_both_less : both ≤ wolf)
  (h_both_less_nobel : both ≤ nobel)
  (h_nobel_less : nobel ≤ total) :
  let non_wolf := total - wolf
  let nobel_non_wolf := nobel - both
  let non_nobel_non_wolf := non_wolf - nobel_non_wolf
  nobel_non_wolf - non_nobel_non_wolf = 3 := by
  sorry

end NUMINAMATH_CALUDE_workshop_prize_difference_l3913_391387


namespace NUMINAMATH_CALUDE_min_dials_for_same_remainder_l3913_391372

/-- A dial is a regular 12-sided polygon with numbers from 1 to 12 -/
def Dial := Fin 12 → Fin 12

/-- A stack of dials -/
def Stack := ℕ → Dial

/-- The sum of numbers in a column of the stack -/
def columnSum (s : Stack) (col : Fin 12) (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => (s i col).val + 1)

/-- Whether all column sums have the same remainder modulo 12 -/
def allColumnsSameRemainder (s : Stack) (n : ℕ) : Prop :=
  ∀ (c₁ c₂ : Fin 12), columnSum s c₁ n % 12 = columnSum s c₂ n % 12

/-- The theorem stating that 12 is the minimum number of dials required -/
theorem min_dials_for_same_remainder :
  ∀ (s : Stack), (∃ (n : ℕ), allColumnsSameRemainder s n) →
  (∃ (m : ℕ), m = 12 ∧ allColumnsSameRemainder s m ∧
    ∀ (k : ℕ), k < m → ¬allColumnsSameRemainder s k) :=
sorry

end NUMINAMATH_CALUDE_min_dials_for_same_remainder_l3913_391372


namespace NUMINAMATH_CALUDE_bobby_candy_consumption_l3913_391301

theorem bobby_candy_consumption (initial : ℕ) (next_day : ℕ) :
  initial = 89 → next_day = 152 → initial + next_day = 241 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_consumption_l3913_391301


namespace NUMINAMATH_CALUDE_cookie_is_circle_with_radius_sqrt35_l3913_391362

-- Define the equation of the cookie's boundary
def cookie_boundary (x y : ℝ) : Prop :=
  x^2 + y^2 + 10 = 6*x + 12*y

-- Theorem stating that the cookie's boundary is a circle with radius √35
theorem cookie_is_circle_with_radius_sqrt35 :
  ∃ (h k : ℝ), ∀ (x y : ℝ),
    cookie_boundary x y ↔ (x - h)^2 + (y - k)^2 = 35 :=
sorry

end NUMINAMATH_CALUDE_cookie_is_circle_with_radius_sqrt35_l3913_391362


namespace NUMINAMATH_CALUDE_intersection_M_N_l3913_391374

open Set Real

def M : Set ℝ := {x | Real.exp (x - 1) > 1}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem intersection_M_N : M ∩ N = Ioo 1 3 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3913_391374


namespace NUMINAMATH_CALUDE_number_equation_l3913_391384

theorem number_equation : ∃ x : ℝ, x / 1500 = 0.016833333333333332 ∧ x = 25.25 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l3913_391384


namespace NUMINAMATH_CALUDE_valid_selections_count_l3913_391316

/-- Represents a 6x6 grid of blocks -/
def Grid := Fin 6 → Fin 6 → Bool

/-- Represents a selection of 4 blocks on the grid -/
def Selection := Fin 4 → (Fin 6 × Fin 6)

/-- Checks if a selection forms an L shape -/
def is_L_shape (s : Selection) : Prop := sorry

/-- Checks if no two blocks in the selection share a row or column -/
def no_shared_row_col (s : Selection) : Prop := sorry

/-- The number of valid selections -/
def num_valid_selections : ℕ := sorry

theorem valid_selections_count :
  num_valid_selections = 1800 := by sorry

end NUMINAMATH_CALUDE_valid_selections_count_l3913_391316


namespace NUMINAMATH_CALUDE_additional_distance_for_target_average_speed_l3913_391339

/-- Proves that given an initial trip of 20 miles at 40 mph, an additional 90 miles
    driven at 60 mph will result in an average speed of 55 mph for the entire trip. -/
theorem additional_distance_for_target_average_speed
  (initial_distance : ℝ) (initial_speed : ℝ) (second_speed : ℝ) (target_avg_speed : ℝ)
  (additional_distance : ℝ) :
  initial_distance = 20 →
  initial_speed = 40 →
  second_speed = 60 →
  target_avg_speed = 55 →
  additional_distance = 90 →
  (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / second_speed)) = target_avg_speed :=
by sorry

end NUMINAMATH_CALUDE_additional_distance_for_target_average_speed_l3913_391339


namespace NUMINAMATH_CALUDE_orange_harvest_existence_l3913_391398

theorem orange_harvest_existence :
  ∃ (A B C D : ℕ), A + B + C + D = 56 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_existence_l3913_391398


namespace NUMINAMATH_CALUDE_grocery_store_price_l3913_391307

/-- The price of a bulk warehouse deal for sparkling water -/
def bulk_price : ℚ := 12

/-- The number of cans in the bulk warehouse deal -/
def bulk_cans : ℕ := 48

/-- The additional cost per can at the grocery store compared to the bulk warehouse -/
def additional_cost : ℚ := 1/4

/-- The number of cans in the grocery store deal -/
def grocery_cans : ℕ := 12

/-- The price of the grocery store deal for sparkling water -/
def grocery_price : ℚ := 6

theorem grocery_store_price :
  grocery_price = (bulk_price / bulk_cans + additional_cost) * grocery_cans :=
by sorry

end NUMINAMATH_CALUDE_grocery_store_price_l3913_391307


namespace NUMINAMATH_CALUDE_final_dog_count_l3913_391333

/-- Calculates the number of dogs remaining in the rescue center at the end of the month -/
def dogsRemaining (initial : ℕ) (arrivals : List ℕ) (adoptions : List ℕ) (returned : ℕ) : ℕ :=
  let weeklyChanges := List.zipWith (λ a b => a - b) arrivals adoptions
  initial + weeklyChanges.sum - returned

theorem final_dog_count :
  let initial : ℕ := 200
  let arrivals : List ℕ := [30, 40, 30]
  let adoptions : List ℕ := [40, 50, 30, 70]
  let returned : ℕ := 20
  dogsRemaining initial arrivals adoptions returned = 90 := by
  sorry

end NUMINAMATH_CALUDE_final_dog_count_l3913_391333


namespace NUMINAMATH_CALUDE_min_value_n_over_m_l3913_391354

theorem min_value_n_over_m (m n : ℝ) :
  (∀ x : ℝ, Real.exp x - m * x + n - 1 ≥ 0) →
  (∃ k : ℝ, k = n / m ∧ k ≥ 0 ∧ ∀ j : ℝ, (∀ x : ℝ, Real.exp x - m * x + j * m - 1 ≥ 0) → j ≥ k) :=
by sorry

end NUMINAMATH_CALUDE_min_value_n_over_m_l3913_391354


namespace NUMINAMATH_CALUDE_fraction_equality_l3913_391313

theorem fraction_equality (a b : ℝ) (x : ℝ) (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) :
  (a + 2*b) / (a - 2*b) = (x + 2) / (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3913_391313
