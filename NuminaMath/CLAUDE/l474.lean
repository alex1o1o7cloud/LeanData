import Mathlib

namespace NUMINAMATH_CALUDE_spinster_cat_problem_l474_47477

theorem spinster_cat_problem (spinsters cats : ℕ) : 
  (spinsters : ℚ) / cats = 2 / 9 →
  cats = spinsters + 63 →
  spinsters = 18 := by
sorry

end NUMINAMATH_CALUDE_spinster_cat_problem_l474_47477


namespace NUMINAMATH_CALUDE_first_three_selected_students_l474_47497

/-- Represents a random number table --/
def RandomNumberTable := List (List Nat)

/-- The given extract of the random number table --/
def givenTable : RandomNumberTable := [
  [8442, 1753, 3157, 2455, 0688, 7704, 7447, 6721, 7633, 5026, 8392],
  [6301, 5316, 5916, 9275, 3862, 9821, 5071, 7512, 8673, 5807, 4439],
  [1326, 3321, 1342, 7864, 1607, 8252, 0744, 3815, 0324, 4299, 7931]
]

/-- The total number of freshmen --/
def totalFreshmen : Nat := 800

/-- The number of students to be randomly selected --/
def sampleSize : Nat := 100

/-- The starting row in the random number table --/
def startRow : Nat := 8

/-- The starting column in the random number table --/
def startColumn : Nat := 7

/-- Function to select students based on the random number table --/
def selectStudents (table : RandomNumberTable) (total : Nat) (size : Nat) (row : Nat) (col : Nat) : List Nat :=
  sorry -- Implementation not required for the statement

/-- Theorem stating that the first three selected students' serial numbers are 165, 538, and 629 --/
theorem first_three_selected_students :
  let selected := selectStudents givenTable totalFreshmen sampleSize startRow startColumn
  (List.take 3 selected) = [165, 538, 629] := by
  sorry

end NUMINAMATH_CALUDE_first_three_selected_students_l474_47497


namespace NUMINAMATH_CALUDE_water_level_lowered_l474_47485

/-- Proves that removing 4500 gallons of water from a 60ft by 20ft pool lowers the water level by 6 inches -/
theorem water_level_lowered (pool_length pool_width : ℝ) 
  (water_removed : ℝ) (conversion_factor : ℝ) :
  pool_length = 60 →
  pool_width = 20 →
  water_removed = 4500 →
  conversion_factor = 7.5 →
  (water_removed / conversion_factor) / (pool_length * pool_width) * 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_water_level_lowered_l474_47485


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l474_47412

/-- Given a rhombus with side length 51 and shorter diagonal 48, prove that its longer diagonal is 90 -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diagonal : ℝ) (longer_diagonal : ℝ) : 
  side = 51 → shorter_diagonal = 48 → longer_diagonal = 90 → 
  side^2 = (shorter_diagonal / 2)^2 + (longer_diagonal / 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l474_47412


namespace NUMINAMATH_CALUDE_product_equals_one_l474_47431

theorem product_equals_one :
  (∀ a b c : ℝ, a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1)) →
  6 * 15 * 11 = 1 := by
sorry

end NUMINAMATH_CALUDE_product_equals_one_l474_47431


namespace NUMINAMATH_CALUDE_parallelogram_area_triangle_area_l474_47424

-- Define the parallelogram
def parallelogram_base : ℝ := 16
def parallelogram_height : ℝ := 25

-- Define the right-angled triangle
def triangle_side1 : ℝ := 3
def triangle_side2 : ℝ := 4

-- Theorem for parallelogram area
theorem parallelogram_area : 
  parallelogram_base * parallelogram_height = 400 := by sorry

-- Theorem for right-angled triangle area
theorem triangle_area : 
  (triangle_side1 * triangle_side2) / 2 = 6 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_triangle_area_l474_47424


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l474_47461

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_line_through_point 
  (given_line : Line) 
  (p : Point) 
  (h_given : given_line.a = 6 ∧ given_line.b = -5 ∧ given_line.c = 3) 
  (h_point : p.x = 1 ∧ p.y = 1) :
  ∃ (result_line : Line), 
    result_line.a = 6 ∧ 
    result_line.b = -5 ∧ 
    result_line.c = -1 ∧ 
    parallel result_line given_line ∧ 
    pointOnLine p result_line := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l474_47461


namespace NUMINAMATH_CALUDE_pitcher_juice_distribution_l474_47498

theorem pitcher_juice_distribution (C : ℝ) (h : C > 0) :
  let juice_in_pitcher := C / 2
  let cups := 4
  let juice_per_cup := juice_in_pitcher / cups
  juice_per_cup / C * 100 = 12.5 := by
sorry

end NUMINAMATH_CALUDE_pitcher_juice_distribution_l474_47498


namespace NUMINAMATH_CALUDE_detergent_for_nine_pounds_l474_47499

/-- The amount of detergent needed for a given weight of clothes -/
def detergent_needed (rate : ℝ) (weight : ℝ) : ℝ := rate * weight

/-- Theorem: Given a rate of 2 ounces of detergent per pound of clothes,
    the amount of detergent needed for 9 pounds of clothes is 18 ounces -/
theorem detergent_for_nine_pounds :
  detergent_needed 2 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_detergent_for_nine_pounds_l474_47499


namespace NUMINAMATH_CALUDE_division_problem_l474_47408

theorem division_problem (remainder quotient divisor dividend : ℕ) : 
  remainder = 5 →
  divisor = 3 * quotient →
  divisor = 3 * remainder + 3 →
  dividend = divisor * quotient + remainder →
  dividend = 113 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l474_47408


namespace NUMINAMATH_CALUDE_house_problem_l474_47407

theorem house_problem (total garage pool both : ℕ) 
  (h_total : total = 65)
  (h_garage : garage = 50)
  (h_pool : pool = 40)
  (h_both : both = 35) :
  total - (garage + pool - both) = 10 := by
  sorry

end NUMINAMATH_CALUDE_house_problem_l474_47407


namespace NUMINAMATH_CALUDE_victory_chain_exists_l474_47457

/-- Represents a chess player in the tournament -/
structure Player :=
  (id : Nat)

/-- Represents the result of a match between two players -/
inductive MatchResult
  | Win
  | Loss
  | Draw

/-- The chess tournament with 2016 players -/
def Tournament := Fin 2016 → Player

/-- The result of a match between two players -/
def matchResult (p1 p2 : Player) : MatchResult := sorry

/-- Condition: If players A and B tie, then every other player loses to either A or B -/
def tieCondition (t : Tournament) : Prop :=
  ∀ a b : Player, matchResult a b = MatchResult.Draw →
    ∀ c : Player, c ≠ a ∧ c ≠ b →
      matchResult c a = MatchResult.Loss ∨ matchResult c b = MatchResult.Loss

/-- There are at least two draws in the tournament -/
def atLeastTwoDraws (t : Tournament) : Prop :=
  ∃ a b c d : Player, a ≠ b ∧ c ≠ d ∧ matchResult a b = MatchResult.Draw ∧ matchResult c d = MatchResult.Draw

/-- A permutation of players where each player defeats the next -/
def victoryChain (t : Tournament) (p : Fin 2016 → Fin 2016) : Prop :=
  ∀ i : Fin 2015, matchResult (t (p i)) (t (p (i + 1))) = MatchResult.Win

/-- Main theorem: If there are at least two draws and the tie condition holds,
    then there exists a permutation where each player defeats the next -/
theorem victory_chain_exists (t : Tournament)
  (h1 : tieCondition t) (h2 : atLeastTwoDraws t) :
  ∃ p : Fin 2016 → Fin 2016, Function.Bijective p ∧ victoryChain t p := by
  sorry

end NUMINAMATH_CALUDE_victory_chain_exists_l474_47457


namespace NUMINAMATH_CALUDE_non_union_women_percent_is_75_percent_l474_47405

/-- Represents the composition of employees in a company -/
structure CompanyComposition where
  total : ℕ
  men_percent : ℚ
  union_percent : ℚ
  union_men_percent : ℚ

/-- Calculates the percentage of women among non-union employees -/
def non_union_women_percent (c : CompanyComposition) : ℚ :=
  let total_men := c.men_percent * c.total
  let total_women := c.total - total_men
  let union_employees := c.union_percent * c.total
  let union_men := c.union_men_percent * union_employees
  let non_union_men := total_men - union_men
  let non_union_total := c.total - union_employees
  let non_union_women := non_union_total - non_union_men
  non_union_women / non_union_total

/-- Theorem stating that given the company composition, 
    the percentage of women among non-union employees is 75% -/
theorem non_union_women_percent_is_75_percent 
  (c : CompanyComposition) 
  (h1 : c.men_percent = 52/100)
  (h2 : c.union_percent = 60/100)
  (h3 : c.union_men_percent = 70/100) :
  non_union_women_percent c = 75/100 := by
  sorry

end NUMINAMATH_CALUDE_non_union_women_percent_is_75_percent_l474_47405


namespace NUMINAMATH_CALUDE_possible_values_of_a_l474_47442

theorem possible_values_of_a (a : ℝ) 
  (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (non_neg : x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0 ∧ x₅ ≥ 0)
  (eq1 : x₁ + 2*x₂ + 3*x₃ + 4*x₄ + 5*x₅ = a)
  (eq2 : x₁ + 8*x₂ + 27*x₃ + 64*x₄ + 125*x₅ = a^2)
  (eq3 : x₁ + 32*x₂ + 243*x₃ + 1024*x₄ + 3125*x₅ = a^3) :
  a ∈ ({0, 1, 4, 9, 16, 25} : Set ℝ) := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l474_47442


namespace NUMINAMATH_CALUDE_tiffany_monday_bags_l474_47437

/-- The number of bags Tiffany had on Monday -/
def monday_bags : ℕ := sorry

/-- The number of bags Tiffany found on Tuesday -/
def tuesday_bags : ℕ := 3

/-- The number of bags Tiffany found on Wednesday -/
def wednesday_bags : ℕ := 7

/-- The total number of bags Tiffany had -/
def total_bags : ℕ := 20

/-- Theorem stating that Tiffany had 10 bags on Monday -/
theorem tiffany_monday_bags : 
  monday_bags + tuesday_bags + wednesday_bags = total_bags ∧ monday_bags = 10 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_monday_bags_l474_47437


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_is_two_sevenths_l474_47465

/-- The probability of two randomly selected diagonals in a nonagon intersecting inside the nonagon -/
def nonagon_diagonal_intersection_probability : ℚ :=
  2 / 7

/-- The number of sides in a nonagon -/
def nonagon_sides : ℕ := 9

/-- The number of diagonals in a nonagon -/
def nonagon_diagonals : ℕ := nonagon_sides.choose 2 - nonagon_sides

/-- The number of ways to choose two diagonals in a nonagon -/
def diagonal_pairs : ℕ := nonagon_diagonals.choose 2

/-- The number of ways to choose four vertices in a nonagon -/
def four_vertex_selections : ℕ := nonagon_sides.choose 4

theorem nonagon_diagonal_intersection_probability_is_two_sevenths :
  nonagon_diagonal_intersection_probability = four_vertex_selections / diagonal_pairs :=
sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_is_two_sevenths_l474_47465


namespace NUMINAMATH_CALUDE_travel_allowance_percentage_l474_47469

theorem travel_allowance_percentage
  (total_employees : ℕ)
  (salary_increase_percentage : ℚ)
  (no_increase : ℕ)
  (h1 : total_employees = 480)
  (h2 : salary_increase_percentage = 1/10)
  (h3 : no_increase = 336) :
  (total_employees - (salary_increase_percentage * total_employees + no_increase : ℚ)) / total_employees = 1/5 :=
by sorry

end NUMINAMATH_CALUDE_travel_allowance_percentage_l474_47469


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l474_47421

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 - 7 * p - 6 = 0) → 
  (3 * q^2 - 7 * q - 6 = 0) → 
  p ≠ q →
  (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l474_47421


namespace NUMINAMATH_CALUDE_negation_of_existence_l474_47484

theorem negation_of_existence (a : ℝ) :
  (¬ ∃ x : ℝ, x > 1 ∧ x^2 - 2*a*x - 1 < 0) ↔ (∀ x : ℝ, x > 1 → x^2 - 2*a*x - 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_l474_47484


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l474_47422

theorem negative_fraction_comparison : -4/5 < -2/3 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l474_47422


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l474_47423

universe u

def U : Set ℕ := {2, 4, 6, 8, 9}
def A : Set ℕ := {2, 4, 9}

theorem complement_of_A_in_U :
  (U \ A) = {6, 8} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l474_47423


namespace NUMINAMATH_CALUDE_restaurant_bill_proof_l474_47434

theorem restaurant_bill_proof (n : ℕ) (extra : ℝ) (total : ℝ) : 
  n = 10 → 
  extra = 3 → 
  (n - 1) * (total / n + extra) = total → 
  total = 270 := by
sorry

end NUMINAMATH_CALUDE_restaurant_bill_proof_l474_47434


namespace NUMINAMATH_CALUDE_even_integer_solution_l474_47417

-- Define the function h for even integers
def h (n : ℕ) : ℕ :=
  if n % 2 = 0 ∧ n ≥ 2 then
    (n / 2) * (2 + n) / 2
  else
    0

-- Theorem statement
theorem even_integer_solution :
  ∃ x : ℕ, x % 2 = 0 ∧ x ≥ 2 ∧ h 18 / h x = 3 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_even_integer_solution_l474_47417


namespace NUMINAMATH_CALUDE_dice_probability_l474_47428

/-- The number of dice -/
def num_dice : ℕ := 4

/-- The number of sides on each die -/
def sides_per_die : ℕ := 8

/-- The probability of all dice showing the same number -/
def prob_all_same : ℚ := 1 / (sides_per_die ^ (num_dice - 1))

theorem dice_probability :
  prob_all_same = 1 / 512 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l474_47428


namespace NUMINAMATH_CALUDE_two_dress_combinations_l474_47410

def num_colors : Nat := 4
def num_patterns : Nat := 5

theorem two_dress_combinations : 
  (num_colors * num_patterns) * ((num_colors - 1) * (num_patterns - 1)) = 240 := by
  sorry

end NUMINAMATH_CALUDE_two_dress_combinations_l474_47410


namespace NUMINAMATH_CALUDE_cylinder_volume_increase_l474_47418

/-- Proves that tripling the height and increasing the radius by 150% results in a volume increase by a factor of 18.75 -/
theorem cylinder_volume_increase (r h : ℝ) (r_pos : 0 < r) (h_pos : 0 < h) : 
  let new_r := 2.5 * r
  let new_h := 3 * h
  π * new_r^2 * new_h = 18.75 * (π * r^2 * h) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_increase_l474_47418


namespace NUMINAMATH_CALUDE_positive_root_cubic_equation_l474_47445

theorem positive_root_cubic_equation :
  ∃ (x : ℝ), x > 0 ∧ x^3 - 5*x^2 - x - Real.sqrt 3 = 0 ∧ x = 3 + (Real.sqrt 21) / 2 := by
sorry

end NUMINAMATH_CALUDE_positive_root_cubic_equation_l474_47445


namespace NUMINAMATH_CALUDE_bag_counter_problem_l474_47470

theorem bag_counter_problem (Y X : ℕ) : 
  (Y > 0) →  -- Y is positive
  (X > 0) →  -- X is positive
  (Y / (Y + 10) = (Y + 2) / (X + Y + 12)) →  -- Proportion remains unchanged
  (Y * X = 20) →  -- Derived from the equality of proportions
  (Y = 1 ∨ Y = 2 ∨ Y = 4 ∨ Y = 5 ∨ Y = 10 ∨ Y = 20) :=
by sorry

end NUMINAMATH_CALUDE_bag_counter_problem_l474_47470


namespace NUMINAMATH_CALUDE_analysis_method_seeks_sufficient_condition_l474_47449

/-- The type of conditions in logical reasoning -/
inductive ConditionType
  | Necessary
  | Sufficient
  | NecessaryAndSufficient
  | NecessaryOrSufficient

/-- Represents the analysis method for proving inequalities -/
structure AnalysisMethod where
  seekCauseFromResult : ConditionType

/-- The theorem stating that "seeking the cause from the result" in the analysis method
    aims to find a sufficient condition -/
theorem analysis_method_seeks_sufficient_condition :
  ∀ (m : AnalysisMethod), m.seekCauseFromResult = ConditionType.Sufficient := by
  sorry

end NUMINAMATH_CALUDE_analysis_method_seeks_sufficient_condition_l474_47449


namespace NUMINAMATH_CALUDE_isabel_paper_count_l474_47471

/-- The number of pieces of paper Isabel used -/
def used_paper : ℕ := 156

/-- The number of pieces of paper Isabel has left -/
def left_paper : ℕ := 744

/-- The initial number of pieces of paper Isabel bought -/
def initial_paper : ℕ := used_paper + left_paper

theorem isabel_paper_count : initial_paper = 900 := by
  sorry

end NUMINAMATH_CALUDE_isabel_paper_count_l474_47471


namespace NUMINAMATH_CALUDE_min_value_quadratic_l474_47406

theorem min_value_quadratic (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2) :
  ∃ m : ℝ, (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l474_47406


namespace NUMINAMATH_CALUDE_parallel_lines_a_equals_two_l474_47425

/-- Two lines in the form Ax + By + C = 0 are parallel if and only if their slopes are equal -/
def parallel (A1 B1 C1 A2 B2 C2 : ℝ) : Prop :=
  A1 / B1 = A2 / B2

/-- Two lines in the form Ax + By + C = 0 are coincident if they have the same slope and y-intercept -/
def coincident (A1 B1 C1 A2 B2 C2 : ℝ) : Prop :=
  A1 / B1 = A2 / B2 ∧ C1 / B1 = C2 / B2

theorem parallel_lines_a_equals_two (a : ℝ) :
  parallel a (a + 2) 2 1 a (-2) ∧
  ¬coincident a (a + 2) 2 1 a (-2) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_equals_two_l474_47425


namespace NUMINAMATH_CALUDE_no_prime_factor_seven_mod_eight_l474_47436

theorem no_prime_factor_seven_mod_eight (n : ℕ+) :
  ∀ p : ℕ, Prime p → p ∣ (2^(n : ℕ) + 1) → p % 8 ≠ 7 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_factor_seven_mod_eight_l474_47436


namespace NUMINAMATH_CALUDE_lake_pleasant_activities_l474_47433

theorem lake_pleasant_activities (total_kids : ℕ) (tubing_fraction : ℚ) (rafting_fraction : ℚ) (kayaking_fraction : ℚ)
  (h_total : total_kids = 40)
  (h_tubing : tubing_fraction = 1/4)
  (h_rafting : rafting_fraction = 1/2)
  (h_kayaking : kayaking_fraction = 1/3) :
  ⌊(total_kids : ℚ) * tubing_fraction * rafting_fraction * kayaking_fraction⌋ = 1 := by
sorry

end NUMINAMATH_CALUDE_lake_pleasant_activities_l474_47433


namespace NUMINAMATH_CALUDE_jeans_to_shirt_cost_ratio_l474_47453

/-- The ratio of the cost of a pair of jeans to the cost of a shirt is 2:1 -/
theorem jeans_to_shirt_cost_ratio :
  ∀ (jeans_cost : ℚ),
  20 * 10 + 10 * jeans_cost = 400 →
  jeans_cost / 10 = 2 / 1 := by
sorry

end NUMINAMATH_CALUDE_jeans_to_shirt_cost_ratio_l474_47453


namespace NUMINAMATH_CALUDE_hexagon_area_proof_l474_47483

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hexagon -/
structure Hexagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point

/-- Calculates the area of a hexagon -/
def hexagonArea (h : Hexagon) : ℝ := sorry

/-- Checks if a hexagon is equilateral -/
def isEquilateral (h : Hexagon) : Prop := sorry

/-- Checks if lines are parallel -/
def areParallel (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Calculates the angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Checks if y-coordinates are distinct elements of a set -/
def distinctYCoordinates (h : Hexagon) (s : Set ℝ) : Prop := sorry

theorem hexagon_area_proof (h : Hexagon) :
  h.A = ⟨0, 0⟩ →
  h.B = ⟨2 * Real.sqrt 3, 3⟩ →
  h.F = ⟨-7 / 2 * Real.sqrt 3, 5⟩ →
  angle h.F h.A h.B = 150 * π / 180 →
  areParallel h.A h.B h.D h.E →
  areParallel h.B h.C h.E h.F →
  areParallel h.C h.D h.F h.A →
  isEquilateral h →
  distinctYCoordinates h {0, 1, 3, 5, 7, 9} →
  hexagonArea h = 77 / 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_proof_l474_47483


namespace NUMINAMATH_CALUDE_lollipop_sugar_calculation_l474_47456

def chocolate_bars : ℕ := 14
def sugar_per_bar : ℕ := 10
def total_sugar : ℕ := 177

def sugar_in_lollipop : ℕ := total_sugar - (chocolate_bars * sugar_per_bar)

theorem lollipop_sugar_calculation :
  sugar_in_lollipop = 37 := by
  sorry

end NUMINAMATH_CALUDE_lollipop_sugar_calculation_l474_47456


namespace NUMINAMATH_CALUDE_collins_savings_l474_47479

-- Define the constants
def cans_at_home : ℕ := 12
def cans_from_neighbor : ℕ := 46
def cans_from_office : ℕ := 250
def price_per_can : ℚ := 25 / 100

-- Define the functions
def cans_at_grandparents : ℕ := 3 * cans_at_home

def total_cans : ℕ := cans_at_home + cans_at_grandparents + cans_from_neighbor + cans_from_office

def total_money : ℚ := (total_cans : ℚ) * price_per_can

def savings_amount : ℚ := total_money / 2

-- Theorem statement
theorem collins_savings : savings_amount = 43 := by
  sorry

end NUMINAMATH_CALUDE_collins_savings_l474_47479


namespace NUMINAMATH_CALUDE_trig_identity_l474_47455

open Real

theorem trig_identity : 
  sin (150 * π / 180) * cos ((-420) * π / 180) + 
  cos ((-690) * π / 180) * sin (600 * π / 180) + 
  tan (405 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l474_47455


namespace NUMINAMATH_CALUDE_cubic_roots_squared_l474_47489

def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 4

def g (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

theorem cubic_roots_squared (b c d : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) →
  (∀ x : ℝ, f x = 0 → g b c d (x^2) = 0) →
  b = 4 ∧ c = -15 ∧ d = -32 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_squared_l474_47489


namespace NUMINAMATH_CALUDE_range_of_a_l474_47480

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x) 
  (h2 : ∃ x : ℝ, x^2 + 4*x + a = 0) : 
  a ∈ Set.Icc (Real.exp 1) 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l474_47480


namespace NUMINAMATH_CALUDE_circle_equation_is_correct_l474_47448

/-- The circle with center M(2, -1) that is tangent to the line x - 2y + 1 = 0 -/
def tangent_circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 1)^2 = 5}

/-- The line x - 2y + 1 = 0 -/
def tangent_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - 2*p.2 + 1 = 0}

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, -1)

theorem circle_equation_is_correct :
  ∃! (r : ℝ), r > 0 ∧
  (∀ p ∈ tangent_circle, dist p circle_center = r) ∧
  (∃ q ∈ tangent_line, dist q circle_center = r) ∧
  (∀ q ∈ tangent_line, dist q circle_center ≥ r) :=
sorry


end NUMINAMATH_CALUDE_circle_equation_is_correct_l474_47448


namespace NUMINAMATH_CALUDE_quaternary_1320_to_binary_l474_47494

/-- Converts a quaternary (base 4) number to decimal (base 10) --/
def quaternary_to_decimal (q : List Nat) : Nat :=
  q.enum.foldl (λ sum (i, digit) => sum + digit * (4 ^ i)) 0

/-- Converts a decimal (base 10) number to binary (base 2) --/
def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec to_binary_aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else to_binary_aux (m / 2) ((m % 2) :: acc)
  to_binary_aux n []

/-- The main theorem stating that 1320₄ in binary is 1111000₂ --/
theorem quaternary_1320_to_binary :
  decimal_to_binary (quaternary_to_decimal [0, 2, 3, 1]) = [1, 1, 1, 1, 0, 0, 0] := by
  sorry


end NUMINAMATH_CALUDE_quaternary_1320_to_binary_l474_47494


namespace NUMINAMATH_CALUDE_piggy_bank_problem_l474_47439

theorem piggy_bank_problem (total_cents : ℕ) (nickel_quarter_diff : ℕ) 
  (h1 : total_cents = 625)
  (h2 : nickel_quarter_diff = 9) : 
  ∃ (nickels quarters : ℕ),
    nickels = quarters + nickel_quarter_diff ∧
    5 * nickels + 25 * quarters = total_cents ∧
    nickels = 28 := by
  sorry

end NUMINAMATH_CALUDE_piggy_bank_problem_l474_47439


namespace NUMINAMATH_CALUDE_card_drawing_problem_l474_47472

theorem card_drawing_problem (n : Nat) (r y b g : Nat) (total_cards : Nat) (drawn_cards : Nat) : 
  n = 12 → r = 3 → y = 3 → b = 3 → g = 3 → 
  total_cards = n → 
  drawn_cards = 3 → 
  (Nat.choose total_cards drawn_cards) - 
  (4 * (Nat.choose r drawn_cards)) - 
  ((Nat.choose r 2) * (Nat.choose (y + b + g) 1)) = 189 := by
sorry

end NUMINAMATH_CALUDE_card_drawing_problem_l474_47472


namespace NUMINAMATH_CALUDE_factor_a_squared_minus_16_l474_47404

theorem factor_a_squared_minus_16 (a : ℝ) : a^2 - 16 = (a - 4) * (a + 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_a_squared_minus_16_l474_47404


namespace NUMINAMATH_CALUDE_expression_equals_seven_l474_47438

theorem expression_equals_seven (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_prod_nonzero : a * b + a * c + b * c ≠ 0) :
  (a^7 + b^7 + c^7) / (a * b * c * (a * b + a * c + b * c)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_seven_l474_47438


namespace NUMINAMATH_CALUDE_taxi_distance_range_l474_47414

/-- Represents the taxi fare structure and ride details -/
structure TaxiRide where
  baseFare : ℝ  -- Base fare in yuan
  baseDistance : ℝ  -- Distance covered by base fare in km
  additionalFarePerUnit : ℝ  -- Additional fare per unit distance in yuan
  additionalDistanceUnit : ℝ  -- Unit of additional distance in km
  fuelSurcharge : ℝ  -- Fuel surcharge in yuan
  totalFare : ℝ  -- Total fare paid in yuan

/-- Theorem stating the range of possible distances for the given fare structure and total fare -/
theorem taxi_distance_range (ride : TaxiRide)
  (h1 : ride.baseFare = 6)
  (h2 : ride.baseDistance = 2)
  (h3 : ride.additionalFarePerUnit = 1)
  (h4 : ride.additionalDistanceUnit = 0.5)
  (h5 : ride.fuelSurcharge = 1)
  (h6 : ride.totalFare = 9) :
  ∃ x : ℝ, 2.5 < x ∧ x ≤ 3 ∧
    ride.totalFare = ride.baseFare + ((x - ride.baseDistance) / ride.additionalDistanceUnit) * ride.additionalFarePerUnit + ride.fuelSurcharge :=
by sorry


end NUMINAMATH_CALUDE_taxi_distance_range_l474_47414


namespace NUMINAMATH_CALUDE_max_erasable_digits_l474_47493

/-- Represents the number of digits in the original number -/
def total_digits : ℕ := 1000

/-- Represents the sum of digits we want to maintain after erasure -/
def target_sum : ℕ := 2018

/-- Represents the repetitive pattern in the original number -/
def pattern : List ℕ := [2, 0, 1, 8]

/-- Represents the sum of digits in one repetition of the pattern -/
def pattern_sum : ℕ := pattern.sum

/-- Represents the number of complete repetitions of the pattern in the original number -/
def repetitions : ℕ := total_digits / pattern.length

theorem max_erasable_digits : 
  ∃ (erasable : ℕ), 
    erasable = total_digits - (target_sum / pattern_sum * pattern.length + target_sum % pattern_sum) ∧
    erasable = 741 := by sorry

end NUMINAMATH_CALUDE_max_erasable_digits_l474_47493


namespace NUMINAMATH_CALUDE_Z_in_third_quadrant_implies_a_range_l474_47458

def Z (a : ℝ) : ℂ := Complex.mk (a^2 - 2*a) (a^2 - a - 2)

def in_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

theorem Z_in_third_quadrant_implies_a_range (a : ℝ) :
  in_third_quadrant (Z a) → 0 < a ∧ a < 2 := by sorry

end NUMINAMATH_CALUDE_Z_in_third_quadrant_implies_a_range_l474_47458


namespace NUMINAMATH_CALUDE_circle_area_l474_47468

theorem circle_area (x y : ℝ) : 
  (2 * x^2 + 2 * y^2 + 10 * x - 6 * y - 18 = 0) → 
  (∃ (center : ℝ × ℝ) (r : ℝ), 
    ((x - center.1)^2 + (y - center.2)^2 = r^2) ∧ 
    (π * r^2 = 35/2 * π)) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_l474_47468


namespace NUMINAMATH_CALUDE_books_remaining_after_loans_and_returns_l474_47466

/-- Calculates the number of books remaining in a special collection after loans and returns. -/
theorem books_remaining_after_loans_and_returns 
  (initial_books : ℕ) 
  (loaned_books : ℕ) 
  (return_rate : ℚ) : 
  initial_books = 75 → 
  loaned_books = 45 → 
  return_rate = 4/5 → 
  initial_books - loaned_books + (return_rate * loaned_books).floor = 66 := by
  sorry

#check books_remaining_after_loans_and_returns

end NUMINAMATH_CALUDE_books_remaining_after_loans_and_returns_l474_47466


namespace NUMINAMATH_CALUDE_february_greatest_difference_l474_47400

/-- Sales data for trumpet and trombone players -/
structure SalesData where
  trumpet : ℕ
  trombone : ℕ

/-- Calculate percentage difference between two numbers -/
def percentDifference (a b : ℕ) : ℚ :=
  (max a b - min a b : ℚ) / (min a b : ℚ) * 100

/-- Months of the year -/
inductive Month
  | Jan | Feb | Mar | Apr | May

/-- Sales data for each month -/
def monthlySales : Month → SalesData
  | Month.Jan => ⟨6, 4⟩
  | Month.Feb => ⟨27, 5⟩  -- Trumpet sales tripled
  | Month.Mar => ⟨8, 5⟩
  | Month.Apr => ⟨7, 8⟩
  | Month.May => ⟨5, 6⟩

/-- February has the greatest percent difference in sales -/
theorem february_greatest_difference :
  ∀ m : Month, m ≠ Month.Feb →
    percentDifference (monthlySales Month.Feb).trumpet (monthlySales Month.Feb).trombone >
    percentDifference (monthlySales m).trumpet (monthlySales m).trombone :=
by sorry

end NUMINAMATH_CALUDE_february_greatest_difference_l474_47400


namespace NUMINAMATH_CALUDE_function_satisfying_property_is_square_l474_47411

open Real

-- Define the property for the function
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = ⨆ y : ℝ, (2 * x * y - f y)

-- Theorem statement
theorem function_satisfying_property_is_square (f : ℝ → ℝ) :
  SatisfiesProperty f → ∀ x : ℝ, f x = x^2 := by
  sorry


end NUMINAMATH_CALUDE_function_satisfying_property_is_square_l474_47411


namespace NUMINAMATH_CALUDE_fried_chicken_dinner_orders_l474_47440

/-- Represents the number of pieces of chicken used in different order types -/
structure ChickenPieces where
  pasta : Nat
  barbecue : Nat
  friedDinner : Nat

/-- Represents the number of orders for each type -/
structure Orders where
  pasta : Nat
  barbecue : Nat
  friedDinner : Nat

/-- Calculates the total number of chicken pieces used -/
def totalChickenPieces (cp : ChickenPieces) (o : Orders) : Nat :=
  cp.pasta * o.pasta + cp.barbecue * o.barbecue + cp.friedDinner * o.friedDinner

/-- The main theorem to prove -/
theorem fried_chicken_dinner_orders
  (cp : ChickenPieces)
  (o : Orders)
  (h1 : cp.pasta = 2)
  (h2 : cp.barbecue = 3)
  (h3 : cp.friedDinner = 8)
  (h4 : o.pasta = 6)
  (h5 : o.barbecue = 3)
  (h6 : totalChickenPieces cp o = 37) :
  o.friedDinner = 2 := by
  sorry

end NUMINAMATH_CALUDE_fried_chicken_dinner_orders_l474_47440


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l474_47416

def normal_distribution (μ σ : ℝ) : Type := ℝ

def probability {α : Type} (p : Set α) : ℝ := sorry

theorem normal_distribution_probability 
  (ξ : normal_distribution 0 3) : 
  probability {x : ℝ | -3 < x ∧ x < 6} = 0.8185 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l474_47416


namespace NUMINAMATH_CALUDE_total_chapter_difference_is_97_l474_47450

-- Define the structure of a book
structure Book where
  chapter1 : ℕ
  chapter2 : ℕ
  chapter3 : ℕ

-- Define the series of books
def book_series : List Book := [
  { chapter1 := 48, chapter2 := 11, chapter3 := 24 },
  { chapter1 := 35, chapter2 := 18, chapter3 := 28 },
  { chapter1 := 62, chapter2 := 19, chapter3 := 12 }
]

-- Define the function to calculate the difference between first and second chapters
def chapter_difference (book : Book) : ℕ :=
  book.chapter1 - book.chapter2

-- Theorem statement
theorem total_chapter_difference_is_97 :
  (List.map chapter_difference book_series).sum = 97 := by
  sorry

end NUMINAMATH_CALUDE_total_chapter_difference_is_97_l474_47450


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l474_47432

theorem sum_of_a_and_b (a b : ℕ+) (h : 143 * a + 500 * b = 2001) : a + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l474_47432


namespace NUMINAMATH_CALUDE_plains_routes_count_l474_47496

/-- Represents the distribution of cities and routes in a country --/
structure CityNetwork where
  total_cities : Nat
  mountainous_cities : Nat
  plains_cities : Nat
  total_routes : Nat
  mountainous_routes : Nat

/-- Theorem stating the number of routes connecting pairs of plains cities --/
theorem plains_routes_count (network : CityNetwork) 
  (h1 : network.total_cities = 100)
  (h2 : network.mountainous_cities = 30)
  (h3 : network.plains_cities = 70)
  (h4 : network.total_routes = 150)
  (h5 : network.mountainous_routes = 21)
  : Nat := by
  sorry

#check plains_routes_count

end NUMINAMATH_CALUDE_plains_routes_count_l474_47496


namespace NUMINAMATH_CALUDE_mary_remaining_money_l474_47475

def initial_money : ℕ := 58
def pie_cost : ℕ := 6

theorem mary_remaining_money :
  initial_money - pie_cost = 52 := by sorry

end NUMINAMATH_CALUDE_mary_remaining_money_l474_47475


namespace NUMINAMATH_CALUDE_range_of_fraction_l474_47447

theorem range_of_fraction (a b : ℝ) (h1 : 0 < a) (h2 : a ≤ 2) (h3 : b ≥ 1) (h4 : b ≤ a^2) :
  ∃ (t : ℝ), t = b / a ∧ 1/2 ≤ t ∧ t ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_fraction_l474_47447


namespace NUMINAMATH_CALUDE_system_solution_l474_47495

theorem system_solution (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ) 
  (eq1 : a₁ * b₁ + a₂ * b₃ = 1)
  (eq2 : a₁ * b₂ + a₂ * b₄ = 0)
  (eq3 : a₃ * b₁ + a₄ * b₃ = 0)
  (eq4 : a₃ * b₂ + a₄ * b₄ = 1)
  (eq5 : a₂ * b₃ = 7) :
  a₄ * b₄ = -6 := by sorry

end NUMINAMATH_CALUDE_system_solution_l474_47495


namespace NUMINAMATH_CALUDE_ellipse_foci_product_range_l474_47402

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the foci
def leftFocus : ℝ × ℝ := sorry
def rightFocus : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_foci_product_range (p : ℝ × ℝ) :
  ellipse p.1 p.2 →
  3 ≤ (distance p leftFocus) * (distance p rightFocus) ∧
  (distance p leftFocus) * (distance p rightFocus) ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_product_range_l474_47402


namespace NUMINAMATH_CALUDE_quarter_circle_square_perimeter_l474_47403

/-- The perimeter of a region bounded by quarter-circular arcs constructed at each corner of a square with sides measuring 4/π is equal to 8. -/
theorem quarter_circle_square_perimeter :
  let square_side : ℝ := 4 / Real.pi
  let quarter_circle_radius : ℝ := square_side
  let quarter_circle_count : ℕ := 4
  let region_perimeter : ℝ := quarter_circle_count * (Real.pi * quarter_circle_radius / 2)
  region_perimeter = 8 := by
  sorry

end NUMINAMATH_CALUDE_quarter_circle_square_perimeter_l474_47403


namespace NUMINAMATH_CALUDE_no_real_solutions_for_sqrt_equation_l474_47426

theorem no_real_solutions_for_sqrt_equation :
  ¬∃ x : ℝ, Real.sqrt (4 + 2*x) + Real.sqrt (6 + 3*x) + Real.sqrt (8 + 4*x) = 9 + 3*x/2 :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_sqrt_equation_l474_47426


namespace NUMINAMATH_CALUDE_complement_union_A_B_l474_47454

open Set

def U : Set ℝ := univ

def A : Set ℝ := {x | (x - 2) * (x + 1) ≤ 0}

def B : Set ℝ := {x | 0 ≤ x ∧ x < 3}

theorem complement_union_A_B : 
  (Uᶜ ∩ (A ∪ B)ᶜ) = {x | x < -1 ∨ x ≥ 3} := by sorry

end NUMINAMATH_CALUDE_complement_union_A_B_l474_47454


namespace NUMINAMATH_CALUDE_xy_value_l474_47452

theorem xy_value (x y : ℝ) (h : (x - 2)^2 + Real.sqrt (y + 2) = 0) : x * y = -4 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l474_47452


namespace NUMINAMATH_CALUDE_frog_ratio_l474_47409

/-- Given two ponds A and B with frogs, prove the ratio of frogs in A to B -/
theorem frog_ratio (total : ℕ) (pond_a : ℕ) (h1 : total = 48) (h2 : pond_a = 32) :
  (pond_a : ℚ) / ((total - pond_a) : ℚ) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_frog_ratio_l474_47409


namespace NUMINAMATH_CALUDE_line_intersects_x_axis_l474_47435

/-- The line equation 2y - 3x = 15 intersects the x-axis at the point (-5, 0) -/
theorem line_intersects_x_axis :
  ∃ (x : ℝ), 2 * 0 - 3 * x = 15 ∧ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_x_axis_l474_47435


namespace NUMINAMATH_CALUDE_marble_arrangement_theorem_l474_47427

/-- The number of blue marbles -/
def blue_marbles : ℕ := 7

/-- The maximum number of yellow marbles that can be arranged with the blue marbles
    such that the number of marbles with same-color right neighbors equals
    the number with different-color right neighbors -/
def max_yellow_marbles : ℕ := 19

/-- The total number of marbles -/
def total_marbles : ℕ := blue_marbles + max_yellow_marbles

/-- The number of ways to arrange the marbles satisfying the condition -/
def arrangement_count : ℕ := Nat.choose (max_yellow_marbles + blue_marbles + 1) blue_marbles

theorem marble_arrangement_theorem :
  arrangement_count % 1000 = 970 := by
  sorry

end NUMINAMATH_CALUDE_marble_arrangement_theorem_l474_47427


namespace NUMINAMATH_CALUDE_eric_running_time_l474_47451

/-- Given Eric's trip to the park and back, prove the time he ran before jogging. -/
theorem eric_running_time (total_time_to_park : ℕ) (running_time : ℕ) : 
  total_time_to_park = running_time + 10 →
  90 = 3 * total_time_to_park →
  running_time = 20 := by
sorry

end NUMINAMATH_CALUDE_eric_running_time_l474_47451


namespace NUMINAMATH_CALUDE_line_equation_point_slope_l474_47482

/-- The point-slope form of a line with given slope and point. -/
def point_slope_form (k : ℝ) (x₀ y₀ : ℝ) (x y : ℝ) : Prop :=
  y - y₀ = k * (x - x₀)

/-- Theorem: The point-slope form of a line with slope 2 passing through (2, -3) is y + 3 = 2(x - 2). -/
theorem line_equation_point_slope : 
  let k : ℝ := 2
  let x₀ : ℝ := 2
  let y₀ : ℝ := -3
  ∀ x y : ℝ, point_slope_form k x₀ y₀ x y ↔ y + 3 = 2 * (x - 2) :=
sorry

end NUMINAMATH_CALUDE_line_equation_point_slope_l474_47482


namespace NUMINAMATH_CALUDE_conference_languages_l474_47473

/-- The proportion of delegates who know both English and Spanish -/
def both_languages (p_english p_spanish : ℝ) : ℝ :=
  p_english + p_spanish - 1

theorem conference_languages :
  let p_english : ℝ := 0.85
  let p_spanish : ℝ := 0.75
  both_languages p_english p_spanish = 0.60 := by
  sorry

end NUMINAMATH_CALUDE_conference_languages_l474_47473


namespace NUMINAMATH_CALUDE_grapes_problem_l474_47492

theorem grapes_problem (bryce_grapes : ℚ) : 
  (∃ (carter_grapes : ℚ), 
    bryce_grapes = carter_grapes + 7 ∧ 
    carter_grapes = bryce_grapes / 3) → 
  bryce_grapes = 21 / 2 := by
sorry

end NUMINAMATH_CALUDE_grapes_problem_l474_47492


namespace NUMINAMATH_CALUDE_infinitely_many_n_divisible_by_sqrt3_d_l474_47486

def d (n : ℕ+) : ℕ := (Nat.divisors n.val).card

theorem infinitely_many_n_divisible_by_sqrt3_d :
  Set.Infinite {n : ℕ+ | ∃ k : ℕ+, n = k * ⌊Real.sqrt 3 * d n⌋} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_n_divisible_by_sqrt3_d_l474_47486


namespace NUMINAMATH_CALUDE_eight_times_ten_y_plus_fourteen_sin_y_l474_47474

theorem eight_times_ten_y_plus_fourteen_sin_y (y : ℝ) (Q : ℝ) 
  (h : 4 * (5 * y + 7 * Real.sin y) = Q) : 
  8 * (10 * y + 14 * Real.sin y) = 4 * Q := by
  sorry

end NUMINAMATH_CALUDE_eight_times_ten_y_plus_fourteen_sin_y_l474_47474


namespace NUMINAMATH_CALUDE_square_root_problem_l474_47441

theorem square_root_problem (x : ℝ) : (Real.sqrt x / 11 = 4) → x = 1936 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l474_47441


namespace NUMINAMATH_CALUDE_N_subset_M_l474_47478

-- Define set M
def M : Set ℝ := {x | ∃ n : ℤ, x = n / 2 + 1}

-- Define set N
def N : Set ℝ := {y | ∃ m : ℤ, y = m + 1 / 2}

-- Theorem statement
theorem N_subset_M : N ⊆ M := by sorry

end NUMINAMATH_CALUDE_N_subset_M_l474_47478


namespace NUMINAMATH_CALUDE_election_percentage_l474_47487

theorem election_percentage (total_votes : ℕ) (second_candidate_votes : ℕ) :
  total_votes = 1200 →
  second_candidate_votes = 240 →
  (total_votes - second_candidate_votes : ℝ) / total_votes * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_election_percentage_l474_47487


namespace NUMINAMATH_CALUDE_function_equality_l474_47419

theorem function_equality (x : ℝ) (h : x ≠ 0) : x^0 = x/x := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l474_47419


namespace NUMINAMATH_CALUDE_largest_even_digit_multiple_of_5_l474_47462

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

theorem largest_even_digit_multiple_of_5 :
  ∃ (n : ℕ), n = 6880 ∧
  has_only_even_digits n ∧
  n < 8000 ∧
  n % 5 = 0 ∧
  ∀ m : ℕ, has_only_even_digits m → m < 8000 → m % 5 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_even_digit_multiple_of_5_l474_47462


namespace NUMINAMATH_CALUDE_muffin_boxes_l474_47467

theorem muffin_boxes (total_muffins : ℕ) (muffins_per_box : ℕ) (available_boxes : ℕ) : 
  total_muffins = 95 →
  muffins_per_box = 5 →
  available_boxes = 10 →
  (total_muffins - available_boxes * muffins_per_box + muffins_per_box - 1) / muffins_per_box = 9 :=
by sorry

end NUMINAMATH_CALUDE_muffin_boxes_l474_47467


namespace NUMINAMATH_CALUDE_exists_coprime_linear_combination_divisible_l474_47460

theorem exists_coprime_linear_combination_divisible (a b p : ℤ) :
  ∃ k l : ℤ, (Nat.gcd k.natAbs l.natAbs = 1) ∧ (∃ m : ℤ, a * k + b * l = p * m) := by
  sorry

end NUMINAMATH_CALUDE_exists_coprime_linear_combination_divisible_l474_47460


namespace NUMINAMATH_CALUDE_cos_alpha_value_l474_47415

theorem cos_alpha_value (α : Real) (h : Real.sin (α / 2) = 1 / 3) : 
  Real.cos α = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l474_47415


namespace NUMINAMATH_CALUDE_kennedy_house_size_l474_47481

theorem kennedy_house_size (benedict_house_size : ℕ) (kennedy_house_size : ℕ) : 
  benedict_house_size = 2350 →
  kennedy_house_size = 4 * benedict_house_size + 600 →
  kennedy_house_size = 10000 := by
sorry

end NUMINAMATH_CALUDE_kennedy_house_size_l474_47481


namespace NUMINAMATH_CALUDE_curve_asymptotes_sum_l474_47476

/-- A curve with equation y = x / (x^3 + Ax^2 + Bx + C) where A, B, and C are integers -/
structure Curve where
  A : ℤ
  B : ℤ
  C : ℤ

/-- The denominator of the curve equation -/
def Curve.denominator (c : Curve) (x : ℝ) : ℝ :=
  x^3 + c.A * x^2 + c.B * x + c.C

/-- A curve has a vertical asymptote at x = a if its denominator is zero at x = a -/
def has_vertical_asymptote (c : Curve) (a : ℝ) : Prop :=
  c.denominator a = 0

theorem curve_asymptotes_sum (c : Curve) 
  (h1 : has_vertical_asymptote c (-1))
  (h2 : has_vertical_asymptote c 2)
  (h3 : has_vertical_asymptote c 3) :
  c.A + c.B + c.C = -3 := by
  sorry

end NUMINAMATH_CALUDE_curve_asymptotes_sum_l474_47476


namespace NUMINAMATH_CALUDE_opposite_of_fraction_l474_47420

theorem opposite_of_fraction (n : ℕ) (hn : n ≠ 0) :
  ∃ x : ℚ, (1 : ℚ) / n + x = 0 → x = -(1 : ℚ) / n := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_fraction_l474_47420


namespace NUMINAMATH_CALUDE_min_value_zero_l474_47430

/-- The quadratic form representing the expression -/
def Q (x y : ℝ) : ℝ := 5 * x^2 - 8 * x * y + 7 * y^2 - 6 * x - 6 * y + 9

/-- The theorem stating that the minimum value of Q is 0 -/
theorem min_value_zero : 
  ∀ x y : ℝ, Q x y ≥ 0 ∧ ∃ x₀ y₀ : ℝ, Q x₀ y₀ = 0 := by sorry

end NUMINAMATH_CALUDE_min_value_zero_l474_47430


namespace NUMINAMATH_CALUDE_brand_a_most_cost_effective_l474_47491

/-- Represents a chocolate bar brand with its price and s'mores per bar -/
structure ChocolateBar where
  price : ℝ
  smoresPerBar : ℕ

/-- Calculates the cost of chocolate bars for a given number of s'mores -/
def calculateCost (bar : ChocolateBar) (numSmores : ℕ) : ℝ :=
  let numBars := (numSmores + bar.smoresPerBar - 1) / bar.smoresPerBar
  let cost := numBars * bar.price
  if numBars ≥ 10 then cost * 0.85 else cost

/-- Proves that Brand A is the most cost-effective option for Ron's scout camp -/
theorem brand_a_most_cost_effective :
  let numScouts : ℕ := 15
  let smoresPerScout : ℕ := 2
  let brandA := ChocolateBar.mk 1.50 3
  let brandB := ChocolateBar.mk 2.10 4
  let brandC := ChocolateBar.mk 3.00 6
  let totalSmores := numScouts * smoresPerScout
  let costA := calculateCost brandA totalSmores
  let costB := calculateCost brandB totalSmores
  let costC := calculateCost brandC totalSmores
  (costA < costB ∧ costA < costC) ∧ costA = 12.75 := by
  sorry

end NUMINAMATH_CALUDE_brand_a_most_cost_effective_l474_47491


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l474_47429

theorem profit_percentage_calculation 
  (tv_cost dvd_cost selling_price : ℕ) : 
  tv_cost = 16000 → 
  dvd_cost = 6250 → 
  selling_price = 35600 → 
  (selling_price - (tv_cost + dvd_cost)) * 100 / (tv_cost + dvd_cost) = 60 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l474_47429


namespace NUMINAMATH_CALUDE_mangoes_sold_to_market_proof_l474_47459

/-- Calculates the amount of mangoes sold to market given total harvest, mangoes per kilogram, and remaining mangoes -/
def mangoes_sold_to_market (total_harvest : ℕ) (mangoes_per_kg : ℕ) (remaining_mangoes : ℕ) : ℕ :=
  let total_mangoes := total_harvest * mangoes_per_kg
  let sold_mangoes := total_mangoes - remaining_mangoes
  sold_mangoes / 2 / mangoes_per_kg

/-- Theorem stating that given the problem conditions, 20 kilograms of mangoes were sold to market -/
theorem mangoes_sold_to_market_proof :
  mangoes_sold_to_market 60 8 160 = 20 := by
  sorry

end NUMINAMATH_CALUDE_mangoes_sold_to_market_proof_l474_47459


namespace NUMINAMATH_CALUDE_two_digit_penultimate_five_l474_47488

/-- A function that returns the penultimate digit of a natural number -/
def penultimateDigit (n : ℕ) : ℕ :=
  (n / 10) % 10

/-- A predicate that checks if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem two_digit_penultimate_five :
  ∀ x : ℕ, isTwoDigit x →
    (∃ k : ℤ, penultimateDigit (x * k.natAbs) = 5) ↔
    (x = 25 ∨ x = 50 ∨ x = 75) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_penultimate_five_l474_47488


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_3402_l474_47443

def largest_perfect_square_factor (n : ℕ) : ℕ := sorry

theorem largest_perfect_square_factor_3402 :
  largest_perfect_square_factor 3402 = 9 := by sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_3402_l474_47443


namespace NUMINAMATH_CALUDE_chess_grandmaster_time_calculation_l474_47464

theorem chess_grandmaster_time_calculation : 
  let time_learn_rules : ℕ := 2
  let time_get_proficient : ℕ := 49 * time_learn_rules
  let time_become_master : ℕ := 100 * (time_learn_rules + time_get_proficient)
  let total_time : ℕ := time_learn_rules + time_get_proficient + time_become_master
  total_time = 10100 := by
sorry

end NUMINAMATH_CALUDE_chess_grandmaster_time_calculation_l474_47464


namespace NUMINAMATH_CALUDE_power_inequality_l474_47490

theorem power_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 2) : a^b < b^a := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l474_47490


namespace NUMINAMATH_CALUDE_sphere_volume_of_hexagonal_prism_l474_47446

/-- A hexagonal prism with specific properties -/
structure HexagonalPrism where
  -- The base is a regular hexagon
  base_is_regular : Bool
  -- Side edges are perpendicular to the base
  edges_perpendicular : Bool
  -- All vertices lie on the same spherical surface
  vertices_on_sphere : Bool
  -- Volume of the prism
  volume : ℝ
  -- Perimeter of the base
  base_perimeter : ℝ

/-- Theorem stating the volume of the sphere containing the hexagonal prism -/
theorem sphere_volume_of_hexagonal_prism (prism : HexagonalPrism)
    (h1 : prism.base_is_regular = true)
    (h2 : prism.edges_perpendicular = true)
    (h3 : prism.vertices_on_sphere = true)
    (h4 : prism.volume = 9/8)
    (h5 : prism.base_perimeter = 3) :
    ∃ (sphere_volume : ℝ), sphere_volume = 4/3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_of_hexagonal_prism_l474_47446


namespace NUMINAMATH_CALUDE_equal_integers_from_cyclic_equation_l474_47401

theorem equal_integers_from_cyclic_equation 
  (n : ℕ+) (p : ℕ) (a b c : ℤ) 
  (h_prime : Nat.Prime p)
  (h_eq1 : a^(n : ℕ) + p * b = b^(n : ℕ) + p * c)
  (h_eq2 : b^(n : ℕ) + p * c = c^(n : ℕ) + p * a) :
  a = b ∧ b = c := by
sorry

end NUMINAMATH_CALUDE_equal_integers_from_cyclic_equation_l474_47401


namespace NUMINAMATH_CALUDE_polynomials_equal_sum_of_squares_is_954_l474_47463

/-- The original polynomial expression -/
def original_polynomial (x : ℝ) : ℝ := 5 * (x^3 - 3*x^2 + 4) - 8 * (2*x^4 - x^3 + x)

/-- The fully simplified polynomial -/
def simplified_polynomial (x : ℝ) : ℝ := -16*x^4 - 3*x^3 - 15*x^2 + 8*x + 20

/-- Theorem stating that the original and simplified polynomials are equal -/
theorem polynomials_equal : ∀ x : ℝ, original_polynomial x = simplified_polynomial x := by sorry

/-- The sum of squares of coefficients of the simplified polynomial -/
def sum_of_squares_of_coefficients : ℕ := 16^2 + 3^2 + 15^2 + 8^2 + 20^2

/-- Theorem stating that the sum of squares of coefficients is 954 -/
theorem sum_of_squares_is_954 : sum_of_squares_of_coefficients = 954 := by sorry

end NUMINAMATH_CALUDE_polynomials_equal_sum_of_squares_is_954_l474_47463


namespace NUMINAMATH_CALUDE_gardener_expenses_l474_47444

/-- Calculates the total expenses for flowers ordered by the gardener at Parc Municipal -/
theorem gardener_expenses : 
  let tulips := 250
  let carnations := 375
  let roses := 320
  let daffodils := 200
  let lilies := 100
  let tulip_price := 2
  let carnation_price := 1.5
  let rose_price := 3
  let daffodil_price := 1
  let lily_price := 4
  tulips * tulip_price + 
  carnations * carnation_price + 
  roses * rose_price + 
  daffodils * daffodil_price + 
  lilies * lily_price = 2622.5 := by
sorry

end NUMINAMATH_CALUDE_gardener_expenses_l474_47444


namespace NUMINAMATH_CALUDE_min_jugs_proof_l474_47413

/-- The capacity of each jug in ounces -/
def jug_capacity : ℕ := 16

/-- The capacity of the container to be filled in ounces -/
def container_capacity : ℕ := 200

/-- The minimum number of jugs needed to fill or exceed the container capacity -/
def min_jugs : ℕ := 13

theorem min_jugs_proof :
  (∀ n : ℕ, n < min_jugs → n * jug_capacity < container_capacity) ∧
  min_jugs * jug_capacity ≥ container_capacity :=
sorry

end NUMINAMATH_CALUDE_min_jugs_proof_l474_47413
