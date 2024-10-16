import Mathlib

namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l1778_177801

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (∀ x, x < 1 → x < 2) ∧ (∃ x, x < 2 ∧ ¬(x < 1)) := by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l1778_177801


namespace NUMINAMATH_CALUDE_missing_figure_proof_l1778_177893

theorem missing_figure_proof : ∃ x : ℝ, (0.25 / 100) * x = 0.04 ∧ x = 16 := by sorry

end NUMINAMATH_CALUDE_missing_figure_proof_l1778_177893


namespace NUMINAMATH_CALUDE_natalia_clip_sales_l1778_177866

/-- The total number of clips Natalia sold in April and May -/
def total_clips (april_sales : ℕ) (may_sales : ℕ) : ℕ := april_sales + may_sales

/-- Theorem stating the total number of clips sold given the conditions -/
theorem natalia_clip_sales : 
  ∀ (april_sales : ℕ), 
  april_sales = 48 → 
  total_clips april_sales (april_sales / 2) = 72 := by
sorry

end NUMINAMATH_CALUDE_natalia_clip_sales_l1778_177866


namespace NUMINAMATH_CALUDE_binary_110011_equals_51_l1778_177861

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110011_equals_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_equals_51_l1778_177861


namespace NUMINAMATH_CALUDE_max_value_location_l1778_177873

theorem max_value_location (f : ℝ → ℝ) (a b : ℝ) (h : a < b) :
  Differentiable ℝ f → ∃ x ∈ Set.Icc a b,
    (∀ y ∈ Set.Icc a b, f y ≤ f x) ∧
    (x = a ∨ x = b ∨ deriv f x = 0) :=
sorry

end NUMINAMATH_CALUDE_max_value_location_l1778_177873


namespace NUMINAMATH_CALUDE_fruit_stand_problem_l1778_177835

/-- Fruit stand problem -/
theorem fruit_stand_problem (apple_price : ℕ) (initial_total : ℕ) (initial_avg : ℕ) (final_avg : ℕ) :
  apple_price = 40 →
  initial_total = 10 →
  initial_avg = 48 →
  final_avg = 45 →
  ∃ (orange_price : ℕ) (apples oranges : ℕ),
    apples + oranges = initial_total ∧
    apple_price * apples + orange_price * oranges = initial_avg * initial_total ∧
    apple_price * apples + orange_price * (oranges - 2) = final_avg * (initial_total - 2) ∧
    orange_price = 60 :=
by sorry

end NUMINAMATH_CALUDE_fruit_stand_problem_l1778_177835


namespace NUMINAMATH_CALUDE_representations_equivalence_distinct_representations_equivalence_l1778_177860

/-- The number of ways to represent a positive integer as a sum of positive integers -/
def numRepresentations (n m : ℕ+) : ℕ :=
  sorry

/-- The number of ways to represent a positive integer as a sum of distinct positive integers -/
def numDistinctRepresentations (n m : ℕ+) : ℕ :=
  sorry

/-- The number of ways to represent a positive integer as a sum of integers from a given set -/
def numRepresentationsFromSet (n : ℕ) (s : Finset ℕ) : ℕ :=
  sorry

theorem representations_equivalence (n m : ℕ+) :
  numRepresentations n m = numRepresentationsFromSet (n - m) (Finset.range m) :=
sorry

theorem distinct_representations_equivalence (n m : ℕ+) :
  numDistinctRepresentations n m = numRepresentationsFromSet (n - m * (m + 1) / 2) (Finset.range n) :=
sorry

end NUMINAMATH_CALUDE_representations_equivalence_distinct_representations_equivalence_l1778_177860


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1778_177887

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 * b + b^2 * c + c^2 * a = 3) :
  a^7 * b + b^7 * c + c^7 * a + a * b^3 + b * c^3 + c * a^3 ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1778_177887


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_l1778_177826

theorem min_sum_reciprocals (w x y z : ℝ) 
  (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : w + x + y + z = 1) :
  1/w + 1/x + 1/y + 1/z ≥ 16 ∧
  (1/w + 1/x + 1/y + 1/z = 16 ↔ w = 1/4 ∧ x = 1/4 ∧ y = 1/4 ∧ z = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_reciprocals_l1778_177826


namespace NUMINAMATH_CALUDE_third_year_sample_size_l1778_177812

/-- Calculates the number of students to be sampled from a specific grade in a stratified sampling. -/
def stratified_sample_size (total_population : ℕ) (grade_population : ℕ) (total_sample : ℕ) : ℕ :=
  (grade_population * total_sample) / total_population

theorem third_year_sample_size :
  let total_population : ℕ := 3000
  let third_year_population : ℕ := 1200
  let total_sample : ℕ := 50
  stratified_sample_size total_population third_year_population total_sample = 20 := by
  sorry

end NUMINAMATH_CALUDE_third_year_sample_size_l1778_177812


namespace NUMINAMATH_CALUDE_circle_diameter_l1778_177881

theorem circle_diameter (A : ℝ) (r : ℝ) (D : ℝ) : 
  A = 100 * Real.pi → A = Real.pi * r^2 → D = 2 * r → D = 20 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l1778_177881


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1778_177879

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - x + m = 0 ∧ 
   ∀ y : ℝ, 2 * y^2 - y + m = 0 → y = x) → 
  m = 1/8 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1778_177879


namespace NUMINAMATH_CALUDE_g_range_l1778_177842

/-- The function f(x) = 2x^2 + 3x - 2 -/
def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x - 2

/-- The function g(x) = f(f(x)) -/
def g (x : ℝ) : ℝ := f (f x)

/-- The domain of g -/
def g_domain : Set ℝ := { x | -1 ≤ x ∧ x ≤ 2 }

theorem g_range :
  ∀ y ∈ g '' g_domain, -2 ≤ y ∧ y ≤ 424 :=
sorry

end NUMINAMATH_CALUDE_g_range_l1778_177842


namespace NUMINAMATH_CALUDE_max_removable_marbles_l1778_177822

/-- Represents the number of marbles of each color in the bag -/
structure MarbleCount where
  yellow : Nat
  red : Nat
  black : Nat

/-- The initial number of marbles in the bag -/
def initialMarbles : MarbleCount := ⟨8, 7, 5⟩

/-- The condition that must be satisfied after removing marbles -/
def satisfiesCondition (mc : MarbleCount) : Prop :=
  (mc.yellow ≥ 4 ∧ (mc.red ≥ 3 ∨ mc.black ≥ 3)) ∨
  (mc.red ≥ 4 ∧ (mc.yellow ≥ 3 ∨ mc.black ≥ 3)) ∨
  (mc.black ≥ 4 ∧ (mc.yellow ≥ 3 ∨ mc.red ≥ 3))

/-- The maximum number of marbles that can be removed -/
def maxRemovable : Nat := 7

theorem max_removable_marbles :
  (∀ (removed : Nat), removed ≤ maxRemovable →
    ∀ (remaining : MarbleCount),
      remaining.yellow + remaining.red + remaining.black = initialMarbles.yellow + initialMarbles.red + initialMarbles.black - removed →
      satisfiesCondition remaining) ∧
  (∀ (removed : Nat), removed > maxRemovable →
    ∃ (remaining : MarbleCount),
      remaining.yellow + remaining.red + remaining.black = initialMarbles.yellow + initialMarbles.red + initialMarbles.black - removed ∧
      ¬satisfiesCondition remaining) := by
  sorry

end NUMINAMATH_CALUDE_max_removable_marbles_l1778_177822


namespace NUMINAMATH_CALUDE_joshua_oranges_expenditure_l1778_177894

/-- The amount Joshua spent on buying oranges -/
def joshua_spent (num_oranges : ℕ) (selling_price profit : ℚ) : ℚ :=
  (num_oranges : ℚ) * (selling_price - profit)

/-- Theorem stating the amount Joshua spent on oranges -/
theorem joshua_oranges_expenditure :
  joshua_spent 25 0.60 0.10 = 12.50 := by
  sorry

end NUMINAMATH_CALUDE_joshua_oranges_expenditure_l1778_177894


namespace NUMINAMATH_CALUDE_lineup_count_l1778_177813

/-- The number of team members -/
def team_size : ℕ := 16

/-- The number of positions in the lineup -/
def lineup_size : ℕ := 5

/-- The number of pre-assigned positions -/
def pre_assigned : ℕ := 2

/-- Calculate the number of ways to choose a lineup -/
def lineup_ways : ℕ :=
  (team_size - pre_assigned) * (team_size - pre_assigned - 1) * (team_size - pre_assigned - 2)

theorem lineup_count : lineup_ways = 2184 := by sorry

end NUMINAMATH_CALUDE_lineup_count_l1778_177813


namespace NUMINAMATH_CALUDE_expression_simplification_l1778_177849

theorem expression_simplification (a : ℝ) (h1 : a ≠ -1) (h2 : a ≠ 2) :
  (a - 3*a/(a+1)) / ((a^2 - 4*a + 4)/(a+1)) = a / (a-2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1778_177849


namespace NUMINAMATH_CALUDE_shirt_count_proof_l1778_177847

/-- The number of different colored neckties -/
def num_neckties : ℕ := 6

/-- The probability that all boxes contain matching necktie-shirt pairs -/
def matching_probability : ℝ := 0.041666666666666664

/-- The number of different colored shirts -/
def num_shirts : ℕ := 2

theorem shirt_count_proof :
  (1 / num_shirts : ℝ) ^ num_neckties = matching_probability ∧
  num_shirts = ⌈(1 / matching_probability) ^ (1 / num_neckties : ℝ)⌉ := by
  sorry

#check shirt_count_proof

end NUMINAMATH_CALUDE_shirt_count_proof_l1778_177847


namespace NUMINAMATH_CALUDE_equations_not_equivalent_l1778_177850

/-- The solution set of the equation 2√(x+5) = x+2 -/
def SolutionSet1 : Set ℝ :=
  {x : ℝ | 2 * Real.sqrt (x + 5) = x + 2}

/-- The solution set of the equation 4(x+5) = (x+2)² -/
def SolutionSet2 : Set ℝ :=
  {x : ℝ | 4 * (x + 5) = (x + 2)^2}

/-- Theorem stating that the equations are not equivalent -/
theorem equations_not_equivalent : SolutionSet1 ≠ SolutionSet2 := by
  sorry

#check equations_not_equivalent

end NUMINAMATH_CALUDE_equations_not_equivalent_l1778_177850


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1778_177854

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

def condition (a : ℕ → ℝ) : Prop :=
  ∀ n, |a (n + 1)| > a n

theorem condition_necessary_not_sufficient (a : ℕ → ℝ) :
  (is_increasing a → condition a) ∧ ¬(condition a → is_increasing a) :=
sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1778_177854


namespace NUMINAMATH_CALUDE_min_value_sin_function_l1778_177846

theorem min_value_sin_function (x : Real) (h : x ∈ Set.Ioo 0 (Real.pi / 2)) :
  (2 * Real.sin x ^ 2 + 1) / Real.sin (2 * x) ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sin_function_l1778_177846


namespace NUMINAMATH_CALUDE_equation_solutions_l1778_177832

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (x₁^2 + 2*x₁ = 0 ∧ x₂^2 + 2*x₂ = 0) ∧ x₁ = 0 ∧ x₂ = -2) ∧
  (∃ x₁ x₂ : ℝ, (2*x₁^2 - 6*x₁ = 3 ∧ 2*x₂^2 - 6*x₂ = 3) ∧ 
    x₁ = (3 + Real.sqrt 15) / 2 ∧ x₂ = (3 - Real.sqrt 15) / 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1778_177832


namespace NUMINAMATH_CALUDE_square_sum_greater_than_quarter_l1778_177858

theorem square_sum_greater_than_quarter (a b : ℝ) (h : a + b = 1) :
  a^2 + b^2 > 1/4 := by
sorry

end NUMINAMATH_CALUDE_square_sum_greater_than_quarter_l1778_177858


namespace NUMINAMATH_CALUDE_cubic_roots_reciprocal_sum_squares_l1778_177817

theorem cubic_roots_reciprocal_sum_squares (a b c d r s t : ℝ) : 
  a ≠ 0 → d ≠ 0 → 
  (∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = r ∨ x = s ∨ x = t) →
  1/r^2 + 1/s^2 + 1/t^2 = (b^2 - 2*a*c) / d^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_reciprocal_sum_squares_l1778_177817


namespace NUMINAMATH_CALUDE_log_equation_l1778_177820

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation : log10 2 * log10 50 + log10 25 - log10 5 * log10 20 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_l1778_177820


namespace NUMINAMATH_CALUDE_equal_pairs_l1778_177892

theorem equal_pairs (x y z : ℝ) (h : xy + z = yz + x ∧ yz + x = zx + y) :
  x = y ∨ y = z ∨ z = x := by
  sorry

end NUMINAMATH_CALUDE_equal_pairs_l1778_177892


namespace NUMINAMATH_CALUDE_correct_number_of_selections_l1778_177853

/-- The number of ways to choose k elements from a set of n elements -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 5 players from a team of 16 players, 
    including a set of twins, where both twins cannot be chosen together -/
def choosePlayersWithTwinRestriction : ℕ := sorry

theorem correct_number_of_selections :
  choosePlayersWithTwinRestriction = 4004 := by sorry

end NUMINAMATH_CALUDE_correct_number_of_selections_l1778_177853


namespace NUMINAMATH_CALUDE_win_trip_l1778_177878

/-- The number of chocolate bars Tom needs to sell to win the trip -/
def total_bars : ℕ := 3465

/-- The number of chocolate bars in each box -/
def bars_per_box : ℕ := 7

/-- The number of boxes Tom needs to sell to win the trip -/
def boxes_needed : ℕ := total_bars / bars_per_box

theorem win_trip : boxes_needed = 495 := by
  sorry

end NUMINAMATH_CALUDE_win_trip_l1778_177878


namespace NUMINAMATH_CALUDE_algebraic_identities_l1778_177889

theorem algebraic_identities (a b : ℝ) : 
  ((-a)^2 * (a^2)^2 / a^3 = a^3) ∧ 
  ((a + b) * (a - b) - (a - b)^2 = 2*a*b - 2*b^2) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identities_l1778_177889


namespace NUMINAMATH_CALUDE_weeks_passed_l1778_177851

/-- Represents the number of weeks passed -/
def weeks : ℕ := sorry

/-- Weekly fixed allowance in dollars -/
def fixed_allowance : ℚ := 20

/-- Extra earning per chore in dollars -/
def extra_chore_earning : ℚ := 1.5

/-- Average number of extra chores per week -/
def avg_extra_chores : ℕ := 15

/-- Total money earned in dollars -/
def total_money : ℚ := 425

/-- Theorem stating that the number of weeks passed is 10 -/
theorem weeks_passed : weeks = 10 := by sorry

end NUMINAMATH_CALUDE_weeks_passed_l1778_177851


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1778_177865

theorem quadratic_roots_relation (b c : ℝ) : 
  (∃ r s : ℝ, (2 * r^2 - 4 * r - 8 = 0) ∧ 
               (2 * s^2 - 4 * s - 8 = 0) ∧ 
               ((r + 3)^2 + b * (r + 3) + c = 0) ∧ 
               ((s + 3)^2 + b * (s + 3) + c = 0)) →
  c = 11 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1778_177865


namespace NUMINAMATH_CALUDE_cost_split_theorem_l1778_177818

/-- Calculates the amount each person should pay when a group buys items and splits the cost equally -/
def calculate_cost_per_person (num_people : ℕ) (item1_count : ℕ) (item1_price : ℕ) (item2_count : ℕ) (item2_price : ℕ) : ℕ :=
  ((item1_count * item1_price + item2_count * item2_price) / num_people)

/-- Proves that when 4 friends buy 5 items at 200 won each and 7 items at 800 won each, 
    and divide the total cost equally, each person should pay 1650 won -/
theorem cost_split_theorem : 
  calculate_cost_per_person 4 5 200 7 800 = 1650 := by
  sorry

end NUMINAMATH_CALUDE_cost_split_theorem_l1778_177818


namespace NUMINAMATH_CALUDE_distinct_values_count_l1778_177898

def expression := 3^3^3^3

def parenthesization1 := 3^(3^(3^3))
def parenthesization2 := 3^((3^3)^3)
def parenthesization3 := ((3^3)^3)^3
def parenthesization4 := (3^(3^3))^3
def parenthesization5 := (3^3)^(3^3)

def distinct_values : Finset ℕ := {parenthesization1, parenthesization2, parenthesization3, parenthesization4, parenthesization5}

theorem distinct_values_count :
  Finset.card distinct_values = 3 :=
sorry

end NUMINAMATH_CALUDE_distinct_values_count_l1778_177898


namespace NUMINAMATH_CALUDE_hockey_season_length_l1778_177828

theorem hockey_season_length 
  (games_per_month : ℕ) 
  (total_games : ℕ) 
  (h1 : games_per_month = 13) 
  (h2 : total_games = 182) : 
  total_games / games_per_month = 14 := by
sorry

end NUMINAMATH_CALUDE_hockey_season_length_l1778_177828


namespace NUMINAMATH_CALUDE_dereks_savings_l1778_177882

theorem dereks_savings (n : ℕ) (a : ℝ) (r : ℝ) : 
  n = 12 → a = 2 → r = 2 → 
  a * (1 - r^n) / (1 - r) = 8190 := by
  sorry

end NUMINAMATH_CALUDE_dereks_savings_l1778_177882


namespace NUMINAMATH_CALUDE_new_student_weight_l1778_177834

theorem new_student_weight (n : ℕ) (w_avg_before w_avg_after w_new : ℝ) :
  n = 19 →
  w_avg_before = 15 →
  w_avg_after = 14.8 →
  w_avg_after * (n + 1) = w_avg_before * n + w_new →
  w_new = 11 := by
  sorry

end NUMINAMATH_CALUDE_new_student_weight_l1778_177834


namespace NUMINAMATH_CALUDE_gcd_of_lcm_and_ratio_l1778_177848

theorem gcd_of_lcm_and_ratio (X Y : ℕ) (h1 : Nat.lcm X Y = 180) (h2 : X * 5 = Y * 4) :
  Nat.gcd X Y = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_lcm_and_ratio_l1778_177848


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1778_177885

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), y = -2*x ∧ y^2/a^2 - x^2/b^2 = 1) →
  (∃ (x y : ℝ), x^2 = 4*Real.sqrt 10*y ∧ x^2 + y^2 = a^2 - b^2) →
  a^2 = 8 ∧ b^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1778_177885


namespace NUMINAMATH_CALUDE_roots_of_unity_cubic_equation_l1778_177870

theorem roots_of_unity_cubic_equation :
  ∃ (c d : ℤ), ∃ (roots : Finset ℂ),
    (∀ z ∈ roots, z^3 = 1) ∧
    (∀ z ∈ roots, z^3 + c*z + d = 0) ∧
    (roots.card = 3) ∧
    (∀ z : ℂ, z^3 = 1 → z^3 + c*z + d = 0 → z ∈ roots) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_unity_cubic_equation_l1778_177870


namespace NUMINAMATH_CALUDE_distance_between_points_l1778_177886

/-- The distance between points (2, 2) and (-1, -1) is 3√2 -/
theorem distance_between_points : Real.sqrt ((2 - (-1))^2 + (2 - (-1))^2) = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1778_177886


namespace NUMINAMATH_CALUDE_total_consumption_theorem_l1778_177843

/-- Represents the amount of liquid consumed by each person --/
structure Consumption where
  elijah : Float
  emilio : Float
  isabella : Float
  xavier_soda : Float
  xavier_fruit_punch : Float

/-- Converts pints to cups --/
def pints_to_cups (pints : Float) : Float := pints * 2

/-- Converts liters to cups --/
def liters_to_cups (liters : Float) : Float := liters * 4.22675

/-- Converts gallons to cups --/
def gallons_to_cups (gallons : Float) : Float := gallons * 16

/-- Calculates the total cups consumed based on the given consumption --/
def total_cups (c : Consumption) : Float :=
  c.elijah + c.emilio + c.isabella + c.xavier_soda + c.xavier_fruit_punch

/-- Theorem stating that the total cups consumed is equal to 80.68025 --/
theorem total_consumption_theorem (c : Consumption)
  (h1 : c.elijah = pints_to_cups 8.5)
  (h2 : c.emilio = pints_to_cups 9.5)
  (h3 : c.isabella = liters_to_cups 3)
  (h4 : c.xavier_soda = gallons_to_cups 2 * 0.6)
  (h5 : c.xavier_fruit_punch = gallons_to_cups 2 * 0.4) :
  total_cups c = 80.68025 := by
  sorry


end NUMINAMATH_CALUDE_total_consumption_theorem_l1778_177843


namespace NUMINAMATH_CALUDE_andrew_donuts_problem_l1778_177839

theorem andrew_donuts_problem (monday tuesday wednesday : ℕ) : 
  tuesday = monday / 2 →
  wednesday = 4 * monday →
  monday + tuesday + wednesday = 49 →
  monday = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_andrew_donuts_problem_l1778_177839


namespace NUMINAMATH_CALUDE_replacement_concentration_l1778_177841

/-- Represents a salt solution with a given concentration -/
structure SaltSolution where
  concentration : ℝ
  concentration_nonneg : 0 ≤ concentration
  concentration_le_one : concentration ≤ 1

/-- The result of mixing two salt solutions -/
def mix_solutions (s1 s2 : SaltSolution) (ratio : ℝ) : SaltSolution where
  concentration := s1.concentration * (1 - ratio) + s2.concentration * ratio
  concentration_nonneg := sorry
  concentration_le_one := sorry

theorem replacement_concentration 
  (original second : SaltSolution)
  (h1 : original.concentration = 0.14)
  (h2 : (mix_solutions original second 0.25).concentration = 0.16) :
  second.concentration = 0.22 := by
  sorry

end NUMINAMATH_CALUDE_replacement_concentration_l1778_177841


namespace NUMINAMATH_CALUDE_average_grade_year_before_l1778_177845

/-- Calculates the average grade for the year before last given the following conditions:
  * The student took 6 courses last year with an average grade of 100 points
  * The student took 5 courses the year before
  * The average grade for the entire two-year period was 86 points
-/
theorem average_grade_year_before (courses_last_year : Nat) (avg_grade_last_year : ℝ)
  (courses_year_before : Nat) (avg_grade_two_years : ℝ) :
  courses_last_year = 6 →
  avg_grade_last_year = 100 →
  courses_year_before = 5 →
  avg_grade_two_years = 86 →
  (courses_year_before * avg_grade_year_before + courses_last_year * avg_grade_last_year) /
    (courses_year_before + courses_last_year) = avg_grade_two_years →
  avg_grade_year_before = 69.2 :=
by
  sorry

#check average_grade_year_before

end NUMINAMATH_CALUDE_average_grade_year_before_l1778_177845


namespace NUMINAMATH_CALUDE_rectangle_area_proof_l1778_177811

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The original rectangle --/
def original : Rectangle := { length := 5, width := 7 }

/-- The rectangle after shortening one side --/
def shortened : Rectangle := { length := 3, width := 7 }

theorem rectangle_area_proof :
  area original = 35 ∧
  area shortened = 21 →
  area { length := 5, width := 5 } = 25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_proof_l1778_177811


namespace NUMINAMATH_CALUDE_range_of_a_inequality_l1778_177806

theorem range_of_a_inequality (a : ℝ) : 
  (∃ x : ℝ, |a| ≥ |x + 1| + |x - 2|) ↔ a ∈ Set.Iic 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_inequality_l1778_177806


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1778_177809

theorem complex_fraction_equality : Complex.I * Complex.I = -1 → (7 - Complex.I) / (3 + Complex.I) = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1778_177809


namespace NUMINAMATH_CALUDE_website_earnings_per_visit_l1778_177831

/-- Calculates the earnings per visit for a website -/
def earnings_per_visit (monthly_visits : ℕ) (daily_earnings : ℚ) : ℚ :=
  (30 * daily_earnings) / monthly_visits

/-- Theorem: Given 30,000 monthly visits and $10 daily earnings, the earnings per visit is $0.01 -/
theorem website_earnings_per_visit : 
  earnings_per_visit 30000 10 = 1/100 := by
  sorry

end NUMINAMATH_CALUDE_website_earnings_per_visit_l1778_177831


namespace NUMINAMATH_CALUDE_fraction_most_compliant_l1778_177859

/-- Represents the compliance of an algebraic expression with standard notation -/
inductive AlgebraicCompliance
  | Compliant
  | NonCompliant

/-- Evaluates the compliance of a mixed number with variable expression -/
def mixedNumberWithVariable (n : ℕ) (m : ℕ) (d : ℕ) (x : String) : AlgebraicCompliance :=
  AlgebraicCompliance.NonCompliant

/-- Evaluates the compliance of a fraction expression -/
def fraction (n : String) (d : String) : AlgebraicCompliance :=
  AlgebraicCompliance.Compliant

/-- Evaluates the compliance of an expression with an attached unit -/
def expressionWithUnit (expr : String) (unit : String) : AlgebraicCompliance :=
  AlgebraicCompliance.NonCompliant

/-- Evaluates the compliance of a multiplication expression -/
def multiplicationExpression (x : String) (n : ℕ) : AlgebraicCompliance :=
  AlgebraicCompliance.NonCompliant

/-- Theorem stating that fraction (b/a) is the most compliant with standard algebraic notation -/
theorem fraction_most_compliant :
  fraction "b" "a" = AlgebraicCompliance.Compliant ∧
  mixedNumberWithVariable 1 1 2 "a" = AlgebraicCompliance.NonCompliant ∧
  expressionWithUnit "3a-1" "个" = AlgebraicCompliance.NonCompliant ∧
  multiplicationExpression "a" 3 = AlgebraicCompliance.NonCompliant :=
by sorry

end NUMINAMATH_CALUDE_fraction_most_compliant_l1778_177859


namespace NUMINAMATH_CALUDE_staircase_ratio_proof_l1778_177805

theorem staircase_ratio_proof (steps_first : ℕ) (step_height : ℚ) (total_height : ℚ) 
  (h1 : steps_first = 20)
  (h2 : step_height = 1/2)
  (h3 : total_height = 45) :
  ∃ (r : ℚ), 
    r * steps_first = (total_height / step_height - steps_first - (r * steps_first - 10)) ∧ 
    r = 2 := by
  sorry

end NUMINAMATH_CALUDE_staircase_ratio_proof_l1778_177805


namespace NUMINAMATH_CALUDE_binomial_12_choose_6_l1778_177804

theorem binomial_12_choose_6 : Nat.choose 12 6 = 1848 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_6_l1778_177804


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1778_177833

theorem cyclic_sum_inequality (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 1 →
  (a^6 / ((a - b) * (a - c))) + (b^6 / ((b - c) * (b - a))) + (c^6 / ((c - a) * (c - b))) > 15 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1778_177833


namespace NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l1778_177825

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Define the concept of axis of symmetry
def AxisOfSymmetry (f : ℝ → ℝ) (a : ℝ) : Prop := 
  ∀ x : ℝ, f (a + x) = f (a - x)

theorem symmetry_of_shifted_even_function (f : ℝ → ℝ) (h : EvenFunction f) :
  AxisOfSymmetry (fun x ↦ f (x + 1)) (-1) := by
sorry

end NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l1778_177825


namespace NUMINAMATH_CALUDE_least_x_divisible_by_three_l1778_177880

theorem least_x_divisible_by_three : 
  ∃ (x : ℕ), x < 10 ∧ 
  (∀ (y : ℕ), y < x → ¬(23 * 100 + y * 10 + 57) % 3 = 0) ∧
  (23 * 100 + x * 10 + 57) % 3 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_x_divisible_by_three_l1778_177880


namespace NUMINAMATH_CALUDE_token_game_1994_token_game_1991_l1778_177869

/-- Represents the state of the token-passing game -/
structure GameState (N : ℕ) where
  tokens : Fin N → ℕ
  total_tokens : ℕ

/-- Defines a single move in the game -/
def move (state : GameState N) (i : Fin N) : GameState N :=
  sorry

/-- Determines if the game has terminated -/
def is_terminated (state : GameState N) : Prop :=
  ∀ i, state.tokens i ≤ 1

/-- Theorem for the token-passing game with 1994 girls -/
theorem token_game_1994 (n : ℕ) :
  (n < 1994 → ∃ (final_state : GameState 1994), is_terminated final_state) ∧
  (n = 1994 → ¬∃ (final_state : GameState 1994), is_terminated final_state) :=
  sorry

/-- Theorem for the token-passing game with 1991 girls -/
theorem token_game_1991 (n : ℕ) :
  n ≤ 1991 → ¬∃ (final_state : GameState 1991), is_terminated final_state :=
  sorry

end NUMINAMATH_CALUDE_token_game_1994_token_game_1991_l1778_177869


namespace NUMINAMATH_CALUDE_max_profit_at_max_price_verify_conditions_l1778_177823

-- Define the linear relationship between price and sales volume
def sales_volume (x : ℝ) : ℝ := -10 * x + 450

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - 15) * (sales_volume x)

-- Define the maximum allowed price
def max_price : ℝ := 28

-- Theorem statement
theorem max_profit_at_max_price :
  (∀ x, x ≤ max_price → profit x ≤ profit max_price) ∧
  profit max_price = 2210 := by
  sorry

-- Verify the conditions given in the problem
theorem verify_conditions :
  sales_volume 20 = 250 ∧
  profit 25 = 2000 ∧
  (∃ k b, ∀ x, sales_volume x = k * x + b) := by
  sorry

end NUMINAMATH_CALUDE_max_profit_at_max_price_verify_conditions_l1778_177823


namespace NUMINAMATH_CALUDE_coin_packing_inequality_l1778_177899

/-- Given a circular table of radius R and n non-overlapping circular coins of radius r
    placed on it such that no more coins can be added, prove that R / r ≤ 2√n + 1 --/
theorem coin_packing_inequality (R r : ℝ) (n : ℕ) 
    (h_positive_R : R > 0) 
    (h_positive_r : r > 0) 
    (h_positive_n : n > 0) 
    (h_non_overlapping : ∀ (i j : ℕ), i < n → j < n → i ≠ j → 
      ∃ (x_i y_i x_j y_j : ℝ), (x_i - x_j)^2 + (y_i - y_j)^2 ≥ 4*r^2)
    (h_within_table : ∀ (i : ℕ), i < n → 
      ∃ (x_i y_i : ℝ), x_i^2 + y_i^2 ≤ (R - r)^2)
    (h_no_more_coins : ∀ (x y : ℝ), x^2 + y^2 ≤ (R - r)^2 → 
      ∃ (i : ℕ), i < n ∧ ∃ (x_i y_i : ℝ), (x - x_i)^2 + (y - y_i)^2 < 4*r^2) :
  R / r ≤ 2 * Real.sqrt n + 1 := by
sorry

end NUMINAMATH_CALUDE_coin_packing_inequality_l1778_177899


namespace NUMINAMATH_CALUDE_shifted_linear_function_equation_l1778_177871

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  slope : ℝ
  yIntercept : ℝ

/-- Shifts a linear function vertically by a given amount -/
def shiftVertically (f : LinearFunction) (shift : ℝ) : LinearFunction :=
  { slope := f.slope, yIntercept := f.yIntercept + shift }

theorem shifted_linear_function_equation 
  (f : LinearFunction) 
  (h1 : f.slope = 2) 
  (h2 : f.yIntercept = -3) :
  (shiftVertically f 3).yIntercept = 0 := by
  sorry

#check shifted_linear_function_equation

end NUMINAMATH_CALUDE_shifted_linear_function_equation_l1778_177871


namespace NUMINAMATH_CALUDE_min_rows_for_hockey_arena_l1778_177837

/-- Represents a hockey arena with rows of seats -/
structure Arena where
  seats_per_row : ℕ
  total_students : ℕ
  max_students_per_school : ℕ

/-- Calculates the minimum number of rows required in the arena -/
def min_rows_required (arena : Arena) : ℕ :=
  sorry

/-- The theorem stating the minimum number of rows required for the given conditions -/
theorem min_rows_for_hockey_arena :
  let arena : Arena := {
    seats_per_row := 168,
    total_students := 2016,
    max_students_per_school := 45
  }
  min_rows_required arena = 16 := by sorry

end NUMINAMATH_CALUDE_min_rows_for_hockey_arena_l1778_177837


namespace NUMINAMATH_CALUDE_triangle_angle_A_l1778_177808

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Theorem statement
theorem triangle_angle_A (t : Triangle) :
  t.a = 3 ∧ t.b = 24/5 ∧ Real.cos t.B = 3/5 → t.A = 30 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l1778_177808


namespace NUMINAMATH_CALUDE_sandy_shopping_total_l1778_177829

/-- The total amount Sandy spent on clothes after discounts, coupon, and tax -/
def total_spent (shorts shirt jacket shoes accessories discount coupon tax : ℚ) : ℚ :=
  let initial_total := shorts + shirt + jacket + shoes + accessories
  let discounted_total := initial_total * (1 - discount)
  let after_coupon := discounted_total - coupon
  let final_total := after_coupon * (1 + tax)
  final_total

/-- Theorem stating the total amount Sandy spent on clothes -/
theorem sandy_shopping_total :
  total_spent 13.99 12.14 7.43 8.50 10.75 0.10 5.00 0.075 = 45.72 := by
  sorry

end NUMINAMATH_CALUDE_sandy_shopping_total_l1778_177829


namespace NUMINAMATH_CALUDE_triangle_side_length_l1778_177802

theorem triangle_side_length (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2)
  (h_thirty_deg : a / c = 1 / 2) (h_hypotenuse : c = 6 * Real.sqrt 2) :
  b = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1778_177802


namespace NUMINAMATH_CALUDE_generalized_spatial_apollonian_problems_l1778_177890

/-- The number of types of objects (sphere, point, plane) --/
def n : ℕ := 3

/-- The number of objects to be chosen --/
def k : ℕ := 4

/-- Combinations with repetition --/
def combinations_with_repetition (n k : ℕ) : ℕ := Nat.choose (n + k - 1) k

/-- The number of generalized spatial Apollonian problems --/
theorem generalized_spatial_apollonian_problems :
  combinations_with_repetition n k = 15 := by
  sorry

end NUMINAMATH_CALUDE_generalized_spatial_apollonian_problems_l1778_177890


namespace NUMINAMATH_CALUDE_milk_dilution_l1778_177814

theorem milk_dilution (initial_volume : ℝ) (pure_milk_added : ℝ) (initial_water_percentage : ℝ) :
  initial_volume = 10 →
  pure_milk_added = 15 →
  initial_water_percentage = 5 →
  let initial_water := initial_volume * (initial_water_percentage / 100)
  let final_volume := initial_volume + pure_milk_added
  let final_water_percentage := (initial_water / final_volume) * 100
  final_water_percentage = 2 := by
  sorry

end NUMINAMATH_CALUDE_milk_dilution_l1778_177814


namespace NUMINAMATH_CALUDE_donnas_truck_weight_l1778_177821

-- Define the given weights and quantities
def bridge_limit : ℕ := 20000
def empty_truck_weight : ℕ := 12000
def soda_crates : ℕ := 20
def soda_crate_weight : ℕ := 50
def dryers : ℕ := 3
def dryer_weight : ℕ := 3000

-- Define the theorem
theorem donnas_truck_weight :
  let soda_weight := soda_crates * soda_crate_weight
  let produce_weight := 2 * soda_weight
  let dryers_weight := dryers * dryer_weight
  let total_weight := empty_truck_weight + soda_weight + produce_weight + dryers_weight
  total_weight = 24000 := by
  sorry

end NUMINAMATH_CALUDE_donnas_truck_weight_l1778_177821


namespace NUMINAMATH_CALUDE_cheryl_m_and_ms_l1778_177815

/-- Cheryl's m&m's problem -/
theorem cheryl_m_and_ms 
  (initial : ℕ) 
  (after_dinner : ℕ) 
  (given_to_sister : ℕ) 
  (h1 : initial = 25) 
  (h2 : after_dinner = 5) 
  (h3 : given_to_sister = 13) :
  initial - (after_dinner + given_to_sister) = 7 :=
by sorry

end NUMINAMATH_CALUDE_cheryl_m_and_ms_l1778_177815


namespace NUMINAMATH_CALUDE_p_implies_m_range_p_and_q_implies_m_range_l1778_177830

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ (a b : ℝ), a > b ∧ a = m - 1 ∧ b = 3 - m ∧ 
  ∀ (x y : ℝ), x^2 / a + y^2 / b = 1 → ∃ (c : ℝ), x^2 + (y^2 - c^2) = a * b

def q (m : ℝ) : Prop := ∀ x : ℝ, ∃ y : ℝ, y = x^2 - m*x + 9/16 ∧ y > 0

-- Theorem 1
theorem p_implies_m_range (m : ℝ) : p m → 1 < m ∧ m < 2 := by sorry

-- Theorem 2
theorem p_and_q_implies_m_range (m : ℝ) : p m ∧ q m → 1 < m ∧ m < 3/2 := by sorry

end NUMINAMATH_CALUDE_p_implies_m_range_p_and_q_implies_m_range_l1778_177830


namespace NUMINAMATH_CALUDE_chloe_picked_42_carrots_l1778_177884

/-- Represents the number of carrots Chloe picked on the second day -/
def carrots_picked_next_day (initial_carrots : ℕ) (carrots_thrown : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (initial_carrots - carrots_thrown)

/-- Theorem stating that Chloe picked 42 carrots the next day -/
theorem chloe_picked_42_carrots : 
  carrots_picked_next_day 48 45 45 = 42 := by
  sorry

#eval carrots_picked_next_day 48 45 45

end NUMINAMATH_CALUDE_chloe_picked_42_carrots_l1778_177884


namespace NUMINAMATH_CALUDE_angle_between_vectors_l1778_177816

/-- Given plane vectors a and b, prove that the angle between a and a+b is π/3 -/
theorem angle_between_vectors (a b : ℝ × ℝ) :
  a = (1, 0) →
  b = (-1/2, Real.sqrt 3/2) →
  let a_plus_b := (a.1 + b.1, a.2 + b.2)
  Real.arccos ((a.1 * a_plus_b.1 + a.2 * a_plus_b.2) / 
    (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (a_plus_b.1^2 + a_plus_b.2^2))) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l1778_177816


namespace NUMINAMATH_CALUDE_yard_trees_l1778_177852

/-- The number of trees in a yard with given specifications -/
def numTrees (yardLength : ℕ) (treeDist : ℕ) : ℕ :=
  (yardLength / treeDist) + 1

theorem yard_trees :
  numTrees 180 18 = 12 :=
by sorry

end NUMINAMATH_CALUDE_yard_trees_l1778_177852


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1778_177891

theorem sqrt_equation_solution (x : ℝ) : 
  x ≥ 2 → 
  (Real.sqrt (x + 5 - 6 * Real.sqrt (x - 2)) + Real.sqrt (x + 12 - 8 * Real.sqrt (x - 2)) = 2) ↔
  (11 ≤ x ∧ x ≤ 27) := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1778_177891


namespace NUMINAMATH_CALUDE_school_growth_difference_l1778_177840

theorem school_growth_difference
  (total_last_year : ℕ)
  (school_yy_last_year : ℕ)
  (xx_growth_rate : ℚ)
  (yy_growth_rate : ℚ)
  (h1 : total_last_year = 4000)
  (h2 : school_yy_last_year = 2400)
  (h3 : xx_growth_rate = 7 / 100)
  (h4 : yy_growth_rate = 3 / 100) :
  let school_xx_last_year := total_last_year - school_yy_last_year
  let xx_growth := (school_xx_last_year : ℚ) * xx_growth_rate
  let yy_growth := (school_yy_last_year : ℚ) * yy_growth_rate
  ⌊xx_growth - yy_growth⌋ = 40 := by
  sorry

end NUMINAMATH_CALUDE_school_growth_difference_l1778_177840


namespace NUMINAMATH_CALUDE_johns_distance_is_285_l1778_177896

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- John's total driving distance -/
def johns_total_distance : ℝ :=
  distance 45 2 + distance 30 0.5 + distance 60 1 + distance 20 1 + distance 50 2

/-- Theorem stating that John's total driving distance is 285 miles -/
theorem johns_distance_is_285 : johns_total_distance = 285 := by
  sorry

end NUMINAMATH_CALUDE_johns_distance_is_285_l1778_177896


namespace NUMINAMATH_CALUDE_unique_reverse_double_minus_one_l1778_177883

/-- A function that reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Theorem stating that 37 is the unique two-digit number that satisfies the given condition -/
theorem unique_reverse_double_minus_one :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ 2 * n - 1 = reverse_digits n :=
by
  sorry

end NUMINAMATH_CALUDE_unique_reverse_double_minus_one_l1778_177883


namespace NUMINAMATH_CALUDE_expression_evaluation_l1778_177877

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 6 - 1
  let y : ℝ := Real.sqrt 6 + 1
  (2*x + y)^2 + (x - y)*(x + y) - 5*x*(x - y) = 45 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1778_177877


namespace NUMINAMATH_CALUDE_equidistant_point_l1778_177803

theorem equidistant_point : ∃ x : ℝ, |x - (-2)| = |x - 4| ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_l1778_177803


namespace NUMINAMATH_CALUDE_remainder_8547_mod_9_l1778_177863

theorem remainder_8547_mod_9 : 8547 % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_8547_mod_9_l1778_177863


namespace NUMINAMATH_CALUDE_survey_result_l1778_177888

theorem survey_result (total : ℕ) (tv_dislike_percent : ℚ) (both_dislike_percent : ℚ)
  (h_total : total = 1500)
  (h_tv_dislike : tv_dislike_percent = 25 / 100)
  (h_both_dislike : both_dislike_percent = 15 / 100) :
  ⌊(total : ℚ) * tv_dislike_percent * both_dislike_percent⌋ = 56 := by
  sorry

end NUMINAMATH_CALUDE_survey_result_l1778_177888


namespace NUMINAMATH_CALUDE_min_cars_with_racing_stripes_l1778_177895

theorem min_cars_with_racing_stripes 
  (total_cars : ℕ) 
  (cars_without_ac : ℕ) 
  (max_ac_no_stripes : ℕ) 
  (h1 : total_cars = 100)
  (h2 : cars_without_ac = 37)
  (h3 : max_ac_no_stripes = 49) :
  ∃ (min_racing_stripes : ℕ), 
    min_racing_stripes = 14 ∧ 
    (∀ (racing_stripes : ℕ), 
      racing_stripes ≥ min_racing_stripes → 
      ∃ (ac_with_stripes ac_no_stripes no_ac_with_stripes no_ac_no_stripes : ℕ),
        ac_with_stripes + ac_no_stripes + no_ac_with_stripes + no_ac_no_stripes = total_cars ∧
        no_ac_with_stripes + no_ac_no_stripes = cars_without_ac ∧
        ac_no_stripes ≤ max_ac_no_stripes ∧
        racing_stripes = ac_with_stripes + no_ac_with_stripes) :=
sorry

end NUMINAMATH_CALUDE_min_cars_with_racing_stripes_l1778_177895


namespace NUMINAMATH_CALUDE_pearl_distribution_l1778_177819

theorem pearl_distribution (n : ℕ) : 
  (∀ m : ℕ, m > n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5)) → 
  n % 8 = 6 → 
  n % 7 = 5 → 
  n % 9 = 0 → 
  n = 54 := by
sorry

end NUMINAMATH_CALUDE_pearl_distribution_l1778_177819


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l1778_177897

-- Define the line l passing through (-2, 1) with slope k
def line (k : ℝ) (x y : ℝ) : Prop :=
  y - 1 = k * (x + 2)

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Define the condition for the line to intersect the parabola at only one point
def unique_intersection (k : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, line k p.1 p.2 ∧ parabola p.1 p.2

-- Theorem statement
theorem line_parabola_intersection (k : ℝ) :
  unique_intersection k → k = 0 ∨ k = -1 ∨ k = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l1778_177897


namespace NUMINAMATH_CALUDE_circles_intersect_l1778_177874

/-- Two circles are intersecting if the distance between their centers is less than the sum of their radii
    and greater than the absolute difference of their radii. -/
def circles_intersecting (center1 center2 : ℝ × ℝ) (radius1 radius2 : ℝ) : Prop :=
  let distance := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)
  distance < radius1 + radius2 ∧ distance > |radius1 - radius2|

/-- Given two circles: (x-a)^2+(y-b)^2=4 and (x-a-1)^2+(y-b-2)^2=1 where a, b ∈ ℝ,
    prove that they are intersecting. -/
theorem circles_intersect (a b : ℝ) : 
  circles_intersecting (a, b) (a+1, b+2) 2 1 := by
  sorry


end NUMINAMATH_CALUDE_circles_intersect_l1778_177874


namespace NUMINAMATH_CALUDE_polygon_sides_l1778_177800

/-- A convex polygon with the sum of all angles except one equal to 2790° has 18 sides -/
theorem polygon_sides (n : ℕ) (angle_sum : ℝ) : 
  n > 2 →
  angle_sum = 2790 →
  (n - 2) * 180 > angle_sum →
  (n - 1) * 180 ≥ angle_sum →
  n = 18 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l1778_177800


namespace NUMINAMATH_CALUDE_man_downstream_speed_l1778_177868

/-- Given a man's upstream speed and the stream speed, calculates his downstream speed -/
def downstream_speed (upstream_speed stream_speed : ℝ) : ℝ :=
  upstream_speed + 2 * stream_speed

/-- Proves that given the specified conditions, the man's downstream speed is 14 kmph -/
theorem man_downstream_speed :
  let upstream_speed : ℝ := 8
  let stream_speed : ℝ := 3
  downstream_speed upstream_speed stream_speed = 14 := by
  sorry

end NUMINAMATH_CALUDE_man_downstream_speed_l1778_177868


namespace NUMINAMATH_CALUDE_mike_arcade_time_mike_play_time_l1778_177857

/-- Given Mike's weekly pay and arcade expenses, calculate his play time in minutes -/
theorem mike_arcade_time (weekly_pay : ℕ) (food_cost : ℕ) (hourly_rate : ℕ) : ℕ :=
  let arcade_budget := weekly_pay / 2
  let token_budget := arcade_budget - food_cost
  let play_hours := token_budget / hourly_rate
  play_hours * 60

/-- Prove that Mike can play for 300 minutes given the specific conditions -/
theorem mike_play_time :
  mike_arcade_time 100 10 8 = 300 := by
  sorry

end NUMINAMATH_CALUDE_mike_arcade_time_mike_play_time_l1778_177857


namespace NUMINAMATH_CALUDE_oil_leak_height_l1778_177872

/-- The speed of oil leaking from a circular cylinder -/
def leak_speed (k : ℝ) (h : ℝ) : ℝ := k * h^2

theorem oil_leak_height (k : ℝ) (h' : ℝ) :
  (k > 0) →
  (leak_speed k 12 = 9 * leak_speed k h') →
  h' = 4 := by
sorry

end NUMINAMATH_CALUDE_oil_leak_height_l1778_177872


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1778_177862

-- Define a structure for parallelograms
structure Parallelogram where
  -- Add necessary fields (for illustration purposes)
  vertices : Fin 4 → ℝ × ℝ

-- Define properties for diagonals
def diagonals_are_equal (p : Parallelogram) : Prop :=
  -- Add definition here
  sorry

def diagonals_bisect_each_other (p : Parallelogram) : Prop :=
  -- Add definition here
  sorry

-- The theorem to prove
theorem negation_of_universal_proposition :
  (¬ ∀ p : Parallelogram, diagonals_are_equal p ∧ diagonals_bisect_each_other p) ↔
  (∃ p : Parallelogram, ¬(diagonals_are_equal p) ∨ ¬(diagonals_bisect_each_other p)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1778_177862


namespace NUMINAMATH_CALUDE_inequality_always_true_l1778_177824

theorem inequality_always_true (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - a * x + 1 > 0) ↔ (-2 * Real.sqrt 2 < a ∧ a < 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_true_l1778_177824


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_1_range_of_a_for_inequality_l1778_177844

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 2|

-- Part 1
theorem solution_set_when_a_is_1 :
  {x : ℝ | f 1 x ≥ 5} = {x : ℝ | x ≤ -3 ∨ x ≥ 2} := by sorry

-- Part 2
theorem range_of_a_for_inequality :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc 0 1 → f a x ≤ |x + 4|) ↔ a ∈ Set.Icc (-1) 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_1_range_of_a_for_inequality_l1778_177844


namespace NUMINAMATH_CALUDE_arrange_sticks_into_triangles_l1778_177838

/-- Represents a stick with a positive length -/
structure Stick where
  length : ℝ
  positive : length > 0

/-- Represents a triangle formed by three sticks -/
structure Triangle where
  side1 : Stick
  side2 : Stick
  side3 : Stick

/-- Checks if three sticks can form a valid triangle -/
def isValidTriangle (s1 s2 s3 : Stick) : Prop :=
  s1.length + s2.length > s3.length ∧
  s1.length + s3.length > s2.length ∧
  s2.length + s3.length > s1.length

/-- Theorem stating that it's always possible to arrange six sticks into two triangles
    with one triangle having sides of one, two, and three sticks -/
theorem arrange_sticks_into_triangles
  (s1 s2 s3 s4 s5 s6 : Stick)
  (h_pairwise_different : s1.length < s2.length ∧ s2.length < s3.length ∧
                          s3.length < s4.length ∧ s4.length < s5.length ∧
                          s5.length < s6.length) :
  ∃ (t1 t2 : Triangle),
    (isValidTriangle t1.side1 t1.side2 t1.side3) ∧
    (isValidTriangle t2.side1 t2.side2 t2.side3) ∧
    ((t1.side1.length = s1.length ∧ t1.side2.length = s3.length + s5.length ∧ t1.side3.length = s2.length + s4.length + s6.length) ∨
     (t2.side1.length = s1.length ∧ t2.side2.length = s3.length + s5.length ∧ t2.side3.length = s2.length + s4.length + s6.length)) :=
by sorry

end NUMINAMATH_CALUDE_arrange_sticks_into_triangles_l1778_177838


namespace NUMINAMATH_CALUDE_vertical_angles_equal_l1778_177864

-- Define a point in a 2D plane
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in a 2D plane using two points
structure Line2D where
  p1 : Point2D
  p2 : Point2D

-- Define an angle
structure Angle where
  vertex : Point2D
  ray1 : Point2D
  ray2 : Point2D

-- Define the intersection of two lines
def intersection (l1 l2 : Line2D) : Point2D :=
  sorry

-- Define vertical angles
def verticalAngles (l1 l2 : Line2D) : (Angle × Angle) :=
  sorry

-- Theorem: Vertical angles are equal
theorem vertical_angles_equal (l1 l2 : Line2D) :
  let (a1, a2) := verticalAngles l1 l2
  a1 = a2 :=
sorry

end NUMINAMATH_CALUDE_vertical_angles_equal_l1778_177864


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1778_177807

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  d > 0 →  -- positive common difference
  a 1 + a 7 = 10 →  -- sum of roots condition
  a 1 * a 7 = 16 →  -- product of roots condition
  a 2 + a 4 + a 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1778_177807


namespace NUMINAMATH_CALUDE_min_value_theorem_l1778_177855

theorem min_value_theorem (a b x : ℝ) (ha : a > 1) (hb : b > 2) (hx : x + b = 5) :
  (∀ y : ℝ, y > 1 ∧ y + b = 5 → (1 / (a - 1) + 9 / (b - 2) ≤ 1 / (y - 1) + 9 / (b - 2))) ∧
  (1 / (a - 1) + 9 / (b - 2) = 8) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1778_177855


namespace NUMINAMATH_CALUDE_income_ratio_proof_l1778_177875

/-- Given two persons P1 and P2 with the following conditions:
    1. The ratio of their expenditures is 3:2
    2. Each saves Rs. 1800 at the end of the year
    3. The income of P1 is Rs. 4500
    Prove that the ratio of their incomes is 5:4 -/
theorem income_ratio_proof (expenditure_ratio : ℚ) (savings : ℕ) (income_p1 : ℕ) :
  expenditure_ratio = 3/2 →
  savings = 1800 →
  income_p1 = 4500 →
  ∃ (income_p2 : ℕ), (income_p1 : ℚ) / income_p2 = 5/4 :=
by sorry

end NUMINAMATH_CALUDE_income_ratio_proof_l1778_177875


namespace NUMINAMATH_CALUDE_sarah_speeding_tickets_l1778_177876

theorem sarah_speeding_tickets (total_tickets : ℕ) (mark_parking : ℕ) :
  total_tickets = 24 →
  mark_parking = 8 →
  ∃ (sarah_speeding : ℕ),
    sarah_speeding = 6 ∧
    sarah_speeding + sarah_speeding + mark_parking + mark_parking / 2 = total_tickets :=
by sorry

end NUMINAMATH_CALUDE_sarah_speeding_tickets_l1778_177876


namespace NUMINAMATH_CALUDE_mirror_area_l1778_177867

/-- The area of a rectangular mirror inside a frame -/
theorem mirror_area (frame_width : ℕ) (frame_height : ℕ) (frame_thickness : ℕ) : 
  frame_width = 90 ∧ frame_height = 70 ∧ frame_thickness = 15 →
  (frame_width - 2 * frame_thickness) * (frame_height - 2 * frame_thickness) = 2400 := by
sorry

end NUMINAMATH_CALUDE_mirror_area_l1778_177867


namespace NUMINAMATH_CALUDE_total_numbers_l1778_177810

theorem total_numbers (average : ℝ) (first_six_average : ℝ) (last_eight_average : ℝ) (eighth_number : ℝ)
  (h1 : average = 60)
  (h2 : first_six_average = 57)
  (h3 : last_eight_average = 61)
  (h4 : eighth_number = 50) :
  ∃ n : ℕ, n = 13 ∧ 
    average * n = first_six_average * 6 + last_eight_average * 8 - eighth_number :=
by
  sorry


end NUMINAMATH_CALUDE_total_numbers_l1778_177810


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1778_177836

theorem linear_equation_solution (x y : ℝ) :
  2 * x + y = 5 → y = -2 * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1778_177836


namespace NUMINAMATH_CALUDE_otimes_twelve_nine_l1778_177856

-- Define the custom operation
def otimes (a b : ℚ) : ℚ := a + (4 * a) / (3 * b)

-- Theorem statement
theorem otimes_twelve_nine : otimes 12 9 = 13 + 7/9 := by
  sorry

end NUMINAMATH_CALUDE_otimes_twelve_nine_l1778_177856


namespace NUMINAMATH_CALUDE_vote_percentages_sum_to_100_l1778_177827

theorem vote_percentages_sum_to_100 (candidate1_percent candidate2_percent candidate3_percent : ℝ) 
  (h1 : candidate1_percent = 25)
  (h2 : candidate2_percent = 45)
  (h3 : candidate3_percent = 30) :
  candidate1_percent + candidate2_percent + candidate3_percent = 100 := by
  sorry

end NUMINAMATH_CALUDE_vote_percentages_sum_to_100_l1778_177827
