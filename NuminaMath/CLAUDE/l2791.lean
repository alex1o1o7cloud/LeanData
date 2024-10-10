import Mathlib

namespace project_completion_time_l2791_279173

/-- The number of days B takes to complete the project -/
def B_days : ℕ := 30

/-- The total number of days when A and B work together with A quitting 5 days before completion -/
def total_days : ℕ := 15

/-- The number of days before completion that A quits -/
def A_quit_days : ℕ := 5

/-- The number of days A can complete the project alone -/
def A_days : ℕ := 20

theorem project_completion_time :
  (total_days - A_quit_days) * (1 / A_days + 1 / B_days) + A_quit_days * (1 / B_days) = 1 :=
by sorry

#check project_completion_time

end project_completion_time_l2791_279173


namespace square_field_side_length_l2791_279114

theorem square_field_side_length (area : ℝ) (side : ℝ) : 
  area = 400 → side ^ 2 = area → side = 20 := by
  sorry

end square_field_side_length_l2791_279114


namespace unique_special_number_l2791_279146

/-- A two-digit number satisfying specific divisibility properties -/
def SpecialNumber (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ 
  2 ∣ n ∧
  3 ∣ (n + 1) ∧
  4 ∣ (n + 2) ∧
  5 ∣ (n + 3)

/-- Theorem stating that 62 is the unique two-digit number satisfying the given conditions -/
theorem unique_special_number : ∃! n, SpecialNumber n :=
  sorry

end unique_special_number_l2791_279146


namespace council_vote_change_l2791_279130

theorem council_vote_change (total : ℕ) (initial_for initial_against : ℕ) 
  (revote_for revote_against : ℕ) :
  total = 350 →
  initial_for + initial_against = total →
  initial_against > initial_for →
  revote_for + revote_against = total →
  revote_for > revote_against →
  (revote_for - revote_against) = 2 * (initial_against - initial_for) →
  revote_for = (10 * initial_against) / 9 →
  revote_for - initial_for = 66 := by
  sorry

end council_vote_change_l2791_279130


namespace min_intersection_cardinality_l2791_279191

-- Define the cardinality of a set
def card (S : Set α) : ℕ := sorry

-- Define the number of subsets of a set
def n (S : Set α) : ℕ := 2^(card S)

-- Define the theorem
theorem min_intersection_cardinality 
  (A B C : Set α) 
  (h1 : n A + n B + n C = n (A ∪ B ∪ C))
  (h2 : card A = 100)
  (h3 : card B = 101)
  (h4 : card (A ∩ B) ≥ 95) :
  96 ≤ card (A ∩ B ∩ C) := by
  sorry

end min_intersection_cardinality_l2791_279191


namespace knight_seating_probability_correct_l2791_279153

/-- The probability of three knights seated at a round table with n chairs
    such that each knight has an empty chair on both sides. -/
def knight_seating_probability (n : ℕ) : ℚ :=
  if n ≥ 6 then
    (n - 4) * (n - 5) / ((n - 1) * (n - 2))
  else
    0

theorem knight_seating_probability_correct (n : ℕ) (h : n ≥ 6) :
  knight_seating_probability n =
    (n - 4) * (n - 5) / ((n - 1) * (n - 2)) :=
by sorry

end knight_seating_probability_correct_l2791_279153


namespace ellipse_k_range_l2791_279105

/-- The range of k for an ellipse with equation x²/(3-k) + y²/(5+k) = 1 and foci on the y-axis -/
theorem ellipse_k_range :
  ∀ k : ℝ,
  (∀ x y : ℝ, x^2 / (3 - k) + y^2 / (5 + k) = 1) →
  (5 + k > 3 - k) →
  (3 - k > 0) →
  (5 + k > 0) →
  -1 < k ∧ k < 3 :=
by sorry

end ellipse_k_range_l2791_279105


namespace congruence_problem_l2791_279161

def binomial_sum (n : ℕ) : ℕ := (Finset.range (n + 1)).sum (λ k => Nat.choose n k * 2^k)

theorem congruence_problem (a b : ℤ) :
  a = binomial_sum 20 ∧ a ≡ b [ZMOD 10] → b = 2011 := by
  sorry

end congruence_problem_l2791_279161


namespace cheese_cookie_price_l2791_279111

/-- Proves that the price of a pack of cheese cookies is $1 -/
theorem cheese_cookie_price (boxes_per_carton : ℕ) (packs_per_box : ℕ) (cost_dozen_cartons : ℕ) 
  (h1 : boxes_per_carton = 12)
  (h2 : packs_per_box = 10)
  (h3 : cost_dozen_cartons = 1440) :
  (cost_dozen_cartons : ℚ) / ((12 * boxes_per_carton * packs_per_box) : ℚ) = 1 := by
  sorry

end cheese_cookie_price_l2791_279111


namespace right_triangle_acute_angle_measure_l2791_279121

/-- In a right triangle where the ratio of the measures of the acute angles is 7:2,
    the measure of the smaller angle is 20°. -/
theorem right_triangle_acute_angle_measure (α β : ℝ) : 
  α > 0 ∧ β > 0 ∧  -- Angles are positive
  α + β + 90 = 180 ∧  -- Sum of angles in a triangle is 180°
  α / β = 7 / 2 ∧  -- Ratio of acute angles
  α > β  -- α is the larger acute angle
  → β = 20 := by sorry

end right_triangle_acute_angle_measure_l2791_279121


namespace herd_size_l2791_279133

/-- Given a herd of cows divided among four sons, prove that the total number of cows is 224 --/
theorem herd_size (herd : ℕ) : herd = 224 :=
  by
  have h1 : (3 : ℚ) / 7 + 1 / 3 + 1 / 6 + (herd - 16 : ℚ) / herd = 1 := by sorry
  have h2 : (herd - 16 : ℚ) / herd = 1 - (3 / 7 + 1 / 3 + 1 / 6) := by sorry
  have h3 : (herd - 16 : ℚ) / herd = 1 / 14 := by sorry
  have h4 : (16 : ℚ) / herd = 1 / 14 := by sorry
  sorry

end herd_size_l2791_279133


namespace special_number_in_list_l2791_279149

theorem special_number_in_list (l : List ℝ) (n : ℝ) (h1 : l.length = 21) 
  (h2 : n ∈ l) (h3 : n = 4 * ((l.sum - n) / 20)) : 
  n = (1 / 6 : ℝ) * l.sum :=
by
  sorry

end special_number_in_list_l2791_279149


namespace average_string_length_l2791_279129

theorem average_string_length :
  let string1 : ℚ := 2
  let string2 : ℚ := 5
  let string3 : ℚ := 3
  let num_strings : ℕ := 3
  (string1 + string2 + string3) / num_strings = 10 / 3 := by
  sorry

end average_string_length_l2791_279129


namespace equilateral_triangle_paths_l2791_279120

/-- Represents the number of paths in an equilateral triangle of side length n --/
def f (n : ℕ) : ℕ := n.factorial

/-- 
Theorem: The number of paths from the top triangle to the middle triangle 
in the bottom row of an equilateral triangle with side length n, 
where paths can only move downward and never revisit a triangle, is equal to n!.
-/
theorem equilateral_triangle_paths (n : ℕ) : f n = n.factorial := by
  sorry

end equilateral_triangle_paths_l2791_279120


namespace problem_solution_l2791_279147

theorem problem_solution (a b : ℝ) (h1 : 4 + a = 5 - b) (h2 : 5 + b = 8 + a) : 4 - a = 3 := by
  sorry

end problem_solution_l2791_279147


namespace club_members_count_l2791_279125

theorem club_members_count (n : ℕ) (h : n > 2) :
  (2 : ℚ) / ((n : ℚ) - 1) = (1 : ℚ) / 5 → n = 11 := by
  sorry

end club_members_count_l2791_279125


namespace skew_lines_sufficient_not_necessary_l2791_279134

-- Define the property of lines being skew
def are_skew_lines (a b : Line3D) : Prop := sorry

-- Define the property of lines having no common points
def have_no_common_points (a b : Line3D) : Prop := sorry

-- Theorem stating that "are_skew_lines" is a sufficient but not necessary condition for "have_no_common_points"
theorem skew_lines_sufficient_not_necessary (a b : Line3D) :
  (are_skew_lines a b → have_no_common_points a b) ∧
  ¬(have_no_common_points a b → are_skew_lines a b) :=
sorry

end skew_lines_sufficient_not_necessary_l2791_279134


namespace calculator_game_result_l2791_279181

def calculator_game (n : Nat) (a b c : Int) : Int :=
  let f1 := fun x => x^3
  let f2 := fun x => x^2
  let f3 := fun x => -x
  (f1^[n] a) + (f2^[n] b) + (f3^[n] c)

theorem calculator_game_result :
  calculator_game 45 1 0 (-2) = 3 := by
  sorry

end calculator_game_result_l2791_279181


namespace folded_carbon_copies_l2791_279157

/-- Represents the number of carbon copies produced given the initial number of sheets,
    carbon papers, and whether the setup is folded or not -/
def carbonCopies (sheets : ℕ) (carbons : ℕ) (folded : Bool) : ℕ :=
  if folded then
    2 * (sheets - 1)
  else
    carbons

/-- Theorem stating that with 3 sheets, 2 carbons, and folded setup, 4 carbon copies are produced -/
theorem folded_carbon_copies :
  carbonCopies 3 2 true = 4 := by sorry

end folded_carbon_copies_l2791_279157


namespace sum_110_is_neg_110_l2791_279190

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  /-- Sum of the first 10 terms -/
  sum_10 : ℤ
  /-- Sum of the first 100 terms -/
  sum_100 : ℤ
  /-- Property: sum of first 10 terms is 100 -/
  prop_10 : sum_10 = 100
  /-- Property: sum of first 100 terms is 10 -/
  prop_100 : sum_100 = 10

/-- Theorem: For the given arithmetic sequence, the sum of the first 110 terms is -110 -/
theorem sum_110_is_neg_110 (seq : ArithmeticSequence) : ℤ :=
  -110

#check sum_110_is_neg_110

end sum_110_is_neg_110_l2791_279190


namespace function_property_l2791_279138

/-- Given a function f(x) = (ln x - k - 1)x where k is a real number and x > 1,
    prove that if x₁ ≠ x₂ and f(x₁) = f(x₂), then x₁x₂ < e^(2k) -/
theorem function_property (k : ℝ) :
  let f : ℝ → ℝ := λ x => (Real.log x - k - 1) * x
  ∀ x₁ x₂, x₁ > 1 → x₂ > 1 → x₁ ≠ x₂ → f x₁ = f x₂ → x₁ * x₂ < Real.exp (2 * k) := by
  sorry


end function_property_l2791_279138


namespace min_value_of_f_l2791_279118

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem min_value_of_f :
  ∃ (min : ℝ), min = -1 / Real.exp 1 ∧ ∀ (x : ℝ), f x ≥ min :=
sorry

end min_value_of_f_l2791_279118


namespace quadratic_distinct_roots_l2791_279113

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  m < -6 ∨ m > 6 := by
sorry

end quadratic_distinct_roots_l2791_279113


namespace range_of_a_l2791_279162

-- Define p and q as predicates on real numbers
def p (a : ℝ) (x : ℝ) : Prop := x ≥ a
def q (x : ℝ) : Prop := x^2 - 2*x - 3 ≥ 0

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x, p a x → q x) ∧ (∃ x, q x ∧ ¬p a x) → a ≥ 3 :=
by sorry

end range_of_a_l2791_279162


namespace dividend_calculation_l2791_279169

theorem dividend_calculation (dividend quotient remainder : ℕ) : 
  dividend / 3 = quotient ∧ 
  dividend % 3 = remainder ∧ 
  quotient = 16 ∧ 
  remainder = 4 → 
  dividend = 52 := by
sorry

end dividend_calculation_l2791_279169


namespace factors_of_60_l2791_279100

/-- The number of positive factors of 60 -/
def num_factors_60 : ℕ := sorry

/-- Theorem stating that the number of positive factors of 60 is 12 -/
theorem factors_of_60 : num_factors_60 = 12 := by sorry

end factors_of_60_l2791_279100


namespace union_of_A_and_B_l2791_279176

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 - x - 2 = 0}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ A, y = x + 3}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {-1, 2, 5} := by sorry

end union_of_A_and_B_l2791_279176


namespace calculate_marked_price_jobber_marked_price_l2791_279122

/-- Calculate the marked price of an article given the original price, purchase discount,
    desired profit margin, and selling discount. -/
theorem calculate_marked_price (original_price : ℝ) (purchase_discount : ℝ) 
    (profit_margin : ℝ) (selling_discount : ℝ) : ℝ :=
  let purchase_price := original_price * (1 - purchase_discount)
  let desired_selling_price := purchase_price * (1 + profit_margin)
  desired_selling_price / (1 - selling_discount)

/-- The marked price of the article should be $50.00 -/
theorem jobber_marked_price : 
  calculate_marked_price 40 0.25 0.5 0.1 = 50 := by
  sorry

end calculate_marked_price_jobber_marked_price_l2791_279122


namespace tower_heights_count_l2791_279104

/-- Represents the dimensions of a brick in inches -/
structure BrickDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of different tower heights achievable -/
def calculateTowerHeights (numBricks : ℕ) (dimensions : BrickDimensions) : ℕ :=
  let minHeight := numBricks * dimensions.length
  let maxAdditionalHeight := numBricks * (dimensions.height - dimensions.length)
  (maxAdditionalHeight / 5 + 1 : ℕ)

/-- Theorem stating the number of different tower heights achievable -/
theorem tower_heights_count (numBricks : ℕ) (dimensions : BrickDimensions) :
  numBricks = 78 →
  dimensions = { length := 3, width := 8, height := 20 } →
  calculateTowerHeights numBricks dimensions = 266 := by
  sorry

end tower_heights_count_l2791_279104


namespace election_votes_proof_l2791_279150

theorem election_votes_proof (total_votes : ℕ) (second_candidate_votes : ℕ) : 
  -- Given conditions
  total_votes = 27500 ∧ 
  (20000 : ℚ) / total_votes = 8011 / 11000 ∧
  total_votes = 2500 + second_candidate_votes + 20000 →
  -- Conclusion
  second_candidate_votes = 5000 := by
sorry


end election_votes_proof_l2791_279150


namespace bart_firewood_calculation_l2791_279197

/-- The number of logs Bart burns per day -/
def logs_per_day : ℕ := 5

/-- The number of days Bart burns logs (Nov 1 through Feb 28) -/
def total_days : ℕ := 120

/-- The number of trees Bart needs to cut down -/
def trees_cut : ℕ := 8

/-- The number of pieces of firewood Bart gets from one tree -/
def firewood_per_tree : ℕ := total_days * logs_per_day / trees_cut

theorem bart_firewood_calculation :
  firewood_per_tree = 75 :=
sorry

end bart_firewood_calculation_l2791_279197


namespace shoe_probability_l2791_279115

theorem shoe_probability (total_pairs : ℕ) (black_pairs brown_pairs gray_pairs : ℕ)
  (h1 : total_pairs = black_pairs + brown_pairs + gray_pairs)
  (h2 : total_pairs = 15)
  (h3 : black_pairs = 8)
  (h4 : brown_pairs = 4)
  (h5 : gray_pairs = 3) :
  let total_shoes := 2 * total_pairs
  let prob_black := (2 * black_pairs / total_shoes) * (black_pairs / (total_shoes - 1))
  let prob_brown := (2 * brown_pairs / total_shoes) * (brown_pairs / (total_shoes - 1))
  let prob_gray := (2 * gray_pairs / total_shoes) * (gray_pairs / (total_shoes - 1))
  prob_black + prob_brown + prob_gray = 89 / 435 :=
by sorry

end shoe_probability_l2791_279115


namespace total_vessels_l2791_279141

theorem total_vessels (x y z w : ℕ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0)
  (hxy : x < y) (hyz : y < z) (hzw : z < w) :
  let cruise_ships := x
  let cargo_ships := y * x
  let sailboats := y * x + z
  let fishing_boats := (y * x + z) / w
  cruise_ships + cargo_ships + sailboats + fishing_boats = x * (2 * y + 1) + z * (1 + 1 / w) :=
by sorry

end total_vessels_l2791_279141


namespace value_of_expression_l2791_279165

theorem value_of_expression (x : ℝ) (h : x = 3) : 5 - 2 * x^2 = -13 := by
  sorry

end value_of_expression_l2791_279165


namespace reseating_women_problem_l2791_279163

/-- Represents the number of ways n women can be reseated under the given conditions --/
def T : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | n + 3 => T (n + 2) + T (n + 1) + if n = 0 then 1 else T n

/-- The problem statement --/
theorem reseating_women_problem :
  T 15 = 987 := by
  sorry

end reseating_women_problem_l2791_279163


namespace perpendicular_line_equation_l2791_279124

/-- The line passing through (-1, 0) and perpendicular to x + y = 0 has equation x - y + 1 = 0 -/
theorem perpendicular_line_equation : 
  let c : ℝ × ℝ := (-1, 0)
  let l₁ : Set (ℝ × ℝ) := {p | p.1 + p.2 = 0}
  let l₂ : Set (ℝ × ℝ) := {p | p.1 - p.2 + 1 = 0}
  (∀ p ∈ l₂, (p.1 - c.1) * (1 + 1) = -(p.2 - c.2)) ∧ 
  c ∈ l₂ :=
by sorry

end perpendicular_line_equation_l2791_279124


namespace vector_operation_l2791_279174

theorem vector_operation (a b : ℝ × ℝ) :
  a = (2, 2) → b = (-1, 3) → 2 • a - b = (5, 1) := by
  sorry

end vector_operation_l2791_279174


namespace expansion_coefficient_l2791_279123

/-- The coefficient of x^5 in the expansion of (2x-√x)^8 -/
def coefficient_x5 : ℕ := 112

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

theorem expansion_coefficient :
  coefficient_x5 = (binomial 8 6) * 2^2 :=
sorry

end expansion_coefficient_l2791_279123


namespace roots_form_parallelogram_l2791_279155

/-- The polynomial whose roots we're investigating -/
def f (b : ℝ) (z : ℂ) : ℂ := z^4 - 8*z^3 + 17*b*z^2 - 2*(3*b^2 + 5*b - 4)*z + 2

/-- Predicate to check if a set of complex numbers forms a parallelogram -/
def forms_parallelogram (s : Set ℂ) : Prop :=
  ∃ (z₁ z₂ z₃ z₄ : ℂ), s = {z₁, z₂, z₃, z₄} ∧ 
    z₁ + z₃ = z₂ + z₄ ∧ z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄

/-- The main theorem stating the condition for the roots to form a parallelogram -/
theorem roots_form_parallelogram (b : ℝ) :
  forms_parallelogram {z : ℂ | f b z = 0} ↔ b = 1 ∨ b = 5/2 :=
sorry

end roots_form_parallelogram_l2791_279155


namespace price_crossover_year_l2791_279175

def price_X (year : ℕ) : ℚ :=
  4.20 + 0.45 * (year - 2001 : ℚ)

def price_Y (year : ℕ) : ℚ :=
  6.30 + 0.20 * (year - 2001 : ℚ)

theorem price_crossover_year : 
  (∀ y : ℕ, y < 2010 → price_X y ≤ price_Y y) ∧ 
  price_X 2010 > price_Y 2010 :=
by sorry

end price_crossover_year_l2791_279175


namespace greatest_prime_factor_of_221_l2791_279168

theorem greatest_prime_factor_of_221 : ∃ p : ℕ, p.Prime ∧ p ∣ 221 ∧ ∀ q : ℕ, q.Prime → q ∣ 221 → q ≤ p ∧ p = 17 := by
  sorry

end greatest_prime_factor_of_221_l2791_279168


namespace theoretical_yield_NaNO3_l2791_279102

/-- Theoretical yield of NaNO3 given initial conditions and overall yield -/
theorem theoretical_yield_NaNO3 (initial_NH4NO3 : ℝ) (initial_NaOH : ℝ) (percent_yield : ℝ) :
  initial_NH4NO3 = 2 →
  initial_NaOH = 2 →
  percent_yield = 0.85 →
  ∃ (theoretical_yield : ℝ),
    theoretical_yield = 289 ∧
    theoretical_yield = initial_NH4NO3 * 2 * 85 * percent_yield :=
by sorry

/-- Molar mass of NaNO3 in g/mol -/
def molar_mass_NaNO3 : ℝ := 85

/-- Theoretical yield in moles of NaNO3 -/
def theoretical_yield_moles (initial_NH4NO3 : ℝ) : ℝ := initial_NH4NO3 * 2

/-- Theoretical yield in grams of NaNO3 -/
def theoretical_yield_grams (theoretical_yield_moles : ℝ) : ℝ :=
  theoretical_yield_moles * molar_mass_NaNO3

/-- Actual yield in grams of NaNO3 considering percent yield -/
def actual_yield_grams (theoretical_yield_grams : ℝ) (percent_yield : ℝ) : ℝ :=
  theoretical_yield_grams * percent_yield

end theoretical_yield_NaNO3_l2791_279102


namespace original_room_population_l2791_279131

theorem original_room_population (initial_population : ℕ) : 
  (initial_population / 3 : ℚ) = 18 → initial_population = 54 :=
by
  intro h
  sorry

#check original_room_population

end original_room_population_l2791_279131


namespace least_subtraction_for_divisibility_l2791_279184

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (hn : n = 5264) (hd : d = 17) :
  ∃ (k : ℕ), k ≤ d - 1 ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
sorry

end least_subtraction_for_divisibility_l2791_279184


namespace basic_astrophysics_is_108_degrees_l2791_279196

/-- Represents the research and development budget allocation --/
structure BudgetAllocation where
  microphotonics : ℝ
  home_electronics : ℝ
  food_additives : ℝ
  genetically_modified_microorganisms : ℝ
  industrial_lubricants : ℝ

/-- The total degrees in a circle --/
def total_degrees : ℝ := 360

/-- Calculate the degrees for basic astrophysics research --/
def basic_astrophysics_degrees (ba : BudgetAllocation) : ℝ :=
  total_degrees * (1 - (ba.microphotonics + ba.home_electronics + ba.food_additives + 
                        ba.genetically_modified_microorganisms + ba.industrial_lubricants))

/-- Theorem stating that the degrees for basic astrophysics research is 108 --/
theorem basic_astrophysics_is_108_degrees (ba : BudgetAllocation) 
    (h1 : ba.microphotonics = 0.09)
    (h2 : ba.home_electronics = 0.14)
    (h3 : ba.food_additives = 0.10)
    (h4 : ba.genetically_modified_microorganisms = 0.29)
    (h5 : ba.industrial_lubricants = 0.08) :
    basic_astrophysics_degrees ba = 108 := by
  sorry

#check basic_astrophysics_is_108_degrees

end basic_astrophysics_is_108_degrees_l2791_279196


namespace base_conversion_three_digits_l2791_279103

theorem base_conversion_three_digits : 
  ∃ (b : ℕ), b > 1 ∧ b^2 ≤ 256 ∧ 256 < b^3 ∧ ∀ (x : ℕ), 1 < x ∧ x < b → (x^2 > 256 ∨ x^3 ≤ 256) :=
by sorry

end base_conversion_three_digits_l2791_279103


namespace rain_probability_implies_very_likely_l2791_279192

-- Define what "very likely" means in terms of probability
def very_likely (p : ℝ) : Prop := p ≥ 0.7

-- Theorem statement
theorem rain_probability_implies_very_likely (p : ℝ) (h : p = 0.8) : very_likely p := by
  sorry

end rain_probability_implies_very_likely_l2791_279192


namespace same_color_difference_l2791_279179

/-- The set of colors used for coloring integers. -/
inductive Color
| Red
| Blue
| Green
| Yellow

/-- A function that colors integers with one of four colors. -/
def ColoringFunction := ℤ → Color

/-- Theorem stating the existence of two integers with the same color and specific difference. -/
theorem same_color_difference (f : ColoringFunction) (x y : ℤ) 
  (h_x_odd : Odd x) (h_y_odd : Odd y) (h_x_y_diff : |x| ≠ |y|) :
  ∃ a b : ℤ, f a = f b ∧ (b - a = x ∨ b - a = y ∨ b - a = x + y ∨ b - a = x - y) := by
  sorry

end same_color_difference_l2791_279179


namespace exists_triangle_101_subdivisions_l2791_279172

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define a function to check if a triangle can be subdivided into n congruent triangles
def can_subdivide (t : Triangle) (n : ℕ) : Prop :=
  ∃ (m : ℕ), m^2 + 1 = n

-- Theorem statement
theorem exists_triangle_101_subdivisions :
  ∃ (t : Triangle), can_subdivide t 101 := by
sorry

end exists_triangle_101_subdivisions_l2791_279172


namespace algebraic_expansions_l2791_279195

theorem algebraic_expansions (x y : ℝ) :
  ((x + 2*y - 3) * (x - 2*y + 3) = x^2 - 4*y^2 + 12*y - 9) ∧
  ((2*x^3*y)^2 * (-2*x*y) + (-2*x^3*y)^3 / (2*x^2) = -12*x^7*y^3) :=
by sorry

end algebraic_expansions_l2791_279195


namespace planes_divide_space_l2791_279194

/-- The number of regions into which n planes can divide space -/
def R (n : ℕ) : ℚ := (n^3 + 5*n + 6) / 6

/-- Theorem stating that R(n) gives the correct number of regions for n planes -/
theorem planes_divide_space (n : ℕ) : 
  R n = (n^3 + 5*n + 6) / 6 := by sorry

end planes_divide_space_l2791_279194


namespace function_bound_l2791_279109

open Real

theorem function_bound (f : ℝ → ℝ) (m : ℝ) : 
  (∀ x ∈ Set.Ioo 0 (π/4), f x = sin (2*x) - Real.sqrt 3 * cos (2*x)) →
  (∀ x ∈ Set.Ioo 0 (π/4), |f x| < m) →
  m ≥ Real.sqrt 3 := by
sorry

end function_bound_l2791_279109


namespace football_team_yardage_l2791_279186

theorem football_team_yardage (initial_loss : ℤ) : 
  (initial_loss < 0) →  -- The team lost some yards initially
  (-initial_loss + 11 = 6) →  -- The team gained 11 yards and ended up with 6 yards progress
  initial_loss = -5 :=  -- The initial loss was 5 yards
by
  sorry

end football_team_yardage_l2791_279186


namespace intersection_of_M_and_N_l2791_279110

def M : Set ℝ := {x | x^2 - x - 2 = 0}
def N : Set ℝ := {-1, 0}

theorem intersection_of_M_and_N : M ∩ N = {-1} := by
  sorry

end intersection_of_M_and_N_l2791_279110


namespace reciprocal_of_golden_ratio_l2791_279136

theorem reciprocal_of_golden_ratio (φ : ℝ) :
  φ = (Real.sqrt 5 + 1) / 2 →
  1 / φ = (Real.sqrt 5 - 1) / 2 := by
  sorry

end reciprocal_of_golden_ratio_l2791_279136


namespace matrix_cube_sum_l2791_279167

/-- Definition of the matrix N -/
def N (a b c : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![a, c, b],
    ![c, b, a],
    ![b, a, c]]

/-- The theorem statement -/
theorem matrix_cube_sum (a b c : ℂ) :
  (N a b c)^2 = 1 → a * b * c = 1 → a^3 + b^3 + c^3 = 2 ∨ a^3 + b^3 + c^3 = 4 := by
  sorry

end matrix_cube_sum_l2791_279167


namespace unanswered_questions_l2791_279128

/-- Represents the scoring system and results for a math contest. -/
structure ContestScoring where
  total_questions : ℕ
  new_correct_points : ℕ
  new_unanswered_points : ℕ
  old_start_points : ℕ
  old_correct_points : ℕ
  old_wrong_points : ℕ
  new_score : ℕ
  old_score : ℕ

/-- Theorem stating that given the contest scoring system and Alice's scores,
    the number of unanswered questions is 8. -/
theorem unanswered_questions (cs : ContestScoring)
  (h1 : cs.total_questions = 30)
  (h2 : cs.new_correct_points = 6)
  (h3 : cs.new_unanswered_points = 3)
  (h4 : cs.old_start_points = 40)
  (h5 : cs.old_correct_points = 5)
  (h6 : cs.old_wrong_points = 2)
  (h7 : cs.new_score = 108)
  (h8 : cs.old_score = 94) :
  ∃ (c w u : ℕ), c + w + u = cs.total_questions ∧
                 cs.new_correct_points * c + cs.new_unanswered_points * u = cs.new_score ∧
                 cs.old_start_points + cs.old_correct_points * c - cs.old_wrong_points * w = cs.old_score ∧
                 u = 8 := by
  sorry

end unanswered_questions_l2791_279128


namespace length_of_24_l2791_279148

def length_of_integer (k : ℕ) : ℕ := sorry

theorem length_of_24 : 
  let k : ℕ := 24
  length_of_integer k = 4 := by sorry

end length_of_24_l2791_279148


namespace non_mundane_primes_characterization_l2791_279139

/-- A prime number is mundane if there exist positive integers a and b less than p/2 
    such that (ab - 1)/p is a positive integer. -/
def IsMundane (p : ℕ) : Prop :=
  ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ a < p / 2 ∧ b < p / 2 ∧ 
  ∃ k : ℕ, 0 < k ∧ k * p = a * b - 1

/-- The set of non-mundane primes -/
def NonMundanePrimes : Set ℕ := {2, 3, 5, 7, 13}

/-- Theorem: A prime number is not mundane if and only if it belongs to the set {2, 3, 5, 7, 13} -/
theorem non_mundane_primes_characterization (p : ℕ) (hp : Nat.Prime p) : 
  ¬ IsMundane p ↔ p ∈ NonMundanePrimes := by
  sorry

end non_mundane_primes_characterization_l2791_279139


namespace colors_in_box_is_seven_l2791_279119

/-- The number of colors in each color box, given the total number of pencils and people who bought a color box. -/
def colors_per_box (total_pencils : ℕ) (total_people : ℕ) : ℕ :=
  total_pencils / total_people

/-- Theorem stating that the number of colors in each color box is 7, given the problem conditions. -/
theorem colors_in_box_is_seven : 
  let total_people : ℕ := 6  -- Chloe and 5 friends
  let total_pencils : ℕ := 42
  colors_per_box total_pencils total_people = 7 := by
  sorry

#eval colors_per_box 42 6  -- This should output 7

end colors_in_box_is_seven_l2791_279119


namespace job_completion_time_l2791_279178

/-- The time taken for two workers to complete a job together -/
def job_time (rate1 rate2 : ℚ) : ℚ := 1 / (rate1 + rate2)

theorem job_completion_time 
  (rate_A rate_B rate_C : ℚ) 
  (h1 : rate_A + rate_B = 1 / 6)  -- A and B can do the job in 6 days
  (h2 : rate_B + rate_C = 1 / 10) -- B and C can do the job in 10 days
  (h3 : rate_A + rate_B + rate_C = 1 / 5) -- A, B, and C can do the job in 5 days
  : job_time rate_A rate_C = 15 / 2 := by
  sorry

end job_completion_time_l2791_279178


namespace arithmetic_equation_proof_l2791_279188

theorem arithmetic_equation_proof : 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 * 9 = 100 := by
  sorry

end arithmetic_equation_proof_l2791_279188


namespace square_painting_size_l2791_279126

/-- Given the total area of an art collection and the areas of non-square paintings,
    prove that the side length of each square painting is 6 feet. -/
theorem square_painting_size 
  (total_area : ℝ) 
  (num_square_paintings : ℕ) 
  (num_small_paintings : ℕ) 
  (small_painting_width small_painting_height : ℝ)
  (num_large_paintings : ℕ)
  (large_painting_width large_painting_height : ℝ) :
  total_area = 282 ∧ 
  num_square_paintings = 3 ∧
  num_small_paintings = 4 ∧
  small_painting_width = 2 ∧
  small_painting_height = 3 ∧
  num_large_paintings = 1 ∧
  large_painting_width = 10 ∧
  large_painting_height = 15 →
  ∃ (square_side : ℝ), 
    square_side = 6 ∧ 
    num_square_paintings * square_side^2 + 
    num_small_paintings * small_painting_width * small_painting_height +
    num_large_paintings * large_painting_width * large_painting_height = total_area :=
by sorry

end square_painting_size_l2791_279126


namespace min_d_value_l2791_279101

theorem min_d_value (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (horder : a < b ∧ b < c ∧ c < d)
  (hunique : ∃! (x y : ℝ), 3 * x + y = 3005 ∧ y = |x - a| + |x - b| + |x - c| + |x - d|) :
  d ≥ 999 ∧ ∃ (a' b' c' : ℕ), a' < b' ∧ b' < c' ∧ c' < 999 ∧
    ∃! (x y : ℝ), 3 * x + y = 3005 ∧ y = |x - a'| + |x - b'| + |x - c'| + |x - 999| :=
by
  sorry


end min_d_value_l2791_279101


namespace certain_number_problem_l2791_279187

theorem certain_number_problem : ∃ x : ℝ, 11*x + 12*x + 15*x + 11 = 125 ∧ x = 3 := by
  sorry

end certain_number_problem_l2791_279187


namespace point_distance_and_inequality_l2791_279154

/-- The value of m for which the point P(m, 3) is at distance 4 from the line 4x-3y+1=0
    and satisfies the inequality 2x+y<3 -/
theorem point_distance_and_inequality (m : ℝ) : 
  (abs (4 * m - 3 * 3 + 1) / Real.sqrt (4^2 + (-3)^2) = 4) ∧ 
  (2 * m + 3 < 3) → 
  m = -3 := by sorry

end point_distance_and_inequality_l2791_279154


namespace ladder_slide_approx_l2791_279144

noncomputable def ladder_slide (ladder_length : Real) (initial_distance : Real) (slip_distance : Real) : Real :=
  let initial_height := Real.sqrt (ladder_length^2 - initial_distance^2)
  let new_height := initial_height - slip_distance
  let new_distance := Real.sqrt (ladder_length^2 - new_height^2)
  new_distance - initial_distance

theorem ladder_slide_approx :
  ∃ (ε : Real), ε > 0 ∧ ε < 0.1 ∧ 
  |ladder_slide 30 11 5 - 3.7| < ε :=
sorry

end ladder_slide_approx_l2791_279144


namespace cucumber_weight_after_evaporation_l2791_279137

/-- Given 100 pounds of cucumbers initially composed of 99% water by weight,
    when the water composition changes to 98% by weight due to evaporation,
    the new total weight of the cucumbers is 50 pounds. -/
theorem cucumber_weight_after_evaporation
  (initial_weight : ℝ)
  (initial_water_percentage : ℝ)
  (final_water_percentage : ℝ)
  (h1 : initial_weight = 100)
  (h2 : initial_water_percentage = 0.99)
  (h3 : final_water_percentage = 0.98)
  : ∃ (final_weight : ℝ), final_weight = 50 ∧
    (1 - initial_water_percentage) * initial_weight =
    (1 - final_water_percentage) * final_weight :=
by sorry

end cucumber_weight_after_evaporation_l2791_279137


namespace purple_cell_count_l2791_279177

/-- Represents the state of a cell on the board -/
inductive CellState
| Unpainted
| Blue
| Red
| Purple

/-- Represents a 2x2 square on the board -/
structure Square :=
  (topLeft : Nat × Nat)

/-- Represents the game board -/
def Board := Fin 2022 → Fin 2022 → CellState

/-- Represents a move in the game -/
structure Move :=
  (square : Square)
  (color : CellState)

/-- The game state -/
structure GameState :=
  (board : Board)
  (moves : List Move)

/-- Count the number of purple cells on the board -/
def countPurpleCells (board : Board) : Nat :=
  sorry

/-- Check if a move is valid -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  sorry

/-- Apply a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

theorem purple_cell_count (finalState : GameState) 
  (h1 : ∀ move ∈ finalState.moves, isValidMove (applyMove finalState move) move)
  (h2 : ∀ i j, finalState.board i j ≠ CellState.Unpainted) :
  countPurpleCells finalState.board = 2022 * 2020 ∨ 
  countPurpleCells finalState.board = 2020 * 2020 :=
sorry

end purple_cell_count_l2791_279177


namespace seating_arrangements_count_l2791_279171

/-- The number of ways to arrange n distinct objects. -/
def arrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange 3 adults and 3 children in 6 seats,
    such that no two people of the same type sit together. -/
def seating_arrangements : ℕ :=
  2 * arrangements 3 * arrangements 3

/-- Theorem stating that the number of seating arrangements is 72. -/
theorem seating_arrangements_count :
  seating_arrangements = 72 := by sorry

end seating_arrangements_count_l2791_279171


namespace vanessa_large_orders_l2791_279185

/-- The number of grams of packing peanuts needed for a large order -/
def large_order_peanuts : ℕ := 200

/-- The number of grams of packing peanuts needed for a small order -/
def small_order_peanuts : ℕ := 50

/-- The total number of grams of packing peanuts used -/
def total_peanuts_used : ℕ := 800

/-- The number of small orders sent -/
def num_small_orders : ℕ := 4

/-- The number of large orders sent -/
def num_large_orders : ℕ := 3

theorem vanessa_large_orders :
  num_large_orders * large_order_peanuts + num_small_orders * small_order_peanuts = total_peanuts_used :=
by sorry

end vanessa_large_orders_l2791_279185


namespace inclined_line_and_triangle_l2791_279112

/-- A line passing through a point with a given angle of inclination -/
structure InclinedLine where
  point : ℝ × ℝ
  angle : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem about an inclined line and the triangle it forms with the axes -/
theorem inclined_line_and_triangle (l : InclinedLine) 
    (h1 : l.point = (0, -2))
    (h2 : l.angle = Real.pi / 3) : 
  ∃ (eq : LineEquation) (area : ℝ),
    eq.a = Real.sqrt 3 ∧ 
    eq.b = -1 ∧ 
    eq.c = -2 ∧
    area = (2 * Real.sqrt 3) / 3 := by
  sorry

end inclined_line_and_triangle_l2791_279112


namespace unique_function_theorem_l2791_279170

/-- A function from positive integers to positive integers -/
def PositiveIntFunction := ℕ+ → ℕ+

/-- Condition: f(n) is a perfect square for all n -/
def IsPerfectSquare (f : PositiveIntFunction) : Prop :=
  ∀ n : ℕ+, ∃ k : ℕ+, f n = k * k

/-- Condition: f(m+n) = f(m) + f(n) + 2mn for all m, n -/
def SatisfiesFunctionalEquation (f : PositiveIntFunction) : Prop :=
  ∀ m n : ℕ+, f (m + n) = f m + f n + 2 * m * n

/-- Theorem: The only function satisfying both conditions is f(n) = n² -/
theorem unique_function_theorem (f : PositiveIntFunction) 
  (h1 : IsPerfectSquare f) (h2 : SatisfiesFunctionalEquation f) :
  ∀ n : ℕ+, f n = n * n :=
by sorry

end unique_function_theorem_l2791_279170


namespace max_band_members_l2791_279140

/-- Represents a rectangular band formation --/
structure BandFormation where
  rows : ℕ
  membersPerRow : ℕ

/-- Checks if a band formation is valid according to the problem conditions --/
def isValidFormation (f : BandFormation) (totalMembers : ℕ) : Prop :=
  totalMembers < 100 ∧
  totalMembers = f.rows * f.membersPerRow + 3 ∧
  totalMembers = (f.rows - 3) * (f.membersPerRow + 2)

/-- Theorem stating the maximum number of band members --/
theorem max_band_members :
  ∃ (m : ℕ) (f : BandFormation),
    isValidFormation f m ∧
    ∀ (n : ℕ) (g : BandFormation), isValidFormation g n → n ≤ m :=
  by sorry

end max_band_members_l2791_279140


namespace arithmetic_sequence_length_l2791_279106

theorem arithmetic_sequence_length :
  ∀ (a₁ aₙ d n : ℤ),
    a₁ = -3 →
    aₙ = 45 →
    d = 4 →
    aₙ = a₁ + (n - 1) * d →
    n = 13 := by
  sorry

end arithmetic_sequence_length_l2791_279106


namespace staircase_arrangement_7_steps_l2791_279158

/-- The number of ways 3 people can stand on a staircase with n steps,
    where each step can accommodate at most 2 people and
    the positions of people on the same step are not distinguished. -/
def staircase_arrangements (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The number of ways 3 people can stand on a 7-step staircase is 336,
    given that each step can accommodate at most 2 people and
    the positions of people on the same step are not distinguished. -/
theorem staircase_arrangement_7_steps :
  staircase_arrangements 7 = 336 := by
  sorry

end staircase_arrangement_7_steps_l2791_279158


namespace point_in_fourth_quadrant_l2791_279199

def point : ℝ × ℝ := (2, -3)

def is_in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : is_in_fourth_quadrant point := by
  sorry

end point_in_fourth_quadrant_l2791_279199


namespace quasi_pythagorean_prime_divisor_l2791_279182

theorem quasi_pythagorean_prime_divisor 
  (a b c : ℕ+) 
  (h : c.val ^ 2 = a.val ^ 2 + a.val * b.val + b.val ^ 2) : 
  ∃ (p : ℕ), p > 5 ∧ Nat.Prime p ∧ p ∣ c.val :=
sorry

end quasi_pythagorean_prime_divisor_l2791_279182


namespace polygon_area_is_1800_l2791_279193

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The vertices of the polygon -/
def vertices : List Point := [
  ⟨0, 0⟩, ⟨15, 0⟩, ⟨45, 30⟩, ⟨45, 45⟩, ⟨30, 45⟩, ⟨0, 15⟩
]

/-- Calculate the area of a polygon given its vertices -/
def polygonArea (vs : List Point) : ℝ :=
  sorry

/-- The theorem stating that the area of the given polygon is 1800 square units -/
theorem polygon_area_is_1800 : polygonArea vertices = 1800 := by
  sorry

end polygon_area_is_1800_l2791_279193


namespace sqrt_problem_l2791_279159

theorem sqrt_problem (x : ℝ) (h : (Real.sqrt x - 8) / 13 = 6) :
  ⌊(x^2 - 45) / 23⌋ = 2380011 := by sorry

end sqrt_problem_l2791_279159


namespace largest_n_divisibility_l2791_279143

theorem largest_n_divisibility : ∃ (n : ℕ), 
  (∀ m : ℕ, m > n → ¬((m + 20) ∣ (m^3 + 200))) ∧ 
  ((n + 20) ∣ (n^3 + 200)) ∧ 
  n = 7780 := by
  sorry

end largest_n_divisibility_l2791_279143


namespace triangle_inequality_l2791_279156

theorem triangle_inequality (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)
  (h7 : a + b + c = 1) :
  a^2 + b^2 + c^2 + 4*a*b*c < 1/2 := by
sorry

end triangle_inequality_l2791_279156


namespace cos_180_degrees_l2791_279160

theorem cos_180_degrees : Real.cos (π) = -1 := by sorry

end cos_180_degrees_l2791_279160


namespace stating_count_quadrilaterals_correct_l2791_279164

/-- 
For a convex n-gon, count_quadrilaterals n returns the number of ways to choose 
four vertices that form a quadrilateral with sides that are diagonals of the n-gon.
-/
def count_quadrilaterals (n : ℕ) : ℕ := 
  n / 4 * Nat.choose (n - 5) 3

/-- 
Theorem stating that count_quadrilaterals correctly counts the number of ways 
to choose four vertices forming a quadrilateral with diagonal sides in an n-gon.
-/
theorem count_quadrilaterals_correct (n : ℕ) : 
  count_quadrilaterals n = n / 4 * Nat.choose (n - 5) 3 := by
  sorry

#eval count_quadrilaterals 10  -- Example evaluation

end stating_count_quadrilaterals_correct_l2791_279164


namespace conic_equation_not_parabola_l2791_279117

/-- Represents a conic section equation of the form mx² + ny² = 1 -/
structure ConicEquation where
  m : ℝ
  n : ℝ

/-- Defines the possible types of conic sections -/
inductive ConicType
  | Circle
  | Ellipse
  | Hyperbola
  | Parabola

/-- States that a conic equation cannot represent a parabola -/
theorem conic_equation_not_parabola (eq : ConicEquation) : 
  ∃ (t : ConicType), t ≠ ConicType.Parabola ∧ 
  (∀ (x y : ℝ), eq.m * x^2 + eq.n * y^2 = 1 → 
    ∃ (a b c d e f : ℝ), a * x^2 + b * y^2 + c * x * y + d * x + e * y + f = 0) :=
sorry

end conic_equation_not_parabola_l2791_279117


namespace cubic_root_sum_l2791_279198

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 15*p^2 + 25*p - 10 = 0 →
  q^3 - 15*q^2 + 25*q - 10 = 0 →
  r^3 - 15*r^2 + 25*r - 10 = 0 →
  p/(2/p + q*r) + q/(2/q + r*p) + r/(2/r + p*q) = 175/12 := by
sorry

end cubic_root_sum_l2791_279198


namespace quadratic_root_conjugate_l2791_279180

theorem quadratic_root_conjugate (a b c : ℚ) :
  (a ≠ 0) →
  (a * (3 + Real.sqrt 2)^2 + b * (3 + Real.sqrt 2) + c = 0) →
  (a * (3 - Real.sqrt 2)^2 + b * (3 - Real.sqrt 2) + c = 0) :=
by sorry

end quadratic_root_conjugate_l2791_279180


namespace first_five_terms_sum_l2791_279135

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem first_five_terms_sum :
  let a : ℚ := 1
  let r : ℚ := 1/2
  let n : ℕ := 5
  geometric_series_sum a r n = 31/16 := by
sorry

end first_five_terms_sum_l2791_279135


namespace sum_four_digit_even_distinct_mod_1000_l2791_279116

/-- A function that generates all four-digit positive integers with distinct even digits -/
def fourDigitEvenDistinct : List Nat := sorry

/-- The sum of all four-digit positive integers with distinct even digits -/
def sumFourDigitEvenDistinct : Nat := (fourDigitEvenDistinct.map id).sum

/-- Theorem: The sum of all four-digit positive integers with distinct even digits,
    when divided by 1000, leaves a remainder of 560 -/
theorem sum_four_digit_even_distinct_mod_1000 :
  sumFourDigitEvenDistinct % 1000 = 560 := by sorry

end sum_four_digit_even_distinct_mod_1000_l2791_279116


namespace hundredth_bracket_numbers_l2791_279152

def bracket_sequence (n : ℕ) : ℕ := 
  if n % 4 = 1 then 1
  else if n % 4 = 2 then 2
  else if n % 4 = 3 then 3
  else 1

def first_number_in_group (group : ℕ) : ℕ := 2 * group - 1

theorem hundredth_bracket_numbers :
  let group := (100 - 1) / 3 + 1
  let first_num := first_number_in_group group - 2
  bracket_sequence 100 = 2 ∧ first_num = 65 ∧ first_num + 2 = 67 := by
  sorry

end hundredth_bracket_numbers_l2791_279152


namespace log_equation_implies_sum_l2791_279151

theorem log_equation_implies_sum (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h : (Real.log x / Real.log 2)^4 + (Real.log y / Real.log 3)^4 + 8 = 
       8 * (Real.log x / Real.log 2) * (Real.log y / Real.log 3)) :
  x^Real.sqrt 2 + y^Real.sqrt 2 = 13 := by
sorry

end log_equation_implies_sum_l2791_279151


namespace carrots_theorem_l2791_279127

/-- The number of carrots Sandy grew -/
def sandy_carrots : ℕ := 6

/-- The number of carrots Sam grew -/
def sam_carrots : ℕ := 3

/-- The total number of carrots grown -/
def total_carrots : ℕ := sandy_carrots + sam_carrots

theorem carrots_theorem : total_carrots = 9 := by sorry

end carrots_theorem_l2791_279127


namespace tangent_circles_radius_l2791_279189

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_O2 (x y r : ℝ) : Prop := (x - 3)^2 + y^2 = r^2

-- Define the tangency condition
def are_tangent (r : ℝ) : Prop := ∃ (x y : ℝ), circle_O1 x y ∧ circle_O2 x y r

-- Theorem statement
theorem tangent_circles_radius (r : ℝ) (h1 : r > 0) (h2 : are_tangent r) : r = 4 := by
  sorry

end tangent_circles_radius_l2791_279189


namespace smallest_prime_factor_in_C_l2791_279145

def C : Set Nat := {37, 39, 42, 43, 47}

theorem smallest_prime_factor_in_C :
  ∃ (n : Nat), n ∈ C ∧ (∀ (m : Nat), m ∈ C → Nat.minFac n ≤ Nat.minFac m) ∧ n = 42 := by
  sorry

end smallest_prime_factor_in_C_l2791_279145


namespace round_trip_percentage_l2791_279108

/-- Represents the distribution of passenger types and classes on a transatlantic ship crossing -/
structure PassengerDistribution where
  /-- Percentage of Type A passengers (round-trip with car) -/
  type_a_percent : ℝ
  /-- Percentage of round-trip passengers not taking cars -/
  no_car_percent : ℝ
  /-- Percentage of round-trip passengers in luxury class -/
  luxury_percent : ℝ
  /-- Percentage of round-trip passengers in economy class -/
  economy_percent : ℝ
  /-- Percentage of Type C passengers in economy class -/
  type_c_economy_percent : ℝ

/-- Theorem stating that given the passenger distribution, the percentage of round-trip passengers is 40% -/
theorem round_trip_percentage (pd : PassengerDistribution)
  (h1 : pd.type_a_percent = 0.2)
  (h2 : pd.no_car_percent = 0.5)
  (h3 : pd.luxury_percent = 0.3)
  (h4 : pd.economy_percent = 0.7)
  (h5 : pd.type_c_economy_percent = 0.4)
  : ℝ :=
  by sorry

end round_trip_percentage_l2791_279108


namespace ratio_first_term_to_common_difference_l2791_279166

/-- An arithmetic progression where the sum of the first 15 terms is three times the sum of the first 8 terms -/
def ArithmeticProgression (a d : ℝ) : Prop :=
  let S : ℕ → ℝ := λ n => n / 2 * (2 * a + (n - 1) * d)
  S 15 = 3 * S 8

theorem ratio_first_term_to_common_difference 
  {a d : ℝ} (h : ArithmeticProgression a d) : 
  a / d = 7 / 3 := by
sorry

end ratio_first_term_to_common_difference_l2791_279166


namespace hcf_of_two_numbers_l2791_279142

theorem hcf_of_two_numbers (a b : ℕ) : 
  (a > 0) → 
  (b > 0) → 
  (a ≤ b) → 
  (b = 1071) → 
  (∃ k : ℕ, Nat.lcm a b = k * 11 * 17) → 
  Nat.gcd a b = 1 := by
  sorry

end hcf_of_two_numbers_l2791_279142


namespace solution_set_f_solution_set_g_l2791_279107

-- Define the quadratic functions
def f (x : ℝ) : ℝ := x^2 - 3*x - 4
def g (x : ℝ) : ℝ := x^2 - x - 6

-- Define the solution sets
def S₁ : Set ℝ := {x | -1 < x ∧ x < 4}
def S₂ : Set ℝ := {x | x < -2 ∨ x > 3}

-- Theorem for the first inequality
theorem solution_set_f : {x : ℝ | f x < 0} = S₁ := by sorry

-- Theorem for the second inequality
theorem solution_set_g : {x : ℝ | g x > 0} = S₂ := by sorry

end solution_set_f_solution_set_g_l2791_279107


namespace expand_expression_l2791_279183

theorem expand_expression (x : ℝ) : (x + 3) * (4 * x - 8) = 4 * x^2 + 4 * x - 24 := by
  sorry

end expand_expression_l2791_279183


namespace square_sum_lower_bound_l2791_279132

theorem square_sum_lower_bound (x y θ : ℝ) 
  (h : (x * Real.cos θ + y * Real.sin θ)^2 + x * Real.sin θ - y * Real.cos θ = 1) : 
  x^2 + y^2 ≥ 3/4 := by
  sorry

end square_sum_lower_bound_l2791_279132
