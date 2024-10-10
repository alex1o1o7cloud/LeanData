import Mathlib

namespace factorization_proof_l3134_313409

theorem factorization_proof (x : ℝ) : 15 * x^2 + 10 * x - 5 = 5 * (3 * x - 1) * (x + 1) := by
  sorry

end factorization_proof_l3134_313409


namespace part_one_part_two_l3134_313418

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - m| - 3

-- Part I
theorem part_one (m : ℝ) :
  (∀ x, f m x ≥ 0 ↔ x ≤ -2 ∨ x ≥ 4) → m = 1 := by sorry

-- Part II
theorem part_two :
  ∀ t, (∃ x, f 1 x ≥ t + |2 - x|) → t ≤ -2 := by sorry

end part_one_part_two_l3134_313418


namespace sphere_volume_increase_l3134_313401

theorem sphere_volume_increase (r₁ r₂ : ℝ) (h : r₂ = 2 * r₁) : 
  (4 / 3) * π * r₂^3 = 8 * ((4 / 3) * π * r₁^3) := by
  sorry

end sphere_volume_increase_l3134_313401


namespace boys_in_class_l3134_313439

theorem boys_in_class (total : ℕ) (diff : ℕ) (boys : ℕ) : 
  total = 485 →
  diff = 69 →
  total = boys + (boys + diff) →
  boys = 208 := by
sorry

end boys_in_class_l3134_313439


namespace odd_square_not_sum_of_five_odd_squares_l3134_313406

theorem odd_square_not_sum_of_five_odd_squares :
  ∀ n a b c d e : ℤ,
  Odd n → Odd a → Odd b → Odd c → Odd d → Odd e →
  ¬(n^2 ≡ a^2 + b^2 + c^2 + d^2 + e^2 [ZMOD 8]) :=
by sorry

end odd_square_not_sum_of_five_odd_squares_l3134_313406


namespace min_value_of_m_l3134_313440

theorem min_value_of_m (a b c m : ℝ) (h1 : a > b) (h2 : b > c) (h3 : m > 0)
  (h4 : ∀ a b c, a > b ∧ b > c → 1 / (a - b) + m / (b - c) ≥ 9 / (a - c)) :
  m ≥ 4 := by
  sorry

end min_value_of_m_l3134_313440


namespace sweater_price_theorem_l3134_313421

/-- The marked price of a sweater in yuan -/
def marked_price : ℝ := 150

/-- The selling price as a percentage of the marked price -/
def selling_percentage : ℝ := 0.8

/-- The profit percentage -/
def profit_percentage : ℝ := 0.2

/-- The purchase price of the sweater in yuan -/
def purchase_price : ℝ := 100

theorem sweater_price_theorem : 
  selling_percentage * marked_price = purchase_price * (1 + profit_percentage) :=
sorry

end sweater_price_theorem_l3134_313421


namespace definite_integral_2x_l3134_313426

theorem definite_integral_2x : ∫ x in (0)..(π/2), 2*x = π^2/4 := by
  sorry

end definite_integral_2x_l3134_313426


namespace profit_ratio_problem_l3134_313481

/-- The profit ratio problem -/
theorem profit_ratio_problem (profit_3_shirts profit_7_shirts_3_sandals : ℚ) 
  (h1 : profit_3_shirts = 21)
  (h2 : profit_7_shirts_3_sandals = 175) :
  (2 * profit_3_shirts) = ((profit_7_shirts_3_sandals - (7 / 3) * profit_3_shirts) / 3 * 2) :=
by sorry

end profit_ratio_problem_l3134_313481


namespace fraction_equation_solution_l3134_313427

theorem fraction_equation_solution :
  ∃! x : ℚ, (x + 5) / (x - 3) = (x - 2) / (x + 4) :=
by
  use -1
  constructor
  · -- Prove that x = -1 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

end fraction_equation_solution_l3134_313427


namespace ceiling_minus_x_l3134_313452

theorem ceiling_minus_x (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 1) :
  ∃ α : ℝ, 0 < α ∧ α < 1 ∧ x = ⌊x⌋ + α ∧ ⌈x⌉ - x = 1 - α :=
sorry

end ceiling_minus_x_l3134_313452


namespace addition_subtraction_elimination_not_factorization_l3134_313438

-- Define the type for factorization methods
inductive FactorizationMethod
  | TakeOutCommonFactor
  | CrossMultiplication
  | Formula
  | AdditionSubtractionElimination

-- Define a predicate to check if a method is a factorization method
def is_factorization_method : FactorizationMethod → Prop
  | FactorizationMethod.TakeOutCommonFactor => true
  | FactorizationMethod.CrossMultiplication => true
  | FactorizationMethod.Formula => true
  | FactorizationMethod.AdditionSubtractionElimination => false

-- Theorem statement
theorem addition_subtraction_elimination_not_factorization :
  ¬(is_factorization_method FactorizationMethod.AdditionSubtractionElimination) :=
by sorry

end addition_subtraction_elimination_not_factorization_l3134_313438


namespace hogwarts_total_students_l3134_313480

-- Define the given conditions
def total_participants : ℕ := 246
def total_boys : ℕ := 255

-- Define the relationship between participating boys and non-participating girls
def boys_participating_girls_not (total_students : ℕ) : Prop :=
  ∃ (boys_participating : ℕ) (girls_not_participating : ℕ),
    boys_participating = girls_not_participating + 11 ∧
    boys_participating ≤ total_boys ∧
    girls_not_participating ≤ total_students - total_boys

-- Theorem statement
theorem hogwarts_total_students : 
  ∃ (total_students : ℕ),
    total_students = 490 ∧
    boys_participating_girls_not total_students :=
by
  sorry


end hogwarts_total_students_l3134_313480


namespace square_sum_from_means_l3134_313419

theorem square_sum_from_means (x y : ℝ) 
  (h_am : (x + y) / 2 = 20) 
  (h_gm : Real.sqrt (x * y) = Real.sqrt 110) : 
  x^2 + y^2 = 1380 := by
sorry

end square_sum_from_means_l3134_313419


namespace cycle_original_price_l3134_313496

/-- Given a cycle sold at a 15% loss for Rs. 1190, prove that the original price was Rs. 1400 -/
theorem cycle_original_price (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1190)
  (h2 : loss_percentage = 15) : 
  ∃ (original_price : ℝ), 
    original_price * (1 - loss_percentage / 100) = selling_price ∧ 
    original_price = 1400 := by
  sorry

end cycle_original_price_l3134_313496


namespace similar_triangle_longest_side_l3134_313433

def triangle_sides (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem similar_triangle_longest_side 
  (a b c : ℝ) 
  (h_triangle : triangle_sides a b c) 
  (h_sides : a = 5 ∧ b = 12 ∧ c = 13) 
  (k : ℝ) 
  (h_similar : k > 0)
  (h_perimeter : k * (a + b + c) = 150) :
  k * max a (max b c) = 65 := by
sorry

end similar_triangle_longest_side_l3134_313433


namespace divisibility_of_sum_and_powers_l3134_313463

theorem divisibility_of_sum_and_powers (a b c : ℤ) 
  (h : 6 ∣ (a + b + c)) : 6 ∣ (a^5 + b^3 + c) := by
  sorry

end divisibility_of_sum_and_powers_l3134_313463


namespace fraction_sum_equals_decimal_l3134_313405

theorem fraction_sum_equals_decimal : 
  (3 : ℚ) / 100 + 5 / 1000 + 8 / 10000 + 2 / 100000 = 0.03582 := by
  sorry

end fraction_sum_equals_decimal_l3134_313405


namespace mean_of_added_numbers_l3134_313447

theorem mean_of_added_numbers (original_count : ℕ) (original_mean : ℚ) 
  (new_count : ℕ) (new_mean : ℚ) (added_count : ℕ) : 
  original_count = 8 →
  original_mean = 72 →
  new_count = 11 →
  new_mean = 85 →
  added_count = 3 →
  (new_count * new_mean - original_count * original_mean) / added_count = 119 + 2/3 := by
  sorry

end mean_of_added_numbers_l3134_313447


namespace clara_cookie_sales_l3134_313417

/-- Represents the number of cookies in a box for each type -/
def cookies_per_box : Fin 3 → ℕ
  | 0 => 12
  | 1 => 20
  | 2 => 16

/-- Represents the number of boxes sold for each type -/
def boxes_sold : Fin 3 → ℕ
  | 0 => 50
  | 1 => 80
  | 2 => 70

/-- Calculates the total number of cookies sold -/
def total_cookies_sold : ℕ :=
  (cookies_per_box 0 * boxes_sold 0) +
  (cookies_per_box 1 * boxes_sold 1) +
  (cookies_per_box 2 * boxes_sold 2)

theorem clara_cookie_sales :
  total_cookies_sold = 3320 := by
  sorry

end clara_cookie_sales_l3134_313417


namespace student_number_problem_l3134_313482

theorem student_number_problem :
  ∃ x : ℝ, 2 * x - 138 = 104 ∧ x = 121 := by sorry

end student_number_problem_l3134_313482


namespace fenced_area_blocks_l3134_313477

def total_blocks : ℕ := 344
def building_blocks : ℕ := 80
def farmhouse_blocks : ℕ := 123
def remaining_blocks : ℕ := 84

theorem fenced_area_blocks :
  total_blocks - building_blocks - farmhouse_blocks - remaining_blocks = 57 := by
  sorry

end fenced_area_blocks_l3134_313477


namespace min_value_sum_squares_l3134_313415

theorem min_value_sum_squares (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 4) 
  (h2 : e * f * g * h = 9) : 
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2 ≥ 24 := by
  sorry

end min_value_sum_squares_l3134_313415


namespace profit_percentage_before_decrease_l3134_313413

/-- Proves that the profit percentage before the decrease in manufacturing cost was 20% --/
theorem profit_percentage_before_decrease
  (selling_price : ℝ)
  (manufacturing_cost_before : ℝ)
  (manufacturing_cost_after : ℝ)
  (h1 : manufacturing_cost_before = 80)
  (h2 : manufacturing_cost_after = 50)
  (h3 : selling_price - manufacturing_cost_after = 0.5 * selling_price) :
  (selling_price - manufacturing_cost_before) / selling_price = 0.2 := by
  sorry

end profit_percentage_before_decrease_l3134_313413


namespace prime_divides_sum_l3134_313460

theorem prime_divides_sum (a b c : ℕ+) (p : ℕ) 
  (h1 : a ^ 3 + 4 * b + c = a * b * c)
  (h2 : a ≥ c)
  (h3 : p = a ^ 2 + 2 * a + 2)
  (h4 : Nat.Prime p) :
  p ∣ (a + 2 * b + 2) := by
sorry

end prime_divides_sum_l3134_313460


namespace max_gcd_of_sequence_l3134_313461

theorem max_gcd_of_sequence (n : ℕ) : 
  Nat.gcd ((8^n - 1) / 7) ((8^(n+1) - 1) / 7) = 1 :=
by sorry

end max_gcd_of_sequence_l3134_313461


namespace twin_brothers_age_l3134_313489

theorem twin_brothers_age :
  ∀ (x : ℕ), 
    (x * x + 9 = (x + 1) * (x + 1)) → 
    x = 4 := by
  sorry

end twin_brothers_age_l3134_313489


namespace min_tablets_to_extract_l3134_313432

/-- Represents the number of tablets of each medicine type in the box -/
def tablets_per_type : ℕ := 10

/-- Represents the minimum number of tablets of each type we want to guarantee -/
def min_tablets_per_type : ℕ := 2

/-- Theorem: The minimum number of tablets to extract to guarantee at least two of each type -/
theorem min_tablets_to_extract :
  tablets_per_type + min_tablets_per_type = 12 := by sorry

end min_tablets_to_extract_l3134_313432


namespace variance_of_dataset_l3134_313473

def dataset : List ℝ := [5, 7, 7, 8, 10, 11]

/-- The variance of the dataset [5, 7, 7, 8, 10, 11] is 4 -/
theorem variance_of_dataset : 
  let n : ℝ := dataset.length
  let mean : ℝ := (dataset.sum) / n
  let variance : ℝ := (dataset.map (λ x => (x - mean)^2)).sum / n
  variance = 4 := by sorry

end variance_of_dataset_l3134_313473


namespace thirty_percent_of_number_l3134_313411

theorem thirty_percent_of_number (x : ℝ) (h : (3 / 7) * x = 0.4 * x + 12) : 0.3 * x = 126 := by
  sorry

end thirty_percent_of_number_l3134_313411


namespace root_equation_problem_l3134_313449

theorem root_equation_problem (a b m p r : ℝ) : 
  (a^2 - m*a + 4 = 0) →
  (b^2 - m*b + 4 = 0) →
  ((a - 1/b)^2 - p*(a - 1/b) + r = 0) →
  ((b - 1/a)^2 - p*(b - 1/a) + r = 0) →
  r = 9/4 := by sorry

end root_equation_problem_l3134_313449


namespace bulls_win_probability_l3134_313414

-- Define the probability of Heat winning a single game
def heat_win_prob : ℚ := 3/4

-- Define the probability of Bulls winning a single game
def bulls_win_prob : ℚ := 1 - heat_win_prob

-- Define the number of games needed to win the series
def games_to_win : ℕ := 4

-- Define the total number of games in a full series
def total_games : ℕ := 7

-- Define the function to calculate the probability of Bulls winning in 7 games
def bulls_win_in_seven : ℚ :=
  -- Probability of 3-3 tie after 6 games
  (Nat.choose 6 3 : ℚ) * bulls_win_prob^3 * heat_win_prob^3 *
  -- Probability of Bulls winning the 7th game
  bulls_win_prob

-- Theorem statement
theorem bulls_win_probability :
  bulls_win_in_seven = 540 / 16384 := by sorry

end bulls_win_probability_l3134_313414


namespace digit_sum_property_l3134_313478

/-- A function that checks if a natural number has no zero digits -/
def has_no_zero_digits (n : ℕ) : Prop := sorry

/-- A function that generates all digit permutations of a natural number -/
def digit_permutations (n : ℕ) : Finset ℕ := sorry

/-- A function that checks if a natural number is composed solely of ones -/
def all_ones (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number has at least one digit 5 or greater -/
def has_digit_ge_5 (n : ℕ) : Prop := sorry

theorem digit_sum_property (n : ℕ) :
  has_no_zero_digits n →
  ∃ (p₁ p₂ p₃ : ℕ), p₁ ∈ digit_permutations n ∧ 
                    p₂ ∈ digit_permutations n ∧ 
                    p₃ ∈ digit_permutations n ∧
                    all_ones (n + p₁ + p₂ + p₃) →
  has_digit_ge_5 n :=
sorry

end digit_sum_property_l3134_313478


namespace collinear_probability_4x5_l3134_313444

/-- Represents a grid of dots -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- Counts the number of sets of 4 collinear dots in a grid -/
def collinearSets (g : Grid) : ℕ := sorry

/-- The probability of selecting 4 collinear dots from a grid -/
def collinearProbability (g : Grid) : ℚ :=
  (collinearSets g : ℚ) / choose (g.rows * g.cols) 4

theorem collinear_probability_4x5 :
  let g : Grid := ⟨4, 5⟩
  collinearProbability g = 9 / 4845 := by sorry

end collinear_probability_4x5_l3134_313444


namespace tiffany_bags_theorem_l3134_313492

/-- The total number of bags Tiffany collected over three days -/
def total_bags (initial : ℕ) (day2 : ℕ) (day3 : ℕ) : ℕ :=
  initial + day2 + day3

/-- Theorem stating that Tiffany's total bags equals 20 given the initial conditions -/
theorem tiffany_bags_theorem :
  total_bags 10 3 7 = 20 := by
  sorry

end tiffany_bags_theorem_l3134_313492


namespace admission_methods_l3134_313498

theorem admission_methods (n : ℕ) (k : ℕ) (s : ℕ) : 
  n = 8 → k = 2 → s = 3 → (n.choose k) * s = 84 :=
by sorry

end admission_methods_l3134_313498


namespace base_conversion_3050_l3134_313423

def base_10_to_base_8 (n : ℕ) : ℕ :=
  5000 + 700 + 50 + 2

theorem base_conversion_3050 :
  base_10_to_base_8 3050 = 5752 := by
  sorry

end base_conversion_3050_l3134_313423


namespace tangent_line_circle_l3134_313435

/-- The line 4x - 3y = 0 is tangent to the circle x^2 + y^2 - 2x + ay + 1 = 0 if and only if a = -1 or a = 4 -/
theorem tangent_line_circle (a : ℝ) : 
  (∀ x y : ℝ, (4 * x - 3 * y = 0 ∧ x^2 + y^2 - 2*x + a*y + 1 = 0) → 
    (∀ x' y' : ℝ, x'^2 + y'^2 - 2*x' + a*y' + 1 = 0 → (x = x' ∧ y = y'))) ↔ 
  (a = -1 ∨ a = 4) :=
sorry

end tangent_line_circle_l3134_313435


namespace x_can_be_negative_one_l3134_313457

theorem x_can_be_negative_one : ∃ (x : ℝ), x = -1 ∧ x^2 ∈ ({0, 1, x} : Set ℝ) := by
  sorry

end x_can_be_negative_one_l3134_313457


namespace min_value_theorem_l3134_313487

/-- Given a function y = a^(1-x) where a > 0 and a ≠ 1, 
    and a point A that lies on both the graph of the function and the line mx + ny - 1 = 0,
    where mn > 0, prove that the minimum value of 1/m + 2/n is 3 + 2√2. -/
theorem min_value_theorem (a : ℝ) (m n : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) (h3 : m * n > 0) (h4 : m + n = 1) :
  (∀ m' n', m' * n' > 0 → m' + n' = 1 → 1 / m' + 2 / n' ≥ 3 + 2 * Real.sqrt 2) ∧
  (∃ m' n', m' * n' > 0 ∧ m' + n' = 1 ∧ 1 / m' + 2 / n' = 3 + 2 * Real.sqrt 2) :=
by sorry

end min_value_theorem_l3134_313487


namespace trigonometric_expression_equality_trigonometric_fraction_simplification_l3134_313491

-- Part 1
theorem trigonometric_expression_equality : 
  2 * Real.cos (π / 2) + Real.tan (π / 4) + 3 * Real.sin 0 + (Real.cos (π / 3))^2 + Real.sin (3 * π / 2) = 1 / 4 := by
  sorry

-- Part 2
theorem trigonometric_fraction_simplification (θ : ℝ) : 
  (Real.sin (2 * π - θ) * Real.cos (π + θ) * Real.cos (π / 2 + θ) * Real.cos (11 * π / 2 - θ)) /
  (Real.cos (π - θ) * Real.sin (3 * π - θ) * Real.sin (-π - θ) * Real.sin (9 * π / 2 + θ)) = -Real.tan θ := by
  sorry

end trigonometric_expression_equality_trigonometric_fraction_simplification_l3134_313491


namespace colored_paper_usage_l3134_313479

theorem colored_paper_usage (initial_sheets : ℕ) (sheets_used : ℕ) : 
  initial_sheets = 82 →
  initial_sheets - sheets_used = sheets_used - 6 →
  sheets_used = 44 := by
  sorry

end colored_paper_usage_l3134_313479


namespace circle_and_tangent_line_equations_l3134_313434

/-- A circle with center C on the line x - y + 1 = 0 passing through points (1, 1) and (2, -2) -/
structure CircleC where
  center : ℝ × ℝ
  center_on_line : center.1 - center.2 + 1 = 0
  passes_through_A : (center.1 - 1)^2 + (center.2 - 1)^2 = (center.1 - 2)^2 + (center.2 + 2)^2

/-- The standard equation of the circle and its tangent line -/
def circle_equation (c : CircleC) : Prop :=
  ∀ (x y : ℝ), (x + 3)^2 + (y + 2)^2 = 25 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = (c.center.1 - 1)^2 + (c.center.2 - 1)^2

def tangent_line_equation (c : CircleC) : Prop :=
  ∀ (x y : ℝ), 4*x + 3*y - 7 = 0 ↔ 
    ((x - 1) * (c.center.1 - 1) + (y - 1) * (c.center.2 - 1) = (c.center.1 - 1)^2 + (c.center.2 - 1)^2) ∧
    ((x, y) ≠ (1, 1))

/-- The main theorem stating that the circle equation and tangent line equation are correct -/
theorem circle_and_tangent_line_equations (c : CircleC) : 
  circle_equation c ∧ tangent_line_equation c :=
sorry

end circle_and_tangent_line_equations_l3134_313434


namespace smallest_a_for_positive_integer_roots_l3134_313454

theorem smallest_a_for_positive_integer_roots : ∃ (a : ℕ),
  (∀ (x₁ x₂ : ℕ), x₁ * x₂ = 2022 ∧ x₁ + x₂ = a → x₁^2 - a*x₁ + 2022 = 0 ∧ x₂^2 - a*x₂ + 2022 = 0) ∧
  (∀ (b : ℕ), b < a →
    ¬∃ (y₁ y₂ : ℕ), y₁ * y₂ = 2022 ∧ y₁ + y₂ = b ∧ y₁^2 - b*y₁ + 2022 = 0 ∧ y₂^2 - b*y₂ + 2022 = 0) ∧
  a = 343 :=
by
  sorry

#check smallest_a_for_positive_integer_roots

end smallest_a_for_positive_integer_roots_l3134_313454


namespace arithmetic_sequence_8th_term_l3134_313446

/-- Proves that the 8th term of an arithmetic sequence with 26 terms, 
    first term 4, and last term 104, is equal to 32. -/
theorem arithmetic_sequence_8th_term 
  (a : ℕ → ℝ) 
  (h1 : a 1 = 4) 
  (h2 : a 26 = 104) 
  (h3 : ∀ n : ℕ, 1 < n → n ≤ 26 → a n - a (n-1) = a 2 - a 1) :
  a 8 = 32 := by
  sorry

end arithmetic_sequence_8th_term_l3134_313446


namespace two_thirds_of_45_minus_7_l3134_313458

theorem two_thirds_of_45_minus_7 : (2 / 3 : ℚ) * 45 - 7 = 23 := by
  sorry

end two_thirds_of_45_minus_7_l3134_313458


namespace sequence_completeness_l3134_313483

theorem sequence_completeness (a : ℕ → ℤ) :
  (∀ n : ℕ, n > 0 → (Finset.range n).card = (Finset.image (λ i => a i % n) (Finset.range n)).card) →
  ∀ k : ℤ, ∃! i : ℕ, a i = k :=
sorry

end sequence_completeness_l3134_313483


namespace equation_solution_l3134_313474

theorem equation_solution (x : ℂ) : 
  (x - 2)^6 + (x - 6)^6 = 64 ↔ x = 4 + Complex.I * Real.sqrt 2 ∨ x = 4 - Complex.I * Real.sqrt 2 := by
  sorry

end equation_solution_l3134_313474


namespace nurses_survey_result_l3134_313471

def total_nurses : ℕ := 150
def high_blood_pressure : ℕ := 90
def heart_trouble : ℕ := 50
def both_conditions : ℕ := 30

theorem nurses_survey_result : 
  (total_nurses - (high_blood_pressure + heart_trouble - both_conditions)) / total_nurses * 100 = 80 / 3 := by
  sorry

end nurses_survey_result_l3134_313471


namespace remaining_water_bottles_samiras_remaining_bottles_l3134_313451

/-- Calculates the number of water bottles remaining after a soccer game --/
theorem remaining_water_bottles (initial_bottles : ℕ) (players : ℕ) 
  (bottles_first_break : ℕ) (bottles_end_game : ℕ) : ℕ :=
  let bottles_after_first_break := initial_bottles - players * bottles_first_break
  let final_remaining_bottles := bottles_after_first_break - players * bottles_end_game
  final_remaining_bottles

/-- Proves that given the specific conditions of Samira's soccer game, 
    15 water bottles remain --/
theorem samiras_remaining_bottles : 
  remaining_water_bottles (4 * 12) 11 2 1 = 15 := by
  sorry

end remaining_water_bottles_samiras_remaining_bottles_l3134_313451


namespace annie_extracurricular_hours_l3134_313469

def chess_hours : ℕ := 2
def drama_hours : ℕ := 8
def glee_hours : ℕ := 3
def total_weeks : ℕ := 12
def sick_weeks : ℕ := 2

def extracurricular_hours_per_week : ℕ := chess_hours + drama_hours + glee_hours
def active_weeks : ℕ := total_weeks - sick_weeks

theorem annie_extracurricular_hours :
  extracurricular_hours_per_week * active_weeks = 130 := by
  sorry

end annie_extracurricular_hours_l3134_313469


namespace bridge_length_at_least_train_length_l3134_313462

/-- Proves that the length of a bridge is at least as long as a train, given the train's length,
    speed, and time to cross the bridge. -/
theorem bridge_length_at_least_train_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 200)
  (h2 : train_speed_kmh = 32)
  (h3 : crossing_time = 20)
  : ∃ (bridge_length : ℝ), bridge_length ≥ train_length :=
by
  sorry

#check bridge_length_at_least_train_length

end bridge_length_at_least_train_length_l3134_313462


namespace box_volume_from_face_centers_l3134_313448

def rectangular_box_volume (a b c : ℝ) : ℝ := 8 * a * b * c

theorem box_volume_from_face_centers 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a^2 + b^2 = 4^2)
  (h2 : b^2 + c^2 = 5^2)
  (h3 : a^2 + c^2 = 6^2) :
  rectangular_box_volume a b c = 90 * Real.sqrt 6 :=
sorry

end box_volume_from_face_centers_l3134_313448


namespace prob_defective_second_draw_specific_l3134_313455

/-- Probability of drawing a defective item on the second draw -/
def prob_defective_second_draw (total : ℕ) (defective : ℕ) (good : ℕ) : ℚ :=
  if total > 0 ∧ good > 0 then
    defective / (total - 1 : ℚ)
  else
    0

theorem prob_defective_second_draw_specific :
  prob_defective_second_draw 10 3 7 = 1/3 :=
by sorry

end prob_defective_second_draw_specific_l3134_313455


namespace probability_point_in_circle_l3134_313476

theorem probability_point_in_circle (s : ℝ) (r : ℝ) (h_s : s = 6) (h_r : r = 1.5) :
  (π * r^2) / (s^2) = π / 16 := by
  sorry

end probability_point_in_circle_l3134_313476


namespace find_Y_l3134_313429

theorem find_Y : ∃ Y : ℝ, (100 + Y / 90) * 90 = 9020 ∧ Y = 20 := by
  sorry

end find_Y_l3134_313429


namespace last_digit_of_one_over_three_to_ten_l3134_313431

theorem last_digit_of_one_over_three_to_ten (n : ℕ) : 
  (1 : ℚ) / (3^10 : ℚ) * 10^n % 10 = 0 :=
by sorry

end last_digit_of_one_over_three_to_ten_l3134_313431


namespace inequality_proof_l3134_313428

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  (1 / (a - b)) + (1 / (b - c)) + (1 / (c - a)) > 0 := by
sorry

end inequality_proof_l3134_313428


namespace if_statement_properties_l3134_313404

-- Define the structure of an IF statement
structure IfStatement where
  has_else : Bool
  has_end_if : Bool

-- Define what makes an IF statement valid
def is_valid_if_statement (stmt : IfStatement) : Prop :=
  stmt.has_end_if ∧ (stmt.has_else ∨ ¬stmt.has_else)

-- Theorem statement
theorem if_statement_properties :
  ∀ (stmt : IfStatement),
    is_valid_if_statement stmt →
    (stmt.has_else ∨ ¬stmt.has_else) ∧ stmt.has_end_if :=
by sorry

end if_statement_properties_l3134_313404


namespace sqrt_nested_expression_l3134_313408

theorem sqrt_nested_expression : 
  Real.sqrt (144 * Real.sqrt (64 * Real.sqrt 36)) = 48 * Real.sqrt 3 := by
  sorry

end sqrt_nested_expression_l3134_313408


namespace upper_bound_y_l3134_313403

theorem upper_bound_y (x y : ℤ) (h1 : 3 < x) (h2 : x < 6) (h3 : 6 < y) 
  (h4 : ∀ (a b : ℤ), 3 < a → a < 6 → 6 < b → b - a ≤ 6) : y ≤ 11 :=
sorry

end upper_bound_y_l3134_313403


namespace root_difference_equals_1993_l3134_313425

theorem root_difference_equals_1993 : ∃ m n : ℝ,
  (1992 * m)^2 - 1991 * 1993 * m - 1 = 0 ∧
  n^2 + 1991 * n - 1992 = 0 ∧
  (∀ x : ℝ, (1992 * x)^2 - 1991 * 1993 * x - 1 = 0 → x ≤ m) ∧
  (∀ y : ℝ, y^2 + 1991 * y - 1992 = 0 → y ≤ n) ∧
  m - n = 1993 :=
sorry

end root_difference_equals_1993_l3134_313425


namespace nested_fraction_evaluation_l3134_313470

theorem nested_fraction_evaluation : 
  (1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = 8 / 21 := by
  sorry

end nested_fraction_evaluation_l3134_313470


namespace raines_change_l3134_313450

/-- Calculates the change Raine receives after purchasing items with a discount --/
theorem raines_change (bracelet_price necklace_price mug_price : ℚ)
  (bracelet_qty necklace_qty mug_qty : ℕ)
  (discount_rate : ℚ)
  (payment : ℚ) :
  bracelet_price = 15 →
  necklace_price = 10 →
  mug_price = 20 →
  bracelet_qty = 3 →
  necklace_qty = 2 →
  mug_qty = 1 →
  discount_rate = 1/10 →
  payment = 100 →
  let total_cost := bracelet_price * bracelet_qty + necklace_price * necklace_qty + mug_price * mug_qty
  let discounted_cost := total_cost * (1 - discount_rate)
  payment - discounted_cost = 23.5 := by sorry

end raines_change_l3134_313450


namespace triangle_angle_problem_l3134_313445

theorem triangle_angle_problem (A B C : ℝ) 
  (h1 : B = A + 10)
  (h2 : C = B + 10)
  (h3 : A + B + C = 180) :
  B = 60 := by
  sorry

end triangle_angle_problem_l3134_313445


namespace inequality_implication_l3134_313475

theorem inequality_implication (a b : ℝ) : a^2 + 2*a*b + b^2 + a + b - 2 ≠ 0 → a + b ≠ 1 := by
  sorry

end inequality_implication_l3134_313475


namespace quadrilateral_equality_l3134_313494

/-- Given a quadrilateral ABCD where AD is parallel to BC, 
    prove that AC^2 + BD^2 = AB^2 + CD^2 + 2AD · BC. -/
theorem quadrilateral_equality (A B C D : ℝ × ℝ) 
    (h_parallel : (D.2 - A.2) / (D.1 - A.1) = (C.2 - B.2) / (C.1 - B.1)) : 
    (A.1 - C.1)^2 + (A.2 - C.2)^2 + (B.1 - D.1)^2 + (B.2 - D.2)^2 = 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 + (C.1 - D.1)^2 + (C.2 - D.2)^2 + 
    2 * ((D.1 - A.1) * (C.1 - B.1) + (D.2 - A.2) * (C.2 - B.2)) := by
  sorry


end quadrilateral_equality_l3134_313494


namespace bisection_diagram_type_l3134_313484

/-- The function we're finding the root for -/
def f (x : ℝ) : ℝ := x^2 - 2

/-- Represents the types of diagrams -/
inductive DiagramType
| ProcessFlowchart
| KnowledgeStructureDiagram
| ProgramFlowchart
| OrganizationalStructureDiagram

/-- Properties of the bisection method -/
structure BisectionMethod where
  continuous : ∀ a b, a < b → ContinuousOn f (Set.Icc a b)
  oppositeSign : ∃ a b, a < b ∧ f a * f b < 0
  iterative : ∀ a b, a < b → ∃ c, a < c ∧ c < b ∧ f c = (f a + f b) / 2

/-- The theorem stating that the bisection method for x^2 - 2 = 0 is represented by a Program Flowchart -/
theorem bisection_diagram_type (bm : BisectionMethod) : 
  ∃ d : DiagramType, d = DiagramType.ProgramFlowchart :=
sorry

end bisection_diagram_type_l3134_313484


namespace total_lunch_cost_l3134_313416

/-- Calculates the total cost of lunch for all students in an elementary school --/
theorem total_lunch_cost (third_grade_classes fourth_grade_classes fifth_grade_classes : ℕ)
  (third_grade_students fourth_grade_students fifth_grade_students : ℕ)
  (hamburger_cost carrot_cost cookie_cost : ℚ) : ℚ :=
  by
  have h1 : third_grade_classes = 5 := by sorry
  have h2 : fourth_grade_classes = 4 := by sorry
  have h3 : fifth_grade_classes = 4 := by sorry
  have h4 : third_grade_students = 30 := by sorry
  have h5 : fourth_grade_students = 28 := by sorry
  have h6 : fifth_grade_students = 27 := by sorry
  have h7 : hamburger_cost = 2.1 := by sorry
  have h8 : carrot_cost = 0.5 := by sorry
  have h9 : cookie_cost = 0.2 := by sorry

  have total_students : ℕ := 
    third_grade_classes * third_grade_students + 
    fourth_grade_classes * fourth_grade_students + 
    fifth_grade_classes * fifth_grade_students

  have lunch_cost_per_student : ℚ := hamburger_cost + carrot_cost + cookie_cost

  have total_cost : ℚ := total_students * lunch_cost_per_student

  exact 1036

end total_lunch_cost_l3134_313416


namespace pentagon_largest_angle_l3134_313486

theorem pentagon_largest_angle (F G H I J : ℝ) : 
  F = 90 → 
  G = 70 → 
  H = I → 
  J = 2 * H + 20 → 
  F + G + H + I + J = 540 → 
  max F (max G (max H (max I J))) = 200 :=
by sorry

end pentagon_largest_angle_l3134_313486


namespace quadratic_reciprocal_roots_l3134_313424

theorem quadratic_reciprocal_roots (p q : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ x*y = 1) →
  ((p ≥ 2 ∨ p ≤ -2) ∧ q = 1) :=
by sorry

end quadratic_reciprocal_roots_l3134_313424


namespace set_equality_sum_l3134_313499

theorem set_equality_sum (x y : ℝ) (A B : Set ℝ) : 
  A = {2, y} → B = {x, 3} → A = B → x + y = 5 := by
  sorry

end set_equality_sum_l3134_313499


namespace circle_center_and_radius_l3134_313400

/-- The equation of the circle is x^2 + y^2 - 2x + 6y + 6 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 6 = 0

/-- The center of the circle -/
def center : ℝ × ℝ := (1, -3)

/-- The radius of the circle -/
def radius : ℝ := 2

theorem circle_center_and_radius :
  ∀ x y : ℝ, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
sorry

end circle_center_and_radius_l3134_313400


namespace rectangular_room_length_l3134_313465

theorem rectangular_room_length (area width : ℝ) (h1 : area = 215.6) (h2 : width = 14) :
  area / width = 15.4 := by
  sorry

end rectangular_room_length_l3134_313465


namespace hearty_beads_count_l3134_313430

/-- The number of packages of blue beads Hearty bought -/
def blue_packages : ℕ := 3

/-- The number of packages of red beads Hearty bought -/
def red_packages : ℕ := 5

/-- The number of beads in each package -/
def beads_per_package : ℕ := 40

/-- The total number of beads Hearty has -/
def total_beads : ℕ := blue_packages * beads_per_package + red_packages * beads_per_package

theorem hearty_beads_count : total_beads = 320 := by
  sorry

end hearty_beads_count_l3134_313430


namespace thabo_book_difference_l3134_313490

/-- Represents the number of books Thabo owns of each type -/
structure BookCollection where
  paperbackFiction : ℕ
  paperbackNonfiction : ℕ
  hardcoverNonfiction : ℕ

/-- The properties of Thabo's book collection -/
def validCollection (books : BookCollection) : Prop :=
  books.paperbackFiction + books.paperbackNonfiction + books.hardcoverNonfiction = 280 ∧
  books.paperbackNonfiction > books.hardcoverNonfiction ∧
  books.paperbackFiction = 2 * books.paperbackNonfiction ∧
  books.hardcoverNonfiction = 55

theorem thabo_book_difference (books : BookCollection) 
  (h : validCollection books) : 
  books.paperbackNonfiction - books.hardcoverNonfiction = 20 := by
  sorry

end thabo_book_difference_l3134_313490


namespace sqrt_meaningful_range_l3134_313488

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2*x - 6) ↔ x ≥ 3 := by sorry

end sqrt_meaningful_range_l3134_313488


namespace det_special_matrix_is_zero_l3134_313422

open Real Matrix

theorem det_special_matrix_is_zero (θ φ : ℝ) : 
  det !![0, cos θ, sin θ; -cos θ, 0, cos φ; -sin θ, -cos φ, 0] = 0 := by
  sorry

end det_special_matrix_is_zero_l3134_313422


namespace equation_holds_iff_l3134_313443

theorem equation_holds_iff (k y : ℝ) : 
  (∀ x : ℝ, -x^2 - (k+10)*x - 8 = -(x - 2)*(x - 4) + (y - 3)*(y - 6)) ↔ 
  (k = -16 ∧ False) :=
by sorry

end equation_holds_iff_l3134_313443


namespace percentage_female_officers_on_duty_l3134_313441

/-- Given the conditions of the police force duty, prove the percentage of female officers on duty -/
theorem percentage_female_officers_on_duty
  (total_on_duty : ℕ)
  (total_female_officers : ℕ)
  (half_on_duty_female : total_on_duty / 2 = total_on_duty - total_on_duty / 2)
  (h_total_on_duty : total_on_duty = 180)
  (h_total_female : total_female_officers = 500) :
  (((total_on_duty / 2 : ℚ) / total_female_officers) * 100 : ℚ) = 18 := by
  sorry

end percentage_female_officers_on_duty_l3134_313441


namespace farm_animals_count_l3134_313495

/-- Represents the farm with goats and sheep -/
structure Farm where
  goats : ℕ
  sheep : ℕ

/-- Calculates the total number of animals on the farm -/
def Farm.total (f : Farm) : ℕ := f.goats + f.sheep

/-- Theorem: Given the conditions, the total number of animals on the farm is 1524 -/
theorem farm_animals_count (f : Farm) 
  (ratio : f.goats * 7 = f.sheep * 5)
  (sale_amount : (f.goats / 2) * 40 + (f.sheep * 2 / 3) * 30 = 7200) : 
  f.total = 1524 := by
  sorry


end farm_animals_count_l3134_313495


namespace train_length_calculation_l3134_313466

/-- Calculates the length of a train given specific conditions of overtaking a motorbike -/
theorem train_length_calculation 
  (initial_train_speed : Real) 
  (train_acceleration : Real)
  (motorbike_speed : Real)
  (overtake_time : Real)
  (motorbike_length : Real)
  (h1 : initial_train_speed = 25)  -- 90 kmph converted to m/s
  (h2 : train_acceleration = 0.5)
  (h3 : motorbike_speed = 20)      -- 72 kmph converted to m/s
  (h4 : overtake_time = 50)
  (h5 : motorbike_length = 2) :
  let final_train_speed := initial_train_speed + train_acceleration * overtake_time
  let train_distance := initial_train_speed * overtake_time + 0.5 * train_acceleration * overtake_time^2
  let motorbike_distance := motorbike_speed * overtake_time
  let train_length := train_distance - motorbike_distance + motorbike_length
  train_length = 877 := by
  sorry

end train_length_calculation_l3134_313466


namespace sine_equation_solution_l3134_313464

theorem sine_equation_solution (x y : ℝ) :
  |Real.sin x - Real.sin y| + Real.sin x * Real.sin y = 0 →
  ∃ k n : ℤ, x = k * Real.pi ∧ y = n * Real.pi := by
sorry

end sine_equation_solution_l3134_313464


namespace john_experience_theorem_l3134_313436

/-- Represents the years of experience for each person -/
structure Experience where
  james : ℕ
  john : ℕ
  mike : ℕ

/-- The conditions of the problem -/
def problem_conditions (e : Experience) : Prop :=
  e.james = 20 ∧
  e.john - 8 = 2 * (e.james - 8) ∧
  e.james + e.john + e.mike = 68

/-- John's experience when Mike started -/
def john_experience_when_mike_started (e : Experience) : ℕ :=
  e.john - e.mike

/-- The theorem to prove -/
theorem john_experience_theorem (e : Experience) :
  problem_conditions e → john_experience_when_mike_started e = 16 := by
  sorry

end john_experience_theorem_l3134_313436


namespace exists_valid_custom_division_l3134_313472

/-- A custom division type that allows introducing additional 7s in intermediate calculations -/
structure CustomDivision where
  dividend : Nat
  divisor : Nat
  quotient : Nat
  intermediate_sevens : List Nat

/-- Checks if a number contains at least one 7 -/
def containsSeven (n : Nat) : Prop :=
  ∃ d, d ∈ n.digits 10 ∧ d = 7

/-- Theorem stating the existence of a valid custom division -/
theorem exists_valid_custom_division :
  ∃ (cd : CustomDivision),
    cd.dividend ≥ 1000000000 ∧ cd.dividend < 10000000000 ∧
    cd.divisor ≥ 100000 ∧ cd.divisor < 1000000 ∧
    cd.quotient ≥ 10000 ∧ cd.quotient < 100000 ∧
    containsSeven cd.dividend ∧
    containsSeven cd.divisor ∧
    cd.dividend = cd.divisor * cd.quotient :=
  sorry

#check exists_valid_custom_division

end exists_valid_custom_division_l3134_313472


namespace choose_four_different_suits_standard_deck_l3134_313437

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (h1 : cards = suits * cards_per_suit)

/-- The number of ways to choose 4 cards from different suits in a standard deck -/
def choose_four_different_suits (d : Deck) : Nat :=
  d.cards_per_suit ^ d.suits

/-- Theorem stating that the number of ways to choose 4 cards from different suits
    in a standard deck of 52 cards is 28,561 -/
theorem choose_four_different_suits_standard_deck :
  ∃ (d : Deck), d.cards = 52 ∧ d.suits = 4 ∧ d.cards_per_suit = 13 ∧
  choose_four_different_suits d = 28561 :=
sorry

end choose_four_different_suits_standard_deck_l3134_313437


namespace odd_divides_power_factorial_minus_one_l3134_313468

theorem odd_divides_power_factorial_minus_one (n : ℕ) (h : Odd n) : n ∣ 2^(n.factorial) - 1 := by
  sorry

end odd_divides_power_factorial_minus_one_l3134_313468


namespace range_f_characterization_l3134_313420

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 1

-- Define the range of f(x) on [0, 2]
def range_f (a : ℝ) : Set ℝ :=
  { y | ∃ x ∈ Set.Icc 0 2, f a x = y }

-- Theorem statement
theorem range_f_characterization (a : ℝ) :
  range_f a =
    if a < 0 then Set.Icc (-1) (3 - 4*a)
    else if a < 1 then Set.Icc (-1 - a^2) (3 - 4*a)
    else if a < 2 then Set.Icc (-1 - a^2) (-1)
    else Set.Icc (3 - 4*a) (-1) := by
  sorry

end range_f_characterization_l3134_313420


namespace original_number_proof_l3134_313459

theorem original_number_proof (x : ℚ) : 1 + 1 / x = 11 / 5 → x = 5 / 6 := by
  sorry

end original_number_proof_l3134_313459


namespace mary_turnips_count_l3134_313442

/-- The number of turnips grown by Sally -/
def sally_turnips : ℕ := 113

/-- The total number of turnips grown by Sally and Mary -/
def total_turnips : ℕ := 242

/-- The number of turnips grown by Mary -/
def mary_turnips : ℕ := total_turnips - sally_turnips

theorem mary_turnips_count : mary_turnips = 129 := by
  sorry

end mary_turnips_count_l3134_313442


namespace howard_window_washing_earnings_l3134_313456

theorem howard_window_washing_earnings
  (initial_amount : ℝ)
  (final_amount : ℝ)
  (cleaning_expenses : ℝ)
  (h1 : initial_amount = 26)
  (h2 : final_amount = 52)
  (h3 : final_amount = initial_amount + earnings - cleaning_expenses) :
  earnings = 26 + cleaning_expenses :=
by sorry

end howard_window_washing_earnings_l3134_313456


namespace log_sum_equality_l3134_313485

theorem log_sum_equality : Real.log 8 / Real.log 10 + 3 * (Real.log 5 / Real.log 10) = 3 := by
  sorry

end log_sum_equality_l3134_313485


namespace garys_final_amount_l3134_313497

/-- Given Gary's initial amount and the amount he received from selling his snake, 
    calculate his final amount. -/
theorem garys_final_amount 
  (initial_amount : ℝ) 
  (snake_sale_amount : ℝ) 
  (h1 : initial_amount = 73.0) 
  (h2 : snake_sale_amount = 55.0) : 
  initial_amount + snake_sale_amount = 128.0 := by
  sorry

end garys_final_amount_l3134_313497


namespace perfect_square_trinomial_l3134_313493

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + (m-1)*x + 9 = (a*x + b)^2) → 
  (m = 7 ∨ m = -5) :=
by sorry

end perfect_square_trinomial_l3134_313493


namespace probability_of_double_domino_l3134_313407

/-- Represents a domino tile with two integers -/
structure Domino :=
  (a b : ℕ)

/-- The set of all possible domino tiles -/
def dominoSet : Set Domino :=
  {d : Domino | d.a ≤ 12 ∧ d.b ≤ 12}

/-- A domino is considered a double if both numbers are the same -/
def isDouble (d : Domino) : Prop :=
  d.a = d.b

/-- The number of unique domino tiles in the complete set -/
def totalDominos : ℕ :=
  (13 * 14) / 2

/-- The number of double dominos in the complete set -/
def doubleDominos : ℕ := 13

theorem probability_of_double_domino :
  (doubleDominos : ℚ) / totalDominos = 13 / 91 :=
sorry

end probability_of_double_domino_l3134_313407


namespace exists_sequence_with_finite_primes_l3134_313410

theorem exists_sequence_with_finite_primes :
  ∃ (a : ℕ → ℕ), 
    (∀ n m : ℕ, n < m → a n < a m) ∧ 
    (∀ k : ℕ, k ≥ 2 → ∃ N : ℕ, ∀ n ≥ N, ¬ Prime (k + a n)) :=
by sorry

end exists_sequence_with_finite_primes_l3134_313410


namespace isosceles_triangle_rectangle_perimeter_difference_unique_d_value_count_impossible_d_values_l3134_313467

theorem isosceles_triangle_rectangle_perimeter_difference 
  (d : ℕ) (w : ℝ) : 
  w > 0 → 
  6 * w > 0 → 
  6 * w + 2 * d = 6 * w + 1236 → 
  d = 618 := by
sorry

theorem unique_d_value : 
  ∃! d : ℕ, ∃ w : ℝ, 
    w > 0 ∧ 
    6 * w > 0 ∧ 
    6 * w + 2 * d = 6 * w + 1236 := by
sorry

theorem count_impossible_d_values : 
  (Nat.card {d : ℕ | ¬∃ w : ℝ, w > 0 ∧ 6 * w > 0 ∧ 6 * w + 2 * d = 6 * w + 1236}) = ℵ₀ := by
sorry

end isosceles_triangle_rectangle_perimeter_difference_unique_d_value_count_impossible_d_values_l3134_313467


namespace golden_ratio_unique_progression_l3134_313453

theorem golden_ratio_unique_progression : ∃! x : ℝ, 
  x > 0 ∧ 
  let b := ⌊x⌋
  let c := x - b
  (c < b) ∧ (b < x) ∧ (c * x = b * b) ∧ x = (Real.sqrt 5 + 1) / 2 := by
  sorry

end golden_ratio_unique_progression_l3134_313453


namespace square_difference_identity_l3134_313402

theorem square_difference_identity : (25 + 15)^2 - (25 - 15)^2 = 1500 := by
  sorry

end square_difference_identity_l3134_313402


namespace missing_number_proof_l3134_313412

theorem missing_number_proof (x : ℝ) : 
  let numbers := [1, 22, 23, 24, 25, 26, x, 2]
  (List.sum numbers) / (List.length numbers) = 20 → x = 37 := by
sorry

end missing_number_proof_l3134_313412
