import Mathlib

namespace eight_squares_exist_l958_95874

/-- Represents a 3x3 square of digits -/
def Square := Matrix (Fin 3) (Fin 3) Nat

/-- Checks if a square uses all digits from 1 to 9 exactly once -/
def uses_all_digits (s : Square) : Prop :=
  ∀ d : Fin 9, ∃! (i j : Fin 3), s i j = d.val + 1

/-- Calculates the sum of a row in a square -/
def row_sum (s : Square) (i : Fin 3) : Nat :=
  (s i 0) + (s i 1) + (s i 2)

/-- Checks if the sum of the first two rows equals the sum of the third row -/
def sum_property (s : Square) : Prop :=
  row_sum s 0 + row_sum s 1 = row_sum s 2

/-- Calculates the difference between row sums -/
def row_sum_diff (s : Square) : Nat :=
  (row_sum s 2) - (row_sum s 1)

/-- The main theorem statement -/
theorem eight_squares_exist : 
  ∃ (squares : Fin 8 → Square),
    (∀ i : Fin 8, uses_all_digits (squares i)) ∧
    (∀ i : Fin 8, sum_property (squares i)) ∧
    (∀ i j : Fin 8, row_sum_diff (squares i) = row_sum_diff (squares j)) ∧
    (∀ i : Fin 8, row_sum_diff (squares i) = 9) :=
  sorry

end eight_squares_exist_l958_95874


namespace square_roots_theorem_l958_95887

theorem square_roots_theorem (x a : ℝ) (hx : x > 0) 
  (h1 : (a + 1) ^ 2 = x) (h2 : (2 * a - 7) ^ 2 = x) : a = 2 := by
  sorry

end square_roots_theorem_l958_95887


namespace project_remaining_time_l958_95811

/-- Given the time spent on various tasks of a project, proves that the remaining time for writing the report is 9 hours. -/
theorem project_remaining_time (total_time research_time proposal_time visual_aids_time editing_time rehearsal_time : ℕ)
  (h_total : total_time = 40)
  (h_research : research_time = 12)
  (h_proposal : proposal_time = 4)
  (h_visual : visual_aids_time = 7)
  (h_editing : editing_time = 5)
  (h_rehearsal : rehearsal_time = 3) :
  total_time - (research_time + proposal_time + visual_aids_time + editing_time + rehearsal_time) = 9 := by
  sorry

end project_remaining_time_l958_95811


namespace insurance_calculation_l958_95828

/-- Insurance calculation parameters --/
structure InsuranceParams where
  baseRate : Float
  noTransitionCoeff : Float
  noMedCertCoeff : Float
  assessedValue : Float
  cadasterValue : Float

/-- Calculate adjusted tariff --/
def calcAdjustedTariff (params : InsuranceParams) : Float :=
  params.baseRate * params.noTransitionCoeff * params.noMedCertCoeff

/-- Determine insurance amount --/
def determineInsuranceAmount (params : InsuranceParams) : Float :=
  max params.assessedValue params.cadasterValue

/-- Calculate insurance premium --/
def calcInsurancePremium (amount : Float) (tariff : Float) : Float :=
  amount * tariff

/-- Main theorem --/
theorem insurance_calculation (params : InsuranceParams) 
  (h1 : params.baseRate = 0.002)
  (h2 : params.noTransitionCoeff = 0.8)
  (h3 : params.noMedCertCoeff = 1.3)
  (h4 : params.assessedValue = 14500000)
  (h5 : params.cadasterValue = 15000000) :
  let adjustedTariff := calcAdjustedTariff params
  let insuranceAmount := determineInsuranceAmount params
  let insurancePremium := calcInsurancePremium insuranceAmount adjustedTariff
  adjustedTariff = 0.00208 ∧ 
  insuranceAmount = 15000000 ∧ 
  insurancePremium = 31200 := by
  sorry

end insurance_calculation_l958_95828


namespace repetend_of_five_elevenths_l958_95821

/-- The decimal representation of 5/11 has a repetend of 45. -/
theorem repetend_of_five_elevenths : ∃ (a b : ℕ), 
  (5 : ℚ) / 11 = (a : ℚ) / 100 + (b : ℚ) / 99 * (1 / 100) ∧ b = 45 := by
  sorry

end repetend_of_five_elevenths_l958_95821


namespace candle_placement_impossibility_l958_95825

theorem candle_placement_impossibility (n : ℕ) (d r : ℝ) (h_n : n = 13) (h_d : d = 10) (h_r : r = 18) :
  ¬ ∃ (points : Fin n → ℝ × ℝ),
    (∀ i, (points i).1^2 + (points i).2^2 = r^2) ∧
    (∀ i j, i ≠ j → Real.sqrt ((points i).1 - (points j).1)^2 + ((points i).2 - (points j).2)^2 ≥ d) :=
by sorry


end candle_placement_impossibility_l958_95825


namespace geometric_sum_five_terms_l958_95869

theorem geometric_sum_five_terms (a r : ℚ) (h1 : a = 1/4) (h2 : r = 1/4) :
  let S := a + a*r + a*r^2 + a*r^3 + a*r^4
  S = 341/1024 := by sorry

end geometric_sum_five_terms_l958_95869


namespace part_one_part_two_l958_95886

/-- The function f(x) = ax^2 - 3ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * a * x + 2

/-- Part 1: Given the solution set of f(x) > 0, prove a and b -/
theorem part_one (a b : ℝ) : 
  (∀ x, f a x > 0 ↔ (x < 1 ∨ x > b)) → a = 1 ∧ b = 2 :=
sorry

/-- Part 2: Given f(x) > 0 for all x, prove the range of a -/
theorem part_two (a : ℝ) :
  (∀ x, f a x > 0) → 0 ≤ a ∧ a < 8/9 :=
sorry

end part_one_part_two_l958_95886


namespace negation_of_existence_negation_of_exp_greater_than_x_l958_95865

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem negation_of_exp_greater_than_x :
  (¬ ∃ x : ℝ, Real.exp x > x) ↔ (∀ x : ℝ, Real.exp x ≤ x) :=
by sorry

end negation_of_existence_negation_of_exp_greater_than_x_l958_95865


namespace largest_number_in_sequence_l958_95832

/-- A sequence of 8 increasing real numbers -/
def IncreasingSequence : Type := { s : Fin 8 → ℝ // ∀ i j, i < j → s i < s j }

/-- Checks if a subsequence of 4 consecutive numbers is an arithmetic progression -/
def IsArithmeticProgression (s : IncreasingSequence) (start : Fin 5) (d : ℝ) : Prop :=
  ∀ i : Fin 3, s.val (start + i + 1) - s.val (start + i) = d

/-- Checks if a subsequence of 4 consecutive numbers is a geometric progression -/
def IsGeometricProgression (s : IncreasingSequence) (start : Fin 5) : Prop :=
  ∃ r : ℝ, ∀ i : Fin 3, s.val (start + i + 1) / s.val (start + i) = r

/-- The main theorem -/
theorem largest_number_in_sequence (s : IncreasingSequence) 
  (h1 : ∃ start1 : Fin 5, IsArithmeticProgression s start1 4)
  (h2 : ∃ start2 : Fin 5, IsArithmeticProgression s start2 36)
  (h3 : ∃ start3 : Fin 5, IsGeometricProgression s start3) :
  s.val 7 = 126 ∨ s.val 7 = 6 := by
  sorry


end largest_number_in_sequence_l958_95832


namespace perpendicular_unit_vectors_l958_95823

def vector_a : ℝ × ℝ := (2, -2)

def is_unit_vector (v : ℝ × ℝ) : Prop :=
  v.1 ^ 2 + v.2 ^ 2 = 1

def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem perpendicular_unit_vectors :
  ∀ v : ℝ × ℝ, is_unit_vector v ∧ is_perpendicular v vector_a →
    v = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ∨ v = (-Real.sqrt 2 / 2, -Real.sqrt 2 / 2) :=
by sorry

end perpendicular_unit_vectors_l958_95823


namespace sum_of_reciprocals_of_roots_l958_95836

theorem sum_of_reciprocals_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - 6*x₁ + 6 = 0 → 
  x₂^2 - 6*x₂ + 6 = 0 → 
  x₁ ≠ x₂ → 
  (1/x₁) + (1/x₂) = 1 := by sorry

end sum_of_reciprocals_of_roots_l958_95836


namespace P_in_second_quadrant_l958_95815

-- Define the point P
def P (x : ℝ) : ℝ × ℝ := (-2, x^2 + 1)

-- Define what it means for a point to be in the second quadrant
def is_in_second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- Theorem stating that P is in the second quadrant for all real x
theorem P_in_second_quadrant (x : ℝ) : is_in_second_quadrant (P x) := by
  sorry


end P_in_second_quadrant_l958_95815


namespace average_score_calculation_l958_95848

/-- Calculates the average score of all students given the proportion of male students,
    the average score of male students, and the average score of female students. -/
def average_score (male_proportion : ℝ) (male_avg : ℝ) (female_avg : ℝ) : ℝ :=
  male_proportion * male_avg + (1 - male_proportion) * female_avg

/-- Theorem stating that when 40% of students are male, with male average score 75
    and female average score 80, the overall average score is 78. -/
theorem average_score_calculation :
  average_score 0.4 75 80 = 78 := by
  sorry

end average_score_calculation_l958_95848


namespace binomial_coefficient_sum_and_constant_term_l958_95847

theorem binomial_coefficient_sum_and_constant_term 
  (x : ℝ) (a : ℝ) (n : ℕ) :
  (1 + a)^n = 32 →
  (∃ (r : ℕ), (n.choose r) * a^r = 80 ∧ 10 - 5*r = 0) →
  a = 2 * Real.sqrt 2 :=
by sorry

end binomial_coefficient_sum_and_constant_term_l958_95847


namespace circle_parameters_l958_95855

/-- Definition of the circle C -/
def circle_equation (x y a b : ℝ) : Prop :=
  x^2 + y^2 + a*x - 2*y + b = 0

/-- Definition of a point being on the circle -/
def point_on_circle (x y a b : ℝ) : Prop :=
  circle_equation x y a b

/-- Definition of the line x + y - 1 = 0 -/
def line_equation (x y : ℝ) : Prop :=
  x + y - 1 = 0

/-- Definition of symmetric point with respect to the line x + y - 1 = 0 -/
def symmetric_point (x y x' y' : ℝ) : Prop :=
  x' + y' = x + y ∧ line_equation ((x + x')/2) ((y + y')/2)

/-- Main theorem -/
theorem circle_parameters :
  ∀ (a b : ℝ),
    (point_on_circle 2 1 a b) →
    (∃ (x' y' : ℝ), symmetric_point 2 1 x' y' ∧ point_on_circle x' y' a b) →
    (a = 0 ∧ b = -3) :=
by sorry

end circle_parameters_l958_95855


namespace units_digit_of_8_power_47_l958_95861

theorem units_digit_of_8_power_47 : Nat.mod (8^47) 10 = 2 := by
  sorry

end units_digit_of_8_power_47_l958_95861


namespace quadratic_roots_theorem_l958_95826

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 - (2*m + 1)*x + m^2 + m

-- Define the condition for the roots
def root_condition (a b : ℝ) : Prop := (2*a + b) * (a + 2*b) = 20

-- Theorem statement
theorem quadratic_roots_theorem (m : ℝ) :
  (∃ a b : ℝ, a ≠ b ∧ quadratic m a = 0 ∧ quadratic m b = 0) ∧
  (∀ a b : ℝ, quadratic m a = 0 → quadratic m b = 0 → root_condition a b → (m = -2 ∨ m = 1)) :=
sorry

end quadratic_roots_theorem_l958_95826


namespace rectangle_longest_side_l958_95819

/-- Given a rectangle with perimeter 240 feet and area equal to 8 times its perimeter,
    the length of its longest side is 96 feet. -/
theorem rectangle_longest_side (l w : ℝ) : 
  l > 0 → w > 0 →
  2 * l + 2 * w = 240 →
  l * w = 8 * (2 * l + 2 * w) →
  max l w = 96 := by
sorry

end rectangle_longest_side_l958_95819


namespace smaller_number_in_ratio_l958_95833

theorem smaller_number_in_ratio (a b d u v : ℝ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : u > 0) (h4 : v > 0)
  (h5 : u / v = b / a) (h6 : u + v = d) : 
  min u v = a * d / (a + b) := by
sorry

end smaller_number_in_ratio_l958_95833


namespace value_of_expression_l958_95813

theorem value_of_expression (x : ℝ) (h : x^2 - 2*x = 1) : 
  2023 + 6*x - 3*x^2 = 2020 := by
  sorry

end value_of_expression_l958_95813


namespace cost_price_is_95_l958_95854

/-- Represents the cost price for one metre of cloth -/
def cost_price_per_metre (total_metres : ℕ) (total_selling_price : ℕ) (loss_per_metre : ℕ) : ℕ :=
  total_selling_price / total_metres + loss_per_metre

/-- Theorem stating that the cost price for one metre of cloth is 95 -/
theorem cost_price_is_95 :
  cost_price_per_metre 200 18000 5 = 95 := by
  sorry

end cost_price_is_95_l958_95854


namespace min_value_of_squared_sum_l958_95852

theorem min_value_of_squared_sum (a b c t : ℝ) 
  (sum_condition : a + b + c = t) 
  (squared_sum_condition : a^2 + b^2 + c^2 = 1) : 
  2 * (a^2 + b^2 + c^2) = 2 := by
  sorry

end min_value_of_squared_sum_l958_95852


namespace counterexample_exists_l958_95899

theorem counterexample_exists : ∃ (a b c : ℤ), a > b ∧ b > c ∧ a + b ≤ c := by
  sorry

end counterexample_exists_l958_95899


namespace polynomial_equality_l958_95888

theorem polynomial_equality (k : ℝ) : 
  (∀ x : ℝ, (x + 6) * (x - 5) = x^2 + k*x - 30) → k = 1 := by
  sorry

end polynomial_equality_l958_95888


namespace molecular_weight_BaBr2_is_297_l958_95844

/-- The molecular weight of BaBr2 in grams per mole -/
def molecular_weight_BaBr2 : ℝ := 297

/-- The number of moles used in the given condition -/
def given_moles : ℝ := 8

/-- The total weight of the given moles of BaBr2 in grams -/
def total_weight : ℝ := 2376

/-- Theorem: The molecular weight of BaBr2 is 297 grams/mole -/
theorem molecular_weight_BaBr2_is_297 :
  molecular_weight_BaBr2 = total_weight / given_moles :=
sorry

end molecular_weight_BaBr2_is_297_l958_95844


namespace praveen_initial_investment_l958_95864

-- Define the initial parameters
def haris_investment : ℕ := 8280
def praveens_time : ℕ := 12
def haris_time : ℕ := 7
def profit_ratio_praveen : ℕ := 2
def profit_ratio_hari : ℕ := 3

-- Define Praveen's investment as a function
def praveens_investment : ℕ := 
  (haris_investment * haris_time * profit_ratio_praveen) / (praveens_time * profit_ratio_hari)

-- Theorem statement
theorem praveen_initial_investment :
  praveens_investment = 3220 :=
sorry

end praveen_initial_investment_l958_95864


namespace sheridan_cats_l958_95822

/-- The number of cats Mrs. Sheridan has after giving some away -/
def remaining_cats (initial : Float) (given_away : Float) : Float :=
  initial - given_away

/-- Theorem: Mrs. Sheridan has 3.0 cats after giving away 14.0 cats from her initial 17.0 cats -/
theorem sheridan_cats : remaining_cats 17.0 14.0 = 3.0 := by
  sorry

end sheridan_cats_l958_95822


namespace solution_xy_l958_95870

theorem solution_xy : ∃ (x y : ℝ), 
  (x + 2*y = (7 - x) + (7 - 2*y)) ∧ 
  (x - y = 2*(x - 2) - (y - 3)) ∧ 
  (x = 1) ∧ (y = 3) := by
  sorry

end solution_xy_l958_95870


namespace number_categorization_l958_95895

def given_numbers : List ℚ := [-18, -3/5, 0, 2023, -22/7, -0.142857, 95/100]

def is_positive (x : ℚ) : Prop := x > 0
def is_negative (x : ℚ) : Prop := x < 0
def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n
def is_fraction (x : ℚ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

def positive_set : Set ℚ := {x | is_positive x}
def negative_set : Set ℚ := {x | is_negative x}
def integer_set : Set ℚ := {x | is_integer x}
def fraction_set : Set ℚ := {x | is_fraction x}

theorem number_categorization :
  (positive_set ∩ given_numbers.toFinset = {2023, 95/100}) ∧
  (negative_set ∩ given_numbers.toFinset = {-18, -3/5, -22/7, -0.142857}) ∧
  (integer_set ∩ given_numbers.toFinset = {-18, 0, 2023}) ∧
  (fraction_set ∩ given_numbers.toFinset = {-3/5, -22/7, -0.142857, 95/100}) :=
by sorry

end number_categorization_l958_95895


namespace grass_area_in_square_plot_l958_95868

theorem grass_area_in_square_plot (perimeter : ℝ) (h_perimeter : perimeter = 40) :
  let side_length := perimeter / 4
  let square_area := side_length ^ 2
  let circle_radius := side_length / 2
  let circle_area := π * circle_radius ^ 2
  let grass_area := square_area - circle_area
  grass_area = 100 - 25 * π :=
by sorry

end grass_area_in_square_plot_l958_95868


namespace smallest_solution_quartic_equation_l958_95876

theorem smallest_solution_quartic_equation :
  ∃ (x : ℝ), x^4 - 40*x^2 + 400 = 0 ∧
  x = -2 * Real.sqrt 5 ∧
  ∀ (y : ℝ), y^4 - 40*y^2 + 400 = 0 → y ≥ x :=
by sorry

end smallest_solution_quartic_equation_l958_95876


namespace largest_integer_under_sqrt_constraint_l958_95849

theorem largest_integer_under_sqrt_constraint : 
  ∀ x : ℤ, (Real.sqrt (x^2 : ℝ) < 15) → x ≤ 14 ∧ ∃ y : ℤ, y > x ∧ ¬(Real.sqrt (y^2 : ℝ) < 15) :=
sorry

end largest_integer_under_sqrt_constraint_l958_95849


namespace vector_perpendicular_and_parallel_l958_95827

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-3, 2]

theorem vector_perpendicular_and_parallel (k : ℝ) :
  (∀ i : Fin 2, (k * a i + b i) * (a i - 3 * b i) = 0) → k = 19 ∧
  (∃ t : ℝ, ∀ i : Fin 2, k * a i + b i = t * (a i - 3 * b i)) → k = -1/3 := by
  sorry

end vector_perpendicular_and_parallel_l958_95827


namespace right_triangle_side_lengths_l958_95824

theorem right_triangle_side_lengths (a : ℝ) : 
  (∃ (x y z : ℝ), x = a + 1 ∧ y = a + 2 ∧ z = a + 3 ∧ 
  x^2 + y^2 = z^2) → a = 2 := by
sorry

end right_triangle_side_lengths_l958_95824


namespace min_area_archimedean_triangle_l958_95858

/-- Represents a parabola with equation y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- Represents a chord of a parabola -/
structure Chord (para : Parabola) where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- Represents the Archimedean triangle of a parabola and chord -/
structure ArchimedeanTriangle (para : Parabola) (chord : Chord para) where
  Q : ℝ × ℝ

/-- Predicate to check if a chord passes through the focus of a parabola -/
def passes_through_focus (para : Parabola) (chord : Chord para) : Prop :=
  ∃ t : ℝ, chord.A = (para.p / 2, t) ∨ chord.B = (para.p / 2, t)

/-- Calculate the area of a triangle given its vertices -/
def triangle_area (A B Q : ℝ × ℝ) : ℝ := sorry

/-- The main theorem: The minimum area of the Archimedean triangle is p^2 -/
theorem min_area_archimedean_triangle (para : Parabola) 
  (chord : Chord para) (arch_tri : ArchimedeanTriangle para chord)
  (h_focus : passes_through_focus para chord) :
  ∃ (min_area : ℝ), 
    (∀ (other_chord : Chord para) (other_tri : ArchimedeanTriangle para other_chord),
      passes_through_focus para other_chord → 
      triangle_area arch_tri.Q chord.A chord.B ≤ triangle_area other_tri.Q other_chord.A other_chord.B) ∧
    min_area = para.p^2 := by
  sorry

end min_area_archimedean_triangle_l958_95858


namespace system_positive_solution_l958_95856

theorem system_positive_solution (a b : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, 
    x₁ - x₂ = a ∧ 
    x₃ - x₄ = b ∧ 
    x₁ + x₂ + x₃ + x₄ = 1 ∧ 
    x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0) ↔ 
  abs a + abs b < 1 :=
by sorry

end system_positive_solution_l958_95856


namespace fraction_comparison_l958_95890

theorem fraction_comparison (x : ℝ) (hx : x > 0) :
  ∀ n : ℕ, (x^n + 1) / (x^(n+1) + 1) > (x^(n+1) + 1) / (x^(n+2) + 1) :=
by sorry

end fraction_comparison_l958_95890


namespace phone_number_guess_probability_l958_95894

theorem phone_number_guess_probability : 
  ∀ (total_digits : ℕ) (correct_digit : ℕ),
  total_digits = 10 →
  correct_digit < total_digits →
  (1 - 1 / total_digits) * (1 / (total_digits - 1)) = 1 / 10 :=
by
  sorry

end phone_number_guess_probability_l958_95894


namespace unknown_table_has_one_leg_l958_95843

/-- The number of legs on the table with the unknown number of legs -/
def unknown_table_legs : ℕ := sorry

/-- The total number of legs in the room -/
def total_legs : ℕ := 40

/-- The number of legs on all furniture except the unknown table -/
def known_legs : ℕ := 
  4 * 4 +  -- 4 tables with 4 legs each
  1 * 4 +  -- 1 sofa with 4 legs
  2 * 4 +  -- 2 chairs with 4 legs each
  3 * 3 +  -- 3 tables with 3 legs each
  1 * 2    -- 1 rocking chair with 2 legs

theorem unknown_table_has_one_leg : 
  unknown_table_legs = 1 :=
by
  sorry

#check unknown_table_has_one_leg

end unknown_table_has_one_leg_l958_95843


namespace slope_range_for_line_l958_95817

/-- Given a line passing through (1, 1) with y-intercept in (0, 2), its slope is in (-1, 1) -/
theorem slope_range_for_line (l : Set (ℝ × ℝ)) (y_intercept : ℝ) (k : ℝ) : 
  (∀ x y, (x, y) ∈ l ↔ y = k * x + (1 - k)) →  -- Line equation
  (1, 1) ∈ l →  -- Line passes through (1, 1)
  0 < y_intercept ∧ y_intercept < 2 →  -- y-intercept in (0, 2)
  y_intercept = 1 - k →  -- y-intercept calculation
  -1 < k ∧ k < 1 :=  -- Slope is in (-1, 1)
by sorry

end slope_range_for_line_l958_95817


namespace coefficient_of_x_l958_95808

theorem coefficient_of_x (x : ℝ) : 
  ∃ (a b c d e : ℝ), (1 + x) * (x - 2/x)^3 = a*x^3 + b*x^2 + (-6)*x + c + d/x + e/(x^2) :=
sorry

end coefficient_of_x_l958_95808


namespace fraction_of_nuts_eaten_l958_95801

def initial_nuts : ℕ := 30
def remaining_nuts : ℕ := 5

theorem fraction_of_nuts_eaten :
  (initial_nuts - remaining_nuts) / initial_nuts = 5 / 6 := by
  sorry

end fraction_of_nuts_eaten_l958_95801


namespace b_joined_after_five_months_l958_95842

/-- Represents the number of months after A started the business that B joined as a partner. -/
def months_before_b_joined : ℕ := 5

/-- Represents A's initial investment in rupees. -/
def a_investment : ℕ := 3500

/-- Represents B's investment in rupees. -/
def b_investment : ℕ := 9000

/-- Represents the total number of months in a year. -/
def months_in_year : ℕ := 12

/-- Theorem stating that B joined the business 5 months after A started, given the conditions. -/
theorem b_joined_after_five_months :
  let a_capital := a_investment * months_in_year
  let b_capital := b_investment * (months_in_year - months_before_b_joined)
  (a_capital : ℚ) / b_capital = 2 / 3 :=
sorry

end b_joined_after_five_months_l958_95842


namespace picnic_attendance_theorem_l958_95805

/-- Represents the percentage of employees who are men -/
def male_percentage : ℝ := 0.5

/-- Represents the percentage of women who attended the picnic -/
def women_attendance_percentage : ℝ := 0.4

/-- Represents the percentage of all employees who attended the picnic -/
def total_attendance_percentage : ℝ := 0.3

/-- Represents the percentage of men who attended the picnic -/
def male_attendance_percentage : ℝ := 0.2

theorem picnic_attendance_theorem :
  male_attendance_percentage * male_percentage + 
  women_attendance_percentage * (1 - male_percentage) = 
  total_attendance_percentage := by
  sorry

#check picnic_attendance_theorem

end picnic_attendance_theorem_l958_95805


namespace polynomial_division_theorem_l958_95863

/-- The polynomial x^4 - 6x^3 + 16x^2 - 25x + 10 -/
def P (x : ℝ) : ℝ := x^4 - 6*x^3 + 16*x^2 - 25*x + 10

/-- The divisor x^2 - 2x + k -/
def D (x k : ℝ) : ℝ := x^2 - 2*x + k

/-- The remainder x + a -/
def R (x a : ℝ) : ℝ := x + a

/-- There exist q such that P(x) = D(x, k) * q(x) + R(x, a) for all x -/
def divides (k a : ℝ) : Prop :=
  ∃ q : ℝ → ℝ, ∀ x, P x = D x k * q x + R x a

theorem polynomial_division_theorem :
  ∀ k a : ℝ, divides k a ↔ k = 5 ∧ a = -5 := by sorry

end polynomial_division_theorem_l958_95863


namespace inequality_proof_l958_95880

theorem inequality_proof (x : ℝ) (n : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ 1) (h3 : 0 < n) :
  (1 + x)^n ≥ (1 - x)^n + 2 * n * x * (1 - x^2)^((n - 1) / 2) := by
  sorry

end inequality_proof_l958_95880


namespace intersection_equals_one_l958_95834

def M : Set ℕ := {0, 1}

def N : Set ℕ := {y | ∃ x ∈ M, y = 2*x + 1}

theorem intersection_equals_one : M ∩ N = {1} := by
  sorry

end intersection_equals_one_l958_95834


namespace disjunction_false_l958_95835

-- Define proposition p
def prop_p (a b : ℝ) : Prop := (a * b > 0) → (|a| + |b| > |a + b|)

-- Define proposition q
def prop_q (a b c : ℝ) : Prop := (c > a^2 + b^2) → (c > 2*a*b)

-- Theorem statement
theorem disjunction_false :
  ¬(∀ a b : ℝ, prop_p a b ∨ ¬(∀ c : ℝ, prop_q a b c)) :=
sorry

end disjunction_false_l958_95835


namespace root_difference_implies_k_value_l958_95837

theorem root_difference_implies_k_value (k : ℝ) : 
  (∃ r s : ℝ, r^2 + k*r + 8 = 0 ∧ s^2 + k*s + 8 = 0 ∧ 
   (r+7)^2 - k*(r+7) + 8 = 0 ∧ (s+7)^2 - k*(s+7) + 8 = 0) → 
  k = 7 := by
  sorry

end root_difference_implies_k_value_l958_95837


namespace debt_work_hours_l958_95898

def initial_debt : ℝ := 100
def payment : ℝ := 40
def hourly_rate : ℝ := 15

theorem debt_work_hours : 
  (initial_debt - payment) / hourly_rate = 4 := by sorry

end debt_work_hours_l958_95898


namespace basketball_volume_after_drilling_l958_95804

/-- The volume of a basketball after drilling holes for handles -/
theorem basketball_volume_after_drilling (d : ℝ) (r1 r2 h : ℝ) :
  d = 50 ∧ r1 = 2 ∧ r2 = 1.5 ∧ h = 10 →
  (4/3 * π * (d/2)^3) - (2 * π * r1^2 * h + 2 * π * r2^2 * h) = (62250/3) * π :=
by sorry

end basketball_volume_after_drilling_l958_95804


namespace correct_arrangement_l958_95831

-- Define the squares
inductive Square
| A | B | C | D | E | F | G | One | Nine

-- Define the arrow directions
def points_to : Square → Square → Prop :=
  fun s1 s2 => match s1, s2 with
    | Square.One, Square.B => True
    | Square.B, Square.E => True
    | Square.E, Square.C => True
    | Square.C, Square.D => True
    | Square.D, Square.A => True
    | Square.A, Square.G => True
    | Square.G, Square.F => True
    | Square.F, Square.Nine => True
    | _, _ => False

-- Define the arrangement
def arrangement : Square → Nat
| Square.A => 6
| Square.B => 2
| Square.C => 4
| Square.D => 5
| Square.E => 3
| Square.F => 8
| Square.G => 7
| Square.One => 1
| Square.Nine => 9

-- Theorem statement
theorem correct_arrangement :
  ∀ s : Square, s ≠ Square.One ∧ s ≠ Square.Nine →
    ∃ s' : Square, points_to s s' ∧ arrangement s' = arrangement s + 1 :=
by sorry

end correct_arrangement_l958_95831


namespace triangle_isosceles_from_quadrilateral_property_l958_95802

structure Triangle where
  angles : Fin 3 → ℝ
  sum_eq_pi : angles 0 + angles 1 + angles 2 = π

structure Quadrilateral where
  angles : Fin 4 → ℝ
  sum_eq_2pi : angles 0 + angles 1 + angles 2 + angles 3 = 2 * π

def has_sum_angle_property (t : Triangle) (q : Quadrilateral) : Prop :=
  ∀ (i j : Fin 3), i ≠ j → ∃ (k : Fin 4), q.angles k = t.angles i + t.angles j

def is_isosceles (t : Triangle) : Prop :=
  ∃ (i j : Fin 3), i ≠ j ∧ t.angles i = t.angles j

theorem triangle_isosceles_from_quadrilateral_property
  (t : Triangle) (q : Quadrilateral) (h : has_sum_angle_property t q) :
  is_isosceles t :=
sorry

end triangle_isosceles_from_quadrilateral_property_l958_95802


namespace quadratic_root_sqrt5_minus_2_l958_95872

theorem quadratic_root_sqrt5_minus_2 :
  ∃ (a b c : ℚ), a = 1 ∧ 
    (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = Real.sqrt 5 - 2 ∨ x = -(Real.sqrt 5) - 2) :=
sorry

end quadratic_root_sqrt5_minus_2_l958_95872


namespace apple_purchase_remainder_l958_95829

theorem apple_purchase_remainder (mark_money carolyn_money apple_cost : ℚ) : 
  mark_money = 2/3 →
  carolyn_money = 1/5 →
  apple_cost = 1/2 →
  mark_money + carolyn_money - apple_cost = 11/30 := by
sorry

end apple_purchase_remainder_l958_95829


namespace workshop_average_salary_l958_95881

-- Define the total number of workers
def total_workers : ℕ := 14

-- Define the number of technicians
def num_technicians : ℕ := 7

-- Define the average salary of technicians
def avg_salary_technicians : ℕ := 12000

-- Define the average salary of other workers
def avg_salary_others : ℕ := 8000

-- Theorem statement
theorem workshop_average_salary :
  (num_technicians * avg_salary_technicians + (total_workers - num_technicians) * avg_salary_others) / total_workers = 10000 := by
  sorry

end workshop_average_salary_l958_95881


namespace middle_part_of_proportional_division_l958_95879

theorem middle_part_of_proportional_division (total : ℕ) (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 120 ∧ a = 3 ∧ b = 5 ∧ c = 7 →
  ∃ (x : ℚ), x > 0 ∧ a * x + b * x + c * x = total ∧ (∃ (n : ℕ), a * x = n ∨ b * x = n ∨ c * x = n) →
  b * x = 40 := by
  sorry

end middle_part_of_proportional_division_l958_95879


namespace det_transformation_l958_95851

/-- Given a 2x2 matrix with determinant -3, prove that a specific transformation of this matrix also has determinant -3 -/
theorem det_transformation (x y z w : ℝ) 
  (h : Matrix.det !![x, y; z, w] = -3) :
  Matrix.det !![x + 2*z, y + 2*w; z, w] = -3 := by
sorry

end det_transformation_l958_95851


namespace exists_four_digit_number_divisible_by_101_when_reversed_l958_95838

/-- Reverses a four-digit number -/
def reverse (n : ℕ) : ℕ :=
  (n % 10) * 1000 + ((n / 10) % 10) * 100 + ((n / 100) % 10) * 10 + (n / 1000)

/-- Checks if a number has distinct non-zero digits -/
def has_distinct_nonzero_digits (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n % 10 ≠ 0) ∧ ((n / 10) % 10 ≠ 0) ∧ ((n / 100) % 10 ≠ 0) ∧ (n / 1000 ≠ 0) ∧
  (n % 10 ≠ (n / 10) % 10) ∧ (n % 10 ≠ (n / 100) % 10) ∧ (n % 10 ≠ n / 1000) ∧
  ((n / 10) % 10 ≠ (n / 100) % 10) ∧ ((n / 10) % 10 ≠ n / 1000) ∧
  ((n / 100) % 10 ≠ n / 1000)

theorem exists_four_digit_number_divisible_by_101_when_reversed :
  ∃ n : ℕ, has_distinct_nonzero_digits n ∧ (n + reverse n) % 101 = 0 :=
sorry

end exists_four_digit_number_divisible_by_101_when_reversed_l958_95838


namespace xyz_inequality_l958_95871

theorem xyz_inequality (x y z : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) 
  (hsum : x + y + z = 1) : 
  0 ≤ x*y + y*z + x*z - 2*x*y*z ∧ x*y + y*z + x*z - 2*x*y*z ≤ 7/27 := by
  sorry

end xyz_inequality_l958_95871


namespace min_value_theorem_l958_95884

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x^2 + 12*x + 108/x^4 ≥ 36 ∧ 
  (x^2 + 12*x + 108/x^4 = 36 ↔ x = 3) :=
by sorry

end min_value_theorem_l958_95884


namespace erin_has_90_dollars_l958_95896

/-- The amount of money Erin has after emptying all machines in her launderette -/
def erins_money_after_emptying (quarters_per_machine : ℕ) (dimes_per_machine : ℕ) (num_machines : ℕ) : ℚ :=
  (quarters_per_machine * (25 : ℚ) / 100 + dimes_per_machine * (10 : ℚ) / 100) * num_machines

/-- Theorem stating that Erin will have $90.00 after emptying all machines -/
theorem erin_has_90_dollars :
  erins_money_after_emptying 80 100 3 = 90 :=
by sorry

end erin_has_90_dollars_l958_95896


namespace unique_solution_for_equation_l958_95885

theorem unique_solution_for_equation :
  ∃! (m n : ℝ), 21 * (m^2 + n) + 21 * Real.sqrt n = 21 * (-m^3 + n^2) + 21 * m^2 * n ∧ m = -1 ∧ n = 0 :=
by sorry

end unique_solution_for_equation_l958_95885


namespace solve_for_b_l958_95857

theorem solve_for_b (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * b) : b = 49 := by
  sorry

end solve_for_b_l958_95857


namespace pentagon_count_l958_95839

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of distinct points on the circle -/
def num_points : ℕ := 15

/-- The number of vertices in a pentagon -/
def pentagon_vertices : ℕ := 5

/-- Theorem: The number of different convex pentagons that can be formed
    by selecting 5 points from 15 distinct points on the circumference of a circle
    is equal to 3003 -/
theorem pentagon_count :
  binomial num_points pentagon_vertices = 3003 := by
  sorry

end pentagon_count_l958_95839


namespace rectangle_perimeter_l958_95841

/-- The perimeter of a rectangle with length 0.54 meters and width 0.08 meters shorter than the length is 2 meters. -/
theorem rectangle_perimeter : 
  let length : ℝ := 0.54
  let width_difference : ℝ := 0.08
  let width : ℝ := length - width_difference
  let perimeter : ℝ := 2 * (length + width)
  perimeter = 2 := by sorry

end rectangle_perimeter_l958_95841


namespace square_difference_l958_95853

theorem square_difference (x y : ℝ) (h1 : x + y = 18) (h2 : x - y = 4) : x^2 - y^2 = 72 := by
  sorry

end square_difference_l958_95853


namespace average_weight_increase_l958_95889

theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 5 * initial_average
  let final_total := initial_total - 40 + 90
  let final_average := final_total / 5
  final_average - initial_average = 10 := by
sorry

end average_weight_increase_l958_95889


namespace smallest_with_eight_factors_l958_95891

/-- The number of distinct positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- The set of distinct positive factors of a positive integer -/
def factors (n : ℕ+) : Finset ℕ := sorry

theorem smallest_with_eight_factors : 
  (∀ m : ℕ+, m < 24 → num_factors m ≠ 8) ∧ 
  num_factors 24 = 8 ∧
  factors 24 = {1, 2, 3, 4, 6, 8, 12, 24} := by sorry

end smallest_with_eight_factors_l958_95891


namespace total_turnips_after_selling_l958_95892

/-- The total number of turnips after selling some -/
def totalTurnipsAfterSelling (melanieTurnips bennyTurnips sarahTurnips davidTurnips melanieSold davidSold : ℕ) : ℕ :=
  (melanieTurnips - melanieSold) + bennyTurnips + sarahTurnips + (davidTurnips - davidSold)

/-- Theorem stating the total number of turnips after selling -/
theorem total_turnips_after_selling :
  totalTurnipsAfterSelling 139 113 195 87 32 15 = 487 := by
  sorry

#eval totalTurnipsAfterSelling 139 113 195 87 32 15

end total_turnips_after_selling_l958_95892


namespace circle_area_equality_l958_95814

theorem circle_area_equality (r₁ r₂ r₃ : ℝ) : 
  r₁ = 17 → r₂ = 27 → r₃ = 10 * Real.sqrt 11 → 
  π * r₃^2 = π * (r₂^2 - r₁^2) := by
  sorry

end circle_area_equality_l958_95814


namespace athlete_stop_point_l958_95878

/-- Represents a rectangular square with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a point on the perimeter of a rectangle -/
structure PerimeterPoint where
  distance : ℝ  -- Distance from a chosen starting point

/-- The athlete's run around the rectangular square -/
def athleteRun (rect : Rectangle) (start : PerimeterPoint) (distance : ℝ) : PerimeterPoint :=
  sorry

theorem athlete_stop_point (rect : Rectangle) (start : PerimeterPoint) :
  let totalDistance : ℝ := 15500  -- 15.5 km in meters
  rect.length = 900 ∧ rect.width = 600 ∧ start.distance = 550 →
  (athleteRun rect start totalDistance).distance = 150 :=
sorry

end athlete_stop_point_l958_95878


namespace point_translation_l958_95877

def initial_point : ℝ × ℝ := (0, 1)
def downward_translation : ℝ := 2
def leftward_translation : ℝ := 4

theorem point_translation :
  (initial_point.1 - leftward_translation, initial_point.2 - downward_translation) = (-4, -1) := by
  sorry

end point_translation_l958_95877


namespace lcm_gcd_275_570_l958_95859

theorem lcm_gcd_275_570 : 
  (Nat.lcm 275 570 = 31350) ∧ (Nat.gcd 275 570 = 5) := by sorry

end lcm_gcd_275_570_l958_95859


namespace geometric_solid_height_l958_95820

-- Define the geometric solid
structure GeometricSolid where
  radius1 : ℝ
  radius2 : ℝ
  water_height1 : ℝ
  water_height2 : ℝ

-- Define the theorem
theorem geometric_solid_height (s : GeometricSolid) 
  (h1 : s.radius1 = 1)
  (h2 : s.radius2 = 3)
  (h3 : s.water_height1 = 20)
  (h4 : s.water_height2 = 28) :
  ∃ (total_height : ℝ), total_height = 29 := by
  sorry

end geometric_solid_height_l958_95820


namespace jordan_list_count_l958_95830

def smallest_square_multiple (n : ℕ) : ℕ := 
  Nat.lcm (n^2) n

def smallest_cube_multiple (n : ℕ) : ℕ := 
  Nat.lcm (n^3) n

theorem jordan_list_count : 
  let lower_bound := smallest_square_multiple 30
  let upper_bound := smallest_cube_multiple 30
  (upper_bound - lower_bound) / 30 + 1 = 871 := by sorry

end jordan_list_count_l958_95830


namespace jims_journey_distance_l958_95816

/-- The total distance of Jim's journey, given the miles driven and miles left to drive -/
def total_distance (miles_driven : ℕ) (miles_left : ℕ) : ℕ :=
  miles_driven + miles_left

/-- Theorem stating that the total distance of Jim's journey is 1200 miles -/
theorem jims_journey_distance :
  total_distance 384 816 = 1200 := by sorry

end jims_journey_distance_l958_95816


namespace inverse_inequality_l958_95883

theorem inverse_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a < b) :
  1 / a > 1 / b := by
  sorry

end inverse_inequality_l958_95883


namespace million_millimeters_equals_one_kilometer_l958_95862

-- Define the conversion factors
def millimeters_per_meter : ℕ := 1000
def meters_per_kilometer : ℕ := 1000

-- Define the question
def million_millimeters : ℕ := 1000000

-- Theorem to prove
theorem million_millimeters_equals_one_kilometer :
  (million_millimeters / millimeters_per_meter) / meters_per_kilometer = 1 := by
  sorry

end million_millimeters_equals_one_kilometer_l958_95862


namespace power_function_inequality_l958_95867

theorem power_function_inequality : let f : ℝ → ℝ := fun x ↦ x^3
  let a : ℝ := f (Real.sqrt 3 / 3)
  let b : ℝ := f (Real.log π)
  let c : ℝ := f (Real.sqrt 2 / 2)
  a < c ∧ c < b := by sorry

end power_function_inequality_l958_95867


namespace triangle_existence_condition_l958_95846

def triangle_exists (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_existence_condition (x : ℕ) :
  x > 0 →
  (triangle_exists 8 11 (2 * x + 1) ↔ x ∈ ({2, 3, 4, 5, 6, 7, 8} : Set ℕ)) :=
by sorry

end triangle_existence_condition_l958_95846


namespace work_completion_days_l958_95812

/-- The number of days B can finish the work -/
def b_days : ℕ := 15

/-- The number of days B worked before leaving -/
def b_worked : ℕ := 10

/-- The number of days A needs to finish the remaining work after B left -/
def a_remaining : ℕ := 2

/-- The number of days A can finish the entire work -/
def a_days : ℕ := 6

theorem work_completion_days :
  (b_worked : ℚ) / b_days + a_remaining / a_days = 1 :=
sorry

end work_completion_days_l958_95812


namespace restaurant_hiring_l958_95873

theorem restaurant_hiring (initial_ratio_cooks initial_ratio_waiters new_ratio_cooks new_ratio_waiters num_cooks : ℕ) 
  (h1 : initial_ratio_cooks = 3)
  (h2 : initial_ratio_waiters = 10)
  (h3 : new_ratio_cooks = 3)
  (h4 : new_ratio_waiters = 14)
  (h5 : num_cooks = 9) :
  ∃ (initial_waiters hired_waiters : ℕ),
    initial_ratio_cooks * initial_waiters = initial_ratio_waiters * num_cooks ∧
    new_ratio_cooks * (initial_waiters + hired_waiters) = new_ratio_waiters * num_cooks ∧
    hired_waiters = 12 := by
  sorry


end restaurant_hiring_l958_95873


namespace square_congruence_mod_four_l958_95897

theorem square_congruence_mod_four (n : ℤ) : (n^2) % 4 = 0 ∨ (n^2) % 4 = 1 := by
  sorry

end square_congruence_mod_four_l958_95897


namespace consecutive_number_pair_l958_95875

theorem consecutive_number_pair (a b : ℤ) : 
  (a = 18 ∨ b = 18) → -- One of the numbers is 18
  abs (a - b) = 1 → -- The numbers are consecutive
  a + b = 35 → -- Their sum is 35
  (a + b) % 5 = 0 → -- The sum is divisible by 5
  (a = 17 ∨ b = 17) := by sorry

end consecutive_number_pair_l958_95875


namespace rectangle_area_l958_95818

theorem rectangle_area (square_side : ℝ) (circle_radius : ℝ) (rectangle_length : ℝ) (rectangle_breadth : ℝ) : 
  square_side ^ 2 = 16 →
  circle_radius = square_side →
  rectangle_length = 5 * circle_radius →
  rectangle_breadth = 11 →
  rectangle_length * rectangle_breadth = 220 := by
sorry

end rectangle_area_l958_95818


namespace wills_remaining_money_l958_95807

-- Define the given amounts
def initial_amount : ℚ := 74
def sweater_cost : ℚ := 9
def tshirt_cost : ℚ := 11
def shoes_cost : ℚ := 30
def refund_percentage : ℚ := 90 / 100

-- Define the theorem
theorem wills_remaining_money :
  let clothes_cost := sweater_cost + tshirt_cost
  let refund := shoes_cost * refund_percentage
  let remaining := initial_amount - clothes_cost - shoes_cost + refund
  remaining = 81 := by sorry

end wills_remaining_money_l958_95807


namespace cuboid_height_l958_95810

/-- Proves that the height of a cuboid with given base area and volume is 7 cm -/
theorem cuboid_height (base_area volume : ℝ) (h_base : base_area = 36) (h_volume : volume = 252) :
  volume / base_area = 7 := by
  sorry

end cuboid_height_l958_95810


namespace both_selected_probability_l958_95866

theorem both_selected_probability (ram_prob ravi_prob : ℚ) 
  (h1 : ram_prob = 2/7)
  (h2 : ravi_prob = 1/5) :
  ram_prob * ravi_prob = 2/35 := by
  sorry

end both_selected_probability_l958_95866


namespace equation_solution_l958_95860

theorem equation_solution : ∃ n : ℝ, 0.03 * n + 0.05 * (30 + n) + 2 = 8.5 ∧ n = 62.5 := by
  sorry

end equation_solution_l958_95860


namespace least_possible_third_side_length_l958_95806

theorem least_possible_third_side_length (a b c : ℝ) : 
  a = 8 → b = 15 → c > 0 → 
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → 
  c ≥ Real.sqrt 161 :=
by sorry

end least_possible_third_side_length_l958_95806


namespace inequality_proof_l958_95850

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) : 
  a + b + Real.sqrt 2 * c ≤ 2 := by
sorry

end inequality_proof_l958_95850


namespace remainder_987543_div_12_l958_95845

theorem remainder_987543_div_12 : 987543 % 12 = 7 := by
  sorry

end remainder_987543_div_12_l958_95845


namespace factorization_identity_l958_95882

theorem factorization_identity (a b : ℝ) : a^2 + a*b = a*(a + b) := by sorry

end factorization_identity_l958_95882


namespace probability_doubled_l958_95809

def total_clips : ℕ := 16
def red_clips : ℕ := 4
def blue_clips : ℕ := 5
def green_clips : ℕ := 7
def removed_clips : ℕ := 12

theorem probability_doubled :
  let initial_prob : ℚ := red_clips / total_clips
  let remaining_clips : ℕ := total_clips - removed_clips
  let new_prob : ℚ := red_clips / remaining_clips
  new_prob = 2 * initial_prob := by sorry

end probability_doubled_l958_95809


namespace friend_score_l958_95803

theorem friend_score (edward_score : ℕ) (total_score : ℕ) (friend_score : ℕ) : 
  edward_score = 7 → 
  total_score = 13 → 
  total_score = edward_score + friend_score →
  friend_score = 6 := by
sorry

end friend_score_l958_95803


namespace jills_age_l958_95893

/-- Given that the sum of Henry and Jill's present ages is 40,
    and 11 years ago Henry was twice the age of Jill,
    prove that Jill's present age is 17 years. -/
theorem jills_age (henry_age jill_age : ℕ) 
  (sum_ages : henry_age + jill_age = 40)
  (past_relation : henry_age - 11 = 2 * (jill_age - 11)) :
  jill_age = 17 := by
  sorry

end jills_age_l958_95893


namespace melissa_bought_four_packs_l958_95840

/-- The number of packs of tennis balls Melissa bought -/
def num_packs : ℕ := sorry

/-- The total cost of all packs in dollars -/
def total_cost : ℕ := 24

/-- The number of balls in each pack -/
def balls_per_pack : ℕ := 3

/-- The cost of each ball in dollars -/
def cost_per_ball : ℕ := 2

/-- Theorem stating that Melissa bought 4 packs of tennis balls -/
theorem melissa_bought_four_packs : num_packs = 4 := by sorry

end melissa_bought_four_packs_l958_95840


namespace rhombus_triangle_inscribed_circle_ratio_l958_95800

/-- Given a rhombus ABCD with acute angle α and a triangle ABC formed by two sides of the rhombus
    and its longer diagonal, this theorem states that the ratio of the radius of the circle
    inscribed in the rhombus to the radius of the circle inscribed in the triangle ABC
    is equal to 1 + cos(α/2). -/
theorem rhombus_triangle_inscribed_circle_ratio (α : Real) 
  (h1 : 0 < α ∧ α < π / 2) : 
  ∃ (r1 r2 : Real), r1 > 0 ∧ r2 > 0 ∧
    (r1 / r2 = 1 + Real.cos (α / 2)) := by
  sorry

end rhombus_triangle_inscribed_circle_ratio_l958_95800
