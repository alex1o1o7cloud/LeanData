import Mathlib

namespace NUMINAMATH_CALUDE_modulus_of_one_plus_i_l1111_111169

theorem modulus_of_one_plus_i :
  let z : ℂ := 1 + Complex.I
  Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_one_plus_i_l1111_111169


namespace NUMINAMATH_CALUDE_one_eighth_of_2_36_equals_2_y_l1111_111188

theorem one_eighth_of_2_36_equals_2_y (y : ℕ) : (1 / 8 : ℝ) * 2^36 = 2^y → y = 33 := by
  sorry

end NUMINAMATH_CALUDE_one_eighth_of_2_36_equals_2_y_l1111_111188


namespace NUMINAMATH_CALUDE_linear_function_kb_positive_l1111_111184

/-- A linear function passing through the second, third, and fourth quadrants -/
structure LinearFunction where
  k : ℝ
  b : ℝ
  second_quadrant : ∃ x y, x < 0 ∧ y > 0 ∧ y = k * x + b
  third_quadrant : ∃ x y, x < 0 ∧ y < 0 ∧ y = k * x + b
  fourth_quadrant : ∃ x y, x > 0 ∧ y < 0 ∧ y = k * x + b

/-- Theorem: For a linear function passing through the second, third, and fourth quadrants, kb > 0 -/
theorem linear_function_kb_positive (f : LinearFunction) : f.k * f.b > 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_kb_positive_l1111_111184


namespace NUMINAMATH_CALUDE_a5_value_l1111_111121

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a5_value (a : ℕ → ℤ) (h_arith : arithmetic_sequence a) 
  (h3 : a 3 = -5) (h7 : a 7 = -1) : a 5 = -3 := by
  sorry

end NUMINAMATH_CALUDE_a5_value_l1111_111121


namespace NUMINAMATH_CALUDE_max_sum_AB_l1111_111123

def is_digit (n : ℕ) : Prop := n ≤ 9

theorem max_sum_AB :
  ∃ (A B C D : ℕ),
    is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    (C + D > 1) ∧
    (∃ k : ℕ, k * (C + D) = A + B) ∧
    (∀ A' B' C' D' : ℕ,
      is_digit A' ∧ is_digit B' ∧ is_digit C' ∧ is_digit D' →
      A' ≠ B' ∧ A' ≠ C' ∧ A' ≠ D' ∧ B' ≠ C' ∧ B' ≠ D' ∧ C' ≠ D' →
      (C' + D' > 1) →
      (∃ k' : ℕ, k' * (C' + D') = A' + B') →
      A' + B' ≤ A + B) →
    A + B = 15 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_AB_l1111_111123


namespace NUMINAMATH_CALUDE_parallel_lines_slope_l1111_111148

/-- If the lines x + 2y = 3 and nx + my = 4 are parallel, then m = 2n -/
theorem parallel_lines_slope (n m : ℝ) : 
  (∀ x y : ℝ, x + 2*y = 3 → nx + m*y = 4) →  -- Lines exist
  (∃ k : ℝ, ∀ x : ℝ, 
    (3 - x) / 2 = (4 - n*x) / m) →           -- Lines are parallel
  m = 2*n :=                                 -- Conclusion
by sorry

end NUMINAMATH_CALUDE_parallel_lines_slope_l1111_111148


namespace NUMINAMATH_CALUDE_population_average_age_l1111_111118

theorem population_average_age 
  (k : ℕ) 
  (h_k_pos : k > 0) 
  (men_count : ℕ := 7 * k)
  (women_count : ℕ := 8 * k)
  (men_avg_age : ℚ := 36)
  (women_avg_age : ℚ := 30) :
  let total_population := men_count + women_count
  let total_age := men_count * men_avg_age + women_count * women_avg_age
  total_age / total_population = 164 / 5 := by
sorry

#eval (164 : ℚ) / 5  -- Should evaluate to 32.8

end NUMINAMATH_CALUDE_population_average_age_l1111_111118


namespace NUMINAMATH_CALUDE_interest_rate_is_one_percent_l1111_111136

/-- Calculate the interest rate given principal, time, and total simple interest -/
def calculate_interest_rate (principal : ℚ) (time : ℚ) (total_interest : ℚ) : ℚ :=
  (total_interest * 100) / (principal * time)

/-- Theorem stating that given the specific values, the interest rate is 1% -/
theorem interest_rate_is_one_percent :
  let principal : ℚ := 133875
  let time : ℚ := 3
  let total_interest : ℚ := 4016.25
  calculate_interest_rate principal time total_interest = 1 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_is_one_percent_l1111_111136


namespace NUMINAMATH_CALUDE_coin_arrangement_count_l1111_111146

def coin_diameter_10_filler : ℕ := 19
def coin_diameter_50_filler : ℕ := 22
def total_length : ℕ := 1000
def min_coins : ℕ := 50

theorem coin_arrangement_count : 
  ∃ (x y : ℕ), 
    x * coin_diameter_10_filler + y * coin_diameter_50_filler = total_length ∧ 
    x + y ≥ min_coins ∧
    Nat.choose (x + y) y = 270725 := by
  sorry

end NUMINAMATH_CALUDE_coin_arrangement_count_l1111_111146


namespace NUMINAMATH_CALUDE_odd_number_representation_l1111_111110

theorem odd_number_representation (k : ℤ) : 
  (k % 2 = 1) → 
  ((∃ n : ℤ, 2 * n + 3 = k) ∧ 
   ¬(∀ k : ℤ, k % 2 = 1 → ∃ n : ℤ, 4 * n - 1 = k)) := by
sorry

end NUMINAMATH_CALUDE_odd_number_representation_l1111_111110


namespace NUMINAMATH_CALUDE_ian_lottery_payment_l1111_111101

theorem ian_lottery_payment (total : ℝ) (left : ℝ) (colin : ℝ) (helen : ℝ) (benedict : ℝ) :
  total = 100 →
  helen = 2 * colin →
  benedict = helen / 2 →
  left = 20 →
  total = colin + helen + benedict + left →
  colin = 20 := by
sorry

end NUMINAMATH_CALUDE_ian_lottery_payment_l1111_111101


namespace NUMINAMATH_CALUDE_eva_orange_count_l1111_111157

/-- Calculates the number of oranges Eva needs to buy given her dietary requirements --/
def calculate_oranges (total_days : ℕ) (orange_frequency : ℕ) : ℕ :=
  total_days / orange_frequency

/-- Theorem stating that Eva needs to buy 10 oranges given her dietary requirements --/
theorem eva_orange_count : calculate_oranges 30 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_eva_orange_count_l1111_111157


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1111_111175

/-- Given a geometric sequence {a_n} with sum S_n = b(-2)^(n-1) - a, prove that a/b = -1/2 -/
theorem geometric_sequence_ratio (b a : ℝ) (S : ℕ → ℝ) (a_n : ℕ → ℝ) :
  (∀ n : ℕ, S n = b * (-2)^(n - 1) - a) →
  (∀ n : ℕ, a_n n = S n - S (n - 1)) →
  (a_n 1 = b - a) →
  a / b = -1 / 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1111_111175


namespace NUMINAMATH_CALUDE_smaller_solution_quadratic_equation_l1111_111134

theorem smaller_solution_quadratic_equation :
  let f : ℝ → ℝ := λ x ↦ x^2 - 15*x - 56
  let sol₁ : ℝ := (15 - Real.sqrt 449) / 2
  let sol₂ : ℝ := (15 + Real.sqrt 449) / 2
  f sol₁ = 0 ∧ f sol₂ = 0 ∧ sol₁ < sol₂ ∧ 
  ∀ x : ℝ, f x = 0 → x = sol₁ ∨ x = sol₂ :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_quadratic_equation_l1111_111134


namespace NUMINAMATH_CALUDE_value_of_expression_l1111_111133

theorem value_of_expression (x : ℝ) (h : x = 2) : (3*x + 4)^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1111_111133


namespace NUMINAMATH_CALUDE_largest_gold_coins_l1111_111192

theorem largest_gold_coins : ∃ n : ℕ, n < 150 ∧ ∃ k : ℕ, n = 13 * k + 3 ∧ 
  ∀ m : ℕ, m < 150 → (∃ j : ℕ, m = 13 * j + 3) → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_gold_coins_l1111_111192


namespace NUMINAMATH_CALUDE_area_ratio_triangle_to_hexagon_l1111_111165

/-- A regular hexagon with vertices labeled A, B, C, D, E, F. -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : sorry

/-- The area of a regular hexagon. -/
def area_hexagon (h : RegularHexagon) : ℝ := sorry

/-- Triangle formed by connecting every second vertex of the hexagon. -/
def triangle_ACE (h : RegularHexagon) : sorry := sorry

/-- The area of triangle ACE. -/
def area_triangle_ACE (h : RegularHexagon) : ℝ := sorry

/-- Theorem stating that the ratio of the area of triangle ACE to the area of the regular hexagon is 2/3. -/
theorem area_ratio_triangle_to_hexagon (h : RegularHexagon) :
  (area_triangle_ACE h) / (area_hexagon h) = 2/3 := by sorry

end NUMINAMATH_CALUDE_area_ratio_triangle_to_hexagon_l1111_111165


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1111_111163

/-- Represents a quadratic function y = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Evaluates the quadratic function at a given x -/
def evaluate (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- The theorem to be proved -/
theorem quadratic_expression_value (f : QuadraticFunction)
  (h1 : evaluate f (-2) = -2.5)
  (h2 : evaluate f (-1) = -5)
  (h3 : evaluate f 0 = -2.5)
  (h4 : evaluate f 1 = 5)
  (h5 : evaluate f 2 = 17.5) :
  16 * f.a - 4 * f.b + f.c = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1111_111163


namespace NUMINAMATH_CALUDE_cos_18_degrees_l1111_111166

theorem cos_18_degrees : Real.cos (18 * π / 180) = (1 + Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_18_degrees_l1111_111166


namespace NUMINAMATH_CALUDE_computer_literate_female_employees_l1111_111122

/-- Given an office in Singapore with the following conditions:
  * There are 1100 total employees
  * 60% of employees are female
  * 50% of male employees are computer literate
  * 62% of all employees are computer literate
  Prove that the number of female employees who are computer literate is 462 -/
theorem computer_literate_female_employees 
  (total_employees : ℕ) 
  (female_percentage : ℚ)
  (male_literate_percentage : ℚ)
  (total_literate_percentage : ℚ)
  (h1 : total_employees = 1100)
  (h2 : female_percentage = 60 / 100)
  (h3 : male_literate_percentage = 50 / 100)
  (h4 : total_literate_percentage = 62 / 100) :
  ↑⌊(total_literate_percentage * total_employees - 
     male_literate_percentage * ((1 - female_percentage) * total_employees))⌋ = 462 := by
  sorry

end NUMINAMATH_CALUDE_computer_literate_female_employees_l1111_111122


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1111_111108

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 6) (h2 : x ≠ -5) :
  (7 * x + 11) / (x^2 - x - 30) = (53 / 11) / (x - 6) + (24 / 11) / (x + 5) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1111_111108


namespace NUMINAMATH_CALUDE_orthogonal_projection_locus_l1111_111152

/-- Given a line (x/a) + (y/b) = 1 where (1/a^2) + (1/b^2) = 1/c^2 (c constant),
    the orthogonal projection of the origin on this line always lies on the circle x^2 + y^2 = c^2 -/
theorem orthogonal_projection_locus (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c > 0) :
  (1 / a^2 + 1 / b^2 = 1 / c^2) →
  ∃ (x y : ℝ), (x / a + y / b = 1) ∧ 
               (y = (a / b) * x) ∧
               (x^2 + y^2 = c^2) :=
by sorry

end NUMINAMATH_CALUDE_orthogonal_projection_locus_l1111_111152


namespace NUMINAMATH_CALUDE_special_polygon_has_eight_sides_l1111_111117

/-- A polygon with n sides where the sum of interior angles is 3 times the sum of exterior angles -/
structure SpecialPolygon where
  n : ℕ
  interior_sum : ℝ
  exterior_sum : ℝ
  h1 : interior_sum = (n - 2) * 180
  h2 : exterior_sum = 360
  h3 : interior_sum = 3 * exterior_sum

/-- Theorem: A SpecialPolygon has 8 sides -/
theorem special_polygon_has_eight_sides (p : SpecialPolygon) : p.n = 8 := by
  sorry

end NUMINAMATH_CALUDE_special_polygon_has_eight_sides_l1111_111117


namespace NUMINAMATH_CALUDE_right_angled_triangle_set_l1111_111199

theorem right_angled_triangle_set : ∃! (a b c : ℝ), 
  ((a = 1 ∧ b = Real.sqrt 2 ∧ c = Real.sqrt 3) ∨
   (a = 1 ∧ b = 2 ∧ c = 3) ∨
   (a = 2 ∧ b = 3 ∧ c = 4) ∨
   (a = Real.sqrt 2 ∧ b = 3 ∧ c = 5)) ∧
  a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_angled_triangle_set_l1111_111199


namespace NUMINAMATH_CALUDE_sin_arccos_12_13_l1111_111154

theorem sin_arccos_12_13 : Real.sin (Real.arccos (12/13)) = 5/13 := by
  sorry

end NUMINAMATH_CALUDE_sin_arccos_12_13_l1111_111154


namespace NUMINAMATH_CALUDE_power_of_two_sum_l1111_111172

theorem power_of_two_sum (m n : ℕ+) (a b : ℝ) 
  (h1 : 2^(m : ℕ) = a) 
  (h2 : 2^(n : ℕ) = b) : 
  2^((m + n : ℕ+) : ℕ) = a * b := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_sum_l1111_111172


namespace NUMINAMATH_CALUDE_polygon_with_900_degree_sum_is_heptagon_l1111_111197

theorem polygon_with_900_degree_sum_is_heptagon :
  ∀ n : ℕ, 
    n ≥ 3 →
    (n - 2) * 180 = 900 →
    n = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_with_900_degree_sum_is_heptagon_l1111_111197


namespace NUMINAMATH_CALUDE_tomato_theorem_l1111_111179

def tomato_problem (plant1 plant2 plant3 plant4 plant5 plant6 plant7 plant8 plant9 : ℕ) : Prop :=
  plant1 = 15 ∧
  plant2 = 2 * plant1 - 8 ∧
  plant3 = (plant1^2) / 3 ∧
  plant4 = (plant1 + plant2) / 2 ∧
  plant5 = 3 * Int.sqrt (plant1 + plant2) ∧
  plant6 = plant5 ∧
  plant7 = (3 * (plant1 + plant2 + plant3)) / 2 ∧
  plant8 = plant7 ∧
  plant9 = plant1 + plant7 + 6 →
  plant1 + plant2 + plant3 + plant4 + plant5 + plant6 + plant7 + plant8 + plant9 = 692

theorem tomato_theorem : ∃ plant1 plant2 plant3 plant4 plant5 plant6 plant7 plant8 plant9 : ℕ,
  tomato_problem plant1 plant2 plant3 plant4 plant5 plant6 plant7 plant8 plant9 :=
by
  sorry

end NUMINAMATH_CALUDE_tomato_theorem_l1111_111179


namespace NUMINAMATH_CALUDE_mMobileCheaperByEleven_l1111_111182

/-- Calculates the cost of a mobile plan given the base cost for two lines, 
    the cost per additional line, and the total number of lines. -/
def mobilePlanCost (baseCost : ℕ) (additionalLineCost : ℕ) (totalLines : ℕ) : ℕ :=
  baseCost + (max (totalLines - 2) 0) * additionalLineCost

/-- Proves that M-Mobile is $11 cheaper than T-Mobile for a family plan with 5 lines. -/
theorem mMobileCheaperByEleven : 
  mobilePlanCost 50 16 5 - mobilePlanCost 45 14 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_mMobileCheaperByEleven_l1111_111182


namespace NUMINAMATH_CALUDE_sum_of_digits_of_product_l1111_111132

/-- Represents a number formed by repeating a pattern a certain number of times -/
def repeatedPattern (pattern : ℕ) (repetitions : ℕ) : ℕ :=
  sorry

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem sum_of_digits_of_product : 
  let a := repeatedPattern 15 1004 * repeatedPattern 3 52008
  sumOfDigits a = 18072 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_product_l1111_111132


namespace NUMINAMATH_CALUDE_watermelon_ratio_l1111_111135

theorem watermelon_ratio (michael_weight john_weight : ℚ) : 
  michael_weight = 8 →
  john_weight = 12 →
  john_weight / (3 * michael_weight) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_ratio_l1111_111135


namespace NUMINAMATH_CALUDE_thursday_steps_l1111_111128

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The target average steps per day -/
def target_average : ℕ := 9000

/-- Steps walked on Sunday -/
def sunday_steps : ℕ := 9400

/-- Steps walked on Monday -/
def monday_steps : ℕ := 9100

/-- Steps walked on Tuesday -/
def tuesday_steps : ℕ := 8300

/-- Steps walked on Wednesday -/
def wednesday_steps : ℕ := 9200

/-- Average steps for Friday and Saturday -/
def friday_saturday_average : ℕ := 9050

/-- Theorem: Given the conditions, Toby must have walked 8900 steps on Thursday to meet his weekly goal -/
theorem thursday_steps : 
  (days_in_week * target_average) - 
  (sunday_steps + monday_steps + tuesday_steps + wednesday_steps + 2 * friday_saturday_average) = 8900 := by
  sorry

end NUMINAMATH_CALUDE_thursday_steps_l1111_111128


namespace NUMINAMATH_CALUDE_triangle_side_difference_l1111_111198

-- Define the triangle sides
def side1 : ℝ := 7
def side2 : ℝ := 10

-- Define the valid range for x
def valid_x (x : ℤ) : Prop :=
  x > 0 ∧ x + side1 > side2 ∧ x + side2 > side1 ∧ side1 + side2 > x

-- Theorem statement
theorem triangle_side_difference :
  (∃ (max min : ℤ), 
    (∀ x : ℤ, valid_x x → x ≤ max) ∧
    (∀ x : ℤ, valid_x x → x ≥ min) ∧
    (∀ x : ℤ, valid_x x → min ≤ x ∧ x ≤ max) ∧
    (max - min = 12)) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_difference_l1111_111198


namespace NUMINAMATH_CALUDE_medium_box_tape_proof_l1111_111124

/-- The amount of tape (in feet) needed to seal a large box -/
def large_box_tape : ℝ := 4

/-- The amount of tape (in feet) needed to seal a small box -/
def small_box_tape : ℝ := 1

/-- The amount of tape (in feet) needed for the address label on any box -/
def label_tape : ℝ := 1

/-- The number of large boxes packed -/
def num_large_boxes : ℕ := 2

/-- The number of medium boxes packed -/
def num_medium_boxes : ℕ := 8

/-- The number of small boxes packed -/
def num_small_boxes : ℕ := 5

/-- The total amount of tape (in feet) used -/
def total_tape : ℝ := 44

/-- The amount of tape (in feet) needed to seal a medium box -/
def medium_box_tape : ℝ := 2

theorem medium_box_tape_proof :
  medium_box_tape * num_medium_boxes + 
  large_box_tape * num_large_boxes + 
  small_box_tape * num_small_boxes + 
  label_tape * (num_large_boxes + num_medium_boxes + num_small_boxes) = 
  total_tape := by sorry

end NUMINAMATH_CALUDE_medium_box_tape_proof_l1111_111124


namespace NUMINAMATH_CALUDE_percentage_proof_l1111_111125

/-- Given a number N and a percentage P, proves that P is 50% when N is 456 and P% of N equals 40% of 120 plus 180. -/
theorem percentage_proof (N : ℝ) (P : ℝ) : 
  N = 456 →
  (P / 100) * N = (40 / 100) * 120 + 180 →
  P = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_proof_l1111_111125


namespace NUMINAMATH_CALUDE_astronaut_stay_duration_l1111_111102

theorem astronaut_stay_duration (days_per_year : ℕ) (seasons_per_year : ℕ) (seasons_stayed : ℕ) : 
  days_per_year = 250 → 
  seasons_per_year = 5 → 
  seasons_stayed = 3 → 
  (days_per_year / seasons_per_year) * seasons_stayed = 150 :=
by sorry

end NUMINAMATH_CALUDE_astronaut_stay_duration_l1111_111102


namespace NUMINAMATH_CALUDE_set_relationship_theorem_l1111_111196

def A : Set ℝ := {-1, 2}

def B (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 = 2}

def whale_swallowing (X Y : Set ℝ) : Prop := X ⊆ Y ∨ Y ⊆ X

def moth_eating (X Y : Set ℝ) : Prop := 
  (∃ x, x ∈ X ∩ Y) ∧ ¬(X ⊆ Y) ∧ ¬(Y ⊆ X)

theorem set_relationship_theorem : 
  {a : ℝ | a ≥ 0 ∧ (whale_swallowing A (B a) ∨ moth_eating A (B a))} = {0, 1/2, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_relationship_theorem_l1111_111196


namespace NUMINAMATH_CALUDE_charge_account_interest_l1111_111195

/-- Calculates the total amount owed after one year with simple interest -/
def total_amount_owed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proves that the total amount owed after one year is $38.15 -/
theorem charge_account_interest :
  let principal : ℝ := 35
  let rate : ℝ := 0.09
  let time : ℝ := 1
  total_amount_owed principal rate time = 38.15 := by
sorry

end NUMINAMATH_CALUDE_charge_account_interest_l1111_111195


namespace NUMINAMATH_CALUDE_square_count_3x3_and_5x5_l1111_111139

/-- Represents a square grid with uniform distance between consecutive dots -/
structure UniformSquareGrid (n : ℕ) :=
  (size : ℕ)
  (uniform_distance : Bool)

/-- Counts the number of squares with all 4 vertices on the dots in a grid -/
def count_squares (grid : UniformSquareGrid n) : ℕ :=
  sorry

theorem square_count_3x3_and_5x5 :
  ∀ (grid3 : UniformSquareGrid 3) (grid5 : UniformSquareGrid 5),
    grid3.size = 3 ∧ grid3.uniform_distance = true →
    grid5.size = 5 ∧ grid5.uniform_distance = true →
    count_squares grid3 = 4 ∧ count_squares grid5 = 50 :=
by sorry

end NUMINAMATH_CALUDE_square_count_3x3_and_5x5_l1111_111139


namespace NUMINAMATH_CALUDE_specific_theater_seats_l1111_111155

/-- A theater with an arithmetic progression of seats per row -/
structure Theater where
  first_row_seats : ℕ
  seat_increase : ℕ
  last_row_seats : ℕ

/-- Calculate the total number of seats in the theater -/
def total_seats (t : Theater) : ℕ :=
  let n := (t.last_row_seats - t.first_row_seats) / t.seat_increase + 1
  n * (t.first_row_seats + t.last_row_seats) / 2

/-- The theorem stating the total number of seats in the specific theater -/
theorem specific_theater_seats :
  let t : Theater := { first_row_seats := 14, seat_increase := 3, last_row_seats := 50 }
  total_seats t = 416 := by
  sorry


end NUMINAMATH_CALUDE_specific_theater_seats_l1111_111155


namespace NUMINAMATH_CALUDE_slope_angle_of_line_l1111_111187

/-- The slope angle of a line passing through points (0,√3) and (2,3√3) is π/3 -/
theorem slope_angle_of_line (A B : ℝ × ℝ) : 
  A = (0, Real.sqrt 3) → 
  B = (2, 3 * Real.sqrt 3) → 
  let slope := (B.2 - A.2) / (B.1 - A.1)
  Real.arctan slope = π / 3 := by sorry

end NUMINAMATH_CALUDE_slope_angle_of_line_l1111_111187


namespace NUMINAMATH_CALUDE_heartsuit_properties_l1111_111150

/-- The heartsuit operation on real numbers -/
def heartsuit (x y : ℝ) : ℝ := 2 * |x - y|

/-- Properties of the heartsuit operation -/
theorem heartsuit_properties :
  (∀ x y : ℝ, heartsuit x y = heartsuit y x) ∧ 
  (∀ x y : ℝ, 3 * (heartsuit x y) = heartsuit (3*x) (3*y)) ∧ 
  (∀ x : ℝ, heartsuit x x = 0) ∧ 
  (∀ x y : ℝ, x ≠ y → heartsuit x y > 0) ∧ 
  (∀ x : ℝ, x ≥ 0 → heartsuit x 0 = 2*x) ∧
  (∃ x : ℝ, heartsuit x 0 ≠ 2*x) :=
by sorry

end NUMINAMATH_CALUDE_heartsuit_properties_l1111_111150


namespace NUMINAMATH_CALUDE_circle_radius_from_area_circumference_ratio_l1111_111164

/-- Given a circle with area X and circumference Y, if X/Y = 10, then the radius is 20 -/
theorem circle_radius_from_area_circumference_ratio (X Y : ℝ) (h1 : X > 0) (h2 : Y > 0) (h3 : X / Y = 10) :
  ∃ r : ℝ, r > 0 ∧ X = π * r^2 ∧ Y = 2 * π * r ∧ r = 20 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_circumference_ratio_l1111_111164


namespace NUMINAMATH_CALUDE_max_value_of_f_l1111_111111

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi / 2 + 2 * x) - 5 * Real.sin x

theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1111_111111


namespace NUMINAMATH_CALUDE_rocket_max_height_l1111_111143

/-- The height function of the rocket -/
def h (t : ℝ) : ℝ := -12 * t^2 + 72 * t + 36

/-- Theorem stating that the maximum height of the rocket is 144 feet -/
theorem rocket_max_height : 
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 144 := by
  sorry

end NUMINAMATH_CALUDE_rocket_max_height_l1111_111143


namespace NUMINAMATH_CALUDE_odd_not_divides_power_plus_one_l1111_111129

theorem odd_not_divides_power_plus_one (n m : ℕ) (h_odd : Odd n) (h_gt_one : n > 1) :
  ¬(n ∣ (m^(n-1) + 1)) := by
  sorry

end NUMINAMATH_CALUDE_odd_not_divides_power_plus_one_l1111_111129


namespace NUMINAMATH_CALUDE_two_digit_number_interchange_property_l1111_111153

theorem two_digit_number_interchange_property (a b k : ℕ) (h1 : 10 * a + b = 2 * k * (a + b)) :
  10 * b + a - 3 * (a + b) = (9 - 4 * k) * (a + b) := by sorry

end NUMINAMATH_CALUDE_two_digit_number_interchange_property_l1111_111153


namespace NUMINAMATH_CALUDE_total_removed_volume_l1111_111103

/-- The edge length of the cube -/
def cube_edge : ℝ := 2

/-- The number of sides in the resulting polygon on each face after slicing -/
def hexadecagon_sides : ℕ := 16

/-- The volume of a single removed tetrahedron -/
noncomputable def tetrahedron_volume : ℝ := 
  let y := 2 * (Real.sqrt 2 - 1)
  let height := 3 - 2 * Real.sqrt 2
  let base_area := (1 / 2) * ((2 - Real.sqrt 2) ^ 2)
  (1 / 3) * base_area * height

/-- The number of corners in a cube -/
def cube_corners : ℕ := 8

/-- Theorem stating the total volume of removed tetrahedra -/
theorem total_removed_volume : 
  cube_corners * tetrahedron_volume = -64 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_total_removed_volume_l1111_111103


namespace NUMINAMATH_CALUDE_square_roots_problem_l1111_111151

theorem square_roots_problem (m : ℝ) : 
  (∃ (x : ℝ), x > 0 ∧ (m - 1)^2 = x ∧ (3*m - 5)^2 = x) → 
  (∃ (y : ℝ), y = 1/4 ∨ y = 1 ∧ (m - 1)^2 = y ∧ (3*m - 5)^2 = y) :=
by sorry

end NUMINAMATH_CALUDE_square_roots_problem_l1111_111151


namespace NUMINAMATH_CALUDE_average_weight_of_children_l1111_111147

theorem average_weight_of_children (num_boys num_girls : ℕ) (avg_weight_boys avg_weight_girls : ℚ) :
  num_boys = 8 →
  num_girls = 6 →
  avg_weight_boys = 160 →
  avg_weight_girls = 130 →
  (num_boys * avg_weight_boys + num_girls * avg_weight_girls) / (num_boys + num_girls) = 147 :=
by sorry

end NUMINAMATH_CALUDE_average_weight_of_children_l1111_111147


namespace NUMINAMATH_CALUDE_negation_of_product_nonzero_l1111_111168

theorem negation_of_product_nonzero (a b : ℝ) :
  ¬(a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0) ↔ (a * b = 0 → a = 0 ∨ b = 0) := by
sorry

end NUMINAMATH_CALUDE_negation_of_product_nonzero_l1111_111168


namespace NUMINAMATH_CALUDE_quadratic_equation_root_zero_l1111_111109

theorem quadratic_equation_root_zero (k : ℝ) : 
  (∀ x : ℝ, (k - 1) * x^2 + 3 * x + k^2 - 1 = 0 → x = 0 ∨ x ≠ 0) →
  ((k - 1) * 0^2 + 3 * 0 + k^2 - 1 = 0) →
  k = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_zero_l1111_111109


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1111_111145

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (3 * x - 6) = 10 → x = 106 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1111_111145


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_equation_l1111_111190

theorem consecutive_odd_numbers_equation (n : ℕ) : 
  let first := 7
  let second := first + 2
  let third := second + 2
  8 * first = 3 * third + 2 * second + 5 :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_equation_l1111_111190


namespace NUMINAMATH_CALUDE_power_simplification_l1111_111140

theorem power_simplification :
  (10^0.6) * (10^0.4) * (10^0.4) * (10^0.1) * (10^0.5) / (10^0.3) = 10^1.7 := by
  sorry

end NUMINAMATH_CALUDE_power_simplification_l1111_111140


namespace NUMINAMATH_CALUDE_pentagon_sides_solutions_l1111_111178

/-- A pentagon with side lengths satisfying the given conditions -/
structure PentagonSides where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  h_one_side : e = 30
  h_arithmetic : b = a + 2 ∧ c = b + 2 ∧ d = c + 2
  h_smallest : a ≤ 7
  h_sum : a + b + c + d + e > e

/-- The theorem stating that only three specific pentagons satisfy the conditions -/
theorem pentagon_sides_solutions :
  { sides : PentagonSides | 
    (sides.a = 5 ∧ sides.b = 7 ∧ sides.c = 9 ∧ sides.d = 11 ∧ sides.e = 30) ∨
    (sides.a = 6 ∧ sides.b = 8 ∧ sides.c = 10 ∧ sides.d = 12 ∧ sides.e = 30) ∨
    (sides.a = 7 ∧ sides.b = 9 ∧ sides.c = 11 ∧ sides.d = 13 ∧ sides.e = 30) } =
  { sides : PentagonSides | True } :=
sorry

end NUMINAMATH_CALUDE_pentagon_sides_solutions_l1111_111178


namespace NUMINAMATH_CALUDE_x_over_y_equals_one_l1111_111112

-- Define a function that represents the nested absolute value expression
def nestedAbs (x y : ℝ) : ℕ → ℝ
  | 0 => x
  | n + 1 => |nestedAbs y x n - x|

-- State the theorem
theorem x_over_y_equals_one
  (x y : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (h : nestedAbs x y 2019 = nestedAbs y x 2019) :
  x / y = 1 :=
sorry

end NUMINAMATH_CALUDE_x_over_y_equals_one_l1111_111112


namespace NUMINAMATH_CALUDE_sum_always_four_digits_l1111_111194

-- Define nonzero digits
def NonzeroDigit := { n : ℕ | 1 ≤ n ∧ n ≤ 9 }

-- Define the sum function
def sum_numbers (C D : NonzeroDigit) : ℕ :=
  3654 + (100 * C.val + 41) + (10 * D.val + 2) + 111

-- Theorem statement
theorem sum_always_four_digits (C D : NonzeroDigit) :
  ∃ n : ℕ, 1000 ≤ sum_numbers C D ∧ sum_numbers C D < 10000 := by
  sorry

end NUMINAMATH_CALUDE_sum_always_four_digits_l1111_111194


namespace NUMINAMATH_CALUDE_cos_2alpha_value_l1111_111127

theorem cos_2alpha_value (α : Real) (h1 : 0 < α ∧ α < π/2) (h2 : Real.sin (α - π/4) = 1/4) : 
  Real.cos (2 * α) = -Real.sqrt 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l1111_111127


namespace NUMINAMATH_CALUDE_solutions_of_z_sixth_power_eq_neg_64_l1111_111186

-- Define the complex number z
variable (z : ℂ)

-- Define the equation
def equation (z : ℂ) : Prop := z^6 = -64

-- State the theorem
theorem solutions_of_z_sixth_power_eq_neg_64 :
  (∀ z : ℂ, equation z ↔ z = 2*I ∨ z = -2*I) :=
sorry

end NUMINAMATH_CALUDE_solutions_of_z_sixth_power_eq_neg_64_l1111_111186


namespace NUMINAMATH_CALUDE_C₁_C₂_tangent_l1111_111170

-- Define the curves in polar coordinates
def C₁ (ρ θ : ℝ) : Prop := ρ * Real.sin θ - 2 = 0
def C₂ (ρ θ : ℝ) : Prop := ρ - 4 * Real.cos θ = 0

-- Define the rectangular form of C₁
def C₁_rect (x y : ℝ) : Prop := y = 2

-- Define the rectangular form of C₂
def C₂_rect (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the center and radius of C₂
def C₂_center : ℝ × ℝ := (2, 0)
def C₂_radius : ℝ := 2

-- Theorem stating that C₁ and C₂ are tangent
theorem C₁_C₂_tangent : 
  ∃ (x y : ℝ), C₁_rect x y ∧ C₂_rect x y ∧ 
  (∀ (x' y' : ℝ), C₁_rect x' y' ∧ C₂_rect x' y' → (x', y') = (x, y)) :=
sorry

end NUMINAMATH_CALUDE_C₁_C₂_tangent_l1111_111170


namespace NUMINAMATH_CALUDE_base_7_23456_equals_6068_l1111_111149

def base_7_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_7_23456_equals_6068 :
  base_7_to_10 [6, 5, 4, 3, 2] = 6068 := by
  sorry

end NUMINAMATH_CALUDE_base_7_23456_equals_6068_l1111_111149


namespace NUMINAMATH_CALUDE_least_satisfying_number_l1111_111137

def is_multiple_of_36 (n : ℕ) : Prop := ∃ k : ℕ, n = 36 * k

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  is_multiple_of_36 n ∧ is_multiple_of_36 (digit_product n)

theorem least_satisfying_number :
  satisfies_condition 1296 ∧ 
  ∀ m : ℕ, m > 0 ∧ m < 1296 → ¬(satisfies_condition m) :=
by sorry

end NUMINAMATH_CALUDE_least_satisfying_number_l1111_111137


namespace NUMINAMATH_CALUDE_tree_height_problem_l1111_111160

theorem tree_height_problem (h₁ h₂ : ℝ) : 
  h₂ = h₁ + 24 →  -- The taller tree is 24 feet higher
  h₁ / h₂ = 5 / 7 →  -- The ratio of heights is 5:7
  h₂ = 84 := by
sorry

end NUMINAMATH_CALUDE_tree_height_problem_l1111_111160


namespace NUMINAMATH_CALUDE_distribution_within_one_std_dev_l1111_111100

-- Define a symmetric distribution type
structure SymmetricDistribution where
  -- The cumulative distribution function (CDF)
  cdf : ℝ → ℝ
  -- The mean of the distribution
  mean : ℝ
  -- The standard deviation of the distribution
  std_dev : ℝ
  -- Symmetry property
  symmetry : ∀ x, cdf (mean - x) + cdf (mean + x) = 1
  -- Property that 84% of the distribution is less than mean + std_dev
  eighty_four_percent : cdf (mean + std_dev) = 0.84

-- Theorem statement
theorem distribution_within_one_std_dev 
  (d : SymmetricDistribution) : 
  d.cdf (d.mean + d.std_dev) - d.cdf (d.mean - d.std_dev) = 0.68 := by
  sorry

end NUMINAMATH_CALUDE_distribution_within_one_std_dev_l1111_111100


namespace NUMINAMATH_CALUDE_antonio_age_is_51_months_l1111_111107

-- Define Isabella's age in months after 18 months
def isabella_age_after_18_months : ℕ := 10 * 12

-- Define the current time difference in months
def time_difference : ℕ := 18

-- Define Isabella's current age in months
def isabella_current_age : ℕ := isabella_age_after_18_months - time_difference

-- Define the relationship between Isabella and Antonio's ages
def antonio_age : ℕ := isabella_current_age / 2

-- Theorem to prove
theorem antonio_age_is_51_months : antonio_age = 51 := by
  sorry


end NUMINAMATH_CALUDE_antonio_age_is_51_months_l1111_111107


namespace NUMINAMATH_CALUDE_parabola_axis_symmetry_l1111_111185

/-- 
Given a parabola defined by y = a * x^2 with axis of symmetry y = -2,
prove that a = 1/8.
-/
theorem parabola_axis_symmetry (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2) → 
  (∀ x : ℝ, -2 = a * x^2) → 
  a = 1/8 := by sorry

end NUMINAMATH_CALUDE_parabola_axis_symmetry_l1111_111185


namespace NUMINAMATH_CALUDE_non_congruent_squares_count_l1111_111144

/-- A lattice point on a 2D grid --/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A square on a lattice grid --/
structure LatticeSquare where
  vertices : Finset LatticePoint
  size : ℕ

/-- The size of the grid --/
def gridSize : ℕ := 6

/-- Function to count standard squares of a given size --/
def countStandardSquares (k : ℕ) : ℕ :=
  (gridSize - k + 1) ^ 2

/-- Function to count 45-degree tilted squares with diagonal of a given size --/
def countTiltedSquares (k : ℕ) : ℕ :=
  (gridSize - k) ^ 2

/-- Function to count 45-degree tilted squares with diagonal of a rectangle --/
def countRectangleDiagonalSquares (w h : ℕ) : ℕ :=
  2 * (gridSize - w) * (gridSize - h)

/-- The total number of non-congruent squares on the grid --/
def totalNonCongruentSquares : ℕ :=
  (countStandardSquares 1) + (countStandardSquares 2) + (countStandardSquares 3) +
  (countStandardSquares 4) + (countStandardSquares 5) +
  (countTiltedSquares 1) + (countTiltedSquares 2) +
  (countRectangleDiagonalSquares 1 2) + (countRectangleDiagonalSquares 1 3)

/-- Theorem: The number of non-congruent squares on a 6x6 grid is 201 --/
theorem non_congruent_squares_count :
  totalNonCongruentSquares = 201 := by
  sorry

end NUMINAMATH_CALUDE_non_congruent_squares_count_l1111_111144


namespace NUMINAMATH_CALUDE_fifth_pythagorean_triple_l1111_111191

/-- Generates the nth Pythagorean triple based on the given pattern -/
def pythagoreanTriple (n : ℕ) : ℕ × ℕ × ℕ :=
  let a := 2 * n + 1
  let b := 2 * n * (n + 1)
  let c := 2 * n * (n + 1) + 1
  (a, b, c)

/-- Checks if a triple of natural numbers forms a Pythagorean triple -/
def isPythagoreanTriple (triple : ℕ × ℕ × ℕ) : Prop :=
  let (a, b, c) := triple
  a * a + b * b = c * c

theorem fifth_pythagorean_triple :
  let triple := pythagoreanTriple 5
  triple = (11, 60, 61) ∧ isPythagoreanTriple triple :=
by sorry

end NUMINAMATH_CALUDE_fifth_pythagorean_triple_l1111_111191


namespace NUMINAMATH_CALUDE_no_real_solutions_l1111_111104

theorem no_real_solutions : ¬∃ (x y z : ℝ), (x + y = 4) ∧ (x * y - 9 * z^2 = -5) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1111_111104


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1111_111173

theorem quadratic_equation_solution (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) :
  (x^2 + 10*x = 45 ∧ x = Real.sqrt a - b) → a + b = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1111_111173


namespace NUMINAMATH_CALUDE_m_greater_than_n_l1111_111138

theorem m_greater_than_n (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (ha1 : a < 1) (hb1 : b < 1) : 
  a * b > a + b - 1 := by
  sorry

end NUMINAMATH_CALUDE_m_greater_than_n_l1111_111138


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1111_111174

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 ∧ x ≠ 3 →
  (1 - 2 / (x - 1)) / ((x^2 - 6*x + 9) / (x^2 - 1)) = (x + 1) / (x - 3) ∧
  (2 + 1) / (2 - 3) = -3 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1111_111174


namespace NUMINAMATH_CALUDE_line_through_point_at_distance_l1111_111131

/-- A line passing through a point (x₀, y₀) and at a distance d from the origin -/
structure DistanceLine where
  x₀ : ℝ
  y₀ : ℝ
  d : ℝ

/-- Check if a line equation ax + by + c = 0 passes through a point (x₀, y₀) -/
def passesThrough (a b c x₀ y₀ : ℝ) : Prop :=
  a * x₀ + b * y₀ + c = 0

/-- Check if a line equation ax + by + c = 0 is at a distance d from the origin -/
def distanceFromOrigin (a b c d : ℝ) : Prop :=
  |c| / Real.sqrt (a^2 + b^2) = d

theorem line_through_point_at_distance (l : DistanceLine) :
  (passesThrough 1 0 (-3) l.x₀ l.y₀ ∧ distanceFromOrigin 1 0 (-3) l.d) ∨
  (passesThrough 8 (-15) 51 l.x₀ l.y₀ ∧ distanceFromOrigin 8 (-15) 51 l.d) :=
by sorry

#check line_through_point_at_distance

end NUMINAMATH_CALUDE_line_through_point_at_distance_l1111_111131


namespace NUMINAMATH_CALUDE_complement_of_A_l1111_111189

-- Define the set A
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}

-- State the theorem
theorem complement_of_A : 
  (Set.univ \ A : Set ℝ) = Set.Icc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1111_111189


namespace NUMINAMATH_CALUDE_candy_solution_l1111_111161

/-- Represents the candy distribution problem --/
def candy_problem (billy_initial caleb_initial andy_initial : ℕ)
                  (new_candies billy_new caleb_new : ℕ) : Prop :=
  let billy_total := billy_initial + billy_new
  let caleb_total := caleb_initial + caleb_new
  let andy_new := new_candies - billy_new - caleb_new
  let andy_total := andy_initial + andy_new
  andy_total - caleb_total = 4

/-- Theorem stating the solution to the candy problem --/
theorem candy_solution :
  candy_problem 6 11 9 36 8 11 := by
  sorry

end NUMINAMATH_CALUDE_candy_solution_l1111_111161


namespace NUMINAMATH_CALUDE_cricket_bat_cost_price_l1111_111193

theorem cricket_bat_cost_price (profit_a_to_b : ℝ) (profit_b_to_c : ℝ) (price_c : ℝ) :
  profit_a_to_b = 0.20 →
  profit_b_to_c = 0.25 →
  price_c = 228 →
  ∃ (cost_price_a : ℝ),
    cost_price_a * (1 + profit_a_to_b) * (1 + profit_b_to_c) = price_c ∧
    cost_price_a = 152 :=
by sorry

end NUMINAMATH_CALUDE_cricket_bat_cost_price_l1111_111193


namespace NUMINAMATH_CALUDE_equation_solution_l1111_111106

theorem equation_solution : ∃! y : ℚ, 2 * (y - 3) - 6 * (2 * y - 1) = -3 * (2 - 5 * y) ∧ y = 6 / 25 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1111_111106


namespace NUMINAMATH_CALUDE_expression_evaluation_l1111_111130

theorem expression_evaluation : ((-3)^2)^4 * (-3)^8 * 2 = 86093442 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1111_111130


namespace NUMINAMATH_CALUDE_probability_ace_second_draw_l1111_111180

/-- The probability of drawing an Ace in the second draw without replacement from a deck of 52 cards, given that an Ace was drawn in the first draw. -/
theorem probability_ace_second_draw (initial_deck_size : ℕ) (initial_aces : ℕ) 
  (h1 : initial_deck_size = 52)
  (h2 : initial_aces = 4)
  (h3 : initial_aces > 0) :
  (initial_aces - 1 : ℚ) / (initial_deck_size - 1 : ℚ) = 1 / 17 := by
  sorry

end NUMINAMATH_CALUDE_probability_ace_second_draw_l1111_111180


namespace NUMINAMATH_CALUDE_gcd_problem_l1111_111183

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 97 * (2 * k)) :
  Int.gcd (3 * b^2 + 41 * b + 74) (b + 19) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1111_111183


namespace NUMINAMATH_CALUDE_digit_puzzle_solution_l1111_111126

/-- Represents a digit in base 10 -/
def Digit := Fin 10

/-- Checks if all elements in a list are distinct -/
def all_distinct (l : List Digit) : Prop :=
  ∀ i j, i ≠ j → l.get i ≠ l.get j

/-- Converts a pair of digits to a two-digit number -/
def to_number (tens digit : Digit) : Nat :=
  10 * tens.val + digit.val

/-- The main theorem -/
theorem digit_puzzle_solution (Y E M T : Digit) 
  (h_distinct : all_distinct [Y, E, M, T])
  (h_equation : to_number Y E * to_number M E = to_number T T * 101) :
  E.val + M.val + T.val + Y.val = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_puzzle_solution_l1111_111126


namespace NUMINAMATH_CALUDE_calendar_sum_l1111_111113

/-- Given three consecutive numbers in a vertical column of a calendar where the top number is n,
    the sum of these three numbers is equal to 3n + 21. -/
theorem calendar_sum (n : ℕ) : n + (n + 7) + (n + 14) = 3 * n + 21 := by
  sorry

end NUMINAMATH_CALUDE_calendar_sum_l1111_111113


namespace NUMINAMATH_CALUDE_three_number_sum_l1111_111158

theorem three_number_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  (a + b + c) / 3 = a + 5 → 
  (a + b + c) / 3 = c - 20 → 
  b = 10 → 
  a + b + c = -15 := by
  sorry

end NUMINAMATH_CALUDE_three_number_sum_l1111_111158


namespace NUMINAMATH_CALUDE_jerry_weller_votes_l1111_111176

theorem jerry_weller_votes 
  (total_votes : ℕ) 
  (vote_difference : ℕ) 
  (h1 : total_votes = 196554)
  (h2 : vote_difference = 20196) :
  ∃ (jerry_votes john_votes : ℕ),
    jerry_votes = 108375 ∧ 
    john_votes + vote_difference = jerry_votes ∧
    jerry_votes + john_votes = total_votes :=
by
  sorry

end NUMINAMATH_CALUDE_jerry_weller_votes_l1111_111176


namespace NUMINAMATH_CALUDE_percentage_increase_l1111_111141

theorem percentage_increase (initial final : ℝ) (h : initial > 0) :
  let increase := (final - initial) / initial * 100
  initial = 150 ∧ final = 210 → increase = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l1111_111141


namespace NUMINAMATH_CALUDE_center_cell_value_l1111_111171

theorem center_cell_value (a b c d e f g h i : ℝ) : 
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) ∧ (e > 0) ∧ (f > 0) ∧ (g > 0) ∧ (h > 0) ∧ (i > 0) →
  (a * b * c = 1) ∧ (d * e * f = 1) ∧ (g * h * i = 1) ∧
  (a * d * g = 1) ∧ (b * e * h = 1) ∧ (c * f * i = 1) →
  (a * b * d * e = 2) ∧ (b * c * e * f = 2) ∧ (d * e * g * h = 2) ∧ (e * f * h * i = 2) →
  e = 1 :=
by sorry

end NUMINAMATH_CALUDE_center_cell_value_l1111_111171


namespace NUMINAMATH_CALUDE_simon_initial_stamps_l1111_111114

/-- The number of stamps Simon has after receiving stamps from friends -/
def total_stamps : ℕ := 61

/-- The number of stamps Simon received from friends -/
def received_stamps : ℕ := 27

/-- The number of stamps Simon initially had -/
def initial_stamps : ℕ := total_stamps - received_stamps

theorem simon_initial_stamps :
  initial_stamps = 34 := by sorry

end NUMINAMATH_CALUDE_simon_initial_stamps_l1111_111114


namespace NUMINAMATH_CALUDE_median_in_middle_interval_l1111_111105

/-- Represents the intervals of scores -/
inductive ScoreInterval
| I60to64
| I65to69
| I70to74
| I75to79
| I80to84

/-- The total number of students -/
def totalStudents : ℕ := 100

/-- The number of intervals -/
def numIntervals : ℕ := 5

/-- The number of students in each interval -/
def studentsPerInterval : ℕ := totalStudents / numIntervals

/-- The position of the median in the ordered list of scores -/
def medianPosition : ℕ := (totalStudents + 1) / 2

/-- Theorem stating that the median score falls in the middle interval -/
theorem median_in_middle_interval :
  medianPosition > 2 * studentsPerInterval ∧
  medianPosition ≤ 3 * studentsPerInterval :=
sorry

end NUMINAMATH_CALUDE_median_in_middle_interval_l1111_111105


namespace NUMINAMATH_CALUDE_fifteen_cells_covered_by_two_l1111_111119

/-- Represents a square on a graph paper --/
structure Square :=
  (side : ℕ)

/-- Represents the configuration of squares on the graph paper --/
structure SquareConfiguration :=
  (squares : List Square)
  (total_area : ℕ)
  (unique_area : ℕ)
  (triple_overlap : ℕ)

/-- Calculates the number of cells covered by exactly two squares --/
def cells_covered_by_two (config : SquareConfiguration) : ℕ :=
  config.total_area - config.unique_area - 2 * config.triple_overlap

/-- Theorem stating that for the given configuration, 15 cells are covered by exactly two squares --/
theorem fifteen_cells_covered_by_two (config : SquareConfiguration) 
  (h1 : config.squares.length = 3)
  (h2 : ∀ s ∈ config.squares, s.side = 5)
  (h3 : config.total_area = 75)
  (h4 : config.unique_area = 56)
  (h5 : config.triple_overlap = 2) :
  cells_covered_by_two config = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_cells_covered_by_two_l1111_111119


namespace NUMINAMATH_CALUDE_terminal_point_coordinates_l1111_111177

/-- Given sin α = 3/5 and cos α = -4/5, the coordinates of the point on the terminal side of angle α are (-4, 3). -/
theorem terminal_point_coordinates (α : Real) 
  (h1 : Real.sin α = 3/5) 
  (h2 : Real.cos α = -4/5) : 
  ∃ (x y : Real), x = -4 ∧ y = 3 ∧ Real.sin α = y / Real.sqrt (x^2 + y^2) ∧ Real.cos α = x / Real.sqrt (x^2 + y^2) := by
  sorry

end NUMINAMATH_CALUDE_terminal_point_coordinates_l1111_111177


namespace NUMINAMATH_CALUDE_domain_of_f_x_squared_l1111_111159

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the domain of f(x+1)
def domain_f_x_plus_1 (f : ℝ → ℝ) : Set ℝ := Set.Icc (-2) 3

-- Define the property that f(x+1) has domain [-2, 3]
def f_x_plus_1_domain (f : ℝ → ℝ) : Prop :=
  ∀ x, x ∈ domain_f_x_plus_1 f ↔ f (x + 1) ≠ 0

-- Theorem statement
theorem domain_of_f_x_squared (f : ℝ → ℝ) 
  (h : f_x_plus_1_domain f) : 
  {x : ℝ | f (x^2) ≠ 0} = Set.Icc (-2) 2 := by sorry

end NUMINAMATH_CALUDE_domain_of_f_x_squared_l1111_111159


namespace NUMINAMATH_CALUDE_max_graduates_few_calls_l1111_111156

/-- The number of graduates -/
def total_graduates : ℕ := 100

/-- The number of universities -/
def num_universities : ℕ := 5

/-- The number of graduates each university attempts to contact -/
def contacts_per_university : ℕ := 50

/-- The total number of contact attempts made by all universities -/
def total_contacts : ℕ := num_universities * contacts_per_university

/-- The maximum number of graduates who received at most 2 calls -/
def max_graduates_with_few_calls : ℕ := 83

theorem max_graduates_few_calls :
  ∀ n : ℕ,
  n ≤ total_graduates →
  2 * n + 5 * (total_graduates - n) ≥ total_contacts →
  n ≤ max_graduates_with_few_calls :=
by sorry

end NUMINAMATH_CALUDE_max_graduates_few_calls_l1111_111156


namespace NUMINAMATH_CALUDE_average_increase_is_4_l1111_111167

/-- Represents the cricketer's score data -/
structure CricketerScore where
  runs_19th_inning : ℕ
  average_after_19 : ℚ

/-- Calculates the increase in average score -/
def average_increase (score : CricketerScore) : ℚ :=
  let total_runs := score.average_after_19 * 19
  let runs_before_19th := total_runs - score.runs_19th_inning
  let average_before_19th := runs_before_19th / 18
  score.average_after_19 - average_before_19th

/-- Theorem stating the increase in average score -/
theorem average_increase_is_4 (score : CricketerScore) 
  (h1 : score.runs_19th_inning = 96)
  (h2 : score.average_after_19 = 24) :
  average_increase score = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_increase_is_4_l1111_111167


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1111_111116

theorem unique_solution_condition (b : ℝ) : 
  (∃! x : ℝ, x^3 - b*x^2 - 4*b*x + b^2 - 4 = 0) ↔ b < 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1111_111116


namespace NUMINAMATH_CALUDE_solution_set_equation_l1111_111115

theorem solution_set_equation : 
  ∀ x : ℝ, ((x - 1) / x)^2 - (7/2) * ((x - 1) / x) + 3 = 0 ↔ x = -1 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equation_l1111_111115


namespace NUMINAMATH_CALUDE_perpendicular_diagonals_imply_square_rectangle_is_not_square_l1111_111120

-- Define a quadrilateral
structure Quadrilateral :=
  (has_right_angles : Bool)
  (opposite_sides_parallel_equal : Bool)
  (diagonals_bisect : Bool)
  (diagonals_perpendicular : Bool)

-- Define a rectangle
def Rectangle : Quadrilateral :=
  { has_right_angles := true,
    opposite_sides_parallel_equal := true,
    diagonals_bisect := true,
    diagonals_perpendicular := false }

-- Define a square
def Square : Quadrilateral :=
  { has_right_angles := true,
    opposite_sides_parallel_equal := true,
    diagonals_bisect := true,
    diagonals_perpendicular := true }

-- Theorem: A quadrilateral with right angles, opposite sides parallel and equal,
-- and perpendicular diagonals that bisect each other is a square
theorem perpendicular_diagonals_imply_square (q : Quadrilateral) :
  q.has_right_angles = true →
  q.opposite_sides_parallel_equal = true →
  q.diagonals_bisect = true →
  q.diagonals_perpendicular = true →
  q = Square := by
  sorry

-- Theorem: A rectangle is not a square
theorem rectangle_is_not_square : Rectangle ≠ Square := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_diagonals_imply_square_rectangle_is_not_square_l1111_111120


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l1111_111162

/-- Given a quadratic inequality with solution set (x₁, x₂), prove certain properties of the roots -/
theorem quadratic_inequality_properties (a : ℝ) (x₁ x₂ : ℝ) 
  (h_sol : ∀ x, a * (x - 1) * (x + 3) + 2 > 0 ↔ x ∈ Set.Ioo x₁ x₂) 
  (h_order : x₁ < x₂) :
  x₁ + x₂ + 2 = 0 ∧ |x₁ - x₂| > 4 ∧ x₁ * x₂ + 3 < 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l1111_111162


namespace NUMINAMATH_CALUDE_minimum_race_distance_l1111_111142

/-- The minimum distance a runner must travel in a race with given constraints -/
theorem minimum_race_distance (wall_length : ℝ) (distance_A : ℝ) (distance_B : ℝ) :
  wall_length = 1500 →
  distance_A = 400 →
  distance_B = 600 →
  let min_distance := Real.sqrt (wall_length ^ 2 + (distance_A + distance_B) ^ 2)
  ⌊min_distance + 0.5⌋ = 1803 := by
  sorry

end NUMINAMATH_CALUDE_minimum_race_distance_l1111_111142


namespace NUMINAMATH_CALUDE_tanning_salon_pricing_l1111_111181

theorem tanning_salon_pricing (first_visit_charge : ℕ) (total_customers : ℕ) (second_visits : ℕ) (third_visits : ℕ) (total_revenue : ℕ) :
  first_visit_charge = 10 →
  total_customers = 100 →
  second_visits = 30 →
  third_visits = 10 →
  total_revenue = 1240 →
  ∃ (subsequent_visit_charge : ℕ),
    subsequent_visit_charge = 6 ∧
    total_revenue = first_visit_charge * total_customers + subsequent_visit_charge * (second_visits + third_visits) :=
by sorry

end NUMINAMATH_CALUDE_tanning_salon_pricing_l1111_111181
