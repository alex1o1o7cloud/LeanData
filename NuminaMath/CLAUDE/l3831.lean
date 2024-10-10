import Mathlib

namespace grid_paths_6_5_l3831_383126

/-- The number of distinct paths on a grid from (0,0) to (m,n) using only right and up moves -/
def gridPaths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) n

theorem grid_paths_6_5 : gridPaths 6 5 = 462 := by
  sorry

end grid_paths_6_5_l3831_383126


namespace arctan_sum_equals_pi_over_four_l3831_383197

theorem arctan_sum_equals_pi_over_four (n : ℕ) :
  (n > 0) →
  (Real.arctan (1/2) + Real.arctan (1/4) + Real.arctan (1/5) + Real.arctan (1/n : ℝ) = π/4) →
  n = 27 := by
  sorry

end arctan_sum_equals_pi_over_four_l3831_383197


namespace mischievous_polynomial_at_two_l3831_383145

/-- A quadratic polynomial of the form x^2 - px + q -/
structure MischievousPolynomial where
  p : ℝ
  q : ℝ

/-- Predicate for a mischievous polynomial -/
def isMischievous (poly : MischievousPolynomial) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    (∀ (x : ℝ), (x^2 - poly.p * (x^2 - poly.p * x + poly.q) + poly.q = 0) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃))

/-- The product of roots of a quadratic polynomial -/
def rootProduct (poly : MischievousPolynomial) : ℝ := poly.q

/-- The value of the polynomial at x = 2 -/
def evalAtTwo (poly : MischievousPolynomial) : ℝ := 4 - 2 * poly.p + poly.q

/-- The main theorem -/
theorem mischievous_polynomial_at_two :
  ∀ (poly : MischievousPolynomial),
    isMischievous poly →
    (∀ (other : MischievousPolynomial), isMischievous other → rootProduct poly ≤ rootProduct other) →
    evalAtTwo poly = -1 := by
  sorry

end mischievous_polynomial_at_two_l3831_383145


namespace arithmetic_progression_proof_l3831_383189

theorem arithmetic_progression_proof (a₁ d : ℕ) : 
  (a₁ * (a₁ + d) * (a₁ + 2*d) = 6) ∧ 
  (a₁ * (a₁ + d) * (a₁ + 2*d) * (a₁ + 3*d) = 24) → 
  (a₁ = 1 ∧ d = 1) := by
sorry

end arithmetic_progression_proof_l3831_383189


namespace last_recess_duration_l3831_383146

/- Define the durations of known breaks -/
def first_recess : ℕ := 15
def second_recess : ℕ := 15
def lunch : ℕ := 30

/- Define the total time spent outside of class -/
def total_outside_time : ℕ := 80

/- Define the duration of the last recess break -/
def last_recess : ℕ := total_outside_time - (first_recess + second_recess + lunch)

/- Theorem to prove -/
theorem last_recess_duration :
  last_recess = 20 :=
by sorry

end last_recess_duration_l3831_383146


namespace sufficient_not_necessary_condition_l3831_383118

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > b^2) ∧ 
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) := by
  sorry

end sufficient_not_necessary_condition_l3831_383118


namespace parabola_point_coordinates_l3831_383115

/-- The parabola y² = 8x with a point P at distance 9 from its focus -/
theorem parabola_point_coordinates :
  ∀ (x y : ℝ),
  y^2 = 8*x →                        -- P lies on the parabola
  (x - 2)^2 + y^2 = 9^2 →            -- Distance from P to focus (2, 0) is 9
  x = 7 ∧ y^2 = 56 := by             -- Coordinates of P are (7, ±2√14)
sorry

end parabola_point_coordinates_l3831_383115


namespace lillys_fish_l3831_383172

theorem lillys_fish (total : ℕ) (rosys_fish : ℕ) (lillys_fish : ℕ) : 
  total = 21 → rosys_fish = 11 → total = rosys_fish + lillys_fish → lillys_fish = 10 := by
  sorry

end lillys_fish_l3831_383172


namespace total_paintable_area_is_1129_l3831_383188

def bedroom_area (length width height : ℕ) : ℕ :=
  2 * (length * height + width * height)

def paintable_area (total_area unpaintable_area : ℕ) : ℕ :=
  total_area - unpaintable_area

theorem total_paintable_area_is_1129 :
  let bedroom1_total := bedroom_area 14 12 9
  let bedroom2_total := bedroom_area 12 11 9
  let bedroom3_total := bedroom_area 13 12 9
  let bedroom1_paintable := paintable_area bedroom1_total 70
  let bedroom2_paintable := paintable_area bedroom2_total 65
  let bedroom3_paintable := paintable_area bedroom3_total 68
  bedroom1_paintable + bedroom2_paintable + bedroom3_paintable = 1129 := by
  sorry

end total_paintable_area_is_1129_l3831_383188


namespace m_plus_n_squared_l3831_383184

theorem m_plus_n_squared (m n : ℤ) (h1 : |m| = 4) (h2 : |n| = 3) (h3 : m - n < 0) :
  (m + n)^2 = 1 ∨ (m + n)^2 = 49 := by
sorry

end m_plus_n_squared_l3831_383184


namespace algebraic_expression_value_l3831_383133

theorem algebraic_expression_value (a b : ℝ) 
  (ha : a = 1 + Real.sqrt 2) 
  (hb : b = 1 - Real.sqrt 2) : 
  a^2 - a*b + b^2 = 7 := by
  sorry

end algebraic_expression_value_l3831_383133


namespace min_students_with_all_characteristics_l3831_383132

theorem min_students_with_all_characteristics
  (total : ℕ) (blue_eyes : ℕ) (lunch_box : ℕ) (glasses : ℕ)
  (h_total : total = 35)
  (h_blue_eyes : blue_eyes = 15)
  (h_lunch_box : lunch_box = 25)
  (h_glasses : glasses = 10) :
  ∃ (n : ℕ), n ≥ 1 ∧ n ≤ min blue_eyes (min lunch_box glasses) ∧
    n ≥ blue_eyes + lunch_box + glasses - 2 * total :=
by sorry

end min_students_with_all_characteristics_l3831_383132


namespace exponent_subtraction_l3831_383176

theorem exponent_subtraction (m : ℝ) : m^2020 / m^2019 = m :=
by sorry

end exponent_subtraction_l3831_383176


namespace polynomial_division_theorem_l3831_383110

theorem polynomial_division_theorem :
  let dividend : Polynomial ℤ := 4 * X^5 - 5 * X^4 + 3 * X^3 - 7 * X^2 + 6 * X - 1
  let divisor : Polynomial ℤ := X^2 + 2 * X + 3
  let quotient : Polynomial ℤ := 4 * X^3 - 13 * X^2 + 35 * X - 104
  let remainder : Polynomial ℤ := 87
  dividend = divisor * quotient + remainder := by sorry

end polynomial_division_theorem_l3831_383110


namespace three_equal_perimeter_triangles_l3831_383163

def stick_lengths : List ℕ := [2, 3, 3, 3, 4, 5, 5, 5, 6, 6, 9]

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def forms_triangle (lengths : List ℕ) : Prop :=
  ∃ (a b c : ℕ), a ∈ lengths ∧ b ∈ lengths ∧ c ∈ lengths ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  is_triangle a b c ∧
  a + b + c = 14

theorem three_equal_perimeter_triangles :
  ∃ (t1 t2 t3 : List ℕ),
    t1 ⊆ stick_lengths ∧
    t2 ⊆ stick_lengths ∧
    t3 ⊆ stick_lengths ∧
    t1 ∩ t2 = ∅ ∧ t2 ∩ t3 = ∅ ∧ t3 ∩ t1 = ∅ ∧
    forms_triangle t1 ∧
    forms_triangle t2 ∧
    forms_triangle t3 :=
  sorry

end three_equal_perimeter_triangles_l3831_383163


namespace reciprocal_of_negative_three_l3831_383123

theorem reciprocal_of_negative_three :
  (1 : ℚ) / (-3 : ℚ) = -1/3 := by sorry

end reciprocal_of_negative_three_l3831_383123


namespace initial_money_calculation_l3831_383159

theorem initial_money_calculation (initial_amount : ℝ) : 
  (initial_amount / 2 - (initial_amount / 2) / 2 = 51) → initial_amount = 204 := by
  sorry

end initial_money_calculation_l3831_383159


namespace tree_planting_problem_l3831_383165

/-- Represents a triangle with given side lengths -/
structure Triangle where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ

/-- Calculates the number of trees that can be planted along a triangle's perimeter -/
def treesAlongPerimeter (t : Triangle) (treeSpacing : ℕ) : ℕ :=
  (t.side1 + t.side2 + t.side3) / treeSpacing

theorem tree_planting_problem :
  let triangle := Triangle.mk 198 180 210
  let treeSpacing := 6
  treesAlongPerimeter triangle treeSpacing = 98 := by
  sorry

end tree_planting_problem_l3831_383165


namespace square_root_equality_l3831_383124

theorem square_root_equality (n : ℕ) :
  (((n * 2021^2) / n : ℝ).sqrt = 2021^2) → n = 2021^2 := by
  sorry

end square_root_equality_l3831_383124


namespace a_5_value_l3831_383129

def S (n : ℕ) : ℤ := n^2 - 10*n

theorem a_5_value : 
  (S 5 : ℤ) - (S 4 : ℤ) = -1 :=
by sorry

end a_5_value_l3831_383129


namespace complex_equation_sum_l3831_383157

theorem complex_equation_sum (m n : ℝ) : 
  (m + n * Complex.I) * (4 - 2 * Complex.I) = 3 * Complex.I + 5 → m + n = 9/5 := by
  sorry

end complex_equation_sum_l3831_383157


namespace average_income_proof_l3831_383192

def daily_incomes : List ℝ := [200, 150, 750, 400, 500]

theorem average_income_proof : 
  (daily_incomes.sum / daily_incomes.length : ℝ) = 400 := by
  sorry

end average_income_proof_l3831_383192


namespace correct_average_after_error_correction_l3831_383162

theorem correct_average_after_error_correction 
  (n : ℕ) 
  (initial_average : ℚ) 
  (wrong_mark correct_mark : ℚ) : 
  n = 10 → 
  initial_average = 100 → 
  wrong_mark = 90 → 
  correct_mark = 10 → 
  (n * initial_average - (wrong_mark - correct_mark)) / n = 92 := by
sorry

end correct_average_after_error_correction_l3831_383162


namespace sqrt_defined_for_five_l3831_383108

theorem sqrt_defined_for_five : ∃ (x : ℝ), x = 5 ∧ x - 4 ≥ 0 := by
  sorry

end sqrt_defined_for_five_l3831_383108


namespace exists_special_number_l3831_383152

def divisor_count (n : ℕ) : ℕ := (Nat.divisors n).card

theorem exists_special_number : 
  ∃ n : ℕ, ∀ i : ℕ, i ≤ 1402 → 
    (divisor_count n : ℝ) / (divisor_count (n + i) : ℝ) > 1401 ∧
    (divisor_count n : ℝ) / (divisor_count (n - i) : ℝ) > 1401 :=
sorry

end exists_special_number_l3831_383152


namespace proposition_intersection_l3831_383120

-- Define the propositions p and q
def p (a : ℝ) : Prop := a^2 - 5*a ≥ 0

def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + 4 ≠ 0

-- Define the range of a
def range_a (a : ℝ) : Prop := -4 < a ∧ a ≤ 0

-- Theorem statement
theorem proposition_intersection (a : ℝ) : p a ∧ q a ↔ range_a a := by sorry

end proposition_intersection_l3831_383120


namespace vegetarian_eaters_count_l3831_383113

/-- Represents the dietary preferences of a family -/
structure DietaryPreferences where
  total : Nat
  vegetarianOnly : Nat
  nonVegetarianOnly : Nat
  bothVegAndNonVeg : Nat
  veganOnly : Nat
  pescatarian : Nat
  specificVegetarian : Nat

/-- Calculates the number of people eating vegetarian food -/
def countVegetarianEaters (prefs : DietaryPreferences) : Nat :=
  prefs.vegetarianOnly + prefs.bothVegAndNonVeg + prefs.veganOnly + prefs.pescatarian + prefs.specificVegetarian

/-- Theorem stating that 29 people eat vegetarian food in the given family -/
theorem vegetarian_eaters_count (prefs : DietaryPreferences)
  (h1 : prefs.total = 35)
  (h2 : prefs.vegetarianOnly = 11)
  (h3 : prefs.nonVegetarianOnly = 6)
  (h4 : prefs.bothVegAndNonVeg = 9)
  (h5 : prefs.veganOnly = 3)
  (h6 : prefs.pescatarian = 4)
  (h7 : prefs.specificVegetarian = 2) :
  countVegetarianEaters prefs = 29 := by
  sorry


end vegetarian_eaters_count_l3831_383113


namespace line_through_5_2_slope_neg1_l3831_383170

/-- The point-slope form equation of a line passing through a given point with a given slope -/
def point_slope_form (x₀ y₀ m : ℝ) (x y : ℝ) : Prop :=
  y - y₀ = m * (x - x₀)

/-- Theorem: The point-slope form equation of the line passing through (5, 2) with slope -1 -/
theorem line_through_5_2_slope_neg1 (x y : ℝ) :
  point_slope_form 5 2 (-1) x y ↔ y - 2 = -(x - 5) :=
by sorry

end line_through_5_2_slope_neg1_l3831_383170


namespace shaded_region_correct_l3831_383196

def shaded_region : Set ℂ := {z : ℂ | Complex.abs z ≤ 1 ∧ Complex.im z ≥ (1/2 : ℝ)}

theorem shaded_region_correct :
  ∀ z : ℂ, z ∈ shaded_region ↔ Complex.abs z ≤ 1 ∧ Complex.im z ≥ (1/2 : ℝ) := by sorry

end shaded_region_correct_l3831_383196


namespace recurring_decimal_multiplication_l3831_383169

theorem recurring_decimal_multiplication : 
  (37 / 999) * (7 / 9) = 259 / 8991 := by sorry

end recurring_decimal_multiplication_l3831_383169


namespace remainder_65_pow_65_plus_65_mod_97_l3831_383102

theorem remainder_65_pow_65_plus_65_mod_97 (h1 : Prime 97) (h2 : 65 < 97) : 
  (65^65 + 65) % 97 = 33 := by
  sorry

end remainder_65_pow_65_plus_65_mod_97_l3831_383102


namespace binomial_9_choose_3_l3831_383101

theorem binomial_9_choose_3 : Nat.choose 9 3 = 84 := by sorry

end binomial_9_choose_3_l3831_383101


namespace sample_size_proof_l3831_383198

theorem sample_size_proof (n : ℕ) : 
  (∃ x : ℚ, 
    x > 0 ∧ 
    2*x + 3*x + 4*x + 6*x + 4*x + x = 1 ∧ 
    (2*x + 3*x + 4*x) * n = 27) → 
  n = 60 := by
sorry

end sample_size_proof_l3831_383198


namespace fraction_sum_bound_l3831_383179

theorem fraction_sum_bound (a b c : ℕ) (h : (1 : ℚ) / a + 1 / b + 1 / c < 1) :
  (1 : ℚ) / a + 1 / b + 1 / c < 41 / 42 := by
  sorry

end fraction_sum_bound_l3831_383179


namespace problem_statement_l3831_383119

theorem problem_statement :
  (¬ ∃ x : ℝ, 0 < x ∧ x < 2 ∧ x^3 - x^2 - x + 2 < 0) ∧
  (¬ ∀ x y : ℝ, x + y > 4 → x > 2 ∧ y > 2) := by
  sorry

end problem_statement_l3831_383119


namespace trees_in_yard_l3831_383139

/-- The number of trees in a yard with given length and tree spacing -/
def numberOfTrees (yardLength : ℕ) (treeSpacing : ℕ) : ℕ :=
  (yardLength / treeSpacing) + 1

/-- Theorem: In a 225-meter long yard with trees spaced 10 meters apart, there are 24 trees -/
theorem trees_in_yard : numberOfTrees 225 10 = 24 := by
  sorry

end trees_in_yard_l3831_383139


namespace nail_color_percentage_difference_l3831_383151

theorem nail_color_percentage_difference :
  let total_nails : ℕ := 20
  let purple_nails : ℕ := 6
  let blue_nails : ℕ := 8
  let striped_nails : ℕ := total_nails - purple_nails - blue_nails
  let blue_percentage : ℚ := blue_nails / total_nails * 100
  let striped_percentage : ℚ := striped_nails / total_nails * 100
  blue_percentage - striped_percentage = 10 := by
  sorry

end nail_color_percentage_difference_l3831_383151


namespace gratuities_calculation_l3831_383193

/-- Calculates the gratuities charged by a restaurant given the total bill, tax rate, and item costs. -/
def calculate_gratuities (total_bill : ℚ) (tax_rate : ℚ) (striploin_cost : ℚ) (wine_cost : ℚ) : ℚ :=
  let bill_before_tax := striploin_cost + wine_cost
  let sales_tax := bill_before_tax * tax_rate
  let bill_with_tax := bill_before_tax + sales_tax
  total_bill - bill_with_tax

/-- Theorem stating that the gratuities charged equals $41 given the problem conditions. -/
theorem gratuities_calculation :
  calculate_gratuities 140 (1/10) 80 10 = 41 := by
  sorry

#eval calculate_gratuities 140 (1/10) 80 10

end gratuities_calculation_l3831_383193


namespace weight_of_brand_a_l3831_383111

/-- The weight of one liter of brand 'b' vegetable ghee in grams -/
def weight_b : ℝ := 800

/-- The ratio of brand 'a' to brand 'b' in the mixture -/
def ratio_a : ℝ := 3
def ratio_b : ℝ := 2

/-- The total volume of the mixture in liters -/
def total_volume : ℝ := 4

/-- The total weight of the mixture in grams -/
def total_weight : ℝ := 3440

/-- The weight of one liter of brand 'a' vegetable ghee in grams -/
def weight_a : ℝ := 580

theorem weight_of_brand_a :
  weight_a * (ratio_a / (ratio_a + ratio_b)) * total_volume +
  weight_b * (ratio_b / (ratio_a + ratio_b)) * total_volume = total_weight :=
by sorry

end weight_of_brand_a_l3831_383111


namespace arithmetic_sequence_implies_equilateral_l3831_383153

-- Define a triangle with sides a, b, and c
structure Triangle :=
  (a b c : ℝ)
  (positive_a : 0 < a)
  (positive_b : 0 < b)
  (positive_c : 0 < c)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)

-- Define the arithmetic sequence property for the sides
def sides_arithmetic_sequence (t : Triangle) : Prop :=
  2 * t.b = t.a + t.c

-- Define the arithmetic sequence property for the square roots of the sides
def sqrt_sides_arithmetic_sequence (t : Triangle) : Prop :=
  2 * Real.sqrt t.b = Real.sqrt t.a + Real.sqrt t.c

-- Define an equilateral triangle
def is_equilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- Theorem statement
theorem arithmetic_sequence_implies_equilateral (t : Triangle) 
  (h1 : sides_arithmetic_sequence t) 
  (h2 : sqrt_sides_arithmetic_sequence t) : 
  is_equilateral t :=
sorry

end arithmetic_sequence_implies_equilateral_l3831_383153


namespace probability_queens_or_aces_l3831_383149

/-- Represents a standard deck of 52 cards -/
def standard_deck : ℕ := 52

/-- Number of queens in a standard deck -/
def num_queens : ℕ := 4

/-- Number of aces in a standard deck -/
def num_aces : ℕ := 4

/-- Number of cards drawn -/
def cards_drawn : ℕ := 3

/-- Probability of drawing all queens or at least 2 aces -/
def prob_queens_or_aces : ℚ := 220 / 581747

theorem probability_queens_or_aces :
  let total_ways := standard_deck.choose cards_drawn
  let ways_all_queens := num_queens.choose cards_drawn
  let ways_two_aces := cards_drawn.choose 2 * num_aces.choose 2 * (standard_deck - num_aces)
  let ways_three_aces := num_aces.choose cards_drawn
  (ways_all_queens + ways_two_aces + ways_three_aces : ℚ) / total_ways = prob_queens_or_aces := by
  sorry

end probability_queens_or_aces_l3831_383149


namespace cone_volume_l3831_383173

/-- The volume of a cone with given conditions -/
theorem cone_volume (r h : ℝ) (hr : r = 3) 
  (hθ : 2 * π * r = 2 * π * r / 3 * 9) : 
  (1 / 3) * π * r^2 * h = 18 * Real.sqrt 2 * π := by
  sorry

end cone_volume_l3831_383173


namespace exist_a_b_satisfying_conditions_l3831_383148

theorem exist_a_b_satisfying_conditions : ∃ (a b : ℝ), 
  a > 0 ∧ b > 0 ∧ a * b * (a - b) = 1 ∧ a^2 + b^2 = Real.sqrt 11 := by
  sorry

end exist_a_b_satisfying_conditions_l3831_383148


namespace equation_solution_l3831_383116

theorem equation_solution :
  ∀ (k m n : ℕ),
  (1/2 : ℝ)^16 * (1/81 : ℝ)^k = (1/18 : ℝ)^16 →
  (1/3 : ℝ)^n * (1/27 : ℝ)^m = (1/18 : ℝ)^k →
  k = 8 ∧ n + 3 * m = 16 :=
by
  sorry

end equation_solution_l3831_383116


namespace rhombus_existence_condition_l3831_383105

/-- A rhombus is a quadrilateral with four equal sides. -/
structure Rhombus where
  /-- The perimeter of the rhombus -/
  k : ℝ
  /-- The sum of the diagonals of the rhombus -/
  u : ℝ
  /-- The perimeter is positive -/
  k_pos : k > 0
  /-- The sum of diagonals is positive -/
  u_pos : u > 0

/-- The condition for the existence of a rhombus given its perimeter and sum of diagonals -/
theorem rhombus_existence_condition (r : Rhombus) : 
  Real.sqrt 2 * r.u ≤ r.k ∧ r.k < 2 * r.u :=
by sorry

end rhombus_existence_condition_l3831_383105


namespace simplify_expression_l3831_383117

theorem simplify_expression (x : ℝ) : (x - 3)^2 - (x + 1)*(x - 1) = -6*x + 10 := by
  sorry

end simplify_expression_l3831_383117


namespace cos_phi_value_l3831_383103

-- Define the function f
variable (f : ℝ → ℝ)

-- Define φ
variable (φ : ℝ)

-- Define x₁
variable (x₁ : ℝ)

-- f(x) - sin(x + φ) is an even function
axiom even_func : ∀ x, f (-x) - Real.sin (-x + φ) = f x - Real.sin (x + φ)

-- f(x) - cos(x + φ) is an odd function
axiom odd_func : ∀ x, f (-x) - Real.cos (-x + φ) = -(f x - Real.cos (x + φ))

-- The slopes of the tangent lines at P and Q are reciprocals
axiom reciprocal_slopes : 
  (deriv f x₁) * (deriv f (x₁ + Real.pi / 2)) = 1

-- Theorem statement
theorem cos_phi_value : Real.cos φ = 1 ∨ Real.cos φ = -1 := by
  sorry

end cos_phi_value_l3831_383103


namespace cubic_root_sum_square_l3831_383142

theorem cubic_root_sum_square (a b c s : ℝ) : 
  a^3 - 15*a^2 + 20*a - 4 = 0 →
  b^3 - 15*b^2 + 20*b - 4 = 0 →
  c^3 - 15*c^2 + 20*c - 4 = 0 →
  s = Real.sqrt a + Real.sqrt b + Real.sqrt c →
  s^4 - 28*s^2 - 20*s = 305 + 2*s^2 - 20*s := by
  sorry

end cubic_root_sum_square_l3831_383142


namespace quadratic_form_equivalence_l3831_383112

theorem quadratic_form_equivalence :
  ∀ x : ℝ, x^2 + 2*x - 2 = (x + 1)^2 - 3 := by
  sorry

end quadratic_form_equivalence_l3831_383112


namespace polynomial_value_at_three_l3831_383131

def f (x : ℝ) : ℝ := 4 * x^5 - 3 * x^3 + 2 * x^2 + 5 * x + 1

theorem polynomial_value_at_three : f 3 = 925 := by
  sorry

end polynomial_value_at_three_l3831_383131


namespace min_throws_correct_l3831_383144

/-- The probability of hitting the target in a single throw -/
def p : ℝ := 0.6

/-- The desired minimum probability of hitting the target at least once -/
def target_prob : ℝ := 0.9

/-- The minimum number of throws needed to exceed the target probability -/
def min_throws : ℕ := 3

/-- Theorem stating that min_throws is the minimum number of throws needed -/
theorem min_throws_correct :
  (∀ n : ℕ, n < min_throws → 1 - (1 - p)^n ≤ target_prob) ∧
  (1 - (1 - p)^min_throws > target_prob) :=
sorry

end min_throws_correct_l3831_383144


namespace dot_no_line_count_l3831_383121

/-- Represents an alphabet with letters containing dots and straight lines -/
structure Alphabet :=
  (total : ℕ)
  (both : ℕ)
  (line_no_dot : ℕ)
  (has_dot_or_line : total = both + line_no_dot + (total - (both + line_no_dot)))

/-- The number of letters containing a dot but not a straight line -/
def dot_no_line (α : Alphabet) : ℕ :=
  α.total - (α.both + α.line_no_dot)

theorem dot_no_line_count (α : Alphabet) 
  (h1 : α.total = 40)
  (h2 : α.both = 11)
  (h3 : α.line_no_dot = 24) :
  dot_no_line α = 5 := by
  sorry

end dot_no_line_count_l3831_383121


namespace greatest_m_value_l3831_383186

theorem greatest_m_value (p m : ℕ) (hp : Nat.Prime p) 
  (heq : p * (p + m) + 2 * p = (m + 2)^3) : 
  m ≤ 28 ∧ ∃ (p' m' : ℕ), Nat.Prime p' ∧ p' * (p' + 28) + 2 * p' = (28 + 2)^3 := by
  sorry

end greatest_m_value_l3831_383186


namespace smallest_a1_l3831_383167

/-- A sequence of positive real numbers satisfying the given recurrence relation. -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n > 1, a n = 13 * a (n - 1) - 2 * n)

/-- The theorem stating the smallest possible value of a₁ in the sequence. -/
theorem smallest_a1 (a : ℕ → ℝ) (h : RecurrenceSequence a) :
    ∀ a₁ : ℝ, a 1 ≥ a₁ → a₁ ≥ 25 / 72 :=
  sorry

end smallest_a1_l3831_383167


namespace trigonometric_identities_l3831_383175

open Real

theorem trigonometric_identities (α : ℝ) (h : tan α = 2) :
  (2 * sin α - cos α) / (sin α + 2 * cos α) = 3/4 ∧
  2 * sin α^2 - sin α * cos α + cos α^2 = 7/5 := by
  sorry

end trigonometric_identities_l3831_383175


namespace brother_catch_up_l3831_383130

/-- The time it takes for the younger brother to reach school (in minutes) -/
def younger_time : ℝ := 25

/-- The time it takes for the older brother to reach school (in minutes) -/
def older_time : ℝ := 15

/-- The time difference between when the older brother leaves after the younger brother (in minutes) -/
def time_difference : ℝ := 8

/-- The time when the older brother catches up to the younger brother (in minutes after the younger brother leaves) -/
def catch_up_time : ℝ := 20

theorem brother_catch_up :
  let younger_speed := 1 / younger_time
  let older_speed := 1 / older_time
  younger_speed * catch_up_time = older_speed * (catch_up_time - time_difference) := by sorry

end brother_catch_up_l3831_383130


namespace distribution_theorem_l3831_383182

/-- The number of ways to distribute 4 students among 4 universities 
    such that exactly two students are admitted to the same university -/
def distribution_count : ℕ := 144

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of universities -/
def num_universities : ℕ := 4

theorem distribution_theorem : 
  (num_students = 4) → 
  (num_universities = 4) → 
  (distribution_count = 144) := by
  sorry

end distribution_theorem_l3831_383182


namespace f_value_at_5pi_3_l3831_383141

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  is_periodic f p ∧ p > 0 ∧ ∀ q, is_periodic f q ∧ q > 0 → p ≤ q

theorem f_value_at_5pi_3 (f : ℝ → ℝ) :
  is_even f →
  smallest_positive_period f π →
  (∀ x ∈ Set.Icc 0 (π/2), f x = Real.sin (x/2)) →
  f (5*π/3) = 1/2 := by sorry

end f_value_at_5pi_3_l3831_383141


namespace share_multiple_problem_l3831_383161

theorem share_multiple_problem (total a b c x : ℚ) : 
  total = 880 →
  c = 160 →
  a + b + c = total →
  4 * a = 5 * b →
  4 * a = x * c →
  x = 10 := by
  sorry

end share_multiple_problem_l3831_383161


namespace set_intersection_and_complement_l3831_383109

def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2 < x ∧ x ≤ 5}

theorem set_intersection_and_complement :
  (A ∩ B = {x | 2 < x ∧ x < 3}) ∧
  ((Set.univ \ A) ∩ B = {x | 3 ≤ x ∧ x ≤ 5}) := by
  sorry

end set_intersection_and_complement_l3831_383109


namespace quadratic_equation_roots_unique_positive_integer_solution_l3831_383178

/-- The quadratic equation x^2 - 2x + 2m - 1 = 0 has real roots -/
def has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - 2*x + 2*m - 1 = 0

/-- m is a positive integer -/
def is_positive_integer (m : ℝ) : Prop :=
  m > 0 ∧ ∃ n : ℕ, m = n

theorem quadratic_equation_roots (m : ℝ) :
  has_real_roots m ↔ m ≤ 1 :=
sorry

theorem unique_positive_integer_solution (m : ℝ) :
  is_positive_integer m ∧ has_real_roots m →
  m = 1 ∧ ∃ x : ℝ, x = 1 ∧ x^2 - 2*x + 2*m - 1 = 0 :=
sorry

end quadratic_equation_roots_unique_positive_integer_solution_l3831_383178


namespace current_visitors_count_l3831_383156

/-- The number of visitors to the Buckingham palace on the previous day -/
def previous_visitors : ℕ := 600

/-- The additional number of visitors compared to the previous day -/
def additional_visitors : ℕ := 61

/-- Theorem: The number of visitors to the Buckingham palace on the current day is 661 -/
theorem current_visitors_count : previous_visitors + additional_visitors = 661 := by
  sorry

end current_visitors_count_l3831_383156


namespace library_books_end_of_month_l3831_383164

theorem library_books_end_of_month 
  (initial_books : ℕ) 
  (loaned_books : ℕ) 
  (return_rate : ℚ) : 
  initial_books = 75 → 
  loaned_books = 50 → 
  return_rate = 70 / 100 → 
  initial_books - (loaned_books - (return_rate * loaned_books).floor) = 60 := by
sorry

end library_books_end_of_month_l3831_383164


namespace two_hour_charge_is_161_l3831_383104

/-- Represents the pricing structure and total charges for a psychologist's therapy sessions. -/
structure TherapyPricing where
  first_hour : ℕ  -- Price for the first hour
  additional_hour : ℕ  -- Price for each additional hour
  first_hour_premium : first_hour = additional_hour + 35  -- First hour costs $35 more
  five_hour_total : first_hour + 4 * additional_hour = 350  -- Total for 5 hours is $350

/-- Calculates the total charge for 2 hours of therapy given the pricing structure. -/
def two_hour_charge (pricing : TherapyPricing) : ℕ :=
  pricing.first_hour + pricing.additional_hour

/-- Theorem stating that the total charge for 2 hours of therapy is $161. -/
theorem two_hour_charge_is_161 (pricing : TherapyPricing) : 
  two_hour_charge pricing = 161 := by
  sorry

end two_hour_charge_is_161_l3831_383104


namespace additional_amount_is_three_l3831_383125

/-- The minimum purchase amount required for free delivery -/
def min_purchase : ℝ := 18

/-- The cost of a quarter-pounder burger -/
def burger_cost : ℝ := 3.20

/-- The cost of large fries -/
def fries_cost : ℝ := 1.90

/-- The cost of a milkshake -/
def milkshake_cost : ℝ := 2.40

/-- The number of each item Danny ordered -/
def quantity : ℕ := 2

/-- The total cost of Danny's current order -/
def order_total : ℝ := quantity * burger_cost + quantity * fries_cost + quantity * milkshake_cost

/-- The additional amount needed for free delivery -/
def additional_amount : ℝ := min_purchase - order_total

theorem additional_amount_is_three :
  additional_amount = 3 :=
by sorry

end additional_amount_is_three_l3831_383125


namespace square_with_1983_nines_l3831_383147

theorem square_with_1983_nines : ∃ N : ℕ,
  (N^2 = 10 * 88) ∧
  (∃ k : ℕ, N = 10^1984 - 1 + k ∧ k < 10^1984) := by
  sorry

end square_with_1983_nines_l3831_383147


namespace complex_subtraction_and_multiplication_l3831_383122

theorem complex_subtraction_and_multiplication (i : ℂ) :
  (7 - 3 * i) - 3 * (2 + 5 * i) = 1 - 18 * i :=
by sorry

end complex_subtraction_and_multiplication_l3831_383122


namespace zhang_qiu_jian_gold_distribution_l3831_383160

/-- Represents the gold distribution problem from "Zhang Qiu Jian Suan Jing" -/
theorem zhang_qiu_jian_gold_distribution 
  (n : ℕ) 
  (gold : ℕ → ℚ) 
  (h1 : n = 10) 
  (h2 : ∀ i j, i < j → gold i < gold j) 
  (h3 : gold 8 + gold 9 + gold 10 = 4) 
  (h4 : gold 4 + gold 5 + gold 6 + gold 7 = 3) 
  (h5 : ∀ i j, j = i + 1 → gold j - gold i = gold (i+1) - gold i) :
  gold 5 + gold 6 + gold 7 = 83/26 := by
  sorry

end zhang_qiu_jian_gold_distribution_l3831_383160


namespace distribute_five_among_three_l3831_383168

/-- The number of ways to distribute n distinguishable objects among k distinct categories -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to distribute 5 distinguishable objects among 3 distinct categories -/
theorem distribute_five_among_three : distribute 5 3 = 243 := by
  sorry

end distribute_five_among_three_l3831_383168


namespace square_sum_given_conditions_l3831_383191

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : (x + y)^2 = 49)
  (h2 : x * y = 12) : 
  x^2 + y^2 = 25 := by
sorry

end square_sum_given_conditions_l3831_383191


namespace noah_painting_sales_l3831_383140

/-- Noah's painting sales calculation -/
theorem noah_painting_sales :
  let large_price : ℕ := 60
  let small_price : ℕ := 30
  let last_month_large : ℕ := 8
  let last_month_small : ℕ := 4
  let this_month_multiplier : ℕ := 2
  
  (large_price * last_month_large + small_price * last_month_small) * this_month_multiplier = 1200 :=
by sorry

end noah_painting_sales_l3831_383140


namespace divisor_sum_theorem_l3831_383135

/-- The number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The property that the sum of divisors of three consecutive numbers is at most 8 -/
def divisor_sum_property (n : ℕ) : Prop :=
  num_divisors (n - 1) + num_divisors n + num_divisors (n + 1) ≤ 8

/-- Theorem stating that the divisor sum property holds if and only if n is 3, 4, or 6 -/
theorem divisor_sum_theorem (n : ℕ) (h : n ≥ 3) :
  divisor_sum_property n ↔ n = 3 ∨ n = 4 ∨ n = 6 := by
  sorry

end divisor_sum_theorem_l3831_383135


namespace courtyard_breadth_l3831_383128

/-- Calculates the breadth of a rectangular courtyard given its length, the number of bricks used, and the dimensions of each brick. -/
theorem courtyard_breadth
  (length : ℝ)
  (num_bricks : ℕ)
  (brick_length brick_width : ℝ)
  (h1 : length = 20)
  (h2 : num_bricks = 16000)
  (h3 : brick_length = 0.2)
  (h4 : brick_width = 0.1) :
  length * (num_bricks : ℝ) * brick_length * brick_width / length = 16 :=
by sorry

end courtyard_breadth_l3831_383128


namespace ant_borya_position_l3831_383187

/-- Represents a point on the coordinate plane -/
structure Point where
  x : Int
  y : Int

/-- Generates the nth point in the spiral sequence -/
def spiral_point (n : Nat) : Point :=
  sorry

/-- The starting point of the sequence -/
def P₀ : Point := { x := 0, y := 0 }

/-- The second point in the sequence -/
def P₁ : Point := { x := 1, y := 0 }

/-- The spiral sequence of points -/
def P : Nat → Point
  | 0 => P₀
  | 1 => P₁
  | n + 2 => spiral_point (n + 2)

theorem ant_borya_position : P 1557 = { x := 20, y := 17 } := by
  sorry

end ant_borya_position_l3831_383187


namespace total_books_equals_135_l3831_383183

def first_day_books : ℕ := 54
def second_day_books : ℕ := 23
def third_day_multiplier : ℕ := 3

def total_books : ℕ :=
  first_day_books +
  (second_day_books + 1) / 2 +
  third_day_multiplier * second_day_books

theorem total_books_equals_135 : total_books = 135 := by
  sorry

end total_books_equals_135_l3831_383183


namespace inverse_function_property_l3831_383155

-- Define the function g
def g (x : ℝ) : ℝ := 1 + 2 * x

-- State the theorem
theorem inverse_function_property (f : ℝ → ℝ) :
  (∀ x, g (f x) = x) ∧ (∀ x, f (g x) = x) →
  f 1 = 0 := by sorry

end inverse_function_property_l3831_383155


namespace mike_remaining_amount_l3831_383154

/-- Calculates the remaining amount for a partner in a profit-sharing scenario -/
def remaining_amount (total_parts : ℕ) (partner_parts : ℕ) (other_partner_amount : ℕ) (spending : ℕ) : ℕ :=
  let part_value := other_partner_amount / (total_parts - partner_parts)
  let partner_amount := part_value * partner_parts
  partner_amount - spending

/-- Theorem stating the remaining amount for Mike in the given profit-sharing scenario -/
theorem mike_remaining_amount :
  remaining_amount 7 2 2500 200 = 800 := by
  sorry

end mike_remaining_amount_l3831_383154


namespace antiderivative_derivative_l3831_383185

/-- The derivative of the antiderivative is equal to the original function -/
theorem antiderivative_derivative (x : ℝ) :
  let f : ℝ → ℝ := λ x => (2*x^3 + 3*x^2 + 3*x + 2) / ((x^2 + x + 1)*(x^2 + 1))
  let F : ℝ → ℝ := λ x => (1/2) * Real.log (abs (x^2 + x + 1)) +
                          (1/Real.sqrt 3) * Real.arctan ((2*x + 1)/Real.sqrt 3) +
                          (1/2) * Real.log (abs (x^2 + 1)) +
                          Real.arctan x
  (deriv F) x = f x :=
by sorry

end antiderivative_derivative_l3831_383185


namespace alpha_sufficient_not_necessary_for_beta_l3831_383143

def α (x y : ℝ) : Prop := x = 1 ∧ y = 2
def β (x y : ℝ) : Prop := x + y = 3

theorem alpha_sufficient_not_necessary_for_beta :
  (∀ x y : ℝ, α x y → β x y) ∧
  (∃ x y : ℝ, β x y ∧ ¬(α x y)) := by
  sorry

end alpha_sufficient_not_necessary_for_beta_l3831_383143


namespace max_value_is_80_l3831_383166

structure Rock :=
  (weight : ℕ)
  (value : ℕ)

def rock_types : List Rock := [
  ⟨6, 20⟩,
  ⟨3, 9⟩,
  ⟨2, 4⟩
]

def max_weight : ℕ := 24

def min_available : ℕ := 10

def optimal_value (rocks : List Rock) (max_w : ℕ) (min_avail : ℕ) : ℕ :=
  sorry

theorem max_value_is_80 :
  optimal_value rock_types max_weight min_available = 80 :=
sorry

end max_value_is_80_l3831_383166


namespace where_is_waldo_books_l3831_383177

/-- The number of "Where's Waldo?" books published -/
def num_books : ℕ := 15

/-- The number of puzzles in each "Where's Waldo?" book -/
def puzzles_per_book : ℕ := 30

/-- The time in minutes to solve one puzzle -/
def time_per_puzzle : ℕ := 3

/-- The total time in minutes to solve all puzzles -/
def total_time : ℕ := 1350

/-- Theorem stating that the number of "Where's Waldo?" books is correct -/
theorem where_is_waldo_books :
  num_books = total_time / (puzzles_per_book * time_per_puzzle) :=
by sorry

end where_is_waldo_books_l3831_383177


namespace polynomial_factors_sum_l3831_383199

/-- Given real numbers a, b, and c, if x^2 + x + 2 is a factor of ax^3 + bx^2 + cx + 5
    and 2x - 1 is a factor of ax^3 + bx^2 + cx - 25/16, then a + b + c = 45/11 -/
theorem polynomial_factors_sum (a b c : ℝ) :
  (∃ d : ℝ, ∀ x : ℝ, a * x^3 + b * x^2 + c * x + 5 = (x^2 + x + 2) * (a * x + d)) →
  (∀ x : ℝ, (2 * x - 1) ∣ (a * x^3 + b * x^2 + c * x - 25/16)) →
  a + b + c = 45/11 := by
  sorry

end polynomial_factors_sum_l3831_383199


namespace log_expression_1_log_expression_2_l3831_383194

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Theorem for the first expression
theorem log_expression_1 :
  2 * log 3 2 - log 3 (32 / 9) + log 3 8 - (5 : ℝ) ^ (log 5 3) = -1 :=
sorry

-- Theorem for the second expression
theorem log_expression_2 :
  log 2 25 * log 3 4 * log 5 9 = 8 :=
sorry

end log_expression_1_log_expression_2_l3831_383194


namespace unique_function_theorem_l3831_383134

-- Define the sum of digits function
def S (n : ℕ+) : ℕ+ :=
  sorry

-- Define the properties of the function f
def satisfies_conditions (f : ℕ+ → ℕ+) : Prop :=
  (∀ n : ℕ+, f n < f (n + 1) ∧ f (n + 1) < f n + 2020) ∧
  (∀ n : ℕ+, S (f n) = f (S n))

-- Theorem statement
theorem unique_function_theorem :
  ∃! f : ℕ+ → ℕ+, satisfies_conditions f ∧ ∀ x : ℕ+, f x = x :=
sorry

end unique_function_theorem_l3831_383134


namespace mobile_phone_purchase_price_l3831_383106

theorem mobile_phone_purchase_price 
  (grinder_price : ℕ) 
  (grinder_loss_percent : ℚ) 
  (mobile_profit_percent : ℚ) 
  (total_profit : ℕ) :
  let mobile_price : ℕ := 8000
  let grinder_sold_price : ℚ := grinder_price * (1 - grinder_loss_percent)
  let mobile_sold_price : ℚ := mobile_price * (1 + mobile_profit_percent)
  grinder_price = 15000 ∧ 
  grinder_loss_percent = 2 / 100 ∧ 
  mobile_profit_percent = 10 / 100 ∧ 
  total_profit = 500 →
  grinder_sold_price + mobile_sold_price = grinder_price + mobile_price + total_profit :=
by sorry

end mobile_phone_purchase_price_l3831_383106


namespace vector_simplification_l3831_383137

variable {V : Type*} [AddCommGroup V]

/-- For any five points A, B, C, D, E in a vector space,
    AC + DE + EB - AB = DC -/
theorem vector_simplification (A B C D E : V) :
  (C - A) + (E - D) + (B - E) - (B - A) = C - D := by sorry

end vector_simplification_l3831_383137


namespace sin_increases_with_angle_l3831_383171

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- angles
  (a b c : Real)  -- sides
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)
  (angle_sum : A + B + C = π)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (law_of_sines : a / Real.sin A = b / Real.sin B)

-- Theorem statement
theorem sin_increases_with_angle (abc : Triangle) (h : abc.A > abc.B) :
  Real.sin abc.A > Real.sin abc.B :=
sorry

end sin_increases_with_angle_l3831_383171


namespace sum_not_prime_l3831_383158

theorem sum_not_prime (a b c d : ℕ+) (h : a * b = c * d) : ¬ Nat.Prime (a + b + c + d) := by
  sorry

end sum_not_prime_l3831_383158


namespace product_of_primes_sum_74_l3831_383100

theorem product_of_primes_sum_74 (p q : ℕ) : 
  Prime p → Prime q → p + q = 74 → p * q = 1369 := by sorry

end product_of_primes_sum_74_l3831_383100


namespace range_of_a_l3831_383107

/-- A function f(x) that depends on a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 3*a*x + 1

/-- The derivative of f(x) with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x + 3*a

/-- The discriminant of f'(x) = 0 -/
def discriminant (a : ℝ) : ℝ := 4*a^2 - 36*a

theorem range_of_a (a : ℝ) :
  (∃ (max min : ℝ), ∀ x, f a x ≤ max ∧ min ≤ f a x) →
  (a < 0 ∨ a > 9) :=
by sorry

end range_of_a_l3831_383107


namespace product_of_repeating_decimal_and_seven_l3831_383150

theorem product_of_repeating_decimal_and_seven :
  ∃ (s : ℚ), (s = 456 / 999) ∧ (s * 7 = 118 / 37) := by
  sorry

end product_of_repeating_decimal_and_seven_l3831_383150


namespace smallest_y_value_l3831_383190

theorem smallest_y_value (y : ℝ) : 
  (12 * y^2 - 56 * y + 48 = 0) → y ≥ 2 :=
by sorry

end smallest_y_value_l3831_383190


namespace value_of_c_l3831_383136

theorem value_of_c (c : ℝ) : 
  4 * ((3.6 * 0.48 * c) / (0.12 * 0.09 * 0.5)) = 3200.0000000000005 → c = 2.5 := by
  sorry

end value_of_c_l3831_383136


namespace range_of_m_l3831_383127

-- Define the conditions
def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- State the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(q x m) → ¬(p x)) →
  (∃ x, p x ∧ ¬(q x m)) →
  m ≥ 9 := by sorry

end range_of_m_l3831_383127


namespace additional_money_needed_l3831_383180

-- Define the given values
def perfume_cost : ℚ := 50
def christian_initial : ℚ := 5
def sue_initial : ℚ := 7
def christian_yards : ℕ := 4
def christian_yard_charge : ℚ := 5
def sue_dogs : ℕ := 6
def sue_dog_charge : ℚ := 2

-- Define the theorem
theorem additional_money_needed : 
  perfume_cost - (christian_initial + sue_initial + 
    (christian_yards : ℚ) * christian_yard_charge + 
    (sue_dogs : ℚ) * sue_dog_charge) = 6 := by
  sorry

end additional_money_needed_l3831_383180


namespace right_triangle_inradius_l3831_383138

/-- The inradius of a right triangle with sides 12, 16, and 20 is 4 -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 12 ∧ b = 16 ∧ c = 20 →  -- Side lengths
  a^2 + b^2 = c^2 →           -- Right triangle condition
  (a + b + c) / 2 * r = (a * b) / 2 →  -- Area formula using inradius
  r = 4 := by
  sorry

end right_triangle_inradius_l3831_383138


namespace symmetric_line_equation_l3831_383181

/-- Given a line symmetric to y = 3x - 2 with respect to the y-axis, prove its equation is y = -3x - 2 -/
theorem symmetric_line_equation (l₁ : Set (ℝ × ℝ)) : 
  (∀ (x y : ℝ), (x, y) ∈ l₁ ↔ (-x, y) ∈ {(x, y) | y = 3 * x - 2}) →
  l₁ = {(x, y) | y = -3 * x - 2} :=
by sorry

end symmetric_line_equation_l3831_383181


namespace lowest_price_pet_food_l3831_383114

def msrp : ℝ := 45.00
def max_regular_discount : ℝ := 0.30
def additional_discount : ℝ := 0.20

theorem lowest_price_pet_food :
  let regular_discounted_price := msrp * (1 - max_regular_discount)
  let final_price := regular_discounted_price * (1 - additional_discount)
  final_price = 25.20 := by sorry

end lowest_price_pet_food_l3831_383114


namespace lcm_hcf_problem_l3831_383174

theorem lcm_hcf_problem (x : ℕ) : 
  Nat.lcm 4 x = 36 → Nat.gcd 4 x = 2 → x = 18 := by
  sorry

end lcm_hcf_problem_l3831_383174


namespace systematic_sampling_41st_number_l3831_383195

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  first_selected : ℕ

/-- The nth number in a systematic sample -/
def nth_number (s : SystematicSampling) (n : ℕ) : ℕ :=
  let part_size := s.total_students / s.sample_size
  (n - 1) * part_size + s.first_selected

theorem systematic_sampling_41st_number 
  (s : SystematicSampling) 
  (h1 : s.total_students = 1000) 
  (h2 : s.sample_size = 50) 
  (h3 : s.first_selected = 10) : 
  nth_number s 41 = 810 := by
  sorry

#eval nth_number { total_students := 1000, sample_size := 50, first_selected := 10 } 41

end systematic_sampling_41st_number_l3831_383195
