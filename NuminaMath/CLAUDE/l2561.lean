import Mathlib

namespace NUMINAMATH_CALUDE_isabellas_hair_length_l2561_256198

theorem isabellas_hair_length (current_length cut_length : ℕ) 
  (h1 : current_length = 9)
  (h2 : cut_length = 9) :
  current_length + cut_length = 18 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_hair_length_l2561_256198


namespace NUMINAMATH_CALUDE_puzzle_solution_l2561_256139

def concatenate (a b c : ℕ) : ℕ := 100 * c + 10 * b + a

def special_sum (a b c : ℕ) : ℕ := 
  10000 * (a * b) + 100 * (a * c) + concatenate c b a

theorem puzzle_solution (h1 : special_sum 5 3 2 = 151022)
                        (h2 : special_sum 9 2 4 = 183652)
                        (h3 : special_sum 7 2 5 = 143547) :
  ∃ x, special_sum 7 2 x = 143547 ∧ x = 5 :=
sorry

end NUMINAMATH_CALUDE_puzzle_solution_l2561_256139


namespace NUMINAMATH_CALUDE_weight_of_bart_and_cindy_l2561_256176

/-- Given the weights of pairs of people, prove the weight of a specific pair -/
theorem weight_of_bart_and_cindy 
  (abby bart cindy damon : ℝ) 
  (h1 : abby + bart = 280) 
  (h2 : cindy + damon = 290) 
  (h3 : abby + damon = 300) : 
  bart + cindy = 270 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_bart_and_cindy_l2561_256176


namespace NUMINAMATH_CALUDE_aunt_wang_money_proof_l2561_256190

/-- The price of apples per kilogram -/
def apple_price : ℝ := 5

/-- The amount of money Aunt Wang has -/
def aunt_wang_money : ℝ := 10.9

/-- Proves that Aunt Wang has 10.9 yuan given the problem conditions -/
theorem aunt_wang_money_proof :
  (2.5 * apple_price - aunt_wang_money = 1.6) ∧
  (aunt_wang_money - 2 * apple_price = 0.9) →
  aunt_wang_money = 10.9 :=
by
  sorry

#check aunt_wang_money_proof

end NUMINAMATH_CALUDE_aunt_wang_money_proof_l2561_256190


namespace NUMINAMATH_CALUDE_equation_solutions_l2561_256100

theorem equation_solutions :
  (∀ x : ℝ, 4 * x^2 - 25 = 0 ↔ x = 5/2 ∨ x = -5/2) ∧
  (∀ x : ℝ, (x + 1)^3 = -27 ↔ x = -4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2561_256100


namespace NUMINAMATH_CALUDE_tan_value_implies_cosine_sine_ratio_l2561_256103

theorem tan_value_implies_cosine_sine_ratio 
  (α : Real) 
  (h : Real.tan α = 1/3) : 
  (Real.cos α)^2 - 2*(Real.sin α)^2 = 7/9 * (Real.cos α)^2 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_implies_cosine_sine_ratio_l2561_256103


namespace NUMINAMATH_CALUDE_square_of_101_l2561_256152

theorem square_of_101 : 101 * 101 = 10201 := by
  sorry

end NUMINAMATH_CALUDE_square_of_101_l2561_256152


namespace NUMINAMATH_CALUDE_arc_length_of_curve_l2561_256115

noncomputable def f (x : ℝ) : ℝ := -Real.arccos x + Real.sqrt (1 - x^2) + 1

theorem arc_length_of_curve (a b : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ 9/16) :
  ∫ x in a..b, Real.sqrt (1 + (((1 - x) / Real.sqrt (1 - x^2))^2)) = 1 / Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_arc_length_of_curve_l2561_256115


namespace NUMINAMATH_CALUDE_smallest_multiple_of_seven_l2561_256131

theorem smallest_multiple_of_seven (x y : ℤ) 
  (h1 : (x + 1) % 7 = 0) 
  (h2 : (y - 5) % 7 = 0) : 
  (∃ n : ℕ+, (x^2 + x*y + y^2 + 3*n) % 7 = 0 ∧ 
    ∀ m : ℕ+, (x^2 + x*y + y^2 + 3*m) % 7 = 0 → n ≤ m) → 
  (∃ n : ℕ+, (x^2 + x*y + y^2 + 3*n) % 7 = 0 ∧ 
    ∀ m : ℕ+, (x^2 + x*y + y^2 + 3*m) % 7 = 0 → n ≤ m) ∧ 
  (∃ n : ℕ+, (x^2 + x*y + y^2 + 3*n) % 7 = 0 ∧ 
    ∀ m : ℕ+, (x^2 + x*y + y^2 + 3*m) % 7 = 0 → n ≤ m) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_seven_l2561_256131


namespace NUMINAMATH_CALUDE_locus_of_midpoints_l2561_256177

-- Define the circles and their properties
def circle1_radius : ℝ := 1
def circle2_radius : ℝ := 3
def distance_between_centers : ℝ := 10

-- Define the locus
def locus_inner_radius : ℝ := 1
def locus_outer_radius : ℝ := 2

-- Theorem statement
theorem locus_of_midpoints (p : ℝ × ℝ) : 
  (∃ (p1 p2 : ℝ × ℝ), 
    (p1.1 - 0)^2 + (p1.2 - 0)^2 = circle1_radius^2 ∧ 
    (p2.1 - distance_between_centers)^2 + p2.2^2 = circle2_radius^2 ∧
    p = ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)) ↔ 
  (locus_inner_radius^2 ≤ (p.1 - distance_between_centers / 2)^2 + p.2^2 ∧ 
   (p.1 - distance_between_centers / 2)^2 + p.2^2 ≤ locus_outer_radius^2) :=
sorry

end NUMINAMATH_CALUDE_locus_of_midpoints_l2561_256177


namespace NUMINAMATH_CALUDE_sum_of_quadratic_roots_sum_of_solutions_is_neg_nine_l2561_256129

theorem sum_of_quadratic_roots (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a :=
by sorry

theorem sum_of_solutions_is_neg_nine :
  let a : ℝ := -3
  let b : ℝ := -27
  let c : ℝ := 54
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -9 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_roots_sum_of_solutions_is_neg_nine_l2561_256129


namespace NUMINAMATH_CALUDE_smaller_number_problem_l2561_256145

theorem smaller_number_problem (a b : ℤ) : 
  a + b = 18 → a - b = 24 → min a b = -3 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l2561_256145


namespace NUMINAMATH_CALUDE_uncle_jerry_tomatoes_l2561_256120

/-- The number of tomatoes Uncle Jerry reaped yesterday -/
def yesterday_tomatoes : ℕ := 120

/-- The additional number of tomatoes Uncle Jerry reaped today compared to yesterday -/
def additional_today : ℕ := 50

/-- The total number of tomatoes Uncle Jerry reaped over two days -/
def total_tomatoes : ℕ := yesterday_tomatoes + (yesterday_tomatoes + additional_today)

theorem uncle_jerry_tomatoes :
  total_tomatoes = 290 :=
sorry

end NUMINAMATH_CALUDE_uncle_jerry_tomatoes_l2561_256120


namespace NUMINAMATH_CALUDE_min_value_theorem_l2561_256154

theorem min_value_theorem (x : ℝ) (hx : x > 0) : 2 * x + 18 / x ≥ 12 ∧ (2 * x + 18 / x = 12 ↔ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2561_256154


namespace NUMINAMATH_CALUDE_investment_growth_l2561_256130

/-- Calculates the future value of an investment -/
def future_value (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem investment_growth (principal : ℝ) (rate : ℝ) (time : ℕ) (future_amount : ℝ) 
  (h1 : principal = 376889.02)
  (h2 : rate = 0.06)
  (h3 : time = 8)
  (h4 : future_amount = 600000) :
  future_value principal rate time = future_amount := by
  sorry

end NUMINAMATH_CALUDE_investment_growth_l2561_256130


namespace NUMINAMATH_CALUDE_common_ratio_is_two_l2561_256166

/-- An increasing geometric sequence with specific conditions -/
def IncreasingGeometricSequence (a : ℕ → ℝ) : Prop :=
  (∃ q : ℝ, q > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * q) ∧
  a 2 = 2 ∧
  a 4 - a 3 = 4

/-- The common ratio of the sequence is 2 -/
theorem common_ratio_is_two (a : ℕ → ℝ) (h : IncreasingGeometricSequence a) :
    ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 2 := by
  sorry

end NUMINAMATH_CALUDE_common_ratio_is_two_l2561_256166


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2561_256169

theorem absolute_value_inequality (x : ℝ) : 
  |2*x - 1| - x ≥ 2 ↔ x ≥ 3 ∨ x ≤ -1/3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2561_256169


namespace NUMINAMATH_CALUDE_mean_height_is_68_25_l2561_256133

def heights : List ℕ := [57, 59, 62, 64, 64, 65, 65, 68, 69, 70, 71, 73, 75, 75, 77, 78]

theorem mean_height_is_68_25 : 
  let total_height : ℕ := heights.sum
  let num_players : ℕ := heights.length
  (total_height : ℚ) / num_players = 68.25 := by
sorry

end NUMINAMATH_CALUDE_mean_height_is_68_25_l2561_256133


namespace NUMINAMATH_CALUDE_difference_of_fractions_difference_for_7000_l2561_256149

theorem difference_of_fractions (n : ℝ) : n * (1 / 10) - n * (1 / 1000) = n * (99 / 1000) :=
by sorry

theorem difference_for_7000 : 7000 * (1 / 10) - 7000 * (1 / 1000) = 693 :=
by sorry

end NUMINAMATH_CALUDE_difference_of_fractions_difference_for_7000_l2561_256149


namespace NUMINAMATH_CALUDE_carpet_shaded_area_l2561_256153

theorem carpet_shaded_area (S T : ℝ) : 
  12 / S = 4 →
  S / T = 4 →
  12 * (T * T) + S * S = 15.75 := by
sorry

end NUMINAMATH_CALUDE_carpet_shaded_area_l2561_256153


namespace NUMINAMATH_CALUDE_euclid_wrote_elements_l2561_256124

/-- The author of "Elements" -/
def author_of_elements : String := "Euclid"

/-- Theorem stating that Euclid is the author of "Elements" -/
theorem euclid_wrote_elements : author_of_elements = "Euclid" := by sorry

end NUMINAMATH_CALUDE_euclid_wrote_elements_l2561_256124


namespace NUMINAMATH_CALUDE_katy_brownies_theorem_l2561_256121

/-- The number of brownies Katy made -/
def total_brownies : ℕ := 15

/-- The number of brownies Katy ate on Monday -/
def monday_brownies : ℕ := 5

/-- The number of brownies Katy ate on Tuesday -/
def tuesday_brownies : ℕ := 2 * monday_brownies

theorem katy_brownies_theorem :
  total_brownies = monday_brownies + tuesday_brownies :=
by sorry

end NUMINAMATH_CALUDE_katy_brownies_theorem_l2561_256121


namespace NUMINAMATH_CALUDE_vector_difference_l2561_256168

/-- Given two vectors AB and AC in 2D space, prove that BC is their difference -/
theorem vector_difference (AB AC : ℝ × ℝ) (h1 : AB = (2, -1)) (h2 : AC = (-4, 1)) :
  AC - AB = (-6, 2) := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_l2561_256168


namespace NUMINAMATH_CALUDE_m_upper_bound_l2561_256161

/-- The function f(x) = a(x^2 + 1) -/
def f (a : ℝ) (x : ℝ) : ℝ := a * (x^2 + 1)

theorem m_upper_bound
  (h1 : ∀ (a : ℝ), a ∈ Set.Ioo (-4) (-2))
  (h2 : ∀ (x : ℝ), x ∈ Set.Icc 1 3)
  (h3 : ∀ (m : ℝ) (a : ℝ) (x : ℝ),
    a ∈ Set.Ioo (-4) (-2) → x ∈ Set.Icc 1 3 →
    m * a - f a x > a^2 + Real.log x) :
  ∀ (m : ℝ), m ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_m_upper_bound_l2561_256161


namespace NUMINAMATH_CALUDE_blue_line_length_is_correct_l2561_256167

/-- The length of the white line in inches -/
def white_line_length : ℝ := 7.67

/-- The difference in length between the white and blue lines in inches -/
def length_difference : ℝ := 4.33

/-- The length of the blue line in inches -/
def blue_line_length : ℝ := white_line_length - length_difference

theorem blue_line_length_is_correct : blue_line_length = 3.34 := by
  sorry

end NUMINAMATH_CALUDE_blue_line_length_is_correct_l2561_256167


namespace NUMINAMATH_CALUDE_common_roots_product_l2561_256117

-- Define the polynomials
def p (C : ℝ) (x : ℝ) : ℝ := x^3 + C*x^2 - 20
def q (D : ℝ) (x : ℝ) : ℝ := x^3 + D*x - 80

-- Define the theorem
theorem common_roots_product (C D : ℝ) :
  ∃ (r₁ r₂ : ℝ) (a b c : ℕ),
    (p C r₁ = 0 ∧ q D r₁ = 0) ∧
    (p C r₂ = 0 ∧ q D r₂ = 0) ∧
    r₁ ≠ r₂ ∧
    (r₁ * r₂ = a * (c ^ (1 / b : ℝ))) ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = 25 :=
  sorry

end NUMINAMATH_CALUDE_common_roots_product_l2561_256117


namespace NUMINAMATH_CALUDE_expression_values_l2561_256147

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := a / abs a + b / abs b + c / abs c + d / abs d + (a * b * c * d) / abs (a * b * c * d)
  expr = 5 ∨ expr = 1 ∨ expr = -3 ∨ expr = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l2561_256147


namespace NUMINAMATH_CALUDE_subtraction_result_l2561_256195

theorem subtraction_result : 2.43 - 1.2 = 1.23 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l2561_256195


namespace NUMINAMATH_CALUDE_fractional_simplification_l2561_256134

theorem fractional_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  2 / (x + 1) - (x + 5) / (x^2 - 1) = (x - 7) / ((x + 1) * (x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_fractional_simplification_l2561_256134


namespace NUMINAMATH_CALUDE_total_tiles_needed_l2561_256106

def room_length : ℕ := 12
def room_width : ℕ := 16
def small_tile_size : ℕ := 1
def large_tile_size : ℕ := 2

theorem total_tiles_needed : 
  (room_length * room_width - (room_length - 2 * small_tile_size) * (room_width - 2 * small_tile_size)) + 
  ((room_length - 2 * small_tile_size) * (room_width - 2 * small_tile_size) / (large_tile_size * large_tile_size)) = 87 := by
  sorry

end NUMINAMATH_CALUDE_total_tiles_needed_l2561_256106


namespace NUMINAMATH_CALUDE_daniel_initial_noodles_l2561_256183

/-- The number of noodles Daniel gave to William -/
def noodles_given : ℝ := 12.0

/-- The number of noodles Daniel had left -/
def noodles_left : ℕ := 42

/-- The initial number of noodles Daniel had -/
def initial_noodles : ℝ := noodles_given + noodles_left

theorem daniel_initial_noodles : initial_noodles = 54.0 := by sorry

end NUMINAMATH_CALUDE_daniel_initial_noodles_l2561_256183


namespace NUMINAMATH_CALUDE_polynomial_not_equal_77_l2561_256172

theorem polynomial_not_equal_77 (x y : ℤ) : 
  x^5 - 4*x^4*y - 5*y^2*x^3 + 20*y^3*x^2 + 4*y^4*x - 16*y^5 ≠ 77 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_not_equal_77_l2561_256172


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2561_256116

theorem system_of_equations_solution :
  ∀ x y : ℝ,
  (4 * x - 2) / (5 * x - 5) = 3 / 4 →
  x + y = 3 →
  x = -7 ∧ y = 10 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2561_256116


namespace NUMINAMATH_CALUDE_grapes_remainder_l2561_256164

theorem grapes_remainder (josiah katelyn liam basket_size : ℕ) 
  (h_josiah : josiah = 54)
  (h_katelyn : katelyn = 67)
  (h_liam : liam = 29)
  (h_basket : basket_size = 15) : 
  (josiah + katelyn + liam) % basket_size = 0 := by
sorry

end NUMINAMATH_CALUDE_grapes_remainder_l2561_256164


namespace NUMINAMATH_CALUDE_anne_weight_is_67_l2561_256109

/-- Douglas's weight in pounds -/
def douglas_weight : ℕ := 52

/-- The difference between Anne's and Douglas's weights in pounds -/
def weight_difference : ℕ := 15

/-- Anne's weight in pounds -/
def anne_weight : ℕ := douglas_weight + weight_difference

theorem anne_weight_is_67 : anne_weight = 67 := by
  sorry

end NUMINAMATH_CALUDE_anne_weight_is_67_l2561_256109


namespace NUMINAMATH_CALUDE_absolute_value_problem_l2561_256194

theorem absolute_value_problem (a b : ℝ) 
  (ha : |a| = 5) 
  (hb : |b| = 2) :
  (a > b → a + b = 7 ∨ a + b = 3) ∧
  (|a + b| = |a| - |b| → (a = -5 ∧ b = 2) ∨ (a = 5 ∧ b = -2)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_problem_l2561_256194


namespace NUMINAMATH_CALUDE_no_solution_in_naturals_l2561_256125

theorem no_solution_in_naturals :
  ¬ ∃ (x y z : ℕ), (2 * x) ^ (2 * x) - 1 = y ^ (z + 1) := by
sorry

end NUMINAMATH_CALUDE_no_solution_in_naturals_l2561_256125


namespace NUMINAMATH_CALUDE_company_capital_growth_l2561_256141

/-- Calculates the final capital after n years given initial capital, growth rate, and yearly consumption --/
def finalCapital (initialCapital : ℝ) (growthRate : ℝ) (yearlyConsumption : ℝ) (years : ℕ) : ℝ :=
  match years with
  | 0 => initialCapital
  | n + 1 => (finalCapital initialCapital growthRate yearlyConsumption n * (1 + growthRate)) - yearlyConsumption

/-- The problem statement --/
theorem company_capital_growth (x : ℝ) : 
  finalCapital 1 0.5 x 3 = 2.9 ↔ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_company_capital_growth_l2561_256141


namespace NUMINAMATH_CALUDE_percentage_increase_l2561_256156

/-- Given two positive real numbers a and b with a ratio of 4:5, 
    and x and m derived from a and b respectively, 
    prove that the percentage increase from a to x is 25% --/
theorem percentage_increase (a b x m : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  a / b = 4 / 5 →
  ∃ p, x = a * (1 + p / 100) →
  m = b * 0.6 →
  m / x = 0.6 →
  p = 25 := by
sorry


end NUMINAMATH_CALUDE_percentage_increase_l2561_256156


namespace NUMINAMATH_CALUDE_pencils_calculation_l2561_256146

/-- Given a setup of pencils and crayons in rows, calculates the number of pencils per row. -/
def pencils_per_row (total_items : ℕ) (rows : ℕ) (crayons_per_row : ℕ) : ℕ :=
  (total_items - rows * crayons_per_row) / rows

theorem pencils_calculation :
  pencils_per_row 638 11 27 = 31 := by
  sorry

end NUMINAMATH_CALUDE_pencils_calculation_l2561_256146


namespace NUMINAMATH_CALUDE_equation_solution_l2561_256197

theorem equation_solution :
  ∃! r : ℚ, (r + 4) / (r - 3) = (r - 2) / (r + 2) ∧ r = -2/11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2561_256197


namespace NUMINAMATH_CALUDE_min_value_problem_l2561_256132

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log 2 * x + Real.log 8 * y = Real.log 2) : 
  1/x + 1/(3*y) ≥ 4 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    Real.log 2 * x₀ + Real.log 8 * y₀ = Real.log 2 ∧ 1/x₀ + 1/(3*y₀) = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l2561_256132


namespace NUMINAMATH_CALUDE_monster_hunt_sum_l2561_256179

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem monster_hunt_sum :
  geometric_sum 2 2 5 = 62 :=
sorry

end NUMINAMATH_CALUDE_monster_hunt_sum_l2561_256179


namespace NUMINAMATH_CALUDE_number_equation_l2561_256112

theorem number_equation (x : ℝ) : 3 * x - 1 = 2 * x ↔ x = 1 := by sorry

end NUMINAMATH_CALUDE_number_equation_l2561_256112


namespace NUMINAMATH_CALUDE_v_2008_value_l2561_256148

-- Define the sequence v_n
def v : ℕ → ℕ 
| n => sorry  -- The exact definition would be complex to write out

-- Define the function g(n) for the last term in a group with n terms
def g (n : ℕ) : ℕ := 2 * n^2 - 3 * n + 2

-- Define the function for the total number of terms up to and including group n
def totalTerms (n : ℕ) : ℕ := n * (n + 1) / 2

-- The theorem to prove
theorem v_2008_value : v 2008 = 7618 := by sorry

end NUMINAMATH_CALUDE_v_2008_value_l2561_256148


namespace NUMINAMATH_CALUDE_sale_discount_theorem_l2561_256105

/-- Proves that applying a 50% discount followed by a 30% discount results in a 65% total discount -/
theorem sale_discount_theorem (original_price : ℝ) (sale_price coupon_price final_price : ℝ) :
  sale_price = 0.5 * original_price →
  coupon_price = 0.7 * sale_price →
  final_price = coupon_price →
  final_price = 0.35 * original_price ∧ (1 - 0.35) * 100 = 65 := by
  sorry

#check sale_discount_theorem

end NUMINAMATH_CALUDE_sale_discount_theorem_l2561_256105


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l2561_256163

/-- An ellipse with given major axis length and eccentricity -/
structure Ellipse where
  major_axis : ℝ
  eccentricity : ℝ

/-- The standard equation of an ellipse -/
inductive StandardEllipseEquation
  | horizontal : StandardEllipseEquation
  | vertical : StandardEllipseEquation

/-- Check if a given equation is the standard equation of the ellipse -/
def is_standard_equation (e : Ellipse) (eq : StandardEllipseEquation) : Prop :=
  match eq with
  | StandardEllipseEquation.horizontal => 
      ∃ (x y : ℝ), x^2 / (e.major_axis/2)^2 + y^2 / ((e.major_axis/2)^2 * (1 - e.eccentricity^2)) = 1
  | StandardEllipseEquation.vertical => 
      ∃ (x y : ℝ), x^2 / ((e.major_axis/2)^2 * (1 - e.eccentricity^2)) + y^2 / (e.major_axis/2)^2 = 1

theorem ellipse_standard_equation (e : Ellipse) 
    (h1 : e.major_axis = 10)
    (h2 : e.eccentricity = 4/5) :
    (is_standard_equation e StandardEllipseEquation.horizontal ∨ 
     is_standard_equation e StandardEllipseEquation.vertical) :=
  sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l2561_256163


namespace NUMINAMATH_CALUDE_smallest_possible_a_l2561_256171

theorem smallest_possible_a (P : ℤ → ℤ) (a : ℕ) (h_a_pos : a > 0) 
  (h_poly : ∀ x : ℤ, ∃ k : ℤ, P x = k)
  (h_odd : P 1 = a ∧ P 3 = a ∧ P 5 = a ∧ P 7 = a)
  (h_even : P 2 = -a ∧ P 4 = -a ∧ P 6 = -a ∧ P 8 = -a ∧ P 10 = -a) :
  945 ≤ a ∧ ∃ Q : ℤ → ℤ, 
    (∀ x : ℤ, ∃ k : ℤ, Q x = k) ∧
    (Q 2 = 126 ∧ Q 4 = -210 ∧ Q 6 = 126 ∧ Q 8 = -18 ∧ Q 10 = 126) ∧
    (∀ x : ℤ, P x - a = (x-1)*(x-3)*(x-5)*(x-7)*(Q x)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_possible_a_l2561_256171


namespace NUMINAMATH_CALUDE_perfect_square_divisibility_l2561_256192

theorem perfect_square_divisibility (a b : ℕ+) 
  (h : (2 * a * b) ∣ (a^2 + b^2 - a)) : 
  ∃ k : ℕ+, a = k^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_divisibility_l2561_256192


namespace NUMINAMATH_CALUDE_total_cost_is_60_l2561_256173

/-- The cost of a set of school supplies -/
structure SchoolSupplies where
  notebook : ℕ
  pen : ℕ
  ruler : ℕ
  pencil : ℕ

/-- The conditions given in the problem -/
structure Conditions where
  supplies : SchoolSupplies
  notebook_pencil_ruler_cost : supplies.notebook + supplies.pencil + supplies.ruler = 47
  notebook_ruler_pen_cost : supplies.notebook + supplies.ruler + supplies.pen = 58
  pen_pencil_cost : supplies.pen + supplies.pencil = 15

/-- The theorem to be proved -/
theorem total_cost_is_60 (c : Conditions) : 
  c.supplies.notebook + c.supplies.pen + c.supplies.ruler + c.supplies.pencil = 60 := by
  sorry

#check total_cost_is_60

end NUMINAMATH_CALUDE_total_cost_is_60_l2561_256173


namespace NUMINAMATH_CALUDE_num_non_mult_6_divisors_l2561_256107

/-- The smallest integer satisfying the given conditions -/
def m : ℕ :=
  2^3 * 3^4 * 5^6

/-- m/2 is a perfect square -/
axiom m_div_2_is_square : ∃ k : ℕ, m / 2 = k^2

/-- m/3 is a perfect cube -/
axiom m_div_3_is_cube : ∃ k : ℕ, m / 3 = k^3

/-- m/5 is a perfect fifth -/
axiom m_div_5_is_fifth : ∃ k : ℕ, m / 5 = k^5

/-- The number of divisors of m -/
def num_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- The number of divisors of m that are multiples of 6 -/
def num_divisors_mult_6 (n : ℕ) : ℕ :=
  (Finset.filter (λ x => x ∣ n ∧ 6 ∣ x) (Finset.range (n + 1))).card

/-- The main theorem -/
theorem num_non_mult_6_divisors :
    num_divisors m - num_divisors_mult_6 m = 56 := by
  sorry

end NUMINAMATH_CALUDE_num_non_mult_6_divisors_l2561_256107


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_zero_l2561_256185

theorem ceiling_floor_sum_zero : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_zero_l2561_256185


namespace NUMINAMATH_CALUDE_negation_of_existence_inequality_l2561_256142

theorem negation_of_existence_inequality (p : Prop) :
  (¬ (∃ x : ℝ, Real.exp x - x - 1 ≤ 0)) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_inequality_l2561_256142


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2561_256150

theorem tan_alpha_value (α : Real) : 
  Real.tan (π / 4 + α) = 1 / 2 → Real.tan α = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2561_256150


namespace NUMINAMATH_CALUDE_sphere_volume_increase_l2561_256187

theorem sphere_volume_increase (r₁ r₂ V₁ V₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ = 2 * r₁) 
  (h₃ : V₁ = (4/3) * π * r₁^3) (h₄ : V₂ = (4/3) * π * r₂^3) : V₂ = 8 * V₁ := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_increase_l2561_256187


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2561_256184

theorem polynomial_expansion :
  ∀ x : ℝ, (3 * x^2 + 4 * x - 5) * (4 * x^3 - 3 * x + 2) = 
    12 * x^5 + 16 * x^4 - 24 * x^3 - 6 * x^2 + 17 * x - 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2561_256184


namespace NUMINAMATH_CALUDE_equal_roots_count_l2561_256165

/-- The number of real values of p for which the quadratic equation
    x^2 - (p+1)x + (p+1)^2 = 0 has equal roots is exactly one. -/
theorem equal_roots_count : ∃! p : ℝ,
  let a : ℝ := 1
  let b : ℝ := -(p + 1)
  let c : ℝ := (p + 1)^2
  b^2 - 4*a*c = 0 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_count_l2561_256165


namespace NUMINAMATH_CALUDE_complement_A_in_U_equals_open_interval_l2561_256143

-- Define the set U
def U : Set ℝ := {x | (x - 2) / x ≤ 1}

-- Define the set A
def A : Set ℝ := {x | 2 - x ≤ 1}

-- Define the complement of A in U
def complement_A_in_U : Set ℝ := U \ A

-- Theorem statement
theorem complement_A_in_U_equals_open_interval :
  complement_A_in_U = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_equals_open_interval_l2561_256143


namespace NUMINAMATH_CALUDE_find_q_l2561_256138

theorem find_q (p q : ℝ) (h1 : p > 1) (h2 : q > 1) (h3 : 1/p + 1/q = 3/2) (h4 : p*q = 9) : q = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_q_l2561_256138


namespace NUMINAMATH_CALUDE_river_trip_longer_than_lake_l2561_256160

/-- Proves that a round trip on a river takes longer than traveling the same distance on a lake -/
theorem river_trip_longer_than_lake (v w : ℝ) (h : v > w) (h_pos : v > 0) :
  (20 * v) / (v^2 - w^2) > 20 / v := by
  sorry

end NUMINAMATH_CALUDE_river_trip_longer_than_lake_l2561_256160


namespace NUMINAMATH_CALUDE_corner_sum_is_164_l2561_256182

/-- Represents a square grid with side length n -/
structure Grid (n : ℕ) where
  size : ℕ
  size_eq : size = n * n

/-- The value of a cell in the grid given its row and column -/
def cellValue (g : Grid 9) (row col : ℕ) : ℕ :=
  (row - 1) * 9 + col

/-- The sum of the corner values in a 9x9 grid -/
def cornerSum (g : Grid 9) : ℕ :=
  cellValue g 1 1 + cellValue g 1 9 + cellValue g 9 1 + cellValue g 9 9

/-- Theorem: The sum of the corner values in a 9x9 grid is 164 -/
theorem corner_sum_is_164 (g : Grid 9) : cornerSum g = 164 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_is_164_l2561_256182


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_perpendicular_l2561_256126

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_perpendicular 
  (m n : Line) (α : Plane) :
  parallel m n → perpendicular n α → perpendicular m α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_perpendicular_l2561_256126


namespace NUMINAMATH_CALUDE_production_decrease_l2561_256188

/-- The number of cars originally planned for production -/
def original_plan : ℕ := 200

/-- The number of doors per car -/
def doors_per_car : ℕ := 5

/-- The total number of doors produced after reductions -/
def total_doors : ℕ := 375

/-- The reduction factor due to pandemic -/
def pandemic_reduction : ℚ := 1/2

theorem production_decrease (x : ℕ) : 
  (pandemic_reduction * (original_plan - x : ℚ)) * doors_per_car = total_doors → 
  x = 50 := by sorry

end NUMINAMATH_CALUDE_production_decrease_l2561_256188


namespace NUMINAMATH_CALUDE_sin_double_angle_l2561_256159

theorem sin_double_angle (x : Real) (h : Real.sin (x - π/4) = 2/3) : 
  Real.sin (2*x) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_l2561_256159


namespace NUMINAMATH_CALUDE_equation_solution_l2561_256108

theorem equation_solution (x : ℚ) : 5 * x + 3 = 2 * x - 4 → 3 * (x^2 + 6) = 103 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2561_256108


namespace NUMINAMATH_CALUDE_linear_function_condition_l2561_256128

/-- Given a linear function f(x) = ax - x - a where a > 0 and a ≠ 1, 
    prove that a > 1. -/
theorem linear_function_condition (a : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) : a > 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_condition_l2561_256128


namespace NUMINAMATH_CALUDE_savings_multiple_l2561_256158

theorem savings_multiple (monthly_pay : ℝ) (savings_fraction : ℝ) : 
  savings_fraction = 0.29411764705882354 →
  monthly_pay > 0 →
  let monthly_savings := monthly_pay * savings_fraction
  let monthly_non_savings := monthly_pay - monthly_savings
  let total_savings := monthly_savings * 12
  total_savings = 5 * monthly_non_savings :=
by sorry

end NUMINAMATH_CALUDE_savings_multiple_l2561_256158


namespace NUMINAMATH_CALUDE_corn_area_theorem_l2561_256110

/-- Represents the farmer's land allocation --/
structure FarmLand where
  total : ℝ
  cleared_percentage : ℝ
  potato_percentage : ℝ
  tomato_percentage : ℝ

/-- Calculates the area of land planted with corn --/
def corn_area (farm : FarmLand) : ℝ :=
  let cleared := farm.total * farm.cleared_percentage
  let potato := cleared * farm.potato_percentage
  let tomato := cleared * farm.tomato_percentage
  cleared - (potato + tomato)

/-- Theorem stating that the corn area is approximately 630 acres --/
theorem corn_area_theorem (farm : FarmLand) 
  (h1 : farm.total = 6999.999999999999)
  (h2 : farm.cleared_percentage = 0.90)
  (h3 : farm.potato_percentage = 0.20)
  (h4 : farm.tomato_percentage = 0.70) :
  ∃ ε > 0, |corn_area farm - 630| < ε :=
sorry

end NUMINAMATH_CALUDE_corn_area_theorem_l2561_256110


namespace NUMINAMATH_CALUDE_white_tree_count_l2561_256113

/-- Represents the number of crepe myrtle trees of each color in the park -/
structure TreeCount where
  total : ℕ
  pink : ℕ
  red : ℕ
  white : ℕ

/-- The conditions of the park's tree distribution -/
def park_conditions (t : TreeCount) : Prop :=
  t.total = 42 ∧
  t.pink = t.total / 3 ∧
  t.red = 2 ∧
  t.white = t.total - t.pink - t.red ∧
  t.white > t.pink ∧ t.white > t.red

/-- Theorem stating that under the given conditions, the number of white trees is 26 -/
theorem white_tree_count (t : TreeCount) (h : park_conditions t) : t.white = 26 := by
  sorry

end NUMINAMATH_CALUDE_white_tree_count_l2561_256113


namespace NUMINAMATH_CALUDE_correct_average_l2561_256127

theorem correct_average (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℕ) :
  n = 10 →
  initial_avg = 16 →
  incorrect_num = 25 →
  correct_num = 45 →
  (n : ℚ) * initial_avg + (correct_num - incorrect_num : ℚ) = n * 18 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l2561_256127


namespace NUMINAMATH_CALUDE_horner_method_V_4_l2561_256199

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial coefficients in descending order of degree -/
def f_coeffs : List ℤ := [3, 5, 6, 79, -8, 35, 12]

/-- The x-value at which to evaluate the polynomial -/
def x_val : ℤ := -4

/-- V_4 in Horner's method is the 5th intermediate value (0-indexed) -/
def V_4 : ℤ := (horner_eval (f_coeffs.take 5) x_val) * x_val + f_coeffs[5]

theorem horner_method_V_4 :
  V_4 = 220 :=
sorry

end NUMINAMATH_CALUDE_horner_method_V_4_l2561_256199


namespace NUMINAMATH_CALUDE_triangle_equilateral_condition_l2561_256102

/-- Triangle ABC with angles A, B, C and sides a, b, c -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : A + B + C = Real.pi
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- A triangle is equilateral if all its sides are equal -/
def Triangle.isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- The theorem stating the conditions and conclusion about the triangle -/
theorem triangle_equilateral_condition (t : Triangle)
    (h1 : t.B = (t.A + t.C) / 2)  -- B is arithmetic mean of A and C
    (h2 : t.b ^ 2 = t.a * t.c)    -- b is geometric mean of a and c
    : t.isEquilateral := by
  sorry

end NUMINAMATH_CALUDE_triangle_equilateral_condition_l2561_256102


namespace NUMINAMATH_CALUDE_millionthDigitOf1Over41_l2561_256137

-- Define the fraction
def fraction : ℚ := 1 / 41

-- Define the function to get the nth digit after the decimal point
noncomputable def nthDigitAfterDecimal (q : ℚ) (n : ℕ) : ℕ := sorry

-- State the theorem
theorem millionthDigitOf1Over41 : 
  nthDigitAfterDecimal fraction 1000000 = 9 := by sorry

end NUMINAMATH_CALUDE_millionthDigitOf1Over41_l2561_256137


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2561_256151

/-- Parabola defined by y = x^2 -/
def parabola (x y : ℝ) : Prop := y = x^2

/-- Point Q -/
def Q : ℝ × ℝ := (10, 25)

/-- Line passing through Q with slope m -/
def line (m x y : ℝ) : Prop := y - Q.2 = m * (x - Q.1)

/-- Line does not intersect parabola -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x y : ℝ, ¬(parabola x y ∧ line m x y)

/-- Theorem statement -/
theorem parabola_line_intersection :
  ∃ r s : ℝ, (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) ∧ r + s = 40 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2561_256151


namespace NUMINAMATH_CALUDE_strategy_is_injective_l2561_256196

-- Define the set of possible numbers
inductive Number : Type
| one : Number
| two : Number
| three : Number

-- Define the set of possible answers
inductive Answer : Type
| yes : Answer
| no : Answer
| dontKnow : Answer

-- Define the strategy function
def strategy : Number → Answer
| Number.one => Answer.yes
| Number.two => Answer.dontKnow
| Number.three => Answer.no

-- Theorem: The strategy function is injective
theorem strategy_is_injective :
  ∀ x y : Number, x ≠ y → strategy x ≠ strategy y := by
  sorry

#check strategy_is_injective

end NUMINAMATH_CALUDE_strategy_is_injective_l2561_256196


namespace NUMINAMATH_CALUDE_max_colors_upper_bound_l2561_256162

/-- 
Given a positive integer n ≥ 2, an n × n × n cube is divided into n³ unit cubes, 
each colored with one color. For each n × n × 1 rectangular prism (in 3 orientations), 
consider the set of colors appearing in this prism. For any color set in one group, 
it also appears in each of the other two groups. 
This theorem states the upper bound for the maximum number of colors.
-/
theorem max_colors_upper_bound (n : ℕ) (h : n ≥ 2) : 
  ∃ C : ℕ, C ≤ n * (n + 1) * (2 * n + 1) / 6 ∧ 
  (∀ D : ℕ, D ≤ n * (n + 1) * (2 * n + 1) / 6) :=
by sorry

end NUMINAMATH_CALUDE_max_colors_upper_bound_l2561_256162


namespace NUMINAMATH_CALUDE_equal_sets_implies_b_minus_a_equals_one_l2561_256135

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {1, 0, a}
def B (a b : ℝ) : Set ℝ := {1/a, |a|, b/a}

-- State the theorem
theorem equal_sets_implies_b_minus_a_equals_one (a b : ℝ) :
  A a = B a b → b - a = 1 := by
  sorry

end NUMINAMATH_CALUDE_equal_sets_implies_b_minus_a_equals_one_l2561_256135


namespace NUMINAMATH_CALUDE_sprinkler_water_usage_5_days_l2561_256175

/-- A sprinkler system for a desert garden -/
structure SprinklerSystem where
  morning_usage : ℕ  -- Water usage in the morning in liters
  evening_usage : ℕ  -- Water usage in the evening in liters

/-- Calculates the total water usage for a given number of days -/
def total_water_usage (s : SprinklerSystem) (days : ℕ) : ℕ :=
  (s.morning_usage + s.evening_usage) * days

/-- Theorem: The sprinkler system uses 50 liters of water in 5 days -/
theorem sprinkler_water_usage_5_days :
  ∃ (s : SprinklerSystem), s.morning_usage = 4 ∧ s.evening_usage = 6 ∧ total_water_usage s 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_sprinkler_water_usage_5_days_l2561_256175


namespace NUMINAMATH_CALUDE_employed_males_percentage_l2561_256178

/-- The percentage of employed people in the population -/
def employed_percentage : ℝ := 64

/-- The percentage of employed people who are female -/
def female_employed_percentage : ℝ := 25

/-- The theorem stating the percentage of the population that are employed males -/
theorem employed_males_percentage :
  (employed_percentage / 100) * (1 - female_employed_percentage / 100) * 100 = 48 := by
  sorry

end NUMINAMATH_CALUDE_employed_males_percentage_l2561_256178


namespace NUMINAMATH_CALUDE_not_equal_to_seven_fifths_l2561_256193

theorem not_equal_to_seven_fifths : ∃ x : ℚ, x ≠ 7/5 ∧
  (x = 1 + 3/8) ∧
  (14/10 = 7/5) ∧
  (1 + 2/5 = 7/5) ∧
  (1 + 6/15 = 7/5) ∧
  (1 + 28/20 = 7/5) :=
by
  sorry

end NUMINAMATH_CALUDE_not_equal_to_seven_fifths_l2561_256193


namespace NUMINAMATH_CALUDE_collinear_vectors_l2561_256174

/-- Given vectors a, b, and c in ℝ², prove that if k*a + b is collinear with c,
    then k = -26/15 -/
theorem collinear_vectors (a b c : ℝ × ℝ) (k : ℝ) 
    (ha : a = (1, 2))
    (hb : b = (2, 3))
    (hc : c = (4, -7))
    (hcollinear : ∃ t : ℝ, t ≠ 0 ∧ k • a + b = t • c) :
    k = -26/15 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_l2561_256174


namespace NUMINAMATH_CALUDE_monochromatic_four_cycle_in_K_5_5_l2561_256104

/-- A complete bipartite graph K_{n,n} --/
def CompleteBipartiteGraph (n : ℕ) := Unit

/-- A coloring of the edges of a graph with two colors --/
def Coloring (G : Type) := G → Bool

/-- A 4-cycle in a graph --/
def FourCycle (G : Type) := List G

/-- Check if a 4-cycle is monochromatic under a given coloring --/
def isMonochromatic (c : Coloring G) (cycle : FourCycle G) : Prop := sorry

/-- The main theorem --/
theorem monochromatic_four_cycle_in_K_5_5 :
  ∀ (c : Coloring (CompleteBipartiteGraph 5)),
  ∃ (cycle : FourCycle (CompleteBipartiteGraph 5)),
  isMonochromatic c cycle :=
sorry

end NUMINAMATH_CALUDE_monochromatic_four_cycle_in_K_5_5_l2561_256104


namespace NUMINAMATH_CALUDE_fraction_division_equals_three_l2561_256136

theorem fraction_division_equals_three : 
  (-1/6 + 3/8 - 1/12) / (1/24) = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_equals_three_l2561_256136


namespace NUMINAMATH_CALUDE_closest_vertex_of_dilated_square_l2561_256155

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  center : Point
  area : ℝ
  verticalSide : Bool

/-- Dilates a point from the origin by a given factor -/
def dilatePoint (p : Point) (factor : ℝ) : Point :=
  { x := p.x * factor, y := p.y * factor }

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Finds the vertex of a dilated square closest to the origin -/
def closestVertexToDilatedSquare (s : Square) (dilationFactor : ℝ) : Point :=
  sorry

theorem closest_vertex_of_dilated_square :
  let originalSquare : Square := {
    center := { x := 5, y := -3 },
    area := 16,
    verticalSide := true
  }
  let dilationFactor : ℝ := 3
  let closestVertex := closestVertexToDilatedSquare originalSquare dilationFactor
  closestVertex.x = 9 ∧ closestVertex.y = -3 := by
  sorry

end NUMINAMATH_CALUDE_closest_vertex_of_dilated_square_l2561_256155


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l2561_256144

/-- The quadratic equation (k-1)x^2 + 2x - 2 = 0 has two distinct real roots if and only if k > 1/2 and k ≠ 1 -/
theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (k - 1) * x^2 + 2 * x - 2 = 0 ∧ (k - 1) * y^2 + 2 * y - 2 = 0) ↔ 
  (k > 1/2 ∧ k ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l2561_256144


namespace NUMINAMATH_CALUDE_linear_function_properties_l2561_256101

theorem linear_function_properties (m k b : ℝ) (h1 : m > 1) 
  (h2 : k * m + b = 1) (h3 : -k + b = m) : k < 0 ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_properties_l2561_256101


namespace NUMINAMATH_CALUDE_binomial_sum_divides_power_of_two_l2561_256122

theorem binomial_sum_divides_power_of_two (n : ℕ) :
  n > 3 →
  (1 + Nat.choose n 1 + Nat.choose n 2 + Nat.choose n 3) ∣ 2^2000 ↔
  n = 7 ∨ n = 23 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_divides_power_of_two_l2561_256122


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l2561_256191

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_formula (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 = 2 →
  (∀ n : ℕ, a (n + 2)^2 + 4 * a n^2 = 4 * a (n + 1)^2) →
  ∀ n : ℕ, a n = 2^((n + 1) / 2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l2561_256191


namespace NUMINAMATH_CALUDE_divisors_of_500_l2561_256180

theorem divisors_of_500 : 
  ∃ (S : Finset Nat), 
    (∀ n ∈ S, n ∣ 500 ∧ 1 ≤ n ∧ n ≤ 500) ∧ 
    (∀ n, n ∣ 500 ∧ 1 ≤ n ∧ n ≤ 500 → n ∈ S) ∧ 
    Finset.card S = 12 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_500_l2561_256180


namespace NUMINAMATH_CALUDE_max_value_implies_a_l2561_256157

def f (a : ℝ) (x : ℝ) : ℝ := -4 * x^2 + 4 * a * x - 4 * a - a^2

theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≤ -5) ∧
  (∃ x ∈ Set.Icc 0 1, f a x = -5) →
  a = 5/4 ∨ a = -5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l2561_256157


namespace NUMINAMATH_CALUDE_farm_animals_difference_l2561_256186

theorem farm_animals_difference (initial_horses : ℕ) (initial_cows : ℕ) : 
  initial_horses = 6 * initial_cows →
  (initial_horses - 30) = 4 * (initial_cows + 30) →
  (initial_horses - 30) - (initial_cows + 30) = 315 := by
sorry

end NUMINAMATH_CALUDE_farm_animals_difference_l2561_256186


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2561_256118

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ),
    x^6 + x^3 + x^3 * y + y = 147^157 ∧
    x^3 + x^3 * y + y^2 + y + z^9 = 157^1177 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2561_256118


namespace NUMINAMATH_CALUDE_books_per_shelf_l2561_256170

def library1_total : ℕ := 24850
def library2_total : ℕ := 55300
def library1_leftover : ℕ := 154
def library2_leftover : ℕ := 175

theorem books_per_shelf :
  Int.gcd (library1_total - library1_leftover) (library2_total - library2_leftover) = 441 :=
by sorry

end NUMINAMATH_CALUDE_books_per_shelf_l2561_256170


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2561_256114

/-- A function satisfying the given functional equation. -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f ((x - y)^2) = x^2 - 2*y*(f x) + (f y)^2

/-- The main theorem stating that functions satisfying the equation are either
    the identity function or the identity function plus one. -/
theorem functional_equation_solution (f : ℝ → ℝ) (hf : SatisfiesEquation f) :
  (∀ x, f x = x) ∨ (∀ x, f x = x + 1) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2561_256114


namespace NUMINAMATH_CALUDE_tree_height_after_three_years_l2561_256111

def tree_height (initial_height : ℝ) (growth_factor : ℝ) (years : ℕ) : ℝ :=
  initial_height * (growth_factor ^ years)

theorem tree_height_after_three_years 
  (initial_height : ℝ)
  (growth_factor : ℝ)
  (h1 : initial_height = 1)
  (h2 : growth_factor = 3)
  (h3 : tree_height initial_height growth_factor 5 = 243) :
  tree_height initial_height growth_factor 3 = 27 := by
sorry

end NUMINAMATH_CALUDE_tree_height_after_three_years_l2561_256111


namespace NUMINAMATH_CALUDE_calculate_expression_l2561_256140

theorem calculate_expression (a b : ℝ) (hb : b ≠ 0) :
  4 * a * (3 * a^2 * b) / (2 * a * b) = 6 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2561_256140


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l2561_256123

theorem no_real_roots_quadratic (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + (a - 5) * x + 2 ≠ 0) → 1 < a ∧ a < 9 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l2561_256123


namespace NUMINAMATH_CALUDE_max_at_one_implies_c_equals_three_l2561_256119

/-- The function f(x) defined as x(x-c)² --/
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

/-- The derivative of f(x) with respect to x --/
def f' (c : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*c*x + c^2

theorem max_at_one_implies_c_equals_three (c : ℝ) :
  (∀ x : ℝ, f c x ≤ f c 1) →
  (f' c 1 = 0) →
  c = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_at_one_implies_c_equals_three_l2561_256119


namespace NUMINAMATH_CALUDE_no_solutions_for_exponential_equations_l2561_256189

theorem no_solutions_for_exponential_equations :
  (∀ n : ℕ, n > 1 → ¬∃ (p m : ℕ), Nat.Prime p ∧ Odd p ∧ m > 0 ∧ p^n + 1 = 2^m) ∧
  (∀ n : ℕ, n > 2 → ¬∃ (p m : ℕ), Nat.Prime p ∧ Odd p ∧ m > 0 ∧ p^n - 1 = 2^m) := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_exponential_equations_l2561_256189


namespace NUMINAMATH_CALUDE_min_green_beads_exact_min_green_beads_l2561_256181

/-- Represents a necklace with red, blue, and green beads. -/
structure Necklace where
  total : Nat
  red : Nat
  blue : Nat
  green : Nat
  sum_eq_total : red + blue + green = total
  red_between_blue : red ≥ blue
  green_between_red : green ≥ red

/-- The minimum number of green beads in a necklace of 80 beads satisfying the given conditions. -/
theorem min_green_beads (n : Necklace) (h : n.total = 80) : n.green ≥ 27 := by
  sorry

/-- The minimum number of green beads is exactly 27. -/
theorem exact_min_green_beads : ∃ n : Necklace, n.total = 80 ∧ n.green = 27 := by
  sorry

end NUMINAMATH_CALUDE_min_green_beads_exact_min_green_beads_l2561_256181
