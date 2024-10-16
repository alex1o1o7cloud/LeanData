import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l546_54667

theorem quadratic_roots_properties (x₁ x₂ : ℝ) 
  (h : x₁^2 - x₁ - 1 = 0 ∧ x₂^2 - x₂ - 1 = 0) : 
  x₁ + x₂ = 1 ∧ x₁ * x₂ = -1 ∧ x₁^2 + x₂^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l546_54667


namespace NUMINAMATH_CALUDE_third_side_length_l546_54614

-- Define a right-angled triangle with side lengths a, b, and x
structure RightTriangle where
  a : ℝ
  b : ℝ
  x : ℝ
  is_right : a^2 + b^2 = x^2 ∨ a^2 + x^2 = b^2 ∨ x^2 + b^2 = a^2

-- Define the theorem
theorem third_side_length (t : RightTriangle) :
  (t.a - 3)^2 + |t.b - 4| = 0 → t.x = 5 ∨ t.x = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_third_side_length_l546_54614


namespace NUMINAMATH_CALUDE_b_share_is_180_l546_54662

/-- Represents the rental arrangement for a pasture -/
structure PastureRental where
  total_rent : ℚ
  a_horses : ℕ
  a_months : ℕ
  b_horses : ℕ
  b_months : ℕ
  c_horses : ℕ
  c_months : ℕ

/-- Calculates the share of rent for person b given a PastureRental arrangement -/
def calculate_b_share (rental : PastureRental) : ℚ :=
  let total_horse_months := rental.a_horses * rental.a_months +
                            rental.b_horses * rental.b_months +
                            rental.c_horses * rental.c_months
  let cost_per_horse_month := rental.total_rent / total_horse_months
  (rental.b_horses * rental.b_months : ℚ) * cost_per_horse_month

/-- Theorem stating that b's share of the rent is 180 for the given arrangement -/
theorem b_share_is_180 (rental : PastureRental)
  (h1 : rental.total_rent = 435)
  (h2 : rental.a_horses = 12) (h3 : rental.a_months = 8)
  (h4 : rental.b_horses = 16) (h5 : rental.b_months = 9)
  (h6 : rental.c_horses = 18) (h7 : rental.c_months = 6) :
  calculate_b_share rental = 180 := by
  sorry

end NUMINAMATH_CALUDE_b_share_is_180_l546_54662


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l546_54618

/-- The quadratic polynomial that satisfies the given conditions -/
def q (x : ℚ) : ℚ := (6/5) * x^2 - (4/5) * x + 8/5

/-- Theorem stating that q(x) satisfies the required conditions -/
theorem q_satisfies_conditions :
  q (-2) = 8 ∧ q 1 = 2 ∧ q 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l546_54618


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l546_54697

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ
  altitude : ℝ
  perimeter : ℝ
  base_to_side_ratio : ℚ
  is_isosceles : base ≠ side
  altitude_value : altitude = 10
  perimeter_value : perimeter = 40
  ratio_value : base_to_side_ratio = 2 / 3

/-- The area of an isosceles triangle with the given properties is 80 -/
theorem isosceles_triangle_area (t : IsoscelesTriangle) : t.base * t.altitude / 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l546_54697


namespace NUMINAMATH_CALUDE_inequality_proof_l546_54606

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hsum : a + b + c ≤ 3) : 
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l546_54606


namespace NUMINAMATH_CALUDE_apples_picked_l546_54602

theorem apples_picked (initial_apples new_apples final_apples : ℕ) 
  (h1 : initial_apples = 11)
  (h2 : new_apples = 2)
  (h3 : final_apples = 6) :
  initial_apples - (initial_apples - new_apples - final_apples) = 7 := by
  sorry

end NUMINAMATH_CALUDE_apples_picked_l546_54602


namespace NUMINAMATH_CALUDE_complex_equation_solution_l546_54647

/-- Given a complex number z and a real number a, if |z| = 2 and (z - a)² = a, then a = 2 -/
theorem complex_equation_solution (z : ℂ) (a : ℝ) 
  (h1 : Complex.abs z = 2) 
  (h2 : (z - a)^2 = a) : 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l546_54647


namespace NUMINAMATH_CALUDE_retailer_profit_percentage_l546_54649

/-- Calculates the overall profit percentage for a retailer selling three items -/
theorem retailer_profit_percentage
  (radio_purchase : ℝ) (tv_purchase : ℝ) (speaker_purchase : ℝ)
  (radio_overhead : ℝ) (tv_overhead : ℝ) (speaker_overhead : ℝ)
  (radio_selling : ℝ) (tv_selling : ℝ) (speaker_selling : ℝ)
  (h1 : radio_purchase = 225)
  (h2 : tv_purchase = 4500)
  (h3 : speaker_purchase = 1500)
  (h4 : radio_overhead = 30)
  (h5 : tv_overhead = 200)
  (h6 : speaker_overhead = 100)
  (h7 : radio_selling = 300)
  (h8 : tv_selling = 5400)
  (h9 : speaker_selling = 1800) :
  let total_cp := radio_purchase + tv_purchase + speaker_purchase +
                  radio_overhead + tv_overhead + speaker_overhead
  let total_sp := radio_selling + tv_selling + speaker_selling
  let profit := total_sp - total_cp
  let profit_percentage := (profit / total_cp) * 100
  abs (profit_percentage - 14.42) < 0.01 := by
sorry


end NUMINAMATH_CALUDE_retailer_profit_percentage_l546_54649


namespace NUMINAMATH_CALUDE_square_side_difference_l546_54646

/-- Given four squares with side lengths s₁ ≥ s₂ ≥ s₃ ≥ s₄, prove that s₁ - s₄ = 29 -/
theorem square_side_difference (s₁ s₂ s₃ s₄ : ℝ) 
  (h₁ : s₁ ≥ s₂) (h₂ : s₂ ≥ s₃) (h₃ : s₃ ≥ s₄)
  (ab : s₁ - s₂ = 11) (cd : s₂ - s₃ = 5) (fe : s₃ - s₄ = 13) :
  s₁ - s₄ = 29 := by
  sorry

end NUMINAMATH_CALUDE_square_side_difference_l546_54646


namespace NUMINAMATH_CALUDE_cone_circumscribed_sphere_surface_area_l546_54623

/-- Given a cone with base area π and lateral area twice the base area, 
    the surface area of its circumscribed sphere is 16π/3 -/
theorem cone_circumscribed_sphere_surface_area 
  (base_area : ℝ) 
  (lateral_area : ℝ) 
  (h1 : base_area = π) 
  (h2 : lateral_area = 2 * base_area) : 
  ∃ (r : ℝ), 
    r > 0 ∧ 
    4 * π * r^2 = 16 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_cone_circumscribed_sphere_surface_area_l546_54623


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_three_elevenths_l546_54657

/-- The repeating decimal 0.27̄ -/
def repeating_decimal : ℚ := 27 / 99

theorem repeating_decimal_equals_three_elevenths : 
  repeating_decimal = 3 / 11 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_three_elevenths_l546_54657


namespace NUMINAMATH_CALUDE_transportation_budget_degrees_l546_54639

theorem transportation_budget_degrees (salaries research_dev utilities equipment supplies : ℝ)
  (h1 : salaries = 60)
  (h2 : research_dev = 9)
  (h3 : utilities = 5)
  (h4 : equipment = 4)
  (h5 : supplies = 2)
  (h6 : salaries + research_dev + utilities + equipment + supplies < 100) :
  let transportation := 100 - (salaries + research_dev + utilities + equipment + supplies)
  360 * (transportation / 100) = 72 := by
  sorry

end NUMINAMATH_CALUDE_transportation_budget_degrees_l546_54639


namespace NUMINAMATH_CALUDE_function_composition_l546_54653

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_composition (x : ℝ) (h : x ≥ -1) :
  f (Real.sqrt x - 1) = x - 2 * Real.sqrt x →
  f x = x^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l546_54653


namespace NUMINAMATH_CALUDE_solution_ratio_l546_54633

theorem solution_ratio (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0)
  (eq1 : 8 * x - 6 * y = c) (eq2 : 12 * y - 18 * x = d) : c / d = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_ratio_l546_54633


namespace NUMINAMATH_CALUDE_equation_solutions_l546_54625

theorem equation_solutions :
  (∀ x : ℝ, 4 * (2 * x - 1)^2 = 36 ↔ x = 2 ∨ x = -1) ∧
  (∀ x : ℝ, (1/4) * (2 * x + 3)^3 - 54 = 0 ↔ x = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l546_54625


namespace NUMINAMATH_CALUDE_binomial_even_iff_power_of_two_l546_54608

theorem binomial_even_iff_power_of_two (n : ℕ) : 
  (∃ m : ℕ, n = 2^m) ↔ 
  (∀ k : ℕ, 1 ≤ k ∧ k < n → Even (n.choose k)) := by
sorry

end NUMINAMATH_CALUDE_binomial_even_iff_power_of_two_l546_54608


namespace NUMINAMATH_CALUDE_tile_placement_theorem_l546_54635

/-- Represents a rectangular grid --/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a rectangular tile --/
structure Tile :=
  (width : ℕ)
  (height : ℕ)

/-- Calculates the maximum number of tiles that can be placed in a grid --/
def max_tiles (g : Grid) (t : Tile) : ℕ :=
  sorry

/-- Calculates the number of cells left unpaved --/
def unpaved_cells (g : Grid) (t : Tile) : ℕ :=
  sorry

theorem tile_placement_theorem (g : Grid) (t : Tile) : 
  g.rows = 14 ∧ g.cols = 14 ∧ t.width = 1 ∧ t.height = 4 →
  max_tiles g t = 48 ∧ unpaved_cells g t = 4 :=
sorry

end NUMINAMATH_CALUDE_tile_placement_theorem_l546_54635


namespace NUMINAMATH_CALUDE_isosceles_triangle_midpoint_property_l546_54655

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the properties of the isosceles triangle
def IsIsosceles (t : Triangle) (a : ℝ) : Prop :=
  dist t.X t.Y = a ∧ dist t.Y t.Z = a

-- Define point M on XZ
def PointOnXZ (t : Triangle) (M : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ M = (1 - k) • t.X + k • t.Z

-- Define the midpoint property of M
def IsMidpoint (t : Triangle) (M : ℝ × ℝ) : Prop :=
  dist t.Y M = dist M t.Z

-- Define the sum of distances property
def SumOfDistances (t : Triangle) (M : ℝ × ℝ) (a : ℝ) : Prop :=
  dist t.X M + dist M t.Z = 2 * a

-- Main theorem
theorem isosceles_triangle_midpoint_property
  (t : Triangle) (M : ℝ × ℝ) (a : ℝ) :
  IsIsosceles t a →
  PointOnXZ t M →
  IsMidpoint t M →
  SumOfDistances t M a →
  dist t.Y M = a / 2 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_midpoint_property_l546_54655


namespace NUMINAMATH_CALUDE_intersection_M_N_l546_54691

def M : Set ℝ := {0, 1, 2}
def N : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_M_N : M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l546_54691


namespace NUMINAMATH_CALUDE_pasture_problem_l546_54648

/-- The number of horses c put in the pasture -/
def c_horses : ℕ := 18

/-- The total cost of the pasture in Rs -/
def total_cost : ℕ := 870

/-- b's payment for the pasture in Rs -/
def b_payment : ℕ := 360

/-- a's horses -/
def a_horses : ℕ := 12

/-- b's horses -/
def b_horses : ℕ := 16

/-- a's months -/
def a_months : ℕ := 8

/-- b's months -/
def b_months : ℕ := 9

/-- c's months -/
def c_months : ℕ := 6

theorem pasture_problem :
  c_horses * c_months * total_cost = 
    b_payment * (a_horses * a_months + b_horses * b_months + c_horses * c_months) - 
    b_horses * b_months * total_cost := by
  sorry

end NUMINAMATH_CALUDE_pasture_problem_l546_54648


namespace NUMINAMATH_CALUDE_product_def_l546_54645

theorem product_def (a b c d e f : ℝ) 
  (h1 : a * b * c = 195)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : (a * f) / (c * d) = 0.75) :
  d * e * f = 250 := by
sorry

end NUMINAMATH_CALUDE_product_def_l546_54645


namespace NUMINAMATH_CALUDE_f_decreasing_on_neg_reals_l546_54669

-- Define the function
def f (x : ℝ) : ℝ := |x - 1|

-- State the theorem
theorem f_decreasing_on_neg_reals : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ < 0 → f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_on_neg_reals_l546_54669


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l546_54654

theorem value_of_a_minus_b (a b : ℤ) 
  (eq1 : 3015 * a + 3019 * b = 3023) 
  (eq2 : 3017 * a + 3021 * b = 3025) : 
  a - b = -3 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l546_54654


namespace NUMINAMATH_CALUDE_claire_balloons_count_l546_54638

/-- The number of balloons Claire has at the end of the fair --/
def claire_balloons : ℕ :=
  let initial := 50
  let given_to_girl := 1
  let floated_away := 12
  let given_away_later := 9
  let grabbed_from_coworker := 11
  initial - given_to_girl - floated_away - given_away_later + grabbed_from_coworker

theorem claire_balloons_count : claire_balloons = 39 := by
  sorry

end NUMINAMATH_CALUDE_claire_balloons_count_l546_54638


namespace NUMINAMATH_CALUDE_circle_properties_l546_54610

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 8*x - 10*y = 10 - y^2 + 6*x

-- Define the center of the circle
def center : ℝ × ℝ := (-1, 5)

-- Define the radius of the circle
def radius : ℝ := 6

-- Theorem to prove
theorem circle_properties :
  (∀ x y : ℝ, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
  center.1 + center.2 + radius = 10 := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l546_54610


namespace NUMINAMATH_CALUDE_two_a_plus_a_equals_three_a_l546_54624

theorem two_a_plus_a_equals_three_a (a : ℝ) : 2 * a + a = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_two_a_plus_a_equals_three_a_l546_54624


namespace NUMINAMATH_CALUDE_at_least_one_square_is_one_l546_54604

theorem at_least_one_square_is_one (a b c : ℤ) 
  (h : |a + b + c| + 2 = |a| + |b| + |c|) : 
  a^2 = 1 ∨ b^2 = 1 ∨ c^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_square_is_one_l546_54604


namespace NUMINAMATH_CALUDE_fraction_sum_equals_point_two_l546_54698

theorem fraction_sum_equals_point_two :
  2 / 40 + 4 / 80 + 6 / 120 + 9 / 180 = (0.2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_point_two_l546_54698


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l546_54617

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def are_perpendicular (m : ℝ) : Prop :=
  3 * m + (2 * m - 1) * m = 0

/-- The condition m = -1 is sufficient for the lines to be perpendicular -/
theorem sufficient_condition (m : ℝ) :
  m = -1 → are_perpendicular m :=
by sorry

/-- The condition m = -1 is not necessary for the lines to be perpendicular -/
theorem not_necessary_condition :
  ∃ m : ℝ, m ≠ -1 ∧ are_perpendicular m :=
by sorry

/-- The condition m = -1 is sufficient but not necessary for the lines to be perpendicular -/
theorem sufficient_but_not_necessary :
  (∀ m : ℝ, m = -1 → are_perpendicular m) ∧
  (∃ m : ℝ, m ≠ -1 ∧ are_perpendicular m) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l546_54617


namespace NUMINAMATH_CALUDE_simplify_expression_l546_54641

theorem simplify_expression : (27 * (10 ^ 12)) / (9 * (10 ^ 4)) = 300000000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l546_54641


namespace NUMINAMATH_CALUDE_initial_money_amount_initial_money_amount_proof_l546_54687

/-- Proves that given the conditions in the problem, the initial amount of money is 160 dollars --/
theorem initial_money_amount : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun your_weekly_savings friend_initial_money friend_weekly_savings weeks initial_money =>
    your_weekly_savings = 7 →
    friend_initial_money = 210 →
    friend_weekly_savings = 5 →
    weeks = 25 →
    initial_money + (your_weekly_savings * weeks) = friend_initial_money + (friend_weekly_savings * weeks) →
    initial_money = 160

/-- The proof of the theorem --/
theorem initial_money_amount_proof :
  initial_money_amount 7 210 5 25 160 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_amount_initial_money_amount_proof_l546_54687


namespace NUMINAMATH_CALUDE_problem_statement_l546_54694

theorem problem_statement (a b : ℝ) (h : (1 / a^2) + (1 / b^2) = 4 / (a^2 + b^2)) :
  (b / a)^2022 - (a / b)^2021 = 0 ∨ (b / a)^2022 - (a / b)^2021 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l546_54694


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l546_54620

theorem purely_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := Complex.mk (a^2 - a - 2) (a^2 - 3*a + 2)
  (z.re = 0 ∧ z.im ≠ 0) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l546_54620


namespace NUMINAMATH_CALUDE_problem_statement_l546_54684

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) : 
  (x - 1)^2 + 16/((x - 1)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l546_54684


namespace NUMINAMATH_CALUDE_equation_solutions_l546_54680

theorem equation_solutions : ∃ (x₁ x₂ : ℚ), 
  (x₁ = -1/2 ∧ x₂ = 3/4) ∧ 
  (∀ x : ℚ, 4*x*(2*x+1) = 3*(2*x+1) ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l546_54680


namespace NUMINAMATH_CALUDE_tangent_speed_l546_54696

/-- Given the equation (a * T) / (a * T - R) = (L + x) / x, where x represents a distance,
    prove that the speed of a point determined by x is equal to a * L / R. -/
theorem tangent_speed (a R L T : ℝ) (x : ℝ) (h : (a * T) / (a * T - R) = (L + x) / x) :
  (x / T) = a * L / R := by
  sorry

end NUMINAMATH_CALUDE_tangent_speed_l546_54696


namespace NUMINAMATH_CALUDE_hen_price_calculation_l546_54651

/-- Proves that given 5 goats and 10 hens with a total cost of 2500,
    and an average price of 400 per goat, the average price of a hen is 50. -/
theorem hen_price_calculation (num_goats num_hens total_cost goat_price : ℕ)
    (h1 : num_goats = 5)
    (h2 : num_hens = 10)
    (h3 : total_cost = 2500)
    (h4 : goat_price = 400) :
    (total_cost - num_goats * goat_price) / num_hens = 50 := by
  sorry

end NUMINAMATH_CALUDE_hen_price_calculation_l546_54651


namespace NUMINAMATH_CALUDE_haley_current_height_l546_54683

/-- Haley's growth rate in inches per year -/
def growth_rate : ℝ := 3

/-- Number of years in the future -/
def years : ℝ := 10

/-- Haley's height after 10 years in inches -/
def future_height : ℝ := 50

/-- Haley's current height in inches -/
def current_height : ℝ := future_height - growth_rate * years

theorem haley_current_height : current_height = 20 := by
  sorry

end NUMINAMATH_CALUDE_haley_current_height_l546_54683


namespace NUMINAMATH_CALUDE_only_2012_is_ternary_l546_54603

def is_ternary (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 3 → d < 3

theorem only_2012_is_ternary :
  is_ternary 2012 ∧
  ¬is_ternary 2013 ∧
  ¬is_ternary 2014 ∧
  ¬is_ternary 2015 :=
by sorry

end NUMINAMATH_CALUDE_only_2012_is_ternary_l546_54603


namespace NUMINAMATH_CALUDE_subtraction_of_like_terms_l546_54679

theorem subtraction_of_like_terms (a : ℝ) : 4 * a - 3 * a = a := by sorry

end NUMINAMATH_CALUDE_subtraction_of_like_terms_l546_54679


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_S_l546_54631

def S : Finset ℕ := {8, 88, 888, 8888, 88888, 888888, 8888888, 88888888, 888888888}

def arithmetic_mean (s : Finset ℕ) : ℚ :=
  (s.sum id) / s.card

def digits (n : ℕ) : Finset ℕ :=
  sorry

theorem arithmetic_mean_of_S :
  arithmetic_mean S = 109728268 ∧
  ∀ d : ℕ, d < 10 → (d ∉ digits 109728268 ↔ d = 4) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_S_l546_54631


namespace NUMINAMATH_CALUDE_floor_painting_dimensions_l546_54622

theorem floor_painting_dimensions :
  ∀ (a b x : ℕ),
  0 < a → 0 < b →
  b > a →
  a + b = 15 →
  (a - 2*x) * (b - 2*x) = 2 * a * b / 3 →
  (a = 8 ∧ b = 7) ∨ (a = 7 ∧ b = 8) :=
by sorry

end NUMINAMATH_CALUDE_floor_painting_dimensions_l546_54622


namespace NUMINAMATH_CALUDE_percentage_of_number_l546_54630

theorem percentage_of_number (x : ℚ) (y : ℕ) (z : ℕ) :
  (x / 100) * y = z → x = 33 + 1/3 → y = 210 → z = 70 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_number_l546_54630


namespace NUMINAMATH_CALUDE_median_name_length_l546_54650

/-- Represents the distribution of name lengths -/
structure NameLengthDistribution where
  three_letter : Nat
  four_letter : Nat
  five_letter : Nat
  six_letter : Nat
  seven_letter : Nat

/-- The median of a list of natural numbers -/
def median (l : List Nat) : Nat :=
  sorry

/-- Generates a list of name lengths based on the distribution -/
def generateNameLengths (d : NameLengthDistribution) : List Nat :=
  sorry

theorem median_name_length (d : NameLengthDistribution) :
  d.three_letter = 6 →
  d.four_letter = 5 →
  d.five_letter = 2 →
  d.six_letter = 4 →
  d.seven_letter = 4 →
  d.three_letter + d.four_letter + d.five_letter + d.six_letter + d.seven_letter = 21 →
  median (generateNameLengths d) = 4 := by
  sorry

end NUMINAMATH_CALUDE_median_name_length_l546_54650


namespace NUMINAMATH_CALUDE_min_PQ_length_l546_54689

/-- Circle C with center (3,4) and radius 2 -/
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

/-- Point P is outside the circle -/
def P_outside_circle (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 > 4

/-- Length of PQ equals distance from P to origin -/
def PQ_equals_PO (x y : ℝ) : Prop := ∃ (qx qy : ℝ), 
  circle_C qx qy ∧ (x - qx)^2 + (y - qy)^2 = x^2 + y^2

/-- Theorem: Minimum value of |PQ| is 17/2 -/
theorem min_PQ_length (x y : ℝ) : 
  circle_C x y → P_outside_circle x y → PQ_equals_PO x y → 
  ∃ (qx qy : ℝ), circle_C qx qy ∧ (x - qx)^2 + (y - qy)^2 ≥ (17/2)^2 :=
sorry

end NUMINAMATH_CALUDE_min_PQ_length_l546_54689


namespace NUMINAMATH_CALUDE_consecutive_hits_theorem_l546_54612

/-- The number of ways to arrange 8 shots with 3 hits, where exactly 2 hits are consecutive -/
def consecutive_hits_arrangements (total_shots : ℕ) (total_hits : ℕ) (consecutive_hits : ℕ) : ℕ :=
  if total_shots = 8 ∧ total_hits = 3 ∧ consecutive_hits = 2 then
    30
  else
    0

/-- Theorem stating that the number of arrangements is 30 -/
theorem consecutive_hits_theorem :
  consecutive_hits_arrangements 8 3 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_hits_theorem_l546_54612


namespace NUMINAMATH_CALUDE_arrangements_count_l546_54670

/-- Number of red flags -/
def red_flags : ℕ := 8

/-- Number of white flags -/
def white_flags : ℕ := 8

/-- Number of black flags -/
def black_flags : ℕ := 1

/-- Total number of flags -/
def total_flags : ℕ := red_flags + white_flags + black_flags

/-- Number of flagpoles -/
def flagpoles : ℕ := 2

/-- Function to calculate the number of distinguishable arrangements -/
def count_arrangements (r w b p : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of distinguishable arrangements is 315 -/
theorem arrangements_count :
  count_arrangements red_flags white_flags black_flags flagpoles = 315 :=
sorry

end NUMINAMATH_CALUDE_arrangements_count_l546_54670


namespace NUMINAMATH_CALUDE_unvisited_route_count_l546_54658

/-- The number of ways to distribute four families among four routes with one route unvisited -/
def unvisited_route_scenarios : ℕ := 144

/-- The number of families -/
def num_families : ℕ := 4

/-- The number of available routes -/
def num_routes : ℕ := 4

theorem unvisited_route_count :
  unvisited_route_scenarios = 
    (Nat.choose num_families 2) * (Nat.factorial num_routes) / Nat.factorial (num_routes - 3) :=
sorry

end NUMINAMATH_CALUDE_unvisited_route_count_l546_54658


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l546_54640

/-- The line equation passing through a fixed point for all values of parameter a -/
def line_equation (a x y : ℝ) : Prop :=
  (a - 1) * x - y + 2 * a + 1 = 0

/-- Theorem stating that the line passes through the point (-2, 3) for all values of a -/
theorem line_passes_through_fixed_point :
  ∀ a : ℝ, line_equation a (-2) 3 := by
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l546_54640


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l546_54681

theorem sqrt_product_simplification (p : ℝ) (hp : p > 0) :
  Real.sqrt (30 * p) * Real.sqrt (5 * p) * Real.sqrt (6 * p) = 30 * p * Real.sqrt p :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l546_54681


namespace NUMINAMATH_CALUDE_prob_all_same_color_l546_54677

/-- The probability of picking all same-colored candies from a jar -/
theorem prob_all_same_color (red blue : ℕ) (h_red : red = 15) (h_blue : blue = 5) :
  let total := red + blue
  let prob_terry_red := (red * (red - 1)) / (total * (total - 1))
  let prob_mary_red_given_terry_red := (red - 2) / (total - 2)
  let prob_all_red := prob_terry_red * prob_mary_red_given_terry_red
  let prob_terry_blue := (blue * (blue - 1)) / (total * (total - 1))
  let prob_mary_blue_given_terry_blue := (blue - 2) / (total - 2)
  let prob_all_blue := prob_terry_blue * prob_mary_blue_given_terry_blue
  prob_all_red + prob_all_blue = 31 / 76 := by
sorry

end NUMINAMATH_CALUDE_prob_all_same_color_l546_54677


namespace NUMINAMATH_CALUDE_steelyard_scale_construction_l546_54676

/-- Represents a steelyard (balance) --/
structure Steelyard where
  l : ℝ  -- length of the steelyard
  Q : ℝ  -- weight of the steelyard
  a : ℝ  -- distance where 1 kg balances the steelyard

/-- Theorem for the steelyard scale construction --/
theorem steelyard_scale_construction (S : Steelyard) (p x : ℝ) 
  (h1 : S.l > 0)
  (h2 : S.Q > 0)
  (h3 : S.a > 0)
  (h4 : S.a < S.l)
  (h5 : x > 0)
  (h6 : x < S.l) :
  p * x / S.a = (S.l - x) / (S.l - S.a) :=
sorry

end NUMINAMATH_CALUDE_steelyard_scale_construction_l546_54676


namespace NUMINAMATH_CALUDE_tree_growth_theorem_l546_54600

/-- Calculates the height of a tree after a given number of months -/
def treeHeight (initialHeight : ℝ) (growthRate : ℝ) (weeksPerMonth : ℕ) (months : ℕ) : ℝ :=
  initialHeight + growthRate * (months * weeksPerMonth : ℝ)

/-- Theorem: A tree with initial height 10 feet, growing 2 feet per week, 
    will be 42 feet tall after 4 months (with 4 weeks per month) -/
theorem tree_growth_theorem : 
  treeHeight 10 2 4 4 = 42 := by
  sorry

end NUMINAMATH_CALUDE_tree_growth_theorem_l546_54600


namespace NUMINAMATH_CALUDE_last_released_position_l546_54636

/-- Represents the state of the ransom process -/
structure RansomState where
  remaining_captives : ℕ
  purses_on_table : ℕ
  last_released_position : ℕ

/-- Simulates the ransom process for Robin Hood's captives -/
def ransom_process (initial_captives : ℕ) : ℕ → RansomState := sorry

/-- Theorem stating the position of the last released captive based on the final number of purses -/
theorem last_released_position 
  (initial_captives : ℕ) 
  (final_purses : ℕ) :
  initial_captives = 7 →
  (final_purses = 28 → (ransom_process initial_captives final_purses).last_released_position = 7) ∧
  (final_purses = 27 → 
    ((ransom_process initial_captives final_purses).last_released_position = 6 ∨
     (ransom_process initial_captives final_purses).last_released_position = 7)) :=
by sorry

end NUMINAMATH_CALUDE_last_released_position_l546_54636


namespace NUMINAMATH_CALUDE_average_weight_problem_l546_54688

/-- Given three weights a, b, c, prove that if their average is 45,
    the average of a and b is 40, and b is 31, then the average of b and c is 43. -/
theorem average_weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  b = 31 →
  (b + c) / 2 = 43 := by
sorry

end NUMINAMATH_CALUDE_average_weight_problem_l546_54688


namespace NUMINAMATH_CALUDE_song_book_cost_l546_54611

/-- The cost of the song book given the costs of other items and the total spent --/
theorem song_book_cost (trumpet_cost music_tool_cost total_spent : ℚ) : 
  trumpet_cost = 149.16 →
  music_tool_cost = 9.98 →
  total_spent = 163.28 →
  total_spent - (trumpet_cost + music_tool_cost) = 4.14 := by
  sorry

end NUMINAMATH_CALUDE_song_book_cost_l546_54611


namespace NUMINAMATH_CALUDE_subtraction_decimal_proof_l546_54628

theorem subtraction_decimal_proof :
  (12.358 : ℝ) - (7.2943 : ℝ) = 5.0637 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_decimal_proof_l546_54628


namespace NUMINAMATH_CALUDE_smallest_y_with_remainders_l546_54616

theorem smallest_y_with_remainders : ∃! y : ℕ, 
  y > 0 ∧ 
  y % 6 = 5 ∧ 
  y % 7 = 6 ∧ 
  y % 8 = 7 ∧
  ∀ z : ℕ, z > 0 ∧ z % 6 = 5 ∧ z % 7 = 6 ∧ z % 8 = 7 → y ≤ z :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_y_with_remainders_l546_54616


namespace NUMINAMATH_CALUDE_complex_intersection_l546_54674

theorem complex_intersection (z : ℂ) (k : ℝ) : 
  k > 0 → 
  Complex.abs (z - 4) = 3 * Complex.abs (z + 4) →
  Complex.abs z = k →
  (∃! z', Complex.abs (z' - 4) = 3 * Complex.abs (z' + 4) ∧ Complex.abs z' = k) →
  k = 4 ∨ k = 14 := by
sorry

end NUMINAMATH_CALUDE_complex_intersection_l546_54674


namespace NUMINAMATH_CALUDE_magic_ink_combinations_l546_54668

/-- The number of valid combinations for a magic ink recipe. -/
def validCombinations (herbTypes : ℕ) (essenceTypes : ℕ) (incompatibleHerbs : ℕ) : ℕ :=
  herbTypes * essenceTypes - incompatibleHerbs

/-- Theorem stating that the number of valid combinations for the magic ink is 21. -/
theorem magic_ink_combinations :
  validCombinations 4 6 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_magic_ink_combinations_l546_54668


namespace NUMINAMATH_CALUDE_quadratic_factorization_l546_54672

theorem quadratic_factorization (y : ℝ) : 16 * y^2 - 40 * y + 25 = (4 * y - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l546_54672


namespace NUMINAMATH_CALUDE_min_trips_to_fill_tank_l546_54643

/-- The minimum number of trips required to fill a cylindrical tank using a hemispherical bucket -/
theorem min_trips_to_fill_tank (tank_radius tank_height bucket_radius : ℝ) 
  (hr : tank_radius = 8) 
  (hh : tank_height = 20) 
  (hb : bucket_radius = 6) : 
  ∃ n : ℕ, (n : ℝ) * ((2/3) * Real.pi * bucket_radius^3) ≥ Real.pi * tank_radius^2 * tank_height ∧ 
  ∀ m : ℕ, m < n → (m : ℝ) * ((2/3) * Real.pi * bucket_radius^3) < Real.pi * tank_radius^2 * tank_height :=
by
  sorry

end NUMINAMATH_CALUDE_min_trips_to_fill_tank_l546_54643


namespace NUMINAMATH_CALUDE_sequence_a_properties_l546_54666

def sequence_a (n : ℕ) : ℝ := sorry

def sum_S (n : ℕ) : ℝ := sorry

axiom arithmetic_mean (n : ℕ) : sequence_a n = (sum_S n + 2) / 2

theorem sequence_a_properties :
  (sequence_a 1 = 2 ∧ sequence_a 2 = 4) ∧
  (∀ n : ℕ, n ≥ 1 → sequence_a n = 2^n) := by sorry

end NUMINAMATH_CALUDE_sequence_a_properties_l546_54666


namespace NUMINAMATH_CALUDE_total_dolls_count_l546_54601

/-- The number of dolls owned by the grandmother -/
def grandmother_dolls : ℕ := 50

/-- The number of dolls owned by the sister -/
def sister_dolls : ℕ := grandmother_dolls + 2

/-- The number of dolls owned by Rene -/
def rene_dolls : ℕ := 3 * sister_dolls

/-- The total number of dolls owned by Rene, her sister, and their grandmother -/
def total_dolls : ℕ := grandmother_dolls + sister_dolls + rene_dolls

theorem total_dolls_count : total_dolls = 258 := by
  sorry

end NUMINAMATH_CALUDE_total_dolls_count_l546_54601


namespace NUMINAMATH_CALUDE_boat_downstream_speed_l546_54619

/-- Represents the speed of a boat in different conditions -/
structure BoatSpeed where
  stillWater : ℝ
  upstream : ℝ

/-- Calculates the downstream speed of a boat given its speed in still water and upstream -/
def downstreamSpeed (boat : BoatSpeed) : ℝ :=
  2 * boat.stillWater - boat.upstream

theorem boat_downstream_speed 
  (boat : BoatSpeed) 
  (h1 : boat.stillWater = 8.5) 
  (h2 : boat.upstream = 4) : 
  downstreamSpeed boat = 13 := by
  sorry

end NUMINAMATH_CALUDE_boat_downstream_speed_l546_54619


namespace NUMINAMATH_CALUDE_number_problem_l546_54613

theorem number_problem (x : ℚ) : (x / 4) * 12 = 9 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l546_54613


namespace NUMINAMATH_CALUDE_intersection_parallel_line_l546_54627

/-- The equation of a line passing through the intersection of two given lines and parallel to a third line -/
theorem intersection_parallel_line (x y : ℝ) : 
  (2 * x - 3 * y + 2 = 0) →  -- l₁
  (3 * x - 4 * y - 2 = 0) →  -- l₂
  ∃ (k : ℝ), (4 * x - 2 * y + k = 0) ∧  -- parallel line
  (2 * x - y - 18 = 0) :=  -- result
by sorry

end NUMINAMATH_CALUDE_intersection_parallel_line_l546_54627


namespace NUMINAMATH_CALUDE_cubic_function_property_l546_54607

/-- Given a cubic function f(x) = ax³ - bx + 5 where a and b are real numbers,
    if f(-3) = -1, then f(3) = 11. -/
theorem cubic_function_property (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x^3 - b * x + 5)
    (h2 : f (-3) = -1) : f 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l546_54607


namespace NUMINAMATH_CALUDE_number_of_sevens_in_Q_l546_54659

/-- Definition of R_k as an integer consisting of k repetitions of the digit 7 -/
def R (k : ℕ) : ℕ := 7 * ((10^k - 1) / 9)

/-- The quotient of R_16 divided by R_2 -/
def Q : ℕ := R 16 / R 2

/-- Count the number of sevens in a natural number -/
def count_sevens (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of sevens in Q is equal to 2 -/
theorem number_of_sevens_in_Q : count_sevens Q = 2 := by sorry

end NUMINAMATH_CALUDE_number_of_sevens_in_Q_l546_54659


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l546_54615

theorem binomial_coefficient_divisibility (n : ℕ) : (n + 1) ∣ Nat.choose (2 * n) n := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l546_54615


namespace NUMINAMATH_CALUDE_sqrt_450_equals_15_l546_54652

theorem sqrt_450_equals_15 : Real.sqrt 450 = 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_equals_15_l546_54652


namespace NUMINAMATH_CALUDE_concyclic_intersecting_lines_ratio_l546_54671

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the condition that A, B, C, D are concyclic
def concyclic (A B C D : ℝ × ℝ) : Prop := sorry

-- Define the condition that lines (AB) and (CD) intersect at E
def intersect_at (A B C D E : ℝ × ℝ) : Prop := sorry

-- Define the distance between two points
def distance (P Q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem concyclic_intersecting_lines_ratio 
  (h1 : concyclic A B C D) 
  (h2 : intersect_at A B C D E) :
  (distance A C / distance B C) * (distance A D / distance B D) = 
  distance A E / distance B E := by sorry

end NUMINAMATH_CALUDE_concyclic_intersecting_lines_ratio_l546_54671


namespace NUMINAMATH_CALUDE_inequality_proof_l546_54637

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (3*x^2 - x)/(1 + x^2) + (3*y^2 - y)/(1 + y^2) + (3*z^2 - z)/(1 + z^2) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l546_54637


namespace NUMINAMATH_CALUDE_odds_calculation_l546_54656

theorem odds_calculation (x : ℝ) (h : (x / (x + 5)) = 0.375) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_odds_calculation_l546_54656


namespace NUMINAMATH_CALUDE_ice_cream_cone_types_l546_54621

theorem ice_cream_cone_types (num_flavors : ℕ) (num_combinations : ℕ) (h1 : num_flavors = 4) (h2 : num_combinations = 8) :
  num_combinations / num_flavors = 2 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_cone_types_l546_54621


namespace NUMINAMATH_CALUDE_factorization_equality_l546_54626

theorem factorization_equality (x y : ℝ) : (x + 2) * (x - 2) - 4 * y * (x - y) = (x - 2*y + 2) * (x - 2*y - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l546_54626


namespace NUMINAMATH_CALUDE_greatest_3digit_base9_divisible_by_7_l546_54682

/-- Converts a base 9 number to decimal --/
def base9ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (9 ^ i)) 0

/-- Checks if a number is a valid 3-digit base 9 number --/
def isValid3DigitBase9 (n : Nat) : Prop :=
  ∃ (d₁ d₂ d₃ : Nat), d₁ ≠ 0 ∧ d₁ < 9 ∧ d₂ < 9 ∧ d₃ < 9 ∧ n = base9ToDecimal [d₃, d₂, d₁]

theorem greatest_3digit_base9_divisible_by_7 :
  let n := base9ToDecimal [8, 8, 8]
  isValid3DigitBase9 n ∧ n % 7 = 0 ∧
  ∀ m, isValid3DigitBase9 m → m % 7 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_3digit_base9_divisible_by_7_l546_54682


namespace NUMINAMATH_CALUDE_missing_sale_is_7562_l546_54693

/-- Calculates the missing sale amount given sales for 5 out of 6 months and the average sale -/
def calculate_missing_sale (sale1 sale2 sale3 sale4 sale6 average_sale : ℕ) : ℕ :=
  6 * average_sale - (sale1 + sale2 + sale3 + sale4 + sale6)

theorem missing_sale_is_7562 (sale1 sale2 sale3 sale4 sale6 average_sale : ℕ) 
  (h1 : sale1 = 7435)
  (h2 : sale2 = 7927)
  (h3 : sale3 = 7855)
  (h4 : sale4 = 8230)
  (h5 : sale6 = 5991)
  (h6 : average_sale = 7500) :
  calculate_missing_sale sale1 sale2 sale3 sale4 sale6 average_sale = 7562 := by
  sorry

#eval calculate_missing_sale 7435 7927 7855 8230 5991 7500

end NUMINAMATH_CALUDE_missing_sale_is_7562_l546_54693


namespace NUMINAMATH_CALUDE_simplify_algebraic_expression_l546_54686

theorem simplify_algebraic_expression (x : ℝ) (h1 : x ≠ -3) (h2 : x ≠ 3) (h3 : x ≠ 1) :
  (1 - 4 / (x + 3)) / ((x^2 - 2*x + 1) / (x^2 - 9)) = (x - 3) / (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_simplify_algebraic_expression_l546_54686


namespace NUMINAMATH_CALUDE_profit_percentage_is_25_l546_54664

/-- 
Given that the cost price of 30 articles is equal to the selling price of 24 articles,
prove that the profit percentage is 25%.
-/
theorem profit_percentage_is_25 (C S : ℝ) (h : 30 * C = 24 * S) : 
  (S - C) / C * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_25_l546_54664


namespace NUMINAMATH_CALUDE_intersection_eq_union_implies_a_eq_3_intersection_eq_nonempty_implies_a_eq_neg_5_div_2_l546_54660

-- Define the sets A, B, and C
def A (a : ℝ) := {x : ℝ | x^2 + (4 - a^2) * x + a + 3 = 0}
def B := {x : ℝ | x^2 - 5 * x + 6 = 0}
def C := {x : ℝ | 2 * x^2 - 5 * x + 2 = 0}

-- Theorem 1
theorem intersection_eq_union_implies_a_eq_3 :
  ∃ a : ℝ, (A a) ∩ B = (A a) ∪ B → a = 3 := by sorry

-- Theorem 2
theorem intersection_eq_nonempty_implies_a_eq_neg_5_div_2 :
  ∃ a : ℝ, (A a) ∩ B = (A a) ∩ C ∧ (A a) ∩ B ≠ ∅ → a = -5/2 := by sorry

end NUMINAMATH_CALUDE_intersection_eq_union_implies_a_eq_3_intersection_eq_nonempty_implies_a_eq_neg_5_div_2_l546_54660


namespace NUMINAMATH_CALUDE_sin_cos_equation_solution_l546_54609

theorem sin_cos_equation_solution (x y : ℝ) : 
  (Real.sin (x + y))^2 - (Real.cos (x - y))^2 = 1 ↔ 
  (∃ (k l : ℤ), x = Real.pi / 2 * (2 * k + l + 1) ∧ y = Real.pi / 2 * (2 * k - l)) ∨
  (∃ (m n : ℤ), x = Real.pi / 2 * (2 * m + n) ∧ y = Real.pi / 2 * (2 * m - n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_equation_solution_l546_54609


namespace NUMINAMATH_CALUDE_smallest_number_l546_54629

/-- Convert a number from base 6 to decimal -/
def base6ToDecimal (n : ℕ) : ℕ := sorry

/-- Convert a number from base 4 to decimal -/
def base4ToDecimal (n : ℕ) : ℕ := sorry

/-- Convert a number from base 2 to decimal -/
def base2ToDecimal (n : ℕ) : ℕ := sorry

theorem smallest_number :
  let n1 := base6ToDecimal 210
  let n2 := base4ToDecimal 1000
  let n3 := base2ToDecimal 111111
  n3 < n1 ∧ n3 < n2 := by sorry

end NUMINAMATH_CALUDE_smallest_number_l546_54629


namespace NUMINAMATH_CALUDE_smallest_gcd_multiple_l546_54644

theorem smallest_gcd_multiple (p q : ℕ+) (h : Nat.gcd p q = 15) :
  (∀ p' q' : ℕ+, Nat.gcd p' q' = 15 → Nat.gcd (8 * p') (18 * q') ≥ 30) ∧
  (∃ p' q' : ℕ+, Nat.gcd p' q' = 15 ∧ Nat.gcd (8 * p') (18 * q') = 30) :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_multiple_l546_54644


namespace NUMINAMATH_CALUDE_max_period_is_14_l546_54661

/-- A function with symmetry properties and a period -/
structure SymmetricPeriodicFunction where
  f : ℝ → ℝ
  period : ℝ
  periodic : ∀ x, f (x + period) = f x
  sym_1 : ∀ x, f (1 + x) = f (1 - x)
  sym_8 : ∀ x, f (8 + x) = f (8 - x)

/-- The maximum period for a SymmetricPeriodicFunction is 14 -/
theorem max_period_is_14 (spf : SymmetricPeriodicFunction) : 
  spf.period ≤ 14 := by sorry

end NUMINAMATH_CALUDE_max_period_is_14_l546_54661


namespace NUMINAMATH_CALUDE_hourly_wage_calculation_l546_54675

/-- Calculates the hourly wage given the total earnings, hours worked, widgets produced, and widget bonus rate. -/
def calculate_hourly_wage (total_earnings : ℚ) (hours_worked : ℚ) (widgets_produced : ℚ) (widget_bonus_rate : ℚ) : ℚ :=
  (total_earnings - widgets_produced * widget_bonus_rate) / hours_worked

theorem hourly_wage_calculation :
  let total_earnings : ℚ := 620
  let hours_worked : ℚ := 40
  let widgets_produced : ℚ := 750
  let widget_bonus_rate : ℚ := 0.16
  calculate_hourly_wage total_earnings hours_worked widgets_produced widget_bonus_rate = 12.5 := by
sorry

#eval calculate_hourly_wage 620 40 750 0.16

end NUMINAMATH_CALUDE_hourly_wage_calculation_l546_54675


namespace NUMINAMATH_CALUDE_largest_nineteen_times_digit_sum_l546_54678

/-- The sum of digits of a positive integer -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: 399 is the largest positive integer equal to 19 times the sum of its digits -/
theorem largest_nineteen_times_digit_sum :
  ∀ n : ℕ, n > 0 → n = 19 * sum_of_digits n → n ≤ 399 := by
  sorry

end NUMINAMATH_CALUDE_largest_nineteen_times_digit_sum_l546_54678


namespace NUMINAMATH_CALUDE_expression_equality_l546_54690

theorem expression_equality : 
  (753^2 + 247^2 - 753 * 247) / (753^3 + 247^3) = 1 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l546_54690


namespace NUMINAMATH_CALUDE_language_learning_hours_difference_l546_54634

def hours_english : ℝ := 2
def hours_chinese : ℝ := 5
def hours_spanish : ℝ := 4
def hours_french : ℝ := 3
def hours_german : ℝ := 1.5

theorem language_learning_hours_difference : 
  (hours_chinese + hours_french) - (hours_german + hours_spanish) = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_language_learning_hours_difference_l546_54634


namespace NUMINAMATH_CALUDE_inequality_system_solution_l546_54695

theorem inequality_system_solution (x : ℝ) :
  (x - 2) / (x - 1) < 1 ∧ -x^2 + x + 2 < 0 → x > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l546_54695


namespace NUMINAMATH_CALUDE_eleanor_childless_descendants_l546_54699

/-- Eleanor's family structure -/
structure EleanorFamily where
  daughters : ℕ
  total_descendants : ℕ
  daughters_with_children : ℕ

/-- The number of Eleanor's daughters and granddaughters with no daughters -/
def childless_descendants (f : EleanorFamily) : ℕ :=
  f.total_descendants - f.daughters_with_children

/-- Theorem stating the number of Eleanor's daughters and granddaughters with no daughters -/
theorem eleanor_childless_descendants :
  ∀ f : EleanorFamily,
  f.daughters = 8 →
  f.total_descendants = 43 →
  f.daughters_with_children * 7 = f.total_descendants - f.daughters →
  childless_descendants f = 38 := by
  sorry

end NUMINAMATH_CALUDE_eleanor_childless_descendants_l546_54699


namespace NUMINAMATH_CALUDE_inequality_proof_l546_54632

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b)^2 / 2 + (a + b) / 4 ≥ a * Real.sqrt b + b * Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l546_54632


namespace NUMINAMATH_CALUDE_raw_materials_cost_l546_54665

/-- The amount spent on raw materials given the total amount, machinery cost, and cash percentage. -/
theorem raw_materials_cost (total : ℝ) (machinery : ℝ) (cash_percent : ℝ) 
  (h1 : total = 1000)
  (h2 : machinery = 400)
  (h3 : cash_percent = 0.1)
  (h4 : ∃ (raw_materials : ℝ), raw_materials + machinery + cash_percent * total = total) :
  ∃ (raw_materials : ℝ), raw_materials = 500 := by
sorry

end NUMINAMATH_CALUDE_raw_materials_cost_l546_54665


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l546_54673

theorem complex_fraction_equals_i : (1 + Complex.I) / (1 - Complex.I) = Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l546_54673


namespace NUMINAMATH_CALUDE_polynomial_factor_l546_54642

theorem polynomial_factor (b : ℚ) : 
  (∀ x : ℚ, (3 * x - 4 = 0) → (9 * x^3 + b * x^2 + 17 * x - 76 = 0)) →
  b = -17/6 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_l546_54642


namespace NUMINAMATH_CALUDE_soccer_basketball_difference_l546_54692

theorem soccer_basketball_difference :
  let soccer_boxes : ℕ := 8
  let basketball_boxes : ℕ := 5
  let balls_per_box : ℕ := 12
  let total_soccer_balls := soccer_boxes * balls_per_box
  let total_basketballs := basketball_boxes * balls_per_box
  total_soccer_balls - total_basketballs = 36 :=
by sorry

end NUMINAMATH_CALUDE_soccer_basketball_difference_l546_54692


namespace NUMINAMATH_CALUDE_sports_club_participation_l546_54605

theorem sports_club_participation (total students_swimming students_basketball students_both : ℕ) 
  (h1 : total = 75)
  (h2 : students_swimming = 46)
  (h3 : students_basketball = 34)
  (h4 : students_both = 22) :
  total - (students_swimming + students_basketball - students_both) = 17 := by
sorry

end NUMINAMATH_CALUDE_sports_club_participation_l546_54605


namespace NUMINAMATH_CALUDE_tan_sum_alpha_beta_l546_54663

theorem tan_sum_alpha_beta (α β : Real) 
  (h1 : Real.sin α + Real.sin β = 1/4)
  (h2 : Real.cos α + Real.cos β = 1/3) : 
  Real.tan (α + β) = 24/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_alpha_beta_l546_54663


namespace NUMINAMATH_CALUDE_triangle_isosceles_if_bcosC_eq_CcosB_l546_54685

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the cosine of an angle in a triangle
def cos_angle (t : Triangle) (angle : Fin 3) : ℝ :=
  sorry

-- Define the length of a side in a triangle
def side_length (t : Triangle) (side : Fin 3) : ℝ :=
  sorry

-- Define an isosceles triangle
def is_isosceles (t : Triangle) : Prop :=
  sorry

-- Theorem statement
theorem triangle_isosceles_if_bcosC_eq_CcosB (t : Triangle) :
  side_length t 1 * cos_angle t 2 = side_length t 2 * cos_angle t 1 →
  is_isosceles t :=
sorry

end NUMINAMATH_CALUDE_triangle_isosceles_if_bcosC_eq_CcosB_l546_54685
