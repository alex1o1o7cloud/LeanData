import Mathlib

namespace NUMINAMATH_CALUDE_multiplication_puzzle_l2840_284009

theorem multiplication_puzzle : ∃ (a b : Nat), 
  a < 10000 ∧ 
  b < 1000 ∧ 
  a / 1000 = 3 ∧ 
  a % 100 = 20 ∧
  b / 100 = 3 ∧
  (a * (b % 10)) % 10000 = 9060 ∧
  ((a * (b / 10)) / 10000) * 10000 + ((a * (b / 10)) % 10000) = 62510 ∧
  a * b = 1157940830 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l2840_284009


namespace NUMINAMATH_CALUDE_line_intersects_parabola_once_l2840_284015

/-- The value of k for which the line x = k intersects the parabola x = 3y² - 7y + 2 at exactly one point -/
def k : ℚ := -25/12

/-- The parabola equation -/
def parabola (y : ℝ) : ℝ := 3*y^2 - 7*y + 2

/-- Theorem stating that k is the unique value for which the line x = k intersects the parabola at exactly one point -/
theorem line_intersects_parabola_once :
  ∀ y : ℝ, (∃! y, parabola y = k) ∧ 
  (∀ k' : ℚ, k' ≠ k → ¬(∃! y, parabola y = k')) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_parabola_once_l2840_284015


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l2840_284068

theorem y_in_terms_of_x (x y : ℝ) (h : y - 2*x = 6) : y = 2*x + 6 := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l2840_284068


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2840_284029

/-- Given a geometric sequence with first term a₁ and common ratio q,
    if the sum of the first three terms equals 3a₁, then q = 1 or q = -2 -/
theorem geometric_sequence_ratio (a₁ : ℝ) (q : ℝ) :
  a₁ ≠ 0 →
  a₁ + a₁ * q + a₁ * q^2 = 3 * a₁ →
  q = 1 ∨ q = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2840_284029


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_l2840_284017

theorem cubic_polynomial_root (x : ℝ) (h : x = Real.rpow 4 (1/3) + 1) : 
  x^3 - 3*x^2 + 3*x - 5 = 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_l2840_284017


namespace NUMINAMATH_CALUDE_f_divisibility_l2840_284072

/-- Sequence a defined recursively -/
def a (r s : ℕ) : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => r * a r s (n + 1) + s * a r s n

/-- Product of first n terms of sequence a -/
def f (r s : ℕ) : ℕ → ℕ
  | 0 => 1
  | (n + 1) => f r s n * a r s (n + 1)

/-- Main theorem -/
theorem f_divisibility (r s n k : ℕ) (hr : r > 0) (hs : s > 0) (hk : k > 0) (hnk : n > k) :
  ∃ m : ℕ, f r s n = m * (f r s k * f r s (n - k)) := by
  sorry

end NUMINAMATH_CALUDE_f_divisibility_l2840_284072


namespace NUMINAMATH_CALUDE_luke_game_rounds_l2840_284047

theorem luke_game_rounds (points_per_round : ℕ) (total_points : ℕ) (h1 : points_per_round = 146) (h2 : total_points = 22922) :
  total_points / points_per_round = 157 := by
  sorry

end NUMINAMATH_CALUDE_luke_game_rounds_l2840_284047


namespace NUMINAMATH_CALUDE_fraction_of_number_seven_fourths_of_48_l2840_284056

theorem fraction_of_number (x y z : ℚ) : x * y = z → (x * y) * 48 = z * 48 := by sorry

theorem seven_fourths_of_48 : (7 / 4 : ℚ) * 48 = 84 := by sorry

end NUMINAMATH_CALUDE_fraction_of_number_seven_fourths_of_48_l2840_284056


namespace NUMINAMATH_CALUDE_remainder_5_pow_2023_mod_11_l2840_284098

theorem remainder_5_pow_2023_mod_11 : 5^2023 % 11 = 4 := by sorry

end NUMINAMATH_CALUDE_remainder_5_pow_2023_mod_11_l2840_284098


namespace NUMINAMATH_CALUDE_value_of_expression_l2840_284031

theorem value_of_expression (x : ℝ) (h : x^2 + 3*x + 5 = 11) : 
  3*x^2 + 9*x + 12 = 30 := by
sorry

end NUMINAMATH_CALUDE_value_of_expression_l2840_284031


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_l2840_284051

theorem least_n_satisfying_inequality : 
  (∃ n : ℕ+, (1 : ℚ) / n - (1 : ℚ) / (n + 2) < (1 : ℚ) / 15) ∧ 
  (∀ m : ℕ+, (1 : ℚ) / m - (1 : ℚ) / (m + 2) < (1 : ℚ) / 15 → m ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_l2840_284051


namespace NUMINAMATH_CALUDE_product_of_functions_l2840_284021

theorem product_of_functions (x : ℝ) (hx : x ≠ 0) : 
  let f : ℝ → ℝ := λ x => 2 * x
  let g : ℝ → ℝ := λ x => -(3 * x - 1) / x
  (f x) * (g x) = -6 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_functions_l2840_284021


namespace NUMINAMATH_CALUDE_no_very_convex_function_l2840_284038

/-- A function is very convex if it satisfies the given inequality for all real x and y -/
def VeryConvex (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f x + f y) / 2 ≥ f ((x + y) / 2) + |x - y|

/-- Theorem stating that no very convex function exists -/
theorem no_very_convex_function : ¬ ∃ f : ℝ → ℝ, VeryConvex f := by
  sorry


end NUMINAMATH_CALUDE_no_very_convex_function_l2840_284038


namespace NUMINAMATH_CALUDE_imo_1993_function_exists_l2840_284040

/-- A strictly increasing function from positive integers to positive integers -/
def StrictlyIncreasing (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, m < n → f m < f n

/-- The existence of a function satisfying the IMO 1993 conditions -/
theorem imo_1993_function_exists : ∃ f : ℕ+ → ℕ+, 
  f 1 = 2 ∧ 
  StrictlyIncreasing f ∧ 
  ∀ n : ℕ+, f (f n) = f n + n :=
sorry

end NUMINAMATH_CALUDE_imo_1993_function_exists_l2840_284040


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l2840_284044

theorem triangle_is_equilateral (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a^2 + b^2 = c^2 + a*b →
  Real.cos A * Real.cos B = 1/4 →
  A = B ∧ B = C ∧ C = π/3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l2840_284044


namespace NUMINAMATH_CALUDE_board_numbers_product_l2840_284026

theorem board_numbers_product (a b c d e : ℤ) : 
  ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} : Finset ℤ) = 
    {6, 9, 10, 13, 13, 14, 17, 17, 20, 21} →
  a * b * c * d * e = 4320 := by
sorry

end NUMINAMATH_CALUDE_board_numbers_product_l2840_284026


namespace NUMINAMATH_CALUDE_fourth_side_length_is_correct_l2840_284095

/-- A quadrilateral inscribed in a circle with radius 150√3, where three sides have length 300 --/
structure InscribedQuadrilateral where
  radius : ℝ
  three_side_length : ℝ
  h_radius : radius = 150 * Real.sqrt 3
  h_three_sides : three_side_length = 300

/-- The length of the fourth side of the inscribed quadrilateral --/
def fourth_side_length (q : InscribedQuadrilateral) : ℝ := 562.5

/-- Theorem stating that the fourth side length is correct --/
theorem fourth_side_length_is_correct (q : InscribedQuadrilateral) :
  fourth_side_length q = 562.5 := by
  sorry

end NUMINAMATH_CALUDE_fourth_side_length_is_correct_l2840_284095


namespace NUMINAMATH_CALUDE_area_midpoint_rectangle_l2840_284097

/-- Given a rectangle EFGH with width w and height h, and points P and Q that are
    midpoints of the longer sides EF and GH respectively, the area of EPGQ is
    half the area of EFGH. -/
theorem area_midpoint_rectangle (w h : ℝ) (hw : w > 0) (hh : h > 0) :
  let rect_area := w * h
  let midpoint_rect_area := (w / 2) * h
  midpoint_rect_area = rect_area / 2 := by
  sorry

#check area_midpoint_rectangle

end NUMINAMATH_CALUDE_area_midpoint_rectangle_l2840_284097


namespace NUMINAMATH_CALUDE_not_necessarily_right_triangle_l2840_284050

/-- A triangle with side lengths proportional to 3:4:6 is not necessarily a right triangle -/
theorem not_necessarily_right_triangle (a b c : ℝ) (h : a / b = 3 / 4 ∧ b / c = 4 / 6) :
  ¬ (a^2 + b^2 = c^2) :=
sorry

end NUMINAMATH_CALUDE_not_necessarily_right_triangle_l2840_284050


namespace NUMINAMATH_CALUDE_calculate_expression_l2840_284018

theorem calculate_expression : (-3 : ℚ) * (1/3) / (-1/3) * 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2840_284018


namespace NUMINAMATH_CALUDE_square_difference_equality_l2840_284085

theorem square_difference_equality : 1013^2 - 1009^2 - 1011^2 + 997^2 = -19924 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l2840_284085


namespace NUMINAMATH_CALUDE_picture_distribution_l2840_284030

theorem picture_distribution (total : ℕ) (transfer : ℕ) 
  (h_total : total = 74) (h_transfer : transfer = 6) : 
  ∃ (wang_original fang_original : ℕ),
    wang_original + fang_original = total ∧
    wang_original - transfer = fang_original + transfer ∧
    wang_original = 43 ∧ 
    fang_original = 31 := by
sorry

end NUMINAMATH_CALUDE_picture_distribution_l2840_284030


namespace NUMINAMATH_CALUDE_rare_coin_value_l2840_284002

/-- Given a collection of rare coins where 4 coins are worth 16 dollars, 
    prove that 20 coins of the same type are worth 80 dollars. -/
theorem rare_coin_value (total_coins : ℕ) (sample_coins : ℕ) (sample_value : ℚ) :
  total_coins = 20 →
  sample_coins = 4 →
  sample_value = 16 →
  (total_coins : ℚ) * (sample_value / sample_coins) = 80 :=
by sorry

end NUMINAMATH_CALUDE_rare_coin_value_l2840_284002


namespace NUMINAMATH_CALUDE_order_relationship_l2840_284096

theorem order_relationship (a b c : ℝ) : 
  a = Real.exp (1/2) - 1 → 
  b = Real.log (3/2) → 
  c = 5/12 → 
  a > c ∧ c > b := by
sorry

end NUMINAMATH_CALUDE_order_relationship_l2840_284096


namespace NUMINAMATH_CALUDE_no_real_c_solution_l2840_284091

/-- Given a polynomial x^2 + bx + c with exactly one real root and b = c + 2,
    prove that there are no real values of c that satisfy these conditions. -/
theorem no_real_c_solution (b c : ℝ) 
    (h1 : ∃! x : ℝ, x^2 + b*x + c = 0)  -- exactly one real root
    (h2 : b = c + 2) :                  -- condition b = c + 2
    False :=                            -- no real c satisfies the conditions
  sorry

end NUMINAMATH_CALUDE_no_real_c_solution_l2840_284091


namespace NUMINAMATH_CALUDE_solve_star_equation_l2840_284076

-- Define the custom operation *
def star (a b : ℝ) : ℝ := a * b + a + b

-- Theorem statement
theorem solve_star_equation (x : ℝ) : star 3 x = 15 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_star_equation_l2840_284076


namespace NUMINAMATH_CALUDE_max_dimes_and_nickels_l2840_284041

def total_amount : ℚ := 485 / 100
def dime_value : ℚ := 10 / 100
def nickel_value : ℚ := 5 / 100

theorem max_dimes_and_nickels :
  ∃ (d : ℕ), d * dime_value + d * nickel_value ≤ total_amount ∧
  ∀ (n : ℕ), n * dime_value + n * nickel_value ≤ total_amount → n ≤ d :=
by sorry

end NUMINAMATH_CALUDE_max_dimes_and_nickels_l2840_284041


namespace NUMINAMATH_CALUDE_range_of_x_max_value_of_a_l2840_284055

/-- The profit calculation function for Product A before upgrade -/
def profit_A_before (raw_materials : ℝ) : ℝ := 1.2 * raw_materials

/-- The profit calculation function for Product A after upgrade -/
def profit_A_after (raw_materials : ℝ) (x : ℝ) : ℝ := 1.2 * (1 + 0.005 * x) * (raw_materials - x)

/-- The profit calculation function for Product B -/
def profit_B (x : ℝ) (a : ℝ) : ℝ := 12 * (a - 0.013 * x) * x

/-- The theorem statement for the range of x -/
theorem range_of_x (x : ℝ) :
  x > 0 →
  (profit_A_after 500 x ≥ profit_A_before 500 ↔ 0 < x ∧ x ≤ 300) :=
sorry

/-- The theorem statement for the maximum value of a -/
theorem max_value_of_a :
  ∃ (a : ℝ), a > 0 ∧ a ≤ 5.5 ∧
  ∀ (x : ℝ) (a' : ℝ), 0 < x → x ≤ 300 → a' > 0 →
  (∀ x', 0 < x' → x' ≤ 300 → profit_B x' a' ≤ profit_A_after 500 x') →
  a' ≤ a :=
sorry

end NUMINAMATH_CALUDE_range_of_x_max_value_of_a_l2840_284055


namespace NUMINAMATH_CALUDE_find_number_l2840_284080

theorem find_number : ∃! x : ℝ, 10 * ((2 * (x^2 + 2) + 3) / 5) = 50 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2840_284080


namespace NUMINAMATH_CALUDE_find_x_l2840_284003

theorem find_x (p q r x : ℝ) 
  (h1 : (p + q + r) / 3 = 4) 
  (h2 : (p + q + r + x) / 4 = 5) : 
  x = 8 := by sorry

end NUMINAMATH_CALUDE_find_x_l2840_284003


namespace NUMINAMATH_CALUDE_savings_calculation_l2840_284005

/-- Represents a person's financial information -/
structure Person where
  income : ℕ
  expenditure : ℕ

/-- The problem setup -/
def financialProblem (p1 p2 : Person) : Prop :=
  p1.income = 3000 ∧
  4 * p1.income = 5 * p2.income ∧
  2 * p1.expenditure = 3 * p2.expenditure ∧
  p1.income - p1.expenditure = p2.income - p2.expenditure

/-- The theorem to prove -/
theorem savings_calculation (p1 p2 : Person) :
  financialProblem p1 p2 → p1.income - p1.expenditure = 1200 := by
  sorry

#check savings_calculation

end NUMINAMATH_CALUDE_savings_calculation_l2840_284005


namespace NUMINAMATH_CALUDE_least_possible_difference_l2840_284014

theorem least_possible_difference (x y z : ℤ) : 
  x < y → y < z → 
  y - x > 5 → 
  Even x → Odd y → Odd z → 
  (∀ d : ℤ, z - x ≥ d → d ≥ 9) :=
by sorry

end NUMINAMATH_CALUDE_least_possible_difference_l2840_284014


namespace NUMINAMATH_CALUDE_percentage_less_than_l2840_284063

theorem percentage_less_than (x y : ℝ) (h : x = 12 * y) :
  (x - y) / x * 100 = (11 / 12) * 100 :=
sorry

end NUMINAMATH_CALUDE_percentage_less_than_l2840_284063


namespace NUMINAMATH_CALUDE_salary_problem_l2840_284073

/-- Proves that A's salary is $3750 given the conditions of the problem -/
theorem salary_problem (a b : ℝ) : 
  a + b = 5000 →
  0.05 * a = 0.15 * b →
  a = 3750 := by
  sorry

end NUMINAMATH_CALUDE_salary_problem_l2840_284073


namespace NUMINAMATH_CALUDE_square_root_range_l2840_284070

theorem square_root_range (x : ℝ) : ∃ y : ℝ, y = Real.sqrt (x - 5) ↔ x ≥ 5 := by sorry

end NUMINAMATH_CALUDE_square_root_range_l2840_284070


namespace NUMINAMATH_CALUDE_chocolate_cost_in_dollars_l2840_284094

/-- The cost of the chocolate in cents -/
def chocolate_cost (money_in_pocket : ℕ) (borrowed : ℕ) (needed : ℕ) : ℕ :=
  money_in_pocket * 100 + borrowed + needed

theorem chocolate_cost_in_dollars :
  let money_in_pocket : ℕ := 4
  let borrowed : ℕ := 59
  let needed : ℕ := 41
  (chocolate_cost money_in_pocket borrowed needed) / 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_cost_in_dollars_l2840_284094


namespace NUMINAMATH_CALUDE_sally_grew_five_onions_l2840_284042

/-- The number of onions grown by Sally, given the number of onions grown by Sara and Fred, and the total number of onions. -/
def sallys_onions (sara_onions fred_onions total_onions : ℕ) : ℕ :=
  total_onions - (sara_onions + fred_onions)

/-- Theorem stating that Sally grew 5 onions given the conditions in the problem. -/
theorem sally_grew_five_onions :
  sallys_onions 4 9 18 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sally_grew_five_onions_l2840_284042


namespace NUMINAMATH_CALUDE_power_product_equality_l2840_284000

theorem power_product_equality (a : ℝ) : 4 * a^2 * a = 4 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l2840_284000


namespace NUMINAMATH_CALUDE_x_plus_y_value_l2840_284022

theorem x_plus_y_value (x y : ℚ) 
  (eq1 : 5 * x - 3 * y = 17) 
  (eq2 : 3 * x + 5 * y = 1) : 
  x + y = 21 / 17 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l2840_284022


namespace NUMINAMATH_CALUDE_manuscript_typing_cost_l2840_284066

/-- The cost per page for revisions -/
def revision_cost : ℝ := 4

/-- The total number of pages in the manuscript -/
def total_pages : ℕ := 100

/-- The number of pages revised once -/
def pages_revised_once : ℕ := 35

/-- The number of pages revised twice -/
def pages_revised_twice : ℕ := 15

/-- The total cost of typing the manuscript -/
def total_cost : ℝ := 860

/-- The cost per page for the first time a page is typed -/
def first_time_cost : ℝ := 6

theorem manuscript_typing_cost :
  first_time_cost * total_pages +
  revision_cost * pages_revised_once +
  2 * revision_cost * pages_revised_twice = total_cost :=
by sorry

end NUMINAMATH_CALUDE_manuscript_typing_cost_l2840_284066


namespace NUMINAMATH_CALUDE_number_of_proper_subsets_of_union_l2840_284084

def A : Finset Nat := {2, 3}
def B : Finset Nat := {2, 4, 5}

theorem number_of_proper_subsets_of_union : (Finset.powerset (A ∪ B)).card - 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_of_proper_subsets_of_union_l2840_284084


namespace NUMINAMATH_CALUDE_divisor_sum_inequality_equality_condition_l2840_284074

theorem divisor_sum_inequality (n : ℕ) (hn : n ≥ 2) :
  let divisors := (Finset.range (n + 1)).filter (λ d => n % d = 0)
  (divisors.sum id) / divisors.card ≥ Real.sqrt (n + 1/4) :=
sorry

theorem equality_condition (n : ℕ) (hn : n ≥ 2) :
  let divisors := (Finset.range (n + 1)).filter (λ d => n % d = 0)
  (divisors.sum id) / divisors.card = Real.sqrt (n + 1/4) ↔ n = 2 :=
sorry

end NUMINAMATH_CALUDE_divisor_sum_inequality_equality_condition_l2840_284074


namespace NUMINAMATH_CALUDE_second_journey_half_time_l2840_284093

/-- Represents a journey with distance and speed -/
structure Journey where
  distance : ℝ
  speed : ℝ

/-- Theorem stating that under given conditions, the time of the second journey is half of the first -/
theorem second_journey_half_time (j1 j2 : Journey) 
  (h1 : j1.distance = 80)
  (h2 : j2.distance = 160)
  (h3 : j2.speed = 4 * j1.speed) :
  (j2.distance / j2.speed) = (1/2) * (j1.distance / j1.speed) := by
  sorry

#check second_journey_half_time

end NUMINAMATH_CALUDE_second_journey_half_time_l2840_284093


namespace NUMINAMATH_CALUDE_todd_remaining_money_l2840_284067

/-- Calculates the remaining money after Todd's purchases -/
def remaining_money (initial_amount : ℚ) (candy_price : ℚ) (candy_count : ℕ) 
  (gum_price : ℚ) (gum_count : ℕ) (soda_price : ℚ) (soda_count : ℕ) 
  (soda_discount : ℚ) : ℚ :=
  let candy_cost := candy_price * candy_count
  let gum_cost := gum_price * gum_count
  let soda_cost := soda_price * soda_count * (1 - soda_discount)
  let total_cost := candy_cost + gum_cost + soda_cost
  initial_amount - total_cost

/-- Theorem stating Todd's remaining money after purchases -/
theorem todd_remaining_money :
  remaining_money 50 2.5 7 1.5 5 3 3 0.2 = 17.8 := by
  sorry

end NUMINAMATH_CALUDE_todd_remaining_money_l2840_284067


namespace NUMINAMATH_CALUDE_quadratic_function_range_l2840_284065

theorem quadratic_function_range (x : ℝ) :
  let y := x^2 - 4*x + 3
  y < 0 ↔ 1 < x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l2840_284065


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2840_284061

/-- Represents the side lengths of squares in the rectangle -/
structure SquareSides where
  smallest : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  largest : ℝ

/-- The rectangle composed of eight squares -/
structure Rectangle where
  sides : SquareSides
  length : ℝ
  width : ℝ

/-- Theorem stating the perimeter of the rectangle -/
theorem rectangle_perimeter (rect : Rectangle) 
  (h1 : rect.sides.smallest = 1)
  (h2 : rect.sides.a = 4)
  (h3 : rect.sides.b = 5)
  (h4 : rect.sides.c = 5)
  (h5 : rect.sides.largest = 14)
  (h6 : rect.length = rect.sides.largest + rect.sides.b)
  (h7 : rect.width = rect.sides.largest) : 
  2 * (rect.length + rect.width) = 66 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2840_284061


namespace NUMINAMATH_CALUDE_unique_adjacent_sums_exist_l2840_284027

def isValidArrangement (arrangement : List Nat) : Prop :=
  arrangement.length = 10 ∧
  (∀ n, n ∈ arrangement → 1 ≤ n ∧ n ≤ 10) ∧
  (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∈ arrangement)

def adjacentSums (arrangement : List Nat) : List Nat :=
  (List.zip arrangement (arrangement.rotateLeft 1)).map (λ (a, b) => a + b)

theorem unique_adjacent_sums_exist : 
  ∃ arrangement : List Nat, 
    isValidArrangement arrangement ∧ 
    (adjacentSums arrangement).Nodup :=
sorry

end NUMINAMATH_CALUDE_unique_adjacent_sums_exist_l2840_284027


namespace NUMINAMATH_CALUDE_racks_fit_on_shelf_l2840_284028

/-- Represents the number of CDs a single rack can hold -/
def cds_per_rack : ℕ := 8

/-- Represents the total number of CDs the shelf can hold -/
def total_cds : ℕ := 32

/-- Calculates the number of racks that can fit on the shelf -/
def racks_on_shelf : ℕ := total_cds / cds_per_rack

/-- Proves that the number of racks that can fit on the shelf is 4 -/
theorem racks_fit_on_shelf : racks_on_shelf = 4 := by
  sorry

end NUMINAMATH_CALUDE_racks_fit_on_shelf_l2840_284028


namespace NUMINAMATH_CALUDE_fiftieth_student_age_l2840_284048

theorem fiftieth_student_age
  (total_students : Nat)
  (average_age : ℝ)
  (group1_count : Nat)
  (group1_avg : ℝ)
  (group2_count : Nat)
  (group2_avg : ℝ)
  (group3_count : Nat)
  (group3_avg : ℝ)
  (group4_count : Nat)
  (group4_avg : ℝ)
  (h1 : total_students = 50)
  (h2 : average_age = 20)
  (h3 : group1_count = 15)
  (h4 : group1_avg = 18)
  (h5 : group2_count = 15)
  (h6 : group2_avg = 22)
  (h7 : group3_count = 10)
  (h8 : group3_avg = 25)
  (h9 : group4_count = 9)
  (h10 : group4_avg = 24)
  (h11 : group1_count + group2_count + group3_count + group4_count = total_students - 1) :
  (total_students : ℝ) * average_age - 
  (group1_count : ℝ) * group1_avg - 
  (group2_count : ℝ) * group2_avg - 
  (group3_count : ℝ) * group3_avg - 
  (group4_count : ℝ) * group4_avg = 66 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_student_age_l2840_284048


namespace NUMINAMATH_CALUDE_trip_cost_calculation_l2840_284025

def initial_odometer : ℕ := 85300
def final_odometer : ℕ := 85335
def fuel_efficiency : ℚ := 25
def gas_price : ℚ := 21/5  -- $4.20 represented as a rational number

def trip_cost : ℚ :=
  (final_odometer - initial_odometer : ℚ) / fuel_efficiency * gas_price

theorem trip_cost_calculation :
  trip_cost = 588/100 := by sorry

end NUMINAMATH_CALUDE_trip_cost_calculation_l2840_284025


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2840_284053

theorem complex_number_quadrant : 
  let z : ℂ := (2 + Complex.I) * Complex.I
  (z.re < 0 ∧ z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2840_284053


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2840_284090

def U : Set Nat := {0, 1, 2, 3, 4}
def A : Set Nat := {0, 1, 3}
def B : Set Nat := {2, 3, 4}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2840_284090


namespace NUMINAMATH_CALUDE_pencil_profit_problem_l2840_284011

theorem pencil_profit_problem (total_pencils : ℕ) (buy_price sell_price : ℚ) (desired_profit : ℚ) :
  total_pencils = 1500 →
  buy_price = 1/10 →
  sell_price = 1/4 →
  desired_profit = 100 →
  ∃ (pencils_sold : ℕ), 
    pencils_sold ≤ total_pencils ∧
    sell_price * pencils_sold - buy_price * total_pencils = desired_profit ∧
    pencils_sold = 1000 :=
by sorry

end NUMINAMATH_CALUDE_pencil_profit_problem_l2840_284011


namespace NUMINAMATH_CALUDE_bobby_paycheck_l2840_284034

/-- Calculates the final amount in Bobby's paycheck after deductions --/
def final_paycheck (gross_salary : ℚ) : ℚ :=
  let federal_tax := gross_salary * (1/3)
  let state_tax := gross_salary * (8/100)
  let local_tax := gross_salary * (5/100)
  let health_insurance := 50
  let life_insurance := 20
  let parking_fee := 10
  let retirement_contribution := gross_salary * (3/100)
  let total_deductions := federal_tax + state_tax + local_tax + health_insurance + life_insurance + parking_fee + retirement_contribution
  gross_salary - total_deductions

/-- Proves that Bobby's final paycheck amount is $148 --/
theorem bobby_paycheck : final_paycheck 450 = 148 := by
  sorry

#eval final_paycheck 450

end NUMINAMATH_CALUDE_bobby_paycheck_l2840_284034


namespace NUMINAMATH_CALUDE_joan_initial_balloons_l2840_284052

/-- The number of blue balloons Joan initially had -/
def initial_balloons : ℕ := sorry

/-- The number of balloons Sally gave to Joan -/
def sally_gave : ℕ := 5

/-- The number of balloons Joan gave to Jessica -/
def joan_gave : ℕ := 2

/-- The number of balloons Joan has now -/
def joan_now : ℕ := 12

theorem joan_initial_balloons :
  initial_balloons + sally_gave - joan_gave = joan_now :=
sorry

end NUMINAMATH_CALUDE_joan_initial_balloons_l2840_284052


namespace NUMINAMATH_CALUDE_no_two_digit_primes_with_digit_sum_12_l2840_284078

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem no_two_digit_primes_with_digit_sum_12 :
  ∀ n : ℕ, is_two_digit n → Nat.Prime n → digit_sum n = 12 → False :=
by
  sorry

end NUMINAMATH_CALUDE_no_two_digit_primes_with_digit_sum_12_l2840_284078


namespace NUMINAMATH_CALUDE_crossing_over_result_l2840_284039

/-- Represents a chromatid with its staining pattern -/
structure Chromatid where
  staining : ℕ → Bool  -- True for darker staining, False for lighter

/-- Represents a chromosome with two sister chromatids -/
structure Chromosome where
  chromatid1 : Chromatid
  chromatid2 : Chromatid

/-- Represents the process of DNA replication with BrdU -/
def dnaReplication (c : Chromosome) : Chromosome :=
  { chromatid1 := { staining := fun _ => true },
    chromatid2 := c.chromatid1 }

/-- Represents the process of crossing over between sister chromatids -/
def crossingOver (c : Chromosome) : Chromosome :=
  { chromatid1 := { staining := fun n => if n % 2 = 0 then c.chromatid1.staining n else c.chromatid2.staining n },
    chromatid2 := { staining := fun n => if n % 2 = 0 then c.chromatid2.staining n else c.chromatid1.staining n } }

/-- Theorem stating the result of the experiment -/
theorem crossing_over_result (initialChromosome : Chromosome) :
  ∃ (n m : ℕ), 
    let finalChromosome := crossingOver (dnaReplication (dnaReplication initialChromosome))
    finalChromosome.chromatid1.staining n ≠ finalChromosome.chromatid1.staining m ∧
    finalChromosome.chromatid2.staining n ≠ finalChromosome.chromatid2.staining m :=
  sorry


end NUMINAMATH_CALUDE_crossing_over_result_l2840_284039


namespace NUMINAMATH_CALUDE_sum_product_theorem_l2840_284045

theorem sum_product_theorem (a b c d : ℝ) 
  (eq1 : a + b + c = -4)
  (eq2 : a + b + d = 2)
  (eq3 : a + c + d = 15)
  (eq4 : b + c + d = 10) :
  a * b + c * d = 485 / 9 := by
sorry

end NUMINAMATH_CALUDE_sum_product_theorem_l2840_284045


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l2840_284049

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (5 * x) + Real.sin (7 * x) = 2 * Real.sin (6 * x) * Real.cos x :=
by sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l2840_284049


namespace NUMINAMATH_CALUDE_transportation_cost_l2840_284032

/-- Transportation problem theorem -/
theorem transportation_cost
  (city_A_supply : ℕ)
  (city_B_supply : ℕ)
  (market_C_demand : ℕ)
  (market_D_demand : ℕ)
  (cost_A_to_C : ℕ)
  (cost_A_to_D : ℕ)
  (cost_B_to_C : ℕ)
  (cost_B_to_D : ℕ)
  (x : ℕ)
  (h1 : city_A_supply = 240)
  (h2 : city_B_supply = 260)
  (h3 : market_C_demand = 200)
  (h4 : market_D_demand = 300)
  (h5 : cost_A_to_C = 20)
  (h6 : cost_A_to_D = 30)
  (h7 : cost_B_to_C = 24)
  (h8 : cost_B_to_D = 32)
  (h9 : x ≤ city_A_supply)
  (h10 : x ≤ market_C_demand) :
  (cost_A_to_C * x +
   cost_A_to_D * (city_A_supply - x) +
   cost_B_to_C * (market_C_demand - x) +
   cost_B_to_D * (market_D_demand - (city_A_supply - x))) =
  13920 - 2 * x :=
by sorry

end NUMINAMATH_CALUDE_transportation_cost_l2840_284032


namespace NUMINAMATH_CALUDE_balls_after_2010_steps_l2840_284064

/-- Converts a natural number to its base-6 representation --/
def toBase6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- Sums the digits in a list --/
def sumDigits (digits : List ℕ) : ℕ :=
  digits.foldl (· + ·) 0

/-- Represents the ball-and-box process --/
def ballBoxProcess (steps : ℕ) : ℕ :=
  sumDigits (toBase6 steps)

/-- Theorem stating that the number of balls after 2010 steps
    is equal to the sum of digits in the base-6 representation of 2010 --/
theorem balls_after_2010_steps :
  ballBoxProcess 2010 = 11 := by sorry

end NUMINAMATH_CALUDE_balls_after_2010_steps_l2840_284064


namespace NUMINAMATH_CALUDE_solution_x_l2840_284092

theorem solution_x (x : Real) 
  (h1 : x ∈ Set.Ioo 0 (π / 2))
  (h2 : 1 / Real.sin x = 1 / Real.sin (2 * x) + 1 / Real.sin (4 * x) + 1 / Real.sin (8 * x)) :
  x = π / 15 ∨ x = π / 5 ∨ x = π / 3 ∨ x = 7 * π / 15 := by
  sorry

end NUMINAMATH_CALUDE_solution_x_l2840_284092


namespace NUMINAMATH_CALUDE_melissa_games_played_l2840_284006

def total_points : ℕ := 81
def points_per_game : ℕ := 27

theorem melissa_games_played :
  total_points / points_per_game = 3 := by sorry

end NUMINAMATH_CALUDE_melissa_games_played_l2840_284006


namespace NUMINAMATH_CALUDE_simplify_expression_l2840_284086

theorem simplify_expression (b c : ℝ) : 
  (2 : ℝ) * (3 * b) * (4 * b^2) * (5 * b^3) * (6 * b^4) * (7 * c^2) = 5040 * b^10 * c^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2840_284086


namespace NUMINAMATH_CALUDE_regression_lines_intersection_l2840_284012

/-- A linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The point (x, y) lies on the regression line -/
def on_line (l : RegressionLine) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem regression_lines_intersection
  (l₁ l₂ : RegressionLine)
  (s t : ℝ)
  (h₁ : on_line l₁ s t)
  (h₂ : on_line l₂ s t) :
  ∃ (x y : ℝ), on_line l₁ x y ∧ on_line l₂ x y ∧ x = s ∧ y = t :=
sorry

end NUMINAMATH_CALUDE_regression_lines_intersection_l2840_284012


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l2840_284060

theorem line_segment_endpoint (y : ℝ) : 
  y > 0 → 
  ((1 - (-8))^2 + (y - 3)^2)^(1/2 : ℝ) = 15 → 
  y = 15 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l2840_284060


namespace NUMINAMATH_CALUDE_intersection_nonempty_iff_l2840_284035

/-- Given sets A and B, prove the condition for their non-empty intersection -/
theorem intersection_nonempty_iff (a : ℝ) : 
  (∃ x : ℝ, x ∈ {x | 1 ≤ x ∧ x ≤ 2} ∩ {x | x ≤ a}) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_iff_l2840_284035


namespace NUMINAMATH_CALUDE_function_min_value_l2840_284007

-- Define the function f(x)
def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

-- State the theorem
theorem function_min_value 
  (m : ℝ) 
  (h_max : ∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 3 ∧ ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y m ≤ 3) :
  ∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -37 ∧ ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y m ≥ -37 :=
sorry

end NUMINAMATH_CALUDE_function_min_value_l2840_284007


namespace NUMINAMATH_CALUDE_binomial_12_choose_10_l2840_284079

theorem binomial_12_choose_10 : Nat.choose 12 10 = 66 := by sorry

end NUMINAMATH_CALUDE_binomial_12_choose_10_l2840_284079


namespace NUMINAMATH_CALUDE_shape_triangle_area_ratio_l2840_284016

/-- A shape with a certain area -/
structure Shape where
  area : ℝ
  area_pos : area > 0

/-- A triangle with a certain area -/
structure Triangle where
  area : ℝ
  area_pos : area > 0

/-- The theorem stating the relationship between the areas of a shape and a triangle -/
theorem shape_triangle_area_ratio 
  (s : Shape) 
  (t : Triangle) 
  (h : s.area / t.area = 2) : 
  s.area = 2 * t.area := by
  sorry

end NUMINAMATH_CALUDE_shape_triangle_area_ratio_l2840_284016


namespace NUMINAMATH_CALUDE_janous_inequality_l2840_284013

theorem janous_inequality (x y : ℝ) (hx : x > -1) (hy : y > -1) (hsum : x + y = 1) :
  x / (y + 1) + y / (x + 1) ≥ 2 / 3 ∧
  (x / (y + 1) + y / (x + 1) = 2 / 3 ↔ x = 1 / 2 ∧ y = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_janous_inequality_l2840_284013


namespace NUMINAMATH_CALUDE_fourth_root_simplification_l2840_284088

theorem fourth_root_simplification :
  (3^7 * 5^3 : ℝ)^(1/4) = 3 * (135 : ℝ)^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_simplification_l2840_284088


namespace NUMINAMATH_CALUDE_leading_digit_logarithm_l2840_284004

-- Define a function to get the leading digit of a real number
noncomputable def leadingDigit (x : ℝ) : ℕ := sorry

-- Define the logarithm base 10
noncomputable def log10 (x : ℝ) : ℝ := sorry

-- State the theorem
theorem leading_digit_logarithm (M : ℝ) (a : ℕ) :
  (leadingDigit (6 * 47 * log10 M) = a) →
  ((leadingDigit (log10 (1000 / M)) = 3 - a) ∨
   (leadingDigit (log10 (1000 / M)) = 2 - a)) :=
by sorry

end NUMINAMATH_CALUDE_leading_digit_logarithm_l2840_284004


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2840_284054

theorem fraction_to_decimal : (21 : ℚ) / 40 = 0.525 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2840_284054


namespace NUMINAMATH_CALUDE_jake_and_sister_weight_l2840_284008

theorem jake_and_sister_weight (jake_weight : ℕ) (h : jake_weight = 188) :
  ∃ (sister_weight : ℕ),
    jake_weight - 8 = 2 * sister_weight ∧
    jake_weight + sister_weight = 278 := by
  sorry

end NUMINAMATH_CALUDE_jake_and_sister_weight_l2840_284008


namespace NUMINAMATH_CALUDE_train_speed_l2840_284058

theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 450) (h2 : time = 8) :
  length / time = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2840_284058


namespace NUMINAMATH_CALUDE_garden_length_l2840_284062

/-- Proves that a rectangular garden with perimeter 500 m and breadth 100 m has length 150 m -/
theorem garden_length (perimeter : ℝ) (breadth : ℝ) (length : ℝ) : 
  perimeter = 500 → 
  breadth = 100 → 
  perimeter = 2 * (length + breadth) → 
  length = 150 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_l2840_284062


namespace NUMINAMATH_CALUDE_alice_probability_after_three_turns_l2840_284071

/-- Represents the probability of Alice having the ball after three turns in the baseball game. -/
def aliceProbabilityAfterThreeTurns : ℚ :=
  let aliceKeepProb : ℚ := 1/2
  let aliceTossProb : ℚ := 1/2
  let bobTossProb : ℚ := 3/5
  let bobKeepProb : ℚ := 2/5
  
  -- Alice passes to Bob, Bob passes to Alice, Alice keeps
  let seq1 : ℚ := aliceTossProb * bobTossProb * aliceKeepProb
  -- Alice passes to Bob, Bob passes to Alice, Alice passes to Bob
  let seq2 : ℚ := aliceTossProb * bobTossProb * aliceTossProb
  -- Alice keeps, Alice keeps, Alice keeps
  let seq3 : ℚ := aliceKeepProb * aliceKeepProb * aliceKeepProb
  -- Alice keeps, Alice passes to Bob, Bob passes to Alice
  let seq4 : ℚ := aliceKeepProb * aliceTossProb * bobTossProb
  
  seq1 + seq2 + seq3 + seq4

/-- Theorem stating that the probability of Alice having the ball after three turns is 23/40. -/
theorem alice_probability_after_three_turns :
  aliceProbabilityAfterThreeTurns = 23/40 := by
  sorry

end NUMINAMATH_CALUDE_alice_probability_after_three_turns_l2840_284071


namespace NUMINAMATH_CALUDE_arithmetic_mean_sqrt3_sqrt2_l2840_284089

theorem arithmetic_mean_sqrt3_sqrt2 :
  let a := Real.sqrt 3 + Real.sqrt 2
  let b := Real.sqrt 3 - Real.sqrt 2
  (a + b) / 2 = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_sqrt3_sqrt2_l2840_284089


namespace NUMINAMATH_CALUDE_unique_prime_with_prime_sums_l2840_284057

theorem unique_prime_with_prime_sums : ∃! p : ℕ, 
  Nat.Prime p ∧ Nat.Prime (p + 10) ∧ Nat.Prime (p + 14) := by sorry

end NUMINAMATH_CALUDE_unique_prime_with_prime_sums_l2840_284057


namespace NUMINAMATH_CALUDE_infinite_solutions_iff_d_equals_five_l2840_284036

theorem infinite_solutions_iff_d_equals_five :
  ∀ d : ℝ, (∀ y : ℝ, 3 * (5 + d * y) = 15 * y + 15) ↔ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_iff_d_equals_five_l2840_284036


namespace NUMINAMATH_CALUDE_intersection_condition_l2840_284024

/-- The line equation -/
def line_equation (x y m : ℝ) : Prop := x - y + m = 0

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 1 = 0

/-- Two distinct intersection points exist -/
def has_two_distinct_intersections (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    line_equation x₁ y₁ m ∧ circle_equation x₁ y₁ ∧
    line_equation x₂ y₂ m ∧ circle_equation x₂ y₂

/-- The theorem statement -/
theorem intersection_condition (m : ℝ) :
  0 < m → m < 1 → has_two_distinct_intersections m :=
by sorry

end NUMINAMATH_CALUDE_intersection_condition_l2840_284024


namespace NUMINAMATH_CALUDE_basketball_match_loss_percentage_l2840_284083

theorem basketball_match_loss_percentage 
  (won lost : ℕ) 
  (h1 : won > 0 ∧ lost > 0) 
  (h2 : won / lost = 7 / 3) : 
  (lost : ℚ) / ((won : ℚ) + lost) * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_basketball_match_loss_percentage_l2840_284083


namespace NUMINAMATH_CALUDE_area_of_K_l2840_284075

/-- The set K in the plane Cartesian coordinate system xOy -/
def K : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (abs p.1 + abs (3 * p.2) - 6) * (abs (3 * p.1) + abs p.2 - 6) ≤ 0}

/-- The area of set K -/
theorem area_of_K : MeasureTheory.volume K = 24 := by
  sorry

end NUMINAMATH_CALUDE_area_of_K_l2840_284075


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2840_284081

theorem sufficient_not_necessary (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, HasDerivAt (fun x => a^x) (a^x * Real.log a) x ∧ a^x * Real.log a < 0) →
  (∀ x : ℝ, HasDerivAt (fun x => (2-a)*x^3) (3*(2-a)*x^2) x ∧ 3*(2-a)*x^2 > 0) ∧
  ¬((∀ x : ℝ, HasDerivAt (fun x => (2-a)*x^3) (3*(2-a)*x^2) x ∧ 3*(2-a)*x^2 > 0) →
    (∀ x : ℝ, HasDerivAt (fun x => a^x) (a^x * Real.log a) x ∧ a^x * Real.log a < 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2840_284081


namespace NUMINAMATH_CALUDE_purchases_total_price_l2840_284087

/-- The total price of a refrigerator and a washing machine -/
def total_price (refrigerator_price washing_machine_price : ℕ) : ℕ :=
  refrigerator_price + washing_machine_price

/-- Theorem: The total price of the purchases is $7060 -/
theorem purchases_total_price :
  let refrigerator_price : ℕ := 4275
  let washing_machine_price : ℕ := refrigerator_price - 1490
  total_price refrigerator_price washing_machine_price = 7060 := by
sorry

end NUMINAMATH_CALUDE_purchases_total_price_l2840_284087


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_third_l2840_284077

theorem opposite_of_negative_one_third :
  let x : ℚ := -1/3
  let opposite (y : ℚ) : ℚ := -y
  opposite x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_third_l2840_284077


namespace NUMINAMATH_CALUDE_smallest_sum_proof_l2840_284037

theorem smallest_sum_proof : 
  let sums : List ℚ := [1/4 + 1/5, 1/4 + 1/6, 1/4 + 1/9, 1/4 + 1/8, 1/4 + 1/7]
  (∀ s ∈ sums, 1/4 + 1/9 ≤ s) ∧ (1/4 + 1/9 = 13/36) := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_proof_l2840_284037


namespace NUMINAMATH_CALUDE_paper_pickup_sum_l2840_284019

theorem paper_pickup_sum : 127.5 + 345.25 + 518.75 = 991.5 := by
  sorry

end NUMINAMATH_CALUDE_paper_pickup_sum_l2840_284019


namespace NUMINAMATH_CALUDE_baker_sales_change_l2840_284069

/-- A baker's weekly pastry sales problem --/
theorem baker_sales_change (price : ℕ) (days_per_week : ℕ) (monday_sales : ℕ) (avg_sales : ℕ) 
  (h1 : price = 5)
  (h2 : days_per_week = 7)
  (h3 : monday_sales = 2)
  (h4 : avg_sales = 5) :
  ∃ (daily_change : ℕ),
    daily_change = 1 ∧
    monday_sales + 
    (monday_sales + daily_change) + 
    (monday_sales + 2 * daily_change) + 
    (monday_sales + 3 * daily_change) + 
    (monday_sales + 4 * daily_change) + 
    (monday_sales + 5 * daily_change) + 
    (monday_sales + 6 * daily_change) = days_per_week * avg_sales :=
by
  sorry

end NUMINAMATH_CALUDE_baker_sales_change_l2840_284069


namespace NUMINAMATH_CALUDE_product_of_g_at_roots_of_f_l2840_284059

theorem product_of_g_at_roots_of_f (x₁ x₂ x₃ x₄ x₅ : ℂ) : 
  (x₁^5 - x₁^3 + 2*x₁^2 + 1 = 0) →
  (x₂^5 - x₂^3 + 2*x₂^2 + 1 = 0) →
  (x₃^5 - x₃^3 + 2*x₃^2 + 1 = 0) →
  (x₄^5 - x₄^3 + 2*x₄^2 + 1 = 0) →
  (x₅^5 - x₅^3 + 2*x₅^2 + 1 = 0) →
  (x₁^2 - 3) * (x₂^2 - 3) * (x₃^2 - 3) * (x₄^2 - 3) * (x₅^2 - 3) = -59 :=
by sorry

end NUMINAMATH_CALUDE_product_of_g_at_roots_of_f_l2840_284059


namespace NUMINAMATH_CALUDE_special_function_properties_l2840_284046

/-- A function satisfying specific properties -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (f 0 ≠ 0) ∧
  (∀ x > 0, f x > 1) ∧
  (∀ a b : ℝ, f (a + b) = f a * f b)

theorem special_function_properties (f : ℝ → ℝ) (hf : SpecialFunction f) :
  (f 0 = 1) ∧
  (∀ x : ℝ, f x > 0) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) := by
  sorry

end NUMINAMATH_CALUDE_special_function_properties_l2840_284046


namespace NUMINAMATH_CALUDE_regularPolygonProperties_givenPolygonSatisfiesProperties_l2840_284099

-- Define a regular polygon
structure RegularPolygon where
  sides : ℕ
  exteriorAngle : ℝ
  interiorAngle : ℝ

-- Define the properties of the given regular polygon
def givenPolygon : RegularPolygon where
  sides := 20
  exteriorAngle := 18
  interiorAngle := 162

-- Theorem statement
theorem regularPolygonProperties (p : RegularPolygon) 
  (h1 : p.exteriorAngle = 18) : 
  p.sides = 20 ∧ p.interiorAngle = 162 := by
  sorry

-- Proof that the given polygon satisfies the theorem
theorem givenPolygonSatisfiesProperties : 
  givenPolygon.sides = 20 ∧ givenPolygon.interiorAngle = 162 := by
  apply regularPolygonProperties givenPolygon
  rfl

end NUMINAMATH_CALUDE_regularPolygonProperties_givenPolygonSatisfiesProperties_l2840_284099


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2840_284020

def U : Set ℤ := {-1, 0, 1, 2}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {0, 2}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {0} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2840_284020


namespace NUMINAMATH_CALUDE_range_of_a_l2840_284043

theorem range_of_a (a : ℝ) : 
  ((∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
   (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0)) ↔ 
  (a ≤ -2 ∨ a = 1) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2840_284043


namespace NUMINAMATH_CALUDE_smallest_max_sum_l2840_284023

theorem smallest_max_sum (a b c d e : ℕ+) (h : a + b + c + d + e = 2020) :
  (∃ (a' b' c' d' e' : ℕ+), 
    a' + b' + c' + d' + e' = 2020 ∧
    max (a' + b') (max (a' + d') (max (b' + e') (c' + d'))) = 1011) ∧
  (∀ (a'' b'' c'' d'' e'' : ℕ+),
    a'' + b'' + c'' + d'' + e'' = 2020 →
    max (a'' + b'') (max (a'' + d'') (max (b'' + e'') (c'' + d''))) ≥ 1011) :=
sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l2840_284023


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2840_284033

theorem quadratic_equation_roots (α β : ℝ) : 
  ((1 + β) / (2 + β) = -1 / α) ∧ 
  ((α * β^2 + 121) / (1 - α^2 * β) = 1) →
  (∃ a b c : ℝ, (a * α^2 + b * α + c = 0) ∧ 
               (a * β^2 + b * β + c = 0) ∧ 
               ((a = 1 ∧ b = 12 ∧ c = 10) ∨ 
                (a = 1 ∧ b = -10 ∧ c = -12))) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2840_284033


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l2840_284082

theorem trigonometric_expression_equals_one : 
  (Real.sin (15 * π / 180) * Real.cos (25 * π / 180) + Real.cos (165 * π / 180) * Real.cos (115 * π / 180)) / 
  (Real.sin (35 * π / 180) * Real.cos (5 * π / 180) + Real.cos (145 * π / 180) * Real.cos (85 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l2840_284082


namespace NUMINAMATH_CALUDE_f_lower_bound_and_g_inequality_l2840_284010

noncomputable section

def f (x : ℝ) := x - Real.log x

def g (x : ℝ) := x^3 + x^2 * (f x) - 16*x

theorem f_lower_bound_and_g_inequality {x : ℝ} (hx : x > 0) :
  f x ≥ 1 ∧ g x > -20 := by sorry

end NUMINAMATH_CALUDE_f_lower_bound_and_g_inequality_l2840_284010


namespace NUMINAMATH_CALUDE_number_equality_l2840_284001

theorem number_equality : ∃ y : ℝ, 0.4 * y = (1/3) * 45 ∧ y = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l2840_284001
