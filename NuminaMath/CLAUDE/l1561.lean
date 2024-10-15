import Mathlib

namespace NUMINAMATH_CALUDE_different_orders_eq_120_l1561_156123

/-- The number of ways to arrange n elements. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of students who won awards. -/
def total_students : ℕ := 6

/-- The number of students whose order is fixed. -/
def fixed_order_students : ℕ := 3

/-- The number of different orders for all students to go on stage. -/
def different_orders : ℕ := permutations total_students / permutations fixed_order_students

theorem different_orders_eq_120 : different_orders = 120 := by
  sorry

end NUMINAMATH_CALUDE_different_orders_eq_120_l1561_156123


namespace NUMINAMATH_CALUDE_power_equality_l1561_156102

theorem power_equality (n b : ℝ) : n = 2^(1/4) → n^b = 16 → b = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l1561_156102


namespace NUMINAMATH_CALUDE_tom_stock_profit_l1561_156149

/-- Calculate Tom's overall profit from stock transactions -/
theorem tom_stock_profit : 
  let stock_a_initial_shares : ℕ := 20
  let stock_a_initial_price : ℚ := 3
  let stock_b_initial_shares : ℕ := 30
  let stock_b_initial_price : ℚ := 5
  let stock_c_initial_shares : ℕ := 15
  let stock_c_initial_price : ℚ := 10
  let commission_rate : ℚ := 2 / 100
  let stock_a_sold_shares : ℕ := 10
  let stock_a_sell_price : ℚ := 4
  let stock_b_sold_shares : ℕ := 20
  let stock_b_sell_price : ℚ := 7
  let stock_c_sold_shares : ℕ := 5
  let stock_c_sell_price : ℚ := 12
  let stock_a_value_increase : ℚ := 2
  let stock_b_value_increase : ℚ := 1.2
  let stock_c_value_decrease : ℚ := 0.9

  let initial_cost := (stock_a_initial_shares * stock_a_initial_price + 
                       stock_b_initial_shares * stock_b_initial_price + 
                       stock_c_initial_shares * stock_c_initial_price) * (1 + commission_rate)

  let sales_revenue := (stock_a_sold_shares * stock_a_sell_price + 
                        stock_b_sold_shares * stock_b_sell_price + 
                        stock_c_sold_shares * stock_c_sell_price) * (1 - commission_rate)

  let remaining_value := (stock_a_initial_shares - stock_a_sold_shares) * stock_a_initial_price * stock_a_value_increase + 
                         (stock_b_initial_shares - stock_b_sold_shares) * stock_b_initial_price * stock_b_value_increase + 
                         (stock_c_initial_shares - stock_c_sold_shares) * stock_c_initial_price * stock_c_value_decrease

  let profit := sales_revenue + remaining_value - initial_cost

  profit = 78
  := by sorry

end NUMINAMATH_CALUDE_tom_stock_profit_l1561_156149


namespace NUMINAMATH_CALUDE_parabola_hyperbola_tangent_l1561_156157

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := x^2 - 2*x + 2

/-- The hyperbola equation -/
def hyperbola (m : ℝ) (x y : ℝ) : Prop := y^2 - m*x^2 = 1

/-- The tangency condition -/
def are_tangent (m : ℝ) : Prop :=
  ∃ x : ℝ, (hyperbola m x (parabola x)) ∧
    (∀ x' : ℝ, x' ≠ x → ¬(hyperbola m x' (parabola x')))

theorem parabola_hyperbola_tangent :
  are_tangent 1 := by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_tangent_l1561_156157


namespace NUMINAMATH_CALUDE_rectangle_area_preservation_l1561_156111

theorem rectangle_area_preservation (original_length original_width : ℝ) 
  (h_length : original_length = 140)
  (h_width : original_width = 40)
  (length_increase_percent : ℝ) 
  (h_increase : length_increase_percent = 30) :
  let new_length := original_length * (1 + length_increase_percent / 100)
  let new_width := (original_length * original_width) / new_length
  let width_decrease_percent := (original_width - new_width) / original_width * 100
  ∃ ε > 0, abs (width_decrease_percent - 23.08) < ε :=
sorry

end NUMINAMATH_CALUDE_rectangle_area_preservation_l1561_156111


namespace NUMINAMATH_CALUDE_pencil_sales_theorem_l1561_156151

/-- The number of pencils initially sold for a rupee when losing 30% --/
def initial_pencils : ℝ := 20

/-- The number of pencils sold for a rupee when gaining 30% --/
def gain_pencils : ℝ := 10.77

/-- The percentage of cost price when losing 30% --/
def loss_percentage : ℝ := 0.7

/-- The percentage of cost price when gaining 30% --/
def gain_percentage : ℝ := 1.3

theorem pencil_sales_theorem :
  initial_pencils * loss_percentage = gain_pencils * gain_percentage := by
  sorry

#check pencil_sales_theorem

end NUMINAMATH_CALUDE_pencil_sales_theorem_l1561_156151


namespace NUMINAMATH_CALUDE_point_N_coordinates_l1561_156189

-- Define the points and lines
def M : ℝ × ℝ := (0, -1)
def N : ℝ × ℝ := (2, 3)

def line1 (x y : ℝ) : Prop := x - y + 1 = 0
def line2 (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- Define the perpendicular property
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem point_N_coordinates :
  line1 N.1 N.2 ∧
  perpendicular 
    ((N.2 - M.2) / (N.1 - M.1)) 
    (-(1 / 2)) →
  N = (2, 3) := by
  sorry

end NUMINAMATH_CALUDE_point_N_coordinates_l1561_156189


namespace NUMINAMATH_CALUDE_plan1_greater_loss_l1561_156180

/-- Probability of minor flooding -/
def p_minor : ℝ := 0.2

/-- Probability of major flooding -/
def p_major : ℝ := 0.05

/-- Cost of building a protective wall -/
def wall_cost : ℝ := 4000

/-- Loss due to major flooding -/
def major_flood_loss : ℝ := 30000

/-- Loss due to minor flooding in Plan 2 -/
def minor_flood_loss : ℝ := 15000

/-- Expected loss for Plan 1 -/
def expected_loss_plan1 : ℝ := major_flood_loss * p_major + wall_cost * p_minor + wall_cost

/-- Expected loss for Plan 2 -/
def expected_loss_plan2 : ℝ := major_flood_loss * p_major + minor_flood_loss * p_minor

/-- Theorem stating that the expected loss of Plan 1 is greater than the expected loss of Plan 2 -/
theorem plan1_greater_loss : expected_loss_plan1 > expected_loss_plan2 :=
  sorry

end NUMINAMATH_CALUDE_plan1_greater_loss_l1561_156180


namespace NUMINAMATH_CALUDE_football_playtime_l1561_156164

def total_playtime_hours : ℝ := 1.5
def basketball_playtime_minutes : ℕ := 60

theorem football_playtime (total_playtime_minutes : ℕ) 
  (h1 : total_playtime_minutes = Int.floor (total_playtime_hours * 60)) 
  (h2 : total_playtime_minutes ≥ basketball_playtime_minutes) : 
  total_playtime_minutes - basketball_playtime_minutes = 30 := by
  sorry

end NUMINAMATH_CALUDE_football_playtime_l1561_156164


namespace NUMINAMATH_CALUDE_half_sum_sequence_common_ratio_l1561_156162

/-- A geometric sequence where each term is half the sum of its next two terms -/
def HalfSumSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0 ∧ a n = (a (n + 1) + a (n + 2)) / 2

/-- The common ratio of a geometric sequence -/
def CommonRatio (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = r * a n

theorem half_sum_sequence_common_ratio (a : ℕ → ℝ) (r : ℝ) :
  HalfSumSequence a → CommonRatio a r → r = 1 := by sorry

end NUMINAMATH_CALUDE_half_sum_sequence_common_ratio_l1561_156162


namespace NUMINAMATH_CALUDE_baseball_game_attendance_l1561_156106

theorem baseball_game_attendance (total : ℕ) (first_team_percent : ℚ) (second_team_percent : ℚ) 
  (h1 : total = 50)
  (h2 : first_team_percent = 40 / 100)
  (h3 : second_team_percent = 34 / 100) :
  total - (total * first_team_percent).floor - (total * second_team_percent).floor = 13 := by
  sorry

end NUMINAMATH_CALUDE_baseball_game_attendance_l1561_156106


namespace NUMINAMATH_CALUDE_problem_3_l1561_156100

theorem problem_3 : (-48) / ((-2)^3) - (-25) * (-4) + (-2)^3 = -102 := by
  sorry

end NUMINAMATH_CALUDE_problem_3_l1561_156100


namespace NUMINAMATH_CALUDE_same_color_prob_six_green_seven_white_l1561_156122

/-- The probability of drawing two balls of the same color from a bag containing 
    6 green balls and 7 white balls. -/
def same_color_probability (green : ℕ) (white : ℕ) : ℚ :=
  let total := green + white
  let p_green := (green / total) * ((green - 1) / (total - 1))
  let p_white := (white / total) * ((white - 1) / (total - 1))
  p_green + p_white

/-- Theorem stating that the probability of drawing two balls of the same color 
    from a bag with 6 green and 7 white balls is 6/13. -/
theorem same_color_prob_six_green_seven_white : 
  same_color_probability 6 7 = 6 / 13 := by
  sorry

end NUMINAMATH_CALUDE_same_color_prob_six_green_seven_white_l1561_156122


namespace NUMINAMATH_CALUDE_three_numbers_sum_to_50_l1561_156165

def number_list : List Nat := [21, 19, 30, 25, 3, 12, 9, 15, 6, 27]

theorem three_numbers_sum_to_50 :
  ∃ (a b c : Nat), a ∈ number_list ∧ b ∈ number_list ∧ c ∈ number_list ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b + c = 50 :=
by sorry

end NUMINAMATH_CALUDE_three_numbers_sum_to_50_l1561_156165


namespace NUMINAMATH_CALUDE_original_cost_of_dvd_pack_l1561_156127

theorem original_cost_of_dvd_pack (discount : ℕ) (price_after_discount : ℕ) 
  (h1 : discount = 25)
  (h2 : price_after_discount = 51) :
  discount + price_after_discount = 76 := by
sorry

end NUMINAMATH_CALUDE_original_cost_of_dvd_pack_l1561_156127


namespace NUMINAMATH_CALUDE_complex_number_problem_l1561_156109

theorem complex_number_problem (a : ℝ) :
  let z : ℂ := a + (10 * Complex.I) / (3 - Complex.I)
  (∃ b : ℝ, z = Complex.I * b) → Complex.abs (a - 2 * Complex.I) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1561_156109


namespace NUMINAMATH_CALUDE_factorization_proof_l1561_156116

theorem factorization_proof (x y : ℝ) : x * (y - 1) + 4 * (1 - y) = (y - 1) * (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1561_156116


namespace NUMINAMATH_CALUDE_min_money_required_l1561_156178

/-- Represents the number of candies of each type -/
structure CandyCounts where
  apple : ℕ
  orange : ℕ
  strawberry : ℕ
  grape : ℕ

/-- Represents the vending machine with given conditions -/
def VendingMachine (c : CandyCounts) : Prop :=
  c.apple = 2 * c.orange ∧
  c.strawberry = 2 * c.grape ∧
  c.apple = 2 * c.strawberry ∧
  c.apple + c.orange + c.strawberry + c.grape = 90

/-- The cost of a single candy -/
def candy_cost : ℚ := 1/10

/-- The minimum number of candies to buy -/
def min_candies_to_buy (c : CandyCounts) : ℕ :=
  min c.grape 10 + 3 + 3 + 3

/-- The theorem to prove -/
theorem min_money_required (c : CandyCounts) (h : VendingMachine c) :
  (min_candies_to_buy c : ℚ) * candy_cost = 19/10 := by
  sorry


end NUMINAMATH_CALUDE_min_money_required_l1561_156178


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l1561_156173

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 169 + y^2 / 144 = 1

-- Define a point on the ellipse
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : is_on_ellipse x y

-- Define the foci of the ellipse
structure Foci where
  f1 : ℝ × ℝ
  f2 : ℝ × ℝ
  are_foci : ∀ (p : PointOnEllipse), 
    Real.sqrt ((p.x - f1.1)^2 + (p.y - f1.2)^2) + 
    Real.sqrt ((p.x - f2.1)^2 + (p.y - f2.2)^2) = 26

-- The theorem to prove
theorem ellipse_triangle_perimeter 
  (p : PointOnEllipse) (f : Foci) : 
  Real.sqrt ((p.x - f.f1.1)^2 + (p.y - f.f1.2)^2) +
  Real.sqrt ((p.x - f.f2.1)^2 + (p.y - f.f2.2)^2) +
  Real.sqrt ((f.f1.1 - f.f2.1)^2 + (f.f1.2 - f.f2.2)^2) = 36 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l1561_156173


namespace NUMINAMATH_CALUDE_unique_solution_l1561_156193

/-- A single digit is a natural number from 0 to 9. -/
def SingleDigit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

/-- The equation that Θ must satisfy. -/
def SatisfiesEquation (Θ : ℕ) : Prop := 504 * Θ = 40 + Θ + Θ^2

theorem unique_solution :
  ∃! Θ : ℕ, SingleDigit Θ ∧ SatisfiesEquation Θ ∧ Θ = 9 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1561_156193


namespace NUMINAMATH_CALUDE_probability_two_non_defective_pens_l1561_156101

theorem probability_two_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) 
  (h1 : total_pens = 12) (h2 : defective_pens = 4) :
  let non_defective_pens := total_pens - defective_pens
  let prob_first := non_defective_pens / total_pens
  let prob_second := (non_defective_pens - 1) / (total_pens - 1)
  prob_first * prob_second = 14 / 33 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_non_defective_pens_l1561_156101


namespace NUMINAMATH_CALUDE_simplify_expressions_l1561_156183

theorem simplify_expressions :
  ((-2.48 + 4.33 + (-7.52) + (-4.33)) = -10) ∧
  ((7/13 * (-9) + 7/13 * (-18) + 7/13) = -14) ∧
  (-20 * (1/19) * 38 = -762) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l1561_156183


namespace NUMINAMATH_CALUDE_angle_A_is_60_degrees_sides_b_c_are_2_l1561_156185

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_condition (t : Triangle) : Prop :=
  t.a * Real.cos t.C + Real.sqrt 3 * t.a * Real.sin t.C - t.b - t.c = 0

def has_area_sqrt_3 (t : Triangle) : Prop :=
  1/2 * t.b * t.c * Real.sin t.A = Real.sqrt 3

-- Theorem 1
theorem angle_A_is_60_degrees (t : Triangle) 
  (h : satisfies_condition t) : t.A = Real.pi / 3 := by
  sorry

-- Theorem 2
theorem sides_b_c_are_2 (t : Triangle) 
  (h1 : satisfies_condition t)
  (h2 : t.a = 2)
  (h3 : has_area_sqrt_3 t) : t.b = 2 ∧ t.c = 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_60_degrees_sides_b_c_are_2_l1561_156185


namespace NUMINAMATH_CALUDE_square_side_increase_l1561_156160

theorem square_side_increase (s : ℝ) (h : s > 0) :
  let new_area := s^2 * 1.5625
  let new_side := s * 1.25
  new_area = new_side^2 := by sorry

end NUMINAMATH_CALUDE_square_side_increase_l1561_156160


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1561_156133

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℚ),
    (P = -4/15 ∧ Q = -11/6 ∧ R = 31/10) ∧
    ∀ (x : ℝ), x ≠ 1 → x ≠ 4 → x ≠ 6 →
      (x^2 - 5) / ((x - 1) * (x - 4) * (x - 6)) =
      P / (x - 1) + Q / (x - 4) + R / (x - 6) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1561_156133


namespace NUMINAMATH_CALUDE_least_positive_value_cubic_equation_l1561_156115

/-- The least positive integer value of a cubic equation with prime number constraints -/
theorem least_positive_value_cubic_equation (x y z w : ℕ) : 
  Prime x → Prime y → Prime z → Prime w →
  x + y + z + w < 50 →
  (∀ a b c d : ℕ, Prime a → Prime b → Prime c → Prime d → 
    a + b + c + d < 50 → 
    24 * a^3 + 16 * b^3 - 7 * c^3 + 5 * d^3 ≥ 24 * x^3 + 16 * y^3 - 7 * z^3 + 5 * w^3) →
  24 * x^3 + 16 * y^3 - 7 * z^3 + 5 * w^3 = 1464 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_value_cubic_equation_l1561_156115


namespace NUMINAMATH_CALUDE_log_equation_sum_l1561_156169

theorem log_equation_sum (A B C : ℕ+) : 
  (Nat.gcd A.val (Nat.gcd B.val C.val) = 1) →
  (A : ℝ) * (Real.log 5 / Real.log 100) + (B : ℝ) * (Real.log 2 / Real.log 100) = C →
  A + B + C = 5 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_sum_l1561_156169


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l1561_156186

/-- Given a triangle DEF with side lengths d, e, and f satisfying certain conditions,
    prove that its largest angle is 120°. -/
theorem largest_angle_in_triangle (d e f : ℝ) (h1 : d + 2*e + 2*f = d^2) (h2 : d + 2*e - 2*f = -9) :
  ∃ (D E F : ℝ), D + E + F = 180 ∧ max D (max E F) = 120 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l1561_156186


namespace NUMINAMATH_CALUDE_percentage_problem_l1561_156119

theorem percentage_problem (x : ℝ) (h : 150 = 250 / 100 * x) : x = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1561_156119


namespace NUMINAMATH_CALUDE_solve_for_y_l1561_156177

theorem solve_for_y (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1561_156177


namespace NUMINAMATH_CALUDE_marble_probability_difference_l1561_156172

/-- The number of red marbles in the box -/
def red_marbles : ℕ := 1500

/-- The number of black marbles in the box -/
def black_marbles : ℕ := 1500

/-- The number of white marbles in the box -/
def white_marbles : ℕ := 1

/-- The total number of marbles in the box -/
def total_marbles : ℕ := red_marbles + black_marbles + white_marbles

/-- The probability of drawing two marbles of the same color (including white pairings) -/
def Ps : ℚ := (red_marbles * (red_marbles - 1) + black_marbles * (black_marbles - 1) + 2 * (red_marbles + black_marbles) * white_marbles) / (total_marbles * (total_marbles - 1))

/-- The probability of drawing two marbles of different colors (excluding white pairings) -/
def Pd : ℚ := (2 * red_marbles * black_marbles) / (total_marbles * (total_marbles - 1))

/-- The theorem stating that the absolute difference between Ps and Pd is 1/3 -/
theorem marble_probability_difference : |Ps - Pd| = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_difference_l1561_156172


namespace NUMINAMATH_CALUDE_arithmetic_progression_first_term_l1561_156198

def is_arithmetic_progression (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def is_increasing (a : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (List.range n).map a |>.sum

theorem arithmetic_progression_first_term (a : ℕ → ℤ) :
  is_arithmetic_progression a →
  is_increasing a →
  let S := sum_first_n_terms a 10
  (a 6 * a 12 > S + 1) →
  (a 7 * a 11 < S + 17) →
  a 1 ∈ ({-6, -5, -4, -2, -1, 0} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_first_term_l1561_156198


namespace NUMINAMATH_CALUDE_max_consecutive_even_sum_l1561_156136

/-- The sum of k consecutive even integers starting from 2n is 156 -/
def ConsecutiveEvenSum (n k : ℕ) : Prop :=
  2 * k * n + k * (k - 1) = 156

/-- The proposition that 4 is the maximum number of consecutive even integers summing to 156 -/
theorem max_consecutive_even_sum :
  (∃ n : ℕ, ConsecutiveEvenSum n 4) ∧
  (∀ k : ℕ, k > 4 → ¬∃ n : ℕ, ConsecutiveEvenSum n k) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_even_sum_l1561_156136


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1561_156184

/-- Represents a quadratic function y = ax² + bx + 3 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  h : a ≠ 0

/-- The axis of symmetry of the quadratic function is x = 1 -/
def axis_of_symmetry (f : QuadraticFunction) : Prop :=
  -f.b / (2 * f.a) = 1

/-- 3 is a root of the quadratic equation ax² + bx + 3 = 0 -/
def is_root_three (f : QuadraticFunction) : Prop :=
  f.a * 3^2 + f.b * 3 + 3 = 0

/-- The maximum value of the quadratic function is 4 -/
def max_value_is_four (f : QuadraticFunction) : Prop :=
  ∀ x, f.a * x^2 + f.b * x + 3 ≤ 4

/-- When x = 2, y = 5 -/
def y_is_five_at_two (f : QuadraticFunction) : Prop :=
  f.a * 2^2 + f.b * 2 + 3 = 5

/-- The main theorem -/
theorem quadratic_function_properties (f : QuadraticFunction) :
  axis_of_symmetry f → is_root_three f → max_value_is_four f →
  ¬(y_is_five_at_two f) := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1561_156184


namespace NUMINAMATH_CALUDE_words_in_page_l1561_156142

/-- The number of words Tom can type per minute -/
def words_per_minute : ℕ := 90

/-- The number of minutes it takes Tom to type 10 pages -/
def minutes_for_ten_pages : ℕ := 50

/-- The number of pages Tom types in the given time -/
def number_of_pages : ℕ := 10

/-- Calculates the number of words in a page -/
def words_per_page : ℕ := (words_per_minute * minutes_for_ten_pages) / number_of_pages

/-- Theorem stating that there are 450 words in a page -/
theorem words_in_page : words_per_page = 450 := by
  sorry

end NUMINAMATH_CALUDE_words_in_page_l1561_156142


namespace NUMINAMATH_CALUDE_complex_power_four_l1561_156166

theorem complex_power_four (i : ℂ) : i * i = -1 → (1 - i)^4 = -4 := by sorry

end NUMINAMATH_CALUDE_complex_power_four_l1561_156166


namespace NUMINAMATH_CALUDE_arcsin_arccos_equation_solution_l1561_156195

theorem arcsin_arccos_equation_solution :
  ∃ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 ∧
  Real.arcsin x + Real.arcsin (2*x) = Real.arccos x + Real.arccos (2*x) ∧
  x = Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_arcsin_arccos_equation_solution_l1561_156195


namespace NUMINAMATH_CALUDE_largest_gold_coins_l1561_156156

theorem largest_gold_coins : 
  ∃ (n : ℕ), n = 146 ∧ 
  (∃ (k : ℕ), n = 13 * k + 3) ∧ 
  n < 150 ∧
  ∀ (m : ℕ), (∃ (j : ℕ), m = 13 * j + 3) → m < 150 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_gold_coins_l1561_156156


namespace NUMINAMATH_CALUDE_truck_and_goods_problem_l1561_156153

theorem truck_and_goods_problem (x : ℕ) (total_goods : ℕ) :
  (3 * x + 5 = total_goods) →  -- Condition 1
  (4 * (x - 5) = total_goods) →  -- Condition 2
  (x = 25 ∧ total_goods = 80) :=  -- Conclusion
by sorry

end NUMINAMATH_CALUDE_truck_and_goods_problem_l1561_156153


namespace NUMINAMATH_CALUDE_only_one_statement_correct_l1561_156135

theorem only_one_statement_correct : 
  ¬(∀ (a b : ℤ), a < b → a^2 < b^2) ∧ 
  ¬(∀ (a : ℤ), a^2 > 0) ∧ 
  ¬(∀ (a : ℤ), -a < 0) ∧ 
  (∀ (a b c : ℤ), a * c^2 < b * c^2 → a < b) :=
by sorry

end NUMINAMATH_CALUDE_only_one_statement_correct_l1561_156135


namespace NUMINAMATH_CALUDE_exists_interest_rate_unique_interest_rate_l1561_156154

/-- The interest rate that satisfies the given conditions --/
def interest_rate_equation (r : ℝ) : Prop :=
  1200 * ((1 + r/2)^2 - 1 - r) = 3

/-- Theorem stating that there exists an interest rate satisfying the equation --/
theorem exists_interest_rate : ∃ r : ℝ, interest_rate_equation r ∧ r > 0 ∧ r < 1 := by
  sorry

/-- Theorem stating that the interest rate solution is unique --/
theorem unique_interest_rate : ∃! r : ℝ, interest_rate_equation r ∧ r > 0 ∧ r < 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_interest_rate_unique_interest_rate_l1561_156154


namespace NUMINAMATH_CALUDE_denise_spending_l1561_156188

/-- Represents the types of dishes available --/
inductive Dish
| Simple
| Meat
| Fish

/-- Represents the types of vitamins available --/
inductive Vitamin
| Milk
| Fruit
| Special

/-- Returns the price of a dish --/
def dishPrice (d : Dish) : ℕ :=
  match d with
  | Dish.Simple => 7
  | Dish.Meat => 11
  | Dish.Fish => 14

/-- Returns the price of a vitamin --/
def vitaminPrice (v : Vitamin) : ℕ :=
  match v with
  | Vitamin.Milk => 6
  | Vitamin.Fruit => 7
  | Vitamin.Special => 9

/-- Calculates the total price of a meal (dish + vitamin) --/
def mealPrice (d : Dish) (v : Vitamin) : ℕ :=
  dishPrice d + vitaminPrice v

/-- Represents a person's meal choice --/
structure MealChoice where
  dish : Dish
  vitamin : Vitamin

/-- The main theorem to prove --/
theorem denise_spending (julio_choice denise_choice : MealChoice)
  (h : mealPrice julio_choice.dish julio_choice.vitamin = 
       mealPrice denise_choice.dish denise_choice.vitamin + 6) :
  mealPrice denise_choice.dish denise_choice.vitamin = 14 ∨
  mealPrice denise_choice.dish denise_choice.vitamin = 17 := by
  sorry


end NUMINAMATH_CALUDE_denise_spending_l1561_156188


namespace NUMINAMATH_CALUDE_expression_equality_l1561_156114

theorem expression_equality : -1^2023 + |Real.sqrt 3 - 2| - 3 * Real.tan (π / 3) = 1 - 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1561_156114


namespace NUMINAMATH_CALUDE_expected_rainfall_is_50_4_l1561_156152

/-- Weather forecast probabilities and rainfall amounts -/
structure WeatherForecast where
  days : ℕ
  prob_sun : ℝ
  prob_rain_3 : ℝ
  prob_rain_8 : ℝ
  amount_rain_3 : ℝ
  amount_rain_8 : ℝ

/-- Expected total rainfall over the forecast period -/
def expectedTotalRainfall (forecast : WeatherForecast) : ℝ :=
  forecast.days * (forecast.prob_rain_3 * forecast.amount_rain_3 + 
                   forecast.prob_rain_8 * forecast.amount_rain_8)

/-- Theorem: The expected total rainfall for the given forecast is 50.4 inches -/
theorem expected_rainfall_is_50_4 (forecast : WeatherForecast) 
  (h1 : forecast.days = 14)
  (h2 : forecast.prob_sun = 0.3)
  (h3 : forecast.prob_rain_3 = 0.4)
  (h4 : forecast.prob_rain_8 = 0.3)
  (h5 : forecast.amount_rain_3 = 3)
  (h6 : forecast.amount_rain_8 = 8)
  (h7 : forecast.prob_sun + forecast.prob_rain_3 + forecast.prob_rain_8 = 1) :
  expectedTotalRainfall forecast = 50.4 := by
  sorry

#eval expectedTotalRainfall { 
  days := 14, 
  prob_sun := 0.3, 
  prob_rain_3 := 0.4, 
  prob_rain_8 := 0.3, 
  amount_rain_3 := 3, 
  amount_rain_8 := 8 
}

end NUMINAMATH_CALUDE_expected_rainfall_is_50_4_l1561_156152


namespace NUMINAMATH_CALUDE_equation_one_real_solution_l1561_156148

theorem equation_one_real_solution :
  ∃! x : ℝ, (3 * x) / (x^2 + 2 * x + 4) + (4 * x) / (x^2 - 4 * x + 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_one_real_solution_l1561_156148


namespace NUMINAMATH_CALUDE_range_of_m_l1561_156159

-- Define the linear function
def f (m : ℝ) (x : ℝ) : ℝ := (m - 3) * x + m + 1

-- Define the condition for passing through first, second, and fourth quadrants
def passes_through_quadrants (m : ℝ) : Prop :=
  ∃ (x₁ x₂ x₄ : ℝ), 
    x₁ > 0 ∧ f m x₁ > 0 ∧  -- First quadrant
    x₂ < 0 ∧ f m x₂ > 0 ∧  -- Second quadrant
    x₄ > 0 ∧ f m x₄ < 0    -- Fourth quadrant

-- Theorem statement
theorem range_of_m (m : ℝ) : 
  passes_through_quadrants m → -1 < m ∧ m < 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1561_156159


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l1561_156187

theorem rectangular_plot_breadth (length breadth : ℝ) 
  (h1 : length * breadth = 15 * breadth) 
  (h2 : length - breadth = 10) : 
  breadth = 5 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l1561_156187


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l1561_156192

/-- Systematic sampling function -/
def systematicSample (totalEmployees : ℕ) (sampleSize : ℕ) (firstSample : ℕ) : ℕ → ℕ :=
  fun n => (n - 1) * (totalEmployees / sampleSize) + firstSample

/-- Theorem: In a systematic sampling of 40 samples from 200 employees, 
    if the 5th sample is 22, then the 10th sample is 47 -/
theorem systematic_sampling_theorem 
  (totalEmployees : ℕ) (sampleSize : ℕ) (groupSize : ℕ) (fifthSample : ℕ) :
  totalEmployees = 200 →
  sampleSize = 40 →
  groupSize = 5 →
  fifthSample = 22 →
  systematicSample totalEmployees sampleSize (fifthSample - (5 - 1) * groupSize) 10 = 47 := by
  sorry

#check systematic_sampling_theorem

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l1561_156192


namespace NUMINAMATH_CALUDE_saree_stripes_l1561_156145

theorem saree_stripes (brown_stripes : ℕ) (gold_stripes : ℕ) (blue_stripes : ℕ) 
  (h1 : gold_stripes = 3 * brown_stripes)
  (h2 : blue_stripes = 5 * gold_stripes)
  (h3 : brown_stripes = 4) : 
  blue_stripes = 60 := by
  sorry

end NUMINAMATH_CALUDE_saree_stripes_l1561_156145


namespace NUMINAMATH_CALUDE_fraction_simplification_l1561_156197

theorem fraction_simplification :
  (36 : ℚ) / 19 * 57 / 40 * 95 / 171 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1561_156197


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l1561_156199

/-- The positive difference between the two largest prime factors of 159137 is 14 -/
theorem largest_prime_factors_difference (n : Nat) : n = 159137 → 
  ∃ (p q : Nat), Prime p ∧ Prime q ∧ p ∣ n ∧ q ∣ n ∧ 
  (∀ (r : Nat), Prime r → r ∣ n → r ≤ p) ∧
  (∀ (r : Nat), Prime r → r ∣ n → r ≠ p → r ≤ q) ∧
  p - q = 14 := by
sorry

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l1561_156199


namespace NUMINAMATH_CALUDE_women_doubles_tournament_handshakes_l1561_156137

/-- The number of handshakes in a women's doubles tennis tournament -/
def num_handshakes (num_teams : ℕ) (team_size : ℕ) : ℕ :=
  let total_players := num_teams * team_size
  let handshakes_per_player := total_players - team_size
  (total_players * handshakes_per_player) / 2

theorem women_doubles_tournament_handshakes :
  num_handshakes 4 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_women_doubles_tournament_handshakes_l1561_156137


namespace NUMINAMATH_CALUDE_smallest_distance_between_circles_l1561_156182

theorem smallest_distance_between_circles (z w : ℂ) 
  (hz : Complex.abs (z + 2 + 4*I) = 2)
  (hw : Complex.abs (w - 6 - 7*I) = 4) :
  ∃ (z' w' : ℂ), 
    Complex.abs (z' + 2 + 4*I) = 2 ∧ 
    Complex.abs (w' - 6 - 7*I) = 4 ∧
    Complex.abs (z' - w') = Real.sqrt 185 - 6 ∧
    ∀ (z'' w'' : ℂ), 
      Complex.abs (z'' + 2 + 4*I) = 2 → 
      Complex.abs (w'' - 6 - 7*I) = 4 → 
      Complex.abs (z'' - w'') ≥ Real.sqrt 185 - 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_between_circles_l1561_156182


namespace NUMINAMATH_CALUDE_video_recorder_price_l1561_156103

/-- Given a wholesale cost, markup percentage, and discount percentage,
    calculate the final price after markup and discount. -/
def finalPrice (wholesaleCost markup discount : ℝ) : ℝ :=
  wholesaleCost * (1 + markup) * (1 - discount)

/-- Theorem stating that for a video recorder with a $200 wholesale cost,
    20% markup, and 25% employee discount, the final price is $180. -/
theorem video_recorder_price :
  finalPrice 200 0.20 0.25 = 180 := by
  sorry

end NUMINAMATH_CALUDE_video_recorder_price_l1561_156103


namespace NUMINAMATH_CALUDE_square_perimeter_l1561_156194

theorem square_perimeter (s : ℝ) (h1 : s > 0) : 
  (2 * s + 2 * (s / 5) = 36) → (4 * s = 60) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l1561_156194


namespace NUMINAMATH_CALUDE_mrs_wong_valentines_l1561_156167

def valentines_problem (initial : ℕ) (given_away : ℕ) : Prop :=
  initial - given_away = 22

theorem mrs_wong_valentines : valentines_problem 30 8 := by
  sorry

end NUMINAMATH_CALUDE_mrs_wong_valentines_l1561_156167


namespace NUMINAMATH_CALUDE_B_subset_A_l1561_156118

-- Define set A
def A : Set ℝ := {x : ℝ | |2*x - 3| > 1}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 + x - 6 > 0}

-- Theorem to prove
theorem B_subset_A : B ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_B_subset_A_l1561_156118


namespace NUMINAMATH_CALUDE_min_positive_temperatures_l1561_156146

theorem min_positive_temperatures (x : ℕ) (y : ℕ) : 
  x * (x - 1) = 110 → 
  y * (y - 1) + (x - y) * (x - 1 - y) = 50 → 
  y ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_min_positive_temperatures_l1561_156146


namespace NUMINAMATH_CALUDE_right_triangle_sin_I_l1561_156161

theorem right_triangle_sin_I (G H I : Real) :
  -- GHI is a right triangle with ∠G = 90°
  G + H + I = Real.pi →
  G = Real.pi / 2 →
  -- sin H = 3/5
  Real.sin H = 3 / 5 →
  -- Prove: sin I = 4/5
  Real.sin I = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sin_I_l1561_156161


namespace NUMINAMATH_CALUDE_rachels_age_l1561_156190

/-- Given that Rachel is 4 years older than Leah and the sum of their ages is 34,
    prove that Rachel is 19 years old. -/
theorem rachels_age (rachel_age leah_age : ℕ) 
    (h1 : rachel_age = leah_age + 4)
    (h2 : rachel_age + leah_age = 34) : 
  rachel_age = 19 := by
  sorry

end NUMINAMATH_CALUDE_rachels_age_l1561_156190


namespace NUMINAMATH_CALUDE_reims_to_chaumont_distance_l1561_156140

/-- Represents a city in the polygon -/
inductive City
  | Chalons
  | Vitry
  | Chaumont
  | SaintQuentin
  | Reims

/-- Represents the distance between two cities -/
def distance (a b : City) : ℕ :=
  match a, b with
  | City.Chalons, City.Vitry => 30
  | City.Vitry, City.Chaumont => 80
  | City.Chaumont, City.SaintQuentin => 236
  | City.SaintQuentin, City.Reims => 86
  | City.Reims, City.Chalons => 40
  | _, _ => 0  -- For simplicity, we set other distances to 0

/-- The theorem stating the distance from Reims to Chaumont -/
theorem reims_to_chaumont_distance :
  distance City.Reims City.Chaumont = 150 :=
by sorry

end NUMINAMATH_CALUDE_reims_to_chaumont_distance_l1561_156140


namespace NUMINAMATH_CALUDE_fifteen_point_figures_l1561_156171

def points : ℕ := 15

-- Define a function to calculate combinations
def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Number of quadrilaterals
def quadrilaterals : ℕ := choose points 4

-- Number of triangles
def triangles : ℕ := choose points 3

-- Total number of figures
def total_figures : ℕ := quadrilaterals + triangles

-- Theorem statement
theorem fifteen_point_figures : total_figures = 1820 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_point_figures_l1561_156171


namespace NUMINAMATH_CALUDE_janet_clarinet_lessons_l1561_156158

/-- Proves that Janet takes 3 hours of clarinet lessons per week -/
theorem janet_clarinet_lessons :
  let clarinet_hourly_rate : ℕ := 40
  let piano_hourly_rate : ℕ := 28
  let piano_hours_per_week : ℕ := 5
  let weeks_per_year : ℕ := 52
  let annual_cost_difference : ℕ := 1040
  ∃ (clarinet_hours_per_week : ℕ),
    clarinet_hours_per_week = 3 ∧
    weeks_per_year * piano_hourly_rate * piano_hours_per_week - 
    weeks_per_year * clarinet_hourly_rate * clarinet_hours_per_week = 
    annual_cost_difference :=
by
  sorry

end NUMINAMATH_CALUDE_janet_clarinet_lessons_l1561_156158


namespace NUMINAMATH_CALUDE_min_value_theorem_l1561_156107

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a + 8) * x + a^2 + a - 12

-- Define the theorem
theorem min_value_theorem (a : ℝ) (h1 : a < 0) 
  (h2 : f a (a^2 - 4) = f a (2*a - 8)) :
  ∀ n : ℕ+, (f a n - 4*a) / (n + 1) ≥ 35/8 ∧ 
  ∃ n : ℕ+, (f a n - 4*a) / (n + 1) = 35/8 := by
sorry


end NUMINAMATH_CALUDE_min_value_theorem_l1561_156107


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1561_156196

-- Define the triangle type
structure Triangle where
  inradius : ℝ
  area : ℝ

-- Theorem statement
theorem triangle_perimeter (t : Triangle) (h1 : t.inradius = 2.5) (h2 : t.area = 75) :
  2 * t.area / t.inradius = 60 := by
  sorry

#check triangle_perimeter

end NUMINAMATH_CALUDE_triangle_perimeter_l1561_156196


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1561_156126

/-- 
For a geometric sequence {a_n}, if a_2 + a_4 = 2, 
then a_1a_3 + 2a_2a_4 + a_3a_5 = 4
-/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1) 
  (h_sum : a 2 + a 4 = 2) : 
  a 1 * a 3 + 2 * a 2 * a 4 + a 3 * a 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1561_156126


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1561_156191

/-- Given a complex number z = i(1-i), prove that it corresponds to a point in the first quadrant of the complex plane. -/
theorem complex_number_in_first_quadrant : 
  let z : ℂ := Complex.I * (1 - Complex.I)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1561_156191


namespace NUMINAMATH_CALUDE_square_mod_nine_not_five_l1561_156179

theorem square_mod_nine_not_five (n : ℤ) : n^2 % 9 ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_square_mod_nine_not_five_l1561_156179


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1561_156168

/-- Given an arithmetic sequence {a_n} with common difference 3,
    where a_1, a_2, a_5 form a geometric sequence, prove that a_10 = 57/2 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℚ) :
  (∀ n, a (n + 1) - a n = 3) →  -- arithmetic sequence with common difference 3
  (a 2)^2 = a 1 * a 5 →         -- a_1, a_2, a_5 form a geometric sequence
  a 10 = 57/2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1561_156168


namespace NUMINAMATH_CALUDE_completing_square_l1561_156181

theorem completing_square (x : ℝ) : x^2 - 4*x + 2 = 0 ↔ (x - 2)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_l1561_156181


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1561_156110

/-- Proves that the speed of a boat in still water is 22 km/hr, given that it travels 54 km downstream in 2 hours with a stream speed of 5 km/hr. -/
theorem boat_speed_in_still_water
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : stream_speed = 5)
  (h2 : downstream_distance = 54)
  (h3 : downstream_time = 2)
  : ∃ (still_water_speed : ℝ),
    still_water_speed = 22 ∧
    downstream_distance = (still_water_speed + stream_speed) * downstream_time :=
by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1561_156110


namespace NUMINAMATH_CALUDE_quadratic_increasing_condition_l1561_156174

/-- A quadratic function f(x) = x^2 + 2mx + 10 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*m*x + 10

/-- The function is increasing on [2, +∞) -/
def increasing_on_interval (m : ℝ) : Prop :=
  ∀ x y, 2 ≤ x → x < y → f m x < f m y

theorem quadratic_increasing_condition (m : ℝ) :
  increasing_on_interval m → m ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_increasing_condition_l1561_156174


namespace NUMINAMATH_CALUDE_seven_sum_problem_l1561_156105

theorem seven_sum_problem :
  ∃ (S : Finset ℕ), (Finset.card S = 108) ∧ 
  (∀ n : ℕ, n ∈ S ↔ 
    ∃ a b c : ℕ, (7 * a + 77 * b + 777 * c = 7000) ∧ 
                 (a + 2 * b + 3 * c = n)) :=
sorry

end NUMINAMATH_CALUDE_seven_sum_problem_l1561_156105


namespace NUMINAMATH_CALUDE_power_division_nineteen_l1561_156124

theorem power_division_nineteen : 19^12 / 19^10 = 361 := by
  sorry

end NUMINAMATH_CALUDE_power_division_nineteen_l1561_156124


namespace NUMINAMATH_CALUDE_square_difference_equals_648_l1561_156131

theorem square_difference_equals_648 : (36 + 9)^2 - (9^2 + 36^2) = 648 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_648_l1561_156131


namespace NUMINAMATH_CALUDE_stream_speed_l1561_156108

/-- Proves that given a boat with a speed of 22 km/hr in still water, 
    traveling 54 km downstream in 2 hours, the speed of the stream is 5 km/hr. -/
theorem stream_speed 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : boat_speed = 22)
  (h2 : distance = 54)
  (h3 : time = 2)
  : ∃ (stream_speed : ℝ), 
    distance = (boat_speed + stream_speed) * time ∧ 
    stream_speed = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l1561_156108


namespace NUMINAMATH_CALUDE_oil_purchase_amount_l1561_156155

/-- Represents the price and quantity of oil before and after a price reduction --/
structure OilPurchase where
  original_price : ℝ
  reduced_price : ℝ
  original_quantity : ℝ
  additional_quantity : ℝ
  price_reduction_percent : ℝ

/-- Calculates the total amount spent on oil after the price reduction --/
def total_spent (purchase : OilPurchase) : ℝ :=
  purchase.reduced_price * (purchase.original_quantity + purchase.additional_quantity)

/-- Theorem stating the total amount spent on oil after the price reduction --/
theorem oil_purchase_amount (purchase : OilPurchase) 
  (h1 : purchase.price_reduction_percent = 25)
  (h2 : purchase.additional_quantity = 5)
  (h3 : purchase.reduced_price = 60)
  (h4 : purchase.reduced_price = purchase.original_price * (1 - purchase.price_reduction_percent / 100)) :
  total_spent purchase = 1200 := by
  sorry

#eval total_spent { original_price := 80, reduced_price := 60, original_quantity := 15, additional_quantity := 5, price_reduction_percent := 25 }

end NUMINAMATH_CALUDE_oil_purchase_amount_l1561_156155


namespace NUMINAMATH_CALUDE_euler_totient_equation_solution_l1561_156128

def euler_totient (n : ℕ) : ℕ := sorry

theorem euler_totient_equation_solution :
  ∀ n : ℕ, n > 0 → (n = euler_totient n + 402 ↔ n = 802 ∨ n = 546) := by
  sorry

end NUMINAMATH_CALUDE_euler_totient_equation_solution_l1561_156128


namespace NUMINAMATH_CALUDE_linear_function_point_relation_l1561_156170

/-- Given a linear function y = -x + 6 and two points A(-1, y₁) and B(2, y₂) on its graph, prove that y₁ > y₂ -/
theorem linear_function_point_relation (y₁ y₂ : ℝ) : 
  (∀ x : ℝ, -x + 6 = y₁ → x = -1) →  -- Point A(-1, y₁) is on the graph
  (∀ x : ℝ, -x + 6 = y₂ → x = 2) →   -- Point B(2, y₂) is on the graph
  y₁ > y₂ := by
sorry

end NUMINAMATH_CALUDE_linear_function_point_relation_l1561_156170


namespace NUMINAMATH_CALUDE_wednesday_thursday_miles_l1561_156120

/-- Represents the mileage reimbursement rate in dollars per mile -/
def reimbursement_rate : ℚ := 36 / 100

/-- Represents the total reimbursement amount in dollars -/
def total_reimbursement : ℚ := 36

/-- Represents the miles driven on Monday -/
def monday_miles : ℕ := 18

/-- Represents the miles driven on Tuesday -/
def tuesday_miles : ℕ := 26

/-- Represents the miles driven on Friday -/
def friday_miles : ℕ := 16

/-- Theorem stating that the miles driven on Wednesday and Thursday combined is 40 -/
theorem wednesday_thursday_miles : 
  (total_reimbursement / reimbursement_rate : ℚ) - 
  (monday_miles + tuesday_miles + friday_miles : ℚ) = 40 := by sorry

end NUMINAMATH_CALUDE_wednesday_thursday_miles_l1561_156120


namespace NUMINAMATH_CALUDE_power_equation_l1561_156130

theorem power_equation (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(2*m + n) = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_l1561_156130


namespace NUMINAMATH_CALUDE_one_dollar_bills_count_l1561_156138

/-- Represents the number of bills of each denomination -/
structure WalletContent where
  ones : ℕ
  twos : ℕ
  fives : ℕ

/-- Calculates the total number of bills -/
def total_bills (w : WalletContent) : ℕ :=
  w.ones + w.twos + w.fives

/-- Calculates the total amount of money -/
def total_money (w : WalletContent) : ℕ :=
  w.ones + 2 * w.twos + 5 * w.fives

/-- Theorem stating that given the conditions, the number of one dollar bills is 20 -/
theorem one_dollar_bills_count (w : WalletContent) 
  (h1 : total_bills w = 60) 
  (h2 : total_money w = 120) : 
  w.ones = 20 := by
  sorry

end NUMINAMATH_CALUDE_one_dollar_bills_count_l1561_156138


namespace NUMINAMATH_CALUDE_base_10_729_equals_base_7_2061_l1561_156112

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Theorem: 729 in base-10 is equal to 2061 in base-7 --/
theorem base_10_729_equals_base_7_2061 :
  729 = base7ToBase10 [1, 6, 0, 2] := by
  sorry

end NUMINAMATH_CALUDE_base_10_729_equals_base_7_2061_l1561_156112


namespace NUMINAMATH_CALUDE_factorial_equation_sum_l1561_156121

theorem factorial_equation_sum : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, ∃ k l : ℕ, n.factorial / 2 = k.factorial + l.factorial) ∧
  (∀ n : ℕ, n ∉ S → ¬∃ k l : ℕ, n.factorial / 2 = k.factorial + l.factorial) ∧
  S.sum id = 10 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_sum_l1561_156121


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1561_156150

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1561_156150


namespace NUMINAMATH_CALUDE_sum_denominator_power_of_two_l1561_156125

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

def sum_series : ℕ → ℚ
  | 0 => 0
  | n + 1 => sum_series n + (double_factorial (2 * (n + 1) - 1) : ℚ) / (double_factorial (2 * (n + 1)) : ℚ)

theorem sum_denominator_power_of_two : 
  ∃ (numerator : ℕ), sum_series 11 = (numerator : ℚ) / 2^8 := by sorry

end NUMINAMATH_CALUDE_sum_denominator_power_of_two_l1561_156125


namespace NUMINAMATH_CALUDE_intersection_trisection_l1561_156134

/-- A line y = mx + b intersecting a circle and a hyperbola -/
structure IntersectingLine where
  m : ℝ
  b : ℝ
  h_m : |m| < 1
  h_b : |b| < 1

/-- Points of intersection with the circle x^2 + y^2 = 1 -/
def circle_intersection (l : IntersectingLine) : Set (ℝ × ℝ) :=
  {(x, y) | y = l.m * x + l.b ∧ x^2 + y^2 = 1}

/-- Points of intersection with the hyperbola x^2 - y^2 = 1 -/
def hyperbola_intersection (l : IntersectingLine) : Set (ℝ × ℝ) :=
  {(x, y) | y = l.m * x + l.b ∧ x^2 - y^2 = 1}

/-- Trisection property of the intersection points -/
def trisects (P Q R S : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t ∈ Set.Icc (0 : ℝ) 1 ∧
    P = (1 - t) • R + t • S ∧
    Q = (1 - t) • S + t • R ∧
    t = 1/3 ∨ t = 2/3

/-- Main theorem: Intersection points trisect implies specific values for m and b -/
theorem intersection_trisection (l : IntersectingLine)
  (hP : P ∈ circle_intersection l) (hQ : Q ∈ circle_intersection l)
  (hR : R ∈ hyperbola_intersection l) (hS : S ∈ hyperbola_intersection l)
  (h_trisect : trisects P Q R S) :
  (l.m = 0 ∧ l.b = 2/5 * Real.sqrt 5) ∨
  (l.m = 0 ∧ l.b = -2/5 * Real.sqrt 5) ∨
  (l.m = 2/5 * Real.sqrt 5 ∧ l.b = 0) ∨
  (l.m = -2/5 * Real.sqrt 5 ∧ l.b = 0) :=
sorry

end NUMINAMATH_CALUDE_intersection_trisection_l1561_156134


namespace NUMINAMATH_CALUDE_jacob_breakfast_calories_l1561_156143

theorem jacob_breakfast_calories 
  (daily_limit : ℕ) 
  (lunch_calories : ℕ) 
  (dinner_calories : ℕ) 
  (exceeded_calories : ℕ) 
  (h1 : daily_limit = 1800)
  (h2 : lunch_calories = 900)
  (h3 : dinner_calories = 1100)
  (h4 : exceeded_calories = 600) :
  daily_limit + exceeded_calories - (lunch_calories + dinner_calories) = 400 := by
sorry

end NUMINAMATH_CALUDE_jacob_breakfast_calories_l1561_156143


namespace NUMINAMATH_CALUDE_prob_even_heads_is_17_25_l1561_156147

/-- Represents an unfair coin where the probability of heads is 4 times the probability of tails -/
structure UnfairCoin where
  p_tails : ℝ
  p_heads : ℝ
  p_tails_pos : 0 < p_tails
  p_heads_pos : 0 < p_heads
  p_sum_one : p_tails + p_heads = 1
  p_heads_four_times : p_heads = 4 * p_tails

/-- The probability of getting an even number of heads when flipping the unfair coin twice -/
def prob_even_heads (c : UnfairCoin) : ℝ :=
  c.p_tails^2 + c.p_heads^2

/-- Theorem stating that the probability of getting an even number of heads
    when flipping the unfair coin twice is 17/25 -/
theorem prob_even_heads_is_17_25 (c : UnfairCoin) :
  prob_even_heads c = 17/25 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_heads_is_17_25_l1561_156147


namespace NUMINAMATH_CALUDE_parabola_translation_l1561_156144

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  equation : ℝ → ℝ

/-- Represents a translation in the xy-plane -/
structure Translation where
  horizontal : ℝ
  vertical : ℝ

/-- Applies a translation to a parabola -/
def apply_translation (p : Parabola) (t : Translation) : Parabola :=
  { equation := fun x => p.equation (x - t.horizontal) + t.vertical }

theorem parabola_translation :
  let original : Parabola := { equation := fun x => x^2 }
  let translation : Translation := { horizontal := 3, vertical := -4 }
  let transformed := apply_translation original translation
  ∀ x, transformed.equation x = (x + 3)^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l1561_156144


namespace NUMINAMATH_CALUDE_cupcake_packages_l1561_156113

theorem cupcake_packages (initial_cupcakes : ℕ) (eaten_cupcakes : ℕ) (cupcakes_per_package : ℕ) :
  initial_cupcakes = 39 →
  eaten_cupcakes = 21 →
  cupcakes_per_package = 3 →
  (initial_cupcakes - eaten_cupcakes) / cupcakes_per_package = 6 :=
by sorry

end NUMINAMATH_CALUDE_cupcake_packages_l1561_156113


namespace NUMINAMATH_CALUDE_f_passes_through_origin_l1561_156176

def f (x : ℝ) : ℝ := -2 * x

theorem f_passes_through_origin : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_passes_through_origin_l1561_156176


namespace NUMINAMATH_CALUDE_range_of_a_and_m_l1561_156104

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + a - 1 = 0}
def C (m : ℝ) : Set ℝ := {x | x^2 - m*x + 2 = 0}

theorem range_of_a_and_m :
  ∀ (a m : ℝ), A ∪ B a = A → A ∩ C m = C m →
  (a = 2 ∨ a = 3) ∧ (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_and_m_l1561_156104


namespace NUMINAMATH_CALUDE_plant_growth_theorem_l1561_156117

structure PlantType where
  seedsPerPacket : ℕ
  growthRate : ℕ
  initialPackets : ℕ

def totalPlants (p : PlantType) : ℕ :=
  p.seedsPerPacket * p.initialPackets

def additionalPacketsNeeded (p : PlantType) (targetPlants : ℕ) : ℕ :=
  max 0 ((targetPlants - totalPlants p + p.seedsPerPacket - 1) / p.seedsPerPacket)

def growthTime (p : PlantType) (targetPlants : ℕ) : ℕ :=
  p.growthRate * max 0 (targetPlants - totalPlants p)

theorem plant_growth_theorem (targetPlants : ℕ) 
  (typeA typeB typeC : PlantType)
  (h1 : typeA = { seedsPerPacket := 3, growthRate := 5, initialPackets := 2 })
  (h2 : typeB = { seedsPerPacket := 6, growthRate := 7, initialPackets := 3 })
  (h3 : typeC = { seedsPerPacket := 9, growthRate := 4, initialPackets := 3 })
  (h4 : targetPlants = 12) : 
  additionalPacketsNeeded typeA targetPlants = 2 ∧ 
  growthTime typeA targetPlants = 5 ∧
  additionalPacketsNeeded typeB targetPlants = 0 ∧
  additionalPacketsNeeded typeC targetPlants = 0 := by
  sorry

end NUMINAMATH_CALUDE_plant_growth_theorem_l1561_156117


namespace NUMINAMATH_CALUDE_boxer_weight_loss_l1561_156163

/-- Given a boxer's initial weight, monthly weight loss, and number of months until the fight,
    calculate the boxer's weight on the day of the fight. -/
def boxerFinalWeight (initialWeight monthlyLoss months : ℕ) : ℕ :=
  initialWeight - monthlyLoss * months

/-- Theorem stating that a boxer weighing 97 kg and losing 3 kg per month for 4 months
    will weigh 85 kg on the day of the fight. -/
theorem boxer_weight_loss : boxerFinalWeight 97 3 4 = 85 := by
  sorry

end NUMINAMATH_CALUDE_boxer_weight_loss_l1561_156163


namespace NUMINAMATH_CALUDE_sphere_surface_area_ratio_l1561_156141

theorem sphere_surface_area_ratio (r₁ r₂ : ℝ) (h : (4 / 3 * Real.pi * r₁^3) / (4 / 3 * Real.pi * r₂^3) = 1 / 27) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_ratio_l1561_156141


namespace NUMINAMATH_CALUDE_boys_in_class_l1561_156175

theorem boys_in_class (total : ℕ) (girls_more : ℕ) (boys : ℕ) 
  (h1 : total = 485) 
  (h2 : girls_more = 69) 
  (h3 : total = boys + (boys + girls_more)) : 
  boys = 208 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_class_l1561_156175


namespace NUMINAMATH_CALUDE_total_cakes_served_l1561_156132

theorem total_cakes_served (lunch_cakes dinner_cakes : ℕ) 
  (h1 : lunch_cakes = 6) 
  (h2 : dinner_cakes = 9) : 
  lunch_cakes + dinner_cakes = 15 := by
sorry

end NUMINAMATH_CALUDE_total_cakes_served_l1561_156132


namespace NUMINAMATH_CALUDE_intersection_condition_l1561_156129

/-- The set M in ℝ² defined by y ≥ x² -/
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 ≥ p.1^2}

/-- The set N in ℝ² defined by x² + (y-a)² ≤ 1 -/
def N (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + (p.2 - a)^2 ≤ 1}

/-- Theorem stating the necessary and sufficient condition for M ∩ N = N -/
theorem intersection_condition (a : ℝ) : M ∩ N a = N a ↔ a ≥ 5/4 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_l1561_156129


namespace NUMINAMATH_CALUDE_nine_sevenths_to_fourth_l1561_156139

theorem nine_sevenths_to_fourth (x : ℚ) : x = 9 * (1 / 7)^4 → x = 9 / 2401 := by
  sorry

end NUMINAMATH_CALUDE_nine_sevenths_to_fourth_l1561_156139
