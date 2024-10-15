import Mathlib

namespace NUMINAMATH_CALUDE_abes_age_l1474_147429

theorem abes_age (present_age : ℕ) : 
  present_age + (present_age - 7) = 35 → present_age = 21 :=
by sorry

end NUMINAMATH_CALUDE_abes_age_l1474_147429


namespace NUMINAMATH_CALUDE_candy_duration_l1474_147462

theorem candy_duration (neighbors_candy : ℝ) (sister_candy : ℝ) (daily_consumption : ℝ) :
  neighbors_candy = 11.0 →
  sister_candy = 5.0 →
  daily_consumption = 8.0 →
  (neighbors_candy + sister_candy) / daily_consumption = 2.0 := by
  sorry

end NUMINAMATH_CALUDE_candy_duration_l1474_147462


namespace NUMINAMATH_CALUDE_store_sale_profit_store_sale_result_l1474_147466

/-- Calculates the money left after a store's inventory sale --/
theorem store_sale_profit (total_items : ℕ) (retail_price : ℚ) (discount_percent : ℚ) 
  (sold_percent : ℚ) (debt : ℚ) : ℚ :=
  let items_sold := total_items * sold_percent
  let discount_amount := retail_price * discount_percent
  let sale_price := retail_price - discount_amount
  let total_revenue := items_sold * sale_price
  let profit := total_revenue - debt
  profit

/-- Proves that the store has $3000 left after the sale --/
theorem store_sale_result : 
  store_sale_profit 2000 50 0.8 0.9 15000 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_store_sale_profit_store_sale_result_l1474_147466


namespace NUMINAMATH_CALUDE_unit_square_quadrilateral_inequalities_l1474_147449

/-- A quadrilateral formed by selecting one point on each side of a unit square -/
structure UnitSquareQuadrilateral where
  a : Real
  b : Real
  c : Real
  d : Real
  a_nonneg : 0 ≤ a
  b_nonneg : 0 ≤ b
  c_nonneg : 0 ≤ c
  d_nonneg : 0 ≤ d
  a_le_one : a ≤ 1
  b_le_one : b ≤ 1
  c_le_one : c ≤ 1
  d_le_one : d ≤ 1

theorem unit_square_quadrilateral_inequalities (q : UnitSquareQuadrilateral) :
  2 ≤ q.a ^ 2 + q.b ^ 2 + q.c ^ 2 + q.d ^ 2 ∧
  q.a ^ 2 + q.b ^ 2 + q.c ^ 2 + q.d ^ 2 ≤ 4 ∧
  2 * Real.sqrt 2 ≤ q.a + q.b + q.c + q.d ∧
  q.a + q.b + q.c + q.d ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_unit_square_quadrilateral_inequalities_l1474_147449


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1474_147493

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x > 2 → x > 0) ∧ 
  (∃ x : ℝ, x > 0 ∧ ¬(x > 2)) := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1474_147493


namespace NUMINAMATH_CALUDE_classroom_gpa_l1474_147496

theorem classroom_gpa (N : ℝ) (h : N > 0) :
  let gpa_one_third := 54
  let gpa_whole := 48
  let gpa_rest := (3 * gpa_whole - gpa_one_third) / 2
  gpa_rest = 45 := by sorry

end NUMINAMATH_CALUDE_classroom_gpa_l1474_147496


namespace NUMINAMATH_CALUDE_grants_test_score_l1474_147410

theorem grants_test_score (hunter_score john_score grant_score : ℕ) :
  hunter_score = 45 →
  john_score = 2 * hunter_score →
  grant_score = john_score + 10 →
  grant_score = 100 := by
sorry

end NUMINAMATH_CALUDE_grants_test_score_l1474_147410


namespace NUMINAMATH_CALUDE_sum_of_special_integers_l1474_147416

/-- A positive integer with exactly two positive divisors -/
def smallest_two_divisor_integer : ℕ+ := sorry

/-- The largest integer less than 150 with exactly three positive divisors -/
def largest_three_divisor_integer_below_150 : ℕ+ := sorry

/-- The sum of the smallest integer with two positive divisors and 
    the largest integer less than 150 with three positive divisors -/
theorem sum_of_special_integers : 
  (smallest_two_divisor_integer : ℕ) + (largest_three_divisor_integer_below_150 : ℕ) = 123 := by sorry

end NUMINAMATH_CALUDE_sum_of_special_integers_l1474_147416


namespace NUMINAMATH_CALUDE_triangle_expression_simplification_l1474_147475

-- Define a triangle with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  -- Triangle inequality conditions
  ab_gt_c : a + b > c
  ac_gt_b : a + c > b
  bc_gt_a : b + c > a

-- Define the theorem
theorem triangle_expression_simplification (t : Triangle) :
  |t.a - t.b - t.c| + |t.b - t.a - t.c| - |t.c - t.a + t.b| = t.a - t.b + t.c :=
by sorry

end NUMINAMATH_CALUDE_triangle_expression_simplification_l1474_147475


namespace NUMINAMATH_CALUDE_salary_calculation_l1474_147421

/-- Represents the monthly salary in Rupees -/
def monthly_salary : ℝ := 1375

/-- Represents the savings rate as a decimal -/
def savings_rate : ℝ := 0.20

/-- Represents the expense increase rate as a decimal -/
def expense_increase_rate : ℝ := 0.20

/-- Represents the new savings amount after expense increase in Rupees -/
def new_savings : ℝ := 220

theorem salary_calculation :
  monthly_salary * savings_rate * (1 - expense_increase_rate) = new_savings :=
by sorry

end NUMINAMATH_CALUDE_salary_calculation_l1474_147421


namespace NUMINAMATH_CALUDE_polynomial_root_sum_product_l1474_147446

theorem polynomial_root_sum_product (c d : ℂ) : 
  (c^4 - 6*c - 3 = 0) → 
  (d^4 - 6*d - 3 = 0) → 
  (c*d + c + d = 3 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_product_l1474_147446


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1474_147450

/-- An arithmetic sequence {a_n} with specific conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 3 = 7 ∧
  a 5 = a 2 + 6

/-- The general term formula for the arithmetic sequence -/
def general_term (n : ℕ) : ℝ := 2 * n + 1

/-- Theorem stating that the general term formula is correct for the given arithmetic sequence -/
theorem arithmetic_sequence_general_term (a : ℕ → ℝ) :
  arithmetic_sequence a → ∀ n : ℕ, a n = general_term n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1474_147450


namespace NUMINAMATH_CALUDE_cherry_price_proof_l1474_147403

-- Define the discount rate
def discount_rate : ℝ := 0.3

-- Define the discounted price for a quarter-pound package
def discounted_quarter_pound_price : ℝ := 2

-- Define the weight of a full pound in terms of quarter-pounds
def full_pound_weight : ℝ := 4

-- Define the regular price for a full pound of cherries
def regular_full_pound_price : ℝ := 11.43

theorem cherry_price_proof :
  (1 - discount_rate) * regular_full_pound_price / full_pound_weight = discounted_quarter_pound_price := by
  sorry

end NUMINAMATH_CALUDE_cherry_price_proof_l1474_147403


namespace NUMINAMATH_CALUDE_limit_of_sequence_a_l1474_147435

def a (n : ℕ) : ℚ := (4*n - 3) / (2*n + 1)

theorem limit_of_sequence_a : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 2| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_sequence_a_l1474_147435


namespace NUMINAMATH_CALUDE_ad_transmission_cost_l1474_147402

/-- The cost of transmitting advertisements during a race -/
theorem ad_transmission_cost
  (num_ads : ℕ)
  (ad_duration : ℕ)
  (cost_per_minute : ℕ)
  (h1 : num_ads = 5)
  (h2 : ad_duration = 3)
  (h3 : cost_per_minute = 4000) :
  num_ads * ad_duration * cost_per_minute = 60000 :=
by sorry

end NUMINAMATH_CALUDE_ad_transmission_cost_l1474_147402


namespace NUMINAMATH_CALUDE_different_color_probability_l1474_147432

/-- The probability of drawing two chips of different colors from a bag containing 
    7 red chips and 4 green chips, when drawing with replacement. -/
theorem different_color_probability :
  let total_chips : ℕ := 7 + 4
  let red_chips : ℕ := 7
  let green_chips : ℕ := 4
  let prob_red : ℚ := red_chips / total_chips
  let prob_green : ℚ := green_chips / total_chips
  prob_red * prob_green + prob_green * prob_red = 56 / 121 := by
sorry

end NUMINAMATH_CALUDE_different_color_probability_l1474_147432


namespace NUMINAMATH_CALUDE_square_gt_abs_l1474_147481

theorem square_gt_abs (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_gt_abs_l1474_147481


namespace NUMINAMATH_CALUDE_area_is_14_4_l1474_147467

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoidWithInscribedCircle where
  /-- Distance from circle center to one end of a non-parallel side -/
  d1 : ℝ
  /-- Distance from circle center to the other end of the same non-parallel side -/
  d2 : ℝ
  /-- Assumption that d1 and d2 are positive -/
  d1_pos : d1 > 0
  d2_pos : d2 > 0

/-- The area of the isosceles trapezoid with an inscribed circle -/
def area (t : IsoscelesTrapezoidWithInscribedCircle) : ℝ :=
  14.4

/-- Theorem stating that the area of the isosceles trapezoid with an inscribed circle is 14.4 cm² -/
theorem area_is_14_4 (t : IsoscelesTrapezoidWithInscribedCircle) 
    (h1 : t.d1 = 2) (h2 : t.d2 = 4) : area t = 14.4 := by
  sorry

end NUMINAMATH_CALUDE_area_is_14_4_l1474_147467


namespace NUMINAMATH_CALUDE_ball_box_theorem_l1474_147428

/-- Given a box with 60 balls where the probability of picking a white ball is 0.25,
    this theorem proves the number of white and black balls, and the number of
    additional white balls needed to change the probability to 2/5. -/
theorem ball_box_theorem (total_balls : ℕ) (prob_white : ℚ) :
  total_balls = 60 →
  prob_white = 1/4 →
  ∃ (white_balls black_balls additional_balls : ℕ),
    white_balls = 15 ∧
    black_balls = 45 ∧
    additional_balls = 15 ∧
    white_balls + black_balls = total_balls ∧
    (white_balls : ℚ) / total_balls = prob_white ∧
    ((white_balls + additional_balls : ℚ) / (total_balls + additional_balls) = 2/5) :=
by sorry

end NUMINAMATH_CALUDE_ball_box_theorem_l1474_147428


namespace NUMINAMATH_CALUDE_sum_of_ages_age_difference_l1474_147456

/-- Tyler's age -/
def tyler_age : ℕ := 7

/-- Tyler's brother's age -/
def brother_age : ℕ := 11 - tyler_age

/-- The sum of Tyler's and his brother's ages -/
theorem sum_of_ages : tyler_age + brother_age = 11 := by sorry

/-- The difference between Tyler's brother's age and Tyler's age -/
theorem age_difference : brother_age - tyler_age = 4 := by sorry

end NUMINAMATH_CALUDE_sum_of_ages_age_difference_l1474_147456


namespace NUMINAMATH_CALUDE_range_x_when_p_false_range_m_when_p_sufficient_for_q_l1474_147415

-- Define propositions p and q
def p (x : ℝ) : Prop := |x - 3| < 1
def q (x m : ℝ) : Prop := m - 2 < x ∧ x < m + 1

-- Part 1: Range of x when p is false
theorem range_x_when_p_false (x : ℝ) :
  ¬(p x) → x ≤ 2 ∨ x ≥ 4 :=
sorry

-- Part 2: Range of m when p is a sufficient condition for q
theorem range_m_when_p_sufficient_for_q (m : ℝ) :
  (∀ x, p x → q x m) → 3 ≤ m ∧ m ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_range_x_when_p_false_range_m_when_p_sufficient_for_q_l1474_147415


namespace NUMINAMATH_CALUDE_ellipse_foci_l1474_147458

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop := 2 * x^2 + y^2 = 8

/-- The foci coordinates -/
def foci_coordinates : Set (ℝ × ℝ) := {(0, 2), (0, -2)}

/-- Theorem: The foci of the ellipse 2x^2 + y^2 = 8 are at (0, ±2) -/
theorem ellipse_foci :
  ∀ (f : ℝ × ℝ), f ∈ foci_coordinates ↔ 
  (∃ (a b c : ℝ), 
    (∀ x y, ellipse_equation x y ↔ (x^2 / a^2 + y^2 / b^2 = 1)) ∧
    (a > b) ∧
    (c^2 = a^2 - b^2) ∧
    (f = (0, c) ∨ f = (0, -c))) :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_l1474_147458


namespace NUMINAMATH_CALUDE_function_properties_l1474_147469

noncomputable def f (x φ : ℝ) : ℝ := Real.cos (2 * x + φ)

theorem function_properties (φ : ℝ) (h : φ > 0) :
  (∀ x, f x φ = f (x + π) φ) ∧ 
  (∃ φ', ∀ x, f x φ' = f (-x) φ') ∧
  (∀ x ∈ Set.Icc (π - φ/2) (3*π/2 - φ/2), ∀ y ∈ Set.Icc (π - φ/2) (3*π/2 - φ/2), 
    x < y → f x φ > f y φ) ∧
  (∀ x, f x φ = Real.cos (2 * (x - φ/2))) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1474_147469


namespace NUMINAMATH_CALUDE_square_difference_formula_l1474_147486

theorem square_difference_formula (x y : ℚ) 
  (h1 : x + y = 8/15) (h2 : x - y = 2/15) : x^2 - y^2 = 16/225 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_formula_l1474_147486


namespace NUMINAMATH_CALUDE_product_of_ratios_l1474_147499

theorem product_of_ratios (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 4*x₁*y₁^2 = 3003)
  (h₂ : y₁^3 - 4*x₁^2*y₁ = 3002)
  (h₃ : x₂^3 - 4*x₂*y₂^2 = 3003)
  (h₄ : y₂^3 - 4*x₂^2*y₂ = 3002)
  (h₅ : x₃^3 - 4*x₃*y₃^2 = 3003)
  (h₆ : y₃^3 - 4*x₃^2*y₃ = 3002) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 3/3002 := by
sorry

end NUMINAMATH_CALUDE_product_of_ratios_l1474_147499


namespace NUMINAMATH_CALUDE_men_count_l1474_147494

/-- The number of women in the arrangement -/
def num_women : ℕ := 2

/-- The number of distinct alternating arrangements -/
def num_arrangements : ℕ := 4

/-- A function that calculates the number of distinct alternating arrangements
    given the number of men and women -/
def calc_arrangements (men women : ℕ) : ℕ := sorry

theorem men_count :
  ∃ (men : ℕ), men > 0 ∧ calc_arrangements men num_women = num_arrangements :=
sorry

end NUMINAMATH_CALUDE_men_count_l1474_147494


namespace NUMINAMATH_CALUDE_club_truncator_probability_l1474_147490

/-- The number of matches played by Club Truncator -/
def num_matches : ℕ := 8

/-- The probability of winning, losing, or tying a single match -/
def single_match_prob : ℚ := 1/3

/-- The probability of finishing with more wins than losses -/
def more_wins_prob : ℚ := 5483/13122

theorem club_truncator_probability :
  (num_matches = 8) →
  (single_match_prob = 1/3) →
  (more_wins_prob = 5483/13122) :=
by sorry

end NUMINAMATH_CALUDE_club_truncator_probability_l1474_147490


namespace NUMINAMATH_CALUDE_four_lattice_points_l1474_147413

-- Define the equation
def equation (x y : ℤ) : Prop := x^2 - y^2 = 53

-- Define a lattice point as a pair of integers
def LatticePoint : Type := ℤ × ℤ

-- Define a function to check if a lattice point satisfies the equation
def satisfies_equation (p : LatticePoint) : Prop :=
  equation p.1 p.2

-- Theorem: There are exactly 4 lattice points satisfying the equation
theorem four_lattice_points : 
  ∃! (s : Finset LatticePoint), (∀ p ∈ s, satisfies_equation p) ∧ s.card = 4 :=
sorry

end NUMINAMATH_CALUDE_four_lattice_points_l1474_147413


namespace NUMINAMATH_CALUDE_ellen_lego_problem_l1474_147411

/-- Represents Ellen's Lego collection and calculations -/
theorem ellen_lego_problem (initial : ℕ) (lost : ℕ) (found : ℕ) 
  (h1 : initial = 12560) (h2 : lost = 478) (h3 : found = 342) :
  let current := initial - lost + found
  (current = 12424) ∧ 
  (((lost : ℚ) / (initial : ℚ)) * 100 = 381 / 100) := by
  sorry

#check ellen_lego_problem

end NUMINAMATH_CALUDE_ellen_lego_problem_l1474_147411


namespace NUMINAMATH_CALUDE_tan_sum_reciprocal_l1474_147470

theorem tan_sum_reciprocal (u v : ℝ) 
  (h1 : (Real.sin u / Real.cos v) + (Real.sin v / Real.cos u) = 2)
  (h2 : (Real.cos u / Real.sin v) + (Real.cos v / Real.sin u) = 3) :
  (Real.tan u / Real.tan v) + (Real.tan v / Real.tan u) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_reciprocal_l1474_147470


namespace NUMINAMATH_CALUDE_percentage_equation_solution_l1474_147498

theorem percentage_equation_solution : 
  ∃ x : ℝ, 45 * x = (35 / 100) * 900 ∧ x = 7 := by sorry

end NUMINAMATH_CALUDE_percentage_equation_solution_l1474_147498


namespace NUMINAMATH_CALUDE_base_6_divisibility_l1474_147471

def base_6_to_10 (a b c d : ℕ) : ℕ := a * 6^3 + b * 6^2 + c * 6 + d

theorem base_6_divisibility :
  ∃! (d : ℕ), d < 6 ∧ (base_6_to_10 3 d d 7) % 13 = 0 :=
sorry

end NUMINAMATH_CALUDE_base_6_divisibility_l1474_147471


namespace NUMINAMATH_CALUDE_simplify_expression_l1474_147445

theorem simplify_expression (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (1 + 1 / (x - 2)) / ((x^2 - 2*x + 1) / (x^2 - 4)) = (x + 2) / (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1474_147445


namespace NUMINAMATH_CALUDE_order_of_a_b_c_l1474_147401

theorem order_of_a_b_c :
  let a := (2 : ℝ) ^ (9/10)
  let b := (3 : ℝ) ^ (2/3)
  let c := Real.log 3 / Real.log (1/2)
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_order_of_a_b_c_l1474_147401


namespace NUMINAMATH_CALUDE_probability_of_odd_product_l1474_147480

def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {0, 1, 3}

def is_product_odd (a b : ℕ) : Bool := (a * b) % 2 = 1

def favorable_outcomes : Finset (ℕ × ℕ) :=
  A.product B |>.filter (fun (a, b) => is_product_odd a b)

theorem probability_of_odd_product :
  (favorable_outcomes.card : ℚ) / ((A.card * B.card) : ℚ) = 4 / 9 := by
  sorry

#eval favorable_outcomes -- To check the favorable outcomes
#eval favorable_outcomes.card -- To check the number of favorable outcomes
#eval A.card * B.card -- To check the total number of outcomes

end NUMINAMATH_CALUDE_probability_of_odd_product_l1474_147480


namespace NUMINAMATH_CALUDE_jim_driven_distance_l1474_147434

theorem jim_driven_distance (total_journey : ℕ) (remaining : ℕ) (driven : ℕ) : 
  total_journey = 1200 →
  remaining = 432 →
  driven = total_journey - remaining →
  driven = 768 := by
sorry

end NUMINAMATH_CALUDE_jim_driven_distance_l1474_147434


namespace NUMINAMATH_CALUDE_reflection_of_circle_center_l1474_147430

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_y_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem reflection_of_circle_center :
  let original_center : ℝ × ℝ := (8, -3)
  let reflected_center : ℝ × ℝ := reflect_about_y_neg_x original_center
  reflected_center = (3, -8) := by sorry

end NUMINAMATH_CALUDE_reflection_of_circle_center_l1474_147430


namespace NUMINAMATH_CALUDE_find_number_l1474_147440

theorem find_number : ∃ x : ℚ, x - (3/5) * x = 62 ∧ x = 155 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1474_147440


namespace NUMINAMATH_CALUDE_fraction_calculation_l1474_147424

theorem fraction_calculation : (0.5 ^ 4) / (0.05 ^ 3) = 500 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l1474_147424


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1474_147419

-- Define variables
variable (x y : ℝ)

-- Theorem for the first expression
theorem simplify_expression_1 : (2*x + 1) - (3 - x) = 3*x - 2 := by sorry

-- Theorem for the second expression
theorem simplify_expression_2 : x^2*y - (2*x*y^2 - 5*x^2*y) + 3*x*y^2 - y^3 = 6*x^2*y + x*y^2 - y^3 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1474_147419


namespace NUMINAMATH_CALUDE_square_sum_equality_l1474_147412

-- Define the problem statement
theorem square_sum_equality (x y : ℝ) 
  (h1 : y + 9 = (x - 3)^2) 
  (h2 : x + 9 = (y - 3)^2) 
  (h3 : x ≠ y) : 
  x^2 + y^2 = 49 := by
sorry

-- Additional helper lemmas if needed
lemma helper_lemma (x y : ℝ) 
  (h1 : y + 9 = (x - 3)^2) 
  (h2 : x + 9 = (y - 3)^2) 
  (h3 : x ≠ y) : 
  x + y = 7 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equality_l1474_147412


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1474_147485

theorem complex_fraction_simplification (x y : ℚ) 
  (hx : x = 3) 
  (hy : y = 4) : 
  (1 / (y + 1)) / (1 / (x - 1)) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1474_147485


namespace NUMINAMATH_CALUDE_peters_leaf_raking_l1474_147497

/-- Given that Peter rakes 3 bags of leaves in 15 minutes at a constant rate,
    prove that it will take him 40 minutes to rake 8 bags of leaves. -/
theorem peters_leaf_raking (rate : ℚ) : 
  (rate * 15 = 3) → (rate * 40 = 8) :=
by sorry

end NUMINAMATH_CALUDE_peters_leaf_raking_l1474_147497


namespace NUMINAMATH_CALUDE_field_trip_buses_l1474_147478

theorem field_trip_buses (total_people : ℕ) (num_vans : ℕ) (people_per_van : ℕ) (people_per_bus : ℕ) 
  (h1 : total_people = 76)
  (h2 : num_vans = 2)
  (h3 : people_per_van = 8)
  (h4 : people_per_bus = 20) :
  (total_people - num_vans * people_per_van) / people_per_bus = 3 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_buses_l1474_147478


namespace NUMINAMATH_CALUDE_nancy_bottle_caps_l1474_147472

theorem nancy_bottle_caps (initial final found : ℕ) : 
  initial = 91 → final = 179 → found = final - initial :=
by sorry

end NUMINAMATH_CALUDE_nancy_bottle_caps_l1474_147472


namespace NUMINAMATH_CALUDE_power_three_equality_l1474_147452

theorem power_three_equality : 3^2012 - 6 * 3^2013 + 2 * 3^2014 = 3^2012 := by
  sorry

end NUMINAMATH_CALUDE_power_three_equality_l1474_147452


namespace NUMINAMATH_CALUDE_bobby_deadlift_increase_l1474_147473

/-- Represents Bobby's deadlift progression --/
structure DeadliftProgress where
  initial_weight : ℕ
  initial_age : ℕ
  final_age : ℕ
  percentage_increase : ℕ
  additional_weight : ℕ

/-- Calculates the average yearly increase in Bobby's deadlift --/
def average_yearly_increase (d : DeadliftProgress) : ℚ :=
  let final_weight := d.initial_weight * (d.percentage_increase : ℚ) / 100 + d.additional_weight
  let total_increase := final_weight - d.initial_weight
  let years := d.final_age - d.initial_age
  total_increase / years

/-- Theorem stating that Bobby's average yearly increase in deadlift is 110 pounds --/
theorem bobby_deadlift_increase :
  let bobby := DeadliftProgress.mk 300 13 18 250 100
  average_yearly_increase bobby = 110 := by
  sorry

end NUMINAMATH_CALUDE_bobby_deadlift_increase_l1474_147473


namespace NUMINAMATH_CALUDE_earbuds_tickets_proof_l1474_147460

/-- The number of tickets Connie spent on earbuds -/
def tickets_on_earbuds (total_tickets : ℕ) (tickets_on_koala : ℕ) (tickets_on_bracelets : ℕ) : ℕ :=
  total_tickets - tickets_on_koala - tickets_on_bracelets

theorem earbuds_tickets_proof :
  let total_tickets : ℕ := 50
  let tickets_on_koala : ℕ := total_tickets / 2
  let tickets_on_bracelets : ℕ := 15
  tickets_on_earbuds total_tickets tickets_on_koala tickets_on_bracelets = 10 := by
  sorry

#eval tickets_on_earbuds 50 25 15

end NUMINAMATH_CALUDE_earbuds_tickets_proof_l1474_147460


namespace NUMINAMATH_CALUDE_car_speed_problem_l1474_147492

/-- Given a car traveling for two hours, prove that if its speed in the second hour
    is 45 km/h and its average speed over the two hours is 55 km/h, then its speed
    in the first hour must be 65 km/h. -/
theorem car_speed_problem (speed_second_hour : ℝ) (average_speed : ℝ)
    (h1 : speed_second_hour = 45)
    (h2 : average_speed = 55) :
    let speed_first_hour := 2 * average_speed - speed_second_hour
    speed_first_hour = 65 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1474_147492


namespace NUMINAMATH_CALUDE_equation_solutions_l1474_147465

theorem equation_solutions : 
  {x : ℝ | (1 / (x^2 + 11*x + 12) + 1 / (x^2 + 2*x + 3) + 1 / (x^2 - 13*x + 14) = 0)} = 
  {-4, -3, 3, 4} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1474_147465


namespace NUMINAMATH_CALUDE_min_like_both_l1474_147423

theorem min_like_both (total : ℕ) (like_mozart : ℕ) (like_beethoven : ℕ)
  (h_total : total = 120)
  (h_mozart : like_mozart = 102)
  (h_beethoven : like_beethoven = 85)
  : ∃ (like_both : ℕ), like_both ≥ 67 ∧ 
    (∀ (x : ℕ), x < like_both → 
      ∃ (only_mozart only_beethoven : ℕ),
        x + only_mozart + only_beethoven ≤ total ∧
        x + only_mozart ≤ like_mozart ∧
        x + only_beethoven ≤ like_beethoven) :=
by sorry

end NUMINAMATH_CALUDE_min_like_both_l1474_147423


namespace NUMINAMATH_CALUDE_triangle_side_validity_l1474_147404

/-- Checks if three lengths can form a valid triangle -/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_side_validity :
  let side1 := 5
  let side2 := 7
  (is_valid_triangle side1 side2 6) ∧
  ¬(is_valid_triangle side1 side2 2) ∧
  ¬(is_valid_triangle side1 side2 17) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_validity_l1474_147404


namespace NUMINAMATH_CALUDE_root_shift_polynomial_l1474_147420

theorem root_shift_polynomial (a b c : ℂ) : 
  (∀ x : ℂ, x^3 - 5*x + 7 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x : ℂ, x^3 - 9*x^2 + 22*x - 5 = 0 ↔ x = a + 3 ∨ x = b + 3 ∨ x = c + 3) :=
by sorry

end NUMINAMATH_CALUDE_root_shift_polynomial_l1474_147420


namespace NUMINAMATH_CALUDE_esperanza_salary_l1474_147487

/-- Calculates the gross monthly salary given the specified expenses and savings. -/
def gross_monthly_salary (rent food_ratio mortgage_ratio savings tax_ratio : ℝ) : ℝ :=
  let food := food_ratio * rent
  let mortgage := mortgage_ratio * food
  let taxes := tax_ratio * savings
  rent + food + mortgage + savings + taxes

/-- Theorem stating the gross monthly salary under given conditions. -/
theorem esperanza_salary : 
  gross_monthly_salary 600 (3/5) 3 2000 (2/5) = 4840 := by
  sorry

end NUMINAMATH_CALUDE_esperanza_salary_l1474_147487


namespace NUMINAMATH_CALUDE_only_C_is_certain_l1474_147400

-- Define the event type
inductive Event
  | A  -- The temperature in Aojiang on June 1st this year is 30 degrees
  | B  -- There are 10 red balls in a box, and any ball taken out must be a white ball
  | C  -- Throwing a stone, the stone will eventually fall
  | D  -- In this math competition, every participating student will score full marks

-- Define what it means for an event to be certain
def is_certain (e : Event) : Prop :=
  match e with
  | Event.C => True
  | _ => False

-- Theorem statement
theorem only_C_is_certain :
  ∀ e : Event, is_certain e ↔ e = Event.C :=
by sorry

end NUMINAMATH_CALUDE_only_C_is_certain_l1474_147400


namespace NUMINAMATH_CALUDE_number_equation_solution_l1474_147483

theorem number_equation_solution : 
  ∀ x : ℝ, (2/5 : ℝ) * x - 3 * ((1/4 : ℝ) * x) + 7 = 14 → x = -20 := by
sorry

end NUMINAMATH_CALUDE_number_equation_solution_l1474_147483


namespace NUMINAMATH_CALUDE_monotonicity_condition_l1474_147482

theorem monotonicity_condition (ω : ℝ) (h_ω_pos : ω > 0) :
  (∀ x ∈ Set.Ioo 0 (π / 3), Monotone (fun x => Real.sin (ω * x + π / 6))) ↔ ω ∈ Set.Ioc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_monotonicity_condition_l1474_147482


namespace NUMINAMATH_CALUDE_common_root_condition_l1474_147408

theorem common_root_condition (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 = 0 ∧ x^2 + x + a = 0) ↔ (a = 1 ∨ a = -2) :=
by sorry

end NUMINAMATH_CALUDE_common_root_condition_l1474_147408


namespace NUMINAMATH_CALUDE_cosine_ratio_equals_one_l1474_147447

theorem cosine_ratio_equals_one :
  (Real.cos (66 * π / 180) * Real.cos (6 * π / 180) + Real.cos (84 * π / 180) * Real.cos (24 * π / 180)) /
  (Real.cos (65 * π / 180) * Real.cos (5 * π / 180) + Real.cos (85 * π / 180) * Real.cos (25 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_ratio_equals_one_l1474_147447


namespace NUMINAMATH_CALUDE_coefficient_sum_l1474_147477

theorem coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x - 2)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = -1 := by
sorry

end NUMINAMATH_CALUDE_coefficient_sum_l1474_147477


namespace NUMINAMATH_CALUDE_night_day_worker_loading_ratio_l1474_147448

theorem night_day_worker_loading_ratio
  (day_workers : ℚ)
  (night_workers : ℚ)
  (total_boxes : ℚ)
  (h1 : night_workers = (4/5) * day_workers)
  (h2 : (5/6) * total_boxes = day_workers * (boxes_per_day_worker : ℚ))
  (h3 : (1/6) * total_boxes = night_workers * (boxes_per_night_worker : ℚ)) :
  boxes_per_night_worker / boxes_per_day_worker = 25/4 := by
  sorry

end NUMINAMATH_CALUDE_night_day_worker_loading_ratio_l1474_147448


namespace NUMINAMATH_CALUDE_correct_seating_count_l1474_147495

/-- Number of Democrats in the Senate committee -/
def num_democrats : ℕ := 6

/-- Number of Republicans in the Senate committee -/
def num_republicans : ℕ := 6

/-- Number of Independents in the Senate committee -/
def num_independents : ℕ := 2

/-- Total number of committee members -/
def total_members : ℕ := num_democrats + num_republicans + num_independents

/-- Function to calculate the number of valid seating arrangements -/
def seating_arrangements : ℕ :=
  12 * (Nat.factorial 10) / 2

/-- Theorem stating the number of valid seating arrangements -/
theorem correct_seating_count :
  seating_arrangements = 21772800 := by sorry

end NUMINAMATH_CALUDE_correct_seating_count_l1474_147495


namespace NUMINAMATH_CALUDE_apple_weight_l1474_147479

/-- Given a bag containing apples, prove the weight of one apple. -/
theorem apple_weight (total_weight : ℝ) (empty_bag_weight : ℝ) (apple_count : ℕ) :
  total_weight = 1.82 →
  empty_bag_weight = 0.5 →
  apple_count = 6 →
  (total_weight - empty_bag_weight) / apple_count = 0.22 := by
  sorry

end NUMINAMATH_CALUDE_apple_weight_l1474_147479


namespace NUMINAMATH_CALUDE_equation_solution_l1474_147451

theorem equation_solution :
  ∀ x : ℝ, Real.sqrt (x + 9) - Real.sqrt (x - 5) - 2 = 0 → x = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1474_147451


namespace NUMINAMATH_CALUDE_fruit_sales_problem_l1474_147426

/-- Fruit sales problem -/
theorem fruit_sales_problem 
  (ponkan_cost fuji_cost : ℚ)
  (h1 : 30 * ponkan_cost + 20 * fuji_cost = 2700)
  (h2 : 50 * ponkan_cost + 40 * fuji_cost = 4800)
  (ponkan_price fuji_price : ℚ)
  (h3 : ponkan_price = 80)
  (h4 : fuji_price = 60)
  (fuji_price_red1 fuji_price_red2 : ℚ)
  (h5 : fuji_price_red1 = fuji_price * (1 - 1/10))
  (h6 : fuji_price_red2 = fuji_price_red1 * (1 - 1/10))
  (profit : ℚ)
  (h7 : profit = 50 * (ponkan_price - ponkan_cost) + 
                 20 * (fuji_price - fuji_cost) +
                 10 * (fuji_price_red1 - fuji_cost) +
                 10 * (fuji_price_red2 - fuji_cost)) :
  ponkan_cost = 60 ∧ fuji_cost = 45 ∧ profit = 1426 := by
sorry

end NUMINAMATH_CALUDE_fruit_sales_problem_l1474_147426


namespace NUMINAMATH_CALUDE_parabola_linear_function_relationship_l1474_147455

-- Define the parabola
def parabola (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x

-- Define the linear function
def linear_function (a b : ℝ) (x : ℝ) : ℝ := (a - b) * x + b

theorem parabola_linear_function_relationship 
  (a b m : ℝ) 
  (h1 : a < 0)  -- parabola opens downwards
  (h2 : m < 0)  -- P(-1, m) is in the third quadrant
  (h3 : parabola a b (-1) = m)  -- parabola passes through P(-1, m)
  (h4 : -b / (2*a) < 0)  -- axis of symmetry is negative (P and origin on opposite sides)
  : ∀ x y : ℝ, x > 0 ∧ y > 0 → linear_function a b x ≠ y :=
by sorry

end NUMINAMATH_CALUDE_parabola_linear_function_relationship_l1474_147455


namespace NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l1474_147438

theorem smallest_multiple_of_6_and_15 : 
  ∃ b : ℕ+, (∀ n : ℕ+, 6 ∣ n ∧ 15 ∣ n → b ≤ n) ∧ 6 ∣ b ∧ 15 ∣ b ∧ b = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l1474_147438


namespace NUMINAMATH_CALUDE_inequality_proof_l1474_147433

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / b) + (b^3 / c^2) + (c^4 / a^3) ≥ -a + 2*b + 2*c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1474_147433


namespace NUMINAMATH_CALUDE_smallest_square_area_l1474_147407

/-- A square in the plane --/
structure RotatedSquare where
  center : ℤ × ℤ
  sideLength : ℝ
  rotation : ℝ

/-- Count the number of lattice points on the boundary of a rotated square --/
def countBoundaryLatticePoints (s : RotatedSquare) : ℕ :=
  sorry

/-- The area of a square --/
def squareArea (s : RotatedSquare) : ℝ :=
  s.sideLength ^ 2

/-- The theorem stating the area of the smallest square meeting the conditions --/
theorem smallest_square_area : 
  ∃ (s : RotatedSquare), 
    (∀ (s' : RotatedSquare), 
      countBoundaryLatticePoints s' = 5 → squareArea s ≤ squareArea s') ∧ 
    countBoundaryLatticePoints s = 5 ∧ 
    squareArea s = 32 :=
  sorry

end NUMINAMATH_CALUDE_smallest_square_area_l1474_147407


namespace NUMINAMATH_CALUDE_engagement_ring_saving_time_l1474_147418

/-- Proves the time required to save for an engagement ring based on annual salary and monthly savings -/
theorem engagement_ring_saving_time 
  (annual_salary : ℕ) 
  (monthly_savings : ℕ) 
  (h1 : annual_salary = 60000)
  (h2 : monthly_savings = 1000) : 
  (2 * (annual_salary / 12)) / monthly_savings = 10 := by
  sorry

end NUMINAMATH_CALUDE_engagement_ring_saving_time_l1474_147418


namespace NUMINAMATH_CALUDE_sample_size_calculation_l1474_147459

/-- Represents the sample size for each school level -/
structure SampleSize where
  elementary : ℕ
  middle : ℕ
  high : ℕ

/-- Calculates the total sample size -/
def totalSampleSize (s : SampleSize) : ℕ :=
  s.elementary + s.middle + s.high

/-- The ratio of elementary:middle:high school students -/
def schoolRatio : Fin 3 → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 5

theorem sample_size_calculation (s : SampleSize) 
  (h_ratio : s.elementary * schoolRatio 1 = s.middle * schoolRatio 0 ∧ 
             s.middle * schoolRatio 2 = s.high * schoolRatio 1)
  (h_middle : s.middle = 150) : 
  totalSampleSize s = 500 := by
sorry

end NUMINAMATH_CALUDE_sample_size_calculation_l1474_147459


namespace NUMINAMATH_CALUDE_bacteria_states_l1474_147463

-- Define the bacteria types
inductive BacteriaType
| Red
| Blue

-- Define the state of the bacteria population
structure BacteriaState where
  red : ℕ
  blue : ℕ

-- Define the transformation rules
def transform (state : BacteriaState) : Set BacteriaState :=
  { BacteriaState.mk (state.red - 2) (state.blue + 1),  -- Two red to one blue
    BacteriaState.mk (state.red + 4) (state.blue - 2),  -- Two blue to four red
    BacteriaState.mk (state.red + 2) (state.blue - 1) } -- One red and one blue to three red

-- Define the initial state
def initial_state (r b : ℕ) : BacteriaState :=
  BacteriaState.mk r b

-- Define the set of possible states
def possible_states (n : ℕ) : Set BacteriaState :=
  {state | ∃ m : ℕ, state.red = n - 2 * m ∧ state.blue = m}

-- Theorem statement
theorem bacteria_states (r b : ℕ) :
  let init := initial_state r b
  let n := r + b
  ∀ state, state ∈ possible_states n ↔ 
    ∃ sequence : ℕ → BacteriaState, 
      sequence 0 = init ∧
      (∀ i, sequence (i + 1) ∈ transform (sequence i)) ∧
      (∃ j, sequence j = state) :=
by sorry

end NUMINAMATH_CALUDE_bacteria_states_l1474_147463


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1474_147491

theorem polynomial_divisibility (a b c d m : ℤ) 
  (h1 : (5 : ℤ) ∣ (a * m^3 + b * m^2 + c * m + d))
  (h2 : ¬((5 : ℤ) ∣ d)) :
  ∃ n : ℤ, (5 : ℤ) ∣ (d * n^3 + c * n^2 + b * n + a) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1474_147491


namespace NUMINAMATH_CALUDE_divisors_of_power_minus_one_l1474_147443

theorem divisors_of_power_minus_one (a b r : ℕ) (ha : a ≥ 2) (hb : b > 0) (hb_composite : ∃ x y, 1 < x ∧ 1 < y ∧ b = x * y) (hr : ∃ (S : Finset ℕ), S.card = r ∧ ∀ x ∈ S, x > 0 ∧ x ∣ b) :
  ∃ (T : Finset ℕ), T.card ≥ r ∧ ∀ x ∈ T, x > 0 ∧ x ∣ (a^b - 1) :=
sorry

end NUMINAMATH_CALUDE_divisors_of_power_minus_one_l1474_147443


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_l1474_147444

theorem chocolate_bar_cost (total_cost : ℚ) (num_chocolate_bars : ℕ) (num_gummy_packs : ℕ) (num_chip_bags : ℕ) 
  (gummy_pack_cost : ℚ) (chip_bag_cost : ℚ) :
  total_cost = 150 →
  num_chocolate_bars = 10 →
  num_gummy_packs = 10 →
  num_chip_bags = 20 →
  gummy_pack_cost = 2 →
  chip_bag_cost = 5 →
  (total_cost - (num_gummy_packs * gummy_pack_cost + num_chip_bags * chip_bag_cost)) / num_chocolate_bars = 3 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_cost_l1474_147444


namespace NUMINAMATH_CALUDE_solution_values_l1474_147476

-- Define the solution sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x : ℝ | x^2 + x - 6 < 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Define the solution set of x^2 + ax + b < 0
def solution_set (a b : ℝ) : Set ℝ := {x : ℝ | x^2 + a*x + b < 0}

-- Theorem statement
theorem solution_values :
  ∃ (a b : ℝ), solution_set a b = A_intersect_B ∧ a = -1 ∧ b = -2 :=
sorry

end NUMINAMATH_CALUDE_solution_values_l1474_147476


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1474_147406

theorem right_triangle_hypotenuse (a b c : ℝ) :
  a = 15 ∧ b = 36 ∧ c^2 = a^2 + b^2 → c = 39 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1474_147406


namespace NUMINAMATH_CALUDE_john_bought_three_reels_l1474_147442

/-- The number of reels John bought -/
def num_reels : ℕ := sorry

/-- The length of fishing line in each reel (in meters) -/
def reel_length : ℕ := 100

/-- The length of each section after cutting (in meters) -/
def section_length : ℕ := 10

/-- The number of sections John got after cutting -/
def num_sections : ℕ := 30

/-- Theorem: John bought 3 reels of fishing line -/
theorem john_bought_three_reels :
  num_reels = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_john_bought_three_reels_l1474_147442


namespace NUMINAMATH_CALUDE_proportional_function_decreasing_l1474_147484

theorem proportional_function_decreasing (k : ℝ) (h1 : k ≠ 0) :
  (∀ x₁ x₂ y₁ y₂ : ℝ, x₁ < x₂ → k * x₁ = y₁ → k * x₂ = y₂ → y₁ > y₂) → k < 0 :=
by sorry

end NUMINAMATH_CALUDE_proportional_function_decreasing_l1474_147484


namespace NUMINAMATH_CALUDE_gcd_n4_plus_125_and_n_plus_5_l1474_147431

theorem gcd_n4_plus_125_and_n_plus_5 (n : ℕ) (h1 : n > 0) (h2 : ¬ 7 ∣ n) :
  (Nat.gcd (n^4 + 5^3) (n + 5) = 1) ∨ (Nat.gcd (n^4 + 5^3) (n + 5) = 3) := by
sorry

end NUMINAMATH_CALUDE_gcd_n4_plus_125_and_n_plus_5_l1474_147431


namespace NUMINAMATH_CALUDE_max_sum_on_circle_max_sum_achievable_l1474_147488

theorem max_sum_on_circle (x y : ℤ) : x^2 + y^2 = 100 → x + y ≤ 14 := by
  sorry

theorem max_sum_achievable : ∃ x y : ℤ, x^2 + y^2 = 100 ∧ x + y = 14 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_max_sum_achievable_l1474_147488


namespace NUMINAMATH_CALUDE_quadratic_root_m_value_l1474_147437

theorem quadratic_root_m_value (m : ℝ) : 
  (∃ x : ℝ, x^2 + 3*x - m = 0 ∧ x = 1) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_m_value_l1474_147437


namespace NUMINAMATH_CALUDE_min_value_product_l1474_147414

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (2 * x + 3 * y) * (2 * y + 3 * z) * (2 * x * z + 1) ≥ 24 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    (2 * x₀ + 3 * y₀) * (2 * y₀ + 3 * z₀) * (2 * x₀ * z₀ + 1) = 24 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_l1474_147414


namespace NUMINAMATH_CALUDE_max_distinct_pairs_l1474_147405

theorem max_distinct_pairs (n : ℕ) (h : n = 2023) :
  let S := Finset.range n
  ∃ (k : ℕ) (pairs : Finset (ℕ × ℕ)),
    k = 809 ∧
    pairs.card = k ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ n) ∧
    (∀ (m : ℕ) (larger_pairs : Finset (ℕ × ℕ)),
      m > k →
      (larger_pairs.card = m →
        ¬((∀ (p : ℕ × ℕ), p ∈ larger_pairs → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) ∧
          (∀ (p q : ℕ × ℕ), p ∈ larger_pairs → q ∈ larger_pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
          (∀ (p q : ℕ × ℕ), p ∈ larger_pairs → q ∈ larger_pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
          (∀ (p : ℕ × ℕ), p ∈ larger_pairs → p.1 + p.2 ≤ n)))) :=
by
  sorry

end NUMINAMATH_CALUDE_max_distinct_pairs_l1474_147405


namespace NUMINAMATH_CALUDE_other_number_is_twenty_l1474_147417

theorem other_number_is_twenty (a b : ℕ) (h1 : a + b = 30) (h2 : a = 10 ∨ b = 10) : 
  (a = 20 ∨ b = 20) :=
by sorry

end NUMINAMATH_CALUDE_other_number_is_twenty_l1474_147417


namespace NUMINAMATH_CALUDE_opposite_def_opposite_of_two_l1474_147441

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The property that defines the opposite of a number -/
theorem opposite_def (x : ℝ) : x + opposite x = 0 := by sorry

/-- Proof that the opposite of 2 is -2 -/
theorem opposite_of_two : opposite 2 = -2 := by sorry

end NUMINAMATH_CALUDE_opposite_def_opposite_of_two_l1474_147441


namespace NUMINAMATH_CALUDE_perfect_squares_ending_in_444_and_4444_l1474_147453

def ends_in_444 (n : ℕ) : Prop := n % 1000 = 444

def ends_in_4444 (n : ℕ) : Prop := n % 10000 = 4444

theorem perfect_squares_ending_in_444_and_4444 :
  (∀ a : ℕ, (∃ k : ℕ, a * a = k) ∧ ends_in_444 (a * a) ↔ ∃ n : ℕ, a = 500 * n + 38 ∨ a = 500 * n - 38) ∧
  (¬ ∃ a : ℕ, (∃ k : ℕ, a * a = k) ∧ ends_in_4444 (a * a)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_ending_in_444_and_4444_l1474_147453


namespace NUMINAMATH_CALUDE_cubic_roots_existence_l1474_147489

theorem cubic_roots_existence (a b c : ℝ) : 
  (a + b + c = 6 ∧ a * b + b * c + c * a = 9) →
  (¬ (a^4 + b^4 + c^4 = 260) ∧ ∃ (x y z : ℝ), x + y + z = 6 ∧ x * y + y * z + z * x = 9 ∧ x^4 + y^4 + z^4 = 210) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_existence_l1474_147489


namespace NUMINAMATH_CALUDE_equation_solution_l1474_147425

theorem equation_solution : 
  ∃ x : ℝ, (1 / 7 + 7 / x = 15 / x + 1 / 15) ∧ (x = 105) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1474_147425


namespace NUMINAMATH_CALUDE_line_equation_proof_l1474_147422

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point lies on a line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- The given line 2x - 3y + 4 = 0 -/
def given_line : Line :=
  { a := 2, b := -3, c := 4 }

/-- The point (-1, 2) -/
def point : (ℝ × ℝ) :=
  (-1, 2)

/-- The equation of the line we want to prove -/
def target_line : Line :=
  { a := 2, b := -3, c := 8 }

theorem line_equation_proof :
  parallel target_line given_line ∧
  point_on_line point.1 point.2 target_line :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1474_147422


namespace NUMINAMATH_CALUDE_percentage_excess_l1474_147464

theorem percentage_excess (x y : ℝ) (h : x = 0.38 * y) :
  (y - x) / x = 0.62 := by
  sorry

end NUMINAMATH_CALUDE_percentage_excess_l1474_147464


namespace NUMINAMATH_CALUDE_function_domain_range_l1474_147439

theorem function_domain_range (a : ℝ) (h1 : a > 1) : 
  (∀ x ∈ Set.Icc 1 a, x^2 - 2*a*x + 5 ∈ Set.Icc 1 a) ∧
  (∀ y ∈ Set.Icc 1 a, ∃ x ∈ Set.Icc 1 a, y = x^2 - 2*a*x + 5) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_function_domain_range_l1474_147439


namespace NUMINAMATH_CALUDE_premier_pups_count_l1474_147474

theorem premier_pups_count :
  let fetch : ℕ := 70
  let jump : ℕ := 40
  let bark : ℕ := 45
  let fetch_and_jump : ℕ := 25
  let jump_and_bark : ℕ := 15
  let fetch_and_bark : ℕ := 20
  let all_three : ℕ := 12
  let none : ℕ := 15
  
  let fetch_only : ℕ := fetch - (fetch_and_jump + fetch_and_bark - all_three)
  let jump_only : ℕ := jump - (fetch_and_jump + jump_and_bark - all_three)
  let bark_only : ℕ := bark - (fetch_and_bark + jump_and_bark - all_three)
  let fetch_jump_only : ℕ := fetch_and_jump - all_three
  let jump_bark_only : ℕ := jump_and_bark - all_three
  let fetch_bark_only : ℕ := fetch_and_bark - all_three

  fetch_only + jump_only + bark_only + fetch_jump_only + jump_bark_only + fetch_bark_only + all_three + none = 122 := by
  sorry

end NUMINAMATH_CALUDE_premier_pups_count_l1474_147474


namespace NUMINAMATH_CALUDE_lemon_pie_degrees_l1474_147454

/-- The number of degrees in a circle --/
def circle_degrees : ℕ := 360

/-- The total number of students in the class --/
def total_students : ℕ := 45

/-- The number of students preferring chocolate pie --/
def chocolate_pref : ℕ := 15

/-- The number of students preferring apple pie --/
def apple_pref : ℕ := 10

/-- The number of students preferring blueberry pie --/
def blueberry_pref : ℕ := 9

/-- Calculate the number of students preferring lemon pie --/
def lemon_pref : ℚ :=
  (total_students - (chocolate_pref + apple_pref + blueberry_pref)) / 2

/-- Theorem: The number of degrees for lemon pie on a pie chart is 44° --/
theorem lemon_pie_degrees : 
  (lemon_pref / total_students) * circle_degrees = 44 := by
  sorry

end NUMINAMATH_CALUDE_lemon_pie_degrees_l1474_147454


namespace NUMINAMATH_CALUDE_grandmothers_age_is_77_l1474_147461

/-- The grandmother's age is obtained by writing the Latin grade twice in a row -/
def grandmothers_age (latin_grade : ℕ) : ℕ := 11 * latin_grade

/-- The morning grade is obtained by dividing the grandmother's age by the number of kittens and subtracting fourteen-thirds -/
def morning_grade (age : ℕ) (kittens : ℕ) : ℚ := age / kittens - 14 / 3

theorem grandmothers_age_is_77 :
  ∃ (latin_grade : ℕ) (kittens : ℕ),
    latin_grade < 10 ∧
    kittens % 3 = 0 ∧
    grandmothers_age latin_grade = 77 ∧
    morning_grade (grandmothers_age latin_grade) kittens = latin_grade :=
by sorry

end NUMINAMATH_CALUDE_grandmothers_age_is_77_l1474_147461


namespace NUMINAMATH_CALUDE_tangent_intersection_y_coordinate_l1474_147409

/-- 
Given a parabola y = x^2 + 1 and two points A and B on it with perpendicular tangents,
this theorem states that the y-coordinate of the intersection point P of these tangents is 3/4.
-/
theorem tangent_intersection_y_coordinate (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 + 1
  let A : ℝ × ℝ := (a, f a)
  let B : ℝ × ℝ := (b, f b)
  let tangent_A : ℝ → ℝ := λ x => 2*a*x - a^2 + 1
  let tangent_B : ℝ → ℝ := λ x => 2*b*x - b^2 + 1
  -- Perpendicularity condition: product of slopes is -1
  2*a * 2*b = -1 →
  -- P is the intersection point of tangents
  let P : ℝ × ℝ := ((a + b) / 2, tangent_A ((a + b) / 2))
  -- The y-coordinate of P is 3/4
  P.2 = 3/4 := by sorry


end NUMINAMATH_CALUDE_tangent_intersection_y_coordinate_l1474_147409


namespace NUMINAMATH_CALUDE_special_numbers_count_l1474_147457

def count_special_numbers (n : ℕ) : ℕ :=
  (n / 12) - (n / 60)

theorem special_numbers_count :
  count_special_numbers 2017 = 135 := by
  sorry

end NUMINAMATH_CALUDE_special_numbers_count_l1474_147457


namespace NUMINAMATH_CALUDE_max_value_cube_root_sum_and_sum_l1474_147468

theorem max_value_cube_root_sum_and_sum (x y : ℝ) :
  (x^(1/3) + y^(1/3) = 2) →
  (x + y = 20) →
  max x y = 10 + 6 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_cube_root_sum_and_sum_l1474_147468


namespace NUMINAMATH_CALUDE_month_mean_profit_l1474_147427

/-- Calculates the mean daily profit for a month given the mean profits of two equal periods -/
def mean_daily_profit (days : ℕ) (mean_profit1 : ℚ) (mean_profit2 : ℚ) : ℚ :=
  (mean_profit1 + mean_profit2) / 2

theorem month_mean_profit : 
  let days : ℕ := 30
  let first_half_mean : ℚ := 275
  let second_half_mean : ℚ := 425
  mean_daily_profit days first_half_mean second_half_mean = 350 := by
sorry

end NUMINAMATH_CALUDE_month_mean_profit_l1474_147427


namespace NUMINAMATH_CALUDE_spring_percentage_is_ten_percent_l1474_147436

/-- The percentage of students who chose Spring -/
def spring_percentage (total : ℕ) (spring : ℕ) : ℚ :=
  (spring : ℚ) / (total : ℚ) * 100

/-- Theorem: The percentage of students who chose Spring is 10% -/
theorem spring_percentage_is_ten_percent :
  spring_percentage 10 1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_spring_percentage_is_ten_percent_l1474_147436
