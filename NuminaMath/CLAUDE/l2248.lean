import Mathlib

namespace zero_properties_l2248_224872

theorem zero_properties : 
  (0 : ℕ) = 0 ∧ (0 : ℤ) = 0 ∧ (0 : ℝ) = 0 ∧ ¬(0 > 0) := by
  sorry

end zero_properties_l2248_224872


namespace ones_divisible_by_power_of_three_l2248_224807

/-- Given a natural number n ≥ 1, the function returns the number formed by 3^n consecutive ones. -/
def number_of_ones (n : ℕ) : ℕ :=
  (10^(3^n) - 1) / 9

/-- Theorem stating that for any natural number n ≥ 1, the number formed by 3^n consecutive ones
    is divisible by 3^n. -/
theorem ones_divisible_by_power_of_three (n : ℕ) (h : n ≥ 1) :
  ∃ k : ℕ, number_of_ones n = 3^n * k :=
sorry

end ones_divisible_by_power_of_three_l2248_224807


namespace last_year_honey_harvest_l2248_224853

/-- 
Given Diane's honey harvest information:
- This year's harvest: 8564 pounds
- Increase from last year: 6085 pounds

Prove that last year's harvest was 2479 pounds.
-/
theorem last_year_honey_harvest 
  (this_year : ℕ) 
  (increase : ℕ) 
  (h1 : this_year = 8564)
  (h2 : increase = 6085) :
  this_year - increase = 2479 := by
sorry

end last_year_honey_harvest_l2248_224853


namespace five_balls_three_boxes_l2248_224841

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 5 ways to distribute 5 indistinguishable balls into 3 indistinguishable boxes -/
theorem five_balls_three_boxes : distribute_balls 5 3 = 5 := by sorry

end five_balls_three_boxes_l2248_224841


namespace complex_midpoint_and_distance_l2248_224843

theorem complex_midpoint_and_distance (z₁ z₂ m : ℂ) (h₁ : z₁ = -7 + 5*I) (h₂ : z₂ = 9 - 11*I) 
  (h_m : m = (z₁ + z₂) / 2) : 
  m = 1 - 3*I ∧ Complex.abs (z₁ - m) = 8*Real.sqrt 2 := by
  sorry

end complex_midpoint_and_distance_l2248_224843


namespace distance_to_left_focus_l2248_224898

-- Define the ellipse
def is_on_ellipse (x y b : ℝ) : Prop :=
  x^2 / 25 + y^2 / b^2 = 1

-- Define the condition for b
def valid_b (b : ℝ) : Prop :=
  0 < b ∧ b < 5

-- Define a point P on the ellipse
def P_on_ellipse (P : ℝ × ℝ) (b : ℝ) : Prop :=
  is_on_ellipse P.1 P.2 b

-- Define the left focus F₁
def F₁ : ℝ × ℝ := sorry

-- Define the condition |OP⃗ + OF₁⃗| = 8
def vector_sum_condition (P : ℝ × ℝ) : Prop :=
  ‖P + F₁‖ = 8

-- Theorem statement
theorem distance_to_left_focus
  (b : ℝ)
  (P : ℝ × ℝ)
  (h_b : valid_b b)
  (h_P : P_on_ellipse P b)
  (h_sum : vector_sum_condition P) :
  ‖P - F₁‖ = 2 :=
sorry

end distance_to_left_focus_l2248_224898


namespace arithmetic_calculation_l2248_224891

theorem arithmetic_calculation : 1323 + 150 / 50 * 3 - 223 = 1109 := by
  sorry

end arithmetic_calculation_l2248_224891


namespace power_of_125_two_thirds_l2248_224875

theorem power_of_125_two_thirds : (125 : ℝ) ^ (2/3) = 25 := by sorry

end power_of_125_two_thirds_l2248_224875


namespace inequality_condition_l2248_224882

theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, 0 < x → x < 2 → x^2 - 2*x + a < 0) → a ≤ 0 := by
  sorry

end inequality_condition_l2248_224882


namespace min_diagonal_rectangle_l2248_224871

/-- Given a rectangle ABCD with perimeter 30 inches and width w ≥ 6 inches,
    the minimum length of diagonal AC is 7.5√2 inches. -/
theorem min_diagonal_rectangle (l w : ℝ) (h1 : l + w = 15) (h2 : w ≥ 6) :
  ∃ (AC : ℝ), AC = 7.5 * Real.sqrt 2 ∧ ∀ (AC' : ℝ), AC' ≥ AC := by
  sorry

end min_diagonal_rectangle_l2248_224871


namespace equation_positive_root_implies_m_equals_3_l2248_224880

-- Define the equation
def equation (m x : ℝ) : Prop :=
  m / (x - 4) - (1 - x) / (4 - x) = 0

-- Define what it means for x to be a positive root
def is_positive_root (m x : ℝ) : Prop :=
  equation m x ∧ x > 0

-- Theorem statement
theorem equation_positive_root_implies_m_equals_3 :
  ∀ m : ℝ, (∃ x : ℝ, is_positive_root m x) → m = 3 :=
by sorry

end equation_positive_root_implies_m_equals_3_l2248_224880


namespace floor_sum_2017_l2248_224840

theorem floor_sum_2017 : 
  let floor (x : ℚ) := ⌊x⌋
  ∀ (isPrime2017 : Nat.Prime 2017),
    (floor (2017 * 3 / 11) : ℤ) + 
    (floor (2017 * 4 / 11) : ℤ) + 
    (floor (2017 * 5 / 11) : ℤ) + 
    (floor (2017 * 6 / 11) : ℤ) + 
    (floor (2017 * 7 / 11) : ℤ) + 
    (floor (2017 * 8 / 11) : ℤ) = 6048 := by
  sorry

end floor_sum_2017_l2248_224840


namespace simplify_sqrt_fraction_l2248_224803

theorem simplify_sqrt_fraction : 
  (Real.sqrt 462 / Real.sqrt 330) + (Real.sqrt 245 / Real.sqrt 175) = 12 * Real.sqrt 35 / 25 := by
  sorry

end simplify_sqrt_fraction_l2248_224803


namespace inequality_proof_l2248_224866

theorem inequality_proof (a b c d : ℝ) 
  (non_neg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) 
  (sum_condition : a*b + a*c + a*d + b*c + b*d + c*d = 6) : 
  1 / (a^2 + 1) + 1 / (b^2 + 1) + 1 / (c^2 + 1) + 1 / (d^2 + 1) ≥ 2 := by
  sorry

end inequality_proof_l2248_224866


namespace arithmetic_geometric_sequence_product_l2248_224847

/-- An arithmetic sequence where each term is not 0 and satisfies a specific condition -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n ≠ 0) ∧
  (∃ d, ∀ n, a (n + 1) = a n + d) ∧
  a 6 - (a 7)^2 + a 8 = 0

/-- A geometric sequence -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, b (n + 1) = r * b n

/-- The main theorem -/
theorem arithmetic_geometric_sequence_product
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (ha : ArithmeticSequence a)
  (hb : GeometricSequence b)
  (h_equal : b 7 = a 7) :
  b 4 * b 7 * b 10 = 8 := by
sorry

end arithmetic_geometric_sequence_product_l2248_224847


namespace irrational_among_given_numbers_l2248_224881

theorem irrational_among_given_numbers :
  let a : ℝ := -1/7
  let b : ℝ := Real.sqrt 11
  let c : ℝ := 0.3
  let d : ℝ := Real.sqrt 25
  Irrational b ∧ ¬(Irrational a ∨ Irrational c ∨ Irrational d) := by
  sorry

end irrational_among_given_numbers_l2248_224881


namespace smallest_b_for_non_range_l2248_224809

theorem smallest_b_for_non_range (b : ℤ) : 
  (∀ x : ℝ, x^2 + b*x + 10 ≠ -6) ↔ b ≤ -7 :=
sorry

end smallest_b_for_non_range_l2248_224809


namespace probability_of_red_bean_l2248_224804

/-- The probability of choosing a red bean from a bag -/
theorem probability_of_red_bean 
  (initial_red : ℕ) 
  (initial_black : ℕ) 
  (added_red : ℕ) 
  (added_black : ℕ) 
  (h1 : initial_red = 5)
  (h2 : initial_black = 9)
  (h3 : added_red = 3)
  (h4 : added_black = 3) : 
  (initial_red + added_red : ℚ) / (initial_red + initial_black + added_red + added_black) = 2 / 5 := by
sorry

end probability_of_red_bean_l2248_224804


namespace eraser_cost_l2248_224846

theorem eraser_cost (total_students : Nat) (buyers : Nat) (erasers_per_student : Nat) (total_cost : Nat) :
  total_students = 36 →
  buyers > total_students / 2 →
  buyers ≤ total_students →
  erasers_per_student > 2 →
  total_cost = 3978 →
  ∃ (cost : Nat), cost > erasers_per_student ∧
                  buyers * erasers_per_student * cost = total_cost ∧
                  cost = 17 :=
by sorry

end eraser_cost_l2248_224846


namespace balloon_difference_l2248_224860

theorem balloon_difference (your_balloons friend_balloons : ℝ) 
  (h1 : your_balloons = -7)
  (h2 : friend_balloons = 4.5) :
  friend_balloons - your_balloons = 11.5 := by
  sorry

end balloon_difference_l2248_224860


namespace circle_area_difference_l2248_224833

theorem circle_area_difference (r₁ r₂ : ℝ) (h₁ : r₁ = 14) (h₂ : r₂ = 10) :
  π * r₁^2 - π * r₂^2 = 96 * π := by
  sorry

end circle_area_difference_l2248_224833


namespace total_profit_calculation_l2248_224879

/-- The total profit of a business partnership given investments and one partner's profit share -/
theorem total_profit_calculation (p_investment q_investment : ℚ) (q_profit_share : ℚ) : 
  p_investment = 54000 →
  q_investment = 36000 →
  q_profit_share = 6001.89 →
  (p_investment + q_investment) / q_investment * q_profit_share = 15004.725 :=
by
  sorry

#eval (54000 + 36000) / 36000 * 6001.89

end total_profit_calculation_l2248_224879


namespace min_value_expression_min_value_achieved_l2248_224800

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 5*x + 1) * (y^2 + 5*y + 1) * (z^2 + 5*z + 1) / (x*y*z) ≥ 343 :=
by sorry

theorem min_value_achieved (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a^2 + 5*a + 1) * (b^2 + 5*b + 1) * (c^2 + 5*c + 1) / (a*b*c) = 343 :=
by sorry

end min_value_expression_min_value_achieved_l2248_224800


namespace longest_segment_in_quarter_circle_l2248_224890

theorem longest_segment_in_quarter_circle (d : ℝ) (h : d = 10) :
  let r := d / 2
  let m := r * Real.sqrt 2
  m ^ 2 = 50 := by
  sorry

end longest_segment_in_quarter_circle_l2248_224890


namespace price_decrease_percentage_l2248_224884

theorem price_decrease_percentage (original_price : ℝ) (h : original_price > 0) :
  let first_sale_price := (4 / 5 : ℝ) * original_price
  let second_sale_price := (1 / 2 : ℝ) * original_price
  let price_difference := first_sale_price - second_sale_price
  let percentage_decrease := (price_difference / first_sale_price) * 100
  percentage_decrease = 37.5 := by
sorry

end price_decrease_percentage_l2248_224884


namespace davids_age_l2248_224829

/-- Given the ages of Uncle Bob, Emily, and David, prove David's age --/
theorem davids_age (uncle_bob_age : ℕ) (emily_age : ℕ) (david_age : ℕ) 
  (h1 : uncle_bob_age = 60)
  (h2 : emily_age = 2 * uncle_bob_age / 3)
  (h3 : david_age = emily_age - 10) : 
  david_age = 30 := by
  sorry

#check davids_age

end davids_age_l2248_224829


namespace least_apples_count_l2248_224844

theorem least_apples_count (b : ℕ) : 
  (b > 0) →
  (b % 3 = 2) → 
  (b % 4 = 3) → 
  (b % 5 = 1) → 
  (∀ n : ℕ, n > 0 ∧ n % 3 = 2 ∧ n % 4 = 3 ∧ n % 5 = 1 → n ≥ b) →
  b = 11 := by
sorry

end least_apples_count_l2248_224844


namespace imaginary_part_of_z_l2248_224813

theorem imaginary_part_of_z (i : ℂ) (z : ℂ) : 
  i ^ 2 = -1 →
  z * (2 - i) = i ^ 3 →
  z.im = -2/5 := by sorry

end imaginary_part_of_z_l2248_224813


namespace arithmetic_sequence_sum_l2248_224877

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a_n where a_2 + a_4 + a_5 + a_6 + a_8 = 25, prove that a_2 + a_8 = 10 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 2 + a 4 + a 5 + a 6 + a 8 = 25) : 
  a 2 + a 8 = 10 := by
  sorry

end arithmetic_sequence_sum_l2248_224877


namespace test_scores_mode_l2248_224864

/-- Represents a stem-and-leaf plot entry -/
structure StemLeafEntry where
  stem : ℕ
  leaves : List ℕ

/-- The stem-and-leaf plot data -/
def testScores : List StemLeafEntry := [
  ⟨4, [5, 5, 5]⟩,
  ⟨5, [2, 6, 6]⟩,
  ⟨6, [1, 3, 3, 3, 3]⟩,
  ⟨7, [2, 4, 5, 5, 5, 5, 5]⟩,
  ⟨8, [0, 3, 6]⟩,
  ⟨9, [1, 1, 4, 7]⟩
]

/-- Convert a stem-leaf entry to a list of full scores -/
def toFullScores (entry : StemLeafEntry) : List ℕ :=
  entry.leaves.map (λ leaf => entry.stem * 10 + leaf)

/-- Find the mode of a list of numbers -/
def mode (numbers : List ℕ) : ℕ := sorry

/-- The main theorem stating that the mode of the test scores is 75 -/
theorem test_scores_mode :
  mode (testScores.bind toFullScores) = 75 := by sorry

end test_scores_mode_l2248_224864


namespace three_digit_squares_ending_with_self_l2248_224826

theorem three_digit_squares_ending_with_self (A : ℕ) : 
  (100 ≤ A ∧ A ≤ 999) ∧ (A^2 ≡ A [ZMOD 1000]) ↔ A = 376 ∨ A = 625 := by
  sorry

end three_digit_squares_ending_with_self_l2248_224826


namespace f_difference_l2248_224834

def f (x : ℝ) : ℝ := x^5 + 3*x^3 + 2*x^2 + 4*x

theorem f_difference : f 3 - f (-3) = 672 := by
  sorry

end f_difference_l2248_224834


namespace trick_decks_total_spent_l2248_224851

/-- The total amount spent by Victor and his friend on trick decks -/
def total_spent (cost_per_deck : ℕ) (victor_decks : ℕ) (friend_decks : ℕ) : ℕ :=
  cost_per_deck * (victor_decks + friend_decks)

/-- Theorem: Victor and his friend spent 64 dollars on trick decks -/
theorem trick_decks_total_spent :
  total_spent 8 6 2 = 64 := by
  sorry

end trick_decks_total_spent_l2248_224851


namespace point_motion_time_l2248_224868

/-- 
Given two points A and B initially separated by distance a, moving along different sides of a right angle 
towards its vertex with constant speed v, where B reaches the vertex t units of time before A, 
this theorem states the time x that A takes to reach the vertex.
-/
theorem point_motion_time (a v t : ℝ) (h : a > v * t) : 
  ∃ x : ℝ, x = (t * v + Real.sqrt (2 * a^2 - v^2 * t^2)) / (2 * v) ∧ 
  x * v = Real.sqrt ((x * v)^2 + ((x - t) * v)^2) :=
sorry

end point_motion_time_l2248_224868


namespace one_third_of_product_l2248_224812

theorem one_third_of_product : (1 / 3 : ℚ) * 7 * 9 * 4 = 84 := by
  sorry

end one_third_of_product_l2248_224812


namespace rectangle_width_l2248_224857

/-- A rectangle with length twice its width and perimeter equal to its area has width 3. -/
theorem rectangle_width (w : ℝ) (h1 : w > 0) : 
  (6 * w = 2 * w ^ 2) → w = 3 :=
by sorry

end rectangle_width_l2248_224857


namespace inscribed_isosceles_tangent_circle_radius_l2248_224852

/-- Given an isosceles triangle inscribed in a circle, with a second circle
    tangent to both legs of the triangle and the first circle, this theorem
    states the radius of the second circle in terms of the base and base angle
    of the isosceles triangle. -/
theorem inscribed_isosceles_tangent_circle_radius
  (a : ℝ) (α : ℝ) (h_a_pos : a > 0) (h_α_pos : α > 0) (h_α_lt_pi_2 : α < π / 2) :
  ∃ (r : ℝ),
    r > 0 ∧
    r = a / (2 * Real.sin α * (1 + Real.cos α)) :=
by sorry

end inscribed_isosceles_tangent_circle_radius_l2248_224852


namespace marcella_shoes_theorem_l2248_224863

/-- Given an initial number of shoe pairs and a number of individual shoes lost,
    calculate the maximum number of complete pairs remaining. -/
def max_pairs_remaining (initial_pairs : ℕ) (shoes_lost : ℕ) : ℕ :=
  initial_pairs - shoes_lost

/-- Theorem: Given 25 initial pairs of shoes and losing 9 individual shoes,
    the maximum number of complete pairs remaining is 16. -/
theorem marcella_shoes_theorem :
  max_pairs_remaining 25 9 = 16 := by
  sorry

#eval max_pairs_remaining 25 9

end marcella_shoes_theorem_l2248_224863


namespace units_digit_power_plus_six_l2248_224888

theorem units_digit_power_plus_six (x : ℕ) : 
  1 ≤ x → x ≤ 9 → (x^75 + 6) % 10 = 9 → x = 3 := by sorry

end units_digit_power_plus_six_l2248_224888


namespace arithmetic_sequence_sum_l2248_224856

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a with a 0 = 3, a 1 = 7, a n = x,
    a (n+1) = y, a (n+2) = t, a (n+3) = 35, and t = 31,
    prove that x + y = 50 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (n : ℕ)
  (h1 : is_arithmetic_sequence a)
  (h2 : a 0 = 3)
  (h3 : a 1 = 7)
  (h4 : a n = x)
  (h5 : a (n+1) = y)
  (h6 : a (n+2) = t)
  (h7 : a (n+3) = 35)
  (h8 : t = 31) :
  x + y = 50 := by
  sorry

end arithmetic_sequence_sum_l2248_224856


namespace cos_two_alpha_zero_l2248_224816

theorem cos_two_alpha_zero (α : Real) 
  (h : Real.sin (π/6 - α) = Real.cos (π/6 + α)) : 
  Real.cos (2 * α) = 0 := by
  sorry

end cos_two_alpha_zero_l2248_224816


namespace rectangle_perimeter_l2248_224811

/-- Given a rectangle formed by 2 rows and 3 columns of identical squares with a total area of 150 cm²,
    prove that its perimeter is 50 cm. -/
theorem rectangle_perimeter (total_area : ℝ) (num_squares : ℕ) (rows cols : ℕ) :
  total_area = 150 ∧ 
  num_squares = 6 ∧ 
  rows = 2 ∧ 
  cols = 3 →
  (2 * rows + 2 * cols) * Real.sqrt (total_area / num_squares) = 50 := by
  sorry

end rectangle_perimeter_l2248_224811


namespace square_area_ratio_l2248_224830

theorem square_area_ratio (x : ℝ) (hx : x > 0) :
  (x^2) / ((3*x)^2) = 1/9 := by sorry

end square_area_ratio_l2248_224830


namespace quadratic_roots_imply_not_prime_l2248_224862

/-- 
Given integers a and b, if the quadratic equation x^2 + ax + b + 1 = 0 
has two positive integer roots, then a^2 + b^2 is not prime.
-/
theorem quadratic_roots_imply_not_prime (a b : ℤ) 
  (h : ∃ p q : ℕ+, p.val ≠ q.val ∧ p.val^2 + a * p.val + b + 1 = 0 ∧ q.val^2 + a * q.val + b + 1 = 0) : 
  ¬ Prime (a^2 + b^2) := by
  sorry

end quadratic_roots_imply_not_prime_l2248_224862


namespace firm_ratio_l2248_224815

theorem firm_ratio (partners associates : ℕ) : 
  partners = 14 ∧ 
  14 * 34 = associates + 35 → 
  (partners : ℚ) / associates = 2 / 63 := by
sorry

end firm_ratio_l2248_224815


namespace common_terms_arithmetic_progression_l2248_224824

/-- Definition of the first arithmetic progression -/
def a (n : ℕ) : ℤ := 4*n - 3

/-- Definition of the second arithmetic progression -/
def b (n : ℕ) : ℤ := 3*n - 1

/-- Function to generate the sequence of common terms -/
def common_terms (m : ℕ) : ℤ := 12*m + 5

/-- Theorem stating that the sequence of common terms forms an arithmetic progression with common difference 12 -/
theorem common_terms_arithmetic_progression :
  ∀ m : ℕ, ∃ n k : ℕ, 
    a n = b k ∧ 
    a n = common_terms m ∧ 
    common_terms (m + 1) - common_terms m = 12 :=
sorry

end common_terms_arithmetic_progression_l2248_224824


namespace alex_academic_year_hours_l2248_224845

/-- Calculates the number of hours Alex needs to work per week during the academic year --/
def academic_year_hours_per_week (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ) 
  (academic_weeks : ℕ) (academic_earnings : ℕ) : ℚ :=
  let summer_total_hours := summer_weeks * summer_hours_per_week
  let hourly_rate := summer_earnings / summer_total_hours
  let academic_total_hours := academic_earnings / hourly_rate
  academic_total_hours / academic_weeks

/-- Theorem stating that Alex needs to work 20 hours per week during the academic year --/
theorem alex_academic_year_hours : 
  academic_year_hours_per_week 8 40 4000 32 8000 = 20 := by
  sorry

end alex_academic_year_hours_l2248_224845


namespace quadratic_root_conditions_l2248_224819

theorem quadratic_root_conditions (k : ℤ) : 
  (∃ x y : ℝ, (k^2 + 1) * x^2 - (4 - k) * x + 1 = 0 ∧
              (k^2 + 1) * y^2 - (4 - k) * y + 1 = 0 ∧
              x > 1 ∧ y < 1) →
  k = -1 ∨ k = 0 := by
sorry

end quadratic_root_conditions_l2248_224819


namespace shower_frequency_l2248_224802

/-- Represents the duration of each shower in minutes -/
def shower_duration : ℝ := 10

/-- Represents the water usage rate in gallons per minute -/
def water_usage_rate : ℝ := 2

/-- Represents the total water usage in 4 weeks in gallons -/
def total_water_usage : ℝ := 280

/-- Represents the number of weeks -/
def num_weeks : ℝ := 4

/-- Theorem stating the frequency of John's showers -/
theorem shower_frequency :
  (total_water_usage / (shower_duration * water_usage_rate)) / num_weeks = 3.5 := by
  sorry

end shower_frequency_l2248_224802


namespace fast_food_order_cost_l2248_224865

/-- Calculates the total cost of a fast-food order with discount and tax --/
theorem fast_food_order_cost
  (burger_cost : ℝ)
  (sandwich_cost : ℝ)
  (smoothie_cost : ℝ)
  (num_smoothies : ℕ)
  (discount_rate : ℝ)
  (discount_threshold : ℝ)
  (tax_rate : ℝ)
  (h1 : burger_cost = 5)
  (h2 : sandwich_cost = 4)
  (h3 : smoothie_cost = 4)
  (h4 : num_smoothies = 2)
  (h5 : discount_rate = 0.15)
  (h6 : discount_threshold = 10)
  (h7 : tax_rate = 0.1) :
  let total_before_discount := burger_cost + sandwich_cost + (smoothie_cost * num_smoothies)
  let discount := if total_before_discount > discount_threshold then total_before_discount * discount_rate else 0
  let total_after_discount := total_before_discount - discount
  let tax := total_after_discount * tax_rate
  let total_cost := total_after_discount + tax
  ∃ (n : ℕ), (n : ℝ) / 100 = total_cost ∧ n = 1590 :=
by sorry


end fast_food_order_cost_l2248_224865


namespace intersection_equals_interval_l2248_224887

-- Define the sets M and N
def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- Define the interval [1,2)
def interval : Set ℝ := {x | 1 ≤ x ∧ x < 2}

-- State the theorem
theorem intersection_equals_interval : M ∩ N = interval := by sorry

end intersection_equals_interval_l2248_224887


namespace second_box_price_l2248_224889

/-- Represents a box of contacts with its quantity and price -/
structure ContactBox where
  quantity : ℕ
  price : ℚ

/-- Calculates the price per contact for a given box -/
def pricePerContact (box : ContactBox) : ℚ :=
  box.price / box.quantity

theorem second_box_price (box1 box2 : ContactBox)
  (h1 : box1.quantity = 50)
  (h2 : box1.price = 25)
  (h3 : box2.quantity = 99)
  (h4 : pricePerContact box2 < pricePerContact box1)
  (h5 : 3 * pricePerContact box2 = 1) :
  box2.price = 99/3 := by
  sorry

#eval (99 : ℚ) / 3  -- Should output 33

end second_box_price_l2248_224889


namespace proportion_proof_1_proportion_proof_2_l2248_224835

theorem proportion_proof_1 : 
  let x : ℚ := 1/12
  (x : ℚ) / (5/9 : ℚ) = (1/20 : ℚ) / (1/3 : ℚ) := by sorry

theorem proportion_proof_2 : 
  let x : ℚ := 5/4
  (x : ℚ) / (1/4 : ℚ) = (1/2 : ℚ) / (1/10 : ℚ) := by sorry

end proportion_proof_1_proportion_proof_2_l2248_224835


namespace parallel_planes_line_parallel_l2248_224876

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation between planes
variable (plane_parallel : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (line_subset_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_line_parallel (α β : Plane) (a : Line) :
  plane_parallel α β → line_subset_plane a β → line_parallel_plane a α :=
sorry

end parallel_planes_line_parallel_l2248_224876


namespace calculation_proof_l2248_224894

theorem calculation_proof : (2468 * 629) / (1234 * 37) = 34 := by
  sorry

end calculation_proof_l2248_224894


namespace six_by_six_checkerboard_half_shaded_l2248_224850

/-- Represents a square grid with checkerboard shading -/
structure CheckerboardGrid :=
  (size : ℕ)
  (startUnshaded : Bool)

/-- Calculates the fraction of shaded squares in a checkerboard grid -/
def shadedFraction (grid : CheckerboardGrid) : ℚ :=
  1/2

/-- Theorem: In a 6x6 checkerboard grid starting with an unshaded square,
    half of the squares are shaded -/
theorem six_by_six_checkerboard_half_shaded :
  let grid : CheckerboardGrid := ⟨6, true⟩
  shadedFraction grid = 1/2 := by
  sorry

end six_by_six_checkerboard_half_shaded_l2248_224850


namespace min_coach_handshakes_l2248_224885

theorem min_coach_handshakes (total_handshakes : ℕ) (h : total_handshakes = 465) :
  ∃ (n : ℕ), n * (n - 1) / 2 = total_handshakes ∧
  (∀ (m₁ m₂ : ℕ), m₁ + m₂ = n → m₁ + m₂ = 0) :=
by sorry

end min_coach_handshakes_l2248_224885


namespace cube_sum_property_l2248_224801

/-- A cube is a three-dimensional geometric shape -/
structure Cube where

/-- The number of edges in a cube -/
def Cube.num_edges (c : Cube) : ℕ := 12

/-- The number of corners in a cube -/
def Cube.num_corners (c : Cube) : ℕ := 8

/-- The number of faces in a cube -/
def Cube.num_faces (c : Cube) : ℕ := 6

/-- The theorem stating that the sum of edges, corners, and faces of a cube is 26 -/
theorem cube_sum_property (c : Cube) : 
  c.num_edges + c.num_corners + c.num_faces = 26 := by
  sorry

end cube_sum_property_l2248_224801


namespace total_percent_decrease_l2248_224858

def year1_decrease : ℝ := 0.20
def year2_decrease : ℝ := 0.10
def year3_decrease : ℝ := 0.15

def compound_decrease (initial_value : ℝ) : ℝ :=
  initial_value * (1 - year1_decrease) * (1 - year2_decrease) * (1 - year3_decrease)

theorem total_percent_decrease (initial_value : ℝ) (h : initial_value > 0) :
  (initial_value - compound_decrease initial_value) / initial_value = 0.388 := by
  sorry

end total_percent_decrease_l2248_224858


namespace number_equals_scientific_rep_l2248_224897

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be represented -/
def number : ℕ := 1300000

/-- The scientific notation representation of the number -/
def scientific_rep : ScientificNotation :=
  { coefficient := 1.3
  , exponent := 6
  , h_coeff := by sorry }

theorem number_equals_scientific_rep :
  (number : ℝ) = scientific_rep.coefficient * (10 : ℝ) ^ scientific_rep.exponent :=
by sorry

end number_equals_scientific_rep_l2248_224897


namespace infinitely_many_unlucky_numbers_l2248_224832

/-- A natural number is unlucky if it cannot be represented as x^2 - 1 or y^2 - 1
    for any natural numbers x, y > 1. -/
def isUnlucky (n : ℕ) : Prop :=
  ∀ x y : ℕ, x > 1 ∧ y > 1 → n ≠ x^2 - 1 ∧ n ≠ y^2 - 1

/-- There are infinitely many unlucky numbers. -/
theorem infinitely_many_unlucky_numbers :
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ isUnlucky n :=
sorry

end infinitely_many_unlucky_numbers_l2248_224832


namespace cars_per_row_section_h_l2248_224825

/-- Prove that the number of cars in each row of Section H is 9 --/
theorem cars_per_row_section_h (
  section_g_rows : ℕ)
  (section_g_cars_per_row : ℕ)
  (section_h_rows : ℕ)
  (cars_per_minute : ℕ)
  (search_time : ℕ)
  (h_section_g_rows : section_g_rows = 15)
  (h_section_g_cars_per_row : section_g_cars_per_row = 10)
  (h_section_h_rows : section_h_rows = 20)
  (h_cars_per_minute : cars_per_minute = 11)
  (h_search_time : search_time = 30)
  : (cars_per_minute * search_time - section_g_rows * section_g_cars_per_row) / section_h_rows = 9 := by
  sorry

end cars_per_row_section_h_l2248_224825


namespace danny_apples_danny_bought_73_apples_l2248_224818

def pinky_apples : ℕ := 36
def total_apples : ℕ := 109

theorem danny_apples : ℕ → Prop :=
  fun x => x = total_apples - pinky_apples

theorem danny_bought_73_apples : danny_apples 73 := by
  sorry

end danny_apples_danny_bought_73_apples_l2248_224818


namespace digit_sum_problem_l2248_224808

theorem digit_sum_problem (P Q : ℕ) : 
  P < 10 → Q < 10 → 
  100 * P + 10 * Q + Q + 
  100 * P + 10 * P + Q + 
  100 * Q + 10 * Q + Q = 876 → 
  P + Q = 5 := by sorry

end digit_sum_problem_l2248_224808


namespace line_passes_through_134_iff_a_gt_third_l2248_224839

/-- A line passes through the first, third, and fourth quadrants if and only if its slope is positive -/
axiom passes_through_134_iff_positive_slope (m : ℝ) (b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) ∨ (x > 0 ∧ y < 0)) ↔ m > 0

/-- The main theorem: the line y = (3a-1)x - 1 passes through the first, third, and fourth quadrants
    if and only if a > 1/3 -/
theorem line_passes_through_134_iff_a_gt_third (a : ℝ) : 
  (∀ x y : ℝ, y = (3*a - 1) * x - 1 → 
    (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) ∨ (x > 0 ∧ y < 0)) ↔ a > 1/3 :=
by sorry

end line_passes_through_134_iff_a_gt_third_l2248_224839


namespace walking_distance_l2248_224806

/-- The distance traveled given a constant speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proof that walking at 4 miles per hour for 2 hours results in 8 miles traveled -/
theorem walking_distance : distance 4 2 = 8 := by
  sorry

end walking_distance_l2248_224806


namespace proportion_sum_l2248_224828

theorem proportion_sum (P Q : ℚ) : 
  (4 : ℚ) / 7 = P / 49 ∧ (4 : ℚ) / 7 = 84 / Q → P + Q = 175 := by
  sorry

end proportion_sum_l2248_224828


namespace christels_initial_dolls_l2248_224842

theorem christels_initial_dolls (debelyn_initial : ℕ) (debelyn_gave : ℕ) (christel_gave : ℕ) :
  debelyn_initial = 20 →
  debelyn_gave = 2 →
  christel_gave = 5 →
  ∃ (christel_initial : ℕ) (andrena_final : ℕ),
    andrena_final = debelyn_gave + christel_gave ∧
    andrena_final = (christel_initial - christel_gave) + 2 ∧
    andrena_final = (debelyn_initial - debelyn_gave) + 3 →
    christel_initial = 10 := by
  sorry

end christels_initial_dolls_l2248_224842


namespace season_games_l2248_224821

/-- The number of hockey games in a season -/
def total_games (games_per_month : ℕ) (season_length : ℕ) : ℕ :=
  games_per_month * season_length

/-- Proof that there are 450 hockey games in the season -/
theorem season_games : total_games 25 18 = 450 := by
  sorry

end season_games_l2248_224821


namespace fence_repair_problem_l2248_224831

theorem fence_repair_problem : ∃ n : ℕ+, 
  (∃ x y : ℕ, x + y = n ∧ 2 * x + 3 * y = 87) ∧
  (∃ a b : ℕ, a + b = n ∧ 3 * a + 5 * b = 94) :=
by sorry

end fence_repair_problem_l2248_224831


namespace constant_function_l2248_224869

theorem constant_function (a : ℝ) (f : ℝ → ℝ) 
  (h1 : f 0 = (1 : ℝ) / 2)
  (h2 : ∀ x y : ℝ, f (x + y) = f x * f (a - y) + f y * f (a - x)) :
  ∀ x : ℝ, f x = (1 : ℝ) / 2 := by
sorry

end constant_function_l2248_224869


namespace book_ratio_is_two_to_one_l2248_224873

/-- Represents the number of books Thabo owns in each category -/
structure BookCounts where
  paperbackFiction : ℕ
  paperbackNonfiction : ℕ
  hardcoverNonfiction : ℕ

/-- Thabo's book collection satisfies the given conditions -/
def satisfiesConditions (books : BookCounts) : Prop :=
  books.paperbackFiction + books.paperbackNonfiction + books.hardcoverNonfiction = 180 ∧
  books.paperbackNonfiction = books.hardcoverNonfiction + 20 ∧
  books.hardcoverNonfiction = 30

/-- The ratio of paperback fiction to paperback nonfiction books is 2:1 -/
def hasRatioTwoToOne (books : BookCounts) : Prop :=
  2 * books.paperbackNonfiction = books.paperbackFiction

theorem book_ratio_is_two_to_one (books : BookCounts) 
  (h : satisfiesConditions books) : hasRatioTwoToOne books := by
  sorry

#check book_ratio_is_two_to_one

end book_ratio_is_two_to_one_l2248_224873


namespace room_dimension_l2248_224854

theorem room_dimension (b h d : ℝ) (hb : b = 8) (hh : h = 9) (hd : d = 17) :
  ∃ l : ℝ, l = 12 ∧ d^2 = l^2 + b^2 + h^2 := by sorry

end room_dimension_l2248_224854


namespace fourth_side_length_l2248_224848

/-- A quadrilateral inscribed in a circle with given side lengths -/
structure InscribedQuadrilateral where
  radius : ℝ
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  radius_positive : radius > 0
  sides_positive : side1 > 0 ∧ side2 > 0 ∧ side3 > 0 ∧ side4 > 0
  inscribed : side1 ≤ 2 * radius ∧ side2 ≤ 2 * radius ∧ side3 ≤ 2 * radius ∧ side4 ≤ 2 * radius

/-- The theorem stating the length of the fourth side -/
theorem fourth_side_length (q : InscribedQuadrilateral) 
    (h1 : q.radius = 250)
    (h2 : q.side1 = 250)
    (h3 : q.side2 = 250)
    (h4 : q.side3 = 100) :
    q.side4 = 200 := by
  sorry

end fourth_side_length_l2248_224848


namespace polynomial_factorization_l2248_224861

theorem polynomial_factorization (x : ℝ) : 
  x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x-1)^4 * (x+1)^4 := by
  sorry

end polynomial_factorization_l2248_224861


namespace value_of_N_l2248_224886

theorem value_of_N : ∃ N : ℝ, (25 / 100) * (N + 100) = (35 / 100) * 1500 ∧ N = 2000 := by
  sorry

end value_of_N_l2248_224886


namespace partition_exists_l2248_224827

/-- The set of weights from 1 to 101 grams -/
def weights : Finset ℕ := Finset.range 101

/-- The sum of all weights from 1 to 101 grams -/
def total_sum : ℕ := weights.sum id

/-- The remaining weights after removing the 19-gram weight -/
def remaining_weights : Finset ℕ := weights.erase 19

/-- The sum of remaining weights -/
def remaining_sum : ℕ := remaining_weights.sum id

/-- A partition of the remaining weights into two subsets -/
structure Partition :=
  (subset1 subset2 : Finset ℕ)
  (partition_complete : subset1 ∪ subset2 = remaining_weights)
  (partition_disjoint : subset1 ∩ subset2 = ∅)
  (equal_size : subset1.card = subset2.card)
  (size_fifty : subset1.card = 50)

/-- The theorem stating that a valid partition exists -/
theorem partition_exists : ∃ (p : Partition), p.subset1.sum id = p.subset2.sum id :=
sorry

end partition_exists_l2248_224827


namespace uninsured_employees_count_l2248_224822

theorem uninsured_employees_count 
  (total : ℕ) 
  (part_time : ℕ) 
  (uninsured_part_time_ratio : ℚ) 
  (neither_uninsured_nor_part_time_prob : ℚ) 
  (h1 : total = 340)
  (h2 : part_time = 54)
  (h3 : uninsured_part_time_ratio = 125 / 1000)
  (h4 : neither_uninsured_nor_part_time_prob = 5735294117647058 / 10000000000000000) :
  ∃ uninsured : ℕ, uninsured = 104 := by
  sorry


end uninsured_employees_count_l2248_224822


namespace parabola_properties_l2248_224883

-- Define the parabola function
def f (x : ℝ) : ℝ := (x + 2)^2 - 1

-- State the theorem
theorem parabola_properties :
  (∀ x y : ℝ, f x ≤ f y → (x + 2)^2 ≤ (y + 2)^2) ∧ -- Opens upwards
  (∀ x : ℝ, f ((-2) + x) = f ((-2) - x)) ∧ -- Axis of symmetry is x = -2
  (∀ x₁ x₂ : ℝ, x₁ > -2 ∧ x₂ > -2 ∧ x₁ < x₂ → f x₁ < f x₂) ∧ -- y increases as x increases when x > -2
  (∀ x : ℝ, f x ≥ f (-2)) ∧ -- Minimum value at x = -2
  (f (-2) = -1) -- Minimum value is -1
  := by sorry

end parabola_properties_l2248_224883


namespace a_pow_b_gt_one_iff_a_minus_one_b_gt_zero_l2248_224836

theorem a_pow_b_gt_one_iff_a_minus_one_b_gt_zero 
  (a b : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) : 
  a^b > 1 ↔ (a - 1) * b > 0 := by sorry

end a_pow_b_gt_one_iff_a_minus_one_b_gt_zero_l2248_224836


namespace circle_tangent_probability_main_theorem_l2248_224867

/-- The probability that two circles have exactly two common tangent lines -/
theorem circle_tangent_probability : Real → Prop := fun p =>
  let r_min : Real := 4
  let r_max : Real := 9
  let circle1_center : Real × Real := (2, -1)
  let circle2_center : Real × Real := (-1, 3)
  let circle1_radius : Real := 2
  let valid_r_min : Real := 3
  let valid_r_max : Real := 7
  p = (valid_r_max - valid_r_min) / (r_max - r_min)

/-- The main theorem -/
theorem main_theorem : circle_tangent_probability (4/5) := by
  sorry

end circle_tangent_probability_main_theorem_l2248_224867


namespace mistake_percentage_l2248_224859

theorem mistake_percentage (n : ℕ) (x : ℕ) : 
  n > 0 ∧ x > 0 ∧ x ≤ n ∧
  (x - 1 : ℚ) / n = 24 / 100 ∧
  (x - 1 : ℚ) / (n - 1) = 25 / 100 →
  (x : ℚ) / n = 28 / 100 :=
by sorry

end mistake_percentage_l2248_224859


namespace school_age_problem_l2248_224895

theorem school_age_problem (num_students : ℕ) (num_teachers : ℕ) (avg_age_students : ℝ) 
  (avg_age_with_teachers : ℝ) (avg_age_with_principal : ℝ) :
  num_students = 30 →
  num_teachers = 3 →
  avg_age_students = 14 →
  avg_age_with_teachers = 16 →
  avg_age_with_principal = 17 →
  ∃ (total_age_teachers : ℝ) (age_principal : ℝ),
    total_age_teachers = 108 ∧ age_principal = 50 := by
  sorry

end school_age_problem_l2248_224895


namespace matrix_self_inverse_l2248_224896

theorem matrix_self_inverse (a b : ℚ) :
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![4, -2; a, b]
  A * A = 1 → a = 7.5 ∧ b = -4 := by
sorry

end matrix_self_inverse_l2248_224896


namespace rhombus_tangent_distance_l2248_224820

/-- A rhombus with an inscribed circle -/
structure RhombusWithCircle where
  /-- Side length of the rhombus -/
  side : ℝ
  /-- Radius of the inscribed circle -/
  radius : ℝ
  /-- Condition that the first diagonal is less than the second diagonal -/
  diag_condition : ℝ → ℝ → Prop

/-- The distance between tangent points on adjacent sides of the rhombus -/
def tangent_distance (r : RhombusWithCircle) : ℝ := sorry

/-- Theorem stating the distance between tangent points on adjacent sides -/
theorem rhombus_tangent_distance
  (r : RhombusWithCircle)
  (h1 : r.side = 5)
  (h2 : r.radius = 2.4)
  (h3 : r.diag_condition (2 * r.radius) (2 * r.side * (1 - r.radius / r.side))) :
  tangent_distance r = 3.84 := by sorry

end rhombus_tangent_distance_l2248_224820


namespace more_green_than_blue_l2248_224870

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the bag of disks -/
structure DiskBag where
  total : Nat
  ratio : Fin 3 → Nat
  sum_ratio : ratio 0 + ratio 1 + ratio 2 = 18

theorem more_green_than_blue (bag : DiskBag) 
  (h_total : bag.total = 54)
  (h_ratio : bag.ratio = ![3, 7, 8]) :
  (bag.total * bag.ratio 2) / 18 - (bag.total * bag.ratio 0) / 18 = 15 := by
  sorry

#check more_green_than_blue

end more_green_than_blue_l2248_224870


namespace integer_condition_l2248_224893

theorem integer_condition (m k n : ℕ) (h1 : 0 < m) (h2 : 0 < k) (h3 : 0 < n)
  (h4 : k < n - 1) (h5 : m ≤ n) :
  ∃ z : ℤ, (n - 3 * k + m : ℚ) / (k + m : ℚ) * (n.choose k : ℚ) = z ↔ 
  ∃ t : ℕ, 2 * m = t * (k + m) :=
by sorry

end integer_condition_l2248_224893


namespace interest_calculation_l2248_224837

/-- Simple interest calculation -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem interest_calculation :
  let principal : ℝ := 10000
  let rate : ℝ := 0.05
  let time : ℝ := 1
  simple_interest principal rate time = 500 := by
sorry

end interest_calculation_l2248_224837


namespace collinear_probability_value_l2248_224817

/-- A 5x5 grid of dots -/
def Grid := Fin 5 × Fin 5

/-- The total number of dots in the grid -/
def total_dots : ℕ := 25

/-- The number of sets of 5 collinear dots in the grid -/
def collinear_sets : ℕ := 12

/-- The number of ways to choose 5 dots from the grid -/
def total_choices : ℕ := Nat.choose total_dots 5

/-- The probability of selecting 5 collinear dots from the grid -/
def collinear_probability : ℚ := collinear_sets / total_choices

theorem collinear_probability_value :
  collinear_probability = 12 / 53130 :=
sorry

end collinear_probability_value_l2248_224817


namespace percentage_decrease_in_people_l2248_224838

/-- Calculates the percentage decrease in the number of people to be fed given initial and new can counts. -/
theorem percentage_decrease_in_people (initial_cans initial_people new_cans : ℕ) : 
  initial_cans = 600 →
  initial_people = 40 →
  new_cans = 420 →
  (1 - (new_cans * initial_people : ℚ) / (initial_cans * initial_people)) * 100 = 30 := by
sorry

end percentage_decrease_in_people_l2248_224838


namespace jeans_price_increase_l2248_224855

theorem jeans_price_increase (C : ℝ) (C_pos : C > 0) : 
  let retailer_price := 1.40 * C
  let customer_price := 1.54 * C
  (customer_price - retailer_price) / retailer_price * 100 = 10 := by
sorry

end jeans_price_increase_l2248_224855


namespace f_properties_l2248_224805

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem f_properties :
  (f 0 = 1) ∧
  (f 1 = 1/2) ∧
  (∀ x : ℝ, 0 < f x ∧ f x ≤ 1) ∧
  (∀ y : ℝ, 0 < y ∧ y ≤ 1 → ∃ x : ℝ, f x = y) := by sorry

end f_properties_l2248_224805


namespace symmetry_of_shifted_even_function_l2248_224899

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def axis_of_symmetry (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem symmetry_of_shifted_even_function (f : ℝ → ℝ) (h : is_even_function (fun x => f (x + 3))) :
  axis_of_symmetry f 3 :=
sorry

end symmetry_of_shifted_even_function_l2248_224899


namespace iesha_book_count_l2248_224849

/-- The number of school books Iesha has -/
def school_books : ℕ := 136

/-- The number of sports books Iesha has -/
def sports_books : ℕ := 208

/-- The total number of books Iesha has -/
def total_books : ℕ := school_books + sports_books

theorem iesha_book_count : total_books = 344 := by
  sorry

end iesha_book_count_l2248_224849


namespace actual_height_of_boy_l2248_224874

/-- Calculates the actual height of a boy in a class given the following conditions:
  * There are 35 boys in the class
  * The initially calculated average height was 182 cm
  * One boy's height was wrongly written as 166 cm
  * The actual average height is 180 cm
-/
theorem actual_height_of_boy (n : ℕ) (initial_avg : ℝ) (wrong_height : ℝ) (actual_avg : ℝ) :
  n = 35 →
  initial_avg = 182 →
  wrong_height = 166 →
  actual_avg = 180 →
  ∃ (x : ℝ), x = 236 ∧ n * actual_avg = (n * initial_avg - wrong_height + x) :=
by sorry

end actual_height_of_boy_l2248_224874


namespace percentage_decrease_l2248_224892

/-- Given a percentage increase P in production value from one year to the next,
    calculate the percentage decrease from the latter year to the former year. -/
theorem percentage_decrease (P : ℝ) : 
  P > -100 → (100 * (1 - 1 / (1 + P / 100))) = P / (1 + P / 100) := by
  sorry

end percentage_decrease_l2248_224892


namespace dans_age_l2248_224823

/-- Given two people, Ben and Dan, where Ben is younger than Dan, 
    their ages sum to 53, and Ben is 25 years old, 
    prove that Dan is 28 years old. -/
theorem dans_age (ben_age dan_age : ℕ) : 
  ben_age < dan_age →
  ben_age + dan_age = 53 →
  ben_age = 25 →
  dan_age = 28 := by
  sorry

end dans_age_l2248_224823


namespace invitation_methods_count_l2248_224878

-- Define the total number of students
def total_students : ℕ := 10

-- Define the number of students to be invited
def invited_students : ℕ := 6

-- Define the function to calculate combinations
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Theorem statement
theorem invitation_methods_count :
  combination total_students invited_students - combination (total_students - 2) (invited_students - 2) = 140 := by
  sorry

end invitation_methods_count_l2248_224878


namespace arithmetic_geometric_inequality_l2248_224814

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem arithmetic_geometric_inequality
  (a b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b)
  (h1 : a 1 = b 1)
  (h1_pos : a 1 > 0)
  (h11 : a 11 = b 11)
  (h11_pos : a 11 > 0) :
  a 6 ≥ b 6 := by
  sorry

end arithmetic_geometric_inequality_l2248_224814


namespace b_91_mod_49_l2248_224810

/-- Definition of the sequence bₙ -/
def b (n : ℕ) : ℕ := 12^n + 14^n

/-- Theorem stating that b₉₁ mod 49 = 38 -/
theorem b_91_mod_49 : b 91 % 49 = 38 := by
  sorry

end b_91_mod_49_l2248_224810
