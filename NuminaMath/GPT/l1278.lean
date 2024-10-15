import Mathlib

namespace NUMINAMATH_GPT_trigonometric_inequalities_l1278_127817

theorem trigonometric_inequalities (θ : ℝ) (h1 : Real.sin (θ + Real.pi) < 0) (h2 : Real.cos (θ - Real.pi) > 0) : 
  Real.sin θ > 0 ∧ Real.cos θ < 0 :=
sorry

end NUMINAMATH_GPT_trigonometric_inequalities_l1278_127817


namespace NUMINAMATH_GPT_ellipse_equation_line_AC_l1278_127816

noncomputable def ellipse_eq (x y a b : ℝ) : Prop := 
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def foci_distance (a c : ℝ) : Prop := 
  a - c = 1 ∧ a + c = 3

noncomputable def b_value (a c b : ℝ) : Prop :=
  b = Real.sqrt (a^2 - c^2)

noncomputable def rhombus_on_line (m : ℝ) : Prop := 
  7 * (2 * m / 7) + 1 - 7 * (3 * m / 7) = 0

theorem ellipse_equation (a b c : ℝ) (h1 : foci_distance a c) (h2 : b_value a c b) :
  ellipse_eq x y a b :=
sorry

theorem line_AC (a b c x y x1 y1 x2 y2 : ℝ) 
  (h1 : ellipse_eq x1 y1 a b)
  (h2 : ellipse_eq x2 y2 a b)
  (h3 : 7 * x1 - 7 * y1 + 1 = 0)
  (h4 : 7 * x2 - 7 * y2 + 1 = 0)
  (h5 : rhombus_on_line y) :
  x + y + 1 = 0 :=
sorry

end NUMINAMATH_GPT_ellipse_equation_line_AC_l1278_127816


namespace NUMINAMATH_GPT_hall_breadth_is_12_l1278_127822

/-- Given a hall with length 15 meters, if the sum of the areas of the floor and the ceiling 
    is equal to the sum of the areas of the four walls and the volume of the hall is 1200 
    cubic meters, then the breadth of the hall is 12 meters. -/
theorem hall_breadth_is_12 (b h : ℝ) (h1 : 15 * b * h = 1200)
  (h2 : 2 * (15 * b) = 2 * (15 * h) + 2 * (b * h)) : b = 12 :=
sorry

end NUMINAMATH_GPT_hall_breadth_is_12_l1278_127822


namespace NUMINAMATH_GPT_part1_part2_l1278_127862

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - a - Real.log x

theorem part1 (a : ℝ) :
  (∀ x > 0, f x a ≥ 0) → a ≤ 1 := sorry

theorem part2 (a : ℝ) (x₁ x₂ : ℝ) (hx : 0 < x₁ ∧ x₁ < x₂) :
  (f x₁ a - f x₂ a) / (x₂ - x₁) < 1 / (x₁ * (x₁ + 1)) := sorry

end NUMINAMATH_GPT_part1_part2_l1278_127862


namespace NUMINAMATH_GPT_ratio_dark_blue_to_total_l1278_127830

-- Definitions based on the conditions
def total_marbles := 63
def red_marbles := 38
def green_marbles := 4
def dark_blue_marbles := total_marbles - red_marbles - green_marbles

-- The statement to be proven
theorem ratio_dark_blue_to_total : (dark_blue_marbles : ℚ) / total_marbles = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_ratio_dark_blue_to_total_l1278_127830


namespace NUMINAMATH_GPT_GCD_of_n_pow_13_sub_n_l1278_127825

theorem GCD_of_n_pow_13_sub_n :
  ∀ n : ℤ, gcd (n^13 - n) 2730 = gcd (n^13 - n) n := sorry

end NUMINAMATH_GPT_GCD_of_n_pow_13_sub_n_l1278_127825


namespace NUMINAMATH_GPT_no_prime_degree_measure_l1278_127858

theorem no_prime_degree_measure :
  ∀ n, 10 ≤ n ∧ n < 20 → ¬ Nat.Prime (180 * (n - 2) / n) :=
by
  intros n h1 h2 
  sorry

end NUMINAMATH_GPT_no_prime_degree_measure_l1278_127858


namespace NUMINAMATH_GPT_difference_between_shares_l1278_127854

def investment_months (amount : ℕ) (months : ℕ) : ℕ :=
  amount * months

def ratio (investment_months : ℕ) (total_investment_months : ℕ) : ℚ :=
  investment_months / total_investment_months

def profit_share (ratio : ℚ) (total_profit : ℝ) : ℝ :=
  ratio * total_profit

theorem difference_between_shares :
  let suresh_investment := 18000
  let rohan_investment := 12000
  let sudhir_investment := 9000
  let suresh_months := 12
  let rohan_months := 9
  let sudhir_months := 8
  let total_profit := 3795
  let suresh_investment_months := investment_months suresh_investment suresh_months
  let rohan_investment_months := investment_months rohan_investment rohan_months
  let sudhir_investment_months := investment_months sudhir_investment sudhir_months
  let total_investment_months := suresh_investment_months + rohan_investment_months + sudhir_investment_months
  let suresh_ratio := ratio suresh_investment_months total_investment_months
  let rohan_ratio := ratio rohan_investment_months total_investment_months
  let sudhir_ratio := ratio sudhir_investment_months total_investment_months
  let rohan_share := profit_share rohan_ratio total_profit
  let sudhir_share := profit_share sudhir_ratio total_profit
  rohan_share - sudhir_share = 345 :=
by
  sorry

end NUMINAMATH_GPT_difference_between_shares_l1278_127854


namespace NUMINAMATH_GPT_amount_of_money_C_l1278_127872

variable (A B C : ℝ)

theorem amount_of_money_C (h1 : A + B + C = 500)
                         (h2 : A + C = 200)
                         (h3 : B + C = 360) :
    C = 60 :=
sorry

end NUMINAMATH_GPT_amount_of_money_C_l1278_127872


namespace NUMINAMATH_GPT_lines_perpendicular_l1278_127870

theorem lines_perpendicular 
  (x y : ℝ)
  (first_angle : ℝ)
  (second_angle : ℝ)
  (h1 : first_angle = 50 + x - y)
  (h2 : second_angle = first_angle - (10 + 2 * x - 2 * y)) :
  first_angle + second_angle = 90 :=
by 
  sorry

end NUMINAMATH_GPT_lines_perpendicular_l1278_127870


namespace NUMINAMATH_GPT_problem_part1_problem_part2_problem_part3_l1278_127888

variable (a b x : ℝ) (p q : ℝ) (n x1 x2 : ℝ)
variable (h1 : x1 = -2) (h2 : x2 = 3)
variable (h3 : x1 < x2)

def equation1 := x + p / x = q
def solution1_p := p = -6
def solution1_q := q = 1

def equation2 := x + 7 / x = 8
def solution2 := x1 = 7

def equation3 := 2 * x + (n^2 - n) / (2 * x - 1) = 2 * n
def solution3 := (2 * x1 - 1) / (2 * x2) = (n - 1) / (n + 1)

theorem problem_part1 : ∀ (x : ℝ), (x + -6 / x = 1) → (p = -6 ∧ q = 1) := by
  sorry

theorem problem_part2 : (max 7 1 = 7) := by
  sorry

theorem problem_part3 : ∀ (n : ℝ), (∃ x1 x2, x1 < x2 ∧ (2 * x1 - 1) / (2 * x2) = (n - 1) / (n + 1)) := by
  sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_problem_part3_l1278_127888


namespace NUMINAMATH_GPT_determine_d_value_l1278_127878

noncomputable def Q (d : ℚ) (x : ℚ) : ℚ := x^3 + 3 * x^2 + d * x + 8

theorem determine_d_value (d : ℚ) : x - 3 ∣ Q d x → d = -62 / 3 := by
  sorry

end NUMINAMATH_GPT_determine_d_value_l1278_127878


namespace NUMINAMATH_GPT_range_of_m_l1278_127821

/-- The quadratic equation x^2 + (2m - 1)x + 4 - 2m = 0 has one root 
greater than 2 and the other less than 2 if and only if m < -3. -/
theorem range_of_m (m : ℝ) :
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 2 ∧ x2 < 2 ∧ x1 ^ 2 + (2 * m - 1) * x1 + 4 - 2 * m = 0 ∧
    x2 ^ 2 + (2 * m - 1) * x2 + 4 - 2 * m = 0) ↔
    m < -3 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l1278_127821


namespace NUMINAMATH_GPT_compare_squares_l1278_127868

theorem compare_squares (a b : ℝ) : 
  (a^2 + b^2) / 2 ≥ (a + b) / 2 * (a + b) / 2 := 
sorry

end NUMINAMATH_GPT_compare_squares_l1278_127868


namespace NUMINAMATH_GPT_no_possible_values_for_b_l1278_127823

theorem no_possible_values_for_b : ¬ ∃ b : ℕ, 2 ≤ b ∧ b^3 ≤ 256 ∧ 256 < b^4 := by
  sorry

end NUMINAMATH_GPT_no_possible_values_for_b_l1278_127823


namespace NUMINAMATH_GPT_soda_cost_original_l1278_127873

theorem soda_cost_original 
  (x : ℚ) -- note: x in rational numbers to capture fractional cost accurately
  (h1 : 3 * (0.90 * x) = 6) :
  x = 20 / 9 :=
by
  sorry

end NUMINAMATH_GPT_soda_cost_original_l1278_127873


namespace NUMINAMATH_GPT_total_tickets_spent_l1278_127892

def tickets_spent_on_hat : ℕ := 2
def tickets_spent_on_stuffed_animal : ℕ := 10
def tickets_spent_on_yoyo : ℕ := 2

theorem total_tickets_spent :
  tickets_spent_on_hat + tickets_spent_on_stuffed_animal + tickets_spent_on_yoyo = 14 := by
  sorry

end NUMINAMATH_GPT_total_tickets_spent_l1278_127892


namespace NUMINAMATH_GPT_symmetry_axis_l1278_127811

noncomputable def y_func (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 4)

theorem symmetry_axis : ∃ a : ℝ, (∀ x : ℝ, y_func (a - x) = y_func (a + x)) ∧ a = Real.pi / 8 :=
by
  sorry

end NUMINAMATH_GPT_symmetry_axis_l1278_127811


namespace NUMINAMATH_GPT_base_conversion_l1278_127886

theorem base_conversion (C D : ℕ) (hC : 0 ≤ C) (hC_lt : C < 8) (hD : 0 ≤ D) (hD_lt : D < 5) :
  (8 * C + D = 5 * D + C) → (8 * C + D = 0) :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_base_conversion_l1278_127886


namespace NUMINAMATH_GPT_triangle_is_isosceles_right_l1278_127898

theorem triangle_is_isosceles_right (A B C a b c : ℝ) 
  (h : a / (Real.cos A) = b / (Real.cos B) ∧ b / (Real.cos B) = c / (Real.sin C)) :
  A = π/4 ∧ B = π/4 ∧ C = π/2 := 
sorry

end NUMINAMATH_GPT_triangle_is_isosceles_right_l1278_127898


namespace NUMINAMATH_GPT_visited_iceland_l1278_127895

variable (total : ℕ) (visitedNorway : ℕ) (visitedBoth : ℕ) (visitedNeither : ℕ)

theorem visited_iceland (h_total : total = 50)
                        (h_visited_norway : visitedNorway = 23)
                        (h_visited_both : visitedBoth = 21)
                        (h_visited_neither : visitedNeither = 23) :
                        (total - (visitedNorway - visitedBoth + visitedNeither) = 25) :=
  sorry

end NUMINAMATH_GPT_visited_iceland_l1278_127895


namespace NUMINAMATH_GPT_sum_is_24000_l1278_127859

theorem sum_is_24000 (P : ℝ) (R : ℝ) (T : ℝ) : 
  (R = 5) → (T = 2) →
  ((P * (1 + R / 100)^T - P) - (P * R * T / 100) = 60) →
  P = 24000 :=
by
  sorry

end NUMINAMATH_GPT_sum_is_24000_l1278_127859


namespace NUMINAMATH_GPT_circle_equation_and_lines_l1278_127839

noncomputable def O : ℝ × ℝ := (0, 0)
noncomputable def A : ℝ × ℝ := (6, 2)
noncomputable def B : ℝ × ℝ := (4, 4)
noncomputable def C_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 10

structure Line (κ β: ℝ) where
  passes_through : ℝ × ℝ → Prop
  definition : Prop

def line_passes_through_point (κ β : ℝ) (p : ℝ × ℝ) : Prop := p.2 = κ * p.1 + β

theorem circle_equation_and_lines : 
  (∀ p : ℝ × ℝ, p = O ∨ p = A ∨ p = B → C_eq p.1 p.2) ∧
  ((∀ p : ℝ × ℝ, line_passes_through_point 0 2 p → C_eq 2 6 ∧ (∃ x1 x2 y : ℝ, C_eq x1 y ∧ C_eq x2 y ∧ ((x1 - x2)^2 + (y - y)^2) = 4)) ∧
   (∀ p : ℝ × ℝ, line_passes_through_point (-7 / 3) (32 / 3) p → C_eq 2 6 ∧ (∃ x1 x2 y : ℝ, C_eq x1 y ∧ C_eq x2 y ∧ ((x1 - x2)^2 + (y - y)^2) = 4))) :=
by 
  sorry

end NUMINAMATH_GPT_circle_equation_and_lines_l1278_127839


namespace NUMINAMATH_GPT_cookie_revenue_l1278_127804

theorem cookie_revenue :
  let robyn_day1_packs := 25
  let robyn_day1_price := 4.0
  let lucy_day1_packs := 17
  let lucy_day1_price := 5.0
  let robyn_day2_packs := 15
  let robyn_day2_price := 3.5
  let lucy_day2_packs := 9
  let lucy_day2_price := 4.5
  let robyn_day3_packs := 23
  let robyn_day3_price := 4.5
  let lucy_day3_packs := 20
  let lucy_day3_price := 3.5
  let robyn_day1_revenue := robyn_day1_packs * robyn_day1_price
  let lucy_day1_revenue := lucy_day1_packs * lucy_day1_price
  let robyn_day2_revenue := robyn_day2_packs * robyn_day2_price
  let lucy_day2_revenue := lucy_day2_packs * lucy_day2_price
  let robyn_day3_revenue := robyn_day3_packs * robyn_day3_price
  let lucy_day3_revenue := lucy_day3_packs * lucy_day3_price
  let robyn_total_revenue := robyn_day1_revenue + robyn_day2_revenue + robyn_day3_revenue
  let lucy_total_revenue := lucy_day1_revenue + lucy_day2_revenue + lucy_day3_revenue
  let total_revenue := robyn_total_revenue + lucy_total_revenue
  total_revenue = 451.5 := 
by
  sorry

end NUMINAMATH_GPT_cookie_revenue_l1278_127804


namespace NUMINAMATH_GPT_sum_of_coefficients_l1278_127887

theorem sum_of_coefficients (x y z : ℤ) (h : x = 1 ∧ y = 1 ∧ z = 1) :
    (x - 2 * y + 3 * z) ^ 12 = 4096 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1278_127887


namespace NUMINAMATH_GPT_find_fraction_l1278_127894

theorem find_fraction (x y : ℝ) (hx : 0 < x) (hy : x < y) (h : x / y + y / x = 8) :
  (x + y) / (x - y) = Real.sqrt 15 / 3 :=
sorry

end NUMINAMATH_GPT_find_fraction_l1278_127894


namespace NUMINAMATH_GPT_average_speed_of_train_l1278_127833

theorem average_speed_of_train (x : ℝ) (h1 : x > 0): 
  (3 * x) / ((x / 40) + (2 * x / 20)) = 24 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_of_train_l1278_127833


namespace NUMINAMATH_GPT_apples_per_basket_l1278_127831

theorem apples_per_basket (total_apples : ℕ) (baskets : ℕ) (h1 : total_apples = 629) (h2 : baskets = 37) :
  total_apples / baskets = 17 :=
by
  sorry

end NUMINAMATH_GPT_apples_per_basket_l1278_127831


namespace NUMINAMATH_GPT_range_of_a_l1278_127834

noncomputable def tangent_slopes (a x0 : ℝ) : ℝ × ℝ :=
  let k1 := (a * x0 + a - 1) * Real.exp x0
  let k2 := (x0 - 2) * Real.exp (-x0)
  (k1, k2)

theorem range_of_a (a x0 : ℝ) (h : x0 ∈ Set.Icc 0 (3 / 2))
  (h_perpendicular : (tangent_slopes a x0).1 * (tangent_slopes a x0).2 = -1)
  : 1 ≤ a ∧ a ≤ 3 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1278_127834


namespace NUMINAMATH_GPT_minimize_sum_dist_l1278_127852

noncomputable section

variables {Q Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 : ℝ}

-- Conditions
def clusters (Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 : ℝ) :=
  Q3 <= Q1 + Q2 + Q4 / 3 ∧ Q3 = (Q1 + 2 * Q2 + 2 * Q4) / 5 ∧
  Q7 <= Q5 + Q6 + Q8 / 3 ∧ Q7 = (Q5 + 2 * Q6 + 2 * Q8) / 5

-- Sum of distances function
def sum_dist (Q : ℝ) (Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 : ℝ) : ℝ :=
  abs (Q - Q1) + abs (Q - Q2) + abs (Q - Q3) + abs (Q - Q4) +
  abs (Q - Q5) + abs (Q - Q6) + abs (Q - Q7) + abs (Q - Q8) + abs (Q - Q9)

-- Theorem
theorem minimize_sum_dist (h : clusters Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9) :
  ∃ Q : ℝ, (∀ Q' : ℝ, sum_dist Q Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 ≤ sum_dist Q' Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9) → Q = Q5 :=
sorry

end NUMINAMATH_GPT_minimize_sum_dist_l1278_127852


namespace NUMINAMATH_GPT_find_N_l1278_127885

theorem find_N : 
  ∀ (a b c N : ℝ), 
  a + b + c = 80 → 
  2 * a = N → 
  b - 10 = N → 
  3 * c = N → 
  N = 38 := 
by sorry

end NUMINAMATH_GPT_find_N_l1278_127885


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_l1278_127844

theorem arithmetic_sequence_general_term (a₁ : ℕ) (d : ℕ) (n : ℕ) (h₁ : a₁ = 2) (h₂ : d = 3) :
  ∃ a_n, a_n = a₁ + (n - 1) * d ∧ a_n = 3 * n - 1 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_term_l1278_127844


namespace NUMINAMATH_GPT_intersection_A_complement_B_l1278_127880

-- Definition of the universal set U
def U : Set ℝ := Set.univ

-- Definition of the set A
def A : Set ℝ := {x | x^2 - 2 * x < 0}

-- Definition of the set B
def B : Set ℝ := {x | x > 1}

-- Definition of the complement of B in U
def complement_B : Set ℝ := {x | x ≤ 1}

-- The intersection A ∩ complement_B
def intersection : Set ℝ := {x | 0 < x ∧ x ≤ 1}

-- The theorem to prove
theorem intersection_A_complement_B : A ∩ complement_B = intersection :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_intersection_A_complement_B_l1278_127880


namespace NUMINAMATH_GPT_percent_of_x_is_y_l1278_127883

theorem percent_of_x_is_y (x y : ℝ) (h : 0.25 * (x - y) = 0.15 * (x + y)) : y = 0.25 * x := by
  sorry

end NUMINAMATH_GPT_percent_of_x_is_y_l1278_127883


namespace NUMINAMATH_GPT_scientific_notation_63000_l1278_127891

theorem scientific_notation_63000 : 63000 = 6.3 * 10^4 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_63000_l1278_127891


namespace NUMINAMATH_GPT_x_value_not_unique_l1278_127843

theorem x_value_not_unique (x y : ℝ) (h1 : y = x) (h2 : y = (|x + y - 2|) / (Real.sqrt 2)) : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
(∃ y1 y2 : ℝ, (y1 = x1 ∧ y2 = x2 ∧ y1 = (|x1 + y1 - 2|) / Real.sqrt 2 ∧ y2 = (|x2 + y2 - 2|) / Real.sqrt 2)) :=
by
  sorry

end NUMINAMATH_GPT_x_value_not_unique_l1278_127843


namespace NUMINAMATH_GPT_tobias_time_spent_at_pool_l1278_127876

-- Define the conditions
def distance_per_interval : ℕ := 100
def time_per_interval : ℕ := 5
def pause_interval : ℕ := 25
def pause_time : ℕ := 5
def total_distance : ℕ := 3000
def total_time_in_hours : ℕ := 3

-- Hypotheses based on the problem conditions
def swimming_time_without_pauses := (total_distance / distance_per_interval) * time_per_interval
def number_of_pauses := (swimming_time_without_pauses / pause_interval)
def total_pause_time := number_of_pauses * pause_time
def total_time := swimming_time_without_pauses + total_pause_time

-- Proof statement
theorem tobias_time_spent_at_pool : total_time / 60 = total_time_in_hours :=
by 
  -- Put proof here
  sorry

end NUMINAMATH_GPT_tobias_time_spent_at_pool_l1278_127876


namespace NUMINAMATH_GPT_sum_of_largest_and_smallest_l1278_127815

theorem sum_of_largest_and_smallest (d1 d2 d3 d4 : ℕ) (h1 : d1 = 1) (h2 : d2 = 6) (h3 : d3 = 3) (h4 : d4 = 9) :
  let largest := 9631
  let smallest := 1369
  largest + smallest = 11000 :=
by
  let largest := 9631
  let smallest := 1369
  sorry

end NUMINAMATH_GPT_sum_of_largest_and_smallest_l1278_127815


namespace NUMINAMATH_GPT_total_heads_l1278_127897

theorem total_heads (h : ℕ) (c : ℕ) (total_feet : ℕ) 
  (h_count : h = 30)
  (hen_feet : h * 2 + c * 4 = total_feet)
  (total_feet_val : total_feet = 140) 
  : h + c = 50 :=
by
  sorry

end NUMINAMATH_GPT_total_heads_l1278_127897


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l1278_127809

variables {a b c e : ℝ}

-- Definition of geometric progression condition for the ellipse axes and focal length
def geometric_progression_condition (a b c : ℝ) : Prop :=
  (2 * b) ^ 2 = 2 * c * 2 * a

-- Eccentricity calculation
def eccentricity {a c : ℝ} (e : ℝ) : Prop :=
  e = (a^2 - c^2) / a^2

-- Theorem that states the eccentricity under the given condition
theorem eccentricity_of_ellipse (h : geometric_progression_condition a b c) : e = (1 + Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l1278_127809


namespace NUMINAMATH_GPT_average_output_assembly_line_l1278_127871

theorem average_output_assembly_line (initial_cogs second_batch_cogs rate1 rate2 : ℕ) (time1 time2 : ℚ)
  (h1 : initial_cogs = 60)
  (h2 : second_batch_cogs = 60)
  (h3 : rate1 = 90)
  (h4 : rate2 = 60)
  (h5 : time1 = 60 / 90)
  (h6 : time2 = 60 / 60)
  (h7 : (120 : ℚ) / (time1 + time2) = (72 : ℚ)) :
  (120 : ℚ) / (time1 + time2) = 72 := by
  sorry

end NUMINAMATH_GPT_average_output_assembly_line_l1278_127871


namespace NUMINAMATH_GPT_find_boys_l1278_127857

-- Variable declarations
variables (B G : ℕ)

-- Conditions
def total_students (B G : ℕ) : Prop := B + G = 466
def more_girls_than_boys (B G : ℕ) : Prop := G = B + 212

-- Proof statement: Prove B = 127 given both conditions
theorem find_boys (h1 : total_students B G) (h2 : more_girls_than_boys B G) : B = 127 :=
sorry

end NUMINAMATH_GPT_find_boys_l1278_127857


namespace NUMINAMATH_GPT_book_pages_l1278_127847

-- Define the number of pages read each day
def pages_yesterday : ℕ := 35
def pages_today : ℕ := pages_yesterday - 5
def pages_tomorrow : ℕ := 35

-- Total number of pages in the book
def total_pages : ℕ := pages_yesterday + pages_today + pages_tomorrow

-- Proof that the total number of pages is 100
theorem book_pages : total_pages = 100 := by
  -- Skip the detailed proof
  sorry

end NUMINAMATH_GPT_book_pages_l1278_127847


namespace NUMINAMATH_GPT_platform_length_l1278_127814

/-- Given:
1. The speed of the train is 72 kmph.
2. The train crosses a platform in 32 seconds.
3. The train crosses a man standing on the platform in 18 seconds.

Prove:
The length of the platform is 280 meters.
-/
theorem platform_length
  (train_speed_kmph : ℕ)
  (cross_platform_time_sec cross_man_time_sec : ℕ)
  (h1 : train_speed_kmph = 72)
  (h2 : cross_platform_time_sec = 32)
  (h3 : cross_man_time_sec = 18) :
  ∃ (L_platform : ℕ), L_platform = 280 :=
by
  sorry

end NUMINAMATH_GPT_platform_length_l1278_127814


namespace NUMINAMATH_GPT_solve_for_x_l1278_127807

theorem solve_for_x (x : ℝ) (h : (x^2 - 36) / 3 = (x^2 + 3 * x + 9) / 6) : x = 9 ∨ x = -9 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l1278_127807


namespace NUMINAMATH_GPT_same_color_difference_perfect_square_l1278_127861

theorem same_color_difference_perfect_square :
  (∃ (f : ℤ → ℕ) (a b : ℤ), f a = f b ∧ a ≠ b ∧ ∃ (k : ℤ), a - b = k * k) :=
sorry

end NUMINAMATH_GPT_same_color_difference_perfect_square_l1278_127861


namespace NUMINAMATH_GPT_negation_of_proposition_l1278_127865

theorem negation_of_proposition : 
  ¬ (∀ x : ℝ, x > 0 → x^2 ≤ 0) ↔ ∃ x : ℝ, x > 0 ∧ x^2 > 0 := by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1278_127865


namespace NUMINAMATH_GPT_eggs_per_group_l1278_127877

-- Conditions
def total_eggs : ℕ := 9
def total_groups : ℕ := 3

-- Theorem statement
theorem eggs_per_group : total_eggs / total_groups = 3 :=
sorry

end NUMINAMATH_GPT_eggs_per_group_l1278_127877


namespace NUMINAMATH_GPT_card_deck_initial_count_l1278_127813

theorem card_deck_initial_count 
  (r b : ℕ)
  (h1 : r / (r + b) = 1 / 4)
  (h2 : r / (r + (b + 6)) = 1 / 5) : 
  r + b = 24 :=
by
  sorry

end NUMINAMATH_GPT_card_deck_initial_count_l1278_127813


namespace NUMINAMATH_GPT_octahedron_volume_l1278_127845

theorem octahedron_volume (a : ℝ) (h1 : a > 0) :
  (∃ V : ℝ, V = (a^3 * Real.sqrt 2) / 3) :=
sorry

end NUMINAMATH_GPT_octahedron_volume_l1278_127845


namespace NUMINAMATH_GPT_actual_order_correct_l1278_127863

-- Define the actual order of the students.
def actual_order := ["E", "D", "A", "C", "B"]

-- Define the first person's prediction and conditions.
def first_person_prediction := ["A", "B", "C", "D", "E"]
def first_person_conditions (pos1 pos2 pos3 pos4 pos5 : String) : Prop :=
  (pos1 ≠ "A") ∧ (pos2 ≠ "B") ∧ (pos3 ≠ "C") ∧ (pos4 ≠ "D") ∧ (pos5 ≠ "E") ∧
  (pos1 ≠ "B") ∧ (pos2 ≠ "A") ∧ (pos2 ≠ "C") ∧ (pos3 ≠ "B") ∧ (pos3 ≠ "D") ∧
  (pos4 ≠ "C") ∧ (pos4 ≠ "E") ∧ (pos5 ≠ "D")

-- Define the second person's prediction and conditions.
def second_person_prediction := ["D", "A", "E", "C", "B"]
def second_person_conditions (pos1 pos2 pos3 pos4 pos5 : String) : Prop :=
  ((pos1 = "D") ∨ (pos2 = "D") ∨ (pos3 = "D") ∨ (pos4 = "D") ∨ (pos5 = "D")) ∧
  ((pos1 = "A") ∨ (pos2 = "A") ∨ (pos3 = "A") ∨ (pos4 = "A") ∨ (pos5 = "A")) ∧
  (pos1 ≠ "D" ∨ pos2 ≠ "A") ∧ (pos2 ≠ "A" ∨ pos3 ≠ "E") ∧ (pos3 ≠ "E" ∨ pos4 ≠ "C") ∧ (pos4 ≠ "C" ∨ pos5 ≠ "B")

-- The theorem to prove the actual order.
theorem actual_order_correct :
  ∃ (pos1 pos2 pos3 pos4 pos5 : String),
    first_person_conditions pos1 pos2 pos3 pos4 pos5 ∧
    second_person_conditions pos1 pos2 pos3 pos4 pos5 ∧
    [pos1, pos2, pos3, pos4, pos5] = actual_order :=
by sorry

end NUMINAMATH_GPT_actual_order_correct_l1278_127863


namespace NUMINAMATH_GPT_find_a_l1278_127890

variable (x y a : ℝ)

theorem find_a (h1 : (a * x + 8 * y) / (x - 2 * y) = 29) (h2 : x / (2 * y) = 3 / 2) : a = 7 :=
sorry

end NUMINAMATH_GPT_find_a_l1278_127890


namespace NUMINAMATH_GPT_simplify_expression_l1278_127824

theorem simplify_expression : 
  8 - (-3) + (-5) + (-7) = 3 + 8 - 7 - 5 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1278_127824


namespace NUMINAMATH_GPT_sum_ai_le_sum_bi_l1278_127849

open BigOperators

variable {α : Type*} [LinearOrderedField α]

theorem sum_ai_le_sum_bi {n : ℕ} {a b : Fin n → α}
  (h1 : ∀ i, 0 < a i)
  (h2 : ∀ i, 0 < b i)
  (h3 : ∑ i, (a i)^2 / b i ≤ ∑ i, b i) :
  ∑ i, a i ≤ ∑ i, b i :=
sorry

end NUMINAMATH_GPT_sum_ai_le_sum_bi_l1278_127849


namespace NUMINAMATH_GPT_find_a_minus_c_l1278_127882

section
variables (a b c : ℝ)
variables (h₁ : (a + b) / 2 = 110) (h₂ : (b + c) / 2 = 170)

theorem find_a_minus_c : a - c = -120 :=
by
  sorry
end

end NUMINAMATH_GPT_find_a_minus_c_l1278_127882


namespace NUMINAMATH_GPT_m_plus_n_eq_five_l1278_127893

theorem m_plus_n_eq_five (m n : ℝ) (h1 : m - 2 = 0) (h2 : 1 + n - 2 * m = 0) : m + n = 5 := 
  by 
  sorry

end NUMINAMATH_GPT_m_plus_n_eq_five_l1278_127893


namespace NUMINAMATH_GPT_prob1_prob2_prob3_l1278_127853

-- Problem 1
theorem prob1 (k : ℝ) (h₀ : k > 0) 
  (h₁ : ∀ x : ℝ, 2 < x ∧ x < 3 → (k * x^2 - 2 * x + 6 * k) < 0) :
  k = 2/5 := 
sorry

-- Problem 2
theorem prob2 (k : ℝ) (h₀ : k > 0) 
  (h₁ : ∀ x : ℝ, 2 < x ∧ x < 3 → (k * x^2 - 2 * x + 6 * k) < 0) :
  0 < k ∧ k ≤ 2/5 := 
sorry

-- Problem 3
theorem prob3 (k : ℝ) (h₀ : k > 0)
  (h₁ : ∀ x : ℝ, 2 < x ∧ x < 3 → (k * x^2 - 2 * x + 6 * k) < 0) :
  k ≥ 2/5 := 
sorry

end NUMINAMATH_GPT_prob1_prob2_prob3_l1278_127853


namespace NUMINAMATH_GPT_compare_a_b_c_l1278_127846

noncomputable def a : ℝ := (1 / 3)^(1 / 3)
noncomputable def b : ℝ := Real.log (1 / 2)
noncomputable def c : ℝ := Real.logb (1 / 3) (1 / 4)

theorem compare_a_b_c : b < a ∧ a < c := by
  sorry

end NUMINAMATH_GPT_compare_a_b_c_l1278_127846


namespace NUMINAMATH_GPT_height_difference_l1278_127838

variable {J L R : ℕ}

theorem height_difference
  (h1 : J = L + 15)
  (h2 : J = 152)
  (h3 : L + R = 295) :
  R - J = 6 :=
sorry

end NUMINAMATH_GPT_height_difference_l1278_127838


namespace NUMINAMATH_GPT_sum_of_tesseract_elements_l1278_127867

noncomputable def tesseract_edges : ℕ := 32
noncomputable def tesseract_vertices : ℕ := 16
noncomputable def tesseract_faces : ℕ := 24

theorem sum_of_tesseract_elements : tesseract_edges + tesseract_vertices + tesseract_faces = 72 := by
  -- proof here
  sorry

end NUMINAMATH_GPT_sum_of_tesseract_elements_l1278_127867


namespace NUMINAMATH_GPT_tan_angle_addition_l1278_127828

theorem tan_angle_addition (x : ℝ) (h : Real.tan x = 3) : Real.tan (x + Real.pi / 3) = - (6 + 5 * Real.sqrt 3) / 13 := by
  sorry

end NUMINAMATH_GPT_tan_angle_addition_l1278_127828


namespace NUMINAMATH_GPT_sum_of_parallelogram_sides_l1278_127848

-- Definitions of the given conditions.
def length_one_side : ℕ := 10
def length_other_side : ℕ := 7

-- Theorem stating the sum of the lengths of the four sides of the parallelogram.
theorem sum_of_parallelogram_sides : 
    (length_one_side + length_one_side + length_other_side + length_other_side) = 34 :=
by
    sorry

end NUMINAMATH_GPT_sum_of_parallelogram_sides_l1278_127848


namespace NUMINAMATH_GPT_divide_pile_l1278_127832

theorem divide_pile (pile : ℝ) (similar : ℝ → ℝ → Prop) :
  (∀ x y, similar x y ↔ x ≤ y * Real.sqrt 2 ∧ y ≤ x * Real.sqrt 2) →
  ¬∃ a b c, a + b + c = pile ∧ similar a b ∧ similar b c ∧ similar a c :=
by sorry

end NUMINAMATH_GPT_divide_pile_l1278_127832


namespace NUMINAMATH_GPT_part_a_first_player_wins_part_b_first_player_wins_l1278_127806

/-- Define the initial state of the game -/
structure GameState :=
(pile1 : Nat) (pile2 : Nat)

/-- Define the moves allowed in Part a) -/
inductive MoveA
| take_from_pile1 : MoveA
| take_from_pile2 : MoveA
| take_from_both  : MoveA

/-- Define the moves allowed in Part b) -/
inductive MoveB
| take_from_pile1 : MoveB
| take_from_pile2 : MoveB
| take_from_both  : MoveB
| transfer_to_pile2 : MoveB

/-- Define what it means for the first player to have a winning strategy in part a) -/
def first_player_wins_a (initial_state : GameState) : Prop := sorry

/-- Define what it means for the first player to have a winning strategy in part b) -/
def first_player_wins_b (initial_state : GameState) : Prop := sorry

/-- Theorem statement for part a) -/
theorem part_a_first_player_wins :
  first_player_wins_a ⟨7, 7⟩ :=
sorry

/-- Theorem statement for part b) -/
theorem part_b_first_player_wins :
  first_player_wins_b ⟨7, 7⟩ :=
sorry

end NUMINAMATH_GPT_part_a_first_player_wins_part_b_first_player_wins_l1278_127806


namespace NUMINAMATH_GPT_remainder_5n_minus_12_l1278_127818

theorem remainder_5n_minus_12 (n : ℤ) (hn : n % 9 = 4) : (5 * n - 12) % 9 = 8 := 
by sorry

end NUMINAMATH_GPT_remainder_5n_minus_12_l1278_127818


namespace NUMINAMATH_GPT_smallest_bottles_needed_l1278_127800

/-- Christine needs at least 60 fluid ounces of milk, the store sells milk in 250 milliliter bottles,
and there are 32 fluid ounces in 1 liter. The smallest number of bottles Christine should purchase
is 8. -/
theorem smallest_bottles_needed
  (fl_oz_needed : ℕ := 60)
  (ml_per_bottle : ℕ := 250)
  (fl_oz_per_liter : ℕ := 32) :
  let liters_needed := fl_oz_needed / fl_oz_per_liter
  let ml_needed := liters_needed * 1000
  let bottles := (ml_needed + ml_per_bottle - 1) / ml_per_bottle
  bottles = 8 :=
by
  sorry

end NUMINAMATH_GPT_smallest_bottles_needed_l1278_127800


namespace NUMINAMATH_GPT_D_180_equals_43_l1278_127866

-- Define D(n) as the number of ways to express the positive integer n
-- as a product of integers strictly greater than 1, where the order of factors matters.
def D (n : Nat) : Nat := sorry  -- The actual implementation is not provided, as per instructions.

theorem D_180_equals_43 : D 180 = 43 :=
by
  sorry  -- The proof is omitted as the task specifies.

end NUMINAMATH_GPT_D_180_equals_43_l1278_127866


namespace NUMINAMATH_GPT_simplify_frac_48_72_l1278_127899

theorem simplify_frac_48_72 : (48 / 72 : ℚ) = 2 / 3 :=
by
  -- In Lean, we prove the equality of the simplified fractions.
  sorry

end NUMINAMATH_GPT_simplify_frac_48_72_l1278_127899


namespace NUMINAMATH_GPT_average_15_19_x_eq_20_l1278_127802

theorem average_15_19_x_eq_20 (x : ℝ) : (15 + 19 + x) / 3 = 20 → x = 26 :=
by
  sorry

end NUMINAMATH_GPT_average_15_19_x_eq_20_l1278_127802


namespace NUMINAMATH_GPT_rotate_right_triangle_along_right_angle_produces_cone_l1278_127827

-- Define a right triangle and the conditions for its rotation
structure RightTriangle (α β γ : ℝ) :=
  (zero_angle : α = 0)
  (ninety_angle_1 : β = 90)
  (ninety_angle_2 : γ = 90)
  (sum_180 : α + β + γ = 180)

-- Define the theorem for the resulting shape when rotating the right triangle
theorem rotate_right_triangle_along_right_angle_produces_cone
  (T : RightTriangle α β γ) (line_of_rotation_contains_right_angle : α = 90 ∨ β = 90 ∨ γ = 90) :
  ∃ shape, shape = "cone" :=
sorry

end NUMINAMATH_GPT_rotate_right_triangle_along_right_angle_produces_cone_l1278_127827


namespace NUMINAMATH_GPT_nine_by_nine_chessboard_dark_light_excess_l1278_127856

theorem nine_by_nine_chessboard_dark_light_excess :
  let board_size := 9
  let odd_row_dark := 5
  let odd_row_light := 4
  let even_row_dark := 4
  let even_row_light := 5
  let num_odd_rows := (board_size + 1) / 2
  let num_even_rows := board_size / 2
  let total_dark_squares := (odd_row_dark * num_odd_rows) + (even_row_dark * num_even_rows)
  let total_light_squares := (odd_row_light * num_odd_rows) + (even_row_light * num_even_rows)
  total_dark_squares - total_light_squares = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_nine_by_nine_chessboard_dark_light_excess_l1278_127856


namespace NUMINAMATH_GPT_prime_factorization_2006_expr_l1278_127896

theorem prime_factorization_2006_expr :
  let a := 2006
  let b := 669
  let c := 1593
  (a^2 * (b + c) - b^2 * (c + a) + c^2 * (a - b)) =
  2 * 3 * 7 * 13 * 29 * 59 * 61 * 191 :=
by
  let a := 2006
  let b := 669
  let c := 1593
  have h1 : 2262 = b + c := by norm_num
  have h2 : 3599 = c + a := by norm_num
  have h3 : 1337 = a - b := by norm_num
  sorry

end NUMINAMATH_GPT_prime_factorization_2006_expr_l1278_127896


namespace NUMINAMATH_GPT_question1_question2_l1278_127864

noncomputable def f1 (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x^2
noncomputable def f2 (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x^2 - 2*x

theorem question1 (a : ℝ) : 
  (∀ x : ℝ, f1 a x = 0 → ∀ y : ℝ, f1 a y = 0 → x = y) ↔ (a = 0 ∨ a < -4 / Real.exp 2) :=
sorry -- Proof of theorem 1

theorem question2 (a m n x0 : ℝ) (h : a ≠ 0) :
  (f2 a x0 = f2 a ((x0 + m) / 2) * (x0 - m) + n ∧ x0 ≠ m) → False :=
sorry -- Proof of theorem 2

end NUMINAMATH_GPT_question1_question2_l1278_127864


namespace NUMINAMATH_GPT_am_minus_hm_lt_bound_l1278_127836

theorem am_minus_hm_lt_bound (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x < y) :
  (x - y)^2 / (2 * (x + y)) < (x - y)^2 / (8 * x) := 
by
  sorry

end NUMINAMATH_GPT_am_minus_hm_lt_bound_l1278_127836


namespace NUMINAMATH_GPT_vlad_taller_than_sister_l1278_127842

theorem vlad_taller_than_sister : 
  ∀ (vlad_height sister_height : ℝ), 
  vlad_height = 190.5 → sister_height = 86.36 → vlad_height - sister_height = 104.14 :=
by
  intros vlad_height sister_height vlad_height_eq sister_height_eq
  rw [vlad_height_eq, sister_height_eq]
  sorry

end NUMINAMATH_GPT_vlad_taller_than_sister_l1278_127842


namespace NUMINAMATH_GPT_factorize_expr_solve_inequality_solve_equation_simplify_expr_l1278_127855

-- Problem 1
theorem factorize_expr (x y m n : ℝ) : x^2 * (3 * m - 2 * n) + y^2 * (2 * n - 3 * m) = (3 * m - 2 * n) * (x + y) * (x - y) := 
sorry

-- Problem 2
theorem solve_inequality (x : ℝ) : 
  (∃ x, (x - 3) / 2 + 3 > x + 1 ∧ 1 - 3 * (x - 1) < 8 - x) → -2 < x ∧ x < 1 :=
sorry

-- Problem 3
theorem solve_equation (x : ℝ) : 
  (∃ x, (3 - x) / (x - 4) + 1 / (4 - x) = 1) → x = 3 :=
sorry

-- Problem 4
theorem simplify_expr (a : ℝ) (h : a = 3) : 
  (2 / (a + 1) + (a + 2) / (a^2 - 1)) / (a / (a - 1)) = 3 / 4 :=
sorry

end NUMINAMATH_GPT_factorize_expr_solve_inequality_solve_equation_simplify_expr_l1278_127855


namespace NUMINAMATH_GPT_candy_vs_chocolate_l1278_127801

theorem candy_vs_chocolate
  (candy1 candy2 chocolate : ℕ)
  (h1 : candy1 = 38)
  (h2 : candy2 = 36)
  (h3 : chocolate = 16) :
  (candy1 + candy2) - chocolate = 58 :=
by
  sorry

end NUMINAMATH_GPT_candy_vs_chocolate_l1278_127801


namespace NUMINAMATH_GPT_geometric_sequence_a6_l1278_127850

theorem geometric_sequence_a6 (a : ℕ → ℝ) (r : ℝ)
  (h₁ : a 4 = 7)
  (h₂ : a 8 = 63)
  (h_geom : ∀ n, a n = a 1 * r^(n - 1)) :
  a 6 = 21 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a6_l1278_127850


namespace NUMINAMATH_GPT_fg_sum_at_2_l1278_127812

noncomputable def f (x : ℚ) : ℚ := (5 * x^3 + 4 * x^2 - 2 * x + 3) / (x^3 - 2 * x^2 + 3 * x + 1)
noncomputable def g (x : ℚ) : ℚ := x^2 - 2

theorem fg_sum_at_2 : f (g 2) + g (f 2) = 468 / 7 := by
  sorry

end NUMINAMATH_GPT_fg_sum_at_2_l1278_127812


namespace NUMINAMATH_GPT_sum_of_original_numbers_l1278_127820

theorem sum_of_original_numbers :
  ∃ a b : ℚ, a = b + 12 ∧ a^2 + b^2 = 169 / 2 ∧ (a^2)^2 - (b^2)^2 = 5070 ∧ a + b = 5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_original_numbers_l1278_127820


namespace NUMINAMATH_GPT_solve_for_x_l1278_127874
-- Lean 4 Statement

theorem solve_for_x (x : ℝ) (h : 2^(3 * x) = Real.sqrt 32) : x = 5 / 6 := 
sorry

end NUMINAMATH_GPT_solve_for_x_l1278_127874


namespace NUMINAMATH_GPT_geometric_series_common_ratio_l1278_127875

theorem geometric_series_common_ratio (a S r : ℝ)
  (h1 : a = 172)
  (h2 : S = 400)
  (h3 : S = a / (1 - r)) :
  r = 57 / 100 := 
sorry

end NUMINAMATH_GPT_geometric_series_common_ratio_l1278_127875


namespace NUMINAMATH_GPT_distances_inequality_l1278_127829

theorem distances_inequality (x y : ℝ) :
  Real.sqrt ((x + 4)^2 + (y + 2)^2) + 
  Real.sqrt ((x - 5)^2 + (y + 4)^2) ≤ 
  Real.sqrt ((x - 2)^2 + (y - 6)^2) + 
  Real.sqrt ((x - 5)^2 + (y - 6)^2) + 20 :=
  sorry

end NUMINAMATH_GPT_distances_inequality_l1278_127829


namespace NUMINAMATH_GPT_find_m_value_l1278_127884

theorem find_m_value (x y m : ℤ) (h₁ : x = 2) (h₂ : y = -3) (h₃ : 5 * x + m * y + 2 = 0) : m = 4 := 
by 
  sorry

end NUMINAMATH_GPT_find_m_value_l1278_127884


namespace NUMINAMATH_GPT_proof_A_proof_C_l1278_127881

theorem proof_A (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a * b ≤ ( (a + b) / 2) ^ 2 := 
sorry

theorem proof_C (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2) : 
  ∃ y, y = x * (4 - x^2).sqrt ∧ y ≤ 2 := 
sorry

end NUMINAMATH_GPT_proof_A_proof_C_l1278_127881


namespace NUMINAMATH_GPT_sandy_initial_payment_l1278_127805

theorem sandy_initial_payment (P : ℝ) (repairs cost: ℝ) (selling_price gain: ℝ) 
  (hc : repairs = 300)
  (hs : selling_price = 1260) 
  (hg : gain = 5)
  (h : selling_price = (P + repairs) * (1 + gain / 100)) : 
  P = 900 :=
sorry

end NUMINAMATH_GPT_sandy_initial_payment_l1278_127805


namespace NUMINAMATH_GPT_bales_in_barn_l1278_127869

theorem bales_in_barn (stacked today total original : ℕ) (h1 : stacked = 67) (h2 : total = 89) (h3 : total = stacked + original) : original = 22 :=
by
  sorry

end NUMINAMATH_GPT_bales_in_barn_l1278_127869


namespace NUMINAMATH_GPT_intersection_A_B_l1278_127810

def A : Set ℝ := { x : ℝ | -1 < x ∧ x < 3 }
def B : Set ℝ := { x : ℝ | x < 2 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | -1 < x ∧ x < 2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1278_127810


namespace NUMINAMATH_GPT_total_points_l1278_127841

theorem total_points (gwen_points_per_4 : ℕ) (lisa_points_per_5 : ℕ) (jack_points_per_7 : ℕ) 
                     (gwen_recycled : ℕ) (lisa_recycled : ℕ) (jack_recycled : ℕ)
                     (gwen_ratio : gwen_points_per_4 = 2) (lisa_ratio : lisa_points_per_5 = 3) 
                     (jack_ratio : jack_points_per_7 = 1) (gwen_pounds : gwen_recycled = 12) 
                     (lisa_pounds : lisa_recycled = 25) (jack_pounds : jack_recycled = 21) 
                     : gwen_points_per_4 * (gwen_recycled / 4) + 
                       lisa_points_per_5 * (lisa_recycled / 5) + 
                       jack_points_per_7 * (jack_recycled / 7) = 24 := by
  sorry

end NUMINAMATH_GPT_total_points_l1278_127841


namespace NUMINAMATH_GPT_jo_integer_max_l1278_127851
noncomputable def jo_integer : Nat :=
  let n := 166
  n

theorem jo_integer_max (n : Nat) (h1 : n < 200) (h2 : ∃ k : Nat, n + 2 = 9 * k) (h3 : ∃ l : Nat, n + 4 = 10 * l) : n ≤ jo_integer := 
by
  unfold jo_integer
  sorry

end NUMINAMATH_GPT_jo_integer_max_l1278_127851


namespace NUMINAMATH_GPT_max_value_seq_l1278_127826

theorem max_value_seq : 
  ∃ a : ℕ → ℝ, 
    a 1 = 1 ∧ 
    a 2 = 4 ∧ 
    (∀ n ≥ 2, 2 * a n = (n - 1) / n * a (n - 1) + (n + 1) / n * a (n + 1)) ∧ 
    ∀ n : ℕ, n > 0 → 
      ∃ m : ℕ, m > 0 ∧ 
        ∀ k : ℕ, k > 0 → (a k) / k ≤ 2 ∧ (a 2) / 2 = 2 :=
sorry

end NUMINAMATH_GPT_max_value_seq_l1278_127826


namespace NUMINAMATH_GPT_travis_apples_l1278_127879

theorem travis_apples
  (price_per_box : ℕ)
  (num_apples_per_box : ℕ)
  (total_money : ℕ)
  (total_boxes : ℕ)
  (total_apples : ℕ)
  (h1 : price_per_box = 35)
  (h2 : num_apples_per_box = 50)
  (h3 : total_money = 7000)
  (h4 : total_boxes = total_money / price_per_box)
  (h5 : total_apples = total_boxes * num_apples_per_box) :
  total_apples = 10000 :=
sorry

end NUMINAMATH_GPT_travis_apples_l1278_127879


namespace NUMINAMATH_GPT_sum_primes_between_20_and_40_l1278_127840

open Nat

def primesBetween20And40 : List Nat := [23, 29, 31, 37]

theorem sum_primes_between_20_and_40 :
  (primesBetween20And40.sum = 120) :=
by
  sorry

end NUMINAMATH_GPT_sum_primes_between_20_and_40_l1278_127840


namespace NUMINAMATH_GPT_ending_number_condition_l1278_127803

theorem ending_number_condition (h : ∃ k : ℕ, k < 21 ∧ 100 < 19 * k) : ∃ n, 21.05263157894737 * 19 = n → n = 399 :=
by
  sorry  -- this is where the proof would go

end NUMINAMATH_GPT_ending_number_condition_l1278_127803


namespace NUMINAMATH_GPT_cos_identity_l1278_127837

theorem cos_identity (α : ℝ) (h : Real.cos (π / 4 - α) = -1 / 3) :
  Real.cos (3 * π / 4 + α) = 1 / 3 :=
sorry

end NUMINAMATH_GPT_cos_identity_l1278_127837


namespace NUMINAMATH_GPT_problem_l1278_127819

open Real

theorem problem (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  (2 + a) * (2 + b) ≥ c * d := 
sorry

end NUMINAMATH_GPT_problem_l1278_127819


namespace NUMINAMATH_GPT_bounded_sequence_exists_l1278_127835

noncomputable def positive_sequence := ℕ → ℝ

variables {a : positive_sequence}

axiom positive_sequence_pos (n : ℕ) : 0 < a n

axiom sequence_condition (k n m l : ℕ) (h : k + n = m + l) : 
  (a k + a n) / (1 + a k * a n) = (a m + a l) / (1 + a m * a l)

theorem bounded_sequence_exists 
  (a : positive_sequence) 
  (h_pos : ∀ n, 0 < a n)
  (h_cond : ∀ (k n m l : ℕ), k + n = m + l → 
              (a k + a n) / (1 + a k * a n) = (a m + a l) / (1 + a m * a l)) :
  ∃ (b c : ℝ), (0 < b) ∧ (0 < c) ∧ (∀ n, b ≤ a n ∧ a n ≤ c) :=
sorry

end NUMINAMATH_GPT_bounded_sequence_exists_l1278_127835


namespace NUMINAMATH_GPT_distance_borya_vasya_l1278_127889

-- Definitions of the houses and distances on the road
def distance_andrey_gena : ℕ := 2450
def race_length : ℕ := 1000

-- Variables to represent the distances
variables (y b : ℕ)

-- Conditions
def start_position := y
def finish_position := b / 2 + 1225

axiom distance_eq : distance_andrey_gena = 2 * y
axiom race_distance_eq : finish_position - start_position = race_length

-- Proving the distance between Borya's and Vasya's houses
theorem distance_borya_vasya :
  ∃ (d : ℕ), d = 450 :=
by
  sorry

end NUMINAMATH_GPT_distance_borya_vasya_l1278_127889


namespace NUMINAMATH_GPT_alice_paper_cranes_l1278_127808

theorem alice_paper_cranes : 
  ∀ (total : ℕ) (half : ℕ) (one_fifth : ℕ) (thirty_percent : ℕ),
    total = 1000 →
    half = total / 2 →
    one_fifth = (total - half) / 5 →
    thirty_percent = ((total - half) - one_fifth) * 3 / 10 →
    total - (half + one_fifth + thirty_percent) = 280 :=
by
  intros total half one_fifth thirty_percent h_total h_half h_one_fifth h_thirty_percent
  sorry

end NUMINAMATH_GPT_alice_paper_cranes_l1278_127808


namespace NUMINAMATH_GPT_inequality_solution_l1278_127860

theorem inequality_solution (x : ℝ) : (2 * x - 3 < x + 1) -> (x < 4) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_inequality_solution_l1278_127860
