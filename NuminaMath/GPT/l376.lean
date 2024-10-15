import Mathlib

namespace NUMINAMATH_GPT_triangle_eq_medians_incircle_l376_37677

-- Define a triangle and the properties of medians and incircle
structure Triangle (α : Type) [Nonempty α] :=
(A B C : α)

def is_equilateral {α : Type} [Nonempty α] (T : Triangle α) : Prop :=
  ∃ (d : α → α → ℝ), d T.A T.B = d T.B T.C ∧ d T.B T.C = d T.C T.A

def medians_segments_equal {α : Type} [Nonempty α] (T : Triangle α) (incr_len : (α → α → ℝ)) : Prop :=
  ∀ (MA MB MC : α), incr_len MA MB = incr_len MB MC ∧ incr_len MB MC = incr_len MC MA

-- The main theorem statement
theorem triangle_eq_medians_incircle {α : Type} [Nonempty α] 
  (T : Triangle α) (incr_len : α → α → ℝ) 
  (h : medians_segments_equal T incr_len) : is_equilateral T :=
sorry

end NUMINAMATH_GPT_triangle_eq_medians_incircle_l376_37677


namespace NUMINAMATH_GPT_rate_percent_simple_interest_l376_37690

theorem rate_percent_simple_interest
  (SI P : ℚ) (T : ℕ) (R : ℚ) : SI = 160 → P = 800 → T = 4 → (P * R * T / 100 = SI) → R = 5 :=
  by
  intros hSI hP hT hFormula
  -- Assertion that R = 5 is correct based on the given conditions and formula
  sorry

end NUMINAMATH_GPT_rate_percent_simple_interest_l376_37690


namespace NUMINAMATH_GPT_shaded_area_is_14_percent_l376_37623

def side_length : ℕ := 20
def rectangle_width : ℕ := 35
def rectangle_height : ℕ := side_length
def rectangle_area : ℕ := rectangle_width * rectangle_height
def overlap_length : ℕ := 2 * side_length - rectangle_width
def shaded_area : ℕ := overlap_length * side_length
def shaded_percentage : ℚ := (shaded_area : ℚ) / rectangle_area * 100

theorem shaded_area_is_14_percent : shaded_percentage = 14 := by
  sorry

end NUMINAMATH_GPT_shaded_area_is_14_percent_l376_37623


namespace NUMINAMATH_GPT_D_is_necessary_but_not_sufficient_condition_for_A_l376_37678

variable (A B C D : Prop)

-- Conditions
axiom A_implies_B : A → B
axiom not_B_implies_A : ¬ (B → A)
axiom B_iff_C : B ↔ C
axiom C_implies_D : C → D
axiom not_D_implies_C : ¬ (D → C)

theorem D_is_necessary_but_not_sufficient_condition_for_A : (A → D) ∧ ¬ (D → A) :=
by sorry

end NUMINAMATH_GPT_D_is_necessary_but_not_sufficient_condition_for_A_l376_37678


namespace NUMINAMATH_GPT_sum_of_divisors_45_l376_37668

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (fun i => n % i = 0) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_45 : sum_of_divisors 45 = 78 := 
  sorry

end NUMINAMATH_GPT_sum_of_divisors_45_l376_37668


namespace NUMINAMATH_GPT_evaluate_ratio_is_negative_two_l376_37686

noncomputable def evaluate_ratio (a b : ℂ) (h : a ≠ 0 ∧ b ≠ 0 ∧ a^4 + a^2 * b^2 + b^4 = 0) : ℂ :=
  (a^15 + b^15) / (a + b)^15

theorem evaluate_ratio_is_negative_two (a b : ℂ) (h : a ≠ 0 ∧ b ≠ 0 ∧ a^4 + a^2 * b^2 + b^4 = 0) : 
  evaluate_ratio a b h = -2 := 
sorry

end NUMINAMATH_GPT_evaluate_ratio_is_negative_two_l376_37686


namespace NUMINAMATH_GPT_triangle_no_two_obtuse_angles_l376_37663

theorem triangle_no_two_obtuse_angles (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A > 90) (h3 : B > 90) (h4 : C > 0) : false :=
by
  sorry

end NUMINAMATH_GPT_triangle_no_two_obtuse_angles_l376_37663


namespace NUMINAMATH_GPT_parametric_to_cartesian_l376_37621

theorem parametric_to_cartesian (t : ℝ) (x y : ℝ) (h1 : x = 5 + 3 * t) (h2 : y = 10 - 4 * t) : 4 * x + 3 * y = 50 :=
by sorry

end NUMINAMATH_GPT_parametric_to_cartesian_l376_37621


namespace NUMINAMATH_GPT_inequality_solution_l376_37699

open Real

theorem inequality_solution (a x : ℝ) :
  (a = 0 ∧ x > 2 ∧ a * x^2 - (2 * a + 2) * x + 4 > 0) ∨
  (a = 1 ∧ ∀ x, ¬ (a * x^2 - (2 * a + 2) * x + 4 > 0)) ∨
  (a < 0 ∧ (x < 2/a ∨ x > 2) ∧ a * x^2 - (2 * a + 2) * x + 4 > 0) ∨
  (0 < a ∧ a < 1 ∧ 2 < x ∧ x < 2/a ∧ a * x^2 - (2 * a + 2) * x + 4 > 0) ∨
  (a > 1 ∧ 2/a < x ∧ x < 2 ∧ a * x^2 - (2 * a + 2) * x + 4 > 0) := 
sorry

end NUMINAMATH_GPT_inequality_solution_l376_37699


namespace NUMINAMATH_GPT_min_ratio_l376_37676

theorem min_ratio (x y : ℕ) 
  (hx : 10 ≤ x ∧ x ≤ 99)
  (hy : 10 ≤ y ∧ y ≤ 99)
  (mean : (x + y) = 110) :
  x / y = 1 / 9 :=
  sorry

end NUMINAMATH_GPT_min_ratio_l376_37676


namespace NUMINAMATH_GPT_ratio_of_p_q_l376_37689

theorem ratio_of_p_q (b : ℝ) (p q : ℝ) (h1 : p = -b / 8) (h2 : q = -b / 12) : p / q = 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_p_q_l376_37689


namespace NUMINAMATH_GPT_simplify_expression_l376_37622

variable (a b : Real)

theorem simplify_expression (a b : Real) : 
    3 * b * (3 * b ^ 2 + 2 * b) - b ^ 2 + 2 * a * (2 * a ^ 2 - 3 * a) - 4 * a * b = 
    9 * b ^ 3 + 5 * b ^ 2 + 4 * a ^ 3 - 6 * a ^ 2 - 4 * a * b := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l376_37622


namespace NUMINAMATH_GPT_max_price_of_product_l376_37638

theorem max_price_of_product (x : ℝ) 
  (cond1 : (x - 10) * 0.1 = (x - 20) * 0.2) : 
  x = 30 := 
by 
  sorry

end NUMINAMATH_GPT_max_price_of_product_l376_37638


namespace NUMINAMATH_GPT_cone_base_radius_and_slant_height_l376_37633

noncomputable def sector_angle := 300
noncomputable def sector_radius := 10
noncomputable def arc_length := (sector_angle / 360) * 2 * Real.pi * sector_radius

theorem cone_base_radius_and_slant_height :
  ∃ (r l : ℝ), arc_length = 2 * Real.pi * r ∧ l = sector_radius ∧ r = 8 ∧ l = 10 :=
by 
  sorry

end NUMINAMATH_GPT_cone_base_radius_and_slant_height_l376_37633


namespace NUMINAMATH_GPT_multiples_of_9_ending_in_5_l376_37635

theorem multiples_of_9_ending_in_5 (n : ℕ) :
  (∃ k : ℕ, n = 9 * k ∧ 0 < n ∧ n < 600 ∧ n % 10 = 5) → 
  ∃ l, l = 7 := 
by
sorry

end NUMINAMATH_GPT_multiples_of_9_ending_in_5_l376_37635


namespace NUMINAMATH_GPT_inconsistent_coordinates_l376_37682

theorem inconsistent_coordinates
  (m n : ℝ) 
  (h1 : m - (5/2)*n + 1 = 0) 
  (h2 : (m + 1/2) - (5/2)*(n + 1) + 1 = 0) :
  false :=
by
  sorry

end NUMINAMATH_GPT_inconsistent_coordinates_l376_37682


namespace NUMINAMATH_GPT_tan_addition_sin_cos_expression_l376_37687

noncomputable def alpha : ℝ := sorry -- this is where alpha would be defined

axiom tan_alpha_eq_two : Real.tan alpha = 2

theorem tan_addition (alpha : ℝ) (h : Real.tan alpha = 2) : (Real.tan (alpha + Real.pi / 4) = -3) :=
by sorry

theorem sin_cos_expression (alpha : ℝ) (h : Real.tan alpha = 2) : 
  (Real.sin (2 * alpha) / (Real.sin (alpha) ^ 2 - Real.cos (2 * alpha) + 1) = 1 / 3) :=
by sorry

end NUMINAMATH_GPT_tan_addition_sin_cos_expression_l376_37687


namespace NUMINAMATH_GPT_hose_removal_rate_l376_37653

def pool_volume (length width depth : ℕ) : ℕ :=
  length * width * depth

def draining_rate (volume time : ℕ) : ℕ :=
  volume / time

theorem hose_removal_rate :
  let length := 150
  let width := 80
  let depth := 10
  let total_volume := pool_volume length width depth
  total_volume = 1200000 ∧
  let time := 2000
  draining_rate total_volume time = 600 :=
by
  sorry

end NUMINAMATH_GPT_hose_removal_rate_l376_37653


namespace NUMINAMATH_GPT_sum_odd_is_13_over_27_l376_37629

-- Define the probability for rolling an odd and an even number
def prob_odd := 1 / 3
def prob_even := 2 / 3

-- Define the probability that the sum of three die rolls is odd
def prob_sum_odd : ℚ :=
  3 * prob_odd * prob_even^2 + prob_odd^3

-- Statement asserting the goal to be proved
theorem sum_odd_is_13_over_27 :
  prob_sum_odd = 13 / 27 :=
by
  sorry

end NUMINAMATH_GPT_sum_odd_is_13_over_27_l376_37629


namespace NUMINAMATH_GPT_find_second_sum_l376_37671

theorem find_second_sum (S : ℝ) (x : ℝ) (h : S = 2704 ∧ 24 * x / 100 = 15 * (S - x) / 100) : (S - x) = 1664 := 
  sorry

end NUMINAMATH_GPT_find_second_sum_l376_37671


namespace NUMINAMATH_GPT_percent_palindromes_containing_7_l376_37605

theorem percent_palindromes_containing_7 : 
  let num_palindromes := 90
  let num_palindrome_with_7 := 19
  (num_palindrome_with_7 / num_palindromes * 100) = 21.11 := 
by
  sorry

end NUMINAMATH_GPT_percent_palindromes_containing_7_l376_37605


namespace NUMINAMATH_GPT_soccer_team_games_played_l376_37641

theorem soccer_team_games_played (t : ℝ) (h1 : 0.40 * t = 63.2) : t = 158 :=
sorry

end NUMINAMATH_GPT_soccer_team_games_played_l376_37641


namespace NUMINAMATH_GPT_wire_weight_l376_37695

theorem wire_weight (w : ℕ → ℕ) (h_proportional : ∀ (x y : ℕ), w (x + y) = w x + w y) : 
  (w 25 = 5) → w 75 = 15 :=
by
  intro h1
  sorry

end NUMINAMATH_GPT_wire_weight_l376_37695


namespace NUMINAMATH_GPT_find_c_l376_37644

noncomputable def y (x c : ℝ) : ℝ := x^3 - 3*x + c

theorem find_c (c : ℝ) (h : ∃ a b : ℝ, a ≠ b ∧ y a c = 0 ∧ y b c = 0) :
  c = -2 ∨ c = 2 :=
by sorry

end NUMINAMATH_GPT_find_c_l376_37644


namespace NUMINAMATH_GPT_minimum_price_to_cover_costs_l376_37692

variable (P : ℝ)

-- Conditions
def prod_cost_A := 80
def ship_cost_A := 2
def prod_cost_B := 60
def ship_cost_B := 3
def fixed_costs := 16200
def units_A := 200
def units_B := 300

-- Cost calculations
def total_cost_A := units_A * prod_cost_A + units_A * ship_cost_A
def total_cost_B := units_B * prod_cost_B + units_B * ship_cost_B
def total_costs := total_cost_A + total_cost_B + fixed_costs

-- Revenue requirement
def revenue (P_A P_B : ℝ) := units_A * P_A + units_B * P_B

theorem minimum_price_to_cover_costs :
  (units_A + units_B) * P ≥ total_costs ↔ P ≥ 103 :=
sorry

end NUMINAMATH_GPT_minimum_price_to_cover_costs_l376_37692


namespace NUMINAMATH_GPT_correct_quadratic_graph_l376_37603

theorem correct_quadratic_graph (a b c : ℝ) (ha : a > 0) (hb : b < 0) (hc : c < 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (-b / (2 * a) > 0) ∧ (c < 0) :=
by
  sorry

end NUMINAMATH_GPT_correct_quadratic_graph_l376_37603


namespace NUMINAMATH_GPT_Sydney_initial_rocks_l376_37602

variable (S₀ : ℕ)

def Conner_initial : ℕ := 723
def Sydney_collects_day1 : ℕ := 4
def Conner_collects_day1 : ℕ := 8 * Sydney_collects_day1
def Sydney_collects_day2 : ℕ := 0
def Conner_collects_day2 : ℕ := 123
def Sydney_collects_day3 : ℕ := 2 * Conner_collects_day1
def Conner_collects_day3 : ℕ := 27

def Total_Sydney_collects : ℕ := Sydney_collects_day1 + Sydney_collects_day2 + Sydney_collects_day3
def Total_Conner_collects : ℕ := Conner_collects_day1 + Conner_collects_day2 + Conner_collects_day3

def Total_Sydney_rocks : ℕ := S₀ + Total_Sydney_collects
def Total_Conner_rocks : ℕ := Conner_initial + Total_Conner_collects

theorem Sydney_initial_rocks :
  Total_Conner_rocks = Total_Sydney_rocks → S₀ = 837 :=
by
  sorry

end NUMINAMATH_GPT_Sydney_initial_rocks_l376_37602


namespace NUMINAMATH_GPT_problem_statement_l376_37661

theorem problem_statement :
  (∃ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ),
    (∀ x : ℝ, 1 + x^5 = a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 + 
              a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + a_5 * (x - 1)^5) ∧
    (a_0 = 2) ∧
    (a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 33)) →
  (∃ a_1 a_2 a_3 a_4 a_5 : ℝ, a_1 + a_2 + a_3 + a_4 + a_5 = 31) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l376_37661


namespace NUMINAMATH_GPT_correct_average_l376_37645

theorem correct_average
  (incorrect_avg : ℝ)
  (incorrect_num correct_num : ℝ)
  (n : ℕ)
  (h1 : incorrect_avg = 16)
  (h2 : incorrect_num = 26)
  (h3 : correct_num = 46)
  (h4 : n = 10) :
  (incorrect_avg * n - incorrect_num + correct_num) / n = 18 :=
sorry

end NUMINAMATH_GPT_correct_average_l376_37645


namespace NUMINAMATH_GPT_distance_between_parallel_lines_l376_37697

theorem distance_between_parallel_lines (A B c1 c2 : Real) (hA : A = 2) (hB : B = 3) 
(hc1 : c1 = -3) (hc2 : c2 = 2) : 
    (abs (c1 - c2) / Real.sqrt (A^2 + B^2)) = (5 * Real.sqrt 13 / 13) := by
  sorry

end NUMINAMATH_GPT_distance_between_parallel_lines_l376_37697


namespace NUMINAMATH_GPT_solve_for_b_l376_37669

theorem solve_for_b 
  (b : ℝ)
  (h : (25 * b^2) - 84 = 0) :
  b = (2 * Real.sqrt 21) / 5 ∨ b = -(2 * Real.sqrt 21) / 5 :=
by sorry

end NUMINAMATH_GPT_solve_for_b_l376_37669


namespace NUMINAMATH_GPT_expression_eq_neg_one_l376_37684

theorem expression_eq_neg_one (a b y : ℝ) (h1 : a ≠ 0) (h2 : b ≠ a) (h3 : y ≠ a) (h4 : y ≠ -a) :
  ( ( (a + b) / (a + y) + y / (a - y) ) / ( (y + b) / (a + y) - a / (a - y) ) = -1 ) ↔ ( y = a - b ) := 
sorry

end NUMINAMATH_GPT_expression_eq_neg_one_l376_37684


namespace NUMINAMATH_GPT_complex_modulus_product_l376_37650

noncomputable def z1 : ℂ := 4 - 3 * Complex.I
noncomputable def z2 : ℂ := 4 + 3 * Complex.I

theorem complex_modulus_product : Complex.abs z1 * Complex.abs z2 = 25 := by 
  sorry

end NUMINAMATH_GPT_complex_modulus_product_l376_37650


namespace NUMINAMATH_GPT_sequence_nth_term_l376_37600

theorem sequence_nth_term (a : ℕ → ℚ) (h : a 1 = 3 / 2 ∧ a 2 = 1 ∧ a 3 = 5 / 8 ∧ a 4 = 3 / 8) :
  ∀ n : ℕ, a n = (n^2 - 11*n + 34) / 16 := by
  sorry

end NUMINAMATH_GPT_sequence_nth_term_l376_37600


namespace NUMINAMATH_GPT_demand_decrease_l376_37614

theorem demand_decrease (original_price_increase effective_price_increase demand_decrease : ℝ)
  (h1 : original_price_increase = 0.2)
  (h2 : effective_price_increase = original_price_increase / 2)
  (h3 : new_price = original_price * (1 + effective_price_increase))
  (h4 : 1 / new_price = original_demand)
  : demand_decrease = 0.0909 := sorry

end NUMINAMATH_GPT_demand_decrease_l376_37614


namespace NUMINAMATH_GPT_count_valid_triangles_l376_37628

def is_triangle (a b c : ℕ) : Prop := 
  a + b > c ∧ b + c > a ∧ c + a > b

def valid_triangle (a b c : ℕ) : Prop :=
  is_triangle a b c ∧ a > 0 ∧ b > 0 ∧ c > 0

theorem count_valid_triangles : 
  (∃ n : ℕ, n = 14 ∧ 
  ∃ (a b c : ℕ), valid_triangle a b c ∧ 
  ((b = 5 ∧ c > 5) ∨ (c = 5 ∧ b > 5)) ∧ 
  (a > 0 ∧ b > 0 ∧ c > 0)) :=
by { sorry }

end NUMINAMATH_GPT_count_valid_triangles_l376_37628


namespace NUMINAMATH_GPT_abs_sum_neq_3_nor_1_l376_37617

theorem abs_sum_neq_3_nor_1 (a b : ℤ) (h₁ : |a| = 3) (h₂ : |b| = 1) : (|a + b| ≠ 3) ∧ (|a + b| ≠ 1) := sorry

end NUMINAMATH_GPT_abs_sum_neq_3_nor_1_l376_37617


namespace NUMINAMATH_GPT_total_distance_traveled_l376_37660

theorem total_distance_traveled :
  let radius := 50
  let angle := 45
  let num_girls := 8
  let cos_135 := Real.cos (135 * Real.pi / 180)
  let distance_one_way := radius * Real.sqrt (2 * (1 - cos_135))
  let distance_one_girl := 4 * distance_one_way
  let total_distance := num_girls * distance_one_girl
  total_distance = 1600 * Real.sqrt (2 + Real.sqrt 2) :=
by
  let radius := 50
  let angle := 45
  let num_girls := 8
  let cos_135 := Real.cos (135 * Real.pi / 180)
  let distance_one_way := radius * Real.sqrt (2 * (1 - cos_135))
  let distance_one_girl := 4 * distance_one_way
  let total_distance := num_girls * distance_one_girl
  show total_distance = 1600 * Real.sqrt (2 + Real.sqrt 2)
  sorry

end NUMINAMATH_GPT_total_distance_traveled_l376_37660


namespace NUMINAMATH_GPT_gcd_372_684_is_12_l376_37694

theorem gcd_372_684_is_12 : gcd 372 684 = 12 := by
  sorry

end NUMINAMATH_GPT_gcd_372_684_is_12_l376_37694


namespace NUMINAMATH_GPT_min_value_pm_pn_l376_37634

theorem min_value_pm_pn (x y : ℝ)
  (h : x ^ 2 - y ^ 2 / 3 = 1) 
  (hx : 1 ≤ x) : (8 * x - 3) = 5 :=
sorry

end NUMINAMATH_GPT_min_value_pm_pn_l376_37634


namespace NUMINAMATH_GPT_count_factors_of_product_l376_37618

theorem count_factors_of_product :
  let n := 8^4 * 7^3 * 9^1 * 5^5
  ∃ (count : ℕ), count = 936 ∧ 
    ∀ f : ℕ, f ∣ n → ∃ a b c d : ℕ,
      a ≤ 12 ∧ b ≤ 2 ∧ c ≤ 5 ∧ d ≤ 3 ∧ 
      f = 2^a * 3^b * 5^c * 7^d :=
by sorry

end NUMINAMATH_GPT_count_factors_of_product_l376_37618


namespace NUMINAMATH_GPT_trigonometric_expression_identity_l376_37664

theorem trigonometric_expression_identity :
  (2 * Real.sin (100 * Real.pi / 180) - Real.cos (70 * Real.pi / 180)) / Real.cos (20 * Real.pi / 180)
  = 2 * Real.sqrt 3 - 1 :=
sorry

end NUMINAMATH_GPT_trigonometric_expression_identity_l376_37664


namespace NUMINAMATH_GPT_inequality_holds_l376_37673

theorem inequality_holds (x : ℝ) : 3 * x^2 + 9 * x ≥ -12 :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_holds_l376_37673


namespace NUMINAMATH_GPT_alice_burger_spending_l376_37659

theorem alice_burger_spending :
  let daily_burgers := 4
  let burger_cost := 13
  let days_in_june := 30
  let mondays_wednesdays := 8
  let fridays := 4
  let fifth_purchase_coupons := 6
  let discount_10_percent := 0.9
  let discount_50_percent := 0.5
  let full_price := days_in_june * daily_burgers * burger_cost
  let discount_10 := mondays_wednesdays * daily_burgers * burger_cost * discount_10_percent
  let fridays_cost := (daily_burgers - 1) * fridays * burger_cost
  let discount_50 := fifth_purchase_coupons * burger_cost * discount_50_percent
  full_price - discount_10 - fridays_cost - discount_50 + fridays_cost = 1146.6 := by sorry

end NUMINAMATH_GPT_alice_burger_spending_l376_37659


namespace NUMINAMATH_GPT_cubic_roots_nature_l376_37674

-- Define the cubic polynomial function
def cubic_poly (x : ℝ) : ℝ := x^3 - 5 * x^2 + 8 * x - 4

-- Define the statement about the roots of the polynomial
theorem cubic_roots_nature :
  ∃ a b c : ℝ, cubic_poly a = 0 ∧ cubic_poly b = 0 ∧ cubic_poly c = 0 
  ∧ 0 < a ∧ 0 < b ∧ 0 < c :=
sorry

end NUMINAMATH_GPT_cubic_roots_nature_l376_37674


namespace NUMINAMATH_GPT_solutions_to_shifted_parabola_l376_37631

noncomputable def solution_equation := ∀ (a b : ℝ) (m : ℝ) (x : ℝ),
  (a ≠ 0) →
  ((a * (x + m) ^ 2 + b = 0) → (x = 2 ∨ x = -1)) →
  (a * (x - m + 2) ^ 2 + b = 0 → (x = -3 ∨ x = 0))

-- We'll leave the proof for this theorem as 'sorry'
theorem solutions_to_shifted_parabola (a b m : ℝ) (h : a ≠ 0)
  (h1 : ∀ (x : ℝ), a * (x + m) ^ 2 + b = 0 → (x = 2 ∨ x = -1)) 
  (x : ℝ) : 
  (a * (x - m + 2) ^ 2 + b = 0 → (x = -3 ∨ x = 0)) := sorry

end NUMINAMATH_GPT_solutions_to_shifted_parabola_l376_37631


namespace NUMINAMATH_GPT_largest_integer_value_l376_37616

theorem largest_integer_value (x : ℤ) : 
  (1/4 : ℚ) < (x : ℚ) / 6 ∧ (x : ℚ) / 6 < 2/3 ∧ (x : ℚ) < 10 → x = 3 := 
by
  sorry

end NUMINAMATH_GPT_largest_integer_value_l376_37616


namespace NUMINAMATH_GPT_weight_of_one_fan_l376_37604

theorem weight_of_one_fan
  (total_weight_with_fans : ℝ)
  (num_fans : ℕ)
  (empty_box_weight : ℝ)
  (h1 : total_weight_with_fans = 11.14)
  (h2 : num_fans = 14)
  (h3 : empty_box_weight = 0.5) :
  (total_weight_with_fans - empty_box_weight) / num_fans = 0.76 :=
by
  simp [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_weight_of_one_fan_l376_37604


namespace NUMINAMATH_GPT_abc_ineq_l376_37615

theorem abc_ineq (a b c : ℝ) (h₁ : a ≥ b) (h₂ : b ≥ c) (h₃ : c > 0) (h₄ : a + b + c = 3) :
  a * b^2 + b * c^2 + c * a^2 ≤ 27 / 8 :=
sorry

end NUMINAMATH_GPT_abc_ineq_l376_37615


namespace NUMINAMATH_GPT_dream_miles_driven_l376_37670

theorem dream_miles_driven (x : ℕ) (h : 4 * x + 4 * (x + 200) = 4000) : x = 400 :=
by
  sorry

end NUMINAMATH_GPT_dream_miles_driven_l376_37670


namespace NUMINAMATH_GPT_solve_r_l376_37652

-- Define E(a, b, c) as given
def E (a b c : ℕ) : ℕ := a * b^c

-- Lean 4 statement for the proof
theorem solve_r (r : ℕ) (r_pos : 0 < r) : E r r 3 = 625 → r = 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_r_l376_37652


namespace NUMINAMATH_GPT_travel_time_third_to_first_l376_37637

variable (boat_speed current_speed : ℝ) -- speeds of the boat and current
variable (d1 d2 d3 : ℝ) -- distances between the docks

-- Conditions
variable (h1 : 30 / 60 = d1 / (boat_speed - current_speed)) -- 30 minutes from one dock to another against current
variable (h2 : 18 / 60 = d2 / (boat_speed + current_speed)) -- 18 minutes from another dock to the third with current
variable (h3 : d1 + d2 = d3) -- Total distance is sum of d1 and d2

theorem travel_time_third_to_first : (d3 / (boat_speed - current_speed)) * 60 = 72 := 
by 
  -- here goes the proof which is omitted
  sorry

end NUMINAMATH_GPT_travel_time_third_to_first_l376_37637


namespace NUMINAMATH_GPT_y_work_days_24_l376_37698

-- Definitions of the conditions
def x_work_days := 36
def y_work_days (d : ℕ) := d
def y_worked_days := 12
def x_remaining_work_days := 18

-- Statement of the theorem
theorem y_work_days_24 : ∃ d : ℕ, (y_worked_days / y_work_days d + x_remaining_work_days / x_work_days = 1) ∧ d = 24 :=
  sorry

end NUMINAMATH_GPT_y_work_days_24_l376_37698


namespace NUMINAMATH_GPT_bianca_initial_cupcakes_l376_37649

theorem bianca_initial_cupcakes (X : ℕ) (h : X - 6 + 17 = 25) : X = 14 := by
  sorry

end NUMINAMATH_GPT_bianca_initial_cupcakes_l376_37649


namespace NUMINAMATH_GPT_cos_product_identity_l376_37610

theorem cos_product_identity :
  (Real.cos (20 * Real.pi / 180)) * (Real.cos (40 * Real.pi / 180)) *
  (Real.cos (60 * Real.pi / 180)) * (Real.cos (80 * Real.pi / 180)) = 1 / 16 := 
by
  sorry

end NUMINAMATH_GPT_cos_product_identity_l376_37610


namespace NUMINAMATH_GPT_total_rent_correct_recoup_investment_period_maximize_average_return_l376_37609

noncomputable def initialInvestment := 720000
noncomputable def firstYearRent := 54000
noncomputable def annualRentIncrease := 4000
noncomputable def maxRentalPeriod := 40

-- Conditions on the rental period
variable (x : ℝ) (hx : 0 < x ∧ x ≤ 40)

-- Function for total rent after x years
noncomputable def total_rent (x : ℝ) := 0.2 * x^2 + 5.2 * x

-- Condition for investment recoup period
noncomputable def recoupInvestmentTime := ∃ x : ℝ, x ≥ 10 ∧ total_rent x ≥ initialInvestment

-- Function for transfer price
noncomputable def transfer_price (x : ℝ) := -0.3 * x^2 + 10.56 * x + 57.6

-- Function for average return on investment
noncomputable def annual_avg_return (x : ℝ) := (transfer_price x + total_rent x - initialInvestment) / x

-- Statement of theorems
theorem total_rent_correct (x : ℝ) (hx : 0 < x ∧ x ≤ 40) :
  total_rent x = 0.2 * x^2 + 5.2 * x := sorry

theorem recoup_investment_period :
  ∃ x : ℝ, x ≥ 10 ∧ total_rent x ≥ initialInvestment := sorry

theorem maximize_average_return :
  ∃ x : ℝ, x = 12 ∧ (∀ y : ℝ, annual_avg_return x ≥ annual_avg_return y) := sorry

end NUMINAMATH_GPT_total_rent_correct_recoup_investment_period_maximize_average_return_l376_37609


namespace NUMINAMATH_GPT_increasing_function_fA_increasing_function_fB_increasing_function_fC_increasing_function_fD_l376_37647

noncomputable def fA (x : ℝ) : ℝ := -x
noncomputable def fB (x : ℝ) : ℝ := (2/3)^x
noncomputable def fC (x : ℝ) : ℝ := x^2
noncomputable def fD (x : ℝ) : ℝ := x^(1/3)

theorem increasing_function_fA : ¬∀ x y : ℝ, x < y → fA x < fA y := sorry
theorem increasing_function_fB : ¬∀ x y : ℝ, x < y → fB x < fB y := sorry
theorem increasing_function_fC : ¬∀ x y : ℝ, x < y → fC x < fC y := sorry
theorem increasing_function_fD : ∀ x y : ℝ, x < y → fD x < fD y := sorry

end NUMINAMATH_GPT_increasing_function_fA_increasing_function_fB_increasing_function_fC_increasing_function_fD_l376_37647


namespace NUMINAMATH_GPT_masha_number_l376_37636

theorem masha_number (x : ℝ) (n : ℤ) (ε : ℝ) (h1 : 0 ≤ ε) (h2 : ε < 1) (h3 : x = n + ε) (h4 : (n : ℝ) = 0.57 * x) : x = 100 / 57 :=
by
  sorry

end NUMINAMATH_GPT_masha_number_l376_37636


namespace NUMINAMATH_GPT_harry_total_payment_in_silvers_l376_37608

-- Definitions for the conditions
def spellbook_gold_cost : ℕ := 5
def spellbook_count : ℕ := 5
def potion_kit_silver_cost : ℕ := 20
def potion_kit_count : ℕ := 3
def owl_gold_cost : ℕ := 28
def silver_per_gold : ℕ := 9

-- Translate the total cost to silver
noncomputable def total_cost_in_silvers : ℕ :=
  spellbook_count * spellbook_gold_cost * silver_per_gold + 
  potion_kit_count * potion_kit_silver_cost + 
  owl_gold_cost * silver_per_gold

-- State the theorem
theorem harry_total_payment_in_silvers : total_cost_in_silvers = 537 :=
by
  unfold total_cost_in_silvers
  sorry

end NUMINAMATH_GPT_harry_total_payment_in_silvers_l376_37608


namespace NUMINAMATH_GPT_smallest_possible_value_of_M_l376_37657

theorem smallest_possible_value_of_M :
  ∃ (N M : ℕ), N > 0 ∧ M > 0 ∧ 
               ∃ (r_6 r_36 r_216 r_M : ℕ), 
               r_6 < 6 ∧ 
               r_6 < r_36 ∧ r_36 < 36 ∧ 
               r_36 < r_216 ∧ r_216 < 216 ∧ 
               r_216 < r_M ∧ 
               r_36 = (r_6 * r) ∧ 
               r_216 = (r_6 * r^2) ∧ 
               r_M = (r_6 * r^3) ∧ 
               Nat.mod N 6 = r_6 ∧ 
               Nat.mod N 36 = r_36 ∧ 
               Nat.mod N 216 = r_216 ∧ 
               Nat.mod N M = r_M ∧ 
               M = 2001 :=
sorry

end NUMINAMATH_GPT_smallest_possible_value_of_M_l376_37657


namespace NUMINAMATH_GPT_cost_per_revision_l376_37648

theorem cost_per_revision
  (x : ℝ)
  (initial_cost : ℝ)
  (revised_once : ℝ)
  (revised_twice : ℝ)
  (total_pages : ℝ)
  (total_cost : ℝ)
  (cost_per_page_first_time : ℝ) :
  initial_cost = cost_per_page_first_time * total_pages →
  revised_once * x + revised_twice * (2 * x) + initial_cost = total_cost →
  revised_once + revised_twice + (total_pages - (revised_once + revised_twice)) = total_pages →
  total_pages = 200 →
  initial_cost = 1000 →
  cost_per_page_first_time = 5 →
  revised_once = 80 →
  revised_twice = 20 →
  total_cost = 1360 →
  x = 3 :=
by
  intros h_initial h_total_cost h_tot_pages h_tot_pages_200 h_initial_1000 h_cost_5 h_revised_once h_revised_twice h_given_cost
  -- Proof steps to be filled
  sorry

end NUMINAMATH_GPT_cost_per_revision_l376_37648


namespace NUMINAMATH_GPT_library_books_total_l376_37626

-- Definitions for the conditions
def books_purchased_last_year : Nat := 50
def books_purchased_this_year : Nat := 3 * books_purchased_last_year
def books_before_last_year : Nat := 100

-- The library's current number of books
def total_books_now : Nat :=
  books_before_last_year + books_purchased_last_year + books_purchased_this_year

-- The proof statement
theorem library_books_total : total_books_now = 300 :=
by
  -- Placeholder for actual proof
  sorry

end NUMINAMATH_GPT_library_books_total_l376_37626


namespace NUMINAMATH_GPT_initial_amount_l376_37662

theorem initial_amount (x : ℝ) (h : 0.015 * x = 750) : x = 50000 :=
by
  sorry

end NUMINAMATH_GPT_initial_amount_l376_37662


namespace NUMINAMATH_GPT_sqrt_43_between_6_and_7_l376_37693

theorem sqrt_43_between_6_and_7 : 6 < Real.sqrt 43 ∧ Real.sqrt 43 < 7 := sorry

end NUMINAMATH_GPT_sqrt_43_between_6_and_7_l376_37693


namespace NUMINAMATH_GPT_solution_set_of_inequality_l376_37691

theorem solution_set_of_inequality :
  { x : ℝ | (x - 4) / (3 - 2*x) < 0 ∧ 3 - 2*x ≠ 0 } = { x : ℝ | x < 3 / 2 ∨ x > 4 } :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l376_37691


namespace NUMINAMATH_GPT_select_best_athlete_l376_37646

theorem select_best_athlete :
  let avg_A := 185
  let var_A := 3.6
  let avg_B := 180
  let var_B := 3.6
  let avg_C := 185
  let var_C := 7.4
  let avg_D := 180
  let var_D := 8.1
  avg_A = 185 ∧ var_A = 3.6 ∧
  avg_B = 180 ∧ var_B = 3.6 ∧
  avg_C = 185 ∧ var_C = 7.4 ∧
  avg_D = 180 ∧ var_D = 8.1 →
  (∃ x, (x = avg_A ∧ avg_A = 185 ∧ var_A = 3.6) ∧
        (∀ (y : ℕ), (y = avg_A) 
        → avg_A = 185 
        ∧ var_A <= var_C ∧ 
        var_A <= var_D 
        ∧ var_A <= var_B)) :=
by {
  sorry
}

end NUMINAMATH_GPT_select_best_athlete_l376_37646


namespace NUMINAMATH_GPT_gumball_water_wednesday_l376_37630

variable (water_Mon_Thu_Sat : ℕ)
variable (water_Tue_Fri_Sun : ℕ)
variable (water_total : ℕ)
variable (water_Wed : ℕ)

theorem gumball_water_wednesday 
  (h1 : water_Mon_Thu_Sat = 9) 
  (h2 : water_Tue_Fri_Sun = 8) 
  (h3 : water_total = 60) 
  (h4 : 3 * water_Mon_Thu_Sat + 3 * water_Tue_Fri_Sun + water_Wed = water_total) : 
  water_Wed = 9 := 
by 
  sorry

end NUMINAMATH_GPT_gumball_water_wednesday_l376_37630


namespace NUMINAMATH_GPT_find_a_l376_37625

theorem find_a (x y : ℝ) (a : ℝ) (h1 : x = 3) (h2 : y = 2) (h3 : a * x + 2 * y = 1) : a = -1 := by
  sorry

end NUMINAMATH_GPT_find_a_l376_37625


namespace NUMINAMATH_GPT_euclidean_remainder_2022_l376_37612

theorem euclidean_remainder_2022 : 
  (2022 ^ (2022 ^ 2022)) % 11 = 5 := 
by sorry

end NUMINAMATH_GPT_euclidean_remainder_2022_l376_37612


namespace NUMINAMATH_GPT_roots_in_interval_l376_37643

theorem roots_in_interval (a b : ℝ) (hb : b > 0) (h_discriminant : a^2 - 4 * b > 0)
  (h_root_interval : ∃ r1 r2 : ℝ, r1 + r2 = -a ∧ r1 * r2 = b ∧ ((-1 ≤ r1 ∧ r1 ≤ 1 ∧ (r2 < -1 ∨ 1 < r2)) ∨ (-1 ≤ r2 ∧ r2 ≤ 1 ∧ (r1 < -1 ∨ 1 < r1)))) : 
  ∃ r : ℝ, (r + a) * r + b = 0 ∧ -b < r ∧ r < b :=
by
  sorry

end NUMINAMATH_GPT_roots_in_interval_l376_37643


namespace NUMINAMATH_GPT_smallest_circle_radius_l376_37639

-- Define the problem as a proposition
theorem smallest_circle_radius (r : ℝ) (R1 R2 : ℝ) (hR1 : R1 = 6) (hR2 : R2 = 4) (h_right_triangle : (r + R2)^2 + (r + R1)^2 = (R2 + R1)^2) : r = 2 := 
sorry

end NUMINAMATH_GPT_smallest_circle_radius_l376_37639


namespace NUMINAMATH_GPT_gcd_factorial_8_10_l376_37632

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorial_8_10 : Nat.gcd (factorial 8) (factorial 10) = 40320 :=
by
  -- these pre-evaluations help Lean understand the factorial values
  have fact_8 : factorial 8 = 40320 := by sorry
  have fact_10 : factorial 10 = 3628800 := by sorry
  rw [fact_8, fact_10]
  -- the actual proof gets skipped here
  sorry

end NUMINAMATH_GPT_gcd_factorial_8_10_l376_37632


namespace NUMINAMATH_GPT_marla_parent_teacher_night_time_l376_37688

def errand_time := 110 -- total minutes on the errand
def driving_time_oneway := 20 -- minutes driving one way to school
def driving_time_return := 20 -- minutes driving one way back home

def total_driving_time := driving_time_oneway + driving_time_return

def time_at_parent_teacher_night := errand_time - total_driving_time

theorem marla_parent_teacher_night_time : time_at_parent_teacher_night = 70 :=
by
  -- Lean proof goes here
  sorry

end NUMINAMATH_GPT_marla_parent_teacher_night_time_l376_37688


namespace NUMINAMATH_GPT_cafe_purchase_max_items_l376_37656

theorem cafe_purchase_max_items (total_money sandwich_cost soft_drink_cost : ℝ) (total_money_pos sandwich_cost_pos soft_drink_cost_pos : total_money > 0 ∧ sandwich_cost > 0 ∧ soft_drink_cost > 0) :
    total_money = 40 ∧ sandwich_cost = 5 ∧ soft_drink_cost = 1.50 →
    ∃ s d : ℕ, s + d = 10 ∧ total_money = sandwich_cost * s + soft_drink_cost * d :=
by
  sorry

end NUMINAMATH_GPT_cafe_purchase_max_items_l376_37656


namespace NUMINAMATH_GPT_remaining_pieces_to_fold_l376_37667

-- Define the initial counts of shirts and shorts
def initial_shirts : ℕ := 20
def initial_shorts : ℕ := 8

-- Define the counts of folded shirts and shorts
def folded_shirts : ℕ := 12
def folded_shorts : ℕ := 5

-- The target theorem to prove the remaining pieces of clothing to fold
theorem remaining_pieces_to_fold :
  initial_shirts + initial_shorts - (folded_shirts + folded_shorts) = 11 := 
by
  sorry

end NUMINAMATH_GPT_remaining_pieces_to_fold_l376_37667


namespace NUMINAMATH_GPT_cannot_achieve_55_cents_with_six_coins_l376_37696

theorem cannot_achieve_55_cents_with_six_coins :
  ¬∃ (a b c d e : ℕ), 
    a + b + c + d + e = 6 ∧ 
    a * 1 + b * 5 + c * 10 + d * 25 + e * 50 = 55 := 
sorry

end NUMINAMATH_GPT_cannot_achieve_55_cents_with_six_coins_l376_37696


namespace NUMINAMATH_GPT_cylinder_height_relationship_l376_37658

theorem cylinder_height_relationship
  (r1 h1 r2 h2 : ℝ)
  (volume_equal : π * r1^2 * h1 = π * r2^2 * h2)
  (radius_relationship : r2 = 1.2 * r1) :
  h1 = 1.44 * h2 :=
by sorry

end NUMINAMATH_GPT_cylinder_height_relationship_l376_37658


namespace NUMINAMATH_GPT_arithmetic_mean_16_24_40_32_l376_37624

theorem arithmetic_mean_16_24_40_32 : (16 + 24 + 40 + 32) / 4 = 28 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_16_24_40_32_l376_37624


namespace NUMINAMATH_GPT_even_heads_probability_is_17_over_25_l376_37672

-- Definition of the probabilities of heads and tails
def prob_tails : ℚ := 1 / 5
def prob_heads : ℚ := 4 * prob_tails

-- Definition of the probability of getting an even number of heads in two flips
def even_heads_prob (p_heads p_tails : ℚ) : ℚ :=
  p_tails * p_tails + p_heads * p_heads

-- Theorem statement
theorem even_heads_probability_is_17_over_25 :
  even_heads_prob prob_heads prob_tails = 17 / 25 := by
  sorry

end NUMINAMATH_GPT_even_heads_probability_is_17_over_25_l376_37672


namespace NUMINAMATH_GPT_sum_of_numbers_l376_37685

theorem sum_of_numbers : 1324 + 2431 + 3142 + 4213 + 1234 = 12344 := sorry

end NUMINAMATH_GPT_sum_of_numbers_l376_37685


namespace NUMINAMATH_GPT_part_a_part_b_l376_37606

def happy (n : ℕ) : Prop :=
  ∃ (a b : ℤ), a^2 + b^2 = n

theorem part_a (t : ℕ) (ht : happy t) : happy (2 * t) := 
sorry

theorem part_b (t : ℕ) (ht : happy t) : ¬ happy (3 * t) := 
sorry

end NUMINAMATH_GPT_part_a_part_b_l376_37606


namespace NUMINAMATH_GPT_market_value_decrease_l376_37640

noncomputable def percentage_decrease_each_year : ℝ :=
  let original_value := 8000
  let value_after_two_years := 3200
  let p := 1 - (value_after_two_years / original_value)^(1 / 2)
  p * 100

theorem market_value_decrease :
  let p := percentage_decrease_each_year
  abs (p - 36.75) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_market_value_decrease_l376_37640


namespace NUMINAMATH_GPT_circumcircle_eq_of_triangle_vertices_l376_37619

theorem circumcircle_eq_of_triangle_vertices (A B C: ℝ × ℝ) (hA : A = (0, 4)) (hB : B = (0, 0)) (hC : C = (3, 0)) :
  ∃ D E F : ℝ,
    x^2 + y^2 + D*x + E*y + F = 0 ∧
    (x - 3/2)^2 + (y - 2)^2 = 25/4 :=
by 
  sorry

end NUMINAMATH_GPT_circumcircle_eq_of_triangle_vertices_l376_37619


namespace NUMINAMATH_GPT_plumber_charge_shower_l376_37683

theorem plumber_charge_shower (S : ℝ) 
  (sink_cost : ℝ := 30) 
  (toilet_cost : ℝ := 50)
  (max_earning : ℝ := 250)
  (first_job_toilets : ℝ := 3) (first_job_sinks : ℝ := 3)
  (second_job_toilets : ℝ := 2) (second_job_sinks : ℝ := 5)
  (third_job_toilets : ℝ := 1) (third_job_showers : ℝ := 2) (third_job_sinks : ℝ := 3) :
  2 * S + 1 * toilet_cost + 3 * sink_cost ≤ max_earning → S ≤ 55 :=
by
  sorry

end NUMINAMATH_GPT_plumber_charge_shower_l376_37683


namespace NUMINAMATH_GPT_sin_double_alpha_l376_37681

theorem sin_double_alpha (α : ℝ) 
  (h : Real.tan (α - Real.pi / 4) = Real.sqrt 2 / 4) : 
  Real.sin (2 * α) = 7 / 9 := by
  sorry

end NUMINAMATH_GPT_sin_double_alpha_l376_37681


namespace NUMINAMATH_GPT_positive_value_m_l376_37666

theorem positive_value_m (m : ℝ) : (∃ x : ℝ, 16 * x^2 + m * x + 4 = 0 ∧ ∀ y : ℝ, 16 * y^2 + m * y + 4 = 0 → y = x) → m = 16 :=
by
  sorry

end NUMINAMATH_GPT_positive_value_m_l376_37666


namespace NUMINAMATH_GPT_sean_total_spending_l376_37601

noncomputable def cost_first_bakery_euros : ℝ :=
  let almond_croissants := 2 * 4.00
  let salami_cheese_croissants := 3 * 5.00
  let total_before_discount := almond_croissants + salami_cheese_croissants
  total_before_discount * 0.90 -- 10% discount

noncomputable def cost_second_bakery_pounds : ℝ :=
  let plain_croissants := 3 * 3.50 -- buy-3-get-1-free
  let focaccia := 5.00
  let total_before_tax := plain_croissants + focaccia
  total_before_tax * 1.05 -- 5% tax

noncomputable def cost_cafe_dollars : ℝ :=
  let lattes := 3 * 3.00
  lattes * 0.85 -- 15% student discount

noncomputable def first_bakery_usd : ℝ :=
  cost_first_bakery_euros * 1.15 -- converting euros to dollars

noncomputable def second_bakery_usd : ℝ :=
  cost_second_bakery_pounds * 1.35 -- converting pounds to dollars

noncomputable def total_cost_sean_spends : ℝ :=
  first_bakery_usd + second_bakery_usd + cost_cafe_dollars

theorem sean_total_spending : total_cost_sean_spends = 53.44 :=
  by
  -- The proof can be handled here
  sorry

end NUMINAMATH_GPT_sean_total_spending_l376_37601


namespace NUMINAMATH_GPT_fish_count_seventh_day_l376_37607

-- Define the initial state and transformations
def fish_count (n: ℕ) :=
  if n = 0 then 6
  else
    if n = 3 then fish_count (n-1) / 3 * 2 * 2 * 2 - fish_count (n-1) / 3
    else if n = 5 then (fish_count (n-1) * 2) / 4 * 3
    else if n = 6 then fish_count (n-1) * 2 + 15
    else fish_count (n-1) * 2

theorem fish_count_seventh_day : fish_count 7 = 207 :=
by
  sorry

end NUMINAMATH_GPT_fish_count_seventh_day_l376_37607


namespace NUMINAMATH_GPT_composite_proposition_l376_37611

theorem composite_proposition :
  (∀ x : ℝ, x^2 ≥ 0) ∧ ¬ (1 < 0) :=
by
  sorry

end NUMINAMATH_GPT_composite_proposition_l376_37611


namespace NUMINAMATH_GPT_cannot_assemble_highlighted_shape_l376_37627

-- Define the rhombus shape with its properties
structure Rhombus :=
  (white_triangle gray_triangle : Prop)

-- Define the assembly condition
def can_rotate (shape : Rhombus) : Prop := sorry

-- Define the specific shape highlighted that Petya cannot form
def highlighted_shape : Prop := sorry

-- The statement we need to prove
theorem cannot_assemble_highlighted_shape (shape : Rhombus) 
  (h_rotate : can_rotate shape)
  (h_highlight : highlighted_shape) : false :=
by sorry

end NUMINAMATH_GPT_cannot_assemble_highlighted_shape_l376_37627


namespace NUMINAMATH_GPT_sampling_method_is_systematic_sampling_l376_37620

-- Definitions based on the problem's conditions
def produces_products (factory : Type) : Prop := sorry
def uses_conveyor_belt (factory : Type) : Prop := sorry
def takes_item_every_5_minutes (inspector : Type) : Prop := sorry

-- Lean 4 statement to prove the question equals the answer given the conditions
theorem sampling_method_is_systematic_sampling
  (factory : Type)
  (inspector : Type)
  (h1 : produces_products factory)
  (h2 : uses_conveyor_belt factory)
  (h3 : takes_item_every_5_minutes inspector) :
  systematic_sampling_method := 
sorry

end NUMINAMATH_GPT_sampling_method_is_systematic_sampling_l376_37620


namespace NUMINAMATH_GPT_emma_chocolates_l376_37679

theorem emma_chocolates 
  (x : ℕ) 
  (h1 : ∃ l : ℕ, x = l + 10) 
  (h2 : ∃ l : ℕ, l = x / 3) : 
  x = 15 := 
  sorry

end NUMINAMATH_GPT_emma_chocolates_l376_37679


namespace NUMINAMATH_GPT_sum_lent_out_l376_37680

theorem sum_lent_out (P R : ℝ) (h1 : 720 = P + (P * R * 2) / 100) (h2 : 1020 = P + (P * R * 7) / 100) : P = 600 := by
  sorry

end NUMINAMATH_GPT_sum_lent_out_l376_37680


namespace NUMINAMATH_GPT_isosceles_triangle_circum_incenter_distance_l376_37651

variable {R r d : ℝ}

/-- The distance \(d\) between the centers of the circumscribed circle and the inscribed circle of an isosceles triangle satisfies \(d = \sqrt{R(R - 2r)}\) --/
theorem isosceles_triangle_circum_incenter_distance (hR : 0 < R) (hr : 0 < r) 
  (hIso : ∃ (A B C : ℝ × ℝ), (A ≠ B) ∧ (A ≠ C) ∧ (B ≠ C) ∧ (dist A B = dist A C)) 
  : d = Real.sqrt (R * (R - 2 * r)) :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_circum_incenter_distance_l376_37651


namespace NUMINAMATH_GPT_range_of_k_l376_37665

theorem range_of_k (k : ℝ) (hₖ : 0 < k) :
  (∃ x : ℝ, 1 = x^2 + (k^2 / x^2)) → 0 < k ∧ k ≤ 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l376_37665


namespace NUMINAMATH_GPT_range_of_a_iff_l376_37675

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ (x : ℝ), 0 < x → (Real.log x / Real.log a) ≤ x ∧ x ≤ a ^ x

theorem range_of_a_iff (a : ℝ) : (a ≥ Real.exp (Real.exp (-1))) ↔ range_of_a a :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_iff_l376_37675


namespace NUMINAMATH_GPT_problem_y_equals_x_squared_plus_x_minus_6_l376_37655

theorem problem_y_equals_x_squared_plus_x_minus_6 (x y : ℝ) :
  (y = x^2 + x - 6 ∧ x = 0 → y = -6) ∧ 
  (y = 0 → x = -3 ∨ x = 2) :=
by
  sorry

end NUMINAMATH_GPT_problem_y_equals_x_squared_plus_x_minus_6_l376_37655


namespace NUMINAMATH_GPT_probability_Xavier_Yvonne_not_Zelda_l376_37642

-- Define the probabilities of success for Xavier, Yvonne, and Zelda
def pXavier := 1 / 5
def pYvonne := 1 / 2
def pZelda := 5 / 8

-- Define the probability that Zelda does not solve the problem
def pNotZelda := 1 - pZelda

-- The desired probability that we want to prove equals 3/80
def desiredProbability := (pXavier * pYvonne * pNotZelda) = (3 / 80)

-- The statement of the problem in Lean
theorem probability_Xavier_Yvonne_not_Zelda :
  desiredProbability := by
  sorry

end NUMINAMATH_GPT_probability_Xavier_Yvonne_not_Zelda_l376_37642


namespace NUMINAMATH_GPT_only_n_equal_one_l376_37654

theorem only_n_equal_one (n : ℕ) (hn : 0 < n) : 
  (5 ^ (n - 1) + 3 ^ (n - 1)) ∣ (5 ^ n + 3 ^ n) → n = 1 := by
  intro h_div
  sorry

end NUMINAMATH_GPT_only_n_equal_one_l376_37654


namespace NUMINAMATH_GPT_range_of_vector_magnitude_l376_37613

variable {V : Type} [NormedAddCommGroup V]

theorem range_of_vector_magnitude
  (A B C : V)
  (h_AB : ‖A - B‖ = 8)
  (h_AC : ‖A - C‖ = 5) :
  3 ≤ ‖B - C‖ ∧ ‖B - C‖ ≤ 13 :=
sorry

end NUMINAMATH_GPT_range_of_vector_magnitude_l376_37613
