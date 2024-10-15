import Mathlib

namespace NUMINAMATH_GPT_tangent_product_value_l1947_194756

theorem tangent_product_value (A B : ℝ) (hA : A = 20) (hB : B = 25) :
    (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = 2 :=
sorry

end NUMINAMATH_GPT_tangent_product_value_l1947_194756


namespace NUMINAMATH_GPT_find_divisor_l1947_194750

theorem find_divisor {x y : ℤ} (h1 : (x - 5) / y = 7) (h2 : (x - 24) / 10 = 3) : y = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l1947_194750


namespace NUMINAMATH_GPT_polynomial_of_degree_2_l1947_194743

noncomputable def polynomialSeq (n : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → 
    ∃ (f_k f_k1 f_k2 : Polynomial ℝ),
      f_k ≠ Polynomial.C 0 ∧ (f_k * f_k1 = f_k1.comp f_k2)

theorem polynomial_of_degree_2 (n : ℕ) (h : n ≥ 3) :
  polynomialSeq n → 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → 
    ∃ f : Polynomial ℝ, f = Polynomial.X ^ 2 :=
sorry

end NUMINAMATH_GPT_polynomial_of_degree_2_l1947_194743


namespace NUMINAMATH_GPT_prob_simultaneous_sequences_l1947_194752

-- Definitions for coin probabilities
def prob_heads_A : ℝ := 0.3
def prob_tails_A : ℝ := 0.7
def prob_heads_B : ℝ := 0.4
def prob_tails_B : ℝ := 0.6

-- Definitions for required sequences
def seq_TTH_A : ℝ := prob_tails_A * prob_tails_A * prob_heads_A
def seq_HTT_B : ℝ := prob_heads_B * prob_tails_B * prob_tails_B

-- Main assertion
theorem prob_simultaneous_sequences :
  seq_TTH_A * seq_HTT_B = 0.021168 :=
by
  sorry

end NUMINAMATH_GPT_prob_simultaneous_sequences_l1947_194752


namespace NUMINAMATH_GPT_sum_of_positive_odd_divisors_of_90_l1947_194708

-- Define the conditions: the odd divisors of 45
def odd_divisors_45 : List ℕ := [1, 3, 5, 9, 15, 45]

-- Define a function to sum the elements of a list
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Now, state the theorem
theorem sum_of_positive_odd_divisors_of_90 : sum_list odd_divisors_45 = 78 := by
  sorry

end NUMINAMATH_GPT_sum_of_positive_odd_divisors_of_90_l1947_194708


namespace NUMINAMATH_GPT_number_of_action_figures_bought_l1947_194716

-- Definitions of conditions
def cost_of_board_game : ℕ := 2
def cost_per_action_figure : ℕ := 7
def total_spent : ℕ := 30

-- The problem to prove
theorem number_of_action_figures_bought : 
  ∃ (n : ℕ), total_spent - cost_of_board_game = n * cost_per_action_figure ∧ n = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_action_figures_bought_l1947_194716


namespace NUMINAMATH_GPT_ancient_china_pentatonic_scale_l1947_194739

theorem ancient_china_pentatonic_scale (a : ℝ) (h : a * (2/3) * (4/3) * (2/3) = 32) : a = 54 :=
by
  sorry

end NUMINAMATH_GPT_ancient_china_pentatonic_scale_l1947_194739


namespace NUMINAMATH_GPT_trader_total_discount_correct_l1947_194759

theorem trader_total_discount_correct :
  let CP_A := 200
  let CP_B := 150
  let CP_C := 100
  let MSP_A := CP_A + 0.50 * CP_A
  let MSP_B := CP_B + 0.50 * CP_B
  let MSP_C := CP_C + 0.50 * CP_C
  let SP_A := 0.99 * CP_A
  let SP_B := 0.97 * CP_B
  let SP_C := 0.98 * CP_C
  let discount_A := MSP_A - SP_A
  let discount_B := MSP_B - SP_B
  let discount_C := MSP_C - SP_C
  let total_discount := discount_A + discount_B + discount_C
  total_discount = 233.5 := by sorry

end NUMINAMATH_GPT_trader_total_discount_correct_l1947_194759


namespace NUMINAMATH_GPT_value_of_m_l1947_194737

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x

theorem value_of_m (a b m : ℝ) (h₀ : m ≠ 0)
  (h₁ : 3 * m^2 + 2 * a * m + b = 0)
  (h₂ : m^2 + a * m + b = 0)
  (h₃ : ∃ x, f x a b = 1/2) :
  m = 3/2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_l1947_194737


namespace NUMINAMATH_GPT_f_positive_l1947_194790

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry
noncomputable def f'' : ℝ → ℝ := sorry

axiom f_monotonically_decreasing : ∀ x y : ℝ, x < y → f x > f y
axiom inequality_condition : ∀ x : ℝ, (f x) / (f'' x) + x < 1

theorem f_positive : ∀ x : ℝ, f x > 0 :=
by sorry

end NUMINAMATH_GPT_f_positive_l1947_194790


namespace NUMINAMATH_GPT_minimum_f_zero_iff_t_is_2sqrt2_l1947_194782

noncomputable def f (x t : ℝ) : ℝ := 4 * x^4 - 6 * t * x^3 + (2 * t + 6) * x^2 - 3 * t * x + 1

theorem minimum_f_zero_iff_t_is_2sqrt2 :
  (∀ x > 0, f x t ≥ 0) ∧ (∃ x > 0, f x t = 0) ↔ t = 2 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_minimum_f_zero_iff_t_is_2sqrt2_l1947_194782


namespace NUMINAMATH_GPT_arithmetic_expression_l1947_194721

theorem arithmetic_expression :
  7 / 2 - 3 - 5 + 3 * 4 = 7.5 :=
by {
  -- We state the main equivalence to be proven
  sorry
}

end NUMINAMATH_GPT_arithmetic_expression_l1947_194721


namespace NUMINAMATH_GPT_number_of_heaps_is_5_l1947_194788

variable (bundles : ℕ) (bunches : ℕ) (heaps : ℕ) (total_removed : ℕ)
variable (sheets_per_bunch : ℕ) (sheets_per_bundle : ℕ) (sheets_per_heap : ℕ)

def number_of_heaps (bundles : ℕ) (sheets_per_bundle : ℕ)
                    (bunches : ℕ) (sheets_per_bunch : ℕ)
                    (total_removed : ℕ) (sheets_per_heap : ℕ) :=
  (total_removed - (bundles * sheets_per_bundle + bunches * sheets_per_bunch)) / sheets_per_heap

theorem number_of_heaps_is_5 :
  number_of_heaps 3 2 2 4 114 20 = 5 :=
by
  unfold number_of_heaps
  sorry

end NUMINAMATH_GPT_number_of_heaps_is_5_l1947_194788


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1947_194734

theorem quadratic_inequality_solution (k : ℝ) :
  (∀ x : ℝ, k * x^2 + k * x - (3 / 4) < 0) ↔ -3 < k ∧ k ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1947_194734


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_l1947_194748

-- Problem 1: Prove that x = 1 given 6x - 7 = 4x - 5
theorem problem1_solution (x : ℝ) (h : 6 * x - 7 = 4 * x - 5) : x = 1 := by
  sorry


-- Problem 2: Prove that x = -1 given (3x - 1) / 4 - 1 = (5x - 7) / 6
theorem problem2_solution (x : ℝ) (h : (3 * x - 1) / 4 - 1 = (5 * x - 7) / 6) : x = -1 := by
  sorry

end NUMINAMATH_GPT_problem1_solution_problem2_solution_l1947_194748


namespace NUMINAMATH_GPT_eval_p_positive_int_l1947_194732

theorem eval_p_positive_int (p : ℕ) : 
  (∃ n : ℕ, n > 0 ∧ (4 * p + 20) = n * (3 * p - 6)) ↔ p = 3 ∨ p = 4 ∨ p = 15 ∨ p = 28 := 
by sorry

end NUMINAMATH_GPT_eval_p_positive_int_l1947_194732


namespace NUMINAMATH_GPT_hyperbola_equation_l1947_194744

theorem hyperbola_equation {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  (∀ {x y : ℝ}, x^2 / 12 + y^2 / 4 = 1 → True) →
  (∀ {x y : ℝ}, x^2 / a^2 - y^2 / b^2 = 1 → True) →
  (∀ {x y : ℝ}, y = Real.sqrt 3 * x → True) →
  (∃ k : ℝ, 4 < k ∧ k < 12 ∧ 2 = 12 - k ∧ 6 = k - 4) →
  a = 2 ∧ b = 6 := by
  intros h_ellipse h_hyperbola h_asymptote h_k
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l1947_194744


namespace NUMINAMATH_GPT_sin_cos_from_tan_in_second_quadrant_l1947_194729

theorem sin_cos_from_tan_in_second_quadrant (α : ℝ) 
  (h1 : Real.tan α = -2) 
  (h2 : α ∈ Set.Ioo (π / 2) π) : 
  Real.sin α = 2 * Real.sqrt 5 / 5 ∧ Real.cos α = -Real.sqrt 5 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_from_tan_in_second_quadrant_l1947_194729


namespace NUMINAMATH_GPT_smallest_five_digit_multiple_of_18_l1947_194733

theorem smallest_five_digit_multiple_of_18 : ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 18 = 0 ∧ ∀ m : ℕ, 10000 ≤ m ∧ m ≤ 99999 ∧ m % 18 = 0 → n ≤ m :=
  sorry

end NUMINAMATH_GPT_smallest_five_digit_multiple_of_18_l1947_194733


namespace NUMINAMATH_GPT_product_of_all_possible_values_l1947_194755

theorem product_of_all_possible_values (x : ℝ) :
  (|16 / x + 4| = 3) → ((x = -16 ∨ x = -16 / 7) →
  (x_1 = -16 ∧ x_2 = -16 / 7) →
  (x_1 * x_2 = 256 / 7)) :=
sorry

end NUMINAMATH_GPT_product_of_all_possible_values_l1947_194755


namespace NUMINAMATH_GPT_steve_speed_ratio_l1947_194711

/-- Define the distance from Steve's house to work. -/
def distance_to_work := 30

/-- Define the total time spent on the road by Steve. -/
def total_time_on_road := 6

/-- Define Steve's speed on the way back from work. -/
def speed_back := 15

/-- Calculate the ratio of Steve's speed on the way back to his speed on the way to work. -/
theorem steve_speed_ratio (v : ℝ) (h_v_pos : v > 0) 
    (h1 : distance_to_work / v + distance_to_work / speed_back = total_time_on_road) :
    speed_back / v = 2 := 
by
  -- We will provide the proof here
  sorry

end NUMINAMATH_GPT_steve_speed_ratio_l1947_194711


namespace NUMINAMATH_GPT_max_value_harmonic_series_l1947_194745

theorem max_value_harmonic_series (k l m : ℕ) (hk : 0 < k) (hl : 0 < l) (hm : 0 < m)
  (h : 1/k + 1/l + 1/m < 1) : 
  (1/2 + 1/3 + 1/7) = 41/42 := 
sorry

end NUMINAMATH_GPT_max_value_harmonic_series_l1947_194745


namespace NUMINAMATH_GPT_find_top_row_number_l1947_194735

theorem find_top_row_number (x z : ℕ) (h1 : 8 = x * 2) (h2 : 16 = 2 * z)
  (h3 : 56 = 8 * 7) (h4 : 112 = 16 * 7) : x = 4 :=
by sorry

end NUMINAMATH_GPT_find_top_row_number_l1947_194735


namespace NUMINAMATH_GPT_range_of_m_l1947_194709

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 9 * x - m

theorem range_of_m (H : ∃ (x_0 : ℝ), x_0 ≠ 0 ∧ f 0 x_0 = f 0 x_0) : 0 < m ∧ m < 1 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1947_194709


namespace NUMINAMATH_GPT_equivalent_modulo_l1947_194725

theorem equivalent_modulo:
  123^2 * 947 % 60 = 3 :=
by
  sorry

end NUMINAMATH_GPT_equivalent_modulo_l1947_194725


namespace NUMINAMATH_GPT_no_negative_roots_but_at_least_one_positive_root_l1947_194727

def f (x : ℝ) : ℝ := x^6 - 3 * x^5 - 6 * x^3 - x + 8

theorem no_negative_roots_but_at_least_one_positive_root :
  (∀ x : ℝ, x < 0 → f x ≠ 0) ∧ (∃ x : ℝ, x > 0 ∧ f x = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_no_negative_roots_but_at_least_one_positive_root_l1947_194727


namespace NUMINAMATH_GPT_simplify_expression_value_at_3_value_at_4_l1947_194799

-- Define the original expression
def original_expr (x : ℕ) : ℚ := (1 - 1 / (x - 1)) / ((x^2 - 4) / (x^2 - 2 * x + 1))

-- Property 1: Simplify the expression
theorem simplify_expression (x : ℕ) (h1 : x ≠ 1) (h2 : x ≠ 2) : 
  original_expr x = (x - 1) / (x + 2) :=
sorry

-- Property 2: Evaluate the expression at x = 3
theorem value_at_3 : original_expr 3 = 2 / 5 :=
sorry

-- Property 3: Evaluate the expression at x = 4
theorem value_at_4 : original_expr 4 = 1 / 2 :=
sorry

end NUMINAMATH_GPT_simplify_expression_value_at_3_value_at_4_l1947_194799


namespace NUMINAMATH_GPT_solve_arcsin_eq_l1947_194775

open Real

noncomputable def problem_statement (x : ℝ) : Prop :=
  arcsin (sin x) = (3 * x) / 4

theorem solve_arcsin_eq(x : ℝ) (h : problem_statement x) (h_range: - (2 * π) / 3 ≤ x ∧ x ≤ (2 * π) / 3) : x = 0 :=
sorry

end NUMINAMATH_GPT_solve_arcsin_eq_l1947_194775


namespace NUMINAMATH_GPT_clothes_prices_l1947_194761

theorem clothes_prices (total_cost : ℕ) (shirt_more : ℕ) (trousers_price : ℕ) (shirt_price : ℕ)
  (h1 : total_cost = 185)
  (h2 : shirt_more = 5)
  (h3 : shirt_price = 2 * trousers_price + shirt_more)
  (h4 : total_cost = shirt_price + trousers_price) : 
  trousers_price = 60 ∧ shirt_price = 125 :=
  by sorry

end NUMINAMATH_GPT_clothes_prices_l1947_194761


namespace NUMINAMATH_GPT_smithtown_left_handed_women_percentage_l1947_194749

theorem smithtown_left_handed_women_percentage
    (x y : ℕ)
    (H1 : 3 * x + x = 4 * x)
    (H2 : 3 * y + 2 * y = 5 * y)
    (H3 : 4 * x = 5 * y) :
    (x / (4 * x)) * 100 = 25 :=
by sorry

end NUMINAMATH_GPT_smithtown_left_handed_women_percentage_l1947_194749


namespace NUMINAMATH_GPT_minimum_3x_4y_l1947_194741

theorem minimum_3x_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 3 * x + 4 * y ≥ 5 :=
by
  sorry

end NUMINAMATH_GPT_minimum_3x_4y_l1947_194741


namespace NUMINAMATH_GPT_greatest_possible_value_l1947_194795

theorem greatest_possible_value (x y : ℝ) (h1 : x^2 + y^2 = 98) (h2 : x * y = 40) : x + y = Real.sqrt 178 :=
by sorry

end NUMINAMATH_GPT_greatest_possible_value_l1947_194795


namespace NUMINAMATH_GPT_VerifyMultiplicationProperties_l1947_194723

theorem VerifyMultiplicationProperties (α : Type) [Semiring α] :
  ((∀ x y z : α, (x * y) * z = x * (y * z)) ∧
   (∀ x y : α, x * y = y * x) ∧
   (∀ x y z : α, x * (y + z) = x * y + x * z) ∧
   (∃ e : α, ∀ x : α, x * e = x)) := by
  sorry

end NUMINAMATH_GPT_VerifyMultiplicationProperties_l1947_194723


namespace NUMINAMATH_GPT_cakes_remaining_l1947_194784

theorem cakes_remaining (cakes_made : ℕ) (cakes_sold : ℕ) (h_made : cakes_made = 149) (h_sold : cakes_sold = 10) :
  (cakes_made - cakes_sold) = 139 :=
by
  cases h_made
  cases h_sold
  sorry

end NUMINAMATH_GPT_cakes_remaining_l1947_194784


namespace NUMINAMATH_GPT_transformation_maps_segment_l1947_194747

variables (C D : ℝ × ℝ) (C' D' : ℝ × ℝ)

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

theorem transformation_maps_segment :
  reflect_x (reflect_y (3, -2)) = (-3, 2) ∧ reflect_x (reflect_y (4, -5)) = (-4, 5) :=
by {
  sorry
}

end NUMINAMATH_GPT_transformation_maps_segment_l1947_194747


namespace NUMINAMATH_GPT_range_of_f_l1947_194713

noncomputable def f (x : ℝ) : ℝ := if x < 1 then 3 * x - 1 else 2 * x ^ 2

theorem range_of_f (a : ℝ) : (f (f a) = 2 * (f a) ^ 2) ↔ (a ≥ 2 / 3 ∨ a = 1 / 2) := 
  sorry

end NUMINAMATH_GPT_range_of_f_l1947_194713


namespace NUMINAMATH_GPT_perpendicular_vectors_k_value_l1947_194730

theorem perpendicular_vectors_k_value (k : ℝ) (a b: ℝ × ℝ)
  (h_a : a = (-1, 3)) (h_b : b = (1, k)) (h_perp : (a.1 * b.1 + a.2 * b.2) = 0) :
  k = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_k_value_l1947_194730


namespace NUMINAMATH_GPT_number_of_students_in_both_ball_and_track_l1947_194706

variable (total studentsSwim studentsTrack studentsBall bothSwimTrack bothSwimBall bothTrackBall : ℕ)
variable (noAllThree : Prop)

theorem number_of_students_in_both_ball_and_track
  (h_total : total = 26)
  (h_swim : studentsSwim = 15)
  (h_track : studentsTrack = 8)
  (h_ball : studentsBall = 14)
  (h_both_swim_track : bothSwimTrack = 3)
  (h_both_swim_ball : bothSwimBall = 3)
  (h_no_all_three : noAllThree) :
  bothTrackBall = 5 := by
  sorry

end NUMINAMATH_GPT_number_of_students_in_both_ball_and_track_l1947_194706


namespace NUMINAMATH_GPT_angle_E_in_quadrilateral_EFGH_l1947_194718

theorem angle_E_in_quadrilateral_EFGH 
  (angle_E angle_F angle_G angle_H : ℝ) 
  (h1 : angle_E = 2 * angle_F)
  (h2 : angle_E = 3 * angle_G)
  (h3 : angle_E = 6 * angle_H)
  (sum_angles : angle_E + angle_F + angle_G + angle_H = 360) : 
  angle_E = 180 :=
by
  sorry

end NUMINAMATH_GPT_angle_E_in_quadrilateral_EFGH_l1947_194718


namespace NUMINAMATH_GPT_volume_relation_l1947_194758

-- Definitions for points and geometry structures
structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Tetrahedron :=
(A B C D : Point3D)

-- Volume function for Tetrahedron
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

-- Given conditions
variable {A B C D D1 A1 B1 C1 : Point3D} 

-- D_1 is the centroid of triangle ABC
axiom centroid_D1 (A B C D1 : Point3D) : D1 = Point3D.mk ((A.x + B.x + C.x) / 3) ((A.y + B.y + C.y) / 3) ((A.z + B.z + C.z) / 3)

-- Line through A parallel to DD_1 intersects plane BCD at A1
axiom A1_condition (A B C D D1 A1 : Point3D) : sorry
-- Line through B parallel to DD_1 intersects plane ACD at B1
axiom B1_condition (A B C D D1 B1 : Point3D) : sorry
-- Line through C parallel to DD_1 intersects plane ABD at C1
axiom C1_condition (A B C D D1 C1 : Point3D) : sorry

-- Volume relation to be proven
theorem volume_relation (t1 t2 : Tetrahedron) (h : t1.A = A ∧ t1.B = B ∧ t1.C = C ∧ t1.D = D ∧
                                                t2.A = A1 ∧ t2.B = B1 ∧ t2.C = C1 ∧ t2.D = D1) :
  volume t1 = 2 * volume t2 := 
sorry

end NUMINAMATH_GPT_volume_relation_l1947_194758


namespace NUMINAMATH_GPT_rate_ratio_l1947_194786

theorem rate_ratio
  (rate_up : ℝ) (time_up : ℝ) (distance_up : ℝ)
  (distance_down : ℝ) (time_down : ℝ) :
  rate_up = 4 → time_up = 2 → distance_up = rate_up * time_up →
  distance_down = 12 → time_down = 2 →
  (distance_down / time_down) / rate_up = 3 / 2 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_rate_ratio_l1947_194786


namespace NUMINAMATH_GPT_find_percentage_loss_l1947_194754

theorem find_percentage_loss 
  (P : ℝ)
  (initial_marbles remaining_marbles : ℝ)
  (h1 : initial_marbles = 100)
  (h2 : remaining_marbles = 20)
  (h3 : (initial_marbles - initial_marbles * P / 100) / 2 = remaining_marbles) :
  P = 60 :=
by
  sorry

end NUMINAMATH_GPT_find_percentage_loss_l1947_194754


namespace NUMINAMATH_GPT_conditions_neither_necessary_nor_sufficient_l1947_194765

theorem conditions_neither_necessary_nor_sufficient :
  (¬(0 < x ∧ x < 2) ↔ (¬(-1 / 2 < x ∨ x < 1)) ∨ (¬(-1 / 2 < x ∧ x < 1))) :=
by sorry

end NUMINAMATH_GPT_conditions_neither_necessary_nor_sufficient_l1947_194765


namespace NUMINAMATH_GPT_smallest_k_multiple_of_180_l1947_194736

def sum_of_squares (k : ℕ) : ℕ :=
  (k * (k + 1) * (2 * k + 1)) / 6

def divisible_by_180 (n : ℕ) : Prop :=
  n % 180 = 0

theorem smallest_k_multiple_of_180 :
  ∃ k : ℕ, k > 0 ∧ divisible_by_180 (sum_of_squares k) ∧ ∀ m : ℕ, m > 0 ∧ divisible_by_180 (sum_of_squares m) → k ≤ m :=
sorry

end NUMINAMATH_GPT_smallest_k_multiple_of_180_l1947_194736


namespace NUMINAMATH_GPT_find_number_l1947_194794

theorem find_number (x : ℝ) (h : 42 - 3 * x = 12) : x = 10 := 
by 
  sorry

end NUMINAMATH_GPT_find_number_l1947_194794


namespace NUMINAMATH_GPT_fruit_weight_sister_and_dad_l1947_194779

-- Defining the problem statement and conditions
variable (strawberries_m blueberries_m raspberries_m : ℝ)
variable (strawberries_d blueberries_d raspberries_d : ℝ)
variable (strawberries_s blueberries_s raspberries_s : ℝ)
variable (total_weight : ℝ)

-- Given initial conditions
def conditions : Prop :=
  strawberries_m = 5 ∧
  blueberries_m = 3 ∧
  raspberries_m = 6 ∧
  strawberries_d = 2 * strawberries_m ∧
  blueberries_d = 2 * blueberries_m ∧
  raspberries_d = 2 * raspberries_m ∧
  strawberries_s = strawberries_m / 2 ∧
  blueberries_s = blueberries_m / 2 ∧
  raspberries_s = raspberries_m / 2 ∧
  total_weight = (strawberries_m + blueberries_m + raspberries_m) + 
                 (strawberries_d + blueberries_d + raspberries_d) + 
                 (strawberries_s + blueberries_s + raspberries_s)

-- Defining the property to prove
theorem fruit_weight_sister_and_dad :
  conditions strawberries_m blueberries_m raspberries_m strawberries_d blueberries_d raspberries_d strawberries_s blueberries_s raspberries_s total_weight →
  (strawberries_d + blueberries_d + raspberries_d) +
  (strawberries_s + blueberries_s + raspberries_s) = 35 := by
  sorry

end NUMINAMATH_GPT_fruit_weight_sister_and_dad_l1947_194779


namespace NUMINAMATH_GPT_find_x_l1947_194704

theorem find_x (a b x : ℝ) (h : ∀ a b, a * b = a + 2 * b) (H : 3 * (4 * x) = 6) : x = -5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1947_194704


namespace NUMINAMATH_GPT_P_plus_Q_is_26_l1947_194768

theorem P_plus_Q_is_26 (P Q : ℝ) (h : ∀ x : ℝ, x ≠ 3 → (P / (x - 3) + Q * (x + 2) = (-2 * x^2 + 8 * x + 34) / (x - 3))) : 
  P + Q = 26 :=
sorry

end NUMINAMATH_GPT_P_plus_Q_is_26_l1947_194768


namespace NUMINAMATH_GPT_top_leftmost_rectangle_is_E_l1947_194793

def rectangle (w x y z : ℕ) : Prop := true

-- Define the rectangles according to the given conditions
def rectangle_A : Prop := rectangle 4 1 6 9
def rectangle_B : Prop := rectangle 1 0 3 6
def rectangle_C : Prop := rectangle 3 8 5 2
def rectangle_D : Prop := rectangle 7 5 4 8
def rectangle_E : Prop := rectangle 9 2 7 0

-- Prove that the top leftmost rectangle is E
theorem top_leftmost_rectangle_is_E : rectangle_E → True :=
by
  sorry

end NUMINAMATH_GPT_top_leftmost_rectangle_is_E_l1947_194793


namespace NUMINAMATH_GPT_range_a_for_false_proposition_l1947_194740

theorem range_a_for_false_proposition :
  {a : ℝ | ¬ ∃ x : ℝ, x^2 + (1 - a) * x < 0} = {1} :=
sorry

end NUMINAMATH_GPT_range_a_for_false_proposition_l1947_194740


namespace NUMINAMATH_GPT_tangent_line_at_e_l1947_194700

noncomputable def f (x : ℝ) := x * Real.log x

theorem tangent_line_at_e : ∀ x y : ℝ, (x = Real.exp 1) → (y = f x) → (y = 2 * x - Real.exp 1) :=
by
  intros x y hx hy
  sorry

end NUMINAMATH_GPT_tangent_line_at_e_l1947_194700


namespace NUMINAMATH_GPT_ratio_of_arithmetic_sums_l1947_194772

theorem ratio_of_arithmetic_sums : 
  let a1 := 4
  let d1 := 4
  let l1 := 48
  let a2 := 2
  let d2 := 3
  let l2 := 35
  let n1 := (l1 - a1) / d1 + 1
  let n2 := (l2 - a2) / d2 + 1
  let S1 := n1 * (a1 + l1) / 2
  let S2 := n2 * (a2 + l2) / 2
  let ratio := S1 / S2
  ratio = 52 / 37 := by sorry

end NUMINAMATH_GPT_ratio_of_arithmetic_sums_l1947_194772


namespace NUMINAMATH_GPT_two_points_same_color_at_distance_one_l1947_194715

theorem two_points_same_color_at_distance_one (color : ℝ × ℝ → ℕ) (h : ∀p : ℝ × ℝ, color p < 3) :
  ∃ (p q : ℝ × ℝ), dist p q = 1 ∧ color p = color q :=
sorry

end NUMINAMATH_GPT_two_points_same_color_at_distance_one_l1947_194715


namespace NUMINAMATH_GPT_composite_function_properties_l1947_194717

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem composite_function_properties
  (a b c : ℝ)
  (h_a_nonzero : a ≠ 0)
  (h_no_real_roots : ∀ x : ℝ, f a b c x ≠ x) :
  (∀ x : ℝ, f a b c (f a b c x) ≠ x) ∧
  (a > 0 → ∀ x : ℝ, f a b c (f a b c x) > x) :=
by sorry

end NUMINAMATH_GPT_composite_function_properties_l1947_194717


namespace NUMINAMATH_GPT_present_age_of_son_l1947_194762

variable (S M : ℕ)

-- Conditions
def condition1 : Prop := M = S + 32
def condition2 : Prop := M + 2 = 2 * (S + 2)

-- Theorem stating the required proof
theorem present_age_of_son : condition1 S M ∧ condition2 S M → S = 30 := by
  sorry

end NUMINAMATH_GPT_present_age_of_son_l1947_194762


namespace NUMINAMATH_GPT_charcoal_drawings_correct_l1947_194738

-- Define the constants based on the problem conditions
def total_drawings : ℕ := 120
def colored_pencils : ℕ := 35
def blending_markers : ℕ := 22
def pastels : ℕ := 15
def watercolors : ℕ := 12

-- Calculate the total number of charcoal drawings
def charcoal_drawings : ℕ := total_drawings - (colored_pencils + blending_markers + pastels + watercolors)

-- The theorem we want to prove is that the number of charcoal drawings is 36
theorem charcoal_drawings_correct : charcoal_drawings = 36 :=
by
  -- The proof goes here (we skip it with 'sorry')
  sorry

end NUMINAMATH_GPT_charcoal_drawings_correct_l1947_194738


namespace NUMINAMATH_GPT_solve_pond_fish_problem_l1947_194796

def pond_fish_problem 
  (tagged_fish : ℕ)
  (second_catch : ℕ)
  (tagged_in_second_catch : ℕ)
  (total_fish : ℕ) : Prop :=
  (tagged_in_second_catch : ℝ) / second_catch = (tagged_fish : ℝ) / total_fish →
  total_fish = 1750

theorem solve_pond_fish_problem : 
  pond_fish_problem 70 50 2 1750 :=
by
  sorry

end NUMINAMATH_GPT_solve_pond_fish_problem_l1947_194796


namespace NUMINAMATH_GPT_cards_per_layer_l1947_194728

theorem cards_per_layer (total_decks : ℕ) (cards_per_deck : ℕ) (layers : ℕ) (h_decks : total_decks = 16) (h_cards_per_deck : cards_per_deck = 52) (h_layers : layers = 32) :
  total_decks * cards_per_deck / layers = 26 :=
by {
  -- To skip the proof
  sorry
}

end NUMINAMATH_GPT_cards_per_layer_l1947_194728


namespace NUMINAMATH_GPT_total_sales_l1947_194767

theorem total_sales (S : ℝ) (remitted : ℝ) : 
  (∀ S, remitted = S - (0.05 * 10000 + 0.04 * (S - 10000)) → remitted = 31100) → S = 32500 :=
by
  sorry

end NUMINAMATH_GPT_total_sales_l1947_194767


namespace NUMINAMATH_GPT_roots_of_unity_l1947_194792

noncomputable def is_root_of_unity (z : ℂ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ z^n = 1

noncomputable def is_cube_root_of_unity (z : ℂ) : Prop :=
  z^3 = 1

theorem roots_of_unity (x y : ℂ) (hx : is_root_of_unity x) (hy : is_root_of_unity y) (hxy : x ≠ y) :
  is_root_of_unity (x + y) ↔ is_cube_root_of_unity (y / x) :=
sorry

end NUMINAMATH_GPT_roots_of_unity_l1947_194792


namespace NUMINAMATH_GPT_find_ac_bc_val_l1947_194757

variable (a b c d : ℚ)
variable (h_neq : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
variable (h1 : (a + c) * (a + d) = 1)
variable (h2 : (b + c) * (b + d) = 1)

theorem find_ac_bc_val : (a + c) * (b + c) = -1 := 
by 
  sorry

end NUMINAMATH_GPT_find_ac_bc_val_l1947_194757


namespace NUMINAMATH_GPT_min_value_expression_l1947_194771

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2*x + y = 1) : 
  ∃ (xy_min : ℝ), xy_min = 9 ∧ (∀ (x y : ℝ), 0 < x ∧ 0 < y ∧ 2*x + y = 1 → (x + 2*y)/(x*y) ≥ xy_min) :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1947_194771


namespace NUMINAMATH_GPT_total_cookies_l1947_194703

-- Conditions
def Paul_cookies : ℕ := 45
def Paula_cookies : ℕ := Paul_cookies - 3

-- Question and Answer
theorem total_cookies : Paul_cookies + Paula_cookies = 87 := by
  sorry

end NUMINAMATH_GPT_total_cookies_l1947_194703


namespace NUMINAMATH_GPT_seq_geom_prog_l1947_194753

theorem seq_geom_prog (a : ℕ → ℝ) (b : ℝ) (h_pos_b : 0 < b)
  (h_pos_a : ∀ n, 0 < a n)
  (h_recurrence : ∀ n, a (n + 2) = (b + 1) * a n * a (n + 1)) :
  (∃ r, ∀ n, a (n + 1) = r * a n) ↔ a 0 = a 1 :=
sorry

end NUMINAMATH_GPT_seq_geom_prog_l1947_194753


namespace NUMINAMATH_GPT_maximum_area_of_triangle_l1947_194764

theorem maximum_area_of_triangle (A B C : ℝ) (a b c : ℝ) (hC : C = π / 6) (hSum : a + b = 12) :
  ∃ (S : ℝ), S = 9 ∧ ∀ S', S' ≤ S := 
sorry

end NUMINAMATH_GPT_maximum_area_of_triangle_l1947_194764


namespace NUMINAMATH_GPT_find_two_digit_number_l1947_194720

theorem find_two_digit_number : 
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ (b = 0 ∨ b = 5) ∧ (10 * a + b = 5 * (a + b)) ∧ (10 * a + b = 45) :=
by
  sorry

end NUMINAMATH_GPT_find_two_digit_number_l1947_194720


namespace NUMINAMATH_GPT_vampire_daily_blood_suction_l1947_194701

-- Conditions from the problem
def vampire_bl_need_per_week : ℕ := 7  -- gallons of blood per week
def blood_per_person_in_pints : ℕ := 2  -- pints of blood per person
def pints_per_gallon : ℕ := 8            -- pints in 1 gallon

-- Theorem statement to prove
theorem vampire_daily_blood_suction :
  let daily_requirement_in_gallons : ℕ := vampire_bl_need_per_week / 7   -- gallons per day
  let daily_requirement_in_pints : ℕ := daily_requirement_in_gallons * pints_per_gallon
  let num_people_needed_per_day : ℕ := daily_requirement_in_pints / blood_per_person_in_pints
  num_people_needed_per_day = 4 :=
by
  sorry

end NUMINAMATH_GPT_vampire_daily_blood_suction_l1947_194701


namespace NUMINAMATH_GPT_solve_Diamond_l1947_194760

theorem solve_Diamond :
  ∀ (Diamond : ℕ), (Diamond * 7 + 4 = Diamond * 8 + 1) → Diamond = 3 :=
by
  intros Diamond h
  sorry

end NUMINAMATH_GPT_solve_Diamond_l1947_194760


namespace NUMINAMATH_GPT_cheryl_gave_mms_to_sister_l1947_194791

-- Definitions for given conditions in the problem
def ate_after_lunch : ℕ := 7
def ate_after_dinner : ℕ := 5
def initial_mms : ℕ := 25

-- The statement to be proved
theorem cheryl_gave_mms_to_sister : (initial_mms - (ate_after_lunch + ate_after_dinner)) = 13 := by
  sorry

end NUMINAMATH_GPT_cheryl_gave_mms_to_sister_l1947_194791


namespace NUMINAMATH_GPT_train_length_l1947_194781

theorem train_length
  (speed_kmph : ℕ)
  (platform_length : ℕ)
  (time_seconds : ℕ)
  (speed_m_per_s : speed_kmph * 1000 / 3600 = 20)
  (distance_covered : 20 * time_seconds = 520)
  (platform_eq : platform_length = 280)
  (time_eq : time_seconds = 26) :
  ∃ L : ℕ, L = 240 := by
  sorry

end NUMINAMATH_GPT_train_length_l1947_194781


namespace NUMINAMATH_GPT_price_of_brand_X_pen_l1947_194724

variable (P : ℝ)

theorem price_of_brand_X_pen :
  (∀ (n : ℕ), n = 12 → 6 * P + 6 * 2.20 = 42 - 13.20) →
  P = 4.80 :=
by
  intro h₁
  have h₂ := h₁ 12 rfl
  sorry

end NUMINAMATH_GPT_price_of_brand_X_pen_l1947_194724


namespace NUMINAMATH_GPT_solve_for_x_l1947_194710

theorem solve_for_x (x : ℝ) (h : x + 2 = 7) : x = 5 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1947_194710


namespace NUMINAMATH_GPT_line_equation_through_points_and_area_l1947_194770

variable (a b S : ℝ)
variable (h_b_gt_a : b > a)
variable (h_area : S = 1/2 * (b - a) * (2 * S / (b - a)))

theorem line_equation_through_points_and_area :
  0 = -2 * S * x + (b - a)^2 * y + 2 * S * a - 2 * S * b := sorry

end NUMINAMATH_GPT_line_equation_through_points_and_area_l1947_194770


namespace NUMINAMATH_GPT_sum_of_roots_l1947_194731

theorem sum_of_roots (f : ℝ → ℝ) :
  (∀ x : ℝ, f (3 + x) = f (3 - x)) →
  (∃ (S : Finset ℝ), S.card = 6 ∧ ∀ x ∈ S, f x = 0) →
  (∃ (S : Finset ℝ), S.sum id = 18) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l1947_194731


namespace NUMINAMATH_GPT_gcd_gx_x_l1947_194769

theorem gcd_gx_x (x : ℤ) (hx : 34560 ∣ x) :
  Int.gcd ((3 * x + 4) * (8 * x + 5) * (15 * x + 11) * (x + 17)) x = 20 := 
by
  sorry

end NUMINAMATH_GPT_gcd_gx_x_l1947_194769


namespace NUMINAMATH_GPT_largest_number_is_y_l1947_194751

def x := 8.1235
def y := 8.12355555555555 -- 8.123\overline{5}
def z := 8.12345454545454 -- 8.123\overline{45}
def w := 8.12345345345345 -- 8.12\overline{345}
def v := 8.12345234523452 -- 8.1\overline{2345}

theorem largest_number_is_y : y > x ∧ y > z ∧ y > w ∧ y > v :=
by
-- Proof steps would go here.
sorry

end NUMINAMATH_GPT_largest_number_is_y_l1947_194751


namespace NUMINAMATH_GPT_rate_of_interest_l1947_194798

theorem rate_of_interest (P R : ℝ) :
  (2 * P * R) / 100 = 320 ∧
  P * ((1 + R / 100) ^ 2 - 1) = 340 →
  R = 12.5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_rate_of_interest_l1947_194798


namespace NUMINAMATH_GPT_isosceles_triangle_angle_between_vectors_l1947_194707

theorem isosceles_triangle_angle_between_vectors 
  (α β γ : ℝ) 
  (h1: α + β + γ = 180)
  (h2: α = 120) 
  (h3: β = γ):
  180 - β = 150 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_angle_between_vectors_l1947_194707


namespace NUMINAMATH_GPT_total_volume_stacked_dice_l1947_194714

def die_volume (width length height : ℕ) : ℕ := 
  width * length * height

def total_dice (horizontal vertical layers : ℕ) : ℕ := 
  horizontal * vertical * layers

theorem total_volume_stacked_dice :
  let width := 1
  let length := 1
  let height := 1
  let horizontal := 7
  let vertical := 5
  let layers := 3
  let single_die_volume := die_volume width length height
  let num_dice := total_dice horizontal vertical layers
  single_die_volume * num_dice = 105 :=
by
  sorry  -- proof to be provided

end NUMINAMATH_GPT_total_volume_stacked_dice_l1947_194714


namespace NUMINAMATH_GPT_jimmy_bread_packs_needed_l1947_194783

theorem jimmy_bread_packs_needed 
  (sandwiches : ℕ)
  (slices_per_sandwich : ℕ)
  (initial_bread_slices : ℕ)
  (slices_per_pack : ℕ)
  (H1 : sandwiches = 8)
  (H2 : slices_per_sandwich = 2)
  (H3 : initial_bread_slices = 0)
  (H4 : slices_per_pack = 4) : 
  (8 * 2) / 4 = 4 := 
sorry

end NUMINAMATH_GPT_jimmy_bread_packs_needed_l1947_194783


namespace NUMINAMATH_GPT_chess_tournament_total_games_l1947_194776

theorem chess_tournament_total_games (n : ℕ) (h : n = 10) : (n * (n - 1)) / 2 = 45 := by
  sorry

end NUMINAMATH_GPT_chess_tournament_total_games_l1947_194776


namespace NUMINAMATH_GPT_max_value_of_f_l1947_194705

noncomputable def f (theta x : ℝ) : ℝ :=
  (Real.cos theta)^2 - 2 * x * Real.cos theta - 1

noncomputable def M (x : ℝ) : ℝ :=
  if 0 <= x then 
    2 * x
  else 
    -2 * x

theorem max_value_of_f {x : ℝ} : 
  ∃ theta : ℝ, Real.cos theta ∈ [-1, 1] ∧ f theta x = M x :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l1947_194705


namespace NUMINAMATH_GPT_common_chord_is_linear_l1947_194777

-- Defining the equations of two intersecting circles
noncomputable def circle1 : ℝ → ℝ → ℝ := sorry
noncomputable def circle2 : ℝ → ℝ → ℝ := sorry

-- Defining a method to eliminate quadratic terms
noncomputable def eliminate_quadratic_terms (eq1 eq2 : ℝ → ℝ → ℝ) : ℝ → ℝ → ℝ := sorry

-- Defining the linear equation representing the common chord
noncomputable def common_chord (eq1 eq2 : ℝ → ℝ → ℝ) : ℝ → ℝ → ℝ := sorry

-- Statement of the problem
theorem common_chord_is_linear (circle1 circle2 : ℝ → ℝ → ℝ) :
  common_chord circle1 circle2 = eliminate_quadratic_terms circle1 circle2 := sorry

end NUMINAMATH_GPT_common_chord_is_linear_l1947_194777


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1947_194766

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a = 1 → (∀ x y : ℝ, ax + 2 * y - 1 = 0 → ∃ x' y' : ℝ, x' + (a + 1) * y' + 4 = 0)) ∧
  ¬ (a = 1 → (∀ x y : ℝ, ax + 2 * y - 1 = 0 ↔ ∃ x' y' : ℝ, x' + (a + 1) * y' + 4 = 0)) := 
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1947_194766


namespace NUMINAMATH_GPT_angles_sum_correct_l1947_194702

-- Definitions from the problem conditions
def identicalSquares (n : Nat) := n = 13

variable (α β γ δ ε ζ η θ : ℝ) -- Angles of interest

def anglesSum :=
  (α + β + γ + δ) + (ε + ζ + η + θ)

-- Lean 4 statement
theorem angles_sum_correct
  (h₁ : identicalSquares 13)
  (h₂ : α = 90) (h₃ : β = 90) (h₄ : γ = 90) (h₅ : δ = 90)
  (h₆ : ε = 90) (h₇ : ζ = 90) (h₈ : η = 45) (h₉ : θ = 45) :
  anglesSum α β γ δ ε ζ η θ = 405 :=
by
  simp [anglesSum]
  sorry

end NUMINAMATH_GPT_angles_sum_correct_l1947_194702


namespace NUMINAMATH_GPT_simplify_fraction_l1947_194778

theorem simplify_fraction (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (1 / (2 * a * b) + b / (4 * a) = (2 + b^2) / (4 * a * b)) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1947_194778


namespace NUMINAMATH_GPT_foreign_stamps_count_l1947_194785

-- Define the conditions
variables (total_stamps : ℕ) (more_than_10_years_old : ℕ) (both_foreign_and_old : ℕ) (neither_foreign_nor_old : ℕ)

theorem foreign_stamps_count 
  (h1 : total_stamps = 200)
  (h2 : more_than_10_years_old = 60)
  (h3 : both_foreign_and_old = 20)
  (h4 : neither_foreign_nor_old = 70) : 
  ∃ (foreign_stamps : ℕ), foreign_stamps = 90 :=
by
  -- let foreign_stamps be the variable representing the number of foreign stamps
  let foreign_stamps := total_stamps - neither_foreign_nor_old - more_than_10_years_old + both_foreign_and_old
  use foreign_stamps
  -- the proof will develop here to show that foreign_stamps = 90
  sorry

end NUMINAMATH_GPT_foreign_stamps_count_l1947_194785


namespace NUMINAMATH_GPT_largest_consecutive_sum_55_l1947_194746

theorem largest_consecutive_sum_55 : 
  ∃ k n : ℕ, k > 0 ∧ (∃ n : ℕ, 55 = k * n + (k * (k - 1)) / 2 ∧ k = 5) :=
sorry

end NUMINAMATH_GPT_largest_consecutive_sum_55_l1947_194746


namespace NUMINAMATH_GPT_weight_of_lightest_weight_l1947_194719

theorem weight_of_lightest_weight (x : ℕ) (y : ℕ) (h1 : 0 < y ∧ y < 9)
  (h2 : (10 : ℕ) * x + 45 - (x + y) = 2022) : x = 220 := by
  sorry

end NUMINAMATH_GPT_weight_of_lightest_weight_l1947_194719


namespace NUMINAMATH_GPT_remainder_3a_plus_b_l1947_194726

theorem remainder_3a_plus_b (p q : ℤ) (a b : ℤ)
  (h1 : a = 98 * p + 92)
  (h2 : b = 147 * q + 135) :
  ((3 * a + b) % 49) = 19 := by
sorry

end NUMINAMATH_GPT_remainder_3a_plus_b_l1947_194726


namespace NUMINAMATH_GPT_difference_in_girls_and_boys_l1947_194780

-- Given conditions as definitions
def boys : ℕ := 40
def ratio_boys_to_girls (b g : ℕ) : Prop := 5 * g = 13 * b

-- Statement of the problem
theorem difference_in_girls_and_boys (g : ℕ) (h : ratio_boys_to_girls boys g) : g - boys = 64 :=
by
  sorry

end NUMINAMATH_GPT_difference_in_girls_and_boys_l1947_194780


namespace NUMINAMATH_GPT_complementary_angle_ratio_l1947_194773

theorem complementary_angle_ratio (x : ℝ) (h1 : 4 * x + x = 90) : x = 18 :=
by {
  sorry
}

end NUMINAMATH_GPT_complementary_angle_ratio_l1947_194773


namespace NUMINAMATH_GPT_part1_solution_set_part2_range_a_l1947_194712

def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : 
  x ≤ 3/2 ∨ x ≥ 11/2 :=
sorry

theorem part2_range_a (a : ℝ) (h : ∃ x, f x a ≥ 4) : 
  a ≤ -1 ∨ a ≥ 3 :=
sorry

end NUMINAMATH_GPT_part1_solution_set_part2_range_a_l1947_194712


namespace NUMINAMATH_GPT_greatest_integer_leq_l1947_194722

theorem greatest_integer_leq (a b : ℝ) (ha : a = 5^150) (hb : b = 3^150) (c d : ℝ) (hc : c = 5^147) (hd : d = 3^147):
  ⌊ (a + b) / (c + d) ⌋ = 124 := 
sorry

end NUMINAMATH_GPT_greatest_integer_leq_l1947_194722


namespace NUMINAMATH_GPT_single_elimination_games_l1947_194787

theorem single_elimination_games (n : Nat) (h : n = 256) : n - 1 = 255 := by
  sorry

end NUMINAMATH_GPT_single_elimination_games_l1947_194787


namespace NUMINAMATH_GPT_tan_double_angle_l1947_194789

theorem tan_double_angle (α : ℝ) (h₁ : Real.sin α = 4/5) (h₂ : α ∈ Set.Ioc (π / 2) π) :
  Real.tan (2 * α) = 24 / 7 := 
  sorry

end NUMINAMATH_GPT_tan_double_angle_l1947_194789


namespace NUMINAMATH_GPT_arc_length_parametric_curve_l1947_194742

noncomputable def arcLength (x y : ℝ → ℝ) (t1 t2 : ℝ) : ℝ :=
  ∫ t in t1..t2, Real.sqrt ((deriv x t)^2 + (deriv y t)^2)

theorem arc_length_parametric_curve :
    (∫ t in (0 : ℝ)..(3 * Real.pi), 
        Real.sqrt ((deriv (fun t => (t ^ 2 - 2) * Real.sin t + 2 * t * Real.cos t) t) ^ 2 +
                   (deriv (fun t => (2 - t ^ 2) * Real.cos t + 2 * t * Real.sin t) t) ^ 2)) =
    9 * Real.pi ^ 3 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_arc_length_parametric_curve_l1947_194742


namespace NUMINAMATH_GPT_cookie_distribution_l1947_194763

theorem cookie_distribution (b m l : ℕ)
  (h1 : b + m + l = 30)
  (h2 : m = 2 * b)
  (h3 : l = b + m) :
  b = 5 ∧ m = 10 ∧ l = 15 := 
by 
  sorry

end NUMINAMATH_GPT_cookie_distribution_l1947_194763


namespace NUMINAMATH_GPT_charge_for_cat_l1947_194774

theorem charge_for_cat (D N_D N_C T C : ℝ) 
  (h1 : D = 60) (h2 : N_D = 20) (h3 : N_C = 60) (h4 : T = 3600)
  (h5 : 20 * D + 60 * C = T) :
  C = 40 := by
  sorry

end NUMINAMATH_GPT_charge_for_cat_l1947_194774


namespace NUMINAMATH_GPT_total_vehicles_correct_l1947_194797

def num_trucks : ℕ := 20
def num_tanks (num_trucks : ℕ) : ℕ := 5 * num_trucks
def total_vehicles (num_trucks : ℕ) (num_tanks : ℕ) : ℕ := num_trucks + num_tanks

theorem total_vehicles_correct : total_vehicles num_trucks (num_tanks num_trucks) = 120 := by
  sorry

end NUMINAMATH_GPT_total_vehicles_correct_l1947_194797
