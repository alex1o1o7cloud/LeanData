import Mathlib

namespace NUMINAMATH_GPT_cars_through_toll_booth_l2397_239789

noncomputable def total_cars_in_week (n_mon n_tue n_wed n_thu n_fri n_sat n_sun : ℕ) : ℕ :=
  n_mon + n_tue + n_wed + n_thu + n_fri + n_sat + n_sun 

theorem cars_through_toll_booth : 
  let n_mon : ℕ := 50
  let n_tue : ℕ := 50
  let n_wed : ℕ := 2 * n_mon
  let n_thu : ℕ := 2 * n_mon
  let n_fri : ℕ := 50
  let n_sat : ℕ := 50
  let n_sun : ℕ := 50
  total_cars_in_week n_mon n_tue n_wed n_thu n_fri n_sat n_sun = 450 := 
by 
  sorry

end NUMINAMATH_GPT_cars_through_toll_booth_l2397_239789


namespace NUMINAMATH_GPT_solve_for_c_l2397_239788

noncomputable def proof_problem (a b c : ℝ) : Prop :=
  (a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) →
  (6 * 15 * c = 1.5) →
  c = 7

theorem solve_for_c : proof_problem 6 15 7 :=
by sorry

end NUMINAMATH_GPT_solve_for_c_l2397_239788


namespace NUMINAMATH_GPT_Miss_Stevie_payment_l2397_239731

theorem Miss_Stevie_payment:
  let painting_hours := 8
  let painting_rate := 15
  let painting_earnings := painting_hours * painting_rate
  let mowing_hours := 6
  let mowing_rate := 10
  let mowing_earnings := mowing_hours * mowing_rate
  let plumbing_hours := 4
  let plumbing_rate := 18
  let plumbing_earnings := plumbing_hours * plumbing_rate
  let total_earnings := painting_earnings + mowing_earnings + plumbing_earnings
  let discount := 0.10 * total_earnings
  let amount_paid := total_earnings - discount
  amount_paid = 226.80 :=
by
  sorry

end NUMINAMATH_GPT_Miss_Stevie_payment_l2397_239731


namespace NUMINAMATH_GPT_monotone_f_iff_l2397_239783

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if h : x < 1 then a^x
  else x^2 + 4 / x + a * Real.log x

theorem monotone_f_iff (a : ℝ) :
  (∀ x₁ x₂, x₁ ≤ x₂ → f a x₁ ≤ f a x₂) ↔ 2 ≤ a ∧ a ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_monotone_f_iff_l2397_239783


namespace NUMINAMATH_GPT_area_of_cos_closed_figure_l2397_239739

theorem area_of_cos_closed_figure :
  ∫ x in (Real.pi / 2)..(3 * Real.pi / 2), Real.cos x = 2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_cos_closed_figure_l2397_239739


namespace NUMINAMATH_GPT_range_of_a_l2397_239742

-- Define the operation ⊗
def tensor (x y : ℝ) : ℝ := x * (1 - y)

-- State the main theorem
theorem range_of_a (a : ℝ) : (∀ x : ℝ, tensor (x - a) (x + a) < 1) → 
  (-((1 : ℝ) / 2) < a ∧ a < (3 : ℝ) / 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2397_239742


namespace NUMINAMATH_GPT_find_m_n_l2397_239761

theorem find_m_n (x : ℝ) (m n : ℝ) 
  (h : (2 * x - 5) * (x + m) = 2 * x^2 - 3 * x + n) :
  m = 1 ∧ n = -5 :=
by
  have h_expand : (2 * x - 5) * (x + m) = 2 * x^2 + (2 * m - 5) * x - 5 * m := by
    ring
  rw [h_expand] at h
  have coeff_eq1 : 2 * m - 5 = -3 := by sorry
  have coeff_eq2 : -5 * m = n := by sorry
  have m_sol : m = 1 := by
    linarith [coeff_eq1]
  have n_sol : n = -5 := by
    rw [m_sol] at coeff_eq2
    linarith
  exact ⟨m_sol, n_sol⟩

end NUMINAMATH_GPT_find_m_n_l2397_239761


namespace NUMINAMATH_GPT_grouping_schemes_count_l2397_239728

/-- Number of possible grouping schemes where each group consists
    of either 2 or 3 students and the total number of students is 25 is 4.-/
theorem grouping_schemes_count : ∃ (x y : ℕ), 2 * x + 3 * y = 25 ∧ 
  (x = 11 ∧ y = 1 ∨ x = 8 ∧ y = 3 ∨ x = 5 ∧ y = 5 ∨ x = 2 ∧ y = 7) :=
sorry

end NUMINAMATH_GPT_grouping_schemes_count_l2397_239728


namespace NUMINAMATH_GPT_linear_function_no_pass_quadrant_I_l2397_239748

theorem linear_function_no_pass_quadrant_I (x y : ℝ) (h : y = -2 * x - 1) : 
  ¬ (0 < x ∧ 0 < y) :=
by 
  sorry

end NUMINAMATH_GPT_linear_function_no_pass_quadrant_I_l2397_239748


namespace NUMINAMATH_GPT_percent_increase_combined_cost_l2397_239772

theorem percent_increase_combined_cost :
  let laptop_last_year := 500
  let tablet_last_year := 200
  let laptop_increase := 10 / 100
  let tablet_increase := 20 / 100
  let new_laptop_cost := laptop_last_year * (1 + laptop_increase)
  let new_tablet_cost := tablet_last_year * (1 + tablet_increase)
  let total_last_year := laptop_last_year + tablet_last_year
  let total_this_year := new_laptop_cost + new_tablet_cost
  let increase := total_this_year - total_last_year
  let percent_increase := (increase / total_last_year) * 100
  percent_increase = 13 :=
by
  sorry

end NUMINAMATH_GPT_percent_increase_combined_cost_l2397_239772


namespace NUMINAMATH_GPT_matrix_pow_minus_l2397_239755

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 4], ![0, 2]]

theorem matrix_pow_minus : B ^ 20 - 3 * (B ^ 19) = ![![0, 4 * (2 ^ 19)], ![0, -(2 ^ 19)]] :=
by
  sorry

end NUMINAMATH_GPT_matrix_pow_minus_l2397_239755


namespace NUMINAMATH_GPT_bubble_sort_probability_r10_r25_l2397_239770

theorem bubble_sort_probability_r10_r25 (n : ℕ) (r : ℕ → ℕ) :
  n = 50 ∧ (∀ i, 1 ≤ i ∧ i ≤ 50 → r i ≠ r (i + 1)) ∧ (∀ i j, i ≠ j → r i ≠ r j) →
  let p := 1
  let q := 650
  p + q = 651 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_bubble_sort_probability_r10_r25_l2397_239770


namespace NUMINAMATH_GPT_range_of_a_l2397_239786

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x + 1 > 2 * x - 2) → (x < a)) → (a ≥ 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2397_239786


namespace NUMINAMATH_GPT_symmetric_point_x_axis_l2397_239730

theorem symmetric_point_x_axis (x y : ℝ) (p : Prod ℝ ℝ) (hx : p = (x, y)) :
  (x, -y) = (1, -2) ↔ (x, y) = (1, 2) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_x_axis_l2397_239730


namespace NUMINAMATH_GPT_total_time_spent_l2397_239745

-- Define the conditions
def number_of_chairs : ℕ := 4
def number_of_tables : ℕ := 2
def time_per_piece : ℕ := 8

-- Prove that the total time spent is 48 minutes
theorem total_time_spent : (number_of_chairs + number_of_tables) * time_per_piece = 48 :=
by
  sorry

end NUMINAMATH_GPT_total_time_spent_l2397_239745


namespace NUMINAMATH_GPT_ordered_triples_lcm_l2397_239724

def lcm_equal (a b n : ℕ) : Prop :=
  a * b / (Nat.gcd a b) = n

theorem ordered_triples_lcm :
  ∀ (x y z : ℕ), 0 < x → 0 < y → 0 < z → 
  lcm_equal x y 48 → lcm_equal x z 900 → lcm_equal y z 180 →
  false :=
by sorry

end NUMINAMATH_GPT_ordered_triples_lcm_l2397_239724


namespace NUMINAMATH_GPT_function_properties_l2397_239790

noncomputable def f (x : ℝ) : ℝ := Real.sin ((13 * Real.pi / 2) - x)

theorem function_properties :
  (∀ x : ℝ, f x = Real.cos x) ∧
  (∀ x : ℝ, f (-x) = f x) ∧
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) ∧
  (forall t: ℝ, (∀ x : ℝ, f (x + t) = f x) → (t = 2 * Real.pi ∨ t = -2 * Real.pi)) :=
by
  sorry

end NUMINAMATH_GPT_function_properties_l2397_239790


namespace NUMINAMATH_GPT_bread_cost_is_30_l2397_239793

variable (cost_sandwich : ℝ)
variable (cost_ham : ℝ)
variable (cost_cheese : ℝ)

def cost_bread (cost_sandwich cost_ham cost_cheese : ℝ) : ℝ :=
  cost_sandwich - cost_ham - cost_cheese

theorem bread_cost_is_30 (H1 : cost_sandwich = 0.90)
  (H2 : cost_ham = 0.25)
  (H3 : cost_cheese = 0.35) :
  cost_bread cost_sandwich cost_ham cost_cheese = 0.30 :=
by
  rw [H1, H2, H3]
  simp [cost_bread]
  sorry

end NUMINAMATH_GPT_bread_cost_is_30_l2397_239793


namespace NUMINAMATH_GPT_triangle_ABC_angles_l2397_239799

theorem triangle_ABC_angles :
  ∃ (θ φ ω : ℝ), θ = 36 ∧ φ = 72 ∧ ω = 72 ∧
  (ω + φ + θ = 180) ∧
  (2 * ω + θ = 180) ∧
  (φ = 2 * θ) :=
by
  sorry

end NUMINAMATH_GPT_triangle_ABC_angles_l2397_239799


namespace NUMINAMATH_GPT_seconds_in_3_hours_25_minutes_l2397_239774

theorem seconds_in_3_hours_25_minutes:
  let hours := 3
  let minutesInAnHour := 60
  let additionalMinutes := 25
  let secondsInAMinute := 60
  (hours * minutesInAnHour + additionalMinutes) * secondsInAMinute = 12300 := 
by
  sorry

end NUMINAMATH_GPT_seconds_in_3_hours_25_minutes_l2397_239774


namespace NUMINAMATH_GPT_range_of_m_l2397_239759

theorem range_of_m (a b c m : ℝ) (h1 : a > b) (h2 : b > c) 
  (h3 : (1 / (a - b)) + (1 / (b - c)) ≥ m / (a - c)) :
  m ≤ 4 :=
sorry

end NUMINAMATH_GPT_range_of_m_l2397_239759


namespace NUMINAMATH_GPT_oranges_to_friend_is_two_l2397_239741

-- Definitions based on the conditions.

def initial_oranges : ℕ := 12

def oranges_to_brother (n : ℕ) : ℕ := n / 3

def remainder_after_brother (n : ℕ) : ℕ := n - oranges_to_brother n

def oranges_to_friend (n : ℕ) : ℕ := remainder_after_brother n / 4

-- Theorem stating the problem to be proven.
theorem oranges_to_friend_is_two : oranges_to_friend initial_oranges = 2 :=
sorry

end NUMINAMATH_GPT_oranges_to_friend_is_two_l2397_239741


namespace NUMINAMATH_GPT_moles_of_H2O_formed_l2397_239700

-- Define the initial conditions
def molesNaOH : ℕ := 2
def molesHCl : ℕ := 2

-- Balanced chemical equation behavior definition
def reaction (x y : ℕ) : ℕ := min x y

-- Statement of the problem to prove
theorem moles_of_H2O_formed :
  reaction molesNaOH molesHCl = 2 := by
  sorry

end NUMINAMATH_GPT_moles_of_H2O_formed_l2397_239700


namespace NUMINAMATH_GPT_extreme_point_properties_l2397_239703

noncomputable def f (x a : ℝ) : ℝ := x * (Real.log x - 2 * a * x)

theorem extreme_point_properties (a x₁ x₂ : ℝ) (h₁ : 0 < a) (h₂ : a < 1 / 4) 
  (h₃ : f a x₁ = 0) (h₄ : f a x₂ = 0) (h₅ : x₁ < x₂) :
  f x₁ a < 0 ∧ f x₂ a > (-1 / 2) := 
sorry

end NUMINAMATH_GPT_extreme_point_properties_l2397_239703


namespace NUMINAMATH_GPT_perp_vector_k_l2397_239758

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem perp_vector_k :
  ∀ k : ℝ, dot_product (1, 2) (-2, k) = 0 → k = 1 :=
by
  intro k h₀
  sorry

end NUMINAMATH_GPT_perp_vector_k_l2397_239758


namespace NUMINAMATH_GPT_books_borrowed_in_a_week_l2397_239792

theorem books_borrowed_in_a_week 
  (daily_avg : ℕ)
  (friday_increase_pct : ℕ)
  (days_open : ℕ)
  (friday_books : ℕ)
  (total_books_week : ℕ)
  (h1 : daily_avg = 40)
  (h2 : friday_increase_pct = 40)
  (h3 : days_open = 5)
  (h4 : friday_books = daily_avg + (daily_avg * friday_increase_pct / 100))
  (h5 : total_books_week = (days_open - 1) * daily_avg + friday_books) :
  total_books_week = 216 :=
by {
  sorry
}

end NUMINAMATH_GPT_books_borrowed_in_a_week_l2397_239792


namespace NUMINAMATH_GPT_algebraic_expression_value_l2397_239796

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y + 3 = 0) : 1 - 2 * x + 4 * y = 7 := 
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l2397_239796


namespace NUMINAMATH_GPT_line_plane_relationship_l2397_239767

variable {ℓ α : Type}
variables (is_line : is_line ℓ) (is_plane : is_plane α) (not_parallel : ¬ parallel ℓ α)

theorem line_plane_relationship (ℓ : Type) (α : Type) [is_line ℓ] [is_plane α] (not_parallel : ¬ parallel ℓ α) : 
  (intersect ℓ α) ∨ (subset ℓ α) :=
sorry

end NUMINAMATH_GPT_line_plane_relationship_l2397_239767


namespace NUMINAMATH_GPT_gasoline_price_increase_l2397_239712

theorem gasoline_price_increase (high low : ℝ) (high_eq : high = 24) (low_eq : low = 18) : 
  ((high - low) / low) * 100 = 33.33 := 
  sorry

end NUMINAMATH_GPT_gasoline_price_increase_l2397_239712


namespace NUMINAMATH_GPT_largest_divisor_of_n_squared_sub_n_squared_l2397_239723

theorem largest_divisor_of_n_squared_sub_n_squared (n : ℤ) : 6 ∣ (n^4 - n^2) :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_n_squared_sub_n_squared_l2397_239723


namespace NUMINAMATH_GPT_original_equation_l2397_239715

theorem original_equation : 9^2 - 8^2 = 17 := by
  sorry

end NUMINAMATH_GPT_original_equation_l2397_239715


namespace NUMINAMATH_GPT_last_three_digits_of_7_exp_1987_l2397_239791

theorem last_three_digits_of_7_exp_1987 : (7 ^ 1987) % 1000 = 543 := by
  sorry

end NUMINAMATH_GPT_last_three_digits_of_7_exp_1987_l2397_239791


namespace NUMINAMATH_GPT_white_balls_count_l2397_239753

theorem white_balls_count (w : ℕ) (h : (w / 15) * ((w - 1) / 14) = (1 : ℚ) / 21) : w = 5 := by
  sorry

end NUMINAMATH_GPT_white_balls_count_l2397_239753


namespace NUMINAMATH_GPT_saree_blue_stripes_l2397_239798

theorem saree_blue_stripes :
  ∀ (brown_stripes gold_stripes blue_stripes : ℕ),
    brown_stripes = 4 →
    gold_stripes = 3 * brown_stripes →
    blue_stripes = 5 * gold_stripes →
    blue_stripes = 60 :=
by
  intros brown_stripes gold_stripes blue_stripes h_brown h_gold h_blue
  sorry

end NUMINAMATH_GPT_saree_blue_stripes_l2397_239798


namespace NUMINAMATH_GPT_vandermonde_identity_combinatorial_identity_l2397_239779

open Nat

-- Problem 1: Vandermonde Identity
theorem vandermonde_identity (m n k : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 0 < k) (h4 : m + n ≥ k) :
  (Finset.range (k + 1)).sum (λ i => Nat.choose m i * Nat.choose n (k - i)) = Nat.choose (m + n) k :=
sorry

-- Problem 2:
theorem combinatorial_identity (p q n : ℕ) (h1 : 0 < p) (h2 : 0 < q) (h3 : 0 < n) :
  (Finset.range (p + 1)).sum (λ k => Nat.choose p k * Nat.choose q k * Nat.choose (n + k) (p + q)) =
  Nat.choose n p * Nat.choose n q :=
sorry

end NUMINAMATH_GPT_vandermonde_identity_combinatorial_identity_l2397_239779


namespace NUMINAMATH_GPT_sebastian_total_payment_l2397_239733

theorem sebastian_total_payment 
  (cost_per_ticket : ℕ) (number_of_tickets : ℕ) (service_fee : ℕ) (total_paid : ℕ)
  (h1 : cost_per_ticket = 44)
  (h2 : number_of_tickets = 3)
  (h3 : service_fee = 18)
  (h4 : total_paid = (number_of_tickets * cost_per_ticket) + service_fee) :
  total_paid = 150 :=
by
  sorry

end NUMINAMATH_GPT_sebastian_total_payment_l2397_239733


namespace NUMINAMATH_GPT_arithmetic_sequence_l2397_239718

theorem arithmetic_sequence (a_n : ℕ → ℕ) (a1 d : ℤ)
  (h1 : 4 * a1 + 6 * d = 0)
  (h2 : a1 + 4 * d = 5) :
  ∀ n : ℕ, a_n n = 2 * n - 5 :=
by
  -- Definitions derived from conditions
  let a_1 := (5 - 4 * d)
  let common_difference := 2
  intro n
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_l2397_239718


namespace NUMINAMATH_GPT_profit_percentage_is_20_l2397_239738

variable (C : ℝ) -- Assuming the cost price C is a real number.

theorem profit_percentage_is_20 
  (h1 : 10 * 1 = 12 * (C / 1)) :  -- Shopkeeper sold 10 articles at the cost price of 12 articles.
  ((12 * C - 10 * C) / (10 * C)) * 100 = 20 := 
by
  sorry

end NUMINAMATH_GPT_profit_percentage_is_20_l2397_239738


namespace NUMINAMATH_GPT_vector_addition_example_l2397_239787

theorem vector_addition_example :
  (⟨-3, 2, -1⟩ : ℝ × ℝ × ℝ) + (⟨1, 5, -3⟩ : ℝ × ℝ × ℝ) = ⟨-2, 7, -4⟩ :=
by
  sorry

end NUMINAMATH_GPT_vector_addition_example_l2397_239787


namespace NUMINAMATH_GPT_smallest_possible_value_of_N_l2397_239706

-- Declares the context and required constraints
theorem smallest_possible_value_of_N (N : ℕ) (h1 : 70 < N) (h2 : 70 ∣ 21 * N) : N = 80 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_value_of_N_l2397_239706


namespace NUMINAMATH_GPT_arithmetic_sequence_30th_term_l2397_239709

theorem arithmetic_sequence_30th_term :
  let a := 3
  let d := 7 - 3
  ∀ n, (n = 30) → (a + (n - 1) * d) = 119 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_30th_term_l2397_239709


namespace NUMINAMATH_GPT_trigonometric_identity_l2397_239732

open Real

theorem trigonometric_identity
  (α : ℝ)
  (h1 : 0 ≤ α ∧ α ≤ π / 2)
  (h2 : cos α = 3 / 5) :
  (1 + sqrt 2 * cos (2 * α - π / 4)) / sin (α + π / 2) = 14 / 5 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2397_239732


namespace NUMINAMATH_GPT_bounce_ratio_l2397_239750

theorem bounce_ratio (r : ℝ) (h₁ : 96 * r^4 = 3) : r = Real.sqrt 2 / 4 :=
by
  sorry

end NUMINAMATH_GPT_bounce_ratio_l2397_239750


namespace NUMINAMATH_GPT_misty_is_three_times_smaller_l2397_239702

-- Define constants representing the favorite numbers of Misty and Glory
def G : ℕ := 450
def total_sum : ℕ := 600

-- Define Misty's favorite number in terms of the total sum and Glory's favorite number
def M : ℕ := total_sum - G

-- The main theorem stating that Misty's favorite number is 3 times smaller than Glory's favorite number
theorem misty_is_three_times_smaller : G / M = 3 := by
  -- Sorry placeholder indicating the need for further proof
  sorry

end NUMINAMATH_GPT_misty_is_three_times_smaller_l2397_239702


namespace NUMINAMATH_GPT_fraction_of_state_quarters_is_two_fifths_l2397_239768

variable (total_quarters state_quarters : ℕ)
variable (is_pennsylvania_percentage : ℚ)
variable (pennsylvania_state_quarters : ℕ)

theorem fraction_of_state_quarters_is_two_fifths
  (h1 : total_quarters = 35)
  (h2 : pennsylvania_state_quarters = 7)
  (h3 : is_pennsylvania_percentage = 1 / 2)
  (h4 : state_quarters = 2 * pennsylvania_state_quarters)
  : (state_quarters : ℚ) / (total_quarters : ℚ) = 2 / 5 :=
sorry

end NUMINAMATH_GPT_fraction_of_state_quarters_is_two_fifths_l2397_239768


namespace NUMINAMATH_GPT_maximize_profit_l2397_239751

def cost_price_A (x y : ℕ) := x = y + 20
def cost_sum_eq_200 (x y : ℕ) := x + 2 * y = 200
def linear_function (m n : ℕ) := m = -((1/2) : ℚ) * n + 90
def profit_function (w n : ℕ) : ℚ := (-((1/2) : ℚ) * ((n : ℚ) - 130)^2) + 1250

theorem maximize_profit
  (x y m n : ℕ)
  (hx : cost_price_A x y)
  (hsum : cost_sum_eq_200 x y)
  (hlin : linear_function m n)
  (hmaxn : 80 ≤ n ∧ n ≤ 120)
  : y = 60 ∧ x = 80 ∧ n = 120 ∧ profit_function 120 120 = 1200 := 
sorry

end NUMINAMATH_GPT_maximize_profit_l2397_239751


namespace NUMINAMATH_GPT_g_3_2_eq_neg3_l2397_239717

noncomputable def f (x y : ℝ) : ℝ := x^3 * y^2 + 4 * x^2 * y - 15 * x

axiom f_symmetric : ∀ x y : ℝ, f x y = f y x
axiom f_2_4_eq_neg2 : f 2 4 = -2

noncomputable def g (x y : ℝ) : ℝ := (x^3 - 3 * x^2 * y + x * y^2) / (x^2 - y^2)

theorem g_3_2_eq_neg3 : g 3 2 = -3 := by
  sorry

end NUMINAMATH_GPT_g_3_2_eq_neg3_l2397_239717


namespace NUMINAMATH_GPT_compute_xy_l2397_239778

theorem compute_xy (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 54) : x * y = -9 := 
by 
  sorry

end NUMINAMATH_GPT_compute_xy_l2397_239778


namespace NUMINAMATH_GPT_correct_option_is_C_l2397_239795

theorem correct_option_is_C (x y : ℝ) :
  ¬(3 * x + 4 * y = 12 * x * y) ∧
  ¬(x^9 / x^3 = x^3) ∧
  ((x^2)^3 = x^6) ∧
  ¬((x - y)^2 = x^2 - y^2) :=
by
  sorry

end NUMINAMATH_GPT_correct_option_is_C_l2397_239795


namespace NUMINAMATH_GPT_fibonacci_odd_index_not_divisible_by_4k_plus_3_l2397_239705

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_odd_index_not_divisible_by_4k_plus_3 (n k : ℕ) (p : ℕ) (h : p = 4 * k + 3) : ¬ (p ∣ fibonacci (2 * n - 1)) :=
by
  sorry

end NUMINAMATH_GPT_fibonacci_odd_index_not_divisible_by_4k_plus_3_l2397_239705


namespace NUMINAMATH_GPT_functions_equiv_l2397_239737

noncomputable def f_D (x : ℝ) : ℝ := Real.log (Real.sqrt x)
noncomputable def g_D (x : ℝ) : ℝ := (1/2) * Real.log x

theorem functions_equiv : ∀ x : ℝ, x > 0 → f_D x = g_D x := by
  intro x h
  sorry

end NUMINAMATH_GPT_functions_equiv_l2397_239737


namespace NUMINAMATH_GPT_youseff_blocks_l2397_239743

theorem youseff_blocks (x : ℕ) (h1 : x = 1 * x) (h2 : (20 / 60 : ℚ) * x = x / 3) (h3 : x = x / 3 + 8) : x = 12 := by
  have : x = x := rfl  -- trivial step to include the equality
  sorry

end NUMINAMATH_GPT_youseff_blocks_l2397_239743


namespace NUMINAMATH_GPT_cricket_average_l2397_239716

theorem cricket_average (x : ℕ) (h : 20 * x + 158 = 21 * (x + 6)) : x = 32 :=
by
  sorry

end NUMINAMATH_GPT_cricket_average_l2397_239716


namespace NUMINAMATH_GPT_find_divisor_l2397_239721

-- Define the conditions
def dividend := 689
def quotient := 19
def remainder := 5

-- Define the division formula
def division_formula (divisor : ℕ) : Prop := 
  dividend = (divisor * quotient) + remainder

-- State the theorem to be proved
theorem find_divisor :
  ∃ divisor : ℕ, division_formula divisor ∧ divisor = 36 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l2397_239721


namespace NUMINAMATH_GPT_intersection_with_negative_y_axis_max_value_at_x3_l2397_239711

theorem intersection_with_negative_y_axis (m : ℝ) (h : 4 - 2 * m < 0) : m > 2 :=
sorry

theorem max_value_at_x3 (m : ℝ) (h : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → 3 * x + 4 - 2 * m ≤ -4) : m = 8.5 :=
sorry

end NUMINAMATH_GPT_intersection_with_negative_y_axis_max_value_at_x3_l2397_239711


namespace NUMINAMATH_GPT_cannot_equal_120_l2397_239773

def positive_even (n : ℕ) : Prop := n > 0 ∧ n % 2 = 0

theorem cannot_equal_120 (a b : ℕ) (ha : positive_even a) (hb : positive_even b) :
  let A := a * b
  let P' := 2 * (a + b) + 6
  A + P' ≠ 120 :=
sorry

end NUMINAMATH_GPT_cannot_equal_120_l2397_239773


namespace NUMINAMATH_GPT_triangle_side_range_l2397_239736

theorem triangle_side_range (AB AC x : ℝ) (hAB : AB = 16) (hAC : AC = 7) (hBC : BC = x) :
  9 < x ∧ x < 23 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_range_l2397_239736


namespace NUMINAMATH_GPT_anya_possible_wins_l2397_239726

-- Define the total rounds played
def total_rounds := 25

-- Define Anya's choices
def anya_rock := 12
def anya_scissors := 6
def anya_paper := 7

-- Define Borya's choices
def borya_rock := 13
def borya_scissors := 9
def borya_paper := 3

-- Define the relationships in rock-paper-scissors game
def rock_beats_scissors := true
def scissors_beat_paper := true
def paper_beats_rock := true

-- Define no draws condition
def no_draws := total_rounds = anya_rock + anya_scissors + anya_paper ∧ total_rounds = borya_rock + borya_scissors + borya_paper

-- Proof problem statement
theorem anya_possible_wins : anya_rock + anya_scissors + anya_paper = total_rounds ∧
                             borya_rock + borya_scissors + borya_paper = total_rounds ∧
                             rock_beats_scissors ∧ scissors_beat_paper ∧ paper_beats_rock ∧
                             no_draws →
                             (9 + 3 + 7 = 19) := by
  sorry

end NUMINAMATH_GPT_anya_possible_wins_l2397_239726


namespace NUMINAMATH_GPT_cost_of_fencing_per_meter_l2397_239707

theorem cost_of_fencing_per_meter (l b : ℕ) (total_cost : ℕ) (cost_per_meter : ℝ) : 
  (l = 66) → 
  (l = b + 32) → 
  (total_cost = 5300) → 
  (2 * l + 2 * b = 200) → 
  (cost_per_meter = total_cost / 200) → 
  cost_per_meter = 26.5 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof is omitted by design
  sorry

end NUMINAMATH_GPT_cost_of_fencing_per_meter_l2397_239707


namespace NUMINAMATH_GPT_P_and_Q_together_l2397_239714

theorem P_and_Q_together (W : ℝ) (H : W > 0) :
  (1 / (1 / 4 + 1 / (1 / 3 * (1 / 4)))) = 3 :=
by
  sorry

end NUMINAMATH_GPT_P_and_Q_together_l2397_239714


namespace NUMINAMATH_GPT_fraction_days_passed_l2397_239797

-- Conditions
def total_days : ℕ := 30
def pills_per_day : ℕ := 2
def total_pills : ℕ := total_days * pills_per_day -- 60 pills
def pills_left : ℕ := 12
def pills_taken : ℕ := total_pills - pills_left -- 48 pills
def days_taken : ℕ := pills_taken / pills_per_day -- 24 days

-- Question and answer
theorem fraction_days_passed :
  (days_taken : ℚ) / (total_days : ℚ) = 4 / 5 := 
by
  sorry

end NUMINAMATH_GPT_fraction_days_passed_l2397_239797


namespace NUMINAMATH_GPT_middle_number_is_45_l2397_239781

open Real

noncomputable def middle_number (l : List ℝ) (h_len : l.length = 13) 
  (h1 : (l.sum / 13) = 9) 
  (h2 : (l.take 6).sum = 30) 
  (h3 : (l.drop 7).sum = 42): ℝ := 
  l.nthLe 6 sorry  -- middle element (index 6 in 0-based index)

theorem middle_number_is_45 (l : List ℝ) (h_len : l.length = 13) 
  (h1 : (l.sum / 13) = 9) 
  (h2 : (l.take 6).sum = 30) 
  (h3 : (l.drop 7).sum = 42) : 
  middle_number l h_len h1 h2 h3 = 45 := 
sorry

end NUMINAMATH_GPT_middle_number_is_45_l2397_239781


namespace NUMINAMATH_GPT_abs_sum_of_roots_l2397_239735

theorem abs_sum_of_roots 
  (a b c m : ℤ) 
  (h1 : a + b + c = 0)
  (h2 : ab + bc + ca = -2023)
  : |a| + |b| + |c| = 102 := 
sorry

end NUMINAMATH_GPT_abs_sum_of_roots_l2397_239735


namespace NUMINAMATH_GPT_interest_rate_calculation_l2397_239727

theorem interest_rate_calculation (P1 P2 I1 I2 : ℝ) (r1 : ℝ) :
  P2 = 1648 ∧ P1 = 2678 - P2 ∧ I2 = P2 * 0.05 * 3 ∧ I1 = P1 * r1 * 8 ∧ I1 = I2 →
  r1 = 0.03 :=
by sorry

end NUMINAMATH_GPT_interest_rate_calculation_l2397_239727


namespace NUMINAMATH_GPT_uncle_fyodor_sandwiches_count_l2397_239785

variable (sandwiches_sharik : ℕ)
variable (sandwiches_matroskin : ℕ := 3 * sandwiches_sharik)
variable (total_sandwiches_eaten : ℕ := sandwiches_sharik + sandwiches_matroskin)
variable (sandwiches_uncle_fyodor : ℕ := 2 * total_sandwiches_eaten)
variable (difference : ℕ := sandwiches_uncle_fyodor - sandwiches_sharik)

theorem uncle_fyodor_sandwiches_count :
  (difference = 21) → sandwiches_uncle_fyodor = 24 := by
  intro h
  sorry

end NUMINAMATH_GPT_uncle_fyodor_sandwiches_count_l2397_239785


namespace NUMINAMATH_GPT_paco_ate_more_sweet_than_salty_l2397_239749

theorem paco_ate_more_sweet_than_salty (s t : ℕ) (h_s : s = 5) (h_t : t = 2) : s - t = 3 :=
by
  sorry

end NUMINAMATH_GPT_paco_ate_more_sweet_than_salty_l2397_239749


namespace NUMINAMATH_GPT_katrina_cookies_left_l2397_239704

theorem katrina_cookies_left (initial_cookies morning_cookies_sold lunch_cookies_sold afternoon_cookies_sold : ℕ)
  (h1 : initial_cookies = 120)
  (h2 : morning_cookies_sold = 36)
  (h3 : lunch_cookies_sold = 57)
  (h4 : afternoon_cookies_sold = 16) :
  initial_cookies - (morning_cookies_sold + lunch_cookies_sold + afternoon_cookies_sold) = 11 := 
by 
  sorry

end NUMINAMATH_GPT_katrina_cookies_left_l2397_239704


namespace NUMINAMATH_GPT_grandmother_mistaken_l2397_239760

-- Definitions of the given conditions:
variables (N : ℕ) (x n : ℕ)
variable (initial_split : N % 4 = 0)

-- Conditions
axiom cows_survived : 4 * (N / 4) / 5 = N / 5
axiom horses_pigs : x = N / 4 - N / 5
axiom rabbit_ratio : (N / 4 - n) = 5 / 14 * (N / 5 + N / 4 + N / 4 - n)

-- Goal: Prove the grandmother is mistaken, i.e., some species avoided casualties
theorem grandmother_mistaken : n = 0 :=
sorry

end NUMINAMATH_GPT_grandmother_mistaken_l2397_239760


namespace NUMINAMATH_GPT_table_sale_price_percentage_l2397_239729

theorem table_sale_price_percentage (W : ℝ) : 
  let S := 1.4 * W
  let P := 0.65 * S
  P = 0.91 * W :=
by
  sorry

end NUMINAMATH_GPT_table_sale_price_percentage_l2397_239729


namespace NUMINAMATH_GPT_assignment_increase_l2397_239765

-- Define what an assignment statement is
def assignment_statement (lhs rhs : ℕ) : ℕ := rhs

-- Define the conditions and the problem
theorem assignment_increase (n : ℕ) : assignment_statement n (n + 1) = n + 1 :=
by
  -- Here we would prove that the assignment statement increases n by 1
  sorry

end NUMINAMATH_GPT_assignment_increase_l2397_239765


namespace NUMINAMATH_GPT_range_of_a_l2397_239762

theorem range_of_a (a : ℝ) :
  (-1 < x ∧ x < 0 → (x^2 - a * x + 2 * a) > 0) ∧
  (0 < x → (x^2 - a * x + 2 * a) < 0) ↔ -1 / 3 < a ∧ a < 0 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2397_239762


namespace NUMINAMATH_GPT_num_boys_is_22_l2397_239708

variable (girls boys total_students : ℕ)

-- Conditions
axiom h1 : total_students = 41
axiom h2 : boys = girls + 3
axiom h3 : total_students = girls + boys

-- Goal: Prove that the number of boys is 22
theorem num_boys_is_22 : boys = 22 :=
by
  sorry

end NUMINAMATH_GPT_num_boys_is_22_l2397_239708


namespace NUMINAMATH_GPT_number_of_days_l2397_239754

def burger_meal_cost : ℕ := 6
def upsize_cost : ℕ := 1
def total_spending : ℕ := 35

/-- The number of days Clinton buys the meal. -/
theorem number_of_days (h1 : burger_meal_cost + upsize_cost = 7) (h2 : total_spending = 35) : total_spending / (burger_meal_cost + upsize_cost) = 5 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_number_of_days_l2397_239754


namespace NUMINAMATH_GPT_fraction_is_percent_l2397_239719

theorem fraction_is_percent (y : ℝ) (hy : y > 0) : (6 * y / 20 + 3 * y / 10) = (60 / 100) * y :=
by
  sorry

end NUMINAMATH_GPT_fraction_is_percent_l2397_239719


namespace NUMINAMATH_GPT_set_intersection_l2397_239766

open Finset

-- Let the universal set U, and sets A and B be defined as follows:
def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Finset ℕ := {1, 2, 3, 5}
def B : Finset ℕ := {2, 4, 6}

-- Define the complement of A with respect to U:
def complement_A : Finset ℕ := U \ A

-- The goal is to prove that B ∩ complement_A = {4, 6}
theorem set_intersection (h : B ∩ complement_A = {4, 6}) : B ∩ complement_A = {4, 6} :=
by exact h

#check set_intersection

end NUMINAMATH_GPT_set_intersection_l2397_239766


namespace NUMINAMATH_GPT_AJ_stamps_l2397_239764

theorem AJ_stamps (A : ℕ)
  (KJ := A / 2)
  (CJ := 2 * KJ + 5)
  (BJ := 3 * A - 3)
  (total_stamps := A + KJ + CJ + BJ)
  (h : total_stamps = 1472) :
  A = 267 :=
  sorry

end NUMINAMATH_GPT_AJ_stamps_l2397_239764


namespace NUMINAMATH_GPT_find_x_l2397_239713

def x : ℕ := 70

theorem find_x :
  x + (5 * 12) / (180 / 3) = 71 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2397_239713


namespace NUMINAMATH_GPT_greatest_possible_subway_takers_l2397_239775

/-- In a company with 48 employees, some part-time and some full-time, exactly (1/3) of the part-time
employees and (1/4) of the full-time employees take the subway to work. Prove that the greatest
possible number of employees who take the subway to work is 15. -/
theorem greatest_possible_subway_takers
  (P F : ℕ)
  (h : P + F = 48)
  (h_subway_part : ∀ p, p = P → 0 ≤ p ∧ p ≤ 48)
  (h_subway_full : ∀ f, f = F → 0 ≤ f ∧ f ≤ 48) :
  ∃ y, y = 15 := 
sorry

end NUMINAMATH_GPT_greatest_possible_subway_takers_l2397_239775


namespace NUMINAMATH_GPT_ellipse_product_l2397_239752

noncomputable def computeProduct (a b : ℝ) : ℝ :=
  let AB := 2 * a
  let CD := 2 * b
  AB * CD

theorem ellipse_product (a b : ℝ) (h1 : a^2 - b^2 = 64) (h2 : a - b = 4) :
  computeProduct a b = 240 := by
sorry

end NUMINAMATH_GPT_ellipse_product_l2397_239752


namespace NUMINAMATH_GPT_minimum_reciprocal_sum_l2397_239747

theorem minimum_reciprocal_sum 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h : x^2 + y^2 = x * y * (x^2 * y^2 + 2)) : 
  (1 / x + 1 / y) ≥ 2 :=
by 
  sorry -- Proof to be completed

end NUMINAMATH_GPT_minimum_reciprocal_sum_l2397_239747


namespace NUMINAMATH_GPT_thabo_book_ratio_l2397_239710

theorem thabo_book_ratio :
  ∃ (P_f P_nf H_nf : ℕ), H_nf = 35 ∧ P_nf = H_nf + 20 ∧ P_f + P_nf + H_nf = 200 ∧ P_f / P_nf = 2 :=
by
  sorry

end NUMINAMATH_GPT_thabo_book_ratio_l2397_239710


namespace NUMINAMATH_GPT_radius_is_100_div_pi_l2397_239720

noncomputable def radius_of_circle (L : ℝ) (θ : ℝ) : ℝ :=
  L * 360 / (θ * 2 * Real.pi)

theorem radius_is_100_div_pi :
  radius_of_circle 25 45 = 100 / Real.pi := 
by
  sorry

end NUMINAMATH_GPT_radius_is_100_div_pi_l2397_239720


namespace NUMINAMATH_GPT_perimeter_of_rectangle_EFGH_l2397_239782

noncomputable def rectangle_ellipse_problem (u v c d : ℝ) : Prop :=
  (u * v = 3000) ∧
  (3000 = c * d) ∧
  ((u + v) = 2 * c) ∧
  ((u^2 + v^2).sqrt = 2 * (c^2 - d^2).sqrt) ∧
  (d = 3000 / c) ∧
  (4 * c = 8 * (1500).sqrt)

theorem perimeter_of_rectangle_EFGH :
  ∃ (u v c d : ℝ), rectangle_ellipse_problem u v c d ∧ 2 * (u + v) = 8 * (1500).sqrt := sorry

end NUMINAMATH_GPT_perimeter_of_rectangle_EFGH_l2397_239782


namespace NUMINAMATH_GPT_rationalize_denominator_l2397_239794

theorem rationalize_denominator :
  ∃ A B C : ℤ, A * B * C = 180 ∧
  (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = A + B * Real.sqrt C :=
sorry

end NUMINAMATH_GPT_rationalize_denominator_l2397_239794


namespace NUMINAMATH_GPT_range_of_c_l2397_239776

variable (c : ℝ)

def p := 2 < 3 * c
def q := ∀ x : ℝ, 2 * x^2 + 4 * c * x + 1 > 0

theorem range_of_c (hp : p c) (hq : q c) : (2 / 3) < c ∧ c < (Real.sqrt 2 / 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_c_l2397_239776


namespace NUMINAMATH_GPT_no_intersection_of_lines_l2397_239757

theorem no_intersection_of_lines :
  ¬ ∃ (s v : ℝ) (x y : ℝ),
    (x = 1 - 2 * s ∧ y = 4 + 6 * s) ∧
    (x = 3 - v ∧ y = 10 + 3 * v) :=
by {
  sorry
}

end NUMINAMATH_GPT_no_intersection_of_lines_l2397_239757


namespace NUMINAMATH_GPT_unique_sum_of_two_primes_l2397_239725

theorem unique_sum_of_two_primes (p1 p2 : ℕ) (hp1_prime : Prime p1) (hp2_prime : Prime p2) (hp1_even : p1 = 2) (sum_eq : p1 + p2 = 10003) : 
  p1 = 2 ∧ p2 = 10001 ∧ (∀ p1' p2', Prime p1' → Prime p2' → p1' + p2' = 10003 → (p1' = 2 ∧ p2' = 10001) ∨ (p1' = 10001 ∧ p2' = 2)) :=
by
  sorry

end NUMINAMATH_GPT_unique_sum_of_two_primes_l2397_239725


namespace NUMINAMATH_GPT_no_two_primes_sum_to_53_l2397_239771

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_two_primes_sum_to_53 :
  ¬ ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 53 :=
by
  sorry

end NUMINAMATH_GPT_no_two_primes_sum_to_53_l2397_239771


namespace NUMINAMATH_GPT_machines_working_time_l2397_239701

theorem machines_working_time (y: ℝ) 
  (h1 : y + 8 > 0)  -- condition for time taken by S
  (h2 : y + 2 > 0)  -- condition for time taken by T
  (h3 : 2 * y > 0)  -- condition for time taken by U
  : (1 / (y + 8) + 1 / (y + 2) + 1 / (2 * y) = 1 / y) ↔ (y = 3 / 2) := 
by
  have h4 : y ≠ 0 := by linarith [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_machines_working_time_l2397_239701


namespace NUMINAMATH_GPT_find_certain_number_l2397_239780

theorem find_certain_number (x : ℤ) (h : x + 34 - 53 = 28) : x = 47 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_certain_number_l2397_239780


namespace NUMINAMATH_GPT_sequence_sum_l2397_239722

theorem sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, S n = n^2 * a n) :
  ∀ n : ℕ, S n = 2 * n / (n + 1) := 
by 
  sorry

end NUMINAMATH_GPT_sequence_sum_l2397_239722


namespace NUMINAMATH_GPT_rainfall_second_week_l2397_239740

theorem rainfall_second_week (r1 r2 : ℝ) (h1 : r1 + r2 = 35) (h2 : r2 = 1.5 * r1) : r2 = 21 := 
  sorry

end NUMINAMATH_GPT_rainfall_second_week_l2397_239740


namespace NUMINAMATH_GPT_increasing_interval_l2397_239746

noncomputable def f (x : ℝ) := Real.logb 2 (5 - 4 * x - x^2)

theorem increasing_interval : ∀ {x : ℝ}, (-5 < x ∧ x ≤ -2) → f x = Real.logb 2 (5 - 4 * x - x^2) := by
  sorry

end NUMINAMATH_GPT_increasing_interval_l2397_239746


namespace NUMINAMATH_GPT_primeFactors_of_3_pow_6_minus_1_l2397_239756

def calcPrimeFactorsSumAndSumOfSquares (n : ℕ) : ℕ × ℕ :=
  let factors := [2, 7, 13]  -- Given directly
  let sum_factors := 2 + 7 + 13
  let sum_squares := 2^2 + 7^2 + 13^2
  (sum_factors, sum_squares)

theorem primeFactors_of_3_pow_6_minus_1 :
  calcPrimeFactorsSumAndSumOfSquares (3^6 - 1) = (22, 222) :=
by
  sorry

end NUMINAMATH_GPT_primeFactors_of_3_pow_6_minus_1_l2397_239756


namespace NUMINAMATH_GPT_multiplication_with_negative_l2397_239763

theorem multiplication_with_negative (a b : Int) (h1 : a = 3) (h2 : b = -2) : a * b = -6 :=
by
  sorry

end NUMINAMATH_GPT_multiplication_with_negative_l2397_239763


namespace NUMINAMATH_GPT_remainder_when_dividing_386_l2397_239784

theorem remainder_when_dividing_386 :
  (386 % 35 = 1) ∧ (386 % 11 = 1) :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_dividing_386_l2397_239784


namespace NUMINAMATH_GPT_min_diff_two_composite_sum_91_l2397_239734

-- Define what it means for a number to be composite
def is_composite (n : ℕ) : Prop :=
  ∃ p q : ℕ, p > 1 ∧ q > 1 ∧ p * q = n

-- Minimum positive difference between two composite numbers that sum up to 91
theorem min_diff_two_composite_sum_91 : ∃ a b : ℕ, 
  is_composite a ∧ 
  is_composite b ∧ 
  a + b = 91 ∧ 
  b - a = 1 :=
by
  sorry

end NUMINAMATH_GPT_min_diff_two_composite_sum_91_l2397_239734


namespace NUMINAMATH_GPT_susans_average_speed_l2397_239769

theorem susans_average_speed :
  ∀ (total_distance first_leg_distance second_leg_distance : ℕ)
    (first_leg_speed second_leg_speed : ℕ)
    (total_time : ℚ),
    first_leg_distance = 40 →
    second_leg_distance = 20 →
    first_leg_speed = 15 →
    second_leg_speed = 60 →
    total_distance = first_leg_distance + second_leg_distance →
    total_time = (first_leg_distance / first_leg_speed : ℚ) + (second_leg_distance / second_leg_speed : ℚ) →
    total_distance / total_time = 20 :=
by
  sorry

end NUMINAMATH_GPT_susans_average_speed_l2397_239769


namespace NUMINAMATH_GPT_inequality_proof_l2397_239744

theorem inequality_proof (a b c d : ℝ) : 
  (a + b + c + d) * (a * b * (c + d) + (a + b) * c * d) - a * b * c * d ≤ 
  (1 / 2) * (a * (b + d) + b * (c + d) + c * (d + a))^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l2397_239744


namespace NUMINAMATH_GPT_tangent_line_at_point_P_l2397_239777

-- Define the curve y = x^3 
def curve (x : ℝ) : ℝ := x ^ 3

-- Define the point P(1,1)
def pointP : ℝ × ℝ := (1, 1)

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 3 * x ^ 2

-- Define the tangent line equation we need to prove
def tangent_line (x y : ℝ) : Prop := 3 * x - y - 2 = 0

theorem tangent_line_at_point_P :
  ∀ (x y : ℝ), 
  pointP = (1, 1) ∧ curve 1 = 1 ∧ curve_derivative 1 = 3 → 
  tangent_line 1 1 := 
by
  intros x y h
  sorry

end NUMINAMATH_GPT_tangent_line_at_point_P_l2397_239777
