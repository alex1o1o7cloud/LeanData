import Mathlib

namespace NUMINAMATH_GPT_find_A_in_triangle_l1860_186076

theorem find_A_in_triangle
  (a b : ℝ) (B A : ℝ)
  (h₀ : a = Real.sqrt 3)
  (h₁ : b = Real.sqrt 2)
  (h₂ : B = Real.pi / 4)
  (h₃ : a / Real.sin A = b / Real.sin B) :
  A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_find_A_in_triangle_l1860_186076


namespace NUMINAMATH_GPT_thickness_of_layer_l1860_186030

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ := Real.pi * r^2 * h

theorem thickness_of_layer (radius_sphere radius_cylinder : ℝ) (volume_sphere volume_cylinder : ℝ) (h : ℝ) : 
  radius_sphere = 3 → 
  radius_cylinder = 10 →
  volume_sphere = volume_of_sphere radius_sphere →
  volume_cylinder = volume_of_cylinder radius_cylinder h →
  volume_sphere = volume_cylinder → 
  h = 9 / 25 :=
by
  intros
  sorry

end NUMINAMATH_GPT_thickness_of_layer_l1860_186030


namespace NUMINAMATH_GPT_sqrt_eq_two_or_neg_two_l1860_186062

theorem sqrt_eq_two_or_neg_two (x : ℝ) (h : x^2 = 4) : x = 2 ∨ x = -2 :=
sorry

end NUMINAMATH_GPT_sqrt_eq_two_or_neg_two_l1860_186062


namespace NUMINAMATH_GPT_remainder_5_pow_2023_mod_11_l1860_186011

theorem remainder_5_pow_2023_mod_11 : (5^2023) % 11 = 4 :=
by
  have h1 : 5^2 % 11 = 25 % 11 := sorry
  have h2 : 25 % 11 = 3 := sorry
  have h3 : (3^5) % 11 = 1 := sorry
  have h4 : 3^1011 % 11 = ((3^5)^202 * 3) % 11 := sorry
  have h5 : ((3^5)^202 * 3) % 11 = (1^202 * 3) % 11 := sorry
  have h6 : (1^202 * 3) % 11 = 3 % 11 := sorry
  have h7 : (5^2023) % 11 = (3 * 5) % 11 := sorry
  have h8 : (3 * 5) % 11 = 15 % 11 := sorry
  have h9 : 15 % 11 = 4 := sorry
  exact h9

end NUMINAMATH_GPT_remainder_5_pow_2023_mod_11_l1860_186011


namespace NUMINAMATH_GPT_average_score_of_juniors_l1860_186055

theorem average_score_of_juniors :
  ∀ (N : ℕ) (junior_percent senior_percent overall_avg senior_avg : ℚ),
  junior_percent = 0.20 →
  senior_percent = 0.80 →
  overall_avg = 86 →
  senior_avg = 85 →
  (N * overall_avg - (N * senior_percent * senior_avg)) / (N * junior_percent) = 90 := 
by
  intros N junior_percent senior_percent overall_avg senior_avg
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_average_score_of_juniors_l1860_186055


namespace NUMINAMATH_GPT_find_x_value_l1860_186074

theorem find_x_value (x : ℝ) (h1 : 0 < x) (h2 : x < 180) :
  (Real.tan (150 - x * Real.pi / 180) = 
   (Real.sin (150 * Real.pi / 180) - Real.sin (x * Real.pi / 180)) /
   (Real.cos (150 * Real.pi / 180) - Real.cos (x * Real.pi / 180))) → 
  x = 110 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_value_l1860_186074


namespace NUMINAMATH_GPT_no_convex_quad_with_given_areas_l1860_186069

theorem no_convex_quad_with_given_areas :
  ¬ ∃ (A B C D M : Type) 
    (T_MAB T_MBC T_MDA T_MDC : ℕ) 
    (H1 : T_MAB = 1) 
    (H2 : T_MBC = 2)
    (H3 : T_MDA = 3) 
    (H4 : T_MDC = 4),
    true :=
by {
  sorry
}

end NUMINAMATH_GPT_no_convex_quad_with_given_areas_l1860_186069


namespace NUMINAMATH_GPT_derivative_at_pi_over_3_l1860_186012

noncomputable def f (x : ℝ) : ℝ := Real.cos x + Real.sqrt 3 * Real.sin x

theorem derivative_at_pi_over_3 : 
  (deriv f) (Real.pi / 3) = 0 := 
by 
  sorry

end NUMINAMATH_GPT_derivative_at_pi_over_3_l1860_186012


namespace NUMINAMATH_GPT_walking_speed_l1860_186084

theorem walking_speed (d : ℝ) (w_speed r_speed : ℝ) (w_time r_time : ℝ)
    (h1 : d = r_speed * r_time)
    (h2 : r_speed = 24)
    (h3 : r_time = 1)
    (h4 : w_time = 3) :
    w_speed = 8 :=
by
  sorry

end NUMINAMATH_GPT_walking_speed_l1860_186084


namespace NUMINAMATH_GPT_four_b_is_222_22_percent_of_a_l1860_186003

-- noncomputable is necessary because Lean does not handle decimal numbers directly
noncomputable def a (b : ℝ) : ℝ := 1.8 * b
noncomputable def four_b (b : ℝ) : ℝ := 4 * b

theorem four_b_is_222_22_percent_of_a (b : ℝ) : four_b b = 2.2222 * a b := 
by
  sorry

end NUMINAMATH_GPT_four_b_is_222_22_percent_of_a_l1860_186003


namespace NUMINAMATH_GPT_wharf_length_l1860_186083

-- Define the constants
def avg_speed := 2 -- average speed in m/s
def travel_time := 16 -- travel time in seconds

-- Define the formula to calculate length of the wharf
def length_of_wharf := 2 * avg_speed * travel_time

-- The goal is to prove that length_of_wharf equals 64
theorem wharf_length : length_of_wharf = 64 :=
by
  -- Proof would be here
  sorry

end NUMINAMATH_GPT_wharf_length_l1860_186083


namespace NUMINAMATH_GPT_sum_of_reciprocals_is_five_l1860_186019

theorem sum_of_reciprocals_is_five (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x - y = 3 * x * y) : 
  (1 / x) + (1 / y) = 5 :=
sorry

end NUMINAMATH_GPT_sum_of_reciprocals_is_five_l1860_186019


namespace NUMINAMATH_GPT_log_inequality_l1860_186093

theorem log_inequality (n : ℕ) (h1 : n > 1) : 
  (1 : ℝ) / (n : ℝ) > Real.log ((n + 1 : ℝ) / n) ∧ 
  Real.log ((n + 1 : ℝ) / n) > (1 : ℝ) / (n + 1) := 
by
  sorry

end NUMINAMATH_GPT_log_inequality_l1860_186093


namespace NUMINAMATH_GPT_range_of_m_l1860_186001

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 5| < m^2 - m) ↔ m < -1 ∨ m > 2 := 
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1860_186001


namespace NUMINAMATH_GPT_hyperbola_asymptote_l1860_186039

theorem hyperbola_asymptote (a : ℝ) (h₀ : a > 0) 
  (h₁ : ∃ (x y : ℝ), (x, y) = (2, 1) ∧ 
       (y = (2 / a) * x ∨ y = -(2 / a) * x)) : a = 4 := by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_l1860_186039


namespace NUMINAMATH_GPT_inequality_one_inequality_system_l1860_186008

theorem inequality_one (x : ℝ) : 2 * x + 3 ≤ 5 * x ↔ x ≥ 1 := sorry

theorem inequality_system (x : ℝ) : 
  (5 * x - 1 ≤ 3 * (x + 1)) ∧ 
  ((2 * x - 1) / 2 - (5 * x - 1) / 4 < 1) ↔ 
  (-5 < x ∧ x ≤ 2) := sorry

end NUMINAMATH_GPT_inequality_one_inequality_system_l1860_186008


namespace NUMINAMATH_GPT_min_f_x_gt_2_solve_inequality_l1860_186057

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 / (x + b)

theorem min_f_x_gt_2 (a b : ℝ) (h1 : ∀ x, f a b x = 2 * x + 3 → x = -2 ∨ x = 3) :
∃ c, ∀ x > 2, f a b x ≥ c ∧ (∀ y, y > 2 → f a b y = c → y = 4 ∧ c = 8) :=
sorry

theorem solve_inequality (a b k : ℝ) (x : ℝ) (h1 : ∀ x, f a b x = 2 * x + 3 → x = -2 ∨ x = 3) :
  f a b x < (k * (x - 1) + 1 - x^2) / (2 - x) ↔ 
  (x < 2 ∧ k = 0) ∨ 
  (-1 < k ∧ k < 0 ∧ 1 - 1 / k < x ∧ x < 2) ∨ 
  ((k > 0 ∨ k < -1) ∧ (1 - 1 / k < x ∧ x < 2) ∨ x > 2) ∨ 
  (k = -1 ∧ x ≠ 2) :=
sorry

end NUMINAMATH_GPT_min_f_x_gt_2_solve_inequality_l1860_186057


namespace NUMINAMATH_GPT_smallest_positive_a_integer_root_l1860_186085

theorem smallest_positive_a_integer_root :
  ∀ x a : ℚ, (exists x : ℚ, (x > 0) ∧ (a > 0) ∧ 
    (
      ((x - a) / 2 + (x - 2 * a) / 3) / ((x + 4 * a) / 5 - (x + 3 * a) / 4) =
      ((x - 3 * a) / 4 + (x - 4 * a) / 5) / ((x + 2 * a) / 3 - (x + a) / 2)
    )
  ) → a = 419 / 421 :=
by sorry

end NUMINAMATH_GPT_smallest_positive_a_integer_root_l1860_186085


namespace NUMINAMATH_GPT_parabolic_arch_properties_l1860_186037

noncomputable def parabolic_arch_height (x : ℝ) : ℝ :=
  let a : ℝ := -4 / 125
  let k : ℝ := 20
  a * x^2 + k

theorem parabolic_arch_properties :
  (parabolic_arch_height 10 = 16.8) ∧ (parabolic_arch_height 10 = parabolic_arch_height 10 → (10 = 10 ∨ 10 = -10)) :=
by
  have h1 : parabolic_arch_height 10 = 16.8 :=
    sorry
  have h2 : parabolic_arch_height 10 = parabolic_arch_height 10 → (10 = 10 ∨ 10 = -10) :=
    sorry
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_parabolic_arch_properties_l1860_186037


namespace NUMINAMATH_GPT_find_m_l1860_186041

variable (m : ℝ)

theorem find_m (h1 : 3 * (-7.5) - y = m) (h2 : -0.4 * (-7.5) + y = 3) : m = -22.5 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1860_186041


namespace NUMINAMATH_GPT_problem_relation_l1860_186091

-- Definitions indicating relationships.
def related₁ : Prop := ∀ (s : ℝ), (s ≥ 0) → (∃ a p : ℝ, a = s^2 ∧ p = 4 * s)
def related₂ : Prop := ∀ (d t : ℝ), (t > 0) → (∃ v : ℝ, d = v * t)
def related₃ : Prop := ∃ (h w : ℝ) (f : ℝ → ℝ), w = f h
def related₄ : Prop := ∀ (h : ℝ) (v : ℝ), False

-- The theorem stating that A, B, and C are related.
theorem problem_relation : 
  related₁ ∧ related₂ ∧ related₃ ∧ ¬ related₄ :=
by sorry

end NUMINAMATH_GPT_problem_relation_l1860_186091


namespace NUMINAMATH_GPT_literature_books_cost_more_l1860_186044

theorem literature_books_cost_more :
  let num_books := 45
  let literature_cost_per_book := 7
  let technology_cost_per_book := 5
  (num_books * literature_cost_per_book) - (num_books * technology_cost_per_book) = 90 :=
by
  sorry

end NUMINAMATH_GPT_literature_books_cost_more_l1860_186044


namespace NUMINAMATH_GPT_impossible_partition_10x10_square_l1860_186072

theorem impossible_partition_10x10_square :
  ¬ ∃ (x y : ℝ), (x - y = 1) ∧ (x * y = 1) ∧ (∃ (n m : ℕ), 10 = n * x + m * y ∧ n + m = 100) :=
by
  sorry

end NUMINAMATH_GPT_impossible_partition_10x10_square_l1860_186072


namespace NUMINAMATH_GPT_count_three_letter_sets_l1860_186047

-- Define the set of letters
def letters := Finset.range 10  -- representing letters A (0) to J (9)

-- Define the condition that J (represented by 9) cannot be the first initial
def valid_first_initials := letters.erase 9  -- remove 9 (J) from 0 to 9

-- Calculate the number of valid three-letter sets of initials
theorem count_three_letter_sets : 
  let first_initials := valid_first_initials
  let second_initials := letters
  let third_initials := letters
  first_initials.card * second_initials.card * third_initials.card = 900 := by
  sorry

end NUMINAMATH_GPT_count_three_letter_sets_l1860_186047


namespace NUMINAMATH_GPT_katherine_savings_multiple_l1860_186035

variable (A K : ℕ)

theorem katherine_savings_multiple
  (h1 : A + K = 750)
  (h2 : A - 150 = 1 / 3 * K) :
  2 * K / A = 3 :=
sorry

end NUMINAMATH_GPT_katherine_savings_multiple_l1860_186035


namespace NUMINAMATH_GPT_base_7_minus_base_8_l1860_186027

def convert_base_7 (n : ℕ) : ℕ :=
  match n with
  | 543210 => 5 * 7^5 + 4 * 7^4 + 3 * 7^3 + 2 * 7^2 + 1 * 7^1 + 0 * 7^0
  | _ => 0

def convert_base_8 (n : ℕ) : ℕ :=
  match n with
  | 45321 => 4 * 8^4 + 5 * 8^3 + 3 * 8^2 + 2 * 8^1 + 1 * 8^0
  | _ => 0

theorem base_7_minus_base_8 : convert_base_7 543210 - convert_base_8 45321 = 75620 := by
  sorry

end NUMINAMATH_GPT_base_7_minus_base_8_l1860_186027


namespace NUMINAMATH_GPT_diagonal_of_rectangle_l1860_186059

theorem diagonal_of_rectangle (a b d : ℝ)
  (h_side : a = 15)
  (h_area : a * b = 120)
  (h_diag : a^2 + b^2 = d^2) :
  d = 17 :=
by
  sorry

end NUMINAMATH_GPT_diagonal_of_rectangle_l1860_186059


namespace NUMINAMATH_GPT_max_possible_median_l1860_186056

theorem max_possible_median (total_cups : ℕ) (total_customers : ℕ) (min_cups_per_customer : ℕ)
  (h1 : total_cups = 310) (h2 : total_customers = 120) (h3 : min_cups_per_customer = 1) :
  ∃ median : ℕ, median = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_possible_median_l1860_186056


namespace NUMINAMATH_GPT_range_of_a_l1860_186045

noncomputable def log_a (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → x < 1 → log_a a (2 - a * x) < log_a a (2 - a * (x / 2))) →
  1 < a ∧ a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1860_186045


namespace NUMINAMATH_GPT_int_n_satisfying_conditions_l1860_186016

theorem int_n_satisfying_conditions : 
  (∃! (n : ℤ), ∃ (k : ℤ), (n + 3 = k^2 * (23 - n)) ∧ n ≠ 23) :=
by
  use 2
  -- Provide a proof for this statement here
  sorry

end NUMINAMATH_GPT_int_n_satisfying_conditions_l1860_186016


namespace NUMINAMATH_GPT_unique_zero_point_mn_l1860_186066

noncomputable def f (a : ℝ) (x : ℝ) := a * (x^2 + 2 / x) - Real.log x

theorem unique_zero_point_mn (a : ℝ) (m n x₀ : ℝ) (hmn : m + 1 = n) (a_pos : 0 < a) (f_zero : f a x₀ = 0) (x0_in_range : m < x₀ ∧ x₀ < n) : m + n = 5 := by
  sorry

end NUMINAMATH_GPT_unique_zero_point_mn_l1860_186066


namespace NUMINAMATH_GPT_solve_for_y_l1860_186023

theorem solve_for_y {y : ℕ} (h : (1000 : ℝ) = (10 : ℝ)^3) : (1000 : ℝ)^4 = (10 : ℝ)^y ↔ y = 12 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1860_186023


namespace NUMINAMATH_GPT_area_of_plot_is_correct_l1860_186065

-- Define the side length of the square plot
def side_length : ℝ := 50.5

-- Define the area of the square plot
def area_of_square (s : ℝ) : ℝ := s * s

-- Theorem stating that the area of a square plot with side length 50.5 m is 2550.25 m²
theorem area_of_plot_is_correct : area_of_square side_length = 2550.25 := by
  sorry

end NUMINAMATH_GPT_area_of_plot_is_correct_l1860_186065


namespace NUMINAMATH_GPT_play_children_count_l1860_186077

theorem play_children_count (cost_adult_ticket cost_children_ticket total_receipts total_attendance adult_count children_count : ℕ) :
  cost_adult_ticket = 25 →
  cost_children_ticket = 15 →
  total_receipts = 7200 →
  total_attendance = 400 →
  adult_count = 280 →
  25 * adult_count + 15 * children_count = total_receipts →
  adult_count + children_count = total_attendance →
  children_count = 120 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_play_children_count_l1860_186077


namespace NUMINAMATH_GPT_ratio_of_ticket_prices_l1860_186009

-- Given conditions
def num_adults := 400
def num_children := 200
def adult_ticket_price : ℕ := 32
def total_amount : ℕ := 16000
def child_ticket_price (C : ℕ) : Prop := num_adults * adult_ticket_price + num_children * C = total_amount

theorem ratio_of_ticket_prices (C : ℕ) (hC : child_ticket_price C) :
  adult_ticket_price / C = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_ticket_prices_l1860_186009


namespace NUMINAMATH_GPT_hours_per_day_l1860_186067

theorem hours_per_day 
  (H : ℕ)
  (h1 : 6 * 8 * H = 48 * H)
  (h2 : 4 * 3 * 8 = 96)
  (h3 : (48 * H) / 75 = 96 / 30) : 
  H = 5 :=
by
  sorry

end NUMINAMATH_GPT_hours_per_day_l1860_186067


namespace NUMINAMATH_GPT_g_at_2_l1860_186049

-- Assuming g is a function from ℝ to ℝ such that it satisfies the given condition.
def g : ℝ → ℝ := sorry

-- Condition of the problem
axiom g_condition : ∀ x : ℝ, g (2 ^ x) + x * g (2 ^ (-x)) = 2

-- The statement we want to prove
theorem g_at_2 : g (2) = 0 :=
by
  sorry

end NUMINAMATH_GPT_g_at_2_l1860_186049


namespace NUMINAMATH_GPT_green_block_weight_l1860_186040

theorem green_block_weight (y g : ℝ) (h1 : y = 0.6) (h2 : y = g + 0.2) : g = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_green_block_weight_l1860_186040


namespace NUMINAMATH_GPT_minimal_rotations_triangle_l1860_186032

/-- Given a triangle with angles α, β, γ at vertices 1, 2, 3 respectively.
    The triangle returns to its original position after 15 rotations around vertex 1 by α,
    and after 6 rotations around vertex 2 by β.
    We need to show that the minimal positive integer n such that the triangle returns
    to its original position after n rotations around vertex 3 by γ is 5. -/
theorem minimal_rotations_triangle :
  ∃ (α β γ : ℝ) (k m l n : ℤ), 
    (15 * α = 360 * k) ∧ 
    (6 * β = 360 * m) ∧ 
    (α + β + γ = 180) ∧ 
    (n * γ = 360 * l) ∧ 
    (∀ n' : ℤ, n' > 0 → (∃ k' m' l' : ℤ, 
      (15 * α = 360 * k') ∧ 
      (6 * β = 360 * m') ∧ 
      (α + β + γ = 180) ∧ 
      (n' * γ = 360 * l') → n <= n')) ∧ 
    n = 5 := by
  sorry

end NUMINAMATH_GPT_minimal_rotations_triangle_l1860_186032


namespace NUMINAMATH_GPT_min_expression_min_expression_achieve_l1860_186097

theorem min_expression (x : ℝ) (hx : 0 < x) : 
  (x^2 + 8 * x + 64 / x^3) ≥ 28 :=
sorry

theorem min_expression_achieve (x : ℝ) (hx : x = 2): 
  (x^2 + 8 * x + 64 / x^3) = 28 :=
sorry

end NUMINAMATH_GPT_min_expression_min_expression_achieve_l1860_186097


namespace NUMINAMATH_GPT_area_enclosed_by_equation_is_96_l1860_186022

-- Definitions based on the conditions
def equation (x y : ℝ) : Prop := |x| + |3 * y| = 12

-- The theorem to prove the area enclosed by the graph is 96 square units
theorem area_enclosed_by_equation_is_96 :
  (∃ x y : ℝ, equation x y) → ∃ A : ℝ, A = 96 :=
sorry

end NUMINAMATH_GPT_area_enclosed_by_equation_is_96_l1860_186022


namespace NUMINAMATH_GPT_equal_numbers_product_l1860_186033

theorem equal_numbers_product :
  ∀ (a b c d : ℕ), 
  (a + b + c + d = 80) → 
  (a = 12) → 
  (b = 22) → 
  (c = d) → 
  (c * d = 529) :=
by
  intros a b c d hsum ha hb hcd
  -- proof skipped
  sorry

end NUMINAMATH_GPT_equal_numbers_product_l1860_186033


namespace NUMINAMATH_GPT_f_13_eq_223_l1860_186014

def f (n : ℕ) : ℕ := n^2 + n + 41

theorem f_13_eq_223 : f 13 = 223 :=
by
  sorry

end NUMINAMATH_GPT_f_13_eq_223_l1860_186014


namespace NUMINAMATH_GPT_smallest_n_terminating_decimal_l1860_186000

theorem smallest_n_terminating_decimal : ∃ n : ℕ, (∀ m : ℕ, m < n → (∀ k : ℕ, (n = 103 + k) → (∃ a b : ℕ, k = 2^a * 5^b)) → (k ≠ 0 → k = 125)) ∧ n = 22 := 
sorry

end NUMINAMATH_GPT_smallest_n_terminating_decimal_l1860_186000


namespace NUMINAMATH_GPT_percentage_difference_l1860_186005

theorem percentage_difference (x y : ℝ) (h : x = 12 * y) : (1 - y / x) * 100 = 91.67 :=
by {
  sorry
}

end NUMINAMATH_GPT_percentage_difference_l1860_186005


namespace NUMINAMATH_GPT_find_jamals_grade_l1860_186031

noncomputable def jamals_grade (n_students : ℕ) (absent_students : ℕ) (test_avg_28_students : ℕ) (new_total_avg_30_students : ℕ) (taqeesha_score : ℕ) : ℕ :=
  let total_28_students := 28 * test_avg_28_students
  let total_30_students := 30 * new_total_avg_30_students
  let combined_score := total_30_students - total_28_students
  combined_score - taqeesha_score

theorem find_jamals_grade :
  jamals_grade 30 2 85 86 92 = 108 :=
by
  sorry

end NUMINAMATH_GPT_find_jamals_grade_l1860_186031


namespace NUMINAMATH_GPT_abs_two_minus_sqrt_five_l1860_186071

noncomputable def sqrt_5 : ℝ := Real.sqrt 5

theorem abs_two_minus_sqrt_five : |2 - sqrt_5| = sqrt_5 - 2 := by
  sorry

end NUMINAMATH_GPT_abs_two_minus_sqrt_five_l1860_186071


namespace NUMINAMATH_GPT_sums_correct_l1860_186046

theorem sums_correct (x : ℕ) (h : x + 2 * x = 48) : x = 16 :=
by
  sorry

end NUMINAMATH_GPT_sums_correct_l1860_186046


namespace NUMINAMATH_GPT_find_x_plus_y_l1860_186006

theorem find_x_plus_y (x y : ℚ) (h1 : |x| + x + y = 12) (h2 : x + |y| - y = 10) : x + y = 26/5 :=
sorry

end NUMINAMATH_GPT_find_x_plus_y_l1860_186006


namespace NUMINAMATH_GPT_bc_sum_condition_l1860_186051

-- Define the conditions as Lean definitions
def is_positive_integer (n : ℕ) : Prop := n > 0
def not_equal_to (x y : ℕ) : Prop := x ≠ y
def less_than_or_equal_to_nine (n : ℕ) : Prop := n ≤ 9

-- Main proof statement
theorem bc_sum_condition (a b c : ℕ) (h_pos_a : is_positive_integer a) (h_pos_b : is_positive_integer b) (h_pos_c : is_positive_integer c)
  (h_a_not_1 : a ≠ 1) (h_b_not_c : b ≠ c) (h_b_le_9 : less_than_or_equal_to_nine b) (h_c_le_9 : less_than_or_equal_to_nine c)
  (h_eq : (10 * a + b) * (10 * a + c) = 100 * a * a + 110 * a + b * c) :
  b + c = 11 := by
  sorry

end NUMINAMATH_GPT_bc_sum_condition_l1860_186051


namespace NUMINAMATH_GPT_even_function_phi_l1860_186080

noncomputable def phi := (3 * Real.pi) / 2

theorem even_function_phi (phi_val : Real) (hphi : 0 ≤ phi_val ∧ phi_val ≤ 2 * Real.pi) :
  (∀ x, Real.sin ((x + phi) / 3) = Real.sin ((-x + phi) / 3)) ↔ phi_val = phi := by
  sorry

end NUMINAMATH_GPT_even_function_phi_l1860_186080


namespace NUMINAMATH_GPT_complex_numbers_satisfying_conditions_l1860_186089

theorem complex_numbers_satisfying_conditions (x y z : ℂ) 
  (h1 : x + y + z = 3) 
  (h2 : x^2 + y^2 + z^2 = 3) 
  (h3 : x^3 + y^3 + z^3 = 3) : x = 1 ∧ y = 1 ∧ z = 1 := 
by sorry

end NUMINAMATH_GPT_complex_numbers_satisfying_conditions_l1860_186089


namespace NUMINAMATH_GPT_first_term_geometric_sequence_l1860_186099

theorem first_term_geometric_sequence (a r : ℕ) (h₁ : a * r^5 = 32) (h₂ : r = 2) : a = 1 := by
  sorry

end NUMINAMATH_GPT_first_term_geometric_sequence_l1860_186099


namespace NUMINAMATH_GPT_mike_falls_short_l1860_186013

theorem mike_falls_short : 
  ∀ (max_marks mike_score : ℕ) (pass_percentage : ℚ),
  pass_percentage = 0.30 → 
  max_marks = 800 → 
  mike_score = 212 → 
  (pass_percentage * max_marks - mike_score) = 28 :=
by
  intros max_marks mike_score pass_percentage h1 h2 h3
  sorry

end NUMINAMATH_GPT_mike_falls_short_l1860_186013


namespace NUMINAMATH_GPT_smallest_quotient_is_1_9_l1860_186096

def is_two_digit_number (n : ℕ) : Prop :=
  10 <= n ∧ n <= 99

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  let x := n / 10
  let y := n % 10
  x + y

noncomputable def quotient (n : ℕ) : ℚ :=
  n / (sum_of_digits n)

theorem smallest_quotient_is_1_9 :
  ∃ n, is_two_digit_number n ∧ (∃ x y, n = 10 * x + y ∧ x ≠ y ∧ 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9) ∧ quotient n = 1.9 := 
sorry

end NUMINAMATH_GPT_smallest_quotient_is_1_9_l1860_186096


namespace NUMINAMATH_GPT_same_function_absolute_value_l1860_186090

theorem same_function_absolute_value :
  (∀ (x : ℝ), |x| = if x > 0 then x else -x) :=
by
  intro x
  split_ifs with h
  · exact abs_of_pos h
  · exact abs_of_nonpos (le_of_not_gt h)

end NUMINAMATH_GPT_same_function_absolute_value_l1860_186090


namespace NUMINAMATH_GPT_lastNumberIsOneOverSeven_l1860_186053

-- Definitions and conditions
def seq (a : ℕ → ℝ) : Prop :=
  ∀ k : ℕ, 2 ≤ k ∧ k ≤ 99 → a k = a (k - 1) * a (k + 1)

def nonZeroSeq (a : ℕ → ℝ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 100 → a k ≠ 0

def firstSeq7 (a : ℕ → ℝ) : Prop :=
  a 1 = 7

-- Theorem statement
theorem lastNumberIsOneOverSeven (a : ℕ → ℝ) :
  seq a → nonZeroSeq a → firstSeq7 a → a 100 = 1 / 7 :=
by
  sorry

end NUMINAMATH_GPT_lastNumberIsOneOverSeven_l1860_186053


namespace NUMINAMATH_GPT_sum_is_correct_l1860_186060

theorem sum_is_correct (a b c d : ℤ) 
  (h : a + 1 = b + 2 ∧ b + 2 = c + 3 ∧ c + 3 = d + 4 ∧ d + 4 = a + b + c + d + 7) : 
  a + b + c + d = -6 := 
by 
  sorry

end NUMINAMATH_GPT_sum_is_correct_l1860_186060


namespace NUMINAMATH_GPT_grilled_cheese_sandwiches_l1860_186058

-- Define the number of ham sandwiches Joan makes
def ham_sandwiches := 8

-- Define the cheese requirements for each type of sandwich
def cheddar_for_ham := 1
def swiss_for_ham := 1
def cheddar_for_grilled := 2
def gouda_for_grilled := 1

-- Total cheese used
def total_cheddar := 40
def total_swiss := 20
def total_gouda := 30

-- Prove the number of grilled cheese sandwiches Joan makes
theorem grilled_cheese_sandwiches (ham_sandwiches : ℕ) (cheddar_for_ham : ℕ) (swiss_for_ham : ℕ)
                                  (cheddar_for_grilled : ℕ) (gouda_for_grilled : ℕ)
                                  (total_cheddar : ℕ) (total_swiss : ℕ) (total_gouda : ℕ) :
    (total_cheddar - ham_sandwiches * cheddar_for_ham) / cheddar_for_grilled = 16 :=
by
  sorry

end NUMINAMATH_GPT_grilled_cheese_sandwiches_l1860_186058


namespace NUMINAMATH_GPT_anoop_joined_after_6_months_l1860_186095

/- Conditions -/
def arjun_investment : ℕ := 20000
def arjun_months : ℕ := 12
def anoop_investment : ℕ := 40000

/- Main theorem -/
theorem anoop_joined_after_6_months (x : ℕ) (h : arjun_investment * arjun_months = anoop_investment * (arjun_months - x)) : 
  x = 6 :=
sorry

end NUMINAMATH_GPT_anoop_joined_after_6_months_l1860_186095


namespace NUMINAMATH_GPT_jessies_original_weight_l1860_186025

theorem jessies_original_weight (current_weight weight_lost original_weight : ℕ) 
  (h_current: current_weight = 27) (h_lost: weight_lost = 101) 
  (h_original: original_weight = current_weight + weight_lost) : 
  original_weight = 128 :=
by
  rw [h_current, h_lost] at h_original
  exact h_original

end NUMINAMATH_GPT_jessies_original_weight_l1860_186025


namespace NUMINAMATH_GPT_belongs_to_one_progression_l1860_186054

-- Define the arithmetic progression and membership property
def is_arith_prog (P : ℕ → Prop) : Prop :=
  ∃ a d, ∀ n, P (a + n * d)

-- Define the given conditions
def condition (P1 P2 P3 : ℕ → Prop) : Prop :=
  is_arith_prog P1 ∧ is_arith_prog P2 ∧ is_arith_prog P3 ∧
  (P1 1 ∨ P2 1 ∨ P3 1) ∧
  (P1 2 ∨ P2 2 ∨ P3 2) ∧
  (P1 3 ∨ P2 3 ∨ P3 3) ∧
  (P1 4 ∨ P2 4 ∨ P3 4) ∧
  (P1 5 ∨ P2 5 ∨ P3 5) ∧
  (P1 6 ∨ P2 6 ∨ P3 6) ∧
  (P1 7 ∨ P2 7 ∨ P3 7) ∧
  (P1 8 ∨ P2 8 ∨ P3 8)

-- Statement to prove
theorem belongs_to_one_progression (P1 P2 P3 : ℕ → Prop) (h : condition P1 P2 P3) : 
  P1 1980 ∨ P2 1980 ∨ P3 1980 := 
by
sorry

end NUMINAMATH_GPT_belongs_to_one_progression_l1860_186054


namespace NUMINAMATH_GPT_largest_divisible_by_3_power_l1860_186020

theorem largest_divisible_by_3_power :
  ∃ n : ℕ, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 100 → ∃ m : ℕ, (3^m ∣ (2*k - 1)) → n = 49) :=
sorry

end NUMINAMATH_GPT_largest_divisible_by_3_power_l1860_186020


namespace NUMINAMATH_GPT_greatest_value_of_x_l1860_186026

theorem greatest_value_of_x
  (x : ℕ)
  (h1 : x % 4 = 0) -- x is a multiple of 4
  (h2 : x > 0) -- x is positive
  (h3 : x^3 < 2000) -- x^3 < 2000
  : x ≤ 12 :=
by
  sorry

end NUMINAMATH_GPT_greatest_value_of_x_l1860_186026


namespace NUMINAMATH_GPT_f_periodic_4_l1860_186004

noncomputable def f : ℝ → ℝ := sorry -- f is some function ℝ → ℝ

theorem f_periodic_4 (h : ∀ x, f x = -f (x + 2)) : f 100 = f 4 := 
by
  sorry

end NUMINAMATH_GPT_f_periodic_4_l1860_186004


namespace NUMINAMATH_GPT_find_f_2_l1860_186002

noncomputable def f (a b x : ℝ) : ℝ := a * x^5 + b * x^3 - x + 2

theorem find_f_2 (a b : ℝ)
  (h : f a b (-2) = 5) : f a b 2 = -1 :=
by 
  sorry

end NUMINAMATH_GPT_find_f_2_l1860_186002


namespace NUMINAMATH_GPT_sum_not_fourteen_l1860_186017

theorem sum_not_fourteen (a b c d : ℕ) (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) (hc : 1 ≤ c ∧ c ≤ 6) (hd : 1 ≤ d ∧ d ≤ 6) 
  (hprod : a * b * c * d = 120) : a + b + c + d ≠ 14 :=
sorry

end NUMINAMATH_GPT_sum_not_fourteen_l1860_186017


namespace NUMINAMATH_GPT_number_of_real_roots_l1860_186075

noncomputable def f (x : ℝ) : ℝ := x^3 - x

theorem number_of_real_roots (a : ℝ) :
    ((|a| < (2 * Real.sqrt 3) / 9) → (∃ x₁ x₂ x₃ : ℝ, f x₁ = a ∧ f x₂ = a ∧ f x₃ = a ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)) ∧
    ((|a| > (2 * Real.sqrt 3) / 9) → (∃ x : ℝ, f x = a ∧ ∀ y : ℝ, f y = a → y = x)) ∧
    ((|a| = (2 * Real.sqrt 3) / 9) → (∃ x₁ x₂ : ℝ, f x₁ = a ∧ f x₂ = a ∧ x₁ ≠ x₂ ∧ ∀ y : ℝ, (f y = a → (y = x₁ ∨ y = x₂)) ∧ (x₁ = x₂ ∨ ∀ z : ℝ, (f z = a → z = x₁ ∨ z = x₂)))) := sorry

end NUMINAMATH_GPT_number_of_real_roots_l1860_186075


namespace NUMINAMATH_GPT_sum_of_fractions_is_correct_l1860_186036

-- Definitions from the conditions
def half_of_third := (1 : ℚ) / 2 * (1 : ℚ) / 3
def third_of_quarter := (1 : ℚ) / 3 * (1 : ℚ) / 4
def quarter_of_fifth := (1 : ℚ) / 4 * (1 : ℚ) / 5
def sum_fractions := half_of_third + third_of_quarter + quarter_of_fifth

-- The theorem to prove
theorem sum_of_fractions_is_correct : sum_fractions = (3 : ℚ) / 10 := by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_is_correct_l1860_186036


namespace NUMINAMATH_GPT_hexagon_side_length_l1860_186081

-- Define the conditions for the side length of a hexagon where the area equals the perimeter
theorem hexagon_side_length (s : ℝ) (h1 : (3 * Real.sqrt 3 / 2) * s^2 = 6 * s) :
  s = 4 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_GPT_hexagon_side_length_l1860_186081


namespace NUMINAMATH_GPT_sin_double_angle_l1860_186061

theorem sin_double_angle (x : ℝ) (h : Real.sin (π / 4 - x) = 3 / 5) : Real.sin (2 * x) = 7 / 25 := by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l1860_186061


namespace NUMINAMATH_GPT_emily_disproved_jacob_by_turnover_5_and_7_l1860_186082

def is_vowel (c : Char) : Prop :=
  c = 'A'

def is_consonant (c : Char) : Prop :=
  ¬ is_vowel c

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

def card_A_is_vowel : Prop := is_vowel 'A'
def card_1_is_odd : Prop := ¬ is_even 1 ∧ ¬ is_prime 1
def card_8_is_even : Prop := is_even 8 ∧ ¬ is_prime 8
def card_R_is_consonant : Prop := is_consonant 'R'
def card_S_is_consonant : Prop := is_consonant 'S'
def card_5_conditions : Prop := ¬ is_even 5 ∧ is_prime 5
def card_7_conditions : Prop := ¬ is_even 7 ∧ is_prime 7

theorem emily_disproved_jacob_by_turnover_5_and_7 :
  card_5_conditions ∧ card_7_conditions →
  (∃ (c : Char), (is_prime 5 ∧ is_consonant c)) ∨
  (∃ (c : Char), (is_prime 7 ∧ is_consonant c)) :=
by sorry

end NUMINAMATH_GPT_emily_disproved_jacob_by_turnover_5_and_7_l1860_186082


namespace NUMINAMATH_GPT_combination_problem_l1860_186088

noncomputable def combination (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.choose n k

theorem combination_problem (x : ℕ) (h : combination 25 (2 * x) = combination 25 (x + 4)) : x = 4 ∨ x = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_combination_problem_l1860_186088


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l1860_186079

open Set

def M : Set ℝ := {x | x ≥ 2}
def N : Set ℝ := {x | x^2 - 25 < 0}
def I : Set ℝ := {x | 2 ≤ x ∧ x < 5}

theorem intersection_of_M_and_N : M ∩ N = I := by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l1860_186079


namespace NUMINAMATH_GPT_Matt_income_from_plantation_l1860_186064

noncomputable def plantation_income :=
  let plantation_area := 500 * 500  -- square feet
  let grams_peanuts_per_sq_ft := 50 -- grams
  let grams_peanut_butter_per_20g_peanuts := 5  -- grams
  let price_per_kg_peanut_butter := 10 -- $

  -- Total revenue calculation
  plantation_area * grams_peanuts_per_sq_ft * grams_peanut_butter_per_20g_peanuts /
  20 / 1000 * price_per_kg_peanut_butter

theorem Matt_income_from_plantation :
  plantation_income = 31250 := sorry

end NUMINAMATH_GPT_Matt_income_from_plantation_l1860_186064


namespace NUMINAMATH_GPT_proof_statement_B_proof_statement_D_proof_statement_E_l1860_186078

def statement_B (x : ℝ) : Prop := x^2 = 0 → x = 0

def statement_D (x : ℝ) : Prop := x^2 < 2 * x → x > 0

def statement_E (x : ℝ) : Prop := x > 2 → x^2 > x

theorem proof_statement_B (x : ℝ) : statement_B x := sorry

theorem proof_statement_D (x : ℝ) : statement_D x := sorry

theorem proof_statement_E (x : ℝ) : statement_E x := sorry

end NUMINAMATH_GPT_proof_statement_B_proof_statement_D_proof_statement_E_l1860_186078


namespace NUMINAMATH_GPT_sum_of_a_and_b_l1860_186070

theorem sum_of_a_and_b (a b : ℕ) (h1 : a > 0) (h2 : b > 1) (h3 : a^b < 500) (h_max : ∀ (a' b' : ℕ), a' > 0 → b' > 1 → a'^b' < 500 → a'^b' ≤ a^b) : a + b = 24 :=
sorry

end NUMINAMATH_GPT_sum_of_a_and_b_l1860_186070


namespace NUMINAMATH_GPT_range_of_a_l1860_186052

noncomputable def f (x a : ℝ) : ℝ := -x^3 + 3 * x + a

theorem range_of_a (a : ℝ) :
  (∃ (m n p : ℝ), m ≠ n ∧ n ≠ p ∧ m ≠ p ∧ f m a = 2024 ∧ f n a = 2024 ∧ f p a = 2024) ↔
  2022 < a ∧ a < 2026 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1860_186052


namespace NUMINAMATH_GPT_find_fx_neg_l1860_186038

variable (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def f_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 0 → f x = x^2 - 2*x

theorem find_fx_neg (h1 : odd_function f) (h2 : f_nonneg f) : 
  ∀ x : ℝ, x < 0 → f x = -x^2 - 2*x := 
by
  sorry

end NUMINAMATH_GPT_find_fx_neg_l1860_186038


namespace NUMINAMATH_GPT_cylindrical_coordinates_cone_shape_l1860_186018

def cylindrical_coordinates := Type

def shape_description (r θ z : ℝ) : Prop :=
θ = 2 * z

theorem cylindrical_coordinates_cone_shape (r θ z : ℝ) :
  shape_description r θ z → θ = 2 * z → Prop := sorry

end NUMINAMATH_GPT_cylindrical_coordinates_cone_shape_l1860_186018


namespace NUMINAMATH_GPT_negation_of_P_l1860_186073

def P : Prop := ∃ x_0 : ℝ, x_0^2 + 2 * x_0 + 2 ≤ 0

theorem negation_of_P : ¬ P ↔ ∀ x : ℝ, x^2 + 2 * x + 2 > 0 :=
by sorry

end NUMINAMATH_GPT_negation_of_P_l1860_186073


namespace NUMINAMATH_GPT_solve_quintic_equation_l1860_186024

theorem solve_quintic_equation :
  {x : ℝ | x * (x - 3)^2 * (5 + x) * (x^2 - 1) = 0} = {0, 3, -5, 1, -1} :=
by
  sorry

end NUMINAMATH_GPT_solve_quintic_equation_l1860_186024


namespace NUMINAMATH_GPT_linear_system_incorrect_statement_l1860_186043

def is_determinant (a b c d : ℝ) := a * d - b * c

def is_solution_system (a1 b1 c1 a2 b2 c2 D Dx Dy : ℝ) :=
  D = is_determinant a1 b1 a2 b2 ∧
  Dx = is_determinant c1 b1 c2 b2 ∧
  Dy = is_determinant a1 c1 a2 c2

def is_solution_linear_system (a1 b1 c1 a2 b2 c2 x y : ℝ) :=
  a1 * x + b1 * y = c1 ∧ a2 * x + b2 * y = c2

theorem linear_system_incorrect_statement :
  ∀ (x y : ℝ),
    is_solution_system 3 (-1) 1 1 3 7 10 10 20 ∧
    is_solution_linear_system 3 (-1) 1 1 3 7 x y →
    x = 1 ∧ y = 2 ∧ ¬(20 = -20) := 
by sorry

end NUMINAMATH_GPT_linear_system_incorrect_statement_l1860_186043


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1860_186021

variables (a : ℕ → ℤ) (q : ℤ)

-- assumption that the sequence is geometric
def geometric_sequence (a : ℕ → ℤ) (q : ℤ) : Prop := 
  ∀ n, a (n + 1) = a n * q

noncomputable def a2 := a 2
noncomputable def a3 := a 3
noncomputable def a4 := a 4
noncomputable def a5 := a 5
noncomputable def a6 := a 6
noncomputable def a7 := a 7

theorem geometric_sequence_sum
  (h_geom : geometric_sequence a q)
  (h1 : a2 + a3 = 1)
  (h2 : a3 + a4 = -2) :
  a5 + a6 + a7 = 24 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1860_186021


namespace NUMINAMATH_GPT_circle_condition_l1860_186098

theorem circle_condition (m : ℝ): (∃ x y : ℝ, (x^2 + y^2 - 2*x - 4*y + m = 0)) ↔ (m < 5) :=
by
  sorry

end NUMINAMATH_GPT_circle_condition_l1860_186098


namespace NUMINAMATH_GPT_prime_factorization_2020_prime_factorization_2021_l1860_186094

theorem prime_factorization_2020 : 2020 = 2^2 * 5 * 101 := by
  sorry

theorem prime_factorization_2021 : 2021 = 43 * 47 := by
  sorry

end NUMINAMATH_GPT_prime_factorization_2020_prime_factorization_2021_l1860_186094


namespace NUMINAMATH_GPT_find_larger_number_l1860_186068

variable (x y : ℝ)
axiom h1 : x + y = 27
axiom h2 : x - y = 5

theorem find_larger_number : x = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_larger_number_l1860_186068


namespace NUMINAMATH_GPT_allocation_of_fabric_l1860_186034

theorem allocation_of_fabric (x : ℝ) (y : ℝ) 
  (fabric_for_top : 3 * x = 2 * x)
  (fabric_for_pants : 3 * y = 3 * (600 - x))
  (total_fabric : x + y = 600)
  (sets_match : (x / 3) * 2 = (y / 3) * 3) : 
  x = 360 ∧ y = 240 := 
by
  sorry

end NUMINAMATH_GPT_allocation_of_fabric_l1860_186034


namespace NUMINAMATH_GPT_find_triangle_side1_l1860_186029

def triangle_side1 (Perimeter Side2 Side3 Side1 : ℕ) : Prop :=
  Perimeter = Side1 + Side2 + Side3

theorem find_triangle_side1 :
  ∀ (Perimeter Side2 Side3 Side1 : ℕ), 
    (Perimeter = 160) → (Side2 = 50) → (Side3 = 70) → triangle_side1 Perimeter Side2 Side3 Side1 → Side1 = 40 :=
by
  intros Perimeter Side2 Side3 Side1 h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_find_triangle_side1_l1860_186029


namespace NUMINAMATH_GPT_min_max_abs_poly_eq_zero_l1860_186087

theorem min_max_abs_poly_eq_zero :
  ∃ y : ℝ, (∀ x : ℝ, 0 ≤ x → x ≤ 1 → |x^2 - x^3 * y| ≤ 0) :=
sorry

end NUMINAMATH_GPT_min_max_abs_poly_eq_zero_l1860_186087


namespace NUMINAMATH_GPT_increase_in_average_weight_l1860_186063

variable {A X : ℝ}

-- Given initial conditions
axiom average_initial_weight_8 : X = (8 * A - 62 + 90) / 8 - A

-- The goal to prove
theorem increase_in_average_weight : X = 3.5 :=
by
  sorry

end NUMINAMATH_GPT_increase_in_average_weight_l1860_186063


namespace NUMINAMATH_GPT_sign_of_ac_l1860_186015

theorem sign_of_ac (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) (h : (a / b) + (c / d) = (a + c) / (b + d)) : a * c < 0 :=
by
  sorry

end NUMINAMATH_GPT_sign_of_ac_l1860_186015


namespace NUMINAMATH_GPT_systematic_sampling_distance_l1860_186092

-- Conditions
def total_students : ℕ := 1200
def sample_size : ℕ := 30

-- Problem: Compute sampling distance
def sampling_distance (n : ℕ) (m : ℕ) : ℕ := n / m

-- The formal proof statement
theorem systematic_sampling_distance :
  sampling_distance total_students sample_size = 40 := by
  sorry

end NUMINAMATH_GPT_systematic_sampling_distance_l1860_186092


namespace NUMINAMATH_GPT_gcd_g10_g13_l1860_186007

-- Define the polynomial function g
def g (x : ℤ) : ℤ := x^3 - 3 * x^2 + x + 2050

-- State the theorem to prove that gcd(g(10), g(13)) is 1
theorem gcd_g10_g13 : Int.gcd (g 10) (g 13) = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_g10_g13_l1860_186007


namespace NUMINAMATH_GPT_find_c_l1860_186086

theorem find_c (x y c : ℝ) (h : x = 5 * y) (h2 : 7 * x + 4 * y = 13 * c) : c = 3 * y :=
by
  sorry

end NUMINAMATH_GPT_find_c_l1860_186086


namespace NUMINAMATH_GPT_maria_workday_end_l1860_186042

def time_in_minutes (h : ℕ) (m : ℕ) : ℕ := h * 60 + m

def start_time : ℕ := time_in_minutes 7 25
def lunch_break : ℕ := 45
def noon : ℕ := time_in_minutes 12 0
def work_hours : ℕ := 8 * 60
def end_time : ℕ := time_in_minutes 16 10

theorem maria_workday_end : start_time + (noon - start_time) + lunch_break + (work_hours - (noon - start_time)) = end_time := by
  sorry

end NUMINAMATH_GPT_maria_workday_end_l1860_186042


namespace NUMINAMATH_GPT_bicycle_helmet_lock_costs_l1860_186028

-- Given total cost, relationships between costs, and the specific costs
theorem bicycle_helmet_lock_costs (H : ℝ) (bicycle helmet lock : ℝ) 
  (h1 : bicycle = 5 * H) 
  (h2 : helmet = H) 
  (h3 : lock = H / 2)
  (total_cost : bicycle + helmet + lock = 360) :
  H = 55.38 ∧ bicycle = 276.90 ∧ lock = 27.72 :=
by 
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_bicycle_helmet_lock_costs_l1860_186028


namespace NUMINAMATH_GPT_amelia_wins_probability_l1860_186010

def amelia_prob_heads : ℚ := 1 / 4
def blaine_prob_heads : ℚ := 3 / 7

def probability_blaine_wins_first_turn : ℚ := blaine_prob_heads

def probability_amelia_wins_first_turn : ℚ :=
  (1 - blaine_prob_heads) * amelia_prob_heads

def probability_amelia_wins_second_turn : ℚ :=
  (1 - blaine_prob_heads) * (1 - amelia_prob_heads) * (1 - blaine_prob_heads) * amelia_prob_heads

def probability_amelia_wins_third_turn : ℚ :=
  (1 - blaine_prob_heads) * (1 - amelia_prob_heads) * (1 - blaine_prob_heads) * 
  (1 - amelia_prob_heads) * (1 - blaine_prob_heads) * amelia_prob_heads

def probability_amelia_wins : ℚ :=
  probability_amelia_wins_first_turn + probability_amelia_wins_second_turn + probability_amelia_wins_third_turn

theorem amelia_wins_probability : probability_amelia_wins = 223 / 784 := by
  sorry

end NUMINAMATH_GPT_amelia_wins_probability_l1860_186010


namespace NUMINAMATH_GPT_vector_equation_l1860_186048

noncomputable def vec_a : (ℝ × ℝ) := (1, -1)
noncomputable def vec_b : (ℝ × ℝ) := (2, 1)
noncomputable def vec_c : (ℝ × ℝ) := (-2, 1)

theorem vector_equation (x y : ℝ) 
  (h : vec_c = (x * vec_a.1 + y * vec_b.1, x * vec_a.2 + y * vec_b.2)) : 
  x - y = -1 := 
by { sorry }

end NUMINAMATH_GPT_vector_equation_l1860_186048


namespace NUMINAMATH_GPT_cos_product_identity_l1860_186050

noncomputable def L : ℝ := 3.418 * (Real.cos (2 * Real.pi / 31)) *
                               (Real.cos (4 * Real.pi / 31)) *
                               (Real.cos (8 * Real.pi / 31)) *
                               (Real.cos (16 * Real.pi / 31)) *
                               (Real.cos (32 * Real.pi / 31))

theorem cos_product_identity : L = 1 / 32 := by
  sorry

end NUMINAMATH_GPT_cos_product_identity_l1860_186050
