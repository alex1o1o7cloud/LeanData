import Mathlib

namespace NUMINAMATH_GPT_total_points_l1037_103758

def points_earned (goblins orcs dragons : ℕ): ℕ :=
  goblins * 3 + orcs * 5 + dragons * 10

theorem total_points :
  points_earned 10 7 1 = 75 :=
by
  sorry

end NUMINAMATH_GPT_total_points_l1037_103758


namespace NUMINAMATH_GPT_amy_balloons_l1037_103705

theorem amy_balloons (james_balloons amy_balloons : ℕ) (h1 : james_balloons = 232) (h2 : james_balloons = amy_balloons + 131) :
  amy_balloons = 101 :=
by
  sorry

end NUMINAMATH_GPT_amy_balloons_l1037_103705


namespace NUMINAMATH_GPT_algebra_minimum_value_l1037_103732

theorem algebra_minimum_value :
  ∀ x y : ℝ, ∃ m : ℝ, (∀ x y : ℝ, x^2 + y^2 + 6*x - 2*y + 12 ≥ m) ∧ m = 2 :=
by
  sorry

end NUMINAMATH_GPT_algebra_minimum_value_l1037_103732


namespace NUMINAMATH_GPT_distance_last_day_l1037_103784

theorem distance_last_day
  (total_distance : ℕ)
  (days : ℕ)
  (initial_distance : ℕ)
  (common_ratio : ℚ)
  (sum_geometric : initial_distance * (1 - common_ratio^days) / (1 - common_ratio) = total_distance) :
  total_distance = 378 → days = 6 → common_ratio = 1/2 → 
  initial_distance = 192 → initial_distance * common_ratio^(days - 1) = 6 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_distance_last_day_l1037_103784


namespace NUMINAMATH_GPT_percentage_increase_of_cube_surface_area_l1037_103778

-- Basic setup definitions and conditions
variable (a : ℝ)

-- Step 1: Initial surface area
def initial_surface_area : ℝ := 6 * a^2

-- Step 2: New edge length after 50% growth
def new_edge_length : ℝ := 1.5 * a

-- Step 3: New surface area after edge growth
def new_surface_area : ℝ := 6 * (new_edge_length a)^2

-- Step 4: Surface area after scaling by 1.5
def scaled_surface_area : ℝ := new_surface_area a * (1.5)^2

-- Prove the percentage increase
theorem percentage_increase_of_cube_surface_area :
  (scaled_surface_area a - initial_surface_area a) / initial_surface_area a * 100 = 406.25 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_of_cube_surface_area_l1037_103778


namespace NUMINAMATH_GPT_determinant_expression_l1037_103717

theorem determinant_expression (a b c d p q r : ℝ)
  (h1: (∃ x: ℝ, x^4 + p*x^2 + q*x + r = 0) → (x = a ∨ x = b ∨ x = c ∨ x = d))
  (h2: a*b + a*c + a*d + b*c + b*d + c*d = p)
  (h3: a*b*c + a*b*d + a*c*d + b*c*d = q)
  (h4: a*b*c*d = -r):
  (Matrix.det ![![1 + a, 1, 1, 1], ![1, 1 + b, 1, 1], ![1, 1, 1 + c, 1], ![1, 1, 1, 1 + d]]) 
  = r + q + p := 
sorry

end NUMINAMATH_GPT_determinant_expression_l1037_103717


namespace NUMINAMATH_GPT_b_over_c_equals_1_l1037_103796

theorem b_over_c_equals_1 (a b c d : ℕ) (ha : a < 4) (hb : b < 4) (hc : c < 4) (hd : d < 4)
    (h : 4^a + 3^b + 2^c + 1^d = 78) : b = c :=
by
  sorry

end NUMINAMATH_GPT_b_over_c_equals_1_l1037_103796


namespace NUMINAMATH_GPT_simplify_expression_l1037_103790

theorem simplify_expression (a b : ℝ) :
  3 * a - 4 * b + 2 * a^2 - (7 * a - 2 * a^2 + 3 * b - 5) = -4 * a - 7 * b + 4 * a^2 + 5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1037_103790


namespace NUMINAMATH_GPT_number_of_4_letter_words_with_B_l1037_103797

-- Define the set of letters.
inductive Alphabet
| A | B | C | D | E

-- The number of 4-letter words with repetition allowed and must include 'B' at least once.
noncomputable def words_with_at_least_one_B : ℕ :=
  let total := 5 ^ 4 -- Total number of 4-letter words.
  let without_B := 4 ^ 4 -- Total number of 4-letter words without 'B'.
  total - without_B

-- The main theorem statement.
theorem number_of_4_letter_words_with_B : words_with_at_least_one_B = 369 :=
  by sorry

end NUMINAMATH_GPT_number_of_4_letter_words_with_B_l1037_103797


namespace NUMINAMATH_GPT_binom_coeffs_not_coprime_l1037_103793

open Nat

theorem binom_coeffs_not_coprime (n k m : ℕ) (h1 : 0 < k) (h2 : k < m) (h3 : m < n) : 
  Nat.gcd (Nat.choose n k) (Nat.choose n m) > 1 := 
sorry

end NUMINAMATH_GPT_binom_coeffs_not_coprime_l1037_103793


namespace NUMINAMATH_GPT_simplify_expression_eq_69_l1037_103787

theorem simplify_expression_eq_69 : 80 - (5 - (6 + 2 * (7 - 8 - 5))) = 69 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_eq_69_l1037_103787


namespace NUMINAMATH_GPT_rectangle_shaded_area_equal_l1037_103760

theorem rectangle_shaded_area_equal {x : ℝ} :
  let total_area := 72
  let shaded_area := 24 + 6*x
  let non_shaded_area := total_area / 2
  shaded_area = non_shaded_area → x = 2 := 
by 
  intros h
  sorry

end NUMINAMATH_GPT_rectangle_shaded_area_equal_l1037_103760


namespace NUMINAMATH_GPT_total_cases_after_three_days_l1037_103731

def initial_cases : ℕ := 2000
def increase_rate : ℝ := 0.20
def recovery_rate : ℝ := 0.02

def day_cases (n : ℕ) : ℝ :=
  match n with
  | 0 => initial_cases
  | n + 1 => 
      let prev_cases := day_cases n
      let new_cases := increase_rate * prev_cases
      let recovered := recovery_rate * prev_cases
      prev_cases + new_cases - recovered

theorem total_cases_after_three_days : day_cases 3 = 3286 := by sorry

end NUMINAMATH_GPT_total_cases_after_three_days_l1037_103731


namespace NUMINAMATH_GPT_polygon_sides_l1037_103772

theorem polygon_sides :
  ∃ (n : ℕ), (n * (n - 3) / 2) = n + 33 ∧ n = 11 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l1037_103772


namespace NUMINAMATH_GPT_circle_n_gon_area_ineq_l1037_103786

variable {n : ℕ} {S S1 S2 : ℝ}

theorem circle_n_gon_area_ineq (h1 : S1 > 0) (h2 : S > 0) (h3 : S2 > 0) : 
  S * S = S1 * S2 := 
sorry

end NUMINAMATH_GPT_circle_n_gon_area_ineq_l1037_103786


namespace NUMINAMATH_GPT_interior_diagonal_length_l1037_103714

theorem interior_diagonal_length (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 26)
  (h2 : 4 * (a + b + c) = 28) : 
  (a^2 + b^2 + c^2) = 23 :=
by
  sorry

end NUMINAMATH_GPT_interior_diagonal_length_l1037_103714


namespace NUMINAMATH_GPT_cube_skew_lines_l1037_103788

theorem cube_skew_lines (cube : Prop) (diagonal : Prop) (edges : Prop) :
  ( ∃ n : ℕ, n = 6 ) :=
by
  sorry

end NUMINAMATH_GPT_cube_skew_lines_l1037_103788


namespace NUMINAMATH_GPT_find_f1_plus_g1_l1037_103789

-- Definition of f being an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = f x

-- Definition of g being an odd function
def is_odd_function (g : ℝ → ℝ) : Prop :=
∀ x : ℝ, g (-x) = -g x

-- Statement of the proof problem
theorem find_f1_plus_g1 
  (f g : ℝ → ℝ) 
  (hf : is_even_function f) 
  (hg : is_odd_function g)
  (hfg : ∀ x : ℝ, f x - g x = x^3 + x^2 + 1) : f 1 + g 1 = 2 :=
sorry

end NUMINAMATH_GPT_find_f1_plus_g1_l1037_103789


namespace NUMINAMATH_GPT_g_three_eighths_l1037_103770

variable (g : ℝ → ℝ)

-- Conditions
axiom g_zero : g 0 = 0
axiom monotonic : ∀ {x y : ℝ}, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y
axiom symmetry : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x
axiom scaling : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3

-- The theorem statement we need to prove
theorem g_three_eighths : g (3 / 8) = 2 / 9 :=
sorry

end NUMINAMATH_GPT_g_three_eighths_l1037_103770


namespace NUMINAMATH_GPT_find_y_coordinate_l1037_103779

-- Define points A, B, C, and D
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (-2, 2)
def C : ℝ × ℝ := (2, 2)
def D : ℝ × ℝ := (4, 0)

-- Define the property that a point P satisfies PA + PD = PB + PC = 10
def satisfies_condition (P : ℝ × ℝ) : Prop :=
  let PA := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
  let PD := Real.sqrt ((P.1 - D.1)^2 + (P.2 - D.2)^2)
  let PB := Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)
  let PC := Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2)
  PA + PD = 10 ∧ PB + PC = 10

-- Lean statement to prove the y-coordinate of P that satisfies the condition
theorem find_y_coordinate :
  ∃ (P : ℝ × ℝ), satisfies_condition P ∧ ∃ (a b c d : ℕ), a = 0 ∧ b = 1 ∧ c = 21 ∧ d = 3 ∧ P.2 = (14 + Real.sqrt 21) / 3 ∧ a + b + c + d = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_y_coordinate_l1037_103779


namespace NUMINAMATH_GPT_value_of_g_13_l1037_103762

def g (n : ℕ) : ℕ := n^2 + 2 * n + 23

theorem value_of_g_13 : g 13 = 218 :=
by 
  sorry

end NUMINAMATH_GPT_value_of_g_13_l1037_103762


namespace NUMINAMATH_GPT_alpha_proportional_l1037_103754

theorem alpha_proportional (alpha beta gamma : ℝ) (h1 : ∀ β γ, (β = 15 ∧ γ = 3) → α = 5)
    (h2 : beta = 30) (h3 : gamma = 6) : alpha = 2.5 :=
sorry

end NUMINAMATH_GPT_alpha_proportional_l1037_103754


namespace NUMINAMATH_GPT_number_of_participants_l1037_103764

theorem number_of_participants (n : ℕ) (hn : n = 862) 
    (h_lower : 575 ≤ n * 2 / 3) 
    (h_upper : n * 7 / 9 ≤ 670) : 
    ∃ p, (575 ≤ p) ∧ (p ≤ 670) ∧ (p % 11 = 0) ∧ ((p - 575) / 11 + 1 = 8) :=
by
  sorry

end NUMINAMATH_GPT_number_of_participants_l1037_103764


namespace NUMINAMATH_GPT_chantel_bracelets_final_count_l1037_103748

-- Definitions for conditions
def bracelets_made_days (days : ℕ) (bracelets_per_day : ℕ) : ℕ :=
  days * bracelets_per_day

def initial_bracelets (days1 : ℕ) (bracelets_per_day1 : ℕ) : ℕ :=
  bracelets_made_days days1 bracelets_per_day1

def after_giving_away1 (initial_count : ℕ) (given_away1 : ℕ) : ℕ :=
  initial_count - given_away1

def additional_bracelets (days2 : ℕ) (bracelets_per_day2 : ℕ) : ℕ :=
  bracelets_made_days days2 bracelets_per_day2

def final_count (remaining_after_giving1 : ℕ) (additional_made : ℕ) (given_away2 : ℕ) : ℕ :=
  remaining_after_giving1 + additional_made - given_away2

-- Main theorem statement
theorem chantel_bracelets_final_count :
  ∀ (days1 days2 bracelets_per_day1 bracelets_per_day2 given_away1 given_away2 : ℕ),
  days1 = 5 →
  bracelets_per_day1 = 2 →
  given_away1 = 3 →
  days2 = 4 →
  bracelets_per_day2 = 3 →
  given_away2 = 6 →
  final_count (after_giving_away1 (initial_bracelets days1 bracelets_per_day1) given_away1)
              (additional_bracelets days2 bracelets_per_day2)
              given_away2 = 13 :=
by
  intros days1 days2 bracelets_per_day1 bracelets_per_day2 given_away1 given_away2 hdays1 hbracelets_per_day1 hgiven_away1 hdays2 hbracelets_per_day2 hgiven_away2
  -- Proof is not required, so we use sorry
  sorry

end NUMINAMATH_GPT_chantel_bracelets_final_count_l1037_103748


namespace NUMINAMATH_GPT_remainder_g10_div_g_l1037_103766

-- Conditions/Definitions
def g (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1
def g10 (x : ℝ) : ℝ := (g (x^10))

-- Theorem/Question
theorem remainder_g10_div_g : (g10 x) % (g x) = 6 :=
by
  sorry

end NUMINAMATH_GPT_remainder_g10_div_g_l1037_103766


namespace NUMINAMATH_GPT_shobha_current_age_l1037_103706

theorem shobha_current_age (S B : ℕ) (h1 : S / B = 4 / 3) (h2 : S + 6 = 26) : B = 15 :=
by
  -- Here we would begin the proof
  sorry

end NUMINAMATH_GPT_shobha_current_age_l1037_103706


namespace NUMINAMATH_GPT_frictional_force_is_12N_l1037_103739

-- Given conditions
variables (m1 m2 a μ : ℝ)
-- Constants
def g : ℝ := 9.8

-- Frictional force on the tank
def F_friction : ℝ := μ * m1 * g

-- Proof statement
theorem frictional_force_is_12N (m1_value : m1 = 3) (m2_value : m2 = 15) (a_value : a = 4) (μ_value : μ = 0.6) :
  m1 * a = 12 :=
by
  sorry

end NUMINAMATH_GPT_frictional_force_is_12N_l1037_103739


namespace NUMINAMATH_GPT_y2_minus_x2_l1037_103713

theorem y2_minus_x2 (x y : ℕ) (hx_pos : x > 0) (hy_pos : y > 0) (h1 : 56 ≤ x + y) (h2 : x + y ≤ 59) (h3 : 9 < 10 * x) (h4 : 10 * x < 91 * y) : y^2 - x^2 = 177 :=
by
  sorry

end NUMINAMATH_GPT_y2_minus_x2_l1037_103713


namespace NUMINAMATH_GPT_dan_minimum_speed_to_beat_cara_l1037_103776

theorem dan_minimum_speed_to_beat_cara
  (distance : ℕ) (cara_speed : ℕ) (dan_delay : ℕ) :
  distance = 120 →
  cara_speed = 30 →
  dan_delay = 1 →
  ∃ (dan_speed : ℕ), dan_speed > 40 :=
by
  sorry

end NUMINAMATH_GPT_dan_minimum_speed_to_beat_cara_l1037_103776


namespace NUMINAMATH_GPT_imo1987_q6_l1037_103783

theorem imo1987_q6 (m n : ℤ) (h : n = m + 2) :
  ⌊(n : ℝ) * Real.sqrt 2⌋ = 2 + ⌊(m : ℝ) * Real.sqrt 2⌋ := 
by
  sorry -- We skip the detailed proof steps here.

end NUMINAMATH_GPT_imo1987_q6_l1037_103783


namespace NUMINAMATH_GPT_part1_part2_l1037_103719

theorem part1 (x y : ℝ) (h1 : y = x + 30) (h2 : 2 * x + 3 * y = 340) : x = 50 ∧ y = 80 :=
by {
  -- Later, we can place the steps to prove x = 50 and y = 80 here.
  sorry
}

theorem part2 (m : ℕ) (h3 : 0 ≤ m ∧ m ≤ 50)
               (h4 : 54 * (50 - m) + 72 * m = 3060) : m = 20 :=
by {
  -- Later, we can place the steps to prove m = 20 here.
  sorry
}

end NUMINAMATH_GPT_part1_part2_l1037_103719


namespace NUMINAMATH_GPT_simplify_tan_cot_fraction_l1037_103722

theorem simplify_tan_cot_fraction :
  let tan45 := 1
  let cot45 := 1
  (tan45^3 + cot45^3) / (tan45 + cot45) = 1 := by
    sorry

end NUMINAMATH_GPT_simplify_tan_cot_fraction_l1037_103722


namespace NUMINAMATH_GPT_Mel_weight_is_70_l1037_103721

-- Definitions and conditions
def MelWeight (M : ℕ) :=
  3 * M + 10

theorem Mel_weight_is_70 (M : ℕ) (h1 : 3 * M + 10 = 220) :
  M = 70 :=
by
  sorry

end NUMINAMATH_GPT_Mel_weight_is_70_l1037_103721


namespace NUMINAMATH_GPT_ordered_concrete_weight_l1037_103765

def weight_of_materials : ℝ := 0.83
def weight_of_bricks : ℝ := 0.17
def weight_of_stone : ℝ := 0.5

theorem ordered_concrete_weight :
  weight_of_materials - (weight_of_bricks + weight_of_stone) = 0.16 := by
  sorry

end NUMINAMATH_GPT_ordered_concrete_weight_l1037_103765


namespace NUMINAMATH_GPT_joan_gave_27_apples_l1037_103709

theorem joan_gave_27_apples (total_apples : ℕ) (current_apples : ℕ)
  (h1 : total_apples = 43) 
  (h2 : current_apples = 16) : 
  total_apples - current_apples = 27 := 
by
  sorry

end NUMINAMATH_GPT_joan_gave_27_apples_l1037_103709


namespace NUMINAMATH_GPT_solve_problem_for_m_n_l1037_103795

theorem solve_problem_for_m_n (m n : ℕ) (h₀ : m > 0) (h₁ : n > 0) (h₂ : m * (n + m) = n * (n - m)) :
  ((∃ h : ℕ, m = (2 * h + 1) * h ∧ n = (2 * h + 1) * (h + 1)) ∨ 
   (∃ h : ℕ, h > 0 ∧ m = 2 * h * (4 * h^2 - 1) ∧ n = 2 * h * (4 * h^2 + 1))) := 
sorry

end NUMINAMATH_GPT_solve_problem_for_m_n_l1037_103795


namespace NUMINAMATH_GPT_integral_solutions_count_l1037_103798

theorem integral_solutions_count (m : ℕ) (h : m > 0) :
  ∃ S : Finset (ℕ × ℕ), S.card = m ∧ 
  ∀ (p : ℕ × ℕ), p ∈ S → (p.1^2 + p.2^2 + 2 * p.1 * p.2 - m * p.1 - m * p.2 - m - 1 = 0) := 
sorry

end NUMINAMATH_GPT_integral_solutions_count_l1037_103798


namespace NUMINAMATH_GPT_find_distance_between_posters_and_wall_l1037_103720

-- Definitions for given conditions
def poster_width : ℝ := 29.05
def num_posters : ℕ := 8
def wall_width : ℝ := 394.4

-- The proof statement: find the distance 'd' between posters and ends
theorem find_distance_between_posters_and_wall :
  ∃ d : ℝ, (wall_width - num_posters * poster_width) / (num_posters + 1) = d ∧ d = 18 := 
by {
  -- The proof would involve showing that this specific d meets the constraints.
  sorry
}

end NUMINAMATH_GPT_find_distance_between_posters_and_wall_l1037_103720


namespace NUMINAMATH_GPT_stock_market_value_l1037_103708

def face_value : ℝ := 100
def dividend_rate : ℝ := 0.05
def yield_rate : ℝ := 0.10

theorem stock_market_value :
  (dividend_rate * face_value / yield_rate = 50) :=
by
  sorry

end NUMINAMATH_GPT_stock_market_value_l1037_103708


namespace NUMINAMATH_GPT_manager_decision_correct_l1037_103791

theorem manager_decision_correct (x : ℝ) (profit : ℝ) 
  (h_condition1 : ∀ (x : ℝ), profit = (2 * x + 20) * (40 - x)) 
  (h_condition2 : 0 ≤ x ∧ x ≤ 40)
  (h_price_reduction : x = 15) :
  profit = 1250 :=
by
  sorry

end NUMINAMATH_GPT_manager_decision_correct_l1037_103791


namespace NUMINAMATH_GPT_range_of_k_l1037_103734

noncomputable def equation (k x : ℝ) : ℝ := 4^x - k * 2^x + k + 3

theorem range_of_k {x : ℝ} (h : ∀ k, equation k x = 0 → ∃! x : ℝ, equation k x = 0) :
  ∃ k : ℝ, (k = 6 ∨ k < -3)∧ (∀ y, equation k y ≠ 0 → (y ≠ x)) :=
sorry

end NUMINAMATH_GPT_range_of_k_l1037_103734


namespace NUMINAMATH_GPT_train_pass_time_l1037_103746

theorem train_pass_time
  (v : ℝ) (l_tunnel l_train : ℝ) (h_v : v = 75) (h_l_tunnel : l_tunnel = 3.5) (h_l_train : l_train = 0.25) :
  (l_tunnel + l_train) / v * 60 = 3 :=
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_train_pass_time_l1037_103746


namespace NUMINAMATH_GPT_largest_three_digit_product_l1037_103745

theorem largest_three_digit_product : 
    ∃ (n : ℕ), 
    (n = 336) ∧ 
    (n > 99 ∧ n < 1000) ∧ 
    (∃ (x y : ℕ), x < 10 ∧ y < 10 ∧ n = x * y * (5 * x + 2 * y) ∧ 
        ∃ (k m : ℕ), k > 1 ∧ m > 1 ∧ k * m = (5 * x + 2 * y)) :=
by
  sorry

end NUMINAMATH_GPT_largest_three_digit_product_l1037_103745


namespace NUMINAMATH_GPT_triangle_angle_ABC_l1037_103740

theorem triangle_angle_ABC
  (ABD CBD ABC : ℝ) 
  (h1 : ABD = 70)
  (h2 : ABD + CBD + ABC = 200)
  (h3 : CBD = 60) : ABC = 70 := 
sorry

end NUMINAMATH_GPT_triangle_angle_ABC_l1037_103740


namespace NUMINAMATH_GPT_shaded_area_proof_l1037_103782

noncomputable def shaded_area (side_length : ℝ) (radius_factor : ℝ) : ℝ :=
  let square_area := side_length * side_length
  let radius := radius_factor * side_length
  let circle_area := Real.pi * (radius * radius)
  square_area - circle_area

theorem shaded_area_proof : shaded_area 8 0.6 = 64 - 23.04 * Real.pi :=
by sorry

end NUMINAMATH_GPT_shaded_area_proof_l1037_103782


namespace NUMINAMATH_GPT_range_of_expression_l1037_103735

theorem range_of_expression (x y : ℝ) (h : x^2 + y^2 = 4) :
  1 ≤ 4 * (x - 1/2)^2 + (y - 1)^2 + 4 * x * y ∧ 4 * (x - 1/2)^2 + (y - 1)^2 + 4 * x * y ≤ 22 + 4 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_range_of_expression_l1037_103735


namespace NUMINAMATH_GPT_bhanu_income_l1037_103780

theorem bhanu_income (I P : ℝ) (h1 : (P / 100) * I = 300) (h2 : (20 / 100) * (I - 300) = 140) : P = 30 := by
  sorry

end NUMINAMATH_GPT_bhanu_income_l1037_103780


namespace NUMINAMATH_GPT_smaller_cube_surface_area_l1037_103769

theorem smaller_cube_surface_area (edge_length : ℝ) (h : edge_length = 12) :
  let sphere_diameter := edge_length
  let smaller_cube_side := sphere_diameter / Real.sqrt 3
  let surface_area := 6 * smaller_cube_side ^ 2
  surface_area = 288 := by
  sorry

end NUMINAMATH_GPT_smaller_cube_surface_area_l1037_103769


namespace NUMINAMATH_GPT_solve_system_for_x_l1037_103761

theorem solve_system_for_x :
  ∃ x y : ℝ, (2 * x + y = 4) ∧ (x + 2 * y = 5) ∧ (x = 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_for_x_l1037_103761


namespace NUMINAMATH_GPT_jellybean_ratio_l1037_103799

theorem jellybean_ratio (jellybeans_large: ℕ) (large_glasses: ℕ) (small_glasses: ℕ) (total_jellybeans: ℕ) (jellybeans_per_large: ℕ) (jellybeans_per_small: ℕ)
  (h1 : jellybeans_large = 50)
  (h2 : large_glasses = 5)
  (h3 : small_glasses = 3)
  (h4 : total_jellybeans = 325)
  (h5 : jellybeans_per_large = jellybeans_large * large_glasses)
  (h6 : jellybeans_per_small * small_glasses = total_jellybeans - jellybeans_per_large)
  : jellybeans_per_small = 25 ∧ jellybeans_per_small / jellybeans_large = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_jellybean_ratio_l1037_103799


namespace NUMINAMATH_GPT_february_saving_l1037_103792

-- Definitions for the conditions
variable {F D : ℝ}

-- Condition 1: Saving in January
def january_saving : ℝ := 2

-- Condition 2: Saving in March
def march_saving : ℝ := 8

-- Condition 3: Total savings after 6 months
def total_savings : ℝ := 126

-- Condition 4: Savings increase by a fixed amount D each month
def fixed_increase : ℝ := D

-- Condition 5: Difference between savings in March and January
def difference_jan_mar : ℝ := 8 - 2

-- The main theorem to prove: Robi saved 50 in February
theorem february_saving : F = 50 :=
by
  -- The required proof is omitted
  sorry

end NUMINAMATH_GPT_february_saving_l1037_103792


namespace NUMINAMATH_GPT_factorize_x_squared_sub_xy_l1037_103744

theorem factorize_x_squared_sub_xy (x y : ℝ) : x^2 - x * y = x * (x - y) :=
sorry

end NUMINAMATH_GPT_factorize_x_squared_sub_xy_l1037_103744


namespace NUMINAMATH_GPT_cricket_run_rate_l1037_103729

theorem cricket_run_rate (initial_run_rate : ℝ) (initial_overs : ℕ) (target : ℕ) (remaining_overs : ℕ) 
    (run_rate_in_remaining_overs : ℝ)
    (h1 : initial_run_rate = 3.2)
    (h2 : initial_overs = 10)
    (h3 : target = 272)
    (h4 : remaining_overs = 40) :
    run_rate_in_remaining_overs = 6 :=
  sorry

end NUMINAMATH_GPT_cricket_run_rate_l1037_103729


namespace NUMINAMATH_GPT_sufficient_condition_not_necessary_condition_l1037_103724

variable {a b : ℝ} 

theorem sufficient_condition (h : a < b ∧ b < 0) : a ^ 2 > b ^ 2 :=
sorry

theorem not_necessary_condition : ¬ (∀ {a b : ℝ}, a ^ 2 > b ^ 2 → a < b ∧ b < 0) :=
sorry

end NUMINAMATH_GPT_sufficient_condition_not_necessary_condition_l1037_103724


namespace NUMINAMATH_GPT_value_of_polynomial_l1037_103752

theorem value_of_polynomial (a b : ℝ) (h : a^2 - 2 * b - 1 = 0) : -2 * a^2 + 4 * b + 2025 = 2023 :=
by
  sorry

end NUMINAMATH_GPT_value_of_polynomial_l1037_103752


namespace NUMINAMATH_GPT_balance_after_transactions_l1037_103702

variable (x : ℝ)

def monday_spent : ℝ := 0.525 * x
def tuesday_spent (remaining : ℝ) : ℝ := 0.106875 * remaining
def wednesday_spent (remaining : ℝ) : ℝ := 0.131297917 * remaining
def thursday_spent (remaining : ℝ) : ℝ := 0.040260605 * remaining

def final_balance (x : ℝ) : ℝ :=
  let after_monday := x - monday_spent x
  let after_tuesday := after_monday - tuesday_spent after_monday
  let after_wednesday := after_tuesday - wednesday_spent after_tuesday
  after_wednesday - thursday_spent after_wednesday

theorem balance_after_transactions (x : ℝ) :
  final_balance x = 0.196566478 * x :=
by
  sorry

end NUMINAMATH_GPT_balance_after_transactions_l1037_103702


namespace NUMINAMATH_GPT_min_value_x2_y2_z2_l1037_103756

theorem min_value_x2_y2_z2 (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : 
  4 ≤ x^2 + y^2 + z^2 :=
sorry

end NUMINAMATH_GPT_min_value_x2_y2_z2_l1037_103756


namespace NUMINAMATH_GPT_roger_gave_candies_l1037_103794

theorem roger_gave_candies :
  ∀ (original_candies : ℕ) (remaining_candies : ℕ) (given_candies : ℕ),
  original_candies = 95 → remaining_candies = 92 → given_candies = original_candies - remaining_candies → given_candies = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_roger_gave_candies_l1037_103794


namespace NUMINAMATH_GPT_probability_red_next_ball_l1037_103718

-- Definitions of initial conditions
def initial_red_balls : ℕ := 50
def initial_blue_balls : ℕ := 50
def initial_yellow_balls : ℕ := 30
def total_pulled_balls : ℕ := 65

-- Condition that Calvin pulled out 5 more red balls than blue balls
def red_balls_pulled (blue_balls_pulled : ℕ) : ℕ := blue_balls_pulled + 5

-- Compute the remaining balls
def remaining_balls (blue_balls_pulled : ℕ) : Prop :=
  let remaining_red_balls := initial_red_balls - red_balls_pulled blue_balls_pulled
  let remaining_blue_balls := initial_blue_balls - blue_balls_pulled
  let remaining_yellow_balls := initial_yellow_balls - (total_pulled_balls - red_balls_pulled blue_balls_pulled - blue_balls_pulled)
  (remaining_red_balls + remaining_blue_balls + remaining_yellow_balls) = 15

-- Main theorem to be proven
theorem probability_red_next_ball (blue_balls_pulled : ℕ) (h : remaining_balls blue_balls_pulled) :
  (initial_red_balls - red_balls_pulled blue_balls_pulled) / 15 = 9 / 26 :=
sorry

end NUMINAMATH_GPT_probability_red_next_ball_l1037_103718


namespace NUMINAMATH_GPT_journey_speed_l1037_103716

theorem journey_speed 
  (total_time : ℝ)
  (total_distance : ℝ)
  (second_half_speed : ℝ)
  (first_half_speed : ℝ) :
  total_time = 30 ∧ total_distance = 400 ∧ second_half_speed = 10 ∧
  2 * (total_distance / 2 / second_half_speed) + total_distance / 2 / first_half_speed = total_time →
  first_half_speed = 20 :=
by
  intros hyp
  sorry

end NUMINAMATH_GPT_journey_speed_l1037_103716


namespace NUMINAMATH_GPT_rabbit_turtle_travel_distance_l1037_103763

-- Define the initial conditions and their values
def rabbit_velocity : ℕ := 40 -- meters per minute when jumping
def rabbit_jump_time : ℕ := 3 -- minutes of jumping
def rabbit_rest_time : ℕ := 2 -- minutes of resting
def rabbit_start_time : ℕ := 9 * 60 -- 9:00 AM in minutes from midnight

def turtle_velocity : ℕ := 10 -- meters per minute
def turtle_start_time : ℕ := 6 * 60 + 40 -- 6:40 AM in minutes from midnight
def lead_time : ℕ := 15 -- turtle leads the rabbit by 15 seconds at the end

-- Define the final distance the turtle traveled by the time rabbit arrives
def distance_traveled_by_turtle (total_time : ℕ) : ℕ :=
  total_time * turtle_velocity

-- Define time intervals for periodic calculations (in minutes)
def time_interval : ℕ := 5

-- Define the total distance rabbit covers in one periodic interval
def rabbit_distance_in_interval : ℕ :=
  rabbit_velocity * rabbit_jump_time

-- Calculate total time taken by the rabbit to close the gap before starting actual run
def initial_time_to_close_gap (gap : ℕ) : ℕ := 
  gap * time_interval / rabbit_distance_in_interval

-- Define the total time the rabbit travels
def total_travel_time : ℕ :=
  initial_time_to_close_gap ((rabbit_start_time - turtle_start_time) * turtle_velocity) + 97

-- Define the total distance condition to be proved as 2370 meters
theorem rabbit_turtle_travel_distance :
  distance_traveled_by_turtle (total_travel_time + lead_time) = 2370 :=
  by sorry

end NUMINAMATH_GPT_rabbit_turtle_travel_distance_l1037_103763


namespace NUMINAMATH_GPT_triangle_arithmetic_geometric_equilateral_l1037_103742

theorem triangle_arithmetic_geometric_equilateral :
  ∀ (α β γ : ℝ), α + β + γ = 180 ∧ (∃ d, β = α + d ∧ γ = α + 2 * d) ∧ (∃ r, β = α * r ∧ γ = α * r^2) →
  α = 60 ∧ β = 60 ∧ γ = 60 :=
by
  sorry

end NUMINAMATH_GPT_triangle_arithmetic_geometric_equilateral_l1037_103742


namespace NUMINAMATH_GPT_ratatouille_cost_per_quart_l1037_103774

theorem ratatouille_cost_per_quart:
  let eggplants_pounds := 5
  let eggplants_cost_per_pound := 2.00
  let zucchini_pounds := 4
  let zucchini_cost_per_pound := 2.00
  let tomatoes_pounds := 4
  let tomatoes_cost_per_pound := 3.50
  let onions_pounds := 3
  let onions_cost_per_pound := 1.00
  let basil_pounds := 1
  let basil_cost_per_half_pound := 2.50
  let total_quarts := 4
  let eggplants_cost := eggplants_pounds * eggplants_cost_per_pound
  let zucchini_cost := zucchini_pounds * zucchini_cost_per_pound
  let tomatoes_cost := tomatoes_pounds * tomatoes_cost_per_pound
  let onions_cost := onions_pounds * onions_cost_per_pound
  let basil_cost := basil_pounds * (basil_cost_per_half_pound / 0.5)
  let total_cost := eggplants_cost + zucchini_cost + tomatoes_cost + onions_cost + basil_cost
  let cost_per_quart := total_cost / total_quarts
  cost_per_quart = 10.00 :=
  by
    sorry

end NUMINAMATH_GPT_ratatouille_cost_per_quart_l1037_103774


namespace NUMINAMATH_GPT_largest_angle_l1037_103777

theorem largest_angle (y : ℝ) (h : 40 + 70 + y = 180) : y = 70 :=
by
  sorry

end NUMINAMATH_GPT_largest_angle_l1037_103777


namespace NUMINAMATH_GPT_cubic_inches_in_two_cubic_feet_l1037_103773

theorem cubic_inches_in_two_cubic_feet :
  (12 ^ 3) * 2 = 3456 := by
  sorry

end NUMINAMATH_GPT_cubic_inches_in_two_cubic_feet_l1037_103773


namespace NUMINAMATH_GPT_quadratic_has_real_root_l1037_103701

theorem quadratic_has_real_root (b : ℝ) : 
  (b^2 - 100 ≥ 0) ↔ (b ≤ -10 ∨ b ≥ 10) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_real_root_l1037_103701


namespace NUMINAMATH_GPT_three_digit_numbers_left_l1037_103768

def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def isABAForm (n : ℕ) : Prop :=
  ∃ A B : ℕ, A ≠ 0 ∧ A ≠ B ∧ n = 100 * A + 10 * B + A

def isAABOrBAAForm (n : ℕ) : Prop :=
  ∃ A B : ℕ, A ≠ 0 ∧ A ≠ B ∧ (n = 100 * A + 10 * A + B ∨ n = 100 * B + 10 * A + A)

def totalThreeDigitNumbers : ℕ := 900

def countABA : ℕ := 81

def countAABAndBAA : ℕ := 153

theorem three_digit_numbers_left : 
  (totalThreeDigitNumbers - countABA - countAABAndBAA) = 666 := 
by
   sorry

end NUMINAMATH_GPT_three_digit_numbers_left_l1037_103768


namespace NUMINAMATH_GPT_max_value_of_k_l1037_103733

theorem max_value_of_k (m : ℝ) (h₁ : 0 < m) (h₂ : m < 1/2) : 
  (1 / m + 2 / (1 - 2 * m)) ≥ 8 :=
sorry

end NUMINAMATH_GPT_max_value_of_k_l1037_103733


namespace NUMINAMATH_GPT_sum_of_interior_angles_of_regular_polygon_l1037_103755

theorem sum_of_interior_angles_of_regular_polygon :
  (∀ (n : ℕ), (n ≠ 0) ∧ ((360 / 45 = n) → (180 * (n - 2) = 1080))) := by sorry

end NUMINAMATH_GPT_sum_of_interior_angles_of_regular_polygon_l1037_103755


namespace NUMINAMATH_GPT_possible_area_l1037_103743

theorem possible_area (A : ℝ) (B : ℝ) (L : ℝ × ℝ) (H₁ : L.1 = 13) (H₂ : L.2 = 14) (area_needed : ℝ) (H₃ : area_needed = 200) : 
∃ x y : ℝ, x = 13 ∧ y = 16 ∧ x * y ≥ area_needed :=
by
  sorry

end NUMINAMATH_GPT_possible_area_l1037_103743


namespace NUMINAMATH_GPT_cost_of_Roger_cookie_l1037_103710

theorem cost_of_Roger_cookie
  (art_cookie_length : ℕ := 4)
  (art_cookie_width : ℕ := 3)
  (art_cookie_count : ℕ := 10)
  (roger_cookie_side : ℕ := 3)
  (art_cookie_price : ℕ := 50)
  (same_dough_used : ℕ := art_cookie_count * art_cookie_length * art_cookie_width)
  (roger_cookie_area : ℕ := roger_cookie_side * roger_cookie_side)
  (roger_cookie_count : ℕ := same_dough_used / roger_cookie_area) :
  (500 / roger_cookie_count) = 38 := by
  sorry

end NUMINAMATH_GPT_cost_of_Roger_cookie_l1037_103710


namespace NUMINAMATH_GPT_mn_value_l1037_103749

theorem mn_value (m n : ℝ) 
  (h1 : m^2 + 1 = 4)
  (h2 : 2 * m + n = 0) :
  m * n = -6 := 
sorry

end NUMINAMATH_GPT_mn_value_l1037_103749


namespace NUMINAMATH_GPT_cafeteria_extra_apples_l1037_103737

-- Define the conditions from the problem
def red_apples : ℕ := 33
def green_apples : ℕ := 23
def students : ℕ := 21

-- Define the total apples and apples given out based on the conditions
def total_apples : ℕ := red_apples + green_apples
def apples_given : ℕ := students

-- Define the extra apples as the difference between total apples and apples given out
def extra_apples : ℕ := total_apples - apples_given

-- The theorem to prove that the number of extra apples is 35
theorem cafeteria_extra_apples : extra_apples = 35 :=
by
  -- The structure of the proof would go here, but is omitted
  sorry

end NUMINAMATH_GPT_cafeteria_extra_apples_l1037_103737


namespace NUMINAMATH_GPT_question1_question2_l1037_103757

def setA : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 3}

def setB (a : ℝ) : Set ℝ := {x : ℝ | abs (x - a) ≤ 1 }

def complementA : Set ℝ := {x : ℝ | x ≤ -1 ∨ x > 3}

theorem question1 : A = setA := sorry

theorem question2 (a : ℝ) : setB a ∩ complementA = setB a → a ∈ Set.union (Set.Iic (-2)) (Set.Ioi 4) := sorry

end NUMINAMATH_GPT_question1_question2_l1037_103757


namespace NUMINAMATH_GPT_solve_for_x_l1037_103728

theorem solve_for_x {x : ℤ} (h : x - 2 * x + 3 * x - 4 * x = 120) : x = -60 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1037_103728


namespace NUMINAMATH_GPT_f_at_3_l1037_103781

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 - b * x + 5

theorem f_at_3 (a b : ℝ) (h : f a b (-3) = -1) : f a b 3 = 11 :=
by
  sorry

end NUMINAMATH_GPT_f_at_3_l1037_103781


namespace NUMINAMATH_GPT_find_other_integer_l1037_103750

theorem find_other_integer (x y : ℤ) (h1 : 3 * x + 2 * y = 85) (h2 : x = 19 ∨ y = 19) : y = 14 ∨ x = 14 :=
  sorry

end NUMINAMATH_GPT_find_other_integer_l1037_103750


namespace NUMINAMATH_GPT_reciprocal_of_neg_one_sixth_is_neg_six_l1037_103767

theorem reciprocal_of_neg_one_sixth_is_neg_six : 1 / (- (1 / 6)) = -6 :=
by sorry

end NUMINAMATH_GPT_reciprocal_of_neg_one_sixth_is_neg_six_l1037_103767


namespace NUMINAMATH_GPT_range_of_n_l1037_103715

theorem range_of_n (n : ℝ) (x : ℝ) (h1 : 180 - n > 0) (h2 : ∀ x, 180 - n != x ∧ 180 - n != x + 24 → 180 - n + x + x + 24 = 180 → 44 ≤ x ∧ x ≤ 52 → 112 ≤ n ∧ n ≤ 128)
  (h3 : ∀ n, 180 - n = max (180 - n) (180 - n) - 24 ∧ min (180 - n) (180 - n) = n - 24 → 104 ≤ n ∧ n ≤ 112)
  (h4 : ∀ n, 180 - n = min (180 - n) (180 - n) ∧ max (180 - n) (180 - n) = 180 - n + 24 → 128 ≤ n ∧ n ≤ 136) :
  104 ≤ n ∧ n ≤ 136 :=
by sorry

end NUMINAMATH_GPT_range_of_n_l1037_103715


namespace NUMINAMATH_GPT_bottom_rightmost_rectangle_is_E_l1037_103712

-- Definitions of the given conditions
structure Rectangle where
  w : ℕ
  y : ℕ

def A : Rectangle := { w := 5, y := 8 }
def B : Rectangle := { w := 2, y := 4 }
def C : Rectangle := { w := 4, y := 6 }
def D : Rectangle := { w := 8, y := 5 }
def E : Rectangle := { w := 10, y := 9 }

-- The theorem we need to prove
theorem bottom_rightmost_rectangle_is_E :
    (E.w = 10) ∧ (E.y = 9) :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_bottom_rightmost_rectangle_is_E_l1037_103712


namespace NUMINAMATH_GPT_households_subscribing_to_F_l1037_103725

theorem households_subscribing_to_F
  (x y : ℕ)
  (hx : x ≥ 1)
  (h_subscriptions : 1 + 4 + 2 + 2 + 2 + y = 2 + 2 + 4 + 3 + 5 + x)
  : y = 6 :=
sorry

end NUMINAMATH_GPT_households_subscribing_to_F_l1037_103725


namespace NUMINAMATH_GPT_ellipse_foci_coordinates_l1037_103730

theorem ellipse_foci_coordinates :
  ∀ x y : ℝ,
  25 * x^2 + 16 * y^2 = 1 →
  (x, y) = (0, 3/20) ∨ (x, y) = (0, -3/20) :=
by
  intro x y h
  sorry

end NUMINAMATH_GPT_ellipse_foci_coordinates_l1037_103730


namespace NUMINAMATH_GPT_problem_equiv_proof_l1037_103707

noncomputable def simplify_and_evaluate (a : ℝ) :=
  ((a + 1) / (a + 2) + 1 / (a - 2)) / (2 / (a^2 - 4))

theorem problem_equiv_proof :
  simplify_and_evaluate (Real.sqrt 2) = 1 := 
  sorry

end NUMINAMATH_GPT_problem_equiv_proof_l1037_103707


namespace NUMINAMATH_GPT_comic_books_l1037_103727

variables (x y : ℤ)

def condition1 (x y : ℤ) : Prop := y + 7 = 5 * (x - 7)
def condition2 (x y : ℤ) : Prop := y - 9 = 3 * (x + 9)

theorem comic_books (x y : ℤ) (h₁ : condition1 x y) (h₂ : condition2 x y) : x = 39 ∧ y = 153 :=
by
  sorry

end NUMINAMATH_GPT_comic_books_l1037_103727


namespace NUMINAMATH_GPT_sum_of_squares_of_extremes_l1037_103751

theorem sum_of_squares_of_extremes
  (a b c : ℕ)
  (h1 : 2*b = 3*a)
  (h2 : 3*b = 4*c)
  (h3 : b = 9) :
  a^2 + c^2 = 180 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_of_extremes_l1037_103751


namespace NUMINAMATH_GPT_die_top_face_after_path_l1037_103747

def opposite_face (n : ℕ) : ℕ :=
  7 - n

def roll_die (start : ℕ) (sequence : List String) : ℕ :=
  sequence.foldl
    (λ top movement =>
      match movement with
      | "left" => opposite_face (7 - top) -- simplified assumption for movements
      | "forward" => opposite_face (top - 1)
      | "right" => opposite_face (7 - top + 1)
      | "back" => opposite_face (top + 1)
      | _ => top) start

theorem die_top_face_after_path : roll_die 3 ["left", "forward", "right", "back", "forward", "back"] = 4 :=
  by
  sorry

end NUMINAMATH_GPT_die_top_face_after_path_l1037_103747


namespace NUMINAMATH_GPT_trader_sold_bags_l1037_103703

-- Define the conditions as constants
def initial_bags : ℕ := 55
def restocked_bags : ℕ := 132
def current_bags : ℕ := 164

-- Define a function to calculate the number of bags sold
def bags_sold (initial restocked current : ℕ) : ℕ :=
  initial + restocked - current

-- Statement of the proof problem
theorem trader_sold_bags : bags_sold initial_bags restocked_bags current_bags = 23 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_trader_sold_bags_l1037_103703


namespace NUMINAMATH_GPT_andrew_total_donation_l1037_103785

/-
Problem statement:
Andrew started donating 7k to an organization on his 11th birthday. Yesterday, Andrew turned 29.
Verify that the total amount Andrew has donated is 126k.
-/

theorem andrew_total_donation 
  (annual_donation : ℕ := 7000) 
  (start_age : ℕ := 11) 
  (current_age : ℕ := 29) 
  (years_donating : ℕ := current_age - start_age) 
  (total_donated : ℕ := annual_donation * years_donating) :
  total_donated = 126000 := 
by 
  sorry

end NUMINAMATH_GPT_andrew_total_donation_l1037_103785


namespace NUMINAMATH_GPT_ratio_of_areas_is_five_l1037_103736

-- Define a convex quadrilateral ABCD
structure Quadrilateral (α : Type) :=
  (A B C D : α)
  (convex : True)  -- We assume convexity

-- Define the additional points B1, C1, D1, A1
structure ExtendedPoints (α : Type) (q : Quadrilateral α) :=
  (B1 C1 D1 A1 : α)
  (BB1_eq_AB : True) -- we assume the conditions BB1 = AB
  (CC1_eq_BC : True) -- CC1 = BC
  (DD1_eq_CD : True) -- DD1 = CD
  (AA1_eq_DA : True) -- AA1 = DA

-- Define the areas of the quadrilaterals
noncomputable def area {α : Type} [MetricSpace α] (A B C D : α) : ℝ := sorry
noncomputable def ratio_of_areas {α : Type} [MetricSpace α] (q : Quadrilateral α) (p : ExtendedPoints α q) : ℝ :=
  (area p.A1 p.B1 p.C1 p.D1) / (area q.A q.B q.C q.D)

theorem ratio_of_areas_is_five {α : Type} [MetricSpace α] (q : Quadrilateral α) (p : ExtendedPoints α q) :
  ratio_of_areas q p = 5 := sorry

end NUMINAMATH_GPT_ratio_of_areas_is_five_l1037_103736


namespace NUMINAMATH_GPT_loss_percentage_is_13_l1037_103759

def cost_price : ℕ := 1500
def selling_price : ℕ := 1305
def loss : ℕ := cost_price - selling_price
def loss_percentage : ℚ := (loss : ℚ) / cost_price * 100

theorem loss_percentage_is_13 :
  loss_percentage = 13 := 
by
  sorry

end NUMINAMATH_GPT_loss_percentage_is_13_l1037_103759


namespace NUMINAMATH_GPT_trajectory_of_circle_center_l1037_103741

theorem trajectory_of_circle_center :
  ∀ (M : ℝ × ℝ), (∃ r : ℝ, (M.1 + r = 1 ∧ M.1 - r = -1) ∧ (M.1 - 1)^2 + (M.2 - 0)^2 = r^2) → M.2^2 = 4 * M.1 :=
by
  intros M h
  sorry

end NUMINAMATH_GPT_trajectory_of_circle_center_l1037_103741


namespace NUMINAMATH_GPT_g_g_g_g_3_eq_101_l1037_103771

def g (m : ℕ) : ℕ :=
  if m < 5 then m^2 + 1 else 2 * m + 3

theorem g_g_g_g_3_eq_101 : g (g (g (g 3))) = 101 :=
  by {
    -- the proof goes here
    sorry
  }

end NUMINAMATH_GPT_g_g_g_g_3_eq_101_l1037_103771


namespace NUMINAMATH_GPT_possible_values_f_l1037_103723

noncomputable def f (x y z : ℝ) : ℝ := (y / (y + x)) + (z / (z + y)) + (x / (x + z))

theorem possible_values_f (x y z : ℝ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : z ≠ x) (h4 : x > 0) (h5 : y > 0) (h6 : z > 0) (h7 : x^2 + y^3 = z^4) : 
  1 < f x y z ∧ f x y z < 2 :=
sorry

end NUMINAMATH_GPT_possible_values_f_l1037_103723


namespace NUMINAMATH_GPT_friends_boat_crossing_impossible_l1037_103704

theorem friends_boat_crossing_impossible : 
  ∀ (friends : Finset ℕ) (boat_capacity : ℕ), friends.card = 5 → boat_capacity ≥ 5 → 
  ¬ (∀ group : Finset ℕ, group ⊆ friends → group ≠ ∅ → group.card ≤ boat_capacity → 
  ∃ crossing : ℕ, (crossing = group.card ∧ group ⊆ friends)) :=
by
  intro friends boat_capacity friends_card boat_capacity_cond goal
  sorry

end NUMINAMATH_GPT_friends_boat_crossing_impossible_l1037_103704


namespace NUMINAMATH_GPT_rectangle_area_l1037_103753

-- Declare the given conditions
def circle_radius : ℝ := 5
def rectangle_width : ℝ := 2 * circle_radius
def length_to_width_ratio : ℝ := 2

-- Given that the length to width ratio is 2:1, calculate the length
def rectangle_length : ℝ := length_to_width_ratio * rectangle_width

-- Define the statement we need to prove
theorem rectangle_area :
  rectangle_length * rectangle_width = 200 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1037_103753


namespace NUMINAMATH_GPT_s_point_condition_l1037_103775

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x)

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 2 * a * x
noncomputable def g_prime (a : ℝ) (x : ℝ) : ℝ := 1 / x

theorem s_point_condition (a : ℝ) (x₀ : ℝ) (h_f_g : f a x₀ = g a x₀) (h_f'g' : f_prime a x₀ = g_prime a x₀) :
  a = 2 / Real.exp 1 :=
by
  sorry

end NUMINAMATH_GPT_s_point_condition_l1037_103775


namespace NUMINAMATH_GPT_sin_2alpha_val_l1037_103700

-- Define the conditions and the problem in Lean 4
theorem sin_2alpha_val (α : ℝ) (h1 : π < α ∨ α < 3 * π / 2)
  (h2 : 2 * (Real.tan α) ^ 2 - 7 * Real.tan α + 3 = 0) :
  (π < α ∧ α < 5 * π / 4 → Real.sin (2 * α) = 4 / 5) ∧ 
  (5 * π / 4 < α ∧ α < 3 * π / 2 → Real.sin (2 * α) = 3 / 5) := 
sorry

end NUMINAMATH_GPT_sin_2alpha_val_l1037_103700


namespace NUMINAMATH_GPT_no_positive_integers_satisfy_l1037_103726

theorem no_positive_integers_satisfy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : 
  x^5 + y^5 + 1 ≠ (x + 2)^5 + (y - 3)^5 :=
sorry

end NUMINAMATH_GPT_no_positive_integers_satisfy_l1037_103726


namespace NUMINAMATH_GPT_minimal_guests_l1037_103738

-- Problem statement: For 120 chairs arranged in a circle,
-- determine the smallest number of guests (N) needed 
-- so that any additional guest must sit next to an already seated guest.

theorem minimal_guests (N : ℕ) : 
  (∀ (chairs : ℕ), chairs = 120 → 
    ∃ (N : ℕ), N = 20 ∧ 
      (∀ (new_guest : ℕ), new_guest + chairs = 120 → 
        new_guest ≤ N + 1 ∧ new_guest ≤ N - 1)) :=
by
  sorry

end NUMINAMATH_GPT_minimal_guests_l1037_103738


namespace NUMINAMATH_GPT_quadratic_inequality_k_range_l1037_103711

variable (k : ℝ)

theorem quadratic_inequality_k_range (h : ∀ x : ℝ, k * x^2 + 2 * k * x - (k + 2) < 0) :
  -1 < k ∧ k < 0 := by
sorry

end NUMINAMATH_GPT_quadratic_inequality_k_range_l1037_103711
