import Mathlib

namespace NUMINAMATH_GPT_trigonometric_sum_l210_21031

theorem trigonometric_sum (θ : ℝ) (h_tan_θ : Real.tan θ = 5 / 12) (h_range : π ≤ θ ∧ θ ≤ 3 * π / 2) : 
  Real.cos θ + Real.sin θ = -17 / 13 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_sum_l210_21031


namespace NUMINAMATH_GPT_initial_blue_balls_l210_21013

theorem initial_blue_balls (B : ℕ) 
  (h1 : 18 - 3 = 15) 
  (h2 : (B - 3) / 15 = 1 / 5) : 
  B = 6 :=
by sorry

end NUMINAMATH_GPT_initial_blue_balls_l210_21013


namespace NUMINAMATH_GPT_negation_of_proposition_l210_21069

theorem negation_of_proposition (a b : ℝ) : ¬ (a > b ∧ a - 1 > b - 1) ↔ a ≤ b ∨ a - 1 ≤ b - 1 :=
by sorry

end NUMINAMATH_GPT_negation_of_proposition_l210_21069


namespace NUMINAMATH_GPT_Katrina_sold_in_morning_l210_21097

theorem Katrina_sold_in_morning :
  ∃ M : ℕ, (120 - 57 - 16 - 11) = M := sorry

end NUMINAMATH_GPT_Katrina_sold_in_morning_l210_21097


namespace NUMINAMATH_GPT_mono_increasing_necessary_not_sufficient_problem_statement_l210_21015

-- Define the function
def f (x : ℝ) (m : ℝ) : ℝ := x^3 + 2*x^2 + m*x + 1

-- Define the first condition of p: f(x) is monotonically increasing in (-∞, +∞)
def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x ≤ f y

-- Define the second condition q: m > 4/3
def m_gt_4_over_3 (m : ℝ) : Prop := m > 4/3

-- State the theorem: 
theorem mono_increasing_necessary_not_sufficient (m : ℝ):
  is_monotonically_increasing (f x) → m_gt_4_over_3 m → 
  (is_monotonically_increasing (f x) ↔ m ≥ 4/3) ∧ (¬ is_monotonically_increasing (f x) → m > 4/3) := 
by
  sorry

-- Main theorem tying the conditions to the conclusion
theorem problem_statement (m : ℝ):
  is_monotonically_increasing (f x) → m_gt_4_over_3 m → 
  (is_monotonically_increasing (f x) ↔ m ≥ 4/3) ∧ (¬ is_monotonically_increasing (f x) → m > 4/3) :=
  by sorry

end NUMINAMATH_GPT_mono_increasing_necessary_not_sufficient_problem_statement_l210_21015


namespace NUMINAMATH_GPT_john_votes_l210_21061

theorem john_votes (J : ℝ) (total_votes : ℝ) (third_candidate_votes : ℝ) (james_votes : ℝ) 
  (h1 : total_votes = 1150) 
  (h2 : third_candidate_votes = J + 150) 
  (h3 : james_votes = 0.70 * (total_votes - J - third_candidate_votes)) 
  (h4 : total_votes = J + james_votes + third_candidate_votes) : 
  J = 500 := 
by 
  rw [h1, h2, h3] at h4 
  sorry

end NUMINAMATH_GPT_john_votes_l210_21061


namespace NUMINAMATH_GPT_community_group_loss_l210_21011

def cookies_bought : ℕ := 800
def cost_per_4_cookies : ℚ := 3 -- dollars per 4 cookies
def sell_per_3_cookies : ℚ := 2 -- dollars per 3 cookies

def cost_per_cookie : ℚ := cost_per_4_cookies / 4
def sell_per_cookie : ℚ := sell_per_3_cookies / 3

def total_cost (n : ℕ) (cost_per_cookie : ℚ) : ℚ := n * cost_per_cookie
def total_revenue (n : ℕ) (sell_per_cookie : ℚ) : ℚ := n * sell_per_cookie

def loss (n : ℕ) (cost_per_cookie sell_per_cookie : ℚ) : ℚ := 
  total_cost n cost_per_cookie - total_revenue n sell_per_cookie

theorem community_group_loss : loss cookies_bought cost_per_cookie sell_per_cookie = 64 := by
  sorry

end NUMINAMATH_GPT_community_group_loss_l210_21011


namespace NUMINAMATH_GPT_light_stripes_total_area_l210_21034

theorem light_stripes_total_area (x : ℝ) (h : 45 * x = 135) :
  2 * x + 4 * x + 6 * x + 8 * x = 60 := 
sorry

end NUMINAMATH_GPT_light_stripes_total_area_l210_21034


namespace NUMINAMATH_GPT_age_of_sisters_l210_21060

theorem age_of_sisters (a b : ℕ) (h1 : 10 * a - 9 * b = 89) 
  (h2 : 10 = 10) : a = 17 ∧ b = 9 :=
by sorry

end NUMINAMATH_GPT_age_of_sisters_l210_21060


namespace NUMINAMATH_GPT_cannot_determine_right_triangle_l210_21086

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

def two_angles_complementary (α β : ℝ) : Prop :=
  α + β = 90

def exterior_angle_is_right (γ : ℝ) : Prop :=
  γ = 90

theorem cannot_determine_right_triangle :
  ¬ (∃ (a b c : ℝ), a = 1 ∧ b = 1 ∧ c = 2 ∧ is_right_triangle a b c) :=
by sorry

end NUMINAMATH_GPT_cannot_determine_right_triangle_l210_21086


namespace NUMINAMATH_GPT_rest_area_milepost_l210_21036

theorem rest_area_milepost (milepost_first : ℕ) (milepost_seventh : ℕ) (h_first : milepost_first = 20) (h_seventh : milepost_seventh = 140) : 
  ∃ milepost_rest : ℕ, milepost_rest = (milepost_first + milepost_seventh) / 2 ∧ milepost_rest = 80 :=
by
  sorry

end NUMINAMATH_GPT_rest_area_milepost_l210_21036


namespace NUMINAMATH_GPT_solve_equation_l210_21082

theorem solve_equation :
  (∃ x : ℝ, (x^2 + 3*x + 5) / (x^2 + 5*x + 6) = x + 3) → (x = -1) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l210_21082


namespace NUMINAMATH_GPT_mina_crafts_total_l210_21012

theorem mina_crafts_total :
  let a₁ := 3
  let d := 4
  let n := 10
  let crafts_sold_on_day (d: ℕ) := a₁ + (d - 1) * d
  let S (n: ℕ) := (n * (2 * a₁ + (n - 1) * d)) / 2
  S n = 210 :=
by
  sorry

end NUMINAMATH_GPT_mina_crafts_total_l210_21012


namespace NUMINAMATH_GPT_problem_solution_l210_21032

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 1 + Real.log (2 - x) / Real.log 2 else 2 ^ (x - 1)

theorem problem_solution : f (-2) + f (Real.log 12 / Real.log 2) = 9 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l210_21032


namespace NUMINAMATH_GPT_arithmetic_sequence_function_positive_l210_21052

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = - (f x)

def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_function_positive
  {f : ℝ → ℝ} {a : ℕ → ℝ}
  (hf_odd : is_odd f)
  (hf_mono : is_monotonically_increasing f)
  (ha_arith : is_arithmetic_sequence a)
  (ha3_pos : a 3 > 0) : 
  f (a 1) + f (a 3) + f (a 5) > 0 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_function_positive_l210_21052


namespace NUMINAMATH_GPT_arrange_chairs_and_stools_l210_21016

-- Definition of the mathematical entities based on the conditions
def num_ways_to_arrange (women men : ℕ) : ℕ :=
  let total := women + men
  (total.factorial) / (women.factorial * men.factorial)

-- Prove that the arrangement yields the correct number of ways
theorem arrange_chairs_and_stools :
  num_ways_to_arrange 7 3 = 120 := by
  -- The specific definitions and steps are not to be included in the Lean statement;
  -- hence, adding a placeholder for the proof.
  sorry

end NUMINAMATH_GPT_arrange_chairs_and_stools_l210_21016


namespace NUMINAMATH_GPT_magnitude_of_Z_l210_21033

-- Define the complex number Z
def Z : ℂ := 3 - 4 * Complex.I

-- Define the theorem to prove the magnitude of Z
theorem magnitude_of_Z : Complex.abs Z = 5 := by
  sorry

end NUMINAMATH_GPT_magnitude_of_Z_l210_21033


namespace NUMINAMATH_GPT_triangle_inequality_l210_21091

theorem triangle_inequality (a b c : ℕ) : 
    a + b > c ∧ a + c > b ∧ b + c > a ↔ 
    (a, b, c) = (2, 3, 4) ∨ (a, b, c) = (3, 4, 7) ∨ (a, b, c) = (4, 6, 2) ∨ (a, b, c) = (7, 10, 2)
    → (a + b > c ∧ a + c > b ∧ b + c > a ↔ (a, b, c) = (2, 3, 4)) ∧
      (a + b = c ∨ a + c = b ∨ b + c = a         ↔ (a, b, c) = (3, 4, 7)) ∧
      (a + b = c ∨ a + c = b ∨ b + c = a        ↔ (a, b, c) = (4, 6, 2)) ∧
      (a + b < c ∨ a + c < b ∨ b + c < a        ↔ (a, b, c) = (7, 10, 2)) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l210_21091


namespace NUMINAMATH_GPT_triangle_area_is_sqrt3_over_4_l210_21075

noncomputable def area_of_triangle (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1/2 * b * c * Real.sin A

theorem triangle_area_is_sqrt3_over_4
  (a b c A B : ℝ)
  (h1 : A = Real.pi / 3)
  (h2 : b = 2 * a * Real.cos B)
  (h3 : c = 1)
  (h4 : B = Real.pi / 3)
  (h5 : a = 1)
  (h6 : b = 1) :
  area_of_triangle a b c A B (Real.pi - A - B) = Real.sqrt 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_is_sqrt3_over_4_l210_21075


namespace NUMINAMATH_GPT_bryden_receives_22_50_dollars_l210_21046

-- Define the face value of a regular quarter
def face_value_regular : ℝ := 0.25

-- Define the number of regular quarters Bryden has
def num_regular_quarters : ℕ := 4

-- Define the face value of the special quarter
def face_value_special : ℝ := face_value_regular * 2

-- The collector pays 15 times the face value for regular quarters
def multiplier : ℝ := 15

-- Calculate the total face value of all quarters
def total_face_value : ℝ := (num_regular_quarters * face_value_regular) + face_value_special

-- Calculate the total amount Bryden will receive
def total_amount_received : ℝ := multiplier * total_face_value

-- Prove that the total amount Bryden will receive is $22.50
theorem bryden_receives_22_50_dollars : total_amount_received = 22.50 :=
by
  sorry

end NUMINAMATH_GPT_bryden_receives_22_50_dollars_l210_21046


namespace NUMINAMATH_GPT_how_many_fewer_girls_l210_21090

def total_students : ℕ := 27
def girls : ℕ := 11
def boys : ℕ := total_students - girls
def fewer_girls_than_boys : ℕ := boys - girls

theorem how_many_fewer_girls :
  fewer_girls_than_boys = 5 :=
sorry

end NUMINAMATH_GPT_how_many_fewer_girls_l210_21090


namespace NUMINAMATH_GPT_find_m_l210_21025

theorem find_m (a b m : ℝ) :
  (∀ x : ℝ, (x^2 - b * x + b^2) / (a * x^2 - b^2) = (m - 1) / (m + 1) → (∀ y : ℝ, x = y ∧ x = -y)) →
  c = b^2 →
  m = (a - 1) / (a + 1) :=
by
  sorry

end NUMINAMATH_GPT_find_m_l210_21025


namespace NUMINAMATH_GPT_roots_sum_one_imp_b_eq_neg_a_l210_21098

theorem roots_sum_one_imp_b_eq_neg_a (a b c : ℝ) (h : a ≠ 0) 
  (hr : ∀ (r s : ℝ), r + s = 1 → (r * s = c / a) → a * (r^2 + (b/a) * r + c/a) = 0) : b = -a :=
sorry

end NUMINAMATH_GPT_roots_sum_one_imp_b_eq_neg_a_l210_21098


namespace NUMINAMATH_GPT_rounding_sum_eq_one_third_probability_l210_21054

noncomputable def rounding_sum_probability : ℝ :=
  (λ (total : ℝ) => 
    let round := (λ (x : ℝ) => if x < 0.5 then 0 else if x < 1.5 then 1 else if x < 2.5 then 2 else 3)
    let interval := (λ (start : ℝ) (end_ : ℝ) => end_ - start)
    let sum_conditions := [((0.5,1.5), 3), ((1.5,2.5), 2)]
    let total_length := 3

    let valid_intervals := sum_conditions.map (λ p => interval (p.fst.fst) (p.fst.snd))
    let total_valid_interval := List.sum valid_intervals
    total_valid_interval / total_length
  ) 3

theorem rounding_sum_eq_one_third_probability : rounding_sum_probability = 2 / 3 := by sorry

end NUMINAMATH_GPT_rounding_sum_eq_one_third_probability_l210_21054


namespace NUMINAMATH_GPT_rational_numbers_cubic_sum_l210_21008

theorem rational_numbers_cubic_sum
  (a b c : ℚ)
  (h1 : a - b + c = 3)
  (h2 : a^2 + b^2 + c^2 = 3) :
  a^3 + b^3 + c^3 = 1 :=
by
  sorry

end NUMINAMATH_GPT_rational_numbers_cubic_sum_l210_21008


namespace NUMINAMATH_GPT_difference_sum_even_odd_1000_l210_21093

open Nat

def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

def sum_first_n_even (n : ℕ) : ℕ :=
  n * (n + 1)

theorem difference_sum_even_odd_1000 :
  sum_first_n_even 1000 - sum_first_n_odd 1000 = 1000 :=
by
  sorry

end NUMINAMATH_GPT_difference_sum_even_odd_1000_l210_21093


namespace NUMINAMATH_GPT_inner_rectangle_length_l210_21057

def inner_rect_width : ℕ := 2

def second_rect_area (x : ℕ) : ℕ := 6 * (x + 4)

def largest_rect_area (x : ℕ) : ℕ := 10 * (x + 8)

def shaded_area_1 (x : ℕ) : ℕ := second_rect_area x - 2 * x

def shaded_area_2 (x : ℕ) : ℕ := largest_rect_area x - second_rect_area x

def in_arithmetic_progression (a b c : ℕ) : Prop := b - a = c - b

theorem inner_rectangle_length (x : ℕ) :
  in_arithmetic_progression (2 * x) (shaded_area_1 x) (shaded_area_2 x) → x = 4 := by
  intros
  sorry

end NUMINAMATH_GPT_inner_rectangle_length_l210_21057


namespace NUMINAMATH_GPT_sum_squares_divisible_by_4_iff_even_l210_21068

theorem sum_squares_divisible_by_4_iff_even (a b c : ℕ) (ha : a % 2 = 0) (hb : b % 2 = 0) (hc : c % 2 = 0) : 
(a^2 + b^2 + c^2) % 4 = 0 ↔ 
  (a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0) :=
sorry

end NUMINAMATH_GPT_sum_squares_divisible_by_4_iff_even_l210_21068


namespace NUMINAMATH_GPT_num_three_digit_numbers_l210_21024

theorem num_three_digit_numbers (a b c : ℕ) :
  a ≠ 0 →
  b = (a + c) / 2 →
  c = a - b →
  ∃ n1 n2 n3 : ℕ, 
    (n1 = 100 * 3 + 10 * 2 + 1) ∧
    (n2 = 100 * 9 + 10 * 6 + 3) ∧
    (n3 = 100 * 6 + 10 * 4 + 2) ∧ 
    3 = 3 := 
sorry  

end NUMINAMATH_GPT_num_three_digit_numbers_l210_21024


namespace NUMINAMATH_GPT_jonas_socks_solution_l210_21040

theorem jonas_socks_solution (p_s p_h n_p n_t n : ℕ) (h_ps : p_s = 20) (h_ph : p_h = 5) (h_np : n_p = 10) (h_nt : n_t = 10) :
  2 * (p_s * 2 + p_h * 2 + n_p + n_t) = 2 * (p_s * 2 + p_h * 2 + n_p + n_t + n * 2) :=
by
  -- skipping the proof part
  sorry

end NUMINAMATH_GPT_jonas_socks_solution_l210_21040


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l210_21014

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (hS : ∀ n, S n = n * (a 1 + a n) / 2)
  (h : a 3 = 20 - a 6) : S 8 = 80 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l210_21014


namespace NUMINAMATH_GPT_angle_B_range_l210_21083

def range_of_angle_B (a b c : ℝ) (A B C : ℝ) : Prop :=
  (0 < B ∧ B ≤ Real.pi / 3)

theorem angle_B_range
  (a b c A B C : ℝ)
  (h1 : b^2 = a * c)
  (h2 : A + B + C = π)
  (h3 : a > 0)
  (h4 : b > 0)
  (h5 : c > 0)
  (h6 : a + b > c)
  (h7 : a + c > b)
  (h8 : b + c > a) :
  range_of_angle_B a b c A B C :=
sorry

end NUMINAMATH_GPT_angle_B_range_l210_21083


namespace NUMINAMATH_GPT_total_pupils_correct_l210_21063

-- Definitions of the number of girls and boys in each school
def girlsA := 542
def boysA := 387
def girlsB := 713
def boysB := 489
def girlsC := 628
def boysC := 361

-- Total pupils in each school
def pupilsA := girlsA + boysA
def pupilsB := girlsB + boysB
def pupilsC := girlsC + boysC

-- Total pupils across all schools
def total_pupils := pupilsA + pupilsB + pupilsC

-- The proof statement (no proof provided, hence sorry)
theorem total_pupils_correct : total_pupils = 3120 := by sorry

end NUMINAMATH_GPT_total_pupils_correct_l210_21063


namespace NUMINAMATH_GPT_remainder_of_f_x10_mod_f_l210_21077

def f (x : ℤ) : ℤ := x^4 + x^3 + x^2 + x + 1

theorem remainder_of_f_x10_mod_f (x : ℤ) : (f (x ^ 10)) % (f x) = 5 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_f_x10_mod_f_l210_21077


namespace NUMINAMATH_GPT_bell_ratio_l210_21039

theorem bell_ratio :
  ∃ (B3 B2 : ℕ), 
  B2 = 2 * 50 ∧ 
  50 + B2 + B3 = 550 ∧ 
  (B3 / B2 = 4) := 
sorry

end NUMINAMATH_GPT_bell_ratio_l210_21039


namespace NUMINAMATH_GPT_range_of_a_l210_21004

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 2| + |x - 1| ≥ a) ↔ a ≤ 3 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l210_21004


namespace NUMINAMATH_GPT_reading_time_difference_l210_21044

theorem reading_time_difference (xanthia_speed molly_speed book_length : ℕ)
  (hx : xanthia_speed = 120) (hm : molly_speed = 60) (hb : book_length = 300) :
  (book_length / molly_speed - book_length / xanthia_speed) * 60 = 150 :=
by
  -- We acknowledge the proof here would use the given values
  sorry

end NUMINAMATH_GPT_reading_time_difference_l210_21044


namespace NUMINAMATH_GPT_problem_EF_fraction_of_GH_l210_21023

theorem problem_EF_fraction_of_GH (E F G H : Type) 
  (GE EH GH GF FH EF : ℝ) 
  (h1 : GE = 3 * EH) 
  (h2 : GF = 8 * FH)
  (h3 : GH = GE + EH)
  (h4 : GH = GF + FH) : 
  EF = 5 / 36 * GH :=
by
  sorry

end NUMINAMATH_GPT_problem_EF_fraction_of_GH_l210_21023


namespace NUMINAMATH_GPT_cars_with_both_features_l210_21002

theorem cars_with_both_features (T P_s P_w N B : ℕ)
  (hT : T = 65) 
  (hPs : P_s = 45) 
  (hPw : P_w = 25) 
  (hN : N = 12) 
  (h_equation : P_s + P_w - B + N = T) :
  B = 17 :=
by
  sorry

end NUMINAMATH_GPT_cars_with_both_features_l210_21002


namespace NUMINAMATH_GPT_kite_area_l210_21042

theorem kite_area (EF GH : ℝ) (FG EH : ℕ) (h1 : FG * FG + EH * EH = 25) : EF * GH = 12 :=
by
  sorry

end NUMINAMATH_GPT_kite_area_l210_21042


namespace NUMINAMATH_GPT_heather_blocks_l210_21021

theorem heather_blocks (initial_blocks : ℕ) (shared_blocks : ℕ) (remaining_blocks : ℕ) :
  initial_blocks = 86 → shared_blocks = 41 → remaining_blocks = initial_blocks - shared_blocks → remaining_blocks = 45 :=
by
  sorry

end NUMINAMATH_GPT_heather_blocks_l210_21021


namespace NUMINAMATH_GPT_number_of_children_is_30_l210_21037

-- Informal statements
def total_guests := 80
def men := 40
def women := men / 2
def adults := men + women
def children := total_guests - adults
def children_after_adding_10 := children + 10

-- Formal proof statement
theorem number_of_children_is_30 :
  children_after_adding_10 = 30 := by
  sorry

end NUMINAMATH_GPT_number_of_children_is_30_l210_21037


namespace NUMINAMATH_GPT_units_produced_today_l210_21081

theorem units_produced_today (n : ℕ) (P : ℕ) (T : ℕ) 
  (h1 : n = 14)
  (h2 : P = 60 * n)
  (h3 : (P + T) / (n + 1) = 62) : 
  T = 90 :=
by
  sorry

end NUMINAMATH_GPT_units_produced_today_l210_21081


namespace NUMINAMATH_GPT_train_passes_bridge_in_128_seconds_l210_21072

/-- A proof problem regarding a train passing a bridge -/
theorem train_passes_bridge_in_128_seconds 
  (train_length : ℕ) 
  (train_speed_kmh : ℕ) 
  (bridge_length : ℕ) 
  (conversion_factor : ℚ) 
  (time_to_pass : ℚ) :
  train_length = 1200 →
  train_speed_kmh = 90 →
  bridge_length = 2000 →
  conversion_factor = (5 / 18) →
  time_to_pass = (train_length + bridge_length) / (train_speed_kmh * conversion_factor) →
  time_to_pass = 128 := 
by
  -- We are skipping the proof itself
  sorry

end NUMINAMATH_GPT_train_passes_bridge_in_128_seconds_l210_21072


namespace NUMINAMATH_GPT_triangle_CD_length_l210_21017

noncomputable def triangle_AB_values : ℝ := 4024
noncomputable def triangle_AC_values : ℝ := 4024
noncomputable def triangle_BC_values : ℝ := 2012
noncomputable def CD_value : ℝ := 504.5

theorem triangle_CD_length 
  (AB AC : ℝ)
  (BC : ℝ)
  (CD : ℝ)
  (h1 : AB = triangle_AB_values)
  (h2 : AC = triangle_AC_values)
  (h3 : BC = triangle_BC_values) :
  CD = CD_value := by
  sorry

end NUMINAMATH_GPT_triangle_CD_length_l210_21017


namespace NUMINAMATH_GPT_find_C_coordinates_l210_21035

variables {A B M L C : ℝ × ℝ}

def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

def on_line_bisector (L B : ℝ × ℝ) : Prop :=
  B.1 = 6  -- Vertical line through B

theorem find_C_coordinates
  (A := (2, 8))
  (M := (4, 11))
  (L := (6, 6))
  (hM : is_midpoint M A B)
  (hL : on_line_bisector L B) :
  C = (6, 14) :=
sorry

end NUMINAMATH_GPT_find_C_coordinates_l210_21035


namespace NUMINAMATH_GPT_projectile_reaches_30m_at_2_seconds_l210_21088

theorem projectile_reaches_30m_at_2_seconds:
  ∀ t : ℝ, -5 * t^2 + 25 * t = 30 → t = 2 ∨ t = 3 :=
by
  sorry

end NUMINAMATH_GPT_projectile_reaches_30m_at_2_seconds_l210_21088


namespace NUMINAMATH_GPT_find_constant_term_l210_21026

-- Definitions based on conditions:
def sum_of_coeffs (n : ℕ) : ℕ := 4 ^ n
def sum_of_binom_coeffs (n : ℕ) : ℕ := 2 ^ n
def P_plus_Q_equals (n : ℕ) : Prop := sum_of_coeffs n + sum_of_binom_coeffs n = 272

-- Constant term in the binomial expansion:
def constant_term (n r : ℕ) : ℕ := Nat.choose n r * (3 ^ (n - r))

-- The proof statement
theorem find_constant_term : 
  ∃ n r : ℕ, P_plus_Q_equals n ∧ n = 4 ∧ r = 1 ∧ constant_term n r = 108 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_constant_term_l210_21026


namespace NUMINAMATH_GPT_problem_statement_l210_21010

theorem problem_statement (a b : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 2 = 2) 
  (h3 : ∀ n : ℕ, a (n + 2) = a n)
  (h_b : ∀ n : ℕ, b (n + 1) - b n = a n)
  (h_repeat : ∀ k : ℕ, ∃ m : ℕ, (b (2 * m) / a m) = k)
  : b 1 = 2 :=
sorry

end NUMINAMATH_GPT_problem_statement_l210_21010


namespace NUMINAMATH_GPT_best_model_is_A_l210_21070

-- Definitions of the models and their R^2 values
def ModelA_R_squared : ℝ := 0.95
def ModelB_R_squared : ℝ := 0.81
def ModelC_R_squared : ℝ := 0.50
def ModelD_R_squared : ℝ := 0.32

-- Definition stating that the best fitting model is the one with the highest R^2 value
def best_fitting_model (R_squared_A R_squared_B R_squared_C R_squared_D: ℝ) : Prop :=
  R_squared_A > R_squared_B ∧ R_squared_A > R_squared_C ∧ R_squared_A > R_squared_D

-- Proof statement
theorem best_model_is_A : best_fitting_model ModelA_R_squared ModelB_R_squared ModelC_R_squared ModelD_R_squared :=
by
  -- Skipping the proof logic
  sorry

end NUMINAMATH_GPT_best_model_is_A_l210_21070


namespace NUMINAMATH_GPT_interval_of_x₀_l210_21007

-- Definition of the problem
variable (x₀ : ℝ)

-- Conditions
def condition_1 := x₀ > 0 ∧ x₀ < Real.pi
def condition_2 := Real.sin x₀ + Real.cos x₀ = 2 / 3

-- Proof problem statement
theorem interval_of_x₀ 
  (h1 : condition_1 x₀)
  (h2 : condition_2 x₀) : 
  x₀ > 7 * Real.pi / 12 ∧ x₀ < 3 * Real.pi / 4 := 
sorry

end NUMINAMATH_GPT_interval_of_x₀_l210_21007


namespace NUMINAMATH_GPT_joe_first_lift_is_400_mike_first_lift_is_450_lisa_second_lift_is_250_l210_21048

-- Defining the weights of Joe's lifts
variable (J1 J2 : ℕ)

-- Conditions for Joe
def joe_conditions : Prop :=
  (J1 + J2 = 900) ∧ (2 * J1 = J2 + 300)

-- Defining the weights of Mike's lifts
variable (M1 M2 : ℕ)

-- Conditions for Mike  
def mike_conditions : Prop :=
  (M1 + M2 = 1100) ∧ (M2 = M1 + 200)

-- Defining the weights of Lisa's lifts
variable (L1 L2 : ℕ)

-- Conditions for Lisa  
def lisa_conditions : Prop :=
  (L1 + L2 = 1000) ∧ (L1 = 3 * L2)

-- Proof statements
theorem joe_first_lift_is_400 (h : joe_conditions J1 J2) : J1 = 400 :=
by
  sorry

theorem mike_first_lift_is_450 (h : mike_conditions M1 M2) : M1 = 450 :=
by
  sorry

theorem lisa_second_lift_is_250 (h : lisa_conditions L1 L2) : L2 = 250 :=
by
  sorry

end NUMINAMATH_GPT_joe_first_lift_is_400_mike_first_lift_is_450_lisa_second_lift_is_250_l210_21048


namespace NUMINAMATH_GPT_initial_money_l210_21058

-- Define the conditions
variable (M : ℝ)
variable (h : (1 / 3) * M = 50)

-- Define the theorem to be proved
theorem initial_money : M = 150 := 
by
  sorry

end NUMINAMATH_GPT_initial_money_l210_21058


namespace NUMINAMATH_GPT_percy_bound_longer_martha_step_l210_21067

theorem percy_bound_longer_martha_step (steps_per_gap_martha: ℕ) (bounds_per_gap_percy: ℕ)
  (gaps: ℕ) (total_distance: ℕ) 
  (step_length_martha: ℝ) (bound_length_percy: ℝ) :
  steps_per_gap_martha = 50 →
  bounds_per_gap_percy = 15 →
  gaps = 50 →
  total_distance = 10560 →
  step_length_martha = total_distance / (steps_per_gap_martha * gaps) →
  bound_length_percy = total_distance / (bounds_per_gap_percy * gaps) →
  (bound_length_percy - step_length_martha) = 10 :=
by
  sorry

end NUMINAMATH_GPT_percy_bound_longer_martha_step_l210_21067


namespace NUMINAMATH_GPT_multiplication_result_l210_21029

theorem multiplication_result :
  121 * 54 = 6534 := by
  sorry

end NUMINAMATH_GPT_multiplication_result_l210_21029


namespace NUMINAMATH_GPT_func_eq_condition_l210_21084

variable (a : ℝ)

theorem func_eq_condition (f : ℝ → ℝ) :
  (∀ x : ℝ, f (Real.sin x) + a * f (Real.cos x) = Real.cos (2 * x)) ↔ a ∈ (Set.univ \ {1} : Set ℝ) :=
by
  sorry

end NUMINAMATH_GPT_func_eq_condition_l210_21084


namespace NUMINAMATH_GPT_ambulance_ride_cost_l210_21095

-- Define the conditions as per the given problem.
def totalBill : ℝ := 5000
def medicationPercentage : ℝ := 0.5
def overnightStayPercentage : ℝ := 0.25
def foodCost : ℝ := 175

-- Define the question to be proved.
theorem ambulance_ride_cost :
  let medicationCost := totalBill * medicationPercentage
  let remainingAfterMedication := totalBill - medicationCost
  let overnightStayCost := remainingAfterMedication * overnightStayPercentage
  let remainingAfterOvernight := remainingAfterMedication - overnightStayCost
  let remainingAfterFood := remainingAfterOvernight - foodCost
  remainingAfterFood = 1700 :=
by
  -- Proof can be completed here
  sorry

end NUMINAMATH_GPT_ambulance_ride_cost_l210_21095


namespace NUMINAMATH_GPT_total_carrot_sticks_l210_21000

-- Define the number of carrot sticks James ate before and after dinner
def carrot_sticks_before_dinner : Nat := 22
def carrot_sticks_after_dinner : Nat := 15

-- Prove that the total number of carrot sticks James ate is 37
theorem total_carrot_sticks : carrot_sticks_before_dinner + carrot_sticks_after_dinner = 37 :=
  by sorry

end NUMINAMATH_GPT_total_carrot_sticks_l210_21000


namespace NUMINAMATH_GPT_points_description_l210_21066

noncomputable def clubsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem points_description (x y : ℝ) : 
  (clubsuit x y = clubsuit y x) ↔ (x = 0) ∨ (y = 0) ∨ (x = y) ∨ (x + y = 0) := 
by 
  sorry

end NUMINAMATH_GPT_points_description_l210_21066


namespace NUMINAMATH_GPT_alice_met_tweedledee_l210_21074

noncomputable def brother_statement (day : ℕ) : Prop :=
  sorry -- Define the exact logical structure of the statement "I am lying today, and my name is Tweedledum" here

theorem alice_met_tweedledee (day : ℕ) : brother_statement day → (∃ (b : String), b = "Tweedledee") :=
by
  sorry -- provide the proof here

end NUMINAMATH_GPT_alice_met_tweedledee_l210_21074


namespace NUMINAMATH_GPT_fraction_undefined_l210_21049

theorem fraction_undefined (x : ℝ) : (x + 1 = 0) ↔ (x = -1) := 
  sorry

end NUMINAMATH_GPT_fraction_undefined_l210_21049


namespace NUMINAMATH_GPT_range_of_x_minus_y_l210_21065

variable (x y : ℝ)
variable (h1 : 2 < x) (h2 : x < 4) (h3 : -1 < y) (h4 : y < 3)

theorem range_of_x_minus_y : -1 < x - y ∧ x - y < 5 := 
by {
  sorry
}

end NUMINAMATH_GPT_range_of_x_minus_y_l210_21065


namespace NUMINAMATH_GPT_age_of_15th_student_l210_21064

theorem age_of_15th_student
  (avg_age_15_students : ℕ)
  (total_students : ℕ)
  (avg_age_5_students : ℕ)
  (students_5 : ℕ)
  (avg_age_9_students : ℕ)
  (students_9 : ℕ)
  (total_age_15_students_eq : avg_age_15_students * total_students = 225)
  (total_age_5_students_eq : avg_age_5_students * students_5 = 70)
  (total_age_9_students_eq : avg_age_9_students * students_9 = 144) :
  (avg_age_15_students * total_students - (avg_age_5_students * students_5 + avg_age_9_students * students_9) = 11) :=
by
  sorry

end NUMINAMATH_GPT_age_of_15th_student_l210_21064


namespace NUMINAMATH_GPT_sin_cos_pow_eq_l210_21053

theorem sin_cos_pow_eq (sin cos : ℝ → ℝ) (x : ℝ) (h₀ : sin x + cos x = -1) (n : ℕ) : 
  sin x ^ n + cos x ^ n = (-1) ^ n :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_pow_eq_l210_21053


namespace NUMINAMATH_GPT_minimum_value_l210_21099

-- Define the expression E(a, b, c)
def E (a b c : ℝ) : ℝ := a^2 + 8 * a * b + 24 * b^2 + 16 * b * c + 6 * c^2

-- State the minimum value theorem
theorem minimum_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  E a b c = 18 :=
sorry

end NUMINAMATH_GPT_minimum_value_l210_21099


namespace NUMINAMATH_GPT_polar_to_rectangular_l210_21006

theorem polar_to_rectangular : 
  ∀ (r θ : ℝ), r = 2 ∧ θ = 2 * Real.pi / 3 → 
  (r * Real.cos θ, r * Real.sin θ) = (-1, Real.sqrt 3) := by
  sorry

end NUMINAMATH_GPT_polar_to_rectangular_l210_21006


namespace NUMINAMATH_GPT_parabola_relationship_l210_21038

theorem parabola_relationship 
  (c : ℝ) (y1 y2 y3 : ℝ) 
  (h1 : y1 = 2*(-2 - 1)^2 + c) 
  (h2 : y2 = 2*(0 - 1)^2 + c) 
  (h3 : y3 = 2*((5:ℝ)/3 - 1)^2 + c):
  y1 > y2 ∧ y2 > y3 :=
by
  sorry

end NUMINAMATH_GPT_parabola_relationship_l210_21038


namespace NUMINAMATH_GPT_trigonometric_inequality_equality_conditions_l210_21045

theorem trigonometric_inequality
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) :
  (1 / (Real.cos α)^2 + 1 / ((Real.sin α)^2 * (Real.sin β)^2 * (Real.cos β)^2)) ≥ 9 :=
sorry

theorem equality_conditions
  (α β : ℝ)
  (hα : α = Real.arctan (Real.sqrt 2))
  (hβ : β = π / 4) :
  (1 / (Real.cos α)^2 + 1 / ((Real.sin α)^2 * (Real.sin β)^2 * (Real.cos β)^2)) = 9 :=
sorry

end NUMINAMATH_GPT_trigonometric_inequality_equality_conditions_l210_21045


namespace NUMINAMATH_GPT_geometric_sequence_b_value_l210_21019

theorem geometric_sequence_b_value (b : ℝ) (h1 : 25 * b = b^2) (h2 : b * (1 / 4) = b / 4) :
  b = 5 / 2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_b_value_l210_21019


namespace NUMINAMATH_GPT_exists_diff_shape_and_color_l210_21050

variable (Pitcher : Type) 
variable (shape color : Pitcher → Prop)
variable (exists_diff_shape : ∃ (A B : Pitcher), shape A ≠ shape B)
variable (exists_diff_color : ∃ (A B : Pitcher), color A ≠ color B)

theorem exists_diff_shape_and_color : ∃ (A B : Pitcher), shape A ≠ shape B ∧ color A ≠ color B :=
  sorry

end NUMINAMATH_GPT_exists_diff_shape_and_color_l210_21050


namespace NUMINAMATH_GPT_simplify_fraction_l210_21051

theorem simplify_fraction (i : ℂ) (h : i^2 = -1) : 
  (2 - i) / (1 + 4 * i) = -2 / 17 - (9 / 17) * i :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l210_21051


namespace NUMINAMATH_GPT_total_number_of_notes_l210_21055

theorem total_number_of_notes 
  (total_money : ℕ)
  (fifty_rupees_notes : ℕ)
  (five_hundred_rupees_notes : ℕ)
  (total_money_eq : total_money = 10350)
  (fifty_rupees_notes_eq : fifty_rupees_notes = 117)
  (money_eq : 50 * fifty_rupees_notes + 500 * five_hundred_rupees_notes = total_money) :
  fifty_rupees_notes + five_hundred_rupees_notes = 126 :=
by sorry

end NUMINAMATH_GPT_total_number_of_notes_l210_21055


namespace NUMINAMATH_GPT_divide_0_24_by_0_004_l210_21018

theorem divide_0_24_by_0_004 : 0.24 / 0.004 = 60 := by
  sorry

end NUMINAMATH_GPT_divide_0_24_by_0_004_l210_21018


namespace NUMINAMATH_GPT_coeff_sum_eq_neg_two_l210_21001

theorem coeff_sum_eq_neg_two (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x^10 + x^4 + 1) = a + a₁ * (x+1) + a₂ * (x+1)^2 + a₃ * (x+1)^3 + a₄ * (x+1)^4 
   + a₅ * (x+1)^5 + a₆ * (x+1)^6 + a₇ * (x+1)^7 + a₈ * (x+1)^8 + a₉ * (x+1)^9 + a₁₀ * (x+1)^10) 
  → (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = -2) := 
by sorry

end NUMINAMATH_GPT_coeff_sum_eq_neg_two_l210_21001


namespace NUMINAMATH_GPT_determine_parabola_coefficients_l210_21041

noncomputable def parabola_coefficients (a b c : ℚ) : Prop :=
  ∀ (x y : ℚ), 
      (y = a * x^2 + b * x + c) ∧
      (
        ((4, 5) = (x, y)) ∧
        ((2, 3) = (x, y))
      )

theorem determine_parabola_coefficients :
  parabola_coefficients (-1/2) 4 (-3) :=
by
  sorry

end NUMINAMATH_GPT_determine_parabola_coefficients_l210_21041


namespace NUMINAMATH_GPT_smallest_value_of_a_b_l210_21096

theorem smallest_value_of_a_b :
  ∃ (a b : ℤ), (∀ x : ℤ, ((x^2 + a*x + 20) = 0 ∨ (x^2 + 17*x + b) = 0) → x < 0) ∧ a + b = -5 :=
sorry

end NUMINAMATH_GPT_smallest_value_of_a_b_l210_21096


namespace NUMINAMATH_GPT_range_of_a_l210_21003

-- Given definitions from the problem
def p (a : ℝ) : Prop :=
  (4 - 4 * a) > 0

def q (a : ℝ) : Prop :=
  (a - 3) * (a + 1) < 0

-- The theorem we want to prove
theorem range_of_a (a : ℝ) : ¬ (p a ∨ q a) ↔ a ≥ 3 := 
by sorry

end NUMINAMATH_GPT_range_of_a_l210_21003


namespace NUMINAMATH_GPT_find_a_l210_21089

variable {x : ℝ} {a b : ℝ}

def setA : Set ℝ := {x | Real.log x / Real.log 2 > 1}
def setB (a : ℝ) : Set ℝ := {x | x < a}
def setIntersection (b : ℝ) : Set ℝ := {x | b < x ∧ x < 2 * b + 3}

theorem find_a (h : setA ∩ setB a = setIntersection b) : a = 7 := 
by
  sorry

end NUMINAMATH_GPT_find_a_l210_21089


namespace NUMINAMATH_GPT_power_calculation_l210_21059

noncomputable def a : ℕ := 3 ^ 1006
noncomputable def b : ℕ := 7 ^ 1007
noncomputable def lhs : ℕ := (a + b)^2 - (a - b)^2
noncomputable def rhs : ℕ := 42 * (10 ^ 1007)

theorem power_calculation : lhs = rhs := by
  sorry

end NUMINAMATH_GPT_power_calculation_l210_21059


namespace NUMINAMATH_GPT_min_expression_value_l210_21094

noncomputable def expression (x y : ℝ) : ℝ := 2*x^2 + 2*y^2 - 8*x + 6*y + 25

theorem min_expression_value : ∃ (x y : ℝ), expression x y = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_min_expression_value_l210_21094


namespace NUMINAMATH_GPT_initial_girls_l210_21009

theorem initial_girls (G : ℕ) (h : G + 682 = 1414) : G = 732 := 
by
  sorry

end NUMINAMATH_GPT_initial_girls_l210_21009


namespace NUMINAMATH_GPT_xiao_hua_seat_correct_l210_21078

-- Define the classroom setup
def classroom : Type := ℤ × ℤ

-- Define the total number of rows and columns in the classroom.
def total_rows : ℤ := 7
def total_columns : ℤ := 8

-- Define the position of Xiao Ming's seat.
def xiao_ming_seat : classroom := (3, 7)

-- Define the position of Xiao Hua's seat.
def xiao_hua_seat : classroom := (5, 2)

-- Prove that Xiao Hua's seat is designated as (5, 2)
theorem xiao_hua_seat_correct : xiao_hua_seat = (5, 2) := by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_xiao_hua_seat_correct_l210_21078


namespace NUMINAMATH_GPT_problem_statement_l210_21076

theorem problem_statement (a x : ℝ) (h_linear_eq : (a + 4) * x ^ |a + 3| + 8 = 0) : a^2 + a - 1 = 1 :=
sorry

end NUMINAMATH_GPT_problem_statement_l210_21076


namespace NUMINAMATH_GPT_part1_part2_min_part2_max_part3_l210_21047

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 2 * a / x - 3 * Real.log x

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := a + 2 * a / (x^2) - 3 / x

theorem part1 (a : ℝ) : f' a 1 = 0 -> a = 1 := sorry

noncomputable def f1 (x : ℝ) : ℝ := x - 2 / x - 3 * Real.log x

noncomputable def f1' (x : ℝ) : ℝ := 1 + 2 / (x^2) - 3 / x

theorem part2_min (h_a : 1 = 1) : 
    ∀ (x : ℝ), (1 ≤ x) ∧ (x ≤ Real.exp 1) -> 
    (f1 2 <= f1 x) := sorry

theorem part2_max (h_a : 1 = 1) : 
    ∀ (x : ℝ), (1 ≤ x) ∧ (x ≤ Real.exp 1) ->
    (f1 x <= f1 1) := sorry

theorem part3 (a : ℝ) : 
    (∀ (x : ℝ), x > 0 -> f' a x ≥ 0) -> a ≥ (3 * Real.sqrt 2) / 4 := sorry

end NUMINAMATH_GPT_part1_part2_min_part2_max_part3_l210_21047


namespace NUMINAMATH_GPT_latus_rectum_of_parabola_l210_21085

theorem latus_rectum_of_parabola (x : ℝ) :
  (∀ x, y = (-1 / 4 : ℝ) * x^2) → y = (-1 / 2 : ℝ) :=
sorry

end NUMINAMATH_GPT_latus_rectum_of_parabola_l210_21085


namespace NUMINAMATH_GPT_unsold_books_l210_21022

-- Definitions from conditions
def books_total : ℕ := 150
def books_sold : ℕ := (2 / 3) * books_total
def book_price : ℕ := 5
def total_received : ℕ := 500

-- Proof statement
theorem unsold_books :
  (books_sold * book_price = total_received) →
  (books_total - books_sold = 50) :=
by
  sorry

end NUMINAMATH_GPT_unsold_books_l210_21022


namespace NUMINAMATH_GPT_cathy_initial_money_l210_21056

-- Definitions of the conditions
def moneyFromDad : Int := 25
def moneyFromMom : Int := 2 * moneyFromDad
def totalMoneyReceived : Int := moneyFromDad + moneyFromMom
def currentMoney : Int := 87

-- Theorem stating the proof problem
theorem cathy_initial_money (initialMoney : Int) :
  initialMoney + totalMoneyReceived = currentMoney → initialMoney = 12 :=
by
  sorry

end NUMINAMATH_GPT_cathy_initial_money_l210_21056


namespace NUMINAMATH_GPT_frog_climb_time_l210_21092

-- Define the problem as an assertion within Lean.
theorem frog_climb_time 
  (well_depth : ℕ) (climb_up : ℕ) (slide_down : ℕ) (time_per_meter: ℕ) (climb_start_time : ℕ) 
  (time_to_slide_multiplier: ℚ)
  (time_to_second_position: ℕ) 
  (final_distance: ℕ) 
  (total_time: ℕ)
  (h_start : well_depth = 12)
  (h_climb_up: climb_up = 3)
  (h_slide_down : slide_down = 1)
  (h_time_per_meter : time_per_meter = 1)
  (h_time_to_slide_multiplier: time_to_slide_multiplier = 1/3)
  (h_time_to_second_position : climb_start_time = 8 * 60 /\ time_to_second_position = 8 * 60 + 17)
  (h_final_distance : final_distance = 3)
  (h_total_time: total_time = 22) :
  
  ∃ (t: ℕ), 
    t = total_time := 
by
  sorry

end NUMINAMATH_GPT_frog_climb_time_l210_21092


namespace NUMINAMATH_GPT_inverse_proportionality_ratio_l210_21030

variable {x y k x1 x2 y1 y2 : ℝ}

theorem inverse_proportionality_ratio
  (h1 : x * y = k)
  (hx1 : x1 ≠ 0)
  (hx2 : x2 ≠ 0)
  (hy1 : y1 ≠ 0)
  (hy2 : y2 ≠ 0)
  (hx_ratio : x1 / x2 = 3 / 4)
  (hxy1 : x1 * y1 = k)
  (hxy2 : x2 * y2 = k) :
  y1 / y2 = 4 / 3 := by
  sorry

end NUMINAMATH_GPT_inverse_proportionality_ratio_l210_21030


namespace NUMINAMATH_GPT_cylinder_cone_volume_ratio_l210_21020

theorem cylinder_cone_volume_ratio (h r_cylinder r_cone : ℝ)
  (hcylinder_csa : π * r_cylinder^2 = π * r_cone^2 / 4):
  (π * r_cylinder^2 * h) / (1 / 3 * π * r_cone^2 * h) = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_cone_volume_ratio_l210_21020


namespace NUMINAMATH_GPT_vertex_of_given_function_l210_21043

-- Definition of the given quadratic function
def given_function (x : ℝ) : ℝ := 2 * (x - 4) ^ 2 + 5

-- Definition of the vertex coordinates
def vertex_coordinates : ℝ × ℝ := (4, 5)

-- Theorem stating the vertex coordinates of the function
theorem vertex_of_given_function : (0, given_function 4) = vertex_coordinates :=
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_vertex_of_given_function_l210_21043


namespace NUMINAMATH_GPT_total_vegetarian_is_33_l210_21087

-- Definitions of the quantities involved
def only_vegetarian : Nat := 19
def both_vegetarian_non_vegetarian : Nat := 12
def vegan_strictly_vegetarian : Nat := 3
def vegan_non_vegetarian : Nat := 2

-- The total number of people consuming vegetarian dishes
def total_vegetarian_consumers : Nat := only_vegetarian + both_vegetarian_non_vegetarian + vegan_non_vegetarian

-- Prove the number of people consuming vegetarian dishes
theorem total_vegetarian_is_33 :
  total_vegetarian_consumers = 33 :=
sorry

end NUMINAMATH_GPT_total_vegetarian_is_33_l210_21087


namespace NUMINAMATH_GPT_closure_of_M_is_closed_interval_l210_21062

noncomputable def U : Set ℝ := Set.univ

noncomputable def M : Set ℝ := {a | a^2 - 2 * a > 0}

theorem closure_of_M_is_closed_interval :
  closure M = {a | 0 ≤ a ∧ a ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_closure_of_M_is_closed_interval_l210_21062


namespace NUMINAMATH_GPT_solve_for_x_l210_21080

def delta (x : ℝ) : ℝ := 4 * x + 5
def phi (x : ℝ) : ℝ := 6 * x + 3

theorem solve_for_x (x : ℝ) (h : delta (phi x) = -1) : x = - 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l210_21080


namespace NUMINAMATH_GPT_least_number_of_cookies_l210_21073

theorem least_number_of_cookies :
  ∃ x : ℕ, x % 6 = 4 ∧ x % 5 = 3 ∧ x % 8 = 6 ∧ x % 9 = 7 ∧ x = 208 :=
by
  sorry

end NUMINAMATH_GPT_least_number_of_cookies_l210_21073


namespace NUMINAMATH_GPT_robin_extra_drinks_l210_21027

-- Conditions
def initial_sodas : ℕ := 22
def initial_energy_drinks : ℕ := 15
def initial_smoothies : ℕ := 12
def drank_sodas : ℕ := 6
def drank_energy_drinks : ℕ := 9
def drank_smoothies : ℕ := 2

-- Total drinks bought
def total_drinks_bought : ℕ :=
  initial_sodas + initial_energy_drinks + initial_smoothies
  
-- Total drinks consumed
def total_drinks_consumed : ℕ :=
  drank_sodas + drank_energy_drinks + drank_smoothies

-- Number of extra drinks
def extra_drinks : ℕ :=
  total_drinks_bought - total_drinks_consumed

-- Theorem to prove
theorem robin_extra_drinks : extra_drinks = 32 :=
  by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_robin_extra_drinks_l210_21027


namespace NUMINAMATH_GPT_average_food_per_week_l210_21071

-- Definitions based on conditions
def food_first_dog := 13
def food_second_dog := 2 * food_first_dog
def food_third_dog := 6
def number_of_dogs := 3

-- Statement of the proof problem
theorem average_food_per_week : 
  (food_first_dog + food_second_dog + food_third_dog) / number_of_dogs = 15 := 
by sorry

end NUMINAMATH_GPT_average_food_per_week_l210_21071


namespace NUMINAMATH_GPT_not_hexagonal_pyramid_l210_21079

-- Definition of the pyramid with slant height, base radius, and height
structure Pyramid where
  r : ℝ  -- Side length of the base equilateral triangle
  h : ℝ  -- Height of the pyramid
  l : ℝ  -- Slant height (lateral edge)
  hypo : h^2 + (r / 2)^2 = l^2

-- The theorem to prove a pyramid with all edges equal cannot be hexagonal
theorem not_hexagonal_pyramid (p : Pyramid) : p.l ≠ p.r :=
sorry

end NUMINAMATH_GPT_not_hexagonal_pyramid_l210_21079


namespace NUMINAMATH_GPT_price_of_item_a_l210_21028

theorem price_of_item_a : 
  let coins_1000 := 7
  let coins_100 := 4
  let coins_10 := 5
  let price_1000 := coins_1000 * 1000
  let price_100 := coins_100 * 100
  let price_10 := coins_10 * 10
  let total_price := price_1000 + price_100 + price_10
  total_price = 7450 := by
    sorry

end NUMINAMATH_GPT_price_of_item_a_l210_21028


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l210_21005

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (h_seq : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_a3 : a 3 = 5) (h_a5 : a 5 = 9) :
  S 7 = 49 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l210_21005
