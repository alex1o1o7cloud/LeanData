import Mathlib

namespace NUMINAMATH_GPT_triangle_XOY_hypotenuse_l0_9

theorem triangle_XOY_hypotenuse (a b : ℝ) (h1 : (a/2)^2 + b^2 = 22^2) (h2 : a^2 + (b/2)^2 = 19^2) :
  Real.sqrt (a^2 + b^2) = 26 :=
sorry

end NUMINAMATH_GPT_triangle_XOY_hypotenuse_l0_9


namespace NUMINAMATH_GPT_number_of_valid_pairs_l0_25

theorem number_of_valid_pairs : 
  ∃ (n : ℕ), n = 1995003 ∧ (∃ b c : ℤ, c < 2000 ∧ b > 2 ∧ (∀ x : ℂ, x^2 - (b:ℝ) * x + (c:ℝ) = 0 → x.re > 1)) := 
sorry

end NUMINAMATH_GPT_number_of_valid_pairs_l0_25


namespace NUMINAMATH_GPT_sum_even_integers_less_than_100_l0_45

theorem sum_even_integers_less_than_100 : 
  let sequence := List.range' 2 98
  let even_seq := sequence.filter (λ x => x % 2 = 0)
  (even_seq.sum) = 2450 :=
by
  sorry

end NUMINAMATH_GPT_sum_even_integers_less_than_100_l0_45


namespace NUMINAMATH_GPT_fraction_studying_japanese_l0_93

variable (J S : ℕ)
variable (hS : S = 3 * J)

def fraction_of_seniors_studying_japanese := (1 / 3 : ℚ) * S
def fraction_of_juniors_studying_japanese := (3 / 4 : ℚ) * J

def total_students := S + J

theorem fraction_studying_japanese (J S : ℕ) (hS : S = 3 * J) :
  ((1 / 3 : ℚ) * S + (3 / 4 : ℚ) * J) / (S + J) = 7 / 16 :=
by {
  -- proof to be filled in
  sorry
}

end NUMINAMATH_GPT_fraction_studying_japanese_l0_93


namespace NUMINAMATH_GPT_number_division_l0_5

theorem number_division (N x : ℕ) 
  (h1 : (N - 5) / x = 7) 
  (h2 : (N - 34) / 10 = 2)
  : x = 7 := 
by
  sorry

end NUMINAMATH_GPT_number_division_l0_5


namespace NUMINAMATH_GPT_binomial_multiplication_subtraction_l0_50

variable (x : ℤ)

theorem binomial_multiplication_subtraction :
  (4 * x - 3) * (x + 6) - ( (2 * x + 1) * (x - 4) ) = 2 * x^2 + 28 * x - 14 := by
  sorry

end NUMINAMATH_GPT_binomial_multiplication_subtraction_l0_50


namespace NUMINAMATH_GPT_total_area_of_paintings_l0_20

-- Definitions based on the conditions
def painting1_area := 3 * (5 * 5) -- 3 paintings of 5 feet by 5 feet
def painting2_area := 10 * 8 -- 1 painting of 10 feet by 8 feet
def painting3_area := 5 * 9 -- 1 painting of 5 feet by 9 feet

-- The proof statement we aim to prove
theorem total_area_of_paintings : painting1_area + painting2_area + painting3_area = 200 :=
by
  sorry

end NUMINAMATH_GPT_total_area_of_paintings_l0_20


namespace NUMINAMATH_GPT_sochi_apartment_price_decrease_l0_46

theorem sochi_apartment_price_decrease (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  let moscow_rub_decrease := 0.2
  let moscow_eur_decrease := 0.4
  let sochi_rub_decrease := 0.1
  let new_moscow_rub := (1 - moscow_rub_decrease) * a
  let new_moscow_eur := (1 - moscow_eur_decrease) * b
  let ruble_to_euro := new_moscow_rub / new_moscow_eur
  let new_sochi_rub := (1 - sochi_rub_decrease) * a
  let new_sochi_eur := new_sochi_rub / ruble_to_euro
  let decrease_percentage := (b - new_sochi_eur) / b * 100
  decrease_percentage = 32.5 :=
by
  sorry

end NUMINAMATH_GPT_sochi_apartment_price_decrease_l0_46


namespace NUMINAMATH_GPT_lives_after_game_l0_27

theorem lives_after_game (l0 : ℕ) (ll : ℕ) (lg : ℕ) (lf : ℕ) : 
  l0 = 10 → ll = 4 → lg = 26 → lf = l0 - ll + lg → lf = 32 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end NUMINAMATH_GPT_lives_after_game_l0_27


namespace NUMINAMATH_GPT_exists_xn_gt_yn_l0_98

noncomputable def x_sequence : ℕ → ℝ := sorry
noncomputable def y_sequence : ℕ → ℝ := sorry

theorem exists_xn_gt_yn
    (x1 x2 y1 y2 : ℝ)
    (hx1 : 1 < x1)
    (hx2 : 1 < x2)
    (hy1 : 1 < y1)
    (hy2 : 1 < y2)
    (h_x_seq : ∀ n, x_sequence (n + 2) = x_sequence n + (x_sequence (n + 1))^2)
    (h_y_seq : ∀ n, y_sequence (n + 2) = (y_sequence n)^2 + y_sequence (n + 1)) :
    ∃ n : ℕ, x_sequence n > y_sequence n :=
sorry

end NUMINAMATH_GPT_exists_xn_gt_yn_l0_98


namespace NUMINAMATH_GPT_modular_inverse_of_2_mod_199_l0_81

theorem modular_inverse_of_2_mod_199 : (2 * 100) % 199 = 1 := 
by sorry

end NUMINAMATH_GPT_modular_inverse_of_2_mod_199_l0_81


namespace NUMINAMATH_GPT_max_value_of_quadratic_l0_82

theorem max_value_of_quadratic :
  ∃ y : ℝ, (∀ x : ℝ, y ≥ -x^2 + 5 * x - 4) ∧ y = 9 / 4 :=
sorry

end NUMINAMATH_GPT_max_value_of_quadratic_l0_82


namespace NUMINAMATH_GPT_smallest_positive_integer_n_l0_4

theorem smallest_positive_integer_n (n : ℕ) :
  (∃ n1 n2 n3 : ℕ, 5 * n = n1 ^ 5 ∧ 6 * n = n2 ^ 6 ∧ 7 * n = n3 ^ 7) →
  n = 2^5 * 3^5 * 5^4 * 7^6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_n_l0_4


namespace NUMINAMATH_GPT_exists_F_squared_l0_57

theorem exists_F_squared (n : ℕ) : ∃ F : ℕ → ℕ, ∀ n : ℕ, (F (F n)) = n^2 := 
sorry

end NUMINAMATH_GPT_exists_F_squared_l0_57


namespace NUMINAMATH_GPT_simplify_expression_l0_40

theorem simplify_expression (n : ℕ) : 
  (3 ^ (n + 5) - 3 * 3 ^ n) / (3 * 3 ^ (n + 4)) = 80 / 27 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l0_40


namespace NUMINAMATH_GPT_multiples_of_15_between_17_and_202_l0_71

theorem multiples_of_15_between_17_and_202 : 
  ∃ n : ℕ, (∀ k : ℤ, 17 < k * 15 ∧ k * 15 < 202 → k = n + 1) ∧ n = 12 :=
sorry

end NUMINAMATH_GPT_multiples_of_15_between_17_and_202_l0_71


namespace NUMINAMATH_GPT_rational_abs_neg_l0_83

theorem rational_abs_neg (a : ℚ) (h : abs a = -a) : a ≤ 0 :=
by 
  sorry

end NUMINAMATH_GPT_rational_abs_neg_l0_83


namespace NUMINAMATH_GPT_debate_club_girls_l0_85

theorem debate_club_girls (B G : ℕ) 
  (h1 : B + G = 22)
  (h2 : B + (1/3 : ℚ) * G = 14) : G = 12 :=
sorry

end NUMINAMATH_GPT_debate_club_girls_l0_85


namespace NUMINAMATH_GPT_sara_received_quarters_correct_l0_59

-- Define the initial number of quarters Sara had
def sara_initial_quarters : ℕ := 21

-- Define the total number of quarters Sara has now
def sara_total_quarters : ℕ := 70

-- Define the number of quarters Sara received from her dad
def sara_received_quarters : ℕ := 49

-- State that the number of quarters Sara received can be deduced by the difference
theorem sara_received_quarters_correct :
  sara_total_quarters = sara_initial_quarters + sara_received_quarters :=
by simp [sara_initial_quarters, sara_total_quarters, sara_received_quarters]

end NUMINAMATH_GPT_sara_received_quarters_correct_l0_59


namespace NUMINAMATH_GPT_gcd_pens_pencils_l0_16

theorem gcd_pens_pencils (pens : ℕ) (pencils : ℕ) (h1 : pens = 1048) (h2 : pencils = 828) : Nat.gcd pens pencils = 4 := 
by
  -- Given: pens = 1048 and pencils = 828
  have h : pens = 1048 := h1
  have h' : pencils = 828 := h2
  sorry

end NUMINAMATH_GPT_gcd_pens_pencils_l0_16


namespace NUMINAMATH_GPT_bridge_length_l0_88

   noncomputable def walking_speed_km_per_hr : ℝ := 6
   noncomputable def walking_time_minutes : ℝ := 15

   noncomputable def length_of_bridge (speed_km_per_hr : ℝ) (time_min : ℝ) : ℝ :=
     (speed_km_per_hr * 1000 / 60) * time_min

   theorem bridge_length :
     length_of_bridge walking_speed_km_per_hr walking_time_minutes = 1500 := 
   by
     sorry
   
end NUMINAMATH_GPT_bridge_length_l0_88


namespace NUMINAMATH_GPT_initial_bottles_l0_8

theorem initial_bottles (x : ℕ) (h1 : x - 8 + 45 = 51) : x = 14 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_initial_bottles_l0_8


namespace NUMINAMATH_GPT_carla_marbles_l0_31

theorem carla_marbles (m : ℕ) : m + 134 = 187 ↔ m = 53 :=
by sorry

end NUMINAMATH_GPT_carla_marbles_l0_31


namespace NUMINAMATH_GPT_min_value_x_plus_one_over_2x_l0_61

theorem min_value_x_plus_one_over_2x (x : ℝ) (hx : x > 0) : 
  x + 1 / (2 * x) ≥ Real.sqrt 2 := sorry

end NUMINAMATH_GPT_min_value_x_plus_one_over_2x_l0_61


namespace NUMINAMATH_GPT_investments_interest_yielded_l0_34

def total_investment : ℝ := 15000
def part_one_investment : ℝ := 8200
def rate_one : ℝ := 0.06
def rate_two : ℝ := 0.075

def part_two_investment : ℝ := total_investment - part_one_investment

def interest_one : ℝ := part_one_investment * rate_one * 1
def interest_two : ℝ := part_two_investment * rate_two * 1

def total_interest : ℝ := interest_one + interest_two

theorem investments_interest_yielded : total_interest = 1002 := by
  sorry

end NUMINAMATH_GPT_investments_interest_yielded_l0_34


namespace NUMINAMATH_GPT_simplify_and_evaluate_expr_l0_99

theorem simplify_and_evaluate_expr (a : ℝ) (h1 : -1 < a) (h2 : a < Real.sqrt 5) (h3 : a = 2) :
  (a - (a^2 / (a^2 - 1))) / (a^2 / (a^2 - 1)) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expr_l0_99


namespace NUMINAMATH_GPT_apples_to_pears_value_l0_37

/-- Suppose 1/2 of 12 apples are worth as much as 10 pears. -/
def apples_per_pears_ratio : ℚ := 10 / (1 / 2 * 12)

/-- Prove that 3/4 of 6 apples are worth as much as 7.5 pears. -/
theorem apples_to_pears_value : (3 / 4 * 6) * apples_per_pears_ratio = 7.5 := 
by
  sorry

end NUMINAMATH_GPT_apples_to_pears_value_l0_37


namespace NUMINAMATH_GPT_percentage_discount_l0_75

theorem percentage_discount (P S : ℝ) (hP : P = 50) (hS : S = 35) : (P - S) / P * 100 = 30 := by
  sorry

end NUMINAMATH_GPT_percentage_discount_l0_75


namespace NUMINAMATH_GPT_problem_statement_l0_97

theorem problem_statement (x : ℤ) (h : Even (3 * x + 1)) : Odd (7 * x + 4) :=
  sorry

end NUMINAMATH_GPT_problem_statement_l0_97


namespace NUMINAMATH_GPT_solve_for_x_l0_21

theorem solve_for_x (x : ℝ) (h: (6 / (x + 1) = 3 / 2)) : x = 3 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l0_21


namespace NUMINAMATH_GPT_ratio_sum_is_four_l0_6

theorem ratio_sum_is_four
  (x y : ℝ)
  (hx : 0 < x) (hy : 0 < y)
  (θ : ℝ)
  (hθ_ne : ∀ n : ℤ, θ ≠ (n * (π / 2)))
  (h1 : (Real.sin θ) / x = (Real.cos θ) / y)
  (h2 : (Real.cos θ)^4 / x^4 + (Real.sin θ)^4 / y^4 = 97 * (Real.sin (2 * θ)) / (x^3 * y + y^3 * x)) :
  (x / y) + (y / x) = 4 := by
  sorry

end NUMINAMATH_GPT_ratio_sum_is_four_l0_6


namespace NUMINAMATH_GPT_set_theorem_1_set_theorem_2_set_theorem_3_set_theorem_4_set_theorem_5_set_theorem_6_set_theorem_7_l0_14

variable {U : Type} [DecidableEq U]
variables (A B C K : Set U)

theorem set_theorem_1 : (A \ K) ∪ (B \ K) = (A ∪ B) \ K := sorry
theorem set_theorem_2 : A \ (B \ C) = (A \ B) ∪ (A ∩ C) := sorry
theorem set_theorem_3 : A \ (A \ B) = A ∩ B := sorry
theorem set_theorem_4 : (A \ B) \ C = (A \ C) \ (B \ C) := sorry
theorem set_theorem_5 : A \ (B ∩ C) = (A \ B) ∪ (A \ C) := sorry
theorem set_theorem_6 : A \ (B ∪ C) = (A \ B) ∩ (A \ C) := sorry
theorem set_theorem_7 : A \ B = (A ∪ B) \ B ∧ A \ B = A \ (A ∩ B) := sorry

end NUMINAMATH_GPT_set_theorem_1_set_theorem_2_set_theorem_3_set_theorem_4_set_theorem_5_set_theorem_6_set_theorem_7_l0_14


namespace NUMINAMATH_GPT_largest_integer_less_than_hundred_with_remainder_five_l0_77

theorem largest_integer_less_than_hundred_with_remainder_five (n : ℤ) :
  n < 100 ∧ n % 8 = 5 → n = 93 :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_less_than_hundred_with_remainder_five_l0_77


namespace NUMINAMATH_GPT_calories_per_one_bar_l0_12

variable (total_calories : ℕ) (num_bars : ℕ)
variable (calories_per_bar : ℕ)

-- Given conditions
axiom total_calories_given : total_calories = 15
axiom num_bars_given : num_bars = 5

-- Mathematical equivalent proof problem
theorem calories_per_one_bar :
  total_calories / num_bars = calories_per_bar →
  calories_per_bar = 3 :=
by
  sorry

end NUMINAMATH_GPT_calories_per_one_bar_l0_12


namespace NUMINAMATH_GPT_approximation_accuracy_l0_68

noncomputable def radius (k : Circle) : ℝ := sorry
def BG_equals_radius (BG : ℝ) (r : ℝ) := BG = r
def DB_equals_radius_sqrt3 (DB DG r : ℝ) := DB = DG ∧ DG = r * Real.sqrt 3
def cos_beta (cos_beta : ℝ) := cos_beta = 1 / (2 * Real.sqrt 3)
def sin_beta (sin_beta : ℝ) := sin_beta = Real.sqrt 11 / (2 * Real.sqrt 3)
def angle_BCH (angle_BCH : ℝ) (beta : ℝ) := angle_BCH = 120 - beta
def side_nonagon (a_9 r : ℝ) := a_9 = 2 * r * Real.sin 20
def bounds_sin_20 (sin_20 : ℝ) := 0.34195 < sin_20 ∧ sin_20 < 0.34205
def error_margin_low (BH_low a_9 r : ℝ) := 0.6839 * r < a_9
def error_margin_high (BH_high a_9 r : ℝ) := a_9 < 0.6841 * r

theorem approximation_accuracy
  (r : ℝ) (BG DB DG : ℝ) (beta : ℝ) (a_9 BH_low BH_high : ℝ)
  (h1 : BG_equals_radius BG r)
  (h2 : DB_equals_radius_sqrt3 DB DG r)
  (h3 : cos_beta (1 / (2 * Real.sqrt 3)))
  (h4 : sin_beta (Real.sqrt 11 / (2 * Real.sqrt 3)))
  (h5 : angle_BCH (120 - beta) beta)
  (h6 : side_nonagon a_9 r)
  (h7 : bounds_sin_20 (Real.sin 20))
  (h8 : error_margin_low BH_low a_9 r)
  (h9 : error_margin_high BH_high a_9 r) : 
  0.6861 * r < BH_high ∧ BH_low < 0.6864 * r :=
sorry

end NUMINAMATH_GPT_approximation_accuracy_l0_68


namespace NUMINAMATH_GPT_distance_between_opposite_vertices_l0_42

noncomputable def calculate_d (a b c v k t : ℝ) : ℝ :=
  (1 / (2 * k)) * Real.sqrt (2 * (k^4 - 16 * t^2 - 8 * v * k))

theorem distance_between_opposite_vertices (a b c v k t d : ℝ)
  (h1 : v = a * b * c)
  (h2 : k = a + b + c)
  (h3 : 16 * t^2 = k * (k - 2 * a) * (k - 2 * b) * (k - 2 * c))
  : d = calculate_d a b c v k t := 
by {
    -- The proof is omitted based on the requirement.
    sorry
}

end NUMINAMATH_GPT_distance_between_opposite_vertices_l0_42


namespace NUMINAMATH_GPT_find_number_l0_36

theorem find_number (x n : ℕ) (h1 : 3 * x + n = 48) (h2 : x = 4) : n = 36 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l0_36


namespace NUMINAMATH_GPT_sum_of_coordinates_of_D_l0_49

theorem sum_of_coordinates_of_D (x y : ℝ) (h1 : (x + 6) / 2 = 2) (h2 : (y + 2) / 2 = 6) :
  x + y = 8 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_D_l0_49


namespace NUMINAMATH_GPT_arithmetic_sequence_properties_l0_76

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n, S n = n * (a 1 + a n) / 2

def condition_S10_pos (S : ℕ → ℝ) : Prop :=
S 10 > 0

def condition_S11_neg (S : ℕ → ℝ) : Prop :=
S 11 < 0

-- Main statement
theorem arithmetic_sequence_properties {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}
  (ar_seq : is_arithmetic_sequence a d)
  (sum_first_n : sum_of_first_n_terms S a)
  (S10_pos : condition_S10_pos S)
  (S11_neg : condition_S11_neg S) :
  (∀ n, (S n) / n = a 1 + (n - 1) / 2 * d) ∧
  (a 2 = 1 → -2 / 7 < d ∧ d < -1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_properties_l0_76


namespace NUMINAMATH_GPT_seq_eq_exp_l0_22

theorem seq_eq_exp (a : ℕ → ℕ) 
  (h₀ : a 1 = 2) 
  (h₁ : ∀ n ≥ 2, a n = 2 * a (n - 1) - 1) :
  ∀ n ≥ 2, a n = 2^(n-1) + 1 := 
  by 
  sorry

end NUMINAMATH_GPT_seq_eq_exp_l0_22


namespace NUMINAMATH_GPT_pyramid_volume_l0_30

noncomputable def volume_of_pyramid (a h : ℝ) : ℝ :=
  (a^2 * h) / (4 * Real.sqrt 3)

theorem pyramid_volume (d x y : ℝ) (a h : ℝ) (edge_distance lateral_face_distance : ℝ)
  (H1 : edge_distance = 2) (H2 : lateral_face_distance = Real.sqrt 12)
  (H3 : x = 2) (H4 : y = 2 * Real.sqrt 3) (H5 : d = (a * Real.sqrt 3) / 6)
  (H6 : h = Real.sqrt (48 / 5)) :
  volume_of_pyramid a h = 216 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_pyramid_volume_l0_30


namespace NUMINAMATH_GPT_ratio_price_16_to_8_l0_96

def price_8_inch := 5
def P : ℝ := sorry
def price_16_inch := 5 * P
def daily_earnings := 3 * price_8_inch + 5 * price_16_inch
def three_day_earnings := 3 * daily_earnings
def total_earnings := 195

theorem ratio_price_16_to_8 : total_earnings = three_day_earnings → P = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_price_16_to_8_l0_96


namespace NUMINAMATH_GPT_selene_sandwiches_l0_70

-- Define the context and conditions in Lean
variables (S : ℕ) (sandwich_cost hamburger_cost hotdog_cost juice_cost : ℕ)
  (selene_cost tanya_cost total_cost : ℕ)

-- Each item prices
axiom sandwich_price : sandwich_cost = 2
axiom hamburger_price : hamburger_cost = 2
axiom hotdog_price : hotdog_cost = 1
axiom juice_price : juice_cost = 2

-- Purchases
axiom selene_purchase : selene_cost = sandwich_cost * S + juice_cost
axiom tanya_purchase : tanya_cost = hamburger_cost * 2 + juice_cost * 2

-- Total spending
axiom total_spending : selene_cost + tanya_cost = 16

-- Goal: Prove that Selene bought 3 sandwiches
theorem selene_sandwiches : S = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_selene_sandwiches_l0_70


namespace NUMINAMATH_GPT_center_of_circle_sum_l0_17
-- Import the entire library

-- Define the problem using declarations for conditions and required proof
theorem center_of_circle_sum (x y : ℝ) 
  (h : ∀ x y : ℝ, x^2 + y^2 - 4 * x + 6 * y = 9 → (x = 2) ∧ (y = -3)) : 
  x + y = -1 := 
by 
  sorry 

end NUMINAMATH_GPT_center_of_circle_sum_l0_17


namespace NUMINAMATH_GPT_find_larger_number_l0_15

theorem find_larger_number (a b : ℝ) (h1 : a + b = 40) (h2 : a - b = 10) : a = 25 :=
  sorry

end NUMINAMATH_GPT_find_larger_number_l0_15


namespace NUMINAMATH_GPT_polygon_sides_l0_90

theorem polygon_sides (n : ℕ) (h : 180 * (n - 2) = 720) : n = 6 :=
sorry

end NUMINAMATH_GPT_polygon_sides_l0_90


namespace NUMINAMATH_GPT_trig_identity_l0_3

theorem trig_identity (α : ℝ) (h0 : Real.tan α = Real.sqrt 3) (h1 : π < α) (h2 : α < 3 * π / 2) :
  Real.cos (2 * α) - Real.sin (π / 2 + α) = 0 :=
sorry

end NUMINAMATH_GPT_trig_identity_l0_3


namespace NUMINAMATH_GPT_sum_of_squares_of_projections_constant_l0_32

-- Defines a function that calculates the sum of the squares of the projections of the edges of a cube onto any plane.
def sum_of_squares_of_projections (a : ℝ) (n : ℝ × ℝ × ℝ) : ℝ :=
  let α := n.1
  let β := n.2.1
  let γ := n.2.2
  4 * (a^2) * (2)

-- Define the theorem statement that proves the sum of the squares of the projections is constant and equal to 8a^2
theorem sum_of_squares_of_projections_constant (a : ℝ) (n : ℝ × ℝ × ℝ) :
  sum_of_squares_of_projections a n = 8 * a^2 :=
by
  -- Since we assume the trigonometric identity holds, directly match the sum_of_squares_of_projections function result.
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_projections_constant_l0_32


namespace NUMINAMATH_GPT_not_divisible_by_11_check_divisibility_by_11_l0_26

theorem not_divisible_by_11 : Nat := 8

theorem check_divisibility_by_11 (n : Nat) (h: n = 98473092) : ¬ (11 ∣ not_divisible_by_11) := by
  sorry

end NUMINAMATH_GPT_not_divisible_by_11_check_divisibility_by_11_l0_26


namespace NUMINAMATH_GPT_average_mark_of_excluded_students_l0_47

noncomputable def average_mark_excluded (A : ℝ) (N : ℕ) (R : ℝ) (excluded_count : ℕ) (remaining_count : ℕ) : ℝ :=
  ((N : ℝ) * A - (remaining_count : ℝ) * R) / (excluded_count : ℝ)

theorem average_mark_of_excluded_students : 
  average_mark_excluded 70 10 90 5 5 = 50 := 
by 
  sorry

end NUMINAMATH_GPT_average_mark_of_excluded_students_l0_47


namespace NUMINAMATH_GPT_neg_p_equiv_l0_43

open Real
open Classical

noncomputable def prop_p : Prop :=
  ∀ x : ℝ, 0 < x → exp x > log x

noncomputable def neg_prop_p : Prop :=
  ∃ x : ℝ, 0 < x ∧ exp x ≤ log x

theorem neg_p_equiv :
  ¬ prop_p ↔ neg_prop_p := by
  sorry

end NUMINAMATH_GPT_neg_p_equiv_l0_43


namespace NUMINAMATH_GPT_no_valid_x_l0_35

theorem no_valid_x (x y : ℝ) (h : y = 2 * x) : ¬(3 * y ^ 2 - 2 * y + 5 = 2 * (6 * x ^ 2 - 3 * y + 3)) :=
by
  sorry

end NUMINAMATH_GPT_no_valid_x_l0_35


namespace NUMINAMATH_GPT_exp_gt_one_l0_29

theorem exp_gt_one (a x y : ℝ) (ha : 1 < a) (hxy : x > y) : a^x > a^y :=
sorry

end NUMINAMATH_GPT_exp_gt_one_l0_29


namespace NUMINAMATH_GPT_ethan_expected_wins_l0_78

-- Define the conditions
def P_win := 2 / 5
def P_tie := 2 / 5
def P_loss := 1 / 5

-- Define the adjusted probabilities
def adj_P_win := P_win / (P_win + P_loss)
def adj_P_loss := P_loss / (P_win + P_loss)

-- Define Ethan's expected number of wins before losing
def expected_wins_before_loss : ℚ := 2

-- The theorem to prove 
theorem ethan_expected_wins :
  ∃ E : ℚ, 
    E = (adj_P_win * (E + 1) + adj_P_loss * 0) ∧ 
    E = expected_wins_before_loss :=
by
  sorry

end NUMINAMATH_GPT_ethan_expected_wins_l0_78


namespace NUMINAMATH_GPT_expedition_ratios_l0_60

theorem expedition_ratios (F : ℕ) (S : ℕ) (L : ℕ) (R : ℕ) 
  (h1 : F = 3) 
  (h2 : S = F + 2) 
  (h3 : F + S + L = 18) 
  (h4 : L = R * S) : 
  R = 2 := 
sorry

end NUMINAMATH_GPT_expedition_ratios_l0_60


namespace NUMINAMATH_GPT_triangle_has_120_degree_l0_91

noncomputable def angles_of_triangle (α β γ : Real) : Prop :=
  α + β + γ = 180

theorem triangle_has_120_degree (α β γ : Real)
    (h1 : angles_of_triangle α β γ)
    (h2 : Real.cos (3 * α) + Real.cos (3 * β) + Real.cos (3 * γ) = 1) :
  γ = 120 :=
  sorry

end NUMINAMATH_GPT_triangle_has_120_degree_l0_91


namespace NUMINAMATH_GPT_will_new_cards_count_l0_11

-- Definitions based on conditions
def cards_per_page := 3
def pages_used := 6
def old_cards := 10

-- Proof statement (no proof, only the statement)
theorem will_new_cards_count : (pages_used * cards_per_page) - old_cards = 8 :=
by sorry

end NUMINAMATH_GPT_will_new_cards_count_l0_11


namespace NUMINAMATH_GPT_sandy_correct_sums_l0_48

theorem sandy_correct_sums (c i : ℕ) (h1 : c + i = 30) (h2 : 3 * c - 2 * i = 55) : c = 23 :=
by
  sorry

end NUMINAMATH_GPT_sandy_correct_sums_l0_48


namespace NUMINAMATH_GPT_find_m_l0_39

theorem find_m (x : ℝ) (m : ℝ) (h : ∃ x, (x - 2) ≠ 0 ∧ (4 - 2 * x) ≠ 0 ∧ (3 / (x - 2) + 1 = m / (4 - 2 * x))) : m = -6 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l0_39


namespace NUMINAMATH_GPT_total_length_of_free_sides_l0_41

theorem total_length_of_free_sides (L W : ℝ) 
  (h1 : L = 2 * W) 
  (h2 : L * W = 128) : 
  L + 2 * W = 32 := by 
sorry

end NUMINAMATH_GPT_total_length_of_free_sides_l0_41


namespace NUMINAMATH_GPT_size_of_each_group_l0_73

theorem size_of_each_group 
  (skittles : ℕ) (erasers : ℕ) (groups : ℕ)
  (h_skittles : skittles = 4502) (h_erasers : erasers = 4276) (h_groups : groups = 154) :
  (skittles + erasers) / groups = 57 :=
by
  sorry

end NUMINAMATH_GPT_size_of_each_group_l0_73


namespace NUMINAMATH_GPT_pen_tip_movement_l0_51

-- Definition of movements
def move_left (x : Int) : Int := -x
def move_right (x : Int) : Int := x

theorem pen_tip_movement :
  move_left 6 + move_right 3 = -3 :=
by
  sorry

end NUMINAMATH_GPT_pen_tip_movement_l0_51


namespace NUMINAMATH_GPT_election_votes_l0_95

theorem election_votes (V : ℝ) (h1 : 0.70 * V - 0.30 * V = 200) : V = 500 :=
sorry

end NUMINAMATH_GPT_election_votes_l0_95


namespace NUMINAMATH_GPT_no_such_integers_exist_l0_65

theorem no_such_integers_exist :
  ¬(∃ (a b c d : ℤ), a * 19^3 + b * 19^2 + c * 19 + d = 1 ∧ a * 62^3 + b * 62^2 + c * 62 + d = 2) :=
by
  sorry

end NUMINAMATH_GPT_no_such_integers_exist_l0_65


namespace NUMINAMATH_GPT_tangent_line_condition_l0_69

theorem tangent_line_condition (a b : ℝ):
  ((a = 1 ∧ b = 1) → ∀ x y : ℝ, x + y = 0 → (x - a)^2 + (y - b)^2 = 2 → x = 0 ∧ y = 0) ∧
  ( (a = -1 ∧ b = -1) → ∀ x y : ℝ, x + y = 0 → (x - a)^2 + (y - b)^2 = 2 → x = 0 ∧ y = 0) →
  (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = -1) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_condition_l0_69


namespace NUMINAMATH_GPT_solve_equation_l0_10

theorem solve_equation (x : ℝ) (h : x ≠ 2) :
  x^2 = (4*x^2 + 4) / (x - 2) ↔ (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 ∨ x = 4) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l0_10


namespace NUMINAMATH_GPT_sample_variance_is_two_l0_72

theorem sample_variance_is_two (a : ℝ) (h_avg : (a + 0 + 1 + 2 + 3) / 5 = 1) : 
  (1 / 5) * ((a - 1)^2 + (0 - 1)^2 + (1 - 1)^2 + (2 - 1)^2 + (3 - 1)^2) = 2 :=
sorry

end NUMINAMATH_GPT_sample_variance_is_two_l0_72


namespace NUMINAMATH_GPT_cos_A_minus_B_eq_nine_eighths_l0_56

theorem cos_A_minus_B_eq_nine_eighths (A B : ℝ)
  (h1 : Real.sin A + Real.sin B = 1 / 2)
  (h2 : Real.cos A + Real.cos B = 2) : 
  Real.cos (A - B) = 9 / 8 := 
by
  sorry

end NUMINAMATH_GPT_cos_A_minus_B_eq_nine_eighths_l0_56


namespace NUMINAMATH_GPT_triangle_non_existence_triangle_existence_l0_94

-- Definition of the triangle inequality theorem for a triangle with given sides.
def triangle_exists (a b c : ℕ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem triangle_non_existence (h : ¬ triangle_exists 2 3 7) : true := by
  sorry

theorem triangle_existence (h : triangle_exists 5 5 5) : true := by
  sorry

end NUMINAMATH_GPT_triangle_non_existence_triangle_existence_l0_94


namespace NUMINAMATH_GPT_distinct_lengths_from_E_to_DF_l0_13

noncomputable def distinct_integer_lengths (DE EF: ℕ) : ℕ :=
if h : DE = 15 ∧ EF = 36 then 24 else 0

theorem distinct_lengths_from_E_to_DF :
  distinct_integer_lengths 15 36 = 24 :=
by {
  sorry
}

end NUMINAMATH_GPT_distinct_lengths_from_E_to_DF_l0_13


namespace NUMINAMATH_GPT_evariste_stairs_l0_92

def num_ways (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 1
  else num_ways (n - 1) + num_ways (n - 2)

theorem evariste_stairs (n : ℕ) : num_ways n = u_n :=
  sorry

end NUMINAMATH_GPT_evariste_stairs_l0_92


namespace NUMINAMATH_GPT_corresponding_angles_equal_l0_67

theorem corresponding_angles_equal (α β γ : ℝ) (h1 : α + β + γ = 180) (h2 : α = 90) :
  (180 - α = 90 ∧ β + γ = 90 ∧ α = 90) :=
by
  sorry

end NUMINAMATH_GPT_corresponding_angles_equal_l0_67


namespace NUMINAMATH_GPT_alex_baked_cherry_pies_l0_1

theorem alex_baked_cherry_pies (total_pies : ℕ) (ratio_apple : ℕ) (ratio_blueberry : ℕ) (ratio_cherry : ℕ)
  (h1 : total_pies = 30)
  (h2 : ratio_apple = 1)
  (h3 : ratio_blueberry = 5)
  (h4 : ratio_cherry = 4) :
  (total_pies * ratio_cherry / (ratio_apple + ratio_blueberry + ratio_cherry) = 12) :=
by {
  sorry
}

end NUMINAMATH_GPT_alex_baked_cherry_pies_l0_1


namespace NUMINAMATH_GPT_polynomial_102_l0_63

/-- Proving the value of the polynomial expression using the Binomial Theorem -/
theorem polynomial_102 :
  102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 100406401 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_102_l0_63


namespace NUMINAMATH_GPT_primes_and_one_l0_62

-- Given conditions:
variables {a n : ℕ}
variable (ha : a > 100 ∧ a % 2 = 1)  -- a is an odd natural number greater than 100
variable (hn_bound : ∀ n ≤ Nat.sqrt (a / 5), Prime (a - n^2) / 4)  -- for all n ≤ √(a / 5), (a - n^2) / 4 is prime

-- Theorem: For all n > √(a / 5), (a - n^2) / 4 is either prime or 1
theorem primes_and_one {a : ℕ} (ha : a > 100 ∧ a % 2 = 1)
  (hn_bound : ∀ n ≤ Nat.sqrt (a / 5), Prime ((a - n^2) / 4)) :
  ∀ n > Nat.sqrt (a / 5), Prime ((a - n^2) / 4) ∨ ((a - n^2) / 4) = 1 :=
sorry

end NUMINAMATH_GPT_primes_and_one_l0_62


namespace NUMINAMATH_GPT_solve_eq_proof_l0_53

noncomputable def solve_equation : List ℚ := [-4, 1, 3 / 2, 2]

theorem solve_eq_proof :
  (∀ x : ℚ, 
    ((x^2 + 3 * x - 4)^2 + (2 * x^2 - 7 * x + 6)^2 = (3 * x^2 - 4 * x + 2)^2) ↔ 
    (x ∈ solve_equation)) :=
by
  sorry

end NUMINAMATH_GPT_solve_eq_proof_l0_53


namespace NUMINAMATH_GPT_find_a_and_c_l0_38

theorem find_a_and_c (a c : ℝ) (h : ∀ x : ℝ, -1/3 < x ∧ x < 1/2 → ax^2 + 2*x + c < 0) :
  a = 12 ∧ c = -2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a_and_c_l0_38


namespace NUMINAMATH_GPT_train_bus_ratio_is_two_thirds_l0_80

def total_distance : ℕ := 1800
def distance_by_plane : ℕ := total_distance / 3
def distance_by_bus : ℕ := 720
def distance_by_train : ℕ := total_distance - (distance_by_plane + distance_by_bus)
def train_to_bus_ratio : ℚ := distance_by_train / distance_by_bus

theorem train_bus_ratio_is_two_thirds :
  train_to_bus_ratio = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_train_bus_ratio_is_two_thirds_l0_80


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l0_89

theorem problem1 : (x : ℝ) → ((x + 1)^2 = 9 → (x = -4 ∨ x = 2)) :=
by
  intro x
  sorry

theorem problem2 : (x : ℝ) → (x^2 - 12*x - 4 = 0 → (x = 6 + 2*Real.sqrt 10 ∨ x = 6 - 2*Real.sqrt 10)) :=
by
  intro x
  sorry

theorem problem3 : (x : ℝ) → (3*(x - 2)^2 = x*(x - 2) → (x = 2 ∨ x = 3)) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l0_89


namespace NUMINAMATH_GPT_truck_covered_distance_l0_28

theorem truck_covered_distance (t : ℝ) (d_bike : ℝ) (d_truck : ℝ) (v_bike : ℝ) (v_truck : ℝ) :
  t = 8 ∧ d_bike = 136 ∧ v_truck = v_bike + 3 ∧ d_bike = v_bike * t →
  d_truck = v_truck * t :=
by
  sorry

end NUMINAMATH_GPT_truck_covered_distance_l0_28


namespace NUMINAMATH_GPT_vending_machine_users_l0_66

theorem vending_machine_users (p_fail p_double p_single : ℚ) (total_snacks : ℕ) (P : ℕ) :
  p_fail = 1 / 6 ∧ p_double = 1 / 10 ∧ p_single = 1 - 1 / 6 - 1 / 10 ∧
  total_snacks = 28 →
  P = 30 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_vending_machine_users_l0_66


namespace NUMINAMATH_GPT_find_triple_abc_l0_74

theorem find_triple_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
    (h_sum : a + b + c = 3)
    (h2 : a^2 - a ≥ 1 - b * c)
    (h3 : b^2 - b ≥ 1 - a * c)
    (h4 : c^2 - c ≥ 1 - a * b) :
    a = 1 ∧ b = 1 ∧ c = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_triple_abc_l0_74


namespace NUMINAMATH_GPT_find_arithmetic_sequence_elements_l0_2

theorem find_arithmetic_sequence_elements :
  ∃ (a b c : ℤ), -1 < a ∧ a < b ∧ b < c ∧ c < 7 ∧
  (∃ d : ℤ, a = -1 + d ∧ b = -1 + 2 * d ∧ c = -1 + 3 * d ∧ 7 = -1 + 4 * d) :=
sorry

end NUMINAMATH_GPT_find_arithmetic_sequence_elements_l0_2


namespace NUMINAMATH_GPT_warehouse_width_l0_84

theorem warehouse_width (L : ℕ) (circles : ℕ) (total_distance : ℕ)
  (hL : L = 600)
  (hcircles : circles = 8)
  (htotal_distance : total_distance = 16000) : 
  ∃ W : ℕ, 2 * L + 2 * W = (total_distance / circles) ∧ W = 400 :=
by
  sorry

end NUMINAMATH_GPT_warehouse_width_l0_84


namespace NUMINAMATH_GPT_sampling_interval_is_9_l0_24

-- Conditions
def books_per_hour : ℕ := 362
def sampled_books_per_hour : ℕ := 40

-- Claim to prove
theorem sampling_interval_is_9 : (360 / sampled_books_per_hour = 9) := by
  sorry

end NUMINAMATH_GPT_sampling_interval_is_9_l0_24


namespace NUMINAMATH_GPT_find_positive_integer_l0_79

variable (z : ℕ)

theorem find_positive_integer
  (h1 : (4 * z)^2 - z = 2345)
  (h2 : 0 < z) :
  z = 7 :=
sorry

end NUMINAMATH_GPT_find_positive_integer_l0_79


namespace NUMINAMATH_GPT_g_50_l0_33

noncomputable def g : ℝ → ℝ :=
sorry

axiom functional_equation (x y : ℝ) : g (x * y) = x * g y
axiom g_2 : g 2 = 10

theorem g_50 : g 50 = 250 :=
sorry

end NUMINAMATH_GPT_g_50_l0_33


namespace NUMINAMATH_GPT_number_of_groups_l0_54

theorem number_of_groups (max_value min_value interval : ℕ) (h_max : max_value = 36) (h_min : min_value = 15) (h_interval : interval = 4) : 
  ∃ groups : ℕ, groups = 6 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_groups_l0_54


namespace NUMINAMATH_GPT_arsenic_acid_concentration_equilibrium_l0_86

noncomputable def dissociation_constants 
  (Kd1 Kd2 Kd3 : ℝ) (H3AsO4 H2AsO4 HAsO4 AsO4 H : ℝ) : Prop :=
  Kd1 = (H * H2AsO4) / H3AsO4 ∧ Kd2 = (H * HAsO4) / H2AsO4 ∧ Kd3 = (H * AsO4) / HAsO4

theorem arsenic_acid_concentration_equilibrium :
  dissociation_constants 5.6e-3 1.7e-7 2.95e-12 0.1 (2e-2) (1.7e-7) (0) (2e-2) :=
by sorry

end NUMINAMATH_GPT_arsenic_acid_concentration_equilibrium_l0_86


namespace NUMINAMATH_GPT_coupon_calculation_l0_18

theorem coupon_calculation :
  let initial_stock : ℝ := 40.0
  let sold_books : ℝ := 20.0
  let coupons_per_book : ℝ := 4.0
  let remaining_books := initial_stock - sold_books
  let total_coupons := remaining_books * coupons_per_book
  total_coupons = 80.0 :=
by
  sorry

end NUMINAMATH_GPT_coupon_calculation_l0_18


namespace NUMINAMATH_GPT_not_all_mages_are_wizards_l0_7

variable (M S W : Type → Prop)

theorem not_all_mages_are_wizards
  (h1 : ∃ x, M x ∧ ¬ S x)
  (h2 : ∀ x, M x ∧ W x → S x) :
  ∃ x, M x ∧ ¬ W x :=
sorry

end NUMINAMATH_GPT_not_all_mages_are_wizards_l0_7


namespace NUMINAMATH_GPT_max_squares_overlap_l0_44

-- Definitions based on conditions.
def side_length_checkerboard_square : ℝ := 0.75
def side_length_card : ℝ := 2
def minimum_overlap : ℝ := 0.25

-- Main theorem to prove.
theorem max_squares_overlap :
  ∃ max_overlap_squares : ℕ, max_overlap_squares = 9 :=
by
  sorry

end NUMINAMATH_GPT_max_squares_overlap_l0_44


namespace NUMINAMATH_GPT__l0_87

/-- This theorem states that if the GCD of 8580 and 330 is diminished by 12, the result is 318. -/
example : (Int.gcd 8580 330) - 12 = 318 :=
by
  sorry

end NUMINAMATH_GPT__l0_87


namespace NUMINAMATH_GPT_intersection_eq_l0_55

def A : Set Int := { -1, 0, 1 }
def B : Set Int := { 0, 1, 2 }

theorem intersection_eq :
  A ∩ B = {0, 1} := 
by 
  sorry

end NUMINAMATH_GPT_intersection_eq_l0_55


namespace NUMINAMATH_GPT_meet_at_starting_point_l0_64

theorem meet_at_starting_point (track_length : Nat) (speed_A_kmph speed_B_kmph : Nat)
  (h_track_length : track_length = 1500)
  (h_speed_A : speed_A_kmph = 36)
  (h_speed_B : speed_B_kmph = 54) :
  let speed_A_mps := speed_A_kmph * 1000 / 3600
  let speed_B_mps := speed_B_kmph * 1000 / 3600
  let time_A := track_length / speed_A_mps
  let time_B := track_length / speed_B_mps
  let lcm_time := Nat.lcm time_A time_B
  lcm_time = 300 :=
by
  sorry

end NUMINAMATH_GPT_meet_at_starting_point_l0_64


namespace NUMINAMATH_GPT_supplementary_angle_60_eq_120_l0_19

def supplementary_angle (α : ℝ) : ℝ :=
  180 - α

theorem supplementary_angle_60_eq_120 :
  supplementary_angle 60 = 120 :=
by
  -- the proof should be filled here
  sorry

end NUMINAMATH_GPT_supplementary_angle_60_eq_120_l0_19


namespace NUMINAMATH_GPT_largest_x_eq_120_div_11_l0_0

theorem largest_x_eq_120_div_11 (x : ℝ) (h : (⌊x⌋ : ℝ) / x = 11 / 12) : x ≤ 120 / 11 :=
sorry

end NUMINAMATH_GPT_largest_x_eq_120_div_11_l0_0


namespace NUMINAMATH_GPT_car_fuel_efficiency_l0_58

theorem car_fuel_efficiency 
  (H C T : ℤ)
  (h₁ : 900 = H * T)
  (h₂ : 600 = C * T)
  (h₃ : C = H - 5) :
  C = 10 := by
  sorry

end NUMINAMATH_GPT_car_fuel_efficiency_l0_58


namespace NUMINAMATH_GPT_find_a_l0_52

def setA : Set ℤ := {-1, 0, 1}

def setB (a : ℤ) : Set ℤ := {a, a ^ 2}

theorem find_a (a : ℤ) (h : setA ∪ setB a = setA) : a = -1 :=
sorry

end NUMINAMATH_GPT_find_a_l0_52


namespace NUMINAMATH_GPT_fraction_simplification_l0_23

theorem fraction_simplification:
  (4 * 7) / (14 * 10) * (5 * 10 * 14) / (4 * 5 * 7) = 1 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_fraction_simplification_l0_23
