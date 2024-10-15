import Mathlib

namespace NUMINAMATH_GPT_f_1_geq_25_l176_17605

-- Define the function f
def f (x : ℝ) (m : ℝ) : ℝ := 4 * x^2 - m * x + 5

-- State that f is increasing on the interval [-2, +∞)
def is_increasing_on_interval (m : ℝ) : Prop :=
  ∀ x y : ℝ, -2 ≤ x → x ≤ y → f x m ≤ f y m

-- Prove that given the function is increasing on [-2, +∞),
-- then f(1) is at least 25.
theorem f_1_geq_25 (m : ℝ) (h : is_increasing_on_interval m) : f 1 m ≥ 25 :=
  sorry

end NUMINAMATH_GPT_f_1_geq_25_l176_17605


namespace NUMINAMATH_GPT_angle_bisector_inequality_l176_17601

theorem angle_bisector_inequality {a b c fa fb fc : ℝ} 
  (h_triangle_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angle_bisectors : fa > 0 ∧ fb > 0 ∧ fc > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  (1 / fa + 1 / fb + 1 / fc > 1 / a + 1 / b + 1 / c) :=
by
  sorry

end NUMINAMATH_GPT_angle_bisector_inequality_l176_17601


namespace NUMINAMATH_GPT_solve_X_l176_17680

def diamond (X Y : ℝ) : ℝ := 4 * X - 3 * Y + 2

theorem solve_X :
  (∃ X : ℝ, diamond X 6 = 35) ↔ (X = 51 / 4) := by
  sorry

end NUMINAMATH_GPT_solve_X_l176_17680


namespace NUMINAMATH_GPT_find_m_l176_17631

-- Define the conditions with variables a, b, and m.
variable (a b m : ℝ)
variable (ha : 2^a = m)
variable (hb : 5^b = m)
variable (hc : 1/a + 1/b = 2)

-- Define the statement to be proven.
theorem find_m : m = Real.sqrt 10 :=
by
  sorry


end NUMINAMATH_GPT_find_m_l176_17631


namespace NUMINAMATH_GPT_base_b_eq_five_l176_17640

theorem base_b_eq_five (b : ℕ) (h1 : 1225 = b^3 + 2 * b^2 + 2 * b + 5) (h2 : 35 = 3 * b + 5) :
    (3 * b + 5)^2 = b^3 + 2 * b^2 + 2 * b + 5 ↔ b = 5 :=
by
  sorry

end NUMINAMATH_GPT_base_b_eq_five_l176_17640


namespace NUMINAMATH_GPT_largest_integer_dividing_sum_of_5_consecutive_integers_l176_17665

theorem largest_integer_dividing_sum_of_5_consecutive_integers :
  ∀ (a : ℤ), ∃ (n : ℤ), n = 5 ∧ 5 ∣ ((a - 2) + (a - 1) + a + (a + 1) + (a + 2)) := by
  sorry

end NUMINAMATH_GPT_largest_integer_dividing_sum_of_5_consecutive_integers_l176_17665


namespace NUMINAMATH_GPT_minimum_reciprocal_sum_l176_17692

noncomputable def minimum_value_of_reciprocal_sum (x y z : ℝ) : ℝ :=
  if x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 2 then 
    max (1/x + 1/y + 1/z) (9/2)
  else
    0
  
theorem minimum_reciprocal_sum (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2): 
  1/x + 1/y + 1/z ≥ 9/2 :=
sorry

end NUMINAMATH_GPT_minimum_reciprocal_sum_l176_17692


namespace NUMINAMATH_GPT_line_passes_through_fixed_point_l176_17659

theorem line_passes_through_fixed_point 
  (a b : ℝ) 
  (h : 2 * a + b = 1) : 
  a * 4 + b * 2 = 2 :=
sorry

end NUMINAMATH_GPT_line_passes_through_fixed_point_l176_17659


namespace NUMINAMATH_GPT_shirt_cost_correct_l176_17694

-- Define the conditions
def pants_cost : ℝ := 9.24
def bill_amount : ℝ := 20
def change_received : ℝ := 2.51

-- Calculate total spent and shirt cost
def total_spent : ℝ := bill_amount - change_received
def shirt_cost : ℝ := total_spent - pants_cost

-- The theorem statement
theorem shirt_cost_correct : shirt_cost = 8.25 := by
  sorry

end NUMINAMATH_GPT_shirt_cost_correct_l176_17694


namespace NUMINAMATH_GPT_append_five_new_number_l176_17636

theorem append_five_new_number (t u : ℕ) (h1 : t < 10) (h2 : u < 10) : 
  10 * (10 * t + u) + 5 = 100 * t + 10 * u + 5 :=
by sorry

end NUMINAMATH_GPT_append_five_new_number_l176_17636


namespace NUMINAMATH_GPT_S_10_value_l176_17626

noncomputable def S (n : ℕ) (a : ℕ → ℕ) : ℕ := n * (a 1 + a n) / 2

theorem S_10_value (a : ℕ → ℕ) (h1 : a 2 = 3) (h2 : a 9 = 17) (h_arith : ∀ n, a (n + 1) = a n + (a 2 - a 1)) : 
  S 10 a = 100 := 
by
  sorry

end NUMINAMATH_GPT_S_10_value_l176_17626


namespace NUMINAMATH_GPT_evaluate_f_5_minus_f_neg_5_l176_17695

def f (x : ℝ) : ℝ := x^4 + x^2 + 5 * x^3

theorem evaluate_f_5_minus_f_neg_5 : f 5 - f (-5) = 1250 :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_f_5_minus_f_neg_5_l176_17695


namespace NUMINAMATH_GPT_remainder_x5_3x3_2x2_x_2_div_x_minus_2_l176_17654

def polynomial (x : ℝ) : ℝ := x^5 + 3*x^3 + 2*x^2 + x + 2

theorem remainder_x5_3x3_2x2_x_2_div_x_minus_2 :
  polynomial 2 = 68 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_x5_3x3_2x2_x_2_div_x_minus_2_l176_17654


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l176_17651

theorem quadratic_inequality_solution (y : ℝ) : 
  (y^2 - 9 * y + 14 ≤ 0) ↔ (2 ≤ y ∧ y ≤ 7) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l176_17651


namespace NUMINAMATH_GPT_milk_water_mixture_initial_volume_l176_17661

theorem milk_water_mixture_initial_volume
  (M W : ℝ)
  (h1 : 2 * M = 3 * W)
  (h2 : 4 * M = 3 * (W + 58)) :
  M + W = 145 := by
  sorry

end NUMINAMATH_GPT_milk_water_mixture_initial_volume_l176_17661


namespace NUMINAMATH_GPT_words_per_page_l176_17674

theorem words_per_page (p : ℕ) (h1 : 150 * p ≡ 270 [MOD 221]) (h2 : p ≤ 120) : p = 107 :=
sorry

end NUMINAMATH_GPT_words_per_page_l176_17674


namespace NUMINAMATH_GPT_sum_n_k_l176_17608

theorem sum_n_k (n k : ℕ) (h1 : 3 = n - 2 * k) (h2 : 15 = 5 * n - 8 * k) : n + k = 3 :=
by
  -- Use the conditions to conclude the proof.
  sorry

end NUMINAMATH_GPT_sum_n_k_l176_17608


namespace NUMINAMATH_GPT_find_number_l176_17666

theorem find_number (x : ℝ) (h_Pos : x > 0) (h_Eq : x + 17 = 60 * (1/x)) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l176_17666


namespace NUMINAMATH_GPT_set_equality_x_plus_y_l176_17615

theorem set_equality_x_plus_y (x y : ℝ) (A B : Set ℝ) (hA : A = {0, |x|, y}) (hB : B = {x, x * y, Real.sqrt (x - y)}) (h : A = B) : x + y = -2 :=
by
  sorry

end NUMINAMATH_GPT_set_equality_x_plus_y_l176_17615


namespace NUMINAMATH_GPT_problem_l176_17600

theorem problem (w : ℝ) (h : 13 = 13 * w / (1 - w)) : w^2 = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_problem_l176_17600


namespace NUMINAMATH_GPT_train_speed_l176_17645

theorem train_speed (len_train len_bridge time : ℝ) (h_len_train : len_train = 120)
  (h_len_bridge : len_bridge = 150) (h_time : time = 26.997840172786177) :
  let total_distance := len_train + len_bridge
  let speed_m_s := total_distance / time
  let speed_km_h := speed_m_s * 3.6
  speed_km_h = 36 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_train_speed_l176_17645


namespace NUMINAMATH_GPT_water_depth_upright_l176_17698

def tank_is_right_cylindrical := true
def tank_height := 18.0
def tank_diameter := 6.0
def tank_initial_position_is_flat := true
def water_depth_flat := 4.0

theorem water_depth_upright : water_depth_flat = 4.0 :=
by
  sorry

end NUMINAMATH_GPT_water_depth_upright_l176_17698


namespace NUMINAMATH_GPT_steve_speed_back_l176_17639

theorem steve_speed_back :
  ∀ (d v_total : ℕ), d = 10 → v_total = 6 →
  (2 * (15 / 6)) = 5 :=
by
  intros d v_total d_eq v_total_eq
  sorry

end NUMINAMATH_GPT_steve_speed_back_l176_17639


namespace NUMINAMATH_GPT_sin_C_value_proof_a2_b2_fraction_proof_sides_sum_comparison_l176_17658

variables (A B C a b c S : ℝ)
variables (h_area : S = (a + b) ^ 2 - c ^ 2) (h_sum : a + b = 4)
variables (h_triangle : ∀ (x : ℝ), x = sin C)

open Real

theorem sin_C_value_proof :
  sin C = 8 / 17 :=
sorry

theorem a2_b2_fraction_proof :
  (a ^ 2 - b ^ 2) / c ^ 2 = sin (A - B) / sin C :=
sorry

theorem sides_sum_comparison :
  a ^ 2 + b ^ 2 + c ^ 2 ≥ 4 * sqrt 3 * S :=
sorry

end NUMINAMATH_GPT_sin_C_value_proof_a2_b2_fraction_proof_sides_sum_comparison_l176_17658


namespace NUMINAMATH_GPT_minimum_height_l176_17630

theorem minimum_height (y : ℝ) (h : ℝ) (S : ℝ) (hS : S = 10 * y^2) (hS_min : S ≥ 150) (h_height : h = 2 * y) : h = 2 * Real.sqrt 15 :=
  sorry

end NUMINAMATH_GPT_minimum_height_l176_17630


namespace NUMINAMATH_GPT_find_original_one_digit_number_l176_17668

theorem find_original_one_digit_number (x : ℕ) (h1 : x < 10) (h2 : (x + 10) * (x + 10) / x = 72) : x = 2 :=
sorry

end NUMINAMATH_GPT_find_original_one_digit_number_l176_17668


namespace NUMINAMATH_GPT_quadratic_complete_square_r_plus_s_l176_17656

theorem quadratic_complete_square_r_plus_s :
  ∃ r s : ℚ, (∀ x : ℚ, 7 * x^2 - 21 * x - 56 = 0 → (x + r)^2 = s) ∧ r + s = 35 / 4 := sorry

end NUMINAMATH_GPT_quadratic_complete_square_r_plus_s_l176_17656


namespace NUMINAMATH_GPT_cube_side_length_ratio_l176_17623

-- Define the conditions and question
variable (s₁ s₂ : ℝ)
variable (weight₁ weight₂ : ℝ)
variable (V₁ V₂ : ℝ)
variable (same_metal : Prop)

-- Conditions
def condition1 (weight₁ : ℝ) : Prop := weight₁ = 4
def condition2 (weight₂ : ℝ) : Prop := weight₂ = 32
def condition3 (V₁ V₂ : ℝ) (s₁ s₂ : ℝ) : Prop := (V₁ = s₁^3) ∧ (V₂ = s₂^3)
def condition4 (same_metal : Prop) : Prop := same_metal

-- Volume definition based on weights and proportion
noncomputable def volume_definition (weight₁ weight₂ V₁ V₂ : ℝ) : Prop :=
(weight₂ / weight₁) = (V₂ / V₁)

-- Define the proof target
theorem cube_side_length_ratio
    (h1 : condition1 weight₁)
    (h2 : condition2 weight₂)
    (h3 : condition3 V₁ V₂ s₁ s₂)
    (h4 : condition4 same_metal)
    (h5 : volume_definition weight₁ weight₂ V₁ V₂) : 
    (s₂ / s₁) = 2 :=
by
  sorry

end NUMINAMATH_GPT_cube_side_length_ratio_l176_17623


namespace NUMINAMATH_GPT_inequality_solution_set_ab2_bc_ca_a3b_ge_1_4_l176_17696

theorem inequality_solution_set (x : ℝ) : (|x - 1| + |2 * x + 5| < 8) ↔ (-4 < x ∧ x < 4 / 3) :=
by
  sorry

theorem ab2_bc_ca_a3b_ge_1_4 (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 1) :
  (a^2 / (b + 3 * c) + b^2 / (c + 3 * a) + c^2 / (a + 3 * b) ≥ 1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_ab2_bc_ca_a3b_ge_1_4_l176_17696


namespace NUMINAMATH_GPT_triangle_square_ratio_l176_17653

theorem triangle_square_ratio (s_t s_s : ℝ) (h : 3 * s_t = 4 * s_s) : s_t / s_s = 4 / 3 := by
  sorry

end NUMINAMATH_GPT_triangle_square_ratio_l176_17653


namespace NUMINAMATH_GPT_solve_for_x_over_z_l176_17667

variables (x y z : ℝ)

theorem solve_for_x_over_z
  (h1 : x + y = 2 * x + z)
  (h2 : x - 2 * y = 4 * z)
  (h3 : x + y + z = 21)
  (h4 : y = 6 * z) :
  x / z = 5 :=
sorry

end NUMINAMATH_GPT_solve_for_x_over_z_l176_17667


namespace NUMINAMATH_GPT_factor_t_sq_minus_64_l176_17664

def isDifferenceOfSquares (a b : Int) : Prop := a = b^2

theorem factor_t_sq_minus_64 (t : Int) (h : isDifferenceOfSquares 64 8) : (t^2 - 64) = (t - 8) * (t + 8) := by
  sorry

end NUMINAMATH_GPT_factor_t_sq_minus_64_l176_17664


namespace NUMINAMATH_GPT_vector_parallel_m_l176_17614

theorem vector_parallel_m {m : ℝ} (h : (2:ℝ) * m - (-1 * -1) = 0) : m = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_vector_parallel_m_l176_17614


namespace NUMINAMATH_GPT_arithmetic_sequence_15th_term_l176_17609

theorem arithmetic_sequence_15th_term (a1 a2 a3 : ℕ) (d : ℕ) (n : ℕ) (h1 : a1 = 3) (h2 : a2 = 14) (h3 : a3 = 25) (h4 : d = a2 - a1) (h5 : a2 - a1 = a3 - a2) (h6 : n = 15) :
  a1 + (n - 1) * d = 157 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_15th_term_l176_17609


namespace NUMINAMATH_GPT_farmer_loss_representative_value_l176_17635

def check_within_loss_range (S L : ℝ) : Prop :=
  (S = 100000) → (20000 ≤ L ∧ L ≤ 25000)

theorem farmer_loss_representative_value : check_within_loss_range 100000 21987.53 :=
by
  intros hs
  sorry

end NUMINAMATH_GPT_farmer_loss_representative_value_l176_17635


namespace NUMINAMATH_GPT_am_gm_inequality_l176_17691

theorem am_gm_inequality
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
by
  sorry

end NUMINAMATH_GPT_am_gm_inequality_l176_17691


namespace NUMINAMATH_GPT_g_of_2_eq_14_l176_17689

theorem g_of_2_eq_14 (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3 * x - 4) = 4 * x + 6) : g 2 = 14 := 
sorry

end NUMINAMATH_GPT_g_of_2_eq_14_l176_17689


namespace NUMINAMATH_GPT_problem_statement_l176_17673

noncomputable def perimeter_rectangle 
  (a b c w : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (area_triangle : ℝ := (1/2) * a * b) 
  (area_rectangle : ℝ := area_triangle) 
  (l : ℝ := area_rectangle / w) : ℝ :=
2 * (w + l)

theorem problem_statement 
  (a b c w : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h_a : a = 9) 
  (h_b : b = 12) 
  (h_c : c = 15) 
  (h_w : w = 6) : 
  perimeter_rectangle a b c w h1 = 30 :=
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l176_17673


namespace NUMINAMATH_GPT_marbles_left_l176_17690

theorem marbles_left (red_marble_count blue_marble_count broken_marble_count : ℕ)
  (h1 : red_marble_count = 156)
  (h2 : blue_marble_count = 267)
  (h3 : broken_marble_count = 115) :
  red_marble_count + blue_marble_count - broken_marble_count = 308 :=
by
  sorry

end NUMINAMATH_GPT_marbles_left_l176_17690


namespace NUMINAMATH_GPT_acute_angle_alpha_range_l176_17684

theorem acute_angle_alpha_range (x : ℝ) (α : ℝ) (h1 : 0 < x) (h2 : x < 90) (h3 : α = 180 - 2 * x) : 0 < α ∧ α < 180 :=
by
  sorry

end NUMINAMATH_GPT_acute_angle_alpha_range_l176_17684


namespace NUMINAMATH_GPT_longest_side_similar_triangle_l176_17663

noncomputable def internal_angle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem longest_side_similar_triangle (a b c A : ℝ) (h₁ : a = 4) (h₂ : b = 6) (h₃ : c = 7) (h₄ : A = 132) :
  let k := Real.sqrt (132 / internal_angle 4 6 7)
  7 * k = 73.5 :=
by
  sorry

end NUMINAMATH_GPT_longest_side_similar_triangle_l176_17663


namespace NUMINAMATH_GPT_reciprocal_neg3_l176_17672

-- Define the problem
def reciprocal (x : ℚ) : ℚ := 1 / x

-- The required proof statement
theorem reciprocal_neg3 : reciprocal (-3) = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_neg3_l176_17672


namespace NUMINAMATH_GPT_increase_by_50_percent_l176_17679

def original : ℕ := 350
def increase_percent : ℕ := 50
def increased_number : ℕ := original * increase_percent / 100
def final_number : ℕ := original + increased_number

theorem increase_by_50_percent : final_number = 525 := 
by
  sorry

end NUMINAMATH_GPT_increase_by_50_percent_l176_17679


namespace NUMINAMATH_GPT_amelia_wins_l176_17637

noncomputable def amelia_wins_probability : ℚ := 21609 / 64328

theorem amelia_wins (h_am_heads : ℚ) (h_bl_heads : ℚ) (game_starts : Prop) (game_alternates : Prop) (win_condition : Prop) :
  h_am_heads = 3/7 ∧ h_bl_heads = 1/3 ∧ game_starts ∧ game_alternates ∧ win_condition →
  amelia_wins_probability = 21609 / 64328 :=
sorry

end NUMINAMATH_GPT_amelia_wins_l176_17637


namespace NUMINAMATH_GPT_polynomial_expansion_correct_l176_17632

open Polynomial

noncomputable def poly1 : Polynomial ℤ := X^2 + 3 * X - 4
noncomputable def poly2 : Polynomial ℤ := 2 * X^2 - X + 5
noncomputable def expected : Polynomial ℤ := 2 * X^4 + 5 * X^3 - 6 * X^2 + 19 * X - 20

theorem polynomial_expansion_correct :
  poly1 * poly2 = expected :=
sorry

end NUMINAMATH_GPT_polynomial_expansion_correct_l176_17632


namespace NUMINAMATH_GPT_complement_A_inter_B_range_of_a_l176_17641

open Set

-- Define sets A and B based on the conditions
def A : Set ℝ := {x | -4 ≤ x - 6 ∧ x - 6 ≤ 0}
def B : Set ℝ := {x | 2 * x - 6 ≥ 3 - x}

-- Define set C based on the conditions
def C (a : ℝ) : Set ℝ := {x | x ≤ a}

-- Problem 1: Prove the complement of (A ∩ B) in ℝ is the set of x where (x < 3 or x > 6)
theorem complement_A_inter_B :
  compl (A ∩ B) = {x | x < 3} ∪ {x | x > 6} :=
sorry

-- Problem 2: Prove that A ∩ C = A implies a ∈ [6, ∞)
theorem range_of_a {a : ℝ} (hC : A ∩ C a = A) :
  6 ≤ a :=
sorry

end NUMINAMATH_GPT_complement_A_inter_B_range_of_a_l176_17641


namespace NUMINAMATH_GPT_daughter_current_age_l176_17642

-- Define the conditions
def mother_current_age := 42
def years_later := 9
def mother_age_in_9_years := mother_current_age + years_later
def daughter_age_in_9_years (D : ℕ) := D + years_later

-- Define the statement we need to prove
theorem daughter_current_age : ∃ D : ℕ, mother_age_in_9_years = 3 * daughter_age_in_9_years D ∧ D = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_daughter_current_age_l176_17642


namespace NUMINAMATH_GPT_find_number_l176_17627

theorem find_number (X : ℝ) (h : 30 = 0.50 * X + 10) : X = 40 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l176_17627


namespace NUMINAMATH_GPT_total_inflation_time_l176_17699

/-- 
  Assume a soccer ball takes 20 minutes to inflate.
  Alexia inflates 20 soccer balls.
  Ermias inflates 5 more balls than Alexia.
  Prove that the total time in minutes taken to inflate all the balls is 900 minutes.
-/
theorem total_inflation_time 
  (alexia_balls : ℕ) (ermias_balls : ℕ) (each_ball_time : ℕ)
  (h1 : alexia_balls = 20)
  (h2 : ermias_balls = alexia_balls + 5)
  (h3 : each_ball_time = 20) :
  (alexia_balls + ermias_balls) * each_ball_time = 900 :=
by
  sorry

end NUMINAMATH_GPT_total_inflation_time_l176_17699


namespace NUMINAMATH_GPT_gain_percent_is_correct_l176_17629

noncomputable def gain_percent (CP SP : ℝ) : ℝ :=
  let gain := SP - CP
  (gain / CP) * 100

theorem gain_percent_is_correct :
  gain_percent 930 1210 = 30.11 :=
by
  sorry

end NUMINAMATH_GPT_gain_percent_is_correct_l176_17629


namespace NUMINAMATH_GPT_satellite_orbit_time_approx_l176_17662

noncomputable def earth_radius_km : ℝ := 6371
noncomputable def satellite_speed_kmph : ℝ := 7000

theorem satellite_orbit_time_approx :
  let circumference := 2 * Real.pi * earth_radius_km 
  let time := circumference / satellite_speed_kmph 
  5.6 < time ∧ time < 5.8 :=
by
  sorry

end NUMINAMATH_GPT_satellite_orbit_time_approx_l176_17662


namespace NUMINAMATH_GPT_ramesh_installation_cost_l176_17687

noncomputable def labelled_price (discounted_price : ℝ) (discount_rate : ℝ) : ℝ :=
  discounted_price / (1 - discount_rate)

noncomputable def selling_price (labelled_price : ℝ) (profit_rate : ℝ) : ℝ :=
  labelled_price * (1 + profit_rate)

def ramesh_total_cost (purchase_price transport_cost : ℝ) (installation_cost : ℝ) : ℝ :=
  purchase_price + transport_cost + installation_cost

theorem ramesh_installation_cost :
  ∀ (purchase_price discounted_price transport_cost labelled_price profit_rate selling_price installation_cost : ℝ),
  discounted_price = 12500 → transport_cost = 125 → profit_rate = 0.18 → selling_price = 18880 →
  labelled_price = discounted_price / (1 - 0.20) →
  selling_price = labelled_price * (1 + profit_rate) →
  ramesh_total_cost purchase_price transport_cost installation_cost = selling_price →
  installation_cost = 6255 :=
by
  intros
  sorry

end NUMINAMATH_GPT_ramesh_installation_cost_l176_17687


namespace NUMINAMATH_GPT_max_value_of_g_l176_17693

noncomputable def g (x : ℝ) : ℝ := min (min (3 * x + 3) ((1 / 3) * x + 1)) (-2 / 3 * x + 8)

theorem max_value_of_g : ∃ x : ℝ, g x = 10 / 3 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_g_l176_17693


namespace NUMINAMATH_GPT_RelativelyPrimeProbability_l176_17616

def relatively_prime_probability_42 : Rat :=
  let n := 42
  let total := n
  let rel_prime_count := total - (21 + 14 + 6 - 7 - 3 - 2 + 1)
  let probability := (rel_prime_count : Rat) / total
  probability

theorem RelativelyPrimeProbability : relatively_prime_probability_42 = 2 / 7 :=
sorry

end NUMINAMATH_GPT_RelativelyPrimeProbability_l176_17616


namespace NUMINAMATH_GPT_three_digit_number_count_l176_17624

theorem three_digit_number_count :
  ∃ n : ℕ, n = 15 ∧
  (∀ a b c : ℕ, (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9) →
    (100 * a + 10 * b + c = 37 * (a + b + c) → ∃ k : ℕ, k = n)) :=
sorry

end NUMINAMATH_GPT_three_digit_number_count_l176_17624


namespace NUMINAMATH_GPT_simplify_expression_l176_17610

theorem simplify_expression (x y : ℝ) :
  (3 * x^2 * y)^3 + (4 * x * y) * y^4 = 27 * x^6 * y^3 + 4 * x * y^5 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l176_17610


namespace NUMINAMATH_GPT_total_cost_correct_l176_17649

def cost_of_cat_toy := 10.22
def cost_of_cage := 11.73
def cost_of_cat_food := 7.50
def cost_of_leash := 5.15
def cost_of_cat_treats := 3.98

theorem total_cost_correct : 
  cost_of_cat_toy + cost_of_cage + cost_of_cat_food + cost_of_leash + cost_of_cat_treats = 38.58 := 
by
  sorry

end NUMINAMATH_GPT_total_cost_correct_l176_17649


namespace NUMINAMATH_GPT_geometric_sequence_sum_63_l176_17638

theorem geometric_sequence_sum_63
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_init : a 1 = 1)
  (h_recurrence : ∀ n, a (n + 2) + 2 * a (n + 1) = 8 * a n) :
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6) = 63 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_63_l176_17638


namespace NUMINAMATH_GPT_fraction_of_students_saying_dislike_actually_like_l176_17619

variables (total_students liking_disliking_students saying_disliking_like_students : ℚ)
          (fraction_like_dislike say_dislike : ℚ)
          (cond1 : 0.7 = liking_disliking_students / total_students) 
          (cond2 : 0.3 = (total_students - liking_disliking_students) / total_students)
          (cond3 : 0.3 * liking_disliking_students = saying_disliking_like_students)
          (cond4 : 0.8 * (total_students - liking_disliking_students) 
                    = say_dislike)

theorem fraction_of_students_saying_dislike_actually_like
    (total_students_eq: total_students = 100) : 
    fraction_like_dislike = 46.67 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_students_saying_dislike_actually_like_l176_17619


namespace NUMINAMATH_GPT_walk_to_lake_park_restaurant_is_zero_l176_17633

noncomputable def time_to_hidden_lake : ℕ := 15
noncomputable def time_to_return_from_hidden_lake : ℕ := 7
noncomputable def total_walk_time_dante : ℕ := 22

theorem walk_to_lake_park_restaurant_is_zero :
  ∃ (x : ℕ), (2 * x + time_to_hidden_lake + time_to_return_from_hidden_lake = total_walk_time_dante) → x = 0 :=
by
  use 0
  intros
  sorry

end NUMINAMATH_GPT_walk_to_lake_park_restaurant_is_zero_l176_17633


namespace NUMINAMATH_GPT_savings_together_vs_separate_l176_17628

def price_per_window : ℕ := 100

def free_windows_per_5_purchased : ℕ := 2

def daves_windows_needed : ℕ := 10

def dougs_windows_needed : ℕ := 11

def total_windows_needed : ℕ := daves_windows_needed + dougs_windows_needed

-- Cost calculation for Dave's windows with the offer
def daves_cost_with_offer : ℕ := 8 * price_per_window

-- Cost calculation for Doug's windows with the offer
def dougs_cost_with_offer : ℕ := 9 * price_per_window

-- Total cost calculation if purchased separately with the offer
def total_cost_separately_with_offer : ℕ := daves_cost_with_offer + dougs_cost_with_offer

-- Total cost calculation if purchased together with the offer
def total_cost_together_with_offer : ℕ := 17 * price_per_window

-- Calculate additional savings if Dave and Doug purchase together rather than separately
def additional_savings_together_vs_separate := 
  total_cost_separately_with_offer - total_cost_together_with_offer = 0

theorem savings_together_vs_separate : additional_savings_together_vs_separate := by
  sorry

end NUMINAMATH_GPT_savings_together_vs_separate_l176_17628


namespace NUMINAMATH_GPT_total_journey_distance_l176_17646

theorem total_journey_distance
  (T : ℝ) (D : ℝ)
  (h1 : T = 20)
  (h2 : (D / 2) / 21 + (D / 2) / 24 = 20) :
  D = 448 :=
by
  sorry

end NUMINAMATH_GPT_total_journey_distance_l176_17646


namespace NUMINAMATH_GPT_log_lt_x_l176_17622

theorem log_lt_x (x : ℝ) (hx : 0 < x) : Real.log (1 + x) < x := 
sorry

end NUMINAMATH_GPT_log_lt_x_l176_17622


namespace NUMINAMATH_GPT_tom_initial_books_l176_17606

theorem tom_initial_books (B : ℕ) (h1 : B - 4 + 38 = 39) : B = 5 :=
by
  sorry

end NUMINAMATH_GPT_tom_initial_books_l176_17606


namespace NUMINAMATH_GPT_prime_ge_7_divides_30_l176_17675

theorem prime_ge_7_divides_30 (p : ℕ) (hp : p ≥ 7) (hp_prime : Nat.Prime p) : 30 ∣ (p^2 - 1) := by
  sorry

end NUMINAMATH_GPT_prime_ge_7_divides_30_l176_17675


namespace NUMINAMATH_GPT_rate_of_grapes_l176_17697

theorem rate_of_grapes (G : ℝ) 
  (h_grapes : 8 * G + 9 * 60 = 1100) : 
  G = 70 := 
by
  sorry

end NUMINAMATH_GPT_rate_of_grapes_l176_17697


namespace NUMINAMATH_GPT_sum_of_possible_values_l176_17678

theorem sum_of_possible_values (x : ℝ) (h : |x - 5| + 2 = 4) :
  x = 7 ∨ x = 3 → x = 10 := 
by sorry

end NUMINAMATH_GPT_sum_of_possible_values_l176_17678


namespace NUMINAMATH_GPT_num_integer_values_satisfying_condition_l176_17676

theorem num_integer_values_satisfying_condition : 
  ∃ s : Finset ℤ, (∀ x ∈ s, 4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) ∧ s.card = 3 :=
by
  sorry

end NUMINAMATH_GPT_num_integer_values_satisfying_condition_l176_17676


namespace NUMINAMATH_GPT_inequality_log_range_of_a_l176_17603

open Real

theorem inequality_log (x : ℝ) (h₀ : 0 < x) : 
  1 - 1 / x ≤ log x ∧ log x ≤ x - 1 := sorry

theorem range_of_a (a : ℝ) (h : ∀ (x : ℝ), 0 < x ∧ x ≤ 1 → a * (1 - x^2) + x^2 * log x ≥ 0) : 
  a ≥ 1/2 := sorry

end NUMINAMATH_GPT_inequality_log_range_of_a_l176_17603


namespace NUMINAMATH_GPT_contrapositive_l176_17613

theorem contrapositive (a b : ℝ) :
  (a > b → a^2 > b^2) → (a^2 ≤ b^2 → a ≤ b) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_contrapositive_l176_17613


namespace NUMINAMATH_GPT_final_cost_is_30_l176_17657

-- Define conditions as constants
def cost_of_repair : ℝ := 7
def sales_tax : ℝ := 0.50
def number_of_tires : ℕ := 4

-- Define the cost for one tire repair
def cost_one_tire : ℝ := cost_of_repair + sales_tax

-- Define the cost for all tires
def total_cost : ℝ := cost_one_tire * number_of_tires

-- Theorem stating that the total cost is $30
theorem final_cost_is_30 : total_cost = 30 :=
by
  sorry

end NUMINAMATH_GPT_final_cost_is_30_l176_17657


namespace NUMINAMATH_GPT_vector_division_by_three_l176_17685

def OA : ℝ × ℝ := (2, 8)
def OB : ℝ × ℝ := (-7, 2)
def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
noncomputable def scalar_mult (k : ℝ) (u : ℝ × ℝ) : ℝ × ℝ := (k * u.1, k * u.2)

theorem vector_division_by_three :
  scalar_mult (1 / 3) (vector_sub OB OA) = (-3, -2) :=
sorry

end NUMINAMATH_GPT_vector_division_by_three_l176_17685


namespace NUMINAMATH_GPT_trigonometric_identity_l176_17611

theorem trigonometric_identity (α : ℝ) (h : Real.sin (α + (Real.pi / 3)) = 3 / 5) :
  Real.cos ((Real.pi / 6) - α) = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l176_17611


namespace NUMINAMATH_GPT_length_of_ac_l176_17612

theorem length_of_ac (a b c d e : ℝ) (ab bc cd de ae ac : ℝ)
  (h1 : ab = 5)
  (h2 : bc = 2 * cd)
  (h3 : de = 8)
  (h4 : ae = 22)
  (h5 : ae = ab + bc + cd + de)
  (h6 : ac = ab + bc) :
  ac = 11 := by
  sorry

end NUMINAMATH_GPT_length_of_ac_l176_17612


namespace NUMINAMATH_GPT_find_ratio_l176_17671

variables {EF GH EH EG EQ ER ES Q R S : ℝ}
variables (x : ℝ)
variables (E F G H : ℝ)

-- Conditions
def is_parallelogram : Prop := 
  -- Placeholder for parallelogram properties, not relevant for this example
  true

def point_on_segment (Q R : ℝ) (segment_length: ℝ) (ratio: ℝ): Prop := Q = segment_length * ratio ∧ R = segment_length * ratio

def intersect (EG QR : ℝ) (S : ℝ): Prop := 
  -- Placeholder for segment intersection properties, not relevant for this example
  true

-- Question
theorem find_ratio 
  (H_parallelogram: is_parallelogram)
  (H_pointQ: point_on_segment EQ ER EF (1/8))
  (H_pointR: point_on_segment ER ES EH (1/9))
  (H_intersection: intersect EG QR ES):
  (ES / EG) = (1/9) := 
by
  sorry

end NUMINAMATH_GPT_find_ratio_l176_17671


namespace NUMINAMATH_GPT_box_weights_l176_17688

theorem box_weights (a b c : ℕ) (h1 : a + b = 132) (h2 : b + c = 135) (h3 : c + a = 137) (h4 : a > 40) (h5 : b > 40) (h6 : c > 40) : a + b + c = 202 :=
by 
  sorry

end NUMINAMATH_GPT_box_weights_l176_17688


namespace NUMINAMATH_GPT_arithmetic_mean_six_expressions_l176_17648

theorem arithmetic_mean_six_expressions (x : ℝ)
  (h : (x + 8 + 15 + 2 * x + 13 + 2 * x + 4 + 3 * x + 5) / 6 = 30) : x = 13.5 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_six_expressions_l176_17648


namespace NUMINAMATH_GPT_money_made_is_40_l176_17683

-- Definitions based on conditions
def BettysStrawberries : ℕ := 16
def MatthewsStrawberries : ℕ := BettysStrawberries + 20
def NataliesStrawberries : ℕ := MatthewsStrawberries / 2
def TotalStrawberries : ℕ := BettysStrawberries + MatthewsStrawberries + NataliesStrawberries
def JarsOfJam : ℕ := TotalStrawberries / 7
def MoneyMade : ℕ := JarsOfJam * 4

-- The theorem to prove
theorem money_made_is_40 : MoneyMade = 40 :=
by
  sorry

end NUMINAMATH_GPT_money_made_is_40_l176_17683


namespace NUMINAMATH_GPT_square_area_from_diagonal_l176_17670

theorem square_area_from_diagonal
  (d : ℝ) (h : d = 10) : ∃ (A : ℝ), A = 50 :=
by {
  -- here goes the proof
  sorry
}

end NUMINAMATH_GPT_square_area_from_diagonal_l176_17670


namespace NUMINAMATH_GPT_cell_phone_bill_l176_17652

-- Definitions
def base_cost : ℝ := 20
def cost_per_text : ℝ := 0.05
def cost_per_extra_minute : ℝ := 0.10
def texts_sent : ℕ := 100
def hours_talked : ℝ := 30.5
def included_hours : ℝ := 30

-- Calculate extra minutes used
def extra_minutes : ℝ := (hours_talked - included_hours) * 60

-- Total cost calculation
def total_cost : ℝ := 
  base_cost + 
  (texts_sent * cost_per_text) + 
  (extra_minutes * cost_per_extra_minute)

-- Proof problem statement
theorem cell_phone_bill : total_cost = 28 := by
  sorry

end NUMINAMATH_GPT_cell_phone_bill_l176_17652


namespace NUMINAMATH_GPT_find_x_l176_17618

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x+1, -x)

def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem find_x (x : ℝ) (h : perpendicular vector_a (vector_b x)) : x = 1 :=
by sorry

end NUMINAMATH_GPT_find_x_l176_17618


namespace NUMINAMATH_GPT_probability_each_box_2_fruits_l176_17602

noncomputable def totalWaysToDistributePears : ℕ := (Nat.choose 8 4)
noncomputable def totalWaysToDistributeApples : ℕ := 5^6

noncomputable def case1 : ℕ := (Nat.choose 5 2) * (Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2))
noncomputable def case2 : ℕ := (Nat.choose 5 1) * (Nat.choose 4 2) * (Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1 * Nat.factorial 1))
noncomputable def case3 : ℕ := (Nat.choose 5 4) * (Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1))

noncomputable def totalFavorableDistributions : ℕ := case1 + case2 + case3
noncomputable def totalPossibleDistributions : ℕ := totalWaysToDistributePears * totalWaysToDistributeApples

noncomputable def probability : ℚ := (totalFavorableDistributions : ℚ) / totalPossibleDistributions * 100

theorem probability_each_box_2_fruits :
  probability = 0.74 := 
sorry

end NUMINAMATH_GPT_probability_each_box_2_fruits_l176_17602


namespace NUMINAMATH_GPT_solution_set_x_l176_17669

theorem solution_set_x (x : ℝ) : 
  (|x^2 - x - 2| + |1 / x| = |x^2 - x - 2 + 1 / x|) ↔ 
  (x ∈ {y : ℝ | -1 ≤ y ∧ y < 0} ∨ x ≥ 2) :=
sorry

end NUMINAMATH_GPT_solution_set_x_l176_17669


namespace NUMINAMATH_GPT_find_t_given_V_S_l176_17650

variables (g V V0 S S0 a t : ℝ)

theorem find_t_given_V_S :
  (V = g * (t - a) + V0) →
  (S = (1 / 2) * g * (t - a) ^ 2 + V0 * (t - a) + S0) →
  t = a + (V - V0) / g :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_find_t_given_V_S_l176_17650


namespace NUMINAMATH_GPT_contrapositive_true_l176_17686

theorem contrapositive_true (h : ∀ x : ℝ, x < 0 → x^2 > 0) : 
  (∀ x : ℝ, ¬ (x^2 > 0) → ¬ (x < 0)) :=
by 
  sorry

end NUMINAMATH_GPT_contrapositive_true_l176_17686


namespace NUMINAMATH_GPT_parametric_curve_C_line_tangent_to_curve_C_l176_17625

open Real

-- Definitions of the curve C and line l
def curve_C (ρ θ : ℝ) : Prop := ρ^2 - 4 * ρ * cos θ + 1 = 0

def line_l (t α x y : ℝ) : Prop := x = 4 + t * sin α ∧ y = t * cos α ∧ 0 ≤ α ∧ α < π

-- Parametric equation of curve C
theorem parametric_curve_C :
  ∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * π →
  ∃ x y : ℝ, (x = 2 + sqrt 3 * cos θ ∧ y = sqrt 3 * sin θ ∧
              curve_C (sqrt (x^2 + y^2)) θ) :=
sorry

-- Tangency condition for line l and curve C
theorem line_tangent_to_curve_C :
  ∀ α : ℝ, 0 ≤ α ∧ α < π →
  (∃ t : ℝ, ∃ x y : ℝ, (line_l t α x y ∧ (x - 2)^2 + y^2 = 3 ∧
                        ((abs (2 * cos α - 4 * cos α) / sqrt (cos α ^ 2 + sin α ^ 2)) = sqrt 3)) →
                       (α = π / 6 ∧ x = 7 / 2 ∧ y = - sqrt 3 / 2)) :=
sorry

end NUMINAMATH_GPT_parametric_curve_C_line_tangent_to_curve_C_l176_17625


namespace NUMINAMATH_GPT_find_number_added_l176_17644

theorem find_number_added (x n : ℕ) (h : (x + x + 2 + x + 4 + x + n + x + 22) / 5 = x + 7) : n = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_number_added_l176_17644


namespace NUMINAMATH_GPT_decreasing_function_iff_m_eq_2_l176_17634

theorem decreasing_function_iff_m_eq_2 
    (m : ℝ) : 
    (∀ x : ℝ, 0 < x → (m^2 - m - 1) * x^(-5*m - 3) < (m^2 - m - 1) * (x + 1)^(-5*m - 3)) ↔ m = 2 := 
sorry

end NUMINAMATH_GPT_decreasing_function_iff_m_eq_2_l176_17634


namespace NUMINAMATH_GPT_calc_x_l176_17677

theorem calc_x : 484 + 2 * 22 * 7 + 49 = 841 := by
  sorry

end NUMINAMATH_GPT_calc_x_l176_17677


namespace NUMINAMATH_GPT_seq_20_eq_5_over_7_l176_17607

theorem seq_20_eq_5_over_7 :
  ∃ (a : ℕ → ℚ), 
    a 1 = 6 / 7 ∧ 
    (∀ n, (0 ≤ a n ∧ a n < 1) → 
      (a (n + 1) = if a n < 1 / 2 then 2 * a n else 2 * a n - 1)) ∧ 
    a 20 = 5 / 7 := 
sorry

end NUMINAMATH_GPT_seq_20_eq_5_over_7_l176_17607


namespace NUMINAMATH_GPT_valid_pairs_l176_17620

theorem valid_pairs
  (x y : ℕ)
  (h_pos_x : x > 0)
  (h_pos_y : y > 0)
  (h_div : ∃ k : ℕ, k > 0 ∧ k * (2 * x + 7 * y) = 7 * x + 2 * y) :
  ∃ a : ℕ, a > 0 ∧ (x = a ∧ y = a ∨ x = 4 * a ∧ y = a ∨ x = 19 * a ∧ y = a) :=
by
  sorry

end NUMINAMATH_GPT_valid_pairs_l176_17620


namespace NUMINAMATH_GPT_find_boys_and_girls_l176_17647

noncomputable def number_of_boys_and_girls (a b c d : Nat) : (Nat × Nat) := sorry

theorem find_boys_and_girls : 
  ∃ m d : Nat,
  (∀ (a b c : Nat), 
    ((a = 15 ∨ b = 18 ∨ c = 13) ∧ 
    (a.mod 4 = 3 ∨ b.mod 4 = 2 ∨ c.mod 4 = 1)) 
    → number_of_boys_and_girls a b c d = (16, 14)) :=
sorry

end NUMINAMATH_GPT_find_boys_and_girls_l176_17647


namespace NUMINAMATH_GPT_increase_in_y_coordinate_l176_17643

theorem increase_in_y_coordinate (m n : ℝ) (h₁ : m = (n / 5) - 2 / 5) : 
  (5 * (m + 3) + 2) - (5 * m + 2) = 15 :=
by
  sorry

end NUMINAMATH_GPT_increase_in_y_coordinate_l176_17643


namespace NUMINAMATH_GPT_quadrilateral_is_parallelogram_l176_17655

theorem quadrilateral_is_parallelogram (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 2 * a * b + 2 * c * d) : a = b ∧ c = d :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_is_parallelogram_l176_17655


namespace NUMINAMATH_GPT_solve_equation_l176_17621

theorem solve_equation :
  {x : ℝ | x * (x - 3)^2 * (5 - x) = 0} = {0, 3, 5} :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l176_17621


namespace NUMINAMATH_GPT_man_swim_distance_downstream_l176_17681

noncomputable def DistanceDownstream (Vm : ℝ) (Vupstream : ℝ) (time : ℝ) : ℝ :=
  let Vs := Vm - Vupstream
  let Vdownstream := Vm + Vs
  Vdownstream * time

theorem man_swim_distance_downstream :
  let Vm : ℝ := 3  -- speed of man in still water in km/h
  let time : ℝ := 6 -- time taken in hours
  let d_upstream : ℝ := 12 -- distance swum upstream in km
  let Vupstream : ℝ := d_upstream / time
  DistanceDownstream Vm Vupstream time = 24 := sorry

end NUMINAMATH_GPT_man_swim_distance_downstream_l176_17681


namespace NUMINAMATH_GPT_a4_plus_a5_eq_27_l176_17604

-- Define the geometric sequence conditions
variables (a : ℕ → ℝ) (q : ℝ)
axiom a_pos : ∀ n, a n > 0
axiom a_2 : a 2 = 1 - a 1
axiom a_4 : a 4 = 9 - a 3

-- Define the geometric sequence property
axiom geom_seq : ∀ n, a (n + 1) = a n * q

theorem a4_plus_a5_eq_27 : a 4 + a 5 = 27 := sorry

end NUMINAMATH_GPT_a4_plus_a5_eq_27_l176_17604


namespace NUMINAMATH_GPT_solve_for_x_l176_17660

theorem solve_for_x : 
  ∀ (x : ℝ), (∀ (a b : ℝ), a * b = 4 * a - 2 * b) → (3 * (6 * x) = -2) → (x = 17 / 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l176_17660


namespace NUMINAMATH_GPT_total_marbles_l176_17682

theorem total_marbles (r b g y : ℝ)
  (h1 : r = 1.35 * b)
  (h2 : g = 1.5 * r)
  (h3 : y = 2 * b) :
  r + b + g + y = 4.72 * r :=
by
  sorry

end NUMINAMATH_GPT_total_marbles_l176_17682


namespace NUMINAMATH_GPT_focus_of_given_parabola_l176_17617

-- Define the given condition as a parameter
def parabola_eq (x y : ℝ) : Prop :=
  y = - (1/2) * x^2

-- Define the property for the focus of the parabola
def is_focus_of_parabola (focus : ℝ × ℝ) : Prop :=
  focus = (0, -1/2)

-- The theorem stating that the given parabola equation has the specific focus
theorem focus_of_given_parabola : 
  (∀ x y : ℝ, parabola_eq x y) → is_focus_of_parabola (0, -1/2) :=
by
  intro h
  unfold parabola_eq at h
  unfold is_focus_of_parabola
  sorry

end NUMINAMATH_GPT_focus_of_given_parabola_l176_17617
