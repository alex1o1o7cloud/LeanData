import Mathlib

namespace NUMINAMATH_GPT_parabola_hyperbola_tangent_l639_63983

noncomputable def parabola : ℝ → ℝ := λ x => x^2 + 5

noncomputable def hyperbola (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y => y^2 - m * x^2 = 1

theorem parabola_hyperbola_tangent (m : ℝ) : 
  (m = 10 + 4*Real.sqrt 6 ∨ m = 10 - 4*Real.sqrt 6) →
  ∃ x y, parabola x = y ∧ hyperbola m x y ∧ 
    ∃ c b a, a * y^2 + b * y + c = 0 ∧ a = 1 ∧ c = 5 * m - 1 ∧ b = -m ∧ b^2 - 4*a*c = 0 :=
by
  sorry

end NUMINAMATH_GPT_parabola_hyperbola_tangent_l639_63983


namespace NUMINAMATH_GPT_marge_final_plants_l639_63982

-- Definitions corresponding to the conditions
def seeds_planted := 23
def seeds_never_grew := 5
def plants_grew := seeds_planted - seeds_never_grew
def plants_eaten := plants_grew / 3
def uneaten_plants := plants_grew - plants_eaten
def plants_strangled := uneaten_plants / 3
def survived_plants := uneaten_plants - plants_strangled
def effective_addition := 1

-- The main statement we need to prove
theorem marge_final_plants : 
  (plants_grew - plants_eaten - plants_strangled + effective_addition) = 9 := 
by
  sorry

end NUMINAMATH_GPT_marge_final_plants_l639_63982


namespace NUMINAMATH_GPT_baseball_cards_given_l639_63911

theorem baseball_cards_given
  (initial_cards : ℕ)
  (maria_take : ℕ)
  (peter_cards : ℕ)
  (paul_triples : ℕ)
  (final_cards : ℕ)
  (h1 : initial_cards = 15)
  (h2 : maria_take = (initial_cards + 1) / 2)
  (h3 : final_cards = 3 * (initial_cards - maria_take - peter_cards))
  (h4 : final_cards = 18) :
  peter_cards = 1 := 
sorry

end NUMINAMATH_GPT_baseball_cards_given_l639_63911


namespace NUMINAMATH_GPT_yellow_faces_of_cube_l639_63910

theorem yellow_faces_of_cube (n : ℕ) (h : 6 * n^2 = (1 / 3) * (6 * n^3)) : n = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_yellow_faces_of_cube_l639_63910


namespace NUMINAMATH_GPT_point_on_y_axis_l639_63938

theorem point_on_y_axis (y : ℝ) :
  let A := (1, 0, 2)
  let B := (1, -3, 1)
  let M := (0, y, 0)
  dist A M = dist B M → y = -1 :=
by sorry

end NUMINAMATH_GPT_point_on_y_axis_l639_63938


namespace NUMINAMATH_GPT_total_books_in_series_l639_63988

-- Definitions for the conditions
def books_read : ℕ := 8
def books_to_read : ℕ := 6

-- Statement to be proved
theorem total_books_in_series : books_read + books_to_read = 14 := by
  sorry

end NUMINAMATH_GPT_total_books_in_series_l639_63988


namespace NUMINAMATH_GPT_average_of_last_three_numbers_l639_63926

theorem average_of_last_three_numbers (nums : List ℝ) (h_len : nums.length = 6) 
    (h_avg6 : nums.sum / 6 = 60) (h_avg3 : (nums.take 3).sum / 3 = 55) : 
    ((nums.drop 3).sum) / 3 = 65 := 
sorry

end NUMINAMATH_GPT_average_of_last_three_numbers_l639_63926


namespace NUMINAMATH_GPT_interest_rate_calculation_l639_63954

theorem interest_rate_calculation
  (P : ℕ) 
  (I : ℕ) 
  (T : ℕ) 
  (R : ℕ) 
  (principal : P = 9200) 
  (time : T = 3) 
  (interest_diff : P - 5888 = I) 
  (interest_formula : I = P * R * T / 100) 
  : R = 12 :=
sorry

end NUMINAMATH_GPT_interest_rate_calculation_l639_63954


namespace NUMINAMATH_GPT_mollys_present_age_l639_63934

theorem mollys_present_age (x : ℤ) (h : x + 18 = 5 * (x - 6)) : x = 12 := by
  sorry

end NUMINAMATH_GPT_mollys_present_age_l639_63934


namespace NUMINAMATH_GPT_Luka_water_requirement_l639_63991

-- Declare variables and conditions
variables (L S W O : ℕ)  -- All variables are natural numbers
-- Conditions
variable (h1 : S = 2 * L)  -- Twice as much sugar as lemon juice
variable (h2 : W = 5 * S)  -- 5 times as much water as sugar
variable (h3 : O = S)      -- Orange juice equals the amount of sugar 
variable (L_eq_5 : L = 5)  -- Lemon juice is 5 cups

-- The goal statement to prove
theorem Luka_water_requirement : W = 50 :=
by
  -- Note: The proof steps would go here, but as per instructions, we leave it as sorry.
  sorry

end NUMINAMATH_GPT_Luka_water_requirement_l639_63991


namespace NUMINAMATH_GPT_minimum_value_of_product_l639_63956

theorem minimum_value_of_product (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sum : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 9) :
  (a / b + b / c + c / a) * (b / a + c / b + a / c) = 30 := 
sorry

end NUMINAMATH_GPT_minimum_value_of_product_l639_63956


namespace NUMINAMATH_GPT_september_first_2021_was_wednesday_l639_63927

-- Defining the main theorem based on the conditions and the question
theorem september_first_2021_was_wednesday
  (doubledCapitalOnWeekdays : ∀ day : Nat, day = 0 % 7 → True)
  (sevenFiftyPercOnWeekends : ∀ day : Nat, day = 5 % 7 → True)
  (millionaireOnLastDayOfYear: ∀ day : Nat, day = 364 % 7 → True)
  : 1 % 7 = 3 % 7 := 
sorry

end NUMINAMATH_GPT_september_first_2021_was_wednesday_l639_63927


namespace NUMINAMATH_GPT_coprime_sum_product_l639_63965

theorem coprime_sum_product (a b : ℤ) (h : Int.gcd a b = 1) : Int.gcd (a + b) (a * b) = 1 := by
  sorry

end NUMINAMATH_GPT_coprime_sum_product_l639_63965


namespace NUMINAMATH_GPT_largest_integer_y_l639_63980

theorem largest_integer_y (y : ℤ) : (y / (4:ℚ) + 3 / 7 < 2 / 3) → y ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_y_l639_63980


namespace NUMINAMATH_GPT_maximize_x3y4_l639_63994

noncomputable def max_product (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 50) : ℝ :=
  x^3 * y^4

theorem maximize_x3y4 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 50) :
  max_product x y hx hy h ≤ max_product (150/7) (200/7) (by norm_num) (by norm_num) (by norm_num) :=
  sorry

end NUMINAMATH_GPT_maximize_x3y4_l639_63994


namespace NUMINAMATH_GPT_red_lights_l639_63901

theorem red_lights (total_lights yellow_lights blue_lights red_lights : ℕ)
  (h1 : total_lights = 95)
  (h2 : yellow_lights = 37)
  (h3 : blue_lights = 32)
  (h4 : red_lights = total_lights - (yellow_lights + blue_lights)) :
  red_lights = 26 := by
  sorry

end NUMINAMATH_GPT_red_lights_l639_63901


namespace NUMINAMATH_GPT_dot_product_a_a_sub_2b_l639_63995

-- Define the vectors a and b
def a : (ℝ × ℝ) := (2, 3)
def b : (ℝ × ℝ) := (-1, 2)

-- Define the subtraction of vector a and 2 * vector b
def a_sub_2b : (ℝ × ℝ) := (a.1 - 2 * b.1, a.2 - 2 * b.2)

-- Define the dot product of two vectors
def dot_product (u v : (ℝ × ℝ)) : ℝ := u.1 * v.1 + u.2 * v.2

-- State that the dot product of a and (a - 2b) is 5
theorem dot_product_a_a_sub_2b : dot_product a a_sub_2b = 5 := 
by 
  -- proof omitted
  sorry

end NUMINAMATH_GPT_dot_product_a_a_sub_2b_l639_63995


namespace NUMINAMATH_GPT_action_figure_collection_complete_l639_63942

theorem action_figure_collection_complete (act_figures : ℕ) (cost_per_fig : ℕ) (extra_money_needed : ℕ) (total_collection : ℕ) 
    (h1 : act_figures = 7) 
    (h2 : cost_per_fig = 8) 
    (h3 : extra_money_needed = 72) : 
    total_collection = 16 :=
by
  sorry

end NUMINAMATH_GPT_action_figure_collection_complete_l639_63942


namespace NUMINAMATH_GPT_students_not_pass_l639_63904

theorem students_not_pass (total_students : ℕ) (percentage_passed : ℕ) (students_passed : ℕ) (students_not_passed : ℕ) :
  total_students = 804 →
  percentage_passed = 75 →
  students_passed = total_students * percentage_passed / 100 →
  students_not_passed = total_students - students_passed →
  students_not_passed = 201 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_students_not_pass_l639_63904


namespace NUMINAMATH_GPT_binary_sum_to_decimal_l639_63915

theorem binary_sum_to_decimal :
  let bin1 := "1101011"
  let bin2 := "1010110"
  let dec1 := 64 + 32 + 0 + 8 + 0 + 2 + 1 -- decimal value of "1101011"
  let dec2 := 64 + 0 + 16 + 0 + 4 + 2 + 0 -- decimal value of "1010110"
  dec1 + dec2 = 193 := by
  sorry

end NUMINAMATH_GPT_binary_sum_to_decimal_l639_63915


namespace NUMINAMATH_GPT_gcd_765432_654321_l639_63962

theorem gcd_765432_654321 :
  Int.gcd 765432 654321 = 3 := 
sorry

end NUMINAMATH_GPT_gcd_765432_654321_l639_63962


namespace NUMINAMATH_GPT_mike_gave_12_pears_l639_63961

variable (P M K N : ℕ)

def initial_pears := 46
def pears_given_to_keith := 47
def pears_left := 11

theorem mike_gave_12_pears (M : ℕ) : 
  initial_pears - pears_given_to_keith + M = pears_left → M = 12 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_mike_gave_12_pears_l639_63961


namespace NUMINAMATH_GPT_range_of_m_l639_63955

variable {x y m : ℝ}

theorem range_of_m (hx : 0 < x) (hy : 0 < y) (h_eq : 1/x + 4/y = 1) (h_ineq : ∃ x y, x + y/4 < m^2 - 3*m) : m < -1 ∨ m > 4 :=
sorry

end NUMINAMATH_GPT_range_of_m_l639_63955


namespace NUMINAMATH_GPT_find_stream_speed_l639_63990

variable (r w : ℝ)

noncomputable def stream_speed:
    Prop := 
    (21 / (r + w) + 4 = 21 / (r - w)) ∧ 
    (21 / (3 * r + w) + 0.5 = 21 / (3 * r - w)) ∧ 
    w = 3 

theorem find_stream_speed : ∃ w, stream_speed r w := 
by
  sorry

end NUMINAMATH_GPT_find_stream_speed_l639_63990


namespace NUMINAMATH_GPT_circle_radius_five_l639_63946

theorem circle_radius_five (c : ℝ) : (∃ x y : ℝ, x^2 + 10 * x + y^2 + 8 * y + c = 0) ∧ 
                                     ((x + 5)^2 + (y + 4)^2 = 25) → c = 16 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_five_l639_63946


namespace NUMINAMATH_GPT_geometric_sequence_sum_l639_63959

theorem geometric_sequence_sum (k : ℕ) (h1 : a_1 = 1) (h2 : a_k = 243) (h3 : q = 3) : S_k = 364 := 
by 
  -- Sorry is used here to skip the proof
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l639_63959


namespace NUMINAMATH_GPT_quiz_show_prob_l639_63929

-- Definitions extracted from the problem conditions
def n : ℕ := 4 -- Number of questions
def p_correct : ℚ := 1 / 4 -- Probability of guessing a question correctly
def p_incorrect : ℚ := 3 / 4 -- Probability of guessing a question incorrectly

-- We need to prove that the probability of answering at least 3 out of 4 questions correctly 
-- by guessing randomly is 13/256.
theorem quiz_show_prob :
  (Nat.choose n 3 * (p_correct ^ 3) * (p_incorrect ^ 1) +
   Nat.choose n 4 * (p_correct ^ 4)) = 13 / 256 :=
by sorry

end NUMINAMATH_GPT_quiz_show_prob_l639_63929


namespace NUMINAMATH_GPT_circle_condition_l639_63925

theorem circle_condition (m : ℝ) : (∃ x y : ℝ, x^2 + y^2 + 4*x - 2*y + 5*m = 0) →
  (m < 1) :=
by
  sorry

end NUMINAMATH_GPT_circle_condition_l639_63925


namespace NUMINAMATH_GPT_price_decrease_percentage_l639_63918

variables (P Q : ℝ)
variables (Q' R R' : ℝ)

-- Condition: the number sold increased by 60%
def quantity_increase_condition : Prop :=
  Q' = Q * (1 + 0.60)

-- Condition: the total revenue increased by 28.000000000000025%
def revenue_increase_condition : Prop :=
  R' = R * (1 + 0.28000000000000025)

-- Definition: the original revenue R
def original_revenue : Prop :=
  R = P * Q

-- The new price P' after decreasing by x%
variables (P' : ℝ) (x : ℝ)
def new_price_condition : Prop :=
  P' = P * (1 - x / 100)

-- The new revenue R'
def new_revenue : Prop :=
  R' = P' * Q'

-- The proof problem
theorem price_decrease_percentage (P Q Q' R R' P' x : ℝ)
  (h1 : quantity_increase_condition Q Q')
  (h2 : revenue_increase_condition R R')
  (h3 : original_revenue P Q R)
  (h4 : new_price_condition P P' x)
  (h5 : new_revenue P' Q' R') :
  x = 20 :=
sorry

end NUMINAMATH_GPT_price_decrease_percentage_l639_63918


namespace NUMINAMATH_GPT_product_of_terms_geometric_sequence_l639_63909

variable {a : ℕ → ℝ}
variable {q : ℝ}
noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem product_of_terms_geometric_sequence
  (ha: geometric_sequence a q)
  (h3_4: a 3 * a 4 = 6) :
  a 2 * a 5 = 6 :=
by
  sorry

end NUMINAMATH_GPT_product_of_terms_geometric_sequence_l639_63909


namespace NUMINAMATH_GPT_grass_knot_segments_butterfly_knot_segments_l639_63972

-- Definitions for the grass knot problem
def outer_loops_cut : Nat := 5
def segments_after_outer_loops_cut : Nat := 6

-- Theorem for the grass knot
theorem grass_knot_segments (n : Nat) (h : n = outer_loops_cut) : (n + 1 = segments_after_outer_loops_cut) :=
sorry

-- Definitions for the butterfly knot problem
def butterfly_wings_loops_per_wing : Nat := 7
def segments_after_butterfly_wings_cut : Nat := 15

-- Theorem for the butterfly knot
theorem butterfly_knot_segments (w : Nat) (h : w = butterfly_wings_loops_per_wing) : ((w * 2 * 2 + 2) / 2 = segments_after_butterfly_wings_cut) :=
sorry

end NUMINAMATH_GPT_grass_knot_segments_butterfly_knot_segments_l639_63972


namespace NUMINAMATH_GPT_peter_fish_caught_l639_63935

theorem peter_fish_caught (n : ℕ) (h : 3 * n = n + 24) : n = 12 :=
sorry

end NUMINAMATH_GPT_peter_fish_caught_l639_63935


namespace NUMINAMATH_GPT_find_q_l639_63985

theorem find_q {q : ℕ} (h : 27^8 = 9^q) : q = 12 := by
  sorry

end NUMINAMATH_GPT_find_q_l639_63985


namespace NUMINAMATH_GPT_inequality_solution_l639_63970

theorem inequality_solution (x y z : ℝ) (h1 : x + 3 * y + 2 * z = 6) :
  (z = 3 - 1/2 * x - 3/2 * y) ∧ (x - 2)^2 + (3 * y - 2)^2 ≤ 4 ∧ 0 ≤ x ∧ x ≤ 4 :=
sorry

end NUMINAMATH_GPT_inequality_solution_l639_63970


namespace NUMINAMATH_GPT_slope_of_line_l639_63984

theorem slope_of_line (a b c : ℝ) (h : 3 * a = 4 * b - 9) : a = 4 / 3 * b - 3 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_l639_63984


namespace NUMINAMATH_GPT_percentage_deficit_of_second_side_l639_63986

theorem percentage_deficit_of_second_side
  (L W : Real)
  (h1 : ∃ (L' : Real), L' = 1.16 * L)
  (h2 : ∃ (W' : Real), (L' * W') = 1.102 * (L * W))
  (h3 : ∃ (x : Real), W' = W * (1 - x / 100)) :
  x = 5 := 
  sorry

end NUMINAMATH_GPT_percentage_deficit_of_second_side_l639_63986


namespace NUMINAMATH_GPT_tetrahedron_inequality_l639_63903

theorem tetrahedron_inequality
  (a b c d h_a h_b h_c h_d V : ℝ)
  (ha : V = 1/3 * a * h_a)
  (hb : V = 1/3 * b * h_b)
  (hc : V = 1/3 * c * h_c)
  (hd : V = 1/3 * d * h_d) :
  (a + b + c + d) * (h_a + h_b + h_c + h_d) >= 48 * V := 
  by sorry

end NUMINAMATH_GPT_tetrahedron_inequality_l639_63903


namespace NUMINAMATH_GPT_period_sine_transformed_l639_63919

theorem period_sine_transformed (x : ℝ) : 
  let y := 3 * Real.sin ((x / 3) + (Real.pi / 4))
  ∃ p : ℝ, (∀ x : ℝ, y = 3 * Real.sin ((x + p) / 3 + (Real.pi / 4)) ↔ y = 3 * Real.sin ((x / 3) + (Real.pi / 4))) ∧ p = 6 * Real.pi :=
sorry

end NUMINAMATH_GPT_period_sine_transformed_l639_63919


namespace NUMINAMATH_GPT_cans_of_chili_beans_ordered_l639_63996

theorem cans_of_chili_beans_ordered (T C : ℕ) (h1 : 2 * T = C) (h2 : T + C = 12) : C = 8 := by
  sorry

end NUMINAMATH_GPT_cans_of_chili_beans_ordered_l639_63996


namespace NUMINAMATH_GPT_proof_sum_q_p_x_l639_63900

def p (x : ℝ) : ℝ := |x| - 3
def q (x : ℝ) : ℝ := -|x|

-- define the list of x values
def x_values : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

-- define q_p_x to apply q to p of each x
def q_p_x : List ℝ := x_values.map (λ x => q (p x))

-- define the sum of q(p(x)) for given x values
def sum_q_p_x : ℝ := q_p_x.sum

theorem proof_sum_q_p_x : sum_q_p_x = -15 := by
  -- steps of solution
  sorry

end NUMINAMATH_GPT_proof_sum_q_p_x_l639_63900


namespace NUMINAMATH_GPT_flag_blue_area_l639_63930

theorem flag_blue_area (A C₁ C₃ : ℝ) (h₀ : A = 1.0) (h₁ : C₁ + C₃ = 0.36 * A) :
  C₃ = 0.02 * A := by
  sorry

end NUMINAMATH_GPT_flag_blue_area_l639_63930


namespace NUMINAMATH_GPT_product_of_05_and_2_3_is_1_3_l639_63913

theorem product_of_05_and_2_3_is_1_3 : (0.5 * (2 / 3) = 1 / 3) :=
by sorry

end NUMINAMATH_GPT_product_of_05_and_2_3_is_1_3_l639_63913


namespace NUMINAMATH_GPT_total_share_proof_l639_63936

variable (P R : ℕ) -- Parker's share and Richie's share
variable (total_share : ℕ) -- Total share

-- Define conditions
def ratio_condition : Prop := (P : ℕ) / 2 = (R : ℕ) / 3
def parker_share_condition : Prop := P = 50

-- Prove the total share is 125
theorem total_share_proof 
  (h1 : ratio_condition P R)
  (h2 : parker_share_condition P) : 
  total_share = 125 :=
sorry

end NUMINAMATH_GPT_total_share_proof_l639_63936


namespace NUMINAMATH_GPT_cost_of_parts_per_tire_repair_is_5_l639_63960

-- Define the given conditions
def charge_per_tire_repair : ℤ := 20
def num_tire_repairs : ℤ := 300
def charge_per_complex_repair : ℤ := 300
def num_complex_repairs : ℤ := 2
def cost_per_complex_repair_parts : ℤ := 50
def retail_shop_profit : ℤ := 2000
def fixed_expenses : ℤ := 4000
def total_profit : ℤ := 3000

-- Define the calculation for total revenue
def total_revenue : ℤ := 
    (charge_per_tire_repair * num_tire_repairs) + 
    (charge_per_complex_repair * num_complex_repairs) + 
    retail_shop_profit

-- Define the calculation for total expenses
def total_expenses : ℤ := total_revenue - total_profit

-- Define the calculation for parts cost of tire repairs
def parts_cost_tire_repairs : ℤ := 
    total_expenses - (cost_per_complex_repair_parts * num_complex_repairs) - fixed_expenses

def cost_per_tire_repair : ℤ := parts_cost_tire_repairs / num_tire_repairs

-- The statement to be proved
theorem cost_of_parts_per_tire_repair_is_5 : cost_per_tire_repair = 5 := by
    sorry

end NUMINAMATH_GPT_cost_of_parts_per_tire_repair_is_5_l639_63960


namespace NUMINAMATH_GPT_balloons_per_school_l639_63978

theorem balloons_per_school (yellow black total : ℕ) 
  (hyellow : yellow = 3414)
  (hblack : black = yellow + 1762)
  (htotal : total = yellow + black)
  (hdivide : total % 10 = 0) : 
  total / 10 = 859 :=
by sorry

end NUMINAMATH_GPT_balloons_per_school_l639_63978


namespace NUMINAMATH_GPT_ab_zero_l639_63997

theorem ab_zero (a b : ℝ) (x : ℝ) (h : ∀ x : ℝ, a * x + b * x ^ 2 = -(a * (-x) + b * (-x) ^ 2)) : a * b = 0 :=
by
  sorry

end NUMINAMATH_GPT_ab_zero_l639_63997


namespace NUMINAMATH_GPT_exists_two_digit_number_N_l639_63933

-- Statement of the problem
theorem exists_two_digit_number_N : 
  ∃ (N : ℕ), (∃ (a b : ℕ), N = 10 * a + b ∧ N = a * b + 2 * (a + b) ∧ 10 ≤ N ∧ N < 100) :=
by
  sorry

end NUMINAMATH_GPT_exists_two_digit_number_N_l639_63933


namespace NUMINAMATH_GPT_prism_sides_plus_two_l639_63993

theorem prism_sides_plus_two (E V S : ℕ) (h1 : E + V = 30) (h2 : E = 3 * S) (h3 : V = 2 * S) : S + 2 = 8 :=
by
  sorry

end NUMINAMATH_GPT_prism_sides_plus_two_l639_63993


namespace NUMINAMATH_GPT_levi_additional_baskets_to_score_l639_63989

def levi_scored_initial := 8
def brother_scored_initial := 12
def brother_likely_to_score := 3
def levi_goal_margin := 5

theorem levi_additional_baskets_to_score : 
  levi_scored_initial + 12 >= brother_scored_initial + brother_likely_to_score + levi_goal_margin :=
by
  sorry

end NUMINAMATH_GPT_levi_additional_baskets_to_score_l639_63989


namespace NUMINAMATH_GPT_tan_ratio_given_sin_equation_l639_63922

theorem tan_ratio_given_sin_equation (α β : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin (2*α + β) = (3/2) * Real.sin β) : 
  Real.tan (α + β) / Real.tan α = 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_tan_ratio_given_sin_equation_l639_63922


namespace NUMINAMATH_GPT_find_x_l639_63916

-- Define the conditions as given in the problem
def angle1 (x : ℝ) : ℝ := 6 * x
def angle2 (x : ℝ) : ℝ := 3 * x
def angle3 (x : ℝ) : ℝ := x
def angle4 (x : ℝ) : ℝ := 5 * x
def sum_of_angles (x : ℝ) : ℝ := angle1 x + angle2 x + angle3 x + angle4 x

-- State the problem: prove that x equals 24 given the sum of angles is 360 degrees
theorem find_x (x : ℝ) (h : sum_of_angles x = 360) : x = 24 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l639_63916


namespace NUMINAMATH_GPT_sum_of_two_longest_altitudes_l639_63973

theorem sum_of_two_longest_altitudes (a b c : ℕ) (h : a^2 + b^2 = c^2) (h1: a = 7) (h2: b = 24) (h3: c = 25) : 
  (a + b = 31) :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_two_longest_altitudes_l639_63973


namespace NUMINAMATH_GPT_find_linear_function_l639_63999

noncomputable def functional_equation (f : ℝ → ℝ) : Prop :=
(∀ (a b c : ℝ), a + b + c ≥ 0 → f (a^3) + f (b^3) + f (c^3) ≥ 3 * f (a * b * c))
∧ (∀ (a b c : ℝ), a + b + c ≤ 0 → f (a^3) + f (b^3) + f (c^3) ≤ 3 * f (a * b * c))

theorem find_linear_function (f : ℝ → ℝ) (h : functional_equation f) : ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end NUMINAMATH_GPT_find_linear_function_l639_63999


namespace NUMINAMATH_GPT_find_length_AB_l639_63975

-- Definitions for the problem conditions.
def angle_B : ℝ := 90
def angle_A : ℝ := 30
def BC : ℝ := 24

-- Main theorem to prove.
theorem find_length_AB (angle_B_eq : angle_B = 90) (angle_A_eq : angle_A = 30) (BC_eq : BC = 24) : 
  ∃ AB : ℝ, AB = 12 := 
by
  sorry

end NUMINAMATH_GPT_find_length_AB_l639_63975


namespace NUMINAMATH_GPT_range_of_a_l639_63920

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 1| + |x - a| ≥ 5) ↔ (a ≤ -6 ∨ a ≥ 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l639_63920


namespace NUMINAMATH_GPT_total_copper_mined_l639_63992

theorem total_copper_mined :
  let daily_production_A := 4500
  let daily_production_B := 6000
  let daily_production_C := 5000
  let daily_production_D := 3500
  let copper_percentage_A := 0.055
  let copper_percentage_B := 0.071
  let copper_percentage_C := 0.147
  let copper_percentage_D := 0.092
  (daily_production_A * copper_percentage_A +
   daily_production_B * copper_percentage_B +
   daily_production_C * copper_percentage_C +
   daily_production_D * copper_percentage_D) = 1730.5 :=
by
  sorry

end NUMINAMATH_GPT_total_copper_mined_l639_63992


namespace NUMINAMATH_GPT_simplify_expression_l639_63974

theorem simplify_expression (w : ℤ) : 
  (-2 * w + 3 - 4 * w + 7 + 6 * w - 5 - 8 * w + 8) = (-8 * w + 13) :=
by {
  sorry
}

end NUMINAMATH_GPT_simplify_expression_l639_63974


namespace NUMINAMATH_GPT_karen_total_nuts_l639_63949

variable (x y : ℝ)
variable (hx : x = 0.25)
variable (hy : y = 0.25)

theorem karen_total_nuts : x + y = 0.50 := by
  rw [hx, hy]
  norm_num

end NUMINAMATH_GPT_karen_total_nuts_l639_63949


namespace NUMINAMATH_GPT_function_increasing_iff_l639_63950

noncomputable def f (x a : ℝ) : ℝ := (1 / 3) * x^3 - a * x

theorem function_increasing_iff (a : ℝ) :
  (∀ x : ℝ, 0 < x^2 - a) ↔ a ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_function_increasing_iff_l639_63950


namespace NUMINAMATH_GPT_puppy_weight_is_3_8_l639_63987

noncomputable def puppy_weight_problem (p s l : ℝ) : Prop :=
  p + 2 * s + l = 38 ∧
  p + l = 3 * s ∧
  p + 2 * s = l

theorem puppy_weight_is_3_8 :
  ∃ p s l : ℝ, puppy_weight_problem p s l ∧ p = 3.8 :=
by
  sorry

end NUMINAMATH_GPT_puppy_weight_is_3_8_l639_63987


namespace NUMINAMATH_GPT_remainder_is_one_l639_63968

theorem remainder_is_one (N : ℤ) (R : ℤ)
  (h1 : N % 100 = R)
  (h2 : N % R = 1) :
  R = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_is_one_l639_63968


namespace NUMINAMATH_GPT_find_p_of_abs_sum_roots_eq_five_l639_63948

theorem find_p_of_abs_sum_roots_eq_five (p : ℝ) : 
  (∃ x y : ℝ, x + y = -p ∧ x * y = -6 ∧ |x| + |y| = 5) → (p = 1 ∨ p = -1) := by
  sorry

end NUMINAMATH_GPT_find_p_of_abs_sum_roots_eq_five_l639_63948


namespace NUMINAMATH_GPT_repeating_decimal_sum_to_fraction_l639_63912

def repeating_decimal_123 : ℚ := 123 / 999
def repeating_decimal_0045 : ℚ := 45 / 9999
def repeating_decimal_000678 : ℚ := 678 / 999999

theorem repeating_decimal_sum_to_fraction :
  repeating_decimal_123 + repeating_decimal_0045 + repeating_decimal_000678 = 128178 / 998001000 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_sum_to_fraction_l639_63912


namespace NUMINAMATH_GPT_gain_percent_calculation_l639_63952

variable (CP SP : ℝ)
variable (gain gain_percent : ℝ)

theorem gain_percent_calculation
  (h₁ : CP = 900) 
  (h₂ : SP = 1180)
  (h₃ : gain = SP - CP)
  (h₄ : gain_percent = (gain / CP) * 100) :
  gain_percent = 31.11 := by
sorry

end NUMINAMATH_GPT_gain_percent_calculation_l639_63952


namespace NUMINAMATH_GPT_unique_integral_root_of_equation_l639_63964

theorem unique_integral_root_of_equation :
  ∀ x : ℤ, (x - 9 / (x - 5) = 7 - 9 / (x - 5)) ↔ (x = 7) :=
by
  sorry

end NUMINAMATH_GPT_unique_integral_root_of_equation_l639_63964


namespace NUMINAMATH_GPT_find_smallest_positive_angle_l639_63917

noncomputable def sin_deg (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def cos_deg (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)

theorem find_smallest_positive_angle :
  ∃ φ > 0, cos_deg φ = sin_deg 45 + cos_deg 37 - sin_deg 23 - cos_deg 11 ∧ φ = 53 := 
by
  sorry

end NUMINAMATH_GPT_find_smallest_positive_angle_l639_63917


namespace NUMINAMATH_GPT_point_on_or_outside_circle_l639_63941

theorem point_on_or_outside_circle (a : ℝ) : 
  let P := (a, 2 - a)
  let r := 2
  let center := (0, 0)
  let distance_square := (P.1 - center.1)^2 + (P.2 - center.2)^2
  distance_square >= r := 
by
  sorry

end NUMINAMATH_GPT_point_on_or_outside_circle_l639_63941


namespace NUMINAMATH_GPT_fundraising_part1_fundraising_part2_l639_63931

-- Problem 1
theorem fundraising_part1 (x y : ℕ) 
(h1 : x + y = 60) 
(h2 : 100 * x + 80 * y = 5600) :
x = 40 ∧ y = 20 := 
by 
  sorry

-- Problem 2
theorem fundraising_part2 (a : ℕ) 
(h1 : 100 * a + 80 * (80 - a) ≤ 6890) :
a ≤ 24 := 
by 
  sorry

end NUMINAMATH_GPT_fundraising_part1_fundraising_part2_l639_63931


namespace NUMINAMATH_GPT_circle_tangent_locus_l639_63943

theorem circle_tangent_locus (a b : ℝ) :
  (∃ r : ℝ, (a ^ 2 + b ^ 2 = (r + 1) ^ 2) ∧ ((a - 3) ^ 2 + b ^ 2 = (5 - r) ^ 2)) →
  3 * a ^ 2 + 4 * b ^ 2 - 14 * a - 49 = 0 := by
  sorry

end NUMINAMATH_GPT_circle_tangent_locus_l639_63943


namespace NUMINAMATH_GPT_bart_pages_bought_l639_63945

theorem bart_pages_bought (total_money : ℝ) (price_per_notepad : ℝ) (pages_per_notepad : ℕ)
  (h1 : total_money = 10) (h2 : price_per_notepad = 1.25) (h3 : pages_per_notepad = 60) :
  total_money / price_per_notepad * pages_per_notepad = 480 :=
by
  sorry

end NUMINAMATH_GPT_bart_pages_bought_l639_63945


namespace NUMINAMATH_GPT_evaluate_expression_l639_63947

theorem evaluate_expression (x y z : ℝ) (hxy : x > y ∧ y > 1) (hz : z > 0) :
  (x^y * y^(x+z)) / (y^(y+z) * x^x) = (x / y)^(y - x) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l639_63947


namespace NUMINAMATH_GPT_division_expression_l639_63914

theorem division_expression :
  (240 : ℚ) / (12 + 12 * 2 - 3) = 240 / 33 := by
  sorry

end NUMINAMATH_GPT_division_expression_l639_63914


namespace NUMINAMATH_GPT_total_fruits_consumed_l639_63923

def starting_cherries : ℝ := 16.5
def remaining_cherries : ℝ := 6.3

def starting_strawberries : ℝ := 10.7
def remaining_strawberries : ℝ := 8.4

def starting_blueberries : ℝ := 20.2
def remaining_blueberries : ℝ := 15.5

theorem total_fruits_consumed 
  (sc : ℝ := starting_cherries)
  (rc : ℝ := remaining_cherries)
  (ss : ℝ := starting_strawberries)
  (rs : ℝ := remaining_strawberries)
  (sb : ℝ := starting_blueberries)
  (rb : ℝ := remaining_blueberries) :
  (sc - rc) + (ss - rs) + (sb - rb) = 17.2 := by
  sorry

end NUMINAMATH_GPT_total_fruits_consumed_l639_63923


namespace NUMINAMATH_GPT_shara_shells_after_vacation_l639_63976

-- Definitions based on conditions
def initial_shells : ℕ := 20
def shells_per_day : ℕ := 5
def days : ℕ := 3
def shells_fourth_day : ℕ := 6

-- Statement of the proof problem
theorem shara_shells_after_vacation : 
  initial_shells + (shells_per_day * days) + shells_fourth_day = 41 := by
  sorry

end NUMINAMATH_GPT_shara_shells_after_vacation_l639_63976


namespace NUMINAMATH_GPT_solve_mod_equiv_l639_63977

theorem solve_mod_equiv : ∃ (n : ℤ), 0 ≤ n ∧ n < 9 ∧ (-2222 ≡ n [ZMOD 9]) → n = 6 := by
  sorry

end NUMINAMATH_GPT_solve_mod_equiv_l639_63977


namespace NUMINAMATH_GPT_increase_in_average_l639_63928

theorem increase_in_average (A : ℤ) (avg_after_12 : ℤ) (score_12th_inning : ℤ) (A : ℤ) : 
  score_12th_inning = 75 → avg_after_12 = 64 → (11 * A + score_12th_inning = 768) → (avg_after_12 - A = 1) :=
by
  intros h_score h_avg h_total
  sorry

end NUMINAMATH_GPT_increase_in_average_l639_63928


namespace NUMINAMATH_GPT_proof_a_squared_plus_b_squared_l639_63908

theorem proof_a_squared_plus_b_squared (a b : ℝ) (h1 : (a + b) ^ 2 = 4) (h2 : a * b = 1) : a ^ 2 + b ^ 2 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_proof_a_squared_plus_b_squared_l639_63908


namespace NUMINAMATH_GPT_check_range_a_l639_63907

open Set

def A : Set ℝ := {x | x < -1/2 ∨ x > 1}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x - 1 ≤ 0}

theorem check_range_a :
  (∃! x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ (x₁ : ℝ) ∈ A ∩ B a ∧ (x₂ : ℝ) ∈ A ∩ B a) →
  a ∈ Icc (4/3 : ℝ) (15/8 : ℝ) :=
sorry

end NUMINAMATH_GPT_check_range_a_l639_63907


namespace NUMINAMATH_GPT_initial_selling_price_l639_63963

theorem initial_selling_price (P : ℝ) : 
    (∀ (c_i c_m p_m r : ℝ),
        c_i = 3 ∧
        c_m = 20 ∧
        p_m = 4 ∧
        r = 50 ∧
        (15 * P + 5 * p_m - 20 * c_i = r)
    ) → 
    P = 6 := by 
    sorry

end NUMINAMATH_GPT_initial_selling_price_l639_63963


namespace NUMINAMATH_GPT_num_true_statements_l639_63940

theorem num_true_statements :
  (if (2 : ℝ) = 2 then (2 : ℝ)^2 - 4 = 0 else false) ∧
  ((∀ (x : ℝ), x^2 - 4 = 0 → x = 2) ∨ (∃ (x : ℝ), x^2 - 4 = 0 ∧ x ≠ 2)) ∧
  ((∀ (x : ℝ), x ≠ 2 → x^2 - 4 ≠ 0) ∨ (∃ (x : ℝ), x ≠ 2 ∧ x^2 - 4 = 0)) ∧
  ((∀ (x : ℝ), x^2 - 4 ≠ 0 → x ≠ 2) ∨ (∃ (x : ℝ), x^2 - 4 ≠ 0 ∧ x = 2)) :=
sorry

end NUMINAMATH_GPT_num_true_statements_l639_63940


namespace NUMINAMATH_GPT_dwarfs_truthful_count_l639_63981

theorem dwarfs_truthful_count :
  ∃ (T L : ℕ), T + L = 10 ∧
    (∀ t : ℕ, t = 10 → t + ((10 - T) * 2 - T) = 16) ∧
    T = 4 :=
by
  sorry

end NUMINAMATH_GPT_dwarfs_truthful_count_l639_63981


namespace NUMINAMATH_GPT_probability_both_asian_selected_probability_A1_but_not_B1_selected_l639_63966

noncomputable def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_both_asian_selected (A1 A2 A3 B1 B2 B3 : Prop) :
  let total_ways := choose 6 2
  let asian_ways := choose 3 2
  asian_ways / total_ways = 1 / 5 := by
  let total_ways := choose 6 2
  let asian_ways := choose 3 2
  sorry

theorem probability_A1_but_not_B1_selected (A1 A2 A3 B1 B2 B3 : Prop) :
  let total_ways := 9
  let valid_ways := 2
  valid_ways / total_ways = 2 / 9 := by
  let total_ways := 9
  let valid_ways := 2
  sorry

end NUMINAMATH_GPT_probability_both_asian_selected_probability_A1_but_not_B1_selected_l639_63966


namespace NUMINAMATH_GPT_intersection_correct_l639_63921

variable (A B : Set ℝ)  -- Define variables A and B as sets of real numbers

-- Define set A as {x | -3 ≤ x < 4}
def setA : Set ℝ := {x | -3 ≤ x ∧ x < 4}

-- Define set B as {x | -2 ≤ x ≤ 5}
def setB : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- The goal is to prove the intersection of A and B is {x | -2 ≤ x < 4}
theorem intersection_correct : setA ∩ setB = {x : ℝ | -2 ≤ x ∧ x < 4} := sorry

end NUMINAMATH_GPT_intersection_correct_l639_63921


namespace NUMINAMATH_GPT_polynomial_degree_bound_l639_63957

theorem polynomial_degree_bound (m n k : ℕ) (P : Polynomial ℤ) 
  (hm_pos : 0 < m)
  (hn_pos : 0 < n)
  (hk_pos : 2 ≤ k)
  (hP_odd : ∀ i, P.coeff i % 2 = 1) 
  (h_div : (X - 1) ^ m ∣ P)
  (hm_bound : m ≥ 2 ^ k) :
  n ≥ 2 ^ (k + 1) - 1 := sorry

end NUMINAMATH_GPT_polynomial_degree_bound_l639_63957


namespace NUMINAMATH_GPT_fred_games_last_year_l639_63958

def total_games : Nat := 47
def games_this_year : Nat := 36

def games_last_year (total games games this year : Nat) : Nat := total_games - games_this_year

theorem fred_games_last_year : games_last_year total_games games_this_year = 11 :=
by
  sorry

end NUMINAMATH_GPT_fred_games_last_year_l639_63958


namespace NUMINAMATH_GPT_solve_for_x_l639_63924

theorem solve_for_x (x : ℝ) : 5 * x - 3 * x = 405 - 9 * (x + 4) → x = 369 / 11 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l639_63924


namespace NUMINAMATH_GPT_evaluate_g_at_5_l639_63998

noncomputable def g (x : ℝ) : ℝ := 2 * x ^ 4 - 15 * x ^ 3 + 24 * x ^ 2 - 18 * x - 72

theorem evaluate_g_at_5 : g 5 = -7 := by
  sorry

end NUMINAMATH_GPT_evaluate_g_at_5_l639_63998


namespace NUMINAMATH_GPT_john_walks_farther_l639_63979

theorem john_walks_farther :
  let john_distance : ℝ := 1.74
  let nina_distance : ℝ := 1.235
  john_distance - nina_distance = 0.505 :=
by
  sorry

end NUMINAMATH_GPT_john_walks_farther_l639_63979


namespace NUMINAMATH_GPT_roof_problem_l639_63951

theorem roof_problem (w l : ℝ) (h1 : l = 4 * w) (h2 : l * w = 900) : l - w = 45 := 
by
  sorry

end NUMINAMATH_GPT_roof_problem_l639_63951


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l639_63905

theorem necessary_but_not_sufficient (x : ℝ) :
  (x < 2 → (x^2 - x - 2 >= 0) ∨ (x >= -1 ∧ x < 2)) ∧ ((-1 < x ∧ x < 2) → x < 2) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l639_63905


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l639_63906

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- Define the first term, common difference and positions of terms in geometric sequence
def a1 : ℤ := -8
def d : ℤ := 2
def a3 := arithmetic_sequence a1 d 2
def a4 := arithmetic_sequence a1 d 3

-- Conditions for the terms forming a geometric sequence
def geometric_condition (a b c : ℤ) : Prop :=
  b^2 = a * c

-- Statement to prove
theorem arithmetic_geometric_sequence :
  geometric_condition a1 a3 a4 → a1 = -8 :=
by
  intro h
  -- Proof can be filled in here
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l639_63906


namespace NUMINAMATH_GPT_sum_of_possible_values_l639_63937

theorem sum_of_possible_values (N : ℝ) (h : N * (N - 8) = 4) : ∃ S : ℝ, S = 8 :=
sorry

end NUMINAMATH_GPT_sum_of_possible_values_l639_63937


namespace NUMINAMATH_GPT_seq_sum_11_l639_63953

noncomputable def S (n : ℕ) : ℕ := sorry

noncomputable def a (n : ℕ) : ℕ := sorry

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

theorem seq_sum_11 :
  (∀ n : ℕ, S n = (n * (a 1 + a n)) / 2) ∧
  (is_arithmetic_sequence a) ∧
  (3 * (a 2 + a 4) + 2 * (a 6 + a 9 + a 12) = 12) →
  S 11 = 11 :=
by
  sorry

end NUMINAMATH_GPT_seq_sum_11_l639_63953


namespace NUMINAMATH_GPT_cost_of_bananas_and_cantaloupe_l639_63971

-- Define prices for different items
variables (a b c d e : ℝ)

-- Define the conditions as hypotheses
theorem cost_of_bananas_and_cantaloupe (h1 : a + b + c + d + e = 30)
    (h2 : d = 3 * a) (h3 : c = a - b) (h4 : e = a + b) :
    b + c = 5 := 
by 
  -- Initial proof setup
  sorry

end NUMINAMATH_GPT_cost_of_bananas_and_cantaloupe_l639_63971


namespace NUMINAMATH_GPT_find_EQ_length_l639_63939

theorem find_EQ_length (a b c d : ℕ) (parallel : Prop) (circle_tangent : Prop) :
  a = 105 ∧ b = 45 ∧ c = 21 ∧ d = 80 ∧ parallel ∧ circle_tangent → (∃ x : ℚ, x = 336 / 5) :=
by
  sorry

end NUMINAMATH_GPT_find_EQ_length_l639_63939


namespace NUMINAMATH_GPT_find_k_l639_63967

variable (a b : ℝ → ℝ → ℝ)
variable {k : ℝ}

-- Defining conditions
axiom a_perpendicular_b : ∀ x y, a x y = 0
axiom a_unit_vector : a 1 0 = 1
axiom b_unit_vector : b 0 1 = 1
axiom sum_perpendicular_to_k_diff : ∀ x y, (a x y + b x y) * (k * a x y - b x y) = 0

theorem find_k : k = 1 :=
sorry

end NUMINAMATH_GPT_find_k_l639_63967


namespace NUMINAMATH_GPT_Sheila_weekly_earnings_l639_63969

-- Definitions based on the conditions
def hours_per_day_MWF : ℕ := 8
def hours_per_day_TT : ℕ := 6
def hourly_wage : ℕ := 7
def days_MWF : ℕ := 3
def days_TT : ℕ := 2

-- Theorem that Sheila earns $252 per week
theorem Sheila_weekly_earnings : (hours_per_day_MWF * hourly_wage * days_MWF) + (hours_per_day_TT * hourly_wage * days_TT) = 252 :=
by 
  sorry

end NUMINAMATH_GPT_Sheila_weekly_earnings_l639_63969


namespace NUMINAMATH_GPT_sin_alpha_minus_3pi_l639_63944

theorem sin_alpha_minus_3pi (α : ℝ) (h : Real.sin α = 3/5) : Real.sin (α - 3 * Real.pi) = -3/5 :=
by
  sorry

end NUMINAMATH_GPT_sin_alpha_minus_3pi_l639_63944


namespace NUMINAMATH_GPT_circumference_proportionality_l639_63902

theorem circumference_proportionality (r : ℝ) (C : ℝ) (k : ℝ) (π : ℝ)
  (h1 : C = k * r)
  (h2 : C = 2 * π * r) :
  k = 2 * π :=
sorry

end NUMINAMATH_GPT_circumference_proportionality_l639_63902


namespace NUMINAMATH_GPT_hyperbola_b_value_l639_63932

theorem hyperbola_b_value (b : ℝ) (h₁ : b > 0) 
  (h₂ : ∃ x y, x^2 - (y^2 / b^2) = 1 ∧ (∀ (c : ℝ), c = Real.sqrt (1 + b^2) → c / 1 = 2)) : b = Real.sqrt 3 :=
by { sorry }

end NUMINAMATH_GPT_hyperbola_b_value_l639_63932
