import Mathlib

namespace find_b_plus_m_l2252_225218

section MatrixPower

open Matrix

-- Define our matrices
def A (b m : ℕ) : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![1, 3, b], 
    ![0, 1, 5], 
    ![0, 0, 1]]

def B : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![1, 27, 3008], 
    ![0, 1, 45], 
    ![0, 0, 1]]

-- The problem statement
noncomputable def power_eq_matrix (b m : ℕ) : Prop :=
  (A b m) ^ m = B

-- The final goal
theorem find_b_plus_m (b m : ℕ) (h : power_eq_matrix b m) : b + m = 283 := sorry

end MatrixPower

end find_b_plus_m_l2252_225218


namespace ways_to_distribute_items_l2252_225207

/-- The number of ways to distribute 5 different items into 4 identical bags, with some bags possibly empty, is 36. -/
theorem ways_to_distribute_items : ∃ (n : ℕ), n = 36 := by
  sorry

end ways_to_distribute_items_l2252_225207


namespace sum_of_interior_angles_increases_l2252_225264

theorem sum_of_interior_angles_increases (n : ℕ) (h : n ≥ 3) : (n-2) * 180 > (n-3) * 180 :=
by
  sorry

end sum_of_interior_angles_increases_l2252_225264


namespace emma_ate_more_than_liam_l2252_225287

-- Definitions based on conditions
def emma_oranges : ℕ := 8
def liam_oranges : ℕ := 1

-- Lean statement to prove the question
theorem emma_ate_more_than_liam : emma_oranges - liam_oranges = 7 := by
  sorry

end emma_ate_more_than_liam_l2252_225287


namespace find_certain_number_l2252_225261

def certain_number (x : ℤ) : Prop := x - 9 = 5

theorem find_certain_number (x : ℤ) (h : certain_number x) : x = 14 :=
by
  sorry

end find_certain_number_l2252_225261


namespace half_angle_in_first_quadrant_l2252_225216

theorem half_angle_in_first_quadrant (α : ℝ) (h : 0 < α ∧ α < π / 2) : 0 < α / 2 ∧ α / 2 < π / 4 :=
by
  sorry

end half_angle_in_first_quadrant_l2252_225216


namespace opposite_of_neg_two_thirds_l2252_225213

theorem opposite_of_neg_two_thirds : -(- (2 / 3)) = (2 / 3) :=
by
  sorry

end opposite_of_neg_two_thirds_l2252_225213


namespace converse_statement_2_true_implies_option_A_l2252_225266

theorem converse_statement_2_true_implies_option_A :
  (∀ x : ℕ, x = 1 ∨ x = 2 → (x^2 - 3 * x + 2 = 0)) →
  (x = 1 ∨ x = 2) :=
by
  intro h
  sorry

end converse_statement_2_true_implies_option_A_l2252_225266


namespace largest_angle_in_pentagon_l2252_225269

theorem largest_angle_in_pentagon (P Q R S T : ℝ) 
          (h1 : P = 70) 
          (h2 : Q = 100)
          (h3 : R = S) 
          (h4 : T = 3 * R - 25)
          (h5 : P + Q + R + S + T = 540) : 
          T = 212 :=
by
  sorry

end largest_angle_in_pentagon_l2252_225269


namespace monotonicity_of_f_l2252_225254

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x^2 + 1

theorem monotonicity_of_f (a x : ℝ) :
  (a > 0 → ((∀ x, (x < -2 * a / 3 → f a x' > f a x) ∧ (x > 0 → f a x' > f a x)) ∧ (∀ x, (-2 * a / 3 < x ∧ x < 0 → f a x' < f a x)))) ∧
  (a = 0 → ∀ x, f a x' > f a x) ∧
  (a < 0 → ((∀ x, (x < 0 → f a x' > f a x) ∧ (x > -2 * a / 3 → f a x' > f a x)) ∧ (∀ x, (0 < x ∧ x < -2 * a / 3 → f a x' < f a x)))) :=
sorry

end monotonicity_of_f_l2252_225254


namespace trees_probability_l2252_225223

theorem trees_probability (num_maple num_oak num_birch total_slots total_trees : ℕ) 
                         (maple_count oak_count birch_count : Prop)
                         (prob_correct : Prop) :
  num_maple = 4 →
  num_oak = 5 →
  num_birch = 6 →
  total_trees = 15 →
  total_slots = 10 →
  maple_count → oak_count → birch_count →
  prob_correct →
  (m + n = 57) :=
by
  intros
  sorry

end trees_probability_l2252_225223


namespace order_of_a_b_c_l2252_225237

noncomputable def a : ℝ := Real.log 2 / Real.log 3 -- a = log_3 2
noncomputable def b : ℝ := Real.log 2 -- b = ln 2
noncomputable def c : ℝ := Real.sqrt 5 -- c = 5^(1/2)

theorem order_of_a_b_c : a < b ∧ b < c := by
  sorry

end order_of_a_b_c_l2252_225237


namespace sum_of_three_squares_l2252_225268

theorem sum_of_three_squares (n : ℕ) (h : n = 100) : 
  ∃ (a b c : ℕ), a = 4 ∧ b^2 + c^2 = 84 ∧ a^2 + b^2 + c^2 = 100 ∧ 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c ∨ (b = c ∧ a ≠ b)) ∧
  (4^2 + 7^2 + 6^2 = 100 ∧ 4^2 + 8^2 + 5^2 = 100 ∧ 4^2 + 9^2 + 1^2 = 100) ∧
  (4^2 + 6^2 + 7^2 ≠ 100 ∧ 4^2 + 5^2 + 8^2 ≠ 100 ∧ 4^2 + 1^2 + 9^2 ≠ 100 ∧ 
   4^2 + 4^2 + 8^2 ≠ 100 ∨ 4^2 + 8^2 + 4^2 ≠ 100) :=
sorry

end sum_of_three_squares_l2252_225268


namespace determine_c_square_of_binomial_l2252_225210

theorem determine_c_square_of_binomial (c : ℝ) : (∀ x : ℝ, 16 * x^2 + 40 * x + c = (4 * x + 5)^2) → c = 25 :=
by
  intro h
  have key := h 0
  -- By substitution, we skip the expansion steps and immediately conclude the value of c
  sorry

end determine_c_square_of_binomial_l2252_225210


namespace polynomial_value_l2252_225272

variable (a b : ℝ)

theorem polynomial_value :
  2 * a + 3 * b = 5 → 6 * a + 9 * b - 12 = 3 :=
by
  intro h
  sorry

end polynomial_value_l2252_225272


namespace arithmetic_sequence_sum_l2252_225267

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (b : ℕ → ℚ) (T : ℕ → ℚ) :
  (a 3 = 7) ∧ (a 5 + a 7 = 26) →
  (∀ n, a n = 2 * n + 1) ∧
  (∀ n, S n = n^2 + 2 * n) ∧
  (∀ n, b n = 1 / ((a n)^2 - 1)) →
  (∀ n, T n = n / (4 * (n + 1))) := sorry

end arithmetic_sequence_sum_l2252_225267


namespace parabola_directrix_l2252_225291

noncomputable def directrix_value (a : ℝ) : ℝ := -1 / (4 * a)

theorem parabola_directrix (a : ℝ) (h : directrix_value a = 2) : a = -1 / 8 :=
by
  sorry

end parabola_directrix_l2252_225291


namespace quadratic_root_conditions_l2252_225274

theorem quadratic_root_conditions (a b : ℝ)
    (h1 : ∃ k : ℝ, ∀ x : ℝ, x^2 + 2 * x + 3 - k = 0)
    (h2 : ∀ α β : ℝ, α * β = 3 - k ∧ k^2 = α * β + 3 * k) : 
    k = 3 := 
sorry

end quadratic_root_conditions_l2252_225274


namespace mark_total_spending_l2252_225262

variable (p_tomato_cost : ℕ) (p_apple_cost : ℕ) 
variable (pounds_tomato : ℕ) (pounds_apple : ℕ)

def total_cost (p_tomato_cost : ℕ) (pounds_tomato : ℕ) (p_apple_cost : ℕ) (pounds_apple : ℕ) : ℕ :=
  (p_tomato_cost * pounds_tomato) + (p_apple_cost * pounds_apple)

theorem mark_total_spending :
  total_cost 5 2 6 5 = 40 :=
by
  sorry

end mark_total_spending_l2252_225262


namespace not_all_polynomials_sum_of_cubes_l2252_225202

theorem not_all_polynomials_sum_of_cubes :
  ¬ ∀ P : Polynomial ℤ, ∃ Q : Polynomial ℤ, P = Q^3 + Q^3 + Q^3 :=
by
  sorry

end not_all_polynomials_sum_of_cubes_l2252_225202


namespace slope_of_line_l2252_225294

theorem slope_of_line (s x y : ℝ) (h1 : 2 * x + 3 * y = 8 * s + 5) (h2 : x + 2 * y = 3 * s + 2) :
  ∃ m c : ℝ, ∀ x y, x = m * y + c ∧ m = -7/2 :=
by
  sorry

end slope_of_line_l2252_225294


namespace semicircle_perimeter_l2252_225240

theorem semicircle_perimeter (r : ℝ) (π : ℝ) (h : 0 < π) (r_eq : r = 14):
  (14 * π + 28) = 14 * π + 28 :=
by
  sorry

end semicircle_perimeter_l2252_225240


namespace infinite_castle_hall_unique_l2252_225288

theorem infinite_castle_hall_unique :
  (∀ (n : ℕ), ∃ hall : ℕ, ∀ m : ℕ, ((m = 2 * n + 1) ∨ (m = 3 * n + 1)) → hall = m) →
  ∀ (hall1 hall2 : ℕ), hall1 = hall2 :=
by
  sorry

end infinite_castle_hall_unique_l2252_225288


namespace neg_neg_eq_pos_l2252_225225

theorem neg_neg_eq_pos : -(-2023) = 2023 :=
by
  sorry

end neg_neg_eq_pos_l2252_225225


namespace color_opposite_gold_is_yellow_l2252_225242

-- Define the colors as a datatype for clarity
inductive Color
| B | Y | O | K | S | G

-- Define the type for each face's color
structure CubeFaces :=
(top front right back left bottom : Color)

-- Given conditions
def first_view (c : CubeFaces) : Prop :=
  c.top = Color.B ∧ c.front = Color.Y ∧ c.right = Color.O

def second_view (c : CubeFaces) : Prop :=
  c.top = Color.B ∧ c.front = Color.K ∧ c.right = Color.O

def third_view (c : CubeFaces) : Prop :=
  c.top = Color.B ∧ c.front = Color.S ∧ c.right = Color.O

-- Problem statement
theorem color_opposite_gold_is_yellow (c : CubeFaces) :
  first_view c → second_view c → third_view c → (c.back = Color.G) → (c.front = Color.Y) :=
by
  sorry

end color_opposite_gold_is_yellow_l2252_225242


namespace no_such_function_exists_l2252_225234

theorem no_such_function_exists 
  (f : ℝ → ℝ) 
  (h_f_pos : ∀ x, 0 < x → 0 < f x) 
  (h_eq : ∀ x y, 0 < x → 0 < y → f (x + y) = f x + f y + (1 / 2012)) : 
  false :=
sorry

end no_such_function_exists_l2252_225234


namespace find_f_value_l2252_225228

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x + 1

theorem find_f_value : f 2019 + f (-2019) = 2 :=
by
  sorry

end find_f_value_l2252_225228


namespace range_of_k_for_domain_real_l2252_225217

theorem range_of_k_for_domain_real (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 6 * k * x + (k + 8) ≥ 0) ↔ (0 ≤ k ∧ k ≤ 1) :=
sorry

end range_of_k_for_domain_real_l2252_225217


namespace sequence_general_term_l2252_225232

/-- Given the sequence {a_n} defined by a_n = 2^n * a_{n-1} for n > 1 and a_1 = 1,
    prove that the general term a_n = 2^((n^2 + n - 2) / 2) -/
theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1) 
  (h2 : ∀ n > 1, a n = 2^n * a (n-1)) :
  ∀ n, a n = 2^((n^2 + n - 2) / 2) :=
sorry

end sequence_general_term_l2252_225232


namespace number_of_white_balls_l2252_225241

theorem number_of_white_balls (x : ℕ) (h : (x : ℚ) / (x + 12) = 2 / 3) : x = 24 :=
sorry

end number_of_white_balls_l2252_225241


namespace camp_cedar_counselors_l2252_225219

theorem camp_cedar_counselors (boys : ℕ) (girls : ℕ) (total_children : ℕ) (counselors : ℕ)
  (h1 : boys = 40)
  (h2 : girls = 3 * boys)
  (h3 : total_children = boys + girls)
  (h4 : counselors = total_children / 8) : 
  counselors = 20 :=
by sorry

end camp_cedar_counselors_l2252_225219


namespace classroom_lamps_total_ways_l2252_225273

theorem classroom_lamps_total_ways (n : ℕ) (h : n = 4) : (2^n - 1) = 15 :=
by
  sorry

end classroom_lamps_total_ways_l2252_225273


namespace num_points_P_on_ellipse_l2252_225206

noncomputable def ellipse : Set (ℝ × ℝ) := {p | (p.1)^2 / 16 + (p.2)^2 / 9 = 1}
noncomputable def line : Set (ℝ × ℝ) := {p | p.1 / 4 + p.2 / 3 = 1}
noncomputable def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := 
  (1 / 2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

theorem num_points_P_on_ellipse (A B : ℝ × ℝ) 
  (hA_on_line : A ∈ line) (hA_on_ellipse : A ∈ ellipse) 
  (hB_on_line : B ∈ line) (hB_on_ellipse : B ∈ ellipse)
  : ∃ P1 P2 : ℝ × ℝ, P1 ∈ ellipse ∧ P2 ∈ ellipse ∧ 
    area_triangle A B P1 = 3 ∧ area_triangle A B P2 = 3 ∧ 
    P1 ≠ P2 ∧ 
    (∀ P : ℝ × ℝ, P ∈ ellipse ∧ area_triangle A B P = 3 → P = P1 ∨ P = P2) := 
sorry

end num_points_P_on_ellipse_l2252_225206


namespace one_fourth_in_one_eighth_l2252_225246

theorem one_fourth_in_one_eighth : (1 / 8) / (1 / 4) = 1 / 2 := by
  sorry

end one_fourth_in_one_eighth_l2252_225246


namespace calculate_f_g2_l2252_225281

def f (x : ℝ) : ℝ := x^2 + 1
def g (x : ℝ) : ℝ := 2 * x^3 - 1

theorem calculate_f_g2 : f (g 2) = 226 := by
  sorry

end calculate_f_g2_l2252_225281


namespace ticket_difference_l2252_225205

/-- 
  Define the initial number of tickets Billy had,
  the number of tickets after buying a yoyo,
  and state the proof that the difference is 16.
--/

theorem ticket_difference (initial_tickets : ℕ) (remaining_tickets : ℕ) 
  (h₁ : initial_tickets = 48) (h₂ : remaining_tickets = 32) : 
  initial_tickets - remaining_tickets = 16 :=
by
  /- This is where the prover would go, 
     no need to implement it as we know the expected result -/
  sorry

end ticket_difference_l2252_225205


namespace parabola_coordinates_l2252_225276

theorem parabola_coordinates (x y : ℝ) (h_parabola : y^2 = 4 * x) (h_distance : (x - 1)^2 + y^2 = 100) :
  (x = 9 ∧ y = 6) ∨ (x = 9 ∧ y = -6) :=
by
  sorry

end parabola_coordinates_l2252_225276


namespace solve_for_x_l2252_225245

theorem solve_for_x (x : ℝ) (h1 : x > 0) (h2 : 3 * x^2 + 8 * x - 16 = 0) : x = 4 / 3 :=
by
  sorry

end solve_for_x_l2252_225245


namespace Haman_initial_trays_l2252_225220

theorem Haman_initial_trays 
  (eggs_in_tray : ℕ)
  (total_eggs_sold : ℕ)
  (trays_dropped : ℕ)
  (additional_trays : ℕ)
  (trays_finally_sold : ℕ)
  (std_trays_sold : total_eggs_sold / eggs_in_tray = trays_finally_sold) 
  (eggs_in_tray_def : eggs_in_tray = 30) 
  (total_eggs_sold_def : total_eggs_sold = 540)
  (trays_dropped_def : trays_dropped = 2)
  (additional_trays_def : additional_trays = 7) :
  trays_finally_sold - additional_trays + trays_dropped = 13 := 
by 
  sorry

end Haman_initial_trays_l2252_225220


namespace average_weight_of_abc_l2252_225200

theorem average_weight_of_abc (A B C : ℝ) (h1 : (A + B) / 2 = 40) (h2 : (B + C) / 2 = 43) (h3 : B = 40) : 
  (A + B + C) / 3 = 42 := 
sorry

end average_weight_of_abc_l2252_225200


namespace garden_length_l2252_225233

theorem garden_length :
  ∀ (w : ℝ) (l : ℝ),
  (l = 2 * w) →
  (2 * l + 2 * w = 150) →
  l = 50 :=
by
  intros w l h1 h2
  sorry

end garden_length_l2252_225233


namespace sharing_watermelons_l2252_225243

theorem sharing_watermelons (h : 8 = people_per_watermelon) : people_for_4_watermelons = 32 :=
by
  let people_per_watermelon := 8
  let watermelons := 4
  let people_for_4_watermelons := people_per_watermelon * watermelons
  sorry

end sharing_watermelons_l2252_225243


namespace train_journey_time_l2252_225285

theorem train_journey_time :
  ∃ T : ℝ, (30 : ℝ) / 60 = (7 / 6 * T) - T ∧ T = 3 :=
by
  sorry

end train_journey_time_l2252_225285


namespace unique_solution_iff_d_ne_4_l2252_225253

theorem unique_solution_iff_d_ne_4 (c d : ℝ) : 
  (∃! (x : ℝ), 4 * x - 7 + c = d * x + 2) ↔ d ≠ 4 := 
by 
  sorry

end unique_solution_iff_d_ne_4_l2252_225253


namespace calc_power_expression_l2252_225211

theorem calc_power_expression (a b c : ℕ) (h₁ : b = 2) (h₂ : c = 3) :
  3^15 * (3^b)^5 / (3^c)^6 = 2187 := 
sorry

end calc_power_expression_l2252_225211


namespace system_of_equations_l2252_225203

theorem system_of_equations (x y k : ℝ) 
  (h1 : x + 2 * y = k + 2) 
  (h2 : 2 * x - 3 * y = 3 * k - 1) : 
  x + 9 * y = 7 :=
  sorry

end system_of_equations_l2252_225203


namespace problem_solution_l2252_225299

theorem problem_solution
  (p q : ℝ)
  (h₁ : p ≠ q)
  (h₂ : (x : ℝ) → (x - 5) * (x + 3) = 24 * x - 72 → x = p ∨ x = q)
  (h₃ : p > q) :
  p - q = 20 :=
sorry

end problem_solution_l2252_225299


namespace find_f_neg2003_l2252_225229

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_neg2003 (f_defined : ∀ x : ℝ, ∃ y : ℝ, f y = x → f y ≠ 0)
  (cond1 : ∀ ⦃x y w : ℝ⦄, x > y → (f x + x ≥ w → w ≥ f y + y → ∃ z, y ≤ z ∧ z ≤ x ∧ f z = w - z))
  (cond2 : ∃ u : ℝ, f u = 0 ∧ ∀ v : ℝ, f v = 0 → u ≤ v)
  (cond3 : f 0 = 1)
  (cond4 : f (-2003) ≤ 2004)
  (cond5 : ∀ x y : ℝ, f x * f y = f (x * f y + y * f x + x * y)) :
  f (-2003) = 2004 :=
sorry

end find_f_neg2003_l2252_225229


namespace solve_inequality_l2252_225259

theorem solve_inequality (a b : ℝ) (h : ∀ x, (x > 1 ∧ x < 2) ↔ (x - a) * (x - b) < 0) : a + b = 3 :=
sorry

end solve_inequality_l2252_225259


namespace trebled_result_of_original_number_is_72_l2252_225284

theorem trebled_result_of_original_number_is_72:
  ∀ (x : ℕ), x = 9 → 3 * (2 * x + 6) = 72 :=
by
  intro x h
  sorry

end trebled_result_of_original_number_is_72_l2252_225284


namespace total_practice_hours_correct_l2252_225292

-- Define the conditions
def daily_practice_hours : ℕ := 5 -- The team practices 5 hours daily
def missed_days : ℕ := 1 -- They missed practicing 1 day this week
def days_in_week : ℕ := 7 -- There are 7 days in a week

-- Calculate the number of days they practiced
def practiced_days : ℕ := days_in_week - missed_days

-- Calculate the total hours practiced
def total_practice_hours : ℕ := practiced_days * daily_practice_hours

-- Theorem to prove the total hours practiced is 30
theorem total_practice_hours_correct : total_practice_hours = 30 := by
  -- Start the proof; skipping the actual proof steps
  sorry

end total_practice_hours_correct_l2252_225292


namespace evaluate_complex_fraction_l2252_225221

def complex_fraction : Prop :=
  let expr : ℚ := 2 + (3 / (4 + (5 / 6)))
  expr = 76 / 29

theorem evaluate_complex_fraction : complex_fraction :=
by
  let expr : ℚ := 2 + (3 / (4 + (5 / 6)))
  show expr = 76 / 29
  sorry

end evaluate_complex_fraction_l2252_225221


namespace circle_area_l2252_225249

theorem circle_area (r : ℝ) (h : 3 * (1 / (2 * π * r)) = r) : 
  π * r^2 = 3 / 2 :=
by
  sorry

end circle_area_l2252_225249


namespace total_rounds_played_l2252_225252

/-- William and Harry played some rounds of tic-tac-toe.
    William won 5 more rounds than Harry.
    William won 10 rounds.
    Prove that the total number of rounds they played is 15. -/
theorem total_rounds_played (williams_wins : ℕ) (harrys_wins : ℕ)
  (h1 : williams_wins = 10)
  (h2 : williams_wins = harrys_wins + 5) :
  williams_wins + harrys_wins = 15 := 
by
  sorry

end total_rounds_played_l2252_225252


namespace contradiction_method_assumption_l2252_225227

theorem contradiction_method_assumption (a b c : ℝ) :
  (¬(a > 0 ∨ b > 0 ∨ c > 0) → false) :=
sorry

end contradiction_method_assumption_l2252_225227


namespace problem_equiv_proof_l2252_225256

variable (a b : ℝ)
variable (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1)

theorem problem_equiv_proof :
  (2 ^ a + 2 ^ b ≥ 2 * Real.sqrt 2) ∧
  (Real.log a / Real.log 2 + Real.log b / Real.log 2 ≤ -2) ∧
  (a ^ 2 + b ^ 2 ≥ 1 / 2) :=
by
  sorry

end problem_equiv_proof_l2252_225256


namespace socks_count_l2252_225298

theorem socks_count :
  ∃ (x y z : ℕ), x + y + z = 12 ∧ x + 3 * y + 4 * z = 24 ∧ 1 ≤ x ∧ 1 ≤ y ∧ 1 <= z ∧ x = 7 :=
by
  sorry

end socks_count_l2252_225298


namespace ratio_of_elements_l2252_225255

theorem ratio_of_elements (total_weight : ℕ) (element_B_weight : ℕ) 
  (h_total : total_weight = 324) (h_B : element_B_weight = 270) :
  (total_weight - element_B_weight) / element_B_weight = 1 / 5 :=
by
  sorry

end ratio_of_elements_l2252_225255


namespace difference_q_r_l2252_225260

-- Conditions
variables (p q r : ℕ) (x : ℕ)
variables (h_ratio : 3 * x = p) (h_ratio2 : 7 * x = q) (h_ratio3 : 12 * x = r)
variables (h_diff_pq : q - p = 3200)

-- Proof problem to solve
theorem difference_q_r : q - p = 3200 → 12 * x - 7 * x = 4000 :=
by 
  intro h_diff_pq
  rw [h_ratio, h_ratio2, h_ratio3] at *
  sorry

end difference_q_r_l2252_225260


namespace rewrite_equation_l2252_225209

theorem rewrite_equation (x y : ℝ) (h : 2 * x - y = 4) : y = 2 * x - 4 :=
by
  sorry

end rewrite_equation_l2252_225209


namespace engagement_ring_savings_l2252_225263

theorem engagement_ring_savings 
  (yearly_salary : ℝ) 
  (monthly_savings : ℝ) 
  (monthly_salary := yearly_salary / 12) 
  (ring_cost := 2 * monthly_salary) 
  (saving_months := ring_cost / monthly_savings) 
  (h_salary : yearly_salary = 60000) 
  (h_savings : monthly_savings = 1000) :
  saving_months = 10 := 
sorry

end engagement_ring_savings_l2252_225263


namespace general_term_formula_l2252_225265

noncomputable def xSeq : ℕ → ℝ
| 0       => 3
| (n + 1) => (xSeq n)^2 + 2 / (2 * (xSeq n) - 1)

theorem general_term_formula (n : ℕ) : 
  xSeq n = (2 * 2^2^n + 1) / (2^2^n - 1) := 
sorry

end general_term_formula_l2252_225265


namespace sum_of_first_5n_l2252_225212

theorem sum_of_first_5n (n : ℕ) : 
  (n * (n + 1) / 2) + 210 = ((4 * n) * (4 * n + 1) / 2) → 
  (5 * n) * (5 * n + 1) / 2 = 465 :=
by sorry

end sum_of_first_5n_l2252_225212


namespace garden_width_l2252_225277

theorem garden_width (L W : ℕ) 
  (area_playground : 192 = 16 * 12)
  (area_garden : 192 = L * W)
  (perimeter_garden : 64 = 2 * L + 2 * W) :
  W = 12 :=
by
  sorry

end garden_width_l2252_225277


namespace sonny_cookie_problem_l2252_225239

theorem sonny_cookie_problem 
  (total_boxes : ℕ) (boxes_sister : ℕ) (boxes_cousin : ℕ) (boxes_left : ℕ) (boxes_brother : ℕ) : 
  total_boxes = 45 → boxes_sister = 9 → boxes_cousin = 7 → boxes_left = 17 → 
  boxes_brother = total_boxes - boxes_left - boxes_sister - boxes_cousin → 
  boxes_brother = 12 :=
by
  intros h_total h_sister h_cousin h_left h_brother
  rw [h_total, h_sister, h_cousin, h_left] at h_brother
  exact h_brother

end sonny_cookie_problem_l2252_225239


namespace actual_distance_traveled_l2252_225297

variable (t : ℝ) -- let t be the actual time in hours
variable (d : ℝ) -- let d be the actual distance traveled at 12 km/hr

-- Conditions
def condition1 := 20 * t = 12 * t + 30
def condition2 := d = 12 * t

-- The target we want to prove
theorem actual_distance_traveled (t : ℝ) (d : ℝ) (h1 : condition1 t) (h2 : condition2 t d) : 
  d = 45 := by
  sorry

end actual_distance_traveled_l2252_225297


namespace arrange_banana_l2252_225270

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l2252_225270


namespace trig_identity_evaluation_l2252_225257

theorem trig_identity_evaluation :
  let θ1 := 70 * Real.pi / 180 -- angle 70 degrees in radians
  let θ2 := 10 * Real.pi / 180 -- angle 10 degrees in radians
  let θ3 := 20 * Real.pi / 180 -- angle 20 degrees in radians
  (Real.tan θ1 * Real.cos θ2 * (Real.sqrt 3 * Real.tan θ3 - 1) = -1) := 
by 
  sorry

end trig_identity_evaluation_l2252_225257


namespace sum_of_digits_l2252_225201

theorem sum_of_digits :
  ∃ (E M V Y : ℕ), 
    (E ≠ M ∧ E ≠ V ∧ E ≠ Y ∧ M ≠ V ∧ M ≠ Y ∧ V ≠ Y) ∧
    (10 * Y + E) * (10 * M + E) = 111 * V ∧ 
    1 ≤ V ∧ V ≤ 9 ∧ 
    E + M + V + Y = 21 :=
by 
  sorry

end sum_of_digits_l2252_225201


namespace sum_of_coefficients_l2252_225279

theorem sum_of_coefficients (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 : ℝ) :
  (∀ x, (x^2 + 1) * (x - 2)^9 = a + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 +
        a5 * (x - 1)^5 + a6 * (x - 1)^6 + a7 * (x - 1)^7 + a8 * (x - 1)^8 + a9 * (x - 1)^9 + a10 * (x - 1)^10 + a11 * (x - 1)^11) →
  a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 = 2 := 
sorry

end sum_of_coefficients_l2252_225279


namespace min_value_frac_l2252_225238

theorem min_value_frac (x y a b c d : ℝ) (hx : 0 < x) (hy : 0 < y)
  (harith : x + y = a + b) (hgeo : x * y = c * d) : (a + b) ^ 2 / (c * d) ≥ 4 := 
by sorry

end min_value_frac_l2252_225238


namespace hunting_dogs_theorem_l2252_225280

noncomputable def hunting_dogs_problem : Prop :=
  ∃ (courtiers : Finset (Finset (Fin 100))) (h1 : courtiers.card = 100),
  ∀ (c1 c2 : Finset (Fin 100)), c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card ≥ 2 → 
  ∃ (c₁ c₂ : Finset (Fin 100)), c₁ ∈ courtiers ∧ c₂ ∈ courtiers ∧ c₁ = c₂

theorem hunting_dogs_theorem : hunting_dogs_problem :=
sorry

end hunting_dogs_theorem_l2252_225280


namespace trig_identity_on_line_l2252_225247

theorem trig_identity_on_line (α : ℝ) (h : Real.tan α = 2) :
  Real.sin α ^ 2 - Real.cos α ^ 2 + Real.sin α * Real.cos α = 1 :=
sorry

end trig_identity_on_line_l2252_225247


namespace smallest_integer_min_value_l2252_225296

theorem smallest_integer_min_value :
  ∃ (A B C D : ℕ), 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ 
  B ≠ C ∧ B ≠ D ∧ 
  C ≠ D ∧ 
  (A + B + C + D) = 288 ∧ 
  D = 90 ∧ 
  (A = 21) := 
sorry

end smallest_integer_min_value_l2252_225296


namespace inequality_proof_l2252_225236

theorem inequality_proof (a b : ℝ) (h₁ : a ≥ b) (h₂ : b > 0) : 
  2 * a ^ 3 - b ^ 3 ≥ 2 * a * b ^ 2 - a ^ 2 * b := 
by
  sorry

end inequality_proof_l2252_225236


namespace additional_votes_in_revote_l2252_225286

theorem additional_votes_in_revote (a b a' b' n : ℕ) :
  a + b = 300 →
  b - a = n →
  a' - b' = 3 * n →
  a' + b' = 300 →
  a' = (7 * b) / 6 →
  a' - a = 55 :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end additional_votes_in_revote_l2252_225286


namespace moving_circle_fixed_point_l2252_225271

def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

def tangent_line (c : ℝ × ℝ) (r : ℝ) : Prop :=
  abs (c.1 + 1) = r

theorem moving_circle_fixed_point :
  ∀ (c : ℝ × ℝ) (r : ℝ),
    parabola c →
    tangent_line c r →
    (1, 0) ∈ {p : ℝ × ℝ | dist c p = r} :=
by
  intro c r hc ht
  sorry

end moving_circle_fixed_point_l2252_225271


namespace minimum_rooms_to_accommodate_fans_l2252_225231

/-
Each hotel room can accommodate no more than 3 people. The hotel manager knows 
that a group of 100 football fans, who support three different teams, will soon 
arrive. A room can only house either men or women; and fans of different teams 
cannot be housed together. Prove that at least 37 rooms are needed to accommodate 
all the fans.
-/

noncomputable def minimum_rooms_needed (total_fans : ℕ) (fans_per_room : ℕ) : ℕ :=
  if h : fans_per_room > 0 then (total_fans + fans_per_room - 1) / fans_per_room else 0

theorem minimum_rooms_to_accommodate_fans :
  ∀ (total_fans : ℕ) (fans_per_room : ℕ)
    (num_teams : ℕ) (num_genders : ℕ),
  total_fans = 100 →
  fans_per_room = 3 →
  num_teams = 3 →
  num_genders = 2 →
  (minimum_rooms_needed total_fans fans_per_room) ≥ 37 :=
by
  intros total_fans fans_per_room num_teams num_genders h_total h_per_room h_teams h_genders
  -- Proof goes here
  sorry

end minimum_rooms_to_accommodate_fans_l2252_225231


namespace not_divisible_by_3_l2252_225278

theorem not_divisible_by_3 (n : ℤ) : (n^2 + 1) % 3 ≠ 0 := by
  sorry

end not_divisible_by_3_l2252_225278


namespace fresh_grapes_water_content_l2252_225208

theorem fresh_grapes_water_content:
  ∀ (P : ℝ), 
  (∀ (x y : ℝ), P = x) → 
  (∃ (fresh_grapes dry_grapes : ℝ), fresh_grapes = 25 ∧ dry_grapes = 3.125 ∧ 
  (100 - P) / 100 * fresh_grapes = 0.8 * dry_grapes ) → 
  P = 90 :=
by 
  sorry

end fresh_grapes_water_content_l2252_225208


namespace total_bottle_caps_l2252_225289

-- Define the conditions
def bottle_caps_per_child : ℕ := 5
def number_of_children : ℕ := 9

-- Define the main statement to be proven
theorem total_bottle_caps : bottle_caps_per_child * number_of_children = 45 :=
by sorry

end total_bottle_caps_l2252_225289


namespace kelly_held_longest_l2252_225258

variable (K : ℕ)

-- Conditions
def Brittany_held (K : ℕ) : ℕ := K - 20
def Buffy_held : ℕ := 120

-- Theorem to prove
theorem kelly_held_longest (h : K > Buffy_held) : K > 120 :=
by sorry

end kelly_held_longest_l2252_225258


namespace segment_order_l2252_225250

def angle_sum_triangle (A B C : ℝ) : Prop := A + B + C = 180

def order_segments (angles_ABC angles_XYZ angles_ZWX : ℝ → ℝ → ℝ) : Prop :=
  let A := angles_ABC 55 60
  let B := angles_XYZ 95 70
  ∀ (XY YZ ZX WX WZ: ℝ), 
    YZ < ZX ∧ ZX < XY ∧ ZX < WZ ∧ WZ < WX

theorem segment_order:
  ∀ (A B C X Y Z W : Type)
  (XYZ_ang ZWX_ang : ℝ), 
  angle_sum_triangle 55 60 65 →
  angle_sum_triangle 95 70 15 →
  order_segments (angles_ABC) (angles_XYZ) (angles_ZWX)
:= sorry

end segment_order_l2252_225250


namespace cost_price_computer_table_l2252_225222

theorem cost_price_computer_table (S : ℝ) (C : ℝ) (h1 : S = C * 1.15) (h2 : S = 5750) : C = 5000 :=
by
  sorry

end cost_price_computer_table_l2252_225222


namespace complex_number_quadrant_l2252_225235

def inSecondQuadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_number_quadrant : inSecondQuadrant (i / (1 - i)) :=
by
  sorry

end complex_number_quadrant_l2252_225235


namespace geometric_series_sum_l2252_225244

-- Define the geometric series
def geometricSeries (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r ^ n) / (1 - r)

-- Define the conditions
def a : ℚ := 1 / 4
def r : ℚ := 1 / 4
def n : ℕ := 5

-- Define the sum of the first n terms using the provided formula
def S_n := geometricSeries a r n

-- State the theorem: the sum S_5 equals the given answer
theorem geometric_series_sum :
  S_n = 1023 / 3072 :=
by
  sorry

end geometric_series_sum_l2252_225244


namespace imaginary_unit_power_l2252_225283

theorem imaginary_unit_power (i : ℂ) (n : ℕ) (h_i : i^2 = -1) : ∃ (n : ℕ), i^n = -1 :=
by
  use 6
  have h1 : i^4 = 1 := by sorry  -- Need to show i^4 = 1
  have h2 : i^6 = -1 := by sorry  -- Use i^4 and additional steps to show i^6 = -1
  exact h2

end imaginary_unit_power_l2252_225283


namespace total_balloons_l2252_225251

-- Define the number of balloons each person has
def joan_balloons : ℕ := 40
def melanie_balloons : ℕ := 41

-- State the theorem about the total number of balloons
theorem total_balloons : joan_balloons + melanie_balloons = 81 :=
by
  sorry

end total_balloons_l2252_225251


namespace min_value_ax_over_rR_l2252_225224

theorem min_value_ax_over_rR (a b c r R : ℝ) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_le_b : a ≤ b) (h_le_c : a ≤ c) (h_inradius : ∀ (a b c : ℝ), r = 2 * area / (a + b + c))
  (h_circumradius : ∀ (a b c : ℝ), R = (a * b * c) / (4 * area))
  (x : ℝ) (h_x : x = (b + c - a) / 2) (area : ℝ) :
  (a * x / (r * R)) ≥ 3 :=
sorry

end min_value_ax_over_rR_l2252_225224


namespace chloe_and_friends_points_l2252_225230

-- Define the conditions as Lean definitions and then state the theorem to be proven.

def total_pounds_recycled : ℕ := 28 + 2

def pounds_per_point : ℕ := 6

def points_earned (total_pounds : ℕ) (pounds_per_point : ℕ) : ℕ :=
  total_pounds / pounds_per_point

theorem chloe_and_friends_points :
  points_earned total_pounds_recycled pounds_per_point = 5 :=
by
  sorry

end chloe_and_friends_points_l2252_225230


namespace find_compound_interest_principal_l2252_225226

noncomputable def SI (P R T: ℝ) := (P * R * T) / 100
noncomputable def CI (P R T: ℝ) := P * (1 + R / 100)^T - P

theorem find_compound_interest_principal :
  let SI_amount := 3500.000000000004
  let SI_years := 2
  let SI_rate := 6
  let CI_years := 2
  let CI_rate := 10
  let SI_value := SI SI_amount SI_rate SI_years
  let P := 4000
  (SI_value = (CI P CI_rate CI_years) / 2) →
  P = 4000 :=
by
  intros
  sorry

end find_compound_interest_principal_l2252_225226


namespace not_true_n_gt_24_l2252_225248

theorem not_true_n_gt_24 (n : ℕ) (h : 1/3 + 1/4 + 1/6 + 1/n = 1) : n ≤ 24 := 
by
  -- Placeholder for the proof
  sorry

end not_true_n_gt_24_l2252_225248


namespace ted_speed_l2252_225295

variables (T F : ℝ)

-- Ted runs two-thirds as fast as Frank
def condition1 : Prop := T = (2 / 3) * F

-- In two hours, Frank runs eight miles farther than Ted
def condition2 : Prop := 2 * F = 2 * T + 8

-- Prove that Ted runs at a speed of 8 mph
theorem ted_speed (h1 : condition1 T F) (h2 : condition2 T F) : T = 8 :=
by
  sorry

end ted_speed_l2252_225295


namespace simplify_fraction_l2252_225290

theorem simplify_fraction (a : ℝ) (h : a ≠ 2) : 
  (a^2 / (a - 2) - (4 * a - 4) / (a - 2)) = a - 2 :=
  sorry

end simplify_fraction_l2252_225290


namespace principal_amount_l2252_225293

theorem principal_amount (P : ℕ) (R : ℕ) (T : ℕ) (SI : ℕ) 
  (h1 : R = 12)
  (h2 : T = 10)
  (h3 : SI = 1500) 
  (h4 : SI = (P * R * T) / 100) : P = 1250 :=
by sorry

end principal_amount_l2252_225293


namespace minimum_value_of_m_plus_n_l2252_225214

-- Define the conditions and goals as a Lean 4 statement with a proof goal.
theorem minimum_value_of_m_plus_n (m n : ℝ) (h : m * n > 0) (hA : m + n = 3 * m * n) : m + n = 4 / 3 :=
sorry

end minimum_value_of_m_plus_n_l2252_225214


namespace fly_distance_from_ceiling_l2252_225215

/-- 
Assume a room where two walls and the ceiling meet at right angles at point P.
Let point P be the origin (0, 0, 0). 
Let the fly's position be (2, 7, z), where z is the distance from the ceiling.
Given the fly is 2 meters from one wall, 7 meters from the other wall, 
and 10 meters from point P, prove that the fly is at a distance sqrt(47) from the ceiling.
-/
theorem fly_distance_from_ceiling : 
  ∀ (z : ℝ), 
  (2^2 + 7^2 + z^2 = 10^2) → 
  z = Real.sqrt 47 :=
by 
  intro z h
  sorry

end fly_distance_from_ceiling_l2252_225215


namespace raghu_investment_l2252_225204

theorem raghu_investment (R T V : ℝ) (h1 : T = 0.9 * R) (h2 : V = 1.1 * T) (h3 : R + T + V = 5780) : R = 2000 :=
by
  sorry

end raghu_investment_l2252_225204


namespace lunch_to_read_ratio_l2252_225282

theorem lunch_to_read_ratio 
  (total_pages : ℕ) (pages_per_hour : ℕ) (lunch_hours : ℕ)
  (h₁ : total_pages = 4000)
  (h₂ : pages_per_hour = 250)
  (h₃ : lunch_hours = 4) :
  lunch_hours / (total_pages / pages_per_hour) = 1 / 4 := by
  sorry

end lunch_to_read_ratio_l2252_225282


namespace calculate_expression_l2252_225275

theorem calculate_expression :
  let a := 2^4
  let b := 2^2
  let c := 2^3
  (a^2 / b^3) * c^3 = 2048 :=
by
  sorry -- Proof is omitted as per instructions

end calculate_expression_l2252_225275
