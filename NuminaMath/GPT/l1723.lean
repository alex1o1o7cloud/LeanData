import Mathlib

namespace b_2030_is_5_l1723_172395

def seq (b : ℕ → ℚ) : Prop :=
  b 1 = 4 ∧ b 2 = 5 ∧ ∀ n ≥ 3, b (n + 1) = b n / b (n - 1)

theorem b_2030_is_5 (b : ℕ → ℚ) (h : seq b) : 
  b 2030 = 5 :=
sorry

end b_2030_is_5_l1723_172395


namespace find_a_l1723_172315

-- Define the conditions given in the problem
def binomial_term (r : ℕ) (a : ℝ) : ℝ :=
  Nat.choose 7 r * 2^(7-r) * (-a)^r

def coefficient_condition (a : ℝ) : Prop :=
  binomial_term 5 a = 84

-- The theorem stating the problem's solution
theorem find_a (a : ℝ) (h : coefficient_condition a) : a = -1 :=
  sorry

end find_a_l1723_172315


namespace first_year_exceeds_threshold_l1723_172300

def P (n : ℕ) : ℝ := 40000 * (1 + 0.2) ^ n
def exceeds_threshold (n : ℕ) : Prop := P n > 120000

theorem first_year_exceeds_threshold : ∃ n : ℕ, exceeds_threshold n ∧ 2013 + n = 2020 := 
by
  sorry

end first_year_exceeds_threshold_l1723_172300


namespace full_price_ticket_revenue_l1723_172354

-- Given conditions
variable {f d p : ℕ}
variable (h1 : f + d = 160)
variable (h2 : f * p + d * (2 * p / 3) = 2800)

-- Goal: Prove the full-price ticket revenue is 1680.
theorem full_price_ticket_revenue : f * p = 1680 :=
sorry

end full_price_ticket_revenue_l1723_172354


namespace total_opponent_scores_is_45_l1723_172312

-- Definitions based on the conditions
def games : Fin 10 := Fin.mk 10 sorry

def team_scores : Fin 10 → ℕ
| ⟨0, _⟩ => 1
| ⟨1, _⟩ => 2
| ⟨2, _⟩ => 3
| ⟨3, _⟩ => 4
| ⟨4, _⟩ => 5
| ⟨5, _⟩ => 6
| ⟨6, _⟩ => 7
| ⟨7, _⟩ => 8
| ⟨8, _⟩ => 9
| ⟨9, _⟩ => 10
| _ => 0  -- Placeholder for out-of-bounds, should not be used

def lost_games : Fin 5 → ℕ
| ⟨0, _⟩ => 1
| ⟨1, _⟩ => 3
| ⟨2, _⟩ => 5
| ⟨3, _⟩ => 7
| ⟨4, _⟩ => 9

def opponent_score_lost : ℕ → ℕ := λ s => s + 1

def won_games : Fin 5 → ℕ
| ⟨0, _⟩ => 2
| ⟨1, _⟩ => 4
| ⟨2, _⟩ => 6
| ⟨3, _⟩ => 8
| ⟨4, _⟩ => 10

def opponent_score_won : ℕ → ℕ := λ s => s / 2

-- Main statement to prove total opponent scores
theorem total_opponent_scores_is_45 :
  let total_lost_scores := (lost_games 0 :: lost_games 1 :: lost_games 2 :: lost_games 3 :: lost_games 4 :: []).map opponent_score_lost
  let total_won_scores  := (won_games 0 :: won_games 1 :: won_games 2 :: won_games 3 :: won_games 4 :: []).map opponent_score_won
  total_lost_scores.sum + total_won_scores.sum = 45 :=
by sorry

end total_opponent_scores_is_45_l1723_172312


namespace problem_statement_l1723_172304

theorem problem_statement :
  ¬(∀ n : ℤ, n ≥ 0 → n = 0) ∧
  ¬(∀ q : ℚ, q ≠ 0 → q > 0 ∨ q < 0) ∧
  ¬(∀ a b : ℝ, abs a = abs b → a = b) ∧
  (∀ a : ℝ, abs a = abs (-a)) :=
by
  sorry

end problem_statement_l1723_172304


namespace minimum_production_quantity_l1723_172333

-- Define the total cost function
def total_cost (x : ℝ) : ℝ := 3000 + 20 * x - 0.1 * x^2

-- Define the revenue function given the selling price per unit
def revenue (x : ℝ) : ℝ := 25 * x

-- Define the interval for x
def x_range (x : ℝ) : Prop := 0 < x ∧ x < 240

-- State the minimum production quantity required to avoid a loss
theorem minimum_production_quantity (x : ℝ) (h : x_range x) : 150 <= x :=
by
  -- Sorry replaces the detailed proof steps
  sorry

end minimum_production_quantity_l1723_172333


namespace lottery_ticket_might_win_l1723_172319

theorem lottery_ticket_might_win (p_win : ℝ) (h : p_win = 0.01) : 
  (∃ (n : ℕ), n = 1 ∧ 0 < p_win ∧ p_win < 1) :=
by 
  sorry

end lottery_ticket_might_win_l1723_172319


namespace pipe_A_fills_tank_in_28_hours_l1723_172384

variable (A B C : ℝ)
-- Conditions
axiom h1 : C = 2 * B
axiom h2 : B = 2 * A
axiom h3 : A + B + C = 1 / 4

theorem pipe_A_fills_tank_in_28_hours : 1 / A = 28 := by
  -- proof omitted for the exercise
  sorry

end pipe_A_fills_tank_in_28_hours_l1723_172384


namespace points_on_line_eqdist_quadrants_l1723_172378

theorem points_on_line_eqdist_quadrants :
  ∀ (x y : ℝ), 4 * x - 3 * y = 12 ∧ |x| = |y| → 
  (x > 0 ∧ y > 0 ∨ x > 0 ∧ y < 0) :=
by
  sorry

end points_on_line_eqdist_quadrants_l1723_172378


namespace pears_count_l1723_172389

theorem pears_count (A F P : ℕ)
  (hA : A = 12)
  (hF : F = 4 * 12 + 3)
  (hP : P = F - A) :
  P = 39 := by
  sorry

end pears_count_l1723_172389


namespace intersection_A_B_l1723_172356

/-- Define the set A -/
def A : Set ℝ := { x | ∃ y, y = Real.log (2 - x) }

/-- Define the set B -/
def B : Set ℝ := { y | ∃ x, y = Real.sqrt x }

/-- Define the intersection of A and B and prove that it equals [0, 2) -/
theorem intersection_A_B : (A ∩ B) = { x | 0 ≤ x ∧ x < 2 } :=
by
  sorry

end intersection_A_B_l1723_172356


namespace irrational_root_exists_l1723_172375

theorem irrational_root_exists 
  (a b c d : ℤ)
  (h_poly : ∀ x : ℚ, a * x^3 + b * x^2 + c * x + d ≠ 0) 
  (h_odd : a * d % 2 = 1) 
  (h_even : b * c % 2 = 0) : 
  ∃ x : ℚ, ¬ ∃ y : ℚ, y ≠ x ∧ y ≠ x ∧ a * x^3 + b * x^2 + c * x + d = 0 :=
sorry

end irrational_root_exists_l1723_172375


namespace no_nat_pairs_satisfy_eq_l1723_172396

theorem no_nat_pairs_satisfy_eq (a b : ℕ) : ¬ (2019 * a ^ 2018 = 2017 + b ^ 2016) :=
sorry

end no_nat_pairs_satisfy_eq_l1723_172396


namespace pairs_of_managers_refusing_l1723_172372

theorem pairs_of_managers_refusing (h_comb : (Nat.choose 8 4) = 70) (h_restriction : 55 = 70 - n * (Nat.choose 6 2)) : n = 1 :=
by
  have h1 : Nat.choose 8 4 = 70 := h_comb
  have h2 : Nat.choose 6 2 = 15 := by sorry -- skipped calculation for (6 choose 2), which is 15
  have h3 : 55 = 70 - n * 15 := h_restriction
  sorry -- proof steps to show n = 1

end pairs_of_managers_refusing_l1723_172372


namespace problem_statement_l1723_172346

theorem problem_statement (x y : ℝ) (h1 : 3 = 0.15 * x) (h2 : 3 = 0.25 * y) : x - y = 8 :=
by
  sorry

end problem_statement_l1723_172346


namespace dan_baseball_cards_total_l1723_172379

-- Define the initial conditions
def initial_baseball_cards : Nat := 97
def torn_baseball_cards : Nat := 8
def sam_bought_cards : Nat := 15
def alex_bought_fraction : Nat := 4
def gift_cards : Nat := 6

-- Define the number of cards    
def non_torn_baseball_cards : Nat := initial_baseball_cards - torn_baseball_cards
def remaining_after_sam : Nat := non_torn_baseball_cards - sam_bought_cards
def remaining_after_alex : Nat := remaining_after_sam - remaining_after_sam / alex_bought_fraction
def final_baseball_cards : Nat := remaining_after_alex + gift_cards

-- The theorem to prove 
theorem dan_baseball_cards_total : final_baseball_cards = 62 := by
  sorry

end dan_baseball_cards_total_l1723_172379


namespace solution_for_x_l1723_172381

theorem solution_for_x (t : ℤ) :
  ∃ x : ℤ, (∃ (k1 k2 k3 : ℤ), 
    (2 * x + 1 = 3 * k1) ∧ (3 * x + 1 = 4 * k2) ∧ (4 * x + 1 = 5 * k3)) :=
  sorry

end solution_for_x_l1723_172381


namespace num_koi_fish_after_3_weeks_l1723_172339

noncomputable def total_days : ℕ := 3 * 7

noncomputable def koi_fish_added_per_day : ℕ := 2
noncomputable def goldfish_added_per_day : ℕ := 5

noncomputable def total_fish_initial : ℕ := 280
noncomputable def goldfish_final : ℕ := 200

noncomputable def koi_fish_total_after_3_weeks : ℕ :=
  let koi_fish_added := koi_fish_added_per_day * total_days
  let goldfish_added := goldfish_added_per_day * total_days
  let goldfish_initial := goldfish_final - goldfish_added
  let koi_fish_initial := total_fish_initial - goldfish_initial
  koi_fish_initial + koi_fish_added

theorem num_koi_fish_after_3_weeks :
  koi_fish_total_after_3_weeks = 227 := by
  sorry

end num_koi_fish_after_3_weeks_l1723_172339


namespace a_8_eq_5_l1723_172302

noncomputable def S (n : ℕ) : ℕ := sorry
noncomputable def a (n : ℕ) : ℕ := sorry

axiom S_eq : ∀ n m : ℕ, S n + S m = S (n + m)
axiom a1 : a 1 = 5
axiom Sn1 : ∀ n : ℕ, S (n + 1) = S n + 5

theorem a_8_eq_5 : a 8 = 5 :=
sorry

end a_8_eq_5_l1723_172302


namespace evaluate_polynomial_at_2_l1723_172387

def polynomial (x : ℝ) := x^2 + 5*x - 14

theorem evaluate_polynomial_at_2 : polynomial 2 = 0 := by
  sorry

end evaluate_polynomial_at_2_l1723_172387


namespace spadesuit_example_l1723_172326

-- Define the operation spadesuit
def spadesuit (a b : ℤ) : ℤ := abs (a - b)

-- Define the specific instance to prove
theorem spadesuit_example : spadesuit 2 (spadesuit 4 7) = 1 :=
by
  sorry

end spadesuit_example_l1723_172326


namespace perimeter_result_l1723_172362

-- Define the side length of the square
def side_length : ℕ := 100

-- Define the dimensions of the rectangle
def rectangle_dim1 : ℕ := side_length
def rectangle_dim2 : ℕ := side_length / 2

-- Perimeter calculation based on the arrangement
def perimeter : ℕ :=
  3 * rectangle_dim1 + 4 * rectangle_dim2

-- The statement of the problem
theorem perimeter_result :
  perimeter = 500 :=
by
  sorry

end perimeter_result_l1723_172362


namespace employed_males_percentage_l1723_172305

theorem employed_males_percentage (P : ℕ) (H1: P > 0)
    (employed_pct : ℝ) (female_pct : ℝ)
    (H_employed_pct : employed_pct = 0.64)
    (H_female_pct : female_pct = 0.140625) :
    (0.859375 * employed_pct * 100) = 54.96 :=
by
  sorry

end employed_males_percentage_l1723_172305


namespace parabola_decreasing_m_geq_neg2_l1723_172367

theorem parabola_decreasing_m_geq_neg2 (m : ℝ) :
  (∀ x ≥ 2, ∃ y, y = -5 * (x + m)^2 - 3 ∧ (∀ x1 y1, x1 ≥ 2 → y1 = -5 * (x1 + m)^2 - 3 → y1 ≤ y)) →
  m ≥ -2 := 
by
  intro h
  sorry

end parabola_decreasing_m_geq_neg2_l1723_172367


namespace delta_value_l1723_172359

theorem delta_value : ∃ Δ : ℤ, 4 * (-3) = Δ + 5 ∧ Δ = -17 := 
by
  use -17
  sorry

end delta_value_l1723_172359


namespace program_arrangements_l1723_172308

/-- Given 5 programs, if A, B, and C appear in a specific order, then the number of different
    arrangements is 20. -/
theorem program_arrangements (A B C A_order : ℕ) : 
  (A + B + C + A_order = 5) → 
  (A_order = 3) → 
  (B = 1) → 
  (C = 1) → 
  (A = 1) → 
  (A * B * C * A_order = 1) :=
  by sorry

end program_arrangements_l1723_172308


namespace mary_can_keep_warm_l1723_172349

def sticks_from_chairs (n_c : ℕ) (c_1 : ℕ) : ℕ := n_c * c_1
def sticks_from_tables (n_t : ℕ) (t_1 : ℕ) : ℕ := n_t * t_1
def sticks_from_cabinets (n_cb : ℕ) (cb_1 : ℕ) : ℕ := n_cb * cb_1
def sticks_from_stools (n_s : ℕ) (s_1 : ℕ) : ℕ := n_s * s_1

def total_sticks (n_c n_t n_cb n_s c_1 t_1 cb_1 s_1 : ℕ) : ℕ :=
  sticks_from_chairs n_c c_1
  + sticks_from_tables n_t t_1 
  + sticks_from_cabinets n_cb cb_1 
  + sticks_from_stools n_s s_1

noncomputable def hours (total_sticks r : ℕ) : ℕ :=
  total_sticks / r

theorem mary_can_keep_warm (n_c n_t n_cb n_s : ℕ) (c_1 t_1 cb_1 s_1 r : ℕ) :
  n_c = 25 → n_t = 12 → n_cb = 5 → n_s = 8 → c_1 = 8 → t_1 = 12 → cb_1 = 16 → s_1 = 3 → r = 7 →
  hours (total_sticks n_c n_t n_cb n_s c_1 t_1 cb_1 s_1) r = 64 :=
by
  intros h_nc h_nt h_ncb h_ns h_c1 h_t1 h_cb1 h_s1 h_r
  sorry

end mary_can_keep_warm_l1723_172349


namespace num_adults_l1723_172337

-- Definitions of the conditions
def num_children : Nat := 11
def child_ticket_cost : Nat := 4
def adult_ticket_cost : Nat := 8
def total_cost : Nat := 124

-- The proof problem statement
theorem num_adults (A : Nat) 
  (h1 : total_cost = num_children * child_ticket_cost + A * adult_ticket_cost) : 
  A = 10 := 
by
  sorry

end num_adults_l1723_172337


namespace overall_rate_of_profit_is_25_percent_l1723_172360

def cost_price_A : ℕ := 50
def selling_price_A : ℕ := 70
def cost_price_B : ℕ := 80
def selling_price_B : ℕ := 100
def cost_price_C : ℕ := 150
def selling_price_C : ℕ := 180

def profit (sp cp : ℕ) : ℕ := sp - cp

def total_cost_price : ℕ := cost_price_A + cost_price_B + cost_price_C
def total_selling_price : ℕ := selling_price_A + selling_price_B + selling_price_C
def total_profit : ℕ := profit selling_price_A cost_price_A +
                        profit selling_price_B cost_price_B +
                        profit selling_price_C cost_price_C

def overall_rate_of_profit : ℚ := (total_profit : ℚ) / (total_cost_price : ℚ) * 100

theorem overall_rate_of_profit_is_25_percent :
  overall_rate_of_profit = 25 :=
by sorry

end overall_rate_of_profit_is_25_percent_l1723_172360


namespace functional_eq_solution_l1723_172358

theorem functional_eq_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (2 * x + f y) = x + y + f x) :
  ∀ x : ℝ, f x = x :=
sorry

end functional_eq_solution_l1723_172358


namespace smallest_n_l1723_172363

theorem smallest_n (n : ℕ) (h : ↑n > 0 ∧ (Real.sqrt (↑n) - Real.sqrt (↑n - 1)) < 0.02) : n = 626 := 
by
  sorry

end smallest_n_l1723_172363


namespace measure_of_angle_4_l1723_172380

theorem measure_of_angle_4 
  (angle1 angle2 angle3 : ℝ)
  (h1 : angle1 = 100)
  (h2 : angle2 = 60)
  (h3 : angle3 = 90)
  (h_sum : angle1 + angle2 + angle3 + angle4 = 360) : 
  angle4 = 110 :=
by
  sorry

end measure_of_angle_4_l1723_172380


namespace brick_length_is_20_cm_l1723_172352

-- Define the conditions given in the problem
def courtyard_length : ℝ := 25
def courtyard_width : ℝ := 16
def num_bricks : ℕ := 20000
def brick_width_cm : ℝ := 10
def total_area_cm2 : ℝ := 4000000

-- Define the goal to prove that the length of each brick is 20 cm
theorem brick_length_is_20_cm :
  (total_area_cm2 = num_bricks * (brick_width_cm * length)) → (length = 20) :=
by
  -- Assume the given conditions
  sorry

end brick_length_is_20_cm_l1723_172352


namespace determine_m_l1723_172353

variable (A B : Set ℝ)
variable (m : ℝ)

theorem determine_m (hA : A = {-1, 3, m}) (hB : B = {3, 4}) (h_inter : B ∩ A = B) : m = 4 :=
sorry

end determine_m_l1723_172353


namespace find_k_value_l1723_172311

theorem find_k_value (k : ℝ) : (∀ (x y : ℝ), (x = 2 ∧ y = 5) → y = k * x + 3) → k = 1 := 
by 
  intro h
  have h1 := h 2 5 ⟨rfl, rfl⟩
  linarith

end find_k_value_l1723_172311


namespace athletes_leave_rate_l1723_172327

theorem athletes_leave_rate (R : ℝ) (h : 300 - 4 * R + 105 = 307) : R = 24.5 :=
  sorry

end athletes_leave_rate_l1723_172327


namespace value_of_a_minus_3_l1723_172388

variable {α : Type*} [Field α] (f : α → α) (a : α)

-- Conditions
variable (h_invertible : Function.Injective f)
variable (h_fa : f a = 3)
variable (h_f3 : f 3 = 6)

-- Statement to prove
theorem value_of_a_minus_3 : a - 3 = -2 :=
by
  sorry

end value_of_a_minus_3_l1723_172388


namespace usual_time_is_120_l1723_172399

variable (S T : ℕ) (h1 : 0 < S) (h2 : 0 < T)
variable (h3 : (4 : ℚ) / 3 = 1 + (40 : ℚ) / T)

theorem usual_time_is_120 : T = 120 := by
  sorry

end usual_time_is_120_l1723_172399


namespace price_per_working_game_l1723_172382

theorem price_per_working_game 
  (total_games : ℕ) (non_working_games : ℕ) (total_earnings : ℕ)
  (h1 : total_games = 16) (h2 : non_working_games = 8) (h3 : total_earnings = 56) :
  total_earnings / (total_games - non_working_games) = 7 :=
by {
  sorry
}

end price_per_working_game_l1723_172382


namespace sequence_an_formula_l1723_172320

theorem sequence_an_formula (a : ℕ → ℕ) (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * a n + 1) :
  ∀ n : ℕ, n ≥ 1 → a n = 2^n - 1 :=
by
  sorry

end sequence_an_formula_l1723_172320


namespace evaluate_expression_l1723_172347

theorem evaluate_expression : (3 + 2) * (3^2 + 2^2) * (3^4 + 2^4) = 6255 := sorry

end evaluate_expression_l1723_172347


namespace greatest_prime_factor_is_5_l1723_172306

-- Define the expression
def expr : Nat := (3^8 + 9^5)

-- State the theorem
theorem greatest_prime_factor_is_5 : ∃ p : Nat, Prime p ∧ p = 5 ∧ ∀ q : Nat, Prime q ∧ q ∣ expr → q ≤ 5 := by
  sorry

end greatest_prime_factor_is_5_l1723_172306


namespace sqrt_square_eq_self_l1723_172318

variable (a : ℝ)

theorem sqrt_square_eq_self (h : a > 0) : Real.sqrt (a ^ 2) = a :=
  sorry

end sqrt_square_eq_self_l1723_172318


namespace jennifer_fish_tank_problem_l1723_172394

theorem jennifer_fish_tank_problem :
  let built_tanks := 3
  let fish_per_built_tank := 15
  let planned_tanks := 3
  let fish_per_planned_tank := 10
  let total_built_fish := built_tanks * fish_per_built_tank
  let total_planned_fish := planned_tanks * fish_per_planned_tank
  let total_fish := total_built_fish + total_planned_fish
  total_fish = 75 := by
    let built_tanks := 3
    let fish_per_built_tank := 15
    let planned_tanks := 3
    let fish_per_planned_tank := 10
    let total_built_fish := built_tanks * fish_per_built_tank
    let total_planned_fish := planned_tanks * fish_per_planned_tank
    let total_fish := total_built_fish + total_planned_fish
    have h₁ : total_built_fish = 45 := by sorry
    have h₂ : total_planned_fish = 30 := by sorry
    have h₃ : total_fish = 75 := by sorry
    exact h₃

end jennifer_fish_tank_problem_l1723_172394


namespace find_x_if_perpendicular_l1723_172331

-- Definitions based on the conditions provided
structure Vector2 := (x : ℚ) (y : ℚ)

def a : Vector2 := ⟨2, 3⟩
def b (x : ℚ) : Vector2 := ⟨x, 4⟩

def dot_product (v1 v2 : Vector2) : ℚ := v1.x * v2.x + v1.y * v2.y

theorem find_x_if_perpendicular :
  ∀ x : ℚ, dot_product a (Vector2.mk (a.x - (b x).x) (a.y - (b x).y)) = 0 → x = 1/2 :=
by
  intro x
  intro h
  sorry

end find_x_if_perpendicular_l1723_172331


namespace other_root_of_equation_l1723_172364

theorem other_root_of_equation (c : ℝ) (h : 3^2 - 5 * 3 + c = 0) : 
  ∃ x : ℝ, x ≠ 3 ∧ x^2 - 5 * x + c = 0 ∧ x = 2 := 
by 
  sorry

end other_root_of_equation_l1723_172364


namespace frac_mul_sub_eq_l1723_172390

/-
  Theorem:
  The result of multiplying 2/9 by 4/5 and then subtracting 1/45 is equal to 7/45.
-/
theorem frac_mul_sub_eq :
  (2/9 * 4/5 - 1/45) = 7/45 :=
by
  sorry

end frac_mul_sub_eq_l1723_172390


namespace number_of_oranges_l1723_172322

-- Definitions of the conditions
def peaches : ℕ := 9
def pears : ℕ := 18
def greatest_num_per_basket : ℕ := 3
def num_baskets_peaches := peaches / greatest_num_per_basket
def num_baskets_pears := pears / greatest_num_per_basket
def min_num_baskets := min num_baskets_peaches num_baskets_pears

-- Proof problem statement
theorem number_of_oranges (O : ℕ) (h1 : O % greatest_num_per_basket = 0) 
  (h2 : O / greatest_num_per_basket = min_num_baskets) : 
  O = 9 :=
by {
  sorry
}

end number_of_oranges_l1723_172322


namespace product_xyz_l1723_172355

noncomputable def xyz_value (x y z : ℝ) :=
  x * y * z

theorem product_xyz (x y z : ℝ) (h1 : x + 1 / y = 3) (h2 : y + 1 / z = 3) :
  xyz_value x y z = -1 :=
by
  sorry

end product_xyz_l1723_172355


namespace find_a_l1723_172374

theorem find_a (a b : ℤ) (h : ∀ x, x^2 - x - 1 = 0 → ax^18 + bx^17 + 1 = 0) : a = 1597 :=
sorry

end find_a_l1723_172374


namespace Lizzy_savings_after_loan_l1723_172310

theorem Lizzy_savings_after_loan :
  ∀ (initial_amount loan_amount : ℕ) (interest_percent : ℕ),
  initial_amount = 30 →
  loan_amount = 15 →
  interest_percent = 20 →
  initial_amount - loan_amount + loan_amount + loan_amount * interest_percent / 100 = 33 :=
by
  intros initial_amount loan_amount interest_percent h1 h2 h3
  sorry

end Lizzy_savings_after_loan_l1723_172310


namespace machine_a_produces_50_parts_in_10_minutes_l1723_172357

/-- 
Given that machine A produces parts twice as fast as machine B,
and machine B produces 100 parts in 40 minutes at a constant rate,
prove that machine A produces 50 parts in 10 minutes.
-/
theorem machine_a_produces_50_parts_in_10_minutes :
  (machine_b_rate : ℕ → ℕ) → 
  (machine_a_rate : ℕ → ℕ) →
  (htwice_as_fast: ∀ t, machine_a_rate t = (2 * machine_b_rate t)) →
  (hconstant_rate_b: ∀ t1 t2, t1 * machine_b_rate t2 = 100 * t2 / 40)→
  machine_a_rate 10 = 50 :=
by
  sorry

end machine_a_produces_50_parts_in_10_minutes_l1723_172357


namespace simplify_expression_l1723_172324

theorem simplify_expression (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 2) :
  (a^2 - 6 * a + 9) / (a^2 - 2 * a) / (1 - 1 / (a - 2)) = (a - 3) / a :=
sorry

end simplify_expression_l1723_172324


namespace investment_ratio_l1723_172386

theorem investment_ratio (A B : ℕ) (hA : A = 12000) (hB : B = 12000) 
  (interest_A : ℕ := 11 * A / 100) (interest_B : ℕ := 9 * B / 100) 
  (total_interest : interest_A + interest_B = 2400) :
  A / B = 1 :=
by
  sorry

end investment_ratio_l1723_172386


namespace marble_count_l1723_172348

noncomputable def total_marbles (blue red white: ℕ) : ℕ := blue + red + white

theorem marble_count (W : ℕ) (h_prob : (9 + W) / (6 + 9 + W : ℝ) = 0.7) : 
  total_marbles 6 9 W = 20 :=
by
  sorry

end marble_count_l1723_172348


namespace jordan_run_7_miles_in_112_div_3_minutes_l1723_172328

noncomputable def time_for_steve (distance : ℝ) : ℝ := 36 / 4.5 * distance
noncomputable def jordan_initial_time (steve_time : ℝ) : ℝ := steve_time / 3
noncomputable def jordan_speed (distance time : ℝ) : ℝ := distance / time
noncomputable def adjusted_speed (speed : ℝ) : ℝ := speed * 0.9
noncomputable def running_time (distance speed : ℝ) : ℝ := distance / speed

theorem jordan_run_7_miles_in_112_div_3_minutes : running_time 7 ((jordan_speed 2.5 (jordan_initial_time (time_for_steve 4.5))) * 0.9) = 112 / 3 :=
by
  sorry

end jordan_run_7_miles_in_112_div_3_minutes_l1723_172328


namespace range_of_f_log_gt_zero_l1723_172383

open Real

noncomputable def f (x : ℝ) : ℝ := -- Placeholder function definition
  sorry

theorem range_of_f_log_gt_zero :
  (∀ x, f x = f (-x)) ∧
  (∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y) ∧
  (f (1 / 3) = 0) →
  {x : ℝ | f ((log x) / (log (1 / 8))) > 0} = 
    (Set.Ioo 0 (1 / 2) ∪ Set.Ioi 2) :=
  sorry

end range_of_f_log_gt_zero_l1723_172383


namespace minimum_value_of_z_l1723_172371

def z (x y : ℝ) : ℝ := 3 * x ^ 2 + 4 * y ^ 2 + 12 * x - 8 * y + 3 * x * y + 30

theorem minimum_value_of_z : ∃ (x y : ℝ), z x y = 8 := 
sorry

end minimum_value_of_z_l1723_172371


namespace relation_of_variables_l1723_172342

theorem relation_of_variables (x y z w : ℝ) 
  (h : (x + 2 * y) / (2 * y + 3 * z) = (3 * z + 4 * w) / (4 * w + x)) : 
  (x = 3 * z) ∨ (x + 2 * y + 4 * w + 3 * z = 0) := 
by
  sorry

end relation_of_variables_l1723_172342


namespace octagon_diagonals_l1723_172345

def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem octagon_diagonals : number_of_diagonals 8 = 20 :=
by
  sorry

end octagon_diagonals_l1723_172345


namespace linear_function_intersects_x_axis_at_2_0_l1723_172392

theorem linear_function_intersects_x_axis_at_2_0
  (f : ℝ → ℝ)
  (h : ∀ x, f x = -x + 2) :
  ∃ x, f x = 0 ∧ x = 2 :=
by
  sorry

end linear_function_intersects_x_axis_at_2_0_l1723_172392


namespace problem_statement_b_problem_statement_c_l1723_172369

def clubsuit (x y : ℝ) : ℝ := |x - y + 3|

theorem problem_statement_b :
  ∃ x y : ℝ, 3 * (clubsuit x y) ≠ clubsuit (3 * x + 3) (3 * y + 3) := by
  sorry

theorem problem_statement_c :
  ∃ x : ℝ, clubsuit x (-3) ≠ x := by
  sorry

end problem_statement_b_problem_statement_c_l1723_172369


namespace additional_track_length_l1723_172398

theorem additional_track_length (elevation_gain : ℝ) (orig_grade new_grade : ℝ) (Δ_track : ℝ) :
  elevation_gain = 800 ∧ orig_grade = 0.04 ∧ new_grade = 0.015 ∧ Δ_track = ((elevation_gain / new_grade) - (elevation_gain / orig_grade)) ->
  Δ_track = 33333 :=
by sorry

end additional_track_length_l1723_172398


namespace theater_ticket_area_l1723_172343

theorem theater_ticket_area
  (P width : ℕ)
  (hP : P = 28)
  (hwidth : width = 6)
  (length : ℕ)
  (hlength : 2 * (length + width) = P) :
  length * width = 48 :=
by
  sorry

end theater_ticket_area_l1723_172343


namespace stock_investment_decrease_l1723_172368

theorem stock_investment_decrease (x : ℝ) (d1 d2 : ℝ) (hx : x > 0)
  (increase : x * 1.30 = 1.30 * x) :
  d1 = 20 ∧ d2 = 3.85 → 1.30 * (1 - d1 / 100) * (1 - d2 / 100) = 1 := by
  sorry

end stock_investment_decrease_l1723_172368


namespace find_k_l1723_172391

theorem find_k (k : ℝ) (h : 32 / k = 4) : k = 8 := sorry

end find_k_l1723_172391


namespace only_B_is_linear_system_l1723_172385

def linear_equation (eq : String) : Prop := 
-- Placeholder for the actual definition
sorry 

def system_B_is_linear : Prop :=
  linear_equation "x + y = 2" ∧ linear_equation "x - y = 4"

theorem only_B_is_linear_system 
: (∀ (A B C D : Prop), 
       (A ↔ (linear_equation "3x + 4y = 6" ∧ linear_equation "5z - 6y = 4")) → 
       (B ↔ (linear_equation "x + y = 2" ∧ linear_equation "x - y = 4")) → 
       (C ↔ (linear_equation "x + y = 2" ∧ linear_equation "x^2 - y^2 = 8")) → 
       (D ↔ (linear_equation "x + y = 2" ∧ linear_equation "1/x - 1/y = 1/2")) → 
       (B ∧ ¬A ∧ ¬C ∧ ¬D))
:= 
sorry

end only_B_is_linear_system_l1723_172385


namespace angle_reduction_l1723_172309

theorem angle_reduction (θ : ℝ) : θ = 1303 → ∃ k : ℤ, θ = 360 * k - 137 := 
by  
  intro h 
  use 4 
  simp [h] 
  sorry

end angle_reduction_l1723_172309


namespace taylor_correct_answers_percentage_l1723_172316

theorem taylor_correct_answers_percentage 
  (N : ℕ := 30)
  (alex_correct_alone_percentage : ℝ := 0.85)
  (alex_overall_percentage : ℝ := 0.83)
  (taylor_correct_alone_percentage : ℝ := 0.95)
  (alex_correct_alone : ℕ := 13)
  (alex_correct_total : ℕ := 25)
  (together_correct : ℕ := 12)
  (taylor_correct_alone : ℕ := 14)
  (taylor_correct_total : ℕ := 26) :
  ((taylor_correct_total : ℝ) / (N : ℝ)) * 100 = 87 :=
by
  sorry

end taylor_correct_answers_percentage_l1723_172316


namespace not_multiple_of_121_l1723_172330

theorem not_multiple_of_121 (n : ℤ) : ¬ ∃ k : ℤ, n^2 + 2*n + 12 = 121*k := 
sorry

end not_multiple_of_121_l1723_172330


namespace probability_at_least_one_six_l1723_172303

theorem probability_at_least_one_six :
  let p_six := 1 / 6
  let p_not_six := 5 / 6
  let p_not_six_three_rolls := p_not_six ^ 3
  let p_at_least_one_six := 1 - p_not_six_three_rolls
  p_at_least_one_six = 91 / 216 :=
by
  sorry

end probability_at_least_one_six_l1723_172303


namespace prime_prod_identity_l1723_172317

theorem prime_prod_identity (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : 3 * p + 7 * q = 41) : (p + 1) * (q - 1) = 12 := 
by 
  sorry

end prime_prod_identity_l1723_172317


namespace coexistence_of_properties_l1723_172393

structure Trapezoid (α : Type _) [Field α] :=
(base1 base2 leg1 leg2 : α)
(height : α)

def isIsosceles {α : Type _} [Field α] (T : Trapezoid α) : Prop :=
T.leg1 = T.leg2

def diagonalsPerpendicular {α : Type _} [Field α] (T : Trapezoid α) : Prop :=
sorry  -- Define this property based on coordinate geometry or vector inner products

def heightsEqual {α : Type _} [Field α] (T : Trapezoid α) : Prop :=
T.base1 = T.base2

def midsegmentEqualHeight {α : Type _} [Field α] (T : Trapezoid α) : Prop :=
(T.base1 + T.base2) / 2 = T.height

theorem coexistence_of_properties (α : Type _) [Field α] (T : Trapezoid α) :
  isIsosceles T → heightsEqual T → midsegmentEqualHeight T → True :=
by sorry

end coexistence_of_properties_l1723_172393


namespace red_ball_probability_l1723_172332

-- Definitions based on conditions
def numBallsA := 10
def redBallsA := 5
def greenBallsA := numBallsA - redBallsA

def numBallsBC := 10
def redBallsBC := 7
def greenBallsBC := numBallsBC - redBallsBC

def probSelectContainer := 1 / 3
def probRedBallA := redBallsA / numBallsA
def probRedBallBC := redBallsBC / numBallsBC

-- Theorem statement to be proved
theorem red_ball_probability : (probSelectContainer * probRedBallA) + (probSelectContainer * probRedBallBC) + (probSelectContainer * probRedBallBC) = 4 / 5 := 
sorry

end red_ball_probability_l1723_172332


namespace shara_shells_final_count_l1723_172344

def initial_shells : ℕ := 20
def first_vacation_found : ℕ := 5 * 3 + 6
def first_vacation_lost : ℕ := 4
def second_vacation_found : ℕ := 4 * 2 + 7
def second_vacation_gifted : ℕ := 3
def third_vacation_found : ℕ := 8 + 4 + 3 * 2
def third_vacation_misplaced : ℕ := 5

def total_shells_after_first_vacation : ℕ :=
  initial_shells + first_vacation_found - first_vacation_lost

def total_shells_after_second_vacation : ℕ :=
  total_shells_after_first_vacation + second_vacation_found - second_vacation_gifted

def total_shells_after_third_vacation : ℕ :=
  total_shells_after_second_vacation + third_vacation_found - third_vacation_misplaced

theorem shara_shells_final_count : total_shells_after_third_vacation = 62 := by
  sorry

end shara_shells_final_count_l1723_172344


namespace value_of_a_l1723_172323

theorem value_of_a (a : ℕ) (h : a^3 = 21 * 49 * 45 * 25) : a = 105 := sorry

end value_of_a_l1723_172323


namespace initial_games_count_l1723_172314

-- Definitions used in conditions
def games_given_away : ℕ := 99
def games_left : ℝ := 22.0

-- Theorem statement for the initial number of games
theorem initial_games_count : games_given_away + games_left = 121.0 := by
  sorry

end initial_games_count_l1723_172314


namespace no_HCl_formed_l1723_172334

-- Definitions
def NaCl_moles : Nat := 3
def HNO3_moles : Nat := 3
def HCl_moles : Nat := 0

-- Hypothetical reaction context
-- if the reaction would produce HCl
axiom hypothetical_reaction : (NaCl_moles = 3) → (HNO3_moles = 3) → (∃ h : Nat, h = 3)

-- Proof under normal conditions that no HCl is formed
theorem no_HCl_formed : (NaCl_moles = 3) → (HNO3_moles = 3) → HCl_moles = 0 := by
  intros hNaCl hHNO3
  sorry

end no_HCl_formed_l1723_172334


namespace value_of_a_squared_b_plus_ab_squared_eq_4_l1723_172366

variable (a b : ℝ)
variable (h_a : a = 2 + Real.sqrt 3)
variable (h_b : b = 2 - Real.sqrt 3)

theorem value_of_a_squared_b_plus_ab_squared_eq_4 :
  a^2 * b + a * b^2 = 4 := by
  sorry

end value_of_a_squared_b_plus_ab_squared_eq_4_l1723_172366


namespace mary_and_joan_marbles_l1723_172341

theorem mary_and_joan_marbles : 9 + 3 = 12 :=
by
  rfl

end mary_and_joan_marbles_l1723_172341


namespace book_total_pages_l1723_172340

theorem book_total_pages (n : ℕ) (h1 : 5 * n / 8 - 3 * n / 7 = 33) : n = n :=
by 
  -- We skip the proof as instructed
  sorry

end book_total_pages_l1723_172340


namespace dale_slices_of_toast_l1723_172373

theorem dale_slices_of_toast
  (slice_cost : ℤ) (egg_cost : ℤ)
  (dale_eggs : ℤ) (andrew_slices : ℤ) (andrew_eggs : ℤ)
  (total_cost : ℤ)
  (cost_eq : slice_cost = 1)
  (egg_cost_eq : egg_cost = 3)
  (dale_eggs_eq : dale_eggs = 2)
  (andrew_slices_eq : andrew_slices = 1)
  (andrew_eggs_eq : andrew_eggs = 2)
  (total_cost_eq : total_cost = 15)
  :
  ∃ T : ℤ, (slice_cost * T + egg_cost * dale_eggs) + (slice_cost * andrew_slices + egg_cost * andrew_eggs) = total_cost ∧ T = 2 :=
by
  sorry

end dale_slices_of_toast_l1723_172373


namespace option_a_correct_option_b_incorrect_option_c_incorrect_option_d_incorrect_l1723_172377

theorem option_a_correct (a : ℝ) : 2 * a^2 - 3 * a^2 = - a^2 :=
by
  sorry

theorem option_b_incorrect : (-3)^2 ≠ 6 :=
by
  sorry

theorem option_c_incorrect (a : ℝ) : 6 * a^3 + 4 * a^4 ≠ 10 * a^7 :=
by
  sorry

theorem option_d_incorrect (a b : ℝ) : 3 * a^2 * b - 3 * b^2 * a ≠ 0 :=
by
  sorry

end option_a_correct_option_b_incorrect_option_c_incorrect_option_d_incorrect_l1723_172377


namespace sin_sum_alpha_pi_over_3_l1723_172301

theorem sin_sum_alpha_pi_over_3 (alpha : ℝ) (h1 : Real.cos (alpha + 2/3 * Real.pi) = 4/5) (h2 : -Real.pi/2 < alpha ∧ alpha < 0) :
  Real.sin (alpha + Real.pi/3) + Real.sin alpha = -4 * Real.sqrt 3 / 5 :=
sorry

end sin_sum_alpha_pi_over_3_l1723_172301


namespace inverse_proposition_vertical_angles_false_l1723_172335

-- Define the statement "Vertical angles are equal"
def vertical_angles_equal (α β : ℝ) : Prop :=
  α = β

-- Define the inverse proposition
def inverse_proposition_vertical_angles : Prop :=
  ∀ α β : ℝ, α = β → vertical_angles_equal α β

-- The proof goal
theorem inverse_proposition_vertical_angles_false : ¬inverse_proposition_vertical_angles :=
by
  sorry

end inverse_proposition_vertical_angles_false_l1723_172335


namespace percentage_books_returned_l1723_172325

theorem percentage_books_returned
    (initial_books : ℝ)
    (end_books : ℝ)
    (loaned_books : ℝ)
    (R : ℝ)
    (Percentage_Returned : ℝ) :
    initial_books = 75 →
    end_books = 65 →
    loaned_books = 50.000000000000014 →
    R = (75 - 65) →
    Percentage_Returned = (R / loaned_books) * 100 →
    Percentage_Returned = 20 :=
by
  intros
  sorry

end percentage_books_returned_l1723_172325


namespace find_numbers_l1723_172329

def seven_digit_number (n : ℕ) : Prop := 10^6 ≤ n ∧ n < 10^7

theorem find_numbers (x y : ℕ) (hx: seven_digit_number x) (hy: seven_digit_number y) :
  10^7 * x + y = 3 * x * y → x = 1666667 ∧ y = 3333334 :=
by
  sorry

end find_numbers_l1723_172329


namespace probability_of_first_spade_or_ace_and_second_ace_l1723_172351

theorem probability_of_first_spade_or_ace_and_second_ace :
  let deck_size := 52
  let aces := 4
  let spades := 13
  let non_ace_spades := spades - 1
  let other_aces := aces - 1
  let prob_first_non_ace_spade := non_ace_spades / deck_size
  let prob_first_ace_not_spade := (aces - 1) / deck_size
  let prob_first_ace_spade := 1 / deck_size
  let prob_second_ace_after_non_ace_spade := aces / (deck_size - 1)
  let prob_second_ace_after_ace_not_spade := (aces - 1) / (deck_size - 1)
  let prob_second_ace_after_ace_spade := (aces - 1) / (deck_size - 1)
  ((prob_first_non_ace_spade * prob_second_ace_after_non_ace_spade) +
   (prob_first_ace_not_spade * prob_second_ace_after_ace_not_spade) +
   (prob_first_ace_spade * prob_second_ace_after_ace_spade)) = 5 / 221 :=
by
  let deck_size := 52
  let aces := 4
  let spades := 13
  let non_ace_spades := spades - 1
  let other_aces := aces - 1
  let prob_first_non_ace_spade := non_ace_spades / deck_size
  let prob_first_ace_not_spade := (aces - 1) / deck_size
  let prob_first_ace_spade := 1 / deck_size
  let prob_second_ace_after_non_ace_spade := aces / (deck_size - 1)
  let prob_second_ace_after_ace_not_spade := (aces - 1) / (deck_size - 1)
  let prob_second_ace_after_ace_spade := (aces - 1) / (deck_size - 1)
  sorry

end probability_of_first_spade_or_ace_and_second_ace_l1723_172351


namespace exists_k_not_divisible_l1723_172336

theorem exists_k_not_divisible (a b c n : ℤ) (hn : n ≥ 3) :
  ∃ k : ℤ, ¬(n ∣ (k + a)) ∧ ¬(n ∣ (k + b)) ∧ ¬(n ∣ (k + c)) :=
sorry

end exists_k_not_divisible_l1723_172336


namespace company_p_employees_in_january_l1723_172397

-- Conditions
def employees_in_december (january_employees : ℝ) : ℝ := january_employees + 0.15 * january_employees

theorem company_p_employees_in_january (january_employees : ℝ) :
  employees_in_december january_employees = 490 → january_employees = 426 :=
by
  intro h
  -- The proof steps will be filled here.
  sorry

end company_p_employees_in_january_l1723_172397


namespace sum_of_hundreds_and_tens_digits_of_product_l1723_172350

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def seq_num (a : ℕ) (x : ℕ) := List.foldr (λ _ acc => acc * 1000 + a) 0 (List.range x)

noncomputable def num_a := seq_num 707 101
noncomputable def num_b := seq_num 909 101

noncomputable def product := num_a * num_b

theorem sum_of_hundreds_and_tens_digits_of_product :
  hundreds_digit product + tens_digit product = 8 := by
  sorry

end sum_of_hundreds_and_tens_digits_of_product_l1723_172350


namespace maximum_ratio_l1723_172370

-- Defining the conditions
def two_digit_positive_integer (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Proving the main theorem
theorem maximum_ratio (x y : ℕ) (hx : two_digit_positive_integer x) (hy : two_digit_positive_integer y) (h_sum : x + y = 100) : 
  ∃ m, m = 9 ∧ ∀ r, r = x / y → r ≤ 9 := sorry

end maximum_ratio_l1723_172370


namespace number_subtracted_eq_l1723_172321

theorem number_subtracted_eq (x n : ℤ) (h1 : x + 1315 + 9211 - n = 11901) (h2 : x = 88320) : n = 86945 :=
by
  sorry

end number_subtracted_eq_l1723_172321


namespace nine_digit_divisible_by_11_l1723_172365

theorem nine_digit_divisible_by_11 (m : ℕ) (k : ℤ) (h1 : 8 + 4 + m + 6 + 8 = 26 + m)
(h2 : 5 + 2 + 7 + 1 = 15)
(h3 : 26 + m - 15 = 11 + m)
(h4 : 11 + m = 11 * k) :
m = 0 := by
  sorry

end nine_digit_divisible_by_11_l1723_172365


namespace arrange_decimals_in_order_l1723_172313

theorem arrange_decimals_in_order 
  (a b c d : ℚ) 
  (h₀ : a = 6 / 10) 
  (h₁ : b = 676 / 1000) 
  (h₂ : c = 677 / 1000) 
  (h₃ : d = 67 / 100) : 
  a < d ∧ d < b ∧ b < c := 
by
  sorry

end arrange_decimals_in_order_l1723_172313


namespace dandelion_dog_puffs_l1723_172361

theorem dandelion_dog_puffs :
  let original_puffs := 40
  let mom_puffs := 3
  let sister_puffs := 3
  let grandmother_puffs := 5
  let friends := 3
  let puffs_per_friend := 9
  original_puffs - (mom_puffs + sister_puffs + grandmother_puffs + friends * puffs_per_friend) = 2 :=
by
  sorry

end dandelion_dog_puffs_l1723_172361


namespace percent_increase_l1723_172307

theorem percent_increase (original value new_value : ℕ) (h1 : original_value = 20) (h2 : new_value = 25) :
  ((new_value - original_value) / original_value) * 100 = 25 :=
by
  -- Proof omitted
  sorry

end percent_increase_l1723_172307


namespace number_of_meetings_l1723_172376

noncomputable def selena_radius : ℝ := 70
noncomputable def bashar_radius : ℝ := 80
noncomputable def selena_speed : ℝ := 200
noncomputable def bashar_speed : ℝ := 240
noncomputable def active_time_together : ℝ := 30

noncomputable def selena_circumference : ℝ := 2 * Real.pi * selena_radius
noncomputable def bashar_circumference : ℝ := 2 * Real.pi * bashar_radius

noncomputable def selena_angular_speed : ℝ := (selena_speed / selena_circumference) * (2 * Real.pi)
noncomputable def bashar_angular_speed : ℝ := (bashar_speed / bashar_circumference) * (2 * Real.pi)

noncomputable def relative_angular_speed : ℝ := selena_angular_speed + bashar_angular_speed
noncomputable def time_to_meet_once : ℝ := (2 * Real.pi) / relative_angular_speed

theorem number_of_meetings : Int := 
    ⌊active_time_together / time_to_meet_once⌋

example : number_of_meetings = 21 := by
  sorry

end number_of_meetings_l1723_172376


namespace pq_plus_p_plus_q_eq_1_l1723_172338

-- Define the polynomial
def poly (x : ℝ) : ℝ := x^4 - 6 * x - 1

-- Prove the target statement
theorem pq_plus_p_plus_q_eq_1 (p q : ℝ) (hpq : poly p = 0) (hq : poly q = 0) :
  p * q + p + q = 1 := by
  sorry

end pq_plus_p_plus_q_eq_1_l1723_172338
