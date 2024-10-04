import Mathlib

namespace p_q_2r_value_l308_308477

variable (p q r : ℝ) (f : ℝ → ℝ)

-- The conditions as definitions
def f_def : f = fun x => p * x^2 + q * x + r := by sorry
def f_at_0 : f 0 = 9 := by sorry
def f_at_1 : f 1 = 6 := by sorry

-- The theorem statement
theorem p_q_2r_value : p + q + 2 * r = 15 :=
by
  -- utilizing the given definitions 
  have h₁ : r = 9 := by sorry
  have h₂ : p + q + r = 6 := by sorry
  -- substitute into p + q + 2r
  sorry

end p_q_2r_value_l308_308477


namespace prob_of_B1_selected_prob_of_D1_in_team_l308_308021

noncomputable def total_teams : ℕ := 20

noncomputable def teams_with_B1 : ℕ := 8

noncomputable def teams_with_D1 : ℕ := 12

theorem prob_of_B1_selected : (teams_with_B1 : ℚ) / total_teams = 2 / 5 := by
  sorry

theorem prob_of_D1_in_team : (teams_with_D1 : ℚ) / total_teams = 3 / 5 := by
  sorry

end prob_of_B1_selected_prob_of_D1_in_team_l308_308021


namespace line_passes_through_fixed_point_l308_308565

theorem line_passes_through_fixed_point :
  ∀ m : ℝ, (m - 1) * (-2) - 3 + 2 * m + 1 = 0 :=
by
  intros m
  sorry

end line_passes_through_fixed_point_l308_308565


namespace meteor_shower_problem_l308_308747

theorem meteor_shower_problem (encounter_time_towards : ℝ) (encounter_time_same : ℝ)
  (h_towards : encounter_time_towards = 7)
  (h_same : encounter_time_same = 13) : 
  let H := 2 * encounter_time_towards * encounter_time_same / (encounter_time_towards + encounter_time_same) in
  H = 9.1 :=
by {
  rw [h_towards, h_same],
  simp [H],
  norm_num,
  sorry
}

end meteor_shower_problem_l308_308747


namespace jelly_beans_in_jar_X_l308_308305

theorem jelly_beans_in_jar_X : 
  ∀ (X Y : ℕ), (X + Y = 1200) → (X = 3 * Y - 400) → X = 800 :=
by
  sorry

end jelly_beans_in_jar_X_l308_308305


namespace pies_from_36_apples_l308_308282

-- Definitions of conditions
def pies_from_apples (apples : Nat) : Nat :=
  apples / 4  -- because 12 apples = 3 pies implies 1 pie = 4 apples

-- Theorem to prove
theorem pies_from_36_apples : pies_from_apples 36 = 9 := by
  sorry

end pies_from_36_apples_l308_308282


namespace determine_sanity_l308_308636

-- Defining the conditions for sanity based on responses to a specific question

-- Define possible responses
inductive Response
| ball : Response
| yes : Response

-- Define sanity based on logical interpretation of an illogical question
def is_sane (response : Response) : Prop :=
  response = Response.ball

-- The theorem stating asking the specific question determines sanity
theorem determine_sanity (response : Response) : is_sane response ↔ response = Response.ball :=
by
  sorry

end determine_sanity_l308_308636


namespace max_n_factoring_polynomial_l308_308800

theorem max_n_factoring_polynomial :
  ∃ n A B : ℤ, (3 * n + A = 217) ∧ (A * B = 72) ∧ (3 * B + A = n) :=
sorry

end max_n_factoring_polynomial_l308_308800


namespace solution_set_of_inequality_l308_308845

variable {f : ℝ → ℝ}

theorem solution_set_of_inequality (h1 : ∀ x : ℝ, deriv f x = 2 * f x)
                                    (h2 : f 0 = 1) :
  { x : ℝ | f (Real.log (x^2 - x)) < 4 } = { x | -1 < x ∧ x < 0 ∨ 1 < x ∧ x < 2 } :=
by {
  sorry
}

end solution_set_of_inequality_l308_308845


namespace product_is_in_A_l308_308971

def is_sum_of_squares (z : Int) : Prop :=
  ∃ t s : Int, z = t^2 + s^2

variable {x y : Int}

theorem product_is_in_A (hx : is_sum_of_squares x) (hy : is_sum_of_squares y) :
  is_sum_of_squares (x * y) :=
sorry

end product_is_in_A_l308_308971


namespace sum_of_transformed_numbers_l308_308304

theorem sum_of_transformed_numbers (a b S : ℕ) (h : a + b = S) :
  3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 :=
by
  sorry

end sum_of_transformed_numbers_l308_308304


namespace complementary_angles_difference_l308_308144

def complementary_angles (θ1 θ2 : ℝ) : Prop :=
  θ1 + θ2 = 90

theorem complementary_angles_difference:
  ∀ (θ1 θ2 : ℝ), 
  (θ1 / θ2 = 4 / 5) → 
  complementary_angles θ1 θ2 → 
  abs (θ2 - θ1) = 10 :=
by
  sorry

end complementary_angles_difference_l308_308144


namespace minimize_expression_l308_308055

variable (a b c : ℝ)
variable (h1 : a > b)
variable (h2 : b > c)
variable (h3 : a ≠ 0)

theorem minimize_expression : 
  (a > b) → (b > c) → (a ≠ 0) → 
  ∃ x : ℝ, x = 4 ∧ ∀ y, y = (a+b)^2 + (b+c)^2 + (c+a)^2 / a^2 → x ≤ y := sorry

end minimize_expression_l308_308055


namespace highest_throw_christine_janice_l308_308345

theorem highest_throw_christine_janice
  (c1 : ℕ) -- Christine's first throw
  (j1 : ℕ) -- Janice's first throw
  (c2 : ℕ) -- Christine's second throw
  (j2 : ℕ) -- Janice's second throw
  (c3 : ℕ) -- Christine's third throw
  (j3 : ℕ) -- Janice's third throw
  (h1 : c1 = 20)
  (h2 : j1 = c1 - 4)
  (h3 : c2 = c1 + 10)
  (h4 : j2 = j1 * 2)
  (h5 : c3 = c2 + 4)
  (h6 : j3 = c1 + 17) :
  max c1 (max c2 (max c3 (max j1 (max j2 j3)))) = 37 := by
  sorry

end highest_throw_christine_janice_l308_308345


namespace count_three_digit_integers_ending_in_7_divisible_by_21_l308_308679

theorem count_three_digit_integers_ending_in_7_divisible_by_21 : 
  let count := (finset.filter (λ n : ℕ, n % 10 = 7 ∧ n % 21 = 0) (finset.Ico 100 1000)).card in
  count = 43 :=
by {
  let count := (finset.filter (λ n : ℕ, n % 10 = 7 ∧ n % 21 = 0) (finset.Ico 100 1000)).card,
  have h : count = 43 := sorry,
  exact h,
}

end count_three_digit_integers_ending_in_7_divisible_by_21_l308_308679


namespace evaluate_expression_l308_308889

theorem evaluate_expression : 4 * (8 - 3) - 7 = 13 := by
  sorry

end evaluate_expression_l308_308889


namespace solve_for_A_l308_308224

variable (a b : ℝ) 

theorem solve_for_A (A : ℝ) (h : (5 * a + 3 * b)^2 = (5 * a - 3 * b)^2 + A) : 
  A = 60 * a * b := by
  sorry

end solve_for_A_l308_308224


namespace probability_point_closer_to_7_than_0_l308_308778

noncomputable def segment_length (a b : ℝ) : ℝ := b - a
noncomputable def closer_segment (a c b : ℝ) : ℝ := segment_length c b

theorem probability_point_closer_to_7_than_0 :
  let a := 0
  let b := 10
  let c := 7
  let midpoint := (a + c) / 2
  let total_length := b - a
  let closer_length := segment_length midpoint b
  (closer_length / total_length) = 0.7 :=
by
  sorry

end probability_point_closer_to_7_than_0_l308_308778


namespace alvin_marble_count_correct_l308_308633

variable (initial_marble_count lost_marble_count won_marble_count final_marble_count : ℕ)

def calculate_final_marble_count (initial : ℕ) (lost : ℕ) (won : ℕ) : ℕ :=
  initial - lost + won

theorem alvin_marble_count_correct :
  initial_marble_count = 57 →
  lost_marble_count = 18 →
  won_marble_count = 25 →
  final_marble_count = calculate_final_marble_count initial_marble_count lost_marble_count won_marble_count →
  final_marble_count = 64 :=
by
  intros h_initial h_lost h_won h_calculate
  rw [h_initial, h_lost, h_won] at h_calculate
  exact h_calculate

end alvin_marble_count_correct_l308_308633


namespace cube_root_of_8_l308_308858

theorem cube_root_of_8 : (∃ x : ℝ, x * x * x = 8) ∧ (∃ y : ℝ, y * y * y = 8 → y = 2) :=
by
  sorry

end cube_root_of_8_l308_308858


namespace binary_sum_in_base_10_l308_308453

theorem binary_sum_in_base_10 :
  (255 : ℕ) + (63 : ℕ) = 318 :=
sorry

end binary_sum_in_base_10_l308_308453


namespace jessica_watermelons_l308_308840

theorem jessica_watermelons (original : ℕ) (eaten : ℕ) (remaining : ℕ) 
    (h1 : original = 35) 
    (h2 : eaten = 27) 
    (h3 : remaining = original - eaten) : 
  remaining = 8 := 
by {
    -- This is where the proof would go
    sorry
}

end jessica_watermelons_l308_308840


namespace problem_statement_l308_308998

theorem problem_statement (x : ℝ) :
  (x - 2)^4 + 5 * (x - 2)^3 + 10 * (x - 2)^2 + 10 * (x - 2) + 5 = (x - 2 + Real.sqrt 2)^4 := by
  sorry

end problem_statement_l308_308998


namespace find_m_l308_308371

theorem find_m (S : ℕ → ℝ) (m : ℝ) (h : ∀ n, S n = m * 2^(n-1) - 3) : m = 6 :=
by
  sorry

end find_m_l308_308371


namespace Tonya_buys_3_lego_sets_l308_308598

-- Definitions based on conditions
def num_sisters : Nat := 2
def num_dolls : Nat := 4
def price_per_doll : Nat := 15
def price_per_lego_set : Nat := 20

-- The amount of money spent on each sister should be the same
def amount_spent_on_younger_sister := num_dolls * price_per_doll
def amount_spent_on_older_sister := (amount_spent_on_younger_sister / price_per_lego_set)

-- Proof statement
theorem Tonya_buys_3_lego_sets : amount_spent_on_older_sister = 3 :=
by
  sorry

end Tonya_buys_3_lego_sets_l308_308598


namespace measure_15_minutes_l308_308150

/-- Given a timer setup with a 7-minute hourglass and an 11-minute hourglass, show that we can measure exactly 15 minutes. -/
theorem measure_15_minutes (h7 : ∃ t : ℕ, t = 7) (h11 : ∃ t : ℕ, t = 11) : ∃ t : ℕ, t = 15 := 
  by 
    sorry

end measure_15_minutes_l308_308150


namespace integer_values_m_l308_308071

theorem integer_values_m (m x y : ℤ) (h1 : x - 2 * y = m) (h2 : 2 * x + 3 * y = 2 * m - 3)
    (h3 : 3 * x + y ≥ 0) (h4 : x + 5 * y < 0) : m = 1 ∨ m = 2 :=
by
  sorry

end integer_values_m_l308_308071


namespace domain_of_function_l308_308004

theorem domain_of_function :
  ∀ x : ℝ, (2 - x > 0) ∧ (2 * x + 1 > 0) ↔ (-1 / 2 < x) ∧ (x < 2) :=
sorry

end domain_of_function_l308_308004


namespace find_monic_polynomial_l308_308329

-- Define the original polynomial
def polynomial_1 (x : ℝ) := x^3 - 4 * x^2 + 9

-- Define the monic polynomial we are seeking
def polynomial_2 (x : ℝ) := x^3 - 12 * x^2 + 243

theorem find_monic_polynomial :
  ∀ (r1 r2 r3 : ℝ), 
    polynomial_1 r1 = 0 → 
    polynomial_1 r2 = 0 → 
    polynomial_1 r3 = 0 → 
    polynomial_2 (3 * r1) = 0 ∧ polynomial_2 (3 * r2) = 0 ∧ polynomial_2 (3 * r3) = 0 :=
by
  intros r1 r2 r3 h1 h2 h3
  sorry

end find_monic_polynomial_l308_308329


namespace book_chapters_not_determinable_l308_308990

variable (pages_initially pages_later pages_total total_pages book_chapters : ℕ)

def problem_statement : Prop :=
  pages_initially = 37 ∧ pages_later = 25 ∧ pages_total = 62 ∧ total_pages = 95 ∧ book_chapters = 0

theorem book_chapters_not_determinable (h: problem_statement pages_initially pages_later pages_total total_pages book_chapters) :
  book_chapters = 0 :=
by
  sorry

end book_chapters_not_determinable_l308_308990


namespace smallest_m_for_integral_roots_l308_308459

theorem smallest_m_for_integral_roots :
  ∃ (m : ℕ), (∃ (p q : ℤ), p * q = 30 ∧ m = 12 * (p + q)) ∧ m = 132 := by
  sorry

end smallest_m_for_integral_roots_l308_308459


namespace cost_for_15_pounds_of_apples_l308_308634

-- Axiom stating the cost of apples per weight
axiom cost_of_apples (pounds : ℕ) : ℕ

-- Condition given in the problem
def rate_apples : Prop := cost_of_apples 5 = 4

-- Statement of the problem
theorem cost_for_15_pounds_of_apples : rate_apples → cost_of_apples 15 = 12 :=
by
  intro h
  -- Proof to be filled in here
  sorry

end cost_for_15_pounds_of_apples_l308_308634


namespace norris_money_left_l308_308725

def sept_savings : ℕ := 29
def oct_savings : ℕ := 25
def nov_savings : ℕ := 31
def hugo_spent  : ℕ := 75
def total_savings : ℕ := sept_savings + oct_savings + nov_savings
def norris_left : ℕ := total_savings - hugo_spent

theorem norris_money_left : norris_left = 10 := by
  unfold norris_left total_savings sept_savings oct_savings nov_savings hugo_spent
  sorry

end norris_money_left_l308_308725


namespace sequence_product_modulo_7_l308_308655

theorem sequence_product_modulo_7 :
  let s := [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]
  s.foldl (*) 1 % 7 = 4 := by
  sorry

end sequence_product_modulo_7_l308_308655


namespace right_triangle_hypotenuse_enlargement_l308_308692

theorem right_triangle_hypotenuse_enlargement
  (a b c : ℝ)
  (h : c^2 = a^2 + b^2) :
  ((5 * a)^2 + (5 * b)^2 = (5 * c)^2) :=
by sorry

end right_triangle_hypotenuse_enlargement_l308_308692


namespace count_integers_satisfying_condition_l308_308245

theorem count_integers_satisfying_condition :
  ({n : ℕ | 300 < n^2 ∧ n^2 < 1000}.card = 14) :=
by
  sorry

end count_integers_satisfying_condition_l308_308245


namespace jared_annual_earnings_l308_308502

-- Defining conditions as constants
def diploma_pay : ℕ := 4000
def degree_multiplier : ℕ := 3
def months_in_year : ℕ := 12

-- Goal: Prove that Jared's annual earnings are $144,000
theorem jared_annual_earnings : diploma_pay * degree_multiplier * months_in_year = 144000 := by
  sorry

end jared_annual_earnings_l308_308502


namespace root_fraction_power_l308_308430

theorem root_fraction_power (a : ℝ) (ha : a = 5) : 
  (a^(1/3)) / (a^(1/5)) = a^(2/15) := by
  sorry

end root_fraction_power_l308_308430


namespace total_share_proof_l308_308123

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

end total_share_proof_l308_308123


namespace no_positive_integer_solutions_l308_308643

theorem no_positive_integer_solutions:
    ∀ x y : ℕ, x > 0 → y > 0 → x^2 + 2 * y^2 = 2 * x^3 - x → false :=
by
  sorry

end no_positive_integer_solutions_l308_308643


namespace factorial_expression_identity_l308_308044

open Nat

theorem factorial_expression_identity : 7! - 6 * 6! - 7! = 0 := by
  sorry

end factorial_expression_identity_l308_308044


namespace alice_unanswered_questions_l308_308176

-- Declare variables for the proof
variables (c w u : ℕ)

-- State the problem in Lean
theorem alice_unanswered_questions :
  50 + 5 * c - 2 * w = 100 ∧
  40 + 7 * c - w - u = 120 ∧
  6 * c + 3 * u = 130 ∧
  c + w + u = 25 →
  u = 20 :=
by
  intros h
  sorry

end alice_unanswered_questions_l308_308176


namespace shorter_side_length_l308_308252

theorem shorter_side_length (L W : ℝ) (h₁ : L * W = 104) (h₂ : 2 * L + 2 * W = 42) : W = 8 :=
by
  have h₃ : L + W = 21 := by linarith
  have h₄ : W = 21 - L := by linarith
  have quad_eq : L^2 - 21*L + 104 = 0 := by linarith
  has_solution_L_one : L = 13 := sorry
  has_solution_L_two : L = 8 := sorry
  use W
  sorry

end shorter_side_length_l308_308252


namespace circle_center_transformation_l308_308934

def original_center : ℤ × ℤ := (3, -4)

def reflect_x_axis (p : ℤ × ℤ) : ℤ × ℤ := (p.1, -p.2)

def translate_right (p : ℤ × ℤ) (d : ℤ) : ℤ × ℤ := (p.1 + d, p.2)

def final_center : ℤ × ℤ := (8, 4)

theorem circle_center_transformation :
  translate_right (reflect_x_axis original_center) 5 = final_center :=
by
  sorry

end circle_center_transformation_l308_308934


namespace triangle_inequality_l308_308572

theorem triangle_inequality 
  (a b c : ℝ) -- lengths of the sides of the triangle
  (α β γ : ℝ) -- angles of the triangle in radians opposite to sides a, b, c
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)  -- positivity of sides
  (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π) (hγ : 0 < γ ∧ γ < π) -- positivity and range of angles
  (h_sum : α + β + γ = π) -- angle sum property of a triangle
: 
  b / Real.sin (γ + α / 3) + c / Real.sin (β + α / 3) > (2 / 3) * (a / Real.sin (α / 3)) :=
sorry

end triangle_inequality_l308_308572


namespace book_total_pages_l308_308078

theorem book_total_pages (x : ℝ) 
  (h1 : ∀ d1 : ℝ, d1 = x * (1/6) + 10)
  (h2 : ∀ remaining1 : ℝ, remaining1 = x - d1)
  (h3 : ∀ d2 : ℝ, d2 = remaining1 * (1/5) + 12)
  (h4 : ∀ remaining2 : ℝ, remaining2 = remaining1 - d2)
  (h5 : ∀ d3 : ℝ, d3 = remaining2 * (1/4) + 14)
  (h6 : ∀ remaining3 : ℝ, remaining3 = remaining2 - d3)
  (h7 : remaining3 = 52) : x = 169 := sorry

end book_total_pages_l308_308078


namespace triangle_inequality_l308_308711

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  a^2 * c * (a - b) + b^2 * a * (b - c) + c^2 * b * (c - a) ≥ 0 :=
sorry

end triangle_inequality_l308_308711


namespace floor_e_eq_2_l308_308209

theorem floor_e_eq_2 : ⌊Real.exp 1⌋ = 2 := by
  sorry

end floor_e_eq_2_l308_308209


namespace beads_problem_l308_308163

noncomputable def number_of_blue_beads (total_beads : ℕ) (beads_with_blue_neighbor : ℕ) (beads_with_green_neighbor : ℕ) : ℕ :=
  let beads_with_both_neighbors := beads_with_blue_neighbor + beads_with_green_neighbor - total_beads
  let beads_with_only_blue_neighbor := beads_with_blue_neighbor - beads_with_both_neighbors
  (2 * beads_with_only_blue_neighbor + beads_with_both_neighbors) / 2

theorem beads_problem : number_of_blue_beads 30 26 20 = 18 := by 
  -- ...
  sorry

end beads_problem_l308_308163


namespace swimming_speed_l308_308626

theorem swimming_speed (v : ℝ) (water_speed : ℝ) (distance : ℝ) (time : ℝ) 
  (h1 : water_speed = 2) 
  (h2 : distance = 14) 
  (h3 : time = 3.5) 
  (h4 : distance = (v - water_speed) * time) : 
  v = 6 := 
by
  sorry

end swimming_speed_l308_308626


namespace inverse_44_mod_53_l308_308227

theorem inverse_44_mod_53 : (44 * 22) % 53 = 1 :=
by
-- Given condition: 19's inverse modulo 53 is 31
have h: (19 * 31) % 53 = 1 := by sorry
-- We should prove the required statement using the given condition.
sorry

end inverse_44_mod_53_l308_308227


namespace correct_transformation_l308_308618

theorem correct_transformation (x : ℝ) : x^2 - 10 * x - 1 = 0 → (x - 5)^2 = 26 :=
  sorry

end correct_transformation_l308_308618


namespace sum_of_n_with_unformable_postage_120_equals_43_l308_308057

theorem sum_of_n_with_unformable_postage_120_equals_43 :
  ∃ n1 n2 : ℕ, n1 = 21 ∧ n2 = 22 ∧ 
  (∀ k : ℕ, k > 120 → ∃ a b c : ℕ, k = 7 * a + n1 * b + (n1 + 1) * c) ∧ 
  (∀ k : ℕ, k > 120 → ∃ a b c : ℕ, k = 7 * a + n2 * b + (n2 + 1) * c) ∧ 
  (120 = 7 * a + n1 * b + (n1 + 1) * c → a = 0 ∧ b = 0 ∧ c = 0) ∧
  (120 = 7 * a + n2 * b + (n2 + 1) * c → a = 0 ∧ b = 0 ∧ c = 0) ∧
  (n1 + n2 = 43) :=
by
  sorry

end sum_of_n_with_unformable_postage_120_equals_43_l308_308057


namespace HunterScoreIs45_l308_308378

variable (G J H : ℕ)
variable (h1 : G = J + 10)
variable (h2 : J = 2 * H)
variable (h3 : G = 100)

theorem HunterScoreIs45 : H = 45 := by
  sorry

end HunterScoreIs45_l308_308378


namespace quadratics_root_k_value_l308_308372

theorem quadratics_root_k_value :
  (∀ k : ℝ, (∀ x : ℝ, x^2 + k * x + 6 = 0 → (x = 2 ∨ ∃ x1 : ℝ, x1 * 2 = 6 ∧ x1 + 2 = k)) → 
  (x = 2 → ∃ x1 : ℝ, x1 = 3 ∧ k = -5)) := 
sorry

end quadratics_root_k_value_l308_308372


namespace min_value_inequality_l308_308562

open Real

theorem min_value_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 3 * y = 4) :
  ∃ z, z = (2 / x + 3 / y) ∧ z = 25 / 4 :=
by
  sorry

end min_value_inequality_l308_308562


namespace rosie_pies_l308_308274

theorem rosie_pies (apples_per_pie : ℕ) (apples_total : ℕ) (pies_initial : ℕ) 
  (h1 : 3 = pies_initial) (h2 : 12 = apples_total) : 
  (36 / (apples_total / pies_initial)) * pies_initial = 27 := 
by
  sorry

end rosie_pies_l308_308274


namespace coprime_divisibility_l308_308162

theorem coprime_divisibility (p q r P Q R : ℕ)
  (hpq : Nat.gcd p q = 1) (hpr : Nat.gcd p r = 1) (hqr : Nat.gcd q r = 1)
  (h : ∃ k : ℤ, (P:ℤ) * (q*r) + (Q:ℤ) * (p*r) + (R:ℤ) * (p*q) = k * (p*q * r)) :
  ∃ a b c : ℤ, (P:ℤ) = a * (p:ℤ) ∧ (Q:ℤ) = b * (q:ℤ) ∧ (R:ℤ) = c * (r:ℤ) :=
by
  sorry

end coprime_divisibility_l308_308162


namespace prob_win_3_1_correct_l308_308763

-- Defining the probability for winning a game
def prob_win_game : ℚ := 2 / 3

-- Defining the probability for losing a game
def prob_lose_game : ℚ := 1 - prob_win_game

-- A function to calculate the probability of winning the match with a 3:1 score
def prob_win_3_1 : ℚ :=
  let combinations := 3 -- Number of ways to lose exactly 1 game in the first 3 games (C_3^1)
  let win_prob := prob_win_game ^ 3 -- Probability for winning 3 games
  let lose_prob := prob_lose_game -- Probability for losing 1 game
  combinations * win_prob * lose_prob

-- The theorem that states the probability that player A wins with a score of 3:1
theorem prob_win_3_1_correct : prob_win_3_1 = 8 / 27 := by
  sorry

end prob_win_3_1_correct_l308_308763


namespace probability_exactly_one_six_probability_at_least_one_six_probability_at_most_one_six_l308_308601

-- Considering a die with 6 faces
def die_faces := 6

-- Total number of possible outcomes when rolling 3 dice
def total_outcomes := die_faces^3

-- 1. Probability of having exactly one die showing a 6 when rolling 3 dice
def prob_exactly_one_six : ℚ :=
  have favorable_outcomes := 3 * 5^2 -- 3 ways to choose which die shows 6, and 25 ways for others to not show 6
  favorable_outcomes / total_outcomes

-- Proof statement
theorem probability_exactly_one_six : prob_exactly_one_six = 25/72 := by 
  sorry

-- 2. Probability of having at least one die showing a 6 when rolling 3 dice
def prob_at_least_one_six : ℚ :=
  have no_six_outcomes := 5^3
  (total_outcomes - no_six_outcomes) / total_outcomes

-- Proof statement
theorem probability_at_least_one_six : prob_at_least_one_six = 91/216 := by 
  sorry

-- 3. Probability of having at most one die showing a 6 when rolling 3 dice
def prob_at_most_one_six : ℚ :=
  have no_six_probability := 125 / total_outcomes
  have one_six_probability := 75 / total_outcomes
  no_six_probability + one_six_probability

-- Proof statement
theorem probability_at_most_one_six : prob_at_most_one_six = 25/27 := by 
  sorry

end probability_exactly_one_six_probability_at_least_one_six_probability_at_most_one_six_l308_308601


namespace proof_of_truth_values_l308_308443

open Classical

variables (x : ℝ)

-- Original proposition: If x = 1, then x^2 = 1.
def original_proposition : Prop := (x = 1) → (x^2 = 1)

-- Converse of the original proposition: If x^2 = 1, then x = 1.
def converse_proposition : Prop := (x^2 = 1) → (x = 1)

-- Inverse of the original proposition: If x ≠ 1, then x^2 ≠ 1.
def inverse_proposition : Prop := (x ≠ 1) → (x^2 ≠ 1)

-- Contrapositive of the original proposition: If x^2 ≠ 1, then x ≠ 1.
def contrapositive_proposition : Prop := (x^2 ≠ 1) → (x ≠ 1)

-- Negation of the original proposition: If x = 1, then x^2 ≠ 1.
def negation_proposition : Prop := (x = 1) → (x^2 ≠ 1)

theorem proof_of_truth_values :
  (original_proposition x) ∧
  (converse_proposition x = False) ∧
  (inverse_proposition x = False) ∧
  (contrapositive_proposition x) ∧
  (negation_proposition x = False) := by
  sorry

end proof_of_truth_values_l308_308443


namespace maximilian_annual_revenue_l308_308121

-- Define the number of units in the building
def total_units : ℕ := 100

-- Define the occupancy rate
def occupancy_rate : ℚ := 3 / 4

-- Define the monthly rent per unit
def monthly_rent : ℚ := 400

-- Calculate the number of occupied units
def occupied_units : ℕ := (occupancy_rate * total_units : ℚ).natAbs

-- Calculate the monthly rent revenue
def monthly_revenue : ℚ := occupied_units * monthly_rent

-- Calculate the annual rent revenue
def annual_revenue : ℚ := monthly_revenue * 12

-- Prove that the annual revenue is $360,000
theorem maximilian_annual_revenue : annual_revenue = 360000 := by
  sorry

end maximilian_annual_revenue_l308_308121


namespace cubed_multiplication_identity_l308_308182

theorem cubed_multiplication_identity : 3^3 * 6^3 = 5832 := by
  sorry

end cubed_multiplication_identity_l308_308182


namespace twenty_percent_of_x_l308_308385

noncomputable def x := 1800 / 1.2

theorem twenty_percent_of_x (h : 1.2 * x = 1800) : 0.2 * x = 300 :=
by
  -- The proof would go here, but we'll replace it with sorry.
  sorry

end twenty_percent_of_x_l308_308385


namespace perimeter_region_l308_308540

theorem perimeter_region (rectangle_height : ℕ) (height_eq_sixteen : rectangle_height = 16) (rect_area_eq : 12 * rectangle_height = 192) (total_area_eq : 12 * rectangle_height - 60 = 132):
  (rectangle_height + 12 + 4 + 6 + 10 * 2) = 54 :=
by
  have h1 : 12 * 16 = 192 := by sorry
  exact sorry


end perimeter_region_l308_308540


namespace coefficient_of_x6_in_expansion_l308_308317

theorem coefficient_of_x6_in_expansion :
  let a := 1
  let b := -3 * (x : ℝ) ^ 3
  let n := 4
  let k := 2
  (1 - 3 * (x : ℝ) ^ 3) ^ 4 = ∑ k in finset.range (n + 1), 
    (nat.choose n k) * a ^ (n - k) * b ^ k →
  is_term_of_degree (1 - 3 * (x : ℝ) ^ 3) ^ 4 x 6 (54 * x ^ 6) :=
by
  sorry

end coefficient_of_x6_in_expansion_l308_308317


namespace num_of_three_digit_integers_with_7_in_units_divisible_by_21_l308_308676

theorem num_of_three_digit_integers_with_7_in_units_divisible_by_21 : 
  let nums := (list.range' 100 900).filter (λ n => n % 21 = 0 ∧ n % 10 = 7) in
  nums.length = 5 := 
by
  let nums := 
    (list.range' 100 900).filter (λ n => n % 21 = 0 ∧ n % 10 = 7)
  show nums.length = 5
  sorry

end num_of_three_digit_integers_with_7_in_units_divisible_by_21_l308_308676


namespace percentage_ethanol_in_fuel_B_l308_308483

-- Definitions from the conditions
def tank_capacity : ℝ := 218
def ethanol_percentage_fuel_A : ℝ := 0.12
def total_ethanol : ℝ := 30
def volume_of_fuel_A : ℝ := 122

-- Expression to calculate ethanol in Fuel A
def ethanol_in_fuel_A : ℝ := ethanol_percentage_fuel_A * volume_of_fuel_A

-- The remaining ethanol in Fuel B = Total ethanol - Ethanol in Fuel A
def ethanol_in_fuel_B : ℝ := total_ethanol - ethanol_in_fuel_A

-- The volume of fuel B used to fill the tank
def volume_of_fuel_B : ℝ := tank_capacity - volume_of_fuel_A

-- Statement to prove:
theorem percentage_ethanol_in_fuel_B : (ethanol_in_fuel_B / volume_of_fuel_B) * 100 = 16 :=
sorry

end percentage_ethanol_in_fuel_B_l308_308483


namespace base8_to_base10_362_eq_242_l308_308923

theorem base8_to_base10_362_eq_242 : 
  let digits := [3, 6, 2]
  let base := 8
  let base10_value := (digits[2] * base^0) + (digits[1] * base^1) + (digits[0] * base^2) 
  base10_value = 242 :=
by
  sorry

end base8_to_base10_362_eq_242_l308_308923


namespace min_value_y_minus_one_over_x_l308_308229

variable {x y : ℝ}

-- Condition 1: x is the median of the dataset
def is_median (x : ℝ) : Prop := 3 ≤ x ∧ x ≤ 5

-- Condition 2: The average of the dataset is 1
def average_is_one (x y : ℝ) : Prop := 1 + 2 + x^2 - y = 4

-- The statement to be proved
theorem min_value_y_minus_one_over_x :
  ∀ (x y : ℝ), is_median x → average_is_one x y → y = x^2 - 1 → (y - 1/x) ≥ 23/3 :=
by 
  -- This is a placeholder for the actual proof
  sorry

end min_value_y_minus_one_over_x_l308_308229


namespace evaluate_expression_l308_308887

theorem evaluate_expression : 4 * (8 - 3) - 7 = 13 := by
  sorry

end evaluate_expression_l308_308887


namespace eval_expression_l308_308894

theorem eval_expression : 4 * (8 - 3) - 7 = 13 :=
by
  sorry

end eval_expression_l308_308894


namespace probability_top_card_is_king_or_queen_l308_308781

-- Defining the basic entities of the problem
def standard_deck_size := 52
def ranks := 13
def suits := 4
def number_of_kings := 4
def number_of_queens := 4
def number_of_kings_and_queens := number_of_kings + number_of_queens

-- Statement: Calculating the probability that the top card is either a King or a Queen
theorem probability_top_card_is_king_or_queen :
  (number_of_kings_and_queens : ℚ) / standard_deck_size = 2 / 13 := by
  -- Skipping the proof for now
  sorry

end probability_top_card_is_king_or_queen_l308_308781


namespace quadratic_inverse_sum_roots_l308_308063

theorem quadratic_inverse_sum_roots (x1 x2 : ℝ) (h1 : x1^2 - 2023 * x1 + 1 = 0) (h2 : x2^2 - 2023 * x2 + 1 = 0) : 
  (1/x1 + 1/x2) = 2023 :=
by
  -- We outline the proof steps that should be accomplished.
  -- These will be placeholders and not part of the actual statement.
  -- sorry allows us to skip the proof.
  sorry

end quadratic_inverse_sum_roots_l308_308063


namespace solve_picnic_problem_l308_308627

def picnic_problem : Prop :=
  ∃ (M W A C : ℕ), 
    M = W + 80 ∧ 
    A = C + 80 ∧ 
    M + W = A ∧ 
    A + C = 240 ∧ 
    M = 120

theorem solve_picnic_problem : picnic_problem :=
  sorry

end solve_picnic_problem_l308_308627


namespace waste_in_scientific_notation_l308_308089

def water_waste_per_person : ℝ := 0.32
def number_of_people : ℝ := 10^6

def total_daily_waste : ℝ := water_waste_per_person * number_of_people

def scientific_notation (x : ℝ) : Prop :=
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ x = a * 10^n

theorem waste_in_scientific_notation :
  scientific_notation total_daily_waste ∧ total_daily_waste = 3.2 * 10^5 :=
by
  sorry

end waste_in_scientific_notation_l308_308089


namespace pies_from_36_apples_l308_308281

-- Definitions of conditions
def pies_from_apples (apples : Nat) : Nat :=
  apples / 4  -- because 12 apples = 3 pies implies 1 pie = 4 apples

-- Theorem to prove
theorem pies_from_36_apples : pies_from_apples 36 = 9 := by
  sorry

end pies_from_36_apples_l308_308281


namespace James_uses_150_sheets_of_paper_l308_308707

-- Define the conditions
def number_of_books := 2
def pages_per_book := 600
def pages_per_side := 4
def sides_per_sheet := 2

-- Statement to prove
theorem James_uses_150_sheets_of_paper :
  number_of_books * pages_per_book / (pages_per_side * sides_per_sheet) = 150 :=
by sorry

end James_uses_150_sheets_of_paper_l308_308707


namespace problem_solution_l308_308098

-- Definitions and assumptions
variables (priceA priceB : ℕ)
variables (numBooksA numBooksB totalBooks : ℕ)
variables (costPriceA : priceA = 45)
variables (costPriceB : priceB = 65)
variables (totalCost : priceA * numBooksA + priceB * numBooksB ≤ 3550)
variables (totalBooksEq : numBooksA + numBooksB = 70)

-- Proof problem
theorem problem_solution :
  priceA = 45 ∧ priceB = 65 ∧ ∃ (numBooksA : ℕ), numBooksA ≥ 50 :=
by
  sorry

end problem_solution_l308_308098


namespace not_necessarily_true_l308_308531

theorem not_necessarily_true (x y : ℝ) (h : x > y) : ¬ (x^2 > y^2) :=
sorry

end not_necessarily_true_l308_308531


namespace Q_at_1_eq_neg_1_l308_308710

def P (x : ℝ) : ℝ := 3 * x^3 - 5 * x^2 + 2 * x - 1

noncomputable def mean_coefficient : ℝ := (3 - 5 + 2 - 1) / 4

noncomputable def Q (x : ℝ) : ℝ := mean_coefficient * x^3 + mean_coefficient * x^2 + mean_coefficient * x + mean_coefficient

theorem Q_at_1_eq_neg_1 : Q 1 = -1 := by
  sorry

end Q_at_1_eq_neg_1_l308_308710


namespace shaded_region_area_l308_308935

theorem shaded_region_area {radius1 radius2 : ℝ} (h1 : radius1 = 4) (h2 : radius2 = 5) :
  let dist_centers := radius1 + radius2,
      circumscribed_radius := dist_centers,
      larger_area := Real.pi * circumscribed_radius ^ 2,
      smaller_area1 := Real.pi * radius1 ^ 2,
      smaller_area2 := Real.pi * radius2 ^ 2,
      shaded_area := larger_area - smaller_area1 - smaller_area2
  in shaded_area = 40 * Real.pi :=
by
  simp [h1, h2]
  sorry

end shaded_region_area_l308_308935


namespace proof_problem_l308_308097

variables (x_A x_B m : ℝ)

-- Condition 1:
def cost_relation : Prop := x_B = x_A + 20

-- Condition 2:
def quantity_relation : Prop := 540 / x_A = 780 / x_B

-- Condition 3:
def total_books := 70
def total_cost := 3550
def min_books_relation : Prop := 45 * m + 65 * (total_books - m) ≤ total_cost 

-- Part 1:
def cost_price_A (x : ℝ) : Prop := x = 45
def cost_price_B (x : ℝ) : Prop := x = 65

-- Part 2:
def min_books_A (m : ℝ) : Prop := m ≥ 50

-- Define the proof problem
theorem proof_problem :
  (cost_relation x_A x_B) ∧ 
  (quantity_relation x_A x_B) →
  (cost_price_A x_A) ∧ 
  (cost_price_B x_B) ∧ 
  (min_books_relation x_A x_B m) → 
  (min_books_A m) :=
by sorry

end proof_problem_l308_308097


namespace range_of_a_l308_308147

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, (x^2 + (a^2 + 1) * x + a - 2 = 0 ∧ y^2 + (a^2 + 1) * y + a - 2 = 0)
    ∧ x > 1 ∧ y < -1) ↔ (-1 < a ∧ a < 0) := sorry

end range_of_a_l308_308147


namespace brownies_left_l308_308596

theorem brownies_left (initial : ℕ) (tina_ate : ℕ) (husband_ate : ℕ) (shared : ℕ) 
                      (h_initial : initial = 24)
                      (h_tina : tina_ate = 10)
                      (h_husband : husband_ate = 5)
                      (h_shared : shared = 4) : 
  initial - tina_ate - husband_ate - shared = 5 :=
by
  rw [h_initial, h_tina, h_husband, h_shared]
  exact Nat.sub_sub_sub_cancel 24 10 5 4 sorry

end brownies_left_l308_308596


namespace sum_floors_arithmetic_sequence_eq_l308_308938

noncomputable def sum_floors_arithmetic_sequence (a d : ℝ) (n : ℕ) : ℤ :=
∑ k in finset.range n, ⌊a + k * d⌋

theorem sum_floors_arithmetic_sequence_eq :
  sum_floors_arithmetic_sequence (-2) 0.7 144 = 6941 :=
by sorry

end sum_floors_arithmetic_sequence_eq_l308_308938


namespace calculate_annual_rent_l308_308120

-- Defining the conditions
def num_units : ℕ := 100
def occupancy_rate : ℚ := 3 / 4
def monthly_rent : ℚ := 400

-- Defining the target annual rent
def annual_rent (units : ℕ) (occupancy : ℚ) (rent : ℚ) : ℚ :=
  let occupied_units := occupancy * units
  let monthly_revenue := occupied_units * rent
  monthly_revenue * 12

-- Proof problem statement
theorem calculate_annual_rent :
  annual_rent num_units occupancy_rate monthly_rent = 360000 := by
  sorry

end calculate_annual_rent_l308_308120


namespace modulo_calculation_l308_308265

theorem modulo_calculation (n : ℕ) (hn : 0 ≤ n ∧ n < 19) (hmod : 5 * n % 19 = 1) : 
  ((3^n)^2 - 3) % 19 = 3 := 
by 
  sorry

end modulo_calculation_l308_308265


namespace tournament_total_games_l308_308987

theorem tournament_total_games (n : ℕ) (k : ℕ) (h_n : n = 30) (h_k : k = 4) : 
  (n * (n - 1) / 2) * k = 1740 := by
  -- Given conditions
  have h1 : n = 30 := h_n
  have h2 : k = 4 := h_k

  -- Calculation using provided values
  sorry

end tournament_total_games_l308_308987


namespace smallest_n_l308_308810

theorem smallest_n (n : ℕ) (h1 : n > 1) (h2 : 2016 ∣ (3 * n^3 + 2013)) : n = 193 := 
sorry

end smallest_n_l308_308810


namespace jared_yearly_earnings_l308_308508

theorem jared_yearly_earnings (monthly_pay_diploma : ℕ) (multiplier : ℕ) (months_in_year : ℕ)
  (h1 : monthly_pay_diploma = 4000) (h2 : multiplier = 3) (h3 : months_in_year = 12) :
  (monthly_pay_diploma * multiplier * months_in_year) = 144000 :=
by
  -- The proof goes here
  sorry

end jared_yearly_earnings_l308_308508


namespace solve_f_g_f_3_l308_308559

def f (x : ℤ) : ℤ := 2 * x + 4

def g (x : ℤ) : ℤ := 5 * x + 2

theorem solve_f_g_f_3 :
  f (g (f 3)) = 108 := by
  sorry

end solve_f_g_f_3_l308_308559


namespace factorize_expr_l308_308952

theorem factorize_expr (x : ℝ) : (x - 1) * (x + 3) + 4 = (x + 1) ^ 2 :=
by
  sorry

end factorize_expr_l308_308952


namespace initial_percentage_decrease_l308_308741

theorem initial_percentage_decrease (x : ℝ) (P : ℝ) (h₁ : P > 0) (h₂ : 1.55 * (1 - x / 100) = 1.24) :
    x = 20 :=
by
  sorry

end initial_percentage_decrease_l308_308741


namespace triangle_median_inequality_l308_308703

-- Defining the parameters and the inequality theorem.
theorem triangle_median_inequality
  (a b c : ℝ)
  (ma mb mc : ℝ)
  (Δ : ℝ)
  (median_medians : ∀ {a b c : ℝ}, ma ≤ mb ∧ mb ≤ mc ∧ a ≥ b ∧ b ≥ c)  :
  a * (-ma + mb + mc) + b * (ma - mb + mc) + c * (ma + mb - mc) ≥ 6 * Δ := 
sorry

end triangle_median_inequality_l308_308703


namespace typist_current_salary_l308_308299

-- Define the initial conditions as given in the problem
def initial_salary : ℝ := 6000
def raise_percentage : ℝ := 0.10
def reduction_percentage : ℝ := 0.05

-- Define the calculations for raised and reduced salaries
def raised_salary := initial_salary * (1 + raise_percentage)
def current_salary := raised_salary * (1 - reduction_percentage)

-- State the theorem to prove the current salary
theorem typist_current_salary : current_salary = 6270 := 
by
  -- Sorry is used to skip proof, overriding with the statement to ensure code builds successfully
  sorry

end typist_current_salary_l308_308299


namespace last_five_digits_l308_308296

theorem last_five_digits : (99 * 10101 * 111 * 1001) % 100000 = 88889 :=
by
  sorry

end last_five_digits_l308_308296


namespace grooming_time_correct_l308_308013

def time_to_groom_poodle : ℕ := 30
def time_to_groom_terrier : ℕ := time_to_groom_poodle / 2
def number_of_poodles : ℕ := 3
def number_of_terriers : ℕ := 8

def total_grooming_time : ℕ :=
  (number_of_poodles * time_to_groom_poodle) + (number_of_terriers * time_to_groom_terrier)

theorem grooming_time_correct :
  total_grooming_time = 210 :=
by
  sorry

end grooming_time_correct_l308_308013


namespace f_of_f_neg1_eq_neg1_f_x_eq_neg1_iff_x_eq_0_or_2_l308_308515

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then x^2 - 1 else -x + 1

-- Prove f[f(-1)] = -1
theorem f_of_f_neg1_eq_neg1 : f (f (-1)) = -1 := sorry

-- Prove that if f(x) = -1, then x = 0 or x = 2
theorem f_x_eq_neg1_iff_x_eq_0_or_2 (x : ℝ) : f x = -1 ↔ x = 0 ∨ x = 2 := sorry

end f_of_f_neg1_eq_neg1_f_x_eq_neg1_iff_x_eq_0_or_2_l308_308515


namespace nine_divides_a2_plus_ab_plus_b2_then_a_b_multiples_of_3_l308_308554

theorem nine_divides_a2_plus_ab_plus_b2_then_a_b_multiples_of_3
  (a b : ℤ)
  (h : 9 ∣ (a^2 + a * b + b^2)) :
  3 ∣ a ∧ 3 ∣ b :=
sorry

end nine_divides_a2_plus_ab_plus_b2_then_a_b_multiples_of_3_l308_308554


namespace num_dimes_l308_308611

/--
Given eleven coins consisting of pennies, nickels, dimes, quarters, and half-dollars,
having a total value of $1.43, with at least one coin of each type,
prove that there must be exactly 4 dimes.
-/
theorem num_dimes (p n d q h : ℕ) :
  1 ≤ p ∧ 1 ≤ n ∧ 1 ≤ d ∧ 1 ≤ q ∧ 1 ≤ h ∧ 
  p + n + d + q + h = 11 ∧ 
  (1 * p + 5 * n + 10 * d + 25 * q + 50 * h) = 143
  → d = 4 :=
by
  sorry

end num_dimes_l308_308611


namespace C_investment_l308_308174

theorem C_investment (A B total_profit A_share : ℝ) (x : ℝ) :
  A = 6300 → B = 4200 → total_profit = 12600 → A_share = 3780 →
  (A / (A + B + x) = A_share / total_profit) → x = 10500 :=
by
  intros hA hB h_total_profit h_A_share h_ratio
  sorry

end C_investment_l308_308174


namespace find_a_plus_b_l308_308866

theorem find_a_plus_b (a b : ℝ) (h1 : 2 * a = -6) (h2 : a^2 - b = 4) : a + b = 2 :=
by
  sorry

end find_a_plus_b_l308_308866


namespace jared_annual_earnings_l308_308507

open Nat

noncomputable def diploma_monthly_pay : ℕ := 4000
noncomputable def months_in_year : ℕ := 12
noncomputable def multiplier : ℕ := 3

theorem jared_annual_earnings :
  let jared_monthly_earnings := diploma_monthly_pay * multiplier
  let jared_annual_earnings := jared_monthly_earnings * months_in_year
  jared_annual_earnings = 144000 :=
by
  let jared_monthly_earnings := diploma_monthly_pay * multiplier
  let jared_annual_earnings := jared_monthly_earnings * months_in_year
  exact sorry

end jared_annual_earnings_l308_308507


namespace solve_for_x_l308_308953

theorem solve_for_x :
  ∃ x : ℝ, 5 * (x - 9) = 3 * (3 - 3 * x) + 9 ∧ x = 4.5 :=
by
  use 4.5
  sorry

end solve_for_x_l308_308953


namespace intersection_M_N_l308_308062

-- Define the sets M and N
def M : Set ℤ := {-1, 0, 1, 2}
def N : Set ℤ := {x | ∃ y ∈ M, |y| = x}

-- The main theorem to prove M ∩ N = {0, 1, 2}
theorem intersection_M_N : M ∩ N = {0, 1, 2} :=
by
  sorry

end intersection_M_N_l308_308062


namespace plant_height_after_year_l308_308771

theorem plant_height_after_year (current_height : ℝ) (monthly_growth : ℝ) (months_in_year : ℕ) (total_growth : ℝ)
  (h1 : current_height = 20)
  (h2 : monthly_growth = 5)
  (h3 : months_in_year = 12)
  (h4 : total_growth = monthly_growth * months_in_year) :
  current_height + total_growth = 80 :=
sorry

end plant_height_after_year_l308_308771


namespace num_words_with_consonant_l308_308077

-- Definitions
def letters : List Char := ['A', 'B', 'C', 'D', 'E']
def vowels : List Char := ['A', 'E']
def consonants : List Char := ['B', 'C', 'D']

-- Total number of 4-letter words without restrictions
def total_words : Nat := 5 ^ 4

-- Number of 4-letter words with only vowels
def vowels_only_words : Nat := 2 ^ 4

-- Number of 4-letter words with at least one consonant
def words_with_consonant : Nat := total_words - vowels_only_words

theorem num_words_with_consonant : words_with_consonant = 609 := by
  -- Add proof steps
  sorry

end num_words_with_consonant_l308_308077


namespace rosie_pies_proof_l308_308286

-- Define the given condition
def pies_per_apples (p: ℕ) (a: ℕ) : ℕ := a / p

-- Given that Rosie can make 3 pies from 12 apples
def given_condition : pies_per_apples 3 12 = 4 := rfl

-- The proof problem statement:
theorem rosie_pies_proof : ∀ (a: ℕ) (n: ℕ), pies_per_apples 3 12 = 4 → pies_per_apples n a = 4 → pies_per_apples n a = 9 :=
begin
  sorry
end

end rosie_pies_proof_l308_308286


namespace min_abc_value_l308_308406

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem min_abc_value
  (a b c : ℕ)
  (h1: is_prime a)
  (h2 : is_prime b)
  (h3 : is_prime c)
  (h4 : a^5 ∣ (b^2 - c))
  (h5 : ∃ k : ℕ, (b + c) = k^2) :
  a * b * c = 1958 := sorry

end min_abc_value_l308_308406


namespace rosie_pies_proof_l308_308288

-- Define the given condition
def pies_per_apples (p: ℕ) (a: ℕ) : ℕ := a / p

-- Given that Rosie can make 3 pies from 12 apples
def given_condition : pies_per_apples 3 12 = 4 := rfl

-- The proof problem statement:
theorem rosie_pies_proof : ∀ (a: ℕ) (n: ℕ), pies_per_apples 3 12 = 4 → pies_per_apples n a = 4 → pies_per_apples n a = 9 :=
begin
  sorry
end

end rosie_pies_proof_l308_308288


namespace complex_modulus_l308_308862

noncomputable def z : ℂ := (1 + 3 * Complex.I) / (1 + Complex.I)

theorem complex_modulus 
  (h : (1 + Complex.I) * z = 1 + 3 * Complex.I) : 
  Complex.abs (z^2) = 5 := 
by
  sorry

end complex_modulus_l308_308862


namespace add_zero_eq_self_l308_308898

theorem add_zero_eq_self (n x : ℤ) (h : n + x = n) : x = 0 := 
sorry

end add_zero_eq_self_l308_308898


namespace probability_three_common_books_l308_308720

open Finset

/-- Probability that Jenna and Marco select exactly 3 common books out of 12 -/
theorem probability_three_common_books :
  let total_ways := (choose 12 4) * (choose 12 4),
      successful_ways := (choose 12 3) * (choose 9 1) * (choose 8 1) in
  (successful_ways : ℚ) / total_ways = (32 / 495 : ℚ) :=
by
  sorry

end probability_three_common_books_l308_308720


namespace speed_of_water_l308_308476

-- Definitions based on conditions
def swim_speed_in_still_water : ℝ := 4
def distance_against_current : ℝ := 6
def time_against_current : ℝ := 3
def effective_speed (v : ℝ) : ℝ := swim_speed_in_still_water - v

-- Theorem to prove the speed of the water
theorem speed_of_water (v : ℝ) : 
  effective_speed v * time_against_current = distance_against_current → 
  v = 2 :=
by
  sorry

end speed_of_water_l308_308476


namespace roots_of_polynomial_in_range_l308_308693

theorem roots_of_polynomial_in_range (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 < -1 ∧ x2 > 1 ∧ x1 * x2 = m^2 - 2 ∧ (x1 + x2) = -(m - 1)) 
  -> 0 < m ∧ m < 1 :=
by
  sorry

end roots_of_polynomial_in_range_l308_308693


namespace circle_center_and_radius_l308_308354

def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 4

theorem circle_center_and_radius :
  (∀ x y : ℝ, circle_eq x y ↔ (x - 2) ^ 2 + y ^ 2 = 4) →
  (exists (h k r : ℝ), (h, k) = (2, 0) ∧ r = 2) :=
by
  sorry

end circle_center_and_radius_l308_308354


namespace problem_statement_l308_308064

theorem problem_statement :
  (¬ (∀ x : ℝ, 2 * x < 3 * x)) ∧ (∃ x : ℝ, x^3 = 1 - x^2) := 
sorry

end problem_statement_l308_308064


namespace multiples_of_9_ending_in_5_l308_308380

theorem multiples_of_9_ending_in_5 (n : ℕ) :
  (∃ k : ℕ, n = 9 * k ∧ 0 < n ∧ n < 600 ∧ n % 10 = 5) → 
  ∃ l, l = 7 := 
by
sorry

end multiples_of_9_ending_in_5_l308_308380


namespace total_collection_value_l308_308946

theorem total_collection_value (total_stickers : ℕ) (partial_stickers : ℕ) (partial_value : ℕ)
  (same_value : ∀ (stickers : ℕ), stickers = total_stickers → stickers * partial_value / partial_stickers = stickers * (partial_value / partial_stickers)):
  partial_value = 24 ∧ partial_stickers = 6 ∧ total_stickers = 18 → total_stickers * (partial_value / partial_stickers) = 72 :=
by {
  sorry
}

end total_collection_value_l308_308946


namespace A_and_C_mutually_exclusive_l308_308220

/-- Definitions for the problem conditions. -/
def A (all_non_defective : Prop) : Prop := all_non_defective
def B (all_defective : Prop) : Prop := all_defective
def C (at_least_one_defective : Prop) : Prop := at_least_one_defective

/-- Theorem stating that A and C are mutually exclusive. -/
theorem A_and_C_mutually_exclusive (all_non_defective at_least_one_defective : Prop) :
  A all_non_defective ∧ C at_least_one_defective → false :=
  sorry

end A_and_C_mutually_exclusive_l308_308220


namespace decrease_in_demand_l308_308165

theorem decrease_in_demand (init_price new_price demand : ℝ) (init_demand : ℕ) (price_increase : ℝ) (original_revenue new_demand : ℝ) :
  init_price = 20 ∧ init_demand = 500 ∧ price_increase = 5 ∧ demand = init_price + price_increase ∧ 
  original_revenue = init_price * init_demand ∧ new_demand ≤ init_demand ∧ 
  new_demand * demand ≥ original_revenue → 
  init_demand - new_demand = 100 :=
by 
  sorry

end decrease_in_demand_l308_308165


namespace austin_hours_on_mondays_l308_308487

-- Define the conditions
def earning_per_hour : ℕ := 5
def hours_wednesday : ℕ := 1
def hours_friday : ℕ := 3
def weeks : ℕ := 6
def bicycle_cost : ℕ := 180

-- Define the proof problem
theorem austin_hours_on_mondays (M : ℕ) :
  earning_per_hour * weeks * (M + hours_wednesday + hours_friday) = bicycle_cost → M = 2 :=
by 
  intro h
  sorry

end austin_hours_on_mondays_l308_308487


namespace tangent_line_at_P_range_of_a_l308_308527

-- Define the function f(x)
noncomputable def f (x a : ℝ) : ℝ := a * (x - 1/x) - Real.log x

-- Problem (Ⅰ): Tangent line equation at P(1, f(1)) for a = 1
theorem tangent_line_at_P (x : ℝ) (h : x = 1) : (∃ y : ℝ, f x 1 = y ∧ x - y - 1 = 0) := sorry

-- Problem (Ⅱ): Range of a for f(x) ≥ 0 ∀ x ≥ 1
theorem range_of_a {a : ℝ} (h : ∀ x : ℝ, x ≥ 1 → f x a ≥ 0) : a ≥ 1/2 := sorry

end tangent_line_at_P_range_of_a_l308_308527


namespace factor_expression_l308_308791

theorem factor_expression (x : ℝ) :
  (3*x^3 + 48*x^2 - 14) - (-9*x^3 + 2*x^2 - 14) =
  2*x^2 * (6*x + 23) :=
by
  sorry

end factor_expression_l308_308791


namespace gemma_amount_given_l308_308512

theorem gemma_amount_given
  (cost_per_pizza : ℕ)
  (number_of_pizzas : ℕ)
  (tip : ℕ)
  (change_back : ℕ)
  (h1 : cost_per_pizza = 10)
  (h2 : number_of_pizzas = 4)
  (h3 : tip = 5)
  (h4 : change_back = 5) :
  number_of_pizzas * cost_per_pizza + tip + change_back = 50 := sorry

end gemma_amount_given_l308_308512


namespace mailman_junk_mail_l308_308149

/-- 
  Given:
    - n = 640 : total number of pieces of junk mail for the block
    - h = 20 : number of houses in the block
  
  Prove:
    - The number of pieces of junk mail given to each house equals 32, when the total number of pieces of junk mail is divided by the number of houses.
--/
theorem mailman_junk_mail (n h : ℕ) (h_total : n = 640) (h_houses : h = 20) :
  n / h = 32 :=
by
  sorry

end mailman_junk_mail_l308_308149


namespace shaded_area_of_circles_l308_308937

theorem shaded_area_of_circles (r1 r2 : ℝ) (h1 : r1 = 4) (h2 : r2 = 5) :
  let R := r1 + r2 in
  let area_large_circle := π * R^2 in
  let area_small_circle1 := π * r1^2 in
  let area_small_circle2 := π * r2^2 in
  area_large_circle - area_small_circle1 - area_small_circle2 = 40 * π :=
by
  sorry

end shaded_area_of_circles_l308_308937


namespace plates_probability_l308_308977

noncomputable def number_of_plates := 12
noncomputable def red_plates := 6
noncomputable def light_blue_plates := 3
noncomputable def dark_blue_plates := 3
noncomputable def total_pairs := number_of_plates * (number_of_plates - 1) / 2
noncomputable def red_pairs := red_plates * (red_plates - 1) / 2
noncomputable def light_blue_pairs := light_blue_plates * (light_blue_plates - 1) / 2
noncomputable def dark_blue_pairs := dark_blue_plates * (dark_blue_plates - 1) / 2
noncomputable def mixed_blue_pairs := light_blue_plates * dark_blue_plates
noncomputable def total_satisfying_pairs := red_pairs + light_blue_pairs + dark_blue_pairs + mixed_blue_pairs
noncomputable def desired_probability := (total_satisfying_pairs : ℚ) / total_pairs

theorem plates_probability :
  desired_probability = 5 / 11 :=
by
  -- Add the proof here
  sorry

end plates_probability_l308_308977


namespace min_shots_to_hit_terrorist_l308_308621

theorem min_shots_to_hit_terrorist : ∀ terrorist_position : ℕ, (1 ≤ terrorist_position ∧ terrorist_position ≤ 10) →
  ∃ shots : ℕ, shots ≥ 6 ∧ (∀ move : ℕ, (shots - move) ≥ 1 → (terrorist_position + move ≤ 10 → terrorist_position % 2 = move % 2)) :=
by
  sorry

end min_shots_to_hit_terrorist_l308_308621


namespace floor_e_equals_2_l308_308213

theorem floor_e_equals_2 : Int.floor Real.exp = 2 := 
sorry

end floor_e_equals_2_l308_308213


namespace work_completion_days_l308_308160

theorem work_completion_days (A B : Type) (A_work_rate B_work_rate : ℝ) :
  (1 / 16 : ℝ) = (1 / 20) + A_work_rate → B_work_rate = (1 / 80) := by
  sorry

end work_completion_days_l308_308160


namespace required_moles_h2so4_l308_308241

-- Defining chemical equation conditions
def balanced_reaction (nacl h2so4 hcl nahso4 : ℕ) : Prop :=
  nacl = h2so4 ∧ hcl = nacl ∧ nahso4 = nacl

-- Theorem statement
theorem required_moles_h2so4 (nacl_needed moles_h2so4 : ℕ) (hcl_produced nahso4_produced : ℕ)
  (h : nacl_needed = 2 ∧ balanced_reaction nacl_needed moles_h2so4 hcl_produced nahso4_produced) :
  moles_h2so4 = 2 :=
  sorry

end required_moles_h2so4_l308_308241


namespace monotonic_decreasing_interval_l308_308438

noncomputable def f (x : ℝ) : ℝ := x^3 - 15 * x^2 - 33 * x + 6

theorem monotonic_decreasing_interval :
  {x : ℝ | ∃ t ∈ Ioo (-1 : ℝ) 11, t = x} ⊆ {x : ℝ | ∃ t, f'(t) < 0} :=
sorry

end monotonic_decreasing_interval_l308_308438


namespace profit_percentage_is_4_l308_308758

-- Define the cost price and selling price
def cost_price : Nat := 600
def selling_price : Nat := 624

-- Calculate profit in dollars
def profit_dollars : Nat := selling_price - cost_price

-- Calculate profit percentage
def profit_percentage : Nat := (profit_dollars * 100) / cost_price

-- Prove that the profit percentage is 4%
theorem profit_percentage_is_4 : profit_percentage = 4 := by
  sorry

end profit_percentage_is_4_l308_308758


namespace value_of_c_in_base8_perfect_cube_l308_308658

theorem value_of_c_in_base8_perfect_cube (c : ℕ) (h : 0 ≤ c ∧ c < 8) :
  4 * 8^2 + c * 8 + 3 = x^3 → c = 0 := by
  sorry

end value_of_c_in_base8_perfect_cube_l308_308658


namespace probability_of_three_blue_beans_l308_308323

-- Define the conditions
def red_jellybeans : ℕ := 10 
def blue_jellybeans : ℕ := 10 
def total_jellybeans : ℕ := red_jellybeans + blue_jellybeans 
def draws : ℕ := 3 

-- Define the events
def P_first_blue : ℚ := blue_jellybeans / total_jellybeans 
def P_second_blue : ℚ := (blue_jellybeans - 1) / (total_jellybeans - 1) 
def P_third_blue : ℚ := (blue_jellybeans - 2) / (total_jellybeans - 2) 
def P_all_three_blue : ℚ := P_first_blue * P_second_blue * P_third_blue 

-- Define the correct answer
def correct_probability : ℚ := 1 / 9.5 

-- State the theorem
theorem probability_of_three_blue_beans : 
  P_all_three_blue = correct_probability := 
sorry

end probability_of_three_blue_beans_l308_308323


namespace value_of_a_l308_308959

theorem value_of_a (a : ℝ) : (∀ x : ℝ, 2 * x^2 + x + a^2 - 1 = 0 → x = 0) → (a = 1 ∨ a = -1) :=
by
  sorry

end value_of_a_l308_308959


namespace student_most_stable_l308_308597

theorem student_most_stable (A B C : ℝ) (hA : A = 0.024) (hB : B = 0.08) (hC : C = 0.015) : C < A ∧ C < B := by
  sorry

end student_most_stable_l308_308597


namespace determine_guilty_resident_l308_308877

structure IslandResident where
  name : String
  is_guilty : Bool
  is_knight : Bool
  is_liar : Bool
  is_normal : Bool -- derived condition: ¬is_knight ∧ ¬is_liar

def A : IslandResident := { name := "A", is_guilty := false, is_knight := false, is_liar := false, is_normal := true }
def B : IslandResident := { name := "B", is_guilty := true, is_knight := true, is_liar := false, is_normal := false }
def C : IslandResident := { name := "C", is_guilty := false, is_knight := false, is_liar := true, is_normal := false }

-- Condition: Only one of them is guilty.
def one_guilty (A B C : IslandResident) : Prop :=
  A.is_guilty ≠ B.is_guilty ∧ A.is_guilty ≠ C.is_guilty ∧ B.is_guilty ≠ C.is_guilty ∧ (A.is_guilty ∨ B.is_guilty ∨ C.is_guilty)

-- Condition: The guilty one is a knight.
def guilty_is_knight (A B C : IslandResident) : Prop :=
  (A.is_guilty → A.is_knight) ∧ (B.is_guilty → B.is_knight) ∧ (C.is_guilty → C.is_knight)

-- Statements made by each resident.
def statements_made (A B C : IslandResident) : Prop :=
  (A.is_guilty = false) ∧ (B.is_guilty = false) ∧ (B.is_normal = false)

theorem determine_guilty_resident (A B C : IslandResident) :
  one_guilty A B C →
  guilty_is_knight A B C →
  statements_made A B C →
  B.is_guilty ∧ B.is_knight :=
by
  sorry

end determine_guilty_resident_l308_308877


namespace bob_pennies_l308_308082

theorem bob_pennies (a b : ℤ) (h1 : b + 1 = 4 * (a - 1)) (h2 : b - 1 = 3 * (a + 1)) : b = 31 :=
by
  sorry

end bob_pennies_l308_308082


namespace intersection_M_N_l308_308973

-- Define set M
def set_M : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}

-- Define set N
def set_N : Set ℤ := {x | ∃ k : ℕ, k > 0 ∧ x = 2 * k - 1}

-- Define the intersection of M and N
def M_intersect_N : Set ℤ := {1, 3}

-- The theorem to prove
theorem intersection_M_N : set_M ∩ set_N = M_intersect_N :=
by sorry

end intersection_M_N_l308_308973


namespace compare_minus_abs_val_l308_308493

theorem compare_minus_abs_val :
  -|(-8)| < -6 := 
sorry

end compare_minus_abs_val_l308_308493


namespace correct_calculation_C_l308_308754

theorem correct_calculation_C (a b y x : ℝ) : 
  (7 * a + a ≠ 8 * a^2) ∧ 
  (5 * y - 3 * y ≠ 2) ∧ 
  (3 * x^2 * y - 2 * x^2 * y = x^2 * y) ∧ 
  (3 * a + 2 * b ≠ 5 * a * b) :=
by {
  split,
  { exact sorry, },
  split,
  { exact sorry, },
  split,
  { exact sorry, },
  { exact sorry, },
}

end correct_calculation_C_l308_308754


namespace range_of_m_l308_308804

theorem range_of_m (m : ℝ) :
  (∃ (x : ℤ), (x > 2 * m ∧ x ≥ m - 3) ∧ x = 1) ↔ 0 ≤ m ∧ m < 0.5 :=
by
  sorry

end range_of_m_l308_308804


namespace smaller_number_is_72_l308_308308

theorem smaller_number_is_72
  (x : ℝ)
  (h1 : (3 * x - 24) / (8 * x - 24) = 4 / 9)
  : 3 * x = 72 :=
sorry

end smaller_number_is_72_l308_308308


namespace rubies_in_chest_l308_308180

theorem rubies_in_chest (R : ℕ) (h₁ : 421 = R + 44) : R = 377 :=
by 
  sorry

end rubies_in_chest_l308_308180


namespace bryan_push_ups_l308_308788

theorem bryan_push_ups (sets : ℕ) (push_ups_per_set : ℕ) (fewer_in_last_set : ℕ) 
  (h1 : sets = 3) (h2 : push_ups_per_set = 15) (h3 : fewer_in_last_set = 5) :
  (sets - 1) * push_ups_per_set + (push_ups_per_set - fewer_in_last_set) = 40 := by 
  -- We are setting sorry here to skip the proof.
  sorry

end bryan_push_ups_l308_308788


namespace total_pies_l308_308289

def apple_Pies (totalApples : ℕ) (applesPerPie : ℕ) (piesPerBatch : ℕ) : ℕ :=
  (totalApples / applesPerPie) * piesPerBatch

def pear_Pies (totalPears : ℕ) (pearsPerPie : ℕ) (piesPerBatch : ℕ) : ℕ :=
  (totalPears / pearsPerPie) * piesPerBatch

theorem total_pies :
  let apples : ℕ := 27
  let pears : ℕ := 30
  let applesPerPie : ℕ := 9
  let pearsPerPie : ℕ := 15
  let applePiesPerBatch : ℕ := 2
  let pearPiesPerBatch : ℕ := 3
  apple_Pies apples applesPerPie applePiesPerBatch + pear_Pies pears pearsPerPie pearPiesPerBatch = 12 :=
by
  sorry

end total_pies_l308_308289


namespace problem_1_solution_problem_2_solution_l308_308638

noncomputable def problem_1 : Real :=
  (-3) + (2 - Real.pi)^0 - (1 / 2)⁻¹

theorem problem_1_solution :
  problem_1 = -4 :=
by
  sorry

noncomputable def problem_2 (a : Real) : Real :=
  (2 * a)^3 - a * a^2 + 3 * a^6 / a^3

theorem problem_2_solution (a : Real) :
  problem_2 a = 10 * a^3 :=
by
  sorry

end problem_1_solution_problem_2_solution_l308_308638


namespace pave_square_with_tiles_l308_308736

theorem pave_square_with_tiles (b c : ℕ) (h_right_triangle : (b > 0) ∧ (c > 0)) :
  (∃ (k : ℕ), k^2 = b^2 + c^2) ↔ (∃ (m n : ℕ), m * c * b = 2 * n^2 * (b^2 + c^2)) := 
sorry

end pave_square_with_tiles_l308_308736


namespace sufficiency_s_for_q_l308_308066

variables {q r s : Prop}

theorem sufficiency_s_for_q (h₁ : r → q) (h₂ : ¬(q → r)) (h₃ : r ↔ s) : s → q ∧ ¬(q → s) :=
by
  sorry

end sufficiency_s_for_q_l308_308066


namespace solve_equation1_solve_equation2_solve_system1_solve_system2_l308_308574

-- Problem 1
theorem solve_equation1 (x : ℚ) : 3 * (x + 8) - 5 = 6 * (2 * x - 1) → x = 25 / 9 :=
by sorry

-- Problem 2
theorem solve_equation2 (x : ℚ) : (3 * x - 2) / 2 = (4 * x + 2) / 3 - 1 → x = 4 :=
by sorry

-- Problem 3
theorem solve_system1 (x y : ℚ) : (3 * x - 7 * y = 8) ∧ (2 * x + y = 11) → x = 5 ∧ y = 1 :=
by sorry

-- Problem 4
theorem solve_system2 (a b c : ℚ) : (a - b + c = 0) ∧ (4 * a + 2 * b + c = 3) ∧ (25 * a + 5 * b + c = 60) → (a = 3) ∧ (b = -2) ∧ (c = -5) :=
by sorry

end solve_equation1_solve_equation2_solve_system1_solve_system2_l308_308574


namespace kids_have_equal_eyes_l308_308101

theorem kids_have_equal_eyes (mom_eyes dad_eyes kids_num total_eyes kids_eyes : ℕ) 
  (h_mom_eyes : mom_eyes = 1) 
  (h_dad_eyes : dad_eyes = 3) 
  (h_kids_num : kids_num = 3) 
  (h_total_eyes : total_eyes = 16) 
  (h_family_eyes : mom_eyes + dad_eyes + kids_num * kids_eyes = total_eyes) :
  kids_eyes = 4 :=
by
  sorry

end kids_have_equal_eyes_l308_308101


namespace b2_b7_product_l308_308109

variable {b : ℕ → ℤ}

-- Define the conditions: b is an arithmetic sequence and b_4 * b_5 = 15
def is_arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

axiom increasing_arithmetic_sequence : is_arithmetic_sequence b
axiom b4_b5_product : b 4 * b 5 = 15

-- The target theorem to prove
theorem b2_b7_product : b 2 * b 7 = -9 :=
sorry

end b2_b7_product_l308_308109


namespace floor_e_eq_two_l308_308217

theorem floor_e_eq_two
  (e_approx : Real ≈ 2.718) :
  ⌊e⌋ = 2 :=
sorry

end floor_e_eq_two_l308_308217


namespace trigo_identity_l308_308382

variable (α : ℝ)

theorem trigo_identity (h : Real.sin (Real.pi / 6 - α) = 1 / 3) :
  Real.cos (Real.pi / 6 + α / 2) ^ 2 = 2 / 3 := by
  sorry

end trigo_identity_l308_308382


namespace total_fireworks_l308_308548

-- Definitions based on conditions
def kobys_boxes := 2
def kobys_sparklers_per_box := 3
def kobys_whistlers_per_box := 5
def cheries_boxes := 1
def cheries_sparklers_per_box := 8
def cheries_whistlers_per_box := 9

-- Calculations
def total_kobys_fireworks := kobys_boxes * (kobys_sparklers_per_box + kobys_whistlers_per_box)
def total_cheries_fireworks := cheries_boxes * (cheries_sparklers_per_box + cheries_whistlers_per_box)

-- Theorem
theorem total_fireworks : total_kobys_fireworks + total_cheries_fireworks = 33 := 
by
  -- Can be elaborated and filled in with steps, if necessary.
  sorry

end total_fireworks_l308_308548


namespace xyz_cubic_expression_l308_308376

theorem xyz_cubic_expression (x y z a b c : ℝ) (h1 : x * y = a) (h2 : x * z = b) (h3 : y * z = c) (h4 : x ≠ 0) (h5 : y ≠ 0) (h6 : z ≠ 0) (h7 : a ≠ 0) (h8 : b ≠ 0) (h9 : c ≠ 0) :
  x^3 + y^3 + z^3 = (a^3 + b^3 + c^3) / (a * b * c) :=
by
  sorry

end xyz_cubic_expression_l308_308376


namespace owen_profit_l308_308413

theorem owen_profit
  (num_boxes : ℕ)
  (cost_per_box : ℕ)
  (pieces_per_box : ℕ)
  (sold_boxes : ℕ)
  (price_per_25_pieces : ℕ)
  (remaining_pieces : ℕ)
  (price_per_10_pieces : ℕ) :
  num_boxes = 12 →
  cost_per_box = 9 →
  pieces_per_box = 50 →
  sold_boxes = 6 →
  price_per_25_pieces = 5 →
  remaining_pieces = 300 →
  price_per_10_pieces = 3 →
  sold_boxes * 2 * price_per_25_pieces + (remaining_pieces / 10) * price_per_10_pieces - num_boxes * cost_per_box = 42 :=
by
  intros h_num h_cost h_pieces h_sold h_price_25 h_remain h_price_10
  sorry

end owen_profit_l308_308413


namespace max_sum_of_squares_diff_l308_308744

theorem max_sum_of_squares_diff {x y : ℕ} (h : x > 0 ∧ y > 0) (h_diff : x^2 - y^2 = 2016) :
  x + y ≤ 1008 ∧ ∃ x' y' : ℕ, x'^2 - y'^2 = 2016 ∧ x' + y' = 1008 :=
sorry

end max_sum_of_squares_diff_l308_308744


namespace read_books_correct_l308_308019

namespace CrazySillySchool

-- Definitions from conditions
def total_books : Nat := 20
def unread_books : Nat := 5
def read_books : Nat := total_books - unread_books

-- Theorem statement
theorem read_books_correct : read_books = 15 :=
by
  -- Mathematical statement that follows from conditions and correct answer
  sorry

end CrazySillySchool

end read_books_correct_l308_308019


namespace f_at_2023_l308_308999

noncomputable def f (a x : ℝ) : ℝ := (a - x) / (a + 2 * x)

noncomputable def g (a x : ℝ) : ℝ := (f a (x - 2023)) + (1 / 2)

def is_odd (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = -g x

variable (a : ℝ)
variable (h_a : a ≠ 0)
variable (h_odd : is_odd (g a))

theorem f_at_2023 : f a 2023 = 1 / 4 :=
sorry

end f_at_2023_l308_308999


namespace smallest_m_for_integral_roots_l308_308458

theorem smallest_m_for_integral_roots :
  ∃ m : ℕ, (∀ x : ℚ, 12 * x^2 - m * x + 360 = 0 → x.den = 1) ∧ 
           (∀ k : ℕ, k < m → ¬∀ x : ℚ, 12 * x^2 - k * x + 360 = 0 → x.den = 1) :=  
begin
  sorry
end

end smallest_m_for_integral_roots_l308_308458


namespace monotonic_decreasing_interval_l308_308437

noncomputable def f (x : ℝ) : ℝ := x^3 - 15 * x^2 - 33 * x + 6

theorem monotonic_decreasing_interval :
  ∃ (a b : ℝ), a = -1 ∧ b = 11 ∧ ∀ x, x > a ∧ x < b → (deriv f x) < 0 :=
by
  sorry

end monotonic_decreasing_interval_l308_308437


namespace a_n_formula_b_n_formula_S_n_formula_l308_308519

noncomputable def a_n (n : ℕ) : ℕ := 3 * n
noncomputable def b_n (n : ℕ) : ℕ := 2^(n-1) + 3 * n
noncomputable def S_n (n : ℕ) : ℕ := 2^n - 1 + (3 * n^2 + 3 * n) / 2

theorem a_n_formula (n : ℕ) : a_n n = 3 * n := by
  unfold a_n
  rfl

theorem b_n_formula (n : ℕ) : b_n n = 2^(n-1) + 3 * n := by
  unfold b_n
  rfl

theorem S_n_formula (n : ℕ) : S_n n = 2^n - 1 + (3 * n^2 + 3 * n) / 2 := by
  unfold S_n
  rfl

end a_n_formula_b_n_formula_S_n_formula_l308_308519


namespace find_area_MOI_l308_308701

noncomputable def incenter_coords (a b c : ℝ) (A B C : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((a * A.1 + b * B.1 + c * C.1) / (a + b + c), (a * A.2 + b * B.2 + c * C.2) / (a + b + c))

noncomputable def shoelace_area (P Q R : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

theorem find_area_MOI :
  let A := (0, 0)
  let B := (8, 0)
  let C := (0, 17)
  let O := (4, 8.5)
  let I := incenter_coords 8 15 17 A B C
  let M := (6.25, 6.25)
  shoelace_area M O I = 25.78125 :=
by
  sorry

end find_area_MOI_l308_308701


namespace missing_number_is_eight_l308_308421

theorem missing_number_is_eight (x : ℤ) : (4 + 3) + (x - 3 - 1) = 11 → x = 8 := by
  intro h
  sorry

end missing_number_is_eight_l308_308421


namespace problem_l308_308357

-- Define the matrix
def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![2, 5, 0], ![0, 2, 3], ![3, 0, 2]]

-- Define the condition that there exists a nonzero vector v such that A * v = k * v
def exists_eigenvector (k : ℝ) : Prop :=
  ∃ (v : Fin 3 → ℝ), v ≠ 0 ∧ A.mulVec v = k • v

theorem problem : ∀ (k : ℝ), exists_eigenvector k ↔ (k = 2 + (45)^(1/3)) :=
sorry

end problem_l308_308357


namespace arrows_from_530_to_533_l308_308662

-- Define what it means for the pattern to be cyclic with period 5
def cycle_period (n m : Nat) : Prop := n % m = 0

-- Define the equivalent points on the circular track
def equiv_point (n : Nat) (m : Nat) : Nat := n % m

-- Given conditions
def arrow_pattern : Prop :=
  ∀ n : Nat, cycle_period n 5 ∧
  (equiv_point 530 5 = 0) ∧ (equiv_point 533 5 = 3)

-- The theorem to be proved
theorem arrows_from_530_to_533 :
  (∃ seq : List (Nat × Nat),
    seq = [(0, 1), (1, 2), (2, 3)]) :=
sorry

end arrows_from_530_to_533_l308_308662


namespace edward_earnings_l308_308498

theorem edward_earnings
    (total_lawns : ℕ := 17)
    (forgotten_lawns : ℕ := 9)
    (total_earnings : ℕ := 32) :
    (total_earnings / (total_lawns - forgotten_lawns) = 4) :=
by
  sorry

end edward_earnings_l308_308498


namespace choir_members_count_l308_308300

theorem choir_members_count : 
  ∃ n : ℕ, 120 ≤ n ∧ n ≤ 300 ∧
    n % 6 = 1 ∧
    n % 8 = 5 ∧
    n % 9 = 2 ∧
    n = 241 :=
by
  -- Proof will follow
  sorry

end choir_members_count_l308_308300


namespace weather_forecast_minutes_l308_308008

theorem weather_forecast_minutes 
  (total_duration : ℕ) 
  (national_news : ℕ) 
  (international_news : ℕ) 
  (sports : ℕ) 
  (advertising : ℕ) 
  (wf : ℕ) :
  total_duration = 30 →
  national_news = 12 →
  international_news = 5 →
  sports = 5 →
  advertising = 6 →
  total_duration - (national_news + international_news + sports + advertising) = wf →
  wf = 2 :=
by
  intros
  sorry

end weather_forecast_minutes_l308_308008


namespace no_monochromatic_10_term_progression_l308_308001

def can_color_without_monochromatic_progression (n k : ℕ) (c : Fin n → Fin k) : Prop :=
  ∀ (a d : ℕ), (a < n) → (a + (9 * d) < n) → (∀ i : ℕ, i < 10 → c ⟨a + (i * d), sorry⟩ = c ⟨a, sorry⟩) → 
    (∃ j i : ℕ, j < 10 ∧ i < 10 ∧ c ⟨a + (i * d), sorry⟩ ≠ c ⟨a + (j * d), sorry⟩)

theorem no_monochromatic_10_term_progression :
  ∃ c : Fin 2008 → Fin 4, can_color_without_monochromatic_progression 2008 4 c :=
sorry

end no_monochromatic_10_term_progression_l308_308001


namespace find_a_and_b_l308_308110

noncomputable def f (x: ℝ) (b: ℝ): ℝ := x^2 + 5*x + b
noncomputable def g (x: ℝ) (b: ℝ): ℝ := 2*b*x + 3

theorem find_a_and_b (a b: ℝ):
  (∀ x: ℝ, f (g x b) b = a * x^2 + 30 * x + 24) →
  a = 900 / 121 ∧ b = 15 / 11 :=
by
  intro H
  -- Proof is omitted as requested
  sorry

end find_a_and_b_l308_308110


namespace parabola_sum_is_neg_fourteen_l308_308007

noncomputable def parabola_sum (a b c : ℝ) : ℝ := a + b + c

theorem parabola_sum_is_neg_fourteen :
  ∃ (a b c : ℝ), 
    (∀ x : ℝ, a * x^2 + b * x + c = -(x + 3)^2 + 2) ∧
    ((-1)^2 = a * (-1 + 3)^2 + 6) ∧ 
    ((-3)^2 = a * (-3 + 3)^2 + 2) ∧
    (parabola_sum a b c = -14) :=
sorry

end parabola_sum_is_neg_fourteen_l308_308007


namespace mike_corvette_average_speed_l308_308715

theorem mike_corvette_average_speed
  (D : ℚ) (v : ℚ) (total_distance : ℚ)
  (first_half_distance : ℚ) (second_half_time_ratio : ℚ)
  (total_time : ℚ) (average_rate : ℚ) :
  total_distance = 640 ∧
  first_half_distance = total_distance / 2 ∧
  second_half_time_ratio = 3 ∧
  average_rate = 40 →
  v = 80 :=
by
  intros h
  have total_distance_eq : total_distance = 640 := h.1
  have first_half_distance_eq : first_half_distance = total_distance / 2 := h.2.1
  have second_half_time_ratio_eq : second_half_time_ratio = 3 := h.2.2.1
  have average_rate_eq : average_rate = 40 := h.2.2.2
  sorry

end mike_corvette_average_speed_l308_308715


namespace number_of_integers_satisfying_l308_308247

theorem number_of_integers_satisfying (k1 k2 : ℕ) (hk1 : k1 = 300) (hk2 : k2 = 1000) :
  ∃ m : ℕ, m = 14 ∧ ∀ n : ℕ, 300 < n^2 → n^2 < 1000 → 18 ≤ n ∧ n ≤ 31 :=
by
  use 14
  sorry

end number_of_integers_satisfying_l308_308247


namespace max_value_negative_one_l308_308383

theorem max_value_negative_one (f : ℝ → ℝ) (hx : ∀ x, x < 1 → f x ≤ -1) :
  ∀ x, x < 1 → ∃ M, (∀ y, y < 1 → f y ≤ M) ∧ f x = M :=
sorry

end max_value_negative_one_l308_308383


namespace combined_height_difference_is_correct_l308_308839

-- Define the initial conditions
def uncle_height : ℕ := 72
def james_initial_height : ℕ := (2 * uncle_height) / 3
def sarah_initial_height : ℕ := (3 * james_initial_height) / 4

-- Define the growth spurts
def james_growth_spurt : ℕ := 10
def sarah_growth_spurt : ℕ := 12

-- Define their heights after growth spurts
def james_final_height : ℕ := james_initial_height + james_growth_spurt
def sarah_final_height : ℕ := sarah_initial_height + sarah_growth_spurt

-- Define the combined height of James and Sarah after growth spurts
def combined_height : ℕ := james_final_height + sarah_final_height

-- Define the combined height difference between uncle and both James and Sarah now
def combined_height_difference : ℕ := combined_height - uncle_height

-- Lean statement to prove the combined height difference
theorem combined_height_difference_is_correct : combined_height_difference = 34 := by
  -- proof omitted
  sorry

end combined_height_difference_is_correct_l308_308839


namespace class_avg_GPA_l308_308142

theorem class_avg_GPA (n : ℕ) (h1 : n > 0) : 
  ((1 / 4 : ℝ) * 92 + (3 / 4 : ℝ) * 76 = 80) :=
sorry

end class_avg_GPA_l308_308142


namespace find_a_and_b_l308_308436

variable {x : ℝ}

/-- The problem statement: Given the function y = b + a * sin x (with a < 0), and the maximum value is -1, and the minimum value is -5,
    find the values of a and b. --/
theorem find_a_and_b (a b : ℝ) (h : a < 0) 
  (h1 : ∀ x, b + a * Real.sin x ≤ -1)
  (h2 : ∀ x, b + a * Real.sin x ≥ -5) : 
  a = -2 ∧ b = -3 := sorry

end find_a_and_b_l308_308436


namespace race_track_width_l308_308143

noncomputable def width_of_race_track (C_inner : ℝ) (r_outer : ℝ) : ℝ :=
  let r_inner := C_inner / (2 * Real.pi)
  r_outer - r_inner

theorem race_track_width : 
  width_of_race_track 880 165.0563499208679 = 25.0492072460867 :=
by
  sorry

end race_track_width_l308_308143


namespace marion_score_is_correct_l308_308868

-- Definition of the problem conditions
def exam_total_items := 40
def ella_incorrect_answers := 4

-- Calculate Ella's score
def ella_score := exam_total_items - ella_incorrect_answers

-- Calculate half of Ella's score
def half_ella_score := ella_score / 2

-- Marion's score is 6 more than half of Ella's score
def marion_score := half_ella_score + 6

-- The theorem we need to prove
theorem marion_score_is_correct : marion_score = 24 := by
  sorry

end marion_score_is_correct_l308_308868


namespace inequality_solution_set_l308_308301

theorem inequality_solution_set (x : ℝ) : (-2 < x ∧ x ≤ 3) ↔ (x - 3) / (x + 2) ≤ 0 := 
sorry

end inequality_solution_set_l308_308301


namespace count_two_digit_integers_with_perfect_square_sum_l308_308976

def valid_pairs : List (ℕ × ℕ) :=
[(2, 9), (3, 8), (4, 7), (5, 6), (6, 5), (7, 4), (8, 3), (9, 2)]

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

def reversed_sum_is_perfect_square (n : ℕ) : Prop :=
  ∃ t u, n = 10 * t + u ∧ t + u = 11

theorem count_two_digit_integers_with_perfect_square_sum :
  Nat.card { n : ℕ // is_two_digit n ∧ reversed_sum_is_perfect_square n } = 8 := 
sorry

end count_two_digit_integers_with_perfect_square_sum_l308_308976


namespace determine_value_of_c_l308_308816

theorem determine_value_of_c (b : ℝ) (h₁ : ∀ x : ℝ, 0 ≤ x^2 + x + b) (h₂ : ∃ m : ℝ, ∀ x : ℝ, x^2 + x + b < c ↔ x = m + 8) : 
    c = 16 :=
sorry

end determine_value_of_c_l308_308816


namespace sequence_product_modulo_7_l308_308656

theorem sequence_product_modulo_7 :
  let s := [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]
  s.foldl (*) 1 % 7 = 4 := by
  sorry

end sequence_product_modulo_7_l308_308656


namespace parabola_equation_with_left_focus_l308_308303

theorem parabola_equation_with_left_focus (x y : ℝ) :
  (∀ x y : ℝ, (x^2)/25 + (y^2)/9 = 1 → (y^2 = -16 * x)) :=
by
  sorry

end parabola_equation_with_left_focus_l308_308303


namespace markers_in_desk_l308_308325

theorem markers_in_desk (pens pencils markers : ℕ) 
  (h_ratio : pens = 2 * pencils ∧ pens = 2 * markers / 5) 
  (h_pens : pens = 10) : markers = 25 :=
by
  sorry

end markers_in_desk_l308_308325


namespace find_n_l308_308484

noncomputable def first_term_1 : ℝ := 12
noncomputable def second_term_1 : ℝ := 4
noncomputable def sum_first_series : ℝ := 18

noncomputable def first_term_2 : ℝ := 12
noncomputable def second_term_2 (n : ℝ) : ℝ := 4 + 2 * n
noncomputable def sum_second_series : ℝ := 90

theorem find_n (n : ℝ) : 
  (first_term_1 = 12) → 
  (second_term_1 = 4) → 
  (sum_first_series = 18) →
  (first_term_2 = 12) →
  (second_term_2 n = 4 + 2 * n) →
  (sum_second_series = 90) →
  (sum_second_series = 5 * sum_first_series) →
  n = 6 :=
by
  intros _ _ _ _ _ _ _
  sorry

end find_n_l308_308484


namespace intersection_of_function_and_inverse_l308_308739

theorem intersection_of_function_and_inverse (m : ℝ) :
  (∀ x y : ℝ, y = Real.sqrt (x - m) ↔ x = y^2 + m) →
  (∃ x : ℝ, Real.sqrt (x - m) = x) ↔ (m ≤ 1 / 4) :=
by
  sorry

end intersection_of_function_and_inverse_l308_308739


namespace parity_of_exponentiated_sum_l308_308010

theorem parity_of_exponentiated_sum
  : (1 ^ 1994 + 9 ^ 1994 + 8 ^ 1994 + 6 ^ 1994) % 2 = 0 := 
by
  sorry

end parity_of_exponentiated_sum_l308_308010


namespace average_runs_in_30_matches_l308_308615

theorem average_runs_in_30_matches (avg_runs_15: ℕ) (avg_runs_20: ℕ) 
    (matches_15: ℕ) (matches_20: ℕ)
    (h1: avg_runs_15 = 30) (h2: avg_runs_20 = 15)
    (h3: matches_15 = 15) (h4: matches_20 = 20) : 
    (matches_15 * avg_runs_15 + matches_20 * avg_runs_20) / (matches_15 + matches_20) = 25 := 
by 
  sorry

end average_runs_in_30_matches_l308_308615


namespace repayment_amount_l308_308105

theorem repayment_amount (borrowed amount : ℝ) (increase_percentage : ℝ) (final_amount : ℝ) 
  (h1 : borrowed_amount = 100) 
  (h2 : increase_percentage = 0.10) :
  final_amount = borrowed_amount * (1 + increase_percentage) :=
by 
  rw [h1, h2]
  norm_num
  exact eq.refl 110


end repayment_amount_l308_308105


namespace visitors_on_previous_day_is_246_l308_308481

def visitors_on_previous_day : Nat := 246
def total_visitors_in_25_days : Nat := 949

theorem visitors_on_previous_day_is_246 :
  visitors_on_previous_day = 246 := 
by
  rfl

end visitors_on_previous_day_is_246_l308_308481


namespace sum_of_digits_l308_308879

variable (a b c d e f : ℕ)

theorem sum_of_digits :
  ∀ (a b c d e f : ℕ),
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧
    100 * a + 10 * b + c + 100 * d + 10 * e + f = 1000 →
    a + b + c + d + e + f = 28 := 
by
  intros a b c d e f h
  sorry

end sum_of_digits_l308_308879


namespace quadratic_inequality_solution_l308_308801

theorem quadratic_inequality_solution {x : ℝ} :
  (x^2 + x - 6 ≤ 0) ↔ (-3 ≤ x ∧ x ≤ 2) :=
by
  sorry

end quadratic_inequality_solution_l308_308801


namespace evaluate_expression_l308_308558

variable {R : Type*} [LinearOrderedField R]

def roots_of_cubic (p q r : R) (a b c : R) :=
  a + b + c = p ∧ a * b + b * c + c * a = q ∧ a * b * c = r

theorem evaluate_expression (a b c : R) 
  (h : roots_of_cubic 15 22 8 a b c) : 
  (a / (1 / a + b * c) + b / (1 / b + c * a) + c / (1 / c + a * b) = 181 / 9) :=
by
  cases h with h_sum h_product;
  cases h_product with h_ab_bc_ca h_abc;
  sorry

end evaluate_expression_l308_308558


namespace pipe_cut_l308_308622

theorem pipe_cut (x : ℝ) (h1 : x + 2 * x = 177) : 2 * x = 118 :=
by
  sorry

end pipe_cut_l308_308622


namespace distinct_real_roots_l308_308944

-- Define the polynomial equation as a Lean function
def polynomial (a x : ℝ) : ℝ :=
  (a + 1) * (x ^ 2 + 1) ^ 2 - (2 * a + 3) * (x ^ 2 + 1) * x + (a + 2) * x ^ 2

-- The theorem we need to prove
theorem distinct_real_roots (a : ℝ) : 
  (∃ (x y : ℝ), x ≠ y ∧ polynomial a x = 0 ∧ polynomial a y = 0) ↔ a ≠ -1 :=
by
  sorry

end distinct_real_roots_l308_308944


namespace fractions_product_l308_308185

theorem fractions_product : 
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 3 / 7 := 
by 
  sorry

end fractions_product_l308_308185


namespace unique_prime_with_conditions_l308_308955

theorem unique_prime_with_conditions (p : ℕ) (hp : Nat.Prime p) (hp2 : Nat.Prime (p + 2)) (hp4 : Nat.Prime (p + 4)) : p = 3 :=
by
  sorry

end unique_prime_with_conditions_l308_308955


namespace sets_relationship_l308_308751

variables {U : Type*} (A B C : Set U)

theorem sets_relationship (h1 : A ∩ B = C) (h2 : B ∩ C = A) : A = C ∧ ∃ B, A ⊆ B := by
  sorry

end sets_relationship_l308_308751


namespace scalene_triangle_angle_obtuse_l308_308986

theorem scalene_triangle_angle_obtuse (a b c : ℝ) 
  (h_scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_longest : a > b ∧ a > c)
  (h_obtuse_angle : a^2 > b^2 + c^2) : 
  ∃ A : ℝ, A = (Real.pi / 2) ∧ (b^2 + c^2 - a^2) / (2 * b * c) < 0 := 
sorry

end scalene_triangle_angle_obtuse_l308_308986


namespace value_of_b_l308_308466

theorem value_of_b (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 35 * 45 * b) : b = 105 :=
sorry

end value_of_b_l308_308466


namespace BillCookingTime_l308_308786

-- Definitions corresponding to the conditions
def chopTimePepper : Nat := 3  -- minutes to chop one pepper
def chopTimeOnion : Nat := 4   -- minutes to chop one onion
def grateTimeCheese : Nat := 1 -- minutes to grate cheese for one omelet
def cookTimeOmelet : Nat := 5  -- minutes to assemble and cook one omelet

def numberOfPeppers : Nat := 4  -- number of peppers Bill needs to chop
def numberOfOnions : Nat := 2   -- number of onions Bill needs to chop
def numberOfOmelets : Nat := 5  -- number of omelets Bill prepares

-- Calculations based on conditions
def totalChopTimePepper : Nat := numberOfPeppers * chopTimePepper
def totalChopTimeOnion : Nat := numberOfOnions * chopTimeOnion
def totalGrateTimeCheese : Nat := numberOfOmelets * grateTimeCheese
def totalCookTimeOmelet : Nat := numberOfOmelets * cookTimeOmelet

-- Total preparation and cooking time
def totalTime : Nat := totalChopTimePepper + totalChopTimeOnion + totalGrateTimeCheese + totalCookTimeOmelet

-- Theorem statement
theorem BillCookingTime :
  totalTime = 50 := by
  sorry

end BillCookingTime_l308_308786


namespace bob_pennies_l308_308083

theorem bob_pennies (a b : ℤ) (h1 : b + 1 = 4 * (a - 1)) (h2 : b - 1 = 3 * (a + 1)) : b = 31 :=
by
  sorry

end bob_pennies_l308_308083


namespace part1_part2_l308_308236

-- Problem statement (1)
theorem part1 (a : ℝ) (h : a = -3) :
  (∀ x : ℝ, (x^2 + a * x + 2) ≤ 0 ↔ 1 ≤ x ∧ x ≤ 2) →
  { x : ℝ // (x^2 + a * x + 2) ≥ 1 - x^2 } = { x : ℝ // x ≤ 1 / 2 ∨ x ≥ 1 } :=
sorry

-- Problem statement (2)
theorem part2 (a : ℝ) :
  (∀ x : ℝ, (x^2 + a * x + 2) + x^2 + 1 = 2 * x^2 + a * x + 3) →
  (∃ x : ℝ, 1 < x ∧ x < 2 ∧ (2 * x^2 + a * x + 3) = 0) →
  -5 < a ∧ a < -2 * Real.sqrt 6 :=
sorry

end part1_part2_l308_308236


namespace cannot_form_set_of_good_friends_of_wang_ming_l308_308921

def is_well_defined_set (description : String) : Prop := sorry  -- Placeholder for the formal definition.

theorem cannot_form_set_of_good_friends_of_wang_ming :
  ¬ is_well_defined_set "Good friends of Wang Ming" :=
sorry

end cannot_form_set_of_good_friends_of_wang_ming_l308_308921


namespace simplify_fractions_l308_308853

theorem simplify_fractions : 
  (150 / 225) + (90 / 135) = 4 / 3 := by 
  sorry

end simplify_fractions_l308_308853


namespace train_length_l308_308632

theorem train_length (L V : ℝ) 
  (h1 : V = L / 10) 
  (h2 : V = (L + 870) / 39) 
  : L = 300 :=
by
  sorry

end train_length_l308_308632


namespace product_of_inverses_l308_308045

theorem product_of_inverses : 
  ((1 - 1 / (3^2)) * (1 - 1 / (5^2)) * (1 - 1 / (7^2)) * (1 - 1 / (11^2)) * (1 - 1 / (13^2)) * (1 - 1 / (17^2))) = 210 / 221 := 
by {
  sorry
}

end product_of_inverses_l308_308045


namespace sin_C_in_right_triangle_l308_308256

theorem sin_C_in_right_triangle
  (A B C : ℝ)
  (sin_A : ℝ)
  (sin_B : ℝ)
  (B_right_angle : B = π / 2)
  (sin_A_value : sin_A = 3 / 5)
  (sin_B_value : sin_B = 1)
  (sin_of_C : ℝ)
  (tri_ABC : A + B + C = π ∧ A > 0 ∧ C > 0) :
    sin_of_C = 4 / 5 :=
by
  -- Skipping the proof
  sorry

end sin_C_in_right_triangle_l308_308256


namespace find_x_collinear_l308_308818

-- Given vectors
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (1, -3)
def vec_c (x : ℝ) : ℝ × ℝ := (-2, x)

-- Definition of vectors being collinear
def collinear (v₁ v₂ : ℝ × ℝ) : Prop :=
∃ k : ℝ, v₁ = (k * v₂.1, k * v₂.2)

-- Question: What is the value of x such that vec_a + vec_b is collinear with vec_c(x)?
theorem find_x_collinear : ∃ x : ℝ, collinear (vec_a.1 + vec_b.1, vec_a.2 + vec_b.2) (vec_c x) ∧ x = 1 :=
by
  sorry

end find_x_collinear_l308_308818


namespace rosie_can_make_nine_pies_l308_308280

theorem rosie_can_make_nine_pies (apples pies : ℕ) (h : apples = 12 ∧ pies = 3) : 36 / (12 / 3) * pies = 9 :=
by
  sorry

end rosie_can_make_nine_pies_l308_308280


namespace exists_set_no_three_ap_l308_308096

theorem exists_set_no_three_ap (n : ℕ) (k : ℕ) :
  (n ≥ 1983) →
  (k ≤ 100000) →
  ∃ S : Finset ℕ,
    S.card = n ∧
    (∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → a < b → b < c → b ≠ (a + c) / 2) :=
sorry

end exists_set_no_three_ap_l308_308096


namespace circles_tangent_l308_308635

/--
Two equal circles each with a radius of 5 are externally tangent to each other and both are internally tangent to a larger circle with a radius of 13. 
Let the points of tangency be A and B. Let AB = m/n where m and n are positive integers and gcd(m, n) = 1. 
We need to prove that m + n = 69.
-/
theorem circles_tangent (r1 r2 r3 : ℝ) (tangent_external : ℝ) (tangent_internal : ℝ) (AB : ℝ) (m n : ℕ) 
  (hmn_coprime : Nat.gcd m n = 1) (hr1 : r1 = 5) (hr2 : r2 = 5) (hr3 : r3 = 13) 
  (ht_external : tangent_external = r1 + r2) (ht_internal : tangent_internal = r3 - r1) 
  (hAB : AB = (130 / 8)): m + n = 69 :=
by
  sorry

end circles_tangent_l308_308635


namespace sequence_1_formula_sequence_2_formula_sequence_3_formula_l308_308757

theorem sequence_1_formula (n : ℕ) (hn : n > 0) : 
  (∃ a : ℕ → ℚ, (a 1 = 1/2) ∧ (a 2 = 1/6) ∧ (a 3 = 1/12) ∧ (a 4 = 1/20) ∧ (∀ n, a n = 1/(n*(n+1)))) :=
by
  sorry

theorem sequence_2_formula (n : ℕ) (hn : n > 0) :
  (∃ a : ℕ → ℕ, (a 1 = 1) ∧ (a 2 = 2) ∧ (a 3 = 4) ∧ (a 4 = 8) ∧ (∀ n, a n = 2^(n-1))) :=
by
  sorry

theorem sequence_3_formula (n : ℕ) (hn : n > 0) :
  (∃ a : ℕ → ℚ, (a 1 = 4/5) ∧ (a 2 = 1/2) ∧ (a 3 = 4/11) ∧ (a 4 = 2/7) ∧ (∀ n, a n = 4/(3*n + 2))) :=
by
  sorry

end sequence_1_formula_sequence_2_formula_sequence_3_formula_l308_308757


namespace solution_pairs_l308_308194

theorem solution_pairs (x y : ℝ) : 
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ (y = -x - 2 ∨ y = -2 * x + 1) := 
by 
  sorry

end solution_pairs_l308_308194


namespace find_m_l308_308958

theorem find_m (m : ℤ) (h1 : -180 < m ∧ m < 180) : 
  ((m = 45) ∨ (m = -135)) ↔ (Real.tan (m * Real.pi / 180) = Real.tan (225 * Real.pi / 180)) := 
by 
  sorry

end find_m_l308_308958


namespace cone_lateral_surface_area_eq_sqrt_17_pi_l308_308525

theorem cone_lateral_surface_area_eq_sqrt_17_pi
  (r_cone r_sphere : ℝ) (h : ℝ)
  (V_sphere V_cone : ℝ)
  (h_cone_radius : r_cone = 1)
  (h_sphere_radius : r_sphere = 1)
  (h_volumes_eq : V_sphere = V_cone)
  (h_sphere_vol : V_sphere = (4 * π) / 3)
  (h_cone_vol : V_cone = (π * r_cone^2 * h) / 3) :
  (π * r_cone * (Real.sqrt (r_cone^2 + h^2))) = Real.sqrt 17 * π :=
sorry

end cone_lateral_surface_area_eq_sqrt_17_pi_l308_308525


namespace neg_abs_value_eq_neg_three_l308_308748

theorem neg_abs_value_eq_neg_three : -|-3| = -3 := 
by sorry

end neg_abs_value_eq_neg_three_l308_308748


namespace postage_cost_correct_l308_308146

-- Conditions
def base_rate : ℕ := 35
def additional_rate_per_ounce : ℕ := 25
def weight_in_ounces : ℚ := 5.25
def first_ounce : ℚ := 1
def fraction_weight : ℚ := weight_in_ounces - first_ounce
def num_additional_charges : ℕ := Nat.ceil (fraction_weight)

-- Question and correct answer
def total_postage_cost : ℕ := base_rate + (num_additional_charges * additional_rate_per_ounce)
def answer_in_cents : ℕ := 160

theorem postage_cost_correct : total_postage_cost = answer_in_cents := by sorry

end postage_cost_correct_l308_308146


namespace lines_from_equation_l308_308205

-- Definitions for the conditions
def satisfies_equation (x y : ℝ) : Prop :=
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Equivalent Lean statement to the proof problem
theorem lines_from_equation :
  (∀ x y : ℝ, satisfies_equation x y → (y = -x - 2) ∨ (y = -2 * x + 1)) :=
by
  intros x y h
  sorry

end lines_from_equation_l308_308205


namespace gold_coins_count_l308_308577

theorem gold_coins_count (n c : ℕ) (h1 : n = 8 * (c - 3))
                                     (h2 : n = 5 * c + 4)
                                     (h3 : c ≥ 10) : n = 54 :=
by
  sorry

end gold_coins_count_l308_308577


namespace car_speeds_and_arrival_times_l308_308486

theorem car_speeds_and_arrival_times
  (x y z u : ℝ)
  (h1 : x^2 = (y + z) * u)
  (h2 : (y + z) / 4 = u)
  (h3 : x / u = y / z)
  (h4 : x + y + z + u = 210) :
  x = 60 ∧ y = 80 ∧ z = 40 ∧ u = 30 := 
by
  sorry

end car_speeds_and_arrival_times_l308_308486


namespace elegant_interval_solution_l308_308830

noncomputable def elegant_interval : ℝ → ℝ × ℝ := sorry

theorem elegant_interval_solution (m : ℝ) (a b : ℕ) (s : ℝ) (p : ℕ) :
  a < m ∧ m < b ∧ a + 1 = b ∧ 3 < s + b ∧ s + b ≤ 13 ∧ s = Real.sqrt a ∧ b * b + a * s = p → p = 33 ∨ p = 127 := 
by sorry

end elegant_interval_solution_l308_308830


namespace linda_age_13_l308_308564

variable (J L : ℕ)

-- Conditions: 
-- 1. Linda is 3 more than 2 times the age of Jane.
-- 2. In five years, the sum of their ages will be 28.
def conditions (J L : ℕ) : Prop :=
  L = 2 * J + 3 ∧ (J + 5) + (L + 5) = 28

-- Question/answer to prove: Linda's current age is 13.
theorem linda_age_13 (J L : ℕ) (h : conditions J L) : L = 13 :=
by
  sorry

end linda_age_13_l308_308564


namespace correct_transformation_l308_308619

theorem correct_transformation (x : ℝ) : (x^2 - 10 * x - 1 = 0) → ((x - 5) ^ 2 = 26) := by
  sorry

end correct_transformation_l308_308619


namespace candy_probability_l308_308169

/-- 
A jar has 15 red candies, 15 blue candies, and 10 green candies. Terry picks three candies at random,
then Mary picks three of the remaining candies at random. Calculate the probability that they get 
the same color combination, irrespective of order, expressed as a fraction $m/n,$ where $m$ and $n$ 
are relatively prime positive integers. Find $m+n.$ -/
theorem candy_probability :
  let num_red := 15
  let num_blue := 15
  let num_green := 10
  let total_candies := num_red + num_blue + num_green
  let Terry_picks := 3
  let Mary_picks := 3
  let prob_equal_comb := (118545 : ℚ) / 2192991
  let m := 118545
  let n := 2192991
  m + n = 2310536 := sorry

end candy_probability_l308_308169


namespace range_of_m_l308_308432

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem range_of_m:
  ∀ m : ℝ, 
  (∀ x, 0 ≤ x ∧ x ≤ m → f x ≤ -3) ∧ 
  (∃ x, 0 ≤ x ∧ x ≤ m ∧ f x = -4) → 
  1 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_l308_308432


namespace rosie_can_make_nine_pies_l308_308278

theorem rosie_can_make_nine_pies (apples pies : ℕ) (h : apples = 12 ∧ pies = 3) : 36 / (12 / 3) * pies = 9 :=
by
  sorry

end rosie_can_make_nine_pies_l308_308278


namespace repay_loan_with_interest_l308_308104

theorem repay_loan_with_interest (amount_borrowed : ℝ) (interest_rate : ℝ) (total_payment : ℝ) 
  (h1 : amount_borrowed = 100) (h2 : interest_rate = 0.10) :
  total_payment = amount_borrowed + (amount_borrowed * interest_rate) :=
by sorry

end repay_loan_with_interest_l308_308104


namespace problem_l308_308520

noncomputable def f : ℝ → ℝ := sorry

variables (x : ℝ) (a b c : ℝ)
  (h1 : ∀ x, f (x + 1) = f (-x + 1))
  (h2 : ∀ x, 1 < x → f x ≤ f (x - 1))
  (ha : a = f 2)
  (hb : b = f (Real.log 2 / Real.log 3))
  (hc : c = f (1 / 2))

theorem problem (h : a = f 2 ∧ b = f (Real.log 2 / Real.log 3) ∧ c = f (1 / 2)) : 
  a < c ∧ c < b := sorry

end problem_l308_308520


namespace find_fourth_root_l308_308878

-- Define the polynomial P(x)
def P (x : ℝ) (a b : ℝ) : ℝ := b * x^3 + (3 * b + a) * x^2 + (a - 2 * b) * x + (5 - b)

-- Define the known roots
def known_roots (x : ℝ) := x = -1 ∨ x = 2 ∨ x = 4 ∨ x = -8

-- Prove the fourth root is -8
theorem find_fourth_root (a b : ℝ) (hx₁ : P (-1) a b = 0) (hx₂ : P 2 a b = 0) 
  (hx₃ : P 4 a b = 0) (hx₄ : P (-8) a b = 0) : known_roots (-8) :=
by {
  -- Proof would go here, but we include sorry since it is not required to solve explicitly.
  sorry
}

end find_fourth_root_l308_308878


namespace percentage_increase_of_return_trip_l308_308777

noncomputable def speed_increase_percentage (initial_speed avg_speed : ℝ) : ℝ :=
  ((2 * avg_speed * initial_speed) / avg_speed - initial_speed) * 100 / initial_speed

theorem percentage_increase_of_return_trip :
  let initial_speed := 30
  let avg_speed := 34.5
  speed_increase_percentage initial_speed avg_speed = 35.294 :=
  sorry

end percentage_increase_of_return_trip_l308_308777


namespace functional_equation_solution_l308_308230

theorem functional_equation_solution (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)) →
  ∀ x : ℝ, f x = a * x^2 + b * x :=
by
  intro h
  intro x
  have : ∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y) := h
  sorry

end functional_equation_solution_l308_308230


namespace teddy_bear_cost_l308_308410

-- Definitions for the given conditions
def num_toys : ℕ := 28
def toy_price : ℕ := 10
def num_teddy_bears : ℕ := 20
def total_money : ℕ := 580

-- The theorem we want to prove
theorem teddy_bear_cost :
  (num_teddy_bears * 15 + num_toys * toy_price = total_money) :=
by
  sorry

end teddy_bear_cost_l308_308410


namespace cos_theta_minus_pi_six_l308_308061

theorem cos_theta_minus_pi_six (θ : ℝ) (h : Real.sin (θ + π / 3) = 2 / 3) : 
  Real.cos (θ - π / 6) = 2 / 3 :=
sorry

end cos_theta_minus_pi_six_l308_308061


namespace solve_trig_eq_l308_308322

theorem solve_trig_eq (x : ℝ) :
  (0.5 * (Real.cos (5 * x) + Real.cos (7 * x)) - Real.cos (2 * x) ^ 2 + Real.sin (3 * x) ^ 2 = 0) →
  (∃ k : ℤ, x = (Real.pi / 2) * (2 * k + 1) ∨ x = (2 * k * Real.pi / 11)) :=
sorry

end solve_trig_eq_l308_308322


namespace inequality_proof_l308_308223

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (n : ℕ) (hn : 0 < n) : 
  (x / (n * x + y + z) + y / (x + n * y + z) + z / (x + y + n * z)) ≤ 3 / (n + 2) :=
sorry

end inequality_proof_l308_308223


namespace shares_sum_4000_l308_308813

variables (w x y z : ℝ)

def relation_z_w : Prop := z = 1.20 * w
def relation_y_z : Prop := y = 1.25 * z
def relation_x_y : Prop := x = 1.35 * y
def w_after_3_years : ℝ := 8 * w
def z_after_3_years : ℝ := 8 * z
def y_after_3_years : ℝ := 8 * y
def x_after_3_years : ℝ := 8 * x

theorem shares_sum_4000 (w : ℝ) :
  relation_z_w w z →
  relation_y_z z y →
  relation_x_y y x →
  x_after_3_years x + y_after_3_years y + z_after_3_years z + w_after_3_years w = 4000 :=
by
  intros h_z_w h_y_z h_x_y
  rw [relation_z_w, relation_y_z, relation_x_y] at *
  sorry

end shares_sum_4000_l308_308813


namespace carlos_paid_l308_308595

theorem carlos_paid (a b c : ℝ) 
  (h1 : a = (1 / 3) * (b + c))
  (h2 : b = (1 / 4) * (a + c))
  (h3 : a + b + c = 120) :
  c = 72 :=
by
-- Proof omitted
sorry

end carlos_paid_l308_308595


namespace sqrt_trig_identity_l308_308969

theorem sqrt_trig_identity
  (α : ℝ)
  (P : ℝ × ℝ)
  (hP: P = (Real.sin 2, Real.cos 2))
  (h_terminal: ∃ (θ : ℝ), P = (Real.cos θ, Real.sin θ)) :
  Real.sqrt (2 * (1 - Real.sin α)) = 2 * Real.sin 1 := 
sorry

end sqrt_trig_identity_l308_308969


namespace average_speed_for_trip_l308_308465

theorem average_speed_for_trip 
  (Speed1 Speed2 : ℝ) 
  (AverageSpeed : ℝ) 
  (h1 : Speed1 = 110) 
  (h2 : Speed2 = 72) 
  (h3 : AverageSpeed = (2 * Speed1 * Speed2) / (Speed1 + Speed2)) :
  AverageSpeed = 87 := 
by
  -- solution steps would go here
  sorry

end average_speed_for_trip_l308_308465


namespace cosine_of_difference_l308_308530

theorem cosine_of_difference (α : ℝ) (h : Real.sin (α + π / 6) = 1 / 3) :
  Real.cos (α - π / 3) = 1 / 3 :=
by
  sorry

end cosine_of_difference_l308_308530


namespace arrangement_possible_l308_308704

noncomputable def exists_a_b : Prop :=
  ∃ a b : ℝ, a + 2*b > 0 ∧ 7*a + 13*b < 0

theorem arrangement_possible : exists_a_b := by
  sorry

end arrangement_possible_l308_308704


namespace find_seventh_value_l308_308485

theorem find_seventh_value (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ)
  (h₁ : x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 9*x₅ + 11*x₆ = 0)
  (h₂ : 3*x₁ + 5*x₂ + 7*x₃ + 9*x₄ + 11*x₅ + 13*x₆ = 10)
  (h₃ : 5*x₁ + 7*x₂ + 9*x₃ + 11*x₄ + 13*x₅ + 15*x₆ = 100) :
  7*x₁ + 9*x₂ + 11*x₃ + 13*x₄ + 15*x₅ + 17*x₆ = 210 :=
sorry

end find_seventh_value_l308_308485


namespace smallest_number_of_eggs_over_150_l308_308901

theorem smallest_number_of_eggs_over_150 
  (d : ℕ) 
  (h1: 12 * d - 3 > 150) 
  (h2: ∀ k < d, 12 * k - 3 ≤ 150) :
  12 * d - 3 = 153 :=
by
  sorry

end smallest_number_of_eggs_over_150_l308_308901


namespace product_of_first_two_terms_l308_308583

-- Given parameters
variables (a d : ℤ) -- a is the first term, d is the common difference

-- Conditions
def fifth_term_condition (a d : ℤ) : Prop := a + 4 * d = 11
def common_difference_condition (d : ℤ) : Prop := d = 1

-- Main statement to prove
theorem product_of_first_two_terms (a d : ℤ) (h1 : fifth_term_condition a d) (h2 : common_difference_condition d) :
  a * (a + d) = 56 :=
by
  sorry

end product_of_first_two_terms_l308_308583


namespace find_x_given_y_l308_308526

noncomputable def constantRatio : Prop :=
  ∃ k : ℚ, ∀ x y : ℚ, (5 * x - 6) / (2 * y + 10) = k

theorem find_x_given_y :
  (constantRatio ∧ (3, 2) ∈ {(x, y) | (5 * x - 6) / (2 * y + 10) = 9 / 14}) →
  ∃ x : ℚ, ((5 * x - 6) / 20 = 9 / 14 ∧ x = 53 / 14) :=
by
  sorry

end find_x_given_y_l308_308526


namespace prob_X_ge_2_l308_308363

open ProbabilityTheory

noncomputable def normal_dist_X : Measure ℝ := measure_theory.measure_gaussian (3 : ℝ) (σ^2 : ℝ)

theorem prob_X_ge_2 : (ProbabilityTheory.probability (normal_dist_X (set.Ici (2 : ℝ)))) = 0.85 :=
begin
  sorry
end

end prob_X_ge_2_l308_308363


namespace no_sqrt_negative_number_l308_308603

theorem no_sqrt_negative_number (a b c d : ℝ) (hA : a = (-3)^2) (hB : b = 0) (hC : c = 1/8) (hD : d = -6^3) : 
  ¬ (∃ x : ℝ, x^2 = d) :=
by
  sorry

end no_sqrt_negative_number_l308_308603


namespace combination_mod_100_l308_308255

def totalDistinctHands : Nat := Nat.choose 60 12

def remainder (n : Nat) (m : Nat) : Nat := n % m

theorem combination_mod_100 :
  remainder totalDistinctHands 100 = R :=
sorry

end combination_mod_100_l308_308255


namespace wilfred_carrots_total_l308_308608

-- Define the number of carrots Wilfred eats each day
def tuesday_carrots := 4
def wednesday_carrots := 6
def thursday_carrots := 5

-- Define the total number of carrots eaten from Tuesday to Thursday
def total_carrots := tuesday_carrots + wednesday_carrots + thursday_carrots

-- The theorem to prove that the total number of carrots is 15
theorem wilfred_carrots_total : total_carrots = 15 := by
  sorry

end wilfred_carrots_total_l308_308608


namespace distinct_integer_roots_l308_308112

-- Definitions of m and the polynomial equation.
def poly (m : ℤ) (x : ℤ) : Prop :=
  x^2 - 2 * (2 * m - 3) * x + 4 * m^2 - 14 * m + 8 = 0

-- Theorem stating that for m = 12 and m = 24, the polynomial has specific roots.
theorem distinct_integer_roots (m x : ℤ) (h1 : 4 < m) (h2 : m < 40) :
  (m = 12 ∨ m = 24) ∧ 
  ((m = 12 ∧ (x = 26 ∨ x = 16) ∧ poly m x) ∨
   (m = 24 ∧ (x = 52 ∨ x = 38) ∧ poly m x)) :=
by
  sorry

end distinct_integer_roots_l308_308112


namespace range_of_sum_coords_on_ellipse_l308_308808

theorem range_of_sum_coords_on_ellipse (x y : ℝ) 
  (h : x^2 / 144 + y^2 / 25 = 1) : 
  -13 ≤ x + y ∧ x + y ≤ 13 := 
sorry

end range_of_sum_coords_on_ellipse_l308_308808


namespace number_of_ways_to_tile_dominos_l308_308340

-- Define the dimensions of the shapes and the criteria for the tiling problem
def L_shaped_area := 24
def size_of_square := 4
def size_of_rectangles := 2 * 10
def number_of_ways_to_tile := 208

-- Theorem statement
theorem number_of_ways_to_tile_dominos :
  (L_shaped_area = size_of_square + size_of_rectangles) →
  number_of_ways_to_tile = 208 :=
by
  intros h
  sorry

end number_of_ways_to_tile_dominos_l308_308340


namespace shaded_square_area_l308_308431

noncomputable def Pythagorean_area (a b c : ℕ) (area_a area_b area_c : ℕ) : Prop :=
  area_a = a^2 ∧ area_b = b^2 ∧ area_c = c^2 ∧ a^2 + b^2 = c^2

theorem shaded_square_area 
  (area1 area2 area3 : ℕ)
  (area_unmarked : ℕ)
  (h1 : area1 = 5)
  (h2 : area2 = 8)
  (h3 : area3 = 32)
  (h_unmarked: area_unmarked = area2 + area3)
  (h_shaded : area1 + area_unmarked = 45) :
  area1 + area_unmarked = 45 :=
by
  exact h_shaded

end shaded_square_area_l308_308431


namespace average_of_consecutive_integers_l308_308291

theorem average_of_consecutive_integers (n m : ℕ) 
  (h1 : m = (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6)) / 7) : 
  (n + 6) = (m + (m+1) + (m+2) + (m+3) + (m+4) + (m+5) + (m+6)) / 7 :=
by
  sorry

end average_of_consecutive_integers_l308_308291


namespace arithmetic_expression_value_l308_308883

theorem arithmetic_expression_value : 4 * (8 - 3) - 7 = 13 := by
  sorry

end arithmetic_expression_value_l308_308883


namespace inequality_proof_equality_condition_l308_308681

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a*b)) :=
sorry

theorem equality_condition (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a*b) ↔ a = b ∧ a < 1) :=
sorry

end inequality_proof_equality_condition_l308_308681


namespace usual_time_catch_bus_l308_308880

-- Define the problem context
variable (S T : ℝ)

-- Hypotheses for the conditions given
def condition1 : Prop := S * T = (4 / 5) * S * (T + 4)
def condition2 : Prop := S ≠ 0

-- Theorem that states the fact we need to prove
theorem usual_time_catch_bus (h1 : condition1 S T) (h2 : condition2 S) : T = 16 :=
by
  -- proof omitted
  sorry

end usual_time_catch_bus_l308_308880


namespace rent_fraction_l308_308912

theorem rent_fraction (B R : ℝ) 
  (food_and_beverages_spent : (1 / 4) * (1 - R) * B = 0.1875 * B) : 
  R = 0.25 :=
by
  -- proof skipped
  sorry

end rent_fraction_l308_308912


namespace smallest_possible_stamps_l308_308849

theorem smallest_possible_stamps (M : ℕ) : 
  ((M % 5 = 2) ∧ (M % 7 = 2) ∧ (M % 9 = 2) ∧ (M > 2)) → M = 317 := 
by 
  sorry

end smallest_possible_stamps_l308_308849


namespace determine_cards_per_friend_l308_308735

theorem determine_cards_per_friend (n_cards : ℕ) (n_friends : ℕ) (h : n_cards = 12) : n_friends > 0 → (n_cards / n_friends) = (12 / n_friends) :=
by
  sorry

end determine_cards_per_friend_l308_308735


namespace no_sqrt_negative_number_l308_308604

theorem no_sqrt_negative_number (a b c d : ℝ) (hA : a = (-3)^2) (hB : b = 0) (hC : c = 1/8) (hD : d = -6^3) : 
  ¬ (∃ x : ℝ, x^2 = d) :=
by
  sorry

end no_sqrt_negative_number_l308_308604


namespace percentage_of_green_ducks_l308_308834

def total_ducks := 100
def green_ducks_smaller_pond := 9
def green_ducks_larger_pond := 22
def total_green_ducks := green_ducks_smaller_pond + green_ducks_larger_pond

theorem percentage_of_green_ducks :
  (total_green_ducks / total_ducks) * 100 = 31 :=
by
  sorry

end percentage_of_green_ducks_l308_308834


namespace total_pictures_l308_308379

theorem total_pictures :
  let Randy_pictures := 5
  let Peter_pictures := Randy_pictures + 3
  let Quincy_pictures := Peter_pictures + 20
  let Susan_pictures := 2 * Quincy_pictures - 7
  let Thomas_pictures := Randy_pictures ^ 3
  Randy_pictures + Peter_pictures + Quincy_pictures + Susan_pictures + Thomas_pictures = 215 := by 
    let Randy_pictures := 5
    let Peter_pictures := Randy_pictures + 3
    let Quincy_pictures := Peter_pictures + 20
    let Susan_pictures := 2 * Quincy_pictures - 7
    let Thomas_pictures := Randy_pictures ^ 3
    sorry

end total_pictures_l308_308379


namespace monthly_cost_per_iguana_l308_308674

theorem monthly_cost_per_iguana
  (gecko_cost snake_cost annual_cost : ℕ)
  (monthly_cost_per_iguana : ℕ)
  (gecko_count iguana_count snake_count : ℕ)
  (annual_cost_eq : annual_cost = 1140)
  (gecko_count_eq : gecko_count = 3)
  (iguana_count_eq : iguana_count = 2)
  (snake_count_eq : snake_count = 4)
  (gecko_cost_eq : gecko_cost = 15)
  (snake_cost_eq : snake_cost = 10)
  (total_annual_cost_eq : gecko_count * gecko_cost + iguana_count * monthly_cost_per_iguana * 12 + snake_count * snake_cost * 12 = annual_cost) :
  monthly_cost_per_iguana = 5 :=
by
  sorry

end monthly_cost_per_iguana_l308_308674


namespace most_prolific_mathematician_is_euler_l308_308756

noncomputable def prolific_mathematician (collected_works_volume_count: ℕ) (publishing_organization: String) : String :=
  if collected_works_volume_count > 75 ∧ publishing_organization = "Swiss Society of Natural Sciences" then
    "Leonhard Euler"
  else
    "Unknown"

theorem most_prolific_mathematician_is_euler :
  prolific_mathematician 76 "Swiss Society of Natural Sciences" = "Leonhard Euler" :=
by
  sorry

end most_prolific_mathematician_is_euler_l308_308756


namespace find_mn_l308_308807

theorem find_mn (m n : ℕ) (h : m > 0 ∧ n > 0) (eq1 : m^2 + n^2 + 4 * m - 46 = 0) :
  mn = 5 ∨ mn = 15 := by
  sorry

end find_mn_l308_308807


namespace find_principal_sum_l308_308137

def compound_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r)^t - P

def simple_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * r * t

theorem find_principal_sum (CI SI : ℝ) (t : ℕ)
  (h1 : CI = 11730) 
  (h2 : SI = 10200) 
  (h3 : t = 2) :
  ∃ P r, P = 17000 ∧
  compound_interest P r t = CI ∧
  simple_interest P r t = SI :=
by
  sorry

end find_principal_sum_l308_308137


namespace original_acid_percentage_l308_308480

variables (a w : ℝ)

-- Conditions from the problem
def cond1 : Prop := a / (a + w + 2) = 0.18
def cond2 : Prop := (a + 2) / (a + w + 4) = 0.36

-- The Lean statement to prove
theorem original_acid_percentage (hc1 : cond1 a w) (hc2 : cond2 a w) : (a / (a + w)) * 100 = 19 :=
sorry

end original_acid_percentage_l308_308480


namespace eval_expression_l308_308896

theorem eval_expression : 4 * (8 - 3) - 7 = 13 :=
by
  sorry

end eval_expression_l308_308896


namespace triangle_sides_length_a_triangle_perimeter_l308_308108

theorem triangle_sides_length_a (A B C : ℝ) (a b c : ℝ) 
  (hA : A = π / 3) (h1 : (b + c) / (Real.sin B + Real.sin C) = 2) :
  a = Real.sqrt 3 :=
sorry

theorem triangle_perimeter (A B C : ℝ) (a b c : ℝ) 
  (hA : A = π / 3) (h1 : (b + c) / (Real.sin B + Real.sin C) = 2) 
  (h2 : (b * c * Real.sin (π / 3)) / 2 = Real.sqrt 3 / 2) :
  a + b + c = 3 + Real.sqrt 3 :=
sorry

end triangle_sides_length_a_triangle_perimeter_l308_308108


namespace root_exists_between_0_and_1_l308_308417

theorem root_exists_between_0_and_1 (a b c : ℝ) (m : ℝ) (hm : 0 < m)
  (h : a / (m + 2) + b / (m + 1) + c / m = 0) : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x ^ 2 + b * x + c = 0 :=
by
  sorry

end root_exists_between_0_and_1_l308_308417


namespace heesu_received_most_sweets_l308_308020

theorem heesu_received_most_sweets
  (total_sweets : ℕ)
  (minsus_sweets : ℕ)
  (jaeyoungs_sweets : ℕ)
  (heesus_sweets : ℕ)
  (h_total : total_sweets = 30)
  (h_minsu : minsus_sweets = 12)
  (h_jaeyoung : jaeyoungs_sweets = 3)
  (h_heesu : heesus_sweets = 15) :
  heesus_sweets = max minsus_sweets (max jaeyoungs_sweets heesus_sweets) :=
by sorry

end heesu_received_most_sweets_l308_308020


namespace problem_l308_308671

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1 / 2) * m * x^2 + 1
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := 2 * Real.log x - (2 * m + 1) * x - 1
noncomputable def h (m : ℝ) (x : ℝ) := f m x + g m x

noncomputable def h_deriv (m : ℝ) (x : ℝ) : ℝ := m * x - (2 * m + 1) + (2 / x)

theorem problem (m : ℝ) : h_deriv m 1 = h_deriv m 3 → m = 2 / 3 :=
by
  sorry

end problem_l308_308671


namespace question1_question2_question3_l308_308779

def f : Nat → Nat → Nat := sorry

axiom condition1 : f 1 1 = 1
axiom condition2 : ∀ m n, f m (n + 1) = f m n + 2
axiom condition3 : ∀ m, f (m + 1) 1 = 2 * f m 1

theorem question1 (n : Nat) : f 1 n = 2 * n - 1 :=
sorry

theorem question2 (m : Nat) : f m 1 = 2 ^ (m - 1) :=
sorry

theorem question3 : f 2002 9 = 2 ^ 2001 + 16 :=
sorry

end question1_question2_question3_l308_308779


namespace tip_calculation_correct_l308_308911

noncomputable def calculate_tip (total_with_tax : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  let bill_before_tax := total_with_tax / (1 + tax_rate)
  bill_before_tax * tip_rate

theorem tip_calculation_correct :
  calculate_tip 226 0.13 0.15 = 30 := 
by
  sorry

end tip_calculation_correct_l308_308911


namespace moon_radius_scientific_notation_l308_308742

noncomputable def moon_radius : ℝ := 1738000

theorem moon_radius_scientific_notation :
  moon_radius = 1.738 * 10^6 :=
by
  sorry

end moon_radius_scientific_notation_l308_308742


namespace work_days_for_A_l308_308623

theorem work_days_for_A (x : ℕ) : 
  (∀ a b, 
    (a = 1 / (x : ℚ)) ∧ 
    (b = 1 / 20) ∧ 
    (8 * (a + b) = 14 / 15) → 
    x = 15) :=
by
  intros a b h
  have ha : a = 1 / (x : ℚ) := h.1
  have hb : b = 1 / 20 := h.2.1
  have hab : 8 * (a + b) = 14 / 15 := h.2.2
  sorry

end work_days_for_A_l308_308623


namespace c_should_pay_l308_308324

-- Define the grazing capacity equivalences
def horse_eq_oxen : ℝ := 2.0
def sheep_eq_oxen : ℝ := 0.5

-- Define the grazing capacities for each person in oxen-months
def a_grazing_oxen_months : ℝ := 10 * 7 + 4 * horse_eq_oxen * 3
def b_grazing_oxen_months : ℝ := 12 * 5
def c_grazing_oxen_months : ℝ := 15 * 3
def d_grazing_oxen_months : ℝ := 18 * 6 + 6 * sheep_eq_oxen * 8
def e_grazing_oxen_months : ℝ := 20 * 4
def f_grazing_oxen_months : ℝ := 5 * horse_eq_oxen * 2 + 10 * sheep_eq_oxen * 4

-- Define the total grazing capacity
def total_grazing_oxen_months : ℝ := a_grazing_oxen_months + b_grazing_oxen_months + c_grazing_oxen_months + d_grazing_oxen_months + e_grazing_oxen_months + f_grazing_oxen_months

-- Define the rent of the pasture
def total_rent : ℝ := 1200

-- Define the amount c should pay
def amount_c_should_pay : ℝ := (c_grazing_oxen_months / total_grazing_oxen_months) * total_rent

-- Prove that the amount c should pay is approximately Rs. 119.73
theorem c_should_pay (h_approx : | amount_c_should_pay - 119.73 | < 0.01) : true := by 
  -- Skip the proof for now
  sorry

end c_should_pay_l308_308324


namespace hitting_at_least_7_rings_hitting_fewer_than_8_rings_l308_308914

-- Definitions of the events and their probabilities
def P_A10 : ℝ := 0.20
def P_A9 : ℝ := 0.22
def P_A8 : ℝ := 0.25
def P_A7 : ℝ := 0.28

-- Probability of hitting at least 7 rings
def P_A : ℝ := P_A10 + P_A9 + P_A8 + P_A7

theorem hitting_at_least_7_rings :
  (P_A = 0.95) :=
by
  -- proof goes here
  sorry

-- Probability of hitting fewer than 8 rings
def P_notA : ℝ := 1 - P_A
def P_B : ℝ := P_A7 + P_notA

theorem hitting_fewer_than_8_rings :
  (P_B = 0.33) :=
by
  -- proof goes here
  sorry

end hitting_at_least_7_rings_hitting_fewer_than_8_rings_l308_308914


namespace find_last_number_l308_308919

theorem find_last_number (A B C D E F G : ℝ)
    (h1 : (A + B + C + D) / 4 = 13)
    (h2 : (D + E + F + G) / 4 = 15)
    (h3 : E + F + G = 55)
    (h4 : D^2 = G) :
  G = 25 := by 
  sorry

end find_last_number_l308_308919


namespace cities_with_fewer_than_500000_residents_l308_308859

theorem cities_with_fewer_than_500000_residents (P Q R : ℕ) 
  (h1 : P + Q + R = 100) 
  (h2 : P = 40) 
  (h3 : Q = 35) 
  (h4 : R = 25) : P + Q = 75 :=
by 
  sorry

end cities_with_fewer_than_500000_residents_l308_308859


namespace roses_in_december_l308_308584

theorem roses_in_december (rOct rNov rJan rFeb : ℕ) 
  (hOct : rOct = 108)
  (hNov : rNov = 120)
  (hJan : rJan = 144)
  (hFeb : rFeb = 156)
  (pattern : (rNov - rOct = 12 ∨ rNov - rOct = 24) ∧ 
             (rJan - rNov = 12 ∨ rJan - rNov = 24) ∧
             (rFeb - rJan = 12 ∨ rFeb - rJan = 24) ∧ 
             (∀ m n, (m - n = 12 ∨ m - n = 24) → 
               ((rNov - rOct) ≠ (rJan - rNov) ↔ 
               (rJan - rNov) ≠ (rFeb - rJan)))) : 
  ∃ rDec : ℕ, rDec = 132 := 
by {
  sorry
}

end roses_in_december_l308_308584


namespace cube_root_of_8_is_2_l308_308857

theorem cube_root_of_8_is_2 : (∛8 : ℝ) = 2 :=
by
  sorry

end cube_root_of_8_is_2_l308_308857


namespace haley_total_lives_l308_308327

-- Define initial conditions
def initial_lives : ℕ := 14
def lives_lost : ℕ := 4
def lives_gained : ℕ := 36

-- Definition to calculate total lives
def total_lives (initial_lives lives_lost lives_gained : ℕ) : ℕ :=
  initial_lives - lives_lost + lives_gained

-- The theorem statement we want to prove
theorem haley_total_lives : total_lives initial_lives lives_lost lives_gained = 46 :=
by 
  sorry

end haley_total_lives_l308_308327


namespace Freddy_age_l308_308189

noncomputable def M : ℕ := 11
noncomputable def R : ℕ := M - 2
noncomputable def F : ℕ := M + 4

theorem Freddy_age : F = 15 :=
  by
    sorry

end Freddy_age_l308_308189


namespace sqrt_expression_meaningful_l308_308737

/--
When is the algebraic expression √(x + 2) meaningful?
To ensure the algebraic expression √(x + 2) is meaningful, 
the expression under the square root, x + 2, must be greater than or equal to 0.
Thus, we need to prove that this condition is equivalent to x ≥ -2.
-/
theorem sqrt_expression_meaningful (x : ℝ) : (x + 2 ≥ 0) ↔ (x ≥ -2) :=
by
  sorry

end sqrt_expression_meaningful_l308_308737


namespace rosie_can_make_nine_pies_l308_308277

theorem rosie_can_make_nine_pies (apples pies : ℕ) (h : apples = 12 ∧ pies = 3) : 36 / (12 / 3) * pies = 9 :=
by
  sorry

end rosie_can_make_nine_pies_l308_308277


namespace ben_has_20_mms_l308_308489

theorem ben_has_20_mms (B_candies Ben_candies : ℕ) 
  (h1 : B_candies = 50) 
  (h2 : B_candies = Ben_candies + 30) : 
  Ben_candies = 20 := 
by
  sorry

end ben_has_20_mms_l308_308489


namespace ice_cream_cost_l308_308910

-- Define the given conditions
def cost_brownie : ℝ := 2.50
def cost_syrup_per_unit : ℝ := 0.50
def cost_nuts : ℝ := 1.50
def cost_total : ℝ := 7.00
def scoops_ice_cream : ℕ := 2
def syrup_units : ℕ := 2

-- Define the hot brownie dessert cost equation
def hot_brownie_cost (cost_ice_cream_per_scoop : ℝ) : ℝ :=
  cost_brownie + (cost_syrup_per_unit * syrup_units) + cost_nuts + (scoops_ice_cream * cost_ice_cream_per_scoop)

-- Define the theorem we want to prove
theorem ice_cream_cost : hot_brownie_cost 1 = cost_total :=
by sorry

end ice_cream_cost_l308_308910


namespace rectangular_sheet_integer_side_l308_308628

theorem rectangular_sheet_integer_side
  (a b : ℝ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (h_cut_a : ∀ x, x ≤ a → ∃ n : ℕ, x = n ∨ x = n + 1)
  (h_cut_b : ∀ y, y ≤ b → ∃ n : ℕ, y = n ∨ y = n + 1) :
  ∃ n m : ℕ, a = n ∨ b = m := 
sorry

end rectangular_sheet_integer_side_l308_308628


namespace y_intercept_of_line_l308_308009

theorem y_intercept_of_line (x y : ℝ) : x + 2 * y + 6 = 0 → x = 0 → y = -3 :=
by
  sorry

end y_intercept_of_line_l308_308009


namespace proof_inequality_l308_308591

theorem proof_inequality (x : ℝ) : (3 ≤ |x + 2| ∧ |x + 2| ≤ 7) ↔ (1 ≤ x ∧ x ≤ 5 ∨ -9 ≤ x ∧ x ≤ -5) :=
by
  sorry

end proof_inequality_l308_308591


namespace concurrency_and_concurrence_point_Q_on_OI_l308_308514

open EuclideanGeometry

variable {A B C I O : Point}

axiom incenter : incenterTriangle A B C I
axiom circumcenter : circumcenterTriangle A B C O

-- $\Gamma_A$ passes through $B$ and $C$ and is tangent to the incircle
axiom GammaA_tangent : ∃ (Gamma_A : Circle), Gamma_A.passes_through B ∧ Gamma_A.passes_through C ∧ tangent Gamma_A (incircle A B C)

-- Similar definitions for $\Gamma_B$ and $\Gamma_C$
axiom GammaB_tangent : ∃ (Gamma_B : Circle), Gamma_B.passes_through A ∧ Gamma_B.passes_through C ∧ tangent Gamma_B (incircle A B C)
axiom GammaC_tangent : ∃ (Gamma_C : Circle), Gamma_C.passes_through A ∧ Gamma_C.passes_through B ∧ tangent Gamma_C (incircle A B C)

-- Intersection points $A'$, $B'$, and $C'$
axiom Aprime_def : ∃ (A' : Point), (GammaB ∩ GammaC = {A, A'})
axiom Bprime_def : ∃ (B' : Point), (GammaA ∩ GammaC = {B, B'})
axiom Cprime_def : ∃ (C' : Point), (GammaA ∩ GammaB = {C, C'})

-- The main theorem we are proving
theorem concurrency_and_concurrence_point_Q_on_OI :
  ∃ (Q : Point), concurrents [line_through A A', line_through B B', line_through C C'] ∧ on_line Q (line_through O I) :=
by
  -- We need a proof here
  sorry

end concurrency_and_concurrence_point_Q_on_OI_l308_308514


namespace monotone_increasing_range_of_a_l308_308590

noncomputable def f (a x : ℝ) : ℝ := x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem monotone_increasing_range_of_a :
  (∀ x y, x ≤ y → f a x ≤ f a y) ↔ (a ∈ Set.Icc (-1 / 3 : ℝ) (1 / 3 : ℝ)) :=
sorry

end monotone_increasing_range_of_a_l308_308590


namespace system_solution_a_l308_308576

theorem system_solution_a (x y z : ℤ) (h1 : x^2 + x * y + y^2 = 7) (h2 : y^2 + y * z + z^2 = 13) (h3 : z^2 + z * x + x^2 = 19) :
  (x = 2 ∧ y = 1 ∧ z = 3) ∨ (x = -2 ∧ y = -1 ∧ z = -3) :=
sorry

end system_solution_a_l308_308576


namespace apples_difference_l308_308917

theorem apples_difference
    (adam_apples : ℕ)
    (jackie_apples : ℕ)
    (h_adam : adam_apples = 10)
    (h_jackie : jackie_apples = 2) :
    adam_apples - jackie_apples = 8 :=
by
    sorry

end apples_difference_l308_308917


namespace exists_3x3_grid_l308_308126

theorem exists_3x3_grid : 
  ∃ (a₁₂ a₂₁ a₂₃ a₃₂ : ℕ), 
  a₁₂ ≠ a₂₁ ∧ a₁₂ ≠ a₂₃ ∧ a₁₂ ≠ a₃₂ ∧ 
  a₂₁ ≠ a₂₃ ∧ a₂₁ ≠ a₃₂ ∧ 
  a₂₃ ≠ a₃₂ ∧ 
  a₁₂ ≤ 25 ∧ a₂₁ ≤ 25 ∧ a₂₃ ≤ 25 ∧ a₃₂ ≤ 25 ∧ 
  a₁₂ > 0 ∧ a₂₁ > 0 ∧ a₂₃ > 0 ∧ a₃₂ > 0 ∧
  (∃ (a₁₁ a₁₃ a₃₁ a₃₃ a₂₂ : ℕ),
  a₁₁ ≤ 25 ∧ a₁₃ ≤ 25 ∧ a₃₁ ≤ 25 ∧ a₃₃ ≤ 25 ∧ a₂₂ ≤ 25 ∧
  a₁₁ > 0 ∧ a₁₃ > 0 ∧ a₃₁ > 0 ∧ a₃₃ > 0 ∧ a₂₂ > 0 ∧
  a₁₁ ≠ a₁₂ ∧ a₁₁ ≠ a₂₁ ∧ a₁₁ ≠ a₁₃ ∧ a₁₁ ≠ a₃₁ ∧ 
  a₁₃ ≠ a₃₃ ∧ a₁₃ ≠ a₂₃ ∧ a₂₁ ≠ a₃₁ ∧ a₃₁ ≠ a₃₂ ∧ 
  a₃₃ ≠ a₂₂ ∧ a₃₃ ≠ a₃₂ ∧ a₂₂ = 1 ∧
  (a₁₂ % a₂₂ = 0 ∨ a₂₂ % a₁₂ = 0) ∧
  (a₂₁ % a₂₂ = 0 ∨ a₂₂ % a₂₁ = 0) ∧
  (a₂₃ % a₂₂ = 0 ∨ a₂₂ % a₂₃ = 0) ∧
  (a₃₂ % a₂₂ = 0 ∨ a₂₂ % a₃₂ = 0) ∧
  (a₁₁ % a₁₂ = 0 ∨ a₁₂ % a₁₁ = 0) ∧
  (a₁₁ % a₂₁ = 0 ∨ a₂₁ % a₁₁ = 0) ∧
  (a₁₃ % a₁₂ = 0 ∨ a₁₂ % a₁₃ = 0) ∧
  (a₁₃ % a₂₃ = 0 ∨ a₂₃ % a₁₃ = 0) ∧
  (a₃₁ % a₂₁ = 0 ∨ a₂₁ % a₃₁ = 0) ∧
  (a₃₁ % a₃₂ = 0 ∨ a₃₂ % a₃₁ = 0) ∧
  (a₃₃ % a₂₃ = 0 ∨ a₂₃ % a₃₃ = 0) ∧
  (a₃₃ % a₃₂ = 0 ∨ a₃₂ % a₃₃ = 0)) 
  :=
sorry

end exists_3x3_grid_l308_308126


namespace count_three_digit_integers_end7_divby21_l308_308675

theorem count_three_digit_integers_end7_divby21 : 
  let count := (List.range' 4 (42 - 4 + 1))
                .map (fun k => 10 * (21 * k + 7) + 7)
                .filter (fun n => n >= 107 ∧ n <= 997) in
  count.length = 39 :=
by
  sorry

end count_three_digit_integers_end7_divby21_l308_308675


namespace positive_integers_between_300_and_1000_squared_l308_308242

theorem positive_integers_between_300_and_1000_squared :
  ∃ n : ℕ, 300 < n^2 ∧ n^2 < 1000 → ∃ m : ℕ, m = 14 := sorry

end positive_integers_between_300_and_1000_squared_l308_308242


namespace find_smallest_number_l308_308457

theorem find_smallest_number 
  : ∃ x : ℕ, (x - 18) % 14 = 0 ∧ (x - 18) % 26 = 0 ∧ (x - 18) % 28 = 0 ∧ (x - 18) / Nat.lcm 14 (Nat.lcm 26 28) = 746 ∧ x = 271562 := by
  sorry

end find_smallest_number_l308_308457


namespace root_expression_equals_181_div_9_l308_308556

noncomputable def polynomial_root_sum (a b c : ℝ)
  (h1 : a + b + c = 15)
  (h2 : a*b + b*c + c*a = 22) 
  (h3 : a*b*c = 8) : ℝ :=
  (a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b)) 

theorem root_expression_equals_181_div_9
  (a b c : ℝ)
  (h1 : a + b + c = 15)
  (h2 : a*b + b*c + c*a = 22)
  (h3 : a*b*c = 8) :
  polynomial_root_sum a b c h1 h2 h3 = 181 / 9 := by 
  sorry

end root_expression_equals_181_div_9_l308_308556


namespace find_q_l308_308377

theorem find_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1 / p + 1 / q = 1) (h4 : p * q = 6) : q = 3 + Real.sqrt 3 :=
by
  sorry

end find_q_l308_308377


namespace xyz_value_l308_308494

theorem xyz_value (x y z : ℝ) (h1 : y = x + 1) (h2 : x + y = 2 * z) (h3 : x = 3) : x * y * z = 42 :=
by
  -- proof here
  sorry

end xyz_value_l308_308494


namespace opposite_of_neg_one_fourth_l308_308441

def opposite_of (x : ℝ) : ℝ := -x

theorem opposite_of_neg_one_fourth :
  opposite_of (-1/4) = 1/4 :=
by
  sorry

end opposite_of_neg_one_fourth_l308_308441


namespace avg_rate_first_half_l308_308717

theorem avg_rate_first_half (total_distance : ℕ) (avg_rate : ℕ) (first_half_distance : ℕ) (second_half_distance : ℕ)
  (rate_first_half : ℕ) (time_first_half : ℕ) (time_second_half : ℕ) (total_time : ℕ) :
  total_distance = 640 →
  second_half_distance = first_half_distance →
  time_second_half = 3 * time_first_half →
  avg_rate = 40 →
  total_distance = first_half_distance + second_half_distance →
  total_time = time_first_half + time_second_half →
  avg_rate = total_distance / total_time →
  time_first_half = first_half_distance / rate_first_half →
  rate_first_half = 80
  :=
  sorry

end avg_rate_first_half_l308_308717


namespace concert_duration_l308_308908

def duration_in_minutes (hours : Int) (extra_minutes : Int) : Int :=
  hours * 60 + extra_minutes

theorem concert_duration : duration_in_minutes 7 45 = 465 :=
by
  sorry

end concert_duration_l308_308908


namespace miles_total_instruments_l308_308118

theorem miles_total_instruments :
  let fingers := 10
  let hands := 2
  let heads := 1
  let trumpets := fingers - 3
  let guitars := hands + 2
  let trombones := heads + 2
  let french_horns := guitars - 1
  (trumpets + guitars + trombones + french_horns) = 17 :=
by
  sorry

end miles_total_instruments_l308_308118


namespace find_n_values_l308_308234

theorem find_n_values (n : ℤ) (hn : ∃ x y : ℤ, x ≠ y ∧ x^2 - 6*x - 4*n^2 - 32*n = 0 ∧ y^2 - 6*y - 4*n^2 - 32*n = 0):
  n = 10 ∨ n = 0 ∨ n = -8 ∨ n = -18 := 
sorry

end find_n_values_l308_308234


namespace last_integer_in_sequence_l308_308873

theorem last_integer_in_sequence : ∀ (n : ℕ), n = 1000000 → (∀ k : ℕ, n = k * 3 → k * 3 < n) → n = 1000000 :=
by
  intro n hn hseq
  have h := hseq 333333 sorry
  exact hn

end last_integer_in_sequence_l308_308873


namespace average_temperature_l308_308428

theorem average_temperature (T_tue T_wed T_thu : ℝ) 
  (h1 : (42 + T_tue + T_wed + T_thu) / 4 = 48)
  (T_fri : ℝ := 34) :
  ((T_tue + T_wed + T_thu + T_fri) / 4 = 46) :=
by
  sorry

end average_temperature_l308_308428


namespace simplify_and_evaluate_l308_308130

theorem simplify_and_evaluate (m : ℝ) (h : m = Real.sqrt 2) :
  (1 + 1 / (m - 2)) / ((m^2 - m) / (m - 2)) = Real.sqrt 2 / 2 :=
by
  sorry

end simplify_and_evaluate_l308_308130


namespace pies_from_36_apples_l308_308283

-- Definitions of conditions
def pies_from_apples (apples : Nat) : Nat :=
  apples / 4  -- because 12 apples = 3 pies implies 1 pie = 4 apples

-- Theorem to prove
theorem pies_from_36_apples : pies_from_apples 36 = 9 := by
  sorry

end pies_from_36_apples_l308_308283


namespace John_is_26_l308_308545

-- Define the variables representing the ages
def John_age : ℕ := 26
def Grandmother_age : ℕ := John_age + 48

-- Conditions
def condition1 : Prop := John_age = Grandmother_age - 48
def condition2 : Prop := John_age + Grandmother_age = 100

-- Main theorem to prove: John is 26 years old
theorem John_is_26 : John_age = 26 :=
by
  have h1 : condition1 := by sorry
  have h2 : condition2 := by sorry
  -- More steps to combine the conditions and prove the theorem would go here
  -- Skipping proof steps with sorry for demonstration
  sorry

end John_is_26_l308_308545


namespace bob_pennies_l308_308085

theorem bob_pennies (a b : ℕ) (h1 : b + 1 = 4 * (a - 1)) (h2 : b - 1 = 3 * (a + 1)) : b = 31 :=
by
  have h3 : 4 * a - b = 5, from sorry,
  have h4 : b - 3 * a = 4, from sorry,
  have h5 : 4 * a - 3 * a = 9, from sorry,
  have h6 : a = 9, from sorry,
  have h7 : b + 1 = 36 - 4, from sorry,
  have h8 : b + 1 = 32, from sorry,
  have h9 : b = 31, from sorry,
  exact h9

end bob_pennies_l308_308085


namespace Freddy_age_l308_308190

noncomputable def M : ℕ := 11
noncomputable def R : ℕ := M - 2
noncomputable def F : ℕ := M + 4

theorem Freddy_age : F = 15 :=
  by
    sorry

end Freddy_age_l308_308190


namespace bob_pennies_l308_308087

theorem bob_pennies (a b : ℕ) (h1 : b + 1 = 4 * (a - 1)) (h2 : b - 1 = 3 * (a + 1)) : b = 31 :=
by
  have h3 : 4 * a - b = 5, from sorry,
  have h4 : b - 3 * a = 4, from sorry,
  have h5 : 4 * a - 3 * a = 9, from sorry,
  have h6 : a = 9, from sorry,
  have h7 : b + 1 = 36 - 4, from sorry,
  have h8 : b + 1 = 32, from sorry,
  have h9 : b = 31, from sorry,
  exact h9

end bob_pennies_l308_308087


namespace probability_no_shaded_square_l308_308031

noncomputable theory

-- Define n as the number of rectangles from a row of 2012 vertical segments
def total_rectangles (n : ℕ) : ℕ := (2012 * 2011) / 2

-- Define m as the number of rectangles containing the shaded square 
def shaded_rectangles (m : ℕ) : ℕ := 1006 * 1006

-- Define the probability calculation
def probability_no_shaded : ℚ := 1 - (shaded_rectangles 1006 / total_rectangles 1006)

-- The theorem to be proven
theorem probability_no_shaded_square : probability_no_shaded = (1005 / 2011) := by
  sorry

end probability_no_shaded_square_l308_308031


namespace original_price_of_RAM_l308_308471

variables (P : ℝ)

-- Conditions extracted from the problem statement
def priceAfterFire (P : ℝ) : ℝ := 1.30 * P
def priceAfterDecrease (P : ℝ) : ℝ := 1.04 * P

-- The given current price
axiom current_price : priceAfterDecrease P = 52

-- Theorem to prove the original price P
theorem original_price_of_RAM : P = 50 :=
sorry

end original_price_of_RAM_l308_308471


namespace find_sum_u_v_l308_308373

theorem find_sum_u_v : ∃ (u v : ℚ), 5 * u - 6 * v = 35 ∧ 3 * u + 5 * v = -10 ∧ u + v = -40 / 43 :=
by
  sorry

end find_sum_u_v_l308_308373


namespace marion_score_correct_l308_308870

-- Definitions based on conditions
def total_items : ℕ := 40
def ella_incorrect : ℕ := 4
def ella_correct : ℕ := total_items - ella_incorrect
def marion_score : ℕ := (ella_correct / 2) + 6

-- Statement of the theorem
theorem marion_score_correct : marion_score = 24 :=
by
  -- proof goes here
  sorry

end marion_score_correct_l308_308870


namespace sqrt_of_square_of_neg_five_eq_five_l308_308931

theorem sqrt_of_square_of_neg_five_eq_five : Real.sqrt ((-5 : ℤ) ^ 2) = 5 := by
  sorry

end sqrt_of_square_of_neg_five_eq_five_l308_308931


namespace cube_edge_length_l308_308429

-- Define edge length and surface area
variables (edge_length surface_area : ℝ)

-- Given condition
def surface_area_condition : Prop := surface_area = 294

-- Cube surface area formula
def cube_surface_area : Prop := surface_area = 6 * edge_length^2

-- Proof statement
theorem cube_edge_length (h1: surface_area_condition surface_area) (h2: cube_surface_area edge_length surface_area) : edge_length = 7 := 
by
  sorry

end cube_edge_length_l308_308429


namespace bricks_required_l308_308774

   -- Definitions from the conditions
   def courtyard_length_meters : ℝ := 42
   def courtyard_width_meters : ℝ := 22
   def brick_length_cm : ℝ := 16
   def brick_width_cm : ℝ := 10

   -- The Lean statement to prove
   theorem bricks_required : (courtyard_length_meters * courtyard_width_meters * 10000) / (brick_length_cm * brick_width_cm) = 57750 :=
   by 
       sorry
   
end bricks_required_l308_308774


namespace find_y_l308_308982

theorem find_y (x y : ℤ) (h₁ : x = 4) (h₂ : 3 * x + 2 * y = 30) : y = 9 := 
by
  sorry

end find_y_l308_308982


namespace geometric_mean_of_negatives_l308_308433

theorem geometric_mean_of_negatives :
  ∃ x : ℝ, x^2 = (-2) * (-8) ∧ (x = 4 ∨ x = -4) := by
  sorry

end geometric_mean_of_negatives_l308_308433


namespace good_number_iff_gcd_phi_l308_308616

def is_good_number (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∀ a : ℤ, ∃ m : ℕ, m > 0 ∧ m^m % n = a % n 

theorem good_number_iff_gcd_phi (n : ℕ) (phi : ℕ := Nat.totient n) :
  is_good_number n ↔ Nat.gcd n phi = 1 := 
sorry

end good_number_iff_gcd_phi_l308_308616


namespace find_x_l308_308541

-- Definitions to capture angles and triangle constraints
def angle_sum_triangle (A B C : ℝ) : Prop := A + B + C = 180

def perpendicular (A B : ℝ) : Prop := A + B = 90

-- Given conditions
axiom angle_ABC : ℝ
axiom angle_BAC : ℝ
axiom angle_BCA : ℝ
axiom angle_DCE : ℝ
axiom angle_x : ℝ

-- Specific values for the angles provided in the problem
axiom angle_ABC_is_70 : angle_ABC = 70
axiom angle_BAC_is_50 : angle_BAC = 50

-- Angle BCA in triangle ABC
axiom angle_sum_ABC : angle_sum_triangle angle_ABC angle_BAC angle_BCA

-- Conditional relationships in triangle CDE
axiom angle_DCE_equals_BCA : angle_DCE = angle_BCA
axiom angle_sum_CDE : perpendicular angle_DCE angle_x

-- The theorem we need to prove
theorem find_x : angle_x = 30 := sorry

end find_x_l308_308541


namespace find_numbers_l308_308630

def seven_digit_number (n : ℕ) : Prop := 10^6 ≤ n ∧ n < 10^7

theorem find_numbers (x y : ℕ) (hx: seven_digit_number x) (hy: seven_digit_number y) :
  10^7 * x + y = 3 * x * y → x = 1666667 ∧ y = 3333334 :=
by
  sorry

end find_numbers_l308_308630


namespace coordinate_equation_solution_l308_308198

theorem coordinate_equation_solution (x y : ℝ) :
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 →
  (y = -x - 2) ∨ (y = -2 * x + 1) :=
by
  sorry

end coordinate_equation_solution_l308_308198


namespace length_of_BC_l308_308482

theorem length_of_BC (a : ℝ) (b_x b_y c_x c_y area : ℝ) 
  (h1 : b_y = b_x ^ 2)
  (h2 : c_y = c_x ^ 2)
  (h3 : b_y = c_y)
  (h4 : area = 64) :
  c_x - b_x = 8 := by
sorry

end length_of_BC_l308_308482


namespace at_least_one_greater_than_16000_l308_308314

open Nat

theorem at_least_one_greater_than_16000 (seq : Fin 20 → ℕ)
  (h_distinct : ∀ i j : Fin 20, i ≠ j → seq i ≠ seq j)
  (h_perfect_square : ∀ i : Fin 19, ∃ k : ℕ, (seq i) * (seq (i + 1)) = k^2)
  (h_first : seq 0 = 42) : ∃ i : Fin 20, seq i > 16000 :=
by
  sorry

end at_least_one_greater_than_16000_l308_308314


namespace jovana_initial_shells_l308_308709

theorem jovana_initial_shells (x : ℕ) (h₁ : x + 12 = 17) : x = 5 :=
by
  -- Proof omitted
  sorry

end jovana_initial_shells_l308_308709


namespace parallel_lines_slope_eq_l308_308297

theorem parallel_lines_slope_eq {a : ℝ} : (∀ x : ℝ, 2*x - 1 = a*x + 1) → a = 2 :=
by
  sorry

end parallel_lines_slope_eq_l308_308297


namespace average_of_remaining_numbers_l308_308294

theorem average_of_remaining_numbers (s : ℝ) (a b c d e f : ℝ)
  (h1: (a + b + c + d + e + f) / 6 = 3.95)
  (h2: (a + b) / 2 = 4.4)
  (h3: (c + d) / 2 = 3.85) :
  ((e + f) / 2 = 3.6) :=
by
  sorry

end average_of_remaining_numbers_l308_308294


namespace part_i_part_ii_l308_308403

-- Define the problem data
noncomputable def a_0 : ℝ := 0
variables {k : ℕ} (a b : Fin k.succ → ℝ)

-- Existence of such polynomials p_n that meet the conditions in part (i)
theorem part_i (n : ℕ) (h : n > k) :
  ∃ (p : ℝ[X]), degree p ≤ n ∧
    (∀ i : Fin k.succ, (derivative^[i] p).eval (-1) = a i) ∧
    (∀ i : Fin k.succ, (derivative^[i] p).eval 1 = b i) ∧
    (∀ x : ℝ, abs x ≤ 1 → abs (p.eval x) ≤ c / n^2) :=
sorry

-- Impossibility of the relation in part (ii)
theorem part_ii :
  ¬ (∀ (n : ℕ), ∃ (p : ℝ[X]), degree p = n ∧
      (∀ i : Fin k.succ, (derivative^[i] p).eval (-1) = a i) ∧
      (∀ i : Fin k.succ, (derivative^[i] p).eval 1 = b i) ∧
      (tendsto (λ n, n^2 * (⨆ x, abs x ≤ 1 → abs (p.eval x))) at_top (nhds 0))) :=
sorry

end part_i_part_ii_l308_308403


namespace evaluate_expression_l308_308886

theorem evaluate_expression : 4 * (8 - 3) - 7 = 13 := by
  sorry

end evaluate_expression_l308_308886


namespace parabola_pass_through_fixed_point_l308_308368

theorem parabola_pass_through_fixed_point
  (p : ℝ) (hp : p > 0)
  (xM yM : ℝ) (hM : (xM, yM) = (1, -2))
  (hMp : yM^2 = 2 * p * xM)
  (xA yA xC yC xB yB xD yD : ℝ)
  (hxA : xA = xC ∨ xA ≠ xC)
  (hxB : xB = xD ∨ xB ≠ xD)
  (x2 y0 : ℝ) (h : (x2, y0) = (2, 0))
  (m1 m2 : ℝ) (hm1m2 : m1 * m2 = -1)
  (l1_intersect_A : xA = m1 * yA + 2)
  (l1_intersect_C : xC = m1 * yC + 2)
  (l2_intersect_B : xB = m2 * yB + 2)
  (l2_intersect_D : xD = m2 * yD + 2)
  (hMidM : (2 * xA + 2 * xC = 4 * xM ∧ 2 * yA + 2 * yC = 4 * yM))
  (hMidN : (2 * xB + 2 * xD = 4 * xM ∧ 2 * yB + 2 * yD = 4 * yM)) :
  (yM^2 = 4 * xM) ∧ 
  (∃ k : ℝ, ∀ x : ℝ, y = k * x ↔ y = xM / (m1 + m2) ∧ y = m1) :=
sorry

end parabola_pass_through_fixed_point_l308_308368


namespace glass_cannot_all_be_upright_l308_308445

def glass_flip_problem :=
  ∀ (g : Fin 6 → ℤ),
    g 0 = 1 ∧ g 1 = 1 ∧ g 2 = 1 ∧ g 3 = 1 ∧ g 4 = 1 ∧ g 5 = -1 →
    (∀ (flip : Fin 4 → Fin 6 → ℤ),
      (∃ (i1 i2 i3 i4: Fin 6), 
        flip 0 = g i1 * -1 ∧ 
        flip 1 = g i2 * -1 ∧
        flip 2 = g i3 * -1 ∧
        flip 3 = g i4 * -1) →
      ∃ j, g j ≠ 1)

theorem glass_cannot_all_be_upright : glass_flip_problem :=
  sorry

end glass_cannot_all_be_upright_l308_308445


namespace sqrt_15_estimate_l308_308500

theorem sqrt_15_estimate : 3 < Real.sqrt 15 ∧ Real.sqrt 15 < 4 :=
by
  sorry

end sqrt_15_estimate_l308_308500


namespace arithmetic_sequence_sum_l308_308389

theorem arithmetic_sequence_sum {a_n : ℕ → ℤ} (d : ℤ) (S : ℕ → ℤ) 
  (h_seq : ∀ n, a_n (n + 1) = a_n n + d)
  (h_sum : ∀ n, S n = (n * (2 * a_n 1 + (n - 1) * d)) / 2)
  (h_condition : a_n 1 = 2 * a_n 3 - 3) : 
  S 9 = 27 :=
sorry

end arithmetic_sequence_sum_l308_308389


namespace correct_equation_for_programmers_l308_308028

theorem correct_equation_for_programmers (x : ℕ) 
  (hB : x > 0) 
  (programmer_b_speed : ℕ := x) 
  (programmer_a_speed : ℕ := 2 * x) 
  (data : ℕ := 2640) :
  (data / programmer_a_speed = data / programmer_b_speed - 120) :=
by
  -- sorry is used to skip the proof, focus on the statement
  sorry

end correct_equation_for_programmers_l308_308028


namespace flight_duration_l308_308991

theorem flight_duration (h m : ℕ) (Hh : h = 2) (Hm : m = 32) : h + m = 34 := by
  sorry

end flight_duration_l308_308991


namespace jello_mix_needed_per_pound_l308_308261

variable (bathtub_volume : ℝ) (gallons_per_cubic_foot : ℝ) 
          (pounds_per_gallon : ℝ) (cost_per_tablespoon : ℝ) 
          (total_cost : ℝ)

theorem jello_mix_needed_per_pound :
  bathtub_volume = 6 ∧
  gallons_per_cubic_foot = 7.5 ∧
  pounds_per_gallon = 8 ∧
  cost_per_tablespoon = 0.50 ∧
  total_cost = 270 →
  (total_cost / cost_per_tablespoon) / 
  (bathtub_volume * gallons_per_cubic_foot * pounds_per_gallon) = 1.5 :=
by
  sorry

end jello_mix_needed_per_pound_l308_308261


namespace coordinate_equation_solution_l308_308201

theorem coordinate_equation_solution (x y : ℝ) :
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 →
  (y = -x - 2) ∨ (y = -2 * x + 1) :=
by
  sorry

end coordinate_equation_solution_l308_308201


namespace arithmetic_sequence_divisible_by_2005_l308_308093

-- Problem Statement
theorem arithmetic_sequence_divisible_by_2005
  (a : ℕ → ℕ) -- Define the arithmetic sequence
  (d : ℕ) -- Common difference
  (h_arith_seq : ∀ n, a (n + 1) = a n + d) -- Arithmetic sequence condition
  (h_product_div_2005 : ∀ n, 2005 ∣ (a n) * (a (n + 31))) -- Given condition on product divisibility
  : ∀ n, 2005 ∣ a n := 
sorry

end arithmetic_sequence_divisible_by_2005_l308_308093


namespace probability_of_one_radio_operator_per_group_l308_308036

def total_ways_to_assign_soldiers_to_groups : ℕ := 27720
def ways_to_assign_radio_operators_to_groups : ℕ := 7560

theorem probability_of_one_radio_operator_per_group :
  (ways_to_assign_radio_operators_to_groups : ℚ) / (total_ways_to_assign_soldiers_to_groups : ℚ) = 3 / 11 := 
sorry

end probability_of_one_radio_operator_per_group_l308_308036


namespace fraction_addition_l308_308948

-- Definitions from conditions
def frac1 : ℚ := 18 / 42
def frac2 : ℚ := 2 / 9
def simplified_frac1 : ℚ := 3 / 7
def simplified_frac2 : ℚ := frac2
def common_denom_frac1 : ℚ := 27 / 63
def common_denom_frac2 : ℚ := 14 / 63

-- The problem statement to prove
theorem fraction_addition :
  frac1 + frac2 = 41 / 63 := by
  sorry

end fraction_addition_l308_308948


namespace sum_6n_is_correct_l308_308984

theorem sum_6n_is_correct {n : ℕ} (h : (5 * n * (5 * n + 1)) / 2 = (n * (n + 1)) / 2 + 200) :
  (6 * n * (6 * n + 1)) / 2 = 300 :=
by sorry

end sum_6n_is_correct_l308_308984


namespace average_distinct_k_values_l308_308231

theorem average_distinct_k_values (k : ℕ) (h : ∃ r1 r2 : ℕ, r1 * r2 = 24 ∧ r1 + r2 = k ∧ r1 > 0 ∧ r2 > 0) : k = 15 :=
sorry

end average_distinct_k_values_l308_308231


namespace electricity_cost_per_kWh_is_14_cents_l308_308995

-- Define the conditions
def powerUsagePerHour : ℕ := 125 -- watts
def dailyUsageHours : ℕ := 4 -- hours
def weeklyCostInCents : ℕ := 49 -- cents
def daysInWeek : ℕ := 7 -- days
def wattsToKilowattsFactor : ℕ := 1000 -- conversion factor

-- Define a function to calculate the cost per kWh
def costPerKwh (powerUsagePerHour : ℕ) (dailyUsageHours : ℕ) (weeklyCostInCents : ℕ) (daysInWeek : ℕ) (wattsToKilowattsFactor : ℕ) : ℕ :=
  let dailyConsumption := powerUsagePerHour * dailyUsageHours
  let weeklyConsumption := dailyConsumption * daysInWeek
  let weeklyConsumptionInKwh := weeklyConsumption / wattsToKilowattsFactor
  weeklyCostInCents / weeklyConsumptionInKwh

-- State the theorem
theorem electricity_cost_per_kWh_is_14_cents :
  costPerKwh powerUsagePerHour dailyUsageHours weeklyCostInCents daysInWeek wattsToKilowattsFactor = 14 :=
by
  sorry

end electricity_cost_per_kWh_is_14_cents_l308_308995


namespace evaluate_expression_l308_308797

theorem evaluate_expression (x : ℝ) (h1 : x^5 + 1 ≠ 0) (h2 : x^5 - 1 ≠ 0) :
  ( ((x^2 - 2*x + 2)^2 * (x^3 - x^2 + 1)^2 / (x^5 + 1)^2)^2 *
    ((x^2 + 2*x + 2)^2 * (x^3 + x^2 + 1)^2 / (x^5 - 1)^2)^2 )
  = 1 := 
by 
  sorry

end evaluate_expression_l308_308797


namespace gcd_256_162_450_l308_308025

theorem gcd_256_162_450 : Nat.gcd (Nat.gcd 256 162) 450 = 2 := sorry

end gcd_256_162_450_l308_308025


namespace coefficient_x_6_in_expansion_l308_308318

-- Define the variable expressions and constraints of the problem
def expansion_expr : ℕ → ℤ := λ k, Nat.choose 4 k * 1^(4 - k) * (-3)^(k)
def term_coefficient_of_x_pow_6 (k : ℕ) : ℕ := if (3 * k = 6) then Nat.choose 4 k * 9 else 0

-- Prove that the coefficient of x^6 in the expansion of (1-3x^3)^4 is 54
theorem coefficient_x_6_in_expansion : term_coefficient_of_x_pow_6 2 = 54 := by
  -- Simplify the expression for the term coefficient of x^6 when k = 2
  simp only [term_coefficient_of_x_pow_6]
  split_ifs
  simp [Nat.choose, Nat.factorial]
  sorry -- one could continue simplifying this manually or provide arithmetic through Lean library

end coefficient_x_6_in_expansion_l308_308318


namespace LaShawn_twice_Kymbrea_after_25_months_l308_308401

theorem LaShawn_twice_Kymbrea_after_25_months : 
  ∀ (x : ℕ), (10 + 6 * x = 2 * (30 + 2 * x)) → x = 25 :=
by
  intro x
  sorry

end LaShawn_twice_Kymbrea_after_25_months_l308_308401


namespace goods_train_length_l308_308909

theorem goods_train_length 
  (v_kmph : ℝ) (L_p : ℝ) (t : ℝ) (v_mps : ℝ) (d : ℝ) (L_t : ℝ) 
  (h1 : v_kmph = 96) 
  (h2 : L_p = 480) 
  (h3 : t = 36) 
  (h4 : v_mps = v_kmph * (5/18)) 
  (h5 : d = v_mps * t) : 
  L_t = d - L_p :=
sorry

end goods_train_length_l308_308909


namespace bob_speed_lt_40_l308_308488

theorem bob_speed_lt_40 (v_b v_a : ℝ) (h1 : v_a > 45) (h2 : 180 / v_a < 180 / v_b - 0.5) :
  v_b < 40 :=
by
  -- Variables and constants
  let distance := 180
  let min_speed_alice := 45
  -- Conditions
  have h_distance := distance
  have h_min_speed_alice := min_speed_alice
  have h_time_alice := (distance : ℝ) / v_a
  have h_time_bob := (distance : ℝ) / v_b
  -- Given conditions inequalities
  have ineq := h2
  have alice_min_speed := h1
  -- Now apply these facts and derived inequalities to prove bob_speed_lt_40
  sorry

end bob_speed_lt_40_l308_308488


namespace number_of_triangles_with_perimeter_20_l308_308298

-- Declare the condition: number of triangles with integer side lengths and perimeter of 20
def integerTrianglesWithPerimeter (n : ℕ) : ℕ :=
  (Finset.range (n/2 + 1)).card

/-- Prove that the number of triangles with integer side lengths and a perimeter of 20 is 8. -/
theorem number_of_triangles_with_perimeter_20 : integerTrianglesWithPerimeter 20 = 8 := 
  sorry

end number_of_triangles_with_perimeter_20_l308_308298


namespace trapezium_area_l308_308903

-- Definitions based on the problem conditions
def length_side_a : ℝ := 20
def length_side_b : ℝ := 18
def distance_between_sides : ℝ := 15

-- Statement of the proof problem
theorem trapezium_area :
  (1 / 2 * (length_side_a + length_side_b) * distance_between_sides) = 285 := by
  sorry

end trapezium_area_l308_308903


namespace currency_conversion_l308_308251

variable (a : ℚ)

theorem currency_conversion
  (h1 : (0.5 / 100) * a = 75 / 100) -- 0.5% of 'a' = 75 paise
  (rate_usd : ℚ := 0.012)          -- Conversion rate (USD/INR)
  (rate_eur : ℚ := 0.010)          -- Conversion rate (EUR/INR)
  (rate_gbp : ℚ := 0.009)          -- Conversion rate (GBP/INR)
  (paise_to_rupees : ℚ := 1 / 100) -- 1 Rupee = 100 paise
  : (a * paise_to_rupees * rate_usd = 1.8) ∧
    (a * paise_to_rupees * rate_eur = 1.5) ∧
    (a * paise_to_rupees * rate_gbp = 1.35) :=
by
  sorry

end currency_conversion_l308_308251


namespace marble_203_is_green_l308_308782

-- Define the conditions
def total_marbles : ℕ := 240
def cycle_length : ℕ := 15
def red_count : ℕ := 6
def green_count : ℕ := 5
def blue_count : ℕ := 4
def marble_pattern (n : ℕ) : String :=
  if n % cycle_length < red_count then "red"
  else if n % cycle_length < red_count + green_count then "green"
  else "blue"

-- Define the color of the 203rd marble
def marble_203 : String := marble_pattern 202

-- State the theorem
theorem marble_203_is_green : marble_203 = "green" :=
by
  sorry

end marble_203_is_green_l308_308782


namespace range_of_a_for_zero_l308_308370

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 2 * x + a

theorem range_of_a_for_zero (a : ℝ) : a ≤ 2 * Real.log 2 - 2 → ∃ x : ℝ, f a x = 0 := by
  sorry

end range_of_a_for_zero_l308_308370


namespace set_of_points_l308_308270

theorem set_of_points (x y : ℝ) (h : x^2 * y - y ≥ 0) :
  (y ≥ 0 ∧ |x| ≥ 1) ∨ (y ≤ 0 ∧ |x| ≤ 1) :=
sorry

end set_of_points_l308_308270


namespace glasses_per_pitcher_l308_308306

theorem glasses_per_pitcher (t p g : ℕ) (ht : t = 54) (hp : p = 9) : g = t / p := by
  rw [ht, hp]
  norm_num
  sorry

end glasses_per_pitcher_l308_308306


namespace minimize_b_plus_4c_l308_308094

noncomputable def triangle := Type

variable {ABC : triangle}
variable (a b c : ℝ) -- sides of the triangle
variable (BAC : ℝ) -- angle BAC
variable (D : triangle → ℝ) -- angle bisector intersecting BC at D
variable (AD : ℝ) -- length of AD
variable (min_bc : ℝ) -- minimum value of b + 4c

-- Conditions
variable (h1 : BAC = 120)
variable (h2 : D ABC = 1)
variable (h3 : AD = 1)

-- Proof statement
theorem minimize_b_plus_4c (h1 : BAC = 120) (h2 : D ABC = 1) (h3 : AD = 1) : min_bc = 9 := 
sorry

end minimize_b_plus_4c_l308_308094


namespace change_is_correct_l308_308918

-- Define the cost of the pencil in cents
def cost_of_pencil : ℕ := 35

-- Define the amount paid in cents
def amount_paid : ℕ := 100

-- State the theorem for the change
theorem change_is_correct : amount_paid - cost_of_pencil = 65 :=
by sorry

end change_is_correct_l308_308918


namespace incorrect_average_l308_308134

theorem incorrect_average (S : ℕ) (A_correct : ℕ) (A_incorrect : ℕ) (S_correct : ℕ) 
  (h1 : S = 135)
  (h2 : A_correct = 19)
  (h3 : A_incorrect = (S + 25) / 10)
  (h4 : S_correct = (S + 55) / 10)
  (h5 : S_correct = A_correct) :
  A_incorrect = 16 :=
by
  -- The proof will go here, which is skipped with a 'sorry'
  sorry

end incorrect_average_l308_308134


namespace find_value_of_squares_l308_308555

-- Defining the conditions
variable (a b c : ℝ)
variable (h1 : a^2 + 3 * b = 10)
variable (h2 : b^2 + 5 * c = 0)
variable (h3 : c^2 + 7 * a = -21)

-- Stating the theorem to prove the desired result
theorem find_value_of_squares : a^2 + b^2 + c^2 = 83 / 4 :=
   sorry

end find_value_of_squares_l308_308555


namespace speed_of_first_train_l308_308331

noncomputable def length_of_first_train : ℝ := 280
noncomputable def speed_of_second_train_kmph : ℝ := 80
noncomputable def length_of_second_train : ℝ := 220.04
noncomputable def time_to_cross : ℝ := 9

noncomputable def relative_speed_mps := (length_of_first_train + length_of_second_train) / time_to_cross

noncomputable def relative_speed_kmph := relative_speed_mps * (3600 / 1000)

theorem speed_of_first_train :
  (relative_speed_kmph - speed_of_second_train_kmph) = 120.016 :=
by
  sorry

end speed_of_first_train_l308_308331


namespace birch_count_is_87_l308_308330

def num_trees : ℕ := 130
def incorrect_signs (B L : ℕ) : Prop := B + L = num_trees ∧ L + 1 = num_trees - 1 ∧ B = 87

theorem birch_count_is_87 (B L : ℕ) (h1 : B + L = num_trees) (h2 : L + 1 = num_trees - 1) :
  B = 87 :=
sorry

end birch_count_is_87_l308_308330


namespace simplify_polynomial_subtraction_l308_308733

variable (x : ℝ)

def P1 : ℝ := 2*x^6 + x^5 + 3*x^4 + x^3 + 5
def P2 : ℝ := x^6 + 2*x^5 + x^4 - x^3 + 7
def P3 : ℝ := x^6 - x^5 + 2*x^4 + 2*x^3 - 2

theorem simplify_polynomial_subtraction : (P1 x - P2 x) = P3 x :=
by
  sorry

end simplify_polynomial_subtraction_l308_308733


namespace gcd_of_product_diff_is_12_l308_308642

theorem gcd_of_product_diff_is_12
  (a b c d : ℤ) : ∃ (D : ℤ), D = 12 ∧
  ∀ (a b c d : ℤ), D ∣ (b - a) * (c - b) * (d - c) * (d - a) * (c - a) * (d - b) :=
by
  use 12
  sorry

end gcd_of_product_diff_is_12_l308_308642


namespace probability_five_common_correct_l308_308836

-- Define the conditions
def compulsory_subjects : ℕ := 3  -- Chinese, Mathematics, and English
def elective_from_physics_history : ℕ := 1  -- Physics and History
def elective_from_four : ℕ := 4  -- Politics, Geography, Chemistry, Biology

def chosen_subjects_by_xiaoming_xiaofang : ℕ := 2  -- two subjects from the four electives

-- Calculate total combinations
noncomputable def total_combinations : ℕ := Nat.choose 4 2 * Nat.choose 4 2

-- Calculate combinations to have exactly five subjects in common
noncomputable def combinations_five_common : ℕ := Nat.choose 4 2 * Nat.choose 2 1 * Nat.choose 2 1

-- Calculate the probability
noncomputable def probability_five_common : ℚ := combinations_five_common / total_combinations

-- The theorem to be proved
theorem probability_five_common_correct : probability_five_common = 2 / 3 := by
  sorry

end probability_five_common_correct_l308_308836


namespace floor_e_equals_two_l308_308208

/-- Prove that the floor of Euler's number is 2. -/
theorem floor_e_equals_two : (⌊Real.exp 1⌋ = 2) :=
sorry

end floor_e_equals_two_l308_308208


namespace fraction_product_simplification_l308_308183

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 3 / 7 := 
by
  sorry

end fraction_product_simplification_l308_308183


namespace probability_green_ball_is_half_l308_308022

def Container := Finset (String × ℚ)

def Set1 : List Container :=
[ {("A", 8 / 10), ("A", 2 / 10)},  -- Container A: 8 green, 2 red
  {("B", 2 / 10), ("B", 8 / 10)},  -- Container B: 2 green, 8 red
  {("C", 2 / 10), ("C", 8 / 10)} ] -- Container C: 2 green, 8 red

def Set2 : List Container :=
[ {("A", 2 / 10), ("A", 8 / 10)},  -- Container A: 2 green, 8 red
  {("B", 8 / 10), ("B", 2 / 10)},  -- Container B: 8 green, 2 red
  {("C", 8 / 10), ("C", 2 / 10)} ] -- Container C: 8 green, 2 red

noncomputable def probabilityOfGreen (sets : List Container) : ℚ :=
  (1 / 2) * (1 / 3 * (8 / 10) + 1 / 3 * (2 / 10) + 1 / 3 * (2 / 10)) +
  (1 / 2) * (1 / 3 * (8 / 10) + 1 / 3 * (8 / 10) + 1 / 3 * (2 / 10))

theorem probability_green_ball_is_half : probabilityOfGreen Set1 = 1 / 2 :=
by
  sorry

end probability_green_ball_is_half_l308_308022


namespace race_time_l308_308092

theorem race_time 
    (v_A v_B t_A t_B : ℝ)
    (h1 : v_A = 1000 / t_A) 
    (h2 : v_B = 940 / t_A)
    (h3 : v_B = 1000 / (t_A + 15)) 
    (h4 : t_B = t_A + 15) :
    t_A = 235 := 
  by
    sorry

end race_time_l308_308092


namespace max_T_n_at_2_l308_308131

noncomputable def geom_seq (a n : ℕ) : ℕ :=
  a * 2 ^ n

noncomputable def S_n (a n : ℕ) : ℕ :=
  a * (2 ^ n - 1)

noncomputable def T_n (a n : ℕ) : ℕ :=
  (17 * S_n a n - S_n a (2 * n)) / geom_seq a n

theorem max_T_n_at_2 (a : ℕ) : (∀ n > 0, T_n a n ≤ T_n a 2) :=
by
  -- proof omitted
  sorry

end max_T_n_at_2_l308_308131


namespace additional_wolves_in_pack_l308_308594

-- Define the conditions
def wolves_out_hunting : ℕ := 4
def meat_per_wolf_per_day : ℕ := 8
def hunting_days : ℕ := 5
def meat_per_deer : ℕ := 200

-- Calculate total meat per wolf for hunting days
def meat_per_wolf_total : ℕ := meat_per_wolf_per_day * hunting_days

-- Calculate wolves fed per deer
def wolves_fed_per_deer : ℕ := meat_per_deer / meat_per_wolf_total

-- Calculate total deer killed by wolves out hunting
def total_deers_killed : ℕ := wolves_out_hunting

-- Calculate total meat provided by hunting wolves
def total_meat_provided : ℕ := total_deers_killed * meat_per_deer

-- Calculate number of wolves fed by total meat provided
def total_wolves_fed : ℕ := total_meat_provided / meat_per_wolf_total

-- Define the main theorem to prove the answer
theorem additional_wolves_in_pack (total_wolves_fed wolves_out_hunting : ℕ) : 
  total_wolves_fed - wolves_out_hunting = 16 :=
by
  sorry

end additional_wolves_in_pack_l308_308594


namespace range_of_a_l308_308404

def A (x : ℝ) : Prop := x^2 - 4 * x + 3 ≤ 0
def B (x : ℝ) (a : ℝ) : Prop := x^2 - a * x < x - a

theorem range_of_a (a : ℝ) :
  (∀ x, A x → B x a) ∧ ∃ x, ¬ (A x → B x a) ↔ 1 ≤ a ∧ a ≤ 3 := 
sorry

end range_of_a_l308_308404


namespace proof_problem_l308_308221

variables {a b c d e : ℝ}

theorem proof_problem (h1 : a * b^2 * c^3 * d^4 * e^5 < 0) (h2 : b^2 ≥ 0) (h3 : d^4 ≥ 0) :
  a * b^2 * c * d^4 * e < 0 :=
sorry

end proof_problem_l308_308221


namespace sum_all_possible_N_values_l308_308495

theorem sum_all_possible_N_values :
  let n := 5 in
  let max_intersections := (n * (n - 1)) / 2 in
  let all_possible_N := finset.range (max_intersections + 1).to_list in
  all_possible_N.sum = 55 :=
by
  sorry

end sum_all_possible_N_values_l308_308495


namespace upper_limit_of_prime_range_l308_308875

theorem upper_limit_of_prime_range : 
  ∃ x : ℝ, (26 / 3 < 11) ∧ (11 < x) ∧ (x < 17) :=
by
  sorry

end upper_limit_of_prime_range_l308_308875


namespace time_taken_by_A_l308_308537

-- Definitions for the problem conditions
def race_distance : ℕ := 1000  -- in meters
def A_beats_B_by_distance : ℕ := 48  -- in meters
def A_beats_B_by_time : ℕ := 12  -- in seconds

-- The formal statement to prove in Lean
theorem time_taken_by_A :
  ∃ T_a : ℕ, (1000 * (T_a + 12) = 952 * T_a) ∧ T_a = 250 :=
by
  sorry

end time_taken_by_A_l308_308537


namespace trigonometric_identity_l308_308462

variable (α : ℝ)

theorem trigonometric_identity :
  4.9 * (Real.sin (7 * Real.pi / 8 - 2 * α))^2 - (Real.sin (9 * Real.pi / 8 - 2 * α))^2 = 
  Real.sin (4 * α) / Real.sqrt 2 :=
by
  sorry

end trigonometric_identity_l308_308462


namespace ellipse_foci_distance_sum_l308_308812

theorem ellipse_foci_distance_sum
    (x y : ℝ)
    (PF1 PF2 : ℝ)
    (a : ℝ)
    (h_ellipse : (x^2 / 36) + (y^2 / 16) = 1)
    (h_foci : ∀F1 F2, ∃e > 0, F1 = (e, 0) ∧ F2 = (-e, 0))
    (h_point_on_ellipse : ∀x y, (x^2 / 36) + (y^2 / 16) = 1 → (x, y) = (PF1, PF2))
    (h_semi_major_axis : a = 6):
    |PF1| + |PF2| = 12 := 
by
  sorry

end ellipse_foci_distance_sum_l308_308812


namespace combined_salaries_l308_308743

theorem combined_salaries (A B C D E : ℝ) 
  (hC : C = 11000) 
  (hAverage : (A + B + C + D + E) / 5 = 8200) : 
  A + B + D + E = 30000 := 
by 
  sorry

end combined_salaries_l308_308743


namespace symmetric_point_P_l308_308571

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Define the function to get the symmetric point with respect to the origin
def symmetric_point (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.1, -point.2)

-- State the theorem that proves the symmetric point of P is (-1, 2)
theorem symmetric_point_P :
  symmetric_point P = (-1, 2) :=
  sorry

end symmetric_point_P_l308_308571


namespace part1_part2_l308_308362

noncomputable def quadratic_eq (m x : ℝ) : Prop := m * x^2 - 2 * x + 1 = 0

theorem part1 (m : ℝ) : 
  (∃ x1 x2 : ℝ, quadratic_eq m x1 ∧ quadratic_eq m x2 ∧ x1 ≠ x2) ↔ (m ≤ 1 ∧ m ≠ 0) :=
by sorry

theorem part2 (m : ℝ) (x1 x2 : ℝ) : 
  (quadratic_eq m x1 ∧ quadratic_eq m x2 ∧ x1 * x2 - x1 - x2 = 1/2) ↔ (m = -2) :=
by sorry

end part1_part2_l308_308362


namespace part1_part2_part3_l308_308364

-- Define the sequences a_n and b_n as described in the problem
def X_sequence (a : ℕ → ℝ) : Prop :=
  (a 1 = 1) ∧ (∀ n : ℕ, n > 0 → (a n = 0 ∨ a n = 1))

def accompanying_sequence (a b : ℕ → ℝ) : Prop :=
  (b 1 = 1) ∧ (∀ n : ℕ, n > 0 → b (n + 1) = abs (a n - (a (n + 1) / 2)) * b n)

-- 1. Prove the values of b_2, b_3, and b_4
theorem part1 (a b : ℕ → ℝ) (h_a : X_sequence a) (h_b : accompanying_sequence a b) :
  a 2 = 1 → a 3 = 0 → a 4 = 1 →
  b 2 = 1 / 2 ∧ b 3 = 1 / 2 ∧ b 4 = 1 / 4 := 
sorry

-- 2. Prove the equivalence for geometric sequence and constant sequence
theorem part2 (a b : ℕ → ℝ) (h_a : X_sequence a) (h_b : accompanying_sequence a b) :
  (∀ n : ℕ, n > 0 → a n = 1) ↔ (∃ r : ℝ, ∀ n : ℕ, n > 0 → b (n + 1) = r * b n) := 
sorry

-- 3. Prove the maximum value of b_2019
theorem part3 (a b : ℕ → ℝ) (h_a : X_sequence a) (h_b : accompanying_sequence a b) :
  b 2019 ≤ 1 / 2^1009 := 
sorry

end part1_part2_part3_l308_308364


namespace correct_transformation_l308_308617

theorem correct_transformation (x : ℝ) : x^2 - 10 * x - 1 = 0 → (x - 5)^2 = 26 :=
  sorry

end correct_transformation_l308_308617


namespace geometric_sequence_term_l308_308699

theorem geometric_sequence_term (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (n : ℕ) :
  (∀ n, S_n n = 3^n - 1) →
  (a_n n = S_n n - S_n (n - 1)) →
  (a_n n = 2 * 3^(n - 1)) :=
by
  intros h1 h2
  sorry

end geometric_sequence_term_l308_308699


namespace triangle_sine_inequality_l308_308702

theorem triangle_sine_inequality
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (habc : a + b > c)
  (hbac : b + c > a)
  (hact : c + a > b)
  : |(a / (a + b)) + (b / (b + c)) + (c / (c + a)) - (3 / 2)| < (8 * Real.sqrt 2 - 5 * Real.sqrt 5) / 6 := 
sorry

end triangle_sine_inequality_l308_308702


namespace cricket_runs_product_l308_308831

theorem cricket_runs_product :
  let runs_first_10 := [11, 6, 7, 5, 12, 8, 3, 10, 9, 4]
  let total_runs_first_10 := runs_first_10.sum
  let total_runs := total_runs_first_10 + 2 + 7
  2 < 15 ∧ 7 < 15 ∧ (total_runs_first_10 + 2) % 11 = 0 ∧ (total_runs_first_10 + 2 + 7) % 12 = 0 →
  (2 * 7) = 14 :=
by
  intros h
  sorry

end cricket_runs_product_l308_308831


namespace count_integers_satisfying_condition_l308_308244

theorem count_integers_satisfying_condition :
  ({n : ℕ | 300 < n^2 ∧ n^2 < 1000}.card = 14) :=
by
  sorry

end count_integers_satisfying_condition_l308_308244


namespace sign_of_f_based_on_C_l308_308712

def is_triangle (a b c : ℝ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem sign_of_f_based_on_C (a b c : ℝ) (R r : ℝ) (A B C : ℝ)
  (h1 : a = 2 * R * Real.sin A) 
  (h2 : b = 2 * R * Real.sin B) 
  (h3 : c = 2 * R * Real.sin C)
  (h4 : r = 4 * R * Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2))
  (h5 : A + B + C = Real.pi)
  (h_triangle : is_triangle a b c)
  : (a + b - 2 * R - 2 * r > 0 ↔ C < Real.pi / 2) ∧
    (a + b - 2 * R - 2 * r = 0 ↔ C = Real.pi / 2) ∧
    (a + b - 2 * R - 2 * r < 0 ↔ C > Real.pi / 2) :=
sorry

end sign_of_f_based_on_C_l308_308712


namespace coefficient_x6_expansion_l308_308316

theorem coefficient_x6_expansion : 
  (∀ x : ℝ, coefficient (expand (1 - 3 * x ^ 3) 4) x 6 = 54) := 
sorry

end coefficient_x6_expansion_l308_308316


namespace jared_annual_earnings_l308_308506

open Nat

noncomputable def diploma_monthly_pay : ℕ := 4000
noncomputable def months_in_year : ℕ := 12
noncomputable def multiplier : ℕ := 3

theorem jared_annual_earnings :
  let jared_monthly_earnings := diploma_monthly_pay * multiplier
  let jared_annual_earnings := jared_monthly_earnings * months_in_year
  jared_annual_earnings = 144000 :=
by
  let jared_monthly_earnings := diploma_monthly_pay * multiplier
  let jared_annual_earnings := jared_monthly_earnings * months_in_year
  exact sorry

end jared_annual_earnings_l308_308506


namespace usual_eggs_accepted_l308_308253

theorem usual_eggs_accepted (A R : ℝ) (h1 : A / R = 1 / 4) (h2 : (A + 12) / (R - 4) = 99 / 1) (h3 : A + R = 400) :
  A = 392 :=
by
  sorry

end usual_eggs_accepted_l308_308253


namespace value_of_f_at_9_l308_308806

def f (n : ℕ) : ℕ := n^3 + n^2 + n + 17

theorem value_of_f_at_9 : f 9 = 836 := sorry

end value_of_f_at_9_l308_308806


namespace kilos_of_bananas_l308_308850

-- Define the conditions
def initial_money := 500
def remaining_money := 426
def cost_per_kilo_potato := 2
def cost_per_kilo_tomato := 3
def cost_per_kilo_cucumber := 4
def cost_per_kilo_banana := 5
def kilos_potato := 6
def kilos_tomato := 9
def kilos_cucumber := 5

-- Total cost of potatoes, tomatoes, and cucumbers
def total_cost_vegetables : ℕ := 
  (kilos_potato * cost_per_kilo_potato) +
  (kilos_tomato * cost_per_kilo_tomato) +
  (kilos_cucumber * cost_per_kilo_cucumber)

-- Money spent on bananas
def money_spent_on_bananas : ℕ := initial_money - remaining_money - total_cost_vegetables

-- The proof problem statement
theorem kilos_of_bananas : money_spent_on_bananas / cost_per_kilo_banana = 14 :=
by
  -- The sorry is a placeholder for the proof
  sorry

end kilos_of_bananas_l308_308850


namespace probability_of_two_distinct_extreme_points_l308_308669

open Set
open Finset

noncomputable def function_has_two_distinct_extreme_points_probability : ℚ :=
  let a_values := {1, 2, 3}
  let b_values := {0, 1, 2}
  let all_combinations := a_values.product b_values
  let favourable_combinations := all_combinations.filter (λ (ab : ℕ × ℕ), ab.1 > ab.2)
  favourable_combinations.card / all_combinations.card

theorem probability_of_two_distinct_extreme_points :
  function_has_two_distinct_extreme_points_probability = 2 / 3 := by
  sorry

end probability_of_two_distinct_extreme_points_l308_308669


namespace symmetric_sum_l308_308852

theorem symmetric_sum (m n : ℤ) (hA : n = 3) (hB : m = -2) : m + n = 1 :=
by
  rw [hA, hB]
  exact rfl

end symmetric_sum_l308_308852


namespace arithmetic_sequence_sum_l308_308988

noncomputable def sum_of_first_n_terms (n : ℕ) (a d : ℝ) : ℝ :=
  n / 2 * (2 * a + (n - 1) * d)

theorem arithmetic_sequence_sum 
  (a_n : ℕ → ℝ) 
  (h_arith : ∃ d, ∀ n, a_n (n + 1) = a_n n + d) 
  (h1 : a_n 1 + a_n 2 + a_n 3 = 3 )
  (h2 : a_n 28 + a_n 29 + a_n 30 = 165 ) 
  : sum_of_first_n_terms 30 (a_n 1) (a_n 2 - a_n 1) = 840 := 
  sorry

end arithmetic_sequence_sum_l308_308988


namespace marion_score_correct_l308_308871

-- Definitions based on conditions
def total_items : ℕ := 40
def ella_incorrect : ℕ := 4
def ella_correct : ℕ := total_items - ella_incorrect
def marion_score : ℕ := (ella_correct / 2) + 6

-- Statement of the theorem
theorem marion_score_correct : marion_score = 24 :=
by
  -- proof goes here
  sorry

end marion_score_correct_l308_308871


namespace ratio_of_numbers_l308_308388

theorem ratio_of_numbers (a b : ℕ) (ha : a = 45) (hb : b = 60) (lcm_ab : Nat.lcm a b = 180) : (a : ℚ) / b = 3 / 4 :=
by
  sorry

end ratio_of_numbers_l308_308388


namespace sum_of_sequence_l308_308226

-- Definitions based on conditions
def a (n : ℕ) := 2 * n - 1
def b (n : ℕ) := 2^(a n) + n
def S (n : ℕ) := (Finset.range n).sum (λ i => b (i + 1))

-- The theorem assertion / problem statement
theorem sum_of_sequence (n : ℕ) : 
  S n = (2 * (4^n - 1)) / 3 + n * (n + 1) / 2 := 
sorry

end sum_of_sequence_l308_308226


namespace senior_employee_bonus_l308_308136

theorem senior_employee_bonus (J S : ℝ) 
  (h1 : S = J + 1200)
  (h2 : J + S = 5000) : 
  S = 3100 :=
sorry

end senior_employee_bonus_l308_308136


namespace remainder_when_dividing_n_by_d_l308_308081

def n : ℕ := 25197638
def d : ℕ := 4
def r : ℕ := 2

theorem remainder_when_dividing_n_by_d :
  n % d = r :=
by
  sorry

end remainder_when_dividing_n_by_d_l308_308081


namespace number_of_different_towers_l308_308773

theorem number_of_different_towers
  (red blue yellow : ℕ)
  (total_height : ℕ)
  (total_cubes : ℕ)
  (discarded_cubes : ℕ)
  (ways_to_leave_out : ℕ)
  (multinomial_coefficient : ℕ) : 
  red = 3 → blue = 4 → yellow = 5 → total_height = 10 → total_cubes = 12 → discarded_cubes = 2 →
  ways_to_leave_out = 66 → multinomial_coefficient = 4200 →
  (ways_to_leave_out * multinomial_coefficient) = 277200 :=
by
  -- proof skipped
  sorry

end number_of_different_towers_l308_308773


namespace additional_matches_l308_308332

theorem additional_matches 
  (avg_runs_first_25 : ℕ → ℚ) 
  (avg_runs_additional : ℕ → ℚ) 
  (avg_runs_all : ℚ) 
  (total_matches_first_25 : ℕ) 
  (total_matches_all : ℕ) 
  (total_runs_first_25 : ℚ) 
  (total_runs_all : ℚ) 
  (x : ℕ)
  (h1 : avg_runs_first_25 25 = 45)
  (h2 : avg_runs_additional x = 15)
  (h3 : avg_runs_all = 38.4375)
  (h4 : total_matches_first_25 = 25)
  (h5 : total_matches_all = 32)
  (h6 : total_runs_first_25 = avg_runs_first_25 25 * 25)
  (h7 : total_runs_all = avg_runs_all * 32)
  (h8 : total_runs_first_25 + avg_runs_additional x * x = total_runs_all) :
  x = 7 :=
sorry

end additional_matches_l308_308332


namespace no_correct_option_l308_308585

-- Define the given table as a list of pairs
def table :=
  [(1, -2), (2, 0), (3, 2), (4, 6), (5, 12), (6, 20)]

-- Define the given functions as potential options
def optionA (x : ℕ) : ℤ := x^2 - 5 * x + 4
def optionB (x : ℕ) : ℤ := x^2 - 3 * x
def optionC (x : ℕ) : ℤ := x^3 - 3 * x^2 + 2 * x
def optionD (x : ℕ) : ℤ := 2 * x^2 - 4 * x - 2
def optionE (x : ℕ) : ℤ := x^2 - 4 * x + 2

-- Prove that there is no correct option among the given options that matches the table
theorem no_correct_option : 
  ¬(∀ p ∈ table, p.snd = optionA p.fst) ∧
  ¬(∀ p ∈ table, p.snd = optionB p.fst) ∧
  ¬(∀ p ∈ table, p.snd = optionC p.fst) ∧
  ¬(∀ p ∈ table, p.snd = optionD p.fst) ∧
  ¬(∀ p ∈ table, p.snd = optionE p.fst) :=
by sorry

end no_correct_option_l308_308585


namespace sum_of_primes_146_sum_of_primes_99_l308_308956

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Prove that there exist tuples of (p1, p2, p3) such that their sum is 146
theorem sum_of_primes_146 :
  ∃ (p1 p2 p3 : ℕ), is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ p1 + p2 + p3 = 146 ∧ p1 ≤ p2 ∧ p2 ≤ p3 :=
sorry

-- Prove that there exist tuples of (p1, p2, p3) such that their sum is 99
theorem sum_of_primes_99 :
  ∃ (p1 p2 p3 : ℕ), is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ p1 + p2 + p3 = 99 ∧ p1 ≤ p2 ∧ p2 ≤ p3 :=
sorry

end sum_of_primes_146_sum_of_primes_99_l308_308956


namespace vector_magnitude_l308_308523

variables (a b : ℝ^3) 

-- Given conditions
noncomputable def angle_ab := 3 * Real.pi / 4
axiom mag_a : ∥a∥ = Real.sqrt 2
axiom mag_b : ∥b∥ = 3

-- The problem statement
theorem vector_magnitude (h : Real.angle a b = angle_ab) : ∥a + 2 • b∥ = Real.sqrt 26 :=
sorry

end vector_magnitude_l308_308523


namespace problem_solution_l308_308962

theorem problem_solution (a : ℝ) (h : a = Real.sqrt 5 - 1) :
  2 * a^3 + 7 * a^2 - 2 * a - 12 = 0 :=
by 
  sorry  -- Proof placeholder

end problem_solution_l308_308962


namespace pete_nickels_spent_l308_308727

-- Definitions based on conditions
def initial_amount_per_person : ℕ := 250 -- 250 cents for $2.50
def total_initial_amount : ℕ := 2 * initial_amount_per_person
def total_expense : ℕ := 200 -- they spent 200 cents in total
def raymond_dimes_left : ℕ := 7
def value_of_dime : ℕ := 10
def raymond_remaining_amount : ℕ := raymond_dimes_left * value_of_dime
def raymond_spent_amount : ℕ := total_expense - raymond_remaining_amount
def value_of_nickel : ℕ := 5

-- Theorem to prove Pete spent 14 nickels
theorem pete_nickels_spent : 
  (total_expense - raymond_spent_amount) / value_of_nickel = 14 :=
by
  sorry

end pete_nickels_spent_l308_308727


namespace sum_formula_l308_308964

open Nat

/-- The sequence a_n defined as (-1)^n * (2 * n - 1) -/
def a_n (n : ℕ) : ℤ :=
  (-1) ^ n * (2 * n - 1)

/-- The partial sum S_n of the first n terms of the sequence a_n -/
def S_n : ℕ → ℤ
| 0     => 0
| (n+1) => S_n n + a_n (n + 1)

/-- The main theorem: For all n in natural numbers, S_n = (-1)^n * n -/
theorem sum_formula (n : ℕ) : S_n n = (-1) ^ n * n := by
  sorry

end sum_formula_l308_308964


namespace area_of_AFCH_l308_308568

-- Define the lengths of the sides of the rectangles
def AB : ℝ := 9
def BC : ℝ := 5
def EF : ℝ := 3
def FG : ℝ := 10

-- Define the problem statement
theorem area_of_AFCH :
  let intersection_area := min BC FG * min EF AB
  let total_area := AB * FG
  let outer_ring_area := total_area - intersection_area
  intersection_area + outer_ring_area / 2 = 52.5 :=
by
  -- Use the values of AB, BC, EF, and FG to compute
  sorry

end area_of_AFCH_l308_308568


namespace no_common_points_implies_parallel_l308_308775

variable (a : Type) (P : Type) [LinearOrder P] [AddGroupWithOne P]
variable (has_no_common_point : a → P → Prop)
variable (is_parallel : a → P → Prop)

theorem no_common_points_implies_parallel (a_line : a) (a_plane : P) :
  has_no_common_point a_line a_plane ↔ is_parallel a_line a_plane :=
sorry

end no_common_points_implies_parallel_l308_308775


namespace positive_integers_between_300_and_1000_squared_l308_308243

theorem positive_integers_between_300_and_1000_squared :
  ∃ n : ℕ, 300 < n^2 ∧ n^2 < 1000 → ∃ m : ℕ, m = 14 := sorry

end positive_integers_between_300_and_1000_squared_l308_308243


namespace find_m_l308_308524

theorem find_m (m : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ((1/3 : ℝ) * x1^3 - 3 * x1 + m = 0) ∧ ((1/3 : ℝ) * x2^3 - 3 * x2 + m = 0)) ↔ (m = -2 * Real.sqrt 3 ∨ m = 2 * Real.sqrt 3) :=
sorry

end find_m_l308_308524


namespace trip_duration_is_6_hours_l308_308769

def distance_1 := 55 * 4
def total_distance (A : ℕ) := distance_1 + 70 * A
def total_time (A : ℕ) := 4 + A
def average_speed (A : ℕ) := total_distance A / total_time A

theorem trip_duration_is_6_hours (A : ℕ) (h : 60 = average_speed A) : total_time A = 6 :=
by
  sorry

end trip_duration_is_6_hours_l308_308769


namespace exists_j_half_for_all_j_l308_308035

def is_j_half (n j : ℕ) : Prop := 
  ∃ (q : ℕ), n = (2 * j + 1) * q + j

theorem exists_j_half_for_all_j (k : ℕ) : 
  ∃ n : ℕ, ∀ j : ℕ, 1 ≤ j ∧ j ≤ k → is_j_half n j :=
by
  sorry

end exists_j_half_for_all_j_l308_308035


namespace passing_marks_l308_308768

variable (T P : ℝ)

theorem passing_marks :
  (0.35 * T = P - 40) →
  (0.60 * T = P + 25) →
  P = 131 :=
by
  intro h1 h2
  -- Proof steps should follow here.
  sorry

end passing_marks_l308_308768


namespace cars_to_hours_l308_308446

def car_interval := 20 -- minutes
def num_cars := 30
def minutes_per_hour := 60

theorem cars_to_hours :
  (car_interval * num_cars) / minutes_per_hour = 10 := by
  sorry

end cars_to_hours_l308_308446


namespace Diego_half_block_time_l308_308492

def problem_conditions_and_solution : Prop :=
  ∃ (D : ℕ), (3 * 60 + D * 60) / 2 = 240 ∧ D = 5

theorem Diego_half_block_time :
  problem_conditions_and_solution :=
by
  sorry

end Diego_half_block_time_l308_308492


namespace inequality_x_solution_l308_308828

theorem inequality_x_solution (a b c d x : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  ( (a^3 / (a^3 + 15 * b * c * d))^(1/2) = a^x / (a^x + b^x + c^x + d^x) ) ↔ x = 15 / 8 := 
sorry

end inequality_x_solution_l308_308828


namespace arithmetic_sequence_sum_ratio_l308_308592

theorem arithmetic_sequence_sum_ratio (a_n : ℕ → ℕ) (S : ℕ → ℕ) 
  (hS : ∀ n, S n = n * a_n 1 + n * (n - 1) / 2 * (a_n 2 - a_n 1)) 
  (h1 : S 6 / S 3 = 4) : S 9 / S 6 = 9 / 4 := 
by 
  sorry

end arithmetic_sequence_sum_ratio_l308_308592


namespace bob_pennies_l308_308086

theorem bob_pennies (a b : ℕ) (h1 : b + 1 = 4 * (a - 1)) (h2 : b - 1 = 3 * (a + 1)) : b = 31 :=
by
  have h3 : 4 * a - b = 5, from sorry,
  have h4 : b - 3 * a = 4, from sorry,
  have h5 : 4 * a - 3 * a = 9, from sorry,
  have h6 : a = 9, from sorry,
  have h7 : b + 1 = 36 - 4, from sorry,
  have h8 : b + 1 = 32, from sorry,
  have h9 : b = 31, from sorry,
  exact h9

end bob_pennies_l308_308086


namespace evaluate_expression_l308_308893

theorem evaluate_expression : 4 * (8 - 3) - 7 = 13 :=
by sorry

end evaluate_expression_l308_308893


namespace area_of_triangle_PQR_l308_308600

structure Point where
  x : ℝ
  y : ℝ

def P : Point := { x := 2, y := 2 }
def Q : Point := { x := 7, y := 2 }
def R : Point := { x := 5, y := 9 }

noncomputable def triangleArea (A B C : Point) : ℝ :=
  (1 / 2) * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

theorem area_of_triangle_PQR : triangleArea P Q R = 17.5 := by
  sorry

end area_of_triangle_PQR_l308_308600


namespace find_x_value_l308_308560

theorem find_x_value (x : ℚ) (h1 : 9 * x ^ 2 + 8 * x - 1 = 0) (h2 : 27 * x ^ 2 + 65 * x - 8 = 0) : x = 1 / 9 :=
sorry

end find_x_value_l308_308560


namespace employee_pays_correct_amount_l308_308913

theorem employee_pays_correct_amount
    (wholesale_cost : ℝ)
    (retail_markup : ℝ)
    (employee_discount : ℝ)
    (weekend_discount : ℝ)
    (sales_tax : ℝ)
    (final_price : ℝ) :
    wholesale_cost = 200 →
    retail_markup = 0.20 →
    employee_discount = 0.05 →
    weekend_discount = 0.10 →
    sales_tax = 0.08 →
    final_price = 221.62 :=
by
  intros h0 h1 h2 h3 h4
  sorry

end employee_pays_correct_amount_l308_308913


namespace initial_pennies_indeterminate_l308_308128

-- Conditions
def initial_nickels : ℕ := 7
def dad_nickels : ℕ := 9
def mom_nickels : ℕ := 2
def total_nickels_now : ℕ := 18

-- Proof problem statement
theorem initial_pennies_indeterminate :
  ∀ (initial_nickels dad_nickels mom_nickels total_nickels_now : ℕ), 
  initial_nickels = 7 → dad_nickels = 9 → mom_nickels = 2 → total_nickels_now = 18 → 
  (∃ (initial_pennies : ℕ), true) → false :=
by
  sorry

end initial_pennies_indeterminate_l308_308128


namespace final_price_of_jacket_l308_308293

noncomputable def original_price : ℝ := 240
noncomputable def initial_discount : ℝ := 0.6
noncomputable def additional_discount : ℝ := 0.25

theorem final_price_of_jacket :
  let price_after_initial_discount := original_price * (1 - initial_discount)
  let final_price := price_after_initial_discount * (1 - additional_discount)
  final_price = 72 := 
by
  sorry

end final_price_of_jacket_l308_308293


namespace amount_needed_for_free_delivery_l308_308448

theorem amount_needed_for_free_delivery :
  let chicken_cost := 1.5 * 6.00
  let lettuce_cost := 3.00
  let tomatoes_cost := 2.50
  let sweet_potatoes_cost := 4 * 0.75
  let broccoli_cost := 2 * 2.00
  let brussel_sprouts_cost := 2.50
  let total_cost := chicken_cost + lettuce_cost + tomatoes_cost + sweet_potatoes_cost + broccoli_cost + brussel_sprouts_cost
  let min_spend_for_free_delivery := 35.00
  min_spend_for_free_delivery - total_cost = 11.00 := sorry

end amount_needed_for_free_delivery_l308_308448


namespace handshakes_4_handshakes_n_l308_308827

-- Defining the number of handshakes for n people
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

-- Proving that the number of handshakes for 4 people is 6
theorem handshakes_4 : handshakes 4 = 6 := by
  sorry

-- Proving that the number of handshakes for n people is (n * (n - 1)) / 2
theorem handshakes_n (n : ℕ) : handshakes n = (n * (n - 1)) / 2 := by 
  sorry

end handshakes_4_handshakes_n_l308_308827


namespace sum_abc_eq_8_l308_308424

theorem sum_abc_eq_8 (a b c : ℝ) 
  (h : (a - 5) ^ 2 + (b - 6) ^ 2 + (c - 7) ^ 2 - 2 * (a - 5) * (b - 6) = 0) : 
  a + b + c = 8 := 
sorry

end sum_abc_eq_8_l308_308424


namespace smaller_sphere_radius_l308_308187

theorem smaller_sphere_radius (R x : ℝ) (h1 : (4/3) * Real.pi * R^3 = (4/3) * Real.pi * x^3 + (4/3) * Real.pi * (2 * x)^3) 
  (h2 : ∀ r₁ r₂ : ℝ, r₁ / r₂ = 1 / 2 → r₁ = x ∧ r₂ = 2 * x) : x = R / 3 :=
by 
  sorry

end smaller_sphere_radius_l308_308187


namespace whitney_total_cost_l308_308607

-- Definitions of the number of items and their costs
def w := 15
def c_w := 14
def f := 12
def c_f := 13
def s := 5
def c_s := 10
def m := 8
def c_m := 3

-- The total cost Whitney spent
theorem whitney_total_cost :
  w * c_w + f * c_f + s * c_s + m * c_m = 440 := by
  sorry

end whitney_total_cost_l308_308607


namespace teddy_bear_cost_l308_308411

theorem teddy_bear_cost : 
  ∀ (n : ℕ) (cost_per_toy : ℕ) 
  (total_cost : ℕ) (num_teddy_bears : ℕ) 
  (amount_in_wallet : ℕ) (cost_per_bear : ℕ),
  n = 28 → 
  cost_per_toy = 10 → 
  num_teddy_bears = 20 → 
  amount_in_wallet = 580 → 
  total_cost = 280 → 
  total_cost = n * cost_per_toy →
  (amount_in_wallet - total_cost) = num_teddy_bears * cost_per_bear →
  cost_per_bear = 15 :=
by 
  intros n cost_per_toy total_cost num_teddy_bears amount_in_wallet cost_per_bear 
         hn hcost_per_toy hnum_teddy_bears hamount_in_wallet htotal_cost htotal_cost_eq
        hbear_cost_eq,
  sorry

end teddy_bear_cost_l308_308411


namespace simplify_expression_l308_308420

variable (y : ℤ)

theorem simplify_expression : 5 * y + 7 * y - 3 * y = 9 * y := by
  sorry

end simplify_expression_l308_308420


namespace common_chord_length_common_chord_diameter_eq_circle_l308_308072

/-
Given two circles C1: x^2 + y^2 - 2x + 10y - 24 = 0 and C2: x^2 + y^2 + 2x + 2y - 8 = 0,
prove that 
1. The length of the common chord is 2 * sqrt(5).
2. The equation of the circle that has the common chord as its diameter is (x + 8/5)^2 + (y - 6/5)^2 = 36/5.
-/

-- Define the first circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 2 * x + 10 * y - 24 = 0

-- Define the second circle C2
def C2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0

-- Prove the length of the common chord
theorem common_chord_length : ∃ d : ℝ, d = 2 * Real.sqrt 5 :=
sorry

-- Prove the equation of the circle that has the common chord as its diameter
theorem common_chord_diameter_eq_circle : ∃ (x y : ℝ → ℝ), (x + 8/5)^2 + (y - 6/5)^2 = 36/5 :=
sorry

end common_chord_length_common_chord_diameter_eq_circle_l308_308072


namespace events_mutually_exclusive_not_complementary_l308_308794

-- Define the set of balls and people
inductive Ball : Type
| b1 | b2 | b3 | b4

inductive Person : Type
| A | B | C | D

-- Define the event types
structure Event :=
  (p : Person)
  (b : Ball)

-- Define specific events as follows
def EventA : Event := { p := Person.A, b := Ball.b1 }
def EventB : Event := { p := Person.B, b := Ball.b1 }

-- We want to prove the relationship between two specific events:
-- "Person A gets ball number 1" and "Person B gets ball number 1"
-- Namely, that they are mutually exclusive but not complementary.

theorem events_mutually_exclusive_not_complementary :
  (∀ e : Event, (e = EventA → ¬ (e = EventB)) ∧ ¬ (e = EventA ∨ e = EventB)) :=
sorry

end events_mutually_exclusive_not_complementary_l308_308794


namespace fractions_of_group_money_l308_308714

def moneyDistribution (m l n o : ℕ) (moeGave : ℕ) (lokiGave : ℕ) (nickGave : ℕ) : Prop :=
  moeGave = 1 / 5 * m ∧
  lokiGave = 1 / 4 * l ∧
  nickGave = 1 / 3 * n ∧
  moeGave = lokiGave ∧
  lokiGave = nickGave ∧
  o = moeGave + lokiGave + nickGave

theorem fractions_of_group_money (m l n o total : ℕ) :
  moneyDistribution m l n o 1 1 1 →
  total = m + l + n →
  (o : ℚ) / total = 1 / 4 :=
by sorry

end fractions_of_group_money_l308_308714


namespace saree_blue_stripes_l308_308313

theorem saree_blue_stripes (brown_stripes gold_stripes blue_stripes : ℕ) 
    (h1 : brown_stripes = 4)
    (h2 : gold_stripes = 3 * brown_stripes)
    (h3 : blue_stripes = 5 * gold_stripes) : 
    blue_stripes = 60 := 
by
  sorry

end saree_blue_stripes_l308_308313


namespace union_of_sets_l308_308972

def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {3, 4, 5}

theorem union_of_sets : M ∪ N = {2, 3, 4, 5} := by
  sorry

end union_of_sets_l308_308972


namespace total_fireworks_l308_308550

-- Definitions of the given conditions
def koby_boxes : Nat := 2
def koby_box_sparklers : Nat := 3
def koby_box_whistlers : Nat := 5
def cherie_boxes : Nat := 1
def cherie_box_sparklers : Nat := 8
def cherie_box_whistlers : Nat := 9

-- Statement to prove the total number of fireworks
theorem total_fireworks : 
  let koby_fireworks := koby_boxes * (koby_box_sparklers + koby_box_whistlers)
  let cherie_fireworks := cherie_boxes * (cherie_box_sparklers + cherie_box_whistlers)
  koby_fireworks + cherie_fireworks = 33 := by
  sorry

end total_fireworks_l308_308550


namespace probability_without_order_knowledge_correct_probability_with_order_knowledge_correct_l308_308400

def TianJi_top {α : Type} [LinearOrder α] (a1 a2 : α) (b1 : α) : Prop :=
  a2 < b1 ∧ b1 < a1

def TianJi_middle {α : Type} [LinearOrder α] (a3 a2 : α) (b2 : α) : Prop :=
  a3 < b2 ∧ b2 < a2

def TianJi_bottom {α : Type} [LinearOrder α] (a3 : α) (b3 : α) : Prop :=
  b3 < a3

def without_order_knowledge_probability (a1 a2 a3 b1 b2 b3 : ℕ) 
  (h_top : TianJi_top a1 a2 b1) 
  (h_middle : TianJi_middle a3 a2 b2) 
  (h_bottom : TianJi_bottom a3 b3) : ℚ :=
  -- Formula for the probability of Tian Ji winning without knowing the order
  1 / 6

theorem probability_without_order_knowledge_correct (a1 a2 a3 b1 b2 b3 : ℕ) 
  (h_top : TianJi_top a1 a2 b1) 
  (h_middle : TianJi_middle a3 a2 b2) 
  (h_bottom : TianJi_bottom a3 b3) : 
  without_order_knowledge_probability a1 a2 a3 b1 b2 b3 h_top h_middle h_bottom = 1 / 6 :=
sorry

def with_order_knowledge_probability (a1 a2 a3 b1 b2 b3 : ℕ) 
  (h_top : TianJi_top a1 a2 b1) 
  (h_middle : TianJi_middle a3 a2 b2) 
  (h_bottom : TianJi_bottom a3 b3) : ℚ :=
  -- Formula for the probability of Tian Ji winning with specific group knowledge
  1 / 2

theorem probability_with_order_knowledge_correct (a1 a2 a3 b1 b2 b3 : ℕ) 
  (h_top : TianJi_top a1 a2 b1) 
  (h_middle : TianJi_middle a3 a2 b2) 
  (h_bottom : TianJi_bottom a3 b3) : 
  with_order_knowledge_probability a1 a2 a3 b1 b2 b3 h_top h_middle h_bottom = 1 / 2 :=
sorry

end probability_without_order_knowledge_correct_probability_with_order_knowledge_correct_l308_308400


namespace find_x_given_y_l308_308978

theorem find_x_given_y (x y : ℤ) (h1 : 16 * (4 : ℝ)^x = 3^(y + 2)) (h2 : y = -2) : x = -2 := by
  sorry

end find_x_given_y_l308_308978


namespace negative_large_base_zero_exponent_l308_308030

-- Define the problem conditions: base number and exponent
def base_number : ℤ := -2023
def exponent : ℕ := 0

-- Prove that (-2023)^0 equals 1
theorem negative_large_base_zero_exponent : base_number ^ exponent = 1 := by
  sorry

end negative_large_base_zero_exponent_l308_308030


namespace tetrahedron_pairs_l308_308680

theorem tetrahedron_pairs (tetra_edges : ℕ) (h_tetra : tetra_edges = 6) :
  ∀ (num_pairs : ℕ), num_pairs = (tetra_edges * (tetra_edges - 1)) / 2 → num_pairs = 15 :=
by
  sorry

end tetrahedron_pairs_l308_308680


namespace total_amount_shared_l308_308124

theorem total_amount_shared 
  (Parker_share : ℤ)
  (ratio_2 : ℤ)
  (ratio_3 : ℤ)
  (total_parts : ℤ)
  (part_value : ℤ)
  (total_amount : ℤ) :
  Parker_share = 50 →
  ratio_2 = 2 →
  ratio_3 = 3 →
  total_parts = ratio_2 + ratio_3 →
  part_value = Parker_share / ratio_2 →
  total_amount = total_parts * part_value →
  total_amount = 125 :=
by 
  intros hParker_share hratio_2 hratio_3 htotal_parts hpart_value htotal_amount
  rw [hParker_share, hratio_2, hratio_3, htotal_parts, hpart_value, htotal_amount]
  sorry

end total_amount_shared_l308_308124


namespace trigonometric_identity_l308_308080

theorem trigonometric_identity (α : ℝ) 
  (h : Real.cos (5 * Real.pi / 12 - α) = Real.sqrt 2 / 3) :
  Real.sqrt 3 * Real.cos (2 * α) - Real.sin (2 * α) = 10 / 9 := sorry

end trigonometric_identity_l308_308080


namespace selling_price_is_correct_l308_308464

-- Definitions based on conditions
def cost_price : ℝ := 280
def profit_percentage : ℝ := 0.3
def profit_amount : ℝ := cost_price * profit_percentage

-- Selling price definition
def selling_price : ℝ := cost_price + profit_amount

-- Theorem statement
theorem selling_price_is_correct : selling_price = 364 := by
  sorry

end selling_price_is_correct_l308_308464


namespace length_of_platform_l308_308463

noncomputable def train_length : ℝ := 300
noncomputable def time_to_cross_platform : ℝ := 39
noncomputable def time_to_cross_pole : ℝ := 9

theorem length_of_platform : ∃ P : ℝ, P = 1000 :=
by
  let train_speed := train_length / time_to_cross_pole
  let total_distance_cross_platform := train_length + 1000
  let platform_length := total_distance_cross_platform - train_length
  existsi platform_length
  sorry

end length_of_platform_l308_308463


namespace norris_money_left_l308_308723

-- Define the amounts saved each month
def september_savings : ℕ := 29
def october_savings : ℕ := 25
def november_savings : ℕ := 31

-- Define the total savings
def total_savings : ℕ := september_savings + october_savings + november_savings

-- Define the amount spent on the online game
def amount_spent : ℕ := 75

-- Define the remaining money
def money_left : ℕ := total_savings - amount_spent

-- The theorem stating the problem and the solution
theorem norris_money_left : money_left = 10 := by
  sorry

end norris_money_left_l308_308723


namespace foci_distance_of_hyperbola_l308_308496

theorem foci_distance_of_hyperbola :
  ∀ (x y : ℝ), (x^2 / 32 - y^2 / 8 = 1) → 2 * (Real.sqrt (32 + 8)) = 4 * Real.sqrt 10 :=
by
  intros x y h
  sorry

end foci_distance_of_hyperbola_l308_308496


namespace tenth_term_is_correct_l308_308454

-- Define the conditions
def first_term : ℚ := 3
def last_term : ℚ := 88
def num_terms : ℕ := 30
def common_difference : ℚ := (last_term - first_term) / (num_terms - 1)

-- Define the function for the n-th term of the arithmetic sequence
def nth_term (n : ℕ) : ℚ := first_term + (n - 1) * common_difference

-- Prove that the 10th term is 852/29
theorem tenth_term_is_correct : nth_term 10 = 852 / 29 := 
by 
  -- Add the proof later, the statement includes the setup and conditions
  sorry

end tenth_term_is_correct_l308_308454


namespace value_of_x_l308_308979

theorem value_of_x (x y : ℝ) (h1 : x - y = 6) (h2 : x + y = 12) : x = 9 :=
by
  sorry

end value_of_x_l308_308979


namespace pizza_slices_count_l308_308315

/-
  We ordered 21 pizzas. Each pizza has 8 slices. 
  Prove that the total number of slices of pizza is 168.
-/

theorem pizza_slices_count :
  (21 * 8) = 168 :=
by
  sorry

end pizza_slices_count_l308_308315


namespace evaluate_expression_l308_308892

theorem evaluate_expression : 4 * (8 - 3) - 7 = 13 :=
by sorry

end evaluate_expression_l308_308892


namespace max_puzzle_sets_l308_308412

theorem max_puzzle_sets 
  (total_logic : ℕ) (total_visual : ℕ) (total_word : ℕ)
  (h1 : total_logic = 36) (h2 : total_visual = 27) (h3 : total_word = 15)
  (x y : ℕ)
  (h4 : 7 ≤ 4 * x + 3 * x + y ∧ 4 * x + 3 * x + y ≤ 12)
  (h5 : 4 * x / 3 * x = 4 / 3)
  (h6 : y ≥ 3 * x / 2) :
  5 ≤ total_logic / (4 * x) ∧ 5 ≤ total_visual / (3 * x) ∧ 5 ≤ total_word / y :=
sorry

end max_puzzle_sets_l308_308412


namespace proof_of_a_b_and_T_l308_308970

-- Define sequences and the given conditions

def a (n : ℕ) : ℕ := 2^(n-1)

def b (n : ℕ) : ℕ := 2 * n

def S (n : ℕ) : ℕ := 2^n - 1

def c (n : ℕ) : ℚ := 1 / ((b n)^2 - 1)

def T (n : ℕ) : ℚ := (n : ℚ) / (2 * n + 1)

axiom b_condition : ∀ n : ℕ, n > 0 → (b n + 2 * n = 2 * (b (n-1)) + 4)

axiom S_condition : ∀ n : ℕ, S n = 2^n - 1

theorem proof_of_a_b_and_T (n : ℕ) (h : n > 0) : 
  (∀ k, a k = 2^(k-1)) ∧ 
  (∀ k, b k = 2 * k) ∧ 
  (∀ k, T k = (k : ℚ) / (2 * k + 1)) := by
  sorry

end proof_of_a_b_and_T_l308_308970


namespace jared_annual_earnings_l308_308505

open Nat

noncomputable def diploma_monthly_pay : ℕ := 4000
noncomputable def months_in_year : ℕ := 12
noncomputable def multiplier : ℕ := 3

theorem jared_annual_earnings :
  let jared_monthly_earnings := diploma_monthly_pay * multiplier
  let jared_annual_earnings := jared_monthly_earnings * months_in_year
  jared_annual_earnings = 144000 :=
by
  let jared_monthly_earnings := diploma_monthly_pay * multiplier
  let jared_annual_earnings := jared_monthly_earnings * months_in_year
  exact sorry

end jared_annual_earnings_l308_308505


namespace max_area_of_triangle_on_parabola_l308_308264

noncomputable def area_of_triangle_ABC (p : ℝ) : ℝ :=
  (1 / 2) * abs (3 * p^2 - 14 * p + 15)

theorem max_area_of_triangle_on_parabola :
  ∃ p : ℝ, 1 ≤ p ∧ p ≤ 3 ∧ area_of_triangle_ABC p = 2 := sorry

end max_area_of_triangle_on_parabola_l308_308264


namespace median_of_consecutive_integers_l308_308752

theorem median_of_consecutive_integers (a b : ℤ) (h : a + b = 50) : 
  (a + b) / 2 = 25 := 
by 
  sorry

end median_of_consecutive_integers_l308_308752


namespace find_x_plus_y_of_parallel_vectors_l308_308672

theorem find_x_plus_y_of_parallel_vectors 
  (x y : ℝ) 
  (a b : ℝ × ℝ × ℝ)
  (ha : a = (x, 2, -2)) 
  (hb : b = (2, y, 4)) 
  (h_parallel : ∃ k : ℝ, a = k • b) 
  : x + y = -5 := 
by 
  sorry

end find_x_plus_y_of_parallel_vectors_l308_308672


namespace new_mean_when_adding_const_to_each_number_l308_308992

theorem new_mean_when_adding_const_to_each_number :
  ∀ (numbers : Fin 15 → ℝ) (m : ℝ),
    (m = (∑ i, numbers i) / 15) →
    m = 40 →
    (∑ i, (numbers i + 10)) / 15 = 50 :=
by
  intros numbers m hm hmean
  sorry

end new_mean_when_adding_const_to_each_number_l308_308992


namespace train_length_approx_l308_308916

noncomputable def length_of_train (speed_km_hr : ℝ) (time_seconds : ℝ) : ℝ :=
  let speed_m_s := (speed_km_hr * 1000) / 3600
  speed_m_s * time_seconds

theorem train_length_approx (speed_km_hr time_seconds : ℝ) (h_speed : speed_km_hr = 120) (h_time : time_seconds = 4) :
  length_of_train speed_km_hr time_seconds = 133.32 :=
by
  sorry

end train_length_approx_l308_308916


namespace find_a_perpendicular_lines_l308_308073

theorem find_a_perpendicular_lines (a : ℝ) :
  (∀ (x y : ℝ),
    a * x + 2 * y + 6 = 0 → 
    x + (a - 1) * y + a^2 - 1 = 0 → (a * 1 + 2 * (a - 1) = 0)) → 
  a = 2/3 :=
by
  intros h
  sorry

end find_a_perpendicular_lines_l308_308073


namespace number_of_people_in_room_l308_308538

theorem number_of_people_in_room (P : ℕ) 
  (h1 : 1/4 * P = P / 4) 
  (h2 : 3/4 * P = 3 * P / 4) 
  (h3 : P / 4 = 20) : 
  P = 80 :=
sorry

end number_of_people_in_room_l308_308538


namespace parabola_eqn_min_distance_l308_308396

theorem parabola_eqn (a b : ℝ) (h_a : 0 < a)
  (h_A : -a + b = 1)
  (h_B : 4 * a + 2 * b = 0) :
  (∀ x : ℝ,  y = a * x^2 + b * x) ↔ (∀ x : ℝ, y = (1/3) * x^2 - (2/3) * x) :=
by
  sorry

theorem min_distance (a b : ℝ) (h_a : 0 < a)
  (h_A : -a + b = 1)
  (h_B : 4 * a + 2 * b = 0)
  (line_eq : ∀ x, (y : ℝ) = x - 25/4) :
  (∀ P : ℝ × ℝ, ∃ P_min : ℝ × ℝ, P_min = (5/2, 5/12)) :=
by
  sorry

end parabola_eqn_min_distance_l308_308396


namespace fraction_meaningful_l308_308452

theorem fraction_meaningful (a : ℝ) : (a + 3 ≠ 0) ↔ (a ≠ -3) :=
by
  sorry

end fraction_meaningful_l308_308452


namespace patty_heavier_before_losing_weight_l308_308272

theorem patty_heavier_before_losing_weight {w_R w_P w_P' x : ℝ}
  (h1 : w_R = 100)
  (h2 : w_P = 100 * x)
  (h3 : w_P' = w_P - 235)
  (h4 : w_P' = w_R + 115) :
  x = 4.5 :=
by
  sorry

end patty_heavier_before_losing_weight_l308_308272


namespace find_added_value_l308_308994

theorem find_added_value (avg_15_numbers : ℤ) (new_avg : ℤ) (x : ℤ)
    (H1 : avg_15_numbers = 40) 
    (H2 : new_avg = 50) 
    (H3 : (600 + 15 * x) / 15 = new_avg) : 
    x = 10 := 
sorry

end find_added_value_l308_308994


namespace algebra_ineq_example_l308_308843

theorem algebra_ineq_example (x y z : ℝ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2 + y^2 + z^2 = x + y + z) :
  x + y + z + 3 ≥ 6 * ( ( (xy + yz + zx) / 3 ) ^ (1/3) ) :=
by
  sorry

end algebra_ineq_example_l308_308843


namespace solve_sausage_problem_l308_308473

def sausage_problem (x y : ℕ) (condition1 : y = x + 300) (condition2 : x = y + 500) : Prop :=
  x + y = 2 * 400

theorem solve_sausage_problem (x y : ℕ) (h1 : y = x + 300) (h2 : x = y + 500) :
  sausage_problem x y h1 h2 :=
by
  sorry

end solve_sausage_problem_l308_308473


namespace smallest_n_7770_l308_308407

theorem smallest_n_7770 (n : ℕ) 
  (h1 : ∀ d ∈ n.digits 10, d = 0 ∨ d = 7)
  (h2 : 15 ∣ n) : 
  n = 7770 := 
sorry

end smallest_n_7770_l308_308407


namespace num_distinct_solutions_l308_308942

theorem num_distinct_solutions : 
  (∃ x : ℝ, |x - 3| = |x + 5|) ∧ 
  (∀ x1 x2 : ℝ, |x1 - 3| = |x1 + 5| → |x2 - 3| = |x2 + 5| → x1 = x2) := 
  sorry

end num_distinct_solutions_l308_308942


namespace boxes_in_pantry_l308_308842

theorem boxes_in_pantry (b p c: ℕ) (h: p = 100) (hc: c = 50) (g: b = 225) (weeks: ℕ) (consumption: ℕ)
    (total_birdseed: ℕ) (new_boxes: ℕ) (initial_boxes: ℕ) : 
    weeks = 12 → consumption = (100 + 50) * weeks → total_birdseed = 1800 →
    new_boxes = 3 → total_birdseed = b * 8 → initial_boxes = 5 :=
by
  sorry

end boxes_in_pantry_l308_308842


namespace average_nums_correct_l308_308135

def nums : List ℕ := [55, 48, 507, 2, 684, 42]

theorem average_nums_correct :
  (List.sum nums) / (nums.length) = 223 := by
  sorry

end average_nums_correct_l308_308135


namespace nancy_balloons_l308_308847

variable (MaryBalloons : ℝ) (NancyBalloons : ℝ)

theorem nancy_balloons (h1 : NancyBalloons = 4 * MaryBalloons) (h2 : MaryBalloons = 1.75) : 
  NancyBalloons = 7 := 
by 
  sorry

end nancy_balloons_l308_308847


namespace cosine_product_inequality_l308_308846

theorem cosine_product_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  8 * Real.cos A * Real.cos B * Real.cos C ≤ 1 := 
sorry

end cosine_product_inequality_l308_308846


namespace perpendicular_condition_line_through_point_l308_308238

-- Definitions for lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := (m + 2) * x + m * y = 6
def l2 (m : ℝ) (x y : ℝ) : Prop := m * x + y = 3

-- Part 1: Prove that l1 is perpendicular to l2 if and only if m = -3 or m = 0
theorem perpendicular_condition (m : ℝ) : 
  (∀ (x : ℝ), ∀ (y : ℝ), (l1 m x y ∧ l2 m x y) → (m = 0 ∨ m = -3)) :=
sorry

-- Part 2: Prove the equations of line l given the conditions
theorem line_through_point (m : ℝ) (l : ℝ → ℝ → Prop) : 
  (∀ (P : ℝ × ℝ), (P = (1, 2*m)) → (l2 m P.1 P.2) → 
  ((∀ (x y : ℝ), l x y → 2 * x - y = 0) ∨ (∀ (x y: ℝ), l x y → x + 2 * y - 5 = 0))) :=
sorry

end perpendicular_condition_line_through_point_l308_308238


namespace total_gum_correct_l308_308547

def num_cousins : ℕ := 4  -- Number of cousins
def gum_per_cousin : ℕ := 5  -- Pieces of gum per cousin

def total_gum : ℕ := num_cousins * gum_per_cousin  -- Total pieces of gum Kim needs

theorem total_gum_correct : total_gum = 20 :=
by sorry

end total_gum_correct_l308_308547


namespace find_integers_l308_308353

def isPerfectSquare (n : ℤ) : Prop := ∃ m : ℤ, m * m = n

theorem find_integers (x : ℤ) (h : isPerfectSquare (x^2 + 19 * x + 95)) : x = -14 ∨ x = -5 := by
  sorry

end find_integers_l308_308353


namespace compare_f_m_plus_2_l308_308361

theorem compare_f_m_plus_2 (a : ℝ) (ha : a > 0) (m : ℝ) 
  (hf : (a * m^2 + 2 * a * m + 1) < 0) : 
  (a * (m + 2)^2 + 2 * a * (m + 2) + 1) > 1 :=
sorry

end compare_f_m_plus_2_l308_308361


namespace value_g2_l308_308111

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (g (x - y)) = g x * g y - g x + g y - x^3 * y^3

theorem value_g2 : g 2 = 8 :=
by sorry

end value_g2_l308_308111


namespace evaluate_expression_l308_308890

theorem evaluate_expression : 4 * (8 - 3) - 7 = 13 :=
by sorry

end evaluate_expression_l308_308890


namespace lars_total_breads_per_day_l308_308103

def loaves_per_hour : ℕ := 10
def baguettes_per_two_hours : ℕ := 30
def hours_per_day : ℕ := 6

theorem lars_total_breads_per_day :
  (loaves_per_hour * hours_per_day) + ((hours_per_day / 2) * baguettes_per_two_hours) = 150 :=
  by 
  sorry

end lars_total_breads_per_day_l308_308103


namespace sum_first_six_terms_l308_308058

noncomputable def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_first_six_terms :
  geometric_series_sum (1/4) (1/4) 6 = 4095 / 12288 :=
by 
  sorry

end sum_first_six_terms_l308_308058


namespace simple_interest_rate_l308_308343

theorem simple_interest_rate (P A : ℝ) (T : ℝ) (SI : ℝ) (R : ℝ) :
  P = 800 → A = 950 → T = 5 → SI = A - P → SI = (P * R * T) / 100 → R = 3.75 :=
  by
  intros hP hA hT hSI h_formula
  sorry

end simple_interest_rate_l308_308343


namespace multiples_of_six_l308_308821

theorem multiples_of_six (a b : ℕ) (h₁ : a = 5) (h₂ : b = 127) :
  ∃ n : ℕ, n = 21 ∧ ∀ x : ℕ, (a < 6 * x ∧ 6 * x < b) ↔ (1 ≤ x ∧ x ≤ 21) :=
by
  sorry

end multiples_of_six_l308_308821


namespace no_rotation_of_11_gears_l308_308835

theorem no_rotation_of_11_gears :
  ∀ (gears : Fin 11 → ℕ → Prop), 
    (∀ i, gears i 0 ∧ gears (i + 1) 1 → gears i 0 = ¬gears (i + 1) 1) →
    gears 10 0 = gears 0 0 →
    False :=
by
  sorry

end no_rotation_of_11_gears_l308_308835


namespace rectangle_area_with_circles_touching_l308_308478

theorem rectangle_area_with_circles_touching
  (r : ℝ)
  (radius_pos : r = 3)
  (short_side : ℝ)
  (long_side : ℝ)
  (dim_rect : short_side = 2 * r ∧ long_side = 4 * r) :
  short_side * long_side = 72 :=
by
  sorry

end rectangle_area_with_circles_touching_l308_308478


namespace saree_blue_stripes_l308_308312

theorem saree_blue_stripes (brown_stripes gold_stripes blue_stripes : ℕ) 
    (h1 : brown_stripes = 4)
    (h2 : gold_stripes = 3 * brown_stripes)
    (h3 : blue_stripes = 5 * gold_stripes) : 
    blue_stripes = 60 := 
by
  sorry

end saree_blue_stripes_l308_308312


namespace smaller_angle_at_3_20_correct_l308_308456

noncomputable def smaller_angle_at_3_20 : Float :=
  let degrees_per_minute_for_minute_hand := 360 / 60
  let degrees_per_minute_for_hour_hand := 360 / (60 * 12)
  let initial_hour_hand_position := 90.0  -- 3 o'clock position
  let minute_past_three := 20
  let minute_hand_movement := minute_past_three * degrees_per_minute_for_minute_hand
  let hour_hand_movement := minute_past_three * degrees_per_minute_for_hour_hand
  let current_hour_hand_position := initial_hour_hand_position + hour_hand_movement
  let angle_between_hands := minute_hand_movement - current_hour_hand_position
  if angle_between_hands < 0 then
    -angle_between_hands
  else
    angle_between_hands

theorem smaller_angle_at_3_20_correct : smaller_angle_at_3_20 = 20.0 := by
  sorry

end smaller_angle_at_3_20_correct_l308_308456


namespace sqrt_mixed_number_eq_l308_308799

noncomputable def mixed_number : ℝ := 8 + 1 / 9

theorem sqrt_mixed_number_eq : Real.sqrt (8 + 1 / 9) = Real.sqrt 73 / 3 := by
  sorry

end sqrt_mixed_number_eq_l308_308799


namespace smallest_palindrome_not_five_digit_l308_308056

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let s := n.toDigits 10
  s = s.reverse

theorem smallest_palindrome_not_five_digit (n : ℕ) :
  (∃ n, is_palindrome n ∧ 100 ≤ n ∧ n < 1000 ∧ ¬is_palindrome (102 * n)) → n = 101 := by
  sorry

end smallest_palindrome_not_five_digit_l308_308056


namespace mike_corvette_average_speed_l308_308716

theorem mike_corvette_average_speed
  (D : ℚ) (v : ℚ) (total_distance : ℚ)
  (first_half_distance : ℚ) (second_half_time_ratio : ℚ)
  (total_time : ℚ) (average_rate : ℚ) :
  total_distance = 640 ∧
  first_half_distance = total_distance / 2 ∧
  second_half_time_ratio = 3 ∧
  average_rate = 40 →
  v = 80 :=
by
  intros h
  have total_distance_eq : total_distance = 640 := h.1
  have first_half_distance_eq : first_half_distance = total_distance / 2 := h.2.1
  have second_half_time_ratio_eq : second_half_time_ratio = 3 := h.2.2.1
  have average_rate_eq : average_rate = 40 := h.2.2.2
  sorry

end mike_corvette_average_speed_l308_308716


namespace rotated_square_vertical_distance_is_correct_l308_308960

-- Define a setup with four 1-inch squares in a straight line
-- and the second square rotated 45 degrees around its center

-- Noncomputable setup
noncomputable def rotated_square_vert_distance : ℝ :=
  let side_length := 1
  let diagonal := side_length * Real.sqrt 2
  -- Calculate the required vertical distance according to given conditions
  Real.sqrt 2 + side_length / 2

-- Theorem statement confirming the calculated vertical distance
theorem rotated_square_vertical_distance_is_correct :
  rotated_square_vert_distance = Real.sqrt 2 + 1 / 2 :=
by
  sorry

end rotated_square_vertical_distance_is_correct_l308_308960


namespace jared_annual_earnings_l308_308503

-- Defining conditions as constants
def diploma_pay : ℕ := 4000
def degree_multiplier : ℕ := 3
def months_in_year : ℕ := 12

-- Goal: Prove that Jared's annual earnings are $144,000
theorem jared_annual_earnings : diploma_pay * degree_multiplier * months_in_year = 144000 := by
  sorry

end jared_annual_earnings_l308_308503


namespace shaded_area_correct_l308_308796

-- Given definitions
def square_side_length : ℝ := 1
def grid_rows : ℕ := 3
def grid_columns : ℕ := 9

def triangle1_area : ℝ := 3
def triangle2_area : ℝ := 1
def triangle3_area : ℝ := 3
def triangle4_area : ℝ := 3

def total_grid_area := (grid_rows * grid_columns : ℕ) * square_side_length^2
def total_unshaded_area := triangle1_area + triangle2_area + triangle3_area + triangle4_area

-- Problem statement
theorem shaded_area_correct :
  total_grid_area - total_unshaded_area = 17 := 
by
  sorry

end shaded_area_correct_l308_308796


namespace car_speed_l308_308613

theorem car_speed
  (v : ℝ)       -- the unknown speed of the car in km/hr
  (time_80 : ℝ := 45)  -- the time in seconds to travel 1 km at 80 km/hr
  (time_plus_10 : ℝ := 55)  -- the time in seconds to travel 1 km at speed v

  (h1 : time_80 = 3600 / 80)
  (h2 : time_plus_10 = time_80 + 10) :
  v = 3600 / (55 / 3600) := sorry

end car_speed_l308_308613


namespace vegetable_plot_area_l308_308876

variable (V W : ℝ)

theorem vegetable_plot_area (h1 : (1/2) * V + (1/3) * W = 13) (h2 : (1/2) * W + (1/3) * V = 12) : V = 18 :=
by
  sorry

end vegetable_plot_area_l308_308876


namespace minimum_value_op_dot_fp_l308_308829

theorem minimum_value_op_dot_fp (x y : ℝ) (h_ellipse : x^2 / 2 + y^2 = 1) :
  let OP := (x, y)
  let FP := (x - 1, y)
  let dot_product := x * (x - 1) + y^2
  dot_product ≥ 1 / 2 :=
by
  sorry

end minimum_value_op_dot_fp_l308_308829


namespace geometric_sequence_fourth_term_l308_308067

theorem geometric_sequence_fourth_term (x : ℝ) (h1 : (2 * x + 2) ^ 2 = x * (3 * x + 3))
  (h2 : x ≠ -1) : (3*x + 3) * (3/2) = -27/2 :=
by
  sorry

end geometric_sequence_fourth_term_l308_308067


namespace geometric_sum_n_eq_3_l308_308745

theorem geometric_sum_n_eq_3 :
  (∃ n : ℕ, (1 / 2) * (1 - (1 / 3) ^ n) = 728 / 2187) ↔ n = 3 :=
by
  sorry

end geometric_sum_n_eq_3_l308_308745


namespace disprove_prime_statement_l308_308328

theorem disprove_prime_statement : ∃ n : ℕ, ((¬ Nat.Prime n) ∧ Nat.Prime (n + 2)) ∨ (Nat.Prime n ∧ ¬ Nat.Prime (n + 2)) :=
sorry

end disprove_prime_statement_l308_308328


namespace hallie_net_earnings_correct_l308_308529

noncomputable def hallieNetEarnings : ℚ :=
  let monday_hours := 7
  let monday_rate := 10
  let monday_tips := 18
  let tuesday_hours := 5
  let tuesday_rate := 12
  let tuesday_tips := 12
  let wednesday_hours := 7
  let wednesday_rate := 10
  let wednesday_tips := 20
  let thursday_hours := 8
  let thursday_rate := 11
  let thursday_tips := 25
  let thursday_discount := 0.10
  let friday_hours := 6
  let friday_rate := 9
  let friday_tips := 15
  let income_tax := 0.05

  let monday_earnings := monday_hours * monday_rate
  let tuesday_earnings := tuesday_hours * tuesday_rate
  let wednesday_earnings := wednesday_hours * wednesday_rate
  let thursday_earnings := thursday_hours * thursday_rate
  let thursday_earnings_after_discount := thursday_earnings * (1 - thursday_discount)
  let friday_earnings := friday_hours * friday_rate

  let total_hourly_earnings := monday_earnings + tuesday_earnings + wednesday_earnings + thursday_earnings + friday_earnings
  let total_tips := monday_tips + tuesday_tips + wednesday_tips + thursday_tips + friday_tips

  let total_tax := total_hourly_earnings * income_tax
  
  let net_earnings := (total_hourly_earnings - total_tax) - (thursday_earnings - thursday_earnings_after_discount) + total_tips
  net_earnings

theorem hallie_net_earnings_correct : hallieNetEarnings = 406.10 := by
  sorry

end hallie_net_earnings_correct_l308_308529


namespace total_amount_after_5_months_l308_308772

-- Definitions from the conditions
def initial_deposit : ℝ := 100
def monthly_interest_rate : ℝ := 0.0036  -- 0.36% expressed as a decimal

-- Definition of the function relationship y with respect to x
def total_amount (x : ℕ) : ℝ := initial_deposit + initial_deposit * monthly_interest_rate * x

-- Prove the total amount after 5 months is 101.8
theorem total_amount_after_5_months : total_amount 5 = 101.8 :=
by
  sorry

end total_amount_after_5_months_l308_308772


namespace books_shelves_l308_308339

def initial_books : ℝ := 40.0
def additional_books : ℝ := 20.0
def books_per_shelf : ℝ := 4.0

theorem books_shelves :
  (initial_books + additional_books) / books_per_shelf = 15 :=
by 
  sorry

end books_shelves_l308_308339


namespace rectangle_shaded_area_fraction_l308_308416

-- Defining necessary parameters and conditions
variables {R : Type} [LinearOrderedField R]

noncomputable def shaded_fraction (length width : R) : R :=
  let P : R × R := (0, width / 2)
  let Q : R × R := (length / 2, width)
  let rect_area := length * width
  let tri_area := (1 / 2) * (length / 2) * (width / 2)
  let shaded_area := rect_area - tri_area
  shaded_area / rect_area

-- The theorem stating our desired proof goal
theorem rectangle_shaded_area_fraction (length width : R) (h_length : 0 < length) (h_width : 0 < width) :
  shaded_fraction length width = 7 / 8 := by
  sorry

end rectangle_shaded_area_fraction_l308_308416


namespace area_of_circle_l308_308697

-- Given condition as a Lean definition
def circle_eq (x y : ℝ) : Prop := 3 * x^2 + 3 * y^2 + 9 * x - 12 * y - 27 = 0

-- Theorem stating the goal
theorem area_of_circle : ∀ (x y : ℝ), circle_eq x y → ∃ r : ℝ, r = 15.25 ∧ ∃ a : ℝ, a = π * r := 
sorry

end area_of_circle_l308_308697


namespace values_of_fractions_l308_308193

theorem values_of_fractions (A B : ℝ) :
  (∀ x : ℝ, 3 * x ^ 2 + 2 * x - 8 ≠ 0) →
  (∀ x : ℝ, (6 * x - 7) / (3 * x ^ 2 + 2 * x - 8) = A / (x - 2) + B / (3 * x + 4)) →
  A = 1 / 2 ∧ B = 4.5 :=
by
  intros h1 h2
  sorry

end values_of_fractions_l308_308193


namespace larger_circle_radius_l308_308867

theorem larger_circle_radius (r R : ℝ) 
  (h : (π * R^2) / (π * r^2) = 5 / 2) : 
  R = r * Real.sqrt 2.5 :=
sorry

end larger_circle_radius_l308_308867


namespace find_f_3_l308_308586

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : f (x + y) = f x + f y
axiom f_4_eq_6 : f 4 = 6

theorem find_f_3 : f 3 = 9 / 2 :=
by sorry

end find_f_3_l308_308586


namespace identify_functions_l308_308920

-- Define the first expression
def expr1 (x : ℝ) : ℝ := x - (x - 3)

-- Define the second expression
noncomputable def expr2 (x : ℝ) : ℝ := Real.sqrt (x - 2) + Real.sqrt (1 - x)

-- Define the third expression
noncomputable def expr3 (x : ℝ) : ℝ :=
if x < 0 then x - 1 else x + 1

-- Define the fourth expression
noncomputable def expr4 (x : ℝ) : ℝ :=
if x ∈ Set.Ioo (-1) 1 then 0 else 1

-- Proof statement
theorem identify_functions :
  (∀ x, ∃! y, expr1 x = y) ∧ (∀ x, ∃! y, expr3 x = y) ∧
  (¬ ∃ x, ∃! y, expr2 x = y) ∧ (¬ ∀ x, ∃! y, expr4 x = y) := by
    sorry

end identify_functions_l308_308920


namespace length_of_train_l308_308435

-- Define the conditions
def bridge_length : ℕ := 200
def train_crossing_time : ℕ := 60
def train_speed : ℕ := 5

-- Define the total distance traveled by the train while crossing the bridge
def total_distance : ℕ := train_speed * train_crossing_time

-- The problem is to show the length of the train
theorem length_of_train :
  total_distance - bridge_length = 100 :=
by sorry

end length_of_train_l308_308435


namespace square_area_l308_308646

theorem square_area (P : ℝ) (hP : P = 32) : ∃ A : ℝ, A = 64 ∧ A = (P / 4) ^ 2 :=
by {
  sorry
}

end square_area_l308_308646


namespace sqrt_of_square_of_neg_five_eq_five_l308_308932

theorem sqrt_of_square_of_neg_five_eq_five : Real.sqrt ((-5 : ℤ) ^ 2) = 5 := by
  sorry

end sqrt_of_square_of_neg_five_eq_five_l308_308932


namespace measure_time_with_hourglasses_l308_308152

def hourglass7 : ℕ := 7
def hourglass11 : ℕ := 11
def target_time : ℕ := 15

theorem measure_time_with_hourglasses :
  ∃ (time_elapsed : ℕ), time_elapsed = target_time :=
by
  use 15
  sorry

end measure_time_with_hourglasses_l308_308152


namespace remainder_prod_mod_7_l308_308653

theorem remainder_prod_mod_7 : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 4 :=
by
  sorry

end remainder_prod_mod_7_l308_308653


namespace students_not_take_test_l308_308719

theorem students_not_take_test
  (total_students : ℕ)
  (q1_correct : ℕ)
  (q2_correct : ℕ)
  (both_correct : ℕ)
  (h_total : total_students = 29)
  (h_q1 : q1_correct = 19)
  (h_q2 : q2_correct = 24)
  (h_both : both_correct = 19)
  : (total_students - (q1_correct + q2_correct - both_correct) = 5) :=
by
  sorry

end students_not_take_test_l308_308719


namespace compute_alpha_powers_l308_308405

variable (α1 α2 α3 : ℂ)

open Complex

-- Given conditions
def condition1 : Prop := α1 + α2 + α3 = 2
def condition2 : Prop := α1^2 + α2^2 + α3^2 = 6
def condition3 : Prop := α1^3 + α2^3 + α3^3 = 14

-- The required proof statement
theorem compute_alpha_powers (h1 : condition1 α1 α2 α3) (h2 : condition2 α1 α2 α3) (h3 : condition3 α1 α2 α3) :
  α1^7 + α2^7 + α3^7 = 46 := by
  sorry

end compute_alpha_powers_l308_308405


namespace definite_integral_abs_poly_l308_308050

theorem definite_integral_abs_poly :
  ∫ x in (-2 : ℝ)..(2 : ℝ), |x^2 - 2*x| = 8 :=
by
  sorry

end definite_integral_abs_poly_l308_308050


namespace highest_throw_is_37_feet_l308_308350

theorem highest_throw_is_37_feet :
  let C1 := 20
  let J1 := C1 - 4
  let C2 := C1 + 10
  let J2 := J1 * 2
  let C3 := C2 + 4
  let J3 := C1 + 17
  max (max C1 (max C2 C3)) (max J1 (max J2 J3)) = 37 := by
  let C1 := 20
  let J1 := C1 - 4
  let C2 := C1 + 10
  let J2 := J1 * 2
  let C3 := C2 + 4
  let J3 := C1 + 17
  sorry

end highest_throw_is_37_feet_l308_308350


namespace highest_throw_christine_janice_l308_308346

theorem highest_throw_christine_janice
  (c1 : ℕ) -- Christine's first throw
  (j1 : ℕ) -- Janice's first throw
  (c2 : ℕ) -- Christine's second throw
  (j2 : ℕ) -- Janice's second throw
  (c3 : ℕ) -- Christine's third throw
  (j3 : ℕ) -- Janice's third throw
  (h1 : c1 = 20)
  (h2 : j1 = c1 - 4)
  (h3 : c2 = c1 + 10)
  (h4 : j2 = j1 * 2)
  (h5 : c3 = c2 + 4)
  (h6 : j3 = c1 + 17) :
  max c1 (max c2 (max c3 (max j1 (max j2 j3)))) = 37 := by
  sorry

end highest_throw_christine_janice_l308_308346


namespace eval_expression_l308_308897

theorem eval_expression : 4 * (8 - 3) - 7 = 13 :=
by
  sorry

end eval_expression_l308_308897


namespace find_b_l308_308659

-- Let's define the real numbers and the conditions given.
variables (b y a : ℝ)

-- Conditions from the problem
def condition1 := abs (b - y) = b + y - a
def condition2 := abs (b + y) = b + a

-- The goal is to find the value of b
theorem find_b (h1 : condition1 b y a) (h2 : condition2 b y a) : b = 1 :=
by
  sorry

end find_b_l308_308659


namespace expenditure_increase_l308_308015

theorem expenditure_increase (x : ℝ) (h₁ : 3 * x / (3 * x + 2 * x) = 3 / 5)
  (h₂ : 2 * x / (3 * x + 2 * x) = 2 / 5)
  (h₃ : ((5 * x) + 0.15 * (5 * x)) = 5.75 * x) 
  (h₄ : (2 * x + 0.06 * 2 * x) = 2.12 * x) 
  : ((3.63 * x - 3 * x) / (3 * x) * 100) = 21 := 
  by
  sorry

end expenditure_increase_l308_308015


namespace initial_cost_of_smartphone_l308_308117

theorem initial_cost_of_smartphone 
(C : ℝ) 
(h : 0.85 * C = 255) : 
C = 300 := 
sorry

end initial_cost_of_smartphone_l308_308117


namespace lines_from_equation_l308_308203

-- Definitions for the conditions
def satisfies_equation (x y : ℝ) : Prop :=
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Equivalent Lean statement to the proof problem
theorem lines_from_equation :
  (∀ x y : ℝ, satisfies_equation x y → (y = -x - 2) ∨ (y = -2 * x + 1)) :=
by
  intros x y h
  sorry

end lines_from_equation_l308_308203


namespace keystone_arch_larger_angle_l308_308864

def isosceles_trapezoid_larger_angle (n : ℕ) : Prop :=
  n = 10 → ∃ (x : ℝ), x = 99

theorem keystone_arch_larger_angle :
  isosceles_trapezoid_larger_angle 10 :=
by
  sorry

end keystone_arch_larger_angle_l308_308864


namespace conic_sections_of_equation_l308_308497

theorem conic_sections_of_equation :
  ∀ x y : ℝ, y^4 - 9*x^6 = 3*y^2 - 1 →
  (∃ y, y^2 - 3*x^3 = 4 ∨ y^2 + 3*x^3 = 0) :=
by 
  sorry

end conic_sections_of_equation_l308_308497


namespace moving_circle_passes_through_fixed_point_l308_308532

-- Define the parabola x^2 = 12y
def parabola (x y : ℝ) : Prop := x^2 = 12 * y

-- Define the directrix line y = -3
def directrix (y : ℝ) : Prop := y = -3

-- The fixed point we need to show the circle always passes through
def fixed_point : ℝ × ℝ := (0, 3)

-- Define the condition that the moving circle is centered on the parabola and tangent to the directrix
def circle_centered_on_parabola_and_tangent_to_directrix (x y : ℝ) (r : ℝ) : Prop :=
  parabola x y ∧ r = abs (y + 3)

-- Main theorem statement
theorem moving_circle_passes_through_fixed_point :
  (∀ (x y r : ℝ), circle_centered_on_parabola_and_tangent_to_directrix x y r → 
    (∃ (px py : ℝ), (px, py) = fixed_point ∧ (px - x)^2 + (py - y)^2 = r^2)) :=
sorry

end moving_circle_passes_through_fixed_point_l308_308532


namespace spider_legs_is_multiple_of_human_legs_l308_308038

def human_legs : ℕ := 2
def spider_legs : ℕ := 8

theorem spider_legs_is_multiple_of_human_legs :
  spider_legs = 4 * human_legs :=
by 
  sorry

end spider_legs_is_multiple_of_human_legs_l308_308038


namespace eval_expression_l308_308355

theorem eval_expression : (49^2 - 25^2 + 10^2) = 1876 := by
  sorry

end eval_expression_l308_308355


namespace trader_profit_percentage_l308_308902

theorem trader_profit_percentage (P : ℝ) (h₀ : 0 ≤ P) : 
  let discount := 0.40
  let increase := 0.80
  let purchase_price := P * (1 - discount)
  let selling_price := purchase_price * (1 + increase)
  let profit := selling_price - P
  (profit / P) * 100 = 8 := 
by
  sorry

end trader_profit_percentage_l308_308902


namespace molecular_weight_constant_l308_308156

-- Given condition
def molecular_weight (compound : Type) : ℝ := 260

-- Proof problem statement (no proof yet)
theorem molecular_weight_constant (compound : Type) : molecular_weight compound = 260 :=
by
  sorry

end molecular_weight_constant_l308_308156


namespace vinegar_used_is_15_l308_308832

noncomputable def vinegar_used (T : ℝ) : ℝ :=
  let water := (3 / 5) * 20
  let total_volume := 27
  let vinegar := total_volume - water
  vinegar

theorem vinegar_used_is_15 (T : ℝ) (h1 : (3 / 5) * 20 = 12) (h2 : 27 - 12 = 15) (h3 : (5 / 6) * T = 15) : vinegar_used T = 15 :=
by
  sorry

end vinegar_used_is_15_l308_308832


namespace tub_volume_ratio_l308_308609

theorem tub_volume_ratio (C D : ℝ) 
  (h₁ : 0 < C) 
  (h₂ : 0 < D)
  (h₃ : (3/4) * C = (2/3) * D) : 
  C / D = 8 / 9 := 
sorry

end tub_volume_ratio_l308_308609


namespace natural_numbers_equal_power_l308_308663

theorem natural_numbers_equal_power
  (a b n : ℕ)
  (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) :
  a = b^n :=
by
  sorry

end natural_numbers_equal_power_l308_308663


namespace find_eccentricity_of_ellipse_l308_308369

theorem find_eccentricity_of_ellipse (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : c = sqrt (a^2 - b^2)) (h4 : ∃ (M : ℝ × ℝ), M.1 = c ∧ M.2 = 2 / 3 * b) :
  eccentricity a b c = sqrt 5 / 3 :=
by
  sorry

end find_eccentricity_of_ellipse_l308_308369


namespace wally_not_all_numbers_l308_308730

def next_wally_number (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    n / 2
  else
    (n + 1001) / 2

def eventually_print(n: ℕ) : Prop :=
  ∃ k: ℕ, (next_wally_number^[k]) 1 = n

theorem wally_not_all_numbers :
  ¬ ∀ n, n ≤ 100 → eventually_print n :=
by
  sorry

end wally_not_all_numbers_l308_308730


namespace tangent_line_at_neg_ln_2_range_of_a_inequality_range_of_a_zero_point_l308_308815

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

/-- Problem 1 -/
theorem tangent_line_at_neg_ln_2 :
  let x := -Real.log 2
  let y := f x
  ∃ k b : ℝ, (y - b) = k * (x - (-Real.log 2)) ∧ k = (Real.exp x - 1) ∧ b = Real.log 2 + 1/2 :=
sorry

/-- Problem 2 -/
theorem range_of_a_inequality :
  ∀ a : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f x > a * x) ↔ a ∈ Set.Iio (Real.exp 1 - 1) :=
sorry

/-- Problem 3 -/
theorem range_of_a_zero_point :
  ∀ a : ℝ, (∃! x : ℝ, f x - a * x = 0) ↔ a ∈ (Set.Iio (-1) ∪ Set.Ioi (Real.exp 1 - 1)) :=
sorry

end tangent_line_at_neg_ln_2_range_of_a_inequality_range_of_a_zero_point_l308_308815


namespace divisible_by_8640_l308_308129

theorem divisible_by_8640 (x : ℤ) : 8640 ∣ (x^9 - 6 * x^7 + 9 * x^5 - 4 * x^3) :=
  sorry

end divisible_by_8640_l308_308129


namespace norris_money_left_l308_308726

def sept_savings : ℕ := 29
def oct_savings : ℕ := 25
def nov_savings : ℕ := 31
def hugo_spent  : ℕ := 75
def total_savings : ℕ := sept_savings + oct_savings + nov_savings
def norris_left : ℕ := total_savings - hugo_spent

theorem norris_money_left : norris_left = 10 := by
  unfold norris_left total_savings sept_savings oct_savings nov_savings hugo_spent
  sorry

end norris_money_left_l308_308726


namespace tan_alpha_l308_308222

theorem tan_alpha {α : ℝ} (h : Real.tan (α + π / 4) = 9) : Real.tan α = 4 / 5 :=
sorry

end tan_alpha_l308_308222


namespace opposite_of_neg_one_fourth_l308_308442

def opposite_of (x : ℝ) : ℝ := -x

theorem opposite_of_neg_one_fourth :
  opposite_of (-1/4) = 1/4 :=
by
  sorry

end opposite_of_neg_one_fourth_l308_308442


namespace resulting_surface_area_l308_308922

-- Defining the initial condition for the cube structure
def cube_surface_area (side_length : ℕ) : ℕ :=
  6 * side_length^2

-- Defining the structure and the modifications
def initial_structure : ℕ :=
  64 * (cube_surface_area 2)

def removed_cubes_exposure : ℕ :=
  4 * (cube_surface_area 2)

-- The final lean statement to prove the surface area after removing central cubes
theorem resulting_surface_area : initial_structure + removed_cubes_exposure = 1632 := by
  sorry

end resulting_surface_area_l308_308922


namespace odd_terms_in_expansion_l308_308688

theorem odd_terms_in_expansion (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  (finset.range 9).filter (λ k, ((nat.choose 8 k) % 2 = 1) ∧ ((p ^ (8 - k) * q ^ k) % 2 = 1)).card = 2 :=
sorry

end odd_terms_in_expansion_l308_308688


namespace point_reflection_correct_l308_308468

def point_reflection_y_axis (x y z : ℝ) : ℝ × ℝ × ℝ :=
  (-x, y, -z)

theorem point_reflection_correct :
  point_reflection_y_axis (-3) 5 2 = (3, 5, -2) :=
by
  -- The proof would go here
  sorry

end point_reflection_correct_l308_308468


namespace right_triangle_count_l308_308975

theorem right_triangle_count (a b : ℕ) (h1 : b < 100) (h2 : a^2 + b^2 = (b + 2)^2) : 
∃ n, n = 10 :=
by sorry

end right_triangle_count_l308_308975


namespace freddy_age_l308_308191

theorem freddy_age
  (mat_age : ℕ)  -- Matthew's age
  (reb_age : ℕ)  -- Rebecca's age
  (fre_age : ℕ)  -- Freddy's age
  (h1 : mat_age = reb_age + 2)
  (h2 : fre_age = mat_age + 4)
  (h3 : mat_age + reb_age + fre_age = 35) :
  fre_age = 15 :=
by sorry

end freddy_age_l308_308191


namespace calculate_expression_value_l308_308881

theorem calculate_expression_value : 
  3 - ((-3 : ℚ) ^ (-3 : ℤ) * 2) = 83 / 27 := 
by
  sorry

end calculate_expression_value_l308_308881


namespace AE_length_l308_308939

theorem AE_length :
  ∀ (A B C D E : Type) 
    (AB CD AC BD AE EC : ℕ),
  AB = 12 → CD = 15 → AC = 18 → BD = 27 → 
  (AE + EC = AC) → 
  (AE * (18 - AE)) = (4 / 9 * 18 * 8) → 
  9 * AE = 72 → 
  AE = 8 := 
by
  intros A B C D E AB CD AC BD AE EC hAB hCD hAC hBD hSum hEqual hSolve
  sorry

end AE_length_l308_308939


namespace cricket_team_members_l308_308750

theorem cricket_team_members (n : ℕ) 
  (captain_age : ℕ) 
  (wicket_keeper_age : ℕ) 
  (team_avg_age : ℕ) 
  (remaining_avg_age : ℕ) 
  (h1 : captain_age = 26)
  (h2 : wicket_keeper_age = 29)
  (h3 : team_avg_age = 23)
  (h4 : remaining_avg_age = 22) 
  (h5 : team_avg_age * n = remaining_avg_age * (n - 2) + captain_age + wicket_keeper_age) : 
  n = 11 := 
sorry

end cricket_team_members_l308_308750


namespace smallest_value_of_expression_l308_308054

variable (a b c : ℝ)
variable (hab : a > b)
variable (hbc : b > c)
variable (ha_nonzero : a ≠ 0)

theorem smallest_value_of_expression :
  ∃ (x : ℝ), x = 6 ∧ 
  (∀ a b c : ℝ, a > b → b > c → a ≠ 0 →
  (a+b)^2 + (b+c)^2 + (c+a)^2 / a^2 ≥ x) :=
begin
  sorry
end

end smallest_value_of_expression_l308_308054


namespace circle_equation_standard_l308_308302

open Real

noncomputable def equation_of_circle : Prop :=
  ∃ R : ℝ, R = sqrt 2 ∧ 
  (∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 → x + y - 2 = 0 → 0 ≤ x ∧ x ≤ 2)

theorem circle_equation_standard :
    equation_of_circle := sorry

end circle_equation_standard_l308_308302


namespace friends_travelled_distance_l308_308409

theorem friends_travelled_distance :
  let lionel_distance : ℝ := 4 * 5280
  let esther_distance : ℝ := 975 * 3
  let niklaus_distance : ℝ := 1287
  let isabella_distance : ℝ := 18 * 1000 * 3.28084
  let sebastian_distance : ℝ := 2400 * 3.28084
  let total_distance := lionel_distance + esther_distance + niklaus_distance + isabella_distance + sebastian_distance
  total_distance = 91261.136 := 
by
  sorry

end friends_travelled_distance_l308_308409


namespace cupcake_combinations_l308_308344

/-- 
Bill needs to purchase exactly seven cupcakes, and the bakery has five types of cupcakes. 
Bill is required to get at least one of each of the first four types. 
We need to prove that the number of ways for Bill to complete his order is 35.
-/
theorem cupcake_combinations : 
  (∑ x in finset.Ico 4 8, (finset.Ico 4 8).choose x) = 35 := by
begin
  sorry
end

end cupcake_combinations_l308_308344


namespace avg_eq_pos_diff_l308_308387

theorem avg_eq_pos_diff (y : ℝ) (h : (35 + y) / 2 = 42) : |35 - y| = 14 := 
sorry

end avg_eq_pos_diff_l308_308387


namespace floor_e_equals_two_l308_308206

/-- Prove that the floor of Euler's number is 2. -/
theorem floor_e_equals_two : (⌊Real.exp 1⌋ = 2) :=
sorry

end floor_e_equals_two_l308_308206


namespace total_students_is_37_l308_308698

-- Let b be the number of blue swim caps 
-- Let r be the number of red swim caps
variables (b r : ℕ)

-- The number of blue swim caps according to the male sports commissioner
def condition1 : Prop := b = 4 * r + 1

-- The number of blue swim caps according to the female sports commissioner
def condition2 : Prop := b = r + 24

-- The total number of students in the 3rd grade
def total_students : ℕ := b + r

theorem total_students_is_37 (h1 : condition1 b r) (h2 : condition2 b r) : total_students b r = 37 :=
by sorry

end total_students_is_37_l308_308698


namespace avg_rate_first_half_l308_308718

theorem avg_rate_first_half (total_distance : ℕ) (avg_rate : ℕ) (first_half_distance : ℕ) (second_half_distance : ℕ)
  (rate_first_half : ℕ) (time_first_half : ℕ) (time_second_half : ℕ) (total_time : ℕ) :
  total_distance = 640 →
  second_half_distance = first_half_distance →
  time_second_half = 3 * time_first_half →
  avg_rate = 40 →
  total_distance = first_half_distance + second_half_distance →
  total_time = time_first_half + time_second_half →
  avg_rate = total_distance / total_time →
  time_first_half = first_half_distance / rate_first_half →
  rate_first_half = 80
  :=
  sorry

end avg_rate_first_half_l308_308718


namespace arithmetic_expression_value_l308_308882

theorem arithmetic_expression_value : 4 * (8 - 3) - 7 = 13 := by
  sorry

end arithmetic_expression_value_l308_308882


namespace five_digit_number_divisibility_l308_308729

theorem five_digit_number_divisibility (a : ℕ) (h1 : 10000 ≤ a) (h2 : a < 100000) : 11 ∣ 100001 * a :=
by
  sorry

end five_digit_number_divisibility_l308_308729


namespace smallest_constant_for_triangle_l308_308359

theorem smallest_constant_for_triangle 
  (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)  
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) :
  (a^2 + b^2 + c^2) / (a*b + b*c + c*a) < 2 := 
  sorry

end smallest_constant_for_triangle_l308_308359


namespace cost_of_brushes_and_canvas_minimum_canvases_cost_effectiveness_l308_308467

-- Part 1: Prove the cost of one box of brushes and one canvas each.
theorem cost_of_brushes_and_canvas (x y : ℕ) 
    (h₁ : 2 * x + 4 * y = 94) (h₂ : 4 * x + 2 * y = 98) :
    x = 17 ∧ y = 15 := by
  sorry

-- Part 2: Prove the minimum number of canvases.
theorem minimum_canvases (m : ℕ) 
    (h₃ : m + (10 - m) = 10) (h₄ : 17 * (10 - m) + 15 * m ≤ 157) :
    m ≥ 7 := by
  sorry

-- Part 3: Prove the cost-effective purchasing plan.
theorem cost_effectiveness (m n : ℕ) 
    (h₃ : m + n = 10) (h₄ : 17 * n + 15 * m ≤ 157) (h₅ : m ≤ 8) :
    (m = 8 ∧ n = 2) := by
  sorry

end cost_of_brushes_and_canvas_minimum_canvases_cost_effectiveness_l308_308467


namespace cheesecakes_sold_l308_308032

theorem cheesecakes_sold
  (initial_display : Nat)
  (initial_fridge : Nat)
  (left_to_sell : Nat)
  (total_cheesecakes := initial_display + initial_fridge)
  (total_after_sales : Nat) :
  initial_display = 10 →
  initial_fridge = 15 →
  left_to_sell = 18 →
  total_after_sales = total_cheesecakes - left_to_sell →
  total_after_sales = 7 := sorry

end cheesecakes_sold_l308_308032


namespace inscribed_square_side_length_l308_308641

-- Define a right triangle
structure RightTriangle :=
  (PQ : ℝ)
  (QR : ℝ)
  (PR : ℝ)
  (is_right : PQ^2 + QR^2 = PR^2)

-- Define the triangle PQR
def trianglePQR : RightTriangle :=
  { PQ := 6, QR := 8, PR := 10, is_right := by norm_num }

-- Define the problem statement
theorem inscribed_square_side_length (t : ℝ) (h : RightTriangle) :
  t = 3 :=
  sorry

end inscribed_square_side_length_l308_308641


namespace total_number_of_wheels_l308_308399

-- Define the conditions as hypotheses
def cars := 2
def wheels_per_car := 4

def bikes := 2
def trashcans := 1
def wheels_per_bike_or_trashcan := 2

def roller_skates_pair := 1
def wheels_per_skate := 4

def tricycle := 1
def wheels_per_tricycle := 3

-- Prove the total number of wheels
theorem total_number_of_wheels :
  cars * wheels_per_car +
  (bikes + trashcans) * wheels_per_bike_or_trashcan +
  (roller_skates_pair * 2) * wheels_per_skate +
  tricycle * wheels_per_tricycle 
  = 25 :=
by
  sorry

end total_number_of_wheels_l308_308399


namespace multiple_of_michael_trophies_l308_308398

-- Conditions
def michael_current_trophies : ℕ := 30
def michael_trophies_increse : ℕ := 100
def total_trophies_in_three_years : ℕ := 430

-- Proof statement
theorem multiple_of_michael_trophies (x : ℕ) :
  (michael_current_trophies + michael_trophies_increse) + (michael_current_trophies * x) = total_trophies_in_three_years → x = 10 := 
by
  sorry

end multiple_of_michael_trophies_l308_308398


namespace bob_pennies_l308_308084

theorem bob_pennies (a b : ℤ) (h1 : b + 1 = 4 * (a - 1)) (h2 : b - 1 = 3 * (a + 1)) : b = 31 :=
by
  sorry

end bob_pennies_l308_308084


namespace rosie_pies_proof_l308_308287

-- Define the given condition
def pies_per_apples (p: ℕ) (a: ℕ) : ℕ := a / p

-- Given that Rosie can make 3 pies from 12 apples
def given_condition : pies_per_apples 3 12 = 4 := rfl

-- The proof problem statement:
theorem rosie_pies_proof : ∀ (a: ℕ) (n: ℕ), pies_per_apples 3 12 = 4 → pies_per_apples n a = 4 → pies_per_apples n a = 9 :=
begin
  sorry
end

end rosie_pies_proof_l308_308287


namespace at_least_one_vowel_l308_308076

-- Define the set of letters
def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'I'}

-- Define the vowels within the set of letters
def vowels : Finset Char := {'A', 'E', 'I'}

-- Define the consonants within the set of letters
def consonants : Finset Char := {'B', 'C', 'D', 'F'}

-- Function to count the total number of 3-letter words from a given set
def count_words (s : Finset Char) (length : Nat) : Nat :=
  s.card ^ length

-- Define the statement of the problem
theorem at_least_one_vowel : count_words letters 3 - count_words consonants 3 = 279 :=
by
  sorry

end at_least_one_vowel_l308_308076


namespace seeds_germination_l308_308501

theorem seeds_germination (seed_plot1 seed_plot2 : ℕ) (germ_rate2 total_germ_rate : ℝ) (germinated_total_pct : ℝ)
  (h1 : seed_plot1 = 300)
  (h2 : seed_plot2 = 200)
  (h3 : germ_rate2 = 0.35)
  (h4 : germinated_total_pct = 28.999999999999996 / 100) :
  (germinated_total_pct * (seed_plot1 + seed_plot2) - germ_rate2 * seed_plot2) / seed_plot1 * 100 = 25 :=
by sorry  -- Proof not required

end seeds_germination_l308_308501


namespace find_two_digit_number_l308_308954

theorem find_two_digit_number : ∃ (y : ℕ), (10 ≤ y ∧ y < 100) ∧ (∃ x : ℕ, x = (y / 10) + (y % 10) ∧ x^3 = y^2) ∧ y = 27 := 
by
  sorry

end find_two_digit_number_l308_308954


namespace inequality_correct_l308_308753

theorem inequality_correct (a b : ℝ) (h₀ : 0 < a) (h₁ : a < b) (h₂ : b < 1) : (1 - a) ^ a > (1 - b) ^ b :=
sorry

end inequality_correct_l308_308753


namespace number_condition_l308_308423

theorem number_condition (x : ℤ) (h : x - 7 = 9) : 5 * x = 80 := by
  sorry

end number_condition_l308_308423


namespace solve_equation_l308_308360

def euler_totient (n : ℕ) : ℕ := sorry  -- Placeholder, Euler's φ function definition
def sigma_function (n : ℕ) : ℕ := sorry  -- Placeholder, σ function definition

theorem solve_equation (x : ℕ) : euler_totient (sigma_function (2^x)) = 2^x → x = 1 := by
  sorry

end solve_equation_l308_308360


namespace grooming_time_correct_l308_308014

def time_to_groom_poodle : ℕ := 30
def time_to_groom_terrier : ℕ := time_to_groom_poodle / 2
def number_of_poodles : ℕ := 3
def number_of_terriers : ℕ := 8

def total_grooming_time : ℕ :=
  (number_of_poodles * time_to_groom_poodle) + (number_of_terriers * time_to_groom_terrier)

theorem grooming_time_correct :
  total_grooming_time = 210 :=
by
  sorry

end grooming_time_correct_l308_308014


namespace initial_students_count_l308_308856

variable (n T : ℕ)
variables (initial_average remaining_average dropped_score : ℚ)
variables (initial_students remaining_students : ℕ)

theorem initial_students_count :
  initial_average = 62.5 →
  remaining_average = 63 →
  dropped_score = 55 →
  T = initial_average * n →
  T - dropped_score = remaining_average * (n - 1) →
  n = 16 :=
by
  intros h_avg_initial h_avg_remaining h_dropped_score h_total h_total_remaining
  sorry

end initial_students_count_l308_308856


namespace find_divisor_l308_308218

-- Definitions from the condition
def original_number : ℕ := 724946
def least_number_subtracted : ℕ := 6
def remaining_number : ℕ := original_number - least_number_subtracted

theorem find_divisor (h1 : remaining_number % least_number_subtracted = 0) :
  Nat.gcd original_number least_number_subtracted = 2 :=
sorry

end find_divisor_l308_308218


namespace fifty_times_reciprocal_of_eight_times_number_three_l308_308386

theorem fifty_times_reciprocal_of_eight_times_number_three (x : ℚ) 
  (h : 8 * x = 3) : 50 * (1 / x) = 133 + 1 / 3 :=
sorry

end fifty_times_reciprocal_of_eight_times_number_three_l308_308386


namespace eggs_volume_correct_l308_308116

def raw_spinach_volume : ℕ := 40
def cooking_reduction_ratio : ℚ := 0.20
def cream_cheese_volume : ℕ := 6
def total_quiche_volume : ℕ := 18
def cooked_spinach_volume := (raw_spinach_volume : ℚ) * cooking_reduction_ratio
def combined_spinach_and_cream_cheese_volume := cooked_spinach_volume + (cream_cheese_volume : ℚ)
def eggs_volume := (total_quiche_volume : ℚ) - combined_spinach_and_cream_cheese_volume

theorem eggs_volume_correct : eggs_volume = 4 := by
  sorry

end eggs_volume_correct_l308_308116


namespace sqrt_mixed_number_eq_l308_308798

noncomputable def mixed_number : ℝ := 8 + 1 / 9

theorem sqrt_mixed_number_eq : Real.sqrt (8 + 1 / 9) = Real.sqrt 73 / 3 := by
  sorry

end sqrt_mixed_number_eq_l308_308798


namespace smallest_scalene_prime_triangle_perimeter_l308_308171

-- Define a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a scalene triangle with distinct side lengths
def is_scalene (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Define the triangle inequality
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define a valid scalene triangle with prime side lengths
def valid_scalene_triangle (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ is_scalene a b c ∧ triangle_inequality a b c

-- Proof statement
theorem smallest_scalene_prime_triangle_perimeter : ∃ (a b c : ℕ), 
  valid_scalene_triangle a b c ∧ a + b + c = 15 := 
sorry

end smallest_scalene_prime_triangle_perimeter_l308_308171


namespace average_trees_planted_l308_308394

def A := 225
def B := A + 48
def C := A - 24
def total_trees := A + B + C
def average := total_trees / 3

theorem average_trees_planted :
  average = 233 := by
  sorry

end average_trees_planted_l308_308394


namespace product_mod_seven_l308_308652

theorem product_mod_seven :
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 4 :=
by
  have h₁ : 3 % 7 = 3 := rfl,
  have h₂ : 13 % 7 = 6 := rfl,
  have h₃ : 23 % 7 = 2 := rfl,
  have h₄ : 33 % 7 = 5 := rfl,
  have h₅ : 43 % 7 = 1 := rfl,
  have h₆ : 53 % 7 = 4 := rfl,
  have h₇ : 63 % 7 = 0 := rfl,
  have h₈ : 73 % 7 = 3 := rfl,
  have h₉ : 83 % 7 = 6 := rfl,
  have h₁₀ : 93 % 7 = 2 := rfl,
  sorry

end product_mod_seven_l308_308652


namespace base_length_of_prism_l308_308694

theorem base_length_of_prism (V : ℝ) (hV : V = 36 * Real.pi) : ∃ (AB : ℝ), AB = 3 * Real.sqrt 3 :=
by
  sorry

end base_length_of_prism_l308_308694


namespace floor_e_equals_2_l308_308214

theorem floor_e_equals_2 : Int.floor Real.exp = 2 := 
sorry

end floor_e_equals_2_l308_308214


namespace length_of_courtyard_l308_308472

-- Given conditions

def width_of_courtyard : ℝ := 14
def brick_length : ℝ := 0.25
def brick_width : ℝ := 0.15
def total_bricks : ℝ := 8960

-- To be proven
theorem length_of_courtyard : brick_length * brick_width * total_bricks / width_of_courtyard = 24 := 
by sorry

end length_of_courtyard_l308_308472


namespace series_sum_equals_four_l308_308351

/-- 
  Proof of the sum of the series: 
  ∑ (n=1 to ∞) (6n² - n + 1) / (n⁵ - n⁴ + n³ - n² + n) = 4 
--/
theorem series_sum_equals_four :
  (∑' n : ℕ, (if n > 0 then (6 * n^2 - n + 1 : ℝ) / (n^5 - n^4 + n^3 - n^2 + n) else 0)) = 4 :=
by
  sorry

end series_sum_equals_four_l308_308351


namespace coordinate_equation_solution_l308_308199

theorem coordinate_equation_solution (x y : ℝ) :
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 →
  (y = -x - 2) ∨ (y = -2 * x + 1) :=
by
  sorry

end coordinate_equation_solution_l308_308199


namespace division_correct_result_l308_308612

theorem division_correct_result (x : ℝ) (h : 8 * x = 56) : 42 / x = 6 := by
  sorry

end division_correct_result_l308_308612


namespace twenty_five_percent_M_eq_thirty_five_percent_1504_l308_308079

theorem twenty_five_percent_M_eq_thirty_five_percent_1504 (M : ℝ) : 
  0.25 * M = 0.35 * 1504 → M = 2105.6 :=
by
  sorry

end twenty_five_percent_M_eq_thirty_five_percent_1504_l308_308079


namespace solution_pairs_l308_308196

theorem solution_pairs (x y : ℝ) : 
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ (y = -x - 2 ∨ y = -2 * x + 1) := 
by 
  sorry

end solution_pairs_l308_308196


namespace minimum_value_of_f_l308_308670

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x ^ 3 - x + (1 / 3)

theorem minimum_value_of_f :
  (∃ m : ℝ, ∀ x : ℝ, f x ≤ 1) → (∀ x : ℝ, f 1 = -(1 / 3)) :=
by
  sorry

end minimum_value_of_f_l308_308670


namespace Tony_slices_left_after_week_l308_308024

-- Define the conditions and problem statement
def Tony_slices_per_day (days : ℕ) : ℕ := days * 2
def Tony_slices_on_Saturday : ℕ := 3 + 2
def Tony_slice_on_Sunday : ℕ := 1
def Total_slices_used (days : ℕ) : ℕ := Tony_slices_per_day days + Tony_slices_on_Saturday + Tony_slice_on_Sunday
def Initial_loaf : ℕ := 22
def Slices_left (days : ℕ) : ℕ := Initial_loaf - Total_slices_used days

-- Prove that Tony has 6 slices left after a week
theorem Tony_slices_left_after_week : Slices_left 5 = 6 := by
  sorry

end Tony_slices_left_after_week_l308_308024


namespace ax_plus_by_equals_d_set_of_solutions_l308_308266

theorem ax_plus_by_equals_d (a b d : ℤ) (u v : ℤ) (h_d : d = a.gcd b) (h_uv : a * u + b * v = d) :
  ∀ (x y : ℤ), (a * x + b * y = d) ↔ ∃ k : ℤ, x = u + k * b ∧ y = v - k * a :=
by
  sorry

theorem set_of_solutions (a b d : ℤ) (u v : ℤ) (h_d : d = a.gcd b) (h_uv : a * u + b * v = d) :
  {p : ℤ × ℤ | a * p.1 + b * p.2 = d} = {p : ℤ × ℤ | ∃ k : ℤ, p = (u + k * b, v - k * a)} :=
by
  sorry

end ax_plus_by_equals_d_set_of_solutions_l308_308266


namespace batsman_average_after_12th_l308_308795

theorem batsman_average_after_12th (runs_12th : ℕ) (average_increase : ℕ) (initial_innings : ℕ)
   (initial_average : ℝ) (runs_before_12th : ℕ → ℕ) 
   (h1 : runs_12th = 48)
   (h2 : average_increase = 2)
   (h3 : initial_innings = 11)
   (h4 : initial_average = 24)
   (h5 : ∀ i, i < initial_innings → runs_before_12th i ≥ 20)
   (h6 : ∃ i, runs_before_12th i = 25 ∧ runs_before_12th (i + 1) = 25) :
   (11 * initial_average + runs_12th) / 12 = 26 :=
by
  sorry

end batsman_average_after_12th_l308_308795


namespace sum_of_roots_l308_308005

theorem sum_of_roots (g : ℝ → ℝ) 
  (h_symmetry : ∀ x : ℝ, g (3 + x) = g (3 - x))
  (h_roots : ∃ s1 s2 s3 s4 : ℝ, 
               g s1 = 0 ∧ 
               g s2 = 0 ∧ 
               g s3 = 0 ∧ 
               g s4 = 0 ∧ 
               s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ 
               s2 ≠ s3 ∧ s2 ≠ s4 ∧ s3 ≠ s4) :
  s1 + s2 + s3 + s4 = 12 :=
by 
  sorry

end sum_of_roots_l308_308005


namespace quadratic_complete_square_l308_308734

theorem quadratic_complete_square (x m n : ℝ) 
  (h : 9 * x^2 - 36 * x - 81 = 0) :
  (x + m)^2 = n ∧ m + n = 11 :=
sorry

end quadratic_complete_square_l308_308734


namespace field_trip_seniors_l308_308427

theorem field_trip_seniors (n : ℕ) 
  (h1 : n < 300) 
  (h2 : n % 17 = 15) 
  (h3 : n % 19 = 12) : 
  n = 202 :=
  sorry

end field_trip_seniors_l308_308427


namespace trains_cross_time_l308_308309

noncomputable def time_to_cross_trains : ℝ :=
  200 / (89.992800575953935 * (1000 / 3600))

theorem trains_cross_time :
  abs (time_to_cross_trains - 8) < 1e-7 :=
by
  sorry

end trains_cross_time_l308_308309


namespace g100_value_l308_308426

-- Define the function g and its properties
def g (x : ℝ) : ℝ := sorry

theorem g100_value 
  (h : ∀ (x y : ℝ), 0 < x → 0 < y → x * g y - y * g x = g (x / y) + x - y) : 
  g 100 = 99 / 2 := 
sorry

end g100_value_l308_308426


namespace grooming_time_correct_l308_308011

def time_to_groom_poodle : ℕ := 30
def time_to_groom_terrier : ℕ := time_to_groom_poodle / 2
def num_poodles : ℕ := 3
def num_terriers : ℕ := 8

def total_grooming_time : ℕ := 
  (num_poodles * time_to_groom_poodle) + (num_terriers * time_to_groom_terrier)

theorem grooming_time_correct : 
  total_grooming_time = 210 := by
  sorry

end grooming_time_correct_l308_308011


namespace product_of_p_r_s_l308_308822

theorem product_of_p_r_s :
  ∃ p r s : ℕ, 3^p + 3^5 = 252 ∧ 2^r + 58 = 122 ∧ 5^3 * 6^s = 117000 ∧ p * r * s = 36 :=
by
  sorry

end product_of_p_r_s_l308_308822


namespace no_equal_refereed_matches_l308_308700

theorem no_equal_refereed_matches {k : ℕ} (h1 : ∀ {n : ℕ}, n > k → n = 2 * k) 
    (h2 : ∀ {n : ℕ}, n > k → ∃ m, m = k * (2 * k - 1))
    (h3 : ∀ {n : ℕ}, n > k → ∃ r, r = (2 * k - 1) / 2): 
    False := 
by
  sorry

end no_equal_refereed_matches_l308_308700


namespace smallest_n_l308_308154

theorem smallest_n (n : ℕ) : 
  (25 * n = (Nat.lcm 10 (Nat.lcm 16 18)) → n = 29) :=
by sorry

end smallest_n_l308_308154


namespace choose_blue_pair_l308_308390

/-- In a drawer, there are 12 distinguishable socks: 5 white, 3 brown, and 4 blue socks.
    Prove that the number of ways to choose a pair of socks such that both socks are blue is 6. -/
theorem choose_blue_pair (total_socks white_socks brown_socks blue_socks : ℕ)
  (h_total : total_socks = 12) (h_white : white_socks = 5) (h_brown : brown_socks = 3) (h_blue : blue_socks = 4) :
  (blue_socks.choose 2) = 6 :=
by
  sorry

end choose_blue_pair_l308_308390


namespace saree_blue_stripes_l308_308310

theorem saree_blue_stripes :
  ∀ (brown_stripes gold_stripes blue_stripes : ℕ),
    gold_stripes = 3 * brown_stripes →
    blue_stripes = 5 * gold_stripes →
    brown_stripes = 4 →
    blue_stripes = 60 :=
by
  intros brown_stripes gold_stripes blue_stripes h_gold h_blue h_brown
  sorry

end saree_blue_stripes_l308_308310


namespace sum_of_first_15_odd_positive_integers_l308_308637

theorem sum_of_first_15_odd_positive_integers :
  let a := 1
  let d := 2
  let n := 15
  let l := a + (n - 1) * d
  let S_n := (n / 2) * (a + l)
  S_n = 225 :=
by
  let a := 1
  let d := 2
  let n := 15
  let l := a + (n - 1) * d
  let S_n := (n / 2) * (a + l)
  have : S_n = 225 := sorry
  exact this

end sum_of_first_15_odd_positive_integers_l308_308637


namespace curve_C_parametric_eq_common_points_count_l308_308161

-- Definition of the circle's equation and transformation
def circle_eq (x y : ℝ) := x^2 + y^2 = 1
def transform (x y : ℝ) := (x, 2*y)
def curve_C_eq (x y : ℝ) := x^2 + y^2 / 4 = 1

-- Theorem to prove the transformation leads to the equation of curve C
theorem curve_C_parametric_eq : 
  ∀ (x_1 y_1 : ℝ), circle_eq x_1 y_1 → (transform x_1 y_1 = (x, y)) → curve_C_eq x y := sorry

-- Definition of the line and the circle in Cartesian coordinates
def line_eq (x y : ℝ) := x + y = 2
def circle_cart_eq (x y : ℝ) := x^2 + y^2 = 4

-- Theorem to prove the number of common points
theorem common_points_count : 
  ∃! (A B : ℝ × ℝ), (line_eq A.1 A.2 ∧ circle_cart_eq A.1 A.2) ∧ (line_eq B.1 B.2 ∧ circle_cart_eq B.1 B.2) ∧ A ≠ B := sorry

end curve_C_parametric_eq_common_points_count_l308_308161


namespace find_a_l308_308235

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ :=
  (x + a) * Real.log x

noncomputable def curve_deriv (a : ℝ) (x : ℝ) : ℝ :=
  Real.log x + (x + a) / x

theorem find_a (a : ℝ) (h : curve (x := 1) a = 2) : a = 1 :=
by
  have eq1 : curve 1 0 = (1 + a) * 0 := by sorry
  have eq2 : curve 1 1 = (1 + a) * Real.log 1 := by sorry
  have eq3 : curve_deriv a 1 = Real.log 1 + (1 + a) / 1 := by sorry
  have eq4 : 2 = 1 + a := by sorry
  sorry -- Complete proof would follow here

end find_a_l308_308235


namespace unique_solution_l308_308358

def satisfies_equation (m n : ℕ) : Prop :=
  15 * m * n = 75 - 5 * m - 3 * n

theorem unique_solution : satisfies_equation 1 6 ∧ ∀ (m n : ℕ), m > 0 → n > 0 → satisfies_equation m n → (m, n) = (1, 6) :=
by {
  sorry
}

end unique_solution_l308_308358


namespace passenger_drop_ratio_l308_308479

theorem passenger_drop_ratio (initial_passengers passengers_at_first passengers_at_second final_passengers x : ℕ)
  (h0 : initial_passengers = 288)
  (h1 : passengers_at_first = initial_passengers - (initial_passengers / 3) + 280)
  (h2 : passengers_at_second = passengers_at_first - x + 12)
  (h3 : final_passengers = 248)
  (h4 : passengers_at_second = final_passengers) :
  x / passengers_at_first = 1 / 2 :=
by
  sorry

end passenger_drop_ratio_l308_308479


namespace sum_of_logs_l308_308682

open Real

noncomputable def log_base (b a : ℝ) : ℝ := log a / log b

theorem sum_of_logs (x y z : ℝ)
  (h1 : log_base 2 (log_base 4 (log_base 5 x)) = 0)
  (h2 : log_base 3 (log_base 5 (log_base 2 y)) = 0)
  (h3 : log_base 4 (log_base 2 (log_base 3 z)) = 0) :
  x + y + z = 666 := sorry

end sum_of_logs_l308_308682


namespace find_point_on_x_axis_l308_308415

theorem find_point_on_x_axis (a : ℝ) (h : abs (3 * a + 6) = 30) : (a = -12) ∨ (a = 8) :=
sorry

end find_point_on_x_axis_l308_308415


namespace vector_magnitude_proof_l308_308522

noncomputable def vector_magnitude (v : ℝˣ ) : ℝ := 
  Real.sqrt (v.dot v)

theorem vector_magnitude_proof (a b : ℝˣ ) 
  (h_angle : ∀ (θ : ℝ), θ = 3 * Real.pi / 4 )
  (h1 : vector_magnitude a = Real.sqrt 2)
  (h2 : vector_magnitude b = 3)
  :
  vector_magnitude (a + (2*b)) = Real.sqrt 26 := 
sorry

end vector_magnitude_proof_l308_308522


namespace product_abc_l308_308367

theorem product_abc (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h : a * b^3 = 180) : a * b * c = 60 * c := 
sorry

end product_abc_l308_308367


namespace number_of_odd_terms_in_expansion_l308_308686

theorem number_of_odd_terms_in_expansion (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  number_of_odd_terms_in_expansion (p + q) 8 = 2 :=
sorry

end number_of_odd_terms_in_expansion_l308_308686


namespace geometric_sequence_common_ratio_l308_308989

theorem geometric_sequence_common_ratio (a : ℕ → ℝ)
    (h1 : a 1 = -1)
    (h2 : a 2 + a 3 = -2) :
    ∃ q : ℝ, (a 2 = a 1 * q) ∧ (a 3 = a 1 * q^2) ∧ (q = -2 ∨ q = 1) :=
sorry

end geometric_sequence_common_ratio_l308_308989


namespace find_pairs_l308_308352

noncomputable def diamond (a b : ℝ) : ℝ :=
  a^2 * b^2 - a^3 * b - a * b^3

theorem find_pairs (x y : ℝ) :
  diamond x y = diamond y x ↔
  x = 0 ∨ y = 0 ∨ x = y ∨ x = -y :=
by
  sorry

end find_pairs_l308_308352


namespace problem_statement_l308_308967

variable (x : ℝ)

-- Definitions based on the conditions
def a := 2005 * x + 2009
def b := 2005 * x + 2010
def c := 2005 * x + 2011

-- Assertion for the problem
theorem problem_statement : a ^ 2 + b ^ 2 + c ^ 2 - a * b - b * c - c * a = 3 := by
  sorry

end problem_statement_l308_308967


namespace graph_properties_l308_308860

theorem graph_properties (x : ℝ) :
  (∃ p : ℝ × ℝ, p = (1, -7) ∧ y = -7 * x) ∧
  (x ≠ 0 → y * x < 0) ∧
  (x > 0 → y < 0) :=
by
  sorry

end graph_properties_l308_308860


namespace opposite_of_neg_quarter_l308_308440

theorem opposite_of_neg_quarter : -(- (1 / 4)) = 1 / 4 :=
by
  sorry

end opposite_of_neg_quarter_l308_308440


namespace chris_eats_donuts_l308_308544

def daily_donuts := 10
def days := 12
def donuts_eaten_per_day := 1
def boxes_filled := 10
def donuts_per_box := 10

-- Define the total number of donuts made.
def total_donuts := daily_donuts * days

-- Define the total number of donuts Jeff eats.
def jeff_total_eats := donuts_eaten_per_day * days

-- Define the remaining donuts after Jeff eats his share.
def remaining_donuts := total_donuts - jeff_total_eats

-- Define the total number of donuts in the boxes.
def donuts_in_boxes := boxes_filled * donuts_per_box

-- The proof problem:
theorem chris_eats_donuts : remaining_donuts - donuts_in_boxes = 8 :=
by
  -- Placeholder for proof
  sorry

end chris_eats_donuts_l308_308544


namespace permutation_average_sum_l308_308511

theorem permutation_average_sum :
  let p := 286
  let q := 11
  p + q = 297 :=
by
  sorry

end permutation_average_sum_l308_308511


namespace common_factor_l308_308157

theorem common_factor (x y : ℝ) : 
  ∃ c : ℝ, c * (3 * x * y^2 - 4 * x^2 * y) = 6 * x^2 * y - 8 * x * y^2 ∧ c = 2 * x * y := 
by 
  sorry

end common_factor_l308_308157


namespace xiao_li_place_l308_308460

def guess_A (place : String) : Prop :=
  place ≠ "first" ∧ place ≠ "second"

def guess_B (place : String) : Prop :=
  place ≠ "first" ∧ place = "third"

def guess_C (place : String) : Prop :=
  place ≠ "third" ∧ place = "first"

def correct_guesses (guess : String → Prop) (place : String) : Prop :=
  guess place

def half_correct_guesses (guess : String → Prop) (place : String) : Prop :=
  (guess "first" = (place = "first")) ∨
  (guess "second" = (place = "second")) ∨
  (guess "third" = (place = "third"))

theorem xiao_li_place :
  ∃ (place : String),
  (correct_guesses guess_A place ∧
   half_correct_guesses guess_B place ∧
   ¬ correct_guesses guess_C place) ∨
  (correct_guesses guess_B place ∧
   half_correct_guesses guess_A place ∧
   ¬ correct_guesses guess_C place) ∨
  (correct_guesses guess_C place ∧
   half_correct_guesses guess_A place ∧
   ¬ correct_guesses guess_B place) ∨
  (correct_guesses guess_C place ∧
   half_correct_guesses guess_B place ∧
   ¬ correct_guesses guess_A place) :=
sorry

end xiao_li_place_l308_308460


namespace walking_rate_ratio_l308_308767

theorem walking_rate_ratio (R R' : ℚ) (D : ℚ) (h1: D = R * 14) (h2: D = R' * 12) : R' / R = 7 / 6 :=
by 
  sorry

end walking_rate_ratio_l308_308767


namespace product_mod_7_zero_l308_308649

theorem product_mod_7_zero : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 0 := 
by sorry

end product_mod_7_zero_l308_308649


namespace cricket_count_l308_308610

theorem cricket_count (x : ℕ) (h : x + 11 = 18) : x = 7 :=
by sorry

end cricket_count_l308_308610


namespace no_integer_coordinates_between_A_and_B_l308_308625

section
variable (A B : ℤ × ℤ)
variable (Aeq : A = (2, 3))
variable (Beq : B = (50, 305))

theorem no_integer_coordinates_between_A_and_B :
  (∀ P : ℤ × ℤ, P.1 > 2 ∧ P.1 < 50 ∧ P.2 = (151 * P.1 - 230) / 24 → False) :=
by
  sorry
end

end no_integer_coordinates_between_A_and_B_l308_308625


namespace road_path_distance_l308_308961

theorem road_path_distance (d_AB d_AC d_BC d_BD : ℕ) 
  (h1 : d_AB = 9) (h2 : d_AC = 13) (h3 : d_BC = 8) (h4 : d_BD = 14) : A_to_D = 19 :=
by
  sorry

end road_path_distance_l308_308961


namespace jack_birth_year_l308_308295

theorem jack_birth_year 
  (first_amc8_year : ℕ) 
  (amc8_annual : ℕ → ℕ → ℕ) 
  (jack_age_ninth_amc8 : ℕ) 
  (ninth_amc8_year : amc8_annual first_amc8_year 9 = 1998) 
  (jack_age_in_ninth_amc8 : jack_age_ninth_amc8 = 15)
  : (1998 - jack_age_ninth_amc8 = 1983) := by
  sorry

end jack_birth_year_l308_308295


namespace distance_AC_l308_308805

theorem distance_AC (south_dist : ℕ) (west_dist : ℕ) (north_dist : ℕ) (east_dist : ℕ) :
  south_dist = 50 → west_dist = 70 → north_dist = 30 → east_dist = 40 →
  Real.sqrt ((south_dist - north_dist)^2 + (west_dist - east_dist)^2) = 36.06 :=
by
  intros h_south h_west h_north h_east
  rw [h_south, h_west, h_north, h_east]
  simp
  norm_num
  sorry

end distance_AC_l308_308805


namespace john_total_shirts_l308_308708

-- Define initial conditions
def initial_shirts : ℕ := 12
def additional_shirts : ℕ := 4

-- Statement of the problem
theorem john_total_shirts : initial_shirts + additional_shirts = 16 := by
  sorry

end john_total_shirts_l308_308708


namespace reverse_geometric_diff_l308_308940

-- A digit must be between 0 and 9
def digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

-- Distinct digits
def distinct_digits (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Reverse geometric sequence 
def reverse_geometric (a b c : ℕ) : Prop := ∃ r : ℚ, b = c * r ∧ a = b * r

-- Check if abc forms a valid 3-digit reverse geometric sequence
def valid_reverse_geometric_number (a b c : ℕ) : Prop :=
  digit a ∧ digit b ∧ digit c ∧ distinct_digits a b c ∧ reverse_geometric a b c

theorem reverse_geometric_diff (a b c d e f : ℕ) 
  (h1: valid_reverse_geometric_number a b c) 
  (h2: valid_reverse_geometric_number d e f) :
  (a * 100 + b * 10 + c) - (d * 100 + e * 10 + f) = 789 :=
sorry

end reverse_geometric_diff_l308_308940


namespace minimum_sum_of_distances_l308_308965

open Real

theorem minimum_sum_of_distances :
  let l1 := (4 * x - 3 * y + 6 = 0),
      l2 := (x = 0),
      parabola := (y^2 = 4 * x),
      distance_to_l2 (a : ℝ) := a^2,
      distance_to_l1 (a : ℝ) := (|4 * a^2 - 6 * a + 6| / 5),
      total_distance (a : ℝ) := (distance_to_l1 a + distance_to_l2 a)
  in
  ∃ a : ℝ, (parabola → ((total_distance a) = 1)) :=
begin
  sorry
end

end minimum_sum_of_distances_l308_308965


namespace algebraic_expression_value_l308_308384

theorem algebraic_expression_value (x : ℝ) (h : x = 2 * Real.sqrt 3 - 1) : x^2 + 2 * x - 3 = 8 :=
by 
  sorry

end algebraic_expression_value_l308_308384


namespace rosie_pies_l308_308275

theorem rosie_pies (apples_per_pie : ℕ) (apples_total : ℕ) (pies_initial : ℕ) 
  (h1 : 3 = pies_initial) (h2 : 12 = apples_total) : 
  (36 / (apples_total / pies_initial)) * pies_initial = 27 := 
by
  sorry

end rosie_pies_l308_308275


namespace product_mod_7_zero_l308_308650

theorem product_mod_7_zero : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 0 := 
by sorry

end product_mod_7_zero_l308_308650


namespace roots_of_polynomial_equation_l308_308053

theorem roots_of_polynomial_equation (x : ℝ) :
  4 * x ^ 4 - 21 * x ^ 3 + 34 * x ^ 2 - 21 * x + 4 = 0 ↔ x = 4 ∨ x = 1 / 4 ∨ x = 1 :=
by
  sorry

end roots_of_polynomial_equation_l308_308053


namespace laura_five_dollar_bills_l308_308552

theorem laura_five_dollar_bills (x y z : ℕ) 
  (h1 : x + y + z = 40) 
  (h2 : x + 2 * y + 5 * z = 120) 
  (h3 : y = 2 * x) : 
  z = 16 := 
by
  sorry

end laura_five_dollar_bills_l308_308552


namespace product_of_two_even_numbers_is_even_product_of_two_odd_numbers_is_odd_product_of_even_and_odd_number_is_even_product_of_odd_and_even_number_is_even_l308_308599

-- Definition of even and odd numbers
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- Theorem statements for each condition

-- Prove that the product of two even numbers is even
theorem product_of_two_even_numbers_is_even (a b : ℤ) :
  is_even a → is_even b → is_even (a * b) :=
by sorry

-- Prove that the product of two odd numbers is odd
theorem product_of_two_odd_numbers_is_odd (c d : ℤ) :
  is_odd c → is_odd d → is_odd (c * d) :=
by sorry

-- Prove that the product of one even and one odd number is even
theorem product_of_even_and_odd_number_is_even (e f : ℤ) :
  is_even e → is_odd f → is_even (e * f) :=
by sorry

-- Prove that the product of one odd and one even number is even
theorem product_of_odd_and_even_number_is_even (g h : ℤ) :
  is_odd g → is_even h → is_even (g * h) :=
by sorry

end product_of_two_even_numbers_is_even_product_of_two_odd_numbers_is_odd_product_of_even_and_odd_number_is_even_product_of_odd_and_even_number_is_even_l308_308599


namespace min_major_axis_ellipse_l308_308968

theorem min_major_axis_ellipse (a b c : ℝ) (h1 : b * c = 1) (h2 : a^2 = b^2 + c^2) :
  2 * a ≥ 2 * Real.sqrt 2 :=
by {
  sorry
}

end min_major_axis_ellipse_l308_308968


namespace jared_yearly_earnings_l308_308509

theorem jared_yearly_earnings (monthly_pay_diploma : ℕ) (multiplier : ℕ) (months_in_year : ℕ)
  (h1 : monthly_pay_diploma = 4000) (h2 : multiplier = 3) (h3 : months_in_year = 12) :
  (monthly_pay_diploma * multiplier * months_in_year) = 144000 :=
by
  -- The proof goes here
  sorry

end jared_yearly_earnings_l308_308509


namespace square_reciprocal_sum_integer_l308_308764

theorem square_reciprocal_sum_integer (a : ℝ) (h : ∃ k : ℤ, a + 1/a = k) : ∃ m : ℤ, a^2 + 1/a^2 = m := by
  sorry

end square_reciprocal_sum_integer_l308_308764


namespace matrix_vector_equation_l308_308107

variables {α : Type*} [CommRing α] (M : Matrix (Fin 2) (Fin 2) α) 
v w u : Vector (Fin 2) α

theorem matrix_vector_equation 
  (Mv : M.mulVec v = ![2, -3])
  (Mw : M.mulVec w = ![-1, 4])
  (Mu : M.mulVec u = ![3, 0]) :
  M.mulVec (3 • v - 4 • w + u) = 
    ![13, -25] := 
    sorry

end matrix_vector_equation_l308_308107


namespace bakery_item_count_l308_308535

theorem bakery_item_count : ∃ (s c : ℕ), 5 * s + 25 * c = 500 ∧ s + c = 12 := by
  sorry

end bakery_item_count_l308_308535


namespace contrapositive_false_of_implication_false_l308_308232

variable (p q : Prop)

-- The statement we need to prove: If "if p then q" is false, 
-- then "if not q then not p" must be false.
theorem contrapositive_false_of_implication_false (h : ¬ (p → q)) : ¬ (¬ q → ¬ p) :=
by
sorry

end contrapositive_false_of_implication_false_l308_308232


namespace tangent_to_parabola_k_l308_308943

theorem tangent_to_parabola_k (k : ℝ) :
  (∃ (x y : ℝ), 4 * x + 7 * y + k = 0 ∧ y^2 = 32 * x ∧ 
  ∀ (a b : ℝ) (ha : a * y^2 + b * y + k = 0), b^2 - 4 * a * k = 0) → k = 98 :=
by
  sorry

end tangent_to_parabola_k_l308_308943


namespace common_root_is_1_neg1_i_negi_l308_308133

open Complex

theorem common_root_is_1_neg1_i_negi (a b c d k : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (a * k^3 + b * k^2 + c * k + d = 0) → (b * k^3 + c * k^2 + d * k + a = 0) →
  k = 1 ∨ k = -1 ∨ k = Complex.i ∨ k = -Complex.i :=
by
  sorry

end common_root_is_1_neg1_i_negi_l308_308133


namespace measure_15_minutes_l308_308151

/-- Given a timer setup with a 7-minute hourglass and an 11-minute hourglass, show that we can measure exactly 15 minutes. -/
theorem measure_15_minutes (h7 : ∃ t : ℕ, t = 7) (h11 : ∃ t : ℕ, t = 11) : ∃ t : ℕ, t = 15 := 
  by 
    sorry

end measure_15_minutes_l308_308151


namespace rosie_can_make_nine_pies_l308_308279

theorem rosie_can_make_nine_pies (apples pies : ℕ) (h : apples = 12 ∧ pies = 3) : 36 / (12 / 3) * pies = 9 :=
by
  sorry

end rosie_can_make_nine_pies_l308_308279


namespace odd_terms_in_expansion_l308_308684

theorem odd_terms_in_expansion (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  (∃ k, k = 2) :=
sorry

end odd_terms_in_expansion_l308_308684


namespace k_equals_three_fourths_l308_308957

theorem k_equals_three_fourths : ∀ a b c d : ℝ, a ∈ Set.Ici (-1) → b ∈ Set.Ici (-1) → c ∈ Set.Ici (-1) → d ∈ Set.Ici (-1) →
  a^3 + b^3 + c^3 + d^3 + 1 ≥ (3 / 4) * (a + b + c + d) :=
by
  intros
  sorry

end k_equals_three_fourths_l308_308957


namespace three_digit_integers_divisible_by_21_with_7_in_units_place_l308_308677

theorem three_digit_integers_divisible_by_21_with_7_in_units_place :
  ∃ k : ℕ, ∀ n : ℕ, (100 ≤ n ∧ n ≤ 999) →
           (n % 10 = 7) →
           (n % 21 = 0) →
           (n ∈ {m | 100 ≤ m ∧ m ≤ 999 ∧ m % 10 = 7 ∧ m % 21 = 0}) →
           k = (finset.card {m | 100 ≤ m ∧ m ≤ 999 ∧ m % 10 = 7 ∧ m % 21 = 0}) := 
sorry

end three_digit_integers_divisible_by_21_with_7_in_units_place_l308_308677


namespace kelly_points_l308_308257

theorem kelly_points (K : ℕ) 
  (h1 : 12 + 2 * 12 + K + 2 * K + 12 / 2 = 69) : K = 9 := by
  sorry

end kelly_points_l308_308257


namespace largest_possible_dividend_l308_308980

theorem largest_possible_dividend (divisor quotient : ℕ) (remainder : ℕ) 
  (h_divisor : divisor = 18)
  (h_quotient : quotient = 32)
  (h_remainder : remainder < divisor) :
  quotient * divisor + remainder = 593 :=
by
  -- No proof here, add sorry to skip the proof
  sorry

end largest_possible_dividend_l308_308980


namespace sector_area_l308_308233

theorem sector_area (r : ℝ) (α : ℝ) (h_r : r = 2) (h_α : α = π / 4) :
  1/2 * r^2 * α = π / 2 :=
by
  subst h_r
  subst h_α
  sorry

end sector_area_l308_308233


namespace arithmetic_expression_value_l308_308884

theorem arithmetic_expression_value : 4 * (8 - 3) - 7 = 13 := by
  sorry

end arithmetic_expression_value_l308_308884


namespace hazel_lemonade_total_l308_308074

theorem hazel_lemonade_total 
  (total_lemonade: ℕ)
  (sold_construction: ℕ := total_lemonade / 2) 
  (sold_kids: ℕ := 18) 
  (gave_friends: ℕ := sold_kids / 2) 
  (drank_herself: ℕ := 1) :
  total_lemonade = 56 :=
  sorry

end hazel_lemonade_total_l308_308074


namespace Eva_arts_marks_difference_l308_308049

noncomputable def marks_difference_in_arts : ℕ := 
  let M1 := 90
  let A2 := 90
  let S1 := 60
  let M2 := 80
  let A1 := A2 - 75
  let S2 := 90
  A2 - A1

theorem Eva_arts_marks_difference : marks_difference_in_arts = 75 := by
  sorry

end Eva_arts_marks_difference_l308_308049


namespace find_value_of_f2_sub_f3_l308_308668

variable (f : ℝ → ℝ)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem find_value_of_f2_sub_f3 (h_odd : is_odd_function f) (h_sum : f (-2) + f 0 + f 3 = 2) :
  f 2 - f 3 = -2 :=
by
  sorry

end find_value_of_f2_sub_f3_l308_308668


namespace determine_n_l308_308178

-- All the terms used in the conditions
variables (S C M : ℝ)
variables (n : ℝ)

-- Define the conditions as hypotheses
def condition1 := M = 1 / 3 * S
def condition2 := M = 1 / n * C

-- The main theorem statement
theorem determine_n (S C M : ℝ) (n : ℝ) (h1 : condition1 S M) (h2 : condition2 M n C) : n = 2 :=
by sorry

end determine_n_l308_308178


namespace largest_possible_x_l308_308026

theorem largest_possible_x :
  ∃ x : ℝ, (3*x^2 + 18*x - 84 = x*(x + 10)) ∧ ∀ y : ℝ, (3*y^2 + 18*y - 84 = y*(y + 10)) → y ≤ x :=
by
  sorry

end largest_possible_x_l308_308026


namespace probability_accurate_forecast_l308_308307

theorem probability_accurate_forecast (p q : ℝ) (h1 : 0 ≤ p ∧ p ≤ 1) (h2 : 0 ≤ q ∧ q ≤ 1) : 
  p * (1 - q) = p * (1 - q) :=
by {
  sorry
}

end probability_accurate_forecast_l308_308307


namespace coefficient_of_xy6_eq_one_l308_308691

theorem coefficient_of_xy6_eq_one (a : ℚ) (h : (7 : ℚ) * a = 1) : a = 1 / 7 :=
by sorry

end coefficient_of_xy6_eq_one_l308_308691


namespace cost_of_kid_ticket_l308_308168

theorem cost_of_kid_ticket (total_people kids adults : ℕ) 
  (adult_ticket_cost kid_ticket_cost : ℕ) 
  (total_sales : ℕ) 
  (h_people : total_people = kids + adults)
  (h_adult_cost : adult_ticket_cost = 28)
  (h_kids : kids = 203)
  (h_total_sales : total_sales = 3864)
  (h_calculate_sales : adults * adult_ticket_cost + kids * kid_ticket_cost = total_sales)
  : kid_ticket_cost = 12 :=
by
  sorry -- Proof will be filled in

end cost_of_kid_ticket_l308_308168


namespace norris_money_left_l308_308721

theorem norris_money_left :
  let september_savings := 29
  let october_savings := 25
  let november_savings := 31
  let total_savings := september_savings + october_savings + november_savings
  let amount_spent := 75
  total_savings - amount_spent = 10 :=
by
  let september_savings := 29
  let october_savings := 25
  let november_savings := 31
  let total_savings := september_savings + october_savings + november_savings
  let amount_spent := 75
  have h1 : total_savings = 85 := by rfl
  have h2 : total_savings - amount_spent = 85 - amount_spent := by rw h1
  have h3 : 85 - amount_spent = 85 - 75 := rfl
  have h4 : 85 - 75 = 10 := rfl
  exact eq.trans (eq.trans h2 h3) h4

end norris_money_left_l308_308721


namespace labor_productivity_increase_l308_308865

noncomputable def regression_equation (x : ℝ) : ℝ := 50 + 60 * x

theorem labor_productivity_increase (Δx : ℝ) (hx : Δx = 1) :
  regression_equation (x + Δx) - regression_equation x = 60 :=
by
  sorry

end labor_productivity_increase_l308_308865


namespace mod_equiv_l308_308263

theorem mod_equiv (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (25 * m + 3 * n) % 83 = 0 ↔ (3 * m + 7 * n) % 83 = 0 :=
by
  sorry

end mod_equiv_l308_308263


namespace principal_sum_correct_l308_308139

noncomputable def principal_sum (CI SI : ℝ) (t : ℕ) : ℝ :=
  let P := ((SI * t) / t) in
  let x := 5100 / P in
  26010000 / ((CI - SI) / t)

theorem principal_sum_correct :
  principal_sum 11730 10200 2 ≈ 16993.46 :=
by
  simp only [principal_sum]
  sorry

end principal_sum_correct_l308_308139


namespace distance_between_home_and_retreat_l308_308802

theorem distance_between_home_and_retreat (D : ℝ) 
  (h1 : D / 50 + D / 75 = 10) : D = 300 :=
sorry

end distance_between_home_and_retreat_l308_308802


namespace mushrooms_used_by_Karla_correct_l308_308060

-- Given conditions
def mushrooms_cut_each_mushroom : ℕ := 4
def mushrooms_cut_total : ℕ := 22 * mushrooms_cut_each_mushroom
def mushrooms_used_by_Kenny : ℕ := 38
def mushrooms_remaining : ℕ := 8
def mushrooms_total_used_by_Kenny_and_remaining : ℕ := mushrooms_used_by_Kenny + mushrooms_remaining
def mushrooms_used_by_Karla : ℕ := mushrooms_cut_total - mushrooms_total_used_by_Kenny_and_remaining

-- Statement to prove
theorem mushrooms_used_by_Karla_correct :
  mushrooms_used_by_Karla = 42 :=
by
  sorry

end mushrooms_used_by_Karla_correct_l308_308060


namespace arithmetic_mean_15_23_37_45_l308_308789

def arithmetic_mean (a b c d : ℕ) : ℕ := (a + b + c + d) / 4

theorem arithmetic_mean_15_23_37_45 :
  arithmetic_mean 15 23 37 45 = 30 :=
by {
  sorry
}

end arithmetic_mean_15_23_37_45_l308_308789


namespace square_area_from_circle_l308_308166

-- Define the conditions for the circle's equation
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 = -2 * y^2 + 8 * x - 8 * y + 28 

-- State the main theorem to prove the area of the square
theorem square_area_from_circle (x y : ℝ) (h : circle_equation x y) :
  ∃ s : ℝ, s^2 = 88 :=
sorry

end square_area_from_circle_l308_308166


namespace tenth_term_is_26_l308_308760

-- Definitions used from the conditions
def first_term : ℤ := 8
def common_difference : ℤ := 2
def term_number : ℕ := 10

-- Define the formula for the nth term of an arithmetic progression
def nth_term (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- Proving that the 10th term is 26 given the conditions
theorem tenth_term_is_26 : nth_term first_term common_difference term_number = 26 := by
  sorry

end tenth_term_is_26_l308_308760


namespace no_real_sqrt_neg_six_pow_three_l308_308605

theorem no_real_sqrt_neg_six_pow_three : 
  ∀ x : ℝ, 
    (¬ ∃ y : ℝ, y * y = -6 ^ 3) :=
by
  sorry

end no_real_sqrt_neg_six_pow_three_l308_308605


namespace eel_count_l308_308644

theorem eel_count 
  (x y z : ℕ)
  (h1 : y + z = 12)
  (h2 : x + z = 14)
  (h3 : x + y = 16) : 
  x + y + z = 21 := 
by 
  sorry

end eel_count_l308_308644


namespace remainder_prod_mod_7_l308_308647

theorem remainder_prod_mod_7 
  (seq : ℕ → ℕ)
  (h_seq : ∀ k, k < 10 → seq k = 10 * k + 3) :
  (∏ k in finset.range 10, seq k) % 7 = 2 :=
by
  have h_seq_form : ∀ k, k < 10 → (seq k % 7 = 3) :=
    by intros k hk; rw [h_seq k hk, add_comm]; exact mod_eq_of_lt (nat.mod_lt _ zero_lt_succ)
  all_goals { sorry }

end remainder_prod_mod_7_l308_308647


namespace g_value_at_2_over_9_l308_308941

theorem g_value_at_2_over_9 (g : ℝ → ℝ) 
  (hg0 : g 0 = 0)
  (hgmono : ∀ ⦃x y⦄, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y)
  (hg_symm : ∀ x, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x)
  (hg_frac : ∀ x, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3) :
  g (2 / 9) = 8 / 27 :=
sorry

end g_value_at_2_over_9_l308_308941


namespace bianca_bags_not_recycled_l308_308926

theorem bianca_bags_not_recycled :
  ∀ (points_per_bag total_bags total_points bags_recycled bags_not_recycled : ℕ),
    points_per_bag = 5 →
    total_bags = 17 →
    total_points = 45 →
    bags_recycled = total_points / points_per_bag →
    bags_not_recycled = total_bags - bags_recycled →
    bags_not_recycled = 8 :=
by
  intros points_per_bag total_bags total_points bags_recycled bags_not_recycled
  intros h_points_per_bag h_total_bags h_total_points h_bags_recycled h_bags_not_recycled
  sorry

end bianca_bags_not_recycled_l308_308926


namespace event_B_is_certain_l308_308320

-- Define the event that the sum of two sides of a triangle is greater than the third side
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the term 'certain event'
def certain_event (E : Prop) : Prop := E

/-- Prove that the event "the sum of two sides of a triangle is greater than the third side" is a certain event -/
theorem event_B_is_certain (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  certain_event (triangle_inequality a b c) :=
sorry

end event_B_is_certain_l308_308320


namespace odd_terms_in_expansion_l308_308687

theorem odd_terms_in_expansion (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  (finset.range 9).filter (λ k, ((nat.choose 8 k) % 2 = 1) ∧ ((p ^ (8 - k) * q ^ k) % 2 = 1)).card = 2 :=
sorry

end odd_terms_in_expansion_l308_308687


namespace parabola_equation_l308_308826

theorem parabola_equation (h k a : ℝ) (same_shape : ∀ x, -2 * x^2 + 2 = a * x^2 + k) (vertex : h = 4 ∧ k = -2) :
  ∀ x, -2 * (x - 4)^2 - 2 = a * (x - h)^2 + k :=
by
  -- This is where the actual proof would go
  simp
  sorry

end parabola_equation_l308_308826


namespace discount_percentage_l308_308172

noncomputable def cost_price : ℝ := 100
noncomputable def profit_with_discount : ℝ := 0.32 * cost_price
noncomputable def profit_without_discount : ℝ := 0.375 * cost_price

noncomputable def sp_with_discount : ℝ := cost_price + profit_with_discount
noncomputable def sp_without_discount : ℝ := cost_price + profit_without_discount

noncomputable def discount_amount : ℝ := sp_without_discount - sp_with_discount
noncomputable def percentage_discount : ℝ := (discount_amount / sp_without_discount) * 100

theorem discount_percentage : percentage_discount = 4 :=
by
  -- proof steps
  sorry

end discount_percentage_l308_308172


namespace jared_annual_earnings_l308_308504

-- Defining conditions as constants
def diploma_pay : ℕ := 4000
def degree_multiplier : ℕ := 3
def months_in_year : ℕ := 12

-- Goal: Prove that Jared's annual earnings are $144,000
theorem jared_annual_earnings : diploma_pay * degree_multiplier * months_in_year = 144000 := by
  sorry

end jared_annual_earnings_l308_308504


namespace work_together_zero_days_l308_308159

theorem work_together_zero_days (a b : ℝ) (ha : a = 1/18) (hb : b = 1/9) (x : ℝ) (hx : 1 - x * a = 2/3) : x = 6 →
  (a - a) * (b - b) = 0 := by
  sorry

end work_together_zero_days_l308_308159


namespace negative_large_base_zero_exponent_l308_308029

-- Define the problem conditions: base number and exponent
def base_number : ℤ := -2023
def exponent : ℕ := 0

-- Prove that (-2023)^0 equals 1
theorem negative_large_base_zero_exponent : base_number ^ exponent = 1 := by
  sorry

end negative_large_base_zero_exponent_l308_308029


namespace maximilian_annual_revenue_l308_308122

-- Define the number of units in the building
def total_units : ℕ := 100

-- Define the occupancy rate
def occupancy_rate : ℚ := 3 / 4

-- Define the monthly rent per unit
def monthly_rent : ℚ := 400

-- Calculate the number of occupied units
def occupied_units : ℕ := (occupancy_rate * total_units : ℚ).natAbs

-- Calculate the monthly rent revenue
def monthly_revenue : ℚ := occupied_units * monthly_rent

-- Calculate the annual rent revenue
def annual_revenue : ℚ := monthly_revenue * 12

-- Prove that the annual revenue is $360,000
theorem maximilian_annual_revenue : annual_revenue = 360000 := by
  sorry

end maximilian_annual_revenue_l308_308122


namespace hours_l308_308337

def mechanic_hours_charged (h : ℕ) : Prop :=
  45 * h + 225 = 450

theorem hours (h : ℕ) : mechanic_hours_charged h → h = 5 :=
by
  intro h_eq
  have : 45 * h + 225 = 450 := h_eq
  sorry

end hours_l308_308337


namespace sqrt_15_between_3_and_4_l308_308499

theorem sqrt_15_between_3_and_4 :
  3 < Real.sqrt 15 ∧ Real.sqrt 15 < 4 :=
by
  have h1 : 3^2 = 9 := by norm_num
  have h2 : 4^2 = 16 := by norm_num
  have h3 : 9 < 15 ∧ 15 < 16 := by split; norm_num
  sorry

end sqrt_15_between_3_and_4_l308_308499


namespace simplify_expression_l308_308947

theorem simplify_expression :
  (2 + 1 / 2) / (1 - 3 / 4) = 10 :=
by
  sorry

end simplify_expression_l308_308947


namespace expand_polynomial_l308_308356

theorem expand_polynomial :
  (5 * x^2 + 3 * x - 4) * 3 * x^3 = 15 * x^5 + 9 * x^4 - 12 * x^3 := 
by
  sorry

end expand_polynomial_l308_308356


namespace jars_needed_l308_308099

-- Definitions based on the given conditions
def total_cherry_tomatoes : ℕ := 56
def cherry_tomatoes_per_jar : ℕ := 8

-- Lean theorem to prove the question
theorem jars_needed (total_cherry_tomatoes cherry_tomatoes_per_jar : ℕ) (h1 : total_cherry_tomatoes = 56) (h2 : cherry_tomatoes_per_jar = 8) : (total_cherry_tomatoes / cherry_tomatoes_per_jar) = 7 := by
  -- Proof omitted
  sorry

end jars_needed_l308_308099


namespace smallest_angle_l308_308392

theorem smallest_angle (largest_angle : ℝ) (a b : ℝ) (h1 : largest_angle = 120) (h2 : 3 * a = 2 * b) (h3 : largest_angle + a + b = 180) : b = 24 := by
  sorry

end smallest_angle_l308_308392


namespace floor_e_eq_2_l308_308211

theorem floor_e_eq_2 : ⌊Real.exp 1⌋ = 2 := by
  sorry

end floor_e_eq_2_l308_308211


namespace directly_proportional_l308_308177

-- Defining conditions
def A (x y : ℝ) : Prop := y = x + 8
def B (x y : ℝ) : Prop := (2 / (5 * y)) = x
def C (x y : ℝ) : Prop := (2 / 3) * x = y

-- Theorem stating that in the given equations, equation C shows direct proportionality
theorem directly_proportional (x y : ℝ) : C x y ↔ (∃ k : ℝ, k ≠ 0 ∧ y = k * x) :=
by
  sorry

end directly_proportional_l308_308177


namespace total_fireworks_l308_308551

-- Definitions of the given conditions
def koby_boxes : Nat := 2
def koby_box_sparklers : Nat := 3
def koby_box_whistlers : Nat := 5
def cherie_boxes : Nat := 1
def cherie_box_sparklers : Nat := 8
def cherie_box_whistlers : Nat := 9

-- Statement to prove the total number of fireworks
theorem total_fireworks : 
  let koby_fireworks := koby_boxes * (koby_box_sparklers + koby_box_whistlers)
  let cherie_fireworks := cherie_boxes * (cherie_box_sparklers + cherie_box_whistlers)
  koby_fireworks + cherie_fireworks = 33 := by
  sorry

end total_fireworks_l308_308551


namespace determinant_roots_cubic_eq_l308_308408

noncomputable def determinant_of_matrix (a b c : ℝ) : ℝ :=
  a * (b * c - 1) - (c - 1) + (1 - b)

theorem determinant_roots_cubic_eq {a b c p q r : ℝ}
  (h1 : a + b + c = p)
  (h2 : a * b + b * c + c * a = q)
  (h3 : a * b * c = r) :
  determinant_of_matrix a b c = r - p + 2 :=
by {
  sorry
}

end determinant_roots_cubic_eq_l308_308408


namespace four_distinct_sum_equal_l308_308855

theorem four_distinct_sum_equal (S : Finset ℕ) (hS : S.card = 10) (hS_subset : S ⊆ Finset.range 38) :
  ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + b = c + d :=
by
  sorry

end four_distinct_sum_equal_l308_308855


namespace division_expression_is_7_l308_308127

noncomputable def evaluate_expression : ℝ :=
  1 / 2 / 3 / 4 / 5 / (6 / 7 / 8 / 9 / 10)

theorem division_expression_is_7 : evaluate_expression = 7 :=
by
  sorry

end division_expression_is_7_l308_308127


namespace transport_cost_l308_308582

theorem transport_cost (mass_g: ℕ) (cost_per_kg : ℕ) (mass_kg : ℝ) 
  (h1 : mass_g = 300) (h2 : mass_kg = (mass_g : ℝ) / 1000) 
  (h3: cost_per_kg = 18000)
  : mass_kg * cost_per_kg = 5400 := by
  sorry

end transport_cost_l308_308582


namespace inequality_proof_l308_308418

variable (x y : ℝ)
variable (hx : 0 < x) (hy : 0 < y)

theorem inequality_proof :
  x / Real.sqrt y + y / Real.sqrt x ≥ Real.sqrt x + Real.sqrt y :=
sorry

end inequality_proof_l308_308418


namespace remainder_prod_mod_7_l308_308654

theorem remainder_prod_mod_7 : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 4 :=
by
  sorry

end remainder_prod_mod_7_l308_308654


namespace difference_is_correct_l308_308567

-- Define the given constants and conditions
def purchase_price : ℕ := 1500
def down_payment : ℕ := 200
def monthly_payment : ℕ := 65
def number_of_monthly_payments : ℕ := 24

-- Define the derived quantities based on the given conditions
def total_monthly_payments : ℕ := monthly_payment * number_of_monthly_payments
def total_amount_paid : ℕ := down_payment + total_monthly_payments
def difference : ℕ := total_amount_paid - purchase_price

-- The statement to be proven
theorem difference_is_correct : difference = 260 := by
  sorry

end difference_is_correct_l308_308567


namespace zeros_of_f_l308_308593

noncomputable def f (x : ℝ) : ℝ := x^3 - 16 * x

theorem zeros_of_f :
  ∃ a b c : ℝ, (a = -4) ∧ (b = 0) ∧ (c = 4) ∧ (f a = 0) ∧ (f b = 0) ∧ (f c = 0) :=
by
  sorry

end zeros_of_f_l308_308593


namespace odd_terms_in_expansion_l308_308683

theorem odd_terms_in_expansion (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  (∃ k, k = 2) :=
sorry

end odd_terms_in_expansion_l308_308683


namespace tsunami_added_sand_l308_308125

noncomputable def dig_rate : ℝ := 8 / 4 -- feet per hour
noncomputable def sand_after_storm : ℝ := 8 / 2 -- feet
noncomputable def time_to_dig_up_treasure : ℝ := 3 -- hours
noncomputable def total_sand_dug_up : ℝ := dig_rate * time_to_dig_up_treasure -- feet

theorem tsunami_added_sand :
  total_sand_dug_up - sand_after_storm = 2 :=
by
  sorry

end tsunami_added_sand_l308_308125


namespace number_of_outfits_l308_308854

def red_shirts : ℕ := 6
def green_shirts : ℕ := 7
def number_pants : ℕ := 9
def blue_hats : ℕ := 10
def red_hats : ℕ := 10

theorem number_of_outfits :
  (red_shirts * number_pants * blue_hats) + (green_shirts * number_pants * red_hats) = 1170 :=
by
  sorry

end number_of_outfits_l308_308854


namespace median_and_mode_l308_308661

open Set

variable (data_set : List ℝ)
variable (mean : ℝ)

noncomputable def median (l : List ℝ) : ℝ := sorry -- Define medial function
noncomputable def mode (l : List ℝ) : ℝ := sorry -- Define mode function

theorem median_and_mode (x : ℝ) (mean_set : (3 + x + 4 + 5 + 8) / 5 = 5) :
  data_set = [3, 4, 5, 5, 8] ∧ median data_set = 5 ∧ mode data_set = 5 :=
by
  have hx : x = 5 := sorry
  have hdata_set : data_set = [3, 4, 5, 5, 8] := sorry
  have hmedian : median data_set = 5 := sorry
  have hmode : mode data_set = 5 := sorry
  exact ⟨hdata_set, hmedian, hmode⟩

end median_and_mode_l308_308661


namespace no_afg_fourth_place_l308_308000

theorem no_afg_fourth_place
  (A B C D E F G : ℕ)
  (h1 : A < B)
  (h2 : A < C)
  (h3 : B < D)
  (h4 : C < E)
  (h5 : A < F ∧ F < B)
  (h6 : B < G ∧ G < C) :
  ¬ (A = 4 ∨ F = 4 ∨ G = 4) :=
by
  sorry

end no_afg_fourth_place_l308_308000


namespace value_of_a_l308_308667

theorem value_of_a (x a : ℤ) (h : x = 4) (h_eq : 5 * (x - 1) - 3 * a = -3) : a = 6 :=
by {
  sorry
}

end value_of_a_l308_308667


namespace option_A_option_C_option_D_l308_308260

variable {A B C a b c : ℝ}

theorem option_A 
  (h1: 3 * b * Real.cos C + 3 * c * Real.cos B = a^2)
  (h2: A + B + C = Real.pi) : 
  a = 3 := sorry

theorem option_C 
  (h1: B = Real.pi - A - C) 
  (h2: C = 2 * A) 
  (h3: 0 < A) (h4: A < Real.pi / 2) 
  (h5: 0 < B) (h6: B < Real.pi / 2)
  (h7: 0 < C) (h8: C < Real.pi / 2) :
  3 * Real.sqrt 2 < c ∧ c < 3 * Real.sqrt 3 :=
  sorry

theorem option_D 
  (h1: A = 2 * C) 
  (h2: Real.sin B = 2 * Real.sin C) 
  (h3: B = Real.pi - A - C) 
  (O : Type) 
  [is_incenter_triangle_O ABC] : 
  area (triangle AOB) = (3 * Real.sqrt 3 - 3) / 4 :=
  sorry

end option_A_option_C_option_D_l308_308260


namespace sum_not_divisible_by_three_times_any_number_l308_308728

theorem sum_not_divisible_by_three_times_any_number (n : ℕ) (a : Fin n → ℕ) (h : n ≥ 3) (distinct : ∀ i j : Fin n, i ≠ j → a i ≠ a j) :
  ∃ (i j : Fin n), i ≠ j ∧ (∀ k : Fin n, ¬ (a i + a j) ∣ (3 * a k)) :=
sorry

end sum_not_divisible_by_three_times_any_number_l308_308728


namespace largest_n_arithmetic_sequences_l308_308925

theorem largest_n_arithmetic_sequences
  (a : ℕ → ℤ) (b : ℕ → ℤ) (x y : ℤ)
  (a_1 : a 1 = 2) (b_1 : b 1 = 3)
  (a_formula : ∀ n : ℕ, a n = 2 + (n - 1) * x)
  (b_formula : ∀ n : ℕ, b n = 3 + (n - 1) * y)
  (x_lt_y : x < y)
  (product_condition : ∃ n : ℕ, a n * b n = 1638) :
  ∃ n : ℕ, a n * b n = 1638 ∧ n = 35 := 
sorry

end largest_n_arithmetic_sequences_l308_308925


namespace min_value_expression_l308_308517

theorem min_value_expression (x : ℝ) (h : x > 1) : 
  ∃ min_val, min_val = 6 ∧ ∀ y > 1, 2 * y + 2 / (y - 1) ≥ min_val :=
by  
  use 6
  sorry

end min_value_expression_l308_308517


namespace price_of_tray_l308_308145

noncomputable def price_per_egg : ℕ := 50
noncomputable def tray_eggs : ℕ := 30
noncomputable def discount_per_egg : ℕ := 10

theorem price_of_tray : (price_per_egg - discount_per_egg) * tray_eggs / 100 = 12 :=
by
  sorry

end price_of_tray_l308_308145


namespace changfei_class_l308_308579

theorem changfei_class (m n : ℕ) (h : m * (m - 1) + m * n + n = 51) : m + n = 9 :=
sorry

end changfei_class_l308_308579


namespace total_fireworks_l308_308549

-- Definitions based on conditions
def kobys_boxes := 2
def kobys_sparklers_per_box := 3
def kobys_whistlers_per_box := 5
def cheries_boxes := 1
def cheries_sparklers_per_box := 8
def cheries_whistlers_per_box := 9

-- Calculations
def total_kobys_fireworks := kobys_boxes * (kobys_sparklers_per_box + kobys_whistlers_per_box)
def total_cheries_fireworks := cheries_boxes * (cheries_sparklers_per_box + cheries_whistlers_per_box)

-- Theorem
theorem total_fireworks : total_kobys_fireworks + total_cheries_fireworks = 33 := 
by
  -- Can be elaborated and filled in with steps, if necessary.
  sorry

end total_fireworks_l308_308549


namespace minimum_f_value_l308_308228

noncomputable def f (x y : ℝ) : ℝ :=
  y / x + 16 * x / (2 * x + y)

theorem minimum_f_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ t, (∀ x y, f x y ≥ t) ∧ t = 6 := sorry

end minimum_f_value_l308_308228


namespace factorize_expression_l308_308951

theorem factorize_expression (a b : ℝ) : a^2 + a * b = a * (a + b) := 
by
  sorry

end factorize_expression_l308_308951


namespace odd_function_m_zero_l308_308250

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^3 + m

theorem odd_function_m_zero (m : ℝ) : (∀ x : ℝ, f (-x) m = -f x m) → m = 0 :=
by
  sorry

end odd_function_m_zero_l308_308250


namespace proof_problem_l308_308365

-- Definitions of the propositions
def p : Prop := ∀ (x y : ℝ), 6 * x + 2 * y - 1 = 0 → y = 5 - 3 * x
def q : Prop := ∀ (x y : ℝ), 6 * x + 2 * y - 1 = 0 → 2 * x + 6 * y - 4 = 0

-- Translate the mathematical proof problem into a Lean theorem
theorem proof_problem : 
  (p ∧ ¬q) ∧ ¬((¬p) ∧ q) :=
by
  -- You can fill in the exact proof steps here
  sorry

end proof_problem_l308_308365


namespace juan_original_number_l308_308546

theorem juan_original_number (n : ℤ) 
  (h : ((2 * (n + 3) - 2) / 2) = 8) : 
  n = 6 := 
sorry

end juan_original_number_l308_308546


namespace sqrt_of_neg_five_squared_l308_308927

theorem sqrt_of_neg_five_squared : Real.sqrt ((-5 : Real) ^ 2) = 5 := 
by 
  sorry

end sqrt_of_neg_five_squared_l308_308927


namespace largest_five_digit_integer_with_conditions_l308_308319

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def digits_product (n : ℕ) : ℕ :=
  (n % 10) * ((n / 10) % 10) * ((n / 100) % 10) * ((n / 1000) % 10) * ((n / 10000) % 10)

def digits_sum (n : ℕ) : ℕ :=
  (n % 10) + ((n / 10) % 10) + ((n / 100) % 10) + ((n / 1000) % 10) + ((n / 10000) % 10)

theorem largest_five_digit_integer_with_conditions :
  ∃ n : ℕ, is_five_digit n ∧ digits_product n = 40320 ∧ digits_sum n < 35 ∧
  ∀ m : ℕ, is_five_digit m ∧ digits_product m = 40320 ∧ digits_sum m < 35 → n ≥ m :=
sorry

end largest_five_digit_integer_with_conditions_l308_308319


namespace lines_from_equation_l308_308202

-- Definitions for the conditions
def satisfies_equation (x y : ℝ) : Prop :=
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Equivalent Lean statement to the proof problem
theorem lines_from_equation :
  (∀ x y : ℝ, satisfies_equation x y → (y = -x - 2) ∨ (y = -2 * x + 1)) :=
by
  intros x y h
  sorry

end lines_from_equation_l308_308202


namespace lines_intersect_lines_perpendicular_lines_parallel_l308_308240

variables (l1 l2 : ℝ) (m : ℝ)

def intersect (m : ℝ) : Prop :=
  m ≠ -1 ∧ m ≠ 3

def perpendicular (m : ℝ) : Prop :=
  m = 1/2

def parallel (m : ℝ) : Prop :=
  m = -1

theorem lines_intersect (m : ℝ) : intersect m :=
by sorry

theorem lines_perpendicular (m : ℝ) : perpendicular m :=
by sorry

theorem lines_parallel (m : ℝ) : parallel m :=
by sorry

end lines_intersect_lines_perpendicular_lines_parallel_l308_308240


namespace new_mean_l308_308993

-- Define the conditions
def mean_of_numbers (n : ℕ) (mean : ℝ) : ℝ := n * mean
def added_to_each (n : ℕ) (addend : ℝ) : ℝ := n * addend

-- The proof problem
theorem new_mean (n : ℕ) (mean addend : ℝ) (h1 : mean_of_numbers n mean = 600) (h2 : added_to_each n addend = 150) (h3 : n = 15) (h4 : mean = 40) (h5 : addend = 10) :
  (mean_of_numbers n mean + added_to_each n addend) / n = 50 :=
by
  sorry

end new_mean_l308_308993


namespace largest_possible_A_l308_308602

-- Define natural numbers
variables (A B C : ℕ)

-- Given conditions
def division_algorithm (A B C : ℕ) : Prop := A = 8 * B + C
def B_equals_C (B C : ℕ) : Prop := B = C

-- The proof statement
theorem largest_possible_A (h1 : division_algorithm A B C) (h2 : B_equals_C B C) : A = 63 :=
by
  -- Proof is omitted
  sorry

end largest_possible_A_l308_308602


namespace bojan_wins_strategy_l308_308341

theorem bojan_wins_strategy (a b : ℕ) (h1: ∀ i, 1 ≤ i ≤ 2016 → ∃ k, a + i = k) (h2: ∀ j, 1 ≤ j ≤ 2016 → ∃ m, b + j = m) :
  let pieces_ana := {a + i | i in (1:ℕ)..2016}
  let pieces_bojan := {b + j | j in (1:ℕ)..2016}
  ∀ x ∈ pieces_ana, ∃ y ∈ pieces_bojan, (x + y) % 2017 = (a + b) % 2017 :=
by
  sorry

end bojan_wins_strategy_l308_308341


namespace remainder_prod_mod_7_l308_308648

theorem remainder_prod_mod_7 
  (seq : ℕ → ℕ)
  (h_seq : ∀ k, k < 10 → seq k = 10 * k + 3) :
  (∏ k in finset.range 10, seq k) % 7 = 2 :=
by
  have h_seq_form : ∀ k, k < 10 → (seq k % 7 = 3) :=
    by intros k hk; rw [h_seq k hk, add_comm]; exact mod_eq_of_lt (nat.mod_lt _ zero_lt_succ)
  all_goals { sorry }

end remainder_prod_mod_7_l308_308648


namespace intersection_of_S_and_T_l308_308375

-- Define S and T based on given conditions
def S : Set ℝ := { x | x^2 + 2 * x = 0 }
def T : Set ℝ := { x | x^2 - 2 * x = 0 }

-- Prove the intersection of S and T
theorem intersection_of_S_and_T : S ∩ T = {0} :=
sorry

end intersection_of_S_and_T_l308_308375


namespace norris_money_left_l308_308722

theorem norris_money_left :
  let september_savings := 29
  let october_savings := 25
  let november_savings := 31
  let total_savings := september_savings + october_savings + november_savings
  let amount_spent := 75
  total_savings - amount_spent = 10 :=
by
  let september_savings := 29
  let october_savings := 25
  let november_savings := 31
  let total_savings := september_savings + october_savings + november_savings
  let amount_spent := 75
  have h1 : total_savings = 85 := by rfl
  have h2 : total_savings - amount_spent = 85 - amount_spent := by rw h1
  have h3 : 85 - amount_spent = 85 - 75 := rfl
  have h4 : 85 - 75 = 10 := rfl
  exact eq.trans (eq.trans h2 h3) h4

end norris_money_left_l308_308722


namespace lateral_surface_area_of_cube_l308_308326

-- Define the side length of the cube
def side_length : ℕ := 12

-- Define the area of one face of the cube
def area_of_one_face (s : ℕ) : ℕ := s * s

-- Define the lateral surface area of the cube
def lateral_surface_area (s : ℕ) : ℕ := 4 * (area_of_one_face s)

-- Prove the lateral surface area of a cube with side length 12 m is equal to 576 m²
theorem lateral_surface_area_of_cube : lateral_surface_area side_length = 576 := by
  sorry

end lateral_surface_area_of_cube_l308_308326


namespace probability_calculation_correct_l308_308164

def total_balls : ℕ := 100
def white_balls : ℕ := 50
def green_balls : ℕ := 20
def yellow_balls : ℕ := 10
def red_balls : ℕ := 17
def purple_balls : ℕ := 3

def number_of_non_red_or_purple_balls : ℕ := total_balls - (red_balls + purple_balls)

def probability_of_non_red_or_purple : ℚ := number_of_non_red_or_purple_balls / total_balls

theorem probability_calculation_correct :
  probability_of_non_red_or_purple = 0.8 := 
  by 
    -- proof goes here
    sorry

end probability_calculation_correct_l308_308164


namespace lcm_220_504_l308_308155

/-- The least common multiple of 220 and 504 is 27720. -/
theorem lcm_220_504 : Nat.lcm 220 504 = 27720 :=
by
  -- This is the final statement of the theorem. The proof is not provided and marked with 'sorry'.
  sorry

end lcm_220_504_l308_308155


namespace total_movies_shown_l308_308475

theorem total_movies_shown (screen1_movies : ℕ) (screen2_movies : ℕ) (screen3_movies : ℕ)
                          (screen4_movies : ℕ) (screen5_movies : ℕ) (screen6_movies : ℕ)
                          (h1 : screen1_movies = 3) (h2 : screen2_movies = 4) 
                          (h3 : screen3_movies = 2) (h4 : screen4_movies = 3) 
                          (h5 : screen5_movies = 5) (h6 : screen6_movies = 2) :
  screen1_movies + screen2_movies + screen3_movies + screen4_movies + screen5_movies + screen6_movies = 19 := 
by
  sorry

end total_movies_shown_l308_308475


namespace norris_money_left_l308_308724

-- Define the amounts saved each month
def september_savings : ℕ := 29
def october_savings : ℕ := 25
def november_savings : ℕ := 31

-- Define the total savings
def total_savings : ℕ := september_savings + october_savings + november_savings

-- Define the amount spent on the online game
def amount_spent : ℕ := 75

-- Define the remaining money
def money_left : ℕ := total_savings - amount_spent

-- The theorem stating the problem and the solution
theorem norris_money_left : money_left = 10 := by
  sorry

end norris_money_left_l308_308724


namespace max_area_of_triangle_l308_308095

theorem max_area_of_triangle (a c : ℝ)
    (h1 : a^2 + c^2 = 16 + a * c) : 
    ∃ s : ℝ, s = 4 * Real.sqrt 3 := by
  sorry

end max_area_of_triangle_l308_308095


namespace deal_saves_customer_two_dollars_l308_308338

-- Define the conditions of the problem
def movie_ticket_price : ℕ := 8
def popcorn_price : ℕ := movie_ticket_price - 3
def drink_price : ℕ := popcorn_price + 1
def candy_price : ℕ := drink_price / 2

def normal_total_price : ℕ := movie_ticket_price + popcorn_price + drink_price + candy_price
def deal_price : ℕ := 20

-- Prove the savings
theorem deal_saves_customer_two_dollars : normal_total_price - deal_price = 2 :=
by
  -- We will fill in the proof here
  sorry

end deal_saves_customer_two_dollars_l308_308338


namespace find_point_M_l308_308148

/-- Define the function f(x) = x^3 + x - 2. -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- Define the derivative of the function, f'(x). -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

/-- Define the condition that the slope of the tangent line is perpendicular to y = -1/4x - 1. -/
def slope_perpendicular_condition (m : ℝ) : Prop := m = 4

/-- Main theorem: The coordinates of the point M are (1, 0) and (-1, -4). -/
theorem find_point_M : 
  ∃ (x₀ y₀ : ℝ), f x₀ = y₀ ∧ slope_perpendicular_condition (f' x₀) ∧ 
  ((x₀ = 1 ∧ y₀ = 0) ∨ (x₀ = -1 ∧ y₀ = -4)) := 
sorry

end find_point_M_l308_308148


namespace highest_throw_among_them_l308_308347

variable (Christine_throw1 Christine_throw2 Christine_throw3 Janice_throw1 Janice_throw2 Janice_throw3 : ℕ)
 
-- Conditions given in the problem
def conditions :=
  Christine_throw1 = 20 ∧
  Janice_throw1 = Christine_throw1 - 4 ∧
  Christine_throw2 = Christine_throw1 + 10 ∧
  Janice_throw2 = Janice_throw1 * 2 ∧
  Christine_throw3 = Christine_throw2 + 4 ∧
  Janice_throw3 = Christine_throw1 + 17

-- The proof statement
theorem highest_throw_among_them : conditions -> max (max (max (max (max Christine_throw1 Christine_throw2) Christine_throw3) Janice_throw1) Janice_throw2) Janice_throw3 = 37 :=
sorry

end highest_throw_among_them_l308_308347


namespace pipe_b_fills_tank_7_times_faster_l308_308851

theorem pipe_b_fills_tank_7_times_faster 
  (time_A : ℝ) 
  (time_B : ℝ)
  (combined_time : ℝ) 
  (hA : time_A = 30)
  (h_combined : combined_time = 3.75) 
  (hB : time_B = time_A / 7) :
  time_B =  30 / 7 :=
by
  sorry

end pipe_b_fills_tank_7_times_faster_l308_308851


namespace cube_root_neg_eighth_l308_308639

theorem cube_root_neg_eighth : ∃ x : ℚ, x^3 = -1 / 8 ∧ x = -1 / 2 :=
by
  sorry

end cube_root_neg_eighth_l308_308639


namespace solve_fraction_sum_l308_308557

noncomputable theory

def roots_of_polynomial : Prop :=
  let a b c := classical.some (roots_of_polynomial_eq (x^3 - 15 * x^2 + 22 * x - 8 = 0)) in
  (a + b + c = 15) ∧ (ab + ac + bc = 22) ∧ (abc = 8)

theorem solve_fraction_sum (a b c : ℝ) (h₁ : a + b + c = 15) (h₂ : ab + ac + bc = 22) (h₃ : abc = 8) :
  (\frac{a}{\frac{1}{a}+bc} + \frac{b}{\frac{1}{b}+ca} + \frac{c}{\frac{1}{c}+ab}) = \frac{181}{9} :=
  by
    sorry

end solve_fraction_sum_l308_308557


namespace find_numbers_l308_308017

def is_solution (a b : ℕ) : Prop :=
  a + b = 432 ∧ (max a b) = 5 * (min a b) ∧ (max a b = 360 ∧ min a b = 72)

theorem find_numbers : ∃ a b : ℕ, is_solution a b :=
by
  sorry

end find_numbers_l308_308017


namespace no_nonzero_integer_solution_l308_308573

theorem no_nonzero_integer_solution (x y z : ℤ) (h : x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) :
  x^2 + y^2 ≠ 3 * z^2 :=
by
  sorry

end no_nonzero_integer_solution_l308_308573


namespace gcd_of_polynomial_and_multiple_l308_308366

-- Definitions based on given conditions
def multiple_of (a b : ℕ) : Prop := ∃ k, a = k * b

-- The main statement of the problem
theorem gcd_of_polynomial_and_multiple (y : ℕ) (h : multiple_of y 56790) :
  Nat.gcd ((3 * y + 2) * (5 * y + 3) * (11 * y + 7) * (y + 17)) y = 714 :=
sorry

end gcd_of_polynomial_and_multiple_l308_308366


namespace optionA_optionB_optionC_optionD_l308_308755

-- Statement for option A
theorem optionA : (∀ x : ℝ, x ≠ 3 → x^2 - 4 * x + 3 ≠ 0) ↔ (x^2 - 4 * x + 3 = 0 → x = 3) := sorry

-- Statement for option B
theorem optionB : (¬ (∀ x : ℝ, x^2 - x + 2 > 0) ↔ ∃ x0 : ℝ, x0^2 - x0 + 2 ≤ 0) := sorry

-- Statement for option C
theorem optionC (p q : Prop) : p ∧ q → p ∧ q := sorry

-- Statement for option D
theorem optionD (x : ℝ) : (x > -1 → x^2 + 4 * x + 3 > 0) ∧ ¬ (∀ x : ℝ, x^2 + 4 * x + 3 > 0 → x > -1) := sorry

end optionA_optionB_optionC_optionD_l308_308755


namespace apples_left_l308_308179

theorem apples_left (initial_apples : ℕ) (ricki_removes : ℕ) (samson_removes : ℕ) 
  (h1 : initial_apples = 74) 
  (h2 : ricki_removes = 14) 
  (h3 : samson_removes = 2 * ricki_removes) : 
  initial_apples - (ricki_removes + samson_removes) = 32 := 
by
  sorry

end apples_left_l308_308179


namespace average_of_original_set_l308_308580

theorem average_of_original_set (A : ℝ) (h1 : (35 * A) = (7 * 75)) : A = 15 := 
by sorry

end average_of_original_set_l308_308580


namespace partI_l308_308267

noncomputable def f (x : ℝ) : ℝ := abs (1 - 1/x)

theorem partI (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) (h4 : f a = f b) :
  a * b > 1 :=
  sorry

end partI_l308_308267


namespace principal_sum_correct_l308_308140

noncomputable def principal_sum (CI SI : ℝ) (t : ℕ) : ℝ :=
  let P := ((SI * t) / t) in
  let x := 5100 / P in
  26010000 / ((CI - SI) / t)

theorem principal_sum_correct :
  principal_sum 11730 10200 2 ≈ 16993.46 :=
by
  simp only [principal_sum]
  sorry

end principal_sum_correct_l308_308140


namespace clock_first_ring_at_midnight_l308_308787

theorem clock_first_ring_at_midnight (rings_every_n_hours : ℕ) (rings_per_day : ℕ) (hours_in_day : ℕ) :
  rings_every_n_hours = 3 ∧ rings_per_day = 8 ∧ hours_in_day = 24 →
  ∃ first_ring_time : Nat, first_ring_time = 0 :=
by
  sorry

end clock_first_ring_at_midnight_l308_308787


namespace total_pizzas_bought_l308_308173

theorem total_pizzas_bought (slices_small : ℕ) (slices_medium : ℕ) (slices_large : ℕ) 
                            (num_small : ℕ) (num_medium : ℕ) (total_slices : ℕ) :
  slices_small = 6 → 
  slices_medium = 8 → 
  slices_large = 12 → 
  num_small = 4 → 
  num_medium = 5 → 
  total_slices = 136 → 
  (total_slices = num_small * slices_small + num_medium * slices_medium + 72) →
  15 = num_small + num_medium + 6 :=
by
  intros
  sorry

end total_pizzas_bought_l308_308173


namespace Gloria_pine_tree_price_l308_308820

theorem Gloria_pine_tree_price :
  ∀ (cabin_cost cash cypress_count pine_count maple_count cypress_price maple_price left_over_price : ℕ)
  (cypress_total maple_total total_required total_from_cypress_and_maple total_needed amount_per_pine : ℕ),
    cabin_cost = 129000 →
    cash = 150 →
    cypress_count = 20 →
    pine_count = 600 →
    maple_count = 24 →
    cypress_price = 100 →
    maple_price = 300 →
    left_over_price = 350 →
    cypress_total = cypress_count * cypress_price →
    maple_total = maple_count * maple_price →
    total_required = cabin_cost - cash + left_over_price →
    total_from_cypress_and_maple = cypress_total + maple_total →
    total_needed = total_required - total_from_cypress_and_maple →
    amount_per_pine = total_needed / pine_count →
    amount_per_pine = 200 :=
by
  intros
  sorry

end Gloria_pine_tree_price_l308_308820


namespace triangle_area_l308_308003

theorem triangle_area (a b c : ℝ) (K : ℝ) (m n p : ℕ) (h1 : a = 10) (h2 : b = 12) (h3 : c = 15)
  (h4 : K = 240 * Real.sqrt 7 / 7)
  (h5 : Int.gcd m p = 1) -- m and p are relatively prime
  (h6 : n ≠ 1 ∧ ¬ (∃ x, x^2 ∣ n ∧ x > 1)) -- n is not divisible by the square of any prime
  : m + n + p = 254 := sorry

end triangle_area_l308_308003


namespace nonagon_blue_quadrilateral_l308_308391

theorem nonagon_blue_quadrilateral :
  ∀ (vertices : Finset ℕ) (red blue : ℕ → ℕ → Prop),
    (vertices.card = 9) →
    (∀ a b, red a b ∨ blue a b) →
    (∀ a b c, (red a b ∧ red b c ∧ red c a) → False) →
    (∃ A B C D, blue A B ∧ blue B C ∧ blue C D ∧ blue D A ∧ blue A C ∧ blue B D) := 
by
  -- Proof goes here
  sorry

end nonagon_blue_quadrilateral_l308_308391


namespace rosie_pies_l308_308273

theorem rosie_pies (apples_per_pie : ℕ) (apples_total : ℕ) (pies_initial : ℕ) 
  (h1 : 3 = pies_initial) (h2 : 12 = apples_total) : 
  (36 / (apples_total / pies_initial)) * pies_initial = 27 := 
by
  sorry

end rosie_pies_l308_308273


namespace freddy_age_l308_308192

theorem freddy_age
  (mat_age : ℕ)  -- Matthew's age
  (reb_age : ℕ)  -- Rebecca's age
  (fre_age : ℕ)  -- Freddy's age
  (h1 : mat_age = reb_age + 2)
  (h2 : fre_age = mat_age + 4)
  (h3 : mat_age + reb_age + fre_age = 35) :
  fre_age = 15 :=
by sorry

end freddy_age_l308_308192


namespace blue_markers_count_l308_308792

-- Definitions based on the problem's conditions
def total_markers : ℕ := 3343
def red_markers : ℕ := 2315

-- Statement to prove
theorem blue_markers_count :
  total_markers - red_markers = 1028 := by
  sorry

end blue_markers_count_l308_308792


namespace floor_e_equals_2_l308_308212

theorem floor_e_equals_2 : Int.floor Real.exp = 2 := 
sorry

end floor_e_equals_2_l308_308212


namespace correct_meteor_passing_time_l308_308746

theorem correct_meteor_passing_time :
  let T1 := 7
  let T2 := 13
  let harmonic_mean := (2 * T1 * T2) / (T1 + T2)
  harmonic_mean = 9.1 := 
by
  sorry

end correct_meteor_passing_time_l308_308746


namespace range_of_a_l308_308239

theorem range_of_a (a : ℝ) :
  (∀ θ : ℝ, 0 < θ ∧ θ < (π / 2) → a ≤ 1 / Real.sin θ + 1 / Real.cos θ) ↔ a ≤ 2 * Real.sqrt 2 :=
sorry

end range_of_a_l308_308239


namespace y_positive_if_and_only_if_x_greater_than_negative_five_over_two_l308_308068

variable (x y : ℝ)

-- Condition: y is defined as a function of x
def y_def := y = 2 * x + 5

-- Theorem: y > 0 if and only if x > -5/2
theorem y_positive_if_and_only_if_x_greater_than_negative_five_over_two 
  (h : y_def x y) : y > 0 ↔ x > -5 / 2 := by sorry

end y_positive_if_and_only_if_x_greater_than_negative_five_over_two_l308_308068


namespace sqrt_of_neg_five_squared_l308_308928

theorem sqrt_of_neg_five_squared : Real.sqrt ((-5 : Real) ^ 2) = 5 := 
by 
  sorry

end sqrt_of_neg_five_squared_l308_308928


namespace thomas_worked_hours_l308_308023

theorem thomas_worked_hours (Toby Thomas Rebecca : ℕ) 
  (h_total : Thomas + Toby + Rebecca = 157) 
  (h_toby : Toby = 2 * Thomas - 10) 
  (h_rebecca_1 : Rebecca = Toby - 8) 
  (h_rebecca_2 : Rebecca = 56) : Thomas = 37 :=
by
  sorry

end thomas_worked_hours_l308_308023


namespace solve_for_x_l308_308825

theorem solve_for_x (x : ℝ) (h : 1 / 4 - 1 / 6 = 4 / x) : x = 48 := 
sorry

end solve_for_x_l308_308825


namespace cubic_expression_l308_308981

theorem cubic_expression (a b c : ℝ) (h₁ : a + b + c = 12) (h₂ : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 1008 :=
sorry

end cubic_expression_l308_308981


namespace five_x_minus_two_l308_308249

theorem five_x_minus_two (x : ℚ) (h : 4 * x - 8 = 13 * x + 3) : 5 * (x - 2) = -145 / 9 := by
  sorry

end five_x_minus_two_l308_308249


namespace total_emails_received_l308_308342

theorem total_emails_received (E : ℝ)
    (h1 : (3/5) * (3/4) * E = 180) :
    E = 400 :=
sorry

end total_emails_received_l308_308342


namespace sum_of_1984_consecutive_integers_not_square_l308_308732

theorem sum_of_1984_consecutive_integers_not_square :
  ∀ n : ℕ, ¬ ∃ k : ℕ, 992 * (2 * n + 1985) = k * k := by
  sorry

end sum_of_1984_consecutive_integers_not_square_l308_308732


namespace certain_number_is_negative_425_l308_308181

theorem certain_number_is_negative_425 (x : ℝ) :
  (3 - (1/5) * x = 88) ∧ (4 - (1/7) * 210 = -26) → x = -425 :=
by
  sorry

end certain_number_is_negative_425_l308_308181


namespace dryer_less_than_washing_machine_by_30_l308_308673

-- Definitions based on conditions
def washing_machine_price : ℝ := 100
def discount_rate : ℝ := 0.10
def total_paid_after_discount : ℝ := 153

-- The equation for price of the dryer
def original_dryer_price (D : ℝ) : Prop :=
  washing_machine_price + D - discount_rate * (washing_machine_price + D) = total_paid_after_discount

-- The statement we need to prove
theorem dryer_less_than_washing_machine_by_30 (D : ℝ) (h : original_dryer_price D) :
  washing_machine_price - D = 30 :=
by 
  sorry

end dryer_less_than_washing_machine_by_30_l308_308673


namespace exists_half_perimeter_area_rectangle_6x1_not_exists_half_perimeter_area_rectangle_2x1_l308_308645

theorem exists_half_perimeter_area_rectangle_6x1 :
  ∃ x₁ x₂ : ℝ, (6 * 1 / 2 = (6 + 1) / 2) ∧
                x₁ * x₂ = 3 ∧
                (x₁ + x₂ = 3.5) ∧
                (x₁ = 2 ∨ x₁ = 1.5) ∧
                (x₂ = 2 ∨ x₂ = 1.5)
:= by
  sorry

theorem not_exists_half_perimeter_area_rectangle_2x1 :
  ¬(∃ x : ℝ, x * (1.5 - x) = 1)
:= by
  sorry

end exists_half_perimeter_area_rectangle_6x1_not_exists_half_perimeter_area_rectangle_2x1_l308_308645


namespace problem1_problem2_l308_308237

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.log x - 1

noncomputable def g (x : ℝ) : ℝ := x / Real.exp x

theorem problem1 (a : ℝ) (h1 : 2 / Real.exp 2 < a) (h2 : a < 1 / Real.exp 1) :
  ∃ (x1 x2 : ℝ), (0 < x1 ∧ x1 < 2) ∧ (0 < x2 ∧ x2 < 2) ∧ x1 ≠ x2 ∧ g x1 = a ∧ g x2 = a :=
sorry

theorem problem2 : ∀ x > 0, f x + 2 / (Real.exp 1 * g x) > 0 :=
sorry

end problem1_problem2_l308_308237


namespace triangle_side_relationship_l308_308402

noncomputable def sin (x : ℝ) : ℝ := sorry
noncomputable def cos (x : ℝ) : ℝ := sorry

theorem triangle_side_relationship 
  (a b c : ℝ)
  (α β γ : ℝ)
  (hα : α = 40 * Real.pi / 180)
  (hβ : β = 60 * Real.pi / 180)
  (hγ : γ = 80 * Real.pi / 180)
  (h_angle_sum : α + β + γ = Real.pi) : 
  a * (a + b + c) = b * (b + c) :=
sorry

end triangle_side_relationship_l308_308402


namespace Owen_profit_l308_308414

/-- 
Owen bought 12 boxes of face masks, each box costing $9 and containing 50 masks. 
He repacked 6 boxes into smaller packs sold for $5 per 25 masks and sold the remaining masks in baggies of 10 pieces for $3 each.
Prove that Owen's profit amounts to $42.
 -/
theorem Owen_profit :
  let box_count := 12
  let cost_per_box := 9
  let masks_per_box := 50
  let repacked_boxes := 6
  let repack_price := 5
  let repack_size := 25
  let baggy_price := 3
  let baggy_size := 10 in
  let total_cost := box_count * cost_per_box in
  let total_masks := box_count * masks_per_box in
  let masks_repacked := repacked_boxes * masks_per_box in
  let repacked_revenue := (masks_repacked / repack_size) * repack_price in
  let remaining_masks := total_masks - masks_repacked in
  let baggy_revenue := (remaining_masks / baggy_size) * baggy_price in
  let total_revenue := repacked_revenue + baggy_revenue in
  let profit := total_revenue - total_cost in
  profit = 42 := by
  sorry

end Owen_profit_l308_308414


namespace point_relationship_on_parabola_neg_x_plus_1_sq_5_l308_308809

theorem point_relationship_on_parabola_neg_x_plus_1_sq_5
  (y_1 y_2 y_3 : ℝ) :
  (A : ℝ × ℝ) = (-2, y_1) →
  (B : ℝ × ℝ) = (1, y_2) →
  (C : ℝ × ℝ) = (2, y_3) →
  (A.2 = -(A.1 + 1)^2 + 5) →
  (B.2 = -(B.1 + 1)^2 + 5) →
  (C.2 = -(C.1 + 1)^2 + 5) →
  y_1 > y_2 ∧ y_2 > y_3 :=
by
  sorry

end point_relationship_on_parabola_neg_x_plus_1_sq_5_l308_308809


namespace sum_n_10_terms_progression_l308_308444

noncomputable def sum_arith_progression (n a d : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_n_10_terms_progression :
  ∃ (a : ℕ), (∃ (n : ℕ), sum_arith_progression n a 3 = 220) ∧
  (2 * a + (10 - 1) * 3) = 43 ∧
  sum_arith_progression 10 a 3 = 215 :=
by sorry

end sum_n_10_terms_progression_l308_308444


namespace quadratic_sum_of_roots_l308_308069

theorem quadratic_sum_of_roots (a b : ℝ)
  (h1: ∀ x: ℝ, x^2 + b * x - a < 0 ↔ 3 < x ∧ x < 4):
  a + b = -19 :=
sorry

end quadratic_sum_of_roots_l308_308069


namespace scientific_notation_of_192M_l308_308566

theorem scientific_notation_of_192M : 192000000 = 1.92 * 10^8 :=
by 
  sorry

end scientific_notation_of_192M_l308_308566


namespace product_mod_seven_l308_308651

theorem product_mod_seven :
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 4 :=
by
  have h₁ : 3 % 7 = 3 := rfl,
  have h₂ : 13 % 7 = 6 := rfl,
  have h₃ : 23 % 7 = 2 := rfl,
  have h₄ : 33 % 7 = 5 := rfl,
  have h₅ : 43 % 7 = 1 := rfl,
  have h₆ : 53 % 7 = 4 := rfl,
  have h₇ : 63 % 7 = 0 := rfl,
  have h₈ : 73 % 7 = 3 := rfl,
  have h₉ : 83 % 7 = 6 := rfl,
  have h₁₀ : 93 % 7 = 2 := rfl,
  sorry

end product_mod_seven_l308_308651


namespace find_third_angle_l308_308395

-- Definitions from the problem conditions
def triangle_angle_sum (a b c : ℝ) : Prop := a + b + c = 180

-- Statement of the proof problem
theorem find_third_angle (a b x : ℝ) (h1 : a = 50) (h2 : b = 45) (h3 : triangle_angle_sum a b x) : x = 85 := sorry

end find_third_angle_l308_308395


namespace min_sum_of_squares_l308_308561

theorem min_sum_of_squares (y1 y2 y3 : ℝ) (h1 : y1 > 0) (h2 : y2 > 0) (h3 : y3 > 0) (h4 : y1 + 3 * y2 + 4 * y3 = 72) : 
  y1^2 + y2^2 + y3^2 ≥ 2592 / 13 ∧ (∃ k, y1 = k ∧ y2 = 3 * k ∧ y3 = 4 * k ∧ k = 36 / 13) :=
sorry

end min_sum_of_squares_l308_308561


namespace a_5_is_31_l308_308528

/-- Define the sequence a_n recursively -/
def a : Nat → Nat
| 0        => 1
| (n + 1)  => 2 * a n + 1

/-- Prove that the 5th term in the sequence is 31 -/
theorem a_5_is_31 : a 5 = 31 := 
sorry

end a_5_is_31_l308_308528


namespace max_value_of_y_l308_308844

noncomputable def maxY (x y : ℝ) : ℝ :=
  if x^2 + y^2 = 10 * x + 60 * y then y else 0

theorem max_value_of_y (x y : ℝ) (h : x^2 + y^2 = 10 * x + 60 * y) : 
  y ≤ 30 + 5 * Real.sqrt 37 :=
sorry

end max_value_of_y_l308_308844


namespace arithmetic_seq_sum_l308_308766

theorem arithmetic_seq_sum (a : ℕ → ℝ) (d : ℝ) 
  (h₁ : ∀ n : ℕ, a (n + 1) = a n + d)
  (h₂ : a 3 + a 7 = 37) :
  a 2 + a 4 + a 6 + a 8 = 74 := 
sorry

end arithmetic_seq_sum_l308_308766


namespace arithmetic_sequence_l308_308258

theorem arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) (h₀ : a 1 = 2) (h₁ : a 2 + a 3 = 13)
    (h₂ : ∀ n, a n = a 1 + (n - 1) * d) : a 5 = 14 :=
by
  sorry

end arithmetic_sequence_l308_308258


namespace sqrt_of_neg_five_squared_l308_308929

theorem sqrt_of_neg_five_squared : Real.sqrt ((-5 : Real) ^ 2) = 5 := 
by 
  sorry

end sqrt_of_neg_five_squared_l308_308929


namespace ice_cream_arrangements_is_correct_l308_308419

-- Let us define the problem: counting the number of unique stacks of ice cream flavors
def ice_cream_scoops_arrangements : ℕ :=
  let total_scoops := 5
  let vanilla_scoops := 2
  Nat.factorial total_scoops / Nat.factorial vanilla_scoops

-- Assertion that needs to be proved
theorem ice_cream_arrangements_is_correct : ice_cream_scoops_arrangements = 60 := by
  -- Proof to be filled in; current placeholder
  sorry

end ice_cream_arrangements_is_correct_l308_308419


namespace hazel_made_56_cups_l308_308075

-- Definitions based on problem conditions:
def sold_to_kids (sold: ℕ) := sold = 18
def gave_away (sold: ℕ) (gave: ℕ) := gave = sold / 2
def drank (drank: ℕ) := drank = 1
def half_total (total: ℕ) (sum_sold_gave_drank: ℕ) := sum_sold_gave_drank = total / 2

-- Main statement that needs to be proved:
theorem hazel_made_56_cups : ∃ (total: ℕ), 
  ∀ (sold gave drank sum_sold_gave_drank: ℕ), 
    sold_to_kids sold → 
    gave_away sold gave → 
    drank drank → 
    half_total total (sold + gave + drank) → 
    total = 56 := 
by sorry

end hazel_made_56_cups_l308_308075


namespace train_cross_time_platform_l308_308039

def speed := 36 -- in kmph
def time_for_pole := 12 -- in seconds
def time_for_platform := 44.99736021118311 -- in seconds

theorem train_cross_time_platform :
  time_for_platform = 44.99736021118311 :=
by
  sorry

end train_cross_time_platform_l308_308039


namespace age_of_15th_person_l308_308905

theorem age_of_15th_person (avg_16 : ℝ) (avg_5 : ℝ) (avg_9 : ℝ) (total_16 : ℝ) (total_5 : ℝ) (total_9 : ℝ) :
  avg_16 = 15 ∧ avg_5 = 14 ∧ avg_9 = 16 ∧
  total_16 = 16 * avg_16 ∧ total_5 = 5 * avg_5 ∧ total_9 = 9 * avg_9 →
  (total_16 - total_5 - total_9) = 26 :=
by
  sorry

end age_of_15th_person_l308_308905


namespace number_of_integers_satisfying_l308_308246

theorem number_of_integers_satisfying (k1 k2 : ℕ) (hk1 : k1 = 300) (hk2 : k2 = 1000) :
  ∃ m : ℕ, m = 14 ∧ ∀ n : ℕ, 300 < n^2 → n^2 < 1000 → 18 ≤ n ∧ n ≤ 31 :=
by
  use 14
  sorry

end number_of_integers_satisfying_l308_308246


namespace total_revenue_full_price_l308_308629

theorem total_revenue_full_price (f d p : ℕ) (h1 : f + d = 200) (h2 : f * p + d * (3 * p) / 4 = 2800) : 
  f * p = 680 :=
by
  -- proof omitted
  sorry

end total_revenue_full_price_l308_308629


namespace sum_of_cubes_application_l308_308516

theorem sum_of_cubes_application : 
  ¬ ((a+1) * (a^2 - a + 1) = a^3 + 1) :=
by
  sorry

end sum_of_cubes_application_l308_308516


namespace cost_per_mile_sunshine_is_018_l308_308578

theorem cost_per_mile_sunshine_is_018 :
  ∀ (x : ℝ) (daily_rate_sunshine daily_rate_city cost_per_mile_city : ℝ),
  daily_rate_sunshine = 17.99 →
  daily_rate_city = 18.95 →
  cost_per_mile_city = 0.16 →
  (daily_rate_sunshine + 48 * x = daily_rate_city + cost_per_mile_city * 48) →
  x = 0.18 :=
by
  intros x daily_rate_sunshine daily_rate_city cost_per_mile_city
  intros h1 h2 h3 h4
  sorry

end cost_per_mile_sunshine_is_018_l308_308578


namespace calculate_value_l308_308924

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

variable (f : ℝ → ℝ)

axiom h : odd_function f
axiom h1 : increasing_on_interval f 3 7
axiom h2 : f 3 = -1
axiom h3 : f 6 = 8

theorem calculate_value : 2 * f (-6) + f (-3) = -15 := by
  sorry

end calculate_value_l308_308924


namespace superdomino_probability_l308_308631

-- Definitions based on conditions
def is_superdomino (a b : ℕ) : Prop := 0 ≤ a ∧ a ≤ 12 ∧ 0 ≤ b ∧ b ≤ 12
def is_superdouble (a b : ℕ) : Prop := a = b
def total_superdomino_count : ℕ := 13 * 13
def superdouble_count : ℕ := 13

-- Proof statement
theorem superdomino_probability : (superdouble_count : ℚ) / total_superdomino_count = 13 / 169 :=
by
  sorry

end superdomino_probability_l308_308631


namespace equation_of_motion_l308_308874

section MotionLaw

variable (t s : ℝ)
variable (v : ℝ → ℝ)
variable (C : ℝ)

-- Velocity function
def velocity (t : ℝ) : ℝ := 6 * t^2 + 1

-- Displacement function (indefinite integral of velocity)
def displacement (t : ℝ) (C : ℝ) : ℝ := 2 * t^3 + t + C

-- Given condition: displacement at t = 3 is 60
axiom displacement_at_3 : displacement 3 C = 60

-- Prove that the equation of motion is s = 2t^3 + t + 3
theorem equation_of_motion :
  ∃ C, displacement t C = 2 * t^3 + t + 3 :=
by
  use 3
  sorry

end MotionLaw

end equation_of_motion_l308_308874


namespace solution_pairs_l308_308195

theorem solution_pairs (x y : ℝ) : 
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ (y = -x - 2 ∨ y = -2 * x + 1) := 
by 
  sorry

end solution_pairs_l308_308195


namespace ratio_of_sleep_l308_308640

theorem ratio_of_sleep (connor_sleep : ℝ) (luke_extra : ℝ) (puppy_sleep : ℝ) 
    (h1 : connor_sleep = 6)
    (h2 : luke_extra = 2)
    (h3 : puppy_sleep = 16) :
    puppy_sleep / (connor_sleep + luke_extra) = 2 := 
by 
  sorry

end ratio_of_sleep_l308_308640


namespace max_value_of_x_plus_y_plus_z_l308_308434

theorem max_value_of_x_plus_y_plus_z : ∀ (x y z : ℤ), (∃ k : ℤ, x = 5 * k ∧ 6 = y * k ∧ z = 2 * k) → x + y + z ≤ 43 :=
by
  intros x y z h
  rcases h with ⟨k, hx, hy, hz⟩
  sorry

end max_value_of_x_plus_y_plus_z_l308_308434


namespace units_digit_of_result_l308_308861

theorem units_digit_of_result (a b c : ℕ) (h1 : a = c + 3) : 
  let original := 100 * a + 10 * b + c
  let reversed := 100 * c + 10 * b + a
  let result := original - reversed
  result % 10 = 7 :=
by
  sorry

end units_digit_of_result_l308_308861


namespace equivalent_problem_l308_308666

variable {a : ℕ → ℝ} {b : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

def is_geometric_sequence (b : ℕ → ℝ) :=
  ∀ n m : ℕ, b (n + 1) / b n = b (m + 1) / b m

theorem equivalent_problem
  (ha : is_arithmetic_sequence a)
  (hb : is_geometric_sequence b)
  (h1 : a 1 + a 3 + a 5 + a 7 + a 9 = 50)
  (h2 : b 4 * b 6 * b 14 * b 16 = 625) :
  (a 2 + a 8) / b 10 = 4 ∨ (a 2 + a 8) / b 10 = -4 := 
sorry

end equivalent_problem_l308_308666


namespace habitat_limits_are_correct_l308_308469

-- Definitions of the conditions
def colonyA_doubling_days : ℕ := 22
def colonyB_tripling_days : ℕ := 30
def tripling_interval : ℕ := 2

-- Definitions to confirm they grow as described
def is_colonyA_habitat_limit_reached (days : ℕ) : Prop := days = colonyA_doubling_days
def is_colonyB_habitat_limit_reached (days : ℕ) : Prop := days = colonyB_tripling_days

-- Proof statement
theorem habitat_limits_are_correct :
  (is_colonyA_habitat_limit_reached colonyA_doubling_days) ∧ (is_colonyB_habitat_limit_reached colonyB_tripling_days) :=
by
  sorry

end habitat_limits_are_correct_l308_308469


namespace problem1_problem2_l308_308518

-- Define the given angle
def given_angle (α : ℝ) : Prop := α = 2010

-- Define the theorem for the first problem
theorem problem1 (α : ℝ) (k : ℤ) (β : ℝ) (h₁ : given_angle α) 
  (h₂ : 0 ≤ β ∧ β < 360) (h₃ : α = k * 360 + β) : 
  -- Assert that α is in the third quadrant
  (190 ≤ β ∧ β < 270 → true) :=
sorry

-- Define the theorem for the second problem
theorem problem2 (α : ℝ) (θ : ℝ) (h₁ : given_angle α)
  (h₂ : -360 ≤ θ ∧ θ < 720)
  (h₃ : ∃ k : ℤ, θ = α + k * 360) : 
  θ = -150 ∨ θ = 210 ∨ θ = 570 :=
sorry

end problem1_problem2_l308_308518


namespace range_of_a_l308_308533

theorem range_of_a (a : ℝ) :
  (∀ x : ℤ, 2 * a * (x : ℝ)^2 - 4 * (x : ℝ) < a * (x : ℝ) - 2 → ∃! x₀ : ℤ, x₀ = x) → 1 ≤ a ∧ a < 2 :=
sorry

end range_of_a_l308_308533


namespace evaluate_at_minus_three_l308_308713

def g (x : ℝ) : ℝ := 3 * x^5 - 5 * x^4 + 9 * x^3 - 6 * x^2 + 15 * x - 210

theorem evaluate_at_minus_three : g (-3) = -1686 :=
by
  sorry

end evaluate_at_minus_three_l308_308713


namespace highest_throw_among_them_l308_308348

variable (Christine_throw1 Christine_throw2 Christine_throw3 Janice_throw1 Janice_throw2 Janice_throw3 : ℕ)
 
-- Conditions given in the problem
def conditions :=
  Christine_throw1 = 20 ∧
  Janice_throw1 = Christine_throw1 - 4 ∧
  Christine_throw2 = Christine_throw1 + 10 ∧
  Janice_throw2 = Janice_throw1 * 2 ∧
  Christine_throw3 = Christine_throw2 + 4 ∧
  Janice_throw3 = Christine_throw1 + 17

-- The proof statement
theorem highest_throw_among_them : conditions -> max (max (max (max (max Christine_throw1 Christine_throw2) Christine_throw3) Janice_throw1) Janice_throw2) Janice_throw3 = 37 :=
sorry

end highest_throw_among_them_l308_308348


namespace chef_makes_10_cakes_l308_308581

def total_eggs : ℕ := 60
def eggs_in_fridge : ℕ := 10
def eggs_per_cake : ℕ := 5

theorem chef_makes_10_cakes :
  (total_eggs - eggs_in_fridge) / eggs_per_cake = 10 := by
  sorry

end chef_makes_10_cakes_l308_308581


namespace point_P_quadrant_l308_308665

theorem point_P_quadrant 
  (h1 : Real.sin (θ / 2) = 3 / 5) 
  (h2 : Real.cos (θ / 2) = -4 / 5) : 
  (0 < Real.cos θ) ∧ (Real.sin θ < 0) :=
by
  sorry

end point_P_quadrant_l308_308665


namespace sequence_sum_l308_308906

open BigOperators

-- Define the general term
def term (n : ℕ) : ℚ := n * (1 - (1 / n))

-- Define the index range for the sequence
def index_range : Finset ℕ := Finset.range 9 \ {0, 1}

-- Lean statement of the problem
theorem sequence_sum : ∑ n in index_range, term (n + 2) = 45 := by
  sorry

end sequence_sum_l308_308906


namespace fractions_product_l308_308186

theorem fractions_product : 
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 3 / 7 := 
by 
  sorry

end fractions_product_l308_308186


namespace fraction_people_over_65_l308_308949

theorem fraction_people_over_65 {T O U: ℚ} 
    (h1: U / T = 3 / 7) -- Condition 1: Under 21 people fraction
    (h2: U = 30)        -- Condition 4: Number of people under 21
    (h3: 50 < T)        -- Condition 3: Total people greater than 50
    (h4: T < 100)       -- Condition 3: Total people less than 100) :
    (h5: O ≤ T - U)     -- Condition 2: Certain fraction of people over 65):

    O / T ≤ 4 / 7 := 
by 
  -- Start with given conditions to derive the answer
  sorry -- Placeholder for the proof

end fraction_people_over_65_l308_308949


namespace additional_spending_required_l308_308450

def cost_of_chicken : ℝ := 1.5 * 6.00
def cost_of_lettuce : ℝ := 3.00
def cost_of_cherry_tomatoes : ℝ := 2.50
def cost_of_sweet_potatoes : ℝ := 4 * 0.75
def cost_of_broccoli : ℝ := 2 * 2.00
def cost_of_brussel_sprouts : ℝ := 2.50
def total_cost : ℝ := cost_of_chicken + cost_of_lettuce + cost_of_cherry_tomatoes + cost_of_sweet_potatoes + cost_of_broccoli + cost_of_brussel_sprouts
def minimum_spending_for_free_delivery : ℝ := 35.00
def additional_amount_needed : ℝ := minimum_spending_for_free_delivery - total_cost

theorem additional_spending_required : additional_amount_needed = 11.00 := by
  sorry

end additional_spending_required_l308_308450


namespace radius_of_circle_l308_308740

noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2
noncomputable def circle_circumference (r : ℝ) : ℝ := 2 * Real.pi * r

theorem radius_of_circle : 
  ∃ r : ℝ, circle_area r = circle_circumference r → r = 2 := 
by 
  sorry

end radius_of_circle_l308_308740


namespace find_v_l308_308793

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, 1;
    3, 0]

noncomputable def v : Matrix (Fin 2) (Fin 1) ℝ :=
  !![0;
    1 / 30.333]

noncomputable def I : Matrix (Fin 2) (Fin 2) ℝ :=
  1

theorem find_v : 
  (A ^ 10 + A ^ 8 + A ^ 6 + A ^ 4 + A ^ 2 + I) * v = !![0; 12] :=
  sorry

end find_v_l308_308793


namespace solution_pairs_l308_308197

theorem solution_pairs (x y : ℝ) : 
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ (y = -x - 2 ∨ y = -2 * x + 1) := 
by 
  sorry

end solution_pairs_l308_308197


namespace baseEight_conversion_l308_308455

-- Base-eight number is given as 1563
def baseEight : Nat := 1563

-- Function to convert a base-eight number to base-ten
noncomputable def baseEightToBaseTen (n : Nat) : Nat :=
  let digit3 := (n / 1000) % 10
  let digit2 := (n / 100) % 10
  let digit1 := (n / 10) % 10
  let digit0 := n % 10
  digit3 * 8^3 + digit2 * 8^2 + digit1 * 8^1 + digit0 * 8^0

theorem baseEight_conversion :
  baseEightToBaseTen baseEight = 883 := by
  sorry

end baseEight_conversion_l308_308455


namespace total_votes_cast_is_8200_l308_308907

variable (V : ℝ) (h1 : 0.35 * V < V) (h2 : 0.35 * V + 2460 = 0.65 * V)

theorem total_votes_cast_is_8200 (V : ℝ)
  (h1 : 0.35 * V < V)
  (h2 : 0.35 * V + 2460 = 0.65 * V) :
  V = 8200 := by
sorry

end total_votes_cast_is_8200_l308_308907


namespace correlation_comparison_l308_308225

-- Definitions of the datasets
def data_XY : List (ℝ × ℝ) := [(10,1), (11.3,2), (11.8,3), (12.5,4), (13,5)]
def data_UV : List (ℝ × ℝ) := [(10,5), (11.3,4), (11.8,3), (12.5,2), (13,1)]

-- Definitions of the linear correlation coefficients
noncomputable def r1 : ℝ := sorry -- Calculation of correlation coefficient between X and Y
noncomputable def r2 : ℝ := sorry -- Calculation of correlation coefficient between U and V

-- The proof statement
theorem correlation_comparison :
  r2 < 0 ∧ 0 < r1 :=
sorry

end correlation_comparison_l308_308225


namespace calculate_annual_rent_l308_308119

-- Defining the conditions
def num_units : ℕ := 100
def occupancy_rate : ℚ := 3 / 4
def monthly_rent : ℚ := 400

-- Defining the target annual rent
def annual_rent (units : ℕ) (occupancy : ℚ) (rent : ℚ) : ℚ :=
  let occupied_units := occupancy * units
  let monthly_revenue := occupied_units * rent
  monthly_revenue * 12

-- Proof problem statement
theorem calculate_annual_rent :
  annual_rent num_units occupancy_rate monthly_rent = 360000 := by
  sorry

end calculate_annual_rent_l308_308119


namespace percentage_decrease_in_spring_l308_308040

-- Given Conditions
variables (initial_members : ℕ) (increased_percent : ℝ) (total_decrease_percent : ℝ)
-- population changes
variables (fall_members : ℝ) (spring_members : ℝ)

-- The initial conditions given by the problem
axiom initial_membership : initial_members = 100
axiom fall_increase : increased_percent = 6
axiom total_decrease : total_decrease_percent = 14.14

-- Derived values based on conditions
axiom fall_members_calculated : fall_members = initial_members * (1 + increased_percent / 100)
axiom spring_members_calculated : spring_members = initial_members * (1 - total_decrease_percent / 100)

-- The correct answer which we need to prove
theorem percentage_decrease_in_spring : 
  ((fall_members - spring_members) / fall_members) * 100 = 19 := by
  sorry

end percentage_decrease_in_spring_l308_308040


namespace amount_needed_for_free_delivery_l308_308447

theorem amount_needed_for_free_delivery :
  let chicken_cost := 1.5 * 6.00
  let lettuce_cost := 3.00
  let tomatoes_cost := 2.50
  let sweet_potatoes_cost := 4 * 0.75
  let broccoli_cost := 2 * 2.00
  let brussel_sprouts_cost := 2.50
  let total_cost := chicken_cost + lettuce_cost + tomatoes_cost + sweet_potatoes_cost + broccoli_cost + brussel_sprouts_cost
  let min_spend_for_free_delivery := 35.00
  min_spend_for_free_delivery - total_cost = 11.00 := sorry

end amount_needed_for_free_delivery_l308_308447


namespace quadratic_nonnegative_quadratic_inv_nonnegative_l308_308219

-- Problem Definitions and Proof Statements

variables {R : Type*} [LinearOrderedField R]

def f (a b c x : R) : R := a * x^2 + 2 * b * x + c

theorem quadratic_nonnegative {a b c : R} (ha : a ≠ 0) (h : ∀ x : R, f a b c x ≥ 0) : 
  a ≥ 0 ∧ c ≥ 0 ∧ a * c - b^2 ≥ 0 :=
sorry

theorem quadratic_inv_nonnegative {a b c : R} (ha : a ≥ 0) (hc : c ≥ 0) (hac : a * c - b^2 ≥ 0) :
  ∀ x : R, f a b c x ≥ 0 :=
sorry

end quadratic_nonnegative_quadratic_inv_nonnegative_l308_308219


namespace arun_age_l308_308689

theorem arun_age (A G M : ℕ) (h1 : (A - 6) / 18 = G) (h2 : G = M - 2) (h3 : M = 5) : A = 60 :=
by
  sorry

end arun_age_l308_308689


namespace find_n_divisible_by_highest_power_of_2_l308_308059

def a_n (n : ℕ) : ℕ :=
  10^n * 999 + 488

theorem find_n_divisible_by_highest_power_of_2:
  ∀ n : ℕ, (n > 0) → (a_n n = 10^n * 999 + 488) → (∃ k : ℕ, 2^(k + 9) ∣ a_n 6) := sorry

end find_n_divisible_by_highest_power_of_2_l308_308059


namespace good_partitions_count_l308_308963

def is_good_partition (A1 A2 A3 : Finset ℕ) : Prop :=
  ∃ (i1 i2 i3 : Finset ℕ) (h: {i1, i2, i3} = {A1, A2, A3}),
    (i1.nonempty ∧ i2.nonempty ∧ i3.nonempty) ∧
    (∃ k (h1: k ∈ i1) (h2: (k + 1) % 3 ∈ i2), true)

def M : Finset ℕ := {n ∈ Finset.range 11 | 0 < n}

def count_good_partitions : ℕ :=
  {p // let (A1, A2, A3, hp) := p in is_good_partition A1 A2 A3}.to_finset.card

theorem good_partitions_count :
  count_good_partitions = 8362 := sorry

end good_partitions_count_l308_308963


namespace total_students_in_both_classrooms_l308_308837

theorem total_students_in_both_classrooms
  (x y : ℕ)
  (hx1 : 80 * x - 250 = 90 * (x - 5))
  (hy1 : 85 * y - 480 = 95 * (y - 8)) :
  x + y = 48 := 
sorry

end total_students_in_both_classrooms_l308_308837


namespace floor_e_eq_two_l308_308215

theorem floor_e_eq_two
  (e_approx : Real ≈ 2.718) :
  ⌊e⌋ = 2 :=
sorry

end floor_e_eq_two_l308_308215


namespace stratified_sampling_b_members_l308_308624

variable (groupA : ℕ) (groupB : ℕ) (groupC : ℕ) (sampleSize : ℕ)

-- Conditions from the problem
def condition1 : groupA = 45 := by sorry
def condition2 : groupB = 45 := by sorry
def condition3 : groupC = 60 := by sorry
def condition4 : sampleSize = 10 := by sorry

-- The proof problem statement
theorem stratified_sampling_b_members : 
  (sampleSize * groupB) / (groupA + groupB + groupC) = 3 :=
by sorry

end stratified_sampling_b_members_l308_308624


namespace smallest_delightful_integer_l308_308731

-- Definition of "delightful" integer
def is_delightful (B : ℤ) : Prop :=
  ∃ (n : ℕ), (n > 0) ∧ ((n + 1) * (2 * B + n)) / 2 = 3050

-- Proving the smallest delightful integer
theorem smallest_delightful_integer : ∃ (B : ℤ), is_delightful B ∧ ∀ (B' : ℤ), is_delightful B' → B ≤ B' :=
  sorry

end smallest_delightful_integer_l308_308731


namespace jane_doe_total_investment_mutual_funds_l308_308262

theorem jane_doe_total_investment_mutual_funds :
  ∀ (c m : ℝ) (total_investment : ℝ),
  total_investment = 250000 → m = 3 * c → c + m = total_investment → m = 187500 :=
by
  intros c m total_investment h_total h_relation h_sum
  sorry

end jane_doe_total_investment_mutual_funds_l308_308262


namespace coordinate_equation_solution_l308_308200

theorem coordinate_equation_solution (x y : ℝ) :
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 →
  (y = -x - 2) ∨ (y = -2 * x + 1) :=
by
  sorry

end coordinate_equation_solution_l308_308200


namespace triangle_has_side_property_l308_308259

theorem triangle_has_side_property (a b c : ℝ) (A B C : ℝ) 
  (h₀ : 3 * b * Real.cos C + 3 * c * Real.cos B = a^2)
  (h₁ : A + B + C = Real.pi)
  (h₂ : a = 3) :
  a = 3 := 
sorry

end triangle_has_side_property_l308_308259


namespace jackson_paintable_area_l308_308706

namespace PaintWallCalculation

def length := 14
def width := 11
def height := 9
def windowArea := 70
def bedrooms := 4

def area_one_bedroom : ℕ :=
  2 * (length * height) + 2 * (width * height)

def paintable_area_one_bedroom : ℕ :=
  area_one_bedroom - windowArea

def total_paintable_area : ℕ :=
  bedrooms * paintable_area_one_bedroom

theorem jackson_paintable_area :
  total_paintable_area = 1520 :=
sorry

end PaintWallCalculation

end jackson_paintable_area_l308_308706


namespace K_travel_time_40_miles_l308_308765

noncomputable def K_time (x : ℝ) : ℝ := 40 / x

theorem K_travel_time_40_miles (x : ℝ) (d : ℝ) (Δt : ℝ)
  (h1 : d = 40)
  (h2 : Δt = 1 / 3)
  (h3 : ∃ (Kmiles_r : ℝ) (Mmiles_r : ℝ), Kmiles_r = x ∧ Mmiles_r = x - 0.5)
  (h4 : ∃ (Ktime : ℝ) (Mtime : ℝ), Ktime = d / x ∧ Mtime = d / (x - 0.5) ∧ Mtime - Ktime = Δt) :
  K_time x = 5 := sorry

end K_travel_time_40_miles_l308_308765


namespace rosie_pies_proof_l308_308285

-- Define the given condition
def pies_per_apples (p: ℕ) (a: ℕ) : ℕ := a / p

-- Given that Rosie can make 3 pies from 12 apples
def given_condition : pies_per_apples 3 12 = 4 := rfl

-- The proof problem statement:
theorem rosie_pies_proof : ∀ (a: ℕ) (n: ℕ), pies_per_apples 3 12 = 4 → pies_per_apples n a = 4 → pies_per_apples n a = 9 :=
begin
  sorry
end

end rosie_pies_proof_l308_308285


namespace tag_sum_is_large_l308_308785

noncomputable def tag_sum : ℝ :=
    let W : ℝ := 200
    let X : ℝ := (2/3) * W
    let Y : ℝ := W + X
    let Z : ℝ := real.sqrt Y
    let P : ℝ := X^3
    let Q : ℝ := nat.factorial W.to_nat / 100000
    W + X + Y + Z + P + Q

theorem tag_sum_is_large :
  let Q : ℝ := nat.factorial 200 / 100000 in
  tag_sum ≈ Q := by
    sorry

end tag_sum_is_large_l308_308785


namespace tricycle_count_l308_308474

variables (b t : ℕ)

theorem tricycle_count :
  b + t = 7 ∧ 2 * b + 3 * t = 19 → t = 5 := by
  intro h
  sorry

end tricycle_count_l308_308474


namespace total_sleep_time_is_correct_l308_308334

-- Define the sleeping patterns of the animals
def cougar_sleep_even_days : ℕ := 4
def cougar_sleep_odd_days : ℕ := 6
def zebra_sleep_more : ℕ := 2

-- Define the distribution of even and odd days in a week
def even_days_in_week : ℕ := 3
def odd_days_in_week : ℕ := 4

-- Define the total weekly sleep time for the cougar
def cougar_total_weekly_sleep : ℕ := 
  (cougar_sleep_even_days * even_days_in_week) + 
  (cougar_sleep_odd_days * odd_days_in_week)

-- Define the total weekly sleep time for the zebra
def zebra_total_weekly_sleep : ℕ := 
  ((cougar_sleep_even_days + zebra_sleep_more) * even_days_in_week) + 
  ((cougar_sleep_odd_days + zebra_sleep_more) * odd_days_in_week)

-- Define the total weekly sleep time for both the cougar and the zebra
def total_weekly_sleep : ℕ := 
  cougar_total_weekly_sleep + zebra_total_weekly_sleep

-- Prove that the total weekly sleep time for both animals is 86 hours
theorem total_sleep_time_is_correct : total_weekly_sleep = 86 :=
by
  -- skipping proof
  sorry

end total_sleep_time_is_correct_l308_308334


namespace exponentiation_problem_l308_308824

variable (x : ℝ) (m n : ℝ)

theorem exponentiation_problem (h1 : x ^ m = 5) (h2 : x ^ n = 1 / 4) :
  x ^ (2 * m - n) = 100 :=
sorry

end exponentiation_problem_l308_308824


namespace coin_difference_l308_308570

-- Define the coin denominations
def coin_denominations : List ℕ := [5, 10, 25, 50]

-- Define the target amount Paul needs to pay
def target_amount : ℕ := 60

-- Define the function to compute the minimum number of coins required
noncomputable def min_coins (target : ℕ) (denominations : List ℕ) : ℕ :=
  sorry -- Implementation of the function is not essential for this statement

-- Define the function to compute the maximum number of coins required
noncomputable def max_coins (target : ℕ) (denominations : List ℕ) : ℕ :=
  sorry -- Implementation of the function is not essential for this statement

-- Define the theorem to state the difference between max and min coins is 10
theorem coin_difference : max_coins target_amount coin_denominations - min_coins target_amount coin_denominations = 10 :=
  sorry

end coin_difference_l308_308570


namespace sarees_original_price_l308_308872

theorem sarees_original_price (P : ℝ) (h : 0.75 * 0.85 * P = 248.625) : P = 390 :=
by
  sorry

end sarees_original_price_l308_308872


namespace calculate_x_value_l308_308490

theorem calculate_x_value : 
  529 + 2 * 23 * 3 + 9 = 676 := 
by
  sorry

end calculate_x_value_l308_308490


namespace probability_exactly_two_sunny_days_l308_308803

-- Define the conditions
def rain_probability : ℝ := 0.8
def sun_probability : ℝ := 1 - rain_probability
def days : ℕ := 5
def sunny_days : ℕ := 2
def rainy_days : ℕ := days - sunny_days

-- Define the combinatorial and probability calculations
def comb (n k : ℕ) : ℕ := Nat.choose n k
def probability_sunny_days : ℝ := comb days sunny_days * (sun_probability ^ sunny_days) * (rain_probability ^ rainy_days)

theorem probability_exactly_two_sunny_days : probability_sunny_days = 51 / 250 := by
  sorry

end probability_exactly_two_sunny_days_l308_308803


namespace evaluate_expression_l308_308888

theorem evaluate_expression : 4 * (8 - 3) - 7 = 13 := by
  sorry

end evaluate_expression_l308_308888


namespace day_of_week_proof_l308_308100

/-- 
January 1, 1978, is a Sunday in the Gregorian calendar.
What day of the week is January 1, 2000, in the Gregorian calendar?
-/
def day_of_week_2000 := "Saturday"

theorem day_of_week_proof :
  let initial_year := 1978
  let target_year := 2000
  let initial_weekday := "Sunday"
  let years_between := target_year - initial_year -- 22 years
  let normal_days := years_between * 365 -- Normal days in these years
  let leap_years := 5 -- Number of leap years in the range
  let total_days := normal_days + leap_years -- Total days considering leap years
  let remainder_days := total_days % 7 -- days modulo 7
  initial_weekday = "Sunday" → remainder_days = 6 → 
  day_of_week_2000 = "Saturday" :=
by
  sorry

end day_of_week_proof_l308_308100


namespace investment_A_l308_308614

-- Define constants B and C's investment values, C's share, and total profit.
def B_investment : ℕ := 8000
def C_investment : ℕ := 9000
def C_share : ℕ := 36000
def total_profit : ℕ := 88000

-- Problem statement to prove
theorem investment_A (A_investment : ℕ) : 
  (A_investment + B_investment + C_investment = 17000) → 
  (C_investment * total_profit = C_share * (A_investment + B_investment + C_investment)) →
  A_investment = 5000 :=
by 
  intros h1 h2
  sorry

end investment_A_l308_308614


namespace count_three_digit_integers_with_7_units_place_div_by_21_l308_308678

theorem count_three_digit_integers_with_7_units_place_div_by_21 :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ n % 10 = 7 ∧ n % 21 = 0}.card = 22 :=
by
  sorry

end count_three_digit_integers_with_7_units_place_div_by_21_l308_308678


namespace correct_transformation_l308_308620

theorem correct_transformation (x : ℝ) : (x^2 - 10 * x - 1 = 0) → ((x - 5) ^ 2 = 26) := by
  sorry

end correct_transformation_l308_308620


namespace interest_rate_is_12_percent_l308_308170

-- Definitions
def SI : ℝ := 5400
def P : ℝ := 15000
def T : ℝ := 3

-- Theorem to prove the interest rate
theorem interest_rate_is_12_percent :
  SI = (P * 12 * T) / 100 :=
by
  sorry

end interest_rate_is_12_percent_l308_308170


namespace drawing_red_ball_is_certain_l308_308158

def certain_event (balls : List String) : Prop :=
  ∀ ball ∈ balls, ball = "red"

theorem drawing_red_ball_is_certain:
  certain_event ["red", "red", "red", "red", "red"] :=
by
  sorry

end drawing_red_ball_is_certain_l308_308158


namespace problem_statement_l308_308065

variable (f : ℕ → ℝ)

theorem problem_statement (hf : ∀ k : ℕ, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2)
  (h : f 4 = 25) : ∀ k : ℕ, k ≥ 4 → f k ≥ k^2 := 
by
  sorry

end problem_statement_l308_308065


namespace digging_foundation_l308_308705

-- Define given conditions
variable (m1 d1 m2 d2 k : ℝ)
variable (md_proportionality : m1 * d1 = k)
variable (k_value : k = 20 * 6)

-- Prove that for 30 men, it takes 4 days to dig the foundation
theorem digging_foundation : m1 = 20 ∧ d1 = 6 ∧ m2 = 30 → d2 = 4 :=
by
  sorry

end digging_foundation_l308_308705


namespace color_plane_with_two_colors_l308_308167

/-- Given a finite set of circles that divides the plane into regions, we can color the plane such that no two adjacent regions have the same color. -/
theorem color_plane_with_two_colors (circles : Finset (Set ℝ)) :
  (∀ (r1 r2 : Set ℝ), (r1 ∩ r2).Nonempty → ∃ (coloring : Set ℝ → Bool), (coloring r1 ≠ coloring r2)) :=
  sorry

end color_plane_with_two_colors_l308_308167


namespace find_smallest_a_l308_308132
open Real

noncomputable def a_min := 2 / 9

theorem find_smallest_a (a b c : ℝ)
  (h1 : (1/4, -9/8) = (1/4, a * (1/4) * (1/4) - 9/8))
  (h2 : ∃ n : ℤ, a + b + c = n)
  (h3 : a > 0)
  (h4 : b = - a / 2)
  (h5 : c = a / 16 - 9 / 8): 
  a = a_min :=
by {
  -- Lean code equivalent to the provided mathematical proof will be placed here.
  sorry
}

end find_smallest_a_l308_308132


namespace valid_parameterizations_l308_308589

def point_on_line (x y : ℝ) : Prop := (y = 2 * x - 5)

def direction_vector_valid (vx vy : ℝ) : Prop := (∃ (k : ℝ), vx = k * 1 ∧ vy = k * 2)

def parametric_option_valid (px py vx vy : ℝ) : Prop := 
  point_on_line px py ∧ direction_vector_valid vx vy

theorem valid_parameterizations : 
  (parametric_option_valid 10 15 5 10) ∧ 
  (parametric_option_valid 3 1 0.5 1) ∧ 
  (parametric_option_valid 7 9 2 4) ∧ 
  (parametric_option_valid 0 (-5) 10 20) :=
  by sorry

end valid_parameterizations_l308_308589


namespace number_of_juniors_l308_308393

theorem number_of_juniors (total_students j_percentage s_percentage : ℚ) (debate_team_ratio : ℚ):
  total_students = 40 →
  j_percentage = 1/5 →
  s_percentage = 1/4 →
  debate_team_ratio = 2 →
  ∃ J S, J + S = total_students ∧ S = debate_team_ratio * j_percentage * J / s_percentage ∧ J = 11 :=
by 
  intros h1 h2 h3 h4
  use 11
  use 18
  split
  exact h1
  split
  calc 18 = (2 : ℚ) * (1 / 5) * 11 / (1 / 4) : by 
    rw [h2, h3, h4]
    ring
  exact rfl
  exact rfl

end number_of_juniors_l308_308393


namespace earning_hours_per_week_l308_308248

theorem earning_hours_per_week (totalEarnings : ℝ) (originalWeeks : ℝ) (missedWeeks : ℝ) 
  (originalHoursPerWeek : ℝ) : 
  missedWeeks = 3 → originalWeeks = 15 → originalHoursPerWeek = 25 → totalEarnings = 3750 → 
  (totalEarnings / ((totalEarnings / (originalWeeks * originalHoursPerWeek)) * (originalWeeks - missedWeeks))) = 31.25 :=
by
  intros
  sorry

end earning_hours_per_week_l308_308248


namespace saree_blue_stripes_l308_308311

theorem saree_blue_stripes :
  ∀ (brown_stripes gold_stripes blue_stripes : ℕ),
    gold_stripes = 3 * brown_stripes →
    blue_stripes = 5 * gold_stripes →
    brown_stripes = 4 →
    blue_stripes = 60 :=
by
  intros brown_stripes gold_stripes blue_stripes h_gold h_blue h_brown
  sorry

end saree_blue_stripes_l308_308311


namespace bobs_highest_success_ratio_l308_308175

def alice_first_day_success_ratio : ℚ := 220 / 400
def alice_second_day_success_ratio : ℚ := 180 / 200
def alice_two_day_total_attempt : ℚ := 600
def alice_two_day_success_ratio : ℚ := 2 / 3

theorem bobs_highest_success_ratio (x y z w : ℕ) 
  (h1 : 0 < x ∧ 0 < z) 
  (h2 : 0 < x / y ∧ x / y < alice_first_day_success_ratio) 
  (h3 : 0 < z / w ∧ z / w < alice_second_day_success_ratio)
  (h4 : y + w = 600) 
  : (x + z) / 600 ≤ 22 / 75 :=
by sorry

end bobs_highest_success_ratio_l308_308175


namespace floor_e_eq_two_l308_308216

theorem floor_e_eq_two
  (e_approx : Real ≈ 2.718) :
  ⌊e⌋ = 2 :=
sorry

end floor_e_eq_two_l308_308216


namespace eval_expression_l308_308895

theorem eval_expression : 4 * (8 - 3) - 7 = 13 :=
by
  sorry

end eval_expression_l308_308895


namespace original_radius_of_cylinder_l308_308542

theorem original_radius_of_cylinder (r z : ℝ) (h : ℝ := 3) :
  z = 3 * π * ((r + 8)^2 - r^2) → z = 8 * π * r^2 → r = 8 :=
by
  intros hz1 hz2
  -- Translate given conditions into their equivalent expressions and equations
  sorry

end original_radius_of_cylinder_l308_308542


namespace kyle_speed_l308_308841

theorem kyle_speed (S : ℝ) (joseph_speed : ℝ) (joseph_time : ℝ) (kyle_time : ℝ) (H1 : joseph_speed = 50) (H2 : joseph_time = 2.5) (H3 : kyle_time = 2) (H4 : joseph_speed * joseph_time = kyle_time * S + 1) : S = 62 :=
by
  sorry

end kyle_speed_l308_308841


namespace probability_is_five_eleven_l308_308985

-- Define the total number of cards
def total_cards : ℕ := 12

-- Define a function to calculate combinations
def comb (n k : ℕ) : ℕ := n.choose k

-- Define the number of favorable outcomes for same letter and same color
def favorable_same_letter : ℕ := 4 * comb 3 2
def favorable_same_color : ℕ := 3 * comb 4 2

-- Total number of favorable outcomes
def total_favorable : ℕ := favorable_same_letter + favorable_same_color

-- Total number of ways to draw 2 cards from 12
def total_ways : ℕ := comb total_cards 2

-- Probability of drawing a winning pair
def probability_winning_pair : ℚ := total_favorable / total_ways

theorem probability_is_five_eleven : probability_winning_pair = 5 / 11 :=
by
  sorry

end probability_is_five_eleven_l308_308985


namespace lines_from_equation_l308_308204

-- Definitions for the conditions
def satisfies_equation (x y : ℝ) : Prop :=
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Equivalent Lean statement to the proof problem
theorem lines_from_equation :
  (∀ x y : ℝ, satisfies_equation x y → (y = -x - 2) ∨ (y = -2 * x + 1)) :=
by
  intros x y h
  sorry

end lines_from_equation_l308_308204


namespace find_alpha_l308_308823

theorem find_alpha (α : ℝ) (h : Real.sin α * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)) = 1) :
  α = 13 * Real.pi / 18 :=
sorry

end find_alpha_l308_308823


namespace product_not_zero_l308_308790

theorem product_not_zero (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 5) : (x - 2) * (x - 5) ≠ 0 := 
by 
  sorry

end product_not_zero_l308_308790


namespace percentage_reduction_l308_308780

theorem percentage_reduction 
  (original_employees : ℝ)
  (new_employees : ℝ)
  (h1 : original_employees = 208.04597701149424)
  (h2 : new_employees = 181) :
  ((original_employees - new_employees) / original_employees) * 100 = 13.00 :=
by
  sorry

end percentage_reduction_l308_308780


namespace zane_total_payment_l308_308461

open Real

noncomputable def shirt1_price := 50.0
noncomputable def shirt2_price := 50.0
noncomputable def discount1 := 0.4 * shirt1_price
noncomputable def discount2 := 0.3 * shirt2_price
noncomputable def price1_after_discount := shirt1_price - discount1
noncomputable def price2_after_discount := shirt2_price - discount2
noncomputable def total_before_tax := price1_after_discount + price2_after_discount
noncomputable def sales_tax := 0.08 * total_before_tax
noncomputable def total_cost := total_before_tax + sales_tax

-- We want to prove:
theorem zane_total_payment : total_cost = 70.20 := by sorry

end zane_total_payment_l308_308461


namespace opposite_of_neg_quarter_l308_308439

theorem opposite_of_neg_quarter : -(- (1 / 4)) = 1 / 4 :=
by
  sorry

end opposite_of_neg_quarter_l308_308439


namespace gcd_372_684_l308_308588

theorem gcd_372_684 : Int.gcd 372 684 = 12 :=
by
  sorry

end gcd_372_684_l308_308588


namespace find_xyz_l308_308690

theorem find_xyz (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
    (h4 : x * y = 16 * real.cbrt 4)
    (h5 : x * z = 28 * real.cbrt 4)
    (h6 : y * z = 112 / real.cbrt 4) :
    x * y * z = 112 * real.sqrt 7 := 
sorry

end find_xyz_l308_308690


namespace equal_areas_triangle_height_l308_308863

theorem equal_areas_triangle_height (l b h : ℝ) (hlb : l > b) 
  (H1 : l * b = (1/2) * l * h) : h = 2 * b :=
by 
  -- skipping proof
  sorry

end equal_areas_triangle_height_l308_308863


namespace pies_from_36_apples_l308_308284

-- Definitions of conditions
def pies_from_apples (apples : Nat) : Nat :=
  apples / 4  -- because 12 apples = 3 pies implies 1 pie = 4 apples

-- Theorem to prove
theorem pies_from_36_apples : pies_from_apples 36 = 9 := by
  sorry

end pies_from_36_apples_l308_308284


namespace grooming_time_correct_l308_308012

def time_to_groom_poodle : ℕ := 30
def time_to_groom_terrier : ℕ := time_to_groom_poodle / 2
def num_poodles : ℕ := 3
def num_terriers : ℕ := 8

def total_grooming_time : ℕ := 
  (num_poodles * time_to_groom_poodle) + (num_terriers * time_to_groom_terrier)

theorem grooming_time_correct : 
  total_grooming_time = 210 := by
  sorry

end grooming_time_correct_l308_308012


namespace vertex_coloring_exists_l308_308037

-- Define the graph with given conditions
variables (V : Type) [Fintype V] [DecidableEq V]
variables (G : SimpleGraph V)
variables (h_card : Fintype.card V = 2004)
variables (h_deg : ∀ v : V, G.degree v ≤ 5)

-- The theorem statement
theorem vertex_coloring_exists :
  ∃ (A B : Finset V), A ∩ B = ∅ ∧ A ∪ B = Finset.univ ∧ (∃ E_cross, E_cross ⊆ G.edgeSet ∧ |E_cross| ≥ 3 / 5 * |G.edgeSet| ∧ ∀ e ∈ E_cross, (e.1 ∈ A ∧ e.2 ∈ B) ∨ (e.1 ∈ B ∧ e.2 ∈ A)) :=
sorry

end vertex_coloring_exists_l308_308037


namespace floor_e_equals_two_l308_308207

/-- Prove that the floor of Euler's number is 2. -/
theorem floor_e_equals_two : (⌊Real.exp 1⌋ = 2) :=
sorry

end floor_e_equals_two_l308_308207


namespace center_of_circle_l308_308738

theorem center_of_circle :
  ∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 → (x, y) = (1, 1) :=
by
  sorry

end center_of_circle_l308_308738


namespace congruent_semicircles_ratio_l308_308536

theorem congruent_semicircles_ratio (N : ℕ) (r : ℝ) (hN : N > 0) 
    (A : ℝ) (B : ℝ) (hA : A = (N * π * r^2) / 2)
    (hB : B = (π * N^2 * r^2) / 2 - (N * π * r^2) / 2)
    (h_ratio : A / B = 1 / 9) : 
    N = 10 :=
by
  -- The proof will be filled in here.
  sorry

end congruent_semicircles_ratio_l308_308536


namespace least_sum_of_variables_l308_308761

theorem least_sum_of_variables (x y z w : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0)  
  (h : 2 * x^2 = 5 * y^3 ∧ 5 * y^3 = 8 * z^4 ∧ 8 * z^4 = 3 * w) : x + y + z + w = 54 := 
sorry

end least_sum_of_variables_l308_308761


namespace marion_score_is_correct_l308_308869

-- Definition of the problem conditions
def exam_total_items := 40
def ella_incorrect_answers := 4

-- Calculate Ella's score
def ella_score := exam_total_items - ella_incorrect_answers

-- Calculate half of Ella's score
def half_ella_score := ella_score / 2

-- Marion's score is 6 more than half of Ella's score
def marion_score := half_ella_score + 6

-- The theorem we need to prove
theorem marion_score_is_correct : marion_score = 24 := by
  sorry

end marion_score_is_correct_l308_308869


namespace solve_for_x_l308_308046

def f (x : ℝ) : ℝ := 3 * x - 4

noncomputable def f_inv (x : ℝ) : ℝ := (x + 4) / 3

theorem solve_for_x : ∃ x : ℝ, f x = f_inv x ∧ x = 2 := by
  sorry

end solve_for_x_l308_308046


namespace complex_distance_l308_308521

theorem complex_distance (i : Complex) (h : i = Complex.I) :
  Complex.abs (3 / (2 - i)^2) = 3 / 5 := 
by
  sorry

end complex_distance_l308_308521


namespace reduced_price_per_dozen_is_approx_2_95_l308_308759

noncomputable def original_price : ℚ := 16 / 39
noncomputable def reduced_price := 0.6 * original_price
noncomputable def reduced_price_per_dozen := reduced_price * 12

theorem reduced_price_per_dozen_is_approx_2_95 :
  abs (reduced_price_per_dozen - 2.95) < 0.01 :=
by
  sorry

end reduced_price_per_dozen_is_approx_2_95_l308_308759


namespace airplane_seat_difference_l308_308043

theorem airplane_seat_difference (F C X : ℕ) 
    (h1 : 387 = F + 310) 
    (h2 : C = 310) 
    (h3 : C = 4 * F + X) :
    X = 2 :=
by
    sorry

end airplane_seat_difference_l308_308043


namespace find_apron_cost_l308_308051

-- Definitions used in the conditions
variables (hand_mitts cost small_knife utensils apron : ℝ)
variables (nieces : ℕ)
variables (total_cost_before_discount total_cost_after_discount : ℝ)

-- Conditions given
def conditions := 
  hand_mitts = 14 ∧ 
  utensils = 10 ∧ 
  small_knife = 2 * utensils ∧
  (total_cost_before_discount : ℝ) = (3 * hand_mitts + 3 * utensils + 3 * small_knife + 3 * apron) ∧
  (total_cost_after_discount : ℝ) = 135 ∧
  total_cost_before_discount * 0.75 = total_cost_after_discount ∧
  nieces = 3

-- Theorem statement (proof problem)
theorem find_apron_cost (h : conditions hand_mitts utensils small_knife apron nieces total_cost_before_discount total_cost_after_discount) : 
  apron = 16 :=
by 
  sorry

end find_apron_cost_l308_308051


namespace remainder_8_pow_900_mod_29_l308_308657

theorem remainder_8_pow_900_mod_29 : 8^900 % 29 = 7 :=
by sorry

end remainder_8_pow_900_mod_29_l308_308657


namespace evaluate_expression_l308_308425

noncomputable def g : ℕ → ℕ := sorry
noncomputable def g_inv : ℕ → ℕ := sorry

axiom g_inverse : ∀ x, g (g_inv x) = x ∧ g_inv (g x) = x

axiom g_1_2 : g 1 = 2
axiom g_4_7 : g 4 = 7
axiom g_3_8 : g 3 = 8

theorem evaluate_expression :
  g_inv (g_inv 8 * g_inv 2) = 3 :=
by
  sorry

end evaluate_expression_l308_308425


namespace hillary_climbing_rate_l308_308974

theorem hillary_climbing_rate :
  ∀ (H : ℕ) (Eddy_rate : ℕ) (Hillary_climb : ℕ) (Hillary_descend_rate : ℕ) (pass_time : ℕ) (start_to_summit : ℕ),
    Eddy_rate = 500 →
    Hillary_climb = 4000 →
    Hillary_descend_rate = 1000 →
    pass_time = 6 →
    start_to_summit = 5000 →
    (Hillary_climb + Eddy_rate * pass_time = Hillary_climb + (pass_time - Hillary_climb / H) * Hillary_descend_rate) →
    H = 800 :=
by
  intros H Eddy_rate Hillary_climb Hillary_descend_rate pass_time start_to_summit
  intro h1 h2 h3 h4 h5 h6
  sorry

end hillary_climbing_rate_l308_308974


namespace cost_price_of_watch_l308_308783

variable (CP : ℝ)
variable (SP_loss SP_gain : ℝ)
variable (h1 : SP_loss = CP * 0.725)
variable (h2 : SP_gain = CP * 1.125)
variable (h3 : SP_gain - SP_loss = 275)

theorem cost_price_of_watch : CP = 687.50 :=
by
  sorry

end cost_price_of_watch_l308_308783


namespace find_function_l308_308563

theorem find_function (f : ℝ → ℝ) :
  (∀ u v : ℝ, f (2 * u) = f (u + v) * f (v - u) + f (u - v) * f (-u - v)) →
  (∀ u : ℝ, 0 ≤ f u) →
  (∀ x : ℝ, f x = 0) := 
  by
    sorry

end find_function_l308_308563


namespace measure_time_with_hourglasses_l308_308153

def hourglass7 : ℕ := 7
def hourglass11 : ℕ := 11
def target_time : ℕ := 15

theorem measure_time_with_hourglasses :
  ∃ (time_elapsed : ℕ), time_elapsed = target_time :=
by
  use 15
  sorry

end measure_time_with_hourglasses_l308_308153


namespace balloons_difference_l308_308042

-- Define the balloons each person brought
def Allan_red := 150
def Allan_blue_total := 75
def Allan_forgotten_blue := 25
def Allan_green := 30

def Jake_red := 100
def Jake_blue := 50
def Jake_green := 45

-- Calculate the actual balloons Allan brought to the park
def Allan_blue := Allan_blue_total - Allan_forgotten_blue
def Allan_total := Allan_red + Allan_blue + Allan_green

-- Calculate the total number of balloons Jake brought
def Jake_total := Jake_red + Jake_blue + Jake_green

-- State the problem: Prove Allan distributed 35 more balloons than Jake
theorem balloons_difference : Allan_total - Jake_total = 35 := 
by
  sorry

end balloons_difference_l308_308042


namespace value_of_expression_l308_308018

theorem value_of_expression : (2 + 4 + 6) - (1 + 3 + 5) = 3 := 
by 
  sorry

end value_of_expression_l308_308018


namespace additional_spending_required_l308_308449

def cost_of_chicken : ℝ := 1.5 * 6.00
def cost_of_lettuce : ℝ := 3.00
def cost_of_cherry_tomatoes : ℝ := 2.50
def cost_of_sweet_potatoes : ℝ := 4 * 0.75
def cost_of_broccoli : ℝ := 2 * 2.00
def cost_of_brussel_sprouts : ℝ := 2.50
def total_cost : ℝ := cost_of_chicken + cost_of_lettuce + cost_of_cherry_tomatoes + cost_of_sweet_potatoes + cost_of_broccoli + cost_of_brussel_sprouts
def minimum_spending_for_free_delivery : ℝ := 35.00
def additional_amount_needed : ℝ := minimum_spending_for_free_delivery - total_cost

theorem additional_spending_required : additional_amount_needed = 11.00 := by
  sorry

end additional_spending_required_l308_308449


namespace highest_throw_is_37_feet_l308_308349

theorem highest_throw_is_37_feet :
  let C1 := 20
  let J1 := C1 - 4
  let C2 := C1 + 10
  let J2 := J1 * 2
  let C3 := C2 + 4
  let J3 := C1 + 17
  max (max C1 (max C2 C3)) (max J1 (max J2 J3)) = 37 := by
  let C1 := 20
  let J1 := C1 - 4
  let C2 := C1 + 10
  let J2 := J1 * 2
  let C3 := C2 + 4
  let J3 := C1 + 17
  sorry

end highest_throw_is_37_feet_l308_308349


namespace how_many_rocks_l308_308838

section see_saw_problem

-- Conditions
def Jack_weight : ℝ := 60
def Anna_weight : ℝ := 40
def rock_weight : ℝ := 4

-- Theorem statement
theorem how_many_rocks : (Jack_weight - Anna_weight) / rock_weight = 5 :=
by
  -- Proof is omitted, just ensuring the theorem statement
  sorry

end see_saw_problem

end how_many_rocks_l308_308838


namespace all_buses_have_same_stoppage_time_l308_308950

-- Define the constants for speeds without and with stoppages
def speed_without_stoppage_bus1 := 50
def speed_without_stoppage_bus2 := 60
def speed_without_stoppage_bus3 := 70

def speed_with_stoppage_bus1 := 40
def speed_with_stoppage_bus2 := 48
def speed_with_stoppage_bus3 := 56

-- Stating the stoppage time per hour for each bus
def stoppage_time_per_hour (speed_without : ℕ) (speed_with : ℕ) : ℚ :=
  1 - (speed_with : ℚ) / (speed_without : ℚ)

-- Theorem to prove the stoppage time correctness
theorem all_buses_have_same_stoppage_time :
  stoppage_time_per_hour speed_without_stoppage_bus1 speed_with_stoppage_bus1 = 0.2 ∧
  stoppage_time_per_hour speed_without_stoppage_bus2 speed_with_stoppage_bus2 = 0.2 ∧
  stoppage_time_per_hour speed_without_stoppage_bus3 speed_with_stoppage_bus3 = 0.2 :=
by
  sorry  -- Proof to be completed

end all_buses_have_same_stoppage_time_l308_308950


namespace gloria_initial_dimes_l308_308819

variable (Q D : ℕ)

theorem gloria_initial_dimes (h1 : D = 5 * Q) 
                             (h2 : (3 * Q) / 5 + D = 392) : 
                             D = 350 := 
by {
  sorry
}

end gloria_initial_dimes_l308_308819


namespace circle_incircle_tangent_radius_l308_308543

theorem circle_incircle_tangent_radius (r1 r2 r3 : ℕ) (k : ℕ) (h1 : r1 = 1) (h2 : r2 = 4) (h3 : r3 = 9) : 
  k = 11 :=
by
  -- Definitions according to the problem
  let k₁ := r1
  let k₂ := r2
  let k₃ := r3
  -- Hypotheses given by the problem
  have h₁ : k₁ = 1 := h1
  have h₂ : k₂ = 4 := h2
  have h₃ : k₃ = 9 := h3
  -- Prove the radius of the incircle k
  sorry

end circle_incircle_tangent_radius_l308_308543


namespace range_of_k_l308_308006

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  ((k+1)*x^2 + (k+3)*x + (2*k-8)) / ((2*k-1)*x^2 + (k+1)*x + (k-4))

theorem range_of_k 
  (k : ℝ) 
  (hk1 : k ≠ -1)
  (hk2 : (k+3)^2 - 4*(k+1)*(2*k-8) ≥ 0)
  (hk3 : (k+1)^2 - 4*(2*k-1)*(k-4) ≤ 0)
  (hk4 : (k+1)/(2*k-1) > 0) :
  k ∈ Set.Iio (-1) ∪ Set.Ioi (1 / 2) ∩ Set.Iic (41 / 7) := 
  sorry

end range_of_k_l308_308006


namespace floor_e_eq_2_l308_308210

theorem floor_e_eq_2 : ⌊Real.exp 1⌋ = 2 := by
  sorry

end floor_e_eq_2_l308_308210


namespace combined_age_in_years_l308_308041

theorem combined_age_in_years (years : ℕ) (adam_age : ℕ) (tom_age : ℕ) (target_age : ℕ) :
  adam_age = 8 → tom_age = 12 → target_age = 44 → (adam_age + tom_age) + 2 * years = target_age → years = 12 :=
by
  intros h_adam h_tom h_target h_combined
  rw [h_adam, h_tom, h_target] at h_combined
  linarith

end combined_age_in_years_l308_308041


namespace evaluate_expression_l308_308891

theorem evaluate_expression : 4 * (8 - 3) - 7 = 13 :=
by sorry

end evaluate_expression_l308_308891


namespace fat_content_whole_milk_l308_308033

open Real

theorem fat_content_whole_milk :
  ∃ (s w : ℝ), 0 < s ∧ 0 < w ∧
  3 / 100 = 0.75 * s / 100 ∧
  s / 100 = 0.8 * w / 100 ∧
  w = 5 :=
by
  sorry

end fat_content_whole_milk_l308_308033


namespace geometric_sequence_from_second_term_l308_308397

theorem geometric_sequence_from_second_term (S : ℕ → ℕ) (a : ℕ → ℕ) :
  S 1 = 1 ∧ S 2 = 2 ∧ (∀ n, n ≥ 2 → S (n + 1) - 3 * S n + 2 * S (n - 1) = 0) →
  (∀ n, n ≥ 2 → a (n + 1) = 2 * a n) :=
by
  sorry

end geometric_sequence_from_second_term_l308_308397


namespace system_of_equations_a_solution_l308_308534

theorem system_of_equations_a_solution (x y a : ℝ) (h1 : 4 * x + y = a) (h2 : 3 * x + 4 * y^2 = 3 * a) (hx : x = 3) : a = 15 ∨ a = 9.75 :=
by
  sorry

end system_of_equations_a_solution_l308_308534


namespace max_profit_price_l308_308899

-- Define the initial conditions
def purchase_price : ℝ := 80
def initial_selling_price : ℝ := 90
def initial_sales_volume : ℝ := 400
def price_increase_effect : ℝ := 1
def sales_volume_decrease : ℝ := 20

-- Define the profit function
def profit (x : ℝ) : ℝ :=
  let selling_price := initial_selling_price + x
  let sales_volume := initial_sales_volume - x * sales_volume_decrease
  let profit_per_item := selling_price - purchase_price
  profit_per_item * sales_volume

-- The statement that needs to be proved
theorem max_profit_price : ∃ x : ℝ, x = 10 ∧ (initial_selling_price + x = 100) := by
  sorry

end max_profit_price_l308_308899


namespace area_constant_k_l308_308016

theorem area_constant_k (l w d : ℝ) (h_ratio : l / w = 5 / 2) (h_diagonal : d = Real.sqrt (l^2 + w^2)) :
  ∃ k : ℝ, (k = 10 / 29) ∧ (l * w = k * d^2) :=
by
  sorry

end area_constant_k_l308_308016


namespace fraction_product_simplification_l308_308184

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 3 / 7 := 
by
  sorry

end fraction_product_simplification_l308_308184


namespace max_sum_a_b_c_l308_308966

noncomputable def f (a b c x : ℝ) : ℝ := a * Real.cos x + b * Real.cos (2 * x) + c * Real.cos (3 * x)

theorem max_sum_a_b_c (a b c : ℝ) (h : ∀ x : ℝ, f a b c x ≥ -1) : a + b + c ≤ 3 :=
sorry

end max_sum_a_b_c_l308_308966


namespace shape_is_cylinder_l308_308660

noncomputable def shape_desc (r θ z a : ℝ) : Prop := r = a

theorem shape_is_cylinder (a : ℝ) (h_a : a > 0) :
  ∀ (r θ z : ℝ), shape_desc r θ z a → ∃ c : Set (ℝ × ℝ × ℝ), c = {p : ℝ × ℝ × ℝ | ∃ θ z, p = (a, θ, z)} :=
by
  sorry

end shape_is_cylinder_l308_308660


namespace complex_number_property_l308_308088

theorem complex_number_property (i : ℂ) (h : i^2 = -1) : (1 + i)^(20) - (1 - i)^(20) = 0 :=
by {
  sorry
}

end complex_number_property_l308_308088


namespace donation_percentage_l308_308776

noncomputable def income : ℝ := 266666.67
noncomputable def remaining_income : ℝ := 0.25 * income
noncomputable def final_amount : ℝ := 40000

theorem donation_percentage :
  ∃ D : ℝ, D = 40 /\ (1 - D / 100) * remaining_income = final_amount :=
by
  sorry

end donation_percentage_l308_308776


namespace no_valid_arrangement_l308_308933

open Nat

theorem no_valid_arrangement :
  ¬ ∃ (f : Fin 30 → ℕ), 
    (∀ (i : Fin 30), 1 ≤ f i ∧ f i ≤ 30) ∧ 
    (∀ (i : Fin 30), ∃ n : ℕ, (f i + f (i + 1) % 30) = n^2) ∧ 
    (∀ i1 i2, i1 ≠ i2 → f i1 ≠ f i2) :=
  sorry

end no_valid_arrangement_l308_308933


namespace trucks_sold_l308_308048

-- Definitions for conditions
def cars_and_trucks_total (T C : Nat) : Prop :=
  T + C = 69

def cars_more_than_trucks (T C : Nat) : Prop :=
  C = T + 27

-- Theorem statement
theorem trucks_sold (T C : Nat) (h1 : cars_and_trucks_total T C) (h2 : cars_more_than_trucks T C) : T = 21 :=
by
  -- This will be replaced by the proof
  sorry

end trucks_sold_l308_308048


namespace min_books_borrowed_l308_308254

theorem min_books_borrowed
  (total_students : ℕ)
  (students_no_books : ℕ)
  (students_one_book : ℕ)
  (students_two_books : ℕ)
  (avg_books_per_student : ℝ)
  (total_students_eq : total_students = 40)
  (students_no_books_eq : students_no_books = 2)
  (students_one_book_eq : students_one_book = 12)
  (students_two_books_eq : students_two_books = 13)
  (avg_books_per_student_eq : avg_books_per_student = 2) :
  ∀ min_books_borrowed : ℕ, 
    (total_students * avg_books_per_student = 80) → 
    (students_one_book * 1 + students_two_books * 2 ≤ 38) → 
    (total_students - students_no_books - students_one_book - students_two_books = 13) →
    min_books_borrowed * 13 = 42 → 
    min_books_borrowed = 4 :=
by
  intros min_books_borrowed total_books_eq books_count_eq remaining_students_eq total_min_books_eq
  sorry

end min_books_borrowed_l308_308254


namespace value_of_f_2_pow_100_l308_308996

def f : ℕ → ℕ :=
sorry

axiom f_base : f 1 = 1
axiom f_recursive : ∀ n : ℕ, f (2 * n) = n * f n

theorem value_of_f_2_pow_100 : f (2^100) = 2^4950 :=
sorry

end value_of_f_2_pow_100_l308_308996


namespace abs_diff_mn_sqrt_eight_l308_308113

theorem abs_diff_mn_sqrt_eight {m n p : ℝ} (h1 : m * n = 6) (h2 : m + n + p = 7) (h3 : p = 1) :
  |m - n| = 2 * Real.sqrt 3 :=
by
  sorry

end abs_diff_mn_sqrt_eight_l308_308113


namespace problem1_problem2_problem3_l308_308470

noncomputable def chi_squared_test (n a b c d α χ_squared_critical : ℝ) : Prop :=
  let χ_squared := n * (a * d - b * c) ^ 2 / ((a + b) * (c + d) * (a + c) * (b + d)) in
  χ_squared > χ_squared_critical

noncomputable def likelihood_ratio (nAB nA_notB : ℝ) : ℝ :=
  nAB / nA_notB

noncomputable def probability_distribution (C_combinations : ℕ → ℕ → ℚ) (C_8_3 : ℚ) :
  (ℚ × ℚ) × (ℚ × ℚ) × (ℚ × ℚ) × (ℚ × ℚ) :=
  let P_X0 := C_combinations 3 3 / C_8_3
  let P_X1 := C_combinations 5 1 * C_combinations 3 2 / C_8_3
  let P_X2 := C_combinations 5 2 * C_combinations 3 1 / C_8_3
  let P_X3 := C_combinations 5 3 / C_8_3
  ((0, P_X0), (1, P_X1), (2, P_X2), (3, P_X3))

noncomputable def expected_value (dist : (ℚ × ℚ) × (ℚ × ℚ) × (ℚ × ℚ) × (ℚ × ℚ)) : ℚ :=
  dist.fst.1 * dist.fst.2 +
  dist.snd.fst * dist.snd.snd +
  dist.snd.snd.fst * dist.snd.snd.snd +
  dist.fst.snd.snd.fst * dist.fst.snd.snd.snd

-- Declarations of constants based on given problem
def n : ℝ := 200
def a : ℝ := 50
def b : ℝ := 30
def c : ℝ := 40
def d : ℝ := 80
def α : ℝ := 0.010
def χ_squared_critical : ℝ := 6.635

def nAB : ℝ := 80
def nA_notB : ℝ := 30

def C_combinations (n k : ℕ) : ℚ := 
  (Finset.range n).card.choose k

def C_8_3 : ℚ := C_combinations 8 3

-- Proof problem statements
theorem problem1 : chi_squared_test n a b c d α χ_squared_critical := by
  sorry

theorem problem2 : likelihood_ratio nAB nA_notB = (8 / 3) := by
  sorry

theorem problem3 : expected_value (probability_distribution C_combinations C_8_3) = (15 / 8) := by
  sorry

end problem1_problem2_problem3_l308_308470


namespace cost_function_discrete_points_l308_308569

def cost (n : ℕ) : ℕ :=
  if n <= 10 then 20 * n
  else if n <= 25 then 18 * n
  else 0

theorem cost_function_discrete_points :
  (∀ n, 1 ≤ n ∧ n ≤ 25 → ∃ y, cost n = y) ∧
  (∀ m n, 1 ≤ m ∧ m ≤ 25 ∧ 1 ≤ n ∧ n ≤ 25 ∧ m ≠ n → cost m ≠ cost n) :=
sorry

end cost_function_discrete_points_l308_308569


namespace m_squared_n_minus_1_l308_308321

theorem m_squared_n_minus_1 (a b m n : ℝ)
  (h1 : a * m^2001 + b * n^2001 = 3)
  (h2 : a * m^2002 + b * n^2002 = 7)
  (h3 : a * m^2003 + b * n^2003 = 24)
  (h4 : a * m^2004 + b * n^2004 = 102) :
  m^2 * (n - 1) = 6 := by
  sorry

end m_squared_n_minus_1_l308_308321


namespace acme_vowel_soup_l308_308784

-- Define the set of vowels
def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U'}

-- Define the number of each vowel
def num_vowels (v : Char) : ℕ := 5

-- Define a function to count the number of five-letter words
def count_five_letter_words : ℕ :=
  (vowels.card) ^ 5

-- Theorem to be proven
theorem acme_vowel_soup :
  count_five_letter_words = 3125 :=
by
  -- Proof omitted
  sorry

end acme_vowel_soup_l308_308784


namespace cheyenne_clay_pots_l308_308188

theorem cheyenne_clay_pots (P : ℕ) (cracked_ratio sold_ratio : ℝ) (total_revenue price_per_pot : ℝ) 
    (P_sold : ℕ) :
  cracked_ratio = (2 / 5) →
  sold_ratio = (3 / 5) →
  total_revenue = 1920 →
  price_per_pot = 40 →
  P_sold = 48 →
  (sold_ratio * P = P_sold) →
  P = 80 :=
by
  sorry

end cheyenne_clay_pots_l308_308188


namespace factor_difference_of_squares_l308_308052

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end factor_difference_of_squares_l308_308052


namespace hyperbola_t_square_l308_308034

theorem hyperbola_t_square (t : ℝ)
  (h1 : ∃ a : ℝ, ∀ (x y : ℝ), (y^2 / 4) - (5 * x^2 / 64) = 1 ↔ ((x, y) = (2, t) ∨ (x, y) = (4, -3) ∨ (x, y) = (0, -2))) :
  t^2 = 21 / 4 :=
by
  -- We need to prove t² = 21/4 given the conditions
  sorry

end hyperbola_t_square_l308_308034


namespace range_of_a_value_of_a_l308_308997

-- Problem 1
theorem range_of_a (a : ℝ) :
  (∃ x, (2 < x ∧ x < 4) ∧ (a < x ∧ x < 3 * a)) ↔ (4 / 3 ≤ a ∧ a < 4) :=
sorry

-- Problem 2
theorem value_of_a (a : ℝ) :
  (∀ x, (2 < x ∧ x < 4) ∨ (a < x ∧ x < 3 * a) ↔ (2 < x ∧ x < 6)) ↔ (a = 2) :=
sorry

end range_of_a_value_of_a_l308_308997


namespace f_neg_m_equals_neg_8_l308_308814

def f (x : ℝ) : ℝ := x^5 + x^3 + 1

theorem f_neg_m_equals_neg_8 (m : ℝ) (h : f m = 10) : f (-m) = -8 :=
by
  sorry

end f_neg_m_equals_neg_8_l308_308814


namespace find_principal_sum_l308_308138

def compound_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r)^t - P

def simple_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * r * t

theorem find_principal_sum (CI SI : ℝ) (t : ℕ)
  (h1 : CI = 11730) 
  (h2 : SI = 10200) 
  (h3 : t = 2) :
  ∃ P r, P = 17000 ∧
  compound_interest P r t = CI ∧
  simple_interest P r t = SI :=
by
  sorry

end find_principal_sum_l308_308138


namespace return_amount_is_correct_l308_308106

-- Define the borrowed amount and the interest rate
def borrowed_amount : ℝ := 100
def interest_rate : ℝ := 10 / 100

-- Define the condition of the increased amount
def increased_amount : ℝ := borrowed_amount * interest_rate

-- Define the total amount to be returned
def total_amount : ℝ := borrowed_amount + increased_amount

-- Lean 4 statement to prove
theorem return_amount_is_correct : total_amount = 110 := by
  -- Borrowing amount definition
  have h1 : borrowed_amount = 100 := rfl
  -- Interest rate definition
  have h2 : interest_rate = 10 / 100 := rfl
  -- Increased amount calculation
  have h3 : increased_amount = borrowed_amount * interest_rate := rfl
  -- Expanded calculation of increased_amount
  have h4 : increased_amount = 100 * (10 / 100) := by rw [h1, h2]
  -- Simplify the increased_amount
  have h5 : increased_amount = 10 := by norm_num [h4]
  -- Total amount calculation
  have h6 : total_amount = borrowed_amount + increased_amount := rfl
  -- Expanded calculation of total_amount
  have h7 : total_amount = 100 + 10 := by rw [h1, h5]
  -- Simplify the total_amount
  show 100 + 10 = 110 from rfl
  sorry

end return_amount_is_correct_l308_308106


namespace product_multiple_of_4_probability_l308_308115

-- Condition representations
def maria_rolls : ℕ := 10
def karim_rolls : ℕ := 6

-- Event that the product of their rolls is a multiple of 4
def is_multiple_of_4 (x y : ℕ) : Prop := (x * y) % 4 = 0

-- Probability calculation as per conditions
def probability {α : Type} (S : Set α) (event : Set α) : ℚ :=
  (event.toFinite.card : ℚ) / (S.toFinite.card : ℚ)

def die_rolls (n : ℕ) : Set ℕ := {k | 1 ≤ k ∧ k ≤ n}

-- Define the desired probability
def desired_probability : ℚ := 11 / 30

-- Proof statement
theorem product_multiple_of_4_probability :
  probability (die_rolls maria_rolls ×ˢ die_rolls karim_rolls) {x | is_multiple_of_4 (x.fst) (x.snd)} = desired_probability :=
sorry

end product_multiple_of_4_probability_l308_308115


namespace solve_q_l308_308271

-- Definitions of conditions
variable (p q : ℝ)
variable (k : ℝ) 

-- Initial conditions
axiom h1 : p = 1500
axiom h2 : q = 0.5
axiom h3 : p * q = k
axiom h4 : k = 750

-- Goal
theorem solve_q (hp : p = 3000) : q = 0.250 :=
by
  -- The proof is omitted.
  sorry

end solve_q_l308_308271


namespace like_terms_exponents_l308_308381

theorem like_terms_exponents (m n : ℤ) (h1 : 2 * n - 1 = m) (h2 : m = 3) : m = 3 ∧ n = 2 :=
by
  sorry

end like_terms_exponents_l308_308381


namespace fiona_correct_answers_l308_308292

-- 5 marks for each correct answer in Questions 1-15
def marks_questions_1_to_15 (correct1 : ℕ) : ℕ := 5 * correct1

-- 6 marks for each correct answer in Questions 16-25
def marks_questions_16_to_25 (correct2 : ℕ) : ℕ := 6 * correct2

-- 1 mark penalty for incorrect answers in Questions 16-20
def penalty_questions_16_to_20 (incorrect1 : ℕ) : ℕ := incorrect1

-- 2 mark penalty for incorrect answers in Questions 21-25
def penalty_questions_21_to_25 (incorrect2 : ℕ) : ℕ := 2 * incorrect2

-- Total marks given correct and incorrect answers
def total_marks (correct1 correct2 incorrect1 incorrect2 : ℕ) : ℕ :=
  marks_questions_1_to_15 correct1 +
  marks_questions_16_to_25 correct2 -
  penalty_questions_16_to_20 incorrect1 -
  penalty_questions_21_to_25 incorrect2

-- Fiona's total score
def fionas_total_score : ℕ := 80

-- The proof problem: Fiona answered 16 questions correctly
theorem fiona_correct_answers (correct1 correct2 incorrect1 incorrect2 : ℕ) :
  total_marks correct1 correct2 incorrect1 incorrect2 = fionas_total_score → 
  (correct1 + correct2 = 16) := sorry

end fiona_correct_answers_l308_308292


namespace parabola_focus_distance_l308_308811

theorem parabola_focus_distance (P : ℝ × ℝ) (hP : P.2^2 = 4 * P.1) (h_dist_y_axis : |P.1| = 4) : 
  dist P (4, 0) = 5 :=
sorry

end parabola_focus_distance_l308_308811


namespace best_marksman_score_l308_308915

def team_size : ℕ := 6
def total_points : ℕ := 497
def hypothetical_best_score : ℕ := 92
def hypothetical_average : ℕ := 84

theorem best_marksman_score :
  let total_with_hypothetical_best := team_size * hypothetical_average
  let difference := total_with_hypothetical_best - total_points
  let actual_best_score := hypothetical_best_score - difference
  actual_best_score = 85 := 
by
  -- Definitions in Lean are correctly set up
  intro total_with_hypothetical_best difference actual_best_score
  sorry

end best_marksman_score_l308_308915


namespace range_of_k_l308_308983

theorem range_of_k (k : ℝ) :
  (∀ x : ℤ, ((x^2 - x - 2 > 0) ∧ (2*x^2 + (2*k + 5)*x + 5*k < 0)) ↔ (x = -2)) -> 
  (-3 ≤ k ∧ k < 2) :=
by 
  sorry

end range_of_k_l308_308983


namespace projectile_time_to_meet_l308_308904

theorem projectile_time_to_meet
  (d v1 v2 : ℝ)
  (hd : d = 1455)
  (hv1 : v1 = 470)
  (hv2 : v2 = 500) :
  (d / (v1 + v2)) * 60 = 90 := by
  sorry

end projectile_time_to_meet_l308_308904


namespace arithmetic_expression_value_l308_308885

theorem arithmetic_expression_value : 4 * (8 - 3) - 7 = 13 := by
  sorry

end arithmetic_expression_value_l308_308885


namespace simplify_expression_l308_308002

variable (a b : ℤ)

theorem simplify_expression :
  (30 * a + 45 * b) + (15 * a + 40 * b) - (20 * a + 55 * b) + (5 * a - 10 * b) = 30 * a + 20 * b :=
by
  sorry

end simplify_expression_l308_308002


namespace no_prime_solutions_for_x2_plus_y3_eq_z4_l308_308945

theorem no_prime_solutions_for_x2_plus_y3_eq_z4 :
  ¬ ∃ (x y z : ℕ), Prime x ∧ Prime y ∧ Prime z ∧ x^2 + y^3 = z^4 := sorry

end no_prime_solutions_for_x2_plus_y3_eq_z4_l308_308945


namespace rosie_pies_l308_308276

theorem rosie_pies (apples_per_pie : ℕ) (apples_total : ℕ) (pies_initial : ℕ) 
  (h1 : 3 = pies_initial) (h2 : 12 = apples_total) : 
  (36 / (apples_total / pies_initial)) * pies_initial = 27 := 
by
  sorry

end rosie_pies_l308_308276


namespace seed_selection_valid_l308_308451

def seeds : List Nat := [63, 01, 63, 78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07]

def extractValidSeeds (lst : List Nat) (startIndex : Nat) (maxValue : Nat) (count : Nat) : List Nat :=
  lst.drop startIndex
  |>.filter (fun n => n < maxValue)
  |>.take count

theorem seed_selection_valid :
  extractValidSeeds seeds 10 850 4 = [169, 555, 671, 105] :=
by
  sorry

end seed_selection_valid_l308_308451


namespace intersection_of_sets_l308_308070

def setA (x : ℝ) : Prop := x^2 - 4 * x - 5 > 0

def setB (x : ℝ) : Prop := 4 - x^2 > 0

theorem intersection_of_sets :
  {x : ℝ | setA x} ∩ {x : ℝ | setB x} = {x : ℝ | -2 < x ∧ x < -1} :=
by
  sorry

end intersection_of_sets_l308_308070


namespace find_number_l308_308900

theorem find_number (x : ℝ) (h : (2 * x - 37 + 25) / 8 = 5) : x = 26 :=
sorry

end find_number_l308_308900


namespace total_pages_in_book_l308_308114

theorem total_pages_in_book :
  ∃ x : ℝ, (x - (1/6 * x + 10) - (1/3 * (x - (1/6 * x + 10)) + 20)
           - (1/2 * ((x - (1/6 * x + 10) - (1/3 * (x - (1/6 * x + 10)) + 20))) + 25) = 120) ∧
           x = 552 :=
by
  sorry

end total_pages_in_book_l308_308114


namespace saving_is_zero_cents_l308_308336

-- Define the in-store and online prices
def in_store_price : ℝ := 129.99
def online_payment_per_installment : ℝ := 29.99
def shipping_and_handling : ℝ := 11.99

-- Define the online total price
def online_total_price : ℝ := 4 * online_payment_per_installment + shipping_and_handling

-- Define the saving in cents
def saving_in_cents : ℝ := (in_store_price - online_total_price) * 100

-- State the theorem to prove the number of cents saved
theorem saving_is_zero_cents : saving_in_cents = 0 := by
  sorry

end saving_is_zero_cents_l308_308336


namespace functional_equation_l308_308268

def f (x : ℝ) : ℝ := x + 1

theorem functional_equation (f : ℝ → ℝ) (h1 : f 0 = 1) (h2 : ∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2) :
  f = (λ x, x + 1) :=
by
  sorry

end functional_equation_l308_308268


namespace no_real_sqrt_neg_six_pow_three_l308_308606

theorem no_real_sqrt_neg_six_pow_three : 
  ∀ x : ℝ, 
    (¬ ∃ y : ℝ, y * y = -6 ^ 3) :=
by
  sorry

end no_real_sqrt_neg_six_pow_three_l308_308606


namespace fraction_of_married_men_is_two_fifths_l308_308695

noncomputable def fraction_of_married_men (W : ℕ) (p : ℚ) (h : p = 1 / 3) : ℚ :=
  let W_s := p * W
  let W_m := W - W_s
  let M_m := W_m
  let T := W + M_m
  M_m / T

theorem fraction_of_married_men_is_two_fifths (W : ℕ) (p : ℚ) (h : p = 1 / 3) (hW : W = 6) : fraction_of_married_men W p h = 2 / 5 :=
by
  sorry

end fraction_of_married_men_is_two_fifths_l308_308695


namespace jared_yearly_earnings_l308_308510

theorem jared_yearly_earnings (monthly_pay_diploma : ℕ) (multiplier : ℕ) (months_in_year : ℕ)
  (h1 : monthly_pay_diploma = 4000) (h2 : multiplier = 3) (h3 : months_in_year = 12) :
  (monthly_pay_diploma * multiplier * months_in_year) = 144000 :=
by
  -- The proof goes here
  sorry

end jared_yearly_earnings_l308_308510


namespace number_of_odd_terms_in_expansion_l308_308685

theorem number_of_odd_terms_in_expansion (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  number_of_odd_terms_in_expansion (p + q) 8 = 2 :=
sorry

end number_of_odd_terms_in_expansion_l308_308685


namespace factory_produces_6500_toys_per_week_l308_308335

theorem factory_produces_6500_toys_per_week
    (days_per_week : ℕ)
    (toys_per_day : ℕ)
    (h1 : days_per_week = 5)
    (h2 : toys_per_day = 1300) :
    days_per_week * toys_per_day = 6500 := 
by 
  sorry

end factory_produces_6500_toys_per_week_l308_308335


namespace symmetric_points_tangent_line_l308_308664

theorem symmetric_points_tangent_line (k : ℝ) (hk : 0 < k) :
  (∃ P Q : ℝ × ℝ, P.2 = Real.exp P.1 ∧ ∃ x₀ : ℝ, 
    Q.2 = k * Q.1 ∧ Q = (P.2, P.1) ∧ 
    Q.1 = x₀ ∧ k = 1 / x₀ ∧ x₀ = Real.exp 1) → k = 1 / Real.exp 1 := 
by 
  sorry

end symmetric_points_tangent_line_l308_308664


namespace weight_of_lightest_dwarf_l308_308422

noncomputable def weight_of_dwarf (n : ℕ) (x : ℝ) : ℝ := 5 - (n - 1) * x

theorem weight_of_lightest_dwarf :
  ∃ x : ℝ, 
    (∀ n : ℕ, n ≥ 1 ∧ n ≤ 101 → weight_of_dwarf 1 x = 5) ∧
    (weight_of_dwarf 76 x + weight_of_dwarf 77 x + weight_of_dwarf 78 x + weight_of_dwarf 79 x + weight_of_dwarf 80 x =
     weight_of_dwarf 96 x + weight_of_dwarf 97 x + weight_of_dwarf 98 x + weight_of_dwarf 99 x + weight_of_dwarf 100 x + weight_of_dwarf 101 x) →
    weight_of_dwarf 101 x = 2.5 :=
by
  sorry

end weight_of_lightest_dwarf_l308_308422


namespace hall_width_l308_308091

theorem hall_width 
  (L H cost total_expenditure : ℕ)
  (W : ℕ)
  (h1 : L = 20)
  (h2 : H = 5)
  (h3 : cost = 20)
  (h4 : total_expenditure = 19000)
  (h5 : total_expenditure = (L * W + 2 * (H * L) + 2 * (H * W)) * cost) :
  W = 25 := 
sorry

end hall_width_l308_308091


namespace equation_of_line_passing_through_points_l308_308047

-- Definition of the points
def point1 : ℝ × ℝ := (-2, -3)
def point2 : ℝ × ℝ := (4, 7)

-- The statement to prove
theorem equation_of_line_passing_through_points :
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ (forall (x y : ℝ), 
  y + 3 = (5 / 3) * (x + 2) → 3 * y - 5 * x = 1) := sorry

end equation_of_line_passing_through_points_l308_308047


namespace union_A_B_eq_real_subset_A_B_l308_308513

def A (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < 3 + a}
def B : Set ℝ := {x : ℝ | x ≤ -1 ∨ x ≥ 1}

theorem union_A_B_eq_real (a : ℝ) : (A a ∪ B) = Set.univ ↔ -2 ≤ a ∧ a ≤ -1 :=
by
  sorry

theorem subset_A_B (a : ℝ) : A a ⊆ B ↔ (a ≤ -4 ∨ a ≥ 1) :=
by
  sorry

end union_A_B_eq_real_subset_A_B_l308_308513


namespace entrance_fee_per_person_l308_308290

theorem entrance_fee_per_person :
  let ticket_price := 50.00
  let processing_fee_rate := 0.15
  let parking_fee := 10.00
  let total_cost := 135.00
  let known_cost := 2 * ticket_price + processing_fee_rate * (2 * ticket_price) + parking_fee
  ∃ entrance_fee_per_person, 2 * entrance_fee_per_person + known_cost = total_cost :=
by
  sorry

end entrance_fee_per_person_l308_308290


namespace overlap_difference_l308_308833

namespace GeometryBiology

noncomputable def total_students : ℕ := 350
noncomputable def geometry_students : ℕ := 210
noncomputable def biology_students : ℕ := 175

theorem overlap_difference : 
    let max_overlap := min geometry_students biology_students;
    let min_overlap := geometry_students + biology_students - total_students;
    max_overlap - min_overlap = 140 := 
by
  sorry

end GeometryBiology

end overlap_difference_l308_308833


namespace ratio_of_rectangles_l308_308141

theorem ratio_of_rectangles (p q : ℝ) (h1 : q ≠ 0) 
    (h2 : q^2 = 1/4 * (2 * p * q  - q^2)) : p / q = 5 / 2 := 
sorry

end ratio_of_rectangles_l308_308141


namespace quadrilateral_circles_l308_308553

theorem quadrilateral_circles (A B C D : Type) (h : True) :
  ∃ (n : ℕ), n = 6 :=
by
  use 6
  sorry

end quadrilateral_circles_l308_308553


namespace solve_inequalities_l308_308575

theorem solve_inequalities (x : ℝ) :
  ( (-x + 3)/2 < x ∧ 2*(x + 6) ≥ 5*x ) ↔ (1 < x ∧ x ≤ 4) :=
by
  sorry

end solve_inequalities_l308_308575


namespace peanut_raising_ratio_l308_308333

theorem peanut_raising_ratio
  (initial_peanuts : ℝ)
  (remove_peanuts_1 : ℝ)
  (add_raisins_1 : ℝ)
  (remove_mixture : ℝ)
  (add_raisins_2 : ℝ)
  (final_peanuts : ℝ)
  (final_raisins : ℝ)
  (ratio : ℝ) :
  initial_peanuts = 10 ∧
  remove_peanuts_1 = 2 ∧
  add_raisins_1 = 2 ∧
  remove_mixture = 2 ∧
  add_raisins_2 = 2 ∧
  final_peanuts = initial_peanuts - remove_peanuts_1 - (remove_mixture * (initial_peanuts - remove_peanuts_1) / (initial_peanuts - remove_peanuts_1 + add_raisins_1)) ∧
  final_raisins = add_raisins_1 - (remove_mixture * add_raisins_1 / (initial_peanuts - remove_peanuts_1 + add_raisins_1)) + add_raisins_2 ∧
  ratio = final_peanuts / final_raisins →
  ratio = 16 / 9 := by
  sorry

end peanut_raising_ratio_l308_308333


namespace identify_A_B_l308_308848

variable {Person : Type}
variable (isTruthful isLiar : Person → Prop)
variable (isBoy isGirl : Person → Prop)

variables (A B : Person)

-- Conditions
axiom truthful_or_liar : ∀ x : Person, isTruthful x ∨ isLiar x
axiom boy_or_girl : ∀ x : Person, isBoy x ∨ isGirl x
axiom not_both_truthful_and_liar : ∀ x : Person, ¬(isTruthful x ∧ isLiar x)
axiom not_both_boy_and_girl : ∀ x : Person, ¬(isBoy x ∧ isGirl x)

-- Statements made by A and B
axiom A_statement : isTruthful A → isLiar B 
axiom B_statement : isBoy B → isGirl A 

-- Goal: prove the identities of A and B
theorem identify_A_B : isTruthful A ∧ isBoy A ∧ isLiar B ∧ isBoy B :=
by {
  sorry
}

end identify_A_B_l308_308848


namespace simplify_fraction_l308_308749

theorem simplify_fraction : (5 + 4 - 3) / (5 + 4 + 3) = 1 / 2 := 
by {
  sorry
}

end simplify_fraction_l308_308749


namespace third_place_prize_correct_l308_308102

-- Define the conditions and formulate the problem
def total_amount_in_pot : ℝ := 210
def third_place_percentage : ℝ := 0.15
def third_place_prize (P : ℝ) : ℝ := third_place_percentage * P

-- The theorem to be proved
theorem third_place_prize_correct : 
  third_place_prize total_amount_in_pot = 31.5 := 
by
  sorry

end third_place_prize_correct_l308_308102


namespace shaded_region_area_l308_308936

noncomputable def radius1 := 4
noncomputable def radius2 := 5
noncomputable def distance := radius1 + radius2
noncomputable def large_radius := radius2 + distance / 2

theorem shaded_region_area :
  ∃ (A : ℝ), A = (π * large_radius ^ 2) - (π * radius1 ^ 2) - (π * radius2 ^ 2) ∧
  A = 49.25 * π :=
by
  sorry

end shaded_region_area_l308_308936


namespace average_goals_l308_308587

def num_goals_3 := 3
def num_players_3 := 2
def num_goals_4 := 4
def num_players_4 := 3
def num_goals_5 := 5
def num_players_5 := 1
def num_goals_6 := 6
def num_players_6 := 1

def total_goals := (num_goals_3 * num_players_3) + (num_goals_4 * num_players_4) + (num_goals_5 * num_players_5) + (num_goals_6 * num_players_6)
def total_players := num_players_3 + num_players_4 + num_players_5 + num_players_6

theorem average_goals :
  (total_goals / total_players : ℚ) = 29 / 7 :=
sorry

end average_goals_l308_308587


namespace problem1421_part1_problem1421_part2_problem1421_part3_problem1421_part4_l308_308696

theorem problem1421_part1 (total_balls : ℕ) (red_balls : ℕ) (yellow_balls : ℕ)
  (h_total : total_balls = 20) (h_red : red_balls = 5) (h_yellow : yellow_balls = 15) :
  (red_balls < yellow_balls) := by 
  sorry  -- Solution Proof for Part 1

theorem problem1421_part2 (total_balls : ℕ) (red_balls : ℕ) (h_total : total_balls = 20) 
  (h_red : red_balls = 5) :
  (red_balls / total_balls = 1 / 4) := by 
  sorry  -- Solution Proof for Part 2

theorem problem1421_part3 (red_balls total_balls m : ℕ) (h_red : red_balls = 5) 
  (h_total : total_balls = 20) :
  ((red_balls + m) / (total_balls + m) = 3 / 4) → (m = 40) := by 
  sorry  -- Solution Proof for Part 3

theorem problem1421_part4 (total_balls red_balls additional_balls x : ℕ) 
  (h_total : total_balls = 20) (h_red : red_balls = 5) (h_additional : additional_balls = 18):
  (total_balls + additional_balls = 38) → ((red_balls + x) / 38 = 1 / 2) → 
  (x = 14) ∧ ((additional_balls - x) = 4) := by 
  sorry  -- Solution Proof for Part 4

end problem1421_part1_problem1421_part2_problem1421_part3_problem1421_part4_l308_308696


namespace shaded_area_percentage_correct_l308_308027

-- Define a square and the conditions provided
def square (side_length : ℕ) : ℕ := side_length ^ 2

-- Define conditions
def EFGH_side_length : ℕ := 6
def total_area : ℕ := square EFGH_side_length

def shaded_area_1 : ℕ := square 2
def shaded_area_2 : ℕ := square 4 - square 3
def shaded_area_3 : ℕ := square 6 - square 5

def total_shaded_area : ℕ := shaded_area_1 + shaded_area_2 + shaded_area_3

def shaded_percentage : ℚ := total_shaded_area / total_area * 100

-- Statement of the theorem to prove
theorem shaded_area_percentage_correct :
  shaded_percentage = 61.11 := by sorry

end shaded_area_percentage_correct_l308_308027


namespace A_wins_3_1_probability_l308_308762

noncomputable def probability_A_wins_3_1 (p : ℚ) : ℚ :=
  let win_3_1 := binomial 4 3 * (p^3) * (1 - p)
  win_3_1

theorem A_wins_3_1_probability : probability_A_wins_3_1 (2/3) = 8/27 := by
  sorry

end A_wins_3_1_probability_l308_308762


namespace jeremy_goal_product_l308_308090

theorem jeremy_goal_product 
  (g1 g2 g3 g4 g5 : ℕ) 
  (total5 : g1 + g2 + g3 + g4 + g5 = 13)
  (g6 g7 : ℕ) 
  (h6 : g6 < 10) 
  (h7 : g7 < 10) 
  (avg6 : (13 + g6) % 6 = 0) 
  (avg7 : (13 + g6 + g7) % 7 = 0) :
  g6 * g7 = 15 := 
sorry

end jeremy_goal_product_l308_308090


namespace find_a_values_l308_308817

def setA (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.snd - 3) / (p.fst - 2) = a + 1}

def setB (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (a^2 - 1) * p.fst + (a - 1) * p.snd = 15}

def sets_disjoint (A B : Set (ℝ × ℝ)) : Prop := ∀ p : ℝ × ℝ, p ∉ A ∪ B

theorem find_a_values (a : ℝ) :
  sets_disjoint (setA a) (setB a) ↔ a = 1 ∨ a = -1 ∨ a = 5/2 ∨ a = -4 :=
sorry

end find_a_values_l308_308817


namespace percentage_singing_l308_308539

def total_rehearsal_time : ℕ := 75
def warmup_time : ℕ := 6
def notes_time : ℕ := 30
def words_time (t : ℕ) : ℕ := t
def singing_time (t : ℕ) : ℕ := total_rehearsal_time - warmup_time - notes_time - words_time t
def singing_percentage (t : ℕ) : ℕ := (singing_time t * 100) / total_rehearsal_time

theorem percentage_singing (t : ℕ) : (singing_percentage t) = (4 * (39 - t)) / 3 :=
by
  sorry

end percentage_singing_l308_308539


namespace selling_price_for_given_profit_selling_price_to_maximize_profit_l308_308770

-- Define the parameters
def cost_price : ℝ := 40
def initial_selling_price : ℝ := 50
def initial_monthly_sales : ℝ := 500
def sales_decrement_per_unit_increase : ℝ := 10

-- Define the function for monthly sales based on price increment
def monthly_sales (x : ℝ) : ℝ := initial_monthly_sales - sales_decrement_per_unit_increase * x

-- Define the function for selling price based on price increment
def selling_price (x : ℝ) : ℝ := initial_selling_price + x

-- Define the function for monthly profit
def monthly_profit (x : ℝ) : ℝ :=
  let total_revenue := monthly_sales x * selling_price x 
  let total_cost := monthly_sales x * cost_price
  total_revenue - total_cost

-- Problem 1: Prove the selling price when monthly profit is 8750 yuan
theorem selling_price_for_given_profit : 
  ∃ x : ℝ, monthly_profit x = 8750 ∧ (selling_price x = 75 ∨ selling_price x = 65) :=
sorry

-- Problem 2: Prove the selling price that maximizes the monthly profit
theorem selling_price_to_maximize_profit : 
  ∀ x : ℝ, monthly_profit x ≤ monthly_profit 20 ∧ selling_price 20 = 70 :=
sorry

end selling_price_for_given_profit_selling_price_to_maximize_profit_l308_308770


namespace multiplicative_inverse_of_550_mod_4319_l308_308374

theorem multiplicative_inverse_of_550_mod_4319 :
  (48^2 + 275^2 = 277^2) → ((550 * 2208) % 4319 = 1) := by
  intro h
  sorry

end multiplicative_inverse_of_550_mod_4319_l308_308374


namespace sqrt_of_square_of_neg_five_eq_five_l308_308930

theorem sqrt_of_square_of_neg_five_eq_five : Real.sqrt ((-5 : ℤ) ^ 2) = 5 := by
  sorry

end sqrt_of_square_of_neg_five_eq_five_l308_308930


namespace money_last_duration_l308_308269

-- Defining the conditions
def money_from_mowing : ℕ := 14
def money_from_weed_eating : ℕ := 26
def money_spent_per_week : ℕ := 5

-- Theorem statement to prove Mike's money will last 8 weeks
theorem money_last_duration : (money_from_mowing + money_from_weed_eating) / money_spent_per_week = 8 := by
  sorry

end money_last_duration_l308_308269


namespace non_adjective_primes_sum_l308_308491

-- We will define the necessary components as identified from our problem

def is_adjective_prime (p : ℕ) [Fact (Nat.Prime p)] : Prop :=
  ∃ a : ℕ → ℕ, ∀ n : ℕ,
    a 0 % p = (1 + (1 / a 1) % p) ∧
    a 1 % p = (1 + (1 / (1 + (1 / a 2) % p)) % p) ∧
    a 2 % p = (1 + (1 / (1 + (1 / (1 + (1 / a 3) % p))) % p))

def is_not_adjective_prime (p : ℕ) [Fact (Nat.Prime p)] : Prop :=
  ¬ is_adjective_prime p

def first_three_non_adjective_primes_sum : ℕ :=
  3 + 7 + 23

theorem non_adjective_primes_sum :
  first_three_non_adjective_primes_sum = 33 := 
  sorry

end non_adjective_primes_sum_l308_308491
